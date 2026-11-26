"""The SFFT fit mask: which pixels of the matched pair the kernel fit may use.

sparse_sfft is SFFT's sparse-flavour selection: SExtractor on both matched images, the two catalogues matched, sources off the
magnitude-offset surface rejected as variables, blend groups admitted, sources near saturated pixels dropped whole, and the selected
footprints grown. Two options: the star guard keeps a saturated star's light, its diffraction spikes above all, out of the fit; the PSF star
test leaves out the selected sources the effective PSF identifies as point sources. build is the band mask: galaxies in, stars and bad
pixels out, within a band of brightness."""
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

COLUMNS = ("NUMBER", "X_IMAGE", "Y_IMAGE", "A_IMAGE", "B_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO", "FLUX_MAX", "FLAGS", "FLUX_RADIUS", "ISOAREA_IMAGE")
LOCUS_OVER_MODEL = 0.92
LOCUS_WINDOW = (0.85, 1.15)
MIN_LOCUS_STARS = 20
_TOOL = None


def sextractor():
    """betasfft's SExtractor: the program built with the package from sfft/spaceflow/prep/_sextractor (SExtractor 2.28.0), run as a separate process
    with its files passed through pipes (sfft.spaceflow.prep.astromatic.bundled.BundledSExtractor). Stops with the build command when the program is
    not built or was built from other sources."""
    global _TOOL
    if _TOOL is None:
        from sfft.spaceflow.prep.astromatic.bundled import BundledSExtractor
        _TOOL = BundledSExtractor()
    return _TOOL


def detect(image, thresh=2.0, minarea=5, mask=None, apertures=None, deblend=True, extra_columns=()):
    """SExtractor on an image: (catalogue, segmentation map). The catalogue holds SExtractor's columns plus 0-based x, y and FLUX_APER for the
    given aperture radii [px]; a segmentation label is its catalogue row + 1. NaN and masked pixels are bad pixels. deblend=False: one
    detection per connected group of pixels above the threshold; extra_columns: more SExtractor columns."""
    from sfft.spaceflow.prep.astromatic.sextractor import SExConfig
    bad = ~np.isfinite(image) if mask is None else (mask | ~np.isfinite(image))
    img = np.where(bad, 0.0, image).astype(np.float32)
    columns = COLUMNS + tuple(extra_columns) + ((f"FLUX_APER({len(apertures)})",) if apertures else ())
    config = SExConfig(gain=0.0, satur_level=1e20, detect_thresh=thresh, analysis_thresh=thresh, detect_minarea=minarea, back_type="AUTO",
                       backphoto_type="LOCAL", weight_type="MAP_WEIGHT", weight_thresh=0.0, rescale_weights="Y",
                       phot_apertures=tuple(2.0 * r for r in apertures) if apertures else (5.0,),
                       extra=(("MEMORY_PIXSTACK", max(300000, img.size // 2)), ("MEMORY_OBJSTACK", 60000)),
                       **({} if deblend else dict(deblend_mincont=1.0)))
    res = sextractor().run(image=img, config=config, columns=columns, checkimages=("SEGMENTATION",), weight=(~bad).astype(np.float32))
    cat = res.catalog
    cat["x"] = np.asarray(cat["X_IMAGE"], float) - 1.0
    cat["y"] = np.asarray(cat["Y_IMAGE"], float) - 1.0
    return cat, np.asarray(res.checkimages["SEGMENTATION"], dtype=np.int32)


def sparse_products(sci_w, ref_w, fwhm_px):
    """SExtractor products of both matched images, made the way SFFT's sparse flavour makes them (sfft.spaceflow.prep.sparse.catalog.image_products):
    the slow part of sparse_sfft. Each image goes to SExtractor in memory: image_products gets the float32 array with missing pixels 0 (what
    SFFT writes to its temporary FITS file) bound to the tool, and no file is written. Missing pixels serve as the saturation mask until
    sparse_mask sets it. Returns dict(ref, sci, nan)."""
    from sfft.spaceflow.prep.sparse.calib import Calib
    from sfft.spaceflow.prep.sparse.catalog import image_products
    from sfft.spaceflow.prep.astromatic.bundled import PREFIX, BoundExtractor
    nan = ~np.isfinite(sci_w) | ~np.isfinite(ref_w)
    calib = Calib(gain=0.0, satur=1e20, satur_raw=1e20, gain_source="value: images in MJy/sr, no Poisson term",
                  satur_source="none: saturated pixels are missing pixels")
    prods = dict(nan=nan)
    for name, img in (("ref", ref_w), ("sci", sci_w)):
        arr = np.where(nan, 0.0, img).astype(np.float32)
        prod = image_products(PREFIX + name, calib, image=np.ascontiguousarray(np.where(nan, np.nan, img).T), fwhm_prior=float(fwhm_px),
                              sextractor=BoundExtractor(sextractor(), arr), identity=("in-memory array", name))
        prod.sat_mask = np.ascontiguousarray(nan.T)
        prods[name] = prod
    return prods


SPIKE_ELL, SPIKE_ANGLE_DEG, SPIKE_REACH = 0.5, 20.0, 1.5


def star_detections(prod, sat, pad):
    """Per flagged core (8-connected region of sat): its centre (x, y) and the radius [px] of the star's own detection in this image, the
    farthest pixel of the labelled islands that come within pad px of the core, spikes included (0 when none does)."""
    core_lab, k = ndimage.label(sat, structure=np.ones((3, 3), bool))
    if not k:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cx, cy = (np.asarray(c, float) for c in zip(*ndimage.center_of_mass(np.ones(sat.shape), core_lab, np.arange(1, k + 1))))
    area = ndimage.sum_labels(np.ones(sat.shape), core_lab, np.arange(1, k + 1))
    island, _ = ndimage.label(prod.seg > 0, structure=np.ones((3, 3), bool))
    near = ndimage.binary_dilation(core_lab > 0, iterations=int(pad)) if pad else core_lab > 0
    which = ndimage.grey_dilation(core_lab, size=2 * int(pad) + 1) if pad else core_lab
    touch = near & (island > 0) & (which > 0)
    pairs = np.unique(np.c_[which[touch], island[touch]], axis=0)
    if len(pairs):
        order = np.lexsort((-area[pairs[:, 0] - 1], pairs[:, 1]))
        pairs = pairs[order]
        pairs = pairs[np.r_[True, pairs[1:, 1] != pairs[:-1, 1]]]
    boxes = ndimage.find_objects(island)
    radius = np.zeros(k)
    for c, i in pairs:
        sl = boxes[i - 1]
        ii, jj = np.nonzero(island[sl] == i)
        radius[c - 1] = max(radius[c - 1], float(np.hypot(ii + sl[0].start - cx[c - 1], jj + sl[1].start - cy[c - 1]).max()))
    return cx, cy, radius


def star_zone(shape, cx, cy, radius, reach=None):
    """(x, y) bool: discs of radius reach x radius around the flagged cores."""
    reach = SPIKE_REACH if reach is None else float(reach)
    zone = np.zeros(shape, bool)
    for x, y, r in zip(cx, cy, reach * np.asarray(radius)):
        if r <= 0:
            continue
        i0, i1 = max(0, int(x - r)), min(shape[0], int(x + r) + 2)
        j0, j1 = max(0, int(y - r)), min(shape[1], int(y + r) + 2)
        ii, jj = np.ogrid[i0:i1, j0:j1]
        zone[i0:i1, j0:j1] |= (ii - x) ** 2 + (jj - y) ** 2 <= r * r
    return zone


def spike_shaped(prod, sat, pad, labels, detections=None):
    """For the given labels of one image's products: True when a label looks like a piece of a saturated star's diffraction spike
    (elongated, its long axis pointing to a flagged core, within SPIKE_REACH times the star's detection radius)."""
    cat = prod.catalog
    out = np.zeros(len(labels), bool)
    cx, cy, radius = star_detections(prod, sat, pad) if detections is None else detections
    if not len(cx) or not len(labels):
        return out
    rows = np.asarray(labels) - 1
    x = np.asarray(cat["X_IMAGE"], float)[rows] - 1.0
    y = np.asarray(cat["Y_IMAGE"], float)[rows] - 1.0
    a, b = np.asarray(cat["A_IMAGE"], float)[rows], np.asarray(cat["B_IMAGE"], float)[rows]
    theta = np.asarray(cat["THETA_IMAGE"], float)[rows]
    ell = 1.0 - b / np.maximum(a, 1e-6)
    d = np.hypot(x[:, None] - cx[None, :], y[:, None] - cy[None, :])
    radial = np.degrees(np.arctan2(y[:, None] - cy[None, :], x[:, None] - cx[None, :]))
    off = np.abs((theta[:, None] - radial + 90.0) % 180.0 - 90.0)
    hit = (d <= SPIKE_REACH * radius[None, :]) & (off <= SPIKE_ANGLE_DEG) & (d > 0)
    return (ell >= SPIKE_ELL) & hit.any(axis=1)


def overlap_mask(sp, prod_r, prod_s, dilate):
    """The active mask from the pixels both images give to the same selected source, grown by dilate pixels only into pixels no label
    covers and that are not prohibited ((x, y) orientation)."""
    seg_r, seg_s = prod_r.seg, prod_s.seg
    n = len(sp.ir_sel)
    new_r = np.zeros(int(seg_r.max()) + 1, np.int32)
    new_s = np.zeros(int(seg_s.max()) + 1, np.int32)
    new_r[sp.ir_sel + 1] = np.arange(1, n + 1)
    new_s[sp.js_sel + 1] = np.arange(1, n + 1)
    uni_r, uni_s = new_r[seg_r], new_s[seg_s]
    both = (uni_r > 0) & (uni_r == uni_s)
    blocked = sp.prohibited | (((seg_r > 0) | (seg_s > 0)) & ~both)
    lm = np.where(both & ~blocked, uni_r, 0)
    if dilate:
        grown = ndimage.maximum_filter(lm, size=2 * int(dilate) + 1)
        lm = np.where(lm > 0, lm, np.where(blocked, 0, grown))
    return lm > 0


SPIKE_W0, SPIKE_DTHETA, SPIKE_HALO, SPIKE_CONTRAST = 12.0, 0.3, 1.2, 3.0


def _ang(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def angular_excess(img, x0, y0, r0, r1, scale=1.0):
    """The light of img by angle around (x0, y0) over the annulus r0..r1, less its running median over 30 deg: what lies along spikes."""
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.hypot(y - y0, x - x0) / scale
    tt = np.degrees(np.arctan2(y - y0, x - x0)) % 360.0
    ok = (rr > r0) & (rr < r1) & np.isfinite(img)
    b = np.floor(tt[ok] * 4).astype(int) % 1440
    s = np.bincount(b, weights=img[ok], minlength=1440)
    c = np.bincount(b, minlength=1440)
    p = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    p = np.where(np.isfinite(p), p, np.nanmedian(p) if np.isfinite(p).any() else 0.0)
    return p - ndimage.median_filter(p, size=121, mode="wrap")


def refine_angles(exc, angles, win):
    """Each angle moved to the intensity-weighted angle of an angular excess profile within +-win deg."""
    grid = (np.arange(1440) + 0.5) / 4
    out = []
    for a in angles:
        m = np.abs(_ang(grid, a)) <= win
        w = np.clip(exc[m], 0, None)
        out.append(float((a + (w * _ang(grid[m], a)).sum() / max(w.sum(), 1e-30)) % 360))
    return np.array(out)


def spike_angles(exc, n=12, sep=8.0):
    """The n strongest peaks of an angular excess profile, at least sep deg apart, each refined within +-1.5 deg."""
    sm = (exc + np.roll(exc, 1) + np.roll(exc, -1)) / 3
    peaks = []
    for i in np.argsort(-sm):
        a = (i + 0.5) / 4
        if all(abs(_ang(a, p)) >= sep for p in peaks):
            peaks.append(a)
        if len(peaks) == n:
            break
    return refine_angles(exc, peaks, 1.5)


def model_spike_angles(psf_nodes, side, x, y, cache=None):
    """The spike angles [deg] of one image's PSF model ('sci' or 'ref' of psf_nodes) at the node nearest (x, y)."""
    xs, ys = np.asarray(psf_nodes["xs"]), np.asarray(psf_nodes["ys"])
    i, j = int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))
    if cache is not None and (side, i, j) in cache:
        return cache[(side, i, j)]
    stamp = np.asarray(psf_nodes[side][i][j], float)
    c = (stamp.shape[0] - 1) / 2
    a = spike_angles(angular_excess(stamp, c, c, 12, 30, int(psf_nodes["oversample"])))
    if cache is not None:
        cache[(side, i, j)] = a
    return a


def _own_detection(seg, x0, y0, r_core, R_det, reach_px):
    half = int(min(R_det, 600)) + 2
    sl = np.s_[max(0, int(y0) - half):min(seg.shape[0], int(y0) + half + 1), max(0, int(x0) - half):min(seg.shape[1], int(x0) + half + 1)]
    sg = seg[sl]
    isl, _ = ndimage.label(sg > 0, structure=np.ones((3, 3), bool))
    yy, xx = np.mgrid[sl]
    r = np.hypot(yy - y0, xx - x0)
    own = np.unique(isl[(r < reach_px) & (isl > 0)])
    return sl, sg, r, np.isin(isl, own) & (isl > 0)


def halo_radius(seg, x0, y0, r_core, R_det, pad=2):
    """Where a saturated star's own detection stops covering the circle: the first radius beyond r_core + 4 px where it covers less than 0.3 of it."""
    _, _, r, mine = _own_detection(seg, x0, y0, r_core, R_det, r_core + pad + 2)
    rh = r_core + 4
    if mine.any():
        cov = np.bincount(np.floor(r[mine]).astype(int)) / np.maximum(2 * np.pi * np.arange(int(r[mine].max()) + 1), 1)
        past = (cov < 0.3) & (np.arange(len(cov)) > r_core + 4)
        rh = max(rh, float(np.argmax(past)) if past.any() else float(r[mine].max()))
    return rh


def star_spikes(img, seg, x0, y0, r_core, R_det, rh, model):
    """One image's far spikes of one saturated star: dict(far, contrast), the model's angles refined on the star's own far light, and how much that light stands out."""
    r0, r1 = max(rh + 10, 0.4 * R_det), min(R_det, 500.0)
    if r1 - r0 < 15 or not len(model):
        return dict(far=np.array(model, float), contrast=0.0)
    sl, sg, _, mine = _own_detection(seg, x0, y0, r_core, R_det, r_core + 6)
    crop = np.where((sg == 0) | mine, img[sl].astype(np.float64), np.nan)
    exc = angular_excess(crop, x0 - sl[1].start, y0 - sl[0].start, r0, r1)
    far = refine_angles(exc, refine_angles(exc, model, 2.5), 1.0)
    grid = (np.arange(1440) + 0.5) / 4
    on = np.zeros(1440, bool)
    for a in far:
        on |= np.abs(_ang(grid, a)) <= 3.0
    spread = 1.4826 * np.median(np.abs(exc[~on] - np.median(exc[~on])))
    con = [float(np.mean(exc[np.abs(_ang(grid, a)) <= 0.5])) / max(spread, 1e-30) for a in far]
    return dict(far=far, contrast=float(np.median(con)))


def guard_stars(sci_w, ref_w, prods, sat_xy, cx, cy, radius, psf_nodes, pad=2):
    """Per flagged core with a detection: the corridors' geometry (the PSF models' spike angles, the star's halo and detection radii) and
    whether its far spike light stands out in both images (ok)."""
    core_lab, k = ndimage.label(sat_xy, structure=np.ones((3, 3), bool))
    area = ndimage.sum_labels(np.ones(sat_xy.shape), core_lab, np.arange(1, k + 1)) if k else np.zeros(0)
    segs = dict(sci=prods["sci"].seg.T, ref=prods["ref"].seg.T)
    imgs = dict(sci=sci_w, ref=ref_w)
    cache, out = {}, []
    for c in np.nonzero(np.asarray(radius) > 0)[0]:
        x0, y0, R_det = float(cx[c]), float(cy[c]), float(radius[c])
        r_core = np.sqrt(area[c] / np.pi)
        rh = max(halo_radius(segs[side], x0, y0, r_core, R_det, pad) for side in ("sci", "ref"))
        st = dict(core=int(c + 1), x=x0, y=y0, R_det=R_det, rh=rh, angles=[], far=[], contrast={})
        for side in ("sci", "ref"):
            model = model_spike_angles(psf_nodes, side, x0, y0, cache)
            sp = star_spikes(imgs[side], segs[side], x0, y0, r_core, R_det, rh, model)
            st["angles"] += [float(a) for a in model]
            st["far"] += [float(a) for a in sp["far"]]
            st["contrast"][side] = sp["contrast"]
        st["ok"] = bool(min(st["contrast"].values()) >= SPIKE_CONTRAST)
        out.append(st)
    return out


def star_corridors(shape, stars, reach=None, w0=None, dtheta=None, halo=None):
    """[y, x] bool: corridors along each ok star's spike angles (half-width w0 + r tan(dtheta)) from its halo radius to reach x its
    detection radius, plus its halo disc; for a star without ok, the disc of reach x its detection radius."""
    reach = SPIKE_REACH if reach is None else float(reach)
    w0 = SPIKE_W0 if w0 is None else float(w0)
    dtheta = SPIKE_DTHETA if dtheta is None else float(dtheta)
    halo = SPIKE_HALO if halo is None else float(halo)
    zone = np.zeros(shape, bool)
    for s in stars:
        Rz = reach * s["R_det"]
        y_lo, y_hi, x_lo, x_hi = max(0, int(s["y"] - Rz)), min(shape[0], int(s["y"] + Rz) + 2), max(0, int(s["x"] - Rz)), min(shape[1], int(s["x"] + Rz) + 2)
        yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
        r = np.hypot(yy - s["y"], xx - s["x"])
        if not s["ok"]:
            zone[y_lo:y_hi, x_lo:x_hi] |= r <= Rz
            continue
        th = np.degrees(np.arctan2(yy - s["y"], xx - s["x"]))
        c = np.zeros(r.shape, bool)
        for a in s["angles"]:
            d = np.radians(_ang(th, a))
            c |= (np.abs(r * np.sin(d)) <= w0 + r * np.tan(np.radians(dtheta))) & (np.cos(d) > 0)
        zone[y_lo:y_hi, x_lo:x_hi] |= (c & (r > s["rh"]) & (r <= Rz)) | (r <= halo * s["rh"])
    return zone


def guard_mode(star_guard):
    """The star guard's geometry from the option's value: False, 'disc' (also True) or 'corridors'."""
    if star_guard is None or star_guard is False or (isinstance(star_guard, (int, float)) and not isinstance(star_guard, bool) and star_guard == 0):
        return False
    if star_guard is True or star_guard == "disc":
        return "disc"
    if star_guard == "corridors":
        return "corridors"
    raise ValueError(f"star_guard must be 'corridors', 'disc' (True) or False, not {star_guard!r}")


STAR_SIZE_PX = 0.3
STAR_DCHI2 = 9.0
STAR_MODEL_ERR = 0.03
STAR_HALF = 4


def psf_stars(img, x, y, psf_fine, oversample, sigma=None, size_px=STAR_SIZE_PX, dchi2=STAR_DCHI2, model_err=STAR_MODEL_ERR, half=STAR_HALF,
              iterations=4):
    """Which sources the PSF shows to be point sources. At each (x, y) two models are fitted on the (2 half + 1)^2 pixels around it, each
    with a free flux, background and sub-pixel position: the effective PSF (psf_fine, oversample samples per pixel) and the PSF convolved with
    a round Gaussian of sigma size_px. A source is a star when the widened model fits worse by more than dchi2; a source whose central 3 x 3
    pixels are not all usable is not tested. Returns a dict of arrays: star, tested, dchi2, chi2_point, chi2_wide, flux, dx, dy, core, npix."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    n = len(x)
    out = dict(star=np.zeros(n, bool), tested=np.zeros(n, bool), dchi2=np.full(n, np.nan), chi2_point=np.full(n, np.nan),
               chi2_wide=np.full(n, np.nan), flux=np.full(n, np.nan), dx=np.full(n, np.nan), dy=np.full(n, np.nan), core=np.full(n, np.nan),
               npix=np.zeros(n, int))
    img = np.asarray(img, float)
    ny, nx = img.shape
    h, os_ = int(half), int(oversample)
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    idx = np.nonzero(np.isfinite(x) & np.isfinite(y) & (ix - h >= 0) & (ix + h < nx) & (iy - h >= 0) & (iy + h < ny))[0]
    if not len(idx):
        return out
    if sigma is None:
        v = img[np.isfinite(img)]
        sigma = float(1.4826 * np.median(np.abs(v - np.median(v))))
    E = np.asarray(psf_fine, float)
    fy, fx = np.fft.fftfreq(E.shape[0])[:, None], np.fft.rfftfreq(E.shape[1])[None, :]
    wide = np.fft.irfft2(np.fft.rfft2(E) * np.exp(-2.0 * np.pi ** 2 * (float(size_px) * os_) ** 2 * (fx ** 2 + fy ** 2)), E.shape)
    coef = [ndimage.spline_filter(M, order=3) for M in (E, wide)]
    cy, cx = (E.shape[0] - 1) / 2.0, (E.shape[1] - 1) / 2.0
    off = np.arange(-h, h + 1)
    V, U = np.meshgrid(off, off, indexing="ij")
    m, P = len(idx), V.size
    T = img[iy[idx, None, None] + V[None], ix[idx, None, None] + U[None]].reshape(m, P)
    good = np.isfinite(T)
    T = np.where(good, T, 0.0)
    edge = ((np.abs(V) == h) | (np.abs(U) == h)).ravel()
    border = np.where(good & edge[None], T, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nan_to_num(np.nanmedian(border, axis=1))
    w = np.where(good, 1.0 / (sigma ** 2 + (model_err * np.maximum(T - bg[:, None], 0.0)) ** 2), 0.0)
    sx, sy = x[idx] - ix[idx], y[idx] - iy[idx]
    one = np.ones((m, P))

    def model(k, px, py):
        rows = cy + (V[None] - py[:, None, None]) * os_
        cols = cx + (U[None] - px[:, None, None]) * os_
        return ndimage.map_coordinates(coef[k], [rows.ravel(), cols.ravel()], order=3, prefilter=False, mode="constant", cval=0.0).reshape(m, P)

    def solve(X):
        A = np.einsum("mpi,mp,mpj->mij", X, w, X)
        A = A + 1e-12 * np.trace(A, axis1=1, axis2=2)[:, None, None] * np.eye(X.shape[2])[None]
        c = np.linalg.solve(A, np.einsum("mpi,mp,mp->mi", X, w, T)[..., None])[..., 0]
        r = T - np.einsum("mpi,mi->mp", X, c)
        return c, np.einsum("mp,mp,mp->m", r, w, r)

    fits, d = [], 0.05
    for k in (0, 1):
        px, py = sx.copy(), sy.copy()
        for _ in range(int(iterations)):
            m0 = model(k, px, py)
            mx = (model(k, px + d, py) - model(k, px - d, py)) / (2 * d)
            my = (model(k, px, py + d) - model(k, px, py - d)) / (2 * d)
            c, _ = solve(np.stack([m0, mx, my, one], axis=2))
            with np.errstate(all="ignore"):
                f = np.where(c[:, 0] != 0, c[:, 0], np.nan)
                px = np.clip(px + np.clip(np.nan_to_num(c[:, 1] / f), -0.3, 0.3), sx - 1.0, sx + 1.0)
                py = np.clip(py + np.clip(np.nan_to_num(c[:, 2] / f), -0.3, 0.3), sy - 1.0, sy + 1.0)
        m0 = model(k, px, py)
        c, chi2 = solve(np.stack([m0, one], axis=2))
        fits.append((c, chi2, px, py, m0))
    (cp, chip, pxp, pyp, mp), (_, chiw, _, _, _) = fits
    centre = ((np.abs(V) <= 1) & (np.abs(U) <= 1)).ravel()
    ok = (good.sum(1) >= 0.5 * P) & good[:, centre].all(1) & (cp[:, 0] > 0) & np.isfinite(chip) & np.isfinite(chiw)
    dchi = chiw - chip
    j = (h + np.clip(np.round(pyp).astype(int), -h, h)) * (2 * h + 1) + (h + np.clip(np.round(pxp).astype(int), -h, h))
    r = np.arange(m)
    with np.errstate(all="ignore"):
        core = (T[r, j] - cp[:, 0] * mp[r, j] - cp[:, 1]) / (cp[:, 0] * mp[r, j])
    out["tested"][idx] = ok
    out["star"][idx] = ok & (dchi > float(dchi2))
    for key, val in (("dchi2", dchi), ("chi2_point", chip), ("chi2_wide", chiw), ("flux", cp[:, 0]), ("dx", pxp - sx), ("dy", pyp - sy), ("core", core)):
        out[key][idx] = np.where(ok, val, np.nan)
    out["npix"][idx] = good.sum(1)
    return out


def _core_residual(test, min_dchi2=100.0, min_stars=5):
    """The median point-model residual at the core over strongly detected stars: near 0 when the working PSF is right; None with fewer than min_stars."""
    if test is None:
        return None
    sel = test["star"] & (test["dchi2"] > min_dchi2)
    return float(np.nanmedian(test["core"][sel])) if sel.sum() >= min_stars else None


def sparse_mask(prods, sci_w, ref_w, gker_hw, grow=None, sat_pad=2, sat_map=None, match_tol_px=None, star_guard=True, exclude_stars=False,
                psf_fine=None, psf_oversample=None, psf_nodes=None, detail=None):
    """SFFT's sparse-flavour mask from sparse_products (sparse.select_subsources and build_masks), with the options: grow (pixels the
    footprints grow by; None: SFFT's), sat_pad (pixels around a saturated pixel within which a source is dropped whole), sat_map (the
    saturated pixels; None: every missing pixel), match_tol_px (catalogue match tolerance; None: SFFT's), star_guard (False, 'disc' or
    'corridors'; needs sat_map, and psf_nodes for 'corridors'), exclude_stars (leave out the sources psf_stars finds to be point sources; needs
    psf_fine and psf_oversample). detail receives the intermediate products. Returns (fit, record)."""
    from sfft.spaceflow.prep.sparse.mask import build_masks
    from sfft.spaceflow.prep.sparse.select import select_subsources
    import sfft.spaceflow.prep.sparse.select as sparse_select
    nan = prods["nan"]
    sat = nan if sat_map is None else np.asarray(sat_map, bool)
    for name in ("ref", "sci"):
        prods[name].sat_mask = np.ascontiguousarray(sat.T)
    match = sparse_select.match_catalogs
    if match_tol_px is not None:
        sparse_select.match_catalogs = lambda cr, cs, fr, fs, tol=None, tol_factor=3.0: match(cr, cs, fr, fs, tol=float(match_tol_px))
    try:
        sel = select_subsources(prods["ref"], prods["sci"])
    finally:
        sparse_select.match_catalogs = match
    keep_sfft = sel.keep.copy()
    spike = np.zeros_like(keep_sfft)
    zone, stars = None, None
    mode = guard_mode(star_guard)
    if mode and sat_map is not None and sat.any():
        sat_xy = np.ascontiguousarray(sat.T)
        det_r, det_s = star_detections(prods["ref"], sat_xy, sat_pad), star_detections(prods["sci"], sat_xy, sat_pad)
        idx = np.nonzero(sel.keep)[0]
        spike[idx] = (spike_shaped(prods["ref"], sat_xy, sat_pad, sel.match.ir[idx] + 1, det_r)
                      | spike_shaped(prods["sci"], sat_xy, sat_pad, sel.match.js[idx] + 1, det_s))
        sel.keep = sel.keep & ~spike
        radius = np.maximum(det_r[2], det_s[2])
        if mode == "corridors":
            if psf_nodes is None:
                raise ValueError("star_guard='corridors' needs the PSF models at the nodes: psf_nodes")
            stars = guard_stars(sci_w, ref_w, prods, sat_xy, det_r[0], det_r[1], radius, psf_nodes, pad=sat_pad)
            zone = np.ascontiguousarray(star_corridors(sci_w.shape, stars).T)
        else:
            zone = star_zone(sat_xy.shape, det_r[0], det_r[1], radius)
    psf_star, test = np.zeros_like(keep_sfft), None
    if exclude_stars:
        if psf_fine is None or not psf_oversample:
            raise ValueError("exclude_stars needs the effective PSF of the difference: psf_fine and its psf_oversample")
        idx = np.nonzero(sel.keep)[0]
        cat_s = prods["sci"].catalog
        test = psf_stars(np.where(nan, np.nan, sci_w), np.asarray(cat_s["X_IMAGE"], float)[sel.match.js[idx]] - 1.0,
                         np.asarray(cat_s["Y_IMAGE"], float)[sel.match.js[idx]] - 1.0, psf_fine, psf_oversample)
        psf_star[idx] = test["star"]
        sel.keep = sel.keep & ~psf_star
    sp = build_masks(np.where(nan, 0.0, ref_w).T, np.where(nan, 0.0, sci_w).T, prods["ref"], prods["sci"], sel,
                     dilate="AUTO" if grow is None else int(grow), ker_hw=int(gker_hw), sat_reject_pad=int(sat_pad))
    if detail is not None:
        detail.update(selection=sel, keep_sfft=keep_sfft, spike=spike, psf_star=psf_star, psf_test=test, prep=sp, zone=zone, guard_stars=stars)
    active = sp.active_mask
    guard_removed = 0
    if zone is not None:
        active = np.where(zone, overlap_mask(sp, prods["ref"], prods["sci"], int(sp.qa.get("dilate", 0))), active)
        guard_removed = int((sp.active_mask & ~active).sum())
    fit = np.ascontiguousarray(active.T) & ~nan
    rec = dict(mask="sfft sparse", tier=int(sel.tier), n_subsources=int(sp.qa.get("n_subsources", 0)), n_sat_rejected=int(sp.qa.get("n_sat_rejected", 0)),
               star_guard=bool(mode and sat_map is not None), star_guard_geometry=(mode or None) if sat_map is not None else None,
               guard_fraction=float(zone.mean()) if zone is not None else 0.0, n_spike_dropped=int(spike.sum()), guard_px_removed=guard_removed,
               n_corridor_stars=int(sum(s["ok"] for s in stars)) if stars is not None else 0,
               n_disc_stars=int(sum(not s["ok"] for s in stars)) if stars is not None else 0,
               exclude_stars=bool(exclude_stars), n_psf_tested=int(test["tested"].sum()) if test is not None else 0, n_psf_stars=int(psf_star.sum()),
               psf_core_residual=_core_residual(test),
               n_match=int(sel.qa.get("n_match", 0)), n_star_pairs=int(sel.qa.get("n_t1", 0)),
               dilate=int(sp.qa.get("dilate", 0)), sat_pad=int(sat_pad), saturation="flag" if sat_map is not None else "missing pixels", match_tol_px=match_tol_px,
               match_tol_used=float(sel.match.tol), locus_ref=float(prods["ref"].locus.peak),
               locus_sci=float(prods["sci"].locus.peak), fit_fraction=float(fit.mean()))
    return fit, rec


def sparse_sfft(sci_w, ref_w, fwhm_px, gker_hw, grow=None, sat_pad=2, sat_map=None, match_tol_px=None, star_guard=True, exclude_stars=False,
                psf_fine=None, psf_oversample=None, psf_nodes=None):
    """The fit mask of SFFT's sparse flavour on the matched pair: sparse_products, then sparse_mask with the given options. Returns (fit, record)."""
    return sparse_mask(sparse_products(sci_w, ref_w, fwhm_px), sci_w, ref_w, gker_hw, grow=grow, sat_pad=sat_pad,
                       sat_map=sat_map, match_tol_px=match_tol_px, star_guard=star_guard, exclude_stars=exclude_stars, psf_fine=psf_fine,
                       psf_oversample=psf_oversample, psf_nodes=psf_nodes)


def ellipticity(cat):
    """1 - B/A of a SExtractor catalogue."""
    return 1.0 - np.asarray(cat["B_IMAGE"], float) / np.maximum(np.asarray(cat["A_IMAGE"], float), 1e-6)


def signal_to_noise(cat):
    """FLUX_AUTO / FLUXERR_AUTO of a SExtractor catalogue (0 where the error is 0)."""
    f, e = np.asarray(cat["FLUX_AUTO"], float), np.asarray(cat["FLUXERR_AUTO"], float)
    return np.divide(f, e, out=np.zeros_like(f), where=e > 0)


def half_light_radius(stamp, oversample=1):
    """Radius [native px] holding half the flux of a centred PSF stamp; a coarse stamp is interpolated onto a finer grid first."""
    p = np.asarray(stamp, float)
    if oversample < 5:
        zoom = int(np.ceil(5 / oversample))
        p = ndimage.zoom(p, zoom, order=3, grid_mode=True, mode="nearest")
        oversample = oversample * zoom
    h = p.shape[0] // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
    r = np.hypot(xx, yy).ravel() / float(oversample)
    order = np.argsort(r)
    return float(np.interp(0.5, np.maximum.accumulate(np.cumsum(p.ravel()[order]) / p.sum()), r[order]))


def point_source_locus(cat, r_model=None, snr_min=50.0, h=0.04):
    """The point-source locus of FLUX_RADIUS near the radius the PSF model predicts (r_model): dict(locus, belt, source, n_bright, n_near, n_sharper)."""
    fr = np.asarray(cat["FLUX_RADIUS"], float)
    flags = np.asarray(cat["FLAGS"]).astype(int)
    ok = (signal_to_noise(cat) > snr_min) & (ellipticity(cat) < 0.3) & np.isfinite(fr) & (fr > 0.3) & ((flags & ~3) == 0)
    predicted = LOCUS_OVER_MODEL * float(r_model) if r_model else None
    lo, hi = (LOCUS_WINDOW[0] * predicted, LOCUS_WINDOW[1] * predicted) if predicted else (0.4, 5.0)
    r = fr[ok & (fr > lo) & (fr < hi)]
    locus, source = None, None
    if r.size:
        grid = np.arange(lo, hi + 0.005, 0.01)
        kde = np.exp(-0.5 * ((grid[:, None] - r[None, :]) / h) ** 2).sum(axis=1)
        for i in np.nonzero((kde[1:-1] > kde[:-2]) & (kde[1:-1] >= kde[2:]) & (kde[1:-1] > 0.25 * kde.max()))[0] + 1:
            if (np.abs(r - grid[i]) < 0.1).sum() >= MIN_LOCUS_STARS:
                locus, source = float(grid[i]), "measured"
                break
    if locus is None and predicted:
        locus, source = predicted, "psf_model"
    near = r[np.abs(r - locus) < 0.2] if locus is not None else r[:0]
    belt = float(np.clip(2.0 * near.std(), 0.08, 0.3)) if near.size > 3 else 0.15
    return dict(locus=locus, belt=belt, source=source, n_bright=int(ok.sum()), n_near=int(near.size), n_sharper=int((ok & (fr <= lo)).sum()))


def build(sci_w, ref_w, sigma, fwhm_px, gker_hw=4, thresh=2.0, star_extra_dilate=2, transients=None, transient_radius=None, fit_band=None, band_sigma=2.0,
          bright_dilate=12, faint_dilate=2, min_rhalf=None, r_psf=None, cats=None):
    """The band mask: galaxies in (detections off the point-source locus), stars and bad pixels out, optionally within a band of smoothed
    brightness (fit_band, in units of the pixel noise) and to sources larger than min_rhalf times the locus. Returns (fit, record, catalogues)."""
    nan = ~np.isfinite(sci_w) | ~np.isfinite(ref_w)
    sig_s, sig_r = sigma if isinstance(sigma, (tuple, list)) else (sigma, sigma)
    sigma = min(sig_s, sig_r)
    if cats is None:
        with ThreadPoolExecutor(2) as pool:
            job_r, job_s = pool.submit(detect, ref_w, thresh, mask=nan), pool.submit(detect, sci_w, thresh, mask=nan)
            cat_r, seg_r = job_r.result()
            cat_s, seg_s = job_s.result()
    else:
        cat_r, seg_r, cat_s, seg_s = cats["cat_ref"], cats["seg_ref"], cats["cat_sci"], cats["seg_sci"]
    loc = point_source_locus(cat_s if sig_s < sig_r else cat_r, r_psf)
    mode, width = loc["locus"], loc["belt"]
    rec = dict(n_ref=int(len(cat_r)), n_sci=int(len(cat_s)), r_psf=r_psf, locus_rhalf=mode, belt_halfwidth=width, locus_source=loc["source"], locus_stars=loc["n_near"], n_sharper_than_psf=loc["n_sharper"],
               n_overflow=int(((np.asarray(cat_r["FLAGS"]).astype(int) & 192) != 0).sum() + ((np.asarray(cat_s["FLAGS"]).astype(int) & 192) != 0).sum()))
    rh_r, rh_s = np.asarray(cat_r["FLUX_RADIUS"], float), np.asarray(cat_s["FLUX_RADIUS"], float)
    star_r = np.zeros(len(cat_r), bool)
    star_s = np.zeros(len(cat_s), bool)
    if mode is not None:
        star_r[:] = (np.abs(rh_r - mode) < width) & (ellipticity(cat_r) < 0.3)
        star_s[:] = (np.abs(rh_s - mode) < width) & (ellipticity(cat_s) < 0.3)
    rec.update(n_star_ref=int(star_r.sum()), n_star_sci=int(star_s.sum()))
    if min_rhalf is not None and mode is not None:
        cut = float(min_rhalf) * mode
        star_r |= rh_r < cut
        star_s |= rh_s < cut
        rec.update(min_rhalf_px=cut, n_compact_ref=int(star_r.sum()) - rec["n_star_ref"], n_compact_sci=int(star_s.sum()) - rec["n_star_sci"])
    dil = max(4, int(gker_hw))
    gal = np.zeros(sci_w.shape, bool)
    stars = np.zeros(sci_w.shape, bool)
    for seg, star in ((seg_r, star_r), (seg_s, star_s)):
        ids = np.arange(1, len(star) + 1)
        gal |= np.isin(seg, ids[~star])
        stars |= np.isin(seg, ids[star])
    gal = ndimage.binary_dilation(gal, iterations=dil)
    stars = ndimage.binary_dilation(stars, iterations=dil + int(star_extra_dilate))
    bad = ndimage.binary_dilation(nan, iterations=dil)
    fit = gal & ~stars & ~bad
    if fit_band is not None:
        sm_r = ndimage.gaussian_filter(np.where(nan, 0.0, ref_w), band_sigma)
        faint = float(fit_band[0]) * sigma if fit_band[0] is not None and float(fit_band[0]) > 0 else None
        bright = float(fit_band[1]) * sigma if fit_band[1] is not None else None
        if bright is not None:
            sm_s = ndimage.gaussian_filter(np.where(nan, 0.0, sci_w), band_sigma)
            too_bright = ndimage.binary_dilation((sm_s > bright) | (sm_r > bright), iterations=int(bright_dilate))
            fit = fit & ~too_bright
            rec.update(bright_fraction=float(too_bright.mean()))
        if faint is not None:
            fit = fit & ndimage.binary_dilation(sm_r > faint, iterations=int(faint_dilate))
        rec.update(fit_band_dn=[faint, bright])
    if transients is not None and transient_radius:
        yy, xx = np.mgrid[0:sci_w.shape[0], 0:sci_w.shape[1]]
        hole = np.zeros(sci_w.shape, bool)
        for (x, y) in transients:
            hole |= (xx - x) ** 2 + (yy - y) ** 2 <= float(transient_radius) ** 2
        fit &= ~hole
        rec["n_transient_holes"] = int(len(transients))
    rec.update(fit_fraction=float(fit.mean()), galaxy_fraction=float(gal.mean()), star_fraction=float(stars.mean()))
    return fit, rec, dict(cat_ref=cat_r, star_ref=star_r, cat_sci=cat_s, star_sci=star_s, seg_ref=seg_r, seg_sci=seg_s)
