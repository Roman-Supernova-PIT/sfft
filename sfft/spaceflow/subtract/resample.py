"""Resampling onto a target grid with drizzle (square kernel, pixel fraction 1), on the CPU with either backend.

A Level 2 image holds a surface brightness: each pixel is a weighted mean of the sky with unit-sum weights, a value at a sky position whatever
the pixel's size. Drizzle carries values: each target pixel is the weighted mean

    image = sum_i a_i w_i d_i / sum_i a_i w_i         (Fruchter and Hook 2002, Equations 4 and 5)

over the input pixels i it overlaps, with a_i the area of input pixel i inside the target pixel (in target pixels) and w_i the input weight
(1 for a usable pixel times the exposure's weight, 0 for an unusable one). So the image goes in as delivered: no pixel-area map and no
division by the target pixel's area. The variance goes through the same pass with squared weights, and the coverage is the summed weight:

    variance = sum_i a_i^2 w_i^2 v_i / (sum_i a_i w_i)^2,        coverage = sum_i a_i w_i.

Several exposures go into one drizzle product, each with its weight: the weighted mean over all of them. A target pixel whose coverage is
below min_coverage times the mean exposure weight has no value (NaN).

Three details of the implementation:
- drizzle (3.0) weights input pixel i by a_i / J_i, the share of the input pixel's own area J_i (in target pixels) that lands in the target
  pixel. The weight map given to drizzle therefore holds w_i J_i, which makes the weight a_i w_i and the coverage the covered share of the
  target pixel (1 inside the input frame, whatever the two pixel scales).
- drizzle scans only the input pixels inside a polygon with straight edges: the input box's corners taken onto the output grid, clipped
  to the output frame, and taken back. Where the distortion between the two grids bows an edge, the straight edges cut into the frames:
  drizzle drops the input frame's outermost row along an edge that bows outward on the output grid, and input pixels next to the output
  grid's edge (measured: edges bow by up to 2.9 pixels for one detector's distortion against a plain TAN grid). The output grid is made
  PAD pixels larger on every side and cut back (enough for edges that bow by up to about 12 pixels), and the input frame gets pixels of
  weight zero on every side, as many as the largest outward bow of its edges on the output grid (edge_bow) plus PAD_IN_MARGIN, so that
  drizzle's cuts fall only on pixels that carry nothing.
- the pixel map (the target position of every input pixel centre) is a bicubic spline of the exact projection through the two headers' WCS
  (astropy, SIP and TPV included) on a lattice STEP pixels apart with a margin; it is within 1e-8 pixel of the exact projection. drizzle
  finds each input pixel's corners from it.

drizzle computes in single precision (float32). Arrays are [y, x]."""
import time

import numpy as np

ARCSEC2_PER_SR = 206264.80624709636 ** 2
PAD = 16                # pixels added to the output grid on every side, then cut away (drizzle's edge convention)
PAD_IN_MARGIN = 4       # pixels of weight zero around the input frame beyond the outward bow of its edges on the target grid
BOW_MAX, BOW_FALLBACK = 32.0, 8.0   # the bow used at most, and when the projection gives no finite value [target pixels]
STEP, MARGIN = 16, 4    # the pixel map's exact lattice: STEP pixels apart, MARGIN cells beyond the input frame
WCS_TOL = 1e-11         # astropy's all_world2pix tolerance [pixels] (its default, 1e-4, is too loose here)


def target_area_sr(hdr):
    """The sky area of a target grid's pixels in sr: one number for a TAN grid without distortion terms, a map from its WCS for a grid
    with SIP terms."""
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_area
    w = WCS(hdr)
    if w.sip is None:
        return float(proj_plane_pixel_area(w)) * (np.pi / 180.0) ** 2
    from .geometry import pixel_area_map
    return pixel_area_map(w, (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))) / ARCSEC2_PER_SR


def exact_projection(hdr_in, hdr_out, x, y):
    """Target pixel positions (0-based) of input pixel positions (0-based), through the sky with the two headers' WCS."""
    from astropy.wcs import WCS
    ra, dec = WCS(hdr_in).all_pix2world(np.asarray(x, float), np.asarray(y, float), 0)
    xo, yo = WCS(hdr_out).all_world2pix(ra, dec, 0, tolerance=WCS_TOL, maxiter=200, adaptive=False, detect_divergence=True, quiet=True)
    return np.asarray(xo, float), np.asarray(yo, float)


def pixel_map(hdr_in, hdr_out, shape_in, step=STEP, margin=MARGIN, with_area=False, pad=0):
    """(ny, nx, 2): the target position (x, y; 0-based) of every input pixel centre, from bicubic splines of the exact projection on a
    coarse lattice; shape_in = (ny, nx). with_area: also each input pixel's area in target pixels, (ny, nx), the determinant of the map's
    derivatives from the same splines. pad: the map of the input frame grown by pad pixels on every side, (ny + 2 pad, nx + 2 pad, 2), its
    element [j, i] the input pixel (i - pad, j - pad)."""
    from scipy.interpolate import RectBivariateSpline
    ny, nx = int(shape_in[0]), int(shape_in[1])
    gx = np.arange(-margin * step, nx - 1 + (margin + 1) * step, step, dtype=float)
    gy = np.arange(-margin * step, ny - 1 + (margin + 1) * step, step, dtype=float)
    GX, GY = np.meshgrid(gx, gy)
    xo, yo = exact_projection(hdr_in, hdr_out, GX.ravel(), GY.ravel())
    sx = RectBivariateSpline(gy, gx, xo.reshape(GX.shape), kx=3, ky=3, s=0)
    sy = RectBivariateSpline(gy, gx, yo.reshape(GX.shape), kx=3, ky=3, s=0)
    yy, xx = np.arange(-pad, ny + pad, dtype=float), np.arange(-pad, nx + pad, dtype=float)
    pm = np.empty((len(yy), len(xx), 2), dtype=np.float64)
    pm[..., 0] = sx(yy, xx, grid=True)
    pm[..., 1] = sy(yy, xx, grid=True)
    if not with_area:
        return pm
    # RectBivariateSpline(gy, gx, .): dx is the derivative order along y (its first coordinate), dy along x
    area = np.abs(sx(yy, xx, dx=0, dy=1, grid=True) * sy(yy, xx, dx=1, dy=0, grid=True)
                  - sx(yy, xx, dx=1, dy=0, grid=True) * sy(yy, xx, dx=0, dy=1, grid=True))
    return pm, area


def edge_bow(hdr_in, hdr_out, shape_in, n=801):
    """The largest outward bow [target pixels] of the input frame's four edges on the target grid: how far the target positions of an
    edge's pixel centres lie outside the straight line through the edge's two corner pixels (0 for a distortion-free pair)."""
    ny, nx = int(shape_in[0]), int(shape_in[1])
    t = np.linspace(0.0, 1.0, n)
    cx, cy = exact_projection(hdr_in, hdr_out, np.array([(nx - 1) / 2.0]), np.array([(ny - 1) / 2.0]))
    centre = np.array([cx[0], cy[0]])
    bow = 0.0
    for x, y in ((t * (nx - 1), 0.0 * t), (t * (nx - 1), 0.0 * t + ny - 1), (0.0 * t, t * (ny - 1)), (0.0 * t + nx - 1, t * (ny - 1))):
        p = np.stack(exact_projection(hdr_in, hdr_out, x, y))
        a, d = p[:, 0], p[:, -1] - p[:, 0]
        normal = np.array([-d[1], d[0]]) / np.hypot(d[0], d[1])
        if np.dot(centre - a, normal) > 0:
            normal = -normal                     # pointing away from the frame's centre
        bow = max(bow, float(np.nanmax((p - a[:, None]).T @ normal)))
    return bow


def _input_box(pm, shape_out):
    """The input pixel box (xmin, xmax, ymin, ymax; 0-based, inclusive) whose centres land within two pixels of the output grid, or None."""
    ny_o, nx_o = shape_out
    inside = (pm[..., 0] > -2.5) & (pm[..., 0] < nx_o + 1.5) & (pm[..., 1] > -2.5) & (pm[..., 1] < ny_o + 1.5)
    if not inside.any():
        return None
    rows = np.nonzero(inside.any(axis=1))[0]
    cols = np.nonzero(inside.any(axis=0))[0]
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def stack(level2_list, hdr_targ, weights=None, min_coverage=0.5):
    """Weighted stack of exposures on the target grid: dict(image, variance, coverage, n_exp, members). Each exposure gives its header
    (with its WCS), image, variance and dnu (its unusable pixels); image and variance are NaN where the coverage is below min_coverage
    times the mean weight."""
    from drizzle.resample import Drizzle
    ny, nx = int(hdr_targ["NAXIS2"]), int(hdr_targ["NAXIS1"])
    shape_out = (ny + 2 * PAD, nx + 2 * PAD)
    cut = (slice(PAD, PAD + ny), slice(PAD, PAD + nx))
    driz = Drizzle(kernel="square", out_shape=shape_out, fillval=0.0, fillval2=0.0, disable_ctx=True)
    weights = [1.0] * len(level2_list) if weights is None else list(weights)
    nexp = np.zeros((ny, nx), dtype=np.int16)
    members = []
    for L, w in zip(level2_list, weights):
        t0 = time.time()
        img = np.asarray(L.image, np.float64)
        var = np.asarray(L.variance, np.float64)
        good = ~np.asarray(L.dnu, bool) & np.isfinite(img) & np.isfinite(var)
        bow = edge_bow(L.header, hdr_targ, img.shape)
        bow = min(bow, BOW_MAX) if np.isfinite(bow) else BOW_FALLBACK     # a projection far outside the target grid can run wild
        q = PAD_IN_MARGIN + int(np.ceil(bow))
        pm, area = pixel_map(L.header, hdr_targ, img.shape, margin=max(MARGIN, -(-(q + STEP) // STEP)), with_area=True, pad=q)
        pm += PAD
        before = np.array(driz.out_wht, dtype=np.float64)
        box = _input_box(pm, shape_out)
        if box is not None:
            xmin, xmax, ymin, ymax = box
            driz.add_image(np.pad(np.where(good, img, 0.0), q).astype(np.float32), exptime=1.0, pixmap=pm,
                           data2=np.pad(np.where(good, var, 0.0), q).astype(np.float32),
                           weight_map=(np.pad(good, q) * float(w) * area).astype(np.float32),
                           pixfrac=1.0, in_units="cps", xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        cov = (np.asarray(driz.out_wht, dtype=np.float64) - before)[cut]
        nexp += (cov > 0)
        members.append(dict(file=L.meta["filename"], weight=float(w), seconds=round(time.time() - t0, 1), covered=float((cov > 0).mean()),
                            input_pad=q))
    den = np.asarray(driz.out_wht, dtype=np.float64)[cut]
    out_img = np.asarray(driz.out_img, dtype=np.float64)[cut]
    out2 = driz.out_img2
    out_var = np.asarray(out2[0], dtype=np.float64)[cut] if out2 and out2[0] is not None else np.zeros((ny, nx))
    good = den >= min_coverage * np.mean(weights)
    return dict(image=np.where(good, out_img, np.nan), variance=np.where(good, out_var, np.nan), coverage=den, n_exp=nexp, members=members)


def exposure_on_own_grid(level2, hdr_targ):
    """One exposure as a stack of itself on its own pixel grid, or on a whole-pixel crop of it: nothing is resampled. Returns the dict of
    stack(), image and variance NaN at unusable pixels. Raises ValueError when hdr_targ is not the exposure's own grid or such a crop."""
    from astropy.wcs import WCS
    t0 = time.time()
    ny, nx = int(hdr_targ["NAXIS2"]), int(hdr_targ["NAXIS1"])
    x0 = float(level2.header["CRPIX1"]) - float(hdr_targ["CRPIX1"])
    y0 = float(level2.header["CRPIX2"]) - float(hdr_targ["CRPIX2"])
    ix, iy = int(round(x0)), int(round(y0))
    inside = abs(x0 - ix) < 1e-6 and abs(y0 - iy) < 1e-6 and ix >= 0 and iy >= 0 and iy + ny <= level2.shape[0] and ix + nx <= level2.shape[1]
    if inside:
        u = np.array([0.0, nx - 1.0, 0.0])
        v = np.array([0.0, 0.0, ny - 1.0])
        a = np.array(WCS(hdr_targ).all_pix2world(u, v, 0))
        b = np.array(WCS(level2.header).all_pix2world(u + ix, v + iy, 0))
        inside = bool(np.max(np.abs(a - b)) < 1e-8)
    if not inside:
        raise ValueError(f"the grid is not {level2.meta['filename']}'s own pixel grid or a whole-pixel cut of it inside the exposure")
    sl = (slice(iy, iy + ny), slice(ix, ix + nx))
    good = ~level2.dnu[sl]
    image = np.where(good, level2.image[sl], np.nan)
    variance = np.where(good, level2.variance[sl], np.nan)
    members = [dict(file=level2.meta["filename"], weight=1.0, seconds=round(time.time() - t0, 1), covered=float(good.mean()), resampled=False)]
    return dict(image=image, variance=variance, coverage=good.astype(np.float64), n_exp=good.astype(np.int16), members=members)
