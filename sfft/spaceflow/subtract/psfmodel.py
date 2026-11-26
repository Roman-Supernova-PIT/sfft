"""The PSF of an exposure on the target grid, as a fine-sampled stamp in the target frame (the working model of the cross-convolution).

The response of a resampled exposure to a source has a part common to all sources and a part that depends on the source's sub-pixel position.
The working model is the common part,

    P = G x box_in x box_in x box_out

(G the PSF without the pixel; one input pixel for the sampling, one for the overlap kernel, one output pixel), turned into the target frame by
the local Jacobian. A PSF that already holds the pixel contributes one box less; an exposure on its own grid has G x box_in. Where the
exposure's pixel lattice keeps nearly the same sub-pixel offset against the target's over a node's area (the same orientation), the model of
the offsets present replaces the mean over offsets: the mean over offsets plus the mean fixed-offset overlap over the offsets present minus its
mean over all offsets. Stamps keep their sum, the share of a point source's light they hold."""
from collections import namedtuple

import numpy as np
from scipy import ndimage

PsfSample = namedtuple("PsfSample", "stamp oversample pixel_integrated")
LOCK_DRIFT_PX = 0.1
COHERENCE_MIN = 1e-3
OFFSET_BINS = 128


class NotCovered(ValueError):
    """No exposure of the group covers this position."""
    pass


def _box_ft(fine_shape, oversample):
    """The pixel's transfer function on a fine FFT grid."""
    fy = np.fft.fftfreq(fine_shape[0])[:, None]
    fx = np.fft.fftfreq(fine_shape[1])[None, :]
    return np.sinc(oversample * fy) * np.sinc(oversample * fx)


def _object_position(level2, ra, dec):
    """The detector position (x, y) of a sky position in an exposure."""
    x, y = level2.wcs.all_world2pix(ra, dec, 0)
    return float(x), float(y)


def source_positions(level2, x, y):
    """The source positions r of points whose WCS positions are (x, y): r = H^-1(eta) for an exposure with a WCS offset vector
    (level2.wcs_offset_vector), r = eta otherwise."""
    H = getattr(getattr(level2, "full", level2), "wcs_offset_vector", None)
    if H is None:
        return x, y
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        if not (np.isfinite(x) and np.isfinite(y)):
            return x, y
        r = H.inverse(np.array([float(x), float(y)]))
        return float(r[0]), float(r[1])
    eta = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float)), axis=-1)
    r = eta.copy()
    ok = np.all(np.isfinite(eta), axis=-1)
    if ok.any():
        r[ok] = H.inverse(eta[ok])
    return r[..., 0], r[..., 1]


def local_jacobian(level2, hdr_targ_wcs, xo, yo, step=2.0):
    """d(target pixel) / d(detector pixel) at a detector position, by central differences through the sky."""
    pts = np.array([[xo - step, yo], [xo + step, yo], [xo, yo - step], [xo, yo + step]])
    ra, dec = level2.wcs.all_pix2world(pts[:, 0], pts[:, 1], 0)
    xt, yt = hdr_targ_wcs.all_world2pix(ra, dec, 0)
    return np.array([[(xt[1] - xt[0]) / (2 * step), (xt[3] - xt[2]) / (2 * step)],
                     [(yt[1] - yt[0]) / (2 * step), (yt[3] - yt[2]) / (2 * step)]])


def phase_lock(level2, w_t, ra, dec, extent):
    """The sub-pixel offset (dx, dy) of the input lattice at a target pixel when the exposure is locked to the target's lattice (a signed
    permutation Jacobian, the offset drifting by less than LOCK_DRIFT_PX over extent target pixels); None otherwise."""
    xo, yo = _object_position(level2, ra, dec)
    jac = local_jacobian(level2, w_t, xo, yo)
    if np.abs(jac - np.round(jac)).max() * extent > LOCK_DRIFT_PX or abs(abs(np.linalg.det(np.round(jac))) - 1.0) > 1e-6:
        return None
    xt, yt = w_t.all_world2pix(ra, dec, 0)
    r0, d0 = w_t.all_pix2world(np.round(xt), np.round(yt), 0)
    xi, yi = level2.wcs.all_world2pix(r0, d0, 0)
    xi, yi = float(xi), float(yi)
    return xi - round(xi), yi - round(yi)


def _overlap_ft(fine_shape, oversample, lock):
    """Transfer function of the overlap kernel at a fixed offset: two taps per axis, weights 1 - |d| and |d|."""
    ky = np.fft.fftfreq(fine_shape[0])[:, None] * oversample
    kx = np.fft.fftfreq(fine_shape[1])[None, :] * oversample

    def axis(k, d):
        return (1.0 - abs(d)) * np.exp(-2j * np.pi * k * d) + abs(d) * np.exp(-2j * np.pi * k * (d - np.sign(d)))

    return axis(ky, lock[1]) * axis(kx, lock[0])


def _axis_overlap(k, d):
    """The fixed-offset overlap along one axis for many offsets: [n_offsets, n_frequencies]."""
    d = np.asarray(d, np.float64)[:, None]
    k = np.asarray(k, np.float64)[None, :]
    return (1.0 - np.abs(d)) * np.exp(-2j * np.pi * k * d) + np.abs(d) * np.exp(-2j * np.pi * k * (d - np.sign(d)))


def _circular_bins(d, w, nbins=OFFSET_BINS):
    """Bins over the part of one pixel the offsets occupy, centred on their circular mean: (each sample's bin, each bin's weighted mean
    offset, the width occupied)."""
    z = np.sum(w * np.exp(2j * np.pi * d))
    c = float(np.angle(z) / (2.0 * np.pi)) if abs(z) > 1e-9 * np.sum(w) else 0.0
    u = (d - c + 0.5) % 1.0 - 0.5
    lo, hi = float(u.min()), float(u.max())
    width = max(hi - lo, 1e-9) / nbins
    idx = np.clip(((u - lo) / width).astype(np.int64), 0, nbins - 1)
    sw = np.bincount(idx, weights=w, minlength=nbins)
    su = np.bincount(idx, weights=w * u, minlength=nbins)
    mean_u = np.where(sw > 0, su / np.where(sw > 0, sw, 1.0), lo + (np.arange(nbins) + 0.5) * width)
    return idx, (mean_u + c + 0.5) % 1.0 - 0.5, hi - lo


def _present_overlap_ft(fine_shape, oversample, dx, dy, w):
    """The mean fixed-offset overlap over the offsets present minus its mean over all offsets, on a stamp's fine FFT grid."""
    dx, dy, w = (np.asarray(a, np.float64).ravel() for a in (dx, dy, w))
    keep = w > 0
    dx, dy, w = dx[keep], dy[keep], w[keep]
    ky = np.fft.fftfreq(fine_shape[0]) * oversample
    kx = np.fft.fftfreq(fine_shape[1]) * oversample
    ix, cx, _ = _circular_bins(dx, w)
    iy, cy, _ = _circular_bins(dy, w)
    rho = np.bincount(iy * OFFSET_BINS + ix, weights=w, minlength=OFFSET_BINS * OFFSET_BINS).reshape(OFFSET_BINS, OFFSET_BINS) / np.sum(w)
    Tbar = _axis_overlap(ky, cy).T @ rho @ _axis_overlap(kx, cx)
    return Tbar - (np.sinc(ky) ** 2)[:, None] * (np.sinc(kx) ** 2)[None, :]


def footprint_offsets(level2, w_t, fx, wx, fy, wy):
    """The sub-pixel offsets of an exposure's pixel lattice at the target pixels one PSF node's filter serves, with the node's
    interpolation weights: ((dx, dy, w), record), or (None, record) when the mean over offsets holds as it is (the lattices are not parallel, or
    the offsets fill a pixel evenly)."""
    from scipy.interpolate import RectBivariateSpline
    L = getattr(level2, "full", level2)
    fx, wx, fy, wy = (np.asarray(a, np.float64) for a in (fx, wx, fy, wy))
    gx, gy = np.linspace(fx[0], fx[-1], 9), np.linspace(fy[0], fy[-1], 9)
    GX, GY = np.meshgrid(gx, gy)
    ra, dec = w_t.all_pix2world(GX.ravel(), GY.ravel(), 0)
    xd, yd = L.wcs.all_world2pix(ra, dec, 0, quiet=True)
    xd, yd = np.asarray(xd, np.float64).reshape(9, 9), np.asarray(yd, np.float64).reshape(9, 9)
    A = np.column_stack([GX.ravel(), GY.ravel(), np.ones(GX.size)])
    J = np.array([np.linalg.lstsq(A, xd.ravel(), rcond=None)[0][:2], np.linalg.lstsq(A, yd.ravel(), rcond=None)[0][:2]])
    J0 = np.round(J)
    rec = dict(coherence=0.0, why=None, spread_px=None)
    if abs(abs(np.linalg.det(J0)) - 1.0) > 1e-6 or np.abs(J0).sum() != 2:
        rec["why"] = "the lattices are not parallel"
        return None, rec
    D = J - J0
    coh = 0.0
    for m in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)):
        g = D.T @ np.array(m)
        coh = max(coh, abs(np.sum(wx * np.exp(2j * np.pi * g[0] * fx))) / np.sum(wx) * abs(np.sum(wy * np.exp(2j * np.pi * g[1] * fy))) / np.sum(wy))
    rec["coherence"] = float(coh)
    if coh < COHERENCE_MIN:
        rec["why"] = "the offsets fill a pixel evenly"
        return None, rec
    step = 1 if max(len(fx), len(fy)) <= 2048 else 2
    sx, sy, swx, swy = fx[::step], fy[::step], wx[::step], wy[::step]
    XD = RectBivariateSpline(gy, gx, xd, kx=3, ky=3)(sy, sx)
    YD = RectBivariateSpline(gy, gx, yd, kx=3, ky=3)(sy, sx)
    ny_d, nx_d = L.shape
    w = swy[:, None] * swx[None, :] * ((XD >= -0.5) & (XD <= nx_d - 0.5) & (YD >= -0.5) & (YD <= ny_d - 0.5))
    if not np.any(w > 0):
        rec["why"] = "no pixel of the area on the exposure"
        return None, rec
    dx, dy = XD - np.round(XD), YD - np.round(YD)
    keep = w > 0
    rec["spread_px"] = [float(_circular_bins(dx[keep], w[keep])[2]), float(_circular_bins(dy[keep], w[keep])[2])]
    return (dx, dy, w), rec


def averaged_exposure_stamp(stamp_yx, os_in, jac, out_oversamp=5, half_width=48, resampled=True, pixel_integrated=False, order=5, lock=None,
                            offset=(0.0, 0.0), offsets=None):
    """The PSF of one exposure in the target frame, [y, x] on a grid of 1/out_oversamp target pixels: the mean over offsets, or with lock
    the response at that fixed offset, or with offsets the model of the offsets present. stamp_yx is on a grid of 1/os_in detector pixels, jac
    is d(target)/d(detector), offset the WCS offset vector at the source. The stamp's sum is kept."""
    E = np.asarray(stamp_yx, np.float64)
    corr = None
    if offsets is not None:
        if lock is not None or not resampled:
            raise ValueError("offsets present apply to a resampled exposure without a lock")
        Fc = np.fft.fft2(E) * _present_overlap_ft(E.shape, os_in, *offsets)
        if not pixel_integrated:
            Fc = Fc * _box_ft(E.shape, os_in)
        corr = np.real(np.fft.ifft2(Fc))
    if lock is not None:
        n_in = 0 if pixel_integrated else 1
        F = np.fft.fft2(E) * _overlap_ft(E.shape, os_in, lock)
        if n_in:
            F = F * _box_ft(E.shape, os_in)
        E = np.real(np.fft.ifft2(F))
        resampled = False
    else:
        n_in = (2 if resampled else 1) - (1 if pixel_integrated else 0)
        if n_in > 0:
            E = np.real(np.fft.ifft2(np.fft.fft2(E) * _box_ft(E.shape, os_in) ** n_in))
    nfy, nfx = E.shape
    cy, cx = (nfy - 1) / 2.0, (nfx - 1) / 2.0
    H = int(round(half_width * out_oversamp))
    ax = np.arange(-H, H + 1) / float(out_oversamp)
    vv, uu = np.meshgrid(ax, ax, indexing="ij")
    Jinv = np.linalg.inv(jac)
    dx = Jinv[0, 0] * uu + Jinv[0, 1] * vv + float(offset[0])
    dy = Jinv[1, 0] * uu + Jinv[1, 1] * vv + float(offset[1])
    coords = np.array([cy + dy * os_in, cx + dx * os_in])
    P = ndimage.map_coordinates(E, coords, order=order, mode="constant", cval=0.0, prefilter=True)
    P *= (float(os_in) / float(out_oversamp)) ** 2 / abs(float(np.linalg.det(jac)))
    if resampled:
        P = np.real(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(P)) * _box_ft(P.shape, out_oversamp)))
        P = np.fft.fftshift(P)
    if corr is not None:
        Pc = ndimage.map_coordinates(corr, coords, order=order, mode="constant", cval=0.0, prefilter=True)
        P = P + Pc * ((float(os_in) / float(out_oversamp)) ** 2 / abs(float(np.linalg.det(jac))))
    return P


def averaged_stamp(level2_list, hdr_targ, ra, dec, epsf_of, out_oversamp=5, half_width=48, weights=None, resampled=True, footprint=None, info=None):
    """The PSF of a stack at (ra, dec): the weighted mean of its exposures' models, each read from epsf_of(exposure, x, y) at the source's
    detector position; with footprint, the model of the offsets present; info receives a record per exposure. Returns (stamp, exposures used,
    of them with a fixed-offset or offsets-present model)."""
    from astropy.wcs import WCS
    w_t = WCS(hdr_targ)
    extent = 0.5 * max(int(hdr_targ["NAXIS1"]), int(hdr_targ["NAXIS2"]))
    weights = [1.0] * len(level2_list) if weights is None else list(weights)
    num, wsum, used, locked = None, 0.0, 0, 0
    for L, w in zip(level2_list, weights):
        L = getattr(L, "full", L)
        xo, yo = _object_position(L, ra, dec)
        if not (0 <= xo <= L.shape[1] - 1 and 0 <= yo <= L.shape[0] - 1):
            continue
        rx, ry = source_positions(L, xo, yo)
        ps = epsf_of(L, rx, ry)
        jac = local_jacobian(L, w_t, xo, yo)
        lock = offs = None
        if resampled and footprint is None:
            lock = phase_lock(L, w_t, ra, dec, extent)
        elif resampled:
            offs, rec = footprint_offsets(L, w_t, *footprint)
            if info is not None:
                info.append(dict(file=getattr(L, "meta", {}).get("filename"), model="offsets present" if offs is not None else "mean over phases",
                                 **rec))
        P = averaged_exposure_stamp(ps.stamp, ps.oversample, jac, out_oversamp, half_width, resampled=resampled,
                                    pixel_integrated=ps.pixel_integrated, lock=lock, offset=(xo - rx, yo - ry), offsets=offs)
        num = w * P if num is None else num + w * P
        wsum += w
        used += 1
        locked += (lock is not None) or (offs is not None)
    if num is None:
        raise NotCovered("no exposure covers this position")
    return num / wsum, used, locked


def node_stamps(level2_list, hdr_targ, xs, ys, epsf_of, out_oversamp=5, half_width=48, weights=None, resampled=True, present_offsets=False,
                info=None, node_psf=None):
    """PSF models at the nodes (xs, ys; 0-based target pixels): [len(ys)][len(xs)] stamps; a node no exposure covers takes the nearest
    covered node's. present_offsets: each resampled exposure's model at a node from the offsets present at the pixels its filter serves.
    node_psf(j, i): the PsfSample of node (j, i), in place of epsf_of."""
    from astropy.wcs import WCS

    from .crossconv import _tri_weights
    w = WCS(hdr_targ)
    if present_offsets:
        WX = _tri_weights(np.arange(int(hdr_targ["NAXIS1"]), dtype=np.float64), np.asarray(xs, np.float64))
        WY = _tri_weights(np.arange(int(hdr_targ["NAXIS2"]), dtype=np.float64), np.asarray(ys, np.float64))
    arr = [[None] * len(xs) for _ in ys]
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ra, dec = w.all_pix2world(float(x), float(y), 0)
            fp = None
            if present_offsets:
                kx, ky = np.nonzero(WX[i] > 0)[0], np.nonzero(WY[j] > 0)[0]
                fp = (kx.astype(np.float64), WX[i][kx], ky.astype(np.float64), WY[j][ky])
            node_info = [] if info is not None else None
            try:
                source = epsf_of if node_psf is None else (lambda L, rx, ry, j=j, i=i: node_psf(j, i))
                arr[j][i] = averaged_stamp(level2_list, hdr_targ, float(ra), float(dec), source, out_oversamp, half_width, weights, resampled,
                                           footprint=fp, info=node_info)[0]
            except NotCovered:
                pass
            if info is not None:
                info.append(dict(node=(j, i), exposures=node_info))
    ok = np.array([[a is not None for a in row] for row in arr])
    if not ok.any():
        raise ValueError("no PSF node of this grid falls on these exposures: the grid (or its crop) does not overlap them")
    jj, ii = np.nonzero(ok)
    for j in range(len(ys)):
        for i in range(len(xs)):
            if arr[j][i] is None:
                m = np.argmin((jj - j) ** 2 + (ii - i) ** 2)
                arr[j][i] = arr[jj[m]][ii[m]]
    return arr


def fwhm_px(stamp_fine, oversample):
    """FWHM in native pixels, from the area above half maximum."""
    p = np.asarray(stamp_fine)
    return float(2.0 * np.sqrt((p >= 0.5 * p.max()).sum() / np.pi) / oversample)
