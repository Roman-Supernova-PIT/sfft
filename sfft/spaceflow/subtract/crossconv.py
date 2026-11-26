"""Exact cross-convolution of a science/reference pair on one grid with fine-sampled PSF models:

    S' = F^-1[A S^],   R' = F^-1[B R^],   A = C r / N,   B = C s / N,

with one common N, so that A Ps = B Pr and a static source cancels for any N. A pixel grid cannot tell a frequency k from k - n (n whole
numbers), so every product of the two PSFs is summed over those folded copies (the folded filters):

    X = sum_n Pr(k - n) conj Ps(k - n),   Y = sum_n |Ps(k - n)|^2,   s = sqrt(Y),   r = X / s,
    N^2 = sig_s^2 |r|^2 + sig_r^2 s^2,   C = conj(r + s) / |r + s| (the phase the two share).

The sums repeat from one edge of the band to the other, so the filters cannot jump at the Nyquist frequency, and the noise stays white. The
filters are built per PSF node and interpolated bilinearly between nodes; A = 1 at zero frequency, so the matched science image keeps a
source's flux."""
import numpy as np
from scipy import fft, ndimage

WORKERS = 8


def otf_on_grid(stamp_fine, oversample, shape, xp=np):
    """Fourier transform of a fine PSF stamp ([y, x], spacing 1/oversample pixel, centred) on the rfft2 frequency grid of an image of shape;
    its zero frequency is the stamp's sum."""
    P = np.asarray(stamp_fine, np.float64)
    h = 1.0 / float(oversample)
    nfy, nfx = P.shape
    xs = (np.arange(nfx) - (nfx - 1) / 2.0) * h
    ys = (np.arange(nfy) - (nfy - 1) / 2.0) * h
    ny, nx = shape
    kx = np.fft.rfftfreq(nx)
    ky = np.fft.fftfreq(ny)
    Ex = xp.asarray(np.exp(-2j * np.pi * np.outer(kx, xs)))
    Ey = xp.asarray(np.exp(-2j * np.pi * np.outer(ky, ys)))
    return (Ey @ xp.asarray(P.astype(np.complex128))) @ Ex.T


def robust_sigma(a):
    """1.4826 times the median absolute deviation of the finite values."""
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else float("nan")


def source_free(images, fwhm_px, nsig=2.0, grow=None):
    """Pixels free of sources in every image (PSF-smoothed S/N below nsig, grown by grow pixels, by default 2 FWHM) and finite in all."""
    good = np.ones(images[0].shape, bool)
    for im in images:
        good &= np.isfinite(im)
    src = np.zeros_like(good)
    for im in images:
        sm = ndimage.gaussian_filter(np.where(good, im, 0.0), fwhm_px / 2.355)
        v = sm[good]
        src |= (sm - np.median(v)) > nsig * robust_sigma(v)
    grow = max(1, int(round(2 * fwhm_px))) if grow is None else int(grow)
    return ~(ndimage.binary_dilation(src, iterations=grow) | ndimage.binary_dilation(~good, iterations=grow))


def node_grid(shape, n):
    """Node centres (x, y) of an n x n grid on an image [ny, nx]: the cell centres."""
    ny, nx = shape
    xs = (np.arange(n) + 0.5) * nx / n - 0.5
    ys = (np.arange(n) + 0.5) * ny / n - 0.5
    return xs, ys


def _tri_weights(coord, nodes):
    """Piecewise-linear partition of unity on 1-D nodes, constant beyond the end nodes: (n_nodes, n_coord)."""
    w = np.zeros((len(nodes), len(coord)))
    for i, c in enumerate(nodes):
        e = np.zeros(len(nodes))
        e[i] = 1.0
        w[i] = np.interp(coord, nodes, e)
    return w


def inpaint_holes(img, sigma=1.5):
    """Fill NaN pixels from their neighbours (normalised Gaussian convolution), falling to zero far from any data."""
    good = np.isfinite(img)
    if good.all():
        return img
    num = ndimage.gaussian_filter(np.where(good, img, 0.0), sigma)
    den = ndimage.gaussian_filter(good.astype(np.float64), sigma)
    fill = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-4) * np.minimum(den / 0.5, 1.0)
    return np.where(good, img, fill)


def matching_filters(Ps, Pr, sig_s, sig_r, floor=1e-5, xp=np, common_phase=True):
    """The filters A (science) and B (reference) from two transfer functions on a lattice: A = C Pr / N, B = C Ps / N,
    N^2 = sig_s^2 |Pr|^2 + sig_r^2 |Ps|^2 plus a floor, C = conj(Ps + Pr) / |Ps + Pr|; both divided by Re A at zero frequency."""
    den = sig_s ** 2 * xp.abs(Pr) ** 2 + sig_r ** 2 * xp.abs(Ps) ** 2 + (floor * (sig_s + sig_r)) ** 2
    invN = 1.0 / xp.sqrt(den)
    A, B = Pr * invN, Ps * invN
    if common_phase:
        m = Ps + Pr
        mag = xp.abs(m)
        C = xp.where(mag > 1e-12, xp.conj(m) / xp.maximum(mag, 1e-300), 1.0 + 0.0j)
        A, B = A * C, B * C
    a0 = xp.real(A[0, 0])
    return A / a0, B / a0


def lattice_products(stamp_s, stamp_r, oversample, shape, xp=np, workers=WORKERS):
    """X = sum_n Pr(k - n) conj Ps(k - n) and Y = sum_n |Ps(k - n)|^2 on the rfft2 grid of shape, as the lattice transforms of the two
    stamps' cross- and auto-correlation at whole-pixel lags (times oversample^2). Both stamps on one fine grid."""
    os_ = int(oversample)
    ps, pr = np.asarray(stamp_s, np.float64), np.asarray(stamp_r, np.float64)
    if ps.shape != pr.shape:
        raise ValueError(f"the two PSF stamps must share one fine grid: shapes {ps.shape} and {pr.shape}")
    nfy, nfx = ps.shape
    ny, nx = shape
    jy, jx = np.arange(-((nfy - 1) // os_), (nfy - 1) // os_ + 1), np.arange(-((nfx - 1) // os_), (nfx - 1) // os_ + 1)
    if len(jy) > ny or len(jx) > nx:
        raise ValueError(f"the lattice {shape} is smaller than the stamps' whole-pixel lags ({len(jy)} x {len(jx)})")
    my, mx = fft.next_fast_len(2 * nfy - 1, real=True), fft.next_fast_len(2 * nfx - 1, real=True)
    Fs = fft.rfft2(ps, (my, mx), workers=workers)
    Fr = fft.rfft2(pr, (my, mx), workers=workers)
    out = []
    for F in (Fr * np.conj(Fs), Fs * np.conj(Fs)):
        c = fft.irfft2(F, (my, mx), workers=workers)
        lat = np.zeros((ny, nx))
        lat[np.ix_(jy % ny, jx % nx)] = c[np.ix_((jy * os_) % my, (jx * os_) % mx)]
        out.append(fft.rfft2(lat, workers=workers) * float(os_ * os_))
    return xp.asarray(out[0]), xp.asarray(np.real(out[1]))


def lattice_filters(X, Y, sig_s, sig_r, floor=1e-5, xp=np):
    """The folded filters: matching_filters with s = sqrt(Y) and r = X / s (s held at 1e-6 s(0) or more in the division)."""
    s = xp.sqrt(xp.maximum(Y, 0.0))
    r = X / xp.maximum(s, 1e-6 * s[0, 0])
    return matching_filters(s, r, sig_s, sig_r, floor, xp=xp)


def seam_taper(shape, alpha, xp=np):
    """A separable taper, 1 on the central fraction alpha of each frequency axis and smoothly 0 at Nyquist (C2 smootherstep)."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"the taper's flat fraction alpha must be between 0 and 1, not {alpha!r}")

    def t1d(k, n):
        u = np.clip((2.0 * k / n - alpha) / (1.0 - alpha), 0.0, 1.0)
        return 1.0 - u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)

    ny, nx = shape
    return xp.asarray(np.outer(t1d(np.abs(np.round(np.fft.fftfreq(ny) * ny)), ny), t1d(np.arange(nx // 2 + 1, dtype=float), nx)))


def effective_psf_fine(psf_sci_fine, psf_ref_fine, oversample, sig_s, sig_r, half=20, lattice=1024, floor=1e-5, taper=None, fold=False):
    """A transient's PSF in the matched science image as a function of its position inside a pixel: a stamp on the fine grid
    ((2 half + 1) oversample samples a side) whose samples every oversample fine pixels, from fine offset q, are the difference's pixels for a
    transient q/oversample pixel from a pixel centre; for unit flux."""
    os_ = int(oversample)
    L = int(lattice)
    if fold:
        A, _ = lattice_filters(*lattice_products(psf_sci_fine, psf_ref_fine, os_, (L, L)), sig_s, sig_r, floor)
    else:
        Ps = otf_on_grid(psf_sci_fine, os_, (L, L))
        Pr = otf_on_grid(psf_ref_fine, os_, (L, L))
        A, _ = matching_filters(Ps, Pr, sig_s, sig_r, floor)
    if taper is not None:
        A = A * seam_taper((L, L), taper)
    E = np.asarray(psf_sci_fine, np.float64)
    c = (E.shape[0] - 1) // 2
    K = c // os_
    n = np.arange(-K, K + 1)
    out = np.zeros(((2 * half + 1) * os_, (2 * half + 1) * os_))
    h = os_ // 2
    for qy in range(-h, os_ - h):
        for qx in range(-h, os_ - h):
            iy = c + n * os_ + qy
            ix = c + n * os_ + qx
            ok_y, ok_x = (iy >= 0) & (iy < E.shape[0]), (ix >= 0) & (ix < E.shape[1])
            s = np.zeros((L, L))
            s[np.ix_(n[ok_y] % L, n[ok_x] % L)] = E[np.ix_(iy[ok_y], ix[ok_x])]
            Sq = fft.rfft2(s)
            t = np.roll(fft.irfft2(A * Sq, (L, L)), (half, half), axis=(0, 1))[:2 * half + 1, :2 * half + 1]
            out[(qy + h)::os_, (qx + h)::os_] = t
    return out * os_ ** 2


def match_pair(sci, ref, psf_sci_nodes, psf_ref_nodes, oversample, sig_s, sig_r, xs=None, ys=None, pad=128, kernel_lattice=1024, backend="gpu", floor=1e-5,
               taper=None, fold=False, filter_kernels=None, kernel_half=32):
    """The cross-convolution of a pair with filters per node, interpolated bilinearly between nodes: dict(sci, ref, otf_eff {(j, i): A Ps on
    the lattice}, lattice, xs, ys). NaN pixels are filled before the transforms and are NaN after. fold: the folded filters; taper: alpha of
    seam_taper on both filters; filter_kernels: a dict that receives {(j, i): (a, b)}, each node's two filters as real-space kernels
    ((2 kernel_half + 1) pixels a side, centred, [y, x] numpy), for propagating variances."""
    if backend == "gpu":
        import cupy as xp
        F, fkw = xp.fft, {}
    else:
        xp = np
        F, fkw = fft, dict(workers=WORKERS)
    if not isinstance(psf_sci_nodes, (list, tuple)):
        psf_sci_nodes, psf_ref_nodes = [[psf_sci_nodes]], [[psf_ref_nodes]]
    nyn, nxn = len(psf_sci_nodes), len(psf_sci_nodes[0])
    ny, nx = sci.shape
    L = int(kernel_lattice)
    nan = ~np.isfinite(sci) | ~np.isfinite(ref)
    shape = (fft.next_fast_len(max(ny + 2 * pad, L), real=True), fft.next_fast_len(max(nx + 2 * pad, L), real=True))
    s = xp.zeros(shape, dtype=xp.float64)
    r = xp.zeros(shape, dtype=xp.float64)
    s[pad:pad + ny, pad:pad + nx] = xp.asarray(inpaint_holes(np.where(nan, np.nan, sci)))
    r[pad:pad + ny, pad:pad + nx] = xp.asarray(inpaint_holes(np.where(nan, np.nan, ref)))
    Sh, Rh = F.rfft2(s, **fkw), F.rfft2(r, **fkw)
    del s, r
    if xs is None:
        xs = node_grid((ny, nx), nxn)[0] if nxn > 1 else np.array([(nx - 1) / 2.0])
        ys = node_grid((ny, nx), nyn)[1] if nyn > 1 else np.array([(ny - 1) / 2.0])
    wx = xp.asarray(_tri_weights(np.arange(nx, dtype=float), xs))
    wy = xp.asarray(_tri_weights(np.arange(ny, dtype=float), ys))
    out_s = xp.zeros((ny, nx), dtype=xp.float64)
    out_r = xp.zeros((ny, nx), dtype=xp.float64)
    otf_eff = {}
    T = None if taper is None else seam_taper((L, L), taper, xp=xp)
    for j in range(nyn):
        for i in range(nxn):
            Ps = otf_on_grid(psf_sci_nodes[j][i], oversample, (L, L), xp=xp)
            if fold:
                A, B = lattice_filters(*lattice_products(psf_sci_nodes[j][i], psf_ref_nodes[j][i], oversample, (L, L), xp=xp), sig_s, sig_r,
                                       floor, xp=xp)
            else:
                Pr = otf_on_grid(psf_ref_nodes[j][i], oversample, (L, L), xp=xp)
                A, B = matching_filters(Ps, Pr, sig_s, sig_r, floor, xp=xp)
                del Pr
            if T is not None:
                A, B = A * T, B * T
            ka, kb = F.irfft2(A, s=(L, L), **fkw), F.irfft2(B, s=(L, L), **fkw)
            FA = F.rfft2(_pad_wrapped(ka, shape, xp), **fkw)
            FB = F.rfft2(_pad_wrapped(kb, shape, xp), **fkw)
            if filter_kernels is not None:
                filter_kernels[(j, i)] = (_centred_crop(ka, kernel_half), _centred_crop(kb, kernel_half))
            ss = F.irfft2(FA * Sh, s=shape, **fkw)[pad:pad + ny, pad:pad + nx]
            rr = F.irfft2(FB * Rh, s=shape, **fkw)[pad:pad + ny, pad:pad + nx]
            weight = xp.outer(wy[j], wx[i])
            out_s += weight * ss
            out_r += weight * rr
            eff = A * Ps
            otf_eff[(j, i)] = eff.get() if backend == "gpu" else eff
            del FA, FB, ss, rr, weight, A, B, Ps, eff
    to_np = (lambda a: a.get()) if backend == "gpu" else (lambda a: a)
    sci_w, ref_w = to_np(out_s), to_np(out_r)
    sci_w[nan] = np.nan
    ref_w[nan] = np.nan
    if backend == "gpu":
        del Sh, Rh, out_s, out_r
        xp.get_default_memory_pool().free_all_blocks()
    return dict(sci=sci_w, ref=ref_w, otf_eff=otf_eff, lattice=L, xs=np.asarray(xs), ys=np.asarray(ys))


def _centred_crop(k, half):
    """A kernel on a periodic lattice (origin at [0, 0]) as a numpy array of (2 half + 1) x (2 half + 1) samples centred on the origin."""
    a = k.get() if hasattr(k, "get") else np.asarray(k)
    return np.roll(a, (half, half), axis=(0, 1))[:2 * half + 1, :2 * half + 1].copy()


def _pad_wrapped(k, shape, xp=np):
    """A kernel on an L x L periodic lattice (origin at [0, 0]) placed on a larger periodic grid."""
    Ly, Lx = k.shape
    if shape[0] < Ly or shape[1] < Lx:
        raise ValueError(f"the padded grid {shape} is smaller than the filter lattice {k.shape}")
    hy, hx = Ly // 2, Lx // 2
    out = xp.zeros(shape, dtype=xp.float64)
    out[:hy, :hx] = k[:hy, :hx]
    out[:hy, -hx:] = k[:hy, -hx:]
    out[-hy:, :hx] = k[-hy:, :hx]
    out[-hy:, -hx:] = k[-hy:, -hx:]
    return out
