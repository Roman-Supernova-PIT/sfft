"""The variance of the difference, propagated from the two input variance maps through the matching filters and SFFT's kernel."""
import numpy as np
from scipy import fft, signal

from . import crossconv as cc


def propagated_variance(v_sci, v_ref, xs, ys, filter_kernels=None, sfft_kernels=None, workers=8):
    """The variance map of the difference ([y, x]):

        V = sum_n w_n (V_sci * a_n^2 + V_ref * (K_n * b_n)^2),

    the science and reference variance maps propagated through each PSF node's matching filters, a_n on the science side and b_n on the
    reference side, and through SFFT's kernel K_n at the node on the reference side; the nodes' results are blended with the bilinear node
    weights w_n the filters are blended with (crossconv._tri_weights). Pixels are taken as independent, so a pixel-to-pixel correlation of an
    input (a resampled image's) is not included.

    v_sci, v_ref: [y, x], NaN where a pixel has no value; such pixels are filled from their neighbours before the convolutions, as the images
    are, and are NaN after. xs, ys: the node centres. filter_kernels: {(j, i): (a, b)}, centred [y, x] kernels (None: no matching, a = b =
    delta). sfft_kernels: {(j, i): K}, centred [y, x] (None: no SFFT kernel)."""
    ny, nx = v_sci.shape
    nan = ~np.isfinite(v_sci) | ~np.isfinite(v_ref)
    vs = cc.inpaint_holes(np.where(nan, np.nan, v_sci))
    vr = cc.inpaint_holes(np.where(nan, np.nan, v_ref))
    wx = cc._tri_weights(np.arange(nx, dtype=np.float64), np.asarray(xs, np.float64))
    wy = cc._tri_weights(np.arange(ny, dtype=np.float64), np.asarray(ys, np.float64))
    out = np.zeros((ny, nx))
    delta = np.ones((1, 1))
    with fft.set_workers(max(1, int(workers))):
        for j in range(len(ys)):
            for i in range(len(xs)):
                kx, ky = np.nonzero(wx[i] > 0)[0], np.nonzero(wy[j] > 0)[0]
                if not len(kx) or not len(ky):
                    continue
                a, b = filter_kernels[(j, i)] if filter_kernels is not None else (delta, delta)
                if sfft_kernels is not None:
                    b = signal.fftconvolve(b, sfft_kernels[(j, i)], mode="full")
                h = max(a.shape[0], a.shape[1], b.shape[0], b.shape[1]) // 2 + 1
                y0, y1, x0, x1 = int(ky[0]), int(ky[-1]) + 1, int(kx[0]), int(kx[-1]) + 1
                Y0, Y1, X0, X1 = max(0, y0 - h), min(ny, y1 + h), max(0, x0 - h), min(nx, x1 + h)
                part = (signal.fftconvolve(vs[Y0:Y1, X0:X1], a * a, mode="same") + signal.fftconvolve(vr[Y0:Y1, X0:X1], b * b, mode="same"))
                out[y0:y1, x0:x1] += np.outer(wy[j][y0:y1], wx[i][x0:x1]) * part[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]
    out[nan] = np.nan
    return out
