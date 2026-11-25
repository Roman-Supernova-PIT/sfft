"""The transient's PSF in the difference, and the matched-filter detection image in units of the local noise."""
import numpy as np
from scipy import fft

from . import crossconv as cc


def effective_psf(otf_eff, lattice, half=20):
    """Real-space stamp ((2 half + 1) pixels a side, centred) of the transient's PSF in the matched science image, from its OTF on the
    filter lattice, for a transient of unit total flux: the stamp's sum is the share of the light inside it."""
    p = fft.irfft2(otf_eff, (lattice, lattice))
    return np.roll(p, (half, half), axis=(0, 1))[:2 * half + 1, :2 * half + 1]


def _filtered(img, kernel, workers):
    H = kernel.shape[0] // 2
    out = fft.irfft2(fft.rfft2(img, workers=workers) * fft.rfft2(kernel[::-1, ::-1], img.shape, workers=workers), img.shape, workers=workers)
    return np.roll(out, (-H, -H), axis=(0, 1))


def score_image(d, psf, free, variance=None, workers=8):
    """The matched-filter detection image of a difference d in units of the local noise: d correlated with psf, divided by the noise the
    variance map predicts through the filter relative to the sky's (never below 1), then by its robust scatter on the source-free pixels free
    (a selection at 3 sigma keeps the scatter unbiased). NaN where d is NaN. Returns (score, its noise in the filtered image's units)."""
    good = np.isfinite(d)
    sc = _filtered(np.where(good, d, 0.0), psf, workers)
    ratio = 1.0
    if variance is not None:
        v = np.where(good & np.isfinite(variance), variance, 0.0)
        sky = np.median(v[free & good])
        if sky > 0:
            ratio = np.sqrt(np.maximum(_filtered(v, psf * psf / np.sum(psf * psf), workers) / sky, 1.0))
    sc = sc / ratio
    sig = cc.robust_sigma(sc[free & good])
    sc = sc / sig
    sc[~good] = np.nan
    return sc, float(sig)
