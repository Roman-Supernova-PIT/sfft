"""Image-stage products of SFFT's sparse selection: SExtractor catalogue, segmentation, and the point-source locus and FWHM."""
from __future__ import annotations

import os
import logging
import hashlib
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import gaussian_kde

from .calib import Calib
from ..astromatic.sextractor import SExtractor, SExConfig, DEFAULT_COLUMNS
from ..stats import top_flatten_weighted_quantile as _tfwq

logger = logging.getLogger('sfft.sparse')

_SCHEMA = 2


@dataclass(frozen=True)
class LocusInfo:
    """The point-source locus: the belt estimator's output + diagnostics."""
    peak: float
    sigma: float
    belt: float
    prominence: float
    fwhm: float
    n_members: int


@dataclass
class ImageProducts:
    """Everything the pair stage needs from one image."""
    catalog: Table
    seg: np.ndarray
    locus: LocusInfo
    star_mask: np.ndarray
    nan_mask: np.ndarray
    sat_mask: np.ndarray
    shape: tuple
    calib: Calib
    cache_key: str
    from_cache: bool = False


def locus_mode(catalog, fwhm_prior=None, snr_min=20.0, fr_max=10.0,
               belt_cap=0.3, ellip_max=0.3):
    """The belt estimator: flux-weighted KDE mode of FLUX_RADIUS.

    Two passes: pass 1 locates the peak with a loose absolute floor; pass 2
    applies the PSF-SCALED sub-PSF guard FR > 0.35 x FWHM (absolute floors
    are fooled by bright artifact belts -- measured). The belt half-width is capped (an uncapped adaptive
    width inflates exactly in bad seeing -- measured). Returns
    (LocusInfo, member_mask).
    """
    FR = np.asarray(catalog['FLUX_RADIUS'], float)
    FLUX = np.asarray(catalog['FLUX_AUTO'], float)
    FLUXERR = np.asarray(catalog['FLUXERR_AUTO'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        SNR = np.where(FLUXERR > 0, FLUX / FLUXERR, 0.0)
    A = np.asarray(catalog['A_IMAGE'], float)
    B = np.asarray(catalog['B_IMAGE'], float)
    ELLIP = (A - B) / (A + B)
    n = len(FR)

    def _peak(fr_min):
        base = (FR > fr_min) & (FR < fr_max) & np.isfinite(FR) & (FLUX > 0)
        hi = base & (SNR > snr_min)
        if hi.sum() < 10:
            hi = base & (SNR > 5.0)
        if hi.sum() < 5:
            return None
        w = np.log10(np.clip(FLUX[hi], 1.0, None))
        kde = gaussian_kde(FR[hi], weights=w, bw_method=0.15)
        grid = np.linspace(fr_min, min(fr_max, np.percentile(FR[hi], 99)), 400)
        dens = kde(grid)
        ipk = int(np.argmax(dens))
        half = dens[ipk] / 2.0
        il = ipk
        while il > 0 and dens[il] > half:
            il -= 1
        ir = ipk
        while ir < len(grid) - 1 and dens[ir] > half:
            ir += 1
        sigma = max((grid[ir] - grid[il]) / 2.355, 0.03)
        prom = float(dens[ipk] / max(np.median(dens), 1e-12))
        return base, grid[ipk], sigma, prom

    res = _peak(0.5)
    if res is None:
        return LocusInfo(np.nan, np.nan, np.nan, 0.0, np.nan, 0), \
            np.zeros(n, bool)
    _, peak1, _, _ = res
    fwhm_guess = fwhm_prior if fwhm_prior else 2.0 * peak1
    res = _peak(max(0.35 * fwhm_guess, 0.5))
    if res is None:
        return LocusInfo(np.nan, np.nan, np.nan, 0.0, np.nan, 0), \
            np.zeros(n, bool)
    base, peak, sigma, prom = res

    belt = float(np.clip(2.0 * sigma, 0.10, belt_cap))
    members = base & (np.abs(FR - peak) < belt) & (ELLIP < ellip_max)
    if members.sum() >= 3:
        fwhm = float(_tfwq(
            np.asarray(catalog['FWHM_IMAGE'], float)[members],
            [0.5], weights=FLUX[members], num_top=30)[0])
    else:
        fwhm = np.nan
    return LocusInfo(peak=float(peak), sigma=float(sigma), belt=belt,
                     prominence=prom, fwhm=fwhm,
                     n_members=int(members.sum())), members


def array_identity(image):
    """Content-hash identity for an in-memory image: stable across runs and
    temp-file materializations (~0.1s at 8.4Mpx float64). Pass the result as
    image_products(identity=...) so array inputs get real cache hits instead
    of a fresh mtime-based key per run."""
    a = np.ascontiguousarray(image)
    return ('sha1', hashlib.sha1(a.tobytes()).hexdigest())


def _cache_key(path, ext, config: SExConfig, columns, identity=None):
    if identity is None:
        st = os.stat(path)
        identity = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    blob = repr((identity, ext, config.content_hash(), tuple(columns),
                 _SCHEMA)).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _baked_config(config, calib: Calib):
    """The SExtractor config actually run: calibration values folded in."""
    config = config or SExConfig()
    return SExConfig(**{**config.__dict__,
                        'gain': calib.gain, 'satur_level': calib.satur_raw})


def products_key(path, calib: Calib, ext=0, config: SExConfig = None,
                 columns=DEFAULT_COLUMNS, identity=None):
    """The exact cache key image_products() would use for these arguments.
    Exposed so a batch layer can dedup CONCURRENT identical computes (many
    tasks sharing one reference) before the disk cache is even consulted."""
    return _cache_key(path, ext, _baked_config(config, calib), columns,
                      identity=identity)


def _cache_paths(cache_dir, path, key):
    stem = os.path.splitext(os.path.basename(path))[0]
    base = os.path.join(cache_dir, f'{stem}.{key}')
    return base + '.npz', base + '.cat.fits'


def prune_cache(cache_dir, keep_latest_per_stem=False, dry_run=False):
    """Housekeeping for a sidecar cache directory. Removes: incomplete
    entries (an .npz without its .cat.fits or vice versa) and leftover
    .tmp files from crashed writers (older than 1h). With
    keep_latest_per_stem=True, additionally keeps only the newest key per
    image stem -- ONLY safe when image basenames are unique in the cache
    (two different images sharing a basename would evict each other; array
    inputs share generic stems, so leave it False for those).
    Returns the list of removed filenames.
    """
    import re
    import time as _time
    pat = re.compile(r'^(?P<stem>.+)\.(?P<key>[0-9a-f]{12})'
                     r'\.(?P<kind>npz|cat\.fits)$')
    entries = {}
    removed = []
    for fn in sorted(os.listdir(cache_dir)):
        fp = os.path.join(cache_dir, fn)
        if fn.endswith('.tmp') or fn.endswith('.tmp.npz'):
            if _time.time() - os.path.getmtime(fp) > 3600:
                removed.append(fn)
                if not dry_run:
                    os.remove(fp)
            continue
        m = pat.match(fn)
        if m:
            entries.setdefault((m['stem'], m['key']), set()).add(m['kind'])
    newest = {}
    if keep_latest_per_stem:
        for (stem, key), kinds in entries.items():
            if kinds == {'npz', 'cat.fits'}:
                mt = os.path.getmtime(
                    os.path.join(cache_dir, f'{stem}.{key}.npz'))
                if stem not in newest or mt > newest[stem][0]:
                    newest[stem] = (mt, key)
    for (stem, key), kinds in entries.items():
        complete = kinds == {'npz', 'cat.fits'}
        superseded = keep_latest_per_stem and \
            (stem in newest and key != newest[stem][1])
        if complete and not superseded:
            continue
        for kind in kinds:
            fn = f'{stem}.{key}.{kind}'
            removed.append(fn)
            if not dry_run:
                os.remove(os.path.join(cache_dir, fn))
    if removed:
        logger.info('prune_cache: removed %d files from %s',
                    len(removed), cache_dir)
    return removed


def image_products(path, calib: Calib, ext=0,
                   config: SExConfig = None, columns=DEFAULT_COLUMNS,
                   fwhm_prior=None, cache_dir=None, cache_compress=False,
                   sextractor: SExtractor = None,
                   image=None, identity=None) -> ImageProducts:
    """Compute (or load from cache) the image-stage products for one image.

    cache_dir: None (default) disables the sidecar cache entirely; a
    directory enables read+write of persistent per-image sidecars.
    cache_compress: False (default) writes fast (~0.05s, ~48MB/image);
    True writes small (~0.45s, ~0.2MB/image -- masks are ~all zeros) for
    persistent campaign-scale caches. Reads accept either transparently.
    image: optional preloaded pixel array in (x, y) orientation matching
    `path` -- skips rereading the FITS for the NaN/sat masks.
    identity: cache identity override; default is (abspath, mtime, size) of
    `path`. Pass array_identity(image) for in-memory inputs materialized to
    temp files, so their cache key is content-based and survives reruns.
    """
    config = _baked_config(config, calib)
    key = _cache_key(path, ext, config, columns, identity=identity)

    if cache_dir:
        npz_p, cat_p = _cache_paths(cache_dir, path, key)
        if os.path.exists(npz_p) and os.path.exists(cat_p):
            z = np.load(npz_p)
            locus = LocusInfo(*[t(z['locus'][i]) for i, t in enumerate(
                (float, float, float, float, float, int))])
            logger.info('image_products cache HIT: %s', os.path.basename(npz_p))
            return ImageProducts(
                catalog=Table.read(cat_p), seg=z['seg'],
                locus=locus, star_mask=z['star_mask'],
                nan_mask=z['nan_mask'], sat_mask=z['sat_mask'],
                shape=tuple(z['seg'].shape), calib=calib, cache_key=key,
                from_cache=True)

    sex = sextractor or SExtractor()
    res = sex.run(path=path, ext=ext, config=config, columns=columns,
                  checkimages=('SEGMENTATION',))
    seg = np.ascontiguousarray(res.checkimages['SEGMENTATION'].T).astype(np.int32)

    if image is not None:
        img = np.asarray(image, np.float64)
    else:
        with fits.open(path) as hdul:
            data = hdul[ext].data.astype(np.float64)
        img = np.ascontiguousarray(data.T)
    nan_mask = ~np.isfinite(img)
    with np.errstate(invalid='ignore'):
        sat_mask = img >= calib.satur

    locus, star_mask = locus_mode(res.catalog, fwhm_prior=fwhm_prior)
    logger.info('image_products: %s -> %d sources, locus %.2f px '
                '(prom %.1f), FWHM %.2f px', os.path.basename(path),
                len(res.catalog), locus.peak, locus.prominence, locus.fwhm)

    prods = ImageProducts(catalog=res.catalog, seg=seg, locus=locus,
                          star_mask=star_mask, nan_mask=nan_mask,
                          sat_mask=sat_mask, shape=img.shape, calib=calib,
                          cache_key=key)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        npz_p, cat_p = _cache_paths(cache_dir, path, key)
        tmp = f'{cat_p}.{os.getpid()}.tmp'
        res.catalog.write(tmp, format='fits', overwrite=True)
        os.replace(tmp, cat_p)
        _savez = np.savez_compressed if cache_compress else np.savez
        tmp = f'{npz_p}.{os.getpid()}.tmp.npz'
        _savez(tmp, seg=seg, star_mask=star_mask, nan_mask=nan_mask,
               sat_mask=sat_mask,
               locus=np.array([locus.peak, locus.sigma, locus.belt,
                               locus.prominence, locus.fwhm, locus.n_members]))
        os.replace(tmp, npz_p)
        logger.info('image_products cache WRITE: %s', os.path.basename(npz_p))
    return prods
