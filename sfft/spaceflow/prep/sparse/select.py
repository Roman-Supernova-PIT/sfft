"""Pair stage of SFFT's sparse selection: catalogue match, magnitude-offset surface, variable rejection, blend groups, tier ladder."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger('sfft.sparse')


@dataclass
class MatchedPair:
    ir: np.ndarray
    js: np.ndarray
    tol: float
    dx_med: float
    dy_med: float
    d_rms: float


def match_catalogs(cat_r, cat_s, fwhm_r, fwhm_s, tol=None, tol_factor=3.0):
    """Mutual-nearest pixel-coordinate match (images assumed aligned)."""
    from scipy.spatial import cKDTree
    xr = np.column_stack([np.asarray(cat_r['X_IMAGE'], float),
                          np.asarray(cat_r['Y_IMAGE'], float)])
    xs = np.column_stack([np.asarray(cat_s['X_IMAGE'], float),
                          np.asarray(cat_s['Y_IMAGE'], float)])
    if tol is None:
        tol = float(np.sqrt((fwhm_r / tol_factor) ** 2 +
                            (fwhm_s / tol_factor) ** 2))
    tr, ts = cKDTree(xr), cKDTree(xs)
    d_rs, j_rs = ts.query(xr, k=1, distance_upper_bound=tol)
    d_sr, j_sr = tr.query(xs, k=1, distance_upper_bound=tol)
    ir = np.where(np.isfinite(d_rs))[0]
    ir = ir[[j_sr[j_rs[i]] == i if np.isfinite(d_sr[j_rs[i]]) else False
             for i in ir]]
    js = j_rs[ir]
    delta = xs[js] - xr[ir]
    return MatchedPair(
        ir=ir, js=js, tol=tol,
        dx_med=float(np.median(delta[:, 0])) if len(ir) else np.nan,
        dy_med=float(np.median(delta[:, 1])) if len(ir) else np.nan,
        d_rms=float(np.sqrt(np.mean(np.sum(delta ** 2, 1)))) if len(ir) else np.nan)


@dataclass
class OffsetSurface:
    """dmag(x, y) = MAG_SCI - MAG_REF as a low-order polynomial surface."""
    coeffs: np.ndarray
    degree: int
    shape: tuple
    scalar: float
    mad: float

    def __call__(self, x, y):
        cx = np.asarray(x, float) / self.shape[0]
        cy = np.asarray(y, float) / self.shape[1]
        terms = [cx ** i * cy ** j
                 for i in range(self.degree + 1)
                 for j in range(self.degree + 1 - i)]
        return np.sum([c * t for c, t in zip(self.coeffs, terms)], axis=0)


def fit_offset_surface(x, y, dmag, shape, degree=1, clip_sigma=3.0,
                       coarse_thresh=0.5, n_pass=2):
    """Robust polynomial fit of dmag(x, y); two clipping passes."""
    x, y, dmag = (np.asarray(a, float) for a in (x, y, dmag))
    scalar = float(np.median(dmag))
    keep = np.abs(dmag - scalar) < coarse_thresh
    cx, cy = x / shape[0], y / shape[1]
    T = np.column_stack([cx ** i * cy ** j
                         for i in range(degree + 1)
                         for j in range(degree + 1 - i)])
    coeffs = None
    for _ in range(n_pass):
        if keep.sum() < T.shape[1] + 2:
            break
        coeffs, *_ = np.linalg.lstsq(T[keep], dmag[keep], rcond=None)
        resid = dmag - T @ coeffs
        mad = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        keep = np.abs(resid) < clip_sigma * max(mad, 1e-4)
    if coeffs is None:
        coeffs = np.zeros(T.shape[1])
        coeffs[0] = scalar
        mad = 1.4826 * np.median(np.abs(dmag - scalar))
    surf = OffsetSurface(coeffs=coeffs, degree=degree, shape=tuple(shape),
                         scalar=scalar, mad=float(mad))
    return surf


def reject_variables(cat_r, cat_s, m: MatchedPair, surf: OffsetSurface,
                     k_sigma=5.0, err_floor=0.01, snr_min=10.0,
                     f_ref=1.0, f_sci=1.0):
    """Upgraded EVREJ: surface-referenced, self-calibrating variance.

    Replaces the legacy EVREJ_SAFE_MAGDEV escape hatch: the propagated
    sigma is inflated by the measured width of the normalized-residual
    distribution before clipping, and a fractional error floor absorbs
    bright-end systematics. Returns (keep_mask over matches, qa dict).

    f_ref/f_sci: aperture error-inflation F-factors for CORRELATED images
    (noise.f_factor of the stationary-ACF stamp; 1 = independent noise).
    They scale the propagated FLUXERR^2 -- with them supplied, the measured
    sigma_inflation should return to ~1 (the free cross-check of 2c).
    """
    F_r = np.asarray(cat_r['FLUX_AUTO'], float)[m.ir]
    F_s = np.asarray(cat_s['FLUX_AUTO'], float)[m.js]
    E_r = np.asarray(cat_r['FLUXERR_AUTO'], float)[m.ir]
    E_s = np.asarray(cat_s['FLUXERR_AUTO'], float)[m.js]
    x = np.asarray(cat_s['X_IMAGE'], float)[m.js]
    y = np.asarray(cat_s['Y_IMAGE'], float)[m.js]

    s = 10.0 ** (surf(x, y) / -2.5)
    dF = F_s - s * F_r
    sig2 = f_ref * (s * E_r) ** 2 + f_sci * E_s ** 2 \
        + (err_floor * np.maximum(s * F_r, F_s)) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        z = dF / np.sqrt(sig2)

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_ok = (F_r / np.where(E_r > 0, E_r, np.inf) > snr_min) & \
                 (F_s / np.where(E_s > 0, E_s, np.inf) > snr_min)
    zz = z[snr_ok & np.isfinite(z)]
    inflation = 1.0
    if zz.size > 20:
        width = 1.4826 * np.median(np.abs(zz - np.median(zz)))
        inflation = max(1.0, float(width))
    keep = snr_ok & np.isfinite(z) & (np.abs(z) < k_sigma * inflation)
    qa = dict(n_snr_ok=int(snr_ok.sum()), n_rejected=int((snr_ok & ~keep).sum()),
              sigma_inflation=round(inflation, 3),
              f_ref=round(float(f_ref), 4), f_sci=round(float(f_sci), 4))
    logger.info('variable rejection: %d/%d rejected (sigma inflation %.2f)',
                qa['n_rejected'], qa['n_snr_ok'], inflation)
    return keep, qa


def _island_groups(seg, labels):
    """Union-find over touching segmentation labels (deblend families)."""
    parent = {int(l): int(l) for l in labels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for sl in ((slice(None, -1), slice(None)), (slice(None), slice(None, -1))):
        a = seg[sl]
        b = seg[1:, :] if sl[0] != slice(None) else seg[:, 1:]
        m = (a > 0) & (b > 0) & (a != b)
        for pa, pb in set(zip(a[m].ravel().tolist(), b[m].ravel().tolist())):
            if pa in parent and pb in parent:
                union(pa, pb)
    return {l: find(int(l)) for l in labels}


def admit_blend_groups(cat_r, cat_s, seg_r, seg_s, m: MatchedPair,
                       surf: OffsetSurface, keep, k_sigma=5.0, err_floor=0.01):
    """Group-level admission of blended (FLAGS 1|2) matched sources.

    Uses FLUX_ISO sums over deblend families (ISO fluxes partition the island
    exactly, so the sum is deblending-independent). A consistent group admits
    all its matched members; an inconsistent group rejects them all.
    Returns updated keep mask + qa.
    """
    if 'FLUX_ISO' not in cat_r.colnames:
        return keep, dict(group_tested=0, note='FLUX_ISO not in catalog')
    FLAGS_r = np.asarray(cat_r['FLAGS'], int)[m.ir]
    FLAGS_s = np.asarray(cat_s['FLAGS'], int)[m.js]
    blended = ((FLAGS_r & 3) > 0) | ((FLAGS_s & 3) > 0)
    if not blended.any():
        return keep, dict(group_tested=0)

    lab_s = np.asarray(cat_s['SEGLABEL'] if 'SEGLABEL' in cat_s.colnames
                       else np.arange(1, len(cat_s) + 1))[m.js]
    lab_s = np.asarray(m.js) + 1
    groups = _island_groups(seg_s, np.unique(lab_s[blended]))
    gid = np.array([groups.get(int(l), int(l)) for l in lab_s])

    F_r = np.asarray(cat_r['FLUX_ISO'], float)[m.ir]
    F_s = np.asarray(cat_s['FLUX_ISO'], float)[m.js]
    E_r = np.asarray(cat_r['FLUXERR_ISO'], float)[m.ir]
    E_s = np.asarray(cat_s['FLUXERR_ISO'], float)[m.js]
    x = np.asarray(cat_s['X_IMAGE'], float)[m.js]
    y = np.asarray(cat_s['Y_IMAGE'], float)[m.js]
    s = 10.0 ** (surf(x, y) / -2.5)

    keep = keep.copy()
    n_admit = n_reject = 0
    for g in np.unique(gid[blended]):
        sel = gid == g
        Gs, Gr = F_s[sel].sum(), (s[sel] * F_r[sel]).sum()
        sig = np.sqrt((s[sel] * E_r[sel]).sum() ** 2 / max(sel.sum(), 1)
                      + (E_s[sel] ** 2).sum()
                      + (err_floor * max(Gs, Gr)) ** 2)
        if abs(Gs - Gr) < k_sigma * sig:
            keep[sel] = True
            n_admit += int(sel.sum())
        else:
            keep[sel] = False
            n_reject += int(sel.sum())
    qa = dict(group_tested=int(blended.sum()), group_admitted=n_admit,
              group_rejected=n_reject)
    logger.info('blend groups: %d members tested, %d admitted, %d rejected',
                qa['group_tested'], n_admit, n_reject)
    return keep, qa


@dataclass
class Selection:
    tier: int
    keep: np.ndarray
    match: MatchedPair
    surface: OffsetSurface
    qa: dict = field(default_factory=dict)


def select_subsources(prod_r, prod_s, min_members=25,
                      prominence_gate=4.5, offset_mad_gate=0.3,
                      f_ref=1.0, f_sci=1.0,
                      **rej_kwargs) -> Selection:
    """The tier ladder with the measured gates.

    T1: belt members on both sides + variable rejection (strict, purity-first)
    T2: all ratio-consistent matched sources incl. blend groups (coverage)
    The caller may combine (T1 stars always in; T2 adds galaxies/blends) --
    the returned keep IS that union when both tiers are healthy.
    Raises SelectionError with diagnostics if even T2 starves.
    """
    cat_r, cat_s = prod_r.catalog, prod_s.catalog
    m = match_catalogs(cat_r, cat_s, prod_r.locus.fwhm, prod_s.locus.fwhm)
    if len(m.ir) < 5:
        raise SelectionError(f'only {len(m.ir)} matched sources '
                             f'(astrometry? tol={m.tol:.2f}px)', qa=vars(m))

    MAG_r = np.asarray(cat_r['MAG_AUTO'], float)[m.ir]
    MAG_s = np.asarray(cat_s['MAG_AUTO'], float)[m.js]
    x = np.asarray(cat_s['X_IMAGE'], float)[m.js]
    y = np.asarray(cat_s['Y_IMAGE'], float)[m.js]
    surf = fit_offset_surface(x, y, MAG_s - MAG_r, prod_s.shape)

    keep, ev_qa = reject_variables(cat_r, cat_s, m, surf,
                                   f_ref=f_ref, f_sci=f_sci, **rej_kwargs)
    keep, grp_qa = admit_blend_groups(cat_r, cat_s, prod_r.seg, prod_s.seg,
                                      m, surf, keep, **rej_kwargs)

    stars = prod_r.star_mask[m.ir] & prod_s.star_mask[m.js]
    t1 = keep & stars

    loci_ok = (prod_r.locus.prominence >= prominence_gate and
               prod_s.locus.prominence >= prominence_gate)
    gates = dict(prom_r=round(prod_r.locus.prominence, 1),
                 prom_s=round(prod_s.locus.prominence, 1),
                 offset_mad=round(surf.mad, 4),
                 n_match=len(m.ir), n_t1=int(t1.sum()), n_keep=int(keep.sum()),
                 astrom=(round(m.dx_med, 3), round(m.dy_med, 3)),
                 **ev_qa, **grp_qa)

    if surf.mad > offset_mad_gate:
        raise SelectionError(
            f'photometric offset scatter {surf.mad:.3f} mag exceeds the gate '
            f'{offset_mad_gate} -- broken pair (bad image / wrong match)?',
            qa=gates)

    if loci_ok and t1.sum() >= min_members:
        tier = 1
        final = keep
    elif keep.sum() >= min_members:
        tier = 2
        final = keep
    else:
        raise SelectionError(
            f'selection starved: {int(keep.sum())} consistent matches '
            f'(< {min_members}); escalate to signal-cut mask (tier 3)',
            qa=gates)
    gates['tier'] = tier
    logger.info('selection tier %d: %d SubSources (%d stars), '
                'offset %.3f+-%.3f mag', tier, int(final.sum()),
                int(t1.sum()), surf.scalar, surf.mad)
    return Selection(tier=tier, keep=final, match=m, surface=surf, qa=gates)


class SelectionError(RuntimeError):
    def __init__(self, message, qa=None):
        self.qa = qa or {}
        super().__init__(message + '\n  QA: %s' % (qa,))
