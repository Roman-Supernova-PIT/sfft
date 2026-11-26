"""Gain and saturation values for SFFT's sparse selection: resolved from given values or header keys, validated up front."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['CalibError', 'Calib', 'resolve_gain', 'resolve_satur',
           'resolve_satur_auto', 'AUTO_SATUR_KEYS']

AUTO_SATUR_KEYS = ('ESATUR', 'SATURATA', 'SATURATB', 'SATURATE')


class CalibError(ValueError):
    pass


@dataclass(frozen=True)
class Calib:
    """Resolved calibration for one image."""
    gain: float
    satur: float
    satur_raw: float
    gain_source: str
    satur_source: str


def _one(header, spec, what):
    """Resolve one spec ('key' or number) to a float."""
    if isinstance(spec, str):
        if header is None:
            raise CalibError(f'{what}: header keyword {spec!r} requested '
                             'but no header was given')
        if spec not in header:
            raise CalibError(f'{what}: keyword {spec!r} not in header -- '
                             'fix the keyword or pass a numeric value '
                             '(estimate externally if unknown)')
        val, src = header[spec], f'header:{spec}'
    else:
        val, src = spec, 'value'
    try:
        val = float(val)
    except (TypeError, ValueError):
        raise CalibError(f'{what}: {src} gave non-numeric value {val!r}')
    if not np.isfinite(val) or val <= 0:
        raise CalibError(f'{what}: {src} gave non-positive/non-finite {val!r}')
    return val, src


def _multi(header, spec, what):
    """spec may be a single key/number or a tuple of them (per-amplifier)."""
    specs = spec if isinstance(spec, (tuple, list)) else (spec,)
    vals, srcs = zip(*(_one(header, s, what) for s in specs))
    return min(vals), ('min(%s)' % ','.join(srcs)) if len(vals) > 1 else srcs[0]


def resolve_gain(header, gain):
    """gain: number | header-key | tuple of either (per-amp -> min).

    Deliberately NO 'AUTO' (2b asymmetry): gain only calibrates the
    source-Poisson term of FLUXERR (never the fit), and no data-driven
    estimate exists for sky-subtracted/coadded inputs. For raw sky-in
    frames, utils.estimators.estimate_gain is available as an explicit,
    user-called step; feed its VALUE here.
    """
    if isinstance(gain, str) and gain.upper() == 'AUTO':
        raise CalibError(
            "gain='AUTO' is not supported inside the packet (by design): "
            "estimate it explicitly with "
            "sfft.utils.estimators.estimate_gain on a SKY-IN frame (it "
            "refuses sky-subtracted inputs) and pass the value, or pass a "
            "header keyword. For coadds, prefer a variance map (weighted "
            "channel) over any scalar gain.")
    return _multi(header, gain, 'GAIN')


def resolve_satur(header, satur, margin=0.9):
    """satur: number | header-key | tuple (per-amp -> min); margin applied.

    Returns (adopted, raw, source): adopted = margin * min(values).
    """
    if isinstance(satur, str) and satur.upper() == 'AUTO':
        raise CalibError(
            "satur='AUTO' needs pixel data and is resolved by the packet "
            "layer (which calls utils.estimators.estimate_esatur and then "
            "resolve_satur_auto). Here, pass a numeric value or header key.")
    if not (0 < margin <= 1):
        raise CalibError(f'satur margin {margin} outside (0, 1]')
    raw, src = _multi(header, satur, 'SATUR')
    return margin * raw, raw, src


def resolve_satur_auto(header, estimate, margin=0.9, disagree_warn=0.05):
    """The 'AUTO' sourcing ladder as a PURE function.

    The caller runs the plateau estimator on the pixels and passes the result
    here (None/NaN = the estimator refused). Semantics:
      adopted = margin * min(header candidates present, estimate)
      estimator refused -> header candidates alone
      neither available -> CalibError
    Returns (adopted, raw, source, warning-or-None); the warning fires when
    header and estimate disagree by more than `disagree_warn` (a data-quality
    alarm per sec 2.2, not an error).
    """
    if not (0 < margin <= 1):
        raise CalibError(f'satur margin {margin} outside (0, 1]')
    cands = []
    if header is not None:
        for k in AUTO_SATUR_KEYS:
            if k in header:
                try:
                    v = float(header[k])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v) and v > 0:
                    cands.append((v, k))
    est_ok = estimate is not None and np.isfinite(estimate) and estimate > 0
    if not cands and not est_ok:
        raise CalibError(
            'SATUR=AUTO: no usable saturation header keyword '
            f'({"/".join(AUTO_SATUR_KEYS)}) and the plateau estimator '
            'refused (no clipped source found or unreliable ceiling) -- '
            'pass a numeric saturation level.')
    pool = list(cands) + ([(float(estimate), 'plateau')] if est_ok else [])
    raw, which = min(pool, key=lambda t: t[0])
    src = f'auto:{"header:" + which if which != "plateau" else "plateau"}(min)'
    warning = None
    if cands and est_ok:
        hdr_min = min(v for v, _ in cands)
        d = abs(hdr_min - float(estimate)) / min(hdr_min, float(estimate))
        if d > disagree_warn:
            warning = (f'header saturation ({hdr_min:.0f}) and plateau '
                       f'estimate ({float(estimate):.0f}) disagree by '
                       f'{100 * d:.1f}% -- check calibration/processing '
                       '(adopted the smaller).')
    return margin * raw, raw, src, warning


def resolve_calib(header, gain, satur, satur_margin=0.9) -> Calib:
    g, gsrc = resolve_gain(header, gain)
    s, sraw, ssrc = resolve_satur(header, satur, satur_margin)
    return Calib(gain=g, satur=s, satur_raw=sraw,
                 gain_source=gsrc, satur_source=ssrc)
