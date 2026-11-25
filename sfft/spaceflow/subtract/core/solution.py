"""Readers of an SFFT solution: the matching kernel, flux scaling and background at a field position, a grid of kernels; saving and loading a solution."""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from .spec import (SFFTSpec, KernelSpec, BackgroundSpec, PolynomialBasis,
                   t_flux_scaling)
from .engine_cpu import SFFTSolution

__all__ = ['save_solution', 'load_solution', 'kernel_at', 'flux_scaling_at',
           'background_at', 'kernel_grid']


def _poly_pairs(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def _resolve_frac(spec, x=None, y=None, frac=None):
    if frac is not None:
        return float(frac[0]), float(frac[1])
    if x is None or y is None:
        return 0.5, 0.5
    return (float(x) + 1.0) / spec.shape[0], (float(y) + 1.0) / spec.shape[1]


def _require_poly(spec, what):
    if not isinstance(spec.kernel.basis, PolynomialBasis) or \
            not isinstance(spec.background.basis, PolynomialBasis):
        raise NotImplementedError(f'{what}: polynomial bases only '
                                  '(B-spline serialization: next round)')


def save_solution(solution: SFFTSolution, path, extra_header=None):
    """Persist a solution WITH its full spec (self-describing FITS)."""
    spec = solution.spec
    _require_poly(spec, 'save_solution')
    ph = fits.PrimaryHDU()
    h = ph.header
    h['N0'] = (spec.shape[0], 'sfft: solve-grid axis 0 (padded)')
    h['N1'] = (spec.shape[1], 'sfft: solve-grid axis 1 (padded)')
    h['KERHW'] = (spec.kernel.hw, 'sfft: kernel half-width')
    h['KDEG'] = (spec.kernel.basis.degree, 'sfft: kernel spatial degree')
    h['BDEG'] = (spec.background.basis.degree, 'sfft: background degree')
    h['SCALING'] = (spec.scaling, 'sfft: scaling mode')
    h['SDEG'] = (spec.scaling_basis.degree
                 if spec.scaling_basis is not None else -1,
                 'sfft: scaling-basis degree (-1 = none)')
    h['NAIJAB'] = (len(solution.a_ijab), 'sfft: kernel dof')
    h['NBPQ'] = (len(solution.b_pq), 'sfft: background dof')
    for k, v in (extra_header or {}).items():
        h[k] = v
    ph.data = np.concatenate([np.asarray(solution.a_ijab, np.float64),
                              np.asarray(solution.b_pq, np.float64)]
                             ).reshape(1, -1)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fits.HDUList([ph]).writeto(path, overwrite=True)
    return path


def load_solution(path):
    """Load a solution saved by save_solution -> (SFFTSolution, meta dict)."""
    with fits.open(path) as hdul:
        h = hdul[0].header
        vec = np.asarray(hdul[0].data, np.float64).ravel()
    for key in ('N0', 'N1', 'KERHW', 'KDEG', 'BDEG', 'SCALING'):
        if key not in h:
            raise ValueError(
                f'{path}: missing {key} -- not a self-describing 2.x solution '
                'file (files written before 2026-07-19 lack the spec; re-run '
                'write_result to regenerate)')
    sdeg = int(h.get('SDEG', -1))
    spec = SFFTSpec(
        shape=(int(h['N0']), int(h['N1'])),
        kernel=KernelSpec(hw=int(h['KERHW']),
                          basis=PolynomialBasis(int(h['KDEG']))),
        background=BackgroundSpec(basis=PolynomialBasis(int(h['BDEG']))),
        scaling=str(h['SCALING']).strip(),
        scaling_basis=PolynomialBasis(sdeg) if sdeg >= 0 else None)
    na, nb = int(h['NAIJAB']), int(h['NBPQ'])
    if vec.size != na + nb:
        raise ValueError(f'{path}: data length {vec.size} != NAIJAB+NBPQ '
                         f'{na + nb}')
    meta = {k: h[k] for k in h if k not in ('SIMPLE', 'BITPIX', 'NAXIS',
                                            'NAXIS1', 'NAXIS2', 'EXTEND')}
    return SFFTSolution(spec=spec, a_ijab=vec[:na], b_pq=vec[na:]), meta


def kernel_at(solution: SFFTSolution, x=None, y=None, frac=None):
    """The matching kernel at a field position, physical normalization
    (sum(kernel) == the flux scaling there).

    Delta-difference basis: non-center coefficients c_ab enter as
    (delta_ab - delta_center), so K[ab != c] = c_ab(x, y) and
    K[center] = c_center(x, y) - sum_others. For SEPARATE-VARYING the
    center coefficient lives on the SCALING basis, handled separately.
    """
    spec = solution.spec
    _require_poly(spec, 'kernel_at')
    xf, yf = _resolve_frac(spec, x, y, frac)
    L0, L1, Fab = spec.kernel.L0, spec.kernel.L1, spec.Fab
    center = spec.kernel.hw * L1 + spec.kernel.hw
    N = float(spec.shape[0] * spec.shape[1])

    kp = _poly_pairs(spec.kernel.basis.degree)
    c = np.zeros(Fab)
    for k, (i, j) in enumerate(kp):
        c += solution.a_ijab[k * Fab:(k + 1) * Fab] * (xf ** i) * (yf ** j)
    if spec.scaling == 'SEPARATE-VARYING':
        sp_ = _poly_pairs(spec.scaling_basis.degree)
        c[center] = 0.0
        for k, (i, j) in enumerate(sp_):
            c[center] += solution.a_ijab[k * Fab + center] \
                * (xf ** i) * (yf ** j)
    K = c.copy()
    K[center] = c[center] - (c.sum() - c[center])
    return (K / N).reshape(L0, L1)


def flux_scaling_at(solution: SFFTSolution, x=None, y=None, frac=None):
    """The photometric flux scaling at a field position (= sum(kernel_at))."""
    spec = solution.spec
    xf, yf = _resolve_frac(spec, x, y, frac)
    return float(t_flux_scaling(spec, xf, yf) @ solution.full)


def background_at(solution: SFFTSolution, x=None, y=None, frac=None):
    """The differential-background model value at a field position."""
    spec = solution.spec
    _require_poly(spec, 'background_at')
    xf, yf = _resolve_frac(spec, x, y, frac)
    val = 0.0
    for k, (p, q) in enumerate(_poly_pairs(spec.background.basis.degree)):
        val += solution.b_pq[k] * (xf ** p) * (yf ** q)
    return float(val)


def kernel_grid(solution: SFFTSolution, nx=8, ny=8):
    """Kernels realized on an nx x ny grid of tile centers.

    Returns (fracs, kernels): fracs (nx*ny, 2) scaled positions, kernels
    (nx*ny, L0, L1) -- the 2.x equivalent of the legacy
    Realize_MatchingKernel over a query grid.
    """
    fx = (np.arange(nx) + 0.5) / nx
    fy = (np.arange(ny) + 0.5) / ny
    fracs, kernels = [], []
    for a in fx:
        for b in fy:
            fracs.append((a, b))
            kernels.append(kernel_at(solution, frac=(a, b)))
    return np.asarray(fracs), np.asarray(kernels)
