"""Spatial bases as data: one-dimensional tables and design matrices of the polynomial and B-spline bases."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import BSpline

from .spec import PolynomialBasis, BSplineBasis


def _coord(N: int) -> np.ndarray:
    return (1.0 + np.arange(N)) / N


def _bspline_eval(N: int, int_knots, degree: int, coord: np.ndarray) -> np.ndarray:
    """Elementary B-spline basis functions at given scaled coordinates.

    Knot vector matches BSplineSFFT.Create_BSplineBasis exactly:
    [0.5]*(k+1) + int_knots + [N+0.5]*(k+1), all divided by N.
    """
    knot = np.concatenate(([0.5] * (degree + 1), list(int_knots),
                           [N + 0.5] * (degree + 1))) / N
    n_c = len(int_knots) + degree + 1
    table = np.empty((n_c, coord.size), dtype=np.float64)
    for idx in range(n_c):
        coeff = (np.arange(n_c) == idx).astype(float)
        table[idx] = BSpline(t=knot, c=coeff, k=degree, extrapolate=False)(coord)
    if not np.isfinite(table).all():
        raise ValueError('B-spline basis evaluated to non-finite values -- '
                         'coordinates/internal knots must lie inside [0.5, N+0.5]')
    return table


def _bspline_table(N: int, int_knots, degree: int) -> np.ndarray:
    return _bspline_eval(N, int_knots, degree, _coord(N))


def build_basis_tables(basis, N0: int, N1: int):
    """Return (U, V, pairs) for a PolynomialBasis or BSplineBasis.

    U: (nU, N0) float64; V: (nV, N1) float64; pairs: (F, 2) int32 in the exact
    legacy enumeration order (triangular for polynomial, row-major full tensor
    for B-spline).
    """
    if isinstance(basis, PolynomialBasis):
        D = basis.degree
        pairs = np.array([(i, j) for i in range(D + 1) for j in range(D + 1 - i)],
                         dtype=np.int32)
        U = np.array([_coord(N0) ** i for i in range(D + 1)])
        V = np.array([_coord(N1) ** j for j in range(D + 1)])
        return U, V, pairs

    if isinstance(basis, BSplineBasis):
        U = _bspline_table(N0, basis.knots_x, basis.degree)
        V = _bspline_table(N1, basis.knots_y, basis.degree)
        pairs = np.array([(i, j) for i in range(U.shape[0]) for j in range(V.shape[0])],
                         dtype=np.int32)
        return U, V, pairs

    raise TypeError(f'unsupported basis type: {type(basis).__name__}')


def basis_design_matrix(basis, cx: np.ndarray, cy: np.ndarray,
                        shape: tuple) -> np.ndarray:
    """(n_terms, n_points) matrix of basis terms at scaled coordinates.

    cx, cy follow the legacy regularization convention (x/N0, y/N1). The
    term enumeration matches build_basis_tables.
    """
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    if isinstance(basis, PolynomialBasis):
        D = basis.degree
        return np.array([cx ** i * cy ** j
                         for i in range(D + 1) for j in range(D + 1 - i)])
    if isinstance(basis, BSplineBasis):
        U = _bspline_eval(shape[0], basis.knots_x, basis.degree, cx)
        V = _bspline_eval(shape[1], basis.knots_y, basis.degree, cy)
        return np.array([U[i] * V[j]
                         for i in range(U.shape[0]) for j in range(V.shape[0])])
    raise TypeError(f'unsupported basis type: {type(basis).__name__}')
