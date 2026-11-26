"""Frozen dataclasses that define an SFFT linear system: grid shape, kernel half-width and spatial basis, background basis, flux-scaling mode and optional kernel regularization."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolynomialBasis:
    """cx^i * cy^j with i + j <= degree (triangular enumeration)."""
    degree: int = 2

    def __post_init__(self):
        if not (0 <= self.degree <= 6):
            raise ValueError(f'polynomial degree {self.degree} outside [0, 6]')

    @property
    def n_terms(self) -> int:
        return ((self.degree + 1) * (self.degree + 2)) // 2


@dataclass(frozen=True)
class BSplineBasis:
    """Tensor product of 1D B-spline elementary bases (full Fi x Fj enumeration).

    Knots are in pixel units, matching the 2024-paper convention:
    boundary knots at 0.5 and N+0.5, coordinates cx = (1+arange(N))/N.
    """
    degree: int = 2
    knots_x: tuple = field(default_factory=tuple)
    knots_y: tuple = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, 'knots_x', tuple(float(k) for k in self.knots_x))
        object.__setattr__(self, 'knots_y', tuple(float(k) for k in self.knots_y))
        if not (0 <= self.degree <= 5):
            raise ValueError(f'B-spline degree {self.degree} outside [0, 5]')
        if self.degree == 0 and (self.knots_x or self.knots_y):
            raise ValueError('degree-0 B-spline with internal knots is discontinuous')

    def n_terms_1d(self, axis: int) -> int:
        knots = self.knots_x if axis == 0 else self.knots_y
        return len(knots) + self.degree + 1

    @property
    def n_terms(self) -> int:
        return self.n_terms_1d(0) * self.n_terms_1d(1)


@dataclass(frozen=True)
class KernelRegularization:
    """Tikhonov penalty on the matching-kernel Laplacian, sampled at chosen positions.

    xy holds (x, y) image positions in pixel units; the penalty is the
    (weighted) mean over them of |Laplacian(kernel)|^2, scaled by lam.
    ignore_center drops the Laplacian rows touching the kernel center pixel
    (the delta component carries the photometric scaling and must stay sharp).
    """
    lam: float = 1e-6
    xy: tuple = ()
    weights: tuple | None = None
    ignore_center: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'xy',
                           tuple((float(x), float(y)) for x, y in self.xy))
        if self.weights is not None:
            object.__setattr__(self, 'weights',
                               tuple(float(w) for w in self.weights))
            if len(self.weights) != len(self.xy):
                raise ValueError(f'{len(self.weights)} weights for '
                                 f'{len(self.xy)} regularization positions')
        if not self.xy:
            raise ValueError('kernel regularization needs at least one (x, y) position')
        if not (self.lam > 0):
            raise ValueError(f'regularization lam {self.lam} must be positive')


@dataclass(frozen=True)
class KernelSpec:
    """Matching-kernel structure: half-width and spatial variation."""
    hw: int = 8
    basis: PolynomialBasis | BSplineBasis = field(default_factory=PolynomialBasis)

    def __post_init__(self):
        if not (1 <= self.hw <= 64):
            raise ValueError(f'kernel half-width {self.hw} outside [1, 64]')

    @property
    def L0(self) -> int:
        return 2 * self.hw + 1

    @property
    def L1(self) -> int:
        return 2 * self.hw + 1

    @property
    def n_pix(self) -> int:
        return self.L0 * self.L1


@dataclass(frozen=True)
class BackgroundSpec:
    """Differential-background spatial variation."""
    basis: PolynomialBasis | BSplineBasis = field(default_factory=PolynomialBasis)


@dataclass(frozen=True)
class SFFTSpec:
    """Everything that determines the SFFT linear system for one grid.

    scaling:
      'ENTANGLED'         -- photometric scaling follows the kernel spatial law
                             (2022 core: ConstPhotRatio=False)
      'SEPARATE-CONSTANT' -- constant photometric scaling; kernel shape still varies
                             (2022 core: ConstPhotRatio=True)
      'SEPARATE-VARYING'  -- photometric scaling varies on its OWN spatial basis
                             (scaling_basis), decoupled from the kernel-shape law
                             (2024 B-spline flavor: SEPARATE_SCALING with DS > 0)
    """
    shape: tuple
    kernel: KernelSpec = field(default_factory=KernelSpec)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    scaling: str = 'SEPARATE-CONSTANT'
    scaling_basis: PolynomialBasis | BSplineBasis | None = None
    regularization: KernelRegularization | None = None

    def __post_init__(self):
        object.__setattr__(self, 'shape', (int(self.shape[0]), int(self.shape[1])))
        if self.scaling not in ('ENTANGLED', 'SEPARATE-CONSTANT', 'SEPARATE-VARYING'):
            raise ValueError(f"scaling '{self.scaling}' not in "
                             "('ENTANGLED', 'SEPARATE-CONSTANT', 'SEPARATE-VARYING')")
        if min(self.shape) < 4 * self.kernel.hw:
            raise ValueError(f'image {self.shape} too small for kernel hw={self.kernel.hw}')
        if self.scaling == 'SEPARATE-VARYING':
            sb, kb = self.scaling_basis, self.kernel.basis
            if sb is None:
                raise ValueError('SEPARATE-VARYING requires a scaling_basis')
            if sb.n_terms > kb.n_terms:
                raise ValueError(f'scaling dof {sb.n_terms} exceeds kernel dof '
                                 f'{kb.n_terms} (unsupported, mirrors legacy)')
            if sb == kb:
                raise ValueError('scaling_basis identical to kernel basis -- '
                                 "this reduces to scaling='ENTANGLED'; use that")
            if isinstance(sb, PolynomialBasis) and sb.degree == 0:
                raise ValueError("degree-0 scaling_basis is scaling='SEPARATE-CONSTANT'; "
                                 'use that')
        elif self.scaling_basis is not None:
            raise ValueError(f"scaling_basis is only used with scaling='SEPARATE-VARYING' "
                             f"(got scaling='{self.scaling}')")

    @property
    def Fij(self) -> int:
        return self.kernel.basis.n_terms

    @property
    def Fpq(self) -> int:
        return self.background.basis.n_terms

    @property
    def Fab(self) -> int:
        return self.kernel.n_pix

    @property
    def Fijab(self) -> int:
        return self.Fij * self.Fab

    @property
    def Fsca(self) -> int:
        """Scaling-field dof (only meaningful for SEPARATE-VARYING)."""
        return self.scaling_basis.n_terms if self.scaling_basis is not None else 0

    @property
    def NEQ(self) -> int:
        return self.Fijab + self.Fpq


def t_flux_scaling(spec: SFFTSpec, x_frac: float = 0.5, y_frac: float = 0.5):
    """The linear functional t with t.a = flux scaling at a field position.

    Mirrors the delta-difference normalization of realize_kernel: the kernel
    sum at (x, y) is c_center(x, y)/N, and c_center is linear in the center
    DOFs of the relevant spatial basis. Polynomial bases only. Use with
    SFFTEngine.solve(..., unc={'tvecs': {'fscal': t}, ...}) for the sandwich
    variance of the flux scaling.
    """
    import numpy as np
    kb = spec.kernel.basis
    if not isinstance(kb, PolynomialBasis):
        raise NotImplementedError('t_flux_scaling: polynomial kernel bases only')
    N = float(spec.shape[0] * spec.shape[1])
    center = spec.kernel.hw * spec.kernel.L1 + spec.kernel.hw
    t = np.zeros(spec.NEQ)
    if spec.scaling == 'SEPARATE-CONSTANT':
        t[center] = 1.0 / N
        return t
    if spec.scaling == 'SEPARATE-VARYING':
        sb = spec.scaling_basis
        pairs = [(i, j) for i in range(sb.degree + 1)
                 for j in range(sb.degree + 1 - i)]
        for k, (i, j) in enumerate(pairs):
            t[k * spec.Fab + center] = (x_frac ** i) * (y_frac ** j) / N
        return t
    pairs = [(i, j) for i in range(kb.degree + 1)
             for j in range(kb.degree + 1 - i)]
    for k, (i, j) in enumerate(pairs):
        t[k * spec.Fab + center] = (x_frac ** i) * (y_frac ** j) / N
    return t
