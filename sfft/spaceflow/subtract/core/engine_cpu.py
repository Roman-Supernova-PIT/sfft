"""SFFT's CPU engine: the least-squares system assembled by streaming over image planes (peak memory a few planes, never a stack of
every term), solved, and the difference built from a few small transforms; scipy's FFT for the transforms, numba for the inner loops. Also
the solution container both engines return."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .spec import SFFTSpec, PolynomialBasis
from .basis import build_basis_tables, basis_design_matrix
from . import _ops_cpu as K

logger = logging.getLogger('sfft.core')


@dataclass
class SFFTSolution:
    spec: SFFTSpec
    a_ijab: np.ndarray
    b_pq: np.ndarray
    tvar: dict = None

    @property
    def full(self) -> np.ndarray:
        return np.concatenate([self.a_ijab, self.b_pq])


def _path_laplacian(n: int) -> np.ndarray:
    """1D path-graph Laplacian (degree on diagonal, -1 on adjacency)."""
    A = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    return np.diag(A.sum(axis=1)) - A


def _ireg_matrix(L0: int, L1: int, ignore_center: bool) -> np.ndarray:
    """Laplacian quadratic form in the delta-difference coefficient basis.

    Kernel pixel values relate to coefficients a as k = T a with
    T[:, c] = e_c - e_c0 (c != c0) and T[:, c0] = e_c0; iR is the
    symmetrized T^T (Lap^T Lap) T (legacy Fill_iREGMAT, factor 2 included).
    """
    Fab = L0 * L1
    c0 = (L0 // 2) * L1 + (L1 // 2)
    LAP = (np.kron(_path_laplacian(L0), np.eye(L1))
           + np.kron(np.eye(L0), _path_laplacian(L1)))
    if ignore_center:
        LAP[[c0 - L1, c0 - 1, c0, c0 + 1, c0 + L1], :] = 0.0
    M = LAP.T @ LAP
    u = M[c0, :] + M[:, c0]
    mcc = M[c0, c0]
    iR = M + M.T - u[:, None] - u[None, :] + 2.0 * mcc
    iR[:, c0] = u - 2.0 * mcc
    iR[c0, :] = u - 2.0 * mcc
    iR[c0, c0] = 2.0 * mcc
    assert Fab == iR.shape[0]
    return iR


class RegularizationPieces:
    """Per-spec precomputation; apply() adds lam * REG to the kernel block."""

    def __init__(self, spec):
        reg = spec.regularization
        N0, N1 = spec.shape
        L0, L1 = spec.kernel.L0, spec.kernel.L1
        xy = np.asarray(reg.xy, dtype=np.float64)
        cx, cy = xy[:, 0] / N0, xy[:, 1] / N1
        if reg.weights is None:
            W = np.eye(len(xy)) / len(xy)
        else:
            w = np.asarray(reg.weights, dtype=np.float64)
            W = np.diag(w / w.sum())

        Fij = spec.Fij
        SP = basis_design_matrix(spec.kernel.basis, cx, cy, spec.shape)
        self.SST = SP @ W @ SP.T
        self.CSST = None
        self.DSST = None
        if spec.scaling == 'SEPARATE-VARYING':
            SSP = basis_design_matrix(spec.scaling_basis, cx, cy, spec.shape)
            Fsca = spec.Fsca
            self.CSST = np.zeros((Fij, Fij))
            self.CSST[:, :Fsca] = SP @ W @ SSP.T
            self.DSST = np.zeros((Fij, Fij))
            self.DSST[:Fsca, :Fsca] = SSP @ W @ SSP.T

        iR = _ireg_matrix(L0, L1, reg.ignore_center)
        self.c0 = (L0 // 2) * L1 + (L1 // 2)
        if spec.scaling != 'SEPARATE-VARYING':
            self.iR = iR
        else:
            c0 = self.c0
            self.iR_nn = iR.copy()
            self.iR_nn[c0, :] = 0.0
            self.iR_nn[:, c0] = 0.0
            self.iR_nc = np.zeros_like(iR)
            self.iR_nc[:, c0] = iR[:, c0]
            self.iR_nc[c0, c0] = 0.0
            self.iR_cn = np.zeros_like(iR)
            self.iR_cn[c0, :] = iR[c0, :]
            self.iR_cn[c0, c0] = 0.0
            self.iR_cc = np.zeros_like(iR)
            self.iR_cc[c0, c0] = iR[c0, c0]

        self.lam = reg.lam
        self.scale2 = 1.0 / float(N0 * N1) ** 2
        self.separate_varying = (spec.scaling == 'SEPARATE-VARYING')
        self.Fij, self.Fab = Fij, spec.Fab

    def apply(self, LHMAT) -> None:
        """LHMAT[:Fijab, :Fijab] += lam * REG, block by block."""
        Fij, Fab = self.Fij, self.Fab
        f = self.lam * self.scale2
        for k in range(Fij):
            for k8 in range(Fij):
                blk = LHMAT[k * Fab:(k + 1) * Fab, k8 * Fab:(k8 + 1) * Fab]
                if not self.separate_varying:
                    blk += (f * self.SST[k, k8]) * self.iR
                else:
                    blk += ((f * self.SST[k, k8]) * self.iR_nn
                            + (f * self.CSST[k, k8]) * self.iR_nc
                            + (f * self.CSST[k8, k]) * self.iR_cn
                            + (f * self.DSST[k, k8]) * self.iR_cc)


class SFFTEngine:
    """CPU (numpy + numba + scipy pocketfft) engine for one grid specification."""

    def __init__(self, spec: SFFTSpec, threads: int = 8):
        self.spec = spec
        self.threads = int(threads)
        N0, N1 = spec.shape

        import numba as nb
        if callable(getattr(nb, 'set_num_threads', None)):
            nb.set_num_threads(self.threads)
        else:
            logger.warning('numba.set_num_threads was clobbered upstream; '
                           'numba thread count left at its default')

        self.kU, self.kV, self.k_pairs = build_basis_tables(spec.kernel.basis, N0, N1)
        self.bU, self.bV, self.b_pairs = build_basis_tables(spec.background.basis, N0, N1)
        self.sU = self.sV = self.s_pairs = None
        if spec.scaling == 'SEPARATE-VARYING':
            self.sU, self.sV, self.s_pairs = build_basis_tables(spec.scaling_basis, N0, N1)

        self._reg = None
        if spec.regularization is not None:
            self._reg = RegularizationPieces(spec)

        hw = spec.kernel.hw
        self.ref_ab = np.array([(a - hw, b - hw)
                                for a in range(spec.kernel.L0)
                                for b in range(spec.kernel.L1)], dtype=np.int32)
        self.center = hw * spec.kernel.L1 + hw

        self.SCALE = 1.0 / (N0 * N1)
        self.SCALE_L = float(N0 * N1)

        self._init_fft(N0, N1)

        self._FI = None
        self._FT = None

    def _init_fft(self, N0, N1):
        from scipy import fft as sfft
        self.buf = np.empty((N0, N1), dtype=np.complex128)
        self._scipy_fft = sfft

    def _fft_inplace(self):
        out = self._scipy_fft.fft2(self.buf, workers=self.threads, overwrite_x=True)
        if out is not self.buf:
            self.buf[:] = out

    def _ifft_inplace(self):
        out = self._scipy_fft.ifft2(self.buf, workers=self.threads, overwrite_x=True)
        if out is not self.buf:
            self.buf[:] = out

    def _modulated_spectrum(self, img, U_i, V_j, out):
        """out = SCALE * FFT2( img * outer(U_i, V_j) ), via the shared buffer."""
        self.buf[:] = 0.0
        np.multiply(img, U_i[:, None], out=self.buf.real)
        self.buf.real *= V_j[None, :]
        self._fft_inplace()
        np.multiply(self.buf, self.SCALE, out=out)

    def _build_spectra(self, img_I, img_J):
        """FI (Fij), FT (Fpq), FJ, FS (Fsca | None) -- the only persistent stacks."""
        N0, N1 = self.spec.shape
        Fij, Fpq = self.spec.Fij, self.spec.Fpq
        FI = np.empty((Fij, N0, N1), dtype=np.complex128)
        FT = np.empty((Fpq, N0, N1), dtype=np.complex128)
        FJ = np.empty((N0, N1), dtype=np.complex128)

        for ij, (i, j) in enumerate(self.k_pairs):
            self._modulated_spectrum(img_I, self.kU[i], self.kV[j], FI[ij])
        for pq, (p, q) in enumerate(self.b_pairs):
            self.buf[:] = 0.0
            self.buf.real[:, :] = self.bU[p][:, None] * self.bV[q][None, :]
            self._fft_inplace()
            np.multiply(self.buf, self.SCALE, out=FT[pq])
        self.buf[:] = img_J
        self._fft_inplace()
        np.multiply(self.buf, self.SCALE, out=FJ)

        FS = None
        if self.s_pairs is not None:
            FS = np.empty((self.spec.Fsca, N0, N1), dtype=np.complex128)
            for k, (i, j) in enumerate(self.s_pairs):
                self._modulated_spectrum(img_I, self.sU[i], self.sV[j], FS[k])
        return FI, FT, FJ, FS

    def solve(self, mI, mJ, unc=None) -> SFFTSolution:
        """Least-squares kernel fit; optionally with sandwich uncertainties.

        unc: dict with
          'tvecs': {name: full-NEQ vector t} -- linear functionals t.a
          'V':     per-pixel variance of the TARGET side, already ZEROED
                   outside the fit mask (the fit only sees masked pixels)
          'acf':   optional stationary ACF stamp of the target side
        For each t: Var(t.a) = c^2 (A u)^T C_J (A u), u = LHMAT^-1 t --
        one extra constrained solve + one model image per functional; the
        LHMAT<->A^T A normalization c is self-calibrated from the solution
        (immune to spectrum-scaling conventions). Regularized LHMAT => the
        variance of the REGULARIZED estimator, as it should be. Results in
        solution.tvar. Noise in the convolved side is NOT propagated (EIV
        regime).
        """
        spec = self.spec
        self._check(mI); self._check(mJ)
        t0 = time.time()
        mJ = np.ascontiguousarray(mJ, np.float64)
        FI, FT, FJ, FS = self._build_spectra(np.ascontiguousarray(mI, np.float64),
                                             mJ)
        logger.info('spectra staged in %.2fs', time.time() - t0)

        LHMAT, RHb = self._assemble_normal(FI, FT, FJ, FS)

        if self._reg is not None:
            self._reg.apply(LHMAT)

        solution = self._solve_constrained(LHMAT, RHb)
        tvar = None
        if unc:
            tvar = self._sandwich(FI, FT, FS, LHMAT, RHb, solution, mJ, unc)
        return SFFTSolution(spec=spec, a_ijab=solution[:spec.Fijab],
                            b_pq=solution[spec.Fijab:], tvar=tvar)

    def _sandwich(self, FI, FT, FS, LHMAT, RHb, solution, mJ, unc):
        """Var(t.a) for each requested functional (see solve docstring)."""
        names = list(unc['tvecs'])
        T = np.stack([np.asarray(unc['tvecs'][k], np.float64)
                      for k in names], axis=1)
        U = self._solve_constrained(LHMAT, T)
        V_J = np.asarray(unc['V'], np.float64)
        acf = unc.get('acf')
        m_sol = self._model_image(FI, FT, FS, solution)
        denom = float(np.sum(m_sol * mJ))
        c = float(solution @ RHb) / denom if denom != 0.0 else 1.0
        tvar = {}
        for j, name in enumerate(names):
            m_u = self._model_image(FI, FT, FS,
                                    np.ascontiguousarray(U[:, j]))
            if acf is None:
                q = float(np.sum(V_J * m_u * m_u))
            else:
                kk = (acf.shape[0] - 1) // 2
                g = np.zeros((m_u.shape[0] + 2 * kk, m_u.shape[1] + 2 * kk))
                g[kk:kk + m_u.shape[0], kk:kk + m_u.shape[1]] = \
                    m_u * np.sqrt(V_J)
                G = np.fft.rfft2(g)
                ac = np.fft.irfft2(G * np.conj(G), s=g.shape)
                q = 0.0
                for dy in range(-kk, kk + 1):
                    for dx in range(-kk, kk + 1):
                        q += acf[dy + kk, dx + kk] * ac[dy % g.shape[0],
                                                        dx % g.shape[1]]
            tvar[name] = c * c * q
        return tvar

    def _assemble_normal(self, FI, FT, FJ, FS):
        """Stream the normal equations (LHMAT, RHb) from the staged spectra."""
        spec = self.spec
        N0, N1 = spec.shape
        Fij, Fpq, Fab, Fijab, NEQ = (spec.Fij, spec.Fpq, spec.Fab,
                                     spec.Fijab, spec.NEQ)
        LHMAT = np.empty((NEQ, NEQ), dtype=np.float64)
        RHb = np.empty(NEQ, dtype=np.float64)
        S2 = self.SCALE * self.SCALE
        pre = self.buf.real
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        Fsca = spec.Fsca
        c0 = self.center

        t1 = time.time()
        for i8j8 in range(Fij):
            for ij in range(Fij):
                K.mul_conj(FI[i8j8], FI[ij], self.buf)
                self._fft_inplace()
                if sepvar:
                    K.fill_omg11_slice(i8j8, ij, pre, S2, self.ref_ab, Fab,
                                       LHMAT, N0, N1)
                else:
                    K.fill_omg_slice(i8j8, ij, pre, S2, self.ref_ab, Fab,
                                     LHMAT, N0, N1)
        if sepvar:
            for i8j8 in range(Fij):
                for k in range(Fsca):
                    K.mul_conj(FI[i8j8], FS[k], self.buf)
                    self._fft_inplace()
                    K.fill_omg10_col(i8j8, k * Fab + c0, pre, S2, self.ref_ab,
                                     Fab, LHMAT, N0, N1)
            for k in range(Fsca):
                for ij in range(Fij):
                    K.mul_conj(FS[k], FI[ij], self.buf)
                    self._fft_inplace()
                    K.fill_omg01_row(k * Fab + c0, ij, pre, S2, self.ref_ab,
                                     Fab, LHMAT, N0, N1)
            for k8 in range(Fsca):
                for k in range(Fsca):
                    LHMAT[k8 * Fab + c0, k * Fab + c0] = \
                        S2 * K.sum_mul_conj_real(FS[k8], FS[k])
        for i8j8 in range(Fij):
            for pq in range(Fpq):
                K.mul_conj(FI[i8j8], FT[pq], self.buf)
                self._fft_inplace()
                K.fill_gam_slice(i8j8, pq, pre, self.SCALE, self.ref_ab,
                                 Fab, Fijab, LHMAT, N0, N1, sepvar)
        for p8q8 in range(Fpq):
            for ij in range(Fij):
                K.mul_conj(FT[p8q8], FI[ij], self.buf)
                self._fft_inplace()
                K.fill_psi_slice(p8q8, ij, pre, self.SCALE, self.ref_ab,
                                 Fab, Fijab, LHMAT, N0, N1, sepvar)
        if sepvar:
            for k in range(Fsca):
                for pq in range(Fpq):
                    LHMAT[k * Fab + c0, Fijab + pq] = \
                        self.SCALE * K.sum_mul_conj_real(FS[k], FT[pq])
            for p8q8 in range(Fpq):
                for k in range(Fsca):
                    LHMAT[Fijab + p8q8, k * Fab + c0] = \
                        self.SCALE * K.sum_mul_conj_real(FT[p8q8], FS[k])
        for p8q8 in range(Fpq):
            for pq in range(Fpq):
                LHMAT[Fijab + p8q8, Fijab + pq] = K.sum_mul_conj_real(FT[p8q8], FT[pq])
        for i8j8 in range(Fij):
            K.mul_conj(FI[i8j8], FJ, self.buf)
            self._fft_inplace()
            K.fill_the_slice(i8j8, pre, self.SCALE, self.ref_ab, Fab, RHb,
                             N0, N1, sepvar)
        if sepvar:
            for k in range(Fsca):
                RHb[k * Fab + c0] = self.SCALE * K.sum_mul_conj_real(FS[k], FJ)
        for p8q8 in range(Fpq):
            RHb[Fijab + p8q8] = K.sum_mul_conj_real(FT[p8q8], FJ)
        logger.info('linear system established in %.2fs', time.time() - t1)
        return LHMAT, RHb

    def _solve_constrained(self, LHMAT, RHb):
        spec = self.spec
        NEQ, Fij, Fab = spec.NEQ, spec.Fij, spec.Fab
        if spec.scaling == 'ENTANGLED':
            return np.linalg.solve(LHMAT, RHb)

        centers = np.arange(Fij) * Fab + self.center
        if spec.scaling == 'SEPARATE-VARYING':
            drop = centers[spec.Fsca:]
            if drop.size == 0:
                return np.linalg.solve(LHMAT, RHb)
            keep = np.setdiff1d(np.arange(NEQ), drop)
            x = np.linalg.solve(LHMAT[np.ix_(keep, keep)], RHb[keep])
            full = np.zeros(RHb.shape)
            full[keep] = x
            return full

        if isinstance(spec.kernel.basis, PolynomialBasis):
            drop = centers[1:]
            keep = np.setdiff1d(np.arange(NEQ), drop)
            x = np.linalg.solve(LHMAT[np.ix_(keep, keep)], RHb[keep])
            full = np.zeros(RHb.shape)
            full[keep] = x
            return full
        keep = np.setdiff1d(np.arange(NEQ), centers)
        NEQr = keep.size + 1
        R = np.zeros((NEQ, NEQr))
        R[keep, np.arange(keep.size)] = 1.0
        R[centers, keep.size] = 1.0
        Lr = R.T @ LHMAT @ R
        br = R.T @ RHb
        x = np.linalg.solve(Lr, br)
        return R @ x

    def subtract(self, I, J, solution: SFFTSolution, qhat=None) -> np.ndarray:
        spec = self.spec
        N0, N1 = spec.shape
        self._check(I); self._check(J)
        Fab = spec.Fab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        t0 = time.time()
        FI, FT, FJ, FS = self._build_spectra(np.ascontiguousarray(I, np.float64),
                                             np.ascontiguousarray(J, np.float64))

        rows = np.mod(self.ref_ab[:, 0], N0)
        cols = np.mod(self.ref_ab[:, 1], N1)
        PVAL = np.zeros((N0, N1), dtype=np.complex128)
        for ij in range(spec.Fij):
            coeffs = solution.a_ijab[ij * Fab: (ij + 1) * Fab]
            self.buf[:] = 0.0
            self.buf[rows, cols] = coeffs
            if sepvar:
                self.buf[rows[self.center], cols[self.center]] = 0.0
            self._fft_inplace()
            S_ij = coeffs.sum() - coeffs[self.center]
            K.fdiff_mac(PVAL, FI[ij], self.buf, complex(S_ij), self.SCALE)
        if sepvar:
            for k in range(spec.Fsca):
                a_c = solution.a_ijab[k * Fab + self.center]
                K.bg_mac(PVAL, FS[k], complex(a_c * self.SCALE))
        for pq in range(spec.Fpq):
            K.bg_mac(PVAL, FT[pq], complex(solution.b_pq[pq]))

        np.subtract(FJ, PVAL, out=self.buf)
        if qhat is not None:
            self.buf *= qhat
        self._ifft_inplace()
        diff = self.buf.real * self.SCALE_L
        logger.info('difference constructed in %.2fs', time.time() - t0)
        return np.ascontiguousarray(diff)

    def run(self, I, J, mI, mJ):
        solution = self.solve(mI, mJ)
        return solution, self.subtract(I, J, solution)


    def _model_image(self, FI, FT, FS, vec):
        """A @ vec: synthesize the model image from a full-NEQ coefficient vector."""
        spec = self.spec
        N0, N1 = spec.shape
        Fab = spec.Fab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        a = vec[:spec.Fijab]
        b = vec[spec.Fijab:]
        rows = np.mod(self.ref_ab[:, 0], N0)
        cols = np.mod(self.ref_ab[:, 1], N1)
        PVAL = np.zeros((N0, N1), dtype=np.complex128)
        for ij in range(spec.Fij):
            coeffs = a[ij * Fab: (ij + 1) * Fab]
            self.buf[:] = 0.0
            self.buf[rows, cols] = coeffs
            if sepvar:
                self.buf[rows[self.center], cols[self.center]] = 0.0
            self._fft_inplace()
            S_ij = coeffs.sum() - coeffs[self.center]
            K.fdiff_mac(PVAL, FI[ij], self.buf, complex(S_ij), self.SCALE)
        if sepvar:
            for k in range(spec.Fsca):
                K.bg_mac(PVAL, FS[k], complex(a[k * Fab + self.center] * self.SCALE))
        for pq in range(spec.Fpq):
            K.bg_mac(PVAL, FT[pq], complex(b[pq]))
        self.buf[:] = PVAL
        self._ifft_inplace()
        return self.buf.real * self.SCALE_L

    def _project(self, FI, FT, FS, X):
        """A.T @ X in the engine's Fourier-normalized inner product.

        NOTE the normalization: this equals (1/N) times the real-domain
        adjoint (Parseval with the SCALE = 1/N spectrum convention). The
        constant is shared by LHMAT and every matvec, so solutions are
        unaffected; only cross-domain comparisons must multiply by SCALE_L.
        """
        spec = self.spec
        N0, N1 = spec.shape
        Fij, Fpq, Fab = spec.Fij, spec.Fpq, spec.Fab
        Fijab = spec.Fijab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        self.buf[:] = X
        self._fft_inplace()
        FX = self.buf * self.SCALE
        g = np.empty(spec.NEQ, dtype=np.float64)
        pre = self.buf.real
        for i8j8 in range(Fij):
            K.mul_conj(FI[i8j8], FX, self.buf)
            self._fft_inplace()
            K.fill_the_slice(i8j8, pre, self.SCALE, self.ref_ab, Fab, g,
                             N0, N1, sepvar)
        if sepvar:
            for k in range(spec.Fsca):
                g[k * Fab + self.center] = self.SCALE * K.sum_mul_conj_real(FS[k], FX)
        for p8q8 in range(Fpq):
            g[Fijab + p8q8] = K.sum_mul_conj_real(FT[p8q8], FX)
        return g

    def _reduction(self):
        """(T, Tt): full <-> reduced coordinate maps for the scaling constraints."""
        spec = self.spec
        NEQ, Fij, Fab = spec.NEQ, spec.Fij, spec.Fab
        if spec.scaling == 'ENTANGLED':
            return (lambda x: x), (lambda g: g), None
        centers = np.arange(Fij) * Fab + self.center
        if spec.scaling == 'SEPARATE-VARYING':
            drop = centers[spec.Fsca:]
            if drop.size == 0:
                return (lambda x: x), (lambda g: g), None
            keep = np.setdiff1d(np.arange(NEQ), drop)
        elif isinstance(spec.kernel.basis, PolynomialBasis):
            keep = np.setdiff1d(np.arange(NEQ), centers[1:])
        else:
            keep_np = np.setdiff1d(np.arange(NEQ), centers)
            R = np.zeros((NEQ, keep_np.size + 1))
            R[keep_np, np.arange(keep_np.size)] = 1.0
            R[centers, keep_np.size] = 1.0
            return (lambda x: R @ x), (lambda g: R.T @ g), R

        def T(x):
            full = np.zeros(NEQ)
            full[keep] = x
            return full
        return T, (lambda g: g[keep]), keep

    def solve_weighted(self, I, J, w, tol=1e-10, maxit=100) -> SFFTSolution:
        """Exactly weighted SFFT solve: min_c sum_x w(x) |J - (Ac)(x)|^2.

        w: per-pixel weights (0 = pixel ignored, else e.g. 1/sigma^2); may be
        continuous (a tapered mask IS a weight map). I must be defect-free
        (inpaint bad reference pixels first).

        With spec.regularization set, minimizes the penalized objective
        sum w|J - Ac|^2 + lam * <Laplacian penalty>, i.e. solves
        (A^T W A + R) c = A^T W J with EXACTLY the R that solve() adds to
        LHMAT (same RegularizationPieces, same normalization convention) --
        so w == 1 reproduces the classic regularized solution, and the
        preconditioner LU(LHMAT + R) stays exact in that limit.
        """
        spec = self.spec
        self._check(I); self._check(J)
        w = np.ascontiguousarray(w, np.float64)
        if w.shape != spec.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError('w must match the image shape, be finite and >= 0')

        t0 = time.time()
        FI, FT, FJ, FS = self._build_spectra(np.ascontiguousarray(I, np.float64),
                                             np.ascontiguousarray(J, np.float64))
        LHMAT, _ = self._assemble_normal(FI, FT, FJ, FS)
        Rfull = None
        if self._reg is not None:
            Rfull = np.zeros((spec.NEQ, spec.NEQ))
            self._reg.apply(Rfull)
            LHMAT += Rfull
        T, Tt, TR = self._reduction()
        from scipy.linalg import lu_factor, lu_solve

        def _reduce(Mfull):
            if TR is None:
                return Mfull
            if isinstance(TR, np.ndarray) and TR.ndim == 2:
                return TR.T @ Mfull @ TR
            return Mfull[np.ix_(TR, TR)]

        M_lu = lu_factor(_reduce(LHMAT))
        Rr = _reduce(Rfull) if Rfull is not None else None
        logger.info('weighted solve: spectra + preconditioner in %.2fs',
                    time.time() - t0)

        t1 = time.time()
        if Rr is None:
            matvec = lambda x: Tt(self._project(
                FI, FT, FS, w * self._model_image(FI, FT, FS, T(x))))
        else:
            matvec = lambda x: Tt(self._project(
                FI, FT, FS, w * self._model_image(FI, FT, FS, T(x)))) + Rr @ x
        b = Tt(self._project(FI, FT, FS, w * np.ascontiguousarray(J, np.float64)))
        x = np.zeros_like(b)
        r = b - matvec(x)
        z = lu_solve(M_lu, r)
        p = z.copy()
        rz = r @ z
        nb = np.sqrt(b @ b)
        n_iter = 0
        for n_iter in range(1, maxit + 1):
            Ap = matvec(p)
            al = rz / (p @ Ap)
            x += al * p
            r -= al * Ap
            if np.sqrt(r @ r) / nb < tol:
                break
            z = lu_solve(M_lu, r)
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        full = T(x)
        logger.info('weighted solve: CG converged in %d iterations (%.2fs)',
                    n_iter, time.time() - t1)
        return SFFTSolution(spec=spec, a_ijab=full[:spec.Fijab],
                            b_pq=full[spec.Fijab:])

    def _check(self, img):
        if img.shape != self.spec.shape:
            raise ValueError(f'image shape {img.shape} != spec shape {self.spec.shape}')
        if not np.isfinite(img).all():
            raise ValueError('input image contains NaN/Inf -- mask or fill first')
