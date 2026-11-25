"""SFFT's GPU engine (CuPy): the twin of engine_cpu, the same system on the device."""
from __future__ import annotations

import logging
import time

import numpy as np
import cupy as cp

from .spec import SFFTSpec
from .basis import build_basis_tables
from .engine_cpu import SFFTSolution, RegularizationPieces

logger = logging.getLogger('sfft.core.gpu')


def _sync():
    cp.cuda.Stream.null.synchronize()


class CuPySFFTEngine:
    """GPU engine for one grid specification. numpy in / numpy out."""

    def __init__(self, spec: SFFTSpec, warmup: bool = True):
        t0 = time.time()
        self.spec = spec
        N0, N1 = spec.shape
        Fab = spec.Fab
        hw = spec.kernel.hw

        kU, kV, self.k_pairs = build_basis_tables(spec.kernel.basis, N0, N1)
        bU, bV, self.b_pairs = build_basis_tables(spec.background.basis, N0, N1)
        self.kU, self.kV = cp.asarray(kU), cp.asarray(kV)
        self.bU, self.bV = cp.asarray(bU), cp.asarray(bV)
        self.sU = self.sV = self.s_pairs = None
        if spec.scaling == 'SEPARATE-VARYING':
            sU, sV, self.s_pairs = build_basis_tables(spec.scaling_basis, N0, N1)
            self.sU, self.sV = cp.asarray(sU), cp.asarray(sV)

        ref_ab = np.array([(a - hw, b - hw)
                           for a in range(spec.kernel.L0)
                           for b in range(spec.kernel.L1)], dtype=np.int64)
        self.center = hw * spec.kernel.L1 + hw
        a, b = ref_ab[:, 0], ref_ab[:, 1]

        self.idxA = (cp.asarray(np.mod(a, N0)), cp.asarray(np.mod(b, N1)))
        self.idxB = (cp.asarray(np.mod(-a, N0)), cp.asarray(np.mod(-b, N1)))
        self.idxD = (cp.asarray(np.mod(a[:, None] - a[None, :], N0)),
                     cp.asarray(np.mod(b[:, None] - b[None, :], N1)))
        self.rows = cp.asarray(np.mod(a, N0))
        self.cols = cp.asarray(np.mod(b, N1))

        self.SCALE = 1.0 / (N0 * N1)
        self.SCALE_L = float(N0 * N1)

        self._reg = None
        if spec.regularization is not None:
            reg = RegularizationPieces(spec)
            for attr in ('SST', 'CSST', 'DSST', 'iR',
                         'iR_nn', 'iR_nc', 'iR_cn', 'iR_cc'):
                val = getattr(reg, attr, None)
                if val is not None:
                    setattr(reg, attr, cp.asarray(val))
            self._reg = reg

        if warmup:
            z = cp.zeros((N0, N1), dtype=np.complex128)
            cp.fft.fft2(z); cp.fft.ifft2(z)
            cp.fft.fft2(cp.zeros((N0, N1), dtype=np.float64))
            cp.linalg.solve(cp.eye(8), cp.ones(8))
            cp.vdot(z.ravel()[:16], z.ravel()[:16])
            del z
            _sync()
        logger.info('engine init (tables + GPU warmup) in %.2fs', time.time() - t0)

    def _build_spectra(self, img_I, img_J):
        spec = self.spec
        N0, N1 = spec.shape
        FI = cp.empty((spec.Fij, N0, N1), dtype=np.complex128)
        FT = cp.empty((spec.Fpq, N0, N1), dtype=np.complex128)
        for ij, (i, j) in enumerate(self.k_pairs):
            mod = img_I * self.kU[i][:, None]
            mod *= self.kV[j][None, :]
            FI[ij] = cp.fft.fft2(mod)
        FI *= self.SCALE
        for pq, (p, q) in enumerate(self.b_pairs):
            FT[pq] = cp.fft.fft2(self.bU[p][:, None] * self.bV[q][None, :])
        FT *= self.SCALE
        FJ = cp.fft.fft2(img_J)
        FJ *= self.SCALE
        FS = None
        if self.s_pairs is not None:
            FS = cp.empty((spec.Fsca, N0, N1), dtype=np.complex128)
            for k, (i, j) in enumerate(self.s_pairs):
                mod = img_I * self.sU[i][:, None]
                mod *= self.sV[j][None, :]
                FS[k] = cp.fft.fft2(mod)
            FS *= self.SCALE
        return FI, FT, FJ, FS

    def _pre(self, A, B):
        """Re[ FFT2( A * conj(B) )] -- the 'pre-Greek' lookup table."""
        return cp.fft.fft2(A * cp.conj(B)).real

    def _omg_block(self, pre):
        """(Fab, Fab) A-block for one basis pair, all four center cases."""
        pA = pre[self.idxA]
        pB = pre[self.idxB]
        p00 = pre[0, 0]
        blk = pre[self.idxD]
        blk -= pA[:, None]
        blk -= pB[None, :]
        blk += p00
        c = self.center
        blk[c, :] = pB - p00
        blk[:, c] = pA - p00
        blk[c, c] = p00
        return blk

    def _col_vals(self, pre):
        """Fab column for GAM-like fills: pre(a8,b8) - p00, center = p00."""
        p00 = pre[0, 0]
        v = pre[self.idxA] - p00
        v[self.center] = p00
        return v

    def _row_vals(self, pre):
        """Fab row for PSI-like fills: pre(-a,-b) - p00, center = p00."""
        p00 = pre[0, 0]
        v = pre[self.idxB] - p00
        v[self.center] = p00
        return v

    def solve(self, mI, mJ, unc=None) -> SFFTSolution:
        """GPU twin of engine_cpu.solve -- see that docstring for `unc`."""
        spec = self.spec
        self._check(mI); self._check(mJ)

        t0 = time.time()
        mJ_d = cp.asarray(mJ, dtype=np.float64)
        FI, FT, FJ, FS = self._build_spectra(
            cp.asarray(mI, dtype=np.float64), mJ_d)
        _sync()
        logger.info('spectra staged in %.2fs', time.time() - t0)

        LHMAT, RHb = self._assemble_normal(FI, FT, FJ, FS)

        if self._reg is not None:
            self._reg.apply(LHMAT)

        t2 = time.time()
        solution_d = self._solve_constrained(LHMAT, RHb)
        solution = cp.asnumpy(solution_d)
        _sync()
        logger.info('linear system solved in %.2fs', time.time() - t2)
        tvar = None
        if unc:
            tvar = self._sandwich(FI, FT, FS, LHMAT, RHb, solution_d,
                                  mJ_d, unc)
        return SFFTSolution(spec=spec, a_ijab=solution[:spec.Fijab],
                            b_pq=solution[spec.Fijab:], tvar=tvar)

    def _sandwich(self, FI, FT, FS, LHMAT, RHb, solution_d, mJ_d, unc):
        """GPU twin of engine_cpu._sandwich."""
        names = list(unc['tvecs'])
        T = cp.stack([cp.asarray(unc['tvecs'][k], np.float64)
                      for k in names], axis=1)
        U = self._solve_constrained(LHMAT, T)
        V_J = cp.asarray(unc['V'], np.float64)
        acf = unc.get('acf')
        m_sol = self._model_image(FI, FT, FS, solution_d)
        denom = float(cp.sum(m_sol * mJ_d))
        c = float(solution_d @ RHb) / denom if denom != 0.0 else 1.0
        tvar = {}
        for j, name in enumerate(names):
            m_u = self._model_image(FI, FT, FS, cp.ascontiguousarray(U[:, j]))
            if acf is None:
                q = float(cp.sum(V_J * m_u * m_u))
            else:
                kk = (acf.shape[0] - 1) // 2
                g = cp.zeros((m_u.shape[0] + 2 * kk, m_u.shape[1] + 2 * kk))
                g[kk:kk + m_u.shape[0], kk:kk + m_u.shape[1]] = \
                    m_u * cp.sqrt(V_J)
                G = cp.fft.rfft2(g)
                ac = cp.fft.irfft2(G * cp.conj(G), s=g.shape)
                dyg, dxg = np.mgrid[-kk:kk + 1, -kk:kk + 1]
                vals = ac[cp.asarray(dyg.ravel() % g.shape[0]),
                          cp.asarray(dxg.ravel() % g.shape[1])]
                q = float(cp.sum(cp.asarray(
                    np.asarray(acf, np.float64).ravel()) * vals))
            tvar[name] = c * c * q
        return tvar

    def _assemble_normal(self, FI, FT, FJ, FS):
        """Stream the normal equations (LHMAT, RHb) from the staged spectra."""
        spec = self.spec
        Fij, Fpq, Fab = spec.Fij, spec.Fpq, spec.Fab
        Fijab, NEQ, Fsca = spec.Fijab, spec.NEQ, spec.Fsca
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        c0 = self.center
        S2 = self.SCALE * self.SCALE

        t1 = time.time()
        LHMAT = cp.empty((NEQ, NEQ), dtype=np.float64)
        RHb = cp.empty(NEQ, dtype=np.float64)

        for i8j8 in range(Fij):
            for ij in range(Fij):
                blk = self._omg_block(self._pre(FI[i8j8], FI[ij]))
                blk *= S2
                LHMAT[i8j8 * Fab:(i8j8 + 1) * Fab, ij * Fab:(ij + 1) * Fab] = blk
        if sepvar:
            for i8j8 in range(Fij):
                for k in range(Fsca):
                    pre = self._pre(FI[i8j8], FS[k])
                    v = S2 * (pre[self.idxA] - pre[0, 0])
                    LHMAT[i8j8 * Fab:(i8j8 + 1) * Fab, k * Fab + c0] = v
            for k in range(Fsca):
                for ij in range(Fij):
                    pre = self._pre(FS[k], FI[ij])
                    v = S2 * (pre[self.idxB] - pre[0, 0])
                    LHMAT[k * Fab + c0, ij * Fab:(ij + 1) * Fab] = v
            for k8 in range(Fsca):
                for k in range(Fsca):
                    LHMAT[k8 * Fab + c0, k * Fab + c0] = \
                        S2 * cp.vdot(FS[k], FS[k8]).real
        for i8j8 in range(Fij):
            for pq in range(Fpq):
                v = self.SCALE * self._col_vals(self._pre(FI[i8j8], FT[pq]))
                LHMAT[i8j8 * Fab:(i8j8 + 1) * Fab, Fijab + pq] = v
        for p8q8 in range(Fpq):
            for ij in range(Fij):
                v = self.SCALE * self._row_vals(self._pre(FT[p8q8], FI[ij]))
                LHMAT[Fijab + p8q8, ij * Fab:(ij + 1) * Fab] = v
        if sepvar:
            for k in range(Fsca):
                for pq in range(Fpq):
                    LHMAT[k * Fab + c0, Fijab + pq] = \
                        self.SCALE * cp.vdot(FT[pq], FS[k]).real
            for p8q8 in range(Fpq):
                for k in range(Fsca):
                    LHMAT[Fijab + p8q8, k * Fab + c0] = \
                        self.SCALE * cp.vdot(FS[k], FT[p8q8]).real
        for p8q8 in range(Fpq):
            for pq in range(Fpq):
                LHMAT[Fijab + p8q8, Fijab + pq] = cp.vdot(FT[pq], FT[p8q8]).real
        for i8j8 in range(Fij):
            v = self.SCALE * self._col_vals(self._pre(FI[i8j8], FJ))
            RHb[i8j8 * Fab:(i8j8 + 1) * Fab] = v
        if sepvar:
            for k in range(Fsca):
                RHb[k * Fab + c0] = self.SCALE * cp.vdot(FJ, FS[k]).real
        for p8q8 in range(Fpq):
            RHb[Fijab + p8q8] = cp.vdot(FJ, FT[p8q8]).real
        _sync()
        logger.info('linear system established in %.2fs', time.time() - t1)
        return LHMAT, RHb

    def _solve_constrained(self, LHMAT, RHb):
        spec = self.spec
        NEQ, Fij, Fab = spec.NEQ, spec.Fij, spec.Fab
        if spec.scaling == 'ENTANGLED':
            return cp.linalg.solve(LHMAT, RHb)

        centers = np.arange(Fij) * Fab + self.center
        from .spec import PolynomialBasis
        if spec.scaling == 'SEPARATE-VARYING':
            drop = centers[spec.Fsca:]
            if drop.size == 0:
                return cp.linalg.solve(LHMAT, RHb)
            keep = cp.asarray(np.setdiff1d(np.arange(NEQ), drop))
            x = cp.linalg.solve(LHMAT[cp.ix_(keep, keep)], RHb[keep])
            full = cp.zeros(RHb.shape)
            full[keep] = x
            return full

        if isinstance(spec.kernel.basis, PolynomialBasis):
            keep = cp.asarray(np.setdiff1d(np.arange(NEQ), centers[1:]))
            x = cp.linalg.solve(LHMAT[cp.ix_(keep, keep)], RHb[keep])
            full = cp.zeros(RHb.shape)
            full[keep] = x
            return full
        keep_np = np.setdiff1d(np.arange(NEQ), centers)
        R = cp.zeros((NEQ, keep_np.size + 1))
        R[cp.asarray(keep_np), cp.arange(keep_np.size)] = 1.0
        R[cp.asarray(centers), keep_np.size] = 1.0
        x = cp.linalg.solve(R.T @ LHMAT @ R, R.T @ RHb)
        return R @ x


    def _model_image(self, FI, FT, FS, vec):
        """A @ vec: synthesize the model image (vec: cp array, full NEQ)."""
        spec = self.spec
        N0, N1 = spec.shape
        Fab = spec.Fab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        a = vec[:spec.Fijab]
        b = vec[spec.Fijab:]
        Kgrid = cp.zeros((N0, N1), dtype=np.complex128)
        PVAL = cp.zeros((N0, N1), dtype=np.complex128)
        for ij in range(spec.Fij):
            coeffs = a[ij * Fab:(ij + 1) * Fab]
            Kgrid[:] = 0.0
            Kgrid[self.rows, self.cols] = coeffs
            if sepvar:
                Kgrid[self.rows[self.center], self.cols[self.center]] = 0.0
            S_ij = coeffs.sum() - coeffs[self.center]
            PVAL += FI[ij] * ((cp.fft.fft2(Kgrid) - S_ij) * self.SCALE)
        if sepvar:
            for k in range(spec.Fsca):
                PVAL += (a[k * Fab + self.center] * self.SCALE) * FS[k]
        for pq in range(spec.Fpq):
            PVAL += b[pq] * FT[pq]
        return cp.fft.ifft2(PVAL).real * self.SCALE_L

    def _project(self, FI, FT, FS, X):
        """A.T @ X in the Fourier-normalized inner product (see CPU twin)."""
        spec = self.spec
        Fij, Fpq, Fab = spec.Fij, spec.Fpq, spec.Fab
        Fijab = spec.Fijab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        FX = cp.fft.fft2(X) * self.SCALE
        g = cp.empty(spec.NEQ, dtype=np.float64)
        for i8j8 in range(Fij):
            v = self.SCALE * self._col_vals(self._pre(FI[i8j8], FX))
            g[i8j8 * Fab:(i8j8 + 1) * Fab] = v
        if sepvar:
            for k in range(spec.Fsca):
                g[k * Fab + self.center] = self.SCALE * cp.vdot(FX, FS[k]).real
        for p8q8 in range(Fpq):
            g[Fijab + p8q8] = cp.vdot(FX, FT[p8q8]).real
        return g

    def _reduction(self):
        """(T, Tt, M_reducer): constraint maps, mirroring the CPU engine."""
        spec = self.spec
        NEQ, Fij, Fab = spec.NEQ, spec.Fij, spec.Fab
        from .spec import PolynomialBasis
        if spec.scaling == 'ENTANGLED':
            return (lambda x: x), (lambda g: g), None
        centers = np.arange(Fij) * Fab + self.center
        if spec.scaling == 'SEPARATE-VARYING':
            drop = centers[spec.Fsca:]
            if drop.size == 0:
                return (lambda x: x), (lambda g: g), None
            keep = cp.asarray(np.setdiff1d(np.arange(NEQ), drop))
        elif isinstance(spec.kernel.basis, PolynomialBasis):
            keep = cp.asarray(np.setdiff1d(np.arange(NEQ), centers[1:]))
        else:
            keep_np = np.setdiff1d(np.arange(NEQ), centers)
            R = cp.zeros((NEQ, keep_np.size + 1))
            R[cp.asarray(keep_np), cp.arange(keep_np.size)] = 1.0
            R[cp.asarray(centers), keep_np.size] = 1.0
            return (lambda x: R @ x), (lambda g: R.T @ g), R

        def T(x):
            full = cp.zeros(NEQ)
            full[keep] = x
            return full
        return T, (lambda g: g[keep]), keep

    def solve_weighted(self, I, J, w, tol=1e-10, maxit=100) -> SFFTSolution:
        """Exactly weighted SFFT solve on GPU (see engine_cpu.solve_weighted).

        Regularization: same contract as the CPU engine -- solves
        (A^T W A + R) c = A^T W J with the identical Tikhonov block R the
        classic solve() adds to LHMAT; preconditioner = LU(LHMAT + R)."""
        spec = self.spec
        self._check(I); self._check(J)
        w_np = np.ascontiguousarray(w, np.float64)
        if w_np.shape != spec.shape or not np.isfinite(w_np).all() or (w_np < 0).any():
            raise ValueError('w must match the image shape, be finite and >= 0')
        wg = cp.asarray(w_np)

        t0 = time.time()
        FI, FT, FJ, FS = self._build_spectra(
            cp.asarray(I, dtype=np.float64), cp.asarray(J, dtype=np.float64))
        LHMAT, _ = self._assemble_normal(FI, FT, FJ, FS)
        Rfull = None
        if self._reg is not None:
            Rfull = cp.zeros((spec.NEQ, spec.NEQ))
            self._reg.apply(Rfull)
            LHMAT += Rfull
        T, Tt, TR = self._reduction()

        def _reduce(Mfull):
            if TR is None:
                return Mfull
            if isinstance(TR, cp.ndarray) and TR.ndim == 2:
                return TR.T @ Mfull @ TR
            return Mfull[cp.ix_(TR, TR)]

        M = _reduce(LHMAT)
        Rr = _reduce(Rfull) if Rfull is not None else None
        try:
            from cupyx.scipy.linalg import lu_factor, lu_solve
            M_lu = lu_factor(M)
            precond = lambda r: lu_solve(M_lu, r)
        except ImportError:
            Minv = cp.linalg.inv(M)
            precond = lambda r: Minv @ r
        _sync()
        logger.info('weighted solve: spectra + preconditioner in %.2fs',
                    time.time() - t0)

        t1 = time.time()
        Jg = cp.asarray(J, dtype=np.float64)
        if Rr is None:
            matvec = lambda x: Tt(self._project(
                FI, FT, FS, wg * self._model_image(FI, FT, FS, T(x))))
        else:
            matvec = lambda x: Tt(self._project(
                FI, FT, FS, wg * self._model_image(FI, FT, FS, T(x)))) + Rr @ x
        b = Tt(self._project(FI, FT, FS, wg * Jg))
        x = cp.zeros_like(b)
        r = b - matvec(x)
        z = precond(r)
        p = z.copy()
        rz = r @ z
        nb = float(cp.sqrt(b @ b))
        n_iter = 0
        for n_iter in range(1, maxit + 1):
            Ap = matvec(p)
            al = rz / (p @ Ap)
            x += al * p
            r -= al * Ap
            if float(cp.sqrt(r @ r)) / nb < tol:
                break
            z = precond(r)
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
        full = cp.asnumpy(T(x))
        _sync()
        logger.info('weighted solve: CG converged in %d iterations (%.2fs)',
                    n_iter, time.time() - t1)
        return SFFTSolution(spec=spec, a_ijab=full[:spec.Fijab],
                            b_pq=full[spec.Fijab:])

    def subtract(self, I, J, solution: SFFTSolution, qhat=None,
                 return_device=False):
        spec = self.spec
        N0, N1 = spec.shape
        self._check(I); self._check(J)
        Fab = spec.Fab
        sepvar = (spec.scaling == 'SEPARATE-VARYING')
        t0 = time.time()
        FI, FT, FJ, FS = self._build_spectra(
            cp.asarray(I, dtype=np.float64), cp.asarray(J, dtype=np.float64))

        a_gpu = cp.asarray(solution.a_ijab)
        b_gpu = cp.asarray(solution.b_pq)
        Kgrid = cp.zeros((N0, N1), dtype=np.complex128)
        PVAL = cp.zeros((N0, N1), dtype=np.complex128)
        for ij in range(spec.Fij):
            coeffs = a_gpu[ij * Fab:(ij + 1) * Fab]
            Kgrid[:] = 0.0
            Kgrid[self.rows, self.cols] = coeffs
            if sepvar:
                Kgrid[self.rows[self.center], self.cols[self.center]] = 0.0
            S_ij = coeffs.sum() - coeffs[self.center]
            PVAL += FI[ij] * ((cp.fft.fft2(Kgrid) - S_ij) * self.SCALE)
        if sepvar:
            for k in range(spec.Fsca):
                PVAL += (a_gpu[k * Fab + self.center] * self.SCALE) * FS[k]
        for pq in range(spec.Fpq):
            PVAL += b_gpu[pq] * FT[pq]

        spec_diff = FJ - PVAL
        if qhat is not None:
            spec_diff *= cp.asarray(qhat)
        diff = cp.fft.ifft2(spec_diff).real
        diff *= self.SCALE_L
        if return_device:
            _sync()
            logger.info('difference constructed in %.2fs (device-resident)',
                        time.time() - t0)
            return diff
        out = cp.asnumpy(diff)
        _sync()
        logger.info('difference constructed in %.2fs', time.time() - t0)
        return out

    def run(self, I, J, mI, mJ):
        solution = self.solve(mI, mJ)
        return solution, self.subtract(I, J, solution)

    def _check(self, img):
        if img.shape != self.spec.shape:
            raise ValueError(f'image shape {img.shape} != spec shape {self.spec.shape}')
        xp = cp.get_array_module(img)
        if not xp.isfinite(img).all():
            raise ValueError('input image contains NaN/Inf -- mask or fill first')
