"""The compiled inner loops (numba, parallel) of the CPU engine."""
import numba as nb
import numpy as np

_OPTS = dict(parallel=True, cache=True)


@nb.njit(**_OPTS)
def mul_conj(A, B, out):
    """out = A * conj(B), elementwise (fused conjugation, no temp stacks)."""
    n0, n1 = A.shape
    for r in nb.prange(n0):
        for c in range(n1):
            out[r, c] = A[r, c] * np.conj(B[r, c])


@nb.njit(**_OPTS)
def sum_mul_conj_real(A, B):
    """Re[ sum(A * conj(B)) ] -- the (0,0) DFT term without an FFT."""
    n0, n1 = A.shape
    acc = 0.0
    for r in nb.prange(n0):
        s = 0.0
        for c in range(n1):
            s += (A[r, c].real * B[r, c].real + A[r, c].imag * B[r, c].imag)
        acc += s
    return acc


@nb.njit(inline='always')
def _mod(v, N):
    t = v % N
    if t < 0:
        t += N
    return t


@nb.njit(**_OPTS)
def fill_omg_slice(i8j8, ij, pre, scale, ref_ab, Fab, LHMAT, N0, N1):
    """A-block rows [i8j8*Fab : +Fab], cols [ij*Fab : +Fab] from one OMG slice."""
    p00 = scale * pre[0, 0]
    for a8b8 in nb.prange(Fab):
        a8 = ref_ab[a8b8, 0]
        b8 = ref_ab[a8b8, 1]
        ROW = i8j8 * Fab + a8b8
        ma8 = _mod(a8, N0)
        mb8 = _mod(b8, N1)
        c8 = (a8 == 0 and b8 == 0)
        pa8b8 = scale * pre[ma8, mb8]
        for ab in range(Fab):
            a = ref_ab[ab, 0]
            b = ref_ab[ab, 1]
            COL = ij * Fab + ab
            c0 = (a == 0 and b == 0)
            if c8 and c0:
                LHMAT[ROW, COL] = p00
            elif c8:
                LHMAT[ROW, COL] = scale * pre[_mod(-a, N0), _mod(-b, N1)] - p00
            elif c0:
                LHMAT[ROW, COL] = pa8b8 - p00
            else:
                LHMAT[ROW, COL] = (- pa8b8
                                   - scale * pre[_mod(-a, N0), _mod(-b, N1)]
                                   + scale * pre[_mod(a8 - a, N0), _mod(b8 - b, N1)]
                                   + p00)


@nb.njit(**_OPTS)
def fill_omg11_slice(i8j8, ij, pre, scale, ref_ab, Fab, LHMAT, N0, N1):
    """fill_omg_slice restricted to non-center rows x non-center cols.

    SEPARATE-VARYING: center DOFs live on the scaling basis, so the
    kernel-basis OMG slice must not touch center rows/cols (they are filled
    from the FS spectra by fill_omg10_col / fill_omg01_row / plain sums).
    """
    p00 = scale * pre[0, 0]
    for a8b8 in nb.prange(Fab):
        a8 = ref_ab[a8b8, 0]
        b8 = ref_ab[a8b8, 1]
        if a8 == 0 and b8 == 0:
            continue
        ROW = i8j8 * Fab + a8b8
        pa8b8 = scale * pre[_mod(a8, N0), _mod(b8, N1)]
        for ab in range(Fab):
            a = ref_ab[ab, 0]
            b = ref_ab[ab, 1]
            if a == 0 and b == 0:
                continue
            LHMAT[ROW, ij * Fab + ab] = (- pa8b8
                                         - scale * pre[_mod(-a, N0), _mod(-b, N1)]
                                         + scale * pre[_mod(a8 - a, N0), _mod(b8 - b, N1)]
                                         + p00)


@nb.njit(**_OPTS)
def fill_omg10_col(i8j8, COL, pre, scale, ref_ab, Fab, LHMAT, N0, N1):
    """Non-center rows of block i8j8 -> one center column, from FI*conj(FS)."""
    p00 = scale * pre[0, 0]
    for a8b8 in nb.prange(Fab):
        a8 = ref_ab[a8b8, 0]
        b8 = ref_ab[a8b8, 1]
        if a8 == 0 and b8 == 0:
            continue
        LHMAT[i8j8 * Fab + a8b8, COL] = scale * pre[_mod(a8, N0), _mod(b8, N1)] - p00


@nb.njit(**_OPTS)
def fill_omg01_row(ROW, ij, pre, scale, ref_ab, Fab, LHMAT, N0, N1):
    """One center row -> non-center cols of block ij, from FS*conj(FI)."""
    p00 = scale * pre[0, 0]
    for ab in nb.prange(Fab):
        a = ref_ab[ab, 0]
        b = ref_ab[ab, 1]
        if a == 0 and b == 0:
            continue
        LHMAT[ROW, ij * Fab + ab] = scale * pre[_mod(-a, N0), _mod(-b, N1)] - p00


@nb.njit(**_OPTS)
def fill_gam_slice(i8j8, pq, pre, scale, ref_ab, Fab, Fijab, LHMAT, N0, N1,
                   skip_center):
    """B-block rows [i8j8*Fab : +Fab], single col Fijab+pq from one GAM slice.

    skip_center: leave the center row untouched (SEPARATE-VARYING fills it
    from the scaling-basis spectrum as a plain sum).
    """
    p00 = scale * pre[0, 0]
    cCOL = Fijab + pq
    for a8b8 in nb.prange(Fab):
        a8 = ref_ab[a8b8, 0]
        b8 = ref_ab[a8b8, 1]
        ROW = i8j8 * Fab + a8b8
        if a8 == 0 and b8 == 0:
            if not skip_center:
                LHMAT[ROW, cCOL] = p00
        else:
            LHMAT[ROW, cCOL] = scale * pre[_mod(a8, N0), _mod(b8, N1)] - p00


@nb.njit(**_OPTS)
def fill_psi_slice(p8q8, ij, pre, scale, ref_ab, Fab, Fijab, LHMAT, N0, N1,
                   skip_center):
    """B#-block single row Fijab+p8q8, cols [ij*Fab : +Fab] from one PSI slice."""
    p00 = scale * pre[0, 0]
    cROW = Fijab + p8q8
    for ab in nb.prange(Fab):
        a = ref_ab[ab, 0]
        b = ref_ab[ab, 1]
        COL = ij * Fab + ab
        if a == 0 and b == 0:
            if not skip_center:
                LHMAT[cROW, COL] = p00
        else:
            LHMAT[cROW, COL] = scale * pre[_mod(-a, N0), _mod(-b, N1)] - p00


@nb.njit(**_OPTS)
def fill_the_slice(i8j8, pre, scale, ref_ab, Fab, RHb, N0, N1, skip_center):
    """D-component rows [i8j8*Fab : +Fab] of RHb from one THE slice."""
    p00 = scale * pre[0, 0]
    for a8b8 in nb.prange(Fab):
        a8 = ref_ab[a8b8, 0]
        b8 = ref_ab[a8b8, 1]
        ROW = i8j8 * Fab + a8b8
        if a8 == 0 and b8 == 0:
            if not skip_center:
                RHb[ROW] = p00
        else:
            RHb[ROW] = scale * pre[_mod(a8, N0), _mod(b8, N1)] - p00


@nb.njit(**_OPTS)
def fdiff_mac(PVAL, FI_ij, K, S_ij, scale):
    """PVAL += FI_ij * (scale * (K - S_ij)) -- one kernel-basis term of FDIFF.

    K is the DFT of the coefficient grid C_ij; S_ij is the sum of the
    non-center coefficients (the exact factorization of the legacy
    Construct_FDIFF delta-basis expression).
    """
    n0, n1 = PVAL.shape
    for r in nb.prange(n0):
        for c in range(n1):
            PVAL[r, c] += FI_ij[r, c] * (scale * (K[r, c] - S_ij))


@nb.njit(**_OPTS)
def bg_mac(PVAL, FT_pq, b_pq):
    """PVAL += b_pq * FT_pq -- one background term of FDIFF."""
    n0, n1 = PVAL.shape
    for r in nb.prange(n0):
        for c in range(n1):
            PVAL[r, c] += b_pq * FT_pq[r, c]
