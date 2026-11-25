"""SFFT on a matched pair with a user-supplied fit mask, with the checks that keep a poorly constrained fit out of the product.

The pair is solved by SFFT's refactored core (subtract.core: the streaming CPU engine, whose peak memory is a few image planes, or its CuPy twin)
with polynomial kernel and background bases, after both images, and both masked images, are placed at the origin of a zero-filled array larger
by at least twice the kernel half-width on each axis: SFFT's convolutions are circular, and without that guard band the convolved reference
wraps around the borders. The legacy packets (Customized_Packet, PureCupy_Customized_Packet) solve the same system; they are not used here. The reference is convolved, so the
difference is science minus convolved reference. The flux scale is one constant over the field by default (const_phot_ratio).

guarded() lowers the kernel's polynomial degree to what the fit mask can constrain and refuses a result that is not sane (a kernel centre far
from a delta function, a raised noise level, a singular fit); the matched difference is then the product. Any other error in the solve (a
fault in the code or the installation, a GPU out of memory) is raised: it says nothing about the data.
"""
import time
from copy import deepcopy

import numpy as np

from . import crossconv as cc

MIN_PIXELS_PER_UNKNOWN = 20
KERNEL_CENTRE_RANGE = (0.5, 1.4)
MAX_NOISE_RATIO = 1.15


def _kernels_from_solution(XY_q, Solution, N0, N1, L0, L1, DK, Fpq):
    """SFFT's matching kernels at the positions XY_q (FortranCoor, [(x, y), ...]) from a solution vector in the legacy layout, as an array
    [n, L0, L1]. The arithmetic of sfft/legacy/utils/SFFTSolutionReader.py (Read_SFFTSolution.FromArray, SVKDict_SFFT2ST.convert and
    Realize_MatchingKernel.FromArray), copied unchanged so that the workflow does not import the legacy package."""
    DKx, DKy = DK, DK
    sXY_q = XY_q.astype(float)          # FortranCoor -> ScaledFortranCoor
    sXY_q[:, 0] /= N0
    sXY_q[:, 1] /= N1
    # Read_SFFTSolution.FromArray: the SFFT dictionary
    w0 = int((L0 - 1) / 2)
    w1 = int((L1 - 1) / 2)
    REF_ij = np.array([(i, j) for i in range(DKx + 1) for j in range(DKy + 1 - i)]).astype(int)
    REF_ab = np.array([(a_pos - w0, b_pos - w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(int)
    SREF_ijab = np.array([(ij, ab) for ij in range(REF_ij.shape[0]) for ab in range(REF_ab.shape[0])]).astype(int)
    Fijab = SREF_ijab.shape[0]
    Sfft_dict = {}
    for i in range(DKx + 1):
        for j in range(DKy + 1 - i):
            Sfft_dict[(i, j)] = np.zeros((L0, L1)).astype(float)
    a_ijab = Solution[:-Fpq]            # drop the differential background
    ac_ijab = a_ijab / (N0 * N1)
    for idx in range(Fijab):
        ij, ab = SREF_ijab[idx]
        a, b = REF_ab[ab]
        i, j = REF_ij[ij]
        Sfft_dict[(i, j)][a + w0, b + w1] = ac_ijab[idx]
    # SVKDict_SFFT2ST.convert: the standard dictionary
    Standard_dict = deepcopy(Sfft_dict)
    for i in range(DKx + 1):
        for j in range(DKy + 1 - i):
            Standard_dict[(i, j)][0 + w0, 0 + w1] = 2 * Sfft_dict[(i, j)][0 + w0, 0 + w1] - np.sum(Sfft_dict[(i, j)])
    # Realize_MatchingKernel.FromArray: the kernels at the positions
    B = np.array([sXY_q[:, 0] ** i * sXY_q[:, 1] ** j for i in range(DKx + 1) for j in range(DKy + 1 - i)])
    SSS = np.array([Standard_dict[(i, j)] for i in range(DKx + 1) for j in range(DKy + 1 - i)])
    return np.tensordot(B, SSS, (0, 0))


def choose_degree(fit_mask, wanted, gker_hw, cells=4, min_cell=50):
    """The highest polynomial degree <= wanted that the fit mask constrains: at least MIN_PIXELS_PER_UNKNOWN fit pixels per unknown, spread
    over at least as many of cells x cells parts of the field (each holding min_cell fit pixels or more) as the polynomial has terms. None
    when even a constant kernel is not constrained."""
    n = int(fit_mask.sum())
    ny, nx = fit_mask.shape
    ey, ex = np.linspace(0, ny, cells + 1).astype(int), np.linspace(0, nx, cells + 1).astype(int)
    occupied = sum(int(fit_mask[ey[j]:ey[j + 1], ex[i]:ex[i + 1]].sum() >= min_cell) for j in range(cells) for i in range(cells))
    for d in range(int(wanted), -1, -1):
        terms = (d + 1) * (d + 2) // 2
        if n >= MIN_PIXELS_PER_UNKNOWN * (2 * gker_hw + 1) ** 2 * terms and occupied >= terms:
            return d
    return None


def guard_band_shape(shape_xy, gker_hw):
    """The solve grid (x, y) for an image of shape_xy: each axis at least the image plus twice the kernel half-width, rounded up to a size the
    FFT handles fast."""
    from scipy.fft import next_fast_len
    return next_fast_len(int(shape_xy[0]) + 2 * int(gker_hw)), next_fast_len(int(shape_xy[1]) + 2 * int(gker_hw))


def subtract(sci_w, ref_w, fit_mask, gker_hw=2, kernel_degree=2, bg_degree=0, const_phot_ratio=True, backend="gpu", device=0, threads=8,
             numba_cache=True):
    """SFFT difference of a matched pair ([y, x] arrays; NaN where an image has no value). fit_mask: True where the fit may use the pixel.
    Returns dict(diff, solution, solve_shape, fscal, kernel_center, runtime, fit_pixels): diff is NaN where either image is NaN; solution is
    the solver's coefficient vector (kernel terms, then background terms) for the solve grid of shape solve_shape (x, y; guard_band_shape),
    the image at its origin; kernel_center is the matching kernel at the solve grid's centre in the solver's (x, y) orientation, with its
    physical normalisation (its sum is the flux scale fscal).

    The solver is subtract.core with the reference convolved: SFFTEngine on the CPU (threads), CuPySFFTEngine on the GPU (device). The flux
    scale is one constant over the field (const_phot_ratio=True: scaling 'SEPARATE-CONSTANT') or follows the kernel's spatial variation
    (False: 'ENTANGLED'). numba_cache is accepted for compatibility: the core's compiled loops are always cached."""
    from .core.spec import BackgroundSpec, KernelSpec, PolynomialBasis, SFFTSpec
    nan = ~np.isfinite(sci_w) | ~np.isfinite(ref_w)
    use = fit_mask & ~nan
    ref_img = np.where(nan, 0.0, ref_w)
    sci_img = np.where(nan, 0.0, sci_w)
    mref = np.where(use, ref_img, 0.0)
    msci = np.where(use, sci_img, 0.0)
    N0, N1 = ref_img.shape[1], ref_img.shape[0]
    hw, dk, db = int(gker_hw), int(kernel_degree), int(bg_degree)
    P0, P1 = guard_band_shape((N0, N1), hw)

    def embed(a):
        out = np.zeros((P0, P1))
        out[:N0, :N1] = a.T
        return out

    spec = SFFTSpec(shape=(P0, P1), kernel=KernelSpec(hw=hw, basis=PolynomialBasis(dk)), background=BackgroundSpec(basis=PolynomialBasis(db)),
                    scaling="SEPARATE-CONSTANT" if const_phot_ratio else "ENTANGLED")
    t0 = time.time()
    if backend == "gpu":
        import cupy as cp
        cp.cuda.Device(int(device)).use()
        from .core.engine_gpu import CuPySFFTEngine
        engine = CuPySFFTEngine(spec)
    else:
        from .core.engine_cpu import SFFTEngine
        engine = SFFTEngine(spec, threads=int(threads))
    solution = engine.solve(embed(mref), embed(msci))
    diff = engine.subtract(embed(ref_img), embed(sci_img), solution)
    runtime = time.time() - t0
    diff = diff.get() if hasattr(diff, "get") else diff
    d = np.array(np.asarray(diff)[:N0, :N1].T, dtype=np.float64)
    d[nan] = np.nan
    sol = solution.full
    sol = np.asarray(sol.get() if hasattr(sol, "get") else sol, dtype=np.float64)
    L = 2 * hw + 1
    kernel = _kernels_from_solution(np.array([[P0 / 2.0, P1 / 2.0]]), sol, P0, P1, L, L, dk, (db + 1) * (db + 2) // 2)[0]
    return dict(diff=d, solution=sol, solve_shape=(P0, P1), fscal=float(np.sum(kernel)), kernel_center=np.asarray(kernel),
                runtime=float(runtime), fit_pixels=int(use.sum()))


def guarded(sci_w, ref_w, fit_mask, free, gker_hw=2, kernel_degree=2, bg_degree=0, const_phot_ratio=True, backend="gpu", device=0, threads=8,
            numba_cache=True):
    """SFFT with its checks: dict(diff, used, problem, fscal, kernel_centre, kernel, kernel_physical, solution, solve_shape, degree, fit_pixels,
    runtime, sigma_before, sigma_after). free: the source-free pixels on which the noise is measured. When the fit is not possible (too few fit
    pixels, a singular linear system) or its result is not sane, diff is the matched difference sci_w - ref_w, used is False and problem says
    why; any other error in the solve is raised. kernel is the kernel divided by the flux scale; kernel_physical the kernel itself, in the
    solver's (x, y) orientation."""
    matched = sci_w - ref_w
    sigma_before = cc.robust_sigma(matched[free])
    use = fit_mask & np.isfinite(matched)
    info = dict(diff=matched, used=False, problem=None, fscal=None, kernel_centre=None, kernel=None, kernel_physical=None, solution=None,
                solve_shape=None, degree=None,
                fit_pixels=int(use.sum()), runtime=0.0, sigma_before=float(sigma_before), sigma_after=float(sigma_before))
    degree = choose_degree(use, kernel_degree, gker_hw)
    if degree is None:
        info["problem"] = f"only {info['fit_pixels']} fit pixels, or too few parts of the field hold any: SFFT skipped"
        return info
    info["degree"] = degree
    try:
        r = subtract(sci_w, ref_w, use, gker_hw=gker_hw, kernel_degree=degree, bg_degree=bg_degree, const_phot_ratio=const_phot_ratio,
                     backend=backend, device=device, threads=threads, numba_cache=numba_cache)
    except np.linalg.LinAlgError as e:      # a singular linear system: the fit mask cannot constrain the kernel (any other error is raised)
        info["problem"] = f"SFFT's linear system is singular ({str(e)[:120]}): SFFT skipped"
        return info
    fscal = float(r["fscal"])
    k = np.asarray(r["kernel_center"], float) / (fscal if np.isfinite(fscal) and fscal != 0 else 1.0)
    centre = float(k[k.shape[0] // 2, k.shape[1] // 2])
    sigma_after = cc.robust_sigma(r["diff"][free])
    info.update(fscal=fscal if np.isfinite(fscal) else None, kernel_centre=centre if np.isfinite(centre) else None,
                kernel=k.tolist() if np.isfinite(k).all() else None, kernel_physical=np.asarray(r["kernel_center"], float), solution=r["solution"], solve_shape=r["solve_shape"],
                runtime=float(r["runtime"]), sigma_after=float(sigma_after))
    if not (np.isfinite(fscal) and fscal > 0 and np.isfinite(centre) and np.isfinite(sigma_after)):
        info["problem"] = "SFFT returned NaN (a singular fit): SFFT result not used"
    elif not KERNEL_CENTRE_RANGE[0] <= centre <= KERNEL_CENTRE_RANGE[1]:
        info["problem"] = (f"kernel centre {centre:.3f} is outside {KERNEL_CENTRE_RANGE}: SFFT result not used. Check the inputs: cosmic rays or "
                           "hot pixels in the images, or a wrong PSF")
    elif sigma_after > MAX_NOISE_RATIO * sigma_before and sigma_after > 1e-6:
        info["problem"] = f"SFFT raised the noise of the difference from {sigma_before:.4f} to {sigma_after:.4f}: SFFT result not used"
    if info["problem"] is None:
        info.update(diff=r["diff"], used=True)
    else:
        info["sigma_after"] = float(sigma_before)
    return info


def kernels_at(solution, solve_shape, positions, degree, bg_degree, gker_hw):
    """SFFT's matching kernel at 0-based image positions [(x, y), ...] (the image at the origin of the solve grid of shape solve_shape, as
    subtract() places it), each a [y, x] array with its physical normalisation (its sum is the flux scale there)."""
    P0, P1 = solve_shape
    L = 2 * int(gker_hw) + 1
    xy = np.array([[float(x) + 1.0, float(y) + 1.0] for x, y in positions])
    K = _kernels_from_solution(xy, np.asarray(solution), int(P0), int(P1), L, L, int(degree), (int(bg_degree) + 1) * (int(bg_degree) + 2) // 2)
    return [np.ascontiguousarray(np.asarray(k).T) for k in K]
