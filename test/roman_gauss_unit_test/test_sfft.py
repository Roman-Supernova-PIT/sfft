import json
import os.path as pa

import pytest
import cupy as cp
import numpy as np
from astropy.io import fits
from sfft.SpaceSFFTCupyFlow import SpaceSFFT_CupyFlow


CDIR = pa.dirname(pa.abspath(__file__))
INPDIR = f"{CDIR}/inputs"
OUTDIR = f"{CDIR}/outputs"


def _load_input_data_and_header(fname):
    data, header = fits.getdata(f"{INPDIR}/{fname}", header=True)
    data = np.ascontiguousarray(data.T)
    return cp.asarray(data), header


def _load_input_data(fname):
    data = fits.getdata(f"{INPDIR}/{fname}")
    data = np.ascontiguousarray(data.T)
    return cp.asarray(data)


def _load_expected(fname):
    expected = fits.getdata(f"{OUTDIR}/{fname}")
    return np.ascontiguousarray(expected.T)


def _to_numpy(array_like):
    if isinstance(array_like, np.ndarray):
        return array_like
    try:
        return cp.asnumpy(array_like)
    except TypeError:
        return np.asarray(array_like)


def _finite_view(a):
    flat = np.asarray(a).reshape(-1)
    return flat[np.isfinite(flat)]


def _safe_stat(func, arr, default=np.nan):
    if arr.size == 0:
        return default
    return float(func(arr))


def _safe_percentile(arr, q):
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _basic_stats(arr):
    arr = np.asarray(arr)
    flat = arr.reshape(-1)
    finite = _finite_view(arr)

    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "size": int(flat.size),
        "finite_count": int(np.isfinite(flat).sum()),
        "nan_count": int(np.isnan(flat).sum()),
        "posinf_count": int(np.isposinf(flat).sum()),
        "neginf_count": int(np.isneginf(flat).sum()),
        "min": _safe_stat(np.min, finite),
        "max": _safe_stat(np.max, finite),
        "mean": _safe_stat(np.mean, finite),
        "median": _safe_stat(np.median, finite),
        "std": _safe_stat(np.std, finite),
        "p01": _safe_percentile(finite, 1),
        "p99": _safe_percentile(finite, 99),
    }


def _error_stats(actual, expected):
    residual = actual - expected
    abs_residual = np.abs(residual)
    rel_residual = abs_residual / (np.abs(expected) + 1e-12)

    abs_finite = _finite_view(abs_residual)
    rel_finite = _finite_view(rel_residual)
    return {
        "mean_abs": _safe_stat(np.mean, abs_finite),
        "median_abs": _safe_stat(np.median, abs_finite),
        "max_abs": _safe_stat(np.max, abs_finite),
        "rmse": _safe_stat(lambda x: np.sqrt(np.mean(x**2)), abs_finite),
        "p99_abs": _safe_percentile(abs_finite, 99),
        "max_rel": _safe_stat(np.max, rel_finite),
        "p99_rel": _safe_percentile(rel_finite, 99),
    }


def _print_basic_stats(name, stats):
    print(f"[{name}]")
    print(f"  shape={stats['shape']} dtype={stats['dtype']} size={stats['size']}")
    print(
        "  bad-values: "
        f"finite={stats['finite_count']} nan={stats['nan_count']} "
        f"+inf={stats['posinf_count']} -inf={stats['neginf_count']}"
    )
    print(
        "  summary: "
        f"min={stats['min']:.6e} p01={stats['p01']:.6e} mean={stats['mean']:.6e} "
        f"median={stats['median']:.6e} std={stats['std']:.6e} p99={stats['p99']:.6e} max={stats['max']:.6e}"
    )


def _assert_matches_fixture(actual, expected_fname, rtol=1e-6, atol=1e-6):
    actual_np = _to_numpy(actual)
    expected_np = _load_expected(expected_fname)
    assert actual_np.shape == expected_np.shape

    comp = _error_stats(actual_np, expected_np)

    print(f"\n=== Comparing fixture: {expected_fname} ===")
    _print_basic_stats("actual", _basic_stats(actual_np))
    _print_basic_stats("expected", _basic_stats(expected_np))
    print(
        "[error]\n"
        f"  mean_abs={comp['mean_abs']:.6e} median_abs={comp['median_abs']:.6e} "
        f"p99_abs={comp['p99_abs']:.6e} max_abs={comp['max_abs']:.6e} rmse={comp['rmse']:.6e}\n"
        f"  p99_rel={comp['p99_rel']:.6e} max_rel={comp['max_rel']:.6e}"
    )

    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=True)


def test_sfft_run_gauss():
    params = json.load(open(f"{INPDIR}/sfft_params.json"))

    PixA_target_GPU, hdr_target = _load_input_data_and_header("gauss_target.fits")
    PixA_object_GPU, hdr_object = _load_input_data_and_header("gauss_object.fits")

    sfftifier = SpaceSFFT_CupyFlow(
        hdr_target=hdr_target,
        hdr_object=hdr_object,
        target_skyrms=params["target_skyrms"],
        object_skyrms=params["object_skyrms"],
        PixA_target_GPU=PixA_target_GPU,
        PixA_object_GPU=PixA_object_GPU,
        PixA_targetVar_GPU=_load_input_data("gauss_targetVar.fits"),
        PixA_objectVar_GPU=_load_input_data("gauss_objectVar.fits"),
        PixA_target_DMASK_GPU=_load_input_data("gauss_targetDmask.fits"),
        PixA_object_DMASK_GPU=_load_input_data("gauss_objectDmask.fits"),
        PSF_target_GPU=_load_input_data("gauss_targetPSF.fits"),
        PSF_object_GPU=_load_input_data("gauss_objectPSF.fits"),
        sci_is_target=params["sci_is_target"],
        GKerHW=params["GKerHW"],
        KerPolyOrder=params["KerPolyOrder"],
        BGPolyOrder=params["BGPolyOrder"],
        ConstPhotRatio=params["ConstPhotRatio"],
        Consider_Matching_Kernel=params["Consider_Matching_Kernel"],
        CUDA_DEVICE_4SUBTRACT=params["CUDA_DEVICE_4SUBTRACT"],
        GAIN=params["GAIN"],
        RANDOM_SEED=params["RANDOM_SEED"],
    )

    sfftifier.resampling_image_mask_psf()
    _assert_matches_fixture(sfftifier.PixA_resamp_object_GPU, "resamp/resamp_object.fits")
    _assert_matches_fixture(sfftifier.PixA_resamp_objectVar_GPU, "resamp/resamp_objectVar.fits")
    _assert_matches_fixture(sfftifier.PixA_resamp_object_DMASK_GPU, "resamp/resamp_objectDmask.fits")
    _assert_matches_fixture(sfftifier.PSF_resamp_object_GPU, "resamp/resamp_objectPSF.fits")
    print("1. Resampling step PASSED.")

    sfftifier.cross_convolution()
    _assert_matches_fixture(sfftifier.PixA_Ctarget_GPU, "crossconv/crossConv_Ctarget.fits")
    _assert_matches_fixture(sfftifier.PSF_Ctarget_GPU, "crossconv/crossConv_PSF_Ctarget.fits")
    _assert_matches_fixture(sfftifier.PixA_Cresamp_object_GPU, "crossconv/crossConv_Cresamp_object.fits")
    print("2. Cross-convolution step PASSED.")

    sfftifier.sfft_subtraction()
    _assert_matches_fixture(sfftifier.Solution_GPU, "sfft_subtract/sfftSub_solution.fits")
    _assert_matches_fixture(sfftifier.PixA_DIFF_GPU, "sfft_subtract/sfftSub_diff.fits")
    print("3.SFFT subtraction step PASSED.")

    sfftifier.find_decorrelation()
    _assert_matches_fixture(sfftifier.MATCH_KERNEL, "find_decorrelation/findDecorr_match_kernel.fits")
    _assert_matches_fixture(sfftifier.FKDECO.real, "find_decorrelation/findDecorr_fkdeco_real.fits")
    _assert_matches_fixture(sfftifier.FKDECO.imag, "find_decorrelation/findDecorr_fkdeco_imag.fits")
    print("4. Finding decorrelation step PASSED.")

    decorrelated_images = [
        (sfftifier.PixA_DIFF_GPU, "diff", "decorrelation/decorrelation_diff.fits"),
        (sfftifier.PixA_Ctarget_GPU, "Ctarget", "decorrelation/decorrelation_Ctarget.fits"),
        (sfftifier.PSF_Ctarget_GPU, "PSF_Ctarget", "decorrelation/decorrelation_PSF_Ctarget.fits"),
    ]
    for image, image_type, expected_path in decorrelated_images:
        decorrelated = sfftifier.apply_decorrelation(image, img_type=image_type)
        _assert_matches_fixture(decorrelated, expected_path)
    print("5. Decorrelation application step PASSED.")

    diff_var = sfftifier.create_variance_image()
    _assert_matches_fixture(diff_var, "create_variance/createVar_dDIFFVar.fits")
    print("6. Variance image creation (of the decorrelated difference) step PASSED.")

    score_image = sfftifier.create_score_image()
    _assert_matches_fixture(score_image, "create_score/createScore_SCORE.fits")
    print("7. Score image creation step PASSED.")

# USAGE: 
# pytest -vs sfft/test/roman_gauss_unit_test/test_sfft.py::test_sfft_run_gauss
