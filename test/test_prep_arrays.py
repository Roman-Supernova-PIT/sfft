"""The romanL2 preparation from arrays (sfft.spaceflow.prep): the WCS as a FITS header or a gWCS, the masks and optional inputs and the
warning each gives when it is missing, the sky and unit switches, the PSF inputs, and SpaceL2_Prep / SpaceL2_Flow end to end on small
synthetic exposures (CPU backend). Tests that run SExtractor (the source mask, the fit mask) are skipped when betasfft's SExtractor program
is not built (python sfft/spaceflow/prep/_sextractor/build.py). Run from the repository root: python -m pytest test"""
import inspect
import json
import os
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sfft.spaceflow import SpaceL2_Flow
from sfft.spaceflow.prep import SpaceL2_Prep
from sfft.spaceflow.prep.astromatic import bundled
from sfft.spaceflow.prep.reader import INPUT_UNIT, Level2, exposure_wcs

N_PIXELS = 256
SCALE_DEG = 0.11 / 3600
SKY = 10.0
SIGMA_TARGET, SIGMA_OBJECT = 1.2, 1.5
needs_sextractor = pytest.mark.skipif(not os.path.exists(bundled.PROGRAM),
                                      reason="betasfft's SExtractor program is not built (python sfft/spaceflow/prep/_sextractor/build.py)")


def tan_header(crval):
    header = fits.Header()
    header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 2, N_PIXELS, N_PIXELS
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRPIX1"] = header["CRPIX2"] = N_PIXELS / 2 + 0.5
    header["CRVAL1"], header["CRVAL2"] = crval
    header["CD1_1"], header["CD1_2"], header["CD2_1"], header["CD2_2"] = -SCALE_DEG, 0.0, 0.0, SCALE_DEG
    return header


def tan_gwcs(crval):
    """The gWCS of tan_header(crval)."""
    import astropy.units as u
    from astropy.coordinates import ICRS
    from astropy.modeling import models
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gw
    crpix0 = N_PIXELS / 2 - 0.5
    transform = ((models.Shift(-crpix0) & models.Shift(-crpix0)) | (models.Scale(-SCALE_DEG) & models.Scale(SCALE_DEG)) | models.Pix2Sky_TAN()
                 | models.RotateNative2Celestial(crval[0], crval[1], 180))
    detector = cf.Frame2D(name="detector", axes_order=(0, 1), unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=ICRS(), name="world", unit=(u.deg, u.deg))
    result = gw.WCS([(detector, transform), (sky, None)])
    result.bounding_box = ((-0.5, N_PIXELS - 0.5), (-0.5, N_PIXELS - 0.5))
    return result


def gaussian_psf(sigma, size=31):
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def star_image(header, stars, sigma, rng):
    wcs = WCS(header)
    y, x = np.mgrid[:N_PIXELS, :N_PIXELS]
    image = SKY + rng.normal(0.0, 1.0, (N_PIXELS, N_PIXELS))
    for ra, dec, flux in stars:
        px, py = wcs.all_world2pix(ra, dec, 0)
        image += flux * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return image


@pytest.fixture(scope="module")
def pair():
    """Two synthetic exposures of 60 stars on a sky of SKY, offset by a fraction of a pixel, with every optional array."""
    rng = np.random.default_rng(2)
    crval_t, crval_o = (150.0, 2.0), (150.0 + 3e-5, 2.0 - 2e-5)
    header_t = tan_header(crval_t)
    stars = []
    for _ in range(60):
        px, py = rng.uniform(20, N_PIXELS - 20, 2)
        ra, dec = WCS(header_t).all_pix2world(px, py, 0)
        stars.append((float(ra), float(dec), rng.uniform(500, 5000)))
    out = dict(crval_target=crval_t, crval_object=crval_o)
    for side, crval, sigma in (("target", crval_t, SIGMA_TARGET), ("object", crval_o, SIGMA_OBJECT)):
        bad = np.zeros((N_PIXELS, N_PIXELS), bool)
        bad[10, 10:20] = True
        jump = np.zeros((N_PIXELS, N_PIXELS), bool)
        jump[30, 30:40] = True
        out[side] = dict(image=star_image(tan_header(crval), stars, sigma, rng), variance=np.ones((N_PIXELS, N_PIXELS)),
                         header=tan_header(crval), bad=bad, saturated=np.zeros((N_PIXELS, N_PIXELS), bool), jump=jump,
                         dumo=np.full((N_PIXELS, N_PIXELS), 0.5), var_poisson=np.ones((N_PIXELS, N_PIXELS)), psf=gaussian_psf(sigma))
    return out


def all_inputs(e, side="target"):
    """Level2's keywords with every optional input of exposure e passed."""
    return dict(side=side, bad_mask=e["bad"], saturated_mask=e["saturated"], jump_mask=e["jump"], dumo=e["dumo"], var_poisson=e["var_poisson"],
                conversion_megajanskys=2.0, pixel_area_sr=float((0.11 / 206264.80624709636) ** 2), detector="WFI01", filter_name="F106")


QUIET = dict(mask_sources=False, sky_model="constant", reject_outliers=False)     # no SExtractor, and one sky level


# ------------------------------------------------------------------------------------------------------------------------------ the WCS
def test_gwcs_and_fits_header_give_the_same_exposure(pair):
    e = pair["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_header = Level2(e["image"], e["variance"], e["header"], **all_inputs(e), **QUIET)
        from_gwcs = Level2(e["image"], e["variance"], tan_gwcs(pair["crval_target"]), **all_inputs(e), **QUIET)
    np.testing.assert_allclose(from_gwcs.image, from_header.image, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(from_gwcs.variance[np.isfinite(from_gwcs.variance)], from_header.variance[np.isfinite(from_header.variance)],
                               rtol=1e-6)
    np.testing.assert_allclose(from_gwcs.area_sr, from_header.area_sr, rtol=1e-6)
    assert from_gwcs.header["CTYPE1"].startswith("RA---TAN")


def test_fits_header_needs_naxis():
    header = tan_header((150.0, 2.0))
    del header["NAXIS1"]
    with pytest.raises(ValueError, match="NAXIS1 and NAXIS2"):
        exposure_wcs(header, (N_PIXELS, N_PIXELS))


def test_fits_header_naxis_must_match_the_image():
    with pytest.raises(ValueError, match="but the image is"):
        exposure_wcs(tan_header((150.0, 2.0)), (N_PIXELS, N_PIXELS + 1))


def test_wcs_of_another_kind_is_refused():
    with pytest.raises(TypeError, match="gWCS or an astropy.io.fits.Header"):
        exposure_wcs({"CRVAL1": 150.0}, (N_PIXELS, N_PIXELS))


def test_mask_of_another_shape_is_refused(pair):
    e = pair["target"]
    with pytest.raises(ValueError, match="PixA_target_BADMASK has shape"):
        Level2(e["image"], e["variance"], e["header"], bad_mask=np.zeros((5, 5), bool), **QUIET)


# --------------------------------------------------------------------------------------------------------------- missing optional inputs
MISSING = [
    ("bad_mask", "no bad-pixel mask (PixA_target_BADMASK)"),
    ("saturated_mask", "no saturation mask (PixA_target_SATMASK)"),
    ("jump_mask", "no cosmic-ray jump mask (PixA_target_JUMPMASK) passed: dumo is added at every pixel"),
    ("dumo", "no dumo array (PixA_target_DUMO)"),
    ("var_poisson", "no var_poisson array (PixA_target_VARPOISSON)"),
    ("conversion_megajanskys", "no flux conversion (conversion_megajanskys_target)"),
]


@pytest.mark.parametrize("left_out, message", MISSING)
def test_each_missing_input_warns(pair, left_out, message):
    e = pair["target"]
    keywords = all_inputs(e)
    keywords.pop(left_out)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Level2(e["image"], e["variance"], e["header"], name="sci", **keywords, **QUIET)
    texts = [str(w.message) for w in caught]
    assert [t for t in texts if t.startswith("sci: " + message)], texts
    assert len([t for t in texts if t.startswith("sci: no ")]) == 1, texts


def test_jump_mask_without_dumo_says_so(pair):
    e = pair["target"]
    keywords = all_inputs(e)
    keywords.pop("jump_mask")
    keywords.pop("dumo")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Level2(e["image"], e["variance"], e["header"], **keywords, **QUIET)
    assert any("it is only used to place dumo, which is not passed either" in str(w.message) for w in caught)


def test_no_missing_input_warning_when_everything_is_passed(pair):
    e = pair["target"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Level2(e["image"], e["variance"], e["header"], **all_inputs(e), **QUIET)
    assert not [str(w.message) for w in caught if ": no " in str(w.message)]


# ----------------------------------------------------------------------------------------------------------------- what the inputs do
def test_bad_mask_marks_pixels_unusable(pair):
    e = pair["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Level2(e["image"], e["variance"], e["header"], **all_inputs(e), **QUIET)
    assert L.dnu[e["bad"]].all()
    assert (L.image[e["bad"]] == 0).all() and np.isinf(L.variance[e["bad"]]).all()


def test_without_bad_mask_only_non_finite_pixels_are_unusable(pair):
    e = pair["target"]
    image = e["image"].copy()
    image[50, 50] = np.nan
    keywords = all_inputs(e)
    keywords.pop("bad_mask")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Level2(image, e["variance"], e["header"], **keywords, **QUIET)
    assert L.dnu[50, 50] and not L.dnu[e["bad"]].any()


def test_dumo_is_not_added_at_jump_pixels(pair):
    e = pair["target"]
    keywords = all_inputs(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_dumo = Level2(e["image"], e["variance"], e["header"], **keywords, **QUIET, subtract_sky=False, convert_units=False)
        keywords["dumo"] = None
        without = Level2(e["image"], e["variance"], e["header"], **keywords, **QUIET, subtract_sky=False, convert_units=False)
    added = with_dumo.image - without.image
    np.testing.assert_allclose(added[e["jump"]], 0.0)
    usable = ~(e["jump"] | e["bad"])
    np.testing.assert_allclose(added[usable], 0.5)


def test_switches_off_keep_the_input(pair):
    e = pair["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Level2(e["image"], e["variance"], e["header"], **{**all_inputs(e), "dumo": None}, **QUIET, subtract_sky=False, convert_units=False)
    good = ~L.dnu
    np.testing.assert_allclose(L.image[good], e["image"][good])
    assert L.unit == INPUT_UNIT and L.sky_file == 0.0


def test_conversion_and_sky(pair):
    e = pair["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Level2(e["image"], e["variance"], e["header"], **{**all_inputs(e), "dumo": None}, **QUIET)
    assert L.unit == "MJy/sr"
    assert abs(L.sky_file - SKY) < 0.1
    good = ~L.dnu
    np.testing.assert_allclose(L.image[good], (e["image"][good] - L.sky_file) * 2.0)
    np.testing.assert_allclose(L.variance[good], 4.0)


def test_light_per_pixel_is_made_a_surface_brightness(pair):
    e = pair["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = Level2(e["image"], e["variance"], e["header"], **{**all_inputs(e), "dumo": None}, is_surface_brightness=False, **QUIET)
    good = ~L.dnu
    expected = (e["image"] - L.sky_file) * 2.0 * L.nominal_area_sr / L.area_sr.astype(np.float64)
    np.testing.assert_allclose(L.image[good], expected[good], rtol=1e-6)


# ------------------------------------------------------------------------------------------------------------------------- SpaceL2_Prep
def prep_inputs(pair, psf=True):
    """SpaceL2_Prep's six positional inputs and its keywords, with every optional array and the caller's PSFs (or detector and filter)."""
    t, o = pair["target"], pair["object"]
    positional = (t["image"], o["image"], t["variance"], o["variance"], t["header"], o["header"])
    keywords = dict(PixA_target_BADMASK=t["bad"], PixA_object_BADMASK=o["bad"], PixA_target_SATMASK=t["saturated"],
                    PixA_object_SATMASK=o["saturated"], PixA_target_JUMPMASK=t["jump"], PixA_object_JUMPMASK=o["jump"],
                    PixA_target_DUMO=t["dumo"], PixA_object_DUMO=o["dumo"], PixA_target_VARPOISSON=t["var_poisson"],
                    PixA_object_VARPOISSON=o["var_poisson"], conversion_megajanskys_target=1.0, conversion_megajanskys_object=1.0,
                    name_target="sci", name_object="ref", add_dumo=False, sky_model="constant")
    if psf:
        keywords.update(PSF_target=t["psf"], PSF_object=o["psf"])
    return positional, keywords


def test_prep_takes_six_positional_inputs_then_keywords():
    parameters = list(inspect.signature(SpaceL2_Prep.__init__).parameters.values())[1:]
    positional = [p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == ["PixA_target", "PixA_object", "PixA_targetVar", "PixA_objectVar", "wcs_target", "wcs_object"]
    assert all(p.kind is p.KEYWORD_ONLY for p in parameters[6:])


def test_flow_takes_six_positional_inputs_then_keywords():
    parameters = list(inspect.signature(SpaceL2_Flow.__init__).parameters.values())[1:]
    positional = [p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD]
    assert positional == ["PixA_target", "PixA_object", "PixA_targetVar", "PixA_objectVar", "wcs_target", "wcs_object"]


def test_flow_refuses_an_unknown_option(pair):
    positional, keywords = prep_inputs(pair)
    with pytest.raises(TypeError, match="unknown option"):
        SpaceL2_Flow(*positional, **keywords, keep_flags=("WARM",))


def test_prep_needs_a_psf_or_detector_and_filter(pair):
    positional, keywords = prep_inputs(pair, psf=False)
    with pytest.raises(ValueError, match="no PSF for the target exposure"):
        SpaceL2_Prep(*positional, **keywords)


def test_prep_refuses_two_filters(pair):
    positional, keywords = prep_inputs(pair)
    with pytest.raises(ValueError, match="different filters"):
        SpaceL2_Prep(*positional, **keywords, filter_target="F106", filter_object="F129")


def test_prep_refuses_4d_psf_at_another_number_of_nodes(pair):
    positional, keywords = prep_inputs(pair)
    keywords["PSF_target"] = np.tile(pair["target"]["psf"], (3, 3, 1, 1))
    with pytest.raises(ValueError, match="at 3 x 3 nodes, but the PSF nodes are 5 x 5"):
        SpaceL2_Prep(*positional, **keywords)


def test_prep_refuses_the_shared_array_psf_source(pair):
    positional, keywords = prep_inputs(pair)
    with pytest.raises(ValueError, match="PSF_target / PSF_object"):
        SpaceL2_Prep(*positional, **keywords, psf=dict(source="array", stamp=pair["target"]["psf"], oversample=1))


@needs_sextractor
@pytest.mark.parametrize("wcs_position_type, extra", [("chief-ray", None), ("first-moment-centroid", "is applied only to PSFs evaluated from STPSF")])
def test_caller_psf_warns_that_it_is_not_corrected(pair, wcs_position_type, extra):
    positional, keywords = prep_inputs(pair)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SpaceL2_Prep(*positional, **keywords, wcs_position_type=wcs_position_type)
    for side in ("target", "object"):
        found = [str(w.message) for w in caught if str(w.message).startswith(f"PSF_{side} was passed directly, not evaluated from STPSF")]
        assert len(found) == 1, [str(w.message) for w in caught]
        assert (extra in found[0]) if extra else ("wcs_position_type=" not in found[0])


@needs_sextractor
def test_outliers_are_shared_only_with_equal_detector_names(pair):
    positional, keywords = prep_inputs(pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unnamed = SpaceL2_Prep(*positional, **keywords)
        named = SpaceL2_Prep(*positional, **keywords, detector_target="WFI01", detector_object="WFI01")
    assert unnamed.science.rejected_shared == 0 and unnamed.reference.rejected_shared == 0
    assert named.psf_sources == dict(target="caller", object="caller")


@needs_sextractor
def test_prep_and_subtraction_run_end_to_end(pair):
    from sfft.spaceflow.subtract import SpaceL2_Sub
    positional, keywords = prep_inputs(pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = SpaceL2_Prep(*positional, **keywords)
        arguments = prep.sub_arguments()
        sub = SpaceL2_Sub(**arguments, BACKEND_4SUBTRACT="Numpy", NUM_CPU_THREADS_4SUBTRACT=4)
        sub.resample_image_mask_psf()
        sub.cross_convolve()
        sub.sfft_subtract(fit_mask=prep.fit_mask(sub))
        sub.find_decorrelation()
        difference = sub.apply_decorrelation(sub.PixA_DIFF)
    assert arguments["PSF_target"] is prep._caller_psfs["target"][0]
    assert difference.shape == (N_PIXELS, N_PIXELS) and np.isfinite(difference).mean() > 0.5
    assert prep.saturation_map is not None and prep.units["bunit"] == "MJy/sr"


@needs_sextractor
def test_flow_runs_and_records_its_inputs_by_hash(pair, tmp_path):
    positional, keywords = prep_inputs(pair)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = SpaceL2_Flow(*positional, output_dir=str(tmp_path), backend="cpu", cpu_threads=4, **keywords)
        record = flow.run()
    assert record["status"] == "done"
    saved = json.loads((tmp_path / "record.json").read_text())
    assert saved["inputs"]["PixA_target"]["shape"] == [N_PIXELS, N_PIXELS] and len(saved["inputs"]["PixA_target"]["sha256"]) == 64
    assert saved["inputs"]["wcs_target"]["kind"] == "FITS header"
    assert saved["settings"]["prep"]["PSF_target"]["shape"] == [31, 31]
    assert saved["psf"]["source"] == dict(target="caller", object="caller")
    with fits.open(tmp_path / "diff.fits") as hdul:
        assert hdul[0].header["SCIFILE"] == "sci" and hdul[0].header["REFFILE"] == "ref"


def test_stpsf_missing_says_how_to_install_it(pair, monkeypatch):
    import importlib.util
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "stpsf" else real_find_spec(name, *a))
    positional, keywords = prep_inputs(pair, psf=False)
    with pytest.raises(ImportError, match=r"pip install 'sfft-romansnpit\[stpsf\]'"):
        SpaceL2_Prep(*positional, **keywords, detector_target="WFI01", filter_target="F106", detector_object="WFI01", filter_object="F106")
