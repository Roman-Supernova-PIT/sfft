"""SpaceSFFT, the wrapper of both engines: the engine choice, the keywords each engine takes, attribute forwarding, and the step sequences
phrosty and sidecar run (on small synthetic images, CPU backend). Run from the repository root: python -m pytest test"""
import inspect
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sfft import SpaceSFFT
from sfft.legacy.SpaceSFFTFlow import SpaceSFFT_Flow as LegacySpaceSFFT_Flow
from sfft.spaceflow.subtract import SpaceL2_Sub

N_PIXELS = 256
SIGMA_TARGET, SIGMA_OBJECT = 1.2, 1.5          # Gaussian PSF widths [pixels]
CPU = dict(BACKEND_4SUBTRACT="Numpy", NUM_CPU_THREADS_4SUBTRACT=4, GKerHW=4)

# what phrosty (phrosty/pipeline.py) and sidecar (sidecar/subtraction.py) read from the subtraction object
PHROSTY_ATTRIBUTES = ("op", "hdr_target", "PixA_DIFF", "PixA_Ctarget", "PixA_Cresamp_object", "PixA_resamp_object",
                      "PixA_resamp_objectVar", "PixA_resamp_object_DMASK", "PSF_target", "PSF_Ctarget", "MATCH_KERNEL", "FKDECO")
SIDECAR_ATTRIBUTES = ("op", "hdr_target", "PixA_DIFF", "PixA_target", "PixA_target_DMASK", "PixA_resamp_object", "PixA_resamp_objectVar",
                      "PixA_resamp_object_DMASK", "PSF_target", "BlankMask", "CROSS_CONVOLVED", "MATCH_KERNEL")


def tan_header(crval):
    header = fits.Header()
    header["NAXIS"], header["NAXIS1"], header["NAXIS2"] = 2, N_PIXELS, N_PIXELS
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRPIX1"] = header["CRPIX2"] = N_PIXELS / 2 + 0.5
    header["CRVAL1"], header["CRVAL2"] = crval
    header["CD1_1"], header["CD1_2"], header["CD2_1"], header["CD2_2"] = -0.11 / 3600, 0.0, 0.0, 0.11 / 3600
    return header


def gaussian_psf(sigma, size=31):
    y, x = np.mgrid[:size, :size] - (size - 1) / 2
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def star_image(header, stars, sigma, rng):
    wcs = WCS(header)
    y, x = np.mgrid[:N_PIXELS, :N_PIXELS]
    image = rng.normal(0.0, 1.0, (N_PIXELS, N_PIXELS))
    for ra, dec, flux in stars:
        px, py = wcs.all_world2pix(ra, dec, 0)
        image += flux * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return image


@pytest.fixture(scope="module")
def arguments():
    """The twelve positional arguments of the subtraction: two exposures of 60 stars, offset by a fraction of a pixel."""
    rng = np.random.default_rng(1)
    header_target, header_object = tan_header((150.0, 2.0)), tan_header((150.0 + 3e-5, 2.0 - 2e-5))
    stars = []
    for _ in range(60):
        px, py = rng.uniform(20, N_PIXELS - 20, 2)
        ra, dec = WCS(header_target).all_pix2world(px, py, 0)
        stars.append((float(ra), float(dec), rng.uniform(500, 5000)))
    image_target = star_image(header_target, stars, SIGMA_TARGET, rng)
    image_object = star_image(header_object, stars, SIGMA_OBJECT, rng)
    variance = np.ones((N_PIXELS, N_PIXELS))
    return (header_target, header_object, 1.0, 1.0, image_target, image_object, variance, variance.copy(), image_target > 5,
            image_object > 5, gaussian_psf(SIGMA_TARGET), gaussian_psf(SIGMA_OBJECT))


# ------------------------------------------------------------------------------------------------------------------------- engine choice
def test_constructor_keeps_the_release_signature_then_adds_engine():
    release = list(inspect.signature(LegacySpaceSFFT_Flow.__init__).parameters.values())[1:]
    wrapper = list(inspect.signature(SpaceSFFT.__init__).parameters.values())[1:]
    assert [(p.name, p.default) for p in wrapper[:len(release)]] == [(p.name, p.default) for p in release]
    assert [p.name for p in wrapper[len(release):]] == ["engine", "PSF_OVERSAMPLING", "PSF_PIXEL_INTEGRATED", "instrument"]
    assert inspect.signature(SpaceSFFT.__init__).parameters["engine"].default == "romanL2"


@pytest.mark.parametrize("given, expected", [("romanL2", "romanL2"), ("ROMANL2", "romanL2"), ("romanl2", "romanL2"),
                                             ("legacy", "legacy"), ("Legacy", "legacy"), (" LEGACY ", "legacy")])
def test_engine_in_any_letter_case(arguments, given, expected):
    sub = SpaceSFFT(*arguments, **CPU, engine=given)
    assert sub.engine == expected
    assert isinstance(sub.sfftengine, SpaceL2_Sub if expected == "romanL2" else LegacySpaceSFFT_Flow)


def test_default_engine_is_romanl2(arguments):
    sub = SpaceSFFT(*arguments, **CPU)
    assert sub.engine == "romanL2"
    assert isinstance(sub.sfftengine, SpaceL2_Sub)


def test_unknown_engine_raises(arguments):
    with pytest.raises(ValueError, match="engine must be one of"):
        SpaceSFFT(*arguments, **CPU, engine="roman")


@pytest.mark.parametrize("keyword, value", [("PSF_OVERSAMPLING", 2), ("PSF_PIXEL_INTEGRATED", False), ("instrument", "Roman-WFI")])
def test_legacy_refuses_romanl2_keywords(arguments, keyword, value):
    with pytest.raises(TypeError, match=keyword):
        SpaceSFFT(*arguments, **CPU, engine="legacy", **{keyword: value})


def test_romanl2_takes_its_keywords(arguments):
    sub = SpaceSFFT(*arguments, **CPU, engine="romanL2", PSF_OVERSAMPLING=1, PSF_PIXEL_INTEGRATED=True, instrument="Roman-WFI")
    assert sub.instrument == "Roman-WFI"


def test_legacy_refuses_fit_mask(arguments):
    sub = SpaceSFFT(*arguments, **CPU, engine="legacy")
    with pytest.raises(TypeError, match="fit_mask"):
        sub.sfft_subtract(fit_mask=np.ones((N_PIXELS, N_PIXELS), bool))


# --------------------------------------------------------------------------------------------------------------------- attribute forwarding
@pytest.mark.parametrize("engine", ["legacy", "romanL2"])
def test_attributes_are_read_from_the_engine(arguments, engine):
    sub = SpaceSFFT(*arguments, **CPU, engine=engine)
    assert sub.hdr_target is sub.sfftengine.hdr_target
    assert sub.op is sub.sfftengine.op


def test_setting_an_attribute_reaches_the_legacy_engine_without_warning(arguments):
    sub = SpaceSFFT(*arguments, **CPU, engine="legacy")
    sub.resample_image_mask_psf()
    replacement = sub.op.asarray(np.zeros((N_PIXELS, N_PIXELS)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub.PixA_resamp_object = replacement
        sub.BlankMask = replacement == 0.0
    assert sub.sfftengine.PixA_resamp_object is replacement
    assert "PixA_resamp_object" not in vars(sub)


@pytest.mark.parametrize("name", ["PixA_resamp_object", "BlankMask"])
def test_setting_an_attribute_romanl2_does_not_read_back_warns(arguments, name):
    sub = SpaceSFFT(*arguments, **CPU, engine="romanL2")
    sub.resample_image_mask_psf()
    replacement = np.zeros((N_PIXELS, N_PIXELS))
    with pytest.warns(UserWarning, match=f"does not read {name} back"):
        setattr(sub, name, replacement)
    assert getattr(sub.sfftengine, name) is replacement


@pytest.mark.parametrize("name", ["engine", "sfftengine"])
def test_own_attributes_cannot_be_changed(arguments, name):
    sub = SpaceSFFT(*arguments, **CPU, engine="legacy")
    with pytest.raises(AttributeError, match="cannot be changed"):
        setattr(sub, name, None)


# ---------------------------------------------------------------------------------------------------------------- phrosty's and sidecar's steps
@pytest.mark.parametrize("engine", ["legacy", "romanL2"])
def test_phrosty_sequence(arguments, engine):
    """phrosty/pipeline.py: resample, cross-convolve, subtract, decorrelation, variance, then the decorrelation applied to three images."""
    sub = SpaceSFFT(*arguments, **CPU, engine=engine)
    sub.resample_image_mask_psf()
    sub.cross_convolve()
    sub.sfft_subtract()
    sub.find_decorrelation()
    variance = sub.create_variance_image()
    assert np.shape(sub.op.asnumpy(variance)) == (N_PIXELS, N_PIXELS)
    for image in (sub.PixA_DIFF, sub.PixA_Ctarget, sub.PSF_Ctarget):
        decorrelated = sub.op.asnumpy(sub.apply_decorrelation(image))
        assert np.shape(decorrelated) == np.shape(sub.op.asnumpy(image))
    difference = sub.op.asnumpy(sub.apply_decorrelation(sub.PixA_DIFF))
    assert np.isfinite(difference).mean() > 0.5
    for name in PHROSTY_ATTRIBUTES:
        getattr(sub, name)


def _sidecar_default_sequence(sub):
    """sidecar/subtraction.py with its defaults (cross_convolve=False, resample_method='bilinear')."""
    sub.resample_image_mask_psf()
    sub.sfft_subtract()
    sub.find_decorrelation()
    sub.apply_decorrelation(sub.PixA_DIFF)
    sub.apply_decorrelation(sub.PSF_target)
    sub.create_score_image()
    sub.create_variance_image()
    for name in SIDECAR_ATTRIBUTES:
        getattr(sub, name)


def test_sidecar_default_sequence_legacy(arguments):
    _sidecar_default_sequence(SpaceSFFT(*arguments, **CPU, engine="legacy"))


@pytest.mark.xfail(raises=RuntimeError, strict=True,
                   reason="romanL2 makes a score image only after cross_convolve(), which sidecar's default skips (DECISIONS.md D1, O12)")
def test_sidecar_default_sequence_romanl2(arguments):
    _sidecar_default_sequence(SpaceSFFT(*arguments, **CPU, engine="romanL2"))
