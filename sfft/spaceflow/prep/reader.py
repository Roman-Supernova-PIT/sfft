"""One Roman WFI Level 2 exposure prepared for the subtraction, from arrays: its image as a surface brightness in MJy/sr with the sky
removed, its variance, its unusable and saturated pixels, the pixel areas from its WCS, and its WCS as a TAN-SIP header (what the resampling
reads). sfft reads no file: the caller reads the exposure (with snappl, or as in examples/read_roman_l2.py) and passes its arrays.

The WCS is a gWCS (turned into a TAN-SIP header of degree 5) or a FITS header with its WCS (used as it is). A flat-fielded Level 2 holds a
surface brightness already; an image of light per pixel (is_surface_brightness=False) is made one with its nominal over its actual pixel
areas. The sky is a smooth mesh (or a constant) measured on the pixels free of sources (SExtractor at 1.5 sigma, grown by 2 pixels). The
caller's bad-pixel mask, and pixels whose value or variance is not finite, are unusable; outlier_pixels adds pixels far below the sky or with
an extreme error. With add_dumo, the exposure's dumo array (the uniform-weight ramp fit minus the optimal-weight one) is added to the image
except at the caller's cosmic-ray jump pixels, and the noise it adds to the variance there: one constant per exposure, the rise of the
scatter that adding dumo causes on the exposure's source-free usable pixels (dumo_variance, in the input unit).

Each optional input that is not passed gives one warning (missing_input_warnings); what is done without it is in Level2's docstring."""
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from ..subtract.geometry import pixel_area_map

SKY_MODELS = ("mesh", "constant")
REJECT_LOW_SIGMA = 10.0
REJECT_ERR_FACTOR = 100.0
SKY_MASK_THRESH = 1.5
SKY_MASK_GROW = 2
SKY_MASK_MIN_FREE = 0.02
REJECT_WARN_SHARE = 1e-3
ARCSEC2_PER_SR = 206264.80624709636 ** 2
ZP_MJY = 8.9 - 2.5 * np.log10(1.0e6)
INPUT_UNIT = "input unit"      # the unit of the images when they are not converted to MJy/sr


def sip_header(gw, shape, degree=5):
    """TAN-SIP FITS header of a gWCS, with NAXIS keywords."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdr = gw.to_fits_sip(degree=degree, inv_degree=degree)
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(shape[1])
    hdr["NAXIS2"] = int(shape[0])
    return hdr


def robust_sky(image, good, step=7):
    """Median and 1.4826 MAD of the usable pixels (subsampled)."""
    s = image[::step, ::step][good[::step, ::step]]
    sky = float(np.median(s))
    return sky, float(1.4826 * np.median(np.abs(s - sky)))


def outlier_pixels(resid, err, good, low_sigma=REJECT_LOW_SIGMA, err_factor=REJECT_ERR_FACTOR):
    """Usable pixels that cannot hold a measurement: (low, noisy), more than low_sigma sky sigma below the sky, or with err above
    err_factor times the median err."""
    _, sigma = robust_sky(resid, good)
    low = good & (resid < -low_sigma * sigma)
    noisy = good & (err > err_factor * float(np.median(err[good])))
    return low, noisy


def share_outliers(levels):
    """Reject a detector pixel in every exposure of its detector once outlier_pixels rejected it in one. An exposure without a detector
    name shares nothing. Returns {detector: (exposures, rejected pixels)}."""
    groups = {}
    for L in levels:
        if getattr(L, "outliers", None) is not None and L.meta["detector"] is not None:
            groups.setdefault((L.meta["detector"], L.shape), []).append(L)
    out = {}
    for (det, shape), group in groups.items():
        union = np.zeros(shape, bool)
        for L in group:
            union |= L.outliers
        for L in group:
            L.rejected_shared = L.reject(union)
        out[det] = (len(group), int(union.sum()))
    return out


def source_mask(data, good, thresh=SKY_MASK_THRESH, grow=SKY_MASK_GROW, minarea=5):
    """The pixels of an exposure's sources, for its sky: SExtractor on data with its sky, grown by grow pixels; None, with a warning, when SExtractor fails."""
    from sfft.spaceflow.prep.astromatic.bundled import BundledSExtractorError
    from .fitmask import detect
    try:
        _, seg = detect(np.where(good, data, np.nan), thresh=thresh, minarea=minarea, mask=~good)
    except BundledSExtractorError:
        raise
    except Exception as e:
        warnings.warn(f"source mask for the sky: SExtractor failed ({type(e).__name__}: {e}); the sky is measured without it", stacklevel=2)
        return None
    objects = np.asarray(seg) > 0
    if grow:
        objects = ndimage.binary_dilation(objects, structure=np.ones((3, 3), bool), iterations=int(grow))
    return objects


def background_mesh(image, good, box=128, filter_size=3, nsig=3.0, iters=3, step=2, min_share=0.2):
    """A smooth background: clipped medians in cells of about box pixels, cells with few usable pixels filled from the nearest,
    median-filtered over filter_size cells, and interpolated with a spline through the cell centres."""
    ny, nx = image.shape
    ey = np.linspace(0, ny, max(1, int(round(ny / box))) + 1).astype(int)
    ex = np.linspace(0, nx, max(1, int(round(nx / box))) + 1).astype(int)
    med = np.full((len(ey) - 1, len(ex) - 1), np.nan)
    for j in range(len(ey) - 1):
        for i in range(len(ex) - 1):
            sl = (slice(ey[j], ey[j + 1], step), slice(ex[i], ex[i + 1], step))
            v = image[sl][good[sl]]
            if v.size < min_share * image[sl].size or v.size < 10:
                continue
            for _ in range(iters):
                m = np.median(v)
                s = 1.4826 * np.median(np.abs(v - m))
                if s <= 0:
                    break
                v = v[np.abs(v - m) < nsig * s]
            med[j, i] = np.median(v)
    missing = ~np.isfinite(med)
    if missing.all():
        raise ValueError("no good pixels to estimate a background from")
    if missing.any():
        idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        med = med[tuple(idx)]
    if filter_size > 1:
        med = ndimage.median_filter(med, size=filter_size, mode="nearest")
    if med.size == 1:
        return np.full((ny, nx), float(med[0, 0]))
    yc, xc = 0.5 * (ey[:-1] + ey[1:]) - 0.5, 0.5 * (ex[:-1] + ex[1:]) - 0.5
    if med.shape[0] == 1 or med.shape[1] == 1:
        med = np.tile(med, (2, 1)) if med.shape[0] == 1 else np.tile(med, (1, 2))
        yc = np.array([0.0, ny - 1.0]) if len(yc) == 1 else yc
        xc = np.array([0.0, nx - 1.0]) if len(xc) == 1 else xc
    spline = RectBivariateSpline(yc, xc, med, kx=min(3, len(yc) - 1), ky=min(3, len(xc) - 1), s=0)
    return spline(np.arange(ny, dtype=float), np.arange(nx, dtype=float))


def is_gwcs(wcs):
    """Whether an exposure's WCS is a gWCS (it can write a TAN-SIP header), rather than a FITS header."""
    return hasattr(wcs, "to_fits_sip")


def exposure_wcs(wcs, shape, degree=5, what="wcs"):
    """(native_wcs, header, sip_error_px) of an exposure's WCS: for a gWCS, the gWCS itself, its TAN-SIP header of degree `degree` and the
    largest error of that header against the gWCS [pixels]; for a FITS header, the astropy WCS it holds, a copy of it and 0. The header
    must give NAXIS1 and NAXIS2, equal to the image's shape."""
    ny, nx = int(shape[0]), int(shape[1])
    if is_gwcs(wcs):
        header = sip_header(wcs, shape, degree)
        sip_error = max(float(header.get("SIPMXERR", 0.0) or 0.0), float(header.get("SIPIVERR", 0.0) or 0.0))
        return wcs, header, sip_error
    if not isinstance(wcs, fits.Header):
        raise TypeError(f"{what} must be a gWCS or an astropy.io.fits.Header, not {type(wcs).__name__}")
    if "NAXIS1" not in wcs or "NAXIS2" not in wcs:
        raise ValueError(f"{what}: the FITS header must give NAXIS1 and NAXIS2 (the image is {nx} x {ny} pixels, NAXIS1 x NAXIS2)")
    if (int(wcs["NAXIS1"]), int(wcs["NAXIS2"])) != (nx, ny):
        raise ValueError(f"{what}: the header gives NAXIS1 x NAXIS2 = {wcs['NAXIS1']} x {wcs['NAXIS2']}, but the image is {nx} x {ny}")
    header = wcs.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = WCS(header)
    return native, header, 0.0


def _float_image(a, what, shape=None):
    """A 2-D input array as float64 (a copy), of the given shape when shape is not None."""
    out = np.array(np.asarray(a), dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{what} must be a 2-D array [y, x], not {out.ndim}-D")
    if shape is not None and out.shape != tuple(shape):
        raise ValueError(f"{what} has shape {out.shape}, but the image has {tuple(shape)}")
    return out


def _bool_mask(a, what, shape):
    """A 2-D input mask as bool, of the image's shape; None stays None."""
    if a is None:
        return None
    out = np.asarray(a).astype(bool)
    if out.shape != tuple(shape):
        raise ValueError(f"{what} has shape {out.shape}, but the image has {tuple(shape)}")
    return out


def _positive_number(value, what):
    """A positive finite number given as an input; None stays None."""
    if value is None:
        return None
    v = float(value)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{what} must be a positive finite number, not {value!r}")
    return v


def missing_input_warnings(side, name, passed, convert_units=True):
    """One warning for each optional input of an exposure that was not passed, saying what is done without it. side: 'target' or
    'object' (the keywords' names); name: the exposure's name in messages; passed: {input: whether it was passed} for BADMASK, SATMASK,
    JUMPMASK, DUMO, VARPOISSON and conversion_megajanskys."""
    def say(text):
        warnings.warn(f"{name}: {text}", stacklevel=4)

    if not passed["BADMASK"]:
        say(f"no bad-pixel mask (PixA_{side}_BADMASK) passed: only pixels whose value or variance is not finite are unusable")
    if not passed["SATMASK"]:
        say(f"no saturation mask (PixA_{side}_SATMASK) passed: the fit mask does not keep saturated stars out")
    if not passed["JUMPMASK"]:
        if passed["DUMO"]:
            say(f"no cosmic-ray jump mask (PixA_{side}_JUMPMASK) passed: dumo is added at every pixel where it is finite, cosmic-ray "
                "pixels included")
        else:
            say(f"no cosmic-ray jump mask (PixA_{side}_JUMPMASK) passed (it is only used to place dumo, which is not passed either)")
    if not passed["DUMO"]:
        say(f"no dumo array (PixA_{side}_DUMO) passed: nothing is added to the image, and no dumo noise to its variance")
    if not passed["VARPOISSON"]:
        say(f"no var_poisson array (PixA_{side}_VARPOISSON) passed (sfft does not use it)")
    if convert_units and not passed["conversion_megajanskys"]:
        say(f"no flux conversion (conversion_megajanskys_{side}) passed: the images stay in the input unit, not MJy/sr")


class Level2:

    """One exposure prepared for the subtraction, from its arrays: image (MJy/sr, sky removed, 0 at unusable pixels), variance (infinite
    there), dnu (unusable pixels), saturated (None when no saturation mask was passed), source_mask, area_sr, header (TAN-SIP), wcs (the
    astropy WCS of header), native_wcs (the WCS as passed: the gWCS, or the astropy WCS of the passed header), and what was done (sky,
    rejected pixels, dumo and dumo_variance, units).

    Inputs (side is 'target' or 'object', naming the keywords in messages):
      image, variance            2-D arrays [y, x]: the exposure's data and its variance, in the input unit
      wcs                        a gWCS, or a FITS header (with NAXIS1, NAXIS2) holding the WCS
      bad_mask                   unusable pixels; None: only pixels whose value or variance is not finite are unusable
      saturated_mask             saturated pixels; None: no saturation guard in the fit mask
      jump_mask                  cosmic-ray jump pixels, where dumo is not added; None: dumo is added wherever it is finite
      dumo                       the uniform-weight ramp fit minus the optimal-weight one; None: nothing is added
      var_poisson                accepted and not used
      detector, filter_name      the detector and filter (needed for STPSF, and to share outliers between exposures of one detector)
      conversion_megajanskys     MJy/sr per input unit; None: the images stay in the input unit
      pixel_area_sr              the nominal pixel area [sr]; None: the median of the pixel-area map from the WCS
      is_surface_brightness      whether the image is a surface brightness already (a flat-fielded Level 2), default True
      subtract_sky, convert_units  False skips the sky subtraction, or the conversion to MJy/sr (for a caller that has done it)
    A missing optional input gives one warning (missing_input_warnings)."""
    def __init__(self, image, variance, wcs, side="target", name=None, bad_mask=None, saturated_mask=None, jump_mask=None, dumo=None,
                 var_poisson=None, detector=None, filter_name=None, conversion_megajanskys=None, pixel_area_sr=None, is_surface_brightness=True,
                 degree=5, sky_model="mesh", add_dumo=True, mesh_px=128, wcs_offset_vector=None, reject_outliers=True, mask_sources=True,
                 subtract_sky=True, convert_units=True):
        if sky_model not in SKY_MODELS:
            raise ValueError(f"sky_model must be one of {SKY_MODELS}")
        if side not in ("target", "object"):
            raise ValueError(f"side must be 'target' or 'object', not {side!r}")
        self.side = side
        self.name = side if name is None else str(name)
        self.meta = dict(filename=self.name, detector=None if detector is None else str(detector),
                         optical_element=None if filter_name is None else str(filter_name))
        self.sky_model = sky_model
        self.wcs_offset_vector = wcs_offset_vector
        data = _float_image(image, f"PixA_{side}")
        shape = data.shape
        var = _float_image(variance, f"PixA_{side}Var", shape)
        bad = _bool_mask(bad_mask, f"PixA_{side}_BADMASK", shape)
        sat = _bool_mask(saturated_mask, f"PixA_{side}_SATMASK", shape)
        jump = _bool_mask(jump_mask, f"PixA_{side}_JUMPMASK", shape)
        dumo = None if dumo is None else _float_image(dumo, f"PixA_{side}_DUMO", shape)
        if var_poisson is not None:
            _float_image(var_poisson, f"PixA_{side}_VARPOISSON", shape)
        self.has_var_poisson = var_poisson is not None
        conversion = _positive_number(conversion_megajanskys, f"conversion_megajanskys_{side}")
        nominal = _positive_number(pixel_area_sr, f"pixel_area_sr_{side}")
        missing_input_warnings(side, self.name, dict(BADMASK=bad is not None, SATMASK=sat is not None, JUMPMASK=jump is not None,
                                                     DUMO=dumo is not None, VARPOISSON=var_poisson is not None,
                                                     conversion_megajanskys=conversion is not None), convert_units=convert_units)
        self.shape = shape
        self.native_wcs, self.header, self.sip_error_px = exposure_wcs(wcs, shape, degree, f"wcs_{side}")
        self.wcs = WCS(self.header)
        self.pixel_area_applied = bool(is_surface_brightness)
        self.area_sr = (pixel_area_map(self.native_wcs, shape) / ARCSEC2_PER_SR).astype(np.float32)
        self.nominal_area_sr = nominal if nominal is not None else float(np.median(self.area_sr))
        with np.errstate(invalid="ignore"):
            err = np.sqrt(var)                     # a negative variance gives NaN: the pixel is unusable
        self.has_dumo = dumo is not None
        self.dumo_added = False
        self.dumo_variance = 0.0
        added = None
        if add_dumo and dumo is not None:
            no_jump = np.ones(shape, bool) if jump is None else ~jump
            added = np.isfinite(dumo) & no_jump
            data = np.where(added, data + dumo, data)
            self.dumo_added = True
        self.dnu = ~np.isfinite(data) | ~np.isfinite(err)
        if bad is not None:
            self.dnu |= bad
        self.saturated = sat
        good = ~self.dnu
        if good.sum() < 100:
            raise ValueError(f"{self.name}: only {int(good.sum())} usable pixels")
        self.mask_sources = bool(mask_sources)
        self.sky_masked_share = None
        clean = good
        objects = source_mask(data, good) if self.mask_sources else None
        self.source_mask = objects
        if objects is not None:
            free = good & ~objects
            share = float(objects[good].mean())
            if free.sum() >= SKY_MASK_MIN_FREE * good.sum():
                clean, self.sky_masked_share = free, share
            else:
                warnings.warn(f"{self.name}: the detected sources cover {share * 100:.1f} % of the usable pixels; the sky is measured "
                              "without masking them", stacklevel=2)
        self.subtract_sky = bool(subtract_sky)
        if not self.subtract_sky:
            self.sky_file = 0.0
        elif sky_model == "mesh":
            background = background_mesh(data, clean, box=mesh_px, min_share=0.2 if clean is good else 0.05)
            self.sky_file = float(np.median(background))
            data = data - background
        else:
            self.sky_file, _ = robust_sky(data, clean)
            data = data - self.sky_file
        self.reject_outliers = bool(reject_outliers)
        self.rejected_low = self.rejected_noisy = self.rejected_shared = 0
        self.outliers = None
        if self.reject_outliers:
            low, noisy = outlier_pixels(data, err, good)
            self.rejected_low, self.rejected_noisy = int(low.sum()), int(noisy.sum())
            self.outliers = low | noisy
            self.dnu |= low | noisy
            good = ~self.dnu
            share = float((low | noisy).mean())
            if share > REJECT_WARN_SHARE:
                warnings.warn(f"{self.name}: {share * 100:.2f} % of the pixels rejected as outliers ({self.rejected_low} below the sky by "
                              f"more than {REJECT_LOW_SIGMA:g} sigma, {self.rejected_noisy} with err above {REJECT_ERR_FACTOR:g} x the median): see "
                              f"sfft.spaceflow.prep.reader.outlier_pixels", stacklevel=2)
        if added is not None:
            on = added & clean & good
            _, s_with = robust_sky(data, on)
            _, s_without = robust_sky(np.where(added, data - dumo, data), on)
            rise = float(s_with) ** 2 - float(s_without) ** 2
            self.dumo_variance = rise if np.isfinite(rise) and rise > 0 else 0.0
            if self.dumo_variance > 0:
                err = np.where(added, np.sqrt(err ** 2 + self.dumo_variance), err)
        self.convert_units = bool(convert_units)
        if self.convert_units and conversion is not None:
            self.conversion, self.unit = conversion, "MJy/sr"
        else:
            self.conversion, self.unit = 1.0, INPUT_UNIT
        if not self.convert_units or self.pixel_area_applied:
            factor = self.conversion
        else:
            factor = self.conversion * self.nominal_area_sr / self.area_sr.astype(np.float64)
        sb, err = data * factor, err * factor
        self.sky = float(self.sky_file * np.median(factor))
        _, self.sky_rms = robust_sky(sb, good)
        self.image = np.where(good, sb, 0.0)
        self.variance = np.where(good, err ** 2, np.inf)
        self.median_err = float(np.median(err[good]))
        self.zp_light = float(ZP_MJY - 2.5 * np.log10(self.nominal_area_sr)) if self.unit == "MJy/sr" else 27.0

    def reject(self, mask):
        """Make the pixels of mask unusable as well; returns how many of them were usable."""
        new = np.asarray(mask, bool) & ~self.dnu
        if new.any():
            self.dnu = self.dnu | new
            self.image = np.where(new, 0.0, self.image)
            self.variance = np.where(new, np.inf, self.variance)
        return int(new.sum())

    @property
    def light(self):
        """The light in each pixel, in units of the nominal pixel: image x pixel area / nominal pixel area."""
        return self.image * (self.area_sr / self.nominal_area_sr)

    @property
    def light_variance(self):
        """The variance of light."""
        return self.variance * (self.area_sr.astype(np.float64) / self.nominal_area_sr) ** 2

    def __repr__(self):
        masked = (f"sources masked over {self.sky_masked_share * 100:.1f} % of the pixels" if self.sky_masked_share is not None
                  else "sources not masked")
        sky = f"sky {self.sky:.4g} {self.unit} ({self.sky_model}, {masked})" if self.subtract_sky else "sky not subtracted"
        return (f"Level2({self.name}: {self.meta['detector']} {self.meta['optical_element']}, {sky}, "
                f"rms {self.sky_rms:.4g}, "
                f"{self.dnu.mean() * 100:.2f}% do-not-use"
                + (f" ({self.rejected_low} below the sky, {self.rejected_noisy} noisy: outliers; {self.rejected_shared} rejected in another exposure)"
                   if self.reject_outliers else " (outliers kept)")
                + (f", dumo added except at jump pixels (the noise it adds, variance {self.dumo_variance:.3g}, added to err^2), "
                   if self.dumo_added else ", dumo not added, ")
                + f"{'surface brightness' if self.pixel_area_applied else 'light per pixel'} in the input, conversion {self.conversion:.5g})")
