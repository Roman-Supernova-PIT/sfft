"""Reading a Roman WFI Level 2 product (`_cal.asdf`): its image as a surface brightness in MJy/sr with the sky removed, its variance, its
unusable and saturated pixels, the pixel areas from its gWCS, and its gWCS as a TAN-SIP header of degree 5 (what the resampling reads).

A flat-fielded Level 2 holds a surface brightness already; a file of light per pixel is made one with its nominal over its actual pixel areas
(pixel_area). The sky is a smooth mesh (or a constant) measured on the pixels free of sources (SExtractor at 1.5 sigma, grown by 2 pixels).
Every data-quality flag except KEEP_FLAGS makes a pixel unusable; outlier_pixels adds pixels far below the sky or with an extreme error.
With add_dumo, the file's dumo array (the uniform-weight ramp fit minus the optimal-weight one) is added to the data except at JUMP_DET
pixels, and the noise it adds to the variance there: one constant per exposure, the rise of the scatter that adding dumo causes on the
exposure's source-free usable pixels (dumo_variance, in the file's unit)."""
import glob
import os
import warnings

import numpy as np
from astropy.wcs import WCS
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

from ..subtract.geometry import pixel_area_map

KEEP_FLAGS = ("WARM", "LOW_QE", "JUMP_DET", "PERSISTENCE")
SKY_MODELS = ("mesh", "constant")
REJECT_LOW_SIGMA = 10.0
REJECT_ERR_FACTOR = 100.0
SKY_MASK_THRESH = 1.5
SKY_MASK_GROW = 2
SKY_MASK_MIN_FREE = 0.02
REJECT_WARN_SHARE = 1e-3
ARCSEC2_PER_SR = 206264.80624709636 ** 2
ZP_MJY = 8.9 - 2.5 * np.log10(1.0e6)


def flux_conversion(meta):
    """MJy/sr per unit of data: 1 for a flux-calibrated file, else photometry.conversion_megajanskys (NaN when missing)."""
    return 1.0 if meta["flux_step"] == "COMPLETE" else meta["conversion_megajanskys"]


def nominal_area_sr(meta, area_sr):
    """The file's nominal pixel area in sr (photometry.pixel_area; the median of area_sr when it has none)."""
    v = meta["pixel_area_sr"]
    return float(v) if np.isfinite(v) and v > 0 else float(np.median(area_sr))


def data_is_surface_brightness(meta, pixel_area):
    """Whether a file's data is a surface brightness, by the option pixel_area: True, False, or 'auto' (a file that names a flat)."""
    return bool(meta["flat_reference"]) if pixel_area == "auto" else bool(pixel_area)


def bad_pixel_flags(keep=KEEP_FLAGS):
    """Bit mask of the data-quality flags that make a pixel unusable."""
    from roman_datamodels import dqflags
    bits = 0
    for flag in dqflags.pixel:
        if flag.name not in keep:
            bits |= int(flag.value)
    return bits


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
    """Reject a detector pixel in every exposure of its detector once outlier_pixels rejected it in one. Returns {detector: (exposures, rejected pixels)}."""
    groups = {}
    for L in levels:
        if getattr(L, "outliers", None) is not None:
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


def _get(node, path, default=None):
    cur = node
    try:
        for p in path.split("."):
            cur = getattr(cur, p)
        return cur
    except Exception:
        return default


def _positive(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


def read_meta(dm, path):
    """The metadata the workflow uses, from a datamodel."""
    m = dm.meta
    flat = str(_get(m, "ref_file.flat", "") or "")
    return dict(filename=str(_get(m, "filename", os.path.basename(path))),
                detector=str(_get(m, "instrument.detector", "WFI??")),
                optical_element=str(_get(m, "instrument.optical_element", "F???")),
                start_time=str(_get(m, "exposure.start_time", "")),
                exposure_time=_positive(_get(m, "exposure.exposure_time")),
                conversion_megajanskys=_positive(_get(m, "photometry.conversion_megajanskys")),
                pixel_area_sr=_positive(_get(m, "photometry.pixel_area")),
                flux_step=str(_get(m, "cal_step.flux", "?")),
                flat_reference=flat if flat.lower() not in ("", "n/a", "none", "?") else "",
                observation=_get(m, "observation.observation"), visit=_get(m, "observation.visit"), exposure=_get(m, "observation.exposure"))


class Scan:

    """A product's metadata, gWCS and shape, without reading its arrays."""
    def __init__(self, path):
        import roman_datamodels as rdm
        self.path = str(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._dm = rdm.open(self.path)
            self.meta = read_meta(self._dm, self.path)
            self.gwcs = self._dm.meta.wcs
            self.shape = tuple(int(n) for n in self._dm.data.shape)
            self.has_dumo = hasattr(self._dm, "dumo")


class Level2:

    """One exposure: image (MJy/sr, sky removed, 0 at unusable pixels), variance (infinite there), dnu (unusable pixels), saturated,
    source_mask, area_sr, header (TAN-SIP), wcs, and what was done (sky, rejected pixels, dumo and dumo_variance, units)."""
    @staticmethod
    def _load(path):
        """The arrays of a product (float64; dq uint32), its gWCS and metadata; dumo and var_poisson are None when the file has none."""
        import roman_datamodels as rdm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm = rdm.open(path)
            data, err = np.asarray(dm.data, np.float64), np.asarray(dm.err, np.float64)
            dq = np.asarray(dm.dq, np.uint32)
            dumo = np.asarray(dm.dumo, np.float64) if hasattr(dm, "dumo") else None
            var_poisson = np.asarray(dm.var_poisson, np.float64) if hasattr(dm, "var_poisson") else None
            return data, err, dq, dumo, var_poisson, dm.meta.wcs, read_meta(dm, path)

    def __init__(self, path, degree=5, keep_flags=KEEP_FLAGS, sky_model="mesh", add_dumo=True, pixel_area="auto", mesh_px=128,
                 wcs_offset_vector=None, reject_outliers=True, mask_sources=True):
        from roman_datamodels import dqflags
        if sky_model not in SKY_MODELS:
            raise ValueError(f"sky_model must be one of {SKY_MODELS}")
        if pixel_area not in (True, False, "auto"):
            raise ValueError("pixel_area must be True, False or 'auto'")
        self.path = str(path)
        self.sky_model = sky_model
        self.wcs_offset_vector = wcs_offset_vector
        data, err, dq, dumo, var_poisson, self.gwcs, self.meta = self._load(self.path)
        self.shape = data.shape
        self.header = sip_header(self.gwcs, self.shape, degree)
        self.sip_error_px = max(float(self.header.get("SIPMXERR", 0.0) or 0.0), float(self.header.get("SIPIVERR", 0.0) or 0.0))
        self.wcs = WCS(self.header)
        self.pixel_area_applied = data_is_surface_brightness(self.meta, pixel_area)
        self.area_sr = (pixel_area_map(self.gwcs, self.shape) / ARCSEC2_PER_SR).astype(np.float32)
        self.nominal_area_sr = nominal_area_sr(self.meta, self.area_sr)
        self.has_dumo = dumo is not None
        self.dumo_added = False
        self.dumo_variance = 0.0
        added = None
        if add_dumo:
            if dumo is None:
                warnings.warn(f"{self.meta['filename']}: add_dumo is on but the file has no dumo array", stacklevel=2)
            elif dumo.shape != data.shape:
                warnings.warn(f"{self.meta['filename']}: dumo has shape {dumo.shape}, data {data.shape}: not added", stacklevel=2)
            else:
                jump = (dq & np.uint32(dqflags.pixel["JUMP_DET"].value)) != 0
                added = np.isfinite(dumo) & ~jump
                data = np.where(added, data + dumo, data)
                self.dumo_added = True
        self.dnu = ((dq & np.uint32(bad_pixel_flags(keep_flags))) != 0) | ~np.isfinite(data) | ~np.isfinite(err)
        self.saturated = (dq & np.uint32(dqflags.pixel["SATURATED"].value)) != 0
        good = ~self.dnu
        if good.sum() < 100:
            raise ValueError(f"{self.meta['filename']}: only {int(good.sum())} usable pixels")
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
                warnings.warn(f"{self.meta['filename']}: the detected sources cover {share * 100:.1f} % of the usable pixels; the sky is measured "
                              "without masking them", stacklevel=2)
        if sky_model == "mesh":
            background = background_mesh(data, clean, box=mesh_px, min_share=0.2 if clean is good else 0.05)
            self.sky_file = float(np.median(background))
        else:
            self.sky_file, _ = robust_sky(data, clean)
            background = self.sky_file
        data = data - background
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
                warnings.warn(f"{self.meta['filename']}: {share * 100:.2f} % of the pixels rejected as outliers ({self.rejected_low} below the sky by "
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
        conv = flux_conversion(self.meta)
        known = bool(np.isfinite(conv) and conv > 0)
        self.conversion = float(conv) if known else 1.0
        self.unit = "MJy/sr" if known else "file unit"
        if not known:
            warnings.warn(f"{self.meta['filename']}: the metadata hold no flux conversion: its images stay in the file's unit", stacklevel=2)
        factor = self.conversion if self.pixel_area_applied else self.conversion * self.nominal_area_sr / self.area_sr.astype(np.float64)
        sb, err = data * factor, err * factor
        self.sky = float(self.sky_file * np.median(factor))
        _, self.sky_rms = robust_sky(sb, good)
        self.image = np.where(good, sb, 0.0)
        self.variance = np.where(good, err ** 2, np.inf)
        self.median_err = float(np.median(err[good]))
        self.zp_light = float(ZP_MJY - 2.5 * np.log10(self.nominal_area_sr)) if known else 27.0

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
        return (f"Level2({self.meta['filename']}: {self.meta['detector']} {self.meta['optical_element']}, sky {self.sky:.4g} {self.unit} ({self.sky_model}, {masked}), "
                f"rms {self.sky_rms:.4g}, "
                f"{self.dnu.mean() * 100:.2f}% do-not-use"
                + (f" ({self.rejected_low} below the sky, {self.rejected_noisy} noisy: outliers; {self.rejected_shared} rejected in another exposure)"
                   if self.reject_outliers else " (outliers kept)")
                + (f", dumo added except at JUMP_DET (the noise it adds, variance {self.dumo_variance:.3g}, added to err^2), " if self.dumo_added else ", dumo not added, ")
                + f"{'surface brightness' if self.pixel_area_applied else 'light per pixel'} in the file, conversion {self.conversion:.5g})")
