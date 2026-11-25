"""SpaceL2_Prep: the inputs of SpaceL2_Sub from two Roman Space Telescope Level 2 exposures given as arrays, and the fit mask of their
matched pair.

    from sfft.spaceflow.prep import SpaceL2_Prep
    from sfft.spaceflow.subtract import SpaceL2_Sub

    prep = SpaceL2_Prep(PixA_target, PixA_object, PixA_targetVar, PixA_objectVar, wcs_target, wcs_object,
                        PixA_target_BADMASK=..., PixA_object_BADMASK=..., detector_target="WFI11", filter_target="F106", ...)
    sub = SpaceL2_Sub(**prep.sub_arguments(), BACKEND_4SUBTRACT="Cupy")
    sub.resample_image_mask_psf()
    sub.cross_convolve()
    sub.sfft_subtract(fit_mask=prep.fit_mask(sub))
    sub.find_decorrelation()
    difference = sub.apply_decorrelation(sub.PixA_DIFF)
    score, variance = sub.create_score_image(), sub.create_variance_image()

sfft.spaceflow.SpaceL2_Flow runs these steps in this order and writes the products. sfft reads no file: the caller reads both exposures (with
snappl, or as in examples/read_roman_l2.py) and passes their arrays and what their files hold as keywords.

Stage 1 (the constructor and sub_arguments) works before the subtraction: each exposure's flux converted to MJy/sr, its sky removed, its
unusable pixels marked (as NaN in the arrays handed over), and its PSF: the caller's (PSF_target, PSF_object), or STPSF's evaluated at the PSF
nodes of the target grid in the exposure's detector frame. Stage 2 (fit_mask) works on the matched pair, which exists only after the
subtraction's cross_convolve().
"""
import logging
import os
import warnings

import numpy as np

_logger = logging.getLogger(__name__)

PSF_DEFAULT = dict(source="stpsf", n_grid=3, oversample=5, half=32, charge_diffusion="openuniverse", sed=None, nlambda=None, nproc=4)
SFFT_KERNEL = dict(GKerHW=2, KerPolyOrder=2, BGPolyOrder=0)
SIDES = ("target", "object")


class SpaceL2_Prep:
    """The inputs of SpaceL2_Sub for one science exposure (the target) and one reference exposure (the object), given as arrays.

    Parameters (each per-exposure keyword exists for the target and for the object: PixA_target_BADMASK and PixA_object_BADMASK, ...)
    ----------
    PixA_target, PixA_object : 2-D arrays [y, x]
        Each exposure's data, in the input unit. The science exposure's own pixel grid (the target's TAN-SIP header) is the grid of every
        product.
    PixA_targetVar, PixA_objectVar : 2-D arrays
        Their variances (the square of the Level 2 err array).
    wcs_target, wcs_object : gWCS or astropy.io.fits.Header
        Each exposure's WCS: a gWCS (turned into a TAN-SIP header of degree 5), or a FITS header with NAXIS1 and NAXIS2 holding the WCS.
    PSF_target, PSF_object : 2-D or 4-D arrays, optional
        The caller's PSF of each exposure, handed to SpaceL2_Sub as it is: 2-D, one stamp for the whole exposure; 4-D [nodes, nodes, n, n],
        one stamp at each PSF node, the cell centres of a nodes x nodes grid over the target grid (after crop_px, over the crop), the object's
        at the detector position of the object that sees the same sky. The stamps are in each exposure's own detector frame, centred where its
        WCS places a star: sfft does not correct a caller's PSF for wcs_position_type, and warns. An exposure without one gets STPSF's, from
        its detector and filter.
    PSF_OVERSAMPLING, PSF_PIXEL_INTEGRATED : one value or a pair (target, object), defaults 1 and True
        Samples per native pixel of the caller's PSFs, and whether they include the pixel's response (as in SpaceL2_Sub).
    PixA_target_BADMASK, PixA_target_SATMASK, PixA_target_JUMPMASK : 2-D bool, optional
        Unusable, saturated and cosmic-ray jump pixels (where dumo is not added).
    PixA_target_DUMO, PixA_target_VARPOISSON : 2-D arrays, optional
        The uniform-weight ramp fit minus the optimal-weight one (added to the data with add_dumo), and the Poisson variance (not used).
    detector_target, filter_target : str, optional
        The detector and filter: needed for STPSF's PSF; two exposures of one detector share their outlier pixels; two filters given must match.
    conversion_megajanskys_target : float, optional
        MJy/sr per input unit (photometry.conversion_megajanskys; 1.0 for data in MJy/sr already).
    pixel_area_sr_target : float, optional
        The nominal pixel area [sr] (photometry.pixel_area); None: the median of the WCS's pixel-area map.
    is_surface_brightness_target : bool, default True
        Whether the data is a surface brightness already (a flat-fielded Level 2), or the light per pixel.
    name_target : str, optional
        The exposure's name in messages and records (default 'target', 'object').
    subtract_sky, convert_units : bool, default True
        False skips the sky subtraction, or the conversion to MJy/sr, of both exposures, for a caller that has done it.
    instrument : str, default 'Roman-WFI'
        The instrument that took both exposures, one of sfft.spaceflow.instruments.INSTRUMENTS (Roman WFI so far).
    cache_dir : str, optional
        Where STPSF's evaluations are kept (default: ~/.cache/sfft_romanl2).
    psf : dict, optional
        STPSF's settings: dict(source='stpsf', n_grid=3, oversample=5, half=32, charge_diffusion='openuniverse', sed=None, nlambda=None,
        nproc=4) (the default): STPSF's WFI model for the exposure's detector and filter at n_grid x n_grid detector positions, interpolated
        bilinearly, with charge diffusion and without the pixel.
    crop_px, crop_centre : optional
        Work on the crop_px x crop_px part of the science exposure around crop_centre ((x, y), 0-based; None: its centre).
    sky_model, mesh_px : 'mesh' (a smooth background in cells of mesh_px) or 'constant'; measured on the pixels free of sources.
    add_dumo : bool
        Add the dumo array to the data, except at jump pixels, and the noise it adds to the variance there (one constant per exposure, the
        rise of the scatter it causes on the exposure's source-free pixels).
    reject_outliers : bool
        Make unusable the pixels far below the sky or with an extreme error, in both exposures when they share a detector.
    wcs_position_type : 'chief-ray', 'first-moment-centroid' or 'gaussian-window-centroid'
        What the WCS places at a star's position: the chief ray (the PSF model's own centre), or a centroid of the given type; for a centroid
        type each STPSF stamp is moved by the exposure's WCS offset vector before it is handed over. wcs_gaussian_window_r50: the Gaussian
        window's half-light radius [native pixels] for 'gaussian-window-centroid' (None: the profile's own).
    nodes : int
        PSF nodes per axis over the target grid.

    A missing optional input gives one warning per exposure (sfft.spaceflow.prep.reader.missing_input_warnings).

    Attributes
    ----------
    instrument : the instrument's name, as in INSTRUMENTS.
    science, reference : the two exposures as prepared (units, sky, unusable pixels, saturation flags, source masks).
    psf_sources : {'target': 'caller' or 'stpsf', 'object': ...}, where each exposure's PSF comes from.
    header, shape : the target grid.
    saturation_map : the pixels either exposure flags as saturated, on the target grid ([y, x], grown by 1 pixel); None without a mask.
    units : the unit, the grid's pixel area and the AB zero point of a sum over a product image.
    mask_record : what the last fit_mask() did.
    """

    def __init__(self, PixA_target, PixA_object, PixA_targetVar, PixA_objectVar, wcs_target, wcs_object, *, PSF_target=None, PSF_object=None,
                 PSF_OVERSAMPLING=1, PSF_PIXEL_INTEGRATED=True, PixA_target_BADMASK=None, PixA_object_BADMASK=None, PixA_target_SATMASK=None,
                 PixA_object_SATMASK=None, PixA_target_JUMPMASK=None, PixA_object_JUMPMASK=None, PixA_target_DUMO=None, PixA_object_DUMO=None,
                 PixA_target_VARPOISSON=None, PixA_object_VARPOISSON=None, detector_target=None, detector_object=None, filter_target=None,
                 filter_object=None, conversion_megajanskys_target=None, conversion_megajanskys_object=None, pixel_area_sr_target=None,
                 pixel_area_sr_object=None, is_surface_brightness_target=True, is_surface_brightness_object=True, name_target=None,
                 name_object=None, subtract_sky=True, convert_units=True, instrument="Roman-WFI", cache_dir=None, psf=None, crop_px=None,
                 crop_centre=None, sky_model="mesh", mesh_px=128, add_dumo=True, reject_outliers=True, wcs_position_type="chief-ray",
                 wcs_gaussian_window_r50=None, nodes=5):
        from ..instruments import instrument_name
        from ..subtract.SpaceL2Sub import _per_side
        self.instrument = instrument_name(instrument)
        # Roman WFI is the only instrument so far: the exposure preparation and the PSF source below are its own
        from . import grid, psfsource, stpsf_roman
        from .reader import Level2, share_outliers
        self.nodes = int(nodes)
        stpsf_roman.normalize_wcs_position_type(wcs_position_type, what="wcs_position_type")
        self.psf_config = dict(PSF_DEFAULT, **(psf or {}))
        if self.psf_config["source"] not in psfsource.PSF_SOURCES:
            raise ValueError(f"psf source must be one of {psfsource.PSF_SOURCES}, not {self.psf_config['source']!r}: pass a PSF of your own "
                             "as PSF_target / PSF_object instead")
        exposures = dict(
            target=dict(image=PixA_target, variance=PixA_targetVar, wcs=wcs_target, psf=PSF_target, bad_mask=PixA_target_BADMASK,
                        saturated_mask=PixA_target_SATMASK, jump_mask=PixA_target_JUMPMASK, dumo=PixA_target_DUMO,
                        var_poisson=PixA_target_VARPOISSON, detector=detector_target, filter_name=filter_target,
                        conversion_megajanskys=conversion_megajanskys_target, pixel_area_sr=pixel_area_sr_target,
                        is_surface_brightness=is_surface_brightness_target, name=name_target),
            object=dict(image=PixA_object, variance=PixA_objectVar, wcs=wcs_object, psf=PSF_object, bad_mask=PixA_object_BADMASK,
                        saturated_mask=PixA_object_SATMASK, jump_mask=PixA_object_JUMPMASK, dumo=PixA_object_DUMO,
                        var_poisson=PixA_object_VARPOISSON, detector=detector_object, filter_name=filter_object,
                        conversion_megajanskys=conversion_megajanskys_object, pixel_area_sr=pixel_area_sr_object,
                        is_surface_brightness=is_surface_brightness_object, name=name_object))
        filters = sorted({str(e["filter_name"]) for e in exposures.values() if e["filter_name"] is not None})
        if len(filters) > 1:
            raise ValueError(f"the two exposures are in different filters, {filters}: difference one filter at a time")
        oversampling = _per_side(PSF_OVERSAMPLING, "PSF_OVERSAMPLING", "int")
        pixel_integrated = _per_side(PSF_PIXEL_INTEGRATED, "PSF_PIXEL_INTEGRATED", "bool")

        # each exposure's PSF: the caller's, handed over as it is, or STPSF's for its detector and filter
        self.psf_sources, self._caller_psfs = {}, {}
        for index, side in enumerate(SIDES):
            e = exposures[side]
            if e["psf"] is not None:
                self._caller_psfs[side] = (_caller_psf(e["psf"], f"PSF_{side}", self.nodes), oversampling[index], pixel_integrated[index])
                self.psf_sources[side] = "caller"
                warnings.warn(_caller_psf_warning(side, wcs_position_type), stacklevel=2)
            elif e["detector"] is None or e["filter_name"] is None:
                raise ValueError(f"no PSF for the {side} exposure: pass PSF_{side}, or detector_{side} and filter_{side} for STPSF's")
            else:
                self.psf_sources[side] = "stpsf"
        self.psf_source = None
        stpsf_sides = [side for side in SIDES if self.psf_sources[side] == "stpsf"]
        if stpsf_sides:
            cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "sfft_romanl2")
            self.psf_source = psfsource.make(self.psf_config, cache_dir, log=_logger.info)
            # STPSF's grids first: a problem with STPSF shows before the exposures are prepared
            self.psf_source.prepare([(exposures[side]["detector"], exposures[side]["filter_name"]) for side in stpsf_sides])

        options = dict(sky_model=sky_model, add_dumo=add_dumo, mesh_px=mesh_px, reject_outliers=reject_outliers, subtract_sky=subtract_sky,
                       convert_units=convert_units)
        prepared = {}
        for side in ("object", "target"):
            e = {key: value for key, value in exposures[side].items() if key != "psf"}
            prepared[side] = Level2(side=side, **e, **options)
        self.reference, self.science = prepared["object"], prepared["target"]
        for side in stpsf_sides:
            L = prepared[side]
            L.wcs_offset_vector = psfsource.wcs_offset_vector(L, self.psf_source, wcs_position_type, L.pixel_area_applied,
                                                              wcs_gaussian_window_r50)
        if reject_outliers:
            share_outliers([self.reference, self.science])
        self.header = grid.science_exposure_grid(self.science.header, crop_px, crop_centre)
        self.shape = (int(self.header["NAXIS2"]), int(self.header["NAXIS1"]))
        self.saturation_map = grid.saturation_map([self.reference, self.science], self.header, self.shape)
        self.units = grid.units_record([self.reference], [self.science], self.header, warn=lambda m: warnings.warn(m, stacklevel=3))
        self.rotation_deg = grid.single_rotation(self.science, self.reference)
        lo, hi = grid.ROTATION_WARNING_DEG
        if lo < abs(self.rotation_deg) < hi:
            warnings.warn(f"the two exposures' pixel grids differ by {abs(self.rotation_deg):.2f} deg: in this range the resampled reference's PSF "
                          "model can only average over a slow beat of the two pixel lattices, so stars and compact objects leave larger "
                          "residuals than at the same orientation or at a large rotation", stacklevel=2)
        worst_sip = max(self.reference.sip_error_px, self.science.sip_error_px)
        if worst_sip > 0.01:
            warnings.warn(f"the TAN-SIP headers follow the exposures' gWCS only to {worst_sip:.3f} px", stacklevel=2)
        self.max_wcs_offset_px = 0.0
        self.mask_record = None

    def _node_psfs(self, level2):
        """STPSF's PSF at the PSF nodes of the target grid, in this exposure's detector frame and centred where its WCS places a point
        source: ([ny, nx, n, n] with NaN where the exposure does not cover a node, samples per pixel, pixel included)."""
        from astropy.wcs import WCS
        from scipy import ndimage
        from ..subtract import crossconv as cc, psfmodel as pm
        w = WCS(self.header)
        xs, ys = cc.node_grid(self.shape, self.nodes)
        out, sample = None, None
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                ra, dec = w.all_pix2world(float(x), float(y), 0)
                xo, yo = pm._object_position(level2, float(ra), float(dec))
                if not (0 <= xo <= level2.shape[1] - 1 and 0 <= yo <= level2.shape[0] - 1):
                    continue
                rx, ry = pm.source_positions(level2, xo, yo)
                sample = self.psf_source(level2, rx, ry)
                stamp = np.asarray(sample.stamp, np.float64)
                if (xo - rx, yo - ry) != (0.0, 0.0):
                    stamp = ndimage.shift(stamp, (-(yo - ry) * sample.oversample, -(xo - rx) * sample.oversample), order=5, mode="constant")
                    self.max_wcs_offset_px = max(self.max_wcs_offset_px, abs(xo - rx), abs(yo - ry))
                if out is None:
                    out = np.full((len(ys), len(xs)) + stamp.shape, np.nan)
                out[j, i] = stamp
        if out is None:
            raise ValueError("no PSF node of the target grid falls on this exposure")
        return out, int(sample.oversample), bool(sample.pixel_integrated)

    def _psf(self, side, level2):
        """(PSF, samples per pixel, pixel included) of one exposure for SpaceL2_Sub: the caller's as given, or STPSF's at the nodes."""
        if side in self._caller_psfs:
            return self._caller_psfs[side]
        return self._node_psfs(level2)

    def sub_arguments(self):
        """The arguments of SpaceL2_Sub for this pair, as a dict: the two headers, the images (MJy/sr, sky removed, NaN at unusable pixels)
        and their variances, the two exposures' source masks as detection masks, the PSFs with their sampling, SFFT's kernel settings
        (SFFT_KERNEL) and the instrument. Arrays are [y, x] (transpose=True, the default)."""
        S, R, hdr = self.science, self.reference, self.header
        ny, nx = self.shape
        ix = int(round(float(S.header["CRPIX1"]) - float(hdr["CRPIX1"])))
        iy = int(round(float(S.header["CRPIX2"]) - float(hdr["CRPIX2"])))
        cut = (slice(iy, iy + ny), slice(ix, ix + nx))
        psf_t, os_t, pix_t = self._psf("target", S)
        psf_o, os_o, pix_o = self._psf("object", R)
        dmask_t = np.ones(S.shape, bool) if S.source_mask is None else S.source_mask
        dmask_o = np.ones(R.shape, bool) if R.source_mask is None else R.source_mask
        return dict(hdr_target=hdr, hdr_object=R.header, target_skyrms=S.sky_rms, object_skyrms=R.sky_rms,
                    PixA_target=np.where(S.dnu, np.nan, S.image)[cut], PixA_object=np.where(R.dnu, np.nan, R.image),
                    PixA_targetVar=np.where(S.dnu, np.nan, S.variance)[cut], PixA_objectVar=np.where(R.dnu, np.nan, R.variance),
                    PixA_target_DMASK=dmask_t[cut], PixA_object_DMASK=dmask_o, PSF_target=psf_t, PSF_object=psf_o,
                    PSF_OVERSAMPLING=(os_t, os_o), PSF_PIXEL_INTEGRATED=(pix_t, pix_o), **SFFT_KERNEL, instrument=self.instrument)

    def fit_mask(self, sub, **options):
        """The fit mask of a SpaceL2_Sub's matched pair (after sub.cross_convolve()), with this pair's saturation map: SFFT's sparse-flavour
        selection with the saturated-star guard and the PSF star test (flowmask.fit_mask_from_flow, whose keyword options it takes).
        Returns the mask for sub.sfft_subtract(fit_mask=...); its record is in mask_record."""
        from .flowmask import fit_mask_from_flow
        fit, self.mask_record = fit_mask_from_flow(sub, saturation_map=self.saturation_map, **options)
        return fit


def _caller_psf(psf, what, nodes):
    """A caller's PSF as float64: 2-D [n, n], or 4-D [nodes, nodes, n, n] with one stamp per PSF node."""
    a = np.array(np.asarray(psf), dtype=np.float64)
    if a.ndim == 4 and a.shape[:2] != (nodes, nodes):
        raise ValueError(f"{what} gives stamps at {a.shape[0]} x {a.shape[1]} nodes, but the PSF nodes are {nodes} x {nodes} (the option nodes): "
                         "give one stamp per node, at the cell centres of a nodes x nodes grid over the target grid (over the crop with crop_px)")
    if a.ndim not in (2, 4):
        raise ValueError(f"{what} must be a 2-D PSF stamp or a 4-D array [nodes, nodes, n, n] of stamps at the PSF nodes, not {a.ndim}-D")
    return a


def _caller_psf_warning(side, wcs_position_type):
    """The warning for a caller's PSF: sfft does not correct it for where the WCS places a star."""
    text = (f"PSF_{side} was passed directly, not evaluated from STPSF by sfft: sfft does not correct it for where the exposure's WCS places a "
            "star (a WCS fitted to star centroids places it off the PSF model's own centre), so it must already be centred where wcs_"
            f"{side} places a star; make that correction outside sfft if it is needed")
    if wcs_position_type != "chief-ray":
        text += f". wcs_position_type={wcs_position_type!r} is applied only to PSFs evaluated from STPSF, not to this one"
    return text
