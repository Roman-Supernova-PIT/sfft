"""SpaceL2_Prep: the inputs of SpaceL2_Sub from two Level 2 exposures (ASDF files), and the fit mask of their matched pair.

    from sfft.spaceflow.prep import SpaceL2_Prep
    from sfft.spaceflow.subtract import SpaceL2_Sub

    prep = SpaceL2_Prep(science_file, reference_file, instrument="Roman-WFI", cache_dir=...)
    sub = SpaceL2_Sub(**prep.sub_arguments(), BACKEND_4SUBTRACT="Cupy")
    sub.resample_image_mask_psf()
    sub.cross_convolve()
    sub.sfft_subtract(fit_mask=prep.fit_mask(sub))
    sub.find_decorrelation()
    difference = sub.apply_decorrelation(sub.PixA_DIFF)
    score, variance = sub.create_score_image(), sub.create_variance_image()

sfft.spaceflow.SpaceL2_Flow runs these steps in this order and writes the products.

Stage 1 (the constructor and sub_arguments) works before the subtraction: each exposure is read, its flux converted to MJy/sr, its sky
removed, its unusable pixels marked (as NaN in the arrays handed over), and the PSF source evaluated at the PSF nodes of the target grid in each
exposure's detector frame. Stage 2 (fit_mask) works on the matched pair, which exists only after the subtraction's cross_convolve().
"""
import logging
import os
import warnings

import numpy as np

_logger = logging.getLogger(__name__)

PSF_DEFAULT = dict(source="stpsf", n_grid=3, oversample=5, half=32, charge_diffusion="openuniverse", sed=None, nlambda=None, nproc=4)
SFFT_KERNEL = dict(GKerHW=2, KerPolyOrder=2, BGPolyOrder=0)


class SpaceL2_Prep:
    """The inputs of SpaceL2_Sub for one science exposure and one reference exposure (Level 2 files; for Roman WFI, `_cal.asdf`).

    Parameters
    ----------
    science_file, reference_file : str
        The two Level 2 files, in one filter. The science exposure's own pixel grid (its TAN-SIP header) is the grid of every product.
    instrument : str, default 'Roman-WFI'
        The instrument that took both exposures, one of sfft.spaceflow.instruments.INSTRUMENTS. It decides how the files are read and where
        the PSF comes from; Roman WFI is the only instrument so far (its reader, sfft.spaceflow.prep.reader, and its PSF sources,
        sfft.spaceflow.prep.psfsource), and the defaults below are its own.
    cache_dir : str, optional
        Where the PSF source keeps its evaluations (default: ~/.cache/sfft_romanl2).
    psf : dict, optional
        The PSF source. dict(source='stpsf', n_grid=3, oversample=5, half=32, charge_diffusion='openuniverse', sed=None, nlambda=None, nproc=4)
        (the default): STPSF's WFI model for the file's detector and filter at n_grid x n_grid detector positions, interpolated bilinearly, with
        charge diffusion and without the pixel. dict(source='array', stamp=<2-D array>, oversample=<samples per pixel>, pixel_integrated=True):
        one PSF for every exposure and position.
    crop_px, crop_centre : optional
        Work on the crop_px x crop_px part of the science exposure around crop_centre ((x, y), 0-based; None: its centre).
    keep_flags : sequence of str
        Data-quality flags whose pixels stay usable; any other flag makes a pixel unusable.
    sky_model, mesh_px : 'mesh' (a smooth background in cells of mesh_px) or 'constant'; measured on the pixels free of sources.
    add_dumo : bool
        Add the file's dumo array to its data, except at JUMP_DET pixels, and the noise it adds to the variance there (one constant per
        exposure, the rise of the scatter it causes on the exposure's source-free pixels).
    pixel_area : True, False or 'auto'
        Whether the data is a surface brightness (a flat-fielded Level 2) or the light per pixel; 'auto' takes a file that names a flat for
        the first kind.
    reject_outliers : bool
        Make unusable the pixels far below the sky or with an extreme error, in both exposures when they share a detector.
    wcs_position_type : 'chief-ray', 'first-moment-centroid' or 'gaussian-window-centroid'
        What the files' WCS places at a star's position: the chief ray (the PSF model's own centre), or a centroid of the given type; for a
        centroid type each PSF stamp is moved by the exposure's WCS offset vector before it is handed over. wcs_gaussian_window_r50: the
        Gaussian window's half-light radius [native pixels] for 'gaussian-window-centroid' (None: the profile's own).
    nodes : int
        PSF nodes per axis over the target grid.

    Attributes
    ----------
    instrument : the instrument's name, as in INSTRUMENTS.
    science, reference : the two exposures as read (units, sky, unusable pixels, saturation flags, source masks).
    header, shape : the target grid.
    saturation_map : the pixels either exposure flags as saturated, on the target grid ([y, x], grown by 1 pixel).
    units : the unit, the grid's pixel area and the AB zero point of a sum over a product image.
    mask_record : what the last fit_mask() did.
    """

    def __init__(self, science_file, reference_file, instrument="Roman-WFI", cache_dir=None, psf=None, crop_px=None, crop_centre=None,
                 keep_flags=("WARM", "LOW_QE", "JUMP_DET", "PERSISTENCE"), sky_model="mesh", mesh_px=128, add_dumo=True, pixel_area="auto",
                 reject_outliers=True, wcs_position_type="chief-ray", wcs_gaussian_window_r50=None, nodes=5):
        from ..instruments import instrument_name
        self.instrument = instrument_name(instrument)
        # Roman WFI is the only instrument so far: the reader and the PSF sources below are its own
        from . import grid, psfsource
        from .reader import Level2, Scan, data_is_surface_brightness, share_outliers
        self.nodes = int(nodes)
        self.psf_config = dict(PSF_DEFAULT, **(psf or {}))
        if self.psf_config["source"] not in psfsource.PSF_SOURCES:
            raise ValueError(f"psf source must be one of {psfsource.PSF_SOURCES}, not {self.psf_config['source']!r}")
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "sfft_romanl2")
        files = [str(reference_file), str(science_file)]
        scans = [Scan(f) for f in files]
        filters = sorted({s.meta["optical_element"] for s in scans})
        if len(filters) > 1:
            raise ValueError(f"the two exposures are in different filters, {filters}: difference one filter at a time")
        self.psf_source = psfsource.make(self.psf_config, cache_dir, log=_logger.info)
        self.psf_source.prepare([(s.meta["detector"], s.meta["optical_element"]) for s in scans])
        offsets = {s.path: psfsource.wcs_offset_vector(s, self.psf_source, wcs_position_type, data_is_surface_brightness(s.meta, pixel_area),
                                                       wcs_gaussian_window_r50) for s in scans}
        options = dict(keep_flags=tuple(keep_flags), sky_model=sky_model, add_dumo=add_dumo, pixel_area=pixel_area, mesh_px=mesh_px,
                       reject_outliers=reject_outliers)
        self.reference = Level2(files[0], **options, wcs_offset_vector=offsets[files[0]])
        self.science = Level2(files[1], **options, wcs_offset_vector=offsets[files[1]])
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
            warnings.warn(f"the TAN-SIP headers follow the files' WCS only to {worst_sip:.3f} px", stacklevel=2)
        self.max_wcs_offset_px = 0.0
        self.mask_record = None

    def _node_psfs(self, level2):
        """The PSF source at the PSF nodes of the target grid, in this exposure's detector frame and centred where its WCS places a point
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

    def sub_arguments(self):
        """The arguments of SpaceL2_Sub for this pair, as a dict: the two headers, the images (MJy/sr, sky removed, NaN at unusable pixels)
        and their variances, the two exposures' source masks as detection masks, the PSFs at the nodes with their sampling, SFFT's kernel
        settings (SFFT_KERNEL) and the instrument. Arrays are [y, x] (transpose=True, the default)."""
        S, R, hdr = self.science, self.reference, self.header
        ny, nx = self.shape
        ix = int(round(float(S.header["CRPIX1"]) - float(hdr["CRPIX1"])))
        iy = int(round(float(S.header["CRPIX2"]) - float(hdr["CRPIX2"])))
        cut = (slice(iy, iy + ny), slice(ix, ix + nx))
        psf_t, os_t, pix_t = self._node_psfs(S)
        psf_o, os_o, pix_o = self._node_psfs(R)
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
