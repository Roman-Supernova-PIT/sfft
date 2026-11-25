"""The single-exposure grid (the science exposure's own pixels, or a crop of them), the saturation map on it, the rotation between the two
exposures' pixel grids, and the record of the images' units."""
import numpy as np

from ..subtract import geometry
from ..subtract.resample import ARCSEC2_PER_SR, target_area_sr
from .reader import ZP_MJY

ROTATION_WARNING_DEG = (0.02, 5.0)
UNITS_CONVENTION = "MJy/sr: Level 2 x conversion, x pixel area (sr), resampled, / target pixel area"


def science_exposure_grid(header, crop_px=None, crop_centre=None):
    """The science exposure's own TAN-SIP header, or the crop_px x crop_px part of it around crop_centre ((x, y), 0-based; None: the centre)."""
    return geometry.crop_header(header, crop_px, crop_centre) if crop_px else header.copy()


def single_rotation(sci, other):
    """The rotation between two exposures' pixel grids at their own centres, modulo 90 deg [deg], from forward WCS transforms only."""
    return float((geometry.axis_angle_deg(other.gwcs, other.shape) - geometry.axis_angle_deg(sci.gwcs, sci.shape) + 45.0) % 90.0 - 45.0)


def saturation_map(levels, hdr, shape):
    """Every exposure's SATURATED pixels on the grid of hdr ([y, x] bool of shape): each flagged pixel's centre through its WCS to the grid,
    rounded, grown by 1 px (resampling spreads a core over its neighbours). None when no exposure carries the map."""
    from astropy.wcs import WCS
    from scipy import ndimage
    maps = [getattr(L, "saturated", None) for L in levels]
    if all(m is None for m in maps):
        return None
    grid = WCS(hdr)
    out = np.zeros(shape, bool)
    for L, sat in zip(levels, maps):
        if sat is None or not sat.any():
            continue
        yy, xx = np.nonzero(sat)
        ra, dec = L.wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
        gx, gy = grid.world_to_pixel_values(ra, dec)
        ix, iy = np.rint(gx).astype(int), np.rint(gy).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
        out[iy[ok], ix[ok]] = True
    return ndimage.binary_dilation(out, iterations=1)


def units_record(refs, scis, hdr, warn=print):
    """The units of the images on the grid: the unit (MJy/sr unless a file lacks its flux conversion), the grid's mean pixel area, zp_sum (the
    AB zero point of a sum over an image: AB = zp_sum - 2.5 log10(sum)), and per side rate_to_sum, the sum over an image that a source of one
    DN/s in total in the files gives, with its spread over that side's exposures."""
    a_out = float(np.mean(target_area_sr(hdr)))
    unit = "MJy/sr" if all(L.unit == "MJy/sr" for L in refs + scis) else "file unit"
    rec = dict(convention=UNITS_CONVENTION, bunit=unit, out_area_sr=a_out, out_area_arcsec2=a_out * ARCSEC2_PER_SR,
               zp_sum=float(ZP_MJY - 2.5 * np.log10(a_out)) if unit == "MJy/sr" else 27.0)
    for name, group in (("reference", refs), ("science", scis)):
        k = np.array([L.conversion * L.nominal_area_sr / a_out for L in group])
        spread = float(k.max() / k.min() - 1.0)
        if spread > 1e-6:
            warn(f"the {name} exposures differ in calibration (conversion x nominal pixel area) by {spread:.2%}: each is converted with its own, "
                 "and a total in DN/s converts to a sum with their mean")
        rec[name] = dict(rate_to_sum=float(k.mean()), spread=spread, conversion=[L.conversion for L in group],
                         nominal_area_arcsec2=[L.nominal_area_sr * ARCSEC2_PER_SR for L in group])
    return rec
