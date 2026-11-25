"""Geometry of a WCS: local Jacobians, pixel-area maps, the angle of the pixel axes, and crops of a grid's header."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ARCSEC_PER_RAD = 206264.80624709636


def _sky(w, x, y):
    """RA, Dec [deg] of 0-based pixel positions, for a gWCS (callable) or an astropy WCS."""
    if hasattr(w, "all_pix2world"):
        ra, dec = w.all_pix2world(x, y, 0)
    else:
        ra, dec = w(x, y, with_bounding_box=False)
    return np.asarray(ra, float), np.asarray(dec, float)


def jacobians(w, x, y):
    """Local Jacobians at pixel positions (u west, v north, arcsec per pixel), by central differences over one pixel."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    xs = np.concatenate([x, x + 1, x - 1, x, x])
    ys = np.concatenate([y, y, y, y + 1, y - 1])
    ra, dec = _sky(w, xs, ys)
    ra = np.radians(ra).reshape(5, n)
    dec = np.radians(dec).reshape(5, n)
    cosdec = np.cos(dec[0])

    def dra(a, b):
        return (ra[a] - ra[b] + np.pi) % (2 * np.pi) - np.pi

    dudx = -0.5 * dra(1, 2) * cosdec * ARCSEC_PER_RAD
    dudy = -0.5 * dra(3, 4) * cosdec * ARCSEC_PER_RAD
    dvdx = 0.5 * (dec[1] - dec[2]) * ARCSEC_PER_RAD
    dvdy = 0.5 * (dec[3] - dec[4]) * ARCSEC_PER_RAD
    return dudx, dudy, dvdx, dvdy


def _pixel_area_spline(w, shape, step=64):
    """Bilinear interpolation [arcsec^2] of the pixel areas, from Jacobians on a coarse grid of the image."""
    from scipy.interpolate import RectBivariateSpline
    gx = np.arange(0, shape[1], step, dtype=float)
    gy = np.arange(0, shape[0], step, dtype=float)
    gx = np.append(gx, shape[1] - 1.0) if gx[-1] != shape[1] - 1 else gx
    gy = np.append(gy, shape[0] - 1.0) if gy[-1] != shape[0] - 1 else gy
    xx, yy = np.meshgrid(gx, gy)
    dudx, dudy, dvdx, dvdy = jacobians(w, xx.ravel(), yy.ravel())
    area = np.abs(dudx * dvdy - dudy * dvdx).reshape(len(gy), len(gx))
    return RectBivariateSpline(gy, gx, area, kx=1, ky=1, s=0)


def pixel_area_map(w, shape, step=64):
    """Sky area of every pixel [arcsec^2], from Jacobians on a coarse grid, bilinearly interpolated."""
    return _pixel_area_spline(w, shape, step)(np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float))


def pixel_area_function(w, shape, step=64):
    """area(x, y): the sky area [arcsec^2] of a virtual pixel centred at any detector position (0-based, fractional), the interpolation of
    pixel_area_map, held at its edge value beyond the detector."""
    spline = _pixel_area_spline(w, shape, step)

    def area(x, y):
        return spline.ev(np.asarray(y, float), np.asarray(x, float))

    return area


def axis_angle_deg(w, shape):
    """The angle of an image's pixel axes from north-up, modulo 90 deg, in [-45, 45)."""
    dudx, _, dvdx, _ = jacobians(w, np.array([(shape[1] - 1) / 2.0]), np.array([(shape[0] - 1) / 2.0]))
    a = float(np.degrees(np.arctan2(dvdx[0], dudx[0])))
    return (a + 45.0) % 90.0 - 45.0


def crop_header(hdr, crop_px, centre=None):
    """The crop_px x crop_px part of a grid around centre ((x, y), 0-based; None: the centre), as a header of its own (CRPIX moved by whole pixels)."""
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    c = int(crop_px)
    cx, cy = ((nx - 1) / 2.0, (ny - 1) / 2.0) if centre is None else centre
    x0 = int(round(cx - c / 2.0))
    y0 = int(round(cy - c / 2.0))
    x0 = min(max(x0, 0), max(nx - c, 0))
    y0 = min(max(y0, 0), max(ny - c, 0))
    out = hdr.copy()
    out["NAXIS1"], out["NAXIS2"] = min(c, nx), min(c, ny)
    out["CRPIX1"] = hdr["CRPIX1"] - x0
    out["CRPIX2"] = hdr["CRPIX2"] - y0
    return out
