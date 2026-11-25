"""STPSF's Roman WFI PSF prepared for the workflow (the recipe ROMAN_RECIPE), and the WCS offset vector.

The recipe requests the oversampled PSF stamp from STPSF, checks its header, and adds charge diffusion once, which gives the PSF without the
pixel as shares of the fine cells, with the stamp's own sum. For an exposure whose WCS was fitted to centroids of stars (first moment, or a
Gaussian-windowed centroid), measure_wcs_offset_vector gives c = eta - r: where the WCS places a source (eta) relative to the source position
the PSF model is centred on (r)."""
from __future__ import annotations

import copy
import logging
import math
import re
import threading
from dataclasses import dataclass

import numpy as np


WCS_POSITION_TYPES = ("first-moment-centroid", "gaussian-window-centroid", "chief-ray")


WCS_TYPE_HELP = ("'gaussian-window-centroid' or 'first-moment-centroid' for a WCS fitted to "
                 "centroids of stars (which of the two is the estimator of that fit), "
                 "'chief-ray' for a WCS that uses the reference chief-ray landing point, as "
                 "in a simulation; a WCS fitted by PSF fitting takes the type of the origin "
                 "of the fitted PSF model, 'chief-ray' for an STPSF model")


def normalize_wcs_position_type(wcs_position_type, *, allow_none=False,
                                what="wcs_position_type=...", stacklevel=3):
    """The declared WCS position type, checked: one of WCS_POSITION_TYPES. None is returned when allow_none; otherwise there is no default."""
    if wcs_position_type is None:
        if allow_none:
            return None
        raise ValueError(
            f"{what}: the WCS position type is required ({WCS_TYPE_HELP}).  There is no "
            "default: only whoever made the WCS can say which detector position it uses "
            "for a star")
    if not isinstance(wcs_position_type, str) or wcs_position_type not in WCS_POSITION_TYPES:
        raise ValueError(f"{what}: the WCS position type must be one of {WCS_POSITION_TYPES} "
                         f"({WCS_TYPE_HELP}), got {wcs_position_type!r}")
    return wcs_position_type


WINDOW_FWHM_TO_SIGMA = 2.3548


_WINDOW_ROUNDS_MAX = 50


_R50_BISECTIONS = 60


def _sampling_xy(value):
    """Positive finite samples per detector pixel, ordered (x, y); a scalar denotes square sampling."""
    a = np.asarray(value, float).ravel()
    if a.size == 1:
        a = np.repeat(a, 2)
    if a.size != 2 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"oversample must be positive finite scalar or (x, y), got {value!r}")
    return a


def _measured_array(stamp, what):
    """A finite 2-D array with a positive sum (the helpers' input check)."""
    E = np.asarray(stamp, float)
    if E.ndim != 2:
        raise ValueError(f"{what}: the PSF array must be 2-d, got shape {E.shape}")
    if not np.all(np.isfinite(E)):
        raise ValueError(f"{what}: the PSF array contains non-finite values (NaN or inf)")
    if not float(E.sum()) > 0:
        raise ValueError(f"{what}: the PSF array must have a positive sum")
    return E


def _block_sums(E, n):
    """Sliding n x n block sums of E: entry (j, i) is the sum of E[j:j + n, i:i + n]."""
    c = np.cumsum(np.cumsum(np.pad(E, ((1, 0), (1, 0))), axis=0), axis=1)
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def _epsf_samples(stamp, oversample, pixel_integrated, what):
    """(B, px, py): effective-PSF values and the positions of their columns and rows in native pixels from the exact array centre."""
    E = _measured_array(stamp, what)
    os_x, os_y = _sampling_xy(oversample)
    if pixel_integrated:
        B = E
    elif os_x == os_y and os_x == int(os_x):
        n = int(os_x)
        if n > min(E.shape):
            raise ValueError(f"{what}: the array holds no {n} x {n} block")
        B = _block_sums(E, n)
    else:
        B = effective_psf(E, oversample)
    ny, nx = B.shape
    px = (np.arange(nx) - (nx - 1) / 2.0) / os_x
    py = (np.arange(ny) - (ny - 1) / 2.0) / os_y
    return B, px, py


def _half_light_radius(B, px, py, c):
    """The radius about c (native pixels) that holds half of the sum of B, each value spread over its cell."""
    B = np.asarray(B, float)
    px, py = np.asarray(px, float), np.asarray(py, float)
    steps = [abs(float(p[1] - p[0])) for p in (px, py) if p.size > 1] or [1.0]
    dx = abs(float(px[1] - px[0])) if px.size > 1 else steps[0]
    dy = abs(float(py[1] - py[0])) if py.size > 1 else steps[0]
    X = px[None, :] - float(c[0])
    Y = py[:, None] - float(c[1])
    rho = np.hypot(X, Y)
    w = np.where(rho > 0, (np.abs(X) * dx + np.abs(Y) * dy) / np.where(rho > 0, rho, 1.0),
                 max(dx, dy))
    k = np.argsort(rho, axis=None, kind="stable")
    rho, w, v = rho.ravel()[k], w.ravel()[k], B.ravel()[k]
    cum = np.concatenate(([0.0], np.cumsum(v)))
    h = 0.5 * float(w.max())
    half = 0.5 * float(v.sum())

    def enclosed(r):
        i0 = int(np.searchsorted(rho, r - h, side="right"))
        i1 = int(np.searchsorted(rho, r + h, side="left"))
        ramp = np.clip((r - rho[i0:i1]) / w[i0:i1] + 0.5, 0.0, 1.0)
        return float(cum[i0]) + float(v[i0:i1] @ ramp)

    lo, hi = 0.0, float(rho[-1]) + 2.0 * h
    for _ in range(_R50_BISECTIONS):
        mid = 0.5 * (lo + hi)
        if enclosed(mid) < half:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _windowed_iteration(B, px, py, c, sigma, tol, maxiter, what):
    """SExtractor's windowed-centroid loop with a Gaussian window of width sigma: (centre, steps)."""
    x, y = float(c[0]), float(c[1])
    inv = 1.0 / (2.0 * sigma * sigma)
    for it in range(1, int(maxiter) + 1):
        dx, dy = px - x, py - y
        gx, gy = np.exp(-dx * dx * inv), np.exp(-dy * dy * inv)
        Bgx = B @ gx
        s = float(gy @ Bgx)
        if not s > 0:
            raise ValueError(f"{what}: the window at ({x:.3f}, {y:.3f}) native px holds "
                             "no positive weight")
        ex = 2.0 * float(gy @ (B @ (gx * dx))) / s
        ey = 2.0 * float((gy * dy) @ Bgx) / s
        x += ex
        y += ey
        if math.hypot(ex, ey) < tol:
            return np.array([x, y]), it
    raise RuntimeError(f"{what}: the windowed centroid did not converge in {maxiter} "
                       f"steps (last step {math.hypot(ex, ey):.2e} native px, tolerance "
                       f"{tol:g})")


def windowed_centroid(stamp, oversample, *, pixel_integrated=False, r50=None, tol=1e-4,
                      maxiter=500):
    """A Gaussian-window centroid relative to the exact array centre, in fine-array samples (x, y)."""
    what = "windowed_centroid"
    B, px, py = _epsf_samples(stamp, oversample, pixel_integrated, what)
    tol = float(tol)
    if not tol > 0:
        raise ValueError(f"{what}: tol must be positive, got {tol!r}")
    c = np.zeros(2)
    if r50 is not None:
        r = float(r50)
        if not r > 0:
            raise ValueError(f"{what}: r50 must be positive, got {r50!r}")
        c, _ = _windowed_iteration(B, px, py, c, 2.0 * r / WINDOW_FWHM_TO_SIGMA, tol,
                                   maxiter, what)
    else:
        for _ in range(_WINDOW_ROUNDS_MAX):
            r = _half_light_radius(B, px, py, c)
            new, _ = _windowed_iteration(B, px, py, c, 2.0 * r / WINDOW_FWHM_TO_SIGMA,
                                         tol, maxiter, what)
            moved = float(np.hypot(*(new - c)))
            c = new
            if moved < tol:
                break
        else:
            raise RuntimeError(f"{what}: the half-light radius and the centroid did not "
                               f"settle in {_WINDOW_ROUNDS_MAX} rounds")
    os_x, os_y = _sampling_xy(oversample)
    return float(c[0] * os_x), float(c[1] * os_y)


def first_moment(stamp):
    """The finite-array first moment in fine-array samples (x, y), relative to the exact array centre."""
    E = _measured_array(stamp, "first_moment")
    ny, nx = E.shape
    s = float(E.sum())
    x = np.arange(nx) - (nx - 1) / 2.0
    y = np.arange(ny) - (ny - 1) / 2.0
    return float(E.sum(axis=0) @ x) / s, float(E.sum(axis=1) @ y) / s


EFFECTIVE_PSF_PAD_MIN = 2.0


def effective_psf(stamp, oversample, charge_diffusion=None, *, pad=None) -> np.ndarray:
    """An optics-only PSF stamp turned into an effective PSF stamp: the pixel integrated and, optionally, charge diffusion added."""
    stamp = np.asarray(stamp, float)
    if stamp.ndim != 2:
        raise ValueError(f"PSF stamp must be 2-d, got shape {stamp.shape}")
    os_x, os_y = _sampling_xy(oversample)
    if charge_diffusion is not None and (not np.isfinite(charge_diffusion)
                                         or charge_diffusion < 0):
        raise ValueError("charge_diffusion must be finite and nonnegative")
    if pad is None:
        reach = max(EFFECTIVE_PSF_PAD_MIN, 0.5 + 6.0 * float(charge_diffusion or 0.0))
        px, py = int(math.ceil(os_x * reach)), int(math.ceil(os_y * reach))
    else:
        px, py = np.broadcast_to(np.asarray(pad), (2,))
        if px != int(px) or py != int(py) or min(px, py) < 0:
            raise ValueError("pad must be nonnegative integer scalar or (x, y)")
        px, py = int(px), int(py)
    padded = np.pad(stamp, ((py, py), (px, px)))
    ny, nx = padded.shape
    fx = np.fft.fftfreq(nx)[None, :] * os_x
    fy = np.fft.fftfreq(ny)[:, None] * os_y
    window = np.sinc(fx) * np.sinc(fy)
    if charge_diffusion is not None:
        s = float(charge_diffusion)
        window = window * np.exp(-2 * np.pi**2 * s**2 * (fx**2 + fy**2))
    return np.fft.ifft2(np.fft.fft2(padded) * window).real


logger = logging.getLogger(__name__)


STPSF_CHECKED_VERSIONS = {"NIRCAM": ("2.1.0", "2.3.0"), "WFI": ("2.1.0", "2.3.0", "2.3.1")}


_WARNED_VERSIONS = set()


_WARNED_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreparedSTPSF:

    """The effective PSF of one PSF stamp, with the record of its recipe."""
    stamp: np.ndarray
    oversample: tuple[float, float]
    origin: tuple[float, float]
    metadata: dict
    header: object = None


def _value(header, key, default=None):
    value = header.get(key, header.get(key.lower(), default))
    return value[0] if isinstance(value, tuple) else value


def check_stpsf_version(version, what="the PSF stamp", instrument=None):
    """Whether version is one the recipe was checked on (warns once if not); None when the stamp carries no version."""
    if version is None or str(version).strip() == "":
        return None
    version = str(version).strip()
    if instrument is None:
        key = None
        checked = sorted(set.intersection(*(set(v) for v in STPSF_CHECKED_VERSIONS.values())))
        whose = "the recipes of this package were"
    else:
        key = str(instrument).strip().upper()
        checked = list(STPSF_CHECKED_VERSIONS.get(key, ()))
        whose = f"the {key} recipe of this package was"
    if version in checked:
        return True
    with _WARNED_LOCK:
        first = (version, key) not in _WARNED_VERSIONS
        _WARNED_VERSIONS.add((version, key))
    if first:
        told = (f"{whose} checked on STPSF {' and '.join(checked)} only" if checked else
                f"this package has no recipe checked for {key}")
        logger.warning(
            "%s comes from STPSF %s; %s. The stamp is used as it is. Check the recipe "
            "against that release (what its extension contains, where its detector position "
            "is) before trusting the kernel.", what, version, told)
    return False


def _request_psf_stamp(instrument, x_init, y_init, *, source, fov_pixels, oversample,
                       extension, options):
    """Ask an STPSF instrument for one PSF stamp at a detector position."""
    if options.get("normalize", "first") != "first":
        raise ValueError("the recipe asks STPSF with normalize='first'; a stamp is never "
                         "normalized to its own sum")
    options = dict(options, normalize="first")
    instrument.detector_position = (x_init, y_init)
    hdul = instrument.calc_psf(source=source, fov_pixels=fov_pixels, oversample=oversample,
                               **options)
    try:
        if extension not in hdul:
            raise ValueError(f"STPSF returned no {extension} extension")
        hdu = hdul[extension]
        stamp = np.array(hdu.data, dtype=float, copy=True)
        header = hdu.header.copy()
    finally:
        close = getattr(hdul, "close", None)
        if close is not None:
            close()
    if str(_value(header, "NORMALIZ", "first")) != "first":
        raise ValueError(f"STPSF's {extension} stamp records NORMALIZ = "
                         f"{_value(header, 'NORMALIZ')!r}, not 'first'")
    if _value(header, "DET_X") is None or _value(header, "DET_Y") is None:
        raise ValueError(f"STPSF's {extension} stamp carries no DET_X, DET_Y: its detector "
                         "position r is unknown")
    check_stpsf_version(_value(header, "VERSION"), f"the {extension} stamp",
                        _value(header, "INSTRUME"))
    return stamp, header


def _snapshot(value):
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("preparation coordinates and samples must be finite")
    result.setflags(write=False)
    return result


def _sampling(value):
    result = np.asarray(value, dtype=float).ravel()
    if result.size == 1:
        result = np.repeat(result, 2)
    if result.shape != (2,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("oversample must be a positive finite scalar or (x, y) pair")
    return result


def _node_values(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape == (2,):
        result = np.broadcast_to(result, shape + (2,))
    elif result.shape == (np.prod(shape), 2):
        result = result.reshape(shape + (2,))
    if result.shape != shape + (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape (2,) or {shape + (2,)}")
    return _snapshot(result)


def _source_field_bounds(value):
    bounds = np.asarray(value, dtype=float)
    if (bounds.shape != (2, 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("source_field_bounds must be ((xmin,xmax),(ymin,ymax)) "
                         "with finite increasing calibrated pixel-center bounds")
    return _snapshot(bounds)


@dataclass(frozen=True)
class WCSOffsetVector:

    """The WCS offset vector c(r) of one exposure at its source positions r (native detector pixels): constant for one position,
    bilinear between the nodes of a grid. forward(r) = r + c(r) is the WCS position eta; inverse(eta) solves for r."""
    source_positions: np.ndarray
    offsets: np.ndarray
    wcs_position_type: str
    metadata: dict | None = None

    def __post_init__(self):
        positions = np.asarray(self.source_positions, dtype=float)
        offsets = np.asarray(self.offsets, dtype=float)
        if positions.shape == (2,):
            positions = positions[None, None]
        if offsets.shape == (2,):
            offsets = offsets[None, None]
        if positions.ndim != 3 or positions.shape[-1] != 2 or offsets.shape != positions.shape:
            raise ValueError("source positions and WCS offset vectors must be (ny, nx, 2)")
        if positions.shape[:2] != (1, 1):
            if min(positions.shape[:2]) < 2:
                raise ValueError("a field of WCS offset vectors needs at least 2 by 2 nodes")
            gx, gy = positions[0, :, 0], positions[:, 0, 1]
            if (np.any(np.diff(gx) <= 0) or np.any(np.diff(gy) <= 0)
                    or not np.allclose(positions[..., 0], gx[None], rtol=0, atol=1e-10)
                    or not np.allclose(positions[..., 1], gy[:, None], rtol=0, atol=1e-10)):
                raise ValueError("source positions must form an increasing tensor grid")
        object.__setattr__(self, "source_positions", _snapshot(positions))
        object.__setattr__(self, "offsets", _snapshot(offsets))
        eta_type = normalize_wcs_position_type(
            self.wcs_position_type, what="WCSOffsetVector(wcs_position_type=...)")
        if eta_type == "chief-ray" and np.any(offsets):
            raise ValueError("a chief-ray WCS uses the landing point itself, eta = r: "
                             "its WCS offset vector is zero")
        object.__setattr__(self, "wcs_position_type", eta_type)
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata or {}))
        if self.is_grid:
            gx, gy = self.grid_x, self.grid_y
            for iy in range(len(gy) - 1):
                for ix in range(len(gx) - 1):
                    points = np.array([[gx[ix], gy[iy]], [gx[ix + 1], gy[iy]],
                                       [gx[ix], gy[iy + 1]], [gx[ix + 1], gy[iy + 1]]])
                    _, derivative = self._interpolate(points, cell=(iy, ix))
                    if np.any(np.linalg.det(derivative + np.eye(2)) <= 0):
                        raise ValueError("the WCS offset vectors fold a cell of source positions")

    @property
    def is_grid(self):
        return self.source_positions.shape[:2] != (1, 1)

    @property
    def grid_x(self):
        return self.source_positions[0, :, 0]

    @property
    def grid_y(self):
        return self.source_positions[:, 0, 1]

    @property
    def wcs_positions(self):
        return self.source_positions + self.offsets

    def _interpolate(self, positions, cell=None, source_field_bounds=None):
        points = np.asarray(positions, dtype=float)
        if points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
            raise ValueError("source positions must be finite (..., 2) coordinates")
        if not self.is_grid:
            return (np.broadcast_to(self.offsets[0, 0], points.shape).copy(),
                    np.zeros(points.shape[:-1] + (2, 2)))
        original = points
        if source_field_bounds is not None:
            bounds = np.asarray(source_field_bounds, float)
            points = np.clip(points, bounds[:, 0], bounds[:, 1])
        gx, gy = self.grid_x, self.grid_y
        x, y = points[..., 0], points[..., 1]
        xc, yc = np.clip(x, gx[0], gx[-1]), np.clip(y, gy[0], gy[-1])
        if cell is None:
            ix = np.clip(np.searchsorted(gx, xc, side="right") - 1, 0, len(gx) - 2)
            iy = np.clip(np.searchsorted(gy, yc, side="right") - 1, 0, len(gy) - 2)
        else:
            iy, ix = cell
        hx, hy = gx[ix + 1] - gx[ix], gy[iy + 1] - gy[iy]
        tx, ty = (xc - gx[ix]) / hx, (yc - gy[iy]) / hy
        v00, v10 = self.offsets[iy, ix], self.offsets[iy, ix + 1]
        v01, v11 = self.offsets[iy + 1, ix], self.offsets[iy + 1, ix + 1]
        value = ((1 - ty)[..., None] * ((1 - tx)[..., None] * v00 + tx[..., None] * v10)
                 + ty[..., None] * ((1 - tx)[..., None] * v01 + tx[..., None] * v11))
        derivative = np.empty(points.shape[:-1] + (2, 2))
        derivative[..., :, 0] = (((1 - ty)[..., None] * (v10 - v00)
                                 + ty[..., None] * (v11 - v01)) / np.asarray(hx)[..., None])
        derivative[..., :, 1] = (((1 - tx)[..., None] * (v01 - v00)
                                 + tx[..., None] * (v11 - v10)) / np.asarray(hy)[..., None])
        derivative[..., :, 0] *= ((x >= gx[0]) & (x <= gx[-1]))[..., None]
        derivative[..., :, 1] *= ((y >= gy[0]) & (y <= gy[-1]))[..., None]
        if source_field_bounds is not None:
            for axis in range(2):
                derivative[..., :, axis] *= ((original[..., axis] >= bounds[axis, 0])
                    & (original[..., axis] <= bounds[axis, 1]))[..., None]
        return value, derivative

    def offset(self, positions, *, source_field_bounds=None):
        """The WCS offset vector c(r), native pixels, at source positions r."""
        bounds = None if source_field_bounds is None else _source_field_bounds(source_field_bounds)
        return self._interpolate(positions, source_field_bounds=bounds)[0]

    def forward(self, positions, *, source_field_bounds=None):
        """H: r -> eta = r + c(r), the detector position used by the WCS."""
        points = np.asarray(positions, dtype=float)
        return points + self.offset(points, source_field_bounds=source_field_bounds)

    def inverse(self, positions, *, tolerance=1e-10, maxiter=30, source_field_bounds=None):
        """The inverse of H: the source position r of a WCS position eta."""
        target = np.asarray(positions, dtype=float)
        bounds = None if source_field_bounds is None else _source_field_bounds(source_field_bounds)
        guess = target - self._interpolate(target, source_field_bounds=bounds)[0]
        for _ in range(int(maxiter)):
            delta, derivative = self._interpolate(guess, source_field_bounds=bounds)
            residual = guess + delta - target
            jac = derivative + np.eye(2)
            det = np.linalg.det(jac)
            if np.any(det <= 0):
                raise ValueError("H is not locally invertible at a requested position")
            if np.max(np.abs(residual), initial=0.0) <= tolerance:
                return guess
            step = np.empty_like(residual)
            step[..., 0] = (jac[..., 1, 1] * residual[..., 0]
                            - jac[..., 0, 1] * residual[..., 1]) / det
            step[..., 1] = (jac[..., 0, 0] * residual[..., 1]
                            - jac[..., 1, 0] * residual[..., 0]) / det
            guess = guess - step
        raise RuntimeError("the inverse of H did not converge; inspect the WCS offset vectors")

    def to_dict(self):
        """JSON-safe representation for a record."""
        return dict(schema="sfft.wcs-offset-vector.v1",
                    source_positions=self.source_positions.tolist(),
                    offsets=self.offsets.tolist(),
                    wcs_position_type=self.wcs_position_type,
                    metadata=copy.deepcopy(self.metadata))

    @classmethod
    def from_dict(cls, value):
        """The vector from its to_dict representation."""
        if value.get("schema") != "sfft.wcs-offset-vector.v1":
            raise ValueError("unsupported WCS-offset-vector schema; measure it again")
        return cls(value["source_positions"], value["offsets"],
                   value["wcs_position_type"], value.get("metadata"))


@dataclass(frozen=True)
class PreparedResponse:

    """The effective PSF at its source positions r, with its provenance."""
    stamps: np.ndarray
    oversample: tuple[float, float]
    origins: np.ndarray
    metadata: dict

    def __post_init__(self):
        stamps = np.asarray(self.stamps, dtype=float)
        if stamps.ndim not in (2, 4) or min(stamps.shape) < 1 or not np.all(np.isfinite(stamps)):
            raise ValueError("prepared stamps must be finite nonempty 2-D or 4-D arrays")
        if np.any(stamps.sum(axis=(-2, -1)) <= 0):
            raise ValueError("prepared stamps must retain positive total weight")
        shape = stamps.shape[:2] if stamps.ndim == 4 else (1, 1)
        origins = _node_values(self.origins, shape, "prepared origins")
        centre = (np.asarray(stamps.shape[-2:][::-1]) - 1) / 2
        if not np.allclose(origins, centre, rtol=0, atol=1e-12):
            raise ValueError("the source position must lie at the exact centre of each array")
        object.__setattr__(self, "stamps", _snapshot(self.stamps))
        object.__setattr__(self, "oversample", tuple(_sampling(self.oversample)))
        object.__setattr__(self, "origins",
                           _snapshot(origins[0, 0] if stamps.ndim == 2 else origins))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def is_grid(self):
        return self.stamps.ndim == 4

    @property
    def source_positions(self):
        shape = self.stamps.shape[:2] if self.is_grid else (1, 1)
        return _node_values(self.metadata["source_positions"], shape, "source_positions")

    @property
    def grid_x(self):
        return self.source_positions[0, :, 0]

    @property
    def grid_y(self):
        return self.source_positions[:, 0, 1]


def _solid_angle_values(provider, points):
    result = np.asarray(provider(points.reshape(-1, 2)), dtype=float)
    if result.size == 1:
        result = np.full(points.shape[:-1], float(result.ravel()[0]))
    else:
        result = result.reshape(points.shape[:-1])
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("Omega_geo must return positive finite geometric solid angles")
    return result


def measure_wcs_offset_vector(stamps, oversample, *, wcs_position_type, origins=None,
                              source_positions=None, Omega_geo=None,
                              gaussian_window_r50=None):

    """The WCS offset vector c for every source position r: the profile e_r(u - r) / Omega_geo(u), its centroid eta by the WCS's
    position type, c = eta - r. No array is shifted."""
    if isinstance(stamps, PreparedResponse):
        raise ValueError("pass the arrays of the effective PSF with their sampling and "
                         "source positions, not a PreparedResponse")
    values = np.asarray(stamps, dtype=float)
    if values.ndim == 2:
        values = values[None, None]
    if values.ndim != 4 or min(values.shape) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("effective PSF arrays must be finite 2-D or (ny,nx,height,width) arrays")
    if np.any(values.sum(axis=(-2, -1)) <= 0):
        raise ValueError("each effective PSF array must retain positive total weight")
    sampling = _sampling(oversample)
    eta_type = normalize_wcs_position_type(
        wcs_position_type, what="measure_wcs_offset_vector(wcs_position_type=...)")
    r50 = gaussian_window_r50
    if r50 is not None:
        r50 = float(r50)
        if not np.isfinite(r50) or r50 <= 0:
            raise ValueError("gaussian_window_r50 must be positive and finite")
        if eta_type != "gaussian-window-centroid":
            raise ValueError("gaussian_window_r50 belongs to "
                             "wcs_position_type='gaussian-window-centroid'")
    shape = values.shape[:2]
    centre = (np.array(values.shape[-2:][::-1]) - 1) / 2
    origin = _node_values(centre if origins is None else origins, shape, "origins")
    if source_positions is None:
        if shape != (1, 1):
            raise ValueError("a field of effective PSFs requires its source_positions")
        positions = np.zeros((1, 1, 2))
    else:
        positions = _node_values(source_positions, shape, "source_positions")
    eta_is_centroid = eta_type != "chief-ray"
    if eta_is_centroid and (Omega_geo is None or source_positions is None):
        raise ValueError("the profile e_r(u - r) / Omega_geo(u) of a surface-brightness image "
                         "requires source_positions and Omega_geo")
    offsets = np.zeros(shape + (2,))
    details = []
    if eta_is_centroid:
        for index in np.ndindex(shape):
            profile = values[index]
            record = {}
            xx, yy = np.meshgrid(
                (np.arange(profile.shape[1]) - origin[index][0]) / sampling[0],
                (np.arange(profile.shape[0]) - origin[index][1]) / sampling[1])
            u = positions[index] + np.stack((xx, yy), axis=-1)
            solid_angle = _solid_angle_values(Omega_geo, u)
            profile = profile / solid_angle
            record["geometric_solid_angle_range"] = [float(solid_angle.min()),
                                                     float(solid_angle.max())]
            if eta_type == "first-moment-centroid":
                centroid = np.asarray(first_moment(profile))
            else:
                centroid = np.asarray(windowed_centroid(
                    profile, tuple(sampling), pixel_integrated=True, r50=r50,
                    tol=1e-10, maxiter=20000))
                radius = float(r50) if r50 is not None else _half_light_radius(
                    profile, (np.arange(profile.shape[1]) - centre[0]) / sampling[0],
                    (np.arange(profile.shape[0]) - centre[1]) / sampling[1], centroid / sampling)
                record["r50_native_pixels"] = radius
            offsets[index] = (centroid + centre - origin[index]) / sampling
            record["wcs_offset_vector_native_pixels"] = offsets[index].tolist()
            details.append(record)
    profile_name = None
    if eta_is_centroid:
        profile_name = "e_r(u - r) / Omega_geo(u)"
    return WCSOffsetVector(positions, offsets, eta_type, dict(
        recipe="c = eta - r",
        wcs_position_type=eta_type,
        gaussian_window_r50=r50, measurements=details, profile=profile_name,
        input_shape=list(values.shape[-2:]), input_origins=origin.tolist(),
        sample_spacing_native_pixels=(1 / sampling).tolist(),
        measurement_support="full supplied finite effective PSF array",
        physical_wcs_calibration_claimed=False,
        interpolation_applied=False, pixel_integration_repeated=False))


ROMAN_RECIPE = "roman-wfi-oversamp-v1"


OU_CD_SCALE = 0.3279


OU_CD_COMPONENTS = ((0.17519, 0.4522), (0.53146, 0.8050), (0.29335, 1.4329))


OU_CD_MIXTURE = tuple((weight, coefficient * OU_CD_SCALE)
                      for weight, coefficient in OU_CD_COMPONENTS)


def roman_diffusion_components(spec, pixel_scale):
    """Gaussian-mixture weights and native-pixel standard deviations of the charge diffusion: None or 0 none, 'openuniverse' a
    three-Gaussian mixture, a scalar a Gaussian sigma in arcsec, or a mapping with sigma_native or sigma_arcsec."""
    if spec is None or (np.isscalar(spec) and not isinstance(spec, str) and spec == 0):
        return None
    if isinstance(spec, str) and spec == "openuniverse":
        return list(OU_CD_MIXTURE)
    if isinstance(spec, str):
        raise ValueError(
            "unknown charge_diffusion; use 'openuniverse', a Gaussian, a fixed kernel, or None"
        )
    if isinstance(spec, dict):
        if spec.get("kind") != "gaussian":
            raise ValueError("Gaussian diffusion requires kind='gaussian'")
        if ("sigma_native" in spec) == ("sigma_arcsec" in spec):
            raise ValueError(
                "Gaussian diffusion requires exactly one of sigma_native or sigma_arcsec"
            )
        sigma = (float(spec["sigma_native"]) if "sigma_native" in spec
                 else float(spec["sigma_arcsec"]) / float(pixel_scale))
    else:
        sigma = float(spec) / float(pixel_scale)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian diffusion sigma must be finite and nonnegative")
    return [(1.0, sigma)] if sigma else None


def apply_roman_diffusion(stamp, oversample, components):
    """Apply one finite-window Gaussian mixture to a fine stamp."""
    from scipy.ndimage import gaussian_filter
    image = np.asarray(stamp, float)
    if components is None:
        return image.copy()
    out = np.zeros_like(image, dtype=float)
    for weight, sigma in components:
        out += weight * gaussian_filter(image, sigma * oversample, mode="constant", cval=0.0)
    return out


def _fixed_kernel(stamp, oversample, specification):
    """Apply a declared discrete linear kernel without changing its sum."""
    from scipy.signal import fftconvolve
    if not isinstance(specification, dict) or specification.get("kind") != "kernel":
        raise ValueError("A supplied detector kernel requires kind='kernel', values, and sampling")
    kernel = np.asarray(specification.get("values"), float)
    if (kernel.ndim != 2 or min(kernel.shape) < 1 or any(n % 2 != 1 for n in kernel.shape)
            or not np.all(np.isfinite(kernel))):
        raise ValueError(
            "A detector kernel must be finite with odd two-dimensional shape and a centered origin"
        )
    if not np.isclose(kernel.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError(
            "A detector spreading kernel must sum to one; it is never silently normalized"
        )
    sampling = specification.get("sampling")
    if sampling == "native":
        expanded = np.zeros(((kernel.shape[0]-1)*oversample+1,
                             (kernel.shape[1]-1)*oversample+1))
        expanded[::oversample, ::oversample] = kernel
        kernel = expanded
    elif sampling == "fine":
        if specification.get("oversample") != oversample:
            raise ValueError("A fine-grid detector kernel must declare the matching oversample")
    else:
        raise ValueError("A detector kernel must declare sampling='native' or 'fine'")
    return fftconvolve(stamp, kernel, mode="same")


def _raw_contract(stamp, header):
    image = np.asarray(stamp, float)
    if image.ndim != 2 or min(image.shape) < 2 or not np.all(np.isfinite(image)):
        raise ValueError("Roman OVERSAMP requires finite two-dimensional samples")
    if str(_value(header, "INSTRUME", "")).upper() != "WFI":
        raise ValueError("Roman adoption supports STPSF WFI OVERSAMP only")
    if str(_value(header, "EXTNAME", "")).upper() != "OVERSAMP":
        raise ValueError(
            "Roman adoption requires raw OVERSAMP, never OVERDIST or a prepared psf_grid"
        )
    detector = str(_value(header, "DETECTOR", ""))
    if re.fullmatch(r"(?:SCA|WFI)\d{2}", detector) is None or not _value(header, "FILTER"):
        raise ValueError("Roman OVERSAMP requires its detector and filter identity")
    os = float(_value(header, "OVERSAMP", np.nan))
    if not np.isfinite(os) or os < 1 or os != int(os):
        raise ValueError("Roman OVERSAMP must specify positive integer oversampling")
    os = int(os)
    if float(_value(header, "DET_SAMP", os)) != os:
        raise ValueError("Roman OVERSAMP and DET_SAMP disagree")
    pitch = float(_value(header, "PIXELSCL", np.nan))
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Roman OVERSAMP needs a positive PIXELSCL")
    if str(_value(header, "NORMALIZ", "")) != "first":
        raise ValueError(
            "Roman OVERSAMP requires normalize='first'; finite images are not normalized"
        )
    position = np.array([_value(header, "DET_X", np.nan), _value(header, "DET_Y", np.nan)], float)
    if not np.all(np.isfinite(position)):
        raise ValueError("Roman OVERSAMP needs DET_X, DET_Y, its detector position r "
                         "(the position of the array's exact centre)")
    asked = [_value(header, "DETXPIXL"), _value(header, "DETYPIXL")]
    if all(v is not None for v in asked):
        expected = 0.5 if (image.shape[0] / os) % 2 == 0 else 0.0
        if not np.allclose(position - np.asarray(asked, float), expected, rtol=0, atol=1e-9):
            raise ValueError(
                f"Roman OVERSAMP: DET_X, DET_Y = {position.tolist()} and DETXPIXL, DETYPIXL = "
                f"{[float(v) for v in asked]} do not differ by {expected:g} px, as a stamp of "
                f"{image.shape[0] / os:g} detector pixels requires")
    if (any(_value(header, key) not in (None, False, 0, "", "None", "none", "N/A")
            for key in ("CHDFTYPE", "CHDFSIGM", "IPCINST", "IPCFILE", "PPCFILE"))
            or bool(_value(header, "PIXINT", False))):
        raise ValueError(
            "Roman OVERSAMP already declares detector effects or pixel integration; refusing "
            "double application"
        )
    if not np.isfinite(image.sum()) or image.sum() <= 0:
        raise ValueError("Roman OVERSAMP must retain positive finite source amplitude")
    return image, os, pitch * os, position.tolist()


def request_wfi_psf_stamp(wfi, x_init, y_init, *, source, fov_pixels, oversample, **options):
    """Ask STPSF for one Roman WFI PSF stamp at a detector position (wfi: an stpsf.roman.WFI with its detector and filter set): (stamp, header)."""
    P_OS, header = _request_psf_stamp(wfi, x_init, y_init, source=source,
                                      fov_pixels=fov_pixels, oversample=oversample,
                                      extension="OVERSAMP", options=options)
    if str(_value(header, "INSTRUME", "WFI")).upper() != "WFI":
        raise ValueError("request_wfi_psf_stamp needs an stpsf.roman.WFI instrument")
    return P_OS, header


def roman_no_pixel_effective_psf(stamp, header, *, diffusion=None, coupling=None):
    """The PSF without the pixel of one WFI stamp, with its record: the band-weighted PSF with the selected charge diffusion, as
    shares of the fine cells."""
    image, os, scale, position = _raw_contract(stamp, header)
    stpsf_version = _value(header, "VERSION")
    version_checked = check_stpsf_version(stpsf_version, "the OVERSAMP stamp", "WFI")
    original_sum = float(image.sum())
    if isinstance(diffusion, dict) and diffusion.get("kind") == "kernel":
        result = _fixed_kernel(image, os, diffusion)
        diffusion_record = dict(diffusion)
    else:
        components = roman_diffusion_components(diffusion, scale)
        result = apply_roman_diffusion(image, os, components)
        diffusion_record = dict(kind="gaussian-mixture" if components else "disabled",
                                components=components)
    if coupling is not None:
        result = _fixed_kernel(result, os, coupling)
    record = dict(
        recipe=ROMAN_RECIPE,
        note="STPSF - Roman/WFI",
        instrument="Roman WFI",
        source_extension="OVERSAMP",
        stpsf_version=None if stpsf_version is None else str(stpsf_version),
        stpsf_version_checked=version_checked,
        stpsf_checked_versions=list(STPSF_CHECKED_VERSIONS["WFI"]),
        source_detector_xy=position,
        sampling_pitches_w_h=[1 / os, 1 / os],
        detector_fine_cell_area=1 / os**2,
        anchor="source position r at the exact array centre",
        geometric_solid_angle=("not in this array: the exposure's geometry divides by "
                               "Omega_geo once, when the kernel is built"),
        finite_normalization=False,
        input_state="PSF stamp P_OS: the shares of the OVERSAMP cells",
        output_state="no-pixel effective PSF e~_r times the detector fine-cell area",
        native_pixel_integrations=0,
        original_sum=original_sum,
        prepared_sum=float(result.sum()),
        detector_effects=dict(
            optical_jitter="retained from producer",
            intrinsic_distortion="retained; no additional SIAF image warp",
            added_diffusion=diffusion_record,
            added_coupling=coupling,
            application_count=1,
        ),
        source_header={
            key: _value(header, key)
            for key in (
                "INSTRUME",
                "FILTER",
                "DETECTOR",
                "VERSION",
                "DATAVERS",
                "NORMALIZ",
                "PIXELSCL",
                "OVERSAMP",
                "DET_SAMP",
                "DET_X",
                "DET_Y",
                "DETXPIXL",
                "DETYPIXL",
                "JITRTYPE",
                "JITRSIGM",
            )
        },
    )
    return result, record


def integrate_roman_pixel(over, oversample):
    """The pixel window, once: the effective PSF from the PSF without the pixel, on the same odd grid."""
    import galsim
    os = int(oversample)
    if os < 1 or os != oversample:
        raise ValueError("Roman pixel integration requires positive integer oversampling")
    image = np.asarray(over, float)
    if image.ndim != 2 or image.shape[0] != image.shape[1] or not np.all(np.isfinite(image)):
        raise ValueError("Roman pixel integration requires a finite square pre-pixel image")
    half = (image.shape[0] // os - 1) // 2
    npix = 2 * half * os + 1
    profile = galsim.InterpolatedImage(galsim.Image(np.ascontiguousarray(image), scale=1/os))
    effective = galsim.Convolve(profile, galsim.Pixel(1.0))
    return effective.drawImage(nx=npix, ny=npix, scale=1/os, method="no_pixel").array.astype(float)


def prepare_stpsf_roman(stamp, header, *, diffusion=None, coupling=None,
                       wcs_position_type=None, gaussian_window_r50=None, Omega_geo=None):
    """The effective PSF of one WFI PSF stamp: charge diffusion applied once and the pixel window added."""
    over, metadata = roman_no_pixel_effective_psf(stamp, header, diffusion=diffusion,
                                                  coupling=coupling)
    os = int(_value(header, "OVERSAMP"))
    prepared = integrate_roman_pixel(over, os)
    origin = tuple((np.asarray(prepared.shape[::-1])-1)/2)
    metadata.update(
        output_state="effective PSF e_r times the detector fine-cell area",
        native_pixel_integrations=1,
        prepared_sum=float(prepared.sum()),
        integration=(
            "the pixel window, once: GalSim InterpolatedImage and one Pixel(1), no_pixel "
            "drawing on the retained odd grid"
        ),
        wcs_offset_vector_computed=wcs_position_type is not None,
        profile_shift_applied=False,
    )
    if wcs_position_type is not None:
        measured = measure_wcs_offset_vector(
            prepared, (float(os), float(os)), wcs_position_type=wcs_position_type,
            origins=origin, source_positions=metadata["source_detector_xy"],
            Omega_geo=Omega_geo,
            gaussian_window_r50=gaussian_window_r50)
        metadata["wcs_offset_vector"] = measured.to_dict()
    return PreparedSTPSF(prepared, (float(os), float(os)), origin, metadata, dict(header))
