"""Where an exposure's PSF comes from: a callable epsf_of(exposure, x, y) -> PsfSample(stamp, oversample, pixel_integrated) for a source at
detector position (x, y). The stamp is [y, x] on a grid of 1/oversample pixel, centred, in the detector frame; its sum is the share of a
point source's light it holds, and is kept.

  'stpsf'  STPSF's Roman WFI model at an n x n grid of detector positions per detector and filter, interpolated bilinearly, with charge
           diffusion and without the pixel; evaluated in spawned worker processes with the optics engine on the CPU, and cached on disk.
  'array'  one PSF array for every exposure and position, with its sampling and whether it holds the pixel.

wcs_offset_vector measures, for a WCS fitted to centroids of stars, where that WCS places a source relative to the model's centre."""
import hashlib
import json
import os
import sys
import time

import numpy as np

from ..subtract.psfmodel import PsfSample, _box_ft

N_DET = 4088
PSF_SOURCES = ("stpsf", "array")


class ArrayPSF:

    """One PSF array for every exposure and position: square, centred at (n - 1) / 2 on each axis, its wings reaching zero at the edges."""
    def __init__(self, stamp, oversample, pixel_integrated=True):
        self.stamp = load_array(stamp)
        self.oversample = int(oversample)
        self.pixel_integrated = bool(pixel_integrated)
        self.info = dict(source="array", shape=list(self.stamp.shape), oversample=self.oversample, pixel_integrated=self.pixel_integrated,
                         sha256=array_digest(self.stamp))

    def prepare(self, pairs):
        pass

    def __call__(self, level2, det_x, det_y):
        return PsfSample(self.stamp, self.oversample, self.pixel_integrated)


def load_array(stamp):
    """The PSF array as float64 [y, x]; refuses what cannot be a PSF."""
    if stamp is None:
        raise ValueError("the PSF source 'array' needs psf.stamp: the PSF as a 2-D NumPy array")
    if isinstance(stamp, dict):
        raise ValueError("psf.stamp must be the PSF as a 2-D NumPy array, not a dict")
    if isinstance(stamp, (str, bytes, os.PathLike)):
        raise ValueError(f"psf.stamp must be the PSF as a NumPy array, not the path {os.fspath(stamp)!r}: load the file first, np.load('my_psf.npy')")
    a = np.ascontiguousarray(np.asarray(stamp, dtype=np.float64))
    if a.ndim != 2 or min(a.shape) < 3:
        raise ValueError(f"psf.stamp must be a 2-D array [y, x] of at least 3 x 3 samples; it has shape {a.shape}")
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"psf.stamp must be square (n x n samples, as the injection's drawing of sources assumes); it has shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("psf.stamp holds NaN or infinite values")
    if not a.sum() > 0:
        raise ValueError("psf.stamp must have a positive sum")
    return a


def array_digest(a):
    """SHA-256 of an array's shape and float64 values."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(repr(a.shape).encode() + a.tobytes()).hexdigest()


def array_warnings(a, oversample):
    """What to say about a PSF array that can be used but is likely wrong: fewer than 2 samples per pixel, an even size, a sum far from 1."""
    out = []
    if oversample < 2:
        out.append(f"the PSF array has {oversample} sample per native pixel: below 2 the package cannot interpolate it accurately (on STPSF's F146 PSF, "
                   "one sample per pixel makes PSF fluxes 2.3 % low); give it at 2 or more")
    if a.shape[0] % 2 == 0 or a.shape[1] % 2 == 0:
        out.append(f"the PSF array has an even size, {a.shape[0]} x {a.shape[1]}: its centre is taken at (n - 1) / 2, between the two middle samples, "
                   "while most tools centre a PSF on a sample, which this would shift by half a sample")
    s = float(a.sum())
    if not 0.5 <= s <= 1.05:
        out.append(f"the PSF array sums to {s:.4g}: the package keeps a PSF's sum, the share of a point source's light it holds, so PSF fluxes "
                   f"come out divided by {s:.4g}; give the PSF of a source of unit flux (for a PSF built from stars, the share of the light its "
                   "stamp holds; photutils' own ePSF sums to oversample^2)")
    return out


def _bilinear(st, xs, ys, det_x, det_y):
    """Bilinear interpolation of grid stamps at a detector position, held at the grid's edge."""
    if len(xs) == 1 and len(ys) == 1:
        return st[0, 0]
    xc = float(np.clip(det_x, xs[0], xs[-1]))
    yc = float(np.clip(det_y, ys[0], ys[-1]))
    i = int(np.clip(np.searchsorted(xs, xc) - 1, 0, len(xs) - 2))
    j = int(np.clip(np.searchsorted(ys, yc) - 1, 0, len(ys) - 2))
    tx = (xc - xs[i]) / (xs[i + 1] - xs[i])
    ty = (yc - ys[j]) / (ys[j + 1] - ys[j])
    return ((1 - ty) * ((1 - tx) * st[j, i] + tx * st[j, i + 1]) + ty * ((1 - tx) * st[j + 1, i] + tx * st[j + 1, i + 1]))


def _stpsf_version():
    """The installed STPSF version."""
    from importlib import metadata
    try:
        return metadata.version("stpsf")
    except metadata.PackageNotFoundError:
        return "unknown"


def _stpsf_data_version():
    """The version of STPSF's data files."""
    path = os.environ.get("STPSF_PATH") or os.path.expanduser("~/data/stpsf-data")
    try:
        with open(os.path.join(path, "version.txt")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def _stpsf_data_problem():
    """None when STPSF's data files are there, else the reason."""
    path = os.environ.get("STPSF_PATH") or os.path.expanduser("~/data/stpsf-data")
    if os.path.isdir(os.path.join(path, "WFI")):
        return None
    return (f"STPSF's data files are not at {path}. Download the data tarball (https://stpsf.readthedocs.io), and set STPSF_PATH to its folder "
            "(for a Jupyter kernel, set it in the kernel's environment).")


def _block_cupy_for_poppy():
    """Make CuPy unimportable in this worker process before poppy loads (poppy chooses its array library when it is imported),
    also when CuPy is imported already, as importing sfft does."""
    for name in ("cupy", "cupyx"):
        sys.modules[name] = None


def recipe_diffusion(spec, pixel_scale_arcsec=0.11):
    """The charge diffusion of the recipe for the option charge_diffusion: None, 'openuniverse' (a three-Gaussian mixture) or a Gaussian sigma in arcsec."""
    if spec is None or (not isinstance(spec, str) and float(spec) == 0):
        return None
    if isinstance(spec, str):
        if spec == "openuniverse":
            return "openuniverse"
        raise ValueError("charge_diffusion must be None, 'openuniverse' or a Gaussian sigma in arcsec")
    return dict(kind="gaussian", sigma_native=float(spec) / pixel_scale_arcsec)


def _stpsf_node(task):
    """Worker: STPSF's PSF without the pixel at one detector position, by the Roman WFI recipe (stpsf_roman), with poppy on the CPU."""
    _block_cupy_for_poppy()
    det, filt, x, y, oversample, fov, sed, nlambda, diffusion = task
    import stpsf
    from . import stpsf_roman
    wfi = stpsf.roman.WFI()
    wfi.filter = filt
    wfi.detector = det
    wfi.options.setdefault("crop_psf", False)
    source = None
    if sed:
        from synphot import SourceSpectrum
        from synphot.models import BlackBodyNorm1D
        if not (sed.startswith("bb") and sed[2:].isdigit()):
            raise ValueError("sed must be None or 'bb<T>' (a blackbody of T kelvin)")
        source = SourceSpectrum(BlackBodyNorm1D, temperature=float(sed[2:]))
    t0 = time.time()
    P_OS, header = stpsf_roman.request_wfi_psf_stamp(wfi, float(x), float(y), source=source, fov_pixels=int(fov), oversample=int(oversample),
                                                     nlambda=nlambda)
    e_tilde, record = stpsf_roman.roman_no_pixel_effective_psf(P_OS, header, diffusion=diffusion)
    return dict(x=float(x), y=float(y), r=[float(v) for v in record["source_detector_xy"]], stamp=np.asarray(e_tilde, np.float64),
                seconds=time.time() - t0, record=record)


class StpsfPSF:

    """STPSF at an n x n grid of detector positions per (detector, filter), interpolated bilinearly. Grids are cached on disk with each node's
    record and computed again when the STPSF version, its data files or the recipe change."""
    def __init__(self, cache_dir, n_grid=3, oversample=5, half=32, charge_diffusion="openuniverse", sed=None, nlambda=None,
                 nproc=4, pixel_scale_arcsec=0.11, log=print):
        from . import stpsf_roman
        self.cache_dir = os.path.join(cache_dir, "stpsf")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.n, self.oversample, self.half = int(n_grid), int(oversample), int(half)
        self.cd, self.sed, self.nlambda, self.nproc = charge_diffusion, sed, nlambda, int(nproc)
        self.diffusion = recipe_diffusion(charge_diffusion, pixel_scale_arcsec)
        self.recipe = stpsf_roman.ROMAN_RECIPE
        self.grids = {}
        self.log = log
        self.info = dict(source="stpsf", n_grid=self.n, oversample=self.oversample, half=self.half, charge_diffusion=str(charge_diffusion),
                         sed=sed, nlambda=nlambda, recipe=dict(name=self.recipe), grids={})

    def _path(self, det, filt):
        return os.path.join(self.cache_dir, f"{det}_{filt}_os{self.oversample}_half{self.half}_n{self.n}_cd-{self.cd}_sed-{self.sed}_nl-{self.nlambda}.npz")

    def grid(self, det, filt):
        """The grid of (det, filt): dict(x, y, stamps, seconds, records), from the cache or computed."""
        key = (det, filt)
        if key in self.grids:
            return self.grids[key]
        path = self._path(det, filt)
        version = f"{_stpsf_version()}, data {_stpsf_data_version()}"
        cached = None
        if os.path.exists(path):
            with np.load(path) as z:
                made_with = str(z["stpsf_version"]) if "stpsf_version" in z.files else "an unrecorded version"
                recipe = str(z["recipe"]) if "recipe" in z.files else None
                if made_with == version and recipe == self.recipe:
                    cached = dict(x=z["x"], y=z["y"], stamps=z["stamps"], seconds=float(z["seconds"]), records=json.loads(str(z["records"])))
                elif recipe != self.recipe:
                    self.log(f"STPSF grid {det} {filt} in the cache was made " + (f"by the recipe {recipe}" if recipe else "before the copied recipe")
                             + f", this is {self.recipe}: computing it again")
                else:
                    self.log(f"STPSF grid {det} {filt} in the cache was made with STPSF {made_with}, this is {version}: computing it again")
        if cached is not None:
            g = cached
            from . import stpsf_roman
            for rec in g["records"]:
                rec["stpsf_version_checked"] = stpsf_roman.check_stpsf_version(rec.get("stpsf_version"), "the cached OVERSAMP stamps", "WFI")
            self.log(f"STPSF grid {det} {filt} from cache {path}")
        else:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            why = _stpsf_data_problem()
            if why:
                raise RuntimeError(why)
            xs = (np.linspace(4.0, N_DET - 5.0, self.n) if self.n > 1 else np.array([(N_DET - 1) / 2.0]))
            ys = xs.copy()
            tasks = [(det, filt, x, y, self.oversample, 2 * self.half + 1, self.sed, self.nlambda, self.diffusion) for y in ys for x in xs]
            nproc = max(1, min(self.nproc, len(tasks)))
            self.log(f"STPSF grid {det} {filt}: {len(tasks)} evaluations on {nproc} processes ...")
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=nproc, mp_context=multiprocessing.get_context("spawn")) as pool:
                res = list(pool.map(_stpsf_node, tasks))
            stamps = np.array([q["stamp"] for q in res]).reshape(self.n, self.n, *res[0]["stamp"].shape)
            r = np.array([q["r"] for q in res], float).reshape(self.n, self.n, 2)
            xs_r, ys_r = r[0, :, 0], r[:, 0, 1]
            if not (np.allclose(r[..., 0], xs_r[None, :], rtol=0, atol=1e-9) and np.allclose(r[..., 1], ys_r[:, None], rtol=0, atol=1e-9)
                    and np.all(np.diff(xs_r) > 0) and np.all(np.diff(ys_r) > 0)):
                raise RuntimeError(f"the detector positions r = (DET_X, DET_Y) that STPSF wrote for the grid {det} {filt} do not form a tensor grid: "
                                   f"{r.reshape(-1, 2).tolist()}")
            records = [q["record"] for q in res]
            g = dict(x=xs_r, y=ys_r, stamps=stamps, seconds=time.time() - t0, records=records)
            np.savez_compressed(path, x=xs_r, y=ys_r, stamps=stamps, seconds=g["seconds"], positions=r.reshape(-1, 2),
                                asked=np.array([[q["x"], q["y"]] for q in res]), stpsf_version=version, recipe=self.recipe,
                                records=json.dumps(records, default=str))
            self.log(f"STPSF grid {det} {filt} done in {g['seconds']:.0f} s, stamp {stamps.shape[-2:]} at 1/{self.oversample} px, stamp sums "
                     f"{stamps.sum(axis=(-2, -1)).min():.4f}-{stamps.sum(axis=(-2, -1)).max():.4f} (kept), cached at {path}")
        first = g["records"][0]
        if first.get("stpsf_version_checked") is False:
            self.log(f"note: the PSF stamps come from STPSF {first.get('stpsf_version')}, which is not among the versions the copied recipe "
                     f"was checked on; they are used as they are (sfft.spaceflow.prep.stpsf_roman.check_stpsf_version)")
        self.grids[key] = g
        self.info["grids"][f"{det} {filt}"] = dict(path=path, seconds=g["seconds"], nodes_x=[float(v) for v in g["x"]], nodes_y=[float(v) for v in g["y"]],
                                                   stpsf_version=first.get("stpsf_version"), stpsf_version_checked=first.get("stpsf_version_checked"),
                                                   stamp_sums=[float(v) for v in np.asarray(g["stamps"]).sum(axis=(-2, -1)).ravel()])
        return g

    def prepare(self, pairs):
        """The grids of these (detector, filter) pairs, now: a problem with STPSF shows before the images are read."""
        for det, filt in sorted(set(pairs)):
            self.grid(det, filt)

    def __call__(self, level2, det_x, det_y):
        g = self.grid(level2.meta["detector"], level2.meta["optical_element"])
        return PsfSample(_bilinear(g["stamps"], g["x"], g["y"], det_x, det_y), self.oversample, False)


def _model_nodes(epsf_of, scan):
    """The detector positions where a PSF source holds its model."""
    if isinstance(epsf_of, StpsfPSF):
        g = epsf_of.grid(scan.meta["detector"], scan.meta["optical_element"])
        return np.asarray(g["x"], float), np.asarray(g["y"], float)
    ny, nx = scan.shape
    return np.array([(nx - 1) / 2.0]), np.array([(ny - 1) / 2.0])


def wcs_offset_vector(scan, epsf_of, wcs_position_type, surface_brightness, gaussian_window_r50=None):
    """The WCS offset vector c(r) of one exposure, or None for 'chief-ray'. For a WCS fitted to centroids of stars, c = eta - r is
    measured on the effective PSF at the source's nodes (divided by the pixels' solid angle for a surface-brightness file), eta by the
    centroid estimator of the type."""
    from . import stpsf_roman
    from ..subtract.geometry import ARCSEC_PER_RAD, pixel_area_function
    eta_type = stpsf_roman.normalize_wcs_position_type(wcs_position_type, what="wcs_position_type")
    if eta_type == "chief-ray":
        return None
    xs, ys = _model_nodes(epsf_of, scan)
    samples = [[epsf_of(scan, float(x), float(y)) for x in xs] for y in ys]
    os_ = int(samples[0][0].oversample)
    e_r = np.array([[np.asarray(s.stamp, np.float64) if s.pixel_integrated
                     else np.real(np.fft.ifft2(np.fft.fft2(np.asarray(s.stamp, np.float64)) * _box_ft(np.shape(s.stamp), os_))) for s in row]
                    for row in samples])
    if surface_brightness:
        area = pixel_area_function(scan.gwcs, scan.shape)

        def Omega_geo(u):
            return area(u[:, 0], u[:, 1]) / ARCSEC_PER_RAD ** 2
    else:
        def Omega_geo(u):
            return np.ones(len(u))
    positions = np.stack(np.meshgrid(xs, ys), axis=-1)
    return stpsf_roman.measure_wcs_offset_vector(e_r, (float(os_), float(os_)), wcs_position_type=eta_type, source_positions=positions,
                                                 Omega_geo=Omega_geo, gaussian_window_r50=gaussian_window_r50)


def make(config, cache_dir, log=print):
    """The PSF source from its configuration: dict(source='stpsf' | 'array', ...)."""
    src = config.get("source", "stpsf")
    if src == "stpsf":
        return StpsfPSF(cache_dir, n_grid=config.get("n_grid", 3), oversample=config.get("oversample", 5), half=config.get("half", 32),
                        charge_diffusion=config.get("charge_diffusion", "openuniverse"), sed=config.get("sed"), nlambda=config.get("nlambda"),
                        nproc=config.get("nproc", 4), log=log)
    if src == "array":
        if config.get("oversample") is None:
            raise ValueError("the PSF source 'array' needs psf.oversample, the samples per native pixel of the array")
        return ArrayPSF(config.get("stamp"), config["oversample"], config.get("pixel_integrated", True))
    raise ValueError(f"psf source must be one of {PSF_SOURCES}, not {src!r}")
