"""SpaceL2_Flow: two Level 2 exposures differenced end to end, from the two files to the difference, its score and its variance.

    from sfft.spaceflow import SpaceL2_Flow

    if __name__ == "__main__":          # a script needs this guard: the STPSF PSF source evaluates PSFs in spawned worker processes
        flow = SpaceL2_Flow(science_file, reference_file, output_dir="difference")
        record = flow.run()
        difference, score, variance = flow.difference, flow.score, flow.variance

run() goes through SpaceL2_Prep (sfft.spaceflow.prep) and SpaceL2_Sub (sfft.spaceflow.subtract) in their documented order: both exposures read
and prepared, and the PSF source evaluated at the PSF nodes (SpaceL2_Prep); the reference resampled onto the science exposure's own pixels
and the pair matched by cross-convolution (SpaceL2_Sub); the fit mask made on the matched pair (SpaceL2_Prep.fit_mask); SFFT's kernel fitted and
the difference formed, science minus reference, so that a transient in the science exposure is positive; then the score and variance
images. The results become attributes of the object; with output_dir, run() also writes them.

Products in output_dir (images on the science exposure's grid, float32, in the unit record['units']['bunit']):
  diff.fits        DIFF (the difference), SCORE (the detection image, in units of the local noise), VARIANCE (the difference's variance),
                   SCI_M and REF_M (the matched pair), FITMASK (the pixels SFFT fitted on), SATURATED (the pixels either exposure flags as
                   saturated, when the files carry that flag)
  sci_stack.fits   IMAGE, VARIANCE: the science exposure as prepared, on its own pixels
  ref_stack.fits   IMAGE, VARIANCE: the reference exposure resampled onto the science exposure's pixels
  matched.fits     SCI_M, REF_M, and D_MATCHED (their difference before SFFT)
  fitmask.fits     the fit mask (1 on the pixels SFFT fitted on)
  psf_diff_eff.npy, psf_diff_eff_fine.npy
                   a transient's PSF in the difference, unit total flux, at native pixels and at 5 samples per pixel
  psf_nodes.npz    both exposures' working PSF models at the PSF nodes: sci and ref [ny, nx, n, n], xs and ys (node centres, 0-based
                   pixels), oversample
  record.json      what was run and what came out: inputs, instrument, settings, versions, backend, grid, units, SFFT's numbers, the fit
                   mask's record, timings; status 'failed' and the error when the run stops
run() removes these files from output_dir before it computes (other files are left alone), so a failed run leaves no earlier products.
"""
import hashlib
import json
import logging
import os
import platform
import sys
import time

import numpy as np

_logger = logging.getLogger(__name__)

PRODUCTS = ("diff.fits", "sci_stack.fits", "ref_stack.fits", "matched.fits", "fitmask.fits", "psf_diff_eff.npy", "psf_diff_eff_fine.npy",
            "psf_nodes.npz", "record.json")
BACKENDS = ("auto", "gpu", "cpu")
MAX_DEFAULT_THREADS = 32


class SpaceL2_Flow:
    """Two Level 2 exposures differenced end to end: SpaceL2_Prep, SpaceL2_Sub, the fit mask, SFFT, the score and variance images.

    Parameters
    ----------
    science_file, reference_file : str
        The two Level 2 files, in one filter. The science exposure's own pixel grid is the grid of every product.
    output_dir : str, optional
        Where run() writes the products (module docstring); None: nothing is written (write() can do it later).
    instrument : str, default 'Roman-WFI'
        The instrument that took both exposures, one of sfft.spaceflow.instruments.INSTRUMENTS; given to SpaceL2_Prep and SpaceL2_Sub.
    backend : 'auto', 'gpu' or 'cpu', default 'auto'
        Where the subtraction runs; 'auto' takes the GPU when one works here, else the CPU. 'gpu' without a usable GPU stops run() before
        anything is read. Resampling runs on the CPU either way.
    gpu_device : int, default 0
        The GPU used with the GPU backend.
    cpu_threads : int, optional
        Threads of the CPU work (resampling, FFTs, SFFT on the CPU); None: the CPUs this process may use, at most 32.
    GKerHW, KerPolyOrder, BGPolyOrder, ConstPhotRatio
        SFFT's kernel half-width, the polynomial degrees of the kernel's and the background's spatial variation, and one flux scale over
        the field (SpaceL2_Sub's parameters; these defaults are SpaceL2_Prep.SFFT_KERNEL's).
    mask_options : dict, optional
        Keyword options of the fit mask (SpaceL2_Prep.fit_mask, i.e. sfft.spaceflow.prep.flowmask.fit_mask_from_flow).
    **prep_options
        SpaceL2_Prep's other keyword options: cache_dir, psf, crop_px, crop_centre, keep_flags, sky_model, mesh_px, add_dumo, pixel_area,
        reject_outliers, wcs_position_type, wcs_gaussian_window_r50, nodes. crop_px=N works on an N x N part of the science exposure, for a
        quick look.

    Attributes after run()
    ----------------------
    prep, sub : the SpaceL2_Prep and SpaceL2_Sub objects (their exposures as read are released before the solve; the subtraction's public
        attributes stay, see SpaceL2_Sub).
    difference, score, variance : 2-D numpy arrays [y, x] on the science exposure's grid (NaN where the difference has no value).
    fit_mask, saturation_map : bool arrays [y, x] (saturation_map None when the files carry no saturation flag).
    header : the grid's FITS header (the science exposure's TAN-SIP WCS).
    record : what run() returns (as record.json).
    products : {file name: path} of the files written.
    The object keeps every array in memory (on the GPU too, with that backend); delete it to release them.
    """

    def __init__(self, science_file, reference_file, output_dir=None, instrument="Roman-WFI", backend="auto", gpu_device=0, cpu_threads=None,
                 GKerHW=2, KerPolyOrder=2, BGPolyOrder=0, ConstPhotRatio=True, mask_options=None, **prep_options):
        import inspect
        from .instruments import instrument_name
        from .prep import SpaceL2_Prep
        self.instrument = instrument_name(instrument)
        if str(backend).lower() not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
        known = set(inspect.signature(SpaceL2_Prep.__init__).parameters) - {"self", "science_file", "reference_file", "instrument"}
        unknown = sorted(set(prep_options) - known)
        if unknown:
            raise TypeError(f"unknown option(s) {unknown}: SpaceL2_Flow takes SpaceL2_Prep's options {sorted(known)}")
        self.science_file, self.reference_file = os.fspath(science_file), os.fspath(reference_file)
        for f in (self.science_file, self.reference_file):
            if not os.path.isfile(f):
                raise FileNotFoundError(f"no such file: {f}")
        self.output_dir = None if output_dir is None else os.fspath(output_dir)
        self.backend = str(backend).lower()
        self.gpu_device = int(gpu_device)
        self.cpu_threads = None if cpu_threads is None else int(cpu_threads)
        self.GKerHW, self.KerPolyOrder, self.BGPolyOrder = int(GKerHW), int(KerPolyOrder), int(BGPolyOrder)
        self.ConstPhotRatio = bool(ConstPhotRatio)
        self.mask_options = dict(mask_options or {})
        self.prep_options = dict(prep_options)
        self.prep = self.sub = None
        self.difference = self.score = self.variance = self.fit_mask = self.saturation_map = self.header = None
        self.record, self.products = None, {}

    # ------------------------------------------------------------------------------------------------------------------------------ run
    def run(self):
        """Difference the two exposures, then write the products when output_dir is set. Returns the record (also in self.record); raises on
        failure (record.json then says status 'failed' and why)."""
        from .subtract import backend as be
        backend = be.resolve(self.backend)          # first: a GPU that cannot be used stops the run before anything is read or removed
        threads = self.cpu_threads or _default_threads()
        rec = dict(status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), instrument=self.instrument,
                   inputs=dict(science=_file_record(self.science_file), reference=_file_record(self.reference_file)),
                   settings=self._settings_record(), versions=_versions(backend),
                   backend=dict(setting=self.backend, used=backend, cpu_threads=threads, gpu_device=self.gpu_device if backend == "gpu" else None),
                   host=platform.node(), python=sys.executable, output_dir=None if self.output_dir is None else os.path.abspath(self.output_dir),
                   timings={})
        self.record, self.products = rec, {}
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            for name in PRODUCTS:
                path = os.path.join(self.output_dir, name)
                if os.path.exists(path):
                    os.remove(path)
        try:
            self._compute(backend, threads)
            if self.output_dir is not None:
                self.write(self.output_dir)
        except BaseException as e:
            rec.update(status="failed", error=f"{type(e).__name__}: {e}", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            if self.output_dir is not None:
                _write_json(os.path.join(self.output_dir, "record.json"), rec)
            raise
        return rec

    def _compute(self, backend, threads):
        from .prep import SpaceL2_Prep
        from .subtract import SpaceL2_Sub
        from .subtract.crossconv import robust_sigma
        rec, T = self.record, self.record["timings"]
        t_all = time.time()
        _logger.info("SpaceL2_Flow: science %s, reference %s, instrument %s, backend %s", os.path.basename(self.science_file),
                     os.path.basename(self.reference_file), self.instrument, backend + (f" (GPU {self.gpu_device})" if backend == "gpu" else
                                                                                        f" ({threads} CPU threads)"))
        # 1. the two exposures and the PSF source
        t0 = time.time()
        prep = SpaceL2_Prep(self.science_file, self.reference_file, instrument=self.instrument, **self.prep_options)
        T["read"] = time.time() - t0
        t0 = time.time()
        args = prep.sub_arguments()
        T["psf_nodes"] = time.time() - t0
        args.update(GKerHW=self.GKerHW, KerPolyOrder=self.KerPolyOrder, BGPolyOrder=self.BGPolyOrder)
        ny, nx = prep.shape
        rec["grid"] = dict(nx=nx, ny=ny, crop_px=self.prep_options.get("crop_px"))
        rec["units"] = dict(prep.units)
        rec["prep"] = dict(rotation_deg=prep.rotation_deg, max_wcs_offset_px=prep.max_wcs_offset_px,
                           saturated_pixels=None if prep.saturation_map is None else int(prep.saturation_map.sum()),
                           psf_oversampling=list(args["PSF_OVERSAMPLING"]), psf_pixel_integrated=list(args["PSF_PIXEL_INTEGRATED"]),
                           psf_nodes=list(np.shape(args["PSF_target"])[:2]))
        _logger.info("read both exposures in %.0f s and the PSF at the nodes in %.0f s; grid %d x %d, rotation between the pixel grids %+.3f deg",
                     T["read"], T["psf_nodes"], nx, ny, prep.rotation_deg)

        # 2. the subtraction
        sub = SpaceL2_Sub(**args, BACKEND_4SUBTRACT="Cupy" if backend == "gpu" else "Numpy", CUDA_DEVICE_4SUBTRACT=str(self.gpu_device),
                          NUM_CPU_THREADS_4SUBTRACT=threads, ConstPhotRatio=self.ConstPhotRatio)
        # the subtraction keeps its own copies; the exposures as read are not needed again (the fit mask uses only prep.saturation_map):
        # free both before the solve, the run's memory peak
        del args
        prep.science = prep.reference = None
        for step in ("resample_image_mask_psf", "cross_convolve"):
            t0 = time.time()
            getattr(sub, step)()
            T[step] = time.time() - t0
        t0 = time.time()
        fit = prep.fit_mask(sub, **self.mask_options)
        T["fit_mask"] = time.time() - t0
        mrec = prep.mask_record or {}
        t0 = time.time()
        sub.sfft_subtract(fit_mask=fit)
        sub.find_decorrelation()
        T["sfft_subtract"] = time.time() - t0
        t0 = time.time()
        diff = sub.apply_decorrelation(sub.PixA_DIFF)
        score = sub.create_score_image()
        variance = sub.create_variance_image()
        T["score_variance"] = time.time() - t0

        # 3. the numbers of the run
        sci_m, ref_m = _yx(sub, sub.PixA_Ctarget), _yx(sub, sub.PixA_Cresamp_object)
        d_matched = sci_m - ref_m
        free = _yx(sub, sub.SourceFreeMask).astype(bool)
        sig_diff = robust_sigma(diff[free & np.isfinite(diff)])
        sig_matched = robust_sigma(d_matched[free & np.isfinite(d_matched)])
        ok = np.isfinite(diff) & np.isfinite(d_matched)
        sfft_change = float(np.max(np.abs(diff[ok] - d_matched[ok]))) if ok.any() else 0.0
        fit_mask = np.asarray(fit, bool)
        rec["sfft"] = dict(used=bool(sub.SFFT_USED), problem=sub.SFFT_PROBLEM, fscal=sub.SFFT_FSCAL, kernel_centre=sub.SFFT_KERNEL_CENTRE,
                           degree=sub.SFFT_DEGREE, fit_pixels=int(sub.SFFT_FIT_PIXELS), gker_hw=int(sub.GKerHW), sigma_before=sig_matched,
                           sigma_after=sig_diff, max_change_in_noise=sfft_change / sig_diff if sig_diff > 0 else None)
        rec["fit_mask"] = dict(pixels=int(fit_mask.sum()), method="band mask (the sparse selection failed)" if mrec.get("sfft_problem") else
                               "SFFT sparse selection with the saturated-star guard and the PSF star test",
                               fallback_problem=mrec.get("sfft_problem"), record={k: v for k, v in mrec.items() if np.isscalar(v) or v is None})
        rec["psf"] = dict(fwhm_science_px=float(sub.PSF_FWHM), work_oversampling=int(sub.PSF_WORK_OVERSAMPLING),
                          source=prep.psf_config.get("source"))
        self.prep, self.sub, self.header = prep, sub, prep.header
        self.difference, self.score, self.variance, self.fit_mask = diff, score, variance, fit_mask
        self.saturation_map = prep.saturation_map
        T["total"] = time.time() - t_all
        rec["status"] = "done"
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if sub.SFFT_USED:
            _logger.info("SFFT: flux scale %.5f, kernel centre %.3f (degree %d), %d fit pixels; difference noise %.5g (matched difference %.5g) %s",
                         sub.SFFT_FSCAL, sub.SFFT_KERNEL_CENTRE, sub.SFFT_DEGREE, sub.SFFT_FIT_PIXELS, sig_diff, sig_matched,
                         rec["units"].get("bunit"))
        else:
            _logger.warning("SFFT was not used (%s); the difference is the matched pair's", sub.SFFT_PROBLEM)
        _logger.info("difference done in %.0f s", T["total"])

    # ---------------------------------------------------------------------------------------------------------------------------- write
    def write(self, output_dir=None):
        """Write the products of the last run() into output_dir (default: the constructor's). Returns {file name: path}."""
        from astropy.io import fits
        import sfft
        if self.sub is None or self.record is None or self.record.get("status") != "done":
            raise RuntimeError("write() needs a finished run(): call run() first")
        output_dir = output_dir if output_dir is not None else self.output_dir
        if output_dir is None:
            raise ValueError("no output_dir: give one to write() or to the constructor")
        output_dir = os.fspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        t0 = time.time()
        sub, rec, hdr = self.sub, self.record, self.header
        bunit = rec["units"]["bunit"]
        prov = dict(SFFTVER=(str(sfft.__version__), "betasfft version"), INSTRUME=(self.instrument, "instrument (sfft.spaceflow)"),
                    REFFILE=(os.path.basename(self.reference_file), "reference exposure"),
                    SCIFILE=(os.path.basename(self.science_file), "science exposure"))
        paths = {name: os.path.join(output_dir, name) for name in PRODUCTS}
        sci_m, ref_m = _yx(sub, sub.PixA_Ctarget), _yx(sub, sub.PixA_Cresamp_object)

        def stack(path, img, var, role):
            h = _hdr(hdr, None, **prov)
            h["ROLE"] = role
            fits.HDUList([fits.PrimaryHDU(header=h), fits.ImageHDU(img.astype(np.float32), header=_hdr(hdr, bunit), name="IMAGE"),
                          fits.ImageHDU(var.astype(np.float32), header=_hdr(hdr, f"({bunit})^2"), name="VARIANCE")]).writeto(path, overwrite=True)

        stack(paths["sci_stack.fits"], _yx(sub, sub.PixA_target), _yx(sub, sub.PixA_targetVar), "science exposure on its own pixels")
        stack(paths["ref_stack.fits"], _yx(sub, sub.PixA_resamp_object), _yx(sub, sub.PixA_resamp_objectVar),
              "reference exposure resampled onto the science pixels")
        fits.HDUList([fits.PrimaryHDU(header=_hdr(hdr, None, **prov)), fits.ImageHDU(sci_m.astype(np.float32), header=_hdr(hdr, bunit), name="SCI_M"),
                      fits.ImageHDU(ref_m.astype(np.float32), header=_hdr(hdr, bunit), name="REF_M"),
                      fits.ImageHDU((sci_m - ref_m).astype(np.float32), header=_hdr(hdr, bunit), name="D_MATCHED")]).writeto(
            paths["matched.fits"], overwrite=True)
        fits.PrimaryHDU(self.fit_mask.astype(np.uint8), header=_hdr(hdr, None, **prov)).writeto(paths["fitmask.fits"], overwrite=True)
        hdr_d = _hdr(hdr, None, CHAIN=("single", "one exposure against one"), SIGDIFF=(rec["sfft"]["sigma_after"], "difference noise on source-free pixels"),
                     SFFTUSED=(bool(sub.SFFT_USED), "SFFT kernel fitted and applied"), **prov)
        if sub.SFFT_USED:
            hdr_d["FSCAL"] = (float(sub.SFFT_FSCAL), "SFFT flux scale")
            hdr_d["KCENTRE"] = (float(sub.SFFT_KERNEL_CENTRE), "SFFT kernel centre")
        hdus = [fits.PrimaryHDU(header=hdr_d), fits.ImageHDU(self.difference.astype(np.float32), header=_hdr(hdr_d, bunit), name="DIFF"),
                fits.ImageHDU(self.score.astype(np.float32), header=_hdr(hdr_d, "sigma"), name="SCORE"),
                fits.ImageHDU(self.variance.astype(np.float32), header=_hdr(hdr_d, f"({bunit})^2"), name="VARIANCE"),
                fits.ImageHDU(sci_m.astype(np.float32), header=_hdr(hdr, bunit), name="SCI_M"),
                fits.ImageHDU(ref_m.astype(np.float32), header=_hdr(hdr, bunit), name="REF_M"),
                fits.ImageHDU(self.fit_mask.astype(np.uint8), header=_hdr(hdr), name="FITMASK")]
        if self.saturation_map is not None:
            hdus.append(fits.ImageHDU(np.asarray(self.saturation_map).astype(np.uint8), header=_hdr(hdr), name="SATURATED"))
        fits.HDUList(hdus).writeto(paths["diff.fits"], overwrite=True)
        np.save(paths["psf_diff_eff.npy"], _yx(sub, sub.PSF_Ctarget))
        np.save(paths["psf_diff_eff_fine.npy"], _yx(sub, sub.PSF_Ctarget_OVERSAMPLED))
        np.savez_compressed(paths["psf_nodes.npz"], sci=_nodes_yx(sub, sub.PSF_target_NODES), ref=_nodes_yx(sub, sub.PSF_resamp_object_NODES),
                            xs=np.asarray(sub.PSF_NODE_X), ys=np.asarray(sub.PSF_NODE_Y), oversample=int(sub.PSF_WORK_OVERSAMPLING))
        rec["timings"]["write"] = time.time() - t0
        rec["output_dir"] = os.path.abspath(output_dir)
        rec["products"] = list(PRODUCTS)
        _write_json(paths["record.json"], rec)
        self.products = paths
        _logger.info("products written in %.0f s: %s", rec["timings"]["write"], output_dir)
        return paths

    # --------------------------------------------------------------------------------------------------------------------------- record
    def _settings_record(self):
        """The settings as JSON: a PSF given as an array is named by its hash and shape, never written out."""
        prep = dict(self.prep_options)
        psf = prep.get("psf")
        if isinstance(psf, dict) and "stamp" in psf:
            psf = dict(psf)
            stamp = np.ascontiguousarray(psf.pop("stamp"), dtype=np.float64)
            psf["stamp_sha256"] = hashlib.sha256(stamp.tobytes() + str(stamp.shape).encode()).hexdigest()
            psf["stamp_shape"] = list(stamp.shape)
            prep["psf"] = psf
        return dict(prep=prep, sfft=dict(GKerHW=self.GKerHW, KerPolyOrder=self.KerPolyOrder, BGPolyOrder=self.BGPolyOrder,
                                         ConstPhotRatio=self.ConstPhotRatio), mask=dict(self.mask_options))


def _yx(sub, a):
    """A public [x, y] backend array of the subtraction as numpy [y, x]."""
    return np.ascontiguousarray(np.asarray(sub.op.asnumpy(a)).T)


def _nodes_yx(sub, a):
    """Public node stamps [nodes_x, nodes_y, n, n] of the subtraction as numpy [nodes_y, nodes_x, n, n]."""
    return np.ascontiguousarray(np.asarray(sub.op.asnumpy(a)).transpose(1, 0, 3, 2))


def _hdr(base, bunit=None, **cards):
    h = base.copy()
    if bunit:
        h["BUNIT"] = bunit
    for k, v in cards.items():
        h[k] = v
    return h


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _default_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 8
    return max(1, min(MAX_DEFAULT_THREADS, n))


def _file_record(path):
    st = os.stat(path)
    return dict(path=os.path.abspath(path), name=os.path.basename(path), size=st.st_size, mtime_ns=st.st_mtime_ns)


def _versions(backend):
    import importlib.metadata as md
    import sfft
    out = dict(sfft=str(sfft.__version__), python=platform.python_version())
    for name in ("numpy", "scipy", "astropy", "drizzle", "asdf", "roman_datamodels", "stpsf") + (("cupy-cuda12x", "cupy") if backend == "gpu" else ()):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out
