"""SpaceL2_Sub: image subtraction of two space-telescope exposures, one resampled onto the other's pixel grid.

The workflow: the object exposure is resampled onto the target exposure's own pixel grid with drizzle (square kernel, pixel fraction 1: each
target pixel the overlap-weighted mean of the surface brightness of the object pixels it overlaps, on the CPU with either backend; module
sfft.spaceflow.subtract.resample); each exposure's PSF is turned into a working model on the target grid at several positions
(the object's model includes the resampling, built from the sub-pixel offsets of its pixel lattice present there); the two images are matched
by an exact cross-convolution with one common denominator, built from the two PSF models with every product summed over the frequencies the
pixel grid folds together, so that a static source cancels and the noise stays white; SFFT then fits a small, spatially varying kernel on a fit
mask of the matched pair and subtracts. The difference is science minus reference, so a transient in the science exposure is positive.

Steps, in order: resample_image_mask_psf(), cross_convolve(), sfft_subtract(), find_decorrelation(), then any of apply_decorrelation(),
create_score_image(), create_variance_image(). The class replaces release 1.6.4's SpaceSFFT_Flow (kept as
sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow) and keeps its constructor arguments, step names and output conventions; see the class docstring for
what each one means now. sfft.spaceflow.SpaceL2_Flow runs it from two Level 2 files.
"""
import logging
import warnings

import numpy as np

__last_update__ = "2026-11-23"
__author__ = "Lei Hu <leihu@andrew.cmu.edu> and Michael Wood-Vasey <wmwv@pitt.edu>"

NODES = 5                   # PSF positions per axis over the target grid when a PSF is given as one 2-D array
WORK_OVERSAMPLING = 5       # samples per target pixel of the working PSF models
WORK_HALF_WIDTH = 48        # half-width of the working PSF models [target pixels]
OLD_RULE_COVER = 0.1        # the old fit rule: the resampled object's detection mask must cover this much of a target pixel
_logger = logging.getLogger(__name__)


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _per_side(value, name, kind):
    pair = tuple(value) if isinstance(value, (tuple, list)) else (value, value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be one value or a pair (target, object), not {value!r}")
    if kind == "int":
        for v in pair:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)) or v != int(v) or v < 1:
                raise ValueError(f"{name} must be a whole number of samples per native pixel, 1 or more; got {value!r}")
        return tuple(int(v) for v in pair)
    for v in pair:
        if not isinstance(v, (bool, np.bool_)):
            raise ValueError(f"{name} must be True or False, or a pair of them; got {value!r}")
    return tuple(bool(v) for v in pair)


class _Operations:
    """The array functions of the chosen backend (attribute `op`, as in release 1.6.4)."""

    def __init__(self, backend, transpose):
        if backend == "Cupy":
            import cupy as xp
            self.asnumpy = xp.asnumpy
        else:
            xp = np
            self.asnumpy = np.asarray
        self.array, self.asarray, self.ascontiguousarray = xp.array, xp.asarray, xp.ascontiguousarray
        self.conj, self.fft, self.isnan, self.logical_and, self.logical_or, self.sum = xp.conj, xp.fft, xp.isnan, xp.logical_and, xp.logical_or, xp.sum
        self.transpose_if_needed = np.transpose if transpose else (lambda x: x)


def _complete_header(header):
    """The header in the one form the flow keeps: with the four CD keys. A header without them (PC + CDELT keys, as astropy writes a WCS,
    leaving out PC elements equal to their defaults) gets them from its WCS, with the PC and CDELT keys removed; a header with all four is
    returned as it is."""
    keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if all(k in header for k in keys):
        return header
    from astropy.wcs import WCS
    w = WCS(header)
    cd = np.array(w.wcs.cd) if w.wcs.has_cd() else np.array(w.wcs.get_pc()) * np.array(w.wcs.cdelt)[:, None]
    out = header.copy()
    for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2"):
        out.remove(k, ignore_missing=True)
    for k, v in zip(keys, cd.ravel()):
        out[k] = float(v)
    return out


class _Exposure:
    """What the workflow reads from an exposure: its header and WCS, its image and variance ([y, x]) and the do-not-use pixels (non-finite
    image or variance; image 0 and variance infinite there)."""

    def __init__(self, header, image, variance, name):
        from astropy.wcs import WCS
        self.header = header
        self.wcs = WCS(header)
        self.shape = image.shape
        bad = ~np.isfinite(image) | ~np.isfinite(variance)
        self.dnu = bad
        self.image = np.where(bad, 0.0, image)
        self.variance = np.where(bad, np.inf, variance)
        self.meta = dict(filename=name, detector="", optical_element="")
        self.wcs_offset_vector = None


class SpaceL2_Sub:
    """Image subtraction of a target exposure and an object exposure resampled onto the target's pixel grid, on the GPU (Cupy) or the CPU (Numpy).

    Parameters (the order and defaults of release 1.6.4; PSF_OVERSAMPLING, PSF_PIXEL_INTEGRATED and instrument are new)
    ----------
    hdr_target, hdr_object : astropy.io.fits.Header
        The two exposures' headers with their WCS (TAN, TAN-SIP or TPV; CD keys, or PC + CDELT keys with default elements left out, as
        astropy writes them) and NAXIS1, NAXIS2. The target's pixel grid is the grid of every product.
    target_skyrms, object_skyrms : float
        Accepted for compatibility and kept as attributes; not used (the noise levels are measured on source-free pixels).
    PixA_target, PixA_object : 2-D arrays (numpy or cupy)
        Sky-subtracted surface brightness of each exposure on its own grid. A NaN marks a pixel without a usable value.
    PixA_targetVar, PixA_objectVar : 2-D arrays
        Their variances; a NaN also marks a pixel without a usable value.
    PixA_target_DMASK, PixA_object_DMASK : 2-D arrays
        Detection masks (true or nonzero on sources), each on its own exposure's grid. Used only when sfft_subtract() is given no fit mask.
    PSF_target, PSF_object : 2-D or 4-D arrays
        Each exposure's PSF in its own detector frame, centred where its WCS places a point source, with odd sides. 2-D: one PSF for the whole
        exposure. 4-D: [nodes_y, nodes_x, n, n], the PSF at the centres of a regular grid of nodes_y x nodes_x cells over the target image;
        a node the exposure does not cover may hold NaN. A PSF array's sum is kept as the share of a point source's light it holds.
    transpose : bool, default True
        True: every image, mask and PSF stamp is [y, x] (FITS order); False: [x, y]. Public array attributes are always [x, y]; the arrays
        apply_decorrelation, create_score_image and create_variance_image return are in the caller's orientation.
    sci_is_target : bool, default True
        True: the target is the science exposure and the object the reference; False: the reverse. Either way the difference is science minus
        reference.
    GKerHW : int, default 9
        Half-width of SFFT's matching kernel. After the cross-convolution the kernel only corrects what the PSF models miss; 2 is enough for
        well-modelled PSFs.
    KerPolyOrder, BGPolyOrder : int, defaults 2 and 0
        Polynomial degrees of the kernel's and the differential background's spatial variation. The kernel's degree is lowered when the fit
        mask cannot constrain it. A background term cannot be fitted on a sparse fit mask; remove backgrounds before.
    ConstPhotRatio : bool, default True
        One flux scale over the field (True) or one that follows the kernel's spatial variation (False).
    Consider_Matching_Kernel : bool, default False
        Must be False: the matched pair has white noise already, and there is no decorrelation to compute.
    BACKEND_4SUBTRACT : str, default 'Cupy'
        'Cupy' ('cupy', 'cp'): GPU; 'Numpy' ('numpy', 'np'): CPU. The resampling (drizzle) runs on the CPU with either backend.
    CUDA_COMPILER : str
        Accepted for compatibility; not used.
    CUDA_DEVICE_4SUBTRACT : str, default '0'
        The GPU used by every step.
    NUM_CPU_THREADS_4SUBTRACT : int, default 8
        Threads of the CPU backend (resampling, FFTs, SFFT).
    NUMBA_CACHE : bool, default True
        Accepted for compatibility; not used (the CPU solver's compiled loops are always cached).
    RANDOM_SEED : int
        Accepted for compatibility; not used.
    PSF_OVERSAMPLING : int or (int, int), default 1
        Samples per native pixel of PSF_target and PSF_object (one value for both, or a pair). Two samples per pixel or more are recommended
        for an undersampled PSF (four for the bluest filters); with one, PSF fits on the difference come out about 2 % low.
    PSF_PIXEL_INTEGRATED : bool or (bool, bool), default True
        Whether each PSF array already includes its exposure's pixel response (an empirical PSF, or any PSF sampled at the pixel scale), or
        not (an optics model).
    instrument : str, default 'Roman-WFI'
        The instrument that took both exposures, one of sfft.spaceflow.instruments.INSTRUMENTS. The steps of the subtraction are the same for
        any exposure with a FITS WCS; the name is checked and kept, and settings that come to differ between instruments will depend on it.

    Public attributes (backend arrays in [x, y], set by the step named)
    -----------------
    __init__: PixA_target, PixA_object, PixA_targetVar, PixA_objectVar, PixA_target_DMASK, PixA_object_DMASK, PSF_target, PSF_object, op,
        instrument (the instrument's name, as in INSTRUMENTS).
    resample_image_mask_psf: PixA_resamp_object, PixA_resamp_objectVar, PixA_resamp_object_DMASK (computed when first read), BlankMask
        (pixels where either image has no value), PSF_resamp_object (the object's working model at the central node, WORK_OVERSAMPLING samples
        per pixel), PSF_target_NODES and PSF_resamp_object_NODES (all working models, [nodes_x, nodes_y, n, n]), PSF_NODE_X, PSF_NODE_Y (node
        centres, 0-based target pixels), PSF_FWHM (the science model's, pixels), PSF_WORK_OVERSAMPLING.
    cross_convolve: PixA_Ctarget, PixA_Cresamp_object (the matched images), PSF_Ctarget (a transient's PSF in the matched images and the
        difference, 41 x 41 pixels, unit total flux), PSF_Ctarget_OVERSAMPLED (the same at WORK_OVERSAMPLING samples per pixel), SourceFreeMask.
    sfft_subtract: PixA_DIFF (NaN where either image has no value), Solution, SOLUTION_SHAPE, FitMask, SFFT_USED, SFFT_PROBLEM, SFFT_FSCAL,
        SFFT_KERNEL_CENTRE, SFFT_DEGREE, SFFT_FIT_PIXELS. The solver works on a zero-padded grid, the image at its origin: Solution belongs
        to that grid, of shape SOLUTION_SHAPE (x, y), so read it with N0, N1 = SOLUTION_SHAPE.
    find_decorrelation: MATCH_KERNEL (SFFT's kernel at the centre of the solve grid, in [x, y], summing to the flux scale), FKDECO (the
        identity filter).
    """

    def __init__(self, hdr_target, hdr_object, target_skyrms, object_skyrms, PixA_target, PixA_object, PixA_targetVar, PixA_objectVar,
                 PixA_target_DMASK, PixA_object_DMASK, PSF_target, PSF_object, transpose=True, sci_is_target=True, GKerHW=9, KerPolyOrder=2,
                 BGPolyOrder=0, ConstPhotRatio=True, Consider_Matching_Kernel=False, BACKEND_4SUBTRACT="Cupy", CUDA_COMPILER="nvrtc",
                 CUDA_DEVICE_4SUBTRACT="0", NUM_CPU_THREADS_4SUBTRACT=8, NUMBA_CACHE=True, RANDOM_SEED=10086, PSF_OVERSAMPLING=1,
                 PSF_PIXEL_INTEGRATED=True, instrument="Roman-WFI"):
        from ..instruments import instrument_name
        from . import backend as be
        self.instrument = instrument_name(instrument)
        lowercase_backend = BACKEND_4SUBTRACT.strip().lower()
        backend_map = {"numpy": "Numpy", "np": "Numpy", "cupy": "Cupy", "cp": "Cupy"}
        standardized_backend = backend_map.get(lowercase_backend)
        if standardized_backend is None:
            raise ValueError(
                "Unsupported BACKEND_4SUBTRACT '%s'. Valid options: numpy, Numpy, np, cupy, Cupy, cp"
                % BACKEND_4SUBTRACT
            )
        try:
            from drizzle.resample import Drizzle  # noqa: F401
        except ImportError as e:
            raise ImportError("SpaceL2_Sub resamples with drizzle 3.0 or later (pip install 'drizzle>=3.0'), which is not installed") from e
        if Consider_Matching_Kernel:
            raise NotImplementedError("Consider_Matching_Kernel=True is not supported: the matched pair has white noise, and no decorrelation "
                                      "filter is computed")
        self.BACKEND_4SUBTRACT = standardized_backend
        self._backend = "gpu" if standardized_backend == "Cupy" else "cpu"
        self.CUDA_COMPILER = CUDA_COMPILER
        self.CUDA_DEVICE_4SUBTRACT = CUDA_DEVICE_4SUBTRACT
        self.NUM_CPU_THREADS_4SUBTRACT = NUM_CPU_THREADS_4SUBTRACT
        self.NUMBA_CACHE = NUMBA_CACHE
        self.RANDOM_SEED = RANDOM_SEED
        self._device = int(CUDA_DEVICE_4SUBTRACT)
        if self._backend == "cpu":
            self._threads = be.configure_cpu(int(NUM_CPU_THREADS_4SUBTRACT), log=_logger.info)
        else:
            why = be.gpu_problem()
            if why:
                raise RuntimeError(f"BACKEND_4SUBTRACT='Cupy', but the GPU cannot be used here: {why}")
            be.use_device("gpu", self._device)
            self._threads = int(NUM_CPU_THREADS_4SUBTRACT)
        self.transpose = bool(transpose)
        self.op = _Operations(self.BACKEND_4SUBTRACT, self.transpose)
        self.hdr_target, self.hdr_object = hdr_target, hdr_object
        self.target_skyrms, self.object_skyrms = target_skyrms, object_skyrms
        self.sci_is_target = bool(sci_is_target)
        self.GKerHW, self.KerPolyOrder, self.BGPolyOrder = int(GKerHW), int(KerPolyOrder), int(BGPolyOrder)
        self.ConstPhotRatio = bool(ConstPhotRatio)
        self.Consider_Matching_Kernel = Consider_Matching_Kernel
        self.CROSS_CONVOLVED = False
        self._oversampling = _per_side(PSF_OVERSAMPLING, "PSF_OVERSAMPLING", "int")
        self._pixel_integrated = _per_side(PSF_PIXEL_INTEGRATED, "PSF_PIXEL_INTEGRATED", "bool")

        img_t, img_o = self._input(PixA_target), self._input(PixA_object)
        var_t, var_o = self._input(PixA_targetVar), self._input(PixA_objectVar)
        nx_t, ny_t = int(hdr_target["NAXIS1"]), int(hdr_target["NAXIS2"])
        if img_t.shape != (ny_t, nx_t) or var_t.shape != img_t.shape:
            raise ValueError(f"PixA_target and PixA_targetVar must have the shape hdr_target gives, NAXIS2 x NAXIS1 = {ny_t} x {nx_t} in [y, x]; "
                             f"got {img_t.shape} and {var_t.shape}")
        if img_o.shape != (int(hdr_object["NAXIS2"]), int(hdr_object["NAXIS1"])) or var_o.shape != img_o.shape:
            raise ValueError("PixA_object and PixA_objectVar must have the shape hdr_object gives (NAXIS2 x NAXIS1 in [y, x])")
        self._dmask_t = self._input(PixA_target_DMASK, None) != 0
        self._dmask_o = self._input(PixA_object_DMASK, None) != 0
        if self._dmask_t.shape != img_t.shape or self._dmask_o.shape != img_o.shape:
            raise ValueError("each detection mask must have its own image's shape")
        self._target = _Exposure(_complete_header(hdr_target), img_t, var_t, "target")
        self._object = _Exposure(_complete_header(hdr_object), img_o, var_o, "object")
        self._psf = dict(target=self._psf_input(PSF_target, "PSF_target"), object=self._psf_input(PSF_object, "PSF_object"))
        grids = {p.shape[:2] for p in self._psf.values() if p.ndim == 4}
        if len(grids) > 1:
            raise ValueError(f"PSF_target and PSF_object give the PSF at different numbers of nodes: {sorted(grids)}")
        self._nodes = grids.pop() if grids else (NODES, NODES)

        self.PixA_target, self.PixA_object = self._pub(img_t), self._pub(img_o)
        self.PixA_targetVar, self.PixA_objectVar = self._pub(var_t), self._pub(var_o)
        self.PixA_target_DMASK = self._pub(self._dmask_t.astype(np.float64))
        self.PixA_object_DMASK = self._pub(self._dmask_o.astype(np.float64))
        self.PSF_target, self.PSF_object = self._pub(self._psf["target"]), self._pub(self._psf["object"])
        self._stacks = self._models = self._matched = self._pair_cache = self._variance_image = None
        self._resampled_dmask = None
        self._sfft = None

    # ------------------------------------------------------------------ conversions
    def _input(self, a, dtype=np.float64):
        a = _to_numpy(a)
        a = a if self.transpose else a.T
        return np.ascontiguousarray(a, dtype=dtype) if dtype is not None else np.ascontiguousarray(a)

    def _psf_input(self, psf, name):
        p = np.asarray(_to_numpy(psf), np.float64)
        if p.ndim == 2:
            p = p if self.transpose else p.T
        elif p.ndim == 4:
            p = p if self.transpose else p.transpose(1, 0, 3, 2)
        else:
            raise ValueError(f"{name} must be a 2-D PSF or a 4-D array of PSFs at nodes, not {p.ndim}-D")
        if p.shape[-1] % 2 == 0 or p.shape[-2] % 2 == 0:
            raise ValueError(f"{name}'s stamps must have odd sides, with the PSF centred on the centre sample; got {p.shape[-2:]}")
        return np.ascontiguousarray(p)

    def _pub(self, a):
        """A [y, x] array (or [ny, nx, n, n] node stamps) as a public attribute: [x, y] on the backend."""
        a = np.asarray(a)
        t = a.T if a.ndim == 2 else a.transpose(1, 0, 3, 2)
        return self.op.ascontiguousarray(self.op.asarray(t))

    def _caller(self, a):
        """A [y, x] array in the caller's orientation, as numpy."""
        return np.ascontiguousarray(a if self.transpose else a.T)

    def _side(self, name):
        """'sci' or 'ref' -> 'target' or 'object'."""
        return {"sci": "target", "ref": "object"}[name] if self.sci_is_target else {"sci": "object", "ref": "target"}[name]

    # ------------------------------------------------------------------ step 0
    def _node_models(self, exposure, stamps, oversample, pixel_integrated, resampled, info):
        from . import psfmodel as pm

        def node_psf(j, i):
            st = stamps[j, i] if stamps.ndim == 4 else stamps
            if not np.isfinite(st).all():
                raise ValueError(f"the PSF given for node ({j}, {i}) holds non-finite values, but its exposure covers that node")
            return pm.PsfSample(st, oversample, pixel_integrated)

        return pm.node_stamps([exposure], self._target.header, self._xs, self._ys, None, WORK_OVERSAMPLING, WORK_HALF_WIDTH, resampled=resampled,
                              present_offsets=resampled, info=info, node_psf=node_psf)

    def resample_image_mask_psf(self):
        """Step 0: the object exposure resampled onto the target grid (image and variance; the target exposure as it is), and the two
        exposures' working PSF models at the nodes of the target grid. Raises ValueError when the object does not overlap the target."""
        from . import crossconv as cc, psfmodel as pm, resample as rs
        hdr = self._target.header
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        stacks = dict(target=rs.exposure_on_own_grid(self._target, hdr), object=rs.stack([self._object], hdr))
        if np.isfinite(stacks["object"]["image"]).sum() < 1:
            raise ValueError("SFFT Error: Projection of object image is completely outside of target image!")
        nny, nnx = self._nodes
        self._xs, self._ys = cc.node_grid((ny, nx), nnx)[0], cc.node_grid((ny, nx), nny)[1]
        self._model_info = []
        models = dict(target=self._node_models(self._target, self._psf["target"], self._oversampling[0], self._pixel_integrated[0], False, None),
                      object=self._node_models(self._object, self._psf["object"], self._oversampling[1], self._pixel_integrated[1], True,
                                               self._model_info))
        self._centre = (nny // 2, nnx // 2)
        self._fwhm = pm.fwhm_px(models[self._side("sci")][self._centre[0]][self._centre[1]], WORK_OVERSAMPLING)
        self._stacks, self._models = stacks, models
        self._resampled_dmask = self._pair_cache = self._matched = self._sfft = self._variance_image = None
        self.CROSS_CONVOLVED = False
        self.PixA_resamp_object = self._pub(stacks["object"]["image"])
        self.PixA_resamp_objectVar = self._pub(stacks["object"]["variance"])
        self.BlankMask = self._pub(~(np.isfinite(stacks["object"]["image"]) & np.isfinite(stacks["target"]["image"])))
        self.PSF_resamp_object = self._pub(models["object"][self._centre[0]][self._centre[1]])
        self.PSF_target_NODES = self._pub(np.array(models["target"]))
        self.PSF_resamp_object_NODES = self._pub(np.array(models["object"]))
        self.PSF_NODE_X, self.PSF_NODE_Y = np.asarray(self._xs), np.asarray(self._ys)
        self.PSF_FWHM = float(self._fwhm)
        self.PSF_WORK_OVERSAMPLING = WORK_OVERSAMPLING
        _logger.info("resampled the object onto the target grid (%d x %d); working PSF models at %d x %d nodes, science FWHM %.2f px",
                     nx, ny, nnx, nny, self._fwhm)

    def _resampled_object_dmask(self):
        """The object's detection mask on the target grid, as the share of each target pixel it covers."""
        from . import resample as rs
        if self._resampled_dmask is None:
            if self._dmask_o.all():
                m = np.isfinite(self._stacks["object"]["image"]).astype(np.float64)
            else:
                E = _Exposure(self._object.header, np.where(self._object.dnu, np.nan, self._dmask_o.astype(np.float64)),
                              np.where(self._object.dnu, np.nan, 0.0), "object detection mask")
                m = np.nan_to_num(rs.stack([E], self._target.header)["image"])
            self._resampled_dmask = m
        return self._resampled_dmask

    @property
    def PixA_resamp_object_DMASK(self):
        if self._stacks is None:
            raise AttributeError("PixA_resamp_object_DMASK exists after resample_image_mask_psf()")
        return self._pub(self._resampled_object_dmask())

    # ------------------------------------------------------------------ step 1
    def _pair(self):
        """The science and reference images on the target grid, where both have values, with their levels on source-free pixels removed;
        the source-free pixels (2 and 3 sigma) and the two noise levels."""
        from . import crossconv as cc
        if getattr(self, "_pair_cache", None) is not None:
            return self._pair_cache
        st = self._stacks
        sci, ref = st[self._side("sci")]["image"], st[self._side("ref")]["image"]
        both = np.isfinite(ref) & np.isfinite(sci)
        s_img = np.where(both, sci, np.nan)
        r_img = np.where(both, ref, np.nan)
        free = cc.source_free([s_img, r_img], self._fwhm)
        free3 = cc.source_free([s_img, r_img], self._fwhm, nsig=3.0)
        off_s, off_r = float(np.median(s_img[free])), float(np.median(r_img[free]))
        s_img, r_img = s_img - off_s, r_img - off_r
        sig_s, sig_r = cc.robust_sigma(s_img[free]), cc.robust_sigma(r_img[free])
        self._pair_cache = dict(both=both, free=free, free3=free3, s_img=s_img, r_img=r_img, levels=(off_s, off_r), sigmas=(sig_s, sig_r))
        return self._pair_cache

    def cross_convolve(self):
        """Step 1: the exact cross-convolution of the science and reference images with one common denominator (the science image's noise for
        both), its filters built at each node from the two working PSF models with every product summed over the folded frequencies and
        interpolated bilinearly between nodes. Both matched images then carry one PSF (PSF_Ctarget) and white noise."""
        from . import crossconv as cc, score as sc
        if self._stacks is None:
            raise RuntimeError("run resample_image_mask_psf() first")
        self._pair_cache = None
        p = self._pair()
        psf_s, psf_r = self._models[self._side("sci")], self._models[self._side("ref")]
        sig = (p["sigmas"][0], p["sigmas"][0])
        kernels = {}
        m = cc.match_pair(p["s_img"], p["r_img"], psf_s, psf_r, WORK_OVERSAMPLING, sig[0], sig[1], xs=self._xs, ys=self._ys, backend=self._backend,
                          taper=None, fold=True, filter_kernels=kernels)
        cj, ci = self._centre
        p_eff = sc.effective_psf(m["otf_eff"][(cj, ci)], m["lattice"])
        p_fine = cc.effective_psf_fine(psf_s[cj][ci], psf_r[cj][ci], WORK_OVERSAMPLING, sig[0], sig[1], lattice=m["lattice"], taper=None, fold=True)
        self._matched = dict(m=m, p_eff=p_eff, p_fine=p_fine, d0=m["sci"] - m["ref"], filter_kernels=kernels)
        self._variance_image = self._sfft = None
        matched = dict(sci=m["sci"], ref=m["ref"])
        self.PixA_Ctarget = self._pub(matched["sci" if self.sci_is_target else "ref"])
        self.PixA_Cresamp_object = self._pub(matched["ref" if self.sci_is_target else "sci"])
        self.PSF_Ctarget = self._pub(p_eff)
        self.PSF_Ctarget_OVERSAMPLED = self._pub(p_fine)
        self.SourceFreeMask = self._pub(p["free"])
        self.CROSS_CONVOLVED = True
        _logger.info("matched the pair: noise science %.4g, reference %.4g", *p["sigmas"])

    # ------------------------------------------------------------------ step 2
    def sfft_subtract(self, fit_mask=None):
        """Step 2: SFFT's kernel fitted on the fit mask and the difference, science minus convolved reference.

        fit_mask : 2-D array on the target grid (caller's orientation), optional
            The pixels SFFT's fit may use. None: the old rule, pixels where the target's detection mask is set and the object's, resampled,
            covers at least OLD_RULE_COVER of the pixel. Pixels without a value in either image never enter the fit.

        Without cross_convolve() first, SFFT runs on the unmatched pair (levels removed). When the fit cannot be constrained or its result is
        not sane, the matched difference is the product, SFFT_USED is False and SFFT_PROBLEM says why (also as a warning). Any other error in
        the solve is raised."""
        from . import backend as be, solve as subtract
        if self._stacks is None:
            raise RuntimeError("run resample_image_mask_psf() first")
        p = self._pair()
        if self.CROSS_CONVOLVED:
            sci_w, ref_w = self._matched["m"]["sci"], self._matched["m"]["ref"]
        else:
            sci_w, ref_w = p["s_img"], p["r_img"]
        if fit_mask is None:
            fit = self._dmask_t & (self._resampled_object_dmask() >= OLD_RULE_COVER)
        else:
            fit = self._input(fit_mask, None) != 0
            if fit.shape != sci_w.shape:
                raise ValueError(f"fit_mask must be on the target grid, {sci_w.shape} in [y, x]; got {fit.shape}")
        r = subtract.guarded(sci_w, ref_w, fit, p["free"], gker_hw=self.GKerHW, kernel_degree=self.KerPolyOrder, bg_degree=self.BGPolyOrder,
                             const_phot_ratio=self.ConstPhotRatio, backend=be.sfft_backend(self._backend), device=self._device,
                             threads=self._threads, numba_cache=self.NUMBA_CACHE)
        if r["used"]:
            keys = [(j, i) for j in range(len(self._ys)) for i in range(len(self._xs))]
            ks = subtract.kernels_at(r["solution"], r["solve_shape"], [(self._xs[i], self._ys[j]) for j, i in keys], r["degree"], self.BGPolyOrder,
                                     self.GKerHW)
            r["node_kernels"] = dict(zip(keys, ks))
        else:
            r["node_kernels"] = None
        self._sfft = r
        self._fit = fit
        self._variance_image = None
        self.PixA_DIFF = self._pub(r["diff"])
        self.Solution = None if r["solution"] is None else self.op.asarray(r["solution"])
        self.SOLUTION_SHAPE = r["solve_shape"]
        self.FitMask = self._pub(fit)
        self.SFFT_USED, self.SFFT_PROBLEM = bool(r["used"]), r["problem"]
        self.SFFT_FSCAL, self.SFFT_KERNEL_CENTRE, self.SFFT_DEGREE = r["fscal"], r["kernel_centre"], r["degree"]
        self.SFFT_FIT_PIXELS = int(r["fit_pixels"])
        if r["problem"]:
            warnings.warn(r["problem"], stacklevel=2)
        else:
            _logger.info("SFFT: flux scale %.5f, kernel centre %.3f (degree %d), %d fit pixels", r["fscal"], r["kernel_centre"], r["degree"],
                         r["fit_pixels"])

    # ------------------------------------------------------------------ steps 3-5
    def find_decorrelation(self):
        """Step 3: nothing is left to decorrelate, since the matched pair has white noise at one level. Sets MATCH_KERNEL and FKDECO, the
        identity filter, so that apply_decorrelation() can be called as before."""
        if self._sfft is None:
            raise RuntimeError("run sfft_subtract() first")
        if self.Solution is not None:
            self.Solution = self.op.asnumpy(self.Solution)
        self.MATCH_KERNEL = None if self._sfft["kernel_physical"] is None else np.asarray(self._sfft["kernel_physical"])
        ny, nx = self._sfft["diff"].shape
        self.FKDECO = np.ones((nx, ny), dtype=np.complex128)

    def apply_decorrelation(self, img):
        """Step 4: the decorrelation filter (the identity) applied to an image or a PSF stamp given as a public attribute ([x, y]); returned as
        numpy in the caller's orientation."""
        return np.array(self.op.transpose_if_needed(np.asarray(self.op.asnumpy(img))), dtype=np.float64)

    def create_score_image(self):
        """The matched-filter detection image of the difference in units of the local noise (caller's orientation; NaN where the difference
        has no value): the difference correlated with PSF_Ctarget, divided by the noise the variance image predicts through the filter
        relative to the sky's, then by its scatter on source-free pixels."""
        from . import score as sc
        if self._sfft is None or self._matched is None:
            raise RuntimeError("create_score_image() needs cross_convolve() and sfft_subtract() first")
        score, self._score_noise = sc.score_image(self._sfft["diff"], self._matched["p_eff"], self._pair()["free3"], self._variance(),
                                                  workers=self._threads)
        return self._caller(score)

    def _variance(self):
        """The two input variance maps summed: the relative noise map of the detection image (the baseline's), not the difference's variance."""
        st = self._stacks
        return st["target"]["variance"] + st["object"]["variance"]

    def create_variance_image(self):
        """The variance of the difference PixA_DIFF on the target grid (caller's orientation; NaN where the difference has no value): the
        science and reference variance maps propagated through the matching filters of each PSF node and, on the reference side, through
        SFFT's kernel at the node (after sfft_subtract()), V = V_sci * a^2 + V_ref * (K * b)^2, blended between nodes as the filters are
        (sfft.spaceflow.subtract.variance.propagated_variance). Pixels are taken as independent: the resampled image's pixel-to-pixel correlation is
        not included, and the level is only as right as the input variances (PixA_targetVar, PixA_objectVar)."""
        from . import variance
        if self._stacks is None:
            raise RuntimeError("run resample_image_mask_psf() first")
        if self._variance_image is None:
            st = self._stacks
            kernels = self._matched["filter_kernels"] if self.CROSS_CONVOLVED else None
            sfft_kernels = self._sfft["node_kernels"] if self._sfft is not None else None
            V = variance.propagated_variance(st[self._side("sci")]["variance"], st[self._side("ref")]["variance"], self._xs, self._ys, kernels,
                                             sfft_kernels, workers=self._threads)
            if self._sfft is not None:
                V[~np.isfinite(self._sfft["diff"])] = np.nan
            self._variance_image = V
        return self._caller(self._variance_image)
