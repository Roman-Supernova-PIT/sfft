"""SpaceSFFT: the subtraction of two Roman Space Telescope exposures, one resampled onto the other's pixel grid, by either of sfft's
engines.

    from sfft import SpaceSFFT

    sub = SpaceSFFT(hdr_target, hdr_object, target_skyrms, object_skyrms, PixA_target, PixA_object, PixA_targetVar, PixA_objectVar,
                    PixA_target_DMASK, PixA_object_DMASK, PSF_target, PSF_object, BACKEND_4SUBTRACT="Numpy", engine="romanL2")
    sub.resample_image_mask_psf()
    sub.cross_convolve()
    sub.sfft_subtract()
    sub.find_decorrelation()
    difference = sub.apply_decorrelation(sub.PixA_DIFF)
    variance = sub.create_variance_image()

The two engines:
  'romanL2'  sfft.spaceflow.subtract.SpaceL2_Sub, the engine for Roman WFI Level 2 exposures (the default)
  'legacy'   sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow, the subtraction of the SFFT 1.6.4 release

Both take the same constructor arguments, in the same order, and the same steps. SpaceSFFT builds the chosen engine and calls it: each step
is the engine's step, and every attribute the engine sets (PixA_DIFF, PSF_Ctarget, op, hdr_target, ...) is read from and written to the
engine. The engine object itself is the attribute sfftengine.
"""
import warnings

ENGINES = ("legacy", "romanL2")
ROMANL2_ONLY = ("PSF_OVERSAMPLING", "PSF_PIXEL_INTEGRATED", "instrument")      # keywords only the romanL2 engine takes
ROMANL2_NOT_READ_BACK = ("PixA_resamp_object", "BlankMask")                     # set by romanL2's resampling, never read by its later steps
_OWN_ATTRIBUTES = ("engine", "sfftengine")                                     # kept on the wrapper itself, never on the engine


def engine_name(engine):
    """The engine's name as in ENGINES ('legacy' or 'romanL2'), from a name in any letter case; raises ValueError for any other value."""
    for name in ENGINES:
        if str(engine).strip().lower() == name.lower():
            return name
    raise ValueError(f"engine must be one of {ENGINES} (in any letter case), not {engine!r}")


class SpaceSFFT:
    """The subtraction of a target exposure and an object exposure resampled onto the target's pixel grid, by the engine `engine`.

    Parameters
    ----------
    hdr_target, ..., RANDOM_SEED
        The arguments of SFFT 1.6.4's SpaceSFFT_Flow, in its order and with its defaults; their meaning for each engine is in that engine's
        class docstring (sfft.spaceflow.subtract.SpaceL2_Sub, sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow).
    engine : 'romanL2' or 'legacy', in any letter case, default 'romanL2'
        The engine that does the subtraction.
    PSF_OVERSAMPLING, PSF_PIXEL_INTEGRATED, instrument : optional
        romanL2's own keywords (see SpaceL2_Sub); None leaves romanL2's default. The legacy engine refuses them (TypeError).

    Attributes
    ----------
    engine : the engine's name, 'romanL2' or 'legacy'.
    sfftengine : the engine object (SpaceL2_Sub or SpaceSFFT_Flow).
    Every other attribute is the engine's: reading it reads the engine's, and setting it sets the engine's. With romanL2, setting
    PixA_resamp_object or BlankMask warns, since romanL2's later steps work from their own copies of these and do not read them back.
    """

    def __init__(self, hdr_target, hdr_object, target_skyrms, object_skyrms, PixA_target, PixA_object, PixA_targetVar, PixA_objectVar,
                 PixA_target_DMASK, PixA_object_DMASK, PSF_target, PSF_object, transpose=True, sci_is_target=True, GKerHW=9, KerPolyOrder=2,
                 BGPolyOrder=0, ConstPhotRatio=True, Consider_Matching_Kernel=False, BACKEND_4SUBTRACT="Cupy", CUDA_COMPILER="nvrtc",
                 CUDA_DEVICE_4SUBTRACT="0", NUM_CPU_THREADS_4SUBTRACT=8, NUMBA_CACHE=True, RANDOM_SEED=10086, engine="romanL2",
                 PSF_OVERSAMPLING=None, PSF_PIXEL_INTEGRATED=None, instrument=None):
        name = engine_name(engine)
        arguments = dict(hdr_target=hdr_target, hdr_object=hdr_object, target_skyrms=target_skyrms, object_skyrms=object_skyrms,
                         PixA_target=PixA_target, PixA_object=PixA_object, PixA_targetVar=PixA_targetVar, PixA_objectVar=PixA_objectVar,
                         PixA_target_DMASK=PixA_target_DMASK, PixA_object_DMASK=PixA_object_DMASK, PSF_target=PSF_target,
                         PSF_object=PSF_object, transpose=transpose, sci_is_target=sci_is_target, GKerHW=GKerHW, KerPolyOrder=KerPolyOrder,
                         BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, Consider_Matching_Kernel=Consider_Matching_Kernel,
                         BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_COMPILER=CUDA_COMPILER, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT,
                         NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, NUMBA_CACHE=NUMBA_CACHE, RANDOM_SEED=RANDOM_SEED)
        given = dict(PSF_OVERSAMPLING=PSF_OVERSAMPLING, PSF_PIXEL_INTEGRATED=PSF_PIXEL_INTEGRATED, instrument=instrument)
        romanl2_options = {key: value for key, value in given.items() if value is not None}
        if name == "legacy":
            if romanl2_options:
                raise TypeError(f"engine='legacy' does not take {sorted(romanl2_options)}: only engine='romanL2' does")
            from .legacy.SpaceSFFTFlow import SpaceSFFT_Flow
            sfftengine = SpaceSFFT_Flow(**arguments)
        else:
            from .spaceflow.subtract import SpaceL2_Sub
            sfftengine = SpaceL2_Sub(**arguments, **romanl2_options)
        object.__setattr__(self, "engine", name)
        object.__setattr__(self, "sfftengine", sfftengine)

    # ------------------------------------------------------------------------------------------------------------------ the engine's attributes
    def __getattr__(self, name):
        # called only for a name the wrapper does not have itself; its own two attributes are never looked up on the engine
        if name in _OWN_ATTRIBUTES:
            raise AttributeError(f"SpaceSFFT has no attribute {name!r} yet: the engine is not built")
        return getattr(self.sfftengine, name)

    def __setattr__(self, name, value):
        if name in _OWN_ATTRIBUTES:
            raise AttributeError(f"SpaceSFFT.{name} is set when the object is built and cannot be changed")
        if self.engine == "romanL2" and name in ROMANL2_NOT_READ_BACK:
            warnings.warn(f"engine='romanL2' does not read {name} back: its later steps work from their own resampled images, so the value "
                          "set here is not used", stacklevel=2)
        setattr(self.sfftengine, name, value)

    # ------------------------------------------------------------------------------------------------------------------------------ the steps
    def resample_image_mask_psf(self):
        """Step 0: the object exposure resampled onto the target grid, with its mask and PSF."""
        return self.sfftengine.resample_image_mask_psf()

    def cross_convolve(self):
        """Step 1: each image convolved with the other's PSF, so that both carry one PSF."""
        return self.sfftengine.cross_convolve()

    def sfft_subtract(self, fit_mask=None):
        """Step 2: SFFT's matching kernel fitted and the difference formed. fit_mask (the pixels the fit may use) only with engine='romanL2'."""
        if fit_mask is None:
            return self.sfftengine.sfft_subtract()
        if self.engine == "legacy":
            raise TypeError("engine='legacy' does not take fit_mask: only engine='romanL2' does")
        return self.sfftengine.sfft_subtract(fit_mask=fit_mask)

    def find_decorrelation(self):
        """Step 3: the decorrelation filter of the difference (the identity with engine='romanL2')."""
        return self.sfftengine.find_decorrelation()

    def apply_decorrelation(self, img):
        """Step 4: the decorrelation filter applied to an image or a PSF stamp."""
        return self.sfftengine.apply_decorrelation(img)

    def create_score_image(self):
        """The detection image of the difference, in units of its noise."""
        return self.sfftengine.create_score_image()

    def create_variance_image(self):
        """The variance image of the difference."""
        return self.sfftengine.create_variance_image()

    def __repr__(self):
        return f"SpaceSFFT(engine={self.engine!r}, sfftengine={type(self.sfftengine).__name__})"
