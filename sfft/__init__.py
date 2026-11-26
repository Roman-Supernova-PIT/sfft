"""betasfft: image subtraction with SFFT (Saccadic Fast Fourier Transform), with two engines.

  sfft.SpaceSFFT  the subtraction of two Roman Space Telescope exposures by either engine, chosen with engine='romanL2' (the default)
                  or engine='legacy'; sfft.SpaceSFFTFlow.SpaceSFFT_Flow is the same class under the release's name
  sfft.spaceflow  the romanL2 engine, for Roman WFI Level 2 exposures given as arrays: SpaceL2_Flow differences two exposures end to end,
                  through SpaceL2_Prep (input preparation, sfft.spaceflow.prep) and SpaceL2_Sub (resampling, matching and SFFT,
                  sfft.spaceflow.subtract)
  sfft.legacy     the SFFT release this package started from (the Roman SN PIT fork of SFFT 1.6.4): the customized, sparse and crowded
                  packets, their solver (sfft.legacy.sfftcore), their tools (sfft.legacy.utils) and the release's SpaceSFFT_Flow
                  (sfft.legacy.SpaceSFFTFlow)

    from sfft import SpaceSFFT
    from sfft.spaceflow import SpaceL2_Flow
    from sfft.legacy import Customized_Packet

`import sfft` imports neither engine; each is imported when it is first used.

The release's import paths keep working: sfft.utils.StampGenerator, sfft.sfftcore.SFFTSubtract, sfft.CustomizedPacket, ... (LEGACY_MODULES)
are the modules of sfft.legacy under their old names, and `from sfft import Customized_Packet` (LEGACY_NAMES) gives the sfft.legacy class.
Each gives a DeprecationWarning naming the sfft.legacy path to use instead.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

from .SpaceSFFT import SpaceSFFT

__version__ = "1.6.4.dev21"
__all__ = ["SpaceSFFT"]

# The release's modules, now in sfft.legacy: sfft.<name> is the module sfft.legacy.<name>. sfft.SpaceSFFTFlow is not among them: it is a
# module of its own, holding the wrapper under the release's name.
LEGACY_MODULES = (
    "AutoCrowdedPrep", "AutoSparsePrep", "BSplineSFFT", "CustomizedPacket", "EasyCrowdedPacket", "EasySparsePacket",
    "MultiEasyCrowdedPacket", "MultiEasySparsePacket", "PureCupyCustomizedPacket",
    "sfftcore", "sfftcore.SFFTConfigure", "sfftcore.SFFTSubtract",
    "utils", "utils.CombineHeader", "utils.ConvKernelConvertion", "utils.CupyWCSTransform", "utils.DeCorrelationCalculator",
    "utils.DeCorrelationCalculatorDep", "utils.HoughDetection", "utils.HoughMorphClassifier", "utils.ImageZoomRotate",
    "utils.NeighboringPixelCovariance", "utils.NumpyFFTKits", "utils.NumpyResampKits", "utils.PatternRotationCalculator",
    "utils.PureCupyDeCorrelationCalculator", "utils.PureCupyFFTKits", "utils.ReadWCS", "utils.ResampKits", "utils.SExSkySubtract",
    "utils.SFFTSolutionReader", "utils.SkyLevelEstimator", "utils.StampGenerator", "utils.SymmetricMatch", "utils.WeightedQuantile",
    "utils.meta", "utils.meta.MultiProc", "utils.meta.TimeoutKit",
    "utils.pyAstroMatic", "utils.pyAstroMatic.AMConfigMaker", "utils.pyAstroMatic.PYSEx", "utils.pyAstroMatic.PYSWarp",
)

# The release's top-level names: `from sfft import <name>` gives the class of the module sfft.legacy.<module>.
LEGACY_NAMES = {
    "PureCupy_Customized_Packet": "PureCupyCustomizedPacket", "Customized_Packet": "CustomizedPacket",
    "Auto_SparsePrep": "AutoSparsePrep", "Auto_CrowdedPrep": "AutoCrowdedPrep",
    "Easy_SparsePacket": "EasySparsePacket", "Easy_CrowdedPacket": "EasyCrowdedPacket",
    "MultiEasy_SparsePacket": "MultiEasySparsePacket", "MultiEasy_CrowdedPacket": "MultiEasyCrowdedPacket",
    "BSpline_Packet": "BSplineSFFT", "BSpline_MatchingKernel": "BSplineSFFT", "BSpline_DeCorrelation": "BSplineSFFT",
    "BSpline_GridConvolve": "BSplineSFFT",
}


class _LegacyPathLoader(importlib.abc.Loader):
    """Loads an old path as its sfft.legacy module itself (the same module object, not a copy), with a DeprecationWarning."""

    def __init__(self, legacy_name):
        self.legacy_name = legacy_name

    def create_module(self, spec):
        # stacklevel 2 reaches the code that imports: the warnings module skips the import machinery's own frames
        warnings.warn(f"{spec.name} is deprecated: import {self.legacy_name} instead", DeprecationWarning, stacklevel=2)
        return importlib.import_module(self.legacy_name)

    def exec_module(self, module):
        # the module was run when sfft.legacy imported it; running it again would make a second copy of its classes
        pass


class _LegacyPathFinder(importlib.abc.MetaPathFinder):
    """Finds the old paths of LEGACY_MODULES. It comes first in sys.meta_path, so that an old module inside an old package (such as
    sfft.utils.StampGenerator inside sfft.utils, whose files are those of sfft.legacy.utils) is not loaded a second time from its file."""

    OLD_PATHS = {"sfft." + name: "sfft.legacy." + name for name in LEGACY_MODULES}

    def find_spec(self, fullname, path=None, target=None):
        legacy_name = self.OLD_PATHS.get(fullname)
        if legacy_name is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _LegacyPathLoader(legacy_name))


def _install_legacy_paths():
    """Puts one _LegacyPathFinder first in sys.meta_path (once, also when sfft is imported again)."""
    if not any(isinstance(finder, _LegacyPathFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyPathFinder())


_install_legacy_paths()


def __getattr__(name):
    # called only for a name the package does not have: the release's top-level names, imported from sfft.legacy when first used
    module = LEGACY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module 'sfft' has no attribute {name!r}")
    warnings.warn(f"sfft.{name} is deprecated: use sfft.legacy.{name} (from sfft.legacy import {name})", DeprecationWarning, stacklevel=2)
    value = getattr(importlib.import_module("sfft.legacy." + module), name)
    # kept as an attribute of the package, so that the name is found directly from now on (a `from sfft import` looks a name up twice)
    globals()[name] = value
    return value
