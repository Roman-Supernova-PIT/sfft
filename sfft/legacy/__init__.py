"""
Remarks on Internal Packages Imports:
    from sfft.legacy.AutoSparsePrep import Auto_SparsePrep
    from sfft.legacy.AutoCrowdedPrep import Auto_CrowdedPrep
    from sfft.legacy.utils.SymmetricMatch import Symmetric_Match
    from sfft.legacy.utils.HoughMorphClassifier import Hough_MorphClassifier
    from sfft.legacy.utils.WeightedQuantile import TopFlatten_Weighted_Quantile
    from sfft.legacy.utils.meta.MultiProc import Multi_Proc
    from sfft.legacy.utils.meta.TimeoutKit import TimeoutAfter
    from sfft.legacy.utils.pyAstroMatic.PYSEx import PY_SEx
    from sfft.legacy.sfftcore.SFFTConfigure import SFFTConfigure
    from sfft.legacy.sfftcore.SFFTSubtract import GeneralSFFTSubtract
"""

__version__ = "1.6.4"
from .PureCupyCustomizedPacket import PureCupy_Customized_Packet
from .CustomizedPacket import Customized_Packet
from .AutoSparsePrep import Auto_SparsePrep
from .AutoCrowdedPrep import Auto_CrowdedPrep
from .EasySparsePacket import Easy_SparsePacket
from .EasyCrowdedPacket import Easy_CrowdedPacket
from .MultiEasySparsePacket import MultiEasy_SparsePacket
from .MultiEasyCrowdedPacket import MultiEasy_CrowdedPacket
from .BSplineSFFT import BSpline_Packet, BSpline_MatchingKernel, BSpline_DeCorrelation, BSpline_GridConvolve
