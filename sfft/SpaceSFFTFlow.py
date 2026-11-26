"""SpaceSFFT_Flow: the name of the subtraction class in the SFFT 1.6.4 release, kept for its users.

    from sfft.SpaceSFFTFlow import SpaceSFFT_Flow

SpaceSFFT_Flow is sfft.SpaceSFFT, which runs either engine: engine='romanL2' (the default) or engine='legacy' (the release's own
subtraction, sfft.legacy.SpaceSFFTFlow.SpaceSFFT_Flow).
"""
from .SpaceSFFT import SpaceSFFT

SpaceSFFT_Flow = SpaceSFFT
