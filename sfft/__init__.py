"""betasfft: image subtraction with SFFT (Saccadic Fast Fourier Transform), in two subpackages.

  sfft.spaceflow  the workflow for space-telescope Level 2 exposures: SpaceL2_Flow differences two exposures end to end, through
                  SpaceL2_Prep (input preparation, sfft.spaceflow.prep) and SpaceL2_Sub (resampling, matching and SFFT, sfft.spaceflow.subtract)
  sfft.legacy     the SFFT release this package started from (the Roman SN PIT fork of SFFT 1.6.4): the customized, sparse and crowded
                  packets, their solver (sfft.legacy.sfftcore), their tools (sfft.legacy.utils) and the release's SpaceSFFT_Flow
                  (sfft.legacy.SpaceSFFTFlow)

    from sfft.spaceflow import SpaceL2_Flow
    from sfft.legacy import Customized_Packet

`import sfft` imports neither subpackage, so the space-telescope workflow does not need the legacy packets' dependencies.
"""
__version__ = "1.6.4.dev20"
