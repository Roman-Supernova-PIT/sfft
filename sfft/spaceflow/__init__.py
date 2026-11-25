"""betasfft's workflow for Roman Space Telescope Level 2 exposures.

  SpaceL2_Flow  two Level 2 exposures, given as arrays, differenced end to end, products written to a folder: the one class most users
                need (SpaceL2Flow)
  prep          input preparation (SpaceL2_Prep): the exposures converted to MJy/sr, sky removed, unusable pixels marked, their PSFs
                taken or evaluated from STPSF, and the fit mask of the matched pair; with SFFT's sparse-flavour selection (sparse), the SExtractor wrappers
                (astromatic) and betasfft's SExtractor program (_sextractor, GNU GPL v3 or later)
  subtract      the subtraction (SpaceL2_Sub): resampling with drizzle, PSF models on the target grid, cross-convolution, SFFT on a fit mask
                (solve, with SFFT's refactored core in core), the score and variance images
  instruments   the instruments handled, named by the `instrument` keyword of the three classes (Roman WFI so far)

    from sfft.spaceflow import SpaceL2_Flow
    from sfft.spaceflow.prep import SpaceL2_Prep
    from sfft.spaceflow.subtract import SpaceL2_Sub

prep uses subtract's geometry, PSF models and cross-convolution; subtract never imports prep. SpaceL2_Sub replaces the SN PIT release's
SpaceSFFT_Flow, which is kept in sfft.legacy.SpaceSFFTFlow."""
from .SpaceL2Flow import SpaceL2_Flow
