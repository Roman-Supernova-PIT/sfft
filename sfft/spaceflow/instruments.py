"""The instruments whose exposures sfft.spaceflow differences, named by the `instrument` keyword of SpaceL2_Prep, SpaceL2_Sub and SpaceL2_Flow.

  'Roman-WFI'  the Wide Field Instrument of the Roman Space Telescope (the default): Level 2 files (`_cal.asdf`) read by
               sfft.spaceflow.prep.reader, PSFs from STPSF's WFI model (sfft.spaceflow.prep.stpsf_roman) or one PSF array.

Another instrument needs its own reader and PSF source in sfft.spaceflow.prep and a name here; until then it is refused.
"""

INSTRUMENTS = ("Roman-WFI",)
DEFAULT_INSTRUMENT = "Roman-WFI"


def _key(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def instrument_name(instrument):
    """The name in INSTRUMENTS of an instrument given in any capitalisation, with a space, hyphen, underscore, slash or nothing between the
    observatory and the instrument ('Roman-WFI', 'roman_wfi', 'Roman WFI'); ValueError for an instrument sfft.spaceflow does not handle."""
    if not isinstance(instrument, str):
        raise ValueError(f"instrument must be a name such as {DEFAULT_INSTRUMENT!r}, not {instrument!r}")
    for name in INSTRUMENTS:
        if _key(name) == _key(instrument):
            return name
    raise ValueError(f"instrument {instrument!r} is not handled: sfft.spaceflow differences exposures of {', '.join(INSTRUMENTS)}; other "
                     "instruments are not implemented yet")
