"""Reading two Roman WFI Level 2 files (`_cal.asdf`) into the inputs of sfft.spaceflow.SpaceL2_Flow (and SpaceL2_Prep), and running it.

sfft reads no file: a caller reads both exposures and passes their arrays, with what the files hold as keywords. In the Roman SN PIT pipeline
snappl does the reading; this script shows the same with roman_datamodels alone, for a quick run on two files:

    python read_roman_l2.py science_cal.asdf reference_cal.asdf --output-dir difference --cache-dir ~/.cache/sfft_romanl2

or, from Python (the folder of this script on the path):

    from read_roman_l2 import flow_inputs
    from sfft.spaceflow import SpaceL2_Flow

    inputs, keywords = flow_inputs("science_cal.asdf", "reference_cal.asdf")
    flow = SpaceL2_Flow(*inputs, output_dir="difference", **keywords)
    record = flow.run()

What each keyword comes from in the file:
  PixA_<side>, PixA_<side>Var           data, and err squared
  wcs_<side>                            meta.wcs (the gWCS)
  PixA_<side>_BADMASK                   dq with any flag outside KEEP_FLAGS
  PixA_<side>_SATMASK, _JUMPMASK        dq & SATURATED, dq & JUMP_DET
  PixA_<side>_DUMO, _VARPOISSON         dumo, var_poisson (left out when the file has none)
  detector_<side>, filter_<side>        meta.instrument.detector, meta.instrument.optical_element
  conversion_megajanskys_<side>         1.0 when cal_step.flux is COMPLETE (the data is in MJy/sr), else photometry.conversion_megajanskys
  pixel_area_sr_<side>                  meta.photometry.pixel_area
  is_surface_brightness_<side>          whether meta.ref_file.flat names a flat (a flat-fielded file holds a surface brightness)
  name_<side>                           meta.filename
<side> is 'target' for the science exposure and 'object' for the reference. Needs roman_datamodels (pip install roman_datamodels).
"""
import argparse
import os
import warnings

import numpy as np

KEEP_FLAGS = ("WARM", "LOW_QE", "JUMP_DET", "PERSISTENCE")      # data-quality flags whose pixels stay usable; any other flag makes a pixel unusable


def _get(node, path, default=None):
    """meta.<path> of a datamodel, or default when it is missing."""
    current = node
    try:
        for part in path.split("."):
            current = getattr(current, part)
        return current
    except AttributeError:
        return default


def _positive(value):
    """A positive finite number, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


def bad_pixel_bits(keep_flags=KEEP_FLAGS):
    """Bit mask of the data-quality flags that make a pixel unusable: every flag except keep_flags."""
    from roman_datamodels import dqflags
    bits = 0
    for flag in dqflags.pixel:
        if flag.name not in keep_flags:
            bits |= int(flag.value)
    return bits


def read_cal_asdf(path, side, keep_flags=KEEP_FLAGS):
    """One exposure of a Roman WFI `_cal.asdf` as the inputs of SpaceL2_Flow: (data, variance, gWCS, keywords), side 'target' (the science
    exposure) or 'object' (the reference)."""
    import roman_datamodels as rdm
    from roman_datamodels import dqflags
    if side not in ("target", "object"):
        raise ValueError(f"side must be 'target' or 'object', not {side!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = rdm.open(path)
        data = np.array(dm.data, dtype=np.float64)
        err = np.array(dm.err, dtype=np.float64)
        dq = np.array(dm.dq, dtype=np.uint32)
        dumo = np.array(dm.dumo, dtype=np.float64) if hasattr(dm, "dumo") else None
        var_poisson = np.array(dm.var_poisson, dtype=np.float64) if hasattr(dm, "var_poisson") else None
        gwcs = dm.meta.wcs
        meta = dm.meta
        flat = str(_get(meta, "ref_file.flat", "") or "")
        flux_step = str(_get(meta, "cal_step.flux", "?"))
        keywords = {
            f"PixA_{side}_BADMASK": (dq & np.uint32(bad_pixel_bits(keep_flags))) != 0,
            f"PixA_{side}_SATMASK": (dq & np.uint32(dqflags.pixel["SATURATED"].value)) != 0,
            f"PixA_{side}_JUMPMASK": (dq & np.uint32(dqflags.pixel["JUMP_DET"].value)) != 0,
            f"detector_{side}": str(_get(meta, "instrument.detector", "")) or None,
            f"filter_{side}": str(_get(meta, "instrument.optical_element", "")) or None,
            f"conversion_megajanskys_{side}": 1.0 if flux_step == "COMPLETE" else _positive(_get(meta, "photometry.conversion_megajanskys")),
            f"pixel_area_sr_{side}": _positive(_get(meta, "photometry.pixel_area")),
            f"is_surface_brightness_{side}": flat.lower() not in ("", "n/a", "none", "?"),
            f"name_{side}": str(_get(meta, "filename", os.path.basename(path))),
        }
    if dumo is not None:
        keywords[f"PixA_{side}_DUMO"] = dumo
    if var_poisson is not None:
        keywords[f"PixA_{side}_VARPOISSON"] = var_poisson
    return data, err ** 2, gwcs, keywords


def flow_inputs(science_path, reference_path, keep_flags=KEEP_FLAGS):
    """The inputs of SpaceL2_Flow (or SpaceL2_Prep) for two `_cal.asdf` files: (the six positional inputs, the keywords)."""
    data_t, var_t, wcs_t, keywords_t = read_cal_asdf(science_path, "target", keep_flags)
    data_o, var_o, wcs_o, keywords_o = read_cal_asdf(reference_path, "object", keep_flags)
    return (data_t, data_o, var_t, var_o, wcs_t, wcs_o), {**keywords_t, **keywords_o}


def main():
    parser = argparse.ArgumentParser(description="Difference two Roman WFI Level 2 files with sfft.spaceflow.SpaceL2_Flow.")
    parser.add_argument("science", help="the science exposure's _cal.asdf (its grid is the products' grid)")
    parser.add_argument("reference", help="the reference exposure's _cal.asdf, same filter")
    parser.add_argument("--output-dir", default="spaceflow_output", help="where the products are written (default %(default)s)")
    parser.add_argument("--cache-dir", default=None, help="folder for the STPSF PSF models (default ~/.cache/sfft_romanl2)")
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"], help="default %(default)s")
    parser.add_argument("--crop-px", type=int, default=None, help="work on a crop_px x crop_px part of the science exposure, for a quick look")
    args = parser.parse_args()

    from sfft.spaceflow import SpaceL2_Flow
    inputs, keywords = flow_inputs(args.science, args.reference)
    flow = SpaceL2_Flow(*inputs, output_dir=args.output_dir, backend=args.backend, crop_px=args.crop_px, cache_dir=args.cache_dir, **keywords)
    record = flow.run()
    print(f"done in {record['timings']['total']:.0f} s on the {record['backend']['used'].upper()}; products in {args.output_dir}")


if __name__ == "__main__":      # the guard is needed: the STPSF PSF source evaluates PSFs in spawned worker processes
    main()
