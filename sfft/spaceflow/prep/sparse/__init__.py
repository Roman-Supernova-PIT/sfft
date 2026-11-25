"""SFFT's sparse-flavour source selection and mask assembly (the parts the Roman single-exposure workflow uses).

  calib.py    gain and saturation values, resolved and validated up front
  catalog.py  image products: SExtractor catalogue, segmentation, point-source locus and FWHM
  select.py   pair stage: catalogue match, offset surface, variable rejection, blend groups, tier ladder
  mask.py     mask assembly: labels, dilation, saturated-object rejection; SparsePrep
"""
from .calib import Calib, CalibError, resolve_calib, resolve_satur_auto, AUTO_SATUR_KEYS
from .catalog import ImageProducts, LocusInfo, image_products, locus_mode, array_identity, prune_cache, products_key
from .select import Selection, SelectionError, select_subsources, match_catalogs, fit_offset_surface
from .mask import SparsePrep, build_masks

__all__ = ['Calib', 'CalibError', 'resolve_calib', 'resolve_satur_auto', 'AUTO_SATUR_KEYS',
           'ImageProducts', 'LocusInfo', 'image_products', 'locus_mode', 'array_identity', 'prune_cache', 'products_key',
           'Selection', 'SelectionError', 'select_subsources', 'match_catalogs', 'fit_offset_surface',
           'SparsePrep', 'build_masks']
