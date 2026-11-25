"""A typed SExtractor wrapper: the configuration as a dataclass written to file, the run as a checked subprocess, products returned."""
from __future__ import annotations

import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ._base import AstromaticTool

DEFAULT_COLUMNS = ('X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO',
                   'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_ISO', 'FLUXERR_ISO',
                   'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE',
                   'B_IMAGE', 'THETA_IMAGE', 'KRON_RADIUS')

CONV_DEFAULT = ("CONV NORM\n"
                "# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n"
                "1 2 1\n"
                "2 4 2\n"
                "1 2 1")


@dataclass(frozen=True)
class SExConfig:
    """Typed SExtractor configuration; rendered to a config file directly.

    Field defaults replicate the legacy PYSEx defaults so golden tests are
    apples-to-apples. `extra` is a passthrough tuple of (KEY, value) pairs
    for anything not modeled as a field.
    """
    gain: float = 0.0
    satur_level: float = 50000.0
    detect_thresh: float = 1.5
    analysis_thresh: float = 1.5
    detect_minarea: int = 5
    detect_maxarea: int = 0
    deblend_nthresh: int = 32
    deblend_mincont: float = 0.005
    clean: str = 'Y'
    use_filter: bool = True
    back_type: str = 'AUTO'
    back_value: float = 0.0
    back_size: int = 64
    back_filtersize: int = 3
    backphoto_type: str = 'LOCAL'
    weight_type: str = 'NONE'
    weight_thresh: float = None
    rescale_weights: str = 'N'
    weight_gain: str = 'N'
    phot_apertures: tuple = (5.0,)
    pixel_scale: float = 1.0
    seeing_fwhm: float = 1.2
    mag_zeropoint: float = 0.0
    verbose_type: str = 'QUIET'
    extra: tuple = ()

    def config_pairs(self):
        """(KEY, value-string) pairs for the config file."""
        pairs = [
            ('GAIN', repr(float(self.gain))),
            ('SATUR_LEVEL', repr(float(self.satur_level))),
            ('GAIN_KEY', 'SFFT_NIL'),
            ('SATUR_KEY', 'SFFT_NIL'),
            ('DETECT_THRESH', str(self.detect_thresh)),
            ('ANALYSIS_THRESH', str(self.analysis_thresh)),
            ('DETECT_MINAREA', str(self.detect_minarea)),
            ('DETECT_MAXAREA', str(self.detect_maxarea)),
            ('DEBLEND_NTHRESH', str(self.deblend_nthresh)),
            ('DEBLEND_MINCONT', str(self.deblend_mincont)),
            ('CLEAN', self.clean),
            ('BACK_TYPE', self.back_type),
            ('BACK_VALUE', str(self.back_value)),
            ('BACK_SIZE', str(self.back_size)),
            ('BACK_FILTERSIZE', str(self.back_filtersize)),
            ('BACKPHOTO_TYPE', self.backphoto_type),
            ('PHOT_APERTURES', ','.join(str(a) for a in self.phot_apertures)),
            ('PIXEL_SCALE', str(self.pixel_scale)),
            ('SEEING_FWHM', str(self.seeing_fwhm)),
            ('MAG_ZEROPOINT', str(self.mag_zeropoint)),
            ('VERBOSE_TYPE', self.verbose_type),
            ('FILTER', 'Y' if self.use_filter else 'N'),
            ('WEIGHT_TYPE', self.weight_type),
        ]
        if self.weight_type != 'NONE':
            pairs.append(('RESCALE_WEIGHTS', self.rescale_weights))
            pairs.append(('WEIGHT_GAIN', self.weight_gain))
        if self.weight_thresh is not None:
            pairs.append(('WEIGHT_THRESH', repr(float(self.weight_thresh))))
        pairs.extend((str(k), str(v)) for k, v in self.extra)
        return pairs

    def content_hash(self) -> str:
        """Stable short hash of the configuration (cache keys, provenance)."""
        blob = repr(sorted(asdict(self).items())).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


@dataclass
class SExResult:
    catalog: Table
    checkimages: dict
    config: SExConfig
    columns: tuple
    command: str
    runtime: float


class SExtractor(AstromaticTool):
    CANDIDATES = ('sex', 'source-extractor', 'sextractor')
    NAME = 'SExtractor'

    def run(self, path=None, image=None, header=None, ext=0,
            config: SExConfig = SExConfig(), columns=DEFAULT_COLUMNS,
            checkimages=(), keep_workspace=False, weight=None):
        """Run SExtractor on a FITS path or an in-memory image.

        path/ext: input FITS (ext selects the HDU -- multi-extension safe)
        image(+header): alternatively an ndarray in FITS (y, x) C-order
        checkimages: e.g. ('SEGMENTATION',); arrays returned in FITS order.
        weight: pixel-weight input for config.weight_type MAP_VAR / MAP_RMS /
          MAP_WEIGHT -- a FITS path or an ndarray in (y, x) C-order (same
          shape as the image; materialized to the workspace). Affects the
          detection threshold normalization AND the error columns; see the
          weight_type field notes for the gain interplay.
        Returns SExResult; the catalog is exactly what the binary produced.
        """
        if (path is None) == (image is None):
            raise ValueError('give exactly one of path= or image=')
        map_types = ('MAP_VAR', 'MAP_RMS', 'MAP_WEIGHT')
        if config.weight_type in map_types and weight is None:
            raise ValueError(f'weight_type={config.weight_type!r} needs '
                             'run(weight=...)')
        if weight is not None and config.weight_type not in map_types:
            raise ValueError('weight= given but config.weight_type is '
                             f'{config.weight_type!r} -- set it to one of '
                             f'{map_types}')
        columns = tuple(dict.fromkeys(tuple(columns) + ('X_IMAGE', 'Y_IMAGE')))
        t0 = time.time()
        ctx = TemporaryDirectory(prefix='sfft_sex_')
        tdir = ctx.name
        try:
            if image is not None:
                path = os.path.join(tdir, 'input.fits')
                fits.PrimaryHDU(data=np.asarray(image),
                                header=header).writeto(path)
                target = path
            else:
                target = path if ext == 0 else '%s[%d]' % (path, ext)

            param_path = os.path.join(tdir, 'sfft.param')
            with open(param_path, 'w') as f:
                f.write('\n'.join(columns))
            lines = ['%-18s %s' % kv for kv in config.config_pairs()]
            if weight is not None:
                if isinstance(weight, (str, os.PathLike)):
                    wpath = os.fspath(weight)
                else:
                    wpath = os.path.join(tdir, 'weight.fits')
                    fits.PrimaryHDU(
                        data=np.asarray(weight, np.float64)).writeto(wpath)
                lines.append('%-18s %s' % ('WEIGHT_IMAGE', wpath))
            lines.append('%-18s %s' % ('PARAMETERS_NAME', param_path))
            if config.use_filter:
                conv_path = os.path.join(tdir, 'sfft.conv')
                with open(conv_path, 'w') as f:
                    f.write(CONV_DEFAULT)
                lines.append('%-18s %s' % ('FILTER_NAME', conv_path))
            cat_path = os.path.join(tdir, 'catalog.fits')
            lines.append('%-18s %s' % ('CATALOG_NAME', cat_path))
            lines.append('%-18s %s' % ('CATALOG_TYPE', 'FITS_LDAC'))
            check_paths = {}
            if checkimages:
                for ctype in checkimages:
                    check_paths[ctype] = os.path.join(
                        tdir, 'check_%s.fits' % ctype.lower())
                lines.append('%-18s %s' % ('CHECKIMAGE_TYPE',
                                           ','.join(checkimages)))
                lines.append('%-18s %s' % ('CHECKIMAGE_NAME',
                                           ','.join(check_paths[c]
                                                    for c in checkimages)))
            else:
                lines.append('%-18s %s' % ('CHECKIMAGE_TYPE', 'NONE'))
            cfg_path = os.path.join(tdir, 'sfft.sex')
            with open(cfg_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')

            argv = [self.binary, target, '-c', cfg_path]
            self._run(argv, cwd=tdir)

            catalog = Table.read(cat_path, hdu=2)
            checks = {c: fits.getdata(p, ext=0)
                      for c, p in check_paths.items()}
            return SExResult(catalog=catalog, checkimages=checks,
                             config=config, columns=columns,
                             command=' '.join(argv), runtime=time.time() - t0)
        finally:
            if keep_workspace:
                ctx._finalizer.detach()
                import logging
                logging.getLogger('sfft.astromatic').warning(
                    'workspace kept: %s', tdir)
            else:
                ctx.cleanup()
