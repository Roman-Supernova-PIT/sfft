"""Mask assembly of SFFT's sparse selection: labels, dilation, saturated-object rejection; SparsePrep."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage as ndimage

from .select import Selection

logger = logging.getLogger('sfft.sparse')

LABEL_BACKGROUND = 0
LABEL_PRIORBAN = -64
LABEL_PROHIBITED = -128


@dataclass
class SparsePrep:
    """Everything the subtraction stage needs; the pair-stage product."""
    masked_ref: np.ndarray
    masked_sci: np.ndarray
    active_mask: np.ndarray
    label_map: np.ndarray
    weight_window: np.ndarray | None
    prohibited: np.ndarray
    nan_union: np.ndarray
    ir_sel: np.ndarray
    js_sel: np.ndarray
    selection: Selection
    fwhm_ref: float
    fwhm_sci: float
    qa: dict = field(default_factory=dict)

    def legacy_dict(self):
        """Compatibility export mirroring the legacy SFFTPrepDict keys."""
        return {'Active-Mask': self.active_mask,
                'SFFT-LabelMap': self.label_map,
                'Union-NaN-Mask': self.nan_union,
                'PixA_mREF': self.masked_ref, 'PixA_mSCI': self.masked_sci,
                'FWHM_REF': self.fwhm_ref, 'FWHM_SCI': self.fwhm_sci,
                'MAG_OFFSET': self.selection.surface.scalar}


def _floor_index(xy, shape):
    """SExtractor 1-based coords -> array indices; drop out-of-bounds."""
    xy = np.atleast_2d(np.asarray(xy, float))
    ii = np.floor(xy[:, 0] - 0.5).astype(int)
    jj = np.floor(xy[:, 1] - 0.5).astype(int)
    ok = (ii >= 0) & (jj >= 0) & (ii < shape[0]) & (jj < shape[1])
    if (~ok).any():
        logger.warning('%d prior coordinates outside the image were dropped',
                       int((~ok).sum()))
    return ii[ok], jj[ok]


def build_masks(img_ref, img_sci, prod_r, prod_s, selection: Selection,
                dilate='AUTO', ker_hw=None, sat_reject_pad=2,
                xy_prior_ban=None, protect_targets=None,
                protect_r0_fwhm=3.0, protect_taper=None,
                dilate_method='BOX') -> SparsePrep:
    """Assemble the fitting mask and masked images.

    dilate: 'AUTO' -> max(4, ker_hw) (requires ker_hw), or an int.
    dilate_method: 'BOX' (separable max-filter, ~0.1s at 8.4Mpx; square
    window, largest-label tie-break like legacy) or 'EDT' (~0.8s; circular,
    each grown pixel joins its NEAREST island -- only the label assignment
    in overlaps differs, the active area is near-identical).
    protect_targets: array (N,2) of science-target coords -> tapered holes
    (cosine ramp, inner radius protect_r0_fwhm x FWHM_SCI, taper length
    protect_taper or 2 x ker_hw). xy_prior_ban: (N,2) known variables ->
    their whole label islands are banned (coords need not be precise).
    """
    m, keep = selection.match, selection.keep
    shape = prod_s.shape
    seg_r, seg_s = prod_r.seg, prod_s.seg

    if dilate in ('AUTO', None):
        if ker_hw is None:
            raise ValueError("dilate='AUTO' requires ker_hw")
        dilate = max(4, int(ker_hw))

    sat_r = prod_r.sat_mask
    if sat_reject_pad:
        sat_r = ndimage.binary_dilation(sat_r, iterations=int(sat_reject_pad))
    lab_r_sel = (m.ir + 1)[keep]
    lab_s_sel = (m.js + 1)[keep]
    sat_labels_r = np.unique(seg_r[sat_r])
    bad = np.isin(lab_r_sel, sat_labels_r[sat_labels_r > 0])
    if bad.any():
        logger.info('REF-side saturated-object rejection: %d SubSources '
                    'dropped', int(bad.sum()))
    lab_r_sel, lab_s_sel = lab_r_sel[~bad], lab_s_sel[~bad]
    n_sub = len(lab_r_sel)

    new_of_r = np.zeros(seg_r.max() + 1, np.int32)
    new_of_s = np.zeros(seg_s.max() + 1, np.int32)
    new_of_r[lab_r_sel] = np.arange(1, n_sub + 1)
    new_of_s[lab_s_sel] = np.arange(1, n_sub + 1)
    uni_r = new_of_r[seg_r]
    uni_s = new_of_s[seg_s]
    lmap = np.maximum(uni_r, uni_s).astype(np.int32)

    non_selected = ((seg_r > 0) & (uni_r == 0)) | ((seg_s > 0) & (uni_s == 0))
    nan_union = prod_r.nan_mask | prod_s.nan_mask
    prohibited = non_selected | nan_union | prod_r.sat_mask | prod_s.sat_mask

    lmap[prohibited] = 0
    if dilate_method == 'BOX':
        grown = ndimage.maximum_filter(lmap, size=2 * dilate + 1)
        lmap = np.where(lmap > 0, lmap, grown)
    elif dilate_method == 'EDT':
        dist, (ix, iy) = ndimage.distance_transform_edt(lmap == 0,
                                                        return_indices=True)
        grow = (dist > 0) & (dist <= dilate)
        lmap[grow] = lmap[ix[grow], iy[grow]]
    else:
        raise ValueError(f"dilate_method {dilate_method!r} not in "
                         "('BOX', 'EDT')")
    lmap[prohibited] = LABEL_PROHIBITED

    n_banned = 0
    if xy_prior_ban is not None and len(xy_prior_ban):
        ii, jj = _floor_index(xy_prior_ban, shape)
        ban_labels = np.unique(lmap[ii, jj])
        ban_labels = ban_labels[ban_labels > 0]
        if ban_labels.size:
            banned = np.isin(lmap, ban_labels)
            lmap[banned] = LABEL_PRIORBAN
            n_banned = int(ban_labels.size)
        logger.info('prior ban: %d islands banned from %d coordinates',
                    n_banned, len(xy_prior_ban))

    active = lmap > 0

    Wwin = None
    n_protected = 0
    if protect_targets is not None and len(protect_targets):
        taper = float(protect_taper if protect_taper is not None
                      else 2 * (ker_hw or 8))
        r0 = protect_r0_fwhm * prod_s.locus.fwhm
        gx, gy = np.mgrid[0:shape[0], 0:shape[1]]
        Wwin = np.ones(shape)
        for x0, y0 in np.atleast_2d(protect_targets):
            rr = np.hypot(gx - (x0 - 1.0), gy - (y0 - 1.0))
            w = np.where(rr < r0, 0.0,
                         np.where(rr < r0 + taper,
                                  0.5 * (1 - np.cos(np.pi * (rr - r0) / taper)),
                                  1.0))
            Wwin *= w
            n_protected += 1
        logger.info('target protection: %d tapered holes (r0=%.1fpx, '
                    'taper=%.0fpx)', n_protected, r0, taper)

    masked_ref = np.where(active, img_ref, 0.0)
    masked_sci = np.where(active, img_sci, 0.0)
    masked_ref[nan_union] = 0.0
    masked_sci[nan_union] = 0.0
    if Wwin is not None:
        masked_ref *= Wwin
        masked_sci *= Wwin

    qa = dict(n_subsources=n_sub, n_sat_rejected=int(bad.sum()),
              n_banned_islands=n_banned, n_protected=n_protected,
              dilate=int(dilate), active_fraction=round(float(active.mean()), 4),
              **selection.qa)
    logger.info('mask: %d SubSources, active fraction %.3f (dilate %d)',
                n_sub, active.mean(), dilate)
    return SparsePrep(masked_ref=masked_ref, masked_sci=masked_sci,
                      active_mask=active, label_map=lmap,
                      weight_window=Wwin, prohibited=prohibited,
                      nan_union=nan_union,
                      ir_sel=lab_r_sel - 1, js_sel=lab_s_sel - 1,
                      selection=selection,
                      fwhm_ref=prod_r.locus.fwhm, fwhm_sci=prod_s.locus.fwhm,
                      qa=qa)
