"""Weighted quantiles."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger('sfft.utils')

__all__ = ['weighted_quantile', 'top_flatten_weighted_quantile']


def weighted_quantile(values, quantiles, weights=None, values_sorted=False):
    """Quantiles of `values` under sample `weights` (quantiles in [0, 1]).

    The estimator interpolates the cumulative weight at the sample points
    (midpoint convention), reducing to the classic weighted percentile; with
    weights=None it is close to (not identical to) np.quantile.
    """
    values = np.asarray(values, np.float64)
    quantiles = np.atleast_1d(np.asarray(quantiles, np.float64))
    if values.size == 0:
        raise ValueError('weighted_quantile: empty sample')
    if np.any(quantiles < 0) or np.any(quantiles > 1):
        raise ValueError('weighted_quantile: quantiles must be in [0, 1]')
    weights = np.ones(values.size) if weights is None \
        else np.asarray(weights, np.float64)
    if not values_sorted:
        order = np.argsort(values)
        values, weights = values[order], weights[order]
    cum = np.cumsum(weights) - 0.5 * weights
    cum /= np.sum(weights)
    return np.interp(quantiles, cum, values)


def top_flatten_weighted_quantile(values, quantiles, weights, num_top=30):
    """Weighted quantiles with the TOP `num_top` weights flattened to 1.

    Rationale: with flux weights, a handful of very
    bright sources can dominate the estimate; clipping the weight scale at
    the num_top-th largest weight keeps them influential but not decisive.
    Samples of <= num_top values fall back to the UNWEIGHTED quantile (all
    weights would flatten anyway); an empty sample raises -- that means the
    upstream selection produced no members, which is the thing to fix.
    """
    values = np.asarray(values, np.float64)
    if values.size == 0:
        raise ValueError(
            'top_flatten_weighted_quantile: EMPTY sample -- typically zero '
            'point sources survived the upstream selection; fix the '
            'selection/detection stage, not this call.')
    quantiles = np.atleast_1d(np.asarray(quantiles, np.float64))
    if values.size <= num_top:
        logger.info('top-flatten quantile: sample of %d <= num_top=%d -> '
                    'uniform-weighted quantile', values.size, num_top)
        return np.percentile(values, 100.0 * quantiles)
    weights = np.asarray(weights, np.float64)
    scale = np.sort(weights)[-num_top]
    return weighted_quantile(values, quantiles,
                             weights=np.clip(weights / scale, 0.0, 1.0))
