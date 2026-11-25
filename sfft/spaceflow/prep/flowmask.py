"""The SFFT fit mask of a SpaceL2_Sub's matched pair, from its public attributes (stage 2 of SpaceL2_Prep, usable with any SpaceL2_Sub)."""
import numpy as np

from .fitmask import build, half_light_radius, sparse_sfft


def fit_mask_from_flow(flow, saturation_map=None, sfft_grow=2, sfft_sat_pad=2, sfft_match_tol_px=None, sfft_star_guard="corridors",
                       sfft_exclude_stars=True, fit_band=(17.0, 170.0), min_rhalf=1.5):
    """The SFFT fit mask of a SpaceL2_Sub (flow) after its cross_convolve(), from its public attributes: SFFT's sparse-flavour selection on
    the matched pair (sparse_sfft) with the saturated-star guard and the PSF star test; the band mask (build) when the sparse selection fails.

    saturation_map: the pixels flagged saturated, on the target grid in the flow's input orientation (None: every missing pixel counts as
    saturated). Returns (fit, record): fit is a bool array on the target grid in the flow's input orientation, for flow.sfft_subtract(fit_mask=fit)."""
    import warnings
    from ..subtract.crossconv import robust_sigma
    from .astromatic.bundled import BundledSExtractorError
    if not getattr(flow, "CROSS_CONVOLVED", False):
        raise ValueError("the fit mask is made from the matched pair: run flow.cross_convolve() first")

    def yx(a):
        return np.ascontiguousarray(np.asarray(flow.op.asnumpy(a)).T)

    def nodes_yx(a):
        return np.ascontiguousarray(np.asarray(flow.op.asnumpy(a)).transpose(1, 0, 3, 2))

    m_t, m_o = yx(flow.PixA_Ctarget), yx(flow.PixA_Cresamp_object)
    psf_t, psf_o = nodes_yx(flow.PSF_target_NODES), nodes_yx(flow.PSF_resamp_object_NODES)
    sci_w, ref_w = (m_t, m_o) if flow.sci_is_target else (m_o, m_t)
    psf_s, psf_r = (psf_t, psf_o) if flow.sci_is_target else (psf_o, psf_t)
    oversample = int(flow.PSF_WORK_OVERSAMPLING)
    nodes = dict(sci=psf_s, ref=psf_r, xs=np.asarray(flow.PSF_NODE_X), ys=np.asarray(flow.PSF_NODE_Y), oversample=oversample)
    sat = None
    if saturation_map is not None:
        sat = np.asarray(saturation_map, bool)
        sat = sat if flow.transpose else sat.T
        if not sat.any():
            warnings.warn("no exposure flags a pixel as saturated, so the sparse mask drops no source as saturated", stacklevel=2)
    gker_hw, fwhm = int(flow.GKerHW), float(flow.PSF_FWHM)
    free = yx(flow.SourceFreeMask).astype(bool)
    p_eff, p_fine = yx(flow.PSF_Ctarget), yx(flow.PSF_Ctarget_OVERSAMPLED)
    sig_sm = robust_sigma(sci_w[free])
    try:
        fit, rec = sparse_sfft(sci_w, ref_w, fwhm, gker_hw, grow=sfft_grow, sat_pad=sfft_sat_pad, sat_map=sat, match_tol_px=sfft_match_tol_px,
                               star_guard=sfft_star_guard, exclude_stars=sfft_exclude_stars, psf_fine=p_fine, psf_oversample=oversample,
                               psf_nodes=nodes)
    except BundledSExtractorError:
        raise                       # betasfft's SExtractor program cannot be used (not built, stale): stop, as no mask can be made without it
    except Exception as e:  # noqa: BLE001  (the sparse selection refuses thin or inconsistent fields; SExtractor can fail)
        problem = f"{type(e).__name__}: {str(e)[:300]}"
        warnings.warn(f"the sparse fit mask failed ({problem}); the band mask is used instead", stacklevel=2)
        fit, rec, _ = build(sci_w, ref_w, sig_sm, fwhm, max(gker_hw, 4), fit_band=tuple(fit_band) if fit_band else None, min_rhalf=min_rhalf,
                            r_psf=half_light_radius(p_eff))
        rec["sfft_problem"] = problem
    return (fit if flow.transpose else fit.T), rec
