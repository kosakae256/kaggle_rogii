# -*- coding: utf-8 -*-
"""v102 affine 較正(v85/v87 verbatim移植)。事前区間(既知prefix)で well GR ≈ a·typewell + b を fit し、
照合GRを typewell スケールに再較正: gr_aligned = (gr − b)/a。gs も aligned 残差で再計算。
create_v95/v97/dec10 が AFFINE=1 のとき、builder 出力 x に対して apply_affine(x, hw, tw_gr, P) を挟む。"""
import numpy as np

AFF_CLIP = (0.3, 3.0)


def affine_cal(kgr, tw_at_k, min_pts=20):
    """kgr ≈ a·tw_at_k + b の polyfit(v52/v85 と同一)。"""
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1.0, (float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0)
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    return float(a), float(b)


def apply_affine(pf, x, hw, tw_gr, P):
    """x(builder出力)の gr を affine 較正。pf=pf_banks module(_smooth_radius_values 使用)。x2 を返す。
       kgr/ktvt=既知prefixの平滑GR/TVT_input, tw_at_k=参照grid gg を ktvt で内挿。"""
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = pf._smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    kgr = gr_sm[kpos].astype(float); ktvt = kn["TVT_input"].to_numpy(float)
    gg = np.asarray(x["gg"], float); tvt_grid = x["gmin"] + x["gst"] * np.arange(len(gg))
    tw_at_k = np.interp(ktvt, tvt_grid, gg)
    a, b = affine_cal(kgr, tw_at_k); a = float(np.clip(a, *AFF_CLIP))
    x2 = dict(x)
    x2["gr"] = (np.asarray(x["gr"], float) - b) / a
    if len(kgr) >= 20:
        resid = (kgr - b) / a - tw_at_k; gs = float(np.nanstd(resid))
        if not np.isfinite(gs) or gs <= 0:
            gs = float(P["gr_sig_def"])
    else:
        gs = float(P["gr_sig_def"])
    x2["gs"] = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    return x2
