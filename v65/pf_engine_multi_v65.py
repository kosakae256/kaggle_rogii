# -*- coding: utf-8 -*-
"""
v65 合成尤度PFエンジン(composite-likelihood) — pf_engine_v65 の派生。本体は無改造。
================================================================================
狙い(ユーザー提案 2026-07-15): tw/self/nbr を「テンプレート平均(=平均禁)」で混ぜるのでなく、
  各行の emission尤度を"積"で合成し、全ソースが同時に満たす位置に粒子を集める。
    lk = Π_s exp(-0.5 · β_s · ((GR - eg_s)/gs)^power)      (s ∈ {tw, self, nbr})
  → 単一 tw だと ±15ft の datum両義が同点でも、self/nbr が片方を否定すれば逐次の中で
    正しい枝にコミットできる(事後選択=v47の56%頭打ち、とは別経路の武器)。GR照合のみ=struct任意で転移安全。

参照の作り方:
  - tw を基準(β=1, mask全1)。TW_SOURCE に "self"/"nbr" が含まれる時だけ各テンプレを追加。
  - self = 自坑井既知帯(TVT_input,GR) の格子bin中央値 / nbr = 近傍≤3本(≤MAX_DIST=1500)の連結。
  - **近傍<2本(=nbr_pairs None)の坑井は nbr層 mask=0 → nbr尤度が中立(=1)**。何でも使わない(ユーザー指示)。
  - self_mix_w / nbr_mix_w を「混合比」でなく **emission指数β(そのソースの信頼度)** として再解釈。
    → pf_engine_v65.suggest_params が TW_SOURCE に応じて既に振るので、**チューナ側のコードは無改造で最適化可**。

純関数(list_wells/build_cache/_bin_pairs/_smooth/_grid/suggest_params/full_params と定数)は本体を再利用。
使い方(既存チューナのコピー tune_pf_multi_v65 経由): TW_SOURCE=tw+self+nbr python v65/tune_pf_multi_v65.py
単一 tw(TW_SOURCE=tw)なら S=1 で pf_engine_v65 の tw PF と同一結果(回帰確認用)。
"""
import os
import numpy as np
import torch
import pf_engine_v65 as base
from pf_engine_v65 import (PROJECT, DATA_DIR, OUT_DIR, DEVICE, DTYPE, PRIOR_NAME, SEED0, _STRUCT,
                           list_wells, build_cache, _bin_pairs, _smooth, _grid,
                           suggest_params, full_params, _V65_DEFAULTS)

CACHE = {}   # このモジュール専用(tuner が pf.CACHE = ... で差し込む)


# ----------------------- 入力(tw/self/nbr を"個別テンプレ+mask"で保持) -----------------------
def make_inputs(c, P):
    """pf_engine_v65.make_inputs と同じ σ較正・tw格子生成を踏むが、self/nbr は混ぜず個別に返す。"""
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf, step=float(P.get("grid_step", 0.2)))
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    bias = 0.0
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
        if int(P.get("gr_debias", 0)):
            b_ = float(np.nanmedian(resid))
            if np.isfinite(b_): bias = b_
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    if bias != 0.0:
        gr_sm = gr_sm - bias
    # ---- 参照スタック: tw(β=1) + self/nbr(β=self/nbr_mix_w)。混ぜず、被覆maskを添える ----
    src = str(P.get("tw_source", "tw"))
    grids = [gg.astype(np.float64)]; masks = [np.ones(len(gg), np.float64)]; betas = [1.0]
    if "self" in src:
        gh = _bin_pairs(c["kn_tvtin"], gr_sm[c["kpos"]].astype(np.float64), gmin, gst, len(gg))
        m = np.isfinite(gh); grids.append(np.where(m, gh, gg)); masks.append(m.astype(np.float64))
        betas.append(float(P.get("self_mix_w", 1.0)))
    if "nbr" in src:
        gh = np.full(len(gg), np.nan)                 # 近傍<2本(nbr_pairs None)ならNaN → mask全false=中立
        if c.get("nbr_pairs") is not None:
            nt, ng = c["nbr_pairs"]
            ngs = _smooth(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
            gh = _bin_pairs(nt, ngs, gmin, gst, len(gg))
        m = np.isfinite(gh); grids.append(np.where(m, gh, gg)); masks.append(m.astype(np.float64))
        betas.append(float(P.get("nbr_mix_w", 1.0)))
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]),
                anc=c["anc"], ancs=c["ancs"],
                ref_grids=np.stack(grids, 0), ref_masks=np.stack(masks, 0), ref_beta=np.asarray(betas, np.float64))


# ----------------------- 合成尤度コア(emission だけ"積"、伝播/錨/リサンプルは本体と同一) -----------------------
def _pf_core_multi(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, refs, dtype=DTYPE):
    B, T = md.shape
    ref_grids = refs["grids"]; ref_masks = refs["masks"]; ref_beta = refs["beta"]; Sref = ref_grids.shape[1]
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    USE_A = bool(P.get("use_anchor", True)); AMUL = float(P.get("anchor_mult", 20.0))
    RAMP = float(P.get("anchor_ramp_md", 0.0)); LKF = float(P.get("lk_floor", 0.0))
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    gidx_last = (glen - 1).clamp_min(0).unsqueeze(1)
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    one = torch.ones((), device=device, dtype=dtype)
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + ((1.0 - ALPHA) * dipp if RT_IR else 0.0) + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype)
        lk = torch.ones_like(w)                              # ★ 尤度は各ソースの積
        for s in range(Sref):
            gsg = ref_grids[:, s, :]; gsm = ref_masks[:, s, :]; bta = ref_beta[s]
            gA = torch.gather(gsg, 1, i0c); gB = torch.gather(gsg, 1, i0c + 1)
            eg = gA * (1 - t) + gB * t
            eg = torch.where(below, gsg[:, 0:1], eg)
            eg = torch.where(above, torch.gather(gsg, 1, gidx_last), eg)
            d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = (d ** GR_POWER) * bta
            lk_s = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
            mA = torch.gather(gsm, 1, i0c); mB = torch.gather(gsm, 1, i0c + 1)
            cov = ((mA > 0) & (mB > 0)) | below | above      # 被覆ある格子のみ寄与。無い所は中立(=1)
            lk = lk * torch.where(cov, lk_s, one)
        if LKF > 0:
            lk = (1.0 - LKF) * lk + LKF
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        if USE_A:
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig; da2 = da * da
            if RAMP > 0:
                tf = ((cur - md[:, 0]).clamp_min(0.0) / RAMP).clamp(0.0, 1.0)[:, None]; da2 = da2 * tf
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pts[:, i] = (w * (pos - zi)).sum(1); pm = torch.where(act, cur, pm)
    return pts, log_lik


# ----------------------- パディング(本体 _pad + 参照スタック[S,G]) -----------------------
def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    Sref = inps[0]["ref_grids"].shape[0]
    rg = torch.zeros((W, Sref, Gmax), dtype=dtype); rm = torch.zeros((W, Sref, Gmax), dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
        anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
        if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
        ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        rg[b, :, :G] = torch.from_numpy(x["ref_grids"].astype("float32"))
        rm[b, :, :G] = torch.from_numpy(x["ref_masks"].astype("float32"))
        if G < Gmax: rg[b, :, G:] = rg[b, :, G - 1:G]        # mask側は0のまま(=格子外は中立)
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                ref_grids=to(rg), ref_masks=to(rm), ref_beta=to(torch.from_numpy(inps[0]["ref_beta"].astype("float32"))))


def run_forward_ens(inps, P, device, seed, n_seeds):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    refs = dict(grids=rep(pad["ref_grids"]), masks=rep(pad["ref_masks"]), beta=pad["ref_beta"])
    pts, ll = _pf_core_multi(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                             pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S),
                             pad["step"].repeat_interleave(S), pad["gs"].repeat_interleave(S),
                             pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                             rep(pad["anc"]), rep(pad["ancs"]), P, N, device, gen, refs)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    outs = []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]; llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); ssum = ww.sum()
            ww = ww / ssum if ssum > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        outs.append((ww[:, None] * pr).sum(0))
    return outs


# ----------------------- 評価(本体と同一API。make_inputs/run_forward_ens は本モジュールを使う) -----------------------
def predict_errs(P, subset, seed, n_seeds, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        preds = run_forward_ens(inps, P, DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            res[w] = preds[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res


def overall_rmse_on(P, subset, seed, n_seeds, chunk):
    errs = predict_errs(P, subset, seed, n_seeds, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))


def perwell_rmse(P, subset, seed, n_seeds, chunk):
    errs = predict_errs(P, subset, seed, n_seeds, chunk)
    out = {}
    for w, e in errs.items():
        e = e[np.isfinite(e)]
        out[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    return out
