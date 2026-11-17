# -*- coding: utf-8 -*-
"""
v88 多チャネル尤度PFエンジン(GPU) — pf_engine_v65 の派生。本体は無改造で再利用。
=========================================================================================
狙い: GRの自己相似(準周期)で datum が両義になるのは、**尤度が LF(低周波=どのサイクルか) と
HF(サイクル形状=両義) を1つの残差・1つのσに混ぜている**ため。HFの分散が大きいので、
判別情報である LF が事実上埋もれる。
→ **チャネルに分離し、各チャネルを自前のσで正規化して重み付き和を指数に載せる**:

    lk = exp( -0.5 · U(tvt) · Σ_ch  w_ch · |(obs_ch − ref_ch(tvt)) / σ_ch|^power )

  ch ∈ {raw, lf, hf} × src ∈ {tw, self, nbr}。σ_ch は各チャネルの既知帯残差から自動較正。
  U(tvt) = 識別力(distinctiveness)重み(E3, 既定1.0=無効)。

W_RAW=1, W_LF=0, W_HF=0, UNIQ_W=0, src=tw のとき **pf_engine_v65 の tw PF と一致**(回帰確認用)。

P に足すキー(既定は全て中立):
  v88_w_raw(1.0) v88_w_lf(0.0) v88_w_hf(0.0)      … チャネル重み
  v88_lf_tvt(4.0) v88_lf_md(100.0)                 … LF平滑半径(参照=TVT ft / 観測=MD ft)
  v88_src("tw")                                    … "tw" / "tw+self" / "tw+nbr" / "tw+self+nbr"
  v88_beta_self(1.0) v88_beta_nbr(1.0)             … 追加ソースの指数(信頼度)
  v88_uniq_w(0.0) v88_uniq_win(6.0) v88_uniq_lag(30.0) v88_uniq_floor(0.3)  … 識別力重み(E3)
"""
import os
import numpy as np
import torch
import pf_engine_v65 as base
from pf_engine_v65 import (PROJECT, DATA_DIR, DEVICE, DTYPE, SEED0, list_wells, build_cache,
                           _bin_pairs, _smooth, _grid, full_params as _base_full, _V65_DEFAULTS)

CACHE = {}

_V88_DEFAULTS = dict(
    v88_w_raw=1.0, v88_w_lf=0.0, v88_w_hf=0.0,
    v88_lf_tvt=4.0, v88_lf_md=100.0,
    v88_src="tw", v88_beta_self=1.0, v88_beta_nbr=1.0,
    v88_uniq_w=0.0, v88_uniq_win=6.0, v88_uniq_lag=30.0, v88_uniq_floor=0.3,
    v88_ctx_k=0, v88_ctx_span=200.0, v88_ctx_w=0.0,
    v88_conf_kcnt=4.0, v88_nbr_affine=1, v88_neutral=1.0,
    v88_self_sig="tw", v88_self_sig_scale=1.0, v88_self_fill=0.0, v88_self_tail=0,
    v88_w_lfref=0.0, v88_w_lfref2=0.0, v88_lf_tvt2=12.0, v88_infow=0.0,
)


def full_params(p):
    q = _base_full(p)
    for k, v in _V88_DEFAULTS.items():
        q.setdefault(k, v)
    return q


def _ma(x, r):
    """半径r(サンプル)の移動平均。端はedge padding。r<=0で素通し。"""
    r = int(r)
    if r <= 0:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    k = 2 * r + 1
    xp = np.pad(x, (r, r), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def _bin_stats(tvt, gr, gmin, step, G):
    """(TVT,GR)ペアを格子binへ集約: 中央値 / 件数 / bin内std。
       ★bin内std = 「同一TVTでGRがどれだけ散るか」= その参照の**既約ノイズ**。
       self参照のσをこれで決める(prefix残差で決めると自己整合で0になり過信するため)。"""
    m = np.isfinite(tvt) & np.isfinite(gr)
    t, g = np.asarray(tvt)[m], np.asarray(gr)[m]
    idx = np.round((t - gmin) / step).astype(int)
    ok = (idx >= 0) & (idx < G); idx, g = idx[ok], g[ok]
    med = np.full(G, np.nan); cnt = np.zeros(G); sd = np.full(G, np.nan)
    if len(g):
        import pandas as _pd
        s = _pd.Series(g).groupby(idx)
        mm = s.median(); cc = s.size(); ss = s.std()
        med[mm.index.to_numpy()] = mm.to_numpy()
        cnt[cc.index.to_numpy()] = cc.to_numpy()
        sd[ss.index.to_numpy()] = ss.to_numpy()
    return med, cnt, sd


def _sig_of(obs_k, ref_at_k, P):
    """チャネル残差(既知帯)から σ を較正。base と同じ clip 規則。"""
    if len(obs_k) < 20:
        s = float(P["gr_sig_def"])
    else:
        s = float(np.nanstd(obs_k - ref_at_k))
        if not np.isfinite(s) or s <= 0:
            s = float(P["gr_sig_def"])
    return float(np.clip(s * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))


def _uniqueness(ref, gst, win_ft, lag_ft, floor):
    """typewell参照の局所識別力 U∈[floor,1]。
       各位置の窓と、±lag以内の他位置の窓との最大相関が高い=自己相似=識別力低い。"""
    G = len(ref)
    hw = max(2, int(round(win_ft / gst)))
    if G < 4 * hw + 4:
        return np.ones(G)
    # 窓行列(正規化)
    idx = np.arange(-hw, hw + 1)
    cen = np.arange(hw, G - hw)
    Wm = ref[cen[:, None] + idx[None, :]]
    Wm = Wm - Wm.mean(1, keepdims=True)
    nrm = np.sqrt((Wm ** 2).sum(1)) + 1e-9
    Wm = Wm / nrm[:, None]
    maxlag = max(1, int(round(lag_ft / gst)))
    minlag = max(1, int(round(2.0 / gst)))          # 近すぎる位置は自明なので除外
    best = np.full(len(cen), -1.0)
    step = max(1, minlag // 2)
    for L in range(minlag, maxlag + 1, step):
        if L >= len(cen):
            break
        c = (Wm[L:] * Wm[:-L]).sum(1)               # ラグLでの窓相関
        best[:-L] = np.maximum(best[:-L], c)
        best[L:] = np.maximum(best[L:], c)
    u = 1.0 - np.clip(best, 0.0, 1.0)               # 相関高→U小(鈍らせる)
    out = np.ones(G)
    out[cen] = u
    out[:hw] = u[0] if len(u) else 1.0
    out[G - hw:] = u[-1] if len(u) else 1.0
    return np.clip(out, floor, 1.0)


def make_inputs(c, P):
    """チャネル(raw/lf/hf) × ソース(tw/self/nbr)を個別obs・個別σ・個別重みで束ねる。"""
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf, step=float(P.get("grid_step", 0.2)))
    gr_full = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"]).astype(np.float64)

    # gs(raw, base互換): 既知帯残差
    kpos = c["kpos"]; ktvt = c["kn_tvtin"]
    ref_at_k_raw = np.interp(ktvt, c["tw_tvt"], tw_gr_pf)
    gs_raw = _sig_of(gr_full[kpos], ref_at_k_raw, P)
    if int(P.get("gr_debias", 0)) and len(kpos) >= 20:
        b_ = float(np.nanmedian(gr_full[kpos] - ref_at_k_raw))
        if np.isfinite(b_):
            gr_full = gr_full - b_

    # ---- 参照ソース(格子)を用意: tw + (self/nbr) ----
    src = str(P.get("v88_src", "tw"))
    G = len(gg); tvt_g = gmin + gst * np.arange(G)
    KC = max(1.0, float(P.get("v88_conf_kcnt", 4.0)))
    srcs = [("tw", gg, np.ones(G), 1.0, None, 0.0)]
    if "self" in src:
        # ★self格子の材料を「eval帯に近い prefix 末尾 TAIL 行」に限定できる。
        #   地質の連続性は距離で減衰するので、ラテラル直前の区間ほど eval帯を代表するはず。
        TAIL = int(P.get("v88_self_tail", 0))
        _kp, _kt = (kpos[-TAIL:], ktvt[-TAIL:]) if (TAIL > 0 and len(kpos) > TAIL) else (kpos, ktvt)
        med, cnt, sd = _bin_stats(_kt, gr_full[_kp], gmin, gst, G)
        m = np.isfinite(med)
        ref = np.where(m, med, gg)
        conf = np.where(m, np.clip(cnt / KC, 0.0, 1.0), 0.0)          # 疎bin=低信頼
        # ★被覆拡張: 被覆binが疎(実測 被覆率27.6%)だと尤度が断続的になりPFが不安定。
        #   被覆bin間を線形補間し、最近傍被覆binまでの距離で信頼度を減衰させて埋める。
        FILL = float(P.get("v88_self_fill", 0.0))
        ci = np.where(m)[0]
        if FILL > 0 and len(ci) >= 2:
            ref = np.interp(np.arange(G), ci, med[ci])
            d_ft = np.abs(np.arange(G)[:, None] - ci[None, :]).min(1) * gst
            conf = np.maximum(conf, np.clip(cnt / KC, 0.0, 1.0).max() * np.exp(-d_ft / FILL))
            conf = np.clip(conf, 0.0, 1.0)
        # ★σ_self / 典型ミスフィット m_self は **prefix内ホールドアウト**で正直に測る。
        #   self格子は自分のprefixから作るので、prefix上の残差は自己整合でほぼ0になり
        #   σもmも測れない(前半で格子を作り後半で評価する)。
        okk = gr_full[kpos]
        nk = len(kpos); half = nk // 2
        sig = gs_raw; mref = 1.0
        if half >= 20:
            medA, cntA, _ = _bin_stats(ktvt[:half], okk[:half], gmin, gst, G)
            mA = np.isfinite(medA)
            if mA.sum() >= 5:
                covB = np.interp(ktvt[half:], tvt_g, mA.astype(float), left=0.0, right=0.0)
                refB = np.interp(ktvt[half:], tvt_g[mA], medA[mA], left=np.nan, right=np.nan)
                okB = (covB > 0.5) & np.isfinite(refB)
                if okB.sum() >= 20:
                    resid = okk[half:][okB] - refB[okB]
                    sv = float(np.nanstd(resid))
                    if np.isfinite(sv) and sv > 0:
                        sig = float(np.clip(sv * P["gr_sig_mult"], P["gr_sig_min"],
                                            max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
                        mref = float(np.nanmedian(np.abs(resid / sig) ** P["gr_power"]))
        sig = sig * float(P.get("v88_self_sig_scale", 1.0))
        srcs.append(("self", ref, conf, float(P.get("v88_beta_self", 1.0)), sig, mref))
    if "nbr" in src:
        ref = gg.copy(); conf = np.zeros(G); sig = gs_raw; mref_n = 1.0
        if c.get("nbr_pairs") is not None:
            nt, ng = c["nbr_pairs"]
            ngs = _smooth(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
            med, cnt, sd = _bin_stats(nt, ngs, gmin, gst, G)
            m = np.isfinite(med)
            if m.any():
                rf = np.where(m, med, gg)
                # ★近傍井はGR較正差があるので、本坑井prefixに対し affine で本坑井スケールへ写像
                if int(P.get("v88_nbr_affine", 1)):
                    at_k = np.interp(ktvt, tvt_g, rf)
                    v = np.isfinite(at_k) & np.isfinite(gr_full[kpos])
                    if v.sum() >= 20 and np.std(at_k[v]) > 1e-6:
                        a, b = np.polyfit(at_k[v], gr_full[kpos][v], 1)
                        a = float(np.clip(a, 0.3, 3.0))
                        rf = np.where(m, a * rf + b, gg)
                # ★σ_nbr は「近傍参照が本坑井prefixをどれだけ予測できたか」の実測残差(well-posed)
                at_k2 = np.interp(ktvt, tvt_g, rf)
                sig = _sig_of(gr_full[kpos], at_k2, P)
                mref_n = float(np.nanmedian(np.abs((gr_full[kpos] - at_k2) / sig) ** P["gr_power"]))
                ref = rf; conf = np.where(m, np.clip(cnt / KC, 0.0, 1.0), 0.0)
        srcs.append(("nbr", ref, conf, float(P.get("v88_beta_nbr", 1.0)), sig, mref_n))

    # ---- チャネル分解(raw / lf / hf) ----
    r_tvt = int(round(float(P.get("v88_lf_tvt", 4.0)) / gst))
    r_tvt2 = int(round(float(P.get("v88_lf_tvt2", 12.0)) / gst))
    r_md = int(round(float(P.get("v88_lf_md", 100.0))))
    w_raw = float(P.get("v88_w_raw", 1.0)); w_lf = float(P.get("v88_w_lf", 0.0)); w_hf = float(P.get("v88_w_hf", 0.0))
    obs_lf_full = _ma(gr_full, r_md); obs_hf_full = gr_full - obs_lf_full

    refs = []; obss = []; sigs = []; ws = []; masks = []; mrefs = []
    for (nm, gref, gmask, beta, sig_ov, mref_ov) in srcs:
        ref_lf = _ma(gref, r_tvt); ref_hf = gref - ref_lf
        tvt_grid = gmin + gst * np.arange(len(gref))
        for (tag, wch, refv, obsv) in (("raw", w_raw, gref, gr_full),
                                       ("lf", w_lf, ref_lf, obs_lf_full),
                                       ("hf", w_hf, ref_hf, obs_hf_full),
                                       # ★lfref: 参照のみTVT平滑・観測は生。観測をMD変換しないので
                                       #   E2の領域不整合(一般則1)を回避しつつ「絶対レベル/低周波」を尤度に入れる。
                                       ("lfref", float(P.get("v88_w_lfref", 0.0)), ref_lf, gr_full),
                                       ("lfref2", float(P.get("v88_w_lfref2", 0.0)), _ma(gref, r_tvt2), gr_full)):
            if wch <= 0:
                continue
            if tag == "raw" and nm == "tw":
                sig = gs_raw            # ★base(pf_engine_v65)と完全同一のσ(格子化前twで較正)=回帰一致のため
            elif tag == "raw" and sig_ov is not None:
                sig = sig_ov            # ★self=bin内std / nbr=prefix実測残差(上で算出)
            else:
                sig = _sig_of(obsv[kpos], np.interp(ktvt, tvt_grid, refv), P)
            refs.append(refv); obss.append(obsv[c["ev_pos"]]); sigs.append(sig)
            ws.append(wch * beta); masks.append(gmask)
            mrefs.append(mref_ov if (tag == "raw" and mref_ov is not None) else 0.0)
    if not refs:      # 全重み0は不正 → rawへフォールバック
        refs = [gg]; obss = [gr_full[c["ev_pos"]]]; sigs = [gs_raw]; ws = [1.0]; masks = [np.ones(len(gg))]; mrefs = [0.0]

    # ★E12 行ごと情報量重み: typewell GR分布での観測値の希少性(-log p)。平均1に正規化。
    rw = np.ones(len(c["ev_pos"]))
    GAM = float(P.get("v88_infow", 0.0))
    if GAM > 0:
        lo, hi = float(np.nanmin(gg)), float(np.nanmax(gg))
        if hi > lo:
            nb = 48
            h, edges = np.histogram(gg, bins=nb, range=(lo, hi))
            dens = h.astype(float) / max(h.sum(), 1) + 1e-3
            oi = np.clip(((gr_full[c["ev_pos"]] - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
            info = -np.log(dens[oi])
            info = info - info.min()
            rw = 1.0 + GAM * (info / max(float(np.mean(info)), 1e-9) - 1.0)
            rw = np.clip(rw, 0.2, 5.0)
    uw = np.ones(len(gg))
    if float(P.get("v88_uniq_w", 0.0)) > 0:
        u = _uniqueness(gg, gst, float(P.get("v88_uniq_win", 6.0)),
                        float(P.get("v88_uniq_lag", 30.0)), float(P.get("v88_uniq_floor", 0.3)))
        a = float(P.get("v88_uniq_w", 0.0))
        uw = (1.0 - a) + a * u        # a=0で1.0(無効), a=1で U そのもの

    return dict(md=c["md"], z=c["z"], gr=gr_full[c["ev_pos"]], gg=gg, gmin=gmin, gst=gst,
                gs=gs_raw, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]),
                anc=c["anc"], ancs=c["ancs"],
                ch_ref=np.stack(refs, 0), ch_obs=np.stack(obss, 0), ch_mask=np.stack(masks, 0),
                ch_sig=np.asarray(sigs, float), ch_w=np.asarray(ws, float),
                ch_mref=np.asarray(mrefs, float), uw=uw, rw=rw)


def _pf_core_v88(md, z, gr, valid, grid, glen, vmin, step, ls, ir, anc, ancs, P, N, device, gen, ch, dtype=DTYPE):
    B, T = md.shape
    C = ch["ref"].shape[1]
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    USE_A = bool(P.get("use_anchor", True)); AMUL = float(P.get("anchor_mult", 20.0))
    RAMP = float(P.get("anchor_ramp_md", 0.0)); LKF = float(P.get("lk_floor", 0.0))
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")
    UNIQ = bool(float(P.get("v88_uniq_w", 0.0)) > 0)
    CTXK = int(P.get("v88_ctx_k", 0)); CTXW = float(P.get("v88_ctx_w", 0.0))
    CTXSTR = max(1, int(round(float(P.get("v88_ctx_span", 200.0)) / max(CTXK, 1))))
    NEUTRAL = float(P.get("v88_neutral", 1.0))

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    gidx_last = (glen - 1).clamp_min(0).unsqueeze(1)
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
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
        obs_ok = act & ~torch.isnan(gr[:, i])
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype)
        dp = torch.zeros_like(w)
        for s in range(C):
            gsg = ch["ref"][:, s, :]; gsm = ch["mask"][:, s, :]
            gA = torch.gather(gsg, 1, i0c); gB = torch.gather(gsg, 1, i0c + 1)
            eg = gA * (1 - t) + gB * t
            eg = torch.where(below, gsg[:, 0:1], eg)
            eg = torch.where(above, torch.gather(gsg, 1, gidx_last), eg)
            oi = ch["obs"][:, s, i][:, None]
            d = ((oi - eg) / ch["sig"][:, s][:, None]).abs()
            contrib = (d ** GR_POWER) * ch["w"][:, s][:, None]
            # 被覆=連続信頼度[0,1](tw は全1 → base回帰を保つ)。
            # ★未被覆に 0 を入れると「ミスフィット0=最良」となり粒子を未観測領域へ引き寄せる(致命的)。
            #   較正済みσなら E|z|^p≈1 なので **中立値 NEUTRAL で埋める**(報酬も罰も与えない)。
            mA = torch.gather(gsm, 1, i0c); mB = torch.gather(gsm, 1, i0c + 1)
            cwt = mA * (1 - t) + mB * t
            cwt = torch.where(below, gsm[:, 0:1], cwt)
            cwt = torch.where(above, torch.gather(gsm, 1, gidx_last), cwt)
            wch = ch["w"][:, s][:, None]
            term = contrib * cwt
            if s > 0:
                # ★行ごとに粒子群平均を引く(部分被覆チャネル専用)。
                #   典型ミスフィット定数を事前推定するのは原理的に困難(被覆重複不足で外れる)。
                #   粒子平均を引けば「全粒子に一律に効く成分」(=被覆の有無そのもの)が消え、
                #   粒子間の**相対的な合致差**だけが残る。tw(s=0)は不変なので base回帰を保つ。
                term = term - term.mean(1, keepdim=True)
            dp = dp + term
        # ---- E4: 粒子自身の rate で MD窓 → TVT窓 に写像する文脈マッチ ----
        # 領域整合の唯一の筋: 粒子の (pos, rate) 仮説の下で「直近L ftのGR系列」が参照と合うかを直接検定。
        if CTXK >= 1 and CTXW > 0:
            ref0 = ch["ref"][:, 0, :]; obs0 = ch["obs"][:, 0, :]; sig0 = ch["sig"][:, 0][:, None]
            acc = torch.zeros_like(w)
            for j in range(1, CTXK + 1):
                il = max(0, i - j * CTXSTR)
                dmd = (cur - md[:, il])[:, None]
                v_t = pos - rate * dmd - z[:, il][:, None]
                iit = (v_t - vmin[:, None]) / step[:, None]; i0t = torch.floor(iit).long()
                blw = i0t < 0; abv = i0t >= (glen - 1)[:, None]
                i0tc = i0t.clamp_min(0); i0tc = torch.minimum(i0tc, (glen - 2).clamp_min(0)[:, None])
                tt = iit - i0tc.to(dtype)
                gAt = torch.gather(ref0, 1, i0tc); gBt = torch.gather(ref0, 1, i0tc + 1)
                egt = gAt * (1 - tt) + gBt * tt
                egt = torch.where(blw, ref0[:, 0:1], egt)
                egt = torch.where(abv, torch.gather(ref0, 1, gidx_last), egt)
                dt = ((obs0[:, il][:, None] - egt) / sig0).abs()
                acc = acc + dt ** GR_POWER
            dp = dp + (CTXW / CTXK) * acc
        if UNIQ:
            uA = torch.gather(ch["uw"], 1, i0c); uB = torch.gather(ch["uw"], 1, i0c + 1)
            ue = uA * (1 - t) + uB * t
            ue = torch.where(below, ch["uw"][:, 0:1], ue)
            ue = torch.where(above, torch.gather(ch["uw"], 1, gidx_last), ue)
            # ★dp を直接掛けると「曖昧な位置(U小)ほど dp が小さい=良い」となり逆効果(E3失敗の原因)。
            #   粒子平均まわりの**コントラストだけ**を U で圧縮する(レベルは不変)。
            dpm = dp.mean(1, keepdim=True)
            dp = dpm + ue * (dp - dpm)
        dp = dp * ch["rw"][:, i][:, None]      # ★E12 行ごと情報量重み(粒子間は一様=行間の証拠力配分を変える)
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if LKF > 0:
            lk = (1.0 - LKF) * lk + LKF
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs_ok, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        lkp = torch.exp(-0.5 * (((rate - dipp) / PHYS_SIG) ** 2)).clamp_min(1e-300) if USE_P else torch.ones_like(w)
        if USE_A:
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig; da2 = da * da
            if RAMP > 0:
                tf = ((cur - md[:, 0]).clamp_min(0.0) / RAMP).clamp(0.0, 1.0)[:, None]; da2 = da2 * tf
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)
        w_new = torch.where(obs_ok[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws_ = w_new.sum(1, keepdim=True)
        w = torch.where(ws_ > 0, w_new / ws_.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
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


def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    C = inps[0]["ch_ref"].shape[0]
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    cref = torch.zeros((W, C, Gmax), dtype=dtype); cmask = torch.zeros((W, C, Gmax), dtype=dtype)
    cobs = torch.zeros((W, C, Tmax), dtype=dtype)
    csig = torch.ones((W, C), dtype=dtype); cw = torch.zeros((W, C), dtype=dtype)
    cmref = torch.zeros((W, C), dtype=dtype)
    uw = torch.ones((W, Gmax), dtype=dtype)
    rw = torch.ones((W, Tmax), dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; ls[b] = x["ls"]; ir[b] = x["ir"]
        anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
        if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
        ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        cref[b, :, :G] = torch.from_numpy(x["ch_ref"].astype("float32"))
        cmask[b, :, :G] = torch.from_numpy(x["ch_mask"].astype("float32"))
        if G < Gmax: cref[b, :, G:] = cref[b, :, G - 1:G]
        cobs[b, :, :Tn] = torch.from_numpy(x["ch_obs"].astype("float32"))
        if Tn < Tmax: cobs[b, :, Tn:] = cobs[b, :, Tn - 1:Tn]
        csig[b] = torch.from_numpy(x["ch_sig"].astype("float32"))
        cw[b] = torch.from_numpy(x["ch_w"].astype("float32"))
        cmref[b] = torch.from_numpy(x["ch_mref"].astype("float32"))
        uw[b, :G] = torch.from_numpy(x["uw"].astype("float32"))
        rw[b, :Tn] = torch.from_numpy(x["rw"].astype("float32"))
        if G < Gmax: uw[b, G:] = uw[b, G - 1]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                ch_ref=to(cref), ch_mask=to(cmask), ch_obs=to(cobs), ch_sig=to(csig), ch_w=to(cw), ch_mref=to(cmref), uw=to(uw), rw=to(rw))


def run_forward_ens(inps, P, device, seed, n_seeds):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    ch = dict(ref=rep(pad["ch_ref"]), mask=rep(pad["ch_mask"]), obs=rep(pad["ch_obs"]),
              sig=rep(pad["ch_sig"]), w=rep(pad["ch_w"]), mref=rep(pad["ch_mref"]), uw=rep(pad["uw"]), rw=rep(pad["rw"]))
    pts, ll = _pf_core_v88(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                           pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S),
                           pad["step"].repeat_interleave(S), pad["ls"].repeat_interleave(S),
                           pad["ir"].repeat_interleave(S), rep(pad["anc"]), rep(pad["ancs"]),
                           P, N, device, gen, ch)
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


def predict_errs(P, subset, seed, n_seeds, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        preds = run_forward_ens(inps, P, DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            res[w] = np.asarray(preds[k], float) - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res


def overall_rmse_on(P, subset, seed, n_seeds, chunk):
    errs = predict_errs(P, subset, seed, n_seeds, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    return 1e3 if e.size == 0 else float(np.sqrt(np.mean(e ** 2)))
