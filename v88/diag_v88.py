# -*- coding: utf-8 -*-
"""
v88 E1診断 — 「今のGRマッチは自己相似性を解けているのか」を PF を回さずに直接測る。
=============================================================================================
真のTVTが分かっている train で、区間ごとに **ずらし量δのミスフィット断面** を作る:
    S_src(δ) = mean_i | (GR_obs(md_i) − ref_src(TVT_true_i + δ)) / σ_src |^power
δ=0 が最小になるべき。ならなければ「その参照ではその区間を原理的に当てられない」。
測る指標:
  - hit@2ft : argmin_δ |S| が |δ|<=2ft に入る区間割合(=判別できている割合)
  - margin  : (|δ|>=5ft での最小S) − S(δ=0)、S(0)で正規化(大きいほど真が際立つ)
  - decoyδ  : 最良デコイの位置(=自己相似の周期)
参照: tw / self(自prefix) / nbr(近傍, affine較正) と、その**合成**(和=尤度の積に対応)。
→ 合成でmargin/hitが上がるなら E5'(合成尤度)に見込みあり。上がらないなら情報が無い。
使い方: NW=200 python v88/diag_v88.py
"""
import os, io, sys, time, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"   # ★build_cacheにnbr_pairsを作らせる(v88_srcとは別)
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

NW    = int(os.environ.get("NW", "200")) or None
SEG   = int(os.environ.get("SEG", "300"))       # 区間長(行)
DMAX  = float(os.environ.get("DMAX", "30"))     # δ探索幅(ft)
DSTEP = float(os.environ.get("DSTEP", "0.5"))
OUT   = PROJECT / "v88" / "artifacts_v88"; OUT.mkdir(parents=True, exist_ok=True)

P = v88.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False
POW = float(P["gr_power"])
DGRID = np.arange(-DMAX, DMAX + 1e-9, DSTEP)
I0 = int(np.argmin(np.abs(DGRID)))
NEUTRAL = 1.0
FAR = np.abs(DGRID) >= 5.0


def refs_of(c):
    """tw / self / nbr の参照格子とσを作る(v88 make_inputs と同じ規則)。"""
    tw_gr_pf = v88._smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = v88._grid(c["tw_tvt"], tw_gr_pf, step=float(P["grid_step"]))
    gr_full = v88._smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"]).astype(np.float64)
    G = len(gg); tvt_g = gmin + gst * np.arange(G)
    kpos, ktvt = c["kpos"], c["kn_tvtin"]
    out = {}
    out["tw"] = (gg, np.ones(G), v88._sig_of(gr_full[kpos], np.interp(ktvt, c["tw_tvt"], tw_gr_pf), P))
    # self
    med, cnt, sd = v88._bin_stats(ktvt, gr_full[kpos], gmin, gst, G)
    m = np.isfinite(med)
    sv = sd[np.isfinite(sd)]
    s_raw = float(np.median(sv)) if sv.size else np.nan
    sig_s = (float(np.clip(s_raw * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
             if np.isfinite(s_raw) else out["tw"][2])
    out["self"] = (np.where(m, med, np.nan), np.where(m, np.clip(cnt / 4.0, 0, 1), 0.0), sig_s)
    # nbr(affine較正)
    rf = np.full(G, np.nan); cf = np.zeros(G); sig_n = out["tw"][2]
    if c.get("nbr_pairs") is not None:
        nt, ng = c["nbr_pairs"]
        ngs = v88._smooth(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
        med2, cnt2, sd2 = v88._bin_stats(nt, ngs, gmin, gst, G)
        m2 = np.isfinite(med2)
        if m2.any():
            r2 = np.where(m2, med2, np.nan)
            at_k = np.interp(ktvt, tvt_g, np.where(m2, med2, np.interp(tvt_g, tvt_g[m2], med2[m2])))
            v = np.isfinite(at_k)
            if v.sum() >= 20 and np.std(at_k[v]) > 1e-6:
                a, b = np.polyfit(at_k[v], gr_full[kpos][v], 1)
                a = float(np.clip(a, 0.3, 3.0)); r2 = a * r2 + b
                at_k = a * at_k + b
            sig_n = v88._sig_of(gr_full[kpos], at_k, P)
            rf = r2; cf = np.where(m2, np.clip(cnt2 / 4.0, 0, 1), 0.0)
    out["nbr"] = (rf, cf, sig_n)
    return out, gr_full, tvt_g


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    BETAS = [0.25, 0.5, 1.0, 2.0, 4.0]
    keys = ["tw", "self", "nbr", "tw+nbr", "tw+self+nbr"] + [f"tw+{b}self" for b in BETAS]
    stats = {k: {"hit": [], "marg": [], "decoy": []} for k in keys}
    nseg = 0; t0 = time.time(); used = 0
    for wi, w in enumerate(wells):
        c = base.build_cache(w)
        if c is None or c["tvt_true"] is None:
            continue
        used += 1
        R, gr_full, tvt_g = refs_of(c)
        obs = gr_full[c["ev_pos"]]
        tt = np.asarray(c["tvt_true"], float)
        n = len(tt)
        for s0 in range(0, n - SEG + 1, SEG):
            sl = slice(s0, s0 + SEG)
            o = obs[sl]; tv = tt[sl]
            ok = np.isfinite(o) & np.isfinite(tv)
            if ok.sum() < SEG // 2:
                continue
            o = o[ok]; tv = tv[ok]
            pos = tv[:, None] + DGRID[None, :]            # (L,D)
            S = {}
            for k, (ref, conf, sig) in R.items():
                if not np.isfinite(ref).any():
                    S[k] = None; continue
                fin = np.isfinite(ref)
                rv = np.interp(pos.ravel(), tvt_g[fin], ref[fin],
                               left=np.nan, right=np.nan).reshape(pos.shape)
                cv = np.interp(pos.ravel(), tvt_g, conf, left=0.0, right=0.0).reshape(pos.shape)
                d = np.abs((o[:, None] - rv) / sig) ** POW
                d = np.where(np.isfinite(d), d, NEUTRAL)
                # ★未被覆は 0 でなく中立値で埋める(0だと未観測位置が最良に見える致命的バイアス)
                S[k] = np.mean(cv * d + (1.0 - cv) * NEUTRAL, axis=0)
            comb = {"tw": S["tw"], "self": S["self"], "nbr": S["nbr"]}
            if S["tw"] is not None and S["self"] is not None:
                for b in BETAS:
                    comb[f"tw+{b}self"] = S["tw"] + b * S["self"]
            for nm, parts in (("tw+nbr", ["tw", "nbr"]),
                              ("tw+self+nbr", ["tw", "self", "nbr"])):
                acc = None
                for p in parts:
                    if S[p] is None: continue
                    acc = S[p].copy() if acc is None else acc + S[p]
                comb[nm] = acc
            for k, prof in comb.items():
                if prof is None or not np.isfinite(prof).any():
                    continue
                pr = np.where(np.isfinite(prof), prof, np.inf)
                am = int(np.argmin(pr))
                stats[k]["hit"].append(1.0 if abs(DGRID[am]) <= 2.0 else 0.0)
                s0v = pr[I0]
                far = pr[FAR]
                if np.isfinite(s0v) and far.size and np.isfinite(far).any():
                    bd = float(np.nanmin(far))
                    stats[k]["marg"].append((bd - s0v) / max(abs(s0v), 1e-9))
                    stats[k]["decoy"].append(float(DGRID[FAR][int(np.nanargmin(far))]))
            nseg += 1
        if (wi + 1) % 50 == 0:
            print(f"  ..{wi+1}/{len(wells)} wells, {nseg} segs ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== v88 E1 曖昧性の解剖 ({used}井, {nseg}区間, 区間長{SEG}行, δ±{DMAX}ft) ===")
    print(f"{'参照':14s} {'hit@2ft':>9s} {'margin中央':>11s} {'margin>0率':>11s} {'|decoyδ|中央':>12s}")
    res = {}
    for k, d in stats.items():
        if not d["hit"]:
            continue
        hit = float(np.mean(d["hit"])); mg = np.asarray(d["marg"], float)
        mgm = float(np.nanmedian(mg)); pos_r = float(np.mean(mg > 0))
        dc = float(np.nanmedian(np.abs(np.asarray(d["decoy"], float))))
        res[k] = dict(hit=hit, margin=mgm, margin_pos=pos_r, decoy=dc, n=len(d["hit"]))
        print(f"{k:14s} {hit*100:8.1f}% {mgm:11.4f} {pos_r*100:10.1f}% {dc:12.1f}")
    json.dump(res, io.open(OUT / f"diag_e1_nw{used}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT / f'diag_e1_nw{used}.json'}")


if __name__ == "__main__":
    main()
