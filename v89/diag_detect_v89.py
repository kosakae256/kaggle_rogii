# -*- coding: utf-8 -*-
"""
v89 D1改 — 正しい問い: misfitランドスケープは「PFが枝を外した区間」を検出できるか(AUC)。
============================================================================================
前回D1で判明: base(PFを信じる)=区間の70.6%が正しい枝。問題は残り~30%の検出。
ここでは λ量子化をやめ、PF軌跡+δ(fine)の多ソースmisfit断面から:
  - 枝当て率: argmin_δ misfit が真オフセット d(=median(true−PF)) の±4ft内か。base(δ0)と比較。
  - ★検出AUC: gap = (misfit(δ0) − min_{|δ|>=8} misfit)/|misfit(δ0)| が pf_wrong(|d|>5ft)を当てるか。
tw単独 vs tw+self(v88)で比較。struct-free。
使い方: NW=200 python v89/diag_detect_v89.py
"""
import os, io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

NW = int(os.environ.get("NW", "200")) or None
SEED = int(os.environ.get("SEED", "20251707")); NSEED = 32
SEG = int(os.environ.get("SEG", "300")); CHUNK = 100
OUT = PROJECT / "v89" / "artifacts_v89"; OUT.mkdir(parents=True, exist_ok=True)

P = v88.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False; P["n_seeds"] = NSEED
P["tw_source"] = "tw"; P["v88_src"] = "tw+self"; P["v88_beta_self"] = 1.0; P["v88_self_fill"] = 20.0
POW = float(P["gr_power"])
DG = np.arange(-30.0, 30.0 + 1e-9, 0.5); I0 = int(np.argmin(np.abs(DG))); FAR = np.abs(DG) >= 8.0


def auc(score, label):
    label = label.astype(bool); o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(len(score))
    npos = int(label.sum()); nneg = len(label) - npos
    if npos == 0 or nneg == 0: return 0.5
    return float((r[label].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def main():
    wells = base.list_wells()
    if NW: wells = wells[:NW]
    cache = {w: base.build_cache(w) for w in wells}
    cache = {w: c for w, c in cache.items() if c is not None and c["tvt_true"] is not None}
    base.CACHE = cache; v88.CACHE = cache
    ALL = list(cache)
    Pc = base.full_params(dict(wu.R0_PARAMS)); Pc["use_anchor"] = False; Pc["n_seeds"] = NSEED; Pc["tw_source"] = "tw"
    pred = {}
    for c0 in range(0, len(ALL), CHUNK):
        sub = ALL[c0:c0 + CHUNK]
        outs = base.run_forward_ens([base.make_inputs(cache[w], Pc) for w in sub], Pc, base.DEVICE, SEED, NSEED)
        for k, w in enumerate(sub): pred[w] = np.asarray(outs[k], float)
    print(f"[v89] wells={len(ALL)} tw PF done", flush=True)

    rows = []
    for w in ALL:
        c = cache[w]; x = v88.make_inputs(c, P)
        ref_tw = np.asarray(x["ch_ref"][0], float)
        ref_sf = np.asarray(x["ch_ref"][1], float) if x["ch_ref"].shape[0] > 1 else ref_tw
        msk_sf = np.asarray(x["ch_mask"][1], float) if x["ch_mask"].shape[0] > 1 else np.ones_like(ref_tw)
        obs = np.asarray(x["ch_obs"][0], float)
        sig_tw = float(x["ch_sig"][0]); sig_sf = float(x["ch_sig"][1]) if len(x["ch_sig"]) > 1 else sig_tw
        gmin, gst = x["gmin"], x["gst"]; G = len(ref_tw); tvt_g = gmin + gst * np.arange(G)
        p = pred[w]; tt = np.asarray(c["tvt_true"], float); n = min(len(p), len(tt))
        for s0 in range(0, n - SEG + 1, SEG):
            sl = slice(s0, s0 + SEG); pp = p[sl]; tv = tt[sl]
            ok = np.isfinite(pp) & np.isfinite(tv)
            if ok.sum() < SEG // 2: continue
            pp = pp[ok]; tv = tv[ok]; ob = obs[sl][ok]
            d = float(np.median(tv - pp)); pf_wrong = abs(d) > 5.0
            m_tw = np.empty(len(DG)); m_ts = np.empty(len(DG))
            for j, sh in enumerate(DG):
                hyp = pp + sh
                r_tw = np.interp(hyp, tvt_g, ref_tw); r_sf = np.interp(hyp, tvt_g, ref_sf)
                cv = np.interp(hyp, tvt_g, msk_sf)
                dt = np.abs((ob - r_tw) / sig_tw) ** POW
                ds = np.abs((ob - r_sf) / sig_sf) ** POW
                m_tw[j] = np.nanmean(dt); m_ts[j] = np.nanmean(dt + np.where(cv > 0.5, ds, dt))

            def gh(m):
                a = int(np.argmin(m)); far = float(np.nanmin(m[FAR]))
                return (m[I0] - far) / max(abs(m[I0]), 1e-9), (abs(DG[a] - d) <= 4.0)
            g_tw, h_tw = gh(m_tw); g_ts, h_ts = gh(m_ts)
            rows.append((pf_wrong, g_tw, g_ts, h_tw, h_ts, abs(d) <= 4.0))
    R = np.array(rows, float); pfw = R[:, 0]
    print(f"\n=== v89 D1改 検出診断 ({len(R)}区間, 区間{SEG}行) PF誤り率(|d|>5ft)={pfw.mean()*100:.1f}% ===")
    print(f"  枝当て率(argmin_δ が真オフセット±4ft内):")
    print(f"    base(δ0=PF信頼) {R[:,5].mean()*100:5.1f}%   landscape-tw {R[:,3].mean()*100:5.1f}%   tw+self {R[:,4].mean()*100:5.1f}%")
    print(f"  ★PF誤り検出 AUC(gapが|d|>5を当てる):  tw={auc(R[:,1],pfw):.3f}   tw+self={auc(R[:,2],pfw):.3f}")
    print(f"  ★誤り区間だけでの枝訂正率:  base 0%(定義上) / landscape-tw {R[pfw>0.5,3].mean()*100:.1f}% / tw+self {R[pfw>0.5,4].mean()*100:.1f}%")
    print(f"  ★正解区間を壊さない率:  landscape-tw {R[pfw<0.5,3].mean()*100:.1f}% / tw+self {R[pfw<0.5,4].mean()*100:.1f}%")
    json.dump(dict(n=len(R), pfw=float(pfw.mean()), auc_tw=auc(R[:,1],pfw), auc_ts=auc(R[:,2],pfw),
                   hit_base=float(R[:,5].mean()), hit_tw=float(R[:,3].mean()), hit_ts=float(R[:,4].mean())),
              io.open(OUT / "diag_detect.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
