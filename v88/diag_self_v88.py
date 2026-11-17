# -*- coding: utf-8 -*-
"""
v88 診断B — self/nbr 参照は「真の位置で実際に当たっている」のか?
=========================================================================
尤度の項として使えるための必要条件は、**真のTVTで残差が小さい**こと。
そうでなければ、その項は正解の粒子を罰する=PFを壊す(E5の症状)。
eval帯の真TVT上で各参照の残差を実測し、σ較正・被覆・典型ミスフィットと合わせて見る。
使い方: NW=60 python v88/diag_self_v88.py
"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v88"))
os.environ["V65_TWSRC"] = "tw+self+nbr"
import pf_engine_v65 as base
import pf_engine_v88 as v88
import warmup_v65 as wu

NW = int(os.environ.get("NW", "60")) or None
P = v88.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False
P["v88_src"] = "tw+self+nbr"
POW = float(P["gr_power"])

rows = []
wells = base.list_wells()
if NW: wells = wells[:NW]
for w in wells:
    c = base.build_cache(w)
    if c is None or c["tvt_true"] is None:
        continue
    x = v88.make_inputs(c, P)
    tt = np.asarray(c["tvt_true"], float)
    obs = np.asarray(x["ch_obs"], float)          # (C,T)
    ref = np.asarray(x["ch_ref"], float)          # (C,G)
    msk = np.asarray(x["ch_mask"], float)
    sig = np.asarray(x["ch_sig"], float)
    mrf = np.asarray(x["ch_mref"], float)
    gmin, gst = x["gmin"], x["gst"]; G = ref.shape[1]
    tvt_g = gmin + gst * np.arange(G)
    ok = np.isfinite(tt)
    if ok.sum() < 50:
        continue
    rec = {"well": w}
    for s in range(ref.shape[0]):
        rv = np.interp(tt[ok], tvt_g, ref[s])
        cv = np.interp(tt[ok], tvt_g, msk[s])
        d = np.abs((obs[s][ok] - rv) / sig[s]) ** POW
        rec[s] = dict(sig=float(sig[s]), mref=float(mrf[s]),
                      cov=float(np.mean(cv > 0.5)),
                      dmed=float(np.nanmedian(d[cv > 0.5])) if (cv > 0.5).any() else np.nan,
                      contrib=float(np.nanmedian((cv * (d - mrf[s]))[np.isfinite(d)])))
    rows.append(rec)

names = ["tw", "self", "nbr"]
print(f"\n=== v88 診断B: 真TVT上での各参照の実測 ({len(rows)}井) ===")
print("  ※ dmed = 真の位置での |残差/σ|^p の中央値。tw基準と比べて大きいほど『真の位置を罰する』")
print(f"{'src':6s} {'σ中央':>8s} {'mref中央':>9s} {'被覆率':>8s} {'dmed中央':>10s} {'寄与中央':>10s} {'dmed>mrefの井戸率':>18s}")
for s, nm in enumerate(names):
    if not rows or s not in rows[0]:
        continue
    sg = np.array([r[s]["sig"] for r in rows]); mr = np.array([r[s]["mref"] for r in rows])
    cv = np.array([r[s]["cov"] for r in rows]); dm = np.array([r[s]["dmed"] for r in rows])
    ct = np.array([r[s]["contrib"] for r in rows])
    bad = np.nanmean((dm > mr).astype(float))
    print(f"{nm:6s} {np.nanmedian(sg):8.2f} {np.nanmedian(mr):9.3f} {np.nanmedian(cv)*100:7.1f}% "
          f"{np.nanmedian(dm):10.3f} {np.nanmedian(ct):10.3f} {bad*100:17.1f}%")
