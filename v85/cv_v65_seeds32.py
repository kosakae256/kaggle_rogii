# -*- coding: utf-8 -*-
"""
v85 — 提出エンジン(v65 GPU, n_seeds=32, struct/anchor off)で train 773井の r0 vs affine blend をCV測定。
LBと同じエンジンに揃える(pf_tuning n_seeds=8のCVはエンジン不一致だった)。
raw と affine を同一井戸で回し、行別誤差→overall RMSE と 50:50/最適 blend。
使い方: python v85/cv_v65_seeds32.py   (NW=0で全773, NW=200で前哨)
"""
import os, io, sys, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ["V65_TWSRC"] = "tw"
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v86"))
import pf_engine_v65 as pf
import warmup_v65 as wu

NW = int(os.environ.get("NW", "0")) or None
NSEED = int(os.environ.get("NSEED", "32"))
SEED = int(os.environ.get("SEED", "20251707"))     # 別seed(採用判断用)
CHUNK = int(os.environ.get("CHUNK", "120"))

P = pf.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False; P["n_seeds"] = NSEED

wells = pf.list_wells()
if NW: wells = wells[:NW]


def errs_for_rep(rep):
    os.environ["V86_REP"] = rep
    pf.CACHE = {}
    keep = []
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            pf.CACHE[w] = c; keep.append(w)
    res = {}
    t0 = time.time()
    for c0 in range(0, len(keep), CHUNK):
        sub = keep[c0:c0 + CHUNK]
        inps = [pf.make_inputs(pf.CACHE[w], P) for w in sub]
        preds = pf.run_forward_ens(inps, P, pf.DEVICE, SEED, NSEED)
        for k, w in enumerate(sub):
            res[w] = np.asarray(preds[k], float) - pf.CACHE[w]["tvt_true"]
        if (c0 // CHUNK) % 2 == 0:
            print(f"  {rep} {min(c0+CHUNK,len(keep))}/{len(keep)} ({time.time()-t0:.0f}s)", flush=True)
    return res, keep


r_raw, keep = errs_for_rep("raw")
r_aff, _ = errs_for_rep("affine")
common = [w for w in keep if w in r_aff]
Eraw = np.concatenate([r_raw[w] for w in common]); Eaff = np.concatenate([r_aff[w] for w in common])
m = np.isfinite(Eraw) & np.isfinite(Eaff)
Eraw, Eaff = Eraw[m], Eaff[m]


def rmse(e): return float(np.sqrt(np.mean(e ** 2)))


b50 = 0.5 * Eraw + 0.5 * Eaff
best = min((rmse((1 - w) * Eraw + w * Eaff), w) for w in np.linspace(0, 1, 101))
print(f"\n=== v65 GPUエンジン n_seeds={NSEED} ({len(common)}井, 別seed{SEED}, 行プール) ===")
print(f"  r0単体        = {rmse(Eraw):.4f}")
print(f"  affine単体    = {rmse(Eaff):.4f}")
print(f"  ★blend 50:50  = {rmse(b50):.4f}  (r0比 {rmse(b50)-rmse(Eraw):+.4f})")
print(f"  (参考)最適blend= {best[0]:.4f} @w_affine={best[1]:.2f}")
print(f"  相関ρ(r0,affine)= {np.corrcoef(Eraw, Eaff)[0,1]:.3f}")
print(f"  (LB実測: r0=7.597 / blend=7.696)")
