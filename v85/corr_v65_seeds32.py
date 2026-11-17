# -*- coding: utf-8 -*-
"""
v85 — 提出エンジン(v65 GPU, n_seeds=32)で raw/affine/logamp の行別誤差相関を全773井で測定。
同一井戸・同一seedで3表現を回し、相関行列 + 各blend(50:50, 最適)を出す。
使い方: NW=0 python v85/corr_v65_seeds32.py
"""
import os, io, sys, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ["V65_TWSRC"] = "tw"
from pathlib import Path
import numpy as np, pickle
sys.path.insert(0, str(Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v86"))
import pf_engine_v65 as pf
import warmup_v65 as wu

NW = int(os.environ.get("NW", "0")) or None
NSEED = int(os.environ.get("NSEED", "32"))
SEED = int(os.environ.get("SEED", "20251707"))
CHUNK = int(os.environ.get("CHUNK", "120"))
os.environ.setdefault("V86_LOG_G0", "40.0")
REPS = ["raw", "affine", "logamp"]

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
    res = {}; t0 = time.time()
    for c0 in range(0, len(keep), CHUNK):
        sub = keep[c0:c0 + CHUNK]
        inps = [pf.make_inputs(pf.CACHE[w], P) for w in sub]
        preds = pf.run_forward_ens(inps, P, pf.DEVICE, SEED, NSEED)
        for k, w in enumerate(sub):
            res[w] = np.asarray(preds[k], float) - pf.CACHE[w]["tvt_true"]
    print(f"  {rep} done {len(keep)}井 ({time.time()-t0:.0f}s)", flush=True)
    return res, keep


E = {}; keeps = {}
for rep in REPS:
    E[rep], keeps[rep] = errs_for_rep(rep)
common = [w for w in keeps["raw"] if all(w in E[r] for r in REPS)]
M = {}
for r in REPS:
    M[r] = np.concatenate([E[r][w] for w in common])
n = min(len(M[r]) for r in REPS)
X = np.stack([M[r][:n] for r in REPS])
m = np.isfinite(X).all(0); X = X[:, m]
pickle.dump({r: M[r] for r in REPS}, open(Path("v85/artifacts_v85/corr_v65_s32.pkl"), "wb"))


def rmse(e): return float(np.sqrt(np.mean(e ** 2)))


print(f"\n=== v65 n_seeds={NSEED} 相関 ({len(common)}井, 別seed{SEED}, 行プール) ===")
print("  単体RMSE:", {r: round(rmse(X[i]), 4) for i, r in enumerate(REPS)})
C = np.corrcoef(X)
print("  相関ρ:")
for i, r in enumerate(REPS):
    print(f"    {r:8s} " + " ".join(f"{REPS[j]}={C[i,j]:.3f}" for j in range(len(REPS))))
raw = X[0]
for i, r in enumerate(REPS[1:], 1):
    b50 = 0.5 * raw + 0.5 * X[i]
    best = min((rmse((1 - w) * raw + w * X[i]), w) for w in np.linspace(0, 1, 101))
    print(f"  raw+{r}: 50:50={rmse(b50):.4f}(r0比{rmse(b50)-rmse(raw):+.4f}) / 最適={best[0]:.4f}@w={best[1]:.2f}")
