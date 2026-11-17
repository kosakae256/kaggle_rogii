# -*- coding: utf-8 -*-
"""PFを現在のbest(v94_pf_{IDX}.json)で確定し、全井誤差 pf_{IDX}_errs.pkl を保存(=次PFの相関比較用)。
   tune_v94 の終端処理と同じ(seed=777, n_seeds=32)。途中でtunerを止めて確定する用。
   env: IDX(必須) / NS(32) / CHUNK(300)。実行: python v94/finalize_pf_v94.py"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
IDX = int(os.environ["IDX"]); NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "300"))
ART = pf.ART
bp = json.load(open(ART / f"v94_pf_{IDX}.json", encoding="utf-8"))["best_params"]
print(f"[finalize pf{IDX}] best確定 use_anchor={bp.get('use_anchor')} gs={bp.get('gs_scale'):.3f} "
      f"lik={bp.get('likelihood_scale'):.2f} -> 全井誤差保存(seed=777,n_seeds={NS})", flush=True)
ALLW = sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None)
errs = pf.predict_errs(bp, ALLW, seed=777, n_seeds=NS, chunk=CHUNK)
(ART / f"pf_{IDX}_errs.pkl").write_bytes(pickle.dumps({w: np.asarray(errs[w], np.float32) for w in ALLW}))
e = np.concatenate([errs[w] for w in ALLW]); e = e[np.isfinite(e)]
print(f"[finalize pf{IDX}] 保存完了 {len(ALLW)}井 全井RMSE={np.sqrt(np.mean(e**2)):.4f} -> pf_{IDX}_errs.pkl", flush=True)
