# -*- coding: utf-8 -*-
"""現PF#3(warmup best 9.70)の全井・別シード相関を確認。PF3@271828 vs PF1(777)/PF2(777)=cross-seed genuine。
   出力: corr3.txt"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "100"))
ART = pf.ART
e1d = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())   # 777
e2d = pickle.loads((ART / "pf_2_errs.pkl").read_bytes())   # 777
bp3 = json.load(open(ART / "v94_pf_3.json", encoding="utf-8"))
print(f"[corr3] PF#3現best RMSE(600井)={bp3['rmse']:.4f} tuning_corr={bp3['corrs']} -> 全井@271828で確認", flush=True)
bp3 = bp3["best_params"]
ALLW = sorted(set(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None) & set(e1d.keys()))
e3d = pf.predict_errs(bp3, ALLW, seed=271828, n_seeds=NS, chunk=CHUNK)


def pooled(d_a, d_b):
    A, B = [], []
    for w in ALLW:
        a = np.asarray(d_a[w], np.float64); b = np.asarray(d_b[w], np.float64)
        m = min(len(a), len(b)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m])
        A.append(a[:m][msk]); B.append(b[:m][msk])
    A = np.concatenate(A); B = np.concatenate(B)
    return float(np.corrcoef(A, B)[0, 1]), np.sqrt(np.mean(A ** 2)), np.sqrt(np.mean(B ** 2))

c31, r3, r1 = pooled(e3d, e1d)
c32, _, r2 = pooled(e3d, e2d)
out = [f"=== PF#3(現best 9.70) 全井相関 (★別シード PF3@271828 vs PF1@777/PF2@777) ===",
       f"  PF#3 全井RMSE = {r3:.4f}  (PF1={r1:.4f} PF2={r2:.4f})",
       f"  ★corr PF3-PF1(off) = {c31:.4f}   (tuning600井では0.796)",
       f"  ★corr PF3-PF2(on)  = {c32:.4f}   (tuning600井では0.602)"]
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "corr3.txt").write_text(txt, encoding="utf-8")
