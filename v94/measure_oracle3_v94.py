# -*- coding: utf-8 -*-
"""現状3PF(PF1 off / PF2 on / PF3 off=0.8版trial78)の oracle score を測る。
   ★全別シード(777/314159/271828), n_seeds32, 全井。
   per-row oracle(各行で最良PF) と per-well oracle(各井で最良PF単体) の pooled RMSE。
   参考: 3本ブレンド(非負最適)も併記。出力: oracle3.txt"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "300"))
ART = pf.ART
SEEDS = {1: 777, 2: 314159, 3: 271828}

e1d = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())
bp2 = json.load(open(ART / "v94_pf_2.json", encoding="utf-8"))["best_params"]
bp3 = json.load(open(ART / "v94_pf_3_relaxed.json", encoding="utf-8"))["best_params"]   # 現PF#3=0.8版
ALLW = sorted(set(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None) & set(e1d.keys()))
print(f"[oracle3] 全{len(ALLW)}井, 別シード, PF3=0.8版(trial78)。PF#2走行...", flush=True)
e2d = pf.predict_errs(bp2, ALLW, seed=SEEDS[2], n_seeds=NS, chunk=CHUNK)
print(f"[oracle3] PF#3走行...", flush=True)
e3d = pf.predict_errs(bp3, ALLW, seed=SEEDS[3], n_seeds=NS, chunk=CHUNK)

# per-well pooled: 井ごとに3PFのRMSEを出し、per-well oracle=最小RMSEのPFを選ぶ
allrows = {1: [], 2: [], 3: []}
well_choice = []
orac_row = []   # per-row oracle誤差
blend_rows = {1: [], 2: [], 3: []}
for w in ALLW:
    a = np.asarray(e1d[w], np.float64); b = np.asarray(e2d[w], np.float64); c = np.asarray(e3d[w], np.float64)
    m = min(len(a), len(b), len(c)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m]) & np.isfinite(c[:m])
    A, B, C = a[:m][msk], b[:m][msk], c[:m][msk]
    if len(A) == 0:
        continue
    allrows[1].append(A); allrows[2].append(B); allrows[3].append(C)
    M = np.stack([A, B, C], 1)
    orac_row.append(M[np.arange(len(M)), np.argmin(np.abs(M), 1)])   # per-row 最小|err|
    rk = [np.sqrt(np.mean(A ** 2)), np.sqrt(np.mean(B ** 2)), np.sqrt(np.mean(C ** 2))]
    best = int(np.argmin(rk)); well_choice.append(best)
    # per-well oracle = その井で最良PFの誤差を採用
    blend_rows[best + 1].append([A, B, C][best])

E1 = np.concatenate(allrows[1]); E2 = np.concatenate(allrows[2]); E3 = np.concatenate(allrows[3])
r = [np.sqrt(np.mean(E1 ** 2)), np.sqrt(np.mean(E2 ** 2)), np.sqrt(np.mean(E3 ** 2))]
rmse_row_oracle = np.sqrt(np.mean(np.concatenate(orac_row) ** 2))
# per-well oracle pooled
pw = np.concatenate([np.concatenate(blend_rows[k]) for k in (1, 2, 3) if blend_rows[k]])
rmse_well_oracle = np.sqrt(np.mean(pw ** 2))
# 3本ブレンド(非負最適, min-var)
E = np.stack([E1, E2, E3], 1); C3 = E.T @ E / len(E)
one = np.ones(3); ci = np.linalg.inv(C3); wv = ci @ one / (one @ ci @ one); r_blend = float(np.sqrt(wv @ C3 @ wv))
cnt = np.bincount(well_choice, minlength=3)

out = []
out.append("=== 現状3PF oracle score (★別シード777/314159/271828, n_seeds32, 全井) ===")
out.append(f"  単体RMSE: PF#1(off)={r[0]:.4f}  PF#2(on)={r[1]:.4f}  PF#3(off,0.8)={r[2]:.4f}")
out.append(f"  3本ブレンド(min-var)   = {r_blend:.4f}")
out.append(f"  ★per-well oracle(井ごと最良PF単体) = {rmse_well_oracle:.4f}  (選択内訳 PF1/PF2/PF3={cnt.tolist()}井)")
out.append(f"  ★per-row oracle(行ごと最良PF)      = {rmse_row_oracle:.4f}  (絶対天井)")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "oracle3.txt").write_text(txt, encoding="utf-8")
