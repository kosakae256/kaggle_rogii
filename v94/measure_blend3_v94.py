# -*- coding: utf-8 -*-
"""PF#1+PF#2+PF#3 の3本ブレンド利得を実測。★全て別シード(粒子ノイズ共有を排除)。
   PF#1=pf_1_errs.pkl(seed777) / PF#2=v94_pf_2.json@314159 / PF#3=v94_pf_3.json(現best)@271828。
   全井・n_seeds=32・chunk=300(full GPU=tuner停止中に)。2本(PF1+PF2)と3本を比較して PF#3 の限界利得を出す。
   出力: blend3.txt"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "300"))
ART = pf.ART
SEEDS = {1: 777, 2: 314159, 3: 271828}   # ★全て別シード

e1d = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())   # PF#1 = seed777
bp2 = json.load(open(ART / "v94_pf_2.json", encoding="utf-8"))["best_params"]
PF3_JSON = os.environ.get("PF3_JSON", str(ART / "v94_pf_3.json"))   # ★PF#3 params差替可(0.75版比較用)
bp3 = json.load(open(PF3_JSON, encoding="utf-8"))["best_params"]
print(f"[blend3] PF#3 params = {os.path.basename(PF3_JSON)}", flush=True)
ALLW = sorted(set(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None) & set(e1d.keys()))
print(f"[blend3] 全{len(ALLW)}井, 別シード PF1={SEEDS[1]}/PF2={SEEDS[2]}/PF3={SEEDS[3]}, n_seeds={NS}", flush=True)
print(f"[blend3] PF#2走行(seed{SEEDS[2]})...", flush=True)
e2d = pf.predict_errs(bp2, ALLW, seed=SEEDS[2], n_seeds=NS, chunk=CHUNK)
print(f"[blend3] PF#3走行(seed{SEEDS[3]}, 現best)...", flush=True)
e3d = pf.predict_errs(bp3, ALLW, seed=SEEDS[3], n_seeds=NS, chunk=CHUNK)

E1, E2, E3 = [], [], []
for w in ALLW:
    a = np.asarray(e1d[w], np.float64); b = np.asarray(e2d[w], np.float64); c = np.asarray(e3d[w], np.float64)
    m = min(len(a), len(b), len(c)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m]) & np.isfinite(c[:m])
    E1.append(a[:m][msk]); E2.append(b[:m][msk]); E3.append(c[:m][msk])
E1 = np.concatenate(E1); E2 = np.concatenate(E2); E3 = np.concatenate(E3)
E = np.stack([E1, E2, E3], 1)   # (N,3)
r = [np.sqrt(np.mean(E[:, i] ** 2)) for i in range(3)]
C = E.T @ E / len(E)            # 平均二乗行列(誤差は~0平均なので≈共分散)
rho = np.corrcoef(E.T)


def blend_minvar(cov):
    one = np.ones(cov.shape[0]); ci = np.linalg.inv(cov)
    w = ci @ one / (one @ ci @ one); return w, float(np.sqrt(w @ cov @ w))


def blend_nonneg(cols):
    # 単体グリッド(0.05刻み)で非負・和1の最適
    from itertools import product
    best = (None, 1e9)
    if cols == 2:
        for w0 in np.linspace(0, 1, 41):
            e = w0 * E[:, 0] + (1 - w0) * E[:, 1]; rr = np.sqrt(np.mean(e ** 2))
            if rr < best[1]: best = ((w0, 1 - w0), rr)
    else:
        for w0 in np.linspace(0, 1, 21):
            for w1 in np.linspace(0, 1 - w0, int(round((1 - w0) / 0.05)) + 1):
                w2 = 1 - w0 - w1
                if w2 < -1e-9: continue
                e = w0 * E[:, 0] + w1 * E[:, 1] + w2 * E[:, 2]; rr = np.sqrt(np.mean(e ** 2))
                if rr < best[1]: best = ((w0, w1, w2), rr)
    return best


w2, r2mv = blend_minvar(C[:2, :2])
w3, r3mv = blend_minvar(C)
(nnw2, r2nn) = blend_nonneg(2)
(nnw3, r3nn) = blend_nonneg(3)

out = []
out.append("=== PF#1+PF#2+PF#3 ブレンド利得 (★全別シード 777/314159/271828, n_seeds32, 全井) ===")
out.append(f"  単体RMSE: PF#1={r[0]:.4f}  PF#2={r[1]:.4f}  PF#3={r[2]:.4f}(現best)")
out.append(f"  相関: PF1-PF2={rho[0,1]:.3f}  PF1-PF3={rho[0,2]:.3f}  PF2-PF3={rho[1,2]:.3f}")
out.append("")
out.append(f"  【2本 PF1+PF2】非負最適 w={tuple(round(x,2) for x in nnw2)} -> RMSE={r2nn:.4f}   (min-var {r2mv:.4f})")
out.append(f"  【3本 +PF3】   非負最適 w={tuple(round(x,2) for x in nnw3)} -> RMSE={r3nn:.4f}   (min-var {r3mv:.4f})")
out.append(f"  ★PF#3 追加の限界利得: {r3nn-r2nn:+.4f}  (非負最適ベース)")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "blend3.txt").write_text(txt, encoding="utf-8")
