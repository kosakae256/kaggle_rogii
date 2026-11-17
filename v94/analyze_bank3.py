# -*- coding: utf-8 -*-
"""v94 振幅バンク #1/#2/#3(=pf_3_amp) の相関・ブレンド利得・oracle を調査(ユーザー指示 2026-07-30)。
★全ペア異シードで genuine: #1@301, #2@302, #3@777。#3@777 は pf_3_errs.pkl(バンク)として保存も兼ねる。
出力: bank3_analysis.txt / bank3_analysis.png"""
import os, sys, io, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94 as pf

ART = pf.ART
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "300"))
p1 = json.load(open(ART / "v94_pf_1.json", encoding="utf-8"))["best_params"]
p2 = json.load(open(ART / "v94_pf_2.json", encoding="utf-8"))["best_params"]
p3 = json.load(open(ART / "v94_pf_3_amp.json", encoding="utf-8"))["best_params"]
ALLW = sorted(w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None)
print(f"[bank3] 全井={len(ALLW)} NS={NS} chunk={CHUNK} 走行: #3@777(bank保存), #1@301, #2@302 …", flush=True)

e3 = pf.predict_errs(p3, ALLW, seed=777, n_seeds=NS, chunk=CHUNK)
ERRS3 = ART / "pf_3_errs.pkl"
ERRS3.write_bytes(pickle.dumps({w: np.asarray(e3[w], np.float32) for w in ALLW}))
ea = np.concatenate([e3[w] for w in ALLW]); ea = ea[np.isfinite(ea)]
print(f"[bank3] #3@777 全井保存 -> {ERRS3.name}  全井RMSE={np.sqrt(np.mean(ea**2)):.4f}", flush=True)
e1 = pf.predict_errs(p1, ALLW, seed=301, n_seeds=NS, chunk=CHUNK); print("[bank3] #1@301 完了", flush=True)
e2 = pf.predict_errs(p2, ALLW, seed=302, n_seeds=NS, chunk=CHUNK); print("[bank3] #2@302 完了", flush=True)

# 井ごとに長さ揃えてプール
E1, E2, E3 = [], [], []; per_well = []
for w in ALLW:
    a = np.asarray(e1[w], np.float64); b = np.asarray(e2[w], np.float64); c = np.asarray(e3[w], np.float64)
    m = min(len(a), len(b), len(c)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m]) & np.isfinite(c[:m])
    if msk.sum() == 0:
        continue
    A, B, C = a[:m][msk], b[:m][msk], c[:m][msk]
    E1.append(A); E2.append(B); E3.append(C); per_well.append((A, B, C))
E1 = np.concatenate(E1); E2 = np.concatenate(E2); E3 = np.concatenate(E3)
E = np.stack([E1, E2, E3], 1)
r = [float(np.sqrt(np.mean(E[:, i] ** 2))) for i in range(3)]
rho = np.corrcoef(E.T)


def minvar(cov):
    one = np.ones(cov.shape[0]); ci = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
    w = ci @ one / (one @ ci @ one); return w, float(np.sqrt(max(0, w @ cov @ w)))


C2 = (E[:, :2].T @ E[:, :2]) / len(E); C3 = (E.T @ E) / len(E)
w2, r2 = minvar(C2); w3, r3 = minvar(C3)
# oracle
orow = []; owell = []; sel = [0, 0, 0]
for A, B, C in per_well:
    M = np.stack([A, B, C], 1)
    orow.append(M[np.arange(len(M)), np.argmin(np.abs(M), 1)])
    rk = [np.sqrt(np.mean(A ** 2)), np.sqrt(np.mean(B ** 2)), np.sqrt(np.mean(C ** 2))]
    bi = int(np.argmin(rk)); sel[bi] += 1; owell.append([A, B, C][bi])
r_row = float(np.sqrt(np.mean(np.concatenate(orow) ** 2)))
r_well = float(np.sqrt(np.mean(np.concatenate(owell) ** 2)))

L = ["=== v94 振幅バンク #1/#2/#3 解析 (★全ペア異シード #1@301/#2@302/#3@777) ==="]
L.append(f"  単体RMSE: #1(off)={r[0]:.4f}  #2(on)={r[1]:.4f}  #3(off,corr≤0.8)={r[2]:.4f}")
L.append(f"  相関ρ: #1-#2={rho[0,1]:.3f}  #1-#3={rho[0,2]:.3f}  #2-#3={rho[1,2]:.3f}")
L.append(f"  2本(#1+#2) min-var = {r2:.4f}  重み={tuple(round(x,2) for x in w2)}")
L.append(f"  3本(+#3)   min-var = {r3:.4f}  重み={tuple(round(x,2) for x in w3)}")
L.append(f"  ★#3追加の限界利得 = {r3-r2:+.4f}")
L.append(f"  per-well oracle = {r_well:.4f}  (最良選択 #1/#2/#3 = {sel}井)")
L.append(f"  per-row  oracle = {r_row:.4f}")
txt = "\n".join(L); print("\n" + txt, flush=True)
(ART / "bank3_analysis.txt").write_text(txt, encoding="utf-8")

# 図: 相関ヒートマップ + RMSE/blend/oracle バー
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im = ax[0].imshow(rho, vmin=-0.2, vmax=1, cmap="RdYlBu_r")
ax[0].set_xticks(range(3)); ax[0].set_yticks(range(3))
ax[0].set_xticklabels(["#1 off", "#2 on", "#3 off'"]); ax[0].set_yticklabels(["#1 off", "#2 on", "#3 off'"])
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{rho[i,j]:.2f}", ha="center", va="center", fontsize=12)
ax[0].set_title("誤差相関 ρ (別シード=genuine)"); fig.colorbar(im, ax=ax[0], fraction=.046)
names = ["#1", "#2", "#3", "2本blend", "3本blend", "well-oracle", "row-oracle"]
vals = [r[0], r[1], r[2], r2, r3, r_well, r_row]
cols = ["C0", "C1", "C2", "gray", "k", "C3", "darkred"]
ax[1].bar(names, vals, color=cols)
for i, v in enumerate(vals):
    ax[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
ax[1].set_ylabel("overall RMSE [ft]"); ax[1].set_title("単体 vs blend vs oracle"); ax[1].grid(axis="y", alpha=.3)
plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right")
plt.tight_layout(); plt.savefig(ART / "bank3_analysis.png", dpi=110)
print("saved bank3_analysis.png", flush=True)
