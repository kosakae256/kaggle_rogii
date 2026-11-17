# -*- coding: utf-8 -*-
"""v89 — 非逐次U-Netのδ分布の top-2 モードに真の枝が入っているか診断。
入っていれば「枝セレクタ」で硬い坑井を叩ける(oracle方向)。SAVE_DIST=1 で保存した分布を使う。"""
import io, sys, pickle, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
from scipy.signal import argrelextrema
A = "v89/artifacts_v89"; V = "v64/artifacts_v64"
ND = 181; NB = 90; STEP = 0.5; OFF = (np.arange(ND) - NB) * STEP
DIST = dict(np.load(f"{V}/v89_dist_f0_dist.npz"))
E = pickle.load(open(f"{A}/learned_boot.pkl", "rb"))
TH = 4.0   # 命中閾値(ft)


def top2_modes(p):
    idx = argrelextrema(p, np.greater_equal, order=3)[0]
    if len(idx) == 0: idx = np.array([int(np.argmax(p))])
    idx = idx[np.argsort(-p[idx])][:2]
    return OFF[idx]


tot = 0; t1 = 0; t2 = 0; hard_tot = 0; hard_t1 = 0; hard_t2 = 0
e_argmax = []; e_oracle2 = []
for w, P in DIST.items():
    if w not in E: continue
    dt = np.asarray(E[w]["dtrue"], float); P = np.asarray(P, float)
    n = min(len(dt), len(P))
    for b in range(n):
        d = dt[b]
        if not np.isfinite(d) or abs(d) > 44: continue
        p = P[b]; am = OFF[int(np.argmax(p))]; modes = top2_modes(p)
        hard = abs(d) > 8   # PF中心からのズレ大=硬い枝
        tot += 1; hard_tot += hard
        h1 = abs(am - d) <= TH; h2 = float(np.min(np.abs(modes - d)) <= TH)
        t1 += h1; t2 += h2; hard_t1 += (h1 and hard); hard_t2 += (h2 and hard)
        e_argmax.append(am - d)
        e_oracle2.append(modes[int(np.argmin(np.abs(modes - d)))] - d)   # top2の良い方
e_argmax = np.array(e_argmax); e_oracle2 = np.array(e_oracle2)
r = lambda e: float(np.sqrt(np.mean(e ** 2)))
print(f"=== v89 δ分布 枝診断 ({tot}bin, うち硬い枝|d|>8ft={hard_tot}) ===")
print(f"  top1(argmax)命中率 = {t1/tot*100:5.1f}%   top2命中率 = {t2/tot*100:5.1f}%")
print(f"  硬い枝: top1 = {hard_t1/max(hard_tot,1)*100:5.1f}%   top2 = {hard_t2/max(hard_tot,1)*100:5.1f}%")
print(f"  δ残差RMSE: argmax = {r(e_argmax):.3f}ft   top2-oracle(良い方選択) = {r(e_oracle2):.3f}ft")
print(f"  → top2 >> top1 なら 枝セレクタで硬い坑井を叩ける(oracle方向)。差が小さいなら分布に真枝が無い")
