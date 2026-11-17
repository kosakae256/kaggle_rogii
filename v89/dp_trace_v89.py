# -*- coding: utf-8 -*-
"""
v89 DP-trace — 画像(MD×δ マッチコスト)の上で「血管を1本トレース」する大域的経路最適化。
==============================================================================================
per-bin U-Netが枝を解けない(局所判断)のに対し、DP(Viterbi)は**既知帯アンカーから連続性を保った
大域最小コスト経路**を選ぶ→枝を大域整合で解く。PF/struct/nbr 不使用・非逐次(全体を一括最適化)。
入力: pf_free_all.pkl の NCC/aff チャネル → マッチスコア S[bin,δ] → cost=−S。
DP: path δ(bin) が Σ cost[bin,δ] + λ|Δδ| を最小化(|Δδ|≤SLOPE)、start は δ≈0(lk直後)へ弱prior。
出力: fold0 val RMSE(U-Net12.89と比較) と 全井RMSE。学習なし=CV過適合なし。
使い方: python v89/dp_trace_v89.py  (env: LAM, SLOPE, STARTW, WCH)
"""
import os, io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v89", "artifacts_v89")
EVD = pickle.load(open(os.path.join(ART, "pf_free_all.pkl"), "rb"))
WELLS = sorted(EVD.keys())
NB = 90; ND = 2 * NB + 1; STEP = 0.5
OFF = (np.arange(ND) - NB) * STEP
# チャネル添字: 0d16,1ncc16,2ncc64,3ncc128,4ncc256,5grad,6aff,7self16,8self64,9selfcov,10lk,11mds,12z
CH = {"ncc16": 1, "ncc64": 2, "ncc128": 3, "ncc256": 4, "grad": 5, "aff": 6, "self16": 7, "self64": 8}
WCH = os.environ.get("WCH", "ncc64:1,ncc128:1,ncc256:1,aff:1,self64:0.5")
LAM = float(os.environ.get("LAM", "0.5"))       # 連続性(|Δδ|/ft)ペナルティ
SLOPE = float(os.environ.get("SLOPE", "6.0"))   # bin間の最大δ変化(ft) = dip制約
STARTW = float(os.environ.get("STARTW", "0.3")) # start δ≈0 prior 強さ
SIG_S = 8.0
_wch = [(k.split(":")[0], float(k.split(":")[1])) for k in WCH.split(",")]


def score_of(ev):
    S = np.zeros((ev.shape[1], ev.shape[2]), np.float64)   # (nb,ND)
    for nm, w in _wch:
        S += w * np.asarray(ev[CH[nm]], np.float64)
    return S


def dp_trace(S):
    nb, nd = S.shape
    cost = -S                                    # 低い=良い
    sl = int(round(SLOPE / STEP))
    dp = cost[0] + STARTW * (OFF / SIG_S) ** 2    # start: δ≈0 prior
    back = np.zeros((nb, nd), np.int32)
    idx = np.arange(nd)
    for t in range(1, nb):
        # 各δについて δ'∈[δ-sl, δ+sl] の (dp_prev + LAM|δ-δ'|) 最小
        prev = dp
        best = np.full(nd, np.inf); arg = np.zeros(nd, np.int32)
        for d in range(-sl, sl + 1):
            shifted = np.full(nd, np.inf)
            if d >= 0:
                shifted[d:] = prev[:nd - d] + LAM * abs(d) * STEP
            else:
                shifted[:nd + d] = prev[-d:] + LAM * abs(d) * STEP
            better = shifted < best
            arg[better] = idx[better] - d
            best = np.minimum(best, shifted)
        dp = cost[t] + best; back[t] = np.clip(arg, 0, nd - 1)
    path = np.zeros(nb, np.int32); path[-1] = int(np.argmin(dp))
    for t in range(nb - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return OFF[path]                              # δ(bin) ft


def rmse(subset):
    es = []
    for w in subset:
        e = EVD[w]; S = score_of(e["EV"])
        dpath = dp_trace(S)
        z = e["st"] + dpath[e["bid"]]
        er = z - e["y_ev"]; es.append(er[np.isfinite(er)])
    ee = np.concatenate(es); return float(np.sqrt(np.mean(ee ** 2)))


import time
t0 = time.time()
r0 = rmse(WELLS[0::5]); rall = rmse(WELLS)
print(f"[DP-trace] WCH={WCH} LAM={LAM} SLOPE={SLOPE} STARTW={STARTW}")
print(f"  fold0 val = {r0:.4f}  (U-Net PFフリー12.89 と比較)   全井 = {rall:.4f}   ({time.time()-t0:.0f}s)")
