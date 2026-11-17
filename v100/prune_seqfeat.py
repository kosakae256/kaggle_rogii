# -*- coding: utf-8 -*-
"""候補プルーニング: 現行K=71 seqfeat から冗長候補(相関>TH)を削り K' に。
基準=①相関(教師なし)で冗長ペア検出 ②ペア内は個別wRMSE良い方を残す(タイブレークのみ target使用)。
CV過適合回避: 閾値は事前固定(引数)、OOFは見ない。seqfeatをD/S/DX列スライスで直接生成(非破壊)。
usage: python prune_seqfeat.py <TH> <OUT_TAG>   例: 0.95 p95"""
import sys, pickle, numpy as np
from pathlib import Path

TH = float(sys.argv[1]); TAG = sys.argv[2]
ART = Path(__file__).resolve().parent / "artifacts_v100"
SRC = ART / "seqfeat_combo_cf.pkl"
d = pickle.loads(SRC.read_bytes())

# --- 全bin pool で個別wRMSE と 相関 ---
Ds = []; Ts = []; Ws = []
for w, v in d.items():
    Ds.append(v["D"]); Ts.append(v["dtrue"]); Ws.append(v["wgt"])
D = np.concatenate(Ds, 0); T = np.concatenate(Ts, 0); W = np.concatenate(Ws, 0)
m = np.isfinite(T) & (W > 0); D = D[m]; T = T[m]; W = W[m]
K = D.shape[1]


def wr(e):
    e = e.astype(np.float64); ok = np.isfinite(e)
    return float(np.sqrt(np.sum(W[ok] * e[ok] ** 2) / np.sum(W[ok])))


rmse = np.array([wr(D[:, k] - T) for k in range(K)])
Dz = np.nan_to_num(D - np.nanmean(np.where(np.isfinite(D), D, np.nan), 0))
C = np.abs(np.corrcoef(Dz.T)); np.fill_diagonal(C, 0.0)

# --- 貪欲: RMSE良い順に残す。既残と corr>TH の劣候補を削除 ---
keep = np.ones(K, bool)
for i in np.argsort(rmse):
    if not keep[i]:
        continue
    for j in range(K):
        if j != i and keep[j] and C[i, j] > TH and rmse[j] >= rmse[i]:
            keep[j] = False
kept = np.where(keep)[0]
print(f"[prune] TH={TH} K={K} -> 残{len(kept)}本 (削除{K-len(kept)})", flush=True)
print(f"[prune] 削除候補idx: {sorted(set(range(K))-set(kept.tolist()))}", flush=True)

# --- seqfeat の D/S/DX を kept 列にスライスして保存(他キーは不変) ---
out = {}
for w, v in d.items():
    nv = dict(v)
    nv["D"] = v["D"][:, kept]
    nv["S"] = v["S"][:, kept]
    if "DX" in v:
        nv["DX"] = v["DX"][:, kept, :]
    out[w] = nv
OUT = ART / f"seqfeat_{TAG}_cf.pkl"
OUT.write_bytes(pickle.dumps(out))
print(f"[prune] saved {OUT.name}  K'={len(kept)}", flush=True)
