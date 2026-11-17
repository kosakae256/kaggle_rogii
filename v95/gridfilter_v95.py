# -*- coding: utf-8 -*-
"""v95 グリッドベイズフィルタ(非PF, 決定論) — 物理項なし/あり トグル。
δ格子(錨からのずれ)上で GR z正規化 emission × dip連続遷移の forward-backward → posterior-mean δ。
PHYS=0: 錨=flat-hold(末端既知TVT水平保持, naive), δ窓広, 錨prior無し(純GR格子)。
PHYS=1: 錨=grfree_anchor(dip整合物理錨), δ窓狭, 錨prior(sig)で0へ引く。
RMSE>15なら実験バグの疑い(要確認)。真TVTは評価のみ。
env: PHYS(0/1) NW(井数) DR(δ半幅) DSTEP GS(GR尺度) TSIG(遷移σ格子単位)
出力: v95/artifacts_v95b/gf_v95_phys{0,1}.npz(pred per row) + テキスト
実行: PHYS=1 NW=200 .venv/Scripts/python.exe v95/gridfilter_v95.py
"""
import os, sys, io, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
TR = "rogii-wellbore-geology-prediction/train"
ART = Path("v95/artifacts_v95")
PHYS = os.environ.get("PHYS", "1") == "1"
NW = int(os.environ.get("NW", "0")) or None
DR = float(os.environ.get("DR", "40" if PHYS else "70"))     # δ半幅ft
DSTEP = float(os.environ.get("DSTEP", "1.0"))
GS = float(os.environ.get("GS", "1.0"))                       # GR尺度(z正規化後なので~1)
TSIG = float(os.environ.get("TSIG", "2.0"))                  # dip連続 遷移σ(格子単位)
ASIG_SCALE = float(os.environ.get("ASIG", "1.0"))            # 錨prior強度(sig倍率)

anchor_d = pickle.loads((ART / "grfree_anchor_train.pkl").read_bytes()) if PHYS else None
delta = np.arange(-DR, DR + 1e-6, DSTEP); D = len(delta)

def gridfilter_well(obs_gr, tw_tvt, tw_gr, anc_tvt, anc_sig):
    """δ格子 forward-backward → posterior-mean δ (絶対TVT = anc + δ)。"""
    T = len(obs_gr)
    pos = anc_tvt[:, None] + delta[None, :]                  # (T,D) 各行×δの絶対TVT
    tw_at = np.interp(pos.ravel(), tw_tvt, tw_gr).reshape(T, D)
    # GR z正規化(井内): obs と tw_at を各々標準化(較正ズレ吸収=GR正規化知見)
    o = (obs_gr - np.nanmean(obs_gr)) / (np.nanstd(obs_gr) + 1e-6)
    twz = (tw_at - np.nanmean(tw_at)) / (np.nanstd(tw_at) + 1e-6)
    d = (o[:, None] - twz) / GS
    E = np.exp(-0.5 * np.clip(d * d, 0, 60))                 # (T,D) emission
    if PHYS and anc_sig is not None:                         # 錨prior: δ=0へ引く(sig格子単位)
        asig = np.clip(anc_sig / DSTEP * ASIG_SCALE, 1.0, D)
        aprior = np.exp(-0.5 * (delta[None, :] / (asig[:, None] * DSTEP)) ** 2)
        E = E * aprior
    E = np.clip(E, 1e-12, None)
    if os.environ.get("MODE", "pmean") == "viterbi":     # ★大域DP MAP経路(別collapse=脱相関狙い)
        logE = np.log(E); W = max(1, int(3 * TSIG)); ks = np.arange(-W, W + 1); pen = 0.5 * (ks / TSIG) ** 2
        logv = np.full((T, D), -1e18); logv[0] = logE[0]; bp = np.zeros((T, D), np.int16)
        for i in range(1, T):
            prev = logv[i - 1]; best = np.full(D, -1e18); bestk = np.zeros(D, int)
            for ki, k in enumerate(ks):
                sh = np.full(D, -1e18)
                if k >= 0: sh[k:] = prev[:D - k]
                else: sh[:D + k] = prev[-k:]
                cand = sh - pen[ki]; upd = cand > best; best[upd] = cand[upd]; bestk[upd] = ki
            logv[i] = logE[i] + best; bp[i] = bestk
        path = np.zeros(T, int); path[T - 1] = int(np.argmax(logv[T - 1]))
        for i in range(T - 1, 0, -1): path[i - 1] = int(np.clip(path[i] - ks[bp[i][path[i]]], 0, D - 1))
        return anc_tvt + delta[path]
    # forward
    fwd = np.empty((T, D)); f = E[0] / E[0].sum(); fwd[0] = f
    for i in range(1, T):
        f = gaussian_filter1d(f, TSIG, mode="nearest") * E[i]
        s = f.sum(); f = f / s if s > 1e-300 else np.full(D, 1.0 / D); fwd[i] = f
    # backward
    bwd = np.empty((T, D)); b = np.ones(D) / D; bwd[T - 1] = b
    for i in range(T - 2, -1, -1):
        b = gaussian_filter1d(b * E[i + 1], TSIG, mode="nearest")
        s = b.sum(); b = b / s if s > 1e-300 else np.full(D, 1.0 / D); bwd[i] = b
    post = fwd * bwd; post /= post.sum(1, keepdims=True) + 1e-300
    pmean = (post * delta[None, :]).sum(1)                    # posterior-mean δ
    return anc_tvt + pmean

def main():
    wells = sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
    if NW: wells = wells[:NW]
    t0 = time.time(); E_all = []; T_all = []; preds = {}; nused = 0; nbad = 0
    for wi, wn in enumerate(wells):
        try:
            hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv")); tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
        except Exception:
            continue
        kn = hw["TVT_input"].notna().to_numpy()
        if kn.sum() < 10 or (~kn).sum() < 30: continue
        hps = int(np.where(kn)[0].max()); ev = slice(hps + 1, len(hw))
        gr = hw["GR"].to_numpy(float)[ev]; tvt = hw["TVT"].to_numpy(float)[ev]
        ti = hw["TVT_input"].to_numpy(float)
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        m = np.isfinite(tt) & np.isfinite(tg); tt, tg = tt[m], tg[m]
        if len(tt) < 8 or len(gr) < 10: continue
        o = np.argsort(tt); tt, tg = tt[o], tg[o]
        gr = pd.Series(gr).interpolate(limit_direction="both").fillna(float(np.nanmean(tg))).to_numpy()
        if PHYS:
            sp = anchor_d.get(wn)
            if sp is None or len(sp["tvt"]) != len(gr): continue
            anc = np.asarray(sp["tvt"], float); asig = np.asarray(sp["sig"], float)
        else:
            anc = np.full(len(gr), float(ti[hps])); asig = None
        pred = gridfilter_well(gr, tt, tg, anc, asig)
        e = pred - tvt; fin = np.isfinite(e)
        E_all.append(e[fin]); T_all.append((pred[fin] - tvt[fin]))
        preds[wn] = pred.astype(np.float32); nused += 1
        if not np.isfinite(pred).all(): nbad += 1
        if (wi + 1) % 50 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(E_all); rmse = float(np.sqrt(np.mean(e ** 2)))
    tag = "物理あり(grfree錨)" if PHYS else "物理なし(flat-hold, 純GR)"
    print(f"\n=== グリッドフィルタ v95 [{tag}] ===")
    print(f"  井={nused} DR={DR} DSTEP={DSTEP} TSIG={TSIG} GS={GS}")
    print(f"  ★overall RMSE = {rmse:.4f}  ({'>15=バグ疑い' if rmse>15 else 'OK(<15)'})")
    print(f"  |e|中央={np.median(np.abs(e)):.2f} / p90={np.percentile(np.abs(e),90):.2f}")
    np.savez(f"v95/artifacts_v95b/gf_v95_phys{int(PHYS)}.npz", **preds)
    print(f"  saved gf_v95_phys{int(PHYS)}.npz ({nused}井)")

if __name__ == "__main__":
    main()
