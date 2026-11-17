# -*- coding: utf-8 -*-
"""
v79 — K16版gf特徴npy生成(v48 gf_feature.npy の置換。submit c2 の _gf_one と同一アルゴリズム)
==============================================================================================
実行順: create_candidates_v79.py → 本スクリプト → train_tabular_v79.ipynb
アルゴリズム: v48 grid filter(stride16, ±45ft/181bin, SIGG6, TAU0.35, LPR0.3)を
帯中心=K16錨(tvt), prior σ=K16 sig で実行(struct完全不使用)。submit側と厳密同一形。
列: gf_delta = (帯中心+d̂)−last_known / gf_std = posterior std / gf_vs_struct = d̂(錨からのオフセット)
行整列: v79 train.parquet の well ブロック順。
出力: v79/artifacts_v79/gf_feature_v79.npy (N_FWD,3) float32
"""
import io, sys, pickle, time, glob, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch
import pyarrow.parquet as pq

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
SRC = P0 / "v79" / "artifacts_v79" / "data" / "train.parquet"
ANC = pickle.loads((P0 / "v77" / "artifacts_v77" / "k16_anchor_train.pkl").read_bytes())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_STRIDE = 16; _Dg = 45.0; _NDg = 181; _GRS = 25; _SIGG = 6.0; _TAU = 0.35; _LPR = 0.3
_gnp = np.linspace(-_Dg, _Dg, _NDg)
_gfgrid = torch.tensor(_gnp, dtype=torch.float32, device=DEV)


def _gf_one(g_ev, struct, sig, tw_tvt, tw_gr):
    n = len(g_ev); nb = (n + _STRIDE - 1) // _STRIDE; bid = np.arange(n) // _STRIDE
    def bm(a):
        o = np.zeros(nb); c = np.zeros(nb); np.add.at(o, bid, a); np.add.at(c, bid, 1); return o / np.maximum(c, 1)
    g = pd.Series(g_ev).rolling(_GRS, center=True, min_periods=1).mean().to_numpy(); g = bm(g)
    st = bm(struct); sg = np.clip(bm(sig), 1, 60)
    z = st[:, None] + _gnp[None, :]; eg = np.interp(z.ravel(), tw_tvt, tw_gr).reshape(nb, _NDg)
    L = np.exp(-0.5 * ((g[:, None] - eg) / _SIGG) ** 2); Pr = np.exp(-0.5 * (_gnp[None, :] / sg[:, None]) ** 2)
    obs = torch.tensor(L * Pr ** _LPR, dtype=torch.float32, device=DEV)
    obs = obs / obs.sum(1, keepdim=True).clamp_min(1e-30)
    dd = _gfgrid[:, None] - _gfgrid[None, :]; M = torch.exp(-0.5 * (dd / _TAU) ** 2)
    M = M / M.sum(0, keepdim=True).clamp_min(1e-30)
    a = torch.zeros(nb, _NDg, device=DEV); a[0] = obs[0]
    for t in range(1, nb):
        pred = M @ a[t - 1]; a[t] = (pred.clamp_min(1e-30) * obs[t]); a[t] = a[t] / a[t].sum().clamp_min(1e-30)
    b = torch.zeros(nb, _NDg, device=DEV); b[-1] = 1.0 / _NDg
    for t in range(nb - 2, -1, -1):
        msg = b[t + 1] * obs[t + 1]; b[t] = (M @ msg).clamp_min(1e-30); b[t] = b[t] / b[t].sum().clamp_min(1e-30)
    gam = a * b; gam = gam / gam.sum(1, keepdim=True).clamp_min(1e-30)
    dh = (gam * _gfgrid[None, :]).sum(1); var = (gam * (_gfgrid[None, :] - dh[:, None]) ** 2).sum(1)
    dhat = dh.cpu().numpy(); dstd = np.sqrt(var.cpu().numpy()); z_bin = st + dhat
    return (z_bin[bid], dstd[bid], dhat[bid])


t0 = time.time()
well = pq.read_table(SRC, columns=["well"])["well"].to_numpy(zero_copy_only=False)
N = len(well)
chg = np.flatnonzero(well[1:] != well[:-1]) + 1
starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [N]])
GF = np.zeros((N, 3), np.float32)
ok = miss = err = 0
for s, e in zip(starts, ends):
    wid = str(well[s])
    sp = ANC.get(wid)
    if sp is None:
        miss += 1; continue
    try:
        hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        ga = hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
        fb = float(np.nanmean(tw["GR"])) if np.isfinite(np.nanmean(tw["GR"])) else 0.0
        ga = np.where(np.isfinite(ga), ga, fb)
        epos = hw.index.get_indexer(ev.index)
        g_ev = ga[epos]
        lastk = float(kn["TVT_input"].iloc[-1])
        L = e - s; m = min(L, len(sp["tvt"]), len(g_ev))
        zr, sr, dr = _gf_one(g_ev[:m], np.asarray(sp["tvt"], float)[:m],
                             np.asarray(sp["sig"], float)[:m],
                             tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))
        GF[s:s + m, 0] = (zr - lastk).astype(np.float32)
        GF[s:s + m, 1] = sr.astype(np.float32)
        GF[s:s + m, 2] = dr.astype(np.float32)
        ok += 1
    except Exception as ex:
        err += 1; print(f"  {wid[:8]}: ERROR {type(ex).__name__} {ex}")
np.save(P0 / "v79" / "artifacts_v79" / "gf_feature_v79.npy", GF)
print(f"[saved] gf_feature_v79.npy ({N}行, 一致{ok}/欠{miss}/エラー{err}, {time.time()-t0:.0f}s)")
