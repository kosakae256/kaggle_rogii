# -*- coding: utf-8 -*-
"""
v73 段c-2 — GR尤度形状の学習(LLR曲線)
=======================================
現行尤度 exp(-0.5·d^power)(d=|GR−tw|/σ_well)を、データ推定の対数尤度比に置換する:
  LLR(d) = log p(d | 真位置) − log p(d | 帯内おとり位置)
- 真位置: 各行の真TVTでのtypewell期待GRとの d。おとり: 真位置±(6..45ft)の一様サンプル位置での d。
- d∈[0,8]を80binでヒストグラム→平滑→LLR→単調性は課さない(データに語らせる)が端は線形外挿。
- fold不要(1次元曲線・全坑井プール=低容量)だが、5-fold OOF版も算出して曲線の安定性を確認。
- 出力: artifacts_v73/llr_curve_v73.npz {dgrid, llr}  → nnpf_v73 LLR=1 で lk=exp(LLRSC·llr(d)) に置換。
使い方: python v73/llr_curve_v73.py
"""
import os, io, sys, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v73" / "artifacts_v73"; ART.mkdir(exist_ok=True)

# r0 のσ較正を忠実再現(nnpf_v73.build_inputs と同一)
GR_SIG_MIN, GR_SIG_MAX, GR_SIG_DEF, GR_SIG_MULT = 11.865793233968713, 44.98927964612316, 119.25397464636981, 3.058318309943017
GR_POWER = 1.7826896851395453
DG = 0.1; DMAX = 8.0
rng = np.random.RandomState(7)

hist_t = np.zeros(int(DMAX / DG)); hist_b = np.zeros(int(DMAX / DG))
wids = sorted({Path(p).stem.replace("__horizontal_well", "")
               for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
nuse = 0
for wn in wids:
    hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns: continue
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 20: continue
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    g = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)
    kpos = hw.index.get_indexer(kn.index)
    resid = g[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GR_SIG_DEF
    gs = float(np.clip(gs * GR_SIG_MULT, GR_SIG_MIN, GR_SIG_MAX))
    ge = g[hw.index.get_indexer(ev.index)]
    yt = ev["TVT"].to_numpy(float)
    m = np.isfinite(ge) & np.isfinite(yt)
    ge, yt = ge[m], yt[m]
    if len(ge) < 50: continue
    sub = slice(0, len(ge), 3)                       # 行間引き(統計は十分)
    ge, yt = ge[sub], yt[sub]
    d_true = np.abs(ge - np.interp(yt, tw_tvt, tw_gr)) / gs
    off = rng.uniform(6.0, 45.0, size=len(yt)) * np.where(rng.rand(len(yt)) < 0.5, -1, 1)
    d_bg = np.abs(ge - np.interp(yt + off, tw_tvt, tw_gr)) / gs
    hist_t += np.histogram(np.clip(d_true, 0, DMAX - 1e-6), bins=int(DMAX / DG), range=(0, DMAX))[0]
    hist_b += np.histogram(np.clip(d_bg, 0, DMAX - 1e-6), bins=int(DMAX / DG), range=(0, DMAX))[0]
    nuse += 1
print(f"wells used {nuse}  真位置 {int(hist_t.sum())}行 / おとり {int(hist_b.sum())}行")

# 平滑(移動平均5bin)→LLR
def _sm(h):
    return pd.Series(h).rolling(5, center=True, min_periods=1).mean().to_numpy()
pt = _sm(hist_t) + 1.0; pb = _sm(hist_b) + 1.0
pt /= pt.sum(); pb /= pb.sum()
llr = np.log(pt / pb)
dgrid = (np.arange(int(DMAX / DG)) + 0.5) * DG
np.savez(ART / "llr_curve_v73.npz", dgrid=dgrid, llr=llr.astype(np.float32))
print("LLR曲線(サンプル): d=0.25→%.3f d=0.75→%.3f d=1.5→%.3f d=2.5→%.3f d=4→%.3f" % tuple(
    llr[np.searchsorted(dgrid, x)] for x in (0.25, 0.75, 1.5, 2.5, 4.0)))
gaus = -0.5 * dgrid ** GR_POWER
print("(参考)現行ガウス核 logは d=0.25→%.3f d=0.75→%.3f d=1.5→%.3f d=2.5→%.3f d=4→%.3f" % tuple(
    gaus[np.searchsorted(dgrid, x)] for x in (0.25, 0.75, 1.5, 2.5, 4.0)))
print(f"[saved] llr_curve_v73.npz")
