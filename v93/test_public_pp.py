# -*- coding: utf-8 -*-
"""public frontier ノートの後処理メソッドを、我々の fwdbwd blend OOF(CV=6.0079)に CPU で適用し
   CV(overall RMSE)が改善するか1つずつ検証する。GPU 不使用。

検証メソッド(cell24/16 より):
  A) Robust deg-3 projection: 井ごとに dU=(tvt+Z)-anchor を正規化MD s で robust polyfit(IRLS) →
     jitter除去+wrong-branch外れ値ダウンウェイト。tvt_fit = (1-bw)*tvt + bw*proj。 (deg=3, bw=0.75)
  B) Savitzky-Golay 平滑: 井ごと tvt に savgol(win, poly)。 (win=17/61, poly=3)
  C) warmup damping: delta *= (1-exp(-md_since/tau))。 (tau=85)
各メソッド単体 + 組合せで CV を比較。
実行: rogii_claude/.venv/Scripts/python.exe v93/test_public_pp.py
"""
import io, sys, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import nnls
from scipy.signal import savgol_filter

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"
DATA = PROJ / "rogii-wellbore-geology-prediction" / "train"
BASE_PQ = ART / "data" / "train.parquet"


def _robfit(s, y, deg=3):
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - np.polyval(c, s)
        sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


# --- 我々の fwdbwd blend OOF 再構成 ---
dg = np.load(ART / "l1_oof_fwdbwd.npz", allow_pickle=True)
gbm = np.mean([dg[k] for k in dg.files if k not in ("y", "well")], 0)
y = dg["y"].astype(np.float64)
dt = np.load(ART / "nn_oof_combo_top192_fwdbwd.npz", allow_pickle=True)
tcn = np.mean([dt[k] for k in dt.files if k.startswith("mdn2")], 0)
dl = np.load(ART / "l5_fwdbwd_oof.npz", allow_pickle=True)
l5 = dl["l5_mean"].astype(np.float64)
M = np.stack([gbm, tcn, l5], 1)
ok = np.all(np.isfinite(M), 1) & np.isfinite(y)
w, _ = nnls(M[ok], y[ok]); w = w / w.sum()
blend = (M @ w).astype(np.float64)   # target 単位(= tvt - lk)
print(f"[blend] nnls w=[GBM {w[0]:.2f}/TCN {w[1]:.2f}/L5 {w[2]:.2f}]")

# --- meta(id, well, last_known_tvt, md_since) ---
tb = pq.read_table(BASE_PQ, columns=["id", "well", "last_known_tvt", "md_since"])
ids = tb.column("id").to_pandas().astype(str).to_numpy()
well = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
mds = tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
N = len(y)
rown = np.array([int(x.split("_")[-1]) for x in ids])   # CSV row index
tvt_raw = lk + blend
true = y + lk


def overall_rmse(pred):
    m = np.isfinite(pred) & np.isfinite(true)
    return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))


print(f"[baseline] raw blend overall RMSE = {overall_rmse(tvt_raw):.4f}  (doc 6.0079 と一致確認)")

# --- 井ごとに CSV(MD/Z/TVT_input)を引いて projection/SG を適用 ---
uniq = pd.unique(well)
tvt_proj = tvt_raw.copy(); tvt_sg17 = tvt_raw.copy(); tvt_sg61 = tvt_raw.copy()
delta = blend.copy()
tvt_warm = tvt_raw.copy()
n_proj = 0
for wi, wid in enumerate(uniq):
    idx = np.where(well == wid)[0]
    try:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
    except Exception:
        continue
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 5:
        continue
    last = kn.iloc[-1]
    anchor = float(last["TVT_input"]) + float(last["Z"])
    ps = float(last["MD"]); end = float(hw["MD"].iloc[-1])
    # 我々の行(idx)を CSV rownum で並べ、その MD/Z を取得
    ri = rown[idx]
    Z = hw["Z"].values[ri].astype(float)
    md = hw["MD"].values[ri].astype(float)
    order = np.argsort(md)                       # MD昇順で fit(井内順序)
    ii = idx[order]; Zo = Z[order]; mdo = md[order]
    tvo = tvt_raw[ii]
    s = (mdo - ps) / max(end - ps, 1e-6)
    # A) projection(deg=3, bw=0.75)
    fit = _robfit(s, (tvo + Zo) - anchor, 3)
    proj_full = (anchor + fit) - Zo
    tvt_proj[ii] = 0.25 * tvo + 0.75 * proj_full
    # B) SG(win17/61)
    for arr, wl0 in ((tvt_sg17, 17), (tvt_sg61, 61)):
        n = len(tvo); wl = min(wl0, n)
        if wl % 2 == 0:
            wl -= 1
        arr[ii] = savgol_filter(tvo, wl, 3) if wl >= 5 else tvo
    n_proj += 1
# C) warmup damping(tau=85): delta を md_since で減衰
for tau in (85.0,):
    d2 = blend * (1.0 - np.exp(-np.maximum(mds, 0.0) / tau))
    tvt_warm = lk + d2

print(f"[projection] 適用 {n_proj}/{len(uniq)} 井")
print("\n=== メソッド別 CV(overall RMSE, target単位) ===")
base = overall_rmse(tvt_raw)
for name, pred in [("raw blend(baseline)", tvt_raw),
                   ("A) robust deg-3 projection(bw0.75)", tvt_proj),
                   ("B) Savitzky-Golay win17", tvt_sg17),
                   ("B) Savitzky-Golay win61", tvt_sg61),
                   ("C) warmup damping tau85", tvt_warm),
                   ("A+B) projection -> SG17", None)]:
    if pred is None:
        # A+B: projection の上に SG17
        pred = tvt_proj.copy()
        for wid in uniq:
            idx = np.where(well == wid)[0]; ri = rown[idx]
            try:
                hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv"); md = hw["MD"].values[ri].astype(float)
            except Exception:
                continue
            order = np.argsort(md); ii = idx[order]; v = pred[ii]
            n = len(v); wl = min(17, n); wl -= (wl % 2 == 0)
            if wl >= 5:
                pred[ii] = savgol_filter(v, wl, 3)
    r = overall_rmse(pred)
    print(f"  {name:38s}: {r:.4f}  ({r-base:+.4f})")
