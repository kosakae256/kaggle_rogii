# -*- coding: utf-8 -*-
"""projection のパラメータ(deg, blend_weight)をスイープして、我々の blend に効く設定があるか確認。
   井geometry を1回キャッシュ→高速スイープ。各脚(GBM/TCN/L5)単体への projection も試す。GPU不使用。"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import nnls

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"; DATA = PROJ / "rogii-wellbore-geology-prediction" / "train"


def robfit(s, y, deg):
    if len(s) < deg + 2:
        return y.copy()
    c = np.polyfit(s, y, deg)
    for _ in range(4):
        r = y - np.polyval(c, s); sc = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(s, y, deg, w=1.0 / (1.0 + (r / (2.0 * sc)) ** 2))
    return np.polyval(c, s)


dg = np.load(ART / "l1_oof_fwdbwd.npz", allow_pickle=True)
gbm = np.mean([dg[k] for k in dg.files if k not in ("y", "well")], 0); y = dg["y"].astype(np.float64)
dt = np.load(ART / "nn_oof_combo_top192_fwdbwd.npz", allow_pickle=True)
tcn = np.mean([dt[k] for k in dt.files if k.startswith("mdn2")], 0)
l5 = np.load(ART / "l5_fwdbwd_oof.npz", allow_pickle=True)["l5_mean"].astype(np.float64)
M = np.stack([gbm, tcn, l5], 1); ok = np.all(np.isfinite(M), 1) & np.isfinite(y)
w, _ = nnls(M[ok], y[ok]); w = w / w.sum(); blend = M @ w

tb = pq.read_table(ART / "data" / "train.parquet", columns=["id", "well", "last_known_tvt"])
ids = tb.column("id").to_pandas().astype(str).to_numpy(); well = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
rown = np.array([int(x.split("_")[-1]) for x in ids]); true = y + lk


def rmse(p):
    m = np.isfinite(p) & np.isfinite(true); return float(np.sqrt(np.mean((p[m] - true[m]) ** 2)))


# geometry キャッシュ(井 -> (ii MD昇順, Zo, s, anchor))
GEO = {}
for wid in pd.unique(well):
    idx = np.where(well == wid)[0]
    try:
        hw = pd.read_csv(DATA / f"{wid}__horizontal_well.csv")
    except Exception:
        continue
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 5:
        continue
    last = kn.iloc[-1]; anchor = float(last["TVT_input"]) + float(last["Z"])
    ps = float(last["MD"]); end = float(hw["MD"].iloc[-1])
    ri = rown[idx]; Z = hw["Z"].values[ri].astype(float); md = hw["MD"].values[ri].astype(float)
    o = np.argsort(md); GEO[wid] = (idx[o], Z[o], (md[o] - ps) / max(end - ps, 1e-6), anchor)


def apply_proj(base_tvt, deg, bw):
    out = base_tvt.copy()
    for wid, (ii, Zo, s, anchor) in GEO.items():
        tvo = base_tvt[ii]
        fit = robfit(s, (tvo + Zo) - anchor, deg)
        out[ii] = (1 - bw) * tvo + bw * ((anchor + fit) - Zo)
    return out


tvt_blend = lk + blend
print(f"baseline raw blend = {rmse(tvt_blend):.4f}  (w GBM{w[0]:.2f}/TCN{w[1]:.2f}/L5{w[2]:.2f})")
print("\n=== projection スイープ(blend に適用) deg x bw ===")
for deg in (3, 5, 7, 9):
    row = []
    for bw in (0.1, 0.25, 0.5, 0.75, 1.0):
        r = rmse(apply_proj(tvt_blend, deg, bw)); row.append(f"bw{bw}={r:.4f}")
    print(f"  deg{deg}: " + " ".join(row))
print("\n=== 各脚単体に projection(deg5,bw0.5)して blend し直す ===")
for name, leg in [("GBM", gbm), ("TCN", tcn), ("L5", l5)]:
    tvt_leg = lk + leg
    pj = apply_proj(tvt_leg, 5, 0.5) - lk   # projection後の delta
    print(f"  {name} 単体: raw={rmse(lk+leg):.4f} -> proj={rmse(lk+pj+lk-lk):.4f}")
