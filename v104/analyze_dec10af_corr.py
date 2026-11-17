# -*- coding: utf-8 -*-
"""anchor-free dec10 3性格の相関 + ブレンド率(overall pooled RMSE 最小)を調査。
delta_1=jumpy / delta_2=sharp / delta_3=smooth。true delta は train_combo(target)。
実行: .venv/Scripts/python.exe v103/analyze_dec10af_corr.py"""
import sys, io
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

V104 = Path(__file__).resolve().parent
ART = V104 / "artifacts_v104"
FWD = ART / "fwd" / "dec10af_train_lean.parquet"
COMBO = V104.parent / "v100" / "artifacts_v100" / "data" / "train_combo.parquet"
NAMES = {1: "jumpy", 2: "sharp", 3: "smooth"}


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


t = pq.read_table(FWD, columns=["id"] + [f"decaf_ancc_delta_{k}" for k in (1, 2, 3)])
did = t.column("id").to_pandas().astype(str).to_numpy()
D = {k: t.column(f"decaf_ancc_delta_{k}").to_numpy(zero_copy_only=False).astype(np.float64) for k in (1, 2, 3)}

c = pq.read_table(COMBO, columns=["id", "target"])
cid = c.column("id").to_pandas().astype(str).to_numpy()
tgt = c.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
pos = {i: j for j, i in enumerate(cid)}
idx = np.array([pos.get(i, -1) for i in did])
ok = idx >= 0
y = np.full(len(did), np.nan); y[ok] = tgt[idx[ok]]
print(f"[data] dec10af {len(did)}行 / train_combo 突合 {ok.sum()}行", flush=True)

# 単体 RMSE
print("\n=== 単体 overall RMSE(true delta 比)===")
for k in (1, 2, 3):
    print(f"  {NAMES[k]:7s}(delta_{k}): {rmse(D[k], y):.4f}")

# 相関行列(finite 共通行)
print("\n=== ピアソン相関(eval行, 予測値δ同士)===")
for a in (1, 2, 3):
    row = []
    for b in (1, 2, 3):
        m = np.isfinite(D[a]) & np.isfinite(D[b])
        r = np.corrcoef(D[a][m], D[b][m])[0, 1]
        row.append(f"{NAMES[b]}={r:.3f}")
    print(f"  {NAMES[a]:7s}: " + "  ".join(row))

# smooth × sharp のブレンド率(overall RMSE 最小)
s_sharp, s_smooth = D[2], D[3]
ws = np.linspace(0, 1, 101)
sc = [rmse(w * s_sharp + (1 - w) * s_smooth, y) for w in ws]
bi = int(np.argmin(sc)); wsh = ws[bi]
m = np.isfinite(s_sharp) & np.isfinite(s_smooth)
r_ss = np.corrcoef(s_sharp[m], s_smooth[m])[0, 1]
print("\n=== smooth × sharp ブレンド(overall RMSE 最小)===")
print(f"  相関 = {r_ss:.4f}")
print(f"  sharp 単体  = {rmse(s_sharp, y):.4f}")
print(f"  smooth 単体 = {rmse(s_smooth, y):.4f}")
print(f"  ★最適 = sharp {wsh:.2f} + smooth {1-wsh:.2f} : RMSE {sc[bi]:.4f}  (等重み0.5={rmse(0.5*s_sharp+0.5*s_smooth, y):.4f})")
