# -*- coding: utf-8 -*-
"""
v78 — v69_4 train.parquet と同一行順の「サイドカー特徴parquet」を生成
====================================================================
入力: v69/artifacts_v69_4/data/train.parquet('well'列のみ読む=省メモリ)
      v77/artifacts_v77/v78_features_train.pkl(773井, eval行aligned 11特徴)
出力: v77/artifacts_v77/v78_sidecar_train.parquet(同一行数・同一行順, well+11列)
train側の使い方: 本体parquetと行順そのままで hstack/concat(axis=1)。
検証: 井戸ブロックごとの行数一致を全井チェック(不一致井は特徴NaN=GBMが自然に無視)。
"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
SRC = P0 / "v69" / "artifacts_v69_4" / "data" / "train.parquet"
F = pickle.loads((ART / "v78_features_train.pkl").read_bytes())

well = pq.read_table(SRC, columns=["well"])["well"].to_numpy(zero_copy_only=False)
N = len(well)
print(f"train.parquet rows={N}")
# 井戸の連続ブロック検出
chg = np.flatnonzero(well[1:] != well[:-1]) + 1
starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [N]])
keys = list(next(iter(F.values())))
cols = {k: np.full(N, np.nan, np.float32) for k in keys}
ok = miss = mism = 0
for s, e in zip(starts, ends):
    wid = str(well[s])
    f = F.get(wid)
    if f is None:
        miss += 1; continue
    L = e - s; Lf = len(f[keys[0]])
    if Lf < L * 0.98:                       # 大幅不足のみ不整合扱い(端数はK16系列の1-2行差)
        mism += 1; continue
    m = min(L, Lf)
    for k in keys:
        cols[k][s:s + m] = f[k][:m]
    ok += 1
print(f"井戸ブロック: 一致{ok} / pkl無し{miss} / 行数不整合{mism}")
tbl = pa.table({"well": well, **{f"v78_{k}": cols[k] for k in keys}})
out = ART / "v78_sidecar_train.parquet"
pq.write_table(tbl, out, compression="zstd")
print(f"[saved] {out.name}  {out.stat().st_size/1e6:.0f}MB  cols={['well']+[f'v78_{k}' for k in keys]}")
nanfrac = float(np.isnan(cols[keys[0]]).mean())
print(f"NaN率(先頭列): {nanfrac*100:.2f}%")
