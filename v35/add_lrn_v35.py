# -*- coding: utf-8 -*-
"""v35: d_meta ランカーの集約軌跡(lrn)を v35 parquet に列追記する。
=====================================================================================
前提: (1) make_v35_parquet.py 実行済(artifacts_v35/data/train.parquet が存在)
      (2) v34/artifacts_v34/q_d_meta_full.parquet(labカーネルから保存した q_oof)が存在
追加列(float32):
  m34_lrn         : d_meta q̂ の softmax(β1.0, 井戸内z)加重平均軌跡(絶対TVT)
  m34_lrn_delta   : m34_lrn − lk
  m34_lrn_vs_cons : m34_lrn − m34_cons(★ランカーが直した「形」の成分だけを分離した列。
                    調査で lrn の価値は datum でなく形と判明したため、この差分が本体)
※ q̂ は OOF なのでリークなし。列が既に在れば置換。
※ 実行後は artifacts_v35/models を削除してから再学習(特徴数が変わるためfold pklキャッシュ無効)。
実行: .venv/Scripts/python.exe v35/add_lrn_v35.py(~1分)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V34_ART = PROJECT / "v34" / "artifacts_v34"
V35_PQ = PROJECT / "v35" / "artifacts_v35" / "data" / "train.parquet"
BETA = 1.0
SRCS = ("tw", "self", "nbr"); FLTS = (("sm", "ps"), ("fw", "pfw"))

table = pq.read_table(V35_PQ)
g = table.column("well").to_pandas().to_numpy()
lk = table.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
has = table.column("pf_nbr_has").to_numpy(zero_copy_only=False).astype(np.float64)
cons_col = table.column("m34_cons").to_numpy(zero_copy_only=False).astype(np.float64)
wid = list(pd.unique(g)); rows_of = {w: np.where(g == w)[0] for w in wid}
nrows = len(g)
DUMPS = {k: np.load(V34_ART / f"seed_dump_pf{k}.npz") for k in (1, 2, 3)}
KEYS = {k: set(DUMPS[k].files) for k in (1, 2, 3)}
qt = pd.read_parquet(V34_ART / "q_d_meta_full.parquet")
qcol = [c for c in qt.columns if c != "well"][0]
QG = {w: sub[qcol].to_numpy(np.float64) for w, sub in qt.groupby("well", sort=False)}
print(f"rows={nrows} wells={len(wid)} | q={qcol}({len(qt)}行)")

C_LRN = np.full(nrows, np.nan, np.float32)
n_done = n_skip = 0
for w in wid:
    r = rows_of[w]
    nbr_ok = np.nanmean(has[r]) > 0.5
    trajs, wts = [], []
    for k in (1, 2, 3):
        z = DUMPS[k]
        for src in SRCS:
            if src == "nbr" and not nbr_ok: continue
            if f"{w}|ps_{src}" not in KEYS[k]: continue
            for flt, pref in FLTS:
                trajs.append(z[f"{w}|{pref}_{src}"])
    if not trajs or w not in QG:
        n_skip += 1; continue
    T = min(min(t_.shape[1] for t_ in trajs), len(r))
    if T < 5:
        n_skip += 1; continue
    ALL = np.concatenate([t_[:, :T].astype(np.float64) for t_ in trajs], 0)
    cons = np.median(ALL, axis=0)
    ds = np.nanmedian(ALL - cons[None, :], axis=1)
    fin = np.isfinite(ds)
    ALLf = ALL[fin]
    q = QG[w]
    if len(q) != len(ALLf):
        print(f"  [warn] {w}: q {len(q)} vs traj {len(ALLf)} 不一致 → skip"); n_skip += 1; continue
    qz = (q - q.mean()) / (q.std() + 1e-9)
    wgt = np.exp(-BETA * qz); wgt /= wgt.sum()
    C_LRN[r[:T]] = (wgt[:, None] * ALLf).sum(0).astype(np.float32)
    n_done += 1
print(f"lrn計算: {n_done}井戸(skip {n_skip})")

new_cols = {
    "m34_lrn": C_LRN,
    "m34_lrn_delta": (C_LRN.astype(np.float64) - lk).astype(np.float32),
    "m34_lrn_vs_cons": (C_LRN.astype(np.float64) - cons_col).astype(np.float32),
}
names = set(table.schema.names)
for c, arr in new_cols.items():
    if c in names:
        table = table.drop_columns([c])
    table = table.append_column(c, pa.array(arr, type=pa.float32()))
pq.write_table(table, V35_PQ)
print(f"saved: {V35_PQ}  cols={table.num_columns}(m34_lrn/m34_lrn_delta/m34_lrn_vs_cons 追記)")
print("★次: artifacts_v35/models を削除してから train_models_v35.ipynb を再実行(特徴数が変わったため)")
