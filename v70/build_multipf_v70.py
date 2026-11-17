# -*- coding: utf-8 -*-
"""
複数PF出力(既存 parquet の pf_ancc_1..K)を v70 cache 行に揃えて保存。
=====================================================================
既定=v69 の3バンク(af/dec10/sharp, median overall 6.72=最強)。バンク間の食い違い(disagreement)も v70 が使う。
v28単体(9.12)より遥かに強い土台。PF再実行なし=既存流用。整列は build_v28pf と同一(::stride, target=残差+lk 検証)。
出力: v70/artifacts_v70/multipf_s{EVAL_STRIDE}.pkl  {wid: {ancc:[K,T], std:[K,T], names:[..]}}
実行: EVAL_STRIDE=4 PF_PARQUET=v69/artifacts_v69/data/train.parquet .venv/Scripts/python.exe v70/build_multipf_v70.py
env: EVAL_STRIDE(4) PF_PARQUET(既定v69)
"""
import os, sys, pickle, collections, re
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
STR = int(os.environ.get("EVAL_STRIDE", "4"))
PQ = Path(os.environ.get("PF_PARQUET", str(PROJ / "v28" / "artifacts_v28" / "data" / "train.parquet")))   # ★既定=v28(複数PF, struct-free)
# 既定=pf_(ancc|self|nbr)_k(=struct-free PF軌跡)。pf_z は弱いので既定除外。PF_COLS でカンマ指定も可
COLPAT = os.environ.get("PF_COLPAT", r"pf_(ancc|self|nbr)_\d+")
CACHE = PROJ / "v68" / "artifacts_v68" / f"cache_v68_s{STR}.pkl"
OUT = PROJ / "v70" / "artifacts_v70"; OUT.mkdir(parents=True, exist_ok=True)

sch = pq.read_schema(PQ)
if os.environ.get("PF_COLS"):
    ancc_cols = [c.strip() for c in os.environ["PF_COLS"].split(",") if c.strip() in sch.names]
else:
    ancc_cols = sorted([c for c in sch.names if re.fullmatch(COLPAT, c)])
if not ancc_cols:
    ancc_cols = [c for c in sch.names if c == "pf_ancc"]           # 単体しか無い parquet
std_cols = [re.sub(r"_(\d+)$", r"_std_\1", c) for c in ancc_cols]   # pf_ancc_1 -> pf_ancc_std_1
std_cols = [c if c in sch.names else None for c in std_cols]
print(f"[multipf] {PQ.parent.parent.name}  banks({len(ancc_cols)})={ancc_cols}")

cols = ancc_cols + [c for c in std_cols if c] + ["well", "target", "last_known_tvt"]
t = pq.read_table(PQ, columns=list(dict.fromkeys(cols)))
well = np.array(t["well"].to_pylist()); tgt = t["target"].to_numpy(); lk = t["last_known_tvt"].to_numpy()
A = np.stack([t[c].to_numpy() for c in ancc_cols], 0)             # [K, N]
S = np.stack([t[c].to_numpy() if c else np.full(len(tgt), 30.0) for c in std_cols], 0)
iw = collections.OrderedDict()
for i, w in enumerate(well): iw.setdefault(w, []).append(i)

cache = pickle.load(open(CACHE, "rb")); cb = {x["wid"]: x for x in cache}
out = {}; miss = 0; bad = 0
for wid, cw in cb.items():
    if wid not in iw:
        miss += 1; continue
    ii = np.array(iw[wid])
    a = A[:, ii][:, ::STR]; s = S[:, ii][:, ::STR]               # [K, Tstr]
    tv = (tgt[ii] + lk[ii])[::STR]
    t70 = np.asarray(cw["target"]); n = min(a.shape[1], len(t70))
    if abs(a.shape[1] - len(t70)) > 2 or np.nanmax(np.abs(tv[:n] - t70[:n])) > 1e-2:
        bad += 1; continue
    out[wid] = dict(ancc=a[:, :len(t70)].astype(np.float32), std=s[:, :len(t70)].astype(np.float32), names=ancc_cols)
p = OUT / f"multipf_s{STR}.pkl"; p.write_bytes(pickle.dumps(out))
# 参考: median overall
pool = []
for wid, cw in cb.items():
    if wid not in out: continue
    med = np.nanmedian(out[wid]["ancc"], 0); tt = np.asarray(cw["target"], float); n = min(len(med), len(tt))
    e = med[:n] - tt[:n]; pool.append(e[np.isfinite(e)])
ov = float(np.sqrt(np.mean(np.concatenate(pool) ** 2)))
print(f"[multipf] saved {p.name}  K={len(ancc_cols)}バンク  aligned={len(out)} 欠け={miss} 整列失敗={bad}  median overall={ov:.3f}")
