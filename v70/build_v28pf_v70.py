# -*- coding: utf-8 -*-
"""
v28 の既存 PF 出力(pf_ancc / pf_ancc_std)を v70 cache(v68, stride間引き)の行に揃えて保存。
=====================================================================
v28 parquet(frac1.0=全eval行, id=well_行, target=残差d)を坑井ごとに並べ、::EVAL_STRIDE で
v70/v68 cache の行に整列(検証済: target=残差+lk が maxdif0, 長さ一致)。PF再実行なし=v28流用。
出力: v70/artifacts_v70/v28pf_s{EVAL_STRIDE}.pkl  {wid: {ancc, std}}
実行: EVAL_STRIDE=4 .venv/Scripts/python.exe v70/build_v28pf_v70.py
"""
import os, sys, pickle, collections
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
STR = int(os.environ.get("EVAL_STRIDE", "4"))
V28 = Path(os.environ.get("V28_PARQUET", str(PROJ / "v28" / "artifacts_v28" / "data" / "train.parquet")))
CACHE = PROJ / "v68" / "artifacts_v68" / f"cache_v68_s{STR}.pkl"
OUT = PROJ / "v70" / "artifacts_v70"; OUT.mkdir(parents=True, exist_ok=True)

print(f"[v28pf] load {V28.name} (well/pf_ancc/pf_ancc_std) ...")
t = pq.read_table(V28, columns=["well", "pf_ancc", "pf_ancc_std", "target", "last_known_tvt"])
well = np.array(t["well"].to_pylist()); anc = t["pf_ancc"].to_numpy(); ast = t["pf_ancc_std"].to_numpy()
tgt = t["target"].to_numpy(); lk = t["last_known_tvt"].to_numpy()
iw = collections.OrderedDict()
for i, w in enumerate(well): iw.setdefault(w, []).append(i)

cache = pickle.load(open(CACHE, "rb")); cb = {x["wid"]: x for x in cache}
print(f"[v28pf] v70 cache(s{STR}) {len(cache)} wells / v28 {len(iw)} wells")

out = {}; miss = 0; bad = 0
for wid, cw in cb.items():
    if wid not in iw:
        miss += 1; continue
    ii = np.array(iw[wid])
    a = anc[ii][::STR]; s = ast[ii][::STR]
    tv = (tgt[ii] + lk[ii])[::STR]                      # 整列検証用(絶対TVT)
    t70 = np.asarray(cw["target"]); n = min(len(a), len(t70))
    if abs(len(a) - len(t70)) > 2 or np.nanmax(np.abs(tv[:n] - t70[:n])) > 1e-2:
        bad += 1; continue                             # 万一ズレたら錨フォールバック(除外)
    out[wid] = dict(ancc=a[:len(t70)].astype(np.float32), std=s[:len(t70)].astype(np.float32))
p = OUT / f"v28pf_s{STR}.pkl"; p.write_bytes(pickle.dumps(out))
print(f"[v28pf] saved {p.name}  aligned={len(out)}  v28欠け={miss} 整列失敗={bad}")
