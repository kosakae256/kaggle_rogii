# -*- coding: utf-8 -*-
"""
v82 — 夜間実験サイドカー生成(ユーザー指示 2026-07-23夜)
========================================================
(a) 素PF列: v69_4 parquet(K16gate無しの素バンク)から各バンクの軌跡系10列×5バンク=50列を
    v79行順(完全一致検証済)で抜き、pl_プレフィクスで保存 → v82a
(b) prefix特徴: バンク別prefixバックテスト統計(rmse/bias/seed散らばり/ll分散, cut65/85)
    +バンク間不一致 = 42列(井戸レベル→行にブロードキャスト), pfx_プレフィクス → v82b
出力: v82/artifacts_v82/v82_plainpf_train.parquet / v82_prefix_train.parquet
"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v82" / "artifacts_v82"; ART.mkdir(parents=True, exist_ok=True)
PQ79 = P0 / "v79" / "artifacts_v79" / "data" / "train.parquet"
PQ69 = P0 / "v69" / "artifacts_v69_4" / "data" / "train.parquet"
A73 = P0 / "v73" / "artifacts_v73"
LS = 10.536319667317654

well = pq.read_table(PQ79, columns=["well"])["well"].to_numpy(zero_copy_only=False)
N = len(well)

# ---- (a) 素PF 50列 ----
HEADS = ["pf_ancc", "pf_ancc_std", "pf_ancc_delta", "pf_z", "pf_z_delta",
         "pf_vs_z", "pf_vs_spatial", "pf_vs_dense", "pf_rate", "pf_acc"]
sch69 = set(pq.read_schema(PQ69).names)
cols = []
for h in HEADS:
    for suf in ["", "_1", "_2", "_3", "_4"]:
        c = h + suf
        if c in sch69: cols.append(c)
print(f"素PF列: {len(cols)}")
tb = pq.read_table(PQ69, columns=cols)
arrs = {"well": well}
for c in cols:
    arrs["pl_" + c] = tb[c].to_numpy(zero_copy_only=False).astype(np.float32)
pq.write_table(pa.table(arrs), ART / "v82_plainpf_train.parquet", compression="zstd")
print(f"[saved] v82_plainpf_train.parquet ({len(cols)}列)")
del tb, arrs

# ---- (b) prefix特徴 42列(井戸レベル) ----
BANKS = ["p1", "p2", "p3", "p4", "p5"]
def stats(d):
    ll = d["ll"].astype(np.float64)
    w = np.exp((ll - ll.max()) / LS); w /= w.sum()
    e = (w[:, None] * d["err"].astype(np.float64)).sum(0)
    e = e[np.isfinite(e)]
    return (float(np.sqrt(np.mean(e ** 2))), float(np.median(e)),
            float(d["err"].astype(np.float64).std(0).mean()), float(ll.std()))
feat = {}; names = []
for tag, cf in (("c65", "pfx65"), ("c85", "pfx85")):
    for b in BANKS:
        D = pickle.loads((A73 / f"seed_dump_{cf}_{b}.pkl").read_bytes())
        for k, nm in enumerate(("rmse", "bias", "sspread", "llstd")):
            cname = f"pfx_{tag}_{b}_{nm}"; names.append(cname)
            for w2, d in D.items():
                feat.setdefault(w2, {})[cname] = stats(d)[k]
# バンク間不一致
for w2 in feat:
    for tag in ("c65", "c85"):
        v = [feat[w2].get(f"pfx_{tag}_{b}_rmse", np.nan) for b in BANKS]
        feat[w2][f"pfx_{tag}_bankstd"] = float(np.nanstd(v))
names += ["pfx_c65_bankstd", "pfx_c85_bankstd"]
chg = np.flatnonzero(well[1:] != well[:-1]) + 1
starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [N]])
arrs = {"well": well}
for nm in names:
    a = np.full(N, np.nan, np.float32)
    for s, e in zip(starts, ends):
        v = feat.get(str(well[s]), {}).get(nm)
        if v is not None: a[s:e] = v
    arrs[nm] = a
pq.write_table(pa.table(arrs), ART / "v82_prefix_train.parquet", compression="zstd")
nan0 = float(np.isnan(arrs[names[0]]).mean())
print(f"[saved] v82_prefix_train.parquet ({len(names)}列, NaN率{nan0*100:.1f}%)")
