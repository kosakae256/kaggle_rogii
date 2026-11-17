# -*- coding: utf-8 -*-
"""v100 逆方向DA を L5 に train側DAとして投入し、3seed で baseline と比較。
段取り: (1)3レグDA parquet(v100/da_data)を train_combo同型に結合(is_backward=1)→ train_aug_bwd_combo.parquet
        (2)aug seqfeat 生成(v100 ED, K=71) (3)L5 +DA 3seed(V98_DA=1, L5_SF_BWD) → l5_da3_oof.npz
        (4)baseline l5_base3(5.6667, DA無し3seed)と OOF overall RMSE 比較。
★リーク防止: aug井は元wellの fold を継承+val除外(train_l456_v98 が処理)。DA は train側のみ。
実行: .venv/Scripts/python.exe v100/run_da_l5_exp.py   (env: SKIP_COMBINE/SKIP_SEQFEAT/SKIP_L5)"""
import os, sys, io, json, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V100 = Path(__file__).resolve().parent
PROJ = V100.parent
V99 = PROJ / "v99"
ART = V100 / "artifacts_v100"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
DAD = V100 / "da_data"
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")

OUT96 = DAD / "v96_bwd.parquet"; OUT97 = DAD / "v97_bwd.parquet"; OUT100 = DAD / "v100_bwd.parquet"
AUGCOMBO = ART / "data" / "train_aug_bwd_combo.parquet"
AUGSF = ART / "seqfeat_aug_bwd_cf.pkl"
DIFF97 = json.loads((V99 / "artifacts_v99" / "v97_diff_cols.json").read_text(encoding="utf-8"))
DIFF100 = json.loads((ART / "v100_diff_cols.json").read_text(encoding="utf-8"))

ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
              + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
              + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V99 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def combine():
    print("\n==== DA 3レグ結合(is_backward=1) ====", flush=True)
    w96 = pq.read_table(OUT96, columns=["well"]).column("well").to_pandas().to_numpy()
    w97 = pq.read_table(OUT97, columns=["well"]).column("well").to_pandas().to_numpy()
    w100 = pq.read_table(OUT100, columns=["well"]).column("well").to_pandas().to_numpy()
    assert np.array_equal(w96, w97) and np.array_equal(w96, w100), "DA レグ間 well 並び不一致"
    pf96 = pq.ParquetFile(OUT96); pf97 = pq.ParquetFile(OUT97); pf100 = pq.ParquetFile(OUT100)
    writer = None; nrow = 0
    for b96, b97, b100 in zip(pf96.iter_batches(batch_size=200000),
                              pf97.iter_batches(batch_size=200000, columns=DIFF97),
                              pf100.iter_batches(batch_size=200000, columns=DIFF100)):
        cols = list(b96.columns); names = list(b96.schema.names)
        for j, c in enumerate(DIFF97):
            cols.append(b97.column(j)); names.append("v97_" + c)
        for j, c in enumerate(DIFF100):
            cols.append(b100.column(j)); names.append("v100_" + c)
        n = b96.num_rows
        cols.append(pa.array(np.ones(n, np.float32))); names.append("is_backward")   # ★DA=1
        cols.append(pa.array(np.zeros(n, np.float32))); names.append("is_fwdaug")
        tbl = pa.Table.from_arrays(cols, names=names)
        if writer is None:
            writer = pq.ParquetWriter(AUGCOMBO, tbl.schema, compression="zstd")
        writer.write_table(tbl); nrow += n
    writer.close()
    print(f"[combine] {AUGCOMBO.name} 行={nrow} 列={len(pq.read_schema(AUGCOMBO).names)}", flush=True)


def rmse(npz):
    p = ART / npz
    if not p.exists():
        return None, 0
    d = np.load(p); y = d["y"]; m = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
    ok = np.isfinite(y) & np.isfinite(m) & (m != 0)
    return float(np.sqrt(np.mean((m[ok] - y[ok]) ** 2))), int(ok.sum())


if os.environ.get("SKIP_COMBINE", "0") != "1" and not AUGCOMBO.exists():
    combine()
if os.environ.get("SKIP_SEQFEAT", "0") != "1":
    _run("build_seqfeat_v97.py", dict(SF_IN=str(AUGCOMBO), SF_OUT=str(AUGSF),
         CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES,
         ROGII_SCRIPTS=str(V99), ROGII_DATA=DATA), "aug seqfeat(K=71)")
if os.environ.get("SKIP_L5", "0") != "1":
    _run("train_l456_v98.py", dict(ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SEEDS="0,1,2",
         V98_DA="1", L5_SF=str(ART / "seqfeat_combo_cf.pkl"), L5_SF_BWD=str(AUGSF),
         L5_OSUF="_da3", ROGII_DATA=DATA), "L5 +DA(K71, 3seed)")

rb, nb = rmse("l5_base3_oof.npz"); rd, nd = rmse("l5_da3_oof.npz")
print("\n" + "=" * 60)
print(f"[結果] L5 baseline (DA無, 3seed) OOF overall RMSE = {rb}  (n={nb})")
print(f"[結果] L5 +逆方向DA (3seed)      OOF overall RMSE = {rd}  (n={nd})")
if rb and rd:
    print(f"[結果] Δ(+DA − baseline) = {rd - rb:+.4f}   ({'改善' if rd < rb else '悪化'})")
print("=" * 60)
