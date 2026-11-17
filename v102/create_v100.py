# -*- coding: utf-8 -*-
"""v100 create: v99(v96+v97, full平滑)に **新5バンク(pf_lib_v69_4)** を新レグとして追加。
- v96/v97 は再生成せず **v99の既存 parquet を流用**(ユーザー指示)。
- 新レグ(v100)= create_v95機構 + v100/artifacts_v100/pf5 の5バンクconfig(full平滑, robust lever調整済)。
  → v100_normal.parquet(774列, 無印)。その pf_/prov_/sig_/tdpf を v100_diff_cols として combine で `v100_` prefix。
- combine: v96_normal(全845, 無印) + v97_normal 差分(427, v97_) + v100_normal 差分(517, v100_) + is_backward/is_fwdaug
  → train_combo.parquet(1791列, 15バンク=6+4+5)。
env: NWELLS(スモーク) / PF_WELL_CHUNK
実行: SYSTEM=normal .venv/Scripts/python.exe v100/create_v100.py
"""
import os, sys, io, json, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V100 = Path(__file__).resolve().parent
PROJ = V100.parent
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
REALDATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
ART = V100 / "artifacts_v100"
V99_DATA = PROJ / "v99" / "artifacts_v99" / "data"      # ★v96/v97 既存parquetの所在(流用)
CHUNK = os.environ.get("PF_WELL_CHUNK", "20")
NWELLS = os.environ.get("NWELLS")

OUT96 = V99_DATA / "v96_normal.parquet"                 # 流用(再生成しない)
OUT97 = V99_DATA / "v97_normal.parquet"                 # 流用(再生成しない)
OUT100 = ART / "data" / "v100_normal.parquet"           # 新レグ(今回生成)
OUTCOMBO = ART / "data" / "train_combo.parquet"
DIFF97 = json.loads((PROJ / "v99" / "artifacts_v99" / "v97_diff_cols.json").read_text(encoding="utf-8"))
DIFF100 = json.loads((ART / "v100_diff_cols.json").read_text(encoding="utf-8"))
ANCHOR = str(PROJ / "v99" / "artifacts_v99" / "v95" / "grfree_anchor_train.pkl")   # use_anchor=Falseなので内容不使用(ロード用)
(ART / "data").mkdir(parents=True, exist_ok=True)


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"; env["PF_WELL_CHUNK"] = CHUNK
    if NWELLS:
        env["NWELLS"] = NWELLS
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(PROJ / "v99" / script)], env=env)   # v99のスクリプト(full平滑+cache版)を使用
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def gen_v100_leg():
    """新5バンクを create_v95機構 + v100(pf5)config で生成 → v100_normal.parquet。"""
    _run("create_v95.py", dict(
        ROGII_ART95=str(ART / "pf5"),
        V93_ANCHOR_PKL=ANCHOR,
        ROGII_DATA=str(REALDATA), ROGII_PROJ=str(PROJ),
        SPLIT="train", OUT_PARQUET=str(OUT100),
        PF_CACHE_DIR=str(ART / "pf5" / "pfcache")), "v100レグ create(full平滑, 5バンク)")


def combine():
    """v96全 + v97差分(v97_) + v100差分(v100_) を行バッチ hstack。"""
    print("\n==== combo 結合(3レグ=15バンク) ====", flush=True)
    w96 = pq.read_table(OUT96, columns=["well"]).column("well").to_pandas().to_numpy()
    w97 = pq.read_table(OUT97, columns=["well"]).column("well").to_pandas().to_numpy()
    w100 = pq.read_table(OUT100, columns=["well"]).column("well").to_pandas().to_numpy()
    assert np.array_equal(w96, w97), "v96/v97 の well 並び不一致"
    assert np.array_equal(w96, w100), "v96/v100 の well 並び不一致(v100レグの坑井順が違う)"
    pf96 = pq.ParquetFile(OUT96); pf97 = pq.ParquetFile(OUT97); pf100 = pq.ParquetFile(OUT100)
    writer = None; nrow = 0
    for b96, b97, b100 in zip(pf96.iter_batches(batch_size=300000),
                              pf97.iter_batches(batch_size=300000, columns=DIFF97),
                              pf100.iter_batches(batch_size=300000, columns=DIFF100)):
        cols = list(b96.columns); names = list(b96.schema.names)
        for j, c in enumerate(DIFF97):
            cols.append(b97.column(j)); names.append("v97_" + c)
        for j, c in enumerate(DIFF100):
            cols.append(b100.column(j)); names.append("v100_" + c)
        n = b96.num_rows
        cols.append(pa.array(np.zeros(n, np.float32))); names.append("is_backward")
        cols.append(pa.array(np.zeros(n, np.float32))); names.append("is_fwdaug")
        tbl = pa.Table.from_arrays(cols, names=names)
        if writer is None:
            writer = pq.ParquetWriter(OUTCOMBO, tbl.schema, compression="zstd")
        writer.write_table(tbl); nrow += n
    writer.close()
    sch = pq.read_schema(OUTCOMBO)
    print(f"[combo] {OUTCOMBO.name} 完成 行={nrow} 列={len(sch.names)}", flush=True)


def main():
    print(f"[create_v100] chunk={CHUNK} NWELLS={NWELLS}  v96/v97流用={V99_DATA}", flush=True)
    assert OUT96.exists() and OUT97.exists(), f"v96/v97 parquet が無い: {OUT96} / {OUT97}"
    gen_v100_leg()
    combine()
    print(f"\n[create_v100] 完了 -> {OUTCOMBO}", flush=True)


if __name__ == "__main__":
    main()
