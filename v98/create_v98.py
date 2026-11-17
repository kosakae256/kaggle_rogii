# -*- coding: utf-8 -*-
"""v98 create オーケストレータ: 1系統(SYSTEM=normal|bwd|fwd)の combo train.parquet を生成。
段取り: (DA錨生成) → create_v95(v96パイプライン, lag192強化) → create_v97(v97パイプライン, lag192)
        → id順hstack で combo 化(v96全列 + v97差分 v97_)→ is_backward/is_fwdaug タグ付与。
- normal: 実train。錨=既存 grfree_anchor_train(OOF)。nn-emission ON(sim_grfree_v97)。
- bwd/fwd: DA CSV(da_bwd/da_fwd)。錨=FOLD_ARTアンサンブル(train専用・sig9.0)。nn-emission OFF(sim中立)。
全系統 同一スキーマ(is_backward/is_fwdaug 両列付与)で train 時に concat 可能。DAは元wellを保持=同fold。
env: SYSTEM / NWELLS(スモーク) / PF_WELL_CHUNK(既定20)
出力: artifacts_v98/data/{train_combo, train_aug_bwd_combo, train_aug_fwd_combo}.parquet
実行: SYSTEM=normal .venv/Scripts/python.exe v98/create_v98.py
"""
import os, sys, io, json, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V98 = Path(__file__).resolve().parent
PROJ = V98.parent
ART = V98 / "artifacts_v98"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
REALDATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
SYSTEM = os.environ.get("SYSTEM", "normal")
NWELLS = os.environ.get("NWELLS")
CHUNK = os.environ.get("PF_WELL_CHUNK", "20")
FOLD_ART = ART / "grfree_fold_art.pkl"
DIFF = json.loads((ART / "v97_diff_cols.json").read_text())
(ART / "data").mkdir(parents=True, exist_ok=True)

OUTNAME = {"normal": "train_combo.parquet", "bwd": "train_aug_bwd_combo.parquet", "fwd": "train_aug_fwd_combo.parquet"}[SYSTEM]
OUTCOMBO = ART / "data" / OUTNAME
OUT96 = ART / "data" / f"v96_{SYSTEM}.parquet"
OUT97 = ART / "data" / f"v97_{SYSTEM}.parquet"


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"; env["PF_WELL_CHUNK"] = CHUNK
    if NWELLS:
        env["NWELLS"] = NWELLS
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V98 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def gen_da_anchor(droot):
    """DA井の錨を FOLD_ART(アンサンブル) で生成し sig=9.0 に統一(train専用・検証されない)。"""
    raw = droot / "grfree_anchor_train_raw.pkl"; final = droot / "grfree_anchor_train.pkl"
    _run("gen_grfree_anchor.py", dict(ROGII_DATA=str(droot), GRF_SPLIT="train",
         GRFREE_FOLD_ART=str(FOLD_ART), GRF_ANCHOR_OUT=str(raw), ROGII_ART95=str(ART / "v95")), f"{SYSTEM}錨(FOLD_ART)")
    anch = pickle.loads(raw.read_bytes())
    fixed = {w: dict(tvt=np.asarray(d["tvt"], np.float32),
                     sig=np.full(len(d["tvt"]), 9.0, np.float32)) for w, d in anch.items()}
    final.write_bytes(pickle.dumps(fixed, protocol=4))
    print(f"[{SYSTEM}錨] {len(fixed)}井 sig=9.0 -> {final.name}", flush=True)
    return final


def resolve():
    """(data_root, anc95, anc97, sim97) を系統別に返す。"""
    if SYSTEM == "normal":
        return REALDATA, ART / "v95" / "grfree_anchor_train.pkl", ART / "v97" / "grfree_anchor_train.pkl", ART / "v97" / "sim_grfree_v97.pkl"
    droot = ART / ("da_bwd" if SYSTEM == "bwd" else "da_fwd")
    anc = gen_da_anchor(droot)
    return droot, anc, anc, Path("___no_sim___")     # DAは sim無し(中立)


def combine():
    """v96全列 + v97差分(v97_) を行バッチ hstack。is_backward/is_fwdaug 付与。"""
    print("\n==== combo 結合 ====", flush=True)
    w96 = pq.read_table(OUT96, columns=["well"]).column("well").to_pandas().to_numpy()
    w97 = pq.read_table(OUT97, columns=["well"]).column("well").to_pandas().to_numpy()
    assert np.array_equal(w96, w97), "v96/v97 の well 並び不一致"
    isb = 1.0 if SYSTEM == "bwd" else 0.0
    isf = 1.0 if SYSTEM == "fwd" else 0.0
    pf96 = pq.ParquetFile(OUT96); pf97 = pq.ParquetFile(OUT97); writer = None; nrow = 0
    for b96, b97 in zip(pf96.iter_batches(batch_size=300000), pf97.iter_batches(batch_size=300000, columns=DIFF)):
        cols = list(b96.columns); names = list(b96.schema.names)
        for j, c in enumerate(DIFF):
            cols.append(b97.column(j)); names.append("v97_" + c)
        n = b96.num_rows
        cols.append(pa.array(np.full(n, isb, np.float32))); names.append("is_backward")
        cols.append(pa.array(np.full(n, isf, np.float32))); names.append("is_fwdaug")
        tbl = pa.Table.from_arrays(cols, names=names)
        if writer is None:
            writer = pq.ParquetWriter(OUTCOMBO, tbl.schema, compression="zstd")
        writer.write_table(tbl); nrow += n
    writer.close()
    sch = pq.read_schema(OUTCOMBO)
    print(f"[combo] {OUTCOMBO.name} 完成 行={nrow} 列={len(sch.names)}", flush=True)


def main():
    print(f"[create_v98] SYSTEM={SYSTEM} chunk={CHUNK} NWELLS={NWELLS} FOLD_ART={FOLD_ART.name}", flush=True)
    if SYSTEM in ("bwd", "fwd") and (V98 / "SKIP_DA").exists():   # ★DA一旦停止(CSV/aug parquetは保持=再開可)
        print(f"[create_v98] ★SKIP_DA: SYSTEM={SYSTEM} をスキップ(DA資産は保持)", flush=True); return
    droot, anc95, anc97, sim97 = resolve()
    _run("create_v95.py", dict(ROGII_DATA=str(droot), ROGII_ART95=str(ART / "v95"), SPLIT="train",
         V93_ANCHOR_PKL=str(anc95), OUT_PARQUET=str(OUT96)), f"{SYSTEM} v96create")
    _run("create_v97.py", dict(ROGII_DATA=str(droot), ROGII_ART97=str(ART / "v97"), SPLIT="train",
         V93_ANCHOR_PKL=str(anc97), SIM_PKL=str(sim97), OUT_PARQUET=str(OUT97)), f"{SYSTEM} v97create")
    combine()
    print(f"\n[create_v98] SYSTEM={SYSTEM} 完了 -> {OUTCOMBO}", flush=True)


if __name__ == "__main__":
    main()
