# -*- coding: utf-8 -*-
"""v102 create オーケストレータ(自己完結・v102フォルダ内で完結, バージョン横断なし)。
v100 との差分は2点のみ:
  (1) NN-emission を **全バンク**に(v96/v100 レグの create_v95 が emission ON、pf5 config も w_nn 0.02)
  (2) **ps_combo**(seed集約 = 尤度×錨カーネル, per-bank config)= 選択強化
full平滑・15バンク(v96:6 + v97:4 + v100:5)。**3レグ全て再生成**(emission+ps_comboで全レグ変わるため)。
段取り: nn_emission(sim) → v96 create → v97 create → v100 create → 3レグ結合 → train_combo.parquet。
★resume(停止/PC落ち対応): 再実行で「①sim既存→スキップ ②完成レグの parquet→丸ごとスキップ ③各レグ内は
  create の PFキャッシュ(pfcache/{split}_{rep}_{bank}.pkl)で完了済 bank×表現を再利用」。途中からで完全再開。
env: NWELLS(スモーク) / PF_WELL_CHUNK(既定20) / PS_COMBO_TAU(config上書き) / FORCE_PF(PFcache無視) / FORCE_LEG(レグ再生成) / FORCE_SIM
実行: .venv/Scripts/python.exe v102/create_v102.py
"""
import os, sys, io, json, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V102 = Path(__file__).resolve().parent
PROJ = V102.parent
A = V102 / "artifacts_v102"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
REALDATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
CHUNK = os.environ.get("PF_WELL_CHUNK", "20")
NWELLS = os.environ.get("NWELLS")

SIM = A / "v97" / "sim_grfree_v97.pkl"                   # 全バンク共通 emission sim
OUT96 = A / "data" / "v96_normal.parquet"
OUT97 = A / "data" / "v97_normal.parquet"
OUT100 = A / "data" / "v100_normal.parquet"
OUTCOMBO = A / "data" / "train_combo.parquet"
DIFF97 = json.loads((A / "v97_diff_cols.json").read_text(encoding="utf-8"))
DIFF100 = json.loads((A / "v100_diff_cols.json").read_text(encoding="utf-8"))
(A / "data").mkdir(parents=True, exist_ok=True)


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"; env["PF_WELL_CHUNK"] = CHUNK
    env["ROGII_DATA"] = str(REALDATA); env["ROGII_PROJ"] = str(PROJ)
    if NWELLS:
        env["NWELLS"] = NWELLS
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V102 / script)], env=env)   # ★v102 内のスクリプトを使用(自己完結)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def gen_emission():
    if SIM.exists() and os.environ.get("FORCE_SIM", "0") != "1":
        print(f"[v102] emission sim 既存流用: {SIM.name}", flush=True); return
    _run("nn_emission_v97.py", dict(NN_SPLIT="train", ROGII_ART97=str(A / "v97"),
         ROGII_ART95=str(A / "v97"), SIM_PKL=str(SIM)), "emission sim 生成(nn50)")


def _valid(p):
    """レグ出力 parquet が完全(読める・行>0)か。crash途中の破損は False=再生成。"""
    try:
        return p.exists() and pq.ParquetFile(p).metadata.num_rows > 0
    except Exception:
        return False


def leg(script, artenv, artdir, out, tag):
    # ★resume: 出力 parquet が既に有効なら丸ごとスキップ(レグ間再開)。レグ内は create側の PFキャッシュで再開。
    if _valid(out) and os.environ.get("FORCE_LEG", "0") != "1":
        print(f"\n[resume] {out.name} 既存(有効, {pq.ParquetFile(out).metadata.num_rows}行)-> レグ丸ごとスキップ", flush=True)
        return
    _run(script, {artenv: str(artdir), "V93_ANCHOR_PKL": str(artdir / "grfree_anchor_train.pkl"),
                  "SIM_PKL": str(SIM), "SPLIT": "train", "OUT_PARQUET": str(out),
                  "PF_CACHE_DIR": str(artdir / "pfcache")}, tag)


def combine():
    print("\n==== combo 結合(3レグ=15バンク, v96全 + v97_ + v100_) ====", flush=True)
    w96 = pq.read_table(OUT96, columns=["well"]).column("well").to_pandas().to_numpy()
    w97 = pq.read_table(OUT97, columns=["well"]).column("well").to_pandas().to_numpy()
    w100 = pq.read_table(OUT100, columns=["well"]).column("well").to_pandas().to_numpy()
    assert np.array_equal(w96, w97) and np.array_equal(w96, w100), "レグ間 well 並び不一致"
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
    print(f"[create_v102] 自己完結生成 chunk={CHUNK} NWELLS={NWELLS} (emission全バンク+ps_combo, full平滑)", flush=True)
    gen_emission()
    leg("create_v95.py", "ROGII_ART95", A / "v95", OUT96, "v96レグ create(6バンク, emission+ps_combo)")
    leg("create_v97.py", "ROGII_ART97", A / "v97", OUT97, "v97レグ create(4バンク, emission+ps_combo)")
    leg("create_v95.py", "ROGII_ART95", A / "pf5", OUT100, "v100レグ create(5バンク, emission+ps_combo)")
    if _valid(OUTCOMBO) and os.environ.get("FORCE_LEG", "0") != "1":
        print(f"\n[resume] {OUTCOMBO.name} 既存(有効)-> combine スキップ", flush=True); return
    combine()
    print(f"\n[create_v102] 完了 -> {OUTCOMBO}", flush=True)


if __name__ == "__main__":
    main()
