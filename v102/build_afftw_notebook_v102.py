# -*- coding: utf-8 -*-
"""v102 affine_tw の Kaggle ノート(per-leg 3本)+ dataset を生成。
各レグ(v96/v97/v100)を別セッションで実行 → {LEG}_afftw.parquet(tw表現のみ affine較正=6番目候補 affine_tw)。
★lean パス(build_forward_v102, create非経由)= CSV毎メソッド再読込/imputer/build_well の無駄なし。full平滑。
★emission ON = 生成済み sim(773井, 1.38GB)を dataset に同梱して流用(Kaggle の nn_emission=0井バグを回避)。
★PF_NGPU=1(単GPU full)= マルチGPU full の cuda:1 "operation not permitted" を回避(v100 対策・全レグ適用で決定的)。
★復元: build_forward の per-(bank,rep) キャッシュ(fwdcache_{LEG}_affine)→ 出力データセット化 → 次回input追加で再開。
持ち出し dataset(afftw_dataset_v102)= 3レグ config + 錨(OOF)+ sim_grfree_v97.pkl。実行: .venv/Scripts/python.exe v102/build_afftw_notebook_v102.py"""
import json, shutil
from pathlib import Path

V102 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v102")
A = V102 / "artifacts_v102"
OUTDIR = V102 / "afftw_kaggle"
OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "afftw_dataset_v102"

SCRIPTS = [f + ".py" for f in [
    "affine_v102", "nn_emission_v97",                      # ★nn_emission 同梱(同梱sim無し/空時のフォールバック)
    "pf_banks_v95", "imputers_v95", "features_v95",
    "pf_banks_v97", "imputers_v97", "features_v97",
    "build_forward_v102", "afftw_leg_v102"]]
LEGS = ["v96", "v97", "v100"]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---- dataset(config3 + 錨3 + 同梱sim) ----
if DS.exists():
    shutil.rmtree(DS)
for leg in ["v95", "v97", "pf5"]:
    (DS / leg).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(A / leg / "pf_banks_config.json", DS / leg / "pf_banks_config.json")
    shutil.copyfile(A / leg / "grfree_anchor_train.pkl", DS / leg / "grfree_anchor_train.pkl")
shutil.copyfile(A / "v97" / "sim_grfree_v97.pkl", DS / "sim_grfree_v97.pkl")   # ★同梱 emission sim(773井)
shutil.copytree(A / "v97" / "nn50", DS / "nn50")                               # ★nn_emission フォールバック用 encoder
_simmb = (A / "v97" / "sim_grfree_v97.pkl").stat().st_size / 1e6
print(f"[dataset] {DS.name} 組立完了(config3 + 錨3 + sim {_simmb:.0f}MB + nn50)")


INTRO_T = """# ROGII v102 affine_tw 生成 — @@LEG@@レグ(順方向, tw表現のみ affine較正, full平滑, emission ON)
照合GRを prefix較正(gr=(gr−b)/a)した **tw のみ**の affine版=6番目候補 `affine_tw`。出力 `/kaggle/working/@@LEG@@_afftw.parquet`(id/well で train_combo 結合, merge時 `afftw_` 接頭)。
★lean パス(build_forward, create非経由)=CSV再読込/imputer/build_well の無駄なし。★emission ON=同梱 sim(773井)流用(Kaggleでnn_emissionを回さない)。★**PF_NGPU=1(単GPU full)**=マルチGPUの CUDA error 回避。
★復元: 1回で終わらなければ Save Version→出力を次回入力に追加→再実行で per-(bank,rep) キャッシュから続き。GPU=**T4**(単GPUでOK。x2でも1枚だけ使う)。
持ち出し = **afftw_dataset_v102**(config3+錨3+sim)+ 競技データ。
"""

ENV_T = r'''# ===== 環境 + 前回キャッシュ復元 + env(sim同梱流用, PF_NGPU=1) =====
import os, sys, glob, shutil
from pathlib import Path
LEG="@@LEG@@"
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():   # ★sim_grfree_v97.pkl と pf5/config を含む dataset ルートを再帰globで(マウント位置非依存)
    for g in sorted(glob.glob("/kaggle/input/**/sim_grfree_v97.pkl",recursive=True)): return os.path.dirname(g)
    for g in sorted(glob.glob("/kaggle/input/**/pf5/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); DSROOT=detect_ds()
print("DATA=",DATA,"| afftw_dataset=",DSROOT)
assert os.path.isdir(os.path.join(DATA,"train")), "競技データ未追加"
assert DSROOT is not None, "afftw_dataset_v102(sim_grfree_v97.pkl / pf5/pf_banks_config.json)未追加"
assert os.path.exists(os.path.join(DSROOT,"sim_grfree_v97.pkl")), "同梱sim(sim_grfree_v97.pkl)がdataset直下に見つからない"
for prev in sorted(glob.glob(f"/kaggle/input/**/fwdcache_{LEG}_affine", recursive=True)):   # ★復元: PFキャッシュ
    dst=WORK/f"fwdcache_{LEG}_affine"
    if not dst.exists() and os.path.isdir(prev):
        try: shutil.copytree(prev,dst); print("[resume] fwdcache 復元:",prev)
        except Exception: pass
for prevpq in sorted(glob.glob(f"/kaggle/input/**/{LEG}_afftw.parquet", recursive=True)):
    if not (WORK/f"{LEG}_afftw.parquet").exists():
        try: shutil.copyfile(prevpq, WORK/f"{LEG}_afftw.parquet"); print("[resume] 完成parquet復元")
        except Exception: pass
os.chdir(WORK); sys.path.insert(0,str(WORK))
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    ROGII_DS=str(DSROOT), OUT_DIR=str(WORK), LEG=LEG, FULL_VRAM_GB="5.0", PF_WELL_CHUNK="40", PYTHONUNBUFFERED="1",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
import torch as _t
_ng=_t.cuda.device_count(); _names=[_t.cuda.get_device_name(i) for i in range(_ng)]
os.environ["PF_NGPU"]="1"      # ★単GPU full=マルチGPUの cuda:1 "operation not permitted" を回避(決定的)
print("[GPU]",_ng,"x",_names,"-> PF_NGPU=1(単GPU full)")
if any("P100" in n for n in _names): raise SystemExit("★P100不可(sm_60)。Accelerator を T4 に変更して再実行")
print("LEG=",LEG)
'''

RUN_T = r'''# ===== affine_tw 生成(1レグ, lean, emission ON=同梱sim, 復元可能) =====
import os, subprocess, sys
r=subprocess.run([sys.executable,"afftw_leg_v102.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"afftw_leg 異常終了 ({r.returncode})")
import pyarrow.parquet as pq
m=pq.ParquetFile("@@LEG@@_afftw.parquet"); print("完成:@@LEG@@_afftw.parquet", m.metadata.num_rows,"行", len(m.schema.names),"列")
print(">> Save Version→出力を次回入力に追加で キャッシュから続き。全レグ揃ったらローカルで結合(afftw_接頭)。")
'''

for leg in LEGS:
    cells = [md(INTRO_T.replace("@@LEG@@", leg)), code(ENV_T.replace("@@LEG@@", leg))]
    for name in SCRIPTS:
        cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
    cells.append(code(RUN_T.replace("@@LEG@@", leg)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (OUTDIR / f"afftw_{leg}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[notebook] afftw_{leg}.ipynb cells={len(cells)}")

print(f"\n完了: {OUTDIR}(afftw_v96/v97/v100.ipynb + afftw_dataset_v102, sim同梱)")
