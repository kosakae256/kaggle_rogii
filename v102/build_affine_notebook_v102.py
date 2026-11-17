# -*- coding: utf-8 -*-
"""v102 affine の Kaggle ノート(per-leg 3本)+ dataset を生成。
各レグ(v96/v97/v100)を別セッションで実行 → {LEG}_affine.parquet(全バンク×5 affine表現)。
emission+ps_combo+full平滑 は v102 と同一(config駆動)+ AFFINE=1(照合GRをprefix較正)。
★復元: PFキャッシュ(pfcache_{LEG})→ 出力データセット化 → 次回input追加で再開。
持ち出し dataset(affine_dataset_v102)= 3レグconfig+錨(OOF)+nn50。実行: .venv/Scripts/python.exe v102/build_affine_notebook_v102.py"""
import json, shutil
from pathlib import Path

V102 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v102")
A = V102 / "artifacts_v102"
OUTDIR = V102 / "affine_kaggle"
OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "affine_dataset_v102"

SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "nn_emission_v97", "affine_v102",
    "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "create_affine_leg"]]
LEGS = ["v96", "v97", "v100"]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---- dataset(config3 + 錨 + nn50) ----
if DS.exists():
    shutil.rmtree(DS)
for leg in ["v95", "v97", "pf5"]:
    (DS / leg).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(A / leg / "pf_banks_config.json", DS / leg / "pf_banks_config.json")
    shutil.copyfile(A / leg / "grfree_anchor_train.pkl", DS / leg / "grfree_anchor_train.pkl")
    shutil.copytree(A / "v97" / "nn50", DS / leg / "nn50")   # 各レグに nn50(emission用)
print(f"[dataset] {DS.name} 組立完了(config3 + 錨 + nn50)")


INTRO_T = """# ROGII v102 affine 生成 — @@LEG@@レグ(順方向, 全バンク×5 affine表現, emission+ps_combo+full平滑)
照合GRを prefix較正(gr=(gr−b)/a)した affine版。出力 `/kaggle/working/@@LEG@@_affine.parquet`(id/well で train_combo 結合, merge時 `aff_` 接頭)。
★復元: 1回で終わらなければ Save Version→出力を次回入力に追加→再実行で pfcache から続き。GPU=**T4 x2**(P100不可)。
持ち出し = **affine_dataset_v102**(config+錨+nn50)+ 競技データ。長時間ジョブ(v102 create同規模)。
"""

ENV_T = r'''# ===== 環境 + 前回pfcache復元 + env =====
import os, sys, glob, shutil
from pathlib import Path
LEG="@@LEG@@"
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/pf5/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); DSROOT=detect_ds()
print("DATA=",DATA,"| affine_dataset=",DSROOT)
assert os.path.isdir(os.path.join(DATA,"train")), "競技データ未追加"
assert DSROOT is not None, "affine_dataset_v102(pf5/pf_banks_config.json)未追加"
for prev in sorted(glob.glob(f"/kaggle/input/**/pfcache_{LEG}", recursive=True)):   # ★復元
    dst=WORK/f"pfcache_{LEG}"
    if not dst.exists() and os.path.isdir(prev):
        try: shutil.copytree(prev,dst); print("[resume] pfcache 復元:",prev)
        except Exception: pass
for prevpq in sorted(glob.glob(f"/kaggle/input/**/{LEG}_affine.parquet", recursive=True)):
    if not (WORK/f"{LEG}_affine.parquet").exists():
        try: shutil.copyfile(prevpq, WORK/f"{LEG}_affine.parquet"); print("[resume] 完成parquet復元")
        except Exception: pass
for preva in sorted(glob.glob(f"/kaggle/input/**/{LEG}_art", recursive=True)):       # sim含むART復元
    if not (WORK/f"{LEG}_art").exists():
        try: shutil.copytree(preva, WORK/f"{LEG}_art"); print("[resume] "+LEG+"_art(sim含) 復元")
        except Exception: pass
os.chdir(WORK); sys.path.insert(0,str(WORK))
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    ROGII_DS=str(DSROOT), OUT_DIR=str(WORK), LEG=LEG, FULL_VRAM_GB="10.0", PF_WELL_CHUNK="20", PYTHONUNBUFFERED="1"))
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng)); _names=[_t.cuda.get_device_name(i) for i in range(_ng)]
print("[GPU]",_ng,"x",_names,"PF_NGPU=",os.environ["PF_NGPU"])
if any("P100" in n for n in _names): raise SystemExit("★P100不可(sm_60)。Accelerator を T4 x2 に変更して再実行")
print("LEG=",LEG)
'''

RUN_T = r'''# ===== affine 生成(1レグ, 復元可能) =====
import os, subprocess, sys
r=subprocess.run([sys.executable,"create_affine_leg.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"create_affine_leg 異常終了 ({r.returncode})")
import pyarrow.parquet as pq
m=pq.ParquetFile("@@LEG@@_affine.parquet"); print("完成:@@LEG@@_affine.parquet", m.metadata.num_rows,"行", len(m.schema.names),"列")
print(">> Save Version→出力を次回入力に追加で pfcache から続き。全レグ揃ったらローカルで結合。")
'''

for leg in LEGS:
    cells = [md(INTRO_T.replace("@@LEG@@", leg)), code(ENV_T.replace("@@LEG@@", leg))]
    for name in SCRIPTS:
        cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
    cells.append(code(RUN_T.replace("@@LEG@@", leg)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    (OUTDIR / f"affine_{leg}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[notebook] affine_{leg}.ipynb cells={len(cells)}")

print(f"\n完了: {OUTDIR}(affine_v96/v97/v100.ipynb + affine_dataset_v102)")
