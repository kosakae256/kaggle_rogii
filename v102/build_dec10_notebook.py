# -*- coding: utf-8 -*-
"""dec10(MD-10間引きPF, 順方向)の Kaggle 生成ノート + dataset を作成(v100 DA 構造を流用)。
best config = GRフリー錨mult12 + Student-t(nu5) + emission + full平滑, DEC=10, 1バンク×5表現。
出力 /kaggle/working/dec10_forward.parquet(pf_ancc/self/nbr/nbr_gr5/self_graft_delta/std_1 等)。
★復元可能: create の PFキャッシュ(pfcache/train_{rep}_dec10.pkl)を working に置く→出力データセット化→次回input追加で再開。
持ち出し dataset(dec10_dataset)= dec10 config + GRフリー錨(OOF) + nn50(emission生成用)。
実行: .venv/Scripts/python.exe v102/build_dec10_notebook.py"""
import json, shutil
from pathlib import Path

V102 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v102")
PROJ = V102.parent
A = V102 / "artifacts_v102"
OUTDIR = V102 / "dec10_kaggle"
OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "dec10_dataset"

SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "nn_emission_v97", "affine_v102",
    "pf_banks_v95", "imputers_v95", "features_v95", "create_dec10"]]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---- dataset 組立(dec10 config + 錨 + nn50) ----
if DS.exists():
    shutil.rmtree(DS)
(DS / "dec10").mkdir(parents=True, exist_ok=True)
shutil.copyfile(A / "dec10" / "pf_banks_config.json", DS / "dec10" / "pf_banks_config.json")
shutil.copyfile(A / "dec10" / "grfree_anchor_train.pkl", DS / "dec10" / "grfree_anchor_train.pkl")
shutil.copytree(A / "v97" / "nn50", DS / "dec10" / "nn50")     # emission encoder
print(f"[dataset] {DS.name} 組立完了(dec10 config + 錨 + nn50)")


INTRO = """# ROGII dec10 生成 — MD-10間引きPF(順方向, best config, full平滑)
GRフリー錨mult12 + Student-t(nu5) + emission + full平滑, DEC=10, 1バンク×5表現(tw/self/nbr/nbr_gr5/self_graft)。
出力 `/kaggle/working/dec10_forward.parquet`(id/well で v102 train_combo に結合=候補追加、merge時 `dec_` 接頭)。
★復元: 1回で終わらなければ Save Version→出力を次回の入力に追加→再実行で pfcache から続き。GPU=**T4 x2**(★P100不可=sm_60非対応)。
持ち出し = **dec10_dataset**(config+錨+nn50)+ 競技データ。
"""

ENV = r'''# ===== 環境 + 前回pfcache復元 + env =====
import os, sys, glob, shutil
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/dec10/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); DSROOT=detect_ds()
print("DATA=",DATA,"| dec10_dataset=",DSROOT)
assert os.path.isdir(os.path.join(DATA,"train")), "競技データ未追加"
assert DSROOT is not None, "dec10_dataset(dec10/pf_banks_config.json)未追加"
CART=WORK/"dec10_art"
if str(WORK).startswith("/kaggle"):
    if not (CART/"dec10").exists(): shutil.copytree(DSROOT+"/dec10", CART/"dec10")
else:
    CART=Path(DSROOT)
for cn in ["dec10_pfcache","dec10aff_pfcache"]:                                 # ★復元: 両pfcache
    for prev in sorted(glob.glob(f"/kaggle/input/**/{cn}", recursive=True)):
        dst=WORK/cn
        if not dst.exists() and os.path.isdir(prev):
            try: shutil.copytree(prev, dst); print("[resume] "+cn+" 復元:", prev)
            except Exception: pass
for pn in ["dec10_forward.parquet","dec10_affine_forward.parquet","sim_grfree_v97.pkl"]:  # ★復元: 完成物+sim
    for prevpq in sorted(glob.glob(f"/kaggle/input/**/{pn}", recursive=True)):
        if not (WORK/pn).exists():
            try: shutil.copyfile(prevpq, WORK/pn); print("[resume] "+pn+" 復元")
            except Exception: pass
os.chdir(WORK); sys.path.insert(0,str(WORK))
ART=str((CART/"dec10")); SIM=str(WORK/"sim_grfree_v97.pkl")
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_ART95=ART, ROGII_ART97=ART,
    SIM_PKL=SIM, SPLIT="train", OUT_PARQUET=str(WORK/"dec10_forward.parquet"),
    PF_CACHE_DIR=str(WORK/"dec10_pfcache"), DEC="10", FULL_VRAM_GB="10.0", PF_WELL_CHUNK="20", PYTHONUNBUFFERED="1"))
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng)); _names=[_t.cuda.get_device_name(i) for i in range(_ng)]
print("[GPU]",_ng,"x",_names,"PF_NGPU=",os.environ["PF_NGPU"])
if any("P100" in n for n in _names): raise SystemExit("★P100不可(sm_60)。Accelerator を T4 x2 に変更して再実行")
print("ART=",ART)
'''

RUN = r'''# ===== emission sim 生成 → dec10 + dec10-affine 生成 =====
import os, subprocess, sys
from pathlib import Path
if not os.path.exists(os.environ["SIM_PKL"]):
    r=subprocess.run([sys.executable,"nn_emission_v97.py"],env=dict(os.environ,NN_SPLIT="train"))
    if r.returncode!=0: raise SystemExit(f"nn_emission 異常終了 ({r.returncode})")
WORK=Path(os.environ["ROGII_PROJ"])
import pyarrow.parquet as pq
# (1) dec10 素(AFFINE=0)
r=subprocess.run([sys.executable,"create_dec10.py"],env=dict(os.environ,AFFINE="0",PF_ONLY_OUT="1",
    OUT_PARQUET=str(WORK/"dec10_forward.parquet"),PF_CACHE_DIR=str(WORK/"dec10_pfcache")))
if r.returncode!=0: raise SystemExit(f"create_dec10(素) 異常終了 ({r.returncode})")
print("dec10素完成:", pq.ParquetFile("dec10_forward.parquet").metadata.num_rows,"行")
# (2) dec10-affine(AFFINE=1)
r=subprocess.run([sys.executable,"create_dec10.py"],env=dict(os.environ,AFFINE="1",PF_ONLY_OUT="1",
    OUT_PARQUET=str(WORK/"dec10_affine_forward.parquet"),PF_CACHE_DIR=str(WORK/"dec10aff_pfcache")))
if r.returncode!=0: raise SystemExit(f"create_dec10(affine) 異常終了 ({r.returncode})")
print("dec10-affine完成:", pq.ParquetFile("dec10_affine_forward.parquet").metadata.num_rows,"行")
print(">> Save Version で出力データセット化。全井終わるまでは出力を次回入力に追加して再実行=pfcacheから続き。")
'''

cells = [md(INTRO), code(ENV)]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = OUTDIR / "dec10_gen.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[notebook] {out.name} cells={len(cells)}")
print(f"\n完了: {OUTDIR}(dec10_gen.ipynb + dec10_dataset)")
