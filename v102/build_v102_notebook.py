# -*- coding: utf-8 -*-
"""v102 提出ノートブック生成(v100参考・自己完結)。全コード%%writefile直書き。
持ち出し = submit_dataset_v102(v102モデル + 3レグconfig[emission全/ps_combo] + 錨 + nn50, blend.json)。
v100との差=create側 emission全バンク + ps_combo(config)。test錨/sim/parquetは実行時再生成。
実行: .venv/Scripts/python.exe v102/build_v102_notebook.py -> v102/submit_v102.ipynb"""
import json
from pathlib import Path
V = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v102")   # ★v102 スクリプト置き場(自己完結)
OUT = V / "submit_v102.ipynb"
SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97", "build_seqfeat_v97",
    "submit_v102"]]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


INTRO = """# ROGII v102 提出ノートブック (combo 3レグ=15バンク, full平滑, emission全バンク + ps_combo)
v100 との差分は2点のみ: **(1) NN-emission を全15バンクに** **(2) ps_combo(seed集約=尤度×錨カーネル, 選択強化)**。
3レグ(v96/v97/v100)で test 特徴生成(full平滑・**全レグ共有 test-sim で emission**・ps_combo は config 自動)→ id順結合 → combo TCN + combo L5 で blend(blend.json)。
持ち出しは **submit_dataset_v102** のみ(v102モデル+3レグconfig+錨+nn50)。test錨/sim/parquetは実行時再生成。
競技データ `rogii-wellbore-geology-prediction` も Add data。出力 `/kaggle/working/submission.csv`。
"""

ENV = r'''# ===== Kaggle 環境 + データを書込可能領域へコピー + env =====
import os, sys, glob, shutil
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_combo():
    for g in sorted(glob.glob("/kaggle/input/**/blend.json",recursive=True)): return os.path.dirname(g)  # v102 datasetの目印
    return None
DATA = detect_data(); DS = detect_combo()
print("DATA=",DATA,"| v102 DS=",DS)
assert os.path.isdir(os.path.join(DATA,"train")) and os.path.isdir(os.path.join(DATA,"test")), "競技データ未追加"
assert DS is not None, "submit_dataset_v102(blend.json)未追加"
CART = WORK/"v102_art"
if str(WORK).startswith("/kaggle"):
    if not CART.exists(): shutil.copytree(DS, CART)
else:
    CART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v102\submit_dataset_v102")
for stale in ["v95/grfree_anchor_test.pkl","v95/data/test.parquet","v97/grfree_anchor_test.pkl",
              "v97/sim_grfree_test_v97.pkl","v97/data/test.parquet","pf5/data/test.parquet",
              "data/test_combo.parquet","seqfeat_test_combo.pkl"]:
    p=CART/stale
    try:
        if p.exists() and str(WORK).startswith("/kaggle"): p.unlink()
    except Exception: pass
os.chdir(WORK)
os.environ["ROGII_DATA"]=str(DATA); os.environ["ROGII_COMBO"]=str(CART); os.environ["ROGII_WORK"]=str(CART)
os.environ["ROGII_SCRIPTS"]=str(WORK); os.environ["ROGII_PY"]=sys.executable
os.environ["GRFREE_GEN"]=str(WORK/"gen_grfree_anchor.py"); os.environ["ROGII_PROJ"]=str(WORK)
os.environ["FORCE_ANCHOR"]="1"; os.environ["FORCE_SIM"]="1"
os.environ["PF_WELL_CHUNK"]="20"   # (full平滑では無視)
os.environ["FULL_VRAM_GB"]="8.0"   # ★full平滑=履歴バッファVRAM予算(T4 15GBで安全)
import torch as _t
_ng=_t.cuda.device_count()
os.environ["PF_NGPU"]=str(max(1,_ng))
print(f"[GPU] device_count={_ng} -> " + (" / ".join(_t.cuda.get_device_name(i) for i in range(_ng)) if _ng else "CPU"))
if _ng<2: print("*** WARNING: GPU<2 -> PFは単一GPU実行。Kaggle Settings->Accelerator を 'GPU T4 x2' に ***")
else: print(f"[GPU] マルチGPU有効: PF_NGPU={os.environ['PF_NGPU']}")
sys.path.insert(0,str(WORK)); print("CART=",CART,"| PF_NGPU=",os.environ["PF_NGPU"])
'''

RUN = r'''# ===== 実行: 3レグtest生成(emission全/ps_combo) + 結合 + combo TCN/L5 + blend =====
import os, subprocess, sys, shutil
from pathlib import Path
r=subprocess.run([sys.executable,"submit_v102.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_v102 異常終了 (code={r.returncode})")
src=Path(os.environ["ROGII_COMBO"])/"submission_combo.csv"
dst=Path("/kaggle/working/submission.csv") if os.path.isdir("/kaggle/working") else Path("submission.csv")
shutil.copyfile(src,dst)
import pandas as pd; sub=pd.read_csv(dst); print("rows:",len(sub),"| finite%:",100*sub["tvt"].notna().mean()); print(sub.head(6).to_string(index=False))
'''

cells = [md(INTRO), code(ENV)]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + (V / name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved {OUT} cells={len(cells)}")
