# -*- coding: utf-8 -*-
"""combo(v96+v97特徴結合, CV5.8881) の Kaggle提出ノートブック生成。
全コード%%writefile直書き / Dataset=combo_dataset(v96/v97/combo, 153MB)のみ。
実行: .venv/Scripts/python.exe v97/build_combo_notebook.py -> v97/submit_combo.ipynb"""
import json
from pathlib import Path
V97 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v98")   # ★v98 スクリプト置き場
OUT = V97 / "submit_v98.ipynb"
SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97", "build_seqfeat_v97",
    "submit_v98"]]

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}

INTRO = """# ROGII v98 提出ノートブック (combo特徴 + 両PF強化 + 双方向DA + smooth192 + 5seed)
v96とv97の両パイプラインでtest特徴を生成→id順結合→**combo TCN + combo L5**でblend(0.476/0.524, GBM重み0)。
全コードは%%writefile。持ち出しは **combo_dataset フォルダのみ**(v96/ v97/ combo/, 153MB)。錨/sim/parquetは実行時再生成。
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
    for g in sorted(glob.glob("/kaggle/input/**/submit_pack_nn_fcombo.json",recursive=True)): return os.path.dirname(g)
    return None
DATA = detect_data(); DS = detect_combo()
print("DATA=",DATA,"| combo DS=",DS)
assert os.path.isdir(os.path.join(DATA,"train")) and os.path.isdir(os.path.join(DATA,"test")), "競技データ未追加"
assert DS is not None, "artifacts_v99(submit_pack_nn_fcombo.json)未追加"
CART = WORK/"combo_art"
if str(WORK).startswith("/kaggle"):
    if not CART.exists(): shutil.copytree(DS, CART)
else:
    CART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v97\combo_dataset")
for stale in ["v95/grfree_anchor_test.pkl","v95/data/test.parquet","v97/grfree_anchor_test.pkl",
              "v97/sim_grfree_test_v97.pkl","v97/data/test.parquet","data/test_combo.parquet",
              "seqfeat_test_combo.pkl"]:
    p=CART/stale
    try:
        if p.exists() and str(WORK).startswith("/kaggle"): p.unlink()
    except Exception: pass
os.chdir(WORK)
os.environ["ROGII_DATA"]=str(DATA); os.environ["ROGII_COMBO"]=str(CART); os.environ["ROGII_WORK"]=str(CART)
os.environ["ROGII_SCRIPTS"]=str(WORK); os.environ["ROGII_PY"]=sys.executable
os.environ["GRFREE_GEN"]=str(WORK/"gen_grfree_anchor.py"); os.environ["ROGII_PROJ"]=str(WORK)
os.environ["FORCE_ANCHOR"]="1"; os.environ["FORCE_SIM"]="1"
os.environ["PF_WELL_CHUNK"]="20"   # ★smooth_lag=192 は VRAM 対策で chunk=20
# ★PF を 2×T4 で実行(実行時間短縮)。錨FOLD_ART(combo/grfree_fold_art.pkl)はロードのみ=GRU再学習しない。
os.environ["PF_NGPU"]=str(max(1, __import__("torch").cuda.device_count()))
sys.path.insert(0,str(WORK)); print("CART=",CART,"| PF_NGPU=",os.environ["PF_NGPU"])
'''

RUN = r'''# ===== 実行: v96生成 + v97生成 + 結合 + combo TCN/L5 + blend =====
import os, subprocess, sys, shutil
from pathlib import Path
r=subprocess.run([sys.executable,"submit_v98.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_v98 異常終了 (code={r.returncode})")
src=Path(os.environ["ROGII_COMBO"])/"submission_combo.csv"
dst=Path("/kaggle/working/submission.csv") if os.path.isdir("/kaggle/working") else Path("submission.csv")
shutil.copyfile(src,dst)
import pandas as pd; sub=pd.read_csv(dst); print("rows:",len(sub),"| finite%:",100*sub["tvt"].notna().mean()); print(sub.head(6).to_string(index=False))
'''

cells=[md(INTRO), code(ENV)]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n"+(V97/name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
print(f"saved {OUT} cells={len(cells)}")
