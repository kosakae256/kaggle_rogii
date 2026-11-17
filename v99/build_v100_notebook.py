# -*- coding: utf-8 -*-
"""combo(v96+v97特徴結合, CV5.8881) の Kaggle提出ノートブック生成。
全コード%%writefile直書き / Dataset=combo_dataset(v96/v97/combo, 153MB)のみ。
実行: .venv/Scripts/python.exe v97/build_combo_notebook.py -> v97/submit_combo.ipynb"""
import json
from pathlib import Path
V97 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v99")   # ★v99 スクリプト置き場
OUT = V97.parent / "v100" / "submit_v100.ipynb"
SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97", "build_seqfeat_v97",
    "submit_v100"]]

def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}

INTRO = """# ROGII v100 提出ノートブック (combo 3レグ=15バンク + full平滑 + 5seed)
v96+v97+**v100(新5バンク pf_lib_v69_4)** の3レグでtest特徴生成(**full平滑, config駆動**)→id順結合(15バンク)→**combo TCN + combo L5**でblend(重みは blend.json)。
全コードは%%writefile。持ち出しは **submit_dataset_v100 フォルダのみ**(v100モデル+3レグconfig+錨, 2.3GB)。test錨/sim/parquetは実行時再生成。
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
    CART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v100\submit_dataset_v100")
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
os.environ["PF_WELL_CHUNK"]="20"   # (full平滑では無視。fixedlag互換のため残置)
os.environ["FULL_VRAM_GB"]="8.0"   # ★full平滑=履歴バッファVRAM予算(T4 15GBで安全)。可変chunkが長坑井を自動縮小しOOM回避
# ★PF を 2×T4 で実行(実行時間短縮)。錨FOLD_ART(combo/grfree_fold_art.pkl)はロードのみ=GRU再学習しない。
import torch as _t
_ng=_t.cuda.device_count()
os.environ["PF_NGPU"]=str(max(1,_ng))
print(f"[GPU] device_count={_ng} -> " + (" / ".join(_t.cuda.get_device_name(i) for i in range(_ng)) if _ng else "CPU"))
if _ng<2: print("*** WARNING: GPU<2 -> PFは単一GPU実行になります。Kaggle Settings->Accelerator を 'GPU T4 x2' に設定してください ***")
else: print(f"[GPU] マルチGPU有効: PF_NGPU={os.environ['PF_NGPU']} (full平滑PFが cuda:0..{_ng-1} に分散)")
sys.path.insert(0,str(WORK)); print("CART=",CART,"| PF_NGPU=",os.environ["PF_NGPU"])
'''

RUN = r'''# ===== 実行: v96生成 + v97生成 + 結合 + combo TCN/L5 + blend =====
import os, subprocess, sys, shutil
from pathlib import Path
r=subprocess.run([sys.executable,"submit_v100.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_v100 異常終了 (code={r.returncode})")
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
