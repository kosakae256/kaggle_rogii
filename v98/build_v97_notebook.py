# -*- coding: utf-8 -*-
"""v97 Kaggle 提出ノートブック(submit_v97.ipynb)生成アセンブラ。
全コード %%writefile 直書き / Dataset=モデルのみ。v96版の v97 対応。
v97差分: nn_emission ON(test-sim を実行時生成=nn_emission_v97 + encoder同梱), 4バンク, 単一 artifacts_v97。
実行: .venv/Scripts/python.exe v97/build_v97_notebook.py -> v97/submit_v97.ipynb"""
import json
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V97 = PROJ / "v97"
OUT_IPYNB = V97 / "submit_v97.ipynb"
SCRIPTS = [(f, V97 / f) for f in [
    "gen_grfree_anchor.py", "gen_grfree_test_anchor_v97.py", "nn_emission_v97.py",
    "pf_banks_v97.py", "imputers_v97.py", "features_v97.py", "create_v97.py",
    "build_seqfeat_v97.py", "submit_base_v97.py", "submit_v97.py"]]

def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}

CELL_INTRO = """# ROGII v97 提出ノートブック
v96パイプライン + **v94調整PF(pf_1/2/3/4) + Student-t/tempering lever + nn_emission ON + smooth16**。
全コードは %%writefile でノート内。持ち出しは **artifact フォルダ(モデル)のみ**。錨/test-sim/parquet は実行時再生成。

**必要 Kaggle Dataset(1個, artifacts_v97 の中身)**:
```
pf_banks_config.json  features_l1.json  submit_pack_nn.json  submit_v97_config.json
models_l1_fwd/*/fold{0..4}.pkl (GBM)   nn/combo_mdn2_s{0,1,2}_fold{0..4}.pt (TCN)
l5_cf_ckpt/l5_s{0,1,2}_f{0..4}_model.pt (L5cf)
nn50/stageA_enccapaug_f{0..4}.pt (★nn_emission encoder=test-sim生成に必要)
grfree_anchor_train.pkl (錨場の学習に使用)  sim_grfree_v97.pkl は不要(train用)
```
競技データ `rogii-wellbore-geology-prediction` も Add data。出力: `/kaggle/working/submission.csv`。
"""

CELL_ENV = r'''# ===== Kaggle 環境検出 + モデルを書込可能領域へコピー + env =====
import os, sys, glob, shutil
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction", "/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c, "train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/pf_banks_config.json", recursive=True)): return os.path.dirname(g)
    return None
DATA = detect_data(); DS = detect_ds()
print("DATA =", DATA, "| DS =", DS)
assert os.path.isdir(os.path.join(DATA, "train")) and os.path.isdir(os.path.join(DATA, "test")), \
    "競技データ未追加。Add data で ROGII を追加してください。"
assert DS is not None, "モデルdataset(pf_banks_config.json)未追加。"
ART = WORK / "art97"
if str(WORK).startswith("/kaggle"):
    if not ART.exists(): shutil.copytree(DS, ART)
else:
    ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v97\artifacts_v97")
(ART / "data").mkdir(parents=True, exist_ok=True)
# 実行時再生成する生成物(dataset に紛れていれば削除)
for stale in ["grfree_anchor_test.pkl", "grfree_anchor_test_raw.pkl", "sim_grfree_test_v97.pkl", "data/test.parquet", "seqfeat_test_v97_cf.pkl"]:
    p = ART / stale
    try:
        if p.exists() and str(WORK).startswith("/kaggle"): p.unlink()
    except Exception: pass
SCRIPTS_DIR = WORK; os.chdir(SCRIPTS_DIR)
os.environ["ROGII_DATA"] = str(DATA); os.environ["ROGII_ART97"] = str(ART); os.environ["ROGII_ART97B"] = str(ART)
os.environ["ROGII_SCRIPTS"] = str(SCRIPTS_DIR); os.environ["ROGII_PY"] = sys.executable
os.environ["GRFREE_GEN"] = str(SCRIPTS_DIR / "gen_grfree_anchor.py"); os.environ["ROGII_PROJ"] = str(WORK)
os.environ["FORCE_ANCHOR"] = "1"; os.environ["FORCE_SIM"] = "1"   # 隠しtest向けに錨/sim必ず再生成
sys.path.insert(0, str(SCRIPTS_DIR))
print("ART =", ART, "| FORCE_ANCHOR/SIM=1")
'''

CELL_RUN = r'''# ===== 実行: 錨 -> test-sim -> test.parquet -> GBM -> TCN -> L5cf -> blend =====
import os, subprocess, sys, shutil
from pathlib import Path
r = subprocess.run([sys.executable, "submit_v97.py"], env=dict(os.environ))
if r.returncode != 0: raise SystemExit(f"submit_v97 異常終了 (code={r.returncode})")
src = Path(os.environ["ROGII_ART97"]) / "submission_v97.csv"
dst = Path("/kaggle/working/submission.csv") if os.path.isdir("/kaggle/working") else Path("submission.csv")
shutil.copyfile(src, dst)
import pandas as pd
sub = pd.read_csv(dst); print("rows:", len(sub), "| tvt finite%:", 100.0*sub["tvt"].notna().mean())
print(sub.head(8).to_string(index=False))
'''

cells = [md(CELL_INTRO), code(CELL_ENV)]
for name, path in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + path.read_text(encoding="utf-8")))
cells.append(code(CELL_RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT_IPYNB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved: {OUT_IPYNB}  cells={len(cells)}")
