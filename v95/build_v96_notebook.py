# -*- coding: utf-8 -*-
"""v96 Kaggle 提出ノートブック(submit_v96.ipynb)を生成するアセンブラ。

方針(ユーザー指示: Kaggle実行 / 全コード直書き / 持ち出しは artifact のみ):
  - v95 実パイプライン(gen_grfree 錨→pf_banks/imputers/features/create→build_seqfeat)と
    GBM/TCN/L5cf 推論(submit_v95/submit_v96)を、全て %%writefile セルでノートブックに直書き。
  - 各スクリプトは env 駆動パス化済(ローカル値 fallback)。ノートブックが Kaggle パスを env で与える。
  - モデル(read-only dataset)は起動時に書込可能な作業ディレクトリへコピー(生成物 test.parquet/錨/seqfeat の
    書込先を確保。ART95/ART95B の read+write 衝突を回避)。
  - 出力: /kaggle/working/submission.csv

Kaggle Dataset の想定レイアウト(1データセット, 2サブフォルダ):
  <dataset>/artifacts_v95/   … models_l1_fwd/ nn/ pf_banks_config.json features_l1.json submit_pack_nn.json
  <dataset>/artifacts_v95b/  … l5_cf_ckpt/ submit_v96_config.json
競技データは /kaggle/input/rogii-wellbore-geology-prediction/ に add。

実行(ローカル): .venv/Scripts/python.exe v95/build_v96_notebook.py  -> v95/submit_v96.ipynb
"""
import json
from pathlib import Path

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V95 = PROJ / "v95"
V92 = PROJ / "v92"
OUT_IPYNB = V95 / "submit_v96.ipynb"

# writefile する全スクリプト(順不同で可。import は実行時に解決)
SCRIPTS = [
    ("gen_grfree_anchor.py", V92 / "gen_grfree_anchor.py"),
    ("gen_grfree_test_anchor_v95.py", V95 / "gen_grfree_test_anchor_v95.py"),
    ("pf_banks_v95.py", V95 / "pf_banks_v95.py"),
    ("imputers_v95.py", V95 / "imputers_v95.py"),
    ("features_v95.py", V95 / "features_v95.py"),
    ("create_v95.py", V95 / "create_v95.py"),
    ("build_seqfeat_v95b.py", V95 / "build_seqfeat_v95b.py"),
    ("submit_v95.py", V95 / "submit_v95.py"),
    ("submit_v96.py", V95 / "submit_v96.py"),
]

# ---- セル生成ヘルパ ----
def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


CELL_INTRO = """# ROGII v96 提出ノートブック (CANDFEAT 3脚 blend, CV 5.9416)

**構成**: GBM + TCN + **L5cf(候補別信頼性 loglik/vs_tw を各候補トークンに付与)** の固定重み 3脚 blend。

- 全コードはこのノートブック内(`%%writefile`)。持ち出しは **artifact フォルダ(モデル)のみ**。
- 錨(GRフリー)は実行時に train 上で場+GRU を再学習して隠しtest を予測(ckpt不要)。
- 出力: `/kaggle/working/submission.csv`

**必要な Kaggle Dataset(1個, 2サブフォルダ。★下記ファイルのみ = 数GB。train.parquet 等の大きい生成物は不要)**:
```
<your-dataset>/artifacts_v95/
    pf_banks_config.json  features_l1.json  submit_pack_nn.json
    models_l1_fwd/{lightgbm-1..3,catboost-1..3,xgboost-1..3}/fold{0..4}.pkl   (GBM 45)
    nn/combo_mdn2_s{0,1,2}_fold{0..4}.pt                                       (TCN 15)
<your-dataset>/artifacts_v95b/
    l5_cf_ckpt/l5_s{0,1,2}_f{0..4}_model.pt                                    (L5cf 15。valpred.npz は不要)
    submit_v96_config.json
```
競技データ `rogii-wellbore-geology-prediction` も Add data すること。
錨(grfree_anchor_test.pkl)/test.parquet/seqfeat は **アップロード不要**(実行時に隠しtest向け再生成)。
"""

CELL_ENV = r'''# ===== Kaggle 環境検出 + パス設定 + モデルを書込可能な作業ディレクトリへコピー =====
import os, sys, glob, shutil
from pathlib import Path

WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()

# --- 競技データ(train/test/sample_submission) ---
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction",
              "/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(c):
            return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)):
        return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"

# --- モデル dataset(artifacts_v95 / artifacts_v95b を含む) ---
def detect_ds_root():
    # pf_banks_config.json を持つ artifacts_v95 の親を探す
    for g in sorted(glob.glob("/kaggle/input/**/artifacts_v95/pf_banks_config.json", recursive=True)):
        return os.path.dirname(os.path.dirname(g))
    for g in sorted(glob.glob("/kaggle/input/**/pf_banks_config.json", recursive=True)):
        # フラット構成(artifacts_v95 サブフォルダ無し)にも対応
        return os.path.dirname(g)
    return None

DATA = detect_data()
DS = detect_ds_root()
print("DATA =", DATA)
print("DS   =", DS)

# ★早期チェック: 競技データ / モデル dataset が無ければ明確に止める(深いエラー回避)
_ntest = len(glob.glob(os.path.join(DATA, "test", "*__horizontal_well.csv")))
_ntrain = len(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
assert os.path.isdir(os.path.join(DATA, "train")) and os.path.isdir(os.path.join(DATA, "test")), (
    f"競技データが見つからない(DATA={DATA})。右パネル Add data で "
    "'ROGII - Wellbore Geology Prediction' を追加してください。")
print(f"  competition data OK: train {_ntrain}井 / test {_ntest}井")
assert DS is not None, ("モデル dataset(pf_banks_config.json を含む)が見つからない。"
                        "Add data で自分の kaggle_dataset_v96 を追加してください。")

# モデルを書込可能な作業ディレクトリへコピー(read-only /kaggle/input への生成物書込を回避)
ART95 = WORK / "art95"
ART95B = WORK / "art95b"
if DS is not None and str(WORK).startswith("/kaggle"):
    src95 = os.path.join(DS, "artifacts_v95") if os.path.isdir(os.path.join(DS, "artifacts_v95")) else DS
    src95b = os.path.join(DS, "artifacts_v95b") if os.path.isdir(os.path.join(DS, "artifacts_v95b")) else DS
    if not ART95.exists():
        shutil.copytree(src95, ART95)
    if not ART95B.exists():
        shutil.copytree(src95b, ART95B)
else:
    # ローカル検証時は既存の artifacts をそのまま指す
    ART95 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95")
    ART95B = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
print("ART95  =", ART95, "| exists:", ART95.exists())
print("ART95B =", ART95B, "| exists:", ART95B.exists())

# --- スクリプト置き場 = このノートの作業ディレクトリ(次セル以降の %%writefile がここに落ちる) ---
SCRIPTS_DIR = WORK
os.chdir(SCRIPTS_DIR)

# --- env(全 v95 スクリプトが参照) ---
os.environ["ROGII_DATA"] = str(DATA)
os.environ["ROGII_ART95"] = str(ART95)
os.environ["ROGII_ART95B"] = str(ART95B)
os.environ["ROGII_SCRIPTS"] = str(SCRIPTS_DIR)
os.environ["ROGII_PY"] = sys.executable
os.environ["GRFREE_GEN"] = str(SCRIPTS_DIR / "gen_grfree_anchor.py")
os.environ["ROGII_PROJ"] = str(WORK)   # fallback 用(実パスは上の個別 env が優先)
# ★隠しtest では錨を必ず再生成(dataset に含まれる可視3井用 grfree_anchor_test.pkl を使わない)
os.environ["FORCE_ANCHOR"] = "1"
# 念のため: コピー済 dataset に残る生成物(可視test の parquet/錨/seqfeat)を削除し再生成を保証
for stale in [ART95 / "grfree_anchor_test.pkl", ART95 / "grfree_anchor_test_raw.pkl",
              ART95 / "data" / "test.parquet", ART95B / "seqfeat_test_cf.pkl"]:
    try:
        if stale.exists() and str(WORK).startswith("/kaggle"):
            stale.unlink()
    except Exception:
        pass
sys.path.insert(0, str(SCRIPTS_DIR))
print("SCRIPTS_DIR =", SCRIPTS_DIR, "| PY =", sys.executable, "| FORCE_ANCHOR=1")
'''

CELL_RUN = r'''# ===== 実行: 錨 -> test.parquet -> GBM -> TCN -> L5cf -> 3脚 blend =====
import os, subprocess, sys, shutil
from pathlib import Path

# submit_v96 を subprocess で(env-before-import を厳守。pf_banks は import 時に錨を読むため)
r = subprocess.run([sys.executable, "submit_v96.py"], env=dict(os.environ))
if r.returncode != 0:
    raise SystemExit(f"submit_v96 が異常終了 (code={r.returncode})")

# submission を /kaggle/working/submission.csv へ
src = Path(os.environ["ROGII_ART95B"]) / "submission_v96.csv"
dst = Path("/kaggle/working/submission.csv") if os.path.isdir("/kaggle/working") else Path("submission.csv")
shutil.copyfile(src, dst)
import pandas as pd
sub = pd.read_csv(dst)
print("=== submission.csv ===")
print("rows:", len(sub), "| cols:", list(sub.columns))
print(sub.head(8).to_string(index=False))
print("tvt finite%:", 100.0 * sub["tvt"].notna().mean())
'''

# ---- ノートブック組み立て ----
cells = [md(CELL_INTRO), code(CELL_ENV)]
for name, path in SCRIPTS:
    body = path.read_text(encoding="utf-8")
    cells.append(code(f"%%writefile {name}\n" + body))
cells.append(code(CELL_RUN))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT_IPYNB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved: {OUT_IPYNB}  cells={len(cells)}")
for i, c in enumerate(cells):
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    first = src.strip().split("\n")[0][:70]
    print(f"  #{i} {c['cell_type']:8s} {src.count(chr(10))+1:5d}行 | {first}")
