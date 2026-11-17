# -*- coding: utf-8 -*-
"""v103 L5R+ext 提出ノートブック(submit_l5rext.ipynb)+ データセット(l5rext_submit_dataset)を生成。
単体提出: v103 の 91候補を L5R 化 + TCN OOF δ・GRフリー錨 δ を候補注入(ext)。5seed×5fold(GroupKFold det)。
持ち出し = l5rext_submit_dataset(v100共有物 + dec10 config/錨 + l5r_v104l5rdetext_ckpt + tcn_v103_ckpt + pack + postproc)+ 競技データ。
実行: .venv/Scripts/python.exe v103/build_submit_l5rext_notebook.py"""
import json, shutil, os, glob, re
from pathlib import Path

V103 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v103")
PROJ = V103.parent
V102 = PROJ / "v102"
V104A = PROJ / "v104" / "artifacts_v104"
A103 = V103 / "artifacts_v103"
V100SUB = PROJ / "v100" / "submit_dataset_v100"
DEC10 = V102 / "artifacts_v102" / "dec10"
OUTDIR = V103 / "submit_kaggle_l5rext"; OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "l5rext_submit_dataset"

# submit_v103 と同じ生成スクリプト群 + submit_v103(import元)+ submit_l5rext(本体)
SCRIPTS_V102 = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "affine_v102", "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "build_seqfeat_v97", "build_forward_v102"]]
SCRIPTS_V103 = ["submit_v103.py", "submit_l5rext.py"]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


def build_dataset():
    if DS.exists():
        shutil.rmtree(DS)
    DS.mkdir(parents=True, exist_ok=True)
    for f in ["grfree_fold_art.pkl", "v97_diff_cols.json", "v100_diff_cols.json", "features_l1_fcombo.json"]:
        if (V100SUB / f).exists():
            shutil.copyfile(V100SUB / f, DS / f)
    for leg in ["v95", "v97", "pf5"]:
        if (V100SUB / leg).exists():
            shutil.copytree(V100SUB / leg, DS / leg)
    shutil.copytree(DEC10, DS / "dec10")
    # L5R ckpt(5seed×5fold=25本)
    ckdir = V104A / "l5r_v104l5rdetext_ckpt"; _n = 0
    if ckdir.exists():
        shutil.copytree(ckdir, DS / "l5r_v104l5rdetext_ckpt")
        _n = len(glob.glob(str(DS / "l5r_v104l5rdetext_ckpt" / "*.pt")))
    # TCN(ext候補=tcn_ext の生成に必須)+ pack
    _tcn_src = sorted(glob.glob(str(A103 / "nn" / "combo_fcombo_mdn2_s*_fold*.pt"))); _tcn = 0
    if _tcn_src:
        (DS / "tcn_v103_ckpt").mkdir(parents=True, exist_ok=True)
        for p in _tcn_src:
            m = re.search(r"mdn2_s(\d+)_fold(\d+)", p)
            shutil.copyfile(p, DS / "tcn_v103_ckpt" / f"mdn2_s{m.group(1)}_fold{m.group(2)}.pt")
        _tcn = 1
    if (A103 / "submit_pack_nn_fcombo.json").exists():
        shutil.copyfile(A103 / "submit_pack_nn_fcombo.json", DS / "submit_pack_v103.json")
    # 後処理 params(OOF fit 済。無ければ恒等で提出)
    pp = V104A / "postproc_l5rext.json"
    if pp.exists():
        shutil.copyfile(pp, DS / "postproc_l5rext.json")
    print(f"[dataset] {DS.name}: v100共有物 + dec10 + L5Rckpt {_n}/25 + TCN={'有' if _tcn else '未'} + postproc={'有' if pp.exists() else '未'}")


if os.environ.get("L5REXT_SKIP_DATASET", "0") == "1":
    print("[dataset] SKIP(ipynb のみ)。学習完了後にフラグ無しで再実行して dataset を組む")
else:
    build_dataset()


INTRO = """# ROGII L5R+ext 提出 — v103の91候補を L5R 化 + TCN OOF δ・GRフリー錨 δ を候補注入
v103 の 15バンク+dec10/tw-affine(test 生成)で 91候補 → **L5R**(候補の leg/rep/family/k 埋込 + 候補間 relational attention)。
さらに **ext候補**として TCN δ(v103 TCN の test 推論)と **GRフリー物理錨 δ**(sim_grfree_test)を注入 → K=93。
**5seed×5fold(GroupKFold, 決定版)** の平均 → 後処理 expansion → submission。

持ち出し = `l5rext_submit_dataset`(v100共有物 + dec10 + l5r_v104l5rdetext_ckpt + tcn_v103_ckpt + pack + postproc)+ 競技データ。GPU=**T4**推奨。
"""

ENV = r'''# ===== 環境 + データ/データセット検出 + env =====
import os, sys, glob
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"test")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/l5r_v104l5rdetext_ckpt/l5r_s0_f0_model.pt",recursive=True)):
        return os.path.dirname(os.path.dirname(g))
    for g in sorted(glob.glob("/kaggle/input/**/dec10/pf_banks_config.json",recursive=True)):
        return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); DSPATH=detect_ds()
print("DATA=",DATA,"| l5rext_submit_dataset=",DSPATH)
assert os.path.isdir(os.path.join(DATA,"test")), "競技データ(test)未追加"
assert DSPATH is not None, "l5rext_submit_dataset 未追加"
assert os.path.isdir(os.path.join(DSPATH,"l5r_v104l5rdetext_ckpt")), "L5R ckpt 未添付"
assert os.path.isdir(os.path.join(DSPATH,"tcn_v103_ckpt")), "TCN ckpt 未添付(ext候補 tcn_ext に必須)"
os.chdir(WORK); sys.path.insert(0,str(WORK))
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    V100SUB=str(DSPATH), V103SUB=str(DSPATH), FULL_VRAM_GB="8.0", PYTHONUNBUFFERED="1",
    NN_SEEDS="5", N_SPLITS="5"))
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
print("[GPU]",_ng,[_t.cuda.get_device_name(i) for i in range(_ng)])
'''

RUN = r'''# ===== 実行: 15バンク+dec/aff生成 + 結合 + ext(TCNδ/錨δ) + L5R(2パスseqfeat) + 後処理 =====
import subprocess, sys
r=subprocess.run([sys.executable,"submit_l5rext.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_l5rext 異常終了 ({r.returncode})")
import pandas as pd, shutil
shutil.copyfile("submission_l5rext.csv","submission.csv")
sub=pd.read_csv("submission_l5rext.csv"); print("submission_l5rext.csv", len(sub), "行  finite%=", 100*sub['tvt'].notna().mean(), "  → submission.csv")
print(sub.head())
'''

cells = [md(INTRO), code(ENV)]
for name in SCRIPTS_V102:
    cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
for name in SCRIPTS_V103:
    cells.append(code(f"%%writefile {name}\n" + (V103 / name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = OUTDIR / "submit_l5rext.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[notebook] {out.name} cells={len(cells)}(scripts={len(SCRIPTS_V102)+len(SCRIPTS_V103)})")
print(f"\n完了: {OUTDIR}(submit_l5rext.ipynb + l5rext_submit_dataset)")
print("★L5R学習完了・postproc_l5rext.json 生成後に再実行すれば ckpt/postproc が dataset に入る")
