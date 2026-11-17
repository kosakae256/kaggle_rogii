# -*- coding: utf-8 -*-
"""v104 提出ノートブック(submit_v104.ipynb)+ データセット(v104_submit_dataset)を生成(v100/v103 参考)。
v103 の 15バンク+dec10+tw-affine に加え、anchor-free dec10(sharp/smooth)を test 生成し、PRUNE(剪定)を適用。
持ち出し = v104_submit_dataset(v100共有物 + dec10/dec10af config + l5_v104 ckpt + TCN v104 + blend_v104)+ 競技データ。
★剪定ループで最終 TAG/PRUNE/GSEEDS が決まったら下の定数を更新して再実行。
実行: .venv/Scripts/python.exe v104/build_submit_v104_notebook.py"""
import json, shutil, os
from pathlib import Path

V104 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v104")
PROJ = V104.parent
V102 = PROJ / "v102"
V103 = PROJ / "v103"
A104 = V104 / "artifacts_v104"
V100SUB = PROJ / "v100" / "submit_dataset_v100"
DEC10 = V102 / "artifacts_v102" / "dec10"
DEC10AF = V104 / "artifacts_v104" / "dec10af"          # anchor-free config(build_dec10af_config.py 生成)
OUTDIR = V104 / "submit_kaggle"; OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "v104_submit_dataset"

# ★v104 最終構成(剪定ループ確定後に更新)
V104_TAG = os.environ.get("V104_TAG", "base")           # 例 "noSelfGraft"
V104_PRUNE = os.environ.get("V104_PRUNE", "")            # 例 "self_graft"
V104_GSEEDS = [int(x) for x in os.environ.get("V104_GSEEDS", "-1").split(",") if x != ""]   # det GroupKFold のみ
L5_CKPTS = [f"l5_v104{V104_TAG}g{gs}_ckpt" for gs in V104_GSEEDS]

SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "affine_v102", "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "build_seqfeat_v97", "build_forward_v102"]]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---------------- データセット組立 ----------------
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
    shutil.copytree(DEC10AF, DS / "dec10af")            # ★anchor-free config(+ import 用 anchor)
    _n = 0
    for cd in L5_CKPTS:
        if (A104 / cd).exists():
            shutil.copytree(A104 / cd, DS / cd); _n += 1
    # TCN: v104 TCN があればそれ、無ければ ★v103 TCN を流用(v103≈v100 TCN 実証済=追加特徴は TCN 無効)。
    import glob as _g, re as _re
    A103 = PROJ / "v103" / "artifacts_v103"
    _tcn_src = sorted(_g.glob(str(A104 / "nn" / "combo_fcombo_mdn2_s*_fold*.pt")))
    _packsrc = A104 / "submit_pack_nn_fcombo.json"
    _tcn_from = "v104"
    if not _tcn_src:                                         # v104 TCN 未学習 → v103 流用
        _tcn_src = sorted(_g.glob(str(A103 / "nn" / "combo_fcombo_mdn2_s*_fold*.pt")))
        _packsrc = A103 / "submit_pack_nn_fcombo.json"; _tcn_from = "v103流用"
    _tcn = 0
    if _tcn_src:
        (DS / "tcn_v104_ckpt").mkdir(parents=True, exist_ok=True)
        for p in _tcn_src:
            m = _re.search(r"mdn2_s(\d+)_fold(\d+)", p)
            shutil.copyfile(p, DS / "tcn_v104_ckpt" / f"mdn2_s{m.group(1)}_fold{m.group(2)}.pt")
        _tcn = 1
    if _packsrc.exists():
        shutil.copyfile(_packsrc, DS / "submit_pack_v104.json")
    if (A104 / "blend_v104.json").exists():
        shutil.copyfile(A104 / "blend_v104.json", DS / "blend_v104.json")
    print(f"[dataset] {DS.name}: v100共有物 + dec10 + dec10af + L5 {_n}/{len(L5_CKPTS)}ckpt(TAG={V104_TAG}) + TCN={_tcn_from if _tcn else '未'}")


if os.environ.get("V104_SKIP_DATASET", "0") == "1":
    print("[dataset] SKIP(ipynb のみ生成)")
else:
    build_dataset()


# ---------------- ノートブック ----------------
INTRO = f"""# ROGII v104 提出 — v103 + anchor-free dec10(sharp/smooth)+ 剪定(PRUNE)
v103(15バンク+dec10+tw-affine)に **anchor-free dec10(sharp/smooth)** を追加し、L5 で **PRUNE 剪定**(多様性を保ち冗長/ノイズ削除)。
L5({len(L5_CKPTS)}構成 GroupKFold 平均)+ TCN → blend + 後処理 expansion → `submission_v104.csv`。
**v103 とは別枠の独立提出**(アンサンブルしない)。持ち出し = `v104_submit_dataset` + 競技データ。GPU=**T4**推奨。
TAG={V104_TAG} / PRUNE='{V104_PRUNE}' / gseeds={V104_GSEEDS}
"""

ENV = r'''# ===== 環境 + データ/データセット検出 + env =====
import os, sys, glob, shutil, json as _json
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"test")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_v104ds():
    for g in sorted(glob.glob("/kaggle/input/**/dec10af/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); V104DS=detect_v104ds()
print("DATA=",DATA,"| v104_submit_dataset=",V104DS)
assert os.path.isdir(os.path.join(DATA,"test")), "競技データ(test)未追加"
assert V104DS is not None, "v104_submit_dataset(dec10af/pf_banks_config.json)未追加"
os.chdir(WORK); sys.path.insert(0,str(WORK))
_CKPTS=[c for c in __L5CKPTS__ if os.path.isdir(os.path.join(V104DS,c))]
assert _CKPTS, "v104 L5 ckpt が dataset に無い"
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    V100SUB=str(V104DS), V104SUB=str(V104DS), FULL_VRAM_GB="8.0", PYTHONUNBUFFERED="1",
    V104_L5_CKPTS=",".join(_CKPTS), V104_PRUNE="__PRUNE__", NN_SEEDS="5",
    V104_TCN=("1" if os.path.isdir(os.path.join(V104DS,"tcn_v104_ckpt")) else "0")))
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
print("[GPU]",_ng,[_t.cuda.get_device_name(i) for i in range(_ng)],"| ckpts=",_CKPTS,"| PRUNE=",os.environ["V104_PRUNE"],"| V104_TCN=",os.environ["V104_TCN"])
'''.replace("__L5CKPTS__", json.dumps(L5_CKPTS)).replace("__PRUNE__", V104_PRUNE)

RUN = r'''# ===== 実行: 15バンク + dec/aff + dec10af 生成 → 結合 → L5(PRUNE)/TCN → blend =====
import subprocess, sys
r=subprocess.run([sys.executable,"submit_v104.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_v104 異常終了 ({r.returncode})")
import pandas as pd, shutil
shutil.copyfile("submission_v104.csv","submission.csv")
sub=pd.read_csv("submission_v104.csv"); print("submission_v104.csv", len(sub), "行  finite%=", 100*sub['tvt'].notna().mean(), " → submission.csv")
print(sub.head())
'''

cells = [md(INTRO), code(ENV)]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
cells.append(code("%%writefile submit_v104.py\n" + (V104 / "submit_v104.py").read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = OUTDIR / "submit_v104.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[notebook] {out.name} cells={len(cells)}(scripts={len(SCRIPTS)}+submit_v104)")
print(f"完了: {OUTDIR}  ★剪定確定後に V104_TAG/PRUNE/GSEEDS を更新し再実行 → l5_v104*_ckpt/TCN が dataset に入る")
