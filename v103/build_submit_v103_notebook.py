# -*- coding: utf-8 -*-
"""v103 提出ノートブック(submit_v103.ipynb)+ データセット(v103_submit_dataset)を生成。
持ち出しデータセットを Kaggle に添付 → ノート実行で submission_v103.csv が出る自己完結型(v100 submit を参考)。
構成:
 - v100 の 15バンク test 特徴生成(v99/v100 と同一 config・create、affine/ps_combo ゲートOFF=v100学習と厳密一致)
 - dec10 + tw-affine を test で build_forward(emission OFF, full平滑)→ 結合
 - L5 = CV grid 全構成(gseed×K)の model 平均 / TCN = v103 TCN(学習後) → blend
データセット v103_submit_dataset = v100の共有物(錨FOLD_ART/diff_cols/configs/features_l1)+ dec10 config+錨
                                 + l5_v103g{gs}k{K}_ckpt(grid)+ tcn_v103_ckpt/submit_pack_v103(学習後)+ blend_v103.json
実行: .venv/Scripts/python.exe v103/build_submit_v103_notebook.py"""
import json, shutil, os
from pathlib import Path

V103 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v103")
PROJ = V103.parent
V102 = PROJ / "v102"
A103 = V103 / "artifacts_v103"
V100SUB = PROJ / "v100" / "submit_dataset_v100"       # v100 提出データセット(流用元)
DEC10 = V102 / "artifacts_v102" / "dec10"
OUTDIR = V103 / "submit_kaggle"; OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "v103_submit_dataset"

# CV grid 構成(run_cvgrid_v103 と一致)
GSEEDS = [0, 1, 2]; KS = [5, 8, 10]
L5_CONFIGS = [[gs, K] for K in KS for gs in GSEEDS]

SCRIPTS = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "affine_v102", "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "build_seqfeat_v97", "build_forward_v102", "submit_v103"]]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---------------- データセット組立 ----------------
def build_dataset():
    if DS.exists():
        shutil.rmtree(DS)
    DS.mkdir(parents=True, exist_ok=True)
    # (a) v100 共有物: FOLD_ART / diff_cols / features_l1 / configs(v95/v97/pf5)
    for f in ["grfree_fold_art.pkl", "v97_diff_cols.json", "v100_diff_cols.json", "features_l1_fcombo.json"]:
        if (V100SUB / f).exists():
            shutil.copyfile(V100SUB / f, DS / f)
    for leg in ["v95", "v97", "pf5"]:
        if (V100SUB / leg).exists():
            shutil.copytree(V100SUB / leg, DS / leg)          # config + 錨(train)+ nn50(v97)
    # (b) dec10 config + 錨
    shutil.copytree(DEC10, DS / "dec10")
    # (c) L5 grid ckpt(有るものだけ; grid 完了後に再実行で全部入る)= strong 提出用
    _ng5 = 0
    for gs, K in L5_CONFIGS:
        ck = A103 / f"l5_v103g{gs}k{K}_ckpt"
        if ck.exists():
            shutil.copytree(ck, DS / f"l5_v103g{gs}k{K}_ckpt"); _ng5 += 1
    # (c2) 決定版 GroupKFold 5-fold ckpt = base 提出用(従来の分割, TCN と同じ)
    _det = 1 if (A103 / "l5_v103all_ckpt").exists() else 0
    if _det:
        shutil.copytree(A103 / "l5_v103all_ckpt", DS / "l5_v103all_ckpt")
    # (d) v103 TCN: train_nn は artifacts_v103/nn/combo_fcombo_mdn2_s{s}_fold{k}.pt + submit_pack_nn_fcombo.json を出力
    #     → submit が期待する tcn_v103_ckpt/mdn2_s{s}_fold{k}.pt + submit_pack_v103.json にマップ
    import glob as _g, re as _re
    _tcn_src = sorted(_g.glob(str(A103 / "nn" / "combo_fcombo_mdn2_s*_fold*.pt")))
    _tcn = 0
    if _tcn_src:
        (DS / "tcn_v103_ckpt").mkdir(parents=True, exist_ok=True)
        for p in _tcn_src:
            m = _re.search(r"mdn2_s(\d+)_fold(\d+)", p)
            shutil.copyfile(p, DS / "tcn_v103_ckpt" / f"mdn2_s{m.group(1)}_fold{m.group(2)}.pt")
        _tcn = 1
    if (A103 / "submit_pack_nn_fcombo.json").exists():
        shutil.copyfile(A103 / "submit_pack_nn_fcombo.json", DS / "submit_pack_v103.json")
    if (A103 / "blend_v103.json").exists():
        shutil.copyfile(A103 / "blend_v103.json", DS / "blend_v103.json")
    print(f"[dataset] {DS.name}: v100共有物 + dec10 + L5grid {_ng5}/{len(L5_CONFIGS)}構成 + det5fold={'有' if _det else '未'} + TCN={'有' if _tcn else '未(学習後)'}")

# V103_SKIP_DATASET=1 で ipynb 生成のみ(grid/TCN 学習中に ckpt を触らない)。学習完了後にフラグ無しで再実行=dataset 完成。
if os.environ.get("V103_SKIP_DATASET", "0") == "1":
    print("[dataset] SKIP(ipynb のみ生成)。grid/TCN 完了後にフラグ無しで再実行して dataset を組む")
else:
    build_dataset()


# ---------------- ノートブック ----------------
INTRO = """# ROGII v103 提出 — v100(15バンク)+ 脱相関新バンク(dec10 + tw-affine, emission OFF)
v100 の 15バンク test 特徴 + dec10/tw-affine(test で build_forward, full平滑, emission OFF)を結合 → L5 + TCN → blend → submission。

**2通りの提出(`SUBMIT_MODE` セルで切替、2回実行して2ファイル)**:
- `strong` = **5-fold+8-fold+10-fold の全 grid モデル**で推論(cross-group アンサンブル)→ `submission_v103_strong.csv`
- `base`   = **従来の GroupKFold 5-fold(決定版)モデルのみ** → `submission_v103_base.csv`
- TCN は GroupKFold(det)学習のみ。両モードとも同じ TCN を blend。

**持ち出し = `v103_submit_dataset`**(v100共有物 + dec10 config + l5_v103 grid/det ckpt + v103 TCN)+ 競技データ。GPU=**T4**推奨。
"""

MODE = '''# ===== 提出モード選択(strong と base を2回実行して2ファイル作る)=====
# "strong" = 5+8+10 fold の全 grid モデル(cross-group アンサンブル, 最強)
# "base"   = 従来 GroupKFold 5-fold(決定版)のみ
SUBMIT_MODE = "strong"     # ← 1回目 "strong" で実行→提出、2回目 "base" に変えて再実行→提出
'''

ENV = r'''# ===== 環境 + データ/データセット検出 + env =====
import os, sys, glob, shutil
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"test")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_v103ds():
    for g in sorted(glob.glob("/kaggle/input/**/dec10/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); V103DS=detect_v103ds()
print("DATA=",DATA,"| v103_submit_dataset=",V103DS)
assert os.path.isdir(os.path.join(DATA,"test")), "競技データ(test)未追加"
assert V103DS is not None, "v103_submit_dataset(dec10/pf_banks_config.json)未追加"
os.chdir(WORK); sys.path.insert(0,str(WORK))
import json as _json
# strong = grid 全構成(存在する ckpt のみ)、base = 決定版 GroupKFold 5-fold(gseed=-1 センチネル)
_CFG = {"strong": [c for c in __L5CONFIGS__ if os.path.isdir(os.path.join(V103DS, f"l5_v103g{c[0]}k{c[1]}_ckpt"))],
        "base":   [[-1, 5]]}
assert SUBMIT_MODE in _CFG, f"SUBMIT_MODE は strong/base: {SUBMIT_MODE}"
_cfg = _CFG[SUBMIT_MODE]
assert _cfg, f"{SUBMIT_MODE} の ckpt が dataset に無い"
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    V100SUB=str(V103DS), V103SUB=str(V103DS), FULL_VRAM_GB="8.0", PYTHONUNBUFFERED="1",
    V103_L5_CONFIGS=_json.dumps(_cfg), V103_OUT_TAG="_"+SUBMIT_MODE, NN_SEEDS="5",
    V103_TCN=("1" if os.path.isdir(os.path.join(V103DS,"tcn_v103_ckpt")) else "0")))
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
print(f"[mode] {SUBMIT_MODE} L5={_cfg}")
print("[GPU]",_ng,[_t.cuda.get_device_name(i) for i in range(_ng)],"| V103_TCN=",os.environ["V103_TCN"])
'''.replace("__L5CONFIGS__", json.dumps(L5_CONFIGS))

RUN = r'''# ===== 実行: 15バンクtest生成 + dec/aff生成 + 結合 + L5(grid)/TCN + blend =====
import subprocess, sys
r=subprocess.run([sys.executable,"submit_v103.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_v103 異常終了 ({r.returncode})")
import pandas as pd, shutil
_tag=os.environ["V103_OUT_TAG"]; _f=f"submission_v103{_tag}.csv"
shutil.copyfile(_f, "submission.csv")   # Kaggle 提出用に submission.csv も出力
sub=pd.read_csv(_f); print(_f, len(sub), "行  finite%=", 100*sub['tvt'].notna().mean(), "  → submission.csv")
print(sub.head())
'''

cells = [md(INTRO), code(MODE), code(ENV)]
for name in SCRIPTS:
    cells.append(code(f"%%writefile {name}\n" + (V102 / name if name != "submit_v103.py" else V103 / name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = OUTDIR / "submit_v103.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[notebook] {out.name} cells={len(cells)}(scripts={len(SCRIPTS)})")
print(f"\n完了: {OUTDIR}(submit_v103.ipynb + v103_submit_dataset)")
print("★grid完了・TCN学習後に再実行すれば l5_v103*_ckpt / tcn_v103 が dataset に入る")
