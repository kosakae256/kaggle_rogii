# -*- coding: utf-8 -*-
"""最終提出ノートブック(submit_final.ipynb, A/B 2モード)+ データセット(final_submit_dataset)を生成。
1ノート・SUBMIT_MODE=A/B の2回実行で submission_final_A.csv / _B.csv。全脚を1回の gen で共有。
持ち出し = final_submit_dataset(v100共有 + dec10 + L5grid/greedy/extR(anchor-only)/TCN ckpt + greedy_keep + blend_final.json)。
実行: .venv/Scripts/python.exe v103/build_submit_final_notebook.py"""
import json, shutil, os, glob, re
from pathlib import Path

V103 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v103")
PROJ = V103.parent
V102 = PROJ / "v102"
V104A = PROJ / "v104" / "artifacts_v104"
A103 = V103 / "artifacts_v103"
V100SUB = PROJ / "v100" / "submit_dataset_v100"
DEC10 = V102 / "artifacts_v102" / "dec10"
OUTDIR = V103 / "submit_kaggle_final"; OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "final_submit_dataset"

GSEEDS = [0, 1, 2]; KS = [5, 8, 10]
GRID = [[gs, K] for K in KS for gs in GSEEDS]

SCRIPTS_V102 = [f + ".py" for f in [
    "gen_grfree_anchor", "gen_grfree_test_anchor_v95", "gen_grfree_test_anchor_v97", "nn_emission_v97",
    "affine_v102", "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "build_seqfeat_v97", "build_forward_v102"]]
V104 = PROJ / "v104"
SCRIPTS_V103 = ["submit_v103.py", "submit_final.py"]   # submit_final は submit_v104 を import
SCRIPTS_V104 = ["submit_v104.py"]


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
    # dec10af(sharp/smooth config = greedy の 103候補再現に必須)
    _decaf = V104 / "artifacts_v104" / "dec10af"
    assert _decaf.exists(), f"dec10af config 無し: {_decaf}"
    shutil.copytree(_decaf, DS / "dec10af")
    # L5grid(det + grid)
    n = 0
    if (A103 / "l5_v103all_ckpt").exists():
        shutil.copytree(A103 / "l5_v103all_ckpt", DS / "l5_v103all_ckpt"); n += 1
    for gs, K in GRID:
        ck = A103 / f"l5_v103g{gs}k{K}_ckpt"
        if ck.exists():
            shutil.copytree(ck, DS / f"l5_v103g{gs}k{K}_ckpt"); n += 1
    # greedy
    g = 0
    if (V104A / "l5_v104greedyg-1_ckpt").exists():
        shutil.copytree(V104A / "l5_v104greedyg-1_ckpt", DS / "l5_v104greedyg-1_ckpt")
        shutil.copyfile(V104A / "greedy_keep.json", DS / "greedy_keep.json"); g = 1
    # extR は fit で重み0=不採用 → dataset に含めない(Kaggle 実行短縮)
    # TCN(v103)
    _src = sorted(glob.glob(str(A103 / "nn" / "combo_fcombo_mdn2_s*_fold*.pt"))); t = 0
    if _src:
        (DS / "tcn_v103_ckpt").mkdir(parents=True, exist_ok=True)
        for p in _src:
            m = re.search(r"mdn2_s(\d+)_fold(\d+)", p)
            shutil.copyfile(p, DS / "tcn_v103_ckpt" / f"mdn2_s{m.group(1)}_fold{m.group(2)}.pt")
        t = 1
    if (A103 / "submit_pack_nn_fcombo.json").exists():
        shutil.copyfile(A103 / "submit_pack_nn_fcombo.json", DS / "submit_pack_v103.json")
    # blend_final.json(OOF fit 済。無ければ後で入れる)
    bf = V104A / "blend_final.json"
    if bf.exists():
        shutil.copyfile(bf, DS / "blend_final.json")
    print(f"[dataset] {DS.name}: v100共有 + dec10 + dec10af + L5grid {n}構成 + greedy={'有' if g else '未'} + TCN={'有' if t else '未'} + blend={'有' if bf.exists() else '未'}")


if os.environ.get("FINAL_SKIP_DATASET", "0") == "1":
    print("[dataset] SKIP(ipynb のみ)")
else:
    build_dataset()


INTRO = """# ROGII 最終提出 — A/B 2モード(SUBMIT_MODE を A/B に変え2回実行)
全脚を test で再現し blend_final.json の重みで合成。**SUBMIT_MODE を A/B に変え2回実行**で2ファイル。
- **A(v104無関係)** = L5grid + TCN + anchor
- **B(最良・v104少量)** = A + greedy(v104 貪欲サブセット, decaf含む=dec10af 生成)
脚: L5grid=v103 grid(91候補) / greedy=v104 keep27(103候補) / TCN=v103 / anchor=物理錨δ。各モード後に expansion 後処理。
持ち出し = `final_submit_dataset` + 競技データ。GPU=**T4**推奨。
"""

MODE = '''# ===== 提出モード選択(A/B を2回実行して2ファイル作る)=====
SUBMIT_MODE = "B"   # ← 1回目 "B"(最良)→提出、2回目 "A"(v104無関係)に変え再実行→提出
'''

ENV = r'''# ===== 環境 + 検出 + env =====
import os, sys, glob
from pathlib import Path
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"test")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/blend_final.json",recursive=True)): return os.path.dirname(g)
    for g in sorted(glob.glob("/kaggle/input/**/dec10/pf_banks_config.json",recursive=True)): return os.path.dirname(os.path.dirname(g))
    return None
DATA=detect_data(); DSPATH=detect_ds()
print("DATA=",DATA,"| final_submit_dataset=",DSPATH,"| SUBMIT_MODE=",SUBMIT_MODE)
assert SUBMIT_MODE in ("A","B"), "SUBMIT_MODE は A か B"
assert os.path.isdir(os.path.join(DATA,"test")), "競技データ未追加"
assert DSPATH is not None, "final_submit_dataset 未追加"
assert os.path.isfile(os.path.join(DSPATH,"blend_final.json")), "blend_final.json 未添付"
os.chdir(WORK); sys.path.insert(0,str(WORK))
import json as _json
# L5grid = 存在する grid ckpt(+ det)
_cfg = [[-1,5]] + [c for c in __GRID__ if os.path.isdir(os.path.join(DSPATH, f"l5_v103g{c[0]}k{c[1]}_ckpt"))]
os.environ.update(dict(ROGII_DATA=str(DATA), ROGII_PROJ=str(WORK), ROGII_SCRIPTS=str(WORK), ROGII_PY=sys.executable,
    V100SUB=str(DSPATH), V103SUB=str(DSPATH), V104SUB=str(DSPATH), FULL_VRAM_GB="8.0", PYTHONUNBUFFERED="1",
    V103_L5_CONFIGS=_json.dumps(_cfg), NN_SEEDS="5", N_SPLITS="5", SUBMIT_MODE=SUBMIT_MODE, V103_TCN="1"))
assert os.path.isdir(os.path.join(DSPATH,"dec10af")), "dec10af 未添付(greedy の 103候補再現に必須)"
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
print(f"[env] L5grid構成={len(_cfg)} greedy={'有' if os.path.isdir(os.path.join(DSPATH,'l5_v104greedyg-1_ckpt')) else '無'} dec10af={'有' if os.path.isdir(os.path.join(DSPATH,'dec10af')) else '無'}")
print("[GPU]",_ng,[_t.cuda.get_device_name(i) for i in range(_ng)])
'''.replace("__GRID__", json.dumps(GRID))

RUN = r'''# ===== 実行(SUBMIT_MODE のモードで submission.csv 生成)=====
import subprocess, sys, shutil, os
r=subprocess.run([sys.executable,"submit_final.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"submit_final 異常終了 ({r.returncode})")
import pandas as pd
_f=f"submission_final_{SUBMIT_MODE}.csv"; shutil.copyfile(_f,"submission.csv")
s=pd.read_csv(_f); print(_f, len(s), "行  finite%=", round(100*s['tvt'].notna().mean(),2), " → submission.csv")
print(s.head())
'''

cells = [md(INTRO), code(MODE), code(ENV)]
for name in SCRIPTS_V102:
    cells.append(code(f"%%writefile {name}\n" + (V102 / name).read_text(encoding="utf-8")))
for name in SCRIPTS_V104:
    cells.append(code(f"%%writefile {name}\n" + (V104 / name).read_text(encoding="utf-8")))
for name in SCRIPTS_V103:
    cells.append(code(f"%%writefile {name}\n" + (V103 / name).read_text(encoding="utf-8")))
cells.append(code(RUN))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
out = OUTDIR / "submit_final.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[notebook] {out.name} cells={len(cells)}(scripts={len(SCRIPTS_V102)+len(SCRIPTS_V103)+len(SCRIPTS_V104)})")
print(f"\n完了: {OUTDIR}(submit_final.ipynb + final_submit_dataset)")
