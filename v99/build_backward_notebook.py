# -*- coding: utf-8 -*-
"""逆方向PF生成の Kaggle ノートブック + 軽量dataset を3レグ分生成。
各レグを別環境(ローカルv96 / KaggleA v97 / KaggleB v100)で並列実行 → bpf_{leg}.parquet を出力。
持ち出し dataset(~82MB/レグ)= config + grfree_anchor_train.pkl + nn50/ + bwd_endpoints.parquet。
emission sim は Kaggle で nn_emission から再生成(nn50使用)。PFは PF_NGPU=device_count()で2GPU分散。
実行: .venv/Scripts/python.exe v99/build_backward_notebook.py"""
import json, shutil
from pathlib import Path

V99 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v99")
PROJ = V99.parent
A99 = PROJ / "v99" / "artifacts_v99"
A100 = PROJ / "v100" / "artifacts_v100"
OUTDIR = PROJ / "v100" / "backward_kaggle"
OUTDIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [f + ".py" for f in [
    "pf_banks_v95", "imputers_v95", "features_v95",
    "pf_banks_v97", "imputers_v97", "features_v97",
    "nn_emission_v97", "build_backward_v100"]]

# レグ別設定: (LEG, fwd接頭, config元dir, 錨元dir, 出力名)
LEGS = {
    "v96":  dict(prefix="",       cfg=A99 / "v95", anchor=A99 / "v95" / "grfree_anchor_train.pkl", banks=6),
    "v97":  dict(prefix="v97_",   cfg=A99 / "v97", anchor=A99 / "v97" / "grfree_anchor_train.pkl", banks=4),
    "v100": dict(prefix="v100_",  cfg=A100 / "pf5", anchor=A99 / "v95" / "grfree_anchor_train.pkl", banks=5),
}


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---- 各レグの軽量 dataset を assemble ----
NN50_SRC = A99 / "v97" / "nn50"   # emission encoder(全レグ共通)
for leg, C in LEGS.items():
    ds = OUTDIR / f"bwd_dataset_{leg}"
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "nn50").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(C["cfg"] / "pf_banks_config.json", ds / "pf_banks_config.json")
    shutil.copyfile(C["anchor"], ds / "grfree_anchor_train.pkl")
    for p in NN50_SRC.glob("*"):
        shutil.copyfile(p, ds / "nn50" / p.name)
    shutil.copyfile(A100 / "bwd_endpoints.parquet", ds / "bwd_endpoints.parquet")
    print(f"[dataset] {ds.name} assembled ({leg}: banks={C['banks']})")


INTRO_T = """# ROGII 逆方向PF生成 — @@LEG@@レグ(smooth16, 5表現, emission込み, 2GPU)
順方向PFの終端(bwd_endpoints.parquet の75far-end)を始点に、eval区間を逆走して bpf_* 特徴を生成。
持ち出し = **bwd_dataset_@@LEG@@**(config+錨+nn50+endpoints, ~82MB)+ 競技データ。
emission sim は nn_emission で再生成(nn50)。PF は PF_NGPU=device_count() で 2×T4 分散。
出力 `/kaggle/working/bpf_@@LEG@@.parquet`(id で train_combo に結合)。
"""

ENV_T = r'''# ===== 環境 + dataset を書込可能領域へ + env =====
import os, sys, glob, shutil
from pathlib import Path
LEG = "@@LEG@@"; FWD_PREFIX = "@@PREFIX@@"
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/bwd_endpoints.parquet",recursive=True)): return os.path.dirname(g)
    return None
DATA = detect_data(); DS = detect_ds()
print("DATA=",DATA,"| bwd DS=",DS)
assert os.path.isdir(os.path.join(DATA,"train")), "競技データ未追加"
assert DS is not None, f"bwd_dataset_{LEG}(bwd_endpoints.parquet)未追加"
ART = WORK/f"bwd_art_{LEG}"
if str(WORK).startswith("/kaggle"):
    if not ART.exists(): shutil.copytree(DS, ART)   # 書込可能へ(sim出力先)
else:
    ART = Path(DS)
os.chdir(WORK); sys.path.insert(0, str(WORK))
os.environ["ROGII_DATA"]=str(DATA); os.environ["ROGII_PROJ"]=str(WORK)
os.environ["ROGII_ART95"]=str(ART); os.environ["ROGII_ART97"]=str(ART)   # config/錨/nn50 は同一 ART
os.environ["LEG"]=LEG; os.environ["BWD_FWD_PREFIX"]=FWD_PREFIX
os.environ["BWD_ENDPOINTS"]=str(ART/"bwd_endpoints.parquet")
os.environ["SIM_PKL"]=str(ART/"sim_grfree_v97.pkl")     # nn_emission が生成
os.environ["OUT_BWD"]=str(WORK/f"bpf_{LEG}.parquet")
os.environ["BWD_SMOOTH_LAG"]="16"; os.environ["BWD_W_NN"]="0.02"   # 逆方向=固定ラグ16 / 全レグemission込み
os.environ["PF_WELL_CHUNK"]="120"; os.environ["PYTHONUNBUFFERED"]="1"
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
print(f"[GPU] device_count={_ng} -> " + (" / ".join(_t.cuda.get_device_name(i) for i in range(_ng)) if _ng else "CPU"))
if _ng<2: print("*** WARNING: GPU<2 -> PF単一GPU。Kaggle Settings->Accelerator を 'GPU T4 x2' に ***")
else: print(f"[GPU] マルチGPU有効: PF_NGPU={os.environ['PF_NGPU']}")
print("ART=",ART,"| LEG=",LEG)
'''

RUN_T = r'''# ===== emission sim 生成(nn50) → 逆方向PF生成 =====
import os, subprocess, sys
# (1) emission sim(train, OOF fold encoder)
os.environ["NN_SPLIT"]="train"
r=subprocess.run([sys.executable,"nn_emission_v97.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"nn_emission 異常終了 ({r.returncode})")
# (2) 逆方向PF生成
r=subprocess.run([sys.executable,"build_backward_v100.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"build_backward_v100 異常終了 ({r.returncode})")
import pandas as pd
d=pd.read_parquet(os.environ["OUT_BWD"])
print("bpf出力:", d.shape, "| bpf列:", sum(1 for c in d.columns if c.startswith("bpf_")))
print(d[["id","well"]].head(3).to_string(index=False))
'''

for leg, C in LEGS.items():
    cells = [md(INTRO_T.replace("@@LEG@@", leg)), code(ENV_T.replace("@@LEG@@", leg).replace("@@PREFIX@@", C["prefix"]))]
    for name in SCRIPTS:
        cells.append(code(f"%%writefile {name}\n" + (V99 / name).read_text(encoding="utf-8")))
    cells.append(code(RUN_T))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    out = OUTDIR / f"backward_{leg}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[notebook] {out.name} cells={len(cells)}")

print(f"\n完了: {OUTDIR} に 3レグ分の notebook + dataset")
