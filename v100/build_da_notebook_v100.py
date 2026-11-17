# -*- coding: utf-8 -*-
"""v100 逆方向DA の Kaggle ノートブック(per-leg 3本)+ 共有dataset を生成。
各レグ(v96/v97/v100)を別セッションで実行(8h制限・P100)。出力 {LEG}_bwd.parquet を /kaggle/working へ。
★復元可能: create の PFキャッシュ(pfcache_{LEG})を working に置く → 出力データセットとして commit →
  次セッションで「前回の出力データセット」を input に追加すれば pfcache を復元し完了済 bank×表現をスキップ再開。
持ち出し dataset(da_dataset_v100, 小)= 3レグconfig+錨 + FOLD_ART(emission用nn50/sim は不要=DAは中立)。
実行: .venv/Scripts/python.exe v100/build_da_notebook_v100.py"""
import json, shutil
from pathlib import Path

V99 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v99")
PROJ = V99.parent
A99 = PROJ / "v99" / "artifacts_v99"
A100 = PROJ / "v100" / "artifacts_v100"
OUTDIR = PROJ / "v100" / "da_kaggle"
OUTDIR.mkdir(parents=True, exist_ok=True)
DS = OUTDIR / "da_dataset_v100"

SCRIPTS = [f + ".py" for f in [
    "gen_da_csvs_v98", "gen_grfree_anchor", "nn_emission_v97",   # ★nn_emission=v97レグ emission用
    "pf_banks_v95", "imputers_v95", "features_v95", "create_v95",
    "pf_banks_v97", "imputers_v97", "features_v97", "create_v97",
    "da_leg_bwd_v100"]]
LEGS = ["v96", "v97", "v100"]


def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}


# ---- 共有 dataset 組み立て(config+錨 + FOLD_ART) ----
if DS.exists():
    shutil.rmtree(DS)
(DS / "v95").mkdir(parents=True, exist_ok=True)
(DS / "v97").mkdir(parents=True, exist_ok=True)
(DS / "pf5").mkdir(parents=True, exist_ok=True)
shutil.copyfile(A99 / "v95" / "pf_banks_config.json", DS / "v95" / "pf_banks_config.json")
shutil.copyfile(A99 / "v95" / "grfree_anchor_train.pkl", DS / "v95" / "grfree_anchor_train.pkl")
shutil.copyfile(A99 / "v97" / "pf_banks_config.json", DS / "v97" / "pf_banks_config.json")
shutil.copyfile(A99 / "v97" / "grfree_anchor_train.pkl", DS / "v97" / "grfree_anchor_train.pkl")
shutil.copyfile(A100 / "pf5" / "pf_banks_config.json", DS / "pf5" / "pf_banks_config.json")
shutil.copyfile(A99 / "v95" / "grfree_anchor_train.pkl", DS / "pf5" / "grfree_anchor_train.pkl")
shutil.copyfile(A99 / "grfree_fold_art.pkl", DS / "grfree_fold_art.pkl")
# ★v97レグ emission ON(normal一致)に必要な nn50 encoder を同梱
shutil.copytree(A99 / "v97" / "nn50", DS / "v97" / "nn50")
print(f"[dataset] {DS.name} 組立完了(config3+錨+FOLD_ART+nn50[v97 emission用])")


INTRO_T = """# ROGII v100 逆方向DA 生成 — @@LEG@@レグ(P100, 8h制限, 復元可能)
eval行反転+toe側既知帯化の合成坑井を全@@LEG@@バンクで PF(full平滑, emission中立)→ `/kaggle/working/@@LEG@@_bwd.parquet`。
**★復元手順**: 1回で終わらなければ commit(Save Version)→ 出来た出力を **入力データセットとして次セッションに追加** →
このノートを再実行すると pfcache が復元され、完了済 bank×表現をスキップして続きから。全レグ揃ったらローカルで結合。
持ち出し = **da_dataset_v100**(config+錨+FOLD_ART)+ 競技データ。GPU=**T4 x2 必須**(★P100はKaggle PyTorchが非対応=sm_60未サポートで CUDAカーネル無し。T4=sm_75はOK)。
"""

ENV_T = r'''# ===== 環境 + 前回出力(pfcache)復元 + env =====
import os, sys, glob, shutil
from pathlib import Path
LEG = "@@LEG@@"
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".").resolve()
def detect_data():
    for c in ["/kaggle/input/rogii-wellbore-geology-prediction","/kaggle/input/competitions/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c,"train")): return c
    for g in sorted(glob.glob("/kaggle/input/**/sample_submission.csv",recursive=True)): return os.path.dirname(g)
    return r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"
def detect_ds():
    for g in sorted(glob.glob("/kaggle/input/**/grfree_fold_art.pkl",recursive=True)): return os.path.dirname(g)
    return None
DATA = detect_data(); DSROOT = detect_ds()
print("DATA=",DATA,"| da_dataset=",DSROOT)
assert os.path.isdir(os.path.join(DATA,"train")), "競技データ未追加"
assert DSROOT is not None, "da_dataset_v100(grfree_fold_art.pkl)未追加"
# ★復元: 前回セッションの出力データセット(pfcache_{LEG}/{LEG}_bwd.parquet 等)を working へ戻す
for prev in sorted(glob.glob(f"/kaggle/input/**/pfcache_{LEG}", recursive=True)):
    dst = WORK/f"pfcache_{LEG}"
    if not dst.exists(): shutil.copytree(prev, dst); print("[resume] pfcache 復元:", prev, "->", dst)
for prevpq in sorted(glob.glob(f"/kaggle/input/**/{LEG}_bwd.parquet", recursive=True)):
    if not (WORK/f"{LEG}_bwd.parquet").exists(): shutil.copyfile(prevpq, WORK/f"{LEG}_bwd.parquet"); print("[resume] 完成parquet 復元:", prevpq)
for prevda in sorted(glob.glob("/kaggle/input/**/da_bwd", recursive=True)):     # DA CSV/錨も復元(あれば)
    if not (WORK/"da_bwd").exists():
        try: shutil.copytree(prevda, WORK/"da_bwd"); print("[resume] da_bwd 復元")
        except Exception: pass
os.chdir(WORK); sys.path.insert(0, str(WORK))
os.environ["ROGII_DATA"]=str(DATA); os.environ["ROGII_DS"]=str(DSROOT); os.environ["ROGII_SCRIPTS"]=str(WORK)
os.environ["ROGII_PROJ"]=str(WORK); os.environ["OUT_DIR"]=str(WORK); os.environ["LEG"]=LEG
os.environ["ROGII_PY"]=sys.executable; os.environ["PYTHONUNBUFFERED"]="1"
os.environ["FULL_VRAM_GB"]=os.environ.get("FULL_VRAM_GB","10.0")   # ★T4=15GB。履歴バッファ予算=大きいほどchunk大=速いがOOM注意。8で安全/10で高速化(~13.5GB使用)。OOM時は下げる
os.environ["PF_WELL_CHUNK"]="20"        # (full平滑では無視)
import torch as _t
_ng=_t.cuda.device_count(); os.environ["PF_NGPU"]=str(max(1,_ng))
_names=[_t.cuda.get_device_name(i) for i in range(_ng)]
print(f"[GPU] {_ng} x " + (" / ".join(_names) if _ng else "CPU") + f" | PF_NGPU={os.environ['PF_NGPU']}")
if any("P100" in n for n in _names):
    raise SystemExit("★P100は Kaggle PyTorch 非対応(sm_60未サポート)。Settings->Accelerator を 'GPU T4 x2' に変更して再実行してください。")
if _ng<2: print("*** 推奨: T4 x2(2GPUでPF分散)。単GPUでも動きますが遅い ***")
print("LEG=",LEG," WORK=",WORK)
'''

RUN_T = r'''# ===== 逆方向DA 生成(1レグ, 復元可能) =====
import os, subprocess, sys
r=subprocess.run([sys.executable,"da_leg_bwd_v100.py"],env=dict(os.environ))
if r.returncode!=0: raise SystemExit(f"da_leg_bwd 異常終了 ({r.returncode})")
import pyarrow.parquet as pq
p=f"@@LEG@@_bwd.parquet"
m=pq.ParquetFile(p); print("完成:", p, "rows=",m.metadata.num_rows, "cols=",len(m.schema.names))
print(">> Save Version(commit)で /kaggle/working が出力データセット化。全レグ完了までは、この出力を次回の入力に追加して再実行=続きから。")
'''

for leg in LEGS:
    cells = [md(INTRO_T.replace("@@LEG@@", leg)), code(ENV_T.replace("@@LEG@@", leg))]
    for name in SCRIPTS:
        cells.append(code(f"%%writefile {name}\n" + (V99 / name).read_text(encoding="utf-8")))
    cells.append(code(RUN_T.replace("@@LEG@@", leg)))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
    out = OUTDIR / f"da_bwd_{leg}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[notebook] {out.name} cells={len(cells)}")

print(f"\n完了: {OUTDIR} に 3レグ分の逆方向DAノート + da_dataset_v100")
