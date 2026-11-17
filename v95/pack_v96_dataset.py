# -*- coding: utf-8 -*-
"""v96 Kaggle Dataset にアップロードするファイルだけを収集して 1フォルダに揃える。
出力: v95/kaggle_dataset_v96/  (これを zip 化 or フォルダごと Kaggle Dataset へ)
  kaggle_dataset_v96/artifacts_v95/   config3 + models_l1_fwd(45) + nn(15)
  kaggle_dataset_v96/artifacts_v95b/  l5_cf_ckpt(model.pt 15) + submit_v96_config.json
実行: .venv/Scripts/python.exe v95/pack_v96_dataset.py
"""
import sys, io, shutil
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
A = PROJ / "v95" / "artifacts_v95"
B = PROJ / "v95" / "artifacts_v95b"
OUT = PROJ / "v95" / "kaggle_dataset_v96"
OA = OUT / "artifacts_v95"
OB = OUT / "artifacts_v95b"
if OUT.exists():
    shutil.rmtree(OUT)
(OA / "models_l1_fwd").mkdir(parents=True, exist_ok=True)
(OA / "nn").mkdir(parents=True, exist_ok=True)
(OB / "l5_cf_ckpt").mkdir(parents=True, exist_ok=True)

VARIANTS = ["lightgbm-1", "lightgbm-2", "lightgbm-3",
            "catboost-1", "catboost-2", "catboost-3",
            "xgboost-1", "xgboost-2", "xgboost-3"]
n = 0
# config 3本
for f in ["pf_banks_config.json", "features_l1.json", "submit_pack_nn.json"]:
    shutil.copyfile(A / f, OA / f); n += 1
# GBM 45
for v in VARIANTS:
    (OA / "models_l1_fwd" / v).mkdir(parents=True, exist_ok=True)
    for k in range(5):
        shutil.copyfile(A / "models_l1_fwd" / v / f"fold{k}.pkl",
                        OA / "models_l1_fwd" / v / f"fold{k}.pkl"); n += 1
# TCN 15
for s in range(3):
    for k in range(5):
        shutil.copyfile(A / "nn" / f"combo_mdn2_s{s}_fold{k}.pt",
                        OA / "nn" / f"combo_mdn2_s{s}_fold{k}.pt"); n += 1
# L5cf 15 (model.pt のみ。valpred は不要)
for s in range(3):
    for k in range(5):
        shutil.copyfile(B / "l5_cf_ckpt" / f"l5_s{s}_f{k}_model.pt",
                        OB / "l5_cf_ckpt" / f"l5_s{s}_f{k}_model.pt"); n += 1
# L5cf config
shutil.copyfile(B / "submit_v96_config.json", OB / "submit_v96_config.json"); n += 1

# 検証
def cnt(p, pat): return len(list(p.rglob(pat)))
print(f"収集 {n} ファイル -> {OUT}")
print(f"  artifacts_v95/config json = {cnt(OA, '*.json')} (=3)")
print(f"  artifacts_v95/models_l1_fwd fold*.pkl = {cnt(OA/'models_l1_fwd', 'fold*.pkl')} (=45)")
print(f"  artifacts_v95/nn *.pt = {cnt(OA/'nn', '*.pt')} (=15)")
print(f"  artifacts_v95b/l5_cf_ckpt *.pt = {cnt(OB/'l5_cf_ckpt', '*.pt')} (=15)")
print(f"  artifacts_v95b/*.json = {cnt(OB, '*.json')} (=1)")
sz = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
print(f"  合計サイズ = {sz/1e6:.0f} MB")
print("\n次: このフォルダを zip 化するか、Kaggle Dataup で folder ごとアップロード。")
