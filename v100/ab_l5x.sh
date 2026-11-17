#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
export ROGII_DATA="C:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction"
export ROGII_PROJ="C:/Users/kosaka256/Documents/rogii_claude"
SF="C:/Users/kosaka256/Documents/rogii_claude/v99/artifacts_v99/seqfeat_combo_cf.pkl"
ART="C:/Users/kosaka256/Documents/rogii_claude/v99/artifacts_v99"
echo "=== L5 vs L5X A/B (fold0, seed0, v99 combo 10バンク) $(date +%H:%M:%S) ==="
for A in l5 l5x; do
  echo "--- ARCH=$A ---"
  ROGII_ART98="$ART" V93_INPUT=combo ARCH=$A SEEDS=0 FOLD_ONLY=0 V98_DA=0 \
    L5_SF="$SF" L5_OSUF="_ab" MAXMIN=30 TOKB=1500 VFRAC=0.15 \
    .venv/Scripts/python.exe v99/train_l456_v98.py 2>&1 | grep -E "val rmse|wells=|ARCH=" | tail -3
done
echo "=== A/B 完了 $(date +%H:%M:%S) ==="
