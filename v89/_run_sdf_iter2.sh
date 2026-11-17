#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
echo "=== build learned_sdf2.pkl (SDF OOF中心) $(date +%T) ==="
OUT=learned_sdf2.pkl CENTER_NPZ="C:/Users/kosaka256/Documents/rogii_claude/v64/artifacts_v64/v89_sdf_ddil_f0_oof.npz,C:/Users/kosaka256/Documents/rogii_claude/v64/artifacts_v64/v89_sdf_ddil_f1_oof.npz,C:/Users/kosaka256/Documents/rogii_claude/v64/artifacts_v64/v89_sdf_ddil_f2_oof.npz,C:/Users/kosaka256/Documents/rogii_claude/v64/artifacts_v64/v89_sdf_ddil_f3_oof.npz,C:/Users/kosaka256/Documents/rogii_claude/v64/artifacts_v64/v89_sdf_ddil_f4_oof.npz" "$PY" v89/build_learned_img_v89.py
echo "=== SDF iter2 学習 5fold $(date +%T) ==="
for F in 0 1 2 3 4; do
  echo "--- iter2 fold$F $(date +%T) ---"
  EVPKL=v89/artifacts_v89/learned_sdf2.pkl FOLD=$F SAVE_OOF=1 SAVE=v89_sdf2_ddil_f$F "$PY" v89/train_sdf_v89.py
done
echo "=== SDF iter2 done $(date +%T) ==="
