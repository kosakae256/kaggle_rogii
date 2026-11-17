#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
for F in 1 2 3 4; do
  echo "=== SDF fold$F start $(date +%T) ==="
  FOLD=$F SAVE_OOF=1 SAVE=v89_sdf_ddil_f$F "$PY" v89/train_sdf_v89.py
done
echo "=== all SDF folds done $(date +%T) ==="
