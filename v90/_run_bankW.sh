#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
for B in p_1 p_2 p_3 p_4 p_5; do
  for W in 0 0.005 0.01 0.02; do
    echo "=== BANK=$B W=$W $(date +%T) ==="
    GRID=emission BANK=$B BETA=$W NW=0 CHUNK=30 "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE"
  done
done
echo "=== done $(date +%T) ==="
