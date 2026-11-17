#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
for B in 0.0 0.01 0.02 0.03; do
  echo "=== emission BETA=$B 全fold0 $(date +%T) ==="
  GRID=emission BETA=$B NW=0 CHUNK=30 "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE"
done
echo "=== done $(date +%T) ==="
