#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
for B in 0.005 0.008; do
  echo "=== 全773 emission BETA=$B $(date +%T) ==="
  GRID=emission ALLWELLS=1 BETA=$B NW=0 CHUNK=30 "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE"
done
echo "=== done $(date +%T) ==="
