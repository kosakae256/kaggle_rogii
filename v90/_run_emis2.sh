#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
echo "### 全773 一般化確認 ###"
for B in 0.0 0.01; do
  echo "=== 全773 emission BETA=$B $(date +%T) ==="
  GRID=emission ALLWELLS=1 BETA=$B NW=0 CHUNK=30 "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE"
done
echo "### fold0 W微調整 ###"
for B in 0.005 0.008 0.012 0.015; do
  echo "=== fold0 emission BETA=$B $(date +%T) ==="
  GRID=emission BETA=$B NW=0 CHUNK=30 "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE"
done
echo "=== done $(date +%T) ==="
