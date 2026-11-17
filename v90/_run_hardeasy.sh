#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
echo "=== BETA=0 全fold0 $(date +%T) ==="
BETA=0 NW=0 CHUNK=30 SAVEPW=pw_b0.npz "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE|pw saved"
echo "=== BETA=0.3 全fold0 $(date +%T) ==="
BETA=0.3 NW=0 CHUNK=30 SAVEPW=pw_b3.npz "$PY" v90/guided_pf_v90.py 2>&1 | grep -aE "overall RMSE|pw saved"
echo "=== done $(date +%T) ==="
