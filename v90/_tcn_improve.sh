#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1
echo "=== topk256(1seed, 特徴増で更に効くか) $(date +%T) ==="
XTAG=topk256 XSEEDS=0 XTOPK=256 "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|OOF raw|blend :|Error"
echo "=== combo3(baseline 3seed=本命) $(date +%T) ==="
XTAG=combo3 XSEEDS=0,1,2 "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|OOF raw|blend :|Error"
echo "=== TCN improve done $(date +%T) ==="
