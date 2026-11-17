#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
OOF="v90/artifacts_v90/gbm_v41_oof.npy"
export PYTHONUNBUFFERED=1
echo "=== wait tabular (gbm_v41_oof.npy) $(date +%T) ==="
until [ -f "$OOF" ]; do sleep 120; done
sleep 30
echo "=== TCN baseline combo_s0 (1seed) $(date +%T) ==="
XTAG=combo_s0 XSEEDS=0 "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|fold|OOF|blend|rmse|Error|Traceback"
echo "=== TCN noemis (emission+SDF除外) $(date +%T) ==="
XTAG=noemis XSEEDS=0 XEXCL="pf_.*_(corr|diff|absdiff)|^sdf_" "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|除外|fold|OOF|blend|rmse|Error"
echo "=== TCN hidrop (DROP0.2) $(date +%T) ==="
XTAG=hidrop XSEEDS=0 XDROP=0.2 "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|fold|OOF|blend|rmse|Error"
echo "=== TCN topk128 $(date +%T) ==="
XTAG=topk128 XSEEDS=0 XTOPK=128 "$PY" v90/train_nn_exp_v90.py 2>&1 | grep -aE "exp\]|fold|OOF|blend|rmse|Error"
echo "=== TCN experiments done $(date +%T) ==="
