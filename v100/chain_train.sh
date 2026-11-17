#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
CB=v100/artifacts_v100/data/train_combo.parquet
echo "[chain] $(date +%H:%M:%S) create完了待ち(train_combo安定)..."
while [ ! -f "$CB" ]; do sleep 120; done
prev=-1
while :; do
  cur=$(stat -c%s "$CB" 2>/dev/null || echo 0)
  [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ] && break
  prev=$cur; sleep 60
done
echo "[chain] $(date +%H:%M:%S) train_combo安定(size=$cur)。run_train_v100起動"
export ROGII_DATA="C:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction"
export ROGII_PROJ="C:/Users/kosaka256/Documents/rogii_claude"
export NN_SEEDS=5
nohup .venv/Scripts/python.exe v99/run_train_v100.py > v100/train_full.log 2>&1 &
echo "[chain] train PID $!"
# blend.json 完成待ち
while [ ! -f v100/artifacts_v100/blend.json ]; do sleep 120; done
sleep 5
echo "=== [chain] V100 TRAIN 完了 $(date +%H:%M:%S) ==="
cat v100/artifacts_v100/blend.json
