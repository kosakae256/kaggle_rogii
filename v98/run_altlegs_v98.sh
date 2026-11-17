#!/usr/bin/env bash
# v98差し込み: 脱相関脚(GRU=安全, Transformer=小型+温度ガード)を学習しblend増分を測る。
# PIPELINE DONE(TCN/L5完了)後に起動する。特徴/fold/mdn2はTCNと同一、胴体だけ差替。
cd /c/Users/kosaka256/Documents/rogii_claude
PY=".venv/Scripts/python.exe"
export ROGII_DATA="C:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction"
export ROGII_ART98="C:/Users/kosaka256/Documents/rogii_claude/v98/artifacts_v98"
export ROGII_PROJ="C:/Users/kosaka256/Documents/rogii_claude"
export V93_INPUT="combo" V98_DA="0"    # normalのみ(SKIP_DA相当)
TEMP_KILL=${TEMP_KILL:-84}             # GPU温度これ超でkill(PC保護)

# 事前スモーク(40井・バグ検出。RAM軽い)
echo "---- smoke gru ----";       SMOKE=1 ALT_ARCH=gru $PY v98/train_altnn_v98.py 2>&1 | tail -4 || { echo "smoke gru NG"; exit 1; }
echo "---- smoke transformer ----"; SMOKE=1 ALT_ARCH=transformer ALT_BATCH=4 $PY v98/train_altnn_v98.py 2>&1 | tail -4 || { echo "smoke tfmr NG"; exit 1; }

echo "########## ALT-LEG GRU $(date +%H:%M) ##########"
ALT_ARCH=gru NN_SEEDS=3 ALT_EPOCHS=45 $PY v98/train_altnn_v98.py
echo "---- blend(after GRU) ----"; $PY v98/blend_compare_v98.py

echo "########## ALT-LEG TRANSFORMER(小型+温度ガード) $(date +%H:%M) ##########"
ALT_ARCH=transformer ALT_BATCH=6 ALT_NLAYERS=4 ALT_NHEAD=6 NN_SEEDS=3 ALT_EPOCHS=40 $PY v98/train_altnn_v98.py &
TPID=$!
while kill -0 $TPID 2>/dev/null; do
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$temp" ] && [ "$temp" -gt "$TEMP_KILL" ]; then
    echo "★GPU ${temp}C > ${TEMP_KILL}C: Transformer停止(PC保護)"
    powershell.exe -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_altnn_v98*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
    break
  fi
  sleep 15
done
wait $TPID 2>/dev/null
echo "---- blend(after Transformer) ----"; $PY v98/blend_compare_v98.py
echo "########## ALT-LEG DONE $(date +%H:%M) ##########"
