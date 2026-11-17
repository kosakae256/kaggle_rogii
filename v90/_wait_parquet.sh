#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PQ="v90/artifacts_v90/data/train.parquet"
echo "waiting for $PQ ... $(date +%T)"
until [ -f "$PQ" ]; do sleep 60; done
sleep 15   # 書き込み完了待ち
sz=$(stat -c %s "$PQ" 2>/dev/null)
echo "train.parquet ready: ${sz} bytes $(date +%T)"
