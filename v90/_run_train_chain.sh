#!/bin/bash
cd /c/Users/kosaka256/Documents/rogii_claude
PY="C:/Users/kosaka256/Documents/rogii_claude/.venv/Scripts/python.exe"
PQ="v90/artifacts_v90/data/train.parquet"
echo "=== wait+verify parquet $(date +%T) ==="
until [ -f "$PQ" ] && "$PY" -c "import pyarrow.parquet as pq,sys; pq.ParquetFile(sys.argv[1]).metadata" "$PQ" 2>/dev/null; do sleep 60; done
echo "=== parquet OK, TABULAR $(date +%T) ==="
"$PY" v90/_run_nb.py v90/train_tabular_v90.ipynb || { echo "TABULAR FAILED"; exit 1; }
echo "=== TCN baseline(combo,1seed) $(date +%T) ==="
XTAG=combo_s0 XSEEDS=0 "$PY" v90/train_nn_exp_v90.py || echo "TCN base FAILED"
echo "=== TCN noemis(emission+SDF除外,1seed) $(date +%T) ==="
XTAG=noemis XSEEDS=0 XEXCL="pf_.*_(corr|diff|absdiff)|^sdf_" "$PY" v90/train_nn_exp_v90.py || echo "TCN noemis FAILED"
echo "=== TCN hidrop(DROP0.2,1seed) $(date +%T) ==="
XTAG=hidrop XSEEDS=0 XDROP=0.2 "$PY" v90/train_nn_exp_v90.py || echo "TCN hidrop FAILED"
echo "=== TCN topk128(1seed) $(date +%T) ==="
XTAG=topk128 XSEEDS=0 XTOPK=128 "$PY" v90/train_nn_exp_v90.py || echo "TCN topk128 FAILED"
echo "=== CHAIN done $(date +%T) ==="
