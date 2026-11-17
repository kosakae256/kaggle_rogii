set -e
export ROGII_DATA="C:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction"
export ROGII_PROJ="C:/Users/kosaka256/Documents/rogii_claude"
export PF_WELL_CHUNK="20"    # smooth_lag=192 VRAM対策
PY=".venv/Scripts/python.exe"
for sys in normal bwd fwd; do
  echo "########## CREATE SYSTEM=$sys $(date +%H:%M) ##########"
  SYSTEM=$sys $PY v98/create_v98.py
done
echo "########## TRAIN (TCN+L5 5seed+DA) $(date +%H:%M) ##########"
$PY v98/run_train_v98.py
echo "########## PIPELINE DONE $(date +%H:%M) ##########"
