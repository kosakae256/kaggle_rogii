cd /c/Users/kosaka256/Documents/rogii_claude
export LEG=v96 BWD_FWD_PREFIX=""
export ROGII_ART95=v100/bwd_local_v96 ROGII_ART97=v100/bwd_local_v96
export ROGII_DATA=rogii-wellbore-geology-prediction ROGII_PROJ=/c/Users/kosaka256/Documents/rogii_claude
export BWD_ENDPOINTS=v100/bwd_local_v96/bwd_endpoints.parquet
export SIM_PKL=v100/bwd_local_v96/sim_grfree_v97.pkl
export OUT_BWD=v100/artifacts_v100/bpf_v96.parquet
export BWD_SMOOTH_LAG=16 BWD_W_NN=0.02 PF_WELL_CHUNK=40 PF_NGPU=1 PYTHONUNBUFFERED=1
echo "=== [1/2] emission sim 生成(nn_emission, train, 773井) $(date +%H:%M:%S) ==="
NN_SPLIT=train .venv/Scripts/python.exe v99/nn_emission_v97.py
if [ $? -ne 0 ]; then echo "nn_emission 失敗"; exit 1; fi
echo "=== [2/2] 逆方向PF生成(v96, 6バンク×5表現, smooth16, emission込み) $(date +%H:%M:%S) ==="
.venv/Scripts/python.exe v99/build_backward_v100.py
echo "=== 完了 $(date +%H:%M:%S) ==="
