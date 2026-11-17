# -*- coding: utf-8 -*-
"""候補別特徴を全10役割にした seqfeat(NF_MAX=10)。ED=v100基準(K=71, 候補は不変)。
出力 seqfeat_nf10_cf.pkl(現行NF2は温存)。GR適合/空間整合/一致度/rate を各候補に付与。"""
import os, subprocess
from pathlib import Path
V98=Path(__file__).resolve().parent.parent/"v99"; ART=Path(__file__).resolve().parent/"artifacts_v100"
PROJ=Path(__file__).resolve().parent.parent; DATA=str(PROJ/"rogii-wellbore-geology-prediction")
PY=str(PROJ/".venv"/"Scripts"/"python.exe")
ED=(["pf_nbr_gr5_delta_%d"%k for k in(1,2,3)]+["pf_self_graft_delta_%d"%k for k in(1,2,3)]
    +["v97_pf_%s_delta_%d"%(r,k) for r in("ancc","self","nbr","nbr_gr5","self_graft") for k in(1,2,3,4)]
    +["v100_pf_%s_delta_%d"%(r,k) for r in("ancc","self","nbr","nbr_gr5","self_graft") for k in(1,2,3,4,5)])
ES=(["pf_nbr_gr5_std_%d"%k for k in(1,2,3)]+["pf_self_graft_std_%d"%k for k in(1,2,3)]
    +["v97_pf_%s_std_%d"%(r,k) for r in("ancc","self","nbr","nbr_gr5","self_graft") for k in(1,2,3,4)]
    +["v100_pf_%s_std_%d"%(r,k) for r in("ancc","self","nbr","nbr_gr5","self_graft") for k in(1,2,3,4,5)])
env=dict(os.environ); env.update(dict(
    SF_IN=str(ART/"data"/"train_combo.parquet"), SF_OUT=str(ART/"seqfeat_nf10_cf.pkl"),
    CANDFEAT="1", NF_MAX="10", V97_NBANK="6", V93_INPUT="fwd",
    EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES),
    ROGII_SCRIPTS=str(V98), ROGII_DATA=DATA, PYTHONUNBUFFERED="1"))
raise SystemExit(subprocess.run([PY,str(V98/"build_seqfeat_v97.py")],env=env).returncode)
