# -*- coding: utf-8 -*-
"""取りこぼし候補を追加した seqfeat を非破壊生成(K71->83)。
現行 v100 ED に + pf_{nbr_gr5,self_graft,self,nbr}_delta_{4,5,6}(12本)を候補追加。
出力: artifacts_v100/seqfeat_x83_cf.pkl(現行 seqfeat_combo_cf.pkl は温存)。"""
import os, subprocess
from pathlib import Path

V98 = Path(__file__).resolve().parent.parent / "v99"
ART = Path(__file__).resolve().parent / "artifacts_v100"
PROJ = Path(__file__).resolve().parent.parent
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")

# --- 現行 v100 ED/ES(run_train_v100.py と一致) ---
ED = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_delta_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_delta_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4, 5)])
ES = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_std_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_std_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4, 5)])

# --- ★取りこぼし候補追加(EDにバンクは在るのにラグ1-3ハードコードで落ちてた4-6) delta_{4,5,6} × 2family = 6本 ---
ADD_D = ["%s_delta_%d" % (fam, k) for fam in ("pf_nbr_gr5", "pf_self_graft") for k in (4, 5, 6)]
ADD_S = ["%s_std_%d" % (fam, k) for fam in ("pf_nbr_gr5", "pf_self_graft") for k in (4, 5, 6)]
ED += ADD_D; ES += ADD_S
print(f"[build_seqfeat_x] 追加候補 {len(ADD_D)}本: {ADD_D}", flush=True)
print(f"[build_seqfeat_x] EXTRA_DELTA 合計 {len(ED)}本(現行51 + 追加6)", flush=True)

env = dict(os.environ)
env.update(dict(
    SF_IN=str(ART / "data" / "train_combo.parquet"),
    SF_OUT=str(ART / "seqfeat_x77_cf.pkl"),
    CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
    EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES),
    ROGII_SCRIPTS=str(V98), ROGII_DATA=DATA, PYTHONUNBUFFERED="1",
))
r = subprocess.run([PY, str(V98 / "build_seqfeat_v97.py")], env=env)
raise SystemExit(r.returncode)
