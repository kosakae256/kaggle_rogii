# -*- coding: utf-8 -*-
"""対照: v100 combo を v102 と同一パイプライン(注入なし)で L5。
私の run_train seqfeat 設定(CANDFEAT/NF_MAX/ED/ES)が v100 の production(l5_fcombo=5.6684)を
再現するかの基準線。再現すれば「注入=害/v100最良」が確定、ズレれば pipeline 差と判明。
出力 seqfeat_v100ctrl_cf.pkl / l5_v100ctrl_oof.npz。実行: .venv/Scripts/python.exe v102/run_v100ctrl_l5.py"""
import os, sys, io, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path

V102 = Path(__file__).resolve().parent
PROJ = V102.parent
ART = V102 / "artifacts_v102"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
NN_SEEDS = os.environ.get("NN_SEEDS", "5")
V100_COMBO = PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet"
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")
ED = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V102 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def main():
    t0 = time.time()
    print(f"[v100ctrl] v100 combo を v102パイプラインで(注入なし)L5({NN_SEEDS}seed)", flush=True)
    _run("build_seqfeat_v97.py", dict(
        SF_IN=str(V100_COMBO), SF_OUT=str(ART / "seqfeat_v100ctrl_cf.pkl"),
        CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
        EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES),
        ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA), "seqfeat[v100ctrl]")
    _run("train_l456_v98.py", dict(
        ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SF_IN=str(V100_COMBO),
        SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA="0",
        L5_SF=str(ART / "seqfeat_v100ctrl_cf.pkl"), L5_OSUF="_v100ctrl", ROGII_DATA=DATA),
        f"L5[v100ctrl]({NN_SEEDS}seed)")
    print(f"\n[v100ctrl] 完了 ({time.time()-t0:.0f}s) -> l5_v100ctrl_oof.npz", flush=True)


if __name__ == "__main__":
    main()
