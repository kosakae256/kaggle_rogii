# -*- coding: utf-8 -*-
"""ハイブリッド: v100 base(v96/pf5=emission OFF=多様, v97=emission)+ v102の v97(emission+ps_combo)を注入 → L5。
・nn_emission を v96/pf5 に掛けたのが v102 の回帰原因(候補多様性 −41%/−45%)と判明。
・修正 = v96/pf5 は v100(多様)のまま、v97 だけ v102(ps_combo版)を足す。
・実装: seqfeat SF_IN=v100 combo(=v96/pf5 多様 + v97 emission)、v102の v97候補(20 delta)を vb_ 注入。
・出力 seqfeat_hybrid_cf.pkl / l5_hybrid_oof.npz(既存を壊さない)。40GB結合 parquet は作らない。id順は完全一致=直接注入。
実行: .venv/Scripts/python.exe v102/run_hybrid_l5.py"""
import os, sys, io, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

V102 = Path(__file__).resolve().parent
PROJ = V102.parent
ART = V102 / "artifacts_v102"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
NN_SEEDS = os.environ.get("NN_SEEDS", "5")
V102_COMBO = ART / "data" / "train_combo.parquet"
V100_COMBO = PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet"
SCRATCH = Path(os.environ.get("SCRATCH", str(ART)))
NPZ = SCRATCH / "vc_v102_v97.npz"

REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")
# seqfeat base(v100 combo)の候補列(=v102 と同名)
ED = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
# 注入 = v102 の v97候補(emission+ps_combo)20本(delta)
INJ = ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V102 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def build_injection_npz():
    if NPZ.exists():
        print(f"[hybrid] 注入npz 既存流用: {NPZ.name}", flush=True); return
    i100 = pq.read_table(V100_COMBO, columns=["id"]).column("id").to_pandas().astype(str).to_numpy()
    i102 = pq.read_table(V102_COMBO, columns=["id"]).column("id").to_pandas().astype(str).to_numpy()
    assert len(i100) == len(i102) and np.array_equal(i100, i102), "id順不一致=直接注入不可"
    print(f"[hybrid] id順一致({len(i100):,})。v102 の v97候補 {len(INJ)}本(emission+ps_combo)を vc_ で読込...", flush=True)
    tb = pq.read_table(V102_COMBO, columns=INJ)
    inj = {"vc_" + c: np.asarray(tb.column(c).to_numpy(zero_copy_only=False), np.float32) for c in INJ}
    del tb
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **inj)
    print(f"[hybrid] 注入npz 保存: {NPZ.name}({len(inj)}候補, {NPZ.stat().st_size/1e6:.0f}MB)", flush=True)


def main():
    t0 = time.time()
    print(f"[hybrid] base=v100 combo(v96/pf5 多様 + v97 emission)+ v102の v97 {len(INJ)}本注入 → L5({NN_SEEDS}seed)", flush=True)
    build_injection_npz()
    # seqfeat: SF_IN=v100 combo(v96/pf5=多様)。v102の v97 を vc_ 注入。
    _run("build_seqfeat_v97.py", dict(
        SF_IN=str(V100_COMBO), SF_OUT=str(ART / "seqfeat_hybrid_cf.pkl"),
        CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
        EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES), EUNET_NPZ=str(NPZ),
        ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA), "seqfeat[hybrid]")
    # L5: target/well は v100 combo(v102と同一)。L5_OSUF=_hybrid で別出力。
    _run("train_l456_v98.py", dict(
        ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SF_IN=str(V100_COMBO),
        SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA="0",
        L5_SF=str(ART / "seqfeat_hybrid_cf.pkl"), L5_OSUF="_hybrid", ROGII_DATA=DATA),
        f"L5[hybrid]({NN_SEEDS}seed)")
    print(f"\n[hybrid] 完了 ({time.time()-t0:.0f}s) -> l5_hybrid_oof.npz", flush=True)


if __name__ == "__main__":
    main()
