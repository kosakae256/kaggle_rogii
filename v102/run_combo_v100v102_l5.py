# -*- coding: utf-8 -*-
"""特徴結合 v102 + v100 → L5(結合ファイルは作らず、v100候補を seqfeat の EUNET_NPZ フックでメモリ注入)。
・v102 combo = 本体(K=71: base14+bpf6+extra51, ps_combo/emission全=シャープ候補)。
・v100 combo = 同じ51候補(delta)を "vb_" 接頭で追加注入(ps_combo無し=多様候補)。id順は完全一致=直接注入。
・出力 seqfeat_combo2_cf.pkl(通常のseqfeatキャッシュ) / l5_combo2_oof.npz(L5_OSUFで別出力=既存を壊さない)。
・40GB の結合 parquet は作らない。注入 npz は scratchpad(transient)。
実行: .venv/Scripts/python.exe v102/run_combo_v100v102_l5.py"""
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
SCRATCH = Path(os.environ.get("SCRATCH", str(V102 / "artifacts_v102")))
NPZ = SCRATCH / "vb_v100_cands.npz"

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


def build_injection_npz():
    if NPZ.exists():
        print(f"[combo] 注入npz 既存流用: {NPZ.name}", flush=True); return
    # id順が完全一致していることを確認(直接注入の前提)
    i102 = pq.read_table(V102_COMBO, columns=["id"]).column("id").to_pandas().astype(str).to_numpy()
    i100 = pq.read_table(V100_COMBO, columns=["id"]).column("id").to_pandas().astype(str).to_numpy()
    assert len(i102) == len(i100) and np.array_equal(i102, i100), "id順不一致=直接注入不可(要align)"
    print(f"[combo] id順一致 確認({len(i102):,}行)。v100 の {len(ED)}候補(delta)を vb_ で読込...", flush=True)
    tb = pq.read_table(V100_COMBO, columns=ED)
    inj = {}
    for c in ED:
        a = np.asarray(tb.column(c).to_numpy(zero_copy_only=False), np.float32)
        inj["vb_" + c] = a
    del tb
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ, **inj)
    print(f"[combo] 注入npz 保存: {NPZ.name}({len(inj)}候補, {NPZ.stat().st_size/1e6:.0f}MB)", flush=True)


def main():
    t0 = time.time()
    print(f"[combo] v102(K=71 シャープ)+ v100({len(ED)} 多様, vb_接頭)= K≈{71+len(ED)} → L5({NN_SEEDS}seed)", flush=True)
    build_injection_npz()
    # seqfeat: v102本体 + EUNET_NPZ で vb_ 注入(STD=None=既定2.5)。出力は別名。
    _run("build_seqfeat_v97.py", dict(
        SF_IN=str(V102_COMBO), SF_OUT=str(ART / "seqfeat_combo2_cf.pkl"),
        CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
        EXTRA_DELTA=",".join(ED), EXTRA_STD=",".join(ES), EUNET_NPZ=str(NPZ),
        ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA), "seqfeat[combo2]")
    # L5: 結合 seqfeat で学習。L5_OSUF=_combo2 で別出力(既存 l5_oof を保持)。
    _run("train_l456_v98.py", dict(
        ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5",
        SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA="0",
        L5_SF=str(ART / "seqfeat_combo2_cf.pkl"), L5_OSUF="_combo2", ROGII_DATA=DATA),
        f"L5[combo2]({NN_SEEDS}seed)")
    print(f"\n[combo] 完了 ({time.time()-t0:.0f}s) -> l5_combo2_oof.npz", flush=True)


if __name__ == "__main__":
    main()
