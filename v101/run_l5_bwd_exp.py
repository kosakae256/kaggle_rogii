# -*- coding: utf-8 -*-
"""v101 実験: 現状の bwd特徴(v96のみ, bpf_v96)を現行最良L5(v100, ARCH=l5, K=71)に候補追加して 3seed で効果を見る。
  baseline = seqfeat_combo_cf.pkl(K71) / +bwd = seqfeat_combo_bwd_cf.pkl(K71+30=101, bpf_{5rep}_delta/std_1..6)。
  両方 3seed(0,1,2) DA無しで L5 OOF を出し、target単位 overall RMSE(=TVT RMSE)を比較。
前提: v101/merge_bwd_v101.py で train_combo_bwd.parquet 生成済み。実行: .venv/Scripts/python.exe v101/run_l5_bwd_exp.py
env: SKIP_SEQFEAT/SKIP_BASE/SKIP_BWD で段階スキップ可。"""
import os, sys, subprocess, time
from pathlib import Path
import numpy as np

V99 = Path("v99"); ART = Path("v100/artifacts_v100"); PY = sys.executable
DATA = str((Path("rogii-wellbore-geology-prediction")).resolve())
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")

ED = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
      + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
      + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
# ★tw(ancc)は warm-init バグで全滅 → 正常な4表現のみ(self/nbr/nbr_gr5/self_graft × 6バンク = 24候補)
SANE = ["self", "nbr", "nbr_gr5", "self_graft"]
BWD_D = ["bpf_%s_delta_%d" % (r, k) for r in SANE for k in range(1, 7)]   # 24
BWD_S = ["bpf_%s_std_%d" % (r, k) for r in SANE for k in range(1, 7)]     # 24
ED_BWD = ED + BWD_D; ES_BWD = ES + BWD_S


def run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ({time.strftime('%H:%M:%S')}) ====", flush=True)
    r = subprocess.run([PY, str(V99 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 code={r.returncode}")


K_ADD = len(BWD_D)
if os.environ.get("SKIP_SEQFEAT", "0") != "1":
    run("build_seqfeat_v97.py", dict(
        SF_IN=str(ART / "data" / "train_combo_bwd.parquet"), SF_OUT=str(ART / "seqfeat_combo_bwdsane_cf.pkl"),
        CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd",
        EXTRA_DELTA=",".join(ED_BWD), EXTRA_STD=",".join(ES_BWD),
        ROGII_SCRIPTS=str(V99), ROGII_DATA=DATA), f"seqfeat_bwdsane(K=71+{K_ADD})")

L5C = dict(ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SEEDS="0,1,2", V98_DA="0", ROGII_DATA=DATA)
if os.environ.get("SKIP_BASE", "0") != "1":
    run("train_l456_v98.py", dict(L5C, L5_SF=str(ART / "seqfeat_combo_cf.pkl"), L5_OSUF="_base3"), "L5 baseline(K71,3seed)")
if os.environ.get("SKIP_BWD", "0") != "1":
    run("train_l456_v98.py", dict(L5C, L5_SF=str(ART / "seqfeat_combo_bwdsane_cf.pkl"), L5_OSUF="_bwdsane3"), f"L5 +bwd正常(K{71+K_ADD},3seed)")


def rmse(npz):
    p = ART / npz
    if not p.exists():
        return None, 0
    d = np.load(p); y = d["y"]; m = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
    ok = np.isfinite(y) & np.isfinite(m) & (m != 0)
    return float(np.sqrt(np.mean((m[ok] - y[ok]) ** 2))), int(ok.sum())


rb, nb = rmse("l5_base3_oof.npz"); rw, nw = rmse("l5_bwdsane3_oof.npz")
print("\n" + "=" * 60)
print(f"[結果] L5 baseline (K71,      3seed) OOF overall RMSE = {rb:.4f}  (n={nb})")
print(f"[結果] L5 +bwd正常 (K{71+len(BWD_D)}, 3seed) OOF overall RMSE = {rw:.4f}  (n={nw})")
if rb and rw:
    print(f"[結果] Δ(+bwd − baseline) = {rw - rb:+.4f}   ({'改善' if rw < rb else '悪化'})")
print("=" * 60)
