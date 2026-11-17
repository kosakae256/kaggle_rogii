# -*- coding: utf-8 -*-
"""v102 学習オーケストレータ(自己完結): seqfeat → TCN(mdn2) → L5 → blend重み。
v100 と同じ 15バンク候補(K=71)。差分は create 側(emission全バンク+ps_combo)なので学習構成は同一。
前提: v102/create_v102.py で artifacts_v102/data/train_combo.parquet 生成済み。
env: NN_SEEDS(既定5) / SKIP_SEQFEAT / SKIP_TCN / SKIP_L5
実行: .venv/Scripts/python.exe v102/run_train_v102.py
"""
import os, sys, io, json, time, subprocess
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
NN_SEEDS = os.environ.get("NN_SEEDS", "5")
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")

# combo seqfeat の候補(v96 nbr_gr5/self_graft + v97全PF + v100新5バンク)= v100 と同一 K=71
ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
              + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])
ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
              + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)])


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V102 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def wait_parquet(p):
    p = Path(p); last = -1
    for _ in range(120):
        if p.exists():
            sz = p.stat().st_size
            if sz == last and sz > 0:
                try:
                    pq.read_schema(p); return True
                except Exception:
                    pass
            last = sz
        time.sleep(5)
    raise SystemExit(f"[wait] {p.name} が安定しない(未完/破損)")


def build_blend():
    nn = ART / "nn_oof_combo_fcombo.npz"; l5 = ART / "l5_fcombo_oof.npz"
    if not (nn.exists() and l5.exists()):
        print("[blend] OOF未検出 -> blend.json スキップ(submitは均等0.5/0.5)"); return
    dn = np.load(nn); dl = np.load(l5)
    tcn = np.mean([dn[k] for k in dn.files if k.startswith("mdn2_s")], 0)
    l5m = dl["l5_mean"] if "l5_mean" in dl.files else np.mean([dl[k] for k in dl.files if k.startswith("l5_s")], 0)
    y = dl["y"] if "y" in dl.files else pq.read_table(ART / "data" / "train_combo.parquet", columns=["target"]).column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    m = np.isfinite(tcn) & np.isfinite(l5m) & np.isfinite(y) & (l5m != 0)
    ws = np.arange(0, 1.0001, 0.02)
    sc = [np.sqrt(np.mean(((w * tcn + (1 - w) * l5m)[m] - y[m]) ** 2)) for w in ws]
    bi = int(np.argmin(sc)); w_tcn = float(ws[bi])
    (ART / "blend.json").write_text(json.dumps({"bw": [w_tcn, 1 - w_tcn], "rmse": float(sc[bi])}), encoding="utf-8")
    print(f"[blend] w_tcn={w_tcn:.2f} w_l5={1-w_tcn:.2f} OOF RMSE={sc[bi]:.4f} "
          f"(TCN {np.sqrt(np.mean((tcn[m]-y[m])**2)):.4f} / L5 {np.sqrt(np.mean((l5m[m]-y[m])**2)):.4f})", flush=True)


def main():
    t0 = time.time()
    print("[run_train_v102] normalのみ(DA無し)。emission全バンク+ps_combo は create 側で反映済み。", flush=True)
    wait_parquet(ART / "data" / "train_combo.parquet")
    if os.environ.get("SKIP_SEQFEAT", "0") != "1":
        _run("build_seqfeat_v97.py", dict(SF_IN=str(ART / "data" / "train_combo.parquet"), SF_OUT=str(ART / "seqfeat_combo_cf.pkl"),
             CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES,
             ROGII_SCRIPTS=str(V102), ROGII_DATA=DATA), "seqfeat[main]")
    if os.environ.get("SKIP_TCN", "0") != "1":
        _run("train_nn_v98.py", dict(ROGII_ART98=str(ART), V93_INPUT="combo", NN_SEEDS=NN_SEEDS,
             V98_DA="0", ROGII_DATA=DATA), f"TCN({NN_SEEDS}seed)")
    if os.environ.get("SKIP_L5", "0") != "1":
        _run("train_l456_v98.py", dict(ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5",
             SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA="0",
             L5_SF=str(ART / "seqfeat_combo_cf.pkl"), ROGII_DATA=DATA), f"L5({NN_SEEDS}seed)")
    build_blend()
    print(f"\n[run_train_v102] 完了 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
