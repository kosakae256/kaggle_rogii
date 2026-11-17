# -*- coding: utf-8 -*-
"""v98 学習オーケストレータ: (1)seqfeat×3(main/bwd/fwd) → (2)TCN 5seed(DA) → (3)L5 5seed(DA)
   → (4)blend重み導出(TCN vs L5, OOF overall RMSE最小) → artifacts_v98/blend.json。
前提: create_v98 で train_combo / train_aug_bwd_combo / train_aug_fwd_combo.parquet が生成済み
     (書込中の取り込み防止: 各parquetのサイズ安定＋pq.read_schema 成功を確認してから学習)。
env: NN_SEEDS(既定5) / SKIP_SEQFEAT / SKIP_TCN / SKIP_L5
実行: .venv/Scripts/python.exe v98/run_train_v98.py
"""
import os, sys, io, json, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

V98 = Path(__file__).resolve().parent
PROJ = V98.parent
ART = V98 / "artifacts_v98"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
NN_SEEDS = os.environ.get("NN_SEEDS", "5")
DATA = str(PROJ / "rogii-wellbore-geology-prediction")

# combo seqfeat の EXTRA(v96 nbr_gr5/self_graft + v97 全PF) = submit と同一
ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_delta_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)])
ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
              + ["v97_pf_%s_std_%d" % (r, k) for r in ("ancc", "self", "nbr", "nbr_gr5", "self_graft") for k in (1, 2, 3, 4)])
SEQ = {"main": ("train_combo.parquet", "seqfeat_combo_cf.pkl"),
       "bwd":  ("train_aug_bwd_combo.parquet", "seqfeat_aug_bwd_cf.pkl"),
       "fwd":  ("train_aug_fwd_combo.parquet", "seqfeat_aug_fwd_cf.pkl")}


def _run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(V98 / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def wait_parquet(p):
    """書込中を避ける: サイズが安定し pq.read_schema が通るまで待つ。"""
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
    """TCN OOF と L5 OOF から overall RMSE 最小の blend 重みを導出。"""
    nn = ART / "nn_oof_combo_fcombo.npz"; l5 = ART / "l5_fcombo_oof.npz"
    if not (nn.exists() and l5.exists()):
        print("[blend] OOF未検出 -> blend.json スキップ(submitは均等0.5/0.5)"); return
    dn = np.load(nn); dl = np.load(l5)
    tcn = np.mean([dn[k] for k in dn.files if k.startswith("mdn2_s")], 0)
    l5m = dl["l5_mean"] if "l5_mean" in dl.files else np.mean([dl[k] for k in dl.files if k.startswith("l5_s")], 0)
    y = dl["y"] if "y" in dl.files else None
    if y is None:
        meta = pq.read_table(ART / "data" / "train_combo.parquet", columns=["target"]); y = meta.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    m = np.isfinite(tcn) & np.isfinite(l5m) & np.isfinite(y) & (l5m != 0)
    ws = np.arange(0, 1.0001, 0.02)
    sc = [np.sqrt(np.mean(((w * tcn + (1 - w) * l5m)[m] - y[m]) ** 2)) for w in ws]
    bi = int(np.argmin(sc)); w_tcn = float(ws[bi])
    (ART / "blend.json").write_text(json.dumps({"bw": [w_tcn, 1 - w_tcn], "rmse": float(sc[bi])}), encoding="utf-8")
    print(f"[blend] w_tcn={w_tcn:.2f} w_l5={1-w_tcn:.2f} OOF RMSE={sc[bi]:.4f} "
          f"(TCN単体 {np.sqrt(np.mean((tcn[m]-y[m])**2)):.4f} / L5単体 {np.sqrt(np.mean((l5m[m]-y[m])**2)):.4f})", flush=True)


def main():
    t0 = time.time()
    skip_da = (V98 / "SKIP_DA").exists()          # ★DA一旦停止=normalのみ学習(DA資産は保持=再開可)
    da = "0" if skip_da else "1"
    seqs = {"main": SEQ["main"]} if skip_da else SEQ
    if skip_da:
        print("[run_train_v98] ★SKIP_DA: normalのみ学習(aug seqfeat/DA連結なし)", flush=True)
    for sysname, (pqn, _sf) in seqs.items():
        wait_parquet(ART / "data" / pqn)
    if os.environ.get("SKIP_SEQFEAT", "0") != "1":
        for sysname, (pqn, sf) in seqs.items():
            _run("build_seqfeat_v97.py", dict(SF_IN=str(ART / "data" / pqn), SF_OUT=str(ART / sf),
                 CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES,
                 ROGII_SCRIPTS=str(V98), ROGII_DATA=DATA), f"seqfeat[{sysname}]")
    if os.environ.get("SKIP_TCN", "0") != "1":
        _run("train_nn_v98.py", dict(ROGII_ART98=str(ART), V93_INPUT="combo", NN_SEEDS=NN_SEEDS,
             V98_DA=da, ROGII_DATA=DATA), f"TCN(5seed{'' if skip_da else '+DA'})")
    if os.environ.get("SKIP_L5", "0") != "1":
        l5env = dict(ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5",
                     SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))), V98_DA=da,
                     L5_SF=str(ART / "seqfeat_combo_cf.pkl"), ROGII_DATA=DATA)
        if not skip_da:
            l5env["L5_SF_BWD"] = str(ART / "seqfeat_aug_bwd_cf.pkl")
            l5env["L5_SF_FWD"] = str(ART / "seqfeat_aug_fwd_cf.pkl")
        _run("train_l456_v98.py", l5env, f"L5(5seed{'' if skip_da else '+DA'})")
    build_blend()
    print(f"\n[run_train_v98] 完了 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
