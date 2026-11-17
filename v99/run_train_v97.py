# -*- coding: utf-8 -*-
"""v97 train オーケストレーション。train.parquet 完成を待って GBM→TCN→seqfeat(CANDFEAT)→L5cf→blend重み。
各ステップ subprocess(env-before-import 厳守)。L5cf 失敗でも GBM/TCN は残る。
実行: PYTHONUNBUFFERED=1 .venv/Scripts/python.exe v97/run_train_v97.py"""
import os, sys, io, json, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V97 = PROJ / "v97"; ART = V97 / "artifacts_v97"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
PQ = ART / "data" / "train.parquet"


def run(name, script, extra_env):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env["V93_INPUT"] = "fwd"; env.update(extra_env)
    print(f"\n===== [{name}] {script} 開始 {time.strftime('%H:%M:%S')} =====", flush=True)
    t0 = time.time()
    log = ART / f"train_{name}.log"
    with io.open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V97 / script)], env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"===== [{name}] rc={rc} ({(time.time()-t0)/60:.1f}min) -> {log.name} =====", flush=True)
    return rc == 0


def main():
    # train.parquet 完成待ち(★サイズが安定するまで待つ=書込途中の誤検知を防ぐ)
    print(f"[wait] {PQ} 完成待ち(サイズ安定判定)...", flush=True)
    last = -1
    while True:
        if PQ.exists():
            sz = PQ.stat().st_size
            if sz == last and sz > 0:
                break            # 前回と同サイズ=書込完了
            last = sz
        time.sleep(30)
    time.sleep(10)
    # 最終確認: 実際に読めるか
    try:
        import pyarrow.parquet as _pq
        _pq.read_schema(PQ)
    except Exception as e:
        print(f"[wait] parquet まだ不完全({e})。再待機...", flush=True)
        time.sleep(60)
    print(f"[ok] train.parquet 検出 ({PQ.stat().st_size/1e6:.0f}MB)。train 開始", flush=True)

    ok_gbm = run("gbm", "train_l1_gbm_v97.py", {})
    ok_tcn = run("tcn", "train_nn_v97.py", {})

    # seqfeat CANDFEAT(4バンク: nbr_gr5/self_graft delta 1..4 を EXTRA)
    ed = ",".join([f"pf_nbr_gr5_delta_{k}" for k in range(1, 5)] + [f"pf_self_graft_delta_{k}" for k in range(1, 5)])
    es = ",".join([f"pf_nbr_gr5_std_{k}" for k in range(1, 5)] + [f"pf_self_graft_std_{k}" for k in range(1, 5)])
    ok_sf = run("seqfeat", "build_seqfeat_v97.py", dict(
        SF_IN=str(PQ), SF_OUT=str(ART / "seqfeat_v97_cf.pkl"), CANDFEAT="1", NF_MAX="2",
        EXTRA_DELTA=ed, EXTRA_STD=es))
    ok_l5 = False
    if ok_sf:
        ok_l5 = run("l5cf", "train_l456_v97.py", dict(
            ARCH="l5", L5_SF=str(ART / "seqfeat_v97_cf.pkl"), L5_OSUF="_cf", SEEDS="0,1,2"))

    # ---- blend 重み(nnls on OOF) ----
    print("\n===== blend 重み算出 =====", flush=True)
    try:
        import pyarrow.parquet as pq
        from scipy.optimize import nnls
        yv = pq.read_table(PQ, columns=["target"]).to_pandas()["target"].to_numpy(float)
        legs, names = [], []
        d1 = np.load(ART / "l1_oof.npz"); gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
        legs.append(gbm); names.append("gbm")
        tcn = np.mean([v for k, v in np.load(ART / "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
        legs.append(tcn); names.append("tcn")
        if (ART / "l5_cf_oof.npz").exists():
            l5 = np.load(ART / "l5_cf_oof.npz")["l5_mean"].astype(float); legs.append(l5); names.append("l5cf")
        M = np.stack(legs, 1); m = np.isfinite(M).all(1) & np.isfinite(yv)
        w, _ = nnls(M[m], yv[m]); w = w / w.sum()
        blend = M @ w
        rmse = float(np.sqrt(np.mean((blend[m] - yv[m]) ** 2)))
        for nm, wt, lg in zip(names, w, legs):
            r1 = float(np.sqrt(np.nanmean((lg[m] - yv[m]) ** 2)))
            print(f"  {nm}: 単体OOF={r1:.4f}  weight={wt:.4f}", flush=True)
        print(f"  ★v97 blend OOF RMSE = {rmse:.4f}", flush=True)
        json.dump(dict(blend_legs=names, blend_weights=[float(x) for x in w], cv_rmse=rmse,
                       note="v97=v94調整PF(pf_1/2/3/4)+lever(pf_1:ν5+β0.85,他:ν5)+nn_emission+smooth16"),
                  open(ART / "submit_v97_config.json", "w"), indent=1)
        print(f"  saved submit_v97_config.json", flush=True)
    except Exception as e:
        print(f"  blend 算出失敗(train個別は完了済): {e}", flush=True)

    print(f"\n===== v97 train 完了: gbm={ok_gbm} tcn={ok_tcn} seqfeat={ok_sf} l5cf={ok_l5} =====", flush=True)


if __name__ == "__main__":
    main()
