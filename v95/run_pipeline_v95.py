# -*- coding: utf-8 -*-
"""v95 学習チェーン(自律実行用): GBM -> build_seqfeat -> TCN -> L5 -> nnls 3脚 blend。
各ステップ: subprocess 実行 -> 出力ファイル確認 -> 失敗で停止(ログ)。forward-only(V93_INPUT=fwd)。
出力: 各 OOF(l1_oof.npz / nn_oof_combo.npz / l5_oof.npz) + blend_pack_v95.json(nnls重み+CV) + pipeline_v95.log
実行: ./.venv/Scripts/python.exe v95/run_pipeline_v95.py            (フル)
      SMOKE_FIRST=1 ... TCN/L5 を SMOKE=1 で先に通してから本番
"""
import os, sys, io, json, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V95 = PROJ / "v95"; ART = V95 / "artifacts_v95"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LOG = ART / "pipeline_v95.log"
SMOKE_FIRST = os.environ.get("SMOKE_FIRST", "0") == "1"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(tag, script, env_extra, expect, smoke=False):
    env = dict(os.environ); env["V93_INPUT"] = "fwd"; env.update(env_extra)
    if smoke:
        env["SMOKE"] = "1"
    log(f"START {tag}  ({script}  env={env_extra}{' SMOKE' if smoke else ''})")
    t0 = time.perf_counter()
    r = subprocess.run([PY, str(V95 / script)], env=env)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        log(f"FAIL {tag} rc={r.returncode} ({dt:.0f}s)"); raise SystemExit(f"{tag} failed")
    if expect and not Path(expect).exists():
        log(f"FAIL {tag}: 出力 {expect} が無い"); raise SystemExit(f"{tag} no output")
    log(f"OK {tag} ({dt:.0f}s)" + (f" -> {Path(expect).name}" if expect else ""))


def main():
    log("==== v95 pipeline start ====")
    pq = ART / "data" / "train.parquet"
    if not pq.exists():
        raise SystemExit(f"train.parquet が無い: {pq}")
    # 1) GBM
    run("GBM", "train_l1_gbm_v95.py", {}, ART / "l1_oof.npz")
    # 2) seqfeat
    run("SEQFEAT", "build_seqfeat_v95.py", {}, ART / "seqfeat_v95.pkl")
    # 3) TCN (先にSMOKEで疎通確認 -> 本番)
    if SMOKE_FIRST:
        run("TCN-smoke", "train_nn_v95.py", {}, None, smoke=True)
    run("TCN", "train_nn_v95.py", {}, ART / "nn_oof_combo.npz")
    # 4) L5
    if SMOKE_FIRST:
        run("L5-smoke", "train_l456_v95.py", {"ARCH": "l5"}, None, smoke=True)
    run("L5", "train_l456_v95.py", {"ARCH": "l5"}, ART / "l5_oof.npz")
    # 5) nnls 3脚 blend
    log("==== nnls 3脚 blend ====")
    blend_report()
    log("==== v95 pipeline done ====")


def _pooled_rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def blend_report():
    from scipy.optimize import nnls
    d1 = np.load(ART / "l1_oof.npz"); y = d1["y"].astype(np.float64)
    var = [k for k in d1.files if k not in ("y", "well")]
    gbm = np.mean([d1[k] for k in var], axis=0).astype(np.float64)     # 9variant 平均
    dn = np.load(ART / "nn_oof_combo.npz")
    tkeys = [k for k in dn.files if k.startswith("mdn2")]
    tcn = np.mean([dn[k] for k in tkeys], axis=0).astype(np.float64)   # seed 平均
    dl = np.load(ART / "l5_oof.npz")
    l5 = (dl["l5_mean"] if "l5_mean" in dl.files else dl[[k for k in dl.files if k.endswith("_mean")][0]]).astype(np.float64)
    n = min(len(gbm), len(tcn), len(l5), len(y))
    gbm, tcn, l5, y = gbm[:n], tcn[:n], l5[:n], y[:n]
    for nm, p in [("GBM", gbm), ("TCN", tcn), ("L5", l5)]:
        log(f"  {nm} OOF rmse = {_pooled_rmse(p, y):.4f}")
    log(f"  等重blend rmse = {_pooled_rmse((gbm+tcn+l5)/3, y):.4f}")
    M = np.stack([gbm, tcn, l5], 1); m = np.isfinite(M).all(1) & np.isfinite(y)
    w, _ = nnls(M[m], y[m]); s = w.sum(); w = w / s if s > 0 else np.array([1/3, 1/3, 1/3])
    bl = M @ w
    log(f"  ★nnls blend rmse = {_pooled_rmse(bl, y):.4f}  重み GBM/TCN/L5 = {tuple(round(float(x),3) for x in w)}")
    (ART / "blend_pack_v95.json").write_text(json.dumps(dict(
        weights=dict(gbm=float(w[0]), tcn=float(w[1]), l5=float(w[2])),
        rmse=dict(gbm=_pooled_rmse(gbm, y), tcn=_pooled_rmse(tcn, y), l5=_pooled_rmse(l5, y),
                  equal=_pooled_rmse((gbm+tcn+l5)/3, y), nnls=_pooled_rmse(bl, y))), indent=1), encoding="utf-8")
    log(f"  saved blend_pack_v95.json")


if __name__ == "__main__":
    main()
