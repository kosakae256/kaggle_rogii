# -*- coding: utf-8 -*-
"""v83 ランナー: v82 NN完了待ち → L5×5fold(seed0) → OOF評価 → v79 GBM/NN OOFとのstack。"""
import time, subprocess, re, os, io, sys, pickle, glob, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V83 = P0 / "v83"; ART = V83 / "artifacts_v83"
PY = r"C:\Users\kosaka256\Documents\rogii_claude\.venv\Scripts\python.exe"
LOG = ART / "night_v83.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")
    print(m, flush=True)


def main():
    log(f"\n[v83] 起動 {time.strftime('%Y-%m-%d %H:%M:%S')} v82NN完了待ち")
    nnlog = P0 / "v82" / "night_v82_nn.log"
    while True:
        t = nnlog.read_text(encoding="utf-8") if nnlog.exists() else ""
        if "done nn_a=" in t: break
        time.sleep(300)
    log("[v83] v82完了検知 → L5学習開始")
    env0 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for f in range(5):
        env = {**env0, "ARCH": "l5", "FOLD": str(f), "SEED": "0", "MAXMIN": "45"}
        t0 = time.time()
        r = subprocess.run([PY, str(V83 / "train_l5_v83.py")], capture_output=True,
                           encoding="utf-8", errors="replace", env=env)
        out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
        (ART / f"train_f{f}_stdout.log").write_text(out, encoding="utf-8")
        tail = [l for l in out.splitlines() if "RMSE" in l or "stop" in l]
        log(f"[v83] fold{f} exit={r.returncode} ({(time.time()-t0)/60:.1f}min) " + " | ".join(tail[-2:]))
        if r.returncode != 0:
            log("[v83] ★fold失敗。中断"); return
    # OOF合成
    SF = pickle.loads((ART / "seqfeat_v83.pkl").read_bytes())
    oof = {}
    for f in range(5):
        oof.update(pickle.loads((ART / "nn83" / f"oof_v83_l5_f{f}_s0.pkl").read_bytes()))
    es = []
    for w, dr in oof.items():
        er = dr.astype(np.float64) - SF[w]["dtrue_rows"].astype(np.float64)
        es.append(er[np.isfinite(er)])
    e_l5 = np.concatenate(es)
    log(f"[v83] ★L5-K16 OOF 行RMSE = {np.sqrt(np.mean(e_l5**2)):.4f} (v62 L5=5.932/v79 GBM=6.394/NN=6.323)")
    # stack: v79 GBM OOF + L5 (誤差空間, Σw=1閉形式)
    import pyarrow.parquet as pq
    tb = pq.read_table(P0 / "v79" / "artifacts_v79" / "data" / "train.parquet",
                       columns=["well", "target", "last_known_tvt"])
    wl = tb["well"].to_numpy(zero_copy_only=False)
    tgt = tb["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    gbm = np.load(P0 / "v79" / "artifacts_v79" / "gbm_v41_oof.npy").astype(np.float64)
    nn_fp = sorted(glob.glob(str(P0 / "v79" / "artifacts_v79" / "nn" / "*oof*.npy")))
    nn = np.load(nn_fp[0]).astype(np.float64) if nn_fp else None
    log(f"[v83] gbm oof rows={len(gbm)} nn={'あり' if nn is not None else 'なし'}({nn_fp[:1]})")
    chg = np.flatnonzero(wl[1:] != wl[:-1]) + 1
    st_ = np.concatenate([[0], chg]); en_ = np.concatenate([chg, [len(wl)]])
    Eg, El, En = [], [], []
    for s0, e0 in zip(st_, en_):
        w = str(wl[s0])
        if w not in oof or w not in SF: continue
        T = min(e0 - s0, SF[w]["T"])
        eg = gbm[s0:s0 + T] - tgt[s0:s0 + T]
        el = (oof[w][:T].astype(np.float64) - SF[w]["dtrue_rows"][:T].astype(np.float64))
        m = np.isfinite(eg) & np.isfinite(el)
        if nn is not None and len(nn) == len(gbm):
            en2 = nn[s0:s0 + T] - tgt[s0:s0 + T]
            m &= np.isfinite(en2); En.append(en2[m])
        Eg.append(eg[m]); El.append(el[m])
    E = [np.concatenate(Eg), np.concatenate(El)] + ([np.concatenate(En)] if En else [])
    names = ["GBM", "L5"] + (["NN"] if En else [])
    Em = np.stack(E)
    G = Em @ Em.T / Em.shape[1]
    import itertools as it
    for k in range(2, len(names) + 1):
        for idx in it.combinations(range(len(names)), k):
            Gs = G[np.ix_(idx, idx)]; n = len(idx)
            A = np.zeros((n + 1, n + 1)); A[:n, :n] = Gs; A[:n, n] = 1; A[n, :n] = 1
            b = np.zeros(n + 1); b[n] = 1
            wv = np.linalg.solve(A, b)[:n]
            r = float(np.sqrt(wv @ Gs @ wv))
            log(f"[v83] stack {[names[i] for i in idx]}: {r:.4f} w={[round(float(x),2) for x in wv]}")
    log(f"[v83] 全完了 {time.strftime('%Y-%m-%d %H:%M:%S')} (v79 combo基準 6.2420)")


if __name__ == "__main__":
    main()
