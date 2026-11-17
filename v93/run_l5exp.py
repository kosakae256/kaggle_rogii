# -*- coding: utf-8 -*-
"""L5 新PF追加実験(fwd, before/after)を自動実行。gen_newpf 完了(newpf_train.npz)を待って:
   [1] add_newpf -> train_plus1pf.parquet(上書きなし)
   [2] build_seqfeat(fwd + extra=newpf) -> seqfeat_v93_plus.pkl
   [3] L5 before(seqfeat_v93.pkl K14) -> l5_before_oof.npz
   [4] L5 after (seqfeat_v93_plus.pkl K15) -> l5_after_oof.npz
   [5] before/after の CV(overall RMSE)を比較して出力。既存 l5_oof.npz は触らない。
   ログ: v93/artifacts_v93/l5exp.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe"); LOG = ART / "l5exp.log"
NPZ = ART / "newpf_train.npz"; PLUS_PQ = ART / "data" / "train_plus1pf.parquet"
SF_PLUS = ART / "seqfeat_v93_plus.pkl"


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(name, script, env_over):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env.update(env_over)
    log(f"=== START {name} ({' '.join(f'{k}={v}' for k,v in env_over.items())}) ===")
    t0 = time.time()
    with io.open(LOG, "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V93 / script)], cwd=str(PROJ), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"=== END {name} rc={rc} {'OK' if rc==0 else 'FAIL'} ({(time.time()-t0)/60:.1f}min) ===")
    return rc == 0


def rmse_of(npz):
    if not npz.exists():
        return None
    d = np.load(npz, allow_pickle=True)
    if "l5_mean" not in d.files or "y" not in d.files:
        return None
    p = d["l5_mean"].astype(np.float64); y = d["y"].astype(np.float64)
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    log("############ L5 新PF追加実験(fwd) 開始 ############")
    log("gen_newpf(newpf_train.npz)の完了待ち...")
    while not NPZ.exists():
        time.sleep(30)
    time.sleep(10)  # 書き込み完了余裕
    if not run("add_newpf", "add_newpf_v93.py", {}):
        log("### add_newpf 失敗 -> 中止"); return
    run("seqfeat_plus", "build_seqfeat_v93.py",
        {"V93_INPUT": "fwd", "SF_IN": str(PLUS_PQ), "SF_OUT": str(SF_PLUS),
         "EXTRA_DELTA": "newpf_delta", "EXTRA_STD": "newpf_std"})
    run("L5_before(K14)", "train_l456_v93.py", {"V93_INPUT": "fwd", "ARCH": "l5", "L5_OSUF": "_before"})
    run("L5_after(K15,+newpf)", "train_l456_v93.py",
        {"V93_INPUT": "fwd", "ARCH": "l5", "L5_SF": str(SF_PLUS), "L5_OSUF": "_after"})
    rb = rmse_of(ART / "l5_before_oof.npz"); ra = rmse_of(ART / "l5_after_oof.npz")
    log("\n########## L5 fwd 新PF追加 before/after ##########")
    log(f"  before(K14, 既存14ch)      : {rb}")
    log(f"  after (K15, +新脱相関PF)   : {ra}")
    if rb is not None and ra is not None:
        log(f"  差: {ra-rb:+.4f}  {'改善' if ra<rb else '悪化' if ra>rb else '同'}")
    log("############ 完了 ############")


if __name__ == "__main__":
    main()
