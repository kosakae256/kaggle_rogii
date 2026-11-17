# -*- coding: utf-8 -*-
"""逆方向(backward)実験の自動オーケストレータ(夜間放置用)。
   Round1 = fwdbwd(warm=forward終端init): GBM -> seqfeat -> TCN -> L5
   Round2 = fwdbwd_anchor(GRフリー錨終端init): build_backward(anchor) -> GBM -> seqfeat -> TCN -> L5
   最後に make_bwd_doc.py で docs/v93_backward.md(fold別含む)を生成。
   各段は失敗しても継続(部分結果でも doc は出す)。ログ: v93/artifacts_v93/pipeline.log
   実行: nohup 相当でバックグラウンド起動。"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import pyarrow.parquet as pq

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"
ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LOG = ART / "pipeline.log"
WARM_PQ = ART / "data" / "train_fwdbwd.parquet"
ANCH_PQ = ART / "data" / "train_fwdbwd_anchor.parquet"


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def bb_running():
    """build_backward_v93.py が走っているか(PowerShell Get-CimInstance。wmic は Win11 で撤去)。"""
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Select-Object -ExpandProperty CommandLine"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
        return "build_backward_v93.py" in (out or "")
    except Exception:
        return False


def has_fullmirror(pqpath):
    try:
        names = set(pq.read_schema(pqpath).names)
    except Exception:
        return False
    return "bpf_arrival_err" in names   # full-mirror の目印


def run_stage(name, script, env_over, extra_args=None):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env.update(env_over)
    cmd = [PY, str(V93 / script)] + (extra_args or [])
    log(f"=== START {name}  ({' '.join(f'{k}={v}' for k,v in env_over.items())}) ===")
    t0 = time.time()
    try:
        with io.open(LOG, "a", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=str(PROJ), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    except Exception as e:
        log(f"### EXCEPTION {name}: {e}")
        return False
    dt = time.time() - t0
    ok = (rc == 0)
    log(f"=== END {name} rc={rc} {'OK' if ok else 'FAIL'} ({dt/60:.1f}min) ===")
    return ok


def wait_for_warm():
    log("Step0: 実行中の warm build_backward の完了を待つ(kill しない)...")
    last_size = -1; stable_since = None
    while True:
        running = bb_running()
        if WARM_PQ.exists():
            sz = WARM_PQ.stat().st_size
            if sz != last_size:
                last_size = sz; stable_since = None
            elif stable_since is None:
                stable_since = time.time()
            stable = (stable_since is not None) and (time.time() - stable_since > 120)
            if stable and not running:
                log(f"  warm 完了: train_fwdbwd.parquet {sz/1e9:.2f}GB, プロセス無し")
                return
        elif not running:
            log("  build_backward プロセスも train_fwdbwd.parquet も無い -> 自前で warm 生成")
            run_stage("build_backward(warm)", "build_backward_v93.py", {"BWD_INIT": "warm"})
            return
        time.sleep(60)


def ensure_fullmirror_warm():
    if not WARM_PQ.exists() or not has_fullmirror(WARM_PQ):
        log("Step0.5: train_fwdbwd.parquet が full-mirror でない/無い -> warm を現行コードで再生成")
        run_stage("build_backward(warm再生成)", "build_backward_v93.py", {"BWD_INIT": "warm"})
    else:
        try:
            ncol = len(pq.read_schema(WARM_PQ).names)
        except Exception:
            ncol = -1
        log(f"Step0.5: train_fwdbwd.parquet は full-mirror(bpf_arrival_err 有り, 総列={ncol}) OK")


def round_train(vinput):
    """GBM -> seqfeat -> TCN -> L5 を vinput(fwdbwd / fwdbwd_anchor)で回す。"""
    run_stage(f"GBM[{vinput}]", "train_l1_gbm_v93.py", {"V93_INPUT": vinput})
    run_stage(f"seqfeat[{vinput}]", "build_seqfeat_v93.py", {"V93_INPUT": vinput})
    run_stage(f"TCN[{vinput}]", "train_nn_v93.py", {"V93_INPUT": vinput})
    run_stage(f"L5[{vinput}]", "train_l456_v93.py", {"V93_INPUT": vinput, "ARCH": "l5"})


def main():
    log("################ backward pipeline 開始 ################")
    # ---- Round1: warm(forward終端 init) ----
    wait_for_warm()
    ensure_fullmirror_warm()
    log(">>> Round1: fwdbwd(warm) 学習開始")
    round_train("fwdbwd")
    run_stage("doc(中間)", "make_bwd_doc.py", {})   # Round1 時点の中間 doc

    # ---- Round2: anchor(GRフリー錨終端 init) ----
    log(">>> Round2: anchor 生成 + 学習開始")
    if not (ANCH_PQ.exists() and has_fullmirror(ANCH_PQ)):
        run_stage("build_backward(anchor)", "build_backward_v93.py", {"BWD_INIT": "anchor"})
    else:
        log("  train_fwdbwd_anchor.parquet 既存(full-mirror) -> 生成スキップ")
    round_train("fwdbwd_anchor")

    # ---- 最終 doc ----
    run_stage("doc(最終)", "make_bwd_doc.py", {})
    log("################ backward pipeline 完了 ################")


if __name__ == "__main__":
    main()
