# -*- coding: utf-8 -*-
"""Round2(anchor)リカバリ・オーケストレータ。
   disk full で落ちた anchor を、既存 backward_feats_anchor.parquet(PF計算済3GB)から再開する。
   手順: [1] Round1完了済みの warm parquet + backward_feats.parquet を削除して空き確保
         [2] merge_anchor.py で train_fwdbwd_anchor.parquet を生成
         [3] backward_feats_anchor.parquet を削除して更に空き確保
         [4] GBM -> seqfeat -> TCN -> L5 (V93_INPUT=fwdbwd_anchor)
         [5] make_bwd_doc.py で最終 doc
   各段失敗しても継続。ログ: v93/artifacts_v93/pipeline_r2.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LOG = ART / "pipeline_r2.log"
ANCH_PQ = ART / "data" / "train_fwdbwd_anchor.parquet"
DEL_FOR_SPACE = [
    ART / "data" / "train_fwdbwd.parquet",   # Round1 完了済(OOF 保存済) -> 削除可
    ART / "backward_feats.parquet",          # warm 標準bpf(冗長) -> 削除可
]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_bytes():
    import shutil
    return shutil.disk_usage(str(ART)).free


def run_stage(name, script, env_over):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env.update(env_over)
    log(f"=== START {name} ({' '.join(f'{k}={v}' for k,v in env_over.items())}) 空き={free_bytes()/1e9:.1f}GB ===")
    t0 = time.time()
    try:
        with io.open(LOG, "a", encoding="utf-8") as fh:
            rc = subprocess.run([PY, str(V93 / script)], cwd=str(PROJ), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
    except Exception as e:
        log(f"### EXCEPTION {name}: {e}"); return False
    ok = (rc == 0)
    log(f"=== END {name} rc={rc} {'OK' if ok else 'FAIL'} ({(time.time()-t0)/60:.1f}min) ===")
    return ok


def main():
    log("############ Round2(anchor) リカバリ開始 ############")
    # [1] 空き確保(Round1完了済みの大物を削除)
    for p in DEL_FOR_SPACE:
        if p.exists():
            sz = p.stat().st_size
            try:
                p.unlink(); log(f"削除: {p.name} ({sz/1e9:.1f}GB) 空き={free_bytes()/1e9:.1f}GB")
            except Exception as e:
                log(f"削除失敗 {p.name}: {e}")
    # [2] anchor を build_backward から再生成(壊れた standalone は復旧不可のため PF から。
    #     SKIP_STANDALONE=1 で 3GB の standalone を書かず merged(11GB)のみ = disk 二重書き回避)
    if not ANCH_PQ.exists():
        ok = run_stage("build_backward(anchor)", "build_backward_v93.py",
                       {"BWD_INIT": "anchor", "SKIP_STANDALONE": "1"})
        if not ok or not ANCH_PQ.exists():
            log("### build_backward(anchor) 失敗 -> 中止(最終doc だけ出す)")
            run_stage("doc(最終)", "make_bwd_doc.py", {}); return
    else:
        log("train_fwdbwd_anchor.parquet 既存 -> 生成skip")
    # [4] 学習
    run_stage("GBM[anchor]", "train_l1_gbm_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("seqfeat[anchor]", "build_seqfeat_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("TCN[anchor]", "train_nn_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("L5[anchor]", "train_l456_v93.py", {"V93_INPUT": "fwdbwd_anchor", "ARCH": "l5"})
    # [5] 最終 doc
    run_stage("doc(最終)", "make_bwd_doc.py", {})
    log("############ Round2(anchor) 完了 ############")


if __name__ == "__main__":
    main()
