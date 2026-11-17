# -*- coding: utf-8 -*-
"""anchor init バグ修正後の再走。
   バグ: BWD_INIT=anchor が x["anc"](physics=False バンクでゼロ化)を使い、pf_1..r1 の
        anchor init が z[-1](TVT≈0相当)= 桁違いに壊れていた(760ft ドリフト)。
   修正: pf._ANCHOR から錨tvt を直接読む(全バンク正しい)。
   本スクリプト: garbage な fwdbwd_anchor 成果物を全削除(RESUME事故防止)-> 修正版で anchor 一式再走。
   ※ current run(bt6rtzhbk)が完全終了してから起動すること(parquet/GPU 競合回避)。
   ログ: v93/artifacts_v93/pipeline_anchorfix.log"""
import io, sys, os, time, glob, shutil, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LOG = ART / "pipeline_anchorfix.log"
ANCH_PQ = ART / "data" / "train_fwdbwd_anchor.parquet"

DEL_FILES = [
    ANCH_PQ,
    ART / "l1_oof_fwdbwd_anchor.npz", ART / "features_l1_fwdbwd_anchor.json",
    ART / "feat_rank_nn_fwdbwd_anchor.json", ART / "seqfeat_v93_fwdbwd_anchor.pkl",
    ART / "nn_oof_combo_top192_fwdbwd_anchor.npz", ART / "nn_oof_combo_fwdbwd_anchor.npz",
    ART / "results_nn_top192_fwdbwd_anchor.json", ART / "submit_pack_nn_top192_fwdbwd_anchor.json",
    ART / "results_nn_fwdbwd_anchor.json", ART / "submit_pack_nn_fwdbwd_anchor.json",
    ART / "l5_fwdbwd_anchor_oof.npz",
]
DEL_DIRS = [ART / "models_l1_fwdbwd_anchor", ART / "l5_fwdbwd_anchor_ckpt"]
DEL_GLOBS = [str(ART / "nn" / "combo_top192_fwdbwd_anchor_*.pt"),
             str(ART / "nn" / "combo_fwdbwd_anchor_*.pt")]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def free_gb():
    return shutil.disk_usage(str(ART)).free / 1e9


def run_stage(name, script, env_over):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env.update(env_over)
    log(f"=== START {name} ({' '.join(f'{k}={v}' for k,v in env_over.items())}) 空き={free_gb():.1f}GB ===")
    t0 = time.time()
    try:
        with io.open(LOG, "a", encoding="utf-8") as fh:
            rc = subprocess.run([PY, str(V93 / script)], cwd=str(PROJ), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
    except Exception as e:
        log(f"### EXCEPTION {name}: {e}"); return False
    log(f"=== END {name} rc={rc} {'OK' if rc==0 else 'FAIL'} ({(time.time()-t0)/60:.1f}min) ===")
    return rc == 0


def cleanup():
    log("garbage anchor 成果物を削除(RESUME事故防止)...")
    for p in DEL_FILES:
        if p.exists():
            sz = p.stat().st_size
            try: p.unlink(); log(f"  del {p.name} ({sz/1e9:.2f}GB)")
            except Exception as e: log(f"  del失敗 {p.name}: {e}")
    for d in DEL_DIRS:
        if d.exists():
            try: shutil.rmtree(d); log(f"  del dir {d.name}/")
            except Exception as e: log(f"  del dir失敗 {d.name}: {e}")
    for g in DEL_GLOBS:
        for fp in glob.glob(g):
            try: os.remove(fp)
            except Exception: pass
        log(f"  del glob {g}")
    log(f"cleanup 完了 空き={free_gb():.1f}GB")


def main():
    log("########### anchor 修正版 再走 開始 ###########")
    cleanup()
    ok = run_stage("build_backward(anchor,fixed)", "build_backward_v93.py",
                   {"BWD_INIT": "anchor", "SKIP_STANDALONE": "1"})
    if not ok or not ANCH_PQ.exists():
        log("### build_backward(anchor) 失敗 -> doc だけ出す"); run_stage("doc", "make_bwd_doc.py", {}); return
    run_stage("GBM[anchor]", "train_l1_gbm_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("seqfeat[anchor]", "build_seqfeat_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("TCN[anchor]", "train_nn_v93.py", {"V93_INPUT": "fwdbwd_anchor"})
    run_stage("L5[anchor]", "train_l456_v93.py", {"V93_INPUT": "fwdbwd_anchor", "ARCH": "l5"})
    run_stage("doc(最終)", "make_bwd_doc.py", {})
    log("########### anchor 修正版 再走 完了 ###########")


if __name__ == "__main__":
    main()
