# -*- coding: utf-8 -*-
"""夜間オーケストレータ2: anchor 学習(L5まで)完了を待って
   [1] anchor の submit(GBM+TCN+L5)を作成 -> submission_fwdbwd_anchor_v93.csv
   [2] tune_decorr(脱相関PFチューナ, N_SEEDS=32既定)を実行
   の順で回す。各段失敗しても継続。ログ: v93/artifacts_v93/pipeline_overnight2.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
LOG = ART / "pipeline_overnight2.log"
# anchor 学習完了の目印(3脚 OOF が揃う)
ANCHOR_DONE = [ART / "l1_oof_fwdbwd_anchor.npz",
               ART / "nn_oof_combo_fwdbwd_anchor.npz",
               ART / "l5_fwdbwd_anchor_oof.npz"]


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def proc_running(name):
    """python プロセスの CommandLine に name を含むものが走っているか(PowerShell)。"""
    try:
        cmd = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
               "Select-Object -ExpandProperty CommandLine"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
        return name in (out or "")
    except Exception:
        return False


def run_stage(name, args, env_over):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env.update(env_over)
    log(f"=== START {name} ({' '.join(f'{k}={v}' for k,v in env_over.items())}) ===")
    t0 = time.time()
    try:
        with io.open(LOG, "a", encoding="utf-8") as fh:
            rc = subprocess.run([PY] + args, cwd=str(PROJ), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
    except Exception as e:
        log(f"### EXCEPTION {name}: {e}"); return False
    log(f"=== END {name} rc={rc} {'OK' if rc==0 else 'FAIL'} ({(time.time()-t0)/60:.1f}min) ===")
    return rc == 0


def wait_anchor():
    log("Step0: anchor 学習(run_bwd_anchor_fix.py)の完了待ち...")
    while True:
        running = proc_running("run_bwd_anchor_fix.py")
        done = all(p.exists() for p in ANCHOR_DONE)
        if done and not running:
            log(f"  anchor 完了(3脚OOF揃い・プロセス無し)"); return True
        if not running:
            # プロセスが無いのに OOF が揃っていない = どこかで失敗して終了
            miss = [p.name for p in ANCHOR_DONE if not p.exists()]
            log(f"  anchor プロセス無し & OOF不足{miss} -> 学習は失敗/未完のまま先へ(submit は可能な範囲で)")
            return False
        time.sleep(90)


def main():
    log("############ overnight2 開始 ############")
    ok_anchor = wait_anchor()
    # [1] anchor submit(3脚モデルが揃っていれば)
    have_models = ((ART / "models_l1_fwdbwd_anchor").exists()
                   and (ART / "submit_pack_nn_fwdbwd_anchor.json").exists()
                   and (ART / "l5_fwdbwd_anchor_ckpt").exists())
    if have_models:
        run_stage("submit[anchor]", [str(V93 / "submit_fwdbwd_v93.py")],
                  {"VARIANT": "anchor", "SKIP_GEN": "1"})   # forward test は既存流用, backward(anchor)は生成
    else:
        log("### anchor モデル不足 -> submit スキップ")
    # [2] tune_decorr を 物理prior なし(off)/あり(on) で別々に(N_SEEDS は tune_decorr 既定=32)
    run_stage("tune_decorr[PHYS=off]", [str(V93 / "tune_decorr_v93.py")], {"PHYS": "off"})
    run_stage("tune_decorr[PHYS=on]", [str(V93 / "tune_decorr_v93.py")], {"PHYS": "on"})
    log("############ overnight2 完了 ############")


if __name__ == "__main__":
    main()
