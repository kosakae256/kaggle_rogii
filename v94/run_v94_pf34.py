# -*- coding: utf-8 -*-
"""PF#3(warmup=9.82版trial78, 150trial, CORR_MAX=0.8) → PF#4(150trial, 0.8) を逐次実行。
   ★PF#3は汚染db退避後の新studyで、9.82を初期値にTPE探索(0.8帯を素直に探る)。
   各stdout=tune_pf{idx}.log。ログ: v94/artifacts_v94/v94_pf34.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V94 = PROJ / "v94"; ART = V94 / "artifacts_v94"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe"); LOG = ART / "v94_pf34.log"


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run_pf(idx, warmup=None, n_trials=150):
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
    env["IDX"] = str(idx); env["N_TRIALS"] = str(n_trials); env["CORR_MAX"] = "0.8"; env["CHUNK"] = "300"
    if warmup:
        env["WARMUP_JSON"] = warmup
    else:
        env.pop("WARMUP_JSON", None)
    phys = "off" if idx % 2 == 1 else "on"
    log(f"=== PF#{idx} (phys={phys}, {n_trials}trial, 0.8{', warmup' if warmup else ''}) 開始 ===")
    t0 = time.time()
    with io.open(ART / f"tune_pf{idx}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V94 / "tune_v94.py")], cwd=str(PROJ), env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    done = (ART / f"pf_{idx}_errs.pkl").exists()
    log(f"=== PF#{idx} 終了 rc={rc} ({(time.time()-t0)/60:.1f}min) errs.pkl={'保存' if done else '★無し'} ===")
    return done


log("###### PF#3(warmup=現best9.70,50trial)→PF#4(150trial) 走査 [目的=corr≤0.8下でRMSE最小] ######")
if run_pf(3, warmup=str(ART / "warmup_pf3_cur.json"), n_trials=50):
    run_pf(4, warmup=None, n_trials=150)
log("###### 完了。採用サマリ ######")
import json
for j in (1, 2, 3, 4):
    p = ART / f"v94_pf_{j}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        log(f"  PF#{j}: phys={d['phys']} RMSE={d['rmse']:.4f} max_corr={d['max_corr']:.3f} corrs={d.get('corrs')}")
