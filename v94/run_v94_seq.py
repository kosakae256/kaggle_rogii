# -*- coding: utf-8 -*-
"""v94 逐次脱相関PF を【4本できるまで】自動走査。CHUNK=300。
   完了マーカー = pf_{i}_errs.pkl(全井誤差保存済み)。tune_v94 が IDX を自動判定(errs.pkl数+1)。
   PF#1(off,制約なし)→#2(on,corr≤0.7)→#3(off)→#4(on)。各stdoutは tune_pf{idx}.log。
   PF#1 は既存 optuna_v94_pf1.db(4trial保存済)を再開して継続。
   ログ: v94/artifacts_v94/v94_seq.log"""
import io, sys, os, time, subprocess, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V94 = PROJ / "v94"; ART = V94 / "artifacts_v94"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe"); SEQLOG = ART / "v94_seq.log"
TARGET = int(os.environ.get("TARGET", "4"))
CHUNK = os.environ.get("CHUNK", "300")


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    with io.open(SEQLOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def count_done():
    return len(glob.glob(str(ART / "pf_*_errs.pkl")))


log(f"###### v94 逐次走査 開始 (TARGET={TARGET}本, CHUNK={CHUNK}) ######")
while count_done() < TARGET:
    before = count_done(); idx = before + 1
    phys = "off" if idx % 2 == 1 else "on"
    log(f"=== PF#{idx} (phys={phys}) tune開始 (現在 完了{before}/{TARGET}) ===")
    env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"; env["CHUNK"] = CHUNK
    t0 = time.time()
    with io.open(ART / f"tune_pf{idx}.log", "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V94 / "tune_v94.py")], cwd=str(PROJ), env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    after = count_done()
    log(f"=== PF#{idx} tune終了 rc={rc} ({(time.time()-t0)/60:.1f}min) 完了{after}/{TARGET} ===")
    if after <= before:
        log(f"★PF#{idx} の完了マーカー(pf_{idx}_errs.pkl)が生成されず。"
            f"feasible(corr≤0.7)失敗の可能性 → 走査中断。CORR_MAX調整 or N_TRIALS増を検討。")
        break
# サマリ
log("###### 走査終了。採用PFサマリ ######")
import json
for j in range(1, TARGET + 1):
    p = ART / f"v94_pf_{j}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        log(f"  PF#{j}: phys={d['phys']} seed={d['seed']} RMSE(600井)={d['rmse']:.4f} "
            f"max_corr={d['max_corr']:.3f} corrs={d.get('corrs')}")
