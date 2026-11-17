# -*- coding: utf-8 -*-
"""L5 新PF(7番bank, tw/self/nbr)追加実験(fwd, before/after)。gen_newpf_bank 完了後に手動起動。
   [1] add_newpf_bank -> train_plus.parquet  [2] seqfeat(fwd + extra 3ch=ancc/self/nbr_7) -> seqfeat_v93_plus.pkl
   [3] L5 before(K14) [4] L5 after(K17) [5] before/after CV比較。既存 l5_oof.npz 等は触らない。
   ログ: v93/artifacts_v93/l5exp2.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe"); LOG = ART / "l5exp2.log"
PLUS = ART / "data" / "train_plus.parquet"; SF = ART / "seqfeat_v93_plus.pkl"


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(name, script, env):
    e = dict(os.environ); e["PYTHONUNBUFFERED"] = "1"; e.update(env)
    log(f"START {name} ({env})"); t0 = time.time()
    with io.open(LOG, "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V93 / script)], cwd=str(PROJ), env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"END {name} rc={rc} ({(time.time()-t0)/60:.1f}min)"); return rc == 0


def rmse_of(p):
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    if "l5_mean" not in d.files:
        return None
    pr = d["l5_mean"].astype(np.float64); y = d["y"].astype(np.float64); m = np.isfinite(pr) & np.isfinite(y)
    return float(np.sqrt(np.mean((pr[m] - y[m]) ** 2)))


ED = "pf_ancc_delta_7,pf_self_delta_7,pf_nbr_delta_7"
ES = "pf_ancc_std_7,pf_self_std_7,pf_nbr_std_7"
log("###### L5 新PF(7番bank) 追加実験 ######")
if run("add_bank", "add_newpf_bank_v93.py", {}):
    run("seqfeat_plus", "build_seqfeat_v93.py", {"V93_INPUT": "fwd", "SF_IN": str(PLUS),
        "SF_OUT": str(SF), "EXTRA_DELTA": ED, "EXTRA_STD": ES})
    run("L5_before(K14)", "train_l456_v93.py", {"V93_INPUT": "fwd", "ARCH": "l5", "L5_OSUF": "_before"})
    run("L5_after(K17)", "train_l456_v93.py", {"V93_INPUT": "fwd", "ARCH": "l5",
        "L5_SF": str(SF), "L5_OSUF": "_after"})
    rb = rmse_of(ART / "l5_before_oof.npz"); ra = rmse_of(ART / "l5_after_oof.npz")
    log(f"\n### L5 fwd before/after (7番bank tw/self/nbr 追加) ###")
    log(f"  before(K14): {rb}")
    log(f"  after (K17): {ra}")
    if rb and ra:
        log(f"  差: {ra-rb:+.4f}  {'改善' if ra<rb else '悪化' if ra>rb else '同'}")
log("###### 完了 ######")
