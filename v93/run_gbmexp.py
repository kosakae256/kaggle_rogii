# -*- coding: utf-8 -*-
"""tabular GBM 新PF(7番bank フル47列)追加実験。before=既存 l1_oof.npz(fwd 6.4954)流用、
   after のみ学習(GBM_IN=train_plus_full, GBM_SUF=_plus)。9モデルblendで before/after比較。
   [1] add_newpf_full -> train_plus_full.parquet  [2] GBM after -> l1_oof_plus.npz  [3] 比較
   既存 l1_oof.npz/train.parquet は触らない。ログ: v93/artifacts_v93/gbmexp.log"""
import io, sys, os, time, subprocess
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); V93 = PROJ / "v93"; ART = V93 / "artifacts_v93"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe"); LOG = ART / "gbmexp.log"
PLUS = ART / "data" / "train_plus_full.parquet"


def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    with io.open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(name, script, env):
    e = dict(os.environ); e["PYTHONUNBUFFERED"] = "1"; e.update(env)
    log(f"START {name} ({env})"); t0 = time.time()
    with io.open(LOG, "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(V93 / script)], cwd=str(PROJ), env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
    log(f"END {name} rc={rc} ({(time.time()-t0)/60:.1f}min)"); return rc == 0


def blend_rmse(npz):
    if not Path(npz).exists():
        return None
    d = np.load(npz, allow_pickle=True)
    ks = [k for k in d.files if k not in ("y", "well")]
    y = d["y"].astype(np.float64); bl = np.mean([d[k].astype(np.float64) for k in ks], 0)
    m = np.isfinite(bl) & np.isfinite(y)
    return float(np.sqrt(np.mean((bl[m] - y[m]) ** 2))), ks


log("###### tabular GBM 新PF(7番bank フル47列) 追加実験 ######")
if run("add_full", "add_newpf_full_v93.py", {}):
    if run("GBM_after(+pf7)", "train_l1_gbm_v93.py",
           {"V93_INPUT": "fwd", "GBM_IN": str(PLUS), "GBM_SUF": "_plus"}):
        rb, _ = blend_rmse(ART / "l1_oof.npz")
        ra, ks = blend_rmse(ART / "l1_oof_plus.npz")
        log(f"\n### tabular GBM fwd before/after (7番bank 47列追加, 9model blend) ###")
        log(f"  before(既存fwd)   : {rb:.4f}")
        log(f"  after (+pf7 47列) : {ra:.4f}")
        log(f"  差: {ra-rb:+.4f}  {'改善' if ra<rb else '悪化' if ra>rb else '同'}")
        # 各モデル別も
        db = np.load(ART / "l1_oof.npz", allow_pickle=True); da = np.load(ART / "l1_oof_plus.npz", allow_pickle=True)
        y = db["y"].astype(np.float64)
        for k in ks:
            mb = np.isfinite(db[k]) & np.isfinite(y); ma = np.isfinite(da[k]) & np.isfinite(y)
            r_b = np.sqrt(np.mean((db[k][mb] - y[mb]) ** 2)); r_a = np.sqrt(np.mean((da[k][ma] - y[ma]) ** 2))
            log(f"    {k:12s}: {r_b:.4f} -> {r_a:.4f} ({r_a-r_b:+.4f})")
log("###### 完了 ######")
