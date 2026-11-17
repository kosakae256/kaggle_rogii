# -*- coding: utf-8 -*-
"""v86 — 既存 optuna study の現best を全773井・struct-free(錨OFF)で評価。
使い方: REP=logamp python v86/eval773_v86.py  (STUDY省略時 {REP}_seeds32_0)"""
import os, io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, optuna
REP = os.environ.get("REP", "logamp")
os.environ["V86_REP"] = REP
os.environ.setdefault("V86_LOG_G0", "40.0")
os.environ["V65_TWSRC"] = "tw"
sys.path.insert(0, str(Path(__file__).parent))
import pf_engine_v65 as pf
import warmup_v65 as wu

NSEED = int(os.environ.get("EVAL_SEEDS", "32"))
CHUNK = int(os.environ.get("EVAL_CHUNK", "120"))
SEED_SEARCH = 20250707; SEED_ALT = SEED_SEARCH + 1000
STUDY = os.environ.get("STUDY", f"{REP}_seeds32_0")
STORAGE = f"sqlite:///{(pf.PROJECT / 'v86' / 'artifacts_v86' / 'optuna.db').as_posix()}"
FIXED_NEUTRAL = dict(use_phys=False, phys_sig=1.0, use_anchor=False, anchor_mult=20.0,
                     anchor_ramp_md=0.0, lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2)

st = optuna.load_study(study_name=STUDY, storage=STORAGE)
done = sum(1 for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE)
bp = {**pf.full_params(st.best_params), **FIXED_NEUTRAL}
print(f"study={STUDY}  done={done}  best_value(探索seed,in-sample120)={st.best_value:.4f}  use_anchor={bp['use_anchor']}")

wells = pf.list_wells()
cache = {}
for w in wells:
    c = pf.build_cache(w)
    if c is not None and c["tvt_true"] is not None:
        cache[w] = c
pf.CACHE = cache
ALL = [w for w in wells if w in cache]
print(f"usable wells: {len(ALL)}  REP={REP}", flush=True)


def rep_eval(tag, seed):
    e = pf.predict_errs(bp, ALL, seed, NSEED, CHUNK)
    ov = np.concatenate([e[w] for w in ALL]); ov = ov[np.isfinite(ov)]
    o = float(np.sqrt(np.mean(ov ** 2)))
    print(f"  [{tag}] overall(773)={o:.4f}", flush=True)
    return o

print(f"\n=== v86 {STUDY} 現best 全{len(ALL)}井 struct-free 評価 ===")
o_s = rep_eval(f"seed{SEED_SEARCH}(探索)", SEED_SEARCH)
o_a = rep_eval(f"seed{SEED_ALT}(別seed)", SEED_ALT)
print(f"\n参考: 素通しlogamp(r0流用,773)=8.68 / raw r0(773)=8.72")
