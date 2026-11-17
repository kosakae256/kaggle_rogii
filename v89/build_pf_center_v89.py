# -*- coding: utf-8 -*-
"""v89 — struct-free な δ枠中心として tw-PF 軌跡(v88/base engine, struct錨なし)を全train坑井で dump。
出力: v89/artifacts_v89/pf_center_v89.npz  {well: (ev行, )float32 の PF TVT}
使い方: NW=0 python v89/build_pf_center_v89.py"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65"))
os.environ["V65_TWSRC"] = "tw"
import pf_engine_v65 as base
import warmup_v65 as wu

NW = int(os.environ.get("NW", "0")) or None
SEED = int(os.environ.get("SEED", "20251707")); NSEED = 32; CHUNK = 120
P = base.full_params(dict(wu.R0_PARAMS)); P["use_anchor"] = False; P["n_seeds"] = NSEED; P["tw_source"] = "tw"
wells = base.list_wells()
if NW: wells = wells[:NW]
cache = {}
for w in wells:
    c = base.build_cache(w)
    if c is not None:
        cache[w] = c
base.CACHE = cache
ALL = [w for w in wells if w in cache]
print(f"[v89] PF center dump {len(ALL)}井 struct-free", flush=True)
out = {}
for c0 in range(0, len(ALL), CHUNK):
    sub = ALL[c0:c0 + CHUNK]
    outs = base.run_forward_ens([base.make_inputs(cache[w], P) for w in sub], P, base.DEVICE, SEED, NSEED)
    for k, w in enumerate(sub):
        out[w] = np.asarray(outs[k], np.float32)
    print(f"  {min(c0+CHUNK,len(ALL))}/{len(ALL)}", flush=True)
OUTP = PROJECT / "v89" / "artifacts_v89" / "pf_center_v89.npz"
np.savez_compressed(OUTP, **out)
print(f"saved {OUTP} ({len(out)}井)")
