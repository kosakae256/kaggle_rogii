# -*- coding: utf-8 -*-
"""v89 — p_1..p_5(v69_4の5本, struct-free)のPF軌跡をダンプ。U-Netのガイドストライプ用。
出力: v89/artifacts_v89/pf_banks_v89.npz  {f"{well}__p{k}": ev行 PF TVT}
使い方: NW=0 python v89/build_pf_banks_v89.py"""
import os, io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65")); sys.path.insert(0, str(PROJECT / "v69"))
os.environ["V65_TWSRC"] = "tw"
import pf_engine_v65 as base
import create_candidates_v69_4 as v69   # PF_LIB(p_1..p_5)を借りる

NW = int(os.environ.get("NW", "0")) or None
SEED = 20251707; NSEED = 32; CHUNK = 120
BANKS = ["p_1", "p_2", "p_3", "p_4", "p_5"]
wells = base.list_wells()
if NW: wells = wells[:NW]
cache = {}
for w in wells:
    c = base.build_cache(w)
    if c is not None:
        cache[w] = c
base.CACHE = cache
ALL = [w for w in wells if w in cache]
print(f"[v89] PF banks dump {len(ALL)}井 × {len(BANKS)}本 struct-free", flush=True)
out = {}
for lab in BANKS:
    P = base.full_params(dict(v69.PF_LIB[lab])); P["use_anchor"] = False; P["n_seeds"] = NSEED; P["tw_source"] = "tw"
    for c0 in range(0, len(ALL), CHUNK):
        sub = ALL[c0:c0 + CHUNK]
        outs = base.run_forward_ens([base.make_inputs(cache[w], P) for w in sub], P, base.DEVICE, SEED, NSEED)
        for k, w in enumerate(sub):
            out[f"{w}__{lab}"] = np.asarray(outs[k], np.float32)
    print(f"  {lab} done", flush=True)
OUTP = PROJECT / "v89" / "artifacts_v89" / "pf_banks_v89.npz"
np.savez_compressed(OUTP, **out)
print(f"saved {OUTP} ({len(out)}エントリ)")
