# -*- coding: utf-8 -*-
"""既存 v93 PF(pf_banks smooth PF)に gs×1.30(GR尤度σ広げ)を仕込み、30井で gs×1.0 vs 1.3 の
   overall RMSE を比較。GPU 軽め(n_seeds=16, 30井, 代表3バンク)。tuner 並走前提。
実行: rogii_claude/.venv/Scripts/python.exe v93/bench_gs.py"""
import os, sys, io, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v93 as pf

DATA = pf.DATA_DIR / "train"
ART = pf.ART
SEED = 4423098
N_SEEDS = int(os.environ.get("NS", "16"))
CHUNK = 30
NW = int(os.environ.get("NW", "30"))
BANKS = ["pf_1", "r0_seed32", "pfA"]

# 30 有効井をキャッシュ
wells_all = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(DATA / "*__horizontal_well.csv"))})
P0 = pf.bank_param("pf_1"); SEL = []
for w in wells_all:
    try:
        hw = pd.read_csv(DATA / f"{w}__horizontal_well.csv"); tw = pd.read_csv(DATA / f"{w}__typewell.csv").sort_values("TVT")
    except Exception:
        continue
    if "TVT" not in hw.columns:
        continue
    if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None:
        continue
    SEL.append((w, hw, tw))
    if len(SEL) >= NW:
        break
print(f"[bench gs] 井={len(SEL)} banks={BANKS} n_seeds={N_SEEDS}", flush=True)
SIMD = pickle.loads((ART / "sim_grfree_v93.pkl").read_bytes()) if (ART / "sim_grfree_v93.pkl").exists() else {}

res = {}
for gss in ["1.0", "1.3"]:
    os.environ["GS_SCALE"] = gss
    for bank in BANKS:
        P = pf.bank_param(bank); w_nn = P["_w_nn"]; phys = P["_physics"]
        inps = []; trues = []
        for (w, hw, tw) in SEL:
            tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
            x = pf.build_smoother_inputs(hw, tt, tg, P)      # ★ここで GS_SCALE が gs に効く
            x = pf.attach_anchor(x, w, phys)
            sd = SIMD.get(w)
            if sd is not None:
                x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
            inps.append(x)
            ev = hw["TVT_input"].isna().to_numpy()
            trues.append(hw["TVT"].to_numpy(float)[ev])
        outs = pf.run_smoother_ext(inps, P, SEED, N_SEEDS, CHUNK, w_nn=w_nn)
        errs = []
        for i, o in enumerate(outs):
            m = min(len(o["mean"]), len(trues[i]))
            e = np.asarray(o["mean"][:m], float) - trues[i][:m]
            errs.append(e[np.isfinite(e)])
        e = np.concatenate(errs); rmse = float(np.sqrt(np.mean(e ** 2)))
        res[(gss, bank)] = rmse
        print(f"  GS_SCALE={gss} {bank:10s} overall RMSE = {rmse:.4f}", flush=True)

print("\n=== gs×1.0 → gs×1.3 の差(30井 overall RMSE) ===")
for bank in BANKS:
    r0 = res[("1.0", bank)]; r1 = res[("1.3", bank)]
    print(f"  {bank:10s}: {r0:.4f} -> {r1:.4f}  ({r1-r0:+.4f})  {'改善' if r1<r0 else '悪化' if r1>r0 else '同'}")
