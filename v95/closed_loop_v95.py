# -*- coding: utf-8 -*-
"""v95 closed-loop (pseudo-satellite)。v30 の思想を v95 の強い blend(5.987)で。
v95 blend(nnls gbm,tcn,l5)を per-row prior として PF の anchor(anc/ancs)へ注入し、
緩い錨σでPFにGR整合+dip連続の局所補正をさせる。出力 cl は「精度向上」でなく
★新特徴(cl_res=cl−blend / cl_spread=seed std / cl_vs_pf)の生成が目的(v30 r1)。

engine再利用: attach_anchor の代わりに x['anc']=blend_tvt, x['ancs']=σ を手動セット、
P['anchor_mult']=1 で asig=σ。log_lik=GRのみ(v30設計=seed加重にprior混ぜない)は engine既定。
出力: v95/artifacts_v95b/closed_loop_v95.npz(cl_{σ}/spread_{σ}) + closed_loop_v95.txt
実行: .venv/Scripts/python.exe v95/closed_loop_v95.py
"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from scipy.optimize import nnls
sys.path.insert(0, str(Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v95"))
import pf_banks_v95 as pf

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = ART / "data" / "train.parquet"
DATA = pf.DATA_DIR / "train"
SEED = 4423098; N_SEED = pf.N_SEED; WELL_CHUNK = 60
BANK = os.environ.get("CL_BANK", "r0_seed32")
SIGS = [float(s) for s in os.environ.get("CL_SIGS", "192,96,64").split(",")]

tb = pq.read_table(PQ, columns=["well", "target", "last_known_tvt", "md_since"]).to_pandas()
g = tb["well"].astype(str).to_numpy(); y = tb["target"].to_numpy(np.float64)
lk = tb["last_known_tvt"].to_numpy(np.float64); tvt_true = lk + y
# blend(nnls gbm,tcn,l5)
d1 = np.load(ART.parent / "artifacts_v95" / "l1_oof.npz")
gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(ART / "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(ART / "l5_ext_oof.npz")["l5_mean"].astype(float)
M = np.stack([gbm, tcn, l5], 1); mfin = np.isfinite(M).all(1) & np.isfinite(y)
wgt, _ = nnls(M[mfin], y[mfin]); wgt = wgt / wgt.sum()
blend_res = M @ wgt; blend_tvt = lk + blend_res
wid = pd.unique(g); rows_of = {w: np.where(g == w)[0] for w in wid}
def rmse(e): e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
base = rmse(blend_tvt - tvt_true)
print(f"[cl] bank={BANK} σ={SIGS} blend overall={base:.4f} wells={len(wid)}", flush=True)

# ---- 入力構築(tw, bank固定)。anc=blend_tvt / ancs=σ を後で差し替え ----
P = {**pf.bank_param(BANK), "anchor_mult": 1.0}      # asig = ancs*1 = σ
print("[cl] 入力構築(tw) ...", flush=True); t0 = time.perf_counter()
inps = []; keep = []
for w in wid:
    hp = DATA / f"{w}__horizontal_well.csv"; tp = DATA / f"{w}__typewell.csv"
    try:
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
    except Exception:
        keep.append(False); inps.append(None); continue
    x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
    if x is None or len(x["md"]) != len(rows_of[w]):
        keep.append(False); inps.append(None); continue
    r = rows_of[w]; x["anc0"] = blend_tvt[r].astype(float); x["_wid"] = w; x["_rows"] = r
    inps.append(x); keep.append(True)
inps_ok = [x for x, k in zip(inps, keep) if k]
print(f"      {len(inps_ok)}/{len(wid)}井 ({time.perf_counter()-t0:.0f}s)", flush=True)

def run_sig(sig):
    for x in inps_ok:
        n = len(x["md"]); x["anc"] = x["anc0"].copy(); x["ancs"] = np.full(n, sig, float)
    outs = pf.run_smoother_ext(inps_ok, P, SEED, N_SEED, WELL_CHUNK, w_nn=0.0, capture=False)
    cl = np.full(len(g), np.nan); sp = np.full(len(g), np.nan)
    for x, o in zip(inps_ok, outs):
        r = x["_rows"]; T = min(len(r), len(o["mean"]))
        cl[r[:T]] = o["mean"][:T]; sp[r[:T]] = o["std"][:T]
    return cl, sp

out_lines = [f"=== v95 closed-loop (pseudo-satellite) bank={BANK} ==="]
out_lines.append(f"blend(prior) overall = {base:.4f}")
store = {}
for sig in SIGS:
    t0 = time.perf_counter(); cl, sp = run_sig(sig)
    store[f"cl_{int(sig)}"] = cl.astype(np.float32); store[f"spread_{int(sig)}"] = sp.astype(np.float32)
    e_cl = cl - tvt_true; e_bl = blend_tvt - tvt_true
    m = np.isfinite(e_cl) & np.isfinite(e_bl)
    rho = float(np.corrcoef(e_cl[m], e_bl[m])[0, 1])
    rms_diff = float(np.sqrt(np.mean((cl[m] - blend_tvt[m]) ** 2)))
    # β blend sweep(粗)
    best = (base, 0.0)
    for b in np.arange(0, 1.01, 0.1):
        p = b * cl + (1 - b) * blend_tvt; r = rmse(p - tvt_true)
        if r < best[0]: best = (r, float(b))
    line = (f"σ={int(sig):3d}: cl単体={rmse(e_cl):.4f}  ρ(cl,blend)={rho:.3f}  RMS(cl−blend)={rms_diff:.2f}  "
            f"★β blend={best[0]:.4f}(β={best[1]:.1f}, {best[0]-base:+.4f})")
    out_lines.append(line); print(line, flush=True)
    if pf.DEVICE == "cuda": torch.cuda.empty_cache()

np.savez(ART / "closed_loop_v95.npz", blend_tvt=blend_tvt.astype(np.float32), lk=lk.astype(np.float32), **store)
txt = "\n".join(out_lines)
(ART / "closed_loop_v95.txt").write_text(txt, encoding="utf-8")
print("\n保存: closed_loop_v95.npz(cl_/spread_ = 新特徴素材) + .txt", flush=True)
print("→ 次: cl_res=cl−blend / cl_spread を GBM特徴に足して OOF利得を見る(v30 r1)", flush=True)
