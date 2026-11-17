# -*- coding: utf-8 -*-
"""tune_decorr の best(PHYS=off)を pf_banks の"7番目のbank(newpf)"として登録し、
   ★他PFと同じ smooth PF で tw/self/nbr(mean+std)を全井生成(forward, 逆走なし, n_seeds=32)。
   出力: newpf_bank_train.npz = {pf_ancc_delta_7/std_7, pf_self_delta_7/std_7, pf_nbr_delta_7/std_7}
         を train.parquet 行に整列(既存ファイル上書きなし)。+ 脱相関corr(smooth版 ancc_7 vs 既存6)を出力。
   P7 = pf_1 の全キー + tuned 22共通を上書き + use_anchor=False。gs_scale は GS_SCALE env。
   実行: rogii_claude/.venv/Scripts/python.exe v93/gen_newpf_bank_v93.py"""
import os, sys, io, glob, json, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, torch, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v93 as pf
import imputers_v93 as imp

DATA = pf.DATA_DIR / "train"; ART = pf.ART
SEED = 4423098; N_SEED = int(os.environ.get("NS", "32")); CHUNK = 40
OUT = ART / "newpf_bank_train.npz"

# --- P7 構築: pf_1 の全キー + tuned 22共通 + use_anchor=False ---
tuned = json.load(open(ART / "v93_decorr_best_off.json", encoding="utf-8"))["best_params"]
P7 = pf.bank_param("pf_1")                     # 全pf_banksキー + _physics/_w_nn/smooth_lag
for k in set(P7) & set(tuned):
    P7[k] = tuned[k]
P7["use_anchor"] = False; P7["_physics"] = False; P7["name"] = "newpf"
os.environ["GS_SCALE"] = str(float(tuned.get("gs_scale", 1.0)))
print(f"[gen bank] P7: lik={P7['likelihood_scale']:.2f} gr_power={P7['gr_power']:.2f} "
      f"gs_scale(env)={os.environ['GS_SCALE']} use_anchor=False n_seeds={N_SEED}", flush=True)

# --- 有効井 + imputer + nbr + sim(build_backward と同一) ---
wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(DATA / "*__horizontal_well.csv"))})
names = []; paths = []
for w in wells:
    hp = DATA / f"{w}__horizontal_well.csv"; tp = DATA / f"{w}__typewell.csv"
    try:
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
    except Exception:
        continue
    if "TVT" not in hw.columns:
        continue
    if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P7) is None:
        continue
    names.append(w); paths.append((hp, tp))
print(f"[gen bank] 有効井={len(names)}", flush=True)
SIMD = pickle.loads((ART / "sim_grfree_v93.pkl").read_bytes()) if (ART / "sim_grfree_v93.pkl").exists() else {}
imp.build_imputers()
NREFS = {}
for (hp, tp) in paths:
    wid = os.path.basename(str(hp)).split("__")[0]
    hwxy = pd.read_csv(hp, usecols=["X", "Y"]); cx, cy = float(hwxy["X"].mean()), float(hwxy["Y"].mean())
    refw = imp.neighbors_of(cx, cy, exclude_wid=wid)
    NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if len(refw) >= imp.NEED_REFS else []


def run_rep(rep):
    inps = []
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        if rep == "tw":
            x = pf.build_smoother_inputs(hw, tt, tg, P7)
        elif rep == "self":
            x = pf.build_inputs_self(hw, tt, tg, P7)
        else:
            x = pf.build_inputs_nbr(hw, tt, tg, P7, NREFS[wid])
        x = pf.attach_anchor(x, wid, P7["_physics"])
        sd = SIMD.get(wid)
        if sd is not None:
            x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
        inps.append(x)
    outs = pf.run_smoother_ext(inps, P7, SEED, N_SEED, CHUNK, w_nn=P7["_w_nn"])
    if pf.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {names[i]: outs[i] for i in range(len(names))}


import time
REP = {}
for rep in ("tw", "self", "nbr"):
    t0 = time.time(); print(f"[gen bank] {rep} ...", end="", flush=True)
    REP[rep] = run_rep(rep)
    print(f" done {time.time()-t0:.0f}s", flush=True)

# --- train.parquet 行整列 + 6列(ancc/self/nbr delta+std) ---
tb = pq.read_table(ART / "data" / "train.parquet", columns=["id", "well", "last_known_tvt", "target"])
well = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
tgt = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
true = tgt + lk; N = len(well)
COLS = {}
for rep, tag in (("tw", "ancc"), ("self", "self"), ("nbr", "nbr")):
    d = np.full(N, np.nan, np.float32); s = np.full(N, np.nan, np.float32)
    for w in names:
        idx = np.where(well == w)[0]; o = REP[rep].get(w)
        if o is None:
            continue
        m = min(len(idx), len(o["mean"])); ii = idx[:m]
        d[ii] = (np.asarray(o["mean"][:m], np.float64) - lk[ii]).astype(np.float32)
        s[ii] = np.asarray(o["std"][:m], np.float32)
    COLS[f"pf_{tag}_delta_7"] = d; COLS[f"pf_{tag}_std_7"] = s
np.savez_compressed(OUT, **COLS, id=tb.column("id").to_pandas().astype(str).to_numpy())
print(f"saved {OUT.name}  6列(ancc/self/nbr delta+std)", flush=True)

# --- 脱相関 corr(smooth版 ancc_7 誤差 vs 既存6PF, 全井) ---
BANKS = [f"pf_ancc_{k}" for k in range(1, 7)]
tb2 = pq.read_table(ART / "data" / "train.parquet", columns=BANKS)
enew = (COLS["pf_ancc_delta_7"] + lk) - true    # 新PF誤差
print("\n=== ★脱相関検証(smooth版 newpf_ancc vs 既存6PF, 全井) ===")
cors = []
for bi, b in enumerate(BANKS):
    v = tb2.column(b).to_numpy(zero_copy_only=False).astype(np.float64) - true
    m = np.isfinite(enew) & np.isfinite(v)
    c = float(np.corrcoef(enew[m], v[m])[0, 1]) if m.sum() > 10 else 0.0
    cors.append(c); print(f"  vs pf_{bi+1}: {c:+.3f}")
rnew = float(np.sqrt(np.mean(enew[np.isfinite(enew)] ** 2)))
print(f"  max_corr={max(cors):.3f}  newpf(smooth)全井RMSE={rnew:.3f}")
print(f"  制約corr≤0.7: {'○満たす' if max(cors)<=0.7 else '×崩れた(forward調整がsmoothに転移せず→Bへ)'}")
