# -*- coding: utf-8 -*-
"""v95 モードゲート用 per-seed dump（v34/dump_seeds_v34.py の v95移植）。
=====================================================================================
- banks = pf_1/pf_2/pf_3（PF_LIB相当の3本smooth）, sources = tw/self/nbr（v34と同一構成9本）。
- create_v95 と同一の SEED / N_SEED / WELL_CHUNK / 井戸順 / 入力ビルダ / attach_anchor で
  run_smoother_ext(capture=True) を回し、平均前の per-seed 平滑ps・forward pfw・ll・ww を吐く。
  w_nn=0.0（create_v95=NN-emission OFF と一致）。
- 自己チェック: dump の out['std'] ≒ 対応 parquet pf_{rep}_std_{k}（std は定数シフト不変=transform非依存）。

出力: v95/artifacts_v95b/seed_dump_pf{1,2,3}.npz
  各井戸 w, src ∈ {tw,self,nbr}:
    {w}|ps_{src}  (S,T) f32 / {w}|pfw_{src} (S,T) f32 / {w}|ll_{src} (S,) f64 / {w}|ww_{src} (S,) f64
    {w}|gg_{src}  参照GR grid / {w}|meta [gmin,gst,gs] / {w}|gr (T,) / {w}|md {w}|z (T,) / {w}|ir
実行: .venv/Scripts/python.exe v95/dump_seeds_v95.py
"""
import os, sys, io, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJ / "v95"))
import pf_banks_v95 as pf
import imputers_v95 as imp

OUT_DIR = PROJ / "v95" / "artifacts_v95b"          # dump 保存先（parquet と同じ場所）
DATA_DIR = pf.DATA_DIR
SEED = 4423098; N_SEED = pf.N_SEED; WELL_CHUNK = 60
BANKS = [b for b in os.environ.get("BANKS", "pf_1,pf_2,pf_3").split(",") if b]
SRCS = [s for s in os.environ.get("SRCS", "tw,self,nbr").split(",") if s]
REPCOL = {"tw": "ancc", "self": "self", "nbr": "nbr"}      # parquet列の表現名（tw=ancc）
PQ = OUT_DIR / "data" / "train.parquet"                    # 自己チェック用（nn版だがstdはほぼ不変）

# ---- 井戸リスト（create_v95 と同一フィルタ・順序）----
base = DATA_DIR / "train"
import glob
wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
P0 = pf.bank_param(pf.BANK_ORDER[0])
names = []; paths = []
for w in wells:
    hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
    try:
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
    except Exception:
        continue
    if "TVT" not in hw.columns:
        continue
    if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None:
        continue
    names.append(w); paths.append((hp, tp))
_NW = os.environ.get("NWELLS")
if _NW:
    names = names[:int(_NW)]; paths = paths[:int(_NW)]
print(f"[v95dump] wells={len(names)} banks={BANKS} × src={SRCS} = {len(BANKS)*len(SRCS)}本 seed={N_SEED} lag={pf.SMOOTH_LAG}", flush=True)

# ---- imputers + 近傍探索（create_v95 と同一）----
imp.build_imputers()
print("[nbr] 近傍探索 ...", flush=True)
NREFS = {}
for (hp, tp) in paths:
    wid = os.path.basename(str(hp)).split("__")[0]
    hw = pd.read_csv(hp)
    cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
    refw = imp.neighbors_of(cx, cy, exclude_wid=wid)
    NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if len(refw) >= imp.NEED_REFS else []
n_nbr = sum(1 for w in names if NREFS[w])
print(f"      近傍あり {n_nbr}/{len(names)}", flush=True)

# ---- 自己チェック用 parquet std ----
try:
    _chkcols = [f"pf_{REPCOL[s]}_std_{k}" for s in SRCS for k in (1, 2, 3)] + ["well"]
    _tbq = pq.read_table(PQ, columns=_chkcols).to_pandas()
    _gq = _tbq["well"].astype(str).to_numpy()
    _rows_of = {w: np.where(_gq == w)[0] for w in names}
    HAVE_CHK = True
except Exception as e:
    print(f"  [check] parquet読めず self-check skip: {e}", flush=True); HAVE_CHK = False

def builder(src, hw, tvt, gr, wid, P):
    if src == "tw":   return pf.build_smoother_inputs(hw, tvt, gr, P)
    if src == "self": return pf.build_inputs_self(hw, tvt, gr, P)
    if src == "nbr":  return pf.build_inputs_nbr(hw, tvt, gr, P, NREFS[wid])
    raise ValueError(src)

for bi, bank in enumerate(BANKS):
    suf = bi + 1
    P = pf.bank_param(bank)
    print(f"[{bank}] 入力構築 ...", flush=True); t0 = time.perf_counter()
    inps = {s: [] for s in SRCS}; grids = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        tvt = tw["TVT"].to_numpy(float); gr = tw["GR"].to_numpy(float)
        gd = {}
        for s in SRCS:
            x = builder(s, hw, tvt, gr, wid, P)
            x = pf.attach_anchor(x, wid, P["_physics"])
            inps[s].append(x)
            gd[f"gg_{s}"] = x["gg"].astype(np.float32)
        bs = inps["tw"][-1]
        gd["meta"] = np.array([bs["gmin"], bs["gst"], bs["gs"]], dtype=np.float64)
        gd["gr"] = bs["gr"].astype(np.float32); gd["md"] = bs["md"].astype(np.float32)
        gd["z"] = bs["z"].astype(np.float32); gd["ir"] = np.float64(bs["ir"])
        grids[wid] = gd
    print(f"      入力 {time.perf_counter()-t0:.0f}s", flush=True)

    pack = {}
    for wid in names:
        gd = grids[wid]
        for s in SRCS: pack[f"{wid}|gg_{s}"] = gd[f"gg_{s}"]
        pack[f"{wid}|meta"] = gd["meta"]; pack[f"{wid}|gr"] = gd["gr"]
        pack[f"{wid}|md"] = gd["md"]; pack[f"{wid}|z"] = gd["z"]; pack[f"{wid}|ir"] = gd["ir"]

    for s in SRCS:
        print(f"      [{bank}-{s}] PF実行(32seed capture) ...", flush=True); t0 = time.perf_counter()
        outs = pf.run_smoother_ext(inps[s], P, SEED, N_SEED, WELL_CHUNK, w_nn=0.0, capture=True)
        print(f"        {time.perf_counter()-t0:.0f}s", flush=True)
        diffs = []
        for wid, o in zip(names, outs):
            pack[f"{wid}|ps_{s}"] = o["ps"]; pack[f"{wid}|pfw_{s}"] = o["pfw"]
            pack[f"{wid}|ll_{s}"] = o["ll"]; pack[f"{wid}|ww_{s}"] = o["ww"]
            if HAVE_CHK:
                r = _rows_of.get(wid); c = f"pf_{REPCOL[s]}_std_{suf}"
                if r is not None and len(r) and c in _tbq.columns:
                    T = min(len(r), o["std"].shape[0])
                    ref = _tbq[c].to_numpy()[r][:T]; got = o["std"][:T]
                    m = np.isfinite(ref) & np.isfinite(got)
                    if m.any(): diffs.append(float(np.nanmedian(np.abs(ref[m] - got[m]))))
        if diffs:
            md = float(np.nanmedian(diffs))
            print(f"        [check pf_{REPCOL[s]}_std_{suf}] |dump std − parquet| median={md:.3f} "
                  + ("→ OK" if md < 0.6 else "→ ★ズレ(nn差 or drift, 要確認)"), flush=True)
        if pf.DEVICE == "cuda": torch.cuda.empty_cache()

    outp = OUT_DIR / f"seed_dump_pf{suf}.npz"
    np.savez(outp, **pack)
    print(f"      saved {outp}  ({sum(v.nbytes for v in pack.values())/1e9:.1f}GB)", flush=True)
    del pack, inps, grids

print("[v95dump] 完了", flush=True)
