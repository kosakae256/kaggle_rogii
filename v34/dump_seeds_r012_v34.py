# -*- coding: utf-8 -*-
"""v34 dump拡張: v19 32seeds_re バンクの r0/r1/r2 パラメータで smooth PF を回し、per-seed を dump。
=====================================================================================
- パラメータ = v19/artifacts_v19_32seeds_re/pf_bank.json の round 0/1/2(スキーマはPF_LIBと同一)。
- ソース = tw/self/nbr × 32seed × sm(平滑)/fw(forward)。dump_seeds_v34.py と同じ出力形式:
  seed_dump_r{0,1,2}.npz({well}|ps_*/pfw_*/ll_*/ww_*/gg_*/meta/gr/md/z/ir)
- 井戸順=parquet順、SEED/lag/chunk は create と同一。
- parquet に対応列が無いので自己チェックの代わりに、各パスの smooth 加重平均の raw overall RMSE
  (真値あり・カバー行)を表示する(バンク精度表との突合に使える)。
実行: .venv/Scripts/python.exe v34/dump_seeds_r012_v34.py(9パス、GPU数時間級)
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v28"))
import create_train_parquet_v28 as C


def run_smoother_dump(inps, P, seed, n_seeds, chunk):
    """dump_seeds_v34.py と同一(importするとdumpが走るためインライン複製)。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = C._v26_pad(sub, C.V26_DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=C.V26_DEVICE); gen.manual_seed(seed)
        pf, ps, ll = C._v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            P, N, C.V26_DEVICE, gen)
        ps = ps.view(w, S, -1).double().cpu().numpy()
        pf = pf.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                lk_ = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk_ / max(ls_scale, 1e-6)); s = ww.sum()
                ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out[c0 + j] = dict(ps=ps[j, :, :T].astype(np.float32),
                               pfw=pf[j, :, :T].astype(np.float32),
                               ll=llw.astype(np.float64), ww=ww.astype(np.float64))
        if C.V26_DEVICE == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    return out

ART = PROJECT / "v34" / "artifacts_v34"
V28_PQ = PROJECT / "v28" / "artifacts_v28" / "data" / "train.parquet"
BASE = C.DATA_DIR_V26 / "train"
BANK = json.load(open(PROJECT / "v19" / "artifacts_v19_32seeds_re" / "pf_bank.json", encoding="utf-8"))
PARAMS = [(f"r{e['round']}", e["params"]) for e in BANK if e["round"] in (0, 1, 2)]
SRCS = ["tw", "self", "nbr"]

tbq = pq.read_table(V28_PQ, columns=["well", "target", "last_known_tvt"])
gq = tbq.column("well").to_pandas().to_numpy()
names = list(pd.unique(gq))
rows_of = {w: np.where(gq == w)[0] for w in names}
tvt_true = (tbq.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
            + tbq.column("target").to_numpy(zero_copy_only=False).astype(np.float64))
print(f"[r012] wells={len(names)} device={C.V26_DEVICE} n_seed={C.N_SEED} lag={C.SMOOTH_LAG} chunk={C.WELL_CHUNK}")
print(f"[r012] params={[lab for lab, _ in PARAMS]}(n_particles={[p['n_particles'] for _, p in PARAMS]})")

print("[nbr] 近傍探索 ...")
NREFS, NBR_HAS = {}, {}
for w in names:
    hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
    cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
    refw = C.neighbors_of(cx, cy, exclude_wid=w)
    use = len(refw) >= C.NEED_REFS
    NREFS[w] = [C._ref_grtvt(r) for r, _ in refw] if use else []
    NBR_HAS[w] = int(use)
print(f"      近傍あり {sum(NBR_HAS.values())}/{len(names)}")

for lab, raw in PARAMS:
    Pk = {**raw, "smooth_lag": C.SMOOTH_LAG}
    print(f"[{lab}] 入力構築 ..."); t0 = time.perf_counter()
    inps = {s: [] for s in SRCS}; grids = {}
    for w in names:
        hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv")
        tw = pd.read_csv(BASE / f"{w}__typewell.csv").sort_values("TVT")
        twt = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
        bs = C.build_smoother_inputs(hw, twt, twg, Pk)
        if bs is None:
            raise RuntimeError(f"build_smoother_inputs None: {w}")
        sf = C.build_inputs_self(hw, twt, twg, Pk)
        nb = C.build_inputs_nbr(hw, twt, twg, Pk, NREFS[w])
        inps["tw"].append(bs); inps["self"].append(sf); inps["nbr"].append(nb)
        grids[w] = dict(gg_tw=bs["gg"].astype(np.float32), gg_self=sf["gg"].astype(np.float32),
                        gg_nbr=(nb["gg"].astype(np.float32) if NBR_HAS[w] else None),
                        meta=np.array([bs["gmin"], bs["gst"], bs["gs"]], dtype=np.float64),
                        gr=bs["gr"].astype(np.float32),
                        md=bs["md"].astype(np.float32), zz=bs["z"].astype(np.float32),
                        ir=np.float64(bs["ir"]))
    print(f"      入力 {time.perf_counter()-t0:.0f}s")
    pack = {}
    for w in names:
        gd = grids[w]
        pack[f"{w}|gg_tw"] = gd["gg_tw"]; pack[f"{w}|gg_self"] = gd["gg_self"]
        if gd["gg_nbr"] is not None:
            pack[f"{w}|gg_nbr"] = gd["gg_nbr"]
        pack[f"{w}|meta"] = gd["meta"]; pack[f"{w}|gr"] = gd["gr"]
        pack[f"{w}|md"] = gd["md"]; pack[f"{w}|z"] = gd["zz"]; pack[f"{w}|ir"] = gd["ir"]
    for src in SRCS:
        print(f"      [smoothPF-{src}] 実行(32seed) ..."); t0 = time.perf_counter()
        outs = run_smoother_dump(inps[src], Pk, C.SEED_V26, C.N_SEED, C.WELL_CHUNK)
        errs = []
        for w, o in zip(names, outs):
            r = rows_of[w]; T = min(o["ps"].shape[1], len(r))
            smean = (o["ww"][:, None] * o["ps"].astype(np.float64)).sum(0)
            e = smean[:T] - tvt_true[r[:T]]
            e = e[np.isfinite(e)]
            if e.size: errs.append(e)
            pack[f"{w}|ps_{src}"] = o["ps"]; pack[f"{w}|pfw_{src}"] = o["pfw"]
            pack[f"{w}|ll_{src}"] = o["ll"]; pack[f"{w}|ww_{src}"] = o["ww"]
        ov = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        print(f"        PF {time.perf_counter()-t0:.0f}s | sm加重平均のraw overall = {ov:.3f}")
    np.savez(ART / f"seed_dump_{lab}.npz", **pack)
    print(f"      saved: {ART / f'seed_dump_{lab}.npz'}")
print("[r012] dump 完了(3param × tw/self/nbr = 9本)")
