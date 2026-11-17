# -*- coding: utf-8 -*-
"""tune_decorr の best(PHYS=off, use_anchor=False)PF を全井で走らせ、pf_ancc(mean)+std を
   train.parquet の行に整列して newpf_train.npz に保存(既存ファイル上書きしない)。
   L5 に「今回調整した脱相関PFを1本追加」する実験用。n_seeds=32(本番同等)。GPU。
実行: rogii_claude/.venv/Scripts/python.exe v93/gen_newpf_v93.py"""
import os, sys, io, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v93 as pf

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude") / "v93" / "artifacts_v93"
OUT = ART / "newpf_train.npz"
N_SEEDS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "60"))
BEST = json.load(open(ART / "v93_decorr_best_off.json", encoding="utf-8"))
P = dict(BEST["best_params"]); P["use_anchor"] = False
P = pf.full_params(P)
print(f"[gen newpf] best(160)RMSE={BEST['rmse']:.3f} gs_scale={P.get('gs_scale'):.3f} "
      f"use_anchor=False n_seeds={N_SEEDS}", flush=True)


def ens_meanstd(inps, seed):
    """run_forward_ens と同一だが per-eval-row の mean と std(尤度加重)を返す。"""
    S = N_SEEDS; W = len(inps); pad = pf._pad(inps, pf.DEVICE)
    rep = lambda t: t.repeat_interleave(S, 0)
    gen = torch.Generator(device=pf.DEVICE); gen.manual_seed(seed)
    pts, ll = pf._pf_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                          pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                          pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                          rep(pad["anc"]), rep(pad["ancs"]), P, int(P["n_particles"]), pf.DEVICE, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    means, stds = [], []
    for w in range(W):
        T = len(inps[w]["md"]); pr = pts[w, :, :T]; llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        mean = (ww[:, None] * pr).sum(0)
        var = (ww[:, None] * (pr - mean) ** 2).sum(0)
        means.append(mean); stds.append(np.sqrt(np.maximum(var, 0.0)))
    return means, stds


wells = [w for w in pf.list_wells() if pf.CACHE.setdefault(w, pf.build_cache(w)) is not None]
print(f"[gen newpf] 有効井={len(wells)} 生成開始...", flush=True)
MEAN = {}; STD = {}
for c0 in range(0, len(wells), CHUNK):
    sub = wells[c0:c0 + CHUNK]
    inps = [pf.make_inputs(pf.CACHE[w], P) for w in sub]
    ms, ss = ens_meanstd(inps, seed=777)
    for k, w in enumerate(sub):
        MEAN[w] = ms[k]; STD[w] = ss[k]
    if pf.DEVICE == "cuda":
        torch.cuda.synchronize()
    print(f"  {min(c0+CHUNK,len(wells))}/{len(wells)}", flush=True)

# train.parquet 行に整列(井ごと positional。engine eval行 = parquet行, 順一致=tunerで確認済)
tb = pq.read_table(ART / "data" / "train.parquet", columns=["id", "well", "last_known_tvt"])
ids = tb.column("id").to_pandas().astype(str).to_numpy()
well = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
N = len(ids)
delta = np.full(N, np.nan, np.float32); std = np.full(N, np.nan, np.float32)
n_mismatch = 0
for w in wells:
    idx = np.where(well == w)[0]
    m = MEAN.get(w)
    if m is None:
        continue
    k = min(len(idx), len(m))
    if k != len(idx):
        n_mismatch += 1
    ii = idx[:k]
    delta[ii] = (np.asarray(m[:k], np.float64) - lk[ii]).astype(np.float32)
    std[ii] = np.asarray(STD[w][:k], np.float32)
if n_mismatch:
    print(f"[align] WARN 長さ不一致 {n_mismatch}井(min切詰)")
np.savez_compressed(OUT, delta=delta, std=std, id=ids)
print(f"saved {OUT.name}  rows={N} finite(delta)={np.isfinite(delta).mean():.3f}")
