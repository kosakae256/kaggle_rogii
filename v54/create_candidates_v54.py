# v54 create — per-seed 候補軌跡の全dump(seed平均で潰さない)
#   6バンク(pf_1/2/3, r0, r1, pfA)× 3フレーバー(tw/nbr/self)× SEEDS(既定32)
#   NN-emission: W=0.02(pfAのみ0.01+struct錨)、smooth lag16(v52と同一土俵)。
# 出力: v54/artifacts_v54/cand_{bank}_{flav}.npz
#   key f"{well}"   : (S, T) float16  per-seed 平滑軌跡(z, ft)
#   key f"{well}|w" : (S,)   float32  per-seed 尤度重み(正規化済)
# キャッシュ: (bank,flav) ファイル単位。全坑井カバレッジ確認でskip(部分キャッシュ事故対策)。
# 使い方:  python v54/create_candidates_v54.py            # 全18パス(~3.5-4h想定)
#          NWELLS=3 SEEDS=4 BANKS=r0_seed32 FLAVORS=tw ... # スモーク
import os, sys, time, pickle
import numpy as np
import pandas as pd
import torch

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
import nnpf_v51 as npf                     # PF_LIB / build_smoother_inputs / SIMD(OOF sim) / _core

ART = os.path.join(P0, "v54", "artifacts_v54"); os.makedirs(ART, exist_ok=True)
BANKS = os.environ.get("BANKS", "pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA").split(",")
FLAVORS = os.environ.get("FLAVORS", "tw,nbr,self").split(",")
SEEDS = int(os.environ.get("SEEDS", "32")); SEED0 = int(os.environ.get("SEED0", "777"))
CHUNK = int(os.environ.get("CHUNK", "12"))
LAG = int(os.environ.get("LAG", "16"))
DEV = npf.V26_DEVICE
npf.SMOOTH_LAG = LAG

# ---------------- 入力構築(v53 create から移植: self/nbr フレーバー) ----------------
def build_inputs_self_local(hw, tw_tvt, tw_gr, P):
    base = npf.build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    ys = kn["TVT_input"].to_numpy(float)
    gs_ = pd.Series(hw["GR"].to_numpy(float)).interpolate(limit_direction="both").to_numpy()
    gs_ = np.where(np.isfinite(gs_), gs_, np.nanmedian(gs_))[hw.index.get_indexer(kn.index)]
    G = len(base["gg"]); gmin = base["gmin"]; gst = base["gst"]
    idx = np.round((ys - gmin) / gst).astype(np.int64)
    ok = (idx >= 0) & (idx < G) & np.isfinite(ys)
    if ok.sum() >= 20:
        ssum = np.zeros(G); scnt = np.zeros(G)
        np.add.at(ssum, idx[ok], gs_[ok]); np.add.at(scnt, idx[ok], 1)
        base = dict(base); base["gg"] = np.where(scnt > 0, ssum / np.maximum(scnt, 1), base["gg"])
    return base

def build_inputs_nbr_local(hw, tw_tvt, tw_gr, P, nbr_pts):
    base = npf.build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None or nbr_pts is None: return base
    ny, ng = nbr_pts
    G = len(base["gg"]); gmin = base["gmin"]; gst = base["gst"]
    idx = np.round((ny - gmin) / gst).astype(np.int64)
    ok = (idx >= 0) & (idx < G)
    if ok.sum() >= 20:
        ssum = np.zeros(G); scnt = np.zeros(G)
        np.add.at(ssum, idx[ok], ng[ok]); np.add.at(scnt, idx[ok], 1)
        base = dict(base); base["gg"] = np.where(scnt > 0, ssum / np.maximum(scnt, 1), base["gg"])
    return base

# ---------------- run_paths: run_smoother の per-seed 版(平均しない) ----------------
def run_paths(inps, sims_w, sgs_w, P, seed, n_seeds, chunk, W_NN, use_anchor=False):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps)
    out_pr = [None] * W; out_ww = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; wsub = sims_w[c0:c0 + chunk]; wq = len(sub)
        Tmax = max(len(x["md"]) for x in sub); Gmax = max(len(x["gg"]) for x in sub)
        md = torch.zeros((wq, Tmax)); z = torch.zeros((wq, Tmax))
        gr = torch.full((wq, Tmax), float("nan")); valid = torch.zeros((wq, Tmax), dtype=torch.bool)
        grid = torch.zeros((wq, Gmax)); glen = torch.zeros(wq, dtype=torch.long)
        vmin = torch.zeros(wq); step = torch.zeros(wq); gs = torch.zeros(wq)
        ls = torch.zeros(wq); ir = torch.zeros(wq)
        stq = torch.zeros((wq, Tmax)); simt = torch.full((wq, Tmax, 2 * npf.NBAND + 1), -1.0, dtype=torch.float16)
        sgq = torch.full((wq, Tmax), 1e9)
        for b, (x, sd, sg_) in enumerate(zip(sub, wsub, sgs_w[c0:c0 + chunk])):
            Tn = len(x["md"]); G = len(x["gg"])
            md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
            if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
            z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
            gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
            grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
            if G < Gmax: grid[b, G:] = grid[b, G - 1]
            glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
            Ts = min(Tn, len(sd["st"]))
            stq[b, :Ts] = torch.from_numpy(sd["st"][:Ts])
            if Ts < Tmax: stq[b, Ts:] = stq[b, Ts - 1]
            simt[b, :Ts] = torch.from_numpy(sd["sim"][:Ts])
            if sg_ is not None:
                Ts2 = min(Tn, len(sg_)); sgq[b, :Ts2] = torch.from_numpy(sg_[:Ts2].astype("float32"))
        to = lambda tt: tt.to(DEV)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        wmap = torch.arange(wq, device=DEV).repeat_interleave(S)
        pf, ps, ll = npf._core(rep(to(md)), rep(to(z)), rep(to(gr)), rep(to(valid)), rep(to(grid)),
                               to(glen).repeat_interleave(S), to(vmin).repeat_interleave(S),
                               to(step).repeat_interleave(S), to(gs).repeat_interleave(S),
                               to(ls).repeat_interleave(S), to(ir).repeat_interleave(S),
                               rep(to(stq)), rep(to(sgq)), to(simt), wmap, W_NN, use_anchor,
                               {**P, "smooth_lag": LAG}, N, DEV, gen)
        ps = ps.view(wq, S, -1).float().cpu().numpy(); ll = ll.view(wq, S).cpu().numpy()
        for b in range(wq):
            T = len(sub[b]["md"])
            pr = ps[b, :, :T]
            llw = ll[b]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                lkw = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lkw / max(ls_scale, 1e-6)); ssum = ww.sum()
                ww = ww / ssum if ssum > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out_pr[c0 + b] = pr.astype(np.float16)
            out_ww[c0 + b] = ww.astype(np.float32)
        if DEV == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    return out_pr, out_ww

if __name__ == "__main__":
    t0 = time.time()
    wells = sorted(npf.SIMD.keys())
    if os.environ.get("NWELLS"): wells = wells[:int(os.environ["NWELLS"])]
    # stageAキャッシュ(既存)から nbr 既知プール
    import importlib.util
    sa_fp = os.path.join(P0, "v50", "artifacts_v50", "cacheA_v50.pkl")
    cache = pickle.loads(open(sa_fp, "rb").read())
    SGD = {w: cache[w]["sg"] for w in cache}
    XY = np.array([cache[w]["xy"] for w in wells])
    KN = {}
    for w in wells:
        c = cache[w]; knm = (~c["ev"]) & np.isfinite(c["y"])
        KN[w] = (c["y"][knm].astype(np.float64), c["graw"][knm].astype(np.float64))
    from scipy.spatial import cKDTree
    _, nidx = cKDTree(XY).query(XY, k=min(9, len(wells)))
    NBRPTS = {}
    for i, w in enumerate(wells):
        js = [j for j in np.atleast_1d(nidx[i]) if j < len(wells) and wells[j] != w][:8]
        NBRPTS[w] = ((np.concatenate([KN[wells[j]][0] for j in js]),
                      np.concatenate([KN[wells[j]][1] for j in js])) if js else None)
    del cache
    DATA = []
    for wn in wells:
        hw = pd.read_csv(npf.TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(npf.TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        if hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 10: continue
        DATA.append((wn, hw, tw))
    print(f"wells {len(DATA)}  banks {BANKS}  flavors {FLAVORS}  seeds {SEEDS} lag {LAG}  ({time.time()-t0:.0f}s)", flush=True)

    for bank in BANKS:
        Praw = {k: v for k, v in npf.PF_LIB[bank].items() if k != "name"}
        wnn = 0.01 if bank == "pfA" else 0.02
        use_anc = (bank == "pfA")
        for flav in FLAVORS:
            fp = os.path.join(ART, f"cand_{bank}_{flav}.npz")
            if os.path.exists(fp):
                have = set(k for k in np.load(fp).files if not k.endswith("|w"))
                if all(wn in have for wn, _, _ in DATA):
                    print(f"[skip cached] {bank}/{flav}"); continue
                print(f"[recompute partial] {bank}/{flav} ({len(have)} wells cached)")
            print(f"[paths] {bank}/{flav} (W_NN={wnn}, anchor={use_anc}) ...", flush=True)
            t1 = time.time(); inps = []; keep = []
            for wn, hw, tw in DATA:
                if flav == "tw":
                    x = npf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw)
                elif flav == "self":
                    x = build_inputs_self_local(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw)
                else:
                    x = build_inputs_nbr_local(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw, NBRPTS.get(wn))
                if x is not None: inps.append(x); keep.append(wn)
            sims_w = [npf.SIMD[w] for w in keep]; sgs_w = [SGD.get(w) for w in keep]
            prs, wws = run_paths(inps, sims_w, sgs_w, Praw, SEED0, SEEDS, CHUNK, wnn, use_anchor=use_anc)
            payload = {}
            for w, pr, ww in zip(keep, prs, wws):
                payload[w] = pr; payload[f"{w}|w"] = ww
            np.savez_compressed(fp, **payload)
            sz = os.path.getsize(fp) / (1 << 20)
            print(f"      saved {os.path.basename(fp)} ({sz:.0f}MB, {time.time()-t1:.0f}s)", flush=True)
    print(f"done ({(time.time()-t0)/60:.1f}min)")
