# -*- coding: utf-8 -*-
"""
v90.0 — 非逐次δ分布ガイド付きPF。PFの粒子重みに U-Net(softmax)のδ分布を各粒子位置で評価した項を掛け、
        両枝を生かしたまま逐次の滑らかさで選ばせる。v65本体は無改造(コア複製で lkn を追加)。
================================================================================
非逐次事前: v89 の δ分布(SAVE_DIST) P_ns(well, bin, δ) と 中心 st(learned pkl)。
  row i の粒子TVT v=pos−z → δ=v−st[i] を offs格子で内挿し P_ns を取得 → lkn=((1-fl)*P_n+fl)^β を重みに乗算。
r0(struct-free tw-PF, use_anchor=off, 32seed)を基準に fold0 val で比較。
env: BETA(0.15) / NSFLOOR(0.1) / NW(0=全fold0val) / SEED / NSEED(32) / CHUNK(20) / DIST(dist_f0) / PKL(learned_boot)
使い方: BETA=0 で r0 再現(8.03) 確認 → BETA走査。
"""
import os, sys, io, time, pickle
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
from pathlib import Path
import numpy as np, torch

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v65"))
os.environ["V65_TWSRC"] = "tw"
import pf_engine_v65 as base
import warmup_v65 as wu

DEV = base.DEVICE; DTYPE = base.DTYPE
NB = 90; ND = 181; STEP = 0.5                       # 非逐次δ格子(offs=(j-NB)*STEP)
BETA = float(os.environ.get("BETA", "0.15"))
NSFLOOR = float(os.environ.get("NSFLOOR", "0.1"))
NSEED = int(os.environ.get("NSEED", "32")); SEED = int(os.environ.get("SEED", "20251707"))
CHUNK = int(os.environ.get("CHUNK", "20")); NW = int(os.environ.get("NW", "0")) or None
A = PROJECT / "v64" / "artifacts_v64"; AV = PROJECT / "v89" / "artifacts_v89"


def _pf_core_guided(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs,
                    ns_ctr, ns_bid, ns_distb, b_base, P, N, device, gen, dtype=DTYPE):
    """_pf_core(v65) の複製 + 非逐次δ分布ガイド lkn。ns_distb:(Wbase,nbmax,ND) row→bin=ns_bid。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")
    GB = float(P.get("ns_beta", 0.0)); GFL = float(P.get("ns_floor", 0.1))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + ((1.0 - ALPHA) * dipp if RT_IR else 0.0) + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        lkp = torch.exp(-0.5 * ((rate - dipp) / PHYS_SIG) ** 2).clamp_min(1e-300) if USE_P else torch.ones_like(w)
        # ★非逐次δ分布ガイド lkn: δ=v−ns_ctr を offs格子で内挿→P_ns→((1-fl)*P_n+fl)^β
        if GB > 0:
            delta = v - ns_ctr[:, i][:, None]                        # (B,N) 粒子のδ
            jf = delta / STEP + NB                                    # 連続index
            oob = (jf < 0) | (jf > ND - 1)
            j0 = torch.floor(jf).long().clamp(0, ND - 2); fr = jf - j0.to(dtype)
            drow = ns_distb[b_base, ns_bid[:, i]]                     # (B,ND) その行のbin分布
            rmax = drow.max(dim=1, keepdim=True).values.clamp_min(1e-9)
            pA = torch.gather(drow, 1, j0); pB = torch.gather(drow, 1, (j0 + 1).clamp(0, ND - 1))
            pv = (pA * (1 - fr) + pB * fr) / rmax                     # モード=1に正規化
            pv = torch.where(oob, torch.zeros_like(pv), pv)
            lkn = ((1.0 - GFL) * pv + GFL) ** GB
        else:
            lkn = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp * lkn, w * lkp * lkn)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pts[:, i] = (w * (pos - zi)).sum(1); pm = torch.where(act, cur, pm)
    return pts, log_lik


def run_guided(inps, nsarr, P, seed, n_seeds):
    """run_forward_ens(v65)の複製 + ns配列を渡す。nsarr=(ns_ctr,ns_bid,ns_distb) base解像度。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = base._pad(inps, DEV)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    ns_ctr, ns_bid, ns_distb = nsarr
    b_base = (torch.arange(W * S, device=DEV) // S)
    pts, ll = _pf_core_guided(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        rep(pad["anc"]), rep(pad["ancs"]),
        rep(ns_ctr), rep(ns_bid), ns_distb, b_base, P, N, DEV, gen)
    pts = pts.view(W, S, -1).double().cpu().numpy(); ll = ll.view(W, S).cpu().numpy()
    outs = []
    for wk in range(W):
        T = len(inps[wk]["md"]); pr = pts[wk, :, :T]; llw = ll[wk]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        outs.append((ww[:, None] * pr).sum(0))
    return outs


GRID = os.environ.get("GRID", "dist")   # dist=非逐次事後(v90.0) / emission=学習emission尤度(v90.1, EVのc_learn ch0)


def build_ns(sub, EVD, DIST, Tmax):
    """chunk分の ns配列(base解像度): ns_ctr(W,Tmax), ns_bid(W,Tmax), ns_distb(W,nbmax,ND)。
    GRID=dist→非逐次δ分布(posterior) / GRID=emission→EVのc_learn(ch0, 学習emission尤度)。"""
    W = len(sub); nbmax = max(int(EVD[w]["bid"].max()) + 1 for w in sub if w in EVD)
    ns_ctr = torch.zeros((W, Tmax), dtype=DTYPE); ns_bid = torch.zeros((W, Tmax), dtype=torch.long)
    ns_distb = torch.zeros((W, nbmax, ND), dtype=DTYPE)
    for b, w in enumerate(sub):
        e = EVD[w]; st = np.asarray(e["st"], np.float32); bid = np.asarray(e["bid"], np.int64)
        if GRID == "emission":
            G = np.asarray(e["EV"][0], np.float32)                   # c_learn (nb,ND) 学習emission
        else:
            G = np.asarray(DIST[w], np.float32)                      # posterior (nb,ND)
        Tn = min(len(st), len(bid))
        ns_ctr[b, :Tn] = torch.from_numpy(st[:Tn])
        if Tn < Tmax: ns_ctr[b, Tn:] = ns_ctr[b, Tn - 1]
        ns_bid[b, :Tn] = torch.from_numpy(bid[:Tn].clip(0, nbmax - 1))
        nb = min(len(G), nbmax); ns_distb[b, :nb] = torch.from_numpy(G[:nb])
    return ns_ctr.to(DEV), ns_bid.to(DEV), ns_distb.to(DEV)


def main():
    t0 = time.time()
    # 非逐次資産(fold0): dist + st/bid(pkl)
    EVD = pickle.loads(open(AV / os.environ.get("PKL", "learned_boot.pkl"), "rb").read())
    if os.environ.get("ALLWELLS") and GRID == "emission":
        DIST = {}; valwells = sorted(EVD.keys())                      # 全773(emissionはpkl EV[0]から, dist不要)
    else:
        DIST = dict(np.load(A / os.environ.get("DIST", "v89_dist_f0_dist.npz")))
        valwells = [w for w in DIST if w in EVD]                      # fold0 val
    P = base.full_params(dict(wu.R0_PARAMS))
    BANK = os.environ.get("BANK")                                    # ★v90.2: p_1..p_5 のパラメータで走査
    if BANK:
        import json as _json
        bp = _json.load(open(PROJECT / "v90" / "pf_banks_v69_4.json"))[BANK]
        P.update(bp)
    P["use_anchor"] = False; P["n_seeds"] = NSEED; P["tw_source"] = "tw"
    P["ns_beta"] = BETA; P["ns_floor"] = NSFLOOR
    # cache
    cache = {}
    for w in valwells:
        c = base.build_cache(w)
        if c is not None: cache[w] = c
    base.CACHE = cache
    ALL = [w for w in valwells if w in cache]
    if NW: ALL = ALL[:NW]
    print(f"[v90.0] guided PF fold0 {len(ALL)}井 BETA={BETA} NSFLOOR={NSFLOOR} ({time.time()-t0:.0f}s)", flush=True)
    errs = []; pw = {}
    for c0 in range(0, len(ALL), CHUNK):
        sub = ALL[c0:c0 + CHUNK]
        inps = [base.make_inputs(cache[w], P) for w in sub]
        Tmax = max(len(x["md"]) for x in inps)
        nsarr = build_ns(sub, EVD, DIST, Tmax)
        preds = run_guided(inps, nsarr, P, SEED, NSEED)
        for k, w in enumerate(sub):
            e = preds[k] - cache[w]["tvt_true"]; ef = e[np.isfinite(e)]
            errs.append(ef); pw[w] = float(np.sqrt(np.mean(ef ** 2)))
        print(f"  {min(c0+CHUNK,len(ALL))}/{len(ALL)} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(errs); rmse = float(np.sqrt(np.mean(e ** 2)))
    print(f"\n[v90.0] guided PF (BETA={BETA}) fold0 overall RMSE = {rmse:.4f}  (r0 fold0=8.03 と比較)")
    if os.environ.get("SAVEPW"):
        np.savez(PROJECT / "v90" / os.environ["SAVEPW"], **pw)
        print(f"[pw] saved {os.environ['SAVEPW']} ({len(pw)}井)")


if __name__ == "__main__":
    main()
