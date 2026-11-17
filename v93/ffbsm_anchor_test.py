# -*- coding: utf-8 -*-
"""
v93 module: REAL FFBSm (Godsill-Doucet-West 2004 marginal smoother) anchor-preserving test.
================================================================================
Faithful self-contained port of v25/tune_ffbsi_r0_v25.py (ffbs_forward_well +
ffbsm_backward_well + reduce_cloud + ensemble_seed). PF = v93 r0_seed32 bank param
(identical values to v25 R0_PARAMS). CV only (true-TVT RMSE on train wells), no LB.

Experiment (overall pooled RMSE vs true TVT, first NW=60 train wells):
  1. forward        : forward-filter posterior mean (blend=0, no backward).
  2. lag192         : v93 fixed-lag smoother (pf.run_smoother_ext, smooth_lag=192)
                      = transfer-safe baseline (the fixed-lag proxy).
  3. FFBSm-full     : full backward marginal smoother, TRANS_INFLATE=32, blend=1,
                      CLAMP_FT=None (raw FFBS, no anchor clamp).
  4. anchor-FFBSm   : same FFBSm clamped to forward +/- band for band in [5,10,20,40] ft
                      (anchor-preserving: refine locally, cannot move datum past band).

Key question: does a tight clamp pull FFBSm back to forward (killing gain = datum-flip)
or retain most gain (gain is local, not a datum move)?

struct-free / GR-free-anchor OFF: r0 has use_anchor=False, so no bank anchor. The
"anchor" here = the forward/known-prefix constraint (CLAMP to forward).

Run: rogii_claude/.venv/Scripts/python.exe v93/ffbsm_anchor_test.py
Env: NW (default 60), NSEED (default 32), NBWD (default 128), TRANS_INFLATE (default 32).
"""
import io, sys, os, glob, time, math
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)  # cp932-safe
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "v93")
import pf_banks_v93 as pf

# ------------------------------ config ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
TR = pf.DATA_DIR / "train"

NW = int(os.environ.get("NW", "60"))
N_SEED = int(os.environ.get("NSEED", "32"))          # v25 used 128; 32 enough for the comparison
N_BWD = int(os.environ.get("NBWD", "128"))           # thin forward cloud before O(N^2) backward
TRANS_INFLATE = float(os.environ.get("TRANS_INFLATE", "32"))  # vn,pn x inflate for backward transition density
ANCHOR_BANDS = (5.0, 10.0, 20.0, 40.0)               # CLAMP_FT bands (ft)
SEED = 0
T_CAP = int(os.environ.get("TCAP", "20000"))         # skip pathologically long wells (memory guard); none expected

P0 = pf.bank_param("r0_seed32")   # keys already match v25 FFBSm engine expectations (n_particles..use_phys)


# ================= FFBS forward (1 well, S seeds, store particle cloud) =================
# --- verbatim port of v25 ffbs_forward_well ---
def ffbs_forward_well(inp, P, seed, S, device, dtype=DTYPE):
    """Forward physics PF over S seeds; store particle cloud (pre-resample) at each step.
    Returns Xpos,Xrate,Wst (S,T,N), DM (T,), loglik (S,), z (T,)."""
    N = int(P["n_particles"]); T = len(inp["md"])
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]

    md = torch.from_numpy(inp["md"].astype("float32")).to(device)
    z = torch.from_numpy(inp["z"].astype("float32")).to(device)
    gr = torch.from_numpy(inp["gr"].astype("float32")).to(device)
    grid = torch.from_numpy(inp["gg"].astype("float32")).to(device); G = grid.numel()
    vmin = float(inp["gmin"]); step = float(inp["gst"]); gs = float(inp["gs"])
    ls = float(inp["ls"]); ir = float(inp["ir"])

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    def rn(): return torch.randn((S, N), generator=gen, device=device, dtype=dtype)
    pos = ls + IS * rn(); rate = ir + IR * rn()
    w = torch.full((S, N), 1.0 / N, device=device, dtype=dtype)
    loglik = torch.zeros(S, device=device, dtype=torch.float64)
    Xpos = torch.empty((S, T, N), device=device, dtype=dtype)
    Xrate = torch.empty((S, T, N), device=device, dtype=dtype)
    Wst = torch.empty((S, T, N), device=device, dtype=dtype)
    DM = torch.empty(T, device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (G - 1) * step + CLIP
    g0 = grid[0]; glast = grid[G - 1]
    zero = torch.zeros((), device=device, dtype=dtype); arN = torch.arange(N, device=device, dtype=dtype)
    pm = float(inp["md"][0]) - 1.0

    for i in range(T):
        cur = float(inp["md"][i]); dm = max(cur - pm, 1.0); DM[i] = dm
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((S, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[i]; tvt = (pos_n - zi).clamp(tvt_lo, tvt_hi); pos_n = tvt + zi
        pos = pos_n; rate = rate_n
        gri = gr[i]; obs = not bool(torch.isnan(gri))
        v = pos - zi; ii = (v - vmin) / step; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (G - 1)
        i0c = i0.clamp(0, G - 2); t = ii - i0c.to(dtype)
        gA = grid[i0c]; gB = grid[i0c + 1]
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri - eg) / gs).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        if obs:
            avg = (w * lk).sum(1)
            loglik = loglik + torch.log(avg.clamp_min(1e-300).double())
        if USE_P:
            dphys = (rate - ir) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = (w * lk * lkp) if obs else (w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))

        # ---- FFBS: store pre-resample cloud + weights ----
        Xpos[:, i, :] = pos; Xrate[:, i, :] = rate; Wst[:, i, :] = w

        ne = (w * w).sum(1); need = (1.0 / ne) < (RESAMP * N)
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((S, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pm = cur

    return Xpos, Xrate, Wst, DM, loglik, z


# --- verbatim port of v25 ffbsm_backward_well (the priority: the FFBSm backward pass) ---
def ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, inflate, device, dtype=DTYPE):
    """FFBSm marginal backward recursion. Returns smoothed posterior-mean TVT per seed (S,T).
    transition f(x_{t+1}|x_t) = N(rate'; a*rate, (vn*inflate)^2) * N(pos'; pos+rate'*dm, (pn*inflate)^2)."""
    S, T, N = Xpos.shape
    ALPHA = P["mom"]; RN = max(P["vn"] * inflate, 1e-9); PN = max(P["pn"] * inflate, 1e-9)
    sm = torch.empty((S, T), device=device, dtype=dtype)
    eps = 1e-30

    wb = Wst[:, T - 1, :].clone()                              # (S,N) smoothing weight = filter weight at T
    wb = wb / wb.sum(1, keepdim=True).clamp_min(eps)
    sm[:, T - 1] = (wb * (Xpos[:, T - 1, :] - z[T - 1])).sum(1)

    for t in range(T - 2, -1, -1):
        dm = DM[t + 1]
        r_i = Xrate[:, t, :]; p_i = Xpos[:, t, :]               # (S,N_i)
        r_j = Xrate[:, t + 1, :]; p_j = Xpos[:, t + 1, :]       # (S,N_j)
        A = (r_j[:, None, :] - ALPHA * r_i[:, :, None]) / RN     # (S,N_i,N_j)
        Bm = (p_j[:, None, :] - p_i[:, :, None] - r_j[:, None, :] * dm) / PN
        logf = -0.5 * (A * A + Bm * Bm)                          # (S,N_i,N_j)
        m = logf.amax(dim=1, keepdim=True)                      # (S,1,N_j) stabilize (ratio cancels)
        f = torch.exp(logf - m)
        w_t = Wst[:, t, :]                                      # (S,N_i)
        vj = (w_t[:, :, None] * f).sum(1).clamp_min(eps)        # (S,N_j)
        ratio = wb / vj                                        # (S,N_j)
        w_sm = w_t * (f * ratio[:, None, :]).sum(2)             # (S,N_i)
        w_sm = w_sm / w_sm.sum(1, keepdim=True).clamp_min(eps)
        wb = w_sm
        sm[:, t] = (wb * (p_i - z[t])).sum(1)
    return sm


# --- verbatim port of v25 reduce_cloud ---
def reduce_cloud(Xpos, Xrate, Wst, n_bwd, gen, device, dtype=DTYPE):
    """Systematic-resample forward cloud N->n_bwd per step (weights become uniform).
    Drops backward from O(N^2) to O(n_bwd^2). Forward baseline computed BEFORE this (unchanged)."""
    S, T, N = Xpos.shape
    cumw = torch.cumsum(Wst, dim=2)
    ar = torch.arange(n_bwd, device=device, dtype=dtype)
    u0 = torch.rand((S, T, 1), generator=gen, device=device, dtype=dtype) * (1.0 / n_bwd)
    u = u0 + ar / n_bwd
    idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
    Xp = torch.gather(Xpos, 2, idx); Xr = torch.gather(Xrate, 2, idx)
    Wr = torch.full((S, T, n_bwd), 1.0 / n_bwd, device=device, dtype=dtype)
    return Xp, Xr, Wr


# --- verbatim port of v25 ensemble_seed ---
def ensemble_seed(vals_ST, loglik_S, P):
    """(S,T) -> (T,) via seed likelihood weighting (same weighting as run_smoother_ext)."""
    S = vals_ST.shape[0]; llw = loglik_S.detach().cpu().numpy()
    if np.isfinite(llw).any():
        mx = np.nanmax(llw[np.isfinite(llw)])
        lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
        ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
        ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
    else:
        ww = np.full(S, 1.0 / S)
    ww_t = torch.from_numpy(ww.astype("float32")).to(vals_ST.device)
    return (ww_t[:, None] * vals_ST).sum(0).double().cpu().numpy()


def rmse_cat(err_list):
    e = np.concatenate(err_list); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# ================================ data ================================
def load_wells():
    wids = sorted({os.path.basename(q).split("__")[0]
                   for q in glob.glob(str(TR / "*__horizontal_well.csv"))})[:NW]
    inps, trues, kept = [], [], []
    for w in wids:
        hw = pd.read_csv(TR / f"{w}__horizontal_well.csv")
        if "TVT" not in hw.columns:
            continue
        tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        m = np.isfinite(tt) & np.isfinite(tg)
        if m.sum() < 8:
            continue
        x = pf.build_smoother_inputs(hw, tt[m], tg[m], P0)  # keys == v25 make_inputs output
        if x is None:
            continue
        pf.attach_anchor(x, w, P0["_physics"])              # r0 physics=False -> ancs=1e9 (anchor OFF)
        ev = hw[hw["TVT_input"].isna()]
        inps.append(x); trues.append(ev["TVT"].to_numpy(float)); kept.append(w)
    return kept, inps, trues


# ================================ main ================================
def main():
    print(f"device={DEVICE}  PF=r0_seed32 (use_phys={P0['use_phys']}, use_anchor={P0['use_anchor']}, "
          f"N={P0['n_particles']})  n_seed={N_SEED}  N_BWD={N_BWD}  TRANS_INFLATE={TRANS_INFLATE}", flush=True)
    kept, inps, trues = load_wells()
    print(f"usable wells: {len(kept)} / requested {NW}", flush=True)

    # ---------- lag192 reference (v93 fixed-lag smoother on the SAME wells) ----------
    print("\n[lag192] v93 fixed-lag smoother (smooth_lag=192) ...", flush=True)
    Pl = dict(P0); Pl["smooth_lag"] = 192
    t0 = time.time()
    res192 = pf.run_smoother_ext(inps, Pl, seed=99999, n_seeds=N_SEED, chunk=30, w_nn=0.0)
    lag192_pred = [r["mean"] for r in res192]
    lag192_rmse = rmse_cat([np.asarray(lag192_pred[i]) - trues[i] for i in range(len(inps))])
    print(f"[lag192] overall RMSE = {lag192_rmse:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------- FFBS forward + real FFBSm backward (per well) ----------
    print("\n[FFBSm] forward filter + real backward marginal smoother (per well) ...", flush=True)
    EF, ESfull = [], []                                  # forward errs, ffbs-full errs
    EA = {b: [] for b in ANCHOR_BANDS}                   # anchor-clamped errs per band
    accF = [0.0, 0]; accS = [0.0, 0]; accA = {b: [0.0, 0] for b in ANCHOR_BANDS}
    skipped = []
    max_mem = 0.0
    gen_bwd = torch.Generator(device=DEVICE); gen_bwd.manual_seed(SEED + 987654321)
    t0 = time.time()
    for k, (wid, inp, true) in enumerate(zip(kept, inps, trues)):
        T = len(inp["md"])
        if T > T_CAP:
            skipped.append((wid, T)); print(f"[{k+1}/{len(kept)}] {wid[:8]} T={T} SKIPPED (>T_CAP={T_CAP})", flush=True); continue
        Xpos, Xrate, Wst, DM, loglik, z = ffbs_forward_well(inp, P0, SEED, N_SEED, DEVICE)
        # forward posterior mean (full particles, unchanged baseline)
        fwd_ST = (Wst * (Xpos - z[None, :, None])).sum(2)     # (S,T)
        fwd = ensemble_seed(fwd_ST, loglik, P0)
        ef = fwd - true; EF.append(ef)
        s, c = _sumsq(ef); accF[0] += s; accF[1] += c

        # thin cloud for O(N^2) backward
        if N_BWD is not None and N_BWD < Xpos.shape[2]:
            Xpos, Xrate, Wst = reduce_cloud(Xpos, Xrate, Wst, int(N_BWD), gen_bwd, DEVICE)

        sm_ST = ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P0, TRANS_INFLATE, DEVICE)
        sm = ensemble_seed(sm_ST, loglik, P0)                 # FFBSm full (CLAMP_FT=None)
        es = sm - true; ESfull.append(es)
        s2, c2 = _sumsq(es); accS[0] += s2; accS[1] += c2

        # anchor-preserving: clamp FFBSm to forward +/- band
        for b in ANCHOR_BANDS:
            bad = np.abs(sm - fwd) > b
            smb = np.where(bad, fwd, sm)
            eb = smb - true; EA[b].append(eb)
            s3, c3 = _sumsq(eb); accA[b][0] += s3; accA[b][1] += c3

        if DEVICE == "cuda":
            max_mem = max(max_mem, torch.cuda.max_memory_allocated() / 1e9)
        del Xpos, Xrate, Wst, fwd_ST, sm_ST
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        rF = math.sqrt(accF[0] / accF[1]) if accF[1] else float("nan")
        rS = math.sqrt(accS[0] / accS[1]) if accS[1] else float("nan")
        el = time.time() - t0; eta = el / (k + 1) * (len(kept) - (k + 1))
        print(f"[{k+1}/{len(kept)}] {wid[:8]} T={T} | run fwd={rF:.3f} ffbs={rS:.3f} | "
              f"{el:.0f}s ETA {eta/60:.1f}m", flush=True)

    fwd_rmse = rmse_cat(EF)
    ffbs_rmse = rmse_cat(ESfull)
    anc_rmse = {b: rmse_cat(EA[b]) for b in ANCHOR_BANDS}

    # ================================ report ================================
    print("\n" + "=" * 60)
    print(f"RESULT  (r0_seed32, {len([w for w in kept if w])} wells minus skipped, n_seed={N_SEED}, "
          f"N_BWD={N_BWD}, TRANS_INFLATE={TRANS_INFLATE})")
    print("=" * 60)
    print(f"  forward (blend=0)        = {fwd_rmse:.4f}")
    print(f"  lag192 (fixed-lag proxy) = {lag192_rmse:.4f}   (dvs fwd {lag192_rmse-fwd_rmse:+.4f})")
    print(f"  FFBSm-full (clamp=None)  = {ffbs_rmse:.4f}   (dvs fwd {ffbs_rmse-fwd_rmse:+.4f})")
    for b in ANCHOR_BANDS:
        print(f"  anchor-FFBSm band=+/-{b:>4.0f}ft = {anc_rmse[b]:.4f}   "
              f"(dvs fwd {anc_rmse[b]-fwd_rmse:+.4f})")
    print("=" * 60)
    # interpretation flags
    print(f"  FFBSm-full beats forward? {'YES' if ffbs_rmse < fwd_rmse else 'NO'}")
    print(f"  FFBSm-full beats lag192?  {'YES' if ffbs_rmse < lag192_rmse else 'NO'}")
    if ffbs_rmse < fwd_rmse:
        gain = fwd_rmse - ffbs_rmse
        b40 = anc_rmse[40.0]; retained = (fwd_rmse - b40) / gain if gain > 1e-9 else float("nan")
        print(f"  gain (fwd - ffbs)  = {gain:+.4f}")
        print(f"  band=+/-40 retains {retained*100:.0f}% of the gain (>~80% => local, not datum-flip)")
    if DEVICE == "cuda":
        print(f"  peak GPU mem = {max_mem:.2f} GB")
    if skipped:
        print(f"  SKIPPED wells (T>{T_CAP}): {skipped}")
    else:
        print(f"  skipped wells: none")


def _sumsq(err):
    e = err[np.isfinite(err)]
    return float(np.sum(e ** 2)), int(e.size)


if __name__ == "__main__":
    main()
