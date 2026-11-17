# -*- coding: utf-8 -*-
"""
v51 — NN-emission PF: PFの尤度に v50 学習emission を乗算(PFの「目」の交換)
==============================================================================
    w *= lk_raw(GRガウス) × exp(W_NN · sim(particle位置)) [× lka(pfAのみv38錨)]
sim = v50 capaug OOFエンコーダの cosine 類似(行×struct±45ft帯, 0.5ft格子)。
v38原則: 強度は外側定数 W_NN のみ。実測: r0 lag16 で W=0.02 → 9.12→7.44。
実行: PFS=pf_1,pf_2 WNNS=0.01,0.02 python v51/nnpf_v51.py
"""
import os, sys, time, glob, pickle, math
import numpy as np, pandas as pd, torch
warnings = __import__("warnings"); warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from pathlib import Path
P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART51 = P0 / "v51" / "artifacts_v51"
V26_DTYPE = torch.float32; V26_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLAMP_FT = None
SEED_V26 = 4423098; N_SEED = int(os.environ.get("NSEED", "16")); WELL_CHUNK = int(os.environ.get("CHUNK", "24"))
SMOOTH_LAG = int(os.environ.get("LAG", "16")); NW = int(os.environ.get("NW", "10000"))
GRID_STEP = 0.2
WNNS = [float(x) for x in os.environ.get("WNNS", "0.01,0.02").split(",")]
SIM_STEP = 0.5; NBAND = 90
SIM_FP = ART51 / "sim_rows_v51.pkl"

PF_LIB = {
    "pf_1": {"gr_power": 2.1947763195255123, "gr_sig_def": 114.11439615641156, "gr_sig_max": 32.79346108797473,
             "gr_sig_min": 7.495879205722816, "gr_sig_mult": 2.7482077023565075, "hgr_smooth_r": 3,
             "init_pos_std": 0.43824435375436277, "init_rate_std": 0.001840222365629066, "jump_prob": 0.002,
             "jump_std": 5.0, "likelihood_scale": 20.0, "mom": 0.998833416480357, "n_particles": 600,
             "pn": 0.006788762926604484, "rate_clip": 2.0, "resamp": 0.060642973044944336, "rp": 0.6864944651997384,
             "rr": 0.00015873788792956448, "tvt_clip_margin": 113.08092552242186, "tw_gr_smooth_r": 0,
             "vn": 0.0013871256161405515, "phys_sig": 0.05095402721954007, "use_phys": False},
    "pf_2": {"gr_sig_def": 71.37105679394953, "gr_sig_max": 44.7373292550486, "gr_sig_min": 6.520200142977573,
             "gr_sig_mult": 1.9000654781073127, "init_pos_std": 1.1039311766848976, "init_rate_std": 0.009213697911348712,
             "likelihood_scale": 18.584958539746772, "mom": 0.9994265753132063, "n_particles": 600,
             "pn": 0.011257078147676399, "resamp": 0.30834451379977984, "rp": 0.49711567429512854,
             "rr": 0.00031680690540158854, "tvt_clip_margin": 63.752292532272826, "vn": 0.0010650490363481595,
             "rate_clip": 0.0, "gr_power": 2.0, "hgr_smooth_r": 0, "tw_gr_smooth_r": 0, "jump_prob": 0.0,
             "jump_std": 10.0, "phys_sig": 0.05095402721954007, "use_phys": False},
    "pf_3": {"gr_power": 2.074149546524975, "gr_sig_def": 66.97913258656452, "gr_sig_max": 50.843656140023995,
             "gr_sig_min": 9.427442041004895, "gr_sig_mult": 2.4485083695548218, "hgr_smooth_r": 0,
             "init_pos_std": 2.3279966476603975, "init_rate_std": 0.03938758391370206, "jump_prob": 0.002,
             "jump_std": 5.0, "likelihood_scale": 14.890876838217093, "mom": 0.9999769474035675, "n_particles": 600,
             "pn": 0.04394815893193551, "rate_clip": 0.0, "resamp": 0.054606961294962585, "rp": 0.42322457711970995,
             "rr": 0.002148810696831248, "tvt_clip_margin": 91.4794941498565, "tw_gr_smooth_r": 0,
             "vn": 0.0006883646035942229, "phys_sig": 0.05095402721954007, "use_phys": False},
    "r0_seed32": {
            "n_particles": 400,
            "likelihood_scale": 10.536319667317654,
            "init_pos_std": 0.9331541486020234,
            "init_rate_std": 0.00047753793163756304,
            "mom": 0.9997911117425017,
            "vn": 0.0006297672017716754,
            "pn": 0.005014009140727263,
            "rp": 0.7627309890952053,
            "rr": 0.00045663430474523125,
            "resamp": 0.032049849143995945,
            "gr_sig_min": 11.865793233968713,
            "gr_sig_max": 44.98927964612316,
            "gr_sig_def": 119.25397464636981,
            "gr_sig_mult": 3.058318309943017,
            "tvt_clip_margin": 150.56111180234504,
            "rate_clip": 2.0,
            "gr_power": 1.7826896851395453,
            "hgr_smooth_r": 0,
            "tw_gr_smooth_r": 0,
            "jump_prob": 0.002,
            "jump_std": 5.0,
            "phys_sig": 0.013666550145707291,
            "use_phys": False
    },
    "r1_seed32": {
      "n_particles": 400,
      "likelihood_scale": 43.06423939744663,
      "init_pos_std": 1.472927959941933,
      "init_rate_std": 0.0006649111513262288,
      "mom": 0.9995184091067627,
      "vn": 0.0012160450940772792,
      "pn": 0.002809741394025188,
      "rp": 0.7070613341717283,
      "rr": 0.0007812869001227403,
      "resamp": 0.036600490411087865,
      "gr_sig_min": 6.324062861615175,
      "gr_sig_max": 29.32935945549601,
      "gr_sig_def": 83.2600219362552,
      "gr_sig_mult": 1.7565330745434495,
      "tvt_clip_margin": 177.98411545472226,
      "rate_clip": 2.5,
      "gr_power": 1.4643521376101742,
      "hgr_smooth_r": 0,
      "tw_gr_smooth_r": 0,
      "jump_prob": 0.002,
      "jump_std": 10.0,
      "phys_sig": 0.010953433688074062,
      "use_phys": False
    },
    "r4_seed32": {
      "n_particles": 400,
      "likelihood_scale": 6.129713574501556,
      "init_pos_std": 2.5405504337778564,
      "init_rate_std": 0.011938889141999285,
      "mom": 0.9992427162272104,
      "vn": 0.001784288208749022,
      "pn": 0.008007715585381402,
      "rp": 0.4461250058810184,
      "rr": 5.030203093680174e-05,
      "resamp": 0.051240562079163804,
      "gr_sig_min": 9.13765299425053,
      "gr_sig_max": 22.588917759650524,
      "gr_sig_def": 149.15869647937143,
      "gr_sig_mult": 2.2465411129565656,
      "tvt_clip_margin": 76.23311161880518,
      "rate_clip": 2.0,
      "gr_power": 3.1939521239252175,
      "hgr_smooth_r": 3,
      "tw_gr_smooth_r": 0,
      "jump_prob": 0.005,
      "jump_std": 10.0,
      "phys_sig": 0.6873032383656374,
      "use_phys": False
    }
}

import json as _json
_V38_PACK = _json.loads((P0 / "v38" / "artifacts_v38" / "v38_submit_pack.json").read_text())
PF_LIB["pfA"] = dict(_V38_PACK["params"])
V41M = float(_V38_PACK["anchor_mult"])
ANCHOR_BANKS = {"pfA"}
PFS = [x for x in os.environ.get("PFS", "r0_seed32").split(",")]

def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    if r <= 0: return s.to_numpy(np.float32)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    g = np.arange(tmin, tmax + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)

def build_smoother_inputs(hw, tw_tvt, tw_gr, P):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf, GRID_STEP)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    kn_tvtin = kn["TVT_input"].to_numpy(float)
    if len(kpos) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[kpos] - np.interp(kn_tvtin, tw_tvt, tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    return dict(md=ev["MD"].to_numpy(float), z=ev["Z"].to_numpy(float), gr=gr_sm[ev_pos].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(P["n_particles"]))

def build_sims():
    os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
    os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
    sys.path.insert(0, str(P0 / "v50"))
    import stageA_v50 as sa
    data = pickle.loads(sa.CACHE.read_bytes())
    wells = sorted(data.keys()); offs = np.arange(-NBAND, NBAND + 1)
    out = {}
    with torch.no_grad():
        for f in range(5):
            ck = torch.load(sa.ART / f"stageA_enccapaug_f{f}.pt", map_location=V26_DEVICE)
            encA = sa.Enc1D(8, sa.CH_ENC, sa.DIM, sa.DILS).to(V26_DEVICE); encA.load_state_dict(ck["encA"]); encA.eval()
            encB = sa.Enc1D(7, sa.CH_ENC, sa.DIM, sa.DILS).to(V26_DEVICE); encB.load_state_dict(ck["encB"]); encB.eval()
            for wn in wells[f::5]:
                c = data[wn]
                ev_idx = np.where(c["ev"])[0]
                a = encA(torch.from_numpy(c["Fh"]).to(V26_DEVICE))[ev_idx]
                b = encB(torch.from_numpy(c["Ft"]).to(V26_DEVICE))
                K = len(c["gg"]); gmin = c["gmin"]
                st = c["st"].astype(np.float64)
                ks = np.round((st - gmin) / SIM_STEP).astype(np.int64)
                cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
                okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
                sims = (a[:, None, :] * b[torch.tensor(cand, device=V26_DEVICE)]).sum(-1).cpu().numpy()
                sims = np.where(okc, sims, -1.0)
                out[wn] = dict(sim=sims.astype(np.float16), st=st.astype(np.float32))
            print(f"  sim fold{f} done", flush=True)
    SIM_FP.write_bytes(pickle.dumps(out, protocol=4))
    print(f"sim saved: {len(out)} wells")
    return out

SIMD = pickle.loads(SIM_FP.read_bytes()) if SIM_FP.exists() else build_sims()

def _core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, stq, sgq, sim, wmap, W_NN, use_anchor,
          P, N, device, gen, dtype=V26_DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
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
        if W_NN > 0:
            jj = torch.round((v - stq[:, i][:, None]) / SIM_STEP).long() + NBAND
            inb = (jj >= 0) & (jj <= 2 * NBAND)
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * NBAND))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lknn = torch.exp(W_NN * sv)
        else:
            lknn = torch.ones_like(w)
        if use_anchor:                                              # pfA: v38 row錨(mult=V41M)
            asig = (sgq[:, i][:, None] * V41M).clamp_min(1e-6)
            da = (v - stq[:, i][:, None]) / asig
            lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
            lknn = lknn * lka
        w_new = torch.where(obs[:, None], w * lk * lknn, w * lknn)
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
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                pts_s[:, i - L] = (w * buf[:, ptr, :]).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik

def run_smoother(inps, sims_w, sgs_w, P, seed, n_seeds, chunk, W_NN, use_anchor=False):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; wsub = sims_w[c0:c0 + chunk]; wq = len(sub)
        Tmax = max(len(x["md"]) for x in sub); Gmax = max(len(x["gg"]) for x in sub)
        md = torch.zeros((wq, Tmax)); z = torch.zeros((wq, Tmax))
        gr = torch.full((wq, Tmax), float("nan")); valid = torch.zeros((wq, Tmax), dtype=torch.bool)
        grid = torch.zeros((wq, Gmax)); glen = torch.zeros(wq, dtype=torch.long)
        vmin = torch.zeros(wq); step = torch.zeros(wq); gs = torch.zeros(wq)
        ls = torch.zeros(wq); ir = torch.zeros(wq)
        stq = torch.zeros((wq, Tmax)); simt = torch.full((wq, Tmax, 2 * NBAND + 1), -1.0, dtype=torch.float16)
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
        dev = V26_DEVICE
        to = lambda tt: tt.to(dev)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        wmap = torch.arange(wq, device=dev).repeat_interleave(S)
        pf, ps, ll = _core(rep(to(md)), rep(to(z)), rep(to(gr)), rep(to(valid)), rep(to(grid)),
                           to(glen).repeat_interleave(S), to(vmin).repeat_interleave(S), to(step).repeat_interleave(S),
                           to(gs).repeat_interleave(S), to(ls).repeat_interleave(S), to(ir).repeat_interleave(S),
                           rep(to(stq)), rep(to(sgq)), to(simt), wmap, W_NN, use_anchor,
                           {**P, "smooth_lag": SMOOTH_LAG}, N, dev, gen)
        ps = ps.view(wq, S, -1).double().cpu().numpy(); ll = ll.view(wq, S).cpu().numpy()
        for b in range(wq):
            T = len(sub[b]["md"]); pr = ps[b, :, :T]
            llw = ll[b]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                lkw = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lkw / max(ls_scale, 1e-6)); ssum = ww.sum()
                ww = ww / ssum if ssum > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out[c0 + b] = (ww[:, None] * pr).sum(0)
        if dev == "cuda": torch.cuda.synchronize()
    return out

if __name__ == "__main__":
    wells = sorted(SIMD.keys())[:NW]
    DATA = []
    for wn in wells:
        hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or (hw["TVT_input"].notna()).sum() < 10: continue
        DATA.append((wn, hw, tw, ev["TVT"].to_numpy(float)))
    print(f"wells {len(DATA)}  LAG {SMOOTH_LAG}  seeds {N_SEED}", flush=True)
    _cache = pickle.loads((P0 / "v50" / "artifacts_v50" / "cacheA_v50.pkl").read_bytes())
    SGD = {w: _cache[w]["sg"] for w in _cache}
    del _cache
    for pname in PFS:
        Praw = {k: v for k, v in PF_LIB[pname].items() if k != "name"}
        inps = []; keep = []
        for wn, hw, tw, tv in DATA:
            x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw)
            if x is not None: inps.append(x); keep.append((wn, tv))
        sims_w = [SIMD[k[0]] for k in keep]
        sgs_w = [SGD.get(k[0]) for k in keep]
        for W_NN in WNNS:
            t0 = time.time()
            preds = run_smoother(inps, sims_w, sgs_w, Praw, SEED_V26, N_SEED, WELL_CHUNK, W_NN,
                                 use_anchor=(pname in ANCHOR_BANKS))
            errs = {k[0]: (p[:len(k[1])] - k[1][:len(p)]).astype(np.float32) for p, k in zip(preds, keep)}
            e = np.concatenate(list(errs.values())); e = e[np.isfinite(e)]
            print(f"[{pname} lag{SMOOTH_LAG}] W_NN={W_NN:g}: overall={np.sqrt(np.mean(e**2)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if os.environ.get("SAVE", "0") == "1":
                np.savez_compressed(ART51 / f"nnpf_errs_{pname}_w{W_NN:g}.npz", **errs)
