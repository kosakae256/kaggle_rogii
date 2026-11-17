# -*- coding: utf-8 -*-
"""v49 案①(実self-PF版): self疑似typewell の範囲外を「selfに較正したregional」で接続(graft較正)。
実self-PF機構(v41 smoother core をコピー)で、PFパラメータ固定・typewell だけ orig↔ext を差し替え比較。
pf_1 / pfA 両方。全773坑井 overall RMSE。実行: NW=10000 python v49/self_ext_pf_v49.py
"""
import os, time, glob, pickle, numpy as np, pandas as pd, torch
warnings=__import__("warnings"); warnings.filterwarnings("ignore"); np.seterr(all="ignore")
P0=__import__("pathlib").Path(r"C:\Users\kosaka256\Documents\rogii_claude"); TRAIN=P0/"rogii-wellbore-geology-prediction"/"train"
V26_DTYPE=torch.float32; V26_DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CLAMP_FT=None; V41_ANCHOR_MULT=30.0; SELF_MIX_W=1.0
SEED_V26=4423098; N_SEED=int(os.environ.get("NSEED","16")); WELL_CHUNK=120
SMOOTH_LAG=int(os.environ.get("LAG","192")); NW=int(os.environ.get("NW","10000"))
GRID_STEP=float(os.environ.get("GSTEP","0.2"))   # ★typewell格子の刻み(細かくすると self-PF が細かいTVTを追える)
# --- PFパラメータ(raw名。smoother core が直接使用)---
PF_LIB={
 "pf_1":{"gr_power":2.1947763195255123,"gr_sig_def":114.11439615641156,"gr_sig_max":32.79346108797473,
   "gr_sig_min":7.495879205722816,"gr_sig_mult":2.7482077023565075,"hgr_smooth_r":3,"init_pos_std":0.43824435375436277,
   "init_rate_std":0.001840222365629066,"jump_prob":0.002,"jump_std":5.0,"likelihood_scale":20.0,"mom":0.998833416480357,
   "n_particles":600,"pn":0.006788762926604484,"rate_clip":2.0,"resamp":0.060642973044944336,"rp":0.6864944651997384,
   "rr":0.00015873788792956448,"tvt_clip_margin":113.08092552242186,"tw_gr_smooth_r":0,"vn":0.0013871256161405515,
   "phys_sig":0.05095402721954007,"use_phys":False},
 "pfA":{"n_particles":600,"likelihood_scale":23.2846346555713,"init_pos_std":1.5702867537220184,"init_rate_std":0.0014487008327630194,
   "mom":0.9995602288940161,"vn":0.0010067691818459045,"pn":0.01314833751510345,"rp":0.20784122815628556,"rr":0.0012067484628367393,
   "resamp":0.24861696525462024,"gr_sig_min":11.95105315170872,"gr_sig_max":48.33000415066859,"gr_sig_def":90.31457587495258,
   "gr_sig_mult":2.8754881845576428,"tvt_clip_margin":67.26811605160809,"rate_clip":0.0,"gr_power":2.5310381370100004,
   "hgr_smooth_r":1,"tw_gr_smooth_r":0,"jump_prob":0.002,"jump_std":10.0,"phys_sig":4.543603757537785,"use_phys":False},
 "r0_seed32":{"n_particles":400,"likelihood_scale":10.536319667317654,"init_pos_std":0.9331541486020234,
   "init_rate_std":0.00047753793163756304,"mom":0.9997911117425017,"vn":0.0006297672017716754,"pn":0.005014009140727263,
   "rp":0.7627309890952053,"rr":0.00045663430474523125,"resamp":0.032049849143995945,"gr_sig_min":11.865793233968713,
   "gr_sig_max":44.98927964612316,"gr_sig_def":119.25397464636981,"gr_sig_mult":3.058318309943017,"tvt_clip_margin":150.56111180234504,
   "rate_clip":2.0,"gr_power":1.7826896851395453,"hgr_smooth_r":0,"tw_gr_smooth_r":0,"jump_prob":0.002,"jump_std":5.0,
   "phys_sig":0.013666550145707291,"use_phys":False},
}

def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(float(fb))
    if r <= 0:
        return s.to_numpy(np.float32)
    return s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def build_smoother_inputs(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力を hw/tw と生PFパラメータから作る(v9の _smooth_radius_values/_grid を再利用)。無効なら None。"""
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
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


def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=V26_DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
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
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
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
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        # ★v41: struct錨(v38 row錨, mult=V41_ANCHOR_MULT)
        asig = (ancs[:, i][:, None] * V41_ANCHOR_MULT).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
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
                old = buf[:, ptr, :]; pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def _v26_pad(inps, device, dtype=V26_DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)   # v41錨
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        if "anc" in x:
            anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
            if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
            ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs))


def run_smoother_ext(inps, P, seed, n_seeds, chunk):
    """各 well: smoothed(seed尤度加重) と std(smoothedのseed間加重std)。forwardはCLAMP基準。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _v26_pad(sub, V26_DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=V26_DEVICE); gen.manual_seed(seed)
        pf, ps, ll = _v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), P, N, V26_DEVICE, gen)
        pf = pf.view(w, S, -1).double().cpu().numpy(); ps = ps.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk / max(ls_scale, 1e-6)); s = ww.sum(); ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            smean = (ww[:, None] * ps[j, :, :T]).sum(0); fmean = (ww[:, None] * pf[j, :, :T]).sum(0)
            sstd = np.sqrt(np.maximum((ww[:, None] * (ps[j, :, :T] - smean[None, :]) ** 2).sum(0), 0.))
            if CLAMP_FT is not None:
                bad = np.abs(smean - fmean) > CLAMP_FT
                if bad.any(): smean = np.where(bad, fmean, smean)
            out[c0 + j] = dict(mean=smean.astype(np.float32), std=sstd.astype(np.float32))
    if V26_DEVICE == "cuda": torch.cuda.synchronize()
    return out

def _bin_grid(t, g, gmin, step, G):
    """(TVT, GR) を grid の TVT ビンに median 集約(疑似typewell)。無データビン=NaN。"""
    m = np.isfinite(t) & np.isfinite(g); t, g = t[m], g[m]
    idx = np.round((t - gmin) / step).astype(int); ok = (idx >= 0) & (idx < G); idx, gg = idx[ok], g[ok]
    out = np.full(G, np.nan)
    if len(gg):
        s = pd.Series(gg).groupby(idx).median(); out[s.index.to_numpy()] = s.to_numpy()
    return out


def build_inputs_self(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力(tw)に、自分の既知prefix GR(TVT_input) を疑似typewell化して gg に上書き。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]
    base = dict(base); base["gg"] = gg; return base


def build_inputs_self_ext(hw, tw_tvt, tw_gr, P):
    """①: build_inputs_self と同じだが、self範囲外の regional を self に較正(scale/shift)して接続。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh)
    if have.sum() >= 6:
        rg = gg[have]; sh = gh[have]; A = np.vstack([rg, np.ones_like(rg)]).T
        try: a, b = np.linalg.lstsq(A, sh, rcond=None)[0]
        except Exception: a, b = 1.0, 0.0
        if (not np.isfinite(a)) or a <= 0.2 or a > 5: a, b = 1.0, 0.0
        gg_cal = a * gg + b
    else:
        gg_cal = gg
    gg2 = gg_cal.copy(); gg2[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]
    base = dict(base); base["gg"] = gg2; return base


STRUCT_keys=sorted(pickle.loads((P0/"v38"/"artifacts_v38"/"struct_prior_train.pkl").read_bytes()).keys())
wells=STRUCT_keys[:NW]
def load(wn):
    hw=pd.read_csv(TRAIN/f"{wn}__horizontal_well.csv"); tw=pd.read_csv(TRAIN/f"{wn}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns: return None
    ev=hw[hw["TVT_input"].isna()]; kn=hw[hw["TVT_input"].notna()]
    if len(ev)==0 or len(kn)<10: return None
    return dict(hw=hw, tw_tvt=tw["TVT"].to_numpy(float), tw_gr=tw["GR"].to_numpy(float), true=ev["TVT"].to_numpy(float))
DATA=[d for d in (load(w) for w in wells) if d is not None]
print(f"wells {len(DATA)}  dev {V26_DEVICE}  N_SEED {N_SEED}  LAG {SMOOTH_LAG}")

def run_pf(builder, Praw):
    Pk={**Praw, "smooth_lag": SMOOTH_LAG}
    inps=[]; keep=[]
    for d in DATA:
        x=builder(d["hw"], d["tw_tvt"], d["tw_gr"], Pk)
        if x is not None: inps.append(x); keep.append(d)
    outs=run_smoother_ext(inps, Pk, SEED_V26, N_SEED, WELL_CHUNK)
    errs=[]
    for o,d in zip(outs,keep):
        pred=o["mean"][:len(d["true"])]; errs.append(pred-d["true"][:len(pred)])
    e=np.concatenate(errs); e=e[np.isfinite(e)]; return float(np.sqrt(np.mean(e**2)))

STEPS=[float(x) for x in os.environ.get("STEPS","0.2,0.1").split(",")]
print(f"\n=== v49: self-PF typewell格子刻み sweep(全{len(DATA)}坑井, LAG={SMOOTH_LAG})===")
for pname in ["pf_1","r0_seed32","pfA"]:
    row=[]
    for st in STEPS:
        globals()["GRID_STEP"]=st; r=run_pf(build_inputs_self, PF_LIB[pname]); row.append(f"{st}:{r:.3f}")
    print(f"  [{pname}]  "+"  ".join(row),flush=True)
