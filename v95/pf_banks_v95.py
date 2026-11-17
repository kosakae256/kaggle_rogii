# -*- coding: utf-8 -*-
"""v93 module1: 6バンク平滑PFエンジン(GPU固定ラグ smoother)。

v52 create の smooth-PF 部分をクリーン再構築。★v93 の唯一の実変更:
  - pfA バンクの錨を「過去struct(v38/v66)」から「GRフリー錨(v91 OOF, sig=9ft較正)」へ差替。
  - 錨強度 anchor_mult は global 定数でなく バンクparam P["anchor_mult"] から取る(pfA=20)。
過去struct は一切読まない([[no-struct-directive]])。錨源は v93/artifacts_v93 のみ。

構成(v52 と数値一致させる要素):
  - build_smoother_inputs / build_inputs_self / build_inputs_nbr : PF入力(tw/self/nbr疑似typewell)
  - attach_anchor : GRフリー錨(anc/ancs)を添付。非physicsバンクは ancs=1e9(錨OFF)
  - _smoother_core : 固定ラグ(L=smooth_lag)平滑PF。GR尤度×phys×錨×NN-emission
  - run_smoother_ext : seed尤度加重で smoothed平均/std を返す
NN-emission(sim/帯中心 st)は create 側が入力dictに _sim/_st を添付する形で受ける(module2が生成)。
"""
import os, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJ = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
DATA_DIR = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
ART = Path(os.environ.get("ROGII_ART95", str(PROJ / "v95" / "artifacts_v95")))  # ★v95: config/錨をv93からコピー済(PF param同一)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---- config(6バンクparam) & GRフリー錨 ----
_CFG = json.loads((ART / "pf_banks_config.json").read_text(encoding="utf-8"))
BANK_ORDER = _CFG["bank_order"]                    # [pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA]
PHYSICS_BANKS = set(_CFG["physics_banks"])         # {pfA}
BANK_PARAMS = _CFG["params"]                       # dict[bank] -> raw param dict
W_NN_BANK = _CFG["w_nn_bank"]                       # {pfA:0.01}
W_NN_DEFAULT = float(_CFG["w_nn_default"])          # 0.02
SMOOTH_LAG = int(_CFG["smooth_lag"])               # 16
N_SEED = int(_CFG["n_seed"])                        # 32

_ANCHOR_PKL = ART / _CFG.get("anchor_pkl", "grfree_anchor_train.pkl")
if os.environ.get("V93_ANCHOR_PKL"):
    _ANCHOR_PKL = Path(os.environ["V93_ANCHOR_PKL"])
_ANCHOR = pickle.loads(_ANCHOR_PKL.read_bytes())   # {wid: {tvt, sig}}
print(f"[v93 pf_banks] 錨={_ANCHOR_PKL.name} {len(_ANCHOR)}井 / banks={BANK_ORDER} / smooth_lag={SMOOTH_LAG}")

# ---- NN-emission 帯(create/module2と共有する幾何定数) ----
SIM_STEP_NN = 0.5
NBAND_NN = 90                                       # ±45ft / 0.5ft = 90

# ---- self/nbr/Z傾斜 定数(v52同値) ----
SELF_MIX_W = 1.0
NBR_MIX_W = 1.0
ZGRAD_R = 25
K_MAX = 3
MAX_DIST = 1500.0
NEED_REFS = 2


def bank_param(bank):
    """バンク生paramに physics/w_nn を付けて返す(create が smoother に渡す P)。"""
    P = dict(BANK_PARAMS[bank])
    P["name"] = bank
    P.setdefault("smooth_lag", SMOOTH_LAG)
    P["_physics"] = bank in PHYSICS_BANKS
    P["_w_nn"] = float(W_NN_BANK.get(bank, W_NN_DEFAULT))
    return P


# ==================== 小物(v52 と同一) ====================
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    if r <= 0:
        return s.to_numpy(np.float32)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _bin_grid(t, g, gmin, step, G):
    """(TVT, GR) を grid の TVT ビンに median 集約(疑似typewell)。無データビン=NaN。"""
    m = np.isfinite(t) & np.isfinite(g); t, g = t[m], g[m]
    idx = np.round((t - gmin) / step).astype(int); ok = (idx >= 0) & (idx < G); idx, gg = idx[ok], g[ok]
    out = np.full(G, np.nan)
    if len(gg):
        s = pd.Series(gg).groupby(idx).median(); out[s.index.to_numpy()] = s.to_numpy()
    return out


# ==================== PF入力(tw / self / nbr) ====================
def build_smoother_inputs(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力を hw/tw と生PFパラメータから作る。eval無しは None。"""
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
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
    gs = gs * float(os.environ.get("GS_SCALE", "1.0"))     # ★post-clip の GR尤度σ 広げ(gs×1.30 検証。既定1.0=無効)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    return dict(md=ev["MD"].to_numpy(float), z=ev["Z"].to_numpy(float), gr=gr_sm[ev_pos].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(P["n_particles"]))


def build_inputs_self(hw, tw_tvt, tw_gr, P):
    """tw入力の gg を、自分の既知prefix GR(TVT_input) 疑似typewell で上書き。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh)
    gg[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]
    base = dict(base); base["gg"] = gg
    return base


def build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs):
    """tw入力の gg を、近傍train坑井の GR(TVT) 疑似typewell で上書き。refs無し=tw fallback。
       refs = [(gr_array, tvt_array), ...](module3 の近傍選択が供給)。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    if refs:
        rt = np.concatenate([r[1] for r in refs])
        rg = np.concatenate([_smooth_radius_values(r[0], float(np.nanmean(r[0])), P["hgr_smooth_r"]).astype(float) for r in refs])
        gh = _bin_grid(rt, rg, base["gmin"], base["gst"], len(base["gg"]))
        gg = base["gg"].copy(); have = np.isfinite(gh)
        gg[have] = NBR_MIX_W * gh[have] + (1 - NBR_MIX_W) * gg[have]
        base = dict(base); base["gg"] = gg
    return base


def build_inputs_self_graft(hw, tw_tvt, tw_gr, P):
    """★v95新規: self(自分のprefix GR)を tw格子に substitute し、prefix範囲外(データ無しビン)を
       tw を self較正(self≈a·tw+b, a∈[0.2,5])で外挿。self被覆を広げる(v49 build_inputs_self_ext 由来)。
       較正係数 a,b・有効フラグ・外挿割合を base['_graft'] にメタ格納(provenance用)。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None:
        return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh)
    a, b, valid = 1.0, 0.0, 0
    if have.sum() >= 8:
        gg[have] = gh[have]                                       # prefix範囲=self実測
        tw_ov = base["gg"][have]; self_ov = gh[have]
        A = np.c_[tw_ov, np.ones(len(tw_ov))]
        try:
            coef, *_ = np.linalg.lstsq(A, self_ov, rcond=None); a, b = float(coef[0]), float(coef[1])
        except Exception:
            a, b = 1.0, 0.0
        if 0.2 <= a <= 5.0 and np.isfinite(a) and np.isfinite(b):
            gg[~have] = a * base["gg"][~have] + b                 # 範囲外=self較正したtwで外挿
            valid = 1
    base = dict(base); base["gg"] = gg
    base["_graft"] = dict(a=float(a), b=float(b), valid=int(valid),
                          cover=float(have.mean()), extrap_frac=float((~have).mean()), nprefix=int(len(kpos)))
    return base


def z_gradient_eval(hw, r=ZGRAD_R):
    """Z傾斜 dZ/dMD を移動平均(断層影響回避)し eval 行のみ返す。"""
    ev = hw["TVT_input"].isna().to_numpy()
    md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float)
    with np.errstate(all="ignore"):
        g = np.gradient(z, md)
    g = pd.Series(g).rolling(2 * int(r) + 1, center=True, min_periods=1).mean().to_numpy()
    return g[ev].astype(np.float32)


# ==================== GRフリー錨 添付 ====================
def attach_anchor(x, wid, physics):
    """PF入力に GRフリー錨(anc/ancs)を添付。physics=False / 錨欠損 / 行数不一致 は錨OFF(ancs=1e9)。
       ★過去struct は使わない。錨源は _ANCHOR(v93 GRフリー, sig=9ft較正)のみ。"""
    n = len(x["md"]); sp = _ANCHOR.get(wid)
    x["_wid"] = wid
    if physics and sp is not None and len(sp["tvt"]) == n:
        x["anc"] = np.asarray(sp["tvt"], float); x["ancs"] = np.asarray(sp["sig"], float)
    else:
        x["anc"] = np.zeros(n); x["ancs"] = np.full(n, 1e9)
    return x


# ==================== GPU 固定ラグ 平滑PF 本体 ====================
def _smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir,
                   anc, ancs, amul, stq, sim, wmap, w_nn, P, N, device, gen, dtype=DTYPE):
    """v52 _v26_smoother_core と数値一致。★v93変更: 錨強度は amul(=P['anchor_mult'])。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

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
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
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
        # ★v93: GRフリー行錨(mult = amul = P['anchor_mult'])。非physicsは ancs=1e9 で無効化。
        asig = (ancs[:, i][:, None] * amul).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        # NN-emission(学習類似度を温度 w_nn で乗算)。帯中心 stq は GRフリー錨tvt(module2)。
        if w_nn > 0:
            jj = torch.round((v - stq[:, i][:, None]) / SIM_STEP_NN).long() + NBAND_NN
            inb = (jj >= 0) & (jj <= 2 * NBAND_NN)
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * NBAND_NN))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lka = lka * torch.exp(w_nn * sv)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
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


def _pad(inps, device, dtype=DTYPE):
    """可変長wellをバッチテンソルへ。anc/ancs/stq/sim(NN)も詰める。sim帯は入力dictの _sim/_st から。"""
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    stq = torch.zeros((W, Tmax), dtype=dtype)
    simt = torch.full((W, Tmax, 2 * NBAND_NN + 1), -1.0, dtype=torch.float16)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        _sim = x.get("_sim"); _st = x.get("_st")
        if _sim is not None and _st is not None:
            _Ts = min(Tn, len(_st))
            stq[b, :_Ts] = torch.from_numpy(np.asarray(_st[:_Ts], np.float32))
            if _Ts < Tmax:
                stq[b, _Ts:] = stq[b, _Ts - 1]
            simt[b, :_Ts] = torch.from_numpy(np.asarray(_sim[:_Ts], np.float16))
        if "anc" in x:
            anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
            if Tn < Tmax:
                anc[b, Tn:] = anc[b, Tn - 1]
            ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax:
            md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax:
            grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                stq=to(stq), simt=to(simt))


def run_smoother_ext(inps, P, seed, n_seeds, chunk, w_nn=0.0, capture=False):
    """各 well: smoothed(seed尤度加重平均)と std(seed間加重std)を返す。
       ★v93: 錨強度 amul は P['anchor_mult'](pfA=20)。非physicsバンクは ancs=1e9 で錨無効。
       ★capture=True: v34モードゲート用に per-seed 平滑ps(S,T)/forward pfw/ll/ww も out に付与(RAM増)。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"]); amul = float(P.get("anchor_mult", 1.0))
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _pad(sub, DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(seed)
        _wmap = torch.arange(w, device=DEVICE).repeat_interleave(S)
        pf, ps, ll = _smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), amul, rep(pad["stq"]), pad["simt"], _wmap, w_nn, P, N, DEVICE, gen)
        pf = pf.view(w, S, -1).double().cpu().numpy(); ps = ps.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk / max(ls_scale, 1e-6)); s = ww.sum()
                ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            smean = (ww[:, None] * ps[j, :, :T]).sum(0)
            sstd = np.sqrt(np.maximum((ww[:, None] * (ps[j, :, :T] - smean[None, :]) ** 2).sum(0), 0.0))
            nobs = int(np.isfinite(sub[j]["gr"][:T]).sum())      # ★GR観測行数(loglik正規化)
            llbest = float(np.nanmax(llw[np.isfinite(llw)])) if np.isfinite(llw).any() else np.nan
            out[c0 + j] = dict(mean=smean.astype(np.float32), std=sstd.astype(np.float32),
                               loglik=np.float32(llbest / max(nobs, 1)))   # per-row平均対数尤度=その表現PFのGR適合
            if capture:                                                    # ★v34ゲート: per-seed 生軌跡
                out[c0 + j].update(ps=ps[j, :, :T].astype(np.float32), pfw=pf[j, :, :T].astype(np.float32),
                                   ll=llw.astype(np.float64), ww=ww.astype(np.float64))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return out


# ==================== 単体検証(python pf_banks_v93.py) ====================
if __name__ == "__main__":
    import glob
    TR = DATA_DIR / "train"
    wids = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    wids = wids[:12]
    for bank in ["r0_seed32", "pfA"]:
        P = bank_param(bank)
        inps, trues = [], []
        for w in wids:
            hw = pd.read_csv(TR / f"{w}__horizontal_well.csv")
            tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
            tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
            m = np.isfinite(tt) & np.isfinite(tg)
            if m.sum() < 8:
                continue
            x = build_smoother_inputs(hw, tt[m], tg[m], P)
            if x is None:
                continue
            attach_anchor(x, w, P["_physics"])
            ev = hw[hw["TVT_input"].isna()]
            inps.append(x); trues.append(ev["TVT"].to_numpy(float))
        res = run_smoother_ext(inps, P, seed=99999, n_seeds=16, chunk=8, w_nn=0.0)
        errs = np.concatenate([res[k]["mean"] - trues[k] for k in range(len(inps))])
        rmse = float(np.sqrt(np.mean(errs[np.isfinite(errs)] ** 2)))
        print(f"  [{bank}] {len(inps)}井 smooth-PF(w_nn=0) overall RMSE = {rmse:.3f}"
              f"  (physics={P['_physics']}, anchor_mult={P.get('anchor_mult')})")
