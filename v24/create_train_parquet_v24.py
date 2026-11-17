"""
v24 train.parquet ビルダー ―― v20(PF→GBM)に heuristic/beam/GSF 由来の特徴を1パスで追加
==========================================================================================
方針(docs/v24.md):
  - PFは v24/pf_parameter.py の PF_LIB 全部(r0..r4, r0_seed32, un, pf_1, pf_3)。**n_seed=128**。
  - 1回の生成で pf/heur/beam/gsf 全部の特徴を作る(2回生成しない。PF出力は pf特徴と heur特徴で共有)。
  - 特徴は origin 接頭辞で命名 → 学習時に prefix でトグル:
      base=(v9既存,無接頭) / pf_ / heur_ / beam_ / gsf_
  - v20の枠組み(v9 build_well で base+pf特徴, GPU物理PF注入)を流用し、build_well 後に
    heur_/beam_/gsf_/追加pf_ 列を評価行に揃えて付与。

使い方: rogii_claude/.venv/Scripts/python.exe v24/create_train_parquet_v24.py
出力: v24/artifacts_v24/data/train.parquet
"""

import os, sys, time, glob, json, math, threading, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore"); np.seterr(all="ignore")

PROJECT_DIR = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT_DIR / "v24"))
from pf_parameter import PF_LIB            # noqa: E402  使うPF(決定済み)
from gsf_parameter import GSF_LIB          # noqa: E402  使うGSF(決定済み)

# ============================== CONFIG ==============================
DATA_DIR    = PROJECT_DIR / "rogii-wellbore-geology-prediction"
V9_NB       = PROJECT_DIR / "v9" / "pf_tuning_create_train_csv_multi_pf_v9.ipynb"
OUT_PARQUET = PROJECT_DIR / "v24" / "artifacts_v24_3" / "data" / "train.parquet"

SPLIT       = "train"       # "train"(真値あり) / "test"
IS_TRAIN    = True
N_WELLS     = None          # None=全坑井
FWD_SEEDS   = 32           # ★n_seed=128(pf特徴も同じ)
HEUR_SCALES = (3.0, 8.0, 20.0)   # heur特徴用 scale(温度)。sharp/soft でspread=不確実性
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float32
SEED        = 0
WELL_CHUNK  = 773           # PF/GSF GPUバッチ坑井数(n_seed128でVRAM調整。OOMなら下げる)
N_JOBS_CPU  = 4
# 後処理(heur最終予測=ヒューリスティックモデル)用
N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
BEST_SCALE = {0: 8.0, 1: 8.0, 2: 5.0, 3: 20.0, 4: 12.0, 5: 5.0}
HOLD_W = {0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.05, 5: 0.05}
PP_TAU = 85.0
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]
# ===================================================================

_TLS = threading.local()
_FWD_ENS = {}    # (wid, pf_name) -> dict(mean, std, scale_{s})
_GSF_ENS = {}    # (wid, gsf_idx) -> (mean, std)
_BEAM_ENS = {}   # wid -> (mean, spread)


# ----------------------- v9 名前空間 + PFセット(PF_LIB) -----------------------
def _patched_run_pf_ancc(hw, tw_tvt, tw_gr, p):
    r = _FWD_ENS.get((getattr(_TLS, "wid", None), p.get("name")))
    if r is None:
        return np.array([], np.float32), np.array([], np.float32)
    return r["mean"].astype(np.float32), r["std"].astype(np.float32)

def load_v9_namespace():
    nb = json.load(open(V9_NB, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = src.replace(
        r'dataset_path = Path(r"C:\Users\kosaka256\Documents\rogii\rogii-wellbore-geology-prediction")',
        f'dataset_path = Path(r"{DATA_DIR}")')
    src = src.replace("cache=True", "cache=False")
    cut = src.find('train_paths = sorted((CFG.dataset_path / "train").glob')
    if cut > 0:
        src = src[:cut]
    ns = {"__name__": "v9lib"}
    print("[v9] exec で v9 定義を読み込み中(FI/DI fit を含む)...")
    exec(compile(src, str(V9_NB), "exec"), ns)
    # PFセット = PF_LIB 全部。raw→p は v9 _make_pf_params, 物理キー付与。
    sets = []
    for k, (lab, raw) in enumerate(PF_LIB.items()):
        r = dict(raw); r["n_seeds"] = FWD_SEEDS
        p = ns["_make_pf_params"](r, k)
        p["PF_USE_PHYS"] = bool(raw.get("use_phys", True))
        p["PF_PHYS_SIG"] = float(raw.get("phys_sig", 1.0))
        p["label"] = lab
        sets.append(p)
    ns["PF_PARAM_SETS"] = sets
    print(f"[v24] PFセット {len(sets)}本: " + ", ".join(f"{p['label']}->{p['name']}" for p in sets))
    return ns


# ----------------------- 物理PF(v20と同一) -----------------------
def build_pf_inputs(hw, tw_tvt, tw_gr, p, V9):
    _smooth = V9["_smooth_radius_values"]; _grid = V9["_grid"]; _gr_sig = V9["_gr_sig"]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs = _gr_sig(hw, tw_tvt, tw_gr_pf, p)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    md_ev = ev["MD"].values.astype(np.float64); z_ev = ev["Z"].values.astype(np.float64)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc = _smooth(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos = hw.index.get_indexer(ev.index)
    gr_ev = gr_proc[ev_pos].astype(np.float64)
    return dict(md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(p["ANCC_N"]))

def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    B, T = md.shape
    ALPHA = p["ANCC_ALPHA"]; RN = p["ANCC_RN"]; PN = p["ANCC_PN"]; IR = p["ANCC_IR"]; IS = p["ANCC_IS"]
    RP = p["ANCC_RP"]; RR = p["ANCC_RR"]; RESAMP = p["PF_RESAMP"]; RATE_CLIP = p["PF_RATE_CLIP"]
    GR_POWER = p["PF_GR_POWER"]; JUMP_PROB = p["PF_JUMP_PROB"]; JUMP_STD = p["PF_JUMP_STD"]; CLIP = p["PF_TVT_CLIP_MARGIN"]
    USE_P = bool(p.get("PF_USE_PHYS", False)); PHYS_SIG = float(p.get("PF_PHYS_SIG", 1.0))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
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
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0col, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg_lk = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg_lk.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp, w * lkp)
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

def _pad_wells(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["md"]); G = len(x["gg"])
        md[b, :T] = torch.from_numpy(x["md"].astype("float32"))
        if T < Tmax: md[b, T:] = md[b, T - 1]
        z[b, :T] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))

def run_forward_ext(inps, p, S, device, seed, chunk, scales):
    """各坑井: pf_ancc(likelihood_scale加重)+std, および HEUR_SCALES 各 scale の pred を返す。"""
    N = int(p["ANCC_N"]); W = len(inps); out = [None] * W
    ls_scale = float(p["PF_LIKELIHOOD_SCALE"])
    for c in range(0, W, chunk):
        sub = inps[c:c + chunk]; w = len(sub); pad = _pad_wells(sub, device)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), rep(pad["vmin"]), rep(pad["step"]),
            rep(pad["gs"]), rep(pad["ls"]), rep(pad["ir"]), p, N, device, gen)
        pts = pts.view(w, S, -1).double().cpu().numpy(); ll = log_lik.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]
            base = ll[j] - np.nanmax(ll[j]) if np.isfinite(ll[j]).any() else np.zeros_like(ll[j])
            def wsum(sc):
                ww = np.exp(np.where(np.isfinite(base), base, -np.inf) / max(sc, 1e-6))
                ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(len(base), 1.0 / len(base))
                return ww
            ww0 = wsum(ls_scale); mean = (ww0[:, None] * pr).sum(0)
            var = (ww0[:, None] * (pr - mean[None, :]) ** 2).sum(0)
            d = dict(mean=mean.astype(np.float32), std=np.sqrt(np.maximum(var, 0.)).astype(np.float32))
            for sc in scales:
                d[f"scale_{sc:g}"] = (wsum(sc)[:, None] * pr).sum(0).astype(np.float32)
            out[c + j] = d
    if device == "cuda": torch.cuda.synchronize()
    return out


# ----------------------- GSF(v23移植, deterministic, mean+std) -----------------------
def _gsf_smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _gsf_grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_gsf_inputs(hw, tw_tvt, tw_gr, P):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 1:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _gsf_smooth(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _gsf_grid(tw_tvt, tw_gr_pf)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _gsf_smooth(gr_full, tw_fb, P["hgr_smooth_r"])
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
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir)

def _gsf_pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["md"]); G = len(x["gg"])
        md[b, :T] = torch.from_numpy(x["md"].astype("float32"))
        if T < Tmax: md[b, T:] = md[b, T - 1]
        z[b, :T] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), Tmax=Tmax)

def _gsf_core(pad, P, K, device, gen, dtype=DTYPE):
    md = pad["md"]; z = pad["z"]; gr = pad["gr"]; valid = pad["valid"]; grid = pad["grid"]
    glen = pad["glen"]; vmin = pad["vmin"]; step = pad["step"]; gs = pad["gs"]; ls = pad["ls"]; ir = pad["ir"]
    B, T = md.shape
    ALPHA = P["mom"]; QP = P["pn"]; QR = P["vn"]; IS = P["init_pos_std"]; IR = P["init_rate_std"]
    SPAN = P["init_span"]; EKF = P["ekf_gain"]; GR_POWER = P["gr_power"]; CLIP = P["tvt_clip_margin"]
    RESAMP = P["resamp"]; JP = P["jitter_pos"]; JR = P["jitter_rate"]; RTEMP = P["read_temp"]
    PHYS_SIG = P["phys_sig"]; USE_P = bool(P.get("use_phys", True))
    arK = torch.arange(K, device=device, dtype=dtype); zero = torch.zeros((), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]; dipp = ir[:, None]
    def grid_hs(v):
        ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        fr = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        h = gA * (1 - fr) + gB * fr; s = (gB - gA) / step[:, None]
        h = torch.where(below, g0, h); h = torch.where(above, glast, h); s = torch.where(below | above, zero, s)
        return h, s
    off = torch.linspace(-1.0, 1.0, K, device=device, dtype=dtype)[None, :] * SPAN
    mpos = ls[:, None] + off; mrate = ir[:, None].expand(B, K).clone()
    Ppp = torch.full((B, K), IS * IS, device=device, dtype=dtype); Prr = torch.full((B, K), IR * IR, device=device, dtype=dtype)
    Ppr = torch.zeros((B, K), device=device, dtype=dtype); w = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)
    pred = torch.zeros((B, T), device=device, dtype=dtype); std = torch.zeros((B, T), device=device, dtype=dtype)
    pm = md[:, 0] - 1.0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)[:, None]
        zi = z[:, i][:, None]; gri = gr[:, i]; obs = act & ~torch.isnan(gri); am = act[:, None]
        mrate_p = ALPHA * mrate; mpos_p = mpos + mrate_p * dm
        Ppp_p = Ppp + 2 * dm * Ppr + dm * dm * Prr + QP * QP
        Ppr_p = ALPHA * (Ppr + dm * Prr); Prr_p = ALPHA * ALPHA * Prr + QR * QR
        tvt_p = (mpos_p - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); mpos_p = tvt_p + zi
        h, s = grid_hs(mpos_p - zi); y = gri[:, None] - h; S = s * s * Ppp_p + gs[:, None] * gs[:, None]
        Kp = EKF * (Ppp_p * s) / S.clamp_min(1e-12); Kr = EKF * (Ppr_p * s) / S.clamp_min(1e-12)
        mpos_u = mpos_p + Kp * y; mrate_u = mrate_p + Kr * y
        u = 1.0 - Kp * s; vv = -Kr * s
        Ppp_u = (u * Ppp_p).clamp_min(1e-9); Ppr_u = 0.5 * (u * Ppr_p + vv * Ppp_p + Ppr_p); Prr_u = (vv * Ppr_p + Prr_p).clamp_min(1e-9)
        tvt_u = (mpos_u - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); mpos_u = tvt_u + zi
        ob = obs[:, None]
        mpos_n = torch.where(ob, mpos_u, mpos_p); mrate_n = torch.where(ob, mrate_u, mrate_p)
        Ppp_n = torch.where(ob, Ppp_u, Ppp_p); Ppr_n = torch.where(ob, Ppr_u, Ppr_p); Prr_n = torch.where(ob, Prr_u, Prr_p)
        mpos = torch.where(am, mpos_n, mpos); mrate = torch.where(am, mrate_n, mrate)
        Ppp = torch.where(am, Ppp_n, Ppp); Ppr = torch.where(am, Ppr_n, Ppr); Prr = torch.where(am, Prr_n, Prr)
        d = (y / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        if USE_P:
            dphys = (mrate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(ob, w * lk * lkp, w * lkp); ws = w_new.sum(1, keepdim=True)
        w_norm = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / K)); w = torch.where(am, w_norm, w)
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * K))
        cumw = torch.cumsum(w, 1); uu = (0.5 / K) + arK[None, :] / K
        idx = torch.searchsorted(cumw, uu.expand(B, K).contiguous(), right=False).clamp(0, K - 1)
        mpos_r = torch.gather(mpos, 1, idx) + JP * torch.randn((B, K), generator=gen, device=device, dtype=dtype)
        mrate_r = torch.gather(mrate, 1, idx) + JR * torch.randn((B, K), generator=gen, device=device, dtype=dtype)
        Ppp_r = torch.gather(Ppp, 1, idx); Ppr_r = torch.gather(Ppr, 1, idx); Prr_r = torch.gather(Prr, 1, idx)
        nm = need[:, None]
        mpos = torch.where(nm, mpos_r, mpos); mrate = torch.where(nm, mrate_r, mrate)
        Ppp = torch.where(nm, Ppp_r, Ppp); Ppr = torch.where(nm, Ppr_r, Ppr); Prr = torch.where(nm, Prr_r, Prr)
        w = torch.where(nm, torch.full_like(w, 1.0 / K), w)
        wt = w.clamp_min(1e-30) ** RTEMP; wt = wt / wt.sum(1, keepdim=True).clamp_min(1e-30)
        tvtk = mpos - zi; mean = (wt * tvtk).sum(1)
        pred[:, i] = mean; std[:, i] = torch.sqrt(((wt * (tvtk - mean[:, None]) ** 2).sum(1)).clamp_min(0))  # 成分spread=多峰度
        pm = torch.where(act, cur, pm)
    return pred, std

def run_gsf(inps, P, device, chunk, jit_seed):
    K = int(P["n_components"]); W = len(inps); out = [None] * W
    for c in range(0, W, chunk):
        sub = inps[c:c + chunk]; w = len(sub); pad = _gsf_pad(sub, device)
        gen = torch.Generator(device=device); gen.manual_seed(int(jit_seed))
        pred, std = _gsf_core(pad, P, K, device, gen)
        pred = pred.double().cpu().numpy(); std = std.double().cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); out[c + j] = (pred[j, :T].astype(np.float32), std[j, :T].astype(np.float32))
    if device == "cuda": torch.cuda.synchronize()
    return out


# ----------------------- beam(7.191移植, numba) -----------------------
from numba import njit  # noqa: E402
@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

def run_beam_well(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 1:
        return None
    last_tvt = float(kn["TVT_input"].iloc[-1])
    fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
    tw_gr = np.where(np.isfinite(tw_gr), tw_gr, fb).astype(np.float64); tw_tvt = tw_tvt.astype(np.float64)
    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(fb).to_numpy(float)
    hgr = gr_all[hw.index.get_indexer(ev.index)]
    si = int(np.abs(tw_tvt - last_tvt).argmin())
    res = []
    for (bs, mc, es, r) in BEAM_CONFIGS:
        sgr = _gsf_smooth(hgr, fb, r).astype(np.float64)
        path = _beam_jit(sgr, tw_gr, si, int(bs), float(mc), float(es))
        res.append(tw_tvt[path].astype(np.float32))
    arr = np.stack(res, 0)
    return arr.mean(0).astype(np.float32), arr.std(0).astype(np.float32)


# ----------------------- カテゴリ/後処理(heur最終予測) -----------------------
def well_code(n_eval, z_span):
    return int(n_eval > N_EVAL_THRESHOLD) + 2 * int(np.searchsorted(Z_SPAN_THRESHOLDS, z_span, side="right"))


# ----------------------- データ組み立て -----------------------
def build_dataset(V9):
    build_well = V9["build_well"]; P_SETS = V9["PF_PARAM_SETS"]
    base = DATA_DIR / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if N_WELLS is not None:
        wells = wells[:N_WELLS]
    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns:
            continue
        if build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P_SETS[0], V9) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[{SPLIT}] 使用坑井: {len(names)}")

    # --- PF: 各PF n_seed=128 で pf_ancc+std+scales ---
    _FWD_ENS.clear()
    for p in P_SETS:
        print(f"[pf] {p['label']}({p['name']}) ..."); t0 = time.perf_counter()
        inps = []
        for (hp, tp) in paths:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            inps.append(build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), p, V9))
        outs = run_forward_ext(inps, p, FWD_SEEDS, DEVICE, SEED, WELL_CHUNK, HEUR_SCALES)
        for i, w in enumerate(names):
            _FWD_ENS[(w, p["name"])] = outs[i]
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # --- GSF: GSF_LIB 各パラメータ ---
    _GSF_ENS.clear()
    for gi, g in enumerate(GSF_LIB):
        print(f"[gsf] round={g.get('round')} idx={gi} ..."); t0 = time.perf_counter()
        Pg = dict(g["params"])
        inps = []
        for (hp, tp) in paths:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            inps.append(make_gsf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pg))
        outs = run_gsf(inps, Pg, DEVICE, WELL_CHUNK, int(g.get("jit_seed", 0)))
        for i, w in enumerate(names):
            _GSF_ENS[(w, gi)] = outs[i]
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # --- beam(CPU/numba) ---
    _BEAM_ENS.clear()
    print("[beam] ..."); t0 = time.perf_counter()
    for (hp, tp) in paths:
        w = os.path.basename(hp).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        _BEAM_ENS[w] = run_beam_well(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float))
    print(f"      done {time.perf_counter()-t0:.1f}s")

    # --- build_well(pf注入) + origin列付与 ---
    V9["run_pf_ancc"] = _patched_run_pf_ancc
    pf_names = [p["name"] for p in P_SETS]
    pf_labels = [p["label"] for p in P_SETS]
    gsf_n = len(GSF_LIB)

    from joblib import Parallel, delayed
    def one(w, hp, tp):
        _TLS.wid = w
        df = build_well(str(hp), str(tp), IS_TRAIN)
        if df is None or len(df) == 0:
            return None
        n = len(df)
        lk = df["last_known_tvt"].to_numpy(float) if "last_known_tvt" in df.columns else np.zeros(n)
        def fit(a):   # 評価行数に長さを合わせる(不一致はNaN)
            a = np.asarray(a, float)
            return a if len(a) == n else np.full(n, np.nan)
        ref_mean = _FWD_ENS.get((w, pf_names[0]), {}).get("mean")   # 参照PF(r0)
        ref = fit(ref_mean) if ref_mean is not None else np.full(n, np.nan)
        # pf: 各PFの posterior std(不確実性) + PF間disagreement + 全PF平均
        for nm, lab in zip(pf_names, pf_labels):
            fw = _FWD_ENS.get((w, nm))
            if fw is not None:
                df[f"pf_std_{lab}"] = fit(fw["std"])           # ★PFごとの不確実性(尤度加重の分散)
        stk = [fit(_FWD_ENS[(w, nm)]["mean"]) for nm in pf_names if (w, nm) in _FWD_ENS]
        if stk:
            M = np.vstack(stk)
            df["pf_disagree"] = np.nanstd(M, axis=0)           # PF間の食い違い=不確実性
            df["pf_meanall_res"] = np.nanmean(M, axis=0) - lk
        # heur: 各PFの scale分布 由来(温度で予測がどれだけ動くか=不確実性), 参照PFの scale残差, ヒューリスティック最終予測
        for nm, lab in zip(pf_names, pf_labels):
            fw = _FWD_ENS.get((w, nm))
            if fw is None:
                continue
            sc_arrs = [fit(fw.get(f"scale_{sc:g}")) for sc in HEUR_SCALES]
            df[f"heur_sprd_{lab}"] = sc_arrs[-1] - sc_arrs[0]                  # soft - sharp = 不確実性
            df[f"heur_sstd_{lab}"] = np.nanstd(np.vstack(sc_arrs), axis=0)      # scale間 std
        r0 = _FWD_ENS.get((w, pf_names[0]))
        if r0 is not None:
            for sc in HEUR_SCALES:
                df[f"heur_scale{sc:g}_res"] = fit(r0.get(f"scale_{sc:g}")) - lk
            # ヒューリスティックモデル最終(カテゴリ別best scale + hold + apply_pp(tau))を r0 で
            zv = df["z"].to_numpy(float) if "z" in df.columns else np.zeros(n)
            zsp = float(np.nanmax(zv) - np.nanmin(zv)) if n else 0.0
            code = well_code(n, zsp); bs = BEST_SCALE.get(code, 8.0); hw_ = HOLD_W.get(code, 0.1)
            # bs に最も近い HEUR_SCALES を使う(無ければ mean)
            near = min(HEUR_SCALES, key=lambda s: abs(s - bs)); pred = fit(r0.get(f"scale_{near:g}"))
            pred = (1 - hw_) * pred + hw_ * lk
            mds = df["md_since"].to_numpy(float) if "md_since" in df.columns else np.zeros(n)
            pred = lk + (pred - lk) * (1.0 - np.exp(-np.maximum(mds, 0.) / PP_TAU))
            df["heur_pred_res"] = pred - lk
        # beam
        b = _BEAM_ENS.get(w)
        if b is not None:
            bm = fit(b[0]); df["beam_res"] = bm - lk; df["beam_gap"] = bm - ref; df["beam_spread"] = fit(b[1])
        # gsf: 各GSFの posterior mean残差 / 成分spread(多峰度=不確実性) / PF乖離 + GSF間disagreement
        gms = []
        for gi in range(gsf_n):
            g = _GSF_ENS.get((w, gi))
            if g is not None:
                gm = fit(g[0]); df[f"gsf_res_{gi}"] = gm - lk; df[f"gsf_std_{gi}"] = fit(g[1]); df[f"gsf_gap_{gi}"] = gm - ref
                gms.append(gm)
        if len(gms) >= 2:
            df["gsf_disagree"] = np.nanstd(np.vstack(gms), axis=0)   # GSF間の食い違い=不確実性
        return df

    print("[build] build_well(pf注入) + origin列 ...")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS_CPU, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1]) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s  parts={len(parts)}")
    if not parts:
        raise RuntimeError("有効な坑井がありません。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"device={DEVICE} fwd_seeds={FWD_SEEDS} PFs={len(PF_LIB)} GSFs={len(GSF_LIB)} chunk={WELL_CHUNK}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    V9 = load_v9_namespace()
    df = build_dataset(V9)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    cols = list(df.columns)
    for pre in ["pf_", "heur_", "beam_", "gsf_"]:
        print(f"  {pre}* = {sum(c.startswith(pre) for c in cols)} 列")
    print(f"\nsaved: {OUT_PARQUET}  rows={len(df)} cols={len(cols)}")


if __name__ == "__main__":
    main()
