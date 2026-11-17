"""
v25 pseudo-satellite 反復 (iterate-then-re-estimate) ― FFBS平滑を prior に forward を再走行
================================================================================
狙い(analog_competitions.md #7):
  FFBSm は「forward雲が張った support の中でしか再重み付けできない」= 覆えてない真値は復元不能。
  → **一周目の平滑軌跡を prior にして forward PF をもう一周**引き寄せて走らせると、
    未来情報で選んだ正しい datum モード周辺へ粒子探索が移動し、一周目が張れなかった support を埋める。
    その新しい雲で再び FFBS = 支持不足の壁を超える(EM的な二周目)。
  GNSS の pseudo-satellite(平滑トラックを prior に再測位して曖昧性を解く)の地質版。

機構(forward の重みにアンカー項を追加):
  w *= exp(-0.5 * ((particle_tvt - ref_tvt_t) / ANCHOR_SIG)^2)
  - ref_tvt = 前反復の平滑TVT軌跡(坑井の (T,) 配列, 全seed共通)。
  - ANCHOR_SIG 大=弱く引く / 小=強く引く(強すぎると平滑をなぞるだけ=循環collapse)。→ スイープ。
  - seed加重の loglik は GR のみ(アンカーは prior なので尤度に混ぜない=比較の一貫性)。
  - クランプは常に「元の素 forward(iter0 の anchor無し forward)」基準で効かせる(安全網を固定)。

評価: overall = 全行プールRMSE。iter0(=pure FFBS) vs iter1..N を並べ、iter1 が iter0 を超えるかが本丸。
  ANCHOR_SIG をスイープ。効けば OOF(GroupKFold by well)で締める(別途)。

出力: v25/artifacts_v25/pseudosat_r0.json
実行: rogii_claude/.venv/Scripts/python.exe v25/tune_pseudosat_r0_v25.py
※ 1坑井あたり forward+FFBS を (1 + len(ANCHOR_SIGS)*N_ITER) 回。重い → まず N_WELLS で絞る/ANCHOR_SIGS少なめ。
"""

import os, time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v25" / "artifacts_v25"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

SEED       = 0
N_SEED     = 32           # FFBS と揃える(r0_seed32 regime)
N_WELLS    = 773          # ★まず絞って感触(None=全773)。重いので注意
INFLATE    = 32.0         # FFBS 遷移帯域(tuner の best)
N_BWD      = 128          # backward 間引き(高速化)
CLAMP_FT   = 60.0         # 元forwardから±これ超で外れた行はforwardに戻す

# --- pseudo-satellite 反復 ---
ANCHOR_SIGS = [20.0, 40.0]  # アンカー帯域(ft) スイープ。小=強く引く/大=弱く。0近傍=平滑なぞり(collapse)
N_ITER      = 1             # 再推定の周回数(まず1で iter1 が iter0=pure FFBS を超えるか)
USE_PHYS    = False

OUT_PATH = OUT_DIR / "pseudosat_r0.json"

R0_PARAMS = {
    "n_particles": 400, "likelihood_scale": 10.536319667317654,
    "init_pos_std": 0.9331541486020234, "init_rate_std": 0.00047753793163756304,
    "mom": 0.9997911117425017, "vn": 0.0006297672017716754, "pn": 0.005014009140727263,
    "rp": 0.7627309890952053, "rr": 0.00045663430474523125, "resamp": 0.032049849143995945,
    "gr_sig_min": 11.865793233968713, "gr_sig_max": 44.98927964612316, "gr_sig_def": 119.25397464636981,
    "gr_sig_mult": 3.058318309943017, "tvt_clip_margin": 150.56111180234504, "rate_clip": 2.0,
    "gr_power": 1.7826896851395453, "hgr_smooth_r": 0, "tw_gr_smooth_r": 0,
    "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 0.013666550145707291, "use_phys": False,
}
# ===================================================================


# ----------------------- データ(自己完結) -----------------------
def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})

def build_cache(wid):
    hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns:
        return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) < 10:
        return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    if len(tw_tvt) < 3:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    z_ev = ev["Z"].values.astype(np.float64)
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev,
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv)

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt, tw_gr, step=0.2):
    a, b = float(tw_tvt.min()), float(tw_tvt.max()); g = np.arange(a, b + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(a), float(step)

def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    return dict(md=c["md"], z=c["z"], gr=gr_sm[c["ev_pos"]].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=c["ls"], ir=c["ir"], N=int(P["n_particles"]))


# ----------------------- FFBS forward(アンカー項つき) -----------------------
def ffbs_forward_well(inp, P, seed, S, device, ref_tvt=None, anchor_sig=None, dtype=DTYPE):
    """
    前向き物理PF(S seed)。粒子雲を保存。
    ref_tvt が与えられたら、各ステップの重みに exp(-0.5*((tvt-ref)/anchor_sig)^2) を掛ける
    (= 前反復の平滑軌跡へ引き寄せる pseudo-satellite prior)。loglik には混ぜない(GRのみ)。
    """
    N = int(P["n_particles"]); T = len(inp["md"])
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]

    md = torch.from_numpy(inp["md"].astype("float32")).to(device)
    z  = torch.from_numpy(inp["z"].astype("float32")).to(device)
    gr = torch.from_numpy(inp["gr"].astype("float32")).to(device)
    grid = torch.from_numpy(inp["gg"].astype("float32")).to(device); G = grid.numel()
    vmin = float(inp["gmin"]); step = float(inp["gst"]); gs = float(inp["gs"])
    ls = float(inp["ls"]); ir = float(inp["ir"])
    use_anchor = ref_tvt is not None and anchor_sig is not None
    if use_anchor:
        ref = torch.from_numpy(np.asarray(ref_tvt, dtype="float32")).to(device); ASIG = float(anchor_sig)

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
            loglik = loglik + torch.log(avg.clamp_min(1e-300).double())   # ★GRのみ(アンカー非計上)
        if USE_P:
            dphys = (rate - ir) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = (w * lk * lkp) if obs else (w * lkp)
        if use_anchor:
            anc = torch.exp(-0.5 * ((tvt - ref[i]) / ASIG) ** 2).clamp_min(1e-300)   # tvt=(pos-zi)
            w_new = w_new * anc
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
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


def reduce_cloud(Xpos, Xrate, Wst, n_bwd, gen, device, dtype=DTYPE):
    S, T, N = Xpos.shape
    cumw = torch.cumsum(Wst, dim=2)
    ar = torch.arange(n_bwd, device=device, dtype=dtype)
    u0 = torch.rand((S, T, 1), generator=gen, device=device, dtype=dtype) * (1.0 / n_bwd)
    u = u0 + ar / n_bwd
    idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
    return torch.gather(Xpos, 2, idx), torch.gather(Xrate, 2, idx), \
        torch.full((S, T, n_bwd), 1.0 / n_bwd, device=device, dtype=dtype)


def ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, inflate, device, dtype=DTYPE):
    S, T, N = Xpos.shape
    ALPHA = P["mom"]; RN = max(P["vn"] * inflate, 1e-9); PN = max(P["pn"] * inflate, 1e-9)
    sm = torch.empty((S, T), device=device, dtype=dtype); eps = 1e-30
    wb = Wst[:, T - 1, :].clone(); wb = wb / wb.sum(1, keepdim=True).clamp_min(eps)
    sm[:, T - 1] = (wb * (Xpos[:, T - 1, :] - z[T - 1])).sum(1)
    for t in range(T - 2, -1, -1):
        dm = DM[t + 1]
        r_i = Xrate[:, t, :]; p_i = Xpos[:, t, :]
        r_j = Xrate[:, t + 1, :]; p_j = Xpos[:, t + 1, :]
        A = (r_j[:, None, :] - ALPHA * r_i[:, :, None]) / RN
        Bm = (p_j[:, None, :] - p_i[:, :, None] - r_j[:, None, :] * dm) / PN
        logf = -0.5 * (A * A + Bm * Bm)
        m = logf.amax(dim=1, keepdim=True); f = torch.exp(logf - m)
        w_t = Wst[:, t, :]
        vj = (w_t[:, :, None] * f).sum(1).clamp_min(eps)
        ratio = wb / vj
        w_sm = w_t * (f * ratio[:, None, :]).sum(2)
        w_sm = w_sm / w_sm.sum(1, keepdim=True).clamp_min(eps)
        wb = w_sm
        sm[:, t] = (wb * (p_i - z[t])).sum(1)
    return sm


def ensemble_seed(vals_ST, loglik_S, P):
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


def _smooth_pass(inp, P, seed, S, inflate, n_bwd, gen_bwd, ref_tvt=None, anchor_sig=None):
    """1回の forward(+anchor) → FFBS。返り値: (forward TVT, smoothed TVT) 各 (T,)。"""
    Xpos, Xrate, Wst, DM, loglik, z = ffbs_forward_well(inp, P, seed, S, DEVICE, ref_tvt, anchor_sig)
    fwd = ensemble_seed((Wst * (Xpos - z[None, :, None])).sum(2), loglik, P)
    if n_bwd is not None and n_bwd < Xpos.shape[2]:
        Xpos, Xrate, Wst = reduce_cloud(Xpos, Xrate, Wst, int(n_bwd), gen_bwd, DEVICE)
    sm = ensemble_seed(ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, inflate, DEVICE), loglik, P)
    del Xpos, Xrate, Wst
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return fwd, sm


def _clamp(sm, fwd_base):
    if CLAMP_FT is None: return sm
    bad = np.abs(sm - fwd_base) > CLAMP_FT
    return np.where(bad, fwd_base, sm) if bad.any() else sm


def _rmse(errs):
    e = np.concatenate(errs); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3


def main():
    global CACHE
    print(f"device={DEVICE}  PF=r0_seed32 n_seed={N_SEED} inflate={INFLATE} N_BWD={N_BWD} "
          f"anchor_sigs={ANCHOR_SIGS} n_iter={N_ITER}")
    wells = list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    print(f"loading {len(wells)} wells ...")
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    aw = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(aw)}")

    P = {**R0_PARAMS, "use_phys": USE_PHYS}
    gen_bwd = torch.Generator(device=DEVICE); gen_bwd.manual_seed(SEED + 987654321)

    # 累積誤差(全行プールrunning): keys = 'fwd','iter0', ('sig',it)
    keys = ["fwd", "iter0"] + [(sig, it) for sig in ANCHOR_SIGS for it in range(1, N_ITER + 1)]
    E = {k: [] for k in keys}   # 誤差配列を貯める
    n = len(aw); t0 = time.time()
    for k, wid in enumerate(aw):
        c = CACHE[wid]; true = c["tvt_true"]; inp = make_inputs(c, P)
        # --- iter0: anchor無し = pure FFBS ---
        fwd, sm0 = _smooth_pass(inp, P, SEED, N_SEED, INFLATE, N_BWD, gen_bwd)
        sm0 = _clamp(sm0, fwd)
        E["fwd"].append(fwd - true); E["iter0"].append(sm0 - true)
        # --- 各 anchor_sig で反復 ---
        for sig in ANCHOR_SIGS:
            ref = sm0.copy()
            for it in range(1, N_ITER + 1):
                _, sm = _smooth_pass(inp, P, SEED, N_SEED, INFLATE, N_BWD, gen_bwd,
                                     ref_tvt=ref, anchor_sig=sig)
                sm = _clamp(sm, fwd)                       # 安全網は常に元forward基準
                E[(sig, it)].append(sm - true)
                ref = sm.copy()
        # 走行中 overall
        run = {}
        run["fwd"] = _rmse(E["fwd"]); run["iter0"] = _rmse(E["iter0"])
        for sig in ANCHOR_SIGS:
            run[(sig, N_ITER)] = _rmse(E[(sig, N_ITER)])
        el = time.time() - t0; eta = el / (k + 1) * (n - (k + 1))
        sigstr = " ".join(f"sig{sig}={run[(sig, N_ITER)]:.3f}" for sig in ANCHOR_SIGS)
        print(f"[{k+1}/{n}] {wid[:8]} | run: fwd={run['fwd']:.3f} ffbs={run['iter0']:.3f} "
              f"{sigstr} | {el:.0f}s ETA {eta/60:.0f}m", flush=True)

    # 最終
    print("\n===== 最終 overall(全行プール) =====")
    fwd_o = _rmse(E["fwd"]); ffbs_o = _rmse(E["iter0"])
    print(f"forward           = {fwd_o:.4f}")
    print(f"iter0 (pure FFBS) = {ffbs_o:.4f}  (Δ vs fwd {ffbs_o-fwd_o:+.4f})")
    table = {"forward": fwd_o, "iter0_ffbs": ffbs_o, "anchor": {}}
    for sig in ANCHOR_SIGS:
        table["anchor"][str(sig)] = {}
        for it in range(1, N_ITER + 1):
            o = _rmse(E[(sig, it)])
            print(f"anchor_sig={sig:>5} iter{it} = {o:.4f}  (Δ vs FFBS {o-ffbs_o:+.4f})")
            table["anchor"][str(sig)][f"iter{it}"] = o
    best = min(((sig, it, _rmse(E[(sig, it)])) for sig in ANCHOR_SIGS for it in range(1, N_ITER + 1)),
               key=lambda x: x[2])
    verdict = "pseudo-sat が FFBS を超えた" if best[2] < ffbs_o - 1e-6 else "FFBS を超えず(効果なし/collapse)"
    print(f"\n判定: {verdict}  best=(sig={best[0]}, iter{best[1]}) {best[2]:.4f}  vs FFBS {ffbs_o:.4f}")
    OUT_PATH.write_text(json.dumps(dict(
        algorithm="pseudo_satellite_iterated_ffbs__r0_fixed",
        n_seed=N_SEED, inflate=INFLATE, n_bwd=N_BWD, clamp_ft=CLAMP_FT,
        anchor_sigs=ANCHOR_SIGS, n_iter=N_ITER, n_wells=len(aw),
        results=table, best=dict(anchor_sig=best[0], iter=best[1], overall=best[2]),
        note="iter0=pure FFBS が基準。anchor iterが下回れば pseudo-sat 有効。効けば OOF/LB へ。",
    ), indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
