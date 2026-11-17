"""
v25 Particle Smoother ― PF を v19 r0 に固定し、smooth パラメータだけ Optuna 調整
================================================================================
狙い(docs/v25.md):
  deep-research の最優先＝FFBSi/Particle Smoother(後ろ向き平滑)。
  「強い前向きPFを後ろ向き平滑で包む」が逐次位置推定コンペの勝ち筋。
  ただし PFフルチューニング(v22 per-cat, v19 full)は過学習=死に筋。
  → **PF を単一の v19 r0 に固定**(LB 7.098 の土台)し、
    **smooth_lag / smooth_blend の2つだけ**を調整する(少数パラメータ=過学習しにくい)。

v23 tune_smoother_v23.py との違い:
  - カテゴリ分岐(6分割)を撤去 → 全坑井を同じ r0 で処理。
  - warm-start(v22 json)を撤去 → PF は下記 R0_PARAMS でハードコード固定。
  - 探索は smooth_lag / smooth_blend のみ。PF パラメータは触らない。
  - 別バージョンからの import は一切しない(エンジンはこのファイルに丸ごと持つ=自己完結)。

固定する r0 (= v19/artifacts_v19_32seeds/pf_bank.json[0] = r0_seed32, use_phys=false):
  LB 7.098(単体最良, docs/scores.md)を出したヒューリスティックの土台PF。
  n_seed=32 でチューニングされたので、ここでも n_seed=32 で回す(★推論regime一致)。
  train overall(forward, seed0, n_seed32) ≈ 8.489 が期待値(pf_bank.json の overall_on_all)。

速度の近道(docs/v25.md「実装の近道」):
  PF は固定なので、前向き軌跡は smooth_lag ごとに1回だけ計算すればよい。
  さらに forward の誤差は **全 lag で同一**(平滑バッファは pos/w/rate に触らない=RNGも不変)。
  → LAGS の各値で1回ずつエンジンを回して forward誤差 ef と smoothed誤差 es[lag] を得れば、
     最終予測 (1-blend)*fwd + blend*smoothed の RMSE は **blend について解析的**。
     Optuna(lag, blend) の各 trial は行列演算だけ=瞬時。エンジン実行は LAGS の本数のみ。

評価(docs の指標ルール順守):
  overall = 全坑井の全行を連結した pooled RMSE(坑井平均ではない)。
  1) in-sample overall で forward-only(=r0素) vs smoothed(best) を比較(平滑が効くか)。
  2) out-of-fold(GroupKFold by well)で per-fold に (lag,blend) を選び直し、val を連結した OOF overall
     を honest な指標として併記(過学習チェック=analog_competitions の grouped-CV-by-well)。

出力: v25/artifacts_v25/smoother_r0.json
実行: rogii_claude/.venv/Scripts/python.exe v25/tune_smoother_r0_v25.py
※ メモリ: バッファ (B=chunk*seeds, L, N)。OOM なら EVAL_CHUNK / N_SEED / smooth_lag 上限を下げる。
"""

import os, time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import optuna

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v25" / "artifacts_v25"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

# --- シード: r0 のチューニング regime(n_seed=32, seed0=0)に一致させる(★重要) ---
SEED       = 0            # シード尤度加重アンサンブルの base seed(pf_bank round0 = seed 0)
N_SEED     = 32           # ★推論と揃える。r0_seed32 = LB7.098。128 は間違いモード寄り(scores.md)
EVAL_CHUNK = 100          # GPUバッチ well 数。バッファ B*L*N があるので控えめ。OOM なら下げる
N_WELLS    = None         # None=全well, 整数で先頭N(デバッグ)

# --- 平滑の探索 ---
LAGS       = [512]   # 固定ラグ候補(先読みステップ数 L)。エンジンはこの本数だけ回す
N_TRIALS   = 60           # Optuna 試行(objective は解析的なので瞬時。2次元探索に十分)
USE_PHYS   = False        # ★r0_seed32 は use_phys=false。r0 に合わせて false 固定

# --- OOF(honest 指標)---
KFOLD      = 5            # GroupKFold by well(各坑井=1グループ。K分割して per-fold に lag,blend を選ぶ)

STORAGE    = f"sqlite:///{(OUT_DIR / 'optuna_smoother_r0.db').as_posix()}"
STATE_PATH = OUT_DIR / "state_smoother_r0.json"
OUT_PATH   = OUT_DIR / "smoother_r0.json"

# --- 固定 PF = v19/artifacts_v19_32seeds/pf_bank.json[0] (r0_seed32, use_phys=false) をハードコード ---
#     (別バージョンから読み込まない方針のためベタ書き。値の出所は上記 json の round0。)
R0_PARAMS = {
    "n_particles":      400,
    "likelihood_scale": 10.536319667317654,
    "init_pos_std":     0.9331541486020234,
    "init_rate_std":    0.00047753793163756304,
    "mom":              0.9997911117425017,
    "vn":               0.0006297672017716754,
    "pn":               0.005014009140727263,
    "rp":               0.7627309890952053,
    "rr":               0.00045663430474523125,
    "resamp":           0.032049849143995945,
    "gr_sig_min":       11.865793233968713,
    "gr_sig_max":       44.98927964612316,
    "gr_sig_def":       119.25397464636981,
    "gr_sig_mult":      3.058318309943017,
    "tvt_clip_margin":  150.56111180234504,
    "rate_clip":        2.0,
    "gr_power":         1.7826896851395453,
    "hgr_smooth_r":     0,
    "tw_gr_smooth_r":   0,
    "jump_prob":        0.002,
    "jump_std":         5.0,
    "phys_sig":         0.013666550145707291,
    "use_phys":         False,
}
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- データ(v22/v23 と同一, 自己完結) -----------------------
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


# ----------------------- 物理PF + 固定ラグ系統スムーザー GPU(v23 と同一エンジン) -----------------------
def _pf_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, P, N, device, gen, dtype=DTYPE):
    """
    前向き物理PF + 固定ラグ系統スムーザー。
    返り値:
      pts_f : (B,T) 前向き filter の事後平均 TVT(= r0 素)
      pts_s : (B,T) 固定ラグ平滑の事後平均 TVT
      log_lik: (B,) 前向き周辺尤度の対数和(シード加重アンサンブル用)
    ※ pts_f と log_lik は smooth_lag L に依存しない(バッファ操作は pos/w/rate/RNG に触れない)。
    """
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))   # 固定ラグ(先読みステップ数)

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

    # 固定ラグ用バッファ: buf[:, k, :] = 現在の粒子identityに整列した「過去の geo位置」(円環バッファ)
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype)
        ptr = 0

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
            if use_sm:
                # 系譜で過去バッファを並べ替え(リサンプリングした行だけ)。idx:(B,N)
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)

        # 前向き filter 読み出し(smooth_lag に依存しない)
        pts_f[:, i] = (w * (pos - zi)).sum(1)

        # 固定ラグ平滑: 現在重み w で「Lステップ前の整列済み位置」を平均 → t=i-L を確定
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]                       # = step i-L の geo位置(現在粒子に整列)
                pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos                            # 現在(step i)の geo位置を書き込む
            ptr = (ptr + 1) % L

        pm = torch.where(act, cur, pm)

    # 末尾Lステップを最終重みで flush(ラグ縮小=未来全部使用)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def _pad(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))

def run_smoother_ens(inps, P, device, seed, n_seeds):
    """シード尤度加重アンサンブル後の forward / smoothed 予測(各 well の eval長配列)を返す。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); pad = _pad(inps, device)
    rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=device); gen.manual_seed(seed)
    pf, ps, ll = _pf_smoother_core(
        rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
        pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
        pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
        P, N, device, gen)
    pf = pf.view(W, S, -1).double().cpu().numpy(); ps = ps.view(W, S, -1).double().cpu().numpy()
    ll = ll.view(W, S).cpu().numpy()
    out_f = []; out_s = []
    for w in range(W):
        T = len(inps[w]["md"]); llw = ll[w]
        if np.isfinite(llw).any():
            mx = np.nanmax(llw[np.isfinite(llw)])
            lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
            ww = np.exp(lk / max(P["likelihood_scale"], 1e-6)); s = ww.sum()
            ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
        else:
            ww = np.full(S, 1.0 / S)
        out_f.append((ww[:, None] * pf[w, :, :T]).sum(0))
        out_s.append((ww[:, None] * ps[w, :, :T]).sum(0))
    return out_f, out_s


def predict_errs_for_lag(lag, subset, seed, n_seeds, chunk):
    """r0 を smooth_lag=lag で走らせ、各 well の (forward誤差, smoothed誤差) を返す。"""
    P = {**R0_PARAMS, "use_phys": USE_PHYS, "smooth_lag": lag}
    ef = {}; es = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        pf, ps = run_smoother_ens(inps, P, DEVICE, seed, n_seeds)
        for k, w in enumerate(sub):
            ef[w] = pf[k] - CACHE[w]["tvt_true"]; es[w] = ps[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return ef, es


def _rmse_cat(err_list):
    if not err_list:
        return 1e3
    e = np.concatenate(err_list); e = e[np.isfinite(e)]
    if e.size == 0:
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))


# ----------------------- 状態(再開用) -----------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  storage={STORAGE}")
    print(f"PF = 固定 r0_seed32 (use_phys={USE_PHYS}, n_particles={R0_PARAMS['n_particles']}), n_seed={N_SEED}, seed={SEED}")

    # --- データ ---
    wells = list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    print(f"loading {len(wells)} wells ...")
    CACHE = {}; t0 = time.time()
    for i, w in enumerate(wells):
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wells)} ({time.time()-t0:.0f}s)")
    all_wells = [w for w in wells if w in CACHE]
    print(f"usable wells: {len(all_wells)}")

    # ============================================================
    # STEP 1: LAGS の各値でエンジンを1回ずつ回し、forward/smoothed 誤差を precompute
    #   forward 誤差 ef は全 lag で同一なので初回だけ採用。
    #   (blend は後段で解析的にスイープするので、エンジン実行は len(LAGS) 本だけ)
    # ============================================================
    print("\n[STEP1] precompute forward/smoothed errors per smooth_lag ...")
    EF = None                      # well -> forward誤差配列(全lag共通)
    ES = {}                        # lag  -> {well: smoothed誤差配列}
    for lag in LAGS:
        t1 = time.time()
        ef, es = predict_errs_for_lag(lag, all_wells, SEED, N_SEED, EVAL_CHUNK)
        ES[lag] = es
        if EF is None:
            EF = ef
        else:
            # 健全性チェック: forward が lag に依らず一致するか(理論上一致)
            d = max(float(np.nanmax(np.abs(ef[w] - EF[w]))) if len(ef[w]) else 0.0 for w in all_wells)
            if d > 1e-3:
                print(f"   [warn] forward が lag={lag} で {d:.4g} ずれた(想定は0)。EF は最初の値を使用。")
        fwd_all = _rmse_cat([EF[w] for w in all_wells])
        smo_all = _rmse_cat([es[w] for w in all_wells])
        print(f"   lag={lag:3d}: forward_overall={fwd_all:.4f}  smoothed_overall(blend=1)={smo_all:.4f}  ({time.time()-t1:.0f}s)")

    FWD_BASE = _rmse_cat([EF[w] for w in all_wells])   # r0 素の overall(基準。pf_bank≈8.489, LB=7.098)
    print(f"[STEP1] forward baseline (r0 素) overall = {FWD_BASE:.4f}  (期待 ≈ 8.489, LB=7.098)")

    # 事前連結(解析ブレンド高速化用): well 順を固定
    order = all_wells
    ef_cat = np.concatenate([EF[w] for w in order])
    es_cat = {lag: np.concatenate([ES[lag][w] for w in order]) for lag in LAGS}

    def blended_overall(lag, blend, mask=None):
        """(1-blend)*forward + blend*smoothed(lag) の overall RMSE。mask で well 部分集合に限定可。"""
        e = (1.0 - blend) * ef_cat + blend * es_cat[lag]
        if mask is not None:
            e = e[mask]
        e = e[np.isfinite(e)]
        if e.size == 0:
            return 1e3
        return float(np.sqrt(np.mean(e ** 2)))

    # ============================================================
    # STEP 2: in-sample で (smooth_lag, smooth_blend) を Optuna 探索(objective は解析的=瞬時)
    # ============================================================
    print("\n[STEP2] Optuna over (smooth_lag, smooth_blend)  [in-sample overall] ...")
    study = optuna.create_study(direction="minimize", study_name="smoother_r0",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    def objective(trial):
        lag = trial.suggest_categorical("smooth_lag", LAGS)
        blend = trial.suggest_float("smooth_blend", 0.0, 1.0)
        return blended_overall(lag, blend)

    remaining = max(0, N_TRIALS - done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best_lag = int(study.best_params["smooth_lag"]); best_blend = float(study.best_params["smooth_blend"])
    best_overall = float(study.best_value)
    smo_best_overall = blended_overall(best_lag, 1.0)     # そのlagで平滑のみ
    print(f"[STEP2] best: lag={best_lag} blend={best_blend:.3f}  "
          f"blended_overall={best_overall:.4f}  (fwd={FWD_BASE:.4f} smo_only={smo_best_overall:.4f})")
    verdict = "平滑が効いている(smo<fwd)" if best_overall < FWD_BASE - 1e-6 else "平滑無効(blend≈0 で fwd 同等)"
    print(f"        判定: {verdict}")

    # 各 lag での blend 最適(参考テーブル)
    lag_table = {}
    grid_bl = np.linspace(0.0, 1.0, 101)
    for lag in LAGS:
        vals = [blended_overall(lag, b) for b in grid_bl]
        j = int(np.argmin(vals))
        lag_table[lag] = dict(best_blend=float(grid_bl[j]), best_overall=float(vals[j]),
                              smo_only_overall=blended_overall(lag, 1.0))

    # ============================================================
    # STEP 3: OOF(GroupKFold by well) honest 指標
    #   各 fold: train wells で (lag,blend) を選び直し、val wells の誤差を連結 → 全 fold の val を pool
    # ============================================================
    print(f"\n[STEP3] OOF GroupKFold(by well, K={KFOLD}) honest 指標 ...")
    n = len(order); fold_of = np.array([i % KFOLD for i in range(n)])   # 決定的な well 分割
    # well ごとの行数(連結配列のスライス用)
    lens = np.array([len(EF[w]) for w in order]); starts = np.concatenate([[0], np.cumsum(lens)])
    row_fold = np.concatenate([np.full(lens[i], fold_of[i]) for i in range(n)])   # 行単位の fold ラベル

    oof_pred_err = np.empty_like(ef_cat)   # 各行の OOF blended 誤差(val 側)
    fold_choices = {}
    for f in range(KFOLD):
        tr_mask = row_fold != f; va_mask = row_fold == f
        if not va_mask.any():
            continue
        # train 上で最良 (lag, blend) を選ぶ(解析グリッド)
        best = (1e9, LAGS[0], 0.0)
        for lag in LAGS:
            for b in grid_bl:
                e = (1.0 - b) * ef_cat[tr_mask] + b * es_cat[lag][tr_mask]
                e = e[np.isfinite(e)]
                r = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e9
                if r < best[0]:
                    best = (r, lag, float(b))
        _, lg, bl = best
        fold_choices[f] = dict(lag=lg, blend=bl, train_rmse=best[0])
        oof_pred_err[va_mask] = ((1.0 - bl) * ef_cat + bl * es_cat[lg])[va_mask]
        print(f"   fold{f}: choose lag={lg} blend={bl:.3f}  (train={best[0]:.4f})")
    e = oof_pred_err[np.isfinite(oof_pred_err)]
    oof_overall = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    # OOF での forward 基準(比較用)
    ef_oof = ef_cat[np.isfinite(ef_cat)]; oof_fwd = float(np.sqrt(np.mean(ef_oof ** 2))) if ef_oof.size else 1e3
    print(f"[STEP3] OOF overall(smoothed) = {oof_overall:.4f}  vs  forward = {oof_fwd:.4f}  "
          f"({'OOFでも改善' if oof_overall < oof_fwd - 1e-6 else 'OOFでは改善せず(過学習疑い)'})")

    # ============================================================
    # 出力
    # ============================================================
    result = dict(
        algorithm="fixed_lag_genealogy_smoother__r0_fixed",
        pf="r0_seed32 (v19/artifacts_v19_32seeds/pf_bank.json[0], use_phys=false)",
        r0_params=R0_PARAMS,
        n_seed=N_SEED, seed=SEED, lags=LAGS, kfold=KFOLD,
        forward_baseline_overall=FWD_BASE,
        insample_best=dict(smooth_lag=best_lag, smooth_blend=best_blend,
                           blended_overall=best_overall, smoothed_only_overall=smo_best_overall),
        per_lag=lag_table,
        oof=dict(overall_smoothed=oof_overall, overall_forward=oof_fwd, fold_choices=fold_choices),
        note="baseline=r0素forward(≈8.489 train / LB7.098). smo<fwd かつ OOFでも下がれば Particle Smoother 本物。",
    )
    save_json(OUT_PATH, result)
    print(f"\nsaved -> {OUT_PATH}")
    print(f"  forward baseline overall = {FWD_BASE:.4f}")
    print(f"  in-sample best  lag={best_lag} blend={best_blend:.3f} -> blended {best_overall:.4f}")
    print(f"  OOF(K={KFOLD}) smoothed   = {oof_overall:.4f}  (forward {oof_fwd:.4f})")


if __name__ == "__main__":
    main()
