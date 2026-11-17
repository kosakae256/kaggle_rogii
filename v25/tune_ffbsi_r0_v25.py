"""
v25 FFBSm（Forward-Filter Backward-Smoother, 平均直出し版）― PF を v19 r0 に固定
================================================================================
狙い(docs/v25.md, analog_competitions.md の最優先=FFBSi/粒子平滑):
  固定ラグ系統スムーザー(tune_smoother_r0_v25.py)は lag を伸ばすほど改善したが、
  ①メモリ O(L)、②系譜退化 でどこかで頭打ち。FFBS(前向きフィルタ・後ろ向き平滑)は
  **forward の粒子雲を使い回して後ろ向きに重み付け直す**ので、
  ・n_seed=32 のまま(新しい seed を足さない=推論regime一致)、
  ・全未来を使用(L→∞ 相当)、
  ・系譜退化なし(遷移密度で全祖先を再考する)。
  逆向きに新しくPFを走らせる訳ではない(=v10 逆向きPFの「トゥにアンカー無し」問題を回避)。

FFBSm（marginal smoother, Godsill-Doucet-West 2004 の平均版）:
  forward で各 t の粒子 {x_t^i}=(pos,rate) と正規化フィルタ重み {w_t^i} を保存(リサンプル前)。
  後ろ向きに:
    w_{T|T}^i = w_T^i
    w_{t|T}^i = w_t^i · Σ_j [ w_{t+1|T}^j · f(x_{t+1}^j | x_t^i) / Σ_l w_t^l f(x_{t+1}^j | x_t^l) ]
    平滑平均 m_t = Σ_i w_{t|T}^i · (pos_t^i − z_t)
  ※ FFBSi(M本サンプリング)ではなく marginal(平均直出し)。RMSEは事後平均で測るので平均が欲しい=こちらが最適。

遷移密度(r0 の力学):
  rate_{t+1} ~ N(α·rate_t, vn²),  pos_{t+1} ~ N(pos_t + rate_{t+1}·dm, pn²)
  f = N_rate · N_pos。正規化定数は比で消えるので未正規化 exp(-0.5(A²+B²)) を使う。
  ※ jump(まれ 0.2%)・clip・resample roughening は遷移密度に含めない(標準的な近似)。
  ★ r0 は vn/pn が極小=遷移がほぼデルタ → FFBSm が系譜スムーザーに近づく。
    それを崩す帯域ノブ = TRANS_INFLATE(vn,pn を ×inflate)。1.0=忠実、>1=平滑強化。LAGS と同じ役割。

評価(docs 指標順守): overall = 全坑井の全行を連結した pooled RMSE。
  最終予測 = (1-blend)*forward + blend*ffbs。TRANS_INFLATE と blend を
  in-sample と OOF(GroupKFold by well)で選ぶ(過学習チェック)。

出力: v25/artifacts_v25/ffbsi_r0.json
実行: rogii_claude/.venv/Scripts/python.exe v25/tune_ffbsi_r0_v25.py
※ メモリ: forward雲を坑井ごとに (S,T,N) ×3 保存 + 後ろ向きで (S,N,N)/step。
  長い坑井 or OOM 時は N_SEED / n_particles を下げる、TRANS_INFLATE を減らす。
※ 速度: 後ろ向きは O(T·N²·S)/坑井。全坑井×inflate本数。重ければ TRANS_INFLATE を絞る/N_SEED を下げる。
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

# --- シード: r0 のチューニング regime(n_seed=32, seed0=0)に一致 ---
SEED       = 0
N_SEED     = 128           # ★推論と揃える。FFBSは forward 雲を使い回すので seed は増えない
N_WELLS    = None         # None=全well, 整数で先頭N(デバッグ)

# --- FFBS 遷移帯域スイープ(LAGS と同じ役割。エンジンはこの本数だけ回す) ---
#   ★inflate=1.0 は遷移がほぼデルタ=数値的に退化(vj≈0で重み暴走→坑井RMSE爆発)。
#     広帯域(4/8/16)は安定かつ効く。1.0/2.0 は捨てる。
TRANS_INFLATE = [32.0]  # vn,pn を ×inflate。大=平滑強化&安定(退化しにくい)
N_TRIALS      = 60        # Optuna(inflate, blend)。objective は解析的=瞬時
USE_PHYS      = False     # r0_seed32 は use_phys=false

# --- フォールバック・クランプ(ヘビーテール対策=最重要) ---
#   FFBS平滑予測が forward から ±CLAMP_FT(ft) を超えて外れた行は forward に戻す。
#   爆発坑井(err数百ft)の下振れをキャップ→二乗プールoverallを守る。大勝ち(datum修正~数十ft)は残す。
CLAMP_FT = 60.0          # None で無効。datum両モード差~15ftより十分大、爆発(数百ft)は確実に切る

# --- 高速化: 後ろ向き前に forward雲を N_BWD 粒子へ間引き(backward は O(N²)なので効く) ---
#   None=全粒子(n_particles=400) / 整数=その粒子数へ系統リサンプル。128 なら 400²→128² で backward ~10倍速。
#   forward(基準)は常に全粒子で計算=基準は不変。平滑のみ近似(精度は N_BWD=None と比較して確認)。
N_BWD = 128

# --- OOF(honest 指標) ---
KFOLD      = 5

STORAGE    = f"sqlite:///{(OUT_DIR / 'optuna_ffbsi_r0.db').as_posix()}"
OUT_PATH   = OUT_DIR / "ffbsi_r0.json"

# --- 固定 PF = v19/artifacts_v19_32seeds/pf_bank.json[0] (r0_seed32, use_phys=false) をハードコード ---
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


# ----------------------- FFBS forward(1坑井・S seed・粒子雲を保存) -----------------------
def ffbs_forward_well(inp, P, seed, S, device, dtype=DTYPE):
    """
    1 坑井の前向き物理PF を S seed(バッチ)で走らせ、各ステップの粒子雲を保存。
    返り値:
      Xpos, Xrate : (S,T,N) 各ステップの粒子(リサンプル前)
      Wst         : (S,T,N) 各ステップの正規化フィルタ重み(リサンプル前)
      DM          : (T,)   ステップ間隔 dm(t-1→t)
      loglik      : (S,)   前向き周辺尤度の対数和(シード加重用)
      z_t         : (T,)   z(TVT=pos-z 用)
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

        # ---- FFBS用: リサンプル前の粒子雲+重みを保存 ----
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


def ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, inflate, device, dtype=DTYPE):
    """
    FFBSm(marginal)後ろ向き再帰。平滑後の事後平均 TVT を seed ごとに返す (S,T)。
    遷移: f(x_{t+1}|x_t) = N(rate'; α rate, (vn·inflate)²) · N(pos'; pos+rate'·dm, (pn·inflate)²)
    """
    S, T, N = Xpos.shape
    ALPHA = P["mom"]; RN = max(P["vn"] * inflate, 1e-9); PN = max(P["pn"] * inflate, 1e-9)
    sm = torch.empty((S, T), device=device, dtype=dtype)
    eps = 1e-30

    wb = Wst[:, T - 1, :].clone()                              # (S,N) 平滑重み(=フィルタ重み at T)
    wb = wb / wb.sum(1, keepdim=True).clamp_min(eps)
    sm[:, T - 1] = (wb * (Xpos[:, T - 1, :] - z[T - 1])).sum(1)

    for t in range(T - 2, -1, -1):
        dm = DM[t + 1]
        r_i = Xrate[:, t, :]; p_i = Xpos[:, t, :]               # (S,N_i)
        r_j = Xrate[:, t + 1, :]; p_j = Xpos[:, t + 1, :]       # (S,N_j)
        A = (r_j[:, None, :] - ALPHA * r_i[:, :, None]) / RN     # (S,N_i,N_j)
        Bm = (p_j[:, None, :] - p_i[:, :, None] - r_j[:, None, :] * dm) / PN
        logf = -0.5 * (A * A + Bm * Bm)                          # (S,N_i,N_j)
        m = logf.amax(dim=1, keepdim=True)                      # (S,1,N_j) 安定化(比で相殺)
        f = torch.exp(logf - m)
        w_t = Wst[:, t, :]                                      # (S,N_i)
        vj = (w_t[:, :, None] * f).sum(1).clamp_min(eps)        # (S,N_j)
        ratio = wb / vj                                        # (S,N_j)
        w_sm = w_t * (f * ratio[:, None, :]).sum(2)             # (S,N_i)
        w_sm = w_sm / w_sm.sum(1, keepdim=True).clamp_min(eps)
        wb = w_sm
        sm[:, t] = (wb * (p_i - z[t])).sum(1)
    return sm


def reduce_cloud(Xpos, Xrate, Wst, n_bwd, gen, device, dtype=DTYPE):
    """
    後ろ向き高速化: 各ステップの forward雲を Wst で N→n_bwd 粒子へ系統リサンプル。
    リサンプル後は重み一様(1/n_bwd)。backward の O(N²) を O(n_bwd²) に落とす近似。
    返り値: Xp,Xr (S,T,n_bwd), Wr (S,T,n_bwd) 一様。
    """
    S, T, N = Xpos.shape
    cumw = torch.cumsum(Wst, dim=2)                                     # (S,T,N)
    ar = torch.arange(n_bwd, device=device, dtype=dtype)
    u0 = torch.rand((S, T, 1), generator=gen, device=device, dtype=dtype) * (1.0 / n_bwd)
    u = u0 + ar / n_bwd                                                 # (S,T,n_bwd)
    idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)      # (S,T,n_bwd)
    Xp = torch.gather(Xpos, 2, idx); Xr = torch.gather(Xrate, 2, idx)
    Wr = torch.full((S, T, n_bwd), 1.0 / n_bwd, device=device, dtype=dtype)
    return Xp, Xr, Wr


def ensemble_seed(vals_ST, loglik_S, P):
    """(S,T) を seed 尤度加重で 1本 (T,) に。run_smoother_ens と同じ加重式。"""
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


def _sumsq(err):
    e = err[np.isfinite(err)]
    return float(np.sum(e ** 2)), int(e.size)

def predict_errs_ffbs(all_wells, inflates, seed, S):
    """
    各坑井で forward を1回走らせ雲を保存→ inflates ごとに後ろ向き平滑。
    返り値: EF(well->forward誤差), ESdict(inflate->{well:ffbs誤差})
    forward は inflate 非依存なので1回。
    ★坑井ごとに逐次表示(その坑井の RMSE + 走行中の overall プールRMSE + ETA)。
    """
    EF = {}; ES = {infl: {} for infl in inflates}
    P = {**R0_PARAMS, "use_phys": USE_PHYS}
    n = len(all_wells); t0 = time.time()
    acc = {"fwd": [0.0, 0]}                       # 累積 (Σ誤差², 行数) = 走行中 overall 用
    for infl in inflates: acc[infl] = [0.0, 0]
    gen_bwd = torch.Generator(device=DEVICE); gen_bwd.manual_seed(seed + 987654321)
    for k, wid in enumerate(all_wells):
        inp = make_inputs(CACHE[wid], P)
        Xpos, Xrate, Wst, DM, loglik, z = ffbs_forward_well(inp, P, seed, S, DEVICE)
        # forward 事後平均(=r0 素): Σ_i w_t^i (pos-z)  ★必ず全粒子で(基準を不変に)
        fwd_ST = (Wst * (Xpos - z[None, :, None])).sum(2)       # (S,T)
        fwd = ensemble_seed(fwd_ST, loglik, P)
        true = CACHE[wid]["tvt_true"]
        ef = fwd - true; EF[wid] = ef
        s, c = _sumsq(ef); acc["fwd"][0] += s; acc["fwd"][1] += c
        wr_fwd = math.sqrt(s / c) if c else float("nan")
        # ---- 高速化: 後ろ向き用に雲を N_BWD へ間引き(inflate 非依存なので1回) ----
        if N_BWD is not None and N_BWD < Xpos.shape[2]:
            Xpos, Xrate, Wst = reduce_cloud(Xpos, Xrate, Wst, int(N_BWD), gen_bwd, DEVICE)
        parts = []
        for infl in inflates:
            sm_ST = ffbsm_backward_well(Xpos, Xrate, Wst, DM, z, P, infl, DEVICE)
            sm = ensemble_seed(sm_ST, loglik, P)
            # ---- フォールバック・クランプ: forward から ±CLAMP_FT 超で外れた行は forward に戻す ----
            nclip = 0
            if CLAMP_FT is not None:
                bad = np.abs(sm - fwd) > CLAMP_FT
                nclip = int(np.count_nonzero(bad))
                if nclip:
                    sm = np.where(bad, fwd, sm)
            es = sm - true; ES[infl][wid] = es
            s2, c2 = _sumsq(es); acc[infl][0] += s2; acc[infl][1] += c2
            parts.append((infl, math.sqrt(s2 / c2) if c2 else float("nan"), nclip))
        del Xpos, Xrate, Wst
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        # 走行中 overall(全行プール) と ETA
        run_fwd = math.sqrt(acc["fwd"][0] / acc["fwd"][1]) if acc["fwd"][1] else float("nan")
        run_sm = " ".join(f"@{infl}={math.sqrt(acc[infl][0]/acc[infl][1]):.3f}"
                          for infl in inflates if acc[infl][1])
        wsm = " ".join(f"@{infl}={r:.2f}" for infl, r, _ in parts)
        clip = " ".join(f"clip@{infl}={nc}" for infl, _, nc in parts if nc)
        el = time.time() - t0; eta = el / (k + 1) * (n - (k + 1))
        print(f"[{k+1}/{n}] {wid[:8]} T={len(ef)} fwd={wr_fwd:.2f} ffbs {wsm}"
              f"{(' [' + clip + ']') if clip else ''} | "
              f"run: fwd={run_fwd:.3f} {run_sm} | {el:.0f}s ETA {eta/60:.0f}m", flush=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return EF, ES


def _rmse_cat(err_list):
    if not err_list:
        return 1e3
    e = np.concatenate(err_list); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  storage={STORAGE}")
    print(f"PF = 固定 r0_seed32 (use_phys={USE_PHYS}, N={R0_PARAMS['n_particles']}), n_seed={N_SEED}, seed={SEED}")
    print(f"TRANS_INFLATE={TRANS_INFLATE}")

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
    # STEP1: forward を1回走らせ、inflate ごとに FFBSm 平滑誤差を precompute
    # ============================================================
    print("\n[STEP1] FFBS forward(1回) + backward per inflate ...")
    EF, ES = predict_errs_ffbs(all_wells, TRANS_INFLATE, SEED, N_SEED)
    FWD_BASE = _rmse_cat([EF[w] for w in all_wells])
    print(f"[STEP1] forward baseline (r0 素) overall = {FWD_BASE:.4f}  (期待 ≈ 8.5〜8.9, LB=7.098)")
    for infl in TRANS_INFLATE:
        smo = _rmse_cat([ES[infl][w] for w in all_wells])
        print(f"   inflate={infl:>4}: ffbs_overall(blend=1)={smo:.4f}  (Δ vs fwd={smo-FWD_BASE:+.4f})")

    order = all_wells
    ef_cat = np.concatenate([EF[w] for w in order])
    es_cat = {infl: np.concatenate([ES[infl][w] for w in order]) for infl in TRANS_INFLATE}

    def blended_overall(infl, blend, mask=None):
        e = (1.0 - blend) * ef_cat + blend * es_cat[infl]
        if mask is not None: e = e[mask]
        e = e[np.isfinite(e)]
        return float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3

    # ============================================================
    # STEP2: in-sample で (inflate, blend) を Optuna(解析的=瞬時)
    # ============================================================
    print("\n[STEP2] Optuna over (trans_inflate, blend)  [in-sample overall] ...")
    study = optuna.create_study(direction="minimize", study_name="ffbsi_r0",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    def objective(trial):
        infl = trial.suggest_categorical("trans_inflate", TRANS_INFLATE)
        blend = trial.suggest_float("blend", 0.0, 1.0)
        return blended_overall(infl, blend)

    remaining = max(0, N_TRIALS - done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best_infl = study.best_params["trans_inflate"]; best_blend = float(study.best_params["blend"])
    best_overall = float(study.best_value)
    smo_best = blended_overall(best_infl, 1.0)
    print(f"[STEP2] best: inflate={best_infl} blend={best_blend:.3f}  "
          f"blended={best_overall:.4f}  (fwd={FWD_BASE:.4f} ffbs_only={smo_best:.4f})")
    verdict = "FFBSが効いている(smo<fwd)" if best_overall < FWD_BASE - 1e-6 else "FFBS無効(blend≈0で fwd 同等)"
    print(f"        判定: {verdict}")

    inflate_table = {}
    grid_bl = np.linspace(0.0, 1.0, 101)
    for infl in TRANS_INFLATE:
        vals = [blended_overall(infl, b) for b in grid_bl]
        j = int(np.argmin(vals))
        inflate_table[str(infl)] = dict(best_blend=float(grid_bl[j]), best_overall=float(vals[j]),
                                        ffbs_only_overall=blended_overall(infl, 1.0))

    # ============================================================
    # STEP3: OOF(GroupKFold by well)
    # ============================================================
    print(f"\n[STEP3] OOF GroupKFold(by well, K={KFOLD}) ...")
    n = len(order); fold_of = np.array([i % KFOLD for i in range(n)])
    lens = np.array([len(EF[w]) for w in order])
    row_fold = np.concatenate([np.full(lens[i], fold_of[i]) for i in range(n)])
    oof_err = np.empty_like(ef_cat); fold_choices = {}
    for fdx in range(KFOLD):
        tr = row_fold != fdx; va = row_fold == fdx
        if not va.any(): continue
        best = (1e9, TRANS_INFLATE[0], 0.0)
        for infl in TRANS_INFLATE:
            for b in grid_bl:
                e = (1.0 - b) * ef_cat[tr] + b * es_cat[infl][tr]; e = e[np.isfinite(e)]
                r = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e9
                if r < best[0]: best = (r, infl, float(b))
        _, ig, bl = best; fold_choices[str(fdx)] = dict(inflate=ig, blend=bl, train_rmse=best[0])
        oof_err[va] = ((1.0 - bl) * ef_cat + bl * es_cat[ig])[va]
        print(f"   fold{fdx}: inflate={ig} blend={bl:.3f} (train={best[0]:.4f})")
    e = oof_err[np.isfinite(oof_err)]; oof_overall = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
    ef_f = ef_cat[np.isfinite(ef_cat)]; oof_fwd = float(np.sqrt(np.mean(ef_f ** 2))) if ef_f.size else 1e3
    print(f"[STEP3] OOF overall(ffbs)={oof_overall:.4f} vs forward={oof_fwd:.4f}  "
          f"({'OOFでも改善' if oof_overall < oof_fwd - 1e-6 else 'OOFでは改善せず'})")

    result = dict(
        algorithm="FFBSm_marginal_smoother__r0_fixed",
        pf="r0_seed32 (v19/artifacts_v19_32seeds/pf_bank.json[0], use_phys=false)",
        r0_params=R0_PARAMS, n_seed=N_SEED, seed=SEED, trans_inflate=TRANS_INFLATE, kfold=KFOLD,
        forward_baseline_overall=FWD_BASE,
        insample_best=dict(trans_inflate=best_infl, blend=best_blend,
                           blended_overall=best_overall, ffbs_only_overall=smo_best),
        per_inflate=inflate_table,
        oof=dict(overall_ffbs=oof_overall, overall_forward=oof_fwd, fold_choices=fold_choices),
        note="FFBSm(平均直出し). forward雲使い回し=n_seed不変・退化なし. smo<fwd かつ OOFでも下がれば本物.",
    )
    save_json(OUT_PATH, result)
    print(f"\nsaved -> {OUT_PATH}")
    print(f"  forward baseline overall = {FWD_BASE:.4f}")
    print(f"  in-sample best inflate={best_infl} blend={best_blend:.3f} -> {best_overall:.4f}")
    print(f"  OOF(K={KFOLD}) ffbs = {oof_overall:.4f}  (forward {oof_fwd:.4f})")


if __name__ == "__main__":
    main()
