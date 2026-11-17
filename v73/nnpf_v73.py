# -*- coding: utf-8 -*-
"""
v73 — PF×ML部分置換 (docs/v73.md)
=====================================
ベース: r0_seed32 + fixed-lag平滑(L=16) + C1学習emission(v51実証, sim_rows_v51.pkl 流用)。
このファイル = PFエンジン(トグル拡張) + 実行モード:
  MODE=base    : Phase0 床の再現。WNN=0(素) / 0.02(C1)。実測期待: 9.12 / 7.44 (NSEED16, v51実測)
  MODE=collect : C2用ロールアウト収集。毎ステップ innovation特徴(F=10) と per-seed 予測(pts_f/pts_s)・
                 log_lik を保存 → artifacts_v73/rollouts_{tag}.pkl
  (C2の学習は train_corrector_v73.py / 補正評価は eval_corrector_v73.py — ロールアウト上で行う)
使い方例:
  MODE=base WNNS=0,0.02 python v73/nnpf_v73.py
  MODE=collect WNN=0.02 NSEED=8 python v73/nnpf_v73.py
注意: RMSE>10 等の変な数字が出たらまず実装バグを疑う(ユーザー指示)。
"""
import os, sys, time, pickle
import numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v73" / "artifacts_v73"; ART.mkdir(parents=True, exist_ok=True)
SIM_FP = Path(os.environ.get("SIMFP", str(P0 / "v51" / "artifacts_v51" / "sim_rows_v51.pkl")))  # 段b: 新emission差し替え用
SIM_FP2 = os.environ.get("SIMFP2", "")           # 段c: 第2学習照合(self等)のsim表
WNN2 = float(os.environ.get("WNN2", "0"))        # 第2照合の強さ: lk *= exp(WNN2·sim2)。sim2=0は中立

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED0 = int(os.environ.get("SEED", "4423098"))   # 別seed再現確認用に上書き可
MODE = os.environ.get("MODE", "base")
N_SEED = int(os.environ.get("NSEED", "32"))   # ★r0_seed32 = 32seed固定(ユーザー指定)
CHUNK = int(os.environ.get("CHUNK", "24"))
SMOOTH_LAG = int(os.environ.get("LAG", "16"))
NW = int(os.environ.get("NW", "10000"))
CUTFRAC = float(os.environ.get("CUTFRAC", "0"))  # v81: >0でprefixをこの割合でカットし残りを疑似eval化(prefixバックテスト)
WNNS = [float(x) for x in os.environ.get("WNNS", os.environ.get("WNN", "0.02")).split(",")]
SIM_STEP = 0.5; NBAND = 90
NFEAT = 10   # innovation特徴数(C2)
# ---- C4: 合成尤度(積emission, v65 pf_engine_multi 忠実移植) ----
TWSRC = os.environ.get("TWSRC", "tw")            # "tw" / "tw+self" / "tw+nbr" / "tw+self+nbr"
SELF_B = float(os.environ.get("SELF_B", "1.0"))  # self emission指数β
NBR_B = float(os.environ.get("NBR_B", "1.0"))    # nbr emission指数β
NEUT = os.environ.get("NEUT", "one")             # 被覆外の扱い: "one"=中立1(v65) / "rowmean"=被覆内平均(逃げ弁封じ)
# ---- C3': 行レベルσ(sigma_net_v73 の OOF mult を消費) ----
SIGROWS = os.environ.get("SIGROWS", "")          # "1"=sig_rows_v73.pkl を使い行レベルσ
SIGGAM = float(os.environ.get("SIGGAM", "1.0"))  # σ_t = gs·mult^γ (γ=0で従来と同一)
# ---- 段a: 伝播のドリフト目標(rate target) ----
RTGT = os.environ.get("RTGT", "zero")            # "zero"(従来=R0) / "ir" / "struct" / "learned" / "oracle" / "trend"(v76)
TWREF = os.environ.get("TWREF", "")              # v76 A2: "1"=既知区間残差の中央値でGRレベル再較正(STRIDE typewell参照項)
LIK = os.environ.get("LIK", "gauss")             # v76 A3: "cauchy"=1/(1+d²) 重い裾尤度
RTK = float(os.environ.get("RTK", "0.001"))      # 引き戻し係数K: rate←mom·rate+K·(target−rate)+noise
RTNOISE = float(os.environ.get("RTNOISE", "0"))  # oracle汚染: 相関ノイズ(rate RMSE単位)で精度→効果曲線を測る
RTNW = int(os.environ.get("RTNW", "101"))        # 汚染ノイズの平滑窓(低周波=予測誤差の現実形)
SGATE = float(os.environ.get("SGATE", "0"))      # >0: structのσでKを行変調 K_t=K·exp(-0.5(sig/SGATE)^2)(structgate)
TPOW = float(os.environ.get("TPOW", "2.0"))      # trustgate: K_t=RTK·P(trust)^TPOW
# ---- 段c: 尤度へのstruct錨項(位置レベルsoft証拠, v65/v38系) ----
ANCM = float(os.environ.get("ANCM", "0"))        # >0で有効: lka=exp(-0.5((TVT−anc)/(σ·ANCM))²)。20=v38 st20相当
# ---- 段c-2: GR尤度形状の学習(LLR曲線置換) ----
LLR = os.environ.get("LLR", "")                  # "1"=llr_curve_v73.npz で lk=exp(LLRSC·LLR(d)) に置換
LLRSC = float(os.environ.get("LLRSC", "1.0"))
_LLR_TAB = None; _LLR_DG = 0.1
STRUCT_FP = Path(os.environ.get("SPFP", str(P0 / "v38" / "artifacts_v38" / "struct_prior_train.pkl")))  # v79: K16錨pkl差替可
MAX_DIST = 1500.0; K_MAX = 3; NEED_REFS = 2      # v65仕様: 近傍≤1500ft・最大3本・2本未満は中立

# r0_seed32 (= v65/warmup_v65.R0_PARAMS = v63 PF_LIB["r0_seed32"])
R0 = {
    "n_particles": 400, "likelihood_scale": 10.536319667317654, "init_pos_std": 0.9331541486020234,
    "init_rate_std": 0.00047753793163756304, "mom": 0.9997911117425017, "vn": 0.0006297672017716754,
    "pn": 0.005014009140727263, "rp": 0.7627309890952053, "rr": 0.00045663430474523125,
    "resamp": 0.032049849143995945, "gr_sig_min": 11.865793233968713, "gr_sig_max": 44.98927964612316,
    "gr_sig_def": 119.25397464636981, "gr_sig_mult": 3.058318309943017,
    "tvt_clip_margin": 150.56111180234504, "rate_clip": 2.0, "gr_power": 1.7826896851395453,
    "hgr_smooth_r": 0, "tw_gr_smooth_r": 0, "jump_prob": 0.002, "jump_std": 5.0,
    "phys_sig": 0.013666550145707291, "use_phys": False,
}

# ---- PFパラメータ切替(v69_4の5本にv73部品を適用する実験用) ----
PFNAME = os.environ.get("PFNAME", "r0")
if PFNAME != "r0":
    import json as _json
    _lib = _json.loads((ART / "pf_lib_v69_4.json").read_text())
    R0 = {**R0, **_lib[PFNAME]}          # 欠キーはr0既定で補完(v69_4 libは全キー有)



# ---------------- 入力構築(v51と同一ロジック) ----------------
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    if r <= 0: return s.to_numpy(np.float32)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)


def _grid(tw_tvt, tw_gr, step=0.2):
    g = np.arange(float(tw_tvt.min()), float(tw_tvt.max()) + step, step)
    return np.interp(g, tw_tvt, tw_gr).astype(np.float64), float(tw_tvt.min()), float(step)


# ---- C4: self/nbr 疑似typewell参照(v65 pf_engine_v65 忠実移植) ----
def _bin_pairs(tvt, gr, gmin, step, G):
    out = np.full(G, np.nan)
    idx = np.round((np.asarray(tvt, float) - gmin) / step).astype(int)
    ok = (idx >= 0) & (idx < G) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        srt = pd.Series(np.asarray(gr, float)[ok]).groupby(idx[ok]).median()
        out[srt.index.to_numpy()] = srt.to_numpy()
    return out


_TRC = None; _REFC = {}

def _train_centroids():
    global _TRC
    if _TRC is None:
        import json, glob as _g
        fp = ART / "train_centroids.json"
        if fp.exists():
            d = __import__("json").loads(fp.read_text())
            _TRC = (d["wids"], np.array(d["xy"]))
        else:
            wids = sorted({Path(p).stem.replace("__horizontal_well", "")
                           for p in _g.glob(str(TRAIN / "*__horizontal_well.csv"))})
            xy = []
            for w in wids:
                h = pd.read_csv(TRAIN / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
                xy.append((float(h["X"].mean()), float(h["Y"].mean())))
            _TRC = (wids, np.array(xy))
            fp.write_text(__import__("json").dumps(dict(wids=wids, xy=[list(p) for p in xy])))
    return _TRC


def _ref_pairs(wid):
    """参照坑井の既知prefix (TVT_input, GR)。"""
    if wid not in _REFC:
        h = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        m = h["TVT_input"].notna().to_numpy() & np.isfinite(h["GR"].to_numpy(float))
        _REFC[wid] = (h["TVT_input"].to_numpy(float)[m], h["GR"].to_numpy(float)[m])
    return _REFC[wid]


def _nbr_pairs_of(wid, cx, cy):
    wids, xy = _train_centroids()
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    refs = []
    for j in np.argsort(d):
        if wids[j] == wid: continue
        if d[j] > MAX_DIST: break
        refs.append(wids[j])
        if len(refs) >= K_MAX: break
    if len(refs) < NEED_REFS: return None
    ts = []; gs_ = []
    for r in refs:
        t_, g_ = _ref_pairs(r); ts.append(t_); gs_.append(g_)
    return (np.concatenate(ts), np.concatenate(gs_))


def build_inputs(hw, tw_tvt, tw_gr, P, wid=None):
    # v81 CUTFRAC: カットはローダ側(tv抽出前)で適用済み
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0: return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf, 0.2)
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
    if TWREF and len(kpos) >= 20:                       # v76 A2: STRIDE typewell参照項(レベル再較正)
        b_ = float(np.nanmedian(gr_sm[kpos] - np.interp(kn_tvtin, tw_tvt, tw_gr_pf)))
        if np.isfinite(b_):
            gr_sm = gr_sm - b_
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    # ---- C4: 参照スタック(tw基準β=1 + self/nbr は個別テンプレ+被覆mask, v65 multi 忠実移植) ----
    grids = [gg.astype(np.float64)]; masks = [np.ones(len(gg), np.float64)]; betas = [1.0]
    if "self" in TWSRC:
        gh = _bin_pairs(kn_tvtin, gr_sm[kpos].astype(np.float64), gmin, gst, len(gg))
        m = np.isfinite(gh); grids.append(np.where(m, gh, gg)); masks.append(m.astype(np.float64))
        betas.append(SELF_B)
    if "nbr" in TWSRC:
        gh = np.full(len(gg), np.nan)
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        np_ = _nbr_pairs_of(wid, cx, cy) if wid is not None else None
        if np_ is not None:
            nt, ng = np_
            ngs = _smooth_radius_values(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
            gh = _bin_pairs(nt, ngs, gmin, gst, len(gg))
        m = np.isfinite(gh); grids.append(np.where(m, gh, gg)); masks.append(m.astype(np.float64))
        betas.append(NBR_B)
    return dict(md=ev["MD"].to_numpy(float), z=ev["Z"].to_numpy(float), gr=gr_sm[ev_pos].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(P["n_particles"]),
                ref_grids=np.stack(grids, 0), ref_masks=np.stack(masks, 0), ref_beta=np.asarray(betas, np.float64))


# ---------------- PFコア(v51 _core + C2特徴収集トグル) ----------------
def _core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, stq, sim, wmap, W_NN,
          P, N, device, gen, refs=None, gsr=None, rtq=None, rtkq=None, ancq=None, ancsq=None,
          sim2=None, W2=0.0, collect_feats=False, dtype=DTYPE):
    """refs={'grids':(B,S,G),'masks':(B,S,G),'beta':(S,)} → emissionを各ソースの積で合成(C4)。
    None なら tw のみ(=grid 単独, 従来と同一計算)。
    gsr=(B,T) 行レベルσ(C3')。None なら井戸定数 gs。
    rtq=(B,T) 段a: rateの行レベル目標(ドリフト先)。None なら従来(ゼロ減衰)。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    L = int(P.get("smooth_lag", 16)); L = max(0, min(L, T))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    feats = torch.zeros((B, T, NFEAT), device=device, dtype=dtype) if collect_feats else None
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    md0 = md[:, 0]
    pm_prev = None; rate_mean_prev = None
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if rtq is not None:                                      # 段a: rate←…+K_t·(target−rate)
            kk = rtkq[:, i][:, None] if rtkq is not None else RTK
            rate_n = rate_n + kk * (rtq[:, i][:, None] - rate)
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
        t = ii - i0c.to(dtype)
        gsd = gsr[:, i][:, None] if gsr is not None else gs[:, None]   # C3': 行レベルσ
        if refs is None:                                        # tw のみ(従来と同一)
            gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
            eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
            d = ((gri[:, None] - eg) / gsd).abs(); dp = d ** GR_POWER
            lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        else:                                                   # ★C4: 各ソース尤度の積(v65 multi 忠実移植)
            one = torch.ones((), device=device, dtype=dtype)
            gidx_last = (glen - 1).clamp_min(0).unsqueeze(1)
            lk = torch.ones_like(w)
            for s in range(refs["grids"].shape[1]):
                gsg = refs["grids"][:, s, :]; gsm = refs["masks"][:, s, :]; bta = refs["beta"][s]
                gA = torch.gather(gsg, 1, i0c); gB = torch.gather(gsg, 1, i0c + 1)
                eg = gA * (1 - t) + gB * t
                eg = torch.where(below, gsg[:, 0:1], eg)
                eg = torch.where(above, torch.gather(gsg, 1, gidx_last), eg)
                d = ((gri[:, None] - eg) / gsd).abs()
                if _LLR_TAB is not None:                        # 段c-2: 学習LLR形状(有界=組込みfloor付き)
                    di = (d / _LLR_DG).long().clamp(0, _LLR_TAB.shape[0] - 1)
                    lk_s = torch.exp((LLRSC * bta) * _LLR_TAB[di])
                elif LIK == "cauchy":                           # v76 A3: Cauchy(重い裾, repeated-bed対策)
                    lk_s = 1.0 / (1.0 + (d * d) * bta)
                else:
                    dp = (d ** GR_POWER) * bta
                    lk_s = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
                mA = torch.gather(gsm, 1, i0c); mB = torch.gather(gsm, 1, i0c + 1)
                cov = ((mA > 0) & (mB > 0)) | below | above     # 被覆格子のみ寄与
                if NEUT == "rowmean" and s > 0:                 # ★逃げ弁封じ: 被覆外=その行の被覆内平均尤度
                    covf = cov.to(dtype)
                    ncov = covf.sum(1, keepdim=True)
                    lmean = (lk_s * covf).sum(1, keepdim=True) / ncov.clamp_min(1.0)
                    fill = torch.where(ncov > 0, lmean, torch.ones_like(lmean))
                    lk = lk * torch.where(cov, lk_s, fill.expand_as(lk_s))
                else:                                           # v65原典: 被覆外=中立(=1)
                    lk = lk * torch.where(cov, lk_s, one)
            lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if W_NN > 0:                                            # C1: 学習emission乗算(v51実証)
            jj = torch.round((v - stq[:, i][:, None]) / SIM_STEP).long() + NBAND
            inb = (jj >= 0) & (jj <= 2 * NBAND)
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * NBAND))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lknn = torch.exp(W_NN * sv)
        else:
            lknn = torch.ones_like(w)
        if sim2 is not None:                                    # 段c: 第2学習照合(self等)。帯外/被覆外=0=中立
            jj2 = torch.round((v - stq[:, i][:, None]) / SIM_STEP).long() + NBAND
            inb2 = (jj2 >= 0) & (jj2 <= 2 * NBAND)
            sim2_i = sim2[:, i, :].to(dtype)[wmap]
            sv2 = torch.gather(sim2_i, 1, jj2.clamp(0, 2 * NBAND))
            sv2 = torch.where(inb2, sv2, torch.zeros_like(sv2))
            lknn = lknn * torch.exp(W2 * sv2)
        if ancq is not None:                                     # 段c: struct位置錨(soft, σ=場の申告×ANCM)
            asig = (ancsq[:, i][:, None] * ANCM).clamp_min(1e-6)
            da = (v - ancq[:, i][:, None]) / asig
            da2 = da * da
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
            lknn = lknn * lka
        w_new = torch.where(obs[:, None], w * lk * lknn, w * lknn)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        # ---- C2: innovation特徴(事後の状態記述, resample前) ----
        if collect_feats:
            pmv = (w * v).sum(1)                                            # 事後平均TVT
            dv = v - pmv[:, None]
            pstd = (w * dv * dv).sum(1).clamp_min(1e-12).sqrt()
            pmad = (w * dv.abs()).sum(1).clamp_min(1e-9)
            ess = 1.0 / (w * w).sum(1).clamp_min(1e-12)
            rate_mean = (w * rate).sum(1)
            # 事後平均位置での typewell GR とその残差
            iiq = (pmv - vmin) / step; i0q = iiq.floor().long().clamp(0, 1_000_000)
            i0q = torch.minimum(i0q.clamp_min(0), (glen - 2).clamp_min(0))
            tq = (iiq - i0q.to(dtype)).clamp(0, 1)
            egq = torch.gather(grid, 1, i0q[:, None]).squeeze(1) * (1 - tq) + \
                  torch.gather(grid, 1, (i0q + 1)[:, None]).squeeze(1) * tq
            grow = torch.where(torch.isnan(gri), torch.zeros_like(gri), gri)
            rres = torch.where(obs, (grow - egq) / gs, torch.zeros_like(gri))
            innov = torch.zeros_like(pmv) if pm_prev is None else \
                (pmv - pm_prev) / dm - (rate_mean_prev if rate_mean_prev is not None else 0.0)
            fs = torch.stack([
                rres,                                            # 0 GR残差(署名つき, /gs)
                rres.abs(),                                      # 1 |GR残差|
                pstd.clamp(0, 50.0),                             # 2 事後std(ft)
                torch.log10((ess / N).clamp_min(1e-6)),          # 3 ESS比(log)
                rate_mean - ir,                                  # 4 rate vs 初期dip
                innov.clamp(-1.0, 1.0),                          # 5 innovation速度
                avg.clamp(0, 1.0),                               # 6 尤度w平均(GRがどこかに合うか)
                lk.max(1).values.clamp(0, 1.0),                  # 7 尤度max
                (pstd / (1.349 * pmad)).clamp(0, 10.0),          # 8 双峰度(std/MAD)
                ((cur - md0) / 5000.0).clamp(0, 3.0),            # 9 錨からの距離
            ], dim=1)
            feats[:, i, :] = torch.where(act[:, None], fs, torch.zeros_like(fs))
            pm_prev = pmv.detach(); rate_mean_prev = rate_mean.detach()
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
    return pts_f, pts_s, log_lik, feats


# ---------------- 実行(チャンク×seed複製, v51と同じ組み方) ----------------
def run_pf(inps, sims_w, P, seed, n_seeds, chunk, W_NN, collect_feats=False, sgr_w=None, rtq_w=None, rtk_w=None,
           anc_w=None, sims2_w=None):
    N = int(P["n_particles"]); S = n_seeds; W = len(inps)
    out_f = [None] * W; out_s = [None] * W; out_ll = [None] * W; out_feats = [None] * W
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; wq = len(sub)
        Tmax = max(len(x["md"]) for x in sub); Gmax = max(len(x["gg"]) for x in sub)
        md = torch.zeros((wq, Tmax)); z = torch.zeros((wq, Tmax))
        gr = torch.full((wq, Tmax), float("nan")); valid = torch.zeros((wq, Tmax), dtype=torch.bool)
        grid = torch.zeros((wq, Gmax)); glen = torch.zeros(wq, dtype=torch.long)
        vmin = torch.zeros(wq); step = torch.zeros(wq); gs = torch.zeros(wq)
        ls = torch.zeros(wq); ir = torch.zeros(wq)
        stq = torch.zeros((wq, Tmax)); simt = torch.full((wq, Tmax, 2 * NBAND + 1), -1.0, dtype=torch.float16)
        sim2t = None
        if sims2_w is not None:
            sim2t = torch.zeros((wq, Tmax, 2 * NBAND + 1), dtype=torch.float16)   # 0=中立
        Sref = sub[0]["ref_grids"].shape[0]
        rg = torch.zeros((wq, Sref, Gmax)); rm = torch.zeros((wq, Sref, Gmax))
        gsrt = None
        if sgr_w is not None:
            gsrt = torch.zeros((wq, Tmax))
        rtqt = None
        if rtq_w is not None:
            rtqt = torch.zeros((wq, Tmax))
        rtkt = None
        if rtk_w is not None:
            rtkt = torch.zeros((wq, Tmax))
        anct = ancst = None
        if anc_w is not None:
            anct = torch.zeros((wq, Tmax)); ancst = torch.full((wq, Tmax), 1e9)
        for b, x in enumerate(sub):
            Tn = len(x["md"]); G = len(x["gg"])
            md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
            if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
            z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
            gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
            grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
            if G < Gmax: grid[b, G:] = grid[b, G - 1]
            glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
            rg[b, :, :G] = torch.from_numpy(x["ref_grids"].astype("float32"))
            rm[b, :, :G] = torch.from_numpy(x["ref_masks"].astype("float32"))
            if G < Gmax: rg[b, :, G:] = rg[b, :, G - 1:G]       # mask側は0のまま(=格子外は中立)
            if gsrt is not None:                                 # C3': σ_t = gs·mult^γ (無い井戸はgs定数)
                gsrt[b, :] = float(x["gs"])
                mv = sgr_w[c0 + b]
                if mv is not None:
                    Ts = min(Tn, len(mv))
                    gsrt[b, :Ts] = float(x["gs"]) * torch.from_numpy(
                        np.power(mv[:Ts].astype(np.float32), SIGGAM))
            if rtqt is not None:                                 # 段a: rate目標(無い行は最終値/ir)
                rtqt[b, :] = float(x["ir"])
                rv = rtq_w[c0 + b]
                if rv is not None:
                    Ts = min(Tn, len(rv))
                    rtqt[b, :Ts] = torch.from_numpy(rv[:Ts].astype(np.float32))
                    if Ts < Tmax: rtqt[b, Ts:] = rtqt[b, Ts - 1]
            if rtkt is not None:                                 # 段a: 行レベルK(σゲート。無い井戸=0=引かない)
                kv = rtk_w[c0 + b]
                if kv is not None:
                    Ts = min(Tn, len(kv))
                    rtkt[b, :Ts] = torch.from_numpy(kv[:Ts].astype(np.float32))
            if anct is not None:                                 # 段c: struct錨(無い井戸=σ1e9=中立)
                av = anc_w[c0 + b]
                if av is not None:
                    at, asg = av; Ts = min(Tn, len(at))
                    anct[b, :Ts] = torch.from_numpy(at[:Ts].astype(np.float32))
                    if Ts < Tmax: anct[b, Ts:] = anct[b, Ts - 1]
                    ancst[b, :Ts] = torch.from_numpy(asg[:Ts].astype(np.float32))
            if W_NN > 0:
                sd = sims_w[c0 + b]; Ts = min(Tn, len(sd["st"]))
                stq[b, :Ts] = torch.from_numpy(sd["st"][:Ts])
                if Ts < Tmax: stq[b, Ts:] = stq[b, Ts - 1]
                simt[b, :Ts] = torch.from_numpy(sd["sim"][:Ts])
            if sim2t is not None:
                sd2 = sims2_w[c0 + b]
                if sd2 is not None:
                    Ts2 = min(Tn, len(sd2["st"]))
                    sim2t[b, :Ts2] = torch.from_numpy(sd2["sim"][:Ts2])
                    if W_NN <= 0:                                # 帯中心stをsim2側から補完
                        stq[b, :Ts2] = torch.from_numpy(sd2["st"][:Ts2])
                        if Ts2 < Tmax: stq[b, Ts2:] = stq[b, Ts2 - 1]
        dev = DEVICE
        to = lambda tt: tt.to(dev); rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        wmap = torch.arange(wq, device=dev).repeat_interleave(S)
        refs = dict(grids=rep(to(rg)), masks=rep(to(rm)),
                    beta=to(torch.from_numpy(sub[0]["ref_beta"].astype("float32"))))
        pf, ps, ll, ft = _core(rep(to(md)), rep(to(z)), rep(to(gr)), rep(to(valid)), rep(to(grid)),
                               to(glen).repeat_interleave(S), to(vmin).repeat_interleave(S),
                               to(step).repeat_interleave(S), to(gs).repeat_interleave(S),
                               to(ls).repeat_interleave(S), to(ir).repeat_interleave(S),
                               rep(to(stq)), to(simt), wmap, W_NN,
                               {**P, "smooth_lag": SMOOTH_LAG}, N, dev, gen,
                               refs=refs, gsr=(rep(to(gsrt)) if gsrt is not None else None),
                               rtq=(rep(to(rtqt)) if rtqt is not None else None),
                               rtkq=(rep(to(rtkt)) if rtkt is not None else None),
                               ancq=(rep(to(anct)) if anct is not None else None),
                               ancsq=(rep(to(ancst)) if ancst is not None else None),
                               sim2=(to(sim2t) if sim2t is not None else None), W2=WNN2,
                               collect_feats=collect_feats)
        pf = pf.view(wq, S, -1).float().cpu().numpy(); ps = ps.view(wq, S, -1).float().cpu().numpy()
        ll = ll.view(wq, S).cpu().numpy()
        if collect_feats:
            ft = ft.view(wq, S, Tmax, NFEAT).half().cpu().numpy()
        for b in range(wq):
            T = len(sub[b]["md"])
            out_f[c0 + b] = pf[b, :, :T]; out_s[c0 + b] = ps[b, :, :T]; out_ll[c0 + b] = ll[b]
            if collect_feats: out_feats[c0 + b] = ft[b, :, :T, :]
        if dev == "cuda": torch.cuda.synchronize()
        print(f"  chunk {c0 // chunk + 1}/{(W + chunk - 1) // chunk} done", flush=True)
    return out_f, out_s, out_ll, out_feats


def seed_combine(pr, llw, ls_scale):
    """per-seed 予測(S,T)を log_lik ソフトマックスで合成(v51と同一)。"""
    S = pr.shape[0]
    if np.isfinite(llw).any():
        mx = np.nanmax(llw[np.isfinite(llw)])
        lkw = np.where(np.isfinite(llw), llw - mx, -np.inf)
        ww = np.exp(lkw / max(ls_scale, 1e-6)); ssum = ww.sum()
        ww = ww / ssum if ssum > 1e-300 else np.full(S, 1.0 / S)
    else:
        ww = np.full(S, 1.0 / S)
    return (ww[:, None] * pr).sum(0), ww


def load_data():
    simd = pickle.loads(SIM_FP.read_bytes()) if any(w > 0 for w in WNNS) else None
    wells = sorted(simd.keys()) if simd is not None else sorted(
        set(f.name.split("__")[0] for f in TRAIN.glob("*__horizontal_well.csv")))
    wells = wells[:NW]
    DATA = []
    for wn in wells:
        hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        if CUTFRAC > 0:                                 # v81: カットはtv抽出前にhwレベルで適用
            kn0 = hw.index[hw["TVT_input"].notna()]
            ci = int(len(kn0) * CUTFRAC)
            if ci < 20 or len(kn0) - ci < 100: continue
            hw = hw.loc[:kn0[-1]].copy()
            hw.loc[kn0[ci + 1:], "TVT_input"] = np.nan
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or (hw["TVT_input"].notna()).sum() < 10: continue
        x = build_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), R0, wid=wn)
        if x is None: continue
        DATA.append((wn, x, ev["TVT"].to_numpy(float)))
    return DATA, simd


if __name__ == "__main__":
    t00 = time.time()
    DATA, SIMD = load_data()
    wns = [d[0] for d in DATA]; inps = [d[1] for d in DATA]; tvs = [d[2] for d in DATA]
    sims_w = [SIMD[w] for w in wns] if SIMD is not None else [None] * len(wns)
    print(f"MODE={MODE} wells={len(DATA)} LAG={SMOOTH_LAG} NSEED={N_SEED} CHUNK={CHUNK} WNNS={WNNS} "
          f"TWSRC={TWSRC} SELF_B={SELF_B} NBR_B={NBR_B}", flush=True)
    ls_scale = float(R0["likelihood_scale"])

    sgr_w = None
    if SIGROWS:
        SM = pickle.loads((ART / "sig_rows_v73.pkl").read_bytes())
        sgr_w = [SM.get(w) for w in wns]
        print(f"SIGROWS on: mult有り {sum(m is not None for m in sgr_w)}/{len(wns)}井  γ={SIGGAM}", flush=True)

    rtq_w = None; rtk_w = None
    if RTGT != "zero":
        SP = pickle.loads(STRUCT_FP.read_bytes()) if RTGT in ("struct", "structgate", "trustgate") else None
        TR = pickle.loads((ART / "trust_rows_v73.pkl").read_bytes()) if RTGT == "trustgate" else None
        if RTGT in ("structgate", "trustgate"):
            rtk_w = []
        MU = pickle.loads((ART / "mu_rows_v73.pkl").read_bytes()) if RTGT == "learned" else None
        rtq_w = []; nst = 0
        for (wn, x, tv) in DATA:
            T = len(x["md"])
            mu = None
            if RTGT in ("struct", "structgate", "trustgate"):
                sp = SP.get(wn)
                if sp is not None and len(sp["tvt"]) == T:
                    sline = np.asarray(sp["tvt"], float) + x["z"]
                    m_ = np.gradient(sline, x["md"])
                    m_ = pd.Series(m_).rolling(21, center=True, min_periods=1).median().to_numpy()
                    mu = np.clip(m_, -2.0, 2.0).astype(np.float32); nst += 1
                    if RTGT == "structgate":                     # σゲート: K_t=RTK·exp(-0.5(sig/SGATE)^2)
                        sig = np.clip(np.asarray(sp["sig"], float), 0.5, 60.0)
                        rtk_w.append((RTK * np.exp(-0.5 * (sig / max(SGATE, 1e-6)) ** 2)).astype(np.float32))
                    elif RTGT == "trustgate":                    # 学習trustゲート: K_t=RTK·P^TPOW (SGATE>0でσゲート併用)
                        pv = TR.get(wn)
                        if pv is not None and len(pv) == T:
                            kt = RTK * np.power(pv.astype(np.float32), TPOW)
                            if SGATE > 0:
                                sig = np.clip(np.asarray(sp["sig"], float), 0.5, 60.0)
                                kt = kt * np.exp(-0.5 * (sig / SGATE) ** 2)
                            rtk_w.append(kt.astype(np.float32))
                        else:
                            rtk_w.append(None)
                elif RTGT in ("structgate", "trustgate"):
                    rtk_w.append(None)                           # struct無し坑井=引かない(K=0)
            elif RTGT == "learned":
                mv = MU.get(wn)
                if mv is not None and len(mv) == T:
                    mu = mv.astype(np.float32); nst += 1
            elif RTGT == "trend":                               # v76 A1: drilled-trend(層追従なら dS/dMD=dZ/dMD)
                m_ = np.gradient(x["z"], x["md"])
                m_ = pd.Series(m_).rolling(101, center=True, min_periods=1).median().to_numpy()
                m_ = np.nan_to_num(m_)
                # ★ビルド区間(|dzdmd|大)は層追従不成立でS≈一定=目標0が正解。clipでなくゼロ戻し
                mu = np.where(np.abs(m_) <= 0.03, m_, 0.0).astype(np.float32); nst += 1
            elif RTGT == "oracle":                              # 診断専用: 真rate=段aの理論天井
                if tv is not None and len(tv) >= T:
                    m_ = np.gradient(tv[:T] + x["z"], x["md"])
                    m_ = pd.Series(m_).rolling(21, center=True, min_periods=1).median().to_numpy()
                    m_ = np.nan_to_num(m_)
                    if RTNOISE > 0:                             # 相関ノイズ汚染(精度→効果曲線)
                        rgn = np.random.RandomState(abs(hash(wn)) % (2**31))
                        nz = pd.Series(rgn.randn(T)).rolling(RTNW, center=True, min_periods=1).mean().to_numpy()
                        sdv = np.std(nz)
                        m_ = m_ + (nz / max(sdv, 1e-9)) * RTNOISE
                    mu = np.clip(m_, -2.0, 2.0).astype(np.float32); nst += 1
            rtq_w.append(mu if mu is not None else np.full(T, x["ir"], np.float32))
        print(f"RTGT={RTGT} RTK={RTK}  target有り {nst}/{len(DATA)}井", flush=True)

    if LLR:
        _z = np.load(ART / "llr_curve_v73.npz")
        globals()["_LLR_TAB"] = torch.tensor(_z["llr"], dtype=DTYPE, device=DEVICE)
        print(f"LLR曲線置換 on (LLRSC={LLRSC})", flush=True)

    sims2_w = None
    if SIM_FP2 and WNN2 != 0:
        SIMD2 = pickle.loads(Path(SIM_FP2).read_bytes())
        sims2_w = [SIMD2.get(w) for w in wns]
        print(f"SIM2: {Path(SIM_FP2).name} 有り{sum(s is not None for s in sims2_w)}/{len(wns)}井 WNN2={WNN2}", flush=True)

    anc_w = None
    if ANCM > 0:
        SPA = pickle.loads(STRUCT_FP.read_bytes())
        anc_w = []
        for (wn, x, tv) in DATA:
            sp = SPA.get(wn); T = len(x["md"])
            if sp is not None and len(sp["tvt"]) == T:
                anc_w.append((np.asarray(sp["tvt"], np.float32),
                              np.clip(np.asarray(sp["sig"], np.float32), 0.5, 60.0)))
            else:
                anc_w.append(None)
        print(f"ANCM={ANCM}  錨有り {sum(a is not None for a in anc_w)}/{len(DATA)}井", flush=True)

    if MODE == "base":
        for W_NN in WNNS:
            t0 = time.time()
            pf, ps, ll, _ = run_pf(inps, sims_w, R0, SEED0, N_SEED, CHUNK, W_NN, sgr_w=sgr_w, rtq_w=rtq_w, rtk_w=rtk_w,
                                   anc_w=anc_w, sims2_w=sims2_w)
            es = []
            for b in range(len(DATA)):
                comb, _ = seed_combine(ps[b], ll[b], ls_scale)
                e = comb[:len(tvs[b])] - tvs[b][:len(comb)]; es.append(e[np.isfinite(e)])
            e = np.concatenate(es)
            tag = PFNAME + " " + TWSRC + (f" RTGT={RTGT}@K{RTK:g}" if RTGT != "zero" else "") + \
                  (f" SG{SGATE:g}" if SGATE > 0 else "") + (f" NZ{RTNOISE:g}" if RTNOISE > 0 else "") + \
                  (f" ANCM{ANCM:g}" if ANCM > 0 else "") + (f" W2={WNN2:g}" if WNN2 != 0 else "") + \
                  (f" LLR@{LLRSC:g}" if LLR else "") + (" TWREF" if TWREF else "") + \
                  (f" LIK={LIK}" if LIK != "gauss" else "")
            print(f"[base r0_seed32 lag{SMOOTH_LAG} {tag}] W_NN={W_NN:g}: overall={np.sqrt(np.mean(e**2)):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elif MODE == "dump":
        # 段g用: per-seed 平滑軌跡+log_lik を保存(seed合成のオフライン掃引に使う)
        W_NN = WNNS[0]; t0 = time.time()
        pf, ps, ll, _ = run_pf(inps, sims_w, R0, SEED0, N_SEED, CHUNK, W_NN, sgr_w=sgr_w, rtq_w=rtq_w, rtk_w=rtk_w,
                               anc_w=anc_w, sims2_w=sims2_w)
        out = {}
        es = []
        for b, wn in enumerate(wns):
            tv = tvs[b][:ps[b].shape[1]].astype(np.float32)
            comb, _ = seed_combine(ps[b], ll[b], ls_scale)
            e = comb[:len(tv)] - tv; es.append(e[np.isfinite(e)])
            # ★per-seed誤差(小値)をf16保存(絶対TVTのf16は量子化±2.7ftで壊れる)。ps復元=err+true
            out[wn] = dict(err=(ps[b] - tv[None, :]).astype(np.float16), ll=ll[b].astype(np.float32), true=tv)
        e = np.concatenate(es)
        tagd = os.environ.get("DUMPTAG", "champ")
        fp = ART / f"seed_dump_{tagd}.pkl"
        fp.write_bytes(pickle.dumps(out, protocol=4))
        print(f"[dump] baseline(combine)={np.sqrt(np.mean(e**2)):.4f}  saved {fp.name} "
              f"({fp.stat().st_size/1e6:.0f}MB, {time.time()-t0:.0f}s)", flush=True)

    elif MODE == "collect":
        W_NN = WNNS[0]
        t0 = time.time()
        pf, ps, ll, ft = run_pf(inps, sims_w, R0, SEED0, N_SEED, CHUNK, W_NN, collect_feats=True)
        out = {}
        es = []
        for b, wn in enumerate(wns):
            tv = tvs[b][:ps[b].shape[1]].astype(np.float32)
            comb, _ = seed_combine(ps[b], ll[b], ls_scale)
            e = comb[:len(tv)] - tv; es.append(e[np.isfinite(e)])
            out[wn] = dict(
                feats=ft[b],                                   # (S,T,F) f16
                e_f=(pf[b] - tv[None, :]).astype(np.float16),  # per-seed forward誤差
                e_s=(ps[b] - tv[None, :]).astype(np.float16),  # per-seed smoothed誤差
                ll=ll[b].astype(np.float32), true=tv)
        e = np.concatenate(es)
        rb = np.sqrt(np.mean(e ** 2))
        tag = f"w{W_NN:g}_s{N_SEED}"
        fp = ART / f"rollouts_{tag}.pkl"
        fp.write_bytes(pickle.dumps(out, protocol=4))
        print(f"[collect] baseline(combine) overall={rb:.4f}  saved {fp.name} "
              f"({fp.stat().st_size/1e9:.2f}GB, {time.time()-t0:.0f}s)", flush=True)

    print(f"total {time.time()-t00:.0f}s")
