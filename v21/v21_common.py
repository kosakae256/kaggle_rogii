"""
v21 共通エンジン ―― Particle Filter に NN を挟む(案A 学習尤度 / 案B 学習遷移)
================================================================================
方針: PFの骨格は維持し、観測尤度(案A)/遷移(案B)を差し替え可能にする。
  - `_pf_core(..., lmap=None, drift=None, lam_drift=0.0)`
      lmap 指定 → 尤度を学習コストマップに置換(案A)。
      drift 指定 → 遷移を学習ドリフトに混合(案B)。両方 None なら素の r0 PF。
  - NN は PF の外で「1坑井1回」走らせて lmap/drift を作る → PFループ内は軽い lookup のみ(GPU高速)。

r0 パラメータ = v19 バンクの round0 のみ。物理項は既定 OFF(NN効果を切り分けるため)。
GPU: 坑井×シードをバッチ化(B=W*S)。lmap/drift は坑井ごと → well_index で参照。

使い方(ノートから):
    import v21_common as C
    ns = C.load_v9_ns()
    p  = C.r0_params(ns, use_phys=False, n_seeds=8)
    wells = C.list_train_wells()
    # 素のPF:
    res = C.eval_wells(wells[:30], ns, p)
    # 案A: lmap_fn(wid, inp) -> (T,G) float32 を渡す
    # 案B: drift_fn(wid, inp) -> (T,) float32 を渡す
"""
import os, glob, json, time, threading
from pathlib import Path
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "rogii-wellbore-geology-prediction"
V9_NB       = PROJECT_DIR / "v9" / "pf_tuning_create_train_csv_multi_pf_v9.ipynb"
BANK_PATH   = PROJECT_DIR / "v19" / "artifacts_v19" / "pf_bank.json"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.float32
SEED     = 0
LMAP_STEP = 0.5      # 学習コストマップ(案A)の tvt グリッド刻み(粗め=VRAM節約)
LMAP_MARGIN = 60.0   # typewell tvt 範囲の外側マージン(ft)


# ============================ v9 名前空間(ヘルパ再利用) ============================
def load_v9_ns():
    """v9ノートを exec して特徴/PFヘルパ(_smooth_radius_values,_grid,_gr_sig,_make_pf_params,FI等)を得る。"""
    nb = json.load(open(V9_NB, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = src.replace(
        r'dataset_path = Path(r"C:\Users\kosaka256\Documents\rogii\rogii-wellbore-geology-prediction")',
        f'dataset_path = Path(r"{DATA_DIR}")')
    src = src.replace("cache=True", "cache=False")
    marker = 'train_paths = sorted((CFG.dataset_path / "train").glob'
    cut = src.find(marker)
    if cut > 0:
        src = src[:cut]
    ns = {"__name__": "v9lib"}
    print("[v9] exec 中(FI/DI fit を含む)...")
    exec(compile(src, "<v9_inlined>", "exec"), ns)
    return ns


def r0_params(ns, use_phys=False, n_seeds=8):
    """v19バンク round0 の raw を v9 _make_pf_params で内部パラメータ p に変換。物理は既定OFF。"""
    bank = json.loads(BANK_PATH.read_text())
    e0 = bank[0]
    raw = dict(e0["params"]); raw["n_seeds"] = n_seeds
    p = ns["_make_pf_params"](raw, 0)
    p["PF_USE_PHYS"] = bool(use_phys and raw.get("use_phys", False))
    p["PF_PHYS_SIG"] = float(raw.get("phys_sig", 1.0))
    p["label"] = f"r{e0['round']}"
    print(f"[v21] r0 params: {p['label']} N={p['ANCC_N']} seeds={n_seeds} phys={'on' if p['PF_USE_PHYS'] else 'off'}")
    return p


def list_train_wells():
    return sorted(os.path.basename(q).split("__")[0]
                  for q in glob.glob(str(DATA_DIR / "train" / "*__horizontal_well.csv")))


# ============================ 坑井データ準備 ============================
import pandas as pd

def _heel_affine(kgr, tw_at_k):
    """既知ゾーンで 水平GR ≈ a*typeGR + b の a,b を返す(ヒール較正, 合法)。"""
    ok = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if ok.sum() < 10 or np.var(tw_at_k[ok]) < 1e-6:
        return 1.0, 0.0
    x = tw_at_k[ok]; y = kgr[ok]
    a = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    a = float(np.clip(a, 0.2, 3.0))                          # 暴走ガード
    b = float(np.mean(y) - a * np.mean(x))
    return a, b


# ---- GR 別表現(MA/AGC/rank/微分/分位点)。水平とtypewellに同じ変換を当て、typewell生GRスケールへ正規化 ----
def _repr_transform(name, h, md_h, tw, tw_tvt, win):
    """raw以外の表現。両側に同変換 → typewell変換分布で正規化 → 生typewell GRスケールへ(PFパラメータ保持)。"""
    nm = name.lower(); win = int(win)
    def T(sig, x):
        s = pd.Series(np.asarray(sig, float))
        if nm.startswith("ma"):                              # 移動平均(平滑)
            return s.rolling(win, center=True, min_periods=1).mean().to_numpy()
        if nm.startswith("agc"):                             # 局所正規化(利得ドリフト吸収)
            m = s.rolling(win, center=True, min_periods=1).mean().to_numpy()
            sd = s.rolling(win, center=True, min_periods=1).std().to_numpy()
            sd = np.where(~np.isfinite(sd) | (sd < 1e-6), 1.0, sd)
            return (np.asarray(sig, float) - m) / sd
        if nm == "rank":                                     # 大域ランク化(単調変換に不変=Spearman的)
            return np.argsort(np.argsort(np.asarray(sig, float))) / max(len(sig) - 1, 1)
        if nm == "deriv":                                    # 微分(オフセット除去・層境界強調)
            return np.gradient(np.asarray(sig, float), np.asarray(x, float))
        return np.asarray(sig, float)
    Th = T(h, md_h); Ttw = T(tw, tw_tvt)
    mt = float(np.nanmean(Ttw)); st = float(np.nanstd(Ttw) + 1e-6)
    gm = float(np.nanmean(tw)); gsd = float(np.nanstd(tw) + 1e-6)
    f = lambda z: (gm + gsd * ((np.asarray(z, float) - mt) / st)).astype(np.float64)
    return f(Th), f(Ttw)


def _quantile_map(h, tw, nq=201):
    """水平GRの各値を typewell GR の分位点に写像(CDFマッチ=非線形較正)。typewellは生のまま。"""
    h = np.asarray(h, float); tw = np.asarray(tw, float)
    qs = np.linspace(0, 1, nq); hq = np.nanquantile(h, qs); tq = np.nanquantile(tw, qs)
    return np.interp(h, hq, tq).astype(np.float64)


def build_inputs(wid, ns, p, split="train", calibrate=False, repr_="raw", repr_win=3):
    """1坑井の PF 入力 + matcher 窓 + 真値(train)を返す。test に実在する列のみから生成。
    repr_: GR表現 = raw / ma / agc / rank / deriv / quantile(両側に同変換、PFが消費)。repr_win=窓(ma/agc用)。
    calibrate=True: 既知ゾーンで GR目盛り(a,b)を合わせ、typewellグリッドを水平スケールへ較正(ヒール較正)。表現と併用可。"""
    base = DATA_DIR / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 5 or len(ev) < 1:
        return None
    tw_tvt = tw["TVT"].to_numpy(np.float64); tw_gr = tw["GR"].to_numpy(np.float64)
    if len(tw_tvt) < 3:
        return None
    _smooth = ns["_smooth_radius_values"]; _grid = ns["_grid"]; _gr_sig = ns["_gr_sig"]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    md_ev = ev["MD"].to_numpy(np.float64); z_ev = ev["Z"].to_numpy(np.float64)
    md_full = hw["MD"].astype(float).to_numpy(float)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(ev.index)

    # ---- GR表現の選択(両側に同変換 → グリッド/gs を一貫構成)----
    REPR = (repr_ or "raw").lower()
    if REPR == "raw":                                            # v9流: PF自身の平滑半径(r0は水平窓3)
        tw_gr_pf = _smooth(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
        gr_proc = _smooth(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"]).astype(np.float64)
        gs = _gr_sig(hw, tw_tvt, tw_gr_pf, p)
    else:
        if REPR == "quantile":                                  # 非線形較正(CDFマッチ)。typewellは生
            tw_gr_pf = _smooth(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
            gr_proc = _quantile_map(_smooth(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"]).astype(np.float64), tw_gr_pf)
        else:                                                   # ma/agc/rank/deriv
            gr_proc, tw_gr_pf = _repr_transform(REPR, gr_full, md_full, tw_gr, tw_tvt, repr_win)
        kn_pos = hw.index.get_indexer(kn.index)                 # gs は表現後の残差から再計算
        resid = gr_proc[kn_pos] - np.interp(kn["TVT_input"].to_numpy(np.float64), tw_tvt, tw_gr_pf)
        g0 = float(np.nanstd(resid)); g0 = g0 if (np.isfinite(g0) and g0 > 0) else float(p["PF_GR_SIG_DEF"])
        gs = float(np.clip(g0 * p["PF_GR_SIG_MULT"], p["PF_GR_SIG_MIN"], p["PF_GR_SIG_MAX"]))
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)                       # 微細typewellグリッド
    gr_ev = gr_proc[ev_pos].astype(np.float64)

    # ヒール較正: 既知ゾーンで a,b を求め、typewell グリッド(=PFの期待GR)を水平スケールへ
    a_cal, b_cal = 1.0, 0.0
    if calibrate:
        kn_pos = hw.index.get_indexer(kn.index)
        kgr = gr_proc[kn_pos].astype(np.float64)
        tw_at_k = np.interp(kn["TVT_input"].to_numpy(np.float64), tw_tvt, tw_gr_pf)
        a_cal, b_cal = _heel_affine(kgr, tw_at_k)
        gg = (a_cal * gg + b_cal).astype(gg.dtype)               # 期待GRを水平スケールへ
        gs = float(max(gs * a_cal, 1e-3))                        # 目盛り倍率に合わせてスケールも調整

    # 学習コストマップ(案A)用の tvt グリッド(粗め)
    lo = float(np.min(tw_tvt) - LMAP_MARGIN); hi = float(np.max(tw_tvt) + LMAP_MARGIN)
    lgrid = np.arange(lo, hi + LMAP_STEP, LMAP_STEP)
    lvmin = float(lgrid[0]); lglen = len(lgrid)

    # 真値(train のみ; eval の TVT)
    tvt_true = ev["TVT"].to_numpy(np.float64) if ("TVT" in ev.columns and ev["TVT"].notna().any()) else None
    last_tvt = float(kn["TVT_input"].iloc[-1])
    md_since = (md_ev - float(kn["MD"].iloc[-1])).astype(np.float64)

    return dict(
        wid=wid, md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir,
        N=int(p["ANCC_N"]), tw_tvt=tw_tvt, tw_gr=tw_gr_pf,
        lvmin=lvmin, lstep=LMAP_STEP, lglen=lglen, lgrid=lgrid,
        tvt_true=tvt_true, last_tvt=last_tvt, md_since=md_since,
        a_cal=a_cal, b_cal=b_cal,
    )


# ============================ GPU PF(lmap/drift 差し替え可能) ============================
def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen,
             lmap=None, lmap_vmin=None, lmap_step=None, lmap_glen=None, well_index=None,
             drift=None, lam_drift=0.0, dtype=DTYPE):
    """順方向GPU PF。lmap指定で尤度を学習版に、drift指定で遷移を混合(案B)。返り pts(B,T), log_lik(B,)。"""
    B, T = md.shape
    ALPHA = p["ANCC_ALPHA"]; RN = p["ANCC_RN"]; PN = p["ANCC_PN"]; IR = p["ANCC_IR"]; IS = p["ANCC_IS"]
    RP = p["ANCC_RP"]; RR = p["ANCC_RR"]; RESAMP = p["PF_RESAMP"]; RATE_CLIP = p["PF_RATE_CLIP"]
    GR_POWER = p["PF_GR_POWER"]; JUMP_PROB = p["PF_JUMP_PROB"]; JUMP_STD = p["PF_JUMP_STD"]; CLIP = p["PF_TVT_CLIP_MARGIN"]
    USE_P = bool(p.get("PF_USE_PHYS", False)); PHYS_SIG = float(p.get("PF_PHYS_SIG", 1.0))
    use_lmap = lmap is not None
    use_drift = drift is not None and lam_drift > 0.0

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

    pos = ls[:, None] + IS * rn()
    rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP
    tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1))
    g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0
    arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    dipp = ir[:, None]

    for i in range(T):
        act = valid[:, i]
        cur = md[:, i]
        dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if use_drift:                                   # 案B: 学習ドリフトを混合
            dr = drift[well_index, i][:, None]
            rate_n = (1.0 - lam_drift) * rate_n + lam_drift * dr
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None])
        pos_n = tvt + zi
        am = act[:, None]
        pos = torch.where(am, pos_n, pos)
        rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]
        obs = act & ~torch.isnan(gri)
        v = pos - zi

        if use_lmap:                                    # 案A: 学習コストマップを lookup
            Lr = lmap[well_index, i]                    # (B, Gl)
            gi = ((v - lmap_vmin[:, None]) / lmap_step).round().long()
            gi = gi.clamp_min(0)
            gi = torch.minimum(gi, (lmap_glen - 1).clamp_min(0)[:, None])
            ll_part = torch.gather(Lr, 1, gi)           # log尤度
            lk = torch.exp(ll_part).clamp_min(1e-300)
        else:                                           # 素: GR距離尤度
            ii = (v - vmin[:, None]) / step[:, None]
            i0 = torch.floor(ii).long()
            below = i0 < 0; above = i0 >= (glen - 1)[:, None]
            i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
            t = ii - i0c.to(dtype)
            gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
            eg = gA * (1 - t) + gB * t
            eg = torch.where(below, g0col, eg); eg = torch.where(above, glast, eg)
            d = ((gri[:, None] - eg) / gs[:, None]).abs()
            dp = d ** GR_POWER
            lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)

        avg_lk = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg_lk.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG
            lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp, w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1)
        need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1)
            u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N
            idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn()
            rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]
            pos = torch.where(nm, pos_rs, pos)
            rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pts[:, i] = (w * (pos - zi)).sum(1)
        pm = torch.where(act, cur, pm)
    return pts, log_lik


def _pad(inps, device, lmaps=None, drifts=None, dtype=DTYPE):
    """坑井リストをパディング。lmaps(list of (T,Gl)) / drifts(list of (T,)) も対応。"""
    W = len(inps)
    Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
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
    pad = dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
               vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), Tmax=Tmax)
    if lmaps is not None:
        Glm = max(x["lglen"] for x in inps)
        lmap = torch.zeros((W, Tmax, Glm), dtype=dtype)
        lvmin = torch.zeros(W, dtype=dtype); lglen = torch.zeros(W, dtype=torch.long)
        for b, x in enumerate(inps):
            T = len(x["md"]); Gl = x["lglen"]
            L = torch.as_tensor(np.asarray(lmaps[b], np.float32))           # (T, Gl)
            lmap[b, :T, :Gl] = L
            lvmin[b] = x["lvmin"]; lglen[b] = Gl
        pad.update(lmap=to(lmap), lmap_vmin=to(lvmin), lmap_step=float(LMAP_STEP), lmap_glen=to(lglen))
    if drifts is not None:
        drift = torch.zeros((W, Tmax), dtype=dtype)
        for b, x in enumerate(inps):
            T = len(x["md"]); drift[b, :T] = torch.as_tensor(np.asarray(drifts[b], np.float32))[:T]
        pad.update(drift=to(drift))
    return pad


def run_forward(inps, p, S, device, seed, chunk, lmaps=None, drifts=None, lam_drift=0.0):
    """各坑井: Sシードの尤度加重アンサンブル pf_ancc(T,) を返す。lmaps/drifts で案A/B。"""
    N = int(p["ANCC_N"]); W = len(inps); out = [None] * W
    for c in range(0, W, chunk):
        sub = inps[c:c + chunk]; w = len(sub)
        lm = lmaps[c:c + chunk] if lmaps is not None else None
        dr = drifts[c:c + chunk] if drifts is not None else None
        pad = _pad(sub, device, lmaps=lm, drifts=dr)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        widx = torch.arange(w, device=device).repeat_interleave(S)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        kw = {}
        if lm is not None:
            kw.update(lmap=pad["lmap"], lmap_vmin=rep(pad["lmap_vmin"]), lmap_step=pad["lmap_step"],
                      lmap_glen=rep(pad["lmap_glen"]), well_index=widx)
        if dr is not None:
            kw.update(drift=pad["drift"], well_index=widx, lam_drift=lam_drift)
        pts, log_lik = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), rep(pad["vmin"]), rep(pad["step"]),
            rep(pad["gs"]), rep(pad["ls"]), rep(pad["ir"]), p, N, device, gen, **kw)
        pts = pts.view(w, S, -1).double().cpu().numpy()
        ll = log_lik.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]
            lk = ll[j] - np.nanmax(ll[j]) if np.isfinite(ll[j]).any() else np.zeros_like(ll[j])
            ww = np.exp(np.where(np.isfinite(lk), lk, -np.inf) / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6))
            ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(S, 1.0 / S)
            out[c + j] = (ww[:, None] * pr).sum(0).astype(np.float32)
    return out


def run_forward_scales(inps, p, S, device, seed, chunk, scales=(3.0, 5.0, 8.0, 12.0)):
    """各坑井: 1回のGPU走行で per-seed 軌跡 + log_lik を取り、スケールごとに seed 再加重して
    {f'pf_scale_{X:g}': pred} と既定 pf_ancc(PFのLIKELIHOOD_SCALE加重)・std を返す。
    = 7.191 の run_pf_lik_ensemble_scales / run_pf_ancc を GPU 一発で代替するための出力。"""
    N = int(p["ANCC_N"]); W = len(inps); out = [None] * W

    def _wsum(base, pr, sc):
        ww = np.exp(np.where(np.isfinite(base), base, -np.inf) / max(float(sc), 1e-6))
        ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(len(base), 1.0 / len(base))
        return ww, (ww[:, None] * pr).sum(0).astype(np.float32)

    for c in range(0, W, chunk):
        sub = inps[c:c + chunk]; w = len(sub)
        pad = _pad(sub, device)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), rep(pad["vmin"]), rep(pad["step"]),
            rep(pad["gs"]), rep(pad["ls"]), rep(pad["ir"]), p, N, device, gen)
        pts = pts.view(w, S, -1).double().cpu().numpy()
        ll = log_lik.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]
            base = ll[j] - np.nanmax(ll[j]) if np.isfinite(ll[j]).any() else np.zeros_like(ll[j])
            d = {}
            for sc in scales:
                d[f"pf_scale_{sc:g}"] = _wsum(base, pr, sc)[1]
            ww0, mean = _wsum(base, pr, float(p["PF_LIKELIHOOD_SCALE"]))   # 既定スケール=pf_ancc
            var = (ww0[:, None] * (pr - mean[None, :]) ** 2).sum(0)
            d["pf_ancc"] = mean; d["std"] = np.sqrt(np.maximum(var, 0.)).astype(np.float32)
            d["pf_mean"] = np.mean([d[f"pf_scale_{sc:g}"] for sc in scales], 0).astype(np.float32)
            out[c + j] = d
    return out


def eval_wells(wells, ns, p, split="train", chunk=4, seed=SEED,
               lmap_fn=None, drift_fn=None, lam_drift=0.0, calibrate=False,
               repr_="raw", repr_win=3, verbose=True):
    """指定坑井で PF を回し overall(全行プール)RMSE と per-well RMSE を返す。
    lmap_fn(wid, inp)->(T,Gl) で案A、drift_fn(wid, inp)->(T,) で案B、calibrate=True でヒール較正、
    repr_=GR表現(raw/ma/agc/rank/deriv/quantile)。真値TVTのある坑井のみ評価。"""
    inps, names = [], []
    for wid in wells:
        x = build_inputs(wid, ns, p, split=split, calibrate=calibrate, repr_=repr_, repr_win=repr_win)
        if x is None or x["tvt_true"] is None:
            continue
        inps.append(x); names.append(wid)
    if not inps:
        raise RuntimeError("有効坑井なし")
    lmaps = [np.asarray(lmap_fn(x["wid"], x), np.float32) for x in inps] if lmap_fn else None
    drifts = [np.asarray(drift_fn(x["wid"], x), np.float32) for x in inps] if drift_fn else None
    t0 = time.perf_counter()
    preds = run_forward(inps, p, int(p["PF_N_SEEDS"]), DEVICE, seed, chunk,
                        lmaps=lmaps, drifts=drifts, lam_drift=lam_drift)
    dt = time.perf_counter() - t0
    perwell = {}; resid_all = []
    for x, pr in zip(inps, preds):
        e = pr - x["tvt_true"]
        e = e[np.isfinite(e)]
        perwell[x["wid"]] = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
        resid_all.append(e)
    rall = np.concatenate(resid_all)
    overall = float(np.sqrt(np.mean(rall ** 2)))
    if verbose:
        tag = "A:lmap" if lmap_fn else ""
        tag += (" B:drift(λ=%.2f)" % lam_drift) if drift_fn else ""
        print(f"[eval] wells={len(names)} overall={overall:.3f}  ({dt:.1f}s, {DEVICE}) {tag}")
    return dict(overall=overall, perwell=perwell, names=names)


HARD_WELLS = ["b8c49c1a","113011eb","91db7070","a2e8e7f6","1b82b665","73348914","86454a6f",
              "e7818f7a","91b301ce","efe96181","70925e23","173ed841","abf460d2","1d3fbf02",
              "4ed93db6","ab112491","4caa7289","55efee7f","708caea9","43b06c70"]
