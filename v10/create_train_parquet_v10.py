"""
v10 train.parquet ビルダー  ―― 逆方向パーティクルフィルタ特徴の実験
==================================================================

アイデア:
  順方向PF(ancc)で得た各 TVT 軌跡の「終端」から、逆方向(終端→手前)に向けて
  もう一度PFを走らせ、その逆方向 TVT 軌跡を特徴量に加える。順方向のみで学習した
  モデルと、逆方向を加えたモデルの精度を比較する実験用。

設計(ユーザー指示を反映):
  - PF以外の特徴量は v9 をそのまま継承(v9 build_well を exec で再利用)。
  - 順方向PF特徴・逆方向PF特徴はどちらも GPU(PyTorch)で生成(シード変動はOKの粒度)。
      → 順方向GPU PF(8シード)のアンサンブルを v9 build_well に「注入」し、
        v9 の順方向PF特徴定義(_add_pf_feature_columns)で特徴化(CPUの順方向PFは使わない)。
      → 各シードの終端から4本ずつ逆方向PF(計 8x4=32 本)→ 逆方向アンサンブル bpf_ancc。
      （順方向8 + 逆方向32 = 40 本の軌跡)
      → pf_z(z粒子フィルタ)は v9 のまま CPU(順方向PF特徴の一部)。
      → 簡易実験のため pf_1 のみ・順方向 n_seeds=8。
  - 列の命名:
      順方向(v9)特徴: v9 のまま(pf_ancc, pf_*, 非PF特徴, target ...)
      逆方向特徴      : 接頭辞 bpf_   (例 bpf_ancc, bpf_gr_abs_min, bpf_rate ...)
      比較特徴        : 接頭辞 cmp_   (順方向 pf_ancc vs 逆方向 bpf_ancc)
  - 3実験は列の接頭辞で選べる:
      (1) 順方向のみ(=v9): bpf_ / cmp_ を除外
      (2) 順方向+逆方向    : cmp_ を除外
      (3) 順方向+逆方向+比較: 全列
  - 出力: artifacts_v10/data/train.parquet (v9 と同じ構造)。well 列を付与。

逆方向PFの物理(モデリング上の選択。必要なら調整可):
  - 評価ゾーンを終端→手前に辿る。MDを反転 (md_bwd = -md[::-1]) して間隔を保持。
  - 初期位置 = 順方向終端の pos(=TVT+Z)、初期レート = -(順方向終端 rate)(向き反転)、
    そこに初期分散を加えて 4 本に分岐。

使い方:
  rogii_claude/.venv/Scripts/python.exe v10/create_train_parquet_v10.py
"""

import os, sys, time, glob, json, types, threading
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 順方向GPUアンサンブルを v9 build_well に注入するための受け渡し。
# build_well の run_pf_ancc(CPU)を、事前計算した GPU 結果を返す関数に差し替える。
_TLS = threading.local()        # スレッドごとに現在処理中の well id を保持
_FWD_ENS = {}                   # (wid, pf_name) -> (pf_ancc, pf_ancc_std)  GPU順方向アンサンブル

def _patched_run_pf_ancc(hw, tw_tvt, tw_gr, p):
    """v9 の run_pf_ancc を差し替えるGPU版。事前計算した順方向アンサンブルを (well, PFセット) 別に返す。"""
    r = _FWD_ENS.get((getattr(_TLS, "wid", None), p.get("name")))
    if r is None:
        return np.array([], np.float32), np.array([], np.float32)
    return r[0].astype(np.float32), r[1].astype(np.float32)

# ============================== CONFIG ==============================
PROJECT_DIR = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR    = PROJECT_DIR / "rogii-wellbore-geology-prediction"          # train/ test/ がある
V9_NB       = PROJECT_DIR / "v9" / "pf_tuning_create_train_csv_multi_pf_v9.ipynb"
OUT_PARQUET = PROJECT_DIR / "v10" / "artifacts_v10_3" / "data" / "train.parquet"

N_WELLS     = None       # None=全坑井, または整数で先頭N(動作確認用に小さく)
N_PF_SETS   = 3          # 使用するPFパラメータセット数(pf_1/2/3)
FWD_SEEDS   = 128          # 順方向シード数(各PFセット)。順方向のみ実験は 128 など
BWD_PER_SEED= 8          # 各順方向終端から伸ばす逆方向の本数(各PFセット)
BUILD_BACKWARD = False    # False=順方向のみ(逆方向bpf_/比較cmp_を作らない)。順方向のみ版はこれをFalseに
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float32
SEED        = 0
WELL_CHUNK  = 130         # GPUバッチに載せる坑井数(逆方向は x32 になるので控えめ)
N_JOBS_CPU  = 4          # v9 build_well(CPU特徴)の並列スレッド数
# ===================================================================


# ======================================================================
# 1) v9 ノートブックを exec で読み込み、v9 の関数群(build_well 等)を得る
# ======================================================================
def load_v9_namespace():
    nb = json.load(open(V9_NB, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    # CFG のデータパスをこのプロジェクトに差し替え(FI/DI の fit がここを読む)
    src = src.replace(
        r'dataset_path = Path(r"C:\Users\kosaka256\Documents\rogii\rogii-wellbore-geology-prediction")',
        f'dataset_path = Path(r"{DATA_DIR}")',
    )
    # numba の @njit(cache=True) は exec 環境だと過去キャッシュのロードに失敗する
    # (ModuleNotFoundError: '<dynamic>')。キャッシュを無効化してその場で再コンパイルさせる。
    src = src.replace("cache=True", "cache=False")
    # 末尾の実行セル(train_paths=... / build_train_csv(...) / display)を切り落とす
    marker = 'train_paths = sorted((CFG.dataset_path / "train").glob'
    cut = src.find(marker)
    if cut > 0:
        src = src[:cut]
    ns = {"__name__": "v9lib"}
    print("[v9] exec で v9 定義を読み込み中(FI/DI の fit を含むため少し時間がかかります)...")
    exec(compile(src, str(V9_NB), "exec"), ns)
    # 3種類のPF(pf_1/2/3)・順方向 n_seeds=FWD_SEEDS に設定(build_well は global PF_PARAM_SETS を参照)
    raws = ns["PF_PARAM_SETS_RAW"][:N_PF_SETS]
    sets = []
    for k, raw in enumerate(raws):
        r = dict(raw); r["n_seeds"] = FWD_SEEDS
        sets.append(ns["_make_pf_params"](r, k))
    ns["PF_PARAM_SETS"] = sets
    print(f"[v9] PF_PARAM_SETS -> {[p['name'] for p in sets]}, "
          f"suffix={[p['suffix'] for p in sets]}, n_seeds={sets[0]['PF_N_SEEDS']}, N={sets[0]['ANCC_N']}")
    return ns


# ======================================================================
# 2) GPU 用 PF 入力作成(v9 run_pf_ancc のシードループ手前を再現)
#    ※ v9 名前空間のヘルパを使い、順方向GPU/逆方向GPU の入力を揃える
# ======================================================================
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
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    md_ev = ev["MD"].values.astype(np.float64); z_ev = ev["Z"].values.astype(np.float64)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc = _smooth(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos = hw.index.get_indexer(ev.index)
    gr_ev = gr_proc[ev_pos].astype(np.float64)
    return dict(md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(p["ANCC_N"]))


# ======================================================================
# 3) GPU PF コア(坑井×シードをバッチ化)。順方向・逆方向で共用。
#    終端状態(pos,rate)も返す(逆方向の初期化に使う)。
# ======================================================================
def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    """各行 row = 坑井×シード。md,z,gr,valid:(B,T)  grid:(B,G)  他:(B,)
       返り値: pts(B,T) 重み付き平均TVT, log_lik(B,), pos_fin(B,), rate_fin(B,)"""
    B, T = md.shape
    ALPHA=p["ANCC_ALPHA"]; RN=p["ANCC_RN"]; PN=p["ANCC_PN"]; IR=p["ANCC_IR"]; IS=p["ANCC_IS"]
    RP=p["ANCC_RP"]; RR=p["ANCC_RR"]; RESAMP=p["PF_RESAMP"]; RATE_CLIP=p["PF_RATE_CLIP"]
    GR_POWER=p["PF_GR_POWER"]; JUMP_PROB=p["PF_JUMP_PROB"]; JUMP_STD=p["PF_JUMP_STD"]; CLIP=p["PF_TVT_CLIP_MARGIN"]

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

    pos = ls[:, None] + IS*rn()
    rate = ir[:, None] + IR*rn()
    w = torch.full((B, N), 1.0/N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)

    tvt_lo = vmin - CLIP
    tvt_hi = vmin + (glen.to(dtype)-1)*step + CLIP
    glast = torch.gather(grid, 1, (glen-1).clamp_min(0).unsqueeze(1))
    g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0
    arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)

    for i in range(T):
        act = valid[:, i]
        cur = md[:, i]
        dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA*rate + RN*rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n*dm[:, None] + PN*rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD*rn(), zero)
        zi = z[:, i][:, None]
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None])
        pos_n = tvt + zi
        am = act[:, None]
        pos = torch.where(am, pos_n, pos)
        rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]
        obs = act & ~torch.isnan(gri)
        v = pos - zi
        ii = (v - vmin[:, None]) / step[:, None]
        i0 = torch.floor(ii).long()
        below = i0 < 0
        above = i0 >= (glen-1)[:, None]
        i0c = i0.clamp_min(0)
        i0c = torch.minimum(i0c, (glen-2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype)
        gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c+1)
        eg = gA*(1-t) + gB*t
        eg = torch.where(below, g0col, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs()
        dp = d**GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5*dp), zero).clamp_min(1e-300)
        avg_lk = (w*lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg_lk.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        w_new = w*lk
        ws = w_new.sum(1, keepdim=True)
        w_upd = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0/N))
        w = torch.where(obs[:, None], w_upd, w)
        ne = (w*w).sum(1)
        need = act & ((1.0/ne) < (RESAMP*N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1)
            u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0/N)
            u = u0 + arN[None, :]/N
            idx = torch.searchsorted(cumw, u, right=False).clamp(0, N-1)
            pos_rs = torch.gather(pos, 1, idx) + RP*rn()
            rate_rs = torch.gather(rate, 1, idx) + RR*rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]
            pos = torch.where(nm, pos_rs, pos)
            rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0/N), w)
        tvt_out = pos - zi
        pts[:, i] = (w*tvt_out).sum(1)
        pm = torch.where(act, cur, pm)
    pos_fin = (w*pos).sum(1)           # 終端の重み付き平均 pos(=TVT+Z)
    rate_fin = (w*rate).sum(1)         # 終端の重み付き平均 rate
    return pts, log_lik, pos_fin, rate_fin


def _pad_wells(inps, device, reverse=False, dtype=DTYPE):
    """坑井ごとの入力を (W,Tmax)/(W,Gmax) にパディング。reverse=True で評価ゾーンを反転。"""
    W = len(inps)
    Tmax = max(len(x["md"]) for x in inps)
    Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["md"]); G = len(x["gg"])
        mdv = x["md"].astype("float32"); zv = x["z"].astype("float32"); grv = x["gr"].astype("float32")
        if reverse:
            mdv = -mdv[::-1].copy(); zv = zv[::-1].copy(); grv = grv[::-1].copy()
        md[b, :T] = torch.from_numpy(mdv)
        if T < Tmax: md[b, T:] = md[b, T-1]
        z[b, :T] = torch.from_numpy(zv)
        gr[b, :T] = torch.from_numpy(grv); valid[b, :T] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G-1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), Tmax=Tmax)


def run_forward(inps, p, S, device, seed, chunk):
    """順方向GPU PF。返り値:
       ens_list[i] = (pf_ancc, pf_ancc_std)  … Sシードの尤度加重アンサンブル(評価行順)
       term_pos, term_rate (W,S)             … 各シードの終端状態(逆方向の初期化用)"""
    N = int(p["ANCC_N"]); W = len(inps)
    ens_list = [None]*W
    term_pos = np.zeros((W, S), np.float32); term_rate = np.zeros((W, S), np.float32)
    for c in range(0, W, chunk):
        sub = inps[c:c+chunk]; w = len(sub)
        pad = _pad_wells(sub, device, reverse=False)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik, pos_fin, rate_fin = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            p, N, device, gen)
        term_pos[c:c+w] = pos_fin.view(w, S).float().cpu().numpy()
        term_rate[c:c+w] = rate_fin.view(w, S).float().cpu().numpy()
        pts = pts.view(w, S, -1).double().cpu().numpy()
        ll = log_lik.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]
            lk = ll[j] - np.nanmax(ll[j])
            ww = np.exp(lk / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6)); ww /= max(ww.sum(), 1e-300)
            mean = (ww[:, None]*pr).sum(0).astype(np.float32)
            var = (ww[:, None]*(pr - mean[None, :])**2).sum(0)
            std = np.sqrt(np.maximum(var, 0.)).astype(np.float32)
            ens_list[c+j] = (mean, std)
    return ens_list, term_pos, term_rate


def run_backward(inps, term_pos, term_rate, p, Sfwd, Sbwd, device, seed, chunk):
    """逆方向GPU PF。順方向終端から各 Sbwd 本ずつ。坑井ごとに 32本アンサンブルした
       bpf_ancc(順方向行順) と bpf_std を返す(リスト)。"""
    N = int(p["ANCC_N"]); W = len(inps); M = Sfwd * Sbwd
    outs = [None]*W
    for c in range(0, W, chunk):
        sub = inps[c:c+chunk]; w = len(sub)
        pad = _pad_wells(sub, device, reverse=True)   # 逆順(終端→手前)
        rep = lambda tt: tt.repeat_interleave(M, dim=0)
        # 各行の初期 = 対応する順方向終端。行順 = 坑井 -> 順方向シード -> 逆方向枝
        tp = torch.from_numpy(term_pos[c:c+w].reshape(-1)).to(device, DTYPE).repeat_interleave(Sbwd)   # (w*M,)
        tr = torch.from_numpy(term_rate[c:c+w].reshape(-1)).to(device, DTYPE).repeat_interleave(Sbwd)
        ls_rows = tp                 # 初期位置 = 終端 pos
        ir_rows = -tr                # 初期レート = -終端レート(向き反転)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik, _, _ = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(M), pad["vmin"].repeat_interleave(M), pad["step"].repeat_interleave(M),
            pad["gs"].repeat_interleave(M), ls_rows, ir_rows,
            p, N, device, gen)
        pts = pts.view(w, M, -1).double().cpu().numpy()      # 逆順の TVT 軌跡
        ll = log_lik.view(w, M).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"])
            pr_rev = pts[j, :, :T]                            # (M,T) 逆順
            pr = pr_rev[:, ::-1]                              # 順方向行順に戻す
            lk = ll[j] - np.nanmax(ll[j])
            ww = np.exp(lk / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6)); ww /= max(ww.sum(), 1e-300)
            mean = (ww[:, None]*pr).sum(0).astype(np.float32)
            var = (ww[:, None]*(pr - mean[None, :])**2).sum(0)
            std = np.sqrt(np.maximum(var, 0.)).astype(np.float32)
            outs[c+j] = (mean, std)
    return outs


# ======================================================================
# 4) 逆方向特徴(bpf_*)・比較特徴(cmp_*)を v9 のヘルパ定義で構築
# ======================================================================
def _add_one_backward_set(out, suffix, bpf_ancc, bpf_std, hgr, tw_tvt, tw_gr, hmd, last_tvt, V9):
    """1つのPFセットの逆方向特徴(bpf{suffix})+比較特徴(cmp{suffix})を out に追加。
       suffix: "" / "_1" / "_2" / "_3"。比較は順方向 pf_ancc{suffix} 列が out にある場合のみ。"""
    PF_OFFS = V9["PF_OFFS"]
    _add_gr_match = V9["_add_gr_match_features"]; _add_shape = V9["_add_path_shape_features"]
    _safe_grad = V9["_safe_grad"]; _rmean = V9["_rolling_mean_np"]
    n = len(out)
    bpf_ancc = np.asarray(bpf_ancc, np.float32)[:n]
    bpf_std  = np.asarray(bpf_std,  np.float32)[:n]
    hgr = np.asarray(hgr, np.float32)[:n]

    prefix = "bpf" if suffix == "" else f"bpf{suffix}"
    bf = {}
    bf[f"bpf_ancc{suffix}"] = bpf_ancc
    bf[f"bpf_ancc_std{suffix}"] = bpf_std
    bf[f"bpf_ancc_delta{suffix}"] = (bpf_ancc - np.float32(last_tvt)).astype(np.float32)
    for o in PF_OFFS:
        bf[f"bpf_tdpf{int(o)}{suffix}"] = (hgr - np.interp(bpf_ancc + o, tw_tvt, tw_gr)).astype(np.float32)
    _add_gr_match(bf, prefix, bpf_ancc, hgr, tw_tvt, tw_gr, PF_OFFS)
    _add_shape(bf, prefix, bpf_ancc, hmd)

    cmp = {}
    fwd_col = f"pf_ancc{suffix}"
    if fwd_col in out.columns:
        pf_ancc = out[fwd_col].to_numpy(np.float32)
        diff = (pf_ancc - bpf_ancc).astype(np.float32)
        cmp[f"cmp_fwd_bwd_diff{suffix}"] = diff
        cmp[f"cmp_fwd_bwd_absdiff{suffix}"] = np.abs(diff).astype(np.float32)
        cmp[f"cmp_fwd_bwd_mean{suffix}"] = ((pf_ancc + bpf_ancc) * 0.5).astype(np.float32)
        cmp[f"cmp_fwd_bwd_rate_diff{suffix}"] = (_safe_grad(pf_ancc, hmd) - _safe_grad(bpf_ancc, hmd)).astype(np.float32)
        cmp[f"cmp_fwd_bwd_absdiff_rmean21{suffix}"] = _rmean(np.abs(diff), 21)
        fwd_gr = f"{('pf' if suffix=='' else 'pf'+suffix)}_gr_abs_min"
        if fwd_gr in out.columns and f"{prefix}_gr_abs_min" in bf:
            cmp[f"cmp_fwd_bwd_grabsmin_diff{suffix}"] = (out[fwd_gr].to_numpy(np.float32) - bf[f"{prefix}_gr_abs_min"]).astype(np.float32)
    fwd_std = f"pf_ancc_std{suffix}"
    if fwd_std in out.columns:
        cmp[f"cmp_fwd_bwd_std_diff{suffix}"] = (out[fwd_std].to_numpy(np.float32) - bpf_std).astype(np.float32)

    for k, v in bf.items():
        out[k] = v
    for k, v in cmp.items():
        out[k] = v
    return out


def add_backward_and_compare_features(df, hw, tw, bwd_by_suffix, V9):
    """df: v9 build_well 出力(評価ゾーン行順、順方向 pf_ancc{suffix} 列を含む)。
       bwd_by_suffix: [(suffix, bpf_ancc, bpf_std), ...]  各PFセット(+set1エイリアス)。"""
    tw = tw.sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(np.float32); tw_gr = tw["GR"].to_numpy(np.float32)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    last_tvt = float(kn["TVT_input"].iloc[-1])
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)      # v9 と同じ定義
    hmd = ev["MD"].to_numpy(np.float32)
    out = df.copy()
    for suffix, bpf_ancc, bpf_std in bwd_by_suffix:
        out = _add_one_backward_set(out, suffix, bpf_ancc, bpf_std, hgr, tw_tvt, tw_gr, hmd, last_tvt, V9)
    return out


# ======================================================================
# 5) v10 データセット構築(train/test 両用)。提出ノートはこの関数を test で呼ぶ。
# ======================================================================
def build_v10_dataset(V9, split="train", is_train=True, n_wells=None):
    """DATA_DIR/<split> の各坑井について v10 特徴量(非PF=v9 / 順方向PF=GPU注入 / 逆方向bpf / 比較cmp)を作り、
       1つの DataFrame(well, id, [target], 特徴量...) を返す。
       - FI/DI は load_v9_namespace 時に DATA_DIR/train で fit 済み(test でも train近傍から補完=リーク無し)。
       - is_train=False(test)では build_well が target を付けない。"""
    build_well = V9["build_well"]; P_SETS = V9["PF_PARAM_SETS"]
    KEEP_ALIAS = bool(V9.get("KEEP_FIRST_PF_ALIAS", True))
    base = DATA_DIR / split
    wells = sorted({os.path.basename(q).split("__")[0]
                    for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if n_wells is not None:
        wells = wells[:n_wells]

    print(f"[{split}] 有効坑井を確定中 ({len(wells)} 坑井)...")
    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if is_train and "TVT" not in hw.columns:
            continue
        if build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P_SETS[0], V9) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[{split}] 使用坑井: {len(names)}")

    # 各PFセットごとに GPU 順方向(注入用ens+終端)→(BUILD_BACKWARD時のみ)逆方向(bpf)
    _FWD_ENS.clear()
    bwd_by_set = {}
    for p in P_SETS:
        mode = "順方向+逆方向" if BUILD_BACKWARD else "順方向のみ"
        print(f"[gpu] {p['name']}: 入力作成 + PF ({mode}) ...")
        t0 = time.perf_counter()
        inps = []
        for (hp, tp) in paths:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            inps.append(build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), p, V9))
        fwd_ens, term_pos, term_rate = run_forward(inps, p, FWD_SEEDS, DEVICE, SEED, WELL_CHUNK)
        for i, w in enumerate(names):
            _FWD_ENS[(w, p["name"])] = fwd_ens[i]
        if BUILD_BACKWARD:
            bwd_by_set[p["name"]] = run_backward(inps, term_pos, term_rate, p, FWD_SEEDS, BWD_PER_SEED, DEVICE, SEED, WELL_CHUNK)
        if DEVICE == "cuda": torch.cuda.synchronize()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    V9["run_pf_ancc"] = _patched_run_pf_ancc   # build_well 内の run_pf_ancc を GPU結果に差し替え

    print(f"[cpu] build_well(非PF=v9 / 順方向PF=GPU注入){' + 逆方向/比較特徴' if BUILD_BACKWARD else ' (順方向のみ)'} を結合...")
    from joblib import Parallel, delayed
    def one(w, hp, tp, idx):
        _TLS.wid = w
        df = build_well(str(hp), str(tp), is_train)
        if df is None or len(df) == 0:
            return None
        if not BUILD_BACKWARD:
            return df                      # 順方向のみ(v9特徴=GPU注入)
        bwd_by_suffix = []
        for si, p in enumerate(P_SETS):
            bwd = bwd_by_set[p["name"]][idx]
            if bwd is None:
                return None
            bpf_ancc, bpf_std = bwd
            bwd_by_suffix.append((p["suffix"], bpf_ancc, bpf_std))
            if KEEP_ALIAS and si == 0:
                bwd_by_suffix.append(("", bpf_ancc, bpf_std))
        m = min([len(df)] + [len(b[1]) for b in bwd_by_suffix])
        if m != len(df):
            df = df.iloc[:m].reset_index(drop=True)
            bwd_by_suffix = [(s, a[:m], st[:m]) for (s, a, st) in bwd_by_suffix]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp)
        return add_backward_and_compare_features(df, hw, tw, bwd_by_suffix, V9)

    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS_CPU, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1], i) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s,  parts={len(parts)}")
    if not parts:
        raise RuntimeError(f"有効な坑井がありません ({split})。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"device={DEVICE}  pf_sets={N_PF_SETS}  fwd_seeds={FWD_SEEDS}  bwd_per_seed={BWD_PER_SEED} "
          f"(=> backward {FWD_SEEDS*BWD_PER_SEED} traj/セット)  chunk={WELL_CHUNK}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    V9 = load_v9_namespace()
    train = build_v10_dataset(V9, split="train", is_train=True, n_wells=N_WELLS)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_PARQUET, index=False)
    cols = list(train.columns)
    n_bpf = sum(c.startswith("bpf_") for c in cols); n_cmp = sum(c.startswith("cmp_") for c in cols)
    print(f"\nsaved: {OUT_PARQUET}")
    print(f"rows={len(train)}  cols={len(cols)}  (bpf_={n_bpf}, cmp_={n_cmp}, それ以外=v9順方向)")


if __name__ == "__main__":
    main()
