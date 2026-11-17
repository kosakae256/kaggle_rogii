"""
v23 グリッドフィルタ(HMM前向き・決定論)チューナー ―― 6カテゴリ別・再開可
==============================================================================
PF(粒子フィルタ)の代替: TVTを格子(G点=typewellグリッド)に離散化した **厳密ベイズ前向きフィルタ**。
  - 状態 = TVT(地質位置)。emission = GR尤度(観測GR vs typewell GR@TVT)。
  - transition = dipドリフト(known dip × scale)＋ガウス過程ノイズ(＋断層用に一様混合)。
  - 各MDで belief(W,G) を「シフト＋ぼかし → 尤度の要素積 → 正規化」。**リサンプリング無し・粒子退化無し・seed不要(決定論)**。
  - 読み出し = 事後の(温度付き)平均 = posterior mean(双峰tailの hedge)。

利点: 全坑井を一括バッチ(W,G)で前向き、シード不要なので **目的関数にノイズが乗らず Optuna が安定**。
チューニング: selector の 6カテゴリ(code 0..5)ごとに独立に Optuna → 6つのパラメータ。残差ブースティング無し。再開可。

出力: v23/artifacts_v23/per_category_grid.json
実行: rogii_claude/.venv/Scripts/python.exe v23/tune_grid_filter_v23.py
※ build_cache/well_code/カテゴリ定義は v22 と同一。エンジンだけ grid filter に置換。
"""

import os, time, json, glob, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ============================== CONFIG ==============================
PROJECT   = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v23" / "artifacts_v23"; OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32

GRID_STEP   = 1.0        # TVT格子の刻み(ft)。粗いほど速い/メモリ小。0.5で精細。
GRID_MARGIN = 60.0       # typewell tvt 範囲の外側マージン(ft)
WELL_CHUNK  = 256        # grid_forward の同時バッチ坑井数(OOMなら下げる)
N_WELLS     = None       # None=全well, 整数で先頭N(デバッグ)
N_TRIALS    = 250        # 1カテゴリあたりの Optuna 探索回数

N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
N_CATS = 6

STORAGE      = f"sqlite:///{(OUT_DIR / 'optuna.db').as_posix()}"
STATE_PATH   = OUT_DIR / "state.json"
PER_CAT_PATH = OUT_DIR / "per_category_grid.json"
CUR_PATH     = OUT_DIR / "current_cat.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------- データ(v22と同一 + last_tvt) -----------------------
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
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    z_ev = ev["Z"].values.astype(np.float64)
    n_eval = int(len(z_ev))
    z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    # dip_prior(known末尾30の (TVT+Z)/MD 傾き)と last_tvt
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    last_tvt = float(kn["TVT_input"].iloc[-1])
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev, gr_filled=gr_filled,
                ev_pos=ev_pos, kpos=kpos, kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ir=ir, last_tvt=last_tvt,
                tvt_true=tv, n_eval=n_eval, z_span=z_span)

def well_code(c):
    n_bin = int(c["n_eval"] > N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(Z_SPAN_THRESHOLDS, c["z_span"], side="right"))
    return n_bin + 2 * z_bin

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)


def make_grid_inputs(c, P):
    """1坑井のグリッド入力。grid_tvt(格子=TVT), gg(typewell GR@格子), gr(eval処理GR), z, dMD, dZ, gs, ir, last_tvt。"""
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    lo = float(c["tw_tvt"].min() - GRID_MARGIN); hi = float(c["tw_tvt"].max() + GRID_MARGIN)
    grid_tvt = np.arange(lo, hi + GRID_STEP, GRID_STEP)
    gg = np.interp(grid_tvt, c["tw_tvt"], tw_gr_pf).astype(np.float64)   # 範囲外は端値(np.interpが端clamp)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    md = c["md"]; z = c["z"]
    dMD = np.diff(md, prepend=md[0]); dZ = np.diff(z, prepend=z[0])   # [0]は0(初期は遷移なし)
    return dict(grid_tvt=grid_tvt.astype(np.float64), gg=gg, gr=gr_sm[c["ev_pos"]].astype(np.float64),
                z=z, dMD=dMD.astype(np.float64), dZ=dZ.astype(np.float64),
                gs=gs, ir=c["ir"], last_tvt=c["last_tvt"])


# ----------------------- グリッド前向きフィルタ(GPU, W×G バッチ) -----------------------
def _pad_grid(inps, device, dtype=DTYPE):
    W = len(inps); Tmax = max(len(x["gr"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    gtvt = torch.zeros((W, Gmax), dtype=dtype); gg = torch.zeros((W, Gmax), dtype=dtype)
    glen = torch.zeros(W, dtype=torch.long)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    dMD = torch.zeros((W, Tmax), dtype=dtype); dZ = torch.zeros((W, Tmax), dtype=dtype)
    gs = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype); last_tvt = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["gr"]); G = len(x["gg"])
        gtvt[b, :G] = torch.from_numpy(x["grid_tvt"].astype("float32"))
        if G < Gmax: gtvt[b, G:] = gtvt[b, G - 1]
        gg[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: gg[b, G:] = gg[b, G - 1]
        glen[b] = G
        gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        dMD[b, :T] = torch.from_numpy(x["dMD"].astype("float32")); dZ[b, :T] = torch.from_numpy(x["dZ"].astype("float32"))
        gs[b] = x["gs"]; ir[b] = x["ir"]; last_tvt[b] = x["last_tvt"]
    to = lambda t: t.to(device)
    return dict(gtvt=to(gtvt), gg=to(gg), glen=to(glen), gr=to(gr), valid=to(valid),
                dMD=to(dMD), dZ=to(dZ), gs=to(gs), ir=to(ir), last_tvt=to(last_tvt), Tmax=Tmax, Gmax=Gmax)

def _gauss_kernel(sigma_cells, device, dtype=DTYPE):
    sig = max(float(sigma_cells), 1e-3); half = max(1, int(math.ceil(3 * sig)))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sig) ** 2); k = k / k.sum()
    return k.view(1, 1, -1), half

def grid_forward(inps, P, device):
    """各坑井の eval TVT 予測(W,T)を返す。前向きグリッドフィルタ(決定論)。"""
    pad = _pad_grid(inps, device); W = len(inps); Gmax = pad["Gmax"]; Tmax = pad["Tmax"]
    step = GRID_STEP
    PROC = float(P["proc_sigma"]); DSCALE = float(P["dip_drift_scale"]); INIT = float(P["init_sigma"])
    TEMP = float(P["post_temp"]); POWER = float(P["gr_power"]); JP = float(P["jump_prob"])
    gtvt = pad["gtvt"]; gg = pad["gg"]; gs = pad["gs"][:, None]; glen = pad["glen"]
    arG = torch.arange(Gmax, device=device, dtype=DTYPE)[None, :]            # (1,G)
    valid_g = (arG < glen[:, None].to(DTYPE))                                # (W,G) 有効格子
    # 初期事後: last_tvt 周りのガウス
    b = torch.exp(-0.5 * ((gtvt - pad["last_tvt"][:, None]) / max(INIT, 1e-3)) ** 2)
    b = torch.where(valid_g, b, torch.zeros_like(b)); b = b / b.sum(1, keepdim=True).clamp_min(1e-30)
    kernel, _ = _gauss_kernel(PROC / step, device)
    unif = (valid_g.to(DTYPE)); unif = unif / unif.sum(1, keepdim=True).clamp_min(1e-30)
    pred = torch.zeros((W, Tmax), device=device, dtype=DTYPE)
    for i in range(Tmax):
        act = pad["valid"][:, i]
        if i > 0:
            drift_cells = ((pad["ir"] * DSCALE) * pad["dMD"][:, i] - pad["dZ"][:, i]) / step   # (W,)
            src = arG - drift_cells[:, None]                                # (W,G) 参照index(分数)
            i0 = src.floor().long().clamp(0, Gmax - 2); frac = (src - i0.to(DTYPE)).clamp(0, 1)
            shifted = torch.gather(b, 1, i0) * (1 - frac) + torch.gather(b, 1, i0 + 1) * frac
            shifted = torch.where((src >= 0) & (src <= Gmax - 1), shifted, torch.zeros_like(shifted))
            blurred = F.conv1d(shifted[:, None, :], kernel, padding=kernel.shape[-1] // 2)[:, 0, :]
            if JP > 0:
                blurred = (1 - JP) * blurred + JP * unif                    # 断層: 一様へ少し漏らす
            b_pred = torch.where(valid_g, blurred, torch.zeros_like(blurred))
            b_pred = b_pred / b_pred.sum(1, keepdim=True).clamp_min(1e-30)
        else:
            b_pred = b
        d = ((pad["gr"][:, i][:, None] - gg) / gs).abs(); dp = (d ** POWER).clamp(max=600.0)
        emit = torch.exp(-0.5 * dp)
        bn = torch.where(valid_g, b_pred * emit, torch.zeros_like(b_pred))
        bn = bn / bn.sum(1, keepdim=True).clamp_min(1e-30)
        b = torch.where(act[:, None], bn, b_pred)                          # 無効ステップは更新せず予測のみ
        bt = b.clamp_min(1e-30) ** TEMP; bt = bt / bt.sum(1, keepdim=True).clamp_min(1e-30)
        pred[:, i] = (bt * gtvt).sum(1)
    out = pred.double().cpu().numpy()
    return [out[w, :len(inps[w]["gr"])] for w in range(W)]


# ----------------------- 評価 -----------------------
def predict_errs(P, subset, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_grid_inputs(CACHE[w], P) for w in sub]
        preds = grid_forward(inps, P, DEVICE)
        for k, w in enumerate(sub):
            res[w] = preds[k] - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res

def overall_rmse_on(P, subset, chunk):
    errs = predict_errs(P, subset, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))


# ----------------------- Optuna 探索空間(グリッド版) -----------------------
def suggest_params(trial):
    return dict(
        gr_sig_min     = trial.suggest_float("gr_sig_min", 4.0, 12.0),
        gr_sig_max     = trial.suggest_float("gr_sig_max", 22.0, 60.0),
        gr_sig_def     = trial.suggest_float("gr_sig_def", 70.0, 160.0),
        gr_sig_mult    = trial.suggest_float("gr_sig_mult", 1.6, 4.0),
        gr_power       = trial.suggest_float("gr_power", 1.4, 3.2),
        hgr_smooth_r   = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3, 4, 6]),
        tw_gr_smooth_r = trial.suggest_categorical("tw_gr_smooth_r", [0, 1, 2]),
        proc_sigma     = trial.suggest_float("proc_sigma", 0.3, 20.0, log=True),   # 過程ノイズ(ft)
        dip_drift_scale= trial.suggest_float("dip_drift_scale", 0.0, 1.3),          # 0=水平化, 1=known dip
        init_sigma     = trial.suggest_float("init_sigma", 1.0, 40.0, log=True),    # 初期分布σ(ft)
        post_temp      = trial.suggest_float("post_temp", 0.5, 4.0, log=True),      # 読み出し温度(>1=commit/<1=hedge)
        jump_prob      = trial.suggest_categorical("jump_prob", [0.0, 0.002, 0.01, 0.03]),  # 断層: 一様漏れ
    )


# ----------------------- 状態(再開用) -----------------------
def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"done_codes": [], "results": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    PER_CAT_PATH.write_text(json.dumps(dict(
        algorithm="grid_forward_filter", grid_step=GRID_STEP, grid_margin=GRID_MARGIN,
        n_eval_threshold=N_EVAL_THRESHOLD, z_span_thresholds=list(Z_SPAN_THRESHOLDS),
        categories={str(code): state["results"][str(code)] for code in range(N_CATS)
                    if str(code) in state["results"]},
    ), indent=2))


# ----------------------- カテゴリ実行 -----------------------
def run_category(code, wells):
    study = optuna.create_study(direction="minimize", study_name=f"cat{code}",
                                storage=STORAGE, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=100 + code))
    done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS - done)
    print(f"[cat {code}] wells={len(wells)} optuna {done}/{N_TRIALS} done, run {remaining} more", flush=True)

    def objective(trial):
        P = suggest_params(trial)
        try:
            rmse = overall_rmse_on(P, wells, WELL_CHUNK)
        except Exception:
            return 1e3
        return rmse if math.isfinite(rmse) else 1e3

    def cb(study, trial):
        try:
            bv = study.best_value
        except Exception:
            return
        nd = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        CUR_PATH.write_text(json.dumps(dict(code=code, trials_done=nd, n_wells=len(wells),
                                            best_value=bv, best_params=study.best_params), indent=2))
        tv = trial.value if trial.value is not None else float("nan")
        print(f"   [cat{code}] {nd}/{N_TRIALS} this={tv:.4f} best={bv:.4f}", flush=True)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[cb])
    bp = dict(study.best_params)
    print(f"[cat {code}] best overall = {study.best_value:.4f}", flush=True)
    return dict(code=code, n_wells=len(wells), overall_on_cat=float(study.best_value), best_params=bp)


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  grid_step={GRID_STEP}  storage={STORAGE}")

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
    cats = {code: [] for code in range(N_CATS)}
    for w in all_wells:
        cats[well_code(CACHE[w])].append(w)
    print(f"usable wells: {len(all_wells)}")
    print("カテゴリ別 well 数:", {code: len(cats[code]) for code in range(N_CATS)})

    state = load_state()
    print(f"resume: done codes = {state['done_codes']}")
    for code in range(N_CATS):
        if code in state["done_codes"]:
            continue
        if len(cats[code]) == 0:
            print(f"[cat {code}] well 0 件 → スキップ")
            state["done_codes"].append(code); save_state(state); continue
        rec = run_category(code, cats[code])
        state["results"][str(code)] = rec; state["done_codes"].append(code); save_state(state)
        print(f"==> cat {code} saved. overall_on_cat={rec['overall_on_cat']:.4f}\n")

    print("ALL CATEGORIES DONE.")
    for code in range(N_CATS):
        r = state["results"].get(str(code))
        if r: print(f"  cat {code}: wells={r['n_wells']:4d}  overall_on_cat={r['overall_on_cat']:.4f}")
    print(f"6つのグリッドFパラメータ -> {PER_CAT_PATH}")


if __name__ == "__main__":
    main()
