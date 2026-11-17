"""
v23 (A) 2Dグリッドフィルタ(TVT × dip)チューナー ―― 6カテゴリ別・再開可
==============================================================================
1Dグリッド(tune_grid_filter_v23.py)は dip を状態に持たず長い評価ゾーンで系統ドリフトした(RMSE~35)。
本版は **dip(geo_z変化率)を状態に追加した2Dグリッド**で、PFの2D状態追跡(pos, rate)を格子で再現する。

状態 = (TVT位置 pos[Gp], dip[Gd])。belief (W, Gp, Gd)。
  - 遷移: pos は **dip依存にシフト**(dTVT = dip*dMD - dZ)→ pos拡散 → dip拡散(過程ノイズ)。
  - emission: GR尤度(pos だけに依存, dip方向にブロードキャスト)。
  - 読み出し: dip周辺化 → pos事後の(温度付き)平均 = posterior mean。
GPU: (W, Gp, Gd) を一括バッチ、時間ループのみ逐次。リサンプリング無し・決定論(seed不要)。
速度: EVAL_STRIDE で評価行を間引いて伝播→予測を全行に補間(隣接行はほぼ重複なので影響小)。

6カテゴリ(code 0..5)ごとに独立 Optuna → 6パラメータ。残差ブースティング無し。再開可(SQLite+state.json)。
出力: v23/artifacts_v23/per_category_grid2d.json
実行: rogii_claude/.venv/Scripts/python.exe v23/tune_grid2d_v23.py
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

POS_STEP    = 1.0        # TVT格子の刻み(ft)
POS_MARGIN  = 60.0       # typewell tvt 範囲の外側マージン(ft)
DIP_MIN, DIP_MAX, N_DIP = -0.8, 1.6, 49   # dip(geo_z変化率)格子。step ~0.05
EVAL_STRIDE = 4          # 評価行の間引き(速度)。1=間引きなし。隣接ほぼ重複なので4で実害小
WELL_CHUNK  = 64         # 同時バッチ坑井数(OOMなら下げる)。メモリ ~ W*Gp*Gd
N_WELLS     = None
N_TRIALS    = 200

N_EVAL_THRESHOLD  = 4840.0
Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
N_CATS = 6

STORAGE      = f"sqlite:///{(OUT_DIR / 'optuna2d.db').as_posix()}"
STATE_PATH   = OUT_DIR / "state2d.json"
PER_CAT_PATH = OUT_DIR / "per_category_grid2d.json"
CUR_PATH     = OUT_DIR / "current_cat2d.json"
# ===================================================================

optuna.logging.set_verbosity(optuna.logging.WARNING)
DIP_GRID = np.linspace(DIP_MIN, DIP_MAX, N_DIP).astype(np.float64)
DIP_STEP = float(DIP_GRID[1] - DIP_GRID[0])


# ----------------------- データ -----------------------
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
    n_eval = int(len(z_ev)); z_span = float(np.nanmax(z_ev) - np.nanmin(z_ev)) if len(z_ev) else 0.0
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    last_tvt = float(kn["TVT_input"].iloc[-1])
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev, gr_filled=gr_filled,
                ev_pos=ev_pos, kpos=kpos, kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ir=ir, last_tvt=last_tvt,
                tvt_true=tv, n_eval=n_eval, z_span=z_span)

def well_code(c):
    return int(c["n_eval"] > N_EVAL_THRESHOLD) + 2 * int(np.searchsorted(Z_SPAN_THRESHOLDS, c["z_span"], side="right"))

def _smooth(vals, fb, r):
    r = int(r); s = pd.Series(vals, dtype="float32").interpolate(limit_direction="both").fillna(float(fb))
    return s.to_numpy(np.float32) if r <= 0 else s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy(np.float32)

def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    lo = float(c["tw_tvt"].min() - POS_MARGIN); hi = float(c["tw_tvt"].max() + POS_MARGIN)
    grid_tvt = np.arange(lo, hi + POS_STEP, POS_STEP)
    gg = np.interp(grid_tvt, c["tw_tvt"], tw_gr_pf).astype(np.float64)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    md = c["md"]; z = c["z"]; gr_full = gr_sm[c["ev_pos"]].astype(np.float64)
    idx = np.arange(0, len(md), EVAL_STRIDE)
    if idx[-1] != len(md) - 1: idx = np.append(idx, len(md) - 1)
    md_s = md[idx]; z_s = z[idx]; gr_s = gr_full[idx]
    dMD = np.diff(md_s, prepend=md_s[0]); dZ = np.diff(z_s, prepend=z_s[0])
    return dict(grid_tvt=grid_tvt.astype(np.float64), gg=gg, gr=gr_s,
                dMD=dMD.astype(np.float64), dZ=dZ.astype(np.float64),
                gs=gs, ir=c["ir"], last_tvt=c["last_tvt"], md_sub=md_s, md_full=md)


# ----------------------- 2Dグリッドフィルタ(GPU) -----------------------
def _gauss_kernel(sigma_cells, device):
    sig = max(float(sigma_cells), 1e-3); half = max(1, int(math.ceil(3 * sig)))
    x = torch.arange(-half, half + 1, device=device, dtype=DTYPE)
    k = torch.exp(-0.5 * (x / sig) ** 2); k = k / k.sum()
    return k.view(1, 1, -1)

def _blur(b, kernel, dim):
    W, Gp, Gd = b.shape; pad = kernel.shape[-1] // 2
    if dim == 1:   # pos
        x = b.permute(0, 2, 1).reshape(W * Gd, 1, Gp)
        y = F.conv1d(x, kernel, padding=pad)[:, 0, :]
        return y.reshape(W, Gd, Gp).permute(0, 2, 1).contiguous()
    x = b.reshape(W * Gp, 1, Gd)                              # dip
    y = F.conv1d(x, kernel, padding=pad)[:, 0, :]
    return y.reshape(W, Gp, Gd)

def _pad(inps, device):
    W = len(inps); Tmax = max(len(x["gr"]) for x in inps); Gp = max(len(x["gg"]) for x in inps)
    gtvt = torch.zeros((W, Gp), dtype=DTYPE); gg = torch.zeros((W, Gp), dtype=DTYPE)
    glen = torch.zeros(W, dtype=torch.long)
    gr = torch.full((W, Tmax), float("nan"), dtype=DTYPE); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    dMD = torch.zeros((W, Tmax), dtype=DTYPE); dZ = torch.zeros((W, Tmax), dtype=DTYPE)
    gs = torch.zeros(W, dtype=DTYPE); ir = torch.zeros(W, dtype=DTYPE); last_tvt = torch.zeros(W, dtype=DTYPE)
    for b, x in enumerate(inps):
        T = len(x["gr"]); G = len(x["gg"])
        gtvt[b, :G] = torch.from_numpy(x["grid_tvt"].astype("float32"));  gtvt[b, G:] = gtvt[b, G - 1] if G < Gp else 0
        gg[b, :G] = torch.from_numpy(x["gg"].astype("float32"));          gg[b, G:] = gg[b, G - 1] if G < Gp else 0
        glen[b] = G; gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        dMD[b, :T] = torch.from_numpy(x["dMD"].astype("float32")); dZ[b, :T] = torch.from_numpy(x["dZ"].astype("float32"))
        gs[b] = x["gs"]; ir[b] = x["ir"]; last_tvt[b] = x["last_tvt"]
    to = lambda t: t.to(device)
    return dict(gtvt=to(gtvt), gg=to(gg), glen=to(glen), gr=to(gr), valid=to(valid),
                dMD=to(dMD), dZ=to(dZ), gs=to(gs), ir=to(ir), last_tvt=to(last_tvt), Tmax=Tmax, Gp=Gp)

def grid2d_forward(inps, P, device):
    pad = _pad(inps, device); W = len(inps); Gp = pad["Gp"]; Tmax = pad["Tmax"]; Gd = N_DIP
    POW = float(P["gr_power"]); TEMP = float(P["post_temp"]); JP = float(P["jump_prob"])
    POSN = float(P["pos_proc_sigma"]); DIPN = float(P["dip_proc_sigma"])
    INIT = float(P["init_sigma"]); INITD = float(P["init_dip_sigma"])
    dipg = torch.tensor(DIP_GRID, device=device, dtype=DTYPE)                 # (Gd,)
    gtvt = pad["gtvt"]; gg = pad["gg"]; gs = pad["gs"][:, None]; glen = pad["glen"]
    arP = torch.arange(Gp, device=device, dtype=DTYPE)
    validP = (arP[None, :] < glen[:, None].to(DTYPE))                          # (W,Gp)
    # 初期 belief = pos prior(last_tvt) × dip prior(ir)
    pos_pr = torch.exp(-0.5 * ((gtvt - pad["last_tvt"][:, None]) / max(INIT, 1e-3)) ** 2)  # (W,Gp)
    pos_pr = torch.where(validP, pos_pr, torch.zeros_like(pos_pr))
    dip_pr = torch.exp(-0.5 * ((dipg[None, :] - pad["ir"][:, None]) / max(INITD, 1e-3)) ** 2)  # (W,Gd)
    b = pos_pr[:, :, None] * dip_pr[:, None, :]
    b = b / b.sum((1, 2), keepdim=True).clamp_min(1e-30)
    kpos = _gauss_kernel(POSN / POS_STEP, device); kdip = _gauss_kernel(DIPN / DIP_STEP, device)
    unifP = validP.to(DTYPE); unifP = unifP / unifP.sum(1, keepdim=True).clamp_min(1e-30)
    pred = torch.zeros((W, Tmax), device=device, dtype=DTYPE)
    arPg = arP[None, :, None]                                                  # (1,Gp,1)
    for i in range(Tmax):
        act = pad["valid"][:, i]
        if i > 0:
            shift = (dipg[None, :] * pad["dMD"][:, i][:, None] - pad["dZ"][:, i][:, None]) / POS_STEP  # (W,Gd)
            src = arPg - shift[:, None, :]                                     # (W,Gp,Gd)
            i0 = src.floor().long().clamp(0, Gp - 2); frac = (src - i0.to(DTYPE)).clamp(0, 1)
            b = torch.gather(b, 1, i0) * (1 - frac) + torch.gather(b, 1, i0 + 1) * frac
            b = torch.where((src >= 0) & (src <= Gp - 1), b, torch.zeros_like(b))
            b = _blur(b, kpos, 1)                                              # pos拡散
            b = _blur(b, kdip, 2)                                              # dip拡散
            if JP > 0:
                b = (1 - JP) * b + JP * (unifP[:, :, None] * dip_pr[:, None, :])
            b = b * validP[:, :, None]
            b = b / b.sum((1, 2), keepdim=True).clamp_min(1e-30)
        d = ((pad["gr"][:, i][:, None] - gg) / gs).abs(); dp = (d ** POW).clamp(max=600.0)
        emit = torch.exp(-0.5 * dp)[:, :, None]                                # (W,Gp,1)
        bn = b * emit; bn = bn * validP[:, :, None]
        bn = bn / bn.sum((1, 2), keepdim=True).clamp_min(1e-30)
        b = torch.where(act[:, None, None], bn, b)
        pm = b.sum(2)                                                          # (W,Gp) pos周辺
        pm = pm.clamp_min(1e-30) ** TEMP; pm = pm / pm.sum(1, keepdim=True).clamp_min(1e-30)
        pred[:, i] = (pm * gtvt).sum(1)
    out = pred.double().cpu().numpy()
    return [out[w, :len(inps[w]["gr"])] for w in range(W)]


# ----------------------- 評価(間引き→全行補間) -----------------------
def predict_errs(P, subset, chunk):
    res = {}
    for c0 in range(0, len(subset), chunk):
        sub = subset[c0:c0 + chunk]
        inps = [make_inputs(CACHE[w], P) for w in sub]
        preds = grid2d_forward(inps, P, DEVICE)
        for k, w in enumerate(sub):
            ps = preds[k]; full = np.interp(inps[k]["md_full"], inps[k]["md_sub"], ps)
            res[w] = full - CACHE[w]["tvt_true"]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return res

def overall_rmse_on(P, subset, chunk):
    errs = predict_errs(P, subset, chunk)
    e = np.concatenate([errs[w] for w in subset]); e = e[np.isfinite(e)]
    if e.size == 0 or not np.isfinite(e).all():
        return 1e3
    return float(np.sqrt(np.mean(e ** 2)))


# ----------------------- Optuna 探索空間 -----------------------
def suggest_params(trial):
    return dict(
        gr_sig_min     = trial.suggest_float("gr_sig_min", 4.0, 12.0),
        gr_sig_max     = trial.suggest_float("gr_sig_max", 22.0, 60.0),
        gr_sig_def     = trial.suggest_float("gr_sig_def", 70.0, 160.0),
        gr_sig_mult    = trial.suggest_float("gr_sig_mult", 1.6, 4.0),
        gr_power       = trial.suggest_float("gr_power", 1.4, 3.2),
        hgr_smooth_r   = trial.suggest_categorical("hgr_smooth_r", [0, 1, 2, 3, 4, 6]),
        tw_gr_smooth_r = trial.suggest_categorical("tw_gr_smooth_r", [0]),
        pos_proc_sigma = trial.suggest_float("pos_proc_sigma", 0.2, 8.0, log=True),    # 位置ジッタ(ft)
        dip_proc_sigma = trial.suggest_float("dip_proc_sigma", 0.002, 0.15, log=True), # dipの変わりやすさ
        init_sigma     = trial.suggest_float("init_sigma", 1.0, 30.0, log=True),
        init_dip_sigma = trial.suggest_float("init_dip_sigma", 0.02, 0.6, log=True),
        post_temp      = trial.suggest_float("post_temp", 0.5, 4.0, log=True),
        jump_prob      = trial.suggest_categorical("jump_prob", [0.0]),
    )


# ----------------------- 状態(再開用) -----------------------
def load_state():
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {"done_codes": [], "results": {}}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))
    PER_CAT_PATH.write_text(json.dumps(dict(
        algorithm="grid2d_pos_dip", pos_step=POS_STEP, pos_margin=POS_MARGIN,
        dip_grid=[DIP_MIN, DIP_MAX, N_DIP], eval_stride=EVAL_STRIDE,
        n_eval_threshold=N_EVAL_THRESHOLD, z_span_thresholds=list(Z_SPAN_THRESHOLDS),
        categories={str(c): state["results"][str(c)] for c in range(N_CATS) if str(c) in state["results"]},
    ), indent=2))


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
            r = overall_rmse_on(P, wells, WELL_CHUNK)
        except Exception:
            return 1e3
        return r if math.isfinite(r) else 1e3

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
    print(f"[cat {code}] best overall = {study.best_value:.4f}", flush=True)
    return dict(code=code, n_wells=len(wells), overall_on_cat=float(study.best_value), best_params=dict(study.best_params))


def main():
    global CACHE
    if DEVICE != "cuda":
        print("WARNING: CUDA not available, will be slow.")
    print(f"device={DEVICE}  pos_step={POS_STEP} dip_grid=[{DIP_MIN},{DIP_MAX},{N_DIP}] stride={EVAL_STRIDE}  storage={STORAGE}")
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
            print(f"[cat {code}] well 0 件 → スキップ"); state["done_codes"].append(code); save_state(state); continue
        rec = run_category(code, cats[code])
        state["results"][str(code)] = rec; state["done_codes"].append(code); save_state(state)
        print(f"==> cat {code} saved. overall_on_cat={rec['overall_on_cat']:.4f}\n")

    print("ALL CATEGORIES DONE.")
    for code in range(N_CATS):
        r = state["results"].get(str(code))
        if r: print(f"  cat {code}: wells={r['n_wells']:4d}  overall_on_cat={r['overall_on_cat']:.4f}")
    print(f"6つの2Dグリッドパラメータ -> {PER_CAT_PATH}")


if __name__ == "__main__":
    main()
