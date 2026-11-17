# -*- coding: utf-8 -*-
"""
v22 ローカル検証: ヒューリスティックモデル(現行仕様)の train overall RMSE アブレーション
========================================================================================
submit_heuristic_v22.ipynb と同じ予測チェーンを train(真値あり)で回し、成分ごとの寄与を測る。

  チェーン: PF(カテゴリ別 best scale)
            → (1-bw)*PF + bw*beam            [beam: codes 2,4,5]
            → (1-hw)*pred + hw*last_known_tvt [hold: 全code]
            → apply_pp(tau ランプ) + sg_smooth

出力: 各段階(pf / +beam / +hold / +applypp=full)の overall RMSE をカテゴリ別 + プールで表示。
      → LB(提出)と同一チェーンなので CV↔LB を対応づけられる。

トグル: USE_R0(r0単体 ↔ 6カテゴリ別調整PF)。beam/hold/applypp はアブレーションとして常に段階表示。
実行: rogii_claude/.venv/Scripts/python.exe v22/compare_percat_vs_scale_v22.py
※ PFエンジンは tune_pf_per_category_v22 を import して再利用(提出ノートと同一)。Claude は実行しない。
"""

import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.signal import savgol_filter
from numba import njit

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v22"))
import tune_pf_per_category_v22 as T22   # noqa: E402  (build_cache/make_inputs/_pf_core/_pad/well_code を再利用)

# ============================== CONFIG(提出ノートと一致) ==============================
PERCAT   = PROJECT / "v22" / "artifacts_v22" / "per_category_pf.json"
DATA_DIR = T22.DATA_DIR
SEEDS    = T22.EVAL_SEEDS        # 32
CHUNK    = T22.EVAL_CHUNK        # 400
DEVICE   = T22.DEVICE
N_CATS   = T22.N_CATS
N_WELLS  = None

USE_R0    = False               # True: r0(v19 round0)単体。False: 6カテゴリ別調整PF
BEST_SCALE = {0: 8.0, 1: 8.0, 2: 5.0, 3: 20.0, 4: 12.0, 5: 5.0}
BEAM_W = {0: 0.0, 1: 0.0, 2: 0.2, 3: 0.0, 4: 0.05, 5: 0.2}
HOLD_W = {0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.05, 5: 0.05}
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1), (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3), (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4), (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]
PP_PARAMS = {"alpha": 1.0, "tau": 85, "w_pf": 0.09}
SG_W, SG_P = 17, 3
R0_PARAMS = {"n_particles": 500, "likelihood_scale": 6.265873932344672, "init_pos_std": 1.0481897997156377, "init_rate_std": 0.004385968881669577, "mom": 0.9995659517817024, "vn": 0.0009771590366402883, "pn": 0.02193433099852571, "rp": 1.266960649917433, "rr": 0.0003377853906820054, "resamp": 0.05083194815648496, "gr_sig_min": 10.293696795145928, "gr_sig_max": 49.46806691781805, "gr_sig_def": 129.97956203556075, "gr_sig_mult": 3.412937310824599, "tvt_clip_margin": 159.57810929194702, "rate_clip": 0.0, "gr_power": 1.5371545258734838, "hgr_smooth_r": 1, "tw_gr_smooth_r": 0, "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 4.0313262188520795, "use_phys": True}
# =====================================================================================


def build_cache(wid):
    """train 坑井: v22 build_cache と同一 + last_known_tvt/md(beam/hold/applypp 用)。"""
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
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum() >= 3 else 0.
    tv = ev["TVT"].values.astype(np.float64) if "TVT" in ev.columns else None
    z_ev = ev["Z"].values.astype(np.float64)
    return dict(md=ev["MD"].values.astype(np.float64), z=z_ev, gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float), tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb,
                ls=ls, ir=ir, tvt_true=tv, n_eval=int(len(z_ev)),
                z_span=float(np.nanmax(z_ev)-np.nanmin(z_ev)) if len(z_ev) else 0.0,
                last_known_md=float(kn["MD"].iloc[-1]), last_known_tvt=float(kn["TVT_input"].iloc[-1]))


def run_pf_scale(inps, P, seed, n_seeds, chunk, scale):
    """PFを1回走らせ scale(温度)で seed 尤度加重 → 各坑井の評価行TVT予測(= 提出ノートと同一)。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None]*W
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0+chunk]; w = len(sub); pad = T22._pad(sub, DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=DEVICE); gen.manual_seed(int(seed))
        pts, ll = T22._pf_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                               pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
                               pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
                               P, N, DEVICE, gen)
        pts = pts.view(w, S, -1).double().cpu().numpy(); ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]; llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                ww = np.exp(np.where(np.isfinite(llw), llw-mx, -np.inf)/max(scale, 1e-6))
                ww = ww/ww.sum() if ww.sum() > 1e-300 else np.full(S, 1.0/S)
            else:
                ww = np.full(S, 1.0/S)
            out[c0+j] = (ww[:, None]*pr).sum(0)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return out


# ===== beam(7.191 移植, 提出ノートと同一) =====
@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS*6
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
                tot = cost + (gv-tw_gr[ni])**2/es + mc*(d if d >= 0 else -d)
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
            for j in range(i+1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n-1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path

def _nn(arr, val): return int(np.abs(arr - val).argmin())

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt)
    sgr = T22._smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, int(bs), float(mc), float(es))
    return tw_tvt[path].astype(np.float32)

def run_beam(c):
    tw_tvt = c["tw_tvt"].astype(np.float64)
    fb = float(np.nanmean(c["tw_gr"])) if np.isfinite(np.nanmean(c["tw_gr"])) else 0.0
    tw_gr = np.where(np.isfinite(c["tw_gr"]), c["tw_gr"], fb).astype(np.float64)
    hgr = c["gr_filled"][c["ev_pos"]]; last_tvt = c["last_known_tvt"]
    res = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r) for (bs, mc, es, r) in BEAM_CONFIGS]
    return np.stack(res, 0).mean(0).astype(np.float64)

def apply_pp(pred, c):
    md_since = np.maximum(c["md"] - c["last_known_md"], 0.0)
    d = pred - c["last_known_tvt"]
    if PP_PARAMS.get("tau"):
        d = d * (1.0 - np.exp(-md_since / float(PP_PARAMS["tau"])))
    d = d * float(PP_PARAMS.get("alpha", 1.0))
    out = c["last_known_tvt"] + d
    n = len(out); wl = min(SG_W, n)
    if wl % 2 == 0: wl -= 1
    if wl >= SG_P + 2:
        out = savgol_filter(out, wl, SG_P)
    return out


def rmse(residlist):
    if not residlist: return float("nan")
    e = np.concatenate(residlist); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")


def main():
    cats = json.loads(PERCAT.read_text()).get("categories", {})
    print(f"device={DEVICE} seeds={SEEDS} | MODEL={'r0(single)' if USE_R0 else 'per-category tuned'}")

    wells = T22.list_wells()
    if N_WELLS: wells = wells[:N_WELLS]
    CACHE = {}
    for w in wells:
        c = build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    cat_wells = {code: [] for code in range(N_CATS)}
    for w, c in CACHE.items():
        cat_wells[T22.well_code(c)].append(w)
    print("カテゴリ別 well 数:", {c: len(cat_wells[c]) for c in range(N_CATS)})

    # 各段階の残差を貯める: L0=pf, L1=+beam, L2=+hold, L3=+applypp(full)
    LV = ["pf", "+beam", "+hold", "+applypp"]
    res = {k: {code: [] for code in range(N_CATS)} for k in LV}
    t0 = time.time()
    for code in range(N_CATS):
        wl = cat_wells[code]; rec = cats.get(str(code))
        if not wl or (rec is None and not USE_R0):
            continue
        P = dict(R0_PARAMS) if USE_R0 else dict(rec["best_params"]); P.setdefault("use_phys", T22.USE_PHYS)
        scale = BEST_SCALE[code]; seed = 100*code; bw = BEAM_W.get(code, 0.0); hw = HOLD_W.get(code, 0.0)
        inps = [T22.make_inputs(CACHE[w], P) for w in wl]
        preds = run_pf_scale(inps, P, seed, SEEDS, CHUNK, scale)
        for w, base in zip(wl, preds):
            c = CACHE[w]; tv = c["tvt_true"]
            p0 = np.asarray(base, float)
            p1 = p0.copy()
            if bw > 0:
                beam = run_beam(c); m = min(len(p1), len(beam)); p1[:m] = (1-bw)*p1[:m] + bw*beam[:m]
            p2 = p1.copy()
            if hw > 0:
                p2 = (1-hw)*p2 + hw*c["last_known_tvt"]
            p3 = apply_pp(p2, c)
            for k, pp in zip(LV, (p0, p1, p2, p3)):
                res[k][code].append(pp - tv)
        print(f"[cat {code}] wells={len(wl)} scale={scale} beam={bw} hold={hw} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== カテゴリ別 overall RMSE(各段階) ===")
    print(f"{'cat':>3} {'n':>5} " + " ".join(f"{k:>9}" for k in LV))
    for code in range(N_CATS):
        wl = cat_wells[code]
        if not res["pf"][code]:
            continue
        vals = [rmse(res[k][code]) for k in LV]
        print(f"{code:>3} {len(wl):>5} " + " ".join(f"{v:>9.3f}" for v in vals))

    print("\n=== プール overall RMSE(全カテゴリ) ===")
    base = None
    for k in LV:
        pooled = rmse([e for code in range(N_CATS) for e in res[k][code]])
        d = "" if base is None else f"  (Δ vs pf = {pooled-base:+.4f})"
        if base is None: base = pooled
        print(f"  {k:>10} : {pooled:.4f}{d}")
    print("\n  ※ +applypp が提出モデル(全成分ON)。beam/hold は codes {2,4,5}/{全code} でのみ効く。")
    print("     LB(submit_heuristic_v22.ipynb 全ON)とこの +applypp が CV↔LB の対応。")


if __name__ == "__main__":
    main()
