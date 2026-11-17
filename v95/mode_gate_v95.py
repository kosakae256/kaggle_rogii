# -*- coding: utf-8 -*-
"""v95 モードゲート（v34/diagnose_modes_v34.ipynb の v95移植）。
per-seed dump(seed_dump_pf1/2/3.npz, tw/self/nbr)から井戸レベルのモードを分解し、
「seed事後質量がshiftモードに乗っているのにblendが動いていない井戸」を学習ゲートで検出、
全モード確率加重 posterior-mean shift を v95 blend に適用して OOF が下がるか検証。

基準 blend = nnls(gbm,tcn,l5) = v95 CV 5.987。真TVTは label/評価のみ。
出力: v95/artifacts_v95b/mode_gate_v95.txt
実行: .venv/Scripts/python.exe v95/mode_gate_v95.py
"""
import os, sys, io, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v95\artifacts_v95b")
PQ = ART / "data" / "train.parquet"
REF_COLS = [("tw", k, f"pf_ancc_delta_{k}") for k in (1, 2, 3)] + \
           [("self", k, f"pf_self_delta_{k}") for k in (1, 2, 3)] + \
           [("nbr", k, f"pf_nbr_delta_{k}") for k in (1, 2, 3)]
need = ["well", "target", "last_known_tvt", "md_since"] + [c for _, _, c in REF_COLS] \
       + ["prov_nbr_nref", "prov_nbr_dist_min"]
sch = set(pq.read_schema(PQ).names)
need = [c for c in need if c in sch]
tb = pq.read_table(PQ, columns=need).to_pandas()
g = tb["well"].astype(str).to_numpy()
y = tb["target"].to_numpy(np.float64); lk = tb["last_known_tvt"].to_numpy(np.float64)
mds = np.maximum(tb["md_since"].to_numpy(np.float64), 0.0)
tvt_true = lk + y
REFD = {c: tb[c].to_numpy(np.float64) for _, _, c in REF_COLS if c in tb.columns}   # delta = value - lk

# ---- v95 blend(nnls gbm,tcn,l5)= 残差予測 ----
d1 = np.load(ART.parent / "artifacts_v95" / "l1_oof.npz")
gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(ART / "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(ART / "l5_ext_oof.npz")["l5_mean"].astype(float)
M = np.stack([gbm, tcn, l5], 1); mfin = np.isfinite(M).all(1) & np.isfinite(y)
wgt, _ = nnls(M[mfin], y[mfin]); wgt = wgt / wgt.sum()
blend_res = M @ wgt                          # 残差空間 blend
blend = lk + blend_res                       # 絶対TVT
def rmse(e):
    e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
wid = pd.unique(g); rows_of = {w: np.where(g == w)[0] for w in wid}
e0 = blend - tvt_true; base_overall = rmse(e0)
per_base = {w: rmse(e0[rows_of[w]]) for w in wid}
print(f"[v95gate] blend nnls重み(gbm,tcn,l5)={tuple(round(float(x),2) for x in wgt)} overall={base_overall:.4f} wells={len(wid)}", flush=True)

# ---- nbr有無(nbr delta が tw delta とほぼ同一=フォールバック=nbr無し)----
WNBR = {}
for w in wid:
    r = rows_of[w]
    if "pf_nbr_delta_1" in REFD and "pf_ancc_delta_1" in REFD:
        WNBR[w] = bool(np.nanmedian(np.abs(REFD["pf_nbr_delta_1"][r] - REFD["pf_ancc_delta_1"][r])) > 0.5)
    else:
        WNBR[w] = False

# ---- 参照9本オフセット(delta - blend_res の井戸中央値)----
OBS = {}
for w in wid:
    r = rows_of[w]; refs = {}
    for src, pset, c in REF_COLS:
        if c not in REFD: continue
        if src == "nbr" and not WNBR[w]: continue
        v = REFD[c][r] - blend_res[r]
        if np.isfinite(v).sum() < max(10, 0.3 * len(r)): continue
        refs[(src, pset)] = float(np.nanmedian(v))
    OBS[w] = refs

DUMPS = {k: np.load(ART / f"seed_dump_pf{k}.npz", allow_pickle=True) for k in (1, 2, 3)}
KEYS = {k: set(DUMPS[k].files) for k in (1, 2, 3)}
n_seed_wells = sum(1 for w in wid if f"{w}|ps_tw" in KEYS[1])
print(f"[v95gate] seed dumpあり {n_seed_wells}井戸", flush=True)

# ---- ★frame確認: dump ps ww加重平均 ≈ (pf_ancc_delta_1 + lk) か ----
_w0 = wid[np.argmax([f"{w}|ps_tw" in KEYS[1] for w in wid])]
_z = DUMPS[1]; _r = rows_of[_w0]; _ps = _z[f"{_w0}|ps_tw"].astype(np.float64); _ww = _z[f"{_w0}|ww_tw"]
_T = min(_ps.shape[1], len(_r)); _mean = (_ww[:, None] * _ps[:, :_T]).sum(0)
_ref = (REFD["pf_ancc_delta_1"][_r][:_T] + lk[_r][:_T])
print(f"[frame] |ps加重平均 − (delta+lk)| median={np.nanmedian(np.abs(_mean - _ref)):.2f}ft (小=ps絶対TVT frame確認)", flush=True)

GAP_LINK = 8.0; MODE_MIN, MODE_MAX = 4.0, 60.0
SRCS = ("tw", "self", "nbr"); FLTS = (("sm", "ps"), ("fw", "pfw"))

ITEMS = {}
for w in wid:
    r = rows_of[w]; it = []
    for (src, pset), d in OBS[w].items():
        it.append(dict(kind="ref", src=src, pset=pset, flt="", d=d, wt=1.0))
    for k in (1, 2, 3):
        z = DUMPS[k]
        for src in SRCS:
            if src == "nbr" and not WNBR[w]: continue
            for flt, pref in FLTS:
                key = f"{w}|{pref}_{src}"
                if key not in KEYS[k]: continue
                ps = z[key]; ww_ = z[f"{w}|ww_{src}"]; T = min(ps.shape[1], len(r))
                dmed = np.nanmedian(ps[:, :T].astype(np.float64) - blend[r[:T]][None, :], axis=1)
                for s in range(ps.shape[0]):
                    if np.isfinite(dmed[s]):
                        it.append(dict(kind="seed", src=src, pset=k, flt=flt, d=float(dmed[s]), wt=float(ww_[s]), sid=s))
    ITEMS[w] = it

def cluster_items(items):
    its = sorted(items, key=lambda x: x["d"])
    if not its: return []
    groups, cur = [], [its[0]]
    for x in its[1:]:
        if x["d"] - cur[-1]["d"] > GAP_LINK: groups.append(cur); cur = [x]
        else: cur.append(x)
    groups.append(cur); out = []
    for m in groups:
        ds = np.array([x["d"] for x in m]); wts = np.array([x["wt"] for x in m])
        o = np.argsort(ds); cw = np.cumsum(wts[o])
        c = float(ds[o][min(np.searchsorted(cw, cw[-1] / 2), len(ds) - 1)])
        out.append(dict(c=c, members=m, spread=float(np.std(ds)) if len(ds) > 1 else 0.0))
    return out

def oracle(sel):
    e = e0.copy(); picks = {}
    for w in wid:
        modes = cluster_items([x for x in ITEMS[w] if sel(x)])
        cands = [0.0] + [m["c"] for m in modes if MODE_MIN <= abs(m["c"]) <= MODE_MAX]
        r = rows_of[w]; ew = e0[r]; sc = [rmse(ew + c) for c in cands]
        j = int(np.argmin(sc)); picks[w] = cands[j]; e[r] = ew + cands[j]
    return rmse(e), picks

or_all, pick = oracle(lambda x: True)
e_fs = e0.copy()
for w in wid:
    r = rows_of[w]; e_fs[r] = e0[r] - np.nanmean(e0[r])
fs_overall = rmse(e_fs)
rec = lambda v: 100 * (base_overall - v) / max(base_overall - fs_overall, 1e-9)
print(f"[v95gate] 自由シフト上限={fs_overall:.4f} / モードoracle併合={or_all:.4f}(回収{rec(or_all):.0f}%) / 非ゼロ選択{sum(1 for w in wid if pick[w]!=0)}井", flush=True)

# ---- 特徴(cell3 物理 + cell4 質量。xmクロス採点は簡略化のためv1では省略)----
def score_traj(traj, gg, gmin, gst, gs, gr):
    G = len(gg); ii = np.clip((np.asarray(traj, float) - gmin) / gst, 0.0, G - 1.000001)
    i0 = np.floor(ii).astype(int); t = ii - i0
    eg = gg[i0] * (1 - t) + gg[np.minimum(i0 + 1, G - 1)] * t
    m = np.isfinite(gr) & np.isfinite(eg)
    if m.sum() < 10: return np.nan
    d = (gr[m] - eg[m]) / gs; return float(np.mean(np.exp(-0.5 * d * d)))

MODES = {w: [m for m in cluster_items(ITEMS[w]) if MODE_MIN <= abs(m["c"]) <= MODE_MAX] for w in wid}
PHYS_KEYS = ("rough_ratio", "dip_dev", "dip_absmax", "geo_jump", "smfw_gap", "edge_frac", "tail_dev", "ll_margin")
XM = {}; PHYS = {}
for w in wid:
    if not MODES[w]: continue
    r = rows_of[w]
    for mi, m in enumerate(MODES[w]):
        acc = {"tw": [], "self": [], "nbr": []}; ph = {kk: [] for kk in PHYS_KEYS}
        for k in (1, 2, 3):
            dpz = DUMPS[k]
            if f"{w}|meta" not in KEYS[k]: continue
            gmin, gst, gs = dpz[f"{w}|meta"]; gr = dpz[f"{w}|gr"].astype(np.float64)
            md_ = dpz[f"{w}|md"].astype(np.float64); zz = dpz[f"{w}|z"].astype(np.float64); ir = float(dpz[f"{w}|ir"])
            T = min(len(r), len(gr), len(md_)); trajs, wts, smfw, llm = [], [], [], []
            for src in SRCS:
                sids_src = {x["sid"] for x in m["members"] if x["kind"] == "seed" and x["pset"] == k and x["src"] == src}
                if sids_src and f"{w}|ll_{src}" in KEYS[k]:
                    ll_ = dpz[f"{w}|ll_{src}"]; fin = np.isfinite(ll_)
                    if fin.sum() > 1 and np.nanstd(ll_[fin]) > 1e-9:
                        llm.append(float((np.nanmean(ll_[list(sids_src)]) - np.nanmean(ll_[fin])) / (np.nanstd(ll_[fin]) + 1e-9)))
                for flt, pref in FLTS:
                    key = f"{w}|{pref}_{src}"
                    if key not in KEYS[k]: continue
                    sids = [x["sid"] for x in m["members"] if x["kind"] == "seed" and x["pset"] == k and x["src"] == src and x["flt"] == flt]
                    if not sids: continue
                    ps = dpz[key]; ww_ = dpz[f"{w}|ww_{src}"]
                    trajs.append(ps[sids, :T].astype(np.float64)); wts.append(np.asarray(ww_[sids], float))
                    other = f"{w}|{'pfw' if pref == 'ps' else 'ps'}_{src}"
                    if other in KEYS[k]:
                        po = dpz[other]; smfw.append(float(np.nanmean(np.abs(ps[sids, :T].astype(np.float64) - po[sids, :T].astype(np.float64)))))
            has_seed = bool(trajs)
            if has_seed:
                Tm = min(t_.shape[1] for t_ in trajs); allt = np.concatenate([t_[:, :Tm] for t_ in trajs], 0)
                allw = np.concatenate(wts); allw = allw / allw.sum() if allw.sum() > 1e-12 else np.full(len(allw), 1.0 / len(allw))
                traj = (allw[:, None] * allt).sum(0); Tk = Tm
            else:
                traj = blend[r[:T]] + m["c"]; Tk = T
            btraj = blend[r[:Tk]]
            if has_seed and Tk >= 5:
                mdk = md_[:Tk]; zk = zz[:Tk]; dmd = np.maximum(np.diff(mdk), 1.0)
                gt = np.diff(traj + zk) / dmd
                ph["dip_dev"].append(abs(float(np.nanmedian(gt)) - ir)); ph["dip_absmax"].append(float(np.nanpercentile(np.abs(gt), 95)))
                ph["geo_jump"].append(float(np.nanmax(np.abs(np.diff(traj + zk)))))
                r2 = np.abs(np.diff(traj, 2)); b2 = np.abs(np.diff(btraj, 2))
                ph["rough_ratio"].append(float((np.nanmean(r2) + 1e-6) / (np.nanmean(b2) + 1e-6)))
                gmax = gmin + (len(dpz[f"{w}|gg_tw"]) - 1) * gst
                ph["edge_frac"].append(float(np.mean((traj < gmin + 5.0) | (traj > gmax - 5.0))))
                th = Tk // 2; ph["tail_dev"].append(abs(float(np.nanmedian((traj - btraj)[th:])) - m["c"]))
            if smfw: ph["smfw_gap"].append(float(np.mean(smfw)))
            if llm: ph["ll_margin"].append(float(np.mean(llm)))
            for nmk, gk in [("tw", "gg_tw"), ("self", "gg_self"), ("nbr", "gg_nbr")]:
                key = f"{w}|{gk}"
                if key not in KEYS[k]: continue
                gg = dpz[key].astype(np.float64)
                sm = score_traj(traj[:Tk], gg, gmin, gst, gs, gr[:Tk]); sb = score_traj(btraj, gg, gmin, gst, gs, gr[:Tk])
                if np.isfinite(sm) and np.isfinite(sb): acc[nmk].append(sm - sb)
        XM[(w, mi)] = {f"xm_{kk}": (float(np.mean(v)) if v else np.nan) for kk, v in acc.items()}
        PHYS[(w, mi)] = {kk: (float(np.mean(v)) if v else np.nan) for kk, v in ph.items()}

rows = []
for w in wid:
    if not MODES[w]: continue
    items = ITEMS[w]; nref = max(sum(1 for x in items if x["kind"] == "ref"), 1)
    n_ext = sum(1 for x in items if x["kind"] == "ref" and x["src"] in ("self", "nbr"))
    tot_wt = sum(x["wt"] for x in items)
    stay_wt = sum(x["wt"] for m in cluster_items(items) if abs(m["c"]) < MODE_MIN for x in m["members"])
    flts = [f for f, _ in FLTS]
    runs_tw = [("tw", k, f) for k in (1, 2, 3) for f in flts]
    runs_ext = ([("self", k, f) for k in (1, 2, 3) for f in flts] + ([("nbr", k, f) for k in (1, 2, 3) for f in flts] if WNBR[w] else []))
    for mi, m in enumerate(MODES[w]):
        refs = [x for x in m["members"] if x["kind"] == "ref"]; seeds = [x for x in m["members"] if x["kind"] == "seed"]
        srcs_ref = {x["src"] for x in refs}; mass = {}
        for x in seeds:
            kk = (x["src"], x["pset"], x["flt"]); mass[kk] = mass.get(kk, 0.0) + x["wt"]
        xm = XM.get((w, mi), {})
        rows.append(dict(
            well=w, c=m["c"], y=int(pick[w] != 0.0 and abs(pick[w] - m["c"]) < 1e-9),
            vote=len(refs) / nref, ext_vote=(sum(1 for x in refs if x["src"] in ("self", "nbr")) / n_ext) if n_ext else 0.0,
            stay_share=stay_wt / max(tot_wt, 1e-9),
            n_src=len(srcs_ref), has_nbr=int("nbr" in srcs_ref), has_self=int("self" in srcs_ref), has_tw=int("tw" in srcs_ref),
            n_pset_ref=len({x["pset"] for x in refs}),
            tw_seed_mass=float(np.mean([mass.get(rn, 0.0) for rn in runs_tw])),
            ext_seed_mass=float(np.mean([mass.get(rn, 0.0) for rn in runs_ext])) if runs_ext else 0.0,
            sm_seed_mass=float(np.mean([v for kk_, v in mass.items() if kk_[2] == "sm"])) if any(kk_[2] == "sm" for kk_ in mass) else 0.0,
            fw_seed_mass=float(np.mean([v for kk_, v in mass.items() if kk_[2] == "fw"])) if any(kk_[2] == "fw" for kk_ in mass) else 0.0,
            seed_frac=len(seeds) / max(sum(1 for x in items if x["kind"] == "seed"), 1),
            n_src_seed=len({x["src"] for x in seeds}), n_pset_seed=len({x["pset"] for x in seeds}), n_flt=len({x["flt"] for x in seeds}),
            c_abs=abs(m["c"]), spread=m["spread"],
            has_opp=int(any(np.sign(x["c"]) != np.sign(m["c"]) for x in MODES[w] if x is not m)),
            xm_tw=xm.get("xm_tw", np.nan), xm_self=xm.get("xm_self", np.nan), xm_nbr=xm.get("xm_nbr", np.nan),
            **PHYS.get((w, mi), {})))
TBL = pd.DataFrame(rows)
print(f"[v95gate] 候補モード {len(TBL)}行 / 分裂井戸 {TBL['well'].nunique()} / 正解モード {int(TBL['y'].sum())}", flush=True)

FEATS = ["vote", "ext_vote", "stay_share", "n_src", "has_nbr", "has_self", "has_tw", "n_pset_ref",
         "tw_seed_mass", "ext_seed_mass", "sm_seed_mass", "fw_seed_mass", "seed_frac", "n_src_seed", "n_pset_seed", "n_flt",
         "c_abs", "spread", "has_opp", "xm_self", "xm_nbr", "xm_tw",
         "rough_ratio", "dip_dev", "dip_absmax", "geo_jump", "smfw_gap", "edge_frac", "tail_dev", "ll_margin"]
FEATS = [f for f in FEATS if f in TBL.columns]
aucs = {}
for f in FEATS:
    v = TBL[f].to_numpy(float); mm = np.isfinite(v)
    if mm.sum() < 30 or TBL["y"][mm].nunique() < 2: continue
    a = roc_auc_score(TBL["y"][mm], v[mm]); aucs[f] = max(a, 1 - a)
A = pd.Series(aucs).sort_values(ascending=False)
print("[v95gate] 単変量AUC top10:\n" + A.head(10).to_string(float_format="%.3f"), flush=True)

X = np.nan_to_num(TBL[FEATS].to_numpy(float), nan=-999.0); yb = TBL["y"].to_numpy(); grp = TBL["well"].to_numpy()
p_lgb = np.zeros(len(TBL)); p_log = np.zeros(len(TBL))
for tr, va in GroupKFold(5).split(X, yb, grp):
    mdl = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=7, min_child_samples=25,
                             subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    mdl.fit(X[tr], yb[tr]); p_lgb[va] = mdl.predict_proba(X[va])[:, 1]
    sc = StandardScaler().fit(X[tr]); lo = LogisticRegression(C=0.3, max_iter=2000).fit(sc.transform(X[tr]), yb[tr])
    p_log[va] = lo.predict_proba(sc.transform(X[va]))[:, 1]
auc_lgb = roc_auc_score(yb, p_lgb); auc_log = roc_auc_score(yb, p_log)
print(f"[v95gate] OOF AUC: LGBM {auc_lgb:.3f} / Logistic {auc_log:.3f}", flush=True)
TBL["p_lgb"], TBL["p_log"] = p_lgb, p_log

def apply_gate(pcol, lam, agg="post", thr=None):
    out = blend.copy()
    for w, sub in TBL.groupby("well"):
        p = sub[pcol].to_numpy(float); c = sub["c"].to_numpy(float)
        if agg == "post":
            w_stay = max(0.0, 1.0 - float(p.max())); norm = w_stay + float(p.sum())
            if norm <= 1e-9: continue
            out[rows_of[w]] += lam * float((p * c).sum()) / norm
        else:
            j = int(np.argmax(p))
            if thr is not None and p[j] < thr: continue
            out[rows_of[w]] += lam * (p[j] * c[j] if agg == "soft" else c[j])
    return out

res = []
for name in ["p_log", "p_lgb"]:
    for lam in [0.4, 0.6, 0.8, 1.0]:
        res.append((f"{name} soft λ{lam}", rmse(apply_gate(name, lam, "soft") - tvt_true)))
    for lam in [0.6, 0.8, 1.0]:
        res.append((f"{name} post λ{lam}", rmse(apply_gate(name, lam, "post") - tvt_true)))
    for thr in [0.5, 0.7, 0.85, 0.95]:                 # ★高精度hard: 確信井のみコミット
        for lam in [0.8, 1.0]:
            n_move = sum(1 for w, sub in TBL.groupby("well") if sub[name].max() >= thr)
            res.append((f"{name} hard λ{lam} p>={thr}({n_move}井)", rmse(apply_gate(name, lam, "hard", thr) - tvt_true)))
res.sort(key=lambda t: t[1])
out = ["=== v95 モードゲート結果 ===",
       f"blend(base)={base_overall:.4f} / 自由シフト上限={fs_overall:.4f} / モードoracle={or_all:.4f}(回収{rec(or_all):.0f}%)",
       f"OOF AUC: LGBM {auc_lgb:.3f} / Logistic {auc_log:.3f}  正解モード{int(TBL['y'].sum())}/{len(TBL)}行",
       "--- 学習ゲート(適用後 overall) ---"]
for nm, v in res:
    out.append(f"  {nm:20s} = {v:.4f}  ({v-base_overall:+.4f})")
best = res[0]
out.append(f"★best = {best[0]} = {best[1]:.4f} (blend比 {best[1]-base_overall:+.4f})")
txt = "\n".join(out); print("\n" + txt, flush=True)
(ART / "mode_gate_v95.txt").write_text(txt, encoding="utf-8")
