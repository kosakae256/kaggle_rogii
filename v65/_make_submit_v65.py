# submit_pf_v65.ipynb を生成(1回実行用)。エンジン計算部は pf_engine_v65.py から抽出して埋め込む。
import json, sys, re
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

eng = open("v65/pf_engine_v65.py", encoding="utf-8").read()
i0 = eng.index("def _bin_pairs(")
i1 = eng.index("def predict_errs(")
CORE = eng[i0:i1].rstrip()          # _smooth/_grid/make_inputs/_pf_core/_pad/run_forward_ens(純計算)

def code(src): return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.splitlines(keepends=True))
def md(src): return dict(cell_type="markdown", metadata={}, source=src.splitlines(keepends=True))

c0 = """# v65 — 生PF提出(拡張パラメータエンジン, ハードコード複数セット平均)
- **PARAM_SETS をハードコード**(tune_pf_v65 の `*_current.json` / `*_best.json` の `params` をそのまま貼る)
- 複数セット指定時は坑井ごとに軌跡を平均(AVG="mean" / "median")。※多峰井戸で平均は谷に落ちる既知リスク
  (v19/v22実測)— セットは「似た系」で組むこと。
- 各セットの `tw_source`(tw/self/nbr/self+nbr)と `self_mix_w`/`nbr_mix_w` で疑似typewell混合を制御
  (nbr: 1500ft以内2本未満は自動でtwフォールバック。grid_step は tw/self/nbr の全格子に共通適用)
- 錨(use_anchor=True)を含むセットがある場合は dataset に struct_field_v38.npz が必要
  (rogii-v64-artifacts 等に同梱済み。無ければ錨セットは自動で錨OFF実行+警告)。"""

c1 = r'''# ===== 設定(ここだけ編集) =====
import os, json, time, glob as _glob, math, warnings, pickle
import numpy as np, pandas as pd
from pathlib import Path
import torch
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

# --- PFパラメータセット(ハードコード。tune_pf_v65 の params をそのまま貼る) ---
PARAM_SETS = [
    # 例: v38 struct錨 mult20 のbest(v65拡張ノブはデフォルト値で付与)。差し替え/追加自由
    {"use_phys": False, "use_anchor": True, "tw_source": "tw", "self_mix_w": 1.0, "nbr_mix_w": 1.0, "anchor_mult": 20.0, "anchor_ramp_md": 0.0, "lk_floor": 0.0, "gr_debias": 0, "rate_target": "zero", "grid_step": 0.2, "n_particles": 600, "likelihood_scale": 29.056396339290437, "init_pos_std": 1.6168048868046325, "init_rate_std": 0.014412715870661067, "mom": 0.9995682504305502, "vn": 0.000873883338794689, "pn": 0.00606209538025288, "rp": 0.14025726060525087, "rr": 0.00398567931602027, "resamp": 0.18645022443117387, "gr_sig_min": 10.63834771185878, "gr_sig_max": 43.95858017409127, "gr_sig_def": 86.87561008087175, "gr_sig_mult": 2.7681905590609426, "tvt_clip_margin": 86.30698497112195, "rate_clip": 0.0, "gr_power": 1.85175446220882, "hgr_smooth_r": 2, "tw_gr_smooth_r": 0, "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 2.3072090423765363},
]
N_SEEDS = 32          # seedアンサンブル(尤度加重平均)
SEED    = 777
AVG     = "mean"      # 複数セットの合成: "mean" | "median"
PP      = None        # 例 dict(tau=40.0, alpha=1.03): d=(pf-lk)*(1-exp(-mds/tau))*alpha。None=生値
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE   = torch.float32
CHUNK   = 60

def _detect_data():
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"]:
        if os.path.exists(_p): return Path(_p)
    _h = _glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0]))
DATA = _detect_data(); TESTD = DATA / "test"
print("DATA:", DATA, "| device:", DEVICE, "| sets:", len(PARAM_SETS))'''

c2 = '''# ===== PFエンジン(pf_engine_v65.py から抽出した純計算部) =====
''' + CORE

c3 = r'''# ===== test側キャッシュ + 構造prior(錨セットがある場合のみ) =====
NEED_ANCHOR = any(bool(P.get("use_anchor", False)) for P in PARAM_SETS)
STRUCT_T = {}
if NEED_ANCHOR:
    _fz = next((p for p in ([str(Path("/kaggle/input")) and q for q in _glob.glob("/kaggle/input/**/struct_field_v38.npz", recursive=True)]
                            + [r"C:\Users\kosaka256\Documents\rogii_claude\v64\artifacts_v64\struct_field_v38.npz"]) if os.path.exists(p)), None)
    if _fz is None:
        print("★警告: struct_field_v38.npz が見つからない → 錨セットは錨OFFで実行")
    else:
        FZ = np.load(_fz)
        FLD = FZ["field"]; KAP = FZ["kappa"]; CEN = FZ["centers"]; SGC = FZ["sigs"]
        TH = float(FZ["theta0"]); K = int(FZ["K"]); KK = int(FZ["knn_k"]); KH = float(FZ["knn_h"])
        GT = float(FZ["gate"]); KB = FZ["kbins"]; SMIN = float(FZ["sig_min"]); SMAX = float(FZ["sig_max"])
        for fp in _glob.glob(str(TESTD / "*__horizontal_well.csv")):
            wid = os.path.basename(fp).split("__")[0]
            try:
                df = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
                X = df["X"].to_numpy(); Y = df["Y"].to_numpy(); Z = df["Z"].to_numpy(); ti = df["TVT_input"].to_numpy()
                fin = np.where(np.isfinite(ti))[0]
                if len(fin) == 0: continue
                s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
                if n < 2: continue
                edges = np.linspace(0, n, K + 1)
                segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
                mid = np.empty((K, 2)); proj = np.empty(K)
                for j in range(K):
                    f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
                    az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
                    mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2); proj[j] = np.cos(np.radians(az - TH))
                Dh = np.empty(K); dd = np.empty(K)
                for j in range(K):
                    d2 = (FLD[:, 0] - mid[j, 0]) ** 2 + (FLD[:, 1] - mid[j, 1]) ** 2
                    sel = np.argpartition(d2, KK)[:KK]
                    wv = np.exp(np.maximum(-d2[sel] / (2 * KH ** 2), -700))
                    Dh[j] = np.sum(wv * FLD[sel, 2]) / np.sum(wv); dd[j] = np.sqrt(np.median(d2[sel]))
                gg = ndz + (Dh * proj)[segid]; gg = np.where(np.abs(proj[segid]) < GT, 0.0, gg)
                kstep = KAP[np.digitize(dd, KB[1:-1])][segid]
                tvt = float(ti[s]) + np.cumsum(kstep * gg)
                sig = np.clip(np.interp(dd[segid], CEN, SGC), SMIN, SMAX)
                STRUCT_T[wid] = dict(tvt=tvt.astype(np.float64), sig=sig.astype(np.float64))
            except Exception:
                continue
        print(f"[struct] test実行時構築: {len(STRUCT_T)} wells")

# --- self/nbr 用: train近傍プール(nbrセットがある場合のみ構築。test近傍=train坑井, ≤1500ft, ≤3本, 2本未満=tw) ---
MAX_DIST = 1500.0; K_MAX = 3; NEED_REFS = 2
NEED_NBR = any("nbr" in str(P.get("tw_source", "tw")) for P in PARAM_SETS)
TRAIND = DATA / "train"
_TRC = None; _REFC = {}
def _train_centroids():
    global _TRC
    if _TRC is None:
        wids = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TRAIND / "*__horizontal_well.csv"))))
        xy = []
        for w_ in wids:
            h_ = pd.read_csv(TRAIND / f"{w_}__horizontal_well.csv", usecols=["X", "Y"])
            xy.append((float(h_["X"].mean()), float(h_["Y"].mean())))
        _TRC = (wids, np.array(xy))
        print(f"[nbr] trainプール: {len(wids)} wells")
    return _TRC
def _ref_pairs(w_):
    if w_ not in _REFC:
        h_ = pd.read_csv(TRAIND / f"{w_}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        m_ = h_["TVT_input"].notna().to_numpy() & np.isfinite(h_["GR"].to_numpy(float))
        _REFC[w_] = (h_["TVT_input"].to_numpy(float)[m_], h_["GR"].to_numpy(float)[m_])
    return _REFC[w_]
def _nbr_pairs_of(cx, cy):
    wids, xy = _train_centroids()
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    order = np.argsort(d)
    refs = [wids[j] for j in order if d[j] <= MAX_DIST][:K_MAX]
    if len(refs) < NEED_REFS: return None
    ts = []; gs_ = []
    for r_ in refs:
        t_, g_ = _ref_pairs(r_); ts.append(t_); gs_.append(g_)
    return (np.concatenate(ts), np.concatenate(gs_))

def build_cache_test(wid):
    """v65 build_cache の test版(TVT不要)。返り値: make_inputs 互換 cache dict + 行index。"""
    hw = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TESTD / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw["TVT_input"].notna(); ev = ~kn
    if kn.sum() < 10 or ev.sum() == 0: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    fin = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[fin], tw_gr[fin]
    if len(tw_tvt) < 3: return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    ev_pos = hw.index.get_indexer(hw[ev].index); kpos = hw.index.get_indexer(hw[kn].index)
    kn_tvtin = hw["TVT_input"].to_numpy(float)[kpos]
    ls = float(kn_tvtin[-1] + hw["Z"].to_numpy(float)[kpos][-1])
    tl = hw[kn].tail(30)
    dt = np.diff(tl["TVT_input"].values); dz = np.diff(tl["Z"].values); dm = np.diff(tl["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    nev = int(ev.sum())
    sp = STRUCT_T.get(wid)
    if sp is not None and len(sp["tvt"]) == nev:
        anc = np.asarray(sp["tvt"], np.float64); ancs = np.asarray(sp["sig"], np.float64)
    else:
        anc = np.zeros(nev); ancs = np.full(nev, 1e9)
    nbr_pairs = _nbr_pairs_of(float(hw["X"].mean()), float(hw["Y"].mean())) if NEED_NBR else None
    return dict(md=hw["MD"].to_numpy(float)[ev_pos], z=hw["Z"].to_numpy(float)[ev_pos],
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos, kn_tvtin=kn_tvtin,
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir,
                anc=anc, ancs=ancs, ev_index=np.where(ev.to_numpy())[0], lk=float(kn_tvtin[-1]),
                nbr_pairs=nbr_pairs)'''

c4 = r'''# ===== 実行: 各セット → seed尤度加重平均 → セット間 AVG 合成 =====
t0 = time.time()
wids = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TESTD / "*__horizontal_well.csv"))))
caches = {}
for w in wids:
    c = build_cache_test(w)
    if c is not None: caches[w] = c
W = sorted(caches.keys())
print(f"test wells: {len(W)}/{len(wids)}")

preds_by_set = []
for si, P0 in enumerate(PARAM_SETS):
    P = dict(P0)
    outs = {}
    for c0_ in range(0, len(W), CHUNK):
        sub = W[c0_:c0_ + CHUNK]
        inps = [make_inputs(caches[w], P) for w in sub]
        res = run_forward_ens(inps, P, DEVICE, SEED, N_SEEDS)
        for j, w in enumerate(sub): outs[w] = np.asarray(res[j], np.float64)
        if DEVICE == "cuda": torch.cuda.empty_cache()
    preds_by_set.append(outs)
    print(f"set{si+1}/{len(PARAM_SETS)} done ({time.time()-t0:.0f}s)", flush=True)

pids = []; pvals = []
for w in W:
    c = caches[w]
    A = np.stack([ps[w] for ps in preds_by_set], 0)            # (S, nev) 各セットのTVT軌跡
    z = np.nanmedian(A, 0) if AVG == "median" else np.nanmean(A, 0)
    if PP:
        mds = np.maximum(c["md"] - c["md"][0] + 1.0, 0.0)
        d = (z - c["lk"]) * (1.0 - np.exp(-mds / float(PP["tau"]))) * float(PP.get("alpha", 1.0))
        z = c["lk"] + d
    pids += [f"{w}_{int(i)}" for i in c["ev_index"]]
    pvals += list(z)
pred_df = pd.DataFrame({"id": pids, "tvt": pvals})
print("predicted rows:", len(pred_df), f"({time.time()-t0:.0f}s)")'''

c5 = r'''# ===== submission(欠け行は lk フォールバック) =====
ss = pd.read_csv(DATA / "sample_submission.csv")
sub = ss[["id"]].merge(pred_df, on="id", how="left")
miss = sub["tvt"].isna()
if miss.any():
    lkmap = {}
    for wid in set(i.rsplit("_", 1)[0] for i in sub.loc[miss, "id"]):
        h = pd.read_csv(TESTD / f"{wid}__horizontal_well.csv", usecols=["TVT_input"])
        v = h["TVT_input"].dropna()
        lkmap[wid] = float(v.iloc[-1]) if len(v) else 0.0
    sub.loc[miss, "tvt"] = [lkmap[i.rsplit("_", 1)[0]] for i in sub.loc[miss, "id"]]
    print(f"lkフォールバック: {int(miss.sum())} rows")
sub.to_csv("submission.csv", index=False)
print("submission.csv:", len(sub), "rows")'''

nb = dict(nbformat=4, nbformat_minor=5,
          metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}},
          cells=[md(c0), code(c1), code(c2), code(c3), code(c4), code(c5)])
json.dump(nb, open("v65/submit_pf_v65.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written v65/submit_pf_v65.ipynb")
