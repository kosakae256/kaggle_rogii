"""
v20 「どのPFを信用するか」信号の総当たり比較
=============================================
GR尤度と直交しうる信号で per-well 選択し、overall RMSE を比較する。

信号(各 PF×well のスコア, 小さいほど信用):
  A knowntail : 既知ゾーン末尾Lftで各PFを走らせた誤差(ラベル付き答え合わせ)
  B krige     : 各PFの geo_z(=TVT+Z) と 近傍坑井からの kriging構造面 との距離
  C consensus : 各PF予測の(中央値からの)距離
  D dipdev    : 各PFの評価dip と 既知dip の差
  E rough     : 各PFの geo_z 粗さ(2階差RMS)
比較: 各信号で per-well 最良PFを選択 / META(LightGBMで全信号結合) / mean / oracle。

PFコア・評価は iterative_pf_boost_v19 を再利用。重い部分(eval残差/knowntail)はキャッシュ。
実行: rogii_claude/.venv/Scripts/python.exe v20/signal_compare_v20.py
"""

import os, sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.append(str(PROJECT / "v19"))
import iterative_pf_boost_v19 as pf
sys.path.append(str(PROJECT / "v9"))

# ============================== CONFIG ==============================
DATA_DIR   = PROJECT / "rogii-wellbore-geology-prediction" / "train"
BANK_PATH  = PROJECT / "v19" / "artifacts_v19" / "pf_bank.json"
OUT_DIR    = PROJECT / "v20" / "artifacts_v20"; OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE      = OUT_DIR / "signal_cache.pkl"
INCLUDE_V9 = True
N_SEEDS    = 32
CHUNK      = 200
L_KNOWN    = 200          # 既知末尾の検証長
KRIGE_STRIDE = 20; KRIGE_K = 48
N_WELLS    = None
# ===================================================================


def items_bank():
    out = []
    for e in json.loads(BANK_PATH.read_text()):
        out.append((f"r{e['round']}", e.get("seed", 0), e["params"]))
    if INCLUDE_V9:
        from pf_gpu_run import PF_PARAM_SETS_RAW
        for name, raw in PF_PARAM_SETS_RAW.items():
            r = dict(raw); r["use_phys"] = False; r.setdefault("phys_sig", 1.0)
            out.append((f"v9_{name}", 0, r))
    return out


def load_full(wid):
    hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns:
        return None
    kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
    if ev.sum() == 0 or kn.sum() < L_KNOWN + 20:
        return None
    md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float)
    x = hw["X"].to_numpy(float); y = hw["Y"].to_numpy(float)
    tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
    return dict(wid=wid, md=md, z=z, x=x, y=y, tvt=tvt, tin=tin, kn=kn, ev=ev,
                tw_tvt=tw["TVT"].to_numpy(float), tw_gr=tw["GR"].to_numpy(float))


def eval_cache(w):
    """pf.make_inputs 互換(評価ゾーン)。"""
    hw = pd.read_csv(DATA_DIR / f"{w['wid']}__horizontal_well.csv")
    return pf.build_cache(w["wid"])   # pf.build_cache は評価ゾーン用 cache を返す


def knowntail_cache(w):
    """既知末尾 L_KNOWN を「検証ゾーン」にした pf.make_inputs 互換 cache。"""
    hw = pd.read_csv(DATA_DIR / f"{w['wid']}__horizontal_well.csv")
    tw = pd.read_csv(DATA_DIR / f"{w['wid']}__typewell.csv").sort_values("TVT")
    ki = np.where(w["kn"])[0]
    anc = ki[:-L_KNOWN]; val = ki[-L_KNOWN:]
    if anc.size < 10:
        return None
    twv = w["tw_tvt"]; twg = w["tw_gr"]
    tw_fb = float(np.nanmean(twg)) if np.isfinite(np.nanmean(twg)) else 0.
    gr_filled = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    g = w["tin"] + w["z"]
    tail = anc[-30:]; dm = np.diff(w["md"][tail]); dg = np.diff(g[tail]); ok = dm > 0
    ir = float(np.median(dg[ok] / dm[ok])) if ok.sum() >= 3 else 0.
    return dict(md=w["md"][val], z=w["z"][val], gr_filled=gr_filled,
                ev_pos=val, kpos=anc, kn_tvtin=w["tin"][anc],
                tw_tvt=twv, tw_gr=twg, tw_fb=tw_fb,
                ls=float(w["tin"][anc[-1]] + w["z"][anc[-1]]), ir=ir,
                tvt_true=w["tin"][val])


def run_pf_preds(cache_by_w, wids, params, seed):
    """各wellの予測軌跡(リスト)を返す。pf.make_inputs + run_forward_ens(チャンク)。"""
    preds = {}
    for c0 in range(0, len(wids), CHUNK):
        sub = wids[c0:c0 + CHUNK]
        inps = [pf.make_inputs(cache_by_w[w], params) for w in sub]
        outs = pf.run_forward_ens(inps, params, pf.DEVICE, seed, N_SEEDS)
        for w, pr in zip(sub, outs):
            preds[w] = np.asarray(pr, float)
    return preds


def krige_predict(coords_tr, g_tr, q, k=KRIGE_K, reg=1e-2):
    tree = cKDTree(coords_tr); k = min(k, len(coords_tr))
    dist, idx = tree.query(q, k=k)
    nb = coords_tr[idx]; gv = g_tr[idx]
    ls = dist.mean(1, keepdims=True) + 1e-6
    wgt = np.exp(-(dist ** 2) / (2 * ls ** 2)) + 1e-12
    dx = (nb[..., 0] - q[:, None, 0]) / ls; dy = (nb[..., 1] - q[:, None, 1]) / ls
    A = np.stack([np.ones_like(dx), dx, dy], -1); WA = A * wgt[..., None]
    AtA = np.einsum('qki,qkj->qij', WA, A); tr = np.einsum('qii->q', AtA)[:, None, None] / 3
    AtA = AtA + reg * tr * np.eye(3)[None]; Atg = np.einsum('qki,qk->qi', WA, gv)
    coef = np.linalg.solve(AtA, Atg[..., None])[..., 0]
    pred = coef[:, 0]
    lo = gv.min(1); hi = gv.max(1); rng = hi - lo
    return np.clip(pred, lo - 0.5 * rng, hi + 0.5 * rng)


def main():
    print(f"device={pf.DEVICE}")
    items = items_bank(); labels = [it[0] for it in items]; K = len(items)
    print(f"PF {K} 本: {labels}")

    wids0 = pf.list_wells()
    if N_WELLS: wids0 = wids0[:N_WELLS]
    WELLS = {}
    for wid in wids0:
        w = load_full(wid)
        if w is not None and np.isfinite(w["tvt"][w["ev"]]).any():
            WELLS[wid] = w
    WID = list(WELLS)
    print(f"usable wells: {len(WID)}")

    pf.CACHE = {w: pf.build_cache(w) for w in WID}            # 評価ゾーン cache(pf.make_inputs用)
    EVAL_C = pf.CACHE
    KT_C = {w: knowntail_cache(WELLS[w]) for w in WID}
    KT_WID = [w for w in WID if KT_C[w] is not None]

    # ---- キャッシュ ----
    store = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}

    # ---- 各PFの 評価予測軌跡 と knowntail予測 ----
    EVAL_PRED = {}; KT_RMSE = np.full((K, len(WID)), 1e3)
    for ki, (lab, seed, params) in enumerate(items):
        ek = f"evalpred_{lab}_n{N_SEEDS}"
        if ek not in store:
            t0 = time.time(); store[ek] = run_pf_preds(EVAL_C, WID, params, seed)
            CACHE.write_bytes(pickle.dumps(store)); print(f"  evalpred {lab} ({time.time()-t0:.0f}s)", flush=True)
        EVAL_PRED[lab] = store[ek]
        kk = f"kt_{lab}_n{N_SEEDS}_L{L_KNOWN}"
        if kk not in store:
            t0 = time.time()
            ktp = run_pf_preds(KT_C, KT_WID, params, seed)
            kr = {}
            for w in KT_WID:
                e = ktp[w] - KT_C[w]["tvt_true"]; e = e[np.isfinite(e)]
                kr[w] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3
            store[kk] = kr; CACHE.write_bytes(pickle.dumps(store)); print(f"  knowntail {lab} ({time.time()-t0:.0f}s)", flush=True)
        for wi, w in enumerate(WID):
            KT_RMSE[ki, wi] = store[kk].get(w, 1e3)

    # ---- 評価: 各PFの per-well 真値RMSE & 残差 ----
    true_ev = {w: WELLS[w]["tvt"][WELLS[w]["ev"]] for w in WID}
    z_ev = {w: WELLS[w]["z"][WELLS[w]["ev"]] for w in WID}
    EVAL_RMSE = np.full((K, len(WID)), 1e3)
    for ki, lab in enumerate(labels):
        for wi, w in enumerate(WID):
            e = EVAL_PRED[lab][w] - true_ev[w]; e = e[np.isfinite(e)]
            EVAL_RMSE[ki, wi] = float(np.sqrt(np.mean(e ** 2))) if e.size else 1e3

    # ---- kriging構造面(評価X,Yで, 自坑井除外) ----
    Xc, Yc, Gc, Wc = [], [], [], []
    for wi, w in enumerate(WID):
        ww = WELLS[w]; geo = (ww["tvt"] + ww["z"])
        m = np.isfinite(geo) & np.isfinite(ww["x"]) & np.isfinite(ww["y"])
        ii = np.where(m)[0][::KRIGE_STRIDE]
        Xc.append(ww["x"][ii]); Yc.append(ww["y"][ii]); Gc.append(geo[ii]); Wc.append(np.full(ii.size, wi))
    Xc = np.concatenate(Xc); Yc = np.concatenate(Yc); Gc = np.concatenate(Gc); Wc = np.concatenate(Wc)
    COORDS = np.stack([Xc, Yc], 1)
    GEOKR = {}
    for wi, w in enumerate(WID):
        ww = WELLS[w]; ev = ww["ev"]
        q = np.stack([ww["x"][ev], ww["y"][ev]], 1)
        sub = Wc != wi
        GEOKR[w] = krige_predict(COORDS[sub], Gc[sub], q)

    # ---- 信号行列 [K, W] (小さいほど信用) ----
    def md_ev(w): return WELLS[w]["md"][WELLS[w]["ev"]]
    KNOWN_DIP = {}
    for w in WID:
        ww = WELLS[w]; ki = np.where(ww["kn"])[0]; g = ww["tin"] + ww["z"]
        tl = ki[-30:]; dm = np.diff(ww["md"][tl]); dg = np.diff(g[tl]); ok = dm > 0
        KNOWN_DIP[w] = float(np.median(dg[ok] / dm[ok])) if ok.sum() >= 3 else 0.0

    A = KT_RMSE.copy()
    B = np.full((K, len(WID)), 1e3); C = B.copy(); D = B.copy(); E = B.copy()
    for wi, w in enumerate(WID):
        mds = md_ev(w); ze = z_ev[w]
        allpred = np.stack([EVAL_PRED[lab][w] for lab in labels], 0)   # [K, T]
        med = np.median(allpred, 0)
        for ki, lab in enumerate(labels):
            p = EVAL_PRED[lab][w]; geo = p + ze
            B[ki, wi] = float(np.mean(np.abs(geo - GEOKR[w])))
            C[ki, wi] = float(np.mean(np.abs(p - med)))
            gd = np.gradient(geo, mds)
            D[ki, wi] = abs(float(np.mean(gd)) - KNOWN_DIP[w])
            E[ki, wi] = float(np.sqrt(np.mean(np.diff(geo, 2) ** 2))) if len(geo) > 2 else 0.0

    # ---- 評価関数: 信号で per-well 選択 → overall RMSE ----
    def overall_from_choice(choice):
        errs = []
        for wi, w in enumerate(WID):
            p = EVAL_PRED[labels[choice[wi]]][w]; e = p - true_ev[w]; errs.append(e[np.isfinite(e)])
        e = np.concatenate(errs); return float(np.sqrt(np.mean(e ** 2)))

    def sel_by(S): return S.argmin(0)
    # baseline
    mean_err = []
    for w in WID:
        m = np.mean(np.stack([EVAL_PRED[lab][w] for lab in labels], 0), 0) - true_ev[w]
        mean_err.append(m[np.isfinite(m)])
    MEAN = float(np.sqrt(np.mean(np.concatenate(mean_err) ** 2)))
    ORACLE = overall_from_choice(sel_by(EVAL_RMSE))

    print("\n=== per-well 選択の overall RMSE (raw TVT) ===")
    print(f"  mean-ensemble   : {MEAN:.3f}")
    for name, S in [("A knowntail", A), ("B krige", B), ("C consensus", C), ("D dipdev", D), ("E rough", E)]:
        print(f"  select by {name:12s}: {overall_from_choice(sel_by(S)):.3f}")
    print(f"  oracle(真値)    : {ORACLE:.3f}")

    # ---- META: 全信号 + パラメータ で well単位 KFold 学習 ----
    try:
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import KFold
        feats = []; tgt = []
        PKEYS = ["likelihood_scale", "gr_power", "init_pos_std", "mom", "resamp", "phys_sig"]
        for ki, (lab, seed, params) in enumerate(items):
            for wi, w in enumerate(WID):
                row = [A[ki, wi], B[ki, wi], C[ki, wi], D[ki, wi], E[ki, wi]]
                row += [float(params.get(k_, 0.0)) for k_ in PKEYS]
                feats.append(row); tgt.append(EVAL_RMSE[ki, wi])
        Xm = np.array(feats, np.float32); ym = np.array(tgt, np.float32)
        widx = np.tile(np.arange(len(WID)), K)
        pred = np.zeros(len(ym))
        for tr, va in KFold(5, shuffle=True, random_state=0).split(np.arange(len(WID))):
            trm = np.isin(widx, tr); vam = np.isin(widx, va)
            m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
            m.fit(Xm[trm], ym[trm]); pred[vam] = m.predict(Xm[vam])
        score = pred.reshape(K, len(WID))
        print(f"  META(LightGBM)  : {overall_from_choice(score.argmin(0)):.3f}")
    except Exception as e:
        print("  META skip:", e)

    print("\n注: raw TVT(apply_pp前)。mean を切る信号があれば selection に脈あり。")


if __name__ == "__main__":
    main()
