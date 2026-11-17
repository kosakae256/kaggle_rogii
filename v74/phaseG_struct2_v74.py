# -*- coding: utf-8 -*-
"""v74 Phase G — struct設計を参考にゼロから作り替えたdip場エンジン(GRフリー)
   参考: v66 build_struct の設計(セグメント・ランプフィット/θ0拘束+条件数適応2D場/
         α·(−ΔZ)+β·field の累積目標LS/PGATE)。既存の成果物(pkl/npz)は不使用。
   元より厳密な点: フィールドと係数(α,β)を GroupKFold の train fold のみで構築・学習
   (元structは全wellで係数フィット=軽い楽観)。
   出力: 井戸ごと {pred(eval行TVT), rates, dd, spread} → phaseG_dump.npz + engine overall
   env: NW(0=全), SEG_LEN(150), KNN_K(15), KNN_H(500), THETA0(118.4), PGATE(0.35)
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
SEG_LEN = float(os.environ.get("SEG_LEN", "150"))
KNN_K = int(os.environ.get("KNN_K", "15"))
KNN_H = float(os.environ.get("KNN_H", "500"))
THETA0 = float(os.environ.get("THETA0", "118.4"))
PGATE = float(os.environ.get("PGATE", "0.35"))
VEC = int(os.environ.get("VEC", "2"))
QW = os.environ.get("QW", "1") == "1"
ROBUST = os.environ.get("ROBUST", "1") == "1"
STEREO = os.environ.get("STEREO", "0") == "1"   # ペア勾配ドナー(v66 2-1の作り替え)
PAIR_RMAX = 1200.0
PAIR_DMIN = 60.0
PAIR_CENT = 2500.0
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
if os.environ.get("REGIMES_EXT", "0") == "1":
    KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0, 3000.0, 4500.0]   # 疎ドナー環境の模擬を強化
NBIN = len(KBINS) - 1
CMAX = 0.30
HUBER_K = 1.345
IRLS_IT = 2
K_SEG_MAX = 64
_C0, _S0 = np.cos(np.radians(THETA0)), np.sin(np.radians(THETA0))

def load_well(fp):
    w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input", "ANCC"])
    ti = w["TVT_input"].to_numpy(float)
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) == 0:
        return None
    s = int(fin.max())
    X, Y, Z = w["X"].to_numpy(float), w["Y"].to_numpy(float), w["Z"].to_numpy(float)
    n = len(X) - 1 - s
    if n < 2:
        return None
    dX = np.diff(X)[s:]; dY = np.diff(Y)[s:]
    ndz = -np.diff(Z)[s:]
    Lxy = np.maximum(np.hypot(dX, dY), 1e-3)
    p_row = (dX * _C0 + dY * _S0) / Lxy
    tvt = w["TVT"].to_numpy(float)
    d = dict(s=s, n=n, dX=dX, dY=dY, ndz=ndz, Lxy=Lxy, arc=np.cumsum(Lxy),
             Xl=X[s:], Yl=Y[s:], anchor=float(ti[s]),
             perp=(np.abs(p_row) < PGATE) if PGATE > 0 else np.zeros(n, bool),
             tvt_tail=tvt[s + 1:], ev_tail=~np.isfinite(ti[s + 1:]))
    d["R0"] = tvt[s + 1:] - tvt[s]
    if STEREO:
        anc = w["ANCC"].to_numpy(float)
        dhf = np.r_[1.0, np.hypot(np.diff(X), np.diff(Y))]
        m = np.isfinite(anc) & (dhf > 0.5)
        anc_s = pd.Series(np.where(np.isfinite(anc), anc, np.nan)).rolling(
            201, center=True, min_periods=1).mean().to_numpy()
        li = np.where(m & np.isfinite(anc_s))[0][::30]
        d["lat_xy"] = np.c_[X[li], Y[li]]; d["lat_anc"] = anc_s[li]
    return d

def segment_well(wd):
    n, A = wd["n"], wd["arc"]; total = float(A[-1])
    K = int(np.clip(round(total / SEG_LEN), 1, min(K_SEG_MAX, max(1, n // 24)))) \
        if SEG_LEN > 0 else min(16, max(1, n // 4))
    edges = np.linspace(0.0, total, K + 1)
    phi = np.column_stack([np.clip(A - edges[j], 0.0, edges[j + 1] - edges[j])
                           for j in range(K)])
    y = wd["R0"] - np.cumsum(wd["ndz"])                  # 累積Δg(ANCC変化)
    m = np.isfinite(y)
    if m.sum() < K + 2:
        return None
    c, res, rank, _ = np.linalg.lstsq(phi[m], y[m], rcond=None)
    dof = max(int(m.sum()) - K, 1)
    ssr = float(res[0]) if len(res) else float(np.sum((phi[m] @ c - y[m]) ** 2))
    cov = (ssr / dof) * np.linalg.pinv(phi[m].T @ phi[m])
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    segid = np.clip(np.searchsorted(edges[1:], A, side="left"), 0, K - 1)
    mid = np.empty((K, 2)); az = np.empty(K)
    Xl, Yl = wd["Xl"], wd["Yl"]
    for j in range(K):
        rows = np.where(segid == j)[0]
        j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)
        az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0])
        mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)
    return segid, mid, az, c, se

def build_field(wells, use_wi):
    """use_wi の井戸のセグメントだけで場を構築(fold-honest)"""
    px, py, pc, paz, pse = [], [], [], [], []
    pwi = []
    for wd in wells:
        if wd["wi"] not in use_wi or wd.get("seg") is None:
            continue
        segid, mid, az, c, se = wd["seg"]
        ok = np.isfinite(c) & (np.abs(c) <= CMAX) & np.isfinite(se)
        px.append(mid[ok, 0]); py.append(mid[ok, 1]); pc.append(c[ok])
        paz.append(az[ok]); pse.append(se[ok])
        pwi.append(np.full(int(ok.sum()), wd["wi"], float))
    F = dict(x=np.concatenate(px), y=np.concatenate(py), c=np.concatenate(pc),
             az=np.concatenate(paz), wi=np.concatenate(pwi), se=np.concatenate(pse))
    F["wi2"] = F["wi"].copy()
    if QW:
        med2 = float(np.median(F["se"])) ** 2
        F["qw"] = 1.0 / (F["se"] ** 2 + med2 + 1e-12)
        F["qw"] /= np.median(F["qw"])
    else:
        F["qw"] = np.ones(len(F["x"]))
    F["ux"], F["uy"] = np.cos(F["az"]), np.sin(F["az"])
    return F

def add_pairs(F, wells):
    """[STEREO] クロスウェルペア勾配ドナー: c=(ΔANCC)/dist をペア方位の方向微分として追加。
    重み: σ_c = datumノイズ/dist の理論形 → qw ∝ dist²(cap)。追加チューニングなし。"""
    from scipy.spatial import cKDTree
    cents = {wd["wi"]: (float(np.mean(wd["Xl"])), float(np.mean(wd["Yl"]))) for wd in wells}
    trees = {wd["wi"]: cKDTree(wd["lat_xy"]) for wd in wells if "lat_xy" in wd and len(wd["lat_xy"]) > 5}
    P = dict(x=[], y=[], c=[], az=[], qw=[], wi=[], wi2=[])
    ws2 = [wd for wd in wells if wd["wi"] in trees]
    for i, wa in enumerate(ws2):
        ca = cents[wa["wi"]]
        for wb in ws2[i + 1:]:
            cb = cents[wb["wi"]]
            if np.hypot(ca[0] - cb[0], ca[1] - cb[1]) > PAIR_CENT:
                continue
            dq, jj = trees[wb["wi"]].query(wa["lat_xy"][::3], distance_upper_bound=PAIR_RMAX)
            ok = np.isfinite(dq) & (dq > PAIR_DMIN)
            if ok.sum() < 3:
                continue
            ia = np.where(ok)[0] * 3
            jb = np.asarray(jj)[ok]
            xa = wa["lat_xy"][ia]; xb = wb["lat_xy"][jb]
            cp = (wb["lat_anc"][jb] - wa["lat_anc"][ia]) / dq[ok]
            good = np.isfinite(cp) & (np.abs(cp) <= CMAX)
            if good.sum() < 3:
                continue
            P["x"].append(((xa[:, 0] + xb[:, 0]) / 2)[good])
            P["y"].append(((xa[:, 1] + xb[:, 1]) / 2)[good])
            P["c"].append(cp[good])
            P["az"].append(np.arctan2(xb[:, 1] - xa[:, 1], xb[:, 0] - xa[:, 0])[good])
            P["qw"].append(np.clip((dq[ok][good] / 400.0) ** 2, 0.05, 1.5))
            P["wi"].append(np.full(good.sum(), wa["wi"], float))
            P["wi2"].append(np.full(good.sum(), wb["wi"], float))
    if not P["x"]:
        return F
    npair = sum(len(v) for v in P["x"])
    for k in ("x", "y", "c", "az", "qw", "wi", "wi2"):
        P[k] = np.concatenate(P[k])
    F2 = dict(x=np.r_[F["x"], P["x"]], y=np.r_[F["y"], P["y"]],
              c=np.r_[F["c"], P["c"]], az=np.r_[F["az"], P["az"]],
              wi=np.r_[F["wi"], P["wi"]], wi2=np.r_[F["wi2"], P["wi2"]],
              qw=np.r_[F["qw"], P["qw"]])
    F2["ux"], F2["uy"] = np.cos(F2["az"]), np.sin(F2["az"])
    print(f"stereo pair donors: +{npair}  (|c|med={np.median(np.abs(P['c'])):.4f})", flush=True)
    return F2

def query_well(F, mids, own, mdbuf=0.0):
    keep = (F["wi"] != own) & (F["wi2"] != own)
    fx, fy = F["x"][keep], F["y"][keep]
    fc, fqw = F["c"][keep], F["qw"][keep]
    fux, fuy = F["ux"][keep], F["uy"][keep]
    fp0 = fux * _C0 + fuy * _S0
    dx = fx[None, :] - mids[:, 0:1]; dy = fy[None, :] - mids[:, 1:2]
    d2 = dx * dx + dy * dy
    if mdbuf > 0:
        d2 = np.where(d2 >= mdbuf * mdbuf, d2, np.inf)
    kk = min(KNN_K, d2.shape[1] - 1) if d2.shape[1] > 1 else 1
    idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    r = np.arange(len(mids))[:, None]
    d2s = d2[r, idx]
    c_s, p0_s = fc[idx], fp0[idx]
    ux_s, uy_s = fux[idx], fuy[idx]
    w = np.exp(np.maximum(-d2s / (2 * KNN_H ** 2), -700.0)) * fqw[idx]
    w[~np.isfinite(d2s)] = 0.0
    wsum = w.sum(1)
    dead = wsum <= 1e-300
    with np.errstate(invalid="ignore"):
        dd = np.sqrt(np.nanmedian(np.where(np.isfinite(d2s), d2s, np.nan), axis=1))
    dd[~np.isfinite(dd)] = KBINS[-2] * 2
    # θ0拘束回帰 + Huber IRLS
    s_hat = (w * c_s * p0_s).sum(1) / ((w * p0_s ** 2).sum(1) + 1e-9)
    with np.errstate(invalid="ignore"):
        for _ in range(IRLS_IT if ROBUST else 0):
            rres = np.where(w > 0, c_s - s_hat[:, None] * p0_s, np.nan)
            med = np.nanmedian(rres, axis=1, keepdims=True)
            scale = 1.4826 * np.nanmedian(np.abs(rres - med), axis=1, keepdims=True) + 1e-6
            hub = np.minimum(1.0, HUBER_K * scale / np.maximum(np.abs(rres - med), 1e-12))
            hub = np.nan_to_num(hub, nan=1.0, posinf=1.0)
            w = w * hub
            s_hat = (w * c_s * p0_s).sum(1) / ((w * p0_s ** 2).sum(1) + 1e-9)
    g = np.column_stack([s_hat * _C0, s_hat * _S0])
    if VEC < 2:
        g[dead] = 0.0
        gn = np.hypot(g[:, 0], g[:, 1])
        big = gn > CMAX
        if big.any():
            g[big] *= (CMAX / gn[big])[:, None]
        pred_don = g[:, 0:1] * ux_s + g[:, 1:2] * uy_s
        spread = np.sqrt(np.maximum((w * (c_s - pred_don) ** 2).sum(1) /
                                    np.maximum(w.sum(1), 1e-12), 0.0))
        spread[dead] = 0.0
        return g, dd, spread
    # 2Dベクトル場(条件数適応でθ0解へ正則化)
    Sxx = (w * ux_s * ux_s).sum(1); Sxy = (w * ux_s * uy_s).sum(1)
    Syy = (w * uy_s * uy_s).sum(1)
    bx = (w * c_s * ux_s).sum(1); by = (w * c_s * uy_s).sum(1)
    tr = Sxx + Syy; det = Sxx * Syy - Sxy * Sxy
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
    lam1 = (tr + disc) / 2; lam2 = (tr - disc) / 2
    ratio = lam2 / np.maximum(lam1, 1e-12)
    lam = lam1 * np.maximum(0.05, 0.25 - ratio)
    A11 = Sxx + lam; A22 = Syy + lam; A12 = Sxy
    b1 = bx + lam * g[:, 0]; b2 = by + lam * g[:, 1]
    detA = A11 * A22 - A12 * A12
    gx = (b1 * A22 - b2 * A12) / np.maximum(detA, 1e-12)
    gy = (b2 * A11 - b1 * A12) / np.maximum(detA, 1e-12)
    g = np.column_stack([gx, gy])
    g[dead] = 0.0
    gn = np.hypot(g[:, 0], g[:, 1])
    big = gn > CMAX
    if big.any():
        g[big] *= (CMAX / gn[big])[:, None]
    pred_don = g[:, 0:1] * ux_s + g[:, 1:2] * uy_s
    spread = np.sqrt(np.maximum((w * (c_s - pred_don) ** 2).sum(1) /
                                np.maximum(wsum, 1e-12), 0.0))
    spread[dead] = 0.0
    return g, dd, spread

def field_rows(wd, g):
    sid = wd["seg"][0]
    return g[sid, 0] * wd["dX"] + g[sid, 1] * wd["dY"]

def design_cols(wd, F):
    """κ推定/予測共通: buffered込みの列を作る素材"""
    pass

def flip_gdict(wd):
    """逆方向DA: 井戸を反転した擬似井戸(anchor=元toe側, trainのみ=R0必須)"""
    if wd.get("seg") is None or "R0" not in wd:
        return None
    n = wd["n"]
    tvt_rev = wd["tvt_tail"][::-1]          # 元toe→元anchor方向
    if not np.isfinite(tvt_rev[0]):
        fin = np.where(np.isfinite(tvt_rev))[0]
        if len(fin) == 0:
            return None
        a0 = float(tvt_rev[fin[0]])
    else:
        a0 = float(tvt_rev[0])
    gd = dict(n=n, dX=-wd["dX"][::-1], dY=-wd["dY"][::-1], ndz=-wd["ndz"][::-1],
              Lxy=wd["Lxy"][::-1], arc=np.cumsum(wd["Lxy"][::-1]),
              Xl=wd["Xl"][::-1], Yl=wd["Yl"][::-1],
              perp=wd["perp"][::-1], wi=wd["wi"],
              R0=np.r_[tvt_rev[1:], wd["anchor"]] - a0)
    gd["seg"] = segment_well(gd)
    return gd

def fit_kappa(F, wells, use_wi):
    npar = 2 * NBIN
    nperp = 2 if PGATE > 0 else 0
    ncol = npar + nperp
    A = np.zeros((ncol, ncol)); yv = np.zeros(ncol)
    fit_list = [wd for wd in wells if wd["wi"] in use_wi and wd.get("seg") is not None]
    if os.environ.get("BACK_DA", "0") == "1":
        flips = [flip_gdict(wd) for wd in fit_list]
        fit_list = fit_list + [g_ for g_ in flips if g_ is not None and g_["seg"] is not None]
    for R in KAPPA_REGIMES:
        for wd in fit_list:
            g, dd, _ = query_well(F, wd["seg"][1], wd["wi"], mdbuf=R)
            f = field_rows(wd, g)
            a = wd["ndz"]
            par = ~wd["perp"]
            b_row = np.digitize(dd, KBINS[1:-1])[wd["seg"][0]]
            cols = []
            for b in range(NBIN):
                m = (b_row == b) & par
                cols.append(np.cumsum(np.where(m, a, 0.0)))
                cols.append(np.cumsum(np.where(m, f, 0.0)))
            if nperp:
                cols.append(np.cumsum(np.where(wd["perp"], a, 0.0)))
                cols.append(np.cumsum(np.where(wd["perp"], f, 0.0)))
            G = np.column_stack(cols)
            ok = np.isfinite(wd["R0"])
            G = np.nan_to_num(G[ok]); y = wd["R0"][ok]
            A += G.T @ G; yv += G.T @ y
    # perp係数の安定化: v38 hard gate(=0)へのリッジ(fold間分散→特異井戸の積分暴走を防ぐ)
    if nperp:
        lam_p = 0.02 * float(np.trace(A)) / ncol
        A[npar, npar] += lam_p; A[npar + 1, npar + 1] += lam_p
    coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    K = dict(alpha=np.clip(coef[0:npar:2], -0.25, 1.5),
             beta=np.clip(coef[1:npar:2], -0.25, 1.5),
             ap=float(np.clip(coef[npar], -0.25, 1.5)) if nperp else 0.0,
             bp=float(np.clip(coef[npar + 1], -0.25, 1.5)) if nperp else 0.0)
    return K

def predict(F, wd, K):
    g, dd, spread = query_well(F, wd["seg"][1], wd["wi"], mdbuf=0.0)
    f = field_rows(wd, g)
    b_row = np.digitize(dd, KBINS[1:-1])[wd["seg"][0]]
    av = K["alpha"][b_row]; bv = K["beta"][b_row]
    if PGATE > 0:
        av = np.where(wd["perp"], K["ap"], av)
        bv = np.where(wd["perp"], K["bp"], bv)
    rate = av * wd["ndz"] + bv * f
    resid = np.clip(np.cumsum(rate), -150.0, 150.0)       # 物理クリップ(暴走保険)
    return wd["anchor"] + resid, rate, dd[wd["seg"][0]], spread[wd["seg"][0]]

def main():
    t0 = time.time()
    fps = sorted(os.listdir(TRAIN))
    wids = sorted(set(f.split("__")[0] for f in fps if f.endswith("__horizontal_well.csv")))
    if NW:
        wids = wids[:NW]
    wells = []
    for wi, w in enumerate(wids):
        wd = load_well(os.path.join(TRAIN, f"{w}__horizontal_well.csv"))
        if wd is None:
            continue
        wd["wi"] = wi; wd["wid"] = w
        wd["seg"] = segment_well(wd)
        wells.append(wd)
    print(f"wells={len(wells)}  ({time.time()-t0:.0f}s)", flush=True)
    gi = np.arange(len(wells))
    FIELD_OUT = os.environ.get("FOLD_MODE", "well") == "field"
    if FIELD_OUT:
        # leave-field-out: XYクラスタ(連結成分5000ft)=field。場もκもvalのfield全体を除外。
        from scipy.spatial import cKDTree
        cxy = np.array([[float(np.mean(wd["Xl"])), float(np.mean(wd["Yl"]))] for wd in wells])
        tree = cKDTree(cxy)
        parent = list(range(len(wells)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in tree.query_pairs(float(os.environ.get("FIELD_LINK", "5000"))):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        fid = np.array([find(i) for i in range(len(wells))])
        uniq = {f: i for i, f in enumerate(sorted(set(fid.tolist())))}
        groups = np.array([uniq[f] for f in fid])
        print(f"FOLD_MODE=field: fields={len(uniq)} 最大={np.bincount(groups).max()}井", flush=True)
        folds = list(GroupKFold(5).split(gi, gi, groups))
    else:
        folds = list(GroupKFold(5).split(gi, gi, gi))
    # well-fold時: フィールドは全井戸(=デプロイregime, クエリで自坑井のみ除外)
    all_wi = set(wd["wi"] for wd in wells)
    if not FIELD_OUT:
        F = build_field(wells, all_wi)
        if STEREO:
            F = add_pairs(F, wells)
        print(f"field donors={len(F['x'])}", flush=True)
    ES, perw = [], {}
    pack = {}
    for fold, (tr_i, va_i) in enumerate(folds):
        tr_wi = set(wells[i]["wi"] for i in tr_i)
        if FIELD_OUT:
            F = build_field(wells, tr_wi)      # 場からvalのfield全体を除外
        K = fit_kappa(F, wells, tr_wi)
        print(f"fold{fold}: α={np.round(K['alpha'],3)} "
              f"β={np.round(K['beta'],3)} ap={K['ap']:.3f} bp={K['bp']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        for i in va_i:
            wd = wells[i]
            if wd.get("seg") is None:
                continue
            pred, rate, ddr, spr = predict(F, wd, K)
            ev = wd["ev_tail"] & np.isfinite(wd["tvt_tail"])
            e = (pred - wd["tvt_tail"])[ev]
            e = e[np.isfinite(e)]
            if len(e):
                ES.append(e); perw[wd["wid"]] = float(np.sqrt(np.mean(e ** 2)))
            pack[wd["wid"] + "__pred"] = pred.astype(np.float32)
            pack[wd["wid"] + "__rate"] = rate.astype(np.float32)
            pack[wd["wid"] + "__dd"] = ddr.astype(np.float32)
            pack[wd["wid"] + "__spread"] = spr.astype(np.float32)
            pack[wd["wid"] + "__s"] = np.int32(wd["s"])
    es = np.concatenate(ES)
    overall = float(np.sqrt(np.mean(es ** 2)))
    dm = float(np.sqrt(np.mean(np.concatenate([e - np.median(e) for e in ES]) ** 2)))
    print(f"\n[Phase G struct2.0] overall = {overall:.4f}  "
          f"(井戸med {np.median(list(perw.values())):.3f}, demean {dm:.4f})  "
          f"[参考: 既存struct(全体fit) 9.70 / 旧v7エンジン 11.56]")
    np.savez_compressed(os.path.join(ART, "phaseG_dump.npz"), **pack)
    json.dump(dict(overall=overall, demean=dm,
                   per_well_median=float(np.median(list(perw.values()))),
                   per_well=perw),
              open(os.path.join(ART, "phaseG_result.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
