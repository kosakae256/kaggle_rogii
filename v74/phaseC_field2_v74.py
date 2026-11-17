# v74 Phase C — struct 2.0(既存structを参考に、ゼロから大幅改変した空間dip場エンジン)v4
# 変更履歴:
#   v1: 素実装(絶対値リッジ→縮退発散で大破143。ステレオが発散止めになり29)
#   v2: 過剰な信頼度縮小で中央値崩壊(31/med18.7)
#   v3: トレース比例リッジで発散根治(base18.4/gate14.7/med7.75)。stereo微逆・prefcal逆効果
#   v4: σ解像度(v66の勝ち筋): conf = 距離減衰 × 方向拘束度qdir × ドナー不一致MAD。
#       ドナー構築1回で設定グリッド評価。
# 予測: ĝ(t)=g(t0)+Σ(gx·Δx+gy·Δy) → TVT̂=ĝ−Z。GR不使用。ANCCはtrainドナー側のみ(合法)。
# env: NW(0=全), GRID(1=グリッド評価/0=単発), K, RMAX, SIG0, RIDGE(比率0.1), SEG, WG, SHRINK
import os, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
GRID = os.environ.get("GRID", "1") == "1"
K0 = int(os.environ.get("K", "64"))
RMAX0 = float(os.environ.get("RMAX", "3000"))
SIG00 = float(os.environ.get("SIG0", "300"))
RIDGE = float(os.environ.get("RIDGE", "0.1"))
SEG = int(os.environ.get("SEG", "50"))
PAIR_RMAX = float(os.environ.get("PAIR_RMAX", "1200"))
PAIR_CENT = float(os.environ.get("PAIR_CENT", "2500"))
SHRINK = float(os.environ.get("SHRINK", "0.5"))
WG0 = float(os.environ.get("WG", "0.25"))

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def load():
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    D = {}
    for w in wells:
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", "ANCC"])
        md = hw["MD"].to_numpy(float); x = hw["X"].to_numpy(float)
        y = hw["Y"].to_numpy(float); z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        anc = hw["ANCC"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            continue
        dxr = np.gradient(lowpass(x, 101)); dyr = np.gradient(lowpass(y, 101))
        dh = np.hypot(dxr, dyr)
        D[w] = dict(md=md, x=x, y=y, z=z, tvt=tvt, tin=tin, anc=anc,
                    kn=kn, ev=ev, t0=int(np.where(kn)[0][-1]),
                    dxr=dxr, dyr=dyr, dh=dh, zs=lowpass(z, 201))
    return D

def build_donors(D, stride=8, win=201):
    ws = sorted(D.keys())
    P_pos, P_az, P_c, P_w, P_well = [], [], [], [], []
    lat_pts = {}
    for wi, w in enumerate(ws):
        d = D[w]
        anc = d["anc"]
        m = np.isfinite(anc) & (d["dh"] > 0.5)
        anc_s = lowpass(np.where(np.isfinite(anc), anc, np.nan), win)
        c = np.gradient(anc_s) / np.maximum(d["dh"], 1e-6)
        idx = np.where(m & np.isfinite(c) & (np.abs(c) < 0.3))[0][::stride]
        if len(idx) < 5:
            continue
        # ドナー品質: ANCC局所粗さ(平滑残差の窓MAD)が大きい所は信用しない(ターゲット不使用)
        rough = pd.Series(np.abs(anc - anc_s)).rolling(201, center=True,
                                                       min_periods=1).median().to_numpy()
        qual = 1.0 / (1.0 + (np.nan_to_num(rough[idx], nan=5.0) / 3.0) ** 2)
        P_pos.append(np.c_[d["x"][idx], d["y"][idx]])
        P_az.append(np.arctan2(d["dyr"][idx], d["dxr"][idx]))
        P_c.append(c[idx]); P_w.append(qual)
        P_well.append(np.full(len(idx), wi, np.int32))
        lat_pts[w] = (np.where(m)[0][::4], anc_s)
    donors = dict(pos=np.concatenate(P_pos), az=np.concatenate(P_az),
                  c=np.concatenate(P_c), w=np.concatenate(P_w),
                  well=np.concatenate(P_well))
    cent = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in ws])
    trees = {w: cKDTree(np.c_[D[w]["x"][lat_pts[w][0]], D[w]["y"][lat_pts[w][0]]])
             for w in ws if w in lat_pts}
    Q = dict(pos=[], az=[], c=[], w=[], wa=[], wb=[])
    for i, wa in enumerate(ws):
        if wa not in lat_pts:
            continue
        d_a = D[wa]; lia, anc_a = lat_pts[wa]
        for j in range(i + 1, len(ws)):
            wb = ws[j]
            if wb not in lat_pts or np.hypot(*(cent[i] - cent[j])) > PAIR_CENT:
                continue
            lib, anc_b = lat_pts[wb]
            sa = lia[::15]
            dq, jj = trees[wb].query(np.c_[d_a["x"][sa], d_a["y"][sa]],
                                     distance_upper_bound=PAIR_RMAX)
            ok = np.isfinite(dq) & (dq > 60)
            if ok.sum() < 3:
                continue
            sa = sa[ok]; dq = dq[ok]; jb = lib[np.asarray(jj)[ok]]
            db = D[wb]
            dxp = db["x"][jb] - d_a["x"][sa]; dyp = db["y"][jb] - d_a["y"][sa]
            cp = (anc_b[jb] - anc_a[sa]) / dq
            good = np.isfinite(cp) & (np.abs(cp) < 0.3)
            if good.sum() < 3:
                continue
            Q["pos"].append(np.c_[(d_a["x"][sa] + db["x"][jb]) / 2,
                                  (d_a["y"][sa] + db["y"][jb]) / 2][good])
            Q["az"].append(np.arctan2(dyp, dxp)[good])
            Q["c"].append(cp[good])
            Q["w"].append(np.clip((dq[good] / 400.0) ** 2, 0.05, 1.5))
            Q["wa"].append(np.full(good.sum(), i, np.int32))
            Q["wb"].append(np.full(good.sum(), j, np.int32))
    pair = {k: np.concatenate(v) for k, v in Q.items()} if Q["pos"] else None
    return ws, donors, pair

def solve_grad(pts_c, pts_az, wts, ridge_frac, u):
    """加重トレース比例リッジで (gx,gy)。返り値: (g, mad(不一致), qdir(方向拘束度))"""
    A = np.c_[np.cos(pts_az), np.sin(pts_az)]
    AtW = A.T * wts
    M = AtW @ A
    sw = max(float(wts.sum()), 1e-9)
    lam = ridge_frac * (M[0, 0] + M[1, 1]) / 2 + 1e-9
    try:
        g = np.linalg.solve(M + np.diag([lam, lam]), AtW @ pts_c)
    except np.linalg.LinAlgError:
        return np.zeros(2), 1.0, 0.0
    r = pts_c - A @ g
    mad = float(np.average(np.abs(r), weights=wts))
    qdir = float(u @ M @ u) / sw                      # ドナーが自分の方位を拘束する度合[0,1]
    return g, mad, qdir

def run(D, ws, donors, pair, tag, use_stereo=0, use_prefcal=0, use_gate=1,
        k=K0, rmax=RMAX0, sig0=SIG00, wg=WG0, conf2=1, decay=2000.0, ga=0.0,
        seg=SEG, sigcap=1e9, dump=None, ridge=None, eval_wells=None):
    tp = cKDTree(donors["pos"])
    tq = cKDTree(pair["pos"]) if (pair is not None and use_stereo) else None
    n_don = len(donors["pos"]); n_pair = 0 if pair is None else len(pair["pos"])
    rg = RIDGE if ridge is None else ridge
    evset = None if eval_wells is None else set(eval_wells)
    ES, perw = [], {}
    for wi, w in enumerate(ws):
        if evset is not None and w not in evset and dump is None:
            continue
        d = D[w]
        n = len(d["md"]); t0 = d["t0"]
        ei_all = np.arange(t0 + 1, n)
        own_np = int((donors["well"] == wi).sum())
        kq = min(k + own_np + 64, n_don)
        own_pp = int(((pair["wa"] == wi) | (pair["wb"] == wi)).sum()) if pair is not None else 0
        kqp = min(k // 2 + own_pp + 32, max(n_pair, 1))
        s0 = max(t0 - 1500, 0)
        centers = np.arange(s0, n, seg)
        rates = np.zeros(len(centers)); confs = np.zeros(len(centers))
        gxs = np.zeros(len(centers)); gys = np.zeros(len(centers))
        for ci, c0 in enumerate(centers):
            p = np.array([d["x"][c0], d["y"][c0]])
            dd, jj = tp.query(p, k=kq, distance_upper_bound=rmax)
            ok = np.isfinite(dd) & (jj < n_don)
            jj = jj[ok]; dd = dd[ok]
            keep = donors["well"][jj] != wi
            jj = jj[keep][:k]; dd = dd[keep][:k]
            cs, azs, wts = donors["c"][jj], donors["az"][jj], donors["w"][jj]
            sig = min(max(sig0, np.median(dd) if len(dd) else sig0), sigcap)
            wts = wts * np.exp(-0.5 * (dd / sig) ** 2)
            if tq is not None:
                dq, jq = tq.query(p, k=kqp, distance_upper_bound=rmax)
                okq = np.isfinite(dq) & (jq < n_pair)
                jq = jq[okq]; dq = dq[okq]
                keepq = (pair["wa"][jq] != wi) & (pair["wb"][jq] != wi)
                jq = jq[keepq][:k // 2]; dq = dq[keepq][:k // 2]
                cs = np.r_[cs, pair["c"][jq]]
                azs = np.r_[azs, pair["az"][jq]]
                wts = np.r_[wts, pair["w"][jq] * np.exp(-0.5 * (dq / sig) ** 2)]
            if len(cs) < 8:
                rates[ci] = np.nan; confs[ci] = 0.0; continue
            ux, uy = d["dxr"][c0], d["dyr"][c0]
            hh = max(np.hypot(ux, uy), 1e-6)
            u = np.array([ux, uy]) / hh
            (gx, gy), mad, qdir = solve_grad(cs, azs, wts, rg, u)
            rates[ci] = gx * u[0] + gy * u[1]
            gxs[ci] = gx; gys[ci] = gy
            dmed = float(np.median(dd)) if len(dd) else 1e9
            cdist = np.exp(-max(dmed - 800.0, 0.0) / 1000.0)
            if conf2:
                # 閾値型: 通常領域では≈1、明確に悪い時だけ落とす(v4の連続乗算は過縮小で全滅)
                qterm = float(np.clip(qdir / 0.10, 0.0, 1.0))
                madterm = float(np.exp(-max(mad - 0.03, 0.0) / 0.03))
                confs[ci] = cdist * qterm * madterm
            else:
                confs[ci] = cdist
        fin = np.isfinite(rates)
        if fin.sum() >= 2:
            rates = np.interp(np.arange(n), centers[fin], rates[fin])
            confs_r = np.interp(np.arange(n), centers, confs)
        else:
            rates = np.zeros(n); confs_r = np.zeros(n)
        g_true_s = lowpass((d["tvt"] + d["z"]), 201)
        r_all = np.gradient(g_true_s) / np.maximum(d["dh"], 1e-6)
        ki = np.where(d["kn"])[0]
        tl = ki[(ki >= t0 - 1500) & (d["dh"][ki] > 0.5)]
        r_pref = float(np.clip(np.median(r_all[tl]), -0.08, 0.08)) if len(tl) > 100 else 0.0
        dcy = np.exp(-np.maximum(np.arange(n) - t0, 0) / decay)
        r_prior = r_pref * dcy
        rates = confs_r * np.clip(rates, -0.08, 0.08) + (1 - confs_r) * r_prior
        if use_prefcal and len(tl) > 300 and np.median(confs_r[tl]) > 0.25:
            bias = np.median((r_all - rates)[tl])
            rates = rates + SHRINK * float(np.clip(bias, -0.006, 0.006))
        if use_gate:
            r_flat = np.gradient(d["zs"]) / np.maximum(d["dh"], 1e-6)
            # 適応ゲート: 場の信頼度が低い所ほど操舵(TVTフラット)に寄せる
            wg_eff = np.clip(wg + ga * (1 - confs_r), 0.0, 0.8)
            rates = (1 - wg_eff) * rates + wg_eff * np.clip(r_flat, -0.15, 0.15)
        step = rates * d["dh"]
        g0 = float(d["tin"][t0] + d["z"][t0])
        g_hat = g0 + np.cumsum(np.r_[0.0, step[t0 + 1:]])[1:]
        pred = g_hat - d["z"][ei_all]
        m = d["ev"][ei_all]
        e = (pred - d["tvt"][ei_all])[m]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e); perw[w] = float(np.sqrt(np.mean(e ** 2)))
        if dump is not None:
            gx_r = np.interp(np.arange(n), centers, gxs)
            gy_r = np.interp(np.arange(n), centers, gys)
            dump[w] = dict(rates=rates.astype(np.float32),
                           conf=confs_r.astype(np.float32),
                           pred=pred.astype(np.float32),     # 評価帯(t0+1..n)のTVT予測
                           gx=gx_r.astype(np.float32), gy=gy_r.astype(np.float32))
    es = np.concatenate(ES)
    overall = float(np.sqrt(np.mean(es ** 2)))
    dm = float(np.sqrt(np.mean(np.concatenate([e - np.median(e) for e in ES]) ** 2)))
    print(f"[{tag}] overall = {overall:.4f}  (井戸med {np.median(list(perw.values())):.3f}, "
          f"demean {dm:.4f})", flush=True)
    return overall, perw, dm

def main():
    t0 = time.time()
    D = load()
    ws, donors, pair = build_donors(D)
    print(f"wells={len(ws)}  point_donors={len(donors['pos'])}  "
          f"pair_donors={0 if pair is None else len(pair['pos'])}  ({time.time()-t0:.0f}s)",
          flush=True)
    res = {}
    if GRID:
        cfgs = [
            ("v6best(参照)", dict(wg=0.40, ga=0.30, seg=25, sigcap=800, k=128)),
            ("v7_wg45", dict(wg=0.45, ga=0.30, seg=25, sigcap=800, k=128)),
            ("v7_wg50ga25", dict(wg=0.50, ga=0.25, seg=25, sigcap=800, k=128)),
            ("v7_wg45ga40", dict(wg=0.45, ga=0.40, seg=25, sigcap=800, k=128)),
            ("v7_r4000k192", dict(wg=0.40, ga=0.30, seg=25, sigcap=800, k=192, rmax=4000)),
        ]
    else:
        cfgs = [("single", dict())]
    best = (None, 1e18)
    for tag, kw in cfgs:
        o, pw, dm = run(D, ws, donors, pair, tag, **kw)
        res[tag] = dict(overall=o, demean=dm,
                        per_well_median=float(np.median(list(pw.values()))), cfg=kw)
        if o < best[1]:
            best = (tag, o); pw_best = pw
    if GRID:
        # ドナー密度倍増(stride4)で best 設定を再評価
        ws4, donors4, pair4 = build_donors(D, stride=4)
        o, pw, dm = run(D, ws4, donors4, pair4, "v7_dense4",
                        wg=0.40, ga=0.30, seg=25, sigcap=800, k=192)
        res["v7_dense4"] = dict(overall=o, demean=dm,
                                per_well_median=float(np.median(list(pw.values()))))
        if o < best[1]:
            best = ("v7_dense4", o); pw_best = pw
    print(f"done ({time.time()-t0:.0f}s)  best={best[0]} {best[1]:.4f}  参考: 既存struct 9.70")
    with open(os.path.join(ART, "phaseC_result.json"), "w", encoding="utf-8") as f:
        json.dump(dict(res, best=best[0], per_well_best=pw_best), f,
                  ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
