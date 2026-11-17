# v74 Phase A2 — 非ML対照: 近傍ドリフト直接コピー(フォーラム助言「近傍のwigglesをシフトコピー」の一般形)
#   TVT̂(t) = TVT_last + γ · Σ_k w_k(t)·dg_k(t) / Σ_k w_k(t),  w_k = exp(−dist_k(t)/λ)
#   dg_k(t) = g_k(現在位置への射影) − g_k(境界位置への射影)(絶対レベル不使用=datum回避)
#   γ(shrink)×λ(距離スケール) をグリッド走査(overall pooled)。ML無しでチャネルの情報量を直接測る。
# env: NW(0=全), NBK(近傍数, 既定5)
import os, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
NBK = int(os.environ.get("NBK", "5"))

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def main():
    t0 = time.time()
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    D = {}
    for w in wells:
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        x = hw["X"].to_numpy(float); y = hw["Y"].to_numpy(float)
        z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            continue
        # ラテラル部のみ射影対象(ビルド区間のg巨大変動の混入=コピー爆死の原因を排除)
        lat = np.abs(np.gradient(lowpass(z, 301))) < 0.3
        li = np.where(lat)[0]
        t0w = int(np.where(kn)[0][-1])
        zs = lowpass(z, 201)
        dz_ev = zs[np.where(ev)[0]] - zs[t0w]                  # 自坑井ΔZ(境界から)
        D[w] = dict(x=x, y=y, g=lowpass(tvt + z, 201), tvt=tvt, tin=tin,
                    kn=kn, ev=ev, t0=t0w, li=li, dz_ev=dz_ev)
    ws = sorted(D.keys())
    cent = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in ws])
    trees = {}
    for w in ws:
        lidx = D[w]["li"][::2]
        if len(lidx) > 100:
            D[w]["lidx"] = lidx
            trees[w] = cKDTree(np.c_[D[w]["x"][lidx], D[w]["y"][lidx]])
    # per-well: dg (n_ev, K), dist (n_ev, K), 自坑井ΔZ, 誤差ベース(hold)
    DG, DIST, Y0, DZ = [], [], [], []
    for i, w in enumerate(ws):
        d = D[w]
        dd = np.hypot(cent[:, 0] - cent[i, 0], cent[:, 1] - cent[i, 1]); dd[i] = 1e18
        nb = [ws[j] for j in np.argsort(dd) if ws[j] in trees][:NBK]
        ei = np.where(d["ev"])[0]
        p0 = np.array([d["x"][d["t0"]], d["y"][d["t0"]]])
        dg = np.zeros((len(ei), NBK)); ds = np.full((len(ei), NBK), 1e9)
        for k, nw in enumerate(nb):
            ndd = D[nw]; lidx = ndd["lidx"]
            _, j0 = trees[nw].query(p0); j0 = int(lidx[int(j0)])
            dq, jj = trees[nw].query(np.c_[d["x"][ei], d["y"][ei]])
            jj = lidx[np.asarray(jj, int)]
            dg[:, k] = np.clip(ndd["g"][jj] - ndd["g"][j0], -80, 80)
            ds[:, k] = dq
        DG.append(dg); DIST.append(ds)
        DZ.append(d["dz_ev"])
        Y0.append(d["tvt"][ei] - float(d["tin"][d["t0"]]))     # 真のドリフト(=hold誤差の逆符号)
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(ws)}  ({time.time()-t0:.0f}s)", flush=True)
    y = np.concatenate(Y0)
    hold = float(np.sqrt(np.mean(y ** 2)))
    print(f"wells={len(ws)}  rows={len(y)}  hold={hold:.4f}  ({time.time()-t0:.0f}s)")
    res = dict(n_wells=len(ws), hold=hold, grid={})
    best = (None, 1e18)
    for lam in (150, 300, 600, 1200, 2400):
        wgt = [np.exp(-ds / lam) for ds in DIST]
        pred = np.concatenate([(dg * w).sum(1) / np.maximum(w.sum(1), 1e-9) - dz
                               for dg, w, dz in zip(DG, wgt, DZ)])
        for gam in (0.2, 0.4, 0.6, 0.8, 1.0):
            r = float(np.sqrt(np.mean((gam * pred - y) ** 2)))
            res["grid"][f"lam{lam}_gam{gam}"] = r
            if r < best[1]:
                best = ((lam, gam), r)
    print(f"best: λ={best[0][0]} γ={best[0][1]} → overall {best[1]:.4f}  (hold {hold:.4f})")
    # 最近傍1本コピー(γ=1)も参考に
    pred1 = np.concatenate([dg[np.arange(len(dg)), np.argmin(ds, 1)] - dz
                            for dg, ds, dz in zip(DG, DIST, DZ)])
    print(f"最近傍1本コピー(γ=1): {float(np.sqrt(np.mean((pred1 - y) ** 2))):.4f}")
    res["best"] = dict(lam=best[0][0], gam=best[0][1], rmse=best[1])
    res["copy1"] = float(np.sqrt(np.mean((pred1 - y) ** 2)))
    with open(os.path.join(ART, "phaseA2_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
