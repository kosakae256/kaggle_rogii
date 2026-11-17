# -*- coding: utf-8 -*-
"""v74 Phase H — test既知帯ドナー(相互プール)の疑似test実験
   fold の155井=疑似test。場・κは残り618井のみ(=デプロイ再現)。
   (a) deploy:      場 = train618のみ
   (b) +prefix:     (a) + 疑似test井の既知帯(TVT_input)から作ったdipドナー
   期待: LB転移1.113の一因「test同輩が場に居ない」の欠損を(b)が埋めるか。
   参考: 現行CV(場=全井LOO)9.832 は (a) より楽観。
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ.update(SEG_LEN="0", VEC="1", QW="0", ROBUST="0", STEREO="0")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseG_struct2_v74 as G

TRAIN = G.TRAIN
NW = int(os.environ.get("NW", "0"))
PRE_WIN = int(os.environ.get("PRE_WIN", "3000"))   # 既知帯ドナーに使う末尾行数

def prefix_pseudo(w):
    """既知帯(TVT_input)末尾から segment_well 互換の擬似wellを作る(test再現: TVT不使用)"""
    hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                     usecols=["X", "Y", "Z", "TVT_input"])
    ti = hw["TVT_input"].to_numpy(float)
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) < 100:
        return None
    s = int(fin.max())
    i0 = max(0, s - PRE_WIN)
    X = hw["X"].to_numpy(float)[i0:s + 1]
    Y = hw["Y"].to_numpy(float)[i0:s + 1]
    Z = hw["Z"].to_numpy(float)[i0:s + 1]
    tin = ti[i0:s + 1]
    n = len(X) - 1
    if n < 50:
        return None
    dX = np.diff(X); dY = np.diff(Y)
    Lxy = np.maximum(np.hypot(dX, dY), 1e-3)
    # ラテラル部のみ(ビルド区間はdip定義が乱れる)
    lat_frac = float(np.mean(Lxy > 0.5))
    wd = dict(n=n, dX=dX, dY=dY, ndz=-np.diff(Z), Lxy=Lxy, arc=np.cumsum(Lxy),
              Xl=X, Yl=Y, R0=(tin[1:] - tin[0]))
    if lat_frac < 0.5:
        return None
    return wd

def traj_donors(w):
    """TVT/GR不使用: 軌道(X,Y,Z 全行=test評価帯も既知)から操舵代理dipドナーを作る。
    c_traj = d(smooth Z)/d弧長(ラテラル・低ビルドレート区間のみ)。"""
    hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                     usecols=["X", "Y", "Z"])
    X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float); Z = hw["Z"].to_numpy(float)
    if len(X) < 300:
        return None
    dX = np.gradient(X); dY = np.gradient(Y)
    dh = np.maximum(np.hypot(dX, dY), 1e-3)
    zs = pd.Series(Z).rolling(201, center=True, min_periods=1).mean().to_numpy()
    c = np.gradient(zs) / dh                                  # dZ/d弧長 ≈ dip(追従時)
    build = np.abs(np.gradient(pd.Series(c).rolling(101, center=True,
                                                    min_periods=1).mean().to_numpy()))
    m = (dh > 0.5) & (np.abs(c) < G.CMAX) & (build < 0.0006) # ラテラル・低ビルドレート
    idx = np.where(m)[0][::30]
    if len(idx) < 5:
        return None
    az = np.arctan2(dY[idx], dX[idx])
    return np.c_[X[idx], Y[idx]], c[idx], az

def main():
    t0 = time.time()
    fps = sorted(os.listdir(TRAIN))
    wids = sorted(set(f.split("__")[0] for f in fps if f.endswith("__horizontal_well.csv")))
    if NW:
        wids = wids[:NW]
    wells = []
    for wi, w in enumerate(wids):
        wd = G.load_well(os.path.join(TRAIN, f"{w}__horizontal_well.csv"))
        if wd is None:
            continue
        wd["wi"] = wi; wd["wid"] = w
        wd["seg"] = G.segment_well(wd)
        wells.append(wd)
    print(f"wells={len(wells)}  ({time.time()-t0:.0f}s)", flush=True)
    # 疑似test用: 既知帯擬似well(全員分作っておく)
    pre = {}
    for wd in wells:
        p = prefix_pseudo(wd["wid"])
        if p is not None:
            p_seg = G.segment_well(p)
            if p_seg is not None:
                p["seg"] = p_seg
                pre[wd["wi"]] = p
    print(f"prefix疑似well={len(pre)}  ({time.time()-t0:.0f}s)", flush=True)
    gi = np.arange(len(wells))
    folds = list(GroupKFold(5).split(gi, gi, gi))
    E = dict(a=[], b=[])
    PERW = {}
    for fold, (tr_i, va_i) in enumerate(folds):
        tr_wi = set(wells[i]["wi"] for i in tr_i)
        F = G.build_field(wells, tr_wi)                      # デプロイ再現: train618のみ
        K = G.fit_kappa(F, wells, tr_wi)
        # (b): 疑似test井の既知帯ドナーを追加
        px, py, pc, paz, pwi, pse = [], [], [], [], [], []
        for i in va_i:
            wi = wells[i]["wi"]
            if wi not in pre:
                continue
            segid, mid, az, c, se = pre[wi]["seg"]
            ok = np.isfinite(c) & (np.abs(c) <= G.CMAX) & np.isfinite(se)
            px.append(mid[ok, 0]); py.append(mid[ok, 1]); pc.append(c[ok])
            paz.append(az[ok]); pse.append(se[ok])
            pwi.append(np.full(int(ok.sum()), wi, float))
        Fb = dict(x=np.r_[F["x"], np.concatenate(px)],
                  y=np.r_[F["y"], np.concatenate(py)],
                  c=np.r_[F["c"], np.concatenate(pc)],
                  az=np.r_[F["az"], np.concatenate(paz)],
                  wi=np.r_[F["wi"], np.concatenate(pwi)],
                  se=np.r_[F["se"], np.concatenate(pse)])
        Fb["wi2"] = Fb["wi"].copy()
        Fb["qw"] = np.ones(len(Fb["x"]))
        Fb["ux"], Fb["uy"] = np.cos(Fb["az"]), np.sin(Fb["az"])
        nadd = len(Fb["x"]) - len(F["x"])
        arm_b = os.environ.get("ARM_B", "1") == "1"
        for i in va_i:
            wd = wells[i]
            if wd.get("seg") is None:
                continue
            ev = wd["ev_tail"] & np.isfinite(wd["tvt_tail"])
            if not ev.any():
                continue
            pa, _, dd_a, _ = G.predict(F, wd, K)
            ea = (pa - wd["tvt_tail"])[ev]
            PERW[wd["wid"]] = (float(np.median(dd_a)),
                               float(np.sqrt(np.nanmean(ea ** 2))), int(ev.sum()))
            E["a"].append(ea)
            if arm_b:
                pb, _, _, _ = G.predict(Fb, wd, K)
                E["b"].append((pb - wd["tvt_tail"])[ev])
        if not arm_b:
            E["b"] = E["a"]
        ra = np.sqrt(np.mean(np.concatenate(E["a"][-len(va_i):] if False else E["a"]) ** 2))
        print(f"fold{fold}: +prefixドナー{nadd}点  ({time.time()-t0:.0f}s)", flush=True)
    rmse = lambda L: float(np.sqrt(np.mean(np.concatenate(L) ** 2)))
    dm = lambda L: float(np.sqrt(np.mean(np.concatenate(
        [e - np.median(e) for e in L]) ** 2)))
    ra, rb = rmse(E["a"]), rmse(E["b"])
    print(f"\n[Phase H 疑似test] (a)deploy場       = {ra:.4f}  (demean {dm(E['a']):.4f})")
    print(f"                   (b)+test既知帯ドナー = {rb:.4f}  (demean {dm(E['b']):.4f})")
    print(f"                   Δ = {rb - ra:+.4f}   [参考: 全井LOO(現行CV) 9.832]")
    json.dump(dict(deploy=ra, with_prefix=rb, per_well=PERW),
              open(os.path.join(G.ART, "phaseH_result.json"), "w"))

if __name__ == "__main__":
    main()
