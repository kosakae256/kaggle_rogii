# v58 DTW lab — GR↔typewell アライメント DP 族の分解検証(docs/v58.md)
#   V0: 単調 subsequence DTW(古典) / V1: 非単調±K DP(枝刈りなし, Z無視)
#   V2: Z結合 DP(移動コスト λ|Δj−ΔZ/step|) / V3: V2のforward-backward soft事後平均
#   共通: 0.5ft格子, lk錨開始, GRコスト ((g−t)/σ_w)^2, σ_w=prefix残差較正
#   評価: beam解剖と同一80井。参照 beam16.6 / PF(pf_1/tw) 8.89
import os, sys, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v58", "artifacts_v58"); os.makedirs(ART, exist_ok=True)
NWELL = int(os.environ.get("NWELL", "80"))
STEP = 0.5; K = int(os.environ.get("K", "8"))       # 遷移窓 ±K格子(=±4ft/行)
LAM = float(os.environ.get("LAM", "1.0"))           # Z結合の強さ
MC = float(os.environ.get("MC", "0.10"))            # V1の移動コスト(格子あたり)

def smooth(v, fb, r=2):
    s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
    return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()

def build(wn):
    hw = pd.read_csv(os.path.join(TR, f"{wn}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv")).sort_values("TVT")
    t_tvt = tw["TVT"].to_numpy(float); t_gr = tw["GR"].to_numpy(float)
    fin = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[fin], t_gr[fin]
    if len(t_tvt) < 50: return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 10 or len(ev) < 100: return None
    fb = float(np.nanmean(t_gr))
    grid = np.arange(t_tvt.min(), t_tvt.max() + STEP, STEP)
    tg = np.interp(grid, t_tvt, smooth(t_gr, fb, 2))
    gr_full = smooth(hw["GR"].to_numpy(float), fb, 2)
    i0 = ev.index[0]
    hgr = gr_full[i0:]
    y = hw["TVT"].to_numpy(float)[i0:]
    z = hw["Z"].to_numpy(float)[i0:]
    lkv = float(kn["TVT_input"].iloc[-1])
    # σ較正(prefixのGR残差)
    kpos = hw.index.get_indexer(kn.index)
    resid = gr_full[kpos] - np.interp(kn["TVT_input"].to_numpy(float), t_tvt, smooth(t_gr, fb, 2))
    sig = float(np.nanstd(resid)); sig = 15.0 if not np.isfinite(sig) or sig < 3 else max(sig, 5.0)
    dz = np.diff(z, prepend=z[0])                   # TVTはZと同方向に増減(v56: TVT=(ANCC−Z)+b → dTVT=−dZ+dANCC…
    dzq = -dz / STEP                                 # 実効: dTVT≈−dZ(浅い変動)。符号はデータ整合で採用
    j0 = int(np.clip(round((lkv - grid[0]) / STEP), 0, len(grid) - 1))
    return dict(tg=tg, grid=grid, hgr=hgr, y=y, sig=sig, dzq=dzq, j0=j0)

def unary(d):
    return ((d["hgr"][:, None] - d["tg"][None, :]) / d["sig"]) ** 2   # (T,G)

def v0_monotone(d):
    U = unary(d); T, G = U.shape
    INF = 1e18
    prev = np.full(G, INF); prev[d["j0"]] = U[0, d["j0"]]
    bp = np.zeros((T, G), np.int16)
    for i in range(1, T):
        cand = np.stack([prev,
                         np.concatenate([[INF], prev[:-1]]),
                         np.concatenate([[INF, INF], prev[:-2]])])
        arg = cand.argmin(0)
        prev = cand.min(0) + U[i]
        bp[i] = arg
    j = int(prev.argmin()); path = np.zeros(T, np.int32); path[-1] = j
    for i in range(T - 1, 0, -1):
        j = j - bp[i, j]; path[i - 1] = j
    return d["grid"][path]

def dp_nonmono(d, use_z, soft):
    U = unary(d); T, G = U.shape
    offs = np.arange(-K, K + 1)
    if use_z:
        mv = lambda i: LAM * np.abs(offs - d["dzq"][i]) * 0.5
    else:
        mv = lambda i: MC * np.abs(offs)
    NEG = -1e18
    if soft:
        la = np.full(G, NEG); la[d["j0"]] = -U[0, d["j0"]]
        A = np.zeros((T, G)); A[0] = la
        for i in range(1, T):
            m = np.full(G, NEG)
            for oi, o in enumerate(offs):
                src = np.roll(la, o)
                if o > 0: src[:o] = NEG
                elif o < 0: src[o:] = NEG
                m = np.logaddexp(m, src - mv(i)[oi])
            la = m - U[i]; A[i] = la
        lb = np.zeros(G)
        P = np.zeros((T, G))
        for i in range(T - 1, -1, -1):
            lg = A[i] + lb
            lg -= lg.max()
            p = np.exp(lg); P[i] = p / p.sum()
            if i > 0:
                m = np.full(G, NEG)
                src0 = lb - U[i]
                for oi, o in enumerate(offs):
                    src = np.roll(src0, -o)
                    if o > 0: src[-o:] = NEG
                    elif o < 0: src[:-o] = NEG
                    m = np.logaddexp(m, src - mv(i)[oi])
                lb = m
        return P @ d["grid"]
    else:
        INF = 1e18
        prev = np.full(G, INF); prev[d["j0"]] = U[0, d["j0"]]
        bp = np.zeros((T, G), np.int16)
        for i in range(1, T):
            best = np.full(G, INF); barg = np.zeros(G, np.int16)
            mvc = mv(i)
            for oi, o in enumerate(offs):
                src = np.roll(prev, o)
                if o > 0: src[:o] = INF
                elif o < 0: src[o:] = INF
                c = src + mvc[oi]
                upd = c < best
                best[upd] = c[upd]; barg[upd] = o
            prev = best + U[i]; bp[i] = barg
        j = int(prev.argmin()); path = np.zeros(T, np.int32); path[-1] = j
        for i in range(T - 1, 0, -1):
            j = j - bp[i, j]; path[i - 1] = j
        return d["grid"][path]

def main():
    t0 = time.time()
    cand = np.load(os.path.join(P0, "v54", "artifacts_v54", "cand_pf_1_tw.npz"))
    wells_all = sorted(k for k in cand.files if not k.endswith("|w"))
    rng = np.random.default_rng(0)
    wells = [wells_all[i] for i in rng.permutation(len(wells_all))[:NWELL]]
    ES = {k: [] for k in ("V0", "V1", "V2", "V3", "PF")}
    for wi, wn in enumerate(wells):
        d = build(wn)
        if d is None: continue
        n = len(d["y"]); finy = np.isfinite(d["y"])
        pf = (cand[f"{wn}|w"][:, None] * cand[wn].astype(np.float64)).sum(0)
        n2 = min(n, len(pf))
        preds = {"V0": v0_monotone(d), "V1": dp_nonmono(d, False, False),
                 "V2": dp_nonmono(d, True, False), "V3": dp_nonmono(d, True, True),
                 "PF": pf[:n2]}
        for k, p in preds.items():
            m = min(len(p), n)
            ES[k].append((np.asarray(p)[:m] - d["y"][:m])[finy[:m]])
        if (wi + 1) % 20 == 0: print(f"  {wi+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    print("\n--- v58 DTW族 pooled RMSE(80井, 参照: beam med 16.6) ---")
    res = {}
    for k in ("V0", "V1", "V2", "V3", "PF"):
        ee = np.concatenate(ES[k]); r = float(np.sqrt(np.mean(ee ** 2))); res[k] = r
        lab = {"V0": "単調DTW(古典)", "V1": "非単調DP(Z無視)", "V2": "Z結合DP(hard)",
               "V3": "Z結合DP(soft FB)", "PF": "PF参照(pf_1/tw)"}[k]
        print(f"  {k} {lab:18s}: {r:8.3f}")
    json.dump(res, open(os.path.join(ART, "dtw_result.json"), "w"), indent=1)
    print(f"done ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
