# v59 — GR-free強化 lab(docs/v59.md): ABS / C勾配積分 / B2自己較正 / D Kalman(RTS) を全773井LOWOで比較
#   面フィット = v56 と同一(局所平面+リッジ+井戸多様性)だが、prefix行にもクエリを張り、勾配ベクトルも出す。
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v59", "artifacts_v59"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
STRIDE = 10; QSTRIDE = 5; K = 128; RMAX = 4000.0; SIG0 = 300.0; LAMS = 5.0
SLOPE_CLIP = 80.0                                   # ft/kft (~4.5°)
DECAY_B2 = float(os.environ.get("DECAY_B2", "800")) # prefixトレンド外挿の減衰(ft)
QS = [0.02, 0.08, 0.3]                              # Kalman過程分散 q (ft^2 / row)

def local_fit(tree, PXY, PA, PW, qxy, own, own_npts):
    n = len(qxy); kq = min(K + own_npts + 64, len(PXY))
    pred = np.full(n, np.nan); a1 = np.zeros(n); a2 = np.zeros(n); mdist = np.full(n, np.nan)
    for c0 in range(0, n, 512):
        q = qxy[c0:c0 + 512]
        dd, ii = tree.query(q, k=kq, distance_upper_bound=RMAX)
        for bi in range(len(q)):
            d = dd[bi]; j = ii[bi]
            ok = np.isfinite(d) & (j < len(PXY))
            j = j[ok]; d = d[ok]
            keep = PW[j] != own
            j = j[keep]; d = d[keep]
            if len(j) < 12: continue
            NBW, PPW = 4, 48
            sel = []; seen = {}
            for idx in range(len(j)):
                wid = PW[j[idx]]
                c = seen.get(wid, 0)
                if c == 0 and len(seen) >= NBW: continue
                if c >= PPW: continue
                seen[wid] = c + 1; sel.append(idx)
                if len(sel) >= NBW * PPW: break
            j = j[sel]; d = d[sel]
            i = c0 + bi
            sig = max(SIG0, float(np.median(d)))
            w = np.exp(-0.5 * (d / sig) ** 2) + 1e-6
            dx = (PXY[j, 0] - qxy[i, 0]) * 1e-3; dy = (PXY[j, 1] - qxy[i, 1]) * 1e-3
            A = np.c_[np.ones(len(j)), dx, dy]; yv = PA[j]
            R = np.diag([0.0, LAMS, LAMS])
            for _ in range(2):
                W_ = w; AtW = A.T * W_
                try: coef = np.linalg.solve(AtW @ A + R, AtW @ yv)
                except np.linalg.LinAlgError: coef = np.array([np.average(yv, weights=W_), 0, 0])
                r = yv - A @ coef
                mad = np.median(np.abs(r - np.median(r))) + 1e-6
                w = W_ * np.clip(1.0 - (r / (4.5 * mad)) ** 2, 0.05, None)
            pred[i] = coef[0]
            a1[i] = np.clip(coef[1], -SLOPE_CLIP, SLOPE_CLIP)
            a2[i] = np.clip(coef[2], -SLOPE_CLIP, SLOPE_CLIP)
            mdist[i] = float(np.median(d))
    return pred, a1, a2, mdist

def sigma_of(dist):
    return np.where(dist < 300, 5.2, np.where(dist < 800, 7.3, np.where(dist < 2000, 10.8, 17.0)))

def main():
    t0 = time.time()
    import nnpf_v51 as npf                    # SIMD: struct st(GR-free)を取得(sim は使わない)
    ST = {w: np.asarray(npf.SIMD[w]["st"], np.float64) for w in npf.SIMD}
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    eval_wells = wells[:NW] if NW else wells
    PX, PA_, PWID = [], [], []; HW = {}
    for wi, w in enumerate(wells):
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "ANCC", "TVT", "TVT_input"])
        HW[w] = hw
        hh = hw.iloc[::STRIDE]
        m = hh[["X", "Y", "ANCC"]].notna().all(axis=1)
        PX.append(hh.loc[m, ["X", "Y"]].to_numpy(float)); PA_.append(hh.loc[m, "ANCC"].to_numpy(float))
        PWID.append(np.full(int(m.sum()), wi, np.int32))
    PXY = np.concatenate(PX); PA = np.concatenate(PA_); PW = np.concatenate(PWID)
    tree = cKDTree(PXY)
    print(f"wells={len(wells)} points={len(PXY)} ({time.time()-t0:.0f}s)", flush=True)
    preds = {k: {} for k in ["ABS", "C", "B2", "ST"] + [f"D_q{q}" for q in QS] + [f"D2_q{q}" for q in QS]}
    YW = {}; DUMP = {}
    for wi, w in enumerate(wells):
        if w not in set(eval_wells): continue
        hw = HW[w]
        x = hw["X"].to_numpy(float); yy = hw["Y"].to_numpy(float); z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        md = hw["MD"].to_numpy(float)
        kn = np.isfinite(tin); ev = ~kn
        if kn.sum() < 10 or ev.sum() < 100: continue
        n = len(hw); ilast = np.where(kn)[0][-1]
        lkv = float(tin[ilast])
        qi = np.unique(np.concatenate([np.arange(0, n, QSTRIDE), [ilast]]))
        ancp_q, a1q, a2q, mdq = local_fit(tree, PXY, PA, PW, np.c_[x[qi], yy[qi]], wi, int((PW == wi).sum()))
        good = np.isfinite(ancp_q)
        if good.sum() < 5: continue
        ancp = np.interp(np.arange(n), qi[good], ancp_q[good])
        a1 = np.interp(np.arange(n), qi[good], a1q[good]); a2 = np.interp(np.arange(n), qi[good], a2q[good])
        dist = np.interp(np.arange(n), qi[good], mdq[good])
        # 圏外フォールバック(v56と同じ): 外部点が遠すぎる井戸は全変種 prefix-hold
        if not np.isfinite(np.nanmedian(dist)) or np.nanmedian(dist) > 1500.0:
            hold = np.full(n, float(tin[np.where(kn)[0][-1]]))
            for kname in list(preds):
                preds[kname][w] = hold[ev]
            YW[w] = tvt[ev]
            DUMP[w] = np.stack([hold[ev]] * 4 + [np.full(int(ev.sum()), 9e3)], 1).astype(np.float32)
            continue
        # ---- ABS: b̂ + ancp − z(prefixでb̂)----
        bhat = np.median(tin[kn] + z[kn] - ancp[kn])
        p_abs = bhat + ancp - z
        # ---- C: 勾配積分(prefix端から)----
        dxs = np.diff(x, prepend=x[0]); dys = np.diff(yy, prepend=yy[0]); dzs = np.diff(z, prepend=z[0])
        dAn = (a1 * dxs + a2 * dys) * 1e-3
        incr = dAn - dzs
        cum = np.cumsum(incr)
        p_c = lkv + (cum - cum[ilast])
        # ---- B2: prefix面残差トレンドの減衰外挿(ABSの上に)----
        kidx = np.where(kn & np.isfinite(ancp))[0][-200:]
        res_k = (tin[kidx] + z[kidx]) - (bhat + ancp[kidx])
        if len(kidx) >= 20:
            A2 = np.c_[np.ones(len(kidx)), md[kidx] - md[ilast]]
            c2, *_ = np.linalg.lstsq(A2, res_k, rcond=None)
            trend = c2[0] + c2[1] * np.clip(md - md[ilast], None, 0)     # prefix側はそのまま
            fwd = (c2[0] + c2[1] * 0.0) * np.exp(-np.maximum(md - md[ilast], 0) / DECAY_B2)
            corr = np.where(md <= md[ilast], trend, fwd)
        else:
            corr = np.zeros(n)
        p_b2 = p_abs + corr
        # ---- D: Kalman RTS(過程=C増分, 観測=B2絶対面)----
        Rv = sigma_of(dist) ** 2
        for q in QS:
            mu = np.zeros(n); P_ = np.zeros(n)
            mu_p = np.zeros(n); P_p = np.zeros(n)
            mu_c = lkv; P_c = 1.0
            for t in range(ilast + 1, n):
                mu_pred = mu_c + incr[t]; P_pred = P_c + q
                Kg = P_pred / (P_pred + Rv[t])
                mu_c = mu_pred + Kg * (p_b2[t] - mu_pred)
                P_c = (1 - Kg) * P_pred
                mu[t] = mu_c; P_[t] = P_c; mu_p[t] = mu_pred; P_p[t] = P_pred
            # RTS平滑(後ろ→前)
            sm = mu.copy()
            for t in range(n - 2, ilast, -1):
                Ck = P_[t] / P_p[t + 1]
                sm[t] = mu[t] + Ck * (sm[t + 1] - mu_p[t + 1])
            pd_ = np.full(n, np.nan); pd_[ilast + 1:] = sm[ilast + 1:]
            preds[f"D_q{q}"][w] = pd_[ev]
        # ---- D2: 過程=Δst(struct増分, GR-free), 観測=B2絶対面 ----
        stw = ST.get(w)
        if stw is not None:
            nev = int(ev.sum())
            m2 = min(nev, len(stw))
            inc2 = np.diff(stw[:m2], prepend=stw[0])
            evi = np.where(ev)[0][:m2]
            for q in QS:
                mu_c = lkv; P_c = 1.0
                mu = np.zeros(m2); P_ = np.zeros(m2); mu_p = np.zeros(m2); P_p = np.zeros(m2)
                for t in range(m2):
                    mu_pred = mu_c + (inc2[t] if t > 0 else stw[0] - stw[0])
                    P_pred = P_c + q
                    Rt = Rv[evi[t]]
                    Kg = P_pred / (P_pred + Rt)
                    mu_c = mu_pred + Kg * (p_b2[evi[t]] - mu_pred)
                    P_c = (1 - Kg) * P_pred
                    mu[t] = mu_c; P_[t] = P_c; mu_p[t] = mu_pred; P_p[t] = P_pred
                sm = mu.copy()
                for t in range(m2 - 2, -1, -1):
                    Ck = P_[t] / max(P_p[t + 1], 1e-12)
                    sm[t] = mu[t] + Ck * (sm[t + 1] - mu_p[t + 1])
                out2 = np.full(nev, np.nan); out2[:m2] = sm
                preds[f"D2_q{q}"][w] = out2
            outst = np.full(nev, np.nan); outst[:m2] = stw[:m2]
            preds["ST"][w] = outst
        preds["ABS"][w] = p_abs[ev]; preds["C"][w] = p_c[ev]; preds["B2"][w] = p_b2[ev]
        YW[w] = tvt[ev]
        DUMP[w] = np.stack([p_abs[ev], p_c[ev], p_b2[ev], preds[f"D_q{QS[1]}"][w], dist[ev]], 1).astype(np.float32)
        if (len(YW)) % 100 == 0: print(f"  {len(YW)}井 ({time.time()-t0:.0f}s)", flush=True)
    def pooled(dh):
        es = [np.asarray(dh[w], np.float64) - YW[w] for w in dh]
        ee = np.concatenate(es); ee = ee[np.isfinite(ee)]
        return float(np.sqrt(np.mean(ee ** 2)))
    print(f"\n=== v59 GR-free 変種(n={len(YW)}井, 参照: struct 9.71 / v56ABS 15.5) ===")
    res = {}
    for k in preds:
        if preds[k]:
            res[k] = pooled(preds[k])
            print(f"  {k:8s}: {res[k]:8.3f}")
    # fold表(最良変種)
    best = min(res, key=res.get)
    wl = sorted(YW)
    for f in range(5):
        val = set(sorted(wl)[f::5])
        es = np.concatenate([preds[best][w] - YW[w] for w in wl if w in val])
        es = es[np.isfinite(es)]
        print(f"  [{best}] fold{f}: {np.sqrt(np.mean(es**2)):.3f}")
    np.savez_compressed(os.path.join(ART, "geofree_rows.npz"), **DUMP)
    json.dump(res, open(os.path.join(ART, "geofree_result.json"), "w"), indent=1)
    print(f"saved geofree_rows.npz ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
