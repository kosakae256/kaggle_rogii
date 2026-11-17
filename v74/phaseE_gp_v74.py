# -*- coding: utf-8 -*-
"""v74 Phase E — フィールドGP定式(GRフリー・struct不使用・ゼロから)
   「dip推定→積分」をやめ、prefix正解点+近傍井戸ANCC点を1つのGP事後に統合する。
     obs: 近傍井戸の ANCC_s(x,y) [train合法] + 自坑井prefixの g=tin+z(= ANCC + b, bは未知定数)
     モデル: y = f(x,y) + b·u + ε   (f~GP 走向異方性カーネル, u=自坑井フラグ, bは大分散で周辺化)
     予測: TVT̂ = f̂ + b̂ − z (評価行)。σも同時に出る。
   走向θ: 近傍dip点のWLS勾配から推定(勾配の直交方向)。ハイパラは少数・粗く(過学習回避)。
   env: NW(0=全), LA(走向方向長さスケール1500), LC(直交400), SF(振幅15), SN(ノイズ3),
        ROBS(3500 取込半径), CAP(1800 最大観測点), PSTRIDE(予測間引き10)
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_field2_v74 as F

ART = F.ART
NW = int(os.environ.get("NW", "0"))
LA = float(os.environ.get("LA", "1500"))
LC = float(os.environ.get("LC", "400"))
SF = float(os.environ.get("SF", "15"))
SN = float(os.environ.get("SN", "3"))
ROBS = float(os.environ.get("ROBS", "3500"))
CAP = int(os.environ.get("CAP", "1800"))
PSTRIDE = int(os.environ.get("PSTRIDE", "10"))
TAU = float(os.environ.get("TAU", "15"))    # 近傍井戸ごとのレベル自由度σ(面差±10-30ftの周辺化)
SB2 = float(os.environ.get("SB2", "100"))   # 残差bias(粗整列後)の事前分散(σ=10ft)

def main():
    t0 = time.time()
    D = F.load()
    ws, donors, pair = F.build_donors(D)       # donorsを観測点+走向推定に流用
    if NW:
        eval_ws = ws[:NW]
    else:
        eval_ws = ws
    dt = cKDTree(donors["pos"])
    n_don = len(donors["pos"])
    ES, perw = [], {}
    for wi, w in enumerate(ws):
        if w not in set(eval_ws):
            continue
        d = D[w]
        n = len(d["md"]); t0w = d["t0"]
        # --- 観測点: 近傍井戸ドナー(トラック近傍ROBS以内)
        track = np.c_[d["x"], d["y"]][::40]
        own_np = int((donors["well"] == wi).sum())
        cand = set()
        for p in track:
            jj = dt.query_ball_point(p, ROBS)
            cand.update(jj)
        cand = np.array(sorted(cand), dtype=np.int64)
        if len(cand) == 0:
            cand = np.zeros(0, np.int64)
        cand = cand[donors["well"][cand] != wi]
        if len(cand) > CAP:
            cand = cand[:: int(np.ceil(len(cand) / CAP))]
        Xn = donors["pos"][cand]
        yn = donors["c"] * 0  # placeholder(値はancが必要 → donorsにはcしかない)
        # donorsはdipなので、レベル観測は自前で再収集する(各近傍井戸のanc_s点)
        # → build: 近傍井戸集合
        nb_wells = sorted(set(donors["well"][cand].tolist()))
        Xo, yo, wo = [], [], []
        for nwi in nb_wells:
            nw_ = ws[nwi]
            nd = D[nw_]
            anc = nd["anc"]
            m = np.isfinite(anc) & (nd["dh"] > 0.5)
            anc_s = F.lowpass(np.where(np.isfinite(anc), anc, np.nan), 201)
            idx = np.where(m & np.isfinite(anc_s))[0][::12]
            if len(idx) < 5:
                continue
            Xo.append(np.c_[nd["x"][idx], nd["y"][idx]])
            yo.append(anc_s[idx])
            wo.append(np.full(len(idx), nwi, np.int32))
        if not Xo:
            # 近傍ゼロ: hold
            ei_all = np.arange(t0w + 1, n)
            pred = np.full(len(ei_all), float(d["tin"][t0w]))
            mev = d["ev"][ei_all]
            e = (pred - d["tvt"][ei_all])[mev]
            e = e[np.isfinite(e)]
            if len(e):
                ES.append(e); perw[w] = float(np.sqrt(np.mean(e ** 2)))
            continue
        Xo = np.concatenate(Xo); yo = np.concatenate(yo); wo = np.concatenate(wo)
        # トラック近傍に限定 + CAP
        tt = cKDTree(np.c_[d["x"], d["y"]][::20])
        dd_, _ = tt.query(Xo)
        keep = dd_ < ROBS
        Xo = Xo[keep]; yo = yo[keep]; wo = wo[keep]
        if len(Xo) > CAP:
            s = int(np.ceil(len(Xo) / CAP))
            Xo = Xo[::s]; yo = yo[::s]; wo = wo[::s]
        # --- 自坑井 prefix 観測(g = tin+z, ラテラル寄り末尾)
        ki = np.where(d["kn"])[0]
        tl = ki[(ki >= t0w - 2000) & (d["dh"][ki] > 0.5)][::8]
        if len(tl) < 10:
            tl = ki[-200::8]
        Xp = np.c_[d["x"][tl], d["y"][tl]]
        yp = (d["tin"][tl] + d["z"][tl])
        okp = np.isfinite(yp)
        Xp = Xp[okp]; yp = yp[okp]
        # --- 走向θ推定(ドナーdipのWLS勾配, トラック中心)
        pc = np.array([np.nanmean(d["x"]), np.nanmean(d["y"])])
        dd2, jj2 = dt.query(pc, k=min(600 + own_np, n_don), distance_upper_bound=RMAXQ())
        ok2 = np.isfinite(dd2) & (jj2 < n_don)
        jj2 = jj2[ok2]; jj2 = jj2[donors["well"][jj2] != wi][:256]
        if len(jj2) >= 12:
            wts2 = np.exp(-0.5 * (dt.data[jj2] * 0).sum(1))  # 等重み
            (gx, gy), _, _ = F.solve_grad(donors["c"][jj2], donors["az"][jj2],
                                          np.ones(len(jj2)), 0.05,
                                          np.array([1.0, 0.0]))
            theta = np.arctan2(gy, gx) + np.pi / 2            # 走向=勾配の直交
        else:
            theta = float(np.arctan2(d["dyr"][t0w], d["dxr"][t0w]))
        ct, st = np.cos(theta), np.sin(theta)

        def rot(X):
            return np.c_[X[:, 0] * ct + X[:, 1] * st, -X[:, 0] * st + X[:, 1] * ct]

        # --- 普遍クリギング: 近傍obsから線形トレンド(地域勾配~500ft/3500ft)を先に除去
        A_o = np.c_[np.ones(len(Xo)), (Xo - Xo.mean(0)) / 1000.0]

        def ridge_fit(A, yv):
            M = A.T @ A + np.diag([1e-6, 3.0, 3.0]) * len(A) * 1e-3
            return np.linalg.solve(M, A.T @ yv)
        coef = ridge_fit(A_o, yo)
        r_tr = yo - A_o @ coef
        m_in = np.abs(r_tr) < 3.5 * (np.std(r_tr) + 1e-9)
        if m_in.sum() > 50:
            coef = ridge_fit(A_o[m_in], yo[m_in])

        def trend(X):
            return np.c_[np.ones(len(X)), (X - Xo.mean(0)) / 1000.0] @ coef
        ro = yo - trend(Xo)                                   # GP対象=トレンド残差
        rp_raw = yp - trend(Xp)
        # datum b: prefix近傍のトレンド残差との中央値差で粗整列(残りは小バイアス項)
        tp = cKDTree(Xo)
        dnear, jnear = tp.query(Xp, k=min(24, len(Xo)))
        b0 = float(np.median(rp_raw) - np.median(ro[np.asarray(jnear).ravel()]))
        rp = rp_raw - b0
        Xall = np.vstack([Xo, Xp])
        u = np.r_[np.zeros(len(Xo)), np.ones(len(Xp))]        # 自坑井フラグ(残差bias)
        wid = np.r_[wo, np.full(len(Xp), -1, np.int32)]       # 近傍井戸id(-1=自坑井/クエリ)
        mu0 = 0.0
        yall = np.r_[ro, rp]
        R = rot(Xall)

        def kern(A, B, ua, ub, wa, wb):
            da = (A[:, 0][:, None] - B[:, 0][None, :]) / LA
            dc = (A[:, 1][:, None] - B[:, 1][None, :]) / LC
            K = SF ** 2 * np.exp(-0.5 * (da ** 2 + dc ** 2))
            K = K + SB2 * (ua[:, None] * ub[None, :])
            # 近傍井戸ごとのレベル自由度(面差の周辺化): 同一近傍井戸同士に +τ²
            same = (wa[:, None] == wb[None, :]) & (wa[:, None] >= 0)
            K = K + (TAU ** 2) * same
            return K
        K = kern(R, R, u, u, wid, wid) + (SN ** 2) * np.eye(len(R))
        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(R)))
        except np.linalg.LinAlgError:
            continue
        al = np.linalg.solve(L.T, np.linalg.solve(L, yall))
        # --- 予測(評価行, PSTRIDE間引き→補間)
        ei_all = np.arange(t0w + 1, n)
        qi = ei_all[::PSTRIDE]
        Q = rot(np.c_[d["x"][qi], d["y"][qi]])
        uq = np.ones(len(Q))                                   # 予測も自坑井(bを含む)
        Ks = kern(Q, R, uq, u, np.full(len(Q), -1, np.int32), wid)
        Xq = np.c_[d["x"][qi], d["y"][qi]]
        fq = Ks @ al + trend(Xq) + b0
        g_hat = np.interp(ei_all, qi, fq)
        pred = g_hat - d["z"][ei_all]
        # 物理クリップ: 既知末尾TVTから±120ft超のドリフトは起こらない
        hold0 = float(d["tin"][t0w])
        pred = np.clip(pred, hold0 - 120.0, hold0 + 120.0)
        mev = d["ev"][ei_all]
        e = (pred - d["tvt"][ei_all])[mev]
        e = e[np.isfinite(e)]
        if len(e):
            ES.append(e); perw[w] = float(np.sqrt(np.mean(e ** 2)))
        if (len(perw)) % 50 == 0:
            print(f"  {len(perw)}/{len(eval_ws)}  ({time.time()-t0:.0f}s)", flush=True)
    es = np.concatenate(ES)
    overall = float(np.sqrt(np.mean(es ** 2)))
    dm = float(np.sqrt(np.mean(np.concatenate([e - np.median(e) for e in ES]) ** 2)))
    print(f"\n[Phase E GP] overall = {overall:.4f}  (井戸med {np.median(list(perw.values())):.3f}, "
          f"demean {dm:.4f})  n={len(perw)}井")
    json.dump(dict(overall=overall, demean=dm, n=len(perw),
                   params=dict(LA=LA, LC=LC, SF=SF, SN=SN, ROBS=ROBS, CAP=CAP),
                   per_well=perw),
              open(os.path.join(ART, "phaseE_gp.json"), "w"), indent=1)

def RMAXQ():
    return 4000.0

if __name__ == "__main__":
    main()
