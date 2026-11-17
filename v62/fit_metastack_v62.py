# メタスタッカー比較(7脚の選択問題として): 固定重み vs ridge vs FWLS vs softmaxゲート
#   全て GroupKFold(5) のメタOOFで正直評価。特徴=既存列+脚統計のみ。
import os, json, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

ART = "v62/artifacts_v62"; ART52 = "v52/artifacts_v52"
LEGS = ["GBM", "TCN", "L3img", "L4", "L5", "L6", "L7"]

def main():
    tb = pq.read_table(f"{ART52}/data/train.parquet",
                       columns=["id", "well", "target", "md_since", "pf_ancc", "last_known_tvt",
                                "pf_ancc_std_1", "pf_ancc_std_4"])
    ids = tb.column("id").to_pandas().astype(str)
    gw = tb.column("well").to_pandas().astype(str).to_numpy()
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    def col(c):
        try: return np.nan_to_num(tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0)
        except Exception: return np.zeros(len(y))
    fx_static = {c: col(c) for c in ["pf_ancc_std_1", "pf_ancc_std_4"]}
    _gfA = np.load("v48/artifacts_v48/gf_feature.npy"); _vbA = np.load(f"{ART52}/v50b_feature.npy")
    fx_static["gf_std"] = np.nan_to_num(_gfA[:, 1], nan=0.0)
    fx_static["v50b_std"] = np.nan_to_num(_vbA[:, 1], nan=0.0)
    fx_static["v50b_margin"] = np.nan_to_num(_vbA[:, 2], nan=0.0)
    gbm = np.load(f"{ART52}/gbm_v41_oof.npy").astype(np.float64)
    nn = np.mean([np.load(f"{ART52}/nn_oof_combo.npz")[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], 0).astype(np.float64)
    pp = json.loads(open(f"{ART52}/pp_params.json").read())
    def ppe(r): return (r * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
    eg = ppe(gbm); et = ppe(nn)
    E = {"L3img": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/v62_oof_err.npz").items()},
         "L4": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l4_oof_err.npz").items()},
         "L5": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l5_oof_err.npz").items()},
         "L6": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l6_oof_err.npz").items()},
         "L7": {w: v.astype(np.float64) for w, v in np.load(f"{ART}/l7_oof_err.npz").items()}}
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    WELLS = sorted(E["L5"])
    # 行列組み立て(行=全評価行): P(N,7)=脚予測誤差、F(N,nf)=メタ特徴、wid
    Pw, Fw, Yw, Wid = [], [], [], []
    for w in WELLS:
        ii = np.where(gw == w)[0]; o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        n0 = min([len(E[n][w]) for n in E] + [len(iio)])
        ms = np.ones(n0, bool)
        for n in E: ms &= np.isfinite(E[n][w][:n0])
        ms &= np.isfinite(eg[iio[:n0]])
        iiom = iio[:n0][ms]
        errs = np.stack([eg[iiom], et[iiom]] + [E[n][w][:n0][ms] for n in ("L3img", "L4", "L5", "L6", "L7")], 1)
        Pw.append(errs)
        med_pred_rel = np.median(errs, 1) + (y[iiom] - lk[iiom])   # = median(pred)−lk(test計算可・yは相殺で不使用相当)
        f = np.stack([mds[iiom] / 1000.0, np.full(ms.sum(), np.log1p(n0)),
                      errs.std(1), np.percentile(errs, 75, 1) - np.percentile(errs, 25, 1),
                      np.abs(med_pred_rel) / 10.0,
                      fx_static["pf_ancc_std_1"][iiom], fx_static["pf_ancc_std_4"][iiom],
                      fx_static["gf_std"][iiom], fx_static["v50b_std"][iiom], fx_static["v50b_margin"][iiom],
                      np.ones(ms.sum())], 1)
        Fw.append(f); Yw.append(np.zeros(ms.sum())); Wid += [w] * int(ms.sum())
    P = np.concatenate(Pw); F = np.concatenate(Fw); Wid = np.array(Wid)
    # 注意: P は「誤差」= pred − y。合成誤差 = Σ w·err。ターゲット0の回帰として扱える。
    N, K = P.shape; NF = F.shape[1]
    print(f"rows={N} legs={K} feats={NF}")
    fold_of = {w: f for f in range(5) for w in WELLS[f::5]}
    fid = np.array([fold_of[w] for w in Wid])
    def pooled(e): return float(np.sqrt(np.mean(e ** 2)))
    res = {}
    # ---- 0) 固定重み(基準) ----
    W0 = json.loads(open(f"{ART}/stack7_config.json").read())["weights"]
    w0 = np.array([W0[n] for n in LEGS]); w0 = w0 / w0.sum()
    res["fixed"] = pooled(P @ w0)
    # ---- 1) ridge on legs(非負制約なし, 切片なし, foldごとにfit) ----
    from numpy.linalg import solve
    def ridge_fit(A, b, lam):
        return solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
    # 和1を厳密に: w = w0 + B u(B=和ゼロ基底 7x6)
    B = np.zeros((K, K - 1))
    for j in range(K - 1):
        B[j, j] = 1.0; B[j + 1, j] = -1.0
    w0v = np.full(K, 1.0 / K)
    for lam_tag, lam in (("1e-2", 1e-2 * N), ("1e-4", 1e-4 * N)):
        e_oof = np.zeros(N)
        for f in range(5):
            tr = fid != f; va = ~tr
            A = P[tr]
            G = (A @ B); e0 = A @ w0v
            u = solve(G.T @ G + lam * np.eye(K - 1), -G.T @ e0)
            w_ = w0v + B @ u
            e_oof[va] = P[va] @ w_
        res[f"ridge_{lam_tag}"] = pooled(e_oof)
        if lam_tag == "1e-4":
            print("ridge重み例(最終fold):", np.round(w_, 3))
    # ---- 2) FWLS: w_k(x) = Σ_j V[k,j]·f_j(x)、Σ_k w_k(x)=1 をペナルティで ----
    # FWLS: w(x) = w0 + B·U·f(x)(行ごとに和1が厳密に成立)。ridgeでUを解く。
    NU = (K - 1) * NF
    for lam_tag, lam in (("3e-3", 3e-3 * N), ("3e-2", 3e-2 * N)):
        e_oof = np.zeros(N)
        for f in range(5):
            tr = np.where(fid != f)[0]; va = fid == f
            G = np.zeros((NU, NU)); r_ = np.zeros(NU)
            for c0 in range(0, len(tr), 300000):
                b = tr[c0:c0 + 300000]
                PB = P[b] @ B                                  # (n,K-1)
                Z = (PB[:, :, None] * F[b][:, None, :]).reshape(len(b), NU)
                e0 = P[b] @ w0v
                G += Z.T @ Z; r_ += Z.T @ e0
            u = solve(G + lam * np.eye(NU), -r_)
            U = u.reshape(K - 1, NF)
            wva = w0v[None, :] + (F[va] @ U.T) @ B.T           # (n,K)
            e_oof[va] = (P[va] * wva).sum(1)
        res[f"fwls_{lam_tag}"] = pooled(e_oof)
    # ---- 3) softmaxゲート(L5方式の7脚版, torch) ----
    import torch, torch.nn as nnm
    for hid_tag, hid in (("lin", 0), ("mlp16", 16)):
        e_oof = np.zeros(N)
        for f in range(5):
            tr = np.where(fid != f)[0]; va = np.where(fid == f)[0]
            Ft = torch.from_numpy(F[tr].astype(np.float32)); Pt = torch.from_numpy(P[tr].astype(np.float32))
            if hid == 0:
                net = nnm.Linear(NF, K)
            else:
                net = nnm.Sequential(nnm.Linear(NF, hid), nnm.GELU(), nnm.Linear(hid, K))
            for p_ in net.parameters(): nnm.init.zeros_(p_) if p_.dim() == 1 else nnm.init.normal_(p_, 0, 0.01)
            # バイアスを固定重みのlogで初期化(=固定ブレンドから出発)
            _wf = np.array([max(json.loads(open(f"{ART}/stack7_config.json").read())["weights"][n], 1e-3) for n in LEGS])
            _bias = torch.log(torch.tensor(_wf / _wf.sum(), dtype=torch.float32))
            with torch.no_grad():
                (net if hid == 0 else net[-1]).bias.copy_(_bias)
            opt = torch.optim.Adam(net.parameters(), lr=0.01)
            idx = np.arange(len(tr))
            rng = np.random.default_rng(f)
            for ep in range(60):
                rng.shuffle(idx)
                for c0 in range(0, len(idx), 262144):
                    b = idx[c0:c0 + 262144]
                    w_ = torch.softmax(net(Ft[b]), -1)
                    loss = ((Pt[b] * w_).sum(1) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                wva = torch.softmax(net(torch.from_numpy(F[va].astype(np.float32))), -1).numpy()
            e_oof[va] = (P[va] * wva).sum(1)
        res[f"gate_{hid_tag}"] = pooled(e_oof)
        # fold別
        print(f"gate_{hid_tag} fold別:", [round(pooled(e_oof[fid == f2]), 4) for f2 in range(5)])
    print("\n=== メタスタッカー比較(メタOOF, 全773井) ===")
    for k, v in sorted(res.items(), key=lambda t: t[1]):
        print(f"  {k:12s}: {v:.4f}")
    print(f"  (oracle井戸別 4.891 / 行別 4.400)")
    json.dump(res, open(f"{ART}/metastack_result.json", "w"), indent=1)

if __name__ == "__main__":
    main()
