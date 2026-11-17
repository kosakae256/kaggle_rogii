# v74 診断3 — 井戸オフセット(structの主誤差7.73ft)を GRフリー観測で当てられるか
#   offset_true_w = median_eval(struct_tvt − TVT)
#   offset_hat_w  = median_eval(struct_tvt − v56_surf_tvt)   ※v56面(近傍レベル, prefix錨b̂)との食い違い
#   補正: struct_tvt − λ·offset_hat_w。λ グローバル走査 + 距離帯別 λ*(=cov/var, 参考の上限)
# 入力: v38 prior pkl(struct trace) + v56/artifacts_v56/lowo_rows.npz(tvt_surf,dist,slope,fmad)
# 出力: v74/artifacts_v74/diag3_offset_obs.{json,png}
import os, json, pickle
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
PKL = os.path.join(P0, "v38", "artifacts_v38", "priors", "prior_v66_leanv1cumul_train.pkl")
NPZ = os.path.join(P0, "v56", "artifacts_v56", "lowo_rows.npz")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
LAMS = np.round(np.arange(0.0, 1.01, 0.05), 2)

def main():
    with open(PKL, "rb") as f:
        struct = pickle.load(f)
    lowo = np.load(NPZ)
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    W = []       # per-well: off_true, off_hat, dist_med, n_ev
    ERR = []     # per-well struct誤差配列(補正RMSEの厳密再計算用)
    nskip = 0
    for w in wells:
        sp = struct.get(w)
        if sp is None or w not in lowo.files:
            nskip += 1; continue
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["TVT", "TVT_input"])
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        evm = ~np.isfinite(tin)
        s_tvt = np.asarray(sp["tvt"], float)
        L = lowo[w].astype(float)                     # (n_ev,4): tvt_surf, dist, slope, fmad
        if len(s_tvt) != int(evm.sum()) or len(L) != len(s_tvt):
            nskip += 1; continue
        tvt_ev = tvt[evm]
        ok = np.isfinite(tvt_ev) & np.isfinite(s_tvt)
        if ok.sum() < 50:
            nskip += 1; continue
        surf = L[:, 0]; dist = L[:, 1]
        oks = ok & np.isfinite(surf)
        e = (s_tvt - tvt_ev)[ok]
        off_true = float(np.median(e))
        if oks.sum() >= 50:
            off_hat = float(np.median((s_tvt - surf)[oks]))
            dmed = float(np.median(dist[oks]))
        else:
            off_hat = 0.0; dmed = 1e9
        W.append(dict(well=w, off_true=off_true, off_hat=off_hat,
                      dmed=dmed, n=int(ok.sum())))
        ERR.append(e)
    print(f"wells={len(W)} (skip {nskip})")
    ot = np.array([x["off_true"] for x in W]); oh = np.array([x["off_hat"] for x in W])
    dm = np.array([x["dmed"] for x in W]); ns = np.array([x["n"] for x in W])

    def rmse_pool(lam_per_well):
        return float(np.sqrt(np.mean(np.concatenate(
            [ERR[i] - lam_per_well[i] * oh[i] for i in range(len(W))]) ** 2)))
    base = rmse_pool(np.zeros(len(W)))
    # 相関(井戸重み=行数)
    def wcorr(a, b, w):
        aw = a - np.average(a, weights=w); bw = b - np.average(b, weights=w)
        return float(np.average(aw * bw, weights=w) /
                     np.sqrt(np.average(aw ** 2, weights=w) * np.average(bw ** 2, weights=w)))
    rho = wcorr(oh, ot, ns)
    print(f"struct as-is(整列済 {len(W)}井) = {base:.4f}")
    print(f"corr(off_hat, off_true) 行数重み = {rho:.3f}  (未重み {np.corrcoef(oh, ot)[0,1]:.3f})")
    # グローバル λ 走査
    curve = [rmse_pool(np.full(len(W), lam)) for lam in LAMS]
    i = int(np.argmin(curve))
    print(f"グローバル λ*={LAMS[i]:.2f}: {curve[i]:.4f}  (λ=1: {curve[-1]:.4f})")
    # 距離帯別 λ*(cov/var, 行数重み — in-sample参考値)
    bands = [(0, 300), (300, 800), (800, 1500), (1500, 3000), (3000, 1e18)]
    lam_band = np.zeros(len(W)); band_info = []
    for a, b in bands:
        m = (dm >= a) & (dm < b)
        if m.sum() >= 5:
            wgt = ns[m]
            ohm = oh[m] - 0.0; otm = ot[m]
            lam = float(np.average(ohm * otm, weights=wgt) /
                        max(np.average(ohm ** 2, weights=wgt), 1e-9))
            lam = float(np.clip(lam, 0.0, 1.0))
        else:
            lam = 0.0
        lam_band[m] = lam
        band_info.append(dict(lo=a, hi=min(b, 1e9), n=int(m.sum()), lam=lam,
                              rho=wcorr(oh[m], ot[m], ns[m]) if m.sum() >= 5 else None))
    e_band = rmse_pool(lam_band)
    print("距離帯別 λ*:", [(bi["lo"], bi["n"], round(bi["lam"], 2),
                          None if bi["rho"] is None else round(bi["rho"], 2)) for bi in band_info])
    print(f"距離帯別補正(in-sample) = {e_band:.4f}")
    # oracle: off_true を完全に当てた場合(=diag2のdemean)
    e_orc = rmse_pool((ot / np.where(np.abs(oh) > 1e-12, oh, 1e9)) * 0 + 0)  # dummy
    e_orc = float(np.sqrt(np.mean(np.concatenate(
        [ERR[i] - ot[i] for i in range(len(W))]) ** 2)))
    print(f"oracle(off_true除去) = {e_orc:.4f}")
    res = dict(n_wells=len(W), base=base, rho_weighted=rho,
               lam_global=float(LAMS[i]), best_global=curve[i], lam1=curve[-1],
               band_info=band_info, best_band=e_band, oracle=e_orc,
               lams=list(map(float, LAMS)), curve=list(map(float, curve)))
    with open(os.path.join(ART, "diag3_offset_obs.json"), "w", encoding="utf-8") as f:
        json.dump(dict(res, per_well=W), f, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    sc = ax[0].scatter(oh, ot, c=np.clip(dm, 0, 3000), s=10, cmap="viridis")
    ax[0].plot([-40, 40], [-40, 40], "r--", lw=0.7)
    ax[0].set_xlim(-40, 40); ax[0].set_ylim(-40, 40)
    ax[0].set_xlabel("off_hat = median(struct − v56面) (ft)")
    ax[0].set_ylabel("off_true = median(struct − 真TVT) (ft)")
    ax[0].set_title(f"井戸オフセット: 観測 vs 真値 (重み付きρ={rho:.3f})")
    plt.colorbar(sc, ax=ax[0], label="近傍距離 med (ft)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(LAMS, curve, marker="o", ms=3)
    ax[1].axhline(base, color="gray", ls="--", lw=0.8, label=f"struct as-is {base:.3f}")
    ax[1].axhline(e_band, color="tab:green", ls=":", lw=1.2, label=f"距離帯λ* {e_band:.3f}")
    ax[1].axhline(e_orc, color="tab:red", ls=":", lw=1.2, label=f"oracle {e_orc:.3f}")
    ax[1].set_xlabel("λ (グローバル補正ゲイン)"); ax[1].set_ylabel("overall RMSE (ft)")
    ax[1].set_title("struct − λ·off_hat の overall RMSE")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("v74 診断3 — 井戸オフセットのGRフリー観測(v56近傍面レベル)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "diag3_offset_obs.png"), dpi=130)
    print("saved:", os.path.join(ART, "diag3_offset_obs.png"))

if __name__ == "__main__":
    main()
