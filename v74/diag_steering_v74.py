# v74 診断0(v2) — 操舵信号のスケール分解: どの波長帯で軌道Zが面gを運ぶか(GRフリー go/no-go)
#   g = TVT + Z (= ANCC + b, v56確定事実)。評価ゾーンの Z は test でも全行既知(合法観測)。
#   スモーク(v1)の知見: corr(HP300(Z),HP300(g))=0.40 だが高周波の振幅は小(0.56ft)。
#   hold(=Z完全追従)11.8 ≪ dip線形外挿45.8 → 勝負は「Zをどのスケールまで追うか」。
# モデル族: pred_g(W,κ) = [低周波] + κ·HP_W(Z)
#   oracle 低周波 = LP_W(g)真値      → 「低周波を他手段(近傍/ML)で解けた時の天井」
#   legal  低周波 = c_w + LP_W(Z)    → hold錨(c_w=既知帯末尾で整合)。κ=1 で hold と一致。
# 出力: v74/artifacts_v74/diag_steering.{json,png}
# env: NW(0=全), LAGMAX(既定150), TAILC(錨に使う既知帯末尾行数, 既定100)
import os, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
LAGMAX = int(os.environ.get("LAGMAX", "150"))
TAILC = int(os.environ.get("TAILC", "100"))
WINS = [101, 301, 1001, 2001, 4001]
KS = np.round(np.arange(0.0, 1.41, 0.05), 2)

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def xcorr_bestlag(a, b, lagmax, step=5):
    best, bl = -2.0, 0
    n = len(a)
    for L in range(-lagmax, lagmax + 1, step):
        if L >= 0:
            x, y = a[L:], b[:n - L]
        else:
            x, y = a[:n + L], b[-L:]
        if len(x) < 50 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, bl = c, L
    return best, bl

def main():
    t0 = time.time()
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    per = []
    # スケール別プール成分: 評価行 err(κ) を線形結合で厳密再構成するための貯め込み
    #   oracle: err = -gh + κ·zh          (gh=HP_W(g), zh=HP_W(Z))
    #   legal : err = (u - c(κ)) + κ·zh   (u = LP_W(Z) - g, c(κ)=median_tail(u+κ·zh))
    pool = {W: dict(gh=[], zh=[], u=[], ukn=[], zkn=[]) for W in WINS}
    EH = []   # hold誤差
    nskip = 0
    for wi, w in enumerate(wells):
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "Z", "TVT", "TVT_input"])
        z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            nskip += 1; continue
        g = tvt + z
        ki = np.where(kn)[0]; ei = np.where(ev)[0]
        tail = ki[-TAILC:]
        EH.append(tin[ki[-1]] - tvt[ei])
        rec = dict(well=w, n_ev=int(ev.sum()))
        for W in WINS:
            zl = lowpass(z, W); gl = lowpass(g, W)
            zh = z - zl; gh = g - gl
            u = zl - g
            pool[W]["gh"].append(gh[ei]); pool[W]["zh"].append(zh[ei])
            pool[W]["u"].append(u[ei])
            pool[W]["ukn"].append(u[tail]); pool[W]["zkn"].append(zh[tail])
            if np.std(zh[ei]) > 1e-9 and np.std(gh[ei]) > 1e-9:
                rec[f"corr_{W}"] = float(np.corrcoef(zh[ei], gh[ei])[0, 1])
            else:
                rec[f"corr_{W}"] = 0.0
        bc, bl = xcorr_bestlag(pool[301]["zh"][-1], pool[301]["gh"][-1], LAGMAX)
        rec["bestlag301"] = int(bl)
        per.append(rec)
        if (wi + 1) % 100 == 0:
            print(f"  {wi+1}/{len(wells)}  ({time.time()-t0:.0f}s)", flush=True)
    eh = np.concatenate(EH)
    nw = len(per)
    print(f"wells={nw} (skip {nskip})  eval_rows={len(eh)}  ({time.time()-t0:.0f}s)", flush=True)

    def rmse(e):
        return float(np.sqrt(np.mean(e ** 2)))
    res = dict(n_wells=nw, n_rows=int(len(eh)), rmse_hold=rmse(eh), wins=WINS,
               ks=list(map(float, KS)))
    scale_rows = []
    curves = {}
    for W in WINS:
        gh = np.concatenate(pool[W]["gh"]); zh = np.concatenate(pool[W]["zh"])
        # oracle 曲線
        cur_o = [rmse(-gh + k * zh) for k in KS]
        # legal 曲線(井戸別に錨 c(κ) を既知帯末尾で取る)
        cur_l = []
        for k in KS:
            es = []
            for i in range(nw):
                c = np.median(pool[W]["ukn"][i] + k * pool[W]["zkn"][i])
                es.append((pool[W]["u"][i] - c) + k * pool[W]["zh"][i])
            cur_l.append(rmse(np.concatenate(es)))
        io, il = int(np.argmin(cur_o)), int(np.argmin(cur_l))
        corr_pool = float(np.corrcoef(zh, gh)[0, 1]) if np.std(zh) > 1e-9 else 0.0
        row = dict(W=W, sig_g=rmse(gh), corr_pool=corr_pool,
                   corr_med=float(np.median([p[f"corr_{W}"] for p in per])),
                   oracle_k0=cur_o[0], oracle_best=cur_o[io], oracle_k=float(KS[io]),
                   legal_best=cur_l[il], legal_k=float(KS[il]), legal_k1=cur_l[-9])  # KS[-9]=1.0
        # KS[-9]: κ=1.0 の位置(1.40 まで 0.05 刻み → index of 1.0)
        row["legal_k1"] = cur_l[int(np.where(KS == 1.0)[0][0])]
        scale_rows.append(row)
        curves[W] = dict(oracle=list(map(float, cur_o)), legal=list(map(float, cur_l)))
        print(f"W={W:5d}  sig_g={row['sig_g']:6.3f}  corr={corr_pool:5.2f}  "
              f"oracle κ*={row['oracle_k']:.2f}: {cur_o[0]:6.3f}->{cur_o[io]:6.3f}  "
              f"legal κ*={row['legal_k']:.2f}: {row['legal_k1']:6.3f}(κ=1)->{cur_l[il]:6.3f}",
              flush=True)
    res["scales"] = scale_rows; res["curves"] = curves
    res["bestlag301_median"] = float(np.median([p["bestlag301"] for p in per]))
    with open(os.path.join(ART, "diag_steering.json"), "w", encoding="utf-8") as f:
        json.dump(dict(res, per_well=per), f, ensure_ascii=False, indent=1)

    # --- グラフ(Meiryo)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))
    # (0,0) スケール別 corr ECDF
    for W in WINS:
        cs = np.sort([p[f"corr_{W}"] for p in per])
        ax[0, 0].plot(cs, np.arange(1, len(cs) + 1) / len(cs), label=f"W={W}")
    ax[0, 0].axvline(0, color="gray", lw=0.5)
    ax[0, 0].set_title("井戸別 corr(HP_W(Z), HP_W(g)) @評価ゾーン ECDF")
    ax[0, 0].set_xlabel("corr"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    # (0,1) スケール別 帯域振幅と oracle 回収
    Ws = [r["W"] for r in scale_rows]
    ax[0, 1].plot(Ws, [r["sig_g"] for r in scale_rows], marker="o", label="HP_W(g) 振幅(=oracle κ=0)")
    ax[0, 1].plot(Ws, [r["oracle_best"] for r in scale_rows], marker="s", label="oracle + κ*·HP_W(Z)")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("W (行≈ft)")
    ax[0, 1].set_ylabel("overall RMSE (ft)")
    ax[0, 1].set_title("スケール別: 低周波さえ解ければ残りはどこまで軌道で回収できるか")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)
    # (1,0) legal 曲線
    for W in WINS:
        ax[1, 0].plot(KS, curves[W]["legal"], marker="o", ms=2.5, label=f"W={W}")
    ax[1, 0].axhline(res["rmse_hold"], color="gray", ls="--", lw=0.8,
                     label=f"hold: {res['rmse_hold']:.2f}")
    ax[1, 0].set_title("legal(hold錨 + LP_W(Z) + κ·HP_W(Z)) の overall RMSE vs κ")
    ax[1, 0].set_xlabel("κ"); ax[1, 0].set_ylabel("overall RMSE (ft)")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)
    # (1,1) oracle 曲線
    for W in WINS:
        ax[1, 1].plot(KS, curves[W]["oracle"], marker="o", ms=2.5, label=f"W={W}")
    ax[1, 1].set_title("oracle(真LP_W(g) + κ·HP_W(Z)) の overall RMSE vs κ")
    ax[1, 1].set_xlabel("κ"); ax[1, 1].set_ylabel("overall RMSE (ft)")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)
    fig.suptitle(f"v74 診断0 v2 — 操舵信号のスケール分解 ({nw}井)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "diag_steering.png"), dpi=130)
    print("saved:", os.path.join(ART, "diag_steering.png"))

if __name__ == "__main__":
    main()
