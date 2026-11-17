# -*- coding: utf-8 -*-
"""多近傍 ANCC面 予測器 — 近傍(他train坑井)のANCC点群をIDW補間して面を組み、既知帯で b較正 → TVT=面−Z。
コピー(単一最近傍)の near-oracle を多近傍IDWでどこまで広げられるか。leave-own-well-out(自坑井のANCCは除外=合法・非リーク)。
比較: hold / pf_ancc_2 / 既存 tvtF_ANCC(v56面特徴)。evalしきい値=最近傍距離。
出力: コンソール + artifacts_v69_2/neighbor_surface_v69_2.png。"""
import io, sys, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = PROJ / "rogii-wellbore-geology-prediction" / "train"
ART = PROJ / "v69" / "artifacts_v69_2"
CACHE = ART / "wells_xyz_ancc_cache.pkl"
SUB = 12      # 面点群の行間引き
K = 12        # IDW近傍点数
EPS = 1.0


def load():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    W = {}
    for i, p in enumerate(sorted(glob.glob(str(DATA / "*__horizontal_well.csv")))):
        wid = Path(p).stem.replace("__horizontal_well", "")
        cols = pd.read_csv(p, nrows=0).columns
        use = [c for c in ["MD", "X", "Y", "Z", "TVT_input", "TVT", "ANCC"] if c in cols]
        df = pd.read_csv(p, usecols=use)
        if "TVT" not in df or "ANCC" not in df or df["TVT"].isna().all(): continue
        W[wid] = {k.lower(): df[k].to_numpy(float) for k in use}
        if (i + 1) % 150 == 0: print(f"  load {i+1}", flush=True)
    CACHE.write_bytes(pickle.dumps(W)); return W


def main():
    W = load(); wids = list(W)
    # 面点群(全坑井のANCCサンプル, 間引き) + wid index
    xs, ys, av, wi = [], [], [], []
    for k, w in enumerate(wids):
        d = W[w]; a = d.get("ancc")
        if a is None: continue
        m = np.isfinite(d["x"]) & np.isfinite(d["y"]) & np.isfinite(a)
        idx = np.where(m)[0][::SUB]
        xs.append(d["x"][idx]); ys.append(d["y"][idx]); av.append(a[idx]); wi.append(np.full(len(idx), k))
    CX, CY, CA, CW = map(np.concatenate, (xs, ys, av, wi))
    tree = cKDTree(np.c_[CX, CY])
    print(f"面点群 {len(CX)} 点 / {len(wids)} 坑井  (SUB={SUB}, K={K})", flush=True)

    import pyarrow.parquet as pq
    pcols = ["well", "target", "last_known_tvt", "pf_ancc_2"] + (["tvtF_ANCC"])
    tb = pq.read_table(ART / "data" / "train.parquet", columns=pcols).to_pandas()
    tb["true"] = tb["last_known_tvt"] + tb["target"]
    g = tb.groupby("well")
    pf_se = g.apply(lambda x: float(np.sum((x["pf_ancc_2"] - x["true"]) ** 2))); pf_n = g.size()
    tvf_se = g.apply(lambda x: float(np.nansum((x["tvtF_ANCC"] - x["true"]) ** 2)))

    def surf_pred(qx, qy, self_k):
        dd, ii = tree.query(np.c_[qx, qy], k=K + 25, workers=-1)
        out = np.full(len(qx), np.nan)
        for r in range(len(qx)):
            jj = ii[r]; keep = CW[jj] != self_k
            jk = jj[keep][:K]; dk = dd[r][keep][:K]
            if len(jk) < 3: continue
            wgt = 1.0 / (dk + EPS)
            out[r] = float(np.sum(wgt * CA[jk]) / np.sum(wgt))
        return out

    rows = []
    cent = {w: (np.nanmean(W[w]["x"]), np.nanmean(W[w]["y"])) for w in wids}
    ctree = cKDTree(np.array([cent[w] for w in wids]))
    for k, w in enumerate(wids):
        d = W[w]; ev = ~np.isfinite(d["tvt_input"]); kn = np.isfinite(d["tvt_input"])
        if ev.sum() < 5 or kn.sum() < 5: continue
        dd, ii = ctree.query(cent[w], k=2); ncent = float(np.hypot(*(np.array(cent[w]) - np.array(cent[wids[ii[1]]]))))
        kx, ky, kz, ktrue = (d["x"][kn], d["y"][kn], d["z"][kn], d["tvt"][kn])
        ex, ey, ez, etrue = (d["x"][ev], d["y"][ev], d["z"][ev], d["tvt"][ev])
        fk = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(kz) & np.isfinite(ktrue)
        fe = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez) & np.isfinite(etrue)
        if fk.sum() < 5 or fe.sum() < 5: continue
        Akn = surf_pred(kx[fk], ky[fk], k); Aev = surf_pred(ex[fe], ey[fe], k)
        okk = np.isfinite(Akn); oke = np.isfinite(Aev)
        if okk.sum() < 5 or oke.sum() < 5: continue
        # b較正(既知帯): TVT = ANCC - Z + b → b = mean(TVT - (ANCC - Z))
        b = float(np.mean(ktrue[fk][okk] - (Akn[okk] - kz[fk][okk])))
        pred = Aev[oke] - ez[fe][oke] + b
        err = pred - etrue[fe][oke]
        rows.append(dict(well=w, ncent=ncent, n=int(oke.sum()),
                         surf_sumse=float(np.sum(err ** 2)),
                         hold_sumse=float(np.sum((ktrue[fk][-1] - etrue[fe][oke]) ** 2))))
    R = pd.DataFrame(rows).set_index("well")
    R["pf_se"] = pf_se.reindex(R.index); R["pf_n"] = pf_n.reindex(R.index)
    R["tvf_se"] = tvf_se.reindex(R.index)
    print(f"評価 {len(R)} 坑井", flush=True)

    def pool(se, n):
        m = np.isfinite(se) & np.isfinite(n) & (n > 0); return float(np.sqrt(se[m].sum() / n[m].sum())) if m.any() else np.nan

    print("\n=== 最近傍(重心)しきい値ごとの overall pooled RMSE ===")
    print(f"  {'閾値':>7} {'坑井':>5} {'hold':>7} {'surf(近傍面)':>11} {'tvtF_ANCC':>10} {'pf_ancc_2':>10}")
    sweep = {}
    for thr in [150, 300, 500, 800, 1500, 1e9]:
        s = R[R["ncent"] < thr]
        vals = (pool(s["hold_sumse"].to_numpy(), s["n"].to_numpy()),
                pool(s["surf_sumse"].to_numpy(), s["n"].to_numpy()),
                pool(s["tvf_se"].to_numpy(), s["pf_n"].to_numpy()),
                pool(s["pf_se"].to_numpy(), s["pf_n"].to_numpy()))
        sweep[thr] = vals
        print(f"  {('全体' if thr>1e8 else '<'+str(int(thr))):>7} {len(s):>5} {vals[0]:>7.3f} {vals[1]:>11.3f} {vals[2]:>10.3f} {vals[3]:>10.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    thrs = [150, 300, 500, 800, 1500]
    for mi, lab in [(0, "hold"), (1, "surf(近傍面)"), (2, "tvtF_ANCC(既存)"), (3, "pf_ancc_2")]:
        ax[0].plot(range(len(thrs)), [sweep[t][mi] for t in thrs], "o-", label=lab)
    ax[0].set_xticks(range(len(thrs))); ax[0].set_xticklabels([f"<{t}" for t in thrs])
    ax[0].set_xlabel("最近傍距離しきい値 (ft)"); ax[0].set_ylabel("overall pooled RMSE"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("近傍ANCC面 vs 既存tvtF_ANCC vs PF")
    # per-well surf vs pf 散布
    rk = lambda s, c: np.sqrt(R[s] / R[c])
    ax[1].scatter(rk("pf_se", "pf_n"), rk("surf_sumse", "n"), s=12, c=np.log10(R["ncent"].clip(30, 3000)), cmap="viridis")
    ax[1].plot([0, 30], [0, 30], "r--", alpha=0.6); ax[1].set_xlim(0, 25); ax[1].set_ylim(0, 25)
    ax[1].set_xlabel("PF per-well RMSE"); ax[1].set_ylabel("近傍面 per-well RMSE")
    ax[1].set_title("坑井別: 対角下=面がPFに勝ち (色=近傍距離log)"); ax[1].grid(alpha=0.3)
    fig.suptitle("多近傍ANCC面 予測 — artifacts_v69_2"); fig.tight_layout()
    fig.savefig(ART / "neighbor_surface_v69_2.png", dpi=115)
    print(f"\n[saved] {ART/'neighbor_surface_v69_2.png'}")
    win = int((rk("surf_sumse", "n") < rk("pf_se", "pf_n")).sum())
    print(f"面がPFに勝つ坑井: {win}/{len(R)}")


if __name__ == "__main__":
    main()
