# -*- coding: utf-8 -*-
"""近傍コピー(フォーラム助言)を「近傍を持つ対象坑井だけ」で試す。
方式(v56の TVT≒構造面−Z 恒等式に沿う):
  最近傍train坑井Nの (X,Y)→値 を KDTree で対象の各eval点(X,Y)に対応付け、開始点(last-known)で定数シフト整列。
  - simple  : TVT_pred = TVT_N(nn) + δ                 (TVTをそのままコピー)
  - struct  : TVT_pred = Z_t - Z_N(nn) + TVT_N(nn) + δ (深さ差を補正=構造面 S=Z-TVT を共有と仮定)
  - struct2 : TVT_pred = TVT_N(nn) + Z_N(nn) - Z_t + δ (逆符号 S=Z+TVT)
評価 = overall pooled RMSE。近傍距離(重心XY)閾値で層別。baseline=hold(=last_known) と pf_ancc_2(parquet)。
出力: コンソール表 + artifacts_v69_2/neighbor_copy_v69_2.png(Meiryo)。"""
import io, sys, glob, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA = PROJ / "rogii-wellbore-geology-prediction" / "train"
ART = PROJ / "v69" / "artifacts_v69_2"
CACHE = ART / "wells_xyz_cache.pkl"


def load_wells():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    W = {}
    for i, p in enumerate(sorted(glob.glob(str(DATA / "*__horizontal_well.csv")))):
        wid = Path(p).stem.replace("__horizontal_well", "")
        df = pd.read_csv(p, usecols=["MD", "X", "Y", "Z", "TVT_input", "TVT"])
        if "TVT" not in df or df["TVT"].isna().all():
            continue
        W[wid] = dict(md=df["MD"].to_numpy(float), x=df["X"].to_numpy(float), y=df["Y"].to_numpy(float),
                      z=df["Z"].to_numpy(float), tvt=df["TVT"].to_numpy(float),
                      tvti=df["TVT_input"].to_numpy(float))
        if (i + 1) % 100 == 0: print(f"  load {i+1}", flush=True)
    CACHE.write_bytes(pickle.dumps(W)); return W


def main():
    W = load_wells()
    wids = list(W)
    cent = np.array([[np.nanmean(W[w]["x"]), np.nanmean(W[w]["y"])] for w in wids])
    tree = cKDTree(cent)
    print(f"坑井 {len(wids)} 本")

    # pf_ancc_2 の per-well sumse(参照)
    import pyarrow.parquet as pq
    tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "last_known_tvt", "pf_ancc_2"]).to_pandas()
    tb["true"] = tb["last_known_tvt"] + tb["target"]; tb["se"] = (tb["pf_ancc_2"] - tb["true"]) ** 2
    pf_g = tb.groupby("well")["se"].agg(["sum", "count"])

    rows = []
    for wi, w in enumerate(wids):
        d = W[w]; ev = ~np.isfinite(d["tvti"]); kn = np.isfinite(d["tvti"])
        if ev.sum() < 5 or kn.sum() < 3: continue
        # 最近傍(自分除く, 重心距離)
        dd, ii = tree.query(cent[wi], k=2)
        nb = wids[ii[1]]; dist = float(dd[1]); N = W[nb]
        okN = np.isfinite(N["x"]) & np.isfinite(N["y"]) & np.isfinite(N["tvt"]) & np.isfinite(N["z"])
        if okN.sum() < 5: continue
        Ntree = cKDTree(np.c_[N["x"][okN], N["y"][okN]])
        Ntvt = N["tvt"][okN]; Nz = N["z"][okN]

        def nn(qx, qy):
            _, j = Ntree.query(np.c_[qx, qy]); return j
        ex, ey, ez, etrue = d["x"][ev], d["y"][ev], d["z"][ev], d["tvt"][ev]
        fin = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez) & np.isfinite(etrue)
        ex, ey, ez, etrue = ex[fin], ey[fin], ez[fin], etrue[fin]
        if len(ex) < 5: continue
        je = nn(ex, ey)
        # last-known(整列基準)
        kidx = np.where(kn)[0][-1]
        lkx, lky, lkz, lktvt = d["x"][kidx], d["y"][kidx], d["z"][kidx], d["tvt"][kidx]
        if not np.isfinite(lktvt): lktvt = d["tvt"][kn][np.isfinite(d["tvt"][kn])][-1]
        jl = nn(np.array([lkx]), np.array([lky]))[0]
        preds = {}
        # simple
        raw = Ntvt[je]; dlt = lktvt - Ntvt[jl]; preds["simple"] = raw + dlt
        # struct S=Z-TVT
        raw = ez - Nz[je] + Ntvt[je]; raw_lk = lkz - Nz[jl] + Ntvt[jl]; preds["struct"] = raw + (lktvt - raw_lk)
        # struct2 S=Z+TVT
        raw = Ntvt[je] + Nz[je] - ez; raw_lk = Ntvt[jl] + Nz[jl] - lkz; preds["struct2"] = raw + (lktvt - raw_lk)
        preds["hold"] = np.full(len(ex), lktvt)
        rec = dict(well=w, dist=dist, n=len(ex))
        for k, pr in preds.items():
            rec[k] = float(np.sum((pr - etrue) ** 2))
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("well")
    R["pf_se"] = pf_g["sum"].reindex(R.index); R["pf_n"] = pf_g["count"].reindex(R.index)
    print(f"\n評価対象 {len(R)} 坑井")

    def pooled(se, n):
        m = np.isfinite(se) & np.isfinite(n) & (n > 0)
        return float(np.sqrt(se[m].sum() / n[m].sum())) if m.any() else float("nan")

    METH = ["hold", "simple", "struct", "struct2"]
    print("\n=== 近傍距離しきい値ごとの overall pooled RMSE(対象=最近傍<thr の坑井) ===")
    print(f"  {'閾値ft':>7} {'坑井':>5} " + " ".join(f"{m:>8}" for m in METH) + f" {'pf_2':>8}")
    sweep = {}
    for thr in [150, 300, 500, 800, 1500, 1e9]:
        sub = R[R["dist"] < thr]
        vals = [pooled(sub[m].to_numpy(), sub["n"].to_numpy()) for m in METH]
        pfv = pooled(sub["pf_se"].to_numpy(), sub["pf_n"].to_numpy())
        sweep[thr] = vals + [pfv]
        lab = "全体" if thr > 1e8 else f"<{int(thr)}"
        print(f"  {lab:>7} {len(sub):>5} " + " ".join(f"{v:>8.3f}" for v in vals) + f" {pfv:>8.3f}")

    # グラフ
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    thrs = [150, 300, 500, 800, 1500]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(thrs))
    for mi, m in enumerate(["hold", "simple", "struct", "pf"]):
        key = m if m != "pf" else None
        ys = [sweep[t][METH.index(m)] if m in METH else sweep[t][-1] for t in thrs]
        ax[0].plot(x, ys, "o-", label={"hold": "hold(錨保持)", "simple": "copy simple", "struct": "copy struct(Z補正)", "pf": "pf_ancc_2"}[m])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"<{t}" for t in thrs]); ax[0].set_xlabel("最近傍距離しきい値 (ft)")
    ax[0].set_ylabel("overall pooled RMSE (ft)"); ax[0].set_title("近傍コピー vs hold vs PF(近い坑井ほど左)")
    ax[0].grid(alpha=0.3); ax[0].legend()
    # 近傍が最も近い数坑井の例(struct copy vs true vs hold)
    ex_wells = R.sort_values("dist").index[:3]
    for k, w in enumerate(ex_wells):
        d = W[w]; ev = ~np.isfinite(d["tvti"]);
        ax[1].plot(d["md"][ev], d["tvt"][ev], color=f"C{k}", lw=2, label=f"{w[:8]} 真({R.loc[w,'dist']:.0f}ft)")
    ax[1].set_xlabel("MD"); ax[1].set_ylabel("TVT"); ax[1].set_title("最近傍が近い3坑井の真値TVT(曲率の様子)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle("近傍コピー予測(対象=近傍あり坑井) — artifacts_v69_2")
    fig.tight_layout(); fig.savefig(ART / "neighbor_copy_v69_2.png", dpi=115)
    print(f"\n[saved] {ART/'neighbor_copy_v69_2.png'}")
    print("\n=== 近い順 上位8坑井(距離/hold/simple/struct RMSE) ===")
    for w in R.sort_values("dist").index[:8]:
        r = R.loc[w]
        f = lambda k: np.sqrt(r[k] / r["n"])
        print(f"  {w[:8]} dist={r['dist']:6.0f}ft n={int(r['n']):5d}  hold={f('hold'):6.2f}  simple={f('simple'):6.2f}  struct={f('struct'):6.2f}  struct2={f('struct2'):6.2f}")


if __name__ == "__main__":
    main()
