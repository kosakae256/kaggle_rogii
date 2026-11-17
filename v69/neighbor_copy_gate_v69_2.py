# -*- coding: utf-8 -*-
"""近傍コピー採否ゲート — 「コピーが当たる坑井だけ」copy、他はPF、の選択予測。
コピー式 = struct2(v56面転写): TVT = (TVT_N + Z_N) - Z_target + δ。δ=既知帯で整列(mean残差)。
★推論時に使える信号だけでゲート:
  - rmse_known : 既知帯に同じコピーを当てた残差RMSE(構造面がこの坑井に合うか。真値TVT不要=TVT_inputで足りる)
  - p90_path   : eval点から近傍経路への最近傍距離の90%点(外挿でないか)
  - d_cent     : 重心距離(弱)
評価: 全773で pred = (ゲート成立→copy / 否→pf_ancc_2)。純PFと overall pooled RMSE 比較。
出力: コンソール + artifacts_v69_2/neighbor_copy_gate_v69_2.png。"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v69" / "artifacts_v69_2"
W = pickle.loads((ART / "wells_xyz_cache.pkl").read_bytes())
wids = list(W)
cent = np.array([[np.nanmean(W[w]["x"]), np.nanmean(W[w]["y"])] for w in wids])
tree = cKDTree(cent)

# pf_ancc_2 per-well 誤差(参照 & 非ゲート坑井の予測元)
import pyarrow.parquet as pq
tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "last_known_tvt", "pf_ancc_2"]).to_pandas()
tb["true"] = tb["last_known_tvt"] + tb["target"]; tb["se"] = (tb["pf_ancc_2"] - tb["true"]) ** 2
pf_sumse = tb.groupby("well")["se"].sum(); pf_n = tb.groupby("well")["se"].count()


def rms(e): e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan


rows = []
for wi, w in enumerate(wids):
    d = W[w]; ev = ~np.isfinite(d["tvti"]); kn = np.isfinite(d["tvti"])
    if ev.sum() < 5 or kn.sum() < 5: continue
    _, ii = tree.query(cent[wi], k=2); nb = wids[ii[1]]; dc = float(np.hypot(*(cent[wi] - cent[ii[1]])))
    N = W[nb]; okN = np.isfinite(N["x"]) & np.isfinite(N["y"]) & np.isfinite(N["tvt"]) & np.isfinite(N["z"])
    if okN.sum() < 5: continue
    Ntree = cKDTree(np.c_[N["x"][okN], N["y"][okN]]); surfN = (N["tvt"] + N["z"])[okN]
    kx, ky, kz, ktrue = d["x"][kn], d["y"][kn], d["z"][kn], d["tvt"][kn]
    ex, ey, ez, etrue = d["x"][ev], d["y"][ev], d["z"][ev], d["tvt"][ev]
    fk = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(kz) & np.isfinite(ktrue)
    fe = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez) & np.isfinite(etrue)
    if fk.sum() < 5 or fe.sum() < 5: continue
    kx, ky, kz, ktrue = kx[fk], ky[fk], kz[fk], ktrue[fk]; ex, ey, ez, etrue = ex[fe], ey[fe], ez[fe], etrue[fe]
    dk, jk = Ntree.query(np.c_[kx, ky]); de, je = Ntree.query(np.c_[ex, ey])
    raw_k = surfN[jk] - kz; raw_e = surfN[je] - ez
    delta = float(np.mean(ktrue - raw_k))                    # 既知帯で整列(mean)
    rmse_known = rms(raw_k + delta - ktrue)                  # ★ゲート信号(推論安全)
    err_e = (raw_e + delta) - etrue                          # eval誤差(評価用)
    rows.append(dict(well=w, d_cent=dc, p90_path=float(np.percentile(de, 90)),
                     med_path=float(np.median(de)), rmse_known=rmse_known,
                     rmse_eval=rms(err_e), copy_sumse=float(np.sum(err_e ** 2)), n=len(ex),
                     pf_sumse=float(pf_sumse.get(w, np.nan)), pf_n=float(pf_n.get(w, np.nan))))
R = pd.DataFrame(rows).set_index("well")
print(f"近傍あり評価対象 {len(R)} 坑井")

# ゲート信号が eval品質を予測するか
c = R[["rmse_known", "rmse_eval", "p90_path", "d_cent"]].corr()["rmse_eval"]
print(f"\n=== rmse_eval との相関 ===  rmse_known={c['rmse_known']:.3f}  p90_path={c['p90_path']:.3f}  d_cent={c['d_cent']:.3f}")


def pooled(sub_copy_wells):
    """copy_wells集合はcopy誤差、それ以外(近傍ありでゲート外)＋近傍なし は pf。全773でpooled。"""
    se = pf_sumse.copy().astype(float); nn = pf_n.copy().astype(float)      # 既定=全部pf
    for w in sub_copy_wells:
        se[w] = R.loc[w, "copy_sumse"]; nn[w] = R.loc[w, "n"]              # ゲート坑井のみcopyに差替
    m = np.isfinite(se) & np.isfinite(nn) & (nn > 0)
    return float(np.sqrt(se[m].sum() / nn[m].sum()))


pf_pool = float(np.sqrt(pf_sumse.sum() / pf_n.sum()))
print(f"\n純PF(pf_ancc_2) 全773 pooled RMSE = {pf_pool:.4f}")

# ゲート掃引: rmse_known < T1 かつ p90_path < T2
print("\n=== ゲート掃引: [rmse_known<T1 & p90_path<T2] の坑井を copy 差替 → 全773 pooled ===")
print(f"  {'T1(known)':>9} {'T2(path)':>8} {'#gate':>6} {'gate_copy':>9} {'gate_pf':>8} {'全773pool':>9} {'vs純PF':>8}")
best = None
for T2 in [100, 150, 250]:
    for T1 in [2, 3, 4, 5, 7, 10]:
        g = R[(R["rmse_known"] < T1) & (R["p90_path"] < T2)]
        if len(g) == 0: continue
        gc = float(np.sqrt(g["copy_sumse"].sum() / g["n"].sum()))
        gp = float(np.sqrt(g["pf_sumse"].sum() / g["pf_n"].sum()))
        pool = pooled(list(g.index)); delta = pool - pf_pool
        print(f"  {T1:>9} {T2:>8} {len(g):>6} {gc:>9.3f} {gp:>8.3f} {pool:>9.4f} {delta:>+8.4f}")
        if best is None or pool < best[0]: best = (pool, T1, T2, len(g), gc, gp)
print(f"\n★最良ゲート: T1(known)<{best[1]} & p90_path<{best[2]}  → {best[3]}坑井 copy差替  "
      f"gate上 copy={best[4]:.3f} vs pf={best[5]:.3f}  全773 {best[0]:.4f}(純PF {pf_pool:.4f}, {best[0]-pf_pool:+.4f})")

# グラフ
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
sc = ax[0].scatter(R["rmse_known"], R["rmse_eval"], c=np.clip(R["p90_path"], 0, 400), s=14, cmap="viridis")
ax[0].plot([0, 20], [0, 20], "r--", lw=1, alpha=0.6); ax[0].set_xlim(0, 20); ax[0].set_ylim(0, 30)
ax[0].axvline(best[1], color="k", ls=":", alpha=0.7)
ax[0].set_xlabel("rmse_known(ゲート信号=既知帯コピー残差)"); ax[0].set_ylabel("rmse_eval(実コピー精度)")
ax[0].set_title(f"既知帯残差は eval精度を予測(相関{c['rmse_known']:.2f})"); fig.colorbar(sc, ax=ax[0], label="p90_path(ft)")
# ゲート人数 vs 全773 pooled
xs, ys, ns = [], [], []
for T1 in np.arange(1, 12, 0.5):
    g = R[(R["rmse_known"] < T1) & (R["p90_path"] < best[2])]
    xs.append(len(g)); ys.append(pooled(list(g.index)))
ax[1].plot(xs, ys, "o-"); ax[1].axhline(pf_pool, color="gray", ls="--", label=f"純PF {pf_pool:.3f}")
ax[1].set_xlabel("copyに差し替えた坑井数"); ax[1].set_ylabel("全773 overall pooled RMSE")
ax[1].set_title(f"選択予測(copy/PF) の全体RMSE  (p90_path<{best[2]})"); ax[1].grid(alpha=0.3); ax[1].legend()
fig.suptitle("近傍コピー採否ゲート — artifacts_v69_2")
fig.tight_layout(); fig.savefig(ART / "neighbor_copy_gate_v69_2.png", dpi=115)
print(f"\n[saved] {ART/'neighbor_copy_gate_v69_2.png'}")
