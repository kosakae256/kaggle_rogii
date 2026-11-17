# -*- coding: utf-8 -*-
"""非逐次 TVT予測器(単体評価) — v56恒等式 TVT=ANCC(X,Y)-Z+b を、ANCC列の局所加重平面フィットで。
逐次推定(PF/filter)を一切使わない。各行の(X,Y)で、★自坑井を除いた他train坑井のANCCを局所平面フィット(dip込み)
→ANCC(X,Y)→既知帯でb較正→TVT=ANCC-Z+b。全行独立=非逐次。leave-one-well-out=過適合を避けた正直評価。
評価は「単体でどれくらい出るか」のみ(blend/選択しない)。overall pooled RMSE + 近傍距離別 + 例。
出力: コンソール + artifacts_v69_2/nonseq_ancc_surface.png"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_2")
W = pickle.loads((ART / "wells_xyz_ancc_cache.pkl").read_bytes())    # {wid:{md,x,y,z,tvt,tvti,ancc}}
wids = list(W)
SUB = 6           # 面点群の行間引き
K_FIT = 24        # 局所平面フィットに使う非自坑井近傍数
K_QUERY = 260     # kNNクエリ数(自坑井を弾いて K_FIT 残す余裕)
RIDGE = 1e-3

# --- ANCC点群(全坑井, 間引き) + wid index ---
xs, ys, av, wi = [], [], [], []
for k, w in enumerate(wids):
    d = W[w]; a = d.get("ancc")
    if a is None: continue
    m = np.isfinite(d["x"]) & np.isfinite(d["y"]) & np.isfinite(a)
    idx = np.where(m)[0][::SUB]
    xs.append(d["x"][idx]); ys.append(d["y"][idx]); av.append(a[idx]); wi.append(np.full(len(idx), k))
CX, CY, CA, CW = map(np.concatenate, (xs, ys, av, wi))
tree = cKDTree(np.c_[CX, CY])
print(f"面点群 {len(CX)} 点 / {len(wids)} 坑井 (SUB={SUB}, K_FIT={K_FIT})", flush=True)


def plane_predict(qx, qy, self_k):
    """qx,qy の各点で、自坑井(self_k)を除く近傍ANCCを加重平面フィット→その点のANCC。非逐次・LOWO。
    ★座標を h でスケール化(悪条件回避)＋局所ANCC範囲でクランプ(外挿暴走防止)。"""
    dd, ii = tree.query(np.c_[qx, qy], k=K_QUERY, workers=-1)
    out = np.full(len(qx), np.nan)
    for r in range(len(qx)):
        jj = ii[r]; keep = CW[jj] != self_k
        jk = jj[keep][:K_FIT]; dk = dd[r][keep][:K_FIT]
        if len(jk) < 4:
            continue
        h = max(np.median(dk), 1.0)
        wt = np.exp(-0.5 * (dk / h) ** 2)
        dX = (CX[jk] - qx[r]) / h; dY = (CY[jk] - qy[r]) / h     # ★hでスケール化
        av = CA[jk]
        D = np.stack([np.ones_like(dX), dX, dY], 1)             # K×3 (切片=その点の値)
        A = (D * wt[:, None]).T @ D + 1e-6 * np.eye(3)
        b = (D * wt[:, None]).T @ av
        try:
            pred = float(np.linalg.solve(A, b)[0])
        except Exception:
            pred = float(np.average(av, weights=wt))
        lo, hi = float(av.min()), float(av.max()); mgn = 0.5 * (hi - lo) + 1.0   # ★局所範囲でクランプ
        out[r] = min(max(pred, lo - mgn), hi + mgn)
    return out


def rmse(e): e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan


# pf_ancc_2(参照, blendしない) & 最近傍距離
import pyarrow.parquet as pq
cent = np.array([[np.nanmean(W[w]["x"]), np.nanmean(W[w]["y"])] for w in wids]); ctree = cKDTree(cent)
tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "last_known_tvt", "pf_ancc_2"]).to_pandas()
tb["true"] = tb["last_known_tvt"] + tb["target"]; tb["se"] = (tb["pf_ancc_2"] - tb["true"]) ** 2
pf_se = tb.groupby("well")["se"].sum(); pf_n = tb.groupby("well")["se"].count()

rows = []
for k, w in enumerate(wids):
    d = W[w]; ev = ~np.isfinite(d["tvt_input"]); kn = np.isfinite(d["tvt_input"])
    if ev.sum() < 5 or kn.sum() < 5: continue
    dd, ii = ctree.query(cent[k], k=2); ncent = float(np.hypot(*(cent[k] - cent[ii[1]])))
    kx, ky, kz, ktrue = d["x"][kn], d["y"][kn], d["z"][kn], d["tvt"][kn]
    ex, ey, ez, etrue = d["x"][ev], d["y"][ev], d["z"][ev], d["tvt"][ev]
    fk = np.isfinite(kx) & np.isfinite(ky) & np.isfinite(kz) & np.isfinite(ktrue)
    fe = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(ez) & np.isfinite(etrue)
    if fk.sum() < 5 or fe.sum() < 5: continue
    Akn = plane_predict(kx[fk], ky[fk], k); Aev = plane_predict(ex[fe], ey[fe], k)
    okk = np.isfinite(Akn); oke = np.isfinite(Aev)
    if okk.sum() < 5 or oke.sum() < 5: continue
    b = float(np.mean(ktrue[fk][okk] - (Akn[okk] - kz[fk][okk])))       # 既知帯で b 較正
    pred = Aev[oke] - ez[fe][oke] + b                                  # ★TVT = ANCC - Z + b(非逐次)
    err = pred - etrue[fe][oke]
    rows.append(dict(well=w, ncent=ncent, n=int(oke.sum()), cover=float(oke.mean()),
                     surf_se=float(np.sum(err ** 2)),
                     hold_se=float(np.sum((ktrue[fk][-1] - etrue[fe][oke]) ** 2)),
                     pf_se=float(pf_se.get(w, np.nan)), pf_n=float(pf_n.get(w, np.nan))))
    if (len(rows)) % 150 == 0: print(f"  {len(rows)} 坑井...", flush=True)

import pandas as pd
R = pd.DataFrame(rows).set_index("well")
def pool(se, n): m = np.isfinite(se) & np.isfinite(n) & (n > 0); return float(np.sqrt(se[m].sum() / n[m].sum())) if m.any() else np.nan

print(f"\n評価 {len(R)} 坑井  (被覆中央 {R['cover'].median()*100:.0f}%)")
print("\n=== ★単体 overall pooled RMSE(非逐次 ANCC面, LOWO) ===")
print(f"  非逐次面(surf) = {pool(R['surf_se'].to_numpy(), R['n'].to_numpy()):.4f}")
print(f"  (参考) hold    = {pool(R['hold_se'].to_numpy(), R['n'].to_numpy()):.4f}")
print(f"  (参考) pf_ancc_2 = {pool(R['pf_se'].to_numpy(), R['pf_n'].to_numpy()):.4f}   ※blendでなく単体比較用")
print(f"  坑井別 中央RMSE surf = {np.median(np.sqrt(R['surf_se']/R['n'])):.4f}")
print("\n=== 最近傍距離別 overall RMSE(非逐次面) ===")
for lo, hi in [(0, 300), (300, 500), (500, 800), (800, 1500), (1500, 9e9)]:
    s = R[(R['ncent'] >= lo) & (R['ncent'] < hi)]
    if len(s): print(f"  {lo:>5}-{hi if hi<1e9 else '∞':>5}ft : {len(s):>4}坑井  surf={pool(s['surf_se'].to_numpy(),s['n'].to_numpy()):7.3f}  (pf={pool(s['pf_se'].to_numpy(),s['pf_n'].to_numpy()):.3f})")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
rk = lambda s, c: np.sqrt(R[s] / R[c])
ax[0].scatter(rk("pf_se", "pf_n"), rk("surf_se", "n"), s=12, c=np.log10(R["ncent"].clip(30, 3000)), cmap="viridis")
ax[0].plot([0, 30], [0, 30], "r--", alpha=.6); ax[0].set_xlim(0, 25); ax[0].set_ylim(0, 25)
ax[0].set_xlabel("PF per-well RMSE"); ax[0].set_ylabel("非逐次面 per-well RMSE"); ax[0].grid(alpha=.3)
ax[0].set_title("坑井別(単体): 対角下=面が勝ち (色=近傍距離log)")
thr = [(0,300),(300,500),(500,800),(800,1500)]
xs2 = [f"{a}-{b}" for a,b in thr]
ax[1].plot(range(len(thr)), [pool(R[(R['ncent']>=a)&(R['ncent']<b)]['surf_se'].to_numpy(), R[(R['ncent']>=a)&(R['ncent']<b)]['n'].to_numpy()) for a,b in thr], "o-", label="非逐次面 surf")
ax[1].plot(range(len(thr)), [pool(R[(R['ncent']>=a)&(R['ncent']<b)]['pf_se'].to_numpy(), R[(R['ncent']>=a)&(R['ncent']<b)]['pf_n'].to_numpy()) for a,b in thr], "s--", label="pf_ancc_2(参考)")
ax[1].set_xticks(range(len(thr))); ax[1].set_xticklabels(xs2); ax[1].set_xlabel("最近傍距離(ft)"); ax[1].set_ylabel("overall RMSE"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("非逐次面の単体性能(近傍が近いほど良い)")
fig.suptitle("非逐次 ANCC面 TVT予測(単体, LOWO) — 逐次推定なし"); fig.tight_layout()
fig.savefig(ART / "nonseq_ancc_surface.png", dpi=115)
print(f"\n[saved] {ART/'nonseq_ancc_surface.png'}")
