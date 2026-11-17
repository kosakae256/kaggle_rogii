# -*- coding: utf-8 -*-
"""v93 spatial leg: STRUCT-FREE 空間相補脚(井戸レベル datum オフセットの空間回帰)。

狙い(別VIEW): コンペ誤差には空間的に滑らかな datum 成分がある(XYで近い井は datum を共有しがち)。
各井の datum オフセット(=その井の eval 行の target 中央値)を、"自分自身のラベルを使わず"
空間的な近傍井から予測し、行へ broadcast して per-row OOF を作る。これは行レベル(GBM)でも
井戸系列(TCN)でもない「井戸レベル空間VIEW」なので、GBM/TCN/L5/image と脱相関するはず。
memory: v42 spatial-datum-mrf は median を改善し tail を悪化 -> 単体でなく脱相関脚として使う。

★struct禁止: struct/v38/v41/gf/v50b 由来は一切使わない。使うのは parquet の (well,id,target) と
  train CSV の XY 重心(imputers_v93._train_centroids)のみ。cp932安全(em-dash不使用)。

モデル(env SMODEL, 既定 "knn"):
  1. knn : 学習なし。各井の datum を「最近傍 K 井(自fold除外=leave-fold-out & leave-self-out)の
           median(target) の Gaussian 加重平均」で空間補間。速い・即答。真の相補判定はここで出る。
  2. gnn : kNN 井グラフ上の小さな message passing。ノード特徴=XY/既知帯集約(eval targetは不使用)。
           GroupKFold(5) で OOF。SMOKE のみ(重ければフル学習しない)。env SMODEL=gnn/both で起動。

fold整合: L1/L2/L5/image と同一の GroupKFold(5).split(zeros(N), y, groups=well)(parquet行順)。
  knn は各fold、近傍プールを「train-fold の他井」に限定 -> leave-fold-out かつ leave-self-out で厳密OOF。
出力: artifacts_v93/spatial_oof.npz  (per-row, target単位, keys sp_knn/[sp_gnn]/sp_mean/y)

実行: ./.venv/Scripts/python.exe v93/train_spatial_v93.py            (knn フル, 即完了)
      SMODEL=both GNN_SMOKE=1 ./.venv/Scripts/python.exe v93/train_spatial_v93.py  (gnn smoke込み)
"""
import io, sys, os
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imputers_v93 import _train_centroids  # per-well XY 重心(train CSV, struct-free)

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"
PQ = ART / "data" / "train.parquet"
N_SPLITS = 5
SMODEL = os.environ.get("SMODEL", "knn").lower()   # knn / gnn / both
KNN_K = int(os.environ.get("KNN_K", "15"))
GNN_SMOKE = os.environ.get("GNN_SMOKE", "0") == "1"


def rmse(pred, y):
    e = pred - y
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


# ==================== データ読み込み(parquet: well,id,target + XY重心) ====================
tb = pq.read_table(PQ, columns=["well", "target"])
well = tb.column("well").to_numpy(zero_copy_only=False)
y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
del tb
N = len(y)
# 行 -> 井 code(parquet行順を保持)
codes, uniq = pd.factorize(well, sort=False)   # codes[i]=井index, uniq=井id配列(出現順)
n_w = len(uniq)
print(f"[data] rows={N}  wells={n_w}  target finite={np.isfinite(y).mean()*100:.2f}%", flush=True)

# 井ごと median(target)(=datum オフセットのロバスト要約)と 行スライス
order = np.argsort(codes, kind="stable")       # 井ごとに行をまとめる
codes_sorted = codes[order]
bounds = np.searchsorted(codes_sorted, np.arange(n_w + 1))   # 井 j の行 = order[bounds[j]:bounds[j+1]]
well_median = np.empty(n_w, np.float64)
for j in range(n_w):
    rows_j = order[bounds[j]:bounds[j + 1]]
    yy = y[rows_j]
    yy = yy[np.isfinite(yy)]
    well_median[j] = np.median(yy) if len(yy) else 0.0

# XY 重心(imputers と同じ mean 重心)を parquet の井順に整列
cent_wids, cent_xy = _train_centroids()
cmap = {w: i for i, w in enumerate(cent_wids)}
XY = np.empty((n_w, 2), np.float64)
miss = 0
for j, w in enumerate(uniq):
    i = cmap.get(w)
    if i is None:
        XY[j] = np.nan; miss += 1
    else:
        XY[j] = cent_xy[i]
print(f"[data] 井 median(target): mean={well_median.mean():.3f} std={well_median.std():.3f} "
      f"(min={well_median.min():.2f} max={well_median.max():.2f}) / XY欠損井={miss}", flush=True)

# ==================== fold map(L1と同一 GroupKFold5, groups=well) ====================
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.zeros(N), y, groups=well))
# 井 -> どの fold の val か(GroupKFoldは各groupを一意に1 valへ)
well_fold = np.full(n_w, -1, np.int32)
for k, (_tr, va) in enumerate(folds):
    for j in np.unique(codes[va]):
        well_fold[j] = k
assert (well_fold >= 0).all(), "全井が fold 割当を持つこと"
print(f"[fold] GroupKFold(5) 井分布 = {np.bincount(well_fold).tolist()} (合計{n_w})", flush=True)


# ==================== 1. knn 空間補間(学習なし, leave-fold-out & leave-self-out) ====================
def knn_spatial(K):
    """各井 datum = 自fold除外の最近傍K井 median(target) の Gaussian(適応帯域)加重平均。"""
    pred_w = np.full(n_w, np.nan, np.float64)
    valid = np.isfinite(XY).all(1)
    global_mean = well_median[valid].mean()
    for k in range(N_SPLITS):
        val_mask = (well_fold == k) & valid
        pool_mask = (well_fold != k) & valid            # train-fold の他井のみ
        pj = np.where(pool_mask)[0]
        vj = np.where(val_mask)[0]
        if len(pj) == 0 or len(vj) == 0:
            continue
        tree = cKDTree(XY[pj])
        kq = min(K, len(pj))
        dist, idx = tree.query(XY[vj], k=kq)          # (len(vj), kq)
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        h = np.maximum(dist[:, -1:], 1e-6)             # 適応帯域=K番目近傍距離
        w = np.exp(-0.5 * (dist / h) ** 2)             # Gaussian kernel
        sw = w.sum(1)
        vals = well_median[pj[idx]]                    # 近傍井の median(target)
        pw = (w * vals).sum(1) / np.where(sw < 1e-12, 1.0, sw)
        pw = np.where(sw < 1e-12, global_mean, pw)
        pred_w[vj] = pw
    pred_w[~np.isfinite(pred_w)] = global_mean         # XY欠損などの保険
    # 行へ broadcast
    row = pred_w[codes]
    return pred_w, row.astype(np.float32)


print("\n[knn] K 感度(井レベル OOF: 予測井datum vs 実median, 参考)", flush=True)
best = None
for K in sorted(set([8, 12, KNN_K, 20, 30])):
    pw, _ = knn_spatial(K)
    r = root_mean_squared_error(well_median, pw)       # 井レベル(datum予測精度)
    print(f"   K={K:3d}: well-level datum RMSE={r:.4f}", flush=True)
    if best is None or r < best[1]:
        best = (K, r)
pred_w_knn, sp_knn = knn_spatial(KNN_K)
row_rmse_knn = rmse(sp_knn.astype(np.float64), y)
print(f"[knn] 採用 K={KNN_K}  井レベル datum RMSE={root_mean_squared_error(well_median, pred_w_knn):.4f}"
      f"  (best K={best[0]} @ {best[1]:.4f})", flush=True)
print(f"[knn] per-row pooled RMSE(sp_knn vs target) = {row_rmse_knn:.4f}   "
      f"(参考: 素の naive last_known=median~15, これは datum のみを説明する部分脚)", flush=True)


# ==================== 2. gnn 空間 message passing(SMOKE) ====================
sp_gnn = None
if SMODEL in ("gnn", "both"):
    try:
        import torch, torch.nn as nn
        torch.manual_seed(0)
        valid = np.isfinite(XY).all(1)
        # ノード特徴(eval target不使用): 正規化XY + 既知帯 last_known_tvt 集約
        try:
            lk = pq.read_table(PQ, columns=["last_known_tvt"]).column(0).to_numpy(zero_copy_only=False).astype(np.float64)
            lk_med = np.array([np.nanmedian(lk[order[bounds[j]:bounds[j + 1]]]) if bounds[j + 1] > bounds[j] else 0.0
                               for j in range(n_w)])
        except Exception:
            lk_med = np.zeros(n_w)
        XYf = np.where(np.isfinite(XY), XY, np.nanmean(XY[valid], 0))
        xy_n = (XYf - XYf[valid].mean(0)) / (XYf[valid].std(0) + 1e-9)
        lk_n = (lk_med - np.nanmean(lk_med)) / (np.nanstd(lk_med) + 1e-9)
        feat = np.column_stack([xy_n, np.nan_to_num(lk_n)]).astype(np.float32)   # (n_w, 3)
        # kNN グラフ(全井, 有向近傍index)
        tree = cKDTree(XYf[valid])
        vidx = np.where(valid)[0]
        Kg = min(KNN_K, len(vidx) - 1)
        _, nn_local = tree.query(XYf, k=Kg + 1)
        nbr = vidx[nn_local[:, 1:]]                     # (n_w, Kg) 各井の近傍(自身除外)

        F = torch.tensor(feat)
        NB = torch.tensor(nbr, dtype=torch.long)
        med = torch.tensor(well_median, dtype=torch.float32)
        foldt = torch.tensor(well_fold, dtype=torch.long)

        class SpatialGNN(nn.Module):
            """近傍の [特徴 + (train限定)ラベル] を message pass。val井ラベルはmaskで遮断=OOF安全。"""
            def __init__(self, fin, h=32):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(fin + 2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
                self.msg = nn.Sequential(nn.Linear(h + 1, h), nn.ReLU())
                self.out = nn.Sequential(nn.Linear(h + h, h), nn.ReLU(), nn.Linear(h, 1))

            def forward(self, F, NB, nbr_label, nbr_lmask):
                # ノード自己埋め込み(自ラベルは絶対に入れない)
                self_in = torch.cat([F, torch.zeros(F.size(0), 2)], 1)
                hself = self.enc(self_in)
                # 近傍メッセージ: 近傍特徴 + (train限定)ラベル
                nb_feat = hself[NB]                                   # (n,K,h) 近傍自己埋め込み
                nb_lab = nbr_label.unsqueeze(-1)                      # (n,K,1)
                nb_lab = nb_lab * nbr_lmask.unsqueeze(-1)             # val近傍のラベルは0で遮断
                m = self.msg(torch.cat([nb_feat, nb_lab], -1)).mean(1)  # 平均集約
                return self.out(torch.cat([hself, m], 1)).squeeze(-1)

        # SMOKE: fold0 のみ, 少epoch。val井=fold0, train井=残り。近傍ラベルmask=近傍がtrainなら1。
        pred_w_gnn = np.full(n_w, np.nan, np.float64)
        SMOKE_FOLDS = [0] if GNN_SMOKE else list(range(N_SPLITS))
        epochs = 60 if GNN_SMOKE else 400
        for k in SMOKE_FOLDS:
            tr = torch.tensor((well_fold != k) & valid)
            vao = torch.tensor((well_fold == k) & valid)
            nbr_fold = foldt[NB]                                     # (n,K)
            nbr_lmask = (nbr_fold != k).float()                     # 近傍がval(=fold k)ならラベル遮断
            nbr_label = med[NB]
            model = SpatialGNN(F.size(1))
            opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
            for ep in range(epochs):
                model.train(); opt.zero_grad()
                out = model(F, NB, nbr_label, nbr_lmask)
                loss = ((out[tr] - med[tr]) ** 2).mean()
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                out = model(F, NB, nbr_label, nbr_lmask).numpy()
            pred_w_gnn[vao.numpy()] = out[vao.numpy()]
        gm = well_median[valid].mean()
        pred_w_gnn[~np.isfinite(pred_w_gnn)] = gm
        sp_gnn = pred_w_gnn[codes].astype(np.float32)
        cov = np.isfinite(pred_w_gnn[well_fold == 0]).mean() if GNN_SMOKE else 1.0
        rr = root_mean_squared_error(well_median[well_fold == 0] if GNN_SMOKE else well_median,
                                     pred_w_gnn[well_fold == 0] if GNN_SMOKE else pred_w_gnn)
        print(f"\n[gnn] SMOKE={GNN_SMOKE} folds={SMOKE_FOLDS} epochs={epochs}  "
              f"井レベル datum RMSE={rr:.4f}  finite={np.isfinite(sp_gnn).mean()*100:.1f}%", flush=True)
        if GNN_SMOKE:
            print("[gnn] smoke: 機構は動作(全fold本学習は未実施)。真の相補判定は knn を参照。", flush=True)
    except Exception as e:
        print(f"[gnn] SKIP(例外): {type(e).__name__}: {e}", flush=True)
        sp_gnn = None


# ==================== 出力 spatial_oof.npz ====================
out = {"sp_knn": sp_knn, "y": y.astype(np.float32)}
members = [sp_knn]
if sp_gnn is not None:
    out["sp_gnn"] = sp_gnn
    members.append(sp_gnn)
out["sp_mean"] = np.mean(members, axis=0).astype(np.float32)
np.savez_compressed(ART / "spatial_oof.npz", **out)
print(f"\n[save] {ART/'spatial_oof.npz'}  keys={list(out.keys())}", flush=True)


# ==================== 相補性(3-way vs 4-way nnls) ====================
print("\n" + "=" * 66, flush=True)
print("★ 相補性チェック(struct-free spatial 脚は脱相関し、寄与するか)", flush=True)
print("=" * 66, flush=True)
gbm = np.load(ART / "gbm_l1_oof.npy").astype(np.float64)
zc = np.load(ART / "nn_oof_combo.npz")
tcn = np.mean([zc[k] for k in zc.files if k.startswith("mdn2_s")], axis=0).astype(np.float64)
zl = np.load(ART / "l5_oof.npz")
l5 = zl["l5_mean"].astype(np.float64)
sp = sp_knn.astype(np.float64)

fin = np.isfinite(gbm) & np.isfinite(tcn) & np.isfinite(l5) & np.isfinite(sp) & np.isfinite(y)
print(f"有限行={fin.sum()}/{N}", flush=True)


def corr(a, b):
    a = a[fin]; b = b[fin]
    return float(np.corrcoef(a, b)[0, 1])


print(f"\n[相関] spatial(sp_knn) vs 各脚 (低いほど脱相関=良い別VIEW):", flush=True)
print(f"   corr(spatial, gbm) = {corr(sp, gbm):.4f}", flush=True)
print(f"   corr(spatial, tcn) = {corr(sp, tcn):.4f}", flush=True)
print(f"   corr(spatial, l5 ) = {corr(sp, l5):.4f}", flush=True)
print(f"   (参考) corr(gbm,tcn)={corr(gbm,tcn):.4f} corr(gbm,l5)={corr(gbm,l5):.4f} "
      f"corr(tcn,l5)={corr(tcn,l5):.4f}", flush=True)


def fit_nnls(cols, names):
    A = np.column_stack([c[fin] for c in cols])
    b = y[fin]
    wts, _ = nnls(A, b)
    pred = A @ wts
    r = float(np.sqrt(np.mean((pred - b) ** 2)))
    return wts, r


img = ART / "image_oof.npz"
w3, r3 = fit_nnls([gbm, tcn, l5], ["gbm", "tcn", "l5"])
w4, r4 = fit_nnls([gbm, tcn, l5, sp], ["gbm", "tcn", "l5", "spatial"])
print(f"\n[nnls 3-way]  {{gbm,tcn,l5}}          RMSE={r3:.4f}  w={np.round(w3,4).tolist()}", flush=True)
print(f"[nnls 4-way]  {{+spatial}}            RMSE={r4:.4f}  w={np.round(w4,4).tolist()}", flush=True)
print(f"   -> 4-way 改善 = {r3 - r4:+.4f}  (spatial 重み={w4[-1]:.4f})", flush=True)

if img.exists():
    zi = np.load(img)
    ik = "image_mean" if "image_mean" in zi.files else [k for k in zi.files if k != "y"][0]
    im = zi[ik].astype(np.float64)
    fin5 = fin & np.isfinite(im)
    def fit5(cols):
        A = np.column_stack([c[fin5] for c in cols]); b = y[fin5]
        wts, _ = nnls(A, b); return wts, float(np.sqrt(np.mean((A @ wts - b) ** 2)))
    w4i, r4i = fit5([gbm, tcn, l5, im])
    w5, r5 = fit5([gbm, tcn, l5, im, sp])
    print(f"\n[nnls 4-way]  {{gbm,tcn,l5,image}}      RMSE={r4i:.4f}  w={np.round(w4i,4).tolist()}", flush=True)
    print(f"[nnls 5-way]  {{+spatial}}            RMSE={r5:.4f}  w={np.round(w5,4).tolist()}", flush=True)
    print(f"   -> 5-way 改善 = {r4i - r5:+.4f}  (spatial 重み={w5[-1]:.4f})", flush=True)
else:
    print("\n[image] image_oof.npz なし -> 5-way はスキップ", flush=True)

gain = r3 - r4
if gain > 0.01 and w4[-1] > 1e-3:
    verdict = f"ADD (spatial は寄与: -{gain:.3f} RMSE, w={w4[-1]:.3f}). 別VIEWの相補脚として採用価値あり。"
elif w4[-1] > 1e-3:
    verdict = f"MARGINAL (spatial 重み {w4[-1]:.3f} だが改善 {gain:+.3f} は僅少)。脱相関はするが利得は小さい。"
else:
    verdict = f"DROP (nnls 重み~0, 改善 {gain:+.3f})。datum 成分は既存脚が既に吸収済み。"
print(f"\n★ VERDICT: {verdict}", flush=True)
print("=" * 66, flush=True)
