# -*- coding: utf-8 -*-
"""v93 module3: 空間imputer(formation-plane / dense-ANCC)と近傍(train)プール選択。

v52 create の空間KNN部分をクリーン再構築。全て train CSV から自己完結で構築(cross-version無し)。
  - FormationPlaneKNN : 各train井のXY重心+層median → 局所平面fit(XY→層深)。self除外(leave-self-out)。
  - DenseANCCImputer  : train井のXY-ANCCを密サンプル → KNN加重。self除外。
  - 近傍プール(neighbors_of): XY重心距離<=MAX_DIST の最近傍<=K_MAX 井(self/nbr疑似typewell用)。
※ 層列(ANCC..BUDA)は train のみ存在するが、imputer は train から prior を作り query井の XY だけで引くため
  test でも安全(test井の層列は不要)。[[kaggle-submit-pitfalls]] の test既知列制約を満たす。
"""
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJ / "rogii-wellbore-geology-prediction"

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10
DENSE_SPW = 60
DENSE_K = 20
K_MAX = 3
MAX_DIST = 1500.0
NEED_REFS = 2


class FormationPlaneKNN:
    """XY→層深 の局所平面fit(近傍K井の逆距離加重)。self除外可。"""

    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y"] + FORMATIONS).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            row = {"wid": wid, "x": float(df["X"].median()), "y": float(df["Y"].median())}
            for c in FORMATIONS:
                row[f"{c}_m"] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df["wid"])}
        xy = self.df[["x", "y"]].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1.0, xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df["x"].to_numpy(); self.ya = self.df["y"].to_numpy()
        self.fa = self.df[[f"{c}_m" for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap:
            dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1.0 / (dk + 1e-3), 0.0).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try:
            coef = np.linalg.solve(A, rhs)
        except Exception:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try:
                    coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except Exception:
                    pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class DenseANCCImputer:
    """train井のXY-ANCCを密サンプルしKNN加重推定。self除外可。"""

    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f"{wid}__horizontal_well.csv"
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) == 0:
                continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s["X"].values); ys.append(s["Y"].values)
            anccs.append(s["ANCC"].values); wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32); self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1.0, self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid:
            dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordr = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordr, 1); ik = np.take_along_axis(idx, ordr, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1.0 / (dk + 1e-3), 0.0)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1.0, sw); an = self.ancc[ik]
        ap = (an * w).sum(1) / safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return (ap.astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
                np.where(vk, dk, np.inf).min(1).astype(np.float32))


# ==================== 近傍(train)プール: XY重心 ====================
_TRAIN_CENT = None
_REFC = {}


def _train_centroids():
    global _TRAIN_CENT
    if _TRAIN_CENT is None:
        tdir = DATA_DIR / "train"
        wids = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(tdir / "*__horizontal_well.csv"))})
        xy = []
        for w in wids:
            h = pd.read_csv(tdir / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
            xy.append((float(h["X"].mean()), float(h["Y"].mean())))
        _TRAIN_CENT = (wids, np.array(xy))
    return _TRAIN_CENT


def ref_grtvt(wid):
    """近傍井の (GR, TVT)。疑似typewell の材料。"""
    if wid not in _REFC:
        h = pd.read_csv(DATA_DIR / "train" / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT"])
        _REFC[wid] = (h["GR"].to_numpy(float), h["TVT"].to_numpy(float))
    return _REFC[wid]


def nearest_dist(cx, cy, exclude_wid=None):
    """最近傍train井までの実距離(孤立度)。"""
    wids, xy = _train_centroids(); d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    if exclude_wid is not None:
        keep = np.array([w != exclude_wid for w in wids]); d = d[keep]
    return float(d.min()) if len(d) else float("nan")


def neighbors_of(cx, cy, exclude_wid=None):
    """XY重心距離<=MAX_DIST の最近傍<=K_MAX 井 [(wid, dist), ...]。"""
    wids, xy = _train_centroids(); d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy); out = []
    for j in np.argsort(d):
        w = wids[j]
        if w == exclude_wid:
            continue
        if d[j] > MAX_DIST:
            break
        out.append((w, float(d[j])))
        if len(out) >= K_MAX:
            break
    return out


def build_imputers():
    """train全井から FI/DI を構築(self除外は impute 時に self_wid で行う)。"""
    TR = DATA_DIR / "train"
    hw_paths = sorted(TR.glob("*__horizontal_well.csv"))
    train_wids = [p.stem.replace("__horizontal_well", "") for p in hw_paths]
    FI = FormationPlaneKNN(train_wids, TR)
    DI = DenseANCCImputer(train_wids, TR)
    return FI, DI, train_wids


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
    FI, DI, wids = build_imputers()
    print(f"[imputers] FI={len(FI.df)}井 / DI点={len(DI.ancc)} / train={len(wids)}井")
    w0 = wids[0]
    h = pd.read_csv(DATA_DIR / "train" / f"{w0}__horizontal_well.csv", usecols=["X", "Y"])
    xy = np.array([[float(h["X"].mean()), float(h["Y"].mean())]])
    pf_pred, pf_d = FI.impute(xy, self_wid=w0)
    ap, asd, ad = DI.impute(xy, self_wid=w0)
    nb = neighbors_of(xy[0, 0], xy[0, 1], exclude_wid=w0)
    print(f"  {w0}: formation pred={np.round(pf_pred[0],1)} knn_d={pf_d[0]:.1f}")
    print(f"       dense ANCC={ap[0]:.1f}±{asd[0]:.1f} d={ad[0]:.1f} / 近傍{len(nb)}={[(w,round(d,0)) for w,d in nb]}")
