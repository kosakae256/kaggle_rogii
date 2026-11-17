# -*- coding: utf-8 -*-
"""夜間実験C(CPU): GR別表現(envelope/微分/roughness)+オフセット相関(tdpf/tdsc/tda)で
per-segment 枝【方向】(+vs−)を当てられるか。amplitude+幾何は0.50(実験B)。GR別表現が独立に方向を割るか。
ユーザー案「GRは波形みたいなもの」の検証。真TVTはlabel/評価のみ。well-CV。
出力: v95/artifacts_v95b/exp_gr_repr_direction.txt"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
B = "v95/artifacts_v95b/"
sch = list(pq.read_schema(B + "data/train.parquet").names)
# GR別表現 + オフセット相関(枝位置ごとのGR一致プロファイル=方向信号の在処)
GRREP = [c for c in ["gr", "gr_d1", "gr_d2", "gr_env", "gr_nrg", "gr_abs_d1", "gr_abs_d2",
                     "gr_rough21", "gr_rough51", "grs5", "grs21", "grs51", "grs101",
                     "gr_d1_rmean21", "gr_abs_d1_rmean21"] if c in sch]
OFF = [c for c in sch if (c.startswith("tdpf") or c.startswith("tdsc") or c.startswith("tda")
                          or c.startswith("tdbc")) and c[-2:] not in ("_4", "_5", "_6")]
GRBEST = [c for c in sch if c.endswith("gr_best_off") or c.endswith("gr_best_resid")
          or c.endswith("gr_abs_min") or c.endswith("gr_resid0")]
GRBEST = GRBEST[:20]
need = list(dict.fromkeys(GRREP + OFF + GRBEST + ["well", "target", "last_known_tvt", "md_since"]
                          + [f"pf_{r}_delta_{k}" for r in ["ancc", "self", "nbr", "nbr_gr5", "self_graft"] for k in range(1, 7)]))
need = [c for c in need if c in sch]
tb = pq.read_table(B + "data/train.parquet", columns=need).to_pandas()
y = tb["target"].to_numpy(float); g = tb["well"].astype(str).to_numpy(); mds = tb["md_since"].to_numpy(float)
PC = [c for c in need if "_delta_" in c]; P = tb[PC].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz")
gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(y)
wj, _ = nnls(Mm[mf], y[mf]); wj = wj / wj.sum(); blend = Mm @ wj
FEATS = GRREP + OFF + GRBEST
print(f"GR別表現{len(GRREP)} + offset相関{len(OFF)} + grbest{len(GRBEST)} = {len(FEATS)}特徴", flush=True)
ch = np.where(g[1:] != g[:-1])[0] + 1; starts = np.concatenate([[0], ch]); ends = np.concatenate([ch, [len(g)]])
STR = 16
# per-segment: 特徴=区間平均, ラベル=真枝方向(shiftedのみ)
Xrows = []; ydir = []; grp = []
Xm = tb[FEATS].to_numpy(np.float32)
for s, e in zip(starts, ends):
    o = np.argsort(mds[s:e], kind="stable"); idx = np.arange(s, e)[o]
    T = e - s; nb = (T + STR - 1) // STR; seg_of = np.minimum(np.arange(T) // STR, nb - 1)
    bl = blend[idx]; yy = y[idx]; Xi = Xm[idx]
    for b in range(nb):
        rows = np.where(seg_of == b)[0]
        seg_bl = np.nanmedian(bl[rows])
        br = int(np.clip(np.round((np.nanmedian(yy[rows]) - seg_bl) / 15.0), -2, 2))
        if br == 0:
            continue                                   # shifted のみ(方向を問う)
        Xrows.append(np.nanmean(Xi[rows], 0)); ydir.append(1 if br > 0 else 0); grp.append(g[s])
Xseg = np.array(Xrows); ydir = np.array(ydir); grp = np.array(grp)
print(f"shifted segment={len(ydir)}  +方向={ydir.mean()*100:.0f}%", flush=True)
oof = np.zeros(len(ydir))
for tr, va in GroupKFold(5).split(Xseg, ydir, grp):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                           subsample=0.8, colsample_bytree=0.5, reg_lambda=2.0,
                           device="cpu", n_jobs=-1, verbose=-1)
    m.fit(Xseg[tr], ydir[tr]); oof[va] = m.predict_proba(Xseg[va])[:, 1]
auc = roc_auc_score(ydir, oof)
# サブセット別AUC: GR別表現のみ / offset相関のみ
def sub_auc(cols):
    ix = [FEATS.index(c) for c in cols if c in FEATS]
    Xs = Xseg[:, ix]; oo = np.zeros(len(ydir))
    for tr, va in GroupKFold(5).split(Xs, ydir, grp):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               colsample_bytree=0.6, reg_lambda=2.0, device="cpu", n_jobs=-1, verbose=-1)
        m.fit(Xs[tr], ydir[tr]); oo[va] = m.predict_proba(Xs[va])[:, 1]
    return roc_auc_score(ydir, oo)
auc_rep = sub_auc(GRREP); auc_off = sub_auc(OFF)
out = ["=== 実験C: GR別表現で枝【方向】を当てられるか (ユーザー案) ==="]
out.append(f"  shifted segment={len(ydir)} / 特徴={len(FEATS)}")
out.append(f"  ★方向AUC(全GR表現+offset) = {auc:.4f}   (実験B幾何+PF=0.5003。>0.55なら独立方向信号)")
out.append(f"    GR別表現のみ(env/微分/rough) = {auc_rep:.4f}")
out.append(f"    offset相関のみ(tdpf/tdsc/tda) = {auc_off:.4f}")
out.append(f"  → >0.55なら壁突破の兆し=候補生成×この方向emissionで4.x射程。0.50近傍なら情報限界確定。")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(B + "exp_gr_repr_direction.txt", "w", encoding="utf-8").write(txt)
