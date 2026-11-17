# -*- coding: utf-8 -*-
"""夜間実験B(CPU): per-segment 枝方向を"幾何+PF分布"特徴の学習分類器で当てられるか(research OQ2/独立軸)。
emission=分類器prob → Viterbi/argmax で 3.22 oracle をどこまで実現するか。真TVTはlabel/評価のみ。
well-CV(random-test前提で正当)。出力: v95/artifacts_v95b/exp_segment_branch_clf.txt"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
B = "v95/artifacts_v95b/"
REPS = ["ancc", "self", "nbr", "nbr_gr5", "self_graft"]
sch = set(pq.read_schema(B + "data/train.parquet").names)
PC = [f"pf_{r}_delta_{k}" for r in REPS for k in range(1, 7) if f"pf_{r}_delta_{k}" in sch]
PROV = [c for c in sch if c.startswith("prov_")]
tb = pq.read_table(B + "data/train.parquet", columns=PC + PROV + ["well", "target", "last_known_tvt", "md_since", "z"]).to_pandas()
y = tb["target"].to_numpy(float); g = tb["well"].astype(str).to_numpy()
lk = tb["last_known_tvt"].to_numpy(float); mds = tb["md_since"].to_numpy(float)
z = tb["z"].to_numpy(float); mdv = mds; P = tb[PC].to_numpy(float)
d1 = np.load(B.replace("v95b", "v95") + "l1_oof.npz")
gbm = np.mean([d1[k] for k in d1.files if k not in ("y", "well")], 0)
tcn = np.mean([v for k, v in np.load(B + "nn_oof_combo.npz").items() if k.startswith("mdn2")], 0)
l5 = np.load(B + "l5_ext_oof.npz")["l5_mean"].astype(float)
Mm = np.stack([gbm, tcn, l5], 1); mf = np.isfinite(Mm).all(1) & np.isfinite(y)
wj, _ = nnls(Mm[mf], y[mf]); wj = wj / wj.sum(); blend = Mm @ wj
def R(p): m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
ch = np.where(g[1:] != g[:-1])[0] + 1; starts = np.concatenate([[0], ch]); ends = np.concatenate([ch, [len(g)]])
STR = 16; BR = np.array([-2, -1, 0, 1, 2]); NB = len(BR)

# ---- per-segment 特徴/ラベル 構築 ----
rows_feat = []; seg_meta = []   # meta: (well, gidx_of_segment_rows, seg_blend_median)
for s, e in zip(starts, ends):
    o = np.argsort(mdv[s:e], kind="stable"); idx = np.arange(s, e)[o]
    T = e - s; nb = (T + STR - 1) // STR; seg_of = np.minimum(np.arange(T) // STR, nb - 1)
    bl = blend[idx]; zz = z[idx]; md_ = mdv[idx]; mdsi = mds[idx]; Pr = P[idx]
    seg_bl = np.array([np.nanmedian(bl[seg_of == b]) for b in range(nb)])
    seg_z = np.array([np.nanmean(zz[seg_of == b]) for b in range(nb)])
    seg_md = np.array([np.nanmean(md_[seg_of == b]) for b in range(nb)])
    seg_mds = np.array([np.nanmean(mdsi[seg_of == b]) for b in range(nb)])
    # 近傍segmentとのdip/連続(前後差)
    dbl = np.gradient(seg_bl) if nb > 1 else np.zeros(nb)
    dz = np.gradient(seg_z) if nb > 1 else np.zeros(nb)
    dmd = np.gradient(seg_md) if nb > 1 else np.ones(nb)
    incl = dz / np.where(np.abs(dmd) > 1, dmd, 1)                       # wellbore傾斜
    dip_impl = (dbl + dz) / np.where(np.abs(dmd) > 1, dmd, 1)           # blend実効dip
    rough = np.abs(np.gradient(dbl)) if nb > 2 else np.zeros(nb)
    for b in range(nb):
        rmask = np.where(seg_of == b)[0]; gr = idx[rmask]
        dd = Pr[rmask] - seg_bl[b]                                      # (n,30)
        votes = [np.mean(np.abs(dd - br * 15.0) < 7.5) for br in BR]    # 枝ごとPF支持
        spread = float(np.nanstd(Pr[rmask]))
        # ラベル: 真の最良枝
        errs = [np.sqrt(np.nanmean(((seg_bl[b] + br * 15.0) - y[gr]) ** 2)) for br in BR]
        ybr = int(BR[int(np.argmin(errs))])
        rows_feat.append(dict(well=g[s], seg=b, nb=nb, seg_frac=b / max(nb - 1, 1),
                              seg_bl=seg_bl[b], seg_z=seg_z[b], seg_mds=seg_mds[b],
                              incl=incl[b], dip_impl=dip_impl[b], rough=rough[b],
                              dbl=dbl[b], spread=spread,
                              v_m2=votes[0], v_m1=votes[1], v_0=votes[2], v_p1=votes[3], v_p2=votes[4],
                              ybr=ybr))
        seg_meta.append((s, gr, seg_bl[b]))
F = pd.DataFrame(rows_feat)
FEATS = ["seg", "nb", "seg_frac", "seg_z", "seg_mds", "incl", "dip_impl", "rough", "dbl", "spread",
         "v_m2", "v_m1", "v_0", "v_p1", "v_p2"]
X = F[FEATS].to_numpy(np.float32); grp = F["well"].to_numpy()
ybr = F["ybr"].to_numpy(int); ycls = ybr + 2                            # 0..4
print(f"segment数={len(F)} 枝分布={np.bincount(ycls, minlength=5)} (0中心が多数)", flush=True)

# ---- 多クラス分類 well-CV → 各segmentの枝prob ----
prob = np.zeros((len(F), NB))
for tr, va in GroupKFold(5).split(X, ycls, grp):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=80,
                           subsample=0.8, colsample_bytree=0.7, objective="multiclass", num_class=NB,
                           device="cpu", n_jobs=-1, verbose=-1)
    m.fit(X[tr], ycls[tr]); prob[va] = m.predict_proba(X[va])
# 方向AUC(shifted segmentで +枝 vs −枝 を当てられるか)
shifted = ybr != 0
if shifted.sum() > 20:
    dir_true = (ybr[shifted] > 0).astype(int)
    dir_score = (prob[shifted][:, 3] + prob[shifted][:, 4]) - (prob[shifted][:, 0] + prob[shifted][:, 1])
    try: dir_auc = roc_auc_score(dir_true, dir_score)
    except Exception: dir_auc = np.nan
else: dir_auc = np.nan
# shifted検出AUC(0 vs 非0)
det_auc = roc_auc_score((ybr != 0).astype(int), 1 - prob[:, 2])

# ---- 適用: argmax枝 / Viterbi(emission=log prob + dip連続) ----
def apply_pred(mode):
    out = blend.copy()
    for (s, gr, seg_bl_b), pr in zip(seg_meta, prob):
        pass
    # segmentごとに well集約してViterbi
    for s, e in zip(starts, ends):
        pass
    return out
# 簡易: argmax（well内Viterbiは meta再構築が要るので別途）
pred_arg = blend.copy()
for (s, gr, seg_bl_b), pr in zip(seg_meta, prob):
    b = int(np.argmax(pr)) - 2; pred_arg[gr] = seg_bl_b + b * 15.0
# Viterbi: well単位でsegment列を復元
vit = blend.copy()
LAM = 2.0; wells_order = list(dict.fromkeys(F["well"].tolist()))
mi = 0
segs_by_well = {}
for i, (wq) in enumerate(F["well"].tolist()):
    segs_by_well.setdefault(wq, []).append(i)
for wq, ilist in segs_by_well.items():
    nb = len(ilist); Elog = np.log(prob[ilist] + 1e-4)
    dp = np.full((nb, NB), -1e9); bp = np.zeros((nb, NB), int); dp[0] = Elog[0]
    for b in range(1, nb):
        for j in range(NB):
            val = dp[b - 1] - LAM * np.abs(BR - BR[j]); k = int(np.argmax(val))
            dp[b, j] = val[k] + Elog[b, j]; bp[b, j] = k
    path = np.zeros(nb, int); path[-1] = int(np.argmax(dp[-1]))
    for b in range(nb - 2, -1, -1): path[b] = bp[b + 1, path[b + 1]]
    for pos, i in enumerate(ilist):
        s, gr, seg_bl_b = seg_meta[i]; vit[gr] = seg_bl_b + BR[path[pos]] * 15.0

out = ["=== 実験B: per-segment 枝方向 学習分類器 (OQ2/独立軸) ==="]
out.append(f"  blend(base)          = {R(blend):.4f}")
out.append(f"  per-segment枝oracle  = 3.2164")
out.append(f"  ★shifted検出AUC(0vs非0) = {det_auc:.4f}")
out.append(f"  ★★枝方向AUC(+vs−, shifted内) = {dir_auc:.4f}  (>0.58なら独立信号=壁突破の兆し)")
out.append(f"  適用 argmax枝        = {R(pred_arg):.4f}  ({R(pred_arg)-R(blend):+.4f})")
out.append(f"  適用 Viterbi(dip連続) = {R(vit):.4f}  ({R(vit)-R(blend):+.4f})")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(B + "exp_segment_branch_clf.txt", "w", encoding="utf-8").write(txt)
