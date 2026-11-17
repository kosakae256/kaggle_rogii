# -*- coding: utf-8 -*-
"""診断D(核心テスト): 「証拠を区間segmentに集約すると枝offsetを当てられるか」を軽い学習で検証。
   仮説: 行レベルの枝判別は~56%限界だが、1区間に~100倍の証拠が集まれば当てられる(診断B: segment oracle 0.18)。
   手順:
     ・各坑井をMD順に固定長S行のsegmentに分割
     ・segmentごとに証拠特徴(GRマッチ/候補spread/不一致/dip)を平均・標準偏差で集約
     ・target = そのsegmentの combo残差 中央値(=−segment oracleが除く量)
     ・GroupKFold(坑井) で LightGBM 回帰 → OOFで segment offset を予測(真値未使用の予測)
     ・combo_corrected = combo − predicted_offset → overall RMSE(行プール)を combo 6.42 と比較
   対照: 同特徴を「行レベル(集約なし)」でも回して、集約が効くのかを分離。
   出力: コンソール + artifacts_v69_4/diag_segment_regress.png"""
import io, sys, re
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")
S = 96          # segment 行数
NFOLD = 5

# combo OOF
g = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)
z = np.load(ART / "nn_oof_combo.npz")
combo = 0.54 * np.mean([z[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], axis=0) + 0.46 * g

# 証拠特徴を名前で選抜(枝判別に効きそうな: GRマッチ/候補spread/不一致/dip)
sch = [f.name for f in pq.read_schema(ART / "data" / "train.parquet")]
CAND = [f"pf_{k}_{i}" for k in ("ancc", "self", "nbr") for i in range(1, 6)]
STD = [f"pf_ancc_std_{i}" for i in range(1, 6)]
EVID = [c for c in sch if re.search(r"gr_(resid0|abs0|abs_min|best_off|best_resid)$", c)]  # GR証拠
DISAG = [c for c in sch if re.search(r"^(pf_(mean|med)_)?vs_|_vs_(spatial|dense|sc_ens|beam)", c)]
DIP = [c for c in sch if c in ("dzdmd", "slp_all", "slp_50", "slp_z", "z_grad", "pf_ancc_delta_1")]
FEATS = sorted(set(EVID + DISAG + DIP + STD))
print(f"集約する証拠特徴 {len(FEATS)}列  候補{len(CAND)}本  S={S}行/segment", flush=True)

need = ["well", "target", "last_known_tvt", "md_since"] + CAND + FEATS
tb = pq.read_table(ART / "data" / "train.parquet", columns=need).to_pandas()
tgt = tb["target"].to_numpy(np.float64)
lk = tb["last_known_tvt"].to_numpy(np.float64)
wcode = tb["well"].astype("category").cat.codes.to_numpy()
mds = tb["md_since"].to_numpy(np.float64)
Ccand = np.stack([tb[c].to_numpy(np.float64) - lk for c in CAND], axis=1)     # target基準候補
F = tb[FEATS].to_numpy(np.float64)
del tb
resid = combo - tgt
N = len(tgt)

# --- 坑井×MD順に整列し segment id を振る ---
order = np.lexsort((mds, wcode))
w_s = wcode[order]
b0 = np.diff(np.r_[0, np.flatnonzero(np.diff(w_s)) + 1, len(w_s)])   # 坑井ごと行数(連続)
segid = np.empty(N, np.int64); sw = []; i = 0; sc = 0
for n in b0:
    k = (np.arange(n) // S)                                          # 坑井内 segment index
    segid[order[i:i + n]] = sc + k
    nseg = k.max() + 1
    sw += [w_s[i]] * nseg                                            # 各segmentの坑井
    sc += nseg; i += n
sw = np.array(sw)
NS = sc
print(f"segment数 {NS} (坑井{len(np.unique(wcode))}本, 平均{NS/len(np.unique(wcode)):.0f} seg/坑井)", flush=True)

# --- segment集約特徴 ---
def agg(a):  # segmentごと mean/std
    mean = np.zeros(NS); std = np.zeros(NS)
    np.add.at(mean, segid, np.where(np.isfinite(a), a, 0))
    cnt = np.zeros(NS); np.add.at(cnt, segid, np.isfinite(a).astype(float))
    mean = mean / np.maximum(cnt, 1)
    sq = np.zeros(NS); np.add.at(sq, segid, np.where(np.isfinite(a), a * a, 0))
    std = np.sqrt(np.maximum(sq / np.maximum(cnt, 1) - mean ** 2, 0))
    return mean, std

Xs = []
for j in range(F.shape[1]):
    m, s = agg(F[:, j]); Xs += [m, s]
# 候補の segment内 中央 と 候補間spread
cand_seg = np.stack([agg(Ccand[:, j])[0] for j in range(Ccand.shape[1])], axis=1)  # (NS,15)
Xs.append(np.nanmedian(cand_seg, axis=1))
Xs.append(np.nanstd(cand_seg, axis=1))                                  # 候補間の枝ばらつき
Xs.append(agg(mds)[0])
Xs.append(agg(combo)[0])
Xseg = np.stack(Xs, axis=1)
Xseg = np.nan_to_num(Xseg, nan=0.0)

# segment target = combo残差の中央値
seg_off = np.zeros(NS)
tmp = [[] for _ in range(NS)]
# 高速に: sortして区間median
o2 = np.argsort(segid); sids = segid[o2]; rr = resid[o2]
bnd = np.r_[0, np.flatnonzero(np.diff(sids)) + 1, NS and len(sids)]
for a, b in zip(bnd[:-1], bnd[1:]):
    seg_off[sids[a]] = np.nanmedian(rr[a:b])
oracle_rmse_row = None

print(f"segment offset: 中央|{np.median(np.abs(seg_off)):.2f}|ft  std {seg_off.std():.2f}", flush=True)

# --- GroupKFold(坑井) で segment offset 予測 ---
gkf = GroupKFold(n_splits=NFOLD)
pred_off = np.zeros(NS)
for tr, va in gkf.split(Xseg, seg_off, groups=sw):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.7, min_child_samples=40,
                          n_jobs=4, verbose=-1)
    m.fit(Xseg[tr], seg_off[tr])
    pred_off[va] = m.predict(Xseg[va])

# --- 予測offsetを行へ展開して補正 ---
row_off_pred = pred_off[segid]
row_off_oracle = seg_off[segid]
fin = np.isfinite(resid)
def rmse(e): e = e[np.isfinite(e)]; return float(np.sqrt(np.mean(e ** 2)))
r_base = rmse(resid[fin])
r_corr = rmse((resid - row_off_pred)[fin])
r_orac = rmse((resid - row_off_oracle)[fin])
corr = np.corrcoef(pred_off, seg_off)[0, 1]
print(f"\n=== segment offset 回帰(非oracle, GroupKFold坑井) ===")
print(f"  combo baseline             RMSE = {r_base:.4f}")
print(f"  combo − pred_offset(非oracle) RMSE = {r_corr:.4f}   ({r_corr - r_base:+.4f})")
print(f"  combo − oracle_offset(S={S})  RMSE = {r_orac:.4f}   ← このsegment粒度の上限")
print(f"  予測offset vs 真offset 相関 = {corr:.3f}   (枝offsetがどれだけ当たるか)")
print(f"  回収率 = {(r_base - r_corr)/(r_base - r_orac)*100:.1f}%  (oracle到達分のうち非oracleで取れた割合)")

# --- 対照: 行レベル(集約なし)で同特徴から残差予測 ---
print(f"\n=== 対照: 行レベル(集約なし)で残差予測 ===")
Xrow = np.nan_to_num(np.column_stack([F, np.nanmedian(Ccand, axis=1), np.nanstd(Ccand, axis=1), mds, combo]), nan=0.0)
sub = np.arange(N) % 4 == 0     # 高速化のため1/4サブサンプルで学習(予測は全行)
pred_row = np.zeros(N)
for tr, va in gkf.split(np.zeros(N), resid, groups=wcode):
    trs = tr[sub[tr]]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.7, min_child_samples=100,
                          n_jobs=4, verbose=-1)
    m.fit(Xrow[trs], resid[trs])
    pred_row[va] = m.predict(Xrow[va])
r_row = rmse((resid - pred_row)[fin])
print(f"  combo − pred_row(行レベル) RMSE = {r_row:.4f}   ({r_row - r_base:+.4f})")
print(f"  → segment集約の上乗せ = {r_row - r_corr:+.4f}ft (負=segmentが行より良い=集約が効く)")

# --- 図 ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].bar(["combo", "−pred\n(seg,非oracle)", "−pred\n(行)", "−oracle\n(seg)"],
          [r_base, r_corr, r_row, r_orac], color=["#900", "#4c9", "#a84", "#48a"])
for k, v in enumerate([r_base, r_corr, r_row, r_orac]): ax[0].text(k, v + .05, f"{v:.3f}", ha="center", fontsize=9)
ax[0].set_ylabel("overall RMSE"); ax[0].set_title(f"segment offset回帰(S={S})")
sm = np.isfinite(seg_off) & np.isfinite(pred_off)
ax[1].scatter(seg_off[sm], pred_off[sm], s=3, alpha=.2)
lim = np.percentile(np.abs(seg_off[sm]), 99)
ax[1].plot([-lim, lim], [-lim, lim], "r--"); ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
ax[1].set_xlabel("真 segment offset"); ax[1].set_ylabel("予測"); ax[1].set_title(f"offset 予測 vs 真 (相関{corr:.2f})"); ax[1].grid(alpha=.3)
ax[2].hist(seg_off[sm], bins=60, alpha=.6, label="真offset")
ax[2].hist(pred_off[sm], bins=60, alpha=.6, label="予測")
ax[2].set_xlim(-lim, lim); ax[2].legend(); ax[2].set_title("offset分布(予測は縮む=当てきれない)")
fig.suptitle("診断D: 証拠を区間に集約して枝offsetを当てられるか(非oracle学習)", fontsize=13)
fig.tight_layout(); fig.savefig(ART / "diag_segment_regress.png", dpi=115)
print(f"\n[saved] {ART/'diag_segment_regress.png'}")
