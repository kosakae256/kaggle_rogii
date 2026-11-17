# -*- coding: utf-8 -*-
"""診断B: 現combo(v69_4)の残差を「3スケール分解」して行レベル推論の害を数値化。
   TVT予測は datum(b, 坑井単位) / branch(サイクル枝, 区間単位) / refine(細, 行単位) の3スケール合成。
   残差 r = combo_pred − target(=真TVT−last_known と同基準なので r=pred_target−true_target)を、
   ① |r|帯別のRMSE²寄与(fine vs branch-scale) ② 坑井中央値除去(=datum oracle) ③ 区間中央値除去(=branch oracle)
   で分解し、各スケールの"賞金"を測る。加えて残差の坑井内自己相関・区間ラン長・中途flip数で「行レベルが不適」を裏取り。
   出力: コンソール + artifacts_v69_4/diag_branch_flip.png"""
import io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import median_filter

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")
BR = 7.5          # branch-scale しきい(1サイクル ~15ft の半分。これ以上=枝違いスケール)
WIN = 101         # 区間中央値(rolling median)の窓行数

# --- OOF: combo = 0.54*mean(nn 3seed) + 0.46*gbm (submit_pack: w_nn=0.54, run=combo) ---
g = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)
z = np.load(ART / "nn_oof_combo.npz")
nn = np.mean([z[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], axis=0)
W_NN = 0.54
combo = W_NN * nn + (1 - W_NN) * g
print(f"OOF rows={len(combo)}  combo=0.54*nn+0.46*gbm", flush=True)

# --- target / well / md order ---
tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "md_since"]).to_pandas()
tgt = tb["target"].to_numpy(np.float64)
wcode = tb["well"].astype("category").cat.codes.to_numpy()
mds = tb["md_since"].to_numpy(np.float64)
r = combo - tgt                                        # 残差(ft, TVT基準と同一)
fin = np.isfinite(r) & np.isfinite(tgt)
print(f"finite={fin.mean()*100:.1f}%  overall RMSE(pre-pp)={np.sqrt(np.mean(r[fin]**2)):.4f}  (post-pp報告値6.448)")


def rmse(x): x = x[np.isfinite(x)]; return float(np.sqrt(np.mean(x**2))) if x.size else np.nan


# ================= ① |r|帯別 RMSE²寄与 =================
print("\n=== ① |残差|帯別の RMSE²寄与(fine vs branch-scale) ===")
bands = [(0, 3), (3, 7.5), (7.5, 12), (12, 20), (20, 1e9)]
tot_sq = np.sum(r[fin] ** 2); N = fin.sum()
for lo, hi in bands:
    m = fin & (np.abs(r) >= lo) & (np.abs(r) < hi)
    sq = np.sum(r[m] ** 2)
    print(f"  |r| {lo:>4}-{hi if hi<1e9 else '∞':>4}ft : {m.sum()/N*100:5.1f}%行  RMSE²寄与 {sq/tot_sq*100:5.1f}%")
mbr = fin & (np.abs(r) >= BR)
print(f"  → branch-scale(|r|≥{BR}ft): {mbr.sum()/N*100:.1f}%の行が RMSE²の {np.sum(r[mbr]**2)/tot_sq*100:.1f}% を占める")

# ================= ②③ 3スケール分解(oracleでb/branch除去) =================
# 坑井順・MD順に整列 → 坑井ごとに median(=datum b) と rolling-median(=区間branch offset) を"真値使って"除去
order = np.lexsort((mds, wcode))
r_s = r[order]; w_s = wcode[order]; fin_s = fin[order]
r_datum = r_s.copy(); r_seg = r_s.copy()
b0 = np.diff(np.r_[0, np.flatnonzero(np.diff(w_s)) + 1, len(w_s)])   # 各坑井の行数(連続ブロック)
i = 0
for n in b0:
    sl = slice(i, i + n); i += n
    rr = r_s[sl]; fm = fin_s[sl]
    if fm.sum() >= 5:
        r_datum[sl] = rr - np.median(rr[fm])                        # ②坑井中央値除去=datum oracle
        rf = median_filter(np.where(fm, rr, np.median(rr[fm])), size=min(WIN, n), mode="nearest")
        r_seg[sl] = rr - rf                                          # ③区間中央値除去=branch(区間offset) oracle
print("\n=== ②③ 3スケール分解(oracleで各スケールの真offsetを除去した時の残RMSE) ===")
print(f"  full(現状)                    RMSE = {rmse(r_s[fin_s]):.4f}")
print(f"  −datum(坑井b, oracle)         RMSE = {rmse(r_datum[fin_s]):.4f}   ← 坑井単位の基準を当てるだけで到達")
print(f"  −segment(区間branch, oracle)  RMSE = {rmse(r_seg[fin_s]):.4f}   ← 区間単位の枝を全部当てた床(=fine限界)")
d_full = rmse(r_s[fin_s]); d_dat = rmse(r_datum[fin_s]); d_seg = rmse(r_seg[fin_s])
print(f"  賞金: datumを当てる={d_full-d_dat:+.3f}ft / segment枝を当てる(追加)={d_dat-d_seg:+.3f}ft / fine床={d_seg:.3f}ft")

# ================= ④ 残差の坑井内構造(=行レベルが不適の証拠) =================
print("\n=== ④ 残差の坑井内構造(白色ノイズなら行レベルでOK/構造化なら区間モデルが要) ===")
ac1 = []; shift = []; nwell = 0
i = 0
for n in b0:
    sl = slice(i, i + n); i += n
    rr = r_s[sl]; fm = fin_s[sl]
    if fm.sum() < 50: continue
    x = rr[fm]; nwell += 1
    ac1.append(np.corrcoef(x[:-1], x[1:])[0, 1])                    # lag-1自己相関
    rf = median_filter(x, size=min(WIN, len(x)), mode="nearest")    # 区間offset
    shift.append(rf.max() - rf.min())                              # 坑井内の最大持続レベルシフト
ac1 = np.array(ac1); shift = np.array(shift)
print(f"  評価坑井 {nwell}本")
print(f"  残差 lag-1 自己相関: 中央 {np.nanmedian(ac1):.3f}  (白色ノイズ=0。高い=区間で同符号に外れ続ける=枝誤り)")
print(f"  坑井内 最大持続レベルシフト: 中央 {np.median(shift):.2f}ft  /  ≥{2*BR}ft(=途中で枝flip)の坑井 {np.mean(shift>=2*BR)*100:.1f}%")

# ================= 図 =================
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig = plt.figure(figsize=(15, 9))
# (a) 帯別RMSE²寄与
ax = fig.add_subplot(2, 3, 1)
lbl = [f"{lo}-{hi if hi<1e9 else '∞'}" for lo, hi in bands]
sq = [np.sum(r[fin & (np.abs(r) >= lo) & (np.abs(r) < hi)] ** 2) / tot_sq * 100 for lo, hi in bands]
ax.bar(lbl, sq, color=["#4c9","#4c9","#e74","#e74","#900"]); ax.set_ylabel("RMSE²寄与 %")
ax.set_title(f"①|残差|帯別RMSE²寄与\n(赤=branch-scale ≥{BR}ft)"); ax.tick_params(axis="x", rotation=30)
# (b) 3スケール分解
ax = fig.add_subplot(2, 3, 2)
ax.bar(["full", "−datum", "−segment"], [d_full, d_dat, d_seg], color=["#900","#e74","#4c9"])
ax.set_ylabel("RMSE (ft)"); ax.set_title("②③3スケール分解(oracle)\n各スケールを当てた時の到達RMSE")
for k, v in enumerate([d_full, d_dat, d_seg]): ax.text(k, v + .05, f"{v:.2f}", ha="center")
# (c) 自己相関ヒスト
ax = fig.add_subplot(2, 3, 3)
ax.hist(ac1[np.isfinite(ac1)], bins=40, color="#48a"); ax.axvline(0, c="k", ls="--")
ax.set_xlabel("残差 lag-1 自己相関"); ax.set_title(f"④残差自己相関(中央{np.nanmedian(ac1):.2f})\n0=白色/右=区間構造化")
# (d) レベルシフトECDF
ax = fig.add_subplot(2, 3, 4)
ss = np.sort(shift); ax.plot(ss, np.arange(len(ss)) / len(ss)); ax.axvline(2 * BR, c="r", ls="--", label=f"枝flip≥{2*BR}ft")
ax.set_xlabel("坑井内 最大持続レベルシフト(ft)"); ax.set_ylabel("ECDF"); ax.set_xlim(0, 40); ax.legend()
ax.set_title("坑井内で残差offsetがどれだけ動くか")
# (e)(f) flip例の坑井2本(shift最大級)
worst = np.argsort(shift)[::-1]
ci = 0; i = 0; blocks = []
for n in b0:
    blocks.append((i, n)); i += n
good = [w for w in worst if fin_s[blocks[w][0]:blocks[w][0]+blocks[w][1]].sum() >= 200][:2] if len(blocks) else []
# blocksはfin>=50でフィルタ前のindexとずれるので、shiftと同じ順で再走査
i = 0; k2 = 0; examples = []
for n in b0:
    sl = slice(i, i + n); i += n
    fm = fin_s[sl]
    if fm.sum() < 50: continue
    examples.append((sl, shift[k2])); k2 += 1
examples.sort(key=lambda t: -t[1])
for j, (sl, sh) in enumerate(examples[:2]):
    ax = fig.add_subplot(2, 3, 5 + j)
    fm = fin_s[sl]; xx = np.arange(sl.stop - sl.start)
    ax.plot(xx[fm], (combo[order][sl])[fm], ".", ms=2, label="combo pred", color="#e74")
    ax.plot(xx[fm], (tgt[order][sl])[fm], ".", ms=2, label="true", color="#333")
    ax.set_title(f"⑤flip例(shift={sh:.1f}ft)"); ax.set_xlabel("坑井内 行(MD順)"); ax.set_ylabel("target(TVT−lastknown)")
    ax.legend(fontsize=8)
fig.suptitle("診断B: v69_4 combo 残差の3スケール分解 — 行レベル推論が捨てているもの", fontsize=13)
fig.tight_layout()
fig.savefig(ART / "diag_branch_flip.png", dpi=115)
print(f"\n[saved] {ART/'diag_branch_flip.png'}")
