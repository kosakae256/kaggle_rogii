# -*- coding: utf-8 -*-
"""診断E: 区間offsetの正体を「連続ドリフト(dip累積誤差)」か「離散flip(枝選択)」に分離する。
   坑井内 residual(=combo−target) から datum(坑井中央値)を除いた within を、
     within = drift(直線, dip/rate系統誤差) + steppy(非線形の段=枝flip) + fine(≈0)
   に分解し、within分散のうち drift/steppy の割合、段差のサイズ分布(サイクル~15ft帯に集まる=枝),
   誤差が anchor からの距離(md_since)で伸びるか(=ドリフト署名) を測る。
   ・drift支配 → レバー= smoother/rate再推定/v36逆DA(系統誤差の補正)
   ・steppy支配 → レバー= PF内multi-cut/情報(枝の離散選択)
   出力: コンソール + artifacts_v69_4/diag_offset_nature.png"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import median_filter

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")
W = 51                       # 段(step_level)を拾う窓

g = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)
z = np.load(ART / "nn_oof_combo.npz")
combo = 0.54 * np.mean([z[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], axis=0) + 0.46 * g
tb = pq.read_table(ART / "data" / "train.parquet", columns=["well", "target", "md_since"]).to_pandas()
tgt = tb["target"].to_numpy(np.float64)
wcode = tb["well"].astype("category").cat.codes.to_numpy()
mds = tb["md_since"].to_numpy(np.float64)
r = combo - tgt
fin = np.isfinite(r) & np.isfinite(mds)

order = np.lexsort((mds, wcode)); w_s = wcode[order]
b0 = np.diff(np.r_[0, np.flatnonzero(np.diff(w_s)) + 1, len(w_s)])
r_s = r[order]; md_s = mds[order]; fin_s = fin[order]

SS_within = SS_drift = SS_steppy = SS_fine = 0.0
slopes = []; jumps = []          # 坑井drift勾配 / 段差サイズ
r2lin = []                       # within の直線当てはまり(高=ドリフト)
md_rmse_num = np.zeros(12); md_rmse_den = np.zeros(12)   # md_since帯別 within RMSE
mdbins = np.linspace(0, np.nanpercentile(mds[fin], 98), 13)
examples = []
i = 0
for n in b0:
    sl = slice(i, i + n); i += n
    fm = fin_s[sl]
    if fm.sum() < 60: continue
    rr = r_s[sl][fm]; mm = md_s[sl][fm]
    within = rr - np.median(rr)                          # datum(坑井b)除去
    # 直線drift当て
    A = np.polyfit(mm, within, 1); lin = np.polyval(A, mm)
    slopes.append(A[0])
    ss_w = np.sum(within ** 2)
    if ss_w < 1e-6: continue
    r2lin.append(1 - np.sum((within - lin) ** 2) / ss_w)
    # 段(低周波)= rolling median、その非線形部 = steppy
    step_level = median_filter(within, size=min(W, len(within)), mode="nearest")
    steppy = step_level - lin
    fine = within - step_level
    SS_within += ss_w
    SS_drift += np.sum(lin ** 2)
    SS_steppy += np.sum(steppy ** 2)
    SS_fine += np.sum(fine ** 2)
    # 段差サイズ: step_level の有意ジャンプ
    d = np.diff(step_level)
    big = np.abs(d) > 0.3
    if big.any():
        # 連続ジャンプをまとめて1段に
        idx = np.flatnonzero(big)
        cur = d[idx[0]];
        for k in range(1, len(idx)):
            if idx[k] == idx[k-1] + 1: cur += d[idx[k]]
            else:
                if abs(cur) > 3: jumps.append(cur)
                cur = d[idx[k]]
        if abs(cur) > 3: jumps.append(cur)
    # md帯別 within RMSE
    bi = np.clip(np.searchsorted(mdbins, mm) - 1, 0, 11)
    np.add.at(md_rmse_num, bi, within ** 2); np.add.at(md_rmse_den, bi, 1)
    if len(examples) < 3 and np.max(np.abs(steppy)) > 8:
        examples.append((mm, within, lin, step_level))

slopes = np.array(slopes); jumps = np.array(jumps); r2lin = np.array(r2lin)
print(f"評価坑井 {len(slopes)}本\n")
print("=== within(坑井内offset)分散の分解 ===")
print(f"  drift(直線=dip/rate系統) : {SS_drift/SS_within*100:5.1f}%")
print(f"  steppy(非線形の段=枝flip) : {SS_steppy/SS_within*100:5.1f}%")
print(f"  fine(≈0)                 : {SS_fine/SS_within*100:5.1f}%")
verdict = "ドリフト支配→smoother/rate/v36逆DA" if SS_drift > SS_steppy else "離散flip支配→PF内multi-cut/情報"
print(f"  → {verdict}")
print(f"\n=== within の直線当てはまり R²(高=ドリフト) ===")
print(f"  中央 {np.median(r2lin):.3f} / R²>0.5 の坑井 {np.mean(r2lin>0.5)*100:.1f}% / R²<0.2 {np.mean(r2lin<0.2)*100:.1f}%")
print(f"  drift勾配 |slope| 中央 {np.median(np.abs(slopes)):.4f} ft/md単位  (md_since全長×勾配≈累積ドリフト量)")
print(f"\n=== 段差サイズ分布(枝flipなら±サイクル~15ft帯に集まる) ===")
if len(jumps):
    aj = np.abs(jumps)
    print(f"  検出段差 {len(jumps)}個  |段差|中央 {np.median(aj):.1f}ft")
    for lo, hi in [(3,8),(8,12),(12,20),(20,1e9)]:
        m = (aj>=lo)&(aj<hi); print(f"    |段差| {lo:>3}-{hi if hi<1e9 else '∞':>3}ft: {m.mean()*100:5.1f}%")
print(f"\n=== md_since(anchorからの距離)帯別 within RMSE(伸びる=ドリフト署名) ===")
mr = np.sqrt(md_rmse_num/np.maximum(md_rmse_den,1))
for k in range(0,12,2):
    print(f"  md {mdbins[k]:6.0f}-{mdbins[k+1]:6.0f}: within RMSE {mr[k]:5.2f}")

# 図
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(2,3,1)
ax.bar(["drift\n(直線)","steppy\n(段=flip)","fine"], [SS_drift/SS_within*100, SS_steppy/SS_within*100, SS_fine/SS_within*100],
       color=["#48a","#e74","#ccc"]); ax.set_ylabel("within分散 %"); ax.set_title(f"①offset分散の分解\n{verdict}")
ax = fig.add_subplot(2,3,2)
ax.hist(r2lin, bins=40, color="#48a"); ax.axvline(np.median(r2lin), c="k", ls="--")
ax.set_xlabel("within の直線当てはまり R²"); ax.set_title(f"②高=ドリフト坑井 (中央{np.median(r2lin):.2f})")
ax = fig.add_subplot(2,3,3)
if len(jumps):
    ax.hist(np.abs(jumps), bins=np.arange(0,40,1.5), color="#e74")
    for c in (15,): ax.axvline(c, c="k", ls="--", label=f"1サイクル~{c}ft")
    ax.legend()
ax.set_xlabel("|段差|(ft)"); ax.set_title("③段差サイズ(枝ならサイクル帯に集中)")
ax = fig.add_subplot(2,3,4)
ax.plot((mdbins[:-1]+mdbins[1:])/2, mr, "o-"); ax.set_xlabel("md_since(anchorからの距離)"); ax.set_ylabel("within RMSE")
ax.set_title("④距離で伸びる=ドリフト署名"); ax.grid(alpha=.3)
for j,(mm,within,lin,step) in enumerate(examples[:2]):
    ax = fig.add_subplot(2,3,5+j)
    ax.plot(mm, within, ".", ms=2, color="#aaa", label="within(offset)")
    ax.plot(mm, lin, "-", color="#48a", label="drift(直線)")
    ax.plot(mm, step, "-", color="#e74", label="step_level")
    ax.set_xlabel("md_since"); ax.set_ylabel("offset(ft)"); ax.legend(fontsize=8); ax.set_title(f"⑤例{j+1}")
fig.suptitle("診断E: 区間offsetの正体 — 連続ドリフト vs 離散flip", fontsize=13)
fig.tight_layout(); fig.savefig(ART/"diag_offset_nature.png", dpi=115)
print(f"\n[saved] {ART/'diag_offset_nature.png'}")
