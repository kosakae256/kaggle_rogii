# -*- coding: utf-8 -*-
"""診断C: 「区間に上げると効くか」を非oracleの後処理で確かめる高速スキャン(学習不要)。
   現combo(6.42)に対し、PF確定候補15本(target基準)と区間平滑を使った補正を当てRMSEが動くか測る。
   - consensus: 15候補のrow中央値(committed=枝にコミット) 単体 & combo↔consensusブレンド掃引
   - snap: comboを各rowで最近傍候補にスナップ(MSE平均→単枝コミット)
   - medfilt: combo/consensus を坑井内メディアンフィルタ(spurious flip除去) 窓掃引
   全て真値未使用。oracle 0.18(診断B)に非oracleでどこまで迫れるかの下見。
   出力: コンソール + artifacts_v69_4/diag_multicut_scan.png"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import median_filter

ART = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")
CANDS = [f"pf_{k}_{i}" for k in ("ancc", "self", "nbr") for i in range(1, 6)]   # 15確定候補

# combo OOF
g = np.load(ART / "gbm_v41_oof.npy").astype(np.float64)
z = np.load(ART / "nn_oof_combo.npz")
nn = np.mean([z[k] for k in ("mdn2_s0", "mdn2_s1", "mdn2_s2")], axis=0)
combo = 0.54 * nn + 0.46 * g

cols = ["well", "target", "last_known_tvt", "md_since"] + CANDS
tb = pq.read_table(ART / "data" / "train.parquet", columns=cols).to_pandas()
tgt = tb["target"].to_numpy(np.float64)
lk = tb["last_known_tvt"].to_numpy(np.float64)
wcode = tb["well"].astype("category").cat.codes.to_numpy()
mds = tb["md_since"].to_numpy(np.float64)
# 候補を target 基準へ: cand_target = pf_abs - last_known
Ccand = np.stack([tb[c].to_numpy(np.float64) - lk for c in CANDS], axis=1)   # (N,15)
del tb
fin = np.isfinite(tgt) & np.isfinite(combo)
N = len(tgt)
print(f"rows={N}  候補15本  finite={fin.mean()*100:.1f}%", flush=True)


def rmse(p):
    e = (p - tgt)[fin & np.isfinite(p)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan


print(f"\nbaseline combo          RMSE = {rmse(combo):.4f}")

# --- consensus: 候補row中央値(NaN無視) ---
cons = np.nanmedian(Ccand, axis=1)
cand_n = np.isfinite(Ccand).sum(1)
print(f"consensus(15候補中央値) RMSE = {rmse(cons):.4f}  (候補数中央 {int(np.median(cand_n))})")

print("\n=== combo ↔ consensus ブレンド掃引 ===")
best = (1e9, None)
for a in np.linspace(0, 1, 11):
    p = (1 - a) * combo + a * cons
    r = rmse(p)
    if r < best[0]: best = (r, a)
    print(f"  a={a:.1f} (a=consensus重み): RMSE {r:.4f}")
print(f"  best blend a={best[1]:.1f} RMSE={best[0]:.4f}  (combo {rmse(combo):.4f} 比 {best[0]-rmse(combo):+.4f})")

# --- snap: comboを最近傍候補へ ---
diff = np.abs(Ccand - combo[:, None])
diff[~np.isfinite(diff)] = 1e9
snap = Ccand[np.arange(N), np.argmin(diff, axis=1)]
snap = np.where(np.isfinite(snap), snap, combo)
print(f"\nsnap(combo→最近傍候補)   RMSE = {rmse(snap):.4f}  ({rmse(snap)-rmse(combo):+.4f})")
# snapを軽くブレンド
for a in (0.25, 0.5):
    print(f"  combo↔snap a={a}: RMSE {rmse((1-a)*combo+a*snap):.4f}")

# --- 坑井内メディアンフィルタ(spurious flip除去) ---
order = np.lexsort((mds, wcode)); w_s = wcode[order]
b0 = np.diff(np.r_[0, np.flatnonzero(np.diff(w_s)) + 1, len(w_s)])
inv = np.argsort(order)


def well_medfilt(arr, win):
    a_s = arr[order]; out = a_s.copy(); i = 0
    for n in b0:
        sl = slice(i, i + n); i += n
        seg = a_s[sl]; m = np.isfinite(seg)
        if m.sum() >= 3:
            f = np.where(m, seg, np.nanmedian(seg[m]) if m.any() else 0.0)
            out[sl] = median_filter(f, size=min(win, n), mode="nearest")
    return out[inv]


print("\n=== 坑井内メディアンフィルタ(窓掃引) ===")
mf_res = {}
for win in (7, 15, 31, 51, 101):
    pc = well_medfilt(combo, win); pk = well_medfilt(cons, win)
    mf_res[win] = (rmse(pc), rmse(pk))
    print(f"  win={win:3d}: combo-medfilt {rmse(pc):.4f} ({rmse(pc)-rmse(combo):+.4f}) / consensus-medfilt {rmse(pk):.4f}")

# --- 図 ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
aa = np.linspace(0, 1, 11)
ax[0].plot(aa, [rmse((1 - a) * combo + a * cons) for a in aa], "o-")
ax[0].axhline(rmse(combo), c="r", ls="--", label=f"combo {rmse(combo):.3f}")
ax[0].set_xlabel("consensus重み a"); ax[0].set_ylabel("overall RMSE"); ax[0].set_title("combo↔consensus ブレンド"); ax[0].legend(); ax[0].grid(alpha=.3)
wins = list(mf_res); ax[1].plot(wins, [mf_res[w][0] for w in wins], "o-", label="combo-medfilt")
ax[1].plot(wins, [mf_res[w][1] for w in wins], "s-", label="consensus-medfilt")
ax[1].axhline(rmse(combo), c="r", ls="--", label=f"combo {rmse(combo):.3f}")
ax[1].set_xlabel("メディアン窓(行)"); ax[1].set_ylabel("overall RMSE"); ax[1].set_title("坑井内メディアンフィルタ"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].bar(["combo", "consensus", "snap", "best\nblend", f"medfilt\nbest"],
          [rmse(combo), rmse(cons), rmse(snap), best[0], min(min(v) for v in mf_res.values())],
          color=["#900", "#48a", "#a84", "#4c9", "#4c9"])
ax[2].axhline(rmse(combo), c="r", ls="--"); ax[2].set_ylabel("RMSE"); ax[2].set_title("非oracle後処理まとめ\n(oracle区間=0.18)")
ax[2].set_ylim(6.0, max(rmse(cons), rmse(snap)) + .3)
fig.suptitle("診断C: 非oracle後処理スキャン — 区間概念でcomboを改善できるか", fontsize=13)
fig.tight_layout(); fig.savefig(ART / "diag_multicut_scan.png", dpi=115)
print(f"\n[saved] {ART/'diag_multicut_scan.png'}")
