# -*- coding: utf-8 -*-
"""
v81 — 事前区間(prefix=TVT_input既知帯)の解剖(CPU)
==================================================
方法を考える前のデータ分析(ユーザー方針):
  ①prefix長(行/MD) ②縦掘り性: |dZ/dMD| prefix vs eval ③TVTスイープ幅: prefixが典型井のTVT範囲を
  どれだけ掃くか(typewell較正に使える情報量) ④prefix TVT範囲とeval TVT範囲の重なり
  ⑤GR品質: prefix残差σとeval誤差の関係(既知) ⑥prefix終端付近のdip vs eval序盤dip の持続性
出力: 図 v81/artifacts_v81/prefix_anatomy.png + 統計print
"""
import io, sys, glob, os, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v81" / "artifacts_v81"; ART.mkdir(parents=True, exist_ok=True)
BARE = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
LS = 10.536319667317654

rows = []
for fp in sorted(glob.glob(str(TRAIN / "*__horizontal_well.csv"))):
    wid = os.path.basename(fp).split("__")[0]
    hw = pd.read_csv(fp, usecols=["MD", "Z", "GR", "TVT", "TVT_input"])
    twf = TRAIN / f"{wid}__typewell.csv"
    tw = pd.read_csv(twf, usecols=["TVT", "GR"]).sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(kn) < 20 or len(ev) < 60: continue
    kmd = kn["MD"].to_numpy(float); kz = kn["Z"].to_numpy(float); ktvt = kn["TVT_input"].to_numpy(float)
    emd = ev["MD"].to_numpy(float); ez = ev["Z"].to_numpy(float); etvt = ev["TVT"].to_numpy(float)
    dzk = np.abs(np.gradient(kz, kmd)); dze = np.abs(np.gradient(ez, emd))
    tw_lo, tw_hi = float(tw["TVT"].min()), float(tw["TVT"].max())
    twspan = max(tw_hi - tw_lo, 1e-6)
    ksp = (np.nanmax(ktvt) - np.nanmin(ktvt))
    esp = (np.nanmax(etvt) - np.nanmin(etvt))
    # 重なり: eval TVT範囲のうちprefixが掃いた範囲に入る割合
    lo, hi = np.nanmin(ktvt), np.nanmax(ktvt)
    cover = float(np.mean((etvt >= lo) & (etvt <= hi)))
    # prefix末尾dip vs eval序盤dip(S空間)
    def dip(md, z, t, n1, n2):
        s = t + z
        if len(s) < abs(n2 - n1) + 3: return np.nan
        seg = slice(n1, n2)
        dm = np.gradient(md[seg]); dsv = np.gradient(s[seg])
        m = dm > 0
        return float(np.median(dsv[m] / dm[m])) if m.sum() > 3 else np.nan
    d_tail = dip(kmd, kz, ktvt, -60, len(kmd))
    d_head = dip(emd, ez, etvt, 0, 60)
    # PF誤差(bare0)
    pf = np.nan
    if wid in BARE:
        d = BARE[wid]; ll = d["ll"].astype(np.float64)
        ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
        e = (ww[:, None] * d["err"].astype(np.float64)).sum(0)
        pf = float(np.sqrt(np.nanmean(e ** 2)))
    rows.append(dict(wid=wid, klen=len(kn), kmd=float(kmd[-1] - kmd[0]), elen=len(ev),
                     dzk=float(np.nanmedian(dzk)), dze=float(np.nanmedian(dze)),
                     ksp=ksp, esp=esp, ksp_frac=ksp / twspan, cover=cover,
                     d_tail=d_tail, d_head=d_head, pf=pf))
df = pd.DataFrame(rows)
df.to_csv(ART / "prefix_anatomy.csv", index=False)
print(f"{len(df)}井")
print(f"prefix行数: 中央{df.klen.median():.0f} p10={df.klen.quantile(.1):.0f} p90={df.klen.quantile(.9):.0f}")
print(f"prefix MD長: 中央{df.kmd.median():.0f}ft")
print(f"縦掘り性 |dZ/dMD| 中央: prefix={df.dzk.median():.3f} vs eval={df.dze.median():.3f}")
print(f"prefix TVTスイープ: 中央{df.ksp.median():.0f}ft (typewell範囲の{df.ksp_frac.median()*100:.0f}%)")
print(f"eval TVTスパン: 中央{df.esp.median():.0f}ft")
print(f"evalのTVTがprefix掃引範囲内にある行率: 中央{df.cover.median()*100:.0f}% / 平均{df.cover.mean()*100:.0f}%")
c = df[["d_tail", "d_head"]].dropna()
print(f"prefix末尾dip vs eval序盤dip: 相関 {np.corrcoef(c.d_tail, c.d_head)[0,1]:.3f} (n={len(c)})")
m = df.dropna(subset=["pf"])
print(f"prefixスイープ幅 vs PF誤差: 相関 {np.corrcoef(m.ksp_frac, m.pf)[0,1]:.3f}")
print(f"prefix行数 vs PF誤差: 相関 {np.corrcoef(m.klen, m.pf)[0,1]:.3f}")
print(f"カバー率 vs PF誤差: 相関 {np.corrcoef(m.cover, m.pf)[0,1]:.3f}")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0,0].hist(df.klen, bins=50); ax[0,0].set_title("prefix行数")
ax[0,1].hist(np.clip(df.dzk,0,1.2), bins=50, alpha=.6, label="prefix")
ax[0,1].hist(np.clip(df.dze,0,1.2), bins=50, alpha=.6, label="eval")
ax[0,1].legend(); ax[0,1].set_title("|dZ/dMD|(縦掘り性)")
ax[0,2].hist(np.clip(df.ksp_frac,0,1.5)*100, bins=50); ax[0,2].set_title("prefixのtypewell範囲スイープ率(%)")
ax[1,0].hist(df.cover*100, bins=50); ax[1,0].set_title("eval行のprefix掃引範囲内率(%)")
c2 = df[["d_tail","d_head"]].dropna()
ax[1,1].scatter(c2.d_tail, c2.d_head, s=4, alpha=.4)
ax[1,1].set_xlabel("prefix末尾dip"); ax[1,1].set_ylabel("eval序盤dip"); ax[1,1].set_title("dip持続性")
ax[1,1].set_xlim(-.1,.1); ax[1,1].set_ylim(-.1,.1)
ax[1,2].scatter(m.cover*100, m.pf, s=4, alpha=.4)
ax[1,2].set_xlabel("カバー率(%)"); ax[1,2].set_ylabel("PF井戸RMSE"); ax[1,2].set_ylim(0,30)
ax[1,2].set_title("カバー率 vs PF誤差")
fig.suptitle("v81 事前区間(prefix)の解剖", fontsize=13); fig.tight_layout()
fig.savefig(ART / "prefix_anatomy.png", dpi=130)
print("[saved] prefix_anatomy.png / prefix_anatomy.csv")
