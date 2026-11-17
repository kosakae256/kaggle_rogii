# -*- coding: utf-8 -*-
"""v77 夜間キャンペーンのサマリ図(3面): 単体RMSE / blend積み上げ / per-well ECDF"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
LS = 10.536319667317654
anc = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
p5a = pickle.loads((ART / "p5_out_dip03_773w.pkl").read_bytes())
p6e = pickle.loads((ART / "p6_out_lp13emt015_773w.pkl").read_bytes())

W = [0.52, 0.12, 0.19, 0.17]
per_pf, per_bl = {}, {}
E = {k: [] for k in ("pf", "anc", "p5", "p6", "bl")}
for wid in p5a:
    if wid not in bare or wid not in p6e or wid not in anc: continue
    d = bare[wid]; ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)
    tv = d["true"]
    T = min(len(epf), len(tv), len(anc[wid]["tvt"]), len(p5a[wid]["pred"]), len(p6e[wid]["pred"]))
    ea = anc[wid]["tvt"][:T] - tv[:T]
    e5 = np.asarray(p5a[wid]["pred"][:T], np.float64) - tv[:T]
    e6 = np.asarray(p6e[wid]["pred"][:T], np.float64) - tv[:T]
    ep = epf[:T]
    ebl = W[0] * ep + W[1] * ea + W[2] * e5 + W[3] * e6
    m = np.isfinite(ea) & np.isfinite(e5) & np.isfinite(e6) & np.isfinite(ep)
    for k, e_ in zip(("pf", "anc", "p5", "p6", "bl"), (ep, ea, e5, e6, ebl)):
        E[k].append(e_[m])
    per_pf[wid] = float(np.sqrt(np.mean(ep[m] ** 2)))
    per_bl[wid] = float(np.sqrt(np.mean(ebl[m] ** 2)))
R = {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in E.items()}

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
# (1) 単体
names = ["素PF", "K16錨", "P5(dip慣性FB)", "P6(低域GR FB\nEMT0.15)"]
vals = [R["pf"], R["anc"], R["p5"], R["p6"]]
b = ax[0].bar(names, vals, color=["#4477aa", "#66ccee", "#228833", "#ccbb44"])
ax[0].bar_label(b, fmt="%.2f"); ax[0].set_ylabel("773井 pooled RMSE (ft)")
ax[0].set_title("単体スコア(773井)")
# (2) 積み上げ
steps = ["素PF\n8.665", "+K16錨\n7.258", "+P5\n7.149", "+P6(EMT)\n7.128"]
sv = [8.665, 7.258, 7.149, 7.128]
ax[1].plot(steps, sv, "o-", lw=2, color="#aa3377")
for x, y in zip(steps, sv): ax[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
ax[1].set_title("blend積み上げ(4脚, w=0.52/0.12/0.19/0.17)")
ax[1].set_ylabel("773井 pooled RMSE (ft)"); ax[1].grid(alpha=0.3)
# (3) ECDF
for lab, per, c in (("素PF", per_pf, "#4477aa"), ("4脚blend", per_bl, "#aa3377")):
    v = np.sort(list(per.values())); ax[2].plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, color=c)
ax[2].set_xlim(0, 30); ax[2].set_xlabel("per-well RMSE (ft)"); ax[2].set_ylabel("ECDF")
ax[2].legend(); ax[2].grid(alpha=0.3); ax[2].set_title("井戸別RMSEのECDF")
med_pf = np.median(list(per_pf.values())); med_bl = np.median(list(per_bl.values()))
ax[2].annotate(f"中央値 {med_pf:.2f}→{med_bl:.2f}", (0.45, 0.15), xycoords="axes fraction")
fig.suptitle("v77: K16物理錨 × 格子復号 — 夜間キャンペーン結果(2026-07-22)", fontsize=13)
fig.tight_layout()
fp = ART / "v77_summary.png"
fig.savefig(fp, dpi=130)
print(f"[saved] {fp}")
print({k: round(v, 4) for k, v in R.items()}, "中央値", round(med_pf, 3), "→", round(med_bl, 3))
