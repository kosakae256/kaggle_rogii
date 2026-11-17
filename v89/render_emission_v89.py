# -*- coding: utf-8 -*-
"""v89 — PFフリー emission(スコア画像)に 真δ経路 と DP経路 を重ねて診断。
真の尾根が emission に立っているか(=どんな大域法でも可能か)を目視。"""
import os, io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Meiryo"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dp_trace_v89 as dp

ART = dp.ART; EVD = dp.EVD; OFF = dp.OFF
# PF誤差でなく flat誤差で easy/hard
def flat_rmse(w):
    e = EVD[w]; er = e["st"] - e["y_ev"]; er = er[np.isfinite(er)]
    return float(np.sqrt(np.mean(er**2))) if len(er) else 1e9
W = sorted(EVD.keys()); W = [w for w in W if len(EVD[w]["dtrue"])>20]
errs = {w: flat_rmse(w) for w in W}
order = sorted(W, key=lambda w: errs[w])
picks = [order[3], order[len(order)//2], order[-30], order[-8]]; tags=["easy","mid","hard","hard2"]
fig, axes = plt.subplots(len(picks), 2, figsize=(14, 3.2*len(picks)))
for r, w in enumerate(picks):
    e = EVD[w]; S = dp.score_of(e["EV"]); nb = S.shape[0]
    dpath = dp.dp_trace(S)
    dtrue = e["dtrue"]  # 真δ per bin
    for cc, (Simg, ttl) in enumerate([(S, "emission(score)"), (S - S.min(1,keepdims=True), "列正規化")]):
        ax = axes[r, cc]
        ax.imshow(Simg.T, aspect="auto", origin="lower", extent=[0,nb,OFF[0],OFF[-1]], cmap="magma")
        ax.plot(np.arange(nb), dtrue, "c-", lw=1.0, label="真δ")
        ax.plot(np.arange(nb), dpath, "w--", lw=0.8, label="DP経路")
        ax.axhline(0, color="lime", lw=0.4, ls=":")
        if r==0: ax.set_title(ttl);
        if cc==0: ax.set_ylabel(f"{tags[r]} flat={errs[w]:.1f}\nδ(ft)", fontsize=8)
        if r==0 and cc==0: ax.legend(fontsize=7, loc="upper right")
plt.suptitle("v89 PFフリー emission: 真δ(シアン)は最明尾根か? DP(白破線)はどこへ行くか", fontsize=11)
plt.tight_layout()
op = os.path.join(ART, "emission_diag.png"); plt.savefig(op, dpi=85)
print(f"saved {op}")
for w,tg in zip(picks,tags): print(f"  {tg}: {w} flat={errs[w]:.1f}")
