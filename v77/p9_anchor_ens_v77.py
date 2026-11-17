# -*- coding: utf-8 -*-
"""
v77 P9 — K16錨変種アンサンブル(GRフリーのまま錨を強化)
使い方: python v77/p9_anchor_ens_v77.py _h250 _h1000 _k30 _seg32 ""(=基準)
各suffixの k16_anchor_train{sfx}.pkl を等重み平均し、773井で LOO RMSE + 素PF blend を評価。
"""
import io, sys, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v77" / "artifacts_v77"
LS = 10.536319667317654
sfxs = sys.argv[1:] if len(sys.argv) > 1 else ["", "_h250", "_h1000", "_k30", "_seg32"]
ancs = [pickle.loads((ART / f"k16_anchor_train{s}.pkl").read_bytes()) for s in sfxs]
bare = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
print("members:", [s or "(base)" for s in sfxs])

ens = {}
em, ep, ea0 = [], [], []
for wid, d in bare.items():
    if any(wid not in a for a in ancs): continue
    T = min(min(len(a[wid]["tvt"]) for a in ancs), len(d["true"]), d["err"].shape[1])
    avg = np.mean([a[wid]["tvt"][:T] for a in ancs], axis=0)
    ens[wid] = dict(tvt=avg, sig=np.mean([a[wid]["sig"][:T] for a in ancs], axis=0), dd=ancs[0][wid]["dd"][:T])
    ll = d["ll"].astype(np.float64)
    ww = np.exp((ll - ll.max()) / LS); ww /= ww.sum()
    epf = (ww[:, None] * d["err"].astype(np.float64)).sum(0)[:T]
    tv = d["true"][:T]
    e = avg - tv; e0 = ancs[0][wid]["tvt"][:T] - tv
    m = np.isfinite(e) & np.isfinite(epf)
    em.append(e[m]); ep.append(epf[m]); ea0.append(e0[m])
em = np.concatenate(em); ep = np.concatenate(ep); ea0 = np.concatenate(ea0)
sm = np.sqrt(np.mean(em ** 2)); sp = np.sqrt(np.mean(ep ** 2))
print(f"{len(ens)}井  ens={sm:.4f}  (member0={np.sqrt(np.mean(ea0**2)):.4f})  素PF={sp:.4f}  ρ={np.corrcoef(em,ep)[0,1]:.3f}")
best = min((float(np.sqrt(np.mean(((1 - w) * ep + w * em) ** 2))), float(w)) for w in np.linspace(0, 1, 101))
print(f"blend: w={best[1]:.2f} → {best[0]:.4f} (素PF比 {best[0]-sp:+.4f})")
(ART / "k16_anchor_train_ens.pkl").write_bytes(pickle.dumps(ens, protocol=4))
print("[saved] k16_anchor_train_ens.pkl")
