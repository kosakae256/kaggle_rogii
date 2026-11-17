# -*- coding: utf-8 -*-
"""物理脚を『v90との相関×blend値』で再評価(standalone精度でなく)。全773井, hedge_data流用。"""
import sys,io,pickle
try: sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
except: pass
import numpy as np
from pathlib import Path
P=Path(r"C:\Users\kosaka256\Documents\rogii_claude")
D=pickle.load(open(P/"v91"/"hedge_data_v91.pkl","rb"));W=list(D)
v90=np.concatenate([D[w]["v90"] for w in W])
snap=np.concatenate([D[w]["snap"] for w in W])
grsel=np.concatenate([D[w]["grsel"] for w in W])
r=lambda e:float(np.sqrt(np.mean(e**2)))
print(f"[blend評価 全{len(W)}井 {len(v90)}行]")
print(f"  v90 base={r(v90):.4f}  snap={r(snap):.4f}(ρ{np.corrcoef(v90,snap)[0,1]:.3f})  grsel={r(grsel):.4f}(ρ{np.corrcoef(v90,grsel)[0,1]:.3f})")
def bestblend(a,b,name):
    best=(r(a),0)
    for w in np.arange(0,0.61,0.02):
        rm=r((1-w)*a+w*b)
        if rm<best[0]: best=(rm,w)
    print(f"  {name}: 最良blend RMSE={best[0]:.4f} (w={best[1]:.2f}, Δ{best[0]-r(a):+.4f})")
    return best
bestblend(v90,snap,"v90×snap")
bestblend(v90,grsel,"v90×grsel")
# 3way: v90 + w1*snap + w2*grsel
best=(r(v90),0,0)
for w1 in np.arange(0,0.4,0.04):
    for w2 in np.arange(0,0.3,0.04):
        if w1+w2>0.6: continue
        rm=r((1-w1-w2)*v90+w1*snap+w2*grsel)
        if rm<best[0]: best=(rm,w1,w2)
print(f"  v90×snap×grsel 3way: RMSE={best[0]:.4f} (w_snap={best[1]:.2f} w_grsel={best[2]:.2f}, Δ{best[0]-r(v90):+.4f})")
