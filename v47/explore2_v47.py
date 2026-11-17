# -*- coding: utf-8 -*-
"""反復2: 反復1のLightGBM枝確率を坑井内で時間平滑し、連続性(sticky)で枝精度/RMSEが上がるか。
= sticky HMM が効くかの安価な先行テスト。v47 artifact のみ参照。"""
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d
ART=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"v47"/"artifacts_v47"
z=np.load(ART/"candidates_v47.npz"); DEL=z["DEL"]; y=z["y"]; g=z["wellcode"].astype(np.int64); mds=z["mds"].astype(np.float64); N=len(y)
proba=np.load(ART/"oof_lgbm_proba.npy")               # 反復1の枝確率
K=3; KT=21//K
a=np.argsort(DEL,1); rep=np.take_along_axis(DEL,a,1).reshape(N,K,KT).mean(2)
gold=np.argmin(np.abs(rep-y[:,None]),1)
def rmse(p): e=p-y; return float(np.sqrt(np.mean(e*e)))
print(f"[基準] 反復1 soft={rmse((proba*rep).sum(1)):.4f} hard={rmse(rep[np.arange(N),proba.argmax(1)]):.4f} acc={np.mean(proba.argmax(1)==gold):.3f}  天井={np.sqrt(np.mean(np.min(np.abs(rep-y[:,None]),1)**2)):.4f}")

# 坑井内 md_since順に平滑(連続性の効果)
order=np.lexsort((mds,g)); inv=np.argsort(order)
gs=g[order]; bnd=np.flatnonzero(np.diff(gs))+1; blocks=np.split(np.arange(N),bnd)  # sorted空間の坑井ブロック
pso=proba[order]
print("\n=== 反復2: 枝確率の時間平滑(窓=行) ===")
for W in [11,31,61,121,241]:
    sm=np.empty_like(pso)
    for bl in blocks: sm[bl]=uniform_filter1d(pso[bl],W,axis=0,mode="nearest")
    sm=sm[inv]; sm/=sm.sum(1,keepdims=True)
    hard=rep[np.arange(N),sm.argmax(1)]; soft=(sm*rep).sum(1); acc=np.mean(sm.argmax(1)==gold)
    print(f"  W={W:3d}: acc={acc:.3f}  hard={rmse(hard):.4f}  soft={rmse(soft):.4f}")
# 坑井ごと単一枝(大域多数決=完全stickyの極限)
pw=proba.copy()
wsum=np.zeros((g.max()+1,K)); np.add.at(wsum,g,proba)
wmode=wsum.argmax(1)[g]
print(f"  坑井単一枝(大域多数決): acc={np.mean(wmode==gold):.3f}  hard={rmse(rep[np.arange(N),wmode]):.4f}")
