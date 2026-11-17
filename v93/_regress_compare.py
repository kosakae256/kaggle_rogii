# -*- coding: utf-8 -*-
"""回帰比較: v52(6井) vs v93(6井) の列ごと最大絶対差。PF非依存base特徴が一致するか。"""
import os, sys, io
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np, pandas as pd
SCR=r"C:\Users\KOSAKA~1\AppData\Local\Temp\claude\c--Users-kosaka256-Documents-rogii-claude\1c67796e-d953-4c10-8874-a500ac704725\scratchpad"
a=pd.read_parquet(os.path.join(SCR,"df_v52_6.parquet"))
b=pd.read_parquet(os.path.join(SCR,"df_v93_6.parquet"))
print(f"v52: {a.shape}  v93: {b.shape}")
# 行整合(well,id で並べ替え)
key=["well","id"] if "id" in a.columns and "id" in b.columns else ["well"]
a=a.sort_values(key).reset_index(drop=True); b=b.sort_values(key).reset_index(drop=True)
assert len(a)==len(b), f"行数不一致 {len(a)} vs {len(b)}"
common=[c for c in a.columns if c in b.columns]
print(f"共通列: {len(common)} (v52のみ{len(set(a.columns)-set(b.columns))} / v93のみ{len(set(b.columns)-set(a.columns))})")
# PF依存列(struct->GRフリーで差が出る想定): pf_ で始まる or NN-emission経由の全PF
def is_pf(c): return c.startswith("pf_") or c.startswith("tdpf") or c=="z_grad"
rows=[]
for c in common:
    if a[c].dtype.kind not in "fiu": continue
    va=a[c].to_numpy(float); vb=b[c].to_numpy(float)
    m=np.isfinite(va)&np.isfinite(vb)
    if m.sum()==0:
        d=0.0
    else:
        d=float(np.max(np.abs(va[m]-vb[m])))
    rows.append((c,d,is_pf(c)))
rows.sort(key=lambda r:-r[1])
base_bad=[(c,d) for c,d,pf in rows if not pf and d>1e-3]
pf_diff =[(c,d) for c,d,pf in rows if pf and d>1e-3]
base_ok =sum(1 for c,d,pf in rows if not pf and d<=1e-3)
print(f"\n=== PF非依存(base)特徴: 一致{base_ok}列 / 乖離{len(base_bad)}列(>1e-3) ===")
if base_bad:
    print("  ★base乖離(要調査):")
    for c,d in base_bad[:30]: print(f"    {c}: max|Δ|={d:.4g}")
else:
    print("  ✓ base特徴すべて一致(<1e-3) = build_well非PF部は忠実再現")
print(f"\n=== PF依存列: 差あり{len(pf_diff)}列(struct->GRフリーで想定内) ===")
for c,d in pf_diff[:8]: print(f"    {c}: max|Δ|={d:.4g}")
print(f"  ... 計{len(pf_diff)}列がPF経由で変化(pfA錨+全バンクNN-emission帯中心の置換)")
