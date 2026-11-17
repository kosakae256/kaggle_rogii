# -*- coding: utf-8 -*-
"""r0_seed32: forward(lag0)/lag16/lag192 と アンカー保持(lag192をforward±bandにクランプ)を比較。真TVT overall RMSE。
lag192=ユーザー実績(CV-0.3/LB-0.3で綺麗に転移)。anchor-clamp=遠端datumコミットを禁じ局所平滑だけ残す。"""
import io,sys,os,glob
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"v93")
import pf_banks_v93 as pf
TR=pf.DATA_DIR/"train"
NW=int(os.environ.get("NW","60")); NSEED=int(os.environ.get("NSEED","16")); CHUNK=int(os.environ.get("CHUNK","40"))
P0=pf.bank_param("r0_seed32")
wells=sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR/"*__horizontal_well.csv"))})[:NW]

def build(lag):
    P=dict(P0); P["smooth_lag"]=lag; inps=[]; trues=[]
    for w in wells:
        hw=pd.read_csv(TR/f"{w}__horizontal_well.csv"); tw=pd.read_csv(TR/f"{w}__typewell.csv").sort_values("TVT")
        tt=tw["TVT"].to_numpy(float); tg=tw["GR"].to_numpy(float); m=np.isfinite(tt)&np.isfinite(tg)
        if m.sum()<8: continue
        x=pf.build_smoother_inputs(hw,tt[m],tg[m],P)
        if x is None: continue
        pf.attach_anchor(x,w,P["_physics"])
        inps.append(x); trues.append(hw[hw["TVT_input"].isna()]["TVT"].to_numpy(float))
    return P,inps,trues
def run(lag):
    P,inps,trues=build(lag)
    res=pf.run_smoother_ext(inps,P,seed=99999,n_seeds=NSEED,chunk=CHUNK,w_nn=0.0)
    return [r["mean"] for r in res], trues
def rmse(preds,trues):
    e=np.concatenate([np.asarray(preds[i])-trues[i] for i in range(len(preds))]); e=e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e**2)))

print(f"[r0_seed32 {len(wells)}井 n_seed{NSEED} chunk{CHUNK}]",flush=True)
fwd,trues=run(0);   print(f"  forward(lag0) = {rmse(fwd,trues):.4f}",flush=True)
n16,_=run(16);      print(f"  lag16         = {rmse(n16,trues):.4f}",flush=True)
big,_=run(192);     print(f"  lag192        = {rmse(big,trues):.4f}  (=ユーザー実績: CV-0.3/LB-0.3 転移)",flush=True)
print("  --- アンカー保持(lag192 を forward の ±band にクランプ) ---",flush=True)
for band in (5.,10.,20.,40.):
    anc=[np.where(np.abs(np.asarray(big[i])-np.asarray(fwd[i]))>band, np.asarray(fwd[i]), np.asarray(big[i])) for i in range(len(big))]
    print(f"  anchor lag192 band=±{band:.0f}ft = {rmse(anc,trues):.4f}",flush=True)
print("\n判定: lag192がforwardより良い=後ろ読み信号。anchorがその利得を保てば「datum固定・局所平滑」成立(=転移安全な形)。",flush=True)
