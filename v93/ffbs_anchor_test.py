# -*- coding: utf-8 -*-
"""r0_seed32 で forward / normal(lag16) / FFBS(full) / アンカー保持FFBS を比較。
アンカー保持FFBS = FFBS を forward推定の ±band ft にクランプ(datum flip=遠端コミット を禁止, 局所平滑は残す)。
真TVT との overall RMSE で比較。GRフリー錨は使わない(r0=use_anchor False)ので純粋に smoother の効果。"""
import io,sys,os,glob
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0,"v93")
import pf_banks_v93 as pf
TR=pf.DATA_DIR/"train"
NW=int(os.environ.get("NW","60")); NSEED=int(os.environ.get("NSEED","16")); CHUNK=int(os.environ.get("CHUNK","30"))
P0=pf.bank_param("r0_seed32")
wells=sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR/"*__horizontal_well.csv"))})[:NW]

def build(lag):
    P=dict(P0); P["smooth_lag"]=lag
    inps=[]; trues=[]
    for w in wells:
        hw=pd.read_csv(TR/f"{w}__horizontal_well.csv"); tw=pd.read_csv(TR/f"{w}__typewell.csv").sort_values("TVT")
        tt=tw["TVT"].to_numpy(float); tg=tw["GR"].to_numpy(float); m=np.isfinite(tt)&np.isfinite(tg)
        if m.sum()<8: continue
        x=pf.build_smoother_inputs(hw,tt[m],tg[m],P)
        if x is None: continue
        pf.attach_anchor(x,w,P["_physics"])  # r0=physics False → ancs=1e9(錨無効)
        ev=hw[hw["TVT_input"].isna()]
        inps.append(x); trues.append(ev["TVT"].to_numpy(float))
    return P,inps,trues

def run(lag):
    P,inps,trues=build(lag)
    res=pf.run_smoother_ext(inps,P,seed=99999,n_seeds=NSEED,chunk=CHUNK,w_nn=0.0)
    return [r["mean"] for r in res], trues

def rmse(preds,trues):
    e=np.concatenate([np.asarray(preds[i])-trues[i] for i in range(len(preds))]); e=e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e**2)))

print(f"[r0_seed32 {len(wells)}井 n_seed{NSEED}]",flush=True)
fwd,trues=run(0);         print(f"  ①forward(lag0)      = {rmse(fwd,trues):.4f}",flush=True)
n16,_=run(16);            print(f"  ②normal(lag16)      = {rmse(n16,trues):.4f}",flush=True)
lag192,_=run(192);       print(f"  ③lag192            = {rmse(lag192,trues):.4f}",flush=True)
ffbs,_=run(100000);       print(f"  ④FFBS(full lag)     = {rmse(ffbs,trues):.4f}",flush=True)
# ④アンカー保持FFBS: FFBS を forward の ±band にクランプ
for band in (5.,10.,20.,40.):
    anc=[np.where(np.abs(np.asarray(ffbs[i])-np.asarray(fwd[i]))>band, np.asarray(fwd[i]), np.asarray(ffbs[i])) for i in range(len(ffbs))]
    print(f"  ④anchor-FFBS band=±{band:.0f}ft = {rmse(anc,trues):.4f}",flush=True)
print("\n狙い: ③FFBS が①forwardより良ければ後ろ読みに信号。④が③に近く①より良ければ「datum固定で局所平滑」が効く形。",flush=True)
