# -*- coding: utf-8 -*-
import os,sys,io,numpy as np,pandas as pd,torch
sys.stdout=io.open(1,"w",encoding="utf-8",closefd=False)
sys.argv=["x"]; sys.path.insert(0,"v89")
import submit_unet_v89 as S
TR="rogii-wellbore-geology-prediction/train"
wells=sorted(w.split("__")[0] for w in os.listdir(TR) if w.endswith("__horizontal_well.csv"))
ENC={f:S.load_enc("capaug",f) for f in range(5)}
MOD={st:{f:S.load_model(st,f) for f in range(5)} for st in ["ddoof","boot","boot2"]}
res={"pmean":[],"argmax":[]}
for wi,wn in enumerate(wells):
    fold=wi%5
    try:
        hw=pd.read_csv(f"{TR}/{wn}__horizontal_well.csv"); tw=pd.read_csv(f"{TR}/{wn}__typewell.csv").sort_values("TVT")
    except: continue
    center=None; Pl=stvl=bidl=evidx=None; ok=True
    for i,st in enumerate(["ddoof","boot","boot2"]):
        r=S.build_ev(hw,tw,center,ENC[fold][0],ENC[fold][1])
        if r is None: ok=False; break
        EV,stv,bid,evidx,T=r
        with torch.no_grad():
            lg,rf=MOD[st][fold](torch.from_numpy(EV)[None].to(S.DEV)); P=torch.softmax(lg[0,:EV.shape[1]],-1)
        center=stv+np.asarray(((P@S.offs_t)+rf[0,:EV.shape[1]]).cpu().numpy(),np.float64)[bid]; Pl,stvl,bidl=P,stv,bid
    if not ok: continue
    y=hw["TVT"].to_numpy(float)[evidx]
    z_am=stvl+np.asarray(S.offs_t[Pl.argmax(-1)].cpu().numpy(),np.float64)[bidl]
    for mode,z in [("pmean",center),("argmax",z_am)]:
        e=z-y; res[mode].append(e[np.isfinite(e)])
    if (wi+1)%150==0: print(f"  {wi+1}/{len(wells)}",flush=True)
for mode in ["pmean","argmax"]:
    e=np.concatenate(res[mode]); print(f"[RESULT] MODE={mode}: 全{len(wells)}井 overall RMSE = {np.sqrt(np.mean(e**2)):.4f}")
