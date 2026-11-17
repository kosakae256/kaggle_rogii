# -*- coding: utf-8 -*-
"""v47 アンサンブル確認: 候補トークン+2軸注意(soft) と CRF(soft周辺) を同一foldで学習し、
OOF(行レベル)を出して単体/平均のoverall RMSEを比較する。v47 artifact のみ参照。
実行: V47_EPOCHS=15 rogii_claude/.venv/Scripts/python.exe v47/ensemble_check_v47.py
"""
import os, json, time, math, gc
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold

ART=Path(r"C:\Users\kosaka256\Documents\rogii_claude")/"v47"/"artifacts_v47"
META=json.loads((ART/"candidates_v47_meta.json").read_text(encoding="utf-8"))
STRIDE=16; T_MAX=int(os.environ.get("V47_TMAX","256")); SCALE,SG=30.0,20.0; N_SPLITS=5
EPOCHS=int(os.environ.get("V47_EPOCHS","15")); BATCH=int(os.environ.get("V47_BATCH","16")); LR=3e-4
NWD=int(os.environ.get("V47_NWELLS","0")); USE_AMP=torch.cuda.is_available()
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(42); np.random.seed(42)
try: torch.cuda.set_per_process_memory_fraction(0.85)
except Exception: pass
CAND=META["cand_names"]; NC=len(CAND)

z=np.load(ART/"candidates_v47.npz")
DEL=z["DEL"]; y=z["y"]; lk=z["lk"].astype(np.float64); mds=z["mds"].astype(np.float64); g=z["wellcode"].astype(np.int64); N=len(y)
STD=z["STD"]; CTX=z["ctx"]; GRC=z["GRC"]; GRC2=z["GRC2"]; GRM=z["GRM"]; STRD=z["STRD"]; STRS=z["STRS"]

order=np.lexsort((mds,g)); gg=g[order]; _,fst=np.unique(gg,return_index=True); fst=np.sort(fst); bnd=list(fst)+[N]
wells=[int(gg[bnd[k]]) for k in range(len(bnd)-1)]
if NWD>0: wells=wells[:NWD]; bnd=bnd[:NWD+1]
DELo=DEL[order]; STDo=STD[order]; yo=y[order]; CTXo=CTX[order]; GRCo=GRC[order]; GRC2o=GRC2[order]; GRMo=GRM[order]; STRDo=STRD[order]; STRSo=STRS[order]; pos_g=order.copy()
def bmean(a,bid,nb):
    o=np.zeros((nb,)+a.shape[1:],np.float32); c=np.zeros(nb,np.float32); np.add.at(o,bid,a); np.add.at(c,bid,1.0)
    return o/np.maximum(c.reshape((nb,)+(1,)*(a.ndim-1)),1.0), c
def build(s,e):
    n=e-s; nb=(n+STRIDE-1)//STRIDE; bid=np.arange(n)//STRIDE
    d,cnt=bmean(DELo[s:e],bid,nb); st,_=bmean(STDo[s:e],bid,nb); yb,_=bmean(yo[s:e],bid,nb)
    gc1,_=bmean(GRCo[s:e],bid,nb); gc2,_=bmean(GRC2o[s:e],bid,nb); gm,_=bmean(GRMo[s:e],bid,nb)
    strd,_=bmean(STRDo[s:e][:,None],bid,nb); strs,_=bmean(STRSo[s:e][:,None],bid,nb); cx,_=bmean(CTXo[s:e],bid,nb)
    med=np.median(d,1,keepdims=True); ddev=d-med; msign=np.sign(ddev); spread=(np.percentile(d,90,1)-np.percentile(d,10,1))[:,None]
    rank=np.argsort(np.argsort(gc1,1),1).astype(np.float32)/(NC-1); sagree=d-strd
    cur=np.abs(d-0.5*(np.pad(d,((1,0),(0,0)),mode="edge")[:-1]+np.pad(d,((0,1),(0,0)),mode="edge")[1:]))
    ctxb=np.repeat(np.stack([np.log1p(np.maximum(cx[:,0],0)),cx[:,1],np.log1p(np.maximum(cx[:,2],0)),np.log1p(np.maximum(strs[:,0],0))],1)[:,None,:],NC,1)
    perc=np.stack([d/SCALE,np.log1p(np.maximum(st,0)),gc1/SG,gc2/SG,gm/SG,ddev/SCALE,msign,rank,sagree/SCALE,cur/SCALE],2)
    ftok=np.concatenate([perc,ctxb],2).astype(np.float32)                                  # [nb,21,14]
    fflat=np.concatenate([d/SCALE,np.log1p(np.maximum(st,0)),gc1/SG,gc2/SG,gm/SG,ddev/SCALE,rank,sagree/SCALE,
                          strd/SCALE,np.log1p(np.maximum(strs,0)),spread/SCALE,
                          np.log1p(np.maximum(cx[:,0:1],0)),cx[:,1:2],np.log1p(np.maximum(cx[:,2:3],0))],1).astype(np.float32)  # [nb,174]
    return ftok,fflat,d.astype(np.float32),yb.astype(np.float32),cnt.astype(np.float32),bid
SEQ=[]; CHW=[]
for k in range(len(wells)):
    s,e=bnd[k],bnd[k+1]; ftok,fflat,d,yb,cnt,bid=build(s,e); nb=len(d); pos=pos_g[s:e]
    for c0 in range(0,nb,T_MAX):
        c1=min(c0+T_MAX,nb); rows=np.where((bid>=c0)&(bid<c1))[0]
        SEQ.append(dict(ftok=ftok[c0:c1],fflat=fflat[c0:c1],dbin=d[c0:c1],ybin=yb[c0:c1],cnt=cnt[c0:c1],bid=(bid[rows]-c0),pos=pos[rows]))
        CHW.append(wells[k])
FTOK=SEQ[0]["ftok"].shape[2]; FFLAT=SEQ[0]["fflat"].shape[1]; CHW=np.array(CHW); LEN=np.array([len(s["dbin"]) for s in SEQ])
del DELo,STDo,yo,CTXo,GRCo,GRC2o,GRMo,STRDo,STRSo,STD,CTX,GRC,GRC2,GRM,STRS; gc.collect()
print(f"chunks {len(SEQ)} ftok {FTOK} fflat {FFLAT}")

def batches(idxs): idxs=sorted(idxs,key=lambda i:LEN[i]); return [idxs[c0:c0+BATCH] for c0 in range(0,len(idxs),BATCH)]
folds=list(GroupKFold(N_SPLITS).split(np.arange(len(SEQ)),groups=CHW))

# ---------- Model A: 候補トークン + 2軸注意 (soft) ----------
class PE(nn.Module):
    def __init__(s,d,m=1024):
        super().__init__(); pe=torch.zeros(m,d); p=torch.arange(m).unsqueeze(1).float(); dv=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d)); pe[:,0::2]=torch.sin(p*dv); pe[:,1::2]=torch.cos(p*dv); s.register_buffer("pe",pe)
    def forward(s,T): return s.pe[:T]
class Axial(nn.Module):
    def __init__(s,d,h,ff,dr):
        super().__init__(); s.ca=nn.MultiheadAttention(d,h,dropout=dr,batch_first=True); s.n1=nn.LayerNorm(d); s.ta=nn.MultiheadAttention(d,h,dropout=dr,batch_first=True); s.n2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d)); s.n3=nn.LayerNorm(d)
    def forward(s,x,tp):
        B,T,C,d=x.shape; h=x.reshape(B*T,C,d); a,_=s.ca(h,h,h); x=s.n1(h+a).reshape(B,T,C,d)
        h=x.permute(0,2,1,3).reshape(B*C,T,d); a,_=s.ta(h,h,h,key_padding_mask=tp.repeat_interleave(C,0)); x=s.n2(h+a).reshape(B,C,T,d).permute(0,2,1,3)
        return s.n3(x+s.ff(x))
class TokNet(nn.Module):
    def __init__(s,fc,d=64,L=2,h=4,ff=128,dr=0.1):
        super().__init__(); s.pr=nn.Sequential(nn.Linear(fc,d),nn.GELU(),nn.LayerNorm(d)); s.emb=nn.Embedding(NC,d); s.pe=PE(d); s.bl=nn.ModuleList([Axial(d,h,ff,dr) for _ in range(L)]); s.hd=nn.Linear(d,1)
    def forward(s,feat,dbin,tp,temp=0.7):
        B,T,C,_=feat.shape; x=s.pr(feat)+s.emb(torch.arange(C,device=feat.device))[None,None]+s.pe(T).to(feat.device)[None,:,None,:]
        for b in s.bl: x=b(x,tp)
        w=torch.softmax(s.hd(x).squeeze(-1)/temp,-1); return (w*dbin).sum(-1), w

def col_tok(bt):
    T=max(len(b["dbin"]) for b in bt); B=len(bt); feat=torch.zeros(B,T,NC,FTOK); dbin=torch.zeros(B,T,NC); ybin=torch.zeros(B,T); cnt=torch.zeros(B,T); mask=torch.ones(B,T,dtype=torch.bool)
    for i,b in enumerate(bt):
        t=len(b["dbin"]); feat[i,:t]=torch.from_numpy(b["ftok"]); dbin[i,:t]=torch.from_numpy(b["dbin"]); ybin[i,:t]=torch.from_numpy(b["ybin"]); cnt[i,:t]=torch.from_numpy(b["cnt"]); mask[i,:t]=False
    return feat,dbin,ybin,cnt,mask
def train_tok():
    oof=np.full(N,np.nan,np.float32); t0=time.time()
    for k,(tr,va) in enumerate(folds):
        net=TokNet(FTOK).to(DEV); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-4); sc=torch.cuda.amp.GradScaler(enabled=USE_AMP)
        trb=batches(list(tr)); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*max(1,len(trb)))
        for ep in range(EPOCHS):
            net.train(); np.random.shuffle(trb)
            for bt in trb:
                sub=[SEQ[i] for i in bt]; feat,dbin,ybin,cnt,mask=[t.to(DEV) for t in col_tok(sub)]
                with torch.autocast("cuda",enabled=USE_AMP):
                    pred,w=net(feat,dbin,mask); v=(~mask).float()*cnt; mse=((pred-ybin)**2*v).sum()/v.sum().clamp_min(1.)
                    logit=torch.log(w.clamp_min(1e-6)); ob=(dbin-ybin.unsqueeze(-1)).abs().argmin(-1); ce=(F.cross_entropy(logit.reshape(-1,NC),ob.reshape(-1),reduction="none").reshape(pred.shape)*v).sum()/v.sum().clamp_min(1.); loss=mse+0.3*ce
                opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),1.); sc.step(opt); sc.update(); sch.step()
        net.eval()
        with torch.no_grad():
            for bt in batches(list(va)):
                sub=[SEQ[i] for i in bt]; feat,dbin,ybin,cnt,mask=[t.to(DEV) for t in col_tok(sub)]
                with torch.autocast("cuda",enabled=USE_AMP): _,w=net(feat,dbin,mask)
                w=w.float().cpu().numpy()
                for bi,b in enumerate(sub): oof[b["pos"]]=(w[bi][b["bid"]]*DEL[b["pos"]]).sum(1).astype(np.float32)
        print(f"  [tok] fold{k} done ({time.time()-t0:.0f}s)",flush=True); del net,opt,sc; gc.collect(); torch.cuda.empty_cache() if DEV=="cuda" else None
    return oof

# ---------- Model B: Encoder + CRF (soft周辺) ----------
class Enc(nn.Module):
    def __init__(s,fc,d=128,L=3,h=8,ff=256,dr=0.1):
        super().__init__(); s.pr=nn.Sequential(nn.Linear(fc,d),nn.GELU(),nn.LayerNorm(d)); s.pe=PE(d,2048); e=nn.TransformerEncoderLayer(d,h,ff,dr,batch_first=True,norm_first=True); s.en=nn.TransformerEncoder(e,L); s.em=nn.Linear(d,NC)
    def forward(s,feat,pad): x=s.pr(feat)+s.pe(feat.size(1)).to(feat.device)[None]; return s.em(s.en(x,src_key_padding_mask=pad))
class CRF(nn.Module):
    def __init__(s):
        super().__init__(); s.tr=nn.Parameter(torch.zeros(NC,NC)); s.g=nn.Parameter(torch.tensor(0.)); s.st=nn.Parameter(torch.zeros(NC)); s.en=nn.Parameter(torch.zeros(NC))
    def _t(s,dbin,t): return s.tr[None]-F.softplus(s.g)*(dbin[:,t][:,None,:]-dbin[:,t-1][:,:,None]).abs()/30.
    def nll(s,em,dbin,mask,tags):
        B,T,C=em.shape; a=s.st[None]+em[:,0]; sc=s.st[tags[:,0]]+em[torch.arange(B),0,tags[:,0]]
        for t in range(1,T):
            tr=s._t(dbin,t); a=torch.where(mask[:,t][:,None].bool(),em[:,t]+torch.logsumexp(a[:,:,None]+tr,1),a)
            step=em[torch.arange(B),t,tags[:,t]]+tr[torch.arange(B),tags[:,t-1],tags[:,t]]; sc=sc+torch.where(mask[:,t].bool(),step,torch.zeros_like(step))
        return (torch.logsumexp(a+s.en[None],1)-(sc+s.en[tags[torch.arange(B),(mask.sum(1)-1).long()]])).mean()
    def marg(s,em,dbin,mask):
        B,T,C=em.shape; al=[s.st[None]+em[:,0]]
        for t in range(1,T): tr=s._t(dbin,t); al.append(torch.where(mask[:,t][:,None].bool(),em[:,t]+torch.logsumexp(al[-1][:,:,None]+tr,1),al[-1]))
        be=[None]*T; be[T-1]=s.en[None].expand(B,C).clone()
        for t in range(T-2,-1,-1): tr=s._t(dbin,t+1); be[t]=torch.where(mask[:,t+1][:,None].bool(),torch.logsumexp(tr+em[:,t+1][:,None,:]+be[t+1][:,None,:],2),be[t+1])
        return torch.softmax(torch.stack(al,1)+torch.stack(be,1),2)
def col_flat(bt):
    T=max(len(b["dbin"]) for b in bt); B=len(bt); feat=torch.zeros(B,T,FFLAT); dbin=torch.zeros(B,T,NC); ybin=torch.zeros(B,T); mask=torch.zeros(B,T)
    for i,b in enumerate(bt):
        t=len(b["dbin"]); feat[i,:t]=torch.from_numpy(b["fflat"]); dbin[i,:t]=torch.from_numpy(b["dbin"]); ybin[i,:t]=torch.from_numpy(b["ybin"]); mask[i,:t]=1.
    return feat,dbin,ybin,mask
def train_crf():
    oof=np.full(N,np.nan,np.float32); t0=time.time()
    for k,(tr,va) in enumerate(folds):
        enc=Enc(FFLAT).to(DEV); crf=CRF().to(DEV); opt=torch.optim.AdamW(list(enc.parameters())+list(crf.parameters()),lr=LR,weight_decay=1e-4); sc=torch.cuda.amp.GradScaler(enabled=USE_AMP)
        trb=batches(list(tr)); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*max(1,len(trb)))
        for ep in range(EPOCHS):
            enc.train(); crf.train(); np.random.shuffle(trb)
            for bt in trb:
                sub=[SEQ[i] for i in bt]; feat,dbin,ybin,mask=[t.to(DEV) for t in col_flat(sub)]; gold=(dbin-ybin.unsqueeze(-1)).abs().argmin(-1)
                with torch.autocast("cuda",enabled=USE_AMP): em=enc(feat,mask==0)
                em=em.float(); p=crf.marg(em,dbin,mask); soft=(p*dbin).sum(-1); mse=(((soft-ybin)**2)*mask).sum()/mask.sum().clamp_min(1.); loss=crf.nll(em,dbin,mask,gold)+0.3*mse
                opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(list(enc.parameters())+list(crf.parameters()),1.); sc.step(opt); sc.update(); sch.step()
        enc.eval(); crf.eval()
        with torch.no_grad():
            for bt in batches(list(va)):
                sub=[SEQ[i] for i in bt]; feat,dbin,ybin,mask=[t.to(DEV) for t in col_flat(sub)]
                with torch.autocast("cuda",enabled=USE_AMP): em=enc(feat,mask==0)
                p=crf.marg(em.float(),dbin,mask); soft=(p*dbin).sum(-1).cpu().numpy()
                for bi,b in enumerate(sub): oof[b["pos"]]=soft[bi][b["bid"]].astype(np.float32)
        print(f"  [crf] fold{k} done ({time.time()-t0:.0f}s)",flush=True); del enc,crf,opt,sc; gc.collect(); torch.cuda.empty_cache() if DEV=="cuda" else None
    return oof

print("=== train Model A (token 2軸 soft) ==="); oofA=train_tok(); np.save(ART/"oof_A_token.npy",oofA)
print("=== train Model B (CRF soft) ===");       oofB=train_crf(); np.save(ART/"oof_B_crfsoft.npy",oofB)
m=np.isfinite(oofA)&np.isfinite(oofB)
def rm(a): e=(a-y)[m]; return float(np.sqrt(np.mean(e*e)))
print("\n=== v47 アンサンブル確認(行レベル overall RMSE) ===")
print(f"  A: token2軸 soft   = {rm(oofA):.4f}")
print(f"  B: CRF soft        = {rm(oofB):.4f}")
print(f"  相関(A,B delta)     = {np.corrcoef(oofA[m],oofB[m])[0,1]:.4f}")
for wa in [0.3,0.4,0.5,0.6,0.7]:
    print(f"  平均 {wa:.1f}A+{1-wa:.1f}B      = {rm(wa*oofA+(1-wa)*oofB):.4f}")
best=min([(rm(wa*oofA+(1-wa)*oofB),wa) for wa in np.linspace(0,1,21)])
print(f"  best blend rmse={best[0]:.4f} @ wA={best[1]:.2f}")
