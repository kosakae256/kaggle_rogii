# -*- coding: utf-8 -*-
"""v93 module4: 特徴エンジン(v52 の v9-INLINE build_well を忠実移植)。

このモジュールは v52 `create_train_parquet_v52.py` の「特徴生成部」を、struct を一切参照せず
クリーンに再構築したもの。数値は v52 と一致させることが最優先(feature engine は v52 verbatim)。

★v93 で変わるのは pfA バンクの錨源だけ(struct → GRフリー錨)。それは module1/module2 の中で
  完結しており、この module4 は struct/v38/v41 を一切 import/参照しない。

構成:
  - CPU PF numba kernels (_interp1/_resamp/_beam_jit/_pf_ancc/_pf_z) : v52 と数値一致(verbatim)。
  - helpers (_gr_sig/_grid/_nn/_smooth/beam_search/_smooth_radius_values)。
  - run_pf_ancc / run_pf_z : マルチシード CPU PF ラッパ(v52 verbatim)。
  - 統計/特徴ヘルパ + build_well(per-well 特徴エンジン, v52 verbatim)。

★smooth-PF 注入(v52 の monkeypatch をクリーン化):
  v52 では run_pf_ancc を _patched_run_pf_ancc に monkeypatch し、前計算した GPU smooth-PF を
  スレッドローカル wid 経由で返していた。ここでは monkeypatch せず、
    - module-global `FWD_ENS` (dict[(wid, bank_name)] -> {mean,std})
    - thread-local `_CUR.wid`(`set_current_well(wid)` で設定)
  を持ち、run_pf_ancc が最初に FWD_ENS を引く。無ければ本物の CPU マルチシード PF にフォールバック
  (=このファイル単体でも動く)。run_pf_z は常に live 計算(build_well 内で本物を回す)。

  self/nbr 列の追加は build_self_nbr_columns(...) が担う(v52 の one() 相当)。
"""
import os, threading, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

# module3(空間 imputer / 近傍プール)から共有定数と imputer 構築を借りる
import imputers_v93 as imp
from imputers_v93 import FORMATIONS, PLANE_K, DENSE_K

warnings.filterwarnings("ignore"); np.seterr(all="ignore")

DATA_DIR = imp.DATA_DIR

# ==================== v9 定数(v52 line 331-345 と同一) ====================
SEED = 0
DENSE_SPW = 60
N_SPLITS = 5
PF_SUFFIX_START = 1
KEEP_FIRST_PF_ALIAS = True

BEAMS = [
    (15, 7.885250829793783, 69.70049362431021, 3, "gr"),
    (20, 49.43859691674568, 148.592415555271, 3, "hard"),
    (12, 11.84508580886079, 58.844708544034425, 2, "loose"),
    (15, 12.769364767855363, 89.75744127131894, 3, "mid"),
    (15, 10.693883733824851, 118.15908775467301, 8, "smooth"),
    (40, 67.99404057104671, 101.2117946585323, 1, "best"),
]

# 候補パスまわりのオフセット配列(v52 line 851-854 と同一)
ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)


# ==================== PF パラメータ写像(v52 line 360-401 verbatim) ====================
def _make_pf_params(raw, idx):
    """Optuna 形式の PF パラメータ dict を、この notebook が使う内部名へ写す。"""
    r = dict(raw)
    p = {}
    p["name"] = str(r.get("name", f"pf_{idx + PF_SUFFIX_START}"))
    p["suffix"] = str(r.get("suffix", f"_{idx + PF_SUFFIX_START}"))
    p["PF_N"] = int(r.get("PF_N", r.get("n_particles", 600)))
    p["ANCC_N"] = int(r.get("ANCC_N", r.get("n_particles", p["PF_N"])))
    p["PF_N_SEEDS"] = int(r.get("PF_N_SEEDS", r.get("n_seeds", 8)))
    p["PF_SEED0"] = int(r.get("PF_SEED0", SEED + idx * 100000))
    p["PF_LIKELIHOOD_SCALE"] = float(r.get("PF_LIKELIHOOD_SCALE", r.get("likelihood_scale", 20.0)))
    p["PF_INIT_SPR"] = float(r.get("PF_INIT_SPR", r.get("init_pos_std", 0.5)))
    p["PF_INIT_V_STD"] = float(r.get("PF_INIT_V_STD", r.get("init_rate_std", 0.01)))
    p["PF_MOM"] = float(r.get("PF_MOM", r.get("mom", 0.999)))
    p["PF_VN"] = float(r.get("PF_VN", r.get("vn", 0.001)))
    p["PF_PN"] = float(r.get("PF_PN", r.get("pn", 0.005)))
    p["PF_ROUGH_P"] = float(r.get("PF_ROUGH_P", r.get("rp", 0.5)))
    p["PF_ROUGH_V"] = float(r.get("PF_ROUGH_V", r.get("rr", 0.001)))
    p["PF_RESAMP"] = float(r.get("PF_RESAMP", r.get("resamp", 0.5)))
    p["PF_GR_SIG_MIN"] = float(r.get("PF_GR_SIG_MIN", r.get("gr_sig_min", 5.0)))
    p["PF_GR_SIG_MAX"] = float(r.get("PF_GR_SIG_MAX", r.get("gr_sig_max", 50.0)))
    p["PF_GR_SIG_DEF"] = float(r.get("PF_GR_SIG_DEF", r.get("gr_sig_def", 100.0)))
    p["PF_GR_SIG_MULT"] = float(r.get("PF_GR_SIG_MULT", r.get("gr_sig_mult", 2.0)))
    p["PF_TVT_CLIP_MARGIN"] = float(r.get("PF_TVT_CLIP_MARGIN", r.get("tvt_clip_margin", 50.0)))
    p["PF_RATE_CLIP"] = float(r.get("PF_RATE_CLIP", r.get("rate_clip", 0.0)))
    p["PF_GR_POWER"] = float(r.get("PF_GR_POWER", r.get("gr_power", 2.0)))
    p["PF_HGR_SMOOTH_R"] = int(r.get("PF_HGR_SMOOTH_R", r.get("hgr_smooth_r", 0)))
    p["PF_TW_GR_SMOOTH_R"] = int(r.get("PF_TW_GR_SMOOTH_R", r.get("tw_gr_smooth_r", 0)))
    p["PF_JUMP_PROB"] = float(r.get("PF_JUMP_PROB", r.get("jump_prob", 0.0)))
    p["PF_JUMP_STD"] = float(r.get("PF_JUMP_STD", r.get("jump_std", 0.0)))
    p["PF_GR_WIN"] = max(1, 2 * int(p["PF_HGR_SMOOTH_R"]) + 1)
    p["PF_GR_WT"] = float(r.get("PF_GR_WT", r.get("gr_wt", 0.3)))

    # ANCC-PF は TVT+Z を追跡。デフォルトは調整済み PF の遷移/初期化/リサンプルを流用。
    p["ANCC_ALPHA"] = float(r.get("ANCC_ALPHA", p["PF_MOM"]))
    p["ANCC_RN"] = float(r.get("ANCC_RN", p["PF_VN"]))
    p["ANCC_PN"] = float(r.get("ANCC_PN", p["PF_PN"]))
    p["ANCC_IR"] = float(r.get("ANCC_IR", p["PF_INIT_V_STD"]))
    p["ANCC_IS"] = float(r.get("ANCC_IS", p["PF_INIT_SPR"]))
    p["ANCC_RP"] = float(r.get("ANCC_RP", p["PF_ROUGH_P"]))
    p["ANCC_RR"] = float(r.get("ANCC_RR", p["PF_ROUGH_V"]))
    return p


# ---- PF_PARAM_SETS を module1 の BANK_ORDER から構築 ----
# ★重要: name をバンク名(pf_1/pf_2/pf_3/r0_seed32/r1_seed32/pfA)にする。
#   これで FWD_ENS/SELF_ENS/NBR_ENS のキー (wid, bank) と build_well 内 p["name"] が一致する。
#   suffix は _1.._6(=特徴列名は v52 と同一)。
import pf_banks_v93 as pf

PF_BANK = list(pf.BANK_ORDER)
PF_PARAM_SETS_RAW = [{**pf.BANK_PARAMS[_b], "name": _b} for _b in PF_BANK]
PF_PARAM_SETS = [_make_pf_params(raw, i) for i, raw in enumerate(PF_PARAM_SETS_RAW)]
if len(PF_PARAM_SETS) == 0:
    raise ValueError("PF_PARAM_SETS_RAW must contain at least one parameter set.")


# ==================== smooth-PF 注入(monkeypatch レス) ====================
FWD_ENS = {}          # (wid, bank_name) -> dict(mean, std)  ← create が populate
_CUR = threading.local()


def set_current_well(wid):
    """build_well 呼び出し前に(スレッドごとに)現在の坑井 wid を設定。"""
    _CUR.wid = wid


# ==================== CPU PF numba kernels(v52 line 410-594 verbatim) ====================
@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search ±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # ±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=False)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IR,IS,RP,RR,RESAMP,
              RATE_CLIP,GR_POWER,JUMP_PROB,JUMP_STD,CLIP_MARGIN,SEED_VALUE):
    if SEED_VALUE >= 0:
        np.random.seed(SEED_VALUE)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+IR*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    log_lik=0.
    tvt_lo=vmin-CLIP_MARGIN
    tvt_hi=vmin+(len(gg)-1)*step+CLIP_MARGIN
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            if RATE_CLIP>0.:
                if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            # Rare jump particles for extreme TVT movement. jump_prob=0 disables this.
            if JUMP_PROB>0. and JUMP_STD>0. and np.random.random()<JUMP_PROB:
                pos[j]+=JUMP_STD*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,tvt_lo); tvt_j=min(tvt_j,tvt_hi)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.; avg_lk=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=abs((gr_v[i]-eg)/gs)
                dp=d**GR_POWER
                lk=max(np.exp(-0.5*dp) if dp<600. else 0.,1e-300)
                old_w=w[j]
                avg_lk+=old_w*lk
                w[j]=old_w*lk; ws+=w[j]
            log_lik+=np.log(max(avg_lk,1e-300))
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            if RATE_CLIP>0.:
                for j in range(N):
                    if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                    elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_,log_lik

@njit(cache=False)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP,IP_STD,IV_STD,
          RATE_CLIP,GR_POWER,JUMP_PROB,JUMP_STD,CLIP_MARGIN,SEED_VALUE):
    if SEED_VALUE >= 0:
        np.random.seed(SEED_VALUE)
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+IP_STD*np.random.randn()
        vel[j]=iv+IV_STD*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            if RATE_CLIP>0.:
                if vel[j] > RATE_CLIP: vel[j] = RATE_CLIP
                elif vel[j] < -RATE_CLIP: vel[j] = -RATE_CLIP
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            if JUMP_PROB>0. and JUMP_STD>0. and np.random.random()<JUMP_PROB:
                pos[j]+=JUMP_STD*np.random.randn()
            pos[j]=max(pos[j],vmin-CLIP_MARGIN); pos[j]=min(pos[j],vmin+(len(gg_p)-1)*step+CLIP_MARGIN)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=abs((gr_v[i]-ep)/gs)
                dpow=dp**GR_POWER
                lp=max(np.exp(-0.5*dpow) if dpow<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=abs((gr_sm_v[i]-es)/(gs*1.5))
                    dsp=ds**GR_POWER
                    ls=max(np.exp(-0.5*dsp) if dsp<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_


# ==================== helpers(v52 line 597-639 verbatim) ====================
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(float(fb))
    if r <= 0:
        return s.to_numpy(np.float32)
    return s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr,p):
    # Match pf_tuning_4_early_resume: estimate sigma using the same smoothed
    # horizontal GR and typewell GR that PF likelihood will use.
    gr_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    hgr=_smooth_radius_values(hw['GR'].astype(float).to_numpy(), gr_fb, p["PF_HGR_SMOOTH_R"])
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20:
        gs=float(p["PF_GR_SIG_DEF"])
    else:
        kpos=hw.index.get_indexer(kn.index)
        resid=hgr[kpos]-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)
        gs=float(np.nanstd(resid))
        if not np.isfinite(gs) or gs<=0:
            gs=float(p["PF_GR_SIG_DEF"])
    return float(np.clip(gs*p["PF_GR_SIG_MULT"],p["PF_GR_SIG_MIN"],p["PF_GR_SIG_MAX"]))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)


# ==================== マルチシード CPU PF ラッパ ====================
def run_pf_ancc(hw, tw_tvt, tw_gr, p):
    """★smooth-PF 注入つき run_pf_ancc。
       まず FWD_ENS[(現在wid, p['name'])] を引き、あればその smooth-PF 平均/std を返す。
       無ければ本物の CPU マルチシード PF(v52 line 641-677 verbatim)を計算(=単体実行可)。"""
    r = FWD_ENS.get((getattr(_CUR, "wid", None), p.get("name")))
    if r is not None:
        return r["mean"].astype(np.float32), r["std"].astype(np.float32)
    # ---- fallback: 本物の CPU マルチシード ANCC-PF(v52 verbatim) ----
    tw_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf=_smooth_radius_values(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs=_gr_sig(hw,tw_tvt,tw_gr_pf,p)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr_pf)
    md_ev=ev['MD'].values.astype(np.float64)
    z_ev=ev['Z'].values.astype(np.float64)
    # Tuning notebook interpolates + optionally smooths horizontal GR before likelihood.
    gr_fb=tw_fb
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(gr_fb).to_numpy(dtype=float)
    gr_proc=_smooth_radius_values(gr_full, gr_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos=hw.index.get_indexer(ev.index)
    gr_ev=gr_proc[ev_pos].astype(np.float64)

    preds=[]; stds=[]; liks=[]
    for s in range(int(p["PF_N_SEEDS"])):
        pts,std,ll=_pf_ancc(md_ev,z_ev,gr_ev,gg,gmin,gst,
                            gs,ls,ir,int(p["ANCC_N"]),
                            p["ANCC_ALPHA"],p["ANCC_RN"],p["ANCC_PN"],p["ANCC_IR"],p["ANCC_IS"],p["ANCC_RP"],p["ANCC_RR"],
                            p["PF_RESAMP"],p["PF_RATE_CLIP"],p["PF_GR_POWER"],p["PF_JUMP_PROB"],p["PF_JUMP_STD"],
                            p["PF_TVT_CLIP_MARGIN"],int(p["PF_SEED0"])+s)
        preds.append(pts); stds.append(std); liks.append(ll)
    pred_arr=np.stack(preds,0)
    std_arr=np.stack(stds,0)
    lik_arr=np.asarray(liks,dtype=np.float64)
    lik_arr=lik_arr-np.nanmax(lik_arr)
    ww=np.exp(lik_arr/max(float(p["PF_LIKELIHOOD_SCALE"]),1e-6))
    ww=ww/max(float(ww.sum()),1e-300)
    ens=(ww[:,None]*pred_arr).sum(0)
    ens_var=(ww[:,None]*(std_arr**2+(pred_arr-ens[None,:])**2)).sum(0)
    return ens.astype(np.float32),np.sqrt(np.maximum(ens_var,0.)).astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,p):
    """Z追跡 PF(常に live 計算, v52 line 679-710 verbatim)。"""
    tw_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf=_smooth_radius_values(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs=_gr_sig(hw,tw_tvt,tw_gr_pf,p)
    tw_s=pd.Series(tw_gr_pf).rolling(p["PF_GR_WIN"],center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr_pf)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_fb=tw_fb
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(gr_fb).to_numpy(dtype=float)
    gr_proc=_smooth_radius_values(gr_full, gr_fb, p["PF_HGR_SMOOTH_R"])
    gr_sm=pd.Series(gr_proc).rolling(p["PF_GR_WIN"],center=True,min_periods=1).mean().to_numpy(dtype=float)
    ev_pos=hw.index.get_indexer(ev.index)
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   gr_proc[ev_pos].astype(np.float64),
                   gr_sm[ev_pos].astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,int(p["PF_N"]),
                   p["PF_MOM"],p["PF_VN"],p["PF_PN"],p["PF_GR_WT"],p["PF_ROUGH_P"],p["PF_ROUGH_V"],p["PF_RESAMP"],
                   p["PF_INIT_SPR"],p["PF_INIT_V_STD"],p["PF_RATE_CLIP"],p["PF_GR_POWER"],p["PF_JUMP_PROB"],p["PF_JUMP_STD"],
                   p["PF_TVT_CLIP_MARGIN"],int(p["PF_SEED0"]))
    return pts.astype(np.float32),std.astype(np.float32)


# ---- numba warmup(v52 line 713-722) ----
_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_p0=PF_PARAM_SETS[0]
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,
         _p0["ANCC_ALPHA"],_p0["ANCC_RN"],_p0["ANCC_PN"],_p0["ANCC_IR"],_p0["ANCC_IS"],_p0["ANCC_RP"],_p0["ANCC_RR"],_p0["PF_RESAMP"],
         _p0["PF_RATE_CLIP"],_p0["PF_GR_POWER"],_p0["PF_JUMP_PROB"],_p0["PF_JUMP_STD"],_p0["PF_TVT_CLIP_MARGIN"],123)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,
      _p0["PF_MOM"],_p0["PF_VN"],_p0["PF_PN"],_p0["PF_GR_WT"],_p0["PF_ROUGH_P"],_p0["PF_ROUGH_V"],_p0["PF_RESAMP"],_p0["PF_INIT_SPR"],_p0["PF_INIT_V_STD"],
      _p0["PF_RATE_CLIP"],_p0["PF_GR_POWER"],_p0["PF_JUMP_PROB"],_p0["PF_JUMP_STD"],_p0["PF_TVT_CLIP_MARGIN"],123)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)


# ==================== 統計 / 特徴ヘルパ(v52 line 724-943 verbatim) ====================
def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

def _rolling_mean_np(x, w):
    return pd.Series(np.asarray(x, np.float32)).rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32)

def _rolling_std_np(x, w):
    return pd.Series(np.asarray(x, np.float32)).rolling(w, center=True, min_periods=1).std().fillna(0.).to_numpy(np.float32)

def _safe_grad(y, x=None):
    y = np.asarray(y, np.float32)
    n = len(y)
    if n <= 1:
        return np.zeros(n, np.float32)
    if x is None:
        g = np.gradient(y).astype(np.float32)
    else:
        x = np.asarray(x, np.float32)
        dx = np.gradient(x).astype(np.float32)
        dx = np.where(np.abs(dx) < 1e-6, np.nan, dx)
        g = np.gradient(y).astype(np.float32) / dx
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return g.astype(np.float32)

def _add_path_shape_features(feats, prefix, path, md=None):
    """Local geometry of a candidate TVT path. No target is used."""
    path = np.asarray(path, np.float32)
    rate = _safe_grad(path, md)
    acc = _safe_grad(rate, md)
    feats[f'{prefix}_rate'] = rate.astype(np.float32)
    feats[f'{prefix}_acc'] = acc.astype(np.float32)
    feats[f'{prefix}_rate_abs'] = np.abs(rate).astype(np.float32)
    feats[f'{prefix}_acc_abs'] = np.abs(acc).astype(np.float32)
    feats[f'{prefix}_rate_rmean21'] = _rolling_mean_np(rate, 21)
    feats[f'{prefix}_rate_rstd21'] = _rolling_std_np(rate, 21)
    feats[f'{prefix}_acc_rmean21'] = _rolling_mean_np(acc, 21)

def _add_gr_match_features(feats, prefix, path, hgr, tw_tvt, tw_gr, offs):
    """How well a candidate TVT path explains the observed horizontal GR."""
    path = np.asarray(path, np.float32)
    hgr = np.asarray(hgr, np.float32)
    offs = np.asarray(offs, np.float32)
    gr_mat = np.stack([np.interp(path + float(o), tw_tvt, tw_gr) for o in offs], axis=1).astype(np.float32)
    resid = (hgr[:, None] - gr_mat).astype(np.float32)
    abs_resid = np.abs(resid)
    zero_idx = int(np.argmin(np.abs(offs)))
    best_idx = abs_resid.argmin(axis=1)
    row_idx = np.arange(len(path))
    r0 = resid[:, zero_idx].astype(np.float32)
    best_signed = resid[row_idx, best_idx].astype(np.float32)
    feats[f'{prefix}_gr_resid0'] = r0
    feats[f'{prefix}_gr_abs0'] = np.abs(r0).astype(np.float32)
    feats[f'{prefix}_gr_abs_min'] = abs_resid[row_idx, best_idx].astype(np.float32)
    feats[f'{prefix}_gr_best_off'] = offs[best_idx].astype(np.float32)
    feats[f'{prefix}_gr_best_resid'] = best_signed
    feats[f'{prefix}_gr_resid0_rmean21'] = _rolling_mean_np(r0, 21)
    feats[f'{prefix}_gr_resid0_rstd21'] = _rolling_std_np(r0, 21)
    feats[f'{prefix}_gr_abs0_rmean21'] = _rolling_mean_np(np.abs(r0), 21)

def _pairwise_abs_stats(mat):
    mat = np.asarray(mat, np.float32)
    k = mat.shape[1]
    if k < 2:
        z = np.zeros(mat.shape[0], np.float32)
        return z, z
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(np.abs(mat[:, i] - mat[:, j]).astype(np.float32))
    diffs = np.stack(vals, axis=1)
    return diffs.mean(axis=1).astype(np.float32), diffs.max(axis=1).astype(np.float32)

def _corr_summary(mat):
    mat = np.asarray(mat, np.float32)
    k = mat.shape[1]
    if k < 2:
        return 0.0, 0.0, 0.0
    vals = []
    for i in range(k):
        xi = mat[:, i]
        xi_std = float(np.std(xi))
        for j in range(i + 1, k):
            xj = mat[:, j]
            xj_std = float(np.std(xj))
            if xi_std < 1e-6 or xj_std < 1e-6:
                vals.append(0.0)
            else:
                vals.append(float(np.corrcoef(xi, xj)[0, 1]))
    vals = np.nan_to_num(np.asarray(vals, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(vals.mean()), float(vals.min()), float(vals.max())


def _add_pf_feature_columns(feats, pf_runs, last_tvt, tvt_f_ancc, tvt_dense, hgr, tw_tvt, tw_gr, sc,
                            hmd=None, beam_mean=None, sc_ens=None, hyb_ref=None):
    """Add PF-derived feature columns for all parameter sets.(v52 line 946-1051 verbatim)"""
    def add_one(r, suffix):
        pf_a = r["pf_ancc"].astype(np.float32)
        std_a = r["pf_ancc_std"].astype(np.float32)
        has_z = bool(r["has_z"])
        pf_z = r["pf_z"].astype(np.float32) if has_z else sc(last_tvt)
        feats[f'pf_ancc{suffix}'] = pf_a
        feats[f'pf_ancc_std{suffix}'] = std_a
        feats[f'pf_ancc_delta{suffix}'] = (pf_a - np.float32(last_tvt)).astype(np.float32)
        feats[f'pf_z{suffix}'] = pf_z
        feats[f'pf_z_delta{suffix}'] = (pf_z - np.float32(last_tvt)).astype(np.float32) if has_z else sc(0.)
        feats[f'pf_vs_z{suffix}'] = (pf_a - pf_z).astype(np.float32) if has_z else sc(0.)
        feats[f'pf_vs_spatial{suffix}'] = (pf_a - tvt_f_ancc).astype(np.float32)
        feats[f'pf_vs_dense{suffix}'] = (pf_a - tvt_dense).astype(np.float32)
        for o in PF_OFFS:
            feats[f'tdpf{int(o)}{suffix}'] = hgr - np.interp(pf_a + o, tw_tvt, tw_gr).astype(np.float32)

        prefix = 'pf' if suffix == '' else f'pf{suffix}'
        _add_gr_match_features(feats, prefix, pf_a, hgr, tw_tvt, tw_gr, PF_OFFS)
        if hmd is not None:
            _add_path_shape_features(feats, prefix, pf_a, hmd)

    if KEEP_FIRST_PF_ALIAS and len(pf_runs) > 0:
        add_one(pf_runs[0], "")
    for r in pf_runs:
        add_one(r, r["suffix"])

    if len(pf_runs) >= 2:
        pf_stack = np.stack([r["pf_ancc"].astype(np.float32) for r in pf_runs], axis=1)
        pf_mean = pf_stack.mean(1).astype(np.float32)
        pf_med = np.median(pf_stack, axis=1).astype(np.float32)
        pf_pair_mean, pf_pair_max = _pairwise_abs_stats(pf_stack)
        c_mean, c_min, c_max = _corr_summary(pf_stack)
        feats['pf_ancc_mean'] = pf_mean
        feats['pf_ancc_med'] = pf_med
        feats['pf_ancc_std_between'] = pf_stack.std(1).astype(np.float32)
        feats['pf_ancc_range_between'] = (pf_stack.max(1) - pf_stack.min(1)).astype(np.float32)
        feats['pf_ancc_iqr_between'] = (np.percentile(pf_stack, 75, axis=1) - np.percentile(pf_stack, 25, axis=1)).astype(np.float32)
        feats['pf_ancc_pair_absmean_between'] = pf_pair_mean
        feats['pf_ancc_pair_absmax_between'] = pf_pair_max
        feats['pf_ancc_mean_delta'] = (pf_mean - np.float32(last_tvt)).astype(np.float32)
        feats['pf_ancc_med_delta'] = (pf_med - np.float32(last_tvt)).astype(np.float32)
        feats['pf_ancc_corr_mean'] = sc(c_mean)
        feats['pf_ancc_corr_min'] = sc(c_min)
        feats['pf_ancc_corr_max'] = sc(c_max)

        for i in range(len(pf_runs)):
            si = pf_runs[i]['suffix']
            feats[f'pf_dev_mean{si}'] = (pf_stack[:, i] - pf_mean).astype(np.float32)
            feats[f'pf_dev_med{si}'] = (pf_stack[:, i] - pf_med).astype(np.float32)
        for i in range(len(pf_runs)):
            for j in range(i + 1, len(pf_runs)):
                si = pf_runs[i]['suffix'].replace('_', '')
                sj = pf_runs[j]['suffix'].replace('_', '')
                diff = (pf_stack[:, i] - pf_stack[:, j]).astype(np.float32)
                feats[f'pf_diff_{si}_{sj}'] = diff
                feats[f'pf_absdiff_{si}_{sj}'] = np.abs(diff).astype(np.float32)

        _add_gr_match_features(feats, 'pf_mean', pf_mean, hgr, tw_tvt, tw_gr, PF_OFFS)
        _add_gr_match_features(feats, 'pf_med', pf_med, hgr, tw_tvt, tw_gr, PF_OFFS)
        if hmd is not None:
            _add_path_shape_features(feats, 'pf_mean', pf_mean, hmd)
            _add_path_shape_features(feats, 'pf_med', pf_med, hmd)

        if beam_mean is not None:
            feats['pf_mean_vs_beam_mean'] = (pf_mean - np.asarray(beam_mean, np.float32)).astype(np.float32)
            feats['pf_med_vs_beam_mean'] = (pf_med - np.asarray(beam_mean, np.float32)).astype(np.float32)
        if sc_ens is not None:
            feats['pf_mean_vs_sc_ens'] = (pf_mean - np.asarray(sc_ens, np.float32)).astype(np.float32)
            feats['pf_med_vs_sc_ens'] = (pf_med - np.asarray(sc_ens, np.float32)).astype(np.float32)
        if hyb_ref is not None:
            feats['pf_mean_vs_hyb'] = (pf_mean - np.asarray(hyb_ref, np.float32)).astype(np.float32)
        feats['pf_mean_vs_spatial'] = (pf_mean - tvt_f_ancc).astype(np.float32)
        feats['pf_mean_vs_dense'] = (pf_mean - tvt_dense).astype(np.float32)

        z_list = [r["pf_z"].astype(np.float32) for r in pf_runs if bool(r["has_z"])]
        if len(z_list) >= 2:
            z_stack = np.stack(z_list, axis=1)
            z_mean = z_stack.mean(1).astype(np.float32)
            z_med = np.median(z_stack, axis=1).astype(np.float32)
            z_pair_mean, z_pair_max = _pairwise_abs_stats(z_stack)
            zc_mean, zc_min, zc_max = _corr_summary(z_stack)
            feats['pf_z_mean'] = z_mean
            feats['pf_z_med'] = z_med
            feats['pf_z_std_between'] = z_stack.std(1).astype(np.float32)
            feats['pf_z_range_between'] = (z_stack.max(1) - z_stack.min(1)).astype(np.float32)
            feats['pf_z_pair_absmean_between'] = z_pair_mean
            feats['pf_z_pair_absmax_between'] = z_pair_max
            feats['pf_z_corr_mean'] = sc(zc_mean)
            feats['pf_z_corr_min'] = sc(zc_min)
            feats['pf_z_corr_max'] = sc(zc_max)
            feats['pf_mean_vs_z_mean'] = (pf_mean - z_mean).astype(np.float32)
            feats['pf_med_vs_z_med'] = (pf_med - z_med).astype(np.float32)


# ==================== imputer(module3)を借りる ====================
_FI = None
_DI = None


def set_imputers(FI, DI):
    """create 側で構築した imputer を注入(二重構築回避)。"""
    global _FI, _DI
    _FI, _DI = FI, DI


def _ensure_imputers():
    """build_well が imputer を必要とするときに、無ければ自前で構築(単体実行可)。"""
    global _FI, _DI
    if _FI is None or _DI is None:
        FI, DI, _ = imp.build_imputers()
        _FI, _DI = FI, DI


# ==================== per-well 特徴エンジン(v52 line 1054-1329 verbatim) ====================
def build_well(hw_path, tw_path, is_train):
    _ensure_imputers()
    global _FI, _DI
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    try:
        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    pf_runs=[]
    for pconf in PF_PARAM_SETS:
        pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr,pconf)
        if len(pf_a)==0: return None
        pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr,pconf)
        has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))
        pf_runs.append({
            "name": pconf["name"],
            "suffix": pconf["suffix"],
            "pf_ancc": pf_a.astype(np.float32),
            "pf_ancc_std": std_a.astype(np.float32),
            "pf_z": pf_z.astype(np.float32) if len(pf_z)==len(pf_a) else np.full(len(pf_a), np.nan, np.float32),
            "pf_z_std": std_z.astype(np.float32) if len(std_z)==len(pf_a) else np.full(len(pf_a), np.nan, np.float32),
            "has_z": has_z,
        })
    pf_primary=pf_runs[0]
    pf_use=pf_primary["pf_ancc"].astype(np.float32)
    std_use=pf_primary["pf_ancc_std"].astype(np.float32)
    pf_z=pf_primary["pf_z"].astype(np.float32)
    has_z=bool(pf_primary["has_z"])

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT ±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    # Robust reference beam: older versions used tags 'cons'/'sm5', this notebook uses 'gr'/'smooth'.
    _beam_a=bpaths['cons'] if 'cons' in bpaths else bpaths.get('gr', next(iter(bpaths.values())))
    _beam_b=bpaths['sm5'] if 'sm5' in bpaths else bpaths.get('smooth', _beam_a)
    beam_ref=(_beam_a+_beam_b)/2.

    # Multi-scale NCC → score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)

    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE
    tvt_fs={}; form_rmse={}; form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)
        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=np.float32(b_full); tvt_fs[f'bww_{fn}']=np.float32(b_wls)
        tvt_fs[f'bw50_{fn}']=np.float32(b_late)
        tvt_fs[f'bw_early_{fn}']=np.float32(b_early)   # NEW: early segment
        tvt_fs[f'bw_mid_{fn}']=np.float32(b_mid)       # NEW: mid segment
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)

    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d =fs.std(1).astype(np.float32)
    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)
    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)
    res_kn=ktvt+z_kn-d_kn
    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))

    candidate_names = [f"pf_ancc{r['suffix']}" for r in pf_runs] + [f"beam_{k}" for k in bpaths.keys()] + [
        'sc8', 'sc15', 'sc25', 'sc_ens', 'spatial_ancc', 'dense_ancc'
    ]
    all_sigs=[r['pf_ancc'] for r in pf_runs]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense]
    sig_mat=np.stack(all_sigs,1).astype(np.float32)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    sig_med=np.median(sig_mat,axis=1).astype(np.float32)
    sig_q10=np.percentile(sig_mat,10,axis=1).astype(np.float32)
    sig_q25=np.percentile(sig_mat,25,axis=1).astype(np.float32)
    sig_q75=np.percentile(sig_mat,75,axis=1).astype(np.float32)
    sig_q90=np.percentile(sig_mat,90,axis=1).astype(np.float32)
    sig_iqr=(sig_q75-sig_q25).astype(np.float32)
    sig_range=(sig_mat.max(1)-sig_mat.min(1)).astype(np.float32)
    sig_mad=np.mean(np.abs(sig_mat-sig_med[:,None]),axis=1).astype(np.float32)
    sig_centered=(sig_mat-sig_mat.mean(1,keepdims=True)).astype(np.float32)
    sig_skew=(np.mean(sig_centered**3,axis=1)/(np.maximum(sig_std,1e-6)**3)).astype(np.float32)

    sig_gr_pred_mat=np.stack([np.interp(sig_mat[:,j],tw_tvt,tw_gr) for j in range(sig_mat.shape[1])],axis=1).astype(np.float32)
    sig_gr_resid_mat=(hgr[:,None]-sig_gr_pred_mat).astype(np.float32)
    sig_gr_abs_mat=np.abs(sig_gr_resid_mat).astype(np.float32)
    sig_gr_best_idx=sig_gr_abs_mat.argmin(axis=1)
    _sig_rows=np.arange(len(hgr))
    sig_gr_best_path=sig_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_best_abs=sig_gr_abs_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_best_resid=sig_gr_resid_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_pred_std=sig_gr_pred_mat.std(1).astype(np.float32)
    sig_gr_resid_mean=sig_gr_resid_mat.mean(1).astype(np.float32)
    sig_gr_resid_std=sig_gr_resid_mat.std(1).astype(np.float32)

    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sig_std':sig_std,'sig_mean_d':sig_mean,
        'sig_med_d':(sig_med-np.float32(last_tvt)).astype(np.float32),
        'sig_q10_d':(sig_q10-np.float32(last_tvt)).astype(np.float32),
        'sig_q25_d':(sig_q25-np.float32(last_tvt)).astype(np.float32),
        'sig_q75_d':(sig_q75-np.float32(last_tvt)).astype(np.float32),
        'sig_q90_d':(sig_q90-np.float32(last_tvt)).astype(np.float32),
        'sig_iqr':sig_iqr,'sig_range':sig_range,'sig_mad':sig_mad,'sig_skew':sig_skew,
        'sig_gr_pred_std':sig_gr_pred_std,
        'sig_gr_resid_mean':sig_gr_resid_mean,'sig_gr_resid_std':sig_gr_resid_std,
        'sig_gr_best_d':(sig_gr_best_path-np.float32(last_tvt)).astype(np.float32),
        'sig_gr_best_idx':sig_gr_best_idx.astype(np.float32),
        'sig_gr_best_abs':sig_gr_best_abs,'sig_gr_best_resid':sig_gr_best_resid,
        'sig_gr_best_vs_med':(sig_gr_best_path-sig_med).astype(np.float32),
        **tvt_fs,
        **{f'frm_rmse_{fn}':sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d':form_mean_d,'form_std_d':form_std_d,'form_rng_d':form_rng_d,
        'spatial_ancc_d':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),
        'spatial_knn_dist':knn_d,
        'dense_ancc':d_ancc,'dense_std':d_std,'dense_dist':d_dist,
        'tvt_dense_d' :(tvt_dense -last_tvt).astype(np.float32),
        'tvt_densew_d':(tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d':(tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse':sc(d_rmse),'dense_bias':sc(d_bias),'dense_nb_std':sc(d_nb_std),
        'spatial_vs_dense':(tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial':(beam_ref-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam':(sc_ens-beam_ref).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 -last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    _add_pf_feature_columns(
        feats, pf_runs, last_tvt, tvt_fs['tvtF_ANCC'], tvt_dense, hgr, tw_tvt, tw_gr, sc,
        hmd=hmd,
        beam_mean=(np.float32(last_tvt) + feats['beam_mean_d']).astype(np.float32),
        sc_ens=sc_ens,
        hyb_ref=hyb_ref,
    )

    # Shape/GR-match features for non-PF ensemble references.
    beam_mean_path=(np.float32(last_tvt)+feats['beam_mean_d']).astype(np.float32)
    beam_med_path=(np.float32(last_tvt)+feats['beam_med_d']).astype(np.float32)
    _add_path_shape_features(feats,'beam_mean',beam_mean_path,hmd)
    _add_path_shape_features(feats,'beam_med',beam_med_path,hmd)
    _add_path_shape_features(feats,'sc_ens',sc_ens,hmd)
    _add_path_shape_features(feats,'hyb',hyb_ref,hmd)
    _add_path_shape_features(feats,'sig_med',sig_med,hmd)
    _add_path_shape_features(feats,'sig_gr_best',sig_gr_best_path,hmd)
    _add_gr_match_features(feats,'beam_mean',beam_mean_path,hgr,tw_tvt,tw_gr,BEAM_OFFS)
    _add_gr_match_features(feats,'beam_med',beam_med_path,hgr,tw_tvt,tw_gr,BEAM_OFFS)
    _add_gr_match_features(feats,'sc_ens',sc_ens,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'hyb',hyb_ref,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'sig_med',sig_med,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'sig_gr_best',sig_gr_best_path,hgr,tw_tvt,tw_gr,SC_OFFS)

    for k,v in rolls.items(): feats[k]=v

    # Additional GR morphology features. These use only observed horizontal/typewell GR.
    feats['gr_z21']=((hgr-rolls['grm21'])/(rolls['grs21']+1e-3)).astype(np.float32)
    feats['gr_z51']=((hgr-rolls['grm51'])/(rolls['grs51']+1e-3)).astype(np.float32)
    feats['gr_abs_d1']=np.abs(gr_d1).astype(np.float32)
    feats['gr_abs_d2']=np.abs(gr_d2).astype(np.float32)
    feats['gr_d1_rmean21']=_rolling_mean_np(gr_d1,21)
    feats['gr_abs_d1_rmean21']=_rolling_mean_np(np.abs(gr_d1),21)
    feats['gr_rough21']=_rolling_mean_np(np.abs(gr_d1),21)
    feats['gr_rough51']=_rolling_mean_np(np.abs(gr_d1),51)
    _tw_gr_sorted=np.sort(tw_gr.astype(np.float32))
    _kg_sorted=np.sort(kgr.astype(np.float32)) if len(kgr)>0 else _tw_gr_sorted
    feats['gr_pct_tw']=(np.searchsorted(_tw_gr_sorted,hgr,side='right')/max(len(_tw_gr_sorted),1)).astype(np.float32)
    feats['gr_pct_known']=(np.searchsorted(_kg_sorted,hgr,side='right')/max(len(_kg_sorted),1)).astype(np.float32)
    feats['gr_z_known']=((hgr-np.float32(np.mean(kgr)))/(np.float32(np.std(kgr))+1e-3)).astype(np.float32)
    feats['gr_z_tw']=((hgr-np.float32(np.mean(tw_gr)))/(np.float32(np.std(tw_gr))+1e-3)).astype(np.float32)
    feats['gr_vs_known_last']=(hgr-np.float32(kgr[-1])).astype(np.float32)
    feats['gr_vs_known_mean']=(hgr-np.float32(np.mean(kgr))).astype(np.float32)

    # Geometry roughness of the actual drilled path.
    step_xy=np.sqrt(dxdmd**2+dydmd**2).astype(np.float32)
    feats['xy_rate']=step_xy
    feats['xyz_rate']=np.sqrt(dxdmd**2+dydmd**2+dzdmd**2).astype(np.float32)
    feats['z_rate_abs']=np.abs(dzdmd).astype(np.float32)
    feats['dzdmd_rmean21']=_rolling_mean_np(dzdmd,21)
    feats['dzdmd_rstd21']=_rolling_std_np(dzdmd,21)
    feats['xy_rate_rmean21']=_rolling_mean_np(step_xy,21)
    feats['xy_rate_rstd21']=_rolling_std_np(step_xy,21)

    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result


# ==================== self/nbr 列追加(v52 one() line 1725-1763 verbatim) ====================
def build_self_nbr_columns(df, wid, SELF_ENS, NBR_ENS, ZGRAD, NBR_META):
    """build_well 出力 df に、self/nbr smooth-PF と Z傾斜・近傍メタの列を追加。
       SELF_ENS/NBR_ENS は (wid, bank_name) -> dict(mean,std)。ZGRAD/NBR_META は wid -> ...。"""
    n = len(df); lk = df["last_known_tvt"].to_numpy(float)
    def col(a):
        out = np.full(n, np.nan, np.float32)
        if a is not None:
            mm = min(len(a), n); out[:mm] = np.asarray(a[:mm], np.float32)
        return out
    mt = NBR_META.get(wid, {})
    S_all = []; N_all = []; T_all = []
    for p in PF_PARAM_SETS:
        suf = p["suffix"]; nm = p["name"]
        twk = df[f"pf_ancc{suf}"].to_numpy(float) if f"pf_ancc{suf}" in df.columns else lk.copy()
        sf = SELF_ENS.get((wid, nm)); nb = NBR_ENS.get((wid, nm))
        ps = col(sf["mean"] if sf else None); pss = col(sf["std"] if sf else None)
        pn = col(nb["mean"] if nb else None); pns = col(nb["std"] if nb else None)
        df[f"pf_self{suf}"] = ps; df[f"pf_self_std{suf}"] = pss; df[f"pf_self_delta{suf}"] = (ps - lk).astype(np.float32)
        df[f"pf_nbr{suf}"]  = pn; df[f"pf_nbr_std{suf}"]  = pns; df[f"pf_nbr_delta{suf}"]  = (pn - lk).astype(np.float32)
        df[f"pf_self_vs_tw{suf}"] = (ps - twk).astype(np.float32)   # ★最有力: baseline tw との不一致
        df[f"pf_nbr_vs_tw{suf}"]  = (pn - twk).astype(np.float32)
        df[f"pf_self_vs_nbr{suf}"] = (ps - pn).astype(np.float32)
        st = np.stack([twk, ps.astype(float), pn.astype(float)], axis=1)
        df[f"pf_agree_std{suf}"]   = np.nanstd(st, axis=1).astype(np.float32)
        df[f"pf_agree_range{suf}"] = (np.nanmax(st, axis=1) - np.nanmin(st, axis=1)).astype(np.float32)
        S_all.append(ps.astype(float)); N_all.append(pn.astype(float)); T_all.append(twk)
    # pf_1/2/3 間の集約(平均 / param間ばらつき / 平均のtw不一致)
    SM = np.stack(S_all, 1); NM = np.stack(N_all, 1); TM = np.stack(T_all, 1)
    df["pf_self_mean"] = np.nanmean(SM, 1).astype(np.float32); df["pf_self_pstd"] = np.nanstd(SM, 1).astype(np.float32)
    df["pf_nbr_mean"]  = np.nanmean(NM, 1).astype(np.float32); df["pf_nbr_pstd"]  = np.nanstd(NM, 1).astype(np.float32)
    df["pf_self_mean_vs_tw"] = (np.nanmean(SM, 1) - np.nanmean(TM, 1)).astype(np.float32)
    df["pf_nbr_mean_vs_tw"]  = (np.nanmean(NM, 1) - np.nanmean(TM, 1)).astype(np.float32)
    # Z傾斜 + 近傍メタ(近傍を実際に使ったか/最近傍距離/本数)
    df["z_grad"] = col(ZGRAD.get(wid))
    df["pf_nbr_has"]    = np.float32(mt.get("has", 0))
    df["pf_nbr_nndist"] = np.float32(mt.get("nn", np.nan))
    df["pf_nbr_nrefs"]  = np.float32(mt.get("nref", 0))
    return df
