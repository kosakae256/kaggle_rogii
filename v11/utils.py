
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from scipy.spatial import cKDTree

# =============================================================================
# Global Configurations & Constants
# =============================================================================

class CFG:
    dataset_path = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    artifacts_path = Path("/kaggle/working/artifacts")
    n_splits = 5
    metric = None

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10; DENSE_SPW = 60; DENSE_K = 20

BEAMS = [
    (10, 20.0, 144.0, 2, "cons"),
    (10,  8.0,  64.0, 2, "loose"),
    ( 8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0,  90.0, 5, "sm5"),
    (20,  4.0,  36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

PF_N = 600; ANCC_N = 600; PF_RESAMP = 0.5
PF_GR_SIG_MIN = 10.; PF_GR_SIG_MAX = 60.; PF_GR_SIG_DEF = 30.
C4B_PF_MOM = 0.993; C4B_PF_VN = 0.005; C4B_PF_PN = 0.01
ANCC_ALPHA = 0.998; ANCC_RN = 0.002; ANCC_PN = 0.005
ANCC_IR = 0.01; ANCC_IS = 0.3; ANCC_RP = 0.1; ANCC_RR = 0.001
PF_INIT_V_STD = 0.02; PF_ROUGH_P = 0.2; PF_ROUGH_V = 0.003; PF_GR_WIN = 5; PF_GR_WT = 0.3

DTW_RADII = (20, 50, 100, 200)
DTW_STOCH_K = 12; DTW_STOCH_TEMP = 3.0

LIKPF_SEEDS = 128; LIKPF_PARTICLES = 500
LIKPF_SCALES = (3.0, 5.0, 8.0, 12.0)
LIKPF_INIT_SPREAD = 4.5; LIKPF_MOM = 0.998; LIKPF_VN = 0.002; LIKPF_PN = 0.005
LIKPF_RP = 0.1; LIKPF_RR = 0.001; LIKPF_RESAMP = 0.5

# =============================================================================
# Numba JIT Core Algorithms
# =============================================================================

@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=True)
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

@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):
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

@njit(cache=True)
def _beam_jit_weighted(sgr, tw_gr, si, BS, mc, es, alpha):
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        w_step=alpha+(1.0-alpha)*(step/max(n-1,1))
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2*w_step/es+mc*(d if d>=0 else -d)
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

@njit(cache=True)
def _dtw_sakoe_chiba(query, ref, radius):
    N = len(query); M = len(ref); INF = 1e18; D = np.full((N, M), INF)
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius); j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            cost = (query[i] - ref[j]) ** 2
            if i == 0 and j == 0: D[i, j] = cost
            elif i == 0:
                prev = D[i, j - 1]
                D[i, j] = cost + (prev if prev < INF else INF)
            elif j == 0:
                prev = D[i - 1, j]
                D[i, j] = cost + (prev if prev < INF else INF)
            else:
                a = D[i - 1, j - 1]; b = D[i - 1, j]; c = D[i, j - 1]
                mn = a if a < b else b
                mn = mn if mn < c else c
                D[i, j] = cost + (mn if mn < INF else INF)
    i = N - 1; j = M - 1
    pi = np.zeros(N + M, np.int64); pj = np.zeros(N + M, np.int64); k = 0
    while i > 0 or j > 0:
        pi[k] = i; pj[k] = j; k += 1
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            a = D[i-1,j-1]; b = D[i-1,j]; c = D[i,j-1]
            if a <= b and a <= c: i -= 1; j -= 1
            elif b <= c: i -= 1
            else: j -= 1
    pi[k] = 0; pj[k] = 0; k += 1
    return D, pi[:k], pj[:k]

@njit(cache=True)
def _dtw_path_to_tvt(pi, pj, tw_tvt, N):
    j_for_i = np.zeros(N, np.int64)
    for k in range(len(pi)): j_for_i[pi[k]] = pj[k]
    result = np.empty(N, np.float32)
    for i in range(N): result[i] = tw_tvt[j_for_i[i]]
    return result

@njit(cache=True)
def _dtw_path_slope(pi, pj, N, smooth_win=5):
    j_for_i = np.zeros(N, np.float64)
    for k in range(len(pi)): j_for_i[pi[k]] = float(pj[k])
    slope = np.zeros(N, np.float32); hw = smooth_win // 2
    for i in range(N):
        i0 = max(0, i - hw); i1 = min(N - 1, i + hw)
        slope[i] = float((j_for_i[i1] - j_for_i[i0]) / (i1 - i0)) if i1 > i0 else 1.0
    return slope

@njit(cache=True)
def _dtw_stochastic_realizations(query, ref, radius, K, temperature, jit_seed=0):
    np.random.seed(jit_seed)
    N = len(query); M = len(ref); INF = 1e18
    slope = (M - 1.0) / max(N - 1.0, 1.0)
    D_base = np.full((N, M), INF)
    for i in range(N):
        j_center = int(round(i * slope))
        j_lo = max(0, j_center - radius); j_hi = min(M - 1, j_center + radius)
        for j in range(j_lo, j_hi + 1):
            D_base[i, j] = (query[i] - ref[j]) ** 2
    paths = np.zeros((K, N), np.int64)
    for k in range(K):
        D = np.full((N, M), INF)
        for i in range(N):
            j_center = int(round(i * slope))
            j_lo = max(0, j_center - radius); j_hi = min(M - 1, j_center + radius)
            for j in range(j_lo, j_hi + 1):
                noise = -temperature * np.log(-np.log(np.random.uniform(1e-10, 1.0)))
                cost = D_base[i, j] + noise
                if i == 0 and j == 0: D[i, j] = cost
                elif i == 0:
                    prev = D[i, j-1]
                    D[i, j] = cost + (prev if prev < INF else INF)
                elif j == 0:
                    prev = D[i-1, j]
                    D[i, j] = cost + (prev if prev < INF else INF)
                else:
                    a = D[i-1,j-1]; b = D[i-1,j]; c = D[i,j-1]
                    mn = a if a < b else b
                    mn = mn if mn < c else c
                    D[i, j] = cost + (mn if mn < INF else INF)
        i = N-1; j = M-1; j_for_i = np.zeros(N, np.int64)
        while i > 0 or j > 0:
            j_for_i[i] = j
            if i == 0: j -= 1
            elif j == 0: i -= 1
            else:
                a = D[i-1,j-1]; b = D[i-1,j]; c = D[i,j-1]
                if a <= b and a <= c: i -= 1; j -= 1
                elif b <= c: i -= 1
                else: j -= 1
        j_for_i[0] = j; paths[k] = j_for_i
    return paths

@njit(cache=True)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP,jit_seed=0):
    np.random.seed(jit_seed)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_

@njit(cache=True)
def _pf_ancc_loglik(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
                    ALPHA,RN,PN,IS,RP,RR,RESAMP,jit_seed=0):
    np.random.seed(jit_seed)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; log_lik=0.0
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            avg_lk=0.0; ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                avg_lk+=w[j]*lk; w[j]*=lk; ws+=w[j]
            log_lik+=np.log(max(avg_lk,1e-300))
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_,log_lik

@njit(cache=True)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP,jit_seed=0):
    np.random.seed(jit_seed)
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls_v=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls_v
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

@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds)
    tmax = vmin + len(gg) * step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
        for j in range(N):
            pos[j] = ls + init_spr * np.random.randn()
            rate[j] = ir + 0.01 * np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM * rate[j] + VN * np.random.randn()
                pos[j] += rate[j] * dm + PN * np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin - 100.: tvt_j = vmin - 100.
                if tvt_j > tmax + 100.: tvt_j = tmax + 100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs; dd = d * d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5 * dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j] * lk; w[j] = w[j] * lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
            neff = 0.0
            for j in range(N): neff += w[j] * w[j]
            neff = 1.0 / neff
            if neff < RESAMP * N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1. / N)
                newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j / N
                    while ci < N - 1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP * np.random.randn()
                    newrate[j] = rate[ci] + RR * np.random.randn()
                for j in range(N):
                    pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1. / N
            est = 0.0
            for j in range(N): est += w[j] * (pos[j] - z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks

# =============================================================================
# JIT Compilation Warm-Up utility
# =============================================================================

def warm_up_jit_compilers():
    """
    Triggers Numba's JIT compilation on a tiny dummy input in the main process,
    saving worker processes from compiling concurrently.
    """
    print("  Pre-compiling Numba JIT functions (warm-up)...")
    dummy_md = np.array([10.0, 11.0], dtype=np.float64)
    dummy_z = np.array([100.0, 101.0], dtype=np.float64)
    dummy_gr = np.array([50.0, 60.0], dtype=np.float64)
    dummy_grid = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)
    
    _ = _interp1(dummy_grid, 1.5, 0.0, 1.0)
    _ = _resamp(np.array([1.0, 2.0]), np.array([0.1, 0.2]), np.array([0.5, 0.5]), 2, 0.1, 0.1)
    _ = _beam_jit(dummy_gr, dummy_grid, 0, 2, 1.0, 1.0)
    _ = _beam_jit_weighted(dummy_gr, dummy_grid, 0, 2, 1.0, 1.0, 0.3)
    _ = _dtw_sakoe_chiba(dummy_gr, dummy_grid, 2)
    _ = _pf_ancc(dummy_md, dummy_z, dummy_gr, dummy_grid, 0.0, 1.0, 1.0, 10.0, 0.1, 10, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5)
    _ = _pf_lik_allseeds(dummy_md, dummy_z, dummy_gr, dummy_grid, 0.0, 1.0, 1.0, 10.0, 0.1, 10, 2, 0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.5, 1.0)
    print("  ✓ JIT Warm-up successful.")

# =============================================================================
# Python Wrappers for Physical Prior Computations
# =============================================================================

def _grid(tw_tvt, tw_gr, step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw, tw_tvt, tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr, v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals, fb, r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def beam_search_weighted(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r, alpha=0.3):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit_weighted(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es),float(alpha))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw, tw_tvt, tw_gr, N=ANCC_N, jit_seed=0):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP,
                      jit_seed)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_z(hw, tw_tvt, tw_gr, N=PF_N, jit_seed=0):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
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
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   C4B_PF_MOM,C4B_PF_VN,C4B_PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP,
                   jit_seed)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_ancc_loglik(hw, tw_tvt, tw_gr, N=150, jit_seed=0):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([]),0.0
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std,ll=_pf_ancc_loglik(
        ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
        ev['GR'].values.astype(np.float64),gg,gmin,gst,
        gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP,
        jit_seed)
    return pts.astype(np.float32),std.astype(np.float32),float(ll)

def run_pf_lik_ensemble(hw, tw_tvt, tw_gr, n_seeds=16,
                        n_particles=150, scale=5.0, base_seed=0):
    preds=[]; logliks=[]
    for s in range(n_seeds):
        seed_s=(base_seed+s*7919)%(2**31)
        pts,_,ll=run_pf_ancc_loglik(hw,tw_tvt,tw_gr,N=n_particles,jit_seed=seed_s)
        if len(pts)==0: return np.array([]),np.array([]),0.0
        preds.append(pts); logliks.append(ll)
    logliks=np.array(logliks,dtype=np.float64)
    weights=np.exp((logliks-logliks.max())/scale); weights/=weights.sum()
    pred_stack=np.stack(preds,0)
    ensemble=(weights[:,None]*pred_stack).sum(0).astype(np.float32)
    ens_std=np.sqrt((weights[:,None]*(pred_stack-ensemble[None,:])**2).sum(0)).astype(np.float32)
    return ensemble,ens_std,float(logliks.max())

def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII):
    N = len(query_gr)
    qn = (query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)
    rn = (tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)
    qn_f = qn.astype(np.float64); rn_f = rn.astype(np.float64)
    dtw_tvts = {}; dtw_slopes = {}; dtw_costs = {}
    inv_cost_sum = 0.0; tvt_stack = []
    for r in radii:
        D, pi, pj = _dtw_sakoe_chiba(qn_f, rn_f, r)
        cost = float(D[len(qn_f)-1, len(rn_f)-1]) / max(len(qn_f)+len(rn_f), 1)
        tvt_pred = _dtw_path_to_tvt(pi[::-1], pj[::-1], tw_tvt.astype(np.float32), N)
        slope = _dtw_path_slope(pi[::-1], pj[::-1], N)
        dtw_tvts[r] = tvt_pred; dtw_slopes[r] = slope; dtw_costs[r] = cost
        ic = 1.0 / (cost + 1e-6); inv_cost_sum += ic
        tvt_stack.append((tvt_pred, ic))
    weights = np.array([ic / inv_cost_sum for _, ic in tvt_stack], dtype=np.float32)
    tvts_mat = np.stack([t for t, _ in tvt_stack], axis=1)
    dtw_ens = (tvts_mat * weights[None, :]).sum(axis=1).astype(np.float32)
    return dtw_tvts, dtw_slopes, dtw_costs, dtw_ens

def run_dtw_stochastic(query_gr, tw_tvt, tw_gr, last_tvt,
                       radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP, jit_seed=0):
    N = len(query_gr)
    qn = ((query_gr - query_gr.mean()) / (query_gr.std() + 1e-6)).astype(np.float64)
    rn = ((tw_gr - tw_gr.mean()) / (tw_gr.std() + 1e-6)).astype(np.float64)
    paths = _dtw_stochastic_realizations(qn, rn, radius, K, temperature, jit_seed)
    tvt_realiz = np.empty((K, N), dtype=np.float32)
    for k in range(K):
        for i in range(N): tvt_realiz[k, i] = tw_tvt[paths[k, i]]
    mean_tvt = tvt_realiz.mean(axis=0).astype(np.float32)
    std_tvt = tvt_realiz.std(axis=0).astype(np.float32)
    cv_tvt = (std_tvt / (np.abs(mean_tvt) + 1e-6)).astype(np.float32)
    return mean_tvt, std_tvt, cv_tvt

def robust_slope(x, y, w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def seg_b_well(ktvt, kz, form_col):
    bv=ktvt+kz-form_col; n=len(bv); b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr, ktvt, hgr, hws=(8,15,25), stride=3):
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
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens

def lik_pf(hw, tw, n_particles=LIKPF_PARTICLES, n_seeds=LIKPF_SEEDS,
           scales=LIKPF_SCALES, init_spr=LIKPF_INIT_SPREAD, seed_base=0):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s['TVT'].values.astype(float)
    tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(float)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return {}, np.array([], dtype=int)
    last = kn.iloc[-1]; ls = float(last['TVT_input']) + float(last['Z'])
    tw_at_k = np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn['GR'].fillna(0).values - tw_at_k), 10., 60.))
    tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values); dz = np.diff(tail['Z'].values)
    dm = np.diff(tail['MD'].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    tmin = float(tw_tvt.min()); tmax_tw = float(tw_tvt.max()); gstep = 0.2
    tvt_g = np.arange(tmin, tmax_tw + gstep, gstep)
    gg = np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64)
    gr_v = hw['GR'].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)[ev.index]
    preds, liks = _pf_lik_allseeds(
        ev['MD'].values.astype(float), ev['Z'].values.astype(float), gr_v,
        gg, tmin, gstep, gs, ls, ir, n_particles, n_seeds, seed_base,
        LIKPF_MOM, LIKPF_VN, LIKPF_PN, LIKPF_RP, LIKPF_RR, LIKPF_RESAMP, init_spr)
    ln = liks - liks.max(); out = {}
    for sc in scales:
        wts = np.exp(ln / float(sc)); wts /= wts.sum()
        out[f"pf_scale_{sc:g}"] = (wts[:, None] * preds).sum(0).astype(np.float32)
    out["pf_mean"] = preds.mean(0).astype(np.float32)
    return out, ev.index.values

# =============================================================================
# Spatial & Geological Formations KNN Imputers
# =============================================================================

class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord_=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord_,1); ik=np.take_along_axis(idx,ord_,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self, xy_q, self_wid=None, k=DENSE_K, nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord_=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord_,1); ik=np.take_along_axis(idx,ord_,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)

# =============================================================================
# 56-Dimensional Feature Builder Module
# =============================================================================

FEATURE_NAMES = [
    "gr", "gr_d1", "gr_env", "dz_dmd", "dx_dmd", "dy_dmd",
    "beam_cons_d", "beam_sm5_d", "beam_cons_w_d", "beam_mean_d", "beam_std_d",
    "pf_ancc_d", "pf_ens_d", "pf_z_d", "pf_std",
    "sc8_d", "sc15_d", "sc25_d", "sc_ens_d", "hyb_d",
    "dtw_ens_d", "dtw_stoch_mean_d", "dtw_r50_d", "dtw_r100_d", "dtw_stoch_std", "dtw_slope_mean",
    "form_mean_d", "form_std_d", "tvtF_ANCC_d", "tvtF_BUDA_d",
    "tvt_dense_d", "tvt_densew_d",
    "likpf_s3_d", "likpf_s5_d", "likpf_s8_d", "likpf_s12_d",
    "tdbc_n5", "tdbc_0", "tdbc_5",
    "tdpf_n5", "tdpf_0", "tdpf_5",
    "tddtw_n5", "tddtw_0", "tddtw_5",
    "dtw_vs_beam", "dtw_vs_pf", "sc_vs_beam", "pf_ens_vs_single", "sig_std", "dtw_stoch_cv",
    "md_since", "frac", "z_rel", "dxy", "azimuth",
]

def build_well_features(hw_path, tw_path, is_train, fi_imputer, di_imputer, seed_multiplier=8):
    """Build 56-dim feature matrix + delta target for one wellbore."""
    wid = Path(hw_path).stem.replace('__horizontal_well', '')
    _well_seed = int(hashlib.md5(wid.encode()).hexdigest(), 16) % (2**31)
    np.random.seed(seed_multiplier + _well_seed)
    
    try:
        hw = pd.read_csv(hw_path)
        tw = pd.read_csv(tw_path).sort_values('TVT')
    except Exception:
        return None
        
    if is_train and 'TVT' not in hw.columns: return None
    kn = hw[hw['TVT_input'].notna()]
    ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw['TVT'].isna().all(): return None
    
    tw_tvt = tw['TVT'].to_numpy(np.float32)
    tw_gr = tw['GR'].fillna(tw['GR'].mean()).to_numpy(np.float32)
    if len(tw_tvt) < 3: return None

    lk = kn.iloc[-1]; last_tvt = float(lk['TVT_input']); nh = len(ev)

    _jit_seed_a = (seed_multiplier + _well_seed) % (2**31)
    _jit_seed_z = (seed_multiplier + _well_seed + 1) % (2**31)
    _jit_seed_dtw = (seed_multiplier + _well_seed + 2) % (2**31)

    pf_a, std_a = run_pf_ancc(hw, tw_tvt, tw_gr, jit_seed=_jit_seed_a)
    if len(pf_a) == 0: return None
    pf_z_pred, _ = run_pf_z(hw, tw_tvt, tw_gr, jit_seed=_jit_seed_z)
    has_z = len(pf_z_pred) == len(pf_a) and not np.any(np.isnan(pf_z_pred))
    pf_ens, pf_ens_std, _ = run_pf_lik_ensemble(hw, tw_tvt, tw_gr, base_seed=_jit_seed_a)

    gr_full = hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)[:nh]
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in BEAMS:
        bpaths[tag] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    bpaths['cons_w'] = beam_search_weighted(hgr, tw_tvt, tw_gr, last_tvt, 10, 20.0, 144.0, 2, alpha=0.3)
    beam_ref = (bpaths['cons'] + bpaths['sm5']) / 2.

    ktvt = kn['TVT_input'].to_numpy(np.float32)
    sc_res, sc_ens = multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, _ = sc_res[0]; sc15, _ = sc_res[1]; sc25, _ = sc_res[2]
    sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    full_gr = gr_full.values.astype(np.float32)
    dtw_tvts, dtw_slopes, _, dtw_ens_ev_full = run_dtw_multiscale(full_gr, tw_tvt, tw_gr, last_tvt)
    dtw_mean_stoch, dtw_std_stoch, dtw_cv_stoch = run_dtw_stochastic(
        full_gr, tw_tvt, tw_gr, last_tvt, radius=50, K=DTW_STOCH_K,
        temperature=DTW_STOCH_TEMP, jit_seed=_jit_seed_dtw)

    ev_start = ev.index[0]
    def _ev_slice(arr): return arr[ev_start:ev_start+nh].astype(np.float32)

    dtw_ens_ev = _ev_slice(dtw_ens_ev_full)
    dtw_mean_ev = _ev_slice(dtw_mean_stoch)
    dtw_std_ev = _ev_slice(dtw_std_stoch)
    dtw_cv_ev = _ev_slice(dtw_cv_stoch)
    dtw_r50_ev = _ev_slice(dtw_tvts[50])
    dtw_r100_ev = _ev_slice(dtw_tvts[100])
    dtw_slope_mean_ev = np.stack([_ev_slice(dtw_slopes[r]) for r in DTW_RADII], 1).mean(1).astype(np.float32)

    swid = wid if is_train else None
    xy_ev = ev[["X", "Y"]].to_numpy(np.float64)
    xy_kn = kn[["X", "Y"]].to_numpy(np.float64)
    form_ev, _ = fi_imputer.impute(xy_ev, self_wid=swid)
    form_kn, _ = fi_imputer.impute(xy_kn, self_wid=swid)
    z_kn = kn['Z'].to_numpy(np.float32); z_ev = ev['Z'].to_numpy(np.float32)

    fi_ancc = FORMATIONS.index("ANCC"); fi_buda = FORMATIONS.index("BUDA")
    b_ancc, _, _, _, _ = seg_b_well(ktvt, z_kn, form_kn[:, fi_ancc])
    b_buda, _, _, _, _ = seg_b_well(ktvt, z_kn, form_kn[:, fi_buda])
    tvtF_ANCC = (-z_ev + form_ev[:, fi_ancc] + b_ancc).astype(np.float32)
    tvtF_BUDA = (-z_ev + form_ev[:, fi_buda] + b_buda).astype(np.float32)

    form_list = []
    for fi2, fn in enumerate(FORMATIONS):
        b_full, _, _, _, _ = seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        form_list.append((-z_ev + form_ev[:, fi2] + b_full).astype(np.float32))
    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)

    d_ancc, _, _ = di_imputer.impute(xy_ev, self_wid=swid)
    d_kn, _, _ = di_imputer.impute(xy_kn, self_wid=swid)
    b_d = float(np.median(ktvt + z_kn - d_kn))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    _, _, _, _, b_dw = seg_b_well(ktvt, z_kn, d_kn)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)

    likpf_out, _ = lik_pf(hw, tw, seed_base=_well_seed)
    likpf_s3 = likpf_out.get("pf_scale_3", np.full(nh, last_tvt, np.float32))
    likpf_s5 = likpf_out.get("pf_scale_5", np.full(nh, last_tvt, np.float32))
    likpf_s8 = likpf_out.get("pf_scale_8", np.full(nh, last_tvt, np.float32))
    likpf_s12 = likpf_out.get("pf_scale_12", np.full(nh, last_tvt, np.float32))

    hmd = ev['MD'].to_numpy(np.float32); md_since = hmd - float(lk['MD'])
    frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)

    gr_s = pd.Series(gr_full.values)
    gr_d1 = gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)

    mdd = hw['MD'].diff().replace(0, np.nan)
    dzdmd = (hw['Z'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw['X'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw['Y'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxy = np.sqrt((ev['X'].values - float(lk['X']))**2 + (ev['Y'].values - float(lk['Y']))**2).astype(np.float32)
    
    _dx_diff = hw['X'].diff().iloc[ev.index].values.astype(np.float64)
    _dy_diff = hw['Y'].diff().iloc[ev.index].values.astype(np.float64)
    azimuth = np.arctan2(_dy_diff, _dx_diff).astype(np.float32); z_rel = (z_ev - float(lk['Z'])).astype(np.float32)

    all_sigs = [pf_a, bpaths['cons'], bpaths['sm5'], sc_ens, tvtF_ANCC, tvt_dense, dtw_ens_ev]
    sig_std = np.stack(all_sigs, 1).std(1).astype(np.float32)

    pf_ens_use = pf_ens if len(pf_ens) > 0 else np.full(nh, last_tvt, np.float32)
    pf_ens_std_use = pf_ens_std if len(pf_ens_std) > 0 else np.zeros(nh, np.float32)
    pf_z_use = pf_z_pred.astype(np.float32) if has_z else np.full(nh, last_tvt, np.float32)

    def _td(pred_arr, offset): return hgr - np.interp(pred_arr + offset, tw_tvt, tw_gr).astype(np.float32)

    feat_matrix = np.column_stack([
        hgr, gr_d1, gr_env, dzdmd, dxdmd, dydmd,
        bpaths['cons'] - last_tvt, bpaths['sm5'] - last_tvt, bpaths['cons_w'] - last_tvt,
        np.stack([p - last_tvt for p in bpaths.values()], 1).mean(1),
        np.stack([p - last_tvt for p in bpaths.values()], 1).std(1),
        pf_a - last_tvt, pf_ens_use - last_tvt, pf_z_use - last_tvt, std_a,
        sc8 - last_tvt, sc15 - last_tvt, sc25 - last_tvt, sc_ens - last_tvt, hyb_ref - last_tvt,
        dtw_ens_ev - last_tvt, dtw_mean_ev - last_tvt, dtw_r50_ev - last_tvt, dtw_r100_ev - last_tvt,
        dtw_std_ev, dtw_slope_mean_ev,
        form_mean_d, form_std_d, tvtF_ANCC - last_tvt, tvtF_BUDA - last_tvt,
        tvt_dense - last_tvt, tvt_densew - last_tvt,
        likpf_s3 - last_tvt, likpf_s5 - last_tvt, likpf_s8 - last_tvt, likpf_s12 - last_tvt,
        _td(beam_ref, -5.), _td(beam_ref, 0.), _td(beam_ref, 5.),
        _td(pf_a, -5.), _td(pf_a, 0.), _td(pf_a, 5.),
        _td(dtw_ens_ev, -5.), _td(dtw_ens_ev, 0.), _td(dtw_ens_ev, 5.),
        dtw_ens_ev - bpaths['cons'], dtw_ens_ev - pf_a,
        sc_ens - bpaths['cons'], (pf_ens_use - pf_a),
        sig_std, dtw_cv_ev,
        md_since, frac, z_rel, dxy, azimuth,
    ]).astype(np.float32)

    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    result = {
        'wid': wid, 'features': feat_matrix, 'last_tvt': last_tvt,
        'md_since': md_since, 'n_eval': nh,
    }
    if is_train:
        result['target'] = ev['TVT'].to_numpy(np.float32) - np.float32(last_tvt)

    return result
