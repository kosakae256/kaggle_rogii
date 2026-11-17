# -*- coding: utf-8 -*-
"""
v78 P32 — 粗→細 段階整列(US11466560の8段coarse-to-fineの2段版, CPU専用)
========================================================================
文献(docs/gr_literature.md #2)の未回収テク: 粗い段が細い段の探索帯を「絞る」。
段1(粗): K16錨帯±45ft, stride16ブロックFB, GR=MA51(低域) → 粗オフセットc(row)+std
段2(細): 帯中心=K16+c(row), 帯±FBAND(既定10ft), 行FB, 生GR, EMT
「低域を別脚にする」(P6)とも「同解像度で自己整合」(P28)とも違う=解像度が段で変わる。
CPU実行(torch cpu, threads=4)。createのGPUを邪魔しない。
使い方: WSTRIDE=5 python v77/p32_coarse2fine_v77.py
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
DEV = "cpu"

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
NW = int(os.environ.get("NW", "60"))
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))
# 段1(粗)
C_STRIDE = int(os.environ.get("C_STRIDE", "16"))
C_BAND = 45.0; C_STEP = 1.0
C_GRS = int(os.environ.get("C_GRS", "51"))
C_TAU = float(os.environ.get("C_TAU", "0.35"))   # ブロック間随歩σ(ft/ブロック)
C_ANCW = float(os.environ.get("C_ANCW", "1.0"))  # 錨prior(σ(dd)使用, LPR相当)
C_LPR = float(os.environ.get("C_LPR", "0.3"))
C_SIGG = float(os.environ.get("C_SIGG", "6.0"))
# 段2(細)
FBAND = float(os.environ.get("FBAND", "10")); FSTEP = 0.5
F_EMT = float(os.environ.get("F_EMT", "0.3"))
F_SRW = float(os.environ.get("F_SRW", "0.02"))
F_ANCW = float(os.environ.get("F_ANCW", "0.02"))  # 段1中心へのprior(σ=段1 std)
POW = 1.7826896851395453
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())


def run_well(wid):
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    kt = kn["TVT_input"].to_numpy(float)
    resid = gr[kpos] - np.interp(kt, tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sdd = np.clip(a["sig"][:T], 1.0, None)
    ge = gr[epos][:T]; tv = ev["TVT"].to_numpy(float)[:T]

    # ===== 段1: ブロックFB(粗) =====
    nb = (T + C_STRIDE - 1) // C_STRIDE; bid = np.arange(T) // C_STRIDE
    def bm(x):
        o = np.zeros(nb); c = np.zeros(nb); np.add.at(o, bid, x); np.add.at(c, bid, 1); return o / np.maximum(c, 1)
    gsm = pd.Series(ge).rolling(C_GRS, center=True, min_periods=1).mean().to_numpy()
    gb = bm(gsm); Ab = bm(A); sb = np.clip(bm(sdd), 1, 60)
    gnp = np.arange(-C_BAND, C_BAND + C_STEP / 2, C_STEP); NC = len(gnp)
    z = Ab[:, None] + gnp[None, :]
    eg = np.interp(z.ravel(), tw_tvt, tw_gr).reshape(nb, NC)
    L = np.exp(-0.5 * ((gb[:, None] - eg) / C_SIGG) ** 2)
    Pr = np.exp(-0.5 * (gnp[None, :] / sb[:, None]) ** 2)
    obs = torch.tensor(L * Pr ** (C_LPR * C_ANCW), dtype=torch.float32)
    obs = obs / obs.sum(1, keepdim=True).clamp_min(1e-30)
    dd_ = torch.tensor(gnp[:, None] - gnp[None, :], dtype=torch.float32)
    M = torch.exp(-0.5 * (dd_ / max(C_TAU * C_STRIDE ** 0.5, 0.3)) ** 2)
    M = M / M.sum(0, keepdim=True).clamp_min(1e-30)
    al = torch.zeros(nb, NC); al[0] = obs[0]
    for t in range(1, nb):
        p = M @ al[t - 1]; al[t] = (p.clamp_min(1e-30) * obs[t]); al[t] /= al[t].sum().clamp_min(1e-30)
    be = torch.zeros(nb, NC); be[-1] = 1.0 / NC
    for t in range(nb - 2, -1, -1):
        msg = be[t + 1] * obs[t + 1]; be[t] = (M @ msg).clamp_min(1e-30); be[t] /= be[t].sum().clamp_min(1e-30)
    gam = al * be; gam = gam / gam.sum(1, keepdim=True).clamp_min(1e-30)
    gt = torch.tensor(gnp, dtype=torch.float32)
    ch = (gam * gt[None, :]).sum(1).numpy()
    cstd = np.sqrt((gam * (gt[None, :] - torch.tensor(ch)[:, None]) ** 2).sum(1).numpy())
    c_row = ch[bid]; cstd_row = np.clip(cstd[bid], 1.5, 20.0)

    # ===== 段2: 再センター帯の行FB(細) =====
    A2 = A + c_row                                     # 段1で再センターした帯中心
    offs = torch.arange(-FBAND, FBAND + FSTEP / 2, FSTEP); S = len(offs)
    At = torch.tensor(A2, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, dtype=torch.float32); tgr = torch.tensor(tw_gr, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(ge, dtype=torch.float32)
    loge = F_EMT * (-0.5 * (((grt[:, None] - egr) / gs).abs()) ** POW)
    sct = torch.tensor(cstd_row, dtype=torch.float32)
    loge = loge - 0.5 * F_ANCW * (offs[None, :] / sct[:, None]) ** 2
    ksig = max(F_SRW / FSTEP, 0.05)
    kw = 2
    kx = torch.arange(-kw, kw + 1, dtype=torch.float32)
    p_ = min((ksig) ** 2, 0.5)
    ker = torch.tensor([0., p_ / 2, 1 - p_, p_ / 2, 0.])            # 分散保存3点
    ker = (ker / ker.sum()).view(1, 1, -1)
    import torch.nn.functional as Fnn
    def blur(v):
        p = torch.exp(v - v.max())
        p = Fnn.conv1d(p.view(1, 1, -1), ker, padding=kw).view(-1)
        return torch.log(p.clamp_min(1e-30))
    alpha = torch.zeros(T, S); beta = torch.zeros(T, S)
    alpha[0] = -0.5 * (offs / 3.0) ** 2 + loge[0]
    for t in range(1, T):
        alpha[t] = blur(alpha[t - 1]) + loge[t]; alpha[t] -= alpha[t].max()
    for t in range(T - 2, -1, -1):
        beta[t] = blur(beta[t + 1] + loge[t + 1]); beta[t] -= beta[t].max()
    post = torch.softmax(alpha + beta, dim=1)
    om = (post * offs[None, :]).sum(1).numpy()
    pred = A2 + om
    return dict(pred=pred.astype(np.float32), err=(pred - tv).astype(np.float32),
                anc_err=(A - tv).astype(np.float32), c1_err=(A2 - tv).astype(np.float32))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, ea, e1, per, outs = [], [], [], {}, {}
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); ea.append(r["anc_err"][m]); e1.append(r["c1_err"][m])
        per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[w] = r
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea); c1 = np.concatenate(e1)
    print(f"\n=== P32 粗→細 ({len(per)}井, CPU) FBAND={FBAND} F_EMT={F_EMT} C_GRS={C_GRS} ===")
    print(f"  段1(粗)のみ = {np.sqrt(np.mean(c1**2)):.4f}  段2(細) = {np.sqrt(np.mean(e**2)):.4f}  (錨 = {np.sqrt(np.mean(a**2)):.4f})")
    print(f"  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p32_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
