# -*- coding: utf-8 -*-
"""
v77 P14 — 2チャンネルemission grid-FB(生GR + 低域GR を1つのFBに統合)
====================================================================
P6の emission を loge = W1·loge_raw + W2·loge_lowpass(GRMA) に拡張。
生の波形一致(細部)と長波長一致(大局)を同時に使う単一の強い脚を狙う。
使い方: NW=60 GRMA=13 W1=1 W2=1 python v77/p14_dualch_v77.py
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NW = int(os.environ.get("NW", "60"))
BAND = float(os.environ.get("BAND", "40"))
STEP = float(os.environ.get("STEP", "0.5"))
SRW = float(os.environ.get("SRW", "0.03"))       # オフセット随歩σ(ft/行)
ANCW = float(os.environ.get("ANCW", "0.002"))    # 錨prior(行ごと弱く)
POW = float(os.environ.get("POW", "1.7826896851395453"))
SIGDIV = float(os.environ.get("SIGDIV", "1.0"))  # GR σ分割
GRMA = int(os.environ.get("GRMA", "13"))         # ch2(低域)のwell GR移動平均(行)
TWMA = float(os.environ.get("TWMA", "6"))        # ch2のtypewell GR移動平均(ft)
W1 = float(os.environ.get("W1", "1.0"))          # ch1(生)重み
W2 = float(os.environ.get("W2", "1.0"))          # ch2(低域)重み
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())

offs = torch.arange(-BAND, BAND + STEP / 2, STEP, device=DEV)   # (S,)
S = len(offs)


def run_well(wid):
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    fb_ = float(np.nanmean(tw_gr))
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb_).to_numpy(float)
    # ch1=生 / ch2=低域(GRMA/TWMA)
    gr2 = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same") if GRMA > 1 else gr.copy()
    tw_gr2 = tw_gr.copy()
    if TWMA > 0:
        step = float(np.median(np.diff(tw_tvt)))
        wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
        tw_gr2 = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same")
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    kt = kn["TVT_input"].to_numpy(float)
    def calib(g_, t_):
        r = g_[kpos] - np.interp(kt, tw_tvt, t_)
        s = float(np.nanstd(r))
        if not np.isfinite(s) or s <= 0: s = GRSIG["df"]
        return float(np.clip(s * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"])) / SIGDIV
    gs = calib(gr, tw_gr); gs2 = calib(gr2, tw_gr2)
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sig_dd = a["sig"][:T]
    tv = ev["TVT"].to_numpy(float)[:T]
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    t0v = ttv[ii - 1]; t1v = ttv[ii]
    wv = ((tvtr - t0v) / (t1v - t0v).clamp_min(1e-6)).clamp(0, 1)
    loge = 0.0
    for g_, t_, s_, w_ in ((gr, tw_gr, gs, W1), (gr2, tw_gr2, gs2, W2)):
        if w_ <= 0: continue
        tgr_ = torch.tensor(t_, device=DEV, dtype=torch.float32)
        egr = tgr_[ii - 1] * (1 - wv) + tgr_[ii] * wv
        grt = torch.tensor(g_[epos][:T], device=DEV, dtype=torch.float32)
        d = ((grt[:, None] - egr) / s_).abs()
        loge = loge + w_ * (-0.5 * d ** POW)
    sdd = torch.tensor(sig_dd, device=DEV, dtype=torch.float32).clamp_min(1.0)
    loge = loge - 0.5 * ANCW * (offs[None, :] / sdd[:, None]) ** 2
    # 遷移: ガウスぼかし(kernel固定: σ=SRW·dm, dm≈1ft)。畳み込みカーネル半幅4σ
    ksig = max(SRW / STEP, 0.05)
    kw = int(max(2, np.ceil(4 * ksig)))
    kx = torch.arange(-kw, kw + 1, device=DEV, dtype=torch.float32)
    ker = torch.exp(-0.5 * (kx / ksig) ** 2); ker = ker / ker.sum()
    ker = ker.view(1, 1, -1)
    def blur(v):                                             # (S,) log→prob域でぼかし
        p = torch.exp(v - v.max())
        p = torch.nn.functional.conv1d(p.view(1, 1, -1), ker, padding=kw).view(-1)
        return torch.log(p.clamp_min(1e-30))
    # forward-backward(log域, 逐次)
    T_ = loge.shape[0]
    alpha = torch.zeros(T_, S, device=DEV); beta = torch.zeros(T_, S, device=DEV)
    prior0 = -0.5 * (offs / 3.0) ** 2                        # 開始: 錨近傍(既知点直後なのでタイト)
    alpha[0] = prior0 + loge[0]
    for t in range(1, T_):
        alpha[t] = blur(alpha[t - 1]) + loge[t]
        alpha[t] -= alpha[t].max()
    beta[T_ - 1] = 0
    for t in range(T_ - 2, -1, -1):
        beta[t] = blur(beta[t + 1] + loge[t + 1])
        beta[t] -= beta[t].max()
    post = alpha + beta
    post = torch.softmax(post, dim=1)
    om = (post * offs[None, :]).sum(1).cpu().numpy()
    pred = A + om
    e = pred - tv
    return dict(pred=pred.astype(np.float32), err=e.astype(np.float32), true=tv,
                anc_err=(A - tv).astype(np.float32))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NW]
    es = []; ea = []; per = {}; outs = {}
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); ea.append(r["anc_err"][m])
        per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2)))
        outs[w] = r
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P14 2ch grid-FB ({len(per)}井) SRW={SRW} ANCW={ANCW} W1={W1} W2={W2} GRMA={GRMA} ===")
    print(f"  P14 pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})")
    print(f"  per-well 中央 = {np.median(list(per.values())):.3f}")
    fp = ART / f"p14_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
