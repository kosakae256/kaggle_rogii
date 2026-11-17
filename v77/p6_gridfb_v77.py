# -*- coding: utf-8 -*-
"""
v77 P6 — K16残差帯の行レベル grid forward-backward(決定的・乱数なし)
====================================================================
状態: K16錨からのオフセット o∈[−BAND,+BAND] (STEP刻み)。
遷移: ガウスぼかし(σ=SRW·Δmd) = オフセットのランダムウォーク。
emission: GR尤度 exp(−0.5((gr−tw(A+o))/gs)^POW) × 錨prior exp(−0.5·ANCW·(o/σ_dd)²)。
出力: posterior mean オフセット → TVT = A + E[o]。exact FB(v48系のstruct→K16リブート)。
使い方: NW=60 python v77/p6_gridfb_v77.py
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
EMT = float(os.environ.get("EMT", "1.0"))        # emission温度(<1=軟化)
JP = float(os.environ.get("JP", "0"))            # P18: ジャンプ確率(行あたり)。0=無効
JSIG = float(os.environ.get("JSIG", "15"))       # P18: ジャンプ幅σ(ft)
DGR = os.environ.get("DGR", "0")                 # P27: 1=勾配照合(dGR/dMD vs dtwGR/dTVT)
QMAP = os.environ.get("QMAP", "0")               # v81-C: 1=prefixペアで井戸GR→twGRの単調分位マップを適用
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))    # >0: wids[::WSTRIDE]の全域標本
GRMA = int(os.environ.get("GRMA", "1"))          # well GRの移動平均(行)。>1で低域照合モード
TWMA = float(os.environ.get("TWMA", "0"))        # typewell GRの移動平均(ft)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads(Path(os.environ.get("ANCPKL", str(ART / "k16_anchor_train.pkl"))).read_bytes())

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
    if GRMA > 1:
        gr = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    if TWMA > 0:
        step = float(np.median(np.diff(tw_tvt)))
        wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
        tw_gr = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same")
    if DGR == "1":                                   # P27: 勾配空間へ(振幅は保持される)
        gr = np.gradient(gr)
        tw_gr = np.gradient(tw_gr, tw_tvt)
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    if QMAP == "1":                                  # v81-C: prefix教師ペアから単調GR較正マップ
        gw = gr[kpos]; gt = np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
        mq = np.isfinite(gw) & np.isfinite(gt)
        if mq.sum() > 200:
            qs = np.linspace(2, 98, 25)
            xq = np.percentile(gw[mq], qs); yq = np.percentile(gt[mq], qs)
            xq, ui = np.unique(xq, return_index=True)
            gr = np.interp(gr, xq, yq[ui])           # 井戸GR全体をtwスケールへ写像(単調=分位対応)
    resid = gr[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    if DGR == "1":
        gs = float(max(gs * GRSIG["mult"], 1e-3)) / SIGDIV
    else:
        gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"])) / SIGDIV
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sig_dd = a["sig"][:T]
    ge = gr[epos][:T]
    md = ev["MD"].to_numpy(float)[:T]
    tv = ev["TVT"].to_numpy(float)[:T]
    # emission行列 (T,S): tw GR at A+o
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgr = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    t0v = ttv[ii - 1]; t1v = ttv[ii]; g0 = tgr[ii - 1]; g1 = tgr[ii]
    wv = ((tvtr - t0v) / (t1v - t0v).clamp_min(1e-6)).clamp(0, 1)
    egr = g0 * (1 - wv) + g1 * wv
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    d = ((grt[:, None] - egr) / gs).abs()
    loge = EMT * (-0.5 * d ** POW)
    sdd = torch.tensor(sig_dd, device=DEV, dtype=torch.float32).clamp_min(1.0)
    loge = loge - 0.5 * ANCW * (offs[None, :] / sdd[:, None]) ** 2
    # 遷移: ガウスぼかし(σ=SRW·dm)。P18: JP>0で混合 (1−p)·N(0,σ_rw)+p·N(0,σ_jump)=まれな大ジャンプ許容
    ksig = max(SRW / STEP, 0.05)
    jsig = JSIG / STEP
    kw = int(max(2, np.ceil(4 * (jsig if JP > 0 else ksig))))
    kx = torch.arange(-kw, kw + 1, device=DEV, dtype=torch.float32)
    kn_ = torch.exp(-0.5 * (kx / ksig) ** 2); kn_ = kn_ / kn_.sum()
    if JP > 0:
        kj_ = torch.exp(-0.5 * (kx / jsig) ** 2); kj_ = kj_ / kj_.sum()
        ker = (1 - JP) * kn_ + JP * kj_
    else:
        ker = kn_
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
    om_t = (post * offs[None, :]).sum(1)
    postd = ((post * (offs[None, :] - om_t[:, None]) ** 2).sum(1)).sqrt().cpu().numpy()
    om = om_t.cpu().numpy()
    pred = A + om
    e = pred - tv
    return dict(pred=pred.astype(np.float32), err=e.astype(np.float32), true=tv,
                anc_err=(A - tv).astype(np.float32), postd=postd.astype(np.float32))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
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
    print(f"\n=== P6 K16残差帯 grid-FB ({len(per)}井) SRW={SRW} ANCW={ANCW} SIGDIV={SIGDIV} BAND={BAND} ===")
    print(f"  P6 pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})")
    print(f"  per-well 中央 = {np.median(list(per.values())):.3f}")
    fp = ART / f"p6_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
