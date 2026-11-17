# -*- coding: utf-8 -*-
"""
v78 P22 — NCC窓emission FB(振幅・レベル不変の証拠汎関数)
==========================================================
P6の点差分emissionを「窓NCC(正規化相互相関)」に置換: 各(行,オフセット)で
well GR窓(WIN行) と typewell GR窓(A+o±半窓ft) をz正規化して相関。
GRレベルシフト/振幅スケールに厳密不変=バイアス系誤差を証拠関数の不変性で殺す。
loge = K_N·ncc + 錨prior。他はP6と同一(EMT不要=nccが有界)。
使い方: WSTRIDE=5 K_N=3 python v77/p22_nccfb_v77.py
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
K_N = float(os.environ.get("K_N", "3.0"))        # P22: NCCロジット倍率
WIN = int(os.environ.get("WIN", "21"))           # P22: 窓半径込みの窓長(行,奇数)
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))    # >0: wids[::WSTRIDE]の全域標本
GRMA = int(os.environ.get("GRMA", "1"))          # well GRの移動平均(行)。>1で低域照合モード
TWMA = float(os.environ.get("TWMA", "0"))        # typewell GRの移動平均(ft)
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
    if GRMA > 1:
        gr = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    if TWMA > 0:
        step = float(np.median(np.diff(tw_tvt)))
        wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
        tw_gr = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same")
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    resid = gr[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"])) / SIGDIV
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sig_dd = a["sig"][:T]
    ge = gr[epos][:T]
    md = ev["MD"].to_numpy(float)[:T]
    tv = ev["TVT"].to_numpy(float)[:T]
    # P22 emission: 窓NCC。well窓(T,WIN) × typewell窓(T,S,WIN) をz正規化して相関
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgr = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    h = WIN // 2
    kidx = torch.arange(-h, h + 1, device=DEV)
    ridx = (torch.arange(T, device=DEV)[:, None] + kidx[None, :]).clamp(0, T - 1)   # (T,WIN)
    wwin = grt[ridx]                                                # (T,WIN)
    q = At[ridx][:, None, :] + offs[None, :, None]                  # (T,S,WIN) 窓内の錨形状を保った照会TVT
    ii = torch.searchsorted(ttv, q.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((q - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    twin = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv                    # (T,S,WIN)
    wz = (wwin - wwin.mean(1, keepdim=True)) / wwin.std(1, keepdim=True).clamp_min(1e-3)
    tz = (twin - twin.mean(2, keepdim=True)) / twin.std(2, keepdim=True).clamp_min(1e-3)
    ncc = (wz[:, None, :] * tz).mean(2)                             # (T,S) ∈[-1,1]
    loge = K_N * ncc
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
    print(f"\n=== P22 NCC窓FB ({len(per)}井) K_N={K_N} WIN={WIN} ANCW={ANCW} SRW={SRW} ===")
    print(f"  P22 pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})")
    print(f"  per-well 中央 = {np.median(list(per.values())):.3f}")
    fp = ART / f"p22_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
