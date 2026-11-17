# -*- coding: utf-8 -*-
"""
v78 P30 — 「非逐次化されたPF」= r0モデルの厳密2次元グリッドFB(決定的・乱数なし)
================================================================================
ユーザー仮説「PFの仕組みを逐次じゃなくする」の忠実実装。
状態: (S偏差 o=pos−S0 [±OMAX,OSTEP格子] × rate [±RMAX,RSTEP格子])。錨なし・絶対空間。
遷移(r0そのまま): rate'=mom·rate+N(0,vn) / pos+=rate·dm+N(0,pn) / まれジャンプ(0.002,σ5ft)。
emission(r0そのまま): gauss^1.7827, σ=prefix残差較正(mult3.058, clip[11.87,44.99])。
init: pos~N(S0,0.933), rate~N(prefix末尾dip,0.000478)。
出力: 全区間平滑posterior mean TVT。 = PFからseed床(±0.2)とlag16打ち切りを除去した理論上位互換。
サブグリッド拡散は分散保存3点カーネル(p=σ²/step²)。mom減衰はDECAY_K行ごとに補間gatherで適用。
使い方: WSTRIDE=5 python v77/p30_exactpf_v77.py
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as Fnn

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NW = int(os.environ.get("NW", "60"))
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))
OMAX = float(os.environ.get("OMAX", "150")); OSTEP = float(os.environ.get("OSTEP", "1.0"))
RMAX = float(os.environ.get("RMAX", "0.09")); RSTEP = float(os.environ.get("RSTEP", "0.003"))
DECAY_K = int(os.environ.get("DECAY_K", "128"))
EMT = float(os.environ.get("EMT", "1.0"))
LAG = int(os.environ.get("LAG", "0"))            # 0=全区間平滑 / >0=固定ラグ平滑(r0はlag16相当)
# r0パラメータ(v73/nnpf_v73.py R0 と同一)
MOM = 0.9997911117425017; VN = 0.0006297672017716754; PN = 0.005014009140727263
IPS = 0.9331541486020234; IRS = 0.00047753793163756304
JP_ = 0.002; JSTD = 5.0
POW = 1.7826896851395453
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)

oo = torch.arange(-OMAX, OMAX + OSTEP / 2, OSTEP, device=DEV); NO = len(oo)
rr = torch.arange(-RMAX, RMAX + RSTEP / 2, RSTEP, device=DEV); NR = len(rr)


def var_kernel(sig, step, dev):
    """σ<stepでも分散を保存する3点/ガウスカーネル"""
    if sig >= step:
        kw = int(np.ceil(4 * sig / step))
        kx = torch.arange(-kw, kw + 1, device=dev, dtype=torch.float32)
        k = torch.exp(-0.5 * (kx * step / sig) ** 2); return (k / k.sum()), kw
    p = min((sig / step) ** 2, 0.5)
    return torch.tensor([p / 2, 1 - p, p / 2], device=dev), 1


KO, KOW = var_kernel(PN, OSTEP, DEV)                 # pos拡散
KJ, KJW = var_kernel(JSTD, OSTEP, DEV)               # ジャンプ
KR, KRW = var_kernel(VN, RSTEP, DEV)                 # rate拡散
# pos-shift(rate·dm): rate列ごとに小数シフト(2点分配)
_sh = (rr / OSTEP)                                   # 行あたり格子シフト量(dm≈1)
_ar = torch.arange(NO, device=DEV)[:, None]
_lo = torch.floor(_sh).long()[None, :]; _fr = (_sh - _lo.float())
_srcA = _ar - _lo; _srcB = _srcA - 1
_mA = (_srcA >= 0) & (_srcA < NO); _mB = (_srcB >= 0) & (_srcB < NO)
_gA = _srcA.clamp(0, NO - 1); _gB = _srcB.clamp(0, NO - 1)
# mom減衰(DECAY_Kごと): rate → mom^K·rate の補間gather
decf = MOM ** DECAY_K
_rq = (rr / decf / RSTEP + (NR - 1) / 2)             # 減衰後グリッドの供給元index(逆写像)
_ri = torch.floor(_rq).long().clamp(0, NR - 2); _rf = (_rq - _ri.float()).clamp(0, 1)


def run_well(wid):
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
    md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
    ge = gr[epos]; tv = ev["TVT"].to_numpy(float)
    T = len(md)
    S0 = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm_ = np.diff(tail["MD"].values)
    mm = dm_ > 0
    d0 = float(np.median((dt + dz)[mm] / dm_[mm])) if mm.sum() >= 3 else 0.0
    d0 = float(np.clip(d0, -RMAX * 0.8, RMAX * 0.8))
    # emission(T,NO): tvt = S0+o−z
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgr = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    tvtr = (S0 + oo[None, :]) - zt[:, None]
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    loge = EMT * (-0.5 * (((grt[:, None] - egr) / gs).abs()) ** POW)   # (T,NO)
    dmv = np.diff(md, prepend=md[0] - 1.0); dmv = np.clip(dmv, 1.0, None)
    dmt = torch.tensor(dmv, device=DEV, dtype=torch.float32)

    def trans(p, dm):
        # rate拡散
        p = Fnn.conv1d(p.reshape(NO, 1, NR), KR.view(1, 1, -1), padding=KRW).reshape(NO, NR)
        # posシフト(rate·dm; dm≈1固定近似, dm>1はループ)
        reps = int(round(float(dm)))
        for _ in range(max(1, min(reps, 3))):
            A = p.gather(0, _gA) * _mA; B = p.gather(0, _gB) * _mB
            p = (1 - _fr) * A + _fr * B
        # pos拡散+ジャンプ混合
        pn_ = Fnn.conv1d(p.t().reshape(NR, 1, NO), KO.view(1, 1, -1), padding=KOW).reshape(NR, NO)
        pj_ = Fnn.conv1d(p.t().reshape(NR, 1, NO), KJ.view(1, 1, -1), padding=KJW).reshape(NR, NO)
        return ((1 - JP_) * pn_ + JP_ * pj_).t()

    def decay(v):
        a = v[:, _ri]; b = v[:, _ri + 1]
        return a * (1 - _rf)[None, :] + b * _rf[None, :]

    lp0 = (-0.5 * ((oo / max(IPS, OSTEP * 0.5)) ** 2))[:, None] + \
          (-0.5 * (((rr - d0) / max(IRS, RSTEP * 0.5)) ** 2))[None, :]
    alpha = torch.zeros(T, NO, NR, device=DEV); beta_last = None
    a = lp0 + loge[0][:, None]
    a -= a.max(); alpha[0] = a
    for t in range(1, T):
        p = torch.exp(alpha[t - 1] - alpha[t - 1].max())
        p = trans(p, float(dmt[t]))
        if t % DECAY_K == 0: p = decay(p)
        a = torch.log(p.clamp_min(1e-30)) + loge[t][:, None]
        a -= a.max(); alpha[t] = a
    def back_step(b, t):
        bb = torch.exp((b + loge[t][:, None]) - (b + loge[t][:, None]).max())
        pn_ = Fnn.conv1d(bb.t().reshape(NR, 1, NO), KO.view(1, 1, -1), padding=KOW).reshape(NR, NO)
        pj_ = Fnn.conv1d(bb.t().reshape(NR, 1, NO), KJ.view(1, 1, -1), padding=KJW).reshape(NR, NO)
        bb = ((1 - JP_) * pn_ + JP_ * pj_).t()
        reps = int(round(float(dmt[t])))
        for _ in range(max(1, min(reps, 3))):
            A2 = bb.gather(0, (_ar + _lo).clamp(0, NO - 1)) * ((_ar + _lo >= 0) & (_ar + _lo < NO))
            B2 = bb.gather(0, (_ar + _lo + 1).clamp(0, NO - 1)) * ((_ar + _lo + 1 >= 0) & (_ar + _lo + 1 < NO))
            bb = (1 - _fr) * A2 + _fr * B2
        bb = Fnn.conv1d(bb.reshape(NO, 1, NR), KR.view(1, 1, -1), padding=KRW).reshape(NO, NR)
        if t % DECAY_K == 0: bb = decay(bb)
        return torch.log(bb.clamp_min(1e-30))

    om = np.empty(T)
    if LAG <= 0:                                     # 全区間平滑
        b = torch.zeros(NO, NR, device=DEV)
        for t in range(T - 1, -1, -1):
            g = torch.softmax((alpha[t] + b).reshape(-1), 0).reshape(NO, NR)
            om[t] = float((g.sum(1) * oo).sum())
            if t > 0: b = back_step(b, t)
    else:                                            # 固定ラグ平滑(ブロック式: 実効ラグ LAG..2LAG)
        for s0 in range(0, T, LAG):
            e0 = min(s0 + LAG, T)
            t_start = min(e0 + LAG, T) - 1
            b = torch.zeros(NO, NR, device=DEV)
            for t in range(t_start, s0 - 1, -1):
                if t < e0:
                    g = torch.softmax((alpha[t] + b).reshape(-1), 0).reshape(NO, NR)
                    om[t] = float((g.sum(1) * oo).sum())
                if t > s0: b = back_step(b, t)
    pred = (S0 + om) - z
    e = pred - tv
    return dict(pred=pred.astype(np.float32), err=e.astype(np.float32))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, per, outs = [], {}, {}
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[w] = r
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es)
    print(f"\n=== P30 exact-PF 2D-FB ({len(per)}井) OMAX={OMAX} OSTEP={OSTEP} NR={NR} EMT={EMT} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (素PF基準: 155標本≈8.0台/773 8.665)  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p30_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
