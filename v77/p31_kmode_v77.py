# -*- coding: utf-8 -*-
"""
v78 P31 — K-best Viterbiモード列挙 + evidence混合(決定的「32seed PF」)
======================================================================
P30の結論(PFの強さ=モードコミット+seed尤度混合)の正しい形:
- r0モデルの格子上で max-product Viterbi(ジャンプ禁止=接続強制)→ 最良経路(モード1)。
- 経路±TUBE ftを封鎖して再実行 → モード2..K(構造的モード列挙、乱数なし)。
- 各モードのスコア(経路log確率)を softmax(温度MT)で混合 → soft TVT。
時間はSUBKブロック化(rate雑音の格子量子化歪みを緩和)。
使い方: WSTRIDE=25 python v77/p31_kmode_v77.py
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
WSTRIDE = int(os.environ.get("WSTRIDE", "0"))
OMAX = float(os.environ.get("OMAX", "150")); OSTEP = float(os.environ.get("OSTEP", "1.0"))
RMAX = float(os.environ.get("RMAX", "0.09")); RSTEP = float(os.environ.get("RSTEP", "0.003"))
SUBK = int(os.environ.get("SUBK", "8"))          # ブロック行数
KMODE = int(os.environ.get("KMODE", "6"))
TUBE = float(os.environ.get("TUBE", "4.0"))      # モード封鎖幅(ft)
MT = float(os.environ.get("MT", "15.0"))         # 混合温度(経路スコア単位)
DRMAX = int(os.environ.get("DRMAX", "2"))        # ブロックあたりrate変化(格子数)
MOM = 0.9997911117425017; VN = 0.0006297672017716754; PN = 0.005014009140727263
IPS = 0.9331541486020234; IRS = 0.00047753793163756304
POW = 1.7826896851395453
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
oo = torch.arange(-OMAX, OMAX + OSTEP / 2, OSTEP, device=DEV); NO = len(oo)
rr = torch.arange(-RMAX, RMAX + RSTEP / 2, RSTEP, device=DEV); NR = len(rr)


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
    d0 = float(np.clip(np.median((dt + dz)[mm] / dm_[mm]) if mm.sum() >= 3 else 0.0, -RMAX * 0.8, RMAX * 0.8))
    # 行emission(T,NO) → ブロック和(NB,NO)
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    tgr = torch.tensor(tw_gr, device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    tvtr = (S0 + oo[None, :]) - zt[:, None]
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(ge, device=DEV, dtype=torch.float32)
    loge_row = -0.5 * (((grt[:, None] - egr) / gs).abs()) ** POW
    NB = (T + SUBK - 1) // SUBK
    pad = NB * SUBK - T
    lr = torch.cat([loge_row, torch.zeros(pad, NO, device=DEV)], 0) if pad else loge_row
    E = lr.reshape(NB, SUBK, NO).sum(1)                              # (NB,NO)
    dmv = np.clip(np.diff(md, prepend=md[0] - 1.0), 1.0, None)
    dmb = torch.tensor(np.concatenate([dmv, np.zeros(pad)]).reshape(NB, SUBK).sum(1),
                       device=DEV, dtype=torch.float32)              # ブロックMD長
    # rate遷移候補(ブロック単位): Δr∈[-DRMAX,DRMAX], pen=0.5(Δr·RSTEP)²/(VN²·dmb)
    drs = torch.arange(-DRMAX, DRMAX + 1, device=DEV)
    # 封鎖マスク付きViterbiをKMODE回
    block_pen = torch.zeros(NB, NO, device=DEV)
    paths, scores = [], []
    ar_o = torch.arange(NO, device=DEV)
    init = (-0.5 * ((oo / max(IPS, 0.5)) ** 2))[:, None] + \
           (-0.5 * (((rr - d0) / max(IRS * 20, RSTEP)) ** 2))[None, :]  # rate初期は緩め(20倍)
    for k in range(KMODE):
        v = init + (E[0] - block_pen[0])[:, None]
        bp = torch.zeros(NB, NO, NR, dtype=torch.int8, device=DEV)
        for b in range(1, NB):
            pen_r = 0.5 * (drs.float() * RSTEP) ** 2 / (VN ** 2 * dmb[b] + 1e-12)
            # o_src = o − r·dmb (rate=遷移後グリッド r)
            shift = (rr * dmb[b] / OSTEP)
            osrc = (ar_o[:, None] - torch.round(shift)[None, :].long()).clamp(0, NO - 1)  # (NO,NR)
            valid = (ar_o[:, None] - torch.round(shift)[None, :].long() >= 0) & \
                    (ar_o[:, None] - torch.round(shift)[None, :].long() <= NO - 1)
            best = torch.full((NO, NR), -1e30, device=DEV); bidx = torch.zeros(NO, NR, dtype=torch.int8, device=DEV)
            for di, dr in enumerate(drs.tolist()):
                rsrc = (torch.arange(NR, device=DEV) - dr).clamp(0, NR - 1)
                okr = ((torch.arange(NR, device=DEV) - dr) >= 0) & ((torch.arange(NR, device=DEV) - dr) <= NR - 1)
                cand = v[osrc, rsrc[None, :].expand(NO, -1)] - pen_r[di]
                cand = torch.where(valid & okr[None, :], cand, torch.full_like(cand, -1e30))
                upd = cand > best
                best = torch.where(upd, cand, best); bidx = torch.where(upd, torch.tensor(di, dtype=torch.int8, device=DEV), bidx)
            v = best + (E[b] - block_pen[b])[:, None]
            bp[b] = bidx
        sc, flat = torch.max(v.reshape(-1), 0)
        oi = int(flat) // NR; ri = int(flat) % NR
        path_o = np.empty(NB, np.int32)
        for b in range(NB - 1, -1, -1):
            path_o[b] = oi
            if b > 0:
                di = int(bp[b, oi, ri]); dr = int(drs[di])
                shift = int(round(float(rr[ri] * dmb[b] / OSTEP)))
                oi = int(np.clip(oi - shift, 0, NO - 1)); ri = int(np.clip(ri - dr, 0, NR - 1))
        paths.append(path_o); scores.append(float(sc))
        # 封鎖: 経路±TUBE
        po = torch.tensor(path_o, device=DEV, dtype=torch.long)
        dist = (oo[None, :] - oo[po][:, None]).abs()
        block_pen = block_pen + torch.where(dist < TUBE, torch.full_like(dist, 30.0), torch.zeros_like(dist))
    sc = np.array(scores)
    wgt = np.exp((sc - sc.max()) / (MT * NB / 100.0)); wgt /= wgt.sum()   # MT=100ブロックあたり温度
    o_blk = np.sum(np.stack([oo.cpu().numpy()[p] for p in paths], 0) * wgt[:, None], 0)   # (NB,)
    o_row = np.repeat(o_blk, SUBK)[:T]
    pred = (S0 + o_row) - z
    e = pred - tv
    return dict(pred=pred.astype(np.float32), err=e.astype(np.float32), wmax=float(wgt.max()))


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, per, outs, wmx = [], {}, {}, []
    t0 = time.time()
    for i, w in enumerate(wids):
        try:
            r = run_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); per[w] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[w] = r
        wmx.append(r["wmax"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es)
    print(f"\n=== P31 K-modeViterbi ({len(per)}井) K={KMODE} TUBE={TUBE} MT={MT} SUBK={SUBK} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}  中央 {np.median(list(per.values())):.3f}  wmax中央 {np.median(wmx):.2f}")
    fp = ART / f"p31_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
