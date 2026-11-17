# -*- coding: utf-8 -*-
"""
v78 P19 — 因子グラフ・バッチ最適化(族2本命, 錨付き)
====================================================
変数: 行ごとの連続オフセット off[T](init=0=錨)。一括最小化(Adam)。
因子: ①GRロバスト尤度 Cauchy log1p((d/γ)²)·W_GR
      ②隣接平滑(ロバスト=Geman-McClure): W_SM·τ²·Δ²/(Δ²+τ²) → 大ジャンプのコスト飽和=区間切替許容
      ③錨因子 ANCW·(off/σ_dd)²
P10(低次元スプライン)との違い: 行変数+ロバスト結合=跳びを表現できる。決定的(seed固定)。
使い方: WSTRIDE=5 python v77/p19_factorgraph_v77.py
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
W_GR = float(os.environ.get("W_GR", "1.0"))
GAMMA = float(os.environ.get("GAMMA", "1.0"))     # Cauchyスケール(σ正規化後)
W_SM = float(os.environ.get("W_SM", "50.0"))      # 平滑因子重み(per step)
TAU = float(os.environ.get("TAU", "1.0"))         # ロバスト平滑の飽和幅(ft/行スケール)
ANCW = float(os.environ.get("ANCW", "1.0"))
GRMA = int(os.environ.get("GRMA", "1"))
STEPS = int(os.environ.get("STEPS", "600"))
LR = float(os.environ.get("LR", "0.3"))
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
torch.manual_seed(777)


def load(wid):
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    if GRMA > 1: gr = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    resid = gr[kpos] - np.interp(kn["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    return dict(A=a["tvt"][:T], sdd=np.clip(a["sig"][:T], 1.0, None), ge=gr[epos][:T],
                tw_tvt=tw_tvt, tw_gr=tw_gr, gs=gs, true=ev["TVT"].to_numpy(float)[:T])


def run(c):
    T = len(c["ge"])
    At = torch.tensor(c["A"], device=DEV, dtype=torch.float32)
    ttv = torch.tensor(c["tw_tvt"], device=DEV, dtype=torch.float32)
    tgr = torch.tensor(c["tw_gr"], device=DEV, dtype=torch.float32)
    grt = torch.tensor(c["ge"], device=DEV, dtype=torch.float32)
    sdd = torch.tensor(c["sdd"], device=DEV, dtype=torch.float32)
    off = torch.zeros(T, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([off], lr=LR)
    for it in range(STEPS):
        q = (At + off).clamp(ttv[0], ttv[-1])
        ii = torch.searchsorted(ttv, q.detach().contiguous()).clamp(1, len(ttv) - 1)
        wv = ((q - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
        egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
        d = (grt - egr) / c["gs"]
        c_gr = W_GR * torch.log1p((d / GAMMA) ** 2)
        c_anc = ANCW * (off / sdd) ** 2
        dlt = off[1:] - off[:-1]
        c_sm = W_SM * (TAU ** 2) * dlt ** 2 / (dlt ** 2 + TAU ** 2)   # Geman-McClure(飽和=跳び許容)
        loss = c_gr.mean() + c_anc.mean() + c_sm.sum() / T
        opt.zero_grad(); loss.backward(); opt.step()
    return c["A"] + off.detach().cpu().numpy()


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = wids[::WSTRIDE] if WSTRIDE > 0 else wids[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    for i, w in enumerate(wids):
        c = load(w)
        if c is None: continue
        try:
            pred = run(c)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); continue
        e = pred - c["true"]; m = np.isfinite(e)
        es.append(e[m]); ea.append((c["A"] - c["true"])[m])
        per[w] = float(np.sqrt(np.mean(e[m] ** 2)))
        outs[w] = dict(pred=pred.astype(np.float32))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P19 因子グラフ ({len(per)}井) W_SM={W_SM} TAU={TAU} ANCW={ANCW} GAMMA={GAMMA} GRMA={GRMA} ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p19_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
