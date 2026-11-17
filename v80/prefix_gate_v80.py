# -*- coding: utf-8 -*-
"""
v80a — prefix-backtestゲート: 井戸内自己検証によるper-well脚選択の転移検証(CPU)
================================================================================
発想(user/ultimate-pfノートの層5): 既知帯(prefix)を50/65/75%でカットし、保留部分(真値既知)で
エンジン変種をバックテスト→一貫して勝つ変種だけをその井戸のeval区間に採用。
= 井戸レベル枝判別(v33/v47の壁)を「井戸内の正直な検証」で解く試み。

v80a実験(錨ファミリーのみ, PFなし, CPU):
  変種 = {K16錨, P5(dip03), P6(lp13 EMT0.15), 等重み(錨+P5+P6)/3}
  カット版エンジン: datum=カット行のTVT_input, K16場/κ/σは既存(自坑井除外)。
  eval側の各脚予測は既存773井pkl(フル prefix版)を使用=選択のみをこの実験の変数にする。
  判定: 固定各脚 / 固定等重み / backtest選択 / evalオラクル選択 の773井pooled比較。
使い方: WSTRIDE=5 python v80/prefix_gate_v80.py  (全773はWSTRIDE=1)
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn.functional as Fnn
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
DEV = "cpu"

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
A77 = P0 / "v77" / "artifacts_v77"
ART = P0 / "v80" / "artifacts_v80"; ART.mkdir(parents=True, exist_ok=True)
WSTRIDE = int(os.environ.get("WSTRIDE", "5"))
CUTS = tuple(float(x) for x in os.environ.get("CUTS", "0.5,0.65,0.75").split(","))
MIN_GAIN = float(os.environ.get("MIN_GAIN", "0.5"))   # ft: 保留区間RMSEでの必要勝ち幅
MIN_HELD = int(os.environ.get("MIN_HELD", "150"))     # 保留区間の最低行数
CONSIST = int(os.environ.get("CONSIST", "2"))         # 必要勝ちカット数(/有効カット)
POW = 1.7826896851395453
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)

# K16場・κ・σ(dd)は本錨と同一(kappa_2再現値)
THETA0 = 118.4; K16 = 16; KNN_K, KNN_H = 15, 500.0; GATE = 0.35; FMP = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA = np.array([0.812, 0.722, 0.618, 0.372, 0.109])
SIGB = np.array([7.26, 9.7, 13.6, 13.72, 11.42])

ANC = pickle.loads((A77 / "k16_anchor_train.pkl").read_bytes())
P5D = pickle.loads((A77 / "p5_out_dip03_773w.pkl").read_bytes())
P6D = pickle.loads((A77 / "p6_out_lp13e015fth_773w.pkl").read_bytes())
BARE = pickle.loads((P0 / "v73" / "artifacts_v73" / "seed_dump_bare0.pkl").read_bytes())
LS = 10.536319667317654

# ---- 全train井の場(deprojected drift field)を構築(1回だけ, 12s級) ----
def _build_field():
    wells = []
    for fp in sorted(glob.glob(str(TRAIN / "*__horizontal_well.csv"))):
        w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
        tvt, ti = w["TVT"].to_numpy(), w["TVT_input"].to_numpy()
        fin = np.where(np.isfinite(ti))[0]
        if len(fin) < 10: continue
        s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
        if n < 30: continue
        U = np.cumsum(ndz); R0 = tvt[s + 1:] - tvt[s]
        t = np.arange(1, n + 1.0); edges = np.linspace(0, n, K16 + 1)
        phi = np.column_stack([np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(K16)])
        mf = np.isfinite(R0)
        c = np.linalg.lstsq(phi[mf], (R0 - U)[mf], rcond=None)[0]
        mid = np.empty((K16, 2)); proj = np.empty(K16)
        for j in range(K16):
            f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
            az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
            mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2)
            proj[j] = np.cos(np.radians(az - THETA0))
        wid = os.path.basename(fp).split("__")[0]
        wells.append((wid, c, mid, proj))
    widx = {w[0]: i for i, w in enumerate(wells)}
    fld = np.array([(w[2][j, 0], w[2][j, 1], w[1][j] / w[3][j], widx[w[0]])
                    for w in wells for j in range(K16) if abs(w[3][j]) > FMP])
    return fld, widx

FLD, WIDX = _build_field()
print(f"field {len(FLD)}点 / {len(WIDX)}井", flush=True)


def k16_over(X, Y, Z, s, n, wid, t0tvt):
    """区間 s+1..s+n の K16錨 tvt/sig を返す(datum=t0tvt at row s)"""
    edges = np.linspace(0, n, K16 + 1)
    segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K16 - 1)
    mid = np.empty((K16, 2)); proj = np.empty(K16)
    for j in range(K16):
        f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j + 1]) - 1, int(edges[j])), len(X) - 1)
        az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
        mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2)
        proj[j] = np.cos(np.radians(az - THETA0))
    own = WIDX.get(wid, -1)
    keep = FLD[:, 3] != own
    fx, fy, fD = FLD[keep, 0], FLD[keep, 1], FLD[keep, 2]
    Dh = np.empty(K16); dd = np.empty(K16)
    for j in range(K16):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        sel = np.argpartition(d2, KNN_K)[:KNN_K]
        w = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
        Dh[j] = np.sum(w * fD[sel]) / np.sum(w)
        dd[j] = np.sqrt(np.median(d2[sel]))
    ndz = -np.diff(Z)[s:s + n]
    g = ndz + (Dh * proj)[segid]
    g = np.where(np.abs(proj[segid]) < GATE, 0.0, g)
    kstep = KAPPA[np.digitize(dd, KBINS[1:-1])][segid]
    tvt = t0tvt + np.cumsum(kstep * g)
    sig = SIGB[np.digitize(dd[segid], KBINS[1:-1])]
    return tvt, np.clip(sig, 1.0, None)


def fb_leg(ge, A, sig, tw_tvt, tw_gr, gs, mode):
    """P6(lp13 emt015) / P5(dip03) のカット版(CPU)。mode='p6'/'p5'"""
    T = len(ge)
    if mode == "p6":
        gsm = pd.Series(ge).rolling(13, center=True, min_periods=1).mean().to_numpy()
        step = float(np.median(np.diff(tw_tvt)))
        wpt = max(3, int(round(6.0 / max(step, 1e-3))) | 1)
        twg = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same")
        emt, ancw, srw, ostep, band = 0.15, 0.01, 0.02, 0.5, 40.0
        gobs = gsm
    else:
        twg = tw_gr; emt, ancw, srw, ostep, band = 1.0, 0.02, 0.02, 1.0, 40.0
        gobs = ge
    offs = torch.arange(-band, band + ostep / 2, ostep); S = len(offs)
    At = torch.tensor(A, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, dtype=torch.float32); tgr = torch.tensor(twg, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    egr = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    grt = torch.tensor(gobs, dtype=torch.float32)
    loge = emt * (-0.5 * (((grt[:, None] - egr) / gs).abs()) ** POW)
    sd = torch.tensor(np.clip(sig, 1.0, None), dtype=torch.float32)
    loge = loge - 0.5 * ancw * (offs[None, :] / sd[:, None]) ** 2
    if mode == "p5":
        # 簡易dip版はコスト高のためv80aではP5もオフセット随歩FBで近似(dip慣性なし, SRW小さめ)
        srw = 0.01
    ksig = max(srw / ostep, 0.05)
    p_ = min(ksig ** 2, 0.5)
    ker = torch.tensor([p_ / 2, 1 - p_, p_ / 2], dtype=torch.float32).view(1, 1, -1)
    def blur(v):
        p = torch.exp(v - v.max())
        p = Fnn.conv1d(p.view(1, 1, -1), ker, padding=1).view(-1)
        return torch.log(p.clamp_min(1e-30))
    alpha = torch.zeros(T, S); beta = torch.zeros(T, S)
    alpha[0] = -0.5 * (offs / 3.0) ** 2 + loge[0]
    for t in range(1, T):
        alpha[t] = blur(alpha[t - 1]) + loge[t]; alpha[t] -= alpha[t].max()
    for t in range(T - 2, -1, -1):
        beta[t] = blur(beta[t + 1] + loge[t + 1]); beta[t] -= beta[t].max()
    post = torch.softmax(alpha + beta, dim=1)
    om = (post * offs[None, :]).sum(1).numpy()
    return A + om


def backtest_well(wid):
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 3 * MIN_HELD: return None                      # prefixが短い井戸はゲート対象外
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    X, Y, Z = hw["X"].to_numpy(float), hw["Y"].to_numpy(float), hw["Z"].to_numpy(float)
    kpos = hw.index.get_indexer(kn.index)
    kt = kn["TVT_input"].to_numpy(float)
    wins = {v: 0 for v in ("anc", "p5", "p6", "eq")}; valid_cuts = 0; rms_log = []
    for f in CUTS:
        ci = int(len(kpos) * f)
        held = kpos[ci + 1:]
        if len(held) < MIN_HELD: continue
        s = int(kpos[ci]); n = len(held)
        if not np.all(held == np.arange(s + 1, s + 1 + n)): continue   # 非連続prefixは対象外
        # σ較正: カット前prefixのみ
        pre = kpos[:ci + 1]
        resid = gr[pre] - np.interp(kt[:ci + 1], tw_tvt, tw_gr)
        gs = float(np.nanstd(resid))
        if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
        gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
        A, sig = k16_over(X, Y, Z, s, n, wid, float(kt[ci]))
        truth = kt[ci + 1:]
        ge = gr[held]
        preds = {"anc": A}
        preds["p6"] = fb_leg(ge, A, sig, tw_tvt, tw_gr, gs, "p6")
        preds["p5"] = fb_leg(ge, A, sig, tw_tvt, tw_gr, gs, "p5")
        preds["eq"] = (preds["anc"] + preds["p5"] + preds["p6"]) / 3
        rm = {v: float(np.sqrt(np.nanmean((p - truth) ** 2))) for v, p in preds.items()}
        valid_cuts += 1
        rms_log.append(rm)
        best = min(rm, key=rm.get)
        for v in rm:
            if rm[v] <= rm[best] + 1e-9 and (rm["eq"] - rm[v]) >= (MIN_GAIN if v != "eq" else -1):
                wins[v] += 1
    if valid_cuts == 0: return None
    # 選択: eq(既定)をMIN_GAIN以上でCONSIST回以上破った変種があれば乗り換え
    sel = "eq"
    cand = {v: w for v, w in wins.items() if v != "eq" and w >= min(CONSIST, valid_cuts)}
    if cand: sel = max(cand, key=cand.get)
    return sel, valid_cuts, rms_log


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = [w for w in wids if w in ANC and w in P5D and w in P6D and w in BARE][::max(WSTRIDE, 1)]
    t0 = time.time()
    sels = {}; details = {}
    for i, w in enumerate(wids):
        try:
            r = backtest_well(w)
        except Exception as ex:
            print(f"  {w}: ERROR {type(ex).__name__} {ex}"); r = None
        sels[w] = r[0] if r else "eq"
        details[w] = r[2] if r else None
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wids)} ({time.time()-t0:.0f}s)", flush=True)
    import collections
    print("選択分布:", dict(collections.Counter(sels.values())))
    # eval区間で評価(既存773 dumpの脚)
    def legs_of(wid):
        d = BARE[wid]; tv = d["true"]
        A = ANC[wid]["tvt"]
        e5 = np.asarray(P5D[wid]["pred"], np.float64); e6 = np.asarray(P6D[wid]["pred"], np.float64)
        T = min(len(tv), len(A), len(e5), len(e6))
        legs = {"anc": A[:T], "p5": e5[:T], "p6": e6[:T]}
        legs["eq"] = (legs["anc"] + legs["p5"] + legs["p6"]) / 3
        return legs, tv[:T]
    SOFTW = float(os.environ.get("SOFTW", "1.0"))   # 1.0=硬い置換 / <1=cap付きソフト混合(ultimate-pf流)
    tot = {k: [] for k in ("anc", "p5", "p6", "eq", "gate", "oracle")}
    for w in wids:
        legs, tv = legs_of(w)
        m = np.isfinite(tv)
        errs = {k: (v[m] - tv[m]) for k, v in legs.items()}
        for k in ("anc", "p5", "p6", "eq"): tot[k].append(errs[k])
        ge = errs["eq"] if sels[w] == "eq" else (1 - SOFTW) * errs["eq"] + SOFTW * errs[sels[w]]
        tot["gate"].append(ge)
        orc = min(errs, key=lambda k: np.sqrt(np.mean(errs[k] ** 2)))
        tot["oracle"].append(errs[orc])
    print(f"\n=== v80a prefix-backtestゲート ({len(wids)}井) CUTS={CUTS} MIN_GAIN={MIN_GAIN} ===")
    for k in ("anc", "p5", "p6", "eq", "gate", "oracle"):
        e = np.concatenate(tot[k])
        print(f"  {k:7s}: {np.sqrt(np.mean(e**2)):.4f}")
    (ART / f"sels_{len(wids)}w.pkl").write_bytes(pickle.dumps(dict(sels=sels, details=details), protocol=4))
    # ★診断: prefix勝者はeval勝者を予測するか
    agree = 0; n_d = 0; regret = []
    for w in wids:
        if details.get(w) is None: continue
        legs, tv = legs_of(w)
        m = np.isfinite(tv)
        ev_rm = {k: float(np.sqrt(np.mean((v[m] - tv[m]) ** 2))) for k, v in legs.items()}
        ev_best = min(ev_rm, key=ev_rm.get)
        bt_sum = {}
        for rm in details[w]:
            for k, v in rm.items(): bt_sum[k] = bt_sum.get(k, 0) + v
        bt_best = min(bt_sum, key=bt_sum.get)
        n_d += 1; agree += int(bt_best == ev_best)
        regret.append(ev_rm[bt_best] - ev_rm[ev_best])
    if n_d:
        print(f"[診断] backtest勝者==eval勝者: {agree}/{n_d} = {agree/n_d*100:.0f}% (チャンス25%)")
        print(f"[診断] backtest勝者採用時のeval後悔: 平均{np.mean(regret):.3f}ft 中央{np.median(regret):.3f}ft")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
