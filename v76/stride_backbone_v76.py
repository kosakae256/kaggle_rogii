# -*- coding: utf-8 -*-
"""
v76 Phase B — STRIDE型バックボーン: (h_len, ΔS) 区間格子ビームデコーダ(struct-free)
====================================================================================
doc2(STRIDE, 現5位)の1.1を我々の枠で実装。全て坑井内情報のみ(struct不使用)。
- 状態: S=TVT+Z(構造面)。区間=(h_len行, ΔS)。開始 S0=last_known TVT+Z, 初期傾き=prefix末尾dip。
- prior: 区間長 log-normal / 傾き persistence(前区間に近い) / near-zero傾き / (全てログ加算)
- 尤度: Cauchy −log(1+(d/γ)²)、d=(GR−tw(TVT))/σ_well(σはprefix残差較正=r0と同一)
- typewell参照項: 既知区間残差中央値でGRレベル再較正(TWREF, STRIDEで−1.26ft相当)
- 復号: ビームサーチ(B本)。出力=ビーム重み付き事後平均(soft) と最良経路(hard)。
評価: pooled RMSE / per-well / (別途)素のPFとの誤差相関→ブレンド。
使い方: NW=60 python v76/stride_backbone_v76.py  (スモーク) / NW=10000 (全773井)
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v76" / "artifacts_v76"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NW = int(os.environ.get("NW", "60"))
BEAM = int(os.environ.get("BEAM", "64"))
HLENS = [int(x) for x in os.environ.get("HLENS", "64,128,256,512").split(",")]
NDS = int(os.environ.get("NDS", "17"))            # ΔS格子の本数(奇数)
DSMAX = float(os.environ.get("DSMAX", "10"))      # h=256行あたりのΔS最大(ft)。hに√比例スケール
GAMMA = float(os.environ.get("GAMMA", "1.0"))     # Cauchyスケール(dは既にσ_wellで正規化済)
W_LEN = float(os.environ.get("W_LEN", "1.0"))     # 区間長prior重み
W_PERS = float(os.environ.get("W_PERS", "20.0"))  # persistence(傾き差)重み: pen=W·(slope−slope_prev)²
W_ZERO = float(os.environ.get("W_ZERO", "1.0"))   # near-zero傾きprior重み: pen=W·slope²
SIGDIV = float(os.environ.get("SIGDIV", "3.0"))   # σ分割(PF用gr_sig_mult≈3.06を打ち消し尤度を先鋭化)
W_ANC = float(os.environ.get("W_ANC", "1.0"))     # datum prior重み: pen=W·((S−(S0+d0·md))/σ(md))², 区間端で評価
ANC_S0 = float(os.environ.get("ANC_S0", "12.0"))  # datum priorのσ切片(ft)
ANC_SR = float(os.environ.get("ANC_SR", "0.008")) # σ成長率(ft/行) — 距離とともに緩む(STRIDE trellis流)
LEN_MU = float(os.environ.get("LEN_MU", "5.5"))   # log-normal μ(log行数)。exp(5.5)≈245行
LEN_SG = float(os.environ.get("LEN_SG", "0.9"))
BTEMP = float(os.environ.get("BTEMP", "30.0"))    # ビーム事後平均のsoftmax温度(※TEMPはWindows予約変数のためBTEMP)
TWREFB = os.environ.get("TWREFB", "1")            # typewell参照項(STRIDE −1.26ft相当)
SUBR = int(os.environ.get("SUBR", "2"))           # 尤度評価の行間引き
# r0のσ較正定数(nnpf_v73と同一)
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)


def load_well(wid):
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    if "TVT" not in hw.columns: return None
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    fb = float(np.nanmean(tw_gr))
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    kt = kn["TVT_input"].to_numpy(float)
    resid = gr[kpos] - np.interp(kt, tw_tvt, tw_gr)
    gs = float(np.nanstd(resid))
    if not np.isfinite(gs) or gs <= 0: gs = GRSIG["df"]
    gs = float(np.clip(gs * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"])) / SIGDIV
    ge = gr[epos].copy()
    if TWREFB == "1":
        b_ = float(np.nanmedian(resid))
        if np.isfinite(b_): ge = ge - b_
    md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
    S0 = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    mm = dm > 0
    d0 = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    tv = ev["TVT"].to_numpy(float)
    return dict(wid=wid, md=md, z=z, gr=ge, gs=gs, S0=S0, d0=np.clip(d0, -0.05, 0.05),
                tw_tvt=tw_tvt, tw_gr=tw_gr, true=tv)


def decode_well(c):
    """(h_len,ΔS)区間ビーム。返り値: soft(事後平均TVT), hard(最良経路TVT)。"""
    md, z, gr = c["md"], c["z"], c["gr"]
    T = len(md)
    ttv = torch.tensor(c["tw_tvt"], device=DEV, dtype=torch.float32)
    tgr = torch.tensor(c["tw_gr"], device=DEV, dtype=torch.float32)
    zt = torch.tensor(z, device=DEV, dtype=torch.float32)
    grt = torch.tensor(gr, device=DEV, dtype=torch.float32)
    gs = float(c["gs"])
    # ビーム状態: S(現在位置), slope_prev, score, 経路S列(行ごと, 遅延構築でなく全行を保持)
    Spath = torch.full((1, T), float("nan"), device=DEV)
    Scur = torch.tensor([c["S0"]], device=DEV); dprev = torch.tensor([c["d0"]], device=DEV)
    score = torch.tensor([0.0], device=DEV); pos = torch.zeros(1, dtype=torch.long, device=DEV)
    HL = torch.tensor(HLENS, device=DEV, dtype=torch.long)
    while True:
        active = pos < T
        if not bool(active.any()): break
        B = Scur.shape[0]
        # 展開: (B, H, D) — h_len×ΔS格子
        rem = (T - pos).clamp(min=1)
        h_eff = torch.minimum(HL[None, :], rem[:, None])                     # (B,H)
        dsg = torch.linspace(-1, 1, NDS, device=DEV) * DSMAX                  # 基本±DSMAX@256行
        scale = (h_eff.float() / 256.0).sqrt()                                # (B,H)
        dS = dsg[None, None, :] * scale[:, :, None]                           # (B,H,D)
        slope = dS / h_eff[:, :, None].float()                                # ft/行(≈ft/ft)
        # priors(ログ罰)
        lp_len = W_LEN * ((torch.log(h_eff.float()) - LEN_MU) ** 2) / (2 * LEN_SG ** 2)   # (B,H)
        lp_pers = W_PERS * (slope - dprev[:, None, None]) ** 2 * 1e4          # slopeスケール補正
        lp_zero = W_ZERO * slope ** 2 * 1e4
        # 尤度: 各候補区間の行ごとの d をCauchy和(SUBR間引き)
        hmax = int(h_eff.max())
        rr = torch.arange(0, hmax, SUBR, device=DEV)                          # 間引き行
        # 行index (B,H,R)
        ridx = pos[:, None, None] + rr[None, None, :]
        valid = rr[None, None, :] < h_eff[:, :, None]
        ridx = ridx.clamp(max=T - 1)
        frac = (rr[None, None, :].float() + 1) / h_eff[:, :, None].float()    # 区間内位置0..1
        Sr = Scur[:, None, None] + dS[:, :, :, None].permute(0, 1, 3, 2) if False else None
        # S(row) = Scur + slope*(row+1)  → TVT = S − z
        Srow = Scur[:, None, None, None] + slope[:, :, :, None] * (rr[None, None, None, :].float() + 1)
        zrow = zt[ridx][:, :, None, :]                                        # (B,H,1,R)
        tvtr = Srow - zrow                                                    # (B,H,D,R)
        # typewell補間(searchsorted)
        ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous())
        ii = ii.clamp(1, len(ttv) - 1)
        t0 = ttv[ii - 1]; t1 = ttv[ii]; g0 = tgr[ii - 1]; g1 = tgr[ii]
        w_ = ((tvtr - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0, 1)
        egr = g0 * (1 - w_) + g1 * w_
        d = (grt[ridx][:, :, None, :] - egr) / gs
        clh = torch.log1p((d / GAMMA) ** 2)                                   # Cauchy −loglik
        clh = torch.where(valid[:, :, None, :], clh, torch.zeros_like(clh))
        lk = clh.sum(-1) * SUBR                                               # (B,H,D) 行数スケール戻し
        # datum prior(区間端で評価): 中心=S0+d0·行, σ=ANC_S0+ANC_SR·行(遠いほど緩い)
        pos_end = (pos[:, None] + h_eff).float()                              # (B,H)
        S_end = Scur[:, None, None] + dS                                      # (B,H,D)
        anc_c = torch.tensor(c["S0"], device=DEV) + torch.tensor(c["d0"], device=DEV) * pos_end
        anc_sg = ANC_S0 + ANC_SR * pos_end
        lp_anc = W_ANC * ((S_end - anc_c[:, :, None]) / anc_sg[:, :, None]) ** 2
        tot = score[:, None, None] + lk + lp_len[:, :, None] + lp_pers + lp_zero + lp_anc
        # 非activeビームは現状維持(拡張しない)
        tot = torch.where(active[:, None, None], tot, torch.full_like(tot, float("inf")))
        flat = tot.reshape(-1)
        k = min(BEAM, int((flat < float("inf")).sum()))
        if k == 0: break
        v_, idx = torch.topk(-flat, k)
        bi = idx // (len(HLENS) * NDS); hi = (idx % (len(HLENS) * NDS)) // NDS; di = idx % NDS
        # 新ビーム構築
        h_new = h_eff[bi, hi]; dS_new = dS[bi, hi, di]; slope_new = slope[bi, hi, di]
        newSpath = Spath[bi].clone()
        for j in range(k):                                                    # 経路書き込み(kは小さい)
            p0 = int(pos[bi[j]]); h_ = int(h_new[j])
            rows = torch.arange(1, h_ + 1, device=DEV, dtype=torch.float32)
            newSpath[j, p0:p0 + h_] = Scur[bi[j]] + slope_new[j] * rows
        keep_inactive = (~active).nonzero().flatten()
        if len(keep_inactive):                                                # 完走ビームも温存
            newSpath = torch.cat([newSpath, Spath[keep_inactive]], 0)
            Scur = torch.cat([Scur[bi] + dS_new, Scur[keep_inactive]], 0)
            dprev = torch.cat([slope_new, dprev[keep_inactive]], 0)
            score = torch.cat([-v_, score[keep_inactive]], 0)
            pos = torch.cat([pos[bi] + h_new, pos[keep_inactive]], 0)
        else:
            Scur = Scur[bi] + dS_new; dprev = slope_new; score = -v_; pos = pos[bi] + h_new
        Spath = newSpath
        # ビーム数制限(完走含めTop-BEAM)
        if Spath.shape[0] > BEAM:
            norm = score / pos.clamp(min=1).float()
            _, kb = torch.topk(-norm, BEAM)
            Spath = Spath[kb]; Scur = Scur[kb]; dprev = dprev[kb]; score = score[kb]; pos = pos[kb]
    # 完走ビームで合成
    done = pos >= T
    if not bool(done.any()): done = torch.ones_like(pos, dtype=torch.bool)
    sp = Spath[done]; sc = score[done] / T
    ww = torch.softmax(-sc / (BTEMP / T), 0) if len(sc) > 1 else torch.ones(1, device=DEV)
    soft = (ww[:, None] * sp).nansum(0) - zt
    hard = sp[int(torch.argmin(sc))] - zt
    return soft.cpu().numpy(), hard.cpu().numpy()


def main():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NW]
    es_s, es_h = [], []
    per = {}
    t0 = time.time()
    outs = {}
    for i, w in enumerate(wids):
        c = load_well(w)
        if c is None: continue
        try:
            soft, hard = decode_well(c)
        except Exception as e:
            print(f"  {w}: ERROR {type(e).__name__} {e}"); continue
        tvv = c["true"]; m = np.isfinite(tvv) & np.isfinite(soft)
        if m.sum() == 0: continue
        es_s.append(soft[m] - tvv[m]); es_h.append(hard[m] - tvv[m])
        per[w] = float(np.sqrt(np.mean((soft[m] - tvv[m]) ** 2)))
        outs[w] = dict(soft=soft.astype(np.float32), hard=hard.astype(np.float32))
        if (i + 1) % 50 == 0:
            e = np.concatenate(es_s)
            print(f"  {i+1}/{len(wids)}  暫定 soft={np.sqrt(np.mean(e**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e_s = np.concatenate(es_s); e_h = np.concatenate(es_h)
    print(f"\n=== STRIDE backbone v1 ({len(per)}井) ===")
    print(f"  soft(ビーム事後平均) pooled = {np.sqrt(np.mean(e_s**2)):.4f}")
    print(f"  hard(最良経路)      pooled = {np.sqrt(np.mean(e_h**2)):.4f}")
    print(f"  per-well 中央 {np.median(list(per.values())):.3f}  (彼のbackbone単体: 9.99)")
    fp = ART / f"backbone_out_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
