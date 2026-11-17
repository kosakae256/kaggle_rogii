# -*- coding: utf-8 -*-
"""
v73 段b — emission(学習照合)の強化ハーネス
==========================================
v50 stageA(capaug=CH128/DIM96/DILS..32/AUG)を土台に、方式レバーをトグルして再学習:
  - SELFCH=1: typewell側encBに self-prefix曲線+被覆mask の2chを追加(7→9ch)。★self活用の本命(積でなくNN融合)
  - HW/HM/TOPK: ゴーストhinge強化(重み/マージン/far側top-k平均)。±1サイクルおとりを直接叩く
  - FOLD_ONLY=0: fold0だけ学習し val 指標で方式比較(安い)。勝者を全5fold→sim表→PF A/B
出力: artifacts_v73/emis_{TAG73}_f{f}.pt / (BUILD_SIM=1で) sim_rows_v73_{TAG73}.pkl (v51形式={well:{sim,st}})
消費: nnpf_v73.py SIMFP=...pkl
使い方例:
  FOLD_ONLY=0 SELFCH=1 TAG73=selfch python v73/emission_v73.py
  TAG73=selfch SELFCH=1 BUILD_SIM=1 python v73/emission_v73.py   (5fold学習→sim表)
"""
import os, io, sys, time, math, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(P0 / "v50"))
os.environ.setdefault("TAG", "v73"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("AUG", "1")
import stageA_v50 as sa                                    # feats/Enc1D/augment/cache を再利用

ART = P0 / "v73" / "artifacts_v73"; ART.mkdir(exist_ok=True)
DEV = sa.DEV; STEP = sa.STEP; NB = sa.NB
TAG73 = os.environ.get("TAG73", "selfch")
SELFCH = int(os.environ.get("SELFCH", "1"))
SELFONLY = int(os.environ.get("SELFONLY", "0"))  # 1=段c-1: encB=self-prefix曲線のみ(2ch)の独立self照合器
HW = float(os.environ.get("HW", "0.5"))          # hinge重み(v50は0.5)
HM = float(os.environ.get("HM", "0.10"))         # hingeマージン(v50は0.10)
TOPK = int(os.environ.get("TOPK", "1"))          # far側 top-k 平均(1=v50同等)
EPOCHS = int(os.environ.get("EPOCHS", "25"))
RPW = sa.RPW; NEAR_B = sa.NEAR_B; FAR_B = sa.FAR_B
FOLD_ONLY = os.environ.get("FOLD_ONLY", "")      # ""=全fold / "0"等=そのfoldのみ
BUILD_SIM = int(os.environ.get("BUILD_SIM", "0"))
LR = sa.LR; SEED = sa.SEED


def _bin_median(tvt, gr, gmin, K):
    out = np.full(K, np.nan)
    idx = np.round((np.asarray(tvt, float) - gmin) / STEP).astype(int)
    ok = (idx >= 0) & (idx < K) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        srt = pd.Series(np.asarray(gr, float)[ok]).groupby(idx[ok]).median()
        out[srt.index.to_numpy()] = srt.to_numpy()
    return out


def self_channels(c):
    """self-prefix曲線(robust-z, 無被覆=0) + 被覆mask の (K,2)。test合法(prefixのみ使用)。"""
    K = len(c["gg"]); gmin = c["gmin"]
    pre = ~c["ev"]
    gh = _bin_median(c["y"][pre], c["graw"][pre], gmin, K)
    m = np.isfinite(gh)
    z = np.zeros(K, np.float32)
    if m.sum() >= 5:
        z[m] = np.clip(sa._rz(gh[m]), -8, 8)
    return np.stack([z, m.astype(np.float32)], 1)


def ft_of(c):
    if SELFONLY:
        return self_channels(c).astype(np.float32)      # (K,2): self曲線rz + 被覆mask
    Ft = c["Ft"]
    if SELFCH:
        Ft = np.concatenate([Ft, self_channels(c).astype(np.float32)], 1)
    return Ft


def train_fold(data, f):
    wells = sorted(data.keys())
    val = wells[f::5]; vs = set(val); trn = [w for w in wells if w not in vs]
    cinB = 2 if SELFONLY else 7 + (2 if SELFCH else 0)
    print(f"[{TAG73} f{f}] train{len(trn)}/val{len(val)} encB cin={cinB} HW={HW} HM={HM} TOPK={TOPK} ep{EPOCHS}", flush=True)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED + f)
    encA = sa.Enc1D(8, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV)
    encB = sa.Enc1D(cinB, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV)
    logit_s = nn.Parameter(torch.tensor(2.3, device=DEV))
    opt = torch.optim.AdamW(list(encA.parameters()) + list(encB.parameters()) + [logit_s], lr=LR, weight_decay=1e-4)
    offs = torch.arange(-NB, NB + 1, device=DEV)
    tgt0 = torch.exp(-0.5 * (offs.float() / 2.0) ** 2); tgt0 = tgt0 / tgt0.sum()
    far_m = (offs.abs() >= FAR_B)
    t0 = time.time()
    FTs = {w: ft_of(data[w]) for w in wells}                # self ch含む typewell特徴(キャッシュ)
    # SELFONLY: 被覆mask(格子bool)。訓練行は真bin±4が被覆済みの行のみ/候補softmaxは被覆外を-1e4
    COVs = {w: (FTs[w][:, 1] > 0) if SELFONLY else np.ones(len(data[w]["gg"]), bool) for w in wells}
    def _cov_rows(kf, lab, covb):
        if not SELFONLY or len(lab) == 0: return lab
        kk = np.round(kf[lab]).astype(int)
        ok = np.array([covb[max(k - 4, 0): k + 5].all() for k in kk], dtype=bool)
        return lab[ok]
    for ep in range(EPOCHS):
        for pg in opt.param_groups:
            pg["lr"] = LR * 0.5 * (1.0 + math.cos(math.pi * ep / max(EPOCHS - 1, 1)))
        encA.train(); encB.train(); losses = []
        for wi in rng.permutation(len(trn)):
            c = data[trn[wi]]
            y = c["y"]; gmin = c["gmin"]; K = len(c["gg"])
            kf = (y - gmin) / STEP
            lab = np.where(np.isfinite(kf) & (kf >= NB) & (kf <= K - 1 - NB))[0]
            lab = _cov_rows(kf, lab, COVs[trn[wi]])
            if len(lab) < 32: continue
            rows = rng.choice(lab, size=min(RPW, len(lab)), replace=False)
            kt = torch.tensor(np.round(kf[rows]).astype(np.int64), device=DEV)
            xh = torch.from_numpy(c["Fh"]).to(DEV)
            if sa.AUG: xh = sa.augment_h(xh)
            a = encA(xh)[rows]
            b = encB(torch.from_numpy(FTs[trn[wi]]).to(DEV))
            cand = kt[:, None] + offs[None, :]
            sims = (a[:, None, :] * b[cand]).sum(-1)
            if SELFONLY:                                        # 被覆外候補はsoftmax/hingeから除外
                covt = torch.from_numpy(COVs[trn[wi]]).to(DEV)
                candcov = covt[cand]
                sims = sims.masked_fill(~candcov, -1e4)
            logits = sims * logit_s.exp().clamp(1.0, 100.0)
            loss = -(tgt0[None, :] * F.log_softmax(logits, dim=1)).sum(1).mean()
            near = sims[:, NB - NEAR_B: NB + NEAR_B + 1].max(1).values
            farv = sims.masked_fill(~far_m[None, :], -1e4)
            if TOPK > 1:
                far = farv.topk(TOPK, dim=1).values.mean(1)
            else:
                far = farv.max(1).values
            loss = loss + HW * F.relu(far - near + HM).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encA.parameters()) + list(encB.parameters()), 1.0)
            opt.step(); losses.append(float(loss))
        if ep % 4 == 3 or ep == EPOCHS - 1:
            print(f"  ep{ep+1}/{EPOCHS} loss={np.mean(losses):.4f} scale={float(logit_s.exp()):.1f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(encA=encA.state_dict(), encB=encB.state_dict(), logit_s=float(logit_s), cinB=cinB),
               ART / f"emis_{TAG73}_f{f}.pt")
    # ---- val 方式比較指標: ranking acc(near>far) / band内 top1 ±3ft ----
    encA.eval(); encB.eval()
    acc_rk = []; acc_t1 = []
    with torch.no_grad():
        for w in val:
            c = data[w]
            y = c["y"]; gmin = c["gmin"]; K = len(c["gg"])
            kf = (y - gmin) / STEP
            lab = np.where(np.isfinite(kf) & (kf >= NB) & (kf <= K - 1 - NB) & c["ev"])[0]
            lab = _cov_rows(kf, lab, COVs[w])
            if len(lab) < 20: continue
            rows = lab[:: max(1, len(lab) // 200)]
            kt = torch.tensor(np.round(kf[rows]).astype(np.int64), device=DEV)
            a = encA(torch.from_numpy(c["Fh"]).to(DEV))[rows]
            b = encB(torch.from_numpy(FTs[w]).to(DEV))
            sims = (a[:, None, :] * b[kt[:, None] + offs[None, :]]).sum(-1)
            if SELFONLY:
                covt = torch.from_numpy(COVs[w]).to(DEV)
                sims = sims.masked_fill(~covt[kt[:, None] + offs[None, :]], -1e4)
            near = sims[:, NB - NEAR_B: NB + NEAR_B + 1].max(1).values
            far = sims.masked_fill(~far_m[None, :], -1e4).max(1).values
            acc_rk.append((near > far).float().mean().item())
            t1 = sims.argmax(1)
            acc_t1.append(((t1 - NB).abs() <= 6).float().mean().item())
    print(f"[{TAG73} f{f}] val ranking(near>far)={np.mean(acc_rk):.4f}  top1±3ft={np.mean(acc_t1):.4f}", flush=True)
    return np.mean(acc_rk), np.mean(acc_t1)


def build_sim(data):
    """5fold OOFエンコーダで v51形式 sim表(band=struct±45ft)を構築。"""
    wells = sorted(data.keys()); offs = np.arange(-NB, NB + 1)
    out = {}
    with torch.no_grad():
        for f in range(5):
            ck = torch.load(ART / f"emis_{TAG73}_f{f}.pt", map_location=DEV, weights_only=False)
            cinB = ck.get("cinB", 7 + (2 if SELFCH else 0))
            encA = sa.Enc1D(8, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encA.load_state_dict(ck["encA"]); encA.eval()
            encB = sa.Enc1D(cinB, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encB.load_state_dict(ck["encB"]); encB.eval()
            for wn in wells[f::5]:
                c = data[wn]
                ev_idx = np.where(c["ev"])[0]
                a = encA(torch.from_numpy(c["Fh"]).to(DEV))[ev_idx]
                ftc = ft_of(c)
                b = encB(torch.from_numpy(ftc).to(DEV))
                K = len(c["gg"]); gmin = c["gmin"]
                st = c["st"].astype(np.float64)
                ks = np.round((st - gmin) / STEP).astype(np.int64)
                cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
                okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
                sims = (a[:, None, :] * b[torch.tensor(cand, device=DEV)]).sum(-1).cpu().numpy()
                if SELFONLY:                                    # 被覆外=0(中立)。帯外も0
                    covb = ftc[:, 1] > 0
                    sims = np.where(okc & covb[cand], sims, 0.0)
                else:
                    sims = np.where(okc, sims, -1.0)
                out[wn] = dict(sim=sims.astype(np.float16), st=st.astype(np.float32))
            print(f"  sim fold{f} done", flush=True)
    fp = ART / f"sim_rows_v73_{TAG73}.pkl"
    fp.write_bytes(pickle.dumps(out, protocol=4))
    print(f"[saved] {fp.name} ({len(out)}井)")


if __name__ == "__main__":
    data = pickle.loads(sa.CACHE.read_bytes()) if sa.CACHE.exists() else sa.build_cache()
    if FOLD_ONLY != "":
        train_fold(data, int(FOLD_ONLY))
    else:
        for f in range(5):
            if not (ART / f"emis_{TAG73}_f{f}.pt").exists():
                train_fold(data, f)
        if BUILD_SIM:
            build_sim(data)
