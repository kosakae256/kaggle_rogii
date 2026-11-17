# -*- coding: utf-8 -*-
"""v93 module2: NN-emission sim 生成(★帯中心=GRフリー錨。過去struct不使用)。

v50/v51 の学習emission(TCN cosine類似度)を、候補帯の中心だけ struct→GRフリー錨tvt に差替。
  - encoder(stageA_enccapaug_f{f}.pt)は凍結再利用(v93/artifacts_v93/nn50 に複製済)。
  - OOF fold: sorted train井の strided 5-fold(wells[f::5])。fold f の井は fold f を hold-out した
    encoder f を使う=OOF-safe(v50/v51と同一分割)。Fh/Ft は encoder が期待する入力なので
    v50 cacheA と数値一致を assert してから使用。
  - 出力: v93/artifacts_v93/sim_grfree_v93.pkl = {wid: {sim(T,181)fp16, st(T,)=GRフリー錨tvt}}。
create 側は各PF入力dictに _sim/_st を添付して smoother に渡す(module1 _pad が消費)。
"""
import os, sys, io, glob, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJ / "rogii-wellbore-geology-prediction"
ART = PROJ / "v93" / "artifacts_v93"
NN50 = ART / "nn50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIM_STEP = 0.5
NBAND = 90                        # ±45ft / 0.5ft

# ---- split 対応(env NN_SPLIT) ----
#   train: 錨=grfree_anchor_train.pkl / OOF fold(wells[f::5]) の encoder f を使用。
#   test : 錨=env V93_ANCHOR_PKL(=GRフリー test 錨, module1 と同一 pkl)。
#          test 井はどの学習fold にも入っていないため encoder は 5本を平均する(OOF ではなく全平均)。
NN_SPLIT = os.environ.get("NN_SPLIT", "train")
if NN_SPLIT == "test":
    _ANCHOR_PKL = Path(os.environ["V93_ANCHOR_PKL"])   # test 錨(必須)
    SIM_FP = ART / "sim_grfree_test_v93.pkl"
else:
    _ANCHOR_PKL = ART / "grfree_anchor_train.pkl"
    SIM_FP = ART / "sim_grfree_v93.pkl"
_ANCHOR = pickle.loads(_ANCHOR_PKL.read_bytes())        # {wid:{tvt,sig}}


# ---- GR特徴(v50/v52 と同一。encoder入力) ----
def _rz(v):
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip((v - med) / max(iqr / 1.349, 1e-6), -8, 8)


def _rm(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()


def _rs(v, w):
    return pd.Series(v).rolling(w, center=True, min_periods=2).std().fillna(0).to_numpy()


def _detr(v, w=301):
    m = _rm(v, w); sd = _rs(v, w)
    return np.clip((v - m) / np.maximum(sd, 1e-3), -6, 6)


def feats_h(gr, md, z):
    g = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    g = np.where(np.isfinite(g), g, np.nanmedian(g)); g = np.where(np.isfinite(g), g, 0.0)
    z0 = _rz(g); s5, s21, s81 = _rm(z0, 5), _rm(z0, 21), _rm(z0, 81)
    d5 = np.gradient(s5) * 10.0; v21 = _rs(z0, 21); dt = _detr(z0, 301)
    dmd = np.maximum(np.gradient(md), 1e-3)
    dzdm = np.clip(np.gradient(z) / dmd, -0.2, 0.2) * 5.0
    X = np.stack([z0, s5, s21, s81, d5, v21, dt, dzdm], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def feats_t(gg):
    z0 = _rz(gg); s3, s9, s33 = _rm(z0, 3), _rm(z0, 9), _rm(z0, 33)
    d3 = np.gradient(s3) * 10.0; v9 = _rs(z0, 9); dt = _detr(z0, 301)
    X = np.stack([z0, s3, s9, s33, d3, v9, dt], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---- encoder(capaug: ch=128, dils 6段。凍結ckpt に一致) ----
class Enc1D(nn.Module):
    def __init__(self, cin, ch=128, dout=96, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = nn.ModuleList()
        for d in dils:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), nn.GELU(), nn.Conv1d(ch, ch, 1)))
        self.norm = nn.ModuleList([nn.GroupNorm(8, ch) for _ in dils])
        self.out = nn.Conv1d(ch, dout, 1)

    def forward(self, x):
        h = self.inp(x.t()[None])
        for blk, nm in zip(self.blocks, self.norm):
            h = nm(h + blk(h))
        return F.normalize(self.out(h)[0].t(), dim=1)


def _load_encoders():
    encs = []
    for f in range(5):
        ck = torch.load(NN50 / f"stageA_enccapaug_f{f}.pt", map_location=DEVICE)
        eA = Enc1D(8).to(DEVICE); eA.load_state_dict(ck["encA"]); eA.eval()
        eB = Enc1D(7).to(DEVICE); eB.load_state_dict(ck["encB"]); eB.eval()
        encs.append((eA, eB))
    return encs


def _well_gg(tw):
    """typewell を 0.5ft TVT格子へ(sim帯の座標系)。返り (gg, gmin) or None。"""
    tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
    m = np.isfinite(tt) & np.isfinite(tg); tt, tg = tt[m], tg[m]
    if len(tt) < 8:
        return None
    o = np.argsort(tt); tt, tg = tt[o], tg[o]
    gmin = float(tt.min())
    gg = np.interp(np.arange(gmin, float(tt.max()) + SIM_STEP, SIM_STEP), tt, tg)
    return gg, gmin


def build_sims(verify_cache=True, split=None):
    """split井の sim を GRフリー錨帯中心で生成。
       train: OOF fold=wells[f::5](井index i の fold=i%5 の encoder f)。
       test : どの学習fold にも入らないため encoder 5本を平均(全平均)。錨=_ANCHOR(=test錨)。"""
    split = split or NN_SPLIT
    is_test = (split == "test")
    TR = DATA_DIR / split
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    encs = _load_encoders()
    offs = np.arange(-NBAND, NBAND + 1)
    # 検証用 cacheA(Fh/Ft の厳密一致確認, train のみ)。無ければskip
    cache = None
    if verify_cache and not is_test:
        cp = PROJ / "v50" / "artifacts_v50" / "cacheA_v50.pkl"
        if cp.exists():
            cache = pickle.loads(cp.read_bytes())
    out = {}; nver = 0; maxdev = 0.0
    print(f"[NN-emission] split={split} wells={len(wells)} 錨={_ANCHOR_PKL.name}({len(_ANCHOR)}井) "
          f"encoder={'全5平均' if is_test else 'OOF fold'} -> {SIM_FP.name}", flush=True)
    with torch.no_grad():
        for i, wn in enumerate(wells):
            enc_use = encs if is_test else [encs[i % 5]]   # test=全5, train=fold i%5 のみ
            sp = _ANCHOR.get(wn)
            hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv")
            tw = pd.read_csv(TR / f"{wn}__typewell.csv")
            evm = hw["TVT_input"].isna().to_numpy()
            if sp is None or len(sp["tvt"]) != int(evm.sum()):
                continue                                # 錨欠損/行数不一致 → sim無し(注入OFF)
            gg_gmin = _well_gg(tw)
            if gg_gmin is None:
                continue
            gg, gmin = gg_gmin; K = len(gg)
            Fh = feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
            Ft = feats_t(gg)
            if cache is not None and wn in cache and nver < 30:
                cFh = np.asarray(cache[wn]["Fh"]); cFt = np.asarray(cache[wn]["Ft"])
                if cFh.shape == Fh.shape and cFt.shape == Ft.shape:
                    dev = max(float(np.abs(cFh - Fh).max()), float(np.abs(cFt - Ft).max()))
                    maxdev = max(maxdev, dev); nver += 1
            ev_idx = np.where(evm)[0]
            st = np.asarray(sp["tvt"], np.float64)      # ★帯中心=GRフリー錨tvt(struct不使用)
            ks = np.round((st - gmin) / SIM_STEP).astype(np.int64)
            cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
            okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
            xh = torch.from_numpy(Fh).to(DEVICE); xt = torch.from_numpy(Ft).to(DEVICE)
            ct = torch.tensor(cand, device=DEVICE)
            sims_acc = None
            for (eA, eB) in enc_use:                     # test は 5本の類似度を平均
                a = eA(xh)[ev_idx]; b = eB(xt)
                s_e = (a[:, None, :] * b[ct]).sum(-1)
                sims_acc = s_e if sims_acc is None else (sims_acc + s_e)
            sims = (sims_acc / len(enc_use)).cpu().numpy()
            sims = np.where(okc, sims, -1.0)
            out[wn] = dict(sim=sims.astype(np.float16), st=st.astype(np.float32))
            if (i + 1) % 100 == 0:
                print(f"  sim {i+1}/{len(wells)} (ready {len(out)})", flush=True)
    if nver:
        print(f"[検証] Fh/Ft vs cacheA 最大乖離={maxdev:.2e} ({nver}井) -> 0付近ならencoder入力一致OK")
    SIM_FP.write_bytes(pickle.dumps(out, protocol=4))
    print(f"sim saved: {len(out)}/{len(wells)} wells -> {SIM_FP.name}")
    return out


if __name__ == "__main__":
    build_sims(verify_cache=(NN_SPLIT != "test"))
