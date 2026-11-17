# -*- coding: utf-8 -*-
"""
v77 P15 — 錨帯マッチ画像の2D CNN(UNet-lite)による大域オフセット直接回帰(非逐次・学習)
=====================================================================================
入力画像 (C=4, T, S=81):
  ch0=生GR照合(-NLL, 行内max正規化) ch1=低域GR照合(GRMA13) ch2=錨prior(-(o/σdd)²/2) ch3=off/40定数場
出力: 行ごと81binロジット → soft-argmax(期待値) = オフセット → TVT = A + off。
学習: 井戸5fold(ソート順%5)。crop=連続768行。soft-CE(σ=2ft)+期待値huber。
評価は fold外モデルで(OOF厳守)。struct不使用。
使い方: MODE=train python v77/p15_bandcnn_v77.py / MODE=apply NW=60 ...
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = os.environ.get("MODE", "train")
NW = int(os.environ.get("NW", "60"))
BAND = 40.0; STEP = 1.0; S = int(2 * BAND / STEP) + 1        # 81
GRMA = 13; TWMA = 6.0
CROP = int(os.environ.get("CROP", "768"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
LR = float(os.environ.get("LR", "1e-3"))
NFOLD = 5
POW = 1.7826896851395453
GRSIG = dict(mn=11.865793233968713, mx=44.98927964612316, df=119.25397464636981, mult=3.058318309943017)
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs_np = np.arange(-BAND, BAND + STEP / 2, STEP)


def build_image(wid, need_true=True):
    """(C,T,S) f32 numpy + A + true"""
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    gr2 = np.convolve(gr, np.ones(GRMA) / GRMA, mode="same")
    step = float(np.median(np.diff(tw_tvt)))
    wpt = max(3, int(round(TWMA / max(step, 1e-3))) | 1)
    tw_gr2 = np.convolve(tw_gr, np.ones(wpt) / wpt, mode="same")
    kpos = hw.index.get_indexer(kn.index); epos = hw.index.get_indexer(ev.index)
    kt = kn["TVT_input"].to_numpy(float)
    def calib(g_, t_):
        r = g_[kpos] - np.interp(kt, tw_tvt, t_)
        s_ = float(np.nanstd(r))
        if not np.isfinite(s_) or s_ <= 0: s_ = GRSIG["df"]
        return float(np.clip(s_ * GRSIG["mult"], GRSIG["mn"], GRSIG["mx"]))
    gs, gs2 = calib(gr, tw_gr), calib(gr2, tw_gr2)
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sdd = np.clip(a["sig"][:T], 1.0, None)
    At = torch.tensor(A, device=DEV, dtype=torch.float32)
    offs = torch.tensor(offs_np, device=DEV, dtype=torch.float32)
    tvtr = At[:, None] + offs[None, :]
    ttv = torch.tensor(tw_tvt, device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, tvtr.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((tvtr - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    chans = []
    for g_, t_, s_ in ((gr, tw_gr, gs), (gr2, tw_gr2, gs2)):
        tgr_ = torch.tensor(t_, device=DEV, dtype=torch.float32)
        egr = tgr_[ii - 1] * (1 - wv) + tgr_[ii] * wv
        grt = torch.tensor(g_[epos][:T], device=DEV, dtype=torch.float32)
        nll = -0.5 * (((grt[:, None] - egr) / s_).abs()) ** POW
        nll = nll - nll.max(1, keepdim=True).values           # 行内正規化(≤0)
        chans.append(nll)
    sd = torch.tensor(sdd, device=DEV, dtype=torch.float32)
    chans.append(-0.5 * (offs[None, :] / sd[:, None]) ** 2)
    chans.append((offs[None, :] / 40.0).expand(T, -1))
    img = torch.stack(chans, 0)                                # (4,T,S)
    tv = ev["TVT"].to_numpy(float)[:T] if need_true else None
    return dict(img=img, A=A, true=tv, sdd=sdd)


class UNetLite(nn.Module):
    def __init__(self, C=4, ch=24):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())
        self.e1 = blk(C, ch); self.e2 = blk(ch, ch * 2); self.e3 = blk(ch * 2, ch * 4)
        self.d2 = blk(ch * 4 + ch * 2, ch * 2); self.d1 = blk(ch * 2 + ch, ch)
        self.head = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):                                      # (B,4,T,S) → (B,T,S) logits
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, (2, 1)))                 # MD方向のみ縮約(帯方向は保持)
        e3 = self.e3(F.avg_pool2d(e2, (2, 1)))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        # 構造bias: 生GR照合ch0をロジットに直結(zero-init headでもGR追従から開始)
        return self.head(d1).squeeze(1) + x[:, 0] + x[:, 2]


def fold_of(wid, wids_sorted): return wids_sorted.index(wid) % NFOLD


def train():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = [w for w in wids if w in ANC]
    offs = torch.tensor(offs_np, device=DEV, dtype=torch.float32)
    t0 = time.time()
    for f in range(NFOLD):
        torch.manual_seed(777 + f)
        rng = np.random.default_rng(777 + f)
        model = UNetLite().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        trw = [w for w in wids if wids.index(w) % NFOLD != f]
        for ep in range(EPOCHS):
            rng.shuffle(trw)
            tot = n = 0
            for wid in trw:
                c = build_image(wid)
                if c is None: continue
                off_true = c["true"] - c["A"]
                T = len(off_true)
                if T < 200: continue
                for _ in range(3):                              # 井戸ごと3crop
                    s0 = int(rng.integers(0, max(1, T - CROP)))
                    sl = slice(s0, min(s0 + CROP, T))
                    ot = off_true[sl]
                    okm = np.isfinite(ot) & (np.abs(ot) < BAND - 2)
                    if okm.sum() < 100: continue
                    x = c["img"][:, sl][None]                   # (1,4,L,S)
                    L = x.shape[2]
                    if L % 4: x = F.pad(x, (0, 0, 0, 4 - L % 4))
                    logits = model(x)[0, :L]
                    ott = torch.tensor(ot, device=DEV, dtype=torch.float32)
                    okt = torch.tensor(okm, device=DEV)
                    tgt = torch.exp(-0.5 * ((offs[None, :] - ott[:, None]) / 2.0) ** 2)
                    tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-12)
                    ce = -(tgt * torch.log_softmax(logits, 1)).sum(1)
                    p = torch.softmax(logits, 1)
                    ev_off = (p * offs[None, :]).sum(1)
                    hub = F.huber_loss(ev_off, torch.nan_to_num(ott), delta=5.0, reduction="none")
                    loss = (ce[okt].mean() + 0.1 * hub[okt].mean())
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss.detach()); n += 1
            print(f"fold{f} ep{ep} loss {tot/max(n,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), ART / f"p15_unet_f{f}.pt")
    print(f"[train done] {time.time()-t0:.0f}s")


@torch.no_grad()
def apply():
    wids_all = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids_all = [w for w in wids_all if w in ANC]
    models = []
    for f in range(NFOLD):
        m = UNetLite().to(DEV); m.load_state_dict(torch.load(ART / f"p15_unet_f{f}.pt", map_location=DEV))
        m.eval(); models.append(m)
    offs = torch.tensor(offs_np, device=DEV, dtype=torch.float32)
    wids = wids_all[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    for i, wid in enumerate(wids):
        c = build_image(wid)
        if c is None: continue
        f = wids_all.index(wid) % NFOLD
        x = c["img"][None]
        L = x.shape[2]
        if L % 4: x = F.pad(x, (0, 0, 0, 4 - L % 4))
        logits = models[f](x)[0, :L]
        p = torch.softmax(logits, 1)
        om = (p * offs[None, :]).sum(1).cpu().numpy()
        pred = c["A"] + om
        e = pred - c["true"]; m = np.isfinite(e)
        es.append(e[m]); ea.append((c["A"] - c["true"])[m])
        per[wid] = float(np.sqrt(np.mean(e[m] ** 2)))
        outs[wid] = dict(pred=pred.astype(np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P15 band-CNN ({len(per)}井, OOF) ===")
    print(f"  pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})  中央 {np.median(list(per.values())):.3f}")
    fp = ART / f"p15_out{os.environ.get('TAG','')}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if MODE == "train": train()
    else: apply()
