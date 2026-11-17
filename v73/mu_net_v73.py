# -*- coding: utf-8 -*-
"""
v73 段a-2 — 学習ドリフト場 μ_θ(rate target net)
=================================================
伝播の rate 目標(ドリフト先)を行レベルで学習。
- 教師: 真rate = d(TVT_true+Z)/dMD(rolling median 21 平滑, clip±2)。真値は学習時のみ。
- 入力(test合法6ch): struct dip / dz/dmd(坑井形状) / 正規化GR / GR std21 / |GR勾配| / md_since
  ＋井戸スカラー ir(prefix dip) を各行にブロードキャスト → 計7ch
- モデル: dilated Conv1D(sigma_netと同型)。5-fold(坑井 sorted%5) OOF。
- 出力: artifacts_v73/mu_rows_v73.pkl {well: μ(ev行,) f32} → nnpf_v73 RTGT=learned で消費。
使い方: python v73/mu_net_v73.py
"""
import os, io, sys, time, pickle, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v73" / "artifacts_v73"; ART.mkdir(exist_ok=True)
STRUCT_FP = P0 / "v38" / "artifacts_v38" / "struct_prior_train.pkl"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "48"))
CH = int(os.environ.get("CH", "48"))
NWX = int(os.environ.get("NW", "10000"))
CIN = 7


def build():
    SP = pickle.loads(STRUCT_FP.read_bytes())
    wids = sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NWX]
    out = {}
    for wn in wids:
        hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10: continue
        md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
        tvt = ev["TVT"].to_numpy(float)
        fb = float(np.nanmean(tw["GR"])) if np.isfinite(np.nanmean(tw["GR"])) else 0.0
        g = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)[
            hw.index.get_indexer(ev.index)]
        # 教師: 真rate(平滑)
        y = np.gradient(tvt + z, md)
        y = pd.Series(y).rolling(21, center=True, min_periods=1).median().to_numpy()
        y = np.clip(y, -2.0, 2.0).astype(np.float32)
        msk = np.isfinite(y)
        # struct dip(平滑) — a-1と同じ作り
        sp = SP.get(wn)
        if sp is not None and len(sp["tvt"]) == len(ev):
            sd = np.gradient(np.asarray(sp["tvt"], float) + z, md)
            sd = pd.Series(sd).rolling(21, center=True, min_periods=1).median().to_numpy()
            sd = np.clip(sd, -2.0, 2.0)
        else:
            sd = np.zeros(len(ev))
        dz = np.gradient(z, md)
        s = pd.Series(g)
        med = np.nanmedian(g); iqr = max(np.nanpercentile(g, 75) - np.nanpercentile(g, 25), 1e-3)
        gn = (g - med) / iqr
        st21 = np.nan_to_num(s.rolling(21, center=True, min_periods=1).std().to_numpy() / iqr)
        grd = np.nan_to_num(np.abs(np.gradient(g)) / iqr)
        mds = (md - md[0]) / 5000.0
        # ir(prefix dip)
        tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dzz = np.diff(tail["Z"].values)
        dm = np.diff(tail["MD"].values); m = dm > 0
        ir = float(np.median((dt + dzz)[m] / dm[m])) if m.sum() >= 3 else 0.0
        X = np.stack([sd, dz, gn, st21, grd, mds, np.full(len(ev), ir)], 0).astype(np.float32)
        out[wn] = dict(X=np.nan_to_num(X), y=y, m=msk)
    return out


class MuNet(nn.Module):
    def __init__(self, cin=CIN, ch=CH):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d) for d in (1, 4, 16, 64)])
        self.norm = nn.ModuleList([nn.GroupNorm(4, ch) for _ in range(4)])
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = F.gelu(self.inp(x))
        for c, nm in zip(self.blocks, self.norm):
            h = h + F.gelu(nm(c(h)))
        return self.head(h).squeeze(1)


def main():
    t0 = time.time()
    cache_fp = ART / "munet_data_v73.pkl"
    if cache_fp.exists():
        D = pickle.loads(cache_fp.read_bytes())
    else:
        D = build(); cache_fp.write_bytes(pickle.dumps(D, protocol=4))
    wells = sorted(D.keys()); fold_of = {w: i % 5 for i, w in enumerate(wells)}
    print(f"wells={len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    preds = {}
    for f in range(5):
        trw = [w for w in wells if fold_of[w] != f]; vaw = [w for w in wells if fold_of[w] == f]
        model = MuNet().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.RandomState(11 + f)
        for ep in range(EPOCHS):
            order = list(trw); rng.shuffle(order); tot = 0.0; nb = 0
            for c0 in range(0, len(order), BATCH):
                ws = order[c0:c0 + BATCH]
                Tmax = max(D[w]["X"].shape[1] for w in ws)
                X = np.zeros((len(ws), CIN, Tmax), np.float32)
                Y = np.zeros((len(ws), Tmax), np.float32); M = np.zeros((len(ws), Tmax), bool)
                for j, w in enumerate(ws):
                    T = D[w]["X"].shape[1]
                    X[j, :, :T] = D[w]["X"]; Y[j, :T] = D[w]["y"]; M[j, :T] = D[w]["m"]
                xb = torch.from_numpy(X).to(DEV); yb = torch.from_numpy(Y).to(DEV); mb = torch.from_numpy(M).to(DEV)
                o = model(xb)
                loss = F.huber_loss(o[mb], yb[mb], delta=0.05)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                tot += loss.item(); nb += 1
            if ep % 2 == 1 or ep == EPOCHS - 1:
                print(f"  fold{f} ep{ep} loss {tot/nb:.5f}", flush=True)
        model.eval()
        with torch.no_grad():
            for w in vaw:
                x = torch.from_numpy(D[w]["X"][None]).to(DEV)
                preds[w] = np.clip(model(x)[0].cpu().numpy(), -2.0, 2.0).astype(np.float32)
        torch.save(model.state_dict(), ART / f"munet_f{f}.pt")
    # OOF品質: corr + RMSE(rate) vs 基準(struct dip / ir / zero)
    cs, rn, rs, ri, rz = [], [], [], [], []
    for w in wells:
        m = D[w]["m"]; y = D[w]["y"][m]; p = preds[w][m]
        sd = D[w]["X"][0][m]; ir = D[w]["X"][6][m]
        if m.sum() > 50:
            cs.append(np.corrcoef(p, y)[0, 1])
        rn.append(np.mean((p - y) ** 2)); rs.append(np.mean((sd - y) ** 2))
        ri.append(np.mean((ir - y) ** 2)); rz.append(np.mean(y ** 2))
    print(f"\nOOF corr(μ_pred, 真rate): 中央 {np.median(cs):.3f}")
    print(f"rate RMSE: μ_net {np.sqrt(np.mean(rn)):.4f} / struct {np.sqrt(np.mean(rs)):.4f} "
          f"/ ir {np.sqrt(np.mean(ri)):.4f} / zero {np.sqrt(np.mean(rz)):.4f}")
    (ART / "mu_rows_v73.pkl").write_bytes(pickle.dumps(preds, protocol=4))
    print(f"[saved] mu_rows_v73.pkl ({len(preds)}井)  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
