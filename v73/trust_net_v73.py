# -*- coding: utf-8 -*-
"""
v73 段a-2' — struct dip の学習trustゲート
==========================================
段a採用構成(structgate=σゲート)の σ を「学習した信頼度」に置換する試み。
- 教師: trust_t = 1[|struct_dip_t − 真rate_t| < THR](THR=0.01。真値は学習時のみ)
- 入力(8ch, test合法): struct_dip / struct_σ / dz/dmd / 正規化GR / GR std21 / |GR勾配| / md_since / ir
- モデル: dilated Conv1D + BCE。5-fold(坑井 sorted%5) OOF。
- 出力: artifacts_v73/trust_rows_v73.pkl {well: P(trust)(ev行,) f32}
  → nnpf_v73 RTGT=trustgate で K_t = RTK·P^TPOW として消費。
使い方: python v73/trust_net_v73.py
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
THR = float(os.environ.get("THR", "0.01"))
CIN = 8


def build():
    SP = pickle.loads(STRUCT_FP.read_bytes())
    wids = sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    out = {}
    for wn in wids:
        hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10: continue
        sp = SP.get(wn)
        if sp is None or len(sp["tvt"]) != len(ev): continue
        md = ev["MD"].to_numpy(float); z = ev["Z"].to_numpy(float)
        tvt = ev["TVT"].to_numpy(float)
        y = np.gradient(tvt + z, md)
        y = pd.Series(y).rolling(21, center=True, min_periods=1).median().to_numpy()
        y = np.clip(y, -2.0, 2.0)
        sd = np.gradient(np.asarray(sp["tvt"], float) + z, md)
        sd = pd.Series(sd).rolling(21, center=True, min_periods=1).median().to_numpy()
        sd = np.clip(sd, -2.0, 2.0)
        sig = np.clip(np.asarray(sp["sig"], float), 0.5, 60.0)
        lab = (np.abs(sd - y) < THR).astype(np.float32)
        msk = np.isfinite(sd - y)
        fb = float(np.nanmean(tw["GR"])) if np.isfinite(np.nanmean(tw["GR"])) else 0.0
        g = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)[
            hw.index.get_indexer(ev.index)]
        dz = np.gradient(z, md)
        s = pd.Series(g)
        med = np.nanmedian(g); iqr = max(np.nanpercentile(g, 75) - np.nanpercentile(g, 25), 1e-3)
        gn = (g - med) / iqr
        st21 = np.nan_to_num(s.rolling(21, center=True, min_periods=1).std().to_numpy() / iqr)
        grd = np.nan_to_num(np.abs(np.gradient(g)) / iqr)
        mds = (md - md[0]) / 5000.0
        tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dzz = np.diff(tail["Z"].values)
        dm = np.diff(tail["MD"].values); m = dm > 0
        ir = float(np.median((dt + dzz)[m] / dm[m])) if m.sum() >= 3 else 0.0
        X = np.stack([sd, np.log10(sig) - 0.7, dz, gn, st21, grd, mds, np.full(len(ev), ir)], 0)
        out[wn] = dict(X=np.nan_to_num(X).astype(np.float32), y=lab, m=msk)
    return out


class TrustNet(nn.Module):
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
        return self.head(h).squeeze(1)          # logits


def main():
    t0 = time.time()
    cache_fp = ART / "trustnet_data_v73.pkl"
    if cache_fp.exists():
        D = pickle.loads(cache_fp.read_bytes())
    else:
        D = build(); cache_fp.write_bytes(pickle.dumps(D, protocol=4))
    wells = sorted(D.keys()); fold_of = {w: i % 5 for i, w in enumerate(wells)}
    base_rate = np.mean([D[w]["y"][D[w]["m"]].mean() for w in wells])
    print(f"wells={len(wells)}  trust率(平均) {base_rate:.3f}  THR={THR} ({time.time()-t0:.0f}s)", flush=True)
    preds = {}
    for f in range(5):
        trw = [w for w in wells if fold_of[w] != f]; vaw = [w for w in wells if fold_of[w] == f]
        model = TrustNet().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.RandomState(23 + f)
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
                loss = F.binary_cross_entropy_with_logits(o[mb], yb[mb])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                tot += loss.item(); nb += 1
            if ep % 3 == 2 or ep == EPOCHS - 1:
                print(f"  fold{f} ep{ep} loss {tot/nb:.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            for w in vaw:
                x = torch.from_numpy(D[w]["X"][None]).to(DEV)
                preds[w] = torch.sigmoid(model(x)[0]).cpu().numpy().astype(np.float32)
        torch.save(model.state_dict(), ART / f"trustnet_f{f}.pt")
    # OOF品質: AUC近似(順位相関) + σゲートとの比較
    from sklearn.metrics import roc_auc_score
    aucs = []; auc_sig = []
    for w in wells:
        m = D[w]["m"]; yv = D[w]["y"][m]
        if 0 < yv.mean() < 1 and m.sum() > 100:
            aucs.append(roc_auc_score(yv, preds[w][m]))
            auc_sig.append(roc_auc_score(yv, -D[w]["X"][1][m]))   # σ小=信頼 の代理と比較
    print(f"\nOOF AUC(trust): 中央 {np.median(aucs):.3f}  /  σ代理ゲートのAUC: 中央 {np.median(auc_sig):.3f}")
    (ART / "trust_rows_v73.pkl").write_bytes(pickle.dumps(preds, protocol=4))
    print(f"[saved] trust_rows_v73.pkl ({len(preds)}井)  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
