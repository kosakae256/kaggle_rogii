# -*- coding: utf-8 -*-
"""
v73 C3' — 行レベルσ学習(ヘテロスケダスティックemission)
========================================================
PFのGR尤度σは現在「井戸ごと1定数」(prefix残差std)。これを行レベルに:
  σ_t = gs_well · mult_t,  mult_t = (pred|resid|_t / 井戸中央pred)^γ
- 教師: |resid_true|_t = |gr_sm_t − tw(TVT_true_t)|(真値は学習時のみ。推論入力はGR文脈だけ=test合法)
- モデル: dilated Conv1D(井戸のGR系列→行ごとlog|resid|予測)。5-fold(坑井, sorted%5)OOF。
- 出力: artifacts_v73/sig_rows_v73.pkl {well: mult(ev行,) f32} → nnpf_v73 が SIGROWS=1 で消費。
根拠: v66「σ(信頼度申告)の解像度が勝因」の行方向展開(docs/v73.md C3)。
使い方: python v73/sigma_net_v73.py            (学習+OOF出力+品質レポート)
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH = int(os.environ.get("BATCH", "48"))
CH = int(os.environ.get("CH", "48"))
NWX = int(os.environ.get("NW", "10000"))


def build():
    """井戸ごとに (入力チャネル(4,T_ev), 教師log|resid|(T_ev,), mask) を作る。"""
    wids = sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})[:NWX]
    out = {}
    for wn in wids:
        hw = pd.read_csv(TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) < 10: continue
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.0
        gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(fb).to_numpy(float)
        evp = hw.index.get_indexer(ev.index)
        g = gr[evp]
        tvt_true = ev["TVT"].to_numpy(float)
        eg = np.interp(tvt_true, tw_tvt, tw_gr)
        resid = g - eg
        msk = np.isfinite(resid)
        # 入力(全てtest合法): 正規化GR / 局所std21 / |勾配| / 局所レンジ63
        s = pd.Series(g)
        med = np.nanmedian(g); iqr = max(np.nanpercentile(g, 75) - np.nanpercentile(g, 25), 1e-3)
        gn = (g - med) / iqr
        st21 = s.rolling(21, center=True, min_periods=1).std().to_numpy() / iqr
        grd = np.abs(np.gradient(g)) / iqr
        rng63 = (s.rolling(63, center=True, min_periods=1).max()
                 - s.rolling(63, center=True, min_periods=1).min()).to_numpy() / iqr
        X = np.stack([gn, np.nan_to_num(st21), np.nan_to_num(grd), np.nan_to_num(rng63)], 0).astype(np.float32)
        y = np.log(np.abs(np.where(msk, resid, 0.0)) + 1.0).astype(np.float32)   # log(|resid|+1)
        out[wn] = dict(X=X, y=y, m=msk)
    return out


class SigNet(nn.Module):
    def __init__(self, cin=4, ch=CH):
        super().__init__()
        self.inp = nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d) for d in (1, 4, 16, 64)])
        self.norm = nn.ModuleList([nn.GroupNorm(4, ch) for _ in range(4)])
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                    # (B,4,T) → (B,T)
        h = F.gelu(self.inp(x))
        for c, nm in zip(self.blocks, self.norm):
            h = h + F.gelu(nm(c(h)))
        return self.head(h).squeeze(1)


def main():
    t0 = time.time()
    cache_fp = ART / "signet_data_v73.pkl"
    if cache_fp.exists():
        D = pickle.loads(cache_fp.read_bytes())
    else:
        D = build(); cache_fp.write_bytes(pickle.dumps(D, protocol=4))
    wells = sorted(D.keys()); fold_of = {w: i % 5 for i, w in enumerate(wells)}
    print(f"wells={len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    preds = {}
    for f in range(5):
        trw = [w for w in wells if fold_of[w] != f]; vaw = [w for w in wells if fold_of[w] == f]
        model = SigNet().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.RandomState(7 + f)
        for ep in range(EPOCHS):
            order = list(trw); rng.shuffle(order); tot = 0.0; nb = 0
            for c0 in range(0, len(order), BATCH):
                ws = order[c0:c0 + BATCH]
                Tmax = max(D[w]["X"].shape[1] for w in ws)
                X = np.zeros((len(ws), 4, Tmax), np.float32)
                Y = np.zeros((len(ws), Tmax), np.float32); M = np.zeros((len(ws), Tmax), bool)
                for j, w in enumerate(ws):
                    T = D[w]["X"].shape[1]
                    X[j, :, :T] = D[w]["X"]; Y[j, :T] = D[w]["y"]; M[j, :T] = D[w]["m"]
                xb = torch.from_numpy(X).to(DEV); yb = torch.from_numpy(Y).to(DEV); mb = torch.from_numpy(M).to(DEV)
                o = model(xb)
                loss = F.huber_loss(o[mb], yb[mb], delta=1.0)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                tot += loss.item(); nb += 1
            print(f"  fold{f} ep{ep} loss {tot/nb:.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            for w in vaw:
                x = torch.from_numpy(D[w]["X"][None]).to(DEV)
                preds[w] = model(x)[0].cpu().numpy()
        torch.save(model.state_dict(), ART / f"signet_f{f}.pt")
    # OOF品質 + mult 生成
    cs = []
    mults = {}
    for w in wells:
        m = D[w]["m"]; p = preds[w]; y = D[w]["y"]
        if m.sum() > 50:
            cs.append(np.corrcoef(p[m], y[m])[0, 1])
        pr = np.exp(p) - 1.0                       # pred |resid|
        mult = pr / max(np.median(pr), 1e-3)       # 井戸内相対(井戸レベルσはgs較正が既に持つ)
        mults[w] = np.clip(mult, 0.5, 3.0).astype(np.float32)
    print(f"\nOOF corr(pred, log|resid|): 中央 {np.median(cs):.3f} / 平均 {np.mean(cs):.3f}")
    (ART / "sig_rows_v73.pkl").write_bytes(pickle.dumps(mults, protocol=4))
    print(f"[saved] sig_rows_v73.pkl ({len(mults)}井)  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
