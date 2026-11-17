# v74 Phase B — GRフリー系列モデル(GRU)fold0プローブ(既存struct資産・GR不使用)
#   1井戸=1系列(評価行, stride4)。target: drift d(t)=TVT(t)−TVT_last。
#   入力ch: 軌道(Zバンド/掘削角度/方位) + prefix静的(dip/操舵κ) + 近傍K=3系列(dg/dist/dip/相対方位/lvl0)
#   DA: 井戸反転(flip)を学習にのみ追加(v36の実証レバー, valid=順のみ)。
#   評価: GroupKFold(5) fold0 の val 井戸 pooled RMSE(stride4行)。GBM fold0(12.46/13.52)との比較。
# env: NW(0=全), STRIDE(4), EPOCHS(40), FLIP(1), HID(96), FOLD(0)
import os, json, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
NW = int(os.environ.get("NW", "0"))
STRIDE = int(os.environ.get("STRIDE", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
FLIP = os.environ.get("FLIP", "1") == "1"
HID = int(os.environ.get("HID", "96"))
FOLD = int(os.environ.get("FOLD", "0"))
NBK = 3
WINS = [301, 1001, 2001, 4001]

def lowpass(x, win):
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()

def robust_slope(md, y):
    if len(md) < 30:
        return 0.0
    c = np.polyfit(md, y, 1)
    r = y - np.polyval(c, md)
    m = np.abs(r) < 2.5 * (np.std(r) + 1e-9)
    if m.sum() >= 20:
        c = np.polyfit(md[m], y[m], 1)
    return float(c[0])

def build():
    wells = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                       if f.endswith("__horizontal_well.csv")))
    if NW:
        wells = wells[:NW]
    D = {}
    for w in wells:
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input"])
        md = hw["MD"].to_numpy(float); x = hw["X"].to_numpy(float)
        y = hw["Y"].to_numpy(float); z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            continue
        D[w] = dict(md=md, x=x, y=y, z=z, tvt=tvt, tin=tin, kn=kn, ev=ev,
                    g=tvt + z, t0=int(np.where(kn)[0][-1]))
    ws = sorted(D.keys())
    cent = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in ws])
    trees = {w: cKDTree(np.c_[D[w]["x"], D[w]["y"]][::2]) for w in ws}
    nbrs = {}
    for i, w in enumerate(ws):
        dd = np.hypot(cent[:, 0] - cent[i, 0], cent[:, 1] - cent[i, 1]); dd[i] = 1e18
        nbrs[w] = [ws[j] for j in np.argsort(dd)[:NBK]]
    seqs = {}
    for w in ws:
        d = D[w]
        md, x, y, z, g = d["md"], d["x"], d["y"], d["z"], d["g"]
        t0 = d["t0"]; ei = np.where(d["ev"])[0][::STRIDE]
        tvt_last = float(d["tin"][t0])
        ki = np.where(d["kn"])[0]
        dips = {L: robust_slope(md[ki[-min(len(ki), L):]], g[ki[-min(len(ki), L):]])
                for L in (500, 1000, 2000)}
        zh301 = z - lowpass(z, 301)
        tl = ki[-min(len(ki), 1000):]
        gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]
        vz = float(np.dot(zt, zt))
        k_pref = float(np.clip(np.dot(zt, gt) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
        az = np.arctan2(np.gradient(y), np.gradient(x))
        az_s = np.arctan2(lowpass(np.sin(az), 301), lowpass(np.cos(az), 301))
        dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9
        incl = np.arctan2(np.gradient(lowpass(z, 301)), dh)
        build_r = np.gradient(lowpass(incl, 201)) * 1000
        bands = {W: lowpass(z, W) for W in WINS}
        msb = (np.arange(len(md)) - t0).astype(float)
        ch = [msb[ei] / 3000.0,
              np.sin(az_s[ei]), np.cos(az_s[ei]),
              (np.unwrap(az_s)[ei] - az_s[t0]),
              incl[ei] * 3, build_r[ei],
              (incl[ei] - incl[t0]) * 3]
        for W in WINS:
            ch.append((bands[W][ei] - bands[W][t0]) / 50.0)
            ch.append((z - bands[W])[ei] / 10.0)
        for L, sl in dips.items():
            ch.append(sl * (md[ei] - md[t0]) / 30.0)
        ch.append(np.full(len(ei), k_pref))
        p0 = np.array([x[t0], y[t0]])
        for nw in nbrs[w]:
            nd = D[nw]; tr = trees[nw]
            _, j0 = tr.query(p0); j0 = int(j0) * 2
            dq, jj = tr.query(np.c_[x[ei], y[ei]])
            jj = (np.asarray(jj) * 2).astype(int)
            gks = lowpass(nd["g"], 201)
            dipk = np.gradient(gks, nd["md"], edge_order=1)
            ch += [np.clip(gks[jj] - gks[j0], -120, 120) / 30.0,
                   np.asarray(dq) / 1000.0,
                   dipk[jj] * 30,
                   np.cos(az_s[ei] - np.arctan2(np.gradient(nd["y"])[jj],
                                                np.gradient(nd["x"])[jj])),
                   np.full(len(ei), float(np.clip(gks[j0] - g[t0], -120, 120)) / 30.0)]
        X = np.stack(ch, 1).astype(np.float32)          # (T, C)
        yv = (d["tvt"][ei] - tvt_last).astype(np.float32)
        seqs[w] = (np.nan_to_num(X), yv)
    return ws, seqs

def main():
    t0 = time.time()
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    ws, seqs = build()
    print(f"wells={len(ws)}  C={seqs[ws[0]][0].shape[1]}  ({time.time()-t0:.0f}s)", flush=True)
    gi = np.arange(len(ws))
    gkf = GroupKFold(5)
    tr_i, va_i = list(gkf.split(gi, gi, gi))[FOLD]
    tr_w = [ws[i] for i in tr_i]; va_w = [ws[i] for i in va_i]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = seqs[ws[0]][0].shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(C + 1, HID)            # +1 = is_backward
            self.gru = nn.GRU(HID, HID, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(HID * 2, 1)
        def forward(self, x):
            h = torch.relu(self.inp(x))
            h, _ = self.gru(h)
            return self.out(h)[..., 0]

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    items = []
    for w in tr_w:
        X, yv = seqs[w]
        items.append((X, yv, 0.0))
        if FLIP:
            items.append((X[::-1].copy(), yv[::-1].copy() - yv[-1], 1.0))
    print(f"train seqs={len(items)} (flip={FLIP})  val wells={len(va_w)}  dev={dev}", flush=True)
    rng = np.random.default_rng(0)
    BS = 12
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(len(items))
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BS):
            batch = [items[i] for i in order[b0:b0 + BS]]
            L = max(len(x[1]) for x in batch)
            xb = np.zeros((len(batch), L, C + 1), np.float32)
            yb = np.zeros((len(batch), L), np.float32)
            mb = np.zeros((len(batch), L), np.float32)
            for i, (X, yv, fl) in enumerate(batch):
                xb[i, :len(yv), :C] = X; xb[i, :len(yv), C] = fl
                yb[i, :len(yv)] = yv; mb[i, :len(yv)] = 1.0
            xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev)
            mb = torch.from_numpy(mb).to(dev)
            pred = net(xb)
            loss = (nn.functional.huber_loss(pred, yb, delta=8.0,
                                             reduction="none") * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        sch.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            net.eval()
            es = []
            with torch.no_grad():
                for w in va_w:
                    X, yv = seqs[w]
                    xb = np.zeros((1, len(yv), C + 1), np.float32)
                    xb[0, :, :C] = X
                    p = net(torch.from_numpy(xb).to(dev))[0].cpu().numpy()
                    es.append(p - yv)
            rm = float(np.sqrt(np.mean(np.concatenate(es) ** 2)))
            print(f"ep{ep+1:3d}  train_loss={tot/nb:.3f}  val overall={rm:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(ART, "phaseB_fold%d.json" % FOLD), "w", encoding="utf-8") as f:
        json.dump(dict(fold=FOLD, val_overall=rm, epochs=EPOCHS, flip=FLIP,
                       hid=HID, stride=STRIDE), f)
    print("done. val overall =", rm)

if __name__ == "__main__":
    main()
