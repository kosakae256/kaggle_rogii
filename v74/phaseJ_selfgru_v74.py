# -*- coding: utf-8 -*-
"""v74 Phase J — field非依存の自己GRU(leave-field-out 9.x への正面攻撃)
   仮説: ドリラーの操舵ナッジ(ビルドレート変化=GRを見て動いた痕跡)を非線形に復号すれば、
         自fieldのドナー無しでもドリフトを部分回収できる(普遍物理は他field+合成から学べる)。
   構成: 入力=field非依存チャネルのみ(軌道バンド/傾角/ビルドレート/prefix統計/dip外挿。
         方位・近傍・座標・場は不使用)。target=drift(TVT−最終既知)。hold基準。
   評価: leave-field-out(FIELD_LINK=2000, 156クラスタ)。基準: hold 15.91 / field-outエンジン 16.7。
   合成: Phase I 生成器(field無し)で SYN_N 本を学習に混合(AUG_SYN比)。
   env: MODE(train), FIELD_LINK(2000), EPOCHS(10), SEEDS(3), STRIDE(2), AUG_SYN(1.0), SYN_N(1200)
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P0 = os.path.dirname(HERE)
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
ART = os.path.join(HERE, "artifacts_v74")
FIELD_LINK = float(os.environ.get("FIELD_LINK", "2000"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
SEEDS = int(os.environ.get("SEEDS", "3"))
STRIDE = int(os.environ.get("STRIDE", "2"))
AUG_SYN = float(os.environ.get("AUG_SYN", "1.0"))
SYN_N = int(os.environ.get("SYN_N", "1200"))
HID = int(os.environ.get("HID", "96"))
WINS = [301, 1001, 2001, 4001]

def lowpass(x, win):
    return pd.Series(x).rolling(int(win) | 1, center=True, min_periods=1).mean().to_numpy()

def self_channels(md, x, y, z, tin, t0, tvt=None):
    """field非依存チャネルのみ(方位・座標・近傍なし)。行=eval行stride。"""
    n = len(md)
    ei = np.arange(t0 + 1, n)[::STRIDE].astype(int)
    tvt_last = float(tin[t0])
    g = np.where(np.isfinite(tin), tin, np.nan) + z
    ki = np.where(np.isfinite(tin))[0]
    zh301 = z - lowpass(z, 301)
    tl = ki[-min(len(ki), 1000):]
    gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]
    fin = np.isfinite(gt)
    vz = float(np.dot(zt[fin], zt[fin]))
    k_pref = float(np.clip(np.dot(zt[fin], gt[fin]) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
    dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9
    incl = np.arctan2(np.gradient(lowpass(z, 301)), dh)
    build_r = np.gradient(lowpass(incl, 201)) * 1000
    build_f = np.gradient(lowpass(incl, 61)) * 1000          # 速いナッジ検出用
    g_s = lowpass(g, 201)
    r_all = np.gradient(np.nan_to_num(g_s, nan=0.0)) / np.maximum(dh, 1e-6)
    tlp = ki[(ki >= t0 - 1500)]
    rv = r_all[tlp]; rv = rv[np.isfinite(g_s[tlp])]
    r_pref = float(np.clip(np.median(rv), -0.08, 0.08)) if len(rv) > 100 else 0.0
    dprior = (r_pref * (md - md[t0]) - (z - z[t0]))
    pre_res = (g - lowpass(g, 301))[tl]
    pre_std = float(np.nanstd(pre_res)) if np.isfinite(pre_res).sum() > 50 else 0.0
    msb = (np.arange(n) - t0).astype(float)
    bands = {W: lowpass(z, W) for W in WINS}
    ch = [msb[ei] / 3000.0,
          np.full(len(ei), k_pref), np.full(len(ei), pre_std / 5.0),
          incl[ei] * 3, build_r[ei], build_f[ei],
          (incl[ei] - incl[t0]) * 3,
          np.clip(dprior[ei], -120, 120) / 30.0]
    for W in WINS:
        ch.append((bands[W][ei] - bands[W][t0]) / 50.0)
        ch.append((z - bands[W])[ei] / 10.0)
    X = np.nan_to_num(np.stack(ch, 1).astype(np.float32))
    if tvt is None:
        return X, ei, tvt_last, None
    drift = (tvt[ei] - tvt_last).astype(np.float32)
    return X, ei, tvt_last, drift

def load_real():
    wids = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN)
                      if f.endswith("__horizontal_well.csv")))
    out = []
    for w in wids:
        hw = pd.read_csv(os.path.join(TRAIN, f"{w}__horizontal_well.csv"),
                         usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        x = hw["X"].to_numpy(float); y = hw["Y"].to_numpy(float)
        z = hw["Z"].to_numpy(float)
        tvt = hw["TVT"].to_numpy(float); tin = hw["TVT_input"].to_numpy(float)
        kn = np.isfinite(tin) & np.isfinite(tvt)
        ev = (~np.isfinite(tin)) & np.isfinite(tvt)
        if kn.sum() < 100 or ev.sum() < 50:
            continue
        t0 = int(np.where(np.isfinite(tin))[0][-1])
        md = np.arange(len(x), dtype=float)
        X, ei, tvt_last, drift = self_channels(md, x, y, z, tin, t0, tvt)
        evm = (~np.isfinite(tin[ei])) & np.isfinite(tvt[ei])
        out.append(dict(wid=w, X=X, drift=drift, evm=evm,
                        cx=float(np.mean(x)), cy=float(np.mean(y))))
    return out

def gen_synth(n_wells, rng):
    import phaseI_synth_v74 as I
    items = []
    tries = 0
    while len(items) < n_wells and tries < n_wells * 3:
        tries += 1
        S, cell, na, nc = I.sample_surface(rng)
        cp = rng.uniform(500, nc * cell - 500)
        nrows = int(rng.uniform(3500, 8000))
        d = I.drill(rng, S, cell, na, nc, cp, nrows, is_query=True)
        t0 = d["t0"]
        if nrows - t0 < 200:
            continue
        md = np.arange(nrows, dtype=float)
        X, ei, tvt_last, drift = self_channels(md, d["x"], d["y"], d["z"],
                                               d["tin"], t0, d["tvt"])
        if len(ei) < 100:
            continue
        items.append((X, drift))
    return items

def main():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    from scipy.spatial import cKDTree
    t0c = time.time()
    wells = load_real()
    print(f"real wells={len(wells)}  C={wells[0]['X'].shape[1]}  ({time.time()-t0c:.0f}s)",
          flush=True)
    cxy = np.array([[w["cx"], w["cy"]] for w in wells])
    tree = cKDTree(cxy)
    parent = list(range(len(wells)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in tree.query_pairs(FIELD_LINK):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    fid = np.array([find(i) for i in range(len(wells))])
    uniq = {f: i for i, f in enumerate(sorted(set(fid.tolist())))}
    groups = np.array([uniq[f] for f in fid])
    print(f"fields={len(uniq)}", flush=True)
    syn_items = []
    if AUG_SYN > 0:
        rng = np.random.default_rng(20260721)
        syn_items = gen_synth(SYN_N, rng)
        print(f"synthetic={len(syn_items)}  ({time.time()-t0c:.0f}s)", flush=True)
    C = wells[0]["X"].shape[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(C + 1, HID)
            self.gru = nn.GRU(HID, HID, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.1)
            self.out = nn.Linear(HID * 2, 1)
        def forward(self, x):
            h = torch.relu(self.inp(x))
            h, _ = self.gru(h)
            return self.out(h)[..., 0]

    gi = np.arange(len(wells))
    folds = list(GroupKFold(5).split(gi, gi, groups))
    all_es, all_hold = [], []
    for fold, (tr_i, va_i) in enumerate(folds):
        items = []
        for i in tr_i:
            X, dr = wells[i]["X"], wells[i]["drift"]
            items.append((X, dr, 0.0))
            items.append((X[::-1].copy(), (dr[::-1] - dr[-1]).copy(), 1.0))
        n_add = int(AUG_SYN * len(items))
        rng2 = np.random.default_rng(fold)
        for j in rng2.choice(len(syn_items), size=min(n_add, len(syn_items)),
                             replace=False) if syn_items else []:
            items.append((syn_items[j][0], syn_items[j][1], 0.0))
        preds = {i: None for i in va_i}
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            net = Net().to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            rng3 = np.random.default_rng(sd)
            BS = 12
            for ep in range(EPOCHS):
                net.train()
                order = rng3.permutation(len(items))
                for b0 in range(0, len(order), BS):
                    batch = [items[k] for k in order[b0:b0 + BS]]
                    L = max(len(b[1]) for b in batch)
                    xb = np.zeros((len(batch), L, C + 1), np.float32)
                    yb = np.zeros((len(batch), L), np.float32)
                    mb = np.zeros((len(batch), L), np.float32)
                    for ii, (X, yv, fl) in enumerate(batch):
                        xb[ii, :len(yv), :C] = X; xb[ii, :len(yv), C] = fl
                        yb[ii, :len(yv)] = yv; mb[ii, :len(yv)] = 1.0
                    xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev)
                    mb = torch.from_numpy(mb).to(dev)
                    loss = (nn.functional.huber_loss(net(xb), yb, delta=8.0,
                                                     reduction="none") * mb).sum() / mb.sum()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                sch.step()
            net.eval()
            with torch.no_grad():
                for i in va_i:
                    X = wells[i]["X"]
                    xb = np.zeros((1, len(X), C + 1), np.float32)
                    xb[0, :, :C] = X
                    p = net(torch.from_numpy(xb).to(dev))[0].cpu().numpy()
                    preds[i] = p if preds[i] is None else preds[i] + p
        for i in va_i:
            w = wells[i]
            p = preds[i] / SEEDS
            m = w["evm"]
            all_es.append((p - w["drift"])[m])
            all_hold.append((-w["drift"])[m])
        rm = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
        print(f"fold{fold} 累積 overall={rm:.4f}  ({time.time()-t0c:.0f}s)", flush=True)
    ov = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
    hd = float(np.sqrt(np.mean(np.concatenate(all_hold) ** 2)))
    print(f"\n[Phase J self-GRU field-out] overall = {ov:.4f}  (hold {hd:.4f} / "
          f"field-outエンジン 16.7)")
    json.dump(dict(overall=ov, hold=hd, aug=AUG_SYN, syn=len(syn_items)),
              open(os.path.join(ART, "phaseJ_result.json"), "w"))

if __name__ == "__main__":
    main()
