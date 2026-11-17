# v74 Phase D - v7 field engine (11.56) + sequence GRU residual learning (GR-free)
#   baseline = Phase C v7 predicted drift d7(t) as input channel; target = true drift - d7 (residual).
# env: NW(0=all), STRIDE(2/4), EPOCHS(10-12), FLIP(1), HID(96), FOLD(-1=all), SEEDS(3)
import os, json, time, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_field2_v74 as F

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v74", "artifacts_v74"); os.makedirs(ART, exist_ok=True)
STRIDE = int(os.environ.get("STRIDE", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
FLIP = os.environ.get("FLIP", "1") == "1"
HID = int(os.environ.get("HID", "96"))
FOLD = int(os.environ.get("FOLD", "0"))
PREFIX_CTX = os.environ.get("PREFIX_CTX", "0") == "1"   # 1=prefix文脈連結(実測+0.14悪化=既定OFF)
MULTI = os.environ.get("MULTI", "0") == "1"             # 1=エンジン3性格のチャネル併給
HUBER = float(os.environ.get("HUBER", "8.0"))           # 残差学習のHuberδ(遠距離圧はδ大)
ENGINE = os.environ.get("ENGINE", "v7")                 # v7=自作場エンジン / struct=既存struct軌跡
DGC = os.environ.get("DGC", "1") == "1"                 # 勾配補正転写チャネル(実測+0.07悪=OFF可)
INP = os.environ.get("INP", "0") == "1"                 # インペインティングOOF注入チャネル
_INPZ = None
NBK = int(os.environ.get("NBK", "3"))
WINS = [301, 1001, 2001, 4001]

lowpass = F.lowpass

def build_seqs(D, ws, dumps):
    cent = np.array([[np.nanmean(D[w]["x"]), np.nanmean(D[w]["y"])] for w in ws])
    trees = {w: cKDTree(np.c_[D[w]["x"], D[w]["y"]][::2]) for w in ws}
    nbrs = {}
    for i, w in enumerate(ws):
        dd = np.hypot(cent[:, 0] - cent[i, 0], cent[:, 1] - cent[i, 1]); dd[i] = 1e18
        nbrs[w] = [ws[j] for j in np.argsort(dd)[:NBK]]
    seqs = {}
    for w in ws:
        d = D[w]
        md, x, y, z, g = d["md"], d["x"], d["y"], d["z"], d["tvt"] + d["z"]
        t0 = d["t0"]; n = len(md)
        pi = (np.arange(max(t0 - 1000, 0), t0 + 1)[::STRIDE] if PREFIX_CTX
              else np.arange(0))
        ei = np.r_[pi, np.arange(t0 + 1, n)[::STRIDE]].astype(int)
        n_pre = len(pi)
        tvt_last = float(d["tin"][t0])
        def eng_channels(dp):
            dfull = dp["pred"] - tvt_last                    # pred drift (t0+1..n)
            dk = np.empty(len(ei), np.float32)
            dk[:n_pre] = d["tin"][pi] - tvt_last             # prefix = known drift (legal)
            dk[n_pre:] = dfull[(ei[n_pre:] - (t0 + 1))]
            return dk, dp["rates"][ei], dp["conf"][ei]
        d7, rates, conf = eng_channels(dumps[0][w])          # 主エンジン(残差の基準)
        extra_eng = [eng_channels(dp[w]) for dp in dumps[1:]] if len(dumps) > 1 else []
        ki = np.where(d["kn"])[0]
        zh301 = z - lowpass(z, 301)
        tl = ki[-min(len(ki), 1000):]
        gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]
        vz = float(np.dot(zt, zt))
        k_pref = float(np.clip(np.dot(zt, gt) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
        az = np.arctan2(np.gradient(y), np.gradient(x))
        az_s = np.arctan2(lowpass(np.sin(az), 301), lowpass(np.cos(az), 301))
        bands = {W: lowpass(z, W) for W in WINS}
        msb = (np.arange(n) - t0).astype(float)
        # 謗伜炎隗貞ｺｦ繝ｻprefix dip螟匁諺繝峨Μ繝輔ヨ(GR繝輔Μ繝ｼ)
        dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9
        incl = np.arctan2(np.gradient(lowpass(z, 301)), dh)
        build_r = np.gradient(lowpass(incl, 201)) * 1000
        g_s = lowpass(g, 201)
        r_all = np.gradient(g_s) / np.maximum(dh, 1e-6)
        tlp = ki[(ki >= t0 - 1500)]
        r_pref = float(np.clip(np.median(r_all[tlp]), -0.08, 0.08)) if len(tlp) > 100 else 0.0
        dprior = (r_pref * (md - md[t0]) - (z - z[t0]))       # dip螟匁諺縺ｮTVT繝峨Μ繝輔ヨ
        kn_flag = np.r_[np.ones(n_pre, np.float32), np.zeros(len(ei) - n_pre, np.float32)]
        ch = [d7 / 20.0, rates * 20.0, conf, kn_flag]
        for dk, rk, ck in extra_eng:
            ch += [dk / 20.0, (dk - d7) / 10.0, rk * 20.0, ck]
        ch += [
              msb[ei] / 3000.0, np.sin(az_s[ei]), np.cos(az_s[ei]),
              np.full(len(ei), k_pref),
              incl[ei] * 3, build_r[ei], (incl[ei] - incl[t0]) * 3,
              np.clip(dprior[ei], -120, 120) / 30.0]
        for W in WINS:
            ch.append((bands[W][ei] - bands[W][t0]) / 50.0)
            ch.append((z - bands[W])[ei] / 10.0)
        p0 = np.array([x[t0], y[t0]])
        gxr = dumps[0][w].get("gx"); gyr = dumps[0][w].get("gy")
        if gxr is None:
            gxr = np.zeros(n, np.float32); gyr = np.zeros(n, np.float32)
        for kk, nw in enumerate(nbrs[w]):
            nd = D[nw]; tr = trees[nw]
            _, j0 = tr.query(p0); j0 = int(j0) * 2
            dq, jj = tr.query(np.c_[x[ei], y[ei]])
            jj = (np.asarray(jj) * 2).astype(int)
            gks = lowpass(nd["tvt"] + nd["z"], 201)
            # 勾配補正付きレベル転写(co-kriging式): g_nbr(Q)+∇g·(P−Q)
            cxy = gxr[ei] * (x[ei] - nd["x"][jj]) + gyr[ei] * (y[ei] - nd["y"][jj])
            cxy0 = float(gxr[t0] * (x[t0] - nd["x"][j0]) + gyr[t0] * (y[t0] - nd["y"][j0]))
            dgc = np.clip((gks[jj] + cxy) - (gks[j0] + cxy0), -120, 120)
            if kk < 3:
                dipk = np.gradient(gks, nd["md"], edge_order=1)
                ch += [np.clip(gks[jj] - gks[j0], -120, 120) / 30.0,
                       np.asarray(dq) / 1000.0, dipk[jj] * 30,
                       np.cos(az_s[ei] - np.arctan2(np.gradient(nd["y"])[jj],
                                                    np.gradient(nd["x"])[jj])),
                       np.full(len(ei), float(np.clip(gks[j0] - g[t0], -120, 120)) / 30.0)]
                if DGC:
                    ch.append(dgc / 30.0)
            else:
                ch += [dgc / 30.0, np.asarray(dq) / 1000.0]
        if INP and _INPZ is not None:
            key = w + "__rows"
            if key in _INPZ.files:
                mfull = np.full(n, np.nan, np.float32)
                mfull[_INPZ[key]] = _INPZ[w + "__pred"]
                idrift = mfull[ei] - tvt_last
                iflag = np.isfinite(idrift).astype(np.float32)
                ch += [np.clip(np.nan_to_num(idrift), -120, 120) / 30.0, iflag]
            else:
                ch += [np.zeros(len(ei), np.float32), np.zeros(len(ei), np.float32)]
        X = np.nan_to_num(np.stack(ch, 1).astype(np.float32))
        drift_true = (d["tvt"][ei] - tvt_last).astype(np.float32)
        resid = drift_true - d7.astype(np.float32)           # 蟄ｦ鄙稚arget(谿句ｷｮ)
        evm = d["ev"][ei]
        seqs[w] = (X, resid, d7.astype(np.float32), drift_true, evm, ei)
    return seqs

def main():
    t0 = time.time()
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    D = F.load()
    ws, donors, pair = F.build_donors(D)
    if ENGINE == "phaseG":
        # 作り替えエンジン(phaseG_struct2_v74.py)のダンプを baseline に使用
        gz = np.load(os.path.join(ART, "phaseG_dump.npz"))
        dp = {}
        kept = []
        for w in ws:
            if w + "__pred" not in gz.files:
                continue
            d = D[w]; n = len(d["md"]); t0w = d["t0"]
            s = int(gz[w + "__s"])
            pred_tail = gz[w + "__pred"]
            if s != t0w or len(pred_tail) != n - t0w - 1:
                continue
            rates_full = np.zeros(n, np.float32)
            rates_full[t0w + 1:] = gz[w + "__rate"]
            conf_full = np.zeros(n, np.float32)
            ddr = gz[w + "__dd"]; spr = gz[w + "__spread"]
            conf_full[t0w + 1:] = (np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) *
                                   np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06))
            dp[w] = dict(pred=pred_tail.astype(np.float32), rates=rates_full,
                         conf=conf_full)
            kept.append(w)
        ws = kept
        print(f"ENGINE=phaseG  wells={len(ws)}", flush=True)
        dumps = [dp]
    elif ENGINE in ("struct", "both"):
        # 既存struct軌跡(GRフリー, prior_v66_leanv1cumul)を baseline エンジンとして使用
        import pickle
        PKL = os.path.join(P0, "v38", "artifacts_v38", "priors",
                           "prior_v66_leanv1cumul_train.pkl")
        with open(PKL, "rb") as f:
            struct = pickle.load(f)
        dp = {}
        kept = []
        for w in ws:
            d = D[w]; sp = struct.get(w)
            n = len(d["md"]); t0w = d["t0"]
            if sp is None or len(np.asarray(sp["tvt"])) != n - t0w - 1:
                continue
            s_tvt = np.asarray(sp["tvt"], np.float32)
            sig = np.asarray(sp["sig"], np.float32)
            g_s = np.empty(n, np.float64)
            g_true = d["tvt"] + d["z"]
            g_s[:t0w + 1] = np.where(np.isfinite(g_true[:t0w + 1]), g_true[:t0w + 1],
                                     float(d["tin"][t0w] + d["z"][t0w]))
            g_s[t0w + 1:] = s_tvt + d["z"][t0w + 1:]
            rates_full = np.gradient(F.lowpass(g_s, 101)) / np.maximum(d["dh"], 1e-6)
            conf_full = np.zeros(n, np.float32)
            conf_full[t0w + 1:] = 1.0 / (1.0 + np.nan_to_num(sig, nan=60.0) / 15.0)
            dp[w] = dict(pred=s_tvt,
                         rates=np.clip(rates_full, -0.15, 0.15).astype(np.float32),
                         conf=conf_full)
            kept.append(w)
        ws = kept
        print(f"ENGINE={ENGINE}  wells={len(ws)}", flush=True)
        dumps = [dp]
        if ENGINE == "both":
            dp7 = {}
            F.run(D, ws, donors, pair, "v7(dump)", seg=25, sigcap=800, k=128,
                  dump=dp7, wg=0.45, ga=0.40)
            dumps.append(dp7)      # 主=struct(残差基準), 副=v7チャネル
    else:
        cfgs = [dict(wg=0.45, ga=0.40)]
        if MULTI:
            cfgs += [dict(wg=0.30, ga=0.20), dict(wg=0.60, ga=0.50)]
        dumps = []
        for ci, cfg in enumerate(cfgs):
            dp = {}
            F.run(D, ws, donors, pair, f"engine{ci}(dump)", seg=25, sigcap=800, k=128,
                  dump=dp, **cfg)
            dumps.append(dp)
    if INP:
        global _INPZ
        _INPZ = np.load(os.path.join(ART, "inpaint_oof.npz"))
        print(f"INP channel on ({len([k for k in _INPZ.files if k.endswith('__rows')])} wells)",
              flush=True)
    seqs = build_seqs(D, ws, dumps)
    C = seqs[ws[0]][0].shape[1]
    print(f"wells={len(ws)}  C={C}  ({time.time()-t0:.0f}s)", flush=True)
    gi = np.arange(len(ws))
    folds = list(GroupKFold(5).split(gi, gi, gi))
    fold_list = range(5) if FOLD < 0 else [FOLD]
    all_es, all_e7 = [], []
    OOF = {}
    for fold in fold_list:
        tr_i, va_i = folds[fold]
        tr_w = [ws[i] for i in tr_i]; va_w = [ws[i] for i in va_i]
        es_f, e7_f, oof_f = train_one(D, ws, seqs, tr_w, va_w, C, fold, t0)
        all_es.append(es_f); all_e7.append(e7_f); OOF.update(oof_f)
    if len(OOF) > 500:
        pack = {}
        for w, (rows, err) in OOF.items():
            pack[w + "__rows"] = rows; pack[w + "__err"] = err
        np.savez_compressed(os.path.join(
            ART, os.environ.get("OOF_NAME", "phaseD_oof.npz")), **pack)
        print("OOF saved:", len(OOF), "wells")
    ov = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
    ov7 = float(np.sqrt(np.mean(np.concatenate(all_e7) ** 2)))
    print(f"\n[Phase D 蜈ｨfold] overall = {ov:.4f}  (v7 {ov7:.4f} / 譌｢蟄・truct 9.70)")
    with open(os.path.join(ART, "phaseD_all.json"), "w", encoding="utf-8") as f:
        json.dump(dict(overall=ov, v7=ov7, epochs=EPOCHS), f)

def train_one(D, ws, seqs, tr_w, va_w, C, fold, t0):
    """SEEDS 譛ｬ蟄ｦ鄙偵＠縺ｦ莠域ｸｬ蟷ｳ蝮・りｿ斐ｊ蛟､=(隧穂ｾ｡陦後・隱､蟾ｮ, v7隱､蟾ｮ)"""
    import torch
    import torch.nn as nn
    SEEDS = int(os.environ.get("SEEDS", "1"))
    preds_sum = {w: None for w in va_w}
    for sd in range(SEEDS):
        torch.manual_seed(sd)
        es_cat, e7 = _train_single(D, ws, seqs, tr_w, va_w, C, fold, t0, sd, preds_sum)
    # preds_sum 縺ｫ蟷ｳ蝮・ｸ医∩莠域ｸｬ縺悟・繧・荳九・_train_single縺ｧ蜉邂・
    es = []
    oof = {}
    for w in va_w:
        X, resid, d7, dtrue, evm, ei = seqs[w]
        p = preds_sum[w] / SEEDS
        err = ((d7 + p) - dtrue)[evm]
        es.append(err)
        oof[w] = (ei[evm].astype(np.int32), err.astype(np.float32))
    e7 = np.concatenate([(seqs[w][2] - seqs[w][3])[seqs[w][4]] for w in va_w])
    es_cat = np.concatenate(es)
    print(f"fold{fold} seed蟷ｳ蝮・{SEEDS}): val={np.sqrt(np.mean(es_cat**2)):.4f}", flush=True)
    return es_cat, e7, oof

def _train_single(D, ws, seqs, tr_w, va_w, C, fold, t0, seed, preds_sum):
    import torch
    import torch.nn as nn
    # 縺薙・fold縺ｮv7繝吶・繧ｹ繝ｩ繧､繝ｳ(隧穂ｾ｡陦後・縺ｿ)
    e7 = np.concatenate([(seqs[w][2] - seqs[w][3])[seqs[w][4]] for w in va_w])
    base7 = float(np.sqrt(np.mean(e7 ** 2)))
    print(f"fold{fold} v7 baseline = {base7:.4f}", flush=True)
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

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    items = []
    for w in tr_w:
        X, resid, d7, dtrue, evm, ei = seqs[w]
        items.append((X, resid, 0.0))
        if FLIP:
            items.append((X[::-1].copy(), (resid[::-1] - resid[-1]).copy(), 1.0))
    rng = np.random.default_rng(seed)
    BS = 12
    best = 1e18
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(len(items))
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BS):
            batch = [items[i] for i in order[b0:b0 + BS]]
            L = max(len(b[1]) for b in batch)
            xb = np.zeros((len(batch), L, C + 1), np.float32)
            yb = np.zeros((len(batch), L), np.float32)
            mb = np.zeros((len(batch), L), np.float32)
            for i, (X, yv, fl) in enumerate(batch):
                xb[i, :len(yv), :C] = X; xb[i, :len(yv), C] = fl
                yb[i, :len(yv)] = yv; mb[i, :len(yv)] = 1.0
            xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev)
            mb = torch.from_numpy(mb).to(dev)
            loss = (nn.functional.huber_loss(net(xb), yb, delta=HUBER,
                                             reduction="none") * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            net.eval()
            es = []
            with torch.no_grad():
                for w in va_w:
                    X, resid, d7, dtrue, evm, ei = seqs[w]
                    xb = np.zeros((1, len(resid), C + 1), np.float32)
                    xb[0, :, :C] = X
                    p = net(torch.from_numpy(xb).to(dev))[0].cpu().numpy()
                    es.append(((d7 + p) - dtrue)[evm])
            es_cat = np.concatenate(es)
            rm = float(np.sqrt(np.mean(es_cat ** 2)))
            best = min(best, rm)
            print(f"fold{fold} s{seed} ep{ep+1:3d}  loss={tot/nb:.3f}  val={rm:.4f}  "
                  f"(v7 {base7:.3f})  ({time.time()-t0:.0f}s)", flush=True)
    # 譛邨Ｆpoch縺ｮ莠域ｸｬ繧・preds_sum 縺ｫ蜉邂・繧ｷ繝ｼ繝牙ｹｳ蝮・畑)
    net.eval()
    with torch.no_grad():
        for w in va_w:
            X, resid, d7, dtrue, evm, ei = seqs[w]
            xb = np.zeros((1, len(resid), C + 1), np.float32)
            xb[0, :, :C] = X
            p = net(torch.from_numpy(xb).to(dev))[0].cpu().numpy()
            preds_sum[w] = p if preds_sum[w] is None else preds_sum[w] + p
    return es_cat, e7

if __name__ == "__main__":
    main()
