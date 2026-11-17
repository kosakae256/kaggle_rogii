# -*- coding: utf-8 -*-
"""v74 Phase F — ANCC面インペインティングUNet(GRフリー・struct不使用・self-supervised)
   全trainトラックを走向整列ラスタに焼き、自トラックをマスクして周囲+自prefixから補完を学習。
   フレーム: Phase E検証済み部品(走向θ=ドナー勾配直交 / リッジ線形トレンド除去(残差std12-29ft) / prefix b0整列)。
   ラスタ: 50ft/cell, along256×across96 (12.8kft×4.8kft)。ch=[近傍残差burn, 近傍mask, dist, prefix残差burn]
   target: 自トラックのANCC残差(全行=train合法)。loss=自トラックcell(評価帯w1.0/prefix w0.3)。
   出力: TVT̂ = (pred_resid+trend) + b̂_prefix − z。評価=eval行 overall pooled(fold OOF)。
   env: MODE(build/train), NW, FOLD(-1=全), EPOCHS(30), SEEDS(1), BS(16)
"""
import io, sys, os, json, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_field2_v74 as F

ART = F.ART
CACHE = os.path.join(ART, "inpaint_cache.npz")
MODE = os.environ.get("MODE", "train")
NW = int(os.environ.get("NW", "0"))
FOLD = int(os.environ.get("FOLD", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
SEEDS = int(os.environ.get("SEEDS", "1"))
BS = int(os.environ.get("BS", "16"))
CELL = 50.0
NA, NC = 256, 96          # along × across cells
ROBS = 4000.0

def build_cache():
    t0 = time.time()
    D = F.load()
    ws, donors, pair = F.build_donors(D)
    dt = cKDTree(donors["pos"])
    n_don = len(donors["pos"])
    # 事前: 各井戸の anc_s と lateral点
    P = {}
    for w in ws:
        d = D[w]
        anc = d["anc"]
        m = np.isfinite(anc) & (d["dh"] > 0.5)
        anc_s = F.lowpass(np.where(np.isfinite(anc), anc, np.nan), 201)
        P[w] = (np.where(m & np.isfinite(anc_s))[0], anc_s)
    pack = {}
    meta = {}
    wsel = ws[:NW] if NW else ws
    for wi, w in enumerate(ws):
        if w not in set(wsel):
            continue
        d = D[w]
        n = len(d["md"]); t0w = d["t0"]
        # 近傍収集(Phase E同型)
        track = np.c_[d["x"], d["y"]][::40]
        cand = set()
        for p in track:
            cand.update(dt.query_ball_point(p, ROBS))
        cand = np.array(sorted(cand), dtype=np.int64)
        cand = cand[donors["well"][cand] != wi]
        nb_wells = sorted(set(donors["well"][cand].tolist()))
        Xo, yo = [], []
        for nwi in nb_wells:
            nw_ = ws[nwi]
            idx, anc_s = P[nw_]
            idx = idx[::4]
            nd = D[nw_]
            Xo.append(np.c_[nd["x"][idx], nd["y"][idx]])
            yo.append(anc_s[idx])
        if not Xo:
            continue
        Xo = np.concatenate(Xo); yo = np.concatenate(yo)
        tt = cKDTree(np.c_[d["x"], d["y"]][::20])
        dd_, _ = tt.query(Xo)
        keep = (dd_ < ROBS) & np.isfinite(yo)
        Xo = Xo[keep]; yo = yo[keep]
        if len(Xo) < 100:
            continue
        # 走向θ(Phase E同型)
        pc = np.array([np.nanmean(d["x"]), np.nanmean(d["y"])])
        dd2, jj2 = dt.query(pc, k=min(900, n_don))
        ok2 = np.isfinite(dd2) & (jj2 < n_don)
        jj2 = jj2[ok2]; jj2 = jj2[donors["well"][jj2] != wi][:256]
        if len(jj2) >= 12:
            (gx, gy), _, _ = F.solve_grad(donors["c"][jj2], donors["az"][jj2],
                                          np.ones(len(jj2)), 0.05, np.array([1.0, 0.0]))
            theta = np.arctan2(gy, gx) + np.pi / 2
        else:
            theta = float(np.arctan2(d["dyr"][t0w], d["dxr"][t0w]))
        ct, st = np.cos(theta), np.sin(theta)

        def rot(X):
            return np.c_[X[:, 0] * ct + X[:, 1] * st, -X[:, 0] * st + X[:, 1] * ct]
        # トレンド(リッジ)
        A_o = np.c_[np.ones(len(Xo)), (Xo - Xo.mean(0)) / 1000.0]

        def ridge_fit(A, yv):
            M = A.T @ A + np.diag([1e-6, 3.0, 3.0]) * len(A) * 1e-3
            return np.linalg.solve(M, A.T @ yv)
        coef = ridge_fit(A_o, yo)
        r_tr = yo - A_o @ coef
        m_in = np.abs(r_tr) < 3.5 * (np.std(r_tr) + 1e-9)
        if m_in.sum() > 50:
            coef = ridge_fit(A_o[m_in], yo[m_in])

        def trend(X):
            return np.c_[np.ones(len(X)), (X - Xo.mean(0)) / 1000.0] @ coef
        ro = yo - trend(Xo)
        # 自トラック(全行) rot座標・グリッド原点=トラック中央
        Xw = np.c_[d["x"], d["y"]]
        Rw = rot(Xw); Ro = rot(Xo)
        a0 = np.nanmedian(Rw[:, 0]) - NA * CELL / 2
        c0 = np.nanmedian(Rw[:, 1]) - NC * CELL / 2
        ia_w = np.floor((Rw[:, 0] - a0) / CELL).astype(int)
        ic_w = np.floor((Rw[:, 1] - c0) / CELL).astype(int)
        inb_w = (ia_w >= 0) & (ia_w < NA) & (ic_w >= 0) & (ic_w < NC)
        ia_o = np.floor((Ro[:, 0] - a0) / CELL).astype(int)
        ic_o = np.floor((Ro[:, 1] - c0) / CELL).astype(int)
        inb_o = (ia_o >= 0) & (ia_o < NA) & (ic_o >= 0) & (ic_o < NC)
        # burn 近傍
        S = np.zeros((NC, NA), np.float32); Cn = np.zeros((NC, NA), np.float32)
        np.add.at(S, (ic_o[inb_o], ia_o[inb_o]), np.clip(ro[inb_o], -60, 60))
        np.add.at(Cn, (ic_o[inb_o], ia_o[inb_o]), 1.0)
        burn = np.where(Cn > 0, S / np.maximum(Cn, 1), 0.0)
        mask = (Cn > 0).astype(np.float32)
        dist = distance_transform_edt(mask < 0.5) * CELL / 1000.0
        # prefix burn(tin基準, b0整列)
        ki = np.where(d["kn"])[0]
        tl = ki[(ki >= t0w - 2500) & (d["dh"][ki] > 0.5)]
        rp_raw = (d["tin"][tl] + d["z"][tl]) - trend(Xw[tl])
        tpn = cKDTree(Xo)
        _, jn = tpn.query(Xw[tl][::4], k=min(24, len(Xo)))
        b0 = float(np.median(rp_raw) - np.median(ro[np.asarray(jn).ravel()]))
        Sp = np.zeros((NC, NA), np.float32); Cp = np.zeros((NC, NA), np.float32)
        okp = inb_w[tl] & np.isfinite(rp_raw)
        np.add.at(Sp, (ic_w[tl][okp], ia_w[tl][okp]), np.clip((rp_raw - b0)[okp], -60, 60))
        np.add.at(Cp, (ic_w[tl][okp], ia_w[tl][okp]), 1.0)
        pburn = np.where(Cp > 0, Sp / np.maximum(Cp, 1), 0.0)
        pmask = (Cp > 0).astype(np.float32)
        # target: 自トラックANCC残差(全行)
        anc_own_s = F.lowpass(np.where(np.isfinite(d["anc"]), d["anc"], np.nan), 201)
        r_own = anc_own_s - trend(Xw)
        St = np.zeros((NC, NA), np.float32); Ct = np.zeros((NC, NA), np.float32)
        Wt = np.zeros((NC, NA), np.float32)
        okt = inb_w & np.isfinite(r_own)
        np.add.at(St, (ic_w[okt], ia_w[okt]), np.clip(r_own[okt], -80, 80))
        np.add.at(Ct, (ic_w[okt], ia_w[okt]), 1.0)
        evrow = d["ev"] & okt
        np.add.at(Wt, (ic_w[evrow], ia_w[evrow]), 1.0)
        tgt = np.where(Ct > 0, St / np.maximum(Ct, 1), 0.0)
        tmask = (Ct > 0).astype(np.float32)
        tw = np.where(Wt > 0, 1.0, 0.3) * tmask       # 評価帯セルw1.0/prefixセルw0.3
        X4 = np.stack([burn / 30.0, mask, np.clip(dist, 0, 4), pburn / 30.0], 0)
        pack[w + "_x"] = X4.astype(np.float16)
        pack[w + "_t"] = (tgt / 30.0).astype(np.float16)
        pack[w + "_w"] = tw.astype(np.float16)
        pack[w + "_pm"] = pmask.astype(np.float16)
        # 評価用: 行→セル対応と再構成材料
        meta[w] = dict(b0=b0, coef=coef.tolist(), xmean=Xo.mean(0).tolist(),
                       theta=float(theta), a0=float(a0), c0=float(c0))
        pack[w + "_rows"] = np.c_[np.arange(n), ia_w, ic_w, inb_w.astype(int)][
            d["ev"] & np.isfinite(d["tvt"])].astype(np.int32)
        pack[w + "_truth"] = d["tvt"][d["ev"] & np.isfinite(d["tvt"])].astype(np.float32)
        pack[w + "_z"] = d["z"][d["ev"] & np.isfinite(d["tvt"])].astype(np.float32)
        pack[w + "_trend_ev"] = trend(Xw[d["ev"] & np.isfinite(d["tvt"])]).astype(np.float32)
        if len(meta) % 100 == 0:
            print(f"  build {len(meta)}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(CACHE, **pack)
    json.dump(meta, open(os.path.join(ART, "inpaint_meta.json"), "w"))
    print(f"cache saved: {len(meta)} wells  ({time.time()-t0:.0f}s)")

def train():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    z = np.load(CACHE)
    meta = json.load(open(os.path.join(ART, "inpaint_meta.json")))
    ws = sorted(meta.keys())
    print(f"wells={len(ws)}", flush=True)
    gi = np.arange(len(ws))
    folds = list(GroupKFold(5).split(gi, gi, gi))
    fold_list = range(5) if FOLD < 0 else [FOLD]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def get(w):
        return (z[w + "_x"].astype(np.float32), z[w + "_t"].astype(np.float32),
                z[w + "_w"].astype(np.float32))

    class Block(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.c1 = nn.Conv2d(ci, co, 3, padding=1)
            self.c2 = nn.Conv2d(co, co, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(co); self.bn2 = nn.BatchNorm2d(co)
        def forward(self, x):
            x = torch.relu(self.bn1(self.c1(x)))
            return torch.relu(self.bn2(self.c2(x)))

    class UNet(nn.Module):
        def __init__(self, C=4, base=32):
            super().__init__()
            self.e1 = Block(C, base); self.e2 = Block(base, base * 2)
            self.e3 = Block(base * 2, base * 4); self.e4 = Block(base * 4, base * 8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
            self.d3 = Block(base * 8, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
            self.d2 = Block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
            self.d1 = Block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)[:, 0]

    all_es = []
    OOF_ROWS, OOF_PRED = {}, {}
    for fold in fold_list:
        tr_w = [ws[i] for i in folds[fold][0]]
        va_w = [ws[i] for i in folds[fold][1]]
        preds_sum = {w: None for w in va_w}
        for sd in range(SEEDS):
            torch.manual_seed(sd)
            net = UNet().to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            rng = np.random.default_rng(sd)
            for ep in range(EPOCHS):
                net.train()
                order = rng.permutation(len(tr_w))
                tot, nb = 0.0, 0
                for b0i in range(0, len(order), BS):
                    xs, ts, ms = [], [], []
                    for i in order[b0i:b0i + BS]:
                        X, T, Wm = get(tr_w[i])
                        if rng.random() < 0.5:           # along flip
                            X = X[:, :, ::-1].copy(); T = T[:, ::-1].copy(); Wm = Wm[:, ::-1].copy()
                        if rng.random() < 0.5:           # across flip
                            X = X[:, ::-1, :].copy(); T = T[::-1, :].copy(); Wm = Wm[::-1, :].copy()
                        xs.append(X); ts.append(T); ms.append(Wm)
                    xb = torch.from_numpy(np.stack(xs)).to(dev)
                    tb = torch.from_numpy(np.stack(ts)).to(dev)
                    mb = torch.from_numpy(np.stack(ms)).to(dev)
                    pred = net(xb)
                    loss = (nn.functional.huber_loss(pred, tb, delta=1.0,
                                                     reduction="none") * mb).sum() / mb.sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss.detach()); nb += 1
                sch.step()
                if (ep + 1) % 10 == 0:
                    print(f"fold{fold} s{sd} ep{ep+1}  loss={tot/nb:.4f}  "
                          f"({time.time()-t0:.0f}s)", flush=True)
            net.eval()
            with torch.no_grad():
                for w in va_w:
                    X, _, _ = get(w)
                    p = net(torch.from_numpy(X[None]).to(dev))[0].cpu().numpy() * 30.0
                    preds_sum[w] = p if preds_sum[w] is None else preds_sum[w] + p
        # fold評価(+OOF保存: 行indexとTVT予測)
        es_f = []
        for w in va_w:
            p = preds_sum[w] / SEEDS
            rows = z[w + "_rows"]; truth = z[w + "_truth"]
            zc = z[w + "_z"]; trend_ev = z[w + "_trend_ev"]
            b0 = meta[w]["b0"]
            ok = rows[:, 3] == 1
            anc_hat = p[rows[ok, 2], rows[ok, 1]] + trend_ev[ok]
            tvt_hat = anc_hat + b0 - zc[ok] + 0.0
            OOF_ROWS[w] = rows[ok, 0].astype(np.int32)
            OOF_PRED[w] = tvt_hat.astype(np.float32)
            e = tvt_hat - truth[ok]
            e = e[np.isfinite(e)]
            if len(e):
                es_f.append(e)
        rm = float(np.sqrt(np.mean(np.concatenate(es_f) ** 2)))
        print(f"fold{fold}: overall={rm:.4f}  (val {len(va_w)}井)", flush=True)
        all_es.append(np.concatenate(es_f))
    ov = float(np.sqrt(np.mean(np.concatenate(all_es) ** 2)))
    print(f"\n[Phase F inpaint] overall = {ov:.4f}  (参考: dipエンジン系 10.405)")
    if len(OOF_ROWS) > 500:
        pack = {}
        for w in OOF_ROWS:
            pack[w + "__rows"] = OOF_ROWS[w]; pack[w + "__pred"] = OOF_PRED[w]
        np.savez_compressed(os.path.join(ART, "inpaint_oof.npz"), **pack)
        print("inpaint OOF saved:", len(OOF_ROWS))
    json.dump(dict(overall=ov, epochs=EPOCHS, seeds=SEEDS),
              open(os.path.join(ART, "phaseF_result.json"), "w"))

if __name__ == "__main__":
    if MODE == "build":
        build_cache()
    else:
        train()
