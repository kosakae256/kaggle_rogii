# -*- coding: utf-8 -*-
"""
v51 探索2 — 相互擬似ラベル: 確信井戸の予測パスを近傍の nbr 証拠に注入して再デコード
=====================================================================================
M1(出力の空間結合)は死 → 証拠レベルで証人を増やす。
  1) combo の井戸別予測 z(全行)と確信度(シフトプロファイルの s=0 卓越 margin)を計算
  2) 確信上位の井戸の (z, GR) eval行を「擬似既知区間」として近傍の nbr グリッドに追加
  3) nbrチャネル(ch8/9)だけ差し替えて同じ combo デコーダで再デコード(再学習なし)
  4) 全体/margin下位(硬い側)での変化を測定
実行: python v51/explore2_pseudonbr_v51.py
"""
import os, sys, json, pickle, time
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "v50"))
import stageB_v50 as sb
import stageA_v50 as sa

DEV = sb.DEV
ART51 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v51\artifacts_v51")
PROF = pickle.loads((ART51 / "prof_v51.pkl").read_bytes())
E50 = dict(np.load(sb.ART / "v50_stageBcombo_errs.npz"))
data = pickle.loads(sa.CACHE.read_bytes())
WELLS = sorted(PROF.keys()); N = len(WELLS)
XY = np.array([data[w]["xy"] for w in WELLS])
SH = np.arange(-20.0, 20.01, 0.5)
CONF_Q = float(os.environ.get("CONF_Q", "0.4"))     # 確信上位 (1-CONF_Q) を注入元に
NBK = 8

# ---------- 確信度: プロファイルの s=0 卓越 margin ----------
margin = {}
for w in WELLS:
    pr = PROF[w]["prof"].astype(np.float64)
    near = pr[np.abs(SH) <= 1.0].max()
    far = pr[np.abs(SH) >= 4.0].max()
    margin[w] = float(near - far)
th = np.quantile(list(margin.values()), CONF_Q)
conf = {w: margin[w] >= th for w in WELLS}
print(f"確信井戸(margin≥{th:.3f}): {sum(conf.values())}/{N}")

# ---------- 予測 z(eval行)を復元(z = y + err) ----------
ZP = {}
for w in WELLS:
    e = sb.EVD[w]
    ZP[w] = e["y_ev"] + E50[w][:len(e["y_ev"])]

# ---------- 擬似nbrグリッドで L_nbr/H_nbr を再構築 → EV 差し替え再デコード ----------
from scipy.spatial import cKDTree
tree = cKDTree(XY)
_, idxs = tree.query(XY, k=NBK + 1)
offs = np.arange(-sa.NB, sa.NB + 1)
offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP

def nbr_points(j):
    """井戸jが提供する (tvt, gr) 点: 既知区間 + (確信なら)予測eval行。"""
    c = data[WELLS[j]]
    knm = (~c["ev"]) & np.isfinite(c["y"])
    ys = [c["y"][knm].astype(np.float64)]; gs = [c["graw"][knm].astype(np.float64)]
    if conf[WELLS[j]]:
        e = sb.EVD[WELLS[j]]
        zj = ZP[WELLS[j]]; fin = np.isfinite(e["y_ev"])
        gr_ev = c["graw"][c["ev"]][:len(zj)]
        ys.append(zj[fin]); gs.append(gr_ev[fin])
    return np.concatenate(ys), np.concatenate(gs)

res = {"before": [], "after": []}
hard = {"before": [], "after": []}
t0 = time.time()
with torch.no_grad():
    for f in range(5):
        net = sb.DecoderB().to(DEV)
        net.load_state_dict(torch.load(sb.ART / f"stageB_dec{sb.BTAG}_f{f}.pt", map_location=DEV))
        net.eval()
        for w in sb.WELLS[f::5]:
            if w not in PROF: continue
            i = WELLS.index(w)
            c = data[w]; e = sb.EVD[w]
            K = len(c["gg"]); gmin = c["gmin"]
            # 擬似nbrグリッド
            ny, ng = [], []
            for j in idxs[i, 1:]:
                a, b = nbr_points(j); ny.append(a); ng.append(b)
            ny = np.concatenate(ny); ng = np.concatenate(ng)
            nbrg = np.full(K, np.nan)
            kyn = np.round((ny - gmin) / sa.STEP).astype(np.int64)
            okn = (kyn >= 0) & (kyn <= K - 1)
            if okn.sum() >= 20:
                nsum = np.zeros(K); ncnt = np.zeros(K)
                np.add.at(nsum, kyn[okn], ng[okn]); np.add.at(ncnt, kyn[okn], 1)
                with np.errstate(invalid="ignore"):
                    nbrg = np.where(ncnt > 0, nsum / np.maximum(ncnt, 1), np.nan)
            st = c["st"].astype(np.float64)
            ks = np.round((st - gmin) / sa.STEP).astype(np.int64)
            cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
            okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
            nc = np.abs(c["g25"][:, None] - nbrg[cand]); has = np.isfinite(nc) & okc
            nc = np.where(has, nc, 60.0)
            T = int(c["ev"].sum()); nb_ = (T + sa.STRIDE - 1) // sa.STRIDE; bid = np.arange(T) // sa.STRIDE
            def bm(x):
                o = np.zeros((nb_,) + x.shape[1:]); cnt = np.zeros(nb_)
                np.add.at(o, bid, x); np.add.at(cnt, bid, 1)
                return o / np.maximum(cnt, 1)[(...,) + (None,) * (x.ndim - 1)]
            L_nbr = np.exp(-0.5 * (bm(nc) / sa.SIGG) ** 2); H_nbr = bm(has.astype(np.float64))
            # EV 差し替え
            EV2 = e["EV"].astype(np.float32).copy()
            EV2[8] = L_nbr.astype(np.float32); EV2[9] = H_nbr.astype(np.float32)
            for tag, ev_use in [("before", e["EV"].astype(np.float32)), ("after", EV2)]:
                x = torch.from_numpy(ev_use)[None].to(DEV)
                m = torch.ones(1, ev_use.shape[1], device=DEV)
                lg, rf = net(x, m)
                P = torch.softmax(lg[0], dim=-1)
                dh = (P * offs_t[None, :]).sum(-1)
                if rf is not None: dh = dh + rf[0]
                z = e["st"] + dh.cpu().numpy()[bid]
                err = z - e["y_ev"]; err = err[np.isfinite(err)]
                res[tag].append(err)
                if not conf[w]: hard[tag].append(err)
        print(f"fold{f} done ({time.time()-t0:.0f}s)", flush=True)

for tag in ["before", "after"]:
    ee = np.concatenate(res[tag]); hh = np.concatenate(hard[tag])
    print(f"[{tag}] overall={np.sqrt(np.mean(ee**2)):.4f}  低margin側({len(hard[tag])}井)={np.sqrt(np.mean(hh**2)):.4f}")
