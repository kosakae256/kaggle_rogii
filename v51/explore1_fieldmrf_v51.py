# -*- coding: utf-8 -*-
"""
v51 探索1 — フィールド同時推定(非学習MRF): 全773井のシフトを空間結合で同時決定
==================================================================================
v50 の残る2.6ft=井戸単位の枝コミット問題。単独井戸に枝の証人はいない(検証器全滅)が、
このデータは単一巨大フィールド → 近傍井戸の証拠を「独立の証人」として使う。
  unary   : U_i(s) = −logmass_i(s)(v50 comboデコーダのシフト質量プロファイル, s∈±20ft)
  pairwise: w_ij·(s_i−s_j)²/(2σ_s²)  (kNN近傍, w=1/(1+d/D0))
  最適化  : ICM(座標降下, init s=0)
  予測    : s_i* のモード条件付き事後平均パス → 全坑井プールRMSE
比較線: base 7.778(≒bin近似7.88)/ oracleB 5.17。v42の失敗(空間だけは死)の再検証を兼ねる。
実行: python v51/explore1_fieldmrf_v51.py
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
SH = np.arange(-20.0, 20.01, 0.5); S = len(SH)
MW = 12
offs_np = (np.arange(-sa.NB, sa.NB + 1) * sa.STEP).astype(np.float64)
PROF_FP = ART51 / "prof_v51.pkl"

# ---------- シフト質量プロファイル+posterior を全坑井で用意(キャッシュ) ----------
def build_prof():
    offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
    out = {}
    with torch.no_grad():
        for f in range(5):
            net = sb.DecoderB().to(DEV)
            net.load_state_dict(torch.load(sb.ART / f"stageB_dec{sb.BTAG}_f{f}.pt", map_location=DEV))
            net.eval()
            for w in sb.WELLS[f::5]:
                e = sb.EVD[w]
                x, m, _, _ = sb.make_batch([w])
                lg, rf = net(x, m)
                T = e["EV"].shape[1]
                P = torch.softmax(lg[0, :T], dim=-1)
                dh = (P * offs_t[None, :]).sum(-1)
                if rf is not None: dh = dh + rf[0, :T]
                dh = dh.cpu().numpy(); Pn = P.cpu().numpy().astype(np.float64)
                nb_ = len(dh)
                csum = np.concatenate([np.zeros((nb_, 1)), np.cumsum(Pn, 1)], 1)
                kk = np.round((dh[:, None] + SH[None, :]) / sa.STEP).astype(np.int64) + sa.NB
                lo = np.clip(kk - MW, 0, 181); hi = np.clip(kk + MW + 1, 0, 181)
                mass = np.take_along_axis(csum, hi, 1) - np.take_along_axis(csum, lo, 1)
                prof = np.log(np.maximum(mass, 1e-9)).mean(0)                 # (S,)
                out[w] = dict(prof=prof.astype(np.float32), dh=dh.astype(np.float32),
                              P=Pn.astype(np.float16))
            print(f"  prof fold{f} done", flush=True)
    PROF_FP.write_bytes(pickle.dumps(out, protocol=4))
    return out

PROF = pickle.loads(PROF_FP.read_bytes()) if PROF_FP.exists() else build_prof()
data = pickle.loads(sa.CACHE.read_bytes())
WELLS = sorted(PROF.keys()); N = len(WELLS)
XY = np.array([data[w]["xy"] for w in WELLS])
print(f"wells {N}")

# ---------- 予測(シフトsのモード条件付き事後平均)と評価素材 ----------
def path_at(w, s):
    p = PROF[w]; Pn = p["P"].astype(np.float64); dh = p["dh"].astype(np.float64)
    nb_ = len(dh)
    ki = np.clip(np.round((dh + s) / sa.STEP).astype(np.int64) + sa.NB, 0, 180)
    l2 = np.maximum(ki - MW, 0); h2 = np.minimum(ki + MW + 1, 181)
    d = np.empty(nb_)
    for t in range(nb_):
        pw = Pn[t, l2[t]:h2[t]]; sw = pw.sum()
        d[t] = (pw * offs_np[l2[t]:h2[t]]).sum() / max(sw, 1e-12)
    return d

def pooled_rmse(shifts):
    num = 0.0; den = 0.0
    for i, w in enumerate(WELLS):
        e = sb.EVD[w]
        z = e["st"] + path_at(w, shifts[i])[e["bid"]]
        err = z - e["y_ev"]; err = err[np.isfinite(err)]
        num += (err ** 2).sum(); den += len(err)
    return float(np.sqrt(num / den))

# ---------- MRF(ICM) ----------
from scipy.spatial import cKDTree
tree = cKDTree(XY)
KNN = 6
dists, idxs = tree.query(XY, k=KNN + 1)
dists, idxs = dists[:, 1:], idxs[:, 1:]
U = -np.stack([PROF[w]["prof"] for w in WELLS])          # (N,S) 小さいほど良い
U = (U - U.min(1, keepdims=True))

def icm(lmbda, sig_s, D0=2000.0, iters=8):
    Wgt = 1.0 / (1.0 + dists / D0)                        # (N,KNN)
    s_idx = np.full(N, S // 2, np.int64)                  # init s=0
    for it in range(iters):
        changed = 0
        for i in np.random.default_rng(it).permutation(N):
            sj = SH[s_idx[idxs[i]]]                       # 近傍の現シフト
            pair = ((SH[None, :] - sj[:, None]) ** 2 / (2 * sig_s ** 2) * Wgt[i][:, None]).sum(0)
            tot = U[i] + lmbda * pair
            new = int(np.argmin(tot))
            if new != s_idx[i]: changed += 1
            s_idx[i] = new
        if changed == 0: break
    return SH[s_idx], changed

print(f"\nbase(s=0, bin近似): {pooled_rmse(np.zeros(N)):.4f}")
print("unaryのみ(λ=0, 各井戸argmin prof):", end=" ")
s_un = SH[np.argmin(U, 1)]
print(f"{pooled_rmse(s_un):.4f}  (|s|>1ft: {(np.abs(s_un) > 1).sum()}井)")
for lmbda, sig_s in [(0.5, 5.0), (1.0, 5.0), (2.0, 5.0), (1.0, 10.0), (3.0, 8.0)]:
    t0 = time.time()
    s_star, _ = icm(lmbda, sig_s)
    r = pooled_rmse(s_star)
    print(f"MRF λ={lmbda:g} σs={sig_s:g}: {r:.4f}  (|s|>1ft: {(np.abs(s_star) > 1).sum()}井, {time.time()-t0:.0f}s)", flush=True)
