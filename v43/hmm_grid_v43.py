# -*- coding: utf-8 -*-
"""
v43 PF-free グリッドHMM on GRマッチ画像(Viterbi / 前向き後ろ向き)
=====================================================================================
PFを一切使わず、GRマッチ費用の画像 C[t,d](縦=struct偏差d, 横=経路位置t)上の
動的計画法で経路を出す。物理prior=struct(v38 K16 dip場)を「錨帯」として画像に埋める。

  emission L[t,d] = -0.5·C[t,d]/σ_gr²  -  0.5·(d/sig_struct[t])²     (GR証拠 + struct prior)
  transition       ガウス平滑(偏差dの連続=dip連続, σ_trans[ft])
  推論  ① Viterbi(MAP経路)  ② 前向き後ろ向き(行ごと事後→posterior mean)
  出力  pred[t] = struct_tvt[t] + d*(t)

PF超えの理由: 厳密(粒子ロスなし・大域最適)/ bimodal datum を後ろ向きで大域整合選択・meanでヘッジ /
struct が正しいdatumへ画像を歪める。検証の厳密HMMスムーザ移植 = v40(学習emission=崩壊)の非学習・厳密版。

出力: v43/artifacts_v43/hmm_result.json + v43_hmm.png(overall/硬い核, MAP vs mean, σスイープ)
実行: .venv/Scripts/python.exe v43/hmm_grid_v43.py(GPU)
"""
import os, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import pyarrow.parquet as pq

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P / "rogii-wellbore-geology-prediction" / "train"
ART = P / "v43" / "artifacts_v43"; ART.mkdir(parents=True, exist_ok=True)
STRUCT = pickle.loads((P / "v38" / "artifacts_v38" / "struct_prior_train.pkl").read_bytes())
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DMAX, DSTEP = 40.0, 0.5
DGRID = np.arange(-DMAX, DMAX + 1e-6, DSTEP); G = len(DGRID)
HGR = 2
CHUNK = 24                       # 同時処理坑井数(padでVRAM)
print(f"[v43] device={DEV}  grid G={G} (±{DMAX}/{DSTEP})")


def robust_z(x):
    m = np.isfinite(x); med = np.nanmedian(x[m]); mad = np.nanmedian(np.abs(x[m] - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-6: mad = np.nanstd(x[m]) or 1.0
    return (x - med) / mad


def _smooth(v, r):
    return v if r <= 0 else pd.Series(v).rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()


# ---------- per-well emission画像を作る ----------
print("[v43] emission画像(GRコスト + struct prior)...")
t0 = time.time()
t = pq.read_table(P / "v28" / "artifacts_v28" / "data" / "train.parquet",
                  columns=["well", "target", "last_known_tvt", "md_since"]).to_pydict()
gw = np.array(t["well"]); lk = np.asarray(t["last_known_tvt"], float); tgt = np.asarray(t["target"], float)
mds = np.asarray(t["md_since"], float); true_all = lk + tgt
rows_of = {w: np.where(gw == w)[0] for w in pd.unique(gw)}
order_of = {w: rows_of[w][np.argsort(mds[rows_of[w]], kind="stable")] for w in rows_of}
try:
    hd = pd.read_csv(P / "v32" / "artifacts_v32" / "hardcore_diag.csv")
    HARD = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
except Exception:
    HARD = set()

WELLS, EMIT, STV, TRU = [], {}, {}, {}   # well -> C[t,d](GRコスト), struct偏差prior二乗, struct_tvt, true
SIGP = {}
for w in pd.unique(gw):
    sp = STRUCT.get(w)
    if sp is None: continue
    r = order_of[w]; stv = np.asarray(sp["tvt"], float); sig = np.asarray(sp["sig"], float)
    if len(stv) != len(r): continue
    try:
        hw = pd.read_csv(TRAIN / f"{w}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        tw = pd.read_csv(TRAIN / f"{w}__typewell.csv", usecols=["TVT", "GR"]).dropna().sort_values("TVT")
    except Exception:
        continue
    if len(tw) < 5: continue
    ev = ~np.isfinite(hw["TVT_input"].to_numpy())
    gr = pd.Series(hw["GR"].to_numpy(float)).interpolate(limit_direction="both").to_numpy()
    gr_ev = _smooth(gr, HGR)[ev]
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    z = robust_z(np.r_[tw_gr, gr_ev]); tw_z = z[:len(tw_gr)]; gr_z = z[len(tw_gr):]
    if len(gr_z) != len(r): continue
    cand = stv[:, None] + DGRID[None, :]                      # (T,G) 絶対TVT
    eg = np.interp(cand.ravel(), tw_tvt, tw_z).reshape(cand.shape)
    C = (gr_z[:, None] - eg) ** 2                             # ★GRマッチ画像 (T,G)
    WELLS.append(w); EMIT[w] = C.astype(np.float32)
    SIGP[w] = np.clip(sig, 3.0, 40.0).astype(np.float32)
    STV[w] = stv.astype(np.float32); TRU[w] = true_all[r].astype(np.float32)
print(f"      wells {len(WELLS)} ({time.time()-t0:.0f}s)")

d_t = torch.tensor(DGRID, dtype=torch.float32, device=DEV)


def tmat(sig_trans):
    dd = (d_t[None, :] - d_t[:, None]) / max(sig_trans / DSTEP * DSTEP, 1e-3)
    K = torch.exp(-0.5 * dd * dd); return K / K.sum(-1, keepdim=True).clamp_min(1e-20)


@torch.no_grad()
def run(sig_gr, sig_trans, lam_struct, mode="mean"):
    """全坑井を chunk 処理。mode: mean(前向き後ろ向き posterior mean)/ map(Viterbi)。"""
    Tm = tmat(sig_trans); logTm = torch.log(Tm.clamp_min(1e-20))
    out = {}
    for c0 in range(0, len(WELLS), CHUNK):
        sub = WELLS[c0:c0 + CHUNK]; B = len(sub)
        Tmax = max(EMIT[w].shape[0] for w in sub)
        Lem = torch.full((B, Tmax, G), -1e9, device=DEV)      # log emission
        msk = torch.zeros((B, Tmax), dtype=torch.bool, device=DEV)
        s0 = torch.zeros((B, G), device=DEV)
        for b, w in enumerate(sub):
            C = torch.tensor(EMIT[w], device=DEV); Tn = C.shape[0]
            sig = torch.tensor(SIGP[w], device=DEV)[:, None]
            L = -0.5 * C / (sig_gr ** 2) - 0.5 * lam_struct * (d_t[None, :] / sig) ** 2
            Lem[b, :Tn] = L; msk[b, :Tn] = True
            s0[b] = torch.softmax(-0.5 * (d_t / sig[0]) ** 2, -1).log()
        if mode == "map":
            # Viterbi(log)
            delta = s0 + Lem[:, 0]
            bp = torch.zeros((B, Tmax, G), dtype=torch.long, device=DEV)
            for tt in range(1, Tmax):
                cand = delta[:, :, None] + logTm[None]         # (B,G_prev,G_cur)
                m, arg = cand.max(1)
                delta = torch.where(msk[:, tt][:, None], m + Lem[:, tt], delta)
                bp[:, tt] = arg
            dev = torch.zeros((B, Tmax), device=DEV)
            idx = delta.argmax(-1)
            for tt in range(Tmax - 1, -1, -1):
                dev[:, tt] = d_t[idx]
                if tt > 0: idx = torch.gather(bp[:, tt], 1, idx[:, None])[:, 0]
        else:
            # 前向き後ろ向き(linear, 各行正規化)
            em = torch.exp(Lem - Lem.max(-1, keepdim=True).values); em = em * msk[:, :, None]
            a = torch.zeros((B, Tmax, G), device=DEV)
            al = torch.softmax(s0, -1) * em[:, 0]; al = al / al.sum(-1, keepdim=True).clamp_min(1e-20); a[:, 0] = al
            for tt in range(1, Tmax):
                pr = al @ Tm; nw = pr * em[:, tt]; nw = nw / nw.sum(-1, keepdim=True).clamp_min(1e-20)
                al = torch.where(msk[:, tt][:, None], nw, al); a[:, tt] = al
            be = torch.ones((B, G), device=DEV); g_ = torch.zeros((B, Tmax, G), device=DEV)
            g_[:, Tmax - 1] = a[:, Tmax - 1]
            for tt in range(Tmax - 2, -1, -1):
                be = (be * em[:, tt + 1]) @ Tm.T; be = be / be.sum(-1, keepdim=True).clamp_min(1e-20)
                gg = a[:, tt] * be; gg = gg / gg.sum(-1, keepdim=True).clamp_min(1e-20)
                g_[:, tt] = gg
            dev = (g_ * d_t[None, None, :]).sum(-1)
        dev = dev.cpu().numpy()
        for b, w in enumerate(sub):
            Tn = EMIT[w].shape[0]; out[w] = STV[w] + dev[b, :Tn]
    return out


def evalp(pred):
    ea, eh = [], []
    for w in WELLS:
        e = pred[w] - TRU[w]; e = e[np.isfinite(e)]; ea.append(e)
        if w in HARD: eh.append(e)
    A = np.concatenate(ea); H = np.concatenate(eh) if eh else np.array([np.nan])
    return float(np.sqrt(np.mean(A ** 2))), float(np.sqrt(np.nanmean(H ** 2)))


print("\n=== σスイープ(mode=mean, lam_struct=0.5)===")
best = None
for sg in [0.7, 1.0, 1.5]:
    for st in [1.0, 2.0, 4.0]:
        pr = run(sg, st, 0.5, "mean"); ov, hdh = evalp(pr)
        print(f"  σ_gr={sg} σ_trans={st}: overall={ov:.4f} hard={hdh:.4f}")
        if best is None or ov < best[0]: best = (ov, hdh, sg, st)
print(f"  → best mean overall={best[0]:.4f} hard={best[1]:.4f} (σ_gr={best[2]} σ_trans={best[3]})")

print("\n=== MAP(Viterbi) vs mean / lam_struct ===")
sg, st = best[2], best[3]; rec = {}
for lam in [0.0, 0.5, 1.0]:
    pm = run(sg, st, lam, "mean"); ovm, hdm = evalp(pm)
    pv = run(sg, st, lam, "map");  ovv, hdv = evalp(pv)
    rec[f"lam{lam}"] = dict(mean=[ovm, hdm], map=[ovv, hdv])
    print(f"  lam_struct={lam}: mean overall={ovm:.4f} hard={hdm:.4f} | MAP overall={ovv:.4f} hard={hdv:.4f}")

(ART / "hmm_result.json").write_text(json.dumps(dict(
    best_mean=dict(overall=best[0], hard=best[1], sig_gr=best[2], sig_trans=best[3]),
    map_vs_mean=rec, n_wells=len(WELLS), n_hard=sum(1 for w in WELLS if w in HARD),
    grid=dict(DMAX=DMAX, DSTEP=DSTEP)), indent=1))
print(f"\n参考: struct単体≈9.7 / pf_1≈9.12 / v38錨PF≈8.13。v43(PF-free)がこの帯に入れば別系統エンジン成立。")
print(f"saved {ART/'hmm_result.json'}")
