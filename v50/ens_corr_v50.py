# -*- coding: utf-8 -*-
"""
v50 capaug hybrid × v38錨PF(seed777, 正直) の単純アンサンブル + 他PFバンクとの誤差相関
=====================================================================================
1) capaug 5-foldエンコーダで全773坑井の行別エラーを再生成 → v50_capaug_errs.npz(保存・再利用)
2) blend: err = w·e_v38 + (1−w)·e_v50 を w=0..1 sweep → 全坑井プールRMSEカーブ
3) 相関: 井戸レベルbias(=行エラー平均)の相関を v50/v38/PFバンク7本(v47 lag16候補のtw)で計測
実行: python v50/ens_corr_v50.py
"""
import os, sys, json, time, math, pickle
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("TAU", "0.25")
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageA_v50 as sa                     # 定数/Enc1D/cache を流用(env設定後にimport)

P0  = sa.P0; ART = sa.ART; DEV = sa.DEV
T_TEMP, T_HW = 0.25, 0.5                    # capaug 確定較正
ERRS_FP = ART / "v50_capaug_errs.npz"

# ---------- 1) v50 capaug 行別エラー(無ければ生成) ----------
if not ERRS_FP.exists():
    data = pickle.loads(sa.CACHE.read_bytes())
    wells = sorted(data.keys())
    offs = np.arange(-sa.NB, sa.NB + 1)
    grid_t = torch.tensor(offs * sa.STEP, dtype=torch.float32, device=DEV)
    ND = len(offs)
    dd = grid_t[:, None] - grid_t[None, :]
    M = torch.exp(-0.5 * (dd / sa.TAU) ** 2); M = M / M.sum(0, keepdim=True).clamp_min(1e-30)
    def fb(obs_np, nb_):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEV)
        obs = obs / obs.sum(1, keepdim=True).clamp_min(1e-30)
        aF = torch.zeros(nb_, ND, device=DEV); aF[0] = obs[0]
        for i in range(1, nb_):
            pr = M @ aF[i - 1]; aF[i] = (pr.clamp_min(1e-30) * obs[i]); aF[i] = aF[i] / aF[i].sum().clamp_min(1e-30)
        bB = torch.zeros(nb_, ND, device=DEV); bB[-1] = 1.0 / ND
        for i in range(nb_ - 2, -1, -1):
            ms = bB[i + 1] * obs[i + 1]; bB[i] = (M @ ms).clamp_min(1e-30); bB[i] = bB[i] / bB[i].sum().clamp_min(1e-30)
        gam = aF * bB; gam = gam / gam.sum(1, keepdim=True).clamp_min(1e-30)
        return (gam * grid_t[None, :]).sum(1).cpu().numpy()
    out = {}; t0 = time.time()
    with torch.no_grad():
        for f in range(5):
            ck = torch.load(ART / f"stageA_enccapaug_f{f}.pt", map_location=DEV)
            encA = sa.Enc1D(8, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encA.load_state_dict(ck["encA"]); encA.eval()
            encB = sa.Enc1D(7, sa.CH_ENC, sa.DIM, sa.DILS).to(DEV); encB.load_state_dict(ck["encB"]); encB.eval()
            ls_ = math.exp(ck["logit_s"])
            for wn in wells[f::5]:
                c = data[wn]
                ev_idx = np.where(c["ev"])[0]; y_ev = c["y"][ev_idx].astype(np.float64)
                a = encA(torch.from_numpy(c["Fh"]).to(DEV))[ev_idx]
                b = encB(torch.from_numpy(c["Ft"]).to(DEV))
                K = len(c["gg"]); gmin = c["gmin"]
                st = c["st"].astype(np.float64); sg = np.clip(c["sg"].astype(np.float64), 1, 60)
                ks = np.round((st - gmin) / sa.STEP).astype(np.int64)
                cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
                okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
                sims = (a[:, None, :] * b[torch.tensor(cand, device=DEV)]).sum(-1).cpu().numpy()
                sims = np.where(okc, sims, -1e4)
                cost = np.abs(c["g25"][:, None] - c["gg"][cand]); cost[~okc] = 1e4
                T = len(ev_idx); nb_ = (T + sa.STRIDE - 1) // sa.STRIDE; bid = np.arange(T) // sa.STRIDE
                def bm(x):
                    o = np.zeros((nb_,) + x.shape[1:]); cnt = np.zeros(nb_)
                    np.add.at(o, bid, x); np.add.at(cnt, bid, 1)
                    return o / np.maximum(cnt, 1)[(...,) + (None,) * (x.ndim - 1)]
                sgb = np.clip(bm(sg), 1, 60); stb = bm(st)
                Pr = np.exp(-0.5 * ((offs * sa.STEP)[None, :] / sgb[:, None]) ** 2)
                L_raw0 = np.exp(-0.5 * (bm(cost) / sa.SIGG) ** 2)
                Prow = torch.softmax(torch.tensor(sims, dtype=torch.float32) * ls_ * T_TEMP, dim=1).numpy()
                Pb = bm(Prow)
                z = (stb + fb(L_raw0 * np.maximum(Pb, 1e-12) ** T_HW * Pr ** sa.LPR, nb_))[bid]
                out[wn] = (z - y_ev).astype(np.float32)
            print(f"  fold{f} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(ERRS_FP, **out)
    print(f"saved {ERRS_FP.name}: {len(out)} wells")
E50 = dict(np.load(ERRS_FP))
e = np.concatenate(list(E50.values())); e = e[np.isfinite(e)]
print(f"v50 capaug 単体(検算): {np.sqrt(np.mean(e**2)):.4f}")

# ---------- 2) v38(seed777, 正直) との blend sweep ----------
E38 = dict(np.load(P0 / "v38" / "artifacts_v38" / "v38_mult30_errs_seed777.npz"))
common = sorted(set(E50) & set(E38))
print(f"共通坑井: {len(common)}")
pairs = []
for wn in common:
    a, b = E50[wn].astype(np.float64), E38[wn].astype(np.float64)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.any(): pairs.append((a[m], b[m]))
A = np.concatenate([p[0] for p in pairs]); B = np.concatenate([p[1] for p in pairs])
ws = np.arange(0, 1.001, 0.05)
curve = [float(np.sqrt(np.mean((w * B + (1 - w) * A) ** 2))) for w in ws]
bi = int(np.argmin(curve))
print("\n=== blend: err = w·v38 + (1−w)·v50 (全坑井プール) ===")
print(f"  v50単体(w=0)   : {curve[0]:.4f}")
print(f"  v38単体(w=1)   : {curve[-1]:.4f}")
print(f"  ★best w={ws[bi]:.2f} : {curve[bi]:.4f}")
print(f"  50/50 (w=0.5)  : {curve[10]:.4f}")
print(f"  行レベル誤差相関 corr(v50, v38) = {np.corrcoef(A, B)[0,1]:.3f}")

# ---------- 3) 井戸レベルbias相関(v50 / v38 / PFバンク7本[lag16 tw]) ----------
z47 = np.load(P0 / "v47" / "artifacts_v47" / "candidates_v47.npz")
meta = json.loads((P0 / "v47" / "artifacts_v47" / "candidates_v47_meta.json").read_text(encoding="utf-8"))
w47 = meta["wells"]; cn = meta["cand_names"]
BANK = {"pf_1": "tw_b1", "pf_2": "tw_b2", "pf_3": "tw_b3", "r0": "tw_b4",
        "r1": "tw_b5", "r4": "tw_b6", "pfA": "tw_b7"}
DEL = z47["DEL"]; yv = z47["y"]; wc = z47["wellcode"]
bias = {}
order = np.argsort(wc, kind="stable")
wc_s = wc[order]
bounds = np.searchsorted(wc_s, np.arange(len(w47) + 1))
for name, col in BANK.items():
    k = cn.index(col); err = (DEL[:, k] - yv)[order]
    bias[name] = {w47[i]: float(np.nanmean(err[bounds[i]:bounds[i + 1]]))
                  for i in range(len(w47)) if bounds[i + 1] > bounds[i]}
bias["v50"] = {wn: float(np.nanmean(E50[wn])) for wn in common}
bias["v38"] = {wn: float(np.nanmean(E38[wn])) for wn in common}
names = ["v38", "pf_1", "pf_2", "pf_3", "r0", "r1", "r4", "pfA"]
print("\n=== 井戸レベル bias 相関(vs v50 capaug)===")
res_corr = {}
for nm in names:
    ws_c = [w for w in common if w in bias[nm]]
    x = np.array([bias["v50"][w] for w in ws_c]); yb = np.array([bias[nm][w] for w in ws_c])
    m = np.isfinite(x) & np.isfinite(yb)
    res_corr[nm] = float(np.corrcoef(x[m], yb[m])[0, 1])
    print(f"  v50 × {nm:5s}: {res_corr[nm]:+.3f}")
ref = {}
for nm in ["pf_1", "r0", "pfA"]:
    ws_c = [w for w in common if w in bias[nm]]
    x = np.array([bias["v38"][w] for w in ws_c]); yb = np.array([bias[nm][w] for w in ws_c])
    m = np.isfinite(x) & np.isfinite(yb)
    ref[nm] = float(np.corrcoef(x[m], yb[m])[0, 1])
print("  (参考) v38 × pf_1/r0/pfA:", " ".join(f"{k}={v:+.3f}" for k, v in ref.items()))

# ---------- plot ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax1.plot(ws, curve, "o-", color="#d62728")
ax1.scatter([ws[bi]], [curve[bi]], s=220, facecolors="none", edgecolors="#d62728", lw=2.5, zorder=5)
ax1.annotate(f"best w={ws[bi]:.2f}\n{curve[bi]:.3f}", (ws[bi], curve[bi]),
             textcoords="offset points", xytext=(0, 14), ha="center")
ax1.axhline(curve[0], ls=":", color="gray"); ax1.text(0.02, curve[0], f"v50単体 {curve[0]:.3f}", fontsize=8, va="bottom")
ax1.axhline(curve[-1], ls=":", color="gray"); ax1.text(0.7, curve[-1], f"v38単体 {curve[-1]:.3f}", fontsize=8, va="bottom")
ax1.set_xlabel("w(v38錨PFの重み)"); ax1.set_ylabel("全坑井プールRMSE")
ax1.set_title("v50 capaug hybrid × v38錨PF(seed777) 単純ブレンド"); ax1.grid(alpha=.3)
ax2.bar(range(len(names)), [res_corr[n] for n in names], color=["#4c4c4c"] + ["#1f77b4"] * 7)
ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names)
ax2.axhline(0, color="k", lw=0.8)
for i, n in enumerate(names): ax2.text(i, res_corr[n] + 0.01, f"{res_corr[n]:+.2f}", ha="center", fontsize=9)
ax2.set_ylim(0, 1); ax2.set_ylabel("井戸レベル bias 相関")
ax2.set_title("v50 capaug と各エンジンの誤差相関\n(低いほど独立=アンサンブル価値大)")
ax2.grid(axis="y", alpha=.3)
plt.tight_layout(); outp = P0 / "v50" / "ens_corr_v50.png"
plt.savefig(outp, dpi=130); print("\nsaved", outp)
json.dump(dict(blend_ws=list(ws), blend_curve=curve, best_w=float(ws[bi]), best=curve[bi],
               corr_v50=res_corr, corr_v38_ref=ref),
          open(ART / "ens_corr_v50.json", "w"), indent=1)
