# -*- coding: utf-8 -*-
"""現 band-pass PF(v94_pf_3.json, tuner中の現best)を全井走行し、PF#1/PF#2との相関＋
   3本ブレンド＋oracle を測る。★全別シード(PF1=777/PF2=777/bp=271828, bpは別なので相関genuine)。
   bp engine使用。出力: bp_blend.txt"""
import os, sys, io, json, pickle
os.environ["GR_BANDPASS_TVT"] = "1"; os.environ.setdefault("BP_SLO_FT", "2"); os.environ.setdefault("BP_SHI_FT", "32")
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_engine_v94_bp as pfb
NS = int(os.environ.get("NS", "32")); CHUNK = int(os.environ.get("CHUNK", "60"))
ART = pfb.ART
e1d = pickle.loads((ART / "pf_1_errs.pkl").read_bytes())
e2d = pickle.loads((ART / "pf_2_errs.pkl").read_bytes())
bp = json.load(open(ART / "v94_pf_3.json", encoding="utf-8"))
print(f"[bp blend] bp PF: RMSE(50井)={bp['rmse']:.4f} corr(tuning)={bp['corrs']} -> 全井@271828走行", flush=True)
bpp = bp["best_params"]
ALLW = sorted(set(w for w in pfb.list_wells() if pfb.CACHE.setdefault(w, pfb.build_cache(w)) is not None) & set(e1d.keys()))
e3d = pfb.predict_errs(bpp, ALLW, seed=271828, n_seeds=NS, chunk=CHUNK)

E1, E2, E3 = [], [], []
for w in ALLW:
    a = np.asarray(e1d[w], np.float64); b = np.asarray(e2d[w], np.float64); c = np.asarray(e3d[w], np.float64)
    m = min(len(a), len(b), len(c)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m]) & np.isfinite(c[:m])
    E1.append(a[:m][msk]); E2.append(b[:m][msk]); E3.append(c[:m][msk])
E1 = np.concatenate(E1); E2 = np.concatenate(E2); E3 = np.concatenate(E3)
E = np.stack([E1, E2, E3], 1)
r = [np.sqrt(np.mean(E[:, i] ** 2)) for i in range(3)]
rho = np.corrcoef(E.T)


def minvar(cov):
    one = np.ones(cov.shape[0]); ci = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))
    w = ci @ one / (one @ ci @ one); return w, float(np.sqrt(max(0, w @ cov @ w)))


C2 = (E[:, :2].T @ E[:, :2]) / len(E); C3 = (E.T @ E) / len(E)
w2, r2 = minvar(C2); w3, r3 = minvar(C3)
# per-well oracle
allrows = {0: [], 1: [], 2: []}; wsel = []; orac_row = []
i0 = 0
# 井ごとに再取得(選択内訳用)
pw = []
off = 0
for w in ALLW:
    a = np.asarray(e1d[w], np.float64); b = np.asarray(e2d[w], np.float64); c = np.asarray(e3d[w], np.float64)
    m = min(len(a), len(b), len(c)); msk = np.isfinite(a[:m]) & np.isfinite(b[:m]) & np.isfinite(c[:m])
    A, B, Cc = a[:m][msk], b[:m][msk], c[:m][msk]
    if len(A) == 0:
        continue
    M = np.stack([A, B, Cc], 1)
    orac_row.append(M[np.arange(len(M)), np.argmin(np.abs(M), 1)])
    rk = [np.sqrt(np.mean(A ** 2)), np.sqrt(np.mean(B ** 2)), np.sqrt(np.mean(Cc ** 2))]
    best = int(np.argmin(rk)); wsel.append(best); pw.append([A, B, Cc][best])
r_roworacle = np.sqrt(np.mean(np.concatenate(orac_row) ** 2))
r_welloracle = np.sqrt(np.mean(np.concatenate(pw) ** 2))
cnt = np.bincount(wsel, minlength=3)

out = ["=== band-pass PF ブレンド/相関 (★別シード PF1=777/PF2=777/bp=271828) ==="]
out.append(f"  単体RMSE: PF#1(off)={r[0]:.4f}  PF#2(on)={r[1]:.4f}  bp={r[2]:.4f}")
out.append(f"  ★bp相関: bp-PF1={rho[0,2]:.3f}  bp-PF2={rho[1,2]:.3f}   (PF1-PF2={rho[0,1]:.3f})")
out.append(f"  2本(PF1+PF2) min-var = {r2:.4f}")
out.append(f"  3本(+bp)      min-var = {r3:.4f}  重み={tuple(round(x,2) for x in w3)}")
out.append(f"  ★bp追加の限界利得 = {r3-r2:+.4f}")
out.append(f"  per-well oracle = {r_welloracle:.4f}  (選択 PF1/PF2/bp = {cnt.tolist()}井)")
out.append(f"  per-row  oracle = {r_roworacle:.4f}")
txt = "\n".join(out); print("\n" + txt, flush=True)
open(ART / "bp_blend.txt", "w", encoding="utf-8").write(txt)
