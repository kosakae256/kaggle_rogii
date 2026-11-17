# -*- coding: utf-8 -*-
"""最終 A/B ブレンド + expansion 後処理を OOF から fit → blend_final.json。
A(v104無関係)= {L5grid, extR(anchor-only), TCN, anchor}
B(最良・v104少量)= A + greedy
各: 座標降下で重み最適化 → expansion(g,k, s=10固定)を Nelder-Mead。extRが効かなければ重み~0で自然に外れる。
出力: v104/artifacts_v104/blend_final.json = {A:{weights,g,k,s}, B:{weights,g,k,s}}
実行: .venv/Scripts/python.exe v104/fit_final.py"""
import os, sys, io, json, glob, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from scipy.optimize import minimize

PROJ = Path(__file__).resolve().parent.parent
V104A = PROJ / "v104" / "artifacts_v104"
EXT_OOF = V104A / "l5r_v104l5rdetanc_oof.npz"     # anchor-only L5R


def l5l(f):
    e = np.load(f); return e["l5_mean"] if "l5_mean" in e.files else np.mean([e[k] for k in e.files if k.startswith("l5_s")], 0)


def main():
    d = np.load(EXT_OOF); Y = d["y"]; N = len(Y); extR = d["l5r_mean"]
    def r(p):
        m = np.isfinite(p) & np.isfinite(Y); return float(np.sqrt(np.mean((p[m]-Y[m])**2)))
    def fitpp(p):
        mm = np.isfinite(p) & np.isfinite(Y); pp = p[mm]; yy = Y[mm]
        res = minimize(lambda t: np.mean((t[0]*pp*(1+t[1]*np.abs(pp)/10)-yy)**2), [1.02, .02], method="Nelder-Mead")
        g, k = res.x; return r(g*p*(1+k*np.abs(p)/10)), float(g), float(k)
    L5det = l5l(str(PROJ/"v103/artifacts_v103/l5_v103all_oof.npz"))
    L5grid = np.nanmean(np.stack([L5det]+[l5l(f) for f in glob.glob(str(PROJ/"v103/artifacts_v103/l5_v103g*k*_oof.npz"))], 0), 0)
    greedy = l5l(str(V104A/"l5_v104greedyg-1_oof.npz"))
    dt = np.load(PROJ/"v103/artifacts_v103/nn_oof_combo_fcombo.npz"); TCN = np.mean([dt[k] for k in dt.files if k.startswith("mdn2_s")], 0)
    Sm = pickle.loads((PROJ/"v97/artifacts_v97/sim_grfree_v97.pkl").read_bytes()); SF = pickle.loads((V104A/"seqfeat_v104_cf.pkl").read_bytes())
    lk = pq.read_table(PROJ/"v100/artifacts_v100/data/train_combo.parquet", columns=["last_known_tvt"]).column(0).to_numpy(zero_copy_only=False).astype(float)
    anc = np.full(N, np.nan)
    for w, dd in Sm.items():
        if w in SF and len(dd["st"]) == len(SF[w]["rows"]):
            anc[SF[w]["rows"]] = np.asarray(dd["st"], float) - lk[SF[w]["rows"]]

    ALL = {"L5grid": L5grid, "greedy": greedy, "extR": extR, "TCN": TCN, "anchor": anc}
    print("=== 各脚 単体OOF ===")
    for nm, a in ALL.items():
        print(f"  {nm:8s} {r(a):.4f}")

    def cd(names):
        M = [ALL[n] for n in names]; w = np.zeros(len(names)); w[0] = 1.0
        def bl(w):
            W = w/w.sum(); return sum(W[i]*M[i] for i in range(len(names)))
        for _ in range(600):
            best = (r(bl(w)), None)
            for i in range(len(names)):
                for dv in (0.01, -0.01, 0.03, -0.03, 0.06, -0.06):
                    w2 = w.copy(); w2[i] = max(0, w2[i]+dv)
                    if w2.sum() <= 0: continue
                    rr = r(bl(w2))
                    if rr < best[0]: best = (rr, (i, dv))
            if best[1] is None: break
            i, dv = best[1]; w[i] = max(0, w[i]+dv)
        W = w/w.sum(); p = bl(w); fpp, g, k = fitpp(p)
        return {n: float(wv) for n, wv in zip(names, W)}, r(p), fpp, g, k

    out = {}
    for mode, names in [("A", ["L5grid", "extR", "TCN", "anchor"]),
                        ("B", ["L5grid", "greedy", "extR", "TCN", "anchor"])]:
        W, raw, fpp, g, k = cd(names)
        out[mode] = {"weights": {n: round(v, 4) for n, v in W.items() if v > 0.003}, "g": round(g, 4), "k": round(k, 4), "s": 10.0}
        wtxt = " ".join(f"{n}={v:.3f}" for n, v in W.items() if v > 0.005)
        print(f"\n[{mode}] raw={raw:.4f} +後処理={fpp:.4f} (g={g:.4f},k={k:.4f})  {wtxt}")

    (V104A / "blend_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n保存 -> {V104A/'blend_final.json'}")


if __name__ == "__main__":
    main()
