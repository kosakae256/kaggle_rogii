# -*- coding: utf-8 -*-
"""v103 blend重み + 後処理(expansion較正)を OOF で fit → blend_v103.json(mode別)。
v100 と同一の後処理: p → g·p·(1+k|p|/s)(δ を非線形 de-shrink, 平均化で縮んだ振幅を戻す)。CV に ~0.02 効く。
★推論結果ごとに最適 gain が違う → base(決定版L5)と strong(cross-group ens)で別々に fit。TCN は det 学習を共有。
出力: {"base": {bw,gain_g,gain_k,gain_s,rmse,rmse_raw}, "strong": {...}}。
env: TCN_OOF(既定 自動検出) / GAIN_S(既定 10.0 固定=v100準拠, k と冗長なので s は固定して g,k を fit)
実行: .venv/Scripts/python.exe v103/fit_blend_v103.py"""
import os, sys, io, json, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts_v103"
GAIN_S = float(os.environ.get("GAIN_S", "10.0"))


def _rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def _load_l5(path):
    d = np.load(path)
    y = d["y"]
    l5 = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
    return l5.astype(np.float64), y.astype(np.float64)


def _load_tcn():
    cand = os.environ.get("TCN_OOF") or ""
    if not cand:
        for g in sorted(glob.glob(str(ART / "*oof*.npz"))):
            d = np.load(g)
            if any(k.startswith("mdn2_s") for k in d.files):
                cand = g; break
    if not cand or not os.path.exists(cand):
        return None
    d = np.load(cand)
    keys = [k for k in d.files if k.startswith("mdn2_s")]
    tcn = np.mean([d[k] for k in keys], 0) if keys else (d["nn_mean"] if "nn_mean" in d.files else None)
    print(f"[tcn oof] {os.path.basename(cand)} ({len(keys)} seed-fold)")
    return None if tcn is None else tcn.astype(np.float64)


def _fit_gain(p, y):
    """expansion 較正 g·p·(1+k|p|/s) を (g,k) について最小二乗(s は GAIN_S 固定)。"""
    from scipy.optimize import minimize
    m = np.isfinite(p) & np.isfinite(y); pp = p[m]; yy = y[m]
    def loss(th):
        g, k = th
        q = g * pp * (1.0 + k * np.abs(pp) / GAIN_S)
        return np.mean((q - yy) ** 2)
    r = minimize(loss, [1.02, 0.02], method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-9, maxiter=4000))
    g, k = float(r.x[0]), float(r.x[1])
    q = g * p * (1.0 + k * np.abs(p) / GAIN_S)
    return g, k, _rmse(q, y)


def _fit_bw(tcn, l5, y):
    if tcn is None:
        return [0.0, 1.0], l5
    ws = np.linspace(0.0, 1.0, 201)
    sc = [_rmse(w * tcn + (1 - w) * l5, y) for w in ws]
    w = float(ws[int(np.argmin(sc))])
    return [w, 1.0 - w], w * tcn + (1 - w) * l5


def fit_mode(name, l5, y, tcn):
    bw, blended = _fit_bw(tcn, l5, y)
    raw = _rmse(blended, y)
    g, k, r = _fit_gain(blended, y)
    print(f"[{name}] bw={bw} raw={raw:.4f} → +後処理 g={g:.4f} k={k:.4f} s={GAIN_S} : {r:.4f}  (Δ{r-raw:+.4f})")
    return {"bw": bw, "gain_g": g, "gain_k": k, "gain_s": GAIN_S, "rmse": r, "rmse_raw": raw}


def main():
    tcn = _load_tcn()
    out = {}
    # base = 決定版 GroupKFold 5-fold L5
    if (ART / "l5_v103all_oof.npz").exists():
        l5, y = _load_l5(ART / "l5_v103all_oof.npz")
        out["base"] = fit_mode("base", l5, y, tcn)
    # strong = cross-group ensemble(全 grid oof の l5_mean 平均)
    gfs = sorted(glob.glob(str(ART / "l5_v103g*k*_oof.npz")))
    if gfs:
        ys = None; preds = []
        for f in gfs:
            l5, y = _load_l5(f); ys = y if ys is None else ys; preds.append(l5)
        l5e = np.nanmean(np.stack(preds, 0), 0)
        out["strong"] = fit_mode(f"strong({len(gfs)}構成)", l5e, ys, tcn)
    (ART / "blend_v103.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[blend_v103.json] {list(out.keys())} 書き出し(TCN={'有' if tcn is not None else '無=L5単独'})")


if __name__ == "__main__":
    main()
