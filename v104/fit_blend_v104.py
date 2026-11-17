# -*- coding: utf-8 -*-
"""v104 blend + 後処理(expansion較正)を OOF で fit。★TCN は v103 を流用(v104 TCN は学習しない=v103≈v100 TCN 実証済)。
L5 OOF = v104 の det GroupKFold OOF(env L5_OOF)。TCN OOF = v103 の nn_oof_combo_fcombo.npz。
出力: v104/artifacts_v104/blend_v104.json = {bw, gain_g, gain_k, gain_s, rmse, rmse_raw}。
env: L5_OOF(必須, v104 L5 OOF npz) / GAIN_S(既定10.0)
実行: L5_OOF=.../l5_v104final_oof.npz .venv/Scripts/python.exe v104/fit_blend_v104.py"""
import os, sys, io, json
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
ART = V104 / "artifacts_v104"
TCN_OOF = PROJ / "v103" / "artifacts_v103" / "nn_oof_combo_fcombo.npz"   # ★v103 TCN 流用
GAIN_S = float(os.environ.get("GAIN_S", "10.0"))


def rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y); return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    l5p = Path(os.environ["L5_OOF"])
    d = np.load(l5p); y = d["y"]
    l5 = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
    dt = np.load(TCN_OOF); tks = [k for k in dt.files if k.startswith("mdn2_s")]
    tcn = np.mean([dt[k] for k in tks], 0)
    print(f"[v104 blend] L5={l5p.name}({rmse(l5,y):.4f}) + TCN=v103流用({rmse(tcn,y):.4f}, {len(tks)}seed)", flush=True)
    # bw fit
    ws = np.linspace(0, 1, 201)
    bi = int(np.argmin([rmse(w * tcn + (1 - w) * l5, y) for w in ws])); w = float(ws[bi])
    bw = [w, 1 - w]; blended = w * tcn + (1 - w) * l5; raw = rmse(blended, y)
    # expansion (g,k) fit
    from scipy.optimize import minimize
    m = np.isfinite(blended) & np.isfinite(y); pp = blended[m]; yy = y[m]
    def loss(th): g, k = th; q = g * pp * (1 + k * np.abs(pp) / GAIN_S); return np.mean((q - yy) ** 2)
    r = minimize(loss, [1.02, 0.02], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-9, maxiter=4000))
    g, k = float(r.x[0]), float(r.x[1])
    fin = rmse(g * blended * (1 + k * np.abs(blended) / GAIN_S), y)
    out = {"bw": bw, "gain_g": g, "gain_k": k, "gain_s": GAIN_S, "rmse": fin, "rmse_raw": raw}
    (ART / "blend_v104.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[v104 blend] bw={bw} raw={raw:.4f} → +後処理 g={g:.4f} k={k:.4f}: {fin:.4f} (Δ{fin-raw:+.4f})", flush=True)
    print(f"  → blend_v104.json 書き出し。v103 strong={5.4577}/base={5.4650} と比較", flush=True)


if __name__ == "__main__":
    main()
