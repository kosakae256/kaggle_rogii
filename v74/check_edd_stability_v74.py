# -*- coding: utf-8 -*-
"""e(dd)曲線の環境安定性検証: LOO環境 vs デプロイ環境(field 20%間引き)"""
import io, sys, os, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np

ART = r"c:\Users\kosaka256\Documents\rogii_claude\v74\artifacts_v74"
gz = np.load(os.path.join(ART, "phaseG_dump.npz"))
resA = json.load(open(os.path.join(ART, "phaseG_result.json")))["per_well"]
wsA = sorted(w for w in resA if w + "__dd" in gz.files)
ddA = np.array([float(np.median(gz[w + "__dd"])) for w in wsA])
nA = np.array([len(gz[w + "__dd"]) for w in wsA], float)
eA = np.array([resA[w] for w in wsA])
ph = json.load(open(os.path.join(ART, "phaseH_result.json")))["per_well"]
wsD = sorted(ph.keys())
ddD = np.array([ph[w][0] for w in wsD])
eD = np.array([ph[w][1] for w in wsD])
nD = np.array([ph[w][2] for w in wsD], float)
bins = [0, 300, 600, 900, 1200, 1800, 3000, 1e9]
print("e(dd)曲線の環境安定性: [ビン] LOO環境 vs デプロイ環境(条件付きRMSE)")
for a, b in zip(bins[:-1], bins[1:]):
    mA = (ddA >= a) & (ddA < b)
    mD = (ddD >= a) & (ddD < b)
    rA = float(np.sqrt(np.sum(nA[mA] * eA[mA] ** 2) / np.sum(nA[mA]))) if mA.sum() > 2 else float("nan")
    rD = float(np.sqrt(np.sum(nD[mD] * eD[mD] ** 2) / np.sum(nD[mD]))) if mD.sum() > 2 else float("nan")
    hi = b if b < 1e9 else 9999
    print(f"  {a:>5.0f}-{hi:>5.0f}ft: LOO {rA:6.2f} ({int(mA.sum()):3d}井)  deploy {rD:6.2f} ({int(mD.sum()):3d}井)")
tau = 2279.0
wD = np.exp(ddD / tau)
lb_pred = float(np.sqrt(np.sum(wD * nD * eD ** 2) / np.sum(wD * nD)))
print(f"deploy誤差 × dd重み(τ2279)の予想LB: {lb_pred:.3f}  (観測10.938)")
