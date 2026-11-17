# -*- coding: utf-8 -*-
"""
v50 Stage C0 — ヘッジ代の実測(go/no-go)
==========================================
combo デコーダの posterior に対して「井戸ごとに正しい枝へコミットできたら何点か」を測る:
  base    : posterior mean(現行 = 7.778)
  oracleA : 井戸定数シフト s∈[-20,20]ft を真値で最適化(自由度1/井戸の上限)
  oracleB : モード条件付き事後平均 E[δ | |δ−(δ̂+s)|≤6ft] の s を真値で最適化
            (=枝を当てられた場合に posterior 自身の局所質量で決まる予測)
oracle が大きく下がる=生成+検証(Stage C)に金脈がある。
実行: BTAG=combo REFH=1 DILT=1,2,4,8,16 BCH=96 python v50/stageC0_v50.py
"""
import os, sys, json, math, pickle
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageB_v50 as sb
import stageA_v50 as sa

DEV = sb.DEV
offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
offs_np = offs_t.cpu().numpy()
SH = np.arange(-20.0, 20.01, 0.5)          # 井戸シフト候補
MW = 12                                    # モード窓 ±12bin = 6ft

res = {k: [] for k in ["base", "orA", "orB"]}
per_well = []
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
            P = torch.softmax(lg[0, :T], dim=-1)                    # (nb,181)
            dh = (P * offs_t[None, :]).sum(-1)
            if rf is not None: dh = dh + rf[0, :T]
            dh = dh.cpu().numpy(); Pn = P.cpu().numpy()
            y = e["y_ev"]; st = e["st"]; bid = e["bid"]
            fin = np.isfinite(y)
            zb = dh                                                  # (nb,) posterior mean δ
            err0 = (st + zb[bid]) - y
            res["base"].append(err0[fin])
            # oracleA: 井戸定数シフト
            eA = err0[fin]
            rA = [np.sqrt(np.mean((eA + s) ** 2)) for s in SH]
            sA = SH[int(np.argmin(rA))]
            res["orA"].append(eA + sA)
            # oracleB: モード条件付き事後平均(シフトsで条件窓を移動)
            best = (1e18, None)
            kk = np.round((zb[:, None] + SH[None, :]) / sa.STEP).astype(np.int64) + sa.NB  # (nb,S)
            for si, s in enumerate(SH):
                ki = np.clip(kk[:, si], 0, 180)
                lo = np.maximum(ki - MW, 0); hi = np.minimum(ki + MW + 1, 181)
                dcond = np.empty(len(zb))
                for t in range(len(zb)):                              # nb~300: ループ許容
                    pw = Pn[t, lo[t]:hi[t]]
                    swt = pw.sum()
                    dcond[t] = (pw * offs_np[lo[t]:hi[t]]).sum() / max(swt, 1e-12)
                eB = ((st + dcond[bid]) - y)[fin]
                r = np.sqrt(np.mean(eB ** 2))
                if r < best[0]: best = (r, eB)
            res["orB"].append(best[1])
            per_well.append(dict(w=w, base=float(np.sqrt(np.mean(err0[fin] ** 2))),
                                 orA=float(np.sqrt(np.mean(res["orA"][-1] ** 2))),
                                 orB=float(np.sqrt(np.mean(res["orB"][-1] ** 2)))))
        print(f"fold{f} done", flush=True)

out = {}
for k in res:
    ee = np.concatenate(res[k]); out[k] = float(np.sqrt(np.mean(ee ** 2)))
imp = [pw["base"] - pw["orB"] for pw in per_well]
out["wells_gain_gt2"] = int(sum(1 for v in imp if v > 2.0))
out["wells_gain_gt5"] = int(sum(1 for v in imp if v > 5.0))
print("\n=== v50 Stage C0: ヘッジ代の実測(全773坑井, comboデコーダ) ===")
print(f"  base(posterior mean)          : {out['base']:.4f}")
print(f"  oracleA(井戸定数シフト)        : {out['orA']:.4f}")
print(f"  oracleB(モード条件付き事後平均): {out['orB']:.4f}")
print(f"  oracleBで >2ft 改善する坑井: {out['wells_gain_gt2']} / >5ft: {out['wells_gain_gt5']}")
json.dump(dict(**out, per_well=per_well), open(sb.ART / "stageC0_results.json", "w"), indent=1)
print("saved -> stageC0_results.json")
