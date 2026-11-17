# -*- coding: utf-8 -*-
"""Stage B 行別エラー保存 + v38(seed777)ブレンド確認。"""
import os, sys, json, math, pickle
os.environ.setdefault("TAG","capaug"); os.environ.setdefault("CH","128")
os.environ.setdefault("DIM","96"); os.environ.setdefault("DILS","1,2,4,8,16,32")
import numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import stageB_v50 as sb
import stageA_v50 as sa

E = {}
offs_t = torch.arange(-sa.NB, sa.NB+1, device=sb.DEV).float()*sa.STEP
with torch.no_grad():
    for f in range(5):
        net = sb.DecoderB().to(sb.DEV)
        net.load_state_dict(torch.load(sb.ART/f"stageB_dec{sb.BTAG}_f{f}.pt", map_location=sb.DEV)); net.eval()
        for w in sb.WELLS[f::5]:
            e = sb.EVD[w]
            x,m,_,_ = sb.make_batch([w])
            lg, rf = net(x, m)
            P = torch.softmax(lg[0,:e["EV"].shape[1]], dim=-1)
            dh = (P*offs_t[None,:]).sum(-1)
            if rf is not None: dh = dh + rf[0,:e["EV"].shape[1]]
            dh = dh.cpu().numpy()
            E[w] = ((e["st"] + dh[e["bid"]]) - e["y_ev"]).astype(np.float32)
np.savez_compressed(sb.ART/f"v50_stageB{sb.BTAG}_errs.npz", **E)
ee = np.concatenate(list(E.values())); ee = ee[np.isfinite(ee)]
print(f"StageB 検算: {np.sqrt(np.mean(ee**2)):.4f}")
E38 = dict(np.load(sa.P0/"v38"/"artifacts_v38"/"v38_mult30_errs_seed777.npz"))
pa, pb = [], []
for w in sorted(set(E)&set(E38)):
    a,b = E[w].astype(np.float64), E38[w].astype(np.float64)
    n = min(len(a),len(b)); a,b = a[:n],b[:n]
    m = np.isfinite(a)&np.isfinite(b)
    pa.append(a[m]); pb.append(b[m])
A = np.concatenate(pa); B = np.concatenate(pb)
ws = np.arange(0,1.001,0.05)
cur = [float(np.sqrt(np.mean((w*B+(1-w)*A)**2))) for w in ws]
bi = int(np.argmin(cur))
print(f"blend StageB×v38: v50B単体={cur[0]:.4f} v38単体={cur[-1]:.4f} ★best w={ws[bi]:.2f} -> {cur[bi]:.4f}")
print(f"corr(row) = {np.corrcoef(A,B)[0,1]:.3f}")
json.dump(dict(curve=cur, ws=list(ws), best_w=float(ws[bi]), best=cur[bi]), open(sb.ART/"blendB_v50.json","w"))
