# -*- coding: utf-8 -*-
"""
v52 — v50B(学習デコーダ combo)特徴を train.parquet 行に整列して生成
====================================================================
3本エンジンの1本 = v50 StageB combo の「推論結果+作れる特徴」:
  v50b_delta  : 予測TVT − last_known(target同スケール)
  v50b_std    : δ事後の標準偏差(不確実性)
  v50b_margin : シフト質量プロファイルの s=0近傍 vs 遠方(≥4ft)最大の対数質量差(枝確信度)
  v50b_maxp   : 行binの最大事後確率の平均展開(尖り)
出力: v52/artifacts_v52/v50b_feature.npy [N,4] + v50b_meta.json(parquet 行順に整列)
実行: python v52/v50b_feature_v52.py   (先に create で train.parquet を生成しておく)
"""
import os, sys, json, math, pickle, time
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np
import torch
import pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "v50"))
import stageB_v50 as sb
import stageA_v50 as sa

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART52 = P0 / "v52" / "artifacts_v52"
PARQUET = ART52 / "data" / "train.parquet"
DEV = sb.DEV
SH = np.arange(-20.0, 20.01, 0.5); MW = 12
offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP

# ---------- 1) 井戸ごとの v50B 出力(bin) ----------
OUT_W = {}
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
            var = (P * (offs_t[None, :] - dh[:, None]) ** 2).sum(-1)
            dh = dh.cpu().numpy(); std = np.sqrt(var.cpu().numpy())
            Pn = P.cpu().numpy().astype(np.float64)
            maxp = Pn.max(1)
            # シフト質量プロファイル → margin(枝確信度, 井戸レベル)
            nb_ = len(dh)
            csum = np.concatenate([np.zeros((nb_, 1)), np.cumsum(Pn, 1)], 1)
            kk = np.round((dh[:, None] + SH[None, :]) / sa.STEP).astype(np.int64) + sa.NB
            lo = np.clip(kk - MW, 0, 181); hi = np.clip(kk + MW + 1, 0, 181)
            mass = np.take_along_axis(csum, hi, 1) - np.take_along_axis(csum, lo, 1)
            prof = np.log(np.maximum(mass, 1e-9)).mean(0)
            margin = float(prof[np.abs(SH) <= 1.0].max() - prof[np.abs(SH) >= 4.0].max())
            z_bin = e["st"] + dh[e["bid"]]                        # 行レベル予測TVT
            std_row = std[e["bid"]]; maxp_row = maxp[e["bid"]]
            OUT_W[w] = (z_bin.astype(np.float64), std_row.astype(np.float32),
                        margin, maxp_row.astype(np.float32))
        print(f"fold{f} done", flush=True)

# ---------- 2) parquet 行に整列(gf_feature と同じ流儀: id → well_rownum) ----------
tb = pq.read_table(PARQUET, columns=["id", "well", "last_known_tvt"])
_nw = tb.column("well").to_pandas().nunique()
if _nw < 700:
    raise SystemExit(f"[GUARD] train.parquet の坑井数が {_nw} 本しかありません(スモーク残骸の疑い)。"
                     f" 本番 create(python v52/create_train_parquet_v52.py, NWELLS未設定)を先に実行してください。")
ids = tb.column("id").to_pandas().astype(str)
wname = tb.column("well").to_pandas().astype(str).to_numpy()
lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
N = len(ids); F = np.full((N, 4), np.nan, np.float32)
import pandas as pd
miss = 0
for wn in pd.unique(wname):
    ii = np.where(wname == wn)[0]
    o = np.argsort(rn[ii], kind="stable"); iis = ii[o]
    d = OUT_W.get(wn)
    if d is None: miss += 1; continue
    z, sd, mg, mp = d
    n0 = min(len(iis), len(z))
    F[iis[:n0], 0] = (z[:n0] - lk[iis[:n0]]).astype(np.float32)   # v50b_delta
    F[iis[:n0], 1] = sd[:n0]                                       # v50b_std
    F[iis[:n0], 2] = mg                                            # v50b_margin(井戸一定)
    F[iis[:n0], 3] = mp[:n0]                                       # v50b_maxp
np.save(ART52 / "v50b_feature.npy", F)
(ART52 / "v50b_meta.json").write_text(json.dumps(dict(
    cols=["v50b_delta", "v50b_std", "v50b_margin", "v50b_maxp"], n=int(N), miss_wells=int(miss))))
fin = np.isfinite(F[:, 0])
tbl_y = pq.read_table(PARQUET, columns=["target"]).column("target").to_numpy(zero_copy_only=False)
print(f"saved v50b_feature.npy [{N},4]  有効 {fin.mean()*100:.1f}%  "
      f"v50b_delta RMSE(vs target) = {np.sqrt(np.nanmean((F[fin,0]-tbl_y[fin])**2)):.4f}  miss={miss}")
