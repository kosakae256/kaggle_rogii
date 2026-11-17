# -*- coding: utf-8 -*-
"""
v83 — L5(曲線トークン)再建用 seqfeat ビルダー(帯中心=K16錨, struct完全不使用)
==============================================================================
v62のseqfeat(帯中心=struct軌跡)をK16基盤で再構築。bin=16行, δ=各エンジン軌跡−K16錨。
トークン(存在するもののみ):
  pf_ancc×5 / pf_self×≤5 / pf_nbr×≤5(いずれもstd付き, v79 parquet=K16gate版)
  beam_mean(+lk), sc_ens(+lk), gf_v79(z_bin=lk+col0), P5a_off, P6f_off(v78 features=既にδ)
出力: v83/artifacts_v83/seqfeat_v83.pkl {well:{D,S,C,dtrue,wgt,st,rows}}
使い方: python v83/build_seqfeat_v83.py
"""
import io, sys, os, pickle, time
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v83" / "artifacts_v83"; ART.mkdir(parents=True, exist_ok=True)
PQ = P0 / "v79" / "artifacts_v79" / "data" / "train.parquet"
STR = 16

t0 = time.time()
sch = set(pq.read_schema(PQ).names)
tw_c = [c for c in ["pf_ancc", "pf_ancc_1", "pf_ancc_2", "pf_ancc_3", "pf_ancc_4"] if c in sch]
tw_s = [c.replace("pf_ancc", "pf_ancc_std") for c in tw_c]
sf_c = [f"pf_self_{k}" for k in range(1, 6) if f"pf_self_{k}" in sch]
sf_s = [c.replace("pf_self_", "pf_self_std_") for c in sf_c]
nb_c = [f"pf_nbr_{k}" for k in range(1, 6) if f"pf_nbr_{k}" in sch]
nb_s = [c.replace("pf_nbr_", "pf_nbr_std_") for c in nb_c]
cols = (["well", "target", "last_known_tvt", "md_since"] + tw_c + tw_s + sf_c + sf_s + nb_c + nb_s +
        [c for c in ["beam_mean_d", "beam_std_d", "sc_ens_d"] if c in sch])
tb = pq.read_table(PQ, columns=cols)
well = tb["well"].to_numpy(zero_copy_only=False)
D_ = {c: tb[c].to_numpy(zero_copy_only=False) for c in cols if c != "well"}
del tb
GF = np.load(P0 / "v79" / "artifacts_v79" / "gf_feature_v79.npy")
F78 = pickle.loads((P0 / "v77" / "artifacts_v77" / "v78_features_train.pkl").read_bytes())
ANC = pickle.loads((P0 / "v77" / "artifacts_v77" / "k16_anchor_train.pkl").read_bytes())
print(f"tokens: tw{len(tw_c)} self{len(sf_c)} nbr{len(nb_c)} +beam+sc+gf+P5a+P6f ({time.time()-t0:.0f}s)", flush=True)

chg = np.flatnonzero(well[1:] != well[:-1]) + 1
starts = np.concatenate([[0], chg]); ends = np.concatenate([chg, [len(well)]])
OUT = {}
for s0, e0 in zip(starts, ends):
    wid = str(well[s0])
    if wid not in ANC or wid not in F78: continue
    n = e0 - s0
    A = ANC[wid]["tvt"]
    T = min(n, len(A))
    st = np.asarray(A[:T], np.float64)
    lk = D_["last_known_tvt"][s0:s0 + T].astype(np.float64)
    tvt_true = lk + D_["target"][s0:s0 + T].astype(np.float64)
    mds = np.maximum(D_["md_since"][s0:s0 + T].astype(np.float64), 0)
    nbn = (T + STR - 1) // STR
    sel = np.clip(np.arange(nbn) * STR + STR // 2, 0, T - 1)
    Ds, Ss = [], []
    def add(vals, wid_):
        Ds.append(vals[sel])
        Ss.append(wid_[sel] if wid_ is not None else np.full(nbn, 2.5))
    for c, cs in zip(tw_c + sf_c + nb_c, tw_s + sf_s + nb_s):
        v = D_[c][s0:s0 + T].astype(np.float64)
        sd = D_[cs][s0:s0 + T].astype(np.float64) if cs in D_ else None
        add(v - st, sd)
    if "beam_mean_d" in D_:
        add(D_["beam_mean_d"][s0:s0 + T].astype(np.float64) + lk - st,
            np.clip(D_["beam_std_d"][s0:s0 + T].astype(np.float64), 1.5, 15))
    if "sc_ens_d" in D_:
        add(D_["sc_ens_d"][s0:s0 + T].astype(np.float64) + lk - st, None)
    add(lk + np.nan_to_num(GF[s0:s0 + T, 0], nan=0.0) - st, np.abs(GF[s0:s0 + T, 1]))
    f78 = F78[wid]
    Tf = min(T, len(f78["p5a_off"]))
    p5 = np.zeros(T); p5[:Tf] = f78["p5a_off"][:Tf]
    p6 = np.zeros(T); p6[:Tf] = f78["p6f_off"][:Tf]
    add(p5, np.clip(np.abs(f78["p6e_postd"][:T]) if len(f78["p6e_postd"]) >= T else np.full(T, 3.0), 1, 15))
    add(p6, None)
    Dm = np.clip(np.nan_to_num(np.stack(Ds, 1), nan=0.0), -60, 60).astype(np.float32)
    Sm = np.clip(np.nan_to_num(np.stack(Ss, 1), nan=3.0), 1.0, 15.0).astype(np.float32)
    wgt = np.minimum(np.full(nbn, STR), T - np.arange(nbn) * STR).astype(np.float32)
    C = np.stack([mds[sel] / 1000.0, (lk - st)[sel] / 10.0, np.log1p(wgt)], 1).astype(np.float32)
    dtrue = (tvt_true - st)[sel].astype(np.float32)
    OUT[wid] = dict(D=Dm, S=Sm, C=C, dtrue=dtrue, wgt=wgt,
                    st=st[sel].astype(np.float64), rows=(np.arange(nbn) * STR + STR // 2).astype(np.int32),
                    T=T, s0=int(s0),
                    dtrue_rows=(tvt_true - st).astype(np.float16),
                    bid=(np.arange(T) // STR).astype(np.int32))
K = OUT[next(iter(OUT))]["D"].shape[1]
fp = ART / "seqfeat_v83.pkl"
fp.write_bytes(pickle.dumps(OUT, protocol=4))
print(f"[saved] seqfeat_v83.pkl {len(OUT)}井 K={K} ({fp.stat().st_size/1e6:.0f}MB, {time.time()-t0:.0f}s)")
