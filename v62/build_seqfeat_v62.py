# v62 L4/L5/L6 共通データ — v52特徴のうち「エンジン軌跡16本(δ枠)+幅+文脈」をbin単位で抽出
#   出力: seqfeat_v62.pkl {well: {D:(nb,16)δ, S:(nb,16)幅, C:(nb,3)文脈}}(メタは imagesF_v62 と共有)
#   16本: pf_ancc_1..6 / pf_self_1..3 / pf_nbr_1..3 / beam_med / sc_ens / gf / v50b
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v62", "artifacts_v62")
ART52 = os.path.join(P0, "v52", "artifacts_v52")
PQ = os.path.join(ART52, "data", "train.parquet")
STR = 16

def main():
    t0 = time.time()
    IM = pickle.loads(open(os.path.join(ART, "imagesF_v62.pkl"), "rb").read())
    B6 = [f"_{k}" for k in range(1, 7)]
    cols = (["id", "well", "last_known_tvt", "md_since"] +
            [f"pf_ancc{s}" for s in B6] + [f"pf_ancc_std{s}" for s in B6] +
            [f"pf_self_{k}" for k in (1, 2, 3)] + [f"pf_self_std_{k}" for k in (1, 2, 3)] +
            [f"pf_nbr_{k}" for k in (1, 2, 3)] + [f"pf_nbr_std_{k}" for k in (1, 2, 3)] +
            ["beam_med_d", "beam_std_d", "sc_ens_d"])
    sch = set(pq.read_schema(PQ).names)
    cols = [c for c in cols if c in sch]
    tb = pq.read_table(PQ, columns=cols)
    D_ = {c: tb.column(c).to_numpy(zero_copy_only=False) for c in cols if c not in ("id", "well")}
    ids = tb.column("id").to_pandas().astype(str)
    gw = tb.column("well").to_pandas().astype(str).to_numpy()
    del tb
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    gf = np.load(os.path.join(P0, "v48", "artifacts_v48", "gf_feature.npy"))
    vb = np.load(os.path.join(ART52, "v50b_feature.npy"))
    OUT = {}
    for wi, (wn, e) in enumerate(sorted(IM.items())):
        ii = np.where(gw == wn)[0]
        if len(ii) == 0: continue
        o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        T = len(e["y_ev"]); n = min(T, len(iio)); iio = iio[:n]
        st = np.asarray(e["st"], np.float64)[:n]
        lk = D_["last_known_tvt"][iio].astype(np.float64)
        mds = np.maximum(D_["md_since"][iio].astype(np.float64), 0)
        nb_ = e["EV"].shape[1]
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, n - 1)
        Ds, Ss = [], []
        def add(vals, wid):
            Ds.append(vals[sel]); Ss.append((wid[sel] if wid is not None else np.full(nb_, 2.5)))
        for s_ in B6:
            add(D_[f"pf_ancc{s_}"][iio].astype(np.float64) - st, D_[f"pf_ancc_std{s_}"][iio].astype(np.float64))
        for k in (1, 2, 3):
            add(D_[f"pf_self_{k}"][iio].astype(np.float64) - st, D_[f"pf_self_std_{k}"][iio].astype(np.float64))
        for k in (1, 2, 3):
            add(D_[f"pf_nbr_{k}"][iio].astype(np.float64) - st, D_[f"pf_nbr_std_{k}"][iio].astype(np.float64))
        add(D_["beam_med_d"][iio].astype(np.float64) + lk - st, np.clip(D_["beam_std_d"][iio].astype(np.float64), 1.5, 15))
        add(D_["sc_ens_d"][iio].astype(np.float64) + lk - st, None)
        add(lk + np.nan_to_num(gf[iio, 0], nan=0.0) - st, np.abs(gf[iio, 1]))
        add(lk + np.nan_to_num(vb[iio, 0], nan=0.0) - st, np.abs(vb[iio, 1]))
        Dm = np.stack(Ds, 1); Sm = np.clip(np.nan_to_num(np.stack(Ss, 1), nan=3.0), 1.0, 15.0)
        Dm = np.clip(np.nan_to_num(Dm, nan=0.0), -60, 60)
        C = np.stack([mds[sel] / 1000.0, (lk - st)[sel] / 10.0, np.log1p(e["wgt"])], 1)
        OUT[wn] = dict(D=Dm.astype(np.float32), S=Sm.astype(np.float32), C=np.nan_to_num(C).astype(np.float32))
        if (wi + 1) % 150 == 0: print(f"  {wi+1}/{len(IM)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUT, open(os.path.join(ART, "seqfeat_v62.pkl"), "wb"), protocol=4)
    k0 = next(iter(OUT.values()))
    print(f"saved seqfeat_v62.pkl: {len(OUT)} wells D{k0['D'].shape} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
