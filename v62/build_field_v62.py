# v62 — Feature-to-Field レンダラ: v52 parquet の PF/beam/sc + gf/v50b npy を
#   (時間×δ) の柔らかい尾根 exp(-((δ-c)/σ)²/2) として描画し、v61 の生GR6ch と結合。
#   出力: imagesF_v62.pkl(EV=(24,nb,181), v60トレーナ互換)
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(P0, "v62", "artifacts_v62"); os.makedirs(ART, exist_ok=True)
ART52 = os.path.join(P0, "v52", "artifacts_v52")
IMR = os.path.join(P0, "v61", "artifacts_v61", "imagesR_v61.pkl")
PQ = os.path.join(ART52, "data", "train.parquet")
NB = 90; ND = 181; STEP = 0.5; STR = 16
OFF = (np.arange(ND) - NB) * STEP

BANKS = ["", "_1", "_2", "_3", "_4", "_5", "_6"]          # pf_ancc{sfx}
SELF = ["_1", "_2", "_3"]; NBR = ["_1", "_2", "_3"]

def stripe(c, s):
    """c,s: (nb,) → (nb,181)。NaN cは0行。"""
    s = np.clip(np.nan_to_num(s, nan=3.0), 1.0, 15.0)
    z = (OFF[None, :] - c[:, None]) / s[:, None]
    img = np.exp(-0.5 * z * z)
    img[~np.isfinite(c)] = 0.0
    return img

def main():
    t0 = time.time()
    IM = pickle.loads(open(IMR, "rb").read())
    cols = (["id", "well", "last_known_tvt"] +
            [f"pf_ancc{s}" for s in BANKS] + [f"pf_ancc_std{s}" for s in BANKS] +
            [f"pf_self{s}" for s in SELF] + [f"pf_self_std{s}" for s in SELF] +
            [f"pf_nbr{s}" for s in NBR] + [f"pf_nbr_std{s}" for s in NBR] +
            ["beam_med_d", "beam_std_d", "sc_ens_d"])
    sch = set(pq.read_schema(PQ).names)
    cols = [c for c in cols if c in sch]
    tb = pq.read_table(PQ, columns=cols)
    D = {c: tb.column(c).to_numpy(zero_copy_only=False) for c in cols if c not in ("id", "well")}
    ids = tb.column("id").to_pandas().astype(str)
    gw = tb.column("well").to_pandas().astype(str).to_numpy()
    del tb
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    gf = np.load(os.path.join(P0, "v48", "artifacts_v48", "gf_feature.npy"))
    vb = np.load(os.path.join(ART52, "v50b_feature.npy"))
    OUTD = {}
    for wi, (wn, e) in enumerate(sorted(IM.items())):
        ii = np.where(gw == wn)[0]
        if len(ii) == 0: continue
        o = np.argsort(rn[ii], kind="stable"); iio = ii[o]
        T = len(e["y_ev"]); n = min(T, len(iio)); iio = iio[:n]
        st = np.asarray(e["st"], np.float64)[:n]
        lk = D["last_known_tvt"][iio].astype(np.float64)
        nb_ = e["EV"].shape[1]
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, n - 1)
        chs = [e["EV"].astype(np.float32)]                # 6ch(生GR)
        def add(centers_row, widths_row):
            c = centers_row[sel]; s = widths_row[sel] if widths_row is not None else np.full(nb_, 2.5)
            chs.append(stripe(c, s)[None].astype(np.float32))
        for s_ in BANKS:
            if f"pf_ancc{s_}" in D:
                add(D[f"pf_ancc{s_}"][iio].astype(np.float64) - st,
                    D.get(f"pf_ancc_std{s_}", None)[iio].astype(np.float64) if f"pf_ancc_std{s_}" in D else None)
        for s_ in SELF:
            if f"pf_self{s_}" in D:
                add(D[f"pf_self{s_}"][iio].astype(np.float64) - st,
                    D[f"pf_self_std{s_}"][iio].astype(np.float64) if f"pf_self_std{s_}" in D else None)
        for s_ in NBR:
            if f"pf_nbr{s_}" in D:
                add(D[f"pf_nbr{s_}"][iio].astype(np.float64) - st,
                    D[f"pf_nbr_std{s_}"][iio].astype(np.float64) if f"pf_nbr_std{s_}" in D else None)
        if "beam_med_d" in D:
            add(D["beam_med_d"][iio].astype(np.float64) + lk - st,
                np.clip(D.get("beam_std_d", None)[iio].astype(np.float64), 1.5, 15) if "beam_std_d" in D else None)
        if "sc_ens_d" in D:
            add(D["sc_ens_d"][iio].astype(np.float64) + lk - st, None)
        add(lk + np.nan_to_num(gf[iio, 0], nan=0.0) - st, np.abs(gf[iio, 1]))
        add(lk + np.nan_to_num(vb[iio, 0], nan=0.0) - st, np.abs(vb[iio, 1]))
        add(lk - st, np.full(n, 1.0))                      # lk hold線
        EV = np.concatenate([chs[0]] + chs[1:], 0).astype(np.float16)
        OUTD[wn] = dict(EV=EV, dtrue=e["dtrue"], wgt=e["wgt"], st=e["st"], bid=e["bid"], y_ev=e["y_ev"])
        if (wi + 1) % 100 == 0: print(f"  {wi+1}/{len(IM)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(OUTD, open(os.path.join(ART, "imagesF_v62.pkl"), "wb"), protocol=4)
    c0 = next(iter(OUTD.values()))["EV"].shape
    print(f"saved imagesF_v62.pkl: {len(OUTD)} wells, EV={c0} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
