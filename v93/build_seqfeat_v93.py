# -*- coding: utf-8 -*-
"""v93 L5(v62 曲線トークン脚)用 seqfeat ビルダー -- STRUCT-FREE 版。

v62 の build_seqfeat_v62.py を忠実に再現するが、以下が違う:
  - 軌跡は 14 本(v62 の 16 本から gf / v50b を DROP)。★struct禁止方針。
    14本 = pf_ancc_1..6 / pf_self_1..3 / pf_nbr_1..3 / beam_med / sc_ens。
  - 基準線(anchor)を v62 の struct `st`(= v38 struct_prior 由来。struct混入!)から
    v93 の struct-free `last_known_tvt`(hold線)へ張り替え。
      * v93 parquet の `*_delta` 列 = (絶対値 - last_known_tvt) なので、そのまま δ として使える。
      * beam_med_d / sc_ens_d は元から last_known からの δ。
      * per-bin ターゲット dtrue = median(target)。ここで target = TVT - last_known_tvt(= parquet 'target')。
        v62 の dtrue = median(y_true - st) の st を last_known に置換したものに一致。
  - bin 構造(nb, bid, sel)は v62/v61 と完全一致(STR=16, sel=clip(arange(nb)*STR+STR//2,0,T-1))。
    v93 parquet の eval 行数は imagesR_v61 と一致(例: well0 = 3836 行 -> nb=240)を確認済み。

出力: v93/artifacts_v93/seqfeat_v93.pkl
  = {well: {D:(nb,14)δ, S:(nb,14)幅, C:(nb,3)文脈, dtrue:(nb,), wgt:(nb,),
            bid:(T,)行->bin, rows:(T,)parquet全体行index, T:int}}
  ※ D/S/C/dtrue/wgt は v62 と同じ役割。bid/rows/T は train 側で per-row OOF を復元するための追加。

struct混入なし: gf_feature(v48)/v50b/struct/v38/v41 は一切読まない。v93 parquet 単体。
"""
import io, sys
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import os, pickle, time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P0 / "v93" / "artifacts_v93"
# --- 入力切替(env V93_INPUT): fwd / fwdbwd / fwdbwd_anchor ---
V93_INPUT = os.environ.get("V93_INPUT", "fwd").lower()
_SMAP = {"fwd": ("train.parquet", ""),
         "fwdbwd": ("train_fwdbwd.parquet", "_fwdbwd"),
         "fwdbwd_anchor": ("train_fwdbwd_anchor.parquet", "_fwdbwd_anchor")}
if V93_INPUT not in _SMAP:
    raise SystemExit(f"[seqfeat] 未知の V93_INPUT={V93_INPUT}")
_fn, _osuf = _SMAP[V93_INPUT]
# 提出(test): SF_IN=test_fwdbwd.parquet / SF_OUT=seqfeat_test... / SF_NO_TARGET=1(target列なし)
NO_TARGET = os.environ.get("SF_NO_TARGET", "0") == "1"
PQ = Path(os.environ["SF_IN"]) if os.environ.get("SF_IN") else ART / "data" / _fn
OUT = Path(os.environ["SF_OUT"]) if os.environ.get("SF_OUT") else ART / (f"seqfeat_v93{_osuf}.pkl")
STR = 16

# --- 14 本の struct-free 軌跡(anchor = last_known_tvt)。gf/v50b は DROP ---
#     v62 の順序を踏襲: pf_ancc_1..6, pf_self_1..3, pf_nbr_1..3, beam_med, sc_ens
BASE_DELTA = ([f"pf_ancc_delta_{k}" for k in range(1, 7)] +
              [f"pf_self_delta_{k}" for k in range(1, 4)] +
              [f"pf_nbr_delta_{k}" for k in range(1, 4)] +
              ["beam_med_d", "sc_ens_d"])
BASE_STD = ([f"pf_ancc_std_{k}" for k in range(1, 7)] +
            [f"pf_self_std_{k}" for k in range(1, 4)] +
            [f"pf_nbr_std_{k}" for k in range(1, 4)] +
            ["beam_std_d", None])
assert len(BASE_DELTA) == 14 and len(BASE_STD) == 14
# --- ★backward(bpf)軌跡ch(fwdbwd* のみ, parquet に在れば追加): ancc6/self3/nbr3 = 12本 ---
BPF_DELTA = ([f"bpf_ancc_delta_{k}" for k in range(1, 7)] +
             [f"bpf_self_delta_{k}" for k in range(1, 4)] +
             [f"bpf_nbr_delta_{k}" for k in range(1, 4)])
BPF_STD = ([f"bpf_ancc_std_{k}" for k in range(1, 7)] +
           [f"bpf_self_std_{k}" for k in range(1, 4)] +
           [f"bpf_nbr_std_{k}" for k in range(1, 4)])


def main():
    t0 = time.time()
    sch = set(pq.read_schema(PQ).names)
    DELTA = list(BASE_DELTA); STD = list(BASE_STD)
    if V93_INPUT != "fwd":                          # bpf 軌跡chを(在るものだけ)追加
        for d, s in zip(BPF_DELTA, BPF_STD):
            if d in sch and s in sch:
                DELTA.append(d); STD.append(s)
    # ★実験: 追加チャネル(env EXTRA_DELTA/EXTRA_STD, カンマ区切り。新PFをL5に足す等)
    _ed = [x for x in os.environ.get("EXTRA_DELTA", "").split(",") if x]
    _es = [x for x in os.environ.get("EXTRA_STD", "").split(",") if x]
    n_extra = 0
    for d, s in zip(_ed, _es):
        if d in sch and s in sch:
            DELTA.append(d); STD.append(s); n_extra += 1
    print(f"[seqfeat] V93_INPUT={V93_INPUT} 入力={PQ.name} 出力={OUT.name} K={len(DELTA)} "
          f"(base14 + bpf{len(DELTA)-14-n_extra} + extra{n_extra})", flush=True)
    # --- ★struct禁止アサート ---
    for c in DELTA + [s for s in STD if s]:
        low = c.lower()
        assert ("gf" not in low) and ("v50b" not in low) and ("struct" not in low), f"banned col: {c}"
    cols = (["id", "well", "last_known_tvt", "md_since"] + (["target"] if not NO_TARGET else [])
            + DELTA + [s for s in STD if s])
    sch = set(pq.read_schema(PQ).names)
    miss = [c for c in cols if c not in sch]
    assert not miss, f"missing cols: {miss}"
    tb = pq.read_table(PQ, columns=cols)
    def get(c):
        return tb.column(c).to_numpy(zero_copy_only=False)
    well = tb.column("well").to_pandas().astype(str).to_numpy()
    ids = tb.column("id").to_pandas().astype(str)
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    tgt = get("target").astype(np.float64) if not NO_TARGET else np.zeros(len(well))   # test は dtrue/wgt ダミー
    mds = np.maximum(get("md_since").astype(np.float64), 0.0)
    DEL = {c: get(c).astype(np.float64) for c in DELTA}
    STw = {c: get(c).astype(np.float64) for c in STD if c}
    del tb

    # well ごとの連続ブロック境界(parquet 上で連続なのを利用。ビュー参照でコピー回避)
    change = np.where(well[1:] != well[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(well)]])
    N = len(well)

    OUTD = {}
    for wi, (s0, e0) in enumerate(zip(starts, ends)):
        wn = well[s0]
        T = e0 - s0
        # bin 順 = rn 昇順(v62 の rn ソートに合わせる。通常は既に昇順)
        loc = np.argsort(rn[s0:e0], kind="stable")
        rows = np.arange(s0, e0)[loc]              # parquet 全体での行 index(bin 順)
        nb_ = (T + STR - 1) // STR
        sel = np.clip(np.arange(nb_) * STR + STR // 2, 0, T - 1)

        # --- D(δ, bin中心行の値) / S(幅) ---
        Ds, Ss = [], []
        for dc, sc in zip(DELTA, STD):
            dv = DEL[dc][rows]
            Ds.append(dv[sel])
            if sc is None:
                Ss.append(np.full(nb_, 2.5))
            elif sc == "beam_std_d":
                Ss.append(np.clip(STw[sc][rows], 1.5, 15.0)[sel])
            else:
                Ss.append(STw[sc][rows][sel])
        Dm = np.stack(Ds, 1)
        Sm = np.clip(np.nan_to_num(np.stack(Ss, 1), nan=3.0), 1.0, 15.0)
        Dm = np.clip(np.nan_to_num(Dm, nan=0.0), -60, 60)

        # --- per-bin ターゲット dtrue / 重み wgt(= v62 の median(y_true - st) 相当。st -> last_known)
        #     bid = arange(T)//STR は連続 16 行チャンクなのでスライスで集計(高速) ---
        tw_ = tgt[rows]
        dtrue = np.full(nb_, np.nan)
        wgt = np.zeros(nb_)
        for b in range(nb_):
            vals = tw_[b * STR: min((b + 1) * STR, T)]
            fin = np.isfinite(vals)
            if fin.any():
                dtrue[b] = float(np.median(vals[fin]))
                wgt[b] = float(fin.sum())

        # --- C(文脈 3ch)。v62: [md_since/1000, (lk-st)/10, log1p(wgt)]
        #     struct-free では anchor==last_known なので (lk-anchor)=0(死にチャネルだが shape 保持) ---
        lk_minus_anchor = np.zeros(nb_)
        C = np.stack([mds[rows][sel] / 1000.0, lk_minus_anchor, np.log1p(wgt)], 1)

        bid = np.minimum(np.arange(T) // STR, nb_ - 1).astype(np.int32)
        OUTD[wn] = dict(
            D=Dm.astype(np.float32), S=Sm.astype(np.float32),
            C=np.nan_to_num(C).astype(np.float32),
            dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
            bid=bid, rows=rows.astype(np.int64), T=int(T))
        if (wi + 1) % 150 == 0:
            print(f"  {wi+1}/{len(starts)} ({time.time()-t0:.0f}s)", flush=True)

    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    k0 = next(iter(OUTD.values()))
    nbs = [v["D"].shape[0] for v in OUTD.values()]
    print(f"saved {OUT.name}: {len(OUTD)} wells  D{k0['D'].shape} (K=14 struct-free)")
    print(f"  N(total rows)={N}  nb: min={min(nbs)} max={max(nbs)} mean={np.mean(nbs):.1f}")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
