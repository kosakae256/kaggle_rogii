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

P0 = Path(os.environ.get("ROGII_PROJ", r"C:\Users\kosaka256\Documents\rogii_claude"))
ART = Path(os.environ.get("ROGII_ART97B", str(P0 / "v97" / "artifacts_v97")))
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
OUT = Path(os.environ["SF_OUT"]) if os.environ.get("SF_OUT") else ART / (f"seqfeat_v95b{_osuf}.pkl")
STR = 16

# --- ★v97: 4バンク構造(pf_1..4)。ancc/self/nbr × 4バンク + beam/sc_ens。schema に無い列は main で自動除外 ---
NB = int(os.environ.get("V97_NBANK", "4"))            # v97 = 4バンク
BASE_DELTA = ([f"pf_ancc_delta_{k}" for k in range(1, NB + 1)] +
              [f"pf_self_delta_{k}" for k in range(1, NB + 1)] +
              [f"pf_nbr_delta_{k}" for k in range(1, NB + 1)] +
              ["beam_med_d", "sc_ens_d"])
BASE_STD = ([f"pf_ancc_std_{k}" for k in range(1, NB + 1)] +
            [f"pf_self_std_{k}" for k in range(1, NB + 1)] +
            [f"pf_nbr_std_{k}" for k in range(1, NB + 1)] +
            ["beam_std_d", None])
assert len(BASE_DELTA) == len(BASE_STD)
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
    _EXTRA_PQ = [p for p in os.environ.get("SF_EXTRA", "").split(",") if p]   # ★追加parqu(id整列)の列も候補に使える
    _extra_map = {}
    for _p in _EXTRA_PQ:
        for _c in pq.read_schema(_p).names:
            if _c not in sch and _c not in _extra_map:
                _extra_map[_c] = _p
    def _has(c):                                          # base schema か SF_EXTRA のどちらかに在れば真
        return (c in sch) or (c in _extra_map)
    # ★v97: base ch も schema に在るものだけ(beam/sc_ens やバンク数の差を自動吸収)
    DELTA = []; STD = []
    for d, s in zip(BASE_DELTA, BASE_STD):
        if d in sch and (s is None or s in sch):
            DELTA.append(d); STD.append(s)
    _ndrop = len(BASE_DELTA) - len(DELTA)
    if _ndrop:
        print(f"[seqfeat] base ch のうち schema 不在 {_ndrop} 本を除外(K={len(DELTA)})", flush=True)
    if V93_INPUT != "fwd":                          # bpf 軌跡chを(在るものだけ)追加
        for d, s in zip(BPF_DELTA, BPF_STD):
            if d in sch and s in sch:
                DELTA.append(d); STD.append(s)
    # ★実験: 追加チャネル(env EXTRA_DELTA/EXTRA_STD, カンマ区切り。新PFをL5に足す等)
    _ed = [x for x in os.environ.get("EXTRA_DELTA", "").split(",") if x]
    _es = [x for x in os.environ.get("EXTRA_STD", "").split(",") if x]
    n_extra = 0
    for d, s in zip(_ed, _es):
        if _has(d) and _has(s):                          # ★SF_EXTRA 側の列も許可(dec/aff 候補を落とさない)
            DELTA.append(d); STD.append(s); n_extra += 1
    print(f"[seqfeat] V93_INPUT={V93_INPUT} 入力={PQ.name} 出力={OUT.name} K={len(DELTA)} "
          f"(base14 + bpf{len(DELTA)-14-n_extra} + extra{n_extra})", flush=True)
    # --- ★struct禁止アサート ---
    for c in DELTA + [s for s in STD if s]:
        low = c.lower()
        assert ("gf" not in low) and ("v50b" not in low) and ("struct" not in low), f"banned col: {c}"
    # ★L5強化: v95既存特徴を C(query context)に追加(env CEXTRA, カンマ区切り。z正規化して付与)
    _cx = [x for x in os.environ.get("CEXTRA", "").split(",") if x and x in sch]
    if _cx:
        print(f"[seqfeat] CEXTRA: +{len(_cx)}ch {_cx}", flush=True)
    # ★候補別特徴(loglik=GR適合/vs_tw=tw乖離)を各候補トークンに付与(env CANDFEAT)。推論値だけでなく信頼性を入れる
    CANDFEAT = os.environ.get("CANDFEAT", "0") == "1"; CFMAP = []; cfcols = []
    _SIB = {"self": "pf_self_vs_nbr", "nbr_gr5": "pf_nbr_gr5_vs_nbr", "self_graft": "pf_self_graft_vs_self"}
    def _ok(c): return c if (c and c in sch) else None
    if CANDFEAT:
        # ★10役割/候補(役割スロット分離=意味混在なし, 0埋め):
        # [0 loglik, 1 vs_tw, 2 兄弟乖離, 3 agree_std, 4 agree_range,
        #  5 ancc gr_abs0, 6 ancc gr_best_resid, 7 ancc rate, 8 ancc vs_spatial, 9 ancc vs_z]
        for dc in DELTA:
            fs = [None] * 10
            px = "v97_" if dc.startswith("v97_") else ""    # ★combo対応: v97_接頭辞を剥がして同prefixで参照
            body = dc[len(px):]
            for rep in ("self", "nbr", "nbr_gr5", "self_graft"):
                if body.startswith(f"pf_{rep}_delta_"):
                    k = body.rsplit("_", 1)[1]
                    fs[0] = _ok(f"{px}pf_{rep}_loglik_{k}"); fs[1] = _ok(f"{px}pf_{rep}_vs_tw_{k}")
                    fs[2] = _ok(f"{px}{_SIB[rep]}_{k}") if rep in _SIB else None
                    fs[3] = _ok(f"{px}pf_agree_std_{k}"); fs[4] = _ok(f"{px}pf_agree_range_{k}")
            if body.startswith("pf_ancc_delta_"):
                k = body.rsplit("_", 1)[1]
                fs[3] = _ok(f"{px}pf_agree_std_{k}"); fs[4] = _ok(f"{px}pf_agree_range_{k}")
                fs[5] = _ok(f"{px}pf_{k}_gr_abs0"); fs[6] = _ok(f"{px}pf_{k}_gr_best_resid")
                fs[7] = _ok(f"{px}pf_{k}_rate"); fs[8] = _ok(f"{px}pf_{k}_vs_spatial"); fs[9] = _ok(f"{px}pf_{k}_vs_dense")
            CFMAP.append(fs)
        _nfm = int(os.environ.get("NF_MAX", "10")); CFMAP = [fs[:_nfm] for fs in CFMAP]   # ★役割数を絞る(sweet spot探索)
        cfcols = sorted({c for fs in CFMAP for c in fs if c})
        print(f"[seqfeat] CANDFEAT: 候補別特徴 {len(cfcols)}列 → 各トークンに{_nfm}役割(nF={_nfm})付与", flush=True)
    cols = (["id", "well", "last_known_tvt", "md_since"] + (["target"] if not NO_TARGET else [])
            + DELTA + [s for s in STD if s] + _cx + cfcols)
    cols = list(dict.fromkeys(cols))                  # ★重複列を排除(std共有等。読込時のKeyError回避)
    sch = set(pq.read_schema(PQ).names)
    _EXTRA_PQ = [p for p in os.environ.get("SF_EXTRA", "").split(",") if p]   # ★追加parquet(id整列済)から候補列を結合
    _extra_map = {}
    for _p in _EXTRA_PQ:
        for _c in pq.read_schema(_p).names:
            if _c not in sch and _c not in _extra_map:
                _extra_map[_c] = _p
    miss = [c for c in cols if c not in sch and c not in _extra_map]
    assert not miss, f"missing cols: {miss}"
    _main_cols = [c for c in cols if c in sch]
    tb = pq.read_table(PQ, columns=_main_cols)
    if _EXTRA_PQ:
        _base_id = tb.column("id").to_pandas().astype(str).to_numpy()
        _nadd = 0
        for _p in _EXTRA_PQ:
            _ec = [c for c in cols if _extra_map.get(c) == _p]
            if not _ec:
                continue
            _etb = pq.read_table(_p, columns=list(dict.fromkeys(["id"] + _ec)))
            _eid = _etb.column("id").to_pandas().astype(str).to_numpy()
            assert len(_eid) == len(_base_id) and bool((_eid == _base_id).all()), f"SF_EXTRA id不一致: {_p}"
            for c in _ec:
                tb = tb.append_column(c, _etb.column(c)); _nadd += 1
        print(f"[seqfeat] SF_EXTRA 結合: +{_nadd}列 from {len(_EXTRA_PQ)}parquet(id整列検証済)", flush=True)
    def get(c):
        return tb.column(c).to_numpy(zero_copy_only=False)
    well = tb.column("well").to_pandas().astype(str).to_numpy()
    ids = tb.column("id").to_pandas().astype(str)
    rn = ids.str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
    tgt = get("target").astype(np.float64) if not NO_TARGET else np.zeros(len(well))   # test は dtrue/wgt ダミー
    mds = np.maximum(get("md_since").astype(np.float64), 0.0)
    DEL = {c: get(c).astype(np.float64) for c in DELTA}
    STw = {c: get(c).astype(np.float64) for c in STD if c}
    CEX = {}                                              # ★CEXTRA: z正規化して保持
    for c in _cx:
        a = get(c).astype(np.float64); mu = np.nanmean(a); sd = np.nanstd(a) + 1e-6
        CEX[c] = np.nan_to_num((a - mu) / sd, nan=0.0)
    CF = {}                                               # ★CANDFEAT: z正規化して保持
    for c in cfcols:
        a = get(c).astype(np.float64); mu = np.nanmean(a); sd = np.nanstd(a) + 1e-6
        CF[c] = np.nan_to_num((a - mu) / sd, nan=0.0)
    del tb

    # ★実験: E-UNet/SDF 出力を外部npz(parquet行順アライメント済 δ)からチャネル注入(env EUNET_NPZ)
    #   npz の各キー = 追加チャネル名, 値 = (N,) の δ(= pred_TVT - last_known)。std は None(既定2.5)。
    if os.environ.get("EUNET_NPZ"):
        _ez = np.load(os.environ["EUNET_NPZ"])
        for _ch in _ez.files:
            _a = _ez[_ch].astype(np.float64)
            assert len(_a) == len(well), f"EUNET ch {_ch} 長さ不一致 {len(_a)}!={len(well)}"
            DEL[_ch] = _a
            DELTA.append(_ch); STD.append(None)
            if CANDFEAT:
                CFMAP.append([None] * int(os.environ.get("NF_MAX", "10")))   # ★注入chはCANDFEAT役割なし=CFMAP長をDELTAに揃える
        print(f"[seqfeat] EUNET_NPZ 注入: +{len(_ez.files)}ch {list(_ez.files)} -> K={len(DELTA)}", flush=True)

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
        if _cx:                                          # ★CEXTRA を bin中心(sel)で C に連結
            Cx = np.stack([CEX[c][rows][sel] for c in _cx], 1)
            C = np.concatenate([C, Cx], 1)

        bid = np.minimum(np.arange(T) // STR, nb_ - 1).astype(np.int32)
        OUTD[wn] = dict(
            D=Dm.astype(np.float32), S=Sm.astype(np.float32),
            C=np.nan_to_num(C).astype(np.float32),
            dtrue=dtrue.astype(np.float32), wgt=wgt.astype(np.float32),
            bid=bid, rows=rows.astype(np.int64), T=int(T))
        if CANDFEAT:                                      # ★候補別特徴 DX(nb,K,nF): 各候補トークンに loglik/vs_tw
            DXs = []
            for ci in range(len(DELTA)):
                ch = [(CF[fc][rows][sel] if fc else np.zeros(nb_)) for fc in CFMAP[ci]]
                DXs.append(np.stack(ch, 1))
            OUTD[wn]["DX"] = np.nan_to_num(np.stack(DXs, 1)).astype(np.float32)   # (nb,K,nF)
        if (wi + 1) % 150 == 0:
            print(f"  {wi+1}/{len(starts)} ({time.time()-t0:.0f}s)", flush=True)

    pickle.dump(OUTD, open(OUT, "wb"), protocol=4)
    import json as _json                                    # ★候補名sidecar(D の列順=DELTA)。L5 の PRUNE 剪定で名前→indexに使う
    _json.dump(list(DELTA), open(str(OUT) + ".cand.json", "w", encoding="utf-8"), ensure_ascii=False)
    k0 = next(iter(OUTD.values()))
    nbs = [v["D"].shape[0] for v in OUTD.values()]
    print(f"saved {OUT.name}: {len(OUTD)} wells  D{k0['D'].shape} (K=14 struct-free)")
    print(f"  N(total rows)={N}  nb: min={min(nbs)} max={max(nbs)} mean={np.mean(nbs):.1f}")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
