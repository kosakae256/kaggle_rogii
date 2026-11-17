# -*- coding: utf-8 -*-
"""v100 backward-PF feature generator (bpf_* columns) — 3レグ対応・5表現・終端ファイルwarm init・emission込み。

build_backward_v93 の忠実な拡張:
  - LEG(env: v96/v97/v100)で pf_banks module / bank config / forward列接頭 / 錨 / 出力 を切替。
  - 5表現: tw / self / nbr / nbr_gr5 / self_graft(create_v95/v97 と同一 builder)。
  - warm init: 軽量 bwd_endpoints.parquet(順方向far-end 75値)から FWD_LAST を読む(重いparquet不要)。
  - emission: SIM_PKL(nn_emission が各レグ錨で生成)を _attach_sim → w_nn(config)で全バンク emission込み。
  - EMIT_VS_FWD=0: 全forward軌跡が無いので vs_fwd族は出さない。窓特徴の先頭エッジは自動NaN。
  - PF は run_smoother_ext を通るので PF_NGPU>=2 で 2GPU 分散(Kaggle T4x2)。

env:
  LEG            v96|v97|v100(必須)
  ROGII_ART      レグ art dir(pf_banks_config.json / grfree_anchor_train.pkl / nn50 / SIM_PKL を含む)
  BWD_ENDPOINTS  bwd_endpoints.parquet のパス(warm init 用)
  BWD_FWD_PREFIX forward列接頭(v96='' / v97='v97_' / v100='v100_')
  SIM_PKL        emission sim pkl(あれば emission込み。無ければ w_nn=0扱い)
  OUT_BWD        出力 bpf parquet
  NWELLS         スモーク(先頭N井)
  ROGII_DATA     競技データ
  PF_NGPU        (run_smoother_ext が読む。Kaggleで device_count)
出力: OUT_BWD = {id, well, bpf_*}(train と行同順)。train_combo への結合は別途 id で。
"""
import os, sys, io, glob, time, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

_SELF = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(_SELF))

LEG = os.environ["LEG"].lower()                      # v96 / v97 / v100
_PFMOD = {"v96": "pf_banks_v95", "v97": "pf_banks_v97", "v100": "pf_banks_v95"}[LEG]
_IMPMOD = {"v96": "imputers_v95", "v97": "imputers_v97", "v100": "imputers_v95"}[LEG]
_FEATMOD = {"v96": "features_v95", "v97": "features_v97", "v100": "features_v95"}[LEG]
pf = __import__(_PFMOD)
imp = __import__(_IMPMOD)
featv = __import__(_FEATMOD)

FWD_PREFIX = os.environ.get("BWD_FWD_PREFIX", {"v96": "", "v97": "v97_", "v100": "v100_"}[LEG])
DATA_DIR = Path(os.environ.get("ROGII_DATA", str(pf.DATA_DIR)))
SEED = 4423098
N_SEED = pf.N_SEED
WELL_CHUNK = int(os.environ.get("PF_WELL_CHUNK", "40"))
BANK_ORDER = pf.BANK_ORDER
SUFFIXES = [f"_{i+1}" for i in range(len(BANK_ORDER))]
DEVICE = pf.DEVICE
METHODS = ["tw", "self", "nbr", "nbr_gr5", "self_graft"]
_REP2FWD = {"tw": "pf_ancc", "self": "pf_self", "nbr": "pf_nbr", "nbr_gr5": "pf_nbr_gr5", "self_graft": "pf_self_graft"}

ENDPOINTS = Path(os.environ["BWD_ENDPOINTS"])
SIM_PKL = Path(os.environ["SIM_PKL"]) if os.environ.get("SIM_PKL") else None
OUT_BWD = Path(os.environ.get("OUT_BWD", str(_SELF.parent / "v100" / "artifacts_v100" / f"bpf_{LEG}.parquet")))
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None
EMIT_VS_FWD = False                                   # ★終端ファイルのみ=全forward軌跡なし → vs_fwd族は出さない
OFFS = featv.PF_OFFS


def _attach_sim(x, wid, SIMD):
    sd = SIMD.get(wid)
    if sd is not None:
        x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    return x


def reverse_inputs(x):
    xr = dict(x)
    md = np.asarray(x["md"], float)
    xr["md"] = md[0] + (md[-1] - md[::-1])
    xr["z"] = np.asarray(x["z"])[::-1].copy()
    xr["gr"] = np.asarray(x["gr"])[::-1].copy()
    if "anc" in x:
        xr["anc"] = np.asarray(x["anc"])[::-1].copy()
        xr["ancs"] = np.asarray(x["ancs"])[::-1].copy()
    if "_st" in x:
        xr["_st"] = np.asarray(x["_st"])[::-1].copy()
    if "_sim" in x:
        xr["_sim"] = np.asarray(x["_sim"])[::-1].copy()
    return xr


def _build_rep(rep, hw, tt, tg, P, wid, NREFS, NREFS_GR5):
    if rep == "tw":
        return pf.build_smoother_inputs(hw, tt, tg, P)
    if rep == "self":
        return pf.build_inputs_self(hw, tt, tg, P)
    if rep == "nbr":
        return pf.build_inputs_nbr(hw, tt, tg, P, NREFS.get(wid, []))
    if rep == "nbr_gr5":
        return pf.build_inputs_nbr(hw, tt, tg, P, NREFS_GR5.get(wid, []))
    if rep == "self_graft":
        return pf.build_inputs_self_graft(hw, tt, tg, P)
    raise ValueError(rep)


def run_bank_rep(paths, names, bank, rep, SIMD, NREFS, NREFS_GR5, sfx, FWD_LAST, HWS, TWS):
    """1 バンク x 1 表現の backward smooth-PF(warm init=順方向far-end)。名前->dict(mean,std) 元行順。
       ★CSVは HWS/TWS キャッシュから(バンク非依存の読込を全パスで使い回し=I/O 1回化)。"""
    P = pf.bank_param(bank); w_nn = P["_w_nn"]; phys = P["_physics"]
    if os.environ.get("BWD_W_NN"):                                  # ★全レグemission込み: w_nnを強制(pf5 config=0.0対策)
        w_nn = float(os.environ["BWD_W_NN"])
    P["smooth_lag"] = int(os.environ.get("BWD_SMOOTH_LAG", "16"))   # ★逆方向=固定ラグ16(config の192/full を上書き)
    P["smooth_mode"] = "fixedlag"
    inps = []
    _tb = time.perf_counter()
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = HWS[wid]; tw = TWS[wid]                                 # ★キャッシュ(pd.read_csv しない)
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        x = _build_rep(rep, hw, tt, tg, P, wid, NREFS, NREFS_GR5)
        x = pf.attach_anchor(x, wid, phys)
        _attach_sim(x, wid, SIMD)
        # warm init: far-end 始点(順方向far-end TVT + far-end Z)、dip 符号反転
        fwd_tvt_last = FWD_LAST.get((wid, sfx, rep))
        if fwd_tvt_last is not None and np.isfinite(fwd_tvt_last):
            x["ls"] = float(fwd_tvt_last) + float(x["z"][-1])
        x["ir"] = -float(x["ir"])
        inps.append(reverse_inputs(x))
    t_build = time.perf_counter() - _tb
    _tp = time.perf_counter()
    outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
    t_pf = time.perf_counter() - _tp
    res = {}
    for i, w in enumerate(names):
        o = outs[i]
        res[w] = dict(mean=o["mean"][::-1].copy(), std=o["std"][::-1].copy())
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res, t_build, t_pf


def main():
    t_all = time.time()
    _actual_lag = int(os.environ.get("BWD_SMOOTH_LAG", "16"))
    print(f"[backward_v100] LEG={LEG} pf={_PFMOD} banks={BANK_ORDER} 5rep={METHODS} "
          f"smooth_lag={_actual_lag}(fixedlag; config{pf.SMOOTH_LAG}を上書き) "
          f"n_seed={N_SEED} FWD_PREFIX='{FWD_PREFIX}' emission={'ON' if SIM_PKL and SIM_PKL.exists() else 'OFF'} "
          f"w_nn={os.environ.get('BWD_W_NN','config')} PF_NGPU={os.environ.get('PF_NGPU','1')}", flush=True)

    base = DATA_DIR / "train"
    # (0) 有効井 + ★CSVキャッシュ(HWS/TWS を1回だけ読む=全パスで使い回し, I/O 30回→1回)
    P0 = pf.bank_param(BANK_ORDER[0])
    names = []; paths = []; HWS = {}; TWS = {}
    hps = sorted(glob.glob(str(base / "*__horizontal_well.csv")))
    _t_read = time.time()
    for hp in hps:
        wid = os.path.basename(hp).split("__")[0]
        tp = str(base / f"{wid}__typewell.csv")
        if not os.path.exists(tp):
            continue
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None:
            continue
        names.append(wid); paths.append((hp, tp)); HWS[wid] = hw; TWS[wid] = tw
        if NWELLS and len(names) >= NWELLS:
            break
    print(f"[backward_v100] CSVキャッシュ {len(names)}井 読込 {time.time()-_t_read:.0f}s", flush=True)
    print(f"[backward_v100] 使用坑井: {len(names)}  passes={len(BANK_ORDER)*len(METHODS)}(bank{len(BANK_ORDER)} x rep5)", flush=True)

    # (0.5) 終端ファイル → FWD_LAST[(wid, sfx, rep)]
    ep = pd.read_parquet(ENDPOINTS)
    ep_by_well = {r["well"]: r for _, r in ep.iterrows()}
    lk_by_well = {w: float(r["last_known_tvt"]) for w, r in ep_by_well.items()}
    FWD_LAST = {}
    for wid in names:
        r = ep_by_well.get(wid)
        if r is None:
            continue
        for k, s in enumerate(SUFFIXES):
            kk = k + 1
            for rep in METHODS:
                col = f"{FWD_PREFIX}{_REP2FWD[rep]}_{kk}"   # ★tw→pf_ancc(旧: pf_{rep}=pf_tw で不在→tw warm-init欠落=tw全滅バグ)
                FWD_LAST[(wid, s, rep)] = float(r[col]) if col in r and np.isfinite(r[col]) else np.nan
    print(f"[backward_v100] warm-init 終端 読込: {len(ep_by_well)}井 x {len(SUFFIXES)*len(METHODS)}軌跡", flush=True)

    # (1) emission sim(★read_bytes()は1.38GBの二重確保 → open+loadでストリーム読み=RAMピーク半減)
    if SIM_PKL and SIM_PKL.exists():
        with open(SIM_PKL, "rb") as _sf:
            SIMD = pickle.load(_sf)
        print(f"[backward_v100 emission] sim load: {len(SIMD)} wells (w_nn=config)", flush=True)
    else:
        SIMD = {}
        print(f"[backward_v100 emission] sim 無し -> emission OFF", flush=True)

    # (2) imputer + 近傍プール(nbr / nbr_gr5)
    FI, DI, _tw = imp.build_imputers()
    NREFS = {}; NREFS_GR5 = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = HWS[wid]
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        cz = float(hw["Z"].mean()) if "Z" in hw else 0.0
        refw = imp.neighbors_of(cx, cy, exclude_wid=wid)
        NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if len(refw) >= imp.NEED_REFS else []
        kn = hw.dropna(subset=["TVT_input", "GR"]) if "TVT_input" in hw else hw
        try:
            ng = imp.neighbors_gr_of(kn["TVT_input"].to_numpy(float), kn["GR"].to_numpy(float), cx, cy, cz, exclude_wid=wid)
            NREFS_GR5[wid] = [imp.ref_grtvt(w) for w, *_ in ng]
        except Exception:
            NREFS_GR5[wid] = []
    print(f"[backward_v100] 近傍プール: nbr={sum(1 for v in NREFS.values() if v)} gr5={sum(1 for v in NREFS_GR5.values() if v)}", flush=True)

    # (3) 各バンク x 各表現 backward smooth-PF
    STORE = {rep: {} for rep in METHODS}
    for k, bank in enumerate(BANK_ORDER):
        s = SUFFIXES[k]
        for rep in METHODS:
            t0 = time.perf_counter()
            r, t_build, t_pf = run_bank_rep(paths, names, bank, rep, SIMD, NREFS, NREFS_GR5, s, FWD_LAST, HWS, TWS)
            for w in names:
                STORE[rep][(w, bank)] = r[w]
            print(f"[backward-{rep}] {bank} done {time.perf_counter()-t0:.1f}s (入力構築{t_build:.1f}s / PF{t_pf:.1f}s)", flush=True)
    import gc
    del SIMD; gc.collect()                               # ★emission sim(~1.38GB)はここで不要 → 解放

    # (4) per-well hgr/hmd/tw グリッド(gr一致・shape・位置に必要)
    HGR = {}; HMD = {}; TWG = {}; IDMAP = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = HWS[wid]; tw = TWS[wid]
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        x0 = pf.build_smoother_inputs(hw, tt, tg, P0)
        if x0 is None:
            continue
        HGR[wid] = np.asarray(x0["gr"], np.float32); HMD[wid] = np.asarray(x0["md"], np.float32)
        TWG[wid] = (tt, tg)
        # ★id = {wid}_{eval行の元CSV行index}(create_v95 と同一: TVT_input NaN の行 index)
        ev_idx = hw.index[hw["TVT_input"].isna()].to_numpy()
        IDMAP[wid] = np.array([f"{wid}_{int(i)}" for i in ev_idx], dtype=object)

    # (5) bpf_ 特徴生成(features_v95 派生器を bpf_ で再利用。5表現 full-mirror)
    # ★FEAT は per-well の numpy(float32)配列を append(旧: Python float の巨大リスト=~27GB OOM)。
    rows_id = []; rows_well = []; FEAT = {}
    def put(name, arr):
        FEAT.setdefault(name, []).append(np.asarray(arr, np.float32))

    n_out = 0
    for wid in names:
        m = None
        # 全表現・全バンクの mean を取り出し、最小長 m に揃える
        bm_tw = {s: np.asarray(STORE["tw"][(wid, BANK_ORDER[k])]["mean"], np.float32) for k, s in enumerate(SUFFIXES)}
        m = min(len(v) for v in bm_tw.values())
        hgr = HGR.get(wid); hmd = HMD.get(wid)
        if hgr is not None: hgr = hgr[:m]
        if hmd is not None: hmd = hmd[:m]
        twp = TWG.get(wid); tw_tvt, tw_gr = (twp if twp else (None, None))
        lk = float(lk_by_well.get(wid, np.nan))
        f = {}
        # --- 各表現 per-bank: mean/std/delta + gr一致 + shape ---
        rep_stacks = {rep: [] for rep in METHODS}
        for k, s in enumerate(SUFFIXES):
            bank = BANK_ORDER[k]
            for rep in METHODS:
                bm = np.asarray(STORE[rep][(wid, bank)]["mean"], np.float32)[:m]
                bs = np.asarray(STORE[rep][(wid, bank)]["std"], np.float32)[:m]
                tag = {"tw": "ancc", "self": "self", "nbr": "nbr", "nbr_gr5": "nbr_gr5", "self_graft": "self_graft"}[rep]
                f[f"bpf_{tag}{s}"] = bm
                f[f"bpf_{tag}_std{s}"] = bs
                f[f"bpf_{tag}_delta{s}"] = (bm - lk).astype(np.float32)
                rep_stacks[rep].append(bm)
                if rep == "tw":  # tw(ancc)にのみ gr一致/shape(forward pf_ancc と同格)
                    pref = f"bpf{s}"
                    if tw_tvt is not None and hgr is not None:
                        featv._add_gr_match_features(f, pref, bm, hgr, tw_tvt, tw_gr, OFFS)
                    if hmd is not None:
                        featv._add_path_shape_features(f, pref, bm, hmd)
                # 表現間 vs_tw / vs_nbr
            twk = f[f"bpf_ancc{s}"]
            for rep, tag in [("self", "self"), ("nbr", "nbr"), ("nbr_gr5", "nbr_gr5"), ("self_graft", "self_graft")]:
                f[f"bpf_{tag}_vs_tw{s}"] = (f[f"bpf_{tag}{s}"] - twk).astype(np.float32)
            f[f"bpf_self_vs_nbr{s}"] = (f["bpf_self" + s] - f["bpf_nbr" + s]).astype(np.float32)
            # 表現一致
            sstk = np.stack([f[f"bpf_{t}{s}"].astype(float) for t in ("ancc", "self", "nbr", "nbr_gr5", "self_graft")], 1)
            f[f"bpf_agree_std{s}"] = np.nanstd(sstk, 1).astype(np.float32)
            f[f"bpf_agree_range{s}"] = (np.nanmax(sstk, 1) - np.nanmin(sstk, 1)).astype(np.float32)
        # --- cross-bank consensus(tw=ancc) ---
        if len(SUFFIXES) >= 2:
            st = np.stack(rep_stacks["tw"], 1)
            pmean = st.mean(1).astype(np.float32); pmed = np.median(st, 1).astype(np.float32)
            f["bpf_ancc_mean"] = pmean; f["bpf_ancc_med"] = pmed
            f["bpf_ancc_std_between"] = st.std(1).astype(np.float32)
            f["bpf_ancc_range_between"] = (st.max(1) - st.min(1)).astype(np.float32)
            f["bpf_ancc_mean_delta"] = (pmean - lk).astype(np.float32)
            f["bpf_ancc_med_delta"] = (pmed - lk).astype(np.float32)
            if tw_tvt is not None and hgr is not None:
                featv._add_gr_match_features(f, "bpf_mean", pmean, hgr, tw_tvt, tw_gr, OFFS)
            if hmd is not None:
                featv._add_path_shape_features(f, "bpf_mean", pmean, hmd)
            # 到達誤差(heel arrival − last_known_tvt)。非リーク(prefixは既知)。
            K = min(20, m)
            f["bpf_arrival_err"] = np.full(m, float(np.nanmean(pmean[:K])) - lk, np.float32)
            f["bpf_arrival_abs"] = np.abs(f["bpf_arrival_err"]).astype(np.float32)
        # --- 表現 集約(mean/pstd) ---
        for rep, tag in [("self", "self"), ("nbr", "nbr"), ("nbr_gr5", "nbr_gr5"), ("self_graft", "self_graft")]:
            RM = np.stack([x.astype(float) for x in rep_stacks[rep]], 1)
            f[f"bpf_{tag}_mean"] = np.nanmean(RM, 1).astype(np.float32)
            f[f"bpf_{tag}_pstd"] = np.nanstd(RM, 1).astype(np.float32)
        # --- 位置(toe からの距離) ---
        if hmd is not None:
            mdt = (hmd[-1] - hmd).astype(np.float32)
            span = float(hmd[-1] - hmd[0]); span = span if span > 1e-6 else 1.0
            f["bpf_md_from_toe"] = mdt; f["bpf_pos_frac"] = (mdt / span).astype(np.float32)
        # --- 出力行(id/well)。id は eval行に対応。build_smoother_inputs の行数=m 前提 ---
        ids = IDMAP.get(wid)
        if ids is not None and len(ids) >= m:
            rows_id.extend(list(ids[:m]))     # 元eval行順(row0=最初のeval行=heel側)
        else:
            rows_id.extend([f"{wid}_{i}" for i in range(m)])
        rows_well.extend([wid] * m)
        for name, arr in f.items():
            put(name, arr)                                # ★numpy配列を append(Python float 化しない)
        n_out += m

    # (6) DataFrame 化・保存(★per-well配列を列ごとに concatenate。欠損列は末尾NaN補正=旧挙動同一)
    L = len(rows_id)
    bpf_cols = sorted(FEAT.keys())
    data = {"id": rows_id, "well": rows_well}
    for c in bpf_cols:
        parts = FEAT[c]
        arr = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if len(arr) != L:
            a = np.full(L, np.nan, np.float32); a[:len(arr)] = arr; arr = a
        data[c] = arr
        FEAT[c] = None                                    # ★列ごとに解放(ピークRAM抑制)
    bdf = pd.DataFrame(data)
    OUT_BWD.parent.mkdir(parents=True, exist_ok=True)
    bdf.to_parquet(OUT_BWD, index=False)
    nb_shape = sum(1 for c in bpf_cols if "_rate" in c or "_acc" in c)
    nb_gr = sum(1 for c in bpf_cols if "_gr_" in c)
    print(f"\nsaved: {OUT_BWD}  rows={len(bdf)} bpf_cols={len(bpf_cols)} (shape={nb_shape} gr={nb_gr}) "
          f"({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
