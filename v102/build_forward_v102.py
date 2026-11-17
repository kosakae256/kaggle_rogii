# -*- coding: utf-8 -*-
"""v102 forward-PF feature generator(lean, PF専用パス)— build_backward_v100.py の順方向版。
create_v95 を通さない=CSV毎メソッド再読込 / imputer fit / build_well(beam/NCC/全タブラー)の無駄が無い。

build_backward からの差分:
  - 逆方向処理を全撤去(reverse_inputs / far-end warm-init / dip符号反転 なし)= 素の順方向 PF。
  - MODE(env)で照合GR加工: plain / affine(prefix較正 gr=(gr-b)/a)/ dec10(MD-DEC間引き→interp戻し)。
  - FWD_METHODS で表現を選択(既定5表現。★affine_tw は FWD_METHODS=tw の1表現のみ)。
  - FWD_SMOOTH_MODE で fixedlag(smooth16)/ full を切替。
  - emission ON(FWD_W_NN で config の w_nn を上書き。pf5 config=0.0 対策)。
  - CSV は最初に1回だけ読んでキャッシュ(HWS/TWS)=Kaggle の遅いFSでも再読込オーバーヘッド無し。
  - ★resume: per-(bank,rep) を STORE_CACHE_DIR に pkl 保存。完了済スキップ=中断再開可。

出力 = {id, well, <PRE>_*}(train と行同順, PF列のみ)。非PF特徴は base v102 と同一=生成しない。

env:
  LEG            v96|v97|v100(必須)
  MODE           plain|affine|dec10(既定 plain)
  FWD_METHODS    tw,self,nbr,nbr_gr5,self_graft のカンマ区切り(既定 全5。affine_tw=tw)
  FWD_SMOOTH_MODE fixedlag|full(既定 fixedlag)
  FWD_SMOOTH_LAG fixedlag のラグ(既定 16)
  DEC            間引き幅(MODE=dec10, 既定 10)
  OUT_PREFIX     出力列接頭(既定 MODE依存: aff/dec/fpf)
  ROGII_ART      レグ art dir(pf_banks_config.json / grfree_anchor_train.pkl / nn50 / SIM_PKL)
  SIM_PKL        emission sim pkl(あれば emission込み)
  FWD_W_NN       emission 強度(既定 0.02。全バンク強制)
  OUT_FWD        出力 parquet
  STORE_CACHE_DIR per-(bank,rep) キャッシュ dir(既定 OUT_FWD隣 fwdcache_{LEG}_{MODE})
  NWELLS         スモーク(先頭N井)
  ROGII_DATA     競技データ / PF_NGPU  run_smoother_ext が読む(Kaggle T4x2)
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
import affine_v102 as _aff

MODE = os.environ.get("MODE", "plain").lower()       # plain | affine | dec10 | affine_dec10(併用)
DO_AFFINE = "affine" in MODE                         # ★affine較正を掛ける
DO_DEC = "dec10" in MODE                             # ★MD-DEC 間引きを掛ける(affineと併用可)
DEC = int(os.environ.get("DEC", "10"))
_ALL_METHODS = ["tw", "self", "nbr", "nbr_gr5", "self_graft"]
METHODS = [m.strip() for m in os.environ.get("FWD_METHODS", ",".join(_ALL_METHODS)).split(",") if m.strip()]
METHODS = [m for m in _ALL_METHODS if m in METHODS]  # 正規順序
SMODE = os.environ.get("FWD_SMOOTH_MODE", "fixedlag").lower()   # fixedlag(smooth16) | full
_PRE_DEFAULT = (("aff" if DO_AFFINE else "") + ("dec" if DO_DEC else "")) or "fpf"
PRE = os.environ.get("OUT_PREFIX", _PRE_DEFAULT)

DATA_DIR = Path(os.environ.get("ROGII_DATA", str(pf.DATA_DIR)))
SEED = 4423098
N_SEED = pf.N_SEED
WELL_CHUNK = int(os.environ.get("PF_WELL_CHUNK", "40"))
BANK_ORDER = pf.BANK_ORDER
SUFFIXES = [f"_{i+1}" for i in range(len(BANK_ORDER))]
DEVICE = pf.DEVICE

SIM_PKL = Path(os.environ["SIM_PKL"]) if os.environ.get("SIM_PKL") else None
OUT_FWD = Path(os.environ.get("OUT_FWD", str(_SELF / "artifacts_v102" / "fwd" / f"{PRE}_{LEG}.parquet")))
STORE_CACHE = Path(os.environ.get("STORE_CACHE_DIR", str(OUT_FWD.parent / f"fwdcache_{LEG}_{MODE}")))
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None
FWD_W_NN = float(os.environ.get("FWD_W_NN", "0.02"))
NEED_NBR = ("nbr" in METHODS) or ("nbr_gr5" in METHODS)
OFFS = featv.PF_OFFS
TAG = {"tw": "ancc", "self": "self", "nbr": "nbr", "nbr_gr5": "nbr_gr5", "self_graft": "self_graft"}


# ---------------- 照合GR加工(affine / dec10) ----------------
def _agg1(x, s):
    x = np.asarray(x, float); n = len(x); cuts = list(range(0, n, s)) + [n]
    return np.array([x[a:b].mean() for a, b in zip(cuts[:-1], cuts[1:])])


def _agg2(X, s):
    X = np.asarray(X, float); n = len(X); cuts = list(range(0, n, s)) + [n]
    return np.stack([X[a:b].mean(0) for a, b in zip(cuts[:-1], cuts[1:])], 0)


def _decimate(x):
    n = len(x["md"])
    if n < DEC * 2:
        xd = dict(x); xd["_dec_md"] = np.asarray(x["md"], float); xd["_full_md"] = xd["_dec_md"]; return xd
    xd = dict(md=_agg1(x["md"], DEC), z=_agg1(x["z"], DEC), gr=_agg1(x["gr"], DEC),
              gg=x["gg"], gmin=x["gmin"], gst=x["gst"], gs=x["gs"], ls=x["ls"], ir=x["ir"], N=x["N"], _wid=x.get("_wid"))
    if "anc" in x:
        xd["anc"] = _agg1(x["anc"], DEC); xd["ancs"] = _agg1(x["ancs"], DEC)
    if "_st" in x:
        xd["_st"] = _agg1(x["_st"], DEC); xd["_sim"] = _agg2(x["_sim"], DEC)
    xd["_full_md"] = np.asarray(x["md"], float); xd["_dec_md"] = xd["md"]
    return xd


def _attach_sim(x, wid, SIMD):
    sd = SIMD.get(wid)
    if sd is not None:
        x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    return x


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


def run_bank_rep(paths, names, bank, rep, SIMD, NREFS, NREFS_GR5, HWS, TWS):
    """1 バンク x 1 表現の順方向 smooth-PF。名前->dict(mean,std) 元行順(heel->toe)。"""
    P = pf.bank_param(bank); phys = P["_physics"]
    w_nn = FWD_W_NN                                     # ★全バンク emission 強制
    if SMODE == "full":
        P["smooth_mode"] = "full"                       # ★full平滑(config の smooth_lag を使用)
    else:
        P["smooth_lag"] = int(os.environ.get("FWD_SMOOTH_LAG", "16"))
        P["smooth_mode"] = "fixedlag"                   # ★smooth16
    inps = []; fmds = []
    _tb = time.perf_counter()
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = HWS[wid]; tw = TWS[wid]
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        x = _build_rep(rep, hw, tt, tg, P, wid, NREFS, NREFS_GR5)
        if DO_AFFINE:
            x = _aff.apply_affine(pf, x, hw, tg, P)     # ★prefix較正(間引き前)
        x = pf.attach_anchor(x, wid, phys)
        _attach_sim(x, wid, SIMD)
        if DO_DEC:
            xd = _decimate(x)                            # ★affine後に間引き(併用時=affine+dec10)
            fmds.append((np.asarray(xd["_dec_md"], float), np.asarray(xd["_full_md"], float)))
            inps.append(xd)
        else:
            fmds.append(None)
            inps.append(x)
    t_build = time.perf_counter() - _tb
    _tp = time.perf_counter()
    outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
    t_pf = time.perf_counter() - _tp
    res = {}
    for i, w in enumerate(names):
        o = outs[i]
        if DO_DEC and fmds[i] is not None:
            dm, fm = fmds[i]
            if len(dm) != len(fm):
                mf = np.interp(fm, dm, np.asarray(o["mean"], float)).astype(np.float32)
                sf = np.interp(fm, dm, np.asarray(o["std"], float)).astype(np.float32)
                res[w] = dict(mean=mf, std=sf); continue
        res[w] = dict(mean=np.asarray(o["mean"], np.float32), std=np.asarray(o["std"], np.float32))
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res, t_build, t_pf


def main():
    t_all = time.time()
    print(f"[fwd_v102] LEG={LEG} MODE={MODE} PRE={PRE} methods={METHODS} smooth={SMODE} "
          f"pf={_PFMOD} banks={BANK_ORDER} n_seed={N_SEED} "
          f"emission={'ON' if SIM_PKL and SIM_PKL.exists() else 'OFF'} w_nn={FWD_W_NN} "
          f"PF_NGPU={os.environ.get('PF_NGPU','1')} DEC={DEC if MODE=='dec10' else '-'}", flush=True)

    base = DATA_DIR / os.environ.get("FWD_SPLIT", "train")   # ★test対応: FWD_SPLIT=test で test/ を読む
    # (0) 有効井 + CSVキャッシュ(1回だけ読む)
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
    print(f"[fwd_v102] CSVキャッシュ {len(names)}井 読込 {time.time()-_t_read:.0f}s / passes={len(BANK_ORDER)*len(METHODS)}", flush=True)

    # (1) emission sim(ストリーム読み)
    if SIM_PKL and SIM_PKL.exists():
        with open(SIM_PKL, "rb") as _sf:
            SIMD = pickle.load(_sf)
        print(f"[fwd_v102 emission] sim load: {len(SIMD)} wells (w_nn={FWD_W_NN})", flush=True)
    else:
        SIMD = {}
        print(f"[fwd_v102 emission] sim 無し -> emission OFF", flush=True)

    # (2) imputer + 近傍プール(nbr表現がある時だけ。tw限定=affine_tw では丸ごと skip=高速)
    NREFS = {}; NREFS_GR5 = {}
    if NEED_NBR:
        imp.build_imputers()
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
        print(f"[fwd_v102] 近傍プール: nbr={sum(1 for v in NREFS.values() if v)} gr5={sum(1 for v in NREFS_GR5.values() if v)}", flush=True)
    else:
        print(f"[fwd_v102] nbr表現なし({METHODS})-> imputer/近傍プール skip", flush=True)

    # (3) 各バンク x 各表現 順方向 smooth-PF(★per-(bank,rep) キャッシュで復元可能)
    STORE_CACHE.mkdir(parents=True, exist_ok=True)
    STORE = {rep: {} for rep in METHODS}
    for k, bank in enumerate(BANK_ORDER):
        for rep in METHODS:
            cpath = STORE_CACHE / f"{bank}_{rep}.pkl"
            if cpath.exists():
                d = pickle.loads(cpath.read_bytes())
                if all(w in d for w in names):
                    for w in names:
                        STORE[rep][(w, bank)] = d[w]
                    print(f"[fwd-{rep}] {bank} [cache] 流用({len(names)}井)", flush=True); continue
            t0 = time.perf_counter()
            r, t_build, t_pf = run_bank_rep(paths, names, bank, rep, SIMD, NREFS, NREFS_GR5, HWS, TWS)
            for w in names:
                STORE[rep][(w, bank)] = r[w]
            cpath.write_bytes(pickle.dumps(r, protocol=4))
            print(f"[fwd-{rep}] {bank} done {time.perf_counter()-t0:.1f}s (入力構築{t_build:.1f}s / PF{t_pf:.1f}s)", flush=True)
    import gc
    del SIMD; gc.collect()

    # (4) per-well hgr/hmd/tw グリッド + id(eval行 index)
    HGR = {}; HMD = {}; TWG = {}; IDMAP = {}; LKMAP = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = HWS[wid]; tw = TWS[wid]
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        x0 = pf.build_smoother_inputs(hw, tt, tg, P0)
        if x0 is None:
            continue
        HGR[wid] = np.asarray(x0["gr"], np.float32); HMD[wid] = np.asarray(x0["md"], np.float32)
        TWG[wid] = (tt, tg)
        ev_idx = hw.index[hw["TVT_input"].isna()].to_numpy()
        IDMAP[wid] = np.array([f"{wid}_{int(i)}" for i in ev_idx], dtype=object)
        kn = hw[hw["TVT_input"].notna()]
        LKMAP[wid] = float(kn["TVT_input"].iloc[-1]) if len(kn) else np.nan

    # (5) <PRE>_ 特徴生成(METHODS 部分集合に対応。affine_tw=tw のみでも動く)
    present = [r for r in _ALL_METHODS if r in METHODS]
    has_tw = "tw" in present
    ref_rep = "tw" if has_tw else present[0]
    rows_id = []; rows_well = []; FEAT = {}
    def put(name, arr):
        FEAT.setdefault(name, []).append(np.asarray(arr, np.float32))

    for wid in names:
        bm_ref = {s: np.asarray(STORE[ref_rep][(wid, BANK_ORDER[k])]["mean"], np.float32) for k, s in enumerate(SUFFIXES)}
        m = min(len(v) for v in bm_ref.values())
        hgr = HGR.get(wid); hmd = HMD.get(wid)
        if hgr is not None: hgr = hgr[:m]
        if hmd is not None: hmd = hmd[:m]
        twp = TWG.get(wid); tw_tvt, tw_gr = (twp if twp else (None, None))
        lk = float(LKMAP.get(wid, np.nan))
        f = {}
        rep_stacks = {rep: [] for rep in present}
        for k, s in enumerate(SUFFIXES):
            bank = BANK_ORDER[k]
            for rep in present:
                tag = TAG[rep]
                bm = np.asarray(STORE[rep][(wid, bank)]["mean"], np.float32)[:m]
                bs = np.asarray(STORE[rep][(wid, bank)]["std"], np.float32)[:m]
                f[f"{PRE}_{tag}{s}"] = bm
                f[f"{PRE}_{tag}_std{s}"] = bs
                f[f"{PRE}_{tag}_delta{s}"] = (bm - lk).astype(np.float32)
                rep_stacks[rep].append(bm)
                if rep == "tw":
                    pref = f"{PRE}{s}"
                    if tw_tvt is not None and hgr is not None:
                        featv._add_gr_match_features(f, pref, bm, hgr, tw_tvt, tw_gr, OFFS)
                    if hmd is not None:
                        featv._add_path_shape_features(f, pref, bm, hmd)
            # 表現間比較(存在する表現のみ)
            if has_tw:
                twk = f[f"{PRE}_ancc{s}"]
                for rep in present:
                    if rep == "tw":
                        continue
                    tag = TAG[rep]
                    f[f"{PRE}_{tag}_vs_tw{s}"] = (f[f"{PRE}_{tag}{s}"] - twk).astype(np.float32)
            if "self" in present and "nbr" in present:
                f[f"{PRE}_self_vs_nbr{s}"] = (f[f"{PRE}_self{s}"] - f[f"{PRE}_nbr{s}"]).astype(np.float32)
            if len(present) >= 2:
                sstk = np.stack([f[f"{PRE}_{TAG[t]}{s}"].astype(float) for t in present], 1)
                f[f"{PRE}_agree_std{s}"] = np.nanstd(sstk, 1).astype(np.float32)
                f[f"{PRE}_agree_range{s}"] = (np.nanmax(sstk, 1) - np.nanmin(sstk, 1)).astype(np.float32)
        # cross-bank consensus(tw=ancc)
        if has_tw and len(SUFFIXES) >= 2:
            st = np.stack(rep_stacks["tw"], 1)
            pmean = st.mean(1).astype(np.float32); pmed = np.median(st, 1).astype(np.float32)
            f[f"{PRE}_ancc_mean"] = pmean; f[f"{PRE}_ancc_med"] = pmed
            f[f"{PRE}_ancc_std_between"] = st.std(1).astype(np.float32)
            f[f"{PRE}_ancc_range_between"] = (st.max(1) - st.min(1)).astype(np.float32)
            f[f"{PRE}_ancc_mean_delta"] = (pmean - lk).astype(np.float32)
            f[f"{PRE}_ancc_med_delta"] = (pmed - lk).astype(np.float32)
            if tw_tvt is not None and hgr is not None:
                featv._add_gr_match_features(f, f"{PRE}_mean", pmean, hgr, tw_tvt, tw_gr, OFFS)
            if hmd is not None:
                featv._add_path_shape_features(f, f"{PRE}_mean", pmean, hmd)
        # 非tw表現の集約(存在すれば)
        for rep in present:
            if rep == "tw":
                continue
            tag = TAG[rep]
            if len(rep_stacks[rep]) >= 2:
                RM = np.stack([x.astype(float) for x in rep_stacks[rep]], 1)
                f[f"{PRE}_{tag}_mean"] = np.nanmean(RM, 1).astype(np.float32)
                f[f"{PRE}_{tag}_pstd"] = np.nanstd(RM, 1).astype(np.float32)
        # 位置(heel からの距離=順方向)
        if hmd is not None:
            mdf = (hmd - hmd[0]).astype(np.float32)
            span = float(hmd[-1] - hmd[0]); span = span if span > 1e-6 else 1.0
            f[f"{PRE}_md_since"] = mdf; f[f"{PRE}_pos_frac"] = (mdf / span).astype(np.float32)
        ids = IDMAP.get(wid)
        if ids is not None and len(ids) >= m:
            rows_id.extend(list(ids[:m]))
        else:
            rows_id.extend([f"{wid}_{i}" for i in range(m)])
        rows_well.extend([wid] * m)
        for name, arr in f.items():
            put(name, arr)

    # (6) DataFrame 化・保存
    L = len(rows_id)
    cols = sorted(FEAT.keys())
    data = {"id": rows_id, "well": rows_well}
    for c in cols:
        parts = FEAT[c]
        arr = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if len(arr) != L:
            a = np.full(L, np.nan, np.float32); a[:len(arr)] = arr; arr = a
        data[c] = arr
        FEAT[c] = None
    fdf = pd.DataFrame(data)
    OUT_FWD.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_parquet(OUT_FWD, index=False)
    print(f"\nsaved: {OUT_FWD}  rows={len(fdf)} {PRE}_cols={len(cols)} wells={fdf['well'].nunique()} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
