# -*- coding: utf-8 -*-
"""v93 backward-PF feature generator (bpf_* columns).

BACKWARD の定義:
  各坑井の eval 区間(TVT_input が NaN の行)を「逆順」にして、v93 の smooth-PF を
  そのまま(同一 param / smooth_lag=16 / n_seed=32 / 同一錨・同一 NN-emission)走らせる。
  つまり far-end(坑井の奥)から prefix 側へ向かって逐次推定した平滑軌跡を得る。
  出力は元の行順に並べ戻し、★forward の pf_* と同格の full-mirror 特徴族を bpf_ prefix で作る
  (features_v93 の派生器 _add_gr_match_features/_add_path_shape_features/consensus/self-nbr を再利用):
    per-bank : bpf_ancc/std/delta, bpf{s}_rate/acc/..(shape), bpf{s}_gr_resid0/best_off/..(GR一致)
    consensus: bpf_ancc_mean/med/std_between/range/iqr/pair/corr/dev/diff/absdiff + mean/med の GR一致・shape
    self/nbr : bpf_self/nbr の std/delta/vs_tw/vs_nbr/agree + 集約(mean/pstd/mean_vs_tw)
  ★backward固有(新規):
    - bpf_arrival_err/abs, bpf_arrival{s}: backward が heel(prefix)に到達した推定 − 既知 last_known_tvt。
      真値を使うが prefix は test でも既知 = 非リーク。backward の信頼性(ドリフト)シグナル。常時出力。
    - bpf_md_from_toe / bpf_pos_frac    : backward の錨(toe)からの距離。fwd(heel錨)/bwd(toe錨)の
      信頼を位置で切替える軸。常時出力。
    - bpf_vs_fwd{s}/_abs, bpf_mean_vs_fwd: forward との差(ユーザー要望)。EMIT_VS_FWD で束ね【既定ON】。
      v10 で cmp_ は有害だった前科があるので、効かなければ EMIT_VS_FWD=0 でこの束だけ外して再学習可。真値不使用=非リーク。

逆順の作り方(faithful reuse。build_smoother_inputs/self/nbr の入力 dict をそのまま流用):
  - md   : mirror 反転 md_rev = md[0] + (md[-1] - md[::-1])。
           これは増加列を保ちつつ「隣接 step の大きさを逆順」にする。
           smoother core の dm = (cur - pm).clamp_min(1.0) は、単純 [::-1](減少列)だと
           dm が全て負->1.0 に潰れ step 情報が死ぬ。mirror なら dm = |逆順 step| となり
           forward と同じ遷移ダイナミクス(step 大きさは真の MD 間隔の逆順)を保てる。
  - z, gr, anc, ancs, _sim, _st : per-eval-row 配列を [::-1] で反転(行 i = 元の行 n-1-i)。
  - gg / gmin / gst / gs / N     : typewell グリッド系は方向非依存なので不変。
  - ls / ir                      : prefix 由来の初期 level/dip。ユーザー指示どおり「同一」で使う
                                   (far-end 始点だが init は forward と同じ値)。
  走らせた後、mean / std を [::-1] で元行順へ戻す。

struct-free: 錨は grfree_anchor_train.pkl(GRフリー)のみ。struct/v38/v41/gf/v50b 系は一切 import しない。
出力:
  - v93/artifacts_v93/backward_feats.parquet         (id, well, bpf_*)  train.parquet と行同順
  - v93/artifacts_v93/data/train_fwdbwd.parquet       train.parquet + bpf_*(GBM が1ファイルで読める)
実行:  rogii_claude/.venv/Scripts/python.exe v93/build_backward_v93.py
スモーク: NWELLS=40 python v93/build_backward_v93.py  (train_fwdbwd マージはスキップ)
"""
import os, sys, io, glob, time, pickle
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v93 as pf
import imputers_v93 as imp
import features_v93 as featv     # forward の派生器(_add_gr_match_features/_add_path_shape_features 等)を bpf_ で再利用

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = pf.DATA_DIR
ART = pf.ART
SPLIT = os.environ.get("SPLIT", "train").lower()   # train / test(提出)
IS_TRAIN = (SPLIT == "train")

SEED = 4423098                 # smoother 用シード(create_v93 と同一)
N_SEED = pf.N_SEED             # 32
WELL_CHUNK = 40                # GPU バッチ井数(lag16 は軽い)
BANK_ORDER = pf.BANK_ORDER     # [pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA]
SUFFIXES = [f"_{i+1}" for i in range(len(BANK_ORDER))]   # _1.._6

# forward parquet / sim は split で切替(env で明示上書き可。test 提出は submit が指定)
FWD_PARQUET = Path(os.environ.get("V93_FWD_PARQUET",
                                  str(ART / "data" / ("train.parquet" if IS_TRAIN else "test.parquet"))))
SIM_PKL = Path(os.environ.get("SIM_PKL",
                              str(ART / ("sim_grfree_v93.pkl" if IS_TRAIN else "sim_grfree_test_v93.pkl"))))
OUT_BWD = ART / f"backward_feats{'' if IS_TRAIN else '_test'}.parquet"
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None
DEVICE = pf.DEVICE

# ---- backward の初期化モード ----------------------------------------------------------------
#   "warm" (既定): far-end 始点に見合う初期化。level を forward-PF の far-end 推定
#                  (parquet の pf_ancc/pf_self/pf_nbr の最終eval行)から取り、dip は -ir(逆走なので符号反転)。
#                  -> 幾何的に正しい far-end 始点で GR 追従する「本物の逆走 PF」。
#   "same"       : ユーザー当初指示どおり ls/ir を forward と完全同一(=prefix datum)で使う。
#                  ★診断: ls = prefix(TVT+Z) は「prefix の構造datum」。far-end では Z が数十〜百ft
#                  ドリフトしており、この始点は真の far-end TVT から ~90-120ft ずれる。GR エイリアスで
#                  そのまま固着し軌跡が平坦化する(smoke で実測: bpf 平坦, vs_fwd absmean~122ft)。
#                  忠実さの記録として残すが、特徴としては劣化。比較用。
#   "anchor"     : ★物理(GRフリー錨)の終端 far-end(x["anc"][-1])を level に使う。forward の答えに
#                  依存しない独立 init = 真の第2推定器(v10 の準重複問題を構造回避)。dip は同じく -ir。
BWD_INIT = os.environ.get("BWD_INIT", "warm").lower()
# ---- variant 出力名(warm=既定は無印, それ以外は _<init>。test は接頭 test_ + OUT_MERGED env 上書き可)----
VSUF = "" if BWD_INIT == "warm" else f"_{BWD_INIT}"
_PRE = "train" if IS_TRAIN else "test"
OUT_BWD = ART / f"backward_feats{'' if IS_TRAIN else '_test'}{VSUF}.parquet"
OUT_MERGED = (Path(os.environ["OUT_MERGED"]) if os.environ.get("OUT_MERGED")
              else ART / "data" / f"{_PRE}_fwdbwd{VSUF}.parquet")
# forward 比較特徴(bpf_vs_fwd{s}/_abs, bpf_mean_vs_fwd)を出すか。
#   ★既定ON(ユーザー要望「forwardとの差/比較」)。v10 で cmp_ は有害だった前科があるので、
#     効かなければ EMIT_VS_FWD=0 でこの束だけ外して再学習できるよう1フラグに束ねてある。
#     (真値は使わない=非リーク。真値を使う到達誤差 bpf_arrival_* は常時出力=別枠。)
EMIT_VS_FWD = os.environ.get("EMIT_VS_FWD", "1") == "1"
_REP2FWD = {"tw": "pf_ancc", "self": "pf_self", "nbr": "pf_nbr"}


def _attach_sim(x, wid, SIMD):
    sd = SIMD.get(wid)
    if sd is not None:
        x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    return x


def reverse_inputs(x):
    """PF入力 dict を eval 区間逆順にする。per-eval-row 配列のみ反転。他はそのまま。"""
    xr = dict(x)
    md = np.asarray(x["md"], float)
    xr["md"] = md[0] + (md[-1] - md[::-1])            # mirror: 増加列 + step 大きさ逆順
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


def run_bank_rep(paths, names, bank, rep, SIMD, NREFS, sfx, FWD_LAST):
    """1 バンク x 1 表現(tw/self/nbr)の backward smooth-PF。名前->dict(mean,std) を返す(元行順)。
       BWD_INIT=='warm' のとき ls/ir を far-end 用に差し替える(forward far-end level + dip 符号反転)。"""
    P = pf.bank_param(bank); w_nn = P["_w_nn"]; phys = P["_physics"]
    inps = []
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        if rep == "tw":
            x = pf.build_smoother_inputs(hw, tt, tg, P)
        elif rep == "self":
            x = pf.build_inputs_self(hw, tt, tg, P)
        else:
            x = pf.build_inputs_nbr(hw, tt, tg, P, NREFS[wid])
        x = pf.attach_anchor(x, wid, phys)
        _attach_sim(x, wid, SIMD)
        if BWD_INIT == "warm":
            # far-end 始点に見合う level/dip へ差し替え(x['z'] は元行順の far-end = 最終要素)。
            fwd_tvt_last = FWD_LAST.get((wid, sfx, rep))
            if fwd_tvt_last is not None and np.isfinite(fwd_tvt_last):
                x["ls"] = float(fwd_tvt_last) + float(x["z"][-1])   # level = far-end TVT + far-end Z
            x["ir"] = -float(x["ir"])                               # 逆走なので dip 符号反転
        elif BWD_INIT == "anchor":
            # ★物理(GRフリー錨)の終端で init = forward の答えに依存しない独立 init。
            #   ★重要: x["anc"] は _physics=False バンク(pf_1..r1)ではゼロ化される(attach_anchor)。
            #   なので錨は pf._ANCHOR から直接読む(全バンクで正しい錨tvt を使う)。
            sp = pf._ANCHOR.get(wid)
            if sp is not None and len(sp["tvt"]) == len(x["z"]) and np.isfinite(sp["tvt"][-1]):
                x["ls"] = float(sp["tvt"][-1]) + float(x["z"][-1])   # level = 錨far-end TVT + far-end Z
            x["ir"] = -float(x["ir"])
        inps.append(reverse_inputs(x))            # ★逆順にして走らせる
    outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
    res = {}
    for i, w in enumerate(names):
        o = outs[i]
        res[w] = dict(mean=o["mean"][::-1].copy(), std=o["std"][::-1].copy())   # ★元行順へ戻す
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    print(f"[v93 backward] device={DEVICE} smooth_lag={pf.SMOOTH_LAG} n_seed={N_SEED} "
          f"chunk={WELL_CHUNK} NWELLS={NWELLS} banks={BANK_ORDER}")
    base = DATA_DIR / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if NWELLS is not None:
        wells = wells[:NWELLS]

    # (0) 有効井(create_v93 と同一フィルタ: 先頭バンク build_smoother_inputs が None でない井)
    P0 = pf.bank_param(BANK_ORDER[0])
    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns:
            continue
        if pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[backward] 使用坑井: {len(names)}  passes={3*len(BANK_ORDER)}(bank{len(BANK_ORDER)} x rep3)")
    print(f"[backward] init mode = {BWD_INIT}  (warm=far-end level 差替+dip反転 / same=ユーザー当初指示の完全同一)")

    # (0.5) forward parquet 読込(整合用 + warm 初期化の far-end level 供給)
    need_cols = (["id", "well", "last_known_tvt"]
                 + [f"pf_ancc{s}" for s in SUFFIXES]
                 + [f"pf_self{s}" for s in SUFFIXES]
                 + [f"pf_nbr{s}" for s in SUFFIXES])
    fwd = pd.read_parquet(FWD_PARQUET, columns=need_cols).reset_index(drop=True)
    N = len(fwd)
    name_set = set(names)
    have = fwd["well"].isin(name_set)                 # 計算済み坑井のみ(full=全True / smoke=先頭N井)
    fwd_have = fwd[have]
    print(f"[align] train.parquet rows={N} wells={fwd['well'].nunique()}  "
          f"computed rows={int(have.sum())} wells={fwd_have['well'].nunique()}")

    # warm 用: (wid, sfx, rep) -> forward far-end(最終eval行)の TVT 推定
    FWD_LAST = {}
    if BWD_INIT == "warm":
        for wid, g in fwd_have.groupby("well", sort=False):
            for s in SUFFIXES:
                for rep in ("tw", "self", "nbr"):
                    col = f"{_REP2FWD[rep]}{s}"
                    v = g[col].to_numpy(np.float64)
                    vv = v[np.isfinite(v)]
                    if len(vv) == 0:                  # self/nbr が全NaN の井は pf_ancc へフォールバック
                        v2 = g[f"pf_ancc{s}"].to_numpy(np.float64); v2 = v2[np.isfinite(v2)]
                        vv = v2
                    FWD_LAST[(wid, s, rep)] = float(vv[-1]) if len(vv) else np.nan

    # (1) sim 読込(NN-emission。forward と同一)
    if SIM_PKL.exists():
        SIMD = pickle.loads(SIM_PKL.read_bytes())
        print(f"[backward NN-emission] sim load: {len(SIMD)} wells")
    else:
        SIMD = {}
        print(f"[backward NN-emission] WARN: {SIM_PKL.name} 無し -> NN-emission OFF")

    # (2) imputer 構築(nbr 表現の近傍プールに必要)
    FI, DI, _tw = imp.build_imputers()
    print(f"[imputers] FI={len(FI.df)}井 / DI点={len(DI.ancc)}")

    # (3) 近傍プール 前処理(param 非依存, 1回。create_v93 と同一)
    print("[backward-nbr] 近傍探索 ..."); NREFS = {}; n_nbr = 0
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hwxy = pd.read_csv(hp, usecols=["X", "Y"])
        cx, cy = float(hwxy["X"].mean()), float(hwxy["Y"].mean())
        refw = imp.neighbors_of(cx, cy, exclude_wid=wid)
        use = len(refw) >= imp.NEED_REFS
        NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if use else []
        if use:
            n_nbr += 1
    print(f"      近傍あり {n_nbr}/{len(names)}")

    # (4) 各バンク x 各表現で backward smooth-PF
    BWD = {}; BSELF = {}; BNBR = {}
    for k, bank in enumerate(BANK_ORDER):
        s = SUFFIXES[k]
        for rep, store in (("tw", BWD), ("self", BSELF), ("nbr", BNBR)):
            t0 = time.perf_counter(); print(f"[backward-{rep}] {bank} ...", end="", flush=True)
            r = run_bank_rep(paths, names, bank, rep, SIMD, NREFS, s, FWD_LAST)
            for w in names:
                store[(w, bank)] = r[w]
            print(f" done {time.perf_counter()-t0:.1f}s")

    # (5) full-mirror 特徴 + backward固有特徴を行同順で組む。
    #   forward の派生器(features_v93)を bpf_ プレフィックスで再利用 = forward pf_* と同格の族。
    #     per-bank: shape(rate/acc/rolling) + gr一致(resid/best_off/rolling) + ancc/std/delta
    #     consensus: mean/med/std_between/range/iqr/pair/corr/dev/diff/absdiff + mean/med の gr一致・shape
    #     self/nbr : std/delta/vs_tw/vs_nbr/agree_std/range + 集約(mean/pstd/mean_vs_tw)
    #   ★backward固有(新規):
    #     bpf_arrival_err/abs, bpf_arrival{s} = backward が heel(prefix)に到達した推定 − 既知 last_known_tvt
    #        (真値使用だが prefix は test でも既知 = 非リーク。backward の信頼性シグナル)
    #     bpf_md_from_toe / bpf_pos_frac       = backwardの錨(toe)からの距離(位置で fwd/bwd の信頼を切替)
    #     bpf_vs_fwd{s}/_abs, bpf_mean_vs_fwd  = forward との差(EMIT_VS_FWD で束ねる。既定ON)
    OFFS = featv.PF_OFFS
    OUT = {}                                        # colname -> full-N float32(遅延確保)
    def put(name, ii, arr):
        a = OUT.get(name)
        if a is None:
            a = np.full(N, np.nan, np.float32); OUT[name] = a
        a[ii] = np.asarray(arr, np.float32)
    lk_all = fwd["last_known_tvt"].to_numpy(np.float32)
    FA = {s: fwd[f"pf_ancc{s}"].to_numpy(np.float32) for s in SUFFIXES}   # 比較用 forward
    vs_diag = np.full(N, np.nan, np.float32)        # 診断(bank1 順逆乖離)

    # per-well hgr/hmd/tw グリッド(gr一致・shape・位置に必要)を build_smoother_inputs(P0) で1回供給
    HGR = {}; HMD = {}; TWG = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        tt = tw["TVT"].to_numpy(float); tg = tw["GR"].to_numpy(float)
        x0 = pf.build_smoother_inputs(hw, tt, tg, P0)
        if x0 is None:
            continue
        HGR[wid] = np.asarray(x0["gr"], np.float32); HMD[wid] = np.asarray(x0["md"], np.float32)
        TWG[wid] = (tt, tg)

    n_len_mismatch = 0
    for wid, g in fwd_have.groupby("well", sort=False):
        idx = g.index.to_numpy(); n = len(idx)
        bm0 = BWD[(wid, BANK_ORDER[0])]["mean"]; m = min(n, len(bm0))
        if m != n:
            n_len_mismatch += 1
        if m == 0:
            continue
        ii = idx[:m]; lk = lk_all[idx][:m]
        hgr = HGR.get(wid); hmd = HMD.get(wid)
        if hgr is not None: hgr = hgr[:m]
        if hmd is not None: hmd = hmd[:m]
        twp = TWG.get(wid); tw_tvt, tw_gr = (twp if twp else (None, None))
        f = {}                                      # 本 well の bpf_ 列(すべて len=m)

        # --- per-bank(tw): ancc/std/delta + gr一致 + shape + 比較 + 到達誤差 ---
        tw_stack = []
        for k, s in enumerate(SUFFIXES):
            bank = BANK_ORDER[k]
            bm = np.asarray(BWD[(wid, bank)]["mean"], np.float32)[:m]
            bs = np.asarray(BWD[(wid, bank)]["std"], np.float32)[:m]
            f[f"bpf_ancc{s}"] = bm
            f[f"bpf_ancc_std{s}"] = bs
            f[f"bpf_ancc_delta{s}"] = (bm - lk).astype(np.float32)
            pref = f"bpf{s}"
            if tw_tvt is not None and hgr is not None:
                featv._add_gr_match_features(f, pref, bm, hgr, tw_tvt, tw_gr, OFFS)
            if hmd is not None:
                featv._add_path_shape_features(f, pref, bm, hmd)
            tw_stack.append(bm)
            if EMIT_VS_FWD:
                d = (bm - FA[s][idx][:m]).astype(np.float32)
                f[f"bpf_vs_fwd{s}"] = d; f[f"bpf_vs_fwd_abs{s}"] = np.abs(d).astype(np.float32)
            K = min(20, m)
            f[f"bpf_arrival{s}"] = np.full(m, float(np.nanmean(bm[:K])) - float(lk[0]), np.float32)
        vs_diag[ii] = (f["bpf_ancc_1"] - FA["_1"][idx][:m])

        # --- cross-bank consensus(tw) ---
        if len(tw_stack) >= 2:
            st = np.stack(tw_stack, 1)
            pmean = st.mean(1).astype(np.float32); pmed = np.median(st, 1).astype(np.float32)
            pmn, pmx = featv._pairwise_abs_stats(st); cmean, cmin, cmax = featv._corr_summary(st)
            f["bpf_ancc_mean"] = pmean; f["bpf_ancc_med"] = pmed
            f["bpf_ancc_std_between"] = st.std(1).astype(np.float32)
            f["bpf_ancc_range_between"] = (st.max(1) - st.min(1)).astype(np.float32)
            f["bpf_ancc_iqr_between"] = (np.percentile(st, 75, 1) - np.percentile(st, 25, 1)).astype(np.float32)
            f["bpf_ancc_pair_absmean_between"] = pmn; f["bpf_ancc_pair_absmax_between"] = pmx
            f["bpf_ancc_mean_delta"] = (pmean - lk).astype(np.float32)
            f["bpf_ancc_med_delta"] = (pmed - lk).astype(np.float32)
            f["bpf_ancc_corr_mean"] = np.full(m, cmean, np.float32)
            f["bpf_ancc_corr_min"] = np.full(m, cmin, np.float32)
            f["bpf_ancc_corr_max"] = np.full(m, cmax, np.float32)
            for k, s in enumerate(SUFFIXES):
                f[f"bpf_dev_mean{s}"] = (st[:, k] - pmean).astype(np.float32)
                f[f"bpf_dev_med{s}"] = (st[:, k] - pmed).astype(np.float32)
            for i in range(len(SUFFIXES)):
                for j in range(i + 1, len(SUFFIXES)):
                    si = SUFFIXES[i].replace("_", ""); sj = SUFFIXES[j].replace("_", "")
                    dd = (st[:, i] - st[:, j]).astype(np.float32)
                    f[f"bpf_diff_{si}_{sj}"] = dd; f[f"bpf_absdiff_{si}_{sj}"] = np.abs(dd).astype(np.float32)
            if tw_tvt is not None and hgr is not None:
                featv._add_gr_match_features(f, "bpf_mean", pmean, hgr, tw_tvt, tw_gr, OFFS)
                featv._add_gr_match_features(f, "bpf_med", pmed, hgr, tw_tvt, tw_gr, OFFS)
            if hmd is not None:
                featv._add_path_shape_features(f, "bpf_mean", pmean, hmd)
                featv._add_path_shape_features(f, "bpf_med", pmed, hmd)
            K = min(20, m)
            f["bpf_arrival_err"] = np.full(m, float(np.nanmean(pmean[:K])) - float(lk[0]), np.float32)
            f["bpf_arrival_abs"] = np.abs(f["bpf_arrival_err"]).astype(np.float32)
            if EMIT_VS_FWD:
                fam = np.nanmean(np.stack([FA[s][idx][:m] for s in SUFFIXES], 1), 1).astype(np.float32)
                f["bpf_mean_vs_fwd"] = (pmean - fam).astype(np.float32)

        # --- self/nbr(per bank + 集約。build_self_nbr_columns の bpf 版) ---
        S_all = []; N_all = []; T_all = []
        for k, s in enumerate(SUFFIXES):
            bank = BANK_ORDER[k]
            ps = np.asarray(BSELF[(wid, bank)]["mean"], np.float32)[:m]; pss = np.asarray(BSELF[(wid, bank)]["std"], np.float32)[:m]
            pn = np.asarray(BNBR[(wid, bank)]["mean"], np.float32)[:m]; pns = np.asarray(BNBR[(wid, bank)]["std"], np.float32)[:m]
            twk = f[f"bpf_ancc{s}"]
            f[f"bpf_self{s}"] = ps; f[f"bpf_self_std{s}"] = pss; f[f"bpf_self_delta{s}"] = (ps - lk).astype(np.float32)
            f[f"bpf_nbr{s}"] = pn;  f[f"bpf_nbr_std{s}"] = pns;  f[f"bpf_nbr_delta{s}"] = (pn - lk).astype(np.float32)
            f[f"bpf_self_vs_tw{s}"] = (ps - twk).astype(np.float32)
            f[f"bpf_nbr_vs_tw{s}"] = (pn - twk).astype(np.float32)
            f[f"bpf_self_vs_nbr{s}"] = (ps - pn).astype(np.float32)
            sstk = np.stack([twk.astype(float), ps.astype(float), pn.astype(float)], 1)
            f[f"bpf_agree_std{s}"] = np.nanstd(sstk, 1).astype(np.float32)
            f[f"bpf_agree_range{s}"] = (np.nanmax(sstk, 1) - np.nanmin(sstk, 1)).astype(np.float32)
            S_all.append(ps.astype(float)); N_all.append(pn.astype(float)); T_all.append(twk.astype(float))
        SM = np.stack(S_all, 1); NM = np.stack(N_all, 1); TM = np.stack(T_all, 1)
        f["bpf_self_mean"] = np.nanmean(SM, 1).astype(np.float32); f["bpf_self_pstd"] = np.nanstd(SM, 1).astype(np.float32)
        f["bpf_nbr_mean"] = np.nanmean(NM, 1).astype(np.float32); f["bpf_nbr_pstd"] = np.nanstd(NM, 1).astype(np.float32)
        f["bpf_self_mean_vs_tw"] = (np.nanmean(SM, 1) - np.nanmean(TM, 1)).astype(np.float32)
        f["bpf_nbr_mean_vs_tw"] = (np.nanmean(NM, 1) - np.nanmean(TM, 1)).astype(np.float32)

        # --- 新規: 位置(backward の錨 = toe からの距離) ---
        if hmd is not None:
            mdt = (hmd[-1] - hmd).astype(np.float32)
            span = float(hmd[-1] - hmd[0]); span = span if span > 1e-6 else 1.0
            f["bpf_md_from_toe"] = mdt; f["bpf_pos_frac"] = (mdt / span).astype(np.float32)

        for name, arr in f.items():
            put(name, ii, arr)
    if n_len_mismatch:
        print(f"[align] WARN: {n_len_mismatch} (well,bank) で長さ不一致(min で切詰)")

    bpf_names = sorted(OUT.keys())
    outcols = OUT
    nb_shape = sum(1 for c in bpf_names if "_rate" in c or "_acc" in c)
    nb_gr = sum(1 for c in bpf_names if "_gr_" in c)
    print(f"[bpf] full-mirror 生成: {len(bpf_names)}列 (shape={nb_shape} gr一致={nb_gr} "
          f"vs_fwd={sum(1 for c in bpf_names if 'vs_fwd' in c)} arrival={sum(1 for c in bpf_names if 'arrival' in c)})")

    bdf = pd.DataFrame({"id": fwd["id"].to_numpy(), "well": fwd["well"].to_numpy(),
                        **{nm: outcols[nm] for nm in bpf_names}})
    bdf = bdf[have.to_numpy()].reset_index(drop=True)   # full=全行 / smoke=計算済み井のみ
    if os.environ.get("SKIP_STANDALONE", "0") == "1":
        print(f"\n[SKIP_STANDALONE=1] {OUT_BWD.name} は書かない(disk節約)。merged のみ生成。bpf_cols={len(bpf_names)}")
    else:
        OUT_BWD.parent.mkdir(parents=True, exist_ok=True)
        bdf.to_parquet(OUT_BWD, index=False)
        print(f"\nsaved: {OUT_BWD}  rows={len(bdf)} bpf_cols={len(bpf_names)}")

    # (6) train_fwdbwd.parquet = train.parquet + bpf_*(row-group streaming で省メモリ)
    if NWELLS is not None:
        print("[merge] スモーク(NWELLS 指定)なので train_fwdbwd マージはスキップ。")
    else:
        bcols = {nm: pa.array(bdf[nm].to_numpy(np.float32)) for nm in bpf_names}
        pfin = pq.ParquetFile(FWD_PARQUET)
        writer = None; off = 0
        for rg in range(pfin.num_row_groups):
            tbl = pfin.read_row_group(rg)
            n = tbl.num_rows
            for nm in bpf_names:
                tbl = tbl.append_column(nm, bcols[nm].slice(off, n))
            off += n
            if writer is None:
                writer = pq.ParquetWriter(OUT_MERGED, tbl.schema)
            writer.write_table(tbl)
        writer.close()
        print(f"saved: {OUT_MERGED}  rows={off}  (+{len(bpf_names)} bpf cols)")

    # (7) sanity check
    print(f"\n[sanity] corr(bpf_ancc_1, forward pf_ancc_1) per well(先頭8井) + 順逆乖離分布 "
          f"[EMIT_VS_FWD={EMIT_VS_FWD}]")
    fa1 = fwd["pf_ancc_1"].to_numpy(np.float32)
    ba1 = outcols["bpf_ancc_1"]
    cnt = 0
    for wid, g in fwd_have.groupby("well", sort=False):
        ix = g.index.to_numpy()
        a = fa1[ix]; b = ba1[ix]
        mm = np.isfinite(a) & np.isfinite(b)
        if mm.sum() > 5 and np.std(a[mm]) > 1e-6 and np.std(b[mm]) > 1e-6:
            c = float(np.corrcoef(a[mm], b[mm])[0, 1])
            print(f"    {wid}: n={mm.sum():5d} corr={c:+.4f}")
            cnt += 1
        if cnt >= 8:
            break
    vs = vs_diag[np.isfinite(vs_diag)]
    print(f"[sanity] 順逆乖離(bank1, 診断のみ/特徴非出力): mean={vs.mean():+.3f} std={vs.std():.3f} "
          f"absmean={np.abs(vs).mean():.3f} p50={np.percentile(np.abs(vs),50):.3f} "
          f"p90={np.percentile(np.abs(vs),90):.3f} max={np.abs(vs).max():.3f}")


if __name__ == "__main__":
    main()
