# -*- coding: utf-8 -*-
"""v93 orchestration: train.parquet ビルダー(v52 build_dataset_v28 を module1-4 に配線)。

★v93 の設計方針(struct 全面禁止 [[no-struct-directive]]):
  - 位置prior はどこでも「GRフリー錨」(module1 attach_anchor / module2 sim帯中心)。
  - 過去 struct(v38/v41/v66/struct_prior_train.pkl/struct_field_v38.npz)は一切読まない。
  - v52 からの唯一の概念変更 = pfA バンクの錨源が struct→GRフリー錨。これは module1/module2 で完結。
    本 orchestration は v52 の特徴エンジン+段取りを忠実再現するだけ。

段取り(v52 build_dataset_v28 と同一構造):
  1. 有効 train 井を列挙(module1.build_smoother_inputs が None でない井)。
  2. sim 読込: v93/artifacts_v93/sim_grfree_v93.pkl(module2 出力)。struct 無し / 再計算無し。
  3. imputer 構築(module3.build_imputers)→ features に注入。
  4. 各バンクで tw/self/nbr smooth-PF を GPU 実行(module1)→ FWD/SELF/NBR_ENS に格納。
  5. Z傾斜(module1.z_gradient_eval)。
  6. 並列(threads)で features.build_well → features.build_self_nbr_columns。
  7. 連結・ガード(>=700井)・parquet 保存・prefix 別列数プリント。

出力: v93/artifacts_v93/data/train.parquet
実行: rogii_claude/.venv/Scripts/python.exe v93/create_v93.py
スモーク: NWELLS=6 python v93/create_v93.py
"""
import os, sys, io, glob, time, pickle
# Windows コンソール安全化(cp932 で em-dash が落ちるので '->' を使う。ここで utf-8 に固定)
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_banks_v93 as pf
import imputers_v93 as imp
import features_v93 as feats

# ==================== CONFIG ====================
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = pf.DATA_DIR
ART = pf.ART
SPLIT = os.environ.get("SPLIT", "train")           # "train" or "test"
IS_TRAIN = (SPLIT == "train")
OUT_PARQUET = (Path(os.environ["OUT_PARQUET"]) if os.environ.get("OUT_PARQUET")
               else ART / "data" / ("train.parquet" if IS_TRAIN else "test.parquet"))
SIM_PKL = (Path(os.environ["SIM_PKL"]) if os.environ.get("SIM_PKL")
           else ART / ("sim_grfree_v93.pkl" if IS_TRAIN else "sim_grfree_test_v93.pkl"))

SEED = 4423098                                   # smoother 用シード(v52 SEED_V26)
N_SEED = pf.N_SEED                               # 32
WELL_CHUNK = 60                                 # GPU バッチ井数(OOM なら下げる)
N_JOBS = 4
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None   # None=全井

DEVICE = pf.DEVICE
BANK_ORDER = pf.BANK_ORDER


def _attach_sim(x, wid, SIMD):
    """PF入力dict に module2 の NN-emission(sim/帯中心 st)を添付。無い井は素通し(注入OFF)。"""
    sd = SIMD.get(wid)
    if sd is not None:
        x["_sim"] = sd["sim"]; x["_st"] = sd["st"]
    return x


def build_dataset():
    base = DATA_DIR / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if NWELLS is not None:
        wells = wells[:NWELLS]

    # (0) 有効井を列挙(先頭バンクの build_smoother_inputs が None でない井のみ)
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
    print(f"[{SPLIT}] 使用坑井: {len(names)}  banks={BANK_ORDER}(tw/self/nbr 各{len(BANK_ORDER)}回=計{3*len(BANK_ORDER)}回)"
          f"  lag={pf.SMOOTH_LAG}  n_seed={N_SEED}")

    # (1) sim 読込(module2 出力。struct 無し / 再計算無し)
    if SIM_PKL.exists():
        SIMD = pickle.loads(SIM_PKL.read_bytes())
        miss = [w for w in names if w not in SIMD]
        print(f"[v93 NN-emission] sim load: {len(SIMD)} wells (対象 miss {len(miss)})")
    else:
        SIMD = {}
        print(f"[v93 NN-emission] WARN: {SIM_PKL.name} 無し -> NN-emission OFF(sim空)")

    # (2) imputer 構築(module3)→ features に注入(二重構築回避)
    FI, DI, _tw = imp.build_imputers()
    feats.set_imputers(FI, DI)
    print(f"[imputers] FI={len(FI.df)}井 / DI点={len(DI.ancc)}")

    # ★PFパス格納先(module4 の global を直接使う)
    FWD_ENS = feats.FWD_ENS
    SELF_ENS = {}; NBR_ENS = {}; ZGRAD = {}; NBR_META = {}
    FWD_ENS.clear()

    # ---- 近傍メタ 前処理(param 非依存, 1回)----
    print("[smoothPF-nbr] 近傍探索 ..."); n_nbr = 0; NREFS = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp, usecols=["X", "Y"])
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        refw = imp.neighbors_of(cx, cy, exclude_wid=(wid if SPLIT == "train" else None))
        use = len(refw) >= imp.NEED_REFS
        NREFS[wid] = [imp.ref_grtvt(r) for r, _ in refw] if use else []
        NBR_META[wid] = dict(has=int(use), nn=imp.nearest_dist(cx, cy, exclude_wid=(wid if SPLIT == "train" else None)), nref=len(refw))
        if use:
            n_nbr += 1
    print(f"      近傍あり {n_nbr}/{len(names)}(残りは通常実行=tw)")

    # (3) 各バンクで tw / self / nbr smooth-PF
    for bank in BANK_ORDER:
        P = pf.bank_param(bank); w_nn = P["_w_nn"]; phys = P["_physics"]
        # (a) tw
        t0 = time.perf_counter(); print(f"[smoothPF-tw] {bank} ...")
        inps = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
            x = pf.attach_anchor(x, wid, phys); _attach_sim(x, wid, SIMD)
            inps.append(x)
        outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
        for i, w in enumerate(names):
            FWD_ENS[(w, bank)] = outs[i]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

        # (b) self
        t0 = time.perf_counter(); print(f"[smoothPF-self] {bank} ...")
        inps = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            x = pf.build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
            x = pf.attach_anchor(x, wid, phys); _attach_sim(x, wid, SIMD)
            inps.append(x)
        outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
        for i, w in enumerate(names):
            SELF_ENS[(w, bank)] = outs[i]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

        # (c) nbr
        t0 = time.perf_counter(); print(f"[smoothPF-nbr] {bank} ...")
        inps = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            x = pf.build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P, NREFS[wid])
            x = pf.attach_anchor(x, wid, phys); _attach_sim(x, wid, SIMD)
            inps.append(x)
        outs = pf.run_smoother_ext(inps, P, SEED, N_SEED, WELL_CHUNK, w_nn=w_nn)
        for i, w in enumerate(names):
            NBR_ENS[(w, bank)] = outs[i]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # (4) Z傾斜
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        ZGRAD[wid] = pf.z_gradient_eval(pd.read_csv(hp))

    # (5) 並列 build_well + self/nbr/z_grad 列
    def one(w, hp, tp):
        feats.set_current_well(w)
        df = feats.build_well(str(hp), str(tp), IS_TRAIN)
        if df is None or len(df) == 0:
            return None
        return feats.build_self_nbr_columns(df, w, SELF_ENS, NBR_ENS, ZGRAD, NBR_META)

    print("[build] build_well(smooth-PF注入) + self/nbr/z_grad 列 ...")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1]) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s  parts={len(parts)}")
    if not parts:
        raise RuntimeError("有効な坑井がありません。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"[v93] device={DEVICE} smooth_lag={pf.SMOOTH_LAG} n_seed={N_SEED} chunk={WELL_CHUNK} NWELLS={NWELLS}")
    df = build_dataset()
    if SPLIT == "train" and NWELLS is None and df["well"].nunique() < 700:
        raise SystemExit(f"[GUARD] 出力坑井数 {df['well'].nunique()} 本は異常(全773本のはず)。")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    cols = list(df.columns)
    print(f"\nsaved: {OUT_PARQUET}  rows={len(df)} cols={len(cols)} wells={df['well'].nunique()}")
    for pre in ["pf_", "beam", "tvtF", "form", "dense", "sc"]:
        print(f"  {pre}* = {sum(c.startswith(pre) for c in cols)} 列")


if __name__ == "__main__":
    main()
