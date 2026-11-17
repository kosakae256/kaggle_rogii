# -*- coding: utf-8 -*-
"""
v87 create — v69_4 の PF(5バンク p_1..p_5, struct-free)に「affine較正版」15本を追加生成。
============================================================================================
方針(ユーザー指示 2026-07-24):
  - v69_4 の PFパラメータ(p_1..p_5)に対し GR表現を affine 較正して同じ PF を回す。合計15回=5×{tw/self/nbr}。
  - self, neighbor は self/neighbor 疑似typewell を適用した"後"に affine(観測hw GR を参照gridへ α,β 再較正)。
  - 同じ PF でも self/nbr/affine で推論が変わる。その違い(共通点)を特徴量に。
  - v69_4 train.parquet を転用: v9特徴・非affine PF は再計算せず、非affineは pfpass_cache_v69.npz から読む。
    affine 15本だけ新規に回し、id 結合用スリムparquetを出力。
出力: v87/artifacts_v87/data/{SPLIT}_affine_v87.parquet
実行: rogii_claude/.venv/Scripts/python.exe v87/create_affine_v87.py
  env: SMOOTH_LAG(★16=v69_4 基底parquet実測値と一致必須) N_WELLS(0=全) WELL_CHUNK(120) SPLIT(train)
       ※v69_4 の基底parquet/submitは lag16 で運用(doc「本番192」は未実行と実測確認済)。affineも16で揃える。
"""
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v69")); sys.path.insert(0, str(PROJECT / "v87"))
import create_candidates_v69_4 as v69          # ★エンジン名前空間(そのまま eng として渡す)
import affine_lib_v87 as aff

SMOOTH_LAG = int(os.environ.get("SMOOTH_LAG", "16"))   # ★v69_4 基底parquet=lag16(実測確認)。submitも16で揃える
N_WELLS    = int(os.environ.get("N_WELLS", "0")) or None
WELL_CHUNK = int(os.environ.get("WELL_CHUNK", "120"))
SPLIT      = os.environ.get("SPLIT", "train"); IS_TRAIN = (SPLIT == "train")
OUT_PARQUET = PROJECT / "v87" / "artifacts_v87" / "data" / f"{SPLIT}_affine_v87.parquet"
CACHE_NPZ   = PROJECT / "v69" / "artifacts_v69_4" / "data" / "pfpass_cache_v69.npz"
v69.SELF_MIX_W = 1.0; v69.NBR_MIX_W = 1.0
print(f"[v87] SMOOTH_LAG={SMOOTH_LAG}(★v69_4 基底parquet=16と一致必須) N_WELLS={N_WELLS} SPLIT={SPLIT} device={v69.V26_DEVICE}")


def build_names_paths():
    base = v69.DATA_DIR_V26 / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if N_WELLS is not None: wells = wells[:N_WELLS]
    names, paths = [], []; P0 = v69.PF_LIB[v69.PF_BANK[0]]
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns: continue
        if v69.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P0) is None: continue
        names.append(w); paths.append((hp, tp))
    return names, paths


def load_nonaffine_cache():
    if not (CACHE_NPZ.exists() and SPLIT == "train"):
        print(f"[v87] 非affineキャッシュ無し。diff特徴は欠損。"); return None
    z = np.load(CACHE_NPZ, allow_pickle=False); src = {"F": {}, "S": {}, "N": {}}
    for k in z.files:
        w, ens, nm, fld = k.split("|")
        if fld == "mean": src[ens].setdefault(w, {})[nm] = z[k]
    print(f"[v87] 非affine cache 復元: {CACHE_NPZ.name}"); return src


def main():
    names, paths = build_names_paths()
    print(f"[{SPLIT}] 使用坑井 {len(names)}  affine 15本 lag={SMOOTH_LAG} n_seed={v69.N_SEED}")
    NON = load_nonaffine_cache()
    def get_nonaff(ens, wid, nm):
        return None if NON is None else NON[ens].get(wid, {}).get(nm)
    AFF, COEF = aff.run_affine(v69, paths, SMOOTH_LAG=SMOOTH_LAG, N_SEED=v69.N_SEED,
                               SEED_V26=v69.SEED_V26, WELL_CHUNK=WELL_CHUNK, split=SPLIT)
    print("[v87] 特徴量組み立て ...")
    out = aff.assemble_affine_features(v69, paths, AFF, COEF, get_nonaff)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(OUT_PARQUET, index=False)
    cols = [c for c in out.columns if c != "id"]
    print(f"\n[v87] saved: {OUT_PARQUET}  rows={len(out)}  affine列={len(cols)}")
    for pre in ["pf_afftw", "pf_affself", "pf_affnbr", "pf_aff_", "aff_a", "aff_b"]:
        print(f"    {pre:14s} = {sum(c.startswith(pre) for c in cols)} 列")


if __name__ == "__main__":
    main()
