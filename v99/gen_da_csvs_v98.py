# -*- coding: utf-8 -*-
"""v98 DataAugmentation の CSV 生成(逆方向DA + 順方向DA)。CPUのみ・高速。
生成した反転/中点開始CSVは、後段で ROGII_DATA を各ルートへ向けて create_v95+create_v97 を通す
(v36の手口=鏡像/半開始の世界で全特徴が自動整合)。well名は保持(GroupKFoldで元井と同fold=リークなし)。

- **逆DA(v36再現)**: evalゾーンのみ抽出→行反転(MD'=MDmax−MD)→toe側K行を疑似既知帯(TVT_input:=TVT,
  K=min(既知帯行数, n_ev//2), 下限30)→残り n_ev−K 行を予測。既知帯行はCSVごと除外。
- **順DA(新規=中点開始)**: 反転なし。元の既知帯 + eval前半 n_ev//2 行を既知帯化(TVT_input:=TVT)、
  eval後半 n_ev−n_ev//2 行のみ予測対象(NaN)。全行保持。

出力: <OUT_ROOT>/{da_bwd,da_fwd}/train/{w}__horizontal_well.csv (+ __typewell.csv コピー)
env: MODE(bwd|fwd|both 既定both) / NWELLS(スモーク) / OUT_ROOT(既定 v98/artifacts_v99)
実行: .venv/Scripts/python.exe v98/gen_da_csvs_v98.py
"""
import os, sys, io, glob, shutil, time
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd

V98 = Path(__file__).resolve().parent
PROJ = V98.parent
DATA = Path(os.environ.get("ROGII_DATA", str(PROJ / "rogii-wellbore-geology-prediction")))
SRC_TRAIN = DATA / "train"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", str(V98 / "artifacts_v99")))
MODE = os.environ.get("MODE", "both")
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None
K_MIN, EV_MIN = 30, 50


def _wells():
    ws = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(SRC_TRAIN / "*__horizontal_well.csv"))})
    return ws[:NWELLS] if NWELLS is not None else ws


def gen_backward(dst_train):
    """v36再現: evalゾーンのみ反転→toe側K行を疑似既知帯。"""
    dst_train.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0; t0 = time.time()
    for w in _wells():
        hp = SRC_TRAIN / f"{w}__horizontal_well.csv"; tp = SRC_TRAIN / f"{w}__typewell.csv"
        if not tp.exists():
            n_skip += 1; continue
        hw = pd.read_csv(hp)
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            n_skip += 1; continue
        kn = hw["TVT_input"].notna().to_numpy(); n_kn = int(kn.sum()); n_ev = int((~kn).sum())
        K = min(n_kn, n_ev // 2)
        if K < K_MIN or (n_ev - K) < EV_MIN:
            n_skip += 1; continue
        ev = hw[~kn]                                              # ★元evalゾーンのみ(既知帯除外)
        hw2 = ev.iloc[::-1].reset_index(drop=True).copy()
        hw2["MD"] = float(ev["MD"].max()) - hw2["MD"]            # MD再び昇順
        head_tvt = hw2["TVT"].to_numpy()[:K]
        if not np.isfinite(head_tvt).all():
            n_skip += 1; continue
        tvt_in = np.full(len(hw2), np.nan); tvt_in[:K] = head_tvt
        hw2["TVT_input"] = tvt_in                                # 予測 = n_ev−K 行
        hw2.to_csv(dst_train / f"{w}__horizontal_well.csv", index=False)
        shutil.copyfile(tp, dst_train / f"{w}__typewell.csv")
        n_ok += 1
    print(f"[bwd] 反転CSV {n_ok}井(skip {n_skip}) {time.time()-t0:.0f}s -> {dst_train}", flush=True)


def gen_forward(dst_train):
    """新規=中点開始: eval前半を既知帯化、後半のみ予測(反転なし・全行保持)。"""
    dst_train.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0; t0 = time.time()
    for w in _wells():
        hp = SRC_TRAIN / f"{w}__horizontal_well.csv"; tp = SRC_TRAIN / f"{w}__typewell.csv"
        if not tp.exists():
            n_skip += 1; continue
        hw = pd.read_csv(hp).reset_index(drop=True)
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            n_skip += 1; continue
        kn = hw["TVT_input"].notna().to_numpy(); ev_idx = np.where(~kn)[0]
        n_ev = len(ev_idx); half = n_ev // 2
        if half < K_MIN or (n_ev - half) < EV_MIN:
            n_skip += 1; continue
        first = ev_idx[:half]                                    # eval前半 → 既知帯化
        tvt = hw["TVT"].to_numpy(float)
        if not np.isfinite(tvt[first]).all():
            n_skip += 1; continue
        hw2 = hw.copy(); ti = hw2["TVT_input"].to_numpy(float).copy()
        ti[first] = tvt[first]                                    # TVT_input := TVT(前半)
        hw2["TVT_input"] = ti                                     # 予測 = eval後半 n_ev−half 行
        hw2.to_csv(dst_train / f"{w}__horizontal_well.csv", index=False)
        shutil.copyfile(tp, dst_train / f"{w}__typewell.csv")
        n_ok += 1
    print(f"[fwd] 中点開始CSV {n_ok}井(skip {n_skip}) {time.time()-t0:.0f}s -> {dst_train}", flush=True)


def main():
    print(f"[DA] SRC={SRC_TRAIN}  MODE={MODE}  NWELLS={NWELLS}", flush=True)
    if MODE in ("bwd", "both"):
        gen_backward(OUT_ROOT / "da_bwd" / "train")
    if MODE in ("fwd", "both"):
        gen_forward(OUT_ROOT / "da_fwd" / "train")


if __name__ == "__main__":
    main()
