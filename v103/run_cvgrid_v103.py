# -*- coding: utf-8 -*-
"""v103 CV grid: model 5seed × [group seed{0,1,2} × fold{5,8,10}] の L5 OOF を集計。
seeded GroupKFold(坑井シャッフル+行数バランス, GROUP_SEED)で分割を変え、fold数も振って CV の頑健性/最良分割を見る。
seqfeat は fold 非依存 → v103 all(v100+dec10+tw-affine)の既存 seqfeat を全構成で再利用。
env: GSEEDS(既定 0,1,2) / KS(既定 5,8,10) / NN_SEEDS(model seed, 既定5)
実行: .venv/Scripts/python.exe v103/run_cvgrid_v103.py"""
import os, sys, io, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

V103 = Path(__file__).resolve().parent
PROJ = V103.parent
V102 = PROJ / "v102"
ART = V103 / "artifacts_v103"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
V100_COMBO = str(PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet")
SF = ART / "seqfeat_v103all_cf.pkl"                      # v103 all の seqfeat(再利用)
NN_SEEDS = int(os.environ.get("NN_SEEDS", "5"))
GSEEDS = [int(x) for x in os.environ.get("GSEEDS", "0,1,2").split(",")]
KS = [int(x) for x in os.environ.get("KS", "5,8,10").split(",")]


def _run_l5(gseed, K, osuf):
    env = dict(os.environ)
    env.update({
        "ROGII_ART98": str(ART), "V93_INPUT": "combo", "ARCH": "l5", "SF_IN": V100_COMBO,
        "SEEDS": ",".join(str(i) for i in range(NN_SEEDS)), "V98_DA": "0",
        "L5_SF": str(SF), "L5_OSUF": osuf, "ROGII_DATA": DATA,
        "GROUP_SEED": str(gseed), "N_SPLITS": str(K), "PYTHONUNBUFFERED": "1",
    })
    print(f"\n==== L5 config: gseed={gseed} K={K} osuf={osuf} ({NN_SEEDS}model-seed)====", flush=True)
    r = subprocess.run([PY, str(V102 / "train_l456_v98.py")], env=env)
    if r.returncode != 0:
        raise SystemExit(f"L5(gseed={gseed},K={K}) 失敗 (code={r.returncode})")


def main():
    t0 = time.time()
    assert SF.exists(), f"★seqfeat_v103all_cf.pkl が無い: {SF}(train_v103 SET=all を先に)"
    print(f"[cvgrid] seqfeat={SF.name} | model {NN_SEEDS}seed × gseed{GSEEDS} × K{KS} = {len(GSEEDS)*len(KS)}構成", flush=True)
    results = {}
    for K in KS:
        for gs in GSEEDS:
            osuf = f"_v103g{gs}k{K}"
            oofp = ART / f"l5{osuf}_oof.npz"
            if not oofp.exists():
                _run_l5(gs, K, osuf)
            d = np.load(oofp); y = d["y"]
            l5m = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
            m = np.isfinite(l5m) & np.isfinite(y)
            rmse = float(np.sqrt(np.mean((l5m[m] - y[m]) ** 2)))
            results[(gs, K)] = rmse
            print(f"[cvgrid] gseed={gs} K={K}: 5seed-ensemble OOF RMSE = {rmse:.4f}", flush=True)
    # 集計
    print("\n===== v103 CV grid 結果(5-model-seed ensemble OOF RMSE)=====", flush=True)
    print("        " + "  ".join(f"K={K}" for K in KS), flush=True)
    for gs in GSEEDS:
        print(f"gseed={gs}: " + "  ".join(f"{results[(gs,K)]:.4f}" for K in KS), flush=True)
    allv = np.array(list(results.values()))
    print(f"\n全構成: 平均={allv.mean():.4f} 最良={allv.min():.4f} 最悪={allv.max():.4f} std={allv.std():.4f}", flush=True)
    print(f"(参考 v100=5.6684 / v103 all(det GroupKFold)=5.5549)", flush=True)
    print(f"\n[cvgrid] 完了 ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
