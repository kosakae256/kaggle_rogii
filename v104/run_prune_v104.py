# -*- coding: utf-8 -*-
"""v104 剪定ループの1単位: あるPRUNE仮説を 3 group-seed × 5-fold × 5model で学習し、3gseed平均OOFを出す。
スーパーセット seqfeat(train_v104.py 生成)を候補軸スライスで剪定(seqfeat 再構築なし=高速)。
判定 = 3 group-seed 平均 overall RMSE(分割運~0.09 を均す)。baseline(PRUNE空)から伸びれば採用。
env: TAG(必須, 出力接尾) / PRUNE(カンマ区切り部分文字列, 空=baseline) / GSEEDS(既定 0,1,2) / NN_SEEDS(既定5) / SKIP_DONE(既定1)
実行: TAG=base .venv/Scripts/python.exe v104/run_prune_v104.py
      TAG=noSelfGraft PRUNE=self_graft .venv/Scripts/python.exe v104/run_prune_v104.py"""
import os, sys, io, subprocess, glob, re
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
V102 = PROJ / "v102"
ART = V104 / "artifacts_v104"
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
SFOUT = str(ART / "seqfeat_v104_cf.pkl")
V100_COMBO = str(PROJ / "v100" / "artifacts_v100" / "data" / "train_combo.parquet")
TAG = os.environ["TAG"]
PRUNE = os.environ.get("PRUNE", "")
GSEEDS = [int(x) for x in os.environ.get("GSEEDS", "0,1,2").split(",") if x != ""]
NN_SEEDS = os.environ.get("NN_SEEDS", "5")


def _rmse(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def main():
    assert Path(SFOUT).exists(), f"seqfeat 無し(先に train_v104.py): {SFOUT}"
    print(f"[prune] TAG={TAG} PRUNE='{PRUNE or '(なし=baseline)'}' gseeds={GSEEDS} seeds={NN_SEEDS}", flush=True)
    oofs = []
    for gs in GSEEDS:
        osuf = f"_v104{TAG}g{gs}"
        oofp = ART / f"l5{osuf}_oof.npz"
        if not (os.environ.get("SKIP_DONE", "1") == "1" and oofp.exists()):
            env = dict(os.environ)
            env.update(dict(ROGII_ART98=str(ART), V93_INPUT="combo", ARCH="l5", SF_IN=V100_COMBO,
                            L5_SF=SFOUT, L5_OSUF=osuf, PRUNE=PRUNE,
                            GROUP_SEED=str(gs), N_SPLITS="5", V98_DA="0",
                            SEEDS=",".join(str(i) for i in range(int(NN_SEEDS))),
                            ROGII_DATA=DATA, PYTHONUNBUFFERED="1"))
            print(f"\n==== L5 {osuf}(gseed={gs}) ====", flush=True)
            r = subprocess.run([PY, str(V102 / "train_l456_v98.py")], env=env)
            if r.returncode != 0:
                raise SystemExit(f"L5 {osuf} 失敗 ({r.returncode})")
        d = np.load(oofp); y = d["y"]
        l5 = d["l5_mean"] if "l5_mean" in d.files else np.mean([d[k] for k in d.files if k.startswith("l5_s")], 0)
        print(f"  gseed={gs}: {_rmse(l5, y):.4f}", flush=True)
        oofs.append((l5, y))
    y0 = oofs[0][1]
    ens = np.nanmean(np.stack([p for p, _ in oofs], 0), 0)
    r_ens = _rmse(ens, y0)
    per = [_rmse(p, y) for p, y in oofs]
    print(f"\n[prune:{TAG}] 3gseed 各={['%.4f'%v for v in per]}  平均pred RMSE={r_ens:.4f}", flush=True)
    print(f"[prune:{TAG}] ★判定値(3gseed cross ens)= {r_ens:.4f}  (baseと比較)", flush=True)


if __name__ == "__main__":
    main()
