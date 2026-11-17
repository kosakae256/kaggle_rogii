# -*- coding: utf-8 -*-
"""v100 逆方向DA — 1レグ分の生成(Kaggle P100, 8h制限, 復元可能)。
1レグ(v96/v97/v100)を独立生成 → {LEG}_bwd.parquet。復元可能: create の PFキャッシュ
(PF_CACHE_DIR=/kaggle/working/pfcache_{LEG}, per-bank×表現)を working に置く → 出力データセットとして
commit → 次セッションで前回出力を input に attach し pfcache を戻せば、完了済 bank×表現をスキップして再開。
逆方向DA = eval行反転(MD'=MDmax−MD)+ toe側K行を疑似既知帯(gen_da_csvs MODE=bwd)。emission中立(sim非添付)。
錨 = FOLD_ART アンサンブル + sig9.0(train専用)。
env: LEG(v96|v97|v100) / OUT_DIR(/kaggle/working) / ROGII_DS(config+FOLD_ART dataset) / ROGII_SCRIPTS / ROGII_DATA / NWELLS
"""
import os, sys, io, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np

LEG = os.environ["LEG"].lower()
WORK = Path(os.environ["OUT_DIR"])
DS = Path(os.environ["ROGII_DS"])
SCR = Path(os.environ.get("ROGII_SCRIPTS", str(Path(__file__).resolve().parent)))
DATA = os.environ["ROGII_DATA"]
PY = os.environ.get("ROGII_PY", sys.executable)
NWELLS = os.environ.get("NWELLS")

CFG = {"v96": DS / "v95", "v97": DS / "v97", "v100": DS / "pf5"}[LEG]
CREATE = {"v96": "create_v95.py", "v97": "create_v97.py", "v100": "create_v95.py"}[LEG]
ARTENV = {"v96": "ROGII_ART95", "v97": "ROGII_ART97", "v100": "ROGII_ART95"}[LEG]
# ★v100 normal の emission: v96=OFF / v97=ON / v100(pf5)=OFF。DAも normal に一致させる(v97のみ emission ON)。
EMISSION = (LEG == "v97")
DA_ROOT = WORK / "da_bwd"
ANCHOR = WORK / "da_bwd_anchor.pkl"
V97ART = WORK / "v97_art"                                # ★書込可能な v97 ART(config+nn50 コピー, 錨=DA錨に差替)
SIM = V97ART / "sim_grfree_v97.pkl"                      # v97 DA の emission sim(反転井, DA錨=帯中心)
PFCACHE = WORK / f"pfcache_{LEG}"
OUT = WORK / f"{LEG}_bwd.parquet"


def run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    if NWELLS:
        env["NWELLS"] = NWELLS
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(SCR / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def main():
    import pyarrow.parquet as pq
    if OUT.exists():                                          # ★resume: レグ完成なら終了
        try:
            if pq.ParquetFile(OUT).metadata.num_rows > 0:
                print(f"[da_leg] {LEG}_bwd.parquet 既存(完成)-> 何もしない", flush=True); return
        except Exception:
            pass
    print(f"[da_leg] LEG={LEG} create={CREATE} cfg={CFG.name} OUT={OUT.name}  (逆方向DA, emission中立, PFcache={PFCACHE.name})", flush=True)
    # (1) 逆方向DA CSV(冪等・CPU高速)
    if not (DA_ROOT / "train").exists():
        run("gen_da_csvs_v98.py", dict(MODE="bwd", OUT_ROOT=str(WORK), ROGII_DATA=DATA), "逆方向DA CSV")
    else:
        print(f"[da_leg] da_bwd CSV 既存流用", flush=True)
    # (2) DA錨(FOLD_ART アンサンブル → sig9.0)
    if not ANCHOR.exists():
        raw = WORK / "da_anchor_raw.pkl"
        run("gen_grfree_anchor.py", dict(ROGII_DATA=str(DA_ROOT), GRF_SPLIT="train",
            GRFREE_FOLD_ART=str(DS / "grfree_fold_art.pkl"), GRF_ANCHOR_OUT=str(raw),
            ROGII_ART95=str(DS / "v95")), "DA錨(FOLD_ART)")
        anch = pickle.loads(raw.read_bytes())
        fixed = {w: dict(tvt=np.asarray(d["tvt"], np.float32), sig=np.full(len(d["tvt"]), 9.0, np.float32))
                 for w, d in anch.items()}
        ANCHOR.write_bytes(pickle.dumps(fixed, protocol=4))
        print(f"[da_leg] DA錨 {len(fixed)}井 sig=9.0 -> {ANCHOR.name}", flush=True)
    else:
        print(f"[da_leg] DA錨 既存流用", flush=True)
    # (2.5) ★v97レグのみ emission sim を反転井+DA錨(帯中心)で生成(v100 normal v97=emission ON に一致)
    import shutil
    create_art = CFG
    if EMISSION:
        # nn_emission は ART(ROGII_ART97)配下の grfree_anchor_train.pkl を帯中心に使い、sim も ART に書く。
        # → 書込可能な V97ART に config+nn50 をコピーし、grfree_anchor_train.pkl を"DA錨"に差し替える(帯中心=DA錨)。
        if not V97ART.exists():
            shutil.copytree(CFG, V97ART)
        shutil.copyfile(ANCHOR, V97ART / "grfree_anchor_train.pkl")   # ★帯中心=DA錨(反転井)
        create_art = V97ART
        if not SIM.exists():
            run("nn_emission_v97.py", dict(NN_SPLIT="train", ROGII_ART97=str(V97ART), ROGII_DATA=str(DA_ROOT)),
                "emission sim(v97 DA=反転井+DA錨帯中心, normal一致)")
        else:
            print(f"[da_leg] emission sim 既存流用", flush=True)
        print(f"[da_leg] ★v97 emission ON(帯中心=DA錨, normal v97 と一致)", flush=True)
    cenv = {ARTENV: str(create_art), "SPLIT": "train", "ROGII_DATA": str(DA_ROOT),
            "V93_ANCHOR_PKL": str(ANCHOR), "OUT_PARQUET": str(OUT), "PF_CACHE_DIR": str(PFCACHE),
            "ROGII_PROJ": str(SCR)}
    if EMISSION:
        cenv["SIM_PKL"] = str(SIM)
    # (3) レグ create(逆方向DA上, PFキャッシュで復元可能)
    PFCACHE.mkdir(parents=True, exist_ok=True)
    run(CREATE, cenv, f"{LEG}レグ create(逆方向DA{', emission ON' if EMISSION else ''})")
    print(f"\n[da_leg] {LEG} 逆方向DA 完了 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
