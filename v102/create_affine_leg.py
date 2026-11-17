# -*- coding: utf-8 -*-
"""v102 affine — 1レグ分の affine版特徴生成(Kaggle, 順方向, 復元可能)。
config+nn50+錨(OOF)を書込可能領域にコピー → emission sim 生成(nn_emission, 帯中心=OOF錨) →
create_v95/v97 を AFFINE=1 で実行(全バンク×5表現に affine較正 gr=(gr−b)/a)→ {LEG}_affine.parquet。
emission+ps_combo+full平滑 は v102 と同一(config駆動)。復元=create の PFキャッシュ。
env: LEG(v96|v97|v100) / OUT_DIR / ROGII_DS(config+錨+nn50 dataset) / ROGII_SCRIPTS / ROGII_DATA / NWELLS"""
import os, sys, io, subprocess, shutil
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path

LEG = os.environ["LEG"].lower()
WORK = Path(os.environ["OUT_DIR"])
DS = Path(os.environ["ROGII_DS"])
SCR = Path(os.environ.get("ROGII_SCRIPTS", str(Path(__file__).resolve().parent)))
DATA = os.environ["ROGII_DATA"]
PY = os.environ.get("ROGII_PY", sys.executable)
NWELLS = os.environ.get("NWELLS")

CFGSRC = {"v96": DS / "v95", "v97": DS / "v97", "v100": DS / "pf5"}[LEG]
CREATE = {"v96": "create_v95.py", "v97": "create_v97.py", "v100": "create_v95.py"}[LEG]
ARTENV = {"v96": "ROGII_ART95", "v97": "ROGII_ART97", "v100": "ROGII_ART95"}[LEG]
ART = WORK / f"{LEG}_art"                                # 書込可能(sim出力先)
SIM = ART / "sim_grfree_v97.pkl"
PFCACHE = WORK / f"pfcache_{LEG}"
OUT = WORK / f"{LEG}_affine.parquet"


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
    if OUT.exists():
        try:
            if pq.ParquetFile(OUT).metadata.num_rows > 0:
                print(f"[affine] {OUT.name} 既存(完成)-> 何もしない", flush=True); return
        except Exception:
            pass
    if not ART.exists():
        shutil.copytree(CFGSRC, ART)                    # config+nn50+錨 を書込可能へ(sim出力先)
    # ★affine は smooth16(fixedlag)で高速化=採点時間対策。full の単体精度優位は脱相関脚には不要。
    import json as _json
    SM = os.environ.get("AFF_SMOOTH_MODE", "fixedlag"); SL = int(os.environ.get("AFF_SMOOTH_LAG", "16"))
    _cfp = ART / "pf_banks_config.json"; _c = _json.loads(_cfp.read_text(encoding="utf-8"))
    _c["smooth_mode"] = SM; _c["smooth_lag"] = SL
    _cfp.write_text(_json.dumps(_c, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[affine] LEG={LEG} create={CREATE} ART={ART.name} OUT={OUT.name} (AFFINE=1, emission+ps_combo, smooth={SM}/lag{SL})", flush=True)
    # emission sim(帯中心=OOF錨 grfree_anchor_train.pkl, NN_SPLIT=train)
    if not SIM.exists():
        run("nn_emission_v97.py", dict(NN_SPLIT="train", ROGII_ART97=str(ART), ROGII_DATA=DATA), "emission sim")
    # create(AFFINE=1)
    PFCACHE.mkdir(parents=True, exist_ok=True)
    run(CREATE, {ARTENV: str(ART), "SPLIT": "train", "ROGII_DATA": DATA, "SIM_PKL": str(SIM),
                 "OUT_PARQUET": str(OUT), "PF_CACHE_DIR": str(PFCACHE), "ROGII_PROJ": str(SCR),
                 "AFFINE": "1", "PF_ONLY_OUT": "1"}, f"{LEG}レグ create(AFFINE, pf_列のみ)")
    print(f"\n[affine] {LEG} affine 完了 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
