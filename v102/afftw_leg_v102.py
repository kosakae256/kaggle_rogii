# -*- coding: utf-8 -*-
"""v102 affine_tw — 1レグ分の生成(Kaggle, 順方向, lean, full平滑, emission ON, 復元可能)。
★create 非経由(build_forward_v102)。affine は tw 表現のみ=6番目候補 affine_tw。
★emission sim: ①同梱 sim(773井)を優先流用(=base v102 と同方式, Kaggle nn_emission=0井 バグを回避)。
              ②同梱 sim が無い/空なら nn_emission で再生成(nn50 同梱・書込可 art)=フォールバックとして機能。
config+錨は read-only dataset を参照。復元 = build_forward の per-(bank,rep) キャッシュ。
env: LEG(v96|v97|v100) / OUT_DIR / ROGII_DS(config+錨+sim+nn50) / ROGII_SCRIPTS / ROGII_DATA / NWELLS
     FORCE_NN_EMISSION=1 で同梱simを無視して必ず nn_emission 再生成(検証用)"""
import os, sys, io, subprocess, shutil, pickle
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

ART_RO = {"v96": DS / "v95", "v97": DS / "v97", "v100": DS / "pf5"}[LEG]   # read-only: config+錨
ARTENV = {"v96": "ROGII_ART95", "v97": "ROGII_ART97", "v100": "ROGII_ART95"}[LEG]
SIM_SHIPPED = DS / "sim_grfree_v97.pkl"        # ★同梱 emission sim(全レグ流用)
NN50_DS = DS / "nn50"                          # nn_emission フォールバック用 encoder
WART = WORK / f"{LEG}_art"                      # 書込可(フォールバック nn_emission の sim 出力先)
SIM_LOCAL = WART / "sim_grfree_v97.pkl"
STORE_CACHE = WORK / f"fwdcache_{LEG}_affine"
OUT = WORK / f"{LEG}_afftw.parquet"
FORCE_NN = os.environ.get("FORCE_NN_EMISSION", "0") == "1"


def run(script, env_extra, tag):
    env = dict(os.environ); env.update(env_extra); env["PYTHONUNBUFFERED"] = "1"
    if NWELLS:
        env["NWELLS"] = NWELLS
    print(f"\n==== {tag}: {script} ====", flush=True)
    r = subprocess.run([PY, str(SCR / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{tag} 失敗 (code={r.returncode})")


def _sim_wells(p):
    try:
        with open(p, "rb") as f:
            return len(pickle.load(f))
    except Exception:
        return -1


def ensure_sim():
    """① 同梱 sim を優先(信頼できる, 773井)。② 無い/空/FORCE時は nn_emission で再生成(フォールバック)。"""
    if not FORCE_NN and SIM_SHIPPED.exists():
        nw = _sim_wells(SIM_SHIPPED)
        if nw > 0:
            print(f"[afftw] 同梱 sim 流用: {SIM_SHIPPED.name}({nw}井)", flush=True)
            return SIM_SHIPPED
        print(f"[afftw] 同梱 sim が空({nw}) -> nn_emission 再生成へ", flush=True)
    else:
        print(f"[afftw] 同梱 sim {'強制無視(FORCE)' if FORCE_NN else '無し'} -> nn_emission 再生成へ", flush=True)
    # フォールバック: nn_emission(書込可 WART に config+錨+nn50 を用意して生成)
    if not WART.exists():
        shutil.copytree(ART_RO, WART)
    if NN50_DS.exists() and not (WART / "nn50").exists():
        shutil.copytree(NN50_DS, WART / "nn50")
    if not SIM_LOCAL.exists() or _sim_wells(SIM_LOCAL) <= 0:
        if SIM_LOCAL.exists():
            SIM_LOCAL.unlink()
        run("nn_emission_v97.py", dict(NN_SPLIT="train", ROGII_ART97=str(WART), ROGII_DATA=DATA), "emission sim(nn_emission fallback)")
    nw = _sim_wells(SIM_LOCAL)
    assert nw > 0, f"★nn_emission も 0井: {SIM_LOCAL}(同梱 sim を使ってください)"
    print(f"[afftw] nn_emission 再生成 sim: {nw}井", flush=True)
    return SIM_LOCAL


def main():
    import pyarrow.parquet as pq
    if OUT.exists():
        try:
            if pq.ParquetFile(OUT).metadata.num_rows > 0:
                print(f"[afftw] {OUT.name} 既存(完成)-> 何もしない", flush=True); return
        except Exception:
            pass
    assert (ART_RO / "grfree_anchor_train.pkl").exists(), f"★錨が見つかりません: {ART_RO}/grfree_anchor_train.pkl"
    SIM = ensure_sim()
    print(f"[afftw] LEG={LEG} ART={ART_RO}(read-only) SIM={SIM} OUT={OUT.name} "
          f"(MODE=affine, FWD_METHODS=tw, FULL, emission ON, PF_NGPU={os.environ.get('PF_NGPU','1')})", flush=True)
    STORE_CACHE.mkdir(parents=True, exist_ok=True)
    run("build_forward_v102.py", {ARTENV: str(ART_RO), "ROGII_DATA": DATA, "ROGII_PROJ": str(SCR),
        "LEG": LEG, "MODE": "affine", "FWD_METHODS": "tw", "FWD_SMOOTH_MODE": "full",
        "FWD_W_NN": "0.02", "OUT_PREFIX": "afftw", "SIM_PKL": str(SIM),
        "OUT_FWD": str(OUT), "STORE_CACHE_DIR": str(STORE_CACHE)}, f"{LEG}レグ build_forward(affine_tw, full, emission ON)")
    print(f"\n[afftw] {LEG} affine_tw 完了 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
