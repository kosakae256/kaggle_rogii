# -*- coding: utf-8 -*-
"""v103 生成: dec10(専用config 1バンク×5表現)+ tw-affine(v100の15バンク×tw)を **emission OFF・full平滑** で生成。
今日の発見: emission は候補多様性を潰し L5 を悪化させる → 脱相関の新バンク(dec10/affine)は emission なしで足す。
出力 = pf-only parquet(id/well + <PRE>_候補列), v100 combo と id整列 → 学習時にメモリ結合(巨大結合ファイルを作らない)。
lean パス(v102/build_forward_v102)を engine に使用(create非経由=無駄なし)。復元 = build_forward の per-(bank,rep)キャッシュ。
実行: .venv/Scripts/python.exe v103/gen_v103.py"""
import os, sys, io, time, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path

V103 = Path(__file__).resolve().parent
PROJ = V103.parent
V102 = PROJ / "v102"                                    # build_forward engine
PY = os.environ.get("ROGII_PY", str(PROJ / ".venv" / "Scripts" / "python.exe"))
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
OUT = V103 / "artifacts_v103" / "fwd"
CACHE = V103 / "artifacts_v103" / "cache"
OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)

V99 = PROJ / "v99" / "artifacts_v99"
V99ANCH = str(V99 / "v95" / "grfree_anchor_train.pkl")   # pf5 用(use_anchor=Falseだがロード用)
ALL5 = "tw,self,nbr,nbr_gr5,self_graft"

# (name, LEG, ARTENV, ART_DIR, MODE, METHODS, PRE, extra_env)
TASKS = [
    # dec10: 専用config(1バンク), 5表現, MODE=affine_dec10(★affine較正+間引き), DEC=10, full, emission OFF
    ("dec10", "v96", "ROGII_ART95", str(V102 / "artifacts_v102" / "dec10"), "affine_dec10", ALL5, "dec", {"DEC": "10"}),
    # tw-affine: v100の15バンク(3レグ)× tw, MODE=affine, full, emission OFF
    ("aff_v96", "v96", "ROGII_ART95", str(V99 / "v95"), "affine", "tw", "aff96", {}),
    ("aff_v97", "v97", "ROGII_ART97", str(V99 / "v97"), "affine", "tw", "aff97", {}),
    ("aff_pf5", "v100", "ROGII_ART95", str(PROJ / "v100" / "artifacts_v100" / "pf5"), "affine", "tw", "affpf5",
     {"V93_ANCHOR_PKL": V99ANCH}),
]


def main():
    t0 = time.time()
    print(f"[gen_v103] dec10(1バンク×5表現)+ tw-affine(15バンク×tw) full平滑・emission OFF。engine=build_forward", flush=True)
    import pyarrow.parquet as pq
    for (name, leg, artenv, artdir, mode, methods, pre, extra) in TASKS:
        outp = OUT / f"{name}.parquet"
        if outp.exists():
            try:
                if pq.ParquetFile(outp).metadata.num_rows > 0:
                    print(f"[gen_v103] {name} 完成済 -> skip", flush=True); continue
            except Exception:
                pass
        env = dict(os.environ)
        env.update({
            artenv: artdir, "ROGII_DATA": DATA, "ROGII_PROJ": str(V102),
            "LEG": leg, "MODE": mode, "FWD_METHODS": methods, "FWD_SMOOTH_MODE": "full",
            "OUT_PREFIX": pre, "OUT_FWD": str(outp),
            "STORE_CACHE_DIR": str(CACHE / f"{name}"), "PYTHONUNBUFFERED": "1",
            "FULL_VRAM_GB": os.environ.get("FULL_VRAM_GB", "8.0"), "PF_NGPU": os.environ.get("PF_NGPU", "1"),
        })
        env.pop("SIM_PKL", None)                          # ★emission OFF(sim渡さない)
        env.update(extra)
        print(f"\n==== 生成: {name}(LEG={leg} MODE={mode} methods={methods} PRE={pre} full, emission OFF)====", flush=True)
        r = subprocess.run([PY, str(V102 / "build_forward_v102.py")], env=env)
        if r.returncode != 0:
            raise SystemExit(f"{name} 失敗 (code={r.returncode})")
        m = pq.ParquetFile(outp).metadata
        print(f"[gen_v103] {name} 完成: {m.num_rows}行 {m.num_columns}列 -> {outp.name}", flush=True)
    print(f"\n[gen_v103] 全生成完了 ({time.time()-t0:.0f}s)。出力: {OUT}", flush=True)


if __name__ == "__main__":
    main()
