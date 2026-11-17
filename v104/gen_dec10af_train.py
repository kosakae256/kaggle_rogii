# -*- coding: utf-8 -*-
"""anchor-free dec10(sharp/smooth)を train で【6表現】生成:他PFと同様に
  raw dec10 = tw/self/nbr/nbr_gr5/self_graft(5表現)+ affine_tw = affine_dec10(tw)= 6表現目。
2バンク(sharp=1, smooth=2)× 6表現 = 12候補。emission OFF・full平滑・低GPU(FULL_VRAM_GB=2)で TCN と並列。
出力: dec10af_train_lean.parquet(id/well + decaf_{5rep}_delta/std_{1,2} + affdec_ancc_delta/std_{1,2})。
実行: .venv/Scripts/python.exe v103/gen_dec10af_train.py"""
import os, sys, subprocess
from pathlib import Path

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
V102 = PROJ / "v102"
PY = str(PROJ / ".venv" / "Scripts" / "python.exe")
DATA = str(PROJ / "rogii-wellbore-geology-prediction")
ART = V104 / "artifacts_v104" / "dec10af"
FWD = V104 / "artifacts_v104" / "fwd"
VRAM = os.environ.get("FULL_VRAM_GB", "2.0")            # ★TCN と並列=低GPU

# (tag, MODE, FWD_METHODS, OUT_PREFIX, cache, out)
PASSES = [
    ("raw", "dec10",        "tw,self,nbr,nbr_gr5,self_graft", "decaf",  "cache_dec10af_train", "dec10af_raw.parquet"),
    ("aff", "affine_dec10", "tw",                             "affdec", "cache_dec10af_aff",   "dec10af_aff.parquet"),
]


def run_pass(tag, mode, methods, pre, cache, outfn):
    outp = FWD / outfn
    if outp.exists():
        print(f"[gen_dec10af:{tag}] 既存 {outfn} 流用", flush=True); return outp
    env = dict(os.environ)
    env.update(dict(
        ROGII_ART95=str(ART), ROGII_DATA=DATA, ROGII_PROJ=str(V102),
        LEG="v96", MODE=mode, DEC="10", FWD_METHODS=methods,
        FWD_SMOOTH_MODE="full", FWD_SPLIT="train",
        OUT_PREFIX=pre, OUT_FWD=str(outp),
        STORE_CACHE_DIR=str(FWD / cache),
        PF_NGPU="1", FULL_VRAM_GB=VRAM, PYTHONUNBUFFERED="1",
    ))
    print(f"[gen_dec10af:{tag}] MODE={mode} methods={methods} PRE={pre} -> {outfn}(FULL_VRAM_GB={VRAM})", flush=True)
    r = subprocess.run([PY, str(V102 / "build_forward_v102.py")], env=env)
    if r.returncode != 0:
        raise SystemExit(f"pass {tag} 失敗 ({r.returncode})")
    return outp


def slim():
    import pyarrow.parquet as pq
    REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")
    raw = FWD / "dec10af_raw.parquet"; aff = FWD / "dec10af_aff.parquet"
    keep_raw = ["id", "well"] + [f"decaf_{r}_delta_{b}" for r in REPS for b in (1, 2)] + [f"decaf_{r}_std_{b}" for r in REPS for b in (1, 2)]
    keep_aff = [f"affdec_ancc_delta_{b}" for b in (1, 2)] + [f"affdec_ancc_std_{b}" for b in (1, 2)]
    tr = pq.read_table(raw, columns=keep_raw)
    ta = pq.read_table(aff, columns=["id"] + keep_aff)
    assert tr.column("id").to_pylist() == ta.column("id").to_pylist(), "id不一致"
    for c in keep_aff:
        tr = tr.append_column(c, ta.column(c))
    out = FWD / "dec10af_train_lean.parquet"
    pq.write_table(tr, out, compression="zstd")
    import os as _os
    print(f"[gen_dec10af] lean -> {out.name}  {tr.num_rows}行 {len(tr.schema.names)}列 ({round(_os.path.getsize(out)/1e6,1)}MB)", flush=True)
    print(f"  候補: raw 5表現×2 + affine_tw×2 = 12 (delta) / +12 std", flush=True)


def main():
    for (tag, mode, methods, pre, cache, outfn) in PASSES:
        run_pass(tag, mode, methods, pre, cache, outfn)
    slim()


if __name__ == "__main__":
    main()
