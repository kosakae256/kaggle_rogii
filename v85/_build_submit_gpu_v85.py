# -*- coding: utf-8 -*-
"""v85/submit_gpu_v85.ipynb を生成: GPU(v65 torchエンジン)で r0単体 と r0+affine(50:50) の2提出。
struct/物理錨/dec 不使用(use_anchor=False, struct pkl無しでも自動フォールバック)。"""
import json, io, re
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
eng = io.open(P0 / "v86" / "pf_engine_v65.py", encoding="utf-8").read()

# --- エンジンをKaggle安全にパッチ ---
# 1) optuna import を任意化
eng = eng.replace("import optuna\n", "try:\n    import optuna\nexcept Exception:\n    optuna = None\n")
eng = eng.replace("optuna.logging.set_verbosity(optuna.logging.WARNING)",
                  "\nif optuna is not None:\n    optuna.logging.set_verbosity(optuna.logging.WARNING)")
# 2) パス設定ブロックをKaggle/ローカル自動検出に置換(OUT_DIRは書込可能な作業dir)
old_cfg = '''PROJECT   = Path(r"C:\\Users\\kosaka256\\Documents\\rogii_claude")
DATA_DIR  = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_DIR   = PROJECT / "v65" / "artifacts_v65"; OUT_DIR.mkdir(parents=True, exist_ok=True)
V38_ART   = PROJECT / "v38" / "artifacts_v38"'''
new_cfg = '''def _find_root():
    for p in ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
              "/kaggle/input/rogii-wellbore-geology-prediction",
              r"C:\\Users\\kosaka256\\Documents\\rogii_claude\\rogii-wellbore-geology-prediction"]:
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            return Path(p)
    import glob as _g
    h = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(h[0])) if h else Path(".")
_DATA_ROOT = _find_root()
PROJECT   = Path(r"C:\\Users\\kosaka256\\Documents\\rogii_claude")
DATA_DIR  = _DATA_ROOT / "train"                      # ★submitでは実行時に test へ差し替える
OUT_DIR   = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
V38_ART   = PROJECT / "v38" / "artifacts_v38"'''
assert old_cfg in eng, "CONFIGブロック不一致"
eng = eng.replace(old_cfg, new_cfg)
# 3) submit時: TVT列(=真値)無しでもbuild_cacheを通す(testはTVT無し。tvt_trueはNoneのまま)
old_tvt = '''    if "TVT" not in hw.columns:
        return None'''
new_tvt = '''    if "TVT" not in hw.columns and os.environ.get("V86_SUBMIT", "0") != "1":
        return None'''
assert old_tvt in eng, "TVTガード不一致"
eng = eng.replace(old_tvt, new_tvt)

c_head = (
    "# v85 submit(GPU) — r0単体 と r0+affine(50:50)blend の2提出\n"
    "# GPU: v65 torchエンジン(pf_engine_v65)を埋込。struct/物理錨/dec 不使用(use_anchor=False, struct無しでも動く)。\n"
    "# 出力: submission.csv(SUBMITで選択) + submission_r0.csv / submission_blend.csv\n"
)

c_cfg = r'''# ===== 設定 =====
import os
from pathlib import Path
import numpy as np, pandas as pd
R0 = dict(
    n_particles=400, likelihood_scale=10.536319667317654, init_pos_std=0.9331541486020234,
    init_rate_std=0.00047753793163756304, mom=0.9997911117425017, vn=0.0006297672017716754,
    pn=0.005014009140727263, rp=0.7627309890952053, rr=0.00045663430474523125,
    resamp=0.032049849143995945, gr_sig_min=11.865793233968713, gr_sig_max=44.98927964612316,
    gr_sig_def=119.25397464636981, gr_sig_mult=3.058318309943017, tvt_clip_margin=150.56111180234504,
    rate_clip=2.0, gr_power=1.7826896851395453, hgr_smooth_r=0, tw_gr_smooth_r=0,
    jump_prob=0.002, jump_std=5.0)
N_SEEDS = 32
BLEND_W = 0.5      # blend = (1-W)*r0 + W*affine
SUBMIT  = "blend"  # submission.csv にする方: "r0" / "blend"
CHUNK   = 120
os.environ["V65_TWSRC"] = "tw"     # self/nbr不使用
os.environ["V86_SUBMIT"] = "1"     # ★test(TVT列無し)でもbuild_cacheを通す
'''

c_run = r'''# ===== 実行(GPU): test各井で r0(raw) と affine を推論 → 2提出 =====
import time
DATA = _DATA_ROOT; TEST = DATA / "test"
OUT = OUT_DIR
print("DEVICE:", DEVICE, "| DATA:", DATA, "| TEST wells:", len(list(TEST.glob("*__horizontal_well.csv"))))
globals_engine = globals()   # 埋込エンジンの関数はグローバルに展開済み

sub = pd.read_csv(DATA / "sample_submission.csv")
sub["well"] = sub["id"].str.rsplit("_", n=1).str[0]
sub["ridx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
wells = sorted(sub["well"].unique())
P = full_params(dict(R0)); P["use_anchor"] = False   # ★錨off
P["n_seeds"] = N_SEEDS

def run_rep(rep):
    os.environ["V86_REP"] = rep
    # test井のcacheを構築(埋込エンジンと同一名前空間なのでDATA_DIR/CACHEを直接差替)
    global DATA_DIR, CACHE
    DATA_DIR = TEST
    CACHE = {}
    pos = {}
    keep = []
    for w in wells:
        c = build_cache(w)
        if c is None: continue
        CACHE[w] = c; pos[w] = c["ev_pos"]; keep.append(w)
    out = {}
    t0 = time.time()
    for c0 in range(0, len(keep), CHUNK):
        chunk = keep[c0:c0 + CHUNK]
        inps = [make_inputs(CACHE[w], P) for w in chunk]
        preds = run_forward_ens(inps, P, DEVICE, 20250707, N_SEEDS)
        for k, w in enumerate(chunk):
            out[w] = (np.asarray(preds[k], float), pos[w])
        print(f"  {rep} {min(c0+CHUNK,len(keep))}/{len(keep)} ({time.time()-t0:.0f}s)", flush=True)
    return out

res_raw = run_rep("raw")
res_aff = run_rep("affine")

def fill(res):
    m = {}
    for w, (pr, ep) in res.items():
        for i, ridx in enumerate(ep):
            if i < len(pr) and np.isfinite(pr[i]): m[(w, int(ridx))] = float(pr[i])
    return np.array([m.get((w, int(r)), 0.0) for w, r in zip(sub["well"], sub["ridx"])])

v_raw = fill(res_raw); v_aff = fill(res_aff)
v_blend = (1 - BLEND_W) * v_raw + BLEND_W * v_aff
pd.DataFrame({"id": sub["id"], "tvt": v_raw}).to_csv(OUT / "submission_r0.csv", index=False)
pd.DataFrame({"id": sub["id"], "tvt": v_blend}).to_csv(OUT / "submission_blend.csv", index=False)
v_sel = v_raw if SUBMIT == "r0" else v_blend
pd.DataFrame({"id": sub["id"], "tvt": v_sel}).to_csv(OUT / "submission.csv", index=False)
print(f"[saved] submission.csv='{SUBMIT}' + submission_r0.csv/submission_blend.csv  ({len(sub)}行)")
print("r0 vs affine 平均絶対差:", float(np.mean(np.abs(v_raw - v_aff))), "ft")
'''


def cell(src, md=False):
    d = {"cell_type": "markdown" if md else "code", "metadata": {}, "source": src.splitlines(keepends=True)}
    if not md:
        d["outputs"] = []; d["execution_count"] = None
    return d


nb = {"cells": [cell(c_head, md=True), cell(c_cfg),
                cell("# ===== v65 GPU PFエンジン(pf_engine_v65)を埋込(Kaggle安全パッチ済) =====\n" + eng),
                cell(c_run)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, io.open(P0 / "v85" / "submit_gpu_v85.ipynb", "w", encoding="utf-8"), ensure_ascii=False)
print("saved v85/submit_gpu_v85.ipynb (cells:", len(nb["cells"]), ")")
