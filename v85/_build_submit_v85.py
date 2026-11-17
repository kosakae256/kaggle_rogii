# -*- coding: utf-8 -*-
"""v85/submit_v85.ipynb を生成: r0単体 と r0+affine(50:50) の2提出。"""
import json, io
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
nb_pf = json.load(io.open(P0 / "pf_tuning" / "pf_tuning_pf01.ipynb", encoding="utf-8"))
pf_impl = "".join(nb_pf["cells"][5]["source"])

c_head = (
    "# v85 submit — r0単体 と r0+affine(50:50)blend の2提出\n"
    "# 目的: r0単体LB と affine混合LBの比較。struct/物理錨/dec 一切不使用。\n"
    "# GRの別表現(affine較正)をPFに通して混ぜる。出力: submission_r0.csv / submission_blend.csv\n"
)

c_cfg = r'''# ===== 設定 =====
import os, glob, time, hashlib, json
from pathlib import Path
from dataclasses import dataclass, asdict, field
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

def _find_data():
    cands = ["/kaggle/input/competitions/rogii-wellbore-geology-prediction",
             "/kaggle/input/rogii-wellbore-geology-prediction",
             r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"]
    for p in cands:
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            return Path(p)
    h = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(h[0])) if h else Path(".")

DATA = _find_data()
TEST = DATA / "test"
OUTDIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print("DATA:", DATA, "| TEST wells:", len(glob.glob(str(TEST / "*__horizontal_well.csv"))))

# r0_seed32(struct/anchor不使用の struct-free PF params)
R0 = dict(
    n_particles=400, n_seeds=32, seed0=0, likelihood_scale=10.536319667317654,
    init_pos_std=0.9331541486020234, init_rate_std=0.00047753793163756304,
    mom=0.9997911117425017, vn=0.0006297672017716754, pn=0.005014009140727263,
    rp=0.7627309890952053, rr=0.00045663430474523125, resamp=0.032049849143995945,
    gr_sig_min=11.865793233968713, gr_sig_max=44.98927964612316, gr_sig_def=119.25397464636981,
    gr_sig_mult=3.058318309943017, tvt_clip_margin=150.56111180234504, rate_clip=2.0,
    gr_power=1.7826896851395453, hgr_smooth_r=0, tw_gr_smooth_r=0, jump_prob=0.002, jump_std=5.0)
BLEND_W = 0.5     # blend = (1-W)*r0 + W*affine
SUBMIT  = "blend" # ★ submission.csv にするもの: "r0"=r0単体 / "blend"=0.5*r0+0.5*affine
'''

c_affine = r'''# ===== affine GR再較正(prefix既知区間で GR_hw≈α·GR_tw(TVT)+β をロバスト最小二乗→(GR−β)/α) =====
def recalibrate_gr(hw, tw):
    kn = hw["TVT_input"].notna().to_numpy()
    ttv = tw["TVT"].to_numpy(float); tgr = tw["GR"].to_numpy(float)
    m = np.isfinite(ttv) & np.isfinite(tgr)
    hw2 = hw.copy()
    if m.sum() < 8 or kn.sum() < 100:
        return hw2, (1.0, 0.0)
    o = np.argsort(ttv[m]); tt, tg = ttv[m][o], tgr[m][o]
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").to_numpy()
    x = np.interp(hw["TVT_input"].to_numpy(float)[kn], tt, tg); y = gr[kn]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return hw2, (1.0, 0.0)
    x, y = x[ok], y[ok]; c = np.polyfit(x, y, 1)
    for _ in range(3):
        r = y - np.polyval(c, x); s = np.median(np.abs(r)) * 1.4826 + 1e-6
        c = np.polyfit(x, y, 1, w=1.0 / (1.0 + (r / (2 * s)) ** 2))
    a, b = float(c[0]), float(c[1])
    if not np.isfinite(a) or a < 0.3 or a > 3.0:
        return hw2, (1.0, 0.0)
    hw2["GR"] = (gr - b) / a
    return hw2, (a, b)
'''

c_run = r'''# ===== 実行: test各井で r0(raw) と affine を推論 → 2提出 =====
sub = pd.read_csv(DATA / "sample_submission.csv")
sub["well"] = sub["id"].str.rsplit("_", n=1).str[0]
sub["ridx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
pred_raw, pred_aff = {}, {}
t0 = time.time(); wells = sorted(sub["well"].unique())
for i, wid in enumerate(wells):
    hw = pd.read_csv(TEST / f"{wid}__horizontal_well.csv").reset_index(drop=True)
    tw = pd.read_csv(TEST / f"{wid}__typewell.csv").reset_index(drop=True)
    hw["__row_id"] = np.arange(len(hw))
    p_raw = run_pf_lik_ensemble_param(hw, tw, params=R0, return_full=True)
    hw_a, ab = recalibrate_gr(hw, tw)
    p_aff = run_pf_lik_ensemble_param(hw_a, tw, params=R0, return_full=True)
    pred_raw[wid] = np.asarray(p_raw, float); pred_aff[wid] = np.asarray(p_aff, float)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)

def fill(pred_map):
    vals = np.zeros(len(sub))
    for j, (wid, ridx) in enumerate(zip(sub["well"].to_numpy(), sub["ridx"].to_numpy())):
        pr = pred_map.get(wid)
        vals[j] = pr[ridx] if (pr is not None and 0 <= ridx < len(pr) and np.isfinite(pr[ridx])) else 0.0
    return vals

v_raw = fill(pred_raw); v_aff = fill(pred_aff)
v_blend = (1 - BLEND_W) * v_raw + BLEND_W * v_aff
# 名前付き2種を保存(比較用)
pd.DataFrame({"id": sub["id"], "tvt": v_raw}).to_csv(OUTDIR / "submission_r0.csv", index=False)
pd.DataFrame({"id": sub["id"], "tvt": v_blend}).to_csv(OUTDIR / "submission_blend.csv", index=False)
# ★Kaggle提出用: submission.csv = SUBMIT で選んだ方
v_sel = v_raw if SUBMIT == "r0" else v_blend
pd.DataFrame({"id": sub["id"], "tvt": v_sel}).to_csv(OUTDIR / "submission.csv", index=False)
print(f"[saved] submission.csv = '{SUBMIT}'  + submission_r0.csv / submission_blend.csv")
print(f"        blend W={BLEND_W}, {len(sub)}行, {time.time()-t0:.0f}s")
print("r0 vs affine 平均絶対差:", float(np.mean(np.abs(v_raw - v_aff))), "ft")
'''


def cell(src, md=False):
    d = {"cell_type": "markdown" if md else "code", "metadata": {}, "source": src.splitlines(keepends=True)}
    if not md:
        d["outputs"] = []; d["execution_count"] = None
    return d


nb = {"cells": [cell(c_head, md=True), cell(c_cfg),
                cell("# ===== struct-free PF実装(pf_tuning_pf01)を埋込 =====\n" + pf_impl),
                cell(c_affine), cell(c_run)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, io.open(P0 / "v85" / "submit_v85.ipynb", "w", encoding="utf-8"), ensure_ascii=False)
print("saved v85/submit_v85.ipynb (cells:", len(nb["cells"]), ")")
