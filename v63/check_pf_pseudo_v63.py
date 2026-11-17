# -*- coding: utf-8 -*-
# 軽量チェック: pseudo井戸のPF精度を donor実井戸と同一条件で比較(1バンク・CPU・数分)
#   バンク=r0_seed32(lag0=普通のPF), tw参照のみ(self/nbr/beam無し), 錨OFF。
#   実行: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 .venv/Scripts/python.exe v63/check_pf_pseudo_v63.py
#   env: NPS(比較する井戸数, default5) NSEEDS(4) TAG(p1)
import os, sys, json, time
import numpy as np
import pandas as pd

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v63"))
import create_v63 as m   # 関数(build_smoother_inputs / run_smoother_ext)だけ利用。main()は走らない

TAG = os.environ.get("TAG", "p1")
NPS = int(os.environ.get("NPS", "5"))
NSEEDS = int(os.environ.get("NSEEDS", "4"))
PSD = os.path.join(P0, "v63", "artifacts_v63", f"pseudo_wells_{TAG}")
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
man = json.loads(open(os.path.join(PSD, "manifest.json"), encoding="utf-8").read())["wells"][:NPS]
P = {**m.PF_LIB["r0_seed32"], "smooth_lag": 0}

def load_inputs(wid, base):
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv")).sort_values("TVT")
    x = m.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
    if x is None: return None, None, None
    x = m._attach_anchor(x, "__no_struct__", physics=False)   # 錨OFF
    ev = hw["TVT_input"].isna().to_numpy()
    return x, hw["TVT"].to_numpy(float), ev

jobs = []   # (label, kind, x, truth, ev)
for e in man:
    x, tv, ev = load_inputs(e["id"], PSD)
    if x is not None: jobs.append((e["id"], f"pseudo/{e['mode']}/{e['noise_cat']}", x, tv, ev))
    x, tv, ev = load_inputs(e["donor"], TRAIN)
    if x is not None: jobs.append((e["donor"], "real(donor)", x, tv, ev))

print(f"[check] r0_seed32 lag0 / seeds={NSEEDS} / device=cpu / wells={len(jobs)}(pseudo{NPS}+donor)")
res = []
t0 = time.time()
for j0 in range(0, len(jobs), 2):
    grp = jobs[j0:j0 + 2]
    outs = m.run_smoother_ext([g[2] for g in grp], P, 4423098, NSEEDS, chunk=2)
    for g, o in zip(grp, outs):
        lbl, kind, _, tv, ev = g
        e = o["mean"].astype(float) - tv[ev]     # PF出力は評価行(TVT_input isna)のみに整列済み
        e = e[np.isfinite(e)]
        res.append((lbl, kind, len(e), float(np.sqrt(np.mean(e ** 2)))))
        print(f"  {lbl:18s} {kind:22s} eval={len(e):5d}  RMSE={res[-1][3]:7.2f} ft   ({time.time()-t0:.0f}s)")

ps_e = [r for r in res if r[1].startswith("pseudo")]
re_e = [r for r in res if r[1] == "real(donor)"]
def pooled(rs): return float(np.sqrt(np.sum([r[2] * r[3] ** 2 for r in rs]) / max(np.sum([r[2] for r in rs]), 1)))
print(f"\npooled RMSE: pseudo {pooled(ps_e):.2f} ft ({len(ps_e)}井戸) / real(donor) {pooled(re_e):.2f} ft ({len(re_e)}井戸)")
print("ペア比較(同donor):")
for e in man:
    a = next((r for r in res if r[0] == e["id"]), None); b = next((r for r in res if r[0] == e["donor"]), None)
    if a and b:
        print(f"  {e['id']:18s} {a[3]:7.2f}  vs donor {e['donor']:10s} {b[3]:7.2f}   ({e['mode']}/{e['noise_cat']})")
