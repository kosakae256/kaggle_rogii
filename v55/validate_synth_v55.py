# v55 T0/T1 ゲート: 同一の純PFエンジン(r0, W_NN=0, dummy sim)を
#   実サンプル NREAL 井 vs 合成 synth_{TAG} に走らせ、難易度(井戸別RMSE)の分布を突合せる。
#   出力: pooled/median/tail率 + ECDF比較PNG。判定目安: T0=合成pooled 7〜11 / T1=tail率が実の×0.5〜2。
# 使い方: TAG=v1 NREAL=100 SEEDS=8 python v55/validate_synth_v55.py
import os, sys, json, time
import numpy as np
import pandas as pd

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(P0, "v51"))
sys.path.insert(0, os.path.join(P0, "v54"))
import nnpf_v51 as npf
from create_candidates_v54 import run_paths

TAG = os.environ.get("TAG", "v1")
SYN = os.path.join(P0, "v55", "artifacts_v55", f"synth_{TAG}")
NREAL = int(os.environ.get("NREAL", "100"))
SEEDS = int(os.environ.get("SEEDS", "8")); CHUNK = int(os.environ.get("CHUNK", "12"))
BANK = os.environ.get("BANK", "r0_seed32")
TRAIN = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")

def dummy_sim(T):
    return dict(st=np.zeros(T, np.float32),
                sim=np.full((T, 2 * npf.NBAND + 1), -1.0, np.float16))

def run_group(files_dir, wells, P):
    inps, keep, ys = [], [], []
    for wn in wells:
        hw = pd.read_csv(os.path.join(files_dir, f"{wn}__horizontal_well.csv"))
        tw = pd.read_csv(os.path.join(files_dir, f"{wn}__typewell.csv")).sort_values("TVT")
        if "TVT" not in hw.columns or hw["TVT_input"].isna().sum() == 0: continue
        x = npf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
        if x is None: continue
        ev = hw[hw["TVT_input"].isna()]
        inps.append(x); keep.append(wn); ys.append(ev["TVT"].to_numpy(float))
    sims = [dummy_sim(len(x["md"])) for x in inps]
    sgs = [None] * len(inps)
    prs, wws = run_paths(inps, sims, sgs, P, 777, SEEDS, CHUNK, 0.0, use_anchor=False)
    per_well = {}; es = []
    for wn, pr, ww, y in zip(keep, prs, wws, ys):
        wm = (ww[:, None] * pr.astype(np.float64)).sum(0)
        e = (wm - y)[np.isfinite(y)]
        per_well[wn] = float(np.sqrt(np.mean(e ** 2))); es.append(e)
    ee = np.concatenate(es)
    return float(np.sqrt(np.mean(ee ** 2))), per_well

def main():
    t0 = time.time()
    P = {k: v for k, v in npf.PF_LIB[BANK].items() if k != "name"}
    rng = np.random.default_rng(1)
    real_all = sorted(set(f.split("__")[0] for f in os.listdir(TRAIN) if f.endswith("__horizontal_well.csv")))
    real = [real_all[i] for i in rng.permutation(len(real_all))[:NREAL]]
    syn = sorted(set(f.split("__")[0] for f in os.listdir(SYN) if f.endswith("__horizontal_well.csv")))
    print(f"bank={BANK} W_NN=0 seeds={SEEDS}  real={len(real)} synth={len(syn)}", flush=True)
    rr, pw_r = run_group(TRAIN, real, P)
    print(f"[real ] pooled={rr:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    rs, pw_s = run_group(SYN, syn, P)
    print(f"[synth] pooled={rs:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    ar = np.array(list(pw_r.values())); as_ = np.array(list(pw_s.values()))
    tail_r = float((ar > 10).mean()); tail_s = float((as_ > 10).mean())
    print(f"井戸別: real med={np.median(ar):.2f} P90={np.percentile(ar,90):.2f} tail>10ft {tail_r:.1%}")
    print(f"        synth med={np.median(as_):.2f} P90={np.percentile(as_,90):.2f} tail>10ft {tail_s:.1%}")
    t0ok = 7.0 <= rs <= 11.0
    t1ok = (0.5 * max(tail_r, 1e-9) <= tail_s <= 2.0 * tail_r) if tail_r > 0 else (tail_s < 0.2)
    print(f"T0(合成pooled 7-11): {'PASS' if t0ok else 'FAIL'}   T1(tail率 ×0.5-2): {'PASS' if t1ok else 'FAIL'}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for arr, lab in ((ar, f"実 (pooled {rr:.2f})"), (as_, f"合成 (pooled {rs:.2f})")):
        xs = np.sort(arr); ax[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab)
    ax[0].set_xlabel("井戸RMSE (ft)"); ax[0].set_ylabel("ECDF"); ax[0].legend(); ax[0].set_xlim(0, 25)
    ax[0].set_title(f"難易度分布 実vs合成 ({BANK}, 純PF, {SEEDS}seed)")
    ax[1].hist(ar, bins=30, range=(0, 25), alpha=0.5, density=True, label="実")
    ax[1].hist(as_, bins=30, range=(0, 25), alpha=0.5, density=True, label="合成")
    ax[1].set_xlabel("井戸RMSE (ft)"); ax[1].legend(); ax[1].set_title("ヒストグラム")
    plt.tight_layout()
    fp = os.path.join(SYN, "validate_ecdf.png")
    plt.savefig(fp, dpi=110)
    json.dump(dict(bank=BANK, seeds=SEEDS, real_pooled=rr, synth_pooled=rs,
                   real_tail=tail_r, synth_tail=tail_s, t0=bool(t0ok), t1=bool(t1ok),
                   per_well_synth=pw_s), open(os.path.join(SYN, "validate_result.json"), "w"), indent=1)
    print(f"saved {fp}")

if __name__ == "__main__":
    main()
