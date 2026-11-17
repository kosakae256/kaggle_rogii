# -*- coding: utf-8 -*-
"""
v66 PFレベル A/B — struct prior 差し替えの効果を、同一パラメータ・同一seedの PF で測る
=====================================================================================
子プロセスで V38_PRIOR を切り替えて v65 エンジンを2回走らせ(pkl は import 時ロードのため)、
per-well RMSE / overall / hardcore を比較する。パラメータは struct20 best(LB 7.881 実証 regime)固定。

実行: .venv/Scripts/python.exe v66/eval_struct_pf_v66.py
env : PRIORS("struct,v66")   比較する prior 名(struct=v38, その他= v38/artifacts_v38/priors/prior_{名}_train.pkl)
      SEED(20251707=別seed既定。採用判断は別seedで=v38規律) SEEDS(32) N_WELLS(0=全) CHUNK(120)
      PARAMS_JSON("")         省略時 warmup_v65.ST20_PARAMS。別パラメータで測るなら params dict の json パス
      FORCE(0)                1=既存結果 json を無視して再実行
出力: v66/artifacts_v66/pf_ab_{prior}_s{seed}.json / pf_ab_compare_s{seed}.png
"""
import sys, os, json, subprocess
from pathlib import Path
import numpy as np

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v66" / "artifacts_v66"; OUT.mkdir(parents=True, exist_ok=True)

PRIORS = os.environ.get("PRIORS", "struct,v66").split(",")
SEED = int(os.environ.get("SEED", "20251707"))
SEEDS = int(os.environ.get("SEEDS", "32"))
N_WELLS = int(os.environ.get("N_WELLS", "0"))
CHUNK = int(os.environ.get("CHUNK", "120"))
PARAMS_JSON = os.environ.get("PARAMS_JSON", "")


def load_hardcore():
    try:
        import pandas as pd
        d = pd.read_csv(PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv")
        return set(d[d["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        return set()


def child(prior):
    """1つの prior で struct20 params の PF を全坑井実行 → per-well RMSE を保存。"""
    sys.path.insert(0, str(PROJECT / "v65"))
    import pf_engine_v65 as pf
    import warmup_v65 as wu
    if PARAMS_JSON:
        P0 = json.loads(Path(PARAMS_JSON).read_text())
        P0 = P0.get("params", P0.get("best_params", P0))
    else:
        P0 = dict(wu.snap_to_space(wu.ST20_PARAMS, wu.parse_engine_space()))
    P0.pop("self_mix_w", None); P0.pop("nbr_mix_w", None)      # A/B は tw 固定(prior 以外を凍結)
    P = pf.full_params(P0)
    wells = pf.list_wells()
    if N_WELLS:
        wells = wells[:N_WELLS]
    cache = {}
    n_anch = 0
    for w in wells:
        c = pf.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            cache[w] = c
            if np.isfinite(c["ancs"]).all() and (c["ancs"] < 1e8).any():
                n_anch += 1
    pf.CACHE = cache
    ALL = [w for w in wells if w in cache]
    print(f"[{prior}] usable {len(ALL)}  構造錨あり {n_anch}  seed={SEED} x{SEEDS}")
    errs = pf.predict_errs(P, ALL, SEED, SEEDS, CHUNK)
    pw = {}
    pool = []
    for w in ALL:
        e = errs[w]; e = e[np.isfinite(e)]
        if e.size:
            pw[w] = float(np.sqrt(np.mean(e ** 2))); pool.append(e)
    pool = np.concatenate(pool)
    hard = load_hardcore() & set(pw)
    hpool = np.concatenate([errs[w][np.isfinite(errs[w])] for w in hard]) if hard else np.array([0.0])
    res = dict(prior=prior, seed=SEED, seeds=SEEDS, n_wells=len(pw), n_anchored=n_anch,
               overall=float(np.sqrt(np.mean(pool ** 2))),
               hardcore=float(np.sqrt(np.mean(hpool ** 2))) if len(hard) else None,
               n_hard=len(hard), per_well=pw)
    fp = OUT / f"pf_ab_{prior}_s{SEED}.json"
    fp.write_text(json.dumps(res, indent=1))
    print(f"[{prior}] overall={res['overall']:.4f}  hardcore({res['n_hard']})="
          f"{res['hardcore'] if res['hardcore'] else float('nan'):.4f}  -> {fp.name}")


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(sys.argv[2]); return
    results = {}
    for pr in [p.strip() for p in PRIORS if p.strip()]:
        fp = OUT / f"pf_ab_{pr}_s{SEED}.json"
        if fp.exists() and os.environ.get("FORCE", "0") != "1":
            print(f"[skip] {fp.name} 既存(FORCE=1 で再実行)")
        else:
            env = dict(os.environ); env["V38_PRIOR"] = pr; env.setdefault("TW_SOURCE", "tw")
            r = subprocess.run([sys.executable, __file__, "--child", pr], env=env)
            if r.returncode != 0:
                print(f"[error] {pr} 側の実行失敗(rc={r.returncode})"); return
        results[pr] = json.loads(fp.read_text())

    if len(results) < 2:
        return
    ks = list(results)
    a, b = results[ks[0]], results[ks[1]]
    common = sorted(set(a["per_well"]) & set(b["per_well"]))
    pa = np.array([a["per_well"][w] for w in common]); pb = np.array([b["per_well"][w] for w in common])
    print(f"\n=== PF A/B({ks[0]} → {ks[1]}, struct20 params, seed{SEED}x{SEEDS}, {len(common)}坑井) ===")
    print(f"  overall : {a['overall']:.4f} -> {b['overall']:.4f}  (Δ {b['overall'] - a['overall']:+.4f})")
    if a.get("hardcore") and b.get("hardcore"):
        print(f"  hardcore: {a['hardcore']:.4f} -> {b['hardcore']:.4f}  (Δ {b['hardcore'] - a['hardcore']:+.4f})")
    print(f"  坑井別中央値: {np.median(pa):.4f} -> {np.median(pb):.4f}   "
          f"{ks[1]} 改善 {int((pb < pa).sum())}/{len(common)}")
    dlt = sorted(zip(pb - pa, common))
    print("  改善 top5:", [(w, round(d, 2)) for d, w in dlt[:5]])
    print("  悪化 top5:", [(w, round(d, 2)) for d, w in dlt[-5:]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].scatter(pa, pb, s=8, alpha=0.5)
        lim = [min(pa.min(), pb.min()) * 0.8, max(pa.max(), pb.max()) * 1.2]
        ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlabel(f"{ks[0]} per-well RMSE"); ax[0].set_ylabel(f"{ks[1]} per-well RMSE")
        ax[0].set_title(f"PF A/B 坑井別(overall {a['overall']:.3f}→{b['overall']:.3f})")
        for arr, lb in [(pa, ks[0]), (pb, ks[1])]:
            xs = np.sort(arr); ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lb)
        ax[1].set_xscale("log"); ax[1].set_xlabel("per-well RMSE [ft]"); ax[1].set_ylabel("ECDF")
        ax[1].legend(); ax[1].set_title("坑井別 RMSE の ECDF")
        fig.tight_layout()
        png = OUT / f"pf_ab_compare_s{SEED}.png"
        fig.savefig(png, dpi=120); plt.close(fig)
        print(f"[png] {png}")
    except Exception as e:
        print(f"[png] スキップ({e})")


if __name__ == "__main__":
    main()
