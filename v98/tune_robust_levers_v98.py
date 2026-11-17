# -*- coding: utf-8 -*-
"""v98: robust側パラメータ(Student-t robust_nu ν / tempering temper_beta β)【だけ】を調整するチューナ。
★base PFパラメータは現状のまま(正しい)= config の値を固定し、ν/β のみ探索する(過去tune_v94はbaseを弄るので不適)。
評価は create と同一経路(pf_banks_v95.run_smoother_ext)を **普通のPF=lagなし(smooth_lag=0)** で回し、
tw表現の eval行 out['mean'] vs 真TVT の **overall RMSE**(全井プール)を最小化。n_seeds=32・ランダムseed。

2段グリッド(過学習回避で最小限):
  stage1: ν ∈ {0,3,4,5,6,8}(β=1.0固定) → 最小RMSEの ν* を採用(0=Gaussian=強化なし。baselineが勝てば自己選択)
  stage2: ν*>0 のとき β ∈ {1.0,0.9,0.85,0.8} → 最小の β* を採用
バンク毎に (ν*,β*) を pf_banks_config(v95) に焼く(base paramsは不変)。
env: NWELLS(スモーク) / N_WELLS(部分井, 既定=全) / SEARCH_NSEED(既定32) / SEARCH_SEED(既定ランダム) / BAKE(既定1)
実行: .venv/Scripts/python.exe v98/tune_robust_levers_v98.py
"""
import os, sys, io, glob, time, json
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd

V98 = Path(__file__).resolve().parent
PROJ = V98.parent
os.environ.setdefault("ROGII_ART95", str(V98 / "artifacts_v98" / "v95"))
os.environ.setdefault("V93_ANCHOR_PKL", str(V98 / "artifacts_v98" / "v95" / "grfree_anchor_train.pkl"))
sys.path.insert(0, str(V98))
import pf_banks_v95 as pf

CFG_PATH = Path(os.environ["ROGII_ART95"]) / "pf_banks_config.json"
DATA = pf.DATA_DIR; TR = DATA / "train"
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None            # スモーク用(先頭N井)
N_WELLS = int(os.environ["N_WELLS"]) if os.environ.get("N_WELLS") else None          # 部分井(ランダム)。既定=全
SEARCH_SEED = int(os.environ["SEARCH_SEED"]) if os.environ.get("SEARCH_SEED") else int.from_bytes(os.urandom(4), "little")
SEARCH_NSEED = int(os.environ.get("SEARCH_NSEED", "32"))       # ★n_seeds=32(ぶれ対策: ユーザー指示)
SEARCH_LAG = int(os.environ.get("SEARCH_LAG", "0"))            # ★普通のPF=lagなし(0)で調整
CHUNK = int(os.environ.get("PF_WELL_CHUNK", "60"))
NU_GRID = [float(x) for x in os.environ.get("NU_GRID", "0,3,4,5,6,8").split(",")]
BETA_GRID = [float(x) for x in os.environ.get("BETA_GRID", "1.0,0.9,0.85,0.8").split(",")]


def load_wells():
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    if NWELLS is not None:
        wells = wells[:NWELLS]
    elif N_WELLS is not None:
        rng = np.random.RandomState(SEARCH_SEED & 0x7fffffff); rng.shuffle(wells); wells = sorted(wells[:N_WELLS])
    data = []
    for w in wells:
        try:
            hw = pd.read_csv(TR / f"{w}__horizontal_well.csv")
            tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            continue
        ev = hw["TVT_input"].isna().to_numpy()
        if ev.sum() == 0:
            continue
        data.append((w, hw, tw, hw["TVT"].to_numpy(float)[ev]))
    return data


def eval_bank(bank, nu, beta, data):
    """base params(config)固定・ν/βのみ差替え・lag=0 で tw表現PFを回し overall RMSE。"""
    P = dict(pf.bank_param(bank)); P["robust_nu"] = nu; P["temper_beta"] = beta; P["smooth_lag"] = SEARCH_LAG
    inps = []; trues = []
    for (w, hw, tw, true) in data:
        x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
        if x is None:
            continue
        x = pf.attach_anchor(x, w, P.get("_physics"))
        inps.append(x); trues.append(true)
    outs = pf.run_smoother_ext(inps, P, SEARCH_SEED, SEARCH_NSEED, CHUNK, w_nn=0.0)
    errs = []
    for o, true in zip(outs, trues):
        pred = np.asarray(o["mean"], float); n = min(len(pred), len(true))
        e = pred[:n] - true[:n]; errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs) if errs else np.array([np.nan])
    return float(np.sqrt(np.mean(e ** 2)))


def tune_bank(bank, data):
    # stage1: ν(β=1.0)
    s1 = []
    for nu in NU_GRID:
        tb = time.time(); r = eval_bank(bank, nu, 1.0, data); s1.append((nu, r))
        print(f"  {bank:12s} [ν] ν={nu:<4g} β=1    RMSE={r:.4f}  ({time.time()-tb:.0f}s)", flush=True)
    nu_star, base = min(s1, key=lambda t: t[1])
    if nu_star == 0.0:
        return dict(robust_nu=0.0, temper_beta=1.0, rmse=base, base_rmse=base, gain=0.0, stage1=s1, stage2=[])
    # stage2: β(ν=ν*)
    s2 = [(1.0, base)]
    for beta in [b for b in BETA_GRID if b != 1.0]:
        tb = time.time(); r = eval_bank(bank, nu_star, beta, data); s2.append((beta, r))
        print(f"  {bank:12s} [β] ν={nu_star:<4g} β={beta:<4g} RMSE={r:.4f}  ({time.time()-tb:.0f}s)", flush=True)
    beta_star, best = min(s2, key=lambda t: t[1])
    base_rmse = [r for nu, r in s1 if nu == 0.0]; base_rmse = base_rmse[0] if base_rmse else float("nan")
    return dict(robust_nu=nu_star, temper_beta=beta_star, rmse=best, base_rmse=base_rmse,
                gain=base_rmse - best, stage1=s1, stage2=s2)


def main():
    t0 = time.time()
    data = load_wells()
    print(f"[robust-tune] 対象井={len(data)} banks={pf.BANK_ORDER} lag={SEARCH_LAG}(普通PF) n_seed={SEARCH_NSEED} "
          f"chunk={CHUNK} seed={SEARCH_SEED}  νグリッド={NU_GRID} βグリッド={BETA_GRID}", flush=True)
    results = {}
    for bank in pf.BANK_ORDER:
        r = tune_bank(bank, data); results[bank] = r
        print(f"  => {bank}: 最良 ν={r['robust_nu']} β={r['temper_beta']} RMSE={r['rmse']:.4f} "
              f"(base {r['base_rmse']:.4f}, −{r['gain']:.4f})\n", flush=True)
    (Path(os.environ["ROGII_ART95"]) / "robust_tune.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if os.environ.get("BAKE", "1") == "1":
        c = json.loads(CFG_PATH.read_text(encoding="utf-8"))
        for bank, r in results.items():
            if r["robust_nu"] > 0:
                c["params"][bank]["robust_nu"] = r["robust_nu"]; c["params"][bank]["temper_beta"] = r["temper_beta"]
            else:                                      # 強化なし採用=leverを外す
                c["params"][bank].pop("robust_nu", None); c["params"][bank].pop("temper_beta", None)
        CFG_PATH.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[bake] robust params 焼込: { {b: (r['robust_nu'], r['temper_beta']) for b, r in results.items()} }", flush=True)
    print(f"[robust-tune] 完了 ({time.time()-t0:.0f}s)  seed={SEARCH_SEED}", flush=True)


if __name__ == "__main__":
    main()
