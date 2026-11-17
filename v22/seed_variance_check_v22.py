# -*- coding: utf-8 -*-
"""
v22 シード変動 / 同一マシン決定性の確認
=======================================
目的: 「同じ調整済みPFでも seed を変えると結果が変わる(=ノイズ床)」を数値化し、
      さらに「同一マシン・同一seedで再実行すればビット一致するか(=このGPUの決定性)」を確認する。
      → クロスGPU(T4 vs RTX5070)はローカルで直接試せないが、seed変動幅が分かれば
        「別GPU/別torchなら少なくともこのseed差以上はズレる」と外挿できる。

やること:
  (1) seedスイープ: 各カテゴリの調整済みPF(per_category_pf.json)で **ensemble seed だけ変えて** pf_ancc を評価。
      seed = s*1000 + 100*code (s=0 が v22 tuner の正準シード=8.337 を再現)。s を数本振り、
      overall(全プール)と cat4 の RMSE の min/max/std(=ノイズ床)を出す。
  (2) 決定性: s=0 をもう一度走らせ、前回の per-well 予測と **最大絶対差**を比較(0なら決定的)。

実行: rogii_claude/.venv/Scripts/python.exe v22/seed_variance_check_v22.py
※ Claude は実行しない(ユーザーが実行)。
"""

import sys, json
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v22"))
import tune_pf_per_category_v22 as T22   # noqa: E402

# ============================== CONFIG ==============================
PERCAT        = PROJECT / "v22" / "artifacts_v22" / "per_category_pf.json"
SEEDS         = T22.EVAL_SEEDS          # アンサンブル本数(32)
CHUNK         = T22.EVAL_CHUNK          # 400
SEED_OFFSETS  = [0, 1, 2, 3, 4]         # s: 各 s で seed = s*1000 + 100*code
DETERMINISM_CHECK = True                # 同一seed再実行の一致確認
DEVICE        = T22.DEVICE
N_CATS        = T22.N_CATS
FOCUS_CODE    = 4                        # 特に注目するカテゴリ(硬い核)
# ===================================================================


def eval_pf_ancc(inps, P, seed, n_seeds, chunk):
    """pf_ancc(likelihood_scale 加重の単一出力)を per-well で返す。run_forward_ens と同一機構。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = T22._pad(sub, DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(int(seed))
        pts, ll = T22._pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            P, N, DEVICE, gen)
        pts = pts.view(w, S, -1).double().cpu().numpy(); ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]; llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                ww = np.exp(np.where(np.isfinite(llw), llw - mx, -np.inf) / max(ls, 1e-6))
                ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out[c0 + j] = (ww[:, None] * pr).sum(0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return out


def rmse(residlist):
    e = np.concatenate(residlist); e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def run_once(cats, CACHE, cat_wells, s):
    """全カテゴリを seed = s*1000 + 100*code で評価。per-well 予測 dict と RMSE(all, focus) を返す。"""
    preds = {}; truth = {}; wcode = {}
    for code in range(N_CATS):
        wl = cat_wells[code]; rec = cats.get(str(code))
        if not wl or not rec:
            continue
        P = dict(rec["best_params"]); P.setdefault("use_phys", T22.USE_PHYS)
        seed = s * 1000 + 100 * code
        inps = [T22.make_inputs(CACHE[w], P) for w in wl]
        res = eval_pf_ancc(inps, P, seed, SEEDS, CHUNK)
        for w, pr in zip(wl, res):
            preds[w] = pr; truth[w] = CACHE[w]["tvt_true"]; wcode[w] = code
    all_r = rmse([preds[w] - truth[w] for w in preds])
    foc_r = rmse([preds[w] - truth[w] for w in preds if wcode[w] == FOCUS_CODE])
    return preds, truth, wcode, all_r, foc_r


def main():
    cats = json.loads(PERCAT.read_text()).get("categories", {})
    print(f"device={DEVICE}  seeds={SEEDS}  offsets={SEED_OFFSETS}")

    wells = T22.list_wells()
    CACHE = {}
    for w in wells:
        c = T22.build_cache(w)
        if c is not None and c["tvt_true"] is not None:
            CACHE[w] = c
    cat_wells = {code: [] for code in range(N_CATS)}
    for w, c in CACHE.items():
        cat_wells[T22.well_code(c)].append(w)
    print("カテゴリ別 well 数:", {c: len(cat_wells[c]) for c in range(N_CATS)})

    # (1) seed スイープ
    print("\n=== (1) seed スイープ(同じ調整済みPF、ensemble seed だけ変える)===")
    print(f"{'s':>3} {'seed(cat4)':>10} {'overall(all)':>13} {f'cat{FOCUS_CODE}':>9}")
    all_list, foc_list, first_preds = [], [], None
    for s in SEED_OFFSETS:
        preds, truth, wcode, all_r, foc_r = run_once(cats, CACHE, cat_wells, s)
        if first_preds is None:
            first_preds = (preds, truth)
        all_list.append(all_r); foc_list.append(foc_r)
        print(f"{s:>3} {s*1000+400:>10} {all_r:>13.4f} {foc_r:>9.4f}")
    a = np.array(all_list); f = np.array(foc_list)
    print(f"\n  overall(all): min={a.min():.4f} max={a.max():.4f} spread={a.max()-a.min():.4f} std={a.std():.4f}")
    print(f"  cat{FOCUS_CODE}      : min={f.min():.4f} max={f.max():.4f} spread={f.max()-f.min():.4f} std={f.std():.4f}")
    print("  → この spread が『seedだけで動く幅=ノイズ床』。別GPU/別torchなら最低これ以上ズレる。")

    # (2) 同一seed再実行の決定性
    if DETERMINISM_CHECK:
        print("\n=== (2) 同一seed(s=0)再実行の決定性 ===")
        preds2, truth2, _, all_r2, _ = run_once(cats, CACHE, cat_wells, 0)
        p1, _ = first_preds
        maxdiff = 0.0; nzero = 0
        for w in p1:
            if w in preds2 and len(p1[w]) == len(preds2[w]):
                dd = float(np.nanmax(np.abs(p1[w] - preds2[w]))) if len(p1[w]) else 0.0
                maxdiff = max(maxdiff, dd); nzero += int(dd > 0)
        print(f"  overall(all) 1回目={all_list[0]:.6f}  2回目={all_r2:.6f}  差={abs(all_list[0]-all_r2):.2e}")
        print(f"  per-well 予測の最大絶対差={maxdiff:.3e}  (差>0 の坑井数={nzero}/{len(p1)})")
        if maxdiff == 0.0:
            print("  → 完全ビット一致。このGPUでは同一seedは決定的(乱数のみが変動要因)。")
        else:
            print("  → 同一seedでも微差あり=CUDAリダクション等の非決定性。別GPUなら更に大きくズレる。")


if __name__ == "__main__":
    main()
