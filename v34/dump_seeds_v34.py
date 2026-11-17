# -*- coding: utf-8 -*-
"""v34: smooth PF の per-seed 平滑軌跡 + log尤度を dump する。
=====================================================================================
- **paramセット = PF_LIB の pf_1 / pf_2 / pf_3**(ユーザー指示。PF_BANK の r0系ではない)。
- **ソースは tw / self / nbr の3種すべて**を各paramで実行 = 計9本(v28 create の PFパートと同構成)。
  「self/nbr含めてどの出力(seed)を信用すべきか」を学習するための素材なので self/nbr の per-seed も必要。
- v28 create と同一条件(SEED_V26 / N_SEED=32 / lag192 / WELL_CHUNK / 井戸順=parquet順 /
  nbr無し井戸も tw フォールバックで nbrパスに参加=createと同じRNG消費)。
- 自己チェック: dump の seed尤度加重平均 ≒ parquet の pf_ancc_k / pf_self_k / pf_nbr_k。
  ここが大きくズレる場合は create の PF_LIB 等が artifacts_v28 構築時からドリフトしている
  可能性がある(結果を使う前に要相談)。

出力: v34/artifacts_v34/seed_dump_pf{1,2,3}.npz
  各井戸 w につき(src ∈ tw/self/nbr):
    {w}|ps_{src}  : (32, T) float32  per-seed 平滑TVT軌跡(固定ラグsmoother, eval行順)
    {w}|pfw_{src} : (32, T) float32  per-seed forward軌跡(平滑なし=素のフィルタ出力)
    {w}|ll_{src} : (32,)  float64  per-seed log尤度
    {w}|ww_{src} : (32,)  float64  seed重み(likelihood_scale 温度付き softmax = create と同一)
    {w}|gg_tw / {w}|gg_self / {w}|gg_nbr : GR grid(gg_nbr は近傍>=2本の井戸のみ)
    {w}|meta : [gmin, gst, gs]   {w}|gr : (T,) 観測GR(このparamの平滑済み)
    {w}|md / {w}|z : (T,) eval行の MD / Z   {w}|ir : 既知帯末尾から回帰した dip prior(物理特徴用)

実行: .venv/Scripts/python.exe v34/dump_seeds_v34.py
(9本=v28 create のPFパートと同等の時間。OOMなら create 側の WELL_CHUNK を下げる)
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v28"))
import create_train_parquet_v28 as C

ART = PROJECT / "v34" / "artifacts_v34"; ART.mkdir(parents=True, exist_ok=True)
V28_PQ = PROJECT / "v28" / "artifacts_v28" / "data" / "train.parquet"
BASE = C.DATA_DIR_V26 / "train"
PARAMS = ["pf_1", "pf_2", "pf_3"]        # ★PF_LIB の pf_1/2/3 を使う(PF_BANK ではない)
SRCS = ["tw", "self", "nbr"]

# ---- 井戸リストと順序 = 学習に使った parquet と同一(チャンク構成が同じ → RNG再現)----
CHK_COLS = [f"pf_ancc_{k}" for k in (1, 2, 3)] + [f"pf_self_{k}" for k in (1, 2, 3)] + \
           [f"pf_nbr_{k}" for k in (1, 2, 3)]
tbq = pq.read_table(V28_PQ, columns=["well"] + CHK_COLS)
gq = tbq.column("well").to_pandas().to_numpy()
names = list(pd.unique(gq))
rows_of = {w: np.where(gq == w)[0] for w in names}
CHK = {c: tbq.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in CHK_COLS}
print(f"[v34] wells={len(names)}  device={C.V26_DEVICE}  n_seed={C.N_SEED}  lag={C.SMOOTH_LAG}  chunk={C.WELL_CHUNK}")
print(f"[v34] params={PARAMS} × src={SRCS} = 計{len(PARAMS)*len(SRCS)}本")


def run_smoother_dump(inps, P, seed, n_seeds, chunk):
    """create の run_smoother_ext と同一のRNG消費で、平均せず per-seed を返す。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = C._v26_pad(sub, C.V26_DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=C.V26_DEVICE); gen.manual_seed(seed)
        pf, ps, ll = C._v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            P, N, C.V26_DEVICE, gen)
        ps = ps.view(w, S, -1).double().cpu().numpy()
        pf = pf.view(w, S, -1).double().cpu().numpy()   # forward(平滑なし)も候補として保持
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk / max(ls_scale, 1e-6)); s = ww.sum()
                ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out[c0 + j] = dict(ps=ps[j, :, :T].astype(np.float32),
                               pfw=pf[j, :, :T].astype(np.float32),
                               ll=llw.astype(np.float64), ww=ww.astype(np.float64))
        if C.V26_DEVICE == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    return out


# ---- 近傍探索(param非依存・1回だけ。create の (3) と同一ロジック)----
print("[nbr] 近傍探索 ...")
NREFS, NBR_HAS = {}, {}
for w in names:
    hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
    cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
    refw = C.neighbors_of(cx, cy, exclude_wid=w)
    use = len(refw) >= C.NEED_REFS
    NREFS[w] = [C._ref_grtvt(r) for r, _ in refw] if use else []
    NBR_HAS[w] = int(use)
print(f"      近傍あり {sum(NBR_HAS.values())}/{len(names)}(無し井戸は tw フォールバックで nbrパスにも参加=create同一)")

# ---- param ごと: 入力構築(tw/self/nbr) → 3ソース実行 → 自己チェック → 保存 ----
for ki, lab in enumerate(PARAMS):
    suf = ki + 1
    Pk = {**C.PF_LIB[lab], "smooth_lag": C.SMOOTH_LAG}
    print(f"[pf_{suf}] {lab}: 入力構築 ...")
    t0 = time.perf_counter()
    inps = {s: [] for s in SRCS}; grids = {}
    for w in names:
        hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv")
        tw = pd.read_csv(BASE / f"{w}__typewell.csv").sort_values("TVT")
        twt = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
        bs = C.build_smoother_inputs(hw, twt, twg, Pk)
        if bs is None:
            raise RuntimeError(f"build_smoother_inputs None: {w}(parquet構築時と条件が違う)")
        sf = C.build_inputs_self(hw, twt, twg, Pk)
        nb = C.build_inputs_nbr(hw, twt, twg, Pk, NREFS[w])   # refs無し → tw フォールバック(create同一)
        inps["tw"].append(bs); inps["self"].append(sf); inps["nbr"].append(nb)
        grids[w] = dict(gg_tw=bs["gg"].astype(np.float32), gg_self=sf["gg"].astype(np.float32),
                        gg_nbr=(nb["gg"].astype(np.float32) if NBR_HAS[w] else None),
                        meta=np.array([bs["gmin"], bs["gst"], bs["gs"]], dtype=np.float64),
                        gr=bs["gr"].astype(np.float32),
                        md=bs["md"].astype(np.float32), zz=bs["z"].astype(np.float32),
                        ir=np.float64(bs["ir"]))
    print(f"      入力 {time.perf_counter()-t0:.0f}s")

    pack = {}
    for w in names:
        gd = grids[w]
        pack[f"{w}|gg_tw"] = gd["gg_tw"]; pack[f"{w}|gg_self"] = gd["gg_self"]
        if gd["gg_nbr"] is not None:
            pack[f"{w}|gg_nbr"] = gd["gg_nbr"]
        pack[f"{w}|meta"] = gd["meta"]; pack[f"{w}|gr"] = gd["gr"]
        pack[f"{w}|md"] = gd["md"]; pack[f"{w}|z"] = gd["zz"]; pack[f"{w}|ir"] = gd["ir"]

    chk_col = {"tw": f"pf_ancc_{suf}", "self": f"pf_self_{suf}", "nbr": f"pf_nbr_{suf}"}
    for src in SRCS:
        print(f"      [smoothPF-{src}] 実行(32seed) ...")
        t0 = time.perf_counter()
        outs = run_smoother_dump(inps[src], Pk, C.SEED_V26, C.N_SEED, C.WELL_CHUNK)
        print(f"        PF {time.perf_counter()-t0:.0f}s")
        diffs = []
        ref = CHK[chk_col[src]]
        for w, o in zip(names, outs):
            r = rows_of[w]; T = min(o["ps"].shape[1], len(r))
            smean = (o["ww"][:, None] * o["ps"].astype(np.float64)).sum(0)
            diffs.append(float(np.nanmedian(np.abs(smean[:T] - ref[r][:T]))))
            if o["ps"].shape[1] != len(r):
                print(f"        [warn] 行数不一致 {w}: dump T={o['ps'].shape[1]} vs parquet {len(r)}")
            pack[f"{w}|ps_{src}"] = o["ps"]; pack[f"{w}|pfw_{src}"] = o["pfw"]
            pack[f"{w}|ll_{src}"] = o["ll"]; pack[f"{w}|ww_{src}"] = o["ww"]
        med = float(np.nanmedian(diffs)); p90 = float(np.nanpercentile(diffs, 90))
        print(f"        [check {chk_col[src]}] |dump加重平均 − parquet| median={med:.3f}ft / p90={p90:.3f}ft"
              + ("  → OK(再現)" if med < 0.5 else "  → ★ズレ大: create の設定ドリフトの疑い(要確認)"))

    np.savez(ART / f"seed_dump_pf{suf}.npz", **pack)
    print(f"      saved: {ART / f'seed_dump_pf{suf}.npz'}")

print("[v34] dump 完了(3param × tw/self/nbr = 9本)")
