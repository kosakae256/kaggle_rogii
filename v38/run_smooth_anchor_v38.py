# -*- coding: utf-8 -*-
"""
v38s: 錨入り smooth PF(lag192)を tw / self / nbr で実行し、CV最良アンサンブルを決める
=====================================================================================
- エンジン = v28 create の smoother(_v26_smoother_core)に v38 の row錨(struct×mult30)を挿した COPY。
  create本体は不変(parquet再現性の聖域)。
- パラメータ = v38_submit_pack.json の調整済み best(+smooth_lag=192)。
- 出力:
    artifacts_v38/smooth_anchor_traj.npz     {w}|tw / {w}|self / {w}|nbr = 平滑軌跡(seed尤度加重平均)
    artifacts_v38/v38_smooth_pack.json       アンサンブル重み(CV最良)+ 各種CV数値(提出ノートが読む)
実行: rogii_claude/.venv/Scripts/python.exe v38/run_smooth_anchor_v38.py   (GPU ~10-25分)
"""
import os, sys, json, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJECT / "v28"))
import create_train_parquet_v28 as C

ART = PROJECT / "v38" / "artifacts_v38"
PACK = json.loads((ART / "v38_submit_pack.json").read_text())
BP = {**PACK["params"], "smooth_lag": C.SMOOTH_LAG}
ANCHOR_MULT = float(PACK["anchor_mult"])
SEED_EVAL, N_SEEDS = 777, 32
STRUCT = pickle.loads((ART / "struct_prior_train.pkl").read_bytes())
BASE = C.DATA_DIR_V26 / "train"
DEV = C.V26_DEVICE
print(f"[v38s] params from {PACK['study']}  mult={ANCHOR_MULT:g}  lag={C.SMOOTH_LAG}  seed={SEED_EVAL}/{N_SEEDS}  dev={DEV}")


# ---------- 錨入り smoother(createのCOPY + lka。create本体は不変) ----------
def _pad_a(inps, device, dtype=C.V26_DTYPE):
    pad = C._v26_pad(inps, device)
    W = len(inps); Tmax = pad["md"].shape[1]
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"])
        anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
        if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
        ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
    pad["anc"] = anc.to(device); pad["ancs"] = ancs.to(device)
    return pad


def _smoother_core_anchor(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs,
                          P, N, device, gen, dtype=C.V26_DTYPE):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    AMUL = float(ANCHOR_MULT)
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        # ★v38 row錨(struct×mult。padded行は ancs=1e9 で実質1)
        asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        w_new = torch.where(obs[:, None], w * lk * lka, w * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]; pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik


def run_smoother_anchor(inps, P, seed, n_seeds, chunk):
    """seed尤度加重平均の平滑軌跡を返す(createの run_smoother_ext 相当+錨)。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _pad_a(sub, DEV)
        rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        pf, ps, ll = _smoother_core_anchor(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), P, N, DEV, gen)
        ps = ps.view(w, S, -1).double().cpu().numpy(); ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)])
                lkv = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lkv / max(ls_scale, 1e-6)); s = ww.sum()
                ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            out[c0 + j] = (ww[:, None] * ps[j, :, :T]).sum(0)
        if DEV == "cuda":
            torch.cuda.synchronize(); torch.cuda.empty_cache()
    return out


def main():
    names = sorted({Path(p).stem.replace("__horizontal_well", "")
                    for p in (BASE).glob("*__horizontal_well.csv")})
    # 近傍(createと同一ロジック, 自己除外)
    print("[nbr] 近傍探索 ...")
    NREFS, NBR_HAS = {}, {}
    for w in names:
        hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        refw = C.neighbors_of(cx, cy, exclude_wid=w)
        use = len(refw) >= C.NEED_REFS
        NREFS[w] = [C._ref_grtvt(r) for r, _ in refw] if use else []
        NBR_HAS[w] = int(use)
    print(f"      近傍あり {sum(NBR_HAS.values())}/{len(names)}")

    print("[build] 入力構築(tw/self/nbr + 錨) ...")
    inps = {s: [] for s in ["tw", "self", "nbr"]}
    truth, wl = {}, []
    for w in names:
        hw = pd.read_csv(BASE / f"{w}__horizontal_well.csv")
        tw = pd.read_csv(BASE / f"{w}__typewell.csv").sort_values("TVT")
        twt = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
        bs = C.build_smoother_inputs(hw, twt, twg, BP)
        if bs is None: continue
        sf = C.build_inputs_self(hw, twt, twg, BP)
        nb = C.build_inputs_nbr(hw, twt, twg, BP, NREFS[w])
        n = len(bs["md"]); sp = STRUCT.get(w)
        if sp is not None and len(sp["tvt"]) == n:
            anc = np.asarray(sp["tvt"], float); ancs = np.asarray(sp["sig"], float)
        else:
            anc = np.zeros(n); ancs = np.full(n, 1e9)
        for d in (bs, sf, nb):
            d["anc"] = anc; d["ancs"] = ancs
        inps["tw"].append(bs); inps["self"].append(sf); inps["nbr"].append(nb)
        ev = hw[hw["TVT_input"].isna()]
        truth[w] = ev["TVT"].to_numpy(float); wl.append(w)
    print(f"      wells {len(wl)}")

    trajs = {}
    for src in ["tw", "self", "nbr"]:
        t0 = time.time()
        outs = run_smoother_anchor(inps[src], BP, SEED_EVAL, N_SEEDS, C.WELL_CHUNK)
        trajs[src] = {w: o for w, o in zip(wl, outs)}
        e = np.concatenate([trajs[src][w] - truth[w] for w in wl]); e = e[np.isfinite(e)]
        print(f"[{src:4s}] overall={np.sqrt(np.mean(e**2)):.4f}  ({time.time()-t0:.0f}s)")

    # ---- CV最良アンサンブル(単純重み, simplex 0.05刻み)----
    try:
        hd = pd.read_csv(PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv")
        hard = set(hd[hd["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        hard = set()
    T3 = {w: np.stack([trajs["tw"][w], trajs["self"][w], trajs["nbr"][w]]) for w in wl}
    def pooled(wts, wells=None):
        tot = 0.0; n = 0
        for w in (wells or wl):
            p = (np.asarray(wts)[:, None] * T3[w]).sum(0)
            e = p - truth[w]; e = e[np.isfinite(e)]
            tot += (e ** 2).sum(); n += e.size
        return float(np.sqrt(tot / n))
    best = (1e9, (1, 0, 0))
    for a in np.arange(0, 1.0001, 0.05):
        for b in np.arange(0, 1.0001 - a, 0.05):
            c = 1.0 - a - b
            r = pooled((a, b, c))
            if r < best[0]: best = (r, (round(float(a), 2), round(float(b), 2), round(float(c), 2)))
    wts = best[1]
    hardw = [w for w in wl if w in hard]
    print(f"\n=== CV最良アンサンブル: w(tw,self,nbr)={wts}  overall={best[0]:.4f}  "
          f"hard={pooled(wts, hardw):.4f} ===")
    for lab, ww_ in [("tw", (1, 0, 0)), ("self", (0, 1, 0)), ("nbr", (0, 0, 1)), ("均等", (1/3, 1/3, 1/3))]:
        print(f"   {lab:4s}: overall={pooled(ww_):.4f}  hard={pooled(ww_, hardw):.4f}")

    pack_np = {}
    for src in ["tw", "self", "nbr"]:
        for w in wl: pack_np[f"{w}|{src}"] = trajs[src][w].astype(np.float32)
    np.savez_compressed(ART / "smooth_anchor_traj.npz", **pack_np)
    sp = dict(PACK, smooth_lag=int(C.SMOOTH_LAG), ens_weights=list(wts),
              cv_overall=best[0], cv_hard=pooled(wts, hardw),
              cv_tw=pooled((1, 0, 0)), cv_self=pooled((0, 1, 0)), cv_nbr=pooled((0, 0, 1)),
              seed=SEED_EVAL, n_seeds=N_SEEDS)
    (ART / "v38_smooth_pack.json").write_text(json.dumps(sp, indent=1))
    print(f"\nsaved smooth_anchor_traj.npz / v38_smooth_pack.json")
    print("次: Kaggleデータセットに v38_smooth_pack.json を追加 → v38/submit_v38s.ipynb を提出")


if __name__ == "__main__":
    main()
