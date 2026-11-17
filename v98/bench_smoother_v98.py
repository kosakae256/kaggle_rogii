# -*- coding: utf-8 -*-
"""full平滑 の 速度＆精度 ベンチ(全井・1バンク・実行環境同ルール)。
3モードを個別に計測:
  plain  : 素PF(smooth_lag=0, forward filterのみ, chunk60)          — run_smoother_ext
  fix192 : smooth=192 固定ラグ(chunk20 ← createと同じVRAMルール)     — run_smoother_ext
  full   : 全区間 系譜平滑(forward+履歴+単一backward, buffer無し, 可変chunk)
各モードの wall-clock(全井) と overall RMSE(eval行, tw表現, pred vs 真TVT)を出す。n_seed=32。
env: BANK(既定pf_1) / NSEED(32) / FULL_VRAM_GB(8) / FIX_CHUNK(20) / PLAIN_CHUNK(60)
実行: .venv/Scripts/python.exe v98/bench_smoother_v98.py"""
import os, sys, io, glob, time
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, torch

V98 = Path(__file__).resolve().parent
os.environ.setdefault("ROGII_ART95", str(V98 / "artifacts_v98" / "v95"))
os.environ.setdefault("V93_ANCHOR_PKL", str(V98 / "artifacts_v98" / "v95" / "grfree_anchor_train.pkl"))
sys.path.insert(0, str(V98))
import pf_banks_v95 as pf
DTYPE = pf.DTYPE; DEVICE = pf.DEVICE; TR = pf.DATA_DIR / "train"
BANK = os.environ.get("BANK", "pf_1")
NSEED = int(os.environ.get("NSEED", "32"))
FULL_VRAM = float(os.environ.get("FULL_VRAM_GB", "8.0"))
FIX_CHUNK = int(os.environ.get("FIX_CHUNK", "20"))
PLAIN_CHUNK = int(os.environ.get("PLAIN_CHUNK", "60"))
SEED = 20250802


@torch.no_grad()
def full_only_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, amul, P, N, device, gen):
    """forward filter + 全履歴保存 + 単一backward祖先sweep(fixed-lagバッファ無し=full本来)。pts_full(B,T)返す。"""
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    NU = float(P.get("robust_nu", 0.0)); BETA = float(P.get("temper_beta", 1.0))

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=DTYPE)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=DTYPE)
    ll = torch.zeros(B, device=device, dtype=torch.float64)
    pos_hist = torch.empty((T, B, N), device=device, dtype=DTYPE)  # ★fp32(精度優先。fp16は+0.10悪化で不採用)
    anc_hist = torch.empty((T, B, N), device=device, dtype=torch.int16)
    lsr = ls[:, None]
    arL = torch.arange(N, device=device, dtype=torch.long)[None, :].expand(B, N)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(DTYPE) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=DTYPE); zero = torch.zeros((), device=device, dtype=DTYPE); dipp = ir[:, None]
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=DTYPE) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        tt = ii - i0c.to(DTYPE); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - tt) + gB * tt; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = ((1.0 + dp / NU) ** (-(NU + 1.0) * 0.5)) if NU > 0.0 else torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if BETA != 1.0:
            lk = lk ** BETA
        lk = lk.clamp_min(1e-300)
        lkp = torch.exp(-0.5 * ((rate - dipp) / PHYS_SIG) ** 2).clamp_min(1e-300) if USE_P else torch.ones_like(w)
        asig = (ancs[:, i][:, None] * amul).clamp_min(1e-6); da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True); w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        need = act & ((1.0 / (w * w).sum(1)) < (RESAMP * N)); anc_step = arL
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=DTYPE) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w); anc_step = torch.where(nm, idx, arL)
        obsd = torch.where(obs, torch.log(( (w_new.sum(1)).clamp_min(1e-300)).double()), torch.zeros((), device=device, dtype=torch.float64))
        ll = ll + obsd
        pos_hist[i] = (pos - lsr); anc_hist[i] = anc_step.to(torch.int16); pm = torch.where(act, cur, pm)
    pts_full = torch.zeros((B, T), device=device, dtype=DTYPE); a = arL.clone(); wfin = w
    for i in range(T - 1, -1, -1):
        pts_full[:, i] = (wfin * (torch.gather(pos_hist[i], 1, a).to(DTYPE) + lsr)).sum(1) - z[:, i]
        a = torch.gather(anc_hist[i], 1, a).long()
    del pos_hist, anc_hist
    return pts_full, ll


def load_wells(bankname):
    P = dict(pf.bank_param(bankname))
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})
    data = []
    for w in wells:
        try:
            hw = pd.read_csv(TR / f"{w}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{w}__typewell.csv").sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            continue
        x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
        if x is None:
            continue
        x = pf.attach_anchor(x, w, P.get("_physics"))
        ev = hw["TVT_input"].isna().to_numpy()
        data.append((w, x, hw["TVT"].to_numpy(float)[ev]))
    return P, data


def rmse_of(means, data):
    errs = []
    for (w, x, true), mn in zip(data, means):
        n = min(len(mn), len(true)); e = np.asarray(mn[:n], float) - true[:n]; errs.append(e[np.isfinite(e)])
    e = np.concatenate(errs); return float(np.sqrt(np.mean(e ** 2)))


def run_ext(P, data, lag, chunk):
    """run_smoother_ext(create同一) で全井。out['mean']返す。"""
    Pm = dict(P); Pm["smooth_lag"] = lag
    os.environ["PF_WELL_CHUNK"] = str(chunk)
    inps = [x for (_, x, _) in data]
    t0 = time.time(); outs = pf.run_smoother_ext(inps, Pm, SEED, NSEED, chunk, w_nn=0.0); dt = time.time() - t0
    return [o["mean"] for o in outs], dt


def run_full(P, data):
    """full_only_core を可変chunk(VRAM予算)で全井。seed尤度加重平均。"""
    N = int(P["n_particles"]); amul = float(P.get("anchor_mult", 1.0)); ls_scale = float(P["likelihood_scale"])
    built = [(w, x, true, len(x["md"])) for (w, x, true) in data]; built.sort(key=lambda r: r[3])
    means = {}; t0 = time.time(); i = 0; budget = FULL_VRAM * 1e9
    while i < len(built):
        Tmax = built[i][3]; W = 0
        while i + W < len(built) and W < 64:
            tmx = max(Tmax, built[i + W][3])
            if (W + 1) * NSEED * tmx * N * 6 > budget and W >= 1:
                break
            Tmax = tmx; W += 1
        sub = built[i:i + W]; i += W; pad = pf._pad([b[1] for b in sub], DEVICE); rep = lambda t: t.repeat_interleave(NSEED, 0)
        gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)
        pfull, ll = full_only_core(rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                                   pad["glen"].repeat_interleave(NSEED), pad["vmin"].repeat_interleave(NSEED), pad["step"].repeat_interleave(NSEED),
                                   pad["gs"].repeat_interleave(NSEED), pad["ls"].repeat_interleave(NSEED), pad["ir"].repeat_interleave(NSEED),
                                   rep(pad["anc"]), rep(pad["ancs"]), amul, P, N, DEVICE, gen)
        pfull = pfull.double().cpu().numpy(); ll = ll.cpu().numpy()
        for j, (w, x, true, T) in enumerate(sub):
            sl = slice(j * NSEED, (j + 1) * NSEED); llw = ll[sl]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); ww = np.exp(np.where(np.isfinite(llw), llw - mx, -np.inf) / max(ls_scale, 1e-6)); ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(NSEED, 1.0 / NSEED)
            else:
                ww = np.full(NSEED, 1.0 / NSEED)
            means[w] = (ww[:, None] * pfull[sl, :T]).sum(0)
        torch.cuda.empty_cache()
    dt = time.time() - t0
    return [means[w] for (w, x, true) in data], dt


def fix192_from_parquet():
    """smooth=192(実行済みPF)は再計算せず parquet(v96_normal)から精度取得(ユーザー指示)。tw各バンクdelta の overall RMSE。"""
    import pyarrow.parquet as pq
    pqp = V98 / "artifacts_v98" / "data" / "v96_normal.parquet"
    if not pqp.exists():
        return None
    names = pq.read_schema(pqp).names
    cand = [c for c in names if c == "pf_ancc_delta" or (c.startswith("pf_ancc_delta_") and c.split("_")[-1].isdigit())]
    t = pq.read_table(pqp, columns=["target"] + cand); y = t.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    out = {}
    for c in cand:
        d = t.column(c).to_numpy(zero_copy_only=False).astype(np.float64); e = d - y; e = e[np.isfinite(e)]
        out[c] = float(np.sqrt(np.mean(e ** 2)))
    return out


def main():
    banks = os.environ.get("BANKS", BANK).split(",")
    order = list(pf.BANK_ORDER)
    fx = fix192_from_parquet() or {}                 # parquet の pf_ancc_delta{_k} = BANK_ORDER[k-1]
    rows = []
    for b in banks:
        P, data = load_wells(b)
        means, dt_p = run_ext(P, data, 0, PLAIN_CHUNK); r_p = rmse_of(means, data)
        means, dt_f = run_full(P, data); r_f = rmse_of(means, data)
        idx = order.index(b) if b in order else -1        # fix192 = parquet 対応列
        col = "pf_ancc_delta" if idx == 0 else f"pf_ancc_delta_{idx+1}"
        r192 = fx.get(col, float("nan"))
        rows.append((b, r_p, r_f, r192, dt_p, dt_f))
        print(f"  {b:11s}: plain(lag0)={r_p:.4f}/{dt_p:.0f}s  full={r_f:.4f}/{dt_f:.0f}s  smooth192(parq)={r192:.4f}", flush=True)
    print(f"\n=== 全井ベンチ まとめ(n_seed={NSEED}) ===", flush=True)
    print(f"{'bank':11s} {'素PF':>7s} {'full':>7s} {'sm192':>7s} | {'full-素':>7s} {'full-192':>8s} | {'plain s':>7s} {'full s':>7s}", flush=True)
    for b, r_p, r_f, r192, dt_p, dt_f in rows:
        print(f"{b:11s} {r_p:7.3f} {r_f:7.3f} {r192:7.3f} | {r_f-r_p:+7.3f} {r_f-r192:+8.3f} | {dt_p:7.0f} {dt_f:7.0f}", flush=True)
    print("※ smooth=192速度=create実測 ~1020s/回。full は概ね ~600s = fix192比 約1.7倍速。", flush=True)


if __name__ == "__main__":
    main()
