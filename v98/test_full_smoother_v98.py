# -*- coding: utf-8 -*-
"""1バンクで full平滑(前方フィルタ+単一backward祖先sweep, L非依存) と fixed-lag=192 の精度を比較。
同一 forward(同一RNG・粒子・リサンプル)から両方を emit するので、差は「平滑法」だけ。overall RMSE(全井プール)で比較。
chunk=1(padding無し=公平)・n_seed少・少数井で、稼働中パイプラインとGPU共存。
env: BANK(既定pf_1) / NWELLS(既定150) / NSEED(既定16) / STRIDE(井間引き既定5)
実行: .venv/Scripts/python.exe v98/test_full_smoother_v98.py"""
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
DTYPE = pf.DTYPE; DEVICE = pf.DEVICE
TR = pf.DATA_DIR / "train"
BANKS = os.environ.get("BANKS", os.environ.get("BANK", "pf_1")).split(",")   # 複数バンク比較可
NWELLS = int(os.environ["NWELLS"]) if os.environ.get("NWELLS") else None      # 既定=全井
NSEED = int(os.environ.get("NSEED", "32"))                                    # ★n_seed32(ぶれ対策)
STRIDE = int(os.environ.get("STRIDE", "1"))
SEED = int(os.environ.get("SEED", "20250802"))
VRAM_BUDGET_GB = float(os.environ.get("VRAM_BUDGET_GB", "7.0"))               # ★履歴用VRAM予算(空GPUなら増やす)
MAXCHUNK = int(os.environ.get("MAXCHUNK", "64"))                              # 可変chunkの上限
LFIX = 192


@torch.no_grad()
def dual_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, amul, P, N, device, gen, L):
    """forward=_smoother_core と同一(w_nn=0)。pos履歴/祖先index保存 → fixed-lag(L)と full(全区間backward)両方返す。"""
    B, T = md.shape; dtype = DTYPE
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    NU = float(P.get("robust_nu", 0.0)); BETA = float(P.get("temper_beta", 1.0))
    L = max(0, min(L, T))

    def rn():
        return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    pos_hist = torch.empty((T, B, N), device=device, dtype=dtype)      # ★append-only(並べ替え無し)
    anc_hist = torch.empty((T, B, N), device=device, dtype=torch.int16)  # ★祖先index(N<32767でint16=VRAM半減)
    arL = torch.arange(N, device=device, dtype=torch.long)[None, :].expand(B, N)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
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
        tt = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - tt) + gB * tt; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        if NU > 0.0:
            lk = (1.0 + dp / NU) ** (-(NU + 1.0) * 0.5)
        else:
            lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if BETA != 1.0:
            lk = lk ** BETA
        lk = lk.clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()), torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        asig = (ancs[:, i][:, None] * amul).clamp_min(1e-6)
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        anc_step = arL
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            anc_step = torch.where(nm, idx, arL)                       # ★親slot(resample井のみ, 他は恒等)
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        pos_hist[i] = pos; anc_hist[i] = anc_step.to(torch.int16)      # ★保存(並べ替え無し, int16)
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
    # ---- ★full平滑: 単一backward祖先sweep(最終重みwで加重, O(T·B·N), L非依存) ----
    pts_full = torch.zeros((B, T), device=device, dtype=dtype)
    a = arL.clone(); wfin = w
    for i in range(T - 1, -1, -1):
        pos_t = torch.gather(pos_hist[i], 1, a)
        pts_full[:, i] = (wfin * pos_t).sum(1) - z[:, i]
        a = torch.gather(anc_hist[i], 1, a).long()                     # int16格納→long索引に戻す
    del pos_hist, anc_hist
    return pts_f, pts_s, pts_full, log_lik


def _prep_wells():
    """全対象井の入力(x)+true+長さ を作り、長さ昇順にソート(可変chunkのpadding最小化)。"""
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(TR / "*__horizontal_well.csv"))})[::STRIDE]
    if NWELLS:
        wells = wells[:NWELLS]
    items = []
    for wn in wells:
        try:
            hw = pd.read_csv(TR / f"{wn}__horizontal_well.csv"); tw = pd.read_csv(TR / f"{wn}__typewell.csv").sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            continue
        ev = hw["TVT_input"].isna().to_numpy(); true = hw["TVT"].to_numpy(float)[ev]
        items.append((wn, hw, tw, true))
    return items


def eval_bank(BANK, items):
    P = dict(pf.bank_param(BANK)); P["smooth_lag"] = LFIX
    N = int(P["n_particles"]); ls_scale = float(P["likelihood_scale"]); amul = float(P.get("anchor_mult", 1.0))
    # 入力構築 → 長さ順ソート
    built = []
    for (wn, hw, tw, true) in items:
        x = pf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P)
        if x is None:
            continue
        x = pf.attach_anchor(x, wn, P.get("_physics")); built.append((wn, x, true, len(x["md"])))
    built.sort(key=lambda r: r[3])                          # 長さ昇順
    e_fix = []; e_full = []; t0 = time.time(); done = 0
    i = 0
    while i < len(built):
        # 可変chunk: VRAM予算 Tmax×(W×S)×N×6byte ≤ budget で W を最大化
        Tmax = built[i][3]; W = 0; budget = VRAM_BUDGET_GB * 1e9
        while i + W < len(built) and W < MAXCHUNK:
            tmx = max(Tmax, built[i + W][3])
            if (W + 1) * NSEED * tmx * N * 6 > budget and W >= 1:
                break
            Tmax = tmx; W += 1
        sub = built[i:i + W]; i += W
        xs = [b[1] for b in sub]
        try:
            pad = pf._pad(xs, DEVICE); rep = lambda tt: tt.repeat_interleave(NSEED, 0)
            gen = torch.Generator(device=DEVICE); gen.manual_seed(SEED)
            pf_, ps_, pfull_, ll = dual_core(
                rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
                pad["glen"].repeat_interleave(NSEED), pad["vmin"].repeat_interleave(NSEED), pad["step"].repeat_interleave(NSEED),
                pad["gs"].repeat_interleave(NSEED), pad["ls"].repeat_interleave(NSEED), pad["ir"].repeat_interleave(NSEED),
                rep(pad["anc"]), rep(pad["ancs"]), amul, P, N, DEVICE, gen, LFIX)
            ll = ll.cpu().numpy(); ps_ = ps_.double().cpu().numpy(); pfull_ = pfull_.double().cpu().numpy()
        except RuntimeError as e:
            print(f"  [OOM?] W={W} Tmax={Tmax} -> skip ({e})", flush=True); torch.cuda.empty_cache(); continue
        for j, (wn, x, true, T) in enumerate(sub):
            sl = slice(j * NSEED, (j + 1) * NSEED); llw = ll[sl]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); ww = np.exp(np.where(np.isfinite(llw), llw - mx, -np.inf) / max(ls_scale, 1e-6))
                ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(NSEED, 1.0 / NSEED)
            else:
                ww = np.full(NSEED, 1.0 / NSEED)
            m_fix = (ww[:, None] * ps_[sl, :T]).sum(0); m_full = (ww[:, None] * pfull_[sl, :T]).sum(0)
            n = min(len(m_fix), len(true))
            e_fix.append(m_fix[:n] - true[:n]); e_full.append(m_full[:n] - true[:n])
        done += len(sub)
        if done % 60 < W:
            print(f"  {BANK} {done}/{len(built)} (W={W},Tmax={Tmax}, {time.time()-t0:.0f}s)", flush=True)
        del ps_, pfull_; torch.cuda.empty_cache()
    ef = np.concatenate(e_fix); ef = ef[np.isfinite(ef)]; efu = np.concatenate(e_full); efu = efu[np.isfinite(efu)]
    return float(np.sqrt(np.mean(ef ** 2))), float(np.sqrt(np.mean(efu ** 2))), len(built)


def main():
    items = _prep_wells()
    print(f"[full-smoother test] banks={BANKS} lag_fix={LFIX} n_seed={NSEED} 井={len(items)} "
          f"VRAM予算={VRAM_BUDGET_GB}GB maxchunk={MAXCHUNK}", flush=True)
    print(f"{'bank':11s} {'井':>4s} {'fixed192':>9s} {'full':>9s} {'差':>8s}  判定", flush=True)
    res = {}
    for BANK in BANKS:
        rf, ru, nw = eval_bank(BANK, items)
        res[BANK] = (rf, ru)
        print(f"{BANK:11s} {nw:4d} {rf:9.4f} {ru:9.4f} {ru-rf:+8.4f}  {'★full改善' if ru<rf else 'fixed優位'}", flush=True)
    print("\n=== まとめ(fixed-lag192 vs full平滑) ===", flush=True)
    for b, (rf, ru) in res.items():
        print(f"  {b:11s}: fixed={rf:.4f} full={ru:.4f} 差={ru-rf:+.4f}", flush=True)


if __name__ == "__main__":
    main()
