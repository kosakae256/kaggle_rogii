# -*- coding: utf-8 -*-
"""
v53 — Evidence Fusion Field の証拠テンソル EV23 を生成(train)
================================================================
ch1-10 : v50 evidenceB3(既存 pkl)
ch11-22: ★PF事後場 12ch = 6バンク × {tw, nbr} の行別粒子質量ヒストグラム
         (NN-emission W=0.02 / pfAのみ W=0.01+struct錨, 16seed平均, forward事後, stride16)
ch23   : v50B(combo 5本平均)の事後 P̄
出力: v53/artifacts_v53/evidenceC_v53.pkl {well: dict(EV(23,nb,181)fp16, dtrue, wgt, st, bid, y_ev)}
実行: python v53/create_evidence_v53.py     (GPU ~2h。BANKS/SEEDS は env で調整可)
"""
import os, sys, time, json, math, pickle
os.environ.setdefault("TAG", "capaug"); os.environ.setdefault("CH", "128")
os.environ.setdefault("DIM", "96"); os.environ.setdefault("DILS", "1,2,4,8,16,32")
os.environ.setdefault("BTAG", "combo"); os.environ.setdefault("REFH", "1")
os.environ.setdefault("DILT", "1,2,4,8,16"); os.environ.setdefault("BCH", "96")
import numpy as np, pandas as pd, torch
from pathlib import Path
P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(P0 / "v51")); sys.path.insert(0, str(P0 / "v50"))
import nnpf_v51 as npf                     # PF_LIB / build_smoother_inputs / SIMD(OOF sim) / 定数
import stageB_v50 as sb                    # EVD(10ch評価証拠) / DecoderB(v50B)
import stageA_v50 as sa

DEV = npf.V26_DEVICE
ART53 = P0 / "v53" / "artifacts_v53"; ART53.mkdir(parents=True, exist_ok=True)
BANKS = os.environ.get("BANKS", "pf_1,pf_2,pf_3,r0_seed32,r1_seed32,pfA").split(",")
FLAVORS = os.environ.get("FLAVORS", "tw,nbr,self").split(",")   # self を足すなら FLAVORS=tw,nbr,self
SEEDS = int(os.environ.get("SEEDS", "16")); CHUNK = int(os.environ.get("CHUNK", "24"))
LAG = 16; HSTRIDE = 16; ND = 181
SEED0 = 4423098
HIST_FP = ART53 / "pfhist_cache_v53.npz"    # PF事後場のパス途中キャッシュ(bank×flavor単位で追記)

# ---------------- hist対応 PFコア(nnpf_v51._core + scatter 追加) ----------------
def _core_h(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, stq, sgq, sim, wmap, W_NN, use_anchor,
            hist, P, N, device, gen, dtype=torch.float32):
    B, T = md.shape
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
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
        jj = torch.round((v - stq[:, i][:, None]) / npf.SIM_STEP).long() + npf.NBAND
        inb = (jj >= 0) & (jj <= 2 * npf.NBAND)
        if W_NN > 0:
            sim_i = sim[:, i, :].to(dtype)[wmap]
            sv = torch.gather(sim_i, 1, jj.clamp(0, 2 * npf.NBAND))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lknn = torch.exp(W_NN * sv)
        else:
            lknn = torch.ones_like(w)
        if use_anchor:
            asig = (sgq[:, i][:, None] * npf.V41M).clamp_min(1e-6)
            da = (v - stq[:, i][:, None]) / asig
            lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
            lknn = lknn * lka
        w_new = torch.where(obs[:, None], w * lk * lknn, w * lknn)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        # ★v53: 行事後を δ場に scatter(リサンプル前の重み=事後)
        if hist is not None:
            wt = torch.where(am & inb, w, zero.expand_as(w))
            hist[:, i // HSTRIDE, :].scatter_add_(1, jj.clamp(0, 2 * npf.NBAND), wt)
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
                pts_s[:, i - L] = (w * buf[:, ptr, :]).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    return pts_f, pts_s, log_lik

def run_hist(inps, sims_w, sgs_w, P, seed, n_seeds, chunk, W_NN, use_anchor=False):
    """各wellの PF事後場 (nb,181) fp16(seed平均・行正規化)を返す。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; wsub = sims_w[c0:c0 + chunk]; wq = len(sub)
        Tmax = max(len(x["md"]) for x in sub); Gmax = max(len(x["gg"]) for x in sub)
        md = torch.zeros((wq, Tmax)); z = torch.zeros((wq, Tmax))
        gr = torch.full((wq, Tmax), float("nan")); valid = torch.zeros((wq, Tmax), dtype=torch.bool)
        grid = torch.zeros((wq, Gmax)); glen = torch.zeros(wq, dtype=torch.long)
        vmin = torch.zeros(wq); step = torch.zeros(wq); gs = torch.zeros(wq)
        ls = torch.zeros(wq); ir = torch.zeros(wq)
        stq = torch.zeros((wq, Tmax)); simt = torch.full((wq, Tmax, ND), -1.0, dtype=torch.float16)
        sgq = torch.full((wq, Tmax), 1e9)
        for b, (x, sd, sg_) in enumerate(zip(sub, wsub, sgs_w[c0:c0 + chunk])):
            Tn = len(x["md"]); G = len(x["gg"])
            md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
            if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
            z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
            gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
            grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
            if G < Gmax: grid[b, G:] = grid[b, G - 1]
            glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
            Ts = min(Tn, len(sd["st"]))
            stq[b, :Ts] = torch.from_numpy(sd["st"][:Ts])
            if Ts < Tmax: stq[b, Ts:] = stq[b, Ts - 1]
            simt[b, :Ts] = torch.from_numpy(sd["sim"][:Ts])
            if sg_ is not None:
                Ts2 = min(Tn, len(sg_)); sgq[b, :Ts2] = torch.from_numpy(sg_[:Ts2].astype("float32"))
        to = lambda tt: tt.to(DEV); rep = lambda tt: tt.repeat_interleave(S, 0)
        gen = torch.Generator(device=DEV); gen.manual_seed(seed)
        wmap = torch.arange(wq, device=DEV).repeat_interleave(S)
        nbH = (Tmax + HSTRIDE - 1) // HSTRIDE
        hist = torch.zeros((wq * S, nbH, ND), device=DEV, dtype=torch.float32)
        _core_h(rep(to(md)), rep(to(z)), rep(to(gr)), rep(to(valid)), rep(to(grid)),
                to(glen).repeat_interleave(S), to(vmin).repeat_interleave(S), to(step).repeat_interleave(S),
                to(gs).repeat_interleave(S), to(ls).repeat_interleave(S), to(ir).repeat_interleave(S),
                rep(to(stq)), rep(to(sgq)), to(simt), wmap, W_NN, use_anchor, hist,
                {**P, "smooth_lag": LAG}, N, DEV, gen)
        hb = hist.view(wq, S, nbH, ND).mean(1)
        hb = hb / hb.sum(-1, keepdim=True).clamp_min(1e-9)          # 行正規化(有効行のみ実質)
        hb = hb.cpu().numpy()
        for b in range(wq):
            nb_i = (len(sub[b]["md"]) + HSTRIDE - 1) // HSTRIDE
            out[c0 + b] = hb[b, :nb_i].astype(np.float16)
        if DEV == "cuda": torch.cuda.empty_cache()
    return out

# ---------------- 入力構築(tw / nbr フレーバー) ----------------
def build_inputs_self_local(hw, tw_tvt, tw_gr, P):
    """typewellグリッドを自坑井既知区間(TVT_input, GR)で上書き(self擬似typewell)。"""
    base = npf.build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    ys = kn["TVT_input"].to_numpy(float)
    gs_ = pd.Series(hw["GR"].to_numpy(float)).interpolate(limit_direction="both").to_numpy()
    gs_ = np.where(np.isfinite(gs_), gs_, np.nanmedian(gs_))[hw.index.get_indexer(kn.index)]
    G = len(base["gg"]); gmin = base["gmin"]; gst = base["gst"]
    idx = np.round((ys - gmin) / gst).astype(np.int64)
    ok = (idx >= 0) & (idx < G) & np.isfinite(ys)
    if ok.sum() >= 20:
        ssum = np.zeros(G); scnt = np.zeros(G)
        np.add.at(ssum, idx[ok], gs_[ok]); np.add.at(scnt, idx[ok], 1)
        base = dict(base); base["gg"] = np.where(scnt > 0, ssum / np.maximum(scnt, 1), base["gg"])
    return base

def build_inputs_nbr_local(hw, tw_tvt, tw_gr, P, nbr_pts):
    """typewellグリッドの nbr データ有ビンを近傍既知プールで上書き(v47 nbr の簡易版)。"""
    base = npf.build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None or nbr_pts is None: return base
    ny, ng = nbr_pts
    G = len(base["gg"]); gmin = base["gmin"]; gst = base["gst"]
    idx = np.round((ny - gmin) / gst).astype(np.int64)
    ok = (idx >= 0) & (idx < G)
    if ok.sum() >= 20:
        ssum = np.zeros(G); scnt = np.zeros(G)
        np.add.at(ssum, idx[ok], ng[ok]); np.add.at(scnt, idx[ok], 1)
        gg2 = np.where(scnt > 0, ssum / np.maximum(scnt, 1), base["gg"])
        base = dict(base); base["gg"] = gg2
    return base

if __name__ == "__main__":
    t0 = time.time()
    wells = sorted(npf.SIMD.keys())
    if os.environ.get("NWELLS"): wells = wells[:int(os.environ["NWELLS"])]   # スモーク用
    cache = pickle.loads(sa.CACHE.read_bytes())
    SGD = {w: cache[w]["sg"] for w in cache}
    XY = np.array([cache[w]["xy"] for w in wells])
    KN = {}
    for w in wells:
        c = cache[w]; knm = (~c["ev"]) & np.isfinite(c["y"])
        KN[w] = (c["y"][knm].astype(np.float64), c["graw"][knm].astype(np.float64))
    from scipy.spatial import cKDTree
    _, nidx = cKDTree(XY).query(XY, k=min(9, len(wells)))
    NBRPTS = {}
    for i, w in enumerate(wells):
        js = [j for j in np.atleast_1d(nidx[i]) if j < len(wells) and wells[j] != w][:8]
        NBRPTS[w] = ((np.concatenate([KN[wells[j]][0] for j in js]),
                      np.concatenate([KN[wells[j]][1] for j in js])) if js else None)
    del cache
    DATA = []
    for wn in wells:
        hw = pd.read_csv(npf.TRAIN / f"{wn}__horizontal_well.csv")
        tw = pd.read_csv(npf.TRAIN / f"{wn}__typewell.csv").sort_values("TVT")
        if "TVT" not in hw.columns: continue
        if hw["TVT_input"].isna().sum() == 0 or hw["TVT_input"].notna().sum() < 10: continue
        DATA.append((wn, hw, tw))
    print(f"wells {len(DATA)}  banks {BANKS}  flavors tw/nbr  seeds {SEEDS}  ({time.time()-t0:.0f}s)", flush=True)

    HISTS = dict(np.load(HIST_FP, allow_pickle=False)) if HIST_FP.exists() else {}
    if HISTS: print(f"hist cache: {len(HISTS)} entries")
    for bi, bank in enumerate(BANKS):
        Praw = {k: v for k, v in npf.PF_LIB[bank].items() if k != "name"}
        wnn = 0.01 if bank == "pfA" else 0.02
        use_anc = (bank == "pfA")
        for fi, flav in enumerate(FLAVORS):
            if all(f"{wn}|{bank}|{flav}" in HISTS for wn, _, _ in DATA):   # ★全坑井カバレッジで判定(部分キャッシュ対策)
                print(f"[skip cached] {bank}/{flav}"); continue
            print(f"[pfhist] {bank}/{flav} (W_NN={wnn}, anchor={use_anc}) ...", flush=True)
            t1 = time.time(); inps = []; keep = []
            for wn, hw, tw in DATA:
                if flav == "tw":
                    x = npf.build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw)
                elif flav == "self":
                    x = build_inputs_self_local(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw)
                else:
                    x = build_inputs_nbr_local(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Praw, NBRPTS.get(wn))
                if x is not None: inps.append(x); keep.append(wn)
            sims_w = [npf.SIMD[w] for w in keep]; sgs_w = [SGD.get(w) for w in keep]
            hs = run_hist(inps, sims_w, sgs_w, Praw, SEED0, SEEDS, CHUNK, wnn, use_anchor=use_anc)
            for w, h in zip(keep, hs): HISTS[f"{w}|{bank}|{flav}"] = h
            np.savez_compressed(HIST_FP, **HISTS)
            print(f"      done {time.time()-t1:.0f}s (cache saved)", flush=True)

    # ---------------- v50B 事後 P̄(ch23) + EV23 組み立て ----------------
    # ★train井戸のch(v50B事後)は必ずOOF(自foldデコーダ1本のみ)。5本平均は残り4本が
    #   その井戸を学習済み=正解リークになる(v52 v50b_feature と同じ流儀。testのみ5本平均可)。
    print("[assemble] v50B ch(OOF) + EV23 ...", flush=True)
    offs_t = torch.arange(-sa.NB, sa.NB + 1, device=DEV).float() * sa.STEP
    DEC = []
    for f in range(5):
        net = sb.DecoderB().to(DEV)
        net.load_state_dict(torch.load(sb.ART / f"stageB_dec{sb.BTAG}_f{f}.pt", map_location=DEV))
        net.eval(); DEC.append(net)
    FOLD_OF = {w: f for f in range(5) for w in sb.WELLS[f::5]}
    OUT = {}
    with torch.no_grad():
        for wi, wn in enumerate(wells):
            if wn not in sb.EVD or wn not in FOLD_OF: continue
            e = sb.EVD[wn]; nb_ = e["EV"].shape[1]
            x, m, _, _ = sb.make_batch([wn])
            lg, rf = DEC[FOLD_OF[wn]](x, m)
            Pbar = torch.softmax(lg[0, :nb_], dim=-1).cpu().numpy().astype(np.float16)
            ch = [e["EV"].astype(np.float16)]                        # (10,nb,181)
            hs = []
            okall = True
            for bank in BANKS:
                for flav in FLAVORS:
                    h = HISTS.get(f"{wn}|{bank}|{flav}")
                    if h is None or len(h) != nb_: okall = False; break
                    hs.append(h[None].astype(np.float16) * 30.0)      # γと同様×30でfp16域へ
                if not okall: break
            if not okall: continue
            EV23 = np.concatenate(ch + hs + [Pbar[None] * 30.0], 0)   # (23,nb,181)
            OUT[wn] = dict(EV=EV23.astype(np.float16), dtrue=e["dtrue"], wgt=e["wgt"],
                           st=e["st"], bid=e["bid"], y_ev=e["y_ev"])
            if (wi + 1) % 150 == 0: print(f"  assemble {wi+1}/{len(wells)}", flush=True)
    (ART53 / "evidenceC_v53.pkl").write_bytes(pickle.dumps(OUT, protocol=4))
    print(f"saved evidenceC_v53.pkl: {len(OUT)} wells  ({time.time()-t0:.0f}s total)")
