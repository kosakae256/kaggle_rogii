"""
v20 train.parquet ビルダー ―― 物理ゲートPF × 複数本(バンク) を v10流の特徴→GBM に
==================================================================================

v10 を踏襲(各PFセットの pf_* 特徴を v9 build_well で生成 → GBMで学習)。
違い:
  - PFを **物理ゲートあり(lk_phys)** に置換(v19 と同じ物理項を _pf_core に追加)。
  - PFセット = **v19バンク(pf_bank.json) + v9(pf_1/2/3) の最大7本**。
  - **どのPFを使うかを SELECT_PFS で選択可**(ラベルで指定, None=全部)。
  - 順方向のみ(逆方向 bpf_/cmp_ は作らない)。

仕組み(v10と同じ):
  - 非PF特徴は v9 build_well を exec 再利用。
  - 各PFセットの順方向GPUアンサンブル(物理PF)を build_well の run_pf_ancc に注入 → pf_*{suffix} 特徴。
  - PFパラメータは v9 の _make_pf_params で raw→p に変換し、物理キー(PF_USE_PHYS/PF_PHYS_SIG)を付与。

使い方: rogii_claude/.venv/Scripts/python.exe v20/create_train_parquet_v20.py
"""

import os, time, glob, json, threading
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_TLS = threading.local()
_FWD_ENS = {}   # (wid, pf_name) -> (pf_ancc, pf_ancc_std)

def _patched_run_pf_ancc(hw, tw_tvt, tw_gr, p):
    r = _FWD_ENS.get((getattr(_TLS, "wid", None), p.get("name")))
    if r is None:
        return np.array([], np.float32), np.array([], np.float32)
    return r[0].astype(np.float32), r[1].astype(np.float32)

# ============================== CONFIG ==============================
PROJECT_DIR = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR    = PROJECT_DIR / "rogii-wellbore-geology-prediction"
V9_NB       = PROJECT_DIR / "v9" / "pf_tuning_create_train_csv_multi_pf_v9.ipynb"
BANK_PATH   = PROJECT_DIR / "v19" / "artifacts_v19" / "pf_bank.json"
OUT_PARQUET = PROJECT_DIR / "v20" / "artifacts_v20_3" / "data" / "train.parquet"

INCLUDE_V9  = False       # v9 pf_1/2/3(物理OFF)もPFセットに含める
SELECT_PFS  = None       # 使うPFのラベル list(None=全部)。例: ["r0","r2","v9_pf1","v9_pf3"]
                         #   ラベル: v19バンク= r{round} (r0..), v9= v9_pf1/2/3
N_WELLS     = None       # None=全坑井
FWD_SEEDS   = 32         # 順方向シード数。※n_seed感度に注意(バンクは8で調整)。train/testで必ず同じ値に。
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float32
SEED        = 0
WELL_CHUNK  = 80        # 順方向のみなので大きめOK(VRAMで調整)
N_JOBS_CPU  = 4
# ===================================================================


def selected_raws(ns):
    """使うPFの (label, raw_params) を返す。v19バンク + (任意で)v9 pf_1/2/3。"""
    items = []
    bank = json.loads(BANK_PATH.read_text())
    for e in bank:
        items.append((f"r{e['round']}", e["params"]))
    if INCLUDE_V9:
        for k, raw in enumerate(ns["PF_PARAM_SETS_RAW"][:3]):
            r = dict(raw); r["use_phys"] = False; r.setdefault("phys_sig", 1.0)
            items.append((f"v9_pf{k+1}", r))
    if SELECT_PFS is not None:
        sel = set(SELECT_PFS)
        items = [(lab, r) for lab, r in items if lab in sel]
    if not items:
        raise RuntimeError("SELECT_PFS に該当するPFがありません。")
    return items


def load_v9_namespace():
    nb = json.load(open(V9_NB, encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    src = src.replace(
        r'dataset_path = Path(r"C:\Users\kosaka256\Documents\rogii\rogii-wellbore-geology-prediction")',
        f'dataset_path = Path(r"{DATA_DIR}")',
    )
    src = src.replace("cache=True", "cache=False")
    marker = 'train_paths = sorted((CFG.dataset_path / "train").glob'
    cut = src.find(marker)
    if cut > 0:
        src = src[:cut]
    ns = {"__name__": "v9lib"}
    print("[v9] exec で v9 定義を読み込み中(FI/DI の fit を含む)...")
    exec(compile(src, str(V9_NB), "exec"), ns)

    # PFセット = 選択した物理バンク。raw→p は v9 _make_pf_params, 物理キーを付与。
    items = selected_raws(ns)
    sets = []
    for k, (lab, raw) in enumerate(items):
        r = dict(raw); r["n_seeds"] = FWD_SEEDS
        p = ns["_make_pf_params"](r, k)              # name=pf_{k+1}, suffix=_{k+1}
        p["PF_USE_PHYS"] = bool(raw.get("use_phys", True))
        p["PF_PHYS_SIG"] = float(raw.get("phys_sig", 1.0))
        p["label"] = lab
        sets.append(p)
    ns["PF_PARAM_SETS"] = sets
    print(f"[v20] PFセット {len(sets)}本: " +
          ", ".join(f"{p['label']}->{p['name']}{p['suffix']}(phys={'on' if p['PF_USE_PHYS'] else 'off'})" for p in sets))
    return ns


def build_pf_inputs(hw, tw_tvt, tw_gr, p, V9):
    _smooth = V9["_smooth_radius_values"]; _grid = V9["_grid"]; _gr_sig = V9["_gr_sig"]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs = _gr_sig(hw, tw_tvt, tw_gr_pf, p)
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    md_ev = ev["MD"].values.astype(np.float64); z_ev = ev["Z"].values.astype(np.float64)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_proc = _smooth(gr_full, tw_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos = hw.index.get_indexer(ev.index)
    gr_ev = gr_proc[ev_pos].astype(np.float64)
    return dict(md=md_ev, z=z_ev, gr=gr_ev, gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(p["ANCC_N"]))


def _pf_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, p, N, device, gen, dtype=DTYPE):
    """順方向GPU PF(物理ゲートあり)。返り値 pts(B,T)重み付き平均TVT, log_lik(B,)。"""
    B, T = md.shape
    ALPHA = p["ANCC_ALPHA"]; RN = p["ANCC_RN"]; PN = p["ANCC_PN"]; IR = p["ANCC_IR"]; IS = p["ANCC_IS"]
    RP = p["ANCC_RP"]; RR = p["ANCC_RR"]; RESAMP = p["PF_RESAMP"]; RATE_CLIP = p["PF_RATE_CLIP"]
    GR_POWER = p["PF_GR_POWER"]; JUMP_PROB = p["PF_JUMP_PROB"]; JUMP_STD = p["PF_JUMP_STD"]; CLIP = p["PF_TVT_CLIP_MARGIN"]
    USE_P = bool(p.get("PF_USE_PHYS", False)); PHYS_SIG = float(p.get("PF_PHYS_SIG", 1.0))

    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)

    pos = ls[:, None] + IS * rn()
    rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP
    tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1))
    g0col = grid[:, 0:1]
    pm = md[:, 0] - 1.0
    arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    dipp = ir[:, None]                    # dip_prior(=既知ゾーン回帰)

    for i in range(T):
        act = valid[:, i]
        cur = md[:, i]
        dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0:
            rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]
        tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None])
        pos_n = tvt + zi
        am = act[:, None]
        pos = torch.where(am, pos_n, pos)
        rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]
        obs = act & ~torch.isnan(gri)
        v = pos - zi
        ii = (v - vmin[:, None]) / step[:, None]
        i0 = torch.floor(ii).long()
        below = i0 < 0
        above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0)
        i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype)
        gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t
        eg = torch.where(below, g0col, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs()
        dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg_lk = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg_lk.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        # 物理尤度(dipが既知dipに近いほど高い)。GR観測の有無に関わらず適用。
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG
            lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        w_new = torch.where(obs[:, None], w * lk * lkp, w * lkp)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1)
        need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1)
            u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N
            idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn()
            rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0:
                rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]
            pos = torch.where(nm, pos_rs, pos)
            rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
        pts[:, i] = (w * (pos - zi)).sum(1)
        pm = torch.where(act, cur, pm)
    return pts, log_lik


def _pad_wells(inps, device, dtype=DTYPE):
    W = len(inps)
    Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    for b, x in enumerate(inps):
        T = len(x["md"]); G = len(x["gg"])
        md[b, :T] = torch.from_numpy(x["md"].astype("float32"))
        if T < Tmax: md[b, T:] = md[b, T - 1]
        z[b, :T] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :T] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :T] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
    to = lambda tt: tt.to(device)
    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir))


def run_forward(inps, p, S, device, seed, chunk):
    """各坑井: Sシードの尤度加重アンサンブル (pf_ancc, pf_ancc_std) を返す。"""
    N = int(p["ANCC_N"]); W = len(inps); ens_list = [None] * W
    for c in range(0, W, chunk):
        sub = inps[c:c + chunk]; w = len(sub)
        pad = _pad_wells(sub, device)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik = _pf_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            p, N, device, gen)
        pts = pts.view(w, S, -1).double().cpu().numpy()
        ll = log_lik.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); pr = pts[j, :, :T]
            lk = ll[j] - np.nanmax(ll[j]) if np.isfinite(ll[j]).any() else np.zeros_like(ll[j])
            ww = np.exp(np.where(np.isfinite(lk), lk, -np.inf) / max(float(p["PF_LIKELIHOOD_SCALE"]), 1e-6))
            ww = ww / ww.sum() if ww.sum() > 1e-300 else np.full(S, 1.0 / S)
            mean = (ww[:, None] * pr).sum(0).astype(np.float32)
            var = (ww[:, None] * (pr - mean[None, :]) ** 2).sum(0)
            std = np.sqrt(np.maximum(var, 0.)).astype(np.float32)
            ens_list[c + j] = (mean, std)
    return ens_list


def build_v20_dataset(V9, split="train", is_train=True, n_wells=None):
    """v20 特徴(非PF=v9 / 各物理PFの順方向アンサンブルを注入→pf_*{suffix})の DataFrame を返す。"""
    build_well = V9["build_well"]; P_SETS = V9["PF_PARAM_SETS"]
    base = DATA_DIR / split
    wells = sorted({os.path.basename(q).split("__")[0]
                    for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if n_wells is not None:
        wells = wells[:n_wells]

    print(f"[{split}] 有効坑井を確定中 ({len(wells)} 坑井)...")
    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if is_train and "TVT" not in hw.columns:
            continue
        if build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P_SETS[0], V9) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[{split}] 使用坑井: {len(names)}")

    _FWD_ENS.clear()
    for p in P_SETS:
        print(f"[gpu] {p['label']}({p['name']}): 入力作成 + 物理PF(順方向) ...")
        t0 = time.perf_counter()
        inps = []
        for (hp, tp) in paths:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            inps.append(build_pf_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), p, V9))
        fwd_ens = run_forward(inps, p, FWD_SEEDS, DEVICE, SEED, WELL_CHUNK)
        for i, w in enumerate(names):
            _FWD_ENS[(w, p["name"])] = fwd_ens[i]
        if DEVICE == "cuda": torch.cuda.synchronize()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    V9["run_pf_ancc"] = _patched_run_pf_ancc

    print("[cpu] build_well(非PF=v9 / 物理PF=GPU注入)を結合...")
    from joblib import Parallel, delayed
    def one(w, hp, tp):
        _TLS.wid = w
        df = build_well(str(hp), str(tp), is_train)
        return df if (df is not None and len(df) > 0) else None
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS_CPU, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1]) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s,  parts={len(parts)}")
    if not parts:
        raise RuntimeError(f"有効な坑井がありません ({split})。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"device={DEVICE}  fwd_seeds={FWD_SEEDS}  include_v9={INCLUDE_V9}  select={SELECT_PFS}  chunk={WELL_CHUNK}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    V9 = load_v9_namespace()
    train = build_v20_dataset(V9, split="train", is_train=True, n_wells=N_WELLS)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_PARQUET, index=False)
    cols = list(train.columns)
    n_pf = sum(c.startswith("pf_") for c in cols)
    print(f"\nsaved: {OUT_PARQUET}")
    print(f"rows={len(train)}  cols={len(cols)}  (pf_={n_pf})")


if __name__ == "__main__":
    main()
