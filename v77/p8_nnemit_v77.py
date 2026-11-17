# -*- coding: utf-8 -*-
"""
v77 P8 — K16帯のNN emission(学習類似)を P5 trellis に注入
==========================================================
学習: (well GR窓33行, typewell GR窓±16ft, スカラー3) → 帯内81オフセットのsoft-CE。
     井戸5fold(決定的: ソート順%5)。OOF厳守=適用は自foldモデル以外(=fold外モデル)。
適用: P5のGR emissionに W_NN·lognn を加算(または置換)して復号A/B。
使い方: MODE=train python v77/p8_nnemit_v77.py
        MODE=apply NW=60 W_NN=1.0 python v77/p8_nnemit_v77.py
struct(v38/v66)不使用。帯中心=K16錨(許可)。
"""
import os, io, sys, glob, time, pickle
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
TRAIN = P0 / "rogii-wellbore-geology-prediction" / "train"
ART = P0 / "v77" / "artifacts_v77"; ART.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = os.environ.get("MODE", "train")
NW = int(os.environ.get("NW", "60"))
BAND = 40.0; STEP = 1.0
NOFF = int(2 * BAND / STEP) + 1                     # 81
WW = 33                                             # well GR窓(行)
TWW = 33                                            # typewell窓(±16ft, 1ft刻み)
EPOCHS = int(os.environ.get("EPOCHS", "3"))
NFOLD = 5
W_NN = float(os.environ.get("W_NN", "1.0"))
REPL = os.environ.get("REPL", "0")                  # 1=GR emission置換, 0=加算
ANC = pickle.loads((ART / "k16_anchor_train.pkl").read_bytes())
offs_np = np.arange(-BAND, BAND + STEP / 2, STEP)


def wid_fold(wid, wids_sorted):
    return wids_sorted.index(wid) % NFOLD


class Emit(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WW + TWW + 3, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):                            # (B,NOFF,F) → (B,NOFF)
        return self.net(x).squeeze(-1)


def build_well(wid, need_true=True):
    """返り値: dict(feat遅延構築用の配列一式) or None"""
    if wid not in ANC: return None
    hw = pd.read_csv(TRAIN / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(TRAIN / f"{wid}__typewell.csv").sort_values("TVT")
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) < 60 or len(kn) < 20: return None
    tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
    m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
    if len(tw_tvt) < 8: return None
    gr = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    epos = hw.index.get_indexer(ev.index)
    a = ANC[wid]
    T = min(len(ev), len(a["tvt"]))
    A = a["tvt"][:T]; sdd = a["sig"][:T]; dd = a["dd"][:T]
    ge = gr[epos][:T]
    gmu, gsd = float(np.nanmean(gr)), float(np.nanstd(gr) + 1e-6)
    tmu, tsd = float(np.nanmean(tw_gr)), float(np.nanstd(tw_gr) + 1e-6)
    tv = ev["TVT"].to_numpy(float)[:T] if need_true else None
    return dict(wid=wid, A=A, sdd=sdd, dd=dd, ge=(ge - gmu) / gsd,
                tw_tvt=tw_tvt, tw_gr=(tw_gr - tmu) / tsd, true=tv)


def feats(c, rows):
    """rows(index配列) → (R,NOFF,F) tensor"""
    R = len(rows)
    ge = torch.tensor(c["ge"], device=DEV, dtype=torch.float32)
    T = len(c["ge"])
    # well窓 (R,WW)
    wi = torch.tensor(rows, device=DEV)[:, None] + torch.arange(-(WW // 2), WW // 2 + 1, device=DEV)[None, :]
    wwin = ge[wi.clamp(0, T - 1)]
    # tw窓 (R,NOFF,TWW): A[r]+off+grid
    At = torch.tensor(c["A"], device=DEV, dtype=torch.float32)[torch.tensor(rows, device=DEV)]
    offs = torch.tensor(offs_np, device=DEV, dtype=torch.float32)
    grid = torch.arange(-(TWW // 2), TWW // 2 + 1, device=DEV, dtype=torch.float32)
    q = At[:, None, None] + offs[None, :, None] + grid[None, None, :]     # (R,NOFF,TWW)
    ttv = torch.tensor(c["tw_tvt"], device=DEV, dtype=torch.float32)
    tgr = torch.tensor(c["tw_gr"], device=DEV, dtype=torch.float32)
    ii = torch.searchsorted(ttv, q.clamp(ttv[0], ttv[-1]).contiguous()).clamp(1, len(ttv) - 1)
    wv = ((q - ttv[ii - 1]) / (ttv[ii] - ttv[ii - 1]).clamp_min(1e-6)).clamp(0, 1)
    twin = tgr[ii - 1] * (1 - wv) + tgr[ii] * wv
    sdd = torch.tensor(c["sdd"], device=DEV, dtype=torch.float32)[torch.tensor(rows, device=DEV)]
    ddt = torch.tensor(c["dd"].astype(np.float32), device=DEV)[torch.tensor(rows, device=DEV)]
    sc = torch.stack([offs[None, :].expand(R, -1) / 40.0,
                      (sdd[:, None] / 15.0).expand(-1, NOFF),
                      (ddt[:, None] / 2000.0).expand(-1, NOFF)], dim=-1)  # (R,NOFF,3)
    x = torch.cat([wwin[:, None, :].expand(-1, NOFF, -1), twin, sc], dim=-1)
    return x


def train():
    wids = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids = [w for w in wids if w in ANC]
    t0 = time.time()
    for f in range(NFOLD):
        torch.manual_seed(777 + f)
        model = Emit().to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trw = [w for w in wids if wids.index(w) % NFOLD != f]
        rng = np.random.default_rng(777 + f)
        for ep in range(EPOCHS):
            rng.shuffle(trw)
            tot = n = 0
            for wid in trw:
                c = build_well(wid)
                if c is None: continue
                off_true = c["true"] - c["A"]                     # 真オフセット
                ok = np.where(np.isfinite(off_true) & (np.abs(off_true) < BAND - 2))[0]
                if len(ok) < 32: continue
                rows = rng.choice(ok, size=min(192, len(ok)), replace=False)
                x = feats(c, rows)
                logit = model(x)
                ot = torch.tensor(off_true[rows], device=DEV, dtype=torch.float32)
                tgt = torch.exp(-0.5 * ((torch.tensor(offs_np, device=DEV, dtype=torch.float32)[None, :]
                                          - ot[:, None]) / 2.0) ** 2)
                tgt = tgt / tgt.sum(1, keepdim=True)
                loss = -(tgt * torch.log_softmax(logit, 1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); n += 1
            print(f"fold{f} ep{ep} loss {tot/max(n,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), ART / f"p8_emit_f{f}.pt")
    print(f"[train done] {time.time()-t0:.0f}s")


@torch.no_grad()
def nn_loge(c, model):
    T = len(c["ge"])
    outs = []
    for st in range(0, T, 512):
        rows = np.arange(st, min(st + 512, T))
        x = feats(c, rows)
        outs.append(torch.log_softmax(model(x), 1))
    return torch.cat(outs, 0)                        # (T,NOFF) log域


def apply():
    # P5復号にNN emissionを注入(このファイル内に最小P5を内蔵せず、P5のemission部だけ再現)
    import importlib.util
    spec = importlib.util.spec_from_file_location("p5", P0 / "v77" / "p5_trellis_v77.py")
    # → P5をimportすると即実行部は無いのでOK(main()ガード済)
    p5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p5)
    wids_all = sorted({os.path.basename(p).split("__")[0] for p in glob.glob(str(TRAIN / "*__horizontal_well.csv"))})
    wids_all = [w for w in wids_all if w in ANC]
    models = []
    for f in range(NFOLD):
        m = Emit().to(DEV); m.load_state_dict(torch.load(ART / f"p8_emit_f{f}.pt", map_location=DEV))
        m.eval(); models.append(m)
    wids = wids_all[:NW]
    es, ea, per, outs = [], [], {}, {}
    t0 = time.time()
    orig_run = p5.run_well
    for i, wid in enumerate(wids):
        c = build_well(wid)
        if c is None: continue
        f = wids_all.index(wid) % NFOLD
        lognn = nn_loge(c, models[f])                # (T,81) 自foldモデル=fold外学習(OOF)
        # P5のrun_wellを直接使わず、emission注入版をここで実行
        r = run_p5_nn(p5, wid, lognn)
        if r is None: continue
        m = np.isfinite(r["err"])
        es.append(r["err"][m]); ea.append(r["anc_err"][m])
        per[wid] = float(np.sqrt(np.mean(r["err"][m] ** 2))); outs[wid] = r
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wids)} 暫定 {np.sqrt(np.mean(np.concatenate(es)**2)):.3f} ({time.time()-t0:.0f}s)", flush=True)
    e = np.concatenate(es); a = np.concatenate(ea)
    print(f"\n=== P8 NN-emit → P5 ({len(per)}井) W_NN={W_NN} REPL={REPL} ===")
    print(f"  P8 pooled = {np.sqrt(np.mean(e**2)):.4f}   (K16錨単体 = {np.sqrt(np.mean(a**2)):.4f})")
    fp = ART / f"p8_out_wnn{W_NN:g}r{REPL}_{len(per)}w.pkl"
    fp.write_bytes(pickle.dumps(outs, protocol=4))
    print(f"[saved] {fp.name}  total {time.time()-t0:.0f}s")


def run_p5_nn(p5, wid, lognn):
    """p5.run_wellのemission行にlognnを合成した再実装(コピー最小)。"""
    import types
    torch_ = torch
    c = build_well(wid)
    if c is None: return None
    # p5の元関数を一時差し替えでなく、emission合成用のフックが無いので、p5.run_wellを再現:
    # → p5.run_wellはANC/typewell読み込み込み。ここでは p5 のグローバルにフックを置く。
    p5._NN_HOOK = (wid, lognn, W_NN, REPL)
    try:
        return p5.run_well(wid)
    finally:
        p5._NN_HOOK = None


if __name__ == "__main__":
    if MODE == "train": train()
    else: apply()
