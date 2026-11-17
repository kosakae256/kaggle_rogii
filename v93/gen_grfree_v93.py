import io, sys, os, pickle, json
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
# -*- coding: utf-8 -*-
"""GRフリー自己完結 v3 = v92/gen_grfree_anchor.py の忠実再現 + fit/predict モード分離。

■ このモデル(v92 GRフリー v2 錨)の中身
   phaseG dip-field エンジン(GR/struct 一切不使用, X,Y,Z,TVT,TVT_input のみ) + GRU 残差。
   ★val 完全除外の正しい 5-fold。fold 毎に [場F・κ・近傍プール・GRU] の全てを
   train fold の井だけで構築 -> val fold 予測。完全ランダム shuffle 分割。
   test = 5fold x 3seed = 15 セット(各 fold の F/κ/GRU)平均。

■ v92 との違い = MODE 分離だけ(数値ロジックは verbatim)
   v92 は 1 実行の中で「5-fold 学習 -> そのまま FOLD_ART で test/train 錨生成」していた。
   つまり test 用/Kaggle 用に毎回 15 個の GRU を再学習していた。
   v3 では学習(fit)と推論(predict)を分け、学習済み 15 モデル + 5 fold エンジンを
   ディスクへ保存 -> predict はそれを load するだけ(再学習なし)。
   -> train で 1 度だけ学習し、test/Kaggle 提出で使い回す。

   GRF_MODE=fit     : 5-fold 学習 -> OOF CV を表示(忠実性ゲート, ~10.03 のはず)
                      -> 15 GRU(.pt) + 5 fold エンジン(.pkl) + meta.json を保存。
                      既存 grfree_anchor_train.pkl は一切触らない(検証済み train 錨)。
   GRF_MODE=predict : 保存済みモデルを load(再学習なし)-> GRF_SPLIT(test|train)の井を
                      v92 の test-anchor 節と同一手順で予測 -> 錨 pkl を保存。
                      ★sig は定数 9.0 に上書き(bug-fix 版 grfree_anchor_train.pkl と一致)。

   struct/GR は一切読まない。Kaggle でも動く(ROOT 自動検出)。
"""
import os, glob, time, io, sys
import numpy as np, pandas as pd
t00 = time.time()

# ===== パス / Kaggle 検出(v92 と同じ ROOT 検出) =====
KAGGLE = os.path.exists('/kaggle/input')
ROOT = glob.glob('/kaggle/input/*rogii*wellbore*')[0] if KAGGLE else r'c:/Users/kosaka256/Documents/rogii_claude/rogii-wellbore-geology-prediction'
TRAIN_DIR = os.path.join(ROOT, 'train'); TEST_DIR = os.path.join(ROOT, 'test'); SAMPLE = os.path.join(ROOT, 'sample_submission.csv')

# 成果物ルート(local 固定 / Kaggle では GRF_MODELS_DIR で上書き可能)
_LOCAL_ART = r'c:/Users/kosaka256/Documents/rogii_claude/v93/artifacts_v93'
ART = os.environ.get('GRF_ART_DIR', _LOCAL_ART)
MODELS_DIR = os.environ.get('GRF_MODELS_DIR', os.path.join(ART, 'grfree_models'))

# ===== 定数(v92 verbatim) =====
PGATE = 0.35; THETA0 = 118.4; KNN_K, KNN_H = 15, 500.0
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]; KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1; CMAX = 0.30; K_FIX = 16; CLIP_DRIFT = 150.0
N_FOLDS = 5; NBK = 3; STRIDE = 2; HID = 96; C = 37
FT_EP = int(os.environ.get('FT_EP', '10')); FT_LR = 1e-3; SEEDS = int(os.environ.get('SEEDS', '3'))
SPLIT_SEED = int(os.environ.get('SPLIT_SEED', '42'))
WINS = [301, 1001, 2001, 4001]; _C0, _S0 = np.cos(np.radians(THETA0)), np.sin(np.radians(THETA0))
def lowpass(x, win): return pd.Series(x).rolling(int(win) | 1, center=True, min_periods=1).mean().to_numpy()

# ===== エンジン(phaseG lean, GR不使用) =====
def make_gdict(X, Y, Z, ti, tvt=None):
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) == 0: return None
    s = int(fin.max()); n = len(X) - 1 - s
    if n < 2: return None
    dX = np.diff(X)[s:]; dY = np.diff(Y)[s:]; Lxy = np.maximum(np.hypot(dX, dY), 1e-3)
    p_row = (dX * _C0 + dY * _S0) / Lxy
    d = dict(s=s, n=n, dX=dX, dY=dY, ndz=-np.diff(Z)[s:], Lxy=Lxy, arc=np.cumsum(Lxy),
             Xl=X[s:], Yl=Y[s:], anchor=float(ti[s]), perp=np.abs(p_row) < PGATE)
    if tvt is not None: d['R0'] = tvt[s + 1:] - tvt[s]
    return d
def segment_well(wd, with_slopes):
    n, A = wd['n'], wd['arc']; K = min(K_FIX, max(1, n // 4)); total = float(A[-1])
    edges = np.linspace(0.0, total, K + 1); segid = np.clip(np.searchsorted(edges[1:], A, side='left'), 0, K - 1)
    mid = np.empty((K, 2)); az = np.empty(K); Xl, Yl = wd['Xl'], wd['Yl']
    for j in range(K):
        rows = np.where(segid == j)[0]; j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)
        az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0]); mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)
    if not with_slopes: return segid, mid, az, None, None
    phi = np.column_stack([np.clip(A - edges[j], 0.0, edges[j + 1] - edges[j]) for j in range(K)])
    y = wd['R0'] - np.cumsum(wd['ndz']); m = np.isfinite(y)
    if m.sum() < K + 2: return segid, mid, az, None, None
    c, res, rank, _ = np.linalg.lstsq(phi[m], y[m], rcond=None); dof = max(int(m.sum()) - K, 1)
    ssr = float(res[0]) if len(res) else float(np.sum((phi[m] @ c - y[m]) ** 2))
    cov = (ssr / dof) * np.linalg.pinv(phi[m].T @ phi[m]); se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return segid, mid, az, c, se
def query_well(F, mids, own, mdbuf=0.0):
    keep = F['wi'] != own; fx, fy = F['x'][keep], F['y'][keep]; fc = F['c'][keep]; fp0 = F['p0'][keep]; fqw = F['qw'][keep]
    dx = fx[None, :] - mids[:, 0:1]; dy = fy[None, :] - mids[:, 1:2]; d2 = dx * dx + dy * dy
    if mdbuf > 0: d2 = np.where(d2 >= mdbuf * mdbuf, d2, np.inf)
    kk = min(KNN_K, d2.shape[1] - 1) if d2.shape[1] > 1 else 1
    idx = np.argpartition(d2, kk - 1, axis=1)[:, :kk]; r = np.arange(len(mids))[:, None]
    d2s = d2[r, idx]; c_s, p0_s = fc[idx], fp0[idx]
    w = np.exp(np.maximum(-d2s / (2 * KNN_H ** 2), -700.0)) * fqw[idx]; w[~np.isfinite(d2s)] = 0.0
    dead = w.sum(1) <= 1e-300
    with np.errstate(invalid='ignore'): dd = np.sqrt(np.nanmedian(np.where(np.isfinite(d2s), d2s, np.nan), axis=1))
    dd[~np.isfinite(dd)] = KBINS[-2] * 2
    s_hat = (w * c_s * p0_s).sum(1) / ((w * p0_s ** 2).sum(1) + 1e-9); g = np.column_stack([s_hat * _C0, s_hat * _S0]); g[dead] = 0.0
    gn = np.hypot(g[:, 0], g[:, 1]); big = gn > CMAX
    if big.any(): g[big] *= (CMAX / gn[big])[:, None]
    pred_don = g[:, 0:1] * np.cos(F['az'][keep][idx]) + g[:, 1:2] * np.sin(F['az'][keep][idx])
    spread = np.sqrt(np.maximum((w * (c_s - pred_don) ** 2).sum(1) / np.maximum(w.sum(1), 1e-12), 0.0)); spread[dead] = 0.0
    return g, dd, spread
def design_blocks(F, wd, mdbuf):
    g, dd, _ = query_well(F, wd['seg'][1], wd['wi'], mdbuf=mdbuf); sid = wd['seg'][0]
    f = g[sid, 0] * wd['dX'] + g[sid, 1] * wd['dY']; a = wd['ndz']; par = ~wd['perp']
    b_row = np.digitize(dd, KBINS[1:-1])[sid]; cols = []
    for b in range(NBIN):
        m = (b_row == b) & par; cols.append(np.cumsum(np.where(m, a, 0.0))); cols.append(np.cumsum(np.where(m, f, 0.0)))
    cols.append(np.cumsum(np.where(wd['perp'], a, 0.0))); cols.append(np.cumsum(np.where(wd['perp'], f, 0.0)))
    G = np.column_stack(cols); ok = np.isfinite(wd['R0']); G = np.nan_to_num(G[ok]); return G.T @ G, G.T @ wd['R0'][ok]
def solve_kappa(A, yv):
    npar = 2 * NBIN; ncol = npar + 2; A = A.copy(); lam_p = 0.02 * float(np.trace(A)) / ncol
    A[npar, npar] += lam_p; A[npar + 1, npar + 1] += lam_p; coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    return dict(alpha=np.clip(coef[0:npar:2], -0.25, 1.5), beta=np.clip(coef[1:npar:2], -0.25, 1.5),
                ap=float(np.clip(coef[npar], -0.25, 1.5)), bp=float(np.clip(coef[npar + 1], -0.25, 1.5)))
def engine_predict(F, wd, K, own=-1):
    g, dd, spread = query_well(F, wd['seg'][1], own, mdbuf=0.0); sid = wd['seg'][0]
    f = g[sid, 0] * wd['dX'] + g[sid, 1] * wd['dY']; b_row = np.digitize(dd, KBINS[1:-1])[sid]
    av = K['alpha'][b_row]; bv = K['beta'][b_row]; av = np.where(wd['perp'], K['ap'], av); bv = np.where(wd['perp'], K['bp'], bv)
    rate = av * wd['ndz'] + bv * f; resid = np.clip(np.cumsum(rate), -CLIP_DRIFT, CLIP_DRIFT)
    return wd['anchor'] + resid, rate, dd[sid], spread[sid]

def build_field(wells):
    px, py, pc, paz, pwi = [], [], [], [], []
    for wd in wells:
        if wd['seg'][3] is None: continue
        segid, mid, az, c, se = wd['seg']; ok = np.isfinite(c) & (np.abs(c) <= CMAX) & np.isfinite(se)
        px.append(mid[ok, 0]); py.append(mid[ok, 1]); pc.append(c[ok]); paz.append(az[ok]); pwi.append(np.full(int(ok.sum()), wd['wi'], float))
    F = dict(x=np.concatenate(px), y=np.concatenate(py), c=np.concatenate(pc), az=np.concatenate(paz), wi=np.concatenate(pwi))
    F['qw'] = np.ones(len(F['x'])); F['p0'] = np.cos(F['az']) * _C0 + np.sin(F['az']) * _S0
    return F
def fit_kappa(F, wells):
    A = 0; yv = 0
    for wd in wells:
        if wd['seg'][3] is None: continue
        for R in KAPPA_REGIMES:
            Ab, yb = design_blocks(F, wd, R); A = A + Ab; yv = yv + yb
    return solve_kappa(A, yv)
def build_pool(wells):
    ids = [wd['wid'] for wd in wells]
    cent = np.array([[np.nanmean(wd['_X']), np.nanmean(wd['_Y'])] for wd in wells])
    from scipy.spatial import cKDTree
    trees = {wd['wid']: cKDTree(np.c_[wd['_X'], wd['_Y']][::2]) for wd in wells}
    gks = {wd['wid']: lowpass((wd['_tvt'] + wd['_Z']).astype(float), 201) for wd in wells}
    mdc = {wd['wid']: np.arange(len(wd['_X']), dtype=float) for wd in wells}
    yx = {wd['wid']: (wd['_X'].astype(float), wd['_Y'].astype(float)) for wd in wells}
    return dict(ids=ids, cent=cent, trees=trees, gks=gks, md=mdc, yx=yx)

def build_channels(md, x, y, z, tin, t0, d7_drift, rates_full, conf_full, POOL, self_wid=None):
    n = len(md); ei = np.arange(t0 + 1, n)[::STRIDE].astype(int); tvt_last = float(tin[t0])
    d7 = d7_drift[(ei - (t0 + 1))]; rates = rates_full[ei]; conf = conf_full[ei]
    g = np.where(np.isfinite(tin), tin, np.nan) + z; ki = np.where(np.isfinite(tin))[0]
    zh301 = z - lowpass(z, 301); tl = ki[-min(len(ki), 1000):]; gt = (g - lowpass(g, 301))[tl]; zt = zh301[tl]
    fin = np.isfinite(gt); vz = float(np.dot(zt[fin], zt[fin])); k_pref = float(np.clip(np.dot(zt[fin], gt[fin]) / vz, 0, 1.5)) if vz > 1e-9 else 0.0
    az = np.arctan2(np.gradient(y), np.gradient(x)); az_s = np.arctan2(lowpass(np.sin(az), 301), lowpass(np.cos(az), 301))
    bands = {W: lowpass(z, W) for W in WINS}; msb = (np.arange(n) - t0).astype(float)
    dh = np.hypot(np.gradient(lowpass(x, 101)), np.gradient(lowpass(y, 101))) + 1e-9
    incl = np.arctan2(np.gradient(lowpass(z, 301)), dh); build_r = np.gradient(lowpass(incl, 201)) * 1000
    g_s = lowpass(g, 201); r_all = np.gradient(np.nan_to_num(g_s, nan=0.0)) / np.maximum(dh, 1e-6)
    tlp = ki[(ki >= t0 - 1500)]; rvals = r_all[tlp]; rvals = rvals[np.isfinite(g_s[tlp])]
    r_pref = float(np.clip(np.median(rvals), -0.08, 0.08)) if len(rvals) > 100 else 0.0
    dprior = (r_pref * (md - md[t0]) - (z - z[t0])); kn_flag = np.zeros(len(ei), np.float32)
    ch = [d7 / 20.0, rates * 20.0, conf, kn_flag, msb[ei] / 3000.0, np.sin(az_s[ei]), np.cos(az_s[ei]),
          np.full(len(ei), k_pref), incl[ei] * 3, build_r[ei], (incl[ei] - incl[t0]) * 3, np.clip(dprior[ei], -120, 120) / 30.0]
    for W in WINS: ch.append((bands[W][ei] - bands[W][t0]) / 50.0); ch.append((z - bands[W])[ei] / 10.0)
    cent = POOL['cent']; ids = POOL['ids']
    dd0 = np.hypot(cent[:, 0] - np.nanmean(x), cent[:, 1] - np.nanmean(y)); order = np.argsort(dd0); nbrs = []
    for j in order:
        if self_wid is not None and ids[j] == self_wid: continue
        nbrs.append(ids[j])
        if len(nbrs) == NBK: break
    p0 = np.array([x[t0], y[t0]]); g0 = float(tin[t0] + z[t0])
    for nw in nbrs:
        tr = POOL['trees'][nw]; gks = POOL['gks'][nw]; mdn = POOL['md'][nw]; nx, ny = POOL['yx'][nw]
        _, j0 = tr.query(p0); j0 = int(j0) * 2
        dq, jj = tr.query(np.c_[x[ei], y[ei]]); jj = (np.asarray(jj) * 2).astype(int)
        dipk = np.gradient(gks, mdn, edge_order=1)
        ch += [np.clip(gks[jj] - gks[j0], -120, 120) / 30.0, np.asarray(dq) / 1000.0, dipk[jj] * 30,
               np.cos(az_s[ei] - np.arctan2(np.gradient(ny)[jj], np.gradient(nx)[jj])),
               np.full(len(ei), float(np.clip(gks[j0] - g0, -120, 120)) / 30.0)]
    ch += [np.zeros(len(ei), np.float32), np.zeros(len(ei), np.float32)]
    X = np.nan_to_num(np.stack(ch, 1).astype(np.float32)); return X, ei, tvt_last

def wseq(gd, F, K, POOL, own):
    X_, Y_, Z_, ti, tvt = gd['_X'], gd['_Y'], gd['_Z'], gd['_ti'], gd['_tvt']
    n = len(X_); d7_full, rate, ddr, spr = engine_predict(F, gd, K, own=own); t0w = gd['s']
    rates_full = np.zeros(n, np.float32); rates_full[t0w + 1:] = rate
    conf_full = np.zeros(n, np.float32)
    conf_full[t0w + 1:] = np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) * np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06)
    md = np.arange(n, dtype=float)
    X, ei, tvt_last = build_channels(md, X_, Y_, Z_, ti, t0w, (d7_full - float(ti[t0w])), rates_full, conf_full, POOL, self_wid=gd['wid'])
    target = (tvt[ei] - d7_full[(ei - (t0w + 1))]).astype(np.float32)
    return X, ei, target, d7_full, t0w

# ===== GRU(v92 verbatim) =====
import torch, torch.nn as nn
from sklearn.model_selection import KFold
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.inp = nn.Linear(C + 1, HID); s.gru = nn.GRU(HID, HID, 2, batch_first=True, bidirectional=True, dropout=0.1); s.out = nn.Linear(HID * 2, 1)
    def forward(s, x):
        h = torch.relu(s.inp(x)); h, _ = s.gru(h); return s.out(h)[..., 0]
def train_loop(net, items, seed):
    opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EP)
    rng = np.random.default_rng(seed); BS = 12
    for ep in range(FT_EP):
        net.train(); order = rng.permutation(len(items))
        for b0 in range(0, len(order), BS):
            batch = [items[i] for i in order[b0:b0 + BS]]; L = max(len(b[1]) for b in batch)
            xb = np.zeros((len(batch), L, C + 1), np.float32); yb = np.zeros((len(batch), L), np.float32); mb = np.zeros((len(batch), L), np.float32)
            for i, (Xx, yv, fl) in enumerate(batch):
                xb[i, :len(yv), :C] = Xx; xb[i, :len(yv), C] = fl; yb[i, :len(yv)] = yv; mb[i, :len(yv)] = 1.0
            xb = torch.from_numpy(xb).to(dev); yb = torch.from_numpy(yb).to(dev); mb = torch.from_numpy(mb).to(dev)
            loss = (nn.functional.huber_loss(net(xb), yb, delta=8.0, reduction='none') * mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        sch.step()
    return net

# =====================================================================
#  train 全ロード(fit で必要 / predict では不要)
# =====================================================================
def load_train_wells():
    wids = sorted(set(os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(TRAIN_DIR, '*__horizontal_well.csv'))))
    WELLS = []
    for wi, w in enumerate(wids):
        hw = pd.read_csv(os.path.join(TRAIN_DIR, f'{w}__horizontal_well.csv'), usecols=['X', 'Y', 'Z', 'TVT', 'TVT_input'])
        X = hw['X'].to_numpy(float); Y = hw['Y'].to_numpy(float); Z = hw['Z'].to_numpy(float); tvt = hw['TVT'].to_numpy(float); ti = hw['TVT_input'].to_numpy(float)
        gd = make_gdict(X, Y, Z, ti, tvt)
        if gd is None: continue
        gd['wi'] = wi; gd['wid'] = w; gd['seg'] = segment_well(gd, with_slopes=True)
        gd['_X'] = X; gd['_Y'] = Y; gd['_Z'] = Z; gd['_ti'] = ti; gd['_tvt'] = tvt
        WELLS.append(gd)
    return WELLS

# =====================================================================
#  MODE = fit : 5-fold 学習 -> OOF CV -> モデル保存
# =====================================================================
def run_fit():
    WELLS = load_train_wells()
    print(f'train wells={len(WELLS)}  ({time.time()-t00:.0f}s)', flush=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    idx = np.arange(len(WELLS))
    FOLDS = list(KFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED).split(idx))   # ★完全ランダム
    FOLD_ART = []   # (F_k, K_k, POOL_k, nets)
    oof_e = []; oof_base = []; fold_cv = []
    for fold, (tr_i, va_i) in enumerate(FOLDS):
        tr_wells = [WELLS[i] for i in tr_i]; va_wells = [WELLS[i] for i in va_i]
        # ★train foldの井だけで 場・κ・近傍プール(val完全除外)
        F_k = build_field(tr_wells); K_k = fit_kappa(F_k, tr_wells); POOL_k = build_pool(tr_wells)
        # train seq(own=wi で自井donor除外) → GRU学習items
        items = []
        for gd in tr_wells:
            try:
                X, ei, tg, d7, t0 = wseq(gd, F_k, K_k, POOL_k, own=gd['wi'])
                items.append((X, tg, 0.0)); items.append((X[::-1].copy(), (tg[::-1] - tg[-1]).copy(), 1.0))
            except Exception: pass
        nets = []
        for sd in range(SEEDS):
            torch.manual_seed(sd + 100 + fold); net = Net().to(dev); net = train_loop(net, items, sd + fold * 7); net.eval(); nets.append(net)
        FOLD_ART.append((F_k, K_k, POOL_k, nets))
        # ---- 学習済みモデルを保存 ----
        with open(os.path.join(MODELS_DIR, f'fold{fold}_engine.pkl'), 'wb') as fh:
            pickle.dump(dict(F=F_k, K=K_k, POOL=POOL_k, tr_i=tr_i, va_i=va_i), fh, protocol=4)
        for s, nt in enumerate(nets):
            torch.save(nt.state_dict(), os.path.join(MODELS_DIR, f'fold{fold}_seed{s}.pt'))
        # val予測(val井は F_k/POOL_k に不在 → own=-1)
        fe = []; fb = []
        for gd in va_wells:
            try:
                X, ei, tg, d7, t0 = wseq(gd, F_k, K_k, POOL_k, own=-1)
            except Exception: continue
            xb = np.zeros((1, len(ei), C + 1), np.float32); xb[0, :, :C] = X
            with torch.no_grad():
                tb = torch.from_numpy(xb).to(dev); p = np.mean([nt(tb)[0].cpu().numpy() for nt in nets], axis=0)
            tvt = gd['_tvt']; pred = d7[(ei - (t0 + 1))] + p
            fe.append(pred - tvt[ei]); fb.append(d7[(ei - (t0 + 1))] - tvt[ei])
        fe = np.concatenate(fe); fb = np.concatenate(fb); oof_e.append(fe); oof_base.append(fb)
        cvf = float(np.sqrt(np.mean(fe ** 2))); fold_cv.append(cvf)
        print(f'  fold{fold}: val井={len(va_wells)}  val CV(GRU)={cvf:.4f}  (engine単体 {float(np.sqrt(np.mean(fb**2))):.4f})  ({time.time()-t00:.0f}s)', flush=True)
    oof_e = np.concatenate(oof_e); oof_base = np.concatenate(oof_base)
    CV = float(np.sqrt(np.mean(oof_e ** 2))); CVeng = float(np.sqrt(np.mean(oof_base ** 2)))
    print('\n' + '=' * 64, flush=True)
    print(f'★★ OOF CV(GRフリー GRU, val完全除外) = {CV:.4f}  (engine単体 {CVeng:.4f})', flush=True)
    print(f'   fold別 val CV = {[round(c,3) for c in fold_cv]}', flush=True)
    print(f'   忠実性ゲート: 期待 ~10.03 (許容 9.5-10.5)', flush=True)
    print('=' * 64, flush=True)
    # meta 保存
    meta = dict(N_FOLDS=N_FOLDS, SEEDS=SEEDS, C=C, HID=HID, STRIDE=STRIDE, NBK=NBK, K_FIX=K_FIX,
                PGATE=PGATE, THETA0=THETA0, KNN_K=KNN_K, KNN_H=KNN_H, KBINS=KBINS,
                KAPPA_REGIMES=KAPPA_REGIMES, NBIN=NBIN, CMAX=CMAX, CLIP_DRIFT=CLIP_DRIFT,
                FT_EP=FT_EP, FT_LR=FT_LR, SPLIT_SEED=SPLIT_SEED, WINS=WINS,
                oof_cv=CV, oof_cv_engine=CVeng, fold_cv=fold_cv, n_train_wells=len(WELLS))
    with open(os.path.join(MODELS_DIR, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f'saved 15 GRU(.pt) + {N_FOLDS} engine(.pkl) + meta.json -> {MODELS_DIR}', flush=True)
    print(f'   (既存 grfree_anchor_train.pkl は未変更)', flush=True)
    return CV

# =====================================================================
#  MODE = predict : 保存モデル load -> SPLIT の井を予測(再学習なし)
# =====================================================================
def load_fold_art():
    FOLD_ART = []
    for fold in range(N_FOLDS):
        with open(os.path.join(MODELS_DIR, f'fold{fold}_engine.pkl'), 'rb') as fh:
            eng = pickle.load(fh)
        F_k, K_k, POOL_k = eng['F'], eng['K'], eng['POOL']
        nets = []
        for s in range(SEEDS):
            net = Net().to(dev)
            sd = torch.load(os.path.join(MODELS_DIR, f'fold{fold}_seed{s}.pt'), map_location=dev)
            net.load_state_dict(sd); net.eval(); nets.append(net)
        FOLD_ART.append((F_k, K_k, POOL_k, nets))
    return FOLD_ART

def run_predict():
    import glob as _glob
    SPLIT = os.environ.get('GRF_SPLIT', 'test')
    DIR = TEST_DIR if SPLIT == 'test' else TRAIN_DIR
    # train は本物 grfree_anchor_train.pkl を守るため refit 名で出す
    default_out = os.path.join(ART, 'grfree_anchor_test.pkl' if SPLIT == 'test' else 'grfree_anchor_train_refit.pkl')
    OUTPKL = os.environ.get('GRF_ANCHOR_OUT', default_out)
    if not os.path.exists(os.path.join(MODELS_DIR, 'fold0_engine.pkl')):
        raise SystemExit(f'モデル未保存: {MODELS_DIR} に fold0_engine.pkl が無い。先に GRF_MODE=fit を実行せよ。')
    FOLD_ART = load_fold_art()
    print(f'loaded FOLD_ART: {len(FOLD_ART)} folds x {SEEDS} nets = {len(FOLD_ART)*SEEDS} GRU (再学習なし)  ({time.time()-t00:.0f}s)', flush=True)

    tids = sorted(set(os.path.basename(f).split('__')[0] for f in _glob.glob(os.path.join(DIR, '*__horizontal_well.csv'))))
    ANCH = {}
    for w in tids:
        hw = pd.read_csv(os.path.join(DIR, f'{w}__horizontal_well.csv'), usecols=['X', 'Y', 'Z', 'TVT_input'])
        X_ = hw['X'].to_numpy(float); Y_ = hw['Y'].to_numpy(float); Z_ = hw['Z'].to_numpy(float); ti = hw['TVT_input'].to_numpy(float); n = len(X_)
        finr = np.where(np.isfinite(ti))[0]
        if len(finr) == 0: continue
        gd = make_gdict(X_, Y_, Z_, ti)
        if gd is None: continue
        gd['seg'] = segment_well(gd, with_slopes=False); gd['wid'] = w; t0w = gd['s']; tvt_last = float(ti[t0w]); md = np.arange(n, dtype=float)
        seedpreds = []
        for (F_k, K_k, POOL_k, nets) in FOLD_ART:
            d7_full, rate, ddr, spr = engine_predict(F_k, gd, K_k, own=-1)
            rates_full = np.zeros(n, np.float32); rates_full[t0w + 1:] = rate
            conf_full = np.zeros(n, np.float32); conf_full[t0w + 1:] = np.exp(-np.maximum(ddr - 800.0, 0.0) / 1000.0) * np.exp(-np.nan_to_num(spr, nan=0.1) / 0.06)
            Xc, ei, _ = build_channels(md, X_, Y_, Z_, ti, t0w, (d7_full - tvt_last), rates_full, conf_full, POOL_k, self_wid=None)
            xb = np.zeros((1, len(ei), C + 1), np.float32); xb[0, :, :C] = Xc
            with torch.no_grad():
                tb = torch.from_numpy(xb).to(dev)
                for nt in nets:
                    pp = nt(tb)[0].cpu().numpy()
                    seedpreds.append(np.interp(np.arange(t0w + 1, n), ei, d7_full[(ei - (t0w + 1))] + pp))  # 全eval行TVT予測(1net)
        if not seedpreds: continue
        S = np.stack(seedpreds, 0)                                # (15, eval行)
        tvt = S.mean(0).astype(np.float32)
        # ★sig は定数 9.0 に上書き(bug-fix 版 grfree_anchor_train.pkl と一致)
        sig = np.full(len(tvt), 9.0, np.float32)
        ANCH[w] = dict(tvt=tvt, sig=sig)
    os.makedirs(os.path.dirname(OUTPKL), exist_ok=True)
    with open(OUTPKL, 'wb') as fh:
        pickle.dump(ANCH, fh, protocol=4)
    print(f'saved {OUTPKL}  {len(ANCH)}井 (GRフリー {SPLIT} アンカー tvt/sig=9.0)', flush=True)
    for w, d in ANCH.items():
        print(f'   {w}: eval行={len(d["tvt"])}  tvt[min/max]={d["tvt"].min():.1f}/{d["tvt"].max():.1f}  sig={d["sig"][0]:.1f}', flush=True)
    return OUTPKL

# =====================================================================
if __name__ == '__main__':
    MODE = os.environ.get('GRF_MODE', 'fit')
    print(f'[GRF_MODE={MODE}] dev={dev}  ROOT={ROOT}  MODELS_DIR={MODELS_DIR}', flush=True)
    if MODE == 'fit':
        run_fit()
    elif MODE == 'predict':
        run_predict()
    else:
        raise SystemExit(f'unknown GRF_MODE={MODE} (fit|predict)')
