# -*- coding: utf-8 -*-
"""
v67 ARF — Autoregressive Neural Filter(自己回帰ニューラルフィルタ)
=====================================================================
PF不使用の逐次モデル。各ステップで「走行位置 P の周りの typewell-GR 照合ストリップ」を観測として
GRU に与え、mdn ヘッドで**補正 Δ=(真位置−現在P)を予測**して P を更新する。
  - 自己回帰: 観測ストリップは"モデル自身の走行 P"の周りで作る(前の予測が次に効く)
  - 自己補正: 予測対象を「真位置−P」にすることで drift を打ち消す(Shrey の drift 失敗を回避)
  - 双峰: mdn(K成分)で datum の両モードを保持 → posterior mean(error_anatomy §4)
物理prior(struct)は STRUCT=1 のとき soft 錨入力として足す(STRUCT=0 が純GR照合の"物理prior無し"版)。

★検証は leave-field-out 既定(XY を KMeans でフィールド化)。井戸 GroupKFold は CV_MODE=well。
  NN脚の LB 非転移(v34/v62 墓標)を早期に見抜く唯一の物差し。

やること: このスクリプトは train(OOF 生成)まで。実行はユーザー。
実行:
  # キャッシュ構築(初回のみ, 生CSV→per-well系列)
  .venv/Scripts/python.exe v67/arf_v67.py cache
  # 学習(STRUCT 0/1 の2本を回して比較 = 物理prior寄与の切り分け)
  STRUCT=0 CV_MODE=field .venv/Scripts/python.exe v67/arf_v67.py train
  STRUCT=1 CV_MODE=field .venv/Scripts/python.exe v67/arf_v67.py train
env: STRUCT(0) CV_MODE(field|well) N_FIELDS(16) EPOCHS(12) BATCH(16) LR(1e-3) HIDDEN(160)
     KMIX(3) DR(48) DSTEP(3) GR_SIG(15) CHUNK(160=TBPTT) LAMBDA_POS(0.2)
     N_WELLS(0=全; スモーク) SEED(20250714) DEVICE(auto) EVAL_STRIDE(1)
出力: v67/artifacts_v67/arf_oof_{struct}_{cv}.npy(行別posterior-mean TVT)+ results_{...}.json + png
"""
import os, sys, json, glob, pickle, math, time
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v67" / "artifacts_v67"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
STRUCT_PKL = PROJECT / "v38" / "artifacts_v38" / "priors" / "prior_v66_leanv1cumul_train.pkl"
CACHE = OUT / "cache_v67.pkl"

STRUCT   = os.environ.get("STRUCT", "0") == "1"
CV_MODE  = os.environ.get("CV_MODE", "field")      # field(leave-field-out) | well(GroupKFold)
N_FIELDS = int(os.environ.get("N_FIELDS", "16"))
EPOCHS   = int(os.environ.get("EPOCHS", "12"))
BATCH    = int(os.environ.get("BATCH", "16"))
LR       = float(os.environ.get("LR", "1e-3"))
HIDDEN   = int(os.environ.get("HIDDEN", "160"))
KMIX     = int(os.environ.get("KMIX", "3"))
DR       = float(os.environ.get("DR", "48"))        # emission δ半幅 ft
DSTEP    = float(os.environ.get("DSTEP", "3"))       # δ刻み ft
GR_SIG   = float(os.environ.get("GR_SIG", "15"))
CHUNK    = int(os.environ.get("CHUNK", "160"))       # TBPTT チャンク長
LAMBDA_POS = float(os.environ.get("LAMBDA_POS", "0.2"))
N_WELLS  = int(os.environ.get("N_WELLS", "0")) or None
SEED     = int(os.environ.get("SEED", "20250714"))
EVAL_STRIDE = int(os.environ.get("EVAL_STRIDE", "1"))
DTVT = 0.5                                            # typewell リサンプル grid 刻み


# ============================ データキャッシュ ============================
def build_cache():
    struct = pickle.loads(STRUCT_PKL.read_bytes()) if STRUCT_PKL.exists() else {}
    wells = []
    fps = sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv")))
    if N_WELLS:
        fps = fps[:N_WELLS]
    for fp in fps:
        wid = os.path.basename(fp).split("__")[0]
        tw_fp = fp.replace("__horizontal_well", "__typewell")
        if not os.path.exists(tw_fp):
            continue
        hw = pd.read_csv(fp); tw = pd.read_csv(tw_fp).sort_values("TVT")
        if "TVT" not in hw.columns or hw["TVT"].isna().all():
            continue
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(kn) < 10 or len(ev) < 5:
            continue
        ev = ev.iloc[::EVAL_STRIDE]
        tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
        m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
        if len(tw_tvt) < 5:
            continue
        # typewell を一様 TVT grid にリサンプル(emission の高速 gather 用)
        tvt0 = float(tw_tvt.min()); tvt1 = float(tw_tvt.max())
        n_grid = int(np.ceil((tvt1 - tvt0) / DTVT)) + 1
        grid_tvt = tvt0 + np.arange(n_grid) * DTVT
        grid_gr = np.interp(grid_tvt, tw_tvt, tw_gr).astype(np.float32)
        # 水平GR(補間補填)
        gr_fill = float(np.nanmean(tw_gr))
        hgr_all = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(gr_fill).to_numpy(float)
        ev_pos = hw.index.get_indexer(ev.index)
        hgr = hgr_all[ev_pos].astype(np.float32)
        Z = ev["Z"].to_numpy(float); MD = ev["MD"].to_numpy(float)
        lk = kn.iloc[-1]; last_tvt = float(lk["TVT_input"]); z_heel = float(lk["Z"]); md_heel = float(lk["MD"])
        # 幾何
        dmd = np.diff(MD, prepend=md_heel).astype(np.float32); dmd = np.maximum(dmd, 1e-3)
        dz_step = np.diff(Z, prepend=z_heel).astype(np.float32)     # ステップ間 ΔZ
        md_since = (MD - md_heel).astype(np.float32)
        ir_tail = kn.tail(30)
        dt = np.diff(ir_tail["TVT_input"].values); dzk = np.diff(ir_tail["Z"].values); dmk = np.diff(ir_tail["MD"].values)
        mm = dmk > 0
        ir = float(np.median((dt + dzk)[mm] / dmk[mm])) if mm.sum() >= 3 else 0.0
        target = ev["TVT"].to_numpy(np.float32)
        if not np.isfinite(target).any():
            continue
        # struct(あれば, eval行に整列)
        sp = struct.get(wid)
        if sp is not None and len(sp["tvt"]) == len(hw[hw["TVT_input"].isna()]):
            s_tvt_full = np.asarray(sp["tvt"], np.float32); s_sig_full = np.asarray(sp["sig"], np.float32)
            # EVAL_STRIDE 適用(ev は上で stride 済 → struct も同じ位置を取る)
            keep = np.arange(len(hw[hw["TVT_input"].isna()]))[::EVAL_STRIDE]
            s_tvt = s_tvt_full[keep]; s_sig = s_sig_full[keep]
        else:
            s_tvt = np.full(len(ev), last_tvt, np.float32); s_sig = np.full(len(ev), 1e9, np.float32)
        wells.append(dict(
            wid=wid, hgr=hgr, dmd=dmd, dz_step=dz_step, md_since=md_since,
            grid_gr=grid_gr, tvt0=np.float32(tvt0), lk=np.float32(last_tvt), ir=np.float32(ir),
            target=target, s_tvt=s_tvt, s_sig=s_sig,
            cx=float(hw["X"].mean()), cy=float(hw["Y"].mean())))
    print(f"[cache] wells {len(wells)}  (struct整列 {sum(1 for w in wells if (w['s_sig']<1e8).any())})")
    CACHE.write_bytes(pickle.dumps(wells))
    print(f"[saved] {CACHE}")
    return wells


def load_cache():
    if not CACHE.exists():
        return build_cache()
    return pickle.loads(CACHE.read_bytes())


def make_fields(wells, seed):
    from sklearn.cluster import KMeans
    xy = np.array([[w["cx"], w["cy"]] for w in wells])
    k = min(N_FIELDS, len(wells))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(xy)
    return km.labels_


# ============================ モデル ============================
import torch
import torch.nn as nn


class ARF(nn.Module):
    """自己回帰ニューラルフィルタ: emission(GR照合ストリップ)→ GRUCell → mdn(Δ補正)。"""
    def __init__(self, d_grid, use_struct, hidden=HIDDEN, kmix=KMIX):
        super().__init__()
        self.use_struct = use_struct
        geom_dim = 4                        # dmd_n, dz_step, md_since_n, dip_prev
        extra = 2 if use_struct else 0      # (s_tvt - P), s_sig_n
        self.in_dim = d_grid + 2 + geom_dim + 1 + extra   # emission + peak(δ,val) + geom + hgr_n + struct
        self.cell = nn.GRUCell(self.in_dim, hidden)
        self.head = nn.Linear(hidden, 3 * kmix)        # π, μ, log_sig
        self.kmix = kmix

    def head_split(self, h):
        o = self.head(h)
        pi = torch.log_softmax(o[:, :self.kmix], dim=1)
        mu = o[:, self.kmix:2 * self.kmix]
        log_sig = o[:, 2 * self.kmix:].clamp(-4.0, 4.0)
        return pi, mu, log_sig


def mdn_nll(pi_log, mu, log_sig, y):
    """y: (B,) 真の補正 Δ。混合ガウス NLL。"""
    y = y[:, None]
    comp = -0.5 * ((y - mu) / torch.exp(log_sig)) ** 2 - log_sig - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(pi_log + comp, dim=1)      # (B,)


def mixture_mean(pi_log, mu):
    return (torch.softmax(pi_log, dim=1) * mu).sum(1)


# ============================ バッチ整形 ============================
def collate(batch, delta, device):
    """batch: wells の list。pad + mask して tensor 化。"""
    T = max(len(w["hgr"]) for w in batch); B = len(batch); D = len(delta)
    def pad(key, dt=np.float32):
        a = np.zeros((B, T), dt)
        for i, w in enumerate(batch): a[i, :len(w[key])] = w[key]
        return a
    hgr = pad("hgr"); dmd = pad("dmd"); dz = pad("dz_step"); mds = pad("md_since")
    tgt = pad("target"); s_tvt = pad("s_tvt"); s_sig = pad("s_sig")
    mask = np.zeros((B, T), np.float32)
    for i, w in enumerate(batch): mask[i, :len(w["hgr"])] = 1.0
    Lg = max(len(w["grid_gr"]) for w in batch)
    grid = np.zeros((B, Lg), np.float32); tvt0 = np.zeros(B, np.float32)
    lk = np.zeros(B, np.float32); ir = np.zeros(B, np.float32)
    for i, w in enumerate(batch):
        grid[i, :len(w["grid_gr"])] = w["grid_gr"]; grid[i, len(w["grid_gr"]):] = w["grid_gr"][-1]
        tvt0[i] = w["tvt0"]; lk[i] = w["lk"]; ir[i] = w["ir"]
    t = lambda a: torch.tensor(a, device=device)
    return dict(hgr=t(hgr), dmd=t(dmd), dz=t(dz), mds=t(mds), tgt=t(tgt), mask=t(mask),
                s_tvt=t(s_tvt), s_sig=t(s_sig), grid=t(grid), tvt0=t(tvt0), lk=t(lk), ir=t(ir),
                Lg=Lg, T=T, B=B)


def emission(P, hgr_t, grid, tvt0, delta, Lg):
    """P:(B,) 走行位置。grid:(B,Lg) typewell-GR。→ 照合尤度ストリップ (B,D)。"""
    cand = P[:, None] + delta[None, :]                 # (B,D)
    idx = ((cand - tvt0[:, None]) / DTVT).round().long().clamp(0, Lg - 1)
    tw_gr = torch.gather(grid, 1, idx)                 # (B,D)
    return torch.exp(-0.5 * ((hgr_t[:, None] - tw_gr) / GR_SIG) ** 2)


def run_well_batch(model, bt, delta, device, opt=None):
    """自己回帰ロールアウト。opt指定時=学習(チャンク毎backward=TBPTT有界メモリ)。→ (mean loss, P軌跡)。"""
    train = opt is not None
    B, T, Lg = bt["B"], bt["T"], bt["Lg"]
    P = bt["lk"].clone(); dip = bt["ir"].clone()
    h = torch.zeros(B, HIDDEN, device=device)
    mds_scale = 1.0 / (bt["mds"].amax(1, keepdim=True) + 1.0)
    traj = torch.zeros(B, T, device=device)
    ch_loss = torch.zeros((), device=device); ch_n = torch.zeros((), device=device)
    tot = 0.0; nb = 0
    for t in range(T):
        em = emission(P, bt["hgr"][:, t], bt["grid"], bt["tvt0"], delta, Lg)   # (B,D)
        pk = em.argmax(1)                              # 照合ピーク位置(=GR が最も合う δ)
        peak = torch.stack([delta[pk] / DR, em.max(1).values], 1)   # (B,2) 明示 localization 特徴
        geom = torch.stack([bt["dmd"][:, t] / 50.0, bt["dz"][:, t] / 10.0,
                            bt["mds"][:, t] * mds_scale[:, 0], dip / 0.05], 1)
        feats = [em, peak, geom, bt["hgr"][:, t:t + 1] / 100.0]
        if model.use_struct:
            sa = ((bt["s_tvt"][:, t] - P) / 20.0).clamp(-5, 5)
            ss = (1.0 / (bt["s_sig"][:, t].clamp(3, 60))).clamp(0, 0.34)
            feats.append(torch.stack([sa, ss], 1))
        h = model.cell(torch.cat(feats, 1), h)
        pi, mu, log_sig = model.head_split(h)
        y = (bt["tgt"][:, t] - P).detach()             # ★補正ターゲット = 真位置 − 現在P(自己補正)
        nll = mdn_nll(pi, mu, log_sig, y)
        dmean = mixture_mean(pi, mu)
        Pn = P + dmean
        m = bt["mask"][:, t]
        ch_loss = ch_loss + (nll * m).sum() + LAMBDA_POS * (((Pn - bt["tgt"][:, t]) ** 2) * m).sum()
        ch_n = ch_n + m.sum()
        dip = (dmean / bt["dmd"][:, t]).detach().clamp(-0.05, 0.05)
        P = Pn; traj[:, t] = P.detach()
        if (t + 1) % CHUNK == 0 or t == T - 1:          # TBPTT: チャンク毎に backward→detach
            if train and float(ch_n) > 0:
                lc = ch_loss / ch_n.clamp_min(1.0)
                opt.zero_grad(); lc.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                tot += lc.detach().item(); nb += 1
            P = P.detach(); h = h.detach(); dip = dip.detach()
            ch_loss = torch.zeros((), device=device); ch_n = torch.zeros((), device=device)
    return (tot / max(nb, 1)), traj


# ============================ 学習 ============================
def train_fold(wells_tr, wells_va, delta, device):
    model = ARF(len(delta), STRUCT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    order = np.arange(len(wells_tr)); rng = np.random.RandomState(SEED)
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(order); tot = 0.0; nb = 0
        for i in range(0, len(order), BATCH):
            bt = collate([wells_tr[j] for j in order[i:i + BATCH]], delta.cpu().numpy(), device)
            loss, _ = run_well_batch(model, bt, delta, device, opt=opt)
            tot += loss; nb += 1
        print(f"    ep{ep+1}/{EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval(); oof = {}
    with torch.no_grad():
        for i in range(0, len(wells_va), BATCH):
            batch = wells_va[i:i + BATCH]
            _, traj = run_well_batch(model, collate(batch, delta.cpu().numpy(), device), delta, device, opt=None)
            for b, w in enumerate(batch):
                oof[w["wid"]] = traj[b, :len(w["hgr"])].cpu().numpy()
    return oof


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "cache":
        build_cache(); return
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if (os.environ.get("DEVICE", "auto") != "cpu" and torch.cuda.is_available()) else "cpu"
    wells = load_cache()
    print(f"[v67] wells {len(wells)}  STRUCT={STRUCT}  CV={CV_MODE}  device={device}  KMIX={KMIX} DR={DR}")
    delta = torch.arange(-DR, DR + 1e-6, DSTEP, device=device, dtype=torch.float32)
    # fold 割当
    if CV_MODE == "field":
        fields = make_fields(wells, SEED)
        folds = sorted(set(fields))
        assign = fields
        print(f"[cv] leave-field-out: {len(folds)} fields  sizes={np.bincount(fields).tolist()}")
    else:
        from sklearn.model_selection import GroupKFold
        g = np.array([w["wid"] for w in wells])
        assign = np.zeros(len(wells), int)
        for k, (_, va) in enumerate(GroupKFold(5).split(wells, groups=g)):
            assign[va] = k
        folds = list(range(5)); print("[cv] well GroupKFold(5)")

    oof_all = {}; t0 = time.time()
    for k in folds:
        va_idx = np.where(assign == k)[0]; tr_idx = np.where(assign != k)[0]
        wtr = [wells[i] for i in tr_idx]; wva = [wells[i] for i in va_idx]
        print(f"  fold {k}: train {len(wtr)} / val {len(wva)}", flush=True)
        oof_all.update(train_fold(wtr, wva, delta, device))

    # 評価(全行プール RMSE + 坑井別)
    perw = {}; pool_e = []
    for w in wells:
        if w["wid"] not in oof_all: continue
        p = oof_all[w["wid"]]; tgt = w["target"]; n = min(len(p), len(tgt))
        e = p[:n] - tgt[:n]; e = e[np.isfinite(e)]
        if e.size: perw[w["wid"]] = float(np.sqrt(np.mean(e ** 2))); pool_e.append(e)
    pool = np.concatenate(pool_e); overall = float(np.sqrt(np.mean(pool ** 2)))
    tag = f"{'struct' if STRUCT else 'nostruct'}_{CV_MODE}"
    print(f"\n[v67 ARF] {tag}  overall RMSE = {overall:.4f}  坑井別中央 {np.median(list(perw.values())):.4f}  "
          f"({len(perw)}坑井, {time.time()-t0:.0f}s)")
    # 保存
    ids, arr = [], []
    for w in wells:
        if w["wid"] in oof_all:
            p = oof_all[w["wid"]]
            for j in range(len(p)): ids.append(f"{w['wid']}_{j}"); arr.append(p[j])
    np.save(OUT / f"arf_oof_{tag}.npy", np.array(arr, np.float32))
    (OUT / f"arf_ids_{tag}.json").write_text(json.dumps(ids))
    (OUT / f"results_{tag}.json").write_text(json.dumps(dict(
        overall=overall, per_well_median=float(np.median(list(perw.values()))),
        n_wells=len(perw), struct=STRUCT, cv=CV_MODE,
        cfg=dict(epochs=EPOCHS, hidden=HIDDEN, kmix=KMIX, dr=DR, dstep=DSTEP, gr_sig=GR_SIG,
                 lambda_pos=LAMBDA_POS, n_fields=N_FIELDS, eval_stride=EVAL_STRIDE)), indent=2))
    print(f"[saved] arf_oof_{tag}.npy / results_{tag}.json")


if __name__ == "__main__":
    main()
