# -*- coding: utf-8 -*-
"""
v66 構造prior(struct場)強化ビルダー
=====================================================================
v38 struct_prior(K16 dip場)の作り直し。エンジン消費契約は不変:
  {well: {"tvt": ev行の絶対TVT予測, "sig": per-row σ}} の pkl
  → v38/artifacts_v38/priors/prior_v66{TAG}_train.pkl に INSTALL され、
    V38_PRIOR=v66{TAG} を付けるだけで v65 エンジン/チューナが無改造で差し替え利用。

アップグレード(全て env flag、既定=全ON。根拠は docs/v66.md):
  U1 SEG_LEN    長さ適応セグメント(v38はK=16固定。長坑井の短波長起伏が消える → v56)
  U2 VEC        場クエリの回帰化 1=θ0拘束回帰 / 2=2Dベクトル場(条件数でθ0解へブレンド)
                → v38 の c/proj deproject(proj→0でノイズ増幅)と |proj|>0.3 の破棄を廃止
  U3 TWO_KAPPA  κ を幾何項α(−ΔZ)とフィールド項β(ΔANCC予測)に分離して buffered-LOO 推定
                (v38 は確定幾何まで単一κ=0.11 に縮めていた)
  U4 QW         セグメント傾きの標準誤差 → 品質重み
  U5 ROBUST     Huber IRLS(断層/外れ坑井の dip 汚染防止)
  U6 ANISO      走向方向に伸ばす異方性カーネル(既定 1.0=OFF。1.5〜2 をアブレーション)

やらないこと(実験済みの死): test情報の場への注入(fullfield LB7.963=中立)・
絶対標高の転写(v31/v42。場は今後も「傾き」のみ、絶対レベルは自坑井prefixから積分)・
cross-field 外挿(error_anatomy)。

実行:
  .venv/Scripts/python.exe v66/build_struct_v66.py            # train LOO + test pkl + 比較レポート
  .venv/Scripts/python.exe v66/build_struct_v66.py parity     # v38 完全再現の自己検証(npz 一致 assert)
env:
  SEG_LEN(150) VEC(2) TWO_KAPPA(1) QW(1) ROBUST(1) ANISO(1.0) PGATE(0.35)
  KNN_K(15) KNN_H(500) THETA0(118.4) TAG("") INSTALL(1) N_WELLS(0=全; スモーク用)
  ※PGATE = perp行(|cos(az-θ0)|<PGATE)の専用係数α_p/β_p+専用σ。v38のhard GATE(α_p=β_p=0)の学習版
出力: v66/artifacts_v66/prior_v66{TAG}_{train|test}.pkl, struct_field_v66{TAG}.npz,
      struct_sigcurve_v66{TAG}.json, struct_loo_compare{TAG}.png
判定: overall LOO struct RMSE(v38 実測 9.709 ft を下回るか)→ 勝ったら PF A/B(eval_struct_pf_v66.py)
"""
import sys, os, glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v66" / "artifacts_v66"; OUT.mkdir(parents=True, exist_ok=True)
V38_ART = PROJECT / "v38" / "artifacts_v38"
TRAIN_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
TEST_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "test"

# ---------------- 設定(v38 と同値の定数はそのまま) ----------------
SEG_LEN = float(os.environ.get("SEG_LEN", "150"))     # [U1] 目標セグメント弧長 ft(0=v38同様の固定16分割)
VEC = int(os.environ.get("VEC", "2"))                  # [U2] 1=θ0拘束回帰 2=2Dベクトル場
TWO_KAPPA = os.environ.get("TWO_KAPPA", "1") == "1"    # [U3]
QW = os.environ.get("QW", "1") == "1"                  # [U4]
ROBUST = os.environ.get("ROBUST", "1") == "1"          # [U5]
ANISO = float(os.environ.get("ANISO", "1.0"))          # [U6] 走向方向レンジ倍率(1=等方)
PGATE = float(os.environ.get("PGATE", "0.35"))         # [U7] perp行(|cos(az-θ0)|<PGATE)を別係数/別σで扱う(0=無効)
SIGMODE = os.environ.get("SIGMODE", "local")           # [U8] σモデル local=v66既定(dd_row) | cumul=ランダムウォーク累積
SIG_CAP = float(os.environ.get("SIG_CAP", "40"))       # σ上限(cumulでは蓄積で大きくなるので緩める余地。既定40=v38)
SPREAD = os.environ.get("SPREAD", "0") == "1"          # [U9] cumul に「ドナー不一致(近傍dip割れ)」の累積項を追加
KNN_K = int(os.environ.get("KNN_K", "15"))
KNN_H = float(os.environ.get("KNN_H", "500"))
THETA0 = float(os.environ.get("THETA0", "118.4"))
TAG = os.environ.get("TAG", "")
INSTALL = os.environ.get("INSTALL", "1") == "1"
N_WELLS = int(os.environ.get("N_WELLS", "0")) or None

KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]
KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1
SIG_MIN, SIG_MAX = 3.0, 40.0
N_SIG_BINS = 8
K_FIX = 16            # SEG_LEN=0 のときの固定分割数(v38互換)
K_SEG_MAX = 64        # 適応分割の上限
CMAX = 0.30           # |dip| ft/ft のサニティ上限(≈17°。超は断層/破綻フィットとして破棄)
HUBER_K = 1.345
IRLS_IT = 2
NAME = f"v66{TAG}"
_C0, _S0 = np.cos(np.radians(THETA0)), np.sin(np.radians(THETA0))


# ================= 井戸ロード =================
def load_well(fp, need_tvt):
    cols = ["X", "Y", "Z", "TVT", "TVT_input"] if need_tvt else ["X", "Y", "Z", "TVT_input"]
    w = pd.read_csv(fp, usecols=cols)
    ti = w["TVT_input"].to_numpy(float)
    fin = np.where(np.isfinite(ti))[0]
    if len(fin) == 0:
        return None
    s = int(fin.max())
    X, Y, Z = w["X"].to_numpy(float), w["Y"].to_numpy(float), w["Z"].to_numpy(float)
    n = len(X) - 1 - s
    if n < 2:
        return None
    dX = np.diff(X)[s:]; dY = np.diff(Y)[s:]
    ndz = -np.diff(Z)[s:]
    Lxy = np.maximum(np.hypot(dX, dY), 1e-3)
    p_row = (dX * _C0 + dY * _S0) / Lxy                  # 行方位のθ0射影(cos)
    d = dict(s=s, n=n, dX=dX, dY=dY, ndz=ndz, Lxy=Lxy, arc=np.cumsum(Lxy),
             Xl=X[s:], Yl=Y[s:], anchor=float(ti[s]),
             perp=(np.abs(p_row) < PGATE) if PGATE > 0 else np.zeros(n, bool),
             nev=int(np.sum(~np.isfinite(ti))))          # エンジンの ev 行数(NaN数)= 整合チェック用
    if need_tvt:
        tvt = w["TVT"].to_numpy(float)
        d["R0"] = tvt[s + 1:] - tvt[s]
    return d


# ================= セグメント分割 + 傾き抽出 =================
def segment_well(wd):
    """弧長基底のランプフィット。c=ft/ft(経路XY距離あたりのANCC変化)、se=各係数の標準誤差。"""
    n, A = wd["n"], wd["arc"]; total = float(A[-1])
    if SEG_LEN <= 0:
        K = min(K_FIX, max(1, n // 4))
    else:
        K = int(np.clip(round(total / SEG_LEN), 1, min(K_SEG_MAX, max(1, n // 24))))
    edges = np.linspace(0.0, total, K + 1)
    phi = np.column_stack([np.clip(A - edges[j], 0.0, edges[j + 1] - edges[j]) for j in range(K)])
    y = wd["R0"] - np.cumsum(wd["ndz"])                 # 累積ΔANCC(= R0 − U)
    m = np.isfinite(y)
    c, res, rank, _ = np.linalg.lstsq(phi[m], y[m], rcond=None)
    dof = max(int(m.sum()) - K, 1)
    ssr = float(res[0]) if len(res) else float(np.sum((phi[m] @ c - y[m]) ** 2))
    cov = (ssr / dof) * np.linalg.pinv(phi[m].T @ phi[m])
    se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    segid = np.clip(np.searchsorted(edges[1:], A, side="left"), 0, K - 1)
    mid = np.empty((K, 2)); az = np.empty(K)
    Xl, Yl = wd["Xl"], wd["Yl"]
    for j in range(K):
        rows = np.where(segid == j)[0]
        j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)
        az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0])
        mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)
    return segid, mid, az, c, se


def build_field(wells):
    """全wellのセグメント → 場の点群 (x, y, c[ft/ft], az[rad], wi, qw)。|c|>CMAX は破棄。"""
    px, py, pc, paz, pwi, pse = [], [], [], [], [], []
    for wd in wells:
        segid, mid, az, c, se = segment_well(wd)
        wd["segid"], wd["mid"], wd["az"] = segid, mid, az
        ok = np.isfinite(c) & (np.abs(c) <= CMAX) & np.isfinite(se)
        px.append(mid[ok, 0]); py.append(mid[ok, 1]); pc.append(c[ok])
        paz.append(az[ok]); pwi.append(np.full(int(ok.sum()), wd["wi"], float)); pse.append(se[ok])
        wd["n_drop"] = int((~ok).sum())
    F = dict(x=np.concatenate(px), y=np.concatenate(py), c=np.concatenate(pc),
             az=np.concatenate(paz), wi=np.concatenate(pwi), se=np.concatenate(pse))
    if QW:
        med2 = float(np.median(F["se"])) ** 2
        F["qw"] = 1.0 / (F["se"] ** 2 + med2 + 1e-12)
        F["qw"] /= np.median(F["qw"])                    # スケール正規化(中央値=1)
    else:
        F["qw"] = np.ones(len(F["x"]))
    F["ux"], F["uy"] = np.cos(F["az"]), np.sin(F["az"])
    F["p0"] = F["ux"] * _C0 + F["uy"] * _S0              # θ0 方向への方位射影
    return F


# ================= 場クエリ(回帰型) =================
def query_well(F, mids, own, mdbuf=0.0):
    """1well の全セグメントを一括クエリ → (g (K,2)[ft/ft 勾配ベクトル], dd (K,)[等方ドナー距離])"""
    keep = F["wi"] != own
    fx, fy = F["x"][keep], F["y"][keep]
    fc, fqw = F["c"][keep], F["qw"][keep]
    fux, fuy, fp0 = F["ux"][keep], F["uy"][keep], F["p0"][keep]
    dx = fx[None, :] - mids[:, 0:1]; dy = fy[None, :] - mids[:, 1:2]
    d2i = dx * dx + dy * dy                               # 等方(dd・buffer 用)
    if ANISO != 1.0:                                      # [U6] 走向(θ0直交)方向を ANISO 倍遠くまで見る
        tpar = dx * _C0 + dy * _S0; tper = -dx * _S0 + dy * _C0
        d2a = tpar * tpar + (tper / ANISO) ** 2
    else:
        d2a = d2i
    if mdbuf > 0:
        d2a = np.where(d2i >= mdbuf * mdbuf, d2a, np.inf)
    kk = min(KNN_K, d2a.shape[1] - 1) if d2a.shape[1] > 1 else 1
    idx = np.argpartition(d2a, kk - 1, axis=1)[:, :kk]
    r = np.arange(len(mids))[:, None]
    d2a_s, d2i_s = d2a[r, idx], d2i[r, idx]
    c_s, p0_s = fc[idx], fp0[idx]
    ux_s, uy_s = fux[idx], fuy[idx]
    w = np.exp(np.maximum(-d2a_s / (2 * KNN_H ** 2), -700.0)) * fqw[idx]
    w[~np.isfinite(d2a_s)] = 0.0
    wsum = w.sum(1)
    dead = wsum <= 1e-300
    with np.errstate(invalid="ignore"):
        dd = np.sqrt(np.nanmedian(np.where(np.isfinite(d2a_s), d2i_s, np.nan), axis=1))
    dd[~np.isfinite(dd)] = KBINS[-2] * 2                  # 候補ゼロ → 最遠ビン扱い
    # --- θ0 拘束回帰(VEC=1 の解 & VEC=2 の正則化中心): c ≈ s·p0 ---
    s_hat = (w * c_s * p0_s).sum(1) / ((w * p0_s ** 2).sum(1) + 1e-9)
    if ROBUST:
        with np.errstate(invalid="ignore"):
            for _ in range(IRLS_IT):                      # [U5](w=0 要素は NaN 化→hub=1 で無害化)
                rres = np.where(w > 0, c_s - s_hat[:, None] * p0_s, np.nan)
                med = np.nanmedian(rres, axis=1, keepdims=True)
                scale = 1.4826 * np.nanmedian(np.abs(rres - med), axis=1, keepdims=True) + 1e-6
                hub = np.minimum(1.0, HUBER_K * scale / np.maximum(np.abs(rres - med), 1e-12))
                hub = np.nan_to_num(hub, nan=1.0, posinf=1.0)
                w2 = w * hub
                s_hat = (w2 * c_s * p0_s).sum(1) / ((w2 * p0_s ** 2).sum(1) + 1e-9)
        w = w2
    g = np.column_stack([s_hat * _C0, s_hat * _S0])
    if VEC >= 2:                                          # [U2] 2Dベクトル場(方位多様なら勾配を解く)
        Sxx = (w * ux_s * ux_s).sum(1); Sxy = (w * ux_s * uy_s).sum(1); Syy = (w * uy_s * uy_s).sum(1)
        bx = (w * c_s * ux_s).sum(1); by = (w * c_s * uy_s).sum(1)
        tr = Sxx + Syy; det = Sxx * Syy - Sxy * Sxy
        disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
        lam1 = (tr + disc) / 2; lam2 = (tr - disc) / 2
        ratio = lam2 / np.maximum(lam1, 1e-12)
        lam = lam1 * np.maximum(0.05, 0.25 - ratio)       # 方位縮退ほど θ0 解へ強く引く
        A11 = Sxx + lam; A22 = Syy + lam; A12 = Sxy
        b1 = bx + lam * g[:, 0]; b2 = by + lam * g[:, 1]
        detA = A11 * A22 - A12 * A12
        gx = (b1 * A22 - b2 * A12) / np.maximum(detA, 1e-12)
        gy = (b2 * A11 - b1 * A12) / np.maximum(detA, 1e-12)
        g = np.column_stack([gx, gy])
    g[dead] = 0.0
    gn = np.hypot(g[:, 0], g[:, 1])
    big = gn > CMAX                                       # 勾配のサニティ(deproject暴走の残滓防止)
    if big.any():
        g[big] *= (CMAX / gn[big])[:, None]
    # [U9] ドナー不一致 = 場勾配 g で説明できないドナー c の加重RMS(ft/ft)。dd と独立な信頼度軸。
    pred_don = g[:, 0:1] * ux_s + g[:, 1:2] * uy_s        # 各ドナー位置での場の期待傾き
    resd = c_s - pred_don
    spread = np.sqrt(np.maximum((w * resd ** 2).sum(1) / np.maximum(wsum, 1e-12), 0.0))
    spread[dead] = 0.0
    return g, dd, spread


def field_rows(wd, g):
    """per-row のフィールド項 f = g·(ΔX,ΔY)(その行の実ステップで評価 → 単位/傾斜の近似が消える)"""
    sid = wd["segid"]
    return g[sid, 0] * wd["dX"] + g[sid, 1] * wd["dY"]


# ================= κ(α/β)推定: buffered-LOO 同時最小二乗 =================
# [U7] PGATE>0 のとき perp行(|cos(az-θ0)|<PGATE)は専用の大域係数 (α_p, β_p) を持つ。
#      v38 の hard GATE(=α_p=β_p=0)を解空間に含む一般化(データが gate の是非を決める)。
def fit_kappa(F, wells):
    npar = 2 * NBIN if TWO_KAPPA else NBIN
    nperp = (2 if TWO_KAPPA else 1) if PGATE > 0 else 0
    ncol = npar + nperp
    A = np.zeros((ncol, ncol)); yv = np.zeros(ncol)
    for R in KAPPA_REGIMES:
        for wd in wells:
            g, dd, _ = query_well(F, wd["mid"], wd["wi"], mdbuf=R)
            f = field_rows(wd, g)
            a = wd["ndz"]
            par = ~wd["perp"]
            b_row = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            cols = []
            for b in range(NBIN):
                m = (b_row == b) & par
                if TWO_KAPPA:
                    cols.append(np.cumsum(np.where(m, a, 0.0)))
                    cols.append(np.cumsum(np.where(m, f, 0.0)))
                else:
                    cols.append(np.cumsum(np.where(m, a + f, 0.0)))
            if nperp:
                if TWO_KAPPA:
                    cols.append(np.cumsum(np.where(wd["perp"], a, 0.0)))
                    cols.append(np.cumsum(np.where(wd["perp"], f, 0.0)))
                else:
                    cols.append(np.cumsum(np.where(wd["perp"], a + f, 0.0)))
            G = np.column_stack(cols)
            ok = np.isfinite(wd["R0"])
            G = np.nan_to_num(G[ok]); y = wd["R0"][ok]
            A += G.T @ G; yv += G.T @ y
    coef = np.linalg.lstsq(A, yv, rcond=None)[0]
    if TWO_KAPPA:
        alpha_raw, beta_raw = coef[0:npar:2].copy(), coef[1:npar:2].copy()
        ap_raw, bp_raw = (float(coef[npar]), float(coef[npar + 1])) if nperp else (0.0, 0.0)
    else:
        alpha_raw = beta_raw = coef[:npar].copy()
        ap_raw = bp_raw = float(coef[npar]) if nperp else 0.0
    K = dict(alpha=np.clip(alpha_raw, -0.25, 1.5), beta=np.clip(beta_raw, -0.25, 1.5),
             ap=float(np.clip(ap_raw, -0.25, 1.5)), bp=float(np.clip(bp_raw, -0.25, 1.5)),
             alpha_raw=alpha_raw, beta_raw=beta_raw, ap_raw=ap_raw, bp_raw=bp_raw)
    return K


def predict_resid(F, wd, K, own):
    g, dd, spread = query_well(F, wd["mid"], own, mdbuf=0.0)
    f = field_rows(wd, g)
    b_row = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
    av = K["alpha"][b_row]; bv = K["beta"][b_row]
    if PGATE > 0:
        av = np.where(wd["perp"], K["ap"], av); bv = np.where(wd["perp"], K["bp"], bv)
    resid = np.cumsum(av * wd["ndz"] + bv * f)
    return resid, dd[wd["segid"]], spread[wd["segid"]]


# ================= 錨品質指標: ガウス NLL =================
# 錨は (tvt, σ) を消費する。RMSE は σ を無視するので「錨としての良さ」の弱い代理。
# NLL = mean[ 0.5(err/σ)^2 + log σ + 0.5 log 2π ] = 較正込みの尤度品質(低いほど良い)。
_LOG2PI = float(np.log(2 * np.pi))
def gauss_nll(err, sig):
    err = np.asarray(err, float); sig = np.maximum(np.asarray(sig, float), 1e-6)
    m = np.isfinite(err) & np.isfinite(sig)
    if not m.any():
        return float("nan")
    return float(np.mean(0.5 * (err[m] / sig[m]) ** 2 + np.log(sig[m]) + 0.5 * _LOG2PI))


# ================= [U8] 累積σ(ランダムウォーク較正) =================
# σ(dd_row) は「その行の局所信頼度」で、悪ゾーンを長く走った後の蓄積バイアスを無視する。
# ここでは Var(err_row) = base + Σ_{j≤row} ρ(bin_j) の非負累積モデルを LOO で較正する。
#   bin_j = par行の dd ビン(NBIN個)、perp行は専用ビン。ρ は per-row 分散増分(≥0)。
# → σ は「その坑井がここまでどんな地形を通ってきたか(履歴)」に応じて単調増加する。
def _row_bins(dd_row, perp):
    """par行=dd digitize(0..NBIN-1)、perp行=NBIN。"""
    b = np.digitize(dd_row, KBINS[1:-1]).astype(int)
    if PGATE > 0:
        b = np.where(perp, NBIN, b)
    return b

def fit_cumul_sigma(per_well_rows, stride=8):
    """per_well_rows: [(dd_row, perp, spread, err)]。NNLS で (base, rho[bins], rho_spread) を較正。
       Var(err_row) = base + Σ_{j≤row} rho(bin_j) [+ rho_spread·Σ_{j≤row} spread_j]  (SPREAD時)。"""
    NB = NBIN + (1 if PGATE > 0 else 0)
    ncol = 1 + NB + (1 if SPREAD else 0)
    A = np.zeros((ncol, ncol)); bvec = np.zeros(ncol)
    for dd_row, perp, spread, err in per_well_rows:
        b = _row_bins(dd_row, perp)
        onehot = np.zeros((len(b), NB))
        onehot[np.arange(len(b)), b] = 1.0
        cum = np.cumsum(onehot, axis=0)                       # cum[i,k] = ビンk のステップ数(≤i)
        cols = [np.ones(len(b)), cum]
        if SPREAD:
            cols.append(np.cumsum(spread)[:, None])           # 累積ドナー不一致(dd と独立な信頼度軸)
        X = np.column_stack(cols)
        y = err ** 2
        fin = np.isfinite(y)
        X, y = X[fin][::stride], y[fin][::stride]             # 自己相関低減のため間引き
        if len(y):
            A += X.T @ X; bvec += X.T @ y
    rho = np.linalg.lstsq(A, bvec, rcond=None)[0]
    rho = np.maximum(rho, 1e-9)                               # 分散増分は非負(単調性を保証)
    rho_spread = float(rho[1 + NB]) if SPREAD else 0.0
    return float(rho[0]), rho[1:1 + NB], rho_spread           # base_var, rho(per bin), rho_spread

def cumul_sigma_of(dd_row, perp, spread, base, rho, rho_spread):
    b = _row_bins(dd_row, perp)
    var = base + np.cumsum(rho[b])
    if rho_spread:
        var = var + rho_spread * np.cumsum(spread)
    return np.clip(np.sqrt(var), SIG_MIN, SIG_CAP).astype(np.float64)


# ================= σ(dd) 較正(par/perp 別, local モード) =================
def calib_sigma(dd_all, err_all, nbins=N_SIG_BINS):
    ok = np.isfinite(dd_all) & np.isfinite(err_all)
    dd_all, err_all = dd_all[ok], err_all[ok]
    if len(dd_all) < 50:
        return np.array([1000.0]), np.array([SIG_MAX])
    edges = np.quantile(dd_all, np.linspace(0, 1, nbins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    centers, sigs = [], []
    for b in range(nbins):
        m = (dd_all >= edges[b]) & (dd_all < edges[b + 1])
        if m.sum() < 50:
            continue
        centers.append(float(np.median(dd_all[m])))
        sigs.append(float(np.sqrt(np.mean(err_all[m] ** 2))))
    if not centers:
        centers, sigs = [float(np.median(dd_all))], [float(np.sqrt(np.mean(err_all ** 2)))]
    return np.array(centers), np.clip(np.array(sigs), SIG_MIN, SIG_MAX)


def make_sig_of(centers, sigs, centers_p, sigs_p):
    def sig_of(dd_row, perp):
        s = np.clip(np.interp(dd_row, centers, sigs), SIG_MIN, SIG_MAX)
        if PGATE > 0 and perp.any():
            sp = np.clip(np.interp(dd_row, centers_p, sigs_p), SIG_MIN, SIG_MAX)
            s = np.where(perp, sp, s)
        return s.astype(np.float64)
    return sig_of


# ================= parity: v38 完全再現の自己検証 =================
def parity():
    sys.path.insert(0, str(PROJECT / "v38"))
    import struct_prior_v38 as sp38                        # 原典をそのまま実行(定義のみ import)
    train, wname, field, kappa = sp38.build_train_field(str(TRAIN_DIR))
    ref = np.load(V38_ART / "struct_field_v38.npz")
    ok_f = field.shape == ref["field"].shape and np.allclose(field, ref["field"], atol=1e-6)
    ok_k = np.allclose(kappa, ref["kappa"], atol=1e-6)
    print(f"[parity] field {field.shape} == npz: {ok_f}   kappa == npz: {ok_k}")
    print(f"[parity] kappa = {np.round(kappa, 4)}  (npz {np.round(ref['kappa'], 4)})")
    assert ok_f and ok_k, "v38 再現失敗(データ/コードが npz 作成時と不一致)"
    print("[parity] OK — v38 の作り方はこの環境で bit 一致再現できる(比較の土台成立)")


# ================= レポート/比較 =================
def load_v38_baseline(wells, wname):
    """既存 v38 pkl の LOO 軌跡から per-well RMSE を再計算(再ビルド不要の公平基準)。"""
    fp = V38_ART / "struct_prior_train.pkl"
    if not fp.exists():
        return None
    pkl = pickle.loads(fp.read_bytes())
    out = {}
    for wd, nm in zip(wells, wname):
        d = pkl.get(nm)
        if d is None or len(d["tvt"]) != wd["n"] or "R0" not in wd:
            continue
        err = (np.asarray(d["tvt"], float) - wd["anchor"]) - wd["R0"]
        sig = np.asarray(d["sig"], float) if "sig" in d else np.full(len(err), np.nan)
        fin = np.isfinite(err)
        if fin.any():
            out[nm] = (float(np.sqrt(np.mean(err[fin] ** 2))), err[fin], sig[fin])
    return out


def load_hardcore():
    try:
        d = pd.read_csv(PROJECT / "v32" / "artifacts_v32" / "hardcore_diag.csv")
        return set(d[d["hard"] == True].iloc[:, 0].astype(str))
    except Exception:
        return set()


def make_png(per_well66, base, centers, sigs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Meiryo"
        common = sorted(set(per_well66) & set(base))
        a = np.array([base[w][0] for w in common]); b = np.array([per_well66[w] for w in common])
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for arr, lb in [(a, f"v38 (med {np.median(a):.2f})"), (b, f"v66 (med {np.median(b):.2f})")]:
            xs = np.sort(arr); ax[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lb)
        ax[0].set_xscale("log"); ax[0].set_xlabel("per-well LOO struct RMSE [ft]")
        ax[0].set_ylabel("ECDF"); ax[0].legend(); ax[0].set_title("坑井別 LOO RMSE の ECDF")
        ax[1].scatter(a, b, s=8, alpha=0.5)
        lim = [min(a.min(), b.min()) * 0.8, max(a.max(), b.max()) * 1.2]
        ax[1].plot(lim, lim, "k--", lw=1); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("v38"); ax[1].set_ylabel("v66")
        ax[1].set_title(f"坑井別 v38 vs v66(下=改善 {int((b < a).sum())}/{len(a)})")
        ax[2].plot(centers, sigs, "o-", label="v66 σ(dd)")
        try:
            j = json.loads((V38_ART / "struct_sigcurve.json").read_text())
            ax[2].plot(j["centers"], j["sigs"], "s--", label="v38 σ(dd)")
        except Exception:
            pass
        ax[2].set_xlabel("ドナー距離 dd [ft]"); ax[2].set_ylabel("σ [ft]")
        ax[2].legend(); ax[2].set_title("σ(dd) 較正カーブ")
        fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
        print(f"[png] {path}")
    except Exception as e:
        print(f"[png] スキップ({e})")


# ================= main =================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "parity":
        parity(); return

    cfg = dict(seg_len=SEG_LEN, vec=VEC, two_kappa=TWO_KAPPA, qw=QW, robust=ROBUST, aniso=ANISO,
               pgate=PGATE, sigmode=SIGMODE, sig_cap=SIG_CAP, knn_k=KNN_K, knn_h=KNN_H,
               theta0=THETA0, cmax=CMAX, kbins=KBINS,
               kappa_regimes=KAPPA_REGIMES, sig_min=SIG_MIN, sig_max=SIG_MAX, tag=TAG)
    print(f"[v66] config: {json.dumps(cfg)}")

    # ---- train ロード & 場 ----
    wells, wname = [], []
    fps = sorted(glob.glob(str(TRAIN_DIR / "*__horizontal_well.csv")))
    if N_WELLS:
        fps = fps[:N_WELLS]
        print(f"[v66] ★N_WELLS={N_WELLS} スモーク(数値は部分場なので参考値。本判定は全坑井で)")
    for fp in fps:
        wd = load_well(fp, need_tvt=True)
        if wd is None:
            continue
        wd["wi"] = len(wells)
        wells.append(wd); wname.append(os.path.basename(fp).split("__")[0])
    F = build_field(wells)
    n_drop = sum(w["n_drop"] for w in wells)
    print(f"[field] wells {len(wells)}  pts {len(F['x'])}(v38は12,296)  |c|>{CMAX} 破棄 {n_drop}")

    # ---- κ(α/β) ----
    K = fit_kappa(F, wells)
    n_perp = sum(int(w["perp"].sum()) for w in wells); n_rows = sum(w["n"] for w in wells)
    print(f"[kappa] alpha(幾何)  raw {np.round(K['alpha_raw'], 3)} -> use {np.round(K['alpha'], 3)}")
    print(f"[kappa] beta (場)    raw {np.round(K['beta_raw'], 3)} -> use {np.round(K['beta'], 3)}")
    if PGATE > 0:
        print(f"[kappa] perp係数(|cosΔaz|<{PGATE}, 行率 {n_perp / max(n_rows, 1):.2%}): "
              f"alpha_p raw {K['ap_raw']:+.3f} -> {K['ap']:+.3f}   beta_p raw {K['bp_raw']:+.3f} -> {K['bp']:+.3f}"
              f"   (v38 GATE 相当 = 両方 0)")
    print(f"        (v38 単一κ = [0.812 0.722 0.618 0.372 0.109])")

    # ---- LOO 軌跡 & σ較正(par/perp別)----
    dd_all, err_all, perp_all, per_well, loo = [], [], [], {}, {}
    n_mismatch = 0
    for wd, nm in zip(wells, wname):
        resid, dd_row, spread_row = predict_resid(F, wd, K, own=wd["wi"])
        err = resid - wd["R0"]
        fin = np.isfinite(err)
        dd_all.append(dd_row[fin]); err_all.append(err[fin]); perp_all.append(wd["perp"][fin])
        per_well[nm] = float(np.sqrt(np.mean(err[fin] ** 2))) if fin.any() else float("nan")
        loo[nm] = dict(tvt=(wd["anchor"] + resid).astype(np.float64), dd_row=dd_row.astype(np.float64),
                       perp=wd["perp"], err=err.astype(np.float64), spread=spread_row.astype(np.float64))
        if wd["n"] != wd["nev"]:
            n_mismatch += 1
    dd_all = np.concatenate(dd_all); err_all = np.concatenate(err_all); perp_all = np.concatenate(perp_all)
    overall = float(np.sqrt(np.mean(err_all ** 2)))

    # ---- σ 較正(SIGMODE で分岐)----
    centers, sigs = calib_sigma(dd_all[~perp_all], err_all[~perp_all])
    centers_p, sigs_p = calib_sigma(dd_all[perp_all], err_all[perp_all], nbins=4)
    base_var, rho, rho_spread = 0.0, np.zeros(NBIN + (1 if PGATE > 0 else 0)), 0.0
    if SIGMODE == "cumul":
        rows = [(loo[nm]["dd_row"], loo[nm]["perp"], loo[nm]["spread"], loo[nm]["err"]) for nm in loo]
        base_var, rho, rho_spread = fit_cumul_sigma(rows)
        sig_of = lambda dd_row, perp, spread: cumul_sigma_of(dd_row, perp, spread, base_var, rho, rho_spread)
    else:
        _mk = make_sig_of(centers, sigs, centers_p, sigs_p)
        sig_of = lambda dd_row, perp, spread: _mk(dd_row, perp)

    # ---- 錨品質 NLL(par/perp, v38 と比較)----
    sig_v66 = np.concatenate([sig_of(loo[nm]["dd_row"], loo[nm]["perp"], loo[nm]["spread"]) for nm in loo])
    nll_v66 = gauss_nll(err_all, sig_v66)
    nll_par = gauss_nll(err_all[~perp_all], sig_v66[~perp_all])
    nll_perp = gauss_nll(err_all[perp_all], sig_v66[perp_all]) if perp_all.any() else float("nan")

    print(f"\n[LOO] overall struct RMSE = {overall:.4f} ft   (v38 実測 9.709 ft)")
    print(f"[LOO] per-well RMSE  中央値 {np.nanmedian(list(per_well.values())):.3f}  "
          f"p90 {np.nanquantile(list(per_well.values()), 0.9):.3f}")
    print(f"[NLL] SIGMODE={SIGMODE}  overall NLL = {nll_v66:.4f}  (par {nll_par:.4f} / perp {nll_perp:.4f})  ★低いほど良い")
    if n_mismatch:
        print(f"[note] ev行数≠lateral行数の坑井 {n_mismatch} 本(エンジン側で錨OFF扱い。v38と同挙動)")
    if SIGMODE == "cumul":
        print(f"   [cumul] base σ={np.sqrt(base_var):.2f}ft  ρ(per-row var増分, ddビン+perp)={np.round(rho, 4)}")
        if SPREAD:
            print(f"           ρ_spread(ドナー不一致の累積係数)={rho_spread:.5f}  ← >0 なら不一致=新しい信頼度軸")
        print(f"           → σ例: 短坑井端 {np.sqrt(base_var + rho.min()*100):.1f}ft / 長坑井端 {np.sqrt(min(base_var + rho.max()*3000, SIG_CAP**2)):.1f}ft")
    else:
        for cc, ss in zip(centers, sigs):
            print(f"   sigma(par) : dd~{cc:6.0f} ft -> {ss:5.2f} ft")
        if PGATE > 0:
            for cc, ss in zip(centers_p, sigs_p):
                print(f"   sigma(perp): dd~{cc:6.0f} ft -> {ss:5.2f} ft")

    # ---- v38 比較 / hardcore ----
    base = load_v38_baseline(wells, wname)
    cmp_js = {}
    if base:
        common = sorted(set(per_well) & set(base))
        e38 = np.concatenate([base[w][1] for w in common])
        e66 = []
        for wd, nm in zip(wells, wname):
            if nm in base:
                resid = loo[nm]["tvt"] - wd["anchor"]; err = resid - wd["R0"]
                e66.append(err[np.isfinite(err)])
        e66 = np.concatenate(e66)
        ov38 = float(np.sqrt(np.mean(e38 ** 2))); ov66 = float(np.sqrt(np.mean(e66 ** 2)))
        # v38 の錨品質 NLL(pkl の sig で)
        s38 = np.concatenate([base[w][2] for w in common])
        nll38 = gauss_nll(e38, s38)
        s66c = np.concatenate([sig_of(loo[w]["dd_row"], loo[w]["perp"], loo[w]["spread"]) for w in common])
        nll66 = gauss_nll(e66, s66c)
        wins = int(sum(per_well[w] < base[w][0] for w in common))
        hard = load_hardcore() & set(common)
        print(f"\n=== v38 → v66(共通 {len(common)} 坑井, pooled) ===")
        print(f"  overall RMSE: {ov38:.4f} -> {ov66:.4f}  (Δ {ov66 - ov38:+.4f})")
        print(f"  ★錨NLL      : {nll38:.4f} -> {nll66:.4f}  (Δ {nll66 - nll38:+.4f}  低いほど良い)")
        print(f"  坑井別勝敗: v66 改善 {wins}/{len(common)}   中央値 {np.median([base[w][0] for w in common]):.3f}"
              f" -> {np.median([per_well[w] for w in common]):.3f}")
        if hard:
            h38m = float(np.median([base[w][0] for w in hard]))
            h66m = float(np.median([per_well[w] for w in hard]))
            print(f"  hardcore({len(hard)}): per-well median v38 {h38m:.3f} -> v66 {h66m:.3f}")
        dlt = sorted(((per_well[w] - base[w][0], w) for w in common))
        print("  改善 top5 :", [(w, round(d, 2)) for d, w in dlt[:5]])
        print("  悪化 top5 :", [(w, round(d, 2)) for d, w in dlt[-5:]])
        cmp_js = dict(overall_v38=ov38, overall_v66=ov66, nll_v38=nll38, nll_v66=nll66,
                      wins=wins, n_common=len(common))
        make_png(per_well, base, centers, sigs, OUT / f"struct_loo_compare{TAG}.png")

    # ---- 保存(train)。sig_of は SIGMODE 分岐で上で定義済み ----
    out_train = {nm: dict(tvt=d["tvt"], sig=sig_of(d["dd_row"], d["perp"], d["spread"]))
                 for nm, d in loo.items()}
    p_train = OUT / f"prior_{NAME}_train.pkl"
    p_train.write_bytes(pickle.dumps(out_train))
    np.savez(OUT / f"struct_field_{NAME}.npz",
             x=F["x"], y=F["y"], c=F["c"], az=F["az"], wi=F["wi"], qw=F["qw"],
             alpha=K["alpha"], beta=K["beta"], alpha_raw=K["alpha_raw"], beta_raw=K["beta_raw"],
             ap=K["ap"], bp=K["bp"], sigmode=SIGMODE, base_var=base_var, rho=rho, rho_spread=rho_spread,
             centers=centers, sigs=sigs, centers_p=centers_p, sigs_p=sigs_p,
             kbins=np.array(KBINS), config=json.dumps(cfg))
    (OUT / f"struct_sigcurve_{NAME}.json").write_text(json.dumps(dict(
        config=cfg, overall_loo_rmse=overall, overall_nll=nll_v66, v38_compare=cmp_js,
        sigmode=SIGMODE, base_var=base_var, rho=rho.tolist(), rho_spread=rho_spread,
        alpha=K["alpha"].tolist(), beta=K["beta"].tolist(),
        alpha_raw=K["alpha_raw"].tolist(), beta_raw=K["beta_raw"].tolist(),
        alpha_perp=K["ap"], beta_perp=K["bp"], alpha_perp_raw=K["ap_raw"], beta_perp_raw=K["bp_raw"],
        centers=centers.tolist(), sigs=sigs.tolist(),
        centers_p=centers_p.tolist(), sigs_p=sigs_p.tolist(),
        per_well=per_well, n_field_pts=int(len(F["x"]))), indent=1))
    print(f"\n[saved] {p_train}")
    print(f"[saved] {OUT / f'struct_field_{NAME}.npz'} / struct_sigcurve_{NAME}.json")
    if INSTALL and not N_WELLS:
        dst = V38_ART / "priors" / f"prior_{NAME}_train.pkl"
        dst.parent.mkdir(exist_ok=True)
        dst.write_bytes(pickle.dumps(out_train))
        print(f"[install] {dst}  → V38_PRIOR={NAME} で v65 エンジンがこの錨を使う")

    # ---- test(own=-1 全場)----
    if mode in ("both", "test") and TEST_DIR.exists():
        out_test = {}
        for fp in sorted(glob.glob(str(TEST_DIR / "*__horizontal_well.csv"))):
            wd = load_well(fp, need_tvt=False)
            if wd is None:
                continue
            segment_out = segment_well_nofit(wd)
            wd.update(segment_out)
            resid, dd_row, spread_row = predict_resid(F, wd, K, own=-1)
            out_test[os.path.basename(fp).split("__")[0]] = dict(
                tvt=(wd["anchor"] + resid).astype(np.float64),
                sig=sig_of(dd_row, wd["perp"], spread_row))
        p_test = OUT / f"prior_{NAME}_test.pkl"
        p_test.write_bytes(pickle.dumps(out_test))
        print(f"[saved] {p_test}  ({len(out_test)} wells, own=-1)")


def segment_well_nofit(wd):
    """test用: TVT が無いのでランプフィット無しで segid/mid/az だけ作る。"""
    n, A = wd["n"], wd["arc"]; total = float(A[-1])
    if SEG_LEN <= 0:
        K = min(K_FIX, max(1, n // 4))
    else:
        K = int(np.clip(round(total / SEG_LEN), 1, min(K_SEG_MAX, max(1, n // 24))))
    edges = np.linspace(0.0, total, K + 1)
    segid = np.clip(np.searchsorted(edges[1:], A, side="left"), 0, K - 1)
    mid = np.empty((K, 2)); az = np.empty(K)
    Xl, Yl = wd["Xl"], wd["Yl"]
    for j in range(K):
        rows = np.where(segid == j)[0]
        j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)
        az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0])
        mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)
    return dict(segid=segid, mid=mid, az=az)


if __name__ == "__main__":
    main()
