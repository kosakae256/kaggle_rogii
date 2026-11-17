"""
v38 構造事前分布(GRフリー K16 spline + kernel-kNN dip場)の precompute
====================================================================
検証/rogii_k16_spline_kernel_knn_adaptive_kappa.py の幾何モデルを、
PFの「絶対TVTソフト錨(log-prior)」に使うために各wellの
  - struct_tvt(row)  : 構造モデルの絶対TVT予測(= last_known_tvt + cumsum(kappa*g))
  - sig_struct(row)  : その予測の不確かさ(ドナー距離 dd 別の LOO 残差で較正)
を作り、build_cache が参照できる pkl に保存する。

錨の「強さ」は Optuna で振らない。σ_struct はモデル自身の実誤差(較正済み)で、
priorの分散として物理的に決まる量。iterative_pf_boost_v38 が ANCHOR_MULT(定数/狭グリッド)
だけを掛ける。→ Optunaが強度を極限まで下げる問題を回避。

出力: v38/artifacts_v38/struct_prior_{train|test}.pkl = {well: {"tvt":arr, "sig":arr}}
      v38/artifacts_v38/struct_sigcurve.json        = σ(dd) 較正カーブ + メタ
実行 : rogii_claude/.venv/Scripts/python.exe v38/struct_prior_v38.py           (train LOO)
       rogii_claude/.venv/Scripts/python.exe v38/struct_prior_v38.py test      (test full-field)
"""
import sys, os, glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd

PROJECT = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
OUT = PROJECT / "v38" / "artifacts_v38"; OUT.mkdir(parents=True, exist_ok=True)

# --- K16 幾何モデル定数(原典と同一) ---
THETA0 = 118.4; K = 16; KNN_K, KNN_H = 15, 500.0; GATE = 0.35; FIELD_MIN_PROJ = 0.3
KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]; KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
NBIN = len(KBINS) - 1
# --- σ(dd) 較正の安全域 ---
SIG_MIN, SIG_MAX = 3.0, 40.0   # priorの分散をこの範囲にclip(過信・無効化の両極を防ぐ)
N_SIG_BINS = 8


def _seg(X, Y, s, n):
    edges = np.linspace(0, n, K + 1)
    segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
    mid = np.empty((K, 2)); proj = np.empty(K)
    for j in range(K):
        f0 = s + 1 + int(edges[j]); f1 = min(s + 1 + max(int(edges[j+1]) - 1, int(edges[j])), len(X) - 1)
        az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
        mid[j] = ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2); proj[j] = np.cos(np.radians(az - THETA0))
    return segid, mid, proj


def _knn(field, own, mid, md=0.0):
    keep = field[:, 3] != own
    fx, fy, fD = field[keep, 0], field[keep, 1], field[keep, 2]
    Dh = np.empty(K); dd = np.empty(K)
    for j in range(K):
        d2 = (fx - mid[j, 0]) ** 2 + (fy - mid[j, 1]) ** 2
        cand = np.where(d2 >= md * md)[0] if md else np.arange(len(d2))
        sel = cand[np.argpartition(d2[cand], KNN_K)[:KNN_K]]
        w = np.exp(np.maximum(-d2[sel] / (2 * KNN_H ** 2), -700))
        Dh[j] = np.sum(w * fD[sel]) / np.sum(w); dd[j] = np.sqrt(np.median(d2[sel]))
    return Dh, dd


def build_train_field(train_dir):
    """train全wellから deprojected dip場 と kappa を作る。true TVT を使う(=学習ラベル, 許可)。"""
    train, wname = [], []
    for fp in sorted(glob.glob(os.path.join(train_dir, "*__horizontal_well.csv"))):
        wid = os.path.basename(fp).split("__")[0]
        w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
        tvt, ti = w["TVT"].to_numpy(), w["TVT_input"].to_numpy()
        fin = np.where(np.isfinite(ti))[0]
        if len(fin) == 0:
            continue
        s = int(fin.max()); ndz = -np.diff(Z)[s:]; n = len(ndz)
        if n < 2:
            continue
        U = np.cumsum(ndz); R0 = tvt[s + 1:] - tvt[s]; t = np.arange(1, n + 1.0)
        edges = np.linspace(0, n, K + 1)
        phi = np.column_stack([np.clip(t - edges[j], 0, edges[j+1] - edges[j]) for j in range(K)])
        c = np.linalg.lstsq(phi, R0 - U, rcond=None)[0]
        segid, mid, proj = _seg(X, Y, s, n)
        train.append(dict(wi=len(train), s=s, n=n, ndz=ndz, R0=R0, c=c, segid=segid, mid=mid,
                          proj=proj, anchor=float(ti[s])))
        wname.append(wid)
    field = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                      for wd in train for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
    # kappa: buffered-LOO 同時最小二乗
    A = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
    for R in KAPPA_REGIMES:
        for wd in train:
            Dh, dd = _knn(field, wd["wi"], wd["mid"], md=R)
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
            A += G.T @ G; yv += G.T @ wd["R0"]
    kappa = np.linalg.solve(A, yv)
    return train, wname, field, kappa


def _predict_one(field, own_wi, s, X, Y, Z, ti_s, kappa):
    """1well の struct_tvt(row)・dd(row) を返す(own_wi=-1 で全field使用=test)。"""
    ndz = -np.diff(Z)[s:]; n = len(ndz)
    segid, mid, proj = _seg(X, Y, s, n)
    Dh, dd = _knn(field, own_wi, mid)
    g = ndz + (Dh * proj)[segid]
    g = np.where(np.abs(proj[segid]) < GATE, 0.0, g)
    kstep = kappa[np.digitize(dd, KBINS[1:-1])][segid]
    resid = np.cumsum(kstep * g)
    return ti_s + resid, dd[segid], resid, ndz, n


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    train_dir = str(PROJECT / "rogii-wellbore-geology-prediction" / "train")
    train, wname, field, kappa = build_train_field(train_dir)
    print(f"[field] train wells {len(train)}  field pts {len(field)}  kappa {np.round(kappa,3)}")

    # ---- σ(dd) 較正: 全train LOO 残差を dd別に集計 ----
    dd_all, err_all = [], []
    loo = {}
    for wd, nm in zip(train, wname):
        Dh, dd = _knn(field, wd["wi"], wd["mid"])   # LOO
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        kstep = kappa[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
        resid = np.cumsum(kstep * g)
        dd_row = dd[wd["segid"]]
        err = resid - wd["R0"]
        dd_all.append(dd_row); err_all.append(err)
        loo[nm] = dict(tvt=(wd["anchor"] + resid).astype(np.float64), dd_row=dd_row.astype(np.float64))
    dd_all = np.concatenate(dd_all); err_all = np.concatenate(err_all)
    ok = np.isfinite(dd_all) & np.isfinite(err_all); dd_all, err_all = dd_all[ok], err_all[ok]
    edges = np.quantile(dd_all, np.linspace(0, 1, N_SIG_BINS + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    centers, sigs = [], []
    for b in range(N_SIG_BINS):
        m = (dd_all >= edges[b]) & (dd_all < edges[b + 1])
        if m.sum() < 50:
            continue
        centers.append(float(np.median(dd_all[m])))
        sigs.append(float(np.sqrt(np.mean(err_all[m] ** 2))))
    centers, sigs = np.array(centers), np.clip(np.array(sigs), SIG_MIN, SIG_MAX)
    print("[sigma(dd) calibration]  RMS |struct-true| per donor-distance bin:")
    for cc, ss in zip(centers, sigs):
        print(f"   dd~{cc:6.0f} ft -> sig {ss:5.2f} ft")

    # ★ 提出時(Kaggle)にtest坑井の構造priorを実行時構築するため、train由来のdip場を同梱成果物として保存
    np.savez(OUT / "struct_field_v38.npz", field=field, kappa=kappa,
             centers=centers, sigs=sigs, theta0=THETA0, K=K, knn_k=KNN_K, knn_h=KNN_H,
             gate=GATE, kbins=np.array(KBINS), sig_min=SIG_MIN, sig_max=SIG_MAX)
    print(f"[saved] {OUT/'struct_field_v38.npz'}  (field {field.shape}, test実行時再構築用)")

    def sig_of(dd_row):
        return np.clip(np.interp(dd_row, centers, sigs), SIG_MIN, SIG_MAX)

    # ---- 出力 ----
    if mode == "train":
        out = {nm: dict(tvt=d["tvt"], sig=sig_of(d["dd_row"]).astype(np.float64)) for nm, d in loo.items()}
        path = OUT / "struct_prior_train.pkl"
    else:  # test: full-field(自己除外なし)
        test_dir = str(PROJECT / "rogii-wellbore-geology-prediction" / "test")
        out = {}
        for fp in sorted(glob.glob(os.path.join(test_dir, "*__horizontal_well.csv"))):
            wid = os.path.basename(fp).split("__")[0]
            w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
            X, Y, Z = w["X"].to_numpy(), w["Y"].to_numpy(), w["Z"].to_numpy()
            ti = w["TVT_input"].to_numpy(); s = int(np.where(np.isfinite(ti))[0].max())
            tvt, dd_row, _, _, _ = _predict_one(field, -1, s, X, Y, Z, float(ti[s]), kappa)
            out[wid] = dict(tvt=tvt.astype(np.float64), sig=sig_of(dd_row).astype(np.float64))
        path = OUT / "struct_prior_test.pkl"

    path.write_bytes(pickle.dumps(out))
    (OUT / "struct_sigcurve.json").write_text(json.dumps(dict(
        centers=centers.tolist(), sigs=sigs.tolist(), sig_min=SIG_MIN, sig_max=SIG_MAX,
        kappa=kappa.tolist(), n_wells=len(out),
        overall_loo_rmse=float(np.sqrt(np.mean(err_all ** 2)))), indent=2))
    print(f"\n[saved] {path}  ({len(out)} wells)")
    print(f"[saved] {OUT/'struct_sigcurve.json'}  (overall LOO struct RMSE {np.sqrt(np.mean(err_all**2)):.3f} ft)")


if __name__ == "__main__":
    main()
