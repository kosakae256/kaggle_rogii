# -*- coding: utf-8 -*-
"""
v69/submit_nn_v69.ipynb の test struct を v38(k16 KNN場)→ v66_leanv1cumul(回帰場 VEC+PGATE+累積σ)に差替。
=====================================================================
create/train は v66 錨(prior_v66_leanv1cumul_train.pkl)なので、submit の test 錨も v66 に揃える(train/test整合)。
- field 読込: struct_field_v38.npz → struct_field_v66_leanv1cumul.npz(v66/artifacts_v66 or Kaggle)。
- v46_build_test_struct の中身を v65/submit_pf_v65.ipynb cell[3] の v66 クエリに置換(出力 {wid:{tvt,sig}} 互換)。
出力 {tvt,sig} のインターフェースは v38 と同一 → _attach_anchor 以降は無改変(ユーザー「v38は下位互換」)。
実行: .venv/Scripts/python.exe v69/_patch_submit_v66_struct.py
"""
import json, re, ast, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
NB = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\submit_nn_v69.ipynb")

d = json.load(open(NB, encoding="utf-8"))
c0 = "".join(d["cells"][0]["source"])

# (1) field 読込を v66_leanv1cumul に(v66/artifacts_v66 と Kaggle を探索)
old_load = '''_FZP = _v46_find("struct_field_v38.npz")
_V41_FIELD = np.load(_FZP) if _FZP is not None else None'''
new_load = '''_FZ_V66 = "struct_field_v66_leanv1cumul.npz"
_FZP = _v46_find(_FZ_V66)
if _FZP is None:
    _p66 = Path(r"C:\\Users\\kosaka256\\Documents\\rogii_claude\\v66\\artifacts_v66") / _FZ_V66
    _FZP = _p66 if _p66.exists() else None
_V41_FIELD = np.load(_FZP, allow_pickle=True) if _FZP is not None else None'''
assert old_load in c0, "field 読込行が見つからない"
c0 = c0.replace(old_load, new_load, 1)

# (2) v46_build_test_struct を v66 クエリに全置換(v65 submit cell[3] と同一ロジック)
v66_fn = r'''def v46_build_test_struct(test_dir):
    """test坑井の構造prior(v66_leanv1cumul: 回帰場 VEC + PGATE + 累積σ)を実行時構築し _V41_STRUCT を上書き。
    出力 {wid:{tvt,sig}} は v38版と同一インターフェース(以降の _attach_anchor 等は無改変)。"""
    global _V41_STRUCT
    FZ = _V41_FIELD; assert FZ is not None, "struct_field_v66_leanv1cumul.npz が見つからない"
    FX = FZ["x"].astype(np.float64); FY = FZ["y"].astype(np.float64)
    FC = FZ["c"].astype(np.float64); FAZ = FZ["az"].astype(np.float64)
    ALPHA_ = np.asarray(FZ["alpha"], np.float64); BETA_ = np.asarray(FZ["beta"], np.float64)
    AP_ = float(FZ["ap"]); BP_ = float(FZ["bp"])
    BASEV = float(FZ["base_var"]); RHO = np.asarray(FZ["rho"], np.float64); RHO_SP = float(FZ["rho_spread"])
    KB = np.asarray(FZ["kbins"], np.float64)
    _cfg = json.loads(str(FZ["config"]))
    TH = float(_cfg["theta0"]); KK = int(_cfg["knn_k"]); KH = float(_cfg["knn_h"])
    CMAX = float(_cfg["cmax"]); PGATE = float(_cfg["pgate"]); VEC = int(_cfg["vec"]); SEG_LEN = float(_cfg["seg_len"])
    SMIN = float(_cfg["sig_min"]); SCAP = float(_cfg.get("sig_cap", 40.0))
    NBIN_ = len(KB) - 1; K_FIX = 16; K_SEG_MAX = 64
    C0_ = np.cos(np.radians(TH)); S0_ = np.sin(np.radians(TH))
    FP0 = FX * 0 + np.cos(FAZ - np.radians(TH))
    FUX = np.cos(FAZ); FUY = np.sin(FAZ)

    def _seg_test(Xl, Yl, arc, n):
        total = float(arc[-1])
        if SEG_LEN <= 0:
            Kn = min(K_FIX, max(1, n // 4))
        else:
            Kn = int(np.clip(round(total / SEG_LEN), 1, min(K_SEG_MAX, max(1, n // 24))))
        edges = np.linspace(0.0, total, Kn + 1)
        segid = np.clip(np.searchsorted(edges[1:], arc, side="left"), 0, Kn - 1)
        mid = np.empty((Kn, 2)); az = np.empty(Kn)
        for j in range(Kn):
            rows = np.where(segid == j)[0]
            j0, j1 = (int(rows[0]), int(rows[-1])) if len(rows) else (0, n - 1)
            az[j] = np.arctan2(Yl[j1 + 1] - Yl[j0], Xl[j1 + 1] - Xl[j0])
            mid[j] = ((Xl[j0] + Xl[j1 + 1]) / 2.0, (Yl[j0] + Yl[j1 + 1]) / 2.0)
        return segid, mid, az

    def _query_test(mids):
        Kn = len(mids); g = np.zeros((Kn, 2)); dd = np.empty(Kn); spread = np.zeros(Kn)
        for j in range(Kn):
            d2 = (FX - mids[j, 0]) ** 2 + (FY - mids[j, 1]) ** 2
            kk = min(KK, len(d2) - 1) if len(d2) > 1 else 1
            sel = np.argpartition(d2, kk - 1)[:kk]
            w = np.exp(np.maximum(-d2[sel] / (2 * KH ** 2), -700.0))
            ws = float(w.sum()); dd[j] = np.sqrt(np.median(d2[sel]))
            if ws <= 1e-300:
                continue
            p0 = FP0[sel]; cc = FC[sel]
            s_hat = float((w * cc * p0).sum() / ((w * p0 * p0).sum() + 1e-9))
            gx, gy = s_hat * C0_, s_hat * S0_
            if VEC >= 2:
                ux = FUX[sel]; uy = FUY[sel]
                Sxx = (w * ux * ux).sum(); Sxy = (w * ux * uy).sum(); Syy = (w * uy * uy).sum()
                bx = (w * cc * ux).sum(); by = (w * cc * uy).sum()
                tr = Sxx + Syy; det = Sxx * Syy - Sxy * Sxy; disc = np.sqrt(max(tr * tr - 4 * det, 0.0))
                lam1 = (tr + disc) / 2; lam2 = (tr - disc) / 2; ratio = lam2 / max(lam1, 1e-12)
                lam = lam1 * max(0.05, 0.25 - ratio)
                A11 = Sxx + lam; A22 = Syy + lam; A12 = Sxy; b1 = bx + lam * gx; b2 = by + lam * gy
                detA = A11 * A22 - A12 * A12
                gx = (b1 * A22 - b2 * A12) / max(detA, 1e-12); gy = (b2 * A11 - b1 * A12) / max(detA, 1e-12)
            gn = math.hypot(gx, gy)
            if gn > CMAX: gx *= CMAX / gn; gy *= CMAX / gn
            g[j, 0] = gx; g[j, 1] = gy
            if RHO_SP:
                resd = cc - (gx * FUX[sel] + gy * FUY[sel])
                spread[j] = np.sqrt(max(float((w * resd ** 2).sum() / max(ws, 1e-12)), 0.0))
        return g, dd, spread

    d = {}
    for fp in _glob.glob(str(Path(test_dir) / "*__horizontal_well.csv")):
        wid = os.path.basename(fp).split("__")[0]
        try:
            df = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
            X = df["X"].to_numpy(float); Y = df["Y"].to_numpy(float); Z = df["Z"].to_numpy(float); ti = df["TVT_input"].to_numpy(float)
            fin = np.where(np.isfinite(ti))[0]
            if len(fin) == 0: continue
            s = int(fin.max()); n = len(X) - 1 - s
            if n < 2: continue
            dX = np.diff(X)[s:]; dY = np.diff(Y)[s:]; ndz = -np.diff(Z)[s:]
            Lxy = np.maximum(np.hypot(dX, dY), 1e-3); arc = np.cumsum(Lxy)
            Xl = X[s:]; Yl = Y[s:]
            p_row = (dX * C0_ + dY * S0_) / Lxy
            perp = (np.abs(p_row) < PGATE) if PGATE > 0 else np.zeros(n, bool)
            segid, mid, az = _seg_test(Xl, Yl, arc, n)
            g, dd, spread = _query_test(mid)
            f = g[segid, 0] * dX + g[segid, 1] * dY
            b_row = np.digitize(dd, KB[1:-1])[segid]
            av = ALPHA_[b_row].copy(); bv = BETA_[b_row].copy()
            if PGATE > 0:
                av = np.where(perp, AP_, av); bv = np.where(perp, BP_, bv)
            resid = np.cumsum(av * ndz + bv * f)
            tvt = float(ti[s]) + resid
            dd_row = dd[segid]
            bb = np.digitize(dd_row, KB[1:-1]).astype(int)
            if PGATE > 0: bb = np.where(perp, NBIN_, bb)
            var = BASEV + np.cumsum(RHO[bb])
            if RHO_SP: var = var + RHO_SP * np.cumsum(spread[segid])
            sig = np.clip(np.sqrt(var), SMIN, SCAP)
            d[wid] = dict(tvt=tvt.astype(np.float64), sig=sig.astype(np.float64))
        except Exception:
            continue
    _V41_STRUCT = d
    print(f"[v69] test struct(v66_leanv1cumul)実行時構築: {len(d)} wells")
    return d'''

c0new = re.sub(r'def v46_build_test_struct\(test_dir\):.*?\n    return d\n', v66_fn + "\n", c0, count=1, flags=re.S)
assert c0new != c0 and "v66_leanv1cumul)実行時構築" in c0new, "v46_build_test_struct 置換失敗"
c0 = c0new

# math import 確認(v66クエリで math.hypot 使用)
if "import math" not in c0:
    c0 = c0.replace("import numpy as np", "import numpy as np\nimport math", 1)

d["cells"][0]["source"] = c0.splitlines(keepends=True)
for c in d["cells"]:
    if c["cell_type"] == "code": ast.parse("".join(c["source"]))
json.dump(d, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"patched {NB.name}: test struct → v66_leanv1cumul(回帰場VEC+PGATE+累積σ)")
print("  field: struct_field_v66_leanv1cumul.npz(v66/artifacts_v66 or Kaggle dataset に添付)")
