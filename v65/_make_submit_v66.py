# -*- coding: utf-8 -*-
"""
submit_pf_v65.ipynb / submit_struct20_v65.ipynb の構造prior構築を v38 → v66_leanv1cumul に差し替える。
=====================================================================
変更は cell[3] の「test側 struct 実行時構築」ブロックのみ(PFエンジンcore・実行ループ・submitは不変)。
v66_leanv1cumul の作り方(回帰場 VEC=1 / PGATE perp別係数 / 累積σ)を忠実に移植。
場は struct_field_v66_leanv1cumul.npz(x,y,c,az + alpha/beta/ap/bp + base_var/rho + centers/sigs + config)。
実行: .venv/Scripts/python.exe v65/_make_submit_v66.py
"""
import json, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
V65 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v65")

# ===== v66_leanv1cumul の test側 struct 構築ブロック(cell[3] 前半を置換) =====
V66_BLOCK = r'''# ===== test側キャッシュ + 構造prior(v66_leanv1cumul: 回帰場 VEC + PGATE + 累積σ) =====
NEED_ANCHOR = any(bool(P.get("use_anchor", False)) for P in PARAM_SETS)
STRUCT_T = {}
if NEED_ANCHOR:
    _fz = next((p for p in (_glob.glob("/kaggle/input/**/struct_field_v66_leanv1cumul.npz", recursive=True)
                            + [r"C:\Users\kosaka256\Documents\rogii_claude\v66\artifacts_v66\struct_field_v66_leanv1cumul.npz"]) if os.path.exists(p)), None)
    if _fz is None:
        print("★警告: struct_field_v66_leanv1cumul.npz が見つからない → 錨セットは錨OFFで実行")
    else:
        FZ = np.load(_fz, allow_pickle=True)
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
        FP0 = FX * 0 + np.cos(FAZ - np.radians(TH))   # ドナー方位のθ0射影 cos(az-θ0)
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
                w = np.exp(np.maximum(-d2[sel] / (2 * KH ** 2), -700.0))   # qw=1(QW off)
                ws = float(w.sum()); dd[j] = np.sqrt(np.median(d2[sel]))
                if ws <= 1e-300:
                    continue
                p0 = FP0[sel]; cc = FC[sel]
                s_hat = float((w * cc * p0).sum() / ((w * p0 * p0).sum() + 1e-9))
                gx, gy = s_hat * C0_, s_hat * S0_
                if VEC >= 2:                             # 2Dベクトル場(leanv1cumulはVEC=1で不使用)
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
                if RHO_SP:                                # ドナー不一致(spread変種のみ)
                    resd = cc - (gx * FUX[sel] + gy * FUY[sel])
                    spread[j] = np.sqrt(max(float((w * resd ** 2).sum() / max(ws, 1e-12)), 0.0))
            return g, dd, spread

        for fp in _glob.glob(str(TESTD / "*__horizontal_well.csv")):
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
                # 累積σ: var = base + Σ_{行≤i} rho(bin_j)[+ rho_spread·Σspread]
                dd_row = dd[segid]
                bb = np.digitize(dd_row, KB[1:-1]).astype(int)
                if PGATE > 0: bb = np.where(perp, NBIN_, bb)
                var = BASEV + np.cumsum(RHO[bb])
                if RHO_SP: var = var + RHO_SP * np.cumsum(spread[segid])
                sig = np.clip(np.sqrt(var), SMIN, SCAP)
                STRUCT_T[wid] = dict(tvt=tvt.astype(np.float64), sig=sig.astype(np.float64))
            except Exception:
                continue
        print(f"[struct v66_leanv1cumul] test実行時構築: {len(STRUCT_T)} wells")

'''

MD_NOTE = ("\n> **構造prior = v66_leanv1cumul**(回帰場+PGATE+累積σ, PF A/B で v38 比 −0.226)。"
           "dataset に `struct_field_v66_leanv1cumul.npz` を添付すること(旧 struct_field_v38.npz は不要)。\n")


def patch(nb_path):
    d = json.load(open(nb_path, encoding="utf-8"))
    c3 = d["cells"][3]; s = "".join(c3["source"])
    key = "# --- self/nbr 用:"
    assert key in s, f"{nb_path}: self/nbr アンカーが無い"
    tail = s[s.index(key):]                     # self/nbr プール部(不変)
    c3["source"] = (V66_BLOCK + tail).splitlines(keepends=True)
    # markdown に v66 注記(重複追記しない)
    c0 = d["cells"][0]; ms = "".join(c0["source"])
    if "v66_leanv1cumul" not in ms:
        c0["source"] = (ms + MD_NOTE).splitlines(keepends=True)
    json.dump(d, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 構文チェック
    import ast
    for c in d["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    print(f"patched {Path(nb_path).name}(cell[3] v66化, 全codeセル parse OK)")


for nb in ["submit_pf_v65.ipynb", "submit_struct20_v65.ipynb"]:
    patch(str(V65 / nb))
print("done")
