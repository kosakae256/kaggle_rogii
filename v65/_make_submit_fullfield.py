# submit_struct20_v65.ipynb → struct場を完全その場構築(npz不要, train CSV+test prefix)する fullfield 版を生成
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

d = json.load(open("v65/submit_struct20_v65.ipynb", encoding="utf-8"))

d["cells"][0]["source"] = (
    "# v65 生PF提出 — **struct20 + fullfield(struct場を npz 無しで完全その場構築, test既知帯込み)**\n"
    "- パラメータは struct20 best(LB 7.881)と同一。**struct場・κ・σ(dd)較正まで全て実行時に構築**:\n"
    "  train 773本のCSVから場を構築(struct_prior_v38 と同一手順)+ **test既知帯の観測点を追加**し、\n"
    "  κ(buffered-LOO)と σ(dd) を「test増強後の場」の下で較正し直す(npz流用版より整合的)。\n"
    "- **dataset 添付不要**(コンペデータのみで完結)。場の構築 ~2-3分。\n"
    "- INCLUDE_TEST_PREFIX=False にすると train-only 再構築場(npz とほぼ同等)での対照実行も可。"
).splitlines(keepends=True)

# cell3(npzベースのstruct構築)を完全ビルダーに差し替え
c3 = d["cells"][3]; s = "".join(c3["source"])
i0 = s.index("NEED_ANCHOR =")
i1 = s.index("# --- self/nbr 用:")
head = s[:i0]
tail = s[i1:]

FIELD = r'''NEED_ANCHOR = any(bool(P.get("use_anchor", False)) for P in PARAM_SETS)
INCLUDE_TEST_PREFIX = True        # False = train-only 再構築場(対照実験用)
STRUCT_T = {}
if NEED_ANCHOR:
    # ===== struct場の完全その場構築(struct_prior_v38 と同一の数学。npz不要) =====
    THETA0 = 118.4; K = 16; KNN_K, KNN_H = 15, 500.0; GATE = 0.35; FIELD_MIN_PROJ = 0.3
    KBINS = [0.0, 750.0, 1500.0, 2500.0, 4000.0, 1e18]; KAPPA_REGIMES = [0.0, 1000.0, 1500.0, 2000.0]
    NBIN = len(KBINS) - 1; SIG_MIN, SIG_MAX = 3.0, 40.0; N_SIG_BINS = 8
    TRAIND_F = DATA / "train"
    t_f = time.time()

    def _seg(X, Y, s_, n):
        edges = np.linspace(0, n, K + 1)
        segid = np.clip(np.searchsorted(edges[1:], np.arange(1, n + 1.0), side="left"), 0, K - 1)
        mid = np.empty((K, 2)); proj = np.empty(K)
        for j in range(K):
            f0 = s_ + 1 + int(edges[j]); f1 = min(s_ + 1 + max(int(edges[j+1]) - 1, int(edges[j])), len(X) - 1)
            az = np.degrees(np.arctan2(Y[f1] - Y[f0], X[f1] - X[f0])) % 360
            mid[j] = ((X[f0] + X[f1]) / 2, (Y[f1 - 1 if False else f1] + Y[f0]) / 2) if False else ((X[f0] + X[f1]) / 2, (Y[f0] + Y[f1]) / 2)
            proj[j] = np.cos(np.radians(az - THETA0))
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

    # ---- train場(true TVT 使用=学習ラベル, 許可) ----
    train_w = []
    for fp in sorted(_glob.glob(str(TRAIND_F / "*__horizontal_well.csv"))):
        w = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        X = w["X"].to_numpy(); Y = w["Y"].to_numpy(); Z = w["Z"].to_numpy()
        tvt = w["TVT"].to_numpy(); ti = w["TVT_input"].to_numpy()
        fin = np.where(np.isfinite(ti))[0]
        if len(fin) == 0: continue
        s_ = int(fin.max()); ndz = -np.diff(Z)[s_:]; n = len(ndz)
        if n < 2: continue
        U = np.cumsum(ndz); R0 = tvt[s_ + 1:] - tvt[s_]; t_ = np.arange(1, n + 1.0)
        edges = np.linspace(0, n, K + 1)
        phi = np.column_stack([np.clip(t_ - edges[j], 0, edges[j+1] - edges[j]) for j in range(K)])
        c_ = np.linalg.lstsq(phi, R0 - U, rcond=None)[0]
        segid, mid, proj = _seg(X, Y, s_, n)
        train_w.append(dict(wi=len(train_w), ndz=ndz, R0=R0, c=c_, segid=segid, mid=mid, proj=proj))
    FLD = np.array([(wd["mid"][j, 0], wd["mid"][j, 1], wd["c"][j] / wd["proj"][j], wd["wi"])
                    for wd in train_w for j in range(K) if abs(wd["proj"][j]) > FIELD_MIN_PROJ])
    n_train_pts = len(FLD)

    # ---- test既知帯の観測点(TVT_input=与えられた入力。水平区間のみ) ----
    n_test_pts = 0
    if INCLUDE_TEST_PREFIX:
        _MIN_LAT = 150
        _pts = []; _wi0 = len(train_w) + 10
        for _k, _fp in enumerate(sorted(_glob.glob(str(Path(TESTD) / "*__horizontal_well.csv")))):
            try:
                _df = pd.read_csv(_fp, usecols=["X", "Y", "Z", "TVT_input"])
                _X = _df["X"].to_numpy(); _Y = _df["Y"].to_numpy(); _Z = _df["Z"].to_numpy()
                _ti = _df["TVT_input"].to_numpy()
                _fin = np.where(np.isfinite(_ti))[0]
                if len(_fin) < _MIN_LAT + 2: continue
                _s = int(_fin.max())
                _dxy = np.hypot(np.diff(_X[:_s + 1]), np.diff(_Y[:_s + 1])); _dz = np.diff(_Z[:_s + 1])
                _idx = np.where(_dxy > 2.0 * np.abs(_dz))[0]
                if len(_idx) < _MIN_LAT: continue
                _j0 = int(_idx[0]); _nn = _s - _j0
                if _nn < _MIN_LAT: continue
                _ndz = -np.diff(_Z)[_j0:_s]
                _U = np.cumsum(_ndz); _R0 = _ti[_j0 + 1:_s + 1] - _ti[_j0]
                _m = np.isfinite(_R0)
                if _m.sum() < _MIN_LAT: continue
                _Kp = int(np.clip(_nn // 200, 1, 8))
                _edges = np.linspace(0, len(_ndz), _Kp + 1); _t = np.arange(1, len(_ndz) + 1.0)
                _phi = np.column_stack([np.clip(_t - _edges[_j], 0, _edges[_j+1] - _edges[_j]) for _j in range(_Kp)])
                _c = np.linalg.lstsq(_phi[_m], (_R0 - _U)[_m], rcond=None)[0]
                for _j in range(_Kp):
                    _f0 = _j0 + 1 + int(_edges[_j]); _f1 = min(_j0 + 1 + max(int(_edges[_j+1]) - 1, int(_edges[_j])), len(_X) - 1)
                    _az = np.degrees(np.arctan2(_Y[_f1] - _Y[_f0], _X[_f1] - _X[_f0])) % 360
                    _pj = np.cos(np.radians(_az - THETA0))
                    if abs(_pj) > FIELD_MIN_PROJ and np.isfinite(_c[_j]):
                        _pts.append(((_X[_f0] + _X[_f1]) / 2, (_Y[_f0] + _Y[_f1]) / 2, float(_c[_j] / _pj), _wi0 + _k))
            except Exception:
                continue
        if _pts:
            FLD = np.vstack([FLD, np.array(_pts, np.float64)]); n_test_pts = len(_pts)

    # ---- κ: buffered-LOO 最小二乗(test増強後の場で較正。train自坑井のみ除外) ----
    A_ = np.zeros((NBIN, NBIN)); yv = np.zeros(NBIN)
    for R_ in KAPPA_REGIMES:
        for wd in train_w:
            Dh, dd = _knn(FLD, wd["wi"], wd["mid"], md=R_)
            g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
            g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
            bstep = np.digitize(dd, KBINS[1:-1])[wd["segid"]]
            G = np.column_stack([np.cumsum(np.where(bstep == b, g, 0.0)) for b in range(NBIN)])
            A_ += G.T @ G; yv += G.T @ wd["R0"]
    KAPPA = np.linalg.solve(A_, yv)

    # ---- σ(dd): LOO残差(同じく増強後の場で) ----
    dd_all, err_all = [], []
    for wd in train_w:
        Dh, dd = _knn(FLD, wd["wi"], wd["mid"])
        g = wd["ndz"] + (Dh * wd["proj"])[wd["segid"]]
        g = np.where(np.abs(wd["proj"][wd["segid"]]) < GATE, 0.0, g)
        kstep = KAPPA[np.digitize(dd, KBINS[1:-1])][wd["segid"]]
        resid = np.cumsum(kstep * g)
        dd_all.append(dd[wd["segid"]]); err_all.append(resid - wd["R0"])
    dd_all = np.concatenate(dd_all); err_all = np.concatenate(err_all)
    ok_ = np.isfinite(dd_all) & np.isfinite(err_all); dd_all, err_all = dd_all[ok_], err_all[ok_]
    edges_ = np.quantile(dd_all, np.linspace(0, 1, N_SIG_BINS + 1)); edges_[0] = -np.inf; edges_[-1] = np.inf
    CEN, SGC = [], []
    for b in range(N_SIG_BINS):
        m_ = (dd_all >= edges_[b]) & (dd_all < edges_[b + 1])
        if m_.sum() < 50: continue
        CEN.append(float(np.median(dd_all[m_]))); SGC.append(float(np.sqrt(np.mean(err_all[m_] ** 2))))
    CEN = np.array(CEN); SGC = np.clip(np.array(SGC), SIG_MIN, SIG_MAX)
    print(f"[fullfield] train {n_train_pts}点 + test既知帯 {n_test_pts}点 = {len(FLD)}点 | "
          f"kappa {np.round(KAPPA, 3)} ({time.time()-t_f:.0f}s)")
    print("  sigma(dd):", " ".join(f"{c_:.0f}ft->{s_:.1f}" for c_, s_ in zip(CEN, SGC)))

    # ---- test各坑井の struct prior(own除外なし=自分のprefix点も使う) ----
    for fp in _glob.glob(str(Path(TESTD) / "*__horizontal_well.csv")):
        wid = os.path.basename(fp).split("__")[0]
        try:
            df = pd.read_csv(fp, usecols=["X", "Y", "Z", "TVT_input"])
            X = df["X"].to_numpy(); Y = df["Y"].to_numpy(); Z = df["Z"].to_numpy(); ti = df["TVT_input"].to_numpy()
            fin = np.where(np.isfinite(ti))[0]
            if len(fin) == 0: continue
            s_ = int(fin.max()); ndz = -np.diff(Z)[s_:]; n = len(ndz)
            if n < 2: continue
            segid, mid, proj = _seg(X, Y, s_, n)
            Dh, dd = _knn(FLD, -1, mid)
            g = ndz + (Dh * proj)[segid]
            g = np.where(np.abs(proj[segid]) < GATE, 0.0, g)
            kstep = KAPPA[np.digitize(dd, KBINS[1:-1])][segid]
            tvt = float(ti[s_]) + np.cumsum(kstep * g)
            sig = np.clip(np.interp(dd[segid], CEN, SGC), SIG_MIN, SIG_MAX)
            STRUCT_T[wid] = dict(tvt=tvt.astype(np.float64), sig=sig.astype(np.float64))
        except Exception:
            continue
    print(f"[struct] test実行時構築(fullfield): {len(STRUCT_T)} wells")

'''
c3["source"] = (head[:head.index("STRUCT_T = {}")] if "STRUCT_T = {}" in head else head)
c3["source"] = (FIELD + tail).splitlines(keepends=True)

json.dump(d, open("v65/submit_struct20_fullfield_v65.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import ast
for c in d["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("written v65/submit_struct20_fullfield_v65.ipynb, all cells parse OK")
