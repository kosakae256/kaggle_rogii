# submit_stack7_v62.ipynb 生成: submit_v52(blend)ベース。cell5 を
# 「全脚計算(重み0のGBM/L4はスキップ)→ stack7_config.json の重みで線形結合」に差し替え。
import json, io, sys, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

nb = json.load(open(r"v52/submit_v52.ipynb", encoding="utf-8"))

cell5 = r'''# ===== 実行: L1-L7 スタック(ローカル確定重み stack7_config.json を適用) =====
t0 = time.time()
v46_build_test_struct(Path(_V26_DATA) / "test")
test_df = build_dataset_v28()
test_df = gf_add_to_testdf(test_df, _V41_STRUCT, Path(_V26_DATA)/"test")
test_df = v50b_add_to_testdf(test_df, Path(_V26_DATA) / "test")
print("test rows:", len(test_df), f"({time.time()-t0:.0f}s)")

_L62_CANDS = [Path("/kaggle/input/rogii-v62-artifacts"),
              Path(r"C:\Users\kosaka256\Documents\rogii_claude\v62\artifacts_v62")]
_L62 = next(p for p in _L62_CANDS if p.exists())
_SCFG = json.loads((_L62 / "stack7_config.json").read_text())
W7 = _SCFG["weights"]; print("stack weights:", W7)

lk  = test_df["last_known_tvt"].to_numpy(np.float64)
mds = np.maximum(test_df["md_since"].to_numpy(np.float64), 0.0)
pf_res = test_df["pf_ancc"].to_numpy(np.float64) - lk
def _apply_pp(res):
    d = res * (1 - PP["w_pf"]) + pf_res * PP["w_pf"]
    if PP.get("tau"): d = d * (1.0 - np.exp(-mds / float(PP["tau"])))
    return lk + d * float(PP.get("alpha", 1.0))
LEGP = {}

# ---- L1 GBM(重み>0のときだけ) ----
if W7.get("GBM", 0) > 0:
    features = json.loads((ART / "features.json").read_text())
    for c in features:
        if c not in test_df.columns: test_df[c] = 0.0
    Xte = test_df[features].astype("float32")
    _mroot = next((ART / m for m in ["models_fwd", "models_bwd", "models"] if (ART / m).exists()), ART / "models")
    preds = []
    for d_ in sorted([Path(p) for p in glob.glob(str(_mroot / "*")) if os.path.isdir(p)]):
        fps = sorted(glob.glob(str(d_ / "fold*.pkl")))
        if fps: preds.append(np.mean([joblib.load(fp).predict(Xte) for fp in fps], axis=0))
    LEGP["GBM"] = _apply_pp(np.mean(preds, axis=0).astype(np.float64))
    print("GBM leg done")

# ---- L2 TCN(v52セルと同一) ----
if W7.get("TCN", 0) > 0:
    _feats29 = PACK29["features"]; _a2r = PACK29["abs2res"]
    _med29 = np.array(PACK29["med"], np.float64); _scl29 = np.array(PACK29["scale"], np.float64)
    _Xnn = np.empty((len(test_df), len(_feats29)), np.float32)
    for _j, _c in enumerate(_feats29):
        _v = test_df[_c].to_numpy(np.float64) if _c in test_df.columns else np.full(len(test_df), np.nan)
        if _a2r[_j]: _v = _v - lk
        _Xnn[:, _j] = (_v - _med29[_j]) / _scl29[_j]
    _Xnn = np.clip(np.nan_to_num(_Xnn, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
    _wc29 = test_df["well"].to_numpy()
    _PREF29 = {}
    if CFG29.get("prefix"):
        _gm = PACK29["prefix_stats"]["gmed"]; _gq = PACK29["prefix_stats"]["giqr"]
        _ts29 = PACK29["t_scale"]; _plen = int(_ARCH["pre_len"])
        for _w in pd.unique(_wc29):
            _fp = Path(_V26_DATA) / "test" / f"{_w}__horizontal_well.csv"
            if not _fp.exists(): continue
            _dfp = pd.read_csv(_fp, usecols=["MD", "Z", "GR", "TVT_input"])
            _kn = _dfp.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(_plen)
            if len(_kn) < 8: continue
            _a = _kn.to_numpy(np.float64)
            _arr = np.stack([np.clip((_a[:, 2] - _gm) / _gq, -8, 8),
                             (_a[:, 3] - _a[-1, 3]) / _ts29,
                             (_a[:, 1] - _a[-1, 1]) / 50.0,
                             (_a[:, 0] - _a[-1, 0]) / 1000.0], axis=1)
            _PREF29[_w] = np.nan_to_num(_arr, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    def _nn_forward29(_xseq, _w):
        _x = torch.from_numpy(np.ascontiguousarray(_xseq.T)[None]).to(DEV29)
        _m = torch.ones(1, 1, _xseq.shape[0], device=DEV29)
        _px = _pm = None
        if CFG29.get("prefix"):
            _pa = _PREF29.get(_w)
            if _pa is None:
                _px = torch.zeros(1, int(_ARCH["pre_c"]), 1, device=DEV29)
                _pm = torch.zeros(1, 1, 1, device=DEV29)
            else:
                _px = torch.from_numpy(np.ascontiguousarray(_pa.T)[None]).to(DEV29)
                _pm = torch.ones(1, 1, len(_pa), device=DEV29)
        _preds = []
        for _mdl in _NN29:
            _o = _mdl(_x, _m, _px, _pm)[0][0].float().cpu().numpy()
            _pl = 1.0 / (1.0 + np.exp(-_o[2]))
            _preds.append(_pl * _o[0] + (1.0 - _pl) * _o[1])
        return np.mean(_preds, axis=0)
    _acc29 = np.zeros(len(test_df), np.float64)
    with torch.no_grad():
        for _w in pd.unique(_wc29):
            _ix = np.where(_wc29 == _w)[0]
            _o29 = _ix[np.argsort(mds[_ix], kind="stable")]
            _p = _nn_forward29(_Xnn[_o29], _w)
            if CFG29.get("flip"):
                _p = 0.5 * (_p + _nn_forward29(np.ascontiguousarray(_Xnn[_o29][::-1]), _w)[::-1])
            _acc29[_o29] = _p
    LEGP["TCN"] = _apply_pp(_acc29 * PACK29["t_scale"] + PACK29["t_mean"])
    print("TCN leg done", f"({time.time()-t0:.0f}s)")

# ---- 共通: seqfeat / 画像用の部材(δ枠) ----
import torch.nn as _nnl
STR5 = 16; NB5 = 90; ND5 = 181; STEP5 = 0.5
OFF5 = (np.arange(ND5) - NB5) * STEP5
_B6 = [f"_{k}" for k in range(1, 7)]
_COLS_D = ([f"pf_ancc{s}" for s in _B6] + [f"pf_self_{k}" for k in (1,2,3)] +
           [f"pf_nbr_{k}" for k in (1,2,3)] + ["beam_med_d", "sc_ens_d", "gf_delta", "v50b_delta"])
_COLS_S = ([f"pf_ancc_std{s}" for s in _B6] + [f"pf_self_std_{k}" for k in (1,2,3)] +
           [f"pf_nbr_std_{k}" for k in (1,2,3)] + ["beam_std_d", None, "gf_std", "v50b_std"])
_DELTA_COLS = {"beam_med_d", "sc_ens_d", "gf_delta", "v50b_delta"}
for c in set([c for c in _COLS_D] + [c for c in _COLS_S if c]):
    if c not in test_df.columns: test_df[c] = np.nan
_rn7 = test_df["id"].astype(str).str.rsplit("_", n=1, expand=True)[1].astype(int).to_numpy()
_gw7 = test_df["well"].astype(str).to_numpy()
DEVL = "cuda" if torch.cuda.is_available() else "cpu"

def _well_frame(wn):
    ii = np.where(_gw7 == wn)[0]
    o = np.argsort(_rn7[ii], kind="stable"); iio = ii[o]
    n = len(iio)
    sp = _V41_STRUCT.get(wn)
    st = (np.asarray(sp["tvt"], float)[:n] if (sp is not None and len(sp["tvt"]) >= n)
          else np.full(n, float(lk[iio[0]])))
    nb_ = (n + STR5 - 1) // STR5
    sel = np.clip(np.arange(nb_) * STR5 + STR5 // 2, 0, n - 1)
    bid = np.minimum(np.arange(n) // STR5, nb_ - 1)
    Ds, Ss = [], []
    for cd, cs in zip(_COLS_D, _COLS_S):
        v = test_df[cd].to_numpy(np.float64)[iio]
        if cd in _DELTA_COLS: v = v + lk[iio]
        dv = np.clip(np.nan_to_num(v - st, nan=0.0), -60, 60)
        sv = (test_df[cs].to_numpy(np.float64)[iio] if cs else np.full(n, 2.5))
        Ds.append(dv); Ss.append(np.clip(np.nan_to_num(sv, nan=3.0), 1.0, 15.0))
    D = np.stack(Ds, 1); S = np.stack(Ss, 1)
    return iio, n, st, nb_, sel, bid, D, S

# ---- L5/L6/L7(seqfeat系) ----
K5 = 16; DIM5 = 64
class L5(_nnl.Module):
    def __init__(self, k=K5, d=DIM5):
        super().__init__()
        self.k = k
        self.enc = _nnl.Sequential(_nnl.Conv1d(2, d, 9, padding=4), _nnl.GELU(),
                                   _nnl.Conv1d(d, d, 9, padding=16, dilation=4), _nnl.GELU(),
                                   _nnl.Conv1d(d, d, 9, padding=64, dilation=16))
        self.emb = _nnl.Embedding(k, d)
        self.qry = _nnl.Sequential(_nnl.Linear(7, d), _nnl.GELU(), _nnl.Linear(d, d))
        self.res = _nnl.Linear(d, 1)
    def core(self, D, S, C):
        B, T, k = D.shape
        med = D.median(2, keepdim=True).values
        x = torch.stack([D - med, torch.log(S)], 2).permute(0, 3, 2, 1).reshape(B * k, 2, T)
        hc = self.enc(x).reshape(B, k, -1, T).permute(0, 3, 1, 2) + self.emb.weight[None, None]
        stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                           (D - med).abs().lt(2).float().mean(2, keepdim=True),
                           torch.zeros_like(med)], -1)
        q = self.qry(stats)
        att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5)
        return att.softmax(-1), q, D
    def forward(self, D, S, C, m):
        w, q, D_ = self.core(D, S, C)
        return (w * D_).sum(-1) + 2.0 * torch.tanh(self.res(q)[..., 0])
class L7(L5):
    def __init__(self):
        super().__init__()
        self.res = None
    def forward(self, D, S, C, m):
        w, q, D_ = self.core(D, S, C)
        return (w * D_).sum(-1)
class L6(_nnl.Module):
    def __init__(self, k=K5, d=DIM5):
        super().__init__()
        cin = (2 * k + 3) * 3
        self.net = _nnl.Sequential(_nnl.Conv1d(cin, d, 5, padding=2), _nnl.GELU(),
                                   _nnl.Conv1d(d, d, 5, padding=8, dilation=4), _nnl.GELU(),
                                   _nnl.Conv1d(d, d, 5, padding=32, dilation=16), _nnl.GELU(),
                                   _nnl.Conv1d(d, d, 5, padding=128, dilation=64))
        self.head = _nnl.Linear(d, 1)
    def forward(self, D, S, C, m):
        med = D.median(2, keepdim=True).values
        base = torch.cat([D - med, torch.log(S), C], -1).transpose(1, 2)
        outs = [base]
        for w_ in (4, 16):
            sm = torch.nn.functional.avg_pool1d(
                torch.nn.functional.pad(base, (w_ // 2, w_ - w_ // 2 - 1), mode="replicate"), w_, stride=1)
            outs.append(base - sm)
        h = self.net(torch.cat(outs, 1)).transpose(1, 2)
        return self.head(h)[..., 0] + med[..., 0]

def _load5(cls, tag):
    ms = []
    for f in range(5):
        net = cls().to(DEVL)
        sd_ = torch.load(_L62 / "nn62" / f"v62_{tag}_f{f}.pt", map_location=DEVL)
        net.load_state_dict(sd_, strict=False)
        net.eval(); ms.append(net)
    return ms

_SEQ_LEGS = [t for t in ("l5", "l6", "l7") if W7.get(t.upper(), 0) > 0]
_MODELS = {t: _load5({"l5": L5, "l6": L6, "l7": L7}[t], t) for t in _SEQ_LEGS}
for t in _SEQ_LEGS: LEGP[t.upper()] = np.zeros(len(test_df), np.float64)
with torch.no_grad():
    for wn in pd.unique(_gw7):
        iio, n, st, nb_, sel, bid, D, S = _well_frame(wn)
        wgt = np.bincount(bid, minlength=nb_).astype(np.float64)
        Dt = torch.from_numpy(D[sel].astype(np.float32))[None].to(DEVL)
        St = torch.from_numpy(S[sel].astype(np.float32))[None].to(DEVL)
        Ct = torch.from_numpy(np.stack([mds[iio][sel] / 1000.0, (lk[iio] - st)[sel] / 10.0,
                                        np.log1p(wgt)], 1).astype(np.float32))[None].to(DEVL)
        mt = torch.ones(1, nb_).to(DEVL)
        for t in _SEQ_LEGS:
            dh = np.mean([m_(Dt, St, Ct, mt)[0, :nb_].cpu().numpy() for m_ in _MODELS[t]], 0)
            LEGP[t.upper()][iio] = st + np.asarray(dh, np.float64)[bid]
print("seq legs done:", _SEQ_LEGS, f"({time.time()-t0:.0f}s)")

# ---- L3img(field-UNet 24ch) ----
if W7.get("L3img", 0) > 0:
    def _smooth61(v, fb, r=2):
        s = pd.Series(v, dtype="float64").interpolate(limit_direction="both").fillna(fb)
        return s.rolling(2 * r + 1, center=True, min_periods=1).mean().to_numpy()
    def _grid_pairs(tvt, gr, gmin, gmax):
        g = np.arange(gmin, gmax + STEP5, STEP5)
        out = np.full(len(g), np.nan)
        idx = np.round((tvt - gmin) / STEP5).astype(int)
        ok = (idx >= 0) & (idx < len(g)) & np.isfinite(gr) & np.isfinite(tvt)
        if ok.sum():
            ssum = np.zeros(len(g)); scnt = np.zeros(len(g))
            np.add.at(ssum, idx[ok], gr[ok]); np.add.at(scnt, idx[ok], 1)
            m = scnt > 0; out[m] = ssum[m] / scnt[m]
        return g, out
    def _winsum(A, W):
        c = np.cumsum(np.concatenate([np.zeros((1,) + A.shape[1:]), A], 0), 0)
        lo = np.clip(np.arange(len(A)) - W // 2, 0, len(A))
        hi = np.clip(np.arange(len(A)) + W - W // 2, 0, len(A))
        return c[hi] - c[lo], (hi - lo).astype(np.float64)
    def _nccimg(g, T, W):
        V = np.isfinite(T); Tz = np.where(V, T, 0.0)
        n, _ = _winsum(V.astype(np.float64), W)
        sg, _ = _winsum(g[:, None] * V, W); st_, _ = _winsum(Tz, W)
        sgg, _ = _winsum((g[:, None] ** 2) * V, W); stt, _ = _winsum(Tz ** 2, W)
        sgt, _ = _winsum(g[:, None] * Tz, W)
        n = np.maximum(n, 1)
        mg = sg / n; mt = st_ / n
        vg = np.maximum(sgg / n - mg ** 2, 1e-6); vt = np.maximum(stt / n - mt ** 2, 1e-6)
        return np.clip((sgt / n - mg * mt) / np.sqrt(vg * vt), -1, 1)
    # 近傍プール(trainのprefix既知行=学習時と同じ構築)
    _TRDIR = Path(_V26_DATA) / "train"
    _KN = {}; _XYS = {}
    for fp in _glob.glob(str(_TRDIR / "*__horizontal_well.csv")):
        wtr = os.path.basename(fp).split("__")[0]
        dtr = pd.read_csv(fp, usecols=["X", "Y", "GR", "TVT_input"])
        knm = dtr["TVT_input"].notna().to_numpy()
        if knm.sum() < 10: continue
        _KN[wtr] = (dtr["TVT_input"].to_numpy(float)[knm], dtr["GR"].to_numpy(float)[knm])
        _XYS[wtr] = (float(dtr["X"].mean()), float(dtr["Y"].mean()))
    from scipy.spatial import cKDTree as _KD
    _tw_names = sorted(_XYS)
    _tree61 = _KD(np.array([_XYS[w] for w in _tw_names]))
    print("nbr pool wells:", len(_tw_names))
    class UNet2D(_nnl.Module):
        def __init__(self, cin=24, ch=128):
            super().__init__()
            def blk(ci, co):
                return _nnl.Sequential(_nnl.Conv2d(ci, co, 3, padding=1), _nnl.GELU(),
                                       _nnl.Conv2d(co, co, 3, padding=1), _nnl.GELU(), _nnl.GroupNorm(8, co))
            c1, c2, c3, c4 = ch // 2, ch, ch, ch
            self.e1 = blk(cin, c1); self.e2 = blk(c1, c2); self.e3 = blk(c2, c3); self.bott = blk(c3, c4)
            self.pool = _nnl.AvgPool2d(2)
            self.u3 = _nnl.ConvTranspose2d(c4, c3, 2, stride=2); self.d3 = blk(c3 + c3, c3)
            self.u2 = _nnl.ConvTranspose2d(c3, c2, 2, stride=2); self.d2 = blk(c2 + c2, c2)
            self.u1 = _nnl.ConvTranspose2d(c2, c1, 2, stride=2); self.d1 = blk(c1 + c1, c1)
            self.gru = _nnl.GRU(c1, ch, batch_first=True, bidirectional=True)
            self.mix = _nnl.Linear(2 * ch, c1)
            self.out = _nnl.Conv2d(c1, 1, 1); self.ref = _nnl.Linear(2 * ch, 1)
        def forward(self, x):
            B, C_, T, Dd = x.shape
            pt = (-T) % 8; pdd = (-Dd) % 8
            if pt or pdd: x = torch.nn.functional.pad(x, (0, pdd, 0, pt))
            h1 = self.e1(x); h2 = self.e2(self.pool(h1)); h3 = self.e3(self.pool(h2)); hb = self.bott(self.pool(h3))
            d3 = self.d3(torch.cat([self.u3(hb), h3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), h1], 1))[:, :, :T, :Dd]
            g = d1.mean(3).transpose(1, 2); g2, _ = self.gru(g)
            h = d1 + self.mix(g2).transpose(1, 2)[:, :, :, None]
            return self.out(h)[:, 0], 2.0 * torch.tanh(self.ref(g2)[:, :, 0])
    NET3 = []
    for f in range(5):
        net = UNet2D().to(DEVL)
        sd_ = torch.load(_L62 / "nn62" / f"v62_field_unet_f{f}.pt", map_location=DEVL)
        net.load_state_dict(sd_, strict=False); net.eval(); NET3.append(net)
    LEGP["L3img"] = np.zeros(len(test_df), np.float64)
    _W16, _W64 = 16, 64
    with torch.no_grad():
        for wn in pd.unique(_gw7):
            iio, n, st, nb_, sel, bid, D, S = _well_frame(wn)
            fp = Path(_V26_DATA) / "test" / f"{wn}__horizontal_well.csv"
            hw_ = pd.read_csv(fp)
            tw_ = pd.read_csv(Path(_V26_DATA) / "test" / f"{wn}__typewell.csv").sort_values("TVT")
            t_tvt = tw_["TVT"].to_numpy(float); t_gr = tw_["GR"].to_numpy(float)
            fin_ = np.isfinite(t_tvt) & np.isfinite(t_gr); t_tvt, t_gr = t_tvt[fin_], t_gr[fin_]
            fb = float(np.nanmean(t_gr))
            kn_ = hw_["TVT_input"].notna().to_numpy(); ev_ = ~kn_
            gr_all = _smooth61(hw_["GR"].to_numpy(float), fb, 2)
            g = gr_all[ev_][:n]
            pos = st[:, None] + OFF5[None, :]
            tws_ = _smooth61(t_gr, fb, 2)
            Ttw = np.interp(pos, t_tvt, tws_)
            gmin, gmax = pos.min() - 1, pos.max() + 1
            kn_tvt = hw_["TVT_input"].to_numpy(float)[kn_]
            gs_, vs_ = _grid_pairs(kn_tvt, gr_all[kn_], gmin, gmax)
            Tself = np.interp(pos, gs_, np.where(np.isfinite(vs_), vs_, np.interp(gs_, t_tvt, tws_)))
            Aself = np.interp(pos, gs_, np.isfinite(vs_).astype(float))
            xq = (float(hw_["X"].mean()), float(hw_["Y"].mean()))
            _, nidx = _tree61.query(xq, k=9)
            js = [_tw_names[j] for j in np.atleast_1d(nidx)][:8]
            if js:
                ny = np.concatenate([_KN[j][0] for j in js]); ng = np.concatenate([_KN[j][1] for j in js])
                gn_, vn_ = _grid_pairs(ny, ng, gmin, gmax)
                Tnbr = np.interp(pos, gn_, np.where(np.isfinite(vn_), vn_, np.interp(gn_, t_tvt, tws_)))
                Anbr = np.interp(pos, gn_, np.isfinite(vn_).astype(float))
            else:
                Tnbr = Ttw; Anbr = np.zeros_like(Ttw)
            resid = gr_all[kn_] - np.interp(kn_tvt, t_tvt, tws_)
            sigw = float(np.nanstd(resid)); sigw = 15.0 if not np.isfinite(sigw) or sigw < 3 else max(sigw, 5.0)
            d16, _ = _winsum((g[:, None] - Ttw), _W16)
            c0 = (d16 / _W16) / sigw
            c1 = _nccimg(g, Ttw, _W16); c2 = _nccimg(g, Ttw, _W64)
            c3 = _nccimg(g, Tself, _W16); c4 = _nccimg(g, Tnbr, _W16)
            a5s, _ = _winsum(Aself, _W16); a5n, _ = _winsum(Anbr, _W16)
            c5 = 0.5 * (a5s / _W16) + 0.5 * (a5n / _W16)
            chs = [np.stack([c[sel] for c in (c0, c1, c2, c3, c4, c5)], 0).astype(np.float32)]
            for kk in range(D.shape[1]):
                cst = D[sel, kk]; sst = S[sel, kk]
                z_ = (OFF5[None, :] - cst[:, None]) / sst[:, None]
                img = np.exp(-0.5 * z_ * z_); img[~np.isfinite(cst)] = 0.0
                chs.append(img[None].astype(np.float32))
            zl = ((lk[iio] - st)[sel])
            z_ = (OFF5[None, :] - zl[:, None]) / 1.0
            chs.append(np.exp(-0.5 * z_ * z_)[None].astype(np.float32))
            EV = np.concatenate(chs, 0)
            x = torch.from_numpy(EV)[None].to(DEVL)
            offs_t = torch.arange(-NB5, NB5 + 1, device=DEVL).float() * STEP5
            dhs = []
            for net in NET3:
                lgt, rf = net(x)
                P = torch.softmax(lgt[0, :nb_], -1)
                dhs.append(((P @ offs_t) + rf[0, :nb_]).cpu().numpy())
            dh = np.mean(dhs, 0)
            LEGP["L3img"][iio] = st + np.asarray(dh, np.float64)[bid]
    print("L3img leg done", f"({time.time()-t0:.0f}s)")

# ---- スタック合成 ----
sw = {k: v for k, v in W7.items() if v > 0 and k in LEGP}
ssum = sum(sw.values())
tvt = sum(sw[k] * LEGP[k] for k in sw) / ssum
print("stacked legs:", {k: round(v, 2) for k, v in sw.items()})
pred_df = pd.DataFrame({"id": test_df["id"].values, "tvt": tvt})
print("predicted rows:", len(pred_df), f"({time.time()-t0:.0f}s)")
'''

nb["cells"][5]["source"] = cell5.splitlines(keepends=True)
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
json.dump(nb, open(r"v62/submit_stack7_v62.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved v62/submit_stack7_v62.ipynb (cells parse OK)")
