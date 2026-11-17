# pf_engine_v65.py に self/neighbor 疑似typewell混合を追加(1回実行用)
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
s = open("v65/pf_engine_v65.py", encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)

# ---- 近傍プール(v47 create と同仕様: XY重心 ≤1500ft, 最大3本, 2本未満=tw fallback) ----
rep('''def _smooth(vals, fb, r):''',
    '''# ---- self/neighbor 疑似typewell(v28/v47 と同仕様) ----
MAX_DIST = 1500.0; K_MAX = 3; NEED_REFS = 2
_TRC = None; _REFC = {}

def _train_centroids():
    """train全坑井のXY重心(初回のみCSV走査、以後 artifacts にキャッシュ)。"""
    global _TRC
    if _TRC is None:
        fp = OUT_DIR / "train_centroids.json"
        if fp.exists():
            d = json.loads(fp.read_text())
            _TRC = (d["wids"], np.array(d["xy"]))
        else:
            wids = list_wells(); xy = []
            for w in wids:
                h = pd.read_csv(DATA_DIR / f"{w}__horizontal_well.csv", usecols=["X", "Y"])
                xy.append((float(h["X"].mean()), float(h["Y"].mean())))
            _TRC = (wids, np.array(xy))
            fp.write_text(json.dumps(dict(wids=wids, xy=[list(p) for p in xy])))
    return _TRC

def _ref_pairs(wid):
    """参照坑井の既知prefix (TVT_input, GR)(raw GR。平滑は make_inputs で hgr_smooth_r)。"""
    if wid not in _REFC:
        h = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        m = h["TVT_input"].notna().to_numpy() & np.isfinite(h["GR"].to_numpy(float))
        _REFC[wid] = (h["TVT_input"].to_numpy(float)[m], h["GR"].to_numpy(float)[m])
    return _REFC[wid]

def _nbr_pairs_of(wid, cx, cy):
    """≤MAX_DISTの近傍(自分除く)最大K_MAX本の(TVT,GR)連結。NEED_REFS未満はNone(=twのまま)。"""
    wids, xy = _train_centroids()
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    refs = []
    for j in np.argsort(d):
        if wids[j] == wid: continue
        if d[j] > MAX_DIST: break
        refs.append(wids[j])
        if len(refs) >= K_MAX: break
    if len(refs) < NEED_REFS: return None
    ts = []; gs_ = []
    for r in refs:
        t_, g_ = _ref_pairs(r); ts.append(t_); gs_.append(g_)
    return (np.concatenate(ts), np.concatenate(gs_))

def _bin_pairs(tvt, gr, gmin, step, G):
    """(TVT,GR)点群 → 格子bin中央値(grid_step共通)。無データbin=NaN。"""
    out = np.full(G, np.nan)
    idx = np.round((np.asarray(tvt, float) - gmin) / step).astype(int)
    ok = (idx >= 0) & (idx < G) & np.isfinite(gr) & np.isfinite(tvt)
    if ok.sum():
        srt = pd.Series(np.asarray(gr, float)[ok]).groupby(idx[ok]).median()
        out[srt.index.to_numpy()] = srt.to_numpy()
    return out

def _smooth(vals, fb, r):''')

# ---- build_cache: XY重心 + nbrペア(必要時のみ) ----
rep('''    return dict(md=ev["MD"].values.astype(np.float64), z=ev["Z"].values.astype(np.float64),
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                anc=anc, ancs=ancs)''',
    '''    nbr_pairs = None
    if "nbr" in os.environ.get("V65_TWSRC", "tw"):
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        nbr_pairs = _nbr_pairs_of(wid, cx, cy)
    return dict(md=ev["MD"].values.astype(np.float64), z=ev["Z"].values.astype(np.float64),
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos,
                kn_tvtin=kn["TVT_input"].values.astype(float),
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir, tvt_true=tv,
                anc=anc, ancs=ancs, nbr_pairs=nbr_pairs)''')

# ---- make_inputs: 混合(σ較正は純twに対して行った後) ----
rep('''    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    if bias != 0.0:
        gr_sm = gr_sm - bias''',
    '''    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    if bias != 0.0:
        gr_sm = gr_sm - bias
    # ---- self/neighbor 疑似typewell混合(σ較正後。grid_step は全格子共通) ----
    src = str(P.get("tw_source", "tw"))
    if src != "tw":
        gg = gg.copy()
        if "nbr" in src and c.get("nbr_pairs") is not None:
            nt, ng = c["nbr_pairs"]
            ngs = _smooth(ng, float(np.nanmean(ng)), P["hgr_smooth_r"]).astype(np.float64)
            gh = _bin_pairs(nt, ngs, gmin, gst, len(gg))
            have = np.isfinite(gh); w = float(P.get("nbr_mix_w", 1.0))
            gg[have] = w * gh[have] + (1 - w) * gg[have]
        if "self" in src:
            gh = _bin_pairs(c["kn_tvtin"], gr_sm[c["kpos"]].astype(np.float64), gmin, gst, len(gg))
            have = np.isfinite(gh); w = float(P.get("self_mix_w", 1.0))
            gg[have] = w * gh[have] + (1 - w) * gg[have]''')

# ---- suggest/full: source は env で固定、mix比は該当時のみ探索 ----
rep('''        rate_target      = trial.suggest_categorical("rate_target", ["zero", "ir"]),
        grid_step        = trial.suggest_categorical("grid_step", [0.2, 0.35, 0.5]),
    )''',
    '''        rate_target      = trial.suggest_categorical("rate_target", ["zero", "ir"]),
        grid_step        = trial.suggest_categorical("grid_step", [0.2, 0.35, 0.5]),
        # ---- self/neighbor(source は env V65_TWSRC で study 単位に固定。混合比のみ探索) ----
        **({"self_mix_w": trial.suggest_float("self_mix_w", 0.1, 1.0)}
           if "self" in os.environ.get("V65_TWSRC", "tw") else {}),
        **({"nbr_mix_w": trial.suggest_float("nbr_mix_w", 0.1, 1.0)}
           if "nbr" in os.environ.get("V65_TWSRC", "tw") else {}),
    )''')
rep('''_V65_DEFAULTS = dict(use_phys=False, use_anchor=True, anchor_mult=20.0, anchor_ramp_md=0.0,
                     lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2)

def full_params(best_params):
    return {**_V65_DEFAULTS, **best_params, "use_phys": USE_PHYS}''',
    '''_V65_DEFAULTS = dict(use_phys=False, use_anchor=True, anchor_mult=20.0, anchor_ramp_md=0.0,
                     lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2,
                     tw_source="tw", self_mix_w=1.0, nbr_mix_w=1.0)

def full_params(best_params):
    return {**_V65_DEFAULTS, **best_params, "use_phys": USE_PHYS,
            "tw_source": os.environ.get("V65_TWSRC", "tw")}''')

open("v65/pf_engine_v65.py", "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print("engine self/nbr mixing added, syntax OK")
