# _make_submit_v65.py に self/nbr 対応を注入して再生成(1回実行用, リテラル焼き込み版に対応)
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
s = open("v65/_make_submit_v65.py", encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:60])
    s = s.replace(old, new)

# 1) CORE抽出を _bin_pairs から(self/nbr混合のmake_inputsが依存)
rep('i0 = eng.index("def _smooth(")', 'i0 = eng.index("def _bin_pairs(")')

# 2) 焼き込み済みリテラルのJSONブール修正 + v65新キー(tw_source/mix)を追加
rep('{"use_phys": false, "use_anchor": true, ',
    '{"use_phys": False, "use_anchor": True, "tw_source": "tw", "self_mix_w": 1.0, "nbr_mix_w": 1.0, ')

# 3) mdセルに tw_source 説明
rep('- 錨(use_anchor=True)を含むセットがある場合は',
    '- 各セットの `tw_source`(tw/self/nbr/self+nbr)と `self_mix_w`/`nbr_mix_w` で疑似typewell混合を制御\n'
    '  (nbr: 1500ft以内2本未満は自動でtwフォールバック。grid_step は tw/self/nbr の全格子に共通適用)\n'
    '- 錨(use_anchor=True)を含むセットがある場合は')

# 4) cell3: train近傍プール + build_cache_test に nbr_pairs
POOL = r'''# --- self/nbr 用: train近傍プール(nbrセットがある場合のみ構築。test近傍=train坑井, ≤1500ft, ≤3本, 2本未満=tw) ---
MAX_DIST = 1500.0; K_MAX = 3; NEED_REFS = 2
NEED_NBR = any("nbr" in str(P.get("tw_source", "tw")) for P in PARAM_SETS)
TRAIND = DATA / "train"
_TRC = None; _REFC = {}
def _train_centroids():
    global _TRC
    if _TRC is None:
        wids = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TRAIND / "*__horizontal_well.csv"))))
        xy = []
        for w_ in wids:
            h_ = pd.read_csv(TRAIND / f"{w_}__horizontal_well.csv", usecols=["X", "Y"])
            xy.append((float(h_["X"].mean()), float(h_["Y"].mean())))
        _TRC = (wids, np.array(xy))
        print(f"[nbr] trainプール: {len(wids)} wells")
    return _TRC
def _ref_pairs(w_):
    if w_ not in _REFC:
        h_ = pd.read_csv(TRAIND / f"{w_}__horizontal_well.csv", usecols=["GR", "TVT_input"])
        m_ = h_["TVT_input"].notna().to_numpy() & np.isfinite(h_["GR"].to_numpy(float))
        _REFC[w_] = (h_["TVT_input"].to_numpy(float)[m_], h_["GR"].to_numpy(float)[m_])
    return _REFC[w_]
def _nbr_pairs_of(cx, cy):
    wids, xy = _train_centroids()
    d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    order = np.argsort(d)
    refs = [wids[j] for j in order if d[j] <= MAX_DIST][:K_MAX]
    if len(refs) < NEED_REFS: return None
    ts = []; gs_ = []
    for r_ in refs:
        t_, g_ = _ref_pairs(r_); ts.append(t_); gs_.append(g_)
    return (np.concatenate(ts), np.concatenate(gs_))

def build_cache_test(wid):'''
rep("def build_cache_test(wid):", POOL)

rep('''    return dict(md=hw["MD"].to_numpy(float)[ev_pos], z=hw["Z"].to_numpy(float)[ev_pos],
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos, kn_tvtin=kn_tvtin,
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir,
                anc=anc, ancs=ancs, ev_index=np.where(ev.to_numpy())[0], lk=float(kn_tvtin[-1]))''',
    '''    nbr_pairs = _nbr_pairs_of(float(hw["X"].mean()), float(hw["Y"].mean())) if NEED_NBR else None
    return dict(md=hw["MD"].to_numpy(float)[ev_pos], z=hw["Z"].to_numpy(float)[ev_pos],
                gr_filled=gr_filled, ev_pos=ev_pos, kpos=kpos, kn_tvtin=kn_tvtin,
                tw_tvt=tw_tvt, tw_gr=tw_gr, tw_fb=tw_fb, ls=ls, ir=ir,
                anc=anc, ancs=ancs, ev_index=np.where(ev.to_numpy())[0], lk=float(kn_tvtin[-1]),
                nbr_pairs=nbr_pairs)''')

open("v65/_make_submit_v65.py", "w", encoding="utf-8").write(s)
print("builder updated")
