# -*- coding: utf-8 -*-
"""v75_2 生成器 — v75 に追加PFソース(self0.5 / nbr0.5 / self0.5+nbr0.5)を足した create/train/submit を生成。
ユーザー指定(2026-07-21夜): 既存 tw+self(1.0)/tw+nbr(1.0) に加え、mix0.5系3種を特徴量として追加。
命名: artifacts_v75_2 / create_candidates_v75_2 / train_*_v75_2 / submit_nn_v75_2。
全パッチ完全一致assert。新列: pf_self50/pf_nbr50/pf_sn50 {,_std,_delta,_vs_tw}×5バンク=60列(train系はスキーマ駆動で自動取込)。"""
import io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V75 = P0 / "v75"


def rep(s, old, new, cnt=1, tag=""):
    k = s.count(old)
    assert k == cnt, f"[{tag}] 期待{cnt}回が{k}回: {old[:70]!r}"
    return s.replace(old, new)


# ================== 共通パッチ(create本文 と submit cell0 の両方に適用) ==================
def patch_engine(s, tag):
    # (a) mixer のパラメータ化
    s = rep(s, 'def build_inputs_self(hw, tw_tvt, tw_gr, P):',
            'def build_inputs_self(hw, tw_tvt, tw_gr, P, mixw=None):', 1, tag + "-selfsig")
    s = rep(s, 'gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]',
            '_mw = SELF_MIX_W if mixw is None else float(mixw)\n'
            '    gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = _mw * gh[have] + (1 - _mw) * gg[have]', 1, tag + "-selfmix")
    s = rep(s, 'def build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs):',
            'def build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs, mixw=None):', 1, tag + "-nbrsig")
    s = rep(s, 'gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = NBR_MIX_W * gh[have] + (1 - NBR_MIX_W) * gg[have]',
            '_mw = NBR_MIX_W if mixw is None else float(mixw)\n'
            '        gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = _mw * gh[have] + (1 - _mw) * gg[have]', 1, tag + "-nbrmix")
    # (b) 複合ソース build_inputs_selfnbr(nbr→self の順で重ねる)
    s = rep(s, '''        base = dict(base); base["gg"] = gg
    return base
''',
            '''        base = dict(base); base["gg"] = gg
    return base

def build_inputs_selfnbr(hw, tw_tvt, tw_gr, P, refs, sw=0.5, nw=0.5):
    """v75_2: tw に nbr(nw)→self(sw) の順で疑似typewellを重ねる(重複binはself後勝ち再混合)。"""
    base = build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs, mixw=nw)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = float(sw) * gh[have] + (1 - float(sw)) * gg[have]
    base = dict(base); base["gg"] = gg; return base
''', 1, tag + "-selfnbr")
    # (c) 新ensグローバル
    s = rep(s, '_SELF_ENS = {}; _NBR_ENS = {}; _ZGRAD = {}; _NBR_META = {}',
            '_SELF_ENS = {}; _NBR_ENS = {}; _ZGRAD = {}; _NBR_META = {}\n'
            '_SELF50_ENS = {}; _NBR50_ENS = {}; _SN50_ENS = {}    # v75_2 追加ソース(mix0.5)', 1, tag + "-ens")
    # (d) キャッシュ復元/保存に新ソース追加
    s = rep(s, '_dst = {"F": _FWD_ENS, "S": _SELF_ENS, "N": _NBR_ENS}[_ens]',
            '_dst = {"F": _FWD_ENS, "S": _SELF_ENS, "N": _NBR_ENS, "S5": _SELF50_ENS, "N5": _NBR50_ENS, "B5": _SN50_ENS}[_ens]',
            1, tag + "-crest")
    s = rep(s, 'for _ens, _dic in (("F", _FWD_ENS), ("S", _SELF_ENS), ("N", _NBR_ENS)):',
            'for _ens, _dic in (("F", _FWD_ENS), ("S", _SELF_ENS), ("N", _NBR_ENS),\n'
            '                           ("S5", _SELF50_ENS), ("N5", _NBR50_ENS), ("B5", _SN50_ENS)):', 1, tag + "-csave")
    # (e) 追加3ソースの smoother 実行ループ(nbrループ後・キャッシュ保存前)
    s = rep(s, '    # ★v41: PFパス結果を保存(train生成時のみ。testはcache無効=保存もしない)',
            '''    # ==== v75_2: 追加ソース self(0.5)/nbr(0.5)/self(0.5)+nbr(0.5) × 各バンク ====
    if not _PF_CACHED:
        _SELF50_ENS.clear(); _NBR50_ENS.clear(); _SN50_ENS.clear()
    for p in ([] if _PF_CACHED else P_SETS):
        Pk = _praw(p)
        print(f"[smoothPF-mix50 self/nbr/sn] {p['label']} ..."); t0 = time.perf_counter()
        in1 = []; in2 = []; in3 = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            _tt = tw["TVT"].to_numpy(float); _tg = tw["GR"].to_numpy(float)
            _x1 = build_inputs_self(hw, _tt, _tg, Pk, mixw=0.5)
            _x2 = build_inputs_nbr(hw, _tt, _tg, Pk, NREFS[wid], mixw=0.5)
            _x3 = build_inputs_selfnbr(hw, _tt, _tg, Pk, NREFS[wid], sw=0.5, nw=0.5)
            for _x, _lst in ((_x1, in1), (_x2, in2), (_x3, in3)):
                _lst.append(_attach_v75(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)), wid, use_sim=False))
        for _lst, _dic in ((in1, _SELF50_ENS), (in2, _NBR50_ENS), (in3, _SN50_ENS)):
            outs = run_smoother_bank(_lst, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)
            for i, w in enumerate(names): _dic[(w, p["name"])] = outs[i]
            if V26_DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # ★v41: PFパス結果を保存(train生成時のみ。testはcache無効=保存もしない)''', 1, tag + "-loops")
    # (f) 列配線(per-bank agree_range の直後)
    s = rep(s, '''            df[f"pf_agree_range{suf}"] = (np.nanmax(st, axis=1) - np.nanmin(st, axis=1)).astype(np.float32)''',
            '''            df[f"pf_agree_range{suf}"] = (np.nanmax(st, axis=1) - np.nanmin(st, axis=1)).astype(np.float32)
            for _t2, _d2 in (("self50", _SELF50_ENS), ("nbr50", _NBR50_ENS), ("sn50", _SN50_ENS)):   # v75_2
                _o2 = _d2.get((w, nm))
                _m2 = col(_o2["mean"] if _o2 else None); _s2 = col(_o2["std"] if _o2 else None)
                df[f"pf_{_t2}{suf}"] = _m2; df[f"pf_{_t2}_std{suf}"] = _s2
                df[f"pf_{_t2}_delta{suf}"] = (_m2 - lk).astype(np.float32)
                df[f"pf_{_t2}_vs_tw{suf}"] = (_m2 - twk).astype(np.float32)''', 1, tag + "-cols")
    return s


# ================== create ==================
src = (V75 / "create_candidates_v75.py").read_text(encoding="utf-8")
src = rep(src, 'artifacts_v75', 'artifacts_v75_2', src.count('artifacts_v75'), "c-path")
src = rep(src, 'pfpass_cache_v75.npz', 'pfpass_cache_v75_2.npz', 1, "c-cache")
src = patch_engine(src, "create")
(V75 / "create_candidates_v75_2.py").write_text(src, encoding="utf-8")
print(f"[create] create_candidates_v75_2.py 生成 ({len(src)}字)")

# ================== train ノート(パス差替) ==================
for s_ in ["train_tabular_v75.ipynb", "train_nn_v75.ipynb"]:
    nb = json.loads((V75 / s_).read_text(encoding="utf-8")); n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code": continue
        t = "".join(cell["source"])
        if "artifacts_v75" in t:
            cell["source"] = t.replace("artifacts_v75", "artifacts_v75_2").splitlines(keepends=True); n += 1
    d = s_.replace("_v75", "_v75_2")
    (V75 / d).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[train] {d} (パス差替 {n}セル)")

# ================== submit ==================
nb = json.loads((V75 / "submit_nn_v75.ipynb").read_text(encoding="utf-8"))
c0 = "".join(nb["cells"][0]["source"])
c0 = patch_engine(c0, "submit")
nb["cells"][0]["source"] = c0.splitlines(keepends=True)
nhit = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code": continue
    t = "".join(cell["source"])
    if "artifacts_v75" in t and "artifacts_v75_2" not in t:
        cell["source"] = t.replace("artifacts_v75", "artifacts_v75_2").splitlines(keepends=True); nhit += 1
(V75 / "submit_nn_v75_2.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[submit] submit_nn_v75_2.ipynb 生成 (パス差替 {nhit}セル)")

# ================== 検証 ==================
import py_compile
py_compile.compile(str(V75 / "create_candidates_v75_2.py"), doraise=True)
t = (V75 / "create_candidates_v75_2.py").read_text(encoding="utf-8")
nb2 = json.loads((V75 / "submit_nn_v75_2.ipynb").read_text(encoding="utf-8"))
for i, c in enumerate(nb2["cells"]):
    if c.get("cell_type") == "code":
        compile("".join(c["source"]), f"cell{i}", "exec")
full2 = "\n".join("".join(c["source"]) for c in nb2["cells"] if c.get("cell_type") == "code")
checks = [
    ("create: 3ソースloop", "[smoothPF-mix50 self/nbr/sn]" in t),
    ("create: 列60本配線", 'pf_{_t2}_vs_tw{suf}' in t),
    ("create: selfnbr複合", "def build_inputs_selfnbr" in t),
    ("create: cache v75_2", "pfpass_cache_v75_2.npz" in t),
    ("create: OUT=v75_2", 'artifacts_v75_2" / "data"' in t.replace(" / ", ' / ') or "artifacts_v75_2" in t),
    ("create: v73部品維持", "rtkq[:, i][:, None] * (rtq[:, i][:, None] - rate)" in t and "V75_WNN * sv" in t),
    ("submit: 3ソースloop+列", "[smoothPF-mix50 self/nbr/sn]" in full2 and 'pf_{_t2}_vs_tw{suf}' in full2),
    ("submit: test sim機構維持", "_v75_build_test_sims" in full2),
    ("submit: ART=v75_2", "artifacts_v75_2" in full2),
]
for lbl, ok in checks: print(("  OK " if ok else "  ×  ") + lbl)
assert all(ok for _, ok in checks)
print("[build_v75_2] 完了(compile OK)")
