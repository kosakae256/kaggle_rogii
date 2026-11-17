# -*- coding: utf-8 -*-
"""v87 submit_nn ノートブック生成: v69_4 submit_nn を土台に(GBM×NN残差ブレンド維持)、
   - cell1: ART→v87
   - build_dataset_v28() の直後に ★affine 15本を計算し test_df に id merge(GBM/NN 両方が affine を使う)
   - 新cell: affine_lib_v87.py を埋込(Kaggleでimport不可)
   - cell0(engine),2(gf),3(TCN),5(submission) は verbatim
   生成: python v87/_make_submit_nn_v87.py → v87/submit_nn_v87.ipynb
"""
import json, io
from pathlib import Path

P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
nb = json.load(io.open(P / "v69" / "submit_nn_v69_4.ipynb", encoding="utf-8"))
AFFLIB = io.open(P / "v87" / "affine_lib_v87.py", encoding="utf-8").read()
cells = nb["cells"]
c0 = "".join(cells[0]["source"]); c1 = "".join(cells[1]["source"]); c2 = "".join(cells[2]["source"])
c3 = "".join(cells[3]["source"]); c4 = "".join(cells[4]["source"]); c5 = "".join(cells[5]["source"])

# cell1: ART を v87 へ
c1 = c1.replace('Path("/kaggle/input/rogii-v69-artifacts")', 'Path("/kaggle/input/rogii-v87-artifacts")')
c1 = c1.replace(r'Path(r"C:\Users\kosaka256\Documents\rogii_claude\v69\artifacts_v69_4")',
                r'Path(r"C:\Users\kosaka256\Documents\rogii_claude\v87\artifacts_v87")')

# affine lib 埋込cell
c_aff = "# ===== v87 affine コア(affine_lib_v87.py 埋込) =====\n" + AFFLIB

# cell4: build_dataset の直後に affine merge を挿入(NNブレンド以降は verbatim)
_anchor = 'print("test rows:", len(test_df), f"({time.time()-t0:.0f}s)")'
assert _anchor in c4, "cell4 アンカー(test rows print)不一致"
_inject = _anchor + r'''

# --- ★v87 affine 15本(tw/self/nbr × PF_BANK)を test で計算し id merge(GBM/NN 両方が使用) ---
import types, glob as _glob
_ENGKEYS = ["build_smoother_inputs","build_inputs_self","build_inputs_nbr","run_smoother_bank",
            "affine_cal","_smooth_radius_values","_attach_anchor","neighbors_of","_ref_grtvt",
            "DEC_BANK","V26_DEVICE","PF_BANK","PF_LIB","NEED_REFS"]
eng = types.SimpleNamespace(**{k: globals()[k] for k in _ENGKEYS})
_tbase = Path(_V26_DATA) / "test"
_wells = sorted({os.path.basename(q).split("__")[0] for q in _glob.glob(str(_tbase / "*__horizontal_well.csv"))})
_paths = []
for _w in _wells:
    _hp = _tbase / f"{_w}__horizontal_well.csv"; _tp = _tbase / f"{_w}__typewell.csv"
    if not _tp.exists(): continue
    try:
        _hw = pd.read_csv(_hp); _tw = pd.read_csv(_tp).sort_values("TVT")
    except Exception:
        continue
    if build_smoother_inputs(_hw, _tw["TVT"].to_numpy(float), _tw["GR"].to_numpy(float), PF_LIB[PF_BANK[0]]) is None:
        continue
    _paths.append((_hp, _tp))
print("★v87 affine対象 test井:", len(_paths), "lag=", SMOOTH_LAG)
_AFF, _COEF = run_affine(eng, _paths, SMOOTH_LAG=SMOOTH_LAG, N_SEED=N_SEED, SEED_V26=SEED_V26, WELL_CHUNK=WELL_CHUNK, split="test")
def _get_nonaff(ens, wid, nm):
    r = {"F": _FWD_ENS, "S": _SELF_ENS, "N": _NBR_ENS}[ens].get((wid, nm))
    return np.asarray(r["mean"], float) if r else None
_aff_df = assemble_affine_features(eng, _paths, _AFF, _COEF, _get_nonaff)
test_df = test_df.merge(_aff_df, on="id", how="left")
print("★v87 affine特徴 merge:", _aff_df.shape[1] - 1, "列 (test_df:", test_df.shape[1], "列)")'''
c4 = c4.replace(_anchor, _inject, 1)

def mk(src):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": src.splitlines(keepends=True)}

out = dict(nb)
out["cells"] = [mk(c0), mk(c1), mk(c2), mk(c3), mk(c_aff), mk(c4), mk(c5)]   # cell3(TCN)維持、affine cellを4の前に
json.dump(out, io.open(P / "v87" / "submit_nn_v87.ipynb", "w", encoding="utf-8"), ensure_ascii=False)
print("saved v87/submit_nn_v87.ipynb  cells:", len(out["cells"]))
