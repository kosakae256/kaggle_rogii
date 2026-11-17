# -*- coding: utf-8 -*-
"""
v47 の train_tabular / train_nn / submit_nn → v69 に移植(パス差替 + submitはPFバンク差替)。
=====================================================================
- train_tabular_v69.ipynb / train_nn_v69.ipynb: **スキーマ駆動**(train.parquet の列をそのまま使う)なので
  バンク数の編集は不要。パス(v47→v69)だけ差し替え。FORCE_PATTERNS は [1-9] ワイルドカードで4バンク対応済。
- submit_nn_v69.ipynb: cell0 が test 用パーケット生成器(create の複製)。**create と同じ差替**
  (PF_LIB 4バンク / PF_BANK / PHYSICS_BANKS / SMOOTH_LAG=192 / per-bank anchor_mult)+ パス。
実行: .venv/Scripts/python.exe v69/_make_train_submit_v69.py
出力: v69/{train_tabular_v69, train_nn_v69, submit_nn_v69}.ipynb
"""
import os, re, ast, json, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJ / "v65"))
import pf_engine_v65 as pf
V47 = PROJ / "v47"; V69 = PROJ / "v69"

# ---- 4バンク PF_LIB literal(create と同一) ----
ART = PROJ / "v65" / "artifacts_v65"
CONFIGS = {
    "af":     ART / "v65pdec10_anchor_faithful_v66_leanv1cumul_current.json",
    "dec10":  ART / "v65dec10_v66_leanv1cumul_tw_current.json",
    "sharp":  ART / "v65pdec10_sharp_matcher_v66_leanv1cumul_current.json",
}
def load_cfg(fp):
    d = json.load(open(fp, encoding="utf-8")); return pf.full_params(d.get("best_params", d.get("params", d)))
PF_LIB = {n: load_cfg(fp) for n, fp in CONFIGS.items()}
def _lit(p): return json.dumps(p, ensure_ascii=False, indent=8).replace(": false", ": False").replace(": true", ": True").replace(": null", ": None")
PFLIB_SRC = "PF_LIB = {\n" + "".join(f'    {json.dumps(n)}: {_lit(p)},\n' for n, p in PF_LIB.items()) + "}"


def path_swap(s):
    """v47 参照 → v69。v48/v38/v41/v29 等の他バージョン参照は変えない。"""
    s = s.replace('artifacts_v47', 'artifacts_v69')
    s = s.replace('/ "v47" /', '/ "v69" /').replace("\\v47\\", "\\v69\\").replace('"v47"', '"v69"')
    s = s.replace('_v47', '_v69')                         # candidates_v47 / submit_v47 等
    s = s.replace('rogii-v47-artifacts', 'rogii-v69-artifacts')
    return s


def bank_swap(s):
    """submit cell0(create複製)を4バンク+lag192+per-bank錨mult に(create の差替と同型)。"""
    # PF_LIB dict
    s2 = re.sub(r"\nPF_LIB = \{.*?\n\}\n", "\n" + PFLIB_SRC + "\n", s, count=1, flags=re.S)
    assert s2 != s and '"af"' in s2, "submit PF_LIB 置換失敗"
    s = s2
    # PF_BANK(7→4)+ per-bank 間引き(dec10の仕組み)
    _blk = '''PF_BANK = ["af", "dec10", "sharp"]   # ★v69: 3バンク。af/dec10のみ錨
DEC_BANK = {"af": 10, "dec10": 10, "sharp": 10}   # ★全バンク MD間引き(=dec10の仕組み, ~10倍速)
def _agg_dec_mean(x, s):
    x = np.asarray(x, float); n = len(x)
    cuts = sorted(set([0, n] + list(range(s, n, s))))
    return np.array([float(x[a:b].mean()) for a, b in zip(cuts[:-1], cuts[1:])])
def _decimate_inp(x, DEC):
    md = np.asarray(x["md"], float)
    if DEC <= 1 or len(md) < DEC * 2:
        y = dict(x); y["_fmd"] = md; y["_dmd"] = md; return y
    y = dict(x)
    y["md"] = _agg_dec_mean(md, DEC); y["z"] = _agg_dec_mean(x["z"], DEC); y["gr"] = _agg_dec_mean(x["gr"], DEC)
    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)
    if "ancs" in x: y["ancs"] = _agg_dec_mean(x["ancs"], DEC)
    y["_fmd"] = md; y["_dmd"] = y["md"]; return y
def run_smoother_bank(inps, P, DEC, seed, n_seeds, chunk):
    if DEC <= 1:
        return run_smoother_ext(inps, P, seed, n_seeds, chunk)
    dinps = [_decimate_inp(x, DEC) for x in inps]
    outs = run_smoother_ext(dinps, P, seed, n_seeds, chunk); res = []
    for i, o in enumerate(outs):
        dm = dinps[i]["_dmd"]; fm = dinps[i]["_fmd"]
        if len(dm) == len(fm): res.append(o)
        else: res.append(dict(mean=np.interp(fm, dm, np.asarray(o["mean"], float)).astype(np.float32),
                              std=np.interp(fm, dm, np.asarray(o["std"], float)).astype(np.float32)))
    return res
'''
    s = re.sub(r'PF_BANK = \["pf_1".*?\n', _blk, s, count=1)
    assert 'PF_BANK = ["af", "dec10", "sharp"]' in s and 'DEC_BANK' in s, "submit PF_BANK/DEC 置換失敗"
    nb = s.count('run_smoother_ext(inps, Pk, SEED_V26, N_SEED, WELL_CHUNK)')
    s = s.replace('run_smoother_ext(inps, Pk, SEED_V26, N_SEED, WELL_CHUNK)',
                  'run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)')
    assert nb == 3, f"submit run_smoother_ext 呼び出しが3箇所でない({nb})"
    # PHYSICS_BANKS
    s = re.sub(r'PHYSICS_BANKS = \{"pfA"\}[^\n]*', 'PHYSICS_BANKS = {"af", "dec10"}   # ★v69', s, count=1)
    assert 'PHYSICS_BANKS = {"af", "dec10"}' in s, "submit PHYSICS_BANKS 置換失敗"
    # SMOOTH_LAG 16→192
    s = re.sub(r'SMOOTH_LAG\s*=\s*16\b', 'SMOOTH_LAG = 192', s, count=1)
    assert 'SMOOTH_LAG = 192' in s, "submit SMOOTH_LAG 置換失敗"
    # per-bank anchor_mult(smoother core)
    s2 = s.replace('* V41_ANCHOR_MULT).clamp_min(1e-6)',
                   '* float(P.get("anchor_mult", V41_ANCHOR_MULT))).clamp_min(1e-6)')
    assert s2 != s, "submit anchor_mult 置換失敗"
    s = s2
    return s


def load_nb(fp): return json.load(open(fp, encoding="utf-8"))
def save_nb(d, fp):
    for c in d["cells"]:
        if c["cell_type"] == "code": ast.parse("".join(c["source"]))
    json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  written {fp.name}  (code cells parse OK)")


# ---- train_tabular / train_nn: パス差替のみ ----
for src, dst in [("train_tabular_v47 copy.ipynb", "train_tabular_v69.ipynb"),
                 ("train_nn_v47 copy.ipynb", "train_nn_v69.ipynb")]:
    d = load_nb(V47 / src)
    for c in d["cells"]:
        c["source"] = path_swap("".join(c["source"])).splitlines(keepends=True)   # ★str/list両対応(char-list化を防ぐ)
    save_nb(d, V69 / dst)

# ---- submit_nn: cell0 バンク差替 + 全セル パス差替 ----
d = load_nb(V47 / "submit_nn_v47.ipynb")
c0 = "".join(d["cells"][0]["source"])
c0 = bank_swap(c0)
d["cells"][0]["source"] = c0.splitlines(keepends=True)
for c in d["cells"]:
    c["source"] = path_swap("".join(c["source"])).splitlines(keepends=True)   # ★str/list両対応(char-list化を防ぐ)
save_nb(d, V69 / "submit_nn_v69.ipynb")

print("\nPF_LIB banks baked:", list(PF_LIB))
print("★注意: submit の test 用 struct場は v47 由来(struct_field_v38.npz)。af/dec10 の錨は本来 v66。")
print("  → 完全整合には test struct を v66 化(v65 submit の runtime 構築)する必要あり=次段で対応。")
