# -*- coding: utf-8 -*-
"""
v47/create_candidates_v47.py → v69/create_candidates_v69.py を生成。
=====================================================================
方針(ユーザー 2026-07-15): v47 と同じ特徴量(v9 build_well + PF banks self/nbr/tw)を作るが、
**PF出力部分だけ指定の4バンク(af/dec10/sharp/smooth)に差し替え、smooth PF lag=192 で回す**。
- PF params は各バンクをハードコード(v65のチューニング済 json から取り込み)。**追加は PF_LIB + PF_BANK に足すだけ**。
- 錨(struct)= v66_leanv1cumul(4バンクは v66 でチューニング)。**anchor_mult は per-bank**(af/dec10のみ錨ON)。
- 他(v9特徴・smootherコア・self/nbr/z_grad・parquet列)は v47 verbatim(バンク数非依存で自動 pf_*_1..4 になる)。
実行: .venv/Scripts/python.exe v69/_make_create_v69.py
出力: v69/create_candidates_v69.py
"""
import os, re, ast, json, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
sys.path.insert(0, str(PROJ / "v65"))
import pf_engine_v65 as pf                         # V38_PRIOR 既定 = v66_leanv1cumul

ART = PROJ / "v65" / "artifacts_v65"
CONFIGS = {                                        # bank名 -> param json(best_params/params/raw)
    "af":     ART / "v65pdec10_anchor_faithful_v66_leanv1cumul_current.json",
    "dec10":  ART / "v65dec10_v66_leanv1cumul_tw_current.json",
    "sharp":  ART / "v65pdec10_sharp_matcher_v66_leanv1cumul_current.json",
}

def load_cfg(fp):
    d = json.load(open(fp, encoding="utf-8"))
    return pf.full_params(d.get("best_params", d.get("params", d)))

# --- PF_LIB literal(4バンク, ハードコード) ---
PF_LIB = {name: load_cfg(fp) for name, fp in CONFIGS.items()}
def _lit(p):
    return json.dumps(p, ensure_ascii=False, indent=8).replace(": false", ": False").replace(": true", ": True").replace(": null", ": None")
pflib_src = "PF_LIB = {\n" + "".join(f'    {json.dumps(n)}: {_lit(p)},\n' for n, p in PF_LIB.items()) + "}"

txt = (PROJ / "v47" / "create_candidates_v47.py").read_text(encoding="utf-8")

# (1) PF_LIB dict を4バンクに全置換
txt2 = re.sub(r"\nPF_LIB = \{.*?\n\}\n", "\n" + pflib_src + "\n", txt, count=1, flags=re.S)
assert txt2 != txt and '"af"' in txt2, "PF_LIB 置換失敗"
txt = txt2

# (2) 錨 struct を v66 に / PHYSICS_BANKS / PF_BANK
txt = txt.replace('_V41_STRUCT = _pickle.loads((_V38_ART / "struct_prior_train.pkl").read_bytes())',
                  '_V41_STRUCT = _pickle.loads((_V38_ART / "priors" / "prior_v66_leanv1cumul_train.pkl").read_bytes())  # ★v69: v66錨')
assert txt.count('prior_v66_leanv1cumul_train.pkl') >= 1, "struct pkl 置換失敗"
txt = txt.replace('PHYSICS_BANKS = {"pfA"}   # ★v46: このバンクだけ struct錨ON。他(r0_seed32/pf_3)は物理無し=錨OFF。',
                  'PHYSICS_BANKS = {"af", "dec10"}   # ★v69: use_anchor=True のバンクのみ錨ON(sharp/smoothは錨OFF)')
assert 'PHYSICS_BANKS = {"af", "dec10"}' in txt, "PHYSICS_BANKS 置換失敗"
_pf_bank_block = '''PF_BANK = ["af", "dec10", "sharp"]   # ★v69: 3バンク×tw/self/nbr=9候補。af/dec10のみ錨。追加はここ+PF_LIB(+錨ならPHYSICS_BANKS)
# ★dec10の仕組み: 全バンク(af/dec10/sharp=MD-10でチューニング)は MD間引きして smooth PF。追加バンクは既定1。
DEC_BANK = {"af": 10, "dec10": 10, "sharp": 10}
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
    """★dec10の本質: DEC>1 なら MD間引き(GR mean=denoise, overconfidence減, ~DEC倍速)→ smooth PF → 全解像度に補間。DEC=1 は素通し。"""
    if DEC <= 1:
        return run_smoother_ext(inps, P, seed, n_seeds, chunk)
    dinps = [_decimate_inp(x, DEC) for x in inps]
    outs = run_smoother_ext(dinps, P, seed, n_seeds, chunk)
    res = []
    for i, o in enumerate(outs):
        dm = dinps[i]["_dmd"]; fm = dinps[i]["_fmd"]
        if len(dm) == len(fm):
            res.append(o)
        else:
            res.append(dict(mean=np.interp(fm, dm, np.asarray(o["mean"], float)).astype(np.float32),
                            std=np.interp(fm, dm, np.asarray(o["std"], float)).astype(np.float32)))
    return res
'''
txt = re.sub(r'PF_BANK = \["pf_1".*?\n', _pf_bank_block, txt, count=1)
assert 'PF_BANK = ["af", "dec10", "sharp"]' in txt and 'DEC_BANK' in txt, "PF_BANK/DEC 置換失敗"
# smooth PF 呼び出しを per-bank 間引き対応に(tw/self/nbr の3ループ)
n_before = txt.count('run_smoother_ext(inps, Pk, SEED_V26, N_SEED, WELL_CHUNK)')
txt = txt.replace('run_smoother_ext(inps, Pk, SEED_V26, N_SEED, WELL_CHUNK)',
                  'run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)')
assert n_before == 3, f"run_smoother_ext 呼び出しが3箇所でない({n_before})"

# (3) pfA を PF_LIB に足す行は不要(4バンクのみ) → 無害化
txt = txt.replace('PF_LIB["pfA"] = dict(_V38_PACK["params"])',
                  '# (v69: pfA不使用。V41_ANCHOR_MULT は per-bank anchor_mult 欠損時のフォールバックとしてのみ残す)')

# (4) anchor_mult を per-bank に(smoother core)
txt = txt.replace('asig = (ancs[:, i][:, None] * V41_ANCHOR_MULT).clamp_min(1e-6)',
                  'asig = (ancs[:, i][:, None] * float(P.get("anchor_mult", V41_ANCHOR_MULT))).clamp_min(1e-6)  # ★v69: per-bank錨mult')
assert 'P.get("anchor_mult", V41_ANCHOR_MULT)' in txt, "anchor_mult 置換失敗"

# (5) SMOOTH_LAG 16 -> 192
txt = re.sub(r'SMOOTH_LAG  = 16\b', 'SMOOTH_LAG  = 192', txt, count=1)
assert 'SMOOTH_LAG  = 192' in txt, "SMOOTH_LAG 置換失敗"

# (6) 出力パス v47 -> v69
txt = txt.replace('OUT_PARQUET = PROJECT_V26 / "v47" / "artifacts_v47" / "data" / "train.parquet"',
                  'OUT_PARQUET = PROJECT_V26 / "v69" / "artifacts_v69" / "data" / "train.parquet"')
txt = txt.replace('"pfpass_cache_v47.npz"', '"pfpass_cache_v69.npz"')
assert 'artifacts_v69' in txt and 'pfpass_cache_v69' in txt, "パス置換失敗"

out = PROJ / "v69" / "create_candidates_v69.py"
out.parent.mkdir(parents=True, exist_ok=True)
ast.parse(txt)                                    # 構文チェック
out.write_text(txt, encoding="utf-8")
print(f"written {out}  ({len(txt.splitlines())} lines)")
print("PF_LIB banks:", list(PF_LIB))
for n, p in PF_LIB.items():
    print(f"  {n:8s} anchor={p.get('use_anchor')} mult={p.get('anchor_mult'):.2f} np={p['n_particles']} ls={p['likelihood_scale']:.1f} hgr_sm={p['hgr_smooth_r']}")
