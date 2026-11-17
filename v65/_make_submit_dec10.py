# -*- coding: utf-8 -*-
"""
submit_pf_v65.ipynb(v66 struct 済)→ dec10版(MD-10 間引きPF)提出ノートを生成。
=====================================================================
変更点:
  (1) PARAM_SETS = dec10 best(v65dec10_..._current.json)。grid0.5/mult12/np400。
  (2) 実行ループに decimation を注入: make_inputs 出力を MD=DEC で bin集約(GR mean)→ PF →
      予測軌跡を full_md に補間(= pf_dec_v65 と同一ロジック = tuning と submit で整合)。
struct は既存 submit_pf_v65 の v66_leanv1cumul(実行時構築)をそのまま使用(dec10 は v66 prior でチューニング)。
実行: .venv/Scripts/python.exe v65/_make_submit_dec10.py
出力: v65/submit_pf_dec10_v65.ipynb
"""
import json, re, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
V65 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v65")

DEC = 10
d = json.load(open(V65 / "submit_pf_v65.ipynb", encoding="utf-8"))
best = json.load(open(V65 / "artifacts_v65" / "v65dec10_v66_leanv1cumul_tw_current.json", encoding="utf-8"))
P = best["best_params"]
pj = json.dumps(P, ensure_ascii=False).replace(": false", ": False").replace(": true", ": True").replace(": null", ": None")

# ---- cell[0] markdown ----
c0 = "".join(d["cells"][0]["source"])
note = (f"\n\n> **★dec10版(MD-{DEC} 間引きPF)**: 各行でなく MD-{DEC} に bin集約(GR=平均=denoise)して PF を回し、\n"
        f"> 予測軌跡を全解像度(MD-1)に補間して提出。`pf_dec_v65` と同一ロジック(tuning=CV 7.5x と整合)。\n"
        f"> struct は v66_leanv1cumul(実行時構築)。dataset に `struct_field_v66_leanv1cumul.npz` を添付。\n")
if "dec10版" not in c0:
    d["cells"][0]["source"] = (c0 + note).splitlines(keepends=True)

# ---- cell[1] PARAM_SETS 差し替え ----
c1 = "".join(d["cells"][1]["source"])
new_ps = ("PARAM_SETS = [\n"
          f"    # ★dec10版 best(v65dec10_..._current, trials={best.get('trials_done')}, 探索best={best.get('best_value'):.4f})。MD-{DEC}間引きで採点\n"
          f"    {pj},\n]")
c1n = re.sub(r"PARAM_SETS = \[.*?\n\]", new_ps, c1, count=1, flags=re.S)
assert c1n != c1, "PARAM_SETS 差し替え失敗"
d["cells"][1]["source"] = c1n.splitlines(keepends=True)

# ---- cell[4] 実行ループに decimation 注入 ----
c4 = "".join(d["cells"][4]["source"])
# (a) dec ヘルパを先頭コメント直後に追加(pf_dec_v65 と同一)
dec_helpers = f'''
# ===== dec{DEC}: MD粗化ヘルパ(pf_dec_v65 と同一ロジック。単相 PHASE=1, GR=mean)=====
DEC = {DEC}
def _agg_dec(x, s, how):
    x = np.asarray(x, float); n = len(x)
    cuts = sorted(set([0, n] + list(range(s, n, s))))          # phase=0(単相)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = x[a:b]; out.append(float(seg.mean()) if how == "mean" else float(seg[(b - a) // 2]))
    return np.array(out)
def make_inputs_dec_sub(cache, P):
    inp = make_inputs(cache, P); md = np.asarray(inp["md"], float)
    if len(md) < DEC * 2:
        di = dict(inp); di["full_md"] = md; di["dec_md"] = md; return di
    di = dict(md=_agg_dec(md, DEC, "mean"), z=_agg_dec(inp["z"], DEC, "mean"), gr=_agg_dec(inp["gr"], DEC, "mean"),
              gg=inp["gg"], gmin=inp["gmin"], gst=inp["gst"], gs=inp["gs"], ls=inp["ls"], ir=inp["ir"],
              N=inp["N"], anc=_agg_dec(inp["anc"], DEC, "mean"), ancs=_agg_dec(inp["ancs"], DEC, "mean"))
    di["full_md"] = md; di["dec_md"] = di["md"]; return di
'''
anchor = '# ===== 実行: 各セット → seed尤度加重平均 → セット間 AVG 合成 ====='
assert anchor in c4
c4 = c4.replace(anchor, anchor + "\n" + dec_helpers, 1)

# (b) run ループの make_inputs → decimation + 補間 に差し替え
old = '''        inps = [make_inputs(caches[w], P) for w in sub]
        res = run_forward_ens(inps, P, DEVICE, SEED, N_SEEDS)
        for j, w in enumerate(sub): outs[w] = np.asarray(res[j], np.float64)'''
new = '''        dinps = [make_inputs_dec_sub(caches[w], P) for w in sub]     # ★MD-DEC 集約
        res = run_forward_ens(dinps, P, DEVICE, SEED, N_SEEDS)
        for j, w in enumerate(sub):
            _dm = dinps[j]["dec_md"]; _fm = dinps[j]["full_md"]; _dt = np.asarray(res[j], np.float64)
            outs[w] = _dt if len(_dm) == len(_fm) else np.interp(_fm, _dm, _dt)   # ★全解像度に補間'''
assert old in c4, "run ループの差し替え箇所が見つからない"
c4 = c4.replace(old, new, 1)
d["cells"][4]["source"] = c4.splitlines(keepends=True)

# ---- 保存 + 構文チェック ----
out = V65 / "submit_pf_dec10_v65.ipynb"
json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import ast
for c in d["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print(f"written {out.name}  (dec{DEC} params baked, decimation 注入, 全codeセル parse OK)")
print(f"  baked: grid={P['grid_step']} mult={P['anchor_mult']:.1f} np={P['n_particles']} anchor={P['use_anchor']}")
