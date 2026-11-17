# -*- coding: utf-8 -*-
"""
submit_pf_v65.ipynb(v66 struct 済)→ struct20 + dec10 の 50:50 平均 提出ノートを生成。
=====================================================================================
2セットを各々回して 1本に平均(ユーザー指示 2026-07-15「struct20と今の奴(dec10)を50:50」):
  set0 = struct20 (mult20, grid0.2, np600) を MD-1(そのまま)で PF
  set1 = dec10    (mult12, grid0.5, np400, tune 7.566) を MD-10 集約→全解像度に補間して PF
  → 各wellで 2軌跡を nanmean = 50:50 平均。CV相関 0.802 の直交ペア(CV blend ~7.11)。
struct は既存 submit_pf_v65 の v66_leanv1cumul(実行時構築)を共用(両セットとも v66 prior でチューニング)。
実行: .venv/Scripts/python.exe v65/_make_submit_blend_struct20_dec10.py
出力: v65/submit_blend_struct20_dec10_v65.ipynb   (dataset に struct_field_v66_leanv1cumul.npz 添付)
"""
import json, re, ast, sys
from pathlib import Path
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
V65 = Path(r"C:\Users\kosaka256\Documents\rogii_claude\v65")

# --- 欠けキーの中立デフォルト(struct20 は use_phys/tw_source 欠け) ---
V65_DFLT = dict(use_phys=False, use_anchor=True, anchor_ramp_md=0.0, lk_floor=0.0,
                gr_debias=0, rate_target="zero", grid_step=0.2,
                tw_source="tw", self_mix_w=1.0, nbr_mix_w=1.0)

p20 = json.load(open(V65 / "artifacts_v65" / "struct20_params.json", encoding="utf-8"))["best_params"]
decj = json.load(open(V65 / "artifacts_v65" / "v65dec10_v66_leanv1cumul_tw_current.json", encoding="utf-8"))
pdec = decj["best_params"]
setA = {**V65_DFLT, **p20, "use_anchor": True}                 # struct20(MD-1)
setB = {**V65_DFLT, **pdec, "use_anchor": True}                # dec10(MD-10)


def lit(p):
    return json.dumps(p, ensure_ascii=False).replace(": false", ": False").replace(": true", ": True").replace(": null", ": None")


d = json.load(open(V65 / "submit_pf_v65.ipynb", encoding="utf-8"))

# ---- cell[0] markdown ----
d["cells"][0]["source"] = (
    "# v65 生PF提出 — **struct20 + dec10 の 50:50 平均**\n"
    "- set0 = struct20(struct錨 mult20, grid0.2, np600)を **MD-1**(そのまま)で PF。単体 LB≈7.9。\n"
    "- set1 = dec10(mult12, grid0.5, np400, 探索best 7.566)を **MD-10 集約→全解像度に補間**して PF。`pf_dec_v65` と同一ロジック。\n"
    "- 各wellで 2軌跡を **nanmean(=50:50)**。CV誤差相関 0.802 の直交ペア(CV blend ~7.11, 単体struct20比 −0.33)。\n"
    "- struct は v66_leanv1cumul(実行時構築)。dataset に `struct_field_v66_leanv1cumul.npz` を添付。\n"
).splitlines(keepends=True)

# ---- cell[1] PARAM_SETS(2セット)+ DEC_SETS ----
c1 = "".join(d["cells"][1]["source"])
new_ps = ("PARAM_SETS = [\n"
          f"    # set0: struct20 (mult20, grid0.2, np600) — MD-1\n    {lit(setA)},\n"
          f"    # set1: dec10 (mult{setB['anchor_mult']:.1f}, grid{setB['grid_step']}, np{setB['n_particles']}, tune{decj.get('best_value'):.3f}) — MD-10間引き\n    {lit(setB)},\n"
          "]\n"
          "DEC_SETS = [1, 10]   # 各セットの間引き倍率(1=MD-1そのまま / 10=MD-10集約→全解像度に補間)")
c1n = re.sub(r"PARAM_SETS = \[.*?\n\]", new_ps, c1, count=1, flags=re.S)
assert c1n != c1, "PARAM_SETS 差し替え失敗"
d["cells"][1]["source"] = c1n.splitlines(keepends=True)

# ---- cell[4] 実行ループを dec 対応の 2セット版に全置換 ----
cell4 = r'''# ===== 実行: struct20(MD-1) + dec10(MD-10間引き) を各々回して 50:50 平均 =====
# --- dec 集約ヘルパ(pf_dec_v65 と同一。単相 PHASE=1, GR=mean) ---
def _agg_dec(x, s, how="mean"):
    x = np.asarray(x, float); n = len(x)
    cuts = sorted(set([0, n] + list(range(s, n, s))))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        seg = x[a:b]; out.append(float(seg.mean()) if how == "mean" else float(seg[(b - a) // 2]))
    return np.array(out)
def make_inputs_dec_sub(cache, P, DEC):
    inp = make_inputs(cache, P); md = np.asarray(inp["md"], float)
    if len(md) < DEC * 2:
        di = dict(inp); di["full_md"] = md; di["dec_md"] = md; return di
    di = dict(md=_agg_dec(md, DEC), z=_agg_dec(inp["z"], DEC), gr=_agg_dec(inp["gr"], DEC),
              gg=inp["gg"], gmin=inp["gmin"], gst=inp["gst"], gs=inp["gs"], ls=inp["ls"], ir=inp["ir"],
              N=inp["N"], anc=_agg_dec(inp["anc"], DEC), ancs=_agg_dec(inp["ancs"], DEC))
    di["full_md"] = md; di["dec_md"] = di["md"]; return di

t0 = time.time()
wids = sorted(set(os.path.basename(f).split("__")[0] for f in _glob.glob(str(TESTD / "*__horizontal_well.csv"))))
caches = {}
for w in wids:
    c = build_cache_test(w)
    if c is not None: caches[w] = c
W = sorted(caches.keys())
print(f"test wells: {len(W)}/{len(wids)}")

preds_by_set = []
for si, P0 in enumerate(PARAM_SETS):
    P = dict(P0); DECi = int(DEC_SETS[si])
    outs = {}
    for c0_ in range(0, len(W), CHUNK):
        sub = W[c0_:c0_ + CHUNK]
        if DECi > 1:                                            # ★MD-DEC 集約→補間
            dinps = [make_inputs_dec_sub(caches[w], P, DECi) for w in sub]
            res = run_forward_ens(dinps, P, DEVICE, SEED, N_SEEDS)
            for j, w in enumerate(sub):
                dm = dinps[j]["dec_md"]; fm = dinps[j]["full_md"]; dt = np.asarray(res[j], np.float64)
                outs[w] = dt if len(dm) == len(fm) else np.interp(fm, dm, dt)   # 全解像度に補間
        else:                                                   # MD-1 そのまま
            inps = [make_inputs(caches[w], P) for w in sub]
            res = run_forward_ens(inps, P, DEVICE, SEED, N_SEEDS)
            for j, w in enumerate(sub): outs[w] = np.asarray(res[j], np.float64)
        if DEVICE == "cuda": torch.cuda.empty_cache()
    preds_by_set.append(outs)
    print(f"set{si+1}/{len(PARAM_SETS)} (dec{DECi}) done ({time.time()-t0:.0f}s)", flush=True)

pids = []; pvals = []
for w in W:
    c = caches[w]
    A = np.stack([ps[w] for ps in preds_by_set], 0)            # (2, nev) struct20 / dec10 の TVT軌跡
    z = np.nanmedian(A, 0) if AVG == "median" else np.nanmean(A, 0)   # ★50:50 平均
    if PP:
        mds = np.maximum(c["md"] - c["md"][0] + 1.0, 0.0)
        d = (z - c["lk"]) * (1.0 - np.exp(-mds / float(PP["tau"]))) * float(PP.get("alpha", 1.0))
        z = c["lk"] + d
    pids += [f"{w}_{int(i)}" for i in c["ev_index"]]
    pvals += list(z)
pred_df = pd.DataFrame({"id": pids, "tvt": pvals})
print("predicted rows:", len(pred_df), f"({time.time()-t0:.0f}s)")
'''
d["cells"][4]["source"] = cell4.splitlines(keepends=True)

# ---- 保存 + 検証 ----
out = V65 / "submit_blend_struct20_dec10_v65.ipynb"
json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for c in d["cells"]:
    if c["cell_type"] == "code":
        src = "".join(c["source"]); ast.parse(src)
        assert ": false" not in src and ": true" not in src, "JSON bool 混入"
s1 = "".join(d["cells"][1]["source"])
assert s1.count("PARAM_SETS = [") == 1 and "DEC_SETS = [1, 10]" in s1
print(f"written {out.name}  (2セット 50:50, 全codeセル parse OK)")
print(f"  set0 struct20: mult={setA['anchor_mult']} grid={setA['grid_step']} np={setA['n_particles']} (MD-1)")
print(f"  set1 dec10   : mult={setB['anchor_mult']:.2f} grid={setB['grid_step']} np={setB['n_particles']} (MD-10, tune {decj.get('best_value'):.3f})")
