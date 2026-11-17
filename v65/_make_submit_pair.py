# submit_pf_v65.ipynb から pfA30 / struct20 の2本の提出ノートを生成(1回実行用)
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

base = json.load(open("v65/submit_pf_v65.ipynb", encoding="utf-8"))
V65_DFLT = dict(use_phys=False, use_anchor=True, anchor_ramp_md=0.0, lk_floor=0.0,
                gr_debias=0, rate_target="zero", grid_step=0.2,
                tw_source="tw", self_mix_w=1.0, nbr_mix_w=1.0)

pack30 = json.loads(open("v38/artifacts_v38/v38_submit_pack.json", encoding="utf-8").read())
setA = {**V65_DFLT, **pack30["params"], "anchor_mult": float(pack30["anchor_mult"]), "use_anchor": True}
p20 = json.loads(open("v38/artifacts_v38/v38_anchor_mult20_current.json", encoding="utf-8").read())["best_params"]
setB = {**V65_DFLT, **p20, "anchor_mult": 20.0, "use_anchor": True}

def bake(nbdict, params, title, note):
    d = json.loads(json.dumps(nbdict))   # deep copy
    # md セル差し替え
    d["cells"][0]["source"] = title.splitlines(keepends=True)
    # cell1 の PARAM_SETS を差し替え
    c = d["cells"][1]; s = "".join(c["source"])
    i0 = s.index("PARAM_SETS = [")
    i1 = s.index("]", s.index("]", i0))  # 最初の閉じ括弧(リスト1要素想定)
    i1 = s.index("\n]", i0) + 2
    lit = json.dumps(params).replace(": false", ": False").replace(": true", ": True")
    new = "PARAM_SETS = [\n    # " + note + "\n    " + lit + ",\n]"
    s = s[:i0] + new + s[i1:]
    c["source"] = s.splitlines(keepends=True)
    return d

nbA = bake(base, setA,
    "# v65 生PF提出 — **pfA(struct錨 mult30, v38_submit_pack)**\n"
    "- v38 単体提出(LB 8.593, seed777/32seed)と同一 regime の v65 エンジン再現。LB がほぼ再現されれば v65 エンジン移植の健全性確認になる。\n"
    "- dataset に struct_field_v38.npz 必須(rogii-v64-artifacts 等)。",
    "pfA = v38_submit_pack(struct錨 mult30, n_particles600)。LB 8.593 の再現基準")
json.dump(nbA, open("v65/submit_pfA30_v65.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written v65/submit_pfA30_v65.ipynb")

nbB = bake(base, setB,
    "# v65 生PF提出 — **struct錨 mult20 best(v38_anchor_mult20)**\n"
    "- 相関調査(2026-07-15)で mult30 比 −0.7ft/258坑井・別seed頑健を実測した構成。prior入り転移則(≈1.06)なら LB 期待 ~8.1-8.2。\n"
    "- dataset に struct_field_v38.npz 必須(rogii-v64-artifacts 等)。",
    "struct20 = v38_anchor_mult20_current best(mult20, n_particles600)。258坑井別seed 7.65-7.72 実測")
json.dump(nbB, open("v65/submit_struct20_v65.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written v65/submit_struct20_v65.ipynb")

# 検証
import ast
for nb in ("v65/submit_pfA30_v65.ipynb", "v65/submit_struct20_v65.ipynb"):
    d = json.load(open(nb, encoding="utf-8"))
    for c in d["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"]); ast.parse(src)
            assert ": false" not in src and ": true" not in src
    s1 = "".join(d["cells"][1]["source"])
    assert s1.count("PARAM_SETS = [") == 1 and '"use_anchor": True' in s1
print("both notebooks parse OK")
