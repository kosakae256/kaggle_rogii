# -*- coding: utf-8 -*-
"""v103 追加脱相関バンク = anchor-free dec10(v65 の3性格: jumpy/sharp/smooth, use_anchor=False)。
今日の法則「錨に寄せない=多様性↑」(emission OFF が効いたのと同型)を dec10 に適用。
v65/artifacts_v65/v65pdec10_{personality}_v66_leanv1cumul_{best|current}.json の params を
build_forward_v102 が読む pf_banks_config.json 形式(3バンク)へ変換。physics_banks=[] で錨OFF。
出力: v103/artifacts_v103/dec10af/pf_banks_config.json(+ 未使用だが import 時ロードされる anchor をコピー)。
実行: .venv/Scripts/python.exe v103/build_dec10af_config.py"""
import json, shutil
from pathlib import Path

V104 = Path(__file__).resolve().parent
PROJ = V104.parent
V65 = PROJ / "v65" / "artifacts_v65"
DEC10 = PROJ / "v102" / "artifacts_v102" / "dec10"   # anchor コピー元(未使用だが必要)
OUT = V104 / "artifacts_v104" / "dec10af"
OUT.mkdir(parents=True, exist_ok=True)

# (personality -> バンク名, config ファイル)。sharp/smooth 採用(jumpy 不採用)。delta_1=sharp, delta_2=smooth。
SRC = [
    ("dec_sharp",  "v65pdec10_sharp_matcher_v66_leanv1cumul_current.json"),
    ("dec_smooth", "v65pdec10_smooth_tracker_v66_leanv1cumul_best.json"),
]

params = {}
for bank, fn in SRC:
    d = json.load(open(V65 / fn, encoding="utf-8"))
    p = dict(d.get("params") or d.get("best_params"))
    assert p, f"params 空: {fn}"
    assert p.get("use_anchor") is False and p.get("use_phys") is False, f"★anchor-free でない: {fn}"
    params[bank] = p
    print(f"  {bank:11s} <- {fn}  gr_power={p['gr_power']:.3f} n_part={p['n_particles']} lik={p['likelihood_scale']:.2f}")

cfg = {
    "bank_order": [b for b, _ in SRC],
    "physics_banks": [],            # ★空 = attach_anchor が anc=0/ancs=1e9(錨OFF)。struct 不使用
    "params": params,
    "w_nn_bank": {},
    "w_nn_default": 0.02,           # emission OFF(SIM_PKL 無し)で実効未使用
    "n_seed": 32,
    "smooth_lag": 192,
    "smooth_mode": "full",
    "anchor_pkl": "grfree_anchor_train.pkl",   # import 時ロードされるが physics_banks=[] なので実効未使用
    "ps_combo_tau": 0.0,
    "ps_combo_off_banks": [b for b, _ in SRC],
}
(OUT / "pf_banks_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
# import 時に必要な anchor(未使用)をコピー
if not (OUT / "grfree_anchor_train.pkl").exists():
    shutil.copyfile(DEC10 / "grfree_anchor_train.pkl", OUT / "grfree_anchor_train.pkl")
print(f"\n[dec10af] {OUT/'pf_banks_config.json'}  banks={cfg['bank_order']} physics_banks=[](錨OFF) smooth=full")
print("生成: LEG=v96, MODE=dec10, FWD_METHODS=tw, emission OFF → decaf_ancc_delta_{1,2,3}(3脱相関候補)")
