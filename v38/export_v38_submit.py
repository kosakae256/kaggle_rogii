# -*- coding: utf-8 -*-
"""
v38 提出パック書き出し: 調整済みPFパラメータ+錨設定を v38_submit_pack.json に固める。
どのstudyを固めるかは環境変数(既定=struct row mult30)。wellmean版が勝ったら再エクスポートして上げ直す。
実行: rogii_claude/.venv/Scripts/python.exe v38/export_v38_submit.py
Kaggleへ上げる物: v38_submit_pack.json + struct_field_v38.npz(データセットは既存 rogii-artifact-v39 でOK)
"""
import os, json
from pathlib import Path
P = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = P / "v38" / "artifacts_v38"

PRIOR = os.environ.get("V38_PRIOR", "struct")
MODE = os.environ.get("V38_AMODE", "row")
MULT = float(os.environ.get("V38_MULT", "30"))
BMULT = float(os.environ.get("V38_BMULT", "1.0"))
study = (f"v38_anchor_mult{MULT:g}" if (PRIOR == "struct" and MODE == "row")
         else f"v38_{PRIOR}_mult{MULT:g}" if MODE == "row" else f"v38_{PRIOR}_wm{BMULT:g}")
src = None
for cand in [ART / f"{study}_best.json", ART / f"{study}_current.json"]:
    if cand.exists(): src = cand; break
assert src is not None, f"{study} の best/current json が無い"
d = json.loads(src.read_text())
params = d.get("params") or d.get("best_params")
assert params, "params が見つからない"
pack = dict(study=study, prior=PRIOR, anchor_mode=MODE, anchor_mult=MULT, bias_mult=BMULT,
            seed=777, n_seeds=32, well_chunk=60, params=params,
            best_value=d.get("best_value") or d.get("best_value_on_all"))
(ART / "v38_submit_pack.json").write_text(json.dumps(pack, indent=1))
print(f"saved v38_submit_pack.json  <- {src.name}")
print(f"  study={study}  mode={MODE} mult={MULT:g} bmult={BMULT:g}  best={pack['best_value']}")
print(f"  Kaggleへ: v38_submit_pack.json + struct_field_v38.npz")
