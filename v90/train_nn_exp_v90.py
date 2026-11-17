# -*- coding: utf-8 -*-
"""TCN懸念実験ランナー — train_nn_v90.ipynb を env ノブ付きで非対話実行。
   XTAG   : 保存/結果キー(既定combo)
   XSEEDS : "0" など(既定0,1,2)。実験は1seedで高速に。
   XTOPK  : 特徴top-K(既定192)
   XDROP  : DROP=CH_DROP(既定0.10)。過学習対策実験用に0.2等。
   XEXCL  : 除外正規表現(rank候補から落とす)。例 emission+SDF除外: "pf_.*_(corr|diff|absdiff)|^sdf_"
   使い方: XTAG=noemis XSEEDS=0 XEXCL="pf_.*_(corr|diff|absdiff)|^sdf_" python v90/train_nn_exp_v90.py
"""
import sys, json, io, os
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_nn_v90.ipynb")
d = json.load(open(NB, encoding="utf-8"))
src = "\n\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code")

# --- patch1: config末尾に env override を差し込む ---
_anchor = 'print(f"RUN={RUN} {CFG} | device={DEVICE} torch={torch.__version__}")'
_ov = _anchor + '''
# === v90 experiment overrides ===
import os as _os
if _os.environ.get("XSEEDS"): SEEDS = [int(x) for x in _os.environ["XSEEDS"].split(",")]
if _os.environ.get("XTOPK"):  TOP_K = int(_os.environ["XTOPK"])
if _os.environ.get("XDROP"):  DROP = float(_os.environ["XDROP"]); CH_DROP = float(_os.environ["XDROP"])
_XEXCL = _os.environ.get("XEXCL", "")
RUN = _os.environ.get("XTAG", RUN)
print(f"[v90 exp] RUN={RUN} SEEDS={SEEDS} TOP_K={TOP_K} DROP={DROP} CH_DROP={CH_DROP} XEXCL={_XEXCL!r}")'''
assert _anchor in src, "config anchor 見つからず"
src = src.replace(_anchor, _ov, 1)

# --- patch2: rank候補から XEXCL を除外 ---
_ra = 'rank = [c for c in rank if c != "last_known_tvt"]'
_rp = _ra + '''
if _XEXCL:
    import re as _re2; _xr = _re2.compile(_XEXCL)
    _before = len(rank); rank = [c for c in rank if not _xr.search(c)]
    print(f"[v90 exp] XEXCL除外: rank {_before}->{len(rank)}")'''
assert _ra in src, "rank anchor 見つからず"
src = src.replace(_ra, _rp, 1)
# patch3: submit_pack export を現RUNに
src = src.replace('EXPORT_RUN = "combo"', 'EXPORT_RUN = RUN')

g = {"__name__": "__main__"}
exec(compile(src, NB, "exec"), g)
