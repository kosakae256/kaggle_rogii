# -*- coding: utf-8 -*-
"""notebook を非対話実行(全codeセルを1名前空間でexec)。CVのprintをそのまま流す。
   使い方: python v90/_run_nb.py v90/train_tabular_v90.ipynb"""
import sys, json, io
try: sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception: pass
nb = sys.argv[1]
d = json.load(open(nb, encoding="utf-8"))
src = "\n\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code")
g = {"__name__": "__main__"}
exec(compile(src, nb, "exec"), g)
