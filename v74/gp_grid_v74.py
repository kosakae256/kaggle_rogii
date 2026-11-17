# -*- coding: utf-8 -*-
"""GPハイパラ感度グリッド(探索用スモーク40井, CAP1200)。本採用前にfoldプロトコルで確認する。"""
import io, sys, os, itertools, importlib
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
os.environ["NW"] = os.environ.get("NW", "40")
os.environ["CAP"] = "1200"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for LA, LC, SF in itertools.product((600, 1000, 1500), (200, 400), (15, 25)):
    os.environ["LA"] = str(LA); os.environ["LC"] = str(LC); os.environ["SF"] = str(SF)
    import phaseE_gp_v74 as E
    importlib.reload(E)
    print(f"--- LA={LA} LC={LC} SF={SF}", flush=True)
    E.main()
