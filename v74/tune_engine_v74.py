# -*- coding: utf-8 -*-
"""v74 エンジン正直チューニング — fold0-3(618井)で選び、fold4(155井)で検証(過学習防止プロトコル)。
   未走査ノブ: ridge(トレース比例リッジ比率) × win(ドナーdip平滑窓)。他はv7best固定。"""
import io, sys, os, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phaseC_field2_v74 as F
from sklearn.model_selection import GroupKFold

D = F.load()
ws_all = sorted(D.keys())
gi = np.arange(len(ws_all))
folds = list(GroupKFold(5).split(gi, gi, gi))
tune_w = [ws_all[i] for f in range(4) for i in folds[f][1]]
val_w = [ws_all[i] for i in folds[4][1]]
print(f"tune={len(tune_w)}井  val={len(val_w)}井")
BASE = dict(wg=0.45, ga=0.40, seg=25, sigcap=800, k=128)
res = {}
best = (None, 1e18)
for win in (101, 201, 401):
    ws, donors, pair = F.build_donors(D, win=win)
    for ridge in (0.05, 0.1, 0.25):
        tag = f"win{win}_ridge{ridge}"
        o, pw, dm = F.run(D, ws, donors, pair, "tune_" + tag,
                          eval_wells=tune_w, ridge=ridge, **BASE)
        res[tag] = dict(tune=o)
        if o < best[1]:
            best = ((win, ridge), o)
print(f"\ntune best: win={best[0][0]} ridge={best[0][1]} → {best[1]:.4f}")
# fold4 検証(参照=現行 win201/ridge0.1)
for win, ridge, lbl in [(201, 0.1, "現行"), (best[0][0], best[0][1], "tuned")]:
    ws, donors, pair = F.build_donors(D, win=win)
    o, pw, dm = F.run(D, ws, donors, pair, f"val_{lbl}_win{win}_ridge{ridge}",
                      eval_wells=val_w, ridge=ridge, **BASE)
    res[f"val_{lbl}"] = dict(overall=o)
json.dump(res, open(os.path.join(F.ART, "tune_engine.json"), "w"), indent=1)
