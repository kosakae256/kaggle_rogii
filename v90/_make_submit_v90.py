# -*- coding: utf-8 -*-
"""v90/submit_v90.ipynb を生成。★v69_4流: 1つのdataset(rogii-v90)だけで動く自己完結。
   dataset(rogii-v90) の中身(これだけアップロード):
     v90/gen_test_features_v90.py  v90/create_candidates_v90.py  v90/infer_v90.py
     v90/artifacts_v90/{models_fwd/, nn/, submit_pack_nn.json, features.json, pp_params.json}
     v50/artifacts_v50/stageA_enccapaug_f0..4.pt
     v64/artifacts_v64/v89_{ddoof,boot,sdf_ddil}_f0..4.pt
   スクリプトは全て自己完結(他ディレクトリのコードを import しない)。V90_ROOT=dataset, V90_WORK=/kaggle/working。"""
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
def code(s): return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.splitlines(keepends=True))
def md(s): return dict(cell_type="markdown", metadata={}, source=s.splitlines(keepends=True))

c0 = """# v90 提出 — emission補正PF(per-bank W)+SDF特徴 × GBM(9var)+TCN(combo3) blend
- CV(OOF overall): tabular 6.5695 / **TCN blend 6.3790**(v69_4 6.448比 −0.069)
- **dataset は rogii-v90 の1つだけ**(下記)。スクリプトは自己完結、V90_ROOT=dataset / V90_WORK=/kaggle/working。
- パイプライン: ①test側 sim_boot/sdf 生成(v50 enc+v89 U-Net/SDF) → ②v90 create(SPLIT=test, emission+SDF) → ③GBM+TCN blend+pp → submission.csv
- ★gf特徴(v48 struct依存)は0埋め初版=小劣化。"""

c1 = r'''# ===== 1) dataset 検出 → V90_ROOT / V90_WORK 設定 =====
import os, sys, glob, subprocess
ON_KAGGLE = os.path.exists("/kaggle/input")
if ON_KAGGLE:
    _cand = glob.glob("/kaggle/input/**/v90/infer_v90.py", recursive=True)
    assert _cand, "dataset(v90/infer_v90.py を含む)が /kaggle/input に見つからない"
    ROOT = os.path.dirname(os.path.dirname(_cand[0]))   # .../v90/infer_v90.py → 親の親=dataset root
    WORK = "/kaggle/working"
else:
    ROOT = os.getcwd(); WORK = os.path.join(os.getcwd(), "v90", "artifacts_v90")
os.makedirs(os.path.join(WORK, "data"), exist_ok=True)
# コンペデータ検出(Kaggleは別input, ローカルはrepo)
DATA = next((d for d in ["/kaggle/input/rogii-wellbore-geology-prediction",
                         "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
                         os.path.join(ROOT, "rogii-wellbore-geology-prediction")] if os.path.exists(d)), None)
if DATA is None:
    import glob as _g; _h = _g.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    DATA = os.path.dirname(_h[0]) if _h else ""
env = dict(os.environ, V90_ROOT=ROOT, V90_WORK=WORK, V90_DATA=DATA)
PY = sys.executable
print("V90_ROOT =", ROOT, "| V90_WORK =", WORK, "| V90_DATA =", DATA)'''

c2 = r'''# ===== 2) test側 sim_boot_test + sdf_pred_test 生成(自己完結: v50 enc + v89 boot/SDF chain) =====
r = subprocess.run([PY, os.path.join(ROOT, "v90", "gen_test_features_v90.py")], env=env, cwd=WORK)
assert r.returncode == 0, "gen_test_features 失敗"'''

c3 = r'''# ===== 3) v90 create(SPLIT=test): emission補正PF + SDF特徴 → test.parquet =====
r = subprocess.run([PY, os.path.join(ROOT, "v90", "create_candidates_v90.py")],
                   env=dict(env, SPLIT="test"), cwd=WORK)
assert r.returncode == 0, "create(test) 失敗"'''

c4 = r'''# ===== 4) 推論: GBM(9var×5fold) + TCN(combo3 3seed×5fold) blend + pp → submission.csv =====
r = subprocess.run([PY, os.path.join(ROOT, "v90", "infer_v90.py")], env=env, cwd=os.getcwd())
assert r.returncode == 0, "infer 失敗"
import pandas as pd
print(pd.read_csv("submission.csv").head()); print("★submission.csv 完成")'''

nb = dict(nbformat=4, nbformat_minor=5,
          metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}},
          cells=[md(c0), code(c1), code(c2), code(c3), code(c4)])
json.dump(nb, open("v90/submit_v90.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written v90/submit_v90.ipynb")
