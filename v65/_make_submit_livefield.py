# submit_struct20_v65.ipynb → struct場をその場で(test prefix込みで)増強する livefield 版を生成
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

d = json.load(open("v65/submit_struct20_v65.ipynb", encoding="utf-8"))

d["cells"][0]["source"] = (
    "# v65 生PF提出 — **struct20 + livefield(struct場を test 既知帯込みでその場構築)**\n"
    "- パラメータは struct20 best(LB 7.881)と同一。**変更点は struct 場のみ**: train場(npz)に、\n"
    "  **全 test 坑井の既知帯(TVT_input prefix)から抽出した dip 観測点を追加**して場を構築する。\n"
    "- 抽出は train 場ビルダー(struct_prior_v38)と同一手順: 水平区間の (R0−U) をランプ基底で最小二乗 → (中点XY, c/proj)。\n"
    "  自坑井の prefix も自分の場に入る(TVT_input は与えられた入力=合法)。κ/σ較正は train 由来を流用。\n"
    "- 判定: LB 7.881(struct20)との差分が「test既知帯で場を増強する価値」。\n"
    "- dataset に struct_field_v38.npz 必須(rogii-v64-artifacts 等)。"
).splitlines(keepends=True)

c3 = d["cells"][3]; s = "".join(c3["source"])

old = '''        FZ = np.load(_fz)
        FLD = FZ["field"]; KAP = FZ["kappa"]; CEN = FZ["centers"]; SGC = FZ["sigs"]'''
new = '''        FZ = np.load(_fz)
        FLD = FZ["field"]; KAP = FZ["kappa"]; CEN = FZ["centers"]; SGC = FZ["sigs"]
        # ===== livefield: test既知帯(prefix)から dip観測点を抽出して場に追加 =====
        _TH = float(FZ["theta0"]); _MINPROJ = 0.3; _MIN_LAT = 150
        _pts = []
        _wi0 = int(FLD[:, 3].max()) + 1 if FLD.shape[1] > 3 else 10 ** 6
        for _k, _fp in enumerate(_glob.glob(str(Path(TESTD) / "*__horizontal_well.csv"))):
            try:
                _df = pd.read_csv(_fp, usecols=["X", "Y", "Z", "TVT_input"])
                _X = _df["X"].to_numpy(); _Y = _df["Y"].to_numpy(); _Z = _df["Z"].to_numpy()
                _ti = _df["TVT_input"].to_numpy()
                _fin = np.where(np.isfinite(_ti))[0]
                if len(_fin) < _MIN_LAT + 2: continue
                _s = int(_fin.max())
                # 既知帯の水平区間(横ステップ > 2|dz| が安定して成り立つ範囲)のみ使用
                _dx = np.diff(_X[:_s + 1]); _dy = np.diff(_Y[:_s + 1]); _dz = np.diff(_Z[:_s + 1])
                _lat = np.hypot(_dx, _dy) > 2.0 * np.abs(_dz)
                _idx = np.where(_lat)[0]
                if len(_idx) < _MIN_LAT: continue
                _j0 = int(_idx[0])
                _n = _s - _j0
                if _n < _MIN_LAT: continue
                _ndz = -np.diff(_Z)[_j0:_s]; _nn = len(_ndz)
                _U = np.cumsum(_ndz)
                _R0 = _ti[_j0 + 1:_s + 1] - _ti[_j0]
                _m = np.isfinite(_R0)
                if _m.sum() < _MIN_LAT: continue
                _Kp = int(np.clip(_nn // 200, 1, 8))
                _edges = np.linspace(0, _nn, _Kp + 1)
                _t = np.arange(1, _nn + 1.0)
                _phi = np.column_stack([np.clip(_t - _edges[_j], 0, _edges[_j + 1] - _edges[_j]) for _j in range(_Kp)])
                _c = np.linalg.lstsq(_phi[_m], (_R0 - _U)[_m], rcond=None)[0]
                for _j in range(_Kp):
                    _f0 = _j0 + 1 + int(_edges[_j]); _f1 = min(_j0 + 1 + max(int(_edges[_j + 1]) - 1, int(_edges[_j])), len(_X) - 1)
                    _az = np.degrees(np.arctan2(_Y[_f1] - _Y[_f0], _X[_f1] - _X[_f0])) % 360
                    _pj = np.cos(np.radians(_az - _TH))
                    if abs(_pj) > _MINPROJ and np.isfinite(_c[_j]):
                        _pts.append(((_X[_f0] + _X[_f1]) / 2, (_Y[_f0] + _Y[_f1]) / 2, float(_c[_j] / _pj), _wi0 + _k))
            except Exception:
                continue
        if _pts:
            _pts = np.array(_pts, np.float64)
            FLD = np.vstack([FLD[:, :4] if FLD.shape[1] >= 4 else np.column_stack([FLD, -np.ones(len(FLD))]), _pts])
            print(f"[livefield] train場 {len(FZ['field'])}点 + test既知帯 {len(_pts)}点"
                  f"({len(set(_pts[:, 3]))}坑井) = {len(FLD)}点")
        else:
            print("[livefield] ★test既知帯から観測点が取れず → train場のみ(通常struct20と同一)")'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
c3["source"] = s.splitlines(keepends=True)

json.dump(d, open("v65/submit_struct20_livefield_v65.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import ast
for c in d["cells"]:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
print("written v65/submit_struct20_livefield_v65.ipynb, all cells parse OK")
