# create_candidates_v47.py → v63/create_v63.py を生成するパッチスクリプト(1回実行用)
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

SRC = "v47/create_candidates_v47.py"
DST = "v63/create_v63.py"
s = open(SRC, encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, f"count({old[:60]!r})={c} (expect {n})"
    s = s.replace(old, new)

# 1) ヘッダ
rep("# v47 = v46 create を 7バンク化(21候補生成用)。PF_BANK=7バンク×tw/self/nbr=pf_ancc/self/nbr_1..7: PF_BANK=[pfA(物理錨,v38 mult30), r0_seed32(物理無), pf_3(物理無)]。\n"
    "# pfAのみ struct錨ON(PHYSICS_BANKS)、他2本は錨OFF。smooth lag=16(実験用軽量)。パラメータは全て v28 PF_LIB / v38 pack のまま。",
    "# v63 = v47 create のコピー。変更点(ユーザー指示 2026-07-14):\n"
    "#   ① SMOOTH_LAG=0 = smooth無し(pts_s=pts_f=純フォワードPF=「普通のPF」)。計算が軽い。\n"
    "#   ② r4_seed32 をバンクから除外 → 6バンク×tw/self/nbr=18ラン。\n"
    "#   ③ PSEUDO モード: 合成井戸(pseudo)ディレクトリを同じパイプラインに通せる。\n"
    "#      env PSEUDO_DIR=合成井戸dir(manifest.json必須) / OUT_PQ=出力parquet。\n"
    "#      近傍プールは常に実train坑井。pseudoはdonorを近傍から除外(リーク防止)。\n"
    "#      struct錨(pfA)は reskin型(軌跡=donor継承)のみ donor の prior を流用、surface型は錨OFF。\n"
    "# 実行例:\n"
    "#   ベースライン: .venv/Scripts/python.exe v63/create_v63.py\n"
    "#   pseudo:      PSEUDO_DIR=v63/artifacts_v63/pseudo_wells_p1 OUT_PQ=v63/artifacts_v63/data/pseudo_p1.parquet \\\n"
    "#                .venv/Scripts/python.exe v63/create_v63.py")

# 2) 出力先
rep('OUT_PARQUET = PROJECT_V26 / "v47" / "artifacts_v47" / "data" / "train.parquet"',
    'OUT_PARQUET = Path(os.environ.get("OUT_PQ", str(PROJECT_V26 / "v63" / "artifacts_v63" / "data" / "train.parquet")))')

# 3) smooth無し
rep("SMOOTH_LAG  = 16            # ★v46: 実験用に軽量化(生成が重いので lag16)。本番は192で再生成。",
    'SMOOTH_LAG  = int(os.environ.get("SMOOTH_LAG", "0"))   # ★v63: 0=smooth無し(純フォワードPF)。>0でv47系smoothに戻せる')

# 4) バンク(r4除外)
rep('PF_BANK = ["pf_1","pf_2","pf_3","r0_seed32","r1_seed32","r4_seed32","pfA"]   # ★v47: 7バンク=21候補(tw/self/nbr×7)。pfAのみ錨',
    'PF_BANK = ["pf_1","pf_2","pf_3","r0_seed32","r1_seed32","pfA"]   # ★v63: 6バンク(r4_seed32除外)。pfAのみ錨')

# 5) PSEUDO 設定ブロック(SPLIT 定義の直後に挿入)
rep('SPLIT       = "train"       # "train"(真値あり) / "test"',
    'SPLIT       = "train"       # "train"(真値あり) / "test"\n'
    '# ---- v63 PSEUDO(合成井戸)モード ----\n'
    '_PSEUDO_DIR = os.environ.get("PSEUDO_DIR")            # 合成井戸dir(None=通常create)\n'
    '_PSEUDO_MODE = _PSEUDO_DIR is not None\n'
    '_PSEUDO_DONOR = {}; _PSEUDO_KIND = {}\n'
    'if _PSEUDO_MODE:\n'
    '    _man = json.loads(open(os.path.join(_PSEUDO_DIR, "manifest.json"), encoding="utf-8").read())\n'
    '    for _e in _man["wells"]:\n'
    '        _PSEUDO_DONOR[_e["id"]] = _e["donor"]; _PSEUDO_KIND[_e["id"]] = _e.get("mode", "surface")\n'
    '    print(f"[v63] PSEUDO mode: {_PSEUDO_DIR} wells={len(_PSEUDO_DONOR)}")\n'
    'def _excl_of(wid):\n'
    '    """近傍プールから除外する実train坑井ID(pseudo=donor / train=自分 / test=None)。"""\n'
    '    if _PSEUDO_MODE: return _PSEUDO_DONOR.get(wid, wid)\n'
    '    return wid if SPLIT == "train" else None\n'
    'def _struct_alias(wid):\n'
    '    """struct錨(_V41_STRUCT)のキー。pseudo: reskin(軌跡=donor)のみdonor流用、surfaceは無効キー。"""\n'
    '    if not _PSEUDO_MODE: return wid\n'
    '    if _PSEUDO_KIND.get(wid) == "reskin": return _PSEUDO_DONOR.get(wid, wid)\n'
    '    return "__no_struct__"')

# 6) struct錨 lookup を alias 経由に
rep('    n = len(x["md"]); sp = _V41_STRUCT.get(wid)',
    '    n = len(x["md"]); sp = _V41_STRUCT.get(_struct_alias(wid))')

# 7) キャッシュ名(v63 + 出力stem別)
rep('_PF_CACHE = OUT_PARQUET.parent / "pfpass_cache_v47.npz"',
    '_PF_CACHE = OUT_PARQUET.parent / f"pfpass_cache_v63_{OUT_PARQUET.stem}.npz"')

# 8) 入力ディレクトリ(pseudo dir override)
rep('    base = DATA_DIR_V26 / SPLIT\n    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})',
    '    base = Path(_PSEUDO_DIR) if _PSEUDO_MODE else (DATA_DIR_V26 / SPLIT)\n'
    '    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})')

# 9) 近傍除外を _excl_of に(nbr探索の2箇所)
rep('        refw = neighbors_of(cx, cy, exclude_wid=(wid if SPLIT == "train" else None))',
    '        refw = neighbors_of(cx, cy, exclude_wid=_excl_of(wid))')
rep('        _NBR_META[wid] = dict(has=int(use), nn=_nearest_dist(cx, cy, exclude_wid=(wid if SPLIT == "train" else None)), nref=len(refw))',
    '        _NBR_META[wid] = dict(has=int(use), nn=_nearest_dist(cx, cy, exclude_wid=_excl_of(wid)), nref=len(refw))')

# 10) pseudo 識別列(pseudoモードのみ付与 → ベースラインparquetはv47と同一スキーマ)
rep('        df["pf_nbr_has"]    = np.float32(mt.get("has", 0))',
    '        if _PSEUDO_MODE:\n'
    '            df["donor"] = _PSEUDO_DONOR.get(w, ""); df["is_pseudo"] = np.float32(1.0)\n'
    '        df["pf_nbr_has"]    = np.float32(mt.get("has", 0))')

# 11) バナー
rep('    print(f"[v28] device={V26_DEVICE} smooth_lag={SMOOTH_LAG} n_seed={N_SEED} PF=r0 self/nbr/z_grad追加 chunk={WELL_CHUNK}")',
    '    print(f"[v63] device={V26_DEVICE} smooth_lag={SMOOTH_LAG}(0=普通のPF) n_seed={N_SEED} banks={len(PF_BANK)} pseudo={_PSEUDO_MODE} chunk={WELL_CHUNK}")')

open(DST, "w", encoding="utf-8").write(s)
print("written:", DST, len(s), "chars")
