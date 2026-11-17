# -*- coding: utf-8 -*-
# v47 = v46 create を 7バンク化(21候補生成用)。PF_BANK=7バンク×tw/self/nbr=pf_ancc/self/nbr_1..7: PF_BANK=[pfA(物理錨,v38 mult30), r0_seed32(物理無), pf_3(物理無)]。
# pfAのみ struct錨ON(PHYSICS_BANKS)、他2本は錨OFF。smooth lag=16(実験用軽量)。パラメータは全て v28 PF_LIB / v38 pack のまま。
"""
v26 train.parquet ビルダー ― v9 の build_well 特徴(そのまま持ち込み)＋ PF を v25 smooth PF(固定ラグ lag512)に注入
==========================================================================================
方針(ユーザー指示):
  - **入れる特徴量は v9(pf_tuning_create_train_csv_multi_pf_v9.ipynb) と同じ**。v9 の build_well を
    **exec でなくコードとして持ち込み(下記 "v9 INLINE" ブロック)**。beam/formation/NCC/pf_z 等を全部生成。
  - **run_pf_ancc だけ v25 smooth PF(固定ラグ SMOOTH_LAG=192)に monkeypatch** して注入。
  - **PF複数**(PF_BANK)。**gsf/heur は入れない**(beam は v9 build_well 内包)。
  - パラメータは冒頭ハードコード(PF_LIB / PF_BANK / SMOOTH_LAG …)。

出力: v26/artifacts_v26/data/train.parquet
実行: rogii_claude/.venv/Scripts/python.exe v26/create_train_parquet_v26.py
※ 生成方法: scratchpad/build_create_v26.py(v9 cell3/5/7 を持ち込み)。lag512はバッファ大→OOMなら WELL_CHUNK↓。
"""

import os, time, glob, json, threading, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore"); np.seterr(all="ignore")

# ============================== CONFIG(v26) ==============================
def _detect_data():   # Kaggle / ローカル 両対応(v9 CFG.dataset_path と共有)
    for _p in ["/kaggle/input/rogii-wellbore-geology-prediction",
               "/kaggle/input/competitions/rogii-wellbore-geology-prediction",
               r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction"]:
        if os.path.exists(_p): return Path(_p)
    _h = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
    return Path(os.path.dirname(_h[0])) if _h else Path(r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction")
_V26_DATA = _detect_data()
PROJECT_V26 = Path(os.environ.get("V90_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR_V26 = _V26_DATA
V26_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V26_DTYPE  = torch.float32

SPLIT       = os.environ.get("SPLIT", "train")      # "train"(真値あり) / "test"
IS_TRAIN    = (SPLIT == "train")
N_WELLS     = int(os.environ["N_WELLS"]) if os.environ.get("N_WELLS") else None   # 例: N_WELLS=3 でスモーク
ART_V90     = PROJECT_V26 / "v90" / "artifacts_v90"                               # ★v90 成果物ルート
WORK_V90    = Path(os.environ.get("V90_WORK", str(ART_V90)))                      # ★書込先(Kaggle=/kaggle/working)
OUT_PARQUET = Path(os.environ.get("OUT_PARQUET",
                   str(WORK_V90 / "data" / f"{SPLIT}.parquet")))                   # ★v90 artifacts/data
SEED_V26    = 4423098
N_SEED      = 32            # ★r0_seed32 regime(6.786)
SMOOTH_LAG  = 16            # ★v46: 実験用に軽量化(生成が重いので lag16)。本番は192で再生成。
CLAMP_FT    = None          # smoothed が forward から±これ超で外れた行は forward に(注入安定化, None=無効)。今回はNone。
WELL_CHUNK  = 120             # GPUバッチ坑井数(OOMなら下げる)
N_JOBS_CPU  = 4


PF_LIB = {
    "p_1": {
        "use_phys": False,
        "use_anchor": False,
        "anchor_mult": 20.0,
        "anchor_ramp_md": 0.0,
        "lk_floor": 1e-05,
        "gr_debias": 0,
        "rate_target": "zero",
        "grid_step": 0.2,
        "tw_source": "tw",
        "self_mix_w": 1.0,
        "nbr_mix_w": 1.0,
        "phys_sig": 0.013666550145707291,
        "gr_power": 2.0758957681159527,
        "gr_sig_def": 103.30294038252535,
        "gr_sig_max": 32.02666295037317,
        "gr_sig_min": 7.316738267002296,
        "gr_sig_mult": 2.9386178298538974,
        "hgr_smooth_r": 2,
        "init_pos_std": 0.28466093602500697,
        "init_rate_std": 0.0016379907956016362,
        "jump_prob": 0.002,
        "jump_std": 5.0,
        "likelihood_scale": 14.722518658710934,
        "mom": 0.999155702506317,
        "n_particles": 700,
        "n_seeds": 32,
        "pn": 0.007818398099945632,
        "rate_clip": 3.0,
        "resamp": 0.04601327417043175,
        "rp": 0.6740096046150705,
        "rr": 0.00015285310269462732,
        "seed0": 0,
        "tvt_clip_margin": 106.95406362143032,
        "tw_gr_smooth_r": 0,
        "vn": 0.001305055874184147
    },
    "p_2": {
        "use_phys": False,
        "use_anchor": False,
        "anchor_mult": 20.0,
        "anchor_ramp_md": 0.0,
        "lk_floor": 1e-05,
        "gr_debias": 0,
        "rate_target": "zero",
        "grid_step": 0.2,
        "tw_source": "tw",
        "self_mix_w": 1.0,
        "nbr_mix_w": 1.0,
        "phys_sig": 0.013666550145707291,
        "gr_power": 2.0089998541184086,
        "gr_sig_def": 89.51748784417867,
        "gr_sig_max": 48.01806291407373,
        "gr_sig_min": 8.498431181857597,
        "gr_sig_mult": 3.4684729668015577,
        "hgr_smooth_r": 3,
        "init_pos_std": 0.13201182717617238,
        "init_rate_std": 0.0019222507112604003,
        "jump_prob": 0.002,
        "jump_std": 5.0,
        "likelihood_scale": 23.48065596476931,
        "mom": 0.9997596435523213,
        "n_particles": 600,
        "n_seeds": 32,
        "pn": 0.00146890326538415,
        "rate_clip": 1.5,
        "resamp": 0.20682162649391647,
        "rp": 0.18590536717989145,
        "rr": 0.00012433094914019284,
        "seed0": 0,
        "tvt_clip_margin": 109.51747834802984,
        "tw_gr_smooth_r": 1,
        "vn": 0.0008747496312790904
    },
    "p_3": {
        "use_phys": False,
        "use_anchor": False,
        "anchor_mult": 20.0,
        "anchor_ramp_md": 0.0,
        "lk_floor": 1e-05,
        "gr_debias": 0,
        "rate_target": "zero",
        "grid_step": 0.2,
        "tw_source": "tw",
        "self_mix_w": 1.0,
        "nbr_mix_w": 1.0,
        "phys_sig": 0.013666550145707291,
        "gr_power": 2.3550608162294835,
        "gr_sig_def": 121.73853430871536,
        "gr_sig_max": 30.805774523251177,
        "gr_sig_min": 6.129204607494874,
        "gr_sig_mult": 2.2239860932585005,
        "hgr_smooth_r": 0,
        "init_pos_std": 0.6110342659815415,
        "init_rate_std": 0.017227215810478393,
        "jump_prob": 0.002,
        "jump_std": 5.0,
        "likelihood_scale": 12.707813697277029,
        "mom": 0.9986521740591398,
        "n_particles": 800,
        "n_seeds": 32,
        "pn": 0.008607026549212823,
        "rate_clip": 0.0,
        "resamp": 0.22027154128563378,
        "rp": 0.4054319477855724,
        "rr": 0.0010342186487653328,
        "seed0": 0,
        "tvt_clip_margin": 159.36279180208282,
        "tw_gr_smooth_r": 0,
        "vn": 0.001585378256518303
    },
    "p_4": {
        "use_phys": False,
        "use_anchor": False,
        "anchor_mult": 20.0,
        "anchor_ramp_md": 0.0,
        "lk_floor": 1e-05,
        "gr_debias": 0,
        "rate_target": "zero",
        "grid_step": 0.2,
        "tw_source": "tw",
        "self_mix_w": 1.0,
        "nbr_mix_w": 1.0,
        "phys_sig": 0.013666550145707291,
        "gr_power": 2.234682872739614,
        "gr_sig_def": 110.40205312045848,
        "gr_sig_max": 38.93150580901071,
        "gr_sig_min": 6.506708672403766,
        "gr_sig_mult": 3.1167683488416547,
        "hgr_smooth_r": 1,
        "init_pos_std": 1.5134044284698351,
        "init_rate_std": 0.025714909854975217,
        "jump_prob": 0.0,
        "jump_std": 20.0,
        "likelihood_scale": 8.812537885864678,
        "mom": 0.9988504590205197,
        "n_particles": 800,
        "n_seeds": 32,
        "pn": 0.004383324658755245,
        "rate_clip": 3.0,
        "resamp": 0.14043506834054098,
        "rp": 0.7020466979285539,
        "rr": 0.0003017339829762014,
        "seed0": 0,
        "tvt_clip_margin": 131.18430273349367,
        "tw_gr_smooth_r": 0,
        "vn": 0.001343307934466307
    },
    "p_5": {
        "use_phys": False,
        "use_anchor": False,
        "anchor_mult": 20.0,
        "anchor_ramp_md": 0.0,
        "lk_floor": 1e-05,
        "gr_debias": 0,
        "rate_target": "zero",
        "grid_step": 0.2,
        "tw_source": "tw",
        "self_mix_w": 1.0,
        "nbr_mix_w": 1.0,
        "phys_sig": 0.013666550145707291,
        "gr_power": 2.4397815281334925,
        "gr_sig_def": 87.99702225335032,
        "gr_sig_max": 31.337803158934324,
        "gr_sig_min": 3.243687583969081,
        "gr_sig_mult": 2.3999196094002033,
        "hgr_smooth_r": 0,
        "init_pos_std": 0.2700316532641177,
        "init_rate_std": 0.009255150808100809,
        "jump_prob": 0.0,
        "jump_std": 10.0,
        "likelihood_scale": 10.536481330927083,
        "mom": 0.9988865005488281,
        "n_particles": 700,
        "n_seeds": 32,
        "pn": 0.006329415092875301,
        "rate_clip": 2.5,
        "resamp": 0.030096123874617093,
        "rp": 0.8173563935947513,
        "rr": 0.0005595516107073813,
        "seed0": 0,
        "tvt_clip_margin": 106.04712272883125,
        "tw_gr_smooth_r": 1,
        "vn": 0.001283716907451738
    }
}
# ★v41: PF = v38錨付き調整済み1本(pfA)。パラメータは v38_submit_pack.json(=LB提出と同一)
import pickle as _pickle
# ★v90自己完結: v38 struct錨は不使用(PHYSICS_BANks空)。ロードをstub化
_V38_PACK = {"anchor_mult": 20.0}
# (v69: pfA不使用。V41_ANCHOR_MULT は per-bank anchor_mult 欠損時のフォールバックとしてのみ残す)
V41_ANCHOR_MULT = float(_V38_PACK["anchor_mult"])
_V41_STRUCT = {}   # struct-free(未使用)
PHYSICS_BANKS = set()   # ★v69_4: 錨なし(全 use_anchor=False)
def _attach_anchor(x, wid, physics=True):
    """smooth PF入力に struct錨(anc/ancs)を添付。physics=False(物理無しバンク)/prior欠損/行数不一致は錨OFF(ancs=1e9)。"""
    n = len(x["md"]); sp = _V41_STRUCT.get(wid)
    if physics and sp is not None and len(sp["tvt"]) == n:
        x["anc"] = np.asarray(sp["tvt"], float); x["ancs"] = np.asarray(sp["sig"], float)
    else:
        x["anc"] = np.zeros(n); x["ancs"] = np.full(n, 1e9)
    return x
PF_BANK = ["p_1", "p_2", "p_3", "p_4", "p_5"]   # ★v69_4: 与えられた5PF(錨なし)。tw/self/nbr各5=15候補
# ★dec10の仕組み: 全バンク(af/dec10/sharp=MD-10でチューニング)は MD間引きして smooth PF。追加バンクは既定1。
DEC_BANK = {"p_1": 1, "p_2": 1, "p_3": 1, "p_4": 1, "p_5": 1}   # v69_4: 間引きなし(MD-1)

# ============ ★v90: struct-free emission(v90.1検証済 c_learn)を per-bank W で PF尤度へ ============
W_NN_BANK = {"p_1": 0.01, "p_2": 0.01, "p_3": 0.005, "p_4": 0.005, "p_5": 0.005}   # per-bank最適W(fold0調査)
NN_FLOOR = float(os.environ.get("NN_FLOOR", "0.1"))
_SIM_BOOT = {}
_SIM_PATH = (ART_V90 / "sim_boot_v90.pkl") if SPLIT == "train" else (WORK_V90 / "sim_boot_test_v90.pkl")
if _SIM_PATH.exists():
    _SIM_BOOT = _pickle.loads(_SIM_PATH.read_bytes())
    print(f"[v90 emission] sim load: {len(_SIM_BOOT)} wells ({_SIM_PATH.name})")
else:
    print(f"[v90 emission] sim NOT FOUND ({_SIM_PATH.name}) → corrected=素PFにフォールバック")
def _attach_sim(x, wid, W):
    """corrected run用: emission sim(band/st/bid)を添付。W<=0 or sim無し→無添付(=素PF)。"""
    if x is not None and W > 0 and wid in _SIM_BOOT:
        s = _SIM_BOOT[wid]
        x["_simband"] = s["band"]; x["_simst"] = s["st"]; x["_simbid"] = s["bid"]
    return x

# ★v90 SDF予測(U-Net→SDF iter1)を特徴に。train=v89_sdf OOF / test=SDFチェーン(sdf_pred_test_v90.npz)
_SDF_PRED = {}
if SPLIT == "train":
    for _f in range(5):
        _p = PROJECT_V26 / "v64" / "artifacts_v64" / f"v89_sdf_ddil_f{_f}_oof.npz"
        if _p.exists():
            _z = np.load(_p)
            for _k in _z.files: _SDF_PRED[_k] = _z[_k]
else:
    _pt = WORK_V90 / "sdf_pred_test_v90.npz"
    if _pt.exists():
        _z = np.load(_pt)
        for _k in _z.files: _SDF_PRED[_k] = _z[_k]
print(f"[v90 SDF] pred load: {len(_SDF_PRED)} wells ({'train OOF' if SPLIT=='train' else 'test'})")

def _agg_dec_mean(x, s):
    x = np.asarray(x, float); n = len(x)
    cuts = sorted(set([0, n] + list(range(s, n, s))))
    return np.array([float(x[a:b].mean()) for a, b in zip(cuts[:-1], cuts[1:])])
def _decimate_inp(x, DEC):
    md = np.asarray(x["md"], float)
    if DEC <= 1 or len(md) < DEC * 2:
        y = dict(x); y["_fmd"] = md; y["_dmd"] = md; return y
    y = dict(x)
    y["md"] = _agg_dec_mean(md, DEC); y["z"] = _agg_dec_mean(x["z"], DEC); y["gr"] = _agg_dec_mean(x["gr"], DEC)
    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)
    if "ancs" in x: y["ancs"] = _agg_dec_mean(x["ancs"], DEC)
    y["_fmd"] = md; y["_dmd"] = y["md"]; return y
def run_smoother_bank(inps, P, DEC, seed, n_seeds, chunk):
    """★dec10の本質: DEC>1 なら MD間引き(GR mean=denoise, overconfidence減, ~DEC倍速)→ smooth PF → 全解像度に補間。DEC=1 は素通し。"""
    if DEC <= 1:
        return run_smoother_ext(inps, P, seed, n_seeds, chunk)
    dinps = [_decimate_inp(x, DEC) for x in inps]
    outs = run_smoother_ext(dinps, P, seed, n_seeds, chunk)
    res = []
    for i, o in enumerate(outs):
        dm = dinps[i]["_dmd"]; fm = dinps[i]["_fmd"]
        if len(dm) == len(fm):
            res.append(o)
        else:
            res.append(dict(mean=np.interp(fm, dm, np.asarray(o["mean"], float)).astype(np.float32),
                            std=np.interp(fm, dm, np.asarray(o["std"], float)).astype(np.float32)))
    return res

# --- v28 追加: self-PF / neighbor-PF / Z傾斜 ---
K_MAX      = 3
MAX_DIST   = 1500.0   # 近傍(XY重心距離)の上限
NEED_REFS  = 2        # 近傍<2本 -> neighbor無し=tw(通常実行v20相当)にフォールバック
SELF_MIX_W = 1.0      # self疑似typewell(自分prefix)の grid置換率
NBR_MIX_W  = 1.0      # neighbor疑似typewell(近傍train)の grid置換率
ZGRAD_R    = 25       # Z傾斜(dZ/dMD)の移動平均半径(断層の影響回避)
# =======================================================================

_TLS = threading.local()
_FWD_ENS = {}    # (wid, pf_name) -> dict(mean=smoothed, std=smoothed_std)

def _patched_run_pf_ancc(hw, tw_tvt, tw_gr, p):
    """v9 build_well が呼ぶ run_pf_ancc を差替。前計算した smooth PF(平滑平均/std)を返す。"""
    r = _FWD_ENS.get((getattr(_TLS, "wid", None), p.get("name")))
    if r is None:
        return np.array([], np.float32), np.array([], np.float32)
    return r["mean"].astype(np.float32), r["std"].astype(np.float32)


# ==================== v9 INLINE(cell3+5+7 をそのまま持込, exec無し) ====================
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from pathlib import Path
from numba import njit
import multiprocessing
import pandas as pd
import numpy as np
import warnings
import time
import gc
import os

warnings.filterwarnings("ignore")


class CFG:
    # Kaggle competition dataset path
    dataset_path = _V26_DATA

    # Output paths
    output_dir = Path(r"C:\Users\kosaka256\Documents\rogii\tuning_pf_output_v9")
    part_dir = output_dir / "train_feature_parts"
    train_csv_path = output_dir / "train.csv"

    # Resume behavior
    overwrite_parts = False
    overwrite_train_csv = True

    # Debug: set None for all wells, or e.g. 5 for quick test
    max_wells = None


SEED=0
# joblib n_jobs=0 is invalid. Keep this safe for both local Windows and Kaggle.
NCPU=max(1, min(4, multiprocessing.cpu_count()))

FORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
PLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5

BEAMS=[
    (15, 7.885250829793783, 69.70049362431021, 3,"gr"),
    (20, 49.43859691674568, 148.592415555271, 3, "hard"),
    (12, 11.84508580886079, 58.844708544034425, 2, "loose"),
    (15, 12.769364767855363, 89.75744127131894, 3, "mid"),
    (15, 10.693883733824851, 118.15908775467301, 8, "smooth"),
    (40, 67.99404057104671,101.2117946585323, 1, "best")
]

# -----------------------------------------------------------------------------
# Particle Filter parameter sets
# -----------------------------------------------------------------------------
# Add 2-3 good PF parameter dictionaries here.
# Columns are created with suffixes: 1st set -> *_1, 2nd set -> *_2, ...
# KEEP_FIRST_PF_ALIAS=True also keeps the old unsuffixed columns for the 1st set
# so existing training notebooks do not break.
PF_SUFFIX_START = 1
KEEP_FIRST_PF_ALIAS = True

PF_PARAM_SETS_RAW = [{k: v for k, v in PF_LIB[_b].items() if k != "name"} for _b in PF_BANK]   # ★v46: 3バンク(位置対応=pf_1/2/3)


def _make_pf_params(raw, idx):
    """Map Optuna-style PF parameter dict to internal names used by this notebook."""
    r = dict(raw)
    p = {}
    p["name"] = str(r.get("name", f"pf_{idx + PF_SUFFIX_START}"))
    p["suffix"] = str(r.get("suffix", f"_{idx + PF_SUFFIX_START}"))
    p["PF_N"] = int(r.get("PF_N", r.get("n_particles", 600)))
    p["ANCC_N"] = int(r.get("ANCC_N", r.get("n_particles", p["PF_N"])))
    p["PF_N_SEEDS"] = int(r.get("PF_N_SEEDS", r.get("n_seeds", 32)))
    p["PF_SEED0"] = int(r.get("PF_SEED0", SEED + idx * 100000))
    p["PF_LIKELIHOOD_SCALE"] = float(r.get("PF_LIKELIHOOD_SCALE", r.get("likelihood_scale", 20.0)))
    p["PF_INIT_SPR"] = float(r.get("PF_INIT_SPR", r.get("init_pos_std", 0.5)))
    p["PF_INIT_V_STD"] = float(r.get("PF_INIT_V_STD", r.get("init_rate_std", 0.01)))
    p["PF_MOM"] = float(r.get("PF_MOM", r.get("mom", 0.999)))
    p["PF_VN"] = float(r.get("PF_VN", r.get("vn", 0.001)))
    p["PF_PN"] = float(r.get("PF_PN", r.get("pn", 0.005)))
    p["PF_ROUGH_P"] = float(r.get("PF_ROUGH_P", r.get("rp", 0.5)))
    p["PF_ROUGH_V"] = float(r.get("PF_ROUGH_V", r.get("rr", 0.001)))
    p["PF_RESAMP"] = float(r.get("PF_RESAMP", r.get("resamp", 0.5)))
    p["PF_GR_SIG_MIN"] = float(r.get("PF_GR_SIG_MIN", r.get("gr_sig_min", 5.0)))
    p["PF_GR_SIG_MAX"] = float(r.get("PF_GR_SIG_MAX", r.get("gr_sig_max", 50.0)))
    p["PF_GR_SIG_DEF"] = float(r.get("PF_GR_SIG_DEF", r.get("gr_sig_def", 100.0)))
    p["PF_GR_SIG_MULT"] = float(r.get("PF_GR_SIG_MULT", r.get("gr_sig_mult", 2.0)))
    p["PF_TVT_CLIP_MARGIN"] = float(r.get("PF_TVT_CLIP_MARGIN", r.get("tvt_clip_margin", 50.0)))
    p["PF_RATE_CLIP"] = float(r.get("PF_RATE_CLIP", r.get("rate_clip", 0.0)))
    p["PF_GR_POWER"] = float(r.get("PF_GR_POWER", r.get("gr_power", 2.0)))
    p["PF_HGR_SMOOTH_R"] = int(r.get("PF_HGR_SMOOTH_R", r.get("hgr_smooth_r", 0)))
    p["PF_TW_GR_SMOOTH_R"] = int(r.get("PF_TW_GR_SMOOTH_R", r.get("tw_gr_smooth_r", 0)))
    p["PF_JUMP_PROB"] = float(r.get("PF_JUMP_PROB", r.get("jump_prob", 0.0)))
    p["PF_JUMP_STD"] = float(r.get("PF_JUMP_STD", r.get("jump_std", 0.0)))
    p["PF_GR_WIN"] = max(1, 2 * int(p["PF_HGR_SMOOTH_R"]) + 1)
    p["PF_GR_WT"] = float(r.get("PF_GR_WT", r.get("gr_wt", 0.3)))

    # ANCC-PF tracks TVT+Z. Defaults reuse the tuned PF transition/init/resampling params.
    p["ANCC_ALPHA"] = float(r.get("ANCC_ALPHA", p["PF_MOM"]))
    p["ANCC_RN"] = float(r.get("ANCC_RN", p["PF_VN"]))
    p["ANCC_PN"] = float(r.get("ANCC_PN", p["PF_PN"]))
    p["ANCC_IR"] = float(r.get("ANCC_IR", p["PF_INIT_V_STD"]))
    p["ANCC_IS"] = float(r.get("ANCC_IS", p["PF_INIT_SPR"]))
    p["ANCC_RP"] = float(r.get("ANCC_RP", p["PF_ROUGH_P"]))
    p["ANCC_RR"] = float(r.get("ANCC_RR", p["PF_ROUGH_V"]))
    return p

PF_PARAM_SETS = [_make_pf_params(raw, i) for i, raw in enumerate(PF_PARAM_SETS_RAW)]
if len(PF_PARAM_SETS) == 0:
    raise ValueError("PF_PARAM_SETS_RAW must contain at least one parameter set.")
print("PF parameter sets:")
for p in PF_PARAM_SETS:
    print(f"  {p['name']}: suffix={p['suffix']}, N={p['PF_N']}, seeds={p['PF_N_SEEDS']}, jump=({p['PF_JUMP_PROB']}, {p['PF_JUMP_STD']})")

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search ±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # ±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=False)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IR,IS,RP,RR,RESAMP,
              RATE_CLIP,GR_POWER,JUMP_PROB,JUMP_STD,CLIP_MARGIN,SEED_VALUE):
    if SEED_VALUE >= 0:
        np.random.seed(SEED_VALUE)
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+IR*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    log_lik=0.
    tvt_lo=vmin-CLIP_MARGIN
    tvt_hi=vmin+(len(gg)-1)*step+CLIP_MARGIN
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            if RATE_CLIP>0.:
                if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            # Rare jump particles for extreme TVT movement. jump_prob=0 disables this.
            if JUMP_PROB>0. and JUMP_STD>0. and np.random.random()<JUMP_PROB:
                pos[j]+=JUMP_STD*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,tvt_lo); tvt_j=min(tvt_j,tvt_hi)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.; avg_lk=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=abs((gr_v[i]-eg)/gs)
                dp=d**GR_POWER
                lk=max(np.exp(-0.5*dp) if dp<600. else 0.,1e-300)
                old_w=w[j]
                avg_lk+=old_w*lk
                w[j]=old_w*lk; ws+=w[j]
            log_lik+=np.log(max(avg_lk,1e-300))
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            if RATE_CLIP>0.:
                for j in range(N):
                    if rate[j] > RATE_CLIP: rate[j] = RATE_CLIP
                    elif rate[j] < -RATE_CLIP: rate[j] = -RATE_CLIP
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_,log_lik

@njit(cache=False)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP,IP_STD,IV_STD,
          RATE_CLIP,GR_POWER,JUMP_PROB,JUMP_STD,CLIP_MARGIN,SEED_VALUE):
    if SEED_VALUE >= 0:
        np.random.seed(SEED_VALUE)
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+IP_STD*np.random.randn()
        vel[j]=iv+IV_STD*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            if RATE_CLIP>0.:
                if vel[j] > RATE_CLIP: vel[j] = RATE_CLIP
                elif vel[j] < -RATE_CLIP: vel[j] = -RATE_CLIP
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            if JUMP_PROB>0. and JUMP_STD>0. and np.random.random()<JUMP_PROB:
                pos[j]+=JUMP_STD*np.random.randn()
            pos[j]=max(pos[j],vmin-CLIP_MARGIN); pos[j]=min(pos[j],vmin+(len(gg_p)-1)*step+CLIP_MARGIN)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=abs((gr_v[i]-ep)/gs)
                dpow=dp**GR_POWER
                lp=max(np.exp(-0.5*dpow) if dpow<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=abs((gr_sm_v[i]-es)/(gs*1.5))
                    dsp=ds**GR_POWER
                    ls=max(np.exp(-0.5*dsp) if dsp<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_

# Dense grid for O(1) typewell lookup
def _smooth_radius_values(vals, fb, r):
    r = int(r)
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(float(fb))
    if r <= 0:
        return s.to_numpy(np.float32)
    return s.rolling(2*r+1, center=True, min_periods=1).mean().to_numpy(np.float32)

def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr,p):
    # Match pf_tuning_4_early_resume: estimate sigma using the same smoothed
    # horizontal GR and typewell GR that PF likelihood will use.
    gr_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    hgr=_smooth_radius_values(hw['GR'].astype(float).to_numpy(), gr_fb, p["PF_HGR_SMOOTH_R"])
    kn=hw[hw['TVT_input'].notna()]
    if len(kn)<20:
        gs=float(p["PF_GR_SIG_DEF"])
    else:
        kpos=hw.index.get_indexer(kn.index)
        resid=hgr[kpos]-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)
        gs=float(np.nanstd(resid))
        if not np.isfinite(gs) or gs<=0:
            gs=float(p["PF_GR_SIG_DEF"])
    return float(np.clip(gs*p["PF_GR_SIG_MULT"],p["PF_GR_SIG_MIN"],p["PF_GR_SIG_MAX"]))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw,tw_tvt,tw_gr,p):
    tw_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf=_smooth_radius_values(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs=_gr_sig(hw,tw_tvt,tw_gr_pf,p)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr_pf)
    md_ev=ev['MD'].values.astype(np.float64)
    z_ev=ev['Z'].values.astype(np.float64)
    # Tuning notebook interpolates + optionally smooths horizontal GR before likelihood.
    gr_fb=tw_fb
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(gr_fb).to_numpy(dtype=float)
    gr_proc=_smooth_radius_values(gr_full, gr_fb, p["PF_HGR_SMOOTH_R"])
    ev_pos=hw.index.get_indexer(ev.index)
    gr_ev=gr_proc[ev_pos].astype(np.float64)

    preds=[]; stds=[]; liks=[]
    for s in range(int(p["PF_N_SEEDS"])):
        pts,std,ll=_pf_ancc(md_ev,z_ev,gr_ev,gg,gmin,gst,
                            gs,ls,ir,int(p["ANCC_N"]),
                            p["ANCC_ALPHA"],p["ANCC_RN"],p["ANCC_PN"],p["ANCC_IR"],p["ANCC_IS"],p["ANCC_RP"],p["ANCC_RR"],
                            p["PF_RESAMP"],p["PF_RATE_CLIP"],p["PF_GR_POWER"],p["PF_JUMP_PROB"],p["PF_JUMP_STD"],
                            p["PF_TVT_CLIP_MARGIN"],int(p["PF_SEED0"])+s)
        preds.append(pts); stds.append(std); liks.append(ll)
    pred_arr=np.stack(preds,0)
    std_arr=np.stack(stds,0)
    lik_arr=np.asarray(liks,dtype=np.float64)
    lik_arr=lik_arr-np.nanmax(lik_arr)
    ww=np.exp(lik_arr/max(float(p["PF_LIKELIHOOD_SCALE"]),1e-6))
    ww=ww/max(float(ww.sum()),1e-300)
    ens=(ww[:,None]*pred_arr).sum(0)
    ens_var=(ww[:,None]*(std_arr**2+(pred_arr-ens[None,:])**2)).sum(0)
    return ens.astype(np.float32),np.sqrt(np.maximum(ens_var,0.)).astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,p):
    tw_fb=float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf=_smooth_radius_values(tw_gr, tw_fb, p["PF_TW_GR_SMOOTH_R"]).astype(np.float64)
    gs=_gr_sig(hw,tw_tvt,tw_gr_pf,p)
    tw_s=pd.Series(tw_gr_pf).rolling(p["PF_GR_WIN"],center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr_pf)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_fb=tw_fb
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(gr_fb).to_numpy(dtype=float)
    gr_proc=_smooth_radius_values(gr_full, gr_fb, p["PF_HGR_SMOOTH_R"])
    gr_sm=pd.Series(gr_proc).rolling(p["PF_GR_WIN"],center=True,min_periods=1).mean().to_numpy(dtype=float)
    ev_pos=hw.index.get_indexer(ev.index)
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   gr_proc[ev_pos].astype(np.float64),
                   gr_sm[ev_pos].astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,int(p["PF_N"]),
                   p["PF_MOM"],p["PF_VN"],p["PF_PN"],p["PF_GR_WT"],p["PF_ROUGH_P"],p["PF_ROUGH_V"],p["PF_RESAMP"],
                   p["PF_INIT_SPR"],p["PF_INIT_V_STD"],p["PF_RATE_CLIP"],p["PF_GR_POWER"],p["PF_JUMP_PROB"],p["PF_JUMP_STD"],
                   p["PF_TVT_CLIP_MARGIN"],int(p["PF_SEED0"]))
    return pts.astype(np.float32),std.astype(np.float32)


_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_p0=PF_PARAM_SETS[0]
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,
         _p0["ANCC_ALPHA"],_p0["ANCC_RN"],_p0["ANCC_PN"],_p0["ANCC_IR"],_p0["ANCC_IS"],_p0["ANCC_RP"],_p0["ANCC_RR"],_p0["PF_RESAMP"],
         _p0["PF_RATE_CLIP"],_p0["PF_GR_POWER"],_p0["PF_JUMP_PROB"],_p0["PF_JUMP_STD"],_p0["PF_TVT_CLIP_MARGIN"],123)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,
      _p0["PF_MOM"],_p0["PF_VN"],_p0["PF_PN"],_p0["PF_GR_WT"],_p0["PF_ROUGH_P"],_p0["PF_ROUGH_V"],_p0["PF_RESAMP"],_p0["PF_INIT_SPR"],_p0["PF_INIT_V_STD"],
      _p0["PF_RATE_CLIP"],_p0["PF_GR_POWER"],_p0["PF_JUMP_PROB"],_p0["PF_JUMP_STD"],_p0["PF_TVT_CLIP_MARGIN"],123)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

class FormationPlaneKNN:
    def __init__(self,well_ids,data_dir):
        rows=[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y']+FORMATIONS).dropna()
            except: continue
            if len(df)==0: continue
            row={'wid':wid,'x':float(df['X'].median()),'y':float(df['Y'].median())}
            for c in FORMATIONS: row[f'{c}_m']=float(df[c].median())
            rows.append(row)
        self.df=pd.DataFrame(rows); self.wmap={w:i for i,w in enumerate(self.df['wid'])}
        xy=self.df[['x','y']].to_numpy(); self.scale=np.where(xy.std(0)<1e-3,1.,xy.std(0))
        self.tree=cKDTree(xy/self.scale)
        self.xa=self.df['x'].to_numpy(); self.ya=self.df['y'].to_numpy()
        self.fa=self.df[[f'{c}_m' for c in FORMATIONS]].to_numpy(np.float64)

    def impute(self,xy_q,self_wid=None,k=PLANE_K):
        q=xy_q/self.scale; nf=min(k+5,len(self.df))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid in self.wmap: dist=np.where(idx==self.wmap[self_wid],np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.).astype(np.float64)
        xn=self.xa[ik]; yn=self.ya[ik]; fn=self.fa[ik]; wx=w*xn; wy=w*yn
        A=np.zeros((len(q),3,3))
        A[:,0,0]=(wx*xn).sum(1); A[:,0,1]=(wx*yn).sum(1); A[:,0,2]=wx.sum(1)
        A[:,1,0]=A[:,0,1]; A[:,1,1]=(wy*yn).sum(1); A[:,1,2]=wy.sum(1)
        A[:,2,0]=A[:,0,2]; A[:,2,1]=A[:,1,2]; A[:,2,2]=w.sum(1)
        A[:,0,0]+=1e-9; A[:,1,1]+=1e-9; A[:,2,2]+=1e-9
        rhs=np.stack([(wx[:,:,None]*fn).sum(1),(wy[:,:,None]*fn).sum(1),(w[:,:,None]*fn).sum(1)],1)
        try: coef=np.linalg.solve(A,rhs)
        except:
            coef=np.zeros((len(q),3,6))
            for r in range(len(q)):
                try: coef[r]=np.linalg.pinv(A[r])@rhs[r]
                except: pass
        Xq=xy_q[:,0]; Yq=xy_q[:,1]
        pred=(Xq[:,None]*coef[:,0,:]+Yq[:,None]*coef[:,1,:]+coef[:,2,:]).astype(np.float32)
        pred[~vk.any(1)]=self.fa.mean(0)
        return pred,np.where(vk,dk,np.inf).min(1).astype(np.float32)

class DenseANCCImputer:
    def __init__(self,well_ids,data_dir,spw=DENSE_SPW):
        xs,ys,anccs,wids=[],[],[],[]
        for wid in well_ids:
            p=data_dir/f'{wid}__horizontal_well.csv'
            try: df=pd.read_csv(p,usecols=['X','Y','ANCC']).dropna()
            except: continue
            if len(df)==0: continue
            ix=np.linspace(0,len(df)-1,min(spw,len(df)),dtype=int); s=df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid]*len(s))
        self.xy=np.column_stack([np.concatenate(xs),np.concatenate(ys)])
        self.ancc=np.concatenate(anccs).astype(np.float32); self.wids=np.array(wids)
        self.scale=np.where(self.xy.std(0)<1e-3,1.,self.xy.std(0))
        self.tree=cKDTree(self.xy/self.scale)

    def impute(self,xy_q,self_wid=None,k=DENSE_K,nfetch=5000):
        xy_q=np.atleast_2d(xy_q); q=xy_q/self.scale; nf=min(nfetch,len(self.ancc))
        dist,idx=self.tree.query(q,k=nf,workers=-1)
        if self_wid: dist=np.where(self.wids[idx]==self_wid,np.inf,dist)
        ord=np.argpartition(dist,min(k-1,nf-1),1)[:,:k]
        dk=np.take_along_axis(dist,ord,1); ik=np.take_along_axis(idx,ord,1)
        vk=np.isfinite(dk); w=np.where(vk,1./(dk+1e-3),0.)
        sw=w.sum(1); safe=np.where(sw<1e-9,1.,sw); an=self.ancc[ik]
        ap=(an*w).sum(1)/safe; ap=np.where(sw<1e-9,float(self.ancc.mean()),ap)
        var=((an-ap[:,None])**2*w).sum(1)/safe
        return ap.astype(np.float32),np.sqrt(np.maximum(var,0.)).astype(np.float32),np.where(vk,dk,np.inf).min(1).astype(np.float32)

hw_paths=sorted((CFG.dataset_path / "train").glob('*__horizontal_well.csv'))
train_wids=[p.stem.replace('__horizontal_well','') for p in hw_paths]
FI=FormationPlaneKNN(train_wids,CFG.dataset_path / "train")
DI=DenseANCCImputer(train_wids,CFG.dataset_path / "train")

_FI=FI; _DI=DI
ANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)
BEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)
SC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)
PF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)


def _rolling_mean_np(x, w):
    return pd.Series(np.asarray(x, np.float32)).rolling(w, center=True, min_periods=1).mean().to_numpy(np.float32)

def _rolling_std_np(x, w):
    return pd.Series(np.asarray(x, np.float32)).rolling(w, center=True, min_periods=1).std().fillna(0.).to_numpy(np.float32)

def _safe_grad(y, x=None):
    y = np.asarray(y, np.float32)
    n = len(y)
    if n <= 1:
        return np.zeros(n, np.float32)
    if x is None:
        g = np.gradient(y).astype(np.float32)
    else:
        x = np.asarray(x, np.float32)
        dx = np.gradient(x).astype(np.float32)
        dx = np.where(np.abs(dx) < 1e-6, np.nan, dx)
        g = np.gradient(y).astype(np.float32) / dx
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return g.astype(np.float32)

def _add_path_shape_features(feats, prefix, path, md=None):
    """Local geometry of a candidate TVT path. No target is used."""
    path = np.asarray(path, np.float32)
    rate = _safe_grad(path, md)
    acc = _safe_grad(rate, md)
    feats[f'{prefix}_rate'] = rate.astype(np.float32)
    feats[f'{prefix}_acc'] = acc.astype(np.float32)
    feats[f'{prefix}_rate_abs'] = np.abs(rate).astype(np.float32)
    feats[f'{prefix}_acc_abs'] = np.abs(acc).astype(np.float32)
    feats[f'{prefix}_rate_rmean21'] = _rolling_mean_np(rate, 21)
    feats[f'{prefix}_rate_rstd21'] = _rolling_std_np(rate, 21)
    feats[f'{prefix}_acc_rmean21'] = _rolling_mean_np(acc, 21)

def _add_gr_match_features(feats, prefix, path, hgr, tw_tvt, tw_gr, offs):
    """How well a candidate TVT path explains the observed horizontal GR."""
    path = np.asarray(path, np.float32)
    hgr = np.asarray(hgr, np.float32)
    offs = np.asarray(offs, np.float32)
    gr_mat = np.stack([np.interp(path + float(o), tw_tvt, tw_gr) for o in offs], axis=1).astype(np.float32)
    resid = (hgr[:, None] - gr_mat).astype(np.float32)
    abs_resid = np.abs(resid)
    zero_idx = int(np.argmin(np.abs(offs)))
    best_idx = abs_resid.argmin(axis=1)
    row_idx = np.arange(len(path))
    r0 = resid[:, zero_idx].astype(np.float32)
    best_signed = resid[row_idx, best_idx].astype(np.float32)
    feats[f'{prefix}_gr_resid0'] = r0
    feats[f'{prefix}_gr_abs0'] = np.abs(r0).astype(np.float32)
    feats[f'{prefix}_gr_abs_min'] = abs_resid[row_idx, best_idx].astype(np.float32)
    feats[f'{prefix}_gr_best_off'] = offs[best_idx].astype(np.float32)
    feats[f'{prefix}_gr_best_resid'] = best_signed
    feats[f'{prefix}_gr_resid0_rmean21'] = _rolling_mean_np(r0, 21)
    feats[f'{prefix}_gr_resid0_rstd21'] = _rolling_std_np(r0, 21)
    feats[f'{prefix}_gr_abs0_rmean21'] = _rolling_mean_np(np.abs(r0), 21)

def _pairwise_abs_stats(mat):
    mat = np.asarray(mat, np.float32)
    k = mat.shape[1]
    if k < 2:
        z = np.zeros(mat.shape[0], np.float32)
        return z, z
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(np.abs(mat[:, i] - mat[:, j]).astype(np.float32))
    diffs = np.stack(vals, axis=1)
    return diffs.mean(axis=1).astype(np.float32), diffs.max(axis=1).astype(np.float32)

def _corr_summary(mat):
    mat = np.asarray(mat, np.float32)
    k = mat.shape[1]
    if k < 2:
        return 0.0, 0.0, 0.0
    vals = []
    for i in range(k):
        xi = mat[:, i]
        xi_std = float(np.std(xi))
        for j in range(i + 1, k):
            xj = mat[:, j]
            xj_std = float(np.std(xj))
            if xi_std < 1e-6 or xj_std < 1e-6:
                vals.append(0.0)
            else:
                vals.append(float(np.corrcoef(xi, xj)[0, 1]))
    vals = np.nan_to_num(np.asarray(vals, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(vals.mean()), float(vals.min()), float(vals.max())


def _add_pf_feature_columns(feats, pf_runs, last_tvt, tvt_f_ancc, tvt_dense, hgr, tw_tvt, tw_gr, sc,
                            hmd=None, beam_mean=None, sc_ens=None, hyb_ref=None):
    """Add PF-derived feature columns for all parameter sets.

    Added feature families:
      - per-PF GR residual quality around the candidate path
      - per-PF local rate/curvature of the candidate path
      - pairwise PF differences/correlations and PF ensemble consensus
      - PF ensemble disagreement vs beam/sc/spatial/dense references

    Naming rule:
      - set 1 -> suffix _1, set 2 -> suffix _2, ...
      - if KEEP_FIRST_PF_ALIAS=True, also add old unsuffixed columns from set 1.
    """
    def add_one(r, suffix):
        pf_a = r["pf_ancc"].astype(np.float32)
        std_a = r["pf_ancc_std"].astype(np.float32)
        has_z = bool(r["has_z"])
        pf_z = r["pf_z"].astype(np.float32) if has_z else sc(last_tvt)
        feats[f'pf_ancc{suffix}'] = pf_a
        feats[f'pf_ancc_std{suffix}'] = std_a
        feats[f'pf_ancc_delta{suffix}'] = (pf_a - np.float32(last_tvt)).astype(np.float32)
        feats[f'pf_z{suffix}'] = pf_z
        feats[f'pf_z_delta{suffix}'] = (pf_z - np.float32(last_tvt)).astype(np.float32) if has_z else sc(0.)
        feats[f'pf_vs_z{suffix}'] = (pf_a - pf_z).astype(np.float32) if has_z else sc(0.)
        feats[f'pf_vs_spatial{suffix}'] = (pf_a - tvt_f_ancc).astype(np.float32)
        feats[f'pf_vs_dense{suffix}'] = (pf_a - tvt_dense).astype(np.float32)
        for o in PF_OFFS:
            feats[f'tdpf{int(o)}{suffix}'] = hgr - np.interp(pf_a + o, tw_tvt, tw_gr).astype(np.float32)

        prefix = 'pf' if suffix == '' else f'pf{suffix}'
        _add_gr_match_features(feats, prefix, pf_a, hgr, tw_tvt, tw_gr, PF_OFFS)
        if hmd is not None:
            _add_path_shape_features(feats, prefix, pf_a, hmd)

    if KEEP_FIRST_PF_ALIAS and len(pf_runs) > 0:
        add_one(pf_runs[0], "")
    for r in pf_runs:
        add_one(r, r["suffix"])

    if len(pf_runs) >= 2:
        pf_stack = np.stack([r["pf_ancc"].astype(np.float32) for r in pf_runs], axis=1)
        pf_mean = pf_stack.mean(1).astype(np.float32)
        pf_med = np.median(pf_stack, axis=1).astype(np.float32)
        pf_pair_mean, pf_pair_max = _pairwise_abs_stats(pf_stack)
        c_mean, c_min, c_max = _corr_summary(pf_stack)
        feats['pf_ancc_mean'] = pf_mean
        feats['pf_ancc_med'] = pf_med
        feats['pf_ancc_std_between'] = pf_stack.std(1).astype(np.float32)
        feats['pf_ancc_range_between'] = (pf_stack.max(1) - pf_stack.min(1)).astype(np.float32)
        feats['pf_ancc_iqr_between'] = (np.percentile(pf_stack, 75, axis=1) - np.percentile(pf_stack, 25, axis=1)).astype(np.float32)
        feats['pf_ancc_pair_absmean_between'] = pf_pair_mean
        feats['pf_ancc_pair_absmax_between'] = pf_pair_max
        feats['pf_ancc_mean_delta'] = (pf_mean - np.float32(last_tvt)).astype(np.float32)
        feats['pf_ancc_med_delta'] = (pf_med - np.float32(last_tvt)).astype(np.float32)
        feats['pf_ancc_corr_mean'] = sc(c_mean)
        feats['pf_ancc_corr_min'] = sc(c_min)
        feats['pf_ancc_corr_max'] = sc(c_max)

        for i in range(len(pf_runs)):
            si = pf_runs[i]['suffix']
            feats[f'pf_dev_mean{si}'] = (pf_stack[:, i] - pf_mean).astype(np.float32)
            feats[f'pf_dev_med{si}'] = (pf_stack[:, i] - pf_med).astype(np.float32)
        for i in range(len(pf_runs)):
            for j in range(i + 1, len(pf_runs)):
                si = pf_runs[i]['suffix'].replace('_', '')
                sj = pf_runs[j]['suffix'].replace('_', '')
                diff = (pf_stack[:, i] - pf_stack[:, j]).astype(np.float32)
                feats[f'pf_diff_{si}_{sj}'] = diff
                feats[f'pf_absdiff_{si}_{sj}'] = np.abs(diff).astype(np.float32)

        _add_gr_match_features(feats, 'pf_mean', pf_mean, hgr, tw_tvt, tw_gr, PF_OFFS)
        _add_gr_match_features(feats, 'pf_med', pf_med, hgr, tw_tvt, tw_gr, PF_OFFS)
        if hmd is not None:
            _add_path_shape_features(feats, 'pf_mean', pf_mean, hmd)
            _add_path_shape_features(feats, 'pf_med', pf_med, hmd)

        if beam_mean is not None:
            feats['pf_mean_vs_beam_mean'] = (pf_mean - np.asarray(beam_mean, np.float32)).astype(np.float32)
            feats['pf_med_vs_beam_mean'] = (pf_med - np.asarray(beam_mean, np.float32)).astype(np.float32)
        if sc_ens is not None:
            feats['pf_mean_vs_sc_ens'] = (pf_mean - np.asarray(sc_ens, np.float32)).astype(np.float32)
            feats['pf_med_vs_sc_ens'] = (pf_med - np.asarray(sc_ens, np.float32)).astype(np.float32)
        if hyb_ref is not None:
            feats['pf_mean_vs_hyb'] = (pf_mean - np.asarray(hyb_ref, np.float32)).astype(np.float32)
        feats['pf_mean_vs_spatial'] = (pf_mean - tvt_f_ancc).astype(np.float32)
        feats['pf_mean_vs_dense'] = (pf_mean - tvt_dense).astype(np.float32)

        z_list = [r["pf_z"].astype(np.float32) for r in pf_runs if bool(r["has_z"])]
        if len(z_list) >= 2:
            z_stack = np.stack(z_list, axis=1)
            z_mean = z_stack.mean(1).astype(np.float32)
            z_med = np.median(z_stack, axis=1).astype(np.float32)
            z_pair_mean, z_pair_max = _pairwise_abs_stats(z_stack)
            zc_mean, zc_min, zc_max = _corr_summary(z_stack)
            feats['pf_z_mean'] = z_mean
            feats['pf_z_med'] = z_med
            feats['pf_z_std_between'] = z_stack.std(1).astype(np.float32)
            feats['pf_z_range_between'] = (z_stack.max(1) - z_stack.min(1)).astype(np.float32)
            feats['pf_z_pair_absmean_between'] = z_pair_mean
            feats['pf_z_pair_absmax_between'] = z_pair_max
            feats['pf_z_corr_mean'] = sc(zc_mean)
            feats['pf_z_corr_min'] = sc(zc_min)
            feats['pf_z_corr_max'] = sc(zc_max)
            feats['pf_mean_vs_z_mean'] = (pf_mean - z_mean).astype(np.float32)
            feats['pf_med_vs_z_med'] = (pf_med - z_med).astype(np.float32)


def build_well(hw_path,tw_path,is_train):
    global _FI,_DI
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    try:
        hw=pd.read_csv(hw_path); tw=pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    pf_runs=[]
    for pconf in PF_PARAM_SETS:
        pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr,pconf)
        if len(pf_a)==0: return None
        pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr,pconf)
        has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))
        pf_runs.append({
            "name": pconf["name"],
            "suffix": pconf["suffix"],
            "pf_ancc": pf_a.astype(np.float32),
            "pf_ancc_std": std_a.astype(np.float32),
            "pf_z": pf_z.astype(np.float32) if len(pf_z)==len(pf_a) else np.full(len(pf_a), np.nan, np.float32),
            "pf_z_std": std_z.astype(np.float32) if len(std_z)==len(pf_a) else np.full(len(pf_a), np.nan, np.float32),
            "has_z": has_z,
        })
    pf_primary=pf_runs[0]
    pf_use=pf_primary["pf_ancc"].astype(np.float32)
    std_use=pf_primary["pf_ancc_std"].astype(np.float32)
    pf_z=pf_primary["pf_z"].astype(np.float32)
    has_z=bool(pf_primary["has_z"])

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT ±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    # Robust reference beam: older versions used tags 'cons'/'sm5', this notebook uses 'gr'/'smooth'.
    _beam_a=bpaths['cons'] if 'cons' in bpaths else bpaths.get('gr', next(iter(bpaths.values())))
    _beam_b=bpaths['sm5'] if 'sm5' in bpaths else bpaths.get('smooth', _beam_a)
    beam_ref=(_beam_a+_beam_b)/2.

    # Multi-scale NCC → score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    swid=wid if is_train else None
    xy_ev=ev[['X','Y']].to_numpy(np.float64); xy_kn=kn[['X','Y']].to_numpy(np.float64)
    form_ev,knn_d=_FI.impute(xy_ev,self_wid=swid)
    form_kn,_   =_FI.impute(xy_kn,self_wid=swid)
    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)

    # Per-formation: segment b_well (early/mid/late/wls) + TVT + known-zone RMSE
    tvt_fs={}; form_rmse={}; form_list=[]
    for fi2,fn in enumerate(FORMATIONS):
        b_full,b_early,b_mid,b_late,b_wls=seg_b_well(ktvt,z_kn,form_kn[:,fi2])
        tvt_f  =(-z_ev+form_ev[:,fi2]+b_full ).astype(np.float32)
        tvt_fw =(-z_ev+form_ev[:,fi2]+b_wls  ).astype(np.float32)
        tvt_f50=(-z_ev+form_ev[:,fi2]+b_late ).astype(np.float32)
        tvt_fs[f'tvtF_{fn}']=tvt_f; tvt_fs[f'tvtFw_{fn}']=tvt_fw
        tvt_fs[f'tvtF50_{fn}']=tvt_f50
        tvt_fs[f'bw_{fn}']=np.float32(b_full); tvt_fs[f'bww_{fn}']=np.float32(b_wls)
        tvt_fs[f'bw50_{fn}']=np.float32(b_late)
        tvt_fs[f'bw_early_{fn}']=np.float32(b_early)   # NEW: early segment
        tvt_fs[f'bw_mid_{fn}']=np.float32(b_mid)       # NEW: mid segment
        form_rmse[fn]=float(np.sqrt(np.mean((ktvt-(-z_kn+form_kn[:,fi2]+b_full))**2)))
        form_list.append(tvt_f)

    fs=np.stack(form_list,1)
    form_mean_d=(fs.mean(1)-last_tvt).astype(np.float32)
    form_std_d =fs.std(1).astype(np.float32)
    form_rng_d =(fs.max(1)-fs.min(1)).astype(np.float32)

    d_ancc,d_std,d_dist=_DI.impute(xy_ev,self_wid=swid)
    d_kn,d_std_kn,_=_DI.impute(xy_kn,self_wid=swid)
    b_vd=ktvt+z_kn-d_kn
    _,b_de,b_dm,b_dl,b_dw=seg_b_well(ktvt,z_kn,d_kn)
    b_d=float(np.median(b_vd))
    tvt_dense  =(-z_ev+d_ancc+b_d  ).astype(np.float32)
    tvt_densew =(-z_ev+d_ancc+b_dw ).astype(np.float32)
    tvt_dense50=(-z_ev+d_ancc+b_dl ).astype(np.float32)
    res_kn=ktvt+z_kn-d_kn
    d_rmse=float(np.sqrt(np.mean(res_kn**2))); d_bias=float(np.mean(res_kn)); d_nb_std=float(np.mean(d_std_kn))

    candidate_names = [f"pf_ancc{r['suffix']}" for r in pf_runs] + [f"beam_{k}" for k in bpaths.keys()] + [
        'sc8', 'sc15', 'sc25', 'sc_ens', 'spatial_ancc', 'dense_ancc'
    ]
    all_sigs=[r['pf_ancc'] for r in pf_runs]+[p for p in bpaths.values()]+[sc8,sc15,sc25,sc_ens,tvt_fs['tvtF_ANCC'],tvt_dense]
    sig_mat=np.stack(all_sigs,1).astype(np.float32)
    sig_std=sig_mat.std(1).astype(np.float32)
    sig_mean=(sig_mat.mean(1)-last_tvt).astype(np.float32)
    sig_med=np.median(sig_mat,axis=1).astype(np.float32)
    sig_q10=np.percentile(sig_mat,10,axis=1).astype(np.float32)
    sig_q25=np.percentile(sig_mat,25,axis=1).astype(np.float32)
    sig_q75=np.percentile(sig_mat,75,axis=1).astype(np.float32)
    sig_q90=np.percentile(sig_mat,90,axis=1).astype(np.float32)
    sig_iqr=(sig_q75-sig_q25).astype(np.float32)
    sig_range=(sig_mat.max(1)-sig_mat.min(1)).astype(np.float32)
    sig_mad=np.mean(np.abs(sig_mat-sig_med[:,None]),axis=1).astype(np.float32)
    sig_centered=(sig_mat-sig_mat.mean(1,keepdims=True)).astype(np.float32)
    sig_skew=(np.mean(sig_centered**3,axis=1)/(np.maximum(sig_std,1e-6)**3)).astype(np.float32)

    sig_gr_pred_mat=np.stack([np.interp(sig_mat[:,j],tw_tvt,tw_gr) for j in range(sig_mat.shape[1])],axis=1).astype(np.float32)
    sig_gr_resid_mat=(hgr[:,None]-sig_gr_pred_mat).astype(np.float32)
    sig_gr_abs_mat=np.abs(sig_gr_resid_mat).astype(np.float32)
    sig_gr_best_idx=sig_gr_abs_mat.argmin(axis=1)
    _sig_rows=np.arange(len(hgr))
    sig_gr_best_path=sig_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_best_abs=sig_gr_abs_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_best_resid=sig_gr_resid_mat[_sig_rows,sig_gr_best_idx].astype(np.float32)
    sig_gr_pred_std=sig_gr_pred_mat.std(1).astype(np.float32)
    sig_gr_resid_mean=sig_gr_resid_mat.mean(1).astype(np.float32)
    sig_gr_resid_std=sig_gr_resid_mat.std(1).astype(np.float32)

    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),  # score-weighted ensemble
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sig_std':sig_std,'sig_mean_d':sig_mean,
        'sig_med_d':(sig_med-np.float32(last_tvt)).astype(np.float32),
        'sig_q10_d':(sig_q10-np.float32(last_tvt)).astype(np.float32),
        'sig_q25_d':(sig_q25-np.float32(last_tvt)).astype(np.float32),
        'sig_q75_d':(sig_q75-np.float32(last_tvt)).astype(np.float32),
        'sig_q90_d':(sig_q90-np.float32(last_tvt)).astype(np.float32),
        'sig_iqr':sig_iqr,'sig_range':sig_range,'sig_mad':sig_mad,'sig_skew':sig_skew,
        'sig_gr_pred_std':sig_gr_pred_std,
        'sig_gr_resid_mean':sig_gr_resid_mean,'sig_gr_resid_std':sig_gr_resid_std,
        'sig_gr_best_d':(sig_gr_best_path-np.float32(last_tvt)).astype(np.float32),
        'sig_gr_best_idx':sig_gr_best_idx.astype(np.float32),
        'sig_gr_best_abs':sig_gr_best_abs,'sig_gr_best_resid':sig_gr_best_resid,
        'sig_gr_best_vs_med':(sig_gr_best_path-sig_med).astype(np.float32),
        **tvt_fs,
        **{f'frm_rmse_{fn}':sc(form_rmse[fn]) for fn in FORMATIONS},
        'form_mean_d':form_mean_d,'form_std_d':form_std_d,'form_rng_d':form_rng_d,
        'spatial_ancc_d':(form_ev[:,0]-np.float32(np.interp(last_tvt,tw_tvt,tw_gr))),
        'spatial_knn_dist':knn_d,
        'dense_ancc':d_ancc,'dense_std':d_std,'dense_dist':d_dist,
        'tvt_dense_d' :(tvt_dense -last_tvt).astype(np.float32),
        'tvt_densew_d':(tvt_densew-last_tvt).astype(np.float32),
        'tvt_dense50_d':(tvt_dense50-last_tvt).astype(np.float32),
        'dense_rmse':sc(d_rmse),'dense_bias':sc(d_bias),'dense_nb_std':sc(d_nb_std),
        'spatial_vs_dense':(tvt_fs['tvtF_ANCC']-tvt_dense).astype(np.float32),
        'beam_vs_spatial':(beam_ref-tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam':(sc_ens-beam_ref).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 -last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    _add_pf_feature_columns(
        feats, pf_runs, last_tvt, tvt_fs['tvtF_ANCC'], tvt_dense, hgr, tw_tvt, tw_gr, sc,
        hmd=hmd,
        beam_mean=(np.float32(last_tvt) + feats['beam_mean_d']).astype(np.float32),
        sc_ens=sc_ens,
        hyb_ref=hyb_ref,
    )

    # Shape/GR-match features for non-PF ensemble references.
    beam_mean_path=(np.float32(last_tvt)+feats['beam_mean_d']).astype(np.float32)
    beam_med_path=(np.float32(last_tvt)+feats['beam_med_d']).astype(np.float32)
    _add_path_shape_features(feats,'beam_mean',beam_mean_path,hmd)
    _add_path_shape_features(feats,'beam_med',beam_med_path,hmd)
    _add_path_shape_features(feats,'sc_ens',sc_ens,hmd)
    _add_path_shape_features(feats,'hyb',hyb_ref,hmd)
    _add_path_shape_features(feats,'sig_med',sig_med,hmd)
    _add_path_shape_features(feats,'sig_gr_best',sig_gr_best_path,hmd)
    _add_gr_match_features(feats,'beam_mean',beam_mean_path,hgr,tw_tvt,tw_gr,BEAM_OFFS)
    _add_gr_match_features(feats,'beam_med',beam_med_path,hgr,tw_tvt,tw_gr,BEAM_OFFS)
    _add_gr_match_features(feats,'sc_ens',sc_ens,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'hyb',hyb_ref,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'sig_med',sig_med,hgr,tw_tvt,tw_gr,SC_OFFS)
    _add_gr_match_features(feats,'sig_gr_best',sig_gr_best_path,hgr,tw_tvt,tw_gr,SC_OFFS)

    for k,v in rolls.items(): feats[k]=v

    # Additional GR morphology features. These use only observed horizontal/typewell GR.
    feats['gr_z21']=((hgr-rolls['grm21'])/(rolls['grs21']+1e-3)).astype(np.float32)
    feats['gr_z51']=((hgr-rolls['grm51'])/(rolls['grs51']+1e-3)).astype(np.float32)
    feats['gr_abs_d1']=np.abs(gr_d1).astype(np.float32)
    feats['gr_abs_d2']=np.abs(gr_d2).astype(np.float32)
    feats['gr_d1_rmean21']=_rolling_mean_np(gr_d1,21)
    feats['gr_abs_d1_rmean21']=_rolling_mean_np(np.abs(gr_d1),21)
    feats['gr_rough21']=_rolling_mean_np(np.abs(gr_d1),21)
    feats['gr_rough51']=_rolling_mean_np(np.abs(gr_d1),51)
    _tw_gr_sorted=np.sort(tw_gr.astype(np.float32))
    _kg_sorted=np.sort(kgr.astype(np.float32)) if len(kgr)>0 else _tw_gr_sorted
    feats['gr_pct_tw']=(np.searchsorted(_tw_gr_sorted,hgr,side='right')/max(len(_tw_gr_sorted),1)).astype(np.float32)
    feats['gr_pct_known']=(np.searchsorted(_kg_sorted,hgr,side='right')/max(len(_kg_sorted),1)).astype(np.float32)
    feats['gr_z_known']=((hgr-np.float32(np.mean(kgr)))/(np.float32(np.std(kgr))+1e-3)).astype(np.float32)
    feats['gr_z_tw']=((hgr-np.float32(np.mean(tw_gr)))/(np.float32(np.std(tw_gr))+1e-3)).astype(np.float32)
    feats['gr_vs_known_last']=(hgr-np.float32(kgr[-1])).astype(np.float32)
    feats['gr_vs_known_mean']=(hgr-np.float32(np.mean(kgr))).astype(np.float32)

    # Geometry roughness of the actual drilled path.
    step_xy=np.sqrt(dxdmd**2+dydmd**2).astype(np.float32)
    feats['xy_rate']=step_xy
    feats['xyz_rate']=np.sqrt(dxdmd**2+dydmd**2+dzdmd**2).astype(np.float32)
    feats['z_rate_abs']=np.abs(dzdmd).astype(np.float32)
    feats['dzdmd_rmean21']=_rolling_mean_np(dzdmd,21)
    feats['dzdmd_rstd21']=_rolling_std_np(dzdmd,21)
    feats['xy_rate_rmean21']=_rolling_mean_np(step_xy,21)
    feats['xy_rate_rstd21']=_rolling_std_np(step_xy,21)

    result=pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target']=(ev['TVT'].to_numpy(np.float32)-np.float32(last_tvt))
    return result

def build_dataset(paths,is_train,label):
    args=[(str(p),str(p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv'),is_train)
          for p in paths
          if (p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0=time.time()
    res=Parallel(n_jobs=NCPU,prefer='threads',verbose=3)(
        delayed(build_well)(hp,tp,it) for hp,tp,it in args)
    parts=[r for r in res if r is not None]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

# ==================== v26 smooth PF エンジン(固定ラグ, tune_smoother_r0_v25 と同一) ====================
def build_smoother_inputs(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力を hw/tw と生PFパラメータから作る(v9の _smooth_radius_values/_grid を再利用)。無効なら None。"""
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return None
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    tw_gr_pf = _smooth_radius_values(tw_gr, tw_fb, P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(tw_tvt, tw_gr_pf)
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    ev_pos = hw.index.get_indexer(ev.index); kpos = hw.index.get_indexer(kn.index)
    kn_tvtin = kn["TVT_input"].to_numpy(float)
    if len(kpos) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[kpos] - np.interp(kn_tvtin, tw_tvt, tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    ls = float(kn["TVT_input"].iloc[-1] + kn["Z"].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    return dict(md=ev["MD"].to_numpy(float), z=ev["Z"].to_numpy(float), gr=gr_sm[ev_pos].astype(np.float64),
                gg=gg, gmin=gmin, gst=gst, gs=gs, ls=ls, ir=ir, N=int(P["n_particles"]))

NN_NB = 90; NN_ND = 181; NN_STEP = 0.5      # ★v90: emission(c_learn)δ格子(offs=(j-NB)*STEP, boot中心)


def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen,
                       ns_ctr=None, ns_bid=None, ns_band=None, b_base=None, dtype=V26_DTYPE):
    B, T = md.shape
    NN_W = float(P.get("nn_w", 0.0)); NN_FL = float(P.get("nn_floor", 0.1))   # ★v90 emission温度/床(per-bank)
    _use_nn = NN_W > 0 and ns_band is not None
    ALPHA = P["mom"]; RN = P["vn"]; PN = P["pn"]; IR = P["init_rate_std"]; IS = P["init_pos_std"]
    RP = P["rp"]; RR = P["rr"]; RESAMP = P["resamp"]; RATE_CLIP = P["rate_clip"]
    GR_POWER = P["gr_power"]; JUMP_PROB = P["jump_prob"]; JUMP_STD = P["jump_std"]; CLIP = P["tvt_clip_margin"]
    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    L = int(P.get("smooth_lag", 32)); L = max(0, min(L, T))
    def rn(): return torch.randn((B, N), generator=gen, device=device, dtype=dtype)
    pos = ls[:, None] + IS * rn(); rate = ir[:, None] + IR * rn()
    w = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    log_lik = torch.zeros(B, device=device, dtype=torch.float64)
    pts_f = torch.zeros((B, T), device=device, dtype=dtype)
    pts_s = torch.zeros((B, T), device=device, dtype=dtype)
    tvt_lo = vmin - CLIP; tvt_hi = vmin + (glen.to(dtype) - 1) * step + CLIP
    glast = torch.gather(grid, 1, (glen - 1).clamp_min(0).unsqueeze(1)); g0 = grid[:, 0:1]
    pm = md[:, 0] - 1.0; arN = torch.arange(N, device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype); dipp = ir[:, None]
    use_sm = L > 0
    if use_sm:
        buf = torch.full((B, L, N), float("nan"), device=device, dtype=dtype); ptr = 0
    for i in range(T):
        act = valid[:, i]; cur = md[:, i]; dm = (cur - pm).clamp_min(1.0)
        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)
        pos_n = pos + rate_n * dm[:, None] + PN * rn()
        if JUMP_PROB > 0 and JUMP_STD > 0:
            jm = torch.rand((B, N), generator=gen, device=device, dtype=dtype) < JUMP_PROB
            pos_n = pos_n + torch.where(jm, JUMP_STD * rn(), zero)
        zi = z[:, i][:, None]; tvt = (pos_n - zi).clamp(tvt_lo[:, None], tvt_hi[:, None]); pos_n = tvt + zi
        am = act[:, None]; pos = torch.where(am, pos_n, pos); rate = torch.where(am, rate_n, rate)
        gri = gr[:, i]; obs = act & ~torch.isnan(gri)
        v = pos - zi; ii = (v - vmin[:, None]) / step[:, None]; i0 = torch.floor(ii).long()
        below = i0 < 0; above = i0 >= (glen - 1)[:, None]
        i0c = i0.clamp_min(0); i0c = torch.minimum(i0c, (glen - 2).clamp_min(0)[:, None])
        t = ii - i0c.to(dtype); gA = torch.gather(grid, 1, i0c); gB = torch.gather(grid, 1, i0c + 1)
        eg = gA * (1 - t) + gB * t; eg = torch.where(below, g0, eg); eg = torch.where(above, glast, eg)
        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)
        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if USE_P:
            dphys = (rate - dipp) / PHYS_SIG; lkp = torch.exp(-0.5 * dphys * dphys).clamp_min(1e-300)
        else:
            lkp = torch.ones_like(w)
        # ★v41: struct錨(v38 row錨, mult=V41_ANCHOR_MULT)
        asig = (ancs[:, i][:, None] * float(P.get("anchor_mult", V41_ANCHOR_MULT))).clamp_min(1e-6)  # ★v69: per-bank錨mult
        da = (v - anc[:, i][:, None]) / asig
        lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        # ★v90 emission: 粒子TVT v を boot中心stまわりの学習emission c_learn で薄く再重み付け
        if _use_nn:
            delta = v - ns_ctr[:, i][:, None]
            jf = delta / NN_STEP + NN_NB
            oob = (jf < 0) | (jf > NN_ND - 1)
            j0 = torch.floor(jf).long().clamp(0, NN_ND - 2); fr = jf - j0.to(dtype)
            drow = ns_band[b_base, ns_bid[:, i]]                       # (B,ND) その行binの学習emission
            rmax = drow.max(dim=1, keepdim=True).values.clamp_min(1e-9)
            pA = torch.gather(drow, 1, j0); pB = torch.gather(drow, 1, (j0 + 1).clamp(0, NN_ND - 1))
            pv = (pA * (1 - fr) + pB * fr) / rmax
            pv = torch.where(oob, torch.zeros_like(pv), pv)
            lkn = ((1.0 - NN_FL) * pv + NN_FL) ** NN_W
            w_new = torch.where(obs[:, None], w * lk * lkp * lka * lkn, w * lkp * lka * lkn)
        else:
            w_new = torch.where(obs[:, None], w * lk * lkp * lka, w * lkp * lka)
        ws = w_new.sum(1, keepdim=True)
        w = torch.where(ws > 0, w_new / ws.clamp_min(1e-300), torch.full_like(w, 1.0 / N))
        ne = (w * w).sum(1); need = act & ((1.0 / ne) < (RESAMP * N))
        if bool(need.any()):
            cumw = torch.cumsum(w, 1); u0 = torch.rand((B, 1), generator=gen, device=device, dtype=dtype) * (1.0 / N)
            u = u0 + arN[None, :] / N; idx = torch.searchsorted(cumw, u, right=False).clamp(0, N - 1)
            pos_rs = torch.gather(pos, 1, idx) + RP * rn(); rate_rs = torch.gather(rate, 1, idx) + RR * rn()
            if RATE_CLIP > 0: rate_rs = rate_rs.clamp(-RATE_CLIP, RATE_CLIP)
            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)
            if use_sm:
                buf_g = torch.gather(buf, 2, idx[:, None, :].expand(B, L, N))
                buf = torch.where(need[:, None, None], buf_g, buf)
        pts_f[:, i] = (w * (pos - zi)).sum(1)
        if use_sm:
            if i >= L:
                old = buf[:, ptr, :]; pts_s[:, i - L] = (w * old).sum(1) - z[:, i - L]
            buf[:, ptr, :] = pos; ptr = (ptr + 1) % L
        pm = torch.where(act, cur, pm)
    if use_sm:
        for j in range(max(0, T - L), T):
            pts_s[:, j] = (w * buf[:, j % L, :]).sum(1) - z[:, j]
    else:
        pts_s = pts_f
    return pts_f, pts_s, log_lik

def _v26_pad(inps, device, dtype=V26_DTYPE):
    W = len(inps); Tmax = max(len(x["md"]) for x in inps); Gmax = max(len(x["gg"]) for x in inps)
    md = torch.zeros((W, Tmax), dtype=dtype); z = torch.zeros((W, Tmax), dtype=dtype)
    gr = torch.full((W, Tmax), float("nan"), dtype=dtype); valid = torch.zeros((W, Tmax), dtype=torch.bool)
    grid = torch.zeros((W, Gmax), dtype=dtype); glen = torch.zeros(W, dtype=torch.long)
    vmin = torch.zeros(W, dtype=dtype); step = torch.zeros(W, dtype=dtype); gs = torch.zeros(W, dtype=dtype)
    ls = torch.zeros(W, dtype=dtype); ir = torch.zeros(W, dtype=dtype)
    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)   # v41錨
    _has_nn = any("_simband" in x for x in inps)                       # ★v90 emission sim
    if _has_nn:
        nbmax = max((int(x["_simbid"].max()) + 1) for x in inps if "_simbid" in x)
        ns_ctr = torch.zeros((W, Tmax), dtype=dtype); ns_bid = torch.zeros((W, Tmax), dtype=torch.long)
        ns_band = torch.zeros((W, nbmax, NN_ND), dtype=dtype)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        if "anc" in x:
            anc[b, :Tn] = torch.from_numpy(x["anc"].astype("float32"))
            if Tn < Tmax: anc[b, Tn:] = anc[b, Tn - 1]
            ancs[b, :Tn] = torch.from_numpy(x["ancs"].astype("float32"))
        md[b, :Tn] = torch.from_numpy(x["md"].astype("float32"))
        if Tn < Tmax: md[b, Tn:] = md[b, Tn - 1]
        z[b, :Tn] = torch.from_numpy(x["z"].astype("float32"))
        gr[b, :Tn] = torch.from_numpy(x["gr"].astype("float32")); valid[b, :Tn] = True
        grid[b, :G] = torch.from_numpy(x["gg"].astype("float32"))
        if G < Gmax: grid[b, G:] = grid[b, G - 1]
        glen[b] = G; vmin[b] = x["gmin"]; step[b] = x["gst"]; gs[b] = x["gs"]; ls[b] = x["ls"]; ir[b] = x["ir"]
        if _has_nn and "_simband" in x:
            Ts = min(Tn, len(x["_simst"]), len(x["_simbid"]))
            st_ = torch.from_numpy(np.asarray(x["_simst"], np.float32))
            ns_ctr[b, :Ts] = st_[:Ts]
            if Ts < Tmax: ns_ctr[b, Ts:] = st_[Ts - 1] if Ts > 0 else 0.0
            bd = torch.from_numpy(np.asarray(x["_simbid"], np.int64).clip(0, nbmax - 1))
            ns_bid[b, :Ts] = bd[:Ts]
            bandb = np.asarray(x["_simband"], np.float32); nb_ = min(len(bandb), nbmax)
            ns_band[b, :nb_] = torch.from_numpy(bandb[:nb_])
    to = lambda tt: tt.to(device)
    out = dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
               vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs))
    if _has_nn:
        out["ns_ctr"] = to(ns_ctr); out["ns_bid"] = to(ns_bid); out["ns_band"] = to(ns_band)
    return out

def run_smoother_ext(inps, P, seed, n_seeds, chunk):
    """各 well: smoothed(seed尤度加重) と std(smoothedのseed間加重std)。forwardはCLAMP基準。"""
    N = int(P["n_particles"]); S = n_seeds; W = len(inps); out = [None] * W
    ls_scale = float(P["likelihood_scale"])
    for c0 in range(0, W, chunk):
        sub = inps[c0:c0 + chunk]; w = len(sub); pad = _v26_pad(sub, V26_DEVICE)
        rep = lambda tt: tt.repeat_interleave(S, 0); gen = torch.Generator(device=V26_DEVICE); gen.manual_seed(seed)
        _nnkw = {}
        if "ns_band" in pad:                                          # ★v90 emission sim を core へ
            _nnkw = dict(ns_ctr=rep(pad["ns_ctr"]), ns_bid=rep(pad["ns_bid"]),
                         ns_band=pad["ns_band"], b_base=(torch.arange(w * S, device=V26_DEVICE) // S))
        pf, ps, ll = _v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), P, N, V26_DEVICE, gen, **_nnkw)
        pf = pf.view(w, S, -1).double().cpu().numpy(); ps = ps.view(w, S, -1).double().cpu().numpy()
        ll = ll.view(w, S).cpu().numpy()
        for j in range(w):
            T = len(sub[j]["md"]); llw = ll[j]
            if np.isfinite(llw).any():
                mx = np.nanmax(llw[np.isfinite(llw)]); lk = np.where(np.isfinite(llw), llw - mx, -np.inf)
                ww = np.exp(lk / max(ls_scale, 1e-6)); s = ww.sum(); ww = ww / s if s > 1e-300 else np.full(S, 1.0 / S)
            else:
                ww = np.full(S, 1.0 / S)
            smean = (ww[:, None] * ps[j, :, :T]).sum(0); fmean = (ww[:, None] * pf[j, :, :T]).sum(0)
            sstd = np.sqrt(np.maximum((ww[:, None] * (ps[j, :, :T] - smean[None, :]) ** 2).sum(0), 0.))
            if CLAMP_FT is not None:
                bad = np.abs(smean - fmean) > CLAMP_FT
                if bad.any(): smean = np.where(bad, fmean, smean)
            out[c0 + j] = dict(mean=smean.astype(np.float32), std=sstd.astype(np.float32))
    if V26_DEVICE == "cuda": torch.cuda.synchronize()
    return out


# ==================== データ組み立て(v9 build_well + smooth PF 注入) ====================
# ==================== v28 追加: self/neighbor grid + Z傾斜 + 近傍(train)プール ====================
_SELF_ENS = {}; _NBR_ENS = {}; _ZGRAD = {}; _NBR_META = {}    # wid -> per-eval-row dict(mean,std) / meta
_FWD_ENS_C = {}; _SELF_ENS_C = {}; _NBR_ENS_C = {}           # ★v90: emission補正版(corrected)

def _bin_grid(t, g, gmin, step, G):
    """(TVT, GR) を grid の TVT ビンに median 集約(疑似typewell)。無データビン=NaN。"""
    m = np.isfinite(t) & np.isfinite(g); t, g = t[m], g[m]
    idx = np.round((t - gmin) / step).astype(int); ok = (idx >= 0) & (idx < G); idx, gg = idx[ok], g[ok]
    out = np.full(G, np.nan)
    if len(gg):
        s = pd.Series(gg).groupby(idx).median(); out[s.index.to_numpy()] = s.to_numpy()
    return out

def build_inputs_self(hw, tw_tvt, tw_gr, P):
    """smooth PF 入力(tw)に、自分の既知prefix GR(TVT_input) を疑似typewell化して gg に上書き。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    kn = hw[hw["TVT_input"].notna()]
    tw_fb = float(np.nanmean(tw_gr)) if np.isfinite(np.nanmean(tw_gr)) else 0.
    gr_full = hw["GR"].astype(float).interpolate(limit_direction="both").fillna(tw_fb).to_numpy(float)
    gr_sm = _smooth_radius_values(gr_full, tw_fb, P["hgr_smooth_r"])
    kpos = hw.index.get_indexer(kn.index)
    gh = _bin_grid(kn["TVT_input"].to_numpy(float), gr_sm[kpos].astype(float), base["gmin"], base["gst"], len(base["gg"]))
    gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = SELF_MIX_W * gh[have] + (1 - SELF_MIX_W) * gg[have]
    base = dict(base); base["gg"] = gg; return base

def build_inputs_nbr(hw, tw_tvt, tw_gr, P, refs):
    """smooth PF 入力(tw)に、近傍train坑井の GR(TVT) を疑似typewell化して gg に上書き。refs無し=tw(fallback)。"""
    base = build_smoother_inputs(hw, tw_tvt, tw_gr, P)
    if base is None: return None
    if refs:
        rt = np.concatenate([r[1] for r in refs])
        rg = np.concatenate([_smooth_radius_values(r[0], float(np.nanmean(r[0])), P["hgr_smooth_r"]).astype(float) for r in refs])
        gh = _bin_grid(rt, rg, base["gmin"], base["gst"], len(base["gg"]))
        gg = base["gg"].copy(); have = np.isfinite(gh); gg[have] = NBR_MIX_W * gh[have] + (1 - NBR_MIX_W) * gg[have]
        base = dict(base); base["gg"] = gg
    return base

def z_gradient_eval(hw, r=ZGRAD_R):
    """Z傾斜 dZ/dMD を移動平均(断層影響回避)し eval 行のみ返す。"""
    ev = hw["TVT_input"].isna().to_numpy()
    md = hw["MD"].to_numpy(float); z = hw["Z"].to_numpy(float)
    with np.errstate(all="ignore"):
        g = np.gradient(z, md)
    g = pd.Series(g).rolling(2 * int(r) + 1, center=True, min_periods=1).mean().to_numpy()
    return g[ev].astype(np.float32)

# --- 近傍(train)プール: XY重心 ---
_TRAIN_CENT = None
def _train_centroids():
    global _TRAIN_CENT
    if _TRAIN_CENT is None:
        tdir = DATA_DIR_V26 / "train"
        wids = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(tdir / "*__horizontal_well.csv"))})
        xy = []
        for w in wids:
            h = pd.read_csv(tdir / f"{w}__horizontal_well.csv", usecols=["X", "Y"]); xy.append((float(h["X"].mean()), float(h["Y"].mean())))
        _TRAIN_CENT = (wids, np.array(xy))
    return _TRAIN_CENT
_REFC = {}
def _ref_grtvt(wid):
    if wid not in _REFC:
        h = pd.read_csv(DATA_DIR_V26 / "train" / f"{wid}__horizontal_well.csv", usecols=["GR", "TVT"])
        _REFC[wid] = (h["GR"].to_numpy(float), h["TVT"].to_numpy(float))
    return _REFC[wid]
def _nearest_dist(cx, cy, exclude_wid=None):
    """最近傍 train 坑井までの実距離(上限なし)。フォールバック坑井でも実数(孤立度)を返す。"""
    wids, xy = _train_centroids(); d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    if exclude_wid is not None:
        keep = np.array([w != exclude_wid for w in wids]); d = d[keep]
    return float(d.min()) if len(d) else float("nan")
def neighbors_of(cx, cy, exclude_wid=None):
    wids, xy = _train_centroids(); d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy); out = []
    for j in np.argsort(d):
        w = wids[j]
        if w == exclude_wid: continue
        if d[j] > MAX_DIST: break
        out.append((w, float(d[j])))
        if len(out) >= K_MAX: break
    return out


def build_dataset_v28():
    base = DATA_DIR_V26 / SPLIT
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(str(base / "*__horizontal_well.csv"))})
    if N_WELLS is not None:
        wells = wells[:N_WELLS]
    P_SETS = []
    for k, lab in enumerate(PF_BANK):
        raw = dict(PF_LIB[lab]); raw["n_seeds"] = N_SEED
        p = _make_pf_params(raw, k); p["PF_USE_PHYS"] = bool(raw.get("use_phys", False))
        p["PF_PHYS_SIG"] = float(raw.get("phys_sig", 1.0)); p["label"] = lab; p["_raw"] = PF_LIB[lab]
        P_SETS.append(p)
    print(f"[v28] PF: " + ", ".join(f"{p['label']}->{p['name']}" for p in P_SETS))

    names = []; paths = []
    for w in wells:
        hp = base / f"{w}__horizontal_well.csv"; tp = base / f"{w}__typewell.csv"
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
        except Exception:
            continue
        if IS_TRAIN and "TVT" not in hw.columns:
            continue
        if build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), P_SETS[0]["_raw"]) is None:
            continue
        names.append(w); paths.append((hp, tp))
    print(f"[{SPLIT}] 使用坑井: {len(names)}  PF={PF_BANK}(smooth, tw/self/nbr各{len(P_SETS)}回=計{3*len(P_SETS)}回)  lag={SMOOTH_LAG}  n_seed={N_SEED}")
    def _praw(p): return {**p["_raw"], "smooth_lag": SMOOTH_LAG}   # per-param(smooth_lag付き)

    # ★v41: PFパス結果キャッシュ(build段で転んでもPF再実行なしで再開できる保険)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)   # 保存先を先に作る(cache save/最終保存 共通)
    _PF_CACHE = OUT_PARQUET.parent / "pfpass_cache_v69.npz"
    _PF_CACHED = _PF_CACHE.exists() and (SPLIT == "train")   # ★cacheはtrain生成専用。testは必ず錨PF実行
    if _PF_CACHED:
        _zc = np.load(_PF_CACHE, allow_pickle=False)
        for _k in _zc.files:
            _w, _ens, _nm, _fld = _k.split("|")
            _dst = {"F": _FWD_ENS, "S": _SELF_ENS, "N": _NBR_ENS,
                    "FC": _FWD_ENS_C, "SC": _SELF_ENS_C, "NC": _NBR_ENS_C}[_ens]
            _dst.setdefault((_w, _nm), {})[_fld] = _zc[_k]
        print(f"[v41] PFパスをキャッシュから復元: {_PF_CACHE.name}")
    # (1) tw smooth PF(v9 build_well 注入用)→ pf_ancc{suffix}
    if not _PF_CACHED:
        _FWD_ENS.clear()
        for p in P_SETS:
            Pk = _praw(p)
            print(f"[smoothPF-tw] {p['label']} ..."); t0 = time.perf_counter()
            inps = []
            for (hp, tp) in paths:
                hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
                _x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
                inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))
            outs = run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)
            for i, w in enumerate(names): _FWD_ENS[(w, p["name"])] = outs[i]
            if V26_DEVICE == "cuda": torch.cuda.empty_cache()
            print(f"      done {time.perf_counter()-t0:.1f}s")

    # (2) self smooth PF(自分prefix grid) × pf_1/2/3
    if not _PF_CACHED:
        _SELF_ENS.clear()
    for p in ([] if _PF_CACHED else P_SETS):
        Pk = _praw(p)
        print(f"[smoothPF-self] {p['label']} ..."); t0 = time.perf_counter()
        inps = []
        for (hp, tp) in paths:
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            _x = build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
            inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))
        outs = run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)
        for i, w in enumerate(names): _SELF_ENS[(w, p["name"])] = outs[i]     # dict(mean,std)
        if V26_DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # (3) neighbor smooth PF × pf_1/2/3(近傍探索はparam非依存で1回、PF実行のみparam別)
    print("[smoothPF-nbr] 近傍探索 ..."); n_nbr = 0; NREFS = {}
    for (hp, tp) in paths:
        wid = os.path.basename(str(hp)).split("__")[0]
        hw = pd.read_csv(hp)
        cx, cy = float(hw["X"].mean()), float(hw["Y"].mean())
        refw = neighbors_of(cx, cy, exclude_wid=(wid if SPLIT == "train" else None))
        use = len(refw) >= NEED_REFS
        NREFS[wid] = [_ref_grtvt(r) for r, _ in refw] if use else []
        _NBR_META[wid] = dict(has=int(use), nn=_nearest_dist(cx, cy, exclude_wid=(wid if SPLIT == "train" else None)), nref=len(refw))
        if use: n_nbr += 1
    print(f"      近傍あり {n_nbr}/{len(names)}(残りは通常実行=tw)")
    if not _PF_CACHED:
        _NBR_ENS.clear()
    for p in ([] if _PF_CACHED else P_SETS):
        Pk = _praw(p)
        print(f"[smoothPF-nbr] {p['label']} ..."); t0 = time.perf_counter()
        inps = []
        for (hp, tp) in paths:
            wid = os.path.basename(str(hp)).split("__")[0]
            hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
            _x = build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk, NREFS[wid])
            inps.append(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)))
        outs = run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)
        for i, w in enumerate(names): _NBR_ENS[(w, p["name"])] = outs[i]     # dict(mean,std)
        if V26_DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"      done {time.perf_counter()-t0:.1f}s")

    # ★v90: corrected runs — emission(per-bank W)を tw/self/nbr に適用(素PFと別に再実行)
    if not _PF_CACHED:
        _FWD_ENS_C.clear(); _SELF_ENS_C.clear(); _NBR_ENS_C.clear()
        for p in P_SETS:
            W = W_NN_BANK.get(p["label"], 0.0)
            Pk = dict(_praw(p)); Pk["nn_w"] = W; Pk["nn_floor"] = NN_FLOOR
            t0 = time.perf_counter()
            for (tag, dst, bfn) in (("tw", _FWD_ENS_C, "tw"), ("self", _SELF_ENS_C, "self"), ("nbr", _NBR_ENS_C, "nbr")):
                inps = []
                for (hp, tp) in paths:
                    wid = os.path.basename(str(hp)).split("__")[0]
                    hw = pd.read_csv(hp); tw = pd.read_csv(tp).sort_values("TVT")
                    tvt = tw["TVT"].to_numpy(float); grr = tw["GR"].to_numpy(float)
                    if bfn == "tw":     _x = build_smoother_inputs(hw, tvt, grr, Pk)
                    elif bfn == "self": _x = build_inputs_self(hw, tvt, grr, Pk)
                    else:               _x = build_inputs_nbr(hw, tvt, grr, Pk, NREFS[wid])
                    _x = _attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS))
                    inps.append(_attach_sim(_x, wid, W))
                outs = run_smoother_bank(inps, Pk, DEC_BANK.get(p["label"], 1), SEED_V26, N_SEED, WELL_CHUNK)
                for i, w in enumerate(names): dst[(w, p["name"])] = outs[i]
                if V26_DEVICE == "cuda": torch.cuda.empty_cache()
            print(f"[smoothPF-corr] {p['label']} W={W} done {time.perf_counter()-t0:.1f}s")

    # ★v41: PFパス結果を保存(train生成時のみ。testはcache無効=保存もしない)
    if (SPLIT == "train") and not _PF_CACHED:
        _pk = {}
        for _ens, _dic in (("F", _FWD_ENS), ("S", _SELF_ENS), ("N", _NBR_ENS),
                           ("FC", _FWD_ENS_C), ("SC", _SELF_ENS_C), ("NC", _NBR_ENS_C)):
            for (_w, _nm), _o in _dic.items():
                for _fld in ("mean", "std"):
                    _pk[f"{_w}|{_ens}|{_nm}|{_fld}"] = np.asarray(_o[_fld], np.float32)
        try:
            np.savez_compressed(_PF_CACHE, **_pk)   # 保存失敗でも build は続行(PF作業を失わない)
            print(f"[v41] PFパスをキャッシュ保存: {_PF_CACHE.name}")
        except Exception as _e:
            print(f"[v41] PFキャッシュ保存失敗(無視してbuild続行): {type(_e).__name__} {_e}")

    # (4) Z傾斜(移動平均)
    for (hp, tp) in paths:
        _ZGRAD[os.path.basename(str(hp)).split("__")[0]] = z_gradient_eval(pd.read_csv(hp))

    # (5) v9 build_well(tw smooth PF注入) + self/nbr(pf_1/2/3)/z_grad を列追加
    globals()["run_pf_ancc"] = _patched_run_pf_ancc
    from joblib import Parallel, delayed
    def one(w, hp, tp):
        _TLS.wid = w
        df = build_well(str(hp), str(tp), IS_TRAIN)
        if df is None or len(df) == 0: return None
        n = len(df); lk = df["last_known_tvt"].to_numpy(float)
        def col(a):
            out = np.full(n, np.nan, np.float32)
            if a is not None:
                mm = min(len(a), n); out[:mm] = np.asarray(a[:mm], np.float32)
            return out
        mt = _NBR_META.get(w, {})
        S_all = []; N_all = []; T_all = []; _TWC_all = []
        for p in P_SETS:
            suf = p["suffix"]; nm = p["name"]
            twk = df[f"pf_ancc{suf}"].to_numpy(float) if f"pf_ancc{suf}" in df.columns else lk.copy()
            sf = _SELF_ENS.get((w, nm)); nb = _NBR_ENS.get((w, nm))
            ps = col(sf["mean"] if sf else None); pss = col(sf["std"] if sf else None)
            pn = col(nb["mean"] if nb else None); pns = col(nb["std"] if nb else None)
            df[f"pf_self{suf}"] = ps; df[f"pf_self_std{suf}"] = pss; df[f"pf_self_delta{suf}"] = (ps - lk).astype(np.float32)
            df[f"pf_nbr{suf}"]  = pn; df[f"pf_nbr_std{suf}"]  = pns; df[f"pf_nbr_delta{suf}"]  = (pn - lk).astype(np.float32)
            df[f"pf_self_vs_tw{suf}"] = (ps - twk).astype(np.float32)   # ★最有力: baseline tw との不一致
            df[f"pf_nbr_vs_tw{suf}"]  = (pn - twk).astype(np.float32)
            df[f"pf_self_vs_nbr{suf}"] = (ps - pn).astype(np.float32)
            st = np.stack([twk, ps.astype(float), pn.astype(float)], axis=1)
            df[f"pf_agree_std{suf}"]   = np.nanstd(st, axis=1).astype(np.float32)
            df[f"pf_agree_range{suf}"] = (np.nanmax(st, axis=1) - np.nanmin(st, axis=1)).astype(np.float32)
            # ★v90 corrected(emission) + diff(=枝flip検出) を tw/self/nbr で
            sfc = _FWD_ENS_C.get((w, nm)); slc = _SELF_ENS_C.get((w, nm)); nbc = _NBR_ENS_C.get((w, nm))
            tw_c = col(sfc["mean"] if sfc else None); tw_cs = col(sfc["std"] if sfc else None)
            ps_c = col(slc["mean"] if slc else None); pn_c = col(nbc["mean"] if nbc else None)
            df[f"pf_tw_corr{suf}"] = tw_c; df[f"pf_tw_corr_std{suf}"] = tw_cs
            df[f"pf_tw_diff{suf}"] = (tw_c - twk).astype(np.float32)
            df[f"pf_tw_absdiff{suf}"] = np.abs(tw_c - twk).astype(np.float32)
            df[f"pf_self_corr{suf}"] = ps_c; df[f"pf_self_diff{suf}"] = (ps_c - ps).astype(np.float32)
            df[f"pf_nbr_corr{suf}"] = pn_c; df[f"pf_nbr_diff{suf}"] = (pn_c - pn).astype(np.float32)
            S_all.append(ps.astype(float)); N_all.append(pn.astype(float)); T_all.append(twk)
            _TWC_all.append(tw_c.astype(float))
        # pf_1/2/3 間の集約(平均 / param間ばらつき / 平均のtw不一致)
        SM = np.stack(S_all, 1); NM = np.stack(N_all, 1); TM = np.stack(T_all, 1)
        df["pf_self_mean"] = np.nanmean(SM, 1).astype(np.float32); df["pf_self_pstd"] = np.nanstd(SM, 1).astype(np.float32)
        df["pf_nbr_mean"]  = np.nanmean(NM, 1).astype(np.float32); df["pf_nbr_pstd"]  = np.nanstd(NM, 1).astype(np.float32)
        df["pf_self_mean_vs_tw"] = (np.nanmean(SM, 1) - np.nanmean(TM, 1)).astype(np.float32)
        df["pf_nbr_mean_vs_tw"]  = (np.nanmean(NM, 1) - np.nanmean(TM, 1)).astype(np.float32)
        # ★v90: corrected tw の集約 + SDF特徴群
        TCM = np.stack(_TWC_all, 1) if _TWC_all else TM
        tw_cmean = np.nanmean(TCM, 1); tw_rmean = np.nanmean(TM, 1)
        df["pf_tw_corr_mean"]  = tw_cmean.astype(np.float32)
        df["pf_tw_corr_pstd"]  = np.nanstd(TCM, 1).astype(np.float32)          # corrected間の一致度(小=確信)
        df["pf_tw_corr_vs_raw_mean"] = (tw_cmean - tw_rmean).astype(np.float32)  # emissionが平均で動かした量
        sdfp = col(_SDF_PRED.get(w))                                            # ★U-Net→SDF予測(TVT)
        df["sdf_pred"]        = sdfp
        df["sdf_delta"]       = (sdfp - lk).astype(np.float32)
        df["sdf_vs_tw_raw"]   = (sdfp - tw_rmean).astype(np.float32)           # SDF vs 素PF平均
        df["sdf_vs_tw_corr"]  = (sdfp - tw_cmean).astype(np.float32)           # SDF vs 補正PF平均
        df["sdf_vs_self_mean"]= (sdfp - np.nanmean(SM, 1)).astype(np.float32)
        df["sdf_vs_nbr_mean"] = (sdfp - np.nanmean(NM, 1)).astype(np.float32)
        _absd = np.abs(sdfp[:, None] - TCM)
        _absd_safe = np.where(np.isfinite(_absd), _absd, np.inf)
        _amin = _absd_safe.min(axis=1); _aarg = _absd_safe.argmin(axis=1)
        df["sdf_vs_pf_absmin"] = np.where(np.isfinite(_amin), _amin, np.nan).astype(np.float32)  # SDFに最も近い補正バンクとの距離
        df["sdf_agree_bank"]   = np.where(np.isfinite(_amin), _aarg, np.nan).astype(np.float32)  # どのバンクがSDFに一番近いか
        # Z傾斜 + 近傍メタ(近傍を実際に使ったか/最近傍距離/本数)
        df["z_grad"] = col(_ZGRAD.get(w))
        df["pf_nbr_has"]    = np.float32(mt.get("has", 0))
        df["pf_nbr_nndist"] = np.float32(mt.get("nn", np.nan))
        df["pf_nbr_nrefs"]  = np.float32(mt.get("nref", 0))
        return df
    print("[build] v9 build_well(smooth PF注入) + self/nbr/z_grad 列 ...")
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS_CPU, prefer="threads", verbose=5)(
        delayed(one)(names[i], paths[i][0], paths[i][1]) for i in range(len(names)))
    parts = [r for r in res if r is not None]
    print(f"      done {time.perf_counter()-t0:.1f}s  parts={len(parts)}")
    if not parts:
        raise RuntimeError("有効な坑井がありません。")
    return pd.concat(parts, ignore_index=True)


def main():
    print(f"[v28] device={V26_DEVICE} smooth_lag={SMOOTH_LAG} n_seed={N_SEED} PF=r0 self/nbr/z_grad追加 chunk={WELL_CHUNK}")
    df = build_dataset_v28()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    cols = list(df.columns)
    print(f"\\nsaved: {OUT_PARQUET}  rows={len(df)} cols={len(cols)}")
    for pre in ["pf_", "beam", "tvtF", "form", "dense", "sc"]:
        print(f"  {pre}* = {sum(c.startswith(pre) for c in cols)} 列")


if __name__ == "__main__":
    main()
