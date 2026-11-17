"""
v18 パーティクル生成(順方向PFのみ・選択器の素材)
==================================================

狙い: 「どのパーティクル(=PFの各出力軌跡)を選べばよいか」を後で LightGBM 等で学習する。
PF出力の oracle(各坑井で真値に最も近い候補を選択)が overall ~5.0 に達することを確認済み
=> 生成は十分良い。あとは「選択」。本スクリプトはその素材(候補軌跡)を事前生成して保存する。

方針(v10 を参考・逆方向なし):
  - v9 の GPU 順方向PF(pf_gpu_run.py)を再利用。
  - 各 paramセット(pf_1/2/3) × 各シード の「アンサンブルせず1本ずつの軌跡」を候補として保存。
    => 1坑井あたり候補数 = len(PARAM_SETS) * N_SEEDS 本。
  - 後段の特徴量/セレクタ用に、評価ゾーンの md,z,GR,真TVT, typewell も一緒に保存。

出力: artifacts_v18/particles/{wid}.npz  (+ meta.json)
  cand[n_cand, T]   : 候補軌跡(評価ゾーンの TVT)
  loglik[n_cand]    : 各候補の対数尤度(PF内部で算出)
  setid[n_cand]     : paramセット番号(0=pf_1,...)
  seedid[n_cand]    : シード番号
  md,z,gr,gr_raw[T] : 評価ゾーンの測定深度/Z/GR(PF処理後)/GR(生補間)
  true[T]           : 真TVT(train。テスト生成時は NaN)
  ev_index[T]       : 元 horizontal_well.csv の行インデックス(提出 id 用)
  tw_tvt,tw_gr      : typewell(GR当てはまり特徴用)

実行: rogii_claude/.venv/Scripts/python.exe v18/generate_particles_v18.py
※ 実行はユーザー。コードは生成のみ。
"""

import os, sys, glob, json, time
import numpy as np
import pandas as pd
import torch

# v9 の PF コアを再利用
_V9 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v9")
sys.path.append(_V9)
from pf_gpu_run import (build_inputs, make_p, PF_PARAM_SETS_RAW,
                        _pad_inputs, _pf_ancc_gpu_core, DTYPE)

# ============================== CONFIG ==============================
DATA_DIR    = r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction\train"
SPLIT       = "train"          # "train"(真TVTあり) / "test"(推論時)
N_WELLS     = None             # None=全坑井, または先頭N
PARAM_SETS  = ["pf_1", "pf_2", "pf_3"]   # 順方向のみ。候補数 = len(PARAM_SETS)*N_SEEDS
N_SEEDS     = 32
SEED        = 0
WELL_CHUNK  = 130              # 1GPUバッチに載せる坑井数(VRAMに応じて)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR     = r"C:\Users\kosaka256\Documents\rogii_claude\v18\artifacts_v18\particles"
# ===================================================================


def run_particles(inps, p, S, device, seed, well_chunk):
    """各坑井について、シードごとの軌跡 pts[S,T] と対数尤度 ll[S] を返す(アンサンブルしない)。"""
    N = int(p["ANCC_N"])
    PTS, LL = [], []
    for c in range(0, len(inps), well_chunk):
        chunk = inps[c:c + well_chunk]
        pad = _pad_inputs(chunk, device); W = len(chunk)
        rep = lambda tt: tt.repeat_interleave(S, dim=0)
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        pts, log_lik = _pf_ancc_gpu_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            p, N, device, gen)
        pts = pts.view(W, S, -1).float().cpu().numpy()
        ll = log_lik.view(W, S).cpu().numpy()
        for w in range(W):
            T = len(chunk[w]["md"])
            PTS.append(pts[w, :, :T].astype(np.float32))   # [S,T]
            LL.append(ll[w].astype(np.float64))            # [S]
    return PTS, LL


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEVICE} sets={PARAM_SETS} seeds={N_SEEDS} chunk={WELL_CHUNK} split={SPLIT}")
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    wells = sorted({os.path.basename(q).split("__")[0]
                    for q in glob.glob(DATA_DIR + "/*__horizontal_well.csv")})
    if N_WELLS is not None:
        wells = wells[:N_WELLS]

    ps_params = {ps: make_p(PF_PARAM_SETS_RAW[ps]) for ps in PARAM_SETS}

    # ---- 1パス: 各坑井の静的配列 + 全paramセットの PF入力を作る(坑井ごとに1回だけ読む) ----
    statics = {}
    inps_sets = {ps: [] for ps in PARAM_SETS}
    wids_order = []
    for wid in wells:
        hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
        tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        twv = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
        built = {ps: build_inputs(hw, twv, twg, ps_params[ps]) for ps in PARAM_SETS}
        if any(built[ps] is None for ps in PARAM_SETS):
            continue
        ev = built["pf_1"]["ev_index"]
        gr_raw = hw["GR"].astype(float).interpolate(limit_direction="both")
        gr_raw = gr_raw.fillna(float(np.nanmean(twg))).to_numpy(float)[ev]
        true = hw["TVT"].to_numpy(float)[ev] if "TVT" in hw.columns else np.full(ev.shape, np.nan)
        statics[wid] = dict(md=built["pf_1"]["md"], z=built["pf_1"]["z"], gr=built["pf_1"]["gr"],
                            gr_raw=gr_raw.astype(np.float32), true=true.astype(np.float32),
                            ev_index=ev.astype(np.int64),
                            tw_tvt=twv.astype(np.float32), tw_gr=twg.astype(np.float32))
        wids_order.append(wid)
        for ps in PARAM_SETS:
            inps_sets[ps].append(built[ps])
    print(f"wells with eval: {len(wids_order)}")

    # ---- 各paramセットで順方向PF(全シード)を実行し、候補を集約 ----
    cand_by_wid = {wid: [] for wid in wids_order}
    ll_by_wid   = {wid: [] for wid in wids_order}
    set_by_wid  = {wid: [] for wid in wids_order}
    for si, ps in enumerate(PARAM_SETS):
        t0 = time.perf_counter()
        PTS, LL = run_particles(inps_sets[ps], ps_params[ps], N_SEEDS, DEVICE, SEED, WELL_CHUNK)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        for j, wid in enumerate(wids_order):
            cand_by_wid[wid].append(PTS[j])                     # [S,T]
            ll_by_wid[wid].append(LL[j])                        # [S]
            set_by_wid[wid].append(np.full(N_SEEDS, si, np.int8))
        print(f"  [{ps}] {len(wids_order)} wells in {time.perf_counter()-t0:.1f}s")

    # ---- 保存 + oracle sanity ----
    seedid = np.tile(np.arange(N_SEEDS, dtype=np.int16), len(PARAM_SETS))
    err_oracle, err_ens = [], []
    for wid in wids_order:
        st = statics[wid]
        cand = np.concatenate(cand_by_wid[wid], axis=0).astype(np.float32)   # [n_cand, T]
        loglik = np.concatenate(ll_by_wid[wid]).astype(np.float64)
        setid = np.concatenate(set_by_wid[wid]).astype(np.int8)
        np.savez_compressed(os.path.join(OUT_DIR, f"{wid}.npz"),
                            cand=cand, loglik=loglik, setid=setid, seedid=seedid,
                            md=st["md"].astype(np.float32), z=st["z"].astype(np.float32),
                            gr=st["gr"].astype(np.float32), gr_raw=st["gr_raw"],
                            true=st["true"], ev_index=st["ev_index"],
                            tw_tvt=st["tw_tvt"], tw_gr=st["tw_gr"])
        # sanity: oracle(各坑井で真値に最も近い候補) と 尤度加重アンサンブル
        t = st["true"]; m = np.isfinite(t)
        if m.any():
            rmse_c = np.sqrt(np.mean((cand[:, m] - t[None, m]) ** 2, axis=1))
            err_oracle.append(cand[int(np.argmin(rmse_c)), m] - t[m])
            lk = loglik - np.nanmax(loglik)
            ww = np.exp(lk / 20.0); ww /= max(ww.sum(), 1e-300)
            ens = (ww[:, None] * cand).sum(0)
            err_ens.append(ens[m] - t[m])

    meta = dict(split=SPLIT, param_sets=PARAM_SETS, n_seeds=N_SEEDS, seed=SEED,
                n_cand=len(PARAM_SETS) * N_SEEDS, wids=wids_order, out_dir=OUT_DIR)
    with open(os.path.join(os.path.dirname(OUT_DIR), "meta.json"), "w") as f:
        json.dump(meta, f)

    if err_oracle:
        eo = np.concatenate(err_oracle); ee = np.concatenate(err_ens)
        print(f"saved {len(wids_order)} wells -> {OUT_DIR}")
        print(f"  ORACLE overall RMSE     = {np.sqrt(np.mean(eo**2)):.3f}  (これがセレクタの到達目標の下限)")
        print(f"  lik-ensemble overall RMSE = {np.sqrt(np.mean(ee**2)):.3f}  (素のPFアンサンブル=現状)")
        print(f"  -> セレクタ学習で ensemble から oracle に近づけるのが v18 の目標")


if __name__ == "__main__":
    main()
