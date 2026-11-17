# -*- coding: utf-8 -*-
# v65 warm-up — r0_seed32(実績LB 7.098系の regime)を初期trialとして enqueue する共通部品
#   - 探索空間は pf_engine_v65.py の suggest_params をソースからパース(ユーザーが範囲を編集しても追随)
#   - r0 の値を現行空間へスナップ(categorical=最近傍/float=クリップ)して enqueue
#   - 既定では「studyが空のときだけ」入れる(再開時の重複防止)。env FORCE_WARMUP=1 で既存studyにも投入
import os, re, ast
from pathlib import Path

# v63/create_v63.py PF_LIB["r0_seed32"] より(2026-07-17 転記。use_phys は engine 側で固定)
R0_PARAMS = {
    "n_particles": 400, "likelihood_scale": 10.536319667317654, "init_pos_std": 0.9331541486020234,
    "init_rate_std": 0.00047753793163756304, "mom": 0.9997911117425017, "vn": 0.0006297672017716754,
    "pn": 0.005014009140727263, "rp": 0.7627309890952053, "rr": 0.00045663430474523125,
    "resamp": 0.032049849143995945, "gr_sig_min": 11.865793233968713, "gr_sig_max": 44.98927964612316,
    "gr_sig_def": 119.25397464636981, "gr_sig_mult": 3.058318309943017,
    "tvt_clip_margin": 150.56111180234504, "rate_clip": 2.0, "gr_power": 1.7826896851395453,
    "hgr_smooth_r": 0, "tw_gr_smooth_r": 0, "jump_prob": 0.002, "jump_std": 5.0,
    "phys_sig": 0.013666550145707291,
    # v65 新ノブは中立値(=v38コア相当の挙動)
    "lk_floor": 1e-5, "gr_debias": 0, "rate_target": "zero", "grid_step": 0.2,
    "use_anchor": False, "anchor_mult": 20.0, "anchor_ramp_md": 0.0,
    # tw_source系(該当時のみ使用)
    "self_mix_w": 1.0, "nbr_mix_w": 1.0,
}


# v38_anchor_mult20 best(LB 7.881 実証済 = 錨系の出発点。2026-07-18転記)
ST20_PARAMS = {"n_particles": 600, "likelihood_scale": 29.056396339290437, "init_pos_std": 1.6168048868046325, "init_rate_std": 0.014412715870661067, "mom": 0.9995682504305502, "vn": 0.000873883338794689, "pn": 0.00606209538025288, "rp": 0.14025726060525087, "rr": 0.00398567931602027, "resamp": 0.18645022443117387, "gr_sig_min": 10.63834771185878, "gr_sig_max": 43.95858017409127, "gr_sig_def": 86.87561008087175, "gr_sig_mult": 2.7681905590609426, "tvt_clip_margin": 86.30698497112195, "rate_clip": 0.0, "gr_power": 1.85175446220882, "hgr_smooth_r": 2, "tw_gr_smooth_r": 0, "jump_prob": 0.002, "jump_std": 5.0, "phys_sig": 2.3072090423765363, "use_anchor": True, "anchor_mult": 20.0, "anchor_ramp_md": 0.0, "lk_floor": 1e-05, "gr_debias": 0, "rate_target": "zero", "grid_step": 0.2, "self_mix_w": 1.0, "nbr_mix_w": 1.0}


def parse_engine_space(engine_path=None):
    """pf_engine_v65.py の suggest_params から {name: ("cat",[..]) | ("f",lo,hi,log)} を抽出。"""
    p = Path(engine_path or (Path(__file__).parent / "pf_engine_v65.py"))
    src = p.read_text(encoding="utf-8")
    m = re.search(r"def suggest_params\(trial\):(.*?)\ndef ", src, re.S)
    body = m.group(1)
    space = {}
    for name, lst in re.findall(r'suggest_categorical\("([\w]+)",\s*(\[[^\]]*\])', body):
        space[name] = ("cat", ast.literal_eval(lst))
    for name, lo, hi, tail in re.findall(
            r'suggest_float\("([\w]+)",\s*([0-9eE\.\-+]+),\s*([0-9eE\.\-+]+)([^\)]*)\)', body):
        space[name] = ("f", float(lo), float(hi), "log=True" in tail)
    return space


def snap_to_space(params, space):
    """r0値を空間へスナップ。cat=数値なら最近傍/非数値は一致時のみ、float=クリップ。空間外キーは捨てる。"""
    q = {}
    for k, spec in space.items():
        if k not in params:
            continue
        v = params[k]
        if spec[0] == "cat":
            cats = spec[1]
            if v in cats:
                q[k] = v
            elif all(isinstance(c, (int, float)) for c in cats) and isinstance(v, (int, float)):
                q[k] = min(cats, key=lambda c: abs(float(c) - float(v)))
            # 非数値カテゴリで不一致 → enqueueしない(Optunaに任せる)
        else:
            lo, hi = spec[1], spec[2]
            q[k] = float(min(max(float(v), lo), hi))
    return q


def enqueue_r0(study, space, label="r0_seed32", extra=None):
    """studyが空(またはFORCE_WARMUP=1)なら r0 warm-up trial を積む。enqueue内容を返す。"""
    import optuna
    n_exist = len(study.trials)
    if n_exist > 0 and os.environ.get("FORCE_WARMUP", "0") != "1":
        return None
    q = snap_to_space({**R0_PARAMS, **(extra or {})}, space)
    study.enqueue_trial(q, skip_if_exists=True)
    snapped = {k: (q[k], R0_PARAMS.get(k)) for k in q if k in R0_PARAMS and q[k] != R0_PARAMS[k]}
    print(f"[warmup] {label} を初期trialとしてenqueue({len(q)}キー"
          + (f", スナップ調整: {snapped}" if snapped else "") + ")", flush=True)
    return q
