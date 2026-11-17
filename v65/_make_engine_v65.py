# v38/iterative_pf_boost_v38.py → v65/pf_engine_v65.py を生成(1回実行用)
import sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)
s = open("v38/iterative_pf_boost_v38.py", encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:70])
    s = s.replace(old, new)

# ---- ヘッダ/CONFIG ----
rep('"""\nv38 構造事前分布つき 物理PF(v19コア + 絶対TVTソフト錨 log-prior)',
    '"""\nv65 PFエンジン — v38コアの拡張パラメータ版(Fable最終便)\n'
    '=====================================================\n'
    'v38 iterative_pf_boost の GPU物理PF を、錨(struct)を P パラメータ化し、\n'
    '誤差解剖の知見に対応する新ノブを追加したもの:\n'
    '  use_anchor / anchor_mult(錨 on/off・強度を trial 単位で探索可能)\n'
    '  anchor_ramp_md(錨の遅延起動: 錨近傍の引きずり対策=v41教訓)\n'
    '  lk_floor(尤度の一様混合床: 洗掘/スパイク/facies不整合で正解粒子が死ぬ問題のロバスト化)\n'
    '  gr_debias(prefix残差中央値でGRオフセット補正=井戸レベル工具バイアス)\n'
    '  rate_target(momentum減衰先 0/初期dip: dip過外挿vs平均回帰)\n'
    '  grid_step(typewell格子 0.2/0.35/0.5: v49「粗が良い」の本線導入)\n'
    'wellmean錨モードは削除(v32: 井戸定数シフトは非転移)。\n'
    '(以下は v38 原文)\n'
    'v38 構造事前分布つき 物理PF(v19コア + 絶対TVTソフト錨 log-prior)')
rep('OUT_DIR   = PROJECT / "v38" / "artifacts_v38"; OUT_DIR.mkdir(parents=True, exist_ok=True)',
    'OUT_DIR   = PROJECT / "v65" / "artifacts_v65"; OUT_DIR.mkdir(parents=True, exist_ok=True)\n'
    'V38_ART   = PROJECT / "v38" / "artifacts_v38"')
rep('PRIOR_NAME  = os.environ.get("V38_PRIOR", "hold")', 'PRIOR_NAME  = os.environ.get("V38_PRIOR", "struct")')
rep('STRUCT_PKL  = (OUT_DIR / "struct_prior_train.pkl" if PRIOR_NAME == "struct"\n'
    '               else OUT_DIR / "priors" / f"prior_{PRIOR_NAME}_train.pkl")',
    'STRUCT_PKL  = (V38_ART / "struct_prior_train.pkl" if PRIOR_NAME == "struct"\n'
    '               else V38_ART / "priors" / f"prior_{PRIOR_NAME}_train.pkl")')
rep('print(f"[v38] PRIOR={PRIOR_NAME}  MODE={ANCHOR_MODE}  MULT={ANCHOR_MULT:g}  BMULT={BIAS_MULT:g}  pkl={STRUCT_PKL.name}")',
    'print(f"[v65] PRIOR={PRIOR_NAME}  pkl={STRUCT_PKL.name}(錨のon/off/mult/rampはPパラメータ)")')

# ---- make_inputs: gr_debias + grid_step ----
rep('''def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf)
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))''',
    '''def make_inputs(c, P):
    tw_gr_pf = _smooth(c["tw_gr"], c["tw_fb"], P["tw_gr_smooth_r"]).astype(np.float64)
    gg, gmin, gst = _grid(c["tw_tvt"], tw_gr_pf, step=float(P.get("grid_step", 0.2)))
    gr_sm = _smooth(c["gr_filled"], c["tw_fb"], P["hgr_smooth_r"])
    bias = 0.0
    if len(c["kpos"]) < 20:
        gs = P["gr_sig_def"]
    else:
        resid = gr_sm[c["kpos"]] - np.interp(c["kn_tvtin"], c["tw_tvt"], tw_gr_pf)
        gs = float(np.nanstd(resid)); gs = P["gr_sig_def"] if (not np.isfinite(gs) or gs <= 0) else gs
        if int(P.get("gr_debias", 0)):
            b_ = float(np.nanmedian(resid))
            if np.isfinite(b_): bias = b_       # 井戸レベルGRオフセット(工具較正差)を尤度前に補正
    gs = float(np.clip(gs * P["gr_sig_mult"], P["gr_sig_min"], max(P["gr_sig_max"], P["gr_sig_min"] + 1e-6)))
    if bias != 0.0:
        gr_sm = gr_sm - bias''')

# ---- _pf_core: P駆動の錨/床/減衰先。wellmean削除 ----
rep('''    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]; AMUL = float(ANCHOR_MULT); USE_A = USE_ANCHOR
    WELLMEAN = (ANCHOR_MODE == "wellmean")''',
    '''    PHYS_SIG = P["phys_sig"]; USE_P = P["use_phys"]
    USE_A = bool(P.get("use_anchor", True)); AMUL = float(P.get("anchor_mult", 20.0))
    RAMP = float(P.get("anchor_ramp_md", 0.0))          # >0: 錨をmdsでランプイン(近錨域の引きずり防止)
    LKF = float(P.get("lk_floor", 0.0))                  # >0: 尤度の一様混合床(外れGRロバスト化)
    RT_IR = (str(P.get("rate_target", "zero")) == "ir")  # momentum減衰先: 0 or 初期dip''')
rep('''    if USE_A and WELLMEAN:
        # σ_b = 坑井のprior σの有効行平均 × BIAS_MULT(prior欠損坑井は σ_b 巨大=錨OFF)
        vf = valid.to(dtype); sig_ok = (ancs < 1e8).to(dtype) * vf
        sb = (ancs * sig_ok).sum(1) / sig_ok.sum(1).clamp_min(1.0)
        sb = torch.where(sig_ok.sum(1) > 0, sb, torch.full_like(sb, 1e9)) * float(BIAS_MULT)
        sb2 = (sb * sb).clamp_min(1e-6)[:, None]                     # (B,1)
        Dm = torch.zeros((B, N), device=device, dtype=dtype)         # 偏差ランニング平均(粒子ごと)
        cnt = torch.zeros((B, 1), device=device, dtype=dtype)
''', '')
rep('''        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''',
    '''        rate_n = ALPHA * rate + ((1.0 - ALPHA) * dipp if RT_IR else 0.0) + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''')
rep('''        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero).clamp_min(1e-300)''',
    '''        d = ((gri[:, None] - eg) / gs[:, None]).abs(); dp = d ** GR_POWER
        lk = torch.where(dp < 600.0, torch.exp(-0.5 * dp), zero)
        if LKF > 0:
            lk = (1.0 - LKF) * lk + LKF                  # 一様混合床: 外れGRで正解粒子が即死しない
        lk = lk.clamp_min(1e-300)''')
rep('''        if USE_A and WELLMEAN:                      # ★⑤ telescoping 井戸レベル錨
            dev = v - anc[:, i][:, None]
            Dn = torch.where(am, (Dm * cnt + dev) / (cnt + 1.0), Dm)
            expo = (-0.5 * (Dn * Dn - Dm * Dm) / sb2).clamp(-60.0, 60.0)
            lka = torch.where(am, torch.exp(expo), torch.ones_like(w))   # 比 N(D_t)/N(D_{t-1})
            Dm = Dn; cnt = cnt + act.to(dtype)[:, None]
        elif USE_A:                                 # ★ row錨(毎行独立掛け, mult=温度補正)
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig    # v = pos - zi = 粒子のTVT
            lka = torch.where(da * da < 1200.0, torch.exp(-0.5 * da * da), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)''',
    '''        if USE_A:                                   # ★ row錨(毎行独立掛け, mult=温度補正, ramp=遅延起動)
            asig = (ancs[:, i][:, None] * AMUL).clamp_min(1e-6)
            da = (v - anc[:, i][:, None]) / asig    # v = pos - zi = 粒子のTVT
            da2 = da * da
            if RAMP > 0:
                tf = ((cur - md[:, 0]).clamp_min(0.0) / RAMP).clamp(0.0, 1.0)[:, None]
                da2 = da2 * tf                       # 温度でランプイン(近錨域は錨を弱く)
            lka = torch.where(da2 < 1200.0, torch.exp(-0.5 * da2), zero).clamp_min(1e-300)
        else:
            lka = torch.ones_like(w)''')
rep('''            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            if USE_A and WELLMEAN:
                Dm = torch.where(nm, torch.gather(Dm, 1, idx), Dm)   # ★Dは粒子の履歴状態→一緒にgather
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)''',
    '''            nm = need[:, None]; pos = torch.where(nm, pos_rs, pos); rate = torch.where(nm, rate_rs, rate)
            w = torch.where(nm, torch.full_like(w, 1.0 / N), w)''')

# ---- suggest/full ----
rep('''def suggest_params(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 500, 600, 800]),''',
    '''def suggest_params(trial):
    return dict(
        n_particles      = trial.suggest_categorical("n_particles", [400, 600, 800, 1200, 1600]),''')
rep('''        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0]),''',
    '''        tw_gr_smooth_r   = trial.suggest_categorical("tw_gr_smooth_r", [0, 1, 2]),''')
rep('''        phys_sig         = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        use_phys         = USE_PHYS,
    )

def full_params(best_params):
    return {**best_params, "use_phys": USE_PHYS}''',
    '''        phys_sig         = trial.suggest_float("phys_sig", 0.003, 5.0, log=True),
        use_phys         = USE_PHYS,
        # ---- v65 拡張ノブ ----
        use_anchor       = trial.suggest_categorical("use_anchor", [True, False]),
        anchor_mult      = trial.suggest_float("anchor_mult", 3.0, 80.0, log=True),
        anchor_ramp_md   = trial.suggest_categorical("anchor_ramp_md", [0.0, 300.0, 600.0, 1200.0, 2400.0]),
        lk_floor         = trial.suggest_float("lk_floor", 1e-5, 0.05, log=True),
        gr_debias        = trial.suggest_categorical("gr_debias", [0, 1]),
        rate_target      = trial.suggest_categorical("rate_target", ["zero", "ir"]),
        grid_step        = trial.suggest_categorical("grid_step", [0.2, 0.35, 0.5]),
    )

_V65_DEFAULTS = dict(use_phys=False, use_anchor=True, anchor_mult=20.0, anchor_ramp_md=0.0,
                     lk_floor=0.0, gr_debias=0, rate_target="zero", grid_step=0.2)

def full_params(best_params):
    return {**_V65_DEFAULTS, **best_params, "use_phys": USE_PHYS}''')

open("v65/pf_engine_v65.py", "w", encoding="utf-8").write(s)
import ast
ast.parse(s)
print("written v65/pf_engine_v65.py, syntax OK")
