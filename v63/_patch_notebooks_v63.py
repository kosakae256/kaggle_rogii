# v47 の学習/提出ノート4本 → v63 版を生成(1回実行用)
import json, sys
sys.stdout = open(1, "w", encoding="utf-8", closefd=False)

def load(nb): return json.load(open(nb, encoding="utf-8"))
def save(d, nb):
    json.dump(d, open(nb, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written:", nb)
def cell_src(c): return "".join(c["source"])
def set_src(c, s): c["source"] = s.splitlines(keepends=True)

def rep_in_nb(d, old, new, expect=None, allow_zero=False):
    n = 0
    for c in d["cells"]:
        s = cell_src(c)
        if old in s:
            n += s.count(old); set_src(c, s.replace(old, new))
    assert allow_zero or n > 0, f"not found: {old[:70]!r}"
    if expect is not None: assert n == expect, (n, expect, old[:70])
    return n

def rep_line_prefix(d, prefix, newline, expect=1):
    """prefix で始まる行を丸ごと newline に置換(コメント差異に頑健)。"""
    n = 0
    for c in d["cells"]:
        lines = cell_src(c).split("\n")
        out = []
        for ln in lines:
            if ln.startswith(prefix):
                out.append(newline); n += 1
            else:
                out.append(ln)
        set_src(c, "\n".join(out))
    assert n == expect, (n, expect, prefix)

# ================= 1) train_tabular_v63 =================
d = load("v47/train_tabular_v47 copy.ipynb")
rep_in_nb(d, 'ART = PROJECT / "v47" / "artifacts_v47"', 'ART = PROJECT / "v63" / "artifacts_v63"')
rep_in_nb(d, 'META = {"well", "id", "target", "is_backward"}',
          'META = {"well", "id", "target", "is_backward", "donor", "is_pseudo"}')
rep_in_nb(d, '_GF = np.load(_GFP)[:N_FWD]',
          '_GF = np.load(_GFP)\n    assert len(_GF) >= N_FWD, ("gf_feature.npy と v63 parquet の行数不一致", len(_GF), N_FWD)\n    _GF = _GF[:N_FWD]')
# 先頭コメント
rep_in_nb(d, "# ===== v36: 順方向+逆方向DA ロード(pyarrow列チャンク→事前確保float32。RAM節約版) =====",
          "# ===== v63 tabular(=v47と同一構成, parquetのみ差替: 6バンク普通PF) =====\n"
          "# ===== v36: 順方向+逆方向DA ロード(pyarrow列チャンク→事前確保float32。RAM節約版) =====")
save(d, "v63/train_tabular_v63.ipynb")

# ================= 2) train_nn_v63 =================
d = load("v47/train_nn_v47 copy.ipynb")
rep_in_nb(d, 'ART     = PROJECT / "v47" / "artifacts_v47"', 'ART     = PROJECT / "v63" / "artifacts_v63"')

# --- cell1 末尾: pseudo 設定 + RUN サフィックス ---
c1 = d["cells"][1]; s = cell_src(c1)
assert 'print(f"RUN={RUN}' in s
s += '''

# ===== v63 pseudo(合成井戸)設定 =====
#   PSEUDO_MODE: off=ベースライン / pretrain=pseudoで事前学習→realでfinetune / da=realにpseudo井戸を混ぜる
PSEUDO_MODE = os.environ.get("PSEUDO_MODE", "off")     # off | pretrain | da
PSEUDO_TAG  = os.environ.get("PSEUDO_TAG", "p1")
PSEUDO_PQ   = ART / "data" / f"pseudo_{PSEUDO_TAG}.parquet"
PSEUDO_DIR  = PROJECT / "v63" / "artifacts_v63" / f"pseudo_wells_{PSEUDO_TAG}"
PRE_EPOCHS  = int(os.environ.get("PRE_EPOCHS", "20"))  # pretrain: pseudoエポック数(flat LR)
DA_RATIO    = float(os.environ.get("DA_RATIO", "0.5")) # da: 1エポックに混ぜるpseudo井戸数の対train比
if PSEUDO_MODE != "off":
    RUN = f"{RUN}_{PSEUDO_MODE}-{PSEUDO_TAG}"          # ckpt/結果キーを分離(ベースラインを上書きしない)
    print(f"PSEUDO: mode={PSEUDO_MODE} tag={PSEUDO_TAG} pq={PSEUDO_PQ.name} pre_ep={PRE_EPOCHS} da_ratio={DA_RATIO}")
'''
set_src(c1, s)

# --- cell3 末尾: pseudo データ読込(realのmed/scl/残差化/正規化を適用) ---
c3 = d["cells"][3]; s = cell_src(c3)
s += '''

# ===== v63 pseudo(合成井戸)読込: real統計で正規化し X/yn/well_rows に追記 =====
PS_WELLS = []; PS_FOLD = {}
if PSEUDO_MODE != "off":
    assert PSEUDO_PQ.exists(), f"pseudo parquet がない: {PSEUDO_PQ}(create_v63.py を PSEUDO_DIR 付きで実行)"
    _man = json.loads((PSEUDO_DIR / "manifest.json").read_text(encoding="utf-8"))
    _donor_of = {e["id"]: e["donor"] for e in _man["wells"]}
    _pqn = set(pq.read_schema(PSEUDO_PQ).names)
    _needp = list(dict.fromkeys(["well", "target", "last_known_tvt", "md_since"] + [c for c in feats_nn if c in _pqn]))
    _tblp = pq.read_table(PSEUDO_PQ, columns=_needp)
    Np = _tblp.num_rows
    gp   = _tblp.column("well").to_pandas().to_numpy()
    yp   = _tblp.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    lkp  = _tblp.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    mdsp = _tblp.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64)
    Xp = np.empty((Np, len(feats_nn)), np.float32)
    for j, c in enumerate(feats_nn):
        if c in _pqn:
            Xp[:, j] = _tblp.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        else:
            Xp[:, j] = np.nan                          # gf等のparquet外列 → 正規化後0(中央値相当)
    del _tblp; gc.collect()
    for j in range(Xp.shape[1]):                       # realと同じ残差化判定を適用
        if abs2res[j]: Xp[:, j] = Xp[:, j] - lkp.astype(np.float32)
    Xp = (Xp - med) / scl
    Xp = np.clip(np.nan_to_num(Xp, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0).astype(np.float32)
    ynp = ((yp - t_mean) / t_scale).astype(np.float32)
    _N0 = len(yn)
    X = np.concatenate([X, Xp]); yn = np.concatenate([yn, ynp])
    _fold_of_well = {w: k_ for k_, (_tw_, _vw_) in enumerate(fold_wells) for w in _vw_}
    for w in pd.unique(gp):
        idx = np.where(gp == w)[0]
        well_rows[w] = (idx[np.argsort(mdsp[idx], kind="stable")] + _N0)
        PS_WELLS.append(w)
        PS_FOLD[w] = _fold_of_well.get(_donor_of.get(w, ""), -1)   # -1=donorがfold外→全foldで使用可
    # prefix系列(既知ゾーン)を pseudo dir から抽出(real統計 gmed/giqr で正規化)
    if CFG["prefix"]:
        _stats_fp = ART / "gr_stats_nn.json"
        if _stats_fp.exists():
            _st = json.loads(_stats_fp.read_text()); gmed_, giqr_ = _st["gmed"], _st["giqr"]
        else:
            _grs = []
            for w in wid_order:
                f = DATA / "train" / f"{w}__horizontal_well.csv"
                if not f.exists(): continue
                _df = pd.read_csv(f, usecols=["GR", "TVT_input"]).dropna()
                if len(_df) >= 8: _grs.append(_df["GR"].to_numpy(np.float64)[-PRE_LEN:])
            _a = np.concatenate(_grs)
            gmed_ = float(np.nanmedian(_a))
            giqr_ = float(max(np.nanpercentile(_a, 75) - np.nanpercentile(_a, 25), 1e-6))
            _stats_fp.write_text(json.dumps(dict(gmed=gmed_, giqr=giqr_)))
        for w in PS_WELLS:
            f = PSEUDO_DIR / f"{w}__horizontal_well.csv"
            if not f.exists(): continue
            _df = pd.read_csv(f, usecols=["MD", "Z", "GR", "TVT_input"])
            _kn = _df.dropna(subset=["MD", "Z", "GR", "TVT_input"]).tail(PRE_LEN)
            if len(_kn) < 8: continue
            _a = _kn.to_numpy(np.float64)
            PREF[w] = np.nan_to_num(np.stack([
                np.clip((_a[:, 2] - gmed_) / giqr_, -8, 8),
                (_a[:, 3] - _a[-1, 3]) / t_scale,
                (_a[:, 1] - _a[-1, 1]) / 50.0,
                (_a[:, 0] - _a[-1, 0]) / 1000.0], axis=1), nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)
    print(f"pseudo: wells={len(PS_WELLS)} rows={Np} fold内訳:",
          {k_: sum(1 for w in PS_WELLS if PS_FOLD[w] == k_) for k_ in range(-1, N_SPLITS)})
'''
set_src(c3, s)

# --- cell5: train_fold に pretrain/da フック ---
c5 = d["cells"][5]; s = cell_src(c5)
old = "    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)\n    trw = list(trw); vaw = list(vaw)\n    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)\n    steps_ep = math.ceil(len(trw) / BATCH_WELLS)"
new = """    torch.manual_seed(1000 * seed + k); rng = np.random.default_rng(1000 * seed + k)
    trw = list(trw); vaw = list(vaw)
    psw = [w for w in PS_WELLS if PS_FOLD.get(w, -1) != k] if PSEUDO_MODE != "off" else []
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n_ep_wells = len(trw) + (min(len(psw), int(len(trw) * DA_RATIO)) if PSEUDO_MODE == "da" else 0)
    steps_ep = math.ceil(n_ep_wells / BATCH_WELLS)"""
assert old in s; s = s.replace(old, new)

old = '    ema = EMA(model, EMA_DECAY); best = (1e18, None); step = 0\n    va_idx = np.concatenate([well_rows[w] for w in vaw])'
new = '''    ema = EMA(model, EMA_DECAY); best = (1e18, None); step = 0
    va_idx = np.concatenate([well_rows[w] for w in vaw])
    if PSEUDO_MODE == "pretrain" and psw:                      # ===== pseudo事前学習(flat LR) =====
        pre_opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        for _pep in range(PRE_EPOCHS):
            model.train(); _po = rng.permutation(len(psw))
            for _pi in range(0, len(_po), BATCH_WELLS):
                x, yb, mb, _, px, pm = make_batch([psw[_pj] for _pj in _po[_pi:_pi + BATCH_WELLS]], True, rng)
                if NOISE: x = x + NOISE * torch.randn_like(x)
                with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
                    out, mo = model(x, mb.unsqueeze(1), px, pm)
                loss, _ = head_loss(out.float(), yb, mb, head)
                pre_opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                pre_opt.step()
        ema = EMA(model, EMA_DECAY)                            # 事前学習済み重みでEMA再初期化'''
assert old in s; s = s.replace(old, new)

old = """        model.train(); order = rng.permutation(len(trw))
        for i in range(0, len(order), BATCH_WELLS):
            x, yb, mb, _, px, pm = make_batch([trw[j] for j in order[i:i + BATCH_WELLS]], True, rng)"""
new = """        model.train()
        ep_wells = list(trw)
        if PSEUDO_MODE == "da" and psw:                        # ===== DA: pseudo井戸をエポック毎に混入 =====
            _nps = min(len(psw), int(len(trw) * DA_RATIO))
            ep_wells += [psw[_j] for _j in rng.choice(len(psw), size=_nps, replace=False)]
        order = rng.permutation(len(ep_wells))
        for i in range(0, len(order), BATCH_WELLS):
            x, yb, mb, _, px, pm = make_batch([ep_wells[j] for j in order[i:i + BATCH_WELLS]], True, rng)"""
assert old in s; s = s.replace(old, new)
set_src(c5, s)
save(d, "v63/train_nn_v63.ipynb")

# ================= 3) submit notebooks =================
for src, dst in [("v47/submit_tabular_v47.ipynb", "v63/submit_tabular_v63.ipynb"),
                 ("v47/submit_nn_v47.ipynb", "v63/submit_nn_v63.ipynb")]:
    d = load(src)
    rep_line_prefix(d, "SMOOTH_LAG  = 16",
                    "SMOOTH_LAG  = 0             # ★v63: smooth無し=純フォワードPF(普通のPF)。学習側 create_v63 と一致させる")
    rep_line_prefix(d, 'PF_BANK = ["pf_1","pf_2","pf_3","r0_seed32","r1_seed32","r4_seed32","pfA"]',
                    'PF_BANK = ["pf_1","pf_2","pf_3","r0_seed32","r1_seed32","pfA"]   # ★v63: 6バンク(r4除外)。学習側と一致')
    n1 = rep_in_nb(d, "rogii-v47-artifacts", "rogii-v63-artifacts")
    n2 = rep_in_nb(d, "artifacts_v47", "artifacts_v63", allow_zero=True)
    n3 = rep_in_nb(d, '"v47"', '"v63"', allow_zero=True)
    n4 = rep_in_nb(d, "\\\\v47\\\\", "\\\\v63\\\\", allow_zero=True)
    n5 = rep_in_nb(d, "\\v47\\", "\\v63\\", allow_zero=True)
    print(f"  {dst}: dataset={n1} artifacts={n2} quoted={n3} bs2={n4} bs1={n5}")
    save(d, dst)
print("done")
