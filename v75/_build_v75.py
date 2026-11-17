# -*- coding: utf-8 -*-
"""v75 生成器 — create_candidates_v69_4.py に v73 部品(structgate + NN emission)を注入して
create_candidates_v75.py を生成 + train_tabular/train_nn を artifacts_v75 へパス差替コピー。
全パッチは完全一致置換(見つからなければ即エラー=サイレント崩れ防止)。設計は docs/v75.md。"""
import io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V69 = P0 / "v69"; V75 = P0 / "v75"; V75.mkdir(exist_ok=True)
src = (V69 / "create_candidates_v69_4.py").read_text(encoding="utf-8")
n0 = len(src)


def rep(s, old, new, cnt=1, tag=""):
    k = s.count(old)
    assert k == cnt, f"[{tag}] 期待{cnt}回のはずが{k}回: {old[:70]!r}"
    return s.replace(old, new)


# ---------- P1: パス/キャッシュ名 ----------
src = rep(src, 'OUT_PARQUET = PROJECT_V26 / "v69" / "artifacts_v69_4" / "data" / "train.parquet"',
          'OUT_PARQUET = PROJECT_V26 / "v75" / "artifacts_v75" / "data" / "train.parquet"', 1, "P1out")
src = rep(src, 'pfpass_cache_v69.npz', 'pfpass_cache_v75.npz', 1, "P1cache")

# ---------- P2: _attach_v75(+ロード) を _attach_anchor 定義の直後に追加 ----------
ANCHOR_DEF_END = '''        x["anc"] = np.zeros(n); x["ancs"] = np.full(n, 1e9)
    return x
'''
V75_BLOCK = ANCHOR_DEF_END + '''
# ==================== v75: v73部品(structgate + NN emission) ====================
V75_RTK = 0.02; V75_SGATE = 6.0; V75_WNN = 0.01     # v73確定値(docs/v73.md)。無調整=v69_4との単純比較
_V75_SP = _pickle.loads((_V38_ART / "struct_prior_train.pkl").read_bytes())          # v38 ev行整列 tvt/sig
_V75_SIM_FP = PROJECT_V26 / "v51" / "artifacts_v51" / "sim_rows_v51.pkl"
_V75_SIM = _pickle.loads(_V75_SIM_FP.read_bytes())                                    # {wid:{sim(f16,T,181), st}}
print(f"[v75] structgate(K={V75_RTK},SG={V75_SGATE}) + NN emission(W={V75_WNN})  SP={len(_V75_SP)}井 SIM={len(_V75_SIM)}井")

def _attach_v75(x, wid, use_sim):
    """smooth PF入力に v73部品を添付。rtq=struct dip目標 / rtk=σゲート付きK(prior無し坑井=0=引かない)
    / use_sim時 stq(帯中心)+simt(181bin, 帯外・欠損=-1)。"""
    if x is None: return None
    n = len(x["md"])
    sp = _V75_SP.get(wid)
    if sp is not None and len(sp["tvt"]) == n:
        sline = np.asarray(sp["tvt"], float) + np.asarray(x["z"], float)
        mu = pd.Series(np.gradient(sline, np.asarray(x["md"], float))).rolling(
            21, center=True, min_periods=1).median().to_numpy()
        x["rtq"] = np.clip(np.nan_to_num(mu), -2.0, 2.0)
        sig = np.clip(np.asarray(sp["sig"], float), 0.5, 60.0)
        x["rtk"] = (V75_RTK * np.exp(-0.5 * (sig / V75_SGATE) ** 2)).astype(float)
    else:
        x["rtq"] = np.zeros(n); x["rtk"] = np.zeros(n)
    if use_sim:
        sd = _V75_SIM.get(wid)
        if sd is not None:
            Ts = min(n, len(sd["st"]))
            st = np.zeros(n); st[:Ts] = np.asarray(sd["st"], float)[:Ts]
            if Ts and Ts < n: st[Ts:] = st[Ts - 1]
            simt = np.full((n, 181), -1.0, np.float16); simt[:Ts] = sd["sim"][:Ts]
            x["stq"] = st; x["simt"] = simt
    return x
'''
src = rep(src, ANCHOR_DEF_END, V75_BLOCK, 1, "P2attach")

# ---------- P3: _v26_pad 拡張 ----------
src = rep(src, '''    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)   # v41錨
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        if "anc" in x:''',
          '''    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)   # v41錨
    rtq = torch.zeros((W, Tmax), dtype=dtype); rtk = torch.zeros((W, Tmax), dtype=dtype)         # v75 structgate
    stq = torch.zeros((W, Tmax), dtype=dtype)
    simt = torch.full((W, Tmax, 181), -1.0, dtype=torch.float16)                                  # v75 NN emission
    has_sim = any("simt" in x for x in inps)
    for b, x in enumerate(inps):
        Tn = len(x["md"]); G = len(x["gg"])
        if "rtq" in x:
            rtq[b, :Tn] = torch.from_numpy(x["rtq"].astype("float32"))
            if Tn < Tmax: rtq[b, Tn:] = rtq[b, Tn - 1]
            rtk[b, :Tn] = torch.from_numpy(x["rtk"].astype("float32"))
        if "simt" in x:
            stq[b, :Tn] = torch.from_numpy(x["stq"].astype("float32"))
            if Tn < Tmax: stq[b, Tn:] = stq[b, Tn - 1]
            simt[b, :Tn] = torch.from_numpy(x["simt"])
        if "anc" in x:''', 1, "P3pad")
src = rep(src, '''    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs))''',
          '''    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                rtq=to(rtq), rtk=to(rtk), stq=to(stq), simt=(to(simt) if has_sim else None))''', 1, "P3ret")

# ---------- P4: _v26_smoother_core(署名 + 伝播引き戻し + sim乗算) ----------
src = rep(src, 'def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=V26_DTYPE):',
          'def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen,\n'
          '                       rtq=None, rtkq=None, stq=None, simt=None, wmap=None, dtype=V26_DTYPE):', 1, "P4sig")
src = rep(src, '''        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''',
          '''        rate_n = ALPHA * rate + RN * rn()
        if rtq is not None:                                   # ★v75 structgate: rate←…+K_t·(dip目標−rate)
            rate_n = rate_n + rtkq[:, i][:, None] * (rtq[:, i][:, None] - rate)
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''', 1, "P4rate")
src = rep(src, '''        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))''',
          '''        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if simt is not None:                                  # ★v75 NN emission(重み更新のみ。log_likは純GR)
            jj = torch.round((v - stq[:, i][:, None]) / 0.5).long() + 90
            inb = (jj >= 0) & (jj <= 180)
            sv = torch.gather(simt[:, i, :].to(dtype)[wmap], 1, jj.clamp(0, 180))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lk = lk * torch.exp(V75_WNN * sv)''', 1, "P4sim")

# ---------- P5: run_smoother_ext(受け渡し + wmap) ----------
src = rep(src, '''        pf, ps, ll = _v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), P, N, V26_DEVICE, gen)''',
          '''        wmap = torch.arange(w, device=V26_DEVICE).repeat_interleave(S)
        pf, ps, ll = _v26_smoother_core(
            rep(pad["md"]), rep(pad["z"]), rep(pad["gr"]), rep(pad["valid"]), rep(pad["grid"]),
            pad["glen"].repeat_interleave(S), pad["vmin"].repeat_interleave(S), pad["step"].repeat_interleave(S),
            pad["gs"].repeat_interleave(S), pad["ls"].repeat_interleave(S), pad["ir"].repeat_interleave(S),
            rep(pad["anc"]), rep(pad["ancs"]), P, N, V26_DEVICE, gen,
            rtq=rep(pad["rtq"]), rtkq=rep(pad["rtk"]), stq=rep(pad["stq"]),
            simt=pad["simt"], wmap=wmap)''', 1, "P5ext")

# ---------- P6: バンクループ3箇所(_attach_v75 ラップ。twのみ use_sim=True) ----------
src = rep(src, '''                _x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
                inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))''',
          '''                _x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
                _wid = os.path.basename(str(hp)).split("__")[0]
                inps.append(_attach_v75(_attach_anchor(_x, _wid, physics=(p["label"] in PHYSICS_BANKS)), _wid, use_sim=True))''', 1, "P6tw")
src = rep(src, '''            _x = build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
            inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))''',
          '''            _x = build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
            _wid = os.path.basename(str(hp)).split("__")[0]
            inps.append(_attach_v75(_attach_anchor(_x, _wid, physics=(p["label"] in PHYSICS_BANKS)), _wid, use_sim=False))''', 1, "P6self")
src = rep(src, '''            _x = build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk, NREFS[wid])
            inps.append(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)))''',
          '''            _x = build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk, NREFS[wid])
            inps.append(_attach_v75(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)), wid, use_sim=False))''', 1, "P6nbr")

# ---------- P7: DECガード + _decimate_inp の rtq/rtk 集約 ----------
src = rep(src, '''    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)''',
          '''    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)
    if "rtq" in x:  y["rtq"]  = _agg_dec_mean(x["rtq"], DEC); y["rtk"] = _agg_dec_mean(x["rtk"], DEC)
    assert "simt" not in x or DEC == 1, "v75: sim使用バンクはDEC=1のみ対応"''', 1, "P7dec")

(V75 / "create_candidates_v75.py").write_text(src, encoding="utf-8")
print(f"[create] create_candidates_v75.py 生成 ({n0}→{len(src)}字, 全パッチ完全一致OK)")

# ---------- train ノート(パス差替コピー) ----------
for s in ["train_tabular_v69_4.ipynb", "train_nn_v69_4.ipynb"]:
    nb = json.loads((V69 / s).read_text(encoding="utf-8")); n = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code": continue
        t = "".join(cell["source"])
        if "artifacts_v69_4" in t:
            cell["source"] = t.replace(r"v69\artifacts_v69_4", r"v75\artifacts_v75").replace(
                "v69/artifacts_v69_4", "v75/artifacts_v75").replace("artifacts_v69_4", "artifacts_v75").splitlines(keepends=True)
            n += 1
    d = s.replace("_v69_4", "_v75")
    (V75 / d).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[train] {d} (パス差替 {n}セル)")

# ---------- 検証 ----------
t = (V75 / "create_candidates_v75.py").read_text(encoding="utf-8")
checks = [
    ("OUT=v75", 'v75" / "artifacts_v75' in t or "artifacts_v75" in t),
    ("_attach_v75定義", "def _attach_v75" in t),
    ("structgate注入", "rtkq[:, i][:, None] * (rtq[:, i][:, None] - rate)" in t),
    ("sim乗算(重みのみ)", "lk = lk * torch.exp(V75_WNN * sv)" in t),
    ("log_likは純GR(sim乗算がavgの後)", t.index("avg = (w * lk).sum(1)") < t.index("lk = lk * torch.exp")),
    ("tw=use_sim True", "use_sim=True" in t),
    ("self/nbr=use_sim False", t.count("use_sim=False") == 2),
    ("cache名v75", "pfpass_cache_v75.npz" in t),
    ("残v69_4パス", "artifacts_v69_4" not in t),
]
for lbl, ok in checks:
    print(("  OK " if ok else "  ×  ") + lbl)
assert all(ok for _, ok in checks)
print("\n[build_v75] 完了。実行はユーザー: python v75/create_candidates_v75.py → train_tabular_v75 → train_nn_v75")
