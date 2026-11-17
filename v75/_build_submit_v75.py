# -*- coding: utf-8 -*-
"""v75 submit 生成器 — submit_nn_v69_4.ipynb に v73部品を注入して submit_nn_v75.ipynb を生成。
- コアパッチ: create と同一(structgate + NN emission)。
- test側 struct: _V41_STRUCT(v46_build_test_struct が実行時に埋める dict)を structgate に流用。
- test側 sim: ★v50エンコーダ(stageA_enccapaug_f0..4.pt)を test坑井GRに前向き実行し、
  5モデルの sim を平均(ユーザー指定)。帯= test struct ±45ft(0.5ft, 181bin)。
検証: 全パッチ完全一致 assert。実行はユーザー(Kaggle)。設計 docs/v75.md。"""
import io, sys, json
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path

P0 = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
V69 = P0 / "v69"; V75 = P0 / "v75"
nb = json.loads((V69 / "submit_nn_v69_4.ipynb").read_text(encoding="utf-8"))


def rep(s, old, new, cnt=1, tag=""):
    k = s.count(old)
    assert k == cnt, f"[{tag}] 期待{cnt}回が{k}回: {old[:70]!r}"
    return s.replace(old, new)


c0 = "".join(nb["cells"][0]["source"])

# ---------- コアパッチ(create _build_v75.py と同一物) ----------
ANCHOR_DEF_END = '''        x["anc"] = np.zeros(n); x["ancs"] = np.full(n, 1e9)
    return x
'''
V75_BLOCK = ANCHOR_DEF_END + '''
# ==================== v75: v73部品(structgate + NN emission) ====================
V75_RTK = 0.02; V75_SGATE = 6.0; V75_WNN = 0.01     # v73確定値(docs/v73.md)
_V75_SIM = {}                                        # test側: _v75_build_test_sims が実行時に埋める

def _attach_v75(x, wid, use_sim):
    """v73部品を添付。struct= _V41_STRUCT(testはv46_build_test_structが埋める)。prior欠損/不一致はK=0。"""
    if x is None: return None
    n = len(x["md"])
    sp = _V41_STRUCT.get(wid)
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

# ---- v75: test側 sim 構築(v50 stageA capaug エンコーダ, 5fold平均=ユーザー指定) ----
def _v75_rz(v):
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip((v - med) / max(iqr / 1.349, 1e-6), -8, 8)
def _v75_rm(v, w): return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()
def _v75_rs(v, w): return pd.Series(v).rolling(w, center=True, min_periods=2).std().fillna(0).to_numpy()
def _v75_detr(v, w=301):
    m = _v75_rm(v, w); s = _v75_rs(v, w)
    return np.clip((v - m) / np.maximum(s, 1e-3), -6, 6)
def _v75_feats_h(gr, md, z):
    g = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    g = np.where(np.isfinite(g), g, np.nanmedian(g)); g = np.where(np.isfinite(g), g, 0.0)
    z0 = _v75_rz(g); s5, s21, s81 = _v75_rm(z0, 5), _v75_rm(z0, 21), _v75_rm(z0, 81)
    d5 = np.gradient(s5) * 10.0; v21 = _v75_rs(z0, 21); dt = _v75_detr(z0, 301)
    dmd = np.maximum(np.gradient(md), 1e-3); dzdm = np.clip(np.gradient(z) / dmd, -0.2, 0.2) * 5.0
    X = np.stack([z0, s5, s21, s81, d5, v21, dt, dzdm], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
def _v75_feats_t(gg):
    z0 = _v75_rz(gg); s3, s9, s33 = _v75_rm(z0, 3), _v75_rm(z0, 9), _v75_rm(z0, 33)
    d3 = np.gradient(s3) * 10.0; v9 = _v75_rs(z0, 9); dt = _v75_detr(z0, 301)
    X = np.stack([z0, s3, s9, s33, d3, v9, dt], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

class _V75Enc1D(torch.nn.Module):
    def __init__(self, cin, ch=128, dout=96, dils=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.inp = torch.nn.Conv1d(cin, ch, 5, padding=2)
        self.blocks = torch.nn.ModuleList()
        for d in dils:
            self.blocks.append(torch.nn.Sequential(
                torch.nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d), torch.nn.GELU(),
                torch.nn.Conv1d(ch, ch, 1)))
        self.norm = torch.nn.ModuleList([torch.nn.GroupNorm(8, ch) for _ in dils])
        self.out = torch.nn.Conv1d(ch, dout, 1)
    def forward(self, x):
        h = self.inp(x.t()[None])
        for blk, nm in zip(self.blocks, self.norm):
            h = nm(h + blk(h))
        return torch.nn.functional.normalize(self.out(h)[0].t(), dim=1)

def _v75_enc_dir():
    import glob as _g
    for pat in [str(PROJECT_V26 / "v50" / "artifacts_v50"), "/kaggle/input/*/artifacts_v50", "/kaggle/input/*"]:
        for d in _g.glob(pat):
            if (Path(d) / "stageA_enccapaug_f0.pt").exists():
                return Path(d)
    raise FileNotFoundError("stageA_enccapaug_f*.pt が見つからない(Kaggleはv50エンコーダckptを添付すること)")

def _v75_build_test_sims(test_dir):
    """test各坑井: 5foldエンコーダの sim を平均して _V75_SIM に格納(帯= test struct ±45ft)。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ed = _v75_enc_dir()
    encs = []
    for f in range(5):
        ck = torch.load(ed / f"stageA_enccapaug_f{f}.pt", map_location=dev)
        eA = _V75Enc1D(8).to(dev); eA.load_state_dict(ck["encA"]); eA.eval()
        eB = _V75Enc1D(7).to(dev); eB.load_state_dict(ck["encB"]); eB.eval()
        encs.append((eA, eB))
    offs = np.arange(-90, 91)
    import glob as _g
    n_ok = 0
    with torch.no_grad():
        for fp in sorted(_g.glob(str(Path(test_dir) / "*__horizontal_well.csv"))):
            wid = os.path.basename(fp).split("__")[0]
            sp = _V41_STRUCT.get(wid)
            if sp is None: continue
            hw = pd.read_csv(fp)
            twp = Path(test_dir) / f"{wid}__typewell.csv"
            if not twp.exists(): continue
            tw = pd.read_csv(twp).sort_values("TVT")
            tw_tvt = tw["TVT"].to_numpy(float); tw_gr = tw["GR"].to_numpy(float)
            m = np.isfinite(tw_tvt) & np.isfinite(tw_gr); tw_tvt, tw_gr = tw_tvt[m], tw_gr[m]
            if len(tw_tvt) < 8: continue
            ev = hw["TVT_input"].isna().to_numpy()
            if ev.sum() == 0 or len(sp["tvt"]) != int(ev.sum()): continue
            gmin, gmax = float(tw_tvt.min()), float(tw_tvt.max())
            gg = np.interp(np.arange(gmin, gmax + 0.5, 0.5), tw_tvt, tw_gr)
            K = len(gg)
            Fh = _v75_feats_h(hw["GR"].to_numpy(float), hw["MD"].to_numpy(float), hw["Z"].to_numpy(float))
            Ft = _v75_feats_t(gg)
            st = np.asarray(sp["tvt"], np.float64)
            ks = np.round((st - gmin) / 0.5).astype(np.int64)
            cand = np.clip(ks[:, None] + offs[None, :], 0, K - 1)
            okc = (ks[:, None] + offs[None, :] >= 0) & (ks[:, None] + offs[None, :] <= K - 1)
            ev_idx = np.where(ev)[0]
            xh = torch.from_numpy(Fh).to(dev); xt = torch.from_numpy(Ft).to(dev)
            ct = torch.tensor(cand, device=dev)
            acc = None
            for eA, eB in encs:                                  # ★5モデル平均(ユーザー指定)
                a = eA(xh)[ev_idx]; b = eB(xt)
                s = (a[:, None, :] * b[ct]).sum(-1)
                acc = s if acc is None else acc + s
            sims = (acc / len(encs)).cpu().numpy()
            sims = np.where(okc, sims, -1.0)
            _V75_SIM[wid] = dict(sim=sims.astype(np.float16), st=st.astype(np.float32))
            n_ok += 1
    print(f"[v75] test sim 構築: {n_ok}井 (5foldエンコーダ平均)")
'''
c0 = rep(c0, ANCHOR_DEF_END, V75_BLOCK, 1, "S2attach")

c0 = rep(c0, '''    anc = torch.zeros((W, Tmax), dtype=dtype); ancs = torch.full((W, Tmax), 1e9, dtype=dtype)   # v41錨
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
        if "anc" in x:''', 1, "S3pad")
c0 = rep(c0, '''    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs))''',
         '''    return dict(md=to(md), z=to(z), gr=to(gr), valid=to(valid), grid=to(grid), glen=to(glen),
                vmin=to(vmin), step=to(step), gs=to(gs), ls=to(ls), ir=to(ir), anc=to(anc), ancs=to(ancs),
                rtq=to(rtq), rtk=to(rtk), stq=to(stq), simt=(to(simt) if has_sim else None))''', 1, "S3ret")
c0 = rep(c0, 'def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen, dtype=V26_DTYPE):',
         'def _v26_smoother_core(md, z, gr, valid, grid, glen, vmin, step, gs, ls, ir, anc, ancs, P, N, device, gen,\n'
         '                       rtq=None, rtkq=None, stq=None, simt=None, wmap=None, dtype=V26_DTYPE):', 1, "S4sig")
c0 = rep(c0, '''        rate_n = ALPHA * rate + RN * rn()
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''',
         '''        rate_n = ALPHA * rate + RN * rn()
        if rtq is not None:                                   # ★v75 structgate
            rate_n = rate_n + rtkq[:, i][:, None] * (rtq[:, i][:, None] - rate)
        if RATE_CLIP > 0: rate_n = rate_n.clamp(-RATE_CLIP, RATE_CLIP)''', 1, "S4rate")
c0 = rep(c0, '''        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))''',
         '''        avg = (w * lk).sum(1)
        log_lik = log_lik + torch.where(obs, torch.log(avg.clamp_min(1e-300).double()),
                                        torch.zeros((), device=device, dtype=torch.float64))
        if simt is not None:                                  # ★v75 NN emission(重み更新のみ)
            jj = torch.round((v - stq[:, i][:, None]) / 0.5).long() + 90
            inb = (jj >= 0) & (jj <= 180)
            sv = torch.gather(simt[:, i, :].to(dtype)[wmap], 1, jj.clamp(0, 180))
            sv = torch.where(inb, sv, torch.full_like(sv, -1.0))
            lk = lk * torch.exp(V75_WNN * sv)''', 1, "S4sim")
c0 = rep(c0, '''        pf, ps, ll = _v26_smoother_core(
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
            simt=pad["simt"], wmap=wmap)''', 1, "S5ext")
# バンクループ3箇所(submitは build_dataset_v28 内=インデントがcreateと同一のはず)
c0 = rep(c0, '''                _x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
                inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))''',
         '''                _x = build_smoother_inputs(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
                _wid = os.path.basename(str(hp)).split("__")[0]
                inps.append(_attach_v75(_attach_anchor(_x, _wid, physics=(p["label"] in PHYSICS_BANKS)), _wid, use_sim=True))''', 1, "S6tw")
c0 = rep(c0, '''            _x = build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
            inps.append(_attach_anchor(_x, os.path.basename(str(hp)).split("__")[0], physics=(p["label"] in PHYSICS_BANKS)))''',
         '''            _x = build_inputs_self(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk)
            _wid = os.path.basename(str(hp)).split("__")[0]
            inps.append(_attach_v75(_attach_anchor(_x, _wid, physics=(p["label"] in PHYSICS_BANKS)), _wid, use_sim=False))''', 1, "S6self")
c0 = rep(c0, '''            _x = build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk, NREFS[wid])
            inps.append(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)))''',
         '''            _x = build_inputs_nbr(hw, tw["TVT"].to_numpy(float), tw["GR"].to_numpy(float), Pk, NREFS[wid])
            inps.append(_attach_v75(_attach_anchor(_x, wid, physics=(p["label"] in PHYSICS_BANKS)), wid, use_sim=False))''', 1, "S6nbr")
c0 = rep(c0, '''    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)''',
         '''    if "anc" in x:  y["anc"]  = _agg_dec_mean(x["anc"], DEC)
    if "rtq" in x:  y["rtq"]  = _agg_dec_mean(x["rtq"], DEC); y["rtk"] = _agg_dec_mean(x["rtk"], DEC)
    assert "simt" not in x or DEC == 1, "v75: sim使用バンクはDEC=1のみ対応"''', 1, "S7dec")
nb["cells"][0]["source"] = c0.splitlines(keepends=True)

# ---------- cell4: test struct 構築の直後に test sim 構築を挿入 ----------
c4 = "".join(nb["cells"][4]["source"])
c4 = rep(c4, 'v46_build_test_struct(Path(_V26_DATA) / "test")        # ★ _V41_STRUCT を test で埋める(錨ON)',
         'v46_build_test_struct(Path(_V26_DATA) / "test")        # ★ _V41_STRUCT を test で埋める(v75: structgateが参照)\n'
         '_v75_build_test_sims(Path(_V26_DATA) / "test")         # ★v75: test sim(5foldエンコーダ平均) → _V75_SIM', 1, "C4sim")
nb["cells"][4]["source"] = c4.splitlines(keepends=True)

# ---------- モデルARTパス差替(v69_4→v75) ----------
nhit = 0
for cell in nb["cells"]:
    if cell.get("cell_type") != "code": continue
    t = "".join(cell["source"])
    if "artifacts_v69_4" in t:
        t2 = t.replace(r"v69\artifacts_v69_4", r"v75\artifacts_v75").replace(
            "v69/artifacts_v69_4", "v75/artifacts_v75").replace("artifacts_v69_4", "artifacts_v75")
        cell["source"] = t2.splitlines(keepends=True); nhit += 1
(V75 / "submit_nn_v75.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[submit] submit_nn_v75.ipynb 生成 (ARTパス差替 {nhit}セル)")

# ---------- 検証 ----------
full = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
checks = [
    ("_attach_v75", "def _attach_v75" in full),
    ("structgate", "rtkq[:, i][:, None] * (rtq[:, i][:, None] - rate)" in full),
    ("sim乗算", "lk = lk * torch.exp(V75_WNN * sv)" in full),
    ("test sim 5fold平均", "_v75_build_test_sims" in full and "acc / len(encs)" in full),
    ("cell4でsim構築呼出", "_v75_build_test_sims(Path(_V26_DATA)" in full),
    ("tw use_sim=True", "use_sim=True" in full),
    ("self/nbr use_sim=False", full.count("use_sim=False") == 2),
    ("ART=v75", "artifacts_v75" in full),
    ("残v69_4", "artifacts_v69_4" not in full),
]
for lbl, ok in checks: print(("  OK " if ok else "  ×  ") + lbl)
assert all(ok for _, ok in checks)
print("[build_submit_v75] 完了(実行はユーザー。KaggleにはstageA_enccapaug_f0..4.pt=v50/artifacts_v50 を添付)")
