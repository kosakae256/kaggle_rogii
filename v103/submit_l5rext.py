# -*- coding: utf-8 -*-
"""v103 L5R+ext 提出オーケストレータ(Kaggle, 単体提出)。
狙い: v103 の L5(91候補)を L5R 化(候補の leg/rep/family/k 埋込 + 候補間relational attention)し、
      さらに **TCN OOF δ** と **GRフリー物理錨 δ** を候補として注入(ext)した単体モデル。5seed×5fold(GroupKFold det)。

流れ(submit_v103 の gen/merge を再利用):
 (1)(2)(3) v100 15バンク + dec10/tw-affine を test 生成 → v103 test_combo(91候補分の特徴)   ← submit_v103 と同一
 (4) ext候補を test で再現:
      - tcn_ext   = v103 TCN の test δ(parquet 行順)
      - anchor_ext= st(GRフリー錨, sim_grfree_test_v97)− last_known(δ)。SF rows 経由で train と同一整列。
 (5) seqfeat 2パス: pass1(ext無, v103 ED)で rows 取得 → ext npz 構築 → pass2(EUNET_NPZ 注入)= K=93 seqfeat。
 (6) L5R 推論: l5r_v104l5rdetext_ckpt の (seed,fold) model を平均。
 (7) 後処理 expansion(postproc_l5rext.json)→ submission。

env: V103SUB(dataset), V100SUB(=同じ), NN_SEEDS(既定5), N_SPLITS(既定5)
"""
import os, sys, io, json, pickle, subprocess
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False); sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch, torch.nn as nn
import submit_v103 as S           # gen/merge/TCN 再利用(module-level env は ENV セルで設定済)

WORK = S.WORK; DATA = S.DATA; PY = S.PY; V103DS = S.V103DS; V97A = S.V97A
DEVICE = S.DEVICE; REPS = S.REPS
CKDIR = V103DS / "l5r_v104l5rdetext_ckpt"     # L5R checkpoints(l5r_s{s}_f{k}_model.pt)


# ============ v103 の 91候補 ED/ES(submit_v103.l5_v103_predict と厳密一致)============
def _v103_ed_es():
    ED = (["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
          + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
          + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
          + ["dec_%s_delta_1" % t for t in REPS]
          + ["aff96_ancc_delta_%d" % k for k in range(1, 7)] + ["aff97_ancc_delta_%d" % k for k in range(1, 5)]
          + ["affpf5_ancc_delta_%d" % k for k in range(1, 6)])
    ES = (["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
          + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
          + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
          + ["dec_%s_std_1" % t for t in REPS]
          + ["aff96_ancc_std_%d" % k for k in range(1, 7)] + ["aff97_ancc_std_%d" % k for k in range(1, 5)]
          + ["affpf5_ancc_std_%d" % k for k in range(1, 6)])
    return ",".join(ED), ",".join(ES)


def _build_seqfeat(pq_path, out_pkl, eunet_npz=None):
    ED, ES = _v103_ed_es()
    env = dict(SF_IN=str(pq_path), SF_OUT=str(out_pkl), SF_NO_TARGET="1", CANDFEAT="1", NF_MAX="2",
               V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES, ROGII_SCRIPTS=str(S.SCR))
    if eunet_npz:
        env["EUNET_NPZ"] = str(eunet_npz)
    S._run("build_seqfeat_v97.py", env, "L5R seqfeat" + (" +ext" if eunet_npz else " (rows)"))
    return pickle.loads(Path(out_pkl).read_bytes())


# ============ ext候補(tcn_ext / anchor_ext)を test で再現 ============
def _build_ext_npz(cpq, ids, lk, sf1):
    """tcn_ext = TCN test δ(parquet行順)/ anchor_ext = st − lk(sim_grfree_test, SF rows経由=train同一)。"""
    N = len(ids)
    tcn_delta = S._tcn_v103_predict(cpq, ids).astype(np.float64)          # (N,) parquet行順 δ
    # 錨: sim_grfree_test_v97.pkl の st を SF1 rows へ整列(train eunet 構築と同一手続き)
    sim_p = V97A / "sim_grfree_test_v97.pkl"
    anc = np.full(N, np.nan)
    if sim_p.exists():
        SIM = pickle.loads(sim_p.read_bytes()); nok = 0
        for w, d in SIM.items():
            if w not in sf1:
                continue
            st = np.asarray(d["st"], np.float64); rows = sf1[w]["rows"]
            if len(st) != len(rows):
                continue
            anc[rows] = st - lk[rows]; nok += 1
        print(f"[ext] anchor_ext: {nok}井整列 finite%={100*np.isfinite(anc).mean():.1f}", flush=True)
    else:
        print("[ext] WARN sim_grfree_test_v97.pkl 無し → anchor_ext=nan(候補欠測)", flush=True)
    npz = WORK / "eunet_test_l5rext.npz"
    np.savez(npz, tcn_ext=tcn_delta.astype(np.float32), anchor_ext=anc.astype(np.float32))
    print(f"[ext] {npz.name}: tcn_ext(δ mean={np.nanmean(tcn_delta):.2f}) + anchor_ext -> EUNET 注入用", flush=True)
    return npz


# ============ L5R(train_l456_v98 VERBATIM 移植)============
def _cand_attr(cand_names, dim):
    import re as _re
    def _attr(nm):
        if nm.endswith("_ext"):
            return "ext", "other", "external", 0
        if nm.startswith("v97_"): leg = "v97"
        elif nm.startswith("v100_"): leg = "v100"
        elif nm.startswith("decaf_"): leg = "decaf"
        elif nm.startswith("affdec_"): leg = "affdec"
        elif nm.startswith("dec_"): leg = "dec"
        elif nm.startswith("aff96_"): leg = "aff96"
        elif nm.startswith("aff97_"): leg = "aff97"
        elif nm.startswith("affpf5_"): leg = "affpf5"
        else: leg = "v96"
        rep = "other"
        for rp in ("nbr_gr5", "self_graft", "ancc", "self", "nbr"):
            if ("_" + rp) in ("_" + nm):
                rep = rp; break
        if leg in ("aff96", "aff97", "affpf5"): fam = "aff"
        elif leg in ("decaf", "affdec"): fam = "anchorfree"
        elif leg == "dec": fam = "dec"
        elif ("beam" in nm) or nm.startswith("sc_"): fam = "special"
        else: fam = "pf"
        _mm = _re.search(r"_(\d+)$", nm); kidx = min(int(_mm.group(1)), 6) if _mm else 0
        return leg, rep, fam, kidx
    _LG, _RP, _FM = {}, {}, {}; legid = []; repid = []; famid = []; kid = []
    for nm in cand_names:
        lg, rp, fm, ki = _attr(nm)
        legid.append(_LG.setdefault(lg, len(_LG))); repid.append(_RP.setdefault(rp, len(_RP)))
        famid.append(_FM.setdefault(fm, len(_FM))); kid.append(ki)
    print(f"[l5r] 候補構造: legs={list(_LG)}({len(_LG)}) reps={list(_RP)}({len(_RP)}) fams={list(_FM)}({len(_FM)})", flush=True)
    return dict(leg=torch.tensor(legid, device=DEVICE), rep=torch.tensor(repid, device=DEVICE),
                fam=torch.tensor(famid, device=DEVICE), kidx=torch.tensor(kid, device=DEVICE),
                n_leg=len(_LG), n_rep=len(_RP), n_fam=len(_FM), n_k=7)


def _mk_l5r(K, CDIM, NF, dim, ATTR):
    class L5R(nn.Module):
        def __init__(s):
            super().__init__(); s.k = K
            s.enc = nn.Sequential(nn.Conv1d(2 + NF, dim, 9, padding=4), nn.GELU(),
                                  nn.Conv1d(dim, dim, 9, padding=16, dilation=4), nn.GELU(),
                                  nn.Conv1d(dim, dim, 9, padding=64, dilation=16))
            s.emb = nn.Embedding(K, dim)
            s.leg_emb = nn.Embedding(ATTR["n_leg"], dim); s.rep_emb = nn.Embedding(ATTR["n_rep"], dim)
            s.fam_emb = nn.Embedding(ATTR["n_fam"], dim); s.kidx_emb = nn.Embedding(ATTR["n_k"], dim)
            s.register_buffer("leg_id", ATTR["leg"]); s.register_buffer("rep_id", ATTR["rep"])
            s.register_buffer("fam_id", ATTR["fam"]); s.register_buffer("kid", ATTR["kidx"])
            s.cq = nn.Linear(dim, dim); s.ck = nn.Linear(dim, dim); s.cv = nn.Linear(dim, dim)
            s.cproj = nn.Linear(dim, dim); s.cln = nn.LayerNorm(dim)
            s.b_leg = nn.Parameter(torch.zeros(())); s.b_rep = nn.Parameter(torch.zeros(())); s.b_fam = nn.Parameter(torch.zeros(()))
            s.qry = nn.Sequential(nn.Linear(CDIM + 4, dim), nn.GELU(), nn.Linear(dim, dim)); s.res = nn.Linear(dim, 1)

        def _cand_emb(s):
            return (s.emb.weight + s.leg_emb(s.leg_id) + s.rep_emb(s.rep_id) + s.fam_emb(s.fam_id) + s.kidx_emb(s.kid))

        def _rel_bias(s):
            L, R, F = s.leg_id, s.rep_id, s.fam_id
            return (s.b_leg * (L[:, None] == L[None, :]).float() + s.b_rep * (R[:, None] == R[None, :]).float()
                    + s.b_fam * (F[:, None] == F[None, :]).float())

        def forward(s, D, Sd, C, m, DX=None):
            B, T, kk = D.shape; med = D.median(2, keepdim=True).values
            x = torch.stack([D - med, torch.log(Sd)] + ([DX[..., i] for i in range(DX.shape[-1])] if DX is not None else []), 2)
            x = x.permute(0, 3, 2, 1).reshape(B * kk, 2 + (DX.shape[-1] if DX is not None else 0), T)
            hc = s.enc(x).reshape(B, kk, -1, T).permute(0, 3, 1, 2) + s._cand_emb()[None, None]
            q_ = s.cq(hc); k_ = s.ck(hc); v_ = s.cv(hc)
            ac = (q_ @ k_.transpose(-1, -2)) / (hc.shape[-1] ** 0.5) + s._rel_bias()[None, None]
            hc = s.cln(hc + s.cproj(ac.softmax(-1) @ v_))
            stats = torch.cat([C, med / 10.0, D.std(2, keepdim=True),
                               (D - med).abs().lt(2).float().mean(2, keepdim=True), torch.zeros_like(med)], -1)
            q = s.qry(stats); att = (hc @ q[..., None])[..., 0] / (hc.shape[-1] ** 0.5); w = att.softmax(-1)
            return (w * D).sum(-1) + 2.0 * torch.tanh(s.res(q)[..., 0])
    return L5R().to(DEVICE)


def l5r_predict(sf2, cand_names):
    wells = list(sf2.keys()); w0 = sf2[wells[0]]
    K = w0["D"].shape[1]; CDIM = w0["C"].shape[1]; NF = w0["DX"].shape[2]
    assert len(cand_names) == K, f"cand数{len(cand_names)}≠K{K}"
    N = int(max(w["rows"].max() for w in sf2.values())) + 1
    acc = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64)
    NS = int(os.environ.get("NN_SEEDS", "5")); NK = int(os.environ.get("N_SPLITS", "5"))
    nck = 0
    for s in range(NS):
        for k in range(NK):
            fp = CKDIR / f"l5r_s{s}_f{k}_model.pt"
            if not fp.exists():
                print(f"  WARN missing {fp.name}"); continue
            ck = torch.load(fp, map_location=DEVICE)
            dim = ck.get("dim", 64) if isinstance(ck, dict) else 64
            ATTR = _cand_attr(cand_names, dim) if nck == 0 else ATTR      # 1回だけ構築(以降流用)
            net = _mk_l5r(K, CDIM, NF, dim, ATTR)
            net.load_state_dict(ck["state"] if isinstance(ck, dict) and "state" in ck else ck); net.eval()
            with torch.no_grad():
                for w in wells:
                    f = sf2[w]; T = f["D"].shape[0]
                    D = torch.from_numpy(f["D"])[None].to(DEVICE); Sd = torch.from_numpy(f["S"])[None].to(DEVICE)
                    C = torch.from_numpy(f["C"])[None].to(DEVICE); m = torch.ones((1, T), device=DEVICE); DX = torch.from_numpy(f["DX"])[None].to(DEVICE)
                    o = net(D, Sd, C, m, DX)[0].cpu().numpy().astype(np.float64)
                    acc[f["rows"]] += o[f["bid"]]; cnt[f["rows"]] += 1
            nck += 1; del net
            if DEVICE == "cuda": torch.cuda.empty_cache()
    out = np.full(N, np.nan); ok = cnt > 0; out[ok] = acc[ok] / cnt[ok]
    print(f"[L5R+ext] {nck}ckpt cover%={100*ok.mean():.1f}", flush=True); return out


def main():
    S._warm_imp_cache()
    v96pq, v97pq, v100pq = S.gen_v100_15banks()
    c15 = S.build_combo15(v96pq, v97pq, v100pq)
    fwd = S.gen_v103_fwd()
    cpq = S.merge_v103(c15, fwd)
    meta = pq.read_table(cpq, columns=["id", "last_known_tvt"]); ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    # pass1: ext無 seqfeat → rows(錨整列に必要)
    sf1 = _build_seqfeat(cpq, WORK / "seqfeat_test_l5rext_rows.pkl")
    ext_npz = _build_ext_npz(cpq, ids, lk, sf1)
    # pass2: EUNET 注入 → K=93 seqfeat(+ .cand.json)
    sf2_p = WORK / "seqfeat_test_l5rext.pkl"
    sf2 = _build_seqfeat(cpq, sf2_p, eunet_npz=ext_npz)
    cand_names = json.loads((Path(str(sf2_p) + ".cand.json")).read_text(encoding="utf-8"))
    pred = l5r_predict(sf2, cand_names)
    # 後処理 expansion(OOF fit 済 params)。無ければ恒等。
    pp = json.loads((V103DS / "postproc_l5rext.json").read_text()) if (V103DS / "postproc_l5rext.json").exists() else {}
    GG = pp.get("g"); GK = pp.get("k"); GS = pp.get("s")
    if GG is not None and GK is not None and GS:
        pred = float(GG) * pred * (1.0 + float(GK) * np.abs(pred) / float(GS))
        print(f"[後処理] expansion g={GG} k={GK} s={GS}", flush=True)
    tvt = lk + pred
    sub = pd.DataFrame({"id": ids.astype(str), "tvt": tvt.astype(np.float64)})
    samp = pd.read_csv(DATA / "sample_submission.csv"); merged = samp[["id"]].merge(sub, on="id", how="left")
    out = WORK / "submission_l5rext.csv"; merged.to_csv(out, index=False)
    print(f"\n==== {out.name} rows={len(merged)} finite%={100*merged['tvt'].notna().mean():.2f} ====", flush=True)
    print(merged.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
