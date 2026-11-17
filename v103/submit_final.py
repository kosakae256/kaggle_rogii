# -*- coding: utf-8 -*-
"""最終提出オーケストレータ(Kaggle, A/B 2モード)。全脚を test で再現し、モード別 blend_final.json で合成。
生成は submit_v104(dec10af=sharp/smooth 込み)を流用 → greedy が学習時と同じ103候補を再現できる。
脚:
 - L5grid  = v103 grid(gseed×K)アンサンブル。**91候補 seqfeat**(v103 ED)で学習=submit_v103.l5_v103_predict をそのまま流用。
 - greedy  = v104 貪欲サブセット(keep27, decaf 4本含む)。**103候補 seqfeat**(v104 ED)で keep → plain L5。
 - TCN     = v103 TCN(出力ブレンド用)。v104 combo の列部分集合で推論。
 - anchor  = GRフリー物理錨 δ(st − last_known)。sim_grfree_test_v97 + 103-seqfeat rows。
モード: A={L5grid,TCN,anchor} / B=A+greedy(blend_final.json の重み)。各モード後に expansion 後処理。
env SUBMIT_MODE=A/B を2回実行して submission_final_A.csv / _B.csv を作る。
"""
import os, sys, io, json, pickle, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False); sys.stderr = io.open(2, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch
import submit_v104 as S104          # gen(dec10af込み)+ _mk_l5 + _tcn(未使用)
import submit_v103 as S103          # l5_v103_predict(L5grid, 91候補)+ _tcn_v103_predict(v103 TCN)

WORK = S104.WORK; DATA = S104.DATA; V104DS = S104.V104DS; V97A = S104.V97A
SCR = S104.SCR; DEVICE = S104.DEVICE; REPS = S104.REPS


def _v104_ed_es():
    """submit_v104.l5_v104_predict と厳密一致の 103候補 ED/ES(v103 71 + sharp/smooth 12)。"""
    ED = ",".join(["pf_nbr_gr5_delta_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_delta_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
                  + ["v100_pf_%s_delta_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
                  + ["dec_%s_delta_1" % t for t in REPS]
                  + ["aff96_ancc_delta_%d" % k for k in range(1, 7)] + ["aff97_ancc_delta_%d" % k for k in range(1, 5)]
                  + ["affpf5_ancc_delta_%d" % k for k in range(1, 6)]
                  + ["decaf_%s_delta_%d" % (t, b) for t in REPS for b in (1, 2)] + ["affdec_ancc_delta_1", "affdec_ancc_delta_2"])
    ES = ",".join(["pf_nbr_gr5_std_%d" % k for k in (1, 2, 3)] + ["pf_self_graft_std_%d" % k for k in (1, 2, 3)]
                  + ["v97_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4)]
                  + ["v100_pf_%s_std_%d" % (r, k) for r in REPS for k in (1, 2, 3, 4, 5)]
                  + ["dec_%s_std_1" % t for t in REPS]
                  + ["aff96_ancc_std_%d" % k for k in range(1, 7)] + ["aff97_ancc_std_%d" % k for k in range(1, 5)]
                  + ["affpf5_ancc_std_%d" % k for k in range(1, 6)]
                  + ["decaf_%s_std_%d" % (t, b) for t in REPS for b in (1, 2)] + ["affdec_ancc_std_1", "affdec_ancc_std_2"])
    return ED, ES


def _build_seqfeat103(cpq):
    """103候補 seqfeat(v104 ED)= greedy 学習時と同一。C も 103 で構築。"""
    SF = WORK / "seqfeat_test_final103.pkl"
    ED, ES = _v104_ed_es()
    S104._run("build_seqfeat_v97.py", dict(SF_IN=str(cpq), SF_OUT=str(SF), SF_NO_TARGET="1",
              CANDFEAT="1", NF_MAX="2", V97_NBANK="6", V93_INPUT="fwd", EXTRA_DELTA=ED, EXTRA_STD=ES,
              ROGII_SCRIPTS=str(SCR)), "final 103-seqfeat")
    sf = pickle.loads(SF.read_bytes())
    names = json.loads((Path(str(SF) + ".cand.json")).read_text(encoding="utf-8"))
    return sf, names


def _l5_keep_predict(sf, names, keep_names, ckpt_dir):
    """103-seqfeat を keep_names で候補軸スライス(名前一致, 学習時と同順)→ plain L5(ckpt_dir)平均。"""
    ck = V104DS / ckpt_dir
    if not ck.exists():
        print(f"[greedy] {ckpt_dir} 無し → skip"); return None
    ks = set(keep_names); keep = [i for i, nm in enumerate(names) if nm in ks]
    miss = [nm for nm in keep_names if nm not in set(names)]
    assert not miss, f"greedy keep が 103候補に無い: {miss}"
    assert len(keep) == len(keep_names), f"keep 数不一致 {len(keep)}!={len(keep_names)}"
    keep = np.asarray(keep, dtype=np.int64)
    d = {w: dict(f) for w, f in sf.items()}
    for w in d:
        f = d[w]; f["D"] = f["D"][:, keep]; f["S"] = f["S"][:, keep]
        if f.get("DX") is not None: f["DX"] = f["DX"][:, keep]
    wells = list(d.keys()); w0 = d[wells[0]]
    K = w0["D"].shape[1]; CDIM = w0["C"].shape[1]; NF = w0["DX"].shape[2]
    N = int(max(w["rows"].max() for w in d.values())) + 1
    acc = np.zeros(N, np.float64); cnt = np.zeros(N, np.float64); nm = 0
    for fp in sorted(glob.glob(str(ck / "l5_s*_f*_model.pt"))):
        c = torch.load(fp, map_location=DEVICE)
        assert (not isinstance(c, dict)) or c.get("K", K) == K, f"greedy ckpt K={c.get('K')}≠{K}"
        net = S103._mk_l5(K, CDIM, NF, c.get("dim", 64) if isinstance(c, dict) else 64)
        net.load_state_dict(c["state"] if isinstance(c, dict) and "state" in c else c); net.eval()
        with torch.no_grad():
            for w in wells:
                f = d[w]; T = f["D"].shape[0]
                D = torch.from_numpy(f["D"])[None].to(DEVICE); Sd = torch.from_numpy(f["S"])[None].to(DEVICE)
                C = torch.from_numpy(f["C"])[None].to(DEVICE); m = torch.ones((1, T), device=DEVICE); DX = torch.from_numpy(f["DX"])[None].to(DEVICE)
                o = net(D, Sd, C, m, DX)[0].cpu().numpy().astype(np.float64)
                acc[f["rows"]] += o[f["bid"]]; cnt[f["rows"]] += 1
        nm += 1; del net
        if DEVICE == "cuda": torch.cuda.empty_cache()
    out = np.full(N, np.nan); ok = cnt > 0; out[ok] = acc[ok] / cnt[ok]
    print(f"[greedy] {nm}model cover%={100*ok.mean():.1f} K={K}", flush=True); return out


def _anchor_delta(sf103, lk, N, eval_mask):
    """GRフリー物理錨 δ = st − last_known。sim_grfree_test の st(=eval行のみ, nn_emission_v97:151)を
       103-seqfeat rows へ整列。train は全行=eval で一致するが、test は prefix(既知TVT)+ eval を含むため
       st(eval行)を **eval行サブセット(bin順)** に整列する(prefix行は提出対象外=NaNのままで可)。"""
    sim_p = V97A / "sim_grfree_test_v97.pkl"; anc = np.full(N, np.nan)
    if not sim_p.exists():
        print("[anchor] sim_grfree_test_v97.pkl 無し → anchor=nan", flush=True); return anc
    SIM = pickle.loads(sim_p.read_bytes())
    nkey = nfull = neval = nskip = 0
    for w, dd in SIM.items():
        if w not in sf103:
            continue
        nkey += 1
        st = np.asarray(dd["st"], np.float64); rows = np.asarray(sf103[w]["rows"])
        if len(st) == len(rows):                          # 全行一致(train型)
            anc[rows] = st - lk[rows]; nfull += 1
        else:                                             # eval行のみ(test型): eval subset(bin順)に整列
            erows = rows[eval_mask[rows]]                 # sf103 rows は既に bin(MD)順 → eval subset も MD順
            if len(erows) == len(st):
                anc[erows] = st - lk[erows]; neval += 1
            else:
                nskip += 1
    print(f"[anchor] key一致={nkey} 全行整列={nfull} eval整列={neval} 長さ不一致skip={nskip} "
          f"finite%={100*np.isfinite(anc).mean():.1f}", flush=True)
    return anc


def main():
    mode = os.environ.get("SUBMIT_MODE", "B").upper(); assert mode in ("A", "B")
    S104._warm_imp_cache()
    v96pq, v97pq, v100pq = S104.gen_v100_15banks()
    c15 = S104.build_combo15(v96pq, v97pq, v100pq)
    fwd = S104.gen_v104_fwd()               # ★dec10 + aff + dec10af(sharp/smooth)を test 生成
    cpq = S104.merge_v104(c15, fwd)         # v104 test_combo(103候補分の全列)
    meta = pq.read_table(cpq, columns=["id", "last_known_tvt"]); ids = meta.column("id").to_pandas().to_numpy()
    lk = meta.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64); N = len(ids)
    samp = pd.read_csv(DATA / "sample_submission.csv")
    _subids = set(samp["id"].astype(str)); eval_mask = np.array([str(i) in _subids for i in ids])  # 提出対象(eval)行
    legs = {}
    # L5grid(91候補=submit_v103 と厳密一致。cpq の decaf 列は ED に無いので無視)
    legs["L5grid"] = S103.l5_v103_predict(cpq)
    # 103-seqfeat(greedy + anchor rows 用)
    sf103, names = _build_seqfeat103(cpq)
    # greedy(mode=B のみ必要。A では計算不要=時短)
    if mode == "B":
        gk = V104DS / "greedy_keep.json"
        if gk.exists():
            kj = json.loads(gk.read_text()); kn = kj["keep"] if isinstance(kj, dict) else kj
            legs["greedy"] = _l5_keep_predict(sf103, names, kn, "l5_v104greedyg-1_ckpt")
    legs["anchor"] = _anchor_delta(sf103, lk, N, eval_mask)
    legs["TCN"] = S103._tcn_v103_predict(cpq, ids)
    # ---- blend(SUBMIT_MODE で A/B)----
    bl = json.loads((V104DS / "blend_final.json").read_text())[mode]
    W = bl["weights"]; pred = np.zeros(N, np.float64); wsum = 0.0; used = {}
    for nm, wv in W.items():
        a = legs.get(nm)
        if a is None or not np.isfinite(a).any():
            raise SystemExit(f"[blend {mode}] 脚 {nm} が欠測(weight={wv})=提出不能。生成/ckpt を確認")
        pred += wv * np.nan_to_num(a, nan=0.0); wsum += wv; used[nm] = wv
    pred /= max(wsum, 1e-9)
    GG = bl.get("g"); GK = bl.get("k"); GS = bl.get("s")
    if GG is not None and GK is not None and GS:
        pred = float(GG) * pred * (1.0 + float(GK) * np.abs(pred) / float(GS))
    for nm in ("L5grid", "TCN"):            # 欠測行の保険
        a = legs.get(nm)
        if a is None: continue
        bad = ~np.isfinite(pred) & np.isfinite(a); pred[bad] = a[bad]
    tvt = lk + pred
    sub = pd.DataFrame({"id": ids.astype(str), "tvt": tvt.astype(np.float64)})
    merged = samp[["id"]].merge(sub, on="id", how="left")
    out = WORK / f"submission_final_{mode}.csv"; merged.to_csv(out, index=False)
    print(f"\n==== {out.name} rows={len(merged)} finite%={100*merged['tvt'].notna().mean():.2f} "
          f"mode={mode} used={used} 後処理=({GG},{GK},{GS}) ====", flush=True)
    print(merged.head(6).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
