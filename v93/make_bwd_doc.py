# -*- coding: utf-8 -*-
"""逆方向(backward)実験の結果を fold 別まで集計して docs/v93_backward.md にまとめる。
   3系統を比較: fwd(baseline) / fwdbwd(warm=forward終端init) / fwdbwd_anchor(GRフリー錨終端init)。
   各系統で L1 GBM(9model blend) / L2 TCN(ens) / L5(mean) の overall + fold別 RMSE、
   等重み&nnls blend、fwd との差、および bpf vs pf の per-well corr(重複ビュー監視)を出す。
   存在しない OOF はスキップ(部分結果でも書ける)。実行: python v93/make_bwd_doc.py"""
import io, sys, os, json, glob
sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

PROJ = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
ART = PROJ / "v93" / "artifacts_v93"
DOC = PROJ / "docs" / "v93_backward.md"
BASE_PQ = ART / "data" / "train.parquet"

# ---- fold(GroupKFold5, 全脚共通)復元 ----
tb = pq.read_table(BASE_PQ, columns=["well", "target"])
well = tb.column("well").to_pandas().astype(str).to_numpy()
y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
N = len(y); del tb
FOLDS = list(GroupKFold(n_splits=5).split(np.zeros(N), y, groups=well))
FOLD_OF = np.full(N, -1, np.int64)
for fi, (_, va) in enumerate(FOLDS):
    FOLD_OF[va] = fi


def rmse(p, t, mask=None):
    m = np.isfinite(p) & np.isfinite(t)
    if mask is not None:
        m &= mask
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2))) if m.any() else float("nan")


def per_fold(p):
    return [rmse(p, y, FOLD_OF == fi) for fi in range(5)]


def load_gbm(suf):
    fp = ART / (f"l1_oof{suf}.npz")
    if not fp.exists():
        return None
    d = np.load(fp, allow_pickle=True)
    keys = [k for k in d.files if k not in ("y", "well")]
    return np.mean([d[k] for k in keys], 0).astype(np.float64)


def load_tcn(insuf):
    # nn_oof_combo_top192{insuf}.npz。fwd(insuf="")は旧名 nn_oof_combo.npz もフォールバックで探す。
    # 全特徴(top-k無し)= nn_oof_combo{insuf}.npz / 従来 top192 = nn_oof_combo_top192{insuf}.npz
    pats = [f"nn_oof_combo{insuf}.npz", f"nn_oof_combo_top192{insuf}.npz"] + (["nn_oof_combo.npz"] if insuf == "" else [])
    fp = None
    for pat in pats:
        if (ART / pat).exists():
            fp = ART / pat; break
    if fp is None:
        return None
    d = np.load(fp, allow_pickle=True)
    keys = [k for k in d.files if k.startswith("mdn2")]
    if not keys:
        return None
    return np.mean([d[k] for k in keys], 0).astype(np.float64)


def load_l5(osuf):
    fp = ART / (f"l5{osuf}_oof.npz")
    if not fp.exists():
        return None
    d = np.load(fp, allow_pickle=True)
    return d["l5_mean"].astype(np.float64) if "l5_mean" in d.files else None


def nnls_blend(mats):
    from scipy.optimize import nnls
    M = np.stack(mats, 1)
    ok = np.all(np.isfinite(M), 1) & np.isfinite(y)
    w, _ = nnls(M[ok], y[ok])
    if w.sum() <= 0:
        w = np.ones(len(mats))
    w = w / w.sum()
    return (M @ w).astype(np.float64), w


# 系統: (表示名, GBM suf, TCN insuf, L5 osuf)
VARIANTS = [
    ("fwd (baseline)", "", "", ""),
    ("fwdbwd (warm=forward終端init)", "_fwdbwd", "_fwdbwd", "_fwdbwd"),
    ("fwdbwd_anchor (GRフリー錨終端init)", "_fwdbwd_anchor", "_fwdbwd_anchor", "_fwdbwd_anchor"),
]

rows = []          # (variant, leg, overall, fold0..4, weight_or_'')
blends = {}        # variant -> overall blend
for vname, gsuf, tinsuf, l5suf in VARIANTS:
    legs = {}
    g = load_gbm(gsuf); t = load_tcn(tinsuf); l = load_l5(l5suf)
    for lname, arr in (("L1 GBM(9blend)", g), ("L2 TCN(ens)", t), ("L5(mean)", l)):
        if arr is None:
            rows.append((vname, lname, None, None, ""))
        else:
            legs[lname] = arr
            rows.append((vname, lname, rmse(arr, y), per_fold(arr), ""))
    if len(legs) >= 2:
        mats = list(legs.values())
        eqb = np.mean(mats, 0)
        rows.append((vname, f"等重み blend({len(mats)})", rmse(eqb, y), per_fold(eqb), ""))
        nb, w = nnls_blend(mats)
        wtxt = " / ".join(f"{k.split('(')[0]}={wi:.2f}" for k, wi in zip(legs, w))
        rows.append((vname, f"nnls blend({len(mats)})", rmse(nb, y), per_fold(nb), wtxt))
        blends[vname] = rmse(nb, y)


# ---- bpf vs pf 重複ビュー診断(corr per well) ----
def corr_diag(pq_path):
    if not Path(pq_path).exists():
        return None
    try:
        sch = set(pq.read_schema(pq_path).names)
    except Exception:
        return None
    if "bpf_ancc_1" not in sch:
        return None
    d = pq.read_table(pq_path, columns=["well", "pf_ancc_1", "bpf_ancc_1"])
    w = d.column("well").to_pandas().astype(str).to_numpy()
    a = d.column("pf_ancc_1").to_numpy(zero_copy_only=False).astype(np.float64)
    b = d.column("bpf_ancc_1").to_numpy(zero_copy_only=False).astype(np.float64)
    cs = []
    _, idx = np.unique(w, return_index=True)
    for wn in w[np.sort(idx)]:
        sel = (w == wn)
        aa, bb = a[sel], b[sel]
        mm = np.isfinite(aa) & np.isfinite(bb)
        if mm.sum() > 5 and np.std(aa[mm]) > 1e-6 and np.std(bb[mm]) > 1e-6:
            cs.append(float(np.corrcoef(aa[mm], bb[mm])[0, 1]))
    return np.array(cs) if cs else None


diag = {
    "fwdbwd (warm)": corr_diag(ART / "data" / "train_fwdbwd.parquet"),
    "fwdbwd_anchor": corr_diag(ART / "data" / "train_fwdbwd_anchor.parquet"),
}

# ---- markdown ----
L = []
L.append("# v93 逆方向(backward)特徴 実験\n")
L.append("> forward smooth-PF に加えて、eval 区間を終端→手前へ逆走させた backward smooth-PF(lag16, n_seed=32)を\n"
         "> forward と同格の full-mirror 特徴(bpf_, ~287列)＋backward固有(到達誤差/位置/fwd比較)にして\n"
         "> L1 GBM / L2 TCN / L5 に投入。init は2通り: **warm**=forward終端 / **anchor**=GRフリー錨終端(独立第2推定)。\n")
L.append("\n## 仮説\n")
L.append("- 逆走は forward と誤差の出方が違う(脱相関した第2の推定)ので blend が改善するはず。\n")
L.append("- v10 で warm(forward終端init)の逆走は CV改善もLB悪化(=forward答えに条件付けされた準重複ビュー)。\n")
L.append("  → **anchor(錨終端init)なら forward に依存しない独立推定**になり、真に効くかを検証する。\n")
L.append("\n## 手法\n")
L.append("- backward: `build_backward_v93.py`(BWD_INIT=warm / anchor)。MDミラー反転・per-eval-row 反転・同一PFパラメータ。\n")
L.append("- 特徴: forward 派生器(`features_v93`)を bpf_ で再利用(shape/gr一致/consensus/self-nbr) + bpf_arrival_*(到達誤差=heel既知値との差, 非リーク) + bpf_md_from_toe/pos_frac + bpf_vs_fwd(EMIT_VS_FWD=1)。\n")
L.append("- 学習: `V93_INPUT=fwdbwd|fwdbwd_anchor` で GBM/TCN/L5(seqfeat は bpf 軌跡ch追加で K=14→26)。CV=GroupKFold5(全脚共通)。\n")

L.append("\n## 結果(overall RMSE = Kaggle指標。target単位)\n")
L.append("\n| 系統 | 脚 | overall | fold0 | fold1 | fold2 | fold3 | fold4 | 備考 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for vname, lname, ov, folds, note in rows:
    if ov is None:
        L.append(f"| {vname} | {lname} | (未生成) | | | | | | |")
    else:
        ftxt = " | ".join(f"{x:.4f}" for x in folds)
        L.append(f"| {vname} | {lname} | **{ov:.4f}** | {ftxt} | {note} |")

if blends:
    L.append("\n## blend(nnls)まとめ\n")
    base = blends.get("fwd (baseline)")
    for vname, bv in blends.items():
        d = "" if base is None else f"(fwd比 {bv-base:+.4f})"
        L.append(f"- **{vname}**: nnls blend overall = **{bv:.4f}** {d}")

L.append("\n## bpf vs pf 重複ビュー診断(per-well corr(bpf_ancc_1, pf_ancc_1))\n")
L.append("> 0.98+ なら逆走=forwardの重複(CV利得は過学習疑い)。0.6-0.9 なら独立性あり=本物の第2推定。\n")
for name, cs in diag.items():
    if cs is None:
        L.append(f"- {name}: (parquet 未生成 or bpf 無し)")
    else:
        L.append(f"- **{name}**: n={len(cs)}井 corr mean={cs.mean():+.3f} "
                 f"p10={np.percentile(cs,10):+.3f} median={np.median(cs):+.3f} p90={np.percentile(cs,90):+.3f} "
                 f"| corr>0.95 の割合={float((cs>0.95).mean()):.2%}")

L.append("\n## 所感(自動生成の骨子。後で追記)\n")
if blends:
    base = blends.get("fwd (baseline)")
    for vname, bv in blends.items():
        if vname == "fwd (baseline)" or base is None:
            continue
        verdict = ("改善" if bv < base - 1e-4 else ("悪化" if bv > base + 1e-4 else "同等"))
        L.append(f"- {vname}: fwd baseline({base:.4f})に対し {verdict}({bv-base:+.4f})。")
L.append("- ★CV改善は割り引いて見る(v10 で CV改善→LB悪化の前科)。重複corr が高いなら CV利得は過学習疑い。\n")

DOC.parent.mkdir(parents=True, exist_ok=True)
DOC.write_text("\n".join(L), encoding="utf-8")
print(f"saved: {DOC}")
print("\n".join(L[:60]))
