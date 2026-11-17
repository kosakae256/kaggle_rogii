# -*- coding: utf-8 -*-
"""候補オラクル診断: L5が選ぶ候補トークン(PF各rep×bankのdelta)で、
各行「真値に最も近い候補を選べたら」の overall RMSE を出す。
→ 現L5(5.7961)との差 = 選択の伸びしろ。差大=選択器改善が金脈 / 差小=候補生成が律速。
CPUのみ(parquet列読み)。省メモリ(候補列を1本ずつ読み、running min-abs-error更新)。
実行: PARQUET=... .venv/Scripts/python.exe v100/cand_oracle.py
"""
import os, sys, io, re
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
import numpy as np, pyarrow.parquet as pq

P = os.environ.get("PARQUET", r"C:\Users\kosaka256\Documents\rogii_claude\v99\artifacts_v99\data\train_combo.parquet")
REPS = ("ancc", "self", "nbr", "nbr_gr5", "self_graft")


def main():
    sch = pq.read_schema(P)
    names = set(sch.names)
    # 候補列 = (v96無印 / v97_) の pf_{rep}_delta_{k}(GR相関PF予測=targetと同じ量)。z_delta等は別量なので除外。
    cands = []
    for pre in ("", "v97_"):
        for rep in REPS:
            k = 1
            while f"{pre}pf_{rep}_delta_{k}" in names:
                cands.append(f"{pre}pf_{rep}_delta_{k}"); k += 1
    print(f"parquet={os.path.basename(P)}")
    print(f"候補列数(rep×bank delta): {len(cands)}")
    y = pq.read_table(P, columns=["target"]).column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    N = len(y); fin_y = np.isfinite(y)
    print(f"行数={N}")

    # 各候補の単体RMSE + oracle(running best-abs)
    best_abs = np.full(N, np.inf); oracle_err = np.zeros(N)
    per_rmse = {}
    for c in cands:
        v = pq.read_table(P, columns=[c]).column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        m = np.isfinite(v) & fin_y
        per_rmse[c] = float(np.sqrt(np.mean((v[m] - y[m]) ** 2))) if m.sum() else np.nan
        err = v - y; ae = np.abs(err)
        upd = np.isfinite(ae) & (ae < best_abs)
        best_abs[upd] = ae[upd]; oracle_err[upd] = err[upd]
    om = np.isfinite(best_abs) & (best_abs < np.inf) & fin_y
    oracle_rmse = float(np.sqrt(np.mean(oracle_err[om] ** 2)))

    # 単体ベスト候補
    bestc = min(per_rmse, key=lambda k: per_rmse[k])
    # top候補5つ
    top = sorted(per_rmse.items(), key=lambda kv: kv[1])[:8]

    print("\n=== 単体候補 RMSE(良い順 top8) ===")
    for c, r in top:
        print(f"  {c:28s} {r:.4f}")
    print(f"\n単体ベスト候補: {bestc} = {per_rmse[bestc]:.4f}")
    print(f"\n★ 候補オラクル(各行 真値最近傍を選択) overall RMSE = {oracle_rmse:.4f}  (被覆 {100*om.mean():.1f}%)")
    print("\n=== 比較 ===")
    print(f"  候補オラクル    : {oracle_rmse:.4f}")
    print(f"  現 L5(選択器)   : 5.7961")
    print(f"  現 TCN(回帰)    : 5.9952")
    print(f"  現 blend        : 5.7038")
    print(f"  → L5の選択ギャップ(L5 - oracle) = {5.7961 - oracle_rmse:+.4f}  (これが選択器改善の理論最大余地)")

    # 参考: top-2オラクル(真値近傍2候補の良い方=枝が2択に絞れたら の近似)は full oracleと同義になるので、
    #      代わり「候補中央値」= 素朴選択 の床
    # メモリ節約のため中央値は概算(top8のみ)ではなく省略。単体ベストとの対比で十分。
    print(f"\n  素朴床(単体ベスト単独) {per_rmse[bestc]:.4f} → オラクル {oracle_rmse:.4f} = 選択で {per_rmse[bestc]-oracle_rmse:.4f} 取れる余地")


if __name__ == "__main__":
    main()
