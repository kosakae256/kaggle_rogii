# 地層別成績 — last_known_tvt 時点の所在地層(typewell Geologyトップ区間)ごとに
#   v52 GBM pp の 正解率(|e|<=5ft行率)/pooled RMSE/hard率 を集計。
#   (A) lk所在地層別(井戸単位)  (B) 行の真TVT地層別  (C) lkの層境界距離 vs 難易度
import os, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

P0 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART52 = os.path.join(P0, "v52", "artifacts_v52")
TR = os.path.join(P0, "rogii-wellbore-geology-prediction", "train")
PQ = os.path.join(ART52, "data", "train.parquet")
ORDER = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "MNSS", "BUDA"]

def main():
    tb = pq.read_table(PQ, columns=["well", "target", "md_since", "pf_ancc", "last_known_tvt"])
    y = tb.column("target").to_numpy(zero_copy_only=False).astype(np.float64)
    g = tb.column("well").to_pandas().astype(str).to_numpy()
    mds = np.maximum(tb.column("md_since").to_numpy(zero_copy_only=False).astype(np.float64), 0.0)
    pfa = tb.column("pf_ancc").to_numpy(zero_copy_only=False).astype(np.float64)
    lk = tb.column("last_known_tvt").to_numpy(zero_copy_only=False).astype(np.float64)
    del tb
    oof = np.load(os.path.join(ART52, "gbm_v41_oof.npy")).astype(np.float64)
    pp = json.loads(open(os.path.join(ART52, "pp_params.json")).read())
    e = (oof * (1 - pp["w_pf"]) + (pfa - lk) * pp["w_pf"]) * (1 - np.exp(-mds / pp["tau"])) * pp["alpha"] - y
    fin = np.isfinite(e)
    ytvt = y + lk                                   # 行の真TVT(絶対)
    wells = pd.unique(g)
    # typewell Geologyトップ
    TOPS = {}
    for wn in wells:
        try:
            tw = pd.read_csv(os.path.join(TR, f"{wn}__typewell.csv"))
        except FileNotFoundError:
            continue
        ggeo = tw.dropna(subset=["Geology"])
        tops = {}
        for _, r in ggeo.iterrows():
            nm = str(r["Geology"]).strip()
            if nm in ORDER and nm not in tops:
                tops[nm] = float(r["TVT"])
        TOPS[wn] = tops
    def formation_of(wn, tvt):
        tops = TOPS.get(wn)
        if not tops or not np.isfinite(tvt): return "NA"
        below = [(nm, tv) for nm, tv in tops.items() if tv <= tvt]
        if not below: return "above_" + min(tops, key=tops.get)
        return max(below, key=lambda t: t[1])[0]
    def bdist_of(wn, tvt):
        tops = TOPS.get(wn)
        if not tops or not np.isfinite(tvt): return np.nan
        return float(min(abs(tvt - tv) for tv in tops.values()))
    # 井戸単位: lk所在地層 + 誤差
    lkw = pd.Series(lk).groupby(pd.Series(g)).first()
    W = pd.DataFrame({"lk": lkw})
    W["form"] = [formation_of(wn, W.loc[wn, "lk"]) for wn in W.index]
    W["bdist"] = [bdist_of(wn, W.loc[wn, "lk"]) for wn in W.index]
    dfe = pd.DataFrame({"w": g[fin], "e": e[fin]})
    ge = dfe.groupby("w")["e"]
    W["rmse"] = np.sqrt(ge.apply(lambda s: float(np.mean(s ** 2))))
    W["n"] = ge.size(); W["acc5"] = ge.apply(lambda s: float((s.abs() <= 5).mean()))
    W["hard"] = W["rmse"] > 10
    print("=== (A) lk時点の所在地層別(井戸単位) ===")
    print(f"{'地層':8s} {'井戸':>4s} {'pooledRMSE':>10s} {'wellRMSE med':>12s} {'正解率|e|<=5':>12s} {'hard率':>7s}")
    for fm in ORDER + ["NA"]:
        m = W["form"] == fm
        if m.sum() == 0: continue
        pooled = float(np.sqrt((W.loc[m, "rmse"] ** 2 * W.loc[m, "n"]).sum() / W.loc[m, "n"].sum()))
        print(f"{fm:8s} {m.sum():4d} {pooled:10.3f} {W.loc[m,'rmse'].median():12.2f} "
              f"{(W.loc[m,'acc5']*W.loc[m,'n']).sum()/W.loc[m,'n'].sum():12.1%} {W.loc[m,'hard'].mean():7.1%}")
    # (B) 行の真TVT地層別
    print("\n=== (B) 行の真TVT地層別(全行) ===")
    forms_row = np.empty(len(g), dtype=object)
    for wn in wells:
        ii = np.where(g == wn)[0]
        tops = TOPS.get(wn)
        if not tops:
            forms_row[ii] = "NA"; continue
        names = sorted(tops, key=tops.get)
        edges = [tops[nm] for nm in names]
        idx = np.searchsorted(edges, ytvt[ii], side="right") - 1
        fr = np.array(["above_" + names[0]] + names, dtype=object)[idx + 1]
        forms_row[ii] = fr
    dfr = pd.DataFrame({"f": forms_row[fin], "e": e[fin]})
    gr = dfr.groupby("f")["e"]
    tab = pd.DataFrame({"rows": gr.size(), "rmse": np.sqrt(gr.apply(lambda s: float(np.mean(s**2)))),
                        "acc5": gr.apply(lambda s: float((s.abs() <= 5).mean()))})
    tab = tab.reindex([f for f in (["above_ANCC"] + ORDER) if f in tab.index])
    for f, r in tab.iterrows():
        print(f"{f:12s} rows={int(r['rows']):8d} ({r['rows']/fin.sum():5.1%})  RMSE={r['rmse']:7.3f}  正解率={r['acc5']:6.1%}")
    # (C) lkの層境界距離 vs 難易度
    print("\n=== (C) lk層境界までの距離 vs 難易度(井戸単位) ===")
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 30), (30, 999)):
        m = (W["bdist"] >= lo) & (W["bdist"] < hi)
        if m.sum() == 0: continue
        print(f"  境界距離 {lo:>3}-{hi:<3}ft: n={m.sum():3d}  wellRMSE med={W.loc[m,'rmse'].median():6.2f}  hard率={W.loc[m,'hard'].mean():6.1%}")
    W.to_csv(os.path.join(ART52, "formation_acc_v52.csv"))
    print("\nsaved formation_acc_v52.csv")

if __name__ == "__main__":
    main()
