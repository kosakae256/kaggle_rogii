# -*- coding: utf-8 -*-
"""事前区間(prefix=既知帯)の使い道 診断。CPUのみ(PF不使用)。
1. prefix TVTスパンが eval TVTレンジを何%被覆するか(self被覆不足の定量化)
2. prefix-dip 外挿誤差 vs eval距離(dip prior の賞味期限。hold=dip0 baseと比較)
3. prefix affine較正(GR_well ≈ a·GR_tw(TVT)+b)の gain/offset/相関分布(較正の効き代)
出力: v100/prefix_diag/*.png + サマリ print
実行: .venv/Scripts/python.exe v100/prefix_diag.py
"""
import os, sys, io, glob
try:
    sys.stdout = io.open(1, "w", encoding="utf-8", closefd=False)
except Exception:
    pass
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Meiryo"

TR = os.environ.get("ROGII_DATA", r"C:\Users\kosaka256\Documents\rogii_claude\rogii-wellbore-geology-prediction") + "/train"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_diag")
os.makedirs(OUT, exist_ok=True)


def robust_line(x, y):
    """robust 1次(Theil-Sen簡易: 係数中央値)。戻り (slope, intercept)。"""
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    # サンプリングして傾き中央値(全ペアは重い)
    rng = np.random.RandomState(0)
    m = min(2000, n * (n - 1) // 2)
    i = rng.randint(0, n, m); j = rng.randint(0, n, m)
    ok = i != j
    dx = x[j[ok]] - x[i[ok]]; dy = y[j[ok]] - y[i[ok]]
    good = np.abs(dx) > 1e-6
    if good.sum() < 5:
        return np.nan, np.nan
    sl = np.median(dy[good] / dx[good])
    ic = np.median(y - sl * x)
    return float(sl), float(ic)


def main():
    wells = sorted({os.path.basename(q).split("__")[0] for q in glob.glob(os.path.join(TR, "*__horizontal_well.csv"))})
    print(f"train坑井: {len(wells)}")

    cov = []                    # 診断1: eval被覆率, span幅
    span_pre = []; span_ev = []; span_ev_out = []
    d_dist = []; d_dip = []; d_hold = []   # 診断2
    gains = []; offs = []; corrs = []; resid = []   # 診断3

    nused = 0
    for w in wells:
        try:
            hw = pd.read_csv(os.path.join(TR, f"{w}__horizontal_well.csv"))
            tw = pd.read_csv(os.path.join(TR, f"{w}__typewell.csv")).sort_values("TVT")
        except Exception:
            continue
        if "TVT" not in hw.columns:
            continue
        pre = hw["TVT_input"].notna().to_numpy()
        ev = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
        if pre.sum() < 8 or ev.sum() < 8:
            continue
        nused += 1
        md = hw["MD"].to_numpy(float); tvt = hw["TVT"].to_numpy(float); gr = hw["GR"].to_numpy(float)

        # --- 診断1: 被覆 ---
        pt = tvt[pre]; et = tvt[ev]
        lo, hi = np.nanmin(pt), np.nanmax(pt)
        inside = (et >= lo) & (et <= hi)
        cov.append(np.nanmean(inside))
        span_pre.append(hi - lo); span_ev.append(np.nanmax(et) - np.nanmin(et))
        span_ev_out.append(np.nanmean(~inside))

        # --- 診断2: prefix-dip 外挿 vs 距離 ---
        # prefix末尾~800ftのrobust dip
        mdp = md[pre]; tvp = tvt[pre]
        last_md = mdp[-1]; last_tvt = tvp[-1]
        recent = mdp >= (last_md - 800)
        if recent.sum() >= 5:
            sl, ic = robust_line(mdp[recent], tvp[recent])
        else:
            sl, ic = robust_line(mdp, tvp)
        if np.isfinite(sl):
            mde = md[ev]; tve = tvt[ev]
            pred = last_tvt + sl * (mde - last_md)
            dist = mde - last_md
            d_dist.append(dist.astype(np.float32))
            d_dip.append(np.abs(pred - tve).astype(np.float32))
            d_hold.append(np.abs(last_tvt - tve).astype(np.float32))

        # --- 診断3: affine較正 GR_well ≈ a·GR_tw(TVT)+b(prefix重なり) ---
        twt = tw["TVT"].to_numpy(float); twg = tw["GR"].to_numpy(float)
        m2 = np.isfinite(pt) & np.isfinite(gr[pre])
        if m2.sum() >= 10 and (pt[m2].min() >= twt.min() - 5) and (pt[m2].max() <= twt.max() + 5):
            gr_tw_at = np.interp(pt[m2], twt, twg)
            gw = gr[pre][m2]
            if np.std(gr_tw_at) > 1e-6 and np.std(gw) > 1e-6:
                a, b = robust_line(gr_tw_at, gw)
                if np.isfinite(a):
                    gains.append(a); offs.append(b)
                    corrs.append(float(np.corrcoef(gw, gr_tw_at)[0, 1]))
                    resid.append(float(np.std(gw - (a * gr_tw_at + b))))

    print(f"使用坑井: {nused}")
    cov = np.array(cov); span_pre = np.array(span_pre); span_ev = np.array(span_ev); span_ev_out = np.array(span_ev_out)
    print("\n=== 診断1: prefix TVTスパンによる eval被覆 ===")
    print(f"  eval行のうち prefix TVT帯内の割合: 中央値={np.median(cov)*100:.1f}%  平均={cov.mean()*100:.1f}%")
    print(f"  eval帯が prefix帯から はみ出す割合(未被覆): 中央値={np.median(span_ev_out)*100:.1f}%")
    print(f"  prefix TVTスパン幅[ft]: 中央値={np.median(span_pre):.1f}  / eval TVTスパン幅: 中央値={np.median(span_ev):.1f}")
    print(f"  被覆率<10%の坑井: {(cov<0.1).mean()*100:.1f}%  / <50%: {(cov<0.5).mean()*100:.1f}%")

    dd = np.concatenate(d_dist); dpe = np.concatenate(d_dip); dh = np.concatenate(d_hold)
    print("\n=== 診断2: prefix-dip外挿誤差 vs eval距離(MAE ft) ===")
    bins = [0, 250, 500, 1000, 2000, 4000, 1e9]; lab = ["0-250", "250-500", "500-1k", "1k-2k", "2k-4k", ">4k"]
    for k in range(len(bins) - 1):
        s = (dd >= bins[k]) & (dd < bins[k + 1])
        if s.sum():
            print(f"  距離 {lab[k]:>7s}ft (n={s.sum():>7d}): dip外挿MAE={dpe[s].mean():6.2f}  hold(dip0)MAE={dh[s].mean():6.2f}  改善={dh[s].mean()-dpe[s].mean():+.2f}")

    gains = np.array(gains); offs = np.array(offs); corrs = np.array(corrs); resid = np.array(resid)
    print("\n=== 診断3: prefix affine較正 GR_well≈a·GR_tw+b ===")
    print(f"  n={len(gains)}  gain a: 中央値={np.median(gains):.3f} [p10={np.percentile(gains,10):.3f} p90={np.percentile(gains,90):.3f}]")
    print(f"  offset b: 中央値={np.median(offs):.2f} [p10={np.percentile(offs,10):.2f} p90={np.percentile(offs,90):.2f}]")
    print(f"  相関 corr(GR_well, GR_tw@prefix): 中央値={np.median(corrs):.3f} [p10={np.percentile(corrs,10):.3f}]")
    print(f"  較正後残差std: 中央値={np.median(resid):.2f}  (GR_wellのstd比で解釈)")

    # ===== グラフ =====
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # 1: 被覆率 ECDF
    xs = np.sort(cov); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax[0, 0].plot(xs * 100, ys, lw=2); ax[0, 0].set_title("診断1: eval被覆率 ECDF(prefix TVT帯内のeval行割合)")
    ax[0, 0].set_xlabel("被覆率 [%]"); ax[0, 0].set_ylabel("坑井の累積割合"); ax[0, 0].grid(alpha=.3)
    ax[0, 0].axvline(np.median(cov) * 100, color="r", ls="--", label=f"中央値{np.median(cov)*100:.0f}%"); ax[0, 0].legend()
    # 2: dip vs hold MAE by distance
    mids = []; dipm = []; holdm = []
    for k in range(len(bins) - 1):
        s = (dd >= bins[k]) & (dd < bins[k + 1])
        if s.sum():
            mids.append(lab[k]); dipm.append(dpe[s].mean()); holdm.append(dh[s].mean())
    x = np.arange(len(mids))
    ax[0, 1].plot(x, dipm, "o-", label="prefix-dip外挿", lw=2)
    ax[0, 1].plot(x, holdm, "s--", label="hold(dip=0)", lw=2)
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(mids); ax[0, 1].set_title("診断2: 外挿MAE vs eval距離")
    ax[0, 1].set_xlabel("prefix末端からのMD距離[ft]"); ax[0, 1].set_ylabel("MAE [ft]"); ax[0, 1].grid(alpha=.3); ax[0, 1].legend()
    # 3: gain hist
    ax[1, 0].hist(np.clip(gains, -1, 3), bins=60, color="teal", alpha=.8); ax[1, 0].axvline(1.0, color="r", ls="--", label="a=1(較正不要)")
    ax[1, 0].set_title("診断3: 較正 gain a 分布"); ax[1, 0].set_xlabel("a"); ax[1, 0].set_ylabel("坑井数"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    # 4: corr hist
    ax[1, 1].hist(np.clip(corrs, -1, 1), bins=60, color="darkorange", alpha=.8)
    ax[1, 1].set_title("診断3: prefix相関 corr(GR_well, GR_tw) 分布"); ax[1, 1].set_xlabel("相関"); ax[1, 1].set_ylabel("坑井数"); ax[1, 1].grid(alpha=.3)
    ax[1, 1].axvline(np.median(corrs), color="r", ls="--", label=f"中央値{np.median(corrs):.2f}"); ax[1, 1].legend()
    plt.tight_layout(); fp = os.path.join(OUT, "prefix_diag.png"); plt.savefig(fp, dpi=110)
    print(f"\nグラフ保存: {fp}")


if __name__ == "__main__":
    main()
