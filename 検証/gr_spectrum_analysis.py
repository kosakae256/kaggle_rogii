# -*- coding: utf-8 -*-
"""
GR 周波数・周期性 検証
======================
狙い: 「GRセンサーの回転成分が本当にデータに乗っているか？」＋「水平井GRが typewell(縦井戸の
      きれいなGR)より高周波側にどれだけ余分な成分(ノイズ)を持つか＝ローパスで削れる量」を数値化。

背景(物理):
  LWD の GR センサーは掘削管に付き、掘削中に回転する。水平井では上向き=上の地層、下向き=下の地層を
  交互に見るので、上下でGRが違うと「回転周波数の周期変調」が乗りうる(アジマスGR)。
  ただし GR が MD 刻み(~0.5〜1ft)でサンプルされ、しかも1回転を平均済みなら、回転周期(ft/rev は
  ROP/RPM で ~0.01ft オーダー)は Nyquist(=2*刻み)より遥かに細かく **見えない/エイリアスする**はず。
  → 実データで (A)サンプリング (B)スペクトル形 (C)周期性 を確認して、仮説の当否と
     「denoise(ローパス)で削れる高周波帯」を測る。

出力:
  - 標準出力に: dMD統計 / 高周波エネルギー比 / 自己相関の周期ピーク / 解釈
  - 図 gr_spectrum.png (このフォルダ): 中央値PSD(水平井 vs typewell) と 自己相関
実行: rogii_claude/.venv/Scripts/python.exe "検証/gr_spectrum_analysis.py"
※ Claude は実行しない(ユーザーが実行)。scipy/matplotlib 不使用でも動く(numpyのみで自前実装)。
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
PROJECT  = Path(r"C:\Users\kosaka256\Documents\rogii_claude")
DATA_DIR = PROJECT / "rogii-wellbore-geology-prediction" / "train"
OUT_PNG  = Path(__file__).resolve().parent / "gr_spectrum.png"

N_WELLS   = 120     # サンプルする坑井数(全体から等間隔に抽出)
NPERSEG   = 512     # Welch 1セグメント長(点)
MIN_LEN   = 1024    # これ未満(リサンプル後)の系列はスキップ
STEP_CLAMP_MIN = 0.1  # 均一リサンプリング刻みの下限(ft)。実データの dMD 中央値を使うが下限で保護
# ===================================================================


def list_wells():
    return sorted({Path(p).stem.replace("__horizontal_well", "")
                   for p in glob.glob(str(DATA_DIR / "*__horizontal_well.csv"))})


def load_series():
    """サンプル坑井の (水平井 md,gr) と (typewell tvt,gr) を読み、dMD 中央値も集める。"""
    wells = list_wells()
    if len(wells) > N_WELLS:
        idx = np.linspace(0, len(wells) - 1, N_WELLS).astype(int)
        wells = [wells[i] for i in idx]
    hw_list, tw_list, dmd_meds = [], [], []
    for wid in wells:
        try:
            hw = pd.read_csv(DATA_DIR / f"{wid}__horizontal_well.csv", usecols=["MD", "GR"])
        except Exception:
            continue
        md = hw["MD"].to_numpy(float); gr = hw["GR"].to_numpy(float)
        ok = np.isfinite(md)
        md, gr = md[ok], gr[ok]
        if len(md) < 50:
            continue
        order = np.argsort(md); md, gr = md[order], gr[order]
        dmd = np.diff(md); dmd = dmd[dmd > 0]
        if len(dmd) == 0:
            continue
        dmd_meds.append(float(np.median(dmd)))
        # GR 欠損は線形補間
        s = pd.Series(gr).interpolate(limit_direction="both")
        gr = s.to_numpy(float)
        hw_list.append((md, gr))
        # typewell
        try:
            tw = pd.read_csv(DATA_DIR / f"{wid}__typewell.csv", usecols=["TVT", "GR"]).sort_values("TVT")
            tvt = tw["TVT"].to_numpy(float); tgr = tw["GR"].to_numpy(float)
            okt = np.isfinite(tvt) & np.isfinite(tgr)
            if okt.sum() >= 50:
                tw_list.append((tvt[okt], tgr[okt]))
        except Exception:
            pass
    return hw_list, tw_list, np.array(dmd_meds)


def resample_uniform(x, y, step):
    """(x,y) を等間隔 step の格子へ線形補間。"""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < step * MIN_LEN:
        return None
    xg = np.arange(lo, hi, step)
    yg = np.interp(xg, x, y)
    return yg


def welch_psd(y, fs, nperseg):
    """numpy だけの Welch PSD(Hann窓, 50%オーバーラップ, セグメント線形除去)。freq[cycles/ft], psd を返す。"""
    y = np.asarray(y, float)
    n = len(y)
    if n < nperseg:
        return None, None
    step = nperseg // 2
    win = np.hanning(nperseg)
    wsum = (win ** 2).sum()
    segs = []
    for start in range(0, n - nperseg + 1, step):
        seg = y[start:start + nperseg]
        # 線形トレンド除去(地質の大局トレンドを落とし、揺らぎのスペクトルを見る)
        t = np.arange(nperseg)
        a, b = np.polyfit(t, seg, 1)
        seg = seg - (a * t + b)
        seg = seg * win
        f = np.fft.rfft(seg)
        p = (np.abs(f) ** 2) / (fs * wsum)
        segs.append(p)
    if not segs:
        return None, None
    psd = np.mean(segs, axis=0)
    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs)   # cycles per ft
    return freq, psd


def median_psd(series, step, nperseg):
    """複数系列の PSD を共通 freq 軸で中央値化。"""
    fs = 1.0 / step
    acc = []
    freq_ref = None
    for x, y in series:
        yg = resample_uniform(x, y, step)
        if yg is None or len(yg) < MIN_LEN:
            continue
        f, p = welch_psd(yg, fs, nperseg)
        if f is None:
            continue
        freq_ref = f
        acc.append(p)
    if not acc:
        return None, None, 0
    return freq_ref, np.median(np.vstack(acc), axis=0), len(acc)


# numpy 2.x で np.trapz → np.trapezoid に改名。両対応。
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

def hf_fraction(freq, psd, cutoff_wavelength):
    """波長 < cutoff_wavelength(ft) の帯域が全パワーに占める割合。"""
    if freq is None:
        return float("nan")
    fc = 1.0 / cutoff_wavelength                     # cycles/ft
    tot = _trapz(psd, freq)
    hi = _trapz(psd[freq >= fc], freq[freq >= fc]) if (freq >= fc).any() else 0.0
    return float(hi / tot) if tot > 0 else float("nan")


def autocorr_periodicity(series, step, max_lag_ft=60.0, min_lag_ft=3.0):
    """等間隔GRの自己相関(線形除去後)を平均し、地質の周期(互層/Milankovitch層厚)を探す。
    回転(~0.01ft)は1ft刻みで不可視なので、狙いは数ft〜数十ftの周期(双峰datumの層厚 ±15ft 等)。"""
    maxlag = int(max_lag_ft / step)
    acc = []
    for x, y in series:
        yg = resample_uniform(x, y, step)
        if yg is None or len(yg) < 4 * maxlag:
            continue
        t = np.arange(len(yg)); a, b = np.polyfit(t, yg, 1); yg = yg - (a * t + b)
        yg = yg - yg.mean()
        v = np.dot(yg, yg)
        if v <= 0:
            continue
        ac = np.correlate(yg, yg, mode="full")[len(yg) - 1: len(yg) - 1 + maxlag + 1] / v
        acc.append(ac)
    if not acc:
        return None, None, None      # 計算不能(サンプル無し)
    ac = np.median(np.vstack(acc), axis=0)
    lags_ft = np.arange(len(ac)) * step
    # min_lag_ft 以降の最大副ピーク(局所極大)を探す
    lo = max(1, int(min_lag_ft / step))
    peak_lag, peak_val = -1.0, -1.0      # 見つからなければ (-1,-1) = 周期無し
    for i in range(lo + 1, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > peak_val:
            peak_val = ac[i]; peak_lag = lags_ft[i]
    return (lags_ft, ac), peak_lag, peak_val


def main():
    print("=" * 70)
    print("GR 周波数・周期性 検証")
    print("=" * 70)
    hw, tw, dmd_meds = load_series()
    print(f"サンプル坑井: 水平井 {len(hw)} / typewell {len(tw)}")
    if len(dmd_meds) == 0:
        print("データ読めず終了"); return

    # --- (A) サンプリング ---
    dmd_g = float(np.median(dmd_meds))
    print("\n--- (A) MD サンプリング間隔 dMD ---")
    print(f"  dMD 中央値(坑井中央値の中央値) = {dmd_g:.4f} ft")
    print(f"  dMD 分布: 5%={np.percentile(dmd_meds,5):.3f} / 50%={np.percentile(dmd_meds,50):.3f} / 95%={np.percentile(dmd_meds,95):.3f} ft")
    step = max(STEP_CLAMP_MIN, round(dmd_g, 2))
    nyq_wl = 2 * step
    print(f"  → 解析の均一刻み step={step} ft、Nyquist波長={nyq_wl:.2f} ft(これより細かい周期は見えない)")
    print(f"    参考: 回転周期 ft/rev = ROP/(RPM) 例)30ft/hr ÷ 60rpm/60 ≈ 0.008 ft/rev ≪ {nyq_wl:.2f}ft")
    print("         → 素の深度サンプリングでは回転そのものは分解不能(平均化/エイリアス)が理論予想。")

    # --- (B) PSD (水平井 vs typewell) ---
    print("\n--- (B) パワースペクトル(中央値) ---")
    f_h, p_h, n_h = median_psd(hw, step, NPERSEG)
    f_t, p_t, n_t = median_psd(tw, step, NPERSEG)
    print(f"  PSD算出: 水平井 {n_h}本 / typewell {n_t}本")
    # 低周波(baseline/offset帯)の支配度: 波長>cutoff のエネルギー比。ここに well間のオフセット差が乗る。
    print("  [低周波帯=baseline/offset。ハイパスで落とす対象]")
    for wl in (200.0, 100.0, 50.0, 20.0):
        lf_h = 1.0 - hf_fraction(f_h, p_h, wl)
        lf_t = (1.0 - hf_fraction(f_t, p_t, wl)) if f_t is not None else float("nan")
        print(f"    波長>{wl:>5.0f}ft のエネルギー比: 水平井={lf_h:.3f}  typewell={lf_t:.3f}")
    # 高周波(wiggle=照合に効く帯)
    print("  [中〜高周波帯=wiggle。ハイパスで残す対象]")
    for wl in (10.0, 4.0, nyq_wl * 1.001):
        fr_h = hf_fraction(f_h, p_h, wl)
        fr_t = hf_fraction(f_t, p_t, wl) if f_t is not None else float("nan")
        print(f"    波長<{wl:>4.1f}ft のエネルギー比: 水平井={fr_h:.3f}  typewell={fr_t:.3f}")

    # --- (C) 自己相関の周期性(地質の互層周期。回転は1ft刻みで不可視なので数ft〜数十ftを見る) ---
    print("\n--- (C) 自己相関の周期ピーク(互層/層厚の周期性) ---")
    ac_h, plag_h, pval_h = autocorr_periodicity(hw, step)
    if ac_h is None:
        print("  自己相関算出できず(サンプル不足)")
    elif plag_h is None or plag_h < 0:
        print("  3〜60ft に明瞭な周期ピーク無し(単調減衰=赤色ノイズ的)。互層の卓越周期は検出されず。")
    else:
        print(f"  水平井: 最大副ピーク = lag {plag_h:.1f} ft, 値 {pval_h:.3f}")
        if pval_h > 0.15:
            print(f"    → 約 {plag_h:.0f}ft 周期の互層構造あり。error_anatomy の双峰datum(±層厚)と符合するか要確認。")
        else:
            print("    → 弱い周期(値<0.15)。地質のゆるい繰り返し程度。")

    # --- (D) 図 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        if f_h is not None:
            ax[0].loglog(f_h[1:], p_h[1:], label=f"horizontal ({n_h})", lw=1.5)
        if f_t is not None:
            ax[0].loglog(f_t[1:], p_t[1:], label=f"typewell ({n_t})", lw=1.5, alpha=0.8)
        ax[0].axvline(1.0 / nyq_wl, color="k", ls=":", lw=1)
        ax[0].set_xlabel("spatial frequency [cycles/ft]"); ax[0].set_ylabel("PSD")
        ax[0].set_title("Median GR power spectrum"); ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)
        if ac_h is not None:
            lags_ft, ac = ac_h
            ax[1].plot(lags_ft, ac, lw=1.5)
            ax[1].axhline(0, color="k", lw=0.6)
            ax[1].set_xlabel("lag [ft]"); ax[1].set_ylabel("autocorr")
            ax[1].set_title("Median GR autocorrelation (horizontal)"); ax[1].grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110)
        print(f"\n図を保存: {OUT_PNG}")
    except Exception as e:
        print(f"\n(図はスキップ: {e})")

    # --- 解釈 ---
    print("\n--- 解釈の指針(ハイパス仮説) ---")
    print("  ・回転(~0.01ft/rev)は 1ft 刻みで原理的に不可視。denoise(ローパス)の対象HFノイズも乏しい。")
    print("  ・要点は逆で **ハイパス**: 低周波帯に well間の baseline/offset 差(GR_h≈a·GR_tw+b の b と緩ドリフト)が乗る。")
    print("    エネルギーは低周波支配だが、その多くは『照合を汚す交絡』。ローパスは交絡を残し wiggle を捨てる=逆効果。")
    print("  ・**ハイパス(=長波長のオフセット/ドリフトを除去し、中〜細スケールの wiggle を残す)** が照合の理屈に合う。")
    print("    これは calibration(a,b fit)を『フィルタで b と低周波ゲインを落とす』形にした等価物。")
    print("  ・カットオフ設計: 波長>数十〜100ft(オフセット/超緩ドリフト)を落とし、~2〜50ft(地質wiggle)は残す。")
    print("    実装候補: GR - 長窓移動平均(=高域通過) を水平井・typewell 両方に適用してから照合。窓長を探索。")


if __name__ == "__main__":
    main()
