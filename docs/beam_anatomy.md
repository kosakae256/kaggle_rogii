# beam search 解剖(2026-07-12, ユーザー依頼調査)

対象: v26/v52 系が内包する `_beam_jit`(6設定 gr/hard/loose/mid/smooth/best、v52 では beam_ 特徴の供給源)。
スクリプト: `v52/beam_anatomy_v52.py` / グラフ: `v52/artifacts_v52/beam_anatomy.png`(80井サンプル)。

## 仕組み
- 状態 = typewell 行インデックス(≈0.5ft/行)。開始 = last_known_tvt 最近傍。評価行1行につき±2行移動可。
- コスト = (GR差)²/es + mc·|移動行数| の累積。ビーム幅12-40 → 最後に最小コスト1本へ hard commit。
- v52 での役割は軌跡でなく特徴(beam_ GR一致・pf_mean_vs_beam 乖離)。

## 実測(80井)
| | pooled RMSE |
|---|---|
| beam 6設定 | 16.0〜16.8(median合議 16.6) |
| PF(pf_1/tw ww平均, 同一井戸) | 8.89 |

- **corr(Δbeam, −ΔZ)=0.010 vs corr(Δ真値, −ΔZ)=0.716** — Zを入力に持たず物理を完全無視(主犯)。
- 移動中央値 0.000ft/行 — move cost が「その場停留」を最安化。実態は lk 横這い+GR気まぐれドリフト
  (naive rmse15 と同族の軌跡で、median合議16.6はそれ以下)。
- |井戸バイアス|>8ft が41%、bias²比率 med 0.47 — 誤差の約半分が定数オフセット(hard commit の枝誤り)。

## 結論
1. 敗因 = ①幾何無視 ②hard commit ③行数単位コスト(typewell密度依存・マジックナンバー要求)。
2. 改修方向(dZ+dip遷移・ft単位・soft化)の行き先は δグリッドHMM = v43(9.7)/v48(8.85)で天井実測済み → **エンジンとしての改修は不採用**。
3. v52 の beam_ 特徴は「弱いが低相関の参照」として現状維持。外す ablation は GBM 再学習1回で可能。
