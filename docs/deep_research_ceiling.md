# deep-research: 5.4ft 天井の突破口(2026-08-05)

109エージェント・全敵対的検証済み。ROGII TVT予測(生GR↔typewell アライン=1D逐次位置推定)が ~5.44ft で停滞している状況への回答。

## ★最重要結論: 天井は本物(datum は GR だけでは解けない)
- **GR類似だけを消費する手法(DTW/フィルタ/CNN)は bimodal datum を原理的に解けない**。文献で datum を解けた唯一の手段は**人間が与える stratigraphic tie-point**で、隠しテスト坑井には無い(Samant 2025 が独立裏付け)。
- 我々の **AUC 0.50 の壁 = 外部文献と一致**。**3.2ft oracle は到達不能。~5.4 は実質限界に近い**。
- refuted(過剰主張として棄却): sMCL は oracle-selector にならない / stochastic-DTW は「両枝サンプル」を実証せず / soft-DTW の汎化優位も未生存。

## 伸び代 = 「脱相関脚」だけ(貪欲 −0.04 と同型。天井破りでなく ens 積み増し)
検証で生き残った、生GR=別情報源の脱相関脚候補(PF候補を再処理しない):

| # | 手法 | 機構 | datum? | コスト | 出典 |
|---|---|---|---|---|---|
| ① | **制約付き Soft-DTW 脚**(本命) | 生GR↔typewell を **slope/step制約付き**で soft-DTW アライン → δ。生類似だけだと over-warp=傾き制約必須。微分可能・二次時間 | 脱相関脚 | 軽 | Cuturi&Blondel soft-DTW arXiv:1703.01541 / DecDTW ICLR2023 arXiv:2303.10778 / Samant 2025(制約必須) |
| ② | stochastic-DTW 曖昧度特徴 | DTW を確率的に多数実現→ばらつきが bimodal ゾーンを検出→L5信頼度特徴/ヘッジ | 曖昧検出 | 軽 | Sena/Al Ibrahim SPWLA2022 |
| ③ | WLFM 自己教師GR埋め込み | masked+層序contrastive で坑井跨ぎ層序埋め込み | 脱相関脚 | 中 | arXiv:2509.18152 |
| ④ | mode-aware(sMCL/rMCL)+微分Viterbi | モード平均を止め分離・ヘッジ。**枝は選べない** | ヘッジのみ | 中 | sMCL NeurIPS2016 / rMCL NeurIPS2023 / Mensch&Blondel ICML2018 |

## open questions(真のボトルネック)
- ①②③のどれが corr≪0.98 の脱相関脚を許容コストで出すか。積み重ねて L5 相関に戻らないか。
- 人間 tie-point 無しの**自動/学習した層序制約**が constrained-DTW の正枝利得を再現できるか(= AUC 0.50 を破れるか)。破れれば真の突破。

## 方針
- **datum 直撃は非推奨(解けない)**。**「制約付きSoft-DTW脚 + 貪欲複数脚」で ens を積む**が費用対効果最良。
- 関連: [[datum-direction-wall-final]](deep-researchが確証)/ [[v104-greedy-decorrelated-leg]](脱相関脚が効く実証)
