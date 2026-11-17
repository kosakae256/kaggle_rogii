# 非タブラー推論への提案4件(2026-07-06 メモ)

> v28 GBM(LB 6.026 / 真のCV=train OOF apply_pp ≈7.0)を超えるための「tabular行回帰ではない」方法の提案。
> 根拠: 誤差の88%(bias57.6%+dip31%)は**井戸レベルの構造**なのに、GBMは行単位の独立回帰 — ここが構造的ギャップ。
> 死亡済みで再挑戦しない路線: PF選択 / 素朴平均 / cross-field kriging / PF per-well フルチューニング /
> heuristic予測スタッキング(v24=8.026) / blind後処理 / GR前処理 / 画像CNN読み出し / PF全置換NN。

## ① 系列モデル(per-well TCN/Transformer) on v28特徴 + GBM出力ブレンド 【本命 → v29 で実装済】
- v28 parquet を井戸ごと系列化し、非因果 dilated TCN+自己注意で残差 `TVT−lk` を全行同時回帰。
  GBMと**同一fold**の OOF 同士で最適ブレンド重み探索。
- 根拠: v12 TCN は弱特徴でも GBM とほぼ同水準(7.87 vs 7.99)、ブレンドで −0.16。系列モデルは
  「上流の self/nbr/tw 一致情報で下流 offset を固定する」井戸内文脈伝播が可能(GBMには原理的に不可)。
- 追加オプション: **MDN 2成分混合ヘッド → 混合平均出力** = 双峰datumの posterior-mean hedge を自動化。
- → 詳細 [v29.md](v29.md)。

## ② GBM→PF 閉ループ(pseudo-satellite 反復)— 最終出力を「prior付きsmoother軌跡」に
- smooth PF の尤度に GBM予測を中心とする prior 項 `exp(−0.5((tvt_p − tvt_gbm)/σ)²)`(σ=5〜10ft)を乗せて再実行。
  **train側の prior は必ず OOF 予測**(リーク防止)。σは OOF overall のみで1個だけ調整(overfit-policy)。
- 最終出力は GBM でなく **prior誘導 smoother の軌跡そのもの**(=逐次推論)。GBM=最良点推定 × PF=GR整合+dip連続、
  の map-matching。出典: GSDC iterate-then-re-estimate(analog_competitions #7, medium)。
- リスク: エコーチェンバー(出力≈GBM で無利得)。まず OOF で 7.0 を切るかだけ確認して安く撤退判定。

## ③ 近傍(nbr)証拠による datum 双峰の明示的解消 + posterior mean 【診断先行】
- 井戸ごとに tw-PF から上位2モード(±10〜25ft の datum 候補2つ)を抽出し、各モードの
  **nbr疑似typewell軌跡との整合度**で `p=softmax(margin/T)` → 井戸全体を p 加重 posterior mean へシフト。
- 根拠: GR だけではモード判定はコインフリップ(souldrive 48.8%, r=0.054)と証明済。唯一の外部情報=近傍井。
  GBMは nbr 特徴を持つが「±15ft 2モードの幾何」を行回帰では表現できない。
- **実装前diagnostic必須**: pf_nbr_has=1 の井戸で「nbr整合が真モードを当てる AUC>0.6」を OOF確認 → ダメなら即撤退
  (pp_lab で blind hedge の回収は −0.05 しかなかった前科)。

## ④ (安い即効薬) v28 GBM × v27 heuristic の低重みブレンド
- graft は heuristic 0.7 で負けた(6.191)が、重みが高すぎただけの可能性。`(1−w)·v28 + w·v27heur` を w≈0.15 で LB 直行。
- 注意: heuristic の scale/beam/hold は train 調整済なので train 上の重み最適化は過大評価になる → 重みは決め打ちで LB 確認。

## 優先順・状況
1. **① = v29 実装済**(MDNヘッドで③の思想も半分取り込み)。
2. ④ は 手間ゼロの LB テスト枠が余った時に。
3. ③ は診断のみ先行(AUC>0.6 が出たら実装)。
4. ② は ①の後、「smoother側の最終出力」枠として。

関連: [error_anatomy.md](error_anatomy.md) §4(posterior mean) / [analog_competitions.md](analog_competitions.md) #7 / [v28.md](v28.md) / [pp_lab の教訓=[[v28-self-nbr-gbm]]]。
