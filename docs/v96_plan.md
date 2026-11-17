# v96 — ChatGPT提案の全手法 評価＋チェックリスト

> `user/rogii_conversation_record.md`(ChatGPTとの会話, 1476行)の全手法を評価し、v96で一個ずつ潰す。
> 済(除外) / 壁(除外) / ★v96候補 で分類。作成 2026-07-31。

## 外部情報(ChatGPTがKaggle引用で提示 — 重要)
- **公開首位 ≈ 5.262**(disc 718670)。我々5.987は射程内。
- **「500粒子で検証井の90%がRMSE<4候補を含むが、順位づけできない」**(disc 707613) = 我々の「候補生成易/選択が壁」を外部確認。
- **逐次PFは序盤ドリフトで一部40超**(disc 723815) = v91知見一致。
- **GR候補は全置換でなくclip小補正で**(disc 722041) = 我々の方針一致。
- 主催者: **PS以前の水平井GRがTypewellより同水平井PS以後GRに対応** = selfテンプレ推奨。

## 評価サマリ
### 済/壁(除外)
- DTW/制約DTW = 試→死(MAP暴走~200)。Kalman = 単峰不適。
- PF/Beam/GBDT/TCN/L5/well-GroupKFold = 済(中核)。
- Candidate ranker/selection(selection regret) = 試→**壁**(v34 6%回収/v47天井/夜間4軸実験で方向0.5 AUC)。
- 近傍地層面dip(空間kriging) = 試→空間粗く marginal(v42/v91)。
- 逐次PFドリフト = 既知。gate(confidence) = v34試→非転移。

### ★v96 チェックリスト(未着手/部分、優先順)
- [ ] **1. ★★Rolling pseudo-PS / Test-time adaptation** (Part3-5, ChatGPT #1優先/「4.x最大の差分」)
      prefixを40/55/70/85%で疑似PS分割→既存予測器をprefix suffixでbacktest(既知TVTで実RMSE)→
      井戸別に良い予測器/重みを推定→本番suffixに適用(global重みへshrink)。catastrophic well(=我々のハード井戸)を直接叩く=リスク制御。
      **これは datum枝選択でなく "どの手法を信じるか"の井戸別選択=壁と別軸**。
- [ ] 2. Global + 井戸固有Ridge補正(階層shrinkage): r_i=b_w+a_w·ΔMD+c_w·ΔZ を prefixで推定→suffix外挿
- [ ] 3. 疑似PS prefix条件付き学習: prefix-encoder→future-decoder(井戸embedding z_w)。TCN/Transformer enc-dec
- [ ] 4. GR-TVT井戸内校正: self-template(方向別 dTVT/dMD>0,<0)+ typewell α(t)滑らかblend(prefix観測密度で)
- [ ] 5. 空間dip候補: dTVT/dMD≈dZ/dMD − g·v (g=地層勾配, v=坑井水平方向)。近傍からg推定→candidate prior
- [ ] 6. 断層/regime-change候補: jump/slope-change/sigmoid を候補集合に明示追加(事前確率低)
- [ ] 7. multi-metric GR loss: NCC/derivative-corr/multi-scale wavelet/soft-DTW/rank-corr/Huber
- [ ] 8. 大域最適化: CMA-ES / differentiable spline over B-spline params(逐次PFでなく全suffix)
- [ ] 9. 類似井戸検索: prefix fingerprint→Siamese/metric learning→挙動warp転写
- [ ] 10. posterior trajectory ensemble: 形状cluster(flat/±dip/curved/fault)→mode別(単峰mean/双峰best/低信頼shrink)
- [ ] 11. catastrophic well制御: confidence gate(top1-2差/mode分離/backtest regret/self-GR非再現)→safe priorへshrink
- [ ] 12. CV: outer well-CV + inner test-time adaptation(検証も本番同様prefix→self適応→suffix)
- [ ] 13. 記録指標: row-weighted/well-median/worst1,5,10%/oracle@K/selection regret/candidate recall<2,4,6/horizon別/prefix長別

### 既にv95に入っている(参考)
- v90 emission尤度(nn_emission) = 済(v95が5.987の理由)。v91 snap物理脚 = v95で−0.027実測(2026-07-31)。

## 進捗ログ
（各項目を潰すごとに追記）

### 進捗
- [x] **1. Rolling pseudo-PS 基盤** → ✗**不発**(2026-07-31)。self-templateはprefix TVT範囲のみ(backtest=新地質で外挿15.9)、prefix backtestはsuffix難度を予測せず(AUC0.488, ρ0.02)。prefixが局所すぎて遠方suffixを予測不能。v96/exp_rolling_ps_v96.py。

- [x] **候補別特徴(CANDFEAT: 各候補トークンにloglik/vs_tw)** → ★★**WIN**(2026-07-31)。L5 seed0 6.385→6.322(−0.063)、blend 5.987→5.967(−0.020, seed0のみ)。ユーザー案「推論値だけでは足りない」実証。CEXTRA(global→query)は失敗、per-candidate→tokenが正解。次=3seed化+L7版+候補特徴追加(gr_resid/rate)。

- [x] **候補別特徴 3seed確定** → ★★★**大WIN**(2026-07-31)。L5cf 3seed単体6.194(vs ext6.311, −0.117)、**blend 5.987→5.9416(−0.0455)**。ユーザー洞察「推論値だけでは足りない/特徴が足りてない」が本物の成果に。次=10役割拡張版(loglik/vs_tw/兄弟/agree×2/ancc gr_abs0/gr_best_resid/rate/vs_spatial/vs_dense)+L7版+snap脚併用で~5.92狙い。

- [x] **候補特徴 sweet spot** → ★3役割(loglik/vs_tw/兄弟乖離)=seed0 6.303が最良(2役割6.322/5役割6.305/10役割6.357過学習)。兄弟乖離(候補が姉妹表現と一致するか)が有効な3つ目。ancc/agreeは過学習。3seed化+snap併用中。

## ★★★v96 確定成果 (2026-07-31)
- **v95 5.9871 → v96 5.9149 (−0.072, 転移安全)**。2成果の合わせ技:
  1. **候補別特徴 CANDFEAT(2役割 loglik/vs_tw を各候補トークンに)**: L5 6.311→6.194(−0.117)、blend −0.0455。well-CV OOF転移安全。ユーザー洞察「推論値だけでは足りない」。CEXTRA(global→query)は失敗、per-candidate→tokenが正解。3役割(兄弟)以上は過学習/seed noiseで2役割がsweet spot。
  2. **snap物理脚 固定weight0.08**: 追加−0.027。非リーク・v91転移則。
- 実装: build_seqfeat_v95b(CANDFEAT env, DX=(nb,K,nF)) / train_l456_v95b(NF自動, enc 2+NF ch, forward DX伝搬) / snap_leg_v95.py。
- 次候補: L7(ランキング)にCANDFEAT / TCN脚は既に全特徴使用 / さらなる候補特徴探索(過学習注意)。
