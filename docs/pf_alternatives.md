# PF の派生・近縁アルゴリズム候補（GPU適性つき）

> PF頭打ち打破に向けて検討した「パーティクルフィルタの派生／近縁アルゴリズム」の一覧と GPU適性。
> 前提: 状態は1次元(TVT位置)・GR照合で多峰・dip滑らか＋断層ジャンプ・look-ahead合法。
> 最終更新: 2026-07-01

## A. PF直系の派生（SMC系）
- **Auxiliary PF (APF)**: リサンプリングを「次の観測を覗いて」先回り。尤度が鋭いとき粒子枯渇を減らす。
  - GPU可（systematic resampling は cumsum→searchsorted→gather で全部ベクトル化済み＝リサンプリングはGPUの障害ではない）。ただし旨味薄く、grid化すれば不要。
- **Regularized/Kernel PF**: リサンプリング後に連続核で散らす（現状の `RP/RR` の上位版）。退化対策。GPU可。
- **Rao-Blackwellized PF (RBPF)**: 状態の一部が条件付き線形ならKalmanで畳み込み、粒子は非線形部分だけ→分散減。
  この問題では **dip(rate)を条件付き線形に**できる。フルGPU。
- **Particle smoother（FFBS/後方シミュレーション）**: 前向きPF＋後ろ向きで未来(eval後方)を使い過去を修正。look-ahead合法と相性◎。O(N²)はGPU可だが重い。

## B. 別系統の逐次推定（Kalman族）
- **KF/EKF/UKF**: 単峰ガウス前提 → **双峰datumを表現できず不適**。
- **★Gaussian Sum Filter (GSF)**: EKF/UKFを複数本束ねた混合ガウス＝多峰。各ガウスが1モード（±15ftの2位置）を追い尤度で重み付け → **posterior mean を自然に出せる**。双峰tail狙い撃ち。バッチKalmanでGPU可。
- **IMM (Interacting Multiple Model)**: 複数モデル＋切替確率。ビルド↔ラテラル(dip≈0)/断層ジャンプのレジーム変化に。GPU可。
- **EnKF（アンサンブルKalman）**: ジオステアリングの定番だが単峰 → 多峰のこの問題ではPF/GSFに劣る。

## C. グリッド/HMM系（1D状態に素直・最もGPU向き…だったが結果は下記）
- **Histogram/Grid filter（離散ベイズ）**: TVTを格子化して厳密前向き更新。**粒子退化無し**、多峰の事後を完全保持。
- **HMM Forward-Backward**: emission=GR尤度、transition=dip滑らかさ。前向き×後ろ向きで完全な事後（look-ahead込み）→ **期待値=posterior mean=hedge** を無料。dense matmul/1D畳み込みでGPU最適、**seed不要(決定論)**。
- **Viterbi / DTW**: MAP経路・アライメント（dipの形を直接解く）。

## ★却下・見送り
- **DTW と Viterbi は全然ダメ（不採用）**: どちらも **MAP/単一アライメント＝1モードに確定**。双峰tailで **posterior mean(hedge) が出せない** → 確定は midpoint より悪化（[error_anatomy.md](error_anatomy.md) §7）。GPU可否以前に**方向違い**。
- **Auxiliary PF も実装・実験して【ダメ】（2026-07-01, [v23.md](v23.md)）**: v22同枠組みでエンジンをAPFに、先読み強度β・予測点外挿量κも振ったが通常PFを超えず。
  難所は**粒子枯渇でなく「GR尤度≠真値」**なので、先読みは"間違ったモードへ賢く先回り"するだけ。→ **PF系内のアルゴリズム改良(grid置換／APF)は両方否定**。
- **グリッド/HMM は実装・実験して見送り**（[v23.md](v23.md), 2026-07-01）:
  - 1Dグリッド(dip固定): proc_sigma小で「傾きが固定で間違った直線」(cat0 best 14.21=ほぼ直線, GR特徴拾えず)、大で蛇行(~35)。grid系の最良。
  - RBPF(B): 実行したがダメ(14.21にも及ばず)。2Dグリッド(A)はコードのみで結論的検証せず見送り。
  - 教訓: **1次元状態でも dip という隠れ状態の追跡が要る**ため「軽い格子」優位は崩れ、**PFの粒子表現(2D状態pos×dipを効率追跡)が優秀**。**アルゴリズム置換でPF頭打ちは抜けない**。

## まとめ（残す指針）
- **アルゴリズム置換ではPFを超えられなかった**。PFはこの問題に適している（粒子で2D状態を効率表現）。
- もしまだ追うなら **GSF（双峰を明示追跡→posterior mean）** か **RBPF（dipをKalmanで畳み込み）** が候補。ただし優先度は低い。
- PF頭打ちの本筋は別レバー: **カテゴリ別PFチューニング(v22) / 双峰hedge=posterior mean / selectorへGPU PF投入 / 構造**。
- 関連: [error_anatomy.md](error_anatomy.md), [v23.md](v23.md), [hard_wells.md](hard_wells.md)。
