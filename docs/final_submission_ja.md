# Final Submission（最終提出の構成）

最終的に選んだ提出は **A / B の 2 つ**で、どちらも **独立な複数の「脚（予測器）」を固定重みでブレンドし、最後に 1 パラメータの expansion 較正（de-shrink）を掛けた**ものです。主提出は **B**（脱相関脚を 1 本多く積んだ版）。

## 脚（legs）

| 脚 | 中身 | A 重み | B 重み |
|---|---|--:|--:|
| **L5grid** | 本命。候補カーブ統合器（row-level ensembler）。**16 の Optuna 調整 PF × 5 GR 表現 × full 平滑 → 91 候補**の TVT 軌跡を、行ごとに attention で soft 平均。group-CV seed（3）× top-K（5/8/10）を変えた **9 構成のクロス・グループ・アンサンブル**。 | 0.828 | 0.634 |
| **greedy** | v104 の**貪欲サブセット**。103 候補（dec10-affine の sharp/smooth 4 本を含む拡張セット）から **keep27** を選抜し、素の L5 に通した脚。単体は弱いが L5grid と脱相関するのでブレンドで効く（B のみ）。 | — | 0.232 |
| **TCN** | 同じ候補特徴に対する per-well の時系列 CNN（残差 / MDN）。系列モデル由来の脱相関脚。 | 0.131 | 0.098 |
| **anchor** | **GR フリー物理 prior** の δ（GR を一切使わず、幾何と近傍 dip だけで作る TVT。厳密 5-fold OOF で構築）。小さな安定化脚。 | 0.041 | 0.037 |

## ブレンド + 後処理

- **A** = 0.828·L5grid + 0.131·TCN + 0.041·anchor
- **B** = 0.634·L5grid + 0.232·greedy + 0.098·TCN + 0.037·anchor
- そのうえで **expansion 較正（de-shrink）**：ŷ = g·p·(1 + k·|p|/s)
  （A: g = 0.964, k = 0.027 ／ B: g = 0.959, k = 0.028、s = 10 固定）。
  統合器の soft 平均で縮んだ振幅を戻す、1〜2 パラメータの大域補正。

## スコア

| 提出 | CV | Public | Private |
|---|--:|--:|--:|
| A | 5.429 | ___ | 5.997 |
| **B（主提出）** | **5.416** | ___ | **5.984** |

## 最終提出を支える主要部品

- **16 の PF パラメータセット**（Optuna 個別最適化。粒子数・GR 尤度σ・gr_power・遷移ノイズ・resample・錨/emission の有無を意図的にばらして脱相関を作る）
- **5 つの GR 表現**：typewell / self（自井戸の既知 prefix）/ neighbor（XY 近傍）/ GR 類似近傍 / self-graft（self を較正外挿）
- **GR フリー物理 prior**（幾何 X,Y,Z ＋ 既知 TVT だけで dip 場を推定・積分。**必ず厳密 5-fold OOF** で構築）
- **full 平滑**（全区間の粒子平滑）。脱相関バンク（greedy 由来の dec10 / tw-affine）は **emission OFF** で多様性を確保
- 予測は最後まで **Particle Filter が生む候補**が主役で、L5 / TCN / anchor はそれを**行レベルでバギング**する役割
