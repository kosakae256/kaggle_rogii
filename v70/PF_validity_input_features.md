# 複数パーティクルフィルタの妥当性判定に用いる入力特徴

## 1. 問題設定

複数のパーティクルフィルタ（Particle Filter; PF）が、それぞれ状態推定値

\[
\hat{x}_t^{(1)}, \hat{x}_t^{(2)}, \ldots, \hat{x}_t^{(K)}
\]

を出力している状況を考える。

目的は、時刻 \(t\) において各PFがどの程度妥当かを推定し、次のいずれかを行うことである。

- 最良のPFを選択する
- 各PFに重みを付けて統合する
- PF統合後の推定値をニューラルネットワークで補正する

重要なのは、PFの推定値そのものだけではなく、以下の情報を入力することである。

1. 観測との整合性
2. 粒子集合の健全性
3. 推定分布の形状
4. 時間方向の安定性
5. PF間の相対的な関係
6. 観測環境やPF設定に関するコンテキスト

ESSや共分散など単一の指標だけでは、PFが真値に近いかを十分には判断できない。複数種類の診断情報を組み合わせる必要がある。

---

## 2. 観測との整合性

PFの妥当性を判断する上で、最も優先度が高い特徴である。

PF \(k\) の事前予測状態を

\[
\hat{x}_{t|t-1}^{(k)}
\]

観測モデルを

\[
y_t = h(x_t) + v_t
\]

とすると、innovationまたは観測残差は

\[
r_t^{(k)}
=
y_t -
h\left(
\hat{x}_{t|t-1}^{(k)}
\right)
\]

で定義される。

### 2.1 基本的な観測残差特徴

各PFについて、以下を入力候補とする。

\[
r_t^{(k)}
\]

\[
\left|r_t^{(k)}\right|
\]

\[
\left(r_t^{(k)}\right)^2
\]

観測次元が複数ある場合は、ベクトルそのものに加え、L1ノルムやL2ノルムも利用できる。

\[
\left\|r_t^{(k)}\right\|_1
\]

\[
\left\|r_t^{(k)}\right\|_2
\]

### 2.2 正規化観測残差

PFごとに観測予測共分散 \(S_t^{(k)}\) が得られる場合、次の正規化残差が有力である。

\[
d_{t,k}^{\mathrm{obs}}
=
r_t^{(k)\top}
\left(S_t^{(k)}\right)^{-1}
r_t^{(k)}
\]

単純な残差では、PFが想定している不確実性を考慮できない。

一方、この量を用いると、

- 残差が大きい
- PFが主張する不確実性に対して残差が大きすぎる

という状況を区別しやすくなる。

### 2.3 Predictive likelihood

非Gaussianな粒子分布を扱う場合は、平均と共分散だけで近似するより、粒子を使って観測尤度を評価する方が自然である。

PF \(k\) の事前粒子を

\[
x_{t|t-1}^{(i,k)}
\]

事前重みを

\[
w_{t|t-1}^{(i,k)}
\]

とすると、predictive likelihoodは

\[
p(y_t \mid y_{1:t-1}, \mathrm{PF}_k)
\approx
\sum_i
w_{t|t-1}^{(i,k)}
p\left(
y_t \mid x_{t|t-1}^{(i,k)}
\right)
\]

となる。

数値安定性のため、入力には通常、対数尤度を使用する。

\[
\ell_{t,k}^{\mathrm{pred}}
=
\log
p(y_t \mid y_{1:t-1}, \mathrm{PF}_k)
\]

観測モデルが適切に定義されている場合、predictive log-likelihoodはPFの妥当性を表す強い特徴になる。

---

## 3. 粒子重みの健全性

粒子重みの偏りは、サンプル退化や推定の不安定化を表す。

### 3.1 Effective Sample Size

PF \(k\) の粒子重みを \(w_t^{(i,k)}\) とすると、

\[
\mathrm{ESS}_{t,k}
=
\frac{1}{
\sum_i
\left(w_t^{(i,k)}\right)^2
}
\]

である。

粒子数 \(N_k\) がPFごとに異なる場合は、

\[
\mathrm{ESSRatio}_{t,k}
=
\frac{\mathrm{ESS}_{t,k}}{N_k}
\]

と正規化する。

### 3.2 Weight entropy

\[
H_{t,k}
=
-
\sum_i
w_t^{(i,k)}
\log
w_t^{(i,k)}
\]

最大値が粒子数に依存するため、

\[
\bar{H}_{t,k}
=
\frac{H_{t,k}}{\log N_k}
\]

として正規化すると扱いやすい。

### 3.3 その他の重み特徴

次も入力候補になる。

- 最大重み
- 上位1%、5%、10%の重み合計
- 重みの分散
- 重みのperplexity
- Gini係数
- resampling実行フラグ
- resamplingからの経過時間

例えば最大重みは

\[
w_{t,k}^{\max}
=
\max_i
w_t^{(i,k)}
\]

である。

上位5%の重み合計は

\[
\mathrm{Top5Mass}_{t,k}
=
\sum_{i \in \mathrm{Top5\%}}
w_t^{(i,k)}
\]

と定義できる。

ただし、ESSが高いことは、PFが真値に近いことを保証しない。

すべての粒子が誤った位置に均等に分布している場合でも、ESSは高くなる。したがって、ESSは観測残差やpredictive likelihoodと組み合わせる必要がある。

---

## 4. 粒子分布の形状

PFの平均と共分散だけでなく、粒子分布の形状を表す特徴も有効である。

### 4.1 平均と共分散

PF \(k\) の重み付き平均は

\[
\mu_{t,k}
=
\sum_i
w_t^{(i,k)}
x_t^{(i,k)}
\]

共分散は

\[
P_{t,k}
=
\sum_i
w_t^{(i,k)}
\left(
x_t^{(i,k)}-\mu_{t,k}
\right)
\left(
x_t^{(i,k)}-\mu_{t,k}
\right)^\top
\]

である。

入力候補は以下である。

- 共分散の対角成分
- \(\operatorname{trace}(P_{t,k})\)
- \(\log \det(P_{t,k}+\epsilon I)\)
- 最大固有値
- 最小固有値
- condition number
- 各状態次元の標準偏差
- credible interval幅

### 4.2 非対称性と裾の重さ

状態次元ごとに、重み付きskewnessやkurtosisを計算できる。

これらは、平均と共分散では表現できない分布形状を補足する。

### 4.3 Multimodality

PFが多峰性を持つ場合、平均状態が実際にはどのモードにも属さないことがある。

入力候補は以下である。

- 粒子クラスタ数
- 最大モードと第2モードの距離
- 最大モードに属する粒子割合
- モード間の重み比
- 分布のdip statistic
- 混合モデルで推定したモード数

ただし、これらを安定して計算するには追加の処理が必要である。

初期実験では、まず平均・共分散・quantileなどの要約統計量を用い、不足する場合にraw particle encoderへ拡張する方がよい。

---

## 5. 予測から更新への変化量

PFが観測によってどれだけ修正されたかも重要な診断情報になる。

事前推定と事後推定の差を

\[
\Delta_{\mathrm{update},t}^{(k)}
=
\hat{x}_{t|t}^{(k)}
-
\hat{x}_{t|t-1}^{(k)}
\]

とする。

正規化量として、

\[
d_{\mathrm{update},t}^{(k)}
=
\Delta_{\mathrm{update},t}^{(k)\top}
\left(
P_{t|t-1}^{(k)}
\right)^{-1}
\Delta_{\mathrm{update},t}^{(k)}
\]

も利用できる。

更新量が大きい場合、次の可能性がある。

- 予測モデルが外れている
- 観測に外れ値がある
- システムのregimeが変化した
- PFが発散しかけている
- 観測が非常に強い情報を持っている

更新量単独では意味が曖昧なため、観測残差、観測品質、他PFの更新量と合わせて評価する。

---

## 6. 時間方向の安定性

PFの妥当性は、単一時刻の情報だけでは判断しにくい。

過去 \(L\) 時刻の履歴を入力する。

\[
Z_{t,k}
=
[
z_{t-L+1,k},
\ldots,
z_{t,k}
]
\]

### 6.1 状態推定履歴

\[
\hat{x}_{t-L+1:t}^{(k)}
\]

に加え、一次差分、

\[
\Delta \hat{x}_t^{(k)}
=
\hat{x}_t^{(k)}
-
\hat{x}_{t-1}^{(k)}
\]

二次差分、

\[
\Delta^2 \hat{x}_t^{(k)}
=
\Delta \hat{x}_t^{(k)}
-
\Delta \hat{x}_{t-1}^{(k)}
\]

を利用する。

対象によっては、速度、加速度、jerkに相当する。

### 6.2 診断統計量の履歴

以下の履歴も有効である。

- innovation
- predictive log-likelihood
- ESS
- weight entropy
- 最大重み
- 共分散
- resampling履歴
- 観測更新量
- PF間距離
- PFの順位

### 6.3 残差の移動統計

観測残差の移動平均は

\[
\bar{r}_{t,k}
=
\frac{1}{L}
\sum_{\tau=t-L+1}^{t}
r_{\tau,k}
\]

である。

残差の自己相関も利用できる。

\[
\rho_{r,k}(1)
=
\operatorname{corr}
\left(
r_{\tau,k},
r_{\tau-1,k}
\right)
\]

残差が継続して同じ方向に偏る場合、systematic biasが存在する可能性がある。

履歴長 \(L\) は、対象システムの時定数、サンプリング周期、PFが破綻してから誤差が表面化するまでの時間を基準に設定する。

---

## 7. PF間の相対的な特徴

各PFを独立に評価するだけでなく、PF同士の相対関係を入力することが重要である。

### 7.1 Consensusからの距離

複数PFのrobust consensusを

\[
m_t
=
\operatorname{median}_k
\hat{x}_t^{(k)}
\]

とすると、PF \(k\) のconsensusからの距離は

\[
d_{t,k}^{\mathrm{consensus}}
=
\left\|
\hat{x}_t^{(k)}
-
m_t
\right\|
\]

である。

状態次元ごとにスケールが異なる場合は、共分散で正規化する。

\[
d_{t,k}^{\mathrm{consensus}}
=
\sqrt{
\left(
\hat{x}_t^{(k)}-m_t
\right)^\top
C_t^{-1}
\left(
\hat{x}_t^{(k)}-m_t
\right)
}
\]

### 7.2 PF間距離行列

\[
D_{t,i,j}
=
\left\|
\hat{x}_t^{(i)}
-
\hat{x}_t^{(j)}
\right\|
\]

を計算する。

ここから、各PFについて次の特徴を作れる。

- 他PFとの平均距離
- 最近傍PFとの距離
- 最遠PFとの距離
- consensusに近いPF数
- 一定距離以内に存在するPF数
- PF間クラスタの所属
- 共分散楕円の重なり

### 7.3 PF内順位

各時刻で以下の順位を入力することも有効である。

- predictive likelihood順位
- 観測残差順位
- ESS順位
- entropy順位
- consensus距離順位
- 共分散trace順位

順位特徴は、異なるスケールを持つPF間の比較を容易にする。

ただし、多数派のPFが同じモデルや同じ観測仮定を共有している場合、誤差が相関する。

したがって、PF間の合意だけで妥当性を判定せず、

\[
\text{PF間合意}
+
\text{観測整合性}
\]

として利用する必要がある。

---

## 8. 観測と環境コンテキスト

PFの統計量だけでは、現在の観測環境や運動状態を十分に表現できない。

共有コンテキストとして以下を入力する。

- 生の観測 \(y_t\)
- 制御入力 \(u_t\)
- サンプリング間隔 \(\Delta t\)
- 欠損マスク
- sensor ID
- sensor confidence
- SNR
- saturation flag
- outlier flag
- 利用可能なsensor数
- 観測数
- 外部環境変数
- 動作モード
- regime情報

欠損値を0で補完するだけではなく、欠損マスク

\[
m_t^{\mathrm{obs}}
\in
\{0,1\}^{d_y}
\]

を別に与える。

例えばGPS遮蔽時に特定PFが強い場合、GPSの利用可否を入力しなければ、そのPFを選択する根拠をモデルが得られない。

---

## 9. PFの設定情報

複数PFが異なるモデルやハイパーパラメータを持つ場合、その情報も入力する。

候補は以下である。

- process noise \(Q_k\)
- measurement noise \(R_k\)
- 粒子数 \(N_k\)
- proposal distribution
- motion model
- 観測モデル
- resampling方式
- resampling threshold
- 使用sensor
- 想定regime
- PFのモデルID

固定されたPF集合だけを扱う場合は、PF ID embeddingでもよい。

未知のPFや新しい設定へ一般化したい場合は、PF番号ではなく、設定値そのものをencodeする方が望ましい。

---

## 10. 推奨する最小入力セット

初期実験では、各PFについて次の特徴から始める。

\[
z_{t,k}
=
\begin{bmatrix}
\hat{x}_{t|t-1}^{(k)}
\\
\hat{x}_{t|t}^{(k)}
\\
\operatorname{diag}
P_{t|t}^{(k)}
\\
r_t^{(k)}
\\
d_{t,k}^{\mathrm{obs}}
\\
\ell_{t,k}^{\mathrm{pred}}
\\
\mathrm{ESS}_{t,k}/N_k
\\
H_{t,k}/\log N_k
\\
\max_i w_t^{(i,k)}
\\
\Delta_{\mathrm{update},t}^{(k)}
\\
d_{t,k}^{\mathrm{consensus}}
\\
\mathrm{resampled}_{t,k}
\end{bmatrix}
\]

共有コンテキストは

\[
c_t
=
[
y_t,
m_t^{\mathrm{obs}},
u_t,
\Delta t,
\mathrm{sensor\ quality}
]
\]

とする。

モデルには過去 \(L\) ステップをまとめて入力する。

\[
Z_{t,k}
=
[
z_{t-L+1,k},
\ldots,
z_{t,k}
]
\in
\mathbb{R}^{L \times F}
\]

---

## 11. iTransformerへ入力する場合

今回の問題では、各PFを1つのtokenとして扱う構成が自然である。

PF \(k\) の履歴特徴を

\[
Z_{t,k}
\in
\mathbb{R}^{L \times F}
\]

とし、temporal encoderまたはMLPでembeddingする。

\[
e_{t,k}
=
\phi_{\mathrm{PF}}
\left(
Z_{t,k}
\right)
\]

PF tokenを並べる。

\[
E_t
=
[
e_{t,1},
e_{t,2},
\ldots,
e_{t,K}
]
\]

PF間Self-Attentionを適用する。

\[
H_t
=
\operatorname{TransformerEncoder}
\left(
E_t
\right)
\]

各PFのスコアを出力する。

\[
s_{t,k}
=
\operatorname{MLP}
\left(
H_{t,k}
\right)
\]

混合重みは

\[
\alpha_{t,k}
=
\operatorname{softmax}_k
\left(
s_{t,k}
\right)
\]

である。

最終推定は

\[
\hat{x}_t
=
\sum_{k=1}^{K}
\alpha_{t,k}
\hat{x}_t^{(k)}
\]

とする。

### 11.1 共有コンテキストの与え方

共有観測コンテキスト \(c_t\) は、次の方法で入力できる。

#### 方法A: 全PF tokenへ結合

\[
e_{t,k}
=
\phi
\left(
[
Z_{t,k},
C_t
]
\right)
\]

#### 方法B: Context embeddingを加算

\[
e_{t,k}
=
\phi_{\mathrm{PF}}
\left(
Z_{t,k}
\right)
+
\phi_{\mathrm{context}}
\left(
C_t
\right)
\]

#### 方法C: Observation tokenを追加

\[
E_t
=
[
e_{t,1},
\ldots,
e_{t,K},
e_t^{\mathrm{obs}}
]
\]

初期実装では、全PF tokenに同じcontext embeddingを加える方法が単純で扱いやすい。

---

## 12. Raw particleを直接使う拡張

要約統計量だけでは、粒子分布の多峰性や非対称性を十分に表せない場合がある。

その場合は、各PFの粒子集合

\[
\mathcal{X}_{t,k}
=
\left\{
\left(
x_t^{(i,k)},
w_t^{(i,k)}
\right)
\right\}_{i=1}^{N_k}
\]

をDeepSetsやSet Transformerでencodeする。

例えば、

\[
h_{t,k}^{\mathrm{particle}}
=
\rho
\left(
\sum_i
\phi
\left(
x_t^{(i,k)},
w_t^{(i,k)}
\right)
\right)
\]

とし、要約統計量のembeddingと結合する。

\[
e_{t,k}
=
[
e_{t,k}^{\mathrm{summary}},
h_{t,k}^{\mathrm{particle}}
]
\]

ただし、raw particle encoderは計算量が増え、学習も不安定になりやすい。

最初から使用するのではなく、要約統計量によるベースラインを構築した後に導入する。

---

## 13. 入力特徴の優先順位

入力を増やしすぎると、過学習や特徴間冗長性が生じる。

次の順序で追加するのがよい。

1. 観測残差とpredictive likelihood
2. PF間距離とconsensusからの偏差
3. ESS、entropy、最大重み
4. 事前推定から事後推定への更新量
5. 時系列履歴
6. 共分散固有値、quantile、multimodality
7. PF設定情報
8. raw particle encoder

特に重要なのは、

\[
\boxed{
\text{観測整合性}
+
\text{PF間相対情報}
+
\text{時間履歴}
}
\]

である。

PF出力とESSだけでは、oracleに近い選択性能を得るのは難しい可能性が高い。

---

## 14. 推奨アブレーション

以下の順序でモデルを比較する。

### Experiment 1: 推定値のみ

\[
\hat{x}_{t}^{(k)}
\]

のみを入力する。

### Experiment 2: 観測整合性を追加

- innovation
- normalized innovation
- predictive log-likelihood

を追加する。

### Experiment 3: 粒子健全性を追加

- ESS
- entropy
- 最大重み
- resampling flag

を追加する。

### Experiment 4: PF間特徴を追加

- consensus距離
- PF間平均距離
- likelihood順位

を追加する。

### Experiment 5: 時系列履歴を追加

過去 \(L\) ステップをGRU、TCN、iTransformerなどでencodeする。

### Experiment 6: 分布形状を追加

- 共分散固有値
- quantile
- skewness
- kurtosis
- multimodality

を追加する。

### Experiment 7: Raw particle encoder

DeepSetsまたはSet Transformerで粒子集合を直接encodeする。

---

## 15. 学習ラベルの設計

時刻 \(t\) の瞬間誤差だけを教師にすると、oracle PFが頻繁に切り替わり、ラベルノイズが大きくなる場合がある。

PF \(k\) の瞬間誤差は

\[
e_{t,k}
=
\left\|
x_t
-
\hat{x}_t^{(k)}
\right\|^2
\]

である。

短区間の平均誤差を教師とする場合は、

\[
e_{t,k}^{(H)}
=
\frac{1}{H}
\sum_{h=0}^{H-1}
\left\|
x_{t+h}
-
\hat{x}_{t+h}^{(k)}
\right\|^2
\]

とする。

これにより、一時的なノイズによるPF切り替えを抑え、継続的に良いPFを学習しやすくなる。

ただし、学習時に未来誤差をラベルとして使うこと自体は可能でも、モデル入力には時刻 \(t\) より未来の情報を含めてはならない。

推論時に利用可能な情報のみを入力する。

---

## 16. 評価指標

最終RMSEだけでなく、以下を評価する。

### 16.1 Hard selector accuracy

oracle PFと同じPFを選べた割合。

\[
\mathrm{Accuracy}
=
\frac{1}{T}
\sum_t
\mathbf{1}
\left[
\hat{k}_t
=
k_t^*
\right]
\]

### 16.2 Selector regret

\[
R
=
\frac{1}{T}
\sum_t
\left(
e_{t,\hat{k}_t}
-
\min_k e_{t,k}
\right)
\]

oracleとの差を直接測る指標である。

### 16.3 Fusion RMSE

\[
\mathrm{RMSE}
=
\sqrt{
\frac{1}{T}
\sum_t
\left\|
x_t
-
\sum_k
\alpha_{t,k}
\hat{x}_t^{(k)}
\right\|^2
}
\]

### 16.4 Calibration

モデルが出力するPF重みや信頼度が、実際の誤差と整合しているかを評価する。

### 16.5 切り替え頻度

\[
\mathrm{SwitchRate}
=
\frac{1}{T-1}
\sum_{t=2}^{T}
\mathbf{1}
\left[
\hat{k}_t
\neq
\hat{k}_{t-1}
\right]
\]

RMSEが同程度でも、切り替えが過剰なモデルは実運用上不安定な場合がある。

---

## 17. 実装上の注意

### 17.1 Ground truth leakage

入力特徴の生成に真値を使ってはならない。

真値は学習ラベルと評価のみに使用する。

### 17.2 状態次元のスケーリング

位置、速度、角度などでスケールが異なる場合、標準化または物理的スケールによる正規化を行う。

### 17.3 PF間の順序

PFの順番に意味がない場合、モデルはpermutation invariantまたはequivariantであることが望ましい。

PF単位tokenにSelf-Attentionを適用する構成は、この条件と相性がよい。

### 17.4 観測品質の変化

学習データにないsensor failureや観測欠損が発生すると、selectorが破綻する可能性がある。

学習時に欠損、外れ値、ノイズ増加などをaugmentationとして含める。

### 17.5 同一系列の分割

時系列データをランダムに時刻単位でtrain/test分割すると、近接時刻の情報が漏洩する。

trajectory、session、subject、environment単位で分割する。

---

## 18. 結論

PFが正しいかを判断する入力として、特に重要なのは次の3群である。

\[
\boxed{
\text{観測との整合性}
}
\]

\[
\boxed{
\text{PF間の相対的な関係}
}
\]

\[
\boxed{
\text{時間方向の安定性}
}
\]

最小構成としては、各PFの推定値に加え、

- innovation
- normalized innovation
- predictive log-likelihood
- ESS ratio
- weight entropy
- 最大粒子重み
- 共分散
- 更新量
- consensusからの距離
- resampling情報
- 過去履歴
- 観測品質

を入力する。

iTransformerを用いる場合は、1つのPFの履歴を1 tokenとして表現し、PF間のSelf-Attentionで相対的な妥当性を推定する構成が適している。

初期段階では要約統計量を使い、性能が飽和した場合にraw particle encoderを追加するのが現実的である。
