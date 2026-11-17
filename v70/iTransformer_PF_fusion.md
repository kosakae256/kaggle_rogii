# 複数パーティクルフィルタ出力の評価・統合におけるiTransformerの適用

**PF間の相関を利用した時系列ゲーティング／フュージョンモデルの設計**

> **要旨**  
> iTransformerは、各変数の時系列履歴をトークン化し、変数間にAttentionを適用するモデルである。この構造は、複数のParticle Filter（PF）が出力する推定値・不確実性・診断統計を比較し、時刻ごとに信頼度を推定する問題と相性がよい。ただし、元のiTransformerをそのまま使用するよりも、PF単位のトークン化、PF内部のTemporal Encoder、cross-PF Attention、重み付き統合損失を組み合わせた構成が適切である。



# 1. iTransformerはこの問題に適しているか

結論として、iTransformerはかなり相性のよい候補である。ただし、元論文の構造をそのまま適用するのではなく、各PFを「変数（variate）」として扱うiTransformer型ゲーティングネットワークに変更するのが適切である。

一般的なTransformerが時刻をトークンとして時間方向にAttentionをかけるのに対し、iTransformerは各変数の一定期間の時系列を1つのトークンとして表現し、変数間にAttentionを適用する。したがって、「複数PFの履歴を比較し、どのPFが信頼できるかを推定する」という目的に自然に対応する。


# 2. PFに適用する場合の基本構成

時刻 t におけるPF kの特徴ベクトルを、次のように定義する。

$$
z_t^{(k)} = \left[\hat{x}_t^{(k)},\, \operatorname{diag}(P_t^{(k)}),\, \mathrm{ESS}_t^{(k)},\, \mathrm{innovation}_t^{(k)},\, \log p(y_t \mid k),\, H(w_t^{(k)})\right]
$$

ここで、推定状態だけでなく、共分散、Effective Sample Size、innovation、観測尤度、particle weightのエントロピーなどを入力に含める。これらはPFの内部的な信頼性を表す診断量である。

過去L時刻分の特徴をまとめ、PFごとの履歴テンソルを作る。

$$
Z_t^{(k)} = \left[z_{t-L+1}^{(k)},\ldots,z_t^{(k)}\right] \in \mathbb{R}^{L \times F}
$$

各PFの履歴を1つの埋め込みベクトルへ変換する。

$$
e_t^{(k)} = \phi\!\left(Z_t^{(k)}\right) \in \mathbb{R}^{d}
$$

その後、PFをトークンとしてSelf-Attentionを適用する。

$$
\left[h_t^{(1)},\ldots,h_t^{(K)}\right]
= \operatorname{TransformerEncoder}\!\left(\left[e_t^{(1)},\ldots,e_t^{(K)}\right]\right)
$$

各PFの表現からスコアと混合重みを計算する。

$$
s_{t,k} = \operatorname{MLP}\!\left(h_t^{(k)}\right)
$$

$$
\alpha_{t,k} = \frac{\exp(s_{t,k}/\tau)}{\sum_j \exp(s_{t,j}/\tau)}
$$

最終的な状態推定は、各PF出力の重み付き和とする。

$$
\hat{x}_t = \sum_k \alpha_{t,k}\hat{x}_t^{(k)}
$$

これは厳密には元のiTransformerそのものではなく、「時系列履歴をトークン化し、変数間Attentionを行う」という設計原理をPF ensembleへ適用したモデルである。


# 3. iTransformerが有効になりやすい理由

複数PFの評価では、個々の推定値だけでなく、PF間の相対関係が重要になる。例えば、以下のような状況である。
- PF 1〜4は近い推定を出しているが、PF 5だけ大きく離れている。
- 特定のPFのinnovationや観測尤度だけが急激に悪化している。
- 複数PFの推定値は近いが、共分散やESSが大きく異なる。
- 特定の観測条件・運動状態・ノイズregimeで、特定モデルのPFだけが破綻する。

cross-PF Attentionは、PF同士の一致、不一致、相関、冗長性、補完関係を明示的に学習できる。全PFの特徴を単純にflattenしてMLPへ入力する方式よりも、PF間比較の帰納バイアスをモデル内部に持たせられる点が利点である。

$$
\operatorname{Attention}\!\left(e^{(i)},e^{(j)}\right)
\;\longrightarrow\;
\text{PF }i\text{ とPF }j\text{ の相対的な信頼関係}
$$


# 4. vanilla iTransformerをそのまま使う場合の問題

元のiTransformerでは、基本的に各scalar variateの履歴を1トークンとして扱う。状態次元D、PF数Kのとき、各PFの各状態次元をすべて独立したトークンにすると、次のグループ構造が曖昧になる。
- どの特徴が同じPFに属しているか。
- どの特徴が同じ状態次元に対応しているか。
- 推定値、共分散、ESS、尤度などの診断量がどのように対応しているか。


## 4.1 PF単位のトークン化

今回の問題では、1つのPFについて、全状態次元と内部統計の履歴をまとめて1トークンとするのが最も自然である。

$$
e_t^{(k)} = \phi\!\left(Z_t^{(k)}\right), \qquad \text{token数}=K
$$

この構成なら、PFの順番に依存しない共有encoderを使いやすく、PF数の変更にも比較的対応しやすい。


## 4.2 階層型iTransformer

状態次元や診断特徴が多い場合は、PF内部とPF間を分けた階層構造も考えられる。

$$
Z_t^{(k)} \rightarrow \text{within-PF encoder} \rightarrow e_t^{(k)} \rightarrow \text{cross-PF Attention}
$$

ただし、初期実験ではモデルを複雑化しすぎず、PF単位トークンから開始する方が比較とデバッグが容易である。


# 5. 学習目標と損失関数

最終RMSEをoracleへ近づけることが目的なら、oracle indexの分類だけでなく、統合後の状態誤差を直接最適化するべきである。

$$
\mathcal{L}_{\mathrm{state}}
= \frac{1}{T}\sum_t
\left\|x_t - \sum_k \alpha_{t,k}\hat{x}_t^{(k)}\right\|_2^2
$$

加えて、学習時の真値を使い、各PFの誤差からsoft oracle分布を作る。

$$
e_{t,k} = \left\|x_t-\hat{x}_t^{(k)}\right\|_2^2
$$

$$
q_{t,k} = \operatorname{softmax}\!\left(-e_{t,k}/\tau_o\right)
$$

$$
\mathcal{L}_{\mathrm{oracle}} = D_{\mathrm{KL}}\!\left(q_t\,\|\,\alpha_t\right)
$$

全体の損失は、例えば次のように構成できる。

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{state}}
+ \lambda_o\mathcal{L}_{\mathrm{oracle}}
+ \lambda_s\sum_t\left\|\alpha_t-\alpha_{t-1}\right\|_1
$$

最初の実験では、λₒを0.1〜1.0程度で探索し、切り替えペナルティλₛは一旦0としてベースラインを作るのが扱いやすい。PFの優劣が時間的に持続することを確認した後に、切り替え抑制を追加する。


# 6. hard-selection oracle RMSE 4は性能上限ではない

現在のoracleが、各時刻に最良のPFを1つだけ選ぶhard-selection oracleである場合、RMSE 4は統合モデルの理論的な性能上限ではない。

$$
k_t^* = \arg\min_k \left\|x_t-\hat{x}_t^{(k)}\right\|^2
$$

重み付き統合では、複数PFの誤差が異なる方向に分布していれば、誤差を相殺できる可能性がある。したがって、convex-fusion oracleも計算するべきである。

$$
\alpha_t^* = \arg\min_{\alpha}
\left\|x_t-\sum_k\alpha_k\hat{x}_t^{(k)}\right\|^2
$$

$$
\text{subject to}\quad \alpha_k\ge 0,\qquad \sum_k\alpha_k=1
$$

$$
\hat{x}_t^{(\mathrm{convex\ oracle})}
= \sum_k \alpha_{t,k}^*\hat{x}_t^{(k)}
$$

評価時には、少なくとも次の4段階を比較する。

| **比較対象**          | **意味**                                       |
|-----------------------|------------------------------------------------|
| Best fixed PF         | 全データで固定した場合に最も良い単一PF。       |
| Hard-selection oracle | 各時刻で真値に最も近いPFを1つ選択。            |
| Convex-fusion oracle  | 各時刻で最適な非負・総和1の混合重みを使用。    |
| Learned fusion        | 推論時に利用可能な特徴だけから混合重みを予測。 |

iTransformer fusionの直接的な性能上限としては、hard-selection oracleよりもconvex-fusion oracleの方が適切な基準になる。


# 7. iTransformerが効きにくい可能性がある条件

以下の条件では、GRUやTCNなどの小規模な時系列モデルの方が安定する可能性がある。
- PF数が2〜4個程度と少なく、PF間Attentionの利点が限定的である。
- 利用する履歴窓が短く、長期的なregime情報がほとんどない。
- PF間の相関よりも、直近数ステップの急変検知が支配的である。
- oracleとなるPFが毎時刻激しく切り替わる。
- 学習データ量が少なく、Transformerの容量に対して不足している。
- 状態空間や観測ノイズのregimeが単純で、固定ルールでも十分に選択できる。

iTransformerは変数間相関のモデリングには強い一方、時間方向の局所パターン抽出を明示的に行うわけではない。そのため、瞬間的な異常や短時間の急変が重要な場合は、前段にTCN、1D-CNN、GRUなどを置く方が合理的である。


# 8. 推奨するハイブリッド構成

本命は純粋なvanilla iTransformerではなく、次のハイブリッドである。

**per-PF Temporal Encoder + cross-PF iTransformer Attention + Gating/Fusion Head**

処理の流れは次のようになる。

$$
Z_t^{(k)} \xrightarrow{\mathrm{TCN/GRU}} e_t^{(k)}
$$

$$
\left[e_t^{(1)},\ldots,e_t^{(K)}\right]
\xrightarrow{\mathrm{Self\mbox{-}Attention}}
\left[h_t^{(1)},\ldots,h_t^{(K)}\right]
$$

$$
h_t^{(k)} \xrightarrow{\mathrm{MLP}} \alpha_{t,k}
$$

$$
\hat{x}_t = \sum_k\alpha_{t,k}\hat{x}_t^{(k)} + \Delta x_t
$$

最後のΔxₜは任意のresidual correctionであり、全PFに共通する系統誤差を補正する。ただし、最初はΔxₜを使わず、純粋なconvex fusionで性能を確認した方がよい。residual headを早期に導入すると、PF選択・統合器が本当に機能しているか判断しにくくなる。


# 9. 推奨する実験順序

| **段階** | **モデル／処理**      | **目的**                                                                                     |
|----------|-----------------------|----------------------------------------------------------------------------------------------|
| 1        | ベースライン          | Best fixed PF、平均、共分散逆数重み、hard-selection oracle、convex-fusion oracleを算出する。 |
| 2        | 小規模モデル          | PF特徴を連結したMLP、GRU gating、TCN gatingを比較する。                                      |
| 3        | PF-token iTransformer | PF単位の履歴をtoken化し、cross-PF Attentionで重みを出す。                                    |
| 4        | 複合損失              | L_stateにsoft-oracle蒸留を追加し、selector regretとRMSEを同時に評価する。                    |
| 5        | 時間平滑化            | PFの優劣が持続する場合のみswitching penaltyを追加する。                                      |
| 6        | 残差補正              | convex-fusionとの差が残る場合に限定してresidual correctionを追加する。                       |


# 10. 結論

iTransformerは、PF間の相関、合意、不一致、補完関係がoracle選択を予測する鍵になる場合に有力である。一方、元の構造をそのまま使うのではなく、PF単位のトークン化と、PF内部の時間特徴抽出を明示的に設計する必要がある。

推奨構成は、per-PF Temporal Encoder、cross-PF iTransformer Attention、Gating/Fusion Headの組み合わせである。学習は最終状態RMSEを直接最小化し、soft-oracle蒸留を補助損失として利用する。さらに、hard-selection oracle RMSE 4を絶対的な上限と見なさず、convex-fusion oracleを算出して、統合による改善余地を切り分けるべきである。


# 参考

iTransformerの基本的な着想は、時点をトークン化する一般的な時系列Transformerとは逆に、各変数の時系列全体をトークンとして扱い、変数間相関をAttentionで学習する点にある。実装時には、元モデルの予測headをPFの信頼度・混合重みを出力するgating headへ置き換える。
