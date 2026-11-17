# 過学習対策方針 — selector/scale/beam/hold の再設計とマジックナンバー方針

> **目的**: Public LB(隠しtest~200坑井の一部)に手で合わせ込んだ定数を排し、**Private でも剥がれない**構成にする。
> 軸: **CV/train-robust で決めたもの＝安全 / Public LB を叩いて決めたもの＝危険**。
> 対象: 7.191系フルパイプライン(`rogii-lb-7-191-copy-n-upgrade-b31810.ipynb`) と v25 の selector 部。
> 最終更新: 2026-07-03

## 背景 / 問題意識
- 現行 selector(7.191由来)の `scale/beam/hold` は **`sp45 patch`, `sel15 vb best`** 等の命名＝**LB/手で反復調整した痕跡**。
  6bin × (scale/beam/hold) ＝実質十数DOF を小さな Public 集合に合わせ込むと **Private でシェイクする**。
- 一方 **PP(OOF Nelder-Mead)・w_struct(CV)・grid平滑(決定論)** は転移設計＝安全。
- → **過適合の主犯候補 = selector の per-bin 値 と gold層のマジックナンバー**。ここを作り直す/落とす。

---

## 1. selector: 「trainに最適化されすぎてない PF を複数用意して調整」

### 方針
- **bin の分け方はこのままでよい**:
  `code = int(n_eval > 4840) + 2*searchsorted((136.73, 185.51), z_span)`（6分割）。
- **`scale / beam / hold` は疑問 → 再導出する**。今の値は特定PF/LBに合わせ込まれてる恐れ。

### やり方（★ここが肝）
1. **train に過適合していない PF パラメータを ~10個用意する**。
   - 例: v19バンク(r0〜)、seed違い、軽めに調整した頑健なもの。**1個の過学習PFに依存しない**ことが目的。
   - 「1本の最強PF」ではなく「**そこそこ動く多様なPF群**」を使う(未適応PFで見る＝test regimeの代理, [[analog-competitions-research]]/転移診断の思想)。
2. **その全PFで train を見て、どのPFでも総合的に良い scale/beam/hold を選ぶ**。
   - ＝「特定PFに最適な値」ではなく「**PF選択にロバストな値**」。これが過適合を避ける核心。
   - 各 bin で `scale/beam/hold` をスイープ → **10個のPF全部での overall(全行プールRMSE)の平均(or 最悪ケース)が最小**になる値を採用。
   - 評価は **GroupKFold by well の OOF**（grouped-CV-by-well, [[analog-competitions-research]] #5）。in-sample は過学習するので不可。
3. **少数精鋭・粗いスイープ**にする（scaleは数点、beam/holdは 0/0.05/0.1/0.15/0.2 程度）。細かくしない＝DOFを増やさない。

### 実装タスク（コードを書く）
- [ ] **robust PF bank(~10)** を定義（過学習してない多様なPF）。
- [ ] `tune_selector_robust.py`: 各PF × 各bin で `scale/beam/hold` をスイープし、**全PF横断で総合最良**の値を bin ごとに1組選ぶ。
      目的関数 = 10PFの overall の**平均(まず)／最悪ケース(より保守)**、評価 = GroupKFold by well。
- [ ] 出力 = 新 `SELECTOR_BIN_VARIANTS`（PF非依存でロバストな scale/beam/hold）。
- [ ] 現行(vb/patch)値との差と、各PFでの改善分布を記録。

### 検証観点
- **PFを差し替えても scale/beam/hold の最適が動かない** ＝ ロバスト＝転移する見込み。
- 特定PFでだけ効く値は**採らない**（＝そのPFへの過適合）。

---

## 2. gold層 / マジックナンバー: 論外。ただし条件付き採用

### 原則
- **gold層(cell40)の手設定定数群(3プロファイル×各10個超: `base/gain_scale/cap/clip_*/delta_soft/p95_hard`…)は原則使わない**。隠しDOFの塊＝Public過適合源。
- 同様に **`sp45=4.5`, projection `deg4/blend0.75`** 等の"patch"定数も疑う。

### 例外（採否の判定基準）
- **train に適用してみて実際にうまくいったものは使ってよい**。
  - **train(できれば OOF/GroupKFold by well)で精度向上を確認** → **向上すれば採用可、向上しなければ使えない**（＝Publicにだけ効いてる過適合と判断）。
- **中身で何をやっているかは必ず確認した上で**採否を決める（ブラックボックスで採用しない）。
  - 参考: gold = 可視prefixバックテストで「候補がdefaultトラッカーに勝つ」坑井だけ、適応的α(`α×ramp×diff` を max_move で clip)で動かす層。Public LBラベルは直接使ってない(ローカルCV的)が、profile選択やマジックナンバー自体がLB合わせなら過適合。

### 実装タスク
- [ ] gold等の各機構を **単体で train-OOF に適用 → overall が下がるか**を測る。下がらなければ捨てる。
- [ ] 採用するものだけ、その効果を記録（何坑井で・どれだけ効いたか）。

---

## 3. blend重み: 直感でよい。細かくしない

- ridge/selector の `0.3/0.7`、projection の `0.25/0.75` 等の **blend重みは直感で決めてOK**。
- **細かく最適化しない**（細かい最適化＝DOF増＝過適合源）。粗い値(0.5, 0.3/0.7 程度)に留める。
- LB を叩いて 0.01 刻みで詰める、はやらない（±0.2 seed床の中で意味がない上に過適合）。

---

## まとめ（一言）
- **selector**: bin分けは維持。**scale/beam/hold は "過学習してない ~10 PF 全部で総合的に良い値" に再導出**（PF選択にロバスト＝転移する）。GroupKFold by well。
- **マジックナンバー(gold等)**: 原則不採用。**train-OOFで効果実証できたものだけ採用**。中身は必ず確認。
- **blend重み**: 直感・粗く固定。細かくしない。

関連: [scores.md](scores.md), [analog_competitions.md](analog_competitions.md)(grouped-CV-by-well/no-naive-average), [[v19-pf-bank-boosting]](PFバンク/選択がボトルネック), [[hard-wells-priority]], `rogii-lb-7-191-copy-n-upgrade-b31810.ipynb`(selector/gold の出所)。
