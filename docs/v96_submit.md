# v96 提出パイプライン（CANDFEAT 3脚 blend, CV 5.9416）

> 作成 2026-07-31。持ち出しは **artifact フォルダ（モデル）のみ** で完結。
> ★v96 の中身は v95 パイプライン + **CANDFEAT L5**（候補別信頼性を各候補トークンに付与）。

## ★提出物 = Kaggle ノートブック `v95/submit_v96.ipynb`
- **全コードを `%%writefile` でノートブック内に直書き**（Dataset はモデルのみ）。ユーザー指示: Kaggle実行 / 全コード直書き / 持ち出しは artifact のみ。
- 生成器 `v95/build_v96_notebook.py`（アセンブラ）。v95 実モジュール9本を writefile セル化 + Kaggle 環境セル + 実行セル。
  - writefile 9本 = gen_grfree_anchor / gen_grfree_test_anchor_v95 / pf_banks_v95 / imputers_v95 / features_v95 / create_v95 / build_seqfeat_v95b / submit_v95 / submit_v96（各 env 駆動パス、ローカル値 fallback）。
  - **v93 提出ノート `v93/submit_v93.ipynb` が構造の元**（Kaggleパス検出 + 全生成直書き + 実行）。v96 は v95 生成 + L5cf + 3脚に差し替え。
- **錨は実行時に再生成**（`gen_grfree_anchor` が train 上で場+15GRU 学習 → 隠しtest 予測。ckpt不要）。`FORCE_ANCHOR=1` で dataset 内の可視3井錨を必ず上書き。
- Kaggle の read-only /kaggle/input 問題 → 起動時にモデルを `/kaggle/working/art95(,b)` へコピーし、生成物(parquet/錨/seqfeat)の書込先を確保。
- 出力: `/kaggle/working/submission.csv`。想定ランタイム: 錨GRU再学習 + create(200井) + 推論 = 数時間（v93ノート実績あり）。
- **Dataset レイアウト（最小・数GB）**: `artifacts_v95/`(config3 + models_l1_fwd 45 + nn 15) + `artifacts_v95b/`(l5_cf_ckpt 15 + submit_v96_config.json)。train.parquet 等の大生成物は不要。
- ローカル検証用の同一ロジック = `v95/submit_v96.py`（env 無し→ローカルパス）。

## 検証(ノートブック)
- writefile 9セルが元ソースと **バイト完全一致**（9/9, `build_v96_notebook.py` 実行時に自己確認可）。
- env 駆動パス解決 OK（Kaggle風 env で別ディレクトリを指すことを確認）。
- 可視test3井の end-to-end スモークは `submit_v96.py` で進行（create が重く低速、tune_v94 とCPU競合）。

## 何を提出するか
- **3脚 blend = GBM + TCN + L5cf（CANDFEAT）**。固定重み（転移安全）。
- **CV(OOF) = 5.9416**（v95 の CANDFEAT無し 3脚 6.0253 から −0.084）。
- snap 物理脚（追加 −0.027 → 5.9149）は **別建て**（下記「snap の扱い」）。

## blend 重み（固定・nnls-on-OOF で確定, 実測再現済み）
| 脚 | 単体 OOF | 重み |
|---|---|---|
| GBM（9variant×5fold, artifacts_v95） | 6.6529 | 0.0474 |
| TCN（mdn2 3seed×5fold, artifacts_v95） | 6.1341 | 0.5094 |
| **L5cf（CANDFEAT, 3seed×5fold, artifacts_v95b）** | **6.1942** | 0.4432 |
| **3脚 blend** | | **5.9416** |

`tvt = last_known_tvt + (w_gbm·gbm + w_tcn·tcn + w_l5cf·l5cf)`

## L5cf（CANDFEAT）の仕様（★提出の要）
- L5 = 曲線トークン cross-attention ランカー（K=20 軌跡の凸結合 + 小残差）。
- **CANDFEAT**: 各候補トークンの符号化(enc)入力に、その候補の**信頼性2特徴**を追加:
  - `loglik`（GR適合度） / `vs_tw`（typewell からの乖離）。= nF=2 役割。
  - self/nbr/nbr_gr5/self_graft の4表現の候補に付与。ancc候補は 0埋め（役割スロット分離）。
  - ユーザー洞察「推論値だけでは足りない・候補の信頼性を入れるべき」の実装。3役割以上は過学習/seed noise で悪化 → **2役割が sweet spot**。
- enc = `Conv1d(2+nF, d, 9)`（nF=2）。qry = `Linear(CDIM+4, d)`（CDIM=3）。

### test seqfeat 生成（build_seqfeat_v95b, env）
```
SF_IN=<test.parquet>  SF_OUT=seqfeat_test_cf.pkl  SF_NO_TARGET=1
CANDFEAT=1  NF_MAX=2  V93_INPUT=fwd
EXTRA_DELTA=pf_nbr_gr5_delta_1,pf_nbr_gr5_delta_2,pf_nbr_gr5_delta_3,pf_self_graft_delta_1,pf_self_graft_delta_2,pf_self_graft_delta_3
EXTRA_STD=pf_nbr_gr5_std_1,pf_nbr_gr5_std_2,pf_nbr_gr5_std_3,pf_self_graft_std_1,pf_self_graft_std_2,pf_self_graft_std_3
```
- K=20（BASE14 + nbr_gr5/self_graft delta 6ch）。**EXTRA を欠くと K=14 で ckpt と不一致**（致命）。
- z正規化は parquet 毎（train/test は同一母集団のランダム分割・行数巨大 → 列統計はほぼ一致, 実害 <0.01 std）。

## 検証（全て実測で確認済み, 2026-07-31）
1. **L5 推論機構**: 既存 seqfeat_v95_cf.pkl + l5_s0_f0_model.pt で fold0 val 予測を再現 → 保存 valpred と **max|diff|=0**。
2. **seqfeat ビルドパス**: 上記 env で train から再ビルド → 既存 seqfeat_v95_cf.pkl と D/S/C/DX **max|diff|=0**（EXTRA順序・CANDFEAT・z-norm 全確定）。
3. **blend 重み**: gbm/tcn/l5cf の OOF に nnls → 重み[0.0474,0.5094,0.4432]・**RMSE 5.9416**（config固定重みでも同値）。
4. モデル保存形式 = dict `{"state":..., "K":20, "arch":"l5", "dim":64}`（bare state_dict でない）→ ローダで `ck["state"]` 対応済み。

## 実行
```
.venv/Scripts/python.exe v95/submit_v96.py
  SKIP_GEN=1       # test.parquet 生成をスキップ(既存使用)
  SKIP_SEQFEAT=1   # L5cf test seqfeat 生成をスキップ(既存使用)
```
段取り: run_gen(GRフリー錨→create SPLIT=test) → GBM推論 → TCN推論 → L5cf(seqfeat生成→15ckpt推論) → 3脚blend → `artifacts_v95b/submission_v96.csv`。

## 持ち出す artifact（これだけで完結）
### artifacts_v95/（GBM/TCN/生成）
- `models_l1_fwd/{lightgbm-1..3,catboost-1..3,xgboost-1..3}/fold{0..4}.pkl`（GBM 45個）
- `nn/combo_mdn2_s{0,1,2}_fold{0..4}.pt`（TCN 15個）
- `grfree_anchor_test.pkl`（GRフリー test 錨）
- `features_l1.json`（GBM 719特徴）/ `submit_pack_nn.json`（TCN 前処理）
- `data/test.parquet`（実行時 create で生成。列に PF候補=loglik/vs_tw/delta を含む）
### artifacts_v95b/（L5cf）
- `l5_cf_ckpt/l5_s{0,1,2}_f{0..4}_model.pt`（L5cf 15個）
- `submit_v96_config.json`（blend重み・seqfeat env）
### コード（同梱必須）
- `v95/submit_v96.py`, `v95/submit_v95.py`（GBM/TCN/run_gen 再利用元）
- `v95/build_seqfeat_v95b.py`（L5cf seqfeat）, `v95/create_v95.py`（test.parquet 生成）
- `v95/gen_grfree_test_anchor_v95.py`（錨。既存 pkl があれば不要）

## snap 物理脚の扱い（5.9416 → 5.9149 の −0.027）
- snap = v91 物理候補（prefix-dip, N=3000, K4, GRフリー, 3seed）を **pf_center_v89（r0）に最近傍snap** → 多seed平均 → 固定 weight 0.08 で blend。ρ0.61 で脱相関・転移安全。
- **障壁**: r0 = `pf_center_v89`（v89 PF forward-ensemble）を **test 200井で走らせる必要**。`build_pf_center_v89.py` は train 固定（`base.list_wells()`）で test 非対応。
  - v89 PF エンジンを test で回す＝T4 で長い隠しtest井の VRAM OOM リスク（[[kaggle-submit-pitfalls]]）。「artifact のみ」から外れコード依存が増える。
- **判断待ち**: +(-0.027) のため v89 PF-on-test 依存を足すか。3脚(5.9416)で提出し snap は次段が無難。
