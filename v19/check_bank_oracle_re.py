"""_re バンク(artifacts_v19_32seeds_re/pf_bank.json)を全PF横断で確認するラッパ。
check_bank_oracle.py を _re バンク・32seeds に向けるだけ(PFコア/ロジックは同一)。
出力: 各PF単体 overall / oracle(各wellで最良PF) / 閾値カバレッジ / 貪欲追加曲線(相補性=何本で頭打ちか)。
実行: rogii_claude/.venv/Scripts/python.exe v19/check_bank_oracle_re.py
"""
import check_bank_oracle as cbo
import iterative_pf_boost_v19 as pf

_P = pf.PROJECT / "v19" / "artifacts_v19_32seeds_re"
cbo.BANK_PATH  = _P / "pf_bank.json"
cbo.CACHE_PATH = _P / "bank_eval_cache.pkl"   # PFごと per-well RMSE を再開キャッシュ
cbo.N_SEEDS_EVAL = 32                          # _re は 32seeds regime
cbo.CHUNK        = 120                         # OOM回避(32seed)
cbo.INCLUDE_V9   = True                        # v9 pf_1/2/3 も加えて相補性比較(_reバンクのみ見たいなら False)

if __name__ == "__main__":
    print(f"bank: {cbo.BANK_PATH}  seeds={cbo.N_SEEDS_EVAL}  +v9={cbo.INCLUDE_V9}")
    cbo.main()
