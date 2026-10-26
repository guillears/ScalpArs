"""B25-prep ML W/L — rebuild the 10-15 market-impulse / during-trade features (exact definitions of
scripts/study_ml_wl_1015_features.py:feats, reused unchanged) for the momentum-long fills the 10-15 feature file does not cover
(B20…B23 master fills + B25 provisional). Kline sources: reports/backtest_cache* + this study's top-up
reports/PREP_B25_ML_WL_fetch (study_ml_wl_b25prep_fetch.py). Output: reports/PREP_B25_ML_WL_features_new.csv.

    venv/bin/python -W ignore scripts/study_ml_wl_b25prep_features.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_wl_1015_common as C  # noqa: E402
from study_ml_wl_b25prep_data import NEWFEAT, ROOT, load_all  # noqa: E402

C.TOPUP = os.path.join(ROOT, "reports/PREP_B25_ML_WL_fetch")
import study_ml_wl_1015_features as F  # noqa: E402

D = load_all(with_features=True, use_new=False)
need = D[D.btc_r15.isna() & ~D.is_probe].copy()
need["ts_open"] = need.ts
need["entry_price"] = pd.to_numeric(need.entry_price, errors="coerce")
rows = []
for r in need.itertuples(index=False):
    f = F.feats(r)
    f.update(block="b25prep", pair=r.pair, opened_at=r.key_open, seed=0)
    rows.append(f)
out = pd.DataFrame(rows)
out.to_csv(NEWFEAT, index=False)
print(f"wrote {NEWFEAT}: {len(out)} rows")
print(out.drop(columns=["block", "pair", "opened_at", "seed"]).notna().mean().round(2).to_string())
