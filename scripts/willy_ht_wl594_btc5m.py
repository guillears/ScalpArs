"""WILLY_HT W/L 594 — BTC 5m klines 2024-12-20 → now (cache reports/WILLY_HT_WL594_data/btc_5m.csv) so the master's strongest
nominal split (entry_btc_trend_gap_pct = BTC 5m EMA13 − EMA50 gap %) can be cross-checked on the rebuild trades. Research only.
≤ 1,200 weight/min (limit 1000 = weight 5; reads X-MBX-USED-WEIGHT-1M)."""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from willy_ht_wl594_build import get, DATA   # noqa: E402

fn = os.path.join(DATA, 'btc_5m.csv')
s = pd.Timestamp('2024-12-20').value // 1_000_000
if os.path.exists(fn):
    old = pd.read_csv(fn); s = int(old.t.max()) + 300_000
else:
    old = None
end = int(time.time() * 1000)
rows = []
while s < end:
    k = get(f'https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={s}&limit=1000')
    if not k:
        break
    rows += [[int(x[0]), float(x[4])] for x in k]
    s = k[-1][0] + 300_000
    time.sleep(0.4)
d = pd.DataFrame(rows, columns=['t', 'c'])
if old is not None:
    d = pd.concat([old, d])
d.drop_duplicates('t').sort_values('t').to_csv(fn, index=False)
print(len(d))
