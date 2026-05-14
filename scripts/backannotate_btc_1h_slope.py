"""Back-annotate `entry_btc_1h_slope` on existing CLOSED Order rows.

Run once after the May 14 deploy to populate the new dimension on historical
trades. Operates on the DB directly. Idempotent — only updates rows where
the field is currently NULL.

Strategy
--------
1. Fetch BTC 1h OHLCV from Binance for the last ~30 days (or longer if needed).
2. Compute EMA20 + 3-candle slope on the full series.
3. For each Order with NULL `entry_btc_1h_slope`, find the most recent 1h
   candle that closed before the trade's `opened_at`, and assign that
   candle's slope value.

Caveats
-------
- Binance API gives you the LAST N 1h candles. If trades are older than the
  fetched window, they stay NULL. Adjust LIMIT_CANDLES below to widen.
- 1h slope changes slowly so single-candle lookup is fine (no need for
  intra-bar interpolation).

Usage
-----
    python3 scripts/backannotate_btc_1h_slope.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt
import pandas as pd
from datetime import datetime, timezone

from database import SessionLocal
from models import Order

LIMIT_CANDLES = 1000  # Binance allows up to 1500. ~41 days at 1h.


def main():
    print(f"Fetching BTC 1h OHLCV ({LIMIT_CANDLES} candles)...")
    ex = ccxt.binance({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    ohlcv = ex.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=LIMIT_CANDLES)
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    print(f"Fetched {len(df)} candles, range {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")

    # EMA20 + slope vs 3 candles back, matching what the live engine computes
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema20_prev3'] = df['ema20'].shift(3)
    df['slope_pct'] = (df['ema20'] - df['ema20_prev3']) / df['ema20_prev3'] * 100
    df = df.dropna(subset=['slope_pct']).reset_index(drop=True)

    earliest_candle = df['ts'].iloc[0]
    print(f"Slope series starts at {earliest_candle}")

    db = SessionLocal()
    try:
        # Pull all CLOSED orders with NULL 1h slope
        orders = db.query(Order).filter(
            Order.status == 'CLOSED',
            Order.entry_btc_1h_slope.is_(None),
        ).all()
        print(f"Found {len(orders)} CLOSED orders needing back-annotation.")

        updated = 0
        too_old = 0
        no_open_ts = 0
        for o in orders:
            if not o.opened_at:
                no_open_ts += 1
                continue
            # Convert opened_at to UTC-aware datetime to match df['ts']
            opened = o.opened_at
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            if opened < earliest_candle:
                too_old += 1
                continue
            # Find the most recent candle ts <= opened
            mask = df['ts'] <= opened
            if not mask.any():
                too_old += 1
                continue
            slope = float(df.loc[mask, 'slope_pct'].iloc[-1])
            o.entry_btc_1h_slope = round(slope, 4)
            updated += 1

        db.commit()
        print(f"Updated: {updated}")
        print(f"Skipped (older than fetched window): {too_old}")
        print(f"Skipped (no opened_at): {no_open_ts}")
    finally:
        db.close()


if __name__ == '__main__':
    main()
