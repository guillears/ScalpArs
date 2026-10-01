# 🔥 FIRST-DIP rule — pre-registration (written 2026-10-01 18:23 UTC, before the unseen episodes were read)

Found while screening 49,992 hot-state entries in 1,307 episodes (pairs 0G… → roughly S; `reports/backtest_cache/hot_entry_features_SEEN_at_prereg.csv`).

**Rule (frozen):** the pair is HOT (5m ATR ≥ 2 %, RSI ≥ 70, price ≥ 3 % above the 5m EMA5, 24 h volume ≥ $20M) · the entry second is within the
first 10 minutes of the episode · the price is ≥ 3 % below its 5-minute high · LONG at that second (entries 30 s apart).
**Exits (frozen, gross):** primary +2.09 % / −1.11 % · secondary +3.09 % / −1.51 % · secondary trail 1.0 % after +1.5 % (stop 1.5 %) · 30-minute limit.
**Costs:** 0.09 % fees + 0.02 % measured slippage.

**In-sample result (exploratory — about 45 subset × exit cells were looked at, so this is NOT evidence yet):**
primary +0.204 % per trade by day [+0.060, +0.348], Jan–Apr +0.237 / May–Sep +0.180, 492 episodes on 184 days.

**Confirmation test (the only one that counts):** every episode whose 1-second data was fetched AFTER this note (pairs not in the SEEN file).
PASS = primary exit day-mean > 0 with its 95 % interval above 0, and both secondaries > 0. Anything else = not confirmed; no re-tuning of
the 3 % / 10 min / exit levels on the confirmation data.
