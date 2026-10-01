# 🔥 Hot-state entries: what separates the ones that sink from the ones that run?

4,674 first-entries of hot episodes · 391 pairs · 241 days. LOSER = fell ≥ 10 % before +0.59 %, or never reached it in 7 days: **6.6 %** (Jan–Apr 6.5 % of 1,772 · May–Sep 6.7 % of 2,902); never reached: 2.0 %.
Quintile edges are cut on Jan–Apr and applied unchanged to May–Sep.

Shuffle null: the best spread (worst − safest quintile) ANY feature shows by luck on Jan–Apr = 6.8 points (95th pct).

| Feature | loser % by quintile, Jan–Apr (low → high) | May–Sep | spread Jan–Apr / May–Sep | safest end same? | candidate |
|---|---|---|---|---|---|
| bar_range | 6 · 6 · 6 · 5 · 9 | 6 · 7 · 4 · 7 · 10 | 3 / 6 | no | — |
| ret_7d | 7 · 7 · 6 · 6 · 7 | 6 · 7 · 5 · 9 · 7 | 1 / 4 | no | — |
| rsi | 6 · 5 · 7 · 7 · 8 | 8 · 4 · 7 · 7 · 7 | 3 / 4 | no | — |
| bar_ret | 5 · 5 · 8 · 7 · 8 | 8 · 5 · 8 · 6 · 6 | 4 / 3 | no | — |
| off_24h_high | 8 · 6 · 6 · 6 · 7 | 7 · 9 · 6 · 5 · 6 | 2 / 3 | no | — |
| ret_4h | 7 · 5 · 5 · 8 · 7 | 7 · 7 · 8 · 5 · 7 | 3 / 3 | no | — |
| hours_since_24h_low | 7 · 5 · 8 · 5 · 6 | 8 · 7 · 5 · 8 · 6 | 3 / 3 | no | — |
| ret_1h | 6 · 5 · 6 · 7 · 8 | 8 · 5 · 6 · 8 · 7 | 4 / 3 | no | — |
| hot_bars_prior_7d | – · 6 · 6 · 8 · 6 | – · 7 · 5 · 7 · 8 | 3 / 3 | no | — |
| stretch | 5 · 8 · 6 · 7 · 7 | 6 · 6 · 8 · 6 · 7 | 2 / 2 | no | — |
| atr | 8 · 4 · 6 · 7 · 8 | 6 · 5 · 6 · 8 · 8 | 4 / 2 | no | — |
| vol_vs_week | 5 · 6 · 9 · 7 · 5 | 8 · 6 · 6 · 7 · 6 | 4 / 2 | no | — |
| ret_72h | 6 · 6 · 7 · 6 · 7 | 8 · 6 · 5 · 7 · 7 | 2 / 2 | no | — |
| vol_1h_share | 7 · 7 · 5 · 6 · 7 | 6 · 7 · 6 · 8 · 6 | 2 / 2 | no | — |
| btc_ret_24h | 6 · 9 · 6 · 7 · 4 | 7 · 6 · 8 · 7 · 6 | 5 / 2 | yes (Q5) | — |
| bar_vol_mult | 6 · 6 · 6 · 8 · 6 | 6 · 7 · 7 · 8 · 6 | 2 / 2 | no | — |
| up_from_24h_low | 5 · 8 · 5 · 6 · 8 | 6 · 6 · 7 · 7 · 7 | 3 / 2 | no | — |
| btc_ret_4h | 6 · 6 · 6 · 6 · 8 | 7 · 8 · 7 · 6 · 7 | 2 / 2 | no | — |
| hour_utc | 8 · 6 · 6 · 7 · 6 | 6 · 7 · 6 · 7 · 8 | 2 / 2 | no | — |
| upper_wick | 9 · 6 · 5 · 6 · 6 | 6 · 7 · 7 · 8 · 6 | 4 / 2 | no | — |
| hot_bars_prior_24h | – · 7 · 4 · 7 · 7 | – · 7 · 7 · 6 · 6 | 3 / 1 | no | — |
| ret_24h | 6 · 7 · 5 · 5 · 9 | 7 · 6 · 7 · 7 · 7 | 4 / 1 | no | — |
| vol24_usd_log | 8 · 6 · 6 · 4 · 8 | 7 · 6 · 6 · 7 · 7 | 4 / 1 | no | — |
| listed_days | 7 · 5 · 7 · 6 · 8 | 6 · 6 · – · – · 7 | 3 / 1 | no | — |

## Candidates — the safe end of each, on the UNSEEN half (May–Sep)

| Rule (edge from Jan–Apr) | May–Sep entries | days | loser % | never reached % | 10× no-stop: average per trade (of margin) | 5× |
|---|---|---|---|---|---|---|
| (no filter) | 2,902 | 149 | 6.7 | 2.0 | -2.06% | -4.39% |

A win at 10× = +5 % of the margin (+0.5 % net); a loser = the margin. Break-even loser rate: 4.8 % at 10×, 2.4 % at 5× (leverage does not change it much: lower leverage survives deeper dips, but this table's LOSER is fixed at −10 %).
