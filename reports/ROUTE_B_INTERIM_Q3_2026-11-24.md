# 📡 Route B — INTERIM Q3 look at the new-sleeve triggers (2026-11-24) — **NOT A VERDICT**

Operator-ordered interim (PREREG Addendum 6). Data: recording start 2026-11-19 23:23 → 2026-11-24 20:00 UTC (paths need 4 h, so signals to 11-24 16:00) · 4.7 days · 1993 final triggers with a recorded book · 8358 priced random-minute controls. Run: `scripts/route_b_signals/interim_q3.py`.

Luck bar (Addendum 7: family max of Δ over the H1/H2/H4/H6/H7 keys and variants that reach ≥ 20 trades on ≥ 5 days, eligible openers = with ≥ 1 final control, × 16 cells, 95th pct of 1,000 shuffles, keys shuffled independently): **– pp** — **no key reaches the floor this run, so nothing is luck-tested**. Keys below the floor show '–' (not tested). Halves = Δ in the two halves of the interim's matched days (the frozen H-a / H-b split → 11-28 12:00 needs data past it). Top pair · day share = the largest group's share of the SIGNED net sum ('–' when the sum ≤ 0); B2 fails at ≥ 50 %.

| key | trades/day | N (with controls) · days with controls · hours | cell | WR | avg net % (hour CI) | random same coin | Δ vs random (hour CI) | luck p | halves | Kelly@floor | top pair · day share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H5_taker_sell_climax | 107.4 | 504 (420) · 4 · 101 | tp2_sl2 | 41 % | -0.66 (-0.95 … -0.36) | -0.45 | **-0.30** (-0.58 … +0.01) | 0.999 | -0.45 / -0.15 | -0.47 | – % · – % |
| H5_taker_buy_climax | 70.8 | 332 (283) · 4 · 99 | tp2_sl2 | 47 % | -0.49 (-0.76 … -0.23) | -0.58 | **+0.05** (-0.22 … +0.33) | 0.292 | +0.01 / +0.08 | -0.41 | – % · – % |
| H4_bid_vacuum | 65.0 | 305 (271) · 4 · 99 | tp1_sl5 | 75 % | -0.26 (-0.59 … +0.03) | -0.23 | **-0.05** (-0.42 … +0.28) | – | +0.04 / -0.16 | -0.57 | – % · – % |
| H4_ask_vacuum | 55.6 | 261 (215) · 4 · 95 | tp5_sl5 | 49 % | +0.17 (-0.33 … +0.69) | -0.69 | **+1.04** (+0.58 … +1.50) | – | +1.20 / +0.81 | -0.03 | 65 % · 279 % |
| H3_imb_bid_heavy_thin | 32.6 | 153 (128) · 4 · 83 | tp2_sl2 | 50 % | -0.08 (-0.43 … +0.25) | -0.15 | **+0.05** (-0.34 … +0.40) | 0.411 | -0.27 / +0.39 | -0.18 | – % · – % |
| H3_imb_ask_heavy_thin | 29.8 | 140 (115) · 4 · 78 | tp2_sl2 | 45 % | -0.44 (-0.81 … -0.07) | -0.48 | **+0.17** (-0.23 … +0.57) | 0.129 | -0.00 / +0.37 | -0.45 | – % · – % |
| H1_liq_long_cascade | 19.2 | 90 (75) · 4 · 28 | tp1_sl5 | 70 % | -0.39 (-1.27 … +0.47) | -0.72 | **+0.50** (-0.34 … +1.21) | – | +0.60 / -0.12 | -0.85 | – % · – % |
| H2_liq_short_squeeze | 14.7 | 69 (62) · 4 · 32 | tp3_sl5 | 52 % | -0.88 (-1.99 … +0.16) | -0.46 | **-0.69** (-1.72 … +0.55) | – | -0.32 / -1.21 | -0.67 | – % · – % |
| H7a_oi_flush_dump | 9.8 | 27 (16) · 3 · 22 | tp5_sl3 | 56 % | +0.59 (-1.05 … +2.18) | +0.21 | **+1.39** (-0.25 … +2.92) | – | +0.00 / +1.49 | -0.15 | 37 % · 111 % |
| H4_bid_vacuum|c | 6.5 | 18 (16) · 3 · 18 | tp1_sl1 | 61 % | -0.32 (-1.25 … +0.44) | -0.33 | **-0.13** (-1.07 … +0.63) | – | +1.78 / -0.25 | -1.08 | – % · – % |
| H6_funding_le_-0.5pct | 4.9 | 23 (15) · 4 · 21 | tp5_sl1 | 22 % | -0.46 (-1.92 … +1.06) | +0.42 | **+0.31** (-1.40 … +2.18) | – | -0.10 / +0.77 | -0.24 | – % · – % |
| H4_ask_vacuum|c | 4.7 | 13 (9) · 2 · 13 | tp5_sl3 | 54 % | +1.41 (-0.40 … +3.19) | -1.47 | **+3.73** (+1.43 … +5.99) | – | +6.11 / +1.83 | +0.02 | 47 % · 89 % |
| H7c_oi_build_pump | 4.7 | 13 (12) · 2 · 11 | tp3_sl5 | 69 % | +0.37 (-1.55 … +2.12) | -1.23 | **+1.40** (-0.20 … +2.92) | – | -0.14 / +1.71 | -0.50 | 59 % · 135 % |
| H6_funding_le_-0.5pct|c | 3.6 | 10 (7) · 3 · 7 | tp5_sl1 | 20 % | +0.13 (-1.34 … +1.55) | -0.31 | **+1.00** (-1.25 … +3.58) | – | +1.21 / +0.83 | -0.14 | 369 % · 195 % |
| H7b_oi_flush_pump | 3.3 | 9 (6) · 2 · 8 | tp2_sl5 | 67 % | -0.63 (-2.54 … +0.97) | -0.34 | **-0.29** (-2.67 … +2.95) | – | +0.64 / -1.22 | -1.43 | – % · – % |
| H2_liq_short_squeeze|c | 2.2 | 6 (6) · 2 · 5 | tp3_sl5 | 67 % | +0.13 (-2.58 … +2.85) | -0.98 | **+1.11** (-1.59 … +4.12) | – | +0.30 / +1.51 | -0.87 | 754 % · 416 % |
| H6_funding_ge_+0.1pct | 1.9 | 9 (9) · 4 · 8 | tp3_sl2 | 78 % | +1.06 (-0.36 … +2.52) | -0.69 | **+1.75** (+0.32 … +3.28) | – | +3.08 / +0.08 | -0.10 | 30 % · 63 % |
| H7a_oi_flush_dump|c | 1.5 | 4 (2) · 1 · 3 | tp3_sl1 | 25 % | -0.40 (-1.29 … +0.47) | -0.01 | **+0.49** (– … –) | – | – / – | -0.43 | – % · – % |
| H1_liq_long_cascade|c | 1.1 | 3 (3) · 2 · 3 | tp5_sl1 | 33 % | +0.61 (-1.69 … +4.83) | +0.50 | **+0.10** (-2.47 … +4.94) | – | -2.47 / +1.39 | -0.21 | 265 % · 173 % |
| H6_funding_ge_+0.1pct|c | 1.1 | 3 (3) · 2 · 3 | tp1_sl2 | 100 % | +0.79 (+0.76 … +0.81) | -1.31 | **+2.10** (+1.31 … +2.98) | – | +2.98 / +1.66 | – | 68 % · 66 % |
| H7b_oi_flush_pump|c | 0.4 | 1 (1) · 1 · 1 | tp5_sl5 | 100 % | +4.84 (– … –) | +0.77 | **+4.07** (– … –) | – | – / – | – | 100 % · 100 % |

## Data completeness per UTC day

| day | openers | no book | book unwalkable | path final | hours with triggers | control stubs | controls final | missing S3 hours |
|---|---|---|---|---|---|---|---|---|
| 2026-11-19 | 8 | 50 % | 12 % | 100 % | 1 | 8 | – % | – |
| 2026-11-20 | 389 | 7 % | 44 % | 100 % | 24 | 35 | 100 % | 0 |
| 2026-11-21 | 559 | 8 % | 50 % | 100 % | 24 | 68 | 100 % | 0 |
| 2026-11-22 | 373 | 7 % | 50 % | 100 % | 24 | 25 | 100 % | 0 |
| 2026-11-23 | 579 | 10 % | 55 % | 100 % | 24 | 92 | 100 % | 0 |
| 2026-11-24 | 323 | 6 % | 47 % | 89 % | 20 | 25 | 0 % | 0 |

## Read (interim, 4.7 days, one BTC regime — nothing here can arm, propose or retire)
- **Data: complete.** 0 missing recording hours; paths final except the last hours of 11-24; controls final except 11-24 (day not
  closed — so every key has ≤ 4 days with controls). ~8 % of triggers had no recorded book (excluded) and **~50 % had a 20-level book
  too shallow to walk $6k** (default 0.10 % slip → their cost is a LOWER bound; thin books are exactly where these triggers fire).
- **Luck test (Addendum 7, counted on openers with a final control): no key reaches ≥ 20 on ≥ 5 days yet → nothing is luck-tested
  this run.** The first eligible run is after 11-24's controls finalise (next daily refresh). The H3 / H5 p-values are their own
  shuffle (HOLDOUT-only keys) and none is below 0.10.
- **H5 (the most trades) and H3 — nothing.** Both lose in absolute terms (−0.08 … −0.66 % per trade) and do not beat random minutes.
- **H4 bid vacuum, H2 squeeze — nothing** (Δ ≤ 0).
- **H4 ask vacuum (LONG, ~56 / day) — the only signal-like row, and it fails concentration:** Δ +1.04 pp vs random same-coin minutes
  (hour CI +0.58 … +1.50), positive in both interim halves (+1.20 / +0.81) and without unwalkable-book rows. But on its own +0.17 %
  per trade (CI −0.33 … +0.69, Kelly < 0) — it "wins" because random longs on those coins lost −0.69. **One coin (BR) carries 65 % of
  its net sum and 11-21 carries 279 %** (the other days net negative): B2's ≥ 50 % rule fails. 70 % of its rows had the shallow book.
