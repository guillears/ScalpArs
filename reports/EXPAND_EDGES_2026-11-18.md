# EXPAND EDGES: more trades per day from FLATBASE_SHORT and EXHAUSTION_SHORT at the same quality? (2026-11-18)

**Pre-registered:** `EXPAND_EDGES_2026-11-18_PREREG.md`, frozen 16:35 UTC before any result. Research only. Nothing in the bot changed. Nothing was committed.

## Question
The operator wants more trades per day at the same quality. The theory: the edge sits in each sleeve's CONDITIONS, and the UNIVERSE gates may be what keeps it narrow. The universe gates are the spike size, the volume multiple, the verified + shortlist selection, the $20 M floor, the time windows and the flatness cut. If so, loosening one universe gate at a time should add trades that are as good as today's.

## Plain answer
1. **No. None of the 17 single-gate relaxations passes, and neither best pair does.** The theory is refuted for both sleeves: the universe gates are part of the edge.
2. **Flatbase cannot get much bigger.** Every relaxation adds at most 0.15 trades a day, which is about 1 trade a week.
   - The added trades are worse than today's (+0.19 to +1.21 %/trade against +0.93). They lose in 2 – 4 half-years, and none has a positive Kelly at the floor.
   - The best is the $10 M floor (F6): 58 added trades in 3.4 years at +1.21. Its total over 3.4 years rises by only +48 %-points, and its max drawdown deepens from −14.7 % to −17.3 %.
3. **Exhaustion gets more trades, but each one is worse.**
   - A lower 24 h volume floor ($5 M) adds 0.75/day at +0.37 %/trade. A lower spike-volume multiple (15×) adds 0.36/day at +0.43 %/trade. Today's rule earns +0.60.
   - Both added sets lose in two half-years, lose their edge once the best 5 % of trades are removed, and sit at about 0 at 0.20 % slippage.
   - Lower spike-return cuts add trades worth about 0 (+0.06 and −0.00). Longer windows add trades that lose money (−0.16 and −0.15).
4. **Re-entering after a TP is already today's rule.** The engine keeps the episode TRACKING after a trade, so there is nothing to add. Second and later entries in an episode do as well as first entries (+0.67 vs +0.57 %/trade).
5. **Verdict: NO for every relaxation.** Keep both universes as they are. More trades at this quality need new CONDITIONS (new edges), not wider universes.

## Method
- **Flatbase:** the verified HIGH_ATR rebuild was reused.
  - It runs the bot's own `services.frenzy` / `services.flatbase` code on today's `trading_config.json`.
  - The data is the 5m store of 917 USDT perps, 2023-05 → 2026-11-17, delisted coins included.
  - The pipeline is the turnover-route proxy → NEAR_BASE 2.09 % → 4 shared slots, with TP 5 / SL 10 / 72 h, flat 0.10 % slippage and real funding.
  - Each variant patches one threshold in memory (`scripts/expand_edges_flatbase.py`).
  - **Parity: 778 trades · 73.7 % WR · +0.93 %/trade, against 778 · 73.5 % · +0.91.** The 1-trade difference comes from a 1-bar-longer S4 window, which is used in every variant, so the comparisons are like for like.
- **Exhaustion:** the EXHAUSTION_EXPAND builder was reused unchanged (patched in memory by `scripts/expand_edges_exh.py`).
  - It runs on bar-rule spike lists (return ≥ X % ∧ hourly volume ≥ Y × normal ∧ ≥ $2 M, with a new spike only after 72 h). There is no verification and no shortlist (the §0 bias rule).
  - The rule is the 15m RSI14 cross 70 → ≤ 70, vd ≥ 0.6366, spot_share_chg < −0.0066 and one position per pair.
  - The exit is today's B|tp10|S2|24, run on 5m bars, stop first.
  - Costs are taker 0.05 / maker 0.02, the live slip model at **$5k notional** on every taker leg, and real funding (a missing month is charged the builder's penalty; 0.4 – 2.2 % of trades).
  - Spot klines for 668 new pair-months were checked on data.binance.vision. 356 exist and were added; the other 312 have no spot file (no spot market that month), so they stay unscored, as live. API weight used: 0.
  - **Parity: 4,144 trades · 39.7 % · +0.60 %/trade, against EXH_EXIT_GRID 3,998 · 40.2 % · +0.65.** The bar-rule list has 3.6 % more trades than the realisable shortlist list, as in EXHAUSTION_EXPAND §0. The study used 1m bars; this one uses 5m bars.
  - **DL 582 (EXH_SSC_OIVOL) is NOT applied.** OI value history was not available for ≥ 90 % of trades, so every Exhaustion number here is pre-582.
- **Added trades** are trades in the relaxed run whose (pair, entry time) is not in the base run. **Displaced** trades are the reverse: a relaxed spike can re-anchor an episode, or slots / pair-busy can move a trade.
- **Union** = the relaxed run itself. DCR uses one size, 0.18 × equity per trade, over every calendar day from 2023-07-01 to 2026-11-17 (1,235 days).
- **Kelly** is computed at the one-sided 95 % Wilson floor with n = distinct entry days.
- **Family-wise null:** a day-level sign flip, max-T over all 17 relaxations, 2,000 permutations.

## Table 1: ADDED trades only (net %/trade; FB at flat 0.10 % slip, EXH at the $5k slip model)

| sleeve | relaxation | N | /day | WR / BE | **avg** | best-5 % out | @0.20 | Kelly floor | 23H2 | 24H1 | 24H2 | 25H1 | 25H2 | 26H1 | 26H2 | displaced | p_fw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FB | *today (base)* | 778 | 0.63 | 73.7 / 67.1 | **+0.93** | +0.71 | +0.85 | +0.09 | +0.66 | +0.89 | +0.26 | +1.91 | +0.08 | +0.68 | +2.19 | – | – |
| FB | F1 ret 4 % | 84 | 0.07 | 69.0 / 67.7 | +0.19 | −0.12 | +0.15 | −0.24 | +2.65 | +1.26 | +0.62 | −0.47 | −4.08 | −0.09 | +4.92 | 29 | 1.00 |
| FB | F2 ret 3 % | 143 | 0.12 | 69.9 / 67.3 | +0.38 | +0.10 | +0.22 | −0.13 | +2.70 | −0.51 | +2.10 | −1.69 | −2.70 | +0.99 | +2.64 | 53 | 0.94 |
| FB | F3 vol 15× | 89 | 0.07 | 69.7 / 66.3 | +0.47 | +0.20 | +0.42 | −0.16 | +1.16 | +2.35 | −1.39 | +2.08 | −0.18 | −2.51 | +2.75 | 31 | 0.97 |
| FB | F4 vol 10× | 184 | 0.15 | 69.0 / 64.8 | +0.57 | +0.31 | +0.44 | −0.06 | +1.10 | +1.84 | −0.11 | −0.58 | +0.92 | −0.48 | +0.38 | 42 | 0.88 |
| FB | F5 no verify / shortlist | 64 | 0.05 | 68.8 / 63.4 | +0.70 | +0.41 | +0.08 | −0.14 | +0.97 | +1.14 | −0.60 | +4.81 | −2.73 | +0.85 | +2.41 | 19 | 0.92 |
| FB | F6 floor $10 M | 58 | 0.05 | 74.1 / 65.1 | **+1.21** | +1.00 | +1.17 | −0.06 | +0.55 | +4.96 | −0.10 | +3.60 | +0.54 | +0.71 | +0.41 | 39 | 0.59 |
| FB | F7 floor $5 M | 66 | 0.05 | 71.2 / 63.7 | +0.96 | +0.69 | +0.91 | −0.08 | +0.55 | +4.28 | −0.10 | +1.86 | +0.90 | −0.11 | +1.84 | 45 | 0.74 |
| FB | F8 window 144 h | 8 | 0.01 | 62.5 / 69.1 | −0.97 | – | −1.01 | −1.11 | +3.43 | +5.01 | – | −2.65 | +4.95 | −5.29 | – | 5 | 1.00 |
| FB | F9 flatness 0.55 | 188 | 0.15 | 66.0 / 66.1 | −0.02 | −0.31 | −0.18 | −0.19 | −5.36 | +0.55 | −1.29 | −1.07 | −0.52 | +0.95 | +3.50 | 10 | 1.00 |
| EXH | *today (base)* | 4,144 | 3.35 | 39.7 / 35.1 | **+0.60** | +0.11 | +0.27 | +0.03 | +0.81 | +0.76 | +0.66 | +0.09 | +0.64 | +0.50 | +0.78 | – | – |
| EXH | X1 ret 4 % | 434 | 0.35 | 30.0 / 29.3 | +0.06 | −0.46 | −0.30 | −0.05 | −0.04 | +0.08 | +0.15 | +1.15 | −0.85 | +0.54 | −0.16 | 25 | 0.86 |
| EXH | X2 ret 3 % | 833 | 0.67 | 30.1 / 30.2 | −0.00 | −0.53 | −0.37 | −0.05 | −0.12 | +0.52 | −0.08 | +0.76 | −0.36 | −0.22 | −0.49 | 42 | 0.84 |
| EXH | X3 vol 15× | 450 | 0.36 | 33.3 / 29.4 | **+0.43** | −0.08 | +0.08 | −0.00 | −0.32 | +0.83 | −0.43 | +0.07 | +1.02 | +1.47 | +0.50 | 6 | 0.18 |
| EXH | X4 vol 10× | 1,186 | 0.96 | 31.2 / 29.9 | +0.14 | −0.38 | −0.22 | −0.02 | −0.18 | +0.02 | +0.19 | +0.21 | +0.47 | +0.33 | −0.02 | 12 | 0.07 |
| EXH | X5 floor $10 M | 652 | 0.53 | 36.7 / 33.9 | +0.31 | −0.20 | −0.04 | −0.02 | +0.53 | −0.17 | −1.46 | +0.41 | −0.28 | +0.56 | +0.87 | 3 | 0.98 |
| EXH | X6 floor $5 M | 924 | 0.75 | 38.3 / 34.9 | **+0.37** | −0.14 | +0.02 | −0.00 | +0.53 | −0.17 | −1.26 | +0.24 | −0.02 | +0.67 | +0.64 | 3 | 0.55 |
| EXH | X7 window 120 h | 78 | 0.06 | 29.5 / 31.0 | −0.16 | −0.71 | −0.53 | −0.14 | +1.74 | +0.52 | −0.62 | −0.82 | +1.56 | −0.21 | −3.07 | 0 | 1.00 |
| EXH | X8 window 168 h | 125 | 0.10 | 30.4 / 31.9 | −0.15 | −0.75 | −0.52 | −0.13 | +1.14 | +0.18 | −1.06 | −0.33 | +1.41 | −0.36 | −2.77 | 0 | 0.99 |
| EXH | X9 repeat after TP | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |

**X9 is already today's rule.** Of the 4,144 base trades, 1,324 are 2nd+ entries in their episode (+0.67, 40.2 % WR). After a TP: 127 at +0.66. After an SL: 1,178 at +0.70. After a time exit: 19 at −1.07.

**Pass items:** no relaxation passes. Every one fails at least the half-years, the Kelly floor and the family-wise null, and most also fail best-5 %-out and/or the 0.20 % slip test. The full item list per row is in the csv `items` column.

## Table 2: UNION with today's rule (= the relaxed run), one size 0.18 × equity

| sleeve | run | N | /day | avg | Δ trades | Δ total %-pts (3.4 y) | DCR %/day | max DD | worst day |
|---|---|---|---|---|---|---|---|---|---|
| FB | today | 778 | 0.63 | +0.93 | – | – | 0.100 | −14.7 % | −5.5 % |
| FB | F4 vol 10× | 920 | 0.74 | +0.87 | +142 | +78 | 0.111 | −15.0 % | −5.5 % |
| FB | F6 floor $10 M | 797 | 0.64 | +0.97 | +19 | +48 | 0.107 | −17.3 % | −5.5 % |
| FB | F7 floor $5 M | 799 | 0.65 | +0.96 | +21 | +43 | 0.107 | −16.5 % | −5.5 % |
| FB | F1 / F2 ret 4 / 3 % | 833 / 868 | 0.67 / 0.70 | +0.81 / +0.80 | +55 / +90 | −43 / −25 | 0.09 / 0.10 | −29.1 % | −7.3 % |
| FB | F9 flatness 0.55 | 956 | 0.77 | +0.72 | +178 | −38 | 0.093 | −17.9 % | −5.5 % |
| EXH | today | 4,144 | 3.35 | +0.60 | – | – | 0.34 | −22.8 % | −9.7 % |
| EXH | X3 vol 15× | 4,588 | 3.71 | +0.59 | +444 | +191 | 0.36 | −24.0 % | −9.7 % |
| EXH | X5 floor $10 M | 4,793 | 3.88 | +0.56 | +649 | +202 | 0.36 | −24.2 % | −12.3 % |
| EXH | X6 floor $5 M | 5,065 | 4.10 | +0.56 | +921 | +340 | 0.38 | −24.9 % | −13.3 % |
| EXH | X4 vol 10× | 5,318 | 4.30 | +0.50 | +1,174 | +173 | 0.35 | −24.2 % | −11.3 % |
| EXH | X2 ret 3 % | 4,935 | 3.99 | +0.51 | +791 | −12 | 0.33 | −28.4 % | −9.5 % |
| EXH | X7 / X8 window 120 / 168 h | 4,222 / 4,269 | 3.42 / 3.45 | +0.59 / +0.58 | +78 / +125 | −13 / −19 | 0.34 / 0.33 | −24.0 / −23.2 % | −9.7 % |

Some unions raise the DCR without deepening max DD by more than 5 points: FB F4 / F6 / F7 and EXH X3 / X5 / X6. That is what the union gate alone looks for. They still fail because the ADDED trades fail on their own: they are not consistent across half-years, depend on a few big winners, and are about 0 at 0.20 % slippage. They also lift the worst day (EXH X6: −9.7 % → −13.3 %).

## Best pair per sleeve (descriptive: fewer than 2 relaxations passed, so this is NOT a pass)

| pair | added N | /day | avg | 23H2 · 24H1 · 24H2 · 25H1 · 25H2 · 26H1 · 26H2 | best-5 % out | @0.20 | Kelly floor | union DCR (today) | union max DD (today) | worst day |
|---|---|---|---|---|---|---|---|---|---|---|
| FB F6 floor $10 M + F5 no verify | 109 | 0.09 | +0.51 | +0.91 · +1.00 · −0.81 · +3.97 · −1.10 · +0.36 · +0.87 | +0.24 | +0.32 | −0.11 | 0.11 (0.10) | −18.9 % (−14.7) | −5.5 % |
| EXH X3 vol 15× + X6 floor $5 M | 1,462 | 1.18 | +0.37 | −0.34 · +0.65 · −0.64 · +0.29 · +0.15 · +0.73 · +0.61 | −0.14 | +0.02 | +0.01 | 0.40 (0.34) | −25.2 % (−22.8) | −13.3 % (−9.7) |

- **How the pairs were picked.** Flatbase's two highest-avg relaxations (F6, F7) are the same gate, so the next distinct gate (F5) was used instead.
- **The Exhaustion pair looks the most interesting.** It adds +1.2 trades/day at +0.37 and the union DCR rises 0.34 → 0.40 %/day.
- **But the added trades fail the pre-registered bar:**
  - 2 negative half-years;
  - best-5 %-out −0.14;
  - about 0 at 0.20 % slippage (+0.02);
  - a Kelly floor of about 0.
- **It is also post-hoc.** Apply the 30–50 % haircut on top.

## Verdict
| sleeve | relaxation | verdict | config |
|---|---|---|---|
| FLATBASE | all (F1–F9, pair) | **NO** | keep `frenzy_spike_ret_pct` 5 · `frenzy_spike_vol_mult` 20 · `flatbase_min_vol_usd` 20 M · `flatbase_max_hours_after_spike` 96 · `flatbase_flat_max` 0.444 |
| EXHAUSTION | all (X1–X8, pair) | **NO** | keep `exhaustion_min_vol_usd` 20 M · `exhaustion_max_hours_after_spike` 72 · the shared spike rule 5 % / 20× |
| EXHAUSTION | X9 re-entry | already in today's rule | none |

None of the relaxations even qualifies for OBSERVE: the pre-registration gives OBSERVE only to a relaxation that fails nothing but the null. The closest is X3 + X6 (Exhaustion, spike volume 15× and a $5 M floor). If the operator wants it tracked anyway, it could be a zero-cost stamp line: the scout tags the signals that pass the relaxed gates but not today's. It must not be armed.

## Assumptions and limits
1. **DL 582 is not applied** to Exhaustion. The OI value history is not available, so Exhaustion numbers are pre-582. DL 582 removed about 16 % of trades in its own study.
2. **Thin books (F6 / F7 / X5 / X6).** There is no order-book history, so the live depth guard cannot be rebuilt.
   - The Q5-based slip model at $5k notional is the proxy (EXH). Flatbase runs at flat 0.10 % plus a 0.20 % stress.
   - The live guard would refuse some thin-book entries. That effect is unknown in sign.
3. **Holdouts are not clean.** 2023-H2 and 2026-H2 were both read in earlier studies of these sleeves. 2023-H1 could not be built for any relaxation: the Flatbase store starts 2023-05, and the Exhaustion E data holds today's episodes only.
4. **Exhaustion uses the bar-rule spike list, not the live verified + shortlist flag.** That list has +3.6 % trades vs the realisable list, at a slightly lower average.
5. **Funding.** A missing funding month is charged a conservative penalty (0.4 – 2.2 % of trades).
6. **Exits run on 5m bars** (the 1m replica matched 4,138 of 4,140 exit kinds in EXH_EXIT_GRID).

## Next step
Stop widening universes. To get more trades at this quality, look for new conditions that create the same state:
- other "pump without follow-through" shapes;
- the post-hoc extreme-ATR RSI7 short from HIGH_ATR §6, which still needs a fresh pre-registration;
- the "state-only" entry proposed in EXHAUSTION_EXPAND.

Each needs its own pre-registered study.

## Files
- **Report:** `reports/EXPAND_EDGES_2026-11-18.md`, `_PREREG.md`, `.csv` (every row: base / added / union, all columns + pass items), `_frontier.png`.
- **Scripts:**
  - `scripts/expand_edges_flatbase.py`
  - `scripts/expand_edges_exh.py`
  - `scripts/expand_edges_eval_fb.py`
  - `scripts/expand_edges_eval_exh.py`
  - `scripts/expand_edges_stats.py`
  - `scripts/expand_edges_report.py`
- **Data:** `reports/EXPAND_EDGES_2026-11-18_data/` (fb sims, exh spike lists + rows, extra spot months, eval pickles, report.txt).
