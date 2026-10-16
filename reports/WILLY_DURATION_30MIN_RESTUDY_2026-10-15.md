# WILLY: does dropping the take-profit after 30 minutes make sense? Duration re-study (2026-10-15)

Research only. No engine, config, template or test was changed. Nothing was committed. Cache-only: no Binance call was made.
The late-exit grid was pre-registered (reports/WILLY_30MIN_RESTUDY_PREREG.txt, frozen 14:10:47Z) before any cell was priced.
This study has not been reviewed, so it supports no arm or ship.

## Plain-English summary

**Verdict: keep today's WILLY exit (+1.25 / −4 / 90 min). No late rule is a candidate.** 40 late rules were tested (5 times × 8
actions). None passes under real trade prices, and none passes under both price rulers. The live 30-minute Scout cells keep their own
frozen live gate, unchanged.

**What the trade durations say (the operator's question)**
1. **Winners are fast.** Half of all take-profits hit within 5 minutes, about 70 % within 10 minutes and about 88 % within 30 minutes.
   This holds in every period (2025, 2026-H1, 2026-H2, master).
2. **Losers are slower, but not by much.** The median stop comes at 12–22 minutes. About a quarter of all stops happen after 30 minutes.
3. **About 1 trade in 5 is still open at minute 30.** Of those, roughly 4 in 10 still reach +1.25, 2–3 in 10 get stopped, and 3–4 in 10
   run to the 90-minute close (average −1.1 to −1.4 %).
4. **The key number: holding vs closing at minute X is a coin flip.** For trades still open at X, the final result averages almost
   exactly their P&L at X ("hold value", table P3):
   - 2025: −0.05 to +0.07 at every X.
   - 2026: −0.03 to +0.13.
   - The split is by regime. In the 2026-H1 bear, holding past 30 min LOST (−0.21). In both bull periods it WON: 2025 +0.04,
     2026-H2 +0.32.
   - A close at X also pays the live market-exit give-back (about −0.17), so on average closing loses.
5. **The detail that kills the "TP → +0.25" idea.** Trades that are already in profit (+0.25 to +1.0) at minute 15–45 mostly go on to
   the full +1.25:
   - 2025 at 30 min: 81 % TP, hold value +0.62.
   - 2026 at 20 min: 83 % TP, +0.39.
   - Dropping the TP to +0.25 sells exactly those trades early. The trades it rescues are the ones that are negative at X and later
     bounce to +0.25 before stopping. They exist, but there are fewer of them.

**What the rules give** (Δ = change per trade vs today, in % of position; tables below)
- **TP → +x after T (x = 0 … 0.5).**
  - 2025: negative at every T (T30 +0.25: −0.016, P 0.20; +0.30: −0.020).
  - 2026: mildly positive at T15–T30 (+0.25 @ 30: +0.025, P 0.85; +0.30 @ 30: +0.028, P 0.88).
  - The 2026 gain comes from the 2026-H1 bear (T20–T30: +0.034 to +0.046). The 2026-H2 bull is about 0 or negative.
  - Under the pre-registered rule, a cell that wins in 2026 and loses in 2025 is regime-dependent, not a candidate. Eleven cells are
    regime-dependent this way, including both live Scout cells +0.25 @ 30 and +0.30 @ 30.
- **Close at T.** About 0 in both years at every T (2025 −0.008 at T30). With the live give-back it is negative (−0.02 to −0.04).
- **Tighter stop after T (−2 or −1).**
  - On trade prices: about 0 (2025 −0.01 to +0.04, 2026 −0.035 to +0.006, none at P ≥ 0.90 in both years). This matches the earlier
    2026 refutation (WILLY_LATE_SL_STUDY).
  - Under the live-slip ruler, T15_SL1 and T45_SL1 clear every gate. That is a ruler artefact. The ruler books every stop 0.20 better
    than its line and every market exit 0.17 worse, so each trade converted from a TP / cap exit into a stop gains +0.37 by assumption.
    That bonus is 0.06 / 0.03 per trade, which is the whole gain (on prints Δ = +0.010 / +0.001).
  - The +0.20 stop benefit comes from 2 live stops that filled after a −4 wick bounced. A slow drift to −1 after 15 minutes has no
    wick to bounce from.
  - The pre-registration requires a pass under both rulers, so neither is a candidate.
- **Is there a market-regime switch?** Every regime variable stamped at entry was read for the regime-dependent cells (25 BTC /
  breadth / volume variables, sign split first, in day units).
  - The pattern is coherent: the late TP drop helps when BTC is weak (1 h RSI < 50, fewer than half of the last 72 h above, low
    global volume) and hurts when BTC is strong.
  - But it is no stronger than chance: 56 flags vs a day-shuffled null of 47 on average (p 0.30).
  - It is a hypothesis only, not a lead to act on.

**Master and live**
- **Master B19–B21 (19 trades).** The TP-drop cells look good: +0.25 @ 30 Δ +0.218. That comes from 2 trades, API3 and ZK (2 saved,
  0 killed).
- **Live (5 trades, 2026-10-15).** No price path is in the cache yet, so every cell is n/a. Three were TPs in 1–7 minutes and STRK
  stopped at 9 minutes, so a 30-minute rule cannot touch them. Only DUSK (stopped at 38 min, peak +1.15) was open at 30. Whether it
  touched +0.25 between minute 30 and 38 is unknown.

**Recommendation (operator decides).**
- Change nothing. Keep the three live 30-minute Scout cells on their frozen gate (N ≥ 20 on ≥ 8 days).
- Read that gate together with this study's 2025 result: the backtest says the rule is a bear-market rule.
- The number worth measuring live is the stop fill vs the stop line. 2 fills say stops fill better and TPs worse. If that holds at
  N ≥ 15, every exit study should be re-priced with it.

## Candidate list

**None.**
- G1 (P ≥ 0.90 in both years) is passed under trade prices (S0) by **no cell**. Under the live-slip ruler (SL) it is passed by
  T15_SL1 and T45_SL1, the ruler artefact above.
- Expected false passes per ruler if the cells were independent: ≈ 0.4.

## Parity (every headline two independent ways)

| check | result |
|---|---|
| Every cell, reused walkers (C.walk_lock / study_willy_late_sl.ev) vs an independent per-event loop | **100 % identical per fill, 40 cells × 1,535 fills** (asserted) |
| Today's exit, loop at T = ∞ vs FX.evaluate | **100 %** (asserted) |
| Exit-CF rule: fills whose today-exit is at or before T show Δ = 0 | **asserted per fill per cell, 0 violations** |
| Walker today vs the replay's recorded pct_new (same kind ∧ \|Δ\| < 0.10) | 2025 **92.7 %** of 899 (tick paths 100 %), 2026 **89.0 %** of 617 (the known 1 Hz monitor vs first-print gap; WILLY_LATE_SL saw 89 %) |
| Avg today, walker vs recorded | 2025 −0.306 vs −0.238 (1m-only fills book stops at the line or the bar open) · 2026 +0.091 vs +0.089 |
| Master walker vs stack_pct | 19 / 19 |
| Known refs (T30_CLOSE / T30_TP0.25 / T30_TP0.3) | 2025 −0.0075 / −0.0156 / −0.0198 ✔ · 2026 (617) −0.0011 / +0.0249 / +0.0284 ✔ (exact to 4 decimals; asserted, halt otherwise) |

## Earlier grids this study re-prices (no silent re-runs)

- **WILLY_LATE_TP_GRID** (2026 only, 606 fills): TP15–30 × 0.2–0.4, best TP30_0.3 +0.022, P 0.83. Re-priced here on both years:
  2025 is negative for every T15–T30 TP cell.
- **WILLY_LATE_SL_STUDY** (2026 only): SL x_y and TP x_z. There, TP30_0.25 was +0.018 (P 0.79) and every late stop was ≤ +0.008. Here, on
  2026 with 617 fills, the S0 late-stop cells are −0.035 … +0.010, consistent.
- **TICK_STUDY_MAKER_LOCK review corrections:** the 2025 / 2026 Scout cell numbers are reproduced exactly (above).

## Could NOT be tested

- **Live block.** N = 5, and no tick / 1m data exists for 2026-10-15 in the cache (not fetched, cache-only), so live cells are n/a and
  the path state at X is unknown. Exit kind and minute are as recorded.
- **2025 non-top-50 (669 fills) are 1m bars only.**
  - On 1m bars a stop / late stop books at its line (or the bar open), which is optimistic for tighter stops.
  - Ticks-only Δ (S0) for the late-stop cells: T15_SL1 2025 −0.023 (vs +0.010 all), T30_SL1 −0.038.
  - Ticks-only Δ for the TP cells: T30_TP0.25 2025 −0.032, T30_TP0.3 −0.036, T30_CLOSE −0.051. These are worse than the
    all-fill 2025 numbers, so the 1m share does not hide a 2025 gain.
- **Queue / book slip of a late TP or close.** It is priced with a flat ruler (S0 = 0, SL = −0.17 market exits / +0.20 stops). The live
  ruler comes from 5 fills: 3 TPs at −0.13…−0.17 and 2 stops at +0.21 / +0.57. Its mechanism for stops (fill after a wick bounce)
  likely does not extend to slow late stops.
- **Slot effects.** An earlier exit frees a slot for another WILLY entry. This is not modelled.
- **Regime read.** The entry stamps are present on all 1,516 BT fills. The breadth "br_*" and bear_* columns are empty in the replay
  and could not be read.

## Exact commands

```
venv/bin/python scripts/study_willy_duration_profile.py --jobs 8   # cohort + today's-exit duration profile → WILLY_DURATION_cohort.csv, _profile_rows.csv, _profile_tables.md
venv/bin/python scripts/study_willy_duration_prereg.py             # freeze grid + gates (refuses to overwrite) → WILLY_30MIN_RESTUDY_PREREG.txt
venv/bin/python scripts/study_willy_duration_cells.py --jobs 8     # 40 cells two ways + asserts → WILLY_DURATION_cell_rows.csv
venv/bin/python scripts/study_willy_duration_report.py             # gates, two-sided, master per batch, regime read + null → WILLY_DURATION_cell_tables.md, _gates.csv, _regime_rows.csv
```
Shared code: scripts/study_willy_duration_common.py. It reuses study_tick_maker_lock_common, scout_frenzy_willy_exit and
study_willy_late_sl, with no new walker logic except the parity loop.

---
# WILLY duration profile — today's exit +1.25 / −4 / 90 min (generated by scripts/study_willy_duration_profile.py)

Exit kind / minute: ③ BT and ② MASTER = this walker on the fill's own path (ticks → 1m tail); ① LIVE = the RECORDED exit (no tick / 1m data for 2026-10-15 in the cache — not fetched). Net % per fill (fees in, no slip).

## P1. How fills end and how long it takes

| block | N | TP | TP min median / p75 / p90 | STOP | STOP min median / p75 / p90 | CAP (90 min) | avg % | avg TP % | avg STOP % | avg CAP % |
|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 5 | 3 (60 %) | 4.1 / 5.7 / 6.6 | 2 (40 %) | 23.8 / 31.0 / 35.3 | 0 (0 %) | -0.787 | +1.10 | -3.61 | – |
| ② MASTER B19 | 14 | 11 (79 %) | 4.9 / 19.2 / 23.4 | 1 (7 %) | 29.2 / 29.2 / 29.2 | 2 (14 %) | +0.451 | +1.27 | -4.00 | -1.81 |
| ② MASTER B20 | 3 | 2 (67 %) | 1.4 / 1.9 / 2.2 | 1 (33 %) | 80.2 / 80.2 / 80.2 | 0 (0 %) | -0.497 | +1.27 | -4.03 | – |
| ② MASTER B21 | 2 | 2 (100 %) | 4.5 / 6.7 / 8.0 | 0 (0 %) | – / – / – | 0 (0 %) | +1.253 | +1.25 | – | – |
| ② MASTER all | 19 | 15 (79 %) | 4.3 / 12.7 / 22.9 | 2 (11 %) | 54.7 / 67.5 / 75.1 | 2 (11 %) | +0.385 | +1.26 | -4.01 | -1.81 |
| ③ BT 2025 | 899 | 607 (68 %) | 4.8 / 13.8 / 31.8 | 223 (25 %) | 11.8 / 27.3 / 52.6 | 69 (8 %) | -0.306 | +1.25 | -4.23 | -1.35 |
| ③ BT 2025 top-50 | 230 | 162 (70 %) | 8.0 / 17.3 / 41.7 | 43 (19 %) | 21.8 / 49.9 / 65.9 | 25 (11 %) | +0.012 | +1.26 | -4.01 | -1.19 |
| ③ BT 2025 rest (1m) | 669 | 445 (67 %) | 4.8 / 12.8 / 30.7 | 180 (27 %) | 9.8 / 24.8 / 46.7 | 44 (7 %) | -0.415 | +1.25 | -4.28 | -1.43 |
| ③ BT 2026-H1 | 410 | 306 (75 %) | 4.1 / 11.3 / 25.6 | 75 (18 %) | 17.0 / 28.3 / 54.2 | 29 (7 %) | +0.118 | +1.27 | -4.01 | -1.31 |
| ③ BT 2026-H2 | 207 | 150 (72 %) | 6.3 / 22.0 / 41.1 | 40 (19 %) | 21.4 / 30.2 / 57.9 | 17 (8 %) | +0.039 | +1.26 | -4.01 | -1.22 |
| ③ BT 2026 | 617 | 456 (74 %) | 4.8 / 14.0 / 31.1 | 115 (19 %) | 17.7 / 29.9 / 56.6 | 46 (7 %) | +0.091 | +1.26 | -4.01 | -1.27 |

## P2. Share of each exit kind already done by minute X (cumulative)

| block | kind | N | ≤5 | ≤10 | ≤15 | ≤20 | ≤30 | ≤45 | ≤60 | ≤90 |
|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | TP | 3 | 67 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| ① LIVE | STOP | 2 | 0 % | 50 % | 50 % | 50 % | 50 % | 100 % | 100 % | 100 % |
| ② MASTER B19 | TP | 11 | 55 % | 64 % | 64 % | 73 % | 100 % | 100 % | 100 % | 100 % |
| ② MASTER B19 | STOP | 1 | 0 % | 0 % | 0 % | 0 % | 100 % | 100 % | 100 % | 100 % |
| ② MASTER B20 | TP | 2 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| ② MASTER B20 | STOP | 1 | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 100 % |
| ② MASTER B21 | TP | 2 | 50 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| ② MASTER all | TP | 15 | 60 % | 73 % | 73 % | 80 % | 100 % | 100 % | 100 % | 100 % |
| ② MASTER all | STOP | 2 | 0 % | 0 % | 0 % | 0 % | 50 % | 50 % | 50 % | 100 % |
| ③ BT 2025 | TP | 607 | 51 % | 69 % | 77 % | 82 % | 88 % | 94 % | 98 % | 100 % |
| ③ BT 2025 | STOP | 223 | 32 % | 46 % | 57 % | 64 % | 77 % | 84 % | 93 % | 100 % |
| ③ BT 2025 top-50 | TP | 162 | 40 % | 60 % | 73 % | 77 % | 84 % | 92 % | 98 % | 100 % |
| ③ BT 2025 top-50 | STOP | 43 | 12 % | 23 % | 40 % | 47 % | 58 % | 70 % | 86 % | 100 % |
| ③ BT 2025 rest (1m) | TP | 445 | 55 % | 71 % | 78 % | 83 % | 90 % | 95 % | 98 % | 100 % |
| ③ BT 2025 rest (1m) | STOP | 180 | 37 % | 52 % | 61 % | 68 % | 82 % | 88 % | 94 % | 100 % |
| ③ BT 2026-H1 | TP | 306 | 56 % | 72 % | 81 % | 89 % | 92 % | 95 % | 98 % | 100 % |
| ③ BT 2026-H1 | STOP | 75 | 15 % | 29 % | 44 % | 64 % | 75 % | 84 % | 95 % | 100 % |
| ③ BT 2026-H2 | TP | 150 | 44 % | 61 % | 69 % | 73 % | 85 % | 91 % | 95 % | 100 % |
| ③ BT 2026-H2 | STOP | 40 | 15 % | 30 % | 40 % | 48 % | 75 % | 82 % | 92 % | 100 % |
| ③ BT 2026 | TP | 456 | 52 % | 68 % | 77 % | 84 % | 89 % | 94 % | 97 % | 100 % |
| ③ BT 2026 | STOP | 115 | 15 % | 30 % | 43 % | 58 % | 75 % | 83 % | 94 % | 100 % |

## P3. Still open at minute X → how it ends (today's exit)

hold value = avg(final % − current % at X): what holding to today's exit earned vs closing at X (no slip; a close at X is a market order, the live ruler adds −0.17 to it).

| block | X | open at X (% of N) | → TP | → STOP | → CAP | avg final % | avg current % at X | hold value |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 10 | 1 (20 %) | 0 % | 100 % | 0 % | -3.43 | – | – |
| ① LIVE | 15 | 1 (20 %) | 0 % | 100 % | 0 % | -3.43 | – | – |
| ① LIVE | 20 | 1 (20 %) | 0 % | 100 % | 0 % | -3.43 | – | – |
| ① LIVE | 30 | 1 (20 %) | 0 % | 100 % | 0 % | -3.43 | – | – |
| ② MASTER B19 | 10 | 7 (50 %) | 57 % | 14 % | 29 % | -0.37 | -0.66 | +0.29 |
| ② MASTER B19 | 15 | 7 (50 %) | 57 % | 14 % | 29 % | -0.37 | -0.85 | +0.47 |
| ② MASTER B19 | 20 | 6 (43 %) | 50 % | 17 % | 33 % | -0.64 | -0.74 | +0.10 |
| ② MASTER B19 | 30 | 2 (14 %) | 0 % | 0 % | 100 % | -1.81 | -0.77 | -1.04 |
| ② MASTER B19 | 45 | 2 (14 %) | 0 % | 0 % | 100 % | -1.81 | -2.76 | +0.95 |
| ② MASTER B19 | 60 | 2 (14 %) | 0 % | 0 % | 100 % | -1.81 | -1.70 | -0.11 |
| ② MASTER B20 | 10 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -0.77 | -3.26 |
| ② MASTER B20 | 15 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -1.60 | -2.43 |
| ② MASTER B20 | 20 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -1.60 | -2.43 |
| ② MASTER B20 | 30 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -2.43 | -1.60 |
| ② MASTER B20 | 45 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -2.61 | -1.42 |
| ② MASTER B20 | 60 | 1 (33 %) | 0 % | 100 % | 0 % | -4.03 | -3.17 | -0.86 |
| ② MASTER all | 10 | 8 (42 %) | 50 % | 25 % | 25 % | -0.83 | -0.67 | -0.16 |
| ② MASTER all | 15 | 8 (42 %) | 50 % | 25 % | 25 % | -0.83 | -0.94 | +0.11 |
| ② MASTER all | 20 | 7 (37 %) | 43 % | 29 % | 29 % | -1.13 | -0.86 | -0.27 |
| ② MASTER all | 30 | 3 (16 %) | 0 % | 33 % | 67 % | -2.55 | -1.32 | -1.23 |
| ② MASTER all | 45 | 3 (16 %) | 0 % | 33 % | 67 % | -2.55 | -2.71 | +0.16 |
| ② MASTER all | 60 | 3 (16 %) | 0 % | 33 % | 67 % | -2.55 | -2.19 | -0.36 |
| ③ BT 2025 | 10 | 380 (42 %) | 50 % | 32 % | 18 % | -0.88 | -0.82 | -0.05 |
| ③ BT 2025 | 15 | 306 (34 %) | 46 % | 31 % | 23 % | -0.98 | -0.93 | -0.05 |
| ③ BT 2025 | 20 | 261 (29 %) | 43 % | 31 % | 26 % | -1.07 | -1.03 | -0.04 |
| ③ BT 2025 | 30 | 192 (21 %) | 38 % | 27 % | 36 % | -1.08 | -1.11 | +0.04 |
| ③ BT 2025 | 45 | 141 (16 %) | 26 % | 25 % | 49 % | -1.32 | -1.35 | +0.02 |
| ③ BT 2025 | 60 | 98 (11 %) | 13 % | 16 % | 70 % | -1.43 | -1.51 | +0.07 |
| ③ BT 2025 top-50 | 10 | 122 (53 %) | 52 % | 27 % | 20 % | -0.67 | -0.66 | -0.01 |
| ③ BT 2025 top-50 | 15 | 95 (41 %) | 46 % | 27 % | 26 % | -0.83 | -0.84 | +0.01 |
| ③ BT 2025 top-50 | 20 | 85 (37 %) | 44 % | 27 % | 29 % | -0.89 | -0.95 | +0.06 |
| ③ BT 2025 top-50 | 30 | 69 (30 %) | 38 % | 26 % | 36 % | -1.00 | -1.17 | +0.17 |
| ③ BT 2025 top-50 | 45 | 51 (22 %) | 25 % | 25 % | 49 % | -1.29 | -1.14 | -0.14 |
| ③ BT 2025 top-50 | 60 | 35 (15 %) | 11 % | 17 % | 71 % | -1.40 | -1.32 | -0.08 |
| ③ BT 2025 rest (1m) | 10 | 258 (39 %) | 49 % | 34 % | 17 % | -0.98 | -0.90 | -0.08 |
| ③ BT 2025 rest (1m) | 15 | 211 (32 %) | 46 % | 33 % | 21 % | -1.05 | -0.97 | -0.08 |
| ③ BT 2025 rest (1m) | 20 | 176 (26 %) | 42 % | 33 % | 25 % | -1.15 | -1.07 | -0.08 |
| ③ BT 2025 rest (1m) | 30 | 123 (18 %) | 37 % | 27 % | 36 % | -1.12 | -1.08 | -0.04 |
| ③ BT 2025 rest (1m) | 45 | 90 (13 %) | 27 % | 24 % | 49 % | -1.34 | -1.46 | +0.12 |
| ③ BT 2025 rest (1m) | 60 | 63 (9 %) | 14 % | 16 % | 70 % | -1.46 | -1.61 | +0.15 |
| ③ BT 2026-H1 | 10 | 168 (41 %) | 51 % | 32 % | 17 % | -0.84 | -0.96 | +0.12 |
| ③ BT 2026-H1 | 15 | 129 (31 %) | 45 % | 33 % | 22 % | -1.03 | -0.90 | -0.13 |
| ③ BT 2026-H1 | 20 | 91 (22 %) | 38 % | 30 % | 32 % | -1.12 | -1.00 | -0.11 |
| ③ BT 2026-H1 | 30 | 73 (18 %) | 34 % | 26 % | 40 % | -1.12 | -0.92 | -0.21 |
| ③ BT 2026-H1 | 45 | 55 (13 %) | 25 % | 22 % | 53 % | -1.24 | -1.08 | -0.16 |
| ③ BT 2026-H1 | 60 | 38 (9 %) | 13 % | 11 % | 76 % | -1.25 | -1.30 | +0.05 |
| ③ BT 2026-H2 | 10 | 103 (50 %) | 56 % | 27 % | 17 % | -0.58 | -0.71 | +0.13 |
| ③ BT 2026-H2 | 15 | 88 (43 %) | 53 % | 27 % | 19 % | -0.66 | -0.78 | +0.12 |
| ③ BT 2026-H2 | 20 | 78 (38 %) | 51 % | 27 % | 22 % | -0.70 | -0.90 | +0.20 |
| ③ BT 2026-H2 | 30 | 50 (24 %) | 46 % | 20 % | 34 % | -0.64 | -0.95 | +0.32 |
| ③ BT 2026-H2 | 45 | 37 (18 %) | 35 % | 19 % | 46 % | -0.87 | -1.11 | +0.24 |
| ③ BT 2026-H2 | 60 | 28 (14 %) | 29 % | 11 % | 61 % | -0.81 | -0.87 | +0.06 |
| ③ BT 2026 | 10 | 271 (44 %) | 53 % | 30 % | 17 % | -0.74 | -0.87 | +0.13 |
| ③ BT 2026 | 15 | 217 (35 %) | 48 % | 30 % | 21 % | -0.88 | -0.85 | -0.03 |
| ③ BT 2026 | 20 | 169 (27 %) | 44 % | 28 % | 27 % | -0.92 | -0.95 | +0.03 |
| ③ BT 2026 | 30 | 123 (20 %) | 39 % | 24 % | 37 % | -0.93 | -0.93 | +0.01 |
| ③ BT 2026 | 45 | 92 (15 %) | 29 % | 21 % | 50 % | -1.09 | -1.09 | +0.00 |
| ③ BT 2026 | 60 | 66 (11 %) | 20 % | 11 % | 70 % | -1.06 | -1.11 | +0.05 |

## P4. Open at X, split by CURRENT net % at X

Cells: N · TP % / STOP % / CAP % · avg final % · hold value (final − current at X). Blocks: ② master pooled, ③ BT per year.

| block | X | < 0 | 0–0.25 | 0.25–1.0 | ≥ 1.0 |
|---|---|---|---|---|---|
| ② MASTER all | 10 | 7 · 43 %/29 %/29 % · -1.13 · -0.22 | – | 1 · 100 %/0 %/0 % · +1.25 · +0.31 | – |
| ② MASTER all | 15 | 6 · 33 %/33 %/33 % · -1.52 · -0.07 | – | 2 · 100 %/0 %/0 % · +1.25 · +0.67 | – |
| ② MASTER all | 20 | 6 · 33 %/33 %/33 % · -1.52 · -0.37 | – | 1 · 100 %/0 %/0 % · +1.25 · +0.38 | – |
| ② MASTER all | 30 | 3 · 0 %/33 %/67 % · -2.55 · -1.23 | – | – | – |
| ② MASTER all | 45 | 3 · 0 %/33 %/67 % · -2.55 · +0.16 | – | – | – |
| ② MASTER all | 60 | 3 · 0 %/33 %/67 % · -2.55 · -0.36 | – | – | – |
| ③ BT 2025 | 10 | 292 · 43 %/38 %/19 % · -1.30 · -0.10 | 33 · 67 %/21 %/12 % · -0.03 · -0.18 | 53 · 79 %/2 %/19 % · +0.83 · +0.29 | 2 · 100 %/0 %/0 % · +1.25 · +0.17 |
| ③ BT 2025 | 15 | 250 · 38 %/37 %/25 % · -1.36 · -0.13 | 18 · 78 %/11 %/11 % · +0.39 · +0.27 | 36 · 83 %/3 %/14 % · +0.84 · +0.32 | 2 · 100 %/0 %/0 % · +1.25 · +0.16 |
| ③ BT 2025 | 20 | 206 · 34 %/39 %/27 % · -1.53 · -0.14 | 30 · 67 %/0 %/33 % · +0.62 · +0.52 | 25 · 84 %/0 %/16 % · +0.78 · +0.19 | – |
| ③ BT 2025 | 30 | 156 · 29 %/32 %/39 % · -1.52 · -0.07 | 13 · 62 %/8 %/31 % · +0.45 · +0.35 | 21 · 81 %/0 %/19 % · +1.03 · +0.62 | 2 · 100 %/0 %/0 % · +1.25 · +0.08 |
| ③ BT 2025 | 45 | 122 · 18 %/29 %/53 % · -1.69 · -0.05 | 5 · 60 %/0 %/40 % · +0.92 · +0.78 | 13 · 85 %/0 %/15 % · +1.07 · +0.46 | 1 · 100 %/0 %/0 % · +1.25 · +0.09 |
| ③ BT 2025 | 60 | 91 · 12 %/18 %/70 % · -1.55 · +0.09 | 3 · 0 %/0 %/100 % · -1.00 · -1.06 | 4 · 50 %/0 %/50 % · +0.94 · +0.42 | – |
| ③ BT 2025 top-50 | 10 | 85 · 40 %/38 %/22 % · -1.34 · -0.21 | 17 · 76 %/6 %/18 % · +0.74 · +0.59 | 19 · 84 %/0 %/16 % · +0.99 · +0.36 | 1 · 100 %/0 %/0 % · +1.26 · +0.11 |
| ③ BT 2025 top-50 | 15 | 75 · 35 %/35 %/31 % · -1.35 · -0.17 | 6 · 83 %/0 %/17 % · +1.11 · +0.99 | 12 · 92 %/0 %/8 % · +1.13 · +0.61 | 2 · 100 %/0 %/0 % · +1.25 · +0.16 |
| ③ BT 2025 top-50 | 20 | 66 · 36 %/35 %/29 % · -1.31 · +0.01 | 9 · 44 %/0 %/56 % · +0.21 · +0.11 | 10 · 90 %/0 %/10 % · +0.90 · +0.33 | – |
| ③ BT 2025 top-50 | 30 | 54 · 28 %/33 %/39 % · -1.55 · +0.03 | 4 · 50 %/0 %/50 % · +0.66 · +0.60 | 11 · 82 %/0 %/18 % · +1.06 · +0.69 | – |
| ③ BT 2025 top-50 | 45 | 43 · 16 %/30 %/53 % · -1.72 · -0.30 | 4 · 50 %/0 %/50 % · +0.84 · +0.71 | 4 · 100 %/0 %/0 % · +1.25 · +0.69 | – |
| ③ BT 2025 top-50 | 60 | 32 · 9 %/19 %/72 % · -1.56 · -0.07 | 1 · 0 %/0 %/100 % · -0.73 · -0.83 | 2 · 50 %/0 %/50 % · +0.82 · +0.22 | – |
| ③ BT 2025 rest (1m) | 10 | 207 · 44 %/39 %/17 % · -1.28 · -0.06 | 16 · 56 %/38 %/6 % · -0.86 · -0.99 | 34 · 76 %/3 %/21 % · +0.74 · +0.25 | 1 · 100 %/0 %/0 % · +1.25 · +0.24 |
| ③ BT 2025 rest (1m) | 15 | 175 · 39 %/38 %/22 % · -1.36 · -0.11 | 12 · 75 %/17 %/8 % · +0.03 · -0.09 | 24 · 79 %/4 %/17 % · +0.69 · +0.18 | – |
| ③ BT 2025 rest (1m) | 20 | 140 · 33 %/41 %/26 % · -1.64 · -0.22 | 21 · 76 %/0 %/24 % · +0.79 · +0.69 | 15 · 80 %/0 %/20 % · +0.69 · +0.10 | – |
| ③ BT 2025 rest (1m) | 30 | 102 · 29 %/31 %/39 % · -1.50 · -0.13 | 9 · 67 %/11 %/22 % · +0.35 · +0.23 | 10 · 80 %/0 %/20 % · +0.99 · +0.55 | 2 · 100 %/0 %/0 % · +1.25 · +0.08 |
| ③ BT 2025 rest (1m) | 45 | 79 · 19 %/28 %/53 % · -1.68 · +0.08 | 1 · 100 %/0 %/0 % · +1.25 · +1.07 | 9 · 78 %/0 %/22 % · +0.99 · +0.36 | 1 · 100 %/0 %/0 % · +1.25 · +0.09 |
| ③ BT 2025 rest (1m) | 60 | 59 · 14 %/17 %/69 % · -1.55 · +0.18 | 2 · 0 %/0 %/100 % · -1.13 · -1.17 | 2 · 50 %/0 %/50 % · +1.07 · +0.63 | – |
| ③ BT 2026-H1 | 10 | 136 · 46 %/35 %/18 % · -1.09 · +0.19 | 11 · 64 %/18 %/18 % · -0.01 · -0.13 | 21 · 76 %/14 %/10 % · +0.30 · -0.17 | – |
| ③ BT 2026-H1 | 15 | 98 · 34 %/43 %/23 % · -1.62 · -0.31 | 9 · 67 %/0 %/33 % · +0.71 · +0.62 | 21 · 86 %/0 %/14 % · +0.88 · +0.37 | 1 · 100 %/0 %/0 % · +1.30 · +0.11 |
| ③ BT 2026-H1 | 20 | 73 · 30 %/37 %/33 % · -1.54 · -0.20 | 6 · 67 %/0 %/33 % · +0.50 · +0.38 | 12 · 75 %/0 %/25 % · +0.65 · +0.14 | – |
| ③ BT 2026-H1 | 30 | 52 · 25 %/33 %/42 % · -1.59 · -0.18 | 11 · 55 %/9 %/36 % · +0.01 · -0.10 | 10 · 60 %/10 %/30 % · +0.07 · -0.47 | – |
| ③ BT 2026-H1 | 45 | 48 · 17 %/25 %/58 % · -1.57 · -0.23 | 1 · 100 %/0 %/0 % · +1.31 · +1.21 | 5 · 80 %/0 %/20 % · +0.95 · +0.26 | 1 · 100 %/0 %/0 % · +1.26 · +0.22 |
| ③ BT 2026-H1 | 60 | 34 · 9 %/12 %/79 % · -1.42 · +0.12 | – | 3 · 67 %/0 %/33 % · +0.35 · -0.26 | 1 · 0 %/0 %/100 % · -0.27 · -1.32 |
| ③ BT 2026-H2 | 10 | 75 · 47 %/36 %/17 % · -1.09 · +0.01 | 14 · 79 %/7 %/14 % · +0.68 · +0.52 | 14 · 86 %/0 %/14 % · +0.90 · +0.41 | – |
| ③ BT 2026-H2 | 15 | 68 · 46 %/31 %/24 % · -0.97 · +0.20 | 4 · 100 %/0 %/0 % · +1.26 · +1.10 | 14 · 71 %/21 %/7 % · +0.03 · -0.53 | 2 · 100 %/0 %/0 % · +1.26 · +0.13 |
| ③ BT 2026-H2 | 20 | 60 · 42 %/33 %/25 % · -1.16 · +0.10 | 7 · 71 %/14 %/14 % · +0.39 · +0.29 | 11 · 91 %/0 %/9 % · +1.13 · +0.66 | – |
| ③ BT 2026-H2 | 30 | 40 · 38 %/25 %/38 % · -1.02 · +0.27 | 3 · 67 %/0 %/33 % · +0.63 · +0.47 | 7 · 86 %/0 %/14 % · +1.05 · +0.52 | – |
| ③ BT 2026-H2 | 45 | 33 · 33 %/21 %/45 % · -1.06 · +0.24 | 1 · 0 %/0 %/100 % · +0.47 · +0.38 | 3 · 67 %/0 %/33 % · +0.75 · +0.14 | – |
| ③ BT 2026-H2 | 60 | 21 · 19 %/14 %/67 % · -1.23 · +0.01 | 4 · 50 %/0 %/50 % · +0.26 · +0.15 | 3 · 67 %/0 %/33 % · +0.75 · +0.27 | – |
| ③ BT 2026 | 10 | 211 · 46 %/36 %/18 % · -1.09 · +0.12 | 25 · 72 %/12 %/16 % · +0.38 · +0.24 | 35 · 80 %/9 %/11 % · +0.54 · +0.06 | – |
| ③ BT 2026 | 15 | 166 · 39 %/38 %/23 % · -1.35 · -0.10 | 13 · 77 %/0 %/23 % · +0.88 · +0.77 | 35 · 80 %/9 %/11 % · +0.54 · +0.01 | 3 · 100 %/0 %/0 % · +1.27 · +0.12 |
| ③ BT 2026 | 20 | 133 · 35 %/35 %/29 % · -1.37 · -0.06 | 13 · 69 %/8 %/23 % · +0.44 · +0.33 | 23 · 83 %/0 %/17 % · +0.88 · +0.39 | – |
| ③ BT 2026 | 30 | 92 · 30 %/29 %/40 % · -1.35 · +0.02 | 14 · 57 %/7 %/36 % · +0.14 · +0.02 | 17 · 71 %/6 %/24 % · +0.47 · -0.06 | – |
| ③ BT 2026 | 45 | 81 · 23 %/23 %/53 % · -1.36 · -0.04 | 2 · 50 %/0 %/50 % · +0.89 · +0.80 | 8 · 75 %/0 %/25 % · +0.87 · +0.21 | 1 · 100 %/0 %/0 % · +1.26 · +0.22 |
| ③ BT 2026 | 60 | 55 · 13 %/13 %/75 % · -1.35 · +0.08 | 4 · 50 %/0 %/50 % · +0.26 · +0.15 | 6 · 67 %/0 %/33 % · +0.55 · +0.00 | 1 · 0 %/0 %/100 % · -0.27 · -1.32 |

## P5. Open at X, split by PEAK-so-far net % at X

Cells: N · TP % / STOP % / CAP % · avg final % · hold value (final − current at X). Blocks: ② master pooled, ③ BT per year.

| block | X | < 0.25 | 0.25–0.5 | 0.5–1.0 | ≥ 1.0 |
|---|---|---|---|---|---|
| ② MASTER all | 10 | 2 · 50 %/0 %/50 % · -0.45 · +1.11 | 4 · 50 %/50 %/0 % · -1.38 · -0.59 | 2 · 50 %/0 %/50 % · -0.11 · -0.55 | – |
| ② MASTER all | 15 | 2 · 50 %/0 %/50 % · -0.45 · +0.89 | 3 · 33 %/67 %/0 % · -2.26 · -0.47 | 1 · 0 %/0 %/100 % · -1.48 · -0.81 | 2 · 100 %/0 %/0 % · +1.25 · +0.67 |
| ② MASTER all | 20 | 2 · 50 %/0 %/50 % · -0.45 · +0.43 | 3 · 33 %/67 %/0 % · -2.26 · -1.27 | 1 · 0 %/0 %/100 % · -1.48 · +0.69 | 1 · 100 %/0 %/0 % · +1.25 · +0.38 |
| ② MASTER all | 30 | 1 · 0 %/0 %/100 % · -2.15 · -1.39 | 1 · 0 %/100 %/0 % · -4.03 · -1.60 | 1 · 0 %/0 %/100 % · -1.48 · -0.69 | – |
| ② MASTER all | 45 | 1 · 0 %/0 %/100 % · -2.15 · +1.00 | 1 · 0 %/100 %/0 % · -4.03 · -1.42 | 1 · 0 %/0 %/100 % · -1.48 · +0.90 | – |
| ② MASTER all | 60 | – | 1 · 0 %/100 %/0 % · -4.03 · -0.86 | 1 · 0 %/0 %/100 % · -1.48 · +0.99 | 1 · 0 %/0 %/100 % · -2.15 · -1.21 |
| ③ BT 2025 | 10 | 130 · 35 %/45 %/21 % · -1.67 · -0.26 | 70 · 53 %/27 %/20 % · -0.65 · +0.21 | 134 · 59 %/23 %/18 % · -0.39 · +0.04 | 46 · 65 %/26 %/9 % · -0.42 · -0.17 |
| ③ BT 2025 | 15 | 92 · 29 %/45 %/26 % · -1.82 · -0.40 | 64 · 50 %/27 %/23 % · -0.70 · +0.32 | 109 · 53 %/24 %/23 % · -0.56 · +0.09 | 41 · 59 %/29 %/12 % · -0.67 · -0.20 |
| ③ BT 2025 | 20 | 65 · 26 %/42 %/32 % · -1.87 · -0.39 | 55 · 42 %/27 %/31 % · -0.92 · +0.21 | 100 · 49 %/27 %/24 % · -0.70 · +0.11 | 41 · 54 %/29 %/17 % · -0.87 · -0.15 |
| ③ BT 2025 | 30 | 44 · 18 %/39 %/43 % · -2.07 · -0.51 | 42 · 36 %/24 %/40 % · -1.01 · +0.25 | 73 · 41 %/23 %/36 % · -0.74 · +0.17 | 33 · 58 %/21 %/21 % · -0.58 · +0.18 |
| ③ BT 2025 | 45 | 30 · 7 %/33 %/60 % · -2.32 · -0.55 | 30 · 27 %/23 %/50 % · -1.33 · +0.36 | 53 · 26 %/23 %/51 % · -1.04 · +0.07 | 28 · 46 %/21 %/32 % · -0.78 · +0.19 |
| ③ BT 2025 | 60 | 23 · 0 %/22 %/78 % · -2.27 · -0.58 | 23 · 17 %/17 %/65 % · -1.43 · +0.22 | 35 · 11 %/14 %/74 % · -1.16 · +0.15 | 17 · 29 %/12 %/59 % · -0.88 · +0.60 |
| ③ BT 2025 top-50 | 10 | 39 · 36 %/31 %/33 % · -1.33 · -0.10 | 26 · 58 %/27 %/15 % · -0.43 · +0.32 | 39 · 59 %/23 %/18 % · -0.32 · +0.04 | 18 · 67 %/28 %/6 % · -0.35 · -0.38 |
| ③ BT 2025 top-50 | 15 | 27 · 30 %/30 %/41 % · -1.51 · -0.42 | 21 · 43 %/29 %/29 % · -0.81 · +0.31 | 32 · 56 %/22 %/22 % · -0.33 · +0.40 | 15 · 60 %/33 %/7 % · -0.67 · -0.46 |
| ③ BT 2025 top-50 | 20 | 21 · 24 %/33 %/43 % · -1.82 · -0.37 | 20 · 40 %/20 %/40 % · -0.64 · +0.17 | 30 · 57 %/23 %/20 % · -0.32 · +0.39 | 14 · 50 %/36 %/14 % · -1.06 · -0.14 |
| ③ BT 2025 top-50 | 30 | 17 · 18 %/35 %/47 % · -2.03 · -0.44 | 15 · 33 %/20 %/47 % · -0.96 · +0.52 | 24 · 42 %/25 %/33 % · -0.62 · +0.37 | 13 · 62 %/23 %/15 % · -0.43 · +0.20 |
| ③ BT 2025 top-50 | 45 | 12 · 0 %/33 %/67 % · -2.52 · -0.97 | 9 · 33 %/0 %/67 % · -0.66 · +0.85 | 18 · 22 %/33 %/44 % · -1.25 · -0.21 | 12 · 50 %/25 %/25 % · -0.58 · +0.04 |
| ③ BT 2025 top-50 | 60 | 10 · 0 %/20 %/80 % · -2.22 · -0.95 | 7 · 14 %/0 %/86 % · -1.21 · +0.29 | 11 · 0 %/27 %/73 % · -1.40 · +0.02 | 7 · 43 %/14 %/43 % · -0.39 · +0.64 |
| ③ BT 2025 rest (1m) | 10 | 91 · 34 %/51 %/15 % · -1.82 · -0.32 | 44 · 50 %/27 %/23 % · -0.77 · +0.14 | 95 · 59 %/23 %/18 % · -0.41 · +0.05 | 28 · 64 %/25 %/11 % · -0.47 · -0.04 |
| ③ BT 2025 rest (1m) | 15 | 65 · 29 %/51 %/20 % · -1.95 · -0.39 | 43 · 53 %/26 %/21 % · -0.64 · +0.33 | 77 · 52 %/25 %/23 % · -0.65 · -0.04 | 26 · 58 %/27 %/15 % · -0.66 · -0.05 |
| ③ BT 2025 rest (1m) | 20 | 44 · 27 %/45 %/27 % · -1.90 · -0.40 | 35 · 43 %/31 %/26 % · -1.07 · +0.23 | 70 · 46 %/29 %/26 % · -0.86 · -0.01 | 27 · 56 %/26 %/19 % · -0.76 · -0.15 |
| ③ BT 2025 rest (1m) | 30 | 27 · 19 %/41 %/41 % · -2.09 · -0.56 | 27 · 37 %/26 %/37 % · -1.04 · +0.10 | 49 · 41 %/22 %/37 % · -0.81 · +0.08 | 20 · 55 %/20 %/25 % · -0.68 · +0.17 |
| ③ BT 2025 rest (1m) | 45 | 18 · 11 %/33 %/56 % · -2.19 · -0.27 | 21 · 24 %/33 %/43 % · -1.61 · +0.15 | 35 · 29 %/17 %/54 % · -0.94 · +0.21 | 16 · 44 %/19 %/38 % · -0.93 · +0.30 |
| ③ BT 2025 rest (1m) | 60 | 13 · 0 %/23 %/77 % · -2.30 · -0.30 | 16 · 19 %/25 %/56 % · -1.52 · +0.19 | 24 · 17 %/8 %/75 % · -1.05 · +0.20 | 10 · 20 %/10 %/70 % · -1.23 · +0.56 |
| ③ BT 2026-H1 | 10 | 55 · 44 %/31 %/25 % · -1.09 · +0.29 | 36 · 47 %/39 %/14 % · -1.19 · +0.01 | 56 · 55 %/27 %/18 % · -0.50 · +0.10 | 21 · 67 %/33 %/0 % · -0.50 · -0.09 |
| ③ BT 2026-H1 | 15 | 37 · 30 %/38 %/32 % · -1.67 · -0.10 | 26 · 46 %/31 %/23 % · -1.03 · -0.21 | 50 · 54 %/24 %/22 % · -0.44 · +0.02 | 16 · 50 %/50 %/0 % · -1.39 · -0.51 |
| ③ BT 2026-H1 | 20 | 25 · 24 %/36 %/40 % · -1.85 · +0.06 | 19 · 42 %/26 %/32 % · -0.99 · -0.08 | 36 · 42 %/25 %/33 % · -0.73 · -0.17 | 11 · 55 %/36 %/9 % · -0.92 · -0.36 |
| ③ BT 2026-H1 | 30 | 21 · 24 %/33 %/43 % · -1.75 · +0.12 | 11 · 27 %/18 %/55 % · -1.20 · +0.06 | 27 · 30 %/26 %/44 % · -1.00 · -0.74 | 14 · 64 %/21 %/14 % · -0.36 · +0.12 |
| ③ BT 2026-H1 | 45 | 12 · 25 %/0 %/75 % · -0.94 · +0.82 | 7 · 0 %/29 %/71 % · -2.38 · -0.53 | 23 · 22 %/22 %/57 % · -1.01 · -0.31 | 13 · 46 %/38 %/15 % · -1.30 · -0.58 |
| ③ BT 2026-H1 | 60 | 9 · 0 %/0 %/100 % · -1.71 · -0.20 | 5 · 0 %/0 %/100 % · -1.73 · +0.30 | 17 · 18 %/18 %/65 % · -0.93 · +0.34 | 7 · 29 %/14 %/57 % · -1.08 · -0.50 |
| ③ BT 2026-H2 | 10 | 24 · 38 %/33 %/29 % · -1.27 · -0.19 | 20 · 50 %/35 %/15 % · -0.89 · +0.30 | 47 · 64 %/26 %/11 % · -0.41 · +0.00 | 12 · 75 %/8 %/17 % · +0.61 · +1.00 |
| ③ BT 2026-H2 | 15 | 17 · 29 %/41 %/29 % · -1.74 · -0.31 | 15 · 40 %/33 %/27 % · -1.03 · +0.17 | 33 · 55 %/27 %/18 % · -0.71 · +0.08 | 23 · 78 %/13 %/9 % · +0.46 · +0.47 |
| ③ BT 2026-H2 | 20 | 15 · 27 %/40 %/33 % · -1.79 · -0.52 | 13 · 38 %/31 %/31 % · -0.97 · +0.22 | 32 · 56 %/25 %/19 % · -0.61 · +0.34 | 18 · 72 %/17 %/11 % · +0.24 · +0.53 |
| ③ BT 2026-H2 | 30 | 11 · 27 %/27 %/45 % · -1.46 · -0.32 | 7 · 43 %/14 %/43 % · -0.51 · +0.98 | 22 · 45 %/23 %/32 % · -0.77 · +0.17 | 10 · 70 %/10 %/20 % · +0.48 · +0.89 |
| ③ BT 2026-H2 | 45 | 6 · 33 %/0 %/67 % · -0.74 · +0.60 | 6 · 17 %/17 %/67 % · -1.15 · +0.01 | 16 · 31 %/25 %/44 % · -1.20 · +0.25 | 9 · 56 %/22 %/22 % · -0.20 · +0.13 |
| ③ BT 2026-H2 | 60 | 4 · 25 %/0 %/75 % · -0.87 · +0.29 | 6 · 17 %/17 %/67 % · -1.15 · -0.20 | 13 · 31 %/8 %/62 % · -0.82 · +0.04 | 5 · 40 %/20 %/40 % · -0.31 · +0.23 |
| ③ BT 2026 | 10 | 79 · 42 %/32 %/27 % · -1.14 · +0.14 | 56 · 48 %/38 %/14 % · -1.08 · +0.12 | 103 · 59 %/26 %/15 % · -0.46 · +0.06 | 33 · 70 %/24 %/6 % · -0.09 · +0.31 |
| ③ BT 2026 | 15 | 54 · 30 %/39 %/31 % · -1.69 · -0.17 | 41 · 44 %/32 %/24 % · -1.03 · -0.07 | 83 · 54 %/25 %/20 % · -0.55 · +0.04 | 39 · 67 %/28 %/5 % · -0.30 · +0.07 |
| ③ BT 2026 | 20 | 40 · 25 %/38 %/38 % · -1.83 · -0.16 | 32 · 41 %/28 %/31 % · -0.99 · +0.05 | 68 · 49 %/25 %/26 % · -0.67 · +0.07 | 29 · 66 %/24 %/10 % · -0.20 · +0.19 |
| ③ BT 2026 | 30 | 32 · 25 %/31 %/44 % · -1.65 · -0.03 | 18 · 33 %/17 %/50 % · -0.93 · +0.42 | 49 · 37 %/24 %/39 % · -0.90 · -0.33 | 24 · 67 %/17 %/17 % · -0.01 · +0.44 |
| ③ BT 2026 | 45 | 18 · 28 %/0 %/72 % · -0.87 · +0.74 | 13 · 8 %/23 %/69 % · -1.82 · -0.28 | 39 · 26 %/23 %/51 % · -1.09 · -0.08 | 22 · 50 %/32 %/18 % · -0.85 · -0.29 |
| ③ BT 2026 | 60 | 13 · 8 %/0 %/92 % · -1.45 · -0.05 | 11 · 9 %/9 %/82 % · -1.42 · +0.03 | 30 · 23 %/13 %/63 % · -0.88 · +0.21 | 12 · 33 %/17 %/50 % · -0.76 · -0.20 |

## P6. Winners vs losers: trough before a TP, peak before a STOP / CAP

| block | TP: N · trough median / p10 | STOP: N · peak median / p75 / share peak ≥ +0.25 / ≥ +0.5 | CAP: N · peak median / share peak ≥ +0.25 · final median |
|---|---|---|---|
| ① LIVE | 3 · -1.31 / -1.92 | 2 · +0.96 / +1.05 / 100 % / 100 % | 0 · – / – · – |
| ② MASTER B19 | 11 · -0.70 / -2.39 | 1 · +0.46 / +0.46 / 100 % / 0 % | 2 · +0.89 / 100 % · -1.81 |
| ② MASTER B20 | 2 · -0.97 / -1.60 | 1 · +0.28 / +0.28 / 100 % / 0 % | 0 · – / – · – |
| ② MASTER B21 | 2 · -0.61 / -0.74 | 0 · – / – / – / – | 0 · – / – · – |
| ② MASTER all | 15 · -0.70 / -2.28 | 2 · +0.37 / +0.41 / 100 % / 0 % | 2 · +0.89 / 100 % · -1.81 |
| ③ BT 2025 | 607 · -0.84 / -2.63 | 223 · +0.04 / +0.58 / 39 % / 28 % | 69 · +0.58 / 80 % · -1.33 |
| ③ BT 2025 top-50 | 162 · -0.72 / -2.37 | 43 · +0.38 / +0.59 / 60 % / 42 % | 25 · +0.45 / 72 % · -1.57 |
| ③ BT 2025 rest (1m) | 445 · -0.90 / -2.68 | 180 · -0.16 / +0.51 / 34 % / 25 % | 44 · +0.58 / 84 % · -1.30 |
| ③ BT 2026-H1 | 306 · -0.72 / -2.60 | 75 · +0.46 / +0.92 / 67 % / 44 % | 29 · +0.53 / 69 % · -1.08 |
| ③ BT 2026-H2 | 150 · -1.01 / -2.77 | 40 · +0.48 / +0.90 / 70 % / 50 % | 17 · +0.64 / 82 % · -1.12 |
| ③ BT 2026 | 456 · -0.83 / -2.64 | 115 · +0.46 / +0.91 / 68 % / 46 % | 46 · +0.58 / 74 % · -1.10 |

# WILLY late-exit cells — tables (generated by scripts/study_willy_duration_report.py)

Δ = cell − today (+1.25 / −4 / 90) on the same fill, net %/fill. S0 = zero slip; SL = live slip ruler (TP / late TP / cap / close −0.17, stop / late stop +0.20). P = day-clustered bootstrap P(mean Δ > 0), 4,000 draws, seed 7. SAVED = today's STOP / CAP fills the cell improves (n · Σ points); KILLED = today's TP fills the cell exits lower; DEEPER = today's STOP / CAP fills the cell makes worse. ① LIVE: no path data for 2026-10-15 in the cache → every cell n/a (DUSK, the only live fill open past 15 min, stopped at 38 min; its path after minute 15–30 is unknown).

## Parity

- 2025: walker today vs the replay's recorded pct_new (same kind ∧ |Δ| < 0.10): 92.7 % of 899 (tick paths 100.0 % of 230); mean walker − recorded -0.068; avg today walker -0.306 vs recorded -0.238.
- 2026: walker today vs the replay's recorded pct_new (same kind ∧ |Δ| < 0.10): 89.0 % of 617 (tick paths 89.0 % of 593); mean walker − recorded +0.002; avg today walker +0.091 vs recorded +0.089.
- MASTER: walker today vs stack_pct same kind ∧ |Δ| < 0.10: 100 % of 19 (recorded close reasons are the OLD exit; stack_pct is today's re-walk).
- Every cell priced two ways (reused walkers vs independent per-event loop): 100 % identical per fill (asserted); loop at T = ∞ == FX.evaluate 100 % (asserted); exit-CF rule Δ = 0 for every fill whose today-exit is ≤ T (asserted per fill per cell).

| year | cell | Δ this study | reference | match |
|---|---|---|---|---|
| 2025 | T30_CLOSE | -0.0075 | -0.0075 | yes |
| 2025 | T30_TP0.25 | -0.0156 | -0.0156 | yes |
| 2025 | T30_TP0.3 | -0.0198 | -0.0198 | yes |
| 2026 | T30_CLOSE | -0.0011 | -0.0011 | yes |
| 2026 | T30_TP0.25 | +0.0249 | +0.0249 | yes |
| 2026 | T30_TP0.3 | +0.0284 | +0.0284 | yes |

## Gate summary — candidates (pass under S0 AND SL): NONE

Cells passing G1 (both years P ≥ 0.90) — S0: none · SL: ['T15_SL1', 'T45_SL1']. Regime-dependent (2026 > 0, 2025 < 0) — S0: ['T15_TP0.25', 'T15_TP0.3', 'T15_TP0.5', 'T20_TP0.25', 'T20_TP0.3', 'T20_TP0.5', 'T30_TP0', 'T30_TP0.1', 'T30_TP0.25', 'T30_TP0.3', 'T30_TP0.5'] · SL: ['T15_TP0.25', 'T15_TP0.3', 'T15_TP0.5', 'T20_TP0.3', 'T20_TP0.5', 'T30_TP0', 'T30_TP0.1', 'T30_TP0.25', 'T30_TP0.3', 'T30_TP0.5'].


## Cells vs today — ruler S0 (③ backtest per year · ② master)

| cell | 2025 Δ (P) | 2025 top-50 Δ | 2025 rest Δ | 2026-H1 Δ (P) | 2026-H2 Δ (P) | 2026 Δ (P) | master Δ (N) | LOMO 25/26 | top day / pair share (pooled) | gates G1–G5 |
|---|---|---|---|---|---|---|---|---|---|---|
| T15_CLOSE | +0.017 (0.65) | -0.005 | +0.024 | +0.040 (0.80) | -0.053 (0.29) | +0.009 (0.57) | -0.046 (19) | n/n | 0.31 / 0.51 | nnnnn |
| T15_TP0 | -0.041 (0.08) | -0.056 | -0.036 | -0.002 (0.47) | -0.020 (0.39) | -0.008 (0.41) | -0.015 (19) | Y/n | 0.12 / 0.12 | nnnnY |
| T15_TP0.1 | -0.049 (0.04) | -0.064 | -0.044 | +0.003 (0.52) | -0.010 (0.44) | -0.002 (0.48) | +0.008 (19) | Y/n | 0.10 / 0.10 | nYnnY |
| T15_TP0.25 | -0.042 (0.05) | -0.070 | -0.033 | +0.015 (0.65) | +0.008 (0.54) | +0.013 (0.64) | +0.039 (19) | Y/n | 0.13 / 0.13 | nYnnY |
| T15_TP0.3 | -0.047 (0.03) | -0.070 | -0.039 | +0.022 (0.72) | +0.016 (0.58) | +0.020 (0.72) | -0.045 (19) | Y/Y | 0.13 / 0.13 | nnYnY |
| T15_TP0.5 | -0.052 (0.01) | -0.074 | -0.045 | +0.019 (0.71) | +0.008 (0.53) | +0.015 (0.69) | -0.007 (19) | Y/Y | 0.08 / 0.08 | nnYnY |
| T15_SL2 | +0.020 (0.72) | -0.019 | +0.033 | +0.029 (0.75) | -0.058 (0.21) | +0.000 (0.50) | +0.153 (19) | Y/n | 0.22 / 0.46 | nYnnn |
| T15_SL1 | +0.010 (0.59) | -0.023 | +0.022 | +0.048 (0.86) | -0.078 (0.21) | +0.006 (0.56) | -0.017 (19) | n/n | 0.43 / 0.82 | nnnnn |
| T20_CLOSE | +0.010 (0.60) | -0.023 | +0.021 | +0.025 (0.73) | -0.075 (0.19) | -0.008 (0.43) | +0.098 (19) | n/n | 1.47 / 2.20 | nYnnn |
| T20_TP0 | -0.047 (0.03) | -0.037 | -0.050 | +0.034 (0.84) | -0.086 (0.06) | -0.006 (0.42) | +0.039 (19) | Y/n | 0.10 / 0.11 | nYnnY |
| T20_TP0.1 | -0.046 (0.03) | -0.048 | -0.046 | +0.036 (0.86) | -0.071 (0.10) | -0.000 (0.50) | +0.062 (19) | Y/n | 0.10 / 0.11 | nYnnY |
| T20_TP0.25 | -0.039 (0.04) | -0.060 | -0.032 | +0.040 (0.90) | -0.077 (0.07) | +0.001 (0.51) | +0.093 (19) | Y/n | 0.10 / 0.12 | nYnnY |
| T20_TP0.3 | -0.045 (0.02) | -0.062 | -0.039 | +0.046 (0.93) | -0.072 (0.08) | +0.007 (0.60) | +0.009 (19) | Y/n | 0.09 / 0.11 | nYnnY |
| T20_TP0.5 | -0.045 (0.01) | -0.054 | -0.042 | +0.044 (0.93) | -0.066 (0.06) | +0.008 (0.61) | +0.041 (19) | Y/n | 0.07 / 0.08 | nYnnY |
| T20_SL2 | +0.040 (0.93) | -0.007 | +0.056 | -0.011 (0.37) | -0.062 (0.17) | -0.028 (0.20) | +0.181 (19) | Y/Y | 0.22 / 0.31 | nYYYn |
| T20_SL1 | +0.027 (0.78) | -0.007 | +0.039 | +0.021 (0.70) | -0.075 (0.19) | -0.011 (0.40) | +0.266 (19) | Y/n | 0.30 / 0.43 | nYnnn |
| T30_CLOSE | -0.008 (0.38) | -0.051 | +0.007 | +0.037 (0.84) | -0.077 (0.13) | -0.001 (0.50) | +0.194 (19) | n/n | 0.81 / 0.79 | nYnnY |
| T30_TP0 | -0.024 (0.11) | -0.015 | -0.027 | +0.039 (0.89) | -0.011 (0.39) | +0.022 (0.81) | +0.191 (19) | Y/Y | 0.43 / 0.47 | nYYnY |
| T30_TP0.1 | -0.016 (0.20) | -0.015 | -0.016 | +0.036 (0.89) | +0.002 (0.50) | +0.025 (0.84) | +0.203 (19) | Y/Y | 4.86 / 6.16 | nYYnY |
| T30_TP0.25 | -0.016 (0.20) | -0.031 | -0.010 | +0.036 (0.90) | +0.002 (0.50) | +0.025 (0.85) | +0.218 (19) | Y/Y | 4.41 / 3.54 | nYYnY |
| T30_TP0.3 | -0.020 (0.13) | -0.036 | -0.014 | +0.041 (0.93) | +0.004 (0.52) | +0.028 (0.88) | +0.129 (19) | Y/Y | 11.80 / 12.05 | nYYnY |
| T30_TP0.5 | -0.032 (0.01) | -0.036 | -0.030 | +0.018 (0.77) | -0.027 (0.19) | +0.003 (0.56) | +0.139 (19) | Y/n | 0.08 / 0.09 | nYnnY |
| T30_SL2 | +0.011 (0.70) | -0.027 | +0.024 | -0.016 (0.30) | -0.074 (0.09) | -0.035 (0.09) | +0.063 (19) | Y/Y | 0.46 / 0.59 | nYYnn |
| T30_SL1 | +0.010 (0.63) | -0.038 | +0.026 | +0.033 (0.85) | -0.096 (0.08) | -0.010 (0.38) | +0.168 (19) | n/n | 1.85 / 2.51 | nYnnn |
| T45_CLOSE | -0.004 (0.42) | +0.032 | -0.016 | +0.021 (0.76) | -0.042 (0.20) | -0.000 (0.50) | -0.026 (19) | n/n | 1.19 / 1.37 | nnnnn |
| T45_TP0 | -0.021 (0.06) | +0.005 | -0.030 | +0.002 (0.53) | -0.022 (0.25) | -0.006 (0.34) | +0.191 (19) | Y/n | 0.11 / 0.17 | nYnnn |
| T45_TP0.1 | -0.020 (0.06) | -0.008 | -0.024 | -0.013 (0.11) | -0.013 (0.33) | -0.013 (0.16) | +0.203 (19) | Y/Y | 0.09 / 0.13 | nYYYY |
| T45_TP0.25 | -0.020 (0.03) | -0.023 | -0.020 | -0.014 (0.09) | +0.000 (0.49) | -0.009 (0.23) | +0.218 (19) | Y/Y | 0.08 / 0.13 | nYYYY |
| T45_TP0.3 | -0.018 (0.04) | -0.019 | -0.018 | -0.014 (0.07) | -0.000 (0.48) | -0.010 (0.21) | +0.129 (19) | Y/Y | 0.09 / 0.13 | nYYYY |
| T45_TP0.5 | -0.020 (0.00) | -0.032 | -0.016 | -0.012 (0.08) | +0.009 (0.59) | -0.005 (0.33) | +0.139 (19) | Y/Y | 0.07 / 0.11 | nYYnY |
| T45_SL2 | +0.014 (0.81) | +0.029 | +0.009 | -0.003 (0.46) | -0.031 (0.23) | -0.012 (0.28) | -0.025 (19) | Y/Y | 0.53 / 0.77 | nnYnY |
| T45_SL1 | +0.001 (0.50) | +0.032 | -0.010 | +0.023 (0.80) | -0.045 (0.18) | -0.000 (0.50) | -0.025 (19) | n/n | 6.11 / 8.70 | nnnnY |
| T60_CLOSE | -0.008 (0.28) | +0.011 | -0.015 | -0.005 (0.39) | -0.008 (0.40) | -0.006 (0.36) | +0.057 (19) | Y/n | 0.37 / 0.32 | nYnnn |
| T60_TP0 | -0.004 (0.29) | +0.005 | -0.007 | +0.005 (0.70) | -0.024 (0.14) | -0.005 (0.31) | +0.078 (19) | n/n | 0.38 / 0.25 | nYnnn |
| T60_TP0.1 | -0.001 (0.43) | +0.007 | -0.004 | +0.007 (0.78) | -0.025 (0.10) | -0.004 (0.33) | +0.084 (19) | n/n | 0.72 / 0.48 | nYnnn |
| T60_TP0.25 | -0.006 (0.14) | -0.013 | -0.003 | +0.005 (0.75) | -0.016 (0.19) | -0.002 (0.40) | +0.092 (19) | Y/n | 0.31 / 0.21 | nYnnY |
| T60_TP0.3 | -0.005 (0.17) | -0.011 | -0.003 | +0.004 (0.70) | -0.019 (0.13) | -0.004 (0.32) | +0.000 (19) | Y/n | 0.28 / 0.20 | nYnnY |
| T60_TP0.5 | -0.005 (0.12) | -0.002 | -0.006 | +0.003 (0.66) | -0.017 (0.09) | -0.004 (0.28) | +0.000 (19) | Y/Y | 0.22 / 0.17 | nYYnY |
| T60_SL2 | +0.001 (0.53) | +0.002 | +0.001 | -0.007 (0.24) | +0.002 (0.54) | -0.004 (0.36) | -0.001 (19) | n/n | 2.19 / 2.19 | nnnnY |
| T60_SL1 | -0.010 (0.23) | +0.014 | -0.018 | -0.007 (0.30) | -0.007 (0.39) | -0.007 (0.30) | +0.052 (19) | Y/n | 0.34 / 0.27 | nYnnn |

## Cells vs today — ruler SL (③ backtest per year · ② master)

| cell | 2025 Δ (P) | 2025 top-50 Δ | 2025 rest Δ | 2026-H1 Δ (P) | 2026-H2 Δ (P) | 2026 Δ (P) | master Δ (N) | LOMO 25/26 | top day / pair share (pooled) | gates G1–G5 |
|---|---|---|---|---|---|---|---|---|---|---|
| T15_CLOSE | -0.023 (0.27) | -0.047 | -0.015 | +0.002 (0.51) | -0.096 (0.15) | -0.030 (0.24) | -0.085 (19) | Y/Y | 0.19 / 0.19 | nnYYY |
| T15_TP0 | -0.050 (0.04) | -0.064 | -0.045 | -0.012 (0.36) | -0.034 (0.30) | -0.019 (0.29) | -0.015 (19) | Y/Y | 0.09 / 0.09 | nnYYY |
| T15_TP0.1 | -0.055 (0.02) | -0.070 | -0.050 | -0.006 (0.42) | -0.024 (0.36) | -0.012 (0.36) | +0.008 (19) | Y/Y | 0.08 / 0.08 | nYYYY |
| T15_TP0.25 | -0.048 (0.03) | -0.075 | -0.038 | +0.007 (0.57) | -0.006 (0.46) | +0.002 (0.52) | +0.039 (19) | Y/n | 0.10 / 0.10 | nYnnY |
| T15_TP0.3 | -0.052 (0.02) | -0.075 | -0.044 | +0.014 (0.65) | +0.001 (0.50) | +0.010 (0.62) | -0.045 (19) | Y/n | 0.09 / 0.09 | nnnnY |
| T15_TP0.5 | -0.056 (0.00) | -0.077 | -0.048 | +0.013 (0.65) | -0.003 (0.46) | +0.008 (0.60) | -0.007 (19) | Y/n | 0.07 / 0.07 | nnnnY |
| T15_SL2 | +0.057 (0.97) | +0.031 | +0.066 | +0.060 (0.94) | -0.011 (0.44) | +0.036 (0.86) | +0.192 (19) | Y/Y | 0.05 / 0.12 | nYYYY |
| T15_SL1 | +0.071 (0.97) | +0.054 | +0.077 | +0.097 (0.99) | +0.002 (0.51) | +0.065 (0.95) | +0.061 (19) | Y/Y | 0.05 / 0.11 | YYYYY |
| T20_CLOSE | -0.023 (0.25) | -0.060 | -0.011 | +0.001 (0.50) | -0.112 (0.08) | -0.037 (0.17) | +0.059 (19) | Y/Y | 0.16 / 0.15 | nYYYY |
| T20_TP0 | -0.052 (0.01) | -0.045 | -0.054 | +0.026 (0.79) | -0.093 (0.04) | -0.014 (0.31) | +0.039 (19) | Y/n | 0.08 / 0.09 | nYnYY |
| T20_TP0.1 | -0.050 (0.01) | -0.054 | -0.049 | +0.028 (0.81) | -0.078 (0.07) | -0.007 (0.39) | +0.062 (19) | Y/n | 0.08 / 0.09 | nYnnY |
| T20_TP0.25 | -0.043 (0.03) | -0.065 | -0.035 | +0.034 (0.87) | -0.083 (0.04) | -0.005 (0.42) | +0.093 (19) | Y/n | 0.08 / 0.10 | nYnnY |
| T20_TP0.3 | -0.048 (0.01) | -0.067 | -0.042 | +0.040 (0.91) | -0.077 (0.05) | +0.001 (0.51) | +0.009 (19) | Y/n | 0.07 / 0.09 | nYnnY |
| T20_TP0.5 | -0.048 (0.00) | -0.058 | -0.044 | +0.039 (0.91) | -0.069 (0.05) | +0.003 (0.54) | +0.041 (19) | Y/n | 0.06 / 0.07 | nYnnY |
| T20_SL2 | +0.072 (1.00) | +0.038 | +0.083 | +0.017 (0.68) | -0.021 (0.37) | +0.004 (0.57) | +0.220 (19) | Y/n | 0.06 / 0.08 | nYnnY |
| T20_SL1 | +0.078 (0.99) | +0.061 | +0.084 | +0.062 (0.96) | -0.005 (0.48) | +0.039 (0.87) | +0.305 (19) | Y/Y | 0.06 / 0.08 | nYYYY |
| T30_CLOSE | -0.028 (0.15) | -0.080 | -0.011 | +0.020 (0.71) | -0.095 (0.07) | -0.019 (0.28) | +0.175 (19) | Y/Y | 0.16 / 0.16 | nYYYY |
| T30_TP0 | -0.028 (0.07) | -0.021 | -0.030 | +0.032 (0.86) | -0.018 (0.32) | +0.015 (0.74) | +0.191 (19) | Y/n | 0.22 / 0.24 | nYnnY |
| T30_TP0.1 | -0.019 (0.15) | -0.021 | -0.019 | +0.031 (0.86) | -0.005 (0.44) | +0.019 (0.79) | +0.203 (19) | Y/Y | 0.55 / 0.59 | nYYnY |
| T30_TP0.25 | -0.019 (0.14) | -0.036 | -0.013 | +0.031 (0.87) | -0.003 (0.45) | +0.020 (0.81) | +0.218 (19) | Y/Y | 0.59 / 0.62 | nYYnY |
| T30_TP0.3 | -0.023 (0.08) | -0.041 | -0.017 | +0.035 (0.91) | -0.002 (0.47) | +0.023 (0.85) | +0.129 (19) | Y/Y | 0.43 / 0.44 | nYYYY |
| T30_TP0.5 | -0.033 (0.01) | -0.039 | -0.031 | +0.015 (0.73) | -0.028 (0.17) | +0.000 (0.50) | +0.139 (19) | Y/n | 0.08 / 0.08 | nYnnY |
| T30_SL2 | +0.038 (0.97) | +0.012 | +0.047 | +0.009 (0.62) | -0.042 (0.21) | -0.008 (0.39) | +0.102 (19) | Y/n | 0.14 / 0.16 | nYnnY |
| T30_SL1 | +0.050 (0.98) | +0.020 | +0.061 | +0.065 (0.99) | -0.040 (0.27) | +0.030 (0.84) | +0.207 (19) | Y/Y | 0.07 / 0.11 | nYYYY |
| T45_CLOSE | -0.018 (0.17) | +0.011 | -0.028 | +0.010 (0.64) | -0.055 (0.13) | -0.012 (0.32) | -0.045 (19) | Y/n | 0.18 / 0.20 | nnnnn |
| T45_TP0 | -0.022 (0.04) | -0.000 | -0.030 | +0.001 (0.52) | -0.024 (0.22) | -0.007 (0.31) | +0.191 (19) | Y/Y | 0.10 / 0.15 | nYYnY |
| T45_TP0.1 | -0.021 (0.05) | -0.011 | -0.024 | -0.013 (0.11) | -0.015 (0.31) | -0.014 (0.14) | +0.203 (19) | Y/Y | 0.08 / 0.13 | nYYYY |
| T45_TP0.25 | -0.021 (0.02) | -0.025 | -0.020 | -0.014 (0.09) | -0.001 (0.47) | -0.010 (0.21) | +0.218 (19) | Y/Y | 0.08 / 0.12 | nYYYY |
| T45_TP0.3 | -0.019 (0.04) | -0.021 | -0.018 | -0.014 (0.07) | -0.002 (0.46) | -0.010 (0.19) | +0.129 (19) | Y/Y | 0.08 / 0.12 | nYYYY |
| T45_TP0.5 | -0.020 (0.00) | -0.032 | -0.016 | -0.012 (0.08) | +0.007 (0.57) | -0.005 (0.30) | +0.139 (19) | Y/Y | 0.07 / 0.11 | nYYYY |
| T45_SL2 | +0.036 (0.99) | +0.060 | +0.028 | +0.018 (0.79) | -0.008 (0.42) | +0.009 (0.69) | +0.014 (19) | Y/Y | 0.07 / 0.12 | nYYnY |
| T45_SL1 | +0.034 (0.96) | +0.079 | +0.019 | +0.051 (0.97) | -0.004 (0.47) | +0.032 (0.92) | +0.014 (19) | Y/Y | 0.07 / 0.10 | YYYYY |
| T60_CLOSE | -0.014 (0.15) | +0.002 | -0.020 | -0.008 (0.30) | -0.013 (0.35) | -0.010 (0.26) | +0.037 (19) | Y/n | 0.20 / 0.18 | nYnnn |
| T60_TP0 | -0.005 (0.27) | +0.004 | -0.007 | +0.005 (0.70) | -0.024 (0.14) | -0.005 (0.31) | +0.078 (19) | n/n | 0.36 / 0.23 | nYnnn |
| T60_TP0.1 | -0.001 (0.41) | +0.006 | -0.004 | +0.007 (0.78) | -0.025 (0.10) | -0.004 (0.33) | +0.084 (19) | n/n | 0.64 / 0.43 | nYnnn |
| T60_TP0.25 | -0.006 (0.14) | -0.013 | -0.003 | +0.005 (0.75) | -0.016 (0.19) | -0.002 (0.40) | +0.092 (19) | Y/n | 0.31 / 0.21 | nYnnY |
| T60_TP0.3 | -0.005 (0.17) | -0.011 | -0.003 | +0.004 (0.70) | -0.019 (0.13) | -0.004 (0.32) | +0.000 (19) | Y/n | 0.28 / 0.20 | nYnnY |
| T60_TP0.5 | -0.005 (0.12) | -0.002 | -0.006 | +0.003 (0.66) | -0.017 (0.09) | -0.004 (0.28) | +0.000 (19) | Y/Y | 0.22 / 0.17 | nYYnY |
| T60_SL2 | +0.019 (0.97) | +0.029 | +0.016 | +0.007 (0.76) | +0.016 (0.76) | +0.010 (0.85) | +0.038 (19) | Y/Y | 0.09 / 0.16 | nYYYY |
| T60_SL1 | +0.018 (0.89) | +0.052 | +0.006 | +0.018 (0.93) | +0.025 (0.79) | +0.020 (0.94) | +0.091 (19) | Y/Y | 0.12 / 0.16 | nYYYY |

## Two-sided: SAVED vs KILLED vs DEEPER — ruler S0 (n · Σ points)

| cell | 2025 saved | 2025 killed | 2025 deeper | 2026 saved | 2026 killed | 2026 deeper | master saved / killed / deeper |
|---|---|---|---|---|---|---|---|
| T15_CLOSE | 140 · +269.6 | 141 · -237.8 | 25 · -16.8 | 98 · +184.7 | 105 · -170.5 | 14 · -8.5 | 4 · +4.7 / 4 · -5.6 / 0 · +0.0 |
| T15_TP0 | 58 · +135.3 | 141 · -167.5 | 9 · -4.8 | 47 · +107.7 | 105 · -112.0 | 2 · -0.7 | 2 · +3.6 / 4 · -3.9 / 0 · +0.0 |
| T15_TP0.1 | 52 · +115.2 | 141 · -155.1 | 7 · -4.0 | 44 · +104.6 | 105 · -105.2 | 2 · -0.5 | 2 · +3.8 / 4 · -3.7 / 0 · +0.0 |
| T15_TP0.25 | 45 · +101.4 | 141 · -136.2 | 6 · -3.2 | 42 · +102.1 | 105 · -94.2 | 1 · -0.2 | 2 · +4.1 / 4 · -3.4 / 0 · +0.0 |
| T15_TP0.3 | 38 · +90.0 | 141 · -129.7 | 6 · -2.8 | 41 · +102.9 | 105 · -90.4 | 1 · -0.2 | 1 · +2.4 / 4 · -3.3 / 0 · +0.0 |
| T15_TP0.5 | 25 · +58.4 | 141 · -103.6 | 5 · -1.8 | 31 · +83.4 | 105 · -73.8 | 0 · +0.0 | 1 · +2.6 / 4 · -2.8 / 0 · +0.0 |
| T15_SL2 | 118 · +177.6 | 38 · -127.4 | 31 · -32.4 | 75 · +120.4 | 29 · -97.9 | 22 · -22.4 | 3 · +3.5 / 0 · +0.0 / 1 · -0.5 |
| T15_SL1 | 131 · +238.1 | 82 · -204.6 | 30 · -24.2 | 87 · +159.4 | 57 · -143.4 | 21 · -12.4 | 4 · +4.3 / 2 · -4.7 / 0 · +0.0 |
| T20_CLOSE | 122 · +226.3 | 111 · -189.3 | 28 · -27.8 | 77 · +137.6 | 75 · -129.5 | 17 · -13.2 | 3 · +5.8 / 3 · -3.2 / 1 · -0.7 |
| T20_TP0 | 46 · +96.1 | 111 · -133.3 | 9 · -4.6 | 40 · +80.9 | 75 · -84.0 | 2 · -0.6 | 2 · +3.6 / 3 · -2.9 / 0 · +0.0 |
| T20_TP0.1 | 44 · +85.9 | 111 · -123.4 | 7 · -4.0 | 38 · +79.2 | 75 · -78.8 | 2 · -0.5 | 2 · +3.8 / 3 · -2.7 / 0 · +0.0 |
| T20_TP0.25 | 36 · +76.2 | 111 · -108.2 | 6 · -3.2 | 34 · +71.3 | 75 · -70.5 | 1 · -0.2 | 2 · +4.1 / 3 · -2.4 / 0 · +0.0 |
| T20_TP0.3 | 30 · +65.7 | 111 · -103.1 | 6 · -2.8 | 33 · +71.7 | 75 · -67.5 | 1 · -0.2 | 1 · +2.4 / 3 · -2.3 / 0 · +0.0 |
| T20_TP0.5 | 20 · +43.0 | 111 · -82.1 | 5 · -1.8 | 25 · +59.7 | 75 · -55.1 | 0 · +0.0 | 1 · +2.6 / 3 · -1.9 / 0 · +0.0 |
| T20_SL2 | 101 · +149.9 | 25 · -83.5 | 32 · -30.6 | 57 · +87.3 | 24 · -80.9 | 21 · -23.8 | 3 · +4.1 / 0 · +0.0 / 1 · -0.7 |
| T20_SL1 | 114 · +201.4 | 58 · -145.5 | 32 · -31.6 | 67 · +119.0 | 42 · -109.2 | 23 · -16.6 | 3 · +5.7 / 0 · +0.0 / 1 · -0.7 |
| T30_CLOSE | 89 · +145.8 | 72 · -122.7 | 31 · -29.9 | 60 · +98.7 | 48 · -87.5 | 15 · -11.9 | 3 · +3.7 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0 | 37 · +69.6 | 72 · -86.8 | 9 · -4.3 | 34 · +68.2 | 48 · -54.1 | 2 · -0.7 | 2 · +3.6 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.1 | 37 · +70.1 | 72 · -80.5 | 7 · -3.7 | 33 · +66.5 | 48 · -50.7 | 2 · -0.5 | 2 · +3.8 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.25 | 30 · +60.1 | 72 · -71.0 | 6 · -3.0 | 29 · +60.7 | 48 · -45.1 | 1 · -0.2 | 2 · +4.1 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.3 | 25 · +52.8 | 72 · -67.8 | 6 · -2.8 | 28 · +60.9 | 48 · -43.2 | 1 · -0.2 | 1 · +2.4 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.5 | 16 · +27.2 | 72 · -54.0 | 5 · -1.8 | 17 · +37.4 | 48 · -35.3 | 0 · +0.0 | 1 · +2.6 / 0 · +0.0 / 0 · +0.0 |
| T30_SL2 | 71 · +90.8 | 15 · -50.4 | 30 · -30.2 | 38 · +54.9 | 16 · -54.7 | 21 · -22.0 | 2 · +1.7 / 0 · +0.0 / 1 · -0.5 |
| T30_SL1 | 81 · +128.9 | 33 · -85.9 | 35 · -34.1 | 47 · +79.8 | 25 · -69.6 | 24 · -16.6 | 3 · +3.2 / 0 · +0.0 / 0 · +0.0 |
| T45_CLOSE | 69 · +84.9 | 37 · -59.1 | 35 · -29.0 | 42 · +63.7 | 27 · -49.2 | 21 · -14.7 | 1 · +1.4 / 0 · +0.0 / 2 · -1.9 |
| T45_TP0 | 22 · +29.7 | 37 · -43.8 | 9 · -4.8 | 22 · +26.0 | 27 · -29.2 | 2 · -0.6 | 2 · +3.6 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.1 | 22 · +27.0 | 37 · -40.6 | 7 · -4.0 | 18 · +19.5 | 27 · -27.3 | 2 · -0.5 | 2 · +3.8 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.25 | 17 · +20.6 | 37 · -35.8 | 6 · -3.2 | 15 · +18.6 | 27 · -24.0 | 1 · -0.2 | 2 · +4.1 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.3 | 16 · +20.6 | 37 · -34.1 | 6 · -2.8 | 13 · +17.2 | 27 · -22.9 | 1 · -0.2 | 1 · +2.4 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.5 | 11 · +11.3 | 37 · -27.5 | 5 · -1.8 | 11 · +15.8 | 27 · -18.8 | 0 · +0.0 | 1 · +2.6 / 0 · +0.0 / 0 · +0.0 |
| T45_SL2 | 51 · +59.7 | 7 · -23.3 | 30 · -23.5 | 27 · +37.7 | 7 · -23.9 | 21 · -21.4 | 1 · +1.4 / 0 · +0.0 / 2 · -1.9 |
| T45_SL1 | 62 · +79.4 | 19 · -48.1 | 36 · -30.6 | 34 · +53.8 | 13 · -36.1 | 26 · -17.7 | 1 · +1.4 / 0 · +0.0 / 2 · -1.9 |
| T60_CLOSE | 51 · +45.7 | 13 · -24.3 | 34 · -28.5 | 34 · +29.2 | 13 · -19.8 | 19 · -13.0 | 2 · +2.1 / 0 · +0.0 / 1 · -1.0 |
| T60_TP0 | 17 · +16.3 | 13 · -15.4 | 9 · -4.6 | 16 · +12.2 | 13 · -14.5 | 2 · -0.7 | 1 · +1.5 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.1 | 18 · +17.2 | 13 · -14.2 | 7 · -3.9 | 14 · +11.7 | 13 · -13.6 | 2 · -0.5 | 1 · +1.6 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.25 | 13 · +10.3 | 13 · -12.4 | 6 · -3.1 | 12 · +11.3 | 13 · -12.2 | 1 · -0.2 | 1 · +1.7 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.3 | 12 · +10.1 | 13 · -11.8 | 6 · -2.8 | 10 · +9.7 | 13 · -11.7 | 1 · -0.2 | 0 · +0.0 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.5 | 9 · +6.7 | 13 · -9.4 | 5 · -1.8 | 8 · +7.4 | 13 · -9.6 | 0 · +0.0 | 0 · +0.0 / 0 · +0.0 / 0 · +0.0 |
| T60_SL2 | 34 · +26.2 | 2 · -6.7 | 24 · -18.7 | 16 · +12.9 | 1 · -3.3 | 13 · -12.1 | 2 · +1.0 / 0 · +0.0 / 1 · -1.0 |
| T60_SL1 | 44 · +39.8 | 8 · -20.2 | 32 · -28.8 | 24 · +21.4 | 4 · -10.1 | 23 · -15.6 | 2 · +2.0 / 0 · +0.0 / 1 · -1.0 |

## Two-sided: SAVED vs KILLED vs DEEPER — ruler SL (n · Σ points)

| cell | 2025 saved | 2025 killed | 2025 deeper | 2026 saved | 2026 killed | 2026 deeper | master saved / killed / deeper |
|---|---|---|---|---|---|---|---|
| T15_CLOSE | 139 · +234.1 | 141 · -237.8 | 26 · -16.9 | 98 · +160.3 | 105 · -170.5 | 14 · -8.5 | 4 · +3.9 / 4 · -5.6 / 0 · +0.0 |
| T15_TP0 | 58 · +127.5 | 141 · -167.5 | 9 · -4.8 | 47 · +100.7 | 105 · -112.0 | 2 · -0.7 | 2 · +3.6 / 4 · -3.9 / 0 · +0.0 |
| T15_TP0.1 | 52 · +109.2 | 141 · -155.1 | 7 · -4.0 | 44 · +98.0 | 105 · -105.2 | 2 · -0.5 | 2 · +3.8 / 4 · -3.7 / 0 · +0.0 |
| T15_TP0.25 | 45 · +96.6 | 141 · -136.2 | 6 · -3.2 | 42 · +95.9 | 105 · -94.2 | 1 · -0.2 | 2 · +4.1 / 4 · -3.4 / 0 · +0.0 |
| T15_TP0.3 | 38 · +85.6 | 141 · -129.7 | 6 · -2.8 | 41 · +96.6 | 105 · -90.4 | 1 · -0.2 | 1 · +2.4 / 4 · -3.3 / 0 · +0.0 |
| T15_TP0.5 | 25 · +55.4 | 141 · -103.6 | 5 · -1.8 | 31 · +78.6 | 105 · -73.8 | 0 · +0.0 | 1 · +2.6 / 4 · -2.8 / 0 · +0.0 |
| T15_SL2 | 126 · +187.3 | 38 · -113.4 | 23 · -22.5 | 78 · +123.9 | 29 · -87.2 | 19 · -14.4 | 3 · +3.8 / 0 · +0.0 / 1 · -0.2 |
| T15_SL1 | 142 · +253.2 | 82 · -174.3 | 19 · -15.3 | 96 · +169.1 | 57 · -122.4 | 12 · -6.5 | 4 · +5.1 / 2 · -3.9 / 0 · +0.0 |
| T20_CLOSE | 121 · +196.4 | 111 · -189.3 | 29 · -27.9 | 77 · +119.8 | 75 · -129.5 | 17 · -13.2 | 3 · +5.1 / 3 · -3.2 / 1 · -0.7 |
| T20_TP0 | 46 · +91.3 | 111 · -133.3 | 9 · -4.6 | 40 · +76.1 | 75 · -84.0 | 2 · -0.6 | 2 · +3.6 / 3 · -2.9 / 0 · +0.0 |
| T20_TP0.1 | 44 · +82.2 | 111 · -123.4 | 7 · -4.0 | 38 · +74.8 | 75 · -78.8 | 2 · -0.5 | 2 · +3.8 / 3 · -2.7 / 0 · +0.0 |
| T20_TP0.25 | 36 · +72.8 | 111 · -108.2 | 6 · -3.2 | 34 · +67.6 | 75 · -70.5 | 1 · -0.2 | 2 · +4.1 / 3 · -2.4 / 0 · +0.0 |
| T20_TP0.3 | 30 · +62.8 | 111 · -103.1 | 6 · -2.8 | 33 · +68.0 | 75 · -67.5 | 1 · -0.2 | 1 · +2.4 / 3 · -2.3 / 0 · +0.0 |
| T20_TP0.5 | 20 · +41.2 | 111 · -82.1 | 5 · -1.8 | 25 · +56.8 | 75 · -55.1 | 0 · +0.0 | 1 · +2.6 / 3 · -1.9 / 0 · +0.0 |
| T20_SL2 | 109 · +158.8 | 25 · -74.2 | 24 · -20.3 | 59 · +90.7 | 24 · -72.0 | 19 · -16.1 | 3 · +4.5 / 0 · +0.0 / 1 · -0.3 |
| T20_SL1 | 121 · +214.3 | 58 · -124.0 | 25 · -20.5 | 76 · +127.9 | 42 · -93.6 | 14 · -10.0 | 3 · +6.1 / 0 · +0.0 / 1 · -0.3 |
| T30_CLOSE | 88 · +127.1 | 72 · -122.7 | 32 · -30.0 | 60 · +88.0 | 48 · -87.5 | 15 · -11.9 | 3 · +3.3 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0 | 37 · +66.3 | 72 · -86.8 | 9 · -4.3 | 34 · +64.1 | 48 · -54.1 | 2 · -0.7 | 2 · +3.6 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.1 | 37 · +66.8 | 72 · -80.5 | 7 · -3.7 | 33 · +62.8 | 48 · -50.7 | 2 · -0.5 | 2 · +3.8 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.25 | 30 · +57.1 | 72 · -71.0 | 6 · -3.0 | 29 · +57.4 | 48 · -45.1 | 1 · -0.2 | 2 · +4.1 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.3 | 25 · +49.8 | 72 · -67.8 | 6 · -2.8 | 28 · +57.6 | 48 · -43.2 | 1 · -0.2 | 1 · +2.4 / 0 · +0.0 / 0 · +0.0 |
| T30_TP0.5 | 16 · +26.1 | 72 · -54.0 | 5 · -1.8 | 17 · +35.6 | 48 · -35.3 | 0 · +0.0 | 1 · +2.6 / 0 · +0.0 / 0 · +0.0 |
| T30_SL2 | 79 · +99.8 | 15 · -44.9 | 22 · -20.7 | 42 · +58.6 | 16 · -48.8 | 17 · -14.6 | 2 · +2.1 / 0 · +0.0 / 1 · -0.2 |
| T30_SL1 | 92 · +142.1 | 33 · -73.7 | 24 · -23.2 | 58 · +88.2 | 25 · -60.3 | 13 · -9.4 | 3 · +3.9 / 0 · +0.0 / 0 · +0.0 |
| T45_CLOSE | 65 · +72.6 | 37 · -59.1 | 39 · -29.7 | 42 · +56.7 | 27 · -49.2 | 21 · -14.7 | 1 · +1.0 / 0 · +0.0 / 2 · -1.9 |
| T45_TP0 | 22 · +28.6 | 37 · -43.8 | 9 · -4.8 | 22 · +25.3 | 27 · -29.2 | 2 · -0.6 | 2 · +3.6 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.1 | 22 · +26.2 | 37 · -40.6 | 7 · -4.0 | 18 · +19.1 | 27 · -27.3 | 2 · -0.5 | 2 · +3.8 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.25 | 17 · +20.2 | 37 · -35.8 | 6 · -3.2 | 15 · +18.2 | 27 · -24.0 | 1 · -0.2 | 2 · +4.1 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.3 | 16 · +20.3 | 37 · -34.1 | 6 · -2.8 | 13 · +16.9 | 27 · -22.9 | 1 · -0.2 | 1 · +2.4 / 0 · +0.0 / 0 · +0.0 |
| T45_TP0.5 | 11 · +11.3 | 37 · -27.5 | 5 · -1.8 | 11 · +15.4 | 27 · -18.8 | 0 · +0.0 | 1 · +2.6 / 0 · +0.0 / 0 · +0.0 |
| T45_SL2 | 60 · +67.4 | 7 · -20.7 | 21 · -14.2 | 29 · +40.7 | 7 · -21.3 | 19 · -13.7 | 1 · +1.4 / 0 · +0.0 / 2 · -1.2 |
| T45_SL1 | 73 · +91.6 | 19 · -41.1 | 25 · -19.5 | 42 · +60.9 | 13 · -31.3 | 18 · -9.7 | 1 · +1.4 / 0 · +0.0 / 2 · -1.2 |
| T60_CLOSE | 50 · +39.9 | 13 · -24.3 | 35 · -28.6 | 33 · +26.8 | 13 · -19.8 | 20 · -13.2 | 2 · +1.7 / 0 · +0.0 / 1 · -1.0 |
| T60_TP0 | 17 · +15.9 | 13 · -15.4 | 9 · -4.6 | 16 · +12.2 | 13 · -14.5 | 2 · -0.7 | 1 · +1.5 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.1 | 18 · +16.8 | 13 · -14.2 | 7 · -3.9 | 14 · +11.7 | 13 · -13.6 | 2 · -0.5 | 1 · +1.6 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.25 | 13 · +10.3 | 13 · -12.4 | 6 · -3.1 | 12 · +11.3 | 13 · -12.2 | 1 · -0.2 | 1 · +1.7 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.3 | 12 · +10.1 | 13 · -11.8 | 6 · -2.8 | 10 · +9.7 | 13 · -11.7 | 1 · -0.2 | 0 · +0.0 / 0 · +0.0 / 0 · +0.0 |
| T60_TP0.5 | 9 · +6.7 | 13 · -9.4 | 5 · -1.8 | 8 · +7.4 | 13 · -9.6 | 0 · +0.0 | 0 · +0.0 / 0 · +0.0 / 0 · +0.0 |
| T60_SL2 | 42 · +34.6 | 2 · -5.9 | 16 · -11.5 | 18 · +16.5 | 1 · -2.9 | 11 · -7.5 | 2 · +1.4 / 0 · +0.0 / 1 · -0.7 |
| T60_SL1 | 54 · +52.6 | 8 · -17.2 | 22 · -19.4 | 35 · +29.8 | 4 · -8.6 | 13 · -8.8 | 2 · +2.4 / 0 · +0.0 / 1 · -0.7 |

## Master per batch (ruler S0 / SL): N · today avg · Δ

| cell | B19 S0 | B19 SL | B20 S0 | B20 SL | B21 S0 | B21 SL |
|---|---|---|---|---|---|---|
| T15_CLOSE | 14 · +0.45 · -0.237 | 14 · +0.31 · -0.263 | 3 · -0.50 · +0.810 | 3 · -0.54 · +0.687 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_TP0 | 14 · +0.45 · -0.020 | 14 · +0.31 · -0.020 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_TP0.1 | 14 · +0.45 · +0.011 | 14 · +0.31 · +0.011 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_TP0.25 | 14 · +0.45 · +0.053 | 14 · +0.31 · +0.053 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_TP0.3 | 14 · +0.45 · -0.061 | 14 · +0.31 · -0.061 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_TP0.5 | 14 · +0.45 · -0.010 | 14 · +0.31 · -0.010 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_SL2 | 14 · +0.45 · +0.065 | 14 · +0.31 · +0.118 | 3 · -0.50 · +0.667 | 3 · -0.54 · +0.667 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T15_SL1 | 14 · +0.45 · -0.196 | 14 · +0.31 · -0.090 | 3 · -0.50 · +0.810 | 3 · -0.54 · +0.810 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_CLOSE | 14 · +0.45 · -0.041 | 14 · +0.31 · -0.068 | 3 · -0.50 · +0.810 | 3 · -0.54 · +0.687 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_TP0 | 14 · +0.45 · +0.053 | 14 · +0.31 · +0.053 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_TP0.1 | 14 · +0.45 · +0.084 | 14 · +0.31 · +0.084 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_TP0.25 | 14 · +0.45 · +0.126 | 14 · +0.31 · +0.126 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_TP0.3 | 14 · +0.45 · +0.012 | 14 · +0.31 · +0.012 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_TP0.5 | 14 · +0.45 · +0.055 | 14 · +0.31 · +0.055 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_SL2 | 14 · +0.45 · +0.103 | 14 · +0.31 · +0.156 | 3 · -0.50 · +0.667 | 3 · -0.54 · +0.667 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T20_SL1 | 14 · +0.45 · +0.187 | 14 · +0.31 · +0.240 | 3 · -0.50 · +0.810 | 3 · -0.54 · +0.810 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_CLOSE | 14 · +0.45 · +0.149 | 14 · +0.31 · +0.149 | 3 · -0.50 · +0.533 | 3 · -0.54 · +0.410 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_TP0 | 14 · +0.45 · +0.259 | 14 · +0.31 · +0.259 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_TP0.1 | 14 · +0.45 · +0.275 | 14 · +0.31 · +0.275 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_TP0.25 | 14 · +0.45 · +0.296 | 14 · +0.31 · +0.296 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_TP0.3 | 14 · +0.45 · +0.175 | 14 · +0.31 · +0.175 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_TP0.5 | 14 · +0.45 · +0.189 | 14 · +0.31 · +0.189 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_SL2 | 14 · +0.45 · -0.029 | 14 · +0.31 · +0.024 | 3 · -0.50 · +0.533 | 3 · -0.54 · +0.533 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T30_SL1 | 14 · +0.45 · +0.114 | 14 · +0.31 · +0.167 | 3 · -0.50 · +0.533 | 3 · -0.54 · +0.533 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_CLOSE | 14 · +0.45 · -0.136 | 14 · +0.31 · -0.136 | 3 · -0.50 · +0.472 | 3 · -0.54 · +0.349 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_TP0 | 14 · +0.45 · +0.259 | 14 · +0.31 · +0.259 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_TP0.1 | 14 · +0.45 · +0.275 | 14 · +0.31 · +0.275 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_TP0.25 | 14 · +0.45 · +0.296 | 14 · +0.31 · +0.296 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_TP0.3 | 14 · +0.45 · +0.175 | 14 · +0.31 · +0.175 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_TP0.5 | 14 · +0.45 · +0.189 | 14 · +0.31 · +0.189 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_SL2 | 14 · +0.45 · -0.135 | 14 · +0.31 · -0.082 | 3 · -0.50 · +0.472 | 3 · -0.54 · +0.472 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T45_SL1 | 14 · +0.45 · -0.135 | 14 · +0.31 · -0.082 | 3 · -0.50 · +0.472 | 3 · -0.54 · +0.472 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_CLOSE | 14 · +0.45 · +0.016 | 14 · +0.31 · +0.016 | 3 · -0.50 · +0.287 | 3 · -0.54 · +0.164 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_TP0 | 14 · +0.45 · +0.105 | 14 · +0.31 · +0.105 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_TP0.1 | 14 · +0.45 · +0.114 | 14 · +0.31 · +0.114 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_TP0.25 | 14 · +0.45 · +0.125 | 14 · +0.31 · +0.125 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_TP0.3 | 14 · +0.45 · +0.000 | 14 · +0.31 · +0.000 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_TP0.5 | 14 · +0.45 · +0.000 | 14 · +0.31 · +0.000 | 3 · -0.50 · +0.000 | 3 · -0.54 · +0.000 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_SL2 | 14 · +0.45 · -0.063 | 14 · +0.31 · -0.010 | 3 · -0.50 · +0.287 | 3 · -0.54 · +0.287 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |
| T60_SL1 | 14 · +0.45 · +0.009 | 14 · +0.31 · +0.061 | 3 · -0.50 · +0.287 | 3 · -0.54 · +0.287 | 2 · +1.25 · +0.000 | 2 · +1.08 · +0.000 |

## Regime read (hypothesis only) — regime-dependent cells: ['T15_TP0.25', 'T15_TP0.3', 'T15_TP0.5', 'T20_TP0.25', 'T20_TP0.3', 'T20_TP0.5', 'T30_TP0', 'T30_TP0.1', 'T30_TP0.25', 'T30_TP0.3', 'T30_TP0.5']

Joined 1516 / 1516 BT fills to their entry stamps. Split at SIGN granularity first (signed vars at 0; RSI 50; ADX 25; bull / bear % 50; above72 % 50; volume ratios 1; others the pooled median, stated). Day units: each (day, side) is one observation; Δ = mean of day means; 'nd' = days. A var is flagged only if the same side carries the higher Δ in BOTH years, that side is > 0 and the other ≤ 0 in both years, each side ≥ 8 days per year.

Day-block permutation null (40 shuffles of Δ across days within each year, same 550 tests): flags mean 47.0, p95 99, max 223; observed 56 → share of null ≥ observed 0.30.

550 split tests (cells × rulers × 25 vars). Flagged: 56. Under the null each split has roughly a 1-in-4 chance to be consistent in direction and a smaller chance to also clear the sign test — the flagged list is a SCREEN.

| cell | ruler | var (split) | 2025 hi Δ · nd | 2025 lo Δ · nd | 2026 hi Δ · nd | 2026 lo Δ · nd | share hi 2025 / 2026 (fills) |
|---|---|---|---|---|---|---|---|
| T15_TP0.25 | S0 | entry_btc_adx (25) | -0.070 · 199 | +0.001 · 241 | -0.020 · 168 | +0.029 · 184 | 42% / 45% |
| T15_TP0.25 | S0 | entry_btc_adx_prev (25) | -0.066 · 202 | +0.005 · 245 | -0.016 · 165 | +0.054 · 186 | 42% / 45% |
| T15_TP0.25 | S0 | entry_btc_eff72 (median 0.033) | -0.084 · 179 | +0.029 · 191 | -0.004 · 144 | +0.073 · 148 | 50% / 50% |
| T15_TP0.25 | S0 | entry_btc_ema50_100_gap_pct (0) | -0.028 · 229 | +0.003 · 183 | -0.000 · 160 | +0.027 · 151 | 59% / 54% |
| T15_TP0.25 | SL | entry_btc_eff72 (median 0.033) | -0.087 · 179 | +0.019 · 191 | -0.013 · 144 | +0.059 · 148 | 50% / 50% |
| T15_TP0.25 | SL | entry_frenzy_gvol (1) | -0.057 · 207 | +0.006 · 250 | -0.003 · 148 | +0.004 · 202 | 43% / 38% |
| T20_TP0.25 | S0 | entry_btc_eff72 (median 0.033) | -0.069 · 179 | +0.001 · 191 | -0.029 · 144 | +0.043 · 148 | 50% / 50% |
| T20_TP0.25 | SL | entry_btc_r72_pct (0) | -0.079 · 185 | +0.010 · 150 | -0.005 · 130 | +0.008 · 129 | 57% / 52% |
| T20_TP0.5 | S0 | entry_btc_above72_pct (50) | -0.065 · 202 | +0.014 · 143 | -0.002 · 148 | +0.043 · 119 | 59% / 59% |
| T20_TP0.5 | SL | entry_btc_above72_pct (50) | -0.066 · 202 | +0.009 · 143 | -0.007 · 148 | +0.037 · 119 | 59% / 59% |
| T30_TP0 | S0 | entry_btc_adx_prev (25) | -0.044 · 202 | +0.014 · 245 | -0.007 · 165 | +0.037 · 186 | 42% / 45% |
| T30_TP0 | S0 | entry_btc_rsi_1h (50) | -0.005 · 220 | +0.006 · 179 | -0.024 · 153 | +0.089 · 150 | 60% / 53% |
| T30_TP0 | S0 | entry_btc_eff72 (median 0.033) | -0.049 · 179 | +0.014 · 191 | -0.005 · 144 | +0.041 · 148 | 50% / 50% |
| T30_TP0 | S0 | entry_btc_above72_pct (50) | -0.034 · 202 | +0.028 · 143 | -0.012 · 148 | +0.062 · 119 | 59% / 59% |
| T30_TP0 | S0 | entry_frenzy_gvol (1) | -0.037 · 207 | +0.007 · 250 | -0.019 · 148 | +0.031 · 202 | 43% / 38% |
| T30_TP0 | SL | entry_btc_adx_prev (25) | -0.046 · 202 | +0.007 · 245 | -0.011 · 165 | +0.028 · 186 | 42% / 45% |
| T30_TP0 | SL | entry_btc_rsi_1h (50) | -0.011 · 220 | +0.002 · 179 | -0.028 · 153 | +0.078 · 150 | 60% / 53% |
| T30_TP0 | SL | entry_btc_r72_pct (0) | -0.078 · 185 | +0.036 · 150 | -0.001 · 130 | +0.027 · 129 | 57% / 52% |
| T30_TP0 | SL | entry_btc_eff72 (median 0.033) | -0.053 · 179 | +0.010 · 191 | -0.009 · 144 | +0.033 · 148 | 50% / 50% |
| T30_TP0 | SL | entry_btc_above72_pct (50) | -0.038 · 202 | +0.024 · 143 | -0.018 · 148 | +0.054 · 119 | 59% / 59% |
| T30_TP0 | SL | entry_frenzy_gvol (1) | -0.040 · 207 | +0.001 · 250 | -0.022 · 148 | +0.022 · 202 | 43% / 38% |
| T30_TP0.1 | S0 | entry_btc_above72_pct (50) | -0.019 · 202 | +0.025 · 143 | -0.001 · 148 | +0.061 · 119 | 59% / 59% |
| T30_TP0.1 | S0 | entry_frenzy_gvol (1) | -0.039 · 207 | +0.020 · 250 | -0.022 · 148 | +0.043 · 202 | 43% / 38% |
| T30_TP0.1 | SL | entry_btc_adx_prev (25) | -0.036 · 202 | +0.015 · 245 | -0.004 · 165 | +0.019 · 186 | 42% / 45% |
| T30_TP0.1 | SL | entry_btc_eff72 (median 0.033) | -0.037 · 179 | +0.012 · 191 | -0.002 · 144 | +0.037 · 148 | 50% / 50% |
| T30_TP0.1 | SL | entry_btc_above72_pct (50) | -0.023 · 202 | +0.020 · 143 | -0.007 · 148 | +0.054 · 119 | 59% / 59% |
| T30_TP0.1 | SL | entry_frenzy_gvol (1) | -0.041 · 207 | +0.014 · 250 | -0.024 · 148 | +0.034 · 202 | 43% / 38% |
| T30_TP0.25 | S0 | entry_btc_rsi_1h_prev (50) | -0.019 · 211 | +0.008 · 187 | -0.006 · 149 | +0.056 · 158 | 58% / 51% |
| T30_TP0.25 | S0 | entry_frenzy_gvol (1) | -0.033 · 207 | +0.022 · 250 | -0.012 · 148 | +0.045 · 202 | 43% / 38% |
| T30_TP0.25 | SL | entry_btc_rsi_1h (50) | -0.004 · 220 | +0.009 · 179 | -0.010 · 153 | +0.085 · 150 | 60% / 53% |
| T30_TP0.25 | SL | entry_btc_rsi_1h_prev (50) | -0.021 · 211 | +0.004 · 187 | -0.010 · 149 | +0.048 · 158 | 58% / 51% |
| T30_TP0.25 | SL | entry_btc_eff72 (median 0.033) | -0.037 · 179 | +0.017 · 191 | -0.001 · 144 | +0.044 · 148 | 50% / 50% |
| T30_TP0.25 | SL | entry_btc_above72_pct (50) | -0.025 · 202 | +0.030 · 143 | -0.002 · 148 | +0.062 · 119 | 59% / 59% |
| T30_TP0.25 | SL | entry_frenzy_gvol (1) | -0.035 · 207 | +0.016 · 250 | -0.015 · 148 | +0.037 · 202 | 43% / 38% |
| T30_TP0.3 | S0 | entry_btc_rsi_1h (50) | -0.008 · 220 | +0.015 · 179 | -0.003 · 153 | +0.102 · 150 | 60% / 53% |
| T30_TP0.3 | S0 | entry_btc_rsi_1h_prev (50) | -0.035 · 211 | +0.011 · 187 | -0.002 · 149 | +0.061 · 158 | 58% / 51% |
| T30_TP0.3 | S0 | entry_frenzy_gvol (1) | -0.034 · 207 | +0.012 · 250 | -0.008 · 148 | +0.050 · 202 | 43% / 38% |
| T30_TP0.3 | SL | entry_btc_rsi_1h (50) | -0.013 · 220 | +0.012 · 179 | -0.006 · 153 | +0.091 · 150 | 60% / 53% |
| T30_TP0.3 | SL | entry_btc_rsi_1h_prev (50) | -0.038 · 211 | +0.008 · 187 | -0.006 · 149 | +0.053 · 158 | 58% / 51% |
| T30_TP0.3 | SL | entry_frenzy_gvol (1) | -0.037 · 207 | +0.007 · 250 | -0.011 · 148 | +0.042 · 202 | 43% / 38% |
| T30_TP0.5 | S0 | entry_btc_ema20_slope (0) | -0.037 · 246 | +0.024 · 196 | -0.006 · 185 | +0.044 · 166 | 59% / 56% |
| T30_TP0.5 | S0 | entry_btc_rsi (50) | -0.038 · 242 | +0.031 · 198 | -0.011 · 186 | +0.031 · 170 | 58% / 54% |
| T30_TP0.5 | S0 | entry_btc_rsi_prev (50) | -0.033 · 239 | +0.024 · 190 | -0.005 · 184 | +0.046 · 167 | 58% / 57% |
| T30_TP0.5 | S0 | entry_bull_pct (50) | -0.045 · 231 | +0.012 · 223 | -0.000 · 151 | +0.019 · 199 | 51% / 42% |
| T30_TP0.5 | S0 | entry_bear_pct (50) | +0.026 · 184 | -0.049 · 259 | +0.036 · 148 | -0.011 · 197 | 35% / 38% |
| T30_TP0.5 | S0 | entry_btc_above72_pct (50) | -0.051 · 202 | +0.015 · 143 | -0.001 · 148 | +0.016 · 119 | 59% / 59% |
| T30_TP0.5 | S0 | entry_btc_1d_ret_pct (0) | -0.038 · 155 | +0.001 · 141 | -0.010 · 114 | +0.027 · 122 | 55% / 47% |
| T30_TP0.5 | S0 | entry_btc_rsi_closed (50) | -0.038 · 234 | +0.023 · 191 | -0.009 · 184 | +0.050 · 166 | 59% / 57% |
| T30_TP0.5 | SL | entry_btc_ema20_slope (0) | -0.040 · 246 | +0.022 · 196 | -0.009 · 185 | +0.039 · 166 | 59% / 56% |
| T30_TP0.5 | SL | entry_btc_rsi (50) | -0.040 · 242 | +0.029 · 198 | -0.014 · 186 | +0.026 · 170 | 58% / 54% |
| T30_TP0.5 | SL | entry_btc_rsi_prev (50) | -0.036 · 239 | +0.022 · 190 | -0.008 · 184 | +0.040 · 167 | 58% / 57% |
| T30_TP0.5 | SL | entry_bull_pct (50) | -0.046 · 231 | +0.010 · 223 | -0.004 · 151 | +0.015 · 199 | 51% / 42% |
| T30_TP0.5 | SL | entry_bear_pct (50) | +0.024 · 184 | -0.050 · 259 | +0.032 · 148 | -0.014 · 197 | 35% / 38% |
| T30_TP0.5 | SL | entry_btc_dist_from_ema13_pct (0) | -0.031 · 248 | +0.012 · 201 | -0.001 · 186 | +0.008 · 169 | 58% / 54% |
| T30_TP0.5 | SL | entry_btc_above72_pct (50) | -0.051 · 202 | +0.013 · 143 | -0.006 · 148 | +0.015 · 119 | 59% / 59% |
| T30_TP0.5 | SL | entry_btc_rsi_closed (50) | -0.040 · 234 | +0.022 · 191 | -0.012 · 184 | +0.044 · 166 | 59% / 57% |
