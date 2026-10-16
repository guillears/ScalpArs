# WILLY "low-VWAP + short-run" zone: filter or exit-modifier review (2026-10-16)

Research only. No engine, config, template or test was changed. Nothing was committed and the bot API was not touched. Backtest and
master were priced from the local cache only. The only Binance calls were 8 public 1m-kline requests (weight 8 in total) for the 8 live
fills, because the cache has no path for 10-15/16. This study has not been reviewed, so it supports no arm or ship.

**Hypothesis (frozen 2026-10-16 from the live batch, not re-fitted).**
- ZONE = a FRENZY_WILLY entry with `entry_frenzy_vs_vwap_pct ≤ 0.10` (price at or below the spike VWAP) AND
  `entry_frenzy_run_pct ≤ 10` (a short run).
- In the live batch, all 3 losers matched it (DUSK, STRK, ICNT) and 1 of 5 winners did (BAT).
- Each leg is also reported alone (VWAP_LEG, RUN_LEG).

**Pricing: today's WILLY exit.**
- Resting maker TP +1.25 net, booked at the line.
- Stop −4, booked at min(print, −4) − 0.2 slip.
- 90-minute cap, closed at market.
- Late TP +0.25 after 30 minutes (DL 297).
- The exit-modifier arm is the same exit, but zone fills take TP +1.0 instead of +1.25.

## Plain-English summary

**Verdict: NOISE. The zone is not a filter candidate, and not worth a watch line either. TP +1.0 for zone fills is also noise.**

1. **In the backtest the zone is an ordinary WILLY trade.**
   - On tick-priced fills (seed 1, 823 fills, 2 years) the zone makes **+0.024 %/fill (341 fills, WR 77.4 %)**. The rest makes
     **+0.028 (482 fills, WR 78.2 %)**.
   - WILLY's breakeven WR is 77.3 %. The zone's 77.4 % sits on it.
2. **The eras disagree in sign.**
   - **2025:** the zone is clearly *better* than the rest: +0.169 vs −0.212 (WR 79 % vs 74 %).
   - **2026-H1:** worse, −0.035 vs +0.169.
   - **2026-H2:** about level, −0.022 vs +0.027.
   - Refusing the zone would change each WILLY fill by **2025 −0.068 · 2026-H1 +0.014 · 2026-H2 +0.009 · all −0.010 %/fill**.
     None comes close to significance: P(Δ>0) ranges from 0.18 to 0.59.
3. **The expectancy filter bar fails in every era.**
   - ① WR below breakeven: fails in 2025 and overall; passes only by 0.4–1.3 points in 2026.
   - ② P(mean < 0) ≥ 0.95: fails everywhere. The best value is 0.58.
   - ③ Concentration: fails in 2026. STO and POL alone carry more than the whole net zone loss.
   - ④ N ≥ 15 and ≥ 8 days: passes.
   - Seeds 2 and 3 repeat every number within ±0.03.
4. **What the frozen cut actually picks up is the RUN leg, and it points the wrong way.**
   - Short-run WILLY fills (run ≤ 10) are *better* than long-run ones in every era: all +0.101 vs −0.068, refuse-Δ −0.056.
   - Leave-one-month-out keeps the sign in every month.
   - The VWAP leg alone is mildly negative: all −0.038 vs +0.146; 2026-H2 −0.141 vs +0.225, P(neg) 0.78. It fails ② everywhere, and in
     2025 the zone side beats the rest. It is a post-hoc leg and not a candidate.
5. **TP +1.0 on zone fills is a wash.**
   - Δ per WILLY fill: 2025 +0.016, 2026-H1 +0.003, 2026-H2 −0.019, all +0.001. P ranges from 0.25 to 0.64.
   - About 1 trade is rescued for every 9 TPs that get trimmed from +1.25 to +1.0 (all: 23 rescued vs 209 trimmed).
6. **Master (19 WILLY fills) refutes the zone.**
   - 9 master fills are in the zone. Under today's exit, **8 of the 9 win** (7 × +1.25, API3 late TP +0.25); only OGN stops.
   - Refusing the zone would cut the master from **+10.82 to +6.05 points (+$1,203 → +$333)**.
   - TP +1.0 would cut it to +9.07 points (+$981).
   - Batch by batch, refusing helps only B20 (OGN). It hurts B19 (7 of 7 zone fills win) and B21.
7. **The live batch (8 fills) is the only block where the zone loses.** Read it as refute-only.
   - Zone: 4 fills, 3 stops plus BAT. Refusing the zone: −5.52 → **+4.39 points (−$139 → +$108)**.
   - TP +1.0 would rescue DUSK (+1.0 at minute 26 on 1m) and trim BAT: −5.52 → −1.26 points (−$139 → −$25).
   - Under today's late-TP rule DUSK is already +0.25, so the 1m re-price of the batch under today's exit is −1.90 points (−$42).
   - 8 fills on 2 days, against 2 years of backtest and the master pointing the other way.

**Recommendation (operator decides): do not arm, and do not add a Scout line.**
- The zone that explains the live losers is a coincidence of a small batch, not a separator. Master plus live together give 13 zone
  fills: 10 wins (incl. DUSK late TP), Σ −2.1 points under today's exit, on 3 loser days.
- If a WILLY-loser line is wanted later, the only directionally consistent piece is the VWAP leg in 2026. Pre-register it fresh
  (vs_vwap ≤ 0.10 alone, frozen now) and judge it only on fills opened after today.

## Data and method

**Backtest**
- Source: `reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv`, FRENZY_WILLY, CLOSED, non-probe.
- Seeds 1 / 2 / 3: 1,518 / 1,512 / 1,508 fills, 0 duplicate (seed, pair, opened_at). The file is already warm-up trimmed.
- 6 UNRES fills were dropped.

**Walkers (reused, not rewritten)**
- `scout_frenzy_willy_exit.load_path` and `evaluate`, plus `scout_willy_timecap.evaluate_late`.
- Loader patches come from `study_tick_maker_lock_common` (2025 ticks and 1m).
- Network is blocked.

**Parity**
- The plain +1.25/−4/90 walk and the today's-exit late-TP walk both reproduce `reports/WILLY_DURATION_cell_rows.csv` on
  **1,535 / 1,535 fills**: base_pct, base_why, T30_TP0.25 and its reason all match. This is asserted in the script.

**Booking (applied to the walker's raw exit)**
- TP and LATE_TP book the line (resting maker, the same net target).
- STOP books min(print, −4) − 0.2.
- CAP books the last print (taker).

**Pricing split**
- *tick* = the path and both exits were decided on tick prints. This is the verdict block.
- *1m* = any 1m bar involved. It can only refute: the 1m stop-first artefact.
- 2025 is 228 tick fills against 668 on 1m. All 228 2025 tick fills are top-50 pair-days, so the 2025-top50 row equals the 2025 row.

**Statistics**
- Breakeven WR = |avg loss| / (avg win + |avg loss|) over all WILLY fills of the block.
- Bootstrap clusters by UTC day (4,000 draws); WILLY fires about once per pair-spike, so a day is the window unit.
- Refuse Δ = mean over all WILLY fills of (−y if zone else 0). Refused winners count against it, so the test is two-sided.
- LOMO = leave-one-month-out.
- Master: `reports/MASTER_POOL_stacked.csv`, 19 fills, all tick paths.
- Live: `~/Downloads/scalpars_orders_live_2026-10-16_13-10-21 (1).csv`, 8 fills. The TP +1.0 re-price uses public 1m klines (entry
  minute skipped). A zone fill keeps its real fill unless TP +1.0 changes its 1m outcome.

**Not priced:** refusing a WILLY frees the global WILLY hold, so other sleeves could open. That is a second-order effect, and it only
matters if the zone were worth refusing, which it is not.

## Headline: three blocks side by side (ZONE, today's exit; % per fill)

| block | N | zone N · days | zone WR | zone avg | rest avg | BE WR | P(zone mean<0) | refuse Δ/fill (P>0) | TP+1.0 Δ/fill (P>0) | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| ① BT 2025 tick (s1) | 228 | 92 · 83 | 79.3 % | **+0.169** | −0.212 | 77.2 % | 0.19 | −0.068 (0.18) | +0.016 (0.64) | ✗ |
| ① BT 2026-H1 tick (s1) | 393 | 161 · 96 | 76.4 % | −0.035 | +0.169 | 76.9 % | 0.58 | +0.014 (0.57) | +0.003 (0.51) | ✗ |
| ① BT 2026-H2 tick (s1) | 202 | 88 · 57 | 77.3 % | −0.022 | +0.027 | 78.6 % | 0.53 | +0.009 (0.52) | −0.019 (0.25) | ✗ |
| ① BT ALL tick (s1) | 823 | 341 · 236 | 77.4 % | +0.024 | +0.028 | 77.3 % | 0.41 | −0.010 (0.41) | +0.001 (0.50) | ✗ |
| ① BT ALL tick (s2 / s3) | 822 / 820 | 339 / 340 | 77.9 / 77.9 % | +0.038 / +0.050 | +0.030 / +0.034 | 77.4 / 77.3 % | 0.36 / 0.31 | −0.016 / −0.021 | +0.001 / −0.000 | ✗ |
| ① BT 2025 1m (refute-only, s1) | 668 | 202 · 144 | 71.8 % | −0.327 | −0.534 | 79.1 % | 0.98 | +0.099 | −0.011 | (zone *better* than rest) |
| ② MASTER (19) | 19 | 9 | 88.9 % | +0.530 | +0.605 | — | — | Σ −4.77 pts (−$870) | Σ −1.75 pts (−$222) | refutes |
| ③ LIVE 10-15/16 (8) | 8 | 4 | 25 % | −2.477 as traded | +1.096 | — | — | Σ +9.91 pts (+$248) | Σ +4.26 pts (+$115) | supports (N 4) |

A note on the 1m 2025 row. It technically clears the bar, because the zone is negative there. But the rest is even more negative
(−0.53), so the zone is *better* than the rest on 1m too. 1m stop-first inflates every stop, so this row cannot support anything, and
it does not refute the "no harm" read.

### Expectancy bar on the zone (seed 1, tick)

| era | zN | WR vs BE | ① | P(mean<0) | ② | days | top day / pair share of the net zone loss | ③ | ④ | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 92 | 79.3 vs 77.2 | ✗ | 0.19 | ✗ | 83 | (zone net positive) | ✓ | ✓ | ✗ |
| 2026-H1 | 161 | 76.4 vs 76.9 | ✓ | 0.58 | ✗ | 96 | 108 % (01-10) / 178 % (STO) | ✗ | ✓ | ✗ |
| 2026-H2 | 88 | 77.3 vs 78.6 | ✓ | 0.53 | ✗ | 57 | 223 % (09-10) / 444 % (POL) | ✗ | ✓ | ✗ |
| 2026 | 249 | 76.7 vs 77.5 | ✓ | 0.59 | ✗ | 153 | 81 % / 133 % (STO) | ✗ | ✓ | ✗ |
| ALL | 341 | 77.4 vs 77.3 | ✗ | 0.41 | ✗ | 236 | (zone net positive) | ✓ | ✓ | ✗ |

A share over 100 % means one day or one pair loses more than the whole zone's net loss: the zone is about flat and a few fills make
its sign.

### Legs alone (seed 1, tick, ALL eras): refuse-Δ per WILLY fill · LOMO sign-stable?

| leg | zone N | zone avg | rest avg | refuse Δ (P>0) | LOMO range | TP+1.0 Δ (P>0) |
|---|---|---|---|---|---|---|
| ZONE (both) | 341 | +0.024 | +0.028 | −0.010 (0.41) | −0.030 … +0.023 (no) | +0.001 (0.50) |
| VWAP_LEG (≤ 0.10) | 537 | −0.038 | +0.146 | +0.025 (0.66) | +0.009 … +0.067 (yes) | +0.013 (0.68) |
| RUN_LEG (≤ 10) | 458 | **+0.101** | −0.068 | **−0.056 (0.13)** | −0.088 … −0.020 (yes) | −0.016 (0.19) |

Per era, the VWAP leg's zone avg is 2025 −0.023 (rest −0.123), 2026-H1 0.000 (rest +0.270) and 2026-H2 −0.141 (rest +0.225). It is
worse than the rest only in 2026, and it never reaches P 0.95. The RUN leg is better than the rest in every era (2025 +0.132 vs −0.325,
2026-H1 +0.059 vs +0.114, 2026-H2 +0.140 vs −0.190), so the run cut works against the hypothesis.

### Per-pair concentration (zone, seed 1, tick, all eras: 341 fills on 176 pairs, Σ +8.05)

- Worst pairs: STO 4·−9.90 · POL 2·−8.42 · IDOL 2·−8.41 · RUNE 2·−8.40 · EGLD 2·−6.30.
- Best pairs: DEXE 5·+5.25 · MMT 4·+5.00 · XLM / B / BOME 4·+4.00.
- No pair-level story. The losses are single stops spread across pairs.

## ② MASTER: per trade (today's exit walked on ticks; $ at each trade's as-traded $ per point)

Zone fills: API3 (B19, as traded −2.44 → today +0.25 late TP) · MINA · ORCA · CFX · CHIP · S · W (B19, all +1.25) · OGN (B20, −4.23
stop) · ZK (B21, +1.25).

| batch | N | zone N | as traded Σ% / $ | as traded, zone refused | today's exit Σ% / $ | today, zone refused | today, zone TP +1.0 |
|---|---|---|---|---|---|---|---|
| B18 | 0 | — | — (WILLY went live at B19's start) | — | — | — | — |
| B19 | 14 | 7 | +8.32 / +$1,063 | +4.72 / +$485 | +10.05 / +$1,132 | +2.30 / +$86 | +8.55 / +$927 |
| B20 | 3 | 1 | −1.49 / −$90 | +2.54 / +$162 | −1.73 / −$105 | +2.50 / +$160 | −1.73 / −$105 |
| B21 | 2 | 1 | +2.51 / +$177 | +1.25 / +$87 | +2.50 / +$177 | +1.25 / +$87 | +2.25 / +$159 |
| **ALL** | 19 | 9 | **+9.33 / +$1,150** | +8.51 / +$735 | **+10.82 / +$1,203** | +6.05 / +$333 | +9.07 / +$981 |

The full per-trade table is in `reports/WILLY_LOWVWAP_SHORTRUN_tables.md` and `reports/WILLY_LOWVWAP_SHORTRUN_master.csv`.

## ③ CURRENT LIVE batch (10-15 → 10-16, 8 WILLY fills): per trade

| pair | zone | vs_vwap · run | as traded | zone refused | zone TP +1.0 (1m) | today's exit (1m) |
|---|---|---|---|---|---|---|
| DUSK 10-15 01:45 | ✓ | +0.07 · 8.0 | STOP −3.43 (−$92) | 0 | **TP +1.00 at min 26 (+$27)** | LATE_TP +0.25 |
| ZK 04:40 | — | +0.87 · 18.1 | TP +1.12 (+$28) | +1.12 | +1.12 | TP +1.25 |
| BLUR 09:30 | — | +2.24 · 22.5 | TP +1.09 (+$28) | +1.09 | +1.09 | TP +1.25 |
| STRK 10:20 | ✓ | −0.26 · 9.5 | STOP −3.79 (−$99) | 0 | −3.79 (peak +0.77, unchanged) | STOP |
| RLC 13:30 | — | +0.40 · 8.9 | TP +1.08 (+$25) | +1.08 | +1.08 | TP +1.25 |
| PYTH 15:30 | — | +0.51 · 12.6 | TP +1.09 (+$27) | +1.09 | +1.09 | TP +1.25 |
| BAT 16:00 | ✓ | −0.23 · 8.4 | TP +1.18 (+$30) | 0 | +1.00 (+$25) | TP +1.25 |
| ICNT 10-16 00:05 | ✓ | −0.07 · 8.0 | STOP −3.86 (−$86) | 0 | −3.86 (peak +0.60, unchanged) | STOP |
| **Σ** | 4 | | **−5.52 / −$139** | **+4.39 / +$108** | **−1.26 / −$25** | −1.90 / −$42 (stops at −4.2) |

The live stops filled **better** than the line, at −3.43 / −3.79 / −3.86, against the −4.2 the backtest books. Booking stops at −4.0
instead would raise the backtest zone and rest roughly equally (stop shares 11–19 % vs 16–21 %), so the verdict does not move.

## Files
- `scripts/study_willy_lowvwap_shortrun.py` walks all paths and asserts parity, writing `reports/WILLY_LOWVWAP_SHORTRUN_rows.csv`
  (4,557 rows).
- `scripts/study_willy_lowvwap_shortrun_report.py <k1m_dir>` writes `reports/WILLY_LOWVWAP_SHORTRUN_tables.md` (every seed × era ×
  pricing × zone/leg, LOMO, buckets, exit mix, master and live per trade), `reports/WILLY_LOWVWAP_SHORTRUN_tests.csv`, `_master.csv`
  and `_live.csv`.
- `scripts/validate_against_master.py`: **ALL CHECKS PASS** (run 2026-10-16 after the study).
