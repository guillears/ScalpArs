# Momentum LONG: follow-up on "BTC's next 15 minutes decides the trade" (2026-10-15)

- **Question (operator):** "We must follow up on that finding, to find a variable as filter, or exit optimization." Extended the same day: "If we cannot predict the entry, optimize the EXIT once we know whether BTC kept rising: exit after 15 min, or lower the TP / tighten the SL when it's clearly not a runner."
- **Pre-registration:** `reports/ML_BTCFWD_PREREG.txt`.
  - It was written before anything was priced. The script refuses to overwrite it.
  - The operator extension ("ARM B2") was appended as a dated section before any B2 cell was priced.
  - 339 exit cells and 56 filter cells, all counted.
- **Builds on:**
  - `ML_WINNERS_LOSERS_2026-10-15.md`: BTC up ≥ 0.10 % in the 15 min after entry → WR 77–88 %; BTC down → 32–48 %.
  - `ML_DELAY_BTC_CONFIRM_2026-10-15.md`: waiting for BTC to confirm is worse.
- **Safety:** nothing shipped. No engine, config, `trading_config.json`, template or test edits, no git, no bot API.
- **Data:**
  - **Cache first.** One Binance top-up for the only field the cache does not hold, taker-buy volume:
    - BTC / ETH / SOL 1m, and every top-50 member's 5m, 2024-12-29 → 2026-10-15;
    - about 30k requests at ≤ 950 used-weight/min, read from the header; no 418 / 429;
    - stored in `reports/ml_btcfwd_fetch/`, outside `backtest_cache`.
  - **Validation:**
    - `validate_against_master.py`: ALL CHECKS PASS.
    - The study's own `study_ml_btcfwd_validate.py`: ALL STUDY CHECKS PASS. It checks source parity, clock shift 0, replica parity, BTC tick state, no-lookahead 500/500, and counts done two ways.

---

## Plain-English summary: the operator's exit question (ARM B2)

1. **"Exit after 15 min when BTC did not keep rising" loses money.**
   - Trades still open at minute 10 or 15 with BTC flat or down since entry do end worse than the others (WR 35–56 % vs 70–83 %).
   - But from that point they lose very little more on average. Their remaining move is about −0.08 to +0.11 %.
   - Selling at market costs a fee plus slippage, so cutting them is worse than letting today's exits handle them:

     | rule | 2025 | 2026 | master |
     |---|---|---|---|
     | exit at minute 15 when BTC ≤ 0 since entry | −0.012 %/fill | −0.043 %/fill | −0.047 %/fill |
     | same rule at minute 10 | — | −0.061 %/fill | −0.097 %/fill |

   - This is the stop-CF rule's warning in action. The trades it cuts that the stop would have taken are saved (+0.13 to +0.22 per fill).
   - But just as many are late recoverers the stop never touches, and those are killed (−0.13 to −0.26).
2. **Lowering the target to +0.2 / +0.3 / +0.5 % once BTC is not rising:**
   - It is roughly break-even in 2025 (+0.001 to +0.007).
   - It is negative in 2026 (to −0.026) and on real master fills (−0.001 to −0.058).
   - The small "take +0.3 now" wins are paid for by cutting the few late runners.
3. **Tightening the stop to −0.3 / −0.5 % or to break-even: negative in every block.**
   - About −0.01 to −0.02 in 2025, −0.02 to −0.06 in 2026, −0.03 to −0.12 on master.
   - Combining a lower target with a tighter stop is worse still.
4. **The mirror ("BTC is running, so let the runner run longer by suspending the trail or widening the stop"): effect ≈ 0.**
   - Every cell is within ±0.01 %/fill. Today's stack has no EMA-cross, signal-lost or BE exits switched on for momentum longs.
   - The only early exits are the stop and the runner trail. Loosening them for 10–15 min changes almost nothing.
5. **Verdict for the operator's idea:**
   - 0 of 72 cells pass. 3 pass in 2025 only (tiny, +0.005) and lose in 2026.
   - **No exit change.** Today's exit stack already handles "not a runner" trades about as well as any post-entry rule we can decide live.

## Plain-English summary: the rest of the study

1. **Can BTC's next 15 minutes be predicted?** Only by a hair, and not usefully.
   - We tested 25 new market variables on every 5-minute moment of 2025 + 2026 (187k moments, 940k at 1-minute):
     - BTC taker-buy pressure;
     - BTC momentum and acceleration;
     - distance from the 1h / 4h / 24h high and low;
     - volatility regime;
     - ETH / SOL leading BTC;
     - top-50 breadth and its acceleration;
     - top-50 taker-buy pressure;
     - hour and weekday.
   - Many are *statistically* real out of sample: AUC 0.51–0.525 against a null of ~0.504, the same sign in both years. They all say the same thing: after BTC rose or was bought hard, it gives back a sliver.
   - **But the sliver is 0.3–1.5 basis points of BTC move** (the pre-registered usefulness bar was 2 bps in both years). That is far too small to move a trade's outcome. **0 of 27 features pass.**
2. **At our actual entry moments these variables do not predict BTC at all.** AUC at fills is 0.46–0.54 and flips sign between 2025, 2026 and master.
   - Worse, the "BTC is overbought" zones are *better* trades on master and 2026-H2. For example, with high BTC taker-buy pressure the zone does +0.33 to +0.39 %/fill better than the rest.
   - That is consistent with the earlier study: momentum longs like strong BTC flow, even though BTC itself mean-reverts a little.
3. **Exiting early on a BTC drop since entry (the original Arm B, 267 cells): nothing passes.**
   - Every one of the 237 exit / tighten cells is ≤ 0 in 2026.
   - The best is +0.011 in 2025 (P 1.00) and −0.011 in 2026 (P 0.04). That is a sign flip, not a regime-stable edge.
   - No day-level regime variable separates the years (24h / 7d / 30d BTC trend sign, volatility; all p_null ≥ 0.13 except one cell at 0.03 out of 12 tests).
4. **One curiosity, NOT a candidate:**
   - **The pattern:** "SOL ran ≥ 0.11 % ahead of BTC over the last 5 min" marks worse momentum-long entries. The zone is worse than the rest in all four blocks:

     | block | zone − rest | P(zone worse) |
     |---|---|---|
     | 2025 | −0.02 | 0.71 |
     | 2026-H1 | −0.09 | 0.83 |
     | 2026-H2 | −0.24 | 0.96 |
     | master | −0.27 | 0.98 |

   - It technically clears every filter gate, including the expectancy bar on master: 35 fills, WR 51 % vs breakeven 63 %, P(avg < 0) 0.97, 23 days.
   - **Why it is not a candidate:**
     - The variable failed Step 1 (it does not predict BTC, so the mechanism is unknown).
     - It is 1 of 56 exploratory cells.
     - Its 2025 separation is weak. 2026 and master cover the same months, so they are not independent.
     - The one live fill (PEPE, +1.16 %) sits in the zone.
   - **Status:** watchlist hypothesis only (frozen definition in §6).
5. **Bottom line:**
   - The market's next 15 minutes cannot be forecast well enough at entry to filter.
   - Once in the trade, no live-decidable reaction to BTC beats today's exit stack.
   - The momentum-long edge question stays where CURRENT_STATE already put it: **the 30-fill sleeve on/off gate at 1×**.

---

## 1. Blocks and fidelity (MOMENTUM LONG, closed, non-probe, % at 1×)

| block | N | days | today avg % (engine) | replica BASE avg | within ±0.05 | sign agree |
|---|---|---|---|---|---|---|
| ① live (post go-live) | 1 | 1 | +1.157 | +1.098 | — | 100 % |
| ② master as-traded | 191 | 77 | +0.027 | +0.045 | 82.7 % | 97.9 % |
| ② master stack_keep | 139 | 66 | +0.141 (stack_pct) | +0.153 | 81.3 % | 98.6 % |
| ③ replay 2025 BULL | 2617 | 301 | −0.187 | −0.199 | 85.1 % | 97.7 % |
| ③ replay 2026-H1 BEAR | 1171 | 154 | −0.085 | −0.085 | 94.7 % | 99.6 % |
| ③ replay 2026-H2 BULL | 521 | 79 | −0.015 | −0.013 | 97.3 % | 99.8 % |

- **The replica:** `scripts/ml_exit_optimize_yr5.py` BASE (parity 89.2 % on the replay, gate ≥ 85 %: PASS).
  - The study's hooked copy of `evaluate()` equals `RX.evaluate` exactly on all 4,501 fills (asserted).
  - BASE equals the delay study's `rep0` exactly.
- **Today's momentum-long exit stack**, verified key by key in `trading_config.json` against the frozen yr6 config, with no exit-key differences:
  - stop −0.70 / −1.00 while the signal is on, widened to −1.5 × ATR and capped at −1.2;
  - recovery hold;
  - hard-TP ladder 1.25 : 0.25 … 4.0 : 0.80;
  - runner trail: arm +0.40, floor max(peak − 1.0 × ATR, +0.10);
  - 180-min no-expansion.
- **OFF:** EMA13-cross exit (long), signal-lost, fast exit, tick-momentum, regime-change, BE levels.
- **Tick coverage:** 2025 is 79 % tick-priced (21 % on 1m bars, flagged); 2026 is 100 %; master 97 %. Tick-only sensitivities are in the tables and change no sign.

## 2. ARM A step 1: can BTC's forward 15 min be predicted? (5m grid, 187,198 moments, walk-forward 2025-04 → 2026-10, 200-shift circular null)

- **Base rates:** P(fwd15 > 0) = 0.498; P(fwd15 ≤ −0.10 %) = 0.247; sd = 0.238 %.
- **Sign:** "sign" is the training-set direction. A negative sign means a high value predicts a lower BTC.
- **Spread:** OOS mean fwd15 in the top minus the bottom training quintile.

| feature | sign | OOS AUC (up) | null 99th | IC 2025 / 2026 | spread bps 2025 / 2026 | P(down) worst quintile vs 0.240 | Step 1 |
|---|---|---|---|---|---|---|---|
| tbr1 (BTC taker-buy 1 min) | − | 0.509 | 0.504 | +0.010 / +0.018 | 0.4 / 0.6 | 0.230 | fail (c) |
| tbr5 | − | 0.514 | 0.503 | +0.025 / +0.021 | 0.8 / 0.3 | 0.232 | fail (c) |
| tbr15 | − | 0.522 | 0.504 | +0.035 / +0.035 | 1.0 / 0.8 | 0.237 | fail (c) |
| tbr15_dev (vs 24 h) | − | 0.521 | 0.504 | +0.034 / +0.034 | 1.0 / 0.8 | 0.234 | fail (c) |
| btc_m1 | − | 0.511 | 0.503 | +0.015 / +0.022 | 0.6 / 0.4 | 0.294 | fail (c) |
| btc_m5 | − | 0.518 | 0.503 | +0.031 / +0.032 | 0.8 / 0.5 | 0.299 | fail (c) |
| btc_acc1 | − | 0.501 | 0.503 | +0.001 / +0.005 | 0.1 / 0.3 | 0.287 | fail (a, c) |
| btc_acc5 | − | 0.503 | 0.503 | +0.006 / +0.006 | 0.3 / 0.1 | 0.291 | fail (c) |
| dhi_1h (dist. from 1 h high) | − | 0.525 | 0.505 | +0.040 / +0.041 | 0.9 / 0.9 | 0.204 | fail (c) |
| dhi_4h | − | 0.526 | 0.504 | +0.029 / +0.048 | 0.4 / 1.5 | 0.209 | fail (c) |
| dhi_24h | − | 0.517 | 0.504 | +0.025 / +0.027 | 0.7 / 0.3 | 0.226 | fail (c) |
| dlo_1h | − | 0.521 | 0.505 | +0.029 / +0.038 | 0.0 / 0.2 | 0.325 | fail (c) |
| dlo_4h | − | 0.522 | 0.505 | +0.024 / +0.043 | 0.3 / 0.3 | 0.327 | fail (c) |
| dlo_24h | − | 0.519 | 0.504 | +0.028 / +0.030 | 0.5 / −0.3 | 0.302 | fail (c) |
| rv_ratio (1 h / 24 h vol) | ± | 0.499 | 0.505 | +0.000 / −0.000 | −0.2 / 0.3 | 0.213 | fail |
| eth_lead1 / eth_lead5 | + / ± | 0.503 / 0.498 | 0.503 | ≈ 0 | 0.0–0.7 / −0.4–0.4 | 0.28 | fail |
| sol_lead1 / sol_lead5 | ± / − | 0.500 / 0.503 | 0.503 | ≈ 0 | negative | 0.27 | fail |
| br_up5 (top-50 breadth) | − | 0.512 | 0.503 | +0.021 / +0.018 | 0.4 / −0.1 | 0.269 | fail (c) |
| br_acc5 (breadth acceleration) | + | 0.499 | 0.503 | −0.001 / −0.000 | −0.2 / 0.3 | 0.250 | fail |
| agg_tbr5 (top-50 taker-buy) | − | 0.516 | 0.504 | +0.026 / +0.025 | 0.7 / 0.4 | 0.246 | fail (c) |
| agg_tbr15 | − | 0.523 | 0.504 | +0.031 / +0.039 | 0.4 / 0.6 | 0.248 | fail (c) |
| hour / dow (target-encoded) | — | 0.499 / 0.499 | 0.503 / 0.502 | ≈ 0 | 0.1 / 0.2–0.3 | 0.26 / 0.25 | fail |
| control btc_r15 / btc_r60 | − | 0.525 / 0.524 | 0.504 / 0.505 | +0.04 / +0.03–0.04 | 0.6 / 0.3–0.7 | 0.30 | fail (c) |
| MULTI (ridge, all 25) | | 0.515 | 0.503 | +0.021 / +0.030 | 0.3 / 1.0 | 0.266 | fail (c) |

- **Read:** a small, real, cross-year-stable mean reversion. After BTC rose, or was bought aggressively, or sits near its 1h / 4h high, the next 15 minutes are a hair weaker.
  - At most 1.5 bps between the extreme quintiles, against a 24 bps standard deviation.
- **P(down ≥ 0.10)** rises in the "after a rise" quintile (0.29–0.33 vs 0.24). That is a volatility effect, not direction: the same quintile also has more ≥ +0.10 moves.
- **1-minute grid** (940k moments, 50-shift null, BTC/ETH/SOL features): the same picture. Best spread 1.4 bps; 0 passers (`ML_BTCFWD_predict_1m.txt`).
- **Funding / OI / order book:** not cached for BTC, ETH or SOL. See §8.

## 3. ARM A step 2: the same variables at momentum-long entries (three blocks; exploratory, since no Step-1 passer)

**(a) Do they predict BTC's next 15 min at the moment we enter?** No.

| feature (2025-frozen sign) | AUC 2025 | AUC 2026 | AUC master |
|---|---|---|---|
| tbr15 | 0.521 | 0.518 | 0.491 |
| btc_r15 | 0.504 | 0.540 | 0.516 |
| dhi_4h | 0.527 | 0.540 | **0.416** |
| agg_tbr15 | 0.495 | 0.508 | 0.491 |
| MULTI | 0.519 | 0.530 | 0.497 |

The full table for all 28 is in `ML_BTCFWD_filter_tables.txt` §(a).

**(b) The frozen worst-quintile (Z1) / worst-decile (Z2) zone as a filter** (Δ %/signal, skip = 0; 56 cells):
- 0 eligible candidates. Verdicts: 29 one-year-only, 26 fail, 1 "passes but ineligible" (sol_lead5 Z1).
- Typical pattern: Δ 2025 +0.04 to +0.13 (P 1.00) and Δ 2026 −0.01 to +0.05, but **Δ master −0.10 to +0.04 and Δ 2026-H2 often negative.**
- The 2025 "gain" is the skip effect on a sleeve that loses in 2025, not selection. See (c).

**(c) Separation lens** (zone − rest, the honest test; Z1). A filter needs the zone below the rest in every block:

| feature | 2025 | 2026-H1 | 2026-H2 | master |
|---|---|---|---|---|
| tbr15 | −0.05 | −0.09 | **+0.27** | **+0.39** |
| btc_r15 | −0.05 | −0.03 | **+0.18** | **+0.30** |
| dhi_1h | −0.04 | −0.05 | **+0.18** | **+0.23** |
| agg_tbr5 | +0.02 | −0.08 | +0.09 | **+0.21** |
| dlo_24h | +0.05 | +0.10 | +0.30 | +0.29 |
| **sol_lead5** | −0.02 (P 0.71) | −0.09 (0.83) | −0.24 (0.96) | −0.27 (0.98) |

- The BTC "overbought" zones are *better* momentum-long entries on master and in 2026-H2. That is the opposite of a filter.
- Only sol_lead5 is worse in all four blocks.

**sol_lead5 Z1 on the three blocks:**

| block | zone N | zone WR | zone avg | rest avg | breakeven WR |
|---|---|---|---|---|---|
| ① live | 1 (PEPE, in zone) | 100 % | +1.157 | — | — |
| ② master as-traded | 35 · 23 days | 51 % | −0.192 | +0.076 | 62.8 % |
| ② master stack_keep | 27 | — | Δ +0.034/signal (P 0.92) | — | — |
| ③ 2025 / 2026-H1 / 2026-H2 | 668 / 166 / 71 | 55 / 55 / 48 % | −0.204 / −0.161 / −0.220 | −0.181 / −0.073 / +0.017 | 68.7 / 65.7 % |

- **Master per batch (zone fills):** the losers are spread over B6, B8, B9, B12–B19; BASE (+0.44) and B3 (+0.04) are positive.
- **Concentration:** top pair 14 %, top day 14 %.

## 4. ARM B: exits driven by BTC / breadth / P&L since entry (267 cells)

**State at minute k** (fills still open). BTC down ≥ 0.10 since entry at k = 5:

| block | share of open fills | final WR | avg |
|---|---|---|---|
| 2025 | 16 % | 31 % | −0.47 |
| 2026 | 17 % | 45 % | −0.29 |
| master | 8 % | 36 % | −0.36 |

BTC up ≥ 0.10: WR 71 / 79 / 86 %. So the conditioning signal is strong.

**Gate result: 0 candidates.**

| family | cells | fail | one-year-only (all 2025) | Δ range 2025 | Δ range 2026 | Δ range master |
|---|---|---|---|---|---|---|
| EXIT at market | 79 | 62 | 17 | −0.025 … +0.011 | −0.137 … −0.000 | −0.237 … +0.009 |
| TIGHTEN stop −0.3 | 79 | 66 | 13 | −0.029 … +0.009 | −0.091 … 0.000 | −0.147 … +0.010 |
| TIGHTEN stop −0.5 | 79 | 72 | 7 | −0.027 … +0.005 | −0.065 … −0.000 | −0.085 … +0.010 |
| mirror: stop held at −1.2 | 10 | 5 | 5 | 0.000 … +0.010 | −0.002 … +0.002 | −0.006 … +0.002 |
| mirror: runner giveback 1.5 × ATR | 10 | 10 | 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| mirror: both | 10 | 8 | 2 | 0.000 … +0.009 | −0.003 … +0.002 | −0.017 … −0.001 |

- **Best two-sided example:** exit at k 3 when BTC ≤ −0.15 and P&L ≤ −0.2.
  - 2025: saved 76 trades (+24.7 points) vs killed 20 (−17.2).
  - 2026: saved 56 (+21.1) vs killed 38 (−21.7) → Δ −0.000.
  - Base-stopped fills gain +0.02 to +0.03; not-stopped fills lose −0.01 to −0.02. This is exactly the stop-CF split.
- **Sensitivities:**
  - Slip 0.02 (optimistic) lifts 2025 by ~0.005 but leaves 2026 ≤ 0.
  - Slip 0.17 makes everything negative.
  - The tick-only subset gives the same signs.
- **Breadth-turned-down exits** (k 5 / 8): 2026 −0.04 to −0.07; master −0.05 to −0.08.
- **Regime check on the one-year-only cells** (DAY units, 500× day-shuffled null): BTC 24h / 7d / 30d-trend sign and volatility. No variable separates (p_null 0.13–0.98; one 0.03 among 12). Hypothesis: none.

## 5. ARM B2: the operator's exit extension (72 cells)

**B2.1 conditional table** (k = 15, collapsed over P&L; full 4 × 4 × 2 tables for k = 5 / 10 / 15 / 20 in `ML_BTCFWD_exit2_tables.txt`).

| block | BTC since entry | N | final WR | avg final | avg remaining move | ends by stop / runner |
|---|---|---|---|---|---|---|
| master | ≤ −0.10 | 17 | 59 % | −0.10 | +0.13 | 41 / 59 % |
| master | −0.10…0 | 34 | 56 % | −0.05 | +0.05 | 44 / 47 % |
| master | 0…+0.10 | 36 | 78 % | +0.18 | +0.15 | 22 / 69 % |
| master | ≥ +0.10 | 30 | 83 % | +0.42 | +0.17 | 20 / 60 % |
| 2025 | ≤ −0.10 | 382 | 35 % | −0.37 | −0.03 | 63 / 32 % |
| 2025 | ≥ +0.10 | 379 | 70 % | +0.10 | −0.06 | 28 / 60 % |
| 2026-H1 | ≤ −0.10 | 235 | 47 % | −0.19 | +0.11 | 53 / 39 % |
| 2026-H2 | ≤ −0.10 | 37 | 38 % | −0.37 | −0.15 | 62 / 35 % |
| 2026-H2 | ≥ +0.10 | 76 | 80 % | +0.34 | +0.16 | 21 / 62 % |

- **Key line:** the "not a runner" trades are already mostly priced in by minute 15. Their *remaining* move is near zero (−0.15 … +0.13), so cutting them saves almost nothing and pays fee + slip.
- **Inside the "BTC down, P&L < −0.3" cell:** final WR 7–29 %, remaining move −0.11 … +0.04.
  - Those are real losers, but the remaining loss is small: the stop is close.
- **Inside the "BTC down, P&L −0.3…0" cell:** final WR 44–86 % and the remaining move is positive in several blocks. Those are the recoverers a cut kills.

**B2.2 / B2.3 gate result: 0 / 72 candidates.** 69 fail; 3 are 2025-only (mirror, +0.005 to +0.007) and lose in 2026. Δ %/fill:

| cell (k, condition) | 2025 (P) | 2026 (P) | 2026-H2 | master | master_keep |
|---|---|---|---|---|---|
| EXIT k15 BTC ≤ 0 | −0.012 (0.10) | −0.043 (0.00) | −0.026 | −0.047 | −0.059 |
| EXIT k10 BTC ≤ 0 | −0.017 (0.05) | −0.061 (0.00) | −0.047 | −0.097 | −0.139 |
| TP +0.3 k15 BTC ≤ 0 | +0.005 (0.82) | −0.006 (0.23) | −0.011 | −0.008 | −0.023 |
| TP +0.5 k15 BTC ≤ +0.05 | +0.005 (0.83) | +0.008 (0.87) | +0.011 | −0.023 | −0.035 |
| SL −0.3 k15 BTC ≤ 0 | −0.010 (0.12) | −0.030 (0.00) | −0.012 | −0.027 | −0.036 |
| SL break-even k15 BTC ≤ 0 | −0.008 (0.19) | −0.037 (0.00) | −0.028 | −0.043 | −0.055 |
| TP +0.3 & SL −0.5 k15 BTC ≤ 0 | −0.004 (0.33) | −0.026 (0.00) | −0.011 | −0.043 | −0.051 |
| mirror: runner + stop suspended 15 min, k10 BTC ≥ +0.10 | +0.007 (0.97) | −0.002 (0.29) | −0.001 | −0.012 | −0.024 |

- **Closest miss:** TP +0.5 at k15 with BTC ≤ +0.05. It is positive in both backtest years (P 0.83 / 0.87, below the 0.90 bar) but **negative on master (−0.023 as-traded, −0.035 on today's stack)**. It fails.
- **Two-sided read (EXIT k15, BTC ≤ 0):**
  - 2026: saved 282 fills (+90.7) vs killed 245 (−164.3).
  - master: saved 25 (+6.7) vs killed 26 (−15.6).
- **Slip:**
  - At 0.02 (optimistic), 2025 turns +0.012, but 2026 (−0.019) and master (−0.025) stay negative.
  - At 0.17: −0.033 / −0.065 / −0.065.
- **Exit-CF rule:** all 38 hooked variants equal BASE on every fill already closed at T (asserted). Unaffected fills show Δ = 0 (asserted per cell).
- **Regime check on the 3 one-year-only mirror cells:** the only nominal hit is BTC 7d sign at p_null 0.03 (1 of 12 tests). It is not a hypothesis worth registering.

## 6. Recommendation

1. **No filter, no exit change, no config change, no Scout line.**
   - 0 of 395 pre-registered cells pass: 267 Arm B + 72 Arm B2 + 56 Arm A step 2, the last ineligible because Step 1 had 0 passers.
   - The operator's "exit after 15 min / lower TP / tighten SL when it's clearly not a runner" was tested in 72 cells on today's real exit stack. It costs money in 2026 and on real fills.
2. **Watchlist only (no code, not a candidate, tally at batch reviews):**
   - **Name:** ML_SOL_RAN_AHEAD.
   - **Definition:** MOMENTUM LONG where SOL's 5-min return minus BTC's at entry is ≥ +0.110 %. The cut is frozen from the 2025 all-bar 80th percentile; it must never be re-fit.
   - **Evidence:**
     - post-hoc, 1 of 56 exploratory cells;
     - zone − rest −0.27 on master, −0.24 in H2, −0.02 (weak) in 2025;
     - mechanism unknown (it does not predict BTC);
     - the live PEPE winner is in the zone.
   - **After the 30–50 % haircut:** about −0.10 to −0.17 %/fill on zone fills.
   - **Arm consideration** requires fresh fills (opened ≥ 2026-10-16), ≥ 15 zone fills on ≥ 8 days, and the full expectancy bar (WR < sleeve breakeven, window-bootstrap P ≥ 0.95, no day / pair ≥ 50 %). Retire if zone avg ≥ rest avg at N 15.
   - It needs SOL 1m data at entry, which is not stamped today, so it is a rebuild-at-review item.
3. **The open lever stays the CURRENT_STATE momentum-long on/off gate (30 fills at 1×).**
   - Three studies now agree:
     - the post-entry BTC path decides the trade;
     - it cannot be forecast at entry;
     - it cannot be waited for;
     - it cannot be reacted to profitably after entry.
   - That is a regime question for the sleeve, not a rule question.

## 7. Multiplicity

| arm | cells | expected lucky passes |
|---|---|---|
| Arm A step 1 | 27 features + 1 multivariate | 99th-percentile null ≈ 0.3 |
| Arm A step 2 | 56 exploratory cells | ~1 lucky pass (1 seen, sol_lead5) |
| Arm B | 267 cells | ~2.7 lucky gate-1 passes (0 seen) |
| Arm B2 | 72 cells | ~0.7 (0 seen) |

## 8. Could NOT test

- **Funding, open interest, liquidations, order book for BTC / ETH / SOL:** not cached (`funding/` holds alts only; no OI, liquidation or depth history). Not fetched: no free historical depth / liquidation source.
- **Taker side of aggTrades:** `ticks_q` has no buyer-maker flag. Taker ratios come from kline taker-buy volume (1m BTC / ETH / SOL, 5m top-50), so they are not sub-minute.
- **Top-50 5m:** 27 of 4,060 membership spans hit repeated request failures. Breadth / aggregate-taker coverage is 99.6 % of grid moments and 98 % of master fills.
  - The universe list ends 2026-10-11, so live-window breadth is NaN. The live fill (10-15) has no breadth stamp.
- **Tick coverage:** 21 % of 2025 fills are priced on 1m o→l/h→c paths (flagged; tick-only sensitivity reported, same signs).
- **Live execution:** latency and queue are not modelled. Rule exits pay 0.10 % slip (0.02 / 0.17 sensitivities). TP-replacement exits keep the taker fee. A resting maker TP would add +0.027 per TP exit, which is +0.004 to +0.009 per fill on these cells.
  - That lifts TP +0.3 k15 (BTC ≤ +0.05) to about +0.006 on master and +0.007 in 2026, but its 2026 P was 0.55, far from 0.90.
  - The closest miss (TP +0.5 k15) stays negative on master (−0.018).
  - No cell would pass.
- **Recovery-hold interaction:** a tightened-stop hit is treated as a new exit (no recovery hold). The RH premise uses BTC 5m closes from the cache.
- **Slot effects:** an earlier exit frees a slot sooner, and other sleeves' fills are not modelled.
- **Blocked signals:** signals today's stack refuses are absent from the replay, so filter zones are read on admitted signals only.
- **Live block:** N = 1. The same holds for every table.

## 9. Exact commands (about 2 h of fetch at the shared weight budget, then about 10 min of compute)

```
venv/bin/python scripts/study_ml_btcfwd_prereg.py                     # write-once pre-registration
venv/bin/python scripts/study_ml_btcfwd_prereg_ext.py                 # appends the dated ARM B2 section once
venv/bin/python -W ignore scripts/study_ml_btcfwd_fetch.py phase1 BTCUSDT   # (one process per symbol: ETHUSDT, SOLUSDT)
venv/bin/python -W ignore scripts/study_ml_btcfwd_fetch.py phase2 <i> 5     # i = 0..4, top-50 5m taker klines (resumable)
venv/bin/python -W ignore scripts/study_ml_btcfwd_features.py         # → reports/ML_BTCFWD_grid5m.csv.gz, ML_BTCFWD_grid1m_btc.csv.gz
venv/bin/python -W ignore scripts/study_ml_btcfwd_predict.py --grid 5m --shifts 200 > reports/ML_BTCFWD_predict_5m.txt
venv/bin/python -W ignore scripts/study_ml_btcfwd_predict.py --grid 1m --shifts 50  > reports/ML_BTCFWD_predict_1m.txt
venv/bin/python -W ignore scripts/study_ml_btcfwd_filter.py > reports/ML_BTCFWD_filter_tables.txt
venv/bin/python -W ignore scripts/study_ml_btcfwd_exit_price.py --jobs 7          # → reports/ML_BTCFWD_exit_rows.csv
venv/bin/python -W ignore scripts/study_ml_btcfwd_exit_analyse.py > reports/ML_BTCFWD_exit_tables.txt
venv/bin/python -W ignore scripts/study_ml_btcfwd_exit2_price.py --jobs 6         # → reports/ML_BTCFWD_exit2_rows.csv
venv/bin/python -W ignore scripts/study_ml_btcfwd_exit2_analyse.py > reports/ML_BTCFWD_exit2_tables.txt
venv/bin/python -W ignore scripts/study_ml_btcfwd_validate.py          # V1–V6 → ALL STUDY CHECKS PASS
venv/bin/python scripts/validate_against_master.py                    # ALL CHECKS PASS
```

**Files:**
- `reports/ML_BTCFWD_PREREG.txt`
- `ML_BTCFWD_predict_{5m,1m}.{txt,csv}`
- `ML_BTCFWD_filter_{tables.txt,cells.csv}`
- `ML_BTCFWD_fill_features.csv`
- `ML_BTCFWD_exit_{rows.csv,cells.csv,tables.txt}`
- `ML_BTCFWD_exit2_{rows.csv,cells.csv,conditional.csv,tables.txt}`
- `ML_BTCFWD_grid5m.csv.gz`
- `ML_BTCFWD_grid1m_btc.csv.gz`
- `reports/ml_btcfwd_fetch/` (top-up klines + logs)

**Scripts:** `scripts/study_ml_btcfwd_{prereg,prereg_ext,fetch,features,predict,filter,common,exit_price,exit_analyse,exit2_price,exit2_analyse,validate}.py`

## Review corrections (deep review, 2026-10-15) — conclusions HOLD (0 / 395), five text corrections
1. B2 "remaining move" from minute k: −0.15 … +0.13 % at k15 and −0.10 … +0.29 % at k10 (the master k10 (−0.10, 0] bucket, remaining
   +0.289, drives master k10 −0.097) — not "−0.08 to +0.11".
2. Arm A: fill-level AUC spans 0.42–0.55 (dhi_4h on master 0.416), not 0.46–0.54.
3. B2 mirror: within ±0.01 on the backtest; master −0.012, master stack_keep −0.025 (k10 STR15).
4. The B2 "master" column is the replica re-priced under TODAY's stack on master fills (correct method, mislabelled as-traded); master
   replica parity 82.7 % within ±0.05 — below the 85 % gate → read the master column with that caveat (pooled replay parity 89.2 %).
5. ML_SOL_RAN_AHEAD: overstated. 2025 (largest block) zone − rest only −0.02 (P 0.71) and positive in 2025-Q4; Spearman with BTC fwd15
   −0.005; the master expectancy pass holds only as-traded — under today's stack (replica) zone −0.116, P(avg < 0) 0.85 → FAILS the 0.95
   bar; the master separation is entirely Sep–Oct (Jun–Aug the zone was BETTER, +0.07) → market-wide variable = ~2 monthly windows, not
   35 observations. → passive watchlist note only (rebuild at batch review), no Scout code line; its arm gate adds a window clause: no
   single month ≥ 50 % of the zone loss, judged on today's-stack P&L.
Verified: B2 cells recomputed independently (exit@15 −0.0122 / −0.0435 / master −0.0467; 282 saved +90.7 vs 245 killed −164.3), today's
momentum stack confirmed (BE / EMA13 / signal-lost / fast / tick / regime exits OFF for these levels; runner trail on), slip only on
rule-added market exits, no lookahead (validate V1–V6 ALL PASS), walk-forward strictly forward.
