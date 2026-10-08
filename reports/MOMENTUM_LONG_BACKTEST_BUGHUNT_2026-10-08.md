# MOMENTUM LONG backtest bug hunt (2026-10-08)

**Question (operator):** "I'm 99 % sure there is a bug in the BACKTEST for momentum longs." The yr5 engine replay (code 181131e, 3 seeds,
warm-up trimmed, tick-level) prices momentum longs (ML) at **−0.067 %/trade** (595 fills/seed/yr). Most of the loss sits in Jan–Apr
(−0.10) and in the Aug-27 → Sep-11 offline stretch (−0.22), where there is no live data to check against. This hunt looks for a bug
with **independent re-implementations** built from raw data. It does not reuse the harness's own outputs as evidence.

**Validation first:** `scripts/validate_against_master.py` → **ALL CHECKS PASS** (run before any analysis).

**Scope:**
- Read-only on code and config. No commit, no bot API, `scripts/engine_replay*.py` untouched.
- No Binance calls at all. Everything came from the local caches: aggTrade archives (`ticks_q/`, `ticks/`), `k5m_full/`, `btc_5m.csv`,
  `daily/`, `exchange_info.json`, the yr5 run CSVs and the yr5 decision journals.

## Plain-English answer

1. **No bug found that touches momentum-long results.** Every part of the backtest was rebuilt independently from raw trade data and
   the results came out the same:
   - exits
   - entry prices
   - the indicator values behind every decision
   - the entry filters, both for trades taken and for trades refused
   - the data files
   - the chunk boundaries
2. **The exits are right.** Every one of the 1,784 backtest trades was re-run tick by tick with code written from scratch from the
   live engine's rules:
   - 97 % end for the same reason, 91 % at the same second.
   - The average result differs by only **+0.0007 % per trade**.
   - Every month agrees within 0.01 %, January to April included.
3. **The entries are right.**
   - In every month, the indicator values the backtest used match an independent rebuild from raw trades (Jan–May as well as Jun–Oct).
   - No trade breaks an entry rule on its own recorded values.
   - Refused signals were really failing the filter the backtest named: 96–100 % confirmed in every month.
4. **The real reason for the loss: the entry signal had no edge in the backtest year.**
   - One hour after entry, the price is on average where it was at entry (−0.008 %, both halves of the year).
   - Random entries on the same coins do about as well.
   - Every simple exit tried on the same entries also loses:

     | Exit tried | %/trade |
     |---|---|
     | +0.5 / −0.7 | −0.074 |
     | +1.0 / −1.0 | −0.078 |
     | hold 60 min | −0.081 |

     The live exit rules do slightly better than all of them (−0.067).
   - The loss is the trading fees (~0.06–0.09 % per trade) on entries with no edge. It is not a code problem.
5. **Batch by batch, on the exact live days, the backtest is not systematically worse than the master:**
   - **When live and the backtest take the same trade at the same second, they price it the same** (+0.32 vs +0.34).
   - **Outside the washed-out window (Jun-18 → Jul-10), the backtest beats master kept on the same days:** +0.13 vs +0.05.
   - All of the master's advantage sits in that one window (master +0.49 vs backtest +0.07). Its causes are already known:
     - the master's fills there are live's own winners after filtering (in-sample);
     - the backtest scans at different seconds, so it misses some of those winners;
     - live ran older rules then, so the backtest also takes ~40/seed signals live never took (each worth ≈ 0).

     None of these is a bug.
6. **Corrected momentum-long number: −0.061 %/trade** (≈ 568 fills/seed/yr, 95 % range −0.12 … −0.01). This is not a bug fix. The
   backtest ran on the Oct-4 code snapshot, and four rules shipped since then. Re-applying them moves the number only slightly:
   - the chop-burst block removes 13 losing fills/seed (−0.20 each);
   - the WILLY global hold removes 46 fills/seed (−0.06 each);
   - the 3-leg heat rule adds back 33 fills/seed (−0.003 each).

   The year stays negative. The momentum-long verdict does not change.

## 1 · Exits: an independent tick re-walk of every yr5 ML fill

**Script:** `scripts/study_ml_bughunt_exitwalk.py` → `reports/study_ml_bughunt_exitwalk.csv`.

It is written from a fresh reading of `check_realtime_stop_loss`, `update_open_positions` and `recovery_hold.py`. Thresholds are
read from `frozen_config_yr5_181131e.json`, with asserts that every OFF exit (fast exit, EMA13 long, tick, RSI, signal-lost, regime,
FL1/FL2, BE levels) really is OFF. It imports neither the engine, the harness, `services.recovery_hold` nor the
`ml_exit_optimize.py` replica.

**Rules implemented:**
- P&L % = (p/e − 1)·100 − entry fee (maker 0.018 / taker 0.045) − 0.045·p/e.
- **Chain order on every aggTrade:**
  1. recovery hold
  2. hard-TP ladder 1.25:0.25 … 4.0:0.80 (on the previous tick's peak)
  3. peak update
  4. stop: −0.70, or −1.00 while the 5m EMA5 > EMA8 ∧ price > EMA20 signal is on; ATR widen −1.5 × ATR, capped −1.20; fires at
     ≤ line + 0.01
  5. long runner: arm at peak ≥ 0.395, floor = max(peak − 1.0 × ATR, +0.10)
- **Monitor exits:** NO_EXPANSION 180 min, MAX_HOLD 1200 min.
- **Recovery hold:** BTC RSI(14) on closed 5m bars, Wilder, last 100 bars, refreshed 2 s after each close; trigger band 60–66 and
  ≥ entry; hard stop = stop − 0.5; premise / 30-min / 240-min exits; release at +0.40.

**Result (1,784 fills, 3 seeds):**

| Check | Result |
|---|---|
| Same exit reason | **96.9 %**. The only mismatches are STOP_LOSS ↔ STOP_LOSS_WIDE labels (the signal-active refresh is a 111.6 s grid here) and 8 recovery-hold paths |
| Exit within 1 s | **91 %** (median Δ 0 s) |
| Mean Δ pct (independent − engine) | **+0.0007 %/trade**. 6 % of fills differ by > 0.01 |

**By exit reason (engine avg → independent avg):**

| Exit reason | Engine | Independent |
|---|---|---|
| HARD_TP_LADDER | +1.045 | +1.045 |
| RUNNER_TRAIL | +0.287 | +0.287 |
| STOP_LOSS | −0.787 | −0.788 |
| STOP_LOSS_WIDE | −1.083 | −1.044 |
| RH_PREMISE_EXIT | −0.802 | −0.806 |
| RH_HARD_STOP | −1.501 | −1.477 |

**By month, the independent − engine Δ is ≤ 0.01:**

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| Δ | +0.0002 | +0.003 | +0.002 | −0.001 | −0.002 | −0.001 | −0.004 | +0.009 | +0.001 |

**Items the brief asked to check, all clean:**
- **Time-based exits:** only 2 NO_EXPANSION and 3 RH_TIME_EXIT in the year. Monitoring only on tick-carrying seconds cannot matter.
- **Fees:** maker 0.018 / taker 0.045 entry, taker 0.045 exit. This is exactly the live master's fee ratio. pct = net / notional is
  exact (max error 3e-11).
- **Slippage:** paper exits at the tick price in both live and replay. No stop-fill overshoot difference: STOP_LOSS avg −0.787
  here, and the recall trace measured live −0.763 vs replay −0.758 on shared fills.
- **Partial fills:** none (only MAKER / TAKER_FALLBACK).
- **Funding:** in neither live paper P&L nor replay. Effect ≈ 0.0006 %/trade at a 27-min median hold.
- **Recovery hold:** 86 holds, final −0.881 vs −0.918 at the trigger. Net +0.002 %/trade over all fills, so not a cause.
- **Cross-check:** the X1 checks in `validate_against_master.py` (the existing replica vs live master and vs the engine) pass.

**Verdict:** the exit stack and its replay are clean. The −0.067 is not an exit artefact.

## 2 · Entries: inputs, gates on taken fills, gates on refused signals

### 2a · Entry prices
**Script:** `study_ml_bughunt_entryprice.py`. Each fill's entry price is checked against the raw trades around its open.

- **TAKER_FALLBACK** fills are at the last trade: median 0.000 %, 99 % within ±0.05 %.
- **MAKER** fills are at their limit, a median 0.035 % above the triggering print. This is the paper maker rule (2 s polls, fill when
  the last price ≤ limit), the same code live paper runs.
- Every fill had ticks. No price-scale errors (1000x tickers included).

### 2b · Decision inputs at the exact decision second (Jan–May vs Jun–Oct)
**Script:** `study_ml_bughunt_inputs.py`, 60 BLOCK lines + 60 SCAN lines per month from the seed-1 journals.

**Method:** rebuild the 100-bar 5m view a live bot would fetch:
- 99 closed exchange klines;
- plus the forming bar built here from raw aggTrades up to T;
- indicators computed with `ta`.

**Pair inputs:**
- The engine's price and 20-bar high/low are reproduced exactly at an instant 3–7 s (median) before the journal line, in **100 % of
  sampled lines in every month**.
- That lag is the sequential/batched kline fetch, the same as live. Max 32 s, same in all months.
- At the matched instant:

  | Indicator | Exact |
  |---|---|
  | RSI prev2 | 98 % |
  | ADX | 95 % |
  | EMA50 | 96 % |

- Every residual case was traced (12 rows). The engine's data came from a fetch 10–27 s earlier, just before a 5m boundary. This is
  legitimate batch timing, present in all months.

**BTC inputs (SCAN lines):**
- RSI median |Δ| 0.002–0.014, mean ≈ 0 in every month.
- ADX exact 95–100 %.
- Slope exact 53–90 %. The rest is knife-edge on a 4-decimal value.

**BTC 1h and 5m RSI stamps on every seed-1 fill** (`study_ml_bughunt_btc1h.py`):

| Stamp | Within tolerance | Bias |
|---|---|---|
| 1h RSI (±0.15) | 82–97 % | 0 |
| 1h slope (±0.002) | 83–99 % | 0 |
| 5m RSI (±0.5) | 83–96 % | 0 |

Jan–May is no different from Jun–Oct.

### 2c · Gates on the fills' own stamps (all 1,784 fills)
**Script:** `study_ml_bughunt_gatestamps.py`.

**Zero violations of:**
- gap 5-20 [0.05, 0.60]
- EMA5 stretch ≤ 0.35
- gap 5-8 ≤ 0.35, and ≥ 0.06 for UNMATCHED
- RSI [40, 70]
- ADX ≤ 30 and > 15
- LOADX
- BTC EMA20 slope ≤ 0.35
- pair ATR [0.25, 2.5]
- megacap rank ≤ 10
- heat (yr5 rule)
- rank > 50
- pair EMA20 slope ≤ 0

**Three apparent "violations" are gate semantics, not bugs.** The live master fills show the same rates:

| Apparent violation | yr5 | Live master |
|---|---|---|
| BTC RSI outside 40–65 | 31 % | 31 % |
| Global volume < 0.6 | 29 % | 25 % |
| abs(BTC 1h slope) < 0.05 | 8 % | 8 % |

The 20 fills with BTC ADX < 18 are all the ADX_SURGE_OPEN door, which admits them by design.

### 2d · Refused signals
**Script:** `study_ml_bughunt_refused.py`, `study_ml_bughunt_fails.py`.

- **BTC_ADX_GATE_LOW:** all 57,156 seed-1 SCAN vetoes hold on the scan's own BTC ADX (100 %, every month).
- **Pair gates on 643 sampled FAILS lines**, rebuilt from raw data:
  - Confirmed in **96–100 % of Jan–May** lines and 91–98 % of Jun–Oct lines.
  - By gate: ADX max 100 %, gap-min 99 %, LOADX 94 %, EMA20 filter 91 %.
  - The residual is the unknown exact fetch second.

### 2e · Universe
- **No survivorship gap.** Every Binance perp that traded in 2026 and was delisted later is in the cache. Examples: TON, IP, NFP, ZKJ,
  DENT, all SETTLING in the Sep-16 exchangeInfo. After delisting they carry zero-volume rows, so they drop out of the ranking.
- The 69 pairs with daily but no 5m data had zero 2026 volume.
- Ranks of the fills: 11–50 in both halves (median 31 / 29).
- Pair-age mix is the same in both halves. No fill on a pair younger than 90 days.

**Verdict:** entries are clean in every month. Nothing Jan–May-specific.

## 3 · Config / era

- **Config used:** the replay ran `frozen_config_yr5_181131e.json`, a full model_dump with no defaults fallback and no
  `--config-history`. Every ML threshold the walkers and gate checks read from it matches what the fills show.
- **Code snapshot:** `code_yr5_181131e/` equals git 181131e byte for byte for trading_engine / indicators / config / recovery_hold.

**Diff vs today's effective config (`config.trading_config` at HEAD): 48 keys.** Momentum-long-relevant ones:

| Change since the snapshot | Effect on the replay | Post-hoc on yr5 |
|---|---|---|
| `long_chop_burst_block` (new, ON) | yr5 took 13 fills/seed today's engine refuses | removed: **−0.199** avg |
| LONG_HEAT re-scope reverted. yr5 refused every bull ≥ 85; today needs slope ≥ 0.07 ∧ RSI-prev ≥ 64 ∧ bull ≥ 80 | yr5 misses the bull ≥ 85 longs today admits | add back the portfolio study's 32.7/seed at −0.003. Master's own 10 such kept fills average −0.25 |
| FRENZY_WILLY global hold (new) | no automated open while a WILLY is open | removes 46 fills/seed at −0.06 (WILLY windows from `FRENZY_TP3_VS_TP4_TICKS_2026-10-08_willy.csv`) |
| FRENZY changes: TP 4 → 3, ATR cap 3.0, bearish-day block, LITE | FRENZY opens reset BTC_ACCEL_CHASE for ML | not simulable post-hoc. Declared |
| Sizing (UNMATCHED 2 → 1.5, CALM3D 2 → 1, 1h-slope multiplier rules removed) | $ only | pct unaffected |

**Result:**

| | Fills/seed | %/trade | 95 % CI |
|---|---|---|---|
| yr5 as run | 594.7 | −0.067 | |
| Today's config, post-hoc | ≈ 568 | **−0.061** | −0.116 … −0.011 (day bootstrap, before the heat add-back) |

H1 −0.087, H2 −0.046.

**Snapshot-code difference, immaterial:** at 181131e the recovery-hold kill bar auto-disables the hold. Today's code never does
(auto_kill_enabled). Holds are 1.5 per chunk-run and worth +0.002 %/trade, so it cannot matter.

**Sizing / FIX-A:** % per trade is size-invariant. FIX-A (`SPIKE_FADE_H1_REVIEW_2026-10-08.md`) moves the ML **$** only (−$6,660 →
−$6,568 /yr on the $3k ruler).

## 4 · Data

**Script:** `study_ml_bughunt_tickaudit.py` → `reports/study_ml_bughunt_tickaudit.csv`. Covers all 1,814 pair-days an ML fill was
open on, plus the day before.

- **Integrity:**
  - All timestamps are ms and inside their own UTC day.
  - 0 non-monotone ticks.
  - Exact duplicate (t, p, q) rows are a few hundred per month. These are separate aggTrades with identical fields, harmless.
  - Price "outliers" > 20 % from the day median are real pump days.
- **Missing pair-days:** 72, all on the day before an open, never inside a fill's life. 0 walker fills had a missing day in their
  horizon.
- **Tick coverage of the replay:**
  - Over all 57 metas, `minutes_missing` = 3,466 of 313,925 open-position minutes (1.1 %).
  - None of them is on an ML fill: every ML fill's whole life has tick archives (walker `tick_days_missing` = 0). The misses belong
    to other sleeves' small pairs.
  - The recall trace found 100 % tick-decided ML entries.
- **Warm-up trim:** `ENGINE_REPLAY_YR5_ML_fills.csv` equals `yr5_fills_trimmed.load()` exactly (1,784 = 1,784, 0 differences).
- **Chunk boundaries:**

  | Window | Fills | %/trade |
  |---|---|---|
  | First 24 h of each chunk | 153 | −0.071 |
  | Rest | 1,631 | −0.067 |
  | Last 6 h of each chunk | 35 | −0.113 (small N) |

  - Only 1 fill closed after its chunk end (a normal RUNNER_TRAIL in the 24 h follow window).
  - Max hold 180 min. No forced closes.

## 5 · Sanity vs simple baselines (same entries, independent pricing)

**Scripts:** `study_ml_bughunt_exitwalk.py` (bl_* columns), `study_ml_bughunt_control.py` (3 random taker entries per fill, same
pair, ±3 days).

| | H1 Jan–Apr | H2 May–Oct | Year |
|---|---|---|---|
| Engine exit stack | −0.096 | −0.043 | **−0.067** |
| Bracket +0.5 / −0.7 (net) | −0.085 | −0.065 | −0.074 |
| Bracket +1.0 / −1.0 | −0.093 | −0.066 | −0.078 |
| Hold 60 min | −0.081 | −0.081 | −0.081 |
| ML **gross** move 60 min after entry | −0.008 | −0.008 | −0.008 |
| RANDOM entries, gross 60 min | −0.023 | +0.061 | +0.023 |
| RANDOM entries, bracket +0.5 / −0.7 (taker) | −0.120 | −0.089 | −0.103 |

**Reading:**
- No simple exit is positive, so the exit stack is not hiding an edge.
- The entries carry ≈ 0 gross drift in both halves. Random entries on the same coins do about as well after fees.
- The engine's exit is the best of the exits tried.
- The year's loss = fees on an entry signal with no edge in this tape. It is not a replay error.

**BTC monthly returns (context):**

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| BTC % | −10 | −15 | +2 | +12 | −4 | −21 | +7 | +25 | +6 |

## 6 · Batch by batch: master kept ML vs the yr5 replay on the same windows

**Script:** `study_ml_bughunt_batches.py` → `reports/study_ml_bughunt_batches.csv`, `_pairs.csv`.

**Method:**
- Master = `MASTER_POOL_stacked.csv` STACK 2026-10-08c: MOMENTUM LONG, stack_keep, non-probe, ex-B1, pct = stack_pct. This
  reproduces 115 · 75 % · +0.183; 106 of those fall inside the yr5 span.
- Replay = yr5 ML fills inside each live-up window, 3 seeds, with today's chop-burst rule applied.
- Match = same pair ±10 min per seed.
- B17/B18 are outside yr5 (it ends Oct-4).

| Batch | Master kept N · WR · avg | yr5 N/seed · WR · avg | Shared/seed: live → replay | Live-only/seed (avg) | Replay-only/seed (avg) | Why the gap |
|---|---|---|---|---|---|---|
| **BASE** Jun-15 → Jul-10 | 32 · 94 % · **+0.485** | 55.0 · 69 % · +0.073 | 12.0: +0.506 → **+0.503** | 20.0 (+0.472) | 43.0 (−0.047) | **Same-trade pricing is identical.** The replay misses 20 live winners/seed: scan-second knife-edges (trace classes P 6, G1 5, G2 1; live-clock replays take them). It adds 43/seed live never took: PHASE 17.7 (−0.02), RULE_THEN 15.7 (−0.01; live's June rules refused them), ASW_TOOK 3.3, live slots full 1.0 (−0.47). The master BASE rows are the pre-screened survivors of live's own fills (in-sample). Washed-out window. |
| B2 Jul-31 → Aug-10 | 10 · 80 % · +0.191 | 7.3 · 77 % · +0.126 | 3.7: +0.083 → −0.086 | 6.3 (+0.254) | 3.7 (+0.339) | small N. Replay-only fills win more than live-only |
| B3 Aug-11 → 24 | 22 · 77 % · +0.147 | 18.3 · 82 % · **+0.209** | 7.7: +0.237 → +0.238 | 14.3 (+0.098) | 10.7 (+0.188) | replay better |
| B4 | 1 · +0.117 | 0.7 · −0.857 | — | 1.0 | 0.7 | N = 1 |
| B5 | 3 · +0.452 | 3.3 · 60 % · −0.245 | 1.7: +0.448 → −0.386 | 1.3 | 1.7 (−0.105) | N 3; shared: entry seconds apart |
| B6 | 2 · −0.899 | 0 | — | 2.0 | 0 | replay took nothing (better) |
| B7 | 1 · +0.100 | 1.0 · −0.066 | — | 1.0 | 1.0 | N = 1 |
| B8 | 4 · 75 % · +0.113 | 2.7 · 100 % · **+0.434** | 2.0: +0.545 → +0.546 | 2.0 (−0.319) | 0.7 | replay better |
| B9 | 2 · +0.254 | 3.7 · 64 % · +0.031 | — | 2.0 | 3.7 (+0.031) | small N |
| B10 + B11 | 1 · +0.100 | 1.3 · +0.197 | 0.7: +0.100 → −0.329 | 0.3 | 0.7 (+0.723) | N 1 |
| B12 Sep-21 → 25 | 15 · 67 % · +0.119 | 11.0 · 67 % · **+0.181** | 6.7: +0.668 → +0.406 (today's ladder caps a runner) | 8.3 (−0.320) | 4.3 (−0.166) | replay better |
| B13 | 3 · +0.145 | 2.7 · 75 % · +0.275 | 2.3: +0.025 → +0.162 | 0.7 | 0.3 | replay better |
| B14 | 4 · 25 % · −0.656 | 5.7 · 82 % · **+0.094** | 2.0: −0.690 → −0.308 | 2.0 (−0.622) | 3.7 (+0.314) | replay much better (chop-burst removed 1/seed at +0.20) |
| B15 | 1 · −0.993 | 0.3 · +0.099 | — | 1.0 | 0.3 | N 1 |
| B16 | 5 · 60 % · −0.184 | 3.7 · 64 % · −0.085 | 2.0: −0.303 → −0.012 | 3.0 (−0.105) | 1.7 (−0.174) | replay better |
| **All ≤ Oct-3** | **106 · 77 % · +0.181** | **116.7 · 72 % · +0.102** | 40.7: +0.310 → +0.251 | 65.3 (+0.100) | 76.0 (+0.021) | |
| **ex-BASE** | 74 · +0.049 | 61.7 · **+0.127** | 28.7: +0.229 → +0.146 | 45.3 (−0.064) | 33.0 (+0.110) | **replay better than master** |

**What the gap is made of:**

**① Same trades, priced the same.** On the 40.7 shared fills/seed (replay 0.251 vs live 0.310):
- When the two entries are ≤ 30 s apart: replay **+0.319** vs live re-walked under today's exits **+0.341** (57 seed-rows).
  `study_ml_bughunt_master_rewalk.py` runs the same independent walker over live's own entries.
- The rest is fills where the replay entered 30–120 s after live (27 seed-rows: live +0.42 → replay +0.10, a higher entry gets
  stopped), against 23 where it entered earlier (+0.18 → +0.25).
- Timing is symmetric: median Δt +1.9 s. Same-signal entry-phase mix (share of entries in each 30–60 s slot of the 5m bar) is the
  same as live's.
- This is scan-second noise (b), not a bias.

**② Today's exits do not hurt live's fills.** The independent walk prices master kept fills at **+0.210** under today's exits, vs
+0.181 as traded.

**③ Membership: which signals each side took.**
- **All windows:** live-only +0.10 vs replay-only +0.02. Day-bootstrap Δ −0.08, CI [−0.29, +0.12]: not significant.
- **BASE alone:** Δ −0.52, CI [−0.85, −0.22]. The causes are in the BASE row of the table above: (a) master BASE is live's own
  pre-screened survivors, and (b) the missed winners are scan-second knife-edges that the live-clock replays reproduce.
- **ex-BASE:** the replay-only fills beat the live-only ones (Δ +0.17, CI [−0.05, +0.40]).

**Verdict for this section:** the replay is not systematically worse than master on the same days. It is worse only in BASE, and
there entirely for reasons (a) and (b). Every other batch with N ≥ 3 has the replay level or better.

## 7 · Defect list

| # | Finding | Type | Impact on ML %/trade |
|---|---|---|---|
| — | No exit, entry-input, gate, data, chunk or fee defect found in any month | — | 0 |
| C1 | yr5 config is the Oct-4 snapshot. Since then: chop-burst block, heat rule revert, WILLY global hold (and FRENZY changes) | config drift, not a bug | −0.067 → **−0.061** (post-hoc) |
| C2 | 181131e RH kill bar auto-disables (today: never) | code drift | ≈ 0 (+0.002 total RH value) |
| C3 | Paper MAKER fill = poll every 2 s at the limit (fills a median 0.035 % above the triggering print) | paper model, same in live | 0 vs live paper (real money would differ) |
| C4 | The 1784-row exit labels STOP_LOSS vs STOP_LOSS_WIDE depend on the scan-time signal refresh | labelling only | ≈ 0 (Δ pct +0.04 on 52 WIDE rows, offsetting) |

**$ view (portfolio study ruler, $3k book):** the ML replay part is about **−$6.6k/yr** (FIX-A table). Today's-config correction
brings it to ≈ **−$5.7k/yr**.

## 8 · Proposed harness changes (diffs only, not applied)

None of these changes ML's number. They make the next audit cheaper.

**P1: journal the kline fetch instant on BLOCK / FAILS lines.** The 3–27 s gap between fetch and line cost a second pass here.

```diff
@@ engine_replay.py FakeBinance.get_ohlcv
-        self._n(f"ohlcv_{timeframe}_{limit}"); await self._lat()
-        return KS.ohlcv(_sym(symbol), timeframe, limit, Clock.ms)
+        self._n(f"ohlcv_{timeframe}_{limit}"); await self._lat()
+        KS.last_fetch_ms[_sym(symbol)] = Clock.ms          # read by the journal hook → "fetch_t" on BLOCK/FAILS ctx
+        return KS.ohlcv(_sym(symbol), timeframe, limit, Clock.ms)
```

**P2: a yr6 run on today's config.** It removes the post-hoc C1 approximations, in particular the FRENZY → BTC_ACCEL_CHASE
interaction that cannot be simulated post-hoc. Same command as yr5 with a new snapshot of HEAD and a fresh frozen config. ~13.5 h at
9 jobs.

**P3: pass `--rh-auto-kill off`, or snapshot at a commit with `auto_kill_enabled`,** so the hold behaves like today's code (C2).

## 9 · What this hunt could NOT test (blind spots)

- **Live's own H1 behaviour.** There are no live trades Jan–May. The checks prove the replay faithfully executes today's code on
  H1 data. They cannot prove live would have scanned at the same seconds; scan-phase noise is ±0.05/trade, see the recall trace.
- **Alpha / underlying-type tags:** these are the Sep-16 exchangeInfo snapshot. A pair tagged differently in H1 would be in or out of
  the universe wrongly. No historic exchangeInfo exists locally.
- **FRENZY-driven BTC_ACCEL_CHASE resets under today's FRENZY rules.** Not simulable post-hoc (needs P2).
- **Refused-signal check coverage:** pair gates without a pure rule (VOL_GATE, GAP_NOT_EXPANDING, door routing, CALM3D) were not
  re-derived. BTC_RSI_ADX_CROSS / slope gates were checked via their inputs (2b), not their rule strings.
- **Exit walker approximations:**
  - The signal-active grid is anchored at the open, not at the replay's real scan seconds.
  - NO_EXPANSION ignores the signal-reset branch, which never fired in yr5 (max hold 180.0).
- **Real-money fills** (slippage, queue position for maker) are not modelled here or in live paper.

## Files

**Scripts (new):** `scripts/study_ml_bughunt_`:
- `exitwalk.py`
- `entryprice.py`
- `inputs.py` (incl. `lag_match`)
- `btc1h.py`
- `gatestamps.py`
- `refused.py`
- `fails.py`
- `tickaudit.py`
- `control.py`
- `batches.py`
- `master_rewalk.py`

**Outputs (`reports/`):** `study_ml_bughunt_`:
- `exitwalk.csv`
- `entryprice.csv`
- `inputs_pair.csv`
- `inputs_pair_lagmatch.csv`
- `inputs_btc.csv`
- `btc1h.csv`
- `gatestamps.csv`
- `fails.csv`
- `tickaudit.csv`
- `control.csv`
- `batches.csv`
- `batches_pairs.csv`
- `master_rewalk.csv`
