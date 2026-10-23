# LITE late × BTC falling (2026-10-23)

Research only. No engine, config, template, test or git change. Binance: 16 public 5m-kline calls (≈ 190 weight, cooldown checked) for
the master/live pairs and AVA / BTC to 10-23 06:00. Everything else came from caches. Pre-registration (written before any result):
`reports/LITE_LATE_BTCFALL_2026-10-23_PREREG.md`. `scripts/validate_against_master.py`: **ALL CHECKS PASS**.

## The answer in plain English

- **Yes, there is a losing cell, but it is not proven well enough to block.** LITE entries **≥ 12 h after the spike with BTC's 1 h
  EMA20 slope ≤ 0** (the engine stamp; that is exactly AVA 10-23) lose in the two-year backtest: 109 fills · 50 % WR · **−0.57 %/fill**
  vs +0.03 for every other LITE fill. That passes the expectancy bar over the whole period, survives a day-shifted-BTC null (family-wise
  p < 0.001), holds on all 3 seeds and with any month left out.
- **But almost all of the loss is 2025.** 2025: −0.97 (57 fills). 2026-H1: **+0.02** (32). 2026-H2: −0.36 (20). On 2026 alone (the
  out-of-sample half) it is −0.12, P(avg < 0) 0.63, which **fails** the pre-registered OOS gate. That is the same pattern as the existing
  LITE_BTC1H_FALLING line, which also fades in 2026-H1.
- **The cell is always worse than the rest of LITE in the same period** (−0.97 vs −0.07 · +0.02 vs +0.14 · −0.36 vs +0.17). That is
  the "route, don't block" shape in the pre-registration. So the verdict is an **observe-only scout line with a frozen gate**. No
  filter now.
- **Each axis alone does nothing useful.** Late alone (≥ 12 h): +0.04, *better* than early. BTC-falling alone: −0.21, which is the
  existing LITE_BTC1H_FALLING line (frozen gate, not met). Only the *combination* separates. That fits the theory: late in an episode,
  the coin's own buying has faded, so BTC direction decides. Late + BTC **rising** is LITE's *best* cell (+0.39).
- **Real fills are far too few to say anything.** Since go-live there are 3 cell fills (+2.80, −2.54, +2.00) plus AVA (−2.85). The
  master has 3. The scout needs ≥ 15 on ≥ 8 days.
- **What a block would be worth:** it removes ~7.5 % of LITE fills. Backtest: +$1.2k over 92 weeks at 8× (≈ $13/week, $6–9 after the
  haircut). **2026 only: +$120–185 over 40 weeks (≈ $3–5/week).** So it's small money either way.
- **AVA itself is a lagging-slope case.** BTC was **up** +0.78 % in the hour before entry (+0.8 % bounce at 02:30), but the 1 h EMA20
  slope still read −0.085. In the backtest those "BTC already bouncing" analogs lost too (65 · −0.45), so the slope stamp still flags
  the bad cohort. Raw BTC returns (15 m / 1 h / 4 h) do **not** separate. The effect is a slow trend read, not "BTC dropping right now".

## Cohort and engine parity

- **③ BT LITE:** yr6 2Y_TODAY engine replay (the real engine code on historical klines; the bearish-day block is on, as live),
  FRENZY_LITE, seeds 1/2/3 (seed 1 = analysis). Each fill is re-priced at **today's exit** by the LITE timecap tick/1m walker
  (`LITE_TIMECAP_2026-10-20_rows.csv` `base`: LITE C TP +3 iff vol ≥ 20 ∧ slope > 0 else +2 · −3 · 12 h · resting maker TP · 0.10 slip).
  Hours (`entry_frenzy_hours`) and slope (`entry_btc_1h_slope`) are the engine's own stamps. The cohort fires on engine signal bars by
  construction (the walker's validation: 96–99 % same exit kind vs the replay).
- **③ BT RETRY:** S15r replica of FRENZY_LITE_RETRY (engine functions, 97.9 % walker parity), shown apart. It has no slope stamp, so
  the slope is rebuilt from BTC 5m (EMA20 incl. the forming hour vs 3 bars back). Rebuilt vs stamp: sign agrees on 99.3 % of BT fills,
  r 0.998.
- **② master:** MASTER_POOL_stacked kept non-probe LITE + RETRY, `stack_pct` (33 fills, B18–B23).
- **① live:** post-go-live exports (export_glob, dedup (opened_at[:19], pair, direction), CLOSED wins), operator-closed out
  (AUSDT MANUAL, ICNT EXTERNAL_CLOSE). 11 fills to 10-21 02:14. AVA 10-23 03:00 isn't in any export. It is added from the operator's
  report (−2.85 %, 14.9 h, slope −0.085, TP +2), with BTC / coin features rebuilt from fetched klines.
- **Coverage:** hours, slope, BTC returns / EMA20 / DI and coin off-peak / vs-EMA50 are 100 % on every block. The 5m-EMA20 slope stamp
  is 0 % on RETRY.
- **LITE breakeven WR** (today's stack, |avg loss| / (avg win + |avg loss|)): BT 58.1 % (TP +2 fills 61.7 %, TP +3 52.1 %), master
  54.3 %. The bar uses 58 %. Every fill in the cell runs TP +2 (LITE C sends slope ≤ 0 to +2), so its real breakeven is ~62 %.

## Before → after, three blocks: primary cell (hours ≥ 12 ∧ BTC 1h slope ≤ 0)

N · days · WR · avg %/fill · Σ% · $ (LITE $1,900 notional at 8×, RETRY $950).

| block | before | cell (blocked) | after | Δ $ |
|---|---|---|---|---|
| ① live (op-closed out) | 11 · 5 d · 64 % · +0.26 · $+16 | 3 · 3 d · 67 % · +0.75 · $+43 | 8 · 62 % · +0.07 · $−27 | −43 |
| ① live + AVA 10-23 | 12 · 6 d · 58 % · −0.00 · $−38 | 4 · 4 d · 50 % · −0.15 · $−11 | 8 · 62 % · +0.07 | +11 |
| ② master all (B18–B23) | 33 · 13 d · 64 % · +0.48 · $+260 | 3 · 3 d · 67 % · +0.49 · $+28 | 30 · 63 % · +0.48 | −28 |
| ② per batch | B22 cell 2 · −0.27 · B23 cell 1 · +2.00 · B18–B21 0 | | | |
| ③ BT seed 1, 2025 | 826 · 55 % · −0.13 | 57 · 42 % · **−0.97** | 769 · −0.07 | +$1,053 |
| ③ BT seed 1, 2026-H1 | 371 · 61 % · +0.13 | 32 · 62 % · **+0.02** | 339 · +0.14 | −$13 |
| ③ BT seed 1, 2026-H2 | 255 · 61 % · +0.13 | 20 · 55 % · **−0.36** | 235 · +0.17 | +$135 |
| ③ BT seed 1, ALL | 1452 · 517 d · 58 % · −0.02 · $−483 | 109 · 92 d · 50 % · −0.57 · $−1,175 | 1343 · +0.03 | **+$1,175** |
| ③ BT seed 2 / 3, ALL | −0.03 / −0.01 | 110 · −0.61 / 109 · −0.60 (Δ vs rest) | | +$1,239 / +$1,171 |
| ③ BT RETRY ALL | 1701 · 54 % · −0.22 · $−3,587 | 133 · 53 % · −0.44 | 1568 · −0.20 | +$560 |
| ③ BT RETRY 2025 / 2026 | | 58 · **−0.07** (rest −0.25) / 75 · −0.73 (rest −0.15) | | |

Full per-seed, per-period and per-batch tables for every candidate are in `_tables.md`.

**Does it add a filter?** It doesn't meet the bar (see below). LITE_RETRY reverses in 2025. In the backtest the whole retry book is
negative anyway (−0.22).

## Pre-registered grid: BT seed 1, 33 tests, in full (`_grid.csv`, `_tables.md`)

Δ = cell − rest. Checks: expectancy bar · all 3 periods < 0 · 2026 OOS · family-wise day-shift null · seeds.

| test | cell N · avg | Δ (day 95 %) | 2025 / H1 / H2 cell avg | 2026 OOS avg · P | fam p | checks |
|---|---|---|---|---|---|---|
| H6 / H9 / H12 alone | 810 · +0.06 / 512 · +0.14 / 296 · +0.04 | +0.18 / +0.24 / +0.07 | — | — | — | none |
| S1 slope ≤ 0 | 479 · −0.21 | −0.29 (−0.59, +0.02) | −0.36 / +0.13 / −0.34 | −0.05 · 0.59 | 0.24 | bar ✔ only |
| S2 BTC 15 m < 0 | 712 · −0.03 | −0.02 | −0.12 / +0.03 / +0.19 | +0.09 | 0.96 | none |
| S3 BTC 1 h < 0 | 592 · −0.15 | −0.22 | −0.22 / −0.16 / +0.12 | −0.05 | 0.41 | none |
| S4 BTC 4 h < 0 | 524 · −0.06 | −0.07 | −0.42 / +0.29 / +0.47 | +0.36 | 0.89 | none |
| S5 BTC < 1 h EMA20 | 463 · −0.18 | −0.24 | −0.39 / +0.18 / −0.23 | +0.03 | 0.33 | none |
| S6 1 h −DI > +DI | 537 · −0.17 | −0.24 | −0.31 / +0.06 / −0.09 | +0.00 | 0.33 | none |
| S7 5m EMA20 slope < 0 | 547 · −0.19 | −0.28 | −0.33 / −0.03 / −0.03 | −0.03 | n/a | periods ✔ |
| C1 coin < 5m EMA50 | 94 · **+0.56** | +0.61 | — | — | — | (wins) |
| C2 coin ≥ 8 % off peak | 223 · −0.27 | −0.30 | −0.60 / +0.38 / +0.00 | +0.20 | — | none |
| H6 ∧ S1..S7 | −0.24 / +0.06 / −0.18 / −0.10 / −0.25 / −0.08 / −0.29 | | | | 0.21–1.00 | H6∧S7: bar ✔ periods ✔ |
| H9 ∧ S1..S7 | −0.22 / +0.13 / −0.09 / −0.09 / −0.23 / +0.02 / −0.18 | | | | 0.34–1.00 | none |
| **H12 ∧ S1 (primary)** | **109 · −0.57** | **−0.59 (−1.09, −0.10)** | **−0.97 / +0.02 / −0.36** | **−0.12 · 0.63** | **0.000** | bar ✔ null ✔ seeds ✔ · periods ✘ OOS ✘ |
| H12 ∧ S5 (< EMA20h) | 109 · −0.61 | −0.64 (−1.12, −0.15) | −1.04 / +0.01 / −0.61 | −0.21 · 0.73 | 0.000 | same as primary (≈ same fills) |
| H12 ∧ S2 / S3 / S4 / S6 / S7 | +0.10 / −0.10 / −0.27 / −0.21 / −0.37 | | | | 0.26–1.00 | none |

- **Survivors of every check: 0.** The primary and its twin S5 (BTC below its 1 h EMA20, ≈ the same fills) pass the bar, the null
  and the seeds. Both fail "negative in every period" (H1 ≈ 0) and 2026 OOS. **ROUTE-pattern** (cell < rest in all 3 periods,
  P(Δ<0) ≥ 0.90): primary, H12∧S5, S1, S3, S6, S7, H6∧S1, H6∧S5, H6∧S7.
- **Sign granularity across BTC measures:** the slow trend reads (1 h EMA slope, below the EMA, DI) carry it. The fast returns
  (15 m, 4 h) do not. 1 h return is weak. So "BTC falling" has to mean the *1 h trend*, not the last candle.
- **Interaction (the theory's test).** Late − early on falling BTC minus the same on rising BTC: 2025 **−1.31** · 2026-H1 **−0.44** ·
  2026-H2 **−0.06**. The sign is right in every period, but the effect shrinks toward zero.
- **Dose (tercile buckets, late row: fall / flat / rise):** ALL −0.27 / +0.32 / +0.43 · 2025 −0.62 / +0.28 / +0.47 · **2026-H1 +0.29 /
  +0.43 / −0.09 (reversed)** · 2026-H2 −0.03 / +0.26 / +0.89.
- **Leave-one-month-out:** Δ stays −0.48 … −0.70 over all 22 months. Month sign: the cell is below rest in 14 of 22 months.
  2026-03 (−4.2) and 2025-06 (−3.8) are the worst. 2026-05 / 06 / 10 are the best for the cell.
- **Concentration (cell, gross loss):** top day 4 %, top pair 5 %. It is spread out, not one crash day (2025-10-10 has 0 cell fills).

## Expectancy bar on the cell (WR < 58 % BE · P(avg<0) ≥ 0.95 day-clustered · ≥ 8 days, no day/pair ≥ 50 % · N ≥ 15)

| scope | cell | WR | P(avg<0) | avg 95 % | ①②③④ |
|---|---|---|---|---|---|
| BT s1 ALL | 109 · 92 d | 50 % | 0.99 | (−1.00, −0.10) | ✔ ✔ ✔ ✔ |
| BT s1 2026 | 52 · 46 d | 60 % | 0.63 | (−0.78, +0.60) | ✘ ✘ ✔ ✔ |
| BT RETRY ALL | 133 · 110 d | 53 % | 0.98 | (−0.89, −0.00) | ✔ ✔ ✔ ✔ (but 2025 reverses) |
| master | 3 · 3 d | 67 % | 0.34 | — | ✘ ✘ ✘ ✘ |
| live + AVA | 4 · 4 d | 50 % | 0.56 | — | ✔ ✘ ✘ ✘ |

**Counterfactual of a block (BT, today's sizing):**
- Full period: +$1,175 / +$1,239 / +$1,171 (seeds 1/2/3) over 92 weeks, or **+$6–9/week after the 30–50 % haircut**.
- 2026 only: +$121 / +$185 / +$185, or about $2–3/week after the haircut.
- It removes 109 fills (7.5 %), and 50 % of them are winners (+2 each).
- A half-size route would keep half of both the loss and the winners: Δ ≈ ½ of the above.

## AVA 2026-10-23 03:00 (the motivating case)

- **Entry stamps:** 14.9 h after the spike · stamped slope −0.085 (my rebuild −0.017, same sign) · LITE C TP +2.
- **BTC at entry:** 15 m +0.13 % · 1 h **+0.78 %** · 4 h +0.77 % · price **above** its 1 h EMA20 (+0.18 %) · 1 h DI spread −8.8.
- **Coin at entry:** 6.5 % below its post-spike high, 1.6 % above its 5m EMA50 at the entry bar. My rebuild doesn't reproduce the
  "below EMA50" reading; the operator's reading is probably from an earlier bar.
- **Which cells it falls in:** the primary, H12 ∧ S6 (DI), and the axes alone. It is *not* in any return-based cell.
- **Analogs in BT seed 1:**
  - late ∧ slope ≤ 0 ∧ BTC already bouncing (1 h return > 0): 65 · 52 % · −0.45.
  - late ∧ slope ≤ 0 ∧ BTC still falling: 44 · 48 % · −0.74.
  - So AVA's type loses in the backtest even with BTC bouncing.
  - Post-hoc (AVA-derived, **not** pre-registered, no evidential weight): late ∧ slope ≤ 0 ∧ coin ≥ 6 % off peak = 45 · 40 % · −1.12.
    This one is a candidate sub-read for the scout, not evidence.
- **Null question:** "real fills show nothing" is a sample-size answer (3–4 fills), not a refutation. The backtest effect is real in
  2025 and weak in 2026.

## Recommendation

**Don't add a filter, and don't change the size now.** Register an **observe-only scout line LITE_LATE_BTCFALL** with the frozen gate
below. It mirrors WILLY_BTC1H, the same structure the operator asked for. If the gate fires, the operator chooses between a **block**
and a **route to half size** (the 2026 data fits "weak, not clearly negative" better, which argues for ½).

It sits **inside** the existing LITE_BTC1H_FALLING line (slope ≤ 0, frozen DL 288, 6 zone fills · +0.48 at the 10-20 scout read). The
new line is its late sub-cell. Both lines keep their own frozen gates. Neither is re-fitted.

### Scout spec: LITE_LATE_BTCFALL (pre-registered 2026-10-23, never re-fit)

- **Cohort:** CLOSED FRENZY_LITE **and** FRENZY_LITE_RETRY fills (each sleeve also read separately), non-MANUAL, operator-closed
  excluded (`scout_gatekit.split_operator_closed`).
  - Opened ≥ the commit that registers the line + 10 min.
  - Pre-registration fills (AVA 10-23 03:00 and earlier) are listed apart and never counted.
  - Dedup (opened_at[:19], pair, direction); CLOSED wins; never id.
- **Cell (frozen):** `entry_frenzy_hours` ≥ **12.0** ∧ `entry_btc_1h_slope` ≤ **0** (the engine stamp). A missing stamp counts as
  UNSCORED, is listed apart, and is never put into "rest".
- **Comparator:** same-period LITE-family fills outside the cell.
- **Context columns (descriptive only, never gated):**
  - early ∧ falling and late ∧ rising sub-splits;
  - `entry_btc_ema20_slope` sign;
  - coin off-peak ≥ 6 %, rebuilt from cached 5m when available, else blank.
- **P&L:** actual `pnl_percentage` (leverage-invariant, today's LITE C exit as traded; no re-walk). $ is shown as traded.
- **Units:** DAY (UTC). The cell is partly a market-wide switch, so per-day concentration is checked like per-pair.
- **🔒 Gate, decided ONCE** at the first (open time, pair) prefix where the cell reaches **N ≥ 15 on ≥ 8 UTC days**:
  - cell avg ≥ rest avg → **RETIRE**;
  - **ARM CANDIDATE (operator decides block vs ½ size)** iff all of these hold:
    - cell WR < LITE breakeven, computed on that same read's scored LITE fills (fallback 62 %, the TP +2 breakeven, while
      LITE N < 30);
    - day-clustered bootstrap P(cell avg < 0) ≥ 0.95 (4,000 draws, seed 7);
    - no day or pair carries ≥ 50 % of the cell's gross loss;
  - otherwise **KEEP OBSERVING**, with no second look at a looser bar.
  - Whatever is armed carries a pre-committed revert: back off if the first 15 post-arm blocked-cell shadow fills average > the kept
    LITE fills.
- **At each batch review it prints:**
  - cell vs rest vs unscored: N · days · WR · avg % · Σ$ · worst;
  - a per-fill list (pair, time, hours, slope, TP, pnl);
  - progress to the gate (N / 15, days / 8);
  - the same split for LITE and RETRY separately;
  - the backtest reference line, not counted: "BT 109 · 50 % · −0.57 (2025 −0.97 · H1 +0.02 · H2 −0.36; 2026 OOS −0.12, P 0.63)".
- **Stamps on live LITE fills (all 9 / 9 post-go-live LITE rows, export-only, no network):**
  - `entry_frenzy_hours`, `entry_btc_1h_slope`, `entry_frenzy_spike_at`, `entry_frenzy_vol_mult`, `frenzy_lite_tp_used`;
  - `entry_btc_ema20_slope`, `entry_btc_trend_gap_pct`, `entry_btc_1d_ret_pct`, `entry_btc_rsi_1h`, `entry_btc_off24h_pct`,
    `entry_btc_r72_pct`;
  - `entry_frenzy_vs_vwap_pct`, `entry_frenzy_run_pct`, `entry_frenzy_bar_ret_pct`, `entry_pair_ema20_ema50_gap_pct`;
  - RETRY also has `entry_frenzy_retry_wait_min` / `_green_ret`.
  - **Not stamped:** BTC 15 m / 1 h / 4 h return, BTC 1 h DI, and coin off-peak since the spike. These are rebuildable from klines
    only. They aren't needed for the gate.

## What could not be tested / blind spots

- **Live fills after 10-21 02:14.** No export exists. AVA 10-23 comes from the operator's report; any other LITE fills 10-21 → 10-23 are
  missing.
- **Backtest vs live.** The BT is the yr6 replay as run under +3 (slot / pair-held interactions under LITE C and RETRY aren't re-run).
  The walker uses 1m bars where ticks are missing (2025 mostly). The RETRY replica isn't the engine.
- **Coin "below 5m EMA50 / % below peak" at the operator's bar.** My rebuild uses the last closed bar at entry and an approximate spike
  time for AVA.
- **The 5m-EMA20 slope (S7)** has no rebuild, so it has no null and isn't on RETRY.
- **2025's universe** uses the backtest's 2025 cache (delisted coins included). The 2026-H1 bear is the period where both BTC-falling
  lines fade. No regime explanation was searched for. Re-fitting one now would be fitting on the data that failed.

## Files

- `reports/LITE_LATE_BTCFALL_2026-10-23_PREREG.md` (frozen before results)
- `reports/LITE_LATE_BTCFALL_2026-10-23_fills.csv` (per fill, all blocks, all features)
- `reports/LITE_LATE_BTCFALL_2026-10-23_grid.csv` and `_tables.md` (every table: coverage, the full grid, before→after for 6
  candidates × all blocks / seeds / periods, the bar, LOMO, CF, AVA)
- Scripts:
  - `scripts/study_lite_late_btcfall_build.py` (cache only)
  - `scripts/study_lite_late_btcfall_fetch.py` (16 budgeted calls)
  - `scripts/study_lite_late_btcfall_analyse.py`
