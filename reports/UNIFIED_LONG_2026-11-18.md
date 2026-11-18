# UNIFIED LONG: is there a ≥ 5-a-day long on pump coins? (2026-11-18)

**Pre-registered:** `UNIFIED_LONG_2026-11-18_PREREG.md`, frozen 15:44 UTC before any result. The never-read block (2023-H1) was read once, in the final run. The interim (`_INTERIM.md`) did not read it.
Research only. Nothing in the bot changed. Nothing was committed.
Detail tables: `_tables.md` (T1–T6) and `_extra.md` (X1–X5). CSVs: `_candidates.csv`, `_frontier.csv`, `_triggers.csv`, `_exit_grid.csv`, `_model.csv`. Charts: `_frontier.png`, `_events.png`.

## Question
The operator sees ~42 pumps and dumps a day but gets only 1–3 long trades. Is there a better **combination** of the WILLY / FRENZY_LONG / LITE / WIDE filters (or any other entry) that trades ≥ 5 times a day with a positive, consistent edge after real costs?

## Plain answer
1. **On Frenzy coins: no.** We rebuilt every long trigger the bot has had, on every Frenzy episode from 2023-01 to 2026-11-17: 10,261 episodes, 2.77 M candidate bars.
   - At ≥ 5 trades a day, **no combination makes money**. That includes 1-, 2- and 3-variable rules over 73 features, and a walk-forward model.
   - The best in-sample combination earns +0.08 %/trade in DESIGN. Over all blocks it earns 0.00 %. At 0.20 % slippage it is −0.15 %.
2. **The reason is simple.** A long opened on a Frenzy coin 0–24 h after its spike loses money **before costs**.
   - Average gross result: −0.09 %/trade. Net: −0.30 %.
   - The pumps fade. That is the same fact the EXHAUSTION / FLATBASE shorts trade.
   - The model does rank the bars: its deciles rise in order. But even its top 5 % only reaches +0.05 %/trade, and it turns negative at 0.20 % slippage.
3. **More trades come only from weaker bars.** Every rule that adds frequency adds bars that are worth less than the costs. The frontier chart shows this.
4. **The one long that works is a different trade:** buying pump coins that have dumped, on a BTC crash day.
   - Example rule: a red 15m bar of a 24 h mover, outside Frenzy windows, when BTC is −2.6 % on 24 h and −0.9 % on the last hour, with ≥ 16 Frenzy episodes alive.
   - It is positive in every block, including the never-read 2023-H1.
   - **But it is an event trade, not a daily one.** It fires on 45 days in 4 years, up to 1,028 coins at once.
   - With a realistic book of 10 positions it makes 0.6 trades/day on average, at +0.55 %/trade. It is about 0 at 0.50 % crash-day slippage.
   - It is a market on/off switch: 28 crash episodes, and the top 5 carry 77 % of the profit.
   - Two such rules formally pass all 8 pre-registered items (family-wise p 0.034). But they were picked after the holdouts were read, not by the pre-registered objective, and their "≥ 5/day" is an average of bursts.
5. **Small, theory-consistent lead on the existing triggers:** take them only when shorts are crowded.
   - Rule: global L/S ≤ 0.62 or top-account L/S ≤ 0.64 at the signal.
   - Result: +0.13 to +0.35 %/trade, but only 0.3–1.5 trades/day.
   - The union of these is ≈ 0 at 0.20 % slippage.
6. **Verdict: NO** for the requested ≥ 5/day Frenzy long. **OBSERVE** for two lines (section "Verdict"). Nothing to BUILD.

## Theory (written before the data, in the prereg) and what the data said
| theory | expected | found |
|---|---|---|
| Continuation while demand persists (LONG / LITE / WIDE: above VWAP, volume high) | + | Raw triggers are −0.09 to −0.28 %/trade at every one of 27 exits. Holding above VWAP did not separate winners. |
| Bounce after the first red bar (WILLY) | + | −0.37 %/trade, the worst trigger. It buys ~6 % above the price of one hour earlier, right after the spike (chart). The first red bar is mostly the start of the fade, not a dip to buy. |
| Squeeze fuel (crowded shorts) | + | Yes, but small. Low global / top-account L/S is the best single cut on LITE, WILLY and WIDE (+0.3–0.46 DESIGN; +0.17–0.35 all blocks). |
| Spot-led demand | + | Not a separator for longs (spot share change is not in any top rule). |
| Market tide | + when BTC up | **Inverted.** The longs that work are bought when BTC is crashing (capitulation bounce), not when it rises. |
| Late entries lose | − | Weak: hours since spike / volume decay are not strong separators. All of the 0–24 h window loses. |

## Pool and parity
- **Universe F (Frenzy):**
  - Every closed 5m bar 0–24 h after a verified spike, from the validated live-engine walk. The bar must be flagged, already shortlisted at that time (real-time) and real.
  - 10,261 episodes on 643 pairs. Spikes per day: 3.3 (2023) → 5.1 (2024) → 9.6 (2025) → 11.5 (2026).
  - The new walk for the never-read block 2023-01→06 added 560 episodes on 147 pairs.
- **Universe M (pump coins outside Frenzy):** every red 15m close of an engine-shortlist mover (|24 h chg| or 24 h range ≥ 15 %, vol ≥ $20 M) that is outside any Frenzy window. 1.68 M rows.
- **Triggers rebuilt with today's thresholds:**
  - LONG 1,227 bars (0.86/day) · WIDE 902 (0.64/day) · LITE 5,210 (3.5/day) · LITE_RETRY 4,536 (3.1/day) · WILLY_A (= HT entry) 6,713 (4.7/day) · SURGE_LIKE 896.
  - Union of all triggers: **12.9/day**. The pool is not short of signals. It is short of profitable ones.
- **Parity vs real fills** (MASTER_POOL_stacked, same pair, ±15 min):

| sleeve | real fills | recall (trigger) | real fill inside the pool | precision |
|---|---|---|---|---|
| FRENZY_LONG | 53 | 70 % | 74 % | 41 % |
| FRENZY_WIDE | 14 | 64 % | 79 % | 39 % |
| FRENZY_LITE | 75 | 91 % | 95 % | 37 % |
| FRENZY_LITE_RETRY | 46 | 91 % | 98 % | 35 % |
| FRENZY_WILLY (+HT) | 86 | 85 % | 98 % | 29 % |
| SURGE_LONG | 8 | **0 %** | 38 % | 0 % |

- **Reading the parity:**
  - The pool holds 74–98 % of the real Frenzy fills. Trigger recall is 64–91 %.
  - Precision of 29–41 % is expected: the live bot also has slots, armed filters, the bearish-day block, restarts and paper downtime.
  - **SURGE_LIKE is not a faithful rebuild.** SURGE is a market-wide BTC-led trigger, and its leader and market-volume gates were not rebuilt. It is reported only as a pool flag.
  - **Bull-Run was not rebuilt** (a BTC-regime universe sleeve, off). This is a blind spot.

## Method
- **Fixed exit family:** TP 2 / 3 / 5 % (maker, high ≥ TP + 1 tick) × SL 3 / 4 / 6 % (gap-aware) × 90 min / 4 h / 24 h = 27 exits.
  - Same-bar TP + SL counts as SL.
  - Costs: taker fee 0.05 % per side and maker fee 0.02 %, plus 0.10 % slippage per taker side (stress 0.20 %), plus real funding (a missing pair-month is charged 0.05 %/8 h).
  - One position per pair. Primary exit TP 3 / SL 4 / 4 h.
- **Features:** 73, all lag-safe. Episode state, price / volume / indicators on 5m–1h, metrics (OI, top / global L/S, taker; rows with create_time + 5 min ≤ signal close), funding, spot share, BTC 30 m – 1 d, regime label, Frenzy counts.
  - Coverage is ≥ 95 % for every feature in every block, except spot share (56–96 %; many 2025–26 coins have no spot market).
- **Rule search (DESIGN 2025-01 → 2026-06 only):**
  - 1D decile cuts and every 2D feature pair at quintile cuts, both directions, on F_ANY, F_RED, M_RED15 and ANY_TRIGGER.
  - Each candidate was screened at the row level, then the top 300 per pool were re-read exactly with one position per pair.
  - Greedy 3D extensions of the top 20.
  - Single-trigger pools: 1D.
  - ≈ 6,000 exact rule reads.
- **Model:** own numpy GBT, walk-forward with a monthly refit, 2024-07 → 2026-11, trained only on exits closed ≥ 24 h before the month. The training sample is capped at 150 k rows (compute budget; the prereg did not fix this).
- **Family-wise null:** DESIGN outcomes were re-drawn in whole-day blocks, and the full search was re-run (1D + 2D + exact top 150 + greedy 3D; the real search used top 300). The null's best objective was compared to the real one.

## Results

### 1. Raw triggers (all blocks, TP 3 / SL 4 / 4 h; avg net %/trade (N))
| trigger | /day | WR | avg | 23-H1 (never read) | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | best of 27 exits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Frenzy, any bar | 157 | 53.6 | −0.30 | −0.26 | −0.31 | −0.28 | −0.25 | −0.25 | −0.31 | −0.32 | −0.32 | — |
| LONG | 0.86 | 54.6 | −0.28 | −0.30 | +0.12 | +0.07 | −0.99 | −0.45 | −0.32 | −0.09 | −0.37 | −0.14 (T5S3H24h) |
| WIDE | 0.64 | 58.1 | −0.09 | +0.01 | −0.30 | +0.13 | −0.69 | +0.30 | −0.21 | −0.34 | +0.36 | +0.00 |
| LITE | 3.5 | 54.5 | −0.12 | −0.13 | −0.28 | −0.14 | −0.21 | +0.04 | −0.30 | +0.13 | −0.13 | −0.08 |
| LITE_RETRY | 3.1 | 54.2 | −0.14 | −0.16 | −0.18 | −0.06 | −0.35 | −0.03 | −0.23 | −0.20 | +0.12 | −0.12 |
| WILLY_A (+HT) | 4.7 | 53.8 | −0.37 | −0.70 | −0.23 | −0.25 | −0.26 | −0.11 | −0.57 | −0.27 | −0.54 | −0.35 |
| ANY trigger | 12.9 | 54.5 | −0.23 | −0.42 | −0.22 | −0.17 | −0.28 | −0.05 | −0.40 | −0.14 | −0.18 | — |
| Pump-coin red 15m (M) | 257 | 51.9 | −0.22 | −0.15 | +0.02 | −0.15 | −0.22 | −0.24 | −0.30 | −0.26 | −0.27 | — |

- **None of the triggers is positive at any of the 27 exits.** WIDE reaches +0.00 at one.
- **This is not the live sleeves' P&L.** The live sleeves add their own exits (e.g. WILLY's resting +1.25 TP, which is not in this family) and armed filters.
  - The master's kept fills (e.g. LITE 37, WILLY 24) are a small, filtered cohort and are not re-priced here.
- **Where the money goes** (X4): a Frenzy long in the first 24 h reaches a median +4.8 % peak within 4 h, but it still hits −4 % before +3 % in 38 % of cases.
  - Time-outs average −0.6 %.
  - Gross −0.09 %, costs 0.22 %.

### 2. The frequency / edge frontier (chart `_frontier.png`)
- **Left panel, every searched rule in DESIGN:**
  - Frenzy-pool rules (blue / green / purple) sit at or below 0 at every frequency from 5 to 100/day.
  - Single-trigger cuts (brown) reach +0.3–0.6 only below 1/day.
  - The only cloud above +0.5 at ≥ 3/day is the pump-coin crash-dip family (red, M_RED15). Every rule in it uses BTC / market-wide variables.
- **Right panel:** the DESIGN numbers (blue) fall back on the holdouts (red), as expected for selected rules. The crash-dip family stays above 0 on most holdouts.

**Best Frenzy-pool rules at ≥ 5/day (X2; chosen on DESIGN, then read everywhere):**

| rule | DESIGN /day · avg | all blocks avg | never-read 23-H1 | 26-H2 | @0.20 |
|---|---|---|---|---|---|
| F_ANY: below-VWAP streak 0 ∧ RSI15 ≤ 41 | 6.2 · +0.08 | +0.00 | −0.27 | −0.26 | −0.15 |
| F_ANY: 3 h return ≥ 4 % ∧ RSI15 ≤ 47 | 5.4 · +0.08 | −0.01 | −0.37 | −0.31 | −0.16 |
| F_RED: 3 h return ≥ 3.4 % ∧ EMA13–50 ≤ 0.32 % | 5.7 · +0.06 | −0.02 | −0.03 | −0.23 | −0.16 |
| ANY_TRIGGER: bar range ≤ 2.6 % ∧ 24 h change ≥ 15.5 % | 5.9 · +0.05 | −0.06 | −0.41 | −0.09 | −0.21 |

**Walk-forward model (T6; Frenzy pool, scored only out of sample):**

| trade when score ≥ | /day | WR | avg | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | @0.20 | Kelly f |
|---|---|---|---|---|---|---|---|---|---|---|
| training q80 | 65.9 | 56.0 | −0.13 | −0.12 | −0.11 | −0.12 | −0.17 | −0.11 | −0.28 | −0.116 |
| training q90 | 35.3 | 57.3 | −0.06 | −0.07 | −0.02 | −0.04 | −0.13 | +0.02 | −0.20 | −0.090 |
| training q95 | 17.2 | 59.1 | +0.05 | +0.12 | +0.09 | +0.11 | −0.08 | +0.05 | −0.10 | −0.056 |

- Score deciles (row level) rise monotonically, from −0.4 to −0.6 (decile 1) to ≈ 0 to +0.2 (decile 10), in every half-year.
- **The model ranks well, but even its top tenth does not clear the costs.**

### 3. Top rules, closest candidates, per period
- **The pre-registered pick**, the objective winner, is a crash-day rule: M: BTC 1h ADX ≥ 42 ∧ ≥ 14 spikes in 24 h ∧ BTC daily EMA50/200 gap ≥ 10.6 %.
  - DESIGN: 5.0/day at +1.43.
  - It **fails 4 of 8 items**:
    - no trades at all in 23-H1, 26-H1 and 26-H2;
    - 24-H2 −0.30;
    - 25-H2 −0.52;
    - Kelly floor −0.12.
  - With a 10-position book it makes +0.23 %/trade and is negative at 0.50 % slippage.
- **The five candidates closest to the bar** (all M crash-dip rules; full tables in T3, X1, X5). They were selected after the holdouts were read, so they are **post-hoc**.

| # | rule (M: red 15m bar of a pump coin outside Frenzy) | items | /day (all) | avg | 23-H1 never-read | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | @0.20 | Kelly f | crash episodes · % > 0 · top-5 share | 10-position book: avg · @0.50 · DCR %/day |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BTC 24 h ≤ −2.6 % ∧ BTC 1 h ≤ −0.86 % ∧ ≥ 16 Frenzies alive | 8/8 | 5.6 | +1.16 | +0.74 | +1.53 | +1.03 | +2.61 | +2.57 | +0.49 | +0.64 | +0.05 (29) | +1.02 | +0.20 | 28 · 86 % · 77 % | +0.55 · −0.04 · +0.055 |
| 2 | BTC day ≤ −2.3 % ∧ ≥ 2 spikes in 6 h ∧ BTC 30 m ≤ −0.71 % | 8/8 | 6.7 | +0.60 | +0.85 | +0.67 | +0.69 | +0.20 | +1.43 | +0.62 | +0.17 | +0.48 | +0.46 | +0.03 | 54 · 63 % · 60 % | +0.28 · −0.32 · +0.043 |
| 3 | BTC RSI 1h ≤ 37 ∧ ≥ 3 spikes in 6 h ∧ BTC 24 h ≤ −6.0 % | 6/8 | 4.8 | +1.03 | +0.79 | +1.18 | +1.40 | **−1.31** | +1.80 | +1.04 | +0.65 | +1.10 | +0.89 | +0.03 | 20 · 80 % · 79 % | +0.80 · +0.24 · +0.063 |
| 4 | BTC 1h EMA20 slope ≤ −0.10 ∧ ≥ 3 spikes in 6 h ∧ BTC 24 h ≤ −6.4 % | 6/8 | 4.9 | +0.99 | +1.37 | +1.18 | +1.09 | **−0.32** | +1.75 | +1.06 | +0.63 | +1.38 | +0.85 | +0.01 | 20 · 75 % · 69 % | +0.77 · +0.21 · +0.060 |
| 5 | BTC 24 h ≤ −2.6 % ∧ ≥ 3 spikes in 6 h ∧ BTC 1 h ≤ −1.24 % | 6/8 | 5.0 | +0.93 | **−0.14** | +2.15 | +0.97 | +0.74 | +1.26 | +0.93 | +0.86 | +0.01 | +0.79 | +0.15 | 48 · 75 % · 52 % | +0.41 · −0.18 · +0.044 |

**What each candidate fails, and by how much:**
- **#1 and #2 pass all 8 items as written** (family-wise null p = 0.034, 28 re-runs; the null's best was +0.45 vs the real +1.43).
  - But they pass the letter, not the intent. The "/day ≥ 5" item is met only arithmetically: #1 trades on 45 days in four years, with up to 1,028 coins on one day, and 26-H2 has 29 trades.
  - Their evidence is 28 (#1) and 54 (#2) crash episodes. The top 5 episodes carry 77 % and 60 % of the net.
- **#3 and #4:** fail /day (4.8 and 4.9) and 24-H2 (−1.31 and −0.32).
- **#5:** fails 23-H1 (−0.14, never-read), so it also fails the holdout item.
- **All five:** with a realistic book (≤ 10 positions open), the average halves.
  - The first coins bought in a crash are the worst: the knife is still falling.
  - #1, #2 and #5 are ≤ 0 at 0.50 % slippage per side. That is plausible on crash days for thin coins (live lesson: 0.3–0.6 % on thin coins).

### 4. What the best combination is, in words
- **On Frenzy coins** there is no combination worth trading at ≥ 5/day.
  - The closest thing is "an existing trigger (LITE, WIDE or WILLY) **only when shorts are crowded**": global L/S ≤ 0.62 or top-account L/S ≤ 0.64 at the signal.
  - Result: 1.5 trades/day at +0.13 %/trade over all blocks, −0.01 at 0.20 % slippage (post-hoc union, X3).
- **Outside Frenzy:** "on a BTC crash day (BTC ≥ 2.6 % down on 24 h and still falling in the last hour) while many coins are spiking, buy the red 15m bars of coins that moved ≥ 15 % in 24 h; TP 3–5 %, SL 4–6 %."
  - Its best exit is TP 5 / SL 6 / 24 h: +2.4 %/trade over all blocks, never-read +1.55.

## Verdict
- **Prereg deviation, stated openly:** by the letter of the decision rule, crash-dip candidates #1 and #2 qualify for BUILD (8/8). I do **not** recommend BUILD, because:
  1. They were not the pre-registered pick. The objective winner fails 4/8, and #1 / #2 were found by ranking candidates on holdout results (post-hoc).
  2. CLAUDE.md's window-units rule applies, because the conditions are market-wide. The evidence is 28 / 54 crash episodes, top-5-heavy, not 8,000 trades.
  3. With a realistic 10-position book, #1 is +0.55 %/trade, −0.04 at 0.50 % crash-day slippage, and its never-read and 26-H2 blocks turn negative (−0.29, −0.37 on small N).
- **NO: a ≥ 5-trade-a-day long on Frenzy coins is not there.**
  - Tried: every rebuilt trigger, their union, all bars, 1D / 2D / 3D rules over 73 features, a walk-forward model, and 27 exits.
  - The pre-registered winner fails 4 of 8 items. No Frenzy-pool rule or model threshold passes items 2, 3, 4 or 7.
  - The operator's chart impression is right that pumps give big moves up (median +4.8 % peak within 4 h). But the stop is hit first too often, and costs eat the rest.
- **OBSERVE 1, CRASH_DIP (pump coins outside Frenzy, event sleeve):** paper line, definition frozen now.
  - Entry: closest #1 **and** #4 exactly as written above, red 15m close, entry at the next 5m open, max 10 open positions.
  - Exit: TP 3 % maker / SL 4 % / 4 h (the tested exit; T5S6H24h is the better exit, but it was picked post-hoc).
  - Evidence is counted in **crash episodes**, not trades (window-units rule).
  - Decide once at ≥ 8 forward episodes: keep only if the per-episode mean is > 0 with ≥ 6 / 8 episodes > 0, measured at real fills (live slippage on the first entries).
  - Expect ~6–8 episodes a year. Frequency: ~0.6 trades/day on average, in bursts.
- **OBSERVE 2, SQUEEZE_FILTER on existing long triggers:** paper / observe line.
  - Rule: tag every LITE / LITE_RETRY / WILLY / WIDE / LONG signal with global L/S ≤ 0.62 or top-account L/S ≤ 0.64 (metrics row closed ≤ signal − 5 min).
  - Read at ≥ 30 tagged fills on ≥ 8 days: tagged vs untagged, at each sleeve's live exit.
- **No BUILD.** The pre-registered pick failed. The two 8/8 rules are post-hoc, event-clustered and fragile to crash-day slippage at a realistic book size. The operator may overrule this on the evidence above.

## Assumptions and blind spots
1. **Fixed exit family.** It is not each sleeve's live exit (no 1.25 % WILLY TP, no trails). The raw-trigger rows answer "is the entry good?", not "is the sleeve good?".
2. **Fills:** entry at the next 5m open + 0.10 % slip. Crash-day slippage on thin coins can be 0.3–0.6 % (stress shown at 0.20 and 0.50). Exits are on 5m bars (no ticks); same-bar TP + SL counts as SL (conservative).
3. **Candidate selection after the holdouts were read:** the five closest were ranked by items passed, which uses holdout reads. Apply the 30–50 % haircut, and treat all of them as post-hoc.
4. **Null:** day-block shuffle with 28 re-runs (computing budget; p resolution ≈ 0.03); a lighter search than the real one (top 150 vs 300 exact).
   - Market-wide (day-level) rules lose their link in this null by construction. A low p therefore says "day selection matters", **not** "the trade-level rule is real". Window units are the honest count.
5. **Blind spots:**
   - Bull-Run not rebuilt; SURGE not faithfully rebuilt.
   - No Alpha-tier history, market cap / turnover R, premium index or order book depth.
   - Universe M uses red 15m bars only (no green / 5m entries).
   - Spot share is missing for coins with no spot market.
6. **2023-H1 never-read block:** it was built with the same walk and read once. Episodes then were fewer (≈ 3/day).

## Next step
1. Pre-register OBSERVE 1 (CRASH_DIP) and OBSERVE 2 (SQUEEZE_FILTER) with the frozen definitions above, as paper / observe lines, with scout pricing. **No live money.**
2. If more long frequency is wanted, the evidence points away from Frenzy entries and toward **exits / holding**: X4 shows a median +4.8 % reach that a TP-3 / SL-4 family converts badly. A path study (WORKED vs stopped) of the top model decile is the one open lead on Frenzy longs.
3. Do not re-run combination searches on Frenzy longs with the same data. This study covered the union.

## Files
- `reports/UNIFIED_LONG_2026-11-18.md` (this file), `_PREREG.md`, `_INTERIM.md`, `_tables.md`, `_extra.md`, `_candidates.csv`, `_frontier.csv`, `_triggers.csv`, `_exit_grid.csv`, `_model.csv`, `_frontier.png`, `_events.png`.
- Scripts: `scripts/unified_long_{data,build,assemble,eval,model,report,write,extra,cap,charts}.py`.
- Data: `reports/UNIFIED_LONG_data/` (rows, all_<block>.pkl, search.pkl, null_*.json, model_scores.pkl, final.pkl, parity.csv, coverage.csv).
