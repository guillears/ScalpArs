# MOMENTUM LONG recall trace: why the yr5 backtest and live disagree on momentum longs (2026-10-08)

**Question.** The yr5 replay (code 181131e, today's frozen config, 3 seeds, warm-up trimmed) prices momentum longs (ML) at
**−0.067 %/trade** (595 fills/seed/year). Live master ML looks strongly positive (kept ML +0.19 %/trade). "The replay reproduces only
~64 % of live ML fills." This report traces every live and every replay ML signal over the period live ran, says which causes are
harness problems and which are real, and gives a calibrated ML %/trade for the 7-sleeve portfolio study.

**Validation first.** `scripts/validate_against_master.py` → **ALL CHECKS PASS** (before and after this study). The study's own
check (V1 in `study_ml_trace_report.py`): every master-kept full-size ML fill B2 → Oct-3 is in the live set (0 missing), 173/173 live
rows equal the master's pct, 1,784/1,784 replay rows equal the yr5 fills.

**Scope.** Read-only on code and config. No bot API, no Binance calls at all (everything came from the existing caches), no commit.

**New files:**
- `scripts/study_ml_trace_build.py` → `reports/study_ml_trace_matchset.csv`, `reports/study_ml_trace_liveup.csv`
- `scripts/study_ml_trace_eval.py` → `reports/study_ml_trace_seedrows.csv` (live-only × seed), `reports/study_ml_trace_extras.csv`
  (replay-only), journal cache `reports/backtest_cache/study_ml_trace_journal.pkl`
- `scripts/study_ml_trace_report.py` → **`reports/study_ml_trace_signals.csv`** (one row per live ML fill, final class + per-seed
  classes) and every table below

## Plain-English answer

1. **The backtest is not broken for momentum longs.** The problems found for fades do not exist here:
   - Every yr5 ML fill all year (100 %, Jan → Oct) was decided on real trade-by-trade data. The "stale 1-minute candle" problem that
     hurt fades does not touch ML (it trades the top-50 coins, which all have tick archives).
   - When live and backtest take the same trade with the same exit, the backtest prices it the same (+0.015 %/trade difference).
     Stopped trades lose the same amount in both (−0.76 vs −0.76): no stop-fill overshoot.
   - The backtest's BTC and coin readings at its scan match live's stamps (no bias, median differences ≈ 1 RSI point = seconds apart).
2. **Cause 1 (biggest): live's "+0.19" is not today's rules on fresh data. It is live's own trades after the losers were removed.**
   - Live really traded 204 full-size ML fills Jun-18 → Oct-3 at **−0.03 %/trade**.
   - Today's filters (LOADX, megacap, heat, CALM3D checks, chop-burst, older screens) were each built by looking at those trades and
     removing the losers: **83 removed fills averaging −0.35 %**. What is left (121 fills) averages +0.19. That is in-sample by design.
   - The backtest correctly does not take the removed trades (it reproduces only 8–15 % of them).
3. **Cause 2: about a third of the live edge sits in one lucky window.** 32 of the 121 kept fills come from the "washed-out" window
   (Jun-18 → Jul-10, every sleeve won) and average **+0.49**. Without it, kept live ML is **+0.08**; without June and July, **+0.05**.
4. **Cause 3: which second the bot looks at the market.** The bot scans each coin every ~112 s. Momentum gates (BTC RSI/ADX cross,
   EMA gap, range position) flip within seconds.
   - Live's scan seconds and the backtest's are different. Per seed, the backtest takes the same signal for only 42 % of the kept
     live fills (59 % in at least one of 3 seeds). Run at live's own scan clock, it takes 69 % of them. The Oct-4 audit's "64 %" was that live-clock view over all live fills.
   - This is noise, not a bug, but it does not cancel out cleanly. The live fills the backtest misses averaged **+0.12**. The extra
     fills the backtest takes at its own seconds averaged **−0.05** (51 per seed). The difference (+0.17, 95 % range −0.02 … +0.36)
     is about what you'd expect when the live side is the in-sample survivor set (cause 2).
5. **Cause 4: today's exit rules.** On the same trades, today's hard-TP ladder caps some big runners (live +2.4 % → +1.0 %), and the
   recovery hold turns a few small winners into losers. Together that is −0.05 %/trade on matched trades. The backtest is right here:
   these are today's rules.
6. **Since August the backtest is not more pessimistic than live. It is MORE optimistic.**
   - Aug: backtest +0.19 vs live kept +0.16.
   - Sep-11 → Sep-25: backtest +0.18 vs live kept +0.05.
   - Sep-26 → Oct-3: backtest +0.09 vs live kept −0.32.
   - Live ML since Sep-11, under rules close to today's: **−0.15 as traded, −0.02 after today's filters (N 47)**. Since LOADX went live
     (Sep-29): −0.14 (N 19).
   - All the backtest's loss is in hours live never traded: Jan–Apr (−0.10), and the Aug-27 → Sep-11 offline stretch (−0.22).
7. **Best estimate of ML's real per-trade result under today's rules: about −0.06 %/trade** (range −0.12 to +0.03). That is
   below the +0.05 the portfolio study says ML needs to add value, even in the upside case.
8. **Can the year-wide loss be trusted?** Yes, as far as a backtest can be. No ML-specific harness defect was found, and the backtest
   matched live's as-traded average when it ran live's own rules (−0.04 vs −0.03, Oct-4 audit). Jan–Apr is the only stretch no
   current rule was tuned on, and it is negative with 95 % confidence. The remaining risk is the market, not the code: Jan–Apr may
   not repeat.

## 1 · Matched set (Jun-15 → Oct-4, the period live ran inside the yr5 window)

**Live side:**
- Every full-size ML fill live actually took, as traded (probes and MANUAL out), CLOSED, deduped on (opened_at, pair, direction).
  BASE comes from the COMBINED raw pool, B1 from BATCH1_FINAL, B2–B16 from the master (all rows, `stack_keep` carried).
- Cross-check: 0 extra rows in any archived batch CSV or the newest Downloads export.
- The window ends Oct-4, so B17/B18 are outside it.

**Replay side:** yr5 MOM-long non-probe fills.

**Match:** same pair, LONG, ±10 min, nearest per seed.

| Live fills that … | N · WR · avg % | yr5 reproduces: ≥1 seed · per seed · all 3 | As-was config at live's clock (asw) | Today's config at live's clock (lp) |
|---|---|---|---|---|
| today's stack KEEPS (master `stack_keep`) | 121 · 78 % · **+0.190** (CI +0.05…+0.33) | 59 % · 42 % · 24 % | 69 % | 56 % |
| today's stack REMOVES | 52 · 48 % · −0.276 | 10 % · 8 % · 6 % | 63 % | 13 % |
| never in master (BASE pre-screen / B1 anchor) | 31 · 29 % · −0.476 | 19 % · 15 % · 13 % | 45 % | 10 % |
| **all, as traded** | **204 · 63 % · −0.030** (CI −0.14…+0.08) | 40 % · 29 % · 18 % | 64 % | 38 % |

**Replay ML, Jun-15 → Oct-4:** 214/seed.
- Hours live was up: **162.7/seed · 67 % · +0.034** (day-bootstrap CI −0.05…+0.12).
- Hours live was down: 51.7/seed · 55 % · −0.215. Of these, Aug-27 → Sep-11 offline = 38.7/seed at −0.22.

**Live-up periods** are the union of the batch spans (`study_ml_trace_liveup.csv`). The real offline stretches are:
- Aug-10 11:30 → Aug-11 20:00
- Aug-25 20:44 → Aug-26 14:32
- Aug-27 16:35 → Sep-11 15:26
- a few sub-day seams after that

**Reproduction against outcome (kept fills):**

| Seeds that reproduced it | Kept fills | Live result |
|---|---|---|
| 3 | 29 | 86 % · +0.424 |
| 1–2 | 42 | +0.12 |
| 0 | 50 | 74 % · +0.114 |

The replay reproduces the strong winners in every seed. Its misses are the marginal signals.

## 2 · How each signal was traced

- **Three real-engine replays of the same days, each with its decision journal:**
  - **yr5:** today's config, synthetic live cadence 111.6 ± 5 s, a random phase per seed, the fixed harness.
  - **asw:** the as-was config (`config_history_v2`) at live's own scan clock (`livephase_scans_v2`).
  - **lp:** today's config (frozen Sep-28 live config; `lp_b1316m` = yr4 frozen config) at live's own scan clock, on the pre-Oct-4
    harness except `lp_b1316m`.
- **The journals give, per scan:**
  - the BTC readings the engine used (SCAN)
  - every failing gate of every pair (FAILS)
  - the counted gate with the engine's pair indicator context (BLOCK ctx)
  - OPEN and ADMIT lines
  - capacity blocks
- **Live-only, per seed:**
  - First, the replay's own book: pair held in any sleeve, ML on the same pair 10 min–3 h away, or 4/4 slots / capacity block.
  - Then the replay's FAILS line for the pair nearest live's decision second (opened_at − 8 s). Its gates are grouped into:
    - BTC forming-bar gates
    - pair forming-candle gates
    - today-rule gates (LOADX / megacap / heat / CALM3D / chop)
    - door, routing and state gates
  - Else "replay read the pair SHORT" / "no LONG candidate" / "not scanned".
  - Data source: whether the pair-day tick archive existed before the chunk's run start (else the 1m rebuild).
- **Replay-only, live up:**
  1. live took it as a 1× probe · live held the pair · live slots full (as-was `max_open`) · live took the pair ≤ 3 h away
  2. else the live-clock replays decide:
     - **ASW_TOOK:** the as-was rules at live's clock take it → a live-side miss.
     - **RULE_THEN:** the as-was rules block it but today's config at live's clock takes it → a genuine rule-era difference.
     - **PHASE:** both live-clock replays block it → yr5's synthetic second caught a knife-edge pass.
  3. The live decision journal (Sep-28 →) names the same gate as the live-clock replay on 10 of the 13 rows it covers. For
     example, ENA Sep-29 12:38: live BLOCK BTC_RSI_ADX_CROSS, the same gate asw names.

## 3 · Live-only: why the replay did not take each live fill

### 3a · Per signal (204 live fills, `study_ml_trace_signals.csv` → `final_class`)

| Class | Fills | WR | Live avg % | Replay avg % (same signal) | Kind |
|---|---|---|---|---|---|
| R3 · reproduced in all 3 seeds | 36 | 81 % | +0.304 | +0.200 | — |
| R12 · reproduced in 1–2 seeds (other seeds: scan phase) | 46 | 74 % | +0.092 | +0.104 | phase noise |
| **E · today's rules refuse it** (master-removed; reasons over all 52 removed: LOADX 13, CALM3D DMI/ATR/re-entry 19, megacap 10, heat 7, chop 2, blacklist 1) | **47** | 45 % | **−0.312** | — | **rule era; the replay is right** |
| **E0 · never in master** (BASE pre-screen 18, B1 anchor 13) | **25** | 28 % | **−0.495** | — | **rule era; the replay is right** |
| P · live-clock replays take it, yr5's phase never does | 26 | 69 % | +0.005 | — | phase (3 seeds too few) |
| G1 · BTC forming-bar gate at yr5's second (BTC_RSI_ADX_CROSS 6, slope 2, ADX low 1) | 9 | 89 % | +0.293 | — | knife-edge |
| G2 · pair forming-candle gate (EMA gap min 2, ADX max 2, range pos 2, EMA5 stretch 1) | 7 | 71 % | +0.358 | — | knife-edge |
| G3 · today-rule gate at yr5's second although the master keeps it (LOADX 1, BTC cross 1) | 2 | 100 % | +0.896 | — | knife-edge (stamp vs second) |
| G4 · door / routing gate (RSICEIL door 2, range pos 1) | 3 | 67 % | −0.221 | — | knife-edge |
| G5 / G6 / C · replay read it SHORT · no LONG candidate · yr5 book | 3 | 67 % | −0.23 | — | — |

**Kept fills the replay never reproduced: 50 · 74 % · +0.114.**
- 26 (+0.005) are pure scan-phase misses: the live-clock replays take them.
- 24 (+0.22) are knife-edge gate flips at the replay's second.
- None is a data problem. All 612 traced (signal × seed) rows had real ticks.

### 3b · Per seed: the 210 kept seed-rows the replay missed (70/seed, live +0.121)

| Gate family at the replay's nearest scan | Seed-rows | Live avg % | Top gates |
|---|---|---|---|
| Pair forming-candle | 64 | +0.290 | PAIR_EMA_GAP_MIN 25 (+0.38) · PAIR_ADX_MAX 14 (+0.40) · PAIR_RANGE_POSITION_MAX 14 (+0.60) · EMA5_STRETCH 6 (−0.57) · GAP_5_20 4 |
| BTC forming-bar | 59 | +0.095 | BTC_RSI_ADX_CROSS 38 (+0.30) · BTC_SLOPE_GATE 11 (−0.06) · BTC_ADX_GATE_LOW 10 (−0.50) |
| Door / routing (RSICEIL door, cross) | 29 | +0.055 | RSICEIL_DOOR_ADXMIN 11 · range pos 11 · BTC cross 7 |
| Today-rule gate at that second | 26 | +0.087 | **LOADX 18 (−0.12)** · BTC cross 6 |
| No LONG candidate / read SHORT | 26 | −0.10 | FLIP_SHORT_BTC30_RISE 7 · FLIP_SHORT_REGIME 3 |
| Capacity / pair held / other time | 6 | +0.10 | NO_BALANCE 2 (the replay's $5k equal-split book) |

**Gate inputs are not biased.** Replay − live at the replay's nearest scan:
- BTC RSI median |Δ| 1.34 (mean −0.28)
- pair RSI median |Δ| 1.24 (mean −0.35)
- pair ADX median |Δ| 0.13
- price median −0.03 %

These are the seconds between the two decisions, not a systematic error. Closed-bar inputs (ADX) are identical.

**LOADX at the second.** The master keeps 18 seed-rows that the replay's LOADX refuses. LOADX compares RSI with RSI two candles
earlier on the forming bar, so it flips within seconds. The master judges it on live's stamps.

## 4 · Replay-only: why live did not take the replay's extra fills (live up, 104.3/seed)

| Class | Per seed | WR | Replay avg % | Σ %/seed | $/seed (1×, $3k book) | Replay live-up avg if removed (now +0.034) | Kind |
|---|---|---|---|---|---|---|---|
| **PHASE**: both live-clock replays block it | **51.3** | 62 % | **−0.082** | −4.23 | −$618 | **+0.087** | scan-phase knife-edge (BTC_RSI_ADX_CROSS 30 rows −0.07, GAP_NOT_EXPANDING 15 −0.44, GAP_MIN 10, SLOPE 10 +0.43, read SHORT 8 −0.73) |
| **RULE_THEN**: live's then-rules blocked it, today's take it | **28.0** | 64 % | +0.004 | +0.10 | +$15 | +0.040 | genuine rule-era difference. 77 of 84 rows are Jun–Jul. CALM3D door (not shipped until Jul-28) 28 rows +0.25 · UNMATCHED 56 rows −0.12 |
| ASW_TOOK: as-was rules at live's clock take it, live did not | 13.0 | 64 % | +0.084 | +1.10 | +$161 | +0.029 | live-side miss (restarts, latency, live's real clock vs anchors) |
| LIVE_SLOTS_FULL: live book at max_open | 3.7 | 45 % | −0.322 | −1.18 | −$173 | +0.042 | genuine capacity (live had other sleeves open) |
| LIVE_PROBE: live took it as a 1× probe | 3.7 | 55 % | −0.007 | −0.02 | −$4 | — | rule era (probe mode) |
| LIVE_PAIR_HELD / LIVE_OTHER_TIME | 4.7 | 72 % | −0.03 | −0.13 | −$20 | — | live state / timing |

- No replay-only fill sits on a pair-day without ticks.
- The live journal (Sep-28 →) names the same gate as the live-clock replay on 10 of the 13 rows it covers.

## 5 · Same-signal pricing (153 kept seed-pairs matched)

**Live +0.285 vs replay +0.207 (Δ −0.078/trade).**

| Why the result differs | Seed-pairs | Live | Replay | Δ/trade | Σ Δ |
|---|---|---|---|---|---|
| same exit reason | 117 (76 %) | +0.217 | +0.232 | **+0.015** | +1.7 |
| exit rule differs: today's hard-TP ladder caps a runner (SXT +2.44 → +1.00, ACT, AAVE), recovery hold (RH_PREMISE_EXIT) turns a small winner into −0.7/−0.8 (DEXE, AAVE B5), era exits (TRAILING_STOP, MANUAL) | 26 | +0.641 | +0.364 | −0.278 | −7.2 |
| entry at a different scan (> 30 s apart, entry +0.1…+0.5 % higher → stopped) | 9 | +0.056 | −0.467 | −0.523 | −4.7 |
| same scan, knife-edge (DOT B12 maker vs taker-fallback, ladder vs stop) | 1 | +0.996 | −0.694 | −1.69 | −1.7 |

- **Stop-fill overshoot: none.** Both stopped STOP_LOSS 16 (live −0.763, replay −0.758); STOP_LOSS_WIDE 3 (−1.018 vs −1.033).
- **Fill types and entry:** MAKER share live 54 % / replay 54 %. Entry price median +0.000 %. Open lag median +1 s.
- **By era (matched Δ):**

  | Era | Δ replay − live |
  |---|---|
  | BASE | −0.01 |
  | B1 | −0.13 |
  | Aug | −0.05 |
  | B4–B5 | −0.83 (5) |
  | B6–B12 | −0.19 |
  | B13–B16 | **+0.20** |

  In B13–B16 the replay held three trades live stopped WIDE: ALGO, ZRO, ONDO. Same-trade differences have no consistent sign.

## 6 · The gap, quantified

### 6a · Waterfall on the hours live was up (per seed, %/trade)

| Step | N | Avg % |
|---|---|---|
| Live kept (today's stack on live's fills, in-sample) | 121 | **+0.190** |
| ① re-price the 51 shared fills at the replay's % (today's ladder / recovery hold, entry second) | 121 | +0.157 |
| ② drop the 70 kept fills this seed's scan seconds never hit (+0.121) | 51 | +0.207 |
| ③ add 7.3 fills the master removes but the replay's engine still takes (knife-edge of stamped vs live-second filters) | 58.3 | +0.169 |
| ④ add RULE_THEN 28 (+0.004): admits today's rules allow that live's rules refused | 86.3 | +0.115 |
| ⑤ add ASW_TOOK 13 (+0.084) + live-state classes 12.0 (−0.11) | 111.3 | +0.087 |
| ⑥ add PHASE 51.3 (−0.082) | 162.7 | **+0.034** |

### 6b · Per era (live-up hours)

| Era | Live as traded | Live kept (today's stack) | yr5 per seed (live-up hours) |
|---|---|---|---|
| BASE Jun-15 → Jul-10 (washed-out window) | 53 · 72 % · +0.149 | 32 · 94 % · **+0.485** | 56.0 · 70 % · +0.080 |
| B1 Jul-11 → 31 | 38 · 53 % · −0.126 | 15 · 80 % · **+0.256** | 43.7 · 52 % · −0.164 |
| B2–B3 Aug-1 → 24 | 53 · 70 % · +0.021 | 32 · 78 % · +0.161 | 25.7 · 81 % · **+0.185** |
| B4–B5 Aug-24 → 27 | 9 · 78 % · +0.062 | 4 · 100 % · +0.368 | 4.3 · 54 % · −0.294 |
| B6–B12 Sep-11 → 25 | 30 · 60 % · −0.070 | 25 · 68 % · +0.046 | 19.7 · 71 % · **+0.176** |
| B13–B16 Sep-26 → Oct-3 | 21 · 38 % · −0.421 | 13 · 46 % · −0.316 | 13.3 · 78 % · **+0.089** |
| **all** | 204 · 63 % · −0.030 | 121 · 78 % · +0.190 | 162.7 · 67 % · +0.034 |

### 6c · Ranked causes of "live +0.19 vs replay −0.067"

| # | Cause | Size | Type |
|---|---|---|---|
| 1 | **Live kept = in-sample survivors.** 83 removed live fills (−0.35) were what today's filters were fitted on (all removed fills predate their rule: Sep-28 attribution). The replay cannot "remove" its own losers in hindsight. | live −0.03 → +0.19 | genuine (methodology), not harness |
| 2 | **Washed-out window** carries 32 kept fills at +0.485. Ex-BASE kept +0.084 (89); ex BASE and B1 +0.049 (74). | ≈ 0.11 of live's +0.19 | genuine regime luck (memory `washed-out window`) |
| 3 | **Hours live never traded.** Replay Jan–Apr −0.096 (271/seed); live-off Jun–Oct −0.215 (52/seed); live-up +0.034 (163/seed). | the whole year-wide negative | genuine (out-of-sample) |
| 4 | **Scan-phase knife-edges.** 70 kept/seed missed (+0.12) and 64/seed taken at the replay's own seconds (PHASE + ASW_TOOK −0.05). Asymmetry +0.17 (CI −0.02…+0.36). | ≈ −0.05 on the live-up average (PHASE alone: +0.087 → +0.034) | noise. Mostly explained by #1 (the live side is the post-fit survivor set) |
| 5 | **Today's exit rules on the same trades** (ladder cap, recovery hold) + entry seconds | −0.08/trade on matched (exit rules −0.05, entry second −0.03) | the replay is right (today's rules); the entry-second part is noise |
| 6 | **Rule-era admits** live could not take (CALM3D door pre-Jul-28 +0.25; UNMATCHED −0.12) | 28/seed at +0.004 | genuine; the master cannot see re-admits (it can only remove) |
| 7 | Live book full (other sleeves) | 3.7/seed at −0.32 | genuine capacity |
| 8 | Harness: forming-candle data | **0** (100 % tick-decided) | — |
| 9 | Harness: replay $5k equal-split book (NO_BALANCE) | 2 seed-rows | harness, trivial |

**Harness-fidelity problems (fixable):** essentially none that move ML. Phase sampling is reduced by more seeds, not by a fix.

**Real live/replay differences:** causes #1–#7.

## 7 · Calibrated ML expectancy under TODAY's rules (paper fills, per trade)

**Evidence:**

| Source | N | %/trade | 95 % CI | Read |
|---|---|---|---|---|
| yr5 year | 595/seed | −0.067 | −0.120…−0.015 | out-of-sample except Jun–Oct |
| yr5 Jan–Apr | 271/seed | −0.096 | −0.182…−0.011 | no current rule tuned there |
| yr5 May–Oct | 324/seed | −0.043 | −0.105…+0.021 | |
| yr5 live-up hours, Jun-15 → Oct-4 | 163/seed | +0.034 | −0.05…+0.12 | partly in-sample. BASE +0.080, ex-BASE +0.009 |
| Live forward, Sep-11 → Oct-3 (rules ≈ today's) | 60 as traded / 47 kept | −0.152 as traded / −0.024 kept | | small N |
| Live forward, Sep-29 → | 19 kept | −0.142 | | small N |
| Live kept, Aug-1 → | 83 | +0.066 | | in-sample filters |
| Harness calibration (Oct-4 audit): as-was replay vs live as traded on the same hours | | −0.043 vs −0.030 | Δ −0.022 [−0.105, +0.056] | |

No adjustment is warranted: the ML-specific fidelity checks here find no correctable bias.
- Ticks: 100 %.
- Same-exit pricing: +0.015.
- Stop overshoot: 0.
- Gate inputs: unbiased.

**Recommended numbers for the 7-sleeve portfolio study:**

| Case | %/trade | Basis |
|---|---|---|
| **Base** | **−0.06** | ≈ yr5 year (−0.067). H2 −0.04 and live forward −0.02…−0.15 bracket it. |
| Stress | −0.12 | ≈ H1 / CI low; live since Sep-29 kept −0.14 |
| Upside | +0.03 | replay on live-up hours, before any in-sample haircut |
| Real money | base − ~0.02 | top-50 coins: no measurable stop overshoot on paper. Live real-money slippage is not measured for ML; use a small haircut. |

- At ≈ 595 fills/seed/year, the base is about **−36 %·trades/year per seed** (≈ −$5.2k/yr on a fixed $3k book at 1×).
- **The portfolio sensitivity says ML needs about +0.05 to add value. Even the upside does not reach it.**
- Live's +0.19 must not be used for planning. It is in-sample and BASE-heavy.

**Uncertainty.**
- Forward live under today's full stack is tiny (19 fills since Sep-29).
- The replay's Sep-11 → Oct-3 hours read +0.14, live −0.02. Over the most recent weeks the replay is the optimistic side.
- Market regime is the dominant unknown: H1 was a down/choppy tape.

**Can the H1 (Jan–May) loss be trusted?** As a statement about today's rules in that market: yes.
- **Code path:** the same engine code, today's config, real ticks for every ML decision. 100 % of H1 ML fills were decided on
  archives that existed before the run.
- **Calibration:** the harness reproduced live's as-traded average on live's hours when given live's rules.
- **No ML defect:** none of the defects found for fades (stale forming candle, 24 h-volume lag) applies to top-50 ML.
- **Not testable:** H1 cannot be checked trade-by-trade (no live then). Its universe membership and paper fills are modelled, not
  observed.
- **Even ignoring H1:** the replay on Jun–Oct is −0.043 overall.

## 8 · Proposed harness changes (diffs only, NOT applied; `scripts/engine_replay.py` untouched)

None of these changes ML's expectancy materially. They make the next trace cheaper and remove the two tiny artefacts.

### FIX-1: phase ensemble instead of one phase per seed (procedure + a flag)

Per-seed recall of kept live fills is 42 % against 59 % for any of 3 seeds. A single phase is one sample of a knife-edge process.
Run ML studies on ≥ 6 seeds, or add a sub-phase re-evaluation so each signal is scored as the average over phases:

```diff
@@ ap.add_argument("--cadence", default=None)
+ap.add_argument("--phase-shift-s", type=float, default=0.0)   # shift every synthetic scan by a constant (phase-ensemble runs:
+                                                                # 0, 18.6, 37.2, … = 111.6/6 steps with the SAME seed → pure phase effect)
@@ elif A.cadence:
-    _t, _st = WARM_MS + int(_rng.uniform(0, _mu) * 1000), []
+    _t, _st = WARM_MS + int(_rng.uniform(0, _mu) * 1000) + int(A.phase_shift_s * 1000), []
```

### FIX-2: journal the forming-candle data source on every FAILS line (diagnostic)

```diff
@@ KlineServer._ohlcv5(self, pair, limit, t_ms)
             fc = TS.candle(pair, cur_open, t_ms)
             if fc is not None:
+                self.src_last[pair] = "TICKS"; self.src_count["TICKS"] += 1
@@
         if has1:
+            self.src_last[pair] = "1M"; self.src_count["1M"] += 1
@@
+        self.src_last[pair] = "CLOSED_ONLY"; self.src_count["CLOSED_ONLY"] += 1
         lo = max(0, n_done - limit)
@@ meta dump
+    meta["ohlcv5_source_counts"] = dict(KS.src_count)
```

The FAILS ctx can then carry `KS.src_last[pair]` through the existing journal hook. Today it took a file-mtime reconstruction to
prove that ML was 100 % tick-decided.

### FIX-3: book floor for %-studies (removes the equal-split NO_BALANCE artefact)

```diff
+ap.add_argument("--balance-floor", type=float, default=0.0)   # %-per-trade studies: never let the compounding $5k chunk book refuse
@@ before each scan
+    if A.balance_floor and ENG.paper_balance < A.balance_floor:
+        ENG.paper_balance = A.balance_floor                    # $ path no longer meaningful; pnl_percentage unaffected
```

### Engine-side, not harness (for the operator)

- Scanner-path capacity refusals (PAIR_HELD / COOLDOWN / NO_BALANCE / GROSS_CAP) are journaled without the pair, live and replay
  alike. Tagging the pair would make capacity attribution exact.
- The master's stack screen judges LOADX on live's stamps. The engine judges it on the forming bar at the scan second. 18 seed-rows
  of master-kept fills are refused by the replay's own LOADX, and 7.3/seed fills the master removes (or never had) are still taken
  by the replay engine. The master's "kept" set is therefore not exactly what the engine would do: a ±knife-edge on 10–15 % of kept fills.

## 9 · What this study could NOT test (blind spots)

- **Live's scan instants before Sep-28.** The live journal starts Sep-28. Earlier, live's phase comes from order anchors (asw/lp at
  `livephase_scans_v2`), not observed scans.
- **The lp runs before Sep-26 use the Sep-28 frozen config on the pre-Oct-4 harness.** It lacks LOADX, chop-burst and FRENZY, and
  has a stale 1h forming bar. RULE_THEN vs PHASE can therefore be mislabelled when the difference is one of those.
- **The live decision second** is modelled as opened_at − 8 s. Maker fills open later (up to ~20 s), so the "nearest scan" can be
  one scan off.
- **Removed/E0 fills' replay gate** is shown, but whether today's engine would refuse them at live's exact second is only proven for
  B13–B16 (`lp_b1316m`: 0/5 master-removed reproduced).
- **H1 has no live counterpart.** Its credibility rests on harness calibration in Jun–Oct, not on H1 trades.
- **Real-money fills and funding are not modelled.** B4 (Aug-24 → 26, real money) has 5 live ML fills only.
