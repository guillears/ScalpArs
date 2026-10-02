# 🪜 RUN SLEEVE — working plan and everything learned so far (started 2026-10-02; resume from here)

Operator goal: automate what he trades by hand on pairs in a volume frenzy (MOVR Sep-29→Oct-2, SAND Oct-2, also GTC / NIGHT). He wants
20×+, a tight stop (8 % "is crazy"), a trailing take-profit, FEW trades ("2–3 big trades are game changers"), longs and shorts. Nothing is
built or armed. No arm recommendation before the reviews below are read. Size by what one stop costs the account, never by leverage.

## Definitions (frozen; code in scripts/)
- LEADER bar / onset: 5m close with 30-min return ≥ +5 % ∧ last-hour quote volume ≥ 20× the pair's normal hour (median hourly volume over the
  30 days ending 1 day earlier) ∧ ≥ $2M, none in the prior 24 h. Anchored VWAP = Σ(typical × volume)/Σ volume from that bar.
- STAIRCASE state (LONG entry): ≥ 2 h (24 bars) after the onset ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥ 100×
  normal ∧ 24 h volume ≥ $100M. Episode ends 24 h after the last state bar. (scripts/staircase_swing_test.py, scout: scripts/scout_staircase.py)
- BREAK SHORT trigger: 5m close below the EMA50 / EMA200 (of 5m closes) with the previous 12 closes above it, 4–96 h after the onset, run
  ≥ +50 %, 24 h volume ≥ $50M. (scripts/break_short_review.py · triggers())
- CONFIRMED EMA200 break (frozen 2026-10-02): EMA50 is 0…2 % above the EMA200 at the break ∧ the run's peak was ≥ 4 h earlier.
  (scripts/ema200_confirmed_short_test.py) ⚠ MOVR's winning break had the EMA50 2.1 % above → misses by 0.1; do NOT move the limit to fit it.

## What the YEAR says (Jan–Sep 2026, all futures pairs; the year cache ends 09-28, so MOVR / SAND are out of sample)
- Scalping inside a frenzy fails in every form: dip / any minute × 5 exits (FRENZY_DIP_TEST), the operator's own +1.0 / −3.0…3.6 (hit 73–77 % vs
  78–81 % needed), staircase scalps (STAIRCASE_TEST), 35 fixed target/stop cells from staircase entries (+0.5 / −8 hits 94 %, needs 95 %).
- STAIRCASE SWING (ride it: exit on a 5m close below the VWAP, 8 % stop): 2,172 trades · 19 % winners · avg win +24.8 / loss −5.4 ·
  +0.24 / +0.13 %/trade (halves), after funding +0.87 / +0.37 (longs RECEIVE funding in frenzies, +0.40 %/trade), by-day CI [−0.20, +1.77].
  ≥ $100M pairs: +1.00 / +1.00. < $20M: −2.2 / −1.7. Leave-one-month-out positive every time. Best 5 % removed −3.64; top-10 trades carry it;
  longest losing run 42; worst drawdown −705 pts at 1×. Looser exits (2 % / 5 % below the VWAP) are worse. Entries ≥ 32 h after the spike carry
  the whole result (+2.7 / +2.75, unreviewed post-hoc cut); entries in the first 32 h lose. (STAIRCASE_SWING_TEST / _REVIEW)
- SHORTS: frenzy "switch-off" swing short failed review (carried by a few trades). Day-after short of yesterday's gainers: 0 of 9 cells.
  EMA200 / EMA50 break with exit on a close back above the line: fails (median trade 12 min). EMA200 first break HELD a day: runs +100–200 %
  → 67 % lower after 24 h, +3.4 / +1.1 after funding with a 15 % stop (N = 81, CI spans 0); smaller runs no edge; > +200 % dangerous.
  EMA50 break (703 of 4,010 fetched, Jan–Mar): first hour falls ≥ 0.7 % in 82 % but RISES ≥ 0.7 % in 81 % → volatility, not direction.
- CONFIRMATION screen (25 cuts of 2,470 EMA200 breaks; a candidate, not a result): EMA50 0–2 % above the EMA200 → short +0.10 / +0.58 at 4 h,
  +1.62 / +1.13 at 24 h, only 29 % go ≥ 3 % against in hour 1; EMA50 > 5 % above → −0.72 / −1.56, 71 % go ≥ 3 % against. Monotonic.
- Sector peers (Binance underlyingSubType tags): buying a spiking pair's sector peers fails; peers moving ≤ 30 min after the leader +0.07 (lead).
- Entry stretch (longs, ≥ $100M): share dipping ≥ 3 % in the first 30 min rises 36 % → 62 % as the entry goes from 0–2 % to > 20 % above the VWAP;
  71 % after a ≥ 10 % 30-min push. Matters for a tight stop, not for the ride-it result.

## MOVR and SAND cases (in-sample; scripts/run_case_study.py → reports/RUN_CASE_STUDY_MOVR_SAND_2026-10-02.md)
- 1-minute candle ≈ 1 % on these pairs → a 0.5 % stop or give-back is inside the noise.
- Stop: 0.5 % stops 15 of 15 MOVR trades; 3 % best (+14.1 % on 15 trades), 2 % +9.2, 1 % +7.3 (3 winners carry it).
- Trail: start at +3 %, give back ~1 % (1 % > 1.5 % > 2 % at every start; +5 % start takes 10 stops).
- MOVR sequence (stop 3, trail 3/1): L −3.1 · S50 +7.3 · L +3.8 · S50 +2.1 · L +3.0 · S50 −3.1 · S50 +3.9 · S50 +4.6 · S50 −3.1 · S200 −3.1 ·
  S50 −3.1 · S50 −3.1 · S200 −3.1 · S50 +2.5 · S200 +8.7. Five stops in a row (all SHORTS, Oct-1 afternoon, pair still climbing).
- First MOVR long lost because it bought 12.8 % above the VWAP right after a +10.6 % 30-min push (5.2 % candle); waiting 15–20 min fixed it,
  5 min did not; on SAND waiting hurt. SAND long: +3.3 % in 8 min from 4.6 % above the line after a pullback.
- MOVR EMA50 breaks #1–#6 were pauses (pair 30–50 % higher 12 h later); only #7–#9 (45–52 h in) were the top.
- SIZING (operator's key point): the same 15 trades that sum +14.1 % on price end −85 % all-in at 20×, +50 % (57 % drawdown) at 25 % of the account,
  ~+4.5 % at 1 % of the account risked per stop. Rule: size each trade so ONE STOP costs ~1 % (max 2 %) of the account.
- Trade limits on MOVR (after the fact): cap 2 per side per day → +22.1 %, losing run 1; pause after 2 stops → +23.5 %. NOT confirmed on the
  year — the operator: "maybe 3 or 4, and maybe after the stops there is a win" → scripts/sleeve_limits_analysis.py answers it.

## Running / to read (all write to reports/, minute data cached in reports/backtest_cache/k1m_break)
1. scripts/break_short_review.py → BREAK_SHORT_REVIEW_2026-10-02.md (EMA50 + EMA200, 3 run buckets, stops 1.5/2/3, trails; re-run after the
   fetch to use the widened TRAILS grid already in the file).
2. scripts/ema200_confirmed_short_test.py → EMA200_CONFIRMED_SHORT_TEST_2026-10-02.md (stops 1/1.5/2/3 × 5 trails × 3 limits + control).
3. scripts/staircase_long_tight_test.py → STAIRCASE_LONG_TIGHT_TEST_2026-10-02.md (same grid; ALL vs NEAR the line).
4. scripts/sleeve_limits_analysis.py → SLEEVE_LIMITS_ANALYSIS_2026-10-02.md (attempt number, result after k stops, cap grid 6 × 5 per leg).
Then: final sleeve specification for the operator's approval (legs in / out, stop, trail, limits, 1 %-risk sizing, kill bar judged at ~40 trades
— a first-10 bar would kill a 20 %-win-rate sleeve by luck 1 time in 8).

## Live helper already shipped
Scout "🪜 Staircase watch" (commit 5e53db1): alert-only list of pairs in the state, ★ at 100× volume, ⏳ when still ON ≥ 32 h.

## YEAR RESULTS — all four tests finished 2026-10-02 (% of position at 1×, after costs; NONE reviewed yet → no arm recommendation)
- **EMA50 break short, run > +200 %** (499 triggers, 62 pairs) is the only leg that passes: 6 of 12 cells. Stop 3 / trail 5-3 +0.52 / +0.40,
  by day +0.80 [+0.24, +1.37], 43 % win; stop 3 / trail 3-2 +0.48 / +0.33; stop 2 / trail 5-3 +0.40 / +0.31 [+0.07, +1.02], 33 % win, 67 % stopped.
  Without its best 5 % of trades every cell is slightly negative (−0.10 to −0.26) → carried by big falls. Runs of +50–200 %: nothing (1 of 24).
- **EMA200 break short**: all runs → only > +200 % stop 3 / trail 1-1 passes (+0.71 / +0.08); May–Sep negative in most cells.
  Confirmed (EMA50 0–2 % above EMA200, peak ≥ 4 h earlier; 753): 0 of 20 pass; stop 3 trail 5/2 +0.23 / +0.18, by day +0.28 [−0.09, +0.66].
- **Staircase long, tight stop** (1,220 entries ≥ $100M): 1 of 80 cells passes — stop 3 / trail 5-1.5 +0.23 / +0.34, by day +0.30 [+0.02, +0.58].
  Stops of 1–2 % are zero or negative; "near the line" entries are WORSE than all entries; max 2 per pair per day made it worse.
- **Attempts (scripts/sleeve_limits_analysis.py, exit fixed stop 3 / trail 3-1)** — the operator was right, the number is not 2:
  LONG: 1st attempt of the day −0.11 (764), 2nd +0.14 (308), 3rd +0.92 (100, both halves); after 1 stop +0.36 (268, both halves), after 2 +0.45 (63).
  Cap 1/day = worst cell (−84 pts); no cap + no pause = best (+69). EMA50 short: 1st attempt is the best (+0.29, both halves), later ones ≈ +0.1;
  after 2–4 stops still positive (+0.21 / +0.19 / +0.51). Confirmed EMA200: 1st +0.19, after stops +0.23 / +0.35.
  → A pause after stops removes winners in every leg. Limits belong in SIZING (1 % of the account per stop), not in an attempt cap.
  CAUTION: the attempt cut was read after the fact on one exit setting, attempts on a pair are not independent → hypothesis, not a rule.
- 2 % stop reality (operator's ask): only EMA50 > +200 % survives it (trail 3-2 or 5-3). Everything else needs 3 %.
- Still owed before any build: hostile review of the EMA50 > +200 % cell (same-statistic null, leave-one-month-out, per-pair concentration,
  overlapping triggers on one pair counted as one event, 0.02 % slippage on stops), break_short_review re-run with the widened TRAILS grid.

## HOSTILE REVIEW of EMA50 short, run > +200 % (scripts/break_short_hostile_review.py → BREAK_SHORT_HOSTILE_REVIEW_2026-10-02.md) — DOES NOT PASS
- Honest ruler (gap-aware stop + 0.10 % slippage): stop 2 trail 5/3 +0.35 → +0.18 (+0.22 / +0.16); stop 3 trail 5/3 +0.27. At 0.30 % slippage ≈ +0.05.
- Every 95 % interval spans zero (by day [−0.12, +0.54], by episode, by pair). 27 of 62 pairs positive; top 3 pairs = 86 % of the total; losing run 14.
- Dose is NOT monotonic: +50–200 % negative, +200–300 % positive (+0.37), +300 %+ negative → the "> +200 %" pass is one band = confound flag.
- FOR it: beats random bars of the same pairs/condition in 100 % of 500 draws (random −0.32, below-EMA50 bars −0.44); the mirrored long loses (−0.37).
  So the break does carry direction information — but the edge left after honest costs is too small and too concentrated to arm.
- After-the-fact cuts (hypotheses only): entry 15–40 % below the peak +0.55/+0.63 both halves; within 15 % of the peak −0.59; volume ≥ $500M −0.29.
- MOVR/SAND 3-day replay at $500 × 20× ($10k position): MOVR long 3 trades 1 won +$444 · EMA50 short 9 trades 2 won −$461 · EMA200 short 3 trades 1 won +$159;
  SAND long 1/1 +$515 · EMA50 short 1/1 +$333 (still open) · EMA200 none. MOVR's run peaked at +194 % and SAND's at +52 % → the "> +200 %" rule takes NEITHER.
- Working name proposed: FRENZY (FRENZY_LONG / FRENZY_SHORT).

## OFF-PEAK SHORT (entry 15–40 % below the run's peak) — frozen, tested on unseen triggers, FAILS (scripts/break_short_offpeak_test.py)
- Primary (EMA50, run +50–200 %, 1,520 trades): −0.16 per trade (+0.08 / −0.35), by day [−0.34, +0.02]; all 4 exits negative.
- EMA200 all runs (1,483): −0.14; EMA200 run > +200 % (156): 0.00 (+0.99 / −0.44).
- In-sample reference (EMA50 > +200 %, 201): +0.58 (+0.52 / +0.62), by day [−0.00, +1.17], top 3 pairs 53 % — does not transfer.
- The one thing that DOES repeat in every set: entries within 15 % of the peak are the worst band (−0.38 / −0.57 / −0.59 / −0.61) → "do not short near the top"
  is a robust AVOID rule, but nothing tested makes the remaining shorts positive.
- Status of the short leg: no codifiable edge found on the year in 4 attempts (EMA50, EMA200, confirmed EMA200, off-peak band).

## MARKET STATE at the EMA50-break short (operator: "now with SAND everything is going down") — scripts/break_short_market_state.py → BREAK_SHORT_MARKET_STATE_2026-10-02.md
- Case (MOVR 9 + SAND 1): the 3 winners all had BTC flat/down over 1 h and 57–66 % of pairs falling; 4 of the 7 stops came with only 14–36 % of pairs falling
  (market rising). But 3 stops also came at 51–68 % falling → fits the case loosely.
- Year (3,949 trades, strict ruler): NO market split turns the short positive. BTC 1 h / 4 h / 24 h sign, BTC vs EMA50 / EMA200, share of pairs falling
  (1 h / 4 h), the pair's own 1 h / 4 h: every state between −0.39 and +0.02 per trade on all runs; "≥ 75 % of pairs falling" is the WORST state (−0.39).
  On run > +200 % the better states are BTC ABOVE its averages (+0.31 / +0.36), the opposite of the hypothesis. Win rate is 28–32 % in every state.
- Read: when everything falls the pumped pair has usually already fallen (the break is late) — market direction at the break does not separate.

## FRENZY design status (2026-10-02 evening) — see reports/FRENZY_SLEEVE_DESIGN.md (sections 8–9) for the review + follow-up
Long as first designed fell to +0.05 on the strict ruler (wiped out at 1×/2×). Best version: ATR ≤ 2 % gate (+0.25 seen / +0.10 unseen, ranges span zero).
Ride-it exit and ATR-scaled stops fail. Shorts observe-only. Operator: long ARMED, no automatic-off rule, wants 2× inv × 1× lev; one stop at 2× = 32 % of the account.
