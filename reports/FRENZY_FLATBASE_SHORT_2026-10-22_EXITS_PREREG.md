# FRENZY flat-base short — EXIT follow-up — PRE-REGISTRATION ADDENDUM (frozen 2026-10-22, before any result of this follow-up)

Operator (via coordinator): the "TP back at the pre-pump price" exit and the multi-day holds are "too ambitious, no need to hold that much
to make a massive win". This is POST-HOC exploration on the same data as the main study (`FRENZY_FLATBASE_SHORT_2026-10-22.md`, REJECT),
so it is protected by an out-of-sample split decided here, before any number is computed.

## Cohort / entries (unchanged, frozen)
FLAT cohort only (FLAT14 ≤ 44.39 %, the frozen label). Entries S2 (exhaustion) / S3 (24 h) / S4 (4 h close below VWAP) — the same signal
bars as the main study (data/signals.pkl). One short per pair at a time per cell (the next signal on a pair is skipped while the previous
short of that cell is open). Position % at 1× size.

## Exit grid (6 × 4 × 3 = 72 exits per entry → 216 cells; 504 after the fixed-delay entries were added, see the second revision)
- take-profit: fixed −5 / −8 / −10 / −15 / −20 % from entry (limit at the level; filled at min(level, bar open) on a gap), OR a trailing
  variant (arms once a low reaches −8 %; from the next bar the stop is the best low + 4 %, never looser than the fixed stop)
- max hold: 12 / 24 / 48 / 72 h (exit at the close)
- stop: 10 / 15 / 25 % above entry, filled at max(level, bar open) + 0.05 % (gap-aware); stop first when stop and TP share a bar;
  loss capped at −100 % (1× liquidation; never reached with these stops except on a > 99 % gap)
- costs: 0.20 % round trip + REAL funding over the actual hold (settlements in (entry, exit]); missing history −0.10 %/8 h (counted)

## Out-of-sample protocol (frozen)
1. SELECTION on 2025 entries ONLY (flat, year block): rank the 216 cells by the 2025 day-clustered bootstrap 95 % lower bound of the mean
   net %; eligible only with 2025 N ≥ 300. FROZEN PICKS = the top 3. Nothing from 2026 is looked at before the picks are written into
   the results file.
2. OOS = 2026-01-01 → 2026-10-01 (H1 and H2 reported separately) + block ② master (10-02 → 10-13) + block ① live (10-14 → now; open
   trades marked to market), ②① on REAL episodes (bot FRENZY evidence).
3. A pick PASSES only if ALL of the original ship bars hold, read on the OOS data (2026-H1 + H2 together unless stated):
   - full-year N ≥ 300 on ≥ 30 days;
   - OOS day-bootstrap 95 % lower bound > 0 (after costs + funding);
   - beats the same-statistic random-bar null on the OOS data at p < 0.05 / 3 (Bonferroni over the 3 picks);
     the null = a random bar in [spike + 2 h, spike + 96 h] of the same flat episodes, same universe gates, same exit, same costs,
     same real funding, AND the same one-short-per-pair rule; 200 shuffles;
   - the 2026-H1 and 2026-H2 means each > 0;
   - ≥ 8 live/master windows, mean of window means > 0 and ≥ half the windows positive;
   - OOS mean × 0.6 ≥ +0.50 %/trade;
   - OOS mean with the best 5 % removed > 0; no pair and no day > 25 % of the OOS profit.
   (The flat-vs-non-flat contrast was already established in the main study and is not re-tested here.)
4. Below the bars → no ship. A pick that passes everything except the live/master window count may at most become an OBSERVE-ONLY line
   with a frozen gate.

## Reported for every one of the 216 cells (all periods, clearly marked in-sample 2025 vs OOS)
N · WR · avg % [day-clustered 95 % CI] · 2025 / 26-H1 / 26-H2 · best 5 % removed · random-bar null mean and p · worst trade · max drawdown
at $3,200 (size: 1st-percentile loss = 5 % of the account, ≤ 5 concurrent shorts, one per pair, daily mark-to-market).

## Note on the main study's random-bar null (for the hostile reviewer)
The main study's null (study_flatbase_short_sim.py) DID include the 0.20 % costs and real funding for each random draw's actual hold, and
the same universe gates; it did NOT apply the one-short-per-pair rule (declared as a blind spot in the main report). This follow-up's null
applies it. The main study's numbers are not edited.

## REVISION (operator, via coordinator) — written BEFORE any result of this follow-up was computed; REPLACES "Out-of-sample protocol" step 1–2
Operator: 2025 data may not be accurate enough to be the selection set. New protocol:
1. **Walk-forward by half-year.** Folds F1 = 2025-H1, F2 = 2025-H2, F3 = 2026-H1, F4 = 2026-H2 (to 10-01), by entry time. For each fold, the
   cell (entry × exit, 216) is SELECTED on the other three folds pooled: highest day-bootstrap 95 % lower bound of the mean net %, eligible
   only with N ≥ 300 on those folds. The selected cell is then scored on the held-out fold. Each fold's pick and held-out result reported.
2. **Candidate** = a cell selected in ≥ 3 of 4 folds AND with a held-out mean > 0 on BOTH 2026-H1 and 2026-H2 (after costs + funding).
3. **Real blocks first.** A candidate must not be negative on ② master (mean of the day-window means ≥ 0, real episodes) and is reported on
   ① live (open = MTM). Then the original ship bars of step 3 above apply with "OOS" = 2026-H1 + H2 (lower bound > 0, random-bar null
   p < 0.05, haircut, best-5 %, concentration, ≥ 8 ①② windows).
4. **2025 reported separately and flagged**, with every measurable reason it may be less reliable (universe reconstruction, delisted
   pairs, funding-history gaps, data gaps, regime).
5. No candidate → no ship. If a candidate passes everything except the ①② window count → OBSERVE-ONLY with a frozen gate at most.

## SECOND REVISION (coordinator, after a hostile review of the main study). Honesty note: the design came from the coordinator before any result; this text was written down AFTER the run that uses it. The first run (old null, S2-S4 only) had been printed but was superseded before analysis
- The random-bar null is NON-ANTICIPATING: draw a delay uniformly in [spike + 2 h, spike + 96 h] first, then enter at the first bar at / after
  it whose gates (24 h volume ≥ $20M on trailing data, pair already seen on the shortlist) hold; no eligible bar → no trade for that draw.
  (The main study's null drew uniformly among bars that are eligible over the whole 96 h — eligibility depends on future volume.)
- Added entries (operator: "the coin choice is the edge, timing is secondary"): D2 / D6 / D12 / D24 = the first eligible bar (gates on data
  up to that bar) at or after spike + D h. They run through the same 72 exits and the same walk-forward; the family is now 7 × 72 = 504 cells.
