# WILLY HT exit study — PRE-REGISTRATION (written 2026-11-12 BEFORE any variant result was read)

Operator ask 11-12 (after INIT HT: TP +1.25 at 38 s, then +6.64 % post-exit peak; vs ONE HT: post-exit +2.5 then −4.2):
does a different FRENZY_WILLY_HT exit capture more of the runners without giving back more on the faders?

## Grid (10 rows = BASE + 9 variants; frozen)
All keep the −4 stop (until a variant moves it), the 90-min cap, and — before the arming line is reached — today's late TP (+0.25 from minute 30).
T = today's V2b target per fill (max(+1.25, min(8, net at spike high × 0.99)); stamped T where the fill ran one, else rebuilt — build_willy_exit_cf).
| id | exit |
|---|---|
| BASE | today's V2b: resting TP at T, −4 stop, late +0.25 from 30 min, 90-min cap |
| TP2 | TP at max(T, +2) (reference) |
| TP3 | TP at max(T, +3) (reference) |
| RUN05 / RUN10 / RUN15 | at the first touch of T: no exit — trail = (highest net since then) − k × entry ATR %, k = 0.5 / 1.0 / 1.5; cap closes at 90 |
| RUN10_BE | RUN10 with the trail never below +0.10 (BE lock) |
| HALF_RUN10 | half closes at T (resting TP), the other half = RUN10 |
| HALF_TP3_BE | half closes at T, the other half: TP max(T, +3), stop moved to +0.10 (BE) |
| TP3_BE | whole position: TP max(T, +3); after touching +1.25 the stop moves to +0.10 |

Trail / BE exits are market exits: booked at the line on a bar event (or the bar open if it opened through it), at the print on a tick
event, MINUS 0.05 % slip (sensitivity 0.00 / 0.10 shown). The trail peak uses EARLIER events only (an event that sets a new high and
touches the trail is judged against the old peak — conservative). Resting TPs book the line (as the CF builder).

## Cohorts
MASTER (reports/MASTER_POOL_stacked.csv stack_keep, non-probe): FRENZY_WILLY_HT (subject, N = 8) and FRENZY_WILLY (own row, N = 17) — never pooled.
B26 forward (exports, not in master): HT and WILLY fills to date. BACKTEST: the 2-year today's-stack replay carries NO WILLY_HT fills
(all entry_vol_mcap_ratio < 1) → its FRENZY_WILLY fills are shown as a WILLY cross-check only (labelled), never as HT evidence.

## Decision bar (frozen)
A variant is a SHIP candidate for HT only if, on MASTER HT: Σ Δ vs BASE > 0 with the day-block bootstrap 95 % CI lower bound > 0, it beats
the best-of-grid luck test (p ≤ 0.10), leave-one-out keeps the sign, no single fill ≥ 50 % of Δ, AND the WILLY master row and the backtest
WILLY cross-check do not show the opposite sign. With N = 8 the CI bar is not expected to pass → the realistic outcome is an OBSERVE line.
