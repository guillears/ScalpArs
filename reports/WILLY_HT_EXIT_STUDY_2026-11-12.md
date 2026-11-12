# WILLY HT exit study — 2026-11-12

Pre-registration (grid + decision bar frozen before any result): `reports/WILLY_HT_EXIT_STUDY_2026-11-12_PREREG.md`.
Full tables: `reports/WILLY_HT_EXIT_STUDY_2026-11-12_results_slip0.05.md` (main) · `_slip0.00.md` / `_slip0.10.md` (sensitivity on the new
market exits). Rows: `reports/WILLY_HT_EXIT_STUDY_2026-11-12_rows.csv`. Script: `scripts/study_willy_ht_exit_1112.py`. validate_against_master: ALL CHECKS PASS.

## Summary (plain English)
- **No exit change passes the pre-registered bar.** Every variant fails the luck test (best-of-9, p ≥ 0.28 across the three slip rulers), and the two cohorts that
  could confirm (HT master vs the WILLY backtest) point in OPPOSITE directions.
- **Best HT candidate — TP2 (take profit at the larger of today's target and +2 %):** HT master kept 8 fills +20.4 → +23.4 pts
  (+3.0, day-boot CI [+0.75, +5.25]; gained on 4, gave back on 0) · %/day at one size **+3.16 → +3.65**; HT all archived 42 fills Δ +7.9
  (CI [+3.0, +13.6]) +0.54 → +1.25 %/day; B26 HT (3) +3.67 → +5.86 %/day (INIT +0.75). **But** the WILLY backtest says the opposite
  (Δ −15.7 pts on 609 fills; B19's TP2 grid read the same), B26 WILLY H loses 5.25 with it (hit +1.25, fell to the stop), and luck p 0.64.
  At one size (HT X 1.96 × $10k) +3.0 pts ≈ +$590 over 12 days on master kept, before the 30–50 % in-sample haircut.
- **Runner trails (RUN05 / RUN10 / RUN10_BE):** lose on HT master (kept −3 to −5 pts, all −7 to −13; RUN10_BE CI wholly below 0) — the
  HT faders give it back; they help INIT-type trades (B26 INIT +3.3 to +4.2) and the WILLY backtest (+45 to +50, but one fill = 70 %).
- **RUN15 (wide 1.5×ATR trail):** HT kept +3.5 / all +12.0 / backtest +92.9 — but on HT one fill carries 76–112 % of the gain (CI spans 0).
- **N is the problem:** 8 kept HT fills on 5 days (42 incl. the ones today's filters block). WILLY rows would add power but pooling is not
  allowed for a decision (each Frenzy sleeve apart).
- **Recommendation: NO CHANGE; register two OBSERVE lines on HT forward fills** — TP2 and RUN15 shadows priced from the stamped path —
  frozen gate: after 15 HT fills, Σ Δ vs today's exit > 0 with the day-boot lower bound > 0 AND the luck test p ≤ 0.10 → operator call.

## Method
- Cohorts (each sleeve apart): MASTER = every archived FRENZY_WILLY_HT / FRENZY_WILLY fill (`build_willy_exit_cf.archived_fills`), split KEPT
  (stack_keep, non-probe — the decision cohort) vs ALL (incl. fills today's filters block — context); B26 forward (exports, 3 HT + 1 WILLY);
  BACKTEST = the 2-year today's-stack replay — **no HT fills** (every replay fill has turnover R < 1) → FRENZY_WILLY cross-check only.
- Path per fill: ticks then 1m (`study_b19_fzexit_common.load_path`, net % = gross − 0.045 − 0.045 r); B26 1m bars fetched once (public, 4
  requests, weight ≤ 8). T per fill = stamped V2b target, else rebuilt from the spike high. BASE = the CF builder's V2b walker.
- Parity: my walker = the CF walker on 121/121 master, 4/4 B26, 609/609 BT; BASE vs master stack_pct within 0.10 on 25/25 kept fills; vs
  booked % 4/4 B26. **Backtest caveat:** only 609 of 1,519 replay fills have a cached path, and BASE matches the replay's booked % on only
  283/609 (the replay's own exit / fill model differs) — the backtest is a weak cross-check here.
- New market exits (trail / BE) pay 0.05 % slip; sensitivity 0.00 / 0.10 does not change any conclusion (TP variants carry no slip; RUN15 HT
  all +12.9 / +11.2, backtest RUN15 +108 / +77).
- Path labels (BASE), HT master all 42: LEFT_ON_TABLE 14 (+41.8 %) · NEVER_WORKED (stop) 13 (−52.3 %) · LATE (+0.25 line) 11 (+15.3 %) ·
  CAPTURED 3 · DRIFT 1. The larger REALISED loss side is the stops (entry), not the exit.

## Earlier work this re-tests
- B19 WILLY TP grid (`reports/B19_WILLY_TP15.txt`, backtest): TP 2 worse than TP 1.25 — today's backtest cross-check agrees (TP2 −15.7).
- WILLY exit V2 (`reports/WILLY_EXITV2_2026-10-23.md`): spike-high target shipped (the V2b T used here as BASE); its arm/floor was removed (DL 358).
- B21 TP +0.3 / 30-min variants (`reports/B21_WILLY_TP03_CAP30_MASTER.txt`): not re-tested (the opposite direction).
- Cohort changed since: WILLY HT did not exist in those studies (DL 341), and the master now carries the B25 HT fills, whose post-exit peaks
  are large (median ≈ +6.9 % within 45 min on the 8 kept HT fills) — the reason the master and the backtest disagree.
