# Bear-Run false-start EXIT study — PRE-REGISTRATION (frozen 2026-10-23, before any result was computed)

Question (operator 2026-10-23): Bear-Run V2 with exit (d) "hold while V2 ON, +2 % stop, 24 h cap" earns on clean slides
(V2-C1 +1.13 %/short, Σ +93; V2-C2 Σ +118, seed-1 replay) and loses about as much on false starts (−158 / −100). Nothing at
ON separates the two (BEARRUN_V2_SEPARATORS_2026-10-22). Can a POST-ON exit rule cut false starts fast and let clean slides
run, so V2 (or today's T0 trigger) beats today's live sleeve?

Research only. No tracked code/config change. Script: `scripts/study_bearrun_fsexit.py`.

## Data (frozen)
- ③ replay: V2 forced replay seed 1 (`reports/backtest_cache/replay/v2f_*_s1_*`, 93 blocks, 2025-01-01 → 2026-10-22),
  cohort/selection exactly as `scripts/study_bearrun_v2_price.py` (`cohort()` E0 entry arm, `select_all()` 2 h pair
  spacing + 4 slots net of the yr6 book). Pricing on 1 m klines (caches + `reports/backtest_cache/v2ext`), fee 0.10 % round
  trip, no extra slippage (same as exit d). Seed 2 blocks finished by the time of the run = later consistency check only.
- Triggers: V2-C1 (slope3 < 0 ∧ −DI > +DI ∧ r6 ≤ −1 %, OFF bounce 1.5 / +DI 2 bars / slope ≥ 0), V2-C2 (breadth ≥ 50 instead
  of r6), and T0 (today's trigger) — windows from `reports/backtest_cache/study_bearrun_early/on_<var>.csv`.
- Window classes (V2 only): `reports/BEARRUN_V2_windows_mech.csv` (clean_slide / true_other / false_start), mapped per fill as
  in the price script.
- ② master: the 2 Bear-Run master fills (10-07 UNI, XLM). ① live: 10-07 UNI/XLM and 10-22 SOL 17:16 @103.45, APT 18:04
  @0.755619, HBAR 18:10 @0.07892 (−$20.67, −$19.07, −$29.01 at 5×); prices from klines.

## Base exit (d) and rule layer
Every short: exit at the first of +2 % adverse (1 m high), the trigger's OFF, 24 h after entry (close of that minute).
A WINDOW rule fires at a kill time K (first minute close at which it is true, measured from the window's ON, before OFF):
all of the window's open sleeve shorts close at that minute's close, and fills with t ≥ K in that window are BLOCKED (no new
sleeve shorts until the next ON). BTC rules use BTC 1 m closes (ON price p0 = BTC close at the ON minute):
- **R1(B, H)**: within the first H h after ON, BTC close ≥ (lowest BTC close since ON, incl. p0) × (1 + B %).
  B ∈ {0.5, 0.75, 1.0}, H ∈ {1, 2, 4} (9 rules).
- **R2(X, H)**: at ON + H h, BTC's lowest 1 m low since ON is still above p0 × (1 − X %) ("no follow-through"); K = ON + H.
  X ∈ {0.5, 1.0}, H ∈ {1, 2} (4 rules).
- **R3**: the first 1 h bar closing after ON (first top-of-hour > ON) closes above p0; K = that hour's close (1 rule).
- **R4(L)**: average mark-to-market P&L of the window's open SELECTED sleeve shorts ≤ −L % while no short of the window has
  yet reached +1 % (1 m low based; once one has, R4 is disarmed for the window). L ∈ {0.5, 1.0} (2 rules). Selection-
  dependent → solved by fixed-point iteration (price → select → kill → reprice), max 4 passes, convergence reported.
- **T (per short)**: once a short's peak gain (1 m low) ≥ +2 %, trail 1 %: exit at peak − 1 % when a later minute's high
  reaches it (stop checked first only while unarmed). Tested alone and combined with every R.
Family per trigger = {d, T, 16 R, 16 R+T} = 34 exits × 3 triggers (V2-C1, V2-C2, T0) = 102 candidates.

## Measures (all at 1×, Σ % per seed, window units where stated)
N · WR · avg %/short · Σ % · window mean (mean of per-window avg) · per half-year (2025-H1, H2, 2026-H1, H2) ·
clean_slide / true_other / false_start split (Σ given up on clean slides vs Σ removed on false starts) · ΔΣ vs exit (d) of
the same trigger and vs T0 (T0 trigger + today's exit a = live sleeve, Σ −7.03 seed 1) with a WINDOW-clustered bootstrap
(10,000, union windows T0 ∪ V2-C1 ∪ V2-C2) · two-sided exit CF vs d: fills whose exit changed split saved / cost, fills the
rule did not touch must show Δ = 0, blocked fills and slot-freed new fills listed separately.

## Walk-forward (decides the candidate — never best-in-sample)
Leave-one-half-out over the 4 halves: on the 3 training halves pick the candidate (of 102) with the largest training ΔΣ vs
T0; record its held-out ΔΣ vs T0. The "chosen rule" = the modal pick (ties → the pick with the larger total held-out).

## Shuffled-timing null (chosen rule)
Permute the rule's kill delays (K − ON, "no kill" included) across the trigger's windows, reprice + reselect, 200 times.
Report the chosen rule's Σ percentile in the null. A rule whose Σ is not above the null's 95th percentile carries no
timing information (it is just "cut earlier somewhere").

## SHIP BAR (frozen)
A V2/T0 + exit candidate ships (as an observe shadow first, per the OBSERVE-FIRST rule) only if ALL hold:
1. walk-forward held-out ΔΣ vs T0 > 0 in ≥ 3 of 4 folds AND in both 2026 folds;
2. full-sample window-bootstrap 95 % CI lower bound of ΔΣ vs T0 > 0 AFTER the haircut, defined as lower bound − 0.5 × the
   point ΔΣ (50 % in-sample haircut; the rule/parameters were derived from this replay);
3. ≥ 8 windows with fills; no single union window ≥ 50 % of the positive ΔΣ;
4. shuffled-timing null: above the null 95th percentile.
Else: report the best candidate and say NO-SHIP.

## Blocks reported
③ replay (all of the above) · ② master 10-07 (2 fills) · ① live 10-07 + 10-22 (SOL/APT/HBAR) priced under every rule
(at 5× and %), the 10-22 window and the last 48 h (10-21 → 10-23) BTC path with each rule's kill time.
