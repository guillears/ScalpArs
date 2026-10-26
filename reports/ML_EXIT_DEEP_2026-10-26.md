# MOMENTUM-LONG EXIT — deep study (2026-10-26)

**Operator's hypothesis:** "maybe the trades are good, the exit strategy is wrong — we could get more on winners, and the same on losers."
**Spec:** `reports/NEXT_BATCH_REVIEW_PLAN.md` §4 "⭐ ML EXIT DEEP REVIEW". Master-led (MASTER OVER BACKTEST); the 10-05 yr5 study is the backtest cross-check only.
**Study only:** nothing shipped, no engine / config / template edit, no git, no bot API. `scripts/validate_against_master.py` → ALL CHECKS PASS (run first, STACK 2026-10-26a).

## Plain-language summary

**The exit is not where the money is leaking. The hypothesis is refuted on the real fills, and the backtest agrees.** Of 36 exit changes tried (22 for the winner side, 12 for the loser side, 2 post-hoc wider stops), **none improves the sleeve by a margin we can trust**. The best whole-cohort change adds +0.02 %/fill, and that comes from one day in the washed-out window. Most changes **lose** money.

Why:

1. **Winners: the money after our exit is real but cannot be caught.**
   - The average winner gives back **+0.43** from its peak (median +0.42; Σ +51.6 % points on 119 winners). That is the "money on the table".
   - After our runner exit, price is a coin flip. At +15 min it is higher than our exit only **42 %** of the time (median −0.08). At +60 min it is higher 50 % of the time (median 0.00).
   - A trail that stays on for another −0.5 after our exit would bank **+0.01 per winner on average** (median −0.12). Only 32 % of runner exits would gain from it.
   - So every wider trail, later arm, higher lock or partial "let it run" pays on the few runners exactly what it gives back on the reversals:
     - arm 0.8: −0.025 %/fill;
     - trail k 1.5: −0.011;
     - half-position trailed at k 2.0: −0.007.
   - The one exception is **HARD_TP_LADDER exits**: those fills keep running (+0.24 realisable each). But there are only 11 of them, so switching the ladder off gives +0.021 %/fill, all from 2 BASE/B1 days.
2. **Losers: they are fast, and a loser cannot be told apart from a slow winner early enough.**
   - 71 % of losers never reach +0.10 in their first 10 minutes. But **20 % of winners look exactly the same** at 10 min (24 of 119; they finish at +0.47 on average).
   - By the time a loser can be identified (10 min), it is already at −0.56. Cutting it there saves only ≈ 0.25 per loser, while each winner cut costs ≈ 0.68.
   - Every tighter stop, never-green exit and time-decay stop is **negative** (−0.008 to −0.063 %/fill). Three are significantly negative even at 95 %: stop −0.6, linear time-decay, and never-green 15 min at +0.10.
   - About half of the winners (47 %) dip below −0.3 before they win, and 21 % dip below −0.5. That is why a tighter stop kills winners.
3. **"The same on losers" already holds.** Losers close at their stop: median trough −0.79, final −0.81. The stop width question was re-checked post-hoc:
   - Stop −1.2: on the 44 live-stopped fills, 12 saved (+11.8 Σ %) and 31 deeper (−12.3). That is a wash, the same as the 10-05 verdict.
4. **The sleeve's problem is still the entries and regime, not the exit.**
   - The same exit earns +0.53 %/fill in BASE and −0.13 in B17–B24.
   - In the losing eras, the losers get *more* numerous (G5/G6 never-green share 83 % / 74 % vs 50–67 % before). The winners' peak and giveback show no era trend (MFE median +0.65…+0.96, giveback +0.38…+0.45 across era groups).
   - That matches the PREP finding that a trade's fate is set by BTC in the 15 minutes after entry.

**One state-conditional result clears the Bonferroni bar: raising the runner lock 0.10 → 0.20 on high-ATR longs (entry ATR ≥ 0.50).**
- It changes 15 of 54 fills, all upward. That is +0.025 %/fill in state, +1.35 % points over 71 days ≈ **+$10 at today's ticket**.
- It fails the G6 leg (0 fires in B17–B24), and the backtest is null (−0.001 on 704 fills).
- It is an OPERATOR CALL. My recommendation is **do not change the exit for it**: it is statistically clean but economically negligible, so I register it as an observe line.

**Recommendation:**
- No ML exit change.
- Retire the 2026-08-26 HIGH-ATR looser-trail watch (③): its own gate is now decided against it.
- Register two frozen observe lines (below).
- Put the ML effort where the evidence points: entries and regime (the PREP B25 lines, DL 273, the post-entry BTC-15-min fate).

## Key numbers (KEEP master = 158 fills on today's stack, % at 1×; B25 provisional = 9; PROBE = 166, separate)

| | value |
|---|---|
| Path coverage | 158/158 master (135 ticks + 23 1m), 9/9 B25 (1m), 166/166 probe. 41 fills topped up with 83 public kline calls (own weight ≈ 166, IP peak 910 → paused 60 s). 0 fills without a path |
| Replica fidelity (AS-WAS vs live) | master tick 126 fills: 100 % same reason, mean Δ −0.002, 94 % within ±0.05 · master 1m 23: 100 %, −0.060 · B25 1m 9: 89 %, +0.063 (LIGHT 1m bar artefact: replica peak 2.86 vs live 1.45) · yr5 BASE through this evaluator = 1679·62 %·−0.057, bit-equal to the 10-05 study |
| Post-exit stamps vs path | live `post_exit_pnl_at_5/15/30min` vs path: corr 0.991 / 0.998 / 1.000, median gap 0.000 |
| Today's-stack replica on master | 158·70 %·+0.127 (recorded stack_pct +0.124) |
| Winners (119) | MFE median +0.77 @ 19 min · giveback median +0.42 / p75 +0.48 / p90 +0.64 · MAE median −0.29 @ 3 min |
| Losers (48) | MFE median +0.04 @ 1 min · MAE median −0.79 @ 15 min · 75 / 71 / 62 % never above +0.10 by 5 / 10 / 15 min |
| After the exit, winners | +5 / +15 / +30 / +60 min median −0.00 / −0.08 / −0.03 / 0.00 (p90 +0.43 / +0.82 / +0.98 / +1.11) · max before −0.5 pullback median +0.38 (p90 +1.14) · realisable by a −0.5 trail: median −0.12, mean +0.01 |
| After the exit, losers | 40 % trade back above entry within 60 min, 23 % reach +0.40 |

## Top candidates — before → after (replica BASE → variant, avg %/fill; Σ Δ at today's ML ticket ≈ $717 notional)

| candidate | master 158 | G0 BASE | G1 B1 | G5 B12–16 | G6 B17–24 | B25 (9) | PROBE (166) | 95 % CI | yr5 Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Ladder off (`W_LAD_off`) | +0.127 → +0.148 (Δ +0.021, $+23) | +0.53→+0.62 | +0.19→+0.29 | −0.04→−0.05 | −0.13→−0.15 | +0.22→+0.22 | −0.14→−0.15 | [−0.017, +0.085] | +0.006 [−0.005, +0.021] | **REJECTED by gate** (ex-washed −0.000, recent groups < 0, one day = 117 % of the net) |
| Trail k 1.0→1.5 after 30 min (`W_TW30_k1.5`) | +0.127 → +0.132 (+0.005, $+6) | +0.53→+0.52 | +0.19→+0.37 | −0.04→−0.07 | −0.13→−0.13 | +0.22→+0.30 | −0.14→−0.14 | [−0.021, +0.046] | −0.005 (opposite sign) | **KEEP OBSERVING** (mechanical; one B1 day carries it) |
| Ladder loose (`W_LAD_loose`) | +0.003 | | | | | | | [−0.015, +0.029] | +0.003 | KEEP OBSERVING (OBS-3) |
| Trail k 1.25 (`W_K1.25`) | +0.002 | | | | | +0.22→+0.43 | | [−0.028, +0.047] | −0.014 (**significant opposite**) | KEEP OBSERVING (mechanical; backtest caveat) |
| Arm 0.40→0.80 (`W_ARM0.8`) | −0.025 | +0.53→+0.41 | +0.19→+0.03 | −0.04→+0.03 | −0.13→−0.14 | +0.22→+0.37 | −0.15 | [−0.106, +0.051] | −0.006 | **REJECTED** |
| Half at exit, half k 2.0 no ladder | −0.007 | | | −0.045 | +0.040 | | | [−0.040, +0.033] | −0.011 | **REJECTED** |
| Never-green 10 min < +0.10 (`L_NG10_0.10`) | −0.044 (21 winners cut, 17 losers saved) | +0.53→+0.44 | | −0.04→−0.16 | −0.13→−0.10 | +0.22→+0.08 | | [−0.106, +0.013] | −0.022 | **REJECTED** |
| Tighter stop −0.6 (`L_SL-0.6`) | −0.063 | | | | | | | [−0.125, −0.006] | −0.025 [−0.050, −0.002] | **REJECTED** (both blocks significantly negative) |
| Wider stop −1.2 (post-hoc) | +0.004 · saved 12 (+11.8) vs deeper 31 (−12.3) | | | −0.027 | +0.011 | | | [−0.056, +0.063] | +0.013 [−0.018, +0.045] | **DECIDED no change** (post-hoc; a wash; 10-05 verdict stands) |
| **Lock 0.10→0.20, only when entry ATR ≥ 0.50** | in-state 54·67 %·+0.084 → +0.109 (+0.025/fill; cohort +0.009 [+0.005, +0.013], Bonferroni low +0.002) | +0.038 | +0.013 | +0.033 | 0.000 (0 fires) | +0.037 (4) | — | P(Δ ≤ 0) 0.000 · top day 9 % · top pair 18 % | −0.001 [−0.020, +0.015] on 704 | **OPERATOR CALL** — passes Bonferroni; fails the G6 leg and the backtest; ≈ $10 over 71 days. Recommend: don't ship, observe (OBS-2) |

Per-batch Σ Δ for the top variants: §E "Per BATCH" below and `ML_EXIT_DEEP_2026-10-26_variant_by_batch.csv`. Haircut (30–50 %) only matters for positive candidates. After the haircut, ladder-off is +0.010…+0.015 and lock-0.20@ATR_hi is +0.013…+0.018 per in-state fill. Neither changes the sleeve's sign in the losing eras.

## Verdicts per candidate (pre-declared bar in `scripts/study_ml_exit_b25_report.py` docstring)

The bar was fixed before the grid was read.
- Ship-grade needs all of:
  - a Bonferroni day-block CI above 0 (K = 216 tests);
  - Δ > 0 in ≥ 5 era groups, including G5 and G6;
  - Δ > 0 without the washed-out window;
  - top day and top pair each < 50 %;
  - stop-type rules read on the live-stopped cohort.
- Below that:
  - nominal CI above 0 → OPERATOR CALL;
  - Δ > 0 with the CI including 0 → KEEP OBSERVING;
  - Δ ≤ 0 → REJECTED.

**DECIDED by the gate:**
- **All 12 loser-side rules: REJECTED.** That covers tighter stops −0.5 / −0.6 / ATR 1.0, never-green 5 / 10 / 15 min × +0.05 / +0.10, and time-decay 15 / 30 / linear. Δ is −0.008 to −0.063.
  - Stop −0.6, linear time-decay and NG15 at +0.10 have the whole 95 % CI below 0.
  - The backtest agrees in sign on all 12.
- **Winner-side, REJECTED (Δ ≤ 0):**
  - arm 0.6 / 0.8 / 1.0;
  - trail k 0.75 / 1.5 / 2.0;
  - lock 0.0 / 0.2 / 0.3 (whole cohort);
  - ladder wide;
  - peak-based widening (k 1.5 / 2.0 after +1.0);
  - time-widening k 2.0 after 30 min;
  - hold 360 min (Δ ≡ 0: no fill reaches the 180-min clock);
  - all three partials.
- **Ladder off: REJECTED as a whole-cohort rule.** Its gain is the washed-out window: one day is 117 % of the net, ex-washed Δ is −0.000, and G5 / G6 are negative.
- **Time-widening k 1.5 after 15 min: REJECTED.** Recent groups are negative and ex-washed is −0.001.
- **Wider stops −1.0 / −1.2 (post-hoc): DECIDED, no change.**
  - On the live-stopped cohort they are a wash (−1.2: saved +11.8 vs deeper −12.3).
  - −1.0 changes 1 fill the replica did not stop. That fill had an ATR-widened −1.2 stop, so a fixed −1.0 is tighter there. It is a real change, not an artefact.
  - The 10-05 stop verdict stands.
- **2026-08-26 HIGH-ATR WATCH ③ (looser 1.5×ATR trail for high-ATR longs): gate RESOLVED, no ship.**
  - Its gate was "N ≥ 15 current-stack with era-consistent direction". It now has 54 high-ATR (≥ 0.50) master fills: k 1.5 gives +0.027 in state, P(Δ ≤ 0) 0.36, positive in only 2 of 7 era groups, recent −0.006. yr5 gives −0.009 [−0.021, +0.005].
  - **Retire it** (log the resolution in DECISION_LOG; delete it from CURRENT_STATE).

**OPERATOR CALL (below the bar):**
- **Lock 0.20 for entry ATR ≥ 0.50** (`W_LOCK0.2 @ ATR_hi`).
  - Mechanism: on high-ATR pairs with modest peaks the +0.10 lock is the operative exit (the ONDO 07-27 anatomy in CURRENT_STATE). At 0.20 the floor banks +0.10 more on fills that fall through it.
  - On master it cut no winner, because those fills never revisited 0.20 and then recovered. In the backtest some do, and the net there is −0.001.
  - Quant view: **not worth a config change** (+$10 over 71 days, G6 has zero fires, backtest null). If the operator wants it anyway, it is a one-number change with a revert gate:
    - revert if, on the first 15 fresh ATR ≥ 0.50 fills that touch the lock, Σ Δ vs the 0.10 shadow ≤ 0.

**KEEP OBSERVING — frozen lines (thresholds fixed 2026-10-26, never re-tuned):**

| id | rule (frozen) | fresh cohort | promote → ship-grade review | retire |
|---|---|---|---|---|
| OBS-1 `ML_EXIT_LADDER_OFF_HIATR` | hard-TP ladder OFF when entry ATR % ≥ 0.50 (runner trail k 1.0 + lock 0.10 still bound it). Master in-state +0.072/fill (P 0.19, top day 97 %); yr5 in-state +0.018 [−0.009, +0.051], positive in all 3 seeds and both halves | ML KEEP fills opened ≥ 2026-10-27 00:00 UTC with entry ATR ≥ 0.50 **whose peak reaches +1.25** (the only fills the rule touches) | ≥ 10 such fills on ≥ 5 days ∧ Σ Δ > 0 ∧ ≥ 60 % of them better ∧ no day ≥ 50 % of Σ Δ | 10 fills with Σ Δ ≤ 0 |
| OBS-2 `ML_EXIT_LOCK020_HIATR` | runner lock +0.20 instead of +0.10 when entry ATR % ≥ 0.50 | ML KEEP fills ≥ 2026-10-27 with entry ATR ≥ 0.50 that arm (peak ≥ 0.40) | ≥ 15 fills whose exit the rule changes, on ≥ 8 days ∧ Σ Δ > 0 ∧ ≤ 20 % of the changed fills worse | 15 changed fills with Σ Δ ≤ 0, or > 30 % worse |
| (OBS-3) `W_TW30_k1.5` / `W_K1.25` / `W_LAD_loose` | mechanical KEEP OBSERVING (Δ +0.002…+0.005; one B1 day carries each; W_K1.25 is significantly opposite in the backtest) | read together with OBS-1 at the same review, same tool | same bar as OBS-2 (N ≥ 15 changed) | same as OBS-2 |

Each line is priced by `scripts/study_ml_exit_b25_run.py` (the validated replica) on the fresh fills. Run `study_ml_exit_b25_fetch.py` first for fills without ticks. Evidence counts once per fill. Haircut 30–50 % at promotion.

## What could NOT be tested

1. **Live exit micro-structure beyond the replica.** Exit slippage / maker-exit fills, latency between the tick crossing and the order, and partial fills. The replica fills at the first tick through the level (validated: mean Δ −0.002 on ticks). A wider trail's extra exits are assumed to fill as well as today's.
2. **1m-path fills (23 master G6 + 9 B25).** Their intrabar order is assumed (green bar o→l→h→c, red o→h→l→c). Replica fidelity there is 100 % same reason but mean Δ −0.06, and the B25 LIGHT path overstates its peak (2.86 vs live 1.45). The G6 and B25 per-variant numbers carry that noise; ticks for those days would remove it (an aggTrades fetch, not run within the shared request budget).
3. **True partial exits.** Splitting an order is not in the engine. The partials are priced as 50 % / 50 % of two independently replicated exits. Min-notional, a second exit fee and the extra order are not modelled.
4. **The recovery hold (RH) on fills before 2026-10-02** uses the entry BTC RSI rebuilt by the live ruler (the stamp exists on only 27 % of fills). Variants that change stops interact with RH; the replica follows the live RH rules.
5. **Post-exit horizon** is capped at 4 h after the exit. Rows not pulled back within 4 h are counted at their 4 h high (`runpb*_cens` in the per-fill CSV).
6. **State splits are 2-way only** (BTC regime SB vs rest; ATR terciles at 0.42 / 0.50). Not tested: BTC post-entry path as an exit trigger (e.g. "exit when BTC turns down"), breadth-based exits, and any exit that uses information the engine does not have at exit time. The PREP study shows BTC's next-15-min move decides the trade. A BTC-reactive exit is the one exit idea this grid does not cover. It needs BTC ticks aligned to every fill (next study, if wanted).
7. **Power.** 158 master fills over 71 days, with 37 in the losing G6 group. A real +0.03 %/fill exit edge is below the detectable size here (95 % CI widths ≈ ±0.05).
8. **No re-entry modelling.** An exit that holds longer can block a later entry on the same pair or slot. Second-order effects on slot occupancy are not priced.

## Files

- **Scripts:**
  - `scripts/study_ml_exit_b25_data.py`: cohort plus path loader. It reuses the PREP_B25 cohort and the yr5 replica; the readers are extended read-only.
  - `scripts/study_ml_exit_b25_fetch.py`: public 1m top-up into `reports/ML_EXIT_DEEP_2026-10-26_fetch/`.
  - `scripts/study_ml_exit_b25_run.py`: per-fill engine with the pre-declared grid. The evaluator is guard-checked bit-equal to `ml_exit_optimize_yr5.evaluate`, with 0 failures.
  - `scripts/study_ml_exit_b25_report.py`: tables and verdicts.
- **CSVs (each `reports/ML_EXIT_DEEP_2026-10-26_<suffix>.csv`):**
  - `_variants` (fill × variant);
  - `_fills_paths`;
  - `_postexit` (per fill: post-exit, MAE / MFE, BASE replica);
  - `_variant_summary` (variant × scope, every gate leg);
  - `_variant_by_group`;
  - `_variant_by_batch`;
  - `_state_summary`;
  - `_yr5_variants`;
  - `_yr5_summary`.
- **Full tables:** below, also in `reports/ML_EXIT_DEEP_2026-10-26_tables.md`.

---

## A. Cohort and price-path coverage

| block / era group | fills | tick path | 1m path | no path | post-exit ≥ 60 min covered | WR | avg % (stack, 1×) |
|---|---|---|---|---|---|---|---|
| KEEP master — all | 158 | 135 | 23 | 0 | 158 | 71% | +0.124 |
|   G0 BASE | 32 | 32 | 0 | 0 | 32 | 94% | +0.485 |
|   G1 B1 | 15 | 15 | 0 | 0 | 15 | 80% | +0.256 |
|   G2 B2 | 10 | 10 | 0 | 0 | 10 | 80% | +0.191 |
|   G3 B3 | 22 | 22 | 0 | 0 | 22 | 77% | +0.147 |
|   G4 B4-B11 | 14 | 14 | 0 | 0 | 14 | 79% | +0.060 |
|   G5 B12-B16 | 28 | 28 | 0 | 0 | 28 | 57% | -0.083 |
|   G6 B17-B24 | 37 | 14 | 23 | 0 | 37 | 49% | -0.091 |
| B25 provisional | 9 | 0 | 9 | 0 | 9 | 78% | +0.156 |
| PROBE (separate) | 166 | 153 | 13 | 0 | 166 | 48% | -0.149 |

Mixed tick/1m paths: 0. Fills with NO path: 0.

## B. Replica fidelity on this cohort (AS-WAS exit params vs the real exit; bot exits from 2026-06-24 20:20 on)

| block | path | N | same exit reason | mean Δ (rep − real) | \|Δ\| ≤ 0.05 | corr |
|---|---|---|---|---|---|---|
| B25 | 1m | 9 | 89% | +0.063 | 78% | 0.939 |
| MASTER | 1m | 23 | 100% | -0.060 | 70% | 0.988 |
| MASTER | tick | 126 | 100% | -0.002 | 94% | 0.991 |
| PROBE | 1m | 12 | 67% | -0.056 | 25% | 0.925 |
| PROBE | tick | 150 | 99% | -0.013 | 97% | 0.910 |

KEEP / B25 fills off by > 0.15 or a different reason: AVAXUSDT BASE (tick) live RUNNER +0.73 vs replica RUNNER +1.07; JTOUSDT BASE (tick) live RUNNER +1.58 vs replica RUNNER +0.52; ENAUSDT B15 (tick) live STOP -0.99 vs replica STOP -0.83; APTUSDT B22 (1m) live RUNNER +0.21 vs replica RUNNER -0.25; INJUSDT B23 (1m) live STOP -0.70 vs replica STOP -0.88; LIGHTUSDT B25 (1m) live LADDER +0.82 vs replica RUNNER +1.61

Today's-stack replica (BASE) vs recorded stack_pct on KEEP master: BASE +0.127 vs recorded +0.124 (old-era fills ran older exits — BASE re-prices them on today's exit).

## C. Post-exit run per WINNER (how far price went after our exit) — KEEP master + B25

Units: P&L points at 1× measured from the live exit price ((p − exit) / entry × 100). `max before −x` = the highest point price reached after our exit before it first pulled back x from its running high (≤ 4 h; censored rows counted at their 4 h high). `realisable by a −x trail` = that high − x (what a trail of width x after our exit would have banked, before fees/slip). Cells: median / p75 / p90 (mean).

| exit reason (live) | N | giveback = peak − exit | +5 min | +15 min | +30 min | +60 min | max before −0.3 | max before −0.5 | max before −1.0 | realisable −0.5 trail |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL winners | 119 | +0.42 / +0.48 / +0.64 (mean +0.43) | -0.00 / +0.18 / +0.43 (mean +0.01) | -0.08 / +0.27 / +0.82 (mean +0.06) | -0.03 / +0.43 / +0.98 (mean +0.12) | +0.00 / +0.61 / +1.11 (mean +0.11) | +0.23 / +0.43 / +0.71 (mean +0.32) | +0.38 / +0.73 / +1.14 (mean +0.51) | +0.75 / +1.36 / +2.11 (mean +1.00) | -0.12 / +0.23 / +0.64 (mean +0.01) |
| FIXED_TP | 1 | +0.00 / +0.00 / +0.00 (mean +0.00) | -0.51 / -0.51 / -0.51 (mean -0.51) | -0.39 / -0.39 / -0.39 (mean -0.39) | -1.72 / -1.72 / -1.72 (mean -1.72) | -1.40 / -1.40 / -1.40 (mean -1.40) | +1.14 / +1.14 / +1.14 (mean +1.14) | +1.14 / +1.14 / +1.14 (mean +1.14) | +1.14 / +1.14 / +1.14 (mean +1.14) | +0.64 / +0.64 / +0.64 (mean +0.64) |
| LADDER | 11 | +0.46 / +0.51 / +0.62 (mean +0.48) | +0.24 / +0.50 / +0.68 (mean +0.30) | +0.44 / +0.95 / +1.48 (mean +0.53) | +0.24 / +0.47 / +0.70 (mean +0.26) | +0.42 / +0.69 / +0.85 (mean +0.28) | +0.19 / +0.38 / +1.09 (mean +0.43) | +0.70 / +1.02 / +1.36 (mean +0.74) | +1.67 / +1.97 / +2.07 (mean +1.85) | +0.20 / +0.52 / +0.86 (mean +0.24) |
| RUNNER | 99 | +0.41 / +0.48 / +0.64 (mean +0.43) | -0.03 / +0.15 / +0.36 (mean -0.02) | -0.12 / +0.24 / +0.78 (mean -0.02) | -0.08 / +0.37 / +0.89 (mean -0.04) | -0.07 / +0.60 / +1.04 (mean -0.02) | +0.23 / +0.44 / +0.66 (mean +0.30) | +0.32 / +0.66 / +1.08 (mean +0.48) | +0.62 / +1.16 / +2.09 (mean +0.89) | -0.18 / +0.16 / +0.58 (mean -0.02) |
| TRAIL_OLD | 8 | +0.46 / +0.58 / +0.68 (mean +0.46) | +0.16 / +0.31 / +0.70 (mean +0.13) | +0.22 / +0.82 / +1.31 (mean +0.51) | +0.12 / +0.98 / +5.34 (mean +2.09) | +0.22 / +0.77 / +4.54 (mean +1.68) | +0.27 / +0.34 / +0.40 (mean +0.27) | +0.41 / +0.59 / +0.90 (mean +0.48) | +0.57 / +1.54 / +2.58 (mean +1.13) | -0.09 / +0.09 / +0.40 (mean -0.02) |

Share of winners where price was HIGHER than our exit at +15 / +30 / +60 min: 42% / 50% / 50% (by ≥ +0.30: 24% / 30% / 38%).

Per era group — money left on the table by winners (Σ over the group's winners, % points at 1×; $ at the B25 ML ticket $717 notional):

| group | winners | Σ giveback (peak − exit) | Σ max-before-−0.5 | Σ realisable −0.5 trail | median realisable / winner | $ realisable at today's ticket |
|---|---|---|---|---|---|---|
| G0 BASE | 30 | +13.41 | +15.35 | +0.35 | -0.17 | $+3 |
| G1 B1 | 12 | +5.13 | +4.67 | -1.33 | -0.16 | $-10 |
| G2 B2 | 8 | +3.62 | +2.30 | -1.70 | -0.25 | $-12 |
| G3 B3 | 17 | +6.95 | +7.98 | -0.52 | -0.07 | $-4 |
| G4 B4-B11 | 11 | +5.24 | +4.60 | -0.90 | -0.12 | $-6 |
| G5 B12-B16 | 16 | +6.73 | +8.78 | +0.78 | -0.05 | $+6 |
| G6 B17-B24 | 18 | +7.51 | +7.75 | -1.25 | -0.19 | $-9 |
| B25 (prov.) | 7 | +3.04 | +8.92 | +5.42 | +0.67 | $+39 |
- live stamp check post_exit_pnl_at_5min − exit vs path +5 min: N 165 · corr 0.991 · median gap -0.000
- live stamp check post_exit_pnl_at_15min − exit vs path +15 min: N 165 · corr 0.998 · median gap -0.000
- live stamp check post_exit_pnl_at_30min − exit vs path +30 min: N 165 · corr 1.000 · median gap -0.000

LOSERS after the exit (N 48): +15 / +60 min median +0.06 / +0.07; share back above ENTRY within 60 min (post-exit max ≥ −exit): 40%; share recovering ≥ +0.40 above entry within 60 min: 23%.

## D. MAE / MFE — winners vs losers (KEEP master + B25, net %, live life of the trade)

| | N | MFE (peak) median / p75 / p90 | MFE minute median | MAE (trough) median / p25 / p10 | MAE minute median |
|---|---|---|---|---|---|
| winners | 119 | +0.77 / +1.21 / +1.63 | 18.9 | -0.29 / -0.46 / -0.66 | 2.9 |
| losers | 48 | +0.04 / +0.17 / +0.25 | 1.1 | -0.79 / -0.91 / -1.13 | 15.1 |

Losers that NEVER went green (peak < x by N minutes):

| threshold | by 5 min | by 10 min | by 15 min |
|---|---|---|---|
| peak < +0.05 | 73% (35/48) | 67% (32/48) | 56% (27/48) |
| peak < +0.10 | 75% (36/48) | 71% (34/48) | 62% (30/48) |

The same cut on WINNERS (the cost side of a never-green exit):

| threshold | by 5 min | by 10 min | by 15 min |
|---|---|---|---|
| peak < +0.05 | 24% (29/119) | 13% (15/119) | 12% (14/119) |
| peak < +0.10 | 35% (42/119) | 20% (24/119) | 18% (21/119) |

Winners that dipped before winning (trough before the first +0.40 touch):

| dip below | share of winners | their avg final % |
|---|---|---|
| -0.3 | 47% (56/119) | +0.490 |
| -0.5 | 21% (25/119) | +0.422 |
| -0.7 | 7% (8/119) | +0.313 |

Live-stamp cross-check: path MFE vs stamped peak_pnl corr 0.985, median gap -0.000; MAE vs trough_pnl corr 0.997.

Per era group (winners MFE median · losers never-green share peak < +0.10 by 10 min):

| group | W N | W MFE med | W giveback med | L N | L never-green@10 |
|---|---|---|---|---|---|
| G0 BASE | 30 | +0.87 | +0.45 | 2 | 50% |
| G1 B1 | 12 | +0.89 | +0.41 | 3 | 67% |
| G2 B2 | 8 | +0.96 | +0.38 | 2 | 50% |
| G3 B3 | 17 | +0.65 | +0.38 | 5 | 60% |
| G4 B4-B11 | 11 | +0.78 | +0.44 | 3 | 67% |
| G5 B12-B16 | 16 | +0.70 | +0.44 | 12 | 83% |
| G6 B17-B24 | 18 | +0.66 | +0.40 | 19 | 74% |
| B25 (prov.) | 7 | +0.81 | +0.42 | 2 | 50% |

## E. Two-sided counterfactuals on today's stack (validated replica) — KEEP master

36 read variants (partials included) × (whole cohort + 5 state-conditional reads) = **K = 216 tests**; Bonferroni α = 0.05 / 216 = 0.00023 (two-sided CI 99.977 %). Day-block bootstrap, 20,000 resamples. Δ = variant − BASE, paired per fill on the same path. 'Rest' fills = fills whose exit the variant did not touch.

| variant | N·WR·avg BASE → after | Δ %/fill [95 % CI] | Bonf. CI | P(Δ≤0) | groups + | G5 / G6 Δ | ex-washed Δ | Σ Δ on BASE winners / losers | winners cut · losers worse · losers saved · winners more | top day / pair share | $ Σ at ticket | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_LAD_off | 158·70%·+0.127 → 70%·+0.148 | +0.021 (haircut +0.010…+0.014) [-0.017, +0.085] | [-0.030, +0.160] | 0.25 | 2.0/7.0 | -0.003 / -0.018 | -0.000 | +3.26 / +0.00 | 15 · 0 · 0 · 3 | 117% (one day/pair > the whole net) / 117% (one day/pair > the whole net) | $+23 | REJECTED (recent < 0, ex-washed ≤ 0) |
| W_TW30_k1.5 | 158·70%·+0.127 → 70%·+0.132 | +0.005 (haircut +0.003…+0.004) [-0.021, +0.046] | [-0.032, +0.096] | 0.43 | 2.0/7.0 | -0.024 / +0.001 | +0.007 | +0.80 / +0.00 | 18 · 0 · 0 · 3 | 339% (one day/pair > the whole net) / 339% (one day/pair > the whole net) | $+6 | KEEP OBSERVING |
| X_SL-1.2 (post-hoc) | 158·70%·+0.127 → 78%·+0.131 | +0.004 (haircut +0.002…+0.003) [-0.056, +0.063] | [-0.098, +0.118] | 0.47 | 4.0/7.0 | -0.027 / +0.011 | +0.004 | +0.00 / +0.57 | 0 · 31 · 14 · 0 | 464% (one day/pair > the whole net) / 370% (one day/pair > the whole net) | $+4 | KEEP OBSERVING |
| W_LAD_loose | 158·70%·+0.127 → 70%·+0.131 | +0.003 (haircut +0.002…+0.002) [-0.015, +0.029] | [-0.027, +0.060] | 0.42 | 2.0/7.0 | -0.003 / -0.018 | +0.004 | +0.51 / +0.00 | 14 · 0 · 0 · 4 | 283% (one day/pair > the whole net) / 283% (one day/pair > the whole net) | $+4 | KEEP OBSERVING |
| W_K1.25 | 158·70%·+0.127 → 70%·+0.129 | +0.002 (haircut +0.001…+0.001) [-0.028, +0.047] | [-0.041, +0.097] | 0.51 | 2.0/7.0 | -0.014 / -0.010 | +0.008 | +0.28 / +0.00 | 49 · 0 · 0 · 9 | 995% (one day/pair > the whole net) / 995% (one day/pair > the whole net) | $+2 | KEEP OBSERVING |
| W_TW15_k1.5 | 158·70%·+0.127 → 70%·+0.128 | +0.001 (haircut +0.000…+0.000) [-0.031, +0.045] | [-0.048, +0.095] | 0.53 | 2.0/7.0 | -0.034 / -0.005 | -0.001 | +0.11 / +0.00 | 34 · 0 · 0 · 9 | 2521% (one day/pair > the whole net) / 2521% (one day/pair > the whole net) | $+1 | REJECTED (recent < 0, ex-washed ≤ 0) |
| W_HOLD360 | 158·70%·+0.127 → 70%·+0.127 | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | 1.00 | 0.0/7.0 | +0.000 / +0.000 | +0.000 | +0.00 / +0.00 | 0 · 0 · 0 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $+0 | REJECTED (Δ ≤ 0) |
| W_TW30_k2.0 | 158·70%·+0.127 → 70%·+0.126 | -0.002 [-0.027, +0.035] | [-0.042, +0.083] | 0.59 | 1.0/7.0 | -0.037 / -0.003 | -0.001 | -0.25 / +0.00 | 18 · 0 · 0 · 3 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-2 | REJECTED (Δ ≤ 0) |
| W_LOCK0.0 | 158·70%·+0.127 → 59%·+0.123 | -0.004 [-0.016, +0.011] | [-0.022, +0.026] | 0.74 | 1.0/7.0 | -0.010 / +0.023 | -0.003 | -0.65 / +0.00 | 24 · 0 · 0 · 3 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-5 | REJECTED (Δ ≤ 0) |
| W_LAD_wide | 158·70%·+0.127 → 70%·+0.122 | -0.005 [-0.023, +0.019] | [-0.040, +0.047] | 0.72 | 1.0/7.0 | -0.003 / -0.018 | -0.001 | -0.86 / +0.00 | 16 · 0 · 0 · 2 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-6 | REJECTED (Δ ≤ 0) |
| W_ARM1.0 | 158·70%·+0.127 → 52%·+0.122 | -0.006 [-0.115, +0.101] | [-0.208, +0.194] | 0.55 | 4.0/7.0 | +0.011 / +0.010 | +0.024 | -1.67 / +0.75 | 31 · 0 · 1 · 33 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-7 | REJECTED (Δ ≤ 0) |
| P_half_K2.0_noLAD | 158·70%·+0.127 → 70%·+0.121 | -0.007 [-0.040, +0.033] | [-0.061, +0.072] | 0.66 | 3.0/7.0 | -0.045 / +0.040 | -0.010 | -1.08 / +0.00 | 59 · 0 · 0 · 20 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-8 | REJECTED (Δ ≤ 0) |
| W_LOCK0.2 | 158·70%·+0.127 → 70%·+0.120 | -0.008 [-0.035, +0.014] | [-0.065, +0.029] | 0.72 | 3.0/7.0 | +0.015 / +0.018 | -0.008 | -1.24 / +0.00 | 8 · 0 · 0 · 40 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-9 | REJECTED (Δ ≤ 0) |
| L_NG10_0.05 | 158·70%·+0.127 → 63%·+0.119 | -0.008 [-0.055, +0.036] | [-0.103, +0.085] | 0.64 | 5.0/7.0 | +0.000 / +0.028 | -0.010 | -7.98 / +6.66 | 13 · 0 · 14 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-9 | REJECTED (Δ ≤ 0) |
| W_K1.5 | 158·70%·+0.127 → 70%·+0.117 | -0.011 [-0.046, +0.036] | [-0.064, +0.089] | 0.72 | 1.0/7.0 | -0.051 / -0.005 | -0.011 | -1.70 / +0.00 | 49 · 0 · 0 · 11 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-12 | REJECTED (Δ ≤ 0) |
| W_PW1.0_k1.5 | 158·70%·+0.127 → 70%·+0.116 | -0.012 [-0.031, +0.009] | [-0.047, +0.031] | 0.87 | 3.0/7.0 | -0.028 / +0.010 | -0.018 | -1.85 / +0.00 | 20 · 0 · 0 · 5 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-13 | REJECTED (Δ ≤ 0) |
| L_NG5_0.05 | 158·70%·+0.127 → 54%·+0.109 | -0.018 [-0.085, +0.046] | [-0.141, +0.097] | 0.70 | 3.0/7.0 | -0.024 / +0.022 | -0.015 | -17.43 / +14.58 | 27 · 0 · 30 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-20 | REJECTED (Δ ≤ 0) |
| W_ARM0.6 | 158·70%·+0.127 → 59%·+0.108 | -0.019 [-0.076, +0.033] | [-0.128, +0.071] | 0.76 | 2.0/7.0 | +0.056 / -0.019 | -0.009 | -3.08 / +0.00 | 17 · 0 · 0 · 21 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-22 | REJECTED (Δ ≤ 0) |
| X_SL-1.0 (post-hoc) | 158·70%·+0.127 → 74%·+0.107 | -0.020 [-0.063, +0.017] | [-0.108, +0.048] | 0.84 | 2.0/7.0 | +0.036 / -0.050 | -0.023 | -2.20 / -1.03 | 1 · 31 · 12 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-23 | REJECTED (Δ ≤ 0) |
| L_SLatr1.0 | 158·70%·+0.127 → 68%·+0.106 | -0.022 [-0.070, +0.010] | [-0.124, +0.020] | 0.85 | 4.0/7.0 | +0.026 / +0.000 | -0.025 | -4.88 / +1.48 | 4 · 0 · 12 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-24 | REJECTED (Δ ≤ 0) |
| W_K0.75 | 158·70%·+0.127 → 71%·+0.105 | -0.022 [-0.068, +0.015] | [-0.115, +0.039] | 0.87 | 2.0/7.0 | -0.014 / +0.042 | -0.005 | -4.00 / +0.52 | 20 · 0 · 1 · 65 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-25 | REJECTED (Δ ≤ 0) |
| P_half_ARM1.0_K2.0_noLAD | 158·70%·+0.127 → 53%·+0.105 | -0.022 [-0.080, +0.035] | [-0.125, +0.095] | 0.78 | 3.0/7.0 | -0.035 / +0.025 | -0.023 | -3.57 / +0.06 | 63 · 0 · 1 · 43 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-25 | REJECTED (Δ ≤ 0) |
| W_PW1.0_k2.0 | 158·70%·+0.127 → 70%·+0.105 | -0.023 [-0.048, +0.002] | [-0.072, +0.026] | 0.96 | 2.0/7.0 | -0.043 / +0.010 | -0.033 | -3.60 / +0.00 | 20 · 0 · 0 · 5 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-26 | REJECTED (Δ ≤ 0) |
| W_ARM0.8 | 158·70%·+0.127 → 56%·+0.103 | -0.025 [-0.106, +0.051] | [-0.174, +0.110] | 0.74 | 2.0/7.0 | +0.071 / -0.006 | -0.002 | -4.65 / +0.75 | 25 · 0 · 1 · 29 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-28 | REJECTED (Δ ≤ 0) |
| P_half_K3.0_noLAD_h360 | 158·70%·+0.127 → 70%·+0.103 | -0.025 [-0.073, +0.031] | [-0.106, +0.088] | 0.83 | 2.0/7.0 | -0.051 / -0.019 | -0.026 | -3.91 / +0.00 | 67 · 0 · 0 · 13 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-28 | REJECTED (Δ ≤ 0) |
| L_NG15_0.05 | 158·70%·+0.127 → 63%·+0.096 | -0.031 [-0.071, +0.003] | [-0.118, +0.030] | 0.96 | 0.0/7.0 | -0.020 / -0.003 | -0.036 | -7.52 / +2.57 | 12 · 0 · 6 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-36 | REJECTED (Δ ≤ 0) |
| L_TD30_-0.2 | 158·70%·+0.127 → 61%·+0.093 | -0.034 [-0.081, +0.007] | [-0.129, +0.038] | 0.95 | 2.0/7.0 | -0.046 / +0.033 | -0.035 | -10.32 / +4.92 | 14 · 0 · 13 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-39 | REJECTED (Δ ≤ 0) |
| W_K2.0 | 158·70%·+0.127 → 70%·+0.093 | -0.034 [-0.072, +0.012] | [-0.096, +0.063] | 0.94 | 1.0/7.0 | -0.090 / -0.005 | -0.037 | -5.44 / +0.00 | 49 · 0 · 0 · 11 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-39 | REJECTED (Δ ≤ 0) |
| W_LOCK0.3 | 158·70%·+0.127 → 71%·+0.087 | -0.040 [-0.096, +0.007] | [-0.146, +0.038] | 0.95 | 3.0/7.0 | +0.040 / -0.041 | -0.013 | -6.92 / +0.52 | 23 · 0 · 1 · 50 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-46 | REJECTED (Δ ≤ 0) |
| L_TD15_-0.4 | 158·70%·+0.127 → 60%·+0.084 | -0.044 [-0.101, +0.004] | [-0.156, +0.034] | 0.96 | 2.0/7.0 | -0.040 / +0.042 | -0.042 | -13.31 / +6.39 | 16 · 1 · 22 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-50 | REJECTED (Δ ≤ 0) |
| L_NG10_0.10 | 158·70%·+0.127 → 59%·+0.083 | -0.044 [-0.106, +0.013] | [-0.164, +0.059] | 0.94 | 4.0/7.0 | -0.114 / +0.028 | -0.035 | -14.19 / +7.25 | 21 · 0 · 17 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-50 | REJECTED (Δ ≤ 0) |
| L_NG15_0.10 | 158·70%·+0.127 → 59%·+0.078 | -0.049 [-0.096, -0.007] | [-0.139, +0.024] | 0.99 | 1.0/7.0 | -0.079 / +0.001 | -0.050 | -10.90 / +3.13 | 18 · 1 · 7 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-56 | REJECTED (Δ ≤ 0) |
| L_SL-0.5 | 158·70%·+0.127 → 57%·+0.076 | -0.051 [-0.117, +0.008] | [-0.189, +0.048] | 0.95 | 1.0/7.0 | -0.016 / +0.067 | -0.045 | -20.33 / +12.28 | 22 · 0 · 43 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-58 | REJECTED (Δ ≤ 0) |
| L_TDlin10-40_-0.2 | 158·70%·+0.127 → 60%·+0.074 | -0.053 [-0.114, -0.004] | [-0.174, +0.028] | 0.98 | 2.0/7.0 | -0.047 / +0.018 | -0.057 | -13.42 / +4.97 | 16 · 4 · 23 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-61 | REJECTED (Δ ≤ 0) |
| L_NG5_0.10 | 158·70%·+0.127 → 47%·+0.067 | -0.061 [-0.140, +0.015] | [-0.215, +0.082] | 0.94 | 2.0/7.0 | -0.152 / +0.017 | -0.050 | -24.98 / +15.38 | 38 · 0 · 32 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-69 | REJECTED (Δ ≤ 0) |
| L_SL-0.6 | 158·70%·+0.127 → 60%·+0.065 | -0.063 [-0.125, -0.006] | [-0.190, +0.035] | 0.99 | 0.0/7.0 | -0.039 / -0.010 | -0.066 | -16.70 / +6.78 | 16 · 2 · 37 · 0 | n/a (net ≤ 0) / n/a (net ≤ 0) | $-71 | REJECTED (Δ ≤ 0) |

Stop-type and early-exit rules on the LIVE-STOPPED cohort (live exit a stop / RH ∧ replica BASE a stop): saved vs deeper; the rest of the cohort is the cost side (winners/flat fills the rule touches):

| variant | live-stopped N | saved (n · Σ %) | deeper (n · Σ %) | non-stopped fills changed · Σ Δ % |
|---|---|---|---|---|
| L_SL-0.5 | 44 | 42 · +12.04 | 0 · +0.00 | 23 · -20.09 |
| L_SL-0.6 | 44 | 36 · +6.90 | 2 · -0.24 | 17 · -16.58 |
| L_SLatr1.0 | 44 | 12 · +1.48 | 0 · +0.00 | 4 · -4.88 |
| L_NG5_0.05 | 44 | 29 · +13.92 | 0 · +0.00 | 28 · -16.77 |
| L_NG5_0.10 | 44 | 30 · +14.25 | 0 · +0.00 | 40 · -23.85 |
| L_NG10_0.05 | 44 | 13 · +6.18 | 0 · +0.00 | 14 · -7.50 |
| L_NG10_0.10 | 44 | 15 · +6.57 | 0 · +0.00 | 23 · -13.52 |
| L_NG15_0.05 | 44 | 5 · +2.15 | 0 · +0.00 | 13 · -7.10 |
| L_NG15_0.10 | 44 | 6 · +2.74 | 0 · +0.00 | 20 · -10.51 |
| L_TD15_-0.4 | 44 | 21 · +6.11 | 0 · +0.00 | 18 · -13.04 |
| L_TD30_-0.2 | 44 | 12 · +4.57 | 0 · +0.00 | 15 · -9.96 |
| L_TDlin10-40_-0.2 | 44 | 22 · +4.88 | 3 · -0.18 | 18 · -13.15 |
| X_SL-1.0 (post-hoc, wider) | 44 | 10 · +5.64 | 31 · -7.72 | 3 · -1.15 |
| X_SL-1.2 (post-hoc, wider) | 44 | 12 · +11.82 | 31 · -12.30 | 2 · +1.05 |

Artefact guard — winner-side rules on fills that never reached the +0.40 arm and were stopped (Δ must be 0): W_ARM0.6 0/45, W_ARM0.8 0/45, W_ARM1.0 0/45, W_K0.75 0/45, W_K1.25 0/45, W_K1.5 0/45, W_K2.0 0/45, W_LOCK0.0 0/45, W_LOCK0.2 0/45, W_LOCK0.3 0/45, W_LAD_off 0/45, W_LAD_loose 0/45, W_LAD_wide 0/45, W_TW15_k1.5 0/45, W_TW30_k1.5 0/45, W_TW30_k2.0 0/45, W_PW1.0_k1.5 0/45, W_PW1.0_k2.0 0/45, W_HOLD360 0/45, P_half_K2.0_noLAD 0/45, P_half_K3.0_noLAD_h360 0/45, P_half_ARM1.0_K2.0_noLAD 0/45.


Wider-stop guard (post-hoc X_ rules): fills the replica BASE did NOT stop must show Δ = 0 — X_SL-1.0 1 changed, X_SL-1.2 0 changed. ⚠ This guard is REPLICA-based; against the LIVE exit, X_SL-1.2 changes 2 fills live did not stop (+1.05 Σ%), larger than its whole-cohort Δ (+0.004) → that +0.004 is artefact. Read −1.2 on the live-stopped cohort only: net −0.48 (+11.82 saved − 12.30 deeper) — DECIDED no change (the variant_summary CSV's 'KEEP OBSERVING' label for X_SL-1.2 is the mechanical label; this narrative verdict overrides it). The 08-26 watch on its own ATR ≥ 0.7 cut: master N 20 Δ −0.028, yr5 N 322 Δ −0.011 [−0.020, −0.003] — retirement confirmed.


Per era group, before → after (replica BASE avg % → variant avg %, N: before→after) — top variants by Δ:

| variant | G0 BASE | G1 B1 | G2 B2 | G3 B3 | G4 B4-B11 | G5 B12-B16 | G6 B17-B24 | B25 | PROBE |
|---|---|---|---|---|---|---|---|---|---|
| W_LAD_off | 32: +0.53→+0.62 | 15: +0.19→+0.29 | 10: +0.19→+0.19 | 22: +0.11→+0.10 | 14: +0.15→+0.13 | 28: -0.04→-0.05 | 37: -0.13→-0.15 | 9: +0.22→+0.22 | 166: -0.14→-0.15 |
| W_TW30_k1.5 | 32: +0.53→+0.52 | 15: +0.19→+0.37 | 10: +0.19→+0.14 | 22: +0.11→+0.10 | 14: +0.15→+0.14 | 28: -0.04→-0.07 | 37: -0.13→-0.13 | 9: +0.22→+0.30 | 166: -0.14→-0.14 |
| X_SL-1.2 | 32: +0.53→+0.53 | 15: +0.19→+0.14 | 10: +0.19→+0.11 | 22: +0.11→+0.21 | 14: +0.15→+0.16 | 28: -0.04→-0.07 | 37: -0.13→-0.12 | 9: +0.22→+0.17 | 166: -0.14→-0.17 |
| W_LAD_loose | 32: +0.53→+0.52 | 15: +0.19→+0.29 | 10: +0.19→+0.19 | 22: +0.11→+0.13 | 14: +0.15→+0.13 | 28: -0.04→-0.05 | 37: -0.13→-0.15 | 9: +0.22→+0.22 | 166: -0.14→-0.15 |
| W_K1.25 | 32: +0.53→+0.50 | 15: +0.19→+0.37 | 10: +0.19→+0.13 | 22: +0.11→+0.12 | 14: +0.15→+0.13 | 28: -0.04→-0.06 | 37: -0.13→-0.14 | 9: +0.22→+0.43 | 166: -0.14→-0.15 |
| W_TW15_k1.5 | 32: +0.53→+0.54 | 15: +0.19→+0.38 | 10: +0.19→+0.09 | 22: +0.11→+0.09 | 14: +0.15→+0.11 | 28: -0.04→-0.08 | 37: -0.13→-0.14 | 9: +0.22→+0.39 | 166: -0.14→-0.14 |
| W_HOLD360 | 32: +0.53→+0.53 | 15: +0.19→+0.19 | 10: +0.19→+0.19 | 22: +0.11→+0.11 | 14: +0.15→+0.15 | 28: -0.04→-0.04 | 37: -0.13→-0.13 | 9: +0.22→+0.18 | 166: -0.14→-0.14 |
| W_TW30_k2.0 | 32: +0.53→+0.52 | 15: +0.19→+0.35 | 10: +0.19→+0.16 | 22: +0.11→+0.09 | 14: +0.15→+0.12 | 28: -0.04→-0.08 | 37: -0.13→-0.13 | 9: +0.22→+0.30 | 166: -0.14→-0.15 |
| L_NG10_0.10 | 32: +0.53→+0.44 | 15: +0.19→+0.20 | 10: +0.19→-0.03 | 22: +0.11→+0.11 | 14: +0.15→+0.16 | 28: -0.04→-0.16 | 37: -0.13→-0.10 | 9: +0.22→+0.08 | 166: -0.14→-0.14 |
| L_NG15_0.10 | 32: +0.53→+0.47 | 15: +0.19→+0.15 | 10: +0.19→-0.03 | 22: +0.11→+0.07 | 14: +0.15→+0.15 | 28: -0.04→-0.12 | 37: -0.13→-0.13 | 9: +0.22→+0.02 | 166: -0.14→-0.15 |
| W_ARM0.8 | 32: +0.53→+0.41 | 15: +0.19→+0.03 | 10: +0.19→+0.07 | 22: +0.11→+0.23 | 14: +0.15→+0.07 | 28: -0.04→+0.03 | 37: -0.13→-0.14 | 9: +0.22→+0.37 | 166: -0.14→-0.15 |
| W_K1.5 | 32: +0.53→+0.51 | 15: +0.19→+0.34 | 10: +0.19→+0.08 | 22: +0.11→+0.11 | 14: +0.15→+0.11 | 28: -0.04→-0.09 | 37: -0.13→-0.14 | 9: +0.22→+0.51 | 166: -0.14→-0.13 |

Per BATCH (era) Σ Δ % points for the top variants (master; B25 separate):

| variant | BASE (32) | B1 (15) | B2 (10) | B3 (22) | B4 (1) | B5 (3) | B6 (2) | B7 (1) | B8 (4) | B9 (2) | B10 (1) | B12 (15) | B13 (3) | B14 (4) | B15 (1) | B16 (5) | B17 (6) | B18 (3) | B19 (4) | B20 (2) | B21 (3) | B22 (6) | B23 (13) | B25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W_LAD_off | +2.93 | +1.44 | +0.00 | -0.10 | +0.00 | +0.00 | +0.00 | +0.00 | -0.20 | -0.07 | +0.00 | -0.08 | +0.00 | +0.00 | +0.00 | +0.00 | -0.09 | +0.00 | +0.00 | +0.00 | +0.00 | -0.57 | +0.00 | +0.00 |
| W_TW30_k1.5 | -0.26 | +2.70 | -0.52 | -0.25 | +0.00 | -0.22 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | -0.44 | -0.03 | -0.22 | +0.00 | +0.00 | -0.14 | +0.00 | -0.14 | +0.00 | +0.00 | +0.00 | +0.31 | +0.71 |
| X_SL-1.2 | +0.21 | -0.78 | -0.77 | +2.10 | +0.00 | +0.00 | -0.50 | +0.00 | -0.46 | +1.10 | +0.00 | -0.58 | +1.37 | -0.67 | -0.36 | -0.50 | +0.13 | -0.15 | -1.07 | -1.10 | +0.38 | +0.29 | +1.94 | -0.43 |
| W_LAD_loose | -0.35 | +1.44 | +0.00 | +0.43 | +0.00 | +0.00 | +0.00 | +0.00 | -0.20 | -0.07 | +0.00 | -0.08 | +0.00 | +0.00 | +0.00 | +0.00 | -0.09 | +0.00 | +0.00 | +0.00 | +0.00 | -0.57 | +0.00 | +0.00 |
| W_K1.25 | -0.87 | +2.70 | -0.62 | +0.13 | +0.00 | -0.22 | -0.09 | +0.00 | +0.00 | +0.00 | +0.00 | +0.22 | +0.00 | -0.12 | +0.00 | -0.49 | -0.14 | +0.00 | -0.07 | +0.00 | +0.00 | -0.01 | -0.15 | +1.94 |
| W_TW15_k1.5 | +0.36 | +2.82 | -0.97 | -0.42 | +0.00 | -0.46 | -0.09 | +0.00 | +0.00 | +0.00 | +0.00 | -0.32 | -0.17 | -0.22 | +0.00 | -0.24 | -0.14 | +0.00 | -0.14 | +0.00 | +0.00 | +0.00 | +0.10 | +1.52 |

## F. By state — every variant applied ONLY inside one state (Δ = 0 elsewhere), KEEP master

States: BTC regime STRONG_BULL vs rest (entry stamp) · entry ATR % terciles (lo < 0.42 ≤ mid < 0.5 ≤ hi). Same Bonferroni K = 216. Shown: every state read with Δ > 0 and nominal 95 % CI above 0, then the state split of the top-5 whole-cohort variants.

| variant @ state | fills in state | in-state BASE → after | Δ/fill cohort [95 % CI] | Bonf. lo | P(Δ≤0) | groups + | recent Δ | top-day share |
|---|---|---|---|---|---|---|---|---|
| W_LOCK0.2 @ ATR_hi | 54 | +0.084 → +0.109 | +0.009 [+0.005, +0.013] | +0.002 | 0.000 | 5/7 | +0.005 | 9% |

**Gate legs — W_LOCK0.2 @ ATR_hi** (passes the Bonferroni bound): in-state 54·67%·+0.084 → 54·67%·+0.109; Δ per in-state fill +0.025 (haircut +0.013…+0.018); Σ +1.35 % pts ≈ $+10 at today's ticket over 37 in-state trading days (of the 71-day span); changed 15 (up 15 / down 0); per group: G0 BASE +0.038 (8), G1 B1 +0.013 (10), G2 B2 +0.000 (5), G3 B3 +0.035 (8), G4 B4-B11 +0.062 (5), G5 B12-B16 +0.033 (10), G6 B17-B24 +0.000 (8); ex-washed +0.023; top day 9%, top pair 18%; B25 in-state {'count': 4.0, 'mean': 0.037}.
  yr5 backtest in-state: N 704 · Δ -0.0014 [-0.0200, +0.0150] · seeds +0.0093/-0.0065/-0.0076.

State split of the top-5 whole-cohort variants (in-state Δ per fill):

| variant | SBULL | notSBULL | ATR_lo | ATR_mid | ATR_hi |
|---|---|---|---|---|---|
| W_LAD_off | 53: -0.012 | 105: +0.037 | 51: -0.004 | 53: -0.008 | 54: +0.072 |
| W_TW30_k1.5 | 53: -0.017 | 105: +0.016 | 51: -0.012 | 53: -0.016 | 54: +0.041 |
| X_SL-1.2 | 53: -0.065 | 105: +0.038 | 51: -0.036 | 53: +0.051 | 54: -0.005 |
| W_LAD_loose | 53: -0.011 | 105: +0.010 | 51: -0.004 | 53: -0.008 | 54: +0.021 |
| W_K1.25 | 53: -0.005 | 105: +0.005 | 51: -0.010 | 53: -0.026 | 54: +0.040 |

## G. Backtest cross-check — the same grid on the yr5 replay fills (today's config, 3 seeds, ex chop∧burst)

yr5 BASE through this evaluator: 1679·62%·-0.057 (the 10-05 study: 1679·62 %·−0.057 — must match).

| variant | master Δ | yr5 Δ [95 % CI] | yr5 H1 / H2 | seeds | same sign | yr5 significant opposite? |
|---|---|---|---|---|---|---|
| W_LAD_off | +0.021 | +0.006 [-0.005, +0.021] | +0.006 / +0.006 | +0.002/+0.002/+0.016 | yes | no |
| W_TW30_k1.5 | +0.005 | -0.005 [-0.012, +0.005] | -0.011 / +0.001 | -0.005/-0.002/-0.007 | NO | no |
| X_SL-1.2 | +0.004 | +0.013 [-0.018, +0.045] | +0.024 / +0.002 | +0.030/+0.008/-0.002 | yes | no |
| W_LAD_loose | +0.003 | +0.003 [-0.005, +0.013] | -0.001 / +0.007 | +0.003/-0.001/+0.008 | yes | no |
| W_K1.25 | +0.002 | -0.014 [-0.020, -0.005] | -0.010 / -0.017 | -0.009/-0.016/-0.016 | NO | YES |
| W_TW15_k1.5 | +0.001 | -0.013 [-0.022, -0.002] | -0.018 / -0.009 | -0.014/-0.009/-0.015 | NO | YES |
| W_HOLD360 | +0.000 | -0.001 [-0.002, +0.000] | -0.002 / +0.000 | -0.001/-0.001/+0.000 | NO | no |
| W_TW30_k2.0 | -0.002 | -0.005 [-0.015, +0.006] | -0.010 / -0.001 | -0.005/-0.004/-0.007 | yes | no |
| W_LOCK0.0 | -0.004 | -0.005 [-0.011, +0.001] | -0.002 / -0.008 | +0.001/-0.006/-0.011 | yes | no |
| W_LAD_wide | -0.005 | -0.001 [-0.008, +0.008] | -0.006 / +0.004 | +0.002/-0.004/-0.000 | yes | no |
| W_ARM1.0 | -0.006 | -0.027 [-0.070, +0.019] | -0.003 / -0.049 | -0.040/-0.001/-0.041 | yes | no |
| P_half_K2.0_noLAD | -0.007 | -0.011 [-0.023, +0.002] | -0.003 / -0.018 | -0.020/-0.008/-0.004 | yes | no |
| W_LOCK0.2 | -0.008 | +0.002 [-0.007, +0.010] | -0.002 / +0.005 | +0.007/-0.004/+0.002 | NO | no |
| L_NG10_0.05 | -0.008 | -0.018 [-0.041, +0.003] | -0.011 / -0.025 | -0.021/-0.025/-0.009 | yes | no |
| W_K1.5 | -0.011 | -0.017 [-0.028, -0.004] | -0.020 / -0.015 | -0.014/-0.013/-0.025 | yes | no |
| W_PW1.0_k1.5 | -0.012 | -0.002 [-0.009, +0.006] | -0.001 / -0.002 | +0.000/+0.001/-0.007 | yes | no |
| L_NG5_0.05 | -0.018 | -0.045 [-0.076, -0.015] | -0.037 / -0.052 | -0.042/-0.047/-0.046 | yes | no |
| W_ARM0.6 | -0.019 | -0.009 [-0.033, +0.017] | +0.011 / -0.026 | -0.002/-0.001/-0.025 | yes | no |
| X_SL-1.0 | -0.020 | -0.006 [-0.026, +0.015] | +0.003 / -0.014 | -0.004/-0.005/-0.010 | yes | no |
| L_SLatr1.0 | -0.022 | -0.006 [-0.018, +0.006] | -0.006 / -0.005 | -0.008/-0.004/-0.005 | yes | no |
| W_K0.75 | -0.022 | +0.003 [-0.009, +0.014] | -0.012 / +0.016 | +0.001/+0.004/+0.004 | NO | no |
| P_half_ARM1.0_K2.0_noLAD | -0.022 | -0.024 [-0.048, +0.002] | -0.003 / -0.041 | -0.045/-0.013/-0.011 | yes | no |
| W_PW1.0_k2.0 | -0.023 | -0.005 [-0.013, +0.005] | -0.001 / -0.008 | -0.002/-0.003/-0.010 | yes | no |
| W_ARM0.8 | -0.025 | -0.006 [-0.039, +0.030] | +0.028 / -0.035 | +0.006/+0.009/-0.033 | yes | no |
| P_half_K3.0_noLAD_h360 | -0.025 | -0.003 [-0.022, +0.018] | +0.009 / -0.014 | -0.011/+0.003/-0.001 | yes | no |
| L_NG15_0.05 | -0.031 | -0.009 [-0.025, +0.006] | -0.005 / -0.013 | -0.011/-0.015/-0.001 | yes | no |
| L_TD30_-0.2 | -0.034 | -0.003 [-0.023, +0.015] | -0.008 / +0.001 | +0.001/+0.004/-0.015 | yes | no |
| W_K2.0 | -0.034 | -0.016 [-0.030, -0.000] | -0.013 / -0.018 | -0.014/-0.008/-0.026 | yes | no |
| W_LOCK0.3 | -0.040 | +0.000 [-0.015, +0.016] | -0.002 / +0.003 | +0.011/-0.010/-0.001 | NO | no |
| L_TD15_-0.4 | -0.044 | -0.009 [-0.032, +0.013] | -0.013 / -0.006 | -0.002/-0.013/-0.014 | yes | no |
| L_NG10_0.10 | -0.044 | -0.022 [-0.048, +0.003] | -0.004 / -0.037 | -0.023/-0.030/-0.012 | yes | no |
| L_NG15_0.10 | -0.049 | -0.012 [-0.031, +0.006] | -0.007 / -0.017 | -0.012/-0.016/-0.008 | yes | no |
| L_SL-0.5 | -0.051 | -0.024 [-0.053, +0.004] | -0.010 / -0.037 | -0.010/-0.040/-0.024 | yes | no |
| L_TDlin10-40_-0.2 | -0.053 | -0.005 [-0.025, +0.015] | -0.009 / -0.001 | +0.003/+0.002/-0.020 | yes | no |
| L_NG5_0.10 | -0.061 | -0.050 [-0.085, -0.017] | -0.031 / -0.067 | -0.044/-0.055/-0.052 | yes | no |
| L_SL-0.6 | -0.063 | -0.025 [-0.050, -0.002] | -0.016 / -0.033 | -0.023/-0.030/-0.022 | yes | no |
