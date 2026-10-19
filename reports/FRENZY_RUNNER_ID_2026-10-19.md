# FRENZY runner identification at entry, 2026-10-19

**Operator:** "my problem is not only the +1.5 % but not being able to identify that this is a good long, to increase the investment size." Add-on: "what if FRENZY_LONG buys at 70× instead of 100×?"

**Scope:** research only. No code, config or git change. No commit.

**Scripts** (all new, read-only on everything else):
- `scripts/study_frenzy_runner_id_build.py`: cohort, path outcome, exits and HTF features.
- `scripts/study_frenzy_runner_id.py`: coverage, screen, null, OOS, model AUC and sizing.
- `scripts/study_frenzy_runner_id_followup.py`: survivors read in day units and priced as sizing rules.
- `scripts/study_frenzy_runner_id_fetch.py`: public 5m/1m klines for the real fills (the caches end 2026-10-12). About 90 requests, used weight checked, no 418/429.
- `scripts/study_frenzy_runner_id_volwalk.py` / `_volprice.py` / `_volreport.py` / `_agt.py`: the volume-threshold walk (§9).

`scripts/validate_against_master.py`: **ALL CHECKS PASS.**

## Answer in plain English

1. **FRENZY_LONG: a runner cannot be identified at entry.**
   - About 36 % of LONG entries run to +5 % before touching −3 within 12 h. This rate is the same in 2025, 2026-H1, 2026-H2 and the real fills (31 %).
   - Nothing available at the entry bar separates those runners from the rest better than luck:
     - 1D screen: familywise p 0.98.
     - 2D screen: p 0.88, with hit counts *below* the null.
     - A model trained on all ~94 features reaches AUC 0.81 in-sample (2025). Out of sample it falls to 0.55 / 0.54 in 2026, and 0.39 on the real fills.
     - 0 of 8 pre-written theories survive. That includes the AGT-type "late-ON multi-day trend": real 6 fills, 17 % runners.
   - **There is no LONG cohort that justifies more size.**
2. **The +1.5 TP does cap winners, but no runner exit fixes it.** Half of the +1.5 winners would have reached +5. The catch: 85 % of LONG entries also touch −3 at some point within 12 h. Runners spike and come back.
   - Half at +1.5 with the rest trailing 3 points: −0.11 %/fill, vs −0.10 for today's exit.
   - A pure trail: −0.14.
   - On the 29 real LONG fills: today's exit Σ +3.0 pts, half/trail −3.2, pure trail −13.1. AGT 10-19 03:45 peaked +44 %, but the trail exits got only +0.9 / +0.4: it dipped 3 points after +1.5 before running.
3. **The only things that do separate runners are BTC-market conditions, and only for FRENZY_LITE.**
   - On days when BTC's 72 h return is positive, LITE runs more often (38 / 43 / 41 % vs 28 / 36 / 33 % in 2025 / 26-H1 / 26-H2) and averages more at today's exit.
   - It is a market-wide switch, not a trade signal, and it **fails the locked Pattern-W bar**: win rate 56–64 % vs the required 70 %.
   - Doubling LITE on those days earns ≈ $0 in 2025 and doubles the worst day.
   - **Recommendation: an observe-only line, not a size change** (§10).
4. **Buying at 70× instead of 100× does not help.**
   - The whole sleeve at 70× vs 100×: +9 pts over 21 months, 95 % CI [−132, +138].
   - Episodes that never reach 100× lose (50×: −0.27 %/fill, CI below 0).
   - Where the episode does reach 100× later, the 70× bar buys ≈ 5 h earlier and 5.7 % cheaper, but the outcome is identical: +0.31 vs +0.31 %/fill.
   - On AGT itself, a 70× bar would not have bought 10-19 at all.

## 0. What is NEW here vs prior work

| prior study | what it did | what this study adds |
|---|---|---|
| FRENZY_WL_SCREEN 2026-10-19 (today) | W/L at the +1.5 outcome (caps winners); no separator; LONG = keep blocker | **path outcome** (MFE before −3, 2/6/12 h, +3/+5/+8); **multi-day HTF features rebuilt from klines** (1h/4h EMA alignment, 1h/4h RSI, 24/72 h/7 d return, hours since 7 d low, up-day streak, prior-day close position, volume vs 7 d); BTC 4h trend; OOS **model AUC**; sizing priced at 2×/3× with runner exits |
| FRENZY_EXIT_SELECTOR 10-05 / RIDE_CAPTURE 10-06 / SIGNATURE_RUNNER 10-05 | ride = peak ≥ +15 % before −3, FRENZY/WIDE first entries on the old lock exit; selector at entry or at the arm; 0 passes | **lower runner bar (+5, with +3/+8)** that today's +1.5 actually leaves on the table; **LITE and WILLY** included; yr6 2-year replay (2025 discovery, 2026 H1/H2 confirmation) + real fills; question = **size**, not exit selection |
| RUNAWAY / LIFTOFF sleeves (Oct-1) | new sleeves on pump starts; 0/128 cells | studies entries the sleeves **already take** |
| STAIRCASE 10-06 | the staircase at 50× ("enter when it turns on" −0.38…−0.49) | re-walks at **50 / 70 / 85 / 100×** with today's rules (ATR 3.0, bearish block), splits episodes that **later reach 100×** vs never, and prices the **paired earlier entry** (§9) |
| YR6_WL_FRENZY | W/L per sleeve × regime at today's exits | runner-specific outcome; HTF features not in that screen |

## 1. Data, three blocks, parity

| block | source | LONG | LITE | WILLY |
|---|---|---|---|---|
| ③ backtest (BT) | `ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv` seed 1, opened < 2026-10-03; 2025 = discovery, 26-H1 / 26-H2 = confirmation | 603 | 1,427 | 1,482 |
| ② master | `MASTER_POOL_stacked.csv` kept, non-probe, CLOSED (B16–B22) | 26 | 24 | 27 |
| ① live | `scalpars_orders_live_2026-10-19_14-54-01.csv` (B23) | 3 | 4 | 4 |

- **Paths:** ticks first (reports/backtest_cache + backtest_cache_2025 ticks_q), then 1m. Real fills use the fetched 1m tail.
  - Stop-first: a 1m bar touching both −3 and the high is a stop, and its high is not counted.
  - Net = gross − 0.045 − 0.045·r, the bot's paper accounting.
  - 4 of 3,595 BT fills could not be priced.
- **Parity** (`_parity.csv`):

| check | result |
|---|---|
| walker +1.5/−3 vs replay `pct_new`, BT LONG | 92.5 % within 0.05, 99.2 % same sign |
| walker +3/−3 vs `pct_new`, BT LITE | 79 % within 0.05 (fill-slip differences), 97.8 % same sign |
| walker today's exit vs booked / stack_pct, real fills | LONG 93 %, LITE 93 %, WILLY 100 % within 0.1; 100 % same sign |
| rebuilt BTC 72 h return vs the live stamp `entry_btc_r72_pct` | r = 0.999 on every block, so the clock and alignment are right |
| volume walk 100× READY bars vs yr6 FRENZY_LONG fills | 93 % of fills on a READY bar; 94 % same-sign outcome; median entry gap 0.13 % |
| AGT live 10-18 00:30 / 10-19 03:45 / 10-19 14:10 vs the walk | same bars; 102.3 / 136.2 / 101.8× vs stamps 102.4 / 136.1 / 101.8×; vs-VWAP 41.0 / 7.8 / 27.8 equal |

**Note:** `entry_btc_1d_ret_pct` (the bearish-day stamp) is the last closed **UTC daily candle**, not a rolling 24 h. My rolling `b_ret24` only correlates r 0.51 with it, so the two are different variables and both were tested.

## 2. Theory (written before the screen; zone predicted to hold MORE runners unless marked)

| id | hypothesis | mechanism |
|---|---|---|
| T1 | late-ON multi-day trend: hours since spike > 24 ∧ 4h EMA20 > EMA50 ∧ price above | AGT 10-19 type: a coin re-accumulating for days is supported by higher-TF buyers |
| T2 | 1h and 4h EMA stacks aligned | higher-TF trend carries the 5m setup |
| T3 | liquid mid-cap ($100M–$1B mcap ∧ vol24 ≥ $100M) | enough depth for a sustained flow, not a one-wallet pump |
| T4 | volume multiple ≥ 200× | stronger participation |
| T5 | BTC tailwind (1h slope > 0 ∧ BTC 4h EMA20 > EMA50) | risk-on market lifts high-beta alts |
| T6 (fewer) | fresh pump: 7-day low ≤ 24 h ago | Oct-1 lesson: fresh pumps mean-revert |
| T7 | multi-day staircase: ≥ 2 consecutive up days | persistence |
| T8 (fewer) | stretched ≥ 13 % above spike VWAP | exhaustion (the STRETCHED line's frozen cut) |

## 3. Base rates (`_base_rates.csv`)

"Run5" = MFE ≥ +5 before −3 within 12 h.

| sleeve | block | N | run3 | **run5** | run8 | run5 in 2 h | touches −3 within 12 h | share of today's winners that ran ≥ +5 | today's exit | half +1.5 / trail | pure trail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LONG | 2025 | 269 | 50 % | **35 %** | 25 % | 26 % | 84 % | 53 % | −0.05 | −0.10 | −0.10 |
| LONG | 26-H1 | 230 | 48 % | **37 %** | 25 % | 33 % | 87 % | 59 % | −0.12 | −0.11 | −0.11 |
| LONG | 26-H2 | 104 | 43 % | **36 %** | 27 % | 33 % | 86 % | 57 % | −0.19 | −0.15 | −0.32 |
| LONG | REAL | 29 | 38 % | **31 %** | 17 % | 31 % | 86 % | 45 % | +0.10 | −0.11 | −0.45 |
| LITE | BT all | 1,427 | 49 % | **37 %** | 23 % | 24 % | 79 % | 64 % | +0.07 | +0.02 | −0.08 |
| LITE | REAL | 28 | 54 % | **43 %** | 25 % | 32 % | 79 % | 71 % | +0.36 | +0.06 | +0.67 |
| WILLY | BT all | 1,482 | 49 % | **37 %** | 26 % | 31 % | 79 % | 49 % | −0.12 | −0.11 | +0.01 |
| WILLY | REAL | 31 | 42 % | **32 %** | 16 % | 32 % | 84 % | 40 % | +0.11 | −0.24 | −0.74 |

**Reading:**
- Runners are common: about 1 in 3 entries in every sleeve, block and period. Median time to the 12 h MFE is 2–4 h.
- Most entries still touch −3 within 12 h. The typical runner is a spike that gives back.
- That is why both runner exits (half/trail, pure trail) are no better than today's fixed exit on the sleeves as a whole.

## 4. Screen vs day-block null (`_null.csv`, `_screen_top.csv`)

**Setup:**
- Discovery is on BT 2025.
- 1D: every feature with ≥ 50 % coverage × {sign, median, low tercile, high tercile}.
- 2D: exhaustive median quadrants over features with ≥ 70 % coverage.
- Statistic: two-proportion z on run5, runner-rich side.
- Null: day-block, 400 draws (1D) / 100 draws (2D).
- Coverage: `_coverage.csv`, 120 features.
  - 100 % on all h_*/b_* HTF rebuilds and every entry_frenzy_* stamp.
  - mcap / turnover / cmc rank: 78–85 % BT, 100 % real.
  - `entry_pair_rank` and `entry_pair_1d_ndi` are 0 % (unstamped), so they were not testable.

| sleeve | layer | tests | max z obs | null 95 % | familywise p | hits obs (z ≥ 3 / 3.5) | null mean / 95 % | hits p |
|---|---|---|---|---|---|---|---|---|
| **LONG** | 1D | 319 | 2.14 | 3.89 | **0.98** | 0 | 1.1 / 4 | 1.00 |
| **LONG** | 2D | 17,484 | 3.52 | 4.83 | **0.88** | 1 | 12.8 / 42 | 0.89 |
| LITE | 1D | 319 | 4.03 | 4.35 | 0.11 | 6 | 1.8 / 7 | 0.09 |
| LITE | 2D | 15,664 | 4.95 | 5.32 | 0.12 | 48 | 15.8 / 53 | 0.07 |
| WILLY | 1D | 312 | 3.00 | 4.39 | 0.61 | 1 | 2.0 / 8 | 0.61 |
| WILLY | 2D | 16,744 | 4.14 | 5.24 | 0.56 | 4 | 22 / 76 | 0.85 |
| pooled | 1D | 320 | 3.69 | 4.65 | 0.24 | 17 | 2.2 / 8 | **0.02** |
| pooled | 2D | 16,380 | 4.81 | 5.23 | 0.21 | 71 | 24 / 71 | 0.06 |

- **LONG has no signal at all.**
- LITE and the pooled set show an excess of hits. **Every top LITE / pooled cell is a BTC-market variable**: BTC 72 h return, BTC 4h RSI, BTC 4h EMA gap, BTC off 30 d high, BTC 24 h return.

## 5. Out of sample (`_oos.csv`, `_model_auc.csv`)

**Survivor rule (frozen):** run5 is higher in **both** 2026 halves at the frozen 2025 cut ∧ pooled-2026 day-bootstrap P ≥ 0.95. Each sleeve was tested on the top 15 1D + top 15 2D cells + the 8 theory cells.

| sleeve | survivors / 38 | what survives |
|---|---|---|
| **LONG** | **0** | none. 8 of the 30 screen cells even keep the sign in both halves, which is about chance |
| LITE | 7 | all **BTC market state**: BTC 24 h > 0 / > +1.2 %, BTC 72 h > +1.07 % (alone and paired with "pair not over-extended on 1h"), T5 BTC tailwind |
| WILLY | 1 | pair 4h EMA20 > EMA50 |
| pooled | 1 | T8 stretched ≥ 13 % → **fewer** runners (2026: 31 % / 17 % vs 40 % / 38 %) |

**Theory on FRENZY_LONG** (run5 zone vs rest; 2025 → 26-H1 → 26-H2 → real):

| theory | 2025 | 26-H1 | 26-H2 | real | P(2026) |
|---|---|---|---|---|---|
| T1 late-ON multi-day (AGT type) | 38 vs 33 | 38 vs 37 | 39 vs 35 | **17 vs 35** (6 fills) | 0.6 |
| T2 1h ∧ 4h aligned | 37 vs 18 | 38 vs 34 | **35 vs 43** | 31 vs 33 | 0.6 |
| T3 liquid mid-cap | 30 vs 39 | 42 vs 35 | 17 vs 35 | 43 vs 20 | 0.5 |
| T4 volume ≥ 200× | 46 vs 33 | 41 vs 36 | 38 vs 35 | 25 vs 35 | 0.8 |
| T5 BTC tailwind | 34 vs 35 | 37 vs 37 | 37 vs 34 | 20 vs 33 | 0.5 |
| T6 fresh pump (fewer) | 33 vs 35 | 50 vs 37 | 0 vs 37 (n 4) | – | 0.7 |
| T7 multi-day staircase | 36 vs 34 | 40 vs 36 | 33 vs 38 | 35 vs 25 | 0.6 |
| T8 stretched (fewer) | 41 vs 34 | 31 vs 38 | 17 vs 38 | 33 vs 31 | 0.9 |

**Whole-feature model** (ridge logistic, trained on 2025, all ≥ 70 %-covered features):

| sleeve | AUC 2025 (in-sample) | 26-H1 | 26-H2 | real | run5 in top 20 % of scores, 2026 H1 / H2 (base) |
|---|---|---|---|---|---|
| **LONG** | 0.81 | **0.55** | **0.54** | **0.39** | 38 / 38 % (37 / 36) |
| LITE | 0.70 | 0.47 | 0.53 | 0.47 | 37 / 39 % (40 / 38) |
| WILLY | 0.69 | 0.55 | 0.49 | 0.58 | 43 / 37 % (41 / 36) |

**The model memorises 2025 and predicts nothing in 2026.** The top-scored fifth of 2026 entries runs at the base rate.

## 6. Survivors read in day units, priced as sizing rules (`_followup.csv`, `_followup_pricing.csv`)

These are market-wide variables, so the WINDOW-UNITS rule applies: each is a sleeve on/off/size switch, counted in days.

### 6a. FRENZY_LITE when BTC's 72 h return > 0 (stamped `entry_btc_r72_pct`, sign granularity: the cleanest form of the survivors)

| block | zone N · days | run5 zone vs rest | today's exit zone vs rest | Δ day-boot P | zone WR | top day / pair share of gain |
|---|---|---|---|---|---|---|
| 2025 | 540 · 184 d | 38 vs 28 % | **+0.02 vs −0.22** | 0.90 | 56 % | 4 % / 4 % |
| 26-H1 | 213 · 86 d | 43 vs 36 % | +0.37 vs +0.06 | 0.88 | 63 % | 6 % / 4 % |
| 26-H2 | 153 · 56 d | 41 vs 33 % | +0.44 vs −0.04 | 0.88 | 64 % | 10 % / 7 % |
| 2026 | 366 · 142 d | 42 vs 35 % | **+0.40 [+0.12, +0.67] vs +0.03** | 0.95 | 64 % | 5 % / 4 % |
| REAL | 10 · 5 d | 50 vs 39 % | +0.50 vs +0.28 | 0.53 | 60 % | 64 % / 18 % |

**Priced as a size-up of the zone** (LITE notional $2,000 ≈ today's 8×; every BT fill counted, no slot cap):

| period | rule | Δ $ vs today | worst day (today) | max drawdown (today) |
|---|---|---|---|---|
| 2025 | zone 2× | **+$176** | −$720 (−$360) | −$3,124 (−$2,317) |
| 2025 | zone 3× | +$353 | −$1,080 | −$4,128 |
| 2026 (Jan–Oct 2) | zone 2× | +$2,927 | −$360 (−$300) | −$793 (−$608) |
| 2026 | zone 3× | +$5,854 | −$540 | −$1,049 |

**Pattern-W bar (N ≥ 30 ∧ WR ≥ 70 ∧ avg ≥ +0.10 ∧ Σ > 0) — FAILS.**
- At today's exit the WR is 56 % / 64 %.
- A +2/+3 vs −3 exit cannot reach 70 % WR, and the bar does not move.
- With the half/trail exit in the zone, WR is 67 % / 71 % and avg +0.01 / +0.21. 2025 fails.
- BE-compatibility is fine: 81 % of zone losers peak ≥ +0.20.

**30–50 % haircut on 2026:**
- Zone avg +0.20 … +0.28 %/fill.
- 2025 says the zone itself is worth ≈ 0. Its edge there is that the *rest* is bad (−0.22), which is a filter story, not a size story.

**Risk at today's $1,225 account:**
- A 2× LITE stop is ≈ −$120, about 10 % of the book.
- The 2025 worst day doubles to −$720 at $2,000 notional, ≈ 59 % of today's book if every fill had traded.
- That is the ruin constraint (memory: scaling roadmap).

**Overlap:**
- LITE C already gives LITE its +3 TP when BTC 1h slope > 0 (same regime family).
- The existing observe line LITE_BTC1H_FALLING (DL 288) watches the mirror side.
- The real fills contradict the rolling-24 h form: 20 zone fills, −0.15 vs +1.63. The stamped 72 h form agrees: +0.50 vs +0.28, 10 fills on 5 days.

### 6b. FRENZY_WILLY when the pair's 4h EMA20 > EMA50

- Runners: 37 / 45 / 36 % vs 29 / 29 / 33 %.
- Today's exit: zone 2025 **−0.15**, 2026 +0.13, real −0.13 (25 fills) vs rest +1.08.
- 2× of the zone: 2025 −$2,236, 2026 +$1,383. **No.**
- The zone carries more runners but WILLY's +1.25 / 90-min exit never sees them. And 2025 loses at 2×.

### 6c. FRENZY_LONG, the closest cells (none survive)

**Volume multiple > 122.6× (2025 high tercile):**
- Runners are higher in every period: 42 / 39 / 43 % vs 31 / 36 / 31 %, BT-all P 0.985.
- But today's exit in 2026 is **−0.13** (vs −0.16 rest), and real is 31 vs 31 %.
- At 2× of $2,000 (sized up from blocker) 2026 = −$390.
- Only a pure trail on this zone is positive (+0.53 / +0.13 per period; real −0.93). That is an exit-selector idea, already refuted family-wide (EXIT_SELECTOR 10-05). **No.**

**BTC 24 h > 0:** LONG 2026 −0.17 vs −0.09. **No.**

## 7. Pricing summary at the locked bars

| candidate | Pattern-W | expectancy / CI | consistency | haircut | verdict |
|---|---|---|---|---|---|
| LONG, any cell / theory | none pass in 2025 + both 2026 halves | sleeve −0.10 %/fill BT | – | – | **no size** |
| LITE ∧ BTC 72 h > 0 | **fails** (WR 56–64 %) | 2026 +0.40 [+0.12, +0.67]; 2025 +0.02 | runner lift 3/3 periods, $ lift 2026 only | +0.20 … +0.28 | **observe-only** |
| WILLY ∧ 4h up | passes 2026 only (79 % · +0.13) | 2025 −0.15 | no | – | no |
| T8 stretched (pooled) | – | fewer runners | 2026 only | – | descriptive: AGT 10-18 (+41 % vs VWAP) stopped |

## 8. AGT: what the entry knew (`_agt_case.csv`, `_real_fills.csv`)

| fill | hours since spike | vol × | vs VWAP | 24 h / 72 h return | 4h aligned | MFE before −3 (12 h) | today's exit | half/trail | pure trail |
|---|---|---|---|---|---|---|---|---|---|
| 10-16 03:40 | 12.4 | 101 | +4.4 | +106 / +114 % | yes | +8.0 | +1.5 | +3.3 | +5.0 |
| 10-18 00:30 | 57.3 | 102 | **+41.0** | +39 / +397 % | yes | **−0.1 (stopped)** | −3.0 | −3.0 | −3.0 |
| 10-19 03:45 | 84.5 | 136 | +7.8 | −10 / +96 % | yes | **+44.2** | +1.5 (live +1.27) | +0.9 | +0.4 |
| 10-19 14:10 | 94.9 | 102 | +27.8 | +14 / +113 % | yes | +4.4 | +1.5 | +2.3 | +3.1 |

**Reading:**
- The three "good" AGT entries and the stopped one look **the same at entry**: late-ON, 4h-aligned, ~100×, multi-day uptrend.
- The stopped one was the most stretched (+41 % vs VWAP), which matches T8.
- The +44 % runner was **not** catchable by a mechanical trail: it dipped > 3 points after +1.5 first.
- AGT is 2 of the 9 real LONG runners. It does not carry any cohort above.

## 9. Volume threshold (operator)

**Construction** (`_volwalk.py`, frozen in its docstring):
- The real `services.frenzy.frenzy_walk` / `frenzy_long_status`, today's thresholds, only `frenzy_state_vol_mult` changed (50 / 70 / 85 / 100).
- Run over every pair of the 2025 + 2026 5m caches (2025-01 → 2026-10-12).
- FRENZY_READY fresh-ON bars, entry at close + 8 s, 1 % dislocation guard, approximate bearish-day block, priced on the same paths and exits as §3.

**Volume as a first-class feature:** the runner screen included `entry_frenzy_vol_mult` (§6c). A higher multiple means slightly MORE runners. Nothing says lower is better.

**READY bars over 21 months:** 50× 1,682 · 70× 1,440 · 85× 1,291 · 100× 1,154.

### 9a. The first lower-threshold ON bar of each episode, by what 100× does later (`_volume_first_on.csv`)

| T | class | share | N | run5 | today's exit avg [CI] | half/trail | pure trail |
|---|---|---|---|---|---|---|---|
| 70× | SAME bar as 100× (no change) | 36 % | 259 | 41 % | −0.05 | −0.01 | +0.07 |
| 70× | 100× comes LATER | 19 % | 136 | 46 % | +0.31 [−0.03, +0.63] | +0.26 | +0.56 |
| 70× | NEVER reaches 100× (new trade) | 44 % | 315 | 30 % | **−0.17** [−0.41, +0.06] | −0.18 | −0.26 |
| 50× | NEVER reaches 100× | 58 % | 476 | 30 % | **−0.27 [−0.46, −0.08]** | −0.34 | −0.45 |
| 85× | NEVER reaches 100× | 32 % | 201 | 33 % | +0.05 [−0.26, +0.34] (H1 +0.43, H2 −0.10) | −0.03 | +0.09 |

**Never-100× episodes, today's exit by period:**

| T | 2025 | 26-H1 | 26-H2 |
|---|---|---|---|
| 70× | −0.13 | −0.17 | −0.30 |
| 50× | −0.16 | −0.26 | −0.61 |

### 9b. Same episode, earlier entry vs waiting for 100× (paired, LATER class; `_volume_paired.csv`)

| T | N · days | buys earlier by (median) | buys cheaper by (median) | run5 at T vs at 100× | today's exit at T vs 100×: Δ [CI] | half/trail Δ |
|---|---|---|---|---|---|---|
| 50× | 131 · 112 | 9.0 h | 9.7 % | 56 vs 43 % | +0.26 vs +0.44: **−0.17 [−0.69, +0.35]** | +0.18 |
| **70×** | 136 · 115 | 5.3 h | 5.7 % | 46 vs 44 % | +0.31 vs +0.31: **0.00 [−0.42, +0.42]** | −0.05 |
| 85× | 131 · 115 | 0.8 h | 0.4 % | 44 vs 40 % | +0.13 vs +0.13: 0.00 [−0.43, +0.43] | −0.09 |

- The 70× Δ by period is +0.14 / −0.43 / +0.30 in 2025 / 26-H1 / 26-H2. There is no consistent sign.
- **Cheaper is not better under a fixed +1.5 / −3:** the earlier bar sits lower in the same noisy staircase.
- The LATER class is the good class (+0.31, 46 % runners). But "this episode will reach 100× later" is hindsight, so not an entry feature.

### 9c. The sleeve run at T instead of 100× (every READY bar, ≤ 3 per pair-day; `_volume_policy.csv`)

| T | N | avg | run5 | Σ pts | Δ Σ vs 100× [95 % day CI] | P(Δ > 0) |
|---|---|---|---|---|---|---|
| 100× (today) | 763 | −0.05 | 37 % | −39 | – | – |
| 85× | 834 | +0.00 | 38 % | +3 | **+42 [−78, +158]** | 0.76 |
| 70× | 956 | −0.03 | 36 % | −30 | **+9 [−132, +138]** | 0.54 |
| 50× | 1,138 | −0.13 | 36 % | −153 | **−114 [−270, +38]** | 0.07 |

- With bearish days included, 50× is −157 (P 0.04) and 70× is −37.
- 85× by period is +2 / +39 / +2.

**AGT 10-19 at 70×** (`_agt_case.csv`): AGT's 70× state was already ON from 10-19 00:25 (a green bar, refused). So at 70×:
- the 03:45 entry is the same as today;
- the 14:10 bar is **not** a fresh ON (no 1 h off), so the 14:10 +1.5 trade would not have happened.

At 85×, the walk adds 10-17 22:30 (READY, ran +40 % before the stop) and 10-18 14:15 (+1.5). It loses no AGT trade. That is one coin, an anecdote. The year says 85× ≈ +42 pts over 21 months, not significant.

**Do earlier entry and runner identification combine into a sizing rule? No.**
- The earlier entry does not raise the runner rate or the per-fill result in the episodes where it differs.
- The new episodes it adds run less often (30 % vs 37 %) and lose.
- Runner identification has no LONG signal to combine with.
- **Keep `frenzy_state_vol_mult` at 100.** 85× is the only non-negative variant: +42 pts, CI spans 0, at blocker size ≈ +$42 over 21 months. Not worth a change.

## 10. Blind spots

- **Post-entry information excluded by design.** The first-minutes path (the RIDE_CAPTURE "arm" features) was screened 10-06 and also failed.
- **Unstamped / untestable:**
  - `entry_pair_rank`, `entry_pair_1d_ndi`: 0 %.
  - order book (`entry_book_*`): live-only, ≈ 23 %.
  - funding history.
  - mcap: 78–85 % in BT, so unscored fills were left out of zone and rest.
- **Concurrency is not modelled in the $ pricing.** Every BT fill is counted at the sleeve notional; live has ≤ 2 LITE open and the WILLY global hold. The $ rows are an upper bound on both the gain and the worst day.
- **Exit mismatches:**
  - The LITE C TP (+2 / +3) is walked per fill from its stamps.
  - WILLY includes the late TP but not the resting-maker fee.
  - LONG's maker TP fee (DL 310) is not credited (+0.027 per TP fill).
- **Volume walk approximations:**
  - The bearish-day block uses rolling 24 h + BTC 5m EMA13 < EMA50 (the stamp is the last UTC daily candle).
  - Not modelled: slots, the shortlist reach, the pair-held-by-LITE refusals, WILLY's global hold.
- **Real fills are few:** 29 / 28 / 31 on 9–13 days. They can only refute.
- **Seeds 2 / 3 not re-walked on paths.** The prior screen showed seed agreement within 0.01 %/fill on sleeve means.
- **The HTF features are new, not stamped.** A live rule would need new `entry_*` stamps (D11).

## 11. Recommendation

| question | answer |
|---|---|
| **FRENZY_LONG: can a runner be identified at entry to size up?** | **No.** 0 survivors, OOS AUC ≈ 0.54, real AUC 0.39, no Pattern-W cell. **Keep blocker size; no multiplier.** Any LONG size-up would be sizing a coin flip with a −0.10 %/fill sleeve mean. |
| **A runner-friendly exit instead?** | **No.** Half/trail and pure trail are ≤ today's exit on every LONG block (real: +3.0 vs −3.2 vs −13.1 pts). Runners give back 3 points before running. |
| **FRENZY_LITE** | **Observe-only line candidate: LITE_BTC72_UP (operator decides; nothing armed).** Zone = FRENZY_LITE fill with `entry_btc_r72_pct` > 0 (stamped, 100 % coverage). 🔒 Frozen gate below. |
| **FRENZY_WILLY** | Nothing. The 4h-trend zone carries more runners that WILLY's 90-min exit cannot use, and it loses in 2025. |
| **Volume threshold 70×** | **Keep 100×.** 70× ≈ 0 (+9 pts [−132, +138]); 50× worse; never-100 episodes lose; the earlier/cheaper entry gains nothing at today's exit. |

**🔒 LITE_BTC72_UP frozen gate** (window units; cohort = LITE fills opened after this report):
- **Read at:** zone ≥ 30 fills on ≥ 15 UTC days.
- **SIZE-UP STUDY candidate only if all of:**
  - ① the locked Pattern-W bar on the zone at LITE's live exit (N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 ∧ Σ > 0);
  - ② zone − rest ≥ +0.30 %/fill at day-bootstrap P ≥ 0.90;
  - ③ no day ≥ 50 % of the zone gain;
  - ④ a 2× zone worst day ≤ 25 % of the book at the then account size.
- **RETIRE** if zone avg ≤ rest avg at the read.
- Never re-fit the sign cut.
- **Honest prior:** ① is structurally hard for a +2/+3 vs −3 exit (BT zone WR 56–64 %). This line most likely ends as a confirmed regime *description*, not a multiplier.

**30–50 % haircut:** applied in §6a. After it, the only positive sizing projection is LITE 2026: +0.20 … +0.28 %/zone fill. 2025 gives ≈ 0.

## CSVs (`reports/FRENZY_RUNNER_ID_2026-10-19_*`)

- `FRENZY_RUNNER_ID_2026-10-19.csv`: per fill, all blocks: outcomes, exits and features.
- `base_rates`
- `parity`
- `coverage`
- `screen_top`
- `null`
- `oos`
- `model_auc`
- `sizing`
- `followup`
- `followup_pricing`
- `real_fills`
- `theory_real`
- `survivors.json`
- `agt_case`
- `volume_first_on`
- `volume_paired`
- `volume_paired_rows`
- `volume_policy`
- `volume_parity`
