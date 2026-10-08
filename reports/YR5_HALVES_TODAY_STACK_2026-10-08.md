# yr5 backtest by strategy — first half vs second half, at TODAY's stack (2026-10-08)

**Question (operator):** "Show me again the backtest results by strategy, first half and second half of the year, with all current filters applied."

**Split:** H1 = 2026-01-04 00:00 → 2026-05-20 00:00 UTC (136 days) · H2 = 2026-05-20 → 2026-10-04 00:00 UTC (137 days), by trade open time.
**Money ruler:** each strategy on its own **flat $3,000 book** (no compounding), today's sizing: $ = 3,000 × 24.375 % (4 slots, 2.5 % fee reserve) × invest × leverage × P&L %. Same ruler as the Oct-6 table. **Avg P&L % is per trade and leverage-invariant** — compare strategies and halves on that column, not on $.
**CI:** day-clustered bootstrap 95 % of the avg % (4,000 draws; the 3 seeds pooled, a day = one cluster).

**Validation first (as required):** `scripts/validate_against_master.py` → **ALL CHECKS PASS** (M1/M2, F1/F2, C1, A1 ×3, Y1, X1 ×3, PS1 ×3, CB1). Harness parity: re-running this script in Oct-6 mode reproduces the Oct-6 table's Net $ for all 10 sleeves to the dollar (MOM_LONG −7,740 … SURGE_SHORT −3,204).

## Plain-English answer

- **At today's settings, the replayed engine sleeves still lose in the first half and make money in the second half.** H1 −$7,320 → H2 +$12,111 for all live sleeves without WILLY (+$4,791 for the year, eleven separate $3k books). Take away BULLRUN, which is one 6-day event in August, and it is −$7,320 → +$6,614 (−$706 for the year).
- **MOM_LONG is still the main drain, in both halves:** −0.086 %/trade in H1 (CI below zero), −0.027 in H2 (CI spans zero), −$6,433 for the year. Today's filters only trim it (−$7,740 → −$6,433).
- **FLIP_SHORT and MOM_SHORT lose in both halves.** FLIP's dollar loss halves only because FAN flips now trade at 10× instead of 20×; its per-trade result is unchanged.
- **The FRENZY family is where today's changes show up**, but read it with care:
  - **FRENZY_LONG:** + in both halves in $ (+$506 / +$1,363), but only −0.02 % / +0.19 % per trade, with CIs far across zero.
  - **The switch back to a fixed +3 % take-profit costs FRENZY_LONG most of its year on this replay:** +0.174 → +0.011 %/trade on the same fills. 45 % of the fills reached +4 and now book +3; only 5 % peaked between +3 and +4 and are rescued.
  - **FRENZY_WIDE** flips from −$1,604 to +$895 because of the hold-green rule. That rule was chosen on this same year (in-sample).
- **FRENZY_LITE** (a study cohort, not the engine replay) is the steadiest line: +0.27 % / +0.29 % per trade in the two halves, +$6,678 for the year. Its CI touches zero, it fails the locked bars, and it was designed on this same year.
- **FRENZY_WILLY was never backtested as shipped.** It is re-priced here for the first time, under its exact current design, on ticks:
  - **Paper (no stop):** +0.037 % (H1) / +0.074 % (H2), 84 % WR, +$9,874 for the year. The CI spans zero.
  - Trigger A alone: +0.112. Trigger B alone: −0.268.
  - **With the live exchange backstop (≈ −2.2 %) acting as a stop, it turns negative:** −0.105 %/trade for the year, CI below zero (−$19,469).
  - On paper, the worst trade was −25.8 %, which is −$3,776, more than the whole $3k book. 192 of 1,263 trades went through the 20× liquidation zone (≤ −4.5 %) before recovering or hitting the 120-min cap.
- **Bottom line:** no sleeve has a CI above zero in both halves. The year's positive total rests on one BULLRUN week, on LITE (an in-sample study cohort) and, if counted, on WILLY in paper mode only.

## Main table (mean of 3 seeds for replay sleeves; study cohorts are one realisation)

| Strategy | Source | H1 N · WR · avg % [95 % CI] · Net $ | H2 N · WR · avg % [95 % CI] · Net $ | Full year N · WR · avg % [95 % CI] · Net $ |
|---|---|---|---|---|
| MOM_LONG | replay + heat re-admits ᵃ | 317 · 59.6 % · **−0.086** [−0.16, −0.01] · −$5,903 | 280 · 65.2 % · **−0.027** [−0.09, +0.04] · −$530 | 597 · 62.2 % · **−0.058** [−0.11, −0.00] · **−$6,433** |
| MOM_SHORT | replay | 103 · 56.8 % · **−0.088** [−0.22, +0.04] · −$1,328 | 58 · 64.0 % · **−0.023** [−0.13, +0.08] · −$199 | 162 · 59.4 % · **−0.064** [−0.15, +0.02] · **−$1,527** |
| FLIP_SHORT | replay | 72 · 58.8 % · **−0.177** [−0.35, −0.01] · −$909 | 65 · 65.1 % · **−0.091** [−0.21, +0.04] · −$471 | 137 · 61.8 % · **−0.137** [−0.24, −0.03] · **−$1,380** |
| SPIKE_FADE | replay ᵇ | 156 · 72.0 % · **−0.091** [−0.19, +0.00] · −$3,299 | 238 · 79.2 % · **+0.044** [−0.05, +0.13] · +$2,716 | 394 · 76.3 % · **−0.009** [−0.08, +0.06] · **−$583** |
| BULLRUN | replay | 0 trades | 147 · 51.1 % · **+0.256** [−0.16, +0.40] · +$5,497 | 147 · 51.1 % · **+0.256** · **+$5,497** (6 days, one event) |
| BEARRUN (5×) | replay | 5 · 56 % · **−0.116** [−0.62, +0.04] · −$23 | 8 · 75 % · **+0.149** [+0.03, +0.45] · +$44 | 13 · 67.5 % · **+0.043** [−0.18, +0.21] · **+$21** |
| FRENZY_LONG | replay ᶜ | 103 · 49.8 % · **−0.021** [−0.51, +0.46] · +$506 | 85 · 53.3 % · **+0.193** [−0.41, +0.75] · +$1,363 | 188 · 51.4 % · **+0.076** [−0.29, +0.44] · **+$1,870** |
| FRENZY_WIDE | replay ᶜ | 60 · 53.3 % · **+0.196** [−0.60, +1.00] · +$344 | 46 · 56.8 % · **+0.406** [−0.61, +1.33] · +$551 | 106 · 54.9 % · **+0.288** [−0.36, +0.91] · **+$895** |
| FRENZY_LITE | **study cohort, not engine replay** ᵈ | 265 · 56.2 % · **+0.266** [−0.16, +0.67] · +$3,089 | 283 · 56.9 % · **+0.289** [−0.05, +0.62] · +$3,589 | 548 · 56.6 % · **+0.278** [−0.00, +0.53] · **+$6,678** |
| FRENZY_WILLY (paper, no stop) | **separate-source tick estimate** ᵉ | 698 · 82.5 % · **+0.037** [−0.15, +0.21] · +$3,730 | 565 · 85.7 % · **+0.074** [−0.16, +0.28] · +$6,143 | 1,263 · 83.9 % · **+0.053** [−0.09, +0.19] · **+$9,874** |
| ↳ WILLY with live backstop ≈ −2.2 % (approx.) | idem | 698 · 65.5 % · **−0.139** [−0.26, −0.02] · −$14,230 | 565 · 67.8 % · **−0.063** [−0.20, +0.07] · −$5,239 | 1,263 · 66.5 % · **−0.105** [−0.19, −0.01] · −$19,469 |
| SURGE_LONG | replay — **trigger and exit MISMATCHED** ᶠ | 28 · 60.2 % · **+0.048** [−0.32, +0.44] · +$201 | 26 · 50.0 % · **−0.118** [−0.63, +0.57] · −$449 | 54 · 55.3 % · **−0.032** [−0.37, +0.34] · **−$247** |
| **TOTAL live, without WILLY** (replay + heat re-admits + LITE) | sum of separate $3k books | 1,110 · 59.0 % · +0.016 [−0.10, +0.14] · **−$7,320** | 1,237 · 62.9 % · +0.120 [+0.01, +0.22] · **+$12,111** | 2,346 · 61.0 % · +0.071 [−0.01, +0.15] · **+$4,791** |
| **TOTAL live, with WILLY (paper)** | idem | 1,808 · 68.1 % · +0.024 [−0.08, +0.13] · **−$3,590** | 1,802 · 70.0 % · +0.106 [+0.00, +0.20] · **+$18,255** | 3,609 · 69.0 % · +0.065 [−0.01, +0.13] · **+$14,665** |
| TOTAL replay sleeves only (no LITE, no WILLY) | idem | 845 · 59.8 % · −0.063 [−0.15, +0.03] · −$10,409 | 954 · 64.7 % · +0.070 [−0.03, +0.15] · +$8,522 | 1,798 · 62.4 % · +0.008 [−0.06, +0.07] · −$1,887 |
| *SURGE_SHORT (OFF — reference, not in totals)* | replay | 220 · 67.6 % · −0.115 [−0.20, −0.03] · −$3,684 | 126 · 76.4 % · +0.026 [−0.07, +0.12] · +$479 | 346 · 70.8 % · −0.064 · −$3,204 |

How to read the TOTAL rows:
- **TOTAL $** is the sum of separate $3k books, one per strategy (about $33k of capital). It is **not** one shared $3k book. The shared-book compounding question was the Oct-6 table and was not re-run here.
- **TOTAL avg %** pools trades from different sleeves. Read the per-sleeve rows for the edge.

Cross-checks for FRENZY (an independent source):
- The vol/mcap study's tick cohort was built from engine-parity entries under today's FRENZY rules: ATR ≤ 3.0, red → LONG, hold-green → WIDE, bearish-day block on, fixed +3/−3, 8 s entry, sequenced 2 slots. Its numbers:
  - **FRENZY 110 · +0.149 (H1) / 85 · +0.361 (H2), year +0.241, +$2,062.**
  - **WIDE 71 · +0.255 / 50 · −0.100, year +0.108, +$384.**
- The engine replay and the tick cohort agree on the sign of the year and roughly on $. They differ on per-trade size: FRENZY_LONG +0.08 here vs +0.24 there.

### Per-seed Net $ (replay sleeves; seeds differ only in tick ordering, so they are not independent years)

| Strategy | H1 s1 / s2 / s3 | H2 s1 / s2 / s3 |
|---|---|---|
| MOM_LONG (replay part) | −5,297 / −4,969 / −4,736 | −1,565 / −1,761 / −1,652 |
| MOM_SHORT | −1,314 / −1,439 / −1,230 | −287 / −480 / +168 |
| FLIP_SHORT | −857 / −1,087 / −783 | −701 / −226 / −484 |
| SPIKE_FADE | −4,753 / −2,969 / −2,174 | **+6,680 / +1,307 / +161** (seed spread ≫ the mean) |
| BULLRUN | — | +5,926 / +2,519 / +8,047 |
| FRENZY_LONG | +419 / +550 / +550 | +1,679 / +1,277 / +1,134 |
| FRENZY_WIDE | +344 / +344 / +344 | +522 / +609 / +522 |
| SURGE_LONG | +926 / +397 / −720 | −535 / −738 / −74 |

## What was re-applied, per strategy (and what could not be)

The yr5 replay ran code 181131e, which was frozen 2026-10-04 14:06 UTC. 55 commits since then. The config was diffed key-by-key against `frozen_config_yr5_181131e.json` and checked against DECISION_LOG 200–255.

**MOM_LONG**

Re-applied:
- **Today's cell sizing** via `build_master_pool.today_size_rule` (frozen STACK 2026-10-08b):
  - UNMATCHED 2× → 1.5×.
  - Crowd-sprint and pair-vol ≥ 0.90 → 1×.
  - NONEXP_CALM3D door 2× → 1×.
  - Mean today multiplier 1.25.
- **LONG_HEAT_BLOCK back to the 3-leg rule** (208): BTC EMA20 slope ≥ 0.07 ∧ BTC RSI-prev ≥ 64 ∧ bull ≥ 80, washed-out (≤ −10 %) exempt. Applied on stamps; removes 22.7 fills/seed, matching DL 208's 23.
- **LONG_CHOP_BURST** (201): eff72 ≤ 0.007 ∧ another kept bot fill of any live sleeve ≤ 120 s earlier. Applied sequentially per seed on today's kept fills; removes 11.3/seed.
- **Re-admits:** the yr5 re-scope blocked signals at bull ≥ 85 that today's 3-leg rule lets through.
  - 68 of 153 signals (36.7/seed; DL 208 said ~41), from `HEAT_YR5_SIGNALS_priced.csv`.
  - They were priced by the live ML exit replica, not the engine, and sized at the sleeve's mean today multiplier.
  - Their BTC legs were rebuilt from the 5m cache: 96.9 % agreement with the stamped heat legs on yr5 ML fills.

Could not:
- Re-admits were not checked against the chop-burst or the other long gates.
- That signal file predates the warm-up trim and may hold warm-up duplicates.
- Path effects are not modelled: a refused fill frees a slot or cooldown the replay gave to someone else.
- Known replay ≠ live gap (DECISION_LOG 204): the replay runs 24 h, and entry timing differs.

**MOM_SHORT:** nothing new since 181131e (pair-vol 0.86 was already live). Every SHORT cell is 1× (factor 1.0 on all fills).

**FLIP_SHORT:**
- FAN flips at 10× (registry lev 0.5; `today_size_rule` FLIP_FAN_LEV) and every cell at 1×. The size factor averages 0.35 vs as-traded.
- No new flip entry filter since 181131e (the scout FLIP_*_BLOCKED lines are observe-only).

**SPIKE_FADE:**
- No new filter. Today's ticket on a $3k book = min($29,250 desired, 0.5 % × 24 h volume, $500k), from the stamped volume (the master's CF_FADE_CAP05 logic).
- The median ticket is 0.79 of desired, against the replay's 0.45 on its larger chunk books. This is why the year improves −$1,593 → −$583: same per-trade %, bigger tickets.
- Optimistic:
  - There is no market-impact haircut.
  - The live stop slip (scout STOP_SLIP: −0.32 % on fades vs ≈ 0 in paper) is not applied.

**BULLRUN:** unchanged. All 147 fills/seed are Aug 19–24.

**BEARRUN:** 5× (bearrun_lev_mult 0.25, operator override 252). The replay's per-trade % is unchanged. The tick-replay review (SURGE_BEARRUN_REVIEW) found −0.24 %/window over 7 windows.

**SURGE_LONG — mismatched, shown as reference:**
- Fills are priced at full size (20×), but the trigger (0.3 % · 5× · market volume ≥ 1 · spacing after a fill, DL 202) and the exit (Bull-Run trail without the +0.2 lock, DL 203) both changed after 181131e. These fills are the OLD trigger and exit.
- The tick replica of today's option B + no-lock exit (DL 203) reads 131 triggers/yr · +0.395 %/trigger (its own halves +0.34 / +0.43). That is a different ruler and is not in the table.

**FRENZY_LONG — engine replay, re-priced:**
- **Exit: fixed +3 / −3 / 12 h** (yr5 ran TP +4). A fill whose recorded net peak reached +3 books +3.
  - This is exact on the path: the run closes at the TP or at −3, so a recorded peak ≥ +3 means +3 was touched before the close.
  - It equals the master's 10-08a rule.
- **ATR cap 3.0:** yr5 had cap 2.5, so red / flat setups with ATR in (2.5, 3.0] became **FRENZY_WIDE fills** in the replay. Those **59 fills/seed are re-coded to FRENZY_LONG** at 6× (10× when strong). 43/seed survive the bearish block.
  - So the "missing" fills come from the replay itself, not from the ATR-cap study.
  - The ATR-cap study counts 63 such signals per year, a close match.
  - Setups the replay's WIDE refused (WIDE slots full) cannot be recovered.
- **Bearish-day block** (BTC last closed UTC daily return < 0 ∧ BTC 5m EMA13−EMA50 gap < 0) on the yr5 stamps. It removes 73 fills/seed.
  - Rebuild check (`study_yr5_halves_btc.py`): 99.5 % agreement with yr5 stamps, 99.3 % with 413 live master fills, 111 / 111 with the vol/mcap study's block.
- The market-volume gate (frenzy_gvol_max 1.0), the red / flat candle rule, the strong-flag 10× and the pair-day cap were already in 181131e.

Could not:
- Re-entries a +3 exit would allow (≤ 3 per pair per day).
- Slot interplay of the re-coded fills.
- The 1,000 vs 1,500-bar window bug (DL 230; the replay's data source is not the live ccxt path).
- Catch-up (no pauses in a replay).

**FRENZY_WIDE:**
- Same exit and bearish block as FRENZY_LONG.
- **Hold-green only** (231): ATR ≤ 3.0 ∧ green candle ∧ above-VWAP streak > 12.
  - The streak is joined from the engine-parity cohort (`FRENZY_ENGINE_COHORT_2026-10-05.csv`) for 92.7 % of WIDE fills; ATR agrees on 99.8 % of joined rows.
  - The 7.3 % with no join (31.7/seed, Jan 4–10 and Sep 27–Oct 4 mostly) are **refused, fail-closed like the master builder**.
- Refused per seed: ATR > 3.0: 221 · green reclaim (streak ≤ 12): 119 · streak unknown: 32 · bearish day: 34. Kept: 106/seed.

**FRENZY_LITE — study cohort, NOT engine replay:**
- The 724-fill "12 closes, judged once" universe (`scratch lite_streak/kept.pkl` K[12]; it reproduces the DL 243 universe 724 = 724). Ticks, live timing (+12 s, exit slip 0.10, fees 0.09).
- The lock pricing was converted to fixed +3/−3: a lock-armed fill (or a 12 h-cap fill ≥ +1.9) had touched +3 net, so it now books +2.90 (+3 minus the ruler's 0.10 exit slip). The rest are unchanged (−3 stop or cap).
- **Bearish-day block** rebuilt from BTC klines: 176 of 724 blocked.
  - ⚠ DL 250 quotes 215 bearish for LITE. The rebuild here is validated three ways (above), so that gap is unexplained.
- Lev 0.32 → 6×.

Could not:
- 2 LITE slots, one position per pair, and the FRENZY-shortlist reachability (DL 243: reachable ≈ 696 · +0.134 under the lock).
- The cohort is unsequenced.

**FRENZY_WILLY — never backtested as shipped. First re-price here (`scripts/study_yr5_halves_willy.py`), labelled an estimate:**
- **Triggers:** the vol/mcap study's WILLY A (1,781 first-flag events with ticks) and WILLY B (1,038 fresh-ON bars that today's FRENZY / WIDE did not take; LITE's takes are not removed).
- **Turnover:** R = vol24h / mcap < 1.0 at the study's trigger-time estimate. R unknown → refused, as live is fail-closed. 470 + 697 blocked, 172 + 135 unread.
- **Entry:** first closed red 5m bar (k5m_full klines) within 60 min of the trigger, entry at the first print ≥ close + 8 s, +0.035 % slip. Dislocation > 1 % → that bar skipped.
- **Exit and costs:** TP +1.0 net (taker 0.045 × 2), **no stop**, 120-min cap (last print − 0.05).
- **Sequencing:** one WILLY at a time (slot 1 plus the hold).
- **Pricer parity:** with the old exit (immediate, −3, 60 min) it reproduces the study pricer **300 / 300 exactly**.

Could not:
- **The global hold's effect on the other sleeves (stated, not modelled).**
- The 431 study events without ticks and the 108 dislocated ones (both dropped upstream).
- Liquidation: paper does not liquidate. Live, the ≈ −2.2 % exchange backstop is the stop (the sensitivity row).

**SURGE_SHORT:** OFF, reference only.

## What changed vs the Oct-6 table (full-year Net $, flat $3k book)

| Strategy | Oct-6 | Today | What moved it |
|---|---|---|---|
| MOM_LONG | −7,740 | **−6,433** | sizing (CALM3D 1×, de-mux) −7,456 → heat 3-leg −7,110 → chop-burst −6,660 → heat re-admits −6,433 |
| MOM_SHORT | −1,527 | −1,527 | nothing |
| FLIP_SHORT | −2,759 | **−1,380** | FAN 20× → 10× (half the risk, same −0.137 %/trade) |
| SPIKE_FADE | −1,593 | **−583** | ticket re-sized on a $3k book (0.5 % cap) — same −0.009 %/trade |
| BULLRUN | +5,497 | +5,497 | nothing |
| BEARRUN | +4 | +21 | 1× → 5× |
| FRENZY_LONG | +2,770 | **+1,870** | TP +4 → +3: **+411** (−0.163 %/trade) → + ATR 2.5–3.0 red fills: +1,922 → bearish block: +1,870 |
| FRENZY_WIDE | −1,604 | **+895** | TP +3: −631 → minus the fills moved to LONG: −1,233 → hold-green only: +1,562 → bearish block: +895 |
| SURGE_LONG | −247 | −247 | nothing re-priceable (mismatched) |
| FRENZY_LITE | — | **+6,678** | new sleeve (study cohort) |
| FRENZY_WILLY | — | **+9,874 paper / −19,469 with the live backstop** | new sleeve (separate tick estimate) |
| TOTAL live (no SURGE_SHORT) | −7,199 | **−1,887** replay only · **+4,791** + LITE · **+14,665** + WILLY | |

## Caveats (labelled)

1. **In-sample.** Most changes since Oct-6 were chosen on this same year: bearish-day block, WIDE hold-green, LITE's design, WILLY's turnover cut (read after the scan), and the heat revert's evidence. Apply the 30–50 % haircut to every improvement vs Oct-6. LITE, WILLY, hold-green and the bearish block all **fail** the locked promotion bars (declared operator overrides).
2. **Replay ≠ live for MOM_LONG** (DECISION_LOG 204): yr5 runs the momentum sleeve 24 h. Live hours ≈ +0.04 %/trade vs off-hours −0.23. Real as-traded ML since Jun-17 is +0.05 %. The table is the replay, not the live expectation.
3. **The fixed +3 exit is priced from the recorded peak.** This is exact for path order, but it books exactly +3.00 with no TP slip. The +3 vs +4 comparison is one replay. DECISION_LOG 199 carries a pre-committed tick re-check of +4 vs +3 on live fills.
4. **LITE and WILLY are study cohorts**, not engine fills. They are unsequenced against the other sleeves, and LITE ignores its 2-slot / reachability limits. For WILLY, the TOTAL with WILLY ignores the global hold, which in reality would block other sleeves' trades while a WILLY is open.
5. **WILLY risk.** The paper result needs no stop and 20× leverage:
   - 15 % of trades dip to −4.5 % or lower (liquidation zone).
   - The worst trade is −25.8 % (−126 % of a $3k book in one trade).
   - With the backstop, the year is negative with 95 % confidence.
6. **Seeds are not independent years.** SPIKE_FADE H2 ranges +$161 … +$6,680 across seeds. BULLRUN is one event.
7. **Flat-book dollars hide ruin.** The Oct-6 shared-book compounding view is not repeated here. Several sleeves' H1 losses exceed their $3k stake.
8. **CIs are day-clustered on pooled seeds.** BULLRUN and BEARRUN have only 6 trading days. Their CIs are not meaningful.

## Files

- `reports/YR5_HALVES_TODAY_STACK_2026-10-08.csv`: every row above × H1 / H2 / FY. Columns: N (mean of seeds), WR, avg %, CI, Net $, days, source, per-seed $.
- `scripts/study_yr5_halves_today.py`: re-application + table (`--oct6` = parity mode).
- `scripts/study_yr5_halves_btc.py`: BTC 1d return / trend gap / heat legs rebuilt from the 5m cache, plus its validation.
- `scripts/study_yr5_halves_willy.py`: WILLY current-design tick pricer.
- `scripts/study_yr5_halves_report.py`: CSV, waterfall and WILLY sensitivity.
- Intermediates are in the session scratchpad `yr5h/` (res.json, F_today.pkl, willy_priced.csv, waterfall.json).
