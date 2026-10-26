# B25 review prep: block ③ BACKTEST (prepared 2026-10-26, before the review)

> **⚖ Evidence weight (operator 2026-10-26, DL 378):** the backtest is less accurate than the master's real fills. These blocks are a SECONDARY cross-check: at the review the master per batch + the current batch lead; a 'refutes' below is a caveat to weigh against the master read, never a veto on its own.

This file covers block ③ of the three blocks (① batch · ② master per batch · ③ backtest) for the first three sleeves in stake order.
It is **research only**. No config, engine or git change was made, nothing was staged, the bot API was not called and no network was used:
every number comes from cached replay, study and path data.

**The backtest only REFUTES.** A backtest result never proves a change on its own. Every Δ below is shown per seed and per period, with a
UTC-day-block bootstrap 95 % CI.

**Scripts** (new files; scripts/engine_replay*.py and scripts/backtest_fetch_ticks.py were not touched):
- scripts/prep_b25_bt_common.py
- scripts/prep_b25_bt_fade.py
- scripts/prep_b25_bt_flatbase.py
- scripts/prep_b25_bt_willy.py

**Row CSVs:**
- reports/PREP_B25_BT_FADE_{summary,monthly,monthcons,rows}.csv
- reports/PREP_B25_BT_FLATBASE_{summary,monthly,monthcons,rows}.csv
- reports/PREP_B25_BT_WILLY_{summary,monthly,monthcons,rows}.csv

**Reproduction checks (passed).** Each pipeline reproduces the number its source study published:
- fade line ④ sub-zone, s1 tick-priced: 121 · 76.9 % · +0.231. Scout registration: 121 · 77 % · +0.23.
- FLATBASE live cell: 2026 all-S4 +0.93 and turnover > 1 +1.65. DL 368 / 348 published the same.
- WILLY: see §3.

---

## 1 · SPIKE_FADE watch lines (DL 374)

**Source:** reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv.
- Coverage: yr6a 2025 + yr6 2026, 2025-01-01 → 2026-10-11.
- Fills: warm-up trimmed, CLOSED only, no probes.
- Seeds: s1 625 · s2 608 · s3 599 fades.
- `s1tick` is seed 1 with its 12 worst stops re-priced on aggTrades (reports/FADE_SIZE_MCAP_2026-10-19_rows.csv). Lines ⑤ ⑥ were discovered on these prices.

**Stack check.** Every `spike_fade_*` key of the replay config (frozen_config_yr6_01419d4.json) equals today's trading_config.json, so the fade entry and exit rules match today's.

**Stack gap: the WILLY global hold.** The replay ran with the hold ON; it is OFF today (DL 340).
- The fades the hold refused are missing from the replay.
- The journals show about 41–50 unique pair-hours per seed of SHORT entries refused by FRENZY_WILLY_HOLD over the 2 years. That count covers all short sleeves, not only fades, so it is an upper bound.
- These fills are not priced. This is a cohort-completeness gap of at most about 8 % of the fade N.

**Units.** $ is on a fixed $2,900 book at the replay's engine sizing (2× cell, about $28k notional), with no compounding. 1× $ = $ / 2. Thresholds are the scout's frozen constants.

**Sleeve baseline (all fades).**

| arm | N | WR | avg % | 95 % CI | $ at 2× |
|---|---|---|---|---|---|
| s1 | 625 | 73.3 % | +0.056 | [−0.04, +0.15] | +$9,824 |
| s2 | 608 | 71.5 % | −0.017 | [−0.11, +0.08] | −$1,333 |
| s3 | 599 | 73.5 % | −0.016 | [−0.12, +0.08] | −$2,551 |
| s1tick | 625 | 73.1 % | +0.060 | | +$10,312 |

In this backtest the sleeve is close to zero, and the seed changes its sign.

### Filter lines ① ② ⑤ ⑥: before → after, BLOCKED vs KEPT (ALL period)

"Bar" is the CLAUDE.md expectancy bar applied to the blocked cohort at 1×: WR below breakeven · P(avg < 0) ≥ 0.95 by day bootstrap · ≥ 8 days · no day or pair ≥ 50 % of the loss · N ≥ 15.

**① ADX Δ < 2.359 (block the weak side)**

| arm | blocked N · WR · avg [CI] | blocked 1× $ | kept N · avg | kept 1× $ | bar |
|---|---|---|---|---|---|
| s1 | 306 · 75.5 % · +0.073 [−0.05, +0.19] | +2,109 | 319 · +0.040 | +2,803 | fails (WR 75.5 > BE 69.3; P(<0) 0.12) |
| s2 | 292 · 73.3 % · +0.020 [−0.09, +0.14] | +202 | 316 · −0.051 | −869 | fails |
| s3 | 285 · 73.7 % · −0.019 [−0.13, +0.09] | −1,223 | 314 · −0.013 | −53 | fails (P 0.63) |

**② pair ADX ≤ 19.2 (block)**

| arm | blocked | blocked 1× $ | kept | kept 1× $ | bar |
|---|---|---|---|---|---|
| s1 | 174 · 74.1 % · +0.089 [−0.09, +0.28] | +2,548 | 451 · +0.044 | +2,364 | fails |
| s2 | 161 · 74.5 % · +0.062 [−0.13, +0.25] | +1,082 | 447 · −0.045 | −1,748 | fails |
| s3 | 163 · 71.8 % · −0.046 [−0.22, +0.13] | −942 | 436 · −0.004 | −334 | fails (P 0.71) |

**⑤ HB ∧ ADX Δ ≤ 2.4341 (block)**

| arm | blocked | blocked 1× $ | kept | kept 1× $ | bar |
|---|---|---|---|---|---|
| s1 (IN-SAMPLE) | 90 · 68.9 % · −0.077 [−0.37, +0.19] | −1,280 | 535 · +0.079 | +6,192 | fails (P 0.71) |
| s2 | 85 · 80.0 % · **+0.205** [−0.01, +0.41] | +1,918 | 523 · −0.053 | −2,585 | fails |
| s3 | 83 · 79.5 % · +0.057 [−0.15, +0.24] | +339 | 516 · −0.027 | −1,615 | fails |

**⑥ HB ∧ BTC ADX prev > 24.5874 (block)**

| arm | blocked | blocked 1× $ | kept | kept 1× $ | bar |
|---|---|---|---|---|---|
| s1 (IN-SAMPLE) | 45 · 68.9 % · −0.219 [−0.65, +0.15] | −1,178 | 580 · +0.078 | +6,090 | fails (P 0.87; top day 23 %) |
| s2 | 46 · 73.9 % · −0.004 [−0.37, +0.33] | −48 | 562 · −0.018 | −618 | fails |
| s3 | 49 · 81.6 % · +0.123 [−0.11, +0.32] | +571 | 550 · −0.028 | −1,847 | fails |

**Month consistency** (blocked worse than kept, months with ≥ 3 fills on each side; source: _monthcons.csv):

| line | s1 | s2 | s3 |
|---|---|---|---|
| ① | 9 / 20 | 8 / 19 | 10 / 20 |
| ② | 10 / 18 | 9 / 18 | 12 / 18 |
| ⑤ | 8 / 12 | 4 / 14 | 5 / 13 |
| ⑥ | 6 / 7 | 3 / 8 | 2 / 9 |

①–② are a coin flip. ⑤–⑥ hold only on seed 1.

**Per period** (2025 bull / 2026-H1 bear / 2026-H2 bull; blocked avg, s1 · s2 · s3):

| line | 2025 | 2026-H1 | 2026-H2 |
|---|---|---|---|
| ① | +0.03 · +0.00 · −0.08 | +0.05 · −0.02 · −0.01 | +0.24 · +0.17 · +0.10 |
| ⑤ | −0.01 · **+0.47** · −0.04 | **−0.25** · −0.04 · +0.06 | +0.29 · +0.39 · +0.23 |
| ⑥ | −0.35 · −0.06 · −0.06 | **−0.36** · +0.10 · +0.12 | +0.33 · −0.12 · +0.39 |

- The only negative cells are 2026-H1 bear on s1 (the discovery seed).
- In the newest half (2026-H2) every blocked cohort is positive except ⑥ s2.

**Concentration.** The top day and top pair carry ≤ 25 % of any blocked cohort's loss. No line is a single-day or single-pair artefact.

### Size lines ③ ④: before → after (ALL period, zone only; the avg % does not move with size)

| line | arm | zone N · WR · avg [CI] | zone $ at 2× → after | Δ$ (2 y) | worst fill % of book, 2× → after | fills > 4 % of book | max DD of the fade book |
|---|---|---|---|---|---|---|---|
| ③ mcap ≥ $200M → 1.5× | s1 | 98 · 75.5 % · +0.105 [−0.11, +0.31] | +3,536 → +2,652 | −884 | −26.0 → −19.5 | 22 → 21 | −6,966 → −6,421 |
| | s2 | 92 · 71.7 % · +0.092 [−0.16, +0.36] | +2,805 → +2,103 | −701 | −26.0 → −19.5 | 25 → 24 | −9,337 → −8,444 |
| | s3 | 88 · 68.2 % · −0.025 [−0.27, +0.24] | −191 → −143 | +48 | −26.0 → −19.5 | 27 → 24 | −13,476 → −12,191 |
| ③ at 1× | s1 / s2 / s3 | (same zone) | | −1,768 / −1,402 / +96 | → −13.0 | 22→16 / 25→19 / 27→21 | −5,877 / −7,550 / −10,906 |
| ④ HEALTHY_BEAR → 1× | s1 | 180 · 73.9 % · **+0.154** [−0.07, +0.37] | +6,960 → +3,480 | −3,480 | −50.4 → −25.2 | 40 → 30 | −6,966 → −6,843 |
| | s2 | 183 · 74.9 % · +0.096 [−0.07, +0.26] | +4,766 → +2,383 | −2,383 | −44.6 → −22.3 | 39 → 30 | −9,337 → −9,307 |
| | s3 | 177 · 79.7 % · +0.135 [−0.03, +0.29] | +5,317 → +2,659 | −2,659 | −44.2 → −22.1 | 31 → 25 | −13,476 → −13,404 |
| ④b HB ∧ trend gap ≥ −0.13 → 1× | s1 / s2 / s3 | 121 / 127 / 122 · +0.216 / +0.121 / +0.165 | | −2,939 / −2,111 / −2,213 | | | DD slightly WORSE (−7,282 / −9,922 / −14,503) |

**Per period:**
- ③ zone avg is positive in 2025 on all seeds (+0.31 / +0.38 / +0.18). In 2026-H1 bear it is +0.01 / **−0.16 / −0.27**, and in 2026-H2 −0.05 / +0.09 / +0.17.
- ④ HB is negative only in 2026-H1 on s1 / s2 (−0.09 / −0.02); otherwise positive.
- Mcap ≥ $200M fills: 98 of 625 (131 unscored).

**Size and the 4 %-of-account test.** The replay's notional is about 9.75× the book. Live B25 ran about 2× invest; its −1.5 % stop cost about $305 on $2,900. At either sizing:
- a single fade beyond the stop already exceeds 4 % of the account;
- a 1.5× cap removes only about 1 of the 22–27 "> 4 %" fills;
- 1× removes about 6.

This agrees with the scout's own note: no size cap alone meets the 4 % bar, so the decision is stop width × size.

### Verdicts, fade

| line | backtest verdict | why |
|---|---|---|
| ① ADX Δ < 2.359 | **REFUTES** the filter | Blocked side ≥ kept on 2 of 3 seeds; WR above breakeven on 2 / 3; never close to P 0.95; months a coin flip. |
| ② pair ADX ≤ 19.2 | **REFUTES** | Blocked better than kept on 2 of 3 seeds. |
| ⑤ HB ∧ ADX flat | **REFUTES out of seed** | The thresholds were fitted on this backtest (seed 1, in-sample). Seeds 2 / 3 flip it to blocked +0.205 / +0.057. |
| ⑥ HB ∧ BTC ADX high | **REFUTES out of seed** | Same reason: s1 −0.22 in-sample; s2 −0.004, s3 +0.123. |
| ③ high-mcap 1.5× | **Does not support the cap on avg %.** It mildly supports it as a ruin lever only. | DL 311's trigger is avg % < 0: the backtest zone is > 0 on 2 of 3 seeds and < 0 only in 2026-H1. The cap costs about $700–880 per 2 y on the winning seeds and trims the max DD by about 8 %. |
| ④ HB at 1× | **REFUTES the size cut** | HB is the BEST-earning regime on all 3 seeds (+0.10…+0.15). The cut costs $2.4–3.5k per 2 y and leaves the max DD unchanged. 2026-H1 is the only weak half. |

### BLIND or reconstructed, fade

- **mcap:** reconstructed. The replay's proxy is today's circulating supply × the entry price (--mcap-proxy bapi_supply.json). Coins whose supply changed are mis-scored, and 21 % of fills are unscored (131 / 625). Live uses the CMC mcap at entry.
- **HEALTHY_BEAR label:** present. The replay engine stamps entry_btc_regime with the same classifier as live, at 100 % coverage.
- **entry_adx_delta, entry_adx, entry_btc_adx_prev, entry_btc_trend_gap_pct:** present at 100 %.
- **Fades blocked by the WILLY hold:** BLIND (see the stack gap above).
- **Price fidelity:** 1-minute path pricing. Tick-tail re-pricing exists for seed 1 only, and only for its 12 worst stops.
- **Seeds:** s2 and s3 share most entries with s1. They are partial replications, not independent samples.

---

## 2 · FLATBASE_SHORT exits (DL 367 / 368 / 370)

> **Scope note (coordinator, 2026-10-26):** the full FLATBASE deep review (theory, every exit alternative, W/L on every variable) runs AT the B25 batch review (NEXT_BATCH_REVIEW_PLAN ②f ⭐). This section is the backtest reference for it (a re-read of the DL 368 / 370 simulator, no new simulation).

**Source:** the deterministic S4 · 72 h simulator of scripts/study_flatbase_tp3_stop5.py. This is a pure re-read; no new simulation was run.
- Inputs: reports/FRENZY_FLATBASE_TP3_STOP5_2026-10-24_trades.csv plus its TP1 re-run reports/FRENZY_FLATBASE_TP1_2026-10-24_trades.csv. They are joined on pair + dt + period, and the live cell agrees row by row between the two runs.
- Coverage: 2025-01 → 2026-10.
- Costs: 0.20 %, with real funding included. The scout's shadows exclude funding.
- Sizing: one short per pair. $ is at today's ≈ $310 notional per fill.

**Today's stack: S4 + turnover > 1, or unknown mcap → trade (DL 348).**
- Turnover is joined from the WINLOSS trades: 717 > 1, 379 unknown, 1,168 ≤ 1 shadow-only.
- 1,084 fills are on today's stack.

**Single realisation.** The simulator has no seed: the entry is deterministic. "Per seed" therefore reduces to one realisation.

**Split rows.** An exit that closes earlier can free the pair for a later episode, so a few rows exist for one variant only. They were merged. On today's stack each shadow has 8–12 shadow-only rows (no LIVE twin); the shadow's own N · WR · avg includes them, the paired Δ excludes them (fixed 10-26 after review: an earlier loop overwrote the frame and printed 0).

### Live exit (TP −8 / stop 15 / 72 h), today's stack

| period | N | WR | avg % | 95 % CI | $ at $310 |
|---|---|---|---|---|---|
| 2025-H1 | 144 | 65.3 % | +0.83 | [−0.73, +2.33] | +368 |
| 2025-H2 | 341 | 70.1 % | +1.29 | [+0.17, +2.31] | +1,361 |
| 2026-H1 | 368 | 72.6 % | +1.70 | [+0.68, +2.70] | +1,943 |
| 2026-H2 | 231 | 72.7 % | +1.56 | [+0.36, +2.65] | +1,114 |
| ALL | 1,084 | 70.8 % | **+1.42** | [+0.84, +2.01] | +4,786 |

### Shadows vs live on the same entries, today's stack, ALL (Δ = shadow − live, %/fill)

| shadow | N | WR | avg | Δ avg [95 % CI] | Bonferroni LB | Δ by half (25H1 / 25H2 / 26H1 / 26H2) | months better / 21 |
|---|---|---|---|---|---|---|---|
| TP5 (−5/15) | 1,092 | 78.7 % | +0.84 | **−0.55** [−0.86, −0.21] | −0.95 | −0.53 / −0.39 / −0.50 / −0.90 | 6 |
| TP3 (−3/15) | 1,094 | 85.8 % | +0.42 | **−0.98** [−1.41, −0.53] | −1.51 | −0.17 / −0.96 / −1.08 / −1.36 | 4 |
| TP5_SL10 | 1,093 | 72.0 % | +0.62 | **−0.76** [−1.16, −0.35] | −1.28 | −0.47 / −0.63 / −0.90 / −0.90 | 4 |
| TP5_SL5 | 1,095 | 53.5 % | +0.07 | **−1.34** [−1.86, −0.84] | −2.01 | −0.39 / −1.15 / −1.83 / −1.46 | 5 |
| TP8_SL10 | 1,092 | 63.2 % | +1.05 | −0.32 [−0.64, −0.01] | −0.73 | **+0.29** / −0.55 / −0.50 / −0.08 | 7 |
| TP1 (−1/15) | 1,096 | 93.2 % | +0.04 | **−1.36** [−1.90, −0.82] | −2.05 | −0.65 / −1.28 / −1.57 / −1.61 | 6 |
| TP1_SL10 | 1,096 | 91.1 % | +0.01 | **−1.40** [−1.96, −0.84] | −2.12 | −0.66 / −1.42 / −1.58 / −1.54 | 4 |
| TP1_SL5 | 1,096 | 84.0 % | −0.09 | **−1.51** [−2.07, −0.94] | −2.23 | −0.85 / −1.34 / −1.81 / −1.68 | 5 |

### Two-sided split of the Δ (Σ %pts, today's stack, ALL)

Live outcomes: 232 stopped · 694 TP · 158 time exits.

| shadow | live-stopped: saved (n · Σ) | live-TP: cut winners (stopped by the shadow) | live-TP: give-back (lower TP) | live-time Σ |
|---|---|---|---|---|
| TP5 | 51 · +1,031 | 0 | 693 · −2,002 | +369 |
| TP3 | 107 · +1,945 | 0 | 691 · −3,322 | +313 |
| TP5_SL10 | 232 · +1,847 | 51 · −902 | 642 · −1,856 | +89 |
| TP5_SL5 | 232 · +2,731 | 211 · −2,655 | 483 · −1,393 | −141 |
| TP8_SL10 | 232 · +1,179 | 63 · −1,114 | 0 | −412 |
| TP1 | 186 · +3,018 | 0 | 693 · −4,665 | +168 |
| TP1_SL10 | 232 · +3,205 | 15 · −267 | 678 · −4,562 | +106 |
| TP1_SL5 | 232 · +3,387 | 69 · −881 | 624 · −4,194 | +50 |

- **No "deeper" fills exist** by construction: the live stop is the widest of all variants.
- **Concentration:** the top day carries ≤ 4 % of the positive Δ, and the top pair ≤ 3 % of the negative Δ.

### Verdict, FLATBASE

**The backtest REFUTES every shadow.**
- All 8 have a negative Δ with the 95 % CI below 0. The only borderline case is TP8_SL10 (upper bound −0.01).
- Every TP-lowering shadow is negative in all 4 halves. TP8_SL10 is positive only in 2025-H1.
- What the shadows save on live stops (+1.0…+3.4k pts) is more than paid back by the give-back on live TP winners.
- The +1 % family is near zero overall (+0.06 / +0.03 / −0.08 %/fill vs +1.42 live). The 93 % WR is real but sits below the ~95 % breakeven against a −15 stop.

This is consistent with DL 368 / 370 (all-S4) and stronger on today's turnover-gated stack, where the live cell is +1.42 rather than +0.72.

The DL 367 / 370 scout verdicts still decide on live fills. This block only says that the backtest prior is against all of them.

### BLIND, FLATBASE

- Kline replica, not an engine replay: no slot or pair-held interaction with FRENZY / WILLY, and no missed-boundary (FLATBASE_LATE) losses.
- Funding is included here; the scout's shadows exclude it.
- 5m bars use stop-first ordering inside a bar. DL 368: about 39 % of 5 % stop-outs are 1m wicks.
- The 4-hour-boundary entry is idealised, with no entry slippage beyond costs.

---

## 3 · WILLY V2b vs shadows (DL 356 / 357 / 358 / 364), WILLY vs WILLY_HT

**Method.** Every fill is re-walked on its own path with the scout's own walker (scout_willy_exit_v2.walk_v2 / shadow_walk), using the live V2b mechanics:
- stop −4, late TP +0.25 from minute 30, 90-minute cap, resting maker TP;
- live target = max(+1.25, min(+8, net at spike-high H × 0.99));
- each shadow exactly as the scout defines it; floor and ladder exits pay 0.05.

Δ = shadow − V2b, in %/fill. In the "V2b vs OLD" row the sign is flipped so that it reads as the shipped change.

**Arm A: engine replay, WILLY only** (WILLY_HIGHTARGET_2026-10-23_bt.csv cohort).
- Data: yr6a 2025 + yr6 2026, seeds 1–3, trimmed, entry A ≤ 60 min. 4,129 fills on 1m paths.
- Reproduction: the re-walk matches the DL 358 study fill by fill — OLD 100 %, V2b 100 %, TGT20 = H1_k2 100 % (within 0.02). NOLATE matches 93 %: the scout's NOLATE also drops the late TP on +1.25 fills, while the study's did not. The reads below use runners, where the two coincide.
- The replay ran the turnover filter, so these are WILLY fills only. **WILLY_HT cannot be told apart in the replay (BLIND there).**
- The 2026 seeds are near-copies (2026-H2 is identical across seeds, 81 runners each). They do not add independent evidence.

**Arm B: WILLY vs WILLY_HT split** (WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv).
- This is the today's-entry-A replica, 2026-01-10 → 10-07, tick paths, one realisation. It is NOT an engine replay: no slots, hold, pair-held or HT single slot.
- WILLY = turnover R_red < 1 (1,093 fills). HT = R_red ≥ 1 (477 fills).
- 163 fills with R unread are excluded (fail-closed: no entry).
- HT is reconstructed by its routing rule. Its 0.25 size does not change %.

The table reads the scout's population, **RUNNER fills (target > +1.25)**, where every shadow except NOLATE / LATE can differ. All-fill reads are in _summary.csv (pop = all).

### V2b baseline and the shipped change (runners)

| arm | N | WR | V2b avg % [CI] | V2b vs OLD Δ: ALL · 2025 · 2026-H1 · 2026-H2 |
|---|---|---|---|---|
| replay s1 | 798 | 59.1 % | −0.300 [−0.55, −0.06] | +0.01 · −0.10 · **+0.26** · +0.03 |
| replay s2 | 793 | 59.3 % | −0.245 | +0.01 · −0.09 · +0.26 · +0.03 |
| replay s3 | 790 | 59.5 % | −0.229 | +0.02 · −0.09 · +0.27 · +0.03 |
| study WILLY (2026) | 441 | 67.1 % | +0.141 [−0.13, +0.40] | +0.06 · – · −0.00 · +0.19 |
| study WILLY_HT (2026) | 227 | 61.7 % | −0.139 [−0.53, +0.24] | +0.03 · – · −0.23 · +0.34 |

On the 2026 replay runners, V2b beats OLD by about +0.20 (CI [−0.05, +0.45]). In 2025 it is worse by about −0.09. This is the DL 358 picture: the backtest does not refute V2b in 2026.

### Shadows vs V2b (runners, ALL; Δ %/fill [day-boot 95 % CI])

| shadow | replay s1 | replay s2 | replay s3 | study WILLY | study HT | replay Δ by period (2025 / 26H1 / 26H2) | months better (replay / WILLY / HT) |
|---|---|---|---|---|---|---|---|
| FLOOR10 | −0.047 [−0.18, +0.09] | −0.052 | −0.060 | −0.114 [−0.30, +0.07] | −0.046 | +0.07 / **−0.33** / −0.11 | 12 / 22 · 4 / 10 · 6 / 10 |
| STEPS | −0.029 | −0.034 | −0.046 | −0.123 | −0.075 | +0.09 / −0.30 / −0.09 | 11 / 22 · 5 · 6 |
| TRAIL10 | −0.033 | −0.038 | −0.050 | −0.122 | −0.065 | +0.09 / −0.31 / −0.09 | 11 · 4 · 6 |
| TRAIL15 | −0.035 | −0.040 | −0.052 | −0.122 | −0.043 | +0.09 / −0.31 / −0.10 | 12 · 4 · 6 |
| HALF | −0.041 | −0.046 | −0.056 | −0.117 | −0.043 | +0.06 / −0.28 / −0.07 | 11 · 4 · 6 |
| NOLATE | −0.014 [−0.11, +0.09] | −0.006 | −0.003 | **+0.048** [−0.06, +0.16] | +0.013 | −0.06 / +0.00 / **+0.26** | 6 / 22 · 7 · 7 |
| TGT15 (k 1.5) | −0.001 [−0.06, +0.06] | +0.000 | −0.001 | −0.020 | −0.070 | +0.03 / −0.07 / −0.02 | 9 · 5 · 5 |
| TGT20 (k 2.0) | **+0.055** [−0.04, +0.15] (P > 0 0.88) | +0.057 | +0.055 | −0.035 [−0.15, +0.09] | −0.068 | +0.10 / −0.05 / +0.06 | **15** / 22 · 4 · 6 |
| LATE45 | −0.016 | −0.013 | −0.012 | +0.026 | −0.036 | −0.04 / −0.02 / +0.17 | 9 · 7 · 6 |
| LATE60 | −0.000 | +0.007 | +0.008 | +0.002 | +0.018 | −0.03 / +0.04 / +0.16 | 9 · 5 · 6 |

### Two-sided split (replay s1 runners, Σ %pts of Δ by V2b exit class; other seeds within ±5 pts)

| shadow | V2b-STOP fills (300): saved | V2b-TP fills (307): give-back | V2b-LATE_TP (165): up / down | CAP (26) |
|---|---|---|---|---|
| FLOOR10 | +426 | −414 | +31 / −98 | +18 |
| TGT20 | +199 | −226 | +55 / −1 | +16 |
| NOLATE | 0 | 0 | +204 / −215 | 0 |
| LATE60 | 0 | 0 | +158 / −158 | 0 |

The floor and the ladders save about as much on stops as they give back on the V2b TP runners, then lose on late TPs. TGT20 wins only by rescuing late TPs.

**Concentration.** No shadow has a day carrying > 10 % or a fill carrying > 6 % of its positive Δ in the replay. In HT: TGT15's top day = 23 %, LATE45's top day = 33 %.

### Verdicts, WILLY / WILLY_HT

The scout's frozen bar is avg Δ ≥ +0.10 %/fill on runners. **No shadow reaches it in any arm.**

| shadow | backtest verdict | why |
|---|---|---|
| FLOOR10, STEPS, TRAIL10, TRAIL15, HALF | **REFUTE** | Negative on every seed and both study sleeves. Worst in 2026-H1 (≈ −0.30); positive only in 2025. |
| TGT15 | **Δ ≈ 0, no support** | |
| TGT20 | **Leans positive in the replay, refuted in the study arm** | Replay: +0.055 on all 3 seeds, 15 / 22 months, but the CI crosses 0, the seeds are near-copies, and 2026-H1 is −0.05. Study arm: WILLY −0.035, HT −0.068. Keep observing on live runners; not a candidate. |
| NOLATE, LATE45, LATE60 | **≈ 0, inside noise** | Replay ≈ 0 except 2026-H2 (+0.16 … +0.26). Study WILLY +0.05 / +0.03 / +0.00. No refutation and no support. |
| WILLY vs HT | **No separate exit indicated** | No shadow differs in sign between the two sleeves, except NOLATE / LATE60, which are tiny. HT is weaker as a sleeve (V2b −0.14 vs WILLY +0.14 on the same 2026 days), which is the turnover filter's own effect (DL 341), not an exit effect. |

### BLIND or reconstructed, WILLY

- **HT in the engine replay:** BLIND. Reconstructed only in the non-engine study cohort (R ≥ 1 routing).
- **Spike high H:** from 1m bars (arm A) or ticks, then 1m (arm B). Live uses frenzy_walk's 5m peak, which can differ slightly.
- **Target fee convention:** the walker's net versus the engine's fill fee; at most about 0.03 pts.
- **Entry fidelity:** arm A uses the replay's own fills; arm B uses an 8 s delay + 0.035 % slip.
- **Path ordering:** 1m paths in arm A test the stop first inside a bar touching both lines. This is why reads stay on runners / V2b classes rather than on whole-cohort stop artefacts.
- **WILLY_HT single slot and the WILLY hold (today OFF):** not modelled in arm B.

---

## Answers per item (summary)

| block | item | backtest before → after | refutes? |
|---|---|---|---|
| FADE | ① ADX Δ < 2.359 | blocked +0.073 / +0.020 / −0.019 vs kept +0.040 / −0.051 / −0.013 | **Yes** |
| FADE | ② pair ADX ≤ 19.2 | blocked +0.089 / +0.062 / −0.046 vs kept +0.044 / −0.045 / −0.004 | **Yes** |
| FADE | ⑤ HB ∧ ADX flat | s1 (in-sample) −0.077; s2 / s3 +0.205 / +0.057 | **Yes, out of seed** |
| FADE | ⑥ HB ∧ BTC ADX high | s1 (in-sample) −0.219; s2 / s3 −0.004 / +0.123 | **Yes, out of seed** |
| FADE | ③ mcap ≥ $200M at 1.5× | zone +0.105 / +0.092 / −0.025 (2026-H1 negative); $ −884 / −701 / +48 per 2 y; max DD −8 % | **Does not support** the avg-% trigger; ruin lever only |
| FADE | ④ HB at 1× | HB +0.154 / +0.096 / +0.135 = the best regime; −$2.4–3.5k per 2 y; DD unchanged | **Yes** (refutes the size cut) |
| FLATBASE | 8 exit shadows | live +1.42 (2026 +1.65); every shadow Δ −0.32 … −1.51 with the CI < 0; negative in all 4 halves except TP8_SL10 2025-H1 | **Yes, all 8** |
| WILLY | ladders / FLOOR10 | −0.03 … −0.12 | **Yes** |
| WILLY | TGT20 | replay +0.055 (CI crosses 0); study −0.035 / −0.068 | Not refuted in the replay, refuted in the study |
| WILLY | TGT15, NOLATE, LATE45 / 60 | ≈ 0 | Noise |

**No new year replay was needed.** If the review wants the WILLY-hold-OFF fade cohort, or HT inside the engine, that needs a new yr6-style replay at today's config. The ENGINE_REPLAY_YEAR_PLAN precedent is about 10–12 h wall-clock per 2-year lane per seed from cache (yr6w: 45 runs, about 18 h). Ticks for 2026-10-12 onward would also need fetching: about 20–40k weight, spread over hours at ≤ 1,200 per minute.
