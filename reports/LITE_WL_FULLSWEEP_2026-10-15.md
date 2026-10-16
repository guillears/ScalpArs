# LITE: full winners-vs-losers separator sweep, every variable, 1D + exhaustive 2D (2026-10-15)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing committed, no bot API call, **no Binance call** (cache only).
- Pre-registered in `reports/LITE_WL_FULLSWEEP_PREREG.txt` (frozen 19:43:51 UTC, with input hashes) before any variable × outcome table was computed. The operator's exit-menu addendum was appended (same file, dated section) **before** any menu cell was priced.
- Not reviewed yet, so it supports no arm and no ship.
- Today's LITE exit: +3 resting maker TP / −3 stop / 12 h cap, 10×.

## Plain-English summary

**Question 1: is there anything at entry that tells LITE winners from losers? Not reliably. This time every variable we hold was tested.**
- **187 variables were tested.** By family: **108 market (MACRO), 63 coin (PAIR), 16 FRENZY episode.**
  - 82 are values the engine stamps on the trade, including 29 the previous LITE studies left out.
  - 41 are extra market columns from the 5-minute impulse grids that had not been used before.
  - 17 are new rebuilds:
    - BTC 1h / 4h / 1d RSI, ADX, EMA gaps and returns;
    - the coin's daily RSI, its 1-day and 7-day return, its distance from the 30-day high, its daily-volume rank and its age;
    - the funding rate.
  - 47 come from the earlier rebuilds: 5-minute candles, episode counts and market impulse.
- **Each variable was tested alone and in every pair.** That is 66,256 tests: 568 single-variable, plus 65,688 two-variable quadrants.
  - **Macro × pair** (the block the operator asked for): 31,960 tests.
  - Macro × macro: 22,887. Pair × pair: 10,841.
- **Each test was scored against 500 day-shuffled copies of the data.** The shuffle keeps each day's trades together, so a market variable is counted in days, not trades.
- **Result: nothing beats chance.**
  - The best pattern overall has a family-wise p of 0.32. The cut-off is 0.05.
  - The best pattern in each block:

    | block | family-wise p |
    |---|---|
    | single variable | 0.31 |
    | macro × macro | 0.13 |
    | **macro × pair** | **0.39** |
    | pair × pair | 0.69 |

  - 7 tests reach the "strong" level (S ≥ 2.5). Pure chance gives 3, and up to 19.
  - **Learn on one year, test on the other:** the 10 best 2025 patterns confirmed 0 times in 2026, and the 10 best 2026 patterns confirmed 0 times in 2025. Chance gives about 0.4–0.7.
  - **Tick-priced trades only** (the most accurate prices): best family-wise p 0.88.
- **The strongest near-miss** is "BTC weak over 72 h (r72 ≤ +0.98 %) **and** a small burst of top-10 buying (taker-buy turn above its median)".
  - 2025: 205 fills, 37 % WR, −0.74 %/fill. 2026: 165 fills, 45 % WR, −0.32 %/fill. Breakeven is 50 %.
  - It fails the expectancy bar in 2026 (P 0.91, below 0.95).
  - **The 20 master trades point the other way:** 10 in the zone at +0.60 avg vs 10 outside at 0.00.
  - It is the same "weak BTC + buying blip" family the EPIC study found and its review demoted (Z3). **No new line.**
- **The existing LITE_BTC1H_FALLING observe line** (BTC 1h slope ≤ 0) shows the same split as on Oct-13:
  - 2025: −0.69 %/fill, N 250, P 1.00.
  - 2026: **+0.03 %/fill**, N 229, WR 50.7 % against a 50.2 % breakeven.
  - So it holds in 2025 and 2026-H2, not in 2026-H1. Keep observing; nothing changes.

**Question 2 (operator addition): do some trades want +1/−1 and others +3/−3? We can't tell which at entry.** See the next section.

**Recommendation (operator decides): change nothing on LITE.**
- No entry filter, and no per-trade exit choice.
- No new Scout line either, because the best candidates belong to a family already demoted.
- A "no separator" claim for LITE is now backed by an exhaustive sweep (this report).
- The remaining blind spots are data we do not have: order book, open interest, liquidations (see "Could NOT test").

## Which trades want a tight exit (+1/−1) and which want +3/−3?

**What we did.** We priced 7 exits on every fill's own tick or 1-minute path, using the resting maker TP and live slippage of −0.20 on stops. The menu: +1/−1, +1.5/−1.5, +2/−2, **+3/−3 (today)**, +1/−3, +2/−3, +3/−2, all with today's 12 h cap. We then asked two things:
1. Which entry states predict that tight beats today?
2. Does a rule "if state Z → tight exit, else today" make money out of sample?

**What each exit earns** (%/fill, live-slip ruler):

| block | N | +1/−1 | +1.5/−1.5 | +2/−2 | **+3/−3 today** | +1/−3 | +2/−3 | +3/−2 |
|---|---|---|---|---|---|---|---|---|
| BT 2025 | 828 | −0.150 | **−0.100** | −0.153 | −0.272 | −0.112 | −0.139 | −0.312 |
| BT 2026-H1 | 369 | −0.093 | −0.050 | +0.099 | **+0.206** | −0.016 | +0.184 | +0.073 |
| BT 2026-H2 | 255 | −0.218 | −0.133 | −0.059 | +0.107 | −0.032 | **+0.131** | −0.102 |
| BT tick-only | 888 | −0.136 | −0.074 | −0.010 | **+0.075** | −0.055 | +0.044 | −0.033 |
| MASTER (20) | 20 | −0.320 | +0.058 | −0.101 | **+0.207** | −0.470 | −0.082 | −0.123 |

**1. The answer depends on the period, not on the trade.**
- In losing 2025, every tighter exit loses less than today: +1.5/−1.5 is +0.17 %/fill better.
- In 2026 and on master, today's +3/−3 wins: +1/−1 is −0.31 %/fill worse on 2026 fills.

**2. The perfect per-trade pick is huge, and none of it is reachable.**
- Knowing in advance the best of the 7 exits for each trade would add +1.5 to +1.8 %/fill.
- But "tight beats today" is almost the same thing as "today's trade is a loser". On today's losers, +1/−1 is +2.68 better; on today's winners it is −2.78 worse.
- So this question collapses into Question 1, and Question 1 has no answer at entry.

**3. The screen: same 187 variables, 66,256 tests, day-shuffle null (300).**
- **Tight +1/−1 vs today:** best family-wise p 0.087 overall.
  - Inside the macro × macro block alone it is 0.017. The block also has 8 strong tests where chance gives 1 (p95 7, p 0.043).
  - The top cell is the mirror image of the Question 1 near-miss: weak BTC over 72 h + top-10 buying blip.
  - Macro × pair: p 0.49. Pair × pair: 0.44.
- **+1.5/−1.5 vs today:** overall 0.22. Macro × macro count excess p 0.010; its best cell has p 0.093.
- **Learn on one year, test on the other:** 0/10 and 0/10 (+1/−1); 0/10 and 1/10 (+1.5/−1.5); chance gives about 0.5.
- **Tick-only:** +1.5/−1.5 shows a count excess (32 strong tests vs a chance p95 of 22, p 0.02). It sits on one ETH/SOL lead variable (`sol_lead5`), and its best cell has p 0.097.
- In plain words: **there is a faint market-wide tilt** (weak BTC → tight exits lose less), **too weak to pick out individual trades.** It is the same tilt as in Question 1, not a new signal.

**4. Policy test, walk-forward.** In each fold the top 10 zones were picked on the train data only; each fold also has a logistic model and a depth-2 tree. That gives 36 out-of-sample evaluations in total (12 per fold).

| fold | policies with OOS Δ > 0 at P ≥ 0.90 | OOS Δ range %/fill | in-sample Δ range | in-sample − OOS gap |
|---|---|---|---|---|
| F1 train 2025 → test 2026 | **0 / 12** | −0.218 … −0.023 | +0.18 … +0.29 | +0.23 … +0.51 |
| F2 train 26-H1 → test 26-H2 | **0 / 12** | −0.231 … −0.003 | +0.13 … +0.18 | +0.17 … +0.38 |
| F3 train 2026 → test 2025 | 4 / 12 (best +0.114, P 0.99) | −0.057 … +0.114 | +0.07 … +0.11 | −0.01 … +0.16 |

- **F3 "wins" only because 2025 favoured tight exits across the board.** Using +1.5/−1.5 on every 2025 fill gives +0.17, more than any selected zone.
- The logistic model and the tree lose out of sample in F1 (−0.22 / −0.22) and F2 (−0.23 / −0.03).
- The 20 master trades are negative for most policies.
- **Gate (positive with P ≥ 0.90 in ALL folds, master not negative, tick-only holds, neighbours positive): 0 of 12 policy slots pass.** No Scout line.
- What it would take to try this live: the engine would have to choose the TP/SL per fill at entry from stamped state. It has nothing to choose with.

## 1. Cohort and outcome

| block | N | WR today | avg %/fill (RM0 ruler, = rm_base_pct) |
|---|---|---|---|
| BT 2025 | 828 | 47.1 % | −0.165 |
| BT 2026-H1 / H2 | 369 / 255 | 55.0 / 53.7 % | +0.297 / +0.201 |
| BT tick-only | 888 | – | +0.170 |
| MASTER B18–B21 | 20 | 55 % | +0.297 |
| LIVE (booked) | 3 | DEEP +2.80 · DUSK −2.54 · EPIC −3.43 | −1.06 |

- ZK is in no export (not closed in the 19-34 export), so the live block is 3 fills.
- Live EPIC and DUSK now carry full engine stamps from the 2026-10-15 19-34 live export. The EPIC study did not have them.

## 2. Variables. The complete list, with coverage per block, is in `reports/lite_wlsweep/tables.md` §A and `reports/lite_wlsweep/variables.csv`

| family | engine stamp | earlier rebuild (B 5m candles / C episode / D market) | new grid columns | new BTC multi-TF | new pair daily | funding | total |
|---|---|---|---|---|---|---|---|
| MACRO | 28 | 29 (D) | 41 | 10 | – | – | **108** |
| PAIR | 42 | 14 (B) | – | – | 6 | 1 | **63** |
| EPISODE | 12 | 4 (C) | – | – | – | – | **16** |

**What the EPIC study left out "by rule", now included:**
- **16 `entry_pattern_*` flags**: excluded there by name.
- **4 categoricals**:
  - `entry_macro_trend` (3 levels), `entry_btc_regime` (8), `entry_long_heat_flags` (4): dropped by the "< 10 distinct values" rule;
  - `entry_quality_score` (6): explicitly excluded.
- **The mcap trio** (`entry_mcap_usd`, `entry_vol_mcap_ratio`, `entry_cmc_rank`): 69 % covered in 2025, below the 80 % rule.
- **`entry_liquidity_cap_notional`**: explicitly excluded as sizing.

**How they were handled here:**
- Categoricals are read per level.
- Low-coverage columns are read on their scored subset only; unscored fills are never counted as "rest". These are: mcap trio 0.69 / 0.91; pair age 0.55 in 2025, because the k1d cache starts 2025-01-01, so older coins are censored; funding 2026 only.

**Excluded by pre-declared rule only:** entry price, VWAP price, desired notional, fee, modelled slippage, strategy and order type (constant), `entry_frenzy_catchup` (constant 0), spike time (= te − hours), and the bracket sizing columns.

## 3. Screen results (Y1 = today's P&L %/fill; S = direction-consistent min t over 2025 / 2026)

| run | tests | best S | null median / p95 of max S | family-wise p | # S ≥ 2.5 (obs / null median / p95) | learn 2025 → test 2026 / learn 2026 → test 2025 confirmed (null mean) |
|---|---|---|---|---|---|---|
| Y1 all, 500 shuffles | 66,256 | 2.86 | 2.73 / 3.28 | **0.32** | 7 / 3 / 19 | 0/10 (0.42) · 0/10 (0.68) |
| Y1 1D only | 568 | 2.01 | 1.89 / 2.48 | 0.31 | 0 / 0 / 0 | – |
| Y1 macro × macro | 22,887 | 2.86 | 2.52 / 3.05 | 0.13 | 5 / 1 / 9 | – |
| **Y1 macro × pair** | **31,960** | **2.67** | 2.59 / 3.17 | **0.39** | 2 / 1 / 11 | – |
| Y1 pair × pair | 10,841 | 2.24 | 2.36 / 2.94 | 0.69 | 0 / 0 / 4 | – |
| Y1 tick-only | 63,867 | 2.48 | 2.75 / 3.24 | 0.88 | 0 / 3 / 25 | 0/10 · 0/10 |
| Y2 +1/−1 vs today | 66,256 | 3.11 | 2.73 / 3.24 | 0.087 (macro × macro block 0.017) | 9 / 3 / 25 (macro × macro 8 / 1 / 7, p 0.043) | 0/10 · 0/10 |
| Y2 +1.5/−1.5 vs today | 66,256 | 2.92 | 2.71 / 3.16 | 0.22 (macro × macro block 0.093) | 11 / 3 / 17 (macro × macro 10 / 0 / 7, p 0.010) | 0/10 · 1/10 |
| Y2 +1/−1, tick-only | 63,867 | 2.66 | 2.75 / 3.31 | 0.65 | 5 / 3.5 / 24 | 1/10 (p 0.37) · 0/10 |
| Y2 +1.5/−1.5, tick-only | 63,867 | 3.11 | 2.75 / 3.21 | 0.097 (macro × macro 0.027) | **32 / 3 / 22, p 0.020** | 2/10 (p 0.07) · 0/10 |

**Top Y1 tests.** All three are "zone worse", direction-consistent in 2025, H1 and H2 — and none is a survivor:

| zone | 2025 N · in / out avg · t | 2026 N · in / out avg · t | S | family-wise p |
|---|---|---|---|---|
| BTC r72 ≤ 0.98 ∧ top-10 taker-buy turn > −0.0039 | 205 · −0.74 / +0.02 · −3.24 | 165 · −0.31 / +0.46 · −2.86 | 2.86 | 0.32 |
| BTC off-24h-high ≤ −0.83 ∧ same turn | 192 · −0.68 / −0.01 · −2.76 | 172 · −0.30 / +0.47 · −2.89 | 2.76 | 0.44 |
| *(macro × pair top)* range_pos12 > 0.55 ∧ BTC 4h RSI > 53.9 → better | 199 · +0.31 / −0.33 · +2.67 | 163 · +0.78 / +0.07 · +2.67 | 2.67 | 0.59 |

More top-8 lists per block, for every outcome, are in `tables.md` §B.

**Reconciliation with earlier work.**
- **YR6_WL (Oct-13).** BTC 1h slope ≤ 0 has S = 1.44 here: t +3.37 in 2025 and +1.44 in 2026, but H1 −0.02 and H2 +2.37. The Oct-13 study discovered within 2025 and confirmed on H2; this sweep demands both years, and H1 breaks it. The two results are consistent.
- **EPIC study.** Its Z3 (BTC 1h RSI ≤ 55 ∧ buy-turn) appears here as a macro × macro cell with S 2.35, family-wise p ≈ 1. The new top cell is a sibling: r72 or off-24h instead of RSI.
- **sweep_separators.py copy (LITE option, `sweepsep_LITE.txt`).** It has no null, so 132/363 1D and 10,354 2D cells come out "consistent". Its top hits are small-N artefacts:
  - `entry_ema50_slope ≤ 0`: N 29 / 28 (these are winners, +0.52 / +1.29);
  - `btc_off24lo > med ∧ dlo_24h ≤ med`: N 5 / 5.
- That is why this study ranks by a day-shuffle null.

## 4. Filter bar (pre-registered for survivors only: there are none. Post-hoc read for the operator)

| zone | year | N | days | WR | breakeven | avg | P(avg < 0) | Δ if refused | LOMO min | winners refused | bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r72 ≤ 0.98 ∧ buy-turn up | 2025 | 205 | 121 | 37.1 % | 50.0 % | −0.738 | 1.00 | +0.183 | +0.153 | 76 | pass |
| | 2026 | 165 | 100 | 44.8 % | 50.2 % | −0.315 | **0.91** | +0.083 | +0.051 | 74 | **fail** |
| LITE_BTC1H_FALLING (slope ≤ 0) | 2025 | 250 | 140 | 38.4 % | 49.8 % | −0.690 | 1.00 | +0.208 | +0.159 | 96 | pass |
| | 2026 | 229 | 131 | 50.7 % | 50.2 % | +0.030 | 0.42 | −0.011 | −0.084 | 116 | **fail** |

- **Top day or pair share of the zone loss:** ≤ 0.04 for every zone. No concentration.
- **Master + live, refute-only:**
  - r72 ∧ buy-turn: master zone 10 fills, 60 % WR, +0.60 vs rest 10 fills, 50 %, 0.00. **This points the wrong way.** Live DUSK is in the zone (−2.54).
  - Slope ≤ 0: master + live N 6, avg −0.03 vs +0.17 for the rest. This agrees, but carries no weight.

## 5. Could NOT test (blind spots)

- **Stamp columns that are empty on every LITE fill** (37). LITE never fills them in the replay, the master exports or the live export:
  - `entry_br_*` breadth-door stamps (9), `entry_bear_*` (5), `entry_surge_*` (4), `entry_pair_rank`, `entry_pair_age_days`, `entry_funding_rate`;
  - `entry_gap_5_20(_prev)_signed_pct`, `entry_gap_5_8_signed_pct`, `entry_btc_4h_ema50_200_gap_pct`, `entry_pair_1d_ndi`;
  - the catch-up and WILLY fields, `entry_chop_burst_prior_fill_s`, `entry_gap_expand_marginal`, `entry_btc_regime_started_at`.
  - The full list is in `notstamped.csv`.
- **Rebuilt in their place:** pair age, funding, BTC 4h EMA50/200, multi-TF BTC. Breadth comes from the grids.
- **Live-only stamps:** `entry_book_spread_pct`, `entry_book_drift_pct`, `entry_fill_vs_book_pct`, `entry_order_lag_s` exist on 2 live fills only.
- **Not in any cache:** order-book depth and imbalance, open interest, liquidations, long/short ratio, spot-vs-perp basis, listing or news events.
- **Coverage gaps:**
  - funding only from about 2025-12-30 (2026 read only);
  - pair age censored for coins listed before 2025-01-01;
  - mcap trio 69 % in 2025 (delisted coins);
  - k1d / funding caches only hold coins still listed (survivorship);
  - master has no `k_all` / `prior_*` and 65 % `gvol`;
  - live has no episode family; grid columns `br_up5` / `agg_tbr*` / `br_acc5` are missing for the live fills.
- **Design limits:**
  - 2D uses median quadrants only, with no 3-way interactions;
  - pattern flags are tagged PAIR, although some mix BTC state;
  - fills only: refused signals and the slot a refused fill frees are not modelled;
  - the 564 2025 non-top-50 fills use 1-minute paths, so tick-only is shown separately;
  - DUSK's 1-minute live path does not resolve +3/−3 (it stopped live at −2.54), so DUSK's today value is its booked P&L.

## 6. Every headline two ways (`reports/lite_wlsweep/verify.txt`)

- **Screen.** 305 tests × 2 years each for Y1 and +1/−1, recomputed by an independent pandas loop: **0 mismatches** (t and N).
- **Menu.** +3/−3 equals LITE_PROTECT `rm_base_pct` on all 1,452 BT fills, max |diff| 0. The way-2 event loop (`walk_loop`) was asserted on every fill × cell. The best-exit counts were recounted with stdlib csv.
- **Policy.** Δ for F1 P1 / LOGIT and F3 P1 / LOGIT recounted from the row file, exact.
- **LITE_BTC1H_FALLING.** Stdlib recount −0.6897 / +0.0300, matching.
- **`scripts/validate_against_master.py`: ALL CHECKS PASS.**

## Exact commands

```
venv/bin/python scripts/study_lite_wlsweep_prereg.py                    # pre-registration (refuses overwrite); --addendum appended the exit menu once
venv/bin/python -W ignore scripts/study_lite_wlsweep_menu.py            # A1: 7-exit menu on every path (27 s, 9 jobs) → lite_wlsweep/menu.csv
venv/bin/python -W ignore scripts/study_lite_wlsweep_build.py           # variable matrix + registry + coverage → matrix.csv, variables.csv, notstamped.csv
venv/bin/python -W ignore scripts/study_lite_wlsweep_screen.py --y y --R 500 [--tick]     # Y1 screen + null + discover/confirm
venv/bin/python -W ignore scripts/study_lite_wlsweep_screen.py --y D1 --R 300 [--tick]    # Y2 +1/-1 vs today
venv/bin/python -W ignore scripts/study_lite_wlsweep_screen.py --y D15 --R 300 [--tick]   # Y2 +1.5/-1.5 vs today
venv/bin/python -W ignore scripts/study_lite_wlsweep_policy.py          # A3 walk-forward policies + models → policy.csv, policy_rows.csv
venv/bin/python -W ignore scripts/study_lite_wlsweep_tables.py          # inventory, top tests per block, master/live tables, post-hoc bar → tables.md
venv/bin/python -W ignore scripts/study_lite_wlsweep_sweepsep.py LITE   # copy of sweep_separators.py with a LITE option → sweepsep_LITE.txt
venv/bin/python -W ignore scripts/study_lite_wlsweep_verify.py          # way-2 recounts → verify.txt
venv/bin/python -W ignore scripts/validate_against_master.py
```

Shared code is `scripts/study_lite_wlsweep_common.py`. Outputs are in `reports/lite_wlsweep/`:
- `screen_*.csv`: every test;
- `null_*.csv`, `summary_*.json`;
- `menu.csv`, `policy*.csv`, `tables.md`, `variables.csv`, `notstamped.csv`, `verify.txt`.
