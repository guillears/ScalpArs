# FRENZY winners vs losers: full-variable screen, 2026-10-19

**Operator asks:**
- "Analyse deeply all FRENZY longs here and in the master batch, see if you find a pattern of winners vs losers, make sure you analyse absolutely ALL variables we have available. Then if there are findings cross check vs backtest."
- Priority update: FRENZY_LONG first. Is there a winner cohort worth sizing up? What size does the sleeve deserve? What does its blocker role do?

**Scope:** research only. No code, config or git changes.

**Scripts:**
- `scripts/study_frenzy_wl_screen_journal.py`: journal union.
- `scripts/study_frenzy_wl_screen_build.py`: real-fill pool and features.
- `scripts/study_frenzy_wl_screen.py`: screen, null, OOS and backtest.
- `scripts/study_frenzy_wl_screen_theory.py`: theory family and existing observe lines.
- `scripts/study_frenzy_wl_screen_long.py`: LONG sizing, Pattern-W search and blocker role.

`validate_against_master.py` passes on every check. The pool's master rows are exactly the kept, non-probe FRENZY rows of `MASTER_POOL_stacked.csv` (STACK 2026-10-19a), with `y = stack_pct`.

## 0. Data and blocks

| block | source | outcome y (% of notional, fee-net) |
|---|---|---|
| ① live B23 | `scalpars_orders_live_2026-10-19_14-54-01.csv` (10-16 20:00 → 10-19 14:50, not yet archived) | booked `pnl_percentage` (today's live config) |
| ② master | `MASTER_POOL_stacked.csv`, kept non-probe, batches B16 to B22 | `stack_pct` = today's exits (LONG +1.5/−3 · LITE C +2 or +3/−3 · WILLY +1.25/−4/90 with late TP) |
| ③ backtest | yr6 2-year engine replay `ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv`. Seed 1 is the analysis seed; seeds 2 and 3 are a re-computation of the headline numbers only. Only days **< 2026-10-03** are used, so the real-fill era is out of sample. | `pct_new`: LONG +1.5/−3 · LITE **+3/−3** (before LITE C) · WILLY +1.25/−4/90 (**no late TP**) |

The journal is the union of 75 decisions exports in `~/Downloads` plus the archived B22 journal. It spans 09-29 → 10-19. Its order-book (BOOK) minute rows start 10-03 16:00. These rows are joined to each fill at the last snapshot ≤ the open minute, within 2 min and in the same paper/live mode.

| sleeve | ① live N | ② master N (batches) | real total | real days | ③ backtest N (< 10-03) |
|---|---|---|---|---|---|
| FRENZY_LONG | 3 | 26 (B16 1 · B17 5 · B19 11 · B20 3 · B21 3 · B22 3) | **29** | 13 | 617 |
| FRENZY_LITE | 4 | 24 (B18 2 · B19 13 · B20 1 · B21 4 · B22 4) | **28** | 12 | 1,427 |
| FRENZY_WILLY | 4 | 27 (B19 14 · B20 3 · B21 2 · B22 8) | **31** | 9 | 1,482 |

**The binding constraint:** each sleeve has about 30 real fills on 9 to 13 days. No real-fill cohort can reach the N ≥ 15 / ≥ 8-window filter bar, or the N ≥ 30 Pattern-W bar. Real data can only *generate* hypotheses here. Confirmation has to come from the backtest at a frozen cut, plus forward observation.

## 1. Method (binding rules applied)

**Theory first (§5).** Each sleeve has 5 or 6 mechanism hypotheses with a fixed predicted direction.
- Honest disclosure: the theory list was written after the 1D screen had printed.
- Three items overlap the screen's top rows and are **not pre-registered**: L2 ADX, T1 above-VWAP share, and W3 BTC 1h slope (W3 is an existing line).

**Coverage first** (`_coverage.csv`):
- Per sleeve: 106 to 114 variables (76–83 stamped `entry_*`, 11–12 derived `x_*`, 19 journal `j_*`).
- Coverage per block is recorded. Unscored fills are excluded from both zone and rest.
- Low-coverage variables:
  - `entry_book_*` / `entry_order_lag_s`: 21–25 %, live only.
  - `entry_frenzy_gvol`: 38–64 %.
  - WILLY journal book: 68 %; its 15-min book change only 13 %, because WILLY pairs are flagged minutes before entry.
- Unstamped on FRENZY fills (0 %): `entry_br_*` breadth door, `entry_pair_rank`, `entry_gap_5_20_*`. All `manual_ob_*` columns are 0 % on FRENZY fills.

**Screen:**
- **1D:** every scored variable at sign / median / low tercile / high tercile, zone vs rest. 281–306 tests per sleeve.
- **2D:** exhaustive all-pairs median quadrants over variables with ≥ 70 % coverage. 10,331–17,960 tests per sleeve.
- **Statistic:** pooled-variance t on pct. A second outcome is the stop-hit indicator. Welch t was dropped because all-TP zones have zero variance at N ≈ 30.

**Null:** day-block shuffle (each UTC day's outcome vector is swapped with a random donor day). 2,000 draws for 1D, 300 for 2D. Reported: the familywise p of max |t|, and the count of |t| ≥ 3 hits (≥ 3.5 for 2D).

**Out of sample:**
- Real date halves, and master → live.
- Then the backtest at the **same numeric cut**: per regime, top-50 split, and day-bootstrap P.

**Also checked:**
- Market-wide variables are read in day units.
- Per-pair and per-day loss share.
- Expectancy bar (sleeve breakeven WR, 95 % window bootstrap, ≥ 8 windows, no window or pair ≥ 50 %, N ≥ 15).
- Pattern-W bar.

## 2. Screen vs null: nothing survives in any sleeve

| sleeve | layer | tests | max \|t\| obs | null 95 % of max \|t\| | familywise p | \|t\|≥3 (2D ≥3.5) hits obs | null mean / 95 % |
|---|---|---|---|---|---|---|---|
| LONG | 1D | 294 | 2.92 | 5.35 | 0.82 | 0 | 3.8 / 13 |
| LONG | 2D | 17,960 | 5.61 | 7.45 | 0.37 | 43 | 63 / 205 |
| LITE | 1D | 306 | 2.55 | 5.51 | 0.98 | 0 | 3.0 / 8 |
| LITE | 2D | 17,213 | 5.24 | 8.53 | 0.49 | 18 | 48 / 131 |
| WILLY | 1D | 281 | 3.25 | 5.46 | 0.61 | 1 | 3.7 / 14 |
| WILLY | 2D | 10,331 | 4.73 | 8.33 | 0.71 | 10 | 44 / 150 |

The stop-hit outcome gives the same picture: familywise p 0.53–0.97. The hit counts are **below** the null mean everywhere.

**On ~30 fills over 9–13 days, the real data contain no winner/loser separator distinguishable from luck.** The null is wide because the days are few and outcomes cluster within a day. That is the honest null for this sample.

The top tests were then cross-checked in the backtest anyway (top 12 1D + top 8 2D per sleeve, `_oos_backtest.csv`), so the operator's question gets a direct answer. Results are in §3–§5.

## 3. FRENZY_LONG (priority)

### 3.1 The sleeve at today's +1.5 / −3: what size it deserves

Breakeven WR at +1.5/−3 is about 66.7 %. CIs are day-bootstrap 95 %.

| block | N | days | WR | BE WR | avg %/fill | 95 % CI | P(avg>0) | Σ pts | worst day pts |
|---|---|---|---|---|---|---|---|---|---|
| ① live B23 (booked) | 3 | 2 | 67 % | 68 % | −0.059 | [−2.95, +1.39] | 0.24 | −0.18 | −2.95 |
| ② master B16 | 1 | 1 | 0 % | – | −3.019 | – | – | −3.02 | −3.02 |
| ② master B17 | 5 | 2 | 60 % | 67 % | −0.301 | [−0.75, −0.00] | 0.00 | −1.51 | −1.50 |
| ② master B19 | 11 | 4 | 73 % | 67 % | +0.271 | [−0.43, +0.75] | 0.75 | +2.98 | −1.51 |
| ② master B20 | 3 | 1 | 100 % | – | +1.500 | – | – | +4.50 | +4.50 |
| ② master B21 | 3 | 2 | 67 % | 67 % | −0.001 | [−3.00, +1.50] | 0.24 | −0.00 | −3.00 |
| ② master B22 | 3 | 1 | 67 % | 61 % | +0.215 | – | – | +0.65 | +0.65 |
| **② master ALL (today's exit)** | **26** | 11 | 69 % | 66 % | **+0.139** | [−0.53, +0.66] | 0.66 | +3.60 | −3.02 |
| ② master ALL *as traded* (old lock / +3 exits) | 26 | 11 | 35 % | 55 % | −1.112 | [−2.03, −0.19] | 0.01 | −28.92 | −9.02 |
| **①+② real ALL** | **29** | 13 | 69 % | 66 % | **+0.118** | [−0.57, +0.64] | 0.64 | +3.42 | −3.02 |
| ③ yr6 s1 2025 BULL | 270 | 149 | 66 % | 67 % | −0.061 | [−0.32, +0.19] | 0.32 | −16.5 | −10.7 |
| ③ yr6 s1 2026-H1 BEAR | 229 | 106 | 63 % | 67 % | −0.179 | [−0.47, +0.09] | 0.09 | −40.9 | −9.1 |
| ③ yr6 s1 2026-H2 BULL | 124 | 67 | 62 % | 68 % | −0.273 | [−0.71, +0.14] | 0.10 | −33.9 | −13.8 |
| **③ yr6 s1 ALL (2025-01 → 2026-10-11)** | **623** | 322 | 64 % | 67 % | **−0.146** | [−0.33, +0.02] | **0.04** | −91.3 | −13.8 |
| ③ yr6 s2 / s3 ALL | 614 / 613 | 318 | 64 % | 67 % | −0.156 / −0.158 | [−0.33, +0.01] | 0.03 | −95.5 / −97.1 | −13.8 |
| ③ yr6 s1 top-50 pair-days | 303 | 206 | 68 % | 67 % | +0.071 | [−0.17, +0.31] | 0.73 | +21.7 | −6.1 |

**Caveat: the real +0.118 is mostly counterfactual.**
- In the master, 9 of the 26 LONG rows are stop-outs credited +1.5 from their recorded peak. Another 5 are +3 winners cut to +1.5.
- Only **9 LONG fills actually traded under the +1.5 exit** (B21 → B23): 6 W / 3 L, **+0.05 %/fill**.

**Size verdict: KEEP BLOCKER SIZE.**
- Expectancy is ≤ 0 on 1,850 backtest fills: three seeds, all three regimes negative, P(avg>0) 0.03–0.04. So the Kelly size is zero. Nothing supports sizing the sleeve up.
- At blocker size (median notional $99.8):
  - its own two-year cost is about **−$91**
  - its worst LONG day is **−$14**
- At the old 2× ($1,414 notional):
  - two-year cost **−$1,290**
  - worst LONG day **−$196**
- Concurrency: 30 % of LONG fills overlap an open LITE or WILLY on another pair. Sizing up would add correlated risk on the days FRENZY already loses. The worst FRENZY-family day is −$2,951 to −$2,971 on the fixed $2,900 book, with LONG at 2× or at blocker size; that day is driven by LITE/WILLY, not LONG.

### 3.2 Winner cohort for a multiplier (Pattern-W): none

1. **Backtest discovery (2025), confirmation in 2026 at the frozen cut** (`_long_patternW_*.csv`):
   - 522 cells tested (every scored variable × 4 cuts × both sides).
   - 25 pass N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 ∧ Σ > 0 in 2025. That is **fewer than the day-block null (mean 77, 95th 198, p 0.75)**.
   - **0 of the 25 pass in both 2026-H1 and 2026-H2.** Only `x_pair_fz_fills_prev7d > 2` passes H1 (58 · 72 % · +0.255), and it fails H2 (18 · 56 % · −0.504).
2. **Real-fill side.** No cell can reach N ≥ 30, because the sleeve has 29 fills.
   - 291 cells have N ≥ 10 ∧ WR ≥ 70 % ∧ avg ≥ +0.10. At the same cut in the backtest, **only one passes Pattern-W**: `entry_slippage_pct ≤ −0.17` (BT 129 · 70.5 % · +0.193).
   - That is an **execution measurement known only after the fill**, and slippage in the replay is simulated. It is not an entry rule and cannot carry a size.
3. **Best-looking real cohort: LONG with pair ADX ≤ 29.45** (median cut). Real 15 · 87 % · +0.94.
   - Backtest at that cut: 209 · **67 %** · **+0.006**. That fails WR ≥ 70 % and avg ≥ +0.10.
   - The BE-compatibility check is moot because there is no cell to stack.

### 3.3 The one consistent LONG separator, and why it must NOT become a filter

**Finding:** pair **ADX > 29.45** (mature trend) is where FRENZY_LONG loses.

| block | zone N · days · WR · avg | rest N · WR · avg |
|---|---|---|
| real (master + live) | 15 · 10 d · 53 % · −0.614 | 14 · 86 % · +0.903 |
| BT 2025 | 184 · 65 % · −0.101 | 86 · 67 % · +0.025 |
| BT 2026-H1 | 145 · 59 % · −0.371 | 84 · 70 % · +0.153 |
| BT 2026-H2 | 65 · 62 % · −0.355 | 39 · 59 % · −0.352 (no separation) |
| **BT ALL < 10-03** | **394 · 254 d · 62.2 % (BE 67.3) · −0.242, CI [−0.47, −0.03], P(avg<0) 0.986**; top pair 6 % / top day 3 % of loss | 209 · 67 % · +0.006 |
| BT top-50 pair-days | 169 · 64.5 % · −0.101 (P 0.73) | 122 · 74 % · +0.317 |

On the backtest, the zone formally clears the expectancy bar: WR below breakeven, P 0.986, more than 8 windows, no concentration, N ≥ 15.

It is **not a ship**, for three reasons:
1. **The cut is in-sample.** The real familywise p is 0.82 to 0.99. It vanishes in the newest regime (H2), and it is not significant inside the top 50.
2. **At blocker size it is worth almost nothing:** 394 × −0.242 % × $100 ≈ **$95 over two years**.
3. **It would destroy the blocker.** In the yr6 LONG/WIDE-off A/B, the LITE trades that a LONG/WIDE kept out were 21 (−30.6 pts at LITE size). **17 of those 21 were blocked by a holder with ADX > 29.45**: 29 % WR, −1.45 %/fill, Σ −24.6 pts (`_long_blocked_lite_by_holder_adx.csv`). A LONG ADX filter would release exactly those bad LITE entries.

The mechanism agrees with YR6_LWOFF_AB: a LONG fails on a coin → the LITE on that coin fails too, and a high-ADX LONG is the one that fails.

### 3.4 Blocker role: how many bad LITE entries it actually blocks

**Backtest (yr6 LONG/WIDE-off A/B, two years):**
- 21 LITE trades were kept out by a LONG/WIDE holder (19 LONG, 2 WIDE): 29 % WR, −1.46 %/fill.
- At today's +1.5 LONG TP, 18 stay blocked, and the trades kept out are worth **−$1,952 at LITE 10×** (≈ −$1,560 at today's 8×).
- That is about 0.85 blocks per month.

**Real journals, 09-29 → 10-19:**
- **0 LITE refusals by a FRENZY_LONG-held pair.**
- The only 2 `FRENZY_LITE_PAIR_HELD` rows:
  - BAT 10-10 07:40, held by its own earlier LITE
  - STRK 10-08 14:45, holder not in any bot archive (B18 excluded 14 MANUAL rows)
- `FRENZY_PAIR_HELD_BY_LITE`: 0.
- The block is a rare, fat-tail insurance: it has not fired once in three real weeks.

**Size does not affect the block.** The engine refuses LITE when the pair holds *any* open position (`services/trading_engine.py`, the `{_bk}_PAIR_HELD` branch). Size only changes LONG's own P&L.

**The exit does affect the block.** The earlier +1.5 TP frees the pair sooner (3 of the 21 were freed). Any future LONG exit change must re-price the blocked LITE.

### 3.5 FRENZY_LONG recommendation

**Keep blocker size (invest 0.2 · lev 0.05 → 1×). No filter. No multiplier cell.**
- There is no Pattern-W cohort.
- The whole sleeve is negative in every backtest regime.
- The only separator (ADX) points at the very coins where the LONG earns its keep by blocking LITE.
- The existing LONG_NONTOP50 observe line remains the right watch. It is consistent here: top-50 pair-days +0.07 vs all −0.15.

**Optional, observe-only, never a ship:** log the holder ADX on the LITE blocks the LONG causes. The scout already sees `FRENZY_LITE_PAIR_HELD`.

## 4. Top real-fill separators cross-checked in the backtest (frozen cuts)

`_oos_backtest.csv` has every row (halves · master/live · BT per regime and top-50). Selected rows:

| sleeve | test (real cut) | real zone vs rest | BT regimes in same direction | BT ALL zone vs rest (P zone < rest) | verdict |
|---|---|---|---|---|---|
| LONG | ADX ≤ 24.39 (T1) | 10 · 100 % · +1.50 vs −0.61 | 3/3 | +0.003 vs −0.196 (0.15) | no Pattern-W (BT 67 %) |
| LONG | BTC 5m RSI-prev6 > 55.87 | −1.22 vs +0.82 | 2/3 | −0.189 vs −0.131 (0.61) | refuted |
| LONG | BTC 1d ret < 0 | 8 · −1.31 vs +0.66 | 1/3 | −0.099 vs −0.188 | refuted (bearish-day block already exists) |
| LONG | book: bid wall ≤ 1.09 % away | 13 · −0.59 vs +1.20 | – | not in backtest | blind spot |
| LITE | frenzy hours ≤ 7 | 10 · 90 % · +2.00 vs −0.54 | 1/3 | −0.022 vs +0.014 | refuted |
| LITE | pair volume ratio ≤ 0.56 | +2.00 vs −0.54 | 0/3 | wrong direction | refuted |
| LITE | book depth +15 % in 15 min | 8 · −1.51 vs +1.03 | – | not in backtest | blind spot |
| **LITE** | **above-VWAP share ≤ 58.97 ∧ RSI ≤ 57.92 (2D)** | **6 · 0 % · −3.00 vs 22 · 77 % · +1.29** | **3/3** | **−0.252 vs +0.102 (0.978)** | **observe-only (§6)** |
| WILLY | vol/mcap ≤ 0.268 | 16 · 100 % · +1.13 vs −0.98 | 1/3 | −0.167 vs −0.057 (wrong direction) | refuted |
| WILLY | cmc rank > 188 | −1.05 vs +0.66 | 2/3 | −0.222 vs +0.090 (0.99), but top-50 +0.016 vs −0.026 | small-cap artefact zone (2025 fidelity) |
| WILLY | BTC 1h slope ≤ 0 | 11 · 3 d · 64 % · −0.83 vs 90 % · +0.63 | 3/3 | −0.246 vs −0.000 (0.97); top-50 −0.043 vs +0.018 | existing WILLY_BTC1H line |

## 5. Theory family (`_theory.csv`; predicted direction fixed; real-median cut frozen into the backtest)

| id | hypothesis (zone worse unless stated) | real zone vs rest | real P(pred) | BT regimes in predicted direction | BT ALL |
|---|---|---|---|---|---|
| L1 | LONG pair RSI > 58.9 (exhaustion) | 14 · 50 % · −0.75 vs +0.93 | 0.996 | 1/3 | −0.135 vs −0.179 ✗ |
| L2* | LONG ADX > 29.45 (mature trend) | 14 · −0.77 vs +0.94 | 0.983 | **3/3** | −0.242 vs +0.006 (P 0.92 one-sided diff) |
| L3 | LONG stretch vs spike VWAP > 3.4 % | −0.12 vs +0.34 | 0.72 | 1/3 | ✗ |
| L4 | LONG BTC 5m RSI > 51.8 | −0.43 vs +0.63 | 0.96 | 2/3 | −0.171 vs −0.141 ✗ |
| L5 | LONG thin pair (vol24 ≤ median) | +0.03 vs +0.22 | 0.61 | 2/3 | −0.217 vs −0.013 (≈ LONG_NONTOP50) |
| L6 | LONG BTC down day | 8 · −1.31 vs +0.66 | 0.998 | 1/3 | ✗ |
| T1* | LITE weak VWAP hold (above share ≤ 59) | 14 · 43 % · −0.72 vs +1.46 | 0.98 | 2/3 | −0.042 vs +0.038; top-50 reversed ✗ |
| T2 | LITE BTC 1h slope ≤ 0 (existing line) | 8 · **+0.81** vs +0.19 | 0.27 | 2/3 | −0.355 vs +0.169 (H1 reverses) |
| T3 | LITE stretched gap > 1.55 | **+1.32** vs −0.58 | 0.03 | 1/3 | ✗ |
| T4 | LITE volume fading | **+0.93** vs −0.19 | 0.03 | 0/3 | ✗ |
| T5 | LITE stale episode > 9.6 h | −0.26 vs +1.00 | 0.95 | 1/3 | ✗ |
| W1 | WILLY turnover high inside R < 1 | 15 · −0.98 vs 16 · 100 % · +1.13 | 1.00 | 1/3 | ✗ (BT says the opposite) |
| W2 | WILLY small cap | −0.19 vs +0.42 | 0.80 | 1/3 | ✗ |
| W3* | WILLY BTC 1h slope ≤ 0 (existing) | 11 · 3 d · −0.83 vs +0.63 | 0.99 | **3/3** | −0.246 vs −0.000 |
| W4 | WILLY deep first red bar (better) | +0.27 vs −0.07 | 0.70 | 1/3 | ✗ |
| W5 | WILLY high pair ATR | +0.21 vs +0.02 | 0.44 | 1/3 | ✗ |

\* = overlaps the screen's top rows (not pre-registered).

16 hypotheses were tested. Only 2 hold their direction in all three backtest regimes (L2, W3). Both are discussed in §3.3 and §6. **The real-fill "obvious" patterns mostly fail out of sample**:
- RSI exhaustion
- BTC down day
- WILLY turnover 16/16
- LITE early-episode

## 6. LITE and WILLY (secondary)

**Sleeve levels (real / backtest < 10-03):**
- **LITE**
  - Real 28 · 12 d · 61 % (BE 54) · +0.370 [−0.62, +1.57]. Live B23 is 4 · 75 % · +0.73.
  - BT 1,427 · 49 % (BE 49) · −0.005 [−0.17, +0.15]; 2025 −0.142 · H1 +0.226 · H2 +0.117. These are BT outcomes at +3/−3, not LITE C.
- **WILLY**
  - Real 31 · 9 d · 81 % (BE 79) · +0.107 [−0.78, +0.73]. Live B23 is 4 · 1 d · 50 % · −1.60 (ALT and DUSK −4 stops on 10-17, both BTC 1h slope ≤ 0).
  - BT 1,482 · 68 % (BE 70) · −0.108 [−0.23, +0.01]; 2025 −0.238 · H1 +0.101 · H2 +0.072; top-50 −0.007.

**FRENZY_LITE: one new observe-only line candidate, LITE_WEAKHOLD.**

*Definition:* `entry_frenzy_above_share` ≤ 58.97 ∧ `entry_rsi` ≤ 57.92. Both are stamped on 100 % of fills and present in the backtest.

*Theory:* a coin that has spent less than ~60 % of its 5m closes above the spike VWAP, and whose RSI is not strong, is distributing. The 12-close reclaim is then a dead-cat leg.

*Evidence:*

| block | zone | rest |
|---|---|---|
| real | 6 fills · 5 d · 4 pairs (SKL ×2, ZK ×2, BAT, C98) · **6 / 6 stops, −3.00**; top pair 33 % / top day 34 % of loss | 22 · 77 % · +1.29 |
| BT 2025 | 264 · 43.6 % · −0.375 [−0.75, −0.00] | −0.033 |
| BT 2026-H1 | 100 · 46 % · +0.010 | +0.306 |
| BT 2026-H2 | 69 · 42 % · −0.160 | +0.237 |
| **BT ALL** | **433 · 294 d · 43.9 % (BE 48.7) · −0.252 [−0.55, +0.05], P(avg<0) 0.950**; top pair 2 % / top day 2 % | 994 · 50.6 % · +0.102 |
| BT top-50 pair-days | 152 · 50 % · −0.002 (**no effect inside the top 50**) | +0.185 |

*Why only observe:*
- It is the third-ranked quadrant of 17,213 on the real fills (familywise p 0.88).
- It was picked from 20 candidates sent to the backtest.
- The backtest mean sits exactly on the P 0.95 edge and is 2025-driven.
- It vanishes inside the top 50 (the 2025 fidelity flag).
- The backtest exit is +3/−3, not LITE C.
- With a 30–50 % haircut, the sleeve gain would be about +0.04–0.05 %/LITE fill.

*It overlaps in spirit with LITE_OFF30H3* ("bought well below the 30-min high"). That line could not be scored here (no 30-min-high stamp), so the overlap is unknown. Report it at the first read.

*🔒 Proposed frozen gate (operator decides; nothing armed):*
- **Cohort:** FRENZY_LITE fills opened after this report (2026-10-19 15:00 UTC). The 6 motivating fills never count.
- **Outcome:** booked pct under LITE C.
- **Read:** at zone N ≥ 15 on ≥ 8 UTC days.
  - **FILTER CANDIDATE** only if:
    - zone WR < LITE breakeven WR (same-period LITE fills)
    - ∧ day-bootstrap P(zone avg < 0) ≥ 0.95
    - ∧ no day or pair ≥ 50 % of the zone loss
    - then the 30–50 % haircut and a tight revert gate.
  - **RETIRE** if zone avg ≥ rest avg at N ≥ 15.
- Never re-fit the cuts.

*Existing LITE lines on all real fills (descriptive; each line's own gate counts from its deploy, via its scout):*
- LITE_BTC1H_FALLING zone: 8 · 6 d · 75 % · **+0.81** vs +0.19. Real fills contradict it so far.
- LITE_STRETCH_REENTRY (approx.): 5 · 60 % · +0.49 vs +0.34.
- GVOL ≥ 1.0: 4 · +1.37 vs +0.07 (10 unscored).

**FRENZY_WILLY: nothing new.**
- The best real separator, low turnover = 16 / 16 winners, is **refuted** by the backtest: the zone is worse there in 2 of 3 regimes.
- The only consistent one is the existing **WILLY_BTC1H** line (DL 275).
  - Real: slope ≤ 0 is 11 fills but **only 3 days** (10-12, 10-15, 10-17; 10-17 = 50 % of the loss) · 64 % · −0.83 vs 20 · 90 % · +0.63.
  - BT 3/3 regimes, ALL −0.246 vs 0.000 (P 0.994).
  - Inside the top 50 it is −0.043 vs +0.018: no effect.
  - It is a market-wide on/off switch, counted in day units; 3 days ≪ 8.
  - **Keep observing on its frozen gate. No change.**
- The DIP_SUB descriptive read: deep first red bar 14 · 86 % · +0.51 vs 17 · 77 % · −0.22.

## 7. Blind spots (what could not be tested and why)

- **N.** About 30 real fills per sleeve on 9–13 days. Every real-fill effect is below the 8-window gate. The null's 95th-percentile max |t| (5.3–8.5) means a single real cell would need an extreme effect to register.
- **Order-book / journal features (19 `j_*`).** Real fills only (WILLY 68 %, WILLY 15-min change 13 %). They are not in any backtest, so book findings (LONG bid wall, LITE depth surge) cannot be cross-checked. **LITE_OB_THIN keeps its own forward gate.**
- **`entry_book_*`, `entry_order_lag_s`, `entry_slippage_pct`.** Live-only execution stamps (≈ 23 %), post-decision, not entry rules.
- **Unstamped on FRENZY fills:**
  - `entry_br_*` breadth (so WILLY_MKTDIP's top-50 breadth is unscored here)
  - `entry_pair_rank` (LONG_NONTOP50 needs daily volumes; `x_log_vol24` is the proxy)
  - `entry_gap_5_20_*`
  - the 30-min high (LITE_OFF30H3)
  - funding history
- **Exit mismatch in the backtest.** LITE at +3/−3 (LITE C untested here); WILLY without the late TP or maker TP; LONG/WIDE replay held to +3 before re-pricing to +1.5 (second-order: earlier pair release is not modelled).
- **2025 non-top-50 fidelity.** Every backtest effect here that is 2025-driven fades inside the top 50: LITE_WEAKHOLD, WILLY_BTC1H, cmc rank.
- **Real outcomes are partly counterfactual.** Master `stack_pct` re-prices old exits: LONG +1.5 from peak, LITE C, WILLY late TP. Only 9 LONG fills (B21 → B23) and a minority of the LITE and WILLY fills (LITE C since 10-15, WILLY late TP since DL 297) actually traded under today's exits.
- **Real-era concurrency features** (`x_open_positions_at_entry` etc.) count master and live rows only (no manual fills).
- **yr5 replay was not used.** Its period is covered by yr6 (newer code, priced to today); memory: no yr4/pre-fix citations.

## 8. Recommendations (plain English)

| sleeve | recommendation |
|---|---|
| **FRENZY_LONG** | **Keep blocker size. No filter, no multiplier.** No winner cohort passes Pattern-W in the backtest. The sleeve loses −0.15 %/fill over two years in every regime and seed, so it deserves no size of its own. Its value is blocking bad LITE (≈ $1.6k per two years at LITE 8×); size does not change that. The one loser separator (ADX > 29.45) must not be used: 17 of 21 bad LITE blocks come from exactly those high-ADX LONGs. |
| **FRENZY_LITE** | **Nothing to ship.** Register (operator's choice) one observe-only line, **LITE_WEAKHOLD** (above-VWAP share ≤ 58.97 ∧ RSI ≤ 57.92), with the frozen gate in §6. It is a low-prior candidate, and dead inside the top 50. Existing lines unchanged. |
| **FRENZY_WILLY** | **Nothing new.** The real-fill "low turnover = all winners" pattern is refuted by the backtest. WILLY_BTC1H is the only consistent signal: keep it on its frozen gate (3 of 8 days so far). |

**CSVs** (`reports/FRENZY_WL_SCREEN_2026-10-19_*`):
- `coverage`
- `tests_1d`
- `tests_2d_top400`
- `null`
- `oos_backtest`
- `theory`
- `existing_lines`
- `long_sleeve`
- `long_concurrency`
- `long_patternW_cells`
- `long_patternW_null`
- `long_real_cells`
- `long_journal_pairheld`
- `long_blocked_lite_by_holder_adx`
