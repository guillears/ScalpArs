# WILLY turnover filter — re-priced under today's live WILLY entry and exit (2026-10-19)

Research only. No code, config or test was touched and nothing was committed. Files written:
- `scripts/study_willy_turnover_reprice.py` (backtest re-price)
- `scripts/study_willy_turnover_reprice_live.py` (live blocked rows and live fills)
- `scripts/study_willy_turnover_reprice_analyze.py` (statistics)
- `reports/WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv`, `reports/WILLY_TURNOVER_REPRICE_2026-10-19_live.csv` and `reports/WILLY_TURNOVER_REPRICE_2026-10-19_grid.csv` (data)
- `reports/WILLY_TURNOVER_REPRICE_2026-10-19_vol24cache.json` (API cache)

`scripts/validate_against_master.py` returned **ALL CHECKS PASS** before the analysis.

Binance use: 15 public 1h-kline requests (weight 1 each). The last X-MBX-USED-WEIGHT-1M reading was 16. The shared cooldown was checked first, and there were no 418 or 429 responses. Everything else came from the local cache.

## Decision

**The filter's frozen revert gate fired** (first 15 blocked entries averaged +0.153 %), so the rule book says **set 0**. Re-pricing the evidence under the exit WILLY actually runs today makes the filter look **stronger, not weaker**:

| read | blocked (R ≥ 1) | kept (R < 1) |
|---|---|---|
| Old study: immediate entry, +1 / −3 / 60 | 477 · −0.189 % | 1,093 · +0.080 % |
| **Today's live entry and exit**: first red 5m bar ≤ 60 min, +1.25 maker TP, late +0.25 from minute 30, −4 stop, 90-min cap | **477 · 73 % WR · −0.237 % [−0.42, −0.06] · P(avg < 0) 0.996** | **1,093 · 79 % · +0.084 % [−0.03, +0.20]** |
| Live, re-walked under today's exit | 15 · 80 % · +0.130 % [−0.70, +0.87] | 31 · 81 % · +0.104 % (walker) / −0.084 % (booked) |

**My recommendation is to keep `frenzy_willy_max_vol_mcap_ratio` at 1.0, for every market cap, as a transparently declared discipline override of the fired gate,** paired with a new, properly powered revert gate (§7). Don't restrict it to a market-cap band. The operator decides.

Why I'm comfortable overriding the gate:
- **The re-priced cohort passes every item of the locked expectancy bar.** Blocked WR is 72.5 % against a breakeven of 77.3 %. Avg < 0 at P 0.996 on a day-block bootstrap. It spans 217 windows (days). The top pair carries 3 % of the loss and the top day 2 %. N = 477.
- **It also beats a shuffled-label null across the whole 12-cell grid** (within-day null p 0.004).
- **The live read cannot tell the two stories apart.** If the backtest were exactly right, a 15-fill blocked average would still land above 0 **36 % of the time**, and at ≥ +0.13 **28 % of the time**. The gate was a coin flip.
- **The live blocked cohort is concentrated.** GTC and EPIC are 7 of the 15 entries, and EPIC alone is 2 of the 3 losses.

**The honest counterweight.** The 1,570 backtest rows are the same rows that generated the idea. Only the pricing is new, so this is still in-sample: apply the 30–50 % haircut. The R = 1 cut was also picked after a scan in the original study (scan null p ≈ 0.05).

## 1. Theory first: why would turnover above market cap hurt a buy-the-spike-dip long?

WILLY buys the first red 5m candle after a fresh FRENZY spike, betting the dip gets bought back to +1.25 within 90 minutes. R = 24 h futures volume ÷ market cap > 1 means the coin's entire market value changed hands in a day, almost all of it on leveraged perps.

1. **Distribution, not re-rating.** A spike on low turnover means a few buyers moved a thin float, and dips get bought by the same flow. A spike after a day of above-float turnover means the move is already crowded. The holders who wanted out have liquidity to sell into, so the first red candle is more often the start of the unwind than a pause.
2. **Leverage overhang.** Futures volume far above market cap means open interest is large relative to spot depth. After the spike, late longs are stopped or liquidated into the dip, and that cascade runs straight through WILLY's −4 stop. Data: the blocked side hits the −4 stop on **24.5 %** of fills vs **17.4 %** on the kept side, with almost the same TP rate. The damage is all in the tail.
3. **Why market cap could matter.** For small caps (< $50M), R > 1 is the normal state of a pumped meme or low float, so the filter sorts "churned and crowded" from "quietly re-rated". Large caps rarely reach R > 1; when they do, it is usually a real catalyst (listing, news) with deep books, which could plausibly continue. The operator's AGT intuition is that case. **The data does not show the large-cap exemption.** Per fill, the blocked side is negative in every bucket (§3), and the ≥ $200M bucket has only 7 fills in 9 months, which can't be judged either way. The filter is mostly a small-cap filter by construction: 80 % of blocked fills are < $50M, and Spearman(R, market cap) = −0.79. It still separates *inside* the < $50M bucket (+0.17 vs −0.19), so it is not just a size proxy.

## 2. Method

| item | detail |
|---|---|
| Cohort | `reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv`, cohort `willyA`: 1,781 first-flag events, Jan 10 → Oct 7 2026, 393 pairs. t0 = the flag bar's close (`study_vol_mcap_cohorts.py`). The old study entered immediately at t0 + 8 s (`study_vol_mcap_willy_price.py`), so **it did not model the red-candle wait**. Here it is modelled. |
| Entry (live rule, `trading_engine._frenzy_willy_eval`) | Armed on the trigger bar. Opens at the first **closed red 5m bar** (close < open; the trigger bar counts) whose open is ≤ armed + 60 min, using `k5m_full`. Entry is the first aggTrades print ≥ red close + 8 s, × 1.00035 slip. The dislocation guard refuses if the move is > 1 % (48 refused). Wait bars: median 1, 0–12. Every event found a red bar inside 60 min. Not modelled: episode-end expiry, global hold, slot. |
| R at the red bar | The engine reads R at the red bar. vol24 = Σ qvol of the 288 closed 5m bars to the red close. Market cap = the study's mcap at t0 × (red close ÷ trigger close). Spearman vs the study's R is 0.997, and 20 of 1,570 rows change side. |
| Exit | Read from `trading_config.json`: `+1.25/-4/90/late30@+0.25/live`. Walked with `scout_willy_timecap.evaluate_late(book="live")` on `study_b19_fzexit_common.load_path`, the same walker `build_willy_exit_cf.py` uses for the master pool. A TP books exactly the line (resting maker TP). The late TP books the line, or the print if already through it at minute 30. The stop is the first print through −4. The cap is the last print at ≤ 90 min. Net = gross − 0.045 − 0.045 × r. |
| Ticks vs 1m | **100 % ticks**: 1,733 / 1,733 backtest paths and 46 / 46 live paths, with 0 exits decided on a 1m bar. One entry used the 1m open. The tick-only sensitivity is therefore identical to the main read (it drops 1 row). |
| Scored | 1,733 priced, of which 1,570 have R and market cap. The 163 unscored fills average +0.114 and are never counted as "rest". |
| Stats | Day-block bootstrap (4,000 draws) for CIs and P(avg < 0); windows = UTC days. Breakeven WR = \|avg loss\| ÷ (avg win + \|avg loss\|) on the whole priced cohort under today's exit = **77.3 %** (avg win +1.087, avg loss −3.701). |

**Parity of the walker vs booked live exits.**
- Since the DL 296 / 297 deploy (exit identical to today's), there are 5 fills. The walker gives the same exit class 5 / 5, mean |Δ| 0.07, Σ −10.51 vs −10.25 booked.
- It reproduces the master's `WILLY_EXIT_CF.csv` on all 27 shared rows. The max |Δ| is 0.057 (ICNT 10-16, now on ticks instead of 1m).
- The walker's stops book the first print, about −4.00 to −4.06. Live booked −3.79 to −3.93, so the walker is slightly harsher (conservative).

## 3. Main tables (backtest, today's entry and exit)

**A. Blocked vs kept.** CIs are day-block 95 %. "Top pair / day share" is the largest single pair's or day's share of the cohort's total loss.

| cohort | N | days | WR | avg % | 95 % CI | P(avg<0) | top pair / day share of loss |
|---|---|---|---|---|---|---|---|
| all scored | 1,570 | 270 | 77 % | −0.014 | [−0.11, +0.08] | 0.61 | 2 % / 2 % |
| kept R < 1 | 1,093 | 268 | 79 % | +0.084 | [−0.03, +0.20] | 0.08 | 2 % / 2 % |
| **blocked R ≥ 1** | **477** | **217** | **73 %** | **−0.237** | **[−0.42, −0.06]** | **0.996** | 3 % / 2 % |

**B. Per market-cap bucket.**

| bucket | kept N · WR · avg | blocked N · days · WR · avg [CI] · P(avg<0) | top pair share of the blocked loss |
|---|---|---|---|
| < $50M | 332 · 81 % · +0.168 | **382 · 195 · 74 % · −0.189 [−0.39, +0.02] · 0.97** | 4 % |
| $50–100M | 284 · 79 % · +0.062 | 67 · 59 · 72 % · −0.273 [−0.79, +0.22] · 0.85 | 11 % |
| $100–200M | 206 · 79 % · +0.052 | 21 · 21 · 57 % · −1.006 [−2.01, −0.00] · 0.98 | 22 % |
| ≥ $200M | 271 · 77 % · +0.027 | 7 · 7 · 71 % · −0.253 [−1.76, +1.25] · 0.64 | 50 % (noise) |

**C. Consistency** (kept vs blocked avg %):
- **2026-H1:** +0.046 (741) vs **−0.262** (259).
- **2026-H2:** +0.163 (352) vs **−0.208** (218).
- **< $50M by half:** H1 +0.148 vs −0.116, H2 +0.206 vs −0.261.
- **By month:** blocked is worse than kept in **8 / 10 months**. The two reversals are Jan (+0.51 vs −0.37, N 24) and Oct (+0.29 vs +0.08, N 11).
- **Leave-one-month-out:** the blocked avg stays between −0.21 and −0.28 for R ≥ 1, and between −0.10 and −0.24 for R ≥ 1 with < $50M.
- **No 2025 data:** the WILLY A cohort only covers 2026, so the requested 2025 half could not be read.

**D. What changed the result: the entry or the exit?** Blocked vs kept, same rows:

| pricing | kept | blocked | Δ |
|---|---|---|---|
| old: immediate entry, +1 / −3 / 60 | +0.080 | −0.189 | +0.268 |
| immediate entry, today's exit | +0.117 | −0.200 | +0.317 |
| red-bar entry, today's exit (= live) | +0.084 | −0.237 | +0.321 |

The separation survives both the red-candle wait and the new exit, and slightly widens.

## 4. R-threshold × market-cap grid (blocked side)

Columns:
- **N / days / WR / avg / P(avg < 0):** the blocked side's own stats.
- **gain/fill:** −Σ blocked ÷ all 1,570 fills, i.e. the change in average P&L per WILLY A fill if this cell is blocked.
- **P(gain ≤ 0):** from the day-block bootstrap.

| R cut | mcap scope | N | days | WR | avg | P(avg<0) | gain/fill [CI] | P(gain≤0) | H1 / H2 avg |
|---|---|---|---|---|---|---|---|---|---|
| **1.0** | **all** | **477** | 217 | 72.5 % | **−0.237** | 0.996 | **+0.072 [+0.019, +0.127]** | 0.005 | −0.26 / −0.21 |
| 1.0 | < $50M | 382 | 195 | 73.6 % | −0.189 | 0.960 | +0.046 [−0.002, +0.097] | 0.030 | −0.12 / −0.26 |
| 1.0 | < $100M | 449 | 210 | 73.3 % | −0.201 | 0.986 | +0.058 [+0.003, +0.111] | 0.020 | −0.22 / −0.18 |
| 1.5 | all | 303 | 180 | 71.6 % | −0.257 | 0.983 | +0.050 | 0.018 | −0.29 / −0.22 |
| 1.5 | < $50M | 247 | 156 | 73.3 % | −0.169 | 0.886 | +0.027 | 0.110 | −0.14 / −0.20 |
| 1.5 | < $100M | 285 | 175 | 73.0 % | −0.186 | 0.922 | +0.034 | 0.079 | −0.23 / −0.14 |
| 2.0 | all | 201 | 149 | 74.1 % | −0.131 | 0.810 | +0.017 | 0.195 | −0.29 / +0.04 |
| 2.0 | < $50M | 161 | 122 | 75.2 % | −0.077 | 0.669 | +0.008 | 0.328 | −0.22 / +0.06 |
| 2.0 | < $100M | 192 | 144 | 75.5 % | −0.059 | 0.642 | +0.007 | 0.366 | −0.20 / +0.09 |
| 3.0 | all | 105 | 91 | 75.2 % | −0.061 | 0.606 | +0.004 | 0.390 | −0.14 / +0.01 |
| 3.0 | < $50M | 89 | 78 | 75.3 % | −0.049 | 0.580 | +0.003 | 0.426 | −0.09 / −0.01 |
| 3.0 | < $100M | 101 | 88 | 77.2 % | +0.043 | 0.425 | −0.003 | 0.588 | +0.02 / +0.07 |

- **The best cell is the one already live** (R ≥ 1, every market cap). Every market-cap restriction *lowers* the gain, because the $50–200M blocked fills also lose.
- **Raising the cut to 1.5 / 2 / 3 lowers the gain monotonically.** The loss is in R 1–2, not only the extreme tail. The earlier decile table agrees: it is not a "very high turnover only" effect.
- **Shuffled-label null for the best cell** (max gain over all 12 cells, P&L shuffled 2,000×):
  - Observed +0.072 per fill.
  - Within-day shuffle: **p = 0.004** (null 95th percentile +0.048).
  - Global shuffle: p = 0.002.
  - The same p for the single pre-registered R ≥ 1 cell.
- **In-sample haircut (30–50 %):** gain is about **+0.036 to +0.050 % per WILLY A fill**, or about **−0.12 to −0.17 %** avoided per blocked entry.

## 5. Live (re-walked under today's full exit, ticks for every row)

| cohort | N | days | WR | avg % | 95 % CI | P(avg<0) | concentration |
|---|---|---|---|---|---|---|---|
| blocked, first 15 (the gate cohort) | 15 | 8 | 80 % | **+0.130** | [−0.70, +0.87] | 0.36 | GTC 4, EPIC 3; one pair carries 67 % of the loss (EPIC) |
| same, the scout's price (no late TP) | 15 | 8 | 80 % | +0.153 | [−0.71, +0.91] | 0.34 | the gate's number |
| same, +0.035 % entry slip | 15 | 8 | 80 % | +0.129 | | | |
| let-through WILLY fills, walker | 31 | 9 | 81 % | +0.104 | [−0.75, +0.71] | 0.38 | |
| let-through WILLY fills, booked | 31 | 9 | 74 % | −0.084 | [−0.84, +0.52] | 0.60 | earlier exits / paper eras |

**Live per market cap.** Blocked market cap = 24 h futures volume (24 closed 1h klines) ÷ R, the DL 314 method.

| bucket | blocked | let-through (walker) |
|---|---|---|
| < $50M | 12 · 75 % · −0.150 | 4 · 50 % · −1.396 |
| $50–100M | 2 · +1.250 (RLC B, OGN) | 5 · 60 % · −1.254 |
| $100–200M | 0 | 13 · 100 % · +1.173 |
| ≥ $200M | 1 · +1.250 (STRK, R 1.06) | 9 · 78 % · −0.020 |

Every live cell has N ≤ 13 on ≤ 7 days, so these are anecdotes. The live blocked cohort (+0.13) and let-through cohort (+0.10) cannot be told apart. Under the backtest's blocked distribution (sd 2.2 % per fill), a 15-fill mean of ≥ +0.13 happens 28 % of the time.

## 6. Expectancy-bar check (CLAUDE.md, filter = Pattern C)

| item | R ≥ 1, all market caps (backtest, today's exit) | R ≥ 1, < $50M only | live 15 |
|---|---|---|---|
| ① WR < breakeven 77.3 % | ✅ 72.5 % | ✅ 73.6 % | ❌ 80 % |
| ② avg < 0 at 95 % window bootstrap | ✅ P 0.996 | ✅ P 0.960 (CI touches +0.02) | ❌ P 0.36 |
| ③ ≥ 8 windows, no window / pair ≥ 50 % of the loss | ✅ 217 days, 3 % / 2 % | ✅ 195 days, 4 % / 3 % | ❌ one pair is 67 % |
| ④ N ≥ 15 | ✅ 477 | ✅ 382 | ✅ 15 |
| null (multiple-test) | p 0.004 (12-cell grid) | (inside the same grid) | n/a |
| haircut gain per WILLY fill | +0.036 to +0.050 | +0.023 to +0.032 | |

## 7. Recommendation

1. **Keep `frenzy_willy_max_vol_mcap_ratio` = 1.0 for every market cap.** This is a transparently declared override of the fired DL 251 revert gate, which by the book says set 0. That gate was written because the evidence was priced under an old exit. Under the exit WILLY really runs, the evidence passes the full expectancy bar, and the gate's 15-fill read had a 36 % chance of firing even if the filter works exactly as backtested. Expected value is about +0.04 % per WILLY A fill after the haircut. Blocking also frees the global WILLY hold for other sleeves, a second-order gain that is not priced here.
2. **Don't restrict it to small caps.** "< $50M only" and "< $100M only" both cut the gain (+0.046 / +0.058 vs +0.072), because blocked $50–200M fills lose too (−0.27, −1.01). Above $200M the filter fires about once a month (7 in 9 months). Exempting that bucket is neither supported nor needed, and AGT-type blocks are rare.
3. **Don't raise the cut.** R ≥ 1.5 / 2 / 3 each lower the gain, and from 2.0 on the gain is not significant.
4. **New pre-committed revert gate.** It is counted from the operator's decision, excludes the 15 already read, and both cohorts are walked by the scout under the live exit (ticks first).
   - **REVERT (set 0)** at the first prefix of **N ≥ 30 fresh rule-blocked entries on ≥ 10 days**, if the blocked mean is ≥ the mean of the let-through WILLY fills opened in the same period. This is a paired, same-window comparison, so a good or bad tape day hits both sides.
   - **CONFIRMED** if the blocked mean is ≤ the let-through mean − 0.20.
   - Otherwise hold, and re-read once at N ≥ 60.
   - Calibration from the backtest distributions: if the filter works as backtested, this gate fires wrongly about 25 % of the time at N 30, about 17 % at N 60, and about 35 % with the haircut. If the filter is useless it fires 50 %. That makes it tighter than standard, as the override rule requires. WILLY's per-fill variance (sd about 2.2 %) means no live gate at this N can be decisive. The gate can only catch a filter that is clearly *anti*-useful.
5. **If the operator prefers the rule book (set 0):** the turnover-blocked scout then sees nothing. Instead, track R ≥ 1 WILLY fills that *open* as an observe line, with the frozen re-arm bar from DL 251 (WR < breakeven, P(avg < 0) ≥ 0.95, ≥ 8 days, no pair or day ≥ 50 %, N ≥ 15). Note that with WILLY's variance that bar needs roughly 100+ fills to have a chance of passing. The backtest expects about −0.24 % per R ≥ 1 fill until then.

## 8. Blind spots (what this could not test)

- **In-sample.** The same 1,570 rows that suggested the filter; only the pricing changed. The original R = 1 cut came after a scan (null p ≈ 0.05). The p = 0.004 here is the 12-cell grid null, not a fresh out-of-sample test.
- **No 2025.** The WILLY A cohort is Jan–Oct 2026 only. The halves read is 2026-H1 vs 2026-H2.
- **Unsequenced.** The WILLY global hold, 1 slot, pair-held, the bearish block (WILLY has none) and the cold-pair rule are not applied. **Episode-end expiry** (the pair unflags before a red bar) is not modelled.
- **Market cap.** Uses constant supply × price (the study's checked assumption, median 1.007 vs stamps). 72 pairs have no bot-source market cap (unscored, survivorship: delisted coins are missing). Live blocked market cap is reconstructed as volume ÷ R. **Live UNREAD refusals** (10 in 9 days, market cap missing, fail-closed) are a separate block not judged here; the backtest unscored fills average +0.114.
- **Fill assumptions.** A resting TP fills on a touch (no queue). The TP touch is judged on taker-fee net, so it is slightly conservative vs maker. Entry slip is 0.035 %. Walker stops book the first print (−4.00 to −4.06) vs live −3.79 to −3.93.
- **Live sample.** 15 blocked and 31 let-through fills on 9 days, with GTC / EPIC repeats. It cannot confirm or refute anything at WILLY's variance.


## Post-review note (DECISION_LOG 314, 2026-10-19)
The live blocked rows' 24 h volume was first summed from 24 CLOSED 1h bars, which leaves out the spike hour (rebuilt mcap 0.23–0.27× the
engine's own stamp on ERA / GTC). `study_willy_turnover_reprice_live.py` now uses the 288 5m bars closed before the refusal's bar (same as
the scout's tb_vol24). Re-run: every blocked row stays in the same mcap bucket (< $50M 12 · −0.15 % · $50–100M 2 · +1.25 · ≥ $200M 1 ·
+1.25 on the live walker), so the live split above is unchanged in counts; individual mcaps moved (ERA $8.6M → $32M, GTC 10-12 $8.1M →
$35.6M). Against the engine's stamp-implied supply: 13 / 14 within ±20 %, median 0.96×. The backtest cohort's R / mcap come from the 10-08
study (bot-source supply × price) and are not affected.
