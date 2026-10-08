# 7-sleeve shared compounding portfolio (operator study, 2026-10-08)

**Validation first:** `scripts/validate_against_master.py` → **ALL CHECKS PASS**.
**Scope:** read-only on code and config. No bot API, no commit, no Binance calls (all inputs were cached).
**Code:** `scripts/study_port7_portfolio.py`. It imports `scripts/study_tp34_portfolio.py` (fills, sizing schedules, caps, metrics) and `scripts/study_yr5_halves_today.py` (replay frame with today's filters) read-only.
**Output:** `reports/PORTFOLIO_7SLEEVE_2026-10-08.csv` (one row per variant × scenario × seed), `_sleeves.csv` (per sleeve) and `_skips.csv` (capacity skips).

## Plain-English answer

1. **The 5-sleeve run is reproduced exactly.** End balance $29,857 [19,324 … 38,397], max drawdown −48.6 %. This confirms the harness matches.
2. **Adding MOM_LONG at today's size, priced by the yr5 replay, wipes out the book in every fade variant.** The book falls 93–97 % in H1.
   - The replay prices MOM_LONG at **−0.067 %/trade**. That is 555 fills per seed at about 1.25× invest and 20×.
   - Each fill costs about −0.4 % of the book, and the losses stack up across 555 fills.
   - This is the same −0.058 %/trade drain the halves study found (`YR5_HALVES_TODAY_STACK_2026-10-08.md`: −$6,433 on a flat $3k book). Compounding makes it worse.
3. **SPIKE_FADE depends entirely on the calibration.**
   - **F1** (calibrated +0.04): adding fades to the 5-sleeve book gives **$35,991** (+1,100 %). That beats the 5-sleeve's $29,857, but the drawdown deepens to **−66 %**.
   - **F0** (as-is, −0.01): **$22,079**, worse than without fades.
   - **F2** (stress −0.05): **$7,203**, much worse.
   - The fades' spread across seeds is large in every variant.
4. **Capacity is not the issue.** Adding MOM_LONG and the fades skips **zero** extra FRENZY or BULLRUN fills under F0 and F1. The skip counts are identical to the 5-sleeve run: BULLRUN 11.0 per seed on global slots; LITE 15 / 13 / 2.3. The new sleeves hold slots only briefly. The damage is purely their P&L.
5. **Sensitivity (not evidence): MOM_LONG would need about +0.05 %/trade or better to add value.**
   - At an ML mean of 0.00, the book ends at $23k with a −94 % drawdown. That is worse than the 5-sleeve run because H1 ML is still negative after a uniform shift.
   - At +0.05: $47k, drawdown −80 %.
   - At +0.10: $89k, drawdown −66 %.
   - Live master ML runs well above that (baseline v20 ML 37 · 86 %). However, the replay reproduces only ~64 % of live ML, and the operator asked for replay pricing.

## Headline table (mean of 3 seeds [range])

"Daily compound (cal.)" is over 273 calendar days. "Per trading day" counts days with at least one entry. "Worst day" is the mean over seeds, with the dates.

| Variant | Scenario | End balance | Total return | Daily compound (cal.) | Per trading day | Max DD | Worst day | H1 · H2 return |
|---|---|---|---|---|---|---|---|---|
| F0 fades as-is | **ALL 7** | **$342** [65 … 634] | −89 % [−98 … −79] | −0.92 % [−1.39 … −0.57] | −0.93 % (272 d) | −98.9 % | −50.0 % (08-15 / 08-24 / 09-04) | H1 −97 % · H2 +488 % [−19 … +1,403] |
| F0 | without BULLRUN | $147 [67 … 266] | −95 % | −1.16 % | −1.16 % (272 d) | −98.9 % | −49.4 % | H1 −97 % · H2 +158 % |
| F0 | without LITE | $25 [24 … 26] | −99 % | −1.74 % | −2.70 % (187 d) | −99.2 % | −38.9 % | H1 −98 % · H2 −49 % |
| F0 | without MOM_LONG | $22,079 [10,341 … 37,127] | +636 % [+245 … +1,138] | +0.68 % [+0.45 … +0.93] | +0.71 % (262 d) | −83.6 % [−91.5 … −77.2] | −32.2 % (04-15 / 05-14 / 06-08) | H1 −56 % · H2 +1,860 % |
| F0 | ALL 7 without heat re-admits | $392 [67 … 782] | −87 % | −0.89 % | −0.90 % | −98.9 % | −50.0 % | H1 −96 % · H2 +496 % |
| **F1 calibrated +0.04** | **ALL 7** | **$6,643** [501 … 9,960] | +121 % [−83 … +232] | +0.07 % [−0.65 … +0.44] | +0.07 % (272 d) | **−98.0 %** | −37.4 % (02-17 / 05-14 / 08-24) | H1 −93 % · H2 +5,334 % [+230 … +13,264] |
| F1 | without BULLRUN | $1,291 [517 … 2,291] | −57 % | −0.37 % | −0.37 % | −98.0 % | −39.5 % | H1 −93 % · H2 +1,136 % |
| F1 | without LITE | $1,364 [114 … 2,328] | −55 % | −0.50 % | −0.51 % (268 d) | −98.8 % | −40.5 % | H1 −96 % · H2 +1,833 % |
| F1 | **without MOM_LONG** | **$35,991** [21,177 … 44,257] | +1,100 % [+606 … +1,375] | **+0.90 %** [+0.72 … +0.99] | +0.93 % (262 d) | −66.3 % [−74.3 … −52.4] | −28.0 % (05-14) | H1 +1 % · H2 +1,217 % |
| F1 | ALL 7 without heat re-admits | $6,800 [510 … 10,476] | +127 % | +0.08 % | +0.08 % | −98.0 % | −37.4 % | H1 −93 % · H2 +4,656 % |
| F2 stress −0.05 | **ALL 7** | **$24** [24 … 25] | −99 % | −1.75 % | −2.70 % (186 d) | −99.2 % | −39.0 % | H1 −98 % · H2 −50 % |
| F2 | without BULLRUN | $25 | −99 % | −1.75 % | −2.70 % | −99.2 % | −39.0 % | H1 −98 % · H2 −50 % |
| F2 | without LITE | $25 | −99 % | −1.75 % | −3.44 % (139 d) | −99.2 % | −35.4 % | H1 −99 % · H2 −30 % |
| F2 | without MOM_LONG | $7,203 [490 … 14,883] | +140 % [−84 … +396] | +0.07 % [−0.66 … +0.59] | +0.07 % (262 d) | −93.9 % | −35.6 % | H1 −81 % · H2 +1,722 % |
| (fade-free) | without SPIKE_FADE (6 sleeves, MOM_LONG kept) | $2,141 [744 … 3,551] | −29 % [−75 … +18] | −0.19 % [−0.51 … +0.06] | −0.19 % (268 d) | −87.5 % | −40.3 % (08-24) | H1 −81 % · H2 +256 % |
| (fade-free) | **5-sleeve (without MOM_LONG and SPIKE_FADE)** — reproduces the 5-sleeve study | **$29,857** [19,324 … 38,397] | +895 % [+544 … +1,180] | +0.83 % [+0.68 … +0.94] | +0.96 % (236 d) | −48.6 % | −20.5 % (05-14) | H1 +91 % · H2 +421 % |

The brief's "without SPIKE_FADE (= the 5-sleeve run)" row is really 6 sleeves, because MOM_LONG stays in. Both rows are shown above. The 5-sleeve numbers are reproduced to the dollar.

**Sensitivity (F1 fades; MOM_LONG pnl % shifted uniformly, not evidence).** This answers "what per-trade MOM_LONG result would this book need?":

| ML mean set to | End balance | Total return | Daily compound (cal.) | Max DD | H1 · H2 |
|---|---|---|---|---|---|
| replay as-is (−0.063) | $6,643 [501 … 9,960] | +121 % | +0.07 % | −98.0 % | H1 −93 % · H2 +5,334 % |
| 0.00 | $23,435 [11,469 … 36,099] | +681 % | +0.72 % | −93.9 % | H1 −79 % · H2 +5,956 % |
| +0.05 | $47,279 [23,571 … 70,396] | +1,476 % | +0.98 % | −80.1 % | H1 −30 % · H2 +2,892 % |
| +0.10 | $89,342 [53,911 … 118,517] | +2,878 % | +1.23 % | −65.8 % | H1 +181 % · H2 +1,009 % |
| +0.20 | $179,446 [137,986 … 226,391] | +5,882 % | +1.50 % | −52.2 % | H1 +904 % · H2 +492 % |

## Per sleeve inside the ALL-7 book (mean of 3 seeds)

Columns: offered / taken · WR · avg % · net $ [range] · H1 $ · H2 $.

**F1 (calibrated fades):**

| Sleeve | Offered / taken | WR | avg % | Net $ [range] | H1 $ · H2 $ |
|---|---|---|---|---|---|
| BULLRUN | 147 / 135.7 | 49.4 % | +0.215 | +5,993 [−154 … +10,310] | 0 · +5,993 |
| BEARRUN | 13 / 13.3 | 67.0 % | +0.035 | +4 [−15 … +27] | −12 · +16 |
| FRENZY_LONG | 188 / 185.0 | 51.2 % | +0.069 | +444 [−477 … +1,249] | −61 · +505 |
| FRENZY_WIDE | 106 / 106.3 | 51.1 % | +0.088 | −97 [−321 … +229] | +123 · −221 |
| FRENZY_LITE | 548 / 517.7 | 54.1 % | +0.254 | +2,358 [+1,794 … +3,197] | +863 · +1,494 |
| MOM_LONG (replay) | 561 / 555.3 | 61.2 % | **−0.067** | **−4,380** [−5,342 … −2,823] | −2,960 · −1,421 |
| MOM_LONG heat re-admits ᵃ | 37 / 32.7 | 76.2 % | −0.003 | −254 [−370 … −48] | −349 · +95 |
| SPIKE_FADE | 394 / 394.0 | 76.7 % | +0.039 | −424 [−2,281 … +1,691] | −409 · −16 |

**F0 (fades as-is):**

| Sleeve | Offered / taken | WR | avg % | Net $ [range] | H1 $ · H2 $ |
|---|---|---|---|---|---|
| BULLRUN | 147 / 135.7 | 49.4 % | +0.215 | +1,013 [−23 … +1,770] | 0 · +1,013 |
| BEARRUN | 13 / 13.3 | 67.0 % | +0.035 | −10 | −11 · +1 |
| FRENZY_LONG | 188 / 185.0 | 51.2 % | +0.069 | −129 | −25 · −104 |
| FRENZY_WIDE | 106 / 106.3 | 51.1 % | +0.088 | +80 | +92 · −12 |
| FRENZY_LITE | 548 / 516.7 | 54.1 % | +0.252 | +862 | +733 · +129 |
| MOM_LONG (replay) | 561 / 554.3 | 61.2 % | −0.068 | −2,870 | −2,637 · −233 |
| MOM_LONG heat re-admits ᵃ | 37 / 32.7 | 76.2 % | −0.003 | −266 | −339 · +73 |
| SPIKE_FADE | 394 / 393.7 | 76.3 % | −0.010 | −1,338 [−1,975 … −842] | −711 · −627 |

**F2:** the book is ruined in H1, so FRENZY, BULLRUN and the rest mostly see a dead book. See `_sleeves.csv`.

The $ figures are small because the book is near zero. Read the avg % column, not the $. Positive-avg sleeves (FRENZY, BULLRUN) earn little $ here because they trade a book that MOM_LONG has already drained.

ᵃ The heat file gives only `n_seeds` (1–3) per signal. Seed k (k = 0, 1, 2) takes the signals with `n_seeds` > k, which gives 68 / 34 / 8 signals, about 37 in the window. The expected count equals the halves study's weight. Exit time = entry + the minutes in `how`. There is no 24 h volume for these signals, so no liquidity cap is applied.

## Capacity skips per seed

| Sleeve | Reason | 5-sleeve run | ALL 7 · F0 | ALL 7 · F1 |
|---|---|---|---|---|
| BULLRUN | global slots | 11.0 | 11.0 | 11.0 |
| FRENZY_LONG | pair held | 3.0 | 3.0 | 3.0 |
| FRENZY_LITE | sleeve slots / pair held / global | 15.0 / 13.0 / 2.3 | 15.0 / 13.0 / 2.3 | 15.0 / 13.0 / 2.3 |
| MOM_LONG | pair held / global | — | 4.0 / 1.3 | 4.0 / 1.3 |
| MOM_LONG re-admit | global / pair held | — | 3.0 / 1.0 | 3.0 / 1.0 |
| SPIKE_FADE | any | — | 0 (0.7 "no balance") | 0 (0.3 "no balance") |

**Extra FRENZY / BULLRUN fills skipped versus the 5-sleeve run: 0 under F0 and F1.**
- MOM_LONG and the fades hold slots only briefly and rarely overlap the FRENZY / BULLRUN bursts.
- The only "losses" in F0 and F1 are about 1 LITE fill and a few ML / fade fills that hit "no balance" while the book was nearly empty.
- In F2 (and F0 without LITE) hundreds of fills are "no balance". That means the book is ruined (ticket < $5), not short of capacity.

## Method (what was added to the 5-sleeve rules)

- **Same book and rules** as `study_tp34_portfolio.py`:
  - $3,000, 2026-01-04 → 2026-10-04, compounding on realized closes.
  - Equal split (equity − reserve schedule − fee reserve) / 4.
  - Leverage schedule 20× / 15× above $25k.
  - 4 global slots. Caps: FRENZY 2 · WIDE 2 · LITE 2 · BULLRUN 4.
  - One position per pair. ≤ 3 entries per pair-day per FRENZY sleeve.
  - Entry-time order, with skips counted.
  - MOM_LONG and SPIKE_FADE have no sleeve cap of their own; they compete for the global 4 slots, as live.
- **MOM_LONG:**
  - Uses the kept fills of `build_replay()` with today's LONG_HEAT 3-leg and LONG_CHOP_BURST filters.
  - Sizing follows today's `today_size_rule`, split into invest mult × leverage mult. Re-priced cells run at lev 1× (20×); unchanged cells keep their cell lev mult. The mean invest × lev multiplier is 1.25.
  - **FIX-A is applied dynamically.** The replay's own fill ratio is dropped. The engine's 0.1 % × 24 h-volume notional cap (ceiling $500k) is applied at the book's size at each entry. A throttled margin below the $100 minimum is skipped, like the engine. No ML fill hit that skip.
- **SPIKE_FADE:**
  - Uses the kept replay fades: invest 2× at 20×. Ticket = min(desired notional, 0.5 % × 24 h volume, $500k), at the book's size.
  - The fade variants shift **every** fade's pnl % by the same Δ. Δ = target − the mean replay pnl % of all offered fades in the window, with the 3 seeds pooled (−0.0093 %, N 1,183).
  - **F1: Δ = +0.0493** (mean → +0.04). **F2: Δ = −0.0407** (mean → −0.05).
  - No trades were added or removed. The taken mean is +0.039 (F1) because a few fades were skipped.
- **BULLRUN / BEARRUN / FRENZY_*:** priced exactly as in the 5-sleeve study, with no liquidity cap there, for parity.
- **Turned off:** MOM_SHORT, FLIP, SURGE and WILLY.

## Caveats (read before quoting any number)

- **MOM_LONG replay-vs-live gap.**
  - The yr5 replay reproduces only about 64 % of live ML fills. Live master ML is strongly positive (baseline v20 ML 37 · 86 % · +$2,612), while the replay's ML is −0.067 %/trade.
  - This study's verdict on ML is the **replay's** verdict. The sensitivity table shows what ML would have to earn per trade.
  - The ML H1 loss is real in the replay (−0.086 %/trade, CI below zero in the halves study). That is why even a 0.00 mean leaves a −94 % drawdown.
- **Fade calibration.**
  - F1's +0.04 comes from the recall trace's stale-candle fix (window estimate +0.05, 95 % range −0.12…+0.23). It is a **uniform shift**, not re-traded fills.
  - Real-money stop slip (≈ −0.02 %/trade across fades, ≈ −0.3 % per stop) puts real money between F1 and F2.
  - The fades' seed spread is wider than their mean in every variant.
- **In-sample.** The FRENZY bearish block, ATR 3.0, WIDE hold-green, the LITE design, the heat 3-leg and the chop-burst filter were all chosen on this year. Haircut the gains by 30–50 %.
- **LITE is a study cohort**, not the engine replay. It is the same cohort in every seed.
- **BULLRUN is one 6-day stretch in August.** It is the biggest single $ contributor whenever the book is still alive by then.
- **Not modelled:** exit slip for the FRENZY family (≈ −0.08 %/fill), funding, liquidation (paper), the WILLY global hold (it would block the other sleeves while open), and the gross-notional cap (25× balance; never binds at 4 slots × ~24 % × 20×).
- **Liquidity cap** is applied only to MOM_LONG (0.1 %) and the fades (0.5 %). The 5-sleeve parity is kept for the others.
- **Worst-day % on a ruined book** (F0 / F2 rows) is a percentage of a few dollars. It is not meaningful in $ terms.
