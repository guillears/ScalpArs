# ML_MKT_HOT — rolling-universe design study (2026-11-06)

## 0 · PRE-REGISTRATION (written 2026-11-06 15:52 UTC, BEFORE any kline fetch or P&L split; not edited afterwards)

**Question.** The ML_MKT_HOT watch line (DECISION_LOG 416) measures breadth on a FROZEN 108-coin list. Pick ONE rolling-universe definition the engine can compute live and stamp on every trade, and check the hot / not-hot split survives on it.

**Cohort (fixed).** Full-size momentum longs: master `reports/MASTER_POOL_stacked.csv` stack_keep ∧ ¬is_probe ∧ MOMENTUM ∧ LONG (expect 158, pct = stack_pct) + current batch B25 = newest `~/Downloads/scalpars_orders_paper_*.csv`, CLOSED MOMENTUM LONG opened ≥ 2026-10-23 18:30, probes out via `sweep_separators._is_probe` (expect 39, pct = pnl_percentage).

**Measure (frozen, never re-fit).** breadth = share of the universe's coins whose last CLOSED 1h close is above the close 4 h earlier (strict >), coins with both closes only; ≥ 30 coins with data else UNSCORED; a last bar > 2 h old = UNSCORED. hot = breadth > 0.50. Blocked = hot, kept = not hot.

**A · Candidate definitions** (all: USDT-margined PERPETUAL only, BTCUSDT / ETHUSDT out, stablecoin / USD-pegged perps out (USDC, FDUSD, TUSD, USDP, BUSD, USD1, USDE, DAI, PYUSD, RLUSD, U, XUSD …), no look-ahead — a day-D universe uses only data closed before 00:00 UTC of D):
- **V1** top-50 by the PREVIOUS UTC day's quote volume, rebuilt daily at 00:00 UTC.
- **V2** top-100, same rule.
- **V3** top-50 by the 7-day AVERAGE quote volume of the 7 previous UTC days (a coin needs ≥ 1 day of data; average over the days it has, max 7).
- **V4** liquidity floor: every perp with 7-day average quote volume ≥ F, capped at the 150 largest. F is picked ONCE = the median, over the frozen 108, of each member's 7-day average quote volume over 2026-10-23 … 2026-10-29 (the 7 days before registration 10-30), then frozen. F is computed and recorded before any P&L split.
- **V5** V3 + listing age ≥ 30 days at D (onboardDate from exchangeInfo; if missing, the first available 1d kline).
- **V6** the bot's own scan universe IF honestly reconstructable from repo data: top-50 by rolling 24 h quote volume at the entry hour, the scan's pre-filters (underlyingType COIN, not Alpha sub-type, onboard ≥ new_listing_filter_days). Reconstructable only with TODAY's config (filter settings changed over the window) — it is reported as "current-config reconstruction", validated against the fills' own entry_pair_rank stamp where present; if that validation fails it is skipped.
- **F108** reference = the frozen list: zones.csv ML_MKT_HOT flag + recomputed here from the same klines (pipeline check).

**B · Selection rules, in this priority order (NOT the biggest gap):**
1. Live-computable cheaply: ≤ 1 universe refresh per day, breadth ≤ 1 batch of 1h klines per hour within budget (weight / hour stated).
2. Adapts: turnover measured (coins in / out per week); no dead / delisted coin can stay in.
3. Faithful: agreement with the F108 hot flag on the cohort (%), and blocked < kept (same sign) in master AND in B25. A sign flip in either block = disqualified. Magnitude reported, not ranked.
4. Stability: smoother membership (lower weekly turnover) preferred.
5. Simplicity as tie-breaker.

**C · Also reported per variant:** coverage (scorable fills), breadth distribution (is 0.50 still "hot"? the cut is NOT re-fit), F108 drift by month (members outside that month's top-100).

**Statistics.** Per block (master, B25, all): blocked N · WR · avg % vs kept N · WR · avg %, gap = kept − blocked, day-clustered bootstrap P(blocked avg < kept avg) (4,000 resamples, seed 7, UTC days), agreement with F108, flips.

---

_Results generated 2026-11-06 16:30 UTC by `scripts/study_ml_mkt_hot_rolling.py` (export scalpars_orders_paper_2026-11-06_13-03-41.csv; 0 API requests this run; 1h matrix 529 coins × 3447 bars to 2026-11-06 15:00 UTC)._

**Clarifications fixed before any result was seen (coded 2026-11-06 ~16:10 UTC, ahead of the first complete run):** B4 is read as the share of members replaced week-over-week (new members vs 7 days earlier ÷ universe size — absolute counts favour small lists); a B4 tie = within 2 points; B5 order V1 < V2 < V3 < V5 < V4 < V6; B2 'dead coin' = a member with no 1h bar in the 24 h before the day start (its stay in days is reported; breadth drops it anyway since it has no close).

**V4 floor (pre-registered rule, computed before the split):** median F108 member 7-day avg quote volume 2026-10-23…29 = **$25.4 M/day**, frozen.

Cohort: master 158 (expect 158) · B25 39 (expect 39). zones.csv F108 flag present on 197 / 197.

## 1 · Pipeline check — F108 recomputed vs zones.csv

Recomputed F108 hot flag agrees with zones.csv on **100.0%** of 197 fills (0 differ). All split tables below use zones.csv for F108 (the registered flag) and this run's klines for V1…V6.

## 2 · Per-variant split (blocked = hot > 0.50 · kept = ≤ 0.50) — N · WR · avg %

| Variant | Block | scored / unscored | blocked | kept | gap kept − blocked | P(blocked < kept) | agree w/ F108 | flips (F108 hot→not / not→hot) |
|---|---|---|---|---|---|---|---|---|
| F108 | MASTER | 158 / 0 | 73 · 66% · +0.001 | 85 · 75% · +0.231 | +0.230 | 0.95 | 100% | 0 / 0 |
| F108 | B25 | 39 / 0 | 18 · 61% · -0.062 | 21 · 81% · +0.418 | +0.480 | 0.92 | 100% | 0 / 0 |
| F108 | all | 197 / 0 | 91 · 65% · -0.012 | 106 · 76% · +0.268 | +0.280 | 0.99 | 100% | 0 / 0 |
| V1 | MASTER | 158 / 0 | 83 · 71% · +0.077 | 75 · 71% · +0.177 | +0.099 | 0.75 | 90% | 3 / 13 |
| V1 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V1 | all | 197 / 0 | 98 · 68% · +0.036 | 99 · 74% · +0.240 | +0.204 | 0.93 | 90% | 6 / 13 |
| V2 | MASTER | 158 / 0 | 71 · 66% · +0.011 | 87 · 75% · +0.217 | +0.207 | 0.93 | 95% | 5 / 3 |
| V2 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V2 | all | 197 / 0 | 86 · 64% · -0.024 | 111 · 77% · +0.265 | +0.290 | 0.99 | 94% | 8 / 3 |
| V3 | MASTER | 158 / 0 | 76 · 67% · +0.052 | 82 · 74% · +0.191 | +0.139 | 0.83 | 93% | 4 / 7 |
| V3 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V3 | all | 197 / 0 | 91 · 65% · +0.012 | 106 · 76% · +0.247 | +0.235 | 0.96 | 93% | 7 / 7 |
| V4 | MASTER | 158 / 0 | 78 · 69% · +0.063 | 80 · 72% · +0.185 | +0.122 | 0.80 | 94% | 2 / 7 |
| V4 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V4 | all | 197 / 0 | 93 · 67% · +0.022 | 104 · 75% · +0.243 | +0.222 | 0.95 | 94% | 5 / 7 |
| V5 | MASTER | 158 / 0 | 75 · 68% · +0.057 | 83 · 73% · +0.185 | +0.128 | 0.81 | 94% | 4 / 6 |
| V5 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V5 | all | 197 / 0 | 90 · 66% · +0.016 | 107 · 76% · +0.242 | +0.226 | 0.96 | 93% | 7 / 6 |
| V6 | MASTER | 158 / 0 | 76 · 70% · +0.046 | 82 · 72% · +0.197 | +0.150 | 0.86 | 94% | 3 / 6 |
| V6 | B25 | 39 / 0 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | 92% | 3 / 0 |
| V6 | all | 197 / 0 | 91 · 67% · +0.007 | 106 · 75% · +0.251 | +0.244 | 0.97 | 94% | 6 / 6 |

## 3 · Breadth distribution, coverage, turnover, live cost

Hourly breadth, every closed hour 2026-06-18 → now (is 0.50 still 'hot'? share of hours hot should sit near F108's):

| Variant | universe size (avg) | hours | median | p25 / p75 | share of hours hot | fills: median breadth | breadth corr w/ F108 (fills) | turnover ins/week (day-over-day summed) | week-over-week new members | longest stay of a dead member (days) | weight / hour (live) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F108 | 108 | 3400 | 0.47 | 0.29 / 0.64 | 45% | 0.48 | 1.00 | 0.0 | 0.0 | 82 | ≈108 (1 kline call/coin, limit 5, w 1) |
| V1 | 50 | 3400 | 0.47 | 0.28 / 0.66 | 45% | 0.50 | 0.95 | 67.3 | 17.1 | 0 | ≈50 + 40/day ticker |
| V2 | 100 | 3400 | 0.47 | 0.30 / 0.63 | 44% | 0.47 | 0.98 | 111.8 | 28.4 | 0 | ≈100 + 40/day ticker |
| V3 | 50 | 3400 | 0.48 | 0.30 / 0.66 | 45% | 0.48 | 0.96 | 20.0 | 14.4 | 0 | ≈50 + 40/day ticker |
| V4 | 82 | 3400 | 0.47 | 0.31 / 0.64 | 45% | 0.46 | 0.97 | 28.0 | 22.2 | 0 | ≈82 + 40/day ticker |
| V5 | 50 | 3400 | 0.48 | 0.30 / 0.66 | 45% | 0.46 | 0.96 | 19.6 | 14.0 | 0 | ≈50 + 40/day ticker (+exchangeInfo 1/day) |
| V6 | 50 | 3399 | 0.46 | 0.26 / 0.68 | 45% | 0.48 | 0.96 | 54.5 | 13.3 | 0 | ≈50 (+ tickers the scan already pulls) |

Weight / hour = breadth needs the last 5 closed 1h bars of every member once per hour (`/fapi/v1/klines` limit 5 = weight 1 each); membership needs ONE `/fapi/v1/ticker/24hr` (all symbols, weight 40) at 00:00:30 UTC per day — at that moment the 24 h rolling volume IS the previous UTC day (V1/V2); V3/V4/V5 keep the last 7 of those daily snapshots (or one 1d-kline call per perp on cold start, ≈550 × weight 1). The engine already pulls 5m/1h data for its scan pairs, so the incremental cost is lower still. Budget headroom: 1,200 weight/min = 72,000/h.

## 4 · V6 (bot scan universe) reconstruction check

Fills carrying entry_pair_rank: 197; with a V6 rank for the pair: 192 · median |bot rank − reconstructed rank| 2.0 · within ±5: 94% · Spearman 0.99. The reconstruction applies TODAY's scan config (COIN-only, no Alpha, onboard ≥ 90 d, top-50 by 24 h volume); the window ran with other settings at times (pairs limit / listing filter / blacklist changes), so V6 on old fills is an approximation, not the as-was scan.

## 5 · F108 drift by month (ranks by monthly total quote volume, BTC/ETH/stables out)

| Month | days | F108 outside top-100 | outside top-50 | no volume (delisted / not listed) | avg daily overlap F108 ∩ V1 (of 50) |
|---|---|---|---|---|---|
| 2026-06 | 30 | 32 | 61 | 1 | 42.7 |
| 2026-07 | 31 | 41 | 68 | 3 | 39.0 |
| 2026-08 | 31 | 37 | 69 | 3 | 38.2 |
| 2026-09 | 30 | 44 | 66 | 2 | 39.4 |
| 2026-10 | 31 | 53 | 73 | 2 | 35.8 |
| 2026-11 | 5 | 50 | 71 | 2 | 35.8 |

F108 members not TRADING in today's exchangeInfo: 2 — IPUSDT (SETTLING), TONUSDT (SETTLING)

## 6 · Coverage & blind spots

- Unscored fills per variant: F108 0 · V1 0 · V2 0 · V3 0 · V4 0 · V5 0 · V6 0.
- Candidate symbols with no 1d data anywhere (fapi 400 and not in the repo daily cache): 1 — GAIBUSDT.
- Delisted coins whose daily volume came from the repo cache (fapi no longer serves them; cache ends 2026-10-11): 0.
- 1h symbols with no 1h klines (delisted before / not served): 0 of 529. A coin delisted mid-window that fapi no longer serves cannot enter breadth → slight survivorship in V1…V6 (and in F108's recompute) for those hours.
- Stablecoin / pegged perps excluded: USDCUSDT.
- V6 uses today's scan config and the current exchangeInfo metadata (onboardDate / subtype) — not the as-was config per day.
- Breadth reads the 1h close series only; no volume weighting (same as the registered measure).

## 7 · Selection walk-through (pre-registered order B1 → B5; impact numbers in §8 are NOT used)

| Variant | B1 live-cheap | B2 adapts (turnover > 0 · dead member stay) | B3 agree w/ F108 · sign master · sign B25 | B4 week-over-week replaced (% of members) | B5 rank | passes B1–B3 |
|---|---|---|---|---|---|---|
| V1 | yes | yes (17.1 new / week · dead stay 0 d) | 90% · blocked<kept · blocked<kept | 34.2% | 1 | ✅ |
| V2 | yes | yes (28.4 new / week · dead stay 0 d) | 94% · blocked<kept · blocked<kept | 28.4% | 2 | ✅ |
| V3 | yes | yes (14.4 new / week · dead stay 0 d) | 93% · blocked<kept · blocked<kept | 28.8% | 3 | ✅ |
| V5 | yes | yes (14.0 new / week · dead stay 0 d) | 93% · blocked<kept · blocked<kept | 28.1% | 4 | ✅ |
| V4 | yes | yes (22.2 new / week · dead stay 0 d) | 94% · blocked<kept · blocked<kept | 27.2% | 5 | ✅ |
| V6 | yes (scan already pulls tickers) | yes (13.3 new / week · dead stay 0 d) | 94% · blocked<kept · blocked<kept | 26.6% | 6 | ✅ |

B4: lowest relative turnover 26.6%; within 2% points: V2, V5, V4, V6 → B5 simplicity → **winner = V2**.

## 8 · Impact of switching the list — every Market-hot-dependent momentum-long line (F108 = before · ★ = the B1–B5 winner)

### 8.1 ML_MKT_HOT itself (DL 416)

Daily compound = the sleeve's DCR with the block ON (B25: real account path, review_math.dcr_account, start 2,900, flow 2026-10-26 13:50 +7,100; master: per-era calendar-day DCR at the 1× notional ruler X1 = 0.183 of equity, mean over eras). No block: B25 +0.183 %/day · master -0.016 %/day.

| List | Block | blocked N · WR · avg % | kept N · WR · avg % | gap | P(blocked < kept) | DCR block ON (%/day) |
|---|---|---|---|---|---|---|
| F108 | MASTER | 73 · 66% · +0.001 | 85 · 75% · +0.231 | +0.230 | 0.95 | +0.013 |
| F108 | B25 | 18 · 61% · -0.062 | 21 · 81% · +0.418 | +0.480 | 0.92 | +0.194 |
| V1 | MASTER | 83 · 71% · +0.077 | 75 · 71% · +0.177 | +0.099 | 0.75 | -0.001 |
| V1 | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |
| **V2 ★** | MASTER | 71 · 66% · +0.011 | 87 · 75% · +0.217 | +0.207 | 0.93 | +0.008 |
| **V2 ★** | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |
| V3 | MASTER | 76 · 67% · +0.052 | 82 · 74% · +0.191 | +0.139 | 0.83 | +0.004 |
| V3 | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |
| V4 | MASTER | 78 · 69% · +0.063 | 80 · 72% · +0.185 | +0.122 | 0.80 | +0.006 |
| V4 | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |
| V5 | MASTER | 75 · 68% · +0.057 | 83 · 73% · +0.185 | +0.128 | 0.81 | +0.004 |
| V5 | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |
| V6 | MASTER | 76 · 70% · +0.046 | 82 · 72% · +0.197 | +0.150 | 0.86 | +0.006 |
| V6 | B25 | 15 · 53% · -0.190 | 24 · 83% · +0.438 | +0.629 | 0.99 | +0.222 |

Frozen gate (CURRENT_STATE 👁 ML_MKT_HOT: forward fills ≥ DL 416 commit + 10 min; zone N ≥ 15 on ≥ 8 days → decided once):

| List | forward zone / rest / unscored | crossing reached? | gate verdict (at crossing, else live) |
|---|---|---|---|
| F108 | 8 / 8 / 0 | no | COLLECTING — zone N 8/15 · 3/8 days |
| V1 | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |
| **V2 ★** | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |
| V3 | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |
| V4 | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |
| V5 | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |
| V6 | 5 / 11 / 0 | no | COLLECTING — zone N 5/15 · 3/8 days |

### 8.2 ML_LEAD_A (DL 481) — cell (EMA20 slope > 0.1481 ∧ RSI prev ≤ 60.9) inside the NOT-hot side, history + B25 + forward pooled

| List | not-hot scored | cell N · WR · avg % | rest of not-hot N · WR · avg % | newest (B25+fwd) cell vs rest | verdict |
|---|---|---|---|---|---|
| F108 | 106 | 30 · 90% · +0.807 | 76 · 71% · +0.055 | 3 · 100% · +1.856 vs 18 · 78% · +0.178 | **MULTIPLIER CANDIDATE (operator decides; never automatic)** — pooled cell 30 · 90% · +0.81 on 22 d vs rest 76 · 71% · +0.06 · P(cell > rest) 1.000 · biggest day / pair 19% of the gain |
| V1 | 99 | 27 · 89% · +0.814 | 72 · 68% · +0.025 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **KEEP OBSERVING** — pooled cell 27 · 89% · +0.81 on 20 d vs rest 72 · 68% · +0.02 · P(cell > rest) 0.999 · biggest day / pair 21% of the gain |
| **V2 ★** | 111 | 31 · 90% · +0.785 | 80 · 71% · +0.064 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **MULTIPLIER CANDIDATE (operator decides; never automatic)** — pooled cell 31 · 90% · +0.78 on 24 d vs rest 80 · 71% · +0.06 · P(cell > rest) 1.000 · biggest day / pair 19% of the gain |
| V3 | 106 | 29 · 90% · +0.802 | 77 · 71% · +0.038 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **KEEP OBSERVING** — pooled cell 29 · 90% · +0.80 on 22 d vs rest 77 · 71% · +0.04 · P(cell > rest) 1.000 · biggest day / pair 20% of the gain |
| V4 | 104 | 29 · 90% · +0.802 | 75 · 69% · +0.027 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **KEEP OBSERVING** — pooled cell 29 · 90% · +0.80 on 22 d vs rest 75 · 69% · +0.03 · P(cell > rest) 1.000 · biggest day / pair 20% of the gain |
| V5 | 107 | 29 · 90% · +0.802 | 78 · 71% · +0.033 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **KEEP OBSERVING** — pooled cell 29 · 90% · +0.80 on 22 d vs rest 78 · 71% · +0.03 · P(cell > rest) 1.000 · biggest day / pair 20% of the gain |
| V6 | 106 | 29 · 90% · +0.802 | 77 · 69% · +0.044 | 3 · 100% · +1.856 vs 21 · 81% · +0.236 | **KEEP OBSERVING** — pooled cell 29 · 90% · +0.80 on 22 d vs rest 77 · 69% · +0.04 · P(cell > rest) 1.000 · biggest day / pair 20% of the gain |

### 8.3 Replace analysis (reports/ML_CELL_REPLACE_LEADA_2026-11-06.md) — daily compound %/day

| List | Block | before (today: 1.5× UNMATCHED, no block) | Market-hot only | + replace Lead A ×1.5 | + replace Lead A ×2 | unscored → kept |
|---|---|---|---|---|---|---|
| F108 | MASTER | -0.016 | +0.015 | +0.022 | +0.032 | 0 |
| F108 | B25 | +0.183 | +0.194 | +0.193 | +0.238 | 0 |
| V1 | MASTER | -0.016 | -0.002 | +0.007 | +0.016 | 0 |
| V1 | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |
| **V2 ★** | MASTER | -0.016 | +0.009 | +0.017 | +0.026 | 0 |
| **V2 ★** | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |
| V3 | MASTER | -0.016 | +0.005 | +0.013 | +0.021 | 0 |
| V3 | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |
| V4 | MASTER | -0.016 | +0.007 | +0.015 | +0.023 | 0 |
| V4 | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |
| V5 | MASTER | -0.016 | +0.005 | +0.012 | +0.021 | 0 |
| V5 | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |
| V6 | MASTER | -0.016 | +0.007 | +0.014 | +0.023 | 0 |
| V6 | B25 | +0.183 | +0.222 | +0.221 | +0.266 | 0 |

### 8.4 ML_FIXED_TP line B (DL 479) — fixed TP on the fills KEPT by Market-hot: avg %/fill today's exit → TP +0.5 / +1 / +2 / +3

| List | Block | N kept | today's exit | TP +0.5 | TP +1 | TP +2 | TP +3 | verdict per TP (pooled) |
|---|---|---|---|---|---|---|---|---|
| F108 | MASTER | 85 | +0.231 | +0.045 | +0.210 | +0.174 | +0.182 | +0.5: NO EDGE · +1: NO EDGE · +2: NO EDGE · +3: KEEP OBSERVING |
| F108 | B25+fwd | 21 | +0.418 | +0.184 | +0.383 | +0.591 | +0.734 |  |
| V1 | MASTER | 75 | +0.177 | +0.000 | +0.170 | +0.172 | +0.185 | +0.5: NO EDGE · +1: NO EDGE · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| V1 | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |
| **V2 ★** | MASTER | 86 | +0.215 | +0.036 | +0.206 | +0.208 | +0.215 | +0.5: NO EDGE · +1: NO EDGE · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| **V2 ★** | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |
| V3 | MASTER | 82 | +0.191 | +0.038 | +0.196 | +0.180 | +0.195 | +0.5: NO EDGE · +1: NO EDGE · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| V3 | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |
| V4 | MASTER | 80 | +0.185 | +0.032 | +0.203 | +0.183 | +0.198 | +0.5: NO EDGE · +1: KEEP OBSERVING · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| V4 | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |
| V5 | MASTER | 83 | +0.185 | +0.034 | +0.190 | +0.174 | +0.189 | +0.5: NO EDGE · +1: NO EDGE · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| V5 | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |
| V6 | MASTER | 82 | +0.197 | +0.013 | +0.188 | +0.177 | +0.185 | +0.5: NO EDGE · +1: NO EDGE · +2: KEEP OBSERVING · +3: KEEP OBSERVING |
| V6 | B25+fwd | 24 | +0.438 | +0.224 | +0.412 | +0.594 | +0.719 |  |

### 8.5 Fills that change side F108 → V2 (11 of 197)

| Block | opened (UTC) | pair | pct % | F108 | V2 | F108 breadth | V2 breadth |
|---|---|---|---|---|---|---|---|
| MASTER | 2026-06-30 06:30 | FARTCOINUSDT | +0.356 | HOT (blocked) | not hot (kept) | 0.51 | 0.45 |
| MASTER | 2026-07-11 18:23 | ARBUSDT | +0.166 | not hot (kept) | HOT (blocked) | 0.48 | 0.54 |
| MASTER | 2026-08-04 02:20 | PUMPUSDT | +0.980 | HOT (blocked) | not hot (kept) | 0.58 | 0.50 |
| MASTER | 2026-08-04 11:18 | ONDOUSDT | +0.746 | not hot (kept) | HOT (blocked) | 0.50 | 0.63 |
| MASTER | 2026-08-04 12:53 | DEXEUSDT | +0.427 | HOT (blocked) | not hot (kept) | 0.53 | 0.50 |
| MASTER | 2026-09-20 22:36 | AAVEUSDT | +0.100 | HOT (blocked) | not hot (kept) | 0.55 | 0.49 |
| MASTER | 2026-10-02 00:18 | XPLUSDT | -0.696 | HOT (blocked) | not hot (kept) | 0.53 | 0.36 |
| MASTER | 2026-10-15 22:10 | DOTUSDT | +0.962 | not hot (kept) | HOT (blocked) | 0.46 | 0.51 |
| B25 | 2026-11-06 09:11 | TAOUSDT | +0.998 | HOT (blocked) | not hot (kept) | 0.62 | 0.48 |
| B25 | 2026-11-06 09:40 | SUIUSDT | +0.435 | HOT (blocked) | not hot (kept) | 0.62 | 0.48 |
| B25 | 2026-11-06 09:47 | FETUSDT | +0.312 | HOT (blocked) | not hot (kept) | 0.62 | 0.48 |

## 9 · Verdict (plain words)

- **The split survives a rolling universe.** Every rolling variant keeps blocked < kept in master AND in B25. Winner V2: master gap +0.207 (P 0.93) vs F108 +0.230 (P 0.95); B25 +0.629 (P 0.99) vs +0.480 (P 0.92); all +0.290 (P 0.99) vs +0.280 (P 0.99). Same sign, similar size — the Market-hot effect is not an artefact of the frozen list. It agrees with F108's hot flag on 94% of the 197 fills.
- **But every rolling list shows a SMALLER master gap than F108** (+0.10 … +0.21 vs +0.23; the 50-coin lists are weakest, V1 +0.10 P 0.75). F108 was the list the split was found on, so part of its master edge is in-sample; the rolling numbers are the more honest estimate. In B25 every rolling list is stronger (+0.63 vs +0.48): 3 fills on 11-06 09:11–09:47 (TAO / SUI / FET, all winners) read 0.62 on F108 but 0.48 on every rolling list.
- **F108 is drifting fast:** members outside that month's top-100 by volume rose from 32 / 108 (June) to 53 (October) and 50 (November, 5 days); outside the top-50 61 → 73; 2 members are no longer trading (IPUSDT, TONUSDT SETTLING); on an average October day only ~36 of the 50 most-traded coins are in F108. Breadth on F108 is increasingly the breadth of coins nobody trades.
- **The 0.50 cut still means 'hot' on every list:** median hourly breadth 0.46–0.48, 44–45 % of hours hot on all variants and on F108 — the cut does not need (and was not given) a re-fit.
- **Decisions that move with the list (see §8):** ML_MKT_HOT — numbers move a little, gate still COLLECTING (forward zone 8 → 5 of 15), no verdict changes today. ML_LEAD_A — on V2 it stays MULTIPLIER CANDIDATE (cell 31 · 90 % · +0.78 vs F108 30 · 90 % · +0.81) but on every OTHER rolling list it drops to KEEP OBSERVING only because the cell falls to N 27–29 (< 30): the verdict sits on the N ≥ 30 edge and is list-fragile — treat it as borderline, not robust. Replace analysis — ranking unchanged (×2 > ×1.5 ≈ Market-hot only > before) on both lists; master levels drop ~0.005 %/day, B25 levels rise ~0.03 %/day. ML_FIXED_TP line B — still no fixed TP beats today's exit on master; +2 moves from NO EDGE to KEEP OBSERVING (master Σ Δ shrinks to ≈ 0), no candidate appears.

### Recommendation

- **Switch the live measure to the rolling list V2** (engine computes and stamps it on every trade, spec below). Do NOT re-freeze a new fixed list — it would drift again (≈ 5 of 108 members a month leave the top-100).
- **Keep the DL 416 frozen gate as registered** (read on F108, pre-committed bars do not move), and add one condition before any ship: a FILTER CANDIDATE on F108 must also show blocked < kept on V2 over the same prefix (else no ship). Once the engine stamp exists, every NEW read uses the stamped V2 value.
- **Drift alarm** (scout line, report every batch): ⚠ when > 40 % of F108 is outside the trailing-30-day top-100 (already true: 53 / 108 in October) OR the trailing-30-day hot-flag agreement F108 vs V2 on momentum-long fills falls below 85 % (today 94%). For the rolling stamp itself: ⚠ when < 90 members have data on an hour (the stamp goes NULL below 30).

## 10 · SPEC — `entry_mkt_breadth_r100` (winner V2; precise enough for the engine)

1. **Universe rule:** the 100 USDT-margined `contractType == PERPETUAL`, `status == TRADING` symbols with the highest quote volume of the PREVIOUS UTC day, after exclusions. One list per UTC day.
2. **Refresh time:** 00:00:30 UTC daily, from ONE `GET /fapi/v1/ticker/24hr` (all symbols, weight 40) — at that moment `quoteVolume` = the previous UTC day (equivalently the 1d kline of D−1, field 7). Keep yesterday's list if the refresh fails; ⚠ log after 2 missed refreshes. (Study: day-over-day ≈ 16 coins swap at the rank-100 boundary; week-over-week 28 of 100 are new.)
3. **Exclusions:** BTCUSDT, ETHUSDT; stablecoin / USD-pegged perps (base in USDC, FDUSD, TUSD, USDP, BUSD, USD1, USDE, DAI, PYUSD, RLUSD, U, XUSD, USDS, EUR, EURI, AEUR, USDD, BFUSD, or a 30-day close within 0.97–1.03 on > 95 % of days); non-PERPETUAL; non-USDT. NOT excluded (as tested): the pair blacklist, Alpha sub-type coins (~21 of 100 a day), new listings, non-COIN underlyings — the measure is the MARKET, not the tradable list.
4. **Bar at entry:** the last CLOSED 1h bar at the entry time (bar close time ≤ entry; never the forming bar). If that bar closed more than 2 h before the entry → NULL.
5. **Breadth formula:** for every member with both closes, r = close(last closed 1h bar) / close(the 1h bar 4 h earlier) − 1; breadth = #(r > 0) ÷ #(members with both closes). Strict `>` (flat counts as not up). Hot = breadth > 0.50 (frozen DL 416 cut).
6. **Minimum coins:** ≥ 30 members with both closes, else the stamp is NULL (never 0, never counted on either side).
7. **Data / cost:** once per hour at HH:00:05+ fetch `GET /fapi/v1/klines?interval=1h&limit=6` for the 100 members (weight 1 each → ≈ 100 weight / hour + 40 / day; budget 72,000 / hour). Cache per hour; every entry in that hour reads the same value (pure function of the hour → identical across sleeves).
8. **Stamps (D13 class: universal):** `entry_mkt_breadth_r100` (float 0–1, NULL as above) + `entry_mkt_breadth_r100_n` (members with data). Also keep stamping the F108 value (`entry_mkt_breadth_f108`) during the transition so the frozen gate can be read from stamps.

