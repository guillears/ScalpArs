# FRENZY: does "24h volume X % below market cap" separate winners from losers? (2026-10-08)

Research only. No code, config, template or test was touched and nothing was committed. New files: `scripts/study_vol_mcap_*.py`, `reports/cache_vol_mcap/` (API cache), this report, and `reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv` (one row per fill with its R values). This is an **unreviewed backtest** and cannot support an arm or ship until the caveman and deep reviews have run (feedback_no_arm_before_review).

`scripts/validate_against_master.py` was run first: **ALL CHECKS PASS**.

## Plain-English answer

**For FRENZY_LONG / WIDE / LITE: no.** The turnover ratio R = 24h volume ÷ market cap does not separate winners from losers, at any X.
- Backtest, today's kept stack (383 scored fills): low-turnover and high-turnover fills earn the same. Quintile averages run from −0.14 to +0.25 %/fill with no order. The best threshold anywhere in the scan is indistinguishable from random labels (null **p = 0.82**; with the bearish block on, p = 0.97). Choosing X on one half of the year and testing it on the other half fails both ways, and leave-one-month-out fails too.
- Operator's literal rule, "volume below market cap" (R ≤ 1), on today's stack with the bearish block: below gives 87 fills, 56 % WR, +0.27 %; above gives 197 fills, 52 % WR, +0.04 %. The confidence ranges overlap almost completely ([−0.40, +0.91] vs [−0.37, +0.45]), and a threshold of 2 points the other way. That is noise.
- Real fills (16 FRENZY fills, Oct 3 → Oct 8, all stamped): if anything the sign is the **opposite** of the hypothesis. R ≤ 1 gives 6 fills, 2 wins, avg −1.00 %. R > 1 gives 10 fills, 6 wins, avg +0.60 %. With N = 16 this is noise either way.
- **Recommendation: reject for the FRENZY sleeves.** No scout line is needed, because the bot already stamps both inputs on every fill, so this can be re-read at any review for free.

**For FRENZY_WILLY entry A (buy the new flag, +1 / −3 / 60 min): yes, weakly, and in the operator's direction.** Higher turnover is worse.
- Volume below market cap (R ≤ 1): 1,139 fills, 75 % WR, **+0.077 %** [−0.03, +0.18]. Volume above market cap (R > 1): 470 fills, 69 % WR, **−0.198 %** [−0.36, −0.04].
- The direction holds in **10 of 10 months** and in both halves of the year:
  - Jan–May: +0.100 vs −0.107
  - Jun–Oct: +0.046 vs −0.262
- The best threshold in the scan is R > 2.37 (decile 10, avg −0.30 %). Scan-corrected null p = **0.052** with shuffled labels and **0.031** with labels shuffled within each day. Both out-of-sample splits and leave-one-month-out hold their direction.
- The R > 1 side passes the expectancy bar against WILLY's breakeven WR (~74–75 %): WR 69 %, P(avg < 0) = 0.99, 216 days, no pair or day above 6 % of the loss, N 470.
- **It does not make WILLY A profitable.** The kept side is +0.08 %/fill before the 30–50 % haircut (about +0.04 after) and its range includes zero. Blocking R > 1 cuts the loss; it does not create an edge.
- X = 100 % was read after the scan had been seen, so it is not a clean pre-registration. Evidence is backtest only, with no real WILLY fills.
- **Recommendation: an observe-only scout line, frozen at X = 100 % (R = vol24h / mcap > 1.0 → "would block").** Read it on real WILLY A fills. Arming is reserved for the locked gates on real fills (OBSERVE-FIRST rule).

**For FRENZY_WILLY entry B (fresh ON bars that FRENZY / WIDE did not take): no separator, and the cohort itself loses.**
- Scored cohort: 903 fills, WR 70 %, **−0.188 %** [−0.32, −0.07]. It loses at every R:
  - R ≤ 1: −0.22
  - R > 1: −0.18
- The threshold scan has null p = 0.72. The "bar passed" line in the appendix only means one side of an already-losing cohort is also losing. **Reject as a separator.** The bigger finding is that entry B is negative on the year at WILLY's exit, which matches the earlier ON-scalp and seconds-delay studies.

## 1. Data and method

| item | detail |
|---|---|
| **R (primary)** | 24h Binance **futures** quote volume ÷ market cap. Real fills use the bot's own stamps, `entry_pair_volume_24h_usd / entry_mcap_usd` (CoinMarketCap data via the Binance Info-panel endpoint, stamped since 2026-09-28). |
| Backtest market cap | Same source as the bot: one request per base symbol to the `services/mcap_service.py` endpoint today (`scripts/study_vol_mcap_bapi.py`, same symbol rules and `expect=` check as the engine). Market cap at entry = today's market cap × (pair close at entry ÷ pair price today), i.e. **today's circulating supply × entry price**. |
| Backtest volume | Sum of `qvol` over the 288 closed 5m bars before entry, from `k5m_full` plus the scratch `k5new` for Oct 4–8. This equals the futures 24h ticker the bot stamps. |
| Constant-supply check (vs stamps) | On **all 67 stamped fills** in the master pool and live export (44 pairs, Sep 28 → Oct 8), reconstructed ÷ stamped market cap has median **1.007** (IQR 1.000–1.013). 94 % are within ±10 %, and the Spearman correlation is 0.999. Worst: 0G 1.73, ENA 1.33. On the 16 FRENZY fills, R rebuilt vs R stamped has Spearman 0.99. |
| Constant-supply check (year) | CoinGecko's own implied supply (market cap ÷ price) at each fill vs today has median **1.00** in every cohort. Only 4–7 % of covered fills had supply grow > 25 % (Jun 2026 median 1.18 on FRENZY). **The assumption holds, so no CoinGecko fallback was needed.** CoinGecko market caps sit ~1.2–1.8× below the CMC values, but that is a definition difference between the sources, not supply growth. Supply-corrected R vs uncorrected R: Spearman 0.96–0.99, and Spearman(R, P&L) does not change. |
| Robustness variants | R_fut = futures volume ÷ CoinGecko daily market cap (last point ≤ entry). R_cg = CoinGecko all-exchange volume ÷ CoinGecko market cap. CoinGecko history is **partial**: 106 of 434 coins were cached before I stopped the fetch. The free tier allowed ~2–5 requests/min with 60 s back-offs; it was no longer needed as a fallback. Results are in the appendix and do not change any conclusion. |
| Requests made | Binance futures `ticker/price` ×1 (weight 2, used-weight 2). Binance Info-panel endpoint: one request per base symbol at ≥ 1.5 s spacing, no 418/429. CoinGecko: ~115 requests at ≥ 3.2 s, backing off 60 s on each 429. Everything is cached under `reports/cache_vol_mcap/`. |
| Stats | Deciles and quintiles. 17-cut threshold scan (10–90 % quantiles, each side ≥ 25 or ≥ 40 fills). Shuffled-label null (1,000×, same scan, max \|Δ\|) and a within-day shuffled null. OOS split at Jun 1 both ways, plus leave-one-month-out. Day-clustered bootstrap CIs (4,000 draws). Expectancy bar on the blocked side. All P&L is % at 1×. |

### Cohorts and pricing

| cohort | source | pricing | N | scored (R) |
|---|---|---|---|---|
| REAL FRENZY | master pool `FRENZY_SLEEVE` kept rows (STACK 2026-10-08a) + `scalpars_orders_paper_2026-10-08_12-57-28.csv`, deduped on (opened_at, pair, direction), CLOSED | fixed +3 / −3 (`build_master_pool.frenzy_fixed_pct`) | 16 (LONG 9, WIDE 3, LITE 4; 6 bearish-day) | 16 / 16 stamped |
| FRENZY backtest, TODAY | engine-parity fresh-ON cohort `FRENZY_ENGINE_COHORT_2026-10-05.csv` via the ATR-cap study loader. Live-eligible, market volume U2 < 1.0, dislocation at 8 s, ATR ≤ 3.0, red → LONG, hold-green → WIDE, sequenced | FIX +3/−3 at 8 s, ticks | 427 | 383 (90 %) |
| … TODAY + bearish block | as above, minus BTC last-daily < 0 ∧ BTC 5m EMA13−EMA50 < 0, re-sequenced | same | 316 | 284 (90 %) |
| … GATED (any ATR) | every gated signal, unsequenced | same | 595 | 522 (88 %) |
| WILLY A | first-flag events (`flag_math/ev2.pkl`, 2,320 engine-reachable) | ticks: entry +8 s, slip 0.035, dislocation > 1 % refused, taker 0.045 ×2, TP +1.0 net, stop −3 net (gap fill − 0.05), 60 min cap | 1,781 priced (431 no ticks, 108 dislocated) | 1,609 (90 %) |
| WILLY B | live-eligible fresh-ON bars not taken by TODAY + bearish (`study_vol_mcap_cohorts.py`) | same WILLY pricing | 1,038 priced | 903 (87 %) |

About the ~850-fill `frenzy_gvr_trades.csv` set: it was a scratch file that no longer exists. The engine-parity cohort above is its successor (same signals, real live gates), so I used that instead.

Unscored fills are **not** counted as "rest". Their averages: TODAY +0.22 (44 fills), WILLY A −0.10 (172), WILLY B −0.15 (135).

**Pairs without a bot-source market cap** (the endpoint returns no `mc` for any candidate symbol; 72 of 493, mostly delisted or renamed): 1000000BOB, 1000RATS, 1000WHY, A2Z, ACU, ACX, ATA, AZTEC, B3, BAN, BASED, BIRB, BROCCOLIF3B, BSB, CARV, CHESS, COLLECT, COS, CROSS, DAM, DEGO, DENT, DODOX, DRIFT, D, EDGE, EPT, FIGHT, FIO, FUN, GWEI, HFT, HIGH, HIPPO, ICX, INX, LRC, LUNA2, MAGMA, MBOX, MLN, NFP, OM, OXT, PHB, PRL, PROMPT, PUMPBTC, RAYSOL, RDNT, RLS, SCRT, SKR, SKYAI, SLX, SPACE, SPORTFUN, STABLE, STG, STORJ, SYS, TANSSI, TOSHI, TRIA, TRU, VANRY, VIC, YALA, ZKJ, ZRC, 我踏马来了, 龙虾 (all USDT). Pairs that are not on Binance futures today also have no price "now", so they are unscored.

## 2. Key results (primary R)

### FRENZY backtest, today's stack (no bearish block), quintiles of R
| R band | N | WR | avg % | day 95 % CI |
|---|---|---|---|---|
| 0.11–0.77 | 77 | 55 % | +0.164 | [−0.52, +0.82] |
| 0.78–1.30 | 76 | 55 % | +0.219 | [−0.54, +0.94] |
| 1.31–2.26 | 77 | 49 % | −0.140 | [−0.80, +0.57] |
| 2.27–4.25 | 76 | 55 % | +0.219 | [−0.47, +0.91] |
| 4.26–30.7 | 77 | 56 % | +0.250 | [−0.37, +0.85] |

- Trade-level Spearman(R, P&L) = −0.02, and the deciles are non-monotone.
- Best scan cut: block R > 1.18, null p = 0.82 (within-day null 0.86).
- OOS:
  - Jan–May picks R > 0.95, which in Jun–Sep blocks +0.19 and keeps +0.09 (**fails**).
  - Jun–Sep picks R ≤ 0.90, which in Jan–May blocks +0.20 and keeps +0.09 (**fails**).
- With the bearish block: p = 0.97, and leave-one-month-out fails.
- Gated any-ATR cohort: p = 0.99.
- Market cap alone and volume alone don't separate either (p 0.29–0.82).
- Turnover is correlated with run/gain % (+0.42…+0.48), hours since spike (+0.38) and market cap (−0.44). It is not correlated with ATR (+0.08), gvol / U2 (−0.03…−0.04) or bearish day (BTC legs ≈ 0). So it is not a proxy for any current filter, and it still does not separate inside today's kept cohort.

### WILLY A, deciles of R
| R band | N | WR | avg % | day 95 % CI |
|---|---|---|---|---|
| 0.010–0.119 | 161 | 77 % | +0.175 | [−0.05, +0.39] |
| 0.12–0.22 | 161 | 73 % | −0.039 | [−0.28, +0.20] |
| 0.22–0.30 | 161 | 72 % | −0.011 | [−0.28, +0.25] |
| 0.30–0.41 | 161 | 76 % | +0.106 | [−0.16, +0.35] |
| 0.41–0.55 | 161 | 80 % | +0.284 | [+0.05, +0.50] |
| 0.55–0.74 | 160 | 74 % | +0.001 | [−0.24, +0.24] |
| 0.74–0.98 | 161 | 74 % | +0.002 | [−0.28, +0.25] |
| 0.98–1.40 | 161 | 70 % | −0.158 | [−0.45, +0.10] |
| 1.41–2.37 | 161 | 71 % | −0.092 | [−0.38, +0.18] |
| 2.38–44.6 | 161 | 68 % | −0.298 | [−0.58, −0.02] |

- Quintile order: Spearman −0.70. The top three deciles (R > ~1) are all negative.
- Blocked side at **X = 100 %** (R > 1): 470 fills, 216 days, WR 69 % against a kept breakeven of 73 %, avg −0.198 [−0.36, −0.04], P(avg < 0) 0.99. Pair concentration 6 % (PORTAL), day concentration 3 %. **Passes every expectancy-bar item.**
- Δ to the whole cohort ≈ +0.08 %/fill in-sample, about +0.04–0.06 after the haircut.
- Monthly R ≤ 1 vs R > 1, avg %:

  | month | R ≤ 1 | R > 1 |
  |---|---|---|
  | Jan | +0.03 | −0.26 |
  | Feb | +0.09 | −0.01 |
  | Mar | +0.04 | −0.42 |
  | Apr | +0.15 | +0.14 |
  | May | +0.16 | −0.11 |
  | Jun | −0.09 | −0.10 |
  | Jul | +0.41 | −0.44 |
  | Aug | −0.03 | −0.43 |
  | Sep | −0.05 | −0.11 |
  | Oct (11 fills above) | +0.37 | +0.26 |

  R > 1 is worse in all 10 months, though some gaps are tiny (Apr, Jun).
- Overlap:
  - R correlates −0.79 with market cap. Market cap alone separates more weakly (null p 0.083, non-monotone quintiles). Volume alone does not separate (p 0.40).
  - So the signal is mostly "small coins with heavy turnover" rather than size alone.
  - ATR and gvol are not stamped in this cohort, so they are a blind spot.
- The fixed X = 1 split under a within-day permutation gives one-sided p = 0.004, but X = 1 was chosen after seeing the scan, so use the scan-corrected p (0.03–0.05) as the honest figure.

### WILLY B
- 903 scored fills, avg −0.188 [−0.32, −0.07].
- Quintile averages: R order Spearman +0.10.
- Scan p = 0.72, within-day 0.67.
- Every R band is negative.

### Real fills (16), each fill
The full table is in the appendix. R_live, the stamped R, ranges from 0.66 (UMA, W, MET) to 29.9 (AIN 10-04, a winner). Of the 8 losers, 6 had R ≤ 1.53. Of the 8 winners, 6 had R ≥ 1.02. **The real fills lean the opposite way to the hypothesis**, but at N = 16 on 6 days that cannot support anything.

## 3. What I could NOT test (blind spots)
- **Supply between Jan and Sep** is checked only through CoinGecko's implied supply, on the ~25 % of pairs whose CoinGecko history was cached. The endpoint's own history does not exist.
- **72 pairs** have no bot-source market cap (delisted or renamed), and pairs not listed on futures today cannot be priced "now". This is survivorship: delisted coins are missing, and they are mostly crashes.
- **FRENZY_LITE** has no backtest cohort here. Its research cohort (HOLD_LOWVOL_EARLY) carries lock pricing, not +3/−3, so LITE is judged on 4 real fills only.
- **WILLY A has no ATR / gvol / 24h-change stamps** in its event file, so R's overlap with those can't be checked there. For FRENZY, it overlaps nothing that is filtered.
- WILLY A and B are **unsequenced**: no 2-slot or pair-held refusal applied.
- WILLY TP is booked at exactly +1.0 net at the crossing print, with no slip on the TP.
- The live 10-08 fills (6 bearish-day ones) traded before the bearish block, so they are shown both ways.
- Trade-level nulls treat fills as independent. The within-day null is stricter, and R is pair-persistent. Per-pair concentration was checked and is low (≤ 10 % in WILLY A).

## 4. Recommendation (per the locked rules)
| sleeve | verdict |
|---|---|
| FRENZY_LONG / WIDE / LITE | **Reject.** No separator at any X: null p 0.82–0.99, OOS fails both ways, real fills lean the other way. Nothing to scout, because the stamps already let any later review re-read it for free. |
| FRENZY_WILLY entry A | **Observe-only scout line, pre-registered and frozen:** R = `entry_pair_volume_24h_usd / entry_mcap_usd` > **1.0** → "would block". Tally WILLY A fills on both sides at each review. Arm only if the R > 1 side meets the expectancy bar on real fills (WR below WILLY's breakeven ≈ 75 %, P(avg < 0) ≥ 95 % window-clustered, ≥ 8 days, no day or pair ≥ 50 %, N ≥ 15). Never re-fit X. Even then, the evidence says the filter **reduces WILLY A's loss** and does not make it a proven earner (kept side ≈ +0.04 %/fill after haircut). |
| FRENZY_WILLY entry B | **Reject as a separator.** The cohort loses at every turnover level (−0.19 %/fill, CI below zero). That is a sleeve-level question for the operator, not a filter question. |

## Files
- `scripts/study_vol_mcap_bapi.py`: bot-source market cap / supply (one request per symbol, cached)
- `scripts/study_vol_mcap_fetch.py`: CoinGecko history (robustness only, partial)
- `scripts/study_vol_mcap_cohorts.py`: builds FRENZY TODAY / TODAY+bearish / GATED and the WILLY A / B inputs
- `scripts/study_vol_mcap_willy_price.py`: WILLY tick pricer
- `scripts/study_vol_mcap_analyze.py`: all tables; writes the per-fill CSV
- `scripts/study_vol_mcap_extra.py`: within-day null, market cap / volume alone, supply growth, supply-corrected R
- Per-fill CSV: `reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv` (columns: cohort, pair, day, pct, R, R_live, R_b, R_fut, R_cg, mcap_b, mcap_cg, vol_fut, …)
- Scratch intermediates: `<scratchpad>/vm/`

---

# Appendix A: full generated tables (`study_vol_mcap_analyze.py`)

Note: the CoinGecko robustness variants (R_fut, R_cg) cover only the ~25 % of pairs whose history was cached. "Bar PASSED" on a cohort whose kept side is also negative (WILLY B) is not a separator.

#### REAL FILLS (master FRENZY_SLEEVE kept + live export, deduped (opened_at, pair, direction), CLOSED, fixed +3/−3 pricing)

Constant-supply check on 67/67 stamped fills (44 pairs, opened 2026-09-28 → 2026-10-08): reconstructed / stamped mcap median 1.007, IQR 1.000–1.013, within ±10 % 94%, within ±25 % 96%, Spearman(rec, stamp) +0.999. Worst: 0GUSDT 1.73, ENAUSDT 1.33, ENAUSDT 1.32, MOVRUSDT 1.15, METUSDT 1.07

| src | opened_at | pair | sleeve | pct | bear | entry_atr_pct | entry_frenzy_gvol | entry_pair_volume_24h_usd | entry_mcap_usd | R_live | R_b | R_cg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| master | 2026-10-03T03:36:19 | ENJUSDT | FRENZY_LONG | -3.02 | False | 1.93 | nan | 1.14e+08 | 7.5e+07 | 1.52 | 1.53 | nan |
| master | 2026-10-04T05:05:16 | SANDUSDT | FRENZY_LONG | 3 | False | 1.32 | 0.979 | 8.12e+08 | 2.26e+08 | 3.59 | 3.47 | nan |
| master | 2026-10-04T11:15:11 | AINUSDT | FRENZY_LONG | 3 | False | 2.14 | 0.614 | 6.82e+08 | 2.28e+07 | 29.9 | 28.3 | nan |
| master | 2026-10-04T14:05:12 | SANDUSDT | FRENZY_LONG | -3.01 | False | 1.24 | 0.919 | 5.26e+08 | 2.39e+08 | 2.2 | 2.19 | nan |
| master | 2026-10-05T09:15:12 | MOVRUSDT | FRENZY_WIDE | 3 | False | 2.45 | 0.782 | 8.36e+07 | 2.49e+07 | 3.36 | 2.93 | nan |
| master | 2026-10-05T12:00:11 | RLCUSDT | FRENZY_WIDE | 3 | False | 2.38 | 0.684 | 4.28e+07 | 4.19e+07 | 1.02 | 1.05 | nan |
| master | 2026-10-06T09:40:08 | ORCAUSDT | FRENZY_LONG | -3 | False | 1.74 | 0.954 | 1.28e+08 | 1.58e+08 | 0.813 | 0.803 | nan |
| master | 2026-10-06T10:10:08 | UMAUSDT | FRENZY_LONG | -3 | False | 1.66 | 0.816 | 3.94e+07 | 6.01e+07 | 0.656 | 0.657 | nan |
| live | 2026-10-06T18:05:08 | ORCAUSDT | FRENZY_LONG | -3.01 | True | 2.44 | 0.78 | 2.61e+08 | 1.9e+08 | 1.37 | 1.37 | nan |
| live | 2026-10-07T05:45:12 | SANDUSDT | FRENZY_LITE | 3 | True | 1.3 | 0.688 | 1.62e+08 | 2.17e+08 | 0.746 | 0.74 | nan |
| live | 2026-10-07T19:10:08 | HEMIUSDT | FRENZY_LITE | 3 | False | 2.27 | 0.509 | 5.1e+07 | 1.65e+07 | 3.09 | 3.05 | nan |
| live | 2026-10-07T19:55:08 | METUSDT | FRENZY_LITE | 3 | False | 2.66 | 0.831 | 1.59e+08 | 2.42e+08 | 0.657 | 0.647 | nan |
| live | 2026-10-07T22:00:13 | MOVRUSDT | FRENZY_LITE | -3 | True | 0.833 | 0.655 | 8.88e+07 | 2.46e+07 | 3.61 | 3.59 | nan |
| live | 2026-10-08T05:50:09 | WUSDT | FRENZY_LONG | -3 | True | 1.79 | 0.698 | 8.13e+07 | 1.21e+08 | 0.674 | 0.675 | nan |
| live | 2026-10-08T06:50:10 | METUSDT | FRENZY_WIDE | 3 | True | 1.91 | 0.842 | 4.46e+08 | 2.62e+08 | 1.7 | 1.58 | nan |
| live | 2026-10-08T11:30:08 | ERAUSDT | FRENZY_LONG | -3.01 | True | 2.45 | 0.773 | 2.43e+07 | 3.16e+07 | 0.767 | 0.794 | nan |

all · lR_live: scored 16/16 · R ≤ median 1.44: N 8 WR 38% avg -0.753 · R > median: N 8 WR 62% avg +0.746 · Spearman +0.15

all · lR_b: scored 16/16 · R ≤ median 1.45: N 8 WR 38% avg -0.753 · R > median: N 8 WR 62% avg +0.746 · Spearman +0.14

bearish-day block applied · lR_live: scored 10/10 · R ≤ median 1.83: N 5 WR 40% avg -0.604 · R > median: N 5 WR 80% avg +1.799 · Spearman +0.38

bearish-day block applied · lR_b: scored 10/10 · R ≤ median 1.83: N 5 WR 40% avg -0.604 · R > median: N 5 WR 80% avg +1.799 · Spearman +0.33

Live-stamp vs rebuilt R on the same real fills: R_live~R_b Spearman +0.99 (N 16), R_live~R_cg Spearman +nan (N 0)

#### FRENZY backtest — TODAY's kept stack WITHOUT the bearish block (ATR ≤ 3.0, sequenced, FIX +3/−3 @ 8 s)

**Supply-drift check (bot-source mcap vs CoinGecko history)**

| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |
|---|---|---|---|---|
| 2026-01 | 13 | 1.861 | 1.63–2.21 | 0% |
| 2026-02 | 5 | 1.269 | 1.17–1.29 | 40% |
| 2026-03 | 11 | 1.253 | 1.14–1.39 | 27% |
| 2026-04 | 7 | 1.513 | 1.17–1.74 | 43% |
| 2026-05 | 9 | 1.144 | 1.04–1.23 | 78% |
| 2026-06 | 7 | 2.015 | 1.59–2.25 | 0% |
| 2026-07 | 2 | 2.562 | 1.86–3.26 | 50% |
| 2026-08 | 12 | 1.251 | 1.16–1.84 | 50% |
| 2026-09 | 4 | 1.063 | 1.02–1.13 | 75% |

Spearman(log R, log R_fut[CG mcap]) = +0.908 on 70 fills.


#### FRENZY backtest — TODAY's kept stack WITHOUT the bearish block (ATR ≤ 3.0, sequenced, FIX +3/−3 @ 8 s) — R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)

Coverage: 383/427 fills scored (90%). Unscored fills: N 44, avg +0.223 (NOT treated as 'rest'). Scored cohort: all scored | 383 | 187 | 157 | 54% | **+0.142** | [-0.16, +0.44] | +54.4 | -3.34 | 9% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 42 | 42 | 67% | +0.903 | 20 · +1.405 | 22 · +0.448 |
| 2026-02 | 33 | 33 | 52% | -0.040 | 17 · -0.345 | 16 · +0.285 |
| 2026-03 | 60 | 60 | 53% | +0.098 | 27 · +0.460 | 33 · -0.198 |
| 2026-04 | 53 | 53 | 43% | -0.493 | 27 · -0.208 | 26 · -0.788 |
| 2026-05 | 50 | 50 | 56% | +0.266 | 31 · -0.386 | 19 · +1.328 |
| 2026-06 | 28 | 28 | 57% | +0.332 | 13 · +0.137 | 15 · +0.500 |
| 2026-07 | 36 | 36 | 47% | -0.269 | 15 · +0.501 | 21 · -0.819 |
| 2026-08 | 42 | 42 | 57% | +0.335 | 19 · -0.254 | 23 · +0.821 |
| 2026-09 | 39 | 39 | 56% | +0.284 | 23 · +0.291 | 16 · +0.273 |

(pooled median R = 1.7)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.105–0.773 | 77 | 68 | 58 | 55% | **+0.164** | [-0.52, +0.82] | +12.6 | -3.16 | 8% |
| Q2 0.78–1.3 | 76 | 62 | 53 | 55% | **+0.219** | [-0.54, +0.94] | +16.6 | -3.16 | 15% |
| Q3 1.31–2.26 | 77 | 64 | 57 | 49% | **-0.140** | [-0.80, +0.57] | -10.8 | -3.18 | 17% |
| Q4 2.27–4.25 | 76 | 68 | 57 | 55% | **+0.219** | [-0.47, +0.91] | +16.7 | -3.14 | 17% |
| Q5 4.26–30.7 | 77 | 55 | 47 | 56% | **+0.250** | [-0.37, +0.85] | +19.2 | -3.34 | 16% |

bucket-order Spearman (bucket index vs avg) = +0.70 · trade-level Spearman(R, P&L) = -0.021

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.105–0.539 | 39 | 35 | 34 | 56% | **+0.288** | [-0.60, +1.15] | +11.2 | -3.14 | 14% |
| Q2 0.544–0.773 | 38 | 37 | 32 | 53% | **+0.037** | [-0.92, +1.05] | +1.4 | -3.16 | 13% |
| Q3 0.78–1.01 | 38 | 33 | 31 | 58% | **+0.379** | [-0.60, +1.30] | +14.4 | -3.12 | 14% |
| Q4 1.02–1.3 | 38 | 35 | 29 | 53% | **+0.058** | [-0.89, +1.08] | +2.2 | -3.16 | 28% |
| Q5 1.31–1.7 | 39 | 34 | 33 | 51% | **-0.026** | [-1.01, +1.01] | -1.0 | -3.18 | 14% |
| Q6 1.71–2.26 | 38 | 36 | 32 | 47% | **-0.257** | [-1.16, +0.64] | -9.7 | -3.14 | 18% |
| Q7 2.27–2.98 | 38 | 37 | 30 | 55% | **+0.220** | [-0.80, +1.17] | +8.4 | -3.13 | 19% |
| Q8 3.06–4.25 | 38 | 38 | 33 | 55% | **+0.219** | [-0.73, +1.17] | +8.3 | -3.14 | 15% |
| Q9 4.26–6.52 | 38 | 31 | 29 | 55% | **+0.214** | [-0.61, +1.07] | +8.1 | -3.34 | 25% |
| Q10 6.63–30.7 | 39 | 30 | 24 | 56% | **+0.284** | [-0.53, +1.10] | +11.1 | -3.17 | 19% |

bucket-order Spearman (bucket index vs avg) = -0.04 · trade-level Spearman(R, P&L) = -0.021

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block HIGH turnover, R > 1.18 (i.e. 24h volume = 118.2 % of mcap); Δ(high − low) = -0.451 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.817**.

**Expectancy bar on the blocked side**: blocked side = R > 1.18 → N 249, WR 51% vs kept breakeven WR 52%, avg -0.016 [-0.38, +0.34], P(avg<0) 0.525, days 145, top day 4% (2026-03-07), top pair 5% (PLAYUSDT); kept N 134 avg +0.435

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = +0.010 %/fill of the whole cohort → after 30–50 % haircut +0.005…+0.007.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 0.951 (in-sample Δ -0.383) | 113 · 55% · +0.193 | 32 · +0.090 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block low R ≤ 0.9 (in-sample Δ +0.465) | 67 · 55% · +0.204 | 171 · +0.094 | FAILS |
| leave-one-month-out (9 months) | per-month re-picked cut | 256 · 60% · +0.512 [+0.13, +0.87] | 127 · -0.603 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.08 | 383 |
| U2 | -0.04 | 383 |
| gvol | -0.03 | 383 |
| vol24 | +0.58 | 383 |
| gain_pct | +0.42 | 383 |
| run_pct | +0.48 | 383 |
| hours | +0.38 | 383 |
| bar_ret | -0.03 | 383 |
| vol_mult | +0.06 | 383 |
| above_streak | -0.04 | 383 |
| btc_1d_ret | -0.04 | 383 |
| btc_gap | +0.01 | 383 |
| lmcap | -0.44 | 383 |
| lvol | +0.58 | 383 |


#### FRENZY backtest — TODAY's kept stack WITHOUT the bearish block (ATR ≤ 3.0, sequenced, FIX +3/−3 @ 8 s) — R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)

Coverage: 70/427 fills scored (16%). Unscored fills: N 357, avg +0.098 (NOT treated as 'rest'). Scored cohort: all scored | 70 | 51 | 29 | 59% | **+0.417** | [-0.40, +1.16] | +29.2 | -3.14 | 22% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 13 | 13 | 92% | +2.448 | 6 · +2.917 | 7 · +2.046 |
| 2026-02 | 5 | 5 | 20% | -1.905 | 2 · -3.107 | 3 · -1.104 |
| 2026-03 | 11 | 11 | 55% | +0.171 | 8 · -0.104 | 3 · +0.904 |
| 2026-04 | 7 | 7 | 43% | -0.521 | 6 · -0.091 | 1 · -3.102 |
| 2026-05 | 9 | 9 | 67% | +0.900 | 5 · +1.706 | 4 · -0.109 |
| 2026-06 | 7 | 7 | 43% | -0.530 | 1 · +2.911 | 6 · -1.104 |
| 2026-07 | 2 | 2 | 100% | +2.905 | 0 · +nan | 2 · +2.905 |
| 2026-08 | 12 | 12 | 50% | -0.094 | 6 · -0.101 | 6 · -0.087 |
| 2026-09 | 4 | 4 | 50% | -0.104 | 1 · -3.104 | 3 · +0.896 |

(pooled median R = 2.05)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.238–1.23 | 14 | 14 | 13 | 64% | **+0.763** | [-0.53, +2.06] | +10.7 | -3.12 | 40% |
| Q2 1.24–1.79 | 14 | 10 | 9 | 43% | **-0.525** | [-2.61, +1.41] | -7.4 | -3.12 | 50% |
| Q3 1.82–2.95 | 14 | 11 | 8 | 71% | **+1.187** | [-0.61, +2.51] | +16.6 | -3.14 | 97% |
| Q4 2.96–5.16 | 14 | 13 | 8 | 50% | **-0.102** | [-1.72, +1.51] | -1.4 | -3.13 | 65% |
| Q5 5.23–30.9 | 14 | 13 | 10 | 64% | **+0.762** | [-0.70, +2.05] | +10.7 | -3.11 | 91% |

bucket-order Spearman (bucket index vs avg) = -0.10 · trade-level Spearman(R, P&L) = -0.041

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 1.81 (i.e. 24h volume = 181.1 % of mcap); Δ(high − low) = +0.497 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.945**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 1.81 → N 28, WR 54% vs kept breakeven WR 52%, avg +0.119 [-1.18, +1.36], P(avg<0) 0.408, days 23, top day 15% (2026-03-07), top pair 15% (BANANAS31USDT); kept N 42 avg +0.616

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.048 %/fill of the whole cohort → after 30–50 % haircut -0.024…-0.033.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | too few | | | |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 32 · 66% · +0.845 [-0.20, +1.85] | 38 · +0.056 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.23 | 70 |
| U2 | -0.27 | 70 |
| gvol | -0.11 | 70 |
| vol24 | +0.37 | 70 |
| gain_pct | +0.45 | 70 |
| run_pct | +0.47 | 70 |
| hours | +0.33 | 70 |
| bar_ret | +0.19 | 70 |
| vol_mult | +0.11 | 70 |
| above_streak | -0.01 | 70 |
| btc_1d_ret | +0.00 | 70 |
| btc_gap | +0.12 | 70 |
| lmcap | -0.51 | 70 |
| lvol | +0.37 | 70 |


#### FRENZY backtest — TODAY's kept stack WITHOUT the bearish block (ATR ≤ 3.0, sequenced, FIX +3/−3 @ 8 s) — R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap

Coverage: 70/427 fills scored (16%). Unscored fills: N 357, avg +0.098 (NOT treated as 'rest'). Scored cohort: all scored | 70 | 51 | 29 | 59% | **+0.417** | [-0.37, +1.19] | +29.2 | -3.14 | 22% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 13 | 13 | 92% | +2.448 | 2 · +2.905 | 11 · +2.365 |
| 2026-02 | 5 | 5 | 20% | -1.905 | 2 · -3.109 | 3 · -1.103 |
| 2026-03 | 11 | 11 | 55% | +0.171 | 6 · -0.102 | 5 · +0.499 |
| 2026-04 | 7 | 7 | 43% | -0.521 | 5 · -0.692 | 2 · -0.095 |
| 2026-05 | 9 | 9 | 67% | +0.900 | 5 · +1.701 | 4 · -0.102 |
| 2026-06 | 7 | 7 | 43% | -0.530 | 5 · -0.703 | 2 · -0.097 |
| 2026-07 | 2 | 2 | 100% | +2.905 | 1 · +2.906 | 1 · +2.903 |
| 2026-08 | 12 | 12 | 50% | -0.094 | 8 · -0.100 | 4 · -0.083 |
| 2026-09 | 4 | 4 | 50% | -0.104 | 1 · +2.901 | 3 · -1.106 |

(pooled median R = 0.481)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0598–0.159 | 14 | 12 | 11 | 36% | **-0.959** | [-2.36, +0.77] | -13.4 | -3.14 | 38% |
| Q2 0.159–0.311 | 14 | 9 | 6 | 71% | **+1.188** | [-0.38, +2.48] | +16.6 | -3.12 | 100% |
| Q3 0.317–0.594 | 14 | 12 | 10 | 57% | **+0.328** | [-0.96, +1.62] | +4.6 | -3.13 | 50% |
| Q4 0.63–1.84 | 14 | 10 | 7 | 64% | **+0.764** | [-1.10, +2.16] | +10.7 | -3.11 | 91% |
| Q5 1.84–13 | 14 | 13 | 10 | 64% | **+0.763** | [-0.79, +2.16] | +10.7 | -3.11 | 64% |

bucket-order Spearman (bucket index vs avg) = +0.30 · trade-level Spearman(R, P&L) = +0.191

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 0.878 (i.e. 24h volume = 87.8 % of mcap); Δ(high − low) = +0.888 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.470**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 0.878 → N 45, WR 53% vs kept breakeven WR 52%, avg +0.100 [-0.87, +1.07], P(avg<0) 0.404, days 33, top day 17% (2026-03-07), top pair 9% (BANANAS31USDT); kept N 25 avg +0.988

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.064 %/fill of the whole cohort → after 30–50 % haircut -0.032…-0.045.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | too few | | | |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 46 · 65% · +0.816 [-0.10, +1.68] | 24 · -0.349 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | -0.06 | 70 |
| U2 | +0.04 | 70 |
| gvol | +0.06 | 70 |
| vol24 | +0.32 | 70 |
| gain_pct | +0.16 | 70 |
| run_pct | +0.16 | 70 |
| hours | +0.43 | 70 |
| bar_ret | +0.23 | 70 |
| vol_mult | -0.11 | 70 |
| above_streak | +0.07 | 70 |
| btc_1d_ret | -0.01 | 70 |
| btc_gap | +0.16 | 70 |
| lmcap | -0.17 | 70 |
| lvol | +0.32 | 70 |


#### FRENZY backtest — TODAY's kept stack WITH the bearish-day block (live today)

**Supply-drift check (bot-source mcap vs CoinGecko history)**

| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |
|---|---|---|---|---|
| 2026-01 | 8 | 1.726 | 1.62–1.95 | 0% |
| 2026-02 | 4 | 1.230 | 1.14–1.33 | 50% |
| 2026-03 | 5 | 1.372 | 1.25–1.42 | 20% |
| 2026-04 | 5 | 1.513 | 1.16–1.61 | 40% |
| 2026-05 | 7 | 1.220 | 1.14–1.27 | 71% |
| 2026-06 | 7 | 2.015 | 1.59–2.25 | 0% |
| 2026-07 | 2 | 2.562 | 1.86–3.26 | 50% |
| 2026-08 | 11 | 1.245 | 1.15–1.94 | 55% |
| 2026-09 | 3 | 1.058 | 0.99–1.19 | 67% |

Spearman(log R, log R_fut[CG mcap]) = +0.948 on 52 fills.


#### FRENZY backtest — TODAY's kept stack WITH the bearish-day block (live today) — R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)

Coverage: 284/316 fills scored (90%). Unscored fills: N 32, avg +0.909 (NOT treated as 'rest'). Scored cohort: all scored | 284 | 151 | 138 | 54% | **+0.109** | [-0.23, +0.44] | +31.0 | -3.34 | 9% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 31 | 31 | 68% | +0.970 | 16 · +1.407 | 15 · +0.503 |
| 2026-02 | 25 | 25 | 52% | -0.021 | 13 · -0.415 | 12 · +0.407 |
| 2026-03 | 39 | 39 | 59% | +0.435 | 18 · +0.571 | 21 · +0.319 |
| 2026-04 | 43 | 43 | 42% | -0.586 | 20 · -0.402 | 23 · -0.747 |
| 2026-05 | 41 | 41 | 56% | +0.272 | 26 · -0.325 | 15 · +1.307 |
| 2026-06 | 19 | 19 | 58% | +0.376 | 8 · +0.657 | 11 · +0.172 |
| 2026-07 | 25 | 25 | 52% | +0.017 | 11 · +0.168 | 14 · -0.102 |
| 2026-08 | 30 | 30 | 47% | -0.295 | 11 · -1.463 | 19 · +0.381 |
| 2026-09 | 31 | 31 | 52% | -0.004 | 19 · +0.374 | 12 · -0.603 |

(pooled median R = 1.69)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.105–0.743 | 57 | 52 | 46 | 54% | **+0.147** | [-0.64, +0.95] | +8.4 | -3.16 | 11% |
| Q2 0.754–1.24 | 57 | 42 | 40 | 54% | **+0.167** | [-0.72, +1.02] | +9.5 | -3.16 | 18% |
| Q3 1.26–2.13 | 56 | 51 | 46 | 46% | **-0.314** | [-1.10, +0.50] | -17.6 | -3.14 | 19% |
| Q4 2.13–4.28 | 57 | 54 | 47 | 58% | **+0.380** | [-0.36, +1.13] | +21.6 | -3.14 | 16% |
| Q5 4.34–29 | 57 | 43 | 39 | 54% | **+0.160** | [-0.54, +0.80] | +9.1 | -3.34 | 14% |

bucket-order Spearman (bucket index vs avg) = +0.30 · trade-level Spearman(R, P&L) = +0.009

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.105–0.524 | 29 | 27 | 27 | 59% | **+0.421** | [-0.66, +1.42] | +12.2 | -3.14 | 20% |
| Q2 0.524–0.743 | 28 | 27 | 21 | 50% | **-0.138** | [-1.24, +0.98] | -3.9 | -3.16 | 22% |
| Q3 0.754–0.981 | 28 | 23 | 23 | 57% | **+0.337** | [-0.85, +1.46] | +9.4 | -3.12 | 27% |
| Q4 0.986–1.24 | 29 | 26 | 24 | 52% | **+0.003** | [-1.18, +1.19] | +0.1 | -3.16 | 17% |
| Q5 1.26–1.69 | 28 | 26 | 26 | 46% | **-0.316** | [-1.49, +0.90] | -8.8 | -3.13 | 15% |
| Q6 1.7–2.13 | 28 | 27 | 24 | 46% | **-0.312** | [-1.39, +0.69] | -8.7 | -3.14 | 23% |
| Q7 2.13–2.92 | 29 | 28 | 26 | 52% | **+0.006** | [-1.04, +1.05] | +0.2 | -3.13 | 18% |
| Q8 2.96–4.28 | 28 | 28 | 25 | 64% | **+0.766** | [-0.31, +1.84] | +21.5 | -3.14 | 37% |
| Q9 4.34–6.52 | 28 | 25 | 23 | 54% | **+0.110** | [-0.89, +1.06] | +3.1 | -3.34 | 21% |
| Q10 6.63–29 | 29 | 22 | 19 | 55% | **+0.208** | [-0.63, +1.04] | +6.0 | -3.17 | 27% |

bucket-order Spearman (bucket index vs avg) = +0.10 · trade-level Spearman(R, P&L) = +0.009

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 2.93 (i.e. 24h volume = 292.7 % of mcap); Δ(high − low) = +0.357 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.966**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 2.93 → N 199, WR 52% vs kept breakeven WR 52%, avg +0.002 [-0.44, +0.42], P(avg<0) 0.488, days 124, top day 3% (2026-02-16), top pair 5% (SKLUSDT); kept N 85 avg +0.359

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.002 %/fill of the whole cohort → after 30–50 % haircut -0.001…-0.001.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block low R ≤ 0.669 (in-sample Δ +0.389) | 13 · 62% · +0.598 | 92 · -0.100 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block high R > 1.16 (in-sample Δ -0.960) | 108 · 55% · +0.179 | 71 · +0.186 | holds |
| leave-one-month-out (9 months) | per-month re-picked cut | 191 · 58% · +0.358 [-0.05, +0.77] | 93 · -0.402 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.05 | 284 |
| U2 | -0.03 | 284 |
| gvol | -0.02 | 284 |
| vol24 | +0.56 | 284 |
| gain_pct | +0.41 | 284 |
| run_pct | +0.47 | 284 |
| hours | +0.39 | 284 |
| bar_ret | -0.04 | 284 |
| vol_mult | +0.04 | 284 |
| above_streak | -0.05 | 284 |
| btc_1d_ret | -0.04 | 284 |
| btc_gap | +0.08 | 284 |
| lmcap | -0.48 | 284 |
| lvol | +0.56 | 284 |


#### FRENZY backtest — TODAY's kept stack WITH the bearish-day block (live today) — R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)

Coverage: 52/316 fills scored (16%). Unscored fills: N 264, avg +0.156 (NOT treated as 'rest'). Scored cohort: all scored | 52 | 40 | 27 | 58% | **+0.365** | [-0.51, +1.16] | +19.0 | -3.14 | 31% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 8 | 8 | 88% | +2.162 | 6 · +2.917 | 2 · -0.102 |
| 2026-02 | 4 | 4 | 25% | -1.605 | 2 · -3.107 | 2 · -0.103 |
| 2026-03 | 5 | 5 | 80% | +1.704 | 3 · +0.900 | 2 · +2.909 |
| 2026-04 | 5 | 5 | 40% | -0.693 | 5 · -0.693 | 0 · +nan |
| 2026-05 | 7 | 7 | 71% | +1.186 | 4 · +1.407 | 3 · +0.892 |
| 2026-06 | 7 | 7 | 43% | -0.530 | 1 · +2.911 | 6 · -1.104 |
| 2026-07 | 2 | 2 | 100% | +2.905 | 0 · +nan | 2 · +2.905 |
| 2026-08 | 11 | 11 | 45% | -0.367 | 4 · -1.602 | 7 · +0.339 |
| 2026-09 | 3 | 3 | 33% | -1.110 | 1 · -3.104 | 2 · -0.114 |

(pooled median R = 1.96)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.238–1.27 | 11 | 11 | 11 | 45% | **-0.369** | [-2.02, +1.28] | -4.1 | -3.12 | 33% |
| Q2 1.4–1.74 | 10 | 7 | 6 | 50% | **-0.094** | [-2.50, +1.99] | -0.9 | -3.11 | 80% |
| Q3 1.79–2.88 | 10 | 8 | 7 | 80% | **+1.703** | [-0.11, +2.91] | +17.0 | -3.14 | 100% |
| Q4 2.96–5.16 | 10 | 10 | 7 | 50% | **-0.103** | [-1.91, +1.71] | -1.0 | -3.13 | 75% |
| Q5 5.53–30.9 | 11 | 10 | 8 | 64% | **+0.725** | [-0.91, +2.31] | +8.0 | -3.11 | 94% |

bucket-order Spearman (bucket index vs avg) = +0.50 · trade-level Spearman(R, P&L) = +0.013

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block HIGH turnover, R > 1.96 (i.e. 24h volume = 196.4 % of mcap); Δ(high − low) = -0.005 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.806**.

**Expectancy bar on the blocked side**: blocked side = R > 1.96 → N 26, WR 58% vs kept breakeven WR 52%, avg +0.362 [-0.75, +1.40], P(avg<0) 0.241, days 23, top day 11% (2026-05-17), top pair 41% (BTRUSDT); kept N 26 avg +0.367

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.181 %/fill of the whole cohort → after 30–50 % haircut -0.091…-0.127.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | too few | | | |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 2 · 100% · +2.905 [+2.90, +2.91] | 0 · +nan | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.28 | 52 |
| U2 | -0.24 | 52 |
| gvol | -0.07 | 52 |
| vol24 | +0.27 | 52 |
| gain_pct | +0.44 | 52 |
| run_pct | +0.45 | 52 |
| hours | +0.30 | 52 |
| bar_ret | +0.14 | 52 |
| vol_mult | +0.16 | 52 |
| above_streak | -0.01 | 52 |
| btc_1d_ret | -0.07 | 52 |
| btc_gap | +0.16 | 52 |
| lmcap | -0.65 | 52 |
| lvol | +0.27 | 52 |


#### FRENZY backtest — TODAY's kept stack WITH the bearish-day block (live today) — R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap

Coverage: 52/316 fills scored (16%). Unscored fills: N 264, avg +0.156 (NOT treated as 'rest'). Scored cohort: all scored | 52 | 40 | 27 | 58% | **+0.365** | [-0.56, +1.18] | +19.0 | -3.14 | 31% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 8 | 8 | 88% | +2.162 | 1 · +2.910 | 7 · +2.056 |
| 2026-02 | 4 | 4 | 25% | -1.605 | 1 · -3.112 | 3 · -1.103 |
| 2026-03 | 5 | 5 | 80% | +1.704 | 4 · +1.403 | 1 · +2.905 |
| 2026-04 | 5 | 5 | 40% | -0.693 | 4 · -0.089 | 1 · -3.109 |
| 2026-05 | 7 | 7 | 71% | +1.186 | 4 · +1.400 | 3 · +0.901 |
| 2026-06 | 7 | 7 | 43% | -0.530 | 4 · -0.103 | 3 · -1.099 |
| 2026-07 | 2 | 2 | 100% | +2.905 | 0 · +nan | 2 · +2.905 |
| 2026-08 | 11 | 11 | 45% | -0.367 | 7 · -0.529 | 4 · -0.083 |
| 2026-09 | 3 | 3 | 33% | -1.110 | 1 · +2.901 | 2 · -3.116 |

(pooled median R = 0.342)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0598–0.159 | 11 | 10 | 9 | 45% | **-0.372** | [-2.11, +1.71] | -4.1 | -3.14 | 41% |
| Q2 0.161–0.276 | 10 | 6 | 4 | 70% | **+1.101** | [-0.86, +2.41] | +11.0 | -3.12 | 100% |
| Q3 0.305–0.546 | 10 | 8 | 8 | 50% | **-0.100** | [-1.61, +1.41] | -1.0 | -3.13 | 64% |
| Q4 0.594–1.73 | 10 | 7 | 6 | 60% | **+0.506** | [-2.25, +2.31] | +5.1 | -3.11 | 75% |
| Q5 1.84–13 | 11 | 11 | 8 | 64% | **+0.726** | [-0.92, +2.37] | +8.0 | -3.11 | 94% |

bucket-order Spearman (bucket index vs avg) = +0.40 · trade-level Spearman(R, P&L) = +0.139

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 0.342 (i.e. 24h volume = 34.2 % of mcap); Δ(high − low) = +0.003 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.850**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 0.342 → N 26, WR 58% vs kept breakeven WR 52%, avg +0.363 [-0.92, +1.47], P(avg<0) 0.258, days 18, top day 25% (2026-08-14), top pair 20% (FIDAUSDT); kept N 26 avg +0.367

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.182 %/fill of the whole cohort → after 30–50 % haircut -0.091…-0.127.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | too few | | | |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 0 · nan% · +nan [+nan, +nan] | 0 · +nan | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | -0.04 | 52 |
| U2 | +0.03 | 52 |
| gvol | +0.08 | 52 |
| vol24 | +0.31 | 52 |
| gain_pct | +0.18 | 52 |
| run_pct | +0.17 | 52 |
| hours | +0.40 | 52 |
| bar_ret | +0.25 | 52 |
| vol_mult | -0.07 | 52 |
| above_streak | +0.09 | 52 |
| btc_1d_ret | -0.04 | 52 |
| btc_gap | +0.18 | 52 |
| lmcap | -0.29 | 52 |
| lvol | +0.31 | 52 |


#### FRENZY backtest — every gated signal, any ATR, unsequenced (larger N, includes ATR-refused)

**Supply-drift check (bot-source mcap vs CoinGecko history)**

| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |
|---|---|---|---|---|
| 2026-01 | 15 | 1.861 | 1.61–2.34 | 0% |
| 2026-02 | 6 | 1.263 | 1.19–1.29 | 33% |
| 2026-03 | 14 | 1.216 | 1.13–1.41 | 36% |
| 2026-04 | 10 | 1.382 | 1.16–1.71 | 40% |
| 2026-05 | 12 | 1.182 | 1.03–1.34 | 67% |
| 2026-06 | 9 | 1.783 | 1.40–2.19 | 11% |
| 2026-07 | 4 | 2.808 | 1.67–3.82 | 25% |
| 2026-08 | 15 | 1.245 | 1.15–1.88 | 53% |
| 2026-09 | 8 | 1.161 | 1.04–1.45 | 50% |

Spearman(log R, log R_fut[CG mcap]) = +0.900 on 93 fills.


#### FRENZY backtest — every gated signal, any ATR, unsequenced (larger N, includes ATR-refused) — R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)

Coverage: 522/595 fills scored (88%). Unscored fills: N 73, avg -0.120 (NOT treated as 'rest'). Scored cohort: all scored | 522 | 216 | 182 | 51% | **-0.033** | [-0.30, +0.23] | -17.1 | -3.34 | 8% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 51 | 51 | 61% | +0.549 | 27 · +0.680 | 24 · +0.402 |
| 2026-02 | 38 | 38 | 58% | +0.351 | 18 · -0.164 | 20 · +0.814 |
| 2026-03 | 70 | 70 | 50% | -0.102 | 33 · +0.361 | 37 · -0.515 |
| 2026-04 | 91 | 91 | 42% | -0.594 | 49 · -0.407 | 42 · -0.812 |
| 2026-05 | 60 | 60 | 52% | +0.003 | 37 · -0.502 | 23 · +0.816 |
| 2026-06 | 46 | 46 | 48% | -0.230 | 19 · -0.254 | 27 · -0.213 |
| 2026-07 | 53 | 53 | 45% | -0.384 | 19 · +0.693 | 34 · -0.986 |
| 2026-08 | 60 | 60 | 58% | +0.404 | 27 · -0.208 | 33 · +0.906 |
| 2026-09 | 53 | 53 | 55% | +0.176 | 32 · +0.265 | 21 · +0.040 |

(pooled median R = 1.77)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0997–0.803 | 105 | 84 | 77 | 51% | **-0.023** | [-0.62, +0.57] | -2.4 | -3.16 | 10% |
| Q2 0.807–1.36 | 104 | 75 | 70 | 56% | **+0.248** | [-0.40, +0.89] | +25.8 | -3.17 | 12% |
| Q3 1.37–2.38 | 104 | 82 | 73 | 42% | **-0.564** | [-1.11, +0.04] | -58.7 | -3.18 | 11% |
| Q4 2.39–4.41 | 104 | 87 | 71 | 56% | **+0.248** | [-0.33, +0.85] | +25.8 | -3.34 | 13% |
| Q5 4.41–30.7 | 105 | 74 | 54 | 50% | **-0.072** | [-0.63, +0.47] | -7.6 | -3.18 | 18% |

bucket-order Spearman (bucket index vs avg) = -0.10 · trade-level Spearman(R, P&L) = -0.026

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0997–0.539 | 53 | 46 | 44 | 53% | **+0.072** | [-0.66, +0.83] | +3.8 | -3.15 | 20% |
| Q2 0.544–0.803 | 52 | 50 | 45 | 50% | **-0.119** | [-0.88, +0.71] | -6.2 | -3.16 | 13% |
| Q3 0.807–1.04 | 52 | 43 | 39 | 58% | **+0.364** | [-0.49, +1.13] | +18.9 | -3.15 | 17% |
| Q4 1.05–1.36 | 52 | 44 | 42 | 54% | **+0.131** | [-0.70, +1.02] | +6.8 | -3.17 | 21% |
| Q5 1.37–1.77 | 52 | 45 | 43 | 44% | **-0.449** | [-1.26, +0.36] | -23.4 | -3.18 | 20% |
| Q6 1.77–2.38 | 52 | 46 | 44 | 40% | **-0.679** | [-1.44, +0.09] | -35.3 | -3.14 | 8% |
| Q7 2.39–3.2 | 52 | 48 | 43 | 56% | **+0.249** | [-0.63, +1.05] | +12.9 | -3.14 | 17% |
| Q8 3.2–4.41 | 52 | 49 | 43 | 56% | **+0.248** | [-0.51, +1.03] | +12.9 | -3.34 | 21% |
| Q9 4.41–6.78 | 52 | 42 | 37 | 48% | **-0.216** | [-0.99, +0.58] | -11.2 | -3.18 | 20% |
| Q10 6.83–30.7 | 53 | 41 | 30 | 53% | **+0.069** | [-0.68, +0.79] | +3.6 | -3.18 | 29% |

bucket-order Spearman (bucket index vs avg) = -0.13 · trade-level Spearman(R, P&L) = -0.026

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block HIGH turnover, R > 1.37 (i.e. 24h volume = 136.8 % of mcap); Δ(high − low) = -0.241 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.992**.

**Expectancy bar on the blocked side**: blocked side = R > 1.37 → N 313, WR 50% vs kept breakeven WR 52%, avg -0.129 [-0.47, +0.19], P(avg<0) 0.789, days 173, top day 3% (2026-04-05), top pair 6% (STOUSDT); kept N 209 avg +0.112

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = +0.077 %/fill of the whole cohort → after 30–50 % haircut +0.039…+0.054.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 0.961 (in-sample Δ -0.353) | 169 · 52% · +0.025 | 43 · -0.034 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block high R > 1.27 (in-sample Δ -0.511) | 183 · 50% · -0.116 | 127 · +0.012 | holds |
| leave-one-month-out (9 months) | per-month re-picked cut | 314 · 57% · +0.322 [-0.02, +0.65] | 208 · -0.568 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.12 | 522 |
| U2 | +0.00 | 522 |
| gvol | +0.01 | 522 |
| vol24 | +0.57 | 522 |
| gain_pct | +0.39 | 522 |
| run_pct | +0.44 | 522 |
| hours | +0.33 | 522 |
| bar_ret | -0.02 | 522 |
| vol_mult | +0.05 | 522 |
| above_streak | -0.08 | 522 |
| btc_1d_ret | -0.04 | 522 |
| btc_gap | +0.01 | 522 |
| lmcap | -0.48 | 522 |
| lvol | +0.57 | 522 |


#### FRENZY backtest — every gated signal, any ATR, unsequenced (larger N, includes ATR-refused) — R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)

Coverage: 93/595 fills scored (16%). Unscored fills: N 502, avg -0.099 (NOT treated as 'rest'). Scored cohort: all scored | 93 | 70 | 36 | 56% | **+0.256** | [-0.41, +0.89] | +23.8 | -3.14 | 24% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 15 | 15 | 87% | +2.109 | 8 · +2.165 | 7 · +2.046 |
| 2026-02 | 6 | 6 | 33% | -1.087 | 3 · -1.103 | 3 · -1.072 |
| 2026-03 | 14 | 14 | 50% | -0.102 | 8 · -0.104 | 6 · -0.099 |
| 2026-04 | 10 | 10 | 40% | -0.695 | 7 · +0.341 | 3 · -3.114 |
| 2026-05 | 12 | 12 | 58% | +0.402 | 8 · +0.657 | 4 · -0.109 |
| 2026-06 | 9 | 9 | 56% | +0.239 | 1 · +2.911 | 8 · -0.095 |
| 2026-07 | 4 | 4 | 75% | +1.403 | 0 · +nan | 4 · +1.403 |
| 2026-08 | 15 | 15 | 47% | -0.296 | 9 · -0.436 | 6 · -0.087 |
| 2026-09 | 8 | 8 | 50% | -0.140 | 3 · -1.198 | 5 · +0.495 |

(pooled median R = 2.12)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.16–1.23 | 19 | 19 | 17 | 63% | **+0.683** | [-0.59, +1.95] | +13.0 | -3.12 | 33% |
| Q2 1.24–1.79 | 18 | 14 | 11 | 44% | **-0.430** | [-2.15, +1.20] | -7.7 | -3.12 | 50% |
| Q3 1.82–2.95 | 19 | 16 | 10 | 53% | **+0.056** | [-1.61, +1.47] | +1.1 | -3.14 | 47% |
| Q4 2.96–5.16 | 18 | 17 | 11 | 61% | **+0.572** | [-0.76, +1.85] | +10.3 | -3.13 | 89% |
| Q5 5.23–30.9 | 19 | 18 | 13 | 58% | **+0.378** | [-0.89, +1.64] | +7.2 | -3.13 | 47% |

bucket-order Spearman (bucket index vs avg) = -0.10 · trade-level Spearman(R, P&L) = -0.009

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 3.34 (i.e. 24h volume = 334.5 % of mcap); Δ(high − low) = +0.444 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.919**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 3.34 → N 60, WR 53% vs kept breakeven WR 52%, avg +0.098 [-0.76, +0.93], P(avg<0) 0.411, days 48, top day 12% (2026-03-07), top pair 18% (AXLUSDT); kept N 33 avg +0.542

WR < kept breakeven WR: FAIL · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = -0.063 %/fill of the whole cohort → after 30–50 % haircut -0.032…-0.044.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 1.71 (in-sample Δ -0.834) | 28 · 61% · +0.546 | 8 · -1.637 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 61 · 62% · +0.640 [-0.21, +1.41] | 32 · -0.477 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.14 | 93 |
| U2 | -0.19 | 93 |
| gvol | -0.03 | 93 |
| vol24 | +0.38 | 93 |
| gain_pct | +0.44 | 93 |
| run_pct | +0.47 | 93 |
| hours | +0.28 | 93 |
| bar_ret | +0.09 | 93 |
| vol_mult | +0.12 | 93 |
| above_streak | -0.04 | 93 |
| btc_1d_ret | +0.03 | 93 |
| btc_gap | +0.14 | 93 |
| lmcap | -0.48 | 93 |
| lvol | +0.38 | 93 |


#### FRENZY backtest — every gated signal, any ATR, unsequenced (larger N, includes ATR-refused) — R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap

Coverage: 93/595 fills scored (16%). Unscored fills: N 502, avg -0.099 (NOT treated as 'rest'). Scored cohort: all scored | 93 | 70 | 36 | 56% | **+0.256** | [-0.42, +0.88] | +23.8 | -3.14 | 24% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 15 | 15 | 87% | +2.109 | 4 · +1.406 | 11 · +2.365 |
| 2026-02 | 6 | 6 | 33% | -1.087 | 2 · -3.109 | 4 · -0.076 |
| 2026-03 | 14 | 14 | 50% | -0.102 | 8 · -0.101 | 6 · -0.103 |
| 2026-04 | 10 | 10 | 40% | -0.695 | 6 · -1.095 | 4 · -0.096 |
| 2026-05 | 12 | 12 | 58% | +0.402 | 7 · +1.190 | 5 · -0.703 |
| 2026-06 | 9 | 9 | 56% | +0.239 | 5 · -0.703 | 4 · +1.417 |
| 2026-07 | 4 | 4 | 75% | +1.403 | 0 · +nan | 4 · +1.403 |
| 2026-08 | 15 | 15 | 47% | -0.296 | 10 · -0.702 | 5 · +0.514 |
| 2026-09 | 8 | 8 | 50% | -0.140 | 5 · +0.439 | 3 · -1.106 |

(pooled median R = 0.366)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0253–0.159 | 19 | 17 | 14 | 32% | **-1.207** | [-2.44, +0.24] | -22.9 | -3.14 | 34% |
| Q2 0.159–0.299 | 18 | 13 | 10 | 67% | **+0.886** | [-0.71, +2.12] | +15.9 | -3.12 | 50% |
| Q3 0.305–0.572 | 19 | 16 | 14 | 58% | **+0.381** | [-0.88, +1.65] | +7.2 | -3.13 | 39% |
| Q4 0.575–1.45 | 18 | 13 | 12 | 67% | **+0.908** | [-0.48, +2.09] | +16.3 | -3.12 | 73% |
| Q5 1.73–13 | 19 | 17 | 12 | 58% | **+0.377** | [-0.98, +1.64] | +7.2 | -3.13 | 39% |

bucket-order Spearman (bucket index vs avg) = +0.30 · trade-level Spearman(R, P&L) = +0.180

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 25): best = block LOW turnover, R ≤ 0.22 (i.e. 24h volume = 22.0 % of mcap); Δ(high − low) = +1.123 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.290**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 0.22 → N 28, WR 43% vs kept breakeven WR 52%, avg -0.529 [-1.80, +0.72], P(avg<0) 0.792, days 22, top day 13% (2026-03-07), top pair 17% (BANANAUSDT); kept N 65 avg +0.594

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = +0.159 %/fill of the whole cohort → after 30–50 % haircut +0.080…+0.112.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block low R ≤ 0.36 (in-sample Δ +0.876) | 20 · 45% · -0.417 | 16 · +0.658 | holds |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 41 · 51% · -0.023 [-1.04, +0.91] | 52 · +0.475 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | -0.13 | 93 |
| U2 | +0.15 | 93 |
| gvol | +0.09 | 93 |
| vol24 | +0.37 | 93 |
| gain_pct | +0.10 | 93 |
| run_pct | +0.11 | 93 |
| hours | +0.42 | 93 |
| bar_ret | +0.13 | 93 |
| vol_mult | -0.15 | 93 |
| above_streak | +0.07 | 93 |
| btc_1d_ret | -0.05 | 93 |
| btc_gap | +0.06 | 93 |
| lmcap | -0.16 | 93 |
| lvol | +0.37 | 93 |


#### FRENZY_WILLY entry A — first flag (+1 net / −3 / 60 min, real ticks)

**Supply-drift check (bot-source mcap vs CoinGecko history)**

| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |
|---|---|---|---|---|
| 2026-01 | 33 | 1.328 | 1.18–1.95 | 36% |
| 2026-02 | 35 | 1.167 | 1.03–1.44 | 51% |
| 2026-03 | 32 | 1.152 | 1.05–1.25 | 72% |
| 2026-04 | 36 | 1.282 | 1.18–1.49 | 44% |
| 2026-05 | 27 | 1.182 | 1.09–1.28 | 70% |
| 2026-06 | 24 | 1.373 | 1.20–1.61 | 29% |
| 2026-07 | 20 | 1.284 | 1.15–1.77 | 45% |
| 2026-08 | 33 | 1.229 | 1.12–1.35 | 52% |
| 2026-09 | 49 | 1.163 | 1.08–1.44 | 55% |
| 2026-10 | 6 | 1.094 | 1.02–1.15 | 100% |

Spearman(log R, log R_fut[CG mcap]) = +0.956 on 295 fills.


#### FRENZY_WILLY entry A — first flag (+1 net / −3 / 60 min, real ticks) — R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)

Coverage: 1609/1781 fills scored (90%). Unscored fills: N 172, avg -0.096 (NOT treated as 'rest'). Scored cohort: all scored | 1609 | 271 | 348 | 73% | **-0.003** | [-0.08, +0.08] | -4.9 | -3.45 | 6% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 141 | 141 | 72% | -0.020 | 95 · +0.046 | 46 · -0.155 |
| 2026-02 | 158 | 158 | 75% | +0.070 | 99 · +0.079 | 59 · +0.055 |
| 2026-03 | 161 | 161 | 71% | -0.072 | 77 · +0.043 | 84 · -0.177 |
| 2026-04 | 206 | 206 | 78% | +0.149 | 97 · -0.026 | 109 · +0.304 |
| 2026-05 | 196 | 196 | 76% | +0.093 | 116 · +0.160 | 80 · -0.005 |
| 2026-06 | 161 | 161 | 72% | -0.094 | 67 · +0.127 | 94 · -0.252 |
| 2026-07 | 141 | 141 | 75% | +0.074 | 51 · +0.449 | 90 · -0.138 |
| 2026-08 | 209 | 209 | 69% | -0.201 | 82 · +0.200 | 127 · -0.459 |
| 2026-09 | 206 | 206 | 72% | -0.066 | 108 · -0.013 | 98 · -0.125 |
| 2026-10 | 30 | 30 | 83% | +0.328 | 13 · +0.376 | 17 · +0.290 |

(pooled median R = 0.549)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.00989–0.219 | 322 | 172 | 130 | 75% | **+0.068** | [-0.11, +0.23] | +21.9 | -3.20 | 14% |
| Q2 0.22–0.409 | 322 | 184 | 147 | 74% | **+0.047** | [-0.14, +0.23] | +15.2 | -3.13 | 10% |
| Q3 0.41–0.735 | 321 | 188 | 152 | 77% | **+0.143** | [-0.02, +0.29] | +45.8 | -3.41 | 12% |
| Q4 0.737–1.4 | 322 | 191 | 141 | 72% | **-0.078** | [-0.27, +0.12] | -25.1 | -3.45 | 9% |
| Q5 1.41–44.6 | 322 | 191 | 109 | 69% | **-0.195** | [-0.40, +0.01] | -62.7 | -3.35 | 14% |

bucket-order Spearman (bucket index vs avg) = -0.70 · trade-level Spearman(R, P&L) = -0.053

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.00989–0.119 | 161 | 110 | 79 | 77% | **+0.175** | [-0.05, +0.39] | +28.2 | -3.09 | 22% |
| Q2 0.12–0.219 | 161 | 114 | 87 | 73% | **-0.039** | [-0.28, +0.20] | -6.3 | -3.20 | 13% |
| Q3 0.22–0.299 | 161 | 123 | 100 | 72% | **-0.011** | [-0.28, +0.25] | -1.8 | -3.13 | 14% |
| Q4 0.3–0.409 | 161 | 123 | 100 | 76% | **+0.106** | [-0.16, +0.35] | +17.0 | -3.13 | 14% |
| Q5 0.41–0.549 | 161 | 121 | 99 | 80% | **+0.284** | [+0.05, +0.50] | +45.7 | -3.41 | 13% |
| Q6 0.553–0.735 | 160 | 122 | 97 | 74% | **+0.001** | [-0.24, +0.24] | +0.1 | -3.26 | 12% |
| Q7 0.737–0.976 | 161 | 123 | 91 | 74% | **+0.002** | [-0.28, +0.25] | +0.3 | -3.45 | 13% |
| Q8 0.977–1.4 | 161 | 119 | 94 | 70% | **-0.158** | [-0.45, +0.10] | -25.4 | -3.15 | 12% |
| Q9 1.41–2.37 | 161 | 123 | 81 | 71% | **-0.092** | [-0.38, +0.18] | -14.8 | -3.35 | 13% |
| Q10 2.38–44.6 | 161 | 132 | 77 | 68% | **-0.298** | [-0.58, -0.02] | -47.9 | -3.17 | 16% |

bucket-order Spearman (bucket index vs avg) = -0.61 · trade-level Spearman(R, P&L) = -0.053

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block HIGH turnover, R > 2.37 (i.e. 24h volume = 237.5 % of mcap); Δ(high − low) = -0.327 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.052**.

**Expectancy bar on the blocked side**: blocked side = R > 2.37 → N 161, WR 68% vs kept breakeven WR 73%, avg -0.298 [-0.58, -0.02], P(avg<0) 0.983, days 132, top day 4% (2026-06-25), top pair 10% (PORTALUSDT); kept N 1448 avg +0.030

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): PASS · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar PASSED.** In-sample Δ from blocking = +0.030 %/fill of the whole cohort → after 30–50 % haircut +0.015…+0.021.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 1.13 (in-sample Δ -0.222) | 242 · 69% · -0.237 | 505 · +0.013 | holds |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block high R > 0.702 (in-sample Δ -0.411) | 289 · 74% · +0.019 | 573 · +0.070 | holds |
| leave-one-month-out (10 months) | per-month re-picked cut | 175 · 69% · -0.240 [-0.51, +0.02] | 1434 · +0.026 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| vol24 | +0.16 | 1609 |
| btc_1d_ret | -0.02 | 1601 |
| btc_gap | -0.03 | 1609 |
| lmcap | -0.79 | 1609 |
| lvol | +0.16 | 1609 |


#### FRENZY_WILLY entry A — first flag (+1 net / −3 / 60 min, real ticks) — R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)

Coverage: 304/1781 fills scored (17%). Unscored fills: N 1477, avg -0.004 (NOT treated as 'rest'). Scored cohort: all scored | 304 | 181 | 68 | 72% | **-0.052** | [-0.25, +0.15] | -15.9 | -3.45 | 24% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 34 | 34 | 79% | +0.298 | 16 · +0.747 | 18 · -0.100 |
| 2026-02 | 35 | 35 | 57% | -0.556 | 10 · -1.076 | 25 · -0.347 |
| 2026-03 | 32 | 32 | 62% | -0.294 | 18 · -0.414 | 14 · -0.140 |
| 2026-04 | 38 | 38 | 82% | +0.280 | 17 · +0.361 | 21 · +0.215 |
| 2026-05 | 30 | 30 | 77% | +0.047 | 20 · +0.349 | 10 · -0.555 |
| 2026-06 | 24 | 24 | 79% | +0.155 | 9 · -0.802 | 15 · +0.730 |
| 2026-07 | 20 | 20 | 75% | +0.092 | 7 · +0.745 | 13 · -0.260 |
| 2026-08 | 33 | 33 | 70% | -0.231 | 12 · +0.324 | 21 · -0.547 |
| 2026-09 | 52 | 52 | 67% | -0.176 | 39 · -0.020 | 13 · -0.643 |
| 2026-10 | 6 | 6 | 83% | +0.319 | 4 · +1.000 | 2 · -1.042 |

(pooled median R = 0.585)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0326–0.252 | 61 | 57 | 20 | 77% | **+0.063** | [-0.37, +0.46] | +3.9 | -3.08 | 43% |
| Q2 0.258–0.456 | 61 | 54 | 30 | 77% | **+0.236** | [-0.20, +0.63] | +14.4 | -3.08 | 32% |
| Q3 0.457–0.764 | 60 | 53 | 28 | 62% | **-0.407** | [-0.89, +0.03] | -24.4 | -3.26 | 32% |
| Q4 0.77–1.69 | 61 | 54 | 29 | 75% | **+0.064** | [-0.39, +0.47] | +3.9 | -3.45 | 33% |
| Q5 1.71–12.1 | 61 | 57 | 24 | 67% | **-0.224** | [-0.71, +0.22] | -13.7 | -3.11 | 42% |

bucket-order Spearman (bucket index vs avg) = -0.30 · trade-level Spearman(R, P&L) = -0.077

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0326–0.135 | 31 | 31 | 13 | 74% | **-0.058** | [-0.68, +0.52] | -1.8 | -3.08 | 47% |
| Q2 0.136–0.252 | 30 | 27 | 12 | 80% | **+0.188** | [-0.40, +0.73] | +5.6 | -3.08 | 72% |
| Q3 0.258–0.367 | 30 | 29 | 16 | 80% | **+0.441** | [-0.02, +0.83] | +13.2 | -3.08 | 91% |
| Q4 0.372–0.456 | 31 | 28 | 22 | 74% | **+0.039** | [-0.63, +0.72] | +1.2 | -3.08 | 37% |
| Q5 0.457–0.584 | 30 | 27 | 19 | 67% | **-0.211** | [-0.86, +0.41] | -6.3 | -3.12 | 33% |
| Q6 0.586–0.764 | 30 | 30 | 17 | 57% | **-0.603** | [-1.33, +0.08] | -18.1 | -3.26 | 49% |
| Q7 0.77–1.19 | 31 | 28 | 17 | 77% | **+0.069** | [-0.58, +0.62] | +2.1 | -3.45 | 55% |
| Q8 1.2–1.69 | 30 | 29 | 19 | 73% | **+0.059** | [-0.57, +0.60] | +1.8 | -3.15 | 44% |
| Q9 1.71–2.51 | 30 | 30 | 18 | 67% | **-0.259** | [-0.94, +0.42] | -7.8 | -3.09 | 44% |
| Q10 2.53–12.1 | 31 | 30 | 15 | 68% | **-0.191** | [-0.84, +0.40] | -5.9 | -3.11 | 44% |

bucket-order Spearman (bucket index vs avg) = -0.42 · trade-level Spearman(R, P&L) = -0.077

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block HIGH turnover, R > 0.372 (i.e. 24h volume = 37.2 % of mcap); Δ(high − low) = -0.343 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.568**.

**Expectancy bar on the blocked side**: blocked side = R > 0.372 → N 213, WR 69% vs kept breakeven WR 73%, avg -0.155 [-0.40, +0.08], P(avg<0) 0.899, days 144, top day 4% (2026-09-30), top pair 15% (ARKUSDT); kept N 91 avg +0.188

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): FAIL · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar NOT passed.** In-sample Δ from blocking = +0.109 %/fill of the whole cohort → after 30–50 % haircut +0.054…+0.076.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 1.25 (in-sample Δ -0.295) | 45 · 71% · -0.110 | 90 · -0.048 | holds |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block high R > 0.353 (in-sample Δ -0.558) | 125 · 70% · -0.054 | 44 · +0.001 | holds |
| leave-one-month-out (10 months) | per-month re-picked cut | 154 · 74% · +0.006 [-0.28, +0.28] | 150 · -0.113 | FAILS |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| vol24 | +0.11 | 304 |
| btc_1d_ret | -0.10 | 304 |
| btc_gap | -0.06 | 304 |
| lmcap | -0.75 | 295 |
| lvol | +0.11 | 304 |


#### FRENZY_WILLY entry A — first flag (+1 net / −3 / 60 min, real ticks) — R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap

Coverage: 304/1781 fills scored (17%). Unscored fills: N 1477, avg -0.004 (NOT treated as 'rest'). Scored cohort: all scored | 304 | 181 | 68 | 72% | **-0.052** | [-0.26, +0.14] | -15.9 | -3.45 | 24% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 34 | 34 | 79% | +0.298 | 13 · +0.657 | 21 · +0.076 |
| 2026-02 | 35 | 35 | 57% | -0.556 | 17 · -1.567 | 18 · +0.400 |
| 2026-03 | 32 | 32 | 62% | -0.294 | 16 · -0.194 | 16 · -0.395 |
| 2026-04 | 38 | 38 | 82% | +0.280 | 17 · +0.391 | 21 · +0.190 |
| 2026-05 | 30 | 30 | 77% | +0.047 | 18 · +0.276 | 12 · -0.296 |
| 2026-06 | 24 | 24 | 79% | +0.155 | 11 · -0.105 | 13 · +0.376 |
| 2026-07 | 20 | 20 | 75% | +0.092 | 7 · +0.420 | 13 · -0.085 |
| 2026-08 | 33 | 33 | 70% | -0.231 | 15 · +0.188 | 18 · -0.580 |
| 2026-09 | 52 | 52 | 67% | -0.176 | 34 · +0.017 | 18 · -0.539 |
| 2026-10 | 6 | 6 | 83% | +0.319 | 4 · +1.000 | 2 · -1.042 |

(pooled median R = 0.295)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.00601–0.101 | 61 | 52 | 25 | 82% | **+0.324** | [-0.09, +0.68] | +19.8 | -3.08 | 58% |
| Q2 0.104–0.222 | 61 | 56 | 38 | 75% | **+0.096** | [-0.35, +0.49] | +5.8 | -3.26 | 26% |
| Q3 0.226–0.381 | 60 | 53 | 30 | 68% | **-0.271** | [-0.76, +0.20] | -16.3 | -3.45 | 35% |
| Q4 0.382–0.798 | 61 | 48 | 38 | 72% | **+0.031** | [-0.42, +0.44] | +1.9 | -3.11 | 20% |
| Q5 0.83–6.01 | 61 | 58 | 28 | 61% | **-0.445** | [-0.91, +0.01] | -27.1 | -3.15 | 27% |

bucket-order Spearman (bucket index vs avg) = -0.90 · trade-level Spearman(R, P&L) = -0.139

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.00601–0.0607 | 31 | 29 | 11 | 84% | **+0.318** | [-0.26, +0.80] | +9.9 | -3.08 | 71% |
| Q2 0.0619–0.101 | 30 | 27 | 22 | 80% | **+0.330** | [-0.22, +0.82] | +9.9 | -3.06 | 67% |
| Q3 0.104–0.148 | 30 | 30 | 25 | 67% | **-0.154** | [-0.80, +0.45] | -4.6 | -3.09 | 39% |
| Q4 0.15–0.222 | 31 | 28 | 22 | 84% | **+0.337** | [-0.24, +0.86] | +10.4 | -3.26 | 44% |
| Q5 0.226–0.291 | 30 | 29 | 20 | 53% | **-0.867** | [-1.58, -0.15] | -26.0 | -3.45 | 32% |
| Q6 0.299–0.381 | 30 | 28 | 18 | 83% | **+0.324** | [-0.26, +0.86] | +9.7 | -3.06 | 57% |
| Q7 0.382–0.573 | 31 | 26 | 26 | 71% | **-0.023** | [-0.72, +0.55] | -0.7 | -3.11 | 28% |
| Q8 0.575–0.798 | 30 | 28 | 24 | 73% | **+0.087** | [-0.48, +0.60] | +2.6 | -3.09 | 41% |
| Q9 0.83–1.34 | 30 | 28 | 20 | 67% | **-0.210** | [-0.91, +0.45] | -6.3 | -3.15 | 34% |
| Q10 1.36–6.01 | 31 | 31 | 18 | 55% | **-0.672** | [-1.39, -0.02] | -20.8 | -3.09 | 37% |

bucket-order Spearman (bucket index vs avg) = -0.52 · trade-level Spearman(R, P&L) = -0.139

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block HIGH turnover, R > 1 (i.e. 24h volume = 100.0 % of mcap); Δ(high − low) = -0.660 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.041**.

**Expectancy bar on the blocked side**: blocked side = R > 1 → N 46, WR 57% vs kept breakeven WR 73%, avg -0.612 [-1.14, -0.09], P(avg<0) 0.987, days 46, top day 6% (2026-07-06), top pair 16% (ARPAUSDT); kept N 258 avg +0.047

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): PASS · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar PASSED.** In-sample Δ from blocking = +0.093 %/fill of the whole cohort → after 30–50 % haircut +0.046…+0.065.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block low R ≤ 0.148 (in-sample Δ +0.185) | 48 · 85% · +0.475 | 87 · -0.368 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block high R > 0.146 (in-sample Δ -0.816) | 127 · 73% · +0.015 | 42 · -0.205 | FAILS |
| leave-one-month-out (10 months) | per-month re-picked cut | 100 · 66% · -0.248 [-0.61, +0.08] | 204 · +0.043 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| vol24 | +0.20 | 304 |
| btc_1d_ret | -0.10 | 304 |
| btc_gap | -0.02 | 304 |
| lmcap | -0.47 | 295 |
| lvol | +0.20 | 304 |


#### FRENZY_WILLY entry B — fresh ON bars not taken by today's FRENZY / WIDE

**Supply-drift check (bot-source mcap vs CoinGecko history)**

| month | N | median mcap_b / mcap_CG | IQR | share within ±25 % |
|---|---|---|---|---|
| 2026-01 | 40 | 1.783 | 1.40–2.09 | 18% |
| 2026-02 | 5 | 1.212 | 1.11–1.26 | 60% |
| 2026-03 | 21 | 1.170 | 1.08–1.24 | 67% |
| 2026-04 | 16 | 1.407 | 1.18–1.51 | 31% |
| 2026-05 | 18 | 1.172 | 0.99–1.47 | 50% |
| 2026-06 | 9 | 1.421 | 1.32–1.56 | 22% |
| 2026-07 | 17 | 1.837 | 1.20–2.68 | 29% |
| 2026-08 | 25 | 1.340 | 1.21–2.02 | 36% |
| 2026-09 | 15 | 1.254 | 1.17–1.42 | 47% |

Spearman(log R, log R_fut[CG mcap]) = +0.924 on 166 fills.


#### FRENZY_WILLY entry B — fresh ON bars not taken by today's FRENZY / WIDE — R — PRIMARY: Binance futures 24h quote vol / bot-source mcap (today's CMC supply × entry price)

Coverage: 903/1038 fills scored (87%). Unscored fills: N 135, avg -0.151 (NOT treated as 'rest'). Scored cohort: all scored | 903 | 243 | 215 | 70% | **-0.188** | [-0.32, -0.07] | -169.9 | -3.16 | 7% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 103 | 103 | 68% | -0.221 | 73 · -0.167 | 30 · -0.353 |
| 2026-02 | 68 | 68 | 72% | -0.111 | 37 · -0.529 | 31 · +0.388 |
| 2026-03 | 113 | 113 | 66% | -0.288 | 60 · +0.001 | 53 · -0.614 |
| 2026-04 | 126 | 126 | 72% | -0.105 | 67 · -0.257 | 59 · +0.067 |
| 2026-05 | 88 | 88 | 72% | -0.161 | 43 · -0.148 | 45 · -0.174 |
| 2026-06 | 71 | 71 | 70% | -0.173 | 27 · -0.428 | 44 · -0.016 |
| 2026-07 | 108 | 108 | 61% | -0.541 | 44 · -0.433 | 64 · -0.615 |
| 2026-08 | 122 | 122 | 72% | -0.137 | 49 · -0.244 | 73 · -0.066 |
| 2026-09 | 104 | 104 | 77% | +0.075 | 52 · -0.382 | 52 · +0.532 |

(pooled median R = 2.12)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0973–0.917 | 181 | 120 | 104 | 70% | **-0.185** | [-0.50, +0.12] | -33.5 | -3.15 | 10% |
| Q2 0.925–1.68 | 180 | 120 | 104 | 67% | **-0.282** | [-0.55, -0.02] | -50.8 | -3.13 | 13% |
| Q3 1.68–2.94 | 181 | 126 | 101 | 66% | **-0.348** | [-0.62, -0.07] | -63.0 | -3.15 | 11% |
| Q4 2.95–5.96 | 180 | 119 | 98 | 77% | **+0.080** | [-0.18, +0.34] | +14.3 | -3.13 | 14% |
| Q5 5.96–57.8 | 181 | 103 | 71 | 70% | **-0.204** | [-0.47, +0.05] | -37.0 | -3.16 | 13% |

bucket-order Spearman (bucket index vs avg) = +0.10 · trade-level Spearman(R, P&L) = +0.036

**Deciles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0973–0.605 | 91 | 72 | 58 | 67% | **-0.259** | [-0.64, +0.10] | -23.6 | -3.15 | 15% |
| Q2 0.617–0.917 | 90 | 75 | 67 | 72% | **-0.110** | [-0.52, +0.28] | -9.9 | -3.10 | 19% |
| Q3 0.925–1.27 | 90 | 73 | 68 | 74% | **+0.024** | [-0.32, +0.36] | +2.1 | -3.13 | 22% |
| Q4 1.28–1.68 | 90 | 72 | 66 | 60% | **-0.588** | [-0.99, -0.19] | -52.9 | -3.12 | 14% |
| Q5 1.68–2.12 | 91 | 77 | 64 | 66% | **-0.368** | [-0.79, +0.02] | -33.5 | -3.15 | 19% |
| Q6 2.12–2.94 | 90 | 74 | 62 | 67% | **-0.328** | [-0.71, +0.04] | -29.5 | -3.14 | 13% |
| Q7 2.95–4.07 | 90 | 70 | 62 | 74% | **+0.017** | [-0.33, +0.35] | +1.6 | -3.09 | 19% |
| Q8 4.08–5.96 | 90 | 71 | 62 | 79% | **+0.142** | [-0.23, +0.48] | +12.7 | -3.13 | 22% |
| Q9 5.96–9.25 | 90 | 68 | 58 | 71% | **-0.159** | [-0.50, +0.17] | -14.3 | -3.12 | 18% |
| Q10 9.4–57.8 | 91 | 59 | 38 | 69% | **-0.249** | [-0.62, +0.09] | -22.7 | -3.16 | 20% |

bucket-order Spearman (bucket index vs avg) = +0.18 · trade-level Spearman(R, P&L) = +0.036

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block LOW turnover, R ≤ 2.94 (i.e. 24h volume = 294.5 % of mcap); Δ(high − low) = +0.209 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.716**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 2.94 → N 542, WR 68% vs kept breakeven WR 75%, avg -0.272 [-0.44, -0.11], P(avg<0) 0.999, days 214, top day 3% (2026-09-13), top pair 4% (ARKUSDT); kept N 361 avg -0.063

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): PASS · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar PASSED.** In-sample Δ from blocking = +0.163 %/fill of the whole cohort → after 30–50 % haircut +0.082…+0.114.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block low R ≤ 0.537 (in-sample Δ +0.198) | 27 · 74% · +0.027 | 378 · -0.212 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | block low R ≤ 2.33 (in-sample Δ +0.328) | 296 · 69% · -0.201 | 202 · -0.152 | holds |
| leave-one-month-out (9 months) | per-month re-picked cut | 498 · 68% · -0.256 [-0.42, -0.09] | 405 · -0.105 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.17 | 903 |
| U2 | -0.02 | 903 |
| gvol | +0.01 | 903 |
| vol24 | +0.60 | 903 |
| bar_ret | +0.05 | 903 |
| btc_1d_ret | +0.04 | 903 |
| btc_gap | +0.03 | 903 |
| lmcap | -0.51 | 903 |
| lvol | +0.60 | 903 |


#### FRENZY_WILLY entry B — fresh ON bars not taken by today's FRENZY / WIDE — R_fut — robustness: Binance futures 24h vol / CoinGecko daily mcap (history, no supply assumption)

Coverage: 166/1038 fills scored (16%). Unscored fills: N 872, avg -0.185 (NOT treated as 'rest'). Scored cohort: all scored | 166 | 95 | 39 | 69% | **-0.173** | [-0.46, +0.12] | -28.7 | -3.15 | 31% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 40 | 40 | 70% | -0.048 | 24 · -0.069 | 16 · -0.017 |
| 2026-02 | 5 | 5 | 60% | -0.660 | 2 · -3.151 | 3 · +1.000 |
| 2026-03 | 21 | 21 | 52% | -0.747 | 12 · -0.029 | 9 · -1.705 |
| 2026-04 | 16 | 16 | 81% | +0.239 | 10 · -0.218 | 6 · +1.000 |
| 2026-05 | 18 | 18 | 78% | +0.099 | 11 · -0.106 | 7 · +0.421 |
| 2026-06 | 9 | 9 | 78% | +0.098 | 0 · +nan | 9 · +0.098 |
| 2026-07 | 17 | 17 | 76% | +0.158 | 5 · +0.585 | 12 · -0.020 |
| 2026-08 | 25 | 25 | 60% | -0.626 | 12 · -0.690 | 13 · -0.567 |
| 2026-09 | 15 | 15 | 73% | -0.087 | 7 · -1.328 | 8 · +1.000 |

(pooled median R = 2.5)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.146–1.15 | 34 | 27 | 18 | 71% | **-0.072** | [-0.59, +0.50] | -2.5 | -3.11 | 43% |
| Q2 1.23–2.06 | 33 | 28 | 14 | 58% | **-0.519** | [-1.14, +0.12] | -17.1 | -3.15 | 44% |
| Q3 2.08–3.33 | 33 | 28 | 18 | 64% | **-0.430** | [-1.10, +0.23] | -14.2 | -3.10 | 30% |
| Q4 3.4–6.94 | 33 | 26 | 18 | 79% | **+0.138** | [-0.55, +0.84] | +4.6 | -3.09 | 59% |
| Q5 6.98–36.9 | 33 | 27 | 13 | 76% | **+0.014** | [-0.49, +0.51] | +0.5 | -3.13 | 65% |

bucket-order Spearman (bucket index vs avg) = +0.60 · trade-level Spearman(R, P&L) = +0.067

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block LOW turnover, R ≤ 3.33 (i.e. 24h volume = 333.4 % of mcap); Δ(high − low) = +0.414 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.532**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 3.33 → N 100, WR 64% vs kept breakeven WR 75%, avg -0.337 [-0.71, +0.05], P(avg<0) 0.954, days 69, top day 11% (2026-09-13), top pair 19% (ARKUSDT); kept N 66 avg +0.076

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): PASS · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar PASSED.** In-sample Δ from blocking = +0.203 %/fill of the whole cohort → after 30–50 % haircut +0.102…+0.142.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block low R ≤ 2.12 (in-sample Δ +0.259) | 22 · 59% · -0.572 | 44 · -0.018 | holds |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 98 · 68% · -0.174 [-0.55, +0.22] | 68 · -0.172 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | +0.24 | 166 |
| U2 | -0.06 | 166 |
| gvol | +0.04 | 166 |
| vol24 | +0.48 | 166 |
| bar_ret | +0.02 | 166 |
| btc_1d_ret | -0.05 | 166 |
| btc_gap | +0.14 | 166 |
| lmcap | -0.48 | 166 |
| lvol | +0.48 | 166 |


#### FRENZY_WILLY entry B — fresh ON bars not taken by today's FRENZY / WIDE — R_cg — robustness: CoinGecko all-exchange 24h vol / CoinGecko mcap

Coverage: 166/1038 fills scored (16%). Unscored fills: N 872, avg -0.185 (NOT treated as 'rest'). Scored cohort: all scored | 166 | 95 | 39 | 69% | **-0.173** | [-0.49, +0.12] | -28.7 | -3.15 | 31% |

| month | N | scored | WR | avg % | avg R ≤ pooled median | avg R > median |
|---|---|---|---|---|---|---|
| 2026-01 | 40 | 40 | 70% | -0.048 | 14 · +0.140 | 26 · -0.150 |
| 2026-02 | 5 | 5 | 60% | -0.660 | 3 · -0.385 | 2 · -1.074 |
| 2026-03 | 21 | 21 | 52% | -0.747 | 9 · -0.352 | 12 · -1.044 |
| 2026-04 | 16 | 16 | 81% | +0.239 | 11 · +0.262 | 5 · +0.189 |
| 2026-05 | 18 | 18 | 78% | +0.099 | 9 · +0.098 | 9 · +0.099 |
| 2026-06 | 9 | 9 | 78% | +0.098 | 2 · -1.036 | 7 · +0.421 |
| 2026-07 | 17 | 17 | 76% | +0.158 | 10 · +0.380 | 7 · -0.158 |
| 2026-08 | 25 | 25 | 60% | -0.626 | 15 · -0.622 | 10 · -0.631 |
| 2026-09 | 15 | 15 | 73% | -0.087 | 11 · -0.482 | 4 · +1.000 |

(pooled median R = 0.598)

**Quintiles**

| bucket | N | days | pairs | WR | avg % | day-CI 95% | sum % | worst | top-2-pair share of net pair loss |
|---|---|---|---|---|---|---|---|---|---|
| Q1 0.0253–0.231 | 34 | 28 | 18 | 62% | **-0.449** | [-1.05, +0.20] | -15.3 | -3.13 | 41% |
| Q2 0.234–0.501 | 33 | 23 | 17 | 73% | **-0.032** | [-0.82, +0.69] | -1.1 | -3.15 | 65% |
| Q3 0.501–0.711 | 33 | 19 | 16 | 58% | **-0.608** | [-1.24, +0.13] | -20.1 | -3.11 | 32% |
| Q4 0.757–2.48 | 33 | 22 | 14 | 73% | **-0.029** | [-0.64, +0.50] | -1.0 | -3.15 | 40% |
| Q5 2.48–13.8 | 33 | 21 | 15 | 82% | **+0.262** | [-0.27, +0.77] | +8.7 | -3.07 | 62% |

bucket-order Spearman (bucket index vs avg) = +0.70 · trade-level Spearman(R, P&L) = +0.093

**Threshold scan** (17 cuts at the 10–90 % quantiles, each side ≥ 40): best = block LOW turnover, R ≤ 1.39 (i.e. 24h volume = 138.9 % of mcap); Δ(high − low) = +0.709 %/fill. Shuffled-label null (1000×, same scan, max |Δ|): **p = 0.088**.

**Expectancy bar on the blocked side**: blocked side = R ≤ 1.39 → N 116, WR 64% vs kept breakeven WR 73%, avg -0.387 [-0.75, -0.03], P(avg<0) 0.982, days 73, top day 9% (2026-09-13), top pair 16% (ARKUSDT); kept N 50 avg +0.323

WR < kept breakeven WR: PASS · P(avg<0) ≥ 95% (day-clustered): PASS · ≥ 8 distinct days: PASS · no day ≥ 50% of loss: PASS · no pair ≥ 50% of loss: PASS · N ≥ 15: PASS

**Bar PASSED.** In-sample Δ from blocking = +0.270 %/fill of the whole cohort → after 30–50 % haircut +0.135…+0.189.

**Out-of-sample**

| split | rule chosen on the pick half | held-out BLOCKED N · WR · avg | held-out KEPT N · avg | verdict |
|---|---|---|---|---|
| pick < 2026-06-01 → test ≥ 2026-06-01 | block high R > 0.592 (in-sample Δ -0.340) | 30 · 73% · -0.084 | 36 · -0.301 | FAILS |
| pick ≥ 2026-06-01 → test < 2026-06-01 | too few | | | |
| leave-one-month-out (9 months) | per-month re-picked cut | 104 · 64% · -0.350 [-0.75, +0.03] | 62 · +0.125 | holds |

**Overlap with stamped / filtered variables**

| variable | Spearman with log R | N |
|---|---|---|
| atr | -0.10 | 166 |
| U2 | +0.09 | 166 |
| gvol | +0.12 | 166 |
| vol24 | +0.45 | 166 |
| bar_ret | -0.01 | 166 |
| btc_1d_ret | +0.01 | 166 |
| btc_gap | -0.02 | 166 |
| lmcap | -0.22 | 166 |
| lvol | +0.45 | 166 |


# Appendix B — robustness add-ons (study_vol_mcap_extra.py)

### Robustness add-ons

### frenzy_today (scored 383)

- primary R, within-day shuffled null: best cut block high R > 1.18, Δ -0.451, **p = 0.864**
- market cap alone: quintile avgs [0.016, 0.299, 0.16, -0.255, 0.487] · best cut block low lmcap ≤ 2.47e+07 Δ +0.532, shuffled p = 0.694
- 24h futures volume alone: quintile avgs [0.635, -0.496, -0.379, 0.457, 0.489] · best cut block low lvol ≤ 3.07e+08 Δ +0.695, shuffled p = 0.445
- CG implied supply today / at fill (covered 73): median 1.002, IQR 1.000–1.062, share > 1.25: 7% · by month median: 2026-01 1.04, 2026-02 1.05, 2026-03 1.00, 2026-04 1.08, 2026-05 1.00, 2026-06 1.18, 2026-07 1.02, 2026-08 1.00, 2026-09 1.00
- corrected vs uncorrected log R Spearman +0.959; on the covered subset: uncorrected Spearman(R, P&L) -0.059, corrected -0.050

### frenzy_today_bear (scored 284)

- primary R, within-day shuffled null: best cut block low R ≤ 2.93, Δ +0.357, **p = 0.975**
- market cap alone: quintile avgs [0.162, -0.045, 0.098, -0.361, 0.693] · best cut block low lmcap ≤ 1.79e+08 Δ +0.807, shuffled p = 0.506
- 24h futures volume alone: quintile avgs [0.468, -0.576, -0.642, 0.695, 0.589] · best cut block low lvol ≤ 1.29e+08 Δ +0.889, shuffled p = 0.286
- CG implied supply today / at fill (covered 55): median 1.017, IQR 1.000–1.073, share > 1.25: 4% · by month median: 2026-01 1.04, 2026-02 1.06, 2026-03 1.00, 2026-04 1.08, 2026-05 1.00, 2026-06 1.18, 2026-07 1.02, 2026-08 1.00, 2026-09 1.00
- corrected vs uncorrected log R Spearman +0.968; on the covered subset: uncorrected Spearman(R, P&L) +0.059, corrected +0.064

### frenzy_gated (scored 522)

- primary R, within-day shuffled null: best cut block high R > 1.37, Δ -0.241, **p = 0.976**
- market cap alone: quintile avgs [-0.303, 0.19, 0.353, -0.563, 0.16] · best cut block low lmcap ≤ 2.13e+07 Δ +0.395, shuffled p = 0.823
- 24h futures volume alone: quintile avgs [0.319, -0.388, -0.565, 0.364, 0.102] · best cut block high lvol > 3.16e+07 Δ -0.622, shuffled p = 0.428
- CG implied supply today / at fill (covered 98): median 1.001, IQR 1.000–1.049, share > 1.25: 6% · by month median: 2026-01 1.04, 2026-02 1.04, 2026-03 1.00, 2026-04 1.07, 2026-05 1.00, 2026-06 1.18, 2026-07 1.02, 2026-08 1.01, 2026-09 1.00
- corrected vs uncorrected log R Spearman +0.968; on the covered subset: uncorrected Spearman(R, P&L) -0.006, corrected +0.010

### willyA (scored 1609)

- primary R, within-day shuffled null: best cut block high R > 2.37, Δ -0.327, **p = 0.031**
- market cap alone: quintile avgs [-0.142, -0.005, 0.115, -0.057, 0.074] · best cut block low lmcap ≤ 1.74e+07 Δ +0.293, shuffled p = 0.083
- 24h futures volume alone: quintile avgs [0.106, 0.127, -0.055, -0.086, -0.108] · best cut block high lvol > 2.11e+07 Δ -0.200, shuffled p = 0.402
- CG implied supply today / at fill (covered 311): median 1.001, IQR 1.000–1.044, share > 1.25: 5% · by month median: 2026-01 1.04, 2026-02 1.00, 2026-03 1.00, 2026-04 1.05, 2026-05 1.00, 2026-06 1.00, 2026-07 1.00, 2026-08 1.00, 2026-09 1.00, 2026-10 1.00
- corrected vs uncorrected log R Spearman +0.993; on the covered subset: uncorrected Spearman(R, P&L) -0.122, corrected -0.125

### willyB (scored 903)

- primary R, within-day shuffled null: best cut block low R ≤ 2.94, Δ +0.209, **p = 0.670**
- market cap alone: quintile avgs [-0.168, -0.191, -0.287, -0.238, -0.057] · best cut block high lmcap > 1.59e+07 Δ -0.180, shuffled p = 0.823
- 24h futures volume alone: quintile avgs [-0.132, -0.434, -0.234, 0.052, -0.192] · best cut block high lvol > 3.46e+07 Δ -0.351, shuffled p = 0.206
- CG implied supply today / at fill (covered 173): median 1.001, IQR 1.000–1.047, share > 1.25: 7% · by month median: 2026-01 1.04, 2026-02 1.00, 2026-03 1.00, 2026-04 1.05, 2026-05 1.00, 2026-06 1.00, 2026-07 1.00, 2026-08 1.00, 2026-09 1.00
- corrected vs uncorrected log R Spearman +0.985; on the covered subset: uncorrected Spearman(R, P&L) +0.020, corrected +0.028

