# LITE: is there a market regime that decides whether it works? (2026-10-15)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing committed, no bot API call.
- Pre-registered in `reports/LITE_REGIME_PREREG.txt` (frozen before any regime × outcome table). The operator's "2D or 3D based on BTC" grid section was appended **before** any grid cell was priced. One feature-QA deviation (alt universe fix, below) was also appended before outcomes were joined.
- Binance was used only for the gap the cache did not cover (2026-10-11 → today: daily + 5-minute candles, 498 calls, peak used-weight under 700/min, no 418/429). Stored in `reports/lite_regime/fetch/`, outside `backtest_cache`.
- Not reviewed yet, so it supports no arm and no ship.
- Outcome = today's LITE exit (+3 resting TP / −3 stop / 12 h), `rm_base_pct`.

## Plain-English summary

**The question: what market condition makes LITE win or lose, and can we see it in time?**

**Short answer: we could not find one that we can see in time.**
- **LITE's good and bad stretches do not last.**
  - A good LITE week says nothing about the next week. Week-to-week correlation is +0.07 in 2025 and −0.04 in 2026, which is chance level.
  - Months differ by no more than random noise would make them differ (shuffle p 0.24 and 0.48).
  - A slow regime would make good weeks follow good weeks. That does not happen inside either year.
- **The one big difference is the YEAR, and it looks the same in every market state.**
  - 2025 lost (−0.165 %/fill, 828 fills). 2026 won (+0.258 %/fill, 624 fills).
  - Split the days into BTC states (up or down, above or below the 50-day average, calm or volatile, near or far from the 30-day high, and so on). 2026 beat 2025 in **28 of 31** of those states, by +0.38 %/fill on average.
  - So the BTC state does not cause the year gap. Something we do not measure changed between the years. With only two years, that is **two observations**, so it cannot be tested.
  - **Part of the gap is how the backtest was priced, not the market.**
    - 2025's non-top-50 fills were priced on 1-minute candles (−0.225 %/fill). Its tick-priced fills averaged −0.017.
    - Using tick-priced fills only, the gap shrinks to −0.04 vs +0.26 %/fill.
- **None of the 31 slow variables beats chance.** They cover:
  - BTC trend, moving averages, volatility and distance from the 30-day high;
  - alt breadth, alt-vs-BTC strength, market volume and dispersion;
  - FRENZY spike counts and recent spike follow-through;
  - LITE's and WILLY's own recent results.
  - Tested in day and week units against a 7-day-block-shuffle null, the best family-wise p is 0.17 in 2025 and 0.20 in 2026.
  - Learn on one year, apply to the other: 25 of 62 variables agree in sign both ways. Chance gives 28.6.
- **The BTC regime map (2D / 3D grids) finds nothing.**
  - No grid separates LITE results better than shuffled labels (p 0.15–0.93).
  - 3 of 41 cells confirm across years (8 of 41 using tick-priced fills only).
  - Every "trade only in good BTC cells" rule and every "skip high-ATR coins in bad cells" rule **loses money in 2026**. The one exception is the alt-vs-BTC grid, which is the same rule as alt_rs7 below (+13 %, not significant).
- **"Pause LITE after a losing streak" fails, and is harmful.**
  - It would have helped in losing 2025 (+87 to +148 % total).
  - It **costs 54 to 122 % total in winning 2026** (P ≤ 0.13).
  - That is what you expect when streaks do not persist: the pause just skips the next, random, batch of trades.
- **High-ATR coins do not react more to the regime.**
  - Median split effect by ATR tercile: low 0.32, mid 0.27, high 0.23.
  - The 2025 vs 2026 high-ATR gap (−0.45 vs +0.47) is the same year effect.
- **The one variable that technically passed the pre-registered screen fails the gate.**
  - The variable is **alt_rs7 > 0**: alts beat BTC over the prior 7 days.
  - LITE is worse after such weeks in both years: −0.45 and −0.48 %/fill.
  - But it is not family-wise significant, and it fails H1 → H2.
  - As a rule, "skip LITE on those days":
    - 2025: +139 % total (P 0.99);
    - 2026: only +13 % (P 0.65), and it turns negative when one month is dropped (LOMO).
  - **No Scout line.** Its cut is recorded here as frozen (> 0), so it cannot be re-fit later.
- **The washed-out band (BTC 15–24 % below its 30-day high) is not a LITE win zone.**
  - Inside the band: 2025 −0.02 / 2026 +0.11.
  - Outside it: −0.19 / +0.28.

**Recommendation (operator decides): no regime switch, no equity-curve pause, nothing new on LITE.**
- LITE_BTC1H_FALLING keeps observing as before.
- The honest reading: LITE's edge is small, and its good and bad periods look like noise around a level. Nothing measurable at day or week start predicts that level.
- What would help next:
  1. The full-year tick re-pricing of the 2025 non-top-50 fills, to remove the pricing gap.
  2. Data we do not have: open interest, liquidations, order book, BTC dominance by market cap, and 2025 funding. See "Could NOT test".

## BTC regime map (2D / 3D) for LITE

**How to read it.**
- Cuts are frozen from the other year (sign, or that year's median).
- A cell is only read if it has ≥ 8 distinct weeks; fewer is marked "thin".
- eta² = share of fill variance explained by the cells. p = 7-day-block label-shuffle null, R = 500.
- Full cell tables, including high-ATR columns: `reports/lite_regime/tables.md` (all fills) and `tables_tick.md` (tick-priced only).

**Grid-level results:**

| grid | 2025 eta² (p) | 2026 eta² (p) | cells confirmed 25→26 | "trade only good cells" Δ total 2026 (cross-year / monthly WF) | "skip high-ATR in bad cells" Δ total 2026 |
|---|---|---|---|---|---|
| G1 r7 × vs 50d MA × vol7 (8) | 0.014 (0.21) | 0.008 (0.70) | — | −99 / −162 | −123 / −75 |
| G2 r30 × off-30d-high band × r3 (12) | 0.009 (0.66) | 0.014 (0.52) | — | −25 / −28 | −36 / −24 |
| G3 1d × 7d × 30d signs (8) | 0.004 (0.93) | 0.009 (0.68) | — | −134 / −116 | −105 / −93 |
| G3c aligned up / mixed / aligned down | 0.001 (0.79) | 0.002 (0.62) | — | −161 / −137 | −115 / −100 |
| G4 vs 200d MA × breadth20 | 0.006 (0.29) | 0.001 (0.85) | — | −164 / −103 | −115 / −30 |
| G5 r7 × spike follow-through 12 h | 0.004 (0.36) | 0.003 (0.60) | — | −92 / −89 | −115 / −97 |
| G6 vol7 × alt-vs-BTC 7d | 0.007 (0.15) | 0.006 (0.36) | — | +13 / +1 (P 0.66 / 0.51) | −115 / −130 |

- 3 of 41 cells confirm across all grids.
- Every grid policy fails the gate.
- Why most "trade-only" rules lose: trained on losing 2025, almost every cell looks bad, so the rule skips most of winning 2026. That is the uniform year shift again.

**The cells LITE trades most often** (all fills; WR · avg %/fill):

| cell (grid) | 2025 N · weeks · WR · avg | 2026 N · weeks · WR · avg |
|---|---|---|
| r7+ · above 50d MA · vol high (G1) | 135 · 18 · 53 · +0.19 | 137 · 16 · 53 · +0.19 |
| r7+ · above 50d MA · vol low (G1) | 231 · 23 · 48 · −0.09 | 106 · 14 · 59 · +0.53 |
| r7− · below 50d MA · vol high (G1) | 216 · 23 · 44 · −0.38 | 110 · 14 · 55 · +0.27 |
| BTC 1d/7d/30d aligned up (G3c) | 253 · 27 · 49 · −0.04 | 197 · 24 · 56 · +0.36 |
| mixed (G3c) | 439 · 50 · 46 · −0.22 | 350 · 38 · 55 · +0.27 |
| aligned down (G3c) | 136 · 23 · 46 · −0.23 | 77 · 20 · 49 · −0.05 |
| r30+ · near 30d high · r3+ (G2) | 275 · 25 · 51 · +0.11 | 235 · 24 · 55 · +0.27 |
| r30− · deep (≤ −15 %) (G2, both r3 cells pooled; weeks summed) | 122 · 13 · 49 · −0.11 | 129 · 18 · 52 · +0.12 |
| vol high · alts beat BTC (G6) | 136 · 20 · 39 · −0.68 | 100 · 17 · 48 · −0.15 |
| vol low · alts lag BTC (G6) | 245 · 33 · 49 · +0.01 | 209 · 28 · 59 · +0.51 |

**H1 → H2 check.** Among cells with ≥ 8 weeks in both halves, 8 of 13 agree in sign. That is a coin flip.

## Tables

### Baseline (today's exit)

| block | N | days | weeks | WR % | avg %/fill | high-ATR N · avg |
|---|---|---|---|---|---|---|
| BT 2025 (all) | 828 | 279 | 51 | 47.1 | −0.165 | 276 · −0.451 |
| BT 2025 tick-priced only | 264 | 166 | 49 | 49.2 | −0.039 | 88 · −0.342 |
| BT 2026 | 624 | 238 | 40 | 54.5 | +0.258 | 244 · +0.470 |
| BT 2026-H1 | 369 | 143 | 25 | 55.0 | +0.297 | 139 · +0.369 |
| BT 2026-H2 | 255 | 95 | 15 | 53.7 | +0.201 | 105 · +0.602 |
| MASTER B18–B21 | 20 | 8 | 2 | 55.0 | +0.297 | 8 · +1.504 |
| LIVE (DEEP, DUSK, EPIC, ZK) | 4 | 2 | 1 | 25.0 | −1.526 | 1 · −3.432 |

ATR terciles are frozen on 2025 BT: low ≤ 1.216 % < mid ≤ 1.827 % < high.

### Screen (day and week units, 7-day-block shuffle null, R = 500)

**Family-wise p** (best test in each stat family; 31 vars × 4 units {next 1d, 3d, 7d, ISO week} = 124 tests per family):

| family | 2025 (all fills) | 2026 | 2025 (tick-priced) |
|---|---|---|---|
| Spearman | 0.17 | 0.51 | 0.68 |
| sign / median split | 0.84 | 0.20 | 0.32 |
| top vs bottom tercile | 0.63 | 0.23 | 0.65 |

**Tests with per-test p ≤ 0.05** (≈ 6 expected by chance per family):

| family | 2025 | 2026 |
|---|---|---|
| sign / median split | 5 | 15 |
| Spearman | 14 | 16 |
| tercile | 10 | 9 |

2026 shows a mild excess of nominally significant tests, but its family-wise p stays at or above 0.20.

**Discover → confirm** (same sign and bootstrap P ≥ 0.90):

| direction | all fills | tick-priced only |
|---|---|---|
| 2025 → 2026 | 48 / 248 | 44 / 248 |
| 2026 → 2025 | 42 / 248 | 33 / 248 |
| H1 → H2 | 31 / 248 | — |

**Sign agreement both ways** (split, 1d/3d): 25 of 62 observed, against a null mean of 28.6 (p95 40). There is no excess.

**Confirmed both ways at 1d/3d:**

| var | unit | Δ 2025 | Δ 2026 | null p 2025 · 2026 | H1→H2 |
|---|---|---|---|---|---|
| **alt_rs7** (alts beat BTC 7d) | 1d | −0.45 | −0.48 | 0.046 · 0.092 | no |
| disp7 | 1d / 3d | −0.32 / −0.37 | −0.37 / −0.42 | 0.26 · 0.22 | no |
| r1 (tercile) | 3d | +0.29 | +0.20 | 0.11 · 0.44 | no |
| spk24 | 3d | −0.39 | −0.26 | 0.52 · 0.31 | no |

- The tick-only run gives the same survivor, alt_rs7: −0.78 / −0.48, null p 0.060 / 0.092.

**Per-variable reads.** Per-fill day-start split per year (N · WR · avg on each side, plus the high-ATR Δ): `reports/lite_regime/tables.md` § "Per-variable NAT split" and `var_splits.csv`.

Notable sign flips between years, i.e. what made 2025 vs 2026 look different:

| variable | 2025 | 2026 |
|---|---|---|
| ft2 / ftr12 ("pumps have been continuing lately") | −0.31 / −0.22 | +0.34 / +0.34 |
| lite_n10 / lite_d3 (own recent results) | +0.41 / +0.33 | −0.27 / −0.35 (anti-persistent) |
| dma200 | +0.07 | −0.27 |
| spk72 / flags96 (many recent FRENZY spikes → worse), same sign both years | −0.47 / −0.39 | −0.21 / −0.36 |

- spk72 / flags96 do not confirm at P ≥ 0.90: in 2026 only 81–84 fills fall on the "quiet" side.

### Policies (skipped fills = 0; cut and side chosen on train only)

| policy | 2025 test: Δ total % · P · LOMO min | 2026 test: Δ total % · P · LOMO min | gate |
|---|---|---|---|
| alt_rs7 > 0 → skip LITE (cross-year) | +139.4 · 0.99 · +111.6 | +13.4 · 0.65 · −4.9 | **fail** (P, LOMO) |
| same, expanding monthly WF | +122.6 · 0.99 · +94.8 | +13.4 · 0.65 · −4.9 | fail |
| alt_rs7 > 0 → skip high-ATR only | +26.3 · 0.79 | +2.6 · 0.52 | fail |
| pause after last 10 < 0, resume after 5 ≥ 0 | +142.1 · 0.99 | −85.1 · 0.03 | fail |
| pause after last 10 < 0, resume after 10 ≥ 0 | +148.1 · 0.98 | −53.8 · 0.13 | fail |
| pause after last 20 < 0, resume after 5 / 10 ≥ 0 | +106.4 / +106.3 | −66.9 / −57.9 | fail |
| skip day when LITE's prior 3 d < 0 | +126.2 · 0.97 | −121.5 · 0.00 | fail |
| skip day when LITE's prior 7 d < 0 | +87.5 · 0.89 | −121.6 · 0.00 | fail |
| 14 BTC-grid policies × 2 WF modes = 28 rows (G1–G6 + G3c × trade / skip-ATR) | mixed | 26 of 28 negative (the 2 positive = G6 trade = the alt_rs7 rule, +13 / +1) | all fail |

- **Two-sided view of alt_rs7 in 2026:** skipping costs +288 % of winners and avoids +302 % of losers, on 197 fills averaging −0.07.
- Full rows (winners lost, losers avoided, weeks carrying the effect, max week share) are in `reports/lite_regime/policies.csv` and `policies_tick.csv`.
- Slot effects of skipped fills are not re-simulated.

### Persistence (post-hoc diagnostic, declared; `persist.txt`, `persist_tick.txt`)

| | 2025 | 2026 |
|---|---|---|
| weekly avg lag-1 / lag-2 / lag-4 autocorrelation | +0.07 / −0.07 / −0.00 (p 0.25 / 0.63 / 0.45) | −0.04 / −0.19 / −0.10 (p 0.53 / 0.83 / 0.67) |
| month eta² vs day-shuffle null | 0.019 vs 0.015 (p 0.24) | 0.015 vs 0.016 (p 0.48) |

- BTC-grid cells with ≥ 8 weeks in both years where 2026 beat 2025: **28 of 31**, mean +0.38 %/fill.
- Tick-only: 18 of 27, +0.21.
- **Uniform degradation** (sleeve-kill checklist ③): the cause is a variable we do not measure, or the pricing method. It is not the BTC regime.

### Monthly timeline

The month-by-month table (LITE avg vs mean BTC r30, 50d MA distance, off-30d-high, vol, breadth, alt-vs-BTC, spikes, follow-through) is in `tables.md`.

Month-level Spearman (22 months):

| variable | Spearman |
|---|---|
| r30 | +0.27 |
| dma50 | +0.25 |
| off30 | +0.19 |
| br_ma20 | +0.10 |
| ft12 | +0.08 |
| rv7 | +0.06 |
| alt_rs7 | −0.04 |
| spk72 | −0.05 |

All are weak. There are 22 months, so this is about 2 independent BTC up/down regimes.

### Master and live (refute-only)

- **All 24 fills (Oct 7–15) sit in one regime.** BTC 7d down (−0.4 to −4.6 %), above the 50d MA, r30 up, 4–7 % below the 30d high, alts lagging BTC (alt_rs7 < 0, the "good" side), spike follow-through low.
- Master is +0.30 %/fill (11 W / 9 L). Live Oct-14/15 is −1.53 (1 W / 3 L), in the same cell.
- **Nothing to refute or support.** It is one window, and the same cell carried both a win cluster and a loss cluster.

## Method notes and parity

**Features.** Script: `scripts/study_lite_regime_features.py` → `reports/lite_regime/day_features.csv`.
- Every value is measured at 00:00 UTC from closed daily bars ≤ d−1, or from spikes or fills fully resolved before d.
- Feature parity against the engine's own stamps on the same fills:

  | my feature | engine stamp | BT fills | master fills |
  |---|---|---|---|
  | r7 | `btc_ret_7d` | r = 1.00 | r = 1.00 |
  | dma20 | `btc_vs_ema20_1d` | r = 0.986 | r = 0.989 |
  | off30 | `entry_btc_off30d_high_pct` | r = 0.985 (intraday vs day-start) | — |

**FRENZY spikes.** Rebuilt market-wide from cached 5-minute candles for 668 pairs using the engine's spike rule: 8,570 spikes from 2024-06 to 2026-10-15.
- Parity: the engine's `entry_frenzy_spike_at` on BT LITE fills matches a rebuilt spike (same pair, ≤ 10 min) in 87.0 % (2025), 79.7 % (H1) and 76.9 % (H2) of fills.
- The episode rule is approximated by a 24 h no-activity gap; the VWAP-hold leg is ignored.

**Deviation 1** (appended pre-outcome). The alt universe ranked by day-before volume was pump-biased (alt_rs7 averaged +13 %). It was changed to top-50 by 30-day average volume, with a median alt return and an IQR dispersion.

**Multiplicity:**
- 372 screen tests per year (31 vars × 4 units × 3 stats);
- 744 discover/confirm tests;
- 41 grid cells + 7 grid-wise tests;
- 4 survivor-policy rows, 6 equity-curve switches, 28 grid policies.

**Validation.** `venv/bin/python scripts/validate_against_master.py` → **ALL CHECKS PASS**.

## Could NOT test

- **BTC dominance by market cap.** Not cached; alt_rs7 is the price proxy.
- **Funding in 2025.** The funding cache starts 2025-12-30. fund3 can only be read inside 2026: high funding +0.36 vs low +0.06 %/fill. No cross-year confirm is possible.
- **Open interest, liquidations, long/short ratio, order book, stablecoin flows.** Not cached.
- **Macro calendar (FOMC / CPI) and Binance listing / delisting calendar.** Not cached.
- **The year-level regime itself.** Two years = two observations (WINDOW-UNITS rule). It cannot be tested or acted on.
- **Pricing confound.** The 2025 non-top-50 fills are 1-minute-priced. Reads were repeated on tick-priced fills only (`*_tick`), and the conclusions are unchanged.
- **Portfolio and slot effects of skipping LITE fills.** Skips are priced as 0 in a fixed BT sequence.

## Exact commands

```
venv/bin/python scripts/study_lite_regime_prereg.py            # base pre-registration (refuses overwrite)
venv/bin/python scripts/study_lite_regime_prereg.py --grids    # BTC grid addendum (once)
venv/bin/python scripts/study_lite_regime_prereg.py --dev1     # deviation 1 (once)
venv/bin/python scripts/study_lite_regime_fetch.py             # gap fetch 2026-10-11 -> (budgeted, resumable)
venv/bin/python scripts/study_lite_regime_features.py --rescan # spikes + day features
venv/bin/python scripts/study_lite_regime_main.py --R 500      # screen / confirm / grids / policies -> tables.md
venv/bin/python scripts/study_lite_regime_main.py --R 500 --tick
venv/bin/python scripts/study_lite_regime_persist.py [--tick]  # post-hoc persistence diagnostic
venv/bin/python scripts/validate_against_master.py
```
