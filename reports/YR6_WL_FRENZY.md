# YR6 two-year winners vs losers: FRENZY family (LITE · WILLY · LONG · WIDE)

2026-10-13 · brief: reports/YR6_WL_STUDY_BRIEF.md (binding) · data: reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv (priced at today's exits, 443a7b5) · **seed 1** = the FRENZY analysis cohort (the FRENZY seeds follow almost the same path); seeds 2+3 and a plain-csv recount are the second computation · y = `pct_new` (% of notional, net of fees, the same at any leverage) · cache only, no network, no config, engine or git changes.
Supporting tables: `reports/YR6_WL_FRENZY_{summary,screen,regime,willy,litegate,lw}.md` and the CSVs `YR6_WL_FRENZY_{coverage,screen_tests,screen_oos,regime_tests,regime_daily_panel}.csv`.

---

## Plain-English summary

1. **No day-level on/off switch exists for any FRENZY sleeve.** I tested 70 ex-ante daily market variables per sleeve, in day units with a shuffled null. They cover BTC returns over 1/3/7/30 days, distance from the 30-day high, realised volatility, BTC versus its 20/50/200-day averages, alt breadth, "alt season" and the pump count, plus the sleeve's own trailing 7/30-day result. None of them survives in 2025 and is then confirmed in 2026-H2. The bad months in 2025 are not marked by any daily macro variable.
2. **LITE's 2025 loss comes from an entry-level separator: BTC falling intraday when LITE enters.** The stamp is `entry_btc_1h_slope ≤ 0` (BTC's 1-hour EMA20 slope). The `entry_btc_off24h_pct ≤ −1.17` stamp carries the same signal.
   - In 2025 these 250 fills lost **−0.674 %/fill** (CI [−1.07, −0.28]). The rest earned +0.088.
   - In 2026-H2 the same cohort lost −0.495 (P 0.91). It is the strongest survivor of the whole screen: day-unit p_adj 0.004, out-of-sample confirmation P 0.975.
   - **But it reverses in the 2026-H1 bear half (+0.321, 143 fills).** It is also weak inside the top-50 subset (2025 −0.336, P 0.85).
   - A pre-registered explanation (BTC weakness hurts only while BTC trades above its 50-day average) was **refuted**: the H1 cohort stays positive either way.
   - Verdict: **observe-only candidate, not a filter.**
3. **WILLY's 2025 loss (−0.238 %/fill in seed 1, −0.204 across 3 seeds) sits almost entirely in pair-days outside the top 50.**
   - Outside the top 50: 669 fills at −0.323. Inside the top 50: 230 fills at +0.011, which matches 2026's top 50 (−0.017 / +0.007).
   - Outside the top 50, 2025 stopped out 27 % of the time against 16 % in 2026. Its 90-minute-cap exits average −1.38 against −0.24 / −0.53 in 2026.
   - Four crash fills, all outside the top 50 and all exited on 1-minute bars, cost −65 percentage points of the −214 total. SAHARA lost −40 % in 46 seconds on 2025-11-29.
   - Mix (top-50 share × BTC slope) explains only 8–14 % of the gap between 2025 and 2026.
   - This is exactly the subset the brief flags for data fidelity, so the cause cannot be pinned down: a real 2025 small-cap regime and a replay artefact look the same in this data. **No change for WILLY.**
4. **LONG / WIDE:** no separator survives the null in any regime.
   - The only split that keeps its sign in all three regimes is LONG on **pair-days outside the top 50** (−0.27 / −0.35 / −0.52 against +0.15 / −0.00 / +0.02). Confirmation P is 0.93 / 0.88 / 0.88, so it misses the 0.90 bar in 2026-H2 by 0.02.
   - I found it after looking at the summary table, so it is post-hoc → observe-only.
   - At today's 2× / 1× sizing, LONG / WIDE cost −$1,290 / −$493 over two years (seed 1, fixed $2,900 book).
5. **LONG-blocks-LITE cannot be measured directly from the stamps.**
   - The replay counted at most **92 refused LITE stretches over two years, whoever held the pair.**
   - A proxy, LITE fills that entered *after* a LONG / WIDE fill in the same episode, made **+0.49 %/fill (66 fills)**, not a loss. That does not support DL 280's "blocked LITE lost 8 of 10" read on yr5.
   - The bound is wide: −$3.2k to +$11.7k against LONG / WIDE's own −$1.8k. The only decisive test is a replay with LONG / WIDE switched off.
   - Keep them ON per the operator rule (memory feedback_keep_long_wide_on); there are no grounds to change it either way.

**Recommendation: no config change.** Two observe lines with pre-committed gates are set out below: the LITE BTC-slope gate and the LONG non-top-50 split. One new replay is suggested (LONG / WIDE off) to price the LITE side effect.

---

## 1. Per-sleeve × regime table (seed 1; second computation in the right-hand columns)

Breakeven WR = |avg loss| / (avg win + |avg loss|), computed on the same fills.

| sleeve | regime | N | days | WR % | avg % | day-CI 95 % | P(avg>0) | breakeven WR % | avg seeds 2+3 | 3-seed ref (_regimes.csv) | $ seed 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LITE | 2025 BULL | 828 | 279 | 47.6 | −0.142 | [−0.347, +0.053] | 0.08 | 50.0 | −0.148 | −0.146 | −8,320 |
| LITE | 2026-H1 BEAR | 370 | 144 | 50.5 | +0.226 | [−0.066, +0.507] | 0.94 | 46.5 | +0.229 | +0.228 | +5,901 |
| LITE | 2026-H2 BULL | 256 | 95 | 49.2 | +0.137 | [−0.300, +0.568] | 0.73 | 46.8 | +0.142 | +0.140 | +2,485 |
| WILLY | 2025 BULL | 899 | 295 | 68.6 | **−0.238** | [−0.415, −0.069] | 0.004 | 73.3 | −0.188 | −0.204 | −15,097 |
| WILLY | 2026-H1 BEAR | 411 | 152 | 65.9 | +0.101 | [−0.065, +0.269] | 0.88 | 63.0 | +0.104 | +0.103 | +2,946 |
| WILLY | 2026-H2 BULL | 208 | 84 | 68.8 | +0.064 | [−0.219, +0.342] | 0.67 | 67.1 | +0.072 | +0.069 | +937 |
| LONG | 2025 BULL | 270 | 149 | 65.6 | −0.061 | [−0.322, +0.192] | 0.32 | 66.9 | −0.088 | −0.079 | −233 |
| LONG | 2026-H1 BEAR | 229 | 106 | 63.3 | −0.179 | [−0.471, +0.093] | 0.09 | 67.2 | −0.182 | −0.181 | −579 |
| LONG | 2026-H2 BULL | 124 | 67 | 62.1 | −0.273 | [−0.714, +0.138] | 0.10 | 67.9 | −0.255 | −0.261 | −479 |
| WIDE | 2025 BULL | 205 | 141 | 63.4 | −0.168 | [−0.460, +0.123] | 0.13 | 67.1 | −0.149 | −0.155 | −243 |
| WIDE | 2026-H1 BEAR | 157 | 85 | 65.0 | −0.080 | [−0.439, +0.265] | 0.33 | 66.7 | −0.080 | −0.080 | −89 |
| WIDE | 2026-H2 BULL | 81 | 51 | 60.5 | −0.282 | [−0.785, +0.188] | 0.12 | 66.7 | −0.282 | −0.282 | −161 |

Over both years the breakeven WR is about **LITE 48.7 % · WILLY 70.3 % · LONG 67.2 % · WIDE 66.9 %**. LONG and WIDE at +1.5 / −3 sit 3–4 WR points below breakeven in every regime.

**Top-50 fidelity split.** A 2025 finding only counts if it holds inside the top 50. Format: avg (N).

| sleeve | 2025 top-50 | 2025 not top-50 | H1 top-50 | H1 not | H2 top-50 | H2 not |
|---|---|---|---|---|---|---|
| LITE | −0.017 (264) | −0.201 (564) | +0.369 (148) | +0.130 (222) | +0.258 (116) | +0.037 (140) |
| WILLY | **+0.011 (230)** | **−0.323 (669)** | −0.017 (158) | +0.176 (253) | +0.007 (83) | +0.102 (125) |
| LONG | +0.154 (134) | −0.273 (136) | −0.002 (113) | −0.351 (116) | +0.022 (56) | −0.516 (68) |
| WIDE | −0.186 (99) | −0.150 (106) | +0.050 (84) | −0.230 (73) | −0.063 (49) | −0.616 (32) |

The monthly table is in YR6_WL_FRENZY_summary.md §③.

## 2. Coverage (YR6_WL_FRENZY_coverage.csv)

- 88 `entry_*` columns carry values; 79 are ≥ 80 % non-null in every sleeve × regime.
- **Below 80 %:**
  - `entry_vol_mcap_ratio`, `entry_mcap_usd`, `entry_cmc_rank`: LITE 2025 0.69, LONG 2025 0.78, WIDE 2025 0.75 (delisted coins have no mcap proxy). These were read separately (summary §⑤), scored terciles only, never zone-vs-rest.
    - The read is non-monotonic and changes sign by regime → no separator.
    - LITE 2025 fills with no mcap score (256, mostly later-delisted coins) earned +0.12 against −0.25 for scored fills.
  - `entry_frenzy_catchup_*`: ≈ 0 %.
  - `entry_frenzy_willy_wait_bars`: WILLY only.
  - `entry_frenzy_di_spread / adx_delta / vol_trend`: not WILLY.
- Every `entry_br_*` (breadth door), `entry_bear_*`, funding rate, pair rank, pair age, BTC 4h EMA50-200 gap, pair 1d −DI and the gap-5-20 stamps are **absent from the yr6 replay** → could not be tested (§7).

## 3. Exhaustive screen (YR6_WL_FRENZY_screen.md)

**How the screen worked:**
- **Pair-level family:** every pair-level column at sign, median and outer-tercile cuts, plus every 2D median-quadrant of the 40 strongest columns. That is 1,700–2,950 tests per sleeve × discovery regime, measured in fills. The null is a day-block resample with 500 draws, taking the max |t| over the whole family.
- **Macro family:** 86 tests, measured in **days** (one observation per day). The null is a day-label shuffle.
- Discovery in one bull regime, confirmation in the other (cuts frozen, one-sided day-bootstrap P ≥ 0.90). The bear half is shown separately.

| sleeve · discovery regime | family | tests | observed max abs(t) | null q95 | survivors (p_adj<0.10) | confirmed |
|---|---|---|---|---|---|---|
| LITE · 2025 | MACRO (day) | 86 | 4.10 | 3.35 | 5 | **3** (all "BTC falling intraday") |
| LITE · 2025 | PAIR | 2,868 | 3.42 | 4.37 | 0 | – |
| LITE · H2 | MACRO (day) | 86 | 3.50 | 3.60 | 1 | **1** (btc_off24h ≤ −1.22, found independently) |
| LITE · H2 | PAIR | 2,878 | 3.80 | 5.26 | 0 | – |
| WILLY · 2025 | PAIR | 2,703 | 4.81 | 4.60 | 4 | 1, **bear-reversed** |
| WILLY · H2 | PAIR | 2,942 | 8.61 | 8.29 | 1 (N 17, 100 % WR) | 0 |
| WILLY · either | MACRO (day) | 86 | 2.60 / 2.92 | 3.21 / 3.67 | 0 | – |
| LONG · either | PAIR / MACRO | 2,795–2,801 / 86 | 4.92–5.08 / 2.65–2.92 | 5.9–6.1 / 3.6 | 0 | – |
| WIDE · either | PAIR / MACRO | 1,717–2,506 / 86 | 3.17–3.95 / 1.70–3.42 | 5.6–5.8 / 3.5–4.0 | 0 | – |

Survivors and their out-of-sample confirmation:

| sleeve | discovery | zone (frozen) | zone vs rest in discovery | confirmation Δ (P) | bear H1 Δ (P same sign) | 2025 top-50 Δ (P) |
|---|---|---|---|---|---|---|
| LITE | 2025 | day: entry_btc_1h_slope > 0 | +0.22 vs −0.94 (t 4.10, p_adj 0.004) | H2 +0.99 (0.975) ✔ | +0.29 (0.75) | fill-level, see §5: −0.336 blocked (0.85) |
| LITE | 2025 | day: entry_btc_off24h_pct ≤ −1.166 | −0.77 vs +0.21 (p_adj 0.018) | H2 −1.61 (1.00) ✔ | −0.01 (0.52) | – |
| LITE | H2 | day: entry_btc_off24h_pct ≤ −1.217 | −1.04 vs +0.59 (p_adj 0.062) | 2025 −1.00 (1.00) ✔ | −0.00 (0.51) | – |
| WILLY | 2025 | entry_adx > 46.6 ∧ entry_ema_gap_5_8 ≤ 0.965 (winner zone) | +0.45 vs −0.41 (p_adj 0.030) | H2 +0.48 (0.95) ✔ | **−0.58 (0.03): reversed** | +0.45 (0.92) |

The LONG / WIDE top-8 tables show only 2D zones at p_adj ≥ 0.13, mostly involving `pattern_*` flags with N 16–49. The `top50f` flag ranked p_adj 1.0 in both LONG discovery screens.

## 4. Regime-switch read (question 1) — YR6_WL_FRENZY_regime.md

- **Ex-ante daily panel** (prior-day closes, 70 tests per sleeve × discovery regime, measured in days, max-t null over 1,000 shuffles):
  - LITE, LONG, WIDE: **0 survivors** in either discovery regime.
  - WILLY: 0 in 2025. In H2 there is 1 (`btc_ret30d > 0` → worse, p_adj 0.055), but it reverses in 2025 (P 0.014) → refuted.
  - Uniform-degradation check: the bad 2025 months are not explained by BTC 30-day return, distance from the 30-day high, realised volatility, breadth, alt season or pump count. Monthly Spearman ρ is ≤ 0.38 over 22 months for LITE. For WILLY, alt_pump20_share has ρ 0.84 in 2026 but 0.01 in 2025, so it fails.
- **LITE bad months (2025-02, 07–12):** in every one of them the **falling-BTC cohort** (slope ≤ 0) lost money: −1.92, −0.64, −0.78, −0.57, −3.03, −0.54, −1.01. The rising-BTC cohort in the same months was −0.27, −0.07, +0.06, −0.31, +0.38, +0.11, −0.12.
  - Over all 2025 LITE fills, the falling-BTC cohort lost −168.6 percentage points against a net of −117.7; the rest made +50.9.
  - So LITE's 2025 loss is an **entry-time state** (BTC weakening at entry), not a day-level regime. The share of falling-BTC entries per month does not track the monthly result (ρ −0.06).
  - The worst 8 days carry 87 % of LITE's 2025 net, but the falling-BTC block is spread out: its single worst day is 3.3 % of the blocked loss and its worst 2 pairs 5.3 %.
- **WILLY 2025:** no day-level marker; ρ with the share of falling-BTC entries is −0.10. See §5 (question 2).
- **Verdict:** the 2025 FRENZY losses are **not on/off-switch effects at day granularity**. LITE's is an entry-level macro-stamp effect that only holds in the two bull halves. WILLY's sits in the fidelity-suspect subset outside the top 50.

## 5. The three specific questions

### Q1 LITE: candidate gate G1 "skip LITE when `entry_btc_1h_slope ≤ 0`" (YR6_WL_FRENZY_litegate.md; two ways)

| scope | blocked N · days | WR blocked | avg blocked [day-CI] | P(avg<0) | kept avg | breakeven WR | Δ$ if blocked (seed 1, today's size, fixed $2,900 book) | expectancy bar |
|---|---|---|---|---|---|---|---|---|
| 2025 all | 250 · 140 | 38.8 | −0.674 [−1.07, −0.28] | 1.000 | +0.088 | 49.9 | +$11,918 | pass |
| 2025 top-50 | 81 · 59 | 44.4 | −0.336 [−0.95, +0.31] | 0.854 | +0.125 | 49.8 | +$1,927 | **fail** (P) |
| 2026-H1 BEAR | 143 · 86 | 51.7 | **+0.321** [−0.14, +0.76] | 0.095 | +0.166 | 46.9 | **−$3,245** | fail (reversed) |
| 2026-H2 BULL | 86 · 45 | 38.4 | −0.495 [−1.23, +0.22] | 0.913 | +0.457 | 46.6 | +$3,008 | fail (P) |
| 2 years | 479 · 271 | 42.6 | −0.345 [−0.63, −0.06] | 0.989 | +0.170 | 48.7 | +$11,681 | pass (pooled only) |
| 2 years top-50 | 172 · 130 | 48.3 | −0.119 | 0.686 | +0.283 | 49.8 | +$1,447 | fail |
| seeds 2+3, 2025 / H1 / H2 | 495 / 285 / 176 | 39.0 / 51.6 / 38.1 | −0.663 / +0.312 / −0.518 | 0.999 / 0.10 / 0.92 | | | | same picture |

- Concentration is fine: worst 2 pairs carry 3.5 % and the worst day 1.9 % of the blocked loss.
- With the 30–50 % haircut, the two-year Δ is about **+$5.8k to +$8.2k** on the fixed book. The bear half shows the gate would have **cost** $3.2k there.
- G2, pre-registered as "only when BTC > SMA50" to explain the bear reversal, is refuted: H1 blocked +0.253 (58 fills); the complement +0.368.
- Master real fills: only 3 slope ≤ 0 LITE fills over 2 days (+1.00 avg) → nothing to read.
- **Status: OBSERVE-ONLY.** It fails the top-50 rule, the bear half reverses it, and H2 confirmation alone is P 0.91.

### Q2 WILLY: 2025 −0.24 vs 2026 +0.10 / +0.06 — what differs (YR6_WL_FRENZY_willy.md)

**Where the gap is:** pair-days outside the top 50.
- 2025: −0.323 (669 fills; P(diff<0) against top-50 = 0.968).
- 2026: +0.176 / +0.102.
- Top-50 WILLY is about 0 in **every** regime: +0.011 / −0.017 / +0.007. WILLY's 2026 profit itself comes only from pairs outside the top 50.

**How it loses there:**
- Stop share is 26.9 % against 16.2 % / 16.0 % in 2026, and 18.7 % for 2025 top-50.
- 90-minute-cap exits average −1.38 against −0.24 / −0.53.
- Average loss is −3.85 against −2.04 / −2.15. WR is almost unchanged (67.4 against 66.0 / 65.6).
- Stops overshoot on 1-minute bars (avg −4.33 against −4.01 on ticks), and so do TPs (+1.39 against +1.26). Clipping both changes the avg by < 0.01, so the exit ruler is **not** the explanation. The stop *rate* is.

**Tail:**
- 4 fills below −5 % (all outside the top 50, all exited on a 1-minute bar ≤ 46 s after entry) = −65.5 of the −213.6 percentage points. SAHARA 2025-11-29 is −40.3; SOON, SOLV and CATI make up the rest.
- Without them 2025 is −0.166.
- The worst 8 days = 57 % of the net. The worst 10 pairs by net carry 19 % of the gross loss; without them the avg is −0.062.

**Entries were more stretched in 2025** (median, 2025 against 2026-H1 / H2):
- +DI 41.3 against 36.5 / 37.4
- EMA gap 5-8: 0.97 against 0.78 / 0.84
- pair EMA13-50 gap: 4.4 against 3.6 / 4.2 (outside the top 50)
- ADX 48 against 42 / 44 (outside the top 50)
- mcap proxy 2× higher, so turnover is lower and the turnover filter is looser, as the brief warns

The 2025 winner zone (ADX > 46.6 ∧ gap 5-8 ≤ 0.965) confirms in H2 but **reverses in the bear half** (−0.58) → regime-conditional on a regime nobody can see in advance → not usable.

**Mix:** re-weighting 2025 to 2026's top-50 × BTC-slope mix closes only 8–14 % of the gap. No daily macro variable marks the bad months (§4).

**Answer:** WILLY's 2025 loss is a within-cell loss in small, non-top-50, more-stretched pumps, with heavier tails.
- It is either (a) a real 2025 small-cap regime that no measured variable sees in advance, or (b) entry-side fidelity, which the brief flags: 2025 data for pairs outside the top 50 is 1-minute bars and an mcap proxy.
- The data cannot separate the two. Inside the top 50, where 2025 fidelity is good, WILLY is flat in all three regimes.
- **No change.** Do not cut WILLY by 2025 evidence that only exists outside the top 50. Equally, WILLY's 2026 edge comes from the same non-top-50 bucket that lost in 2025, so treat it as unproven across regimes.

### Q3 LONG / WIDE: any separator worth keeping them for, and the LITE side effect (YR6_WL_FRENZY_lw.md)

**Exhaustive screen:** 0 survivors in either bull regime, pair or macro family. The fill-level BTC stamps also fail: LONG slope ≤ 0 blocked is −0.06 / +0.01 / −0.72, mixed sign.

**At +1.5:**
- 62–66 % of fills reach the TP, but breakeven is 67–68 %.
- Peaks ≥ +3 dropped from 50 % (2025) to 40 % (LONG H2).
- Old +3 against new +1.5: LONG 2025 +0.02 → −0.06, H1 −0.12 → −0.18, H2 −0.69 → −0.27. WIDE is +0.02 / +0.02 / +0.04 → −0.17 / −0.08 / −0.28.
- So +1.5 helps LONG only in H2 and hurts WIDE in every regime. This is a pointer for the open FRENZY_TP_REVERT scout gate, not a new proposal.

**Pair universe** (block = pair-day not in the top 50, post-hoc). The figures in brackets are the second computation (seeds 2+3).

| | 2025 | H1 | H2 | 2026 both | 2 years |
|---|---|---|---|---|---|
| LONG blocked avg (N) | −0.273 (136) | −0.351 (116) | −0.516 (68) | −0.412 (184) | −0.353 (320) [−0.360] |
| LONG kept avg | +0.154 | −0.002 | +0.022 | +0.006 | +0.071 |
| P(diff<0) | 0.93 | 0.88 | 0.88 | – | – |
| expectancy bar (blocked) | fail (P 0.91) | pass | pass | pass (P 0.993) | pass (P 0.996, WR 60 < 66.7) |
| WIDE blocked avg | −0.150 | −0.230 | −0.616 | −0.348 | −0.248 (P 0.95, fails WR/P) |

- The 24 h-volume terciles for LONG are non-monotonic in 2025 and H1 (the middle tercile is the best) and monotonic only in H2.
- So LONG's top-50 split is a **consistent-sign but unconfirmed** separator: H2 P 0.881 < 0.90. It did not survive the family-wise null, and I found it after looking. At 2× it is worth only $1.6k over 2 years before the haircut.
- **Status: OBSERVE-ONLY.**

**LONG-blocks-LITE:**
- (a) Replay counters (seed 1, including warm-up so an upper bound): **FRENZY_LITE_PAIR_HELD 51 (2025) + 41 (2026) = ≤ 92** refused LITE stretches from *any* holder (LONG, WIDE, WILLY, momentum, fades). The reverse direction, FRENZY / WIDE refused because LITE held the pair, is 12.
- (b) Proxy: LITE fills entering *after* a LONG / WIDE fill in the same (pair, spike) episode made **+0.491 %/fill, 57.6 % WR, N 66, 53 days** (2025 −0.13 · H1 +0.51 · H2 +1.38). LITE first-in-episode fills made −0.017 (N 1,161).
- (c) Bound: LITE is ≈ $71 per 1 % at 10× on $2,900. Even if all 92 were held by LONG / WIDE:
  - at the proxy +0.49 %, LONG / WIDE **cost** LITE ≈ $3.2k;
  - at DL 280's yr5 rate (8 of 10 lost, about −1.8 %), they **saved** ≤ $11.7k.
  - LONG / WIDE's own two-year result: −$1,783 at 2× / 1×.
- **The yr6 stamps cannot settle the sign.** The proxy argues against the DL 280 "free filter" read, but it is entries *after* LONG closed, not while LONG held the pair.
- Decisive test: replay both years with `frenzy_long_enabled` / `frenzy_wide_enabled` off and price the LITE fills that appear. That needs an operator go-ahead to run the replay.

## 6. Recommendation (no change now) and pre-committed gates

1. **LITE: no filter.** Open a Scout observe line **LITE_BTC1H_FALLING**.
   - Zone (frozen): FRENZY_LITE fill with `entry_btc_1h_slope ≤ 0`; unscored → excluded.
   - Counted from the next export, CLOSED, deduplicated on `(opened_at, pair, direction)`.
   - Decide once, at zone N ≥ 20 on ≥ 8 days:
     - ① zone avg ≥ kept avg → RETIRE.
     - ② else FILTER CANDIDATE for the operator if all of these hold: zone WR < LITE breakeven on the kept fills ∧ day-bootstrap P(zone avg < 0) ≥ 0.95 ∧ no pair or day ≥ 50 % of the zone loss. The 30–50 % haircut applies, and the backtest expectation is the two-year +$5.8–8.2k with the bear half at −$3.2k.
     - ③ else keep observing.
   - Never re-fit the 0 cut.
2. **LONG: no change.** Observe line **LONG_NONTOP50**.
   - Zone (frozen): FRENZY_LONG fill whose pair-day is not in the same-day or prior-day top-50 by quote volume (live equivalent: the engine's `entry_pair_rank > 50`, or 24 h-volume rank from the universe cache).
   - Same bar, N ≥ 20 on ≥ 8 days. Retire if zone avg ≥ rest avg.
   - LONG / WIDE stay ON at 2× / 1× (operator rule).
3. **WILLY: no change.** A reading for the next batch review: split live WILLY fills by top-50, because the 2026 replay says its edge is outside the top 50.
4. **Suggested next run (operator go-ahead):** a LONG / WIDE-off replay for 2025 and 2026, to price the LITE side effect directly.

## 7. What could NOT be tested

- **Columns not in the yr6 replay:**
  - all `entry_br_*` breadth-door stamps, `entry_bear_*`, `entry_funding_rate`;
  - `entry_pair_rank`, `entry_pair_age_days`;
  - `entry_btc_4h_ema50_200_gap_pct`, `entry_pair_1d_ndi`;
  - `entry_gap_5_20_*`, `entry_gap_5_8_signed_pct`.
- **Low coverage:** mcap / turnover / CMC rank (≤ 0.78 in 2025 LITE / LONG / WIDE; delisted coins have no proxy) were read only on scored fills.
- **2025 fidelity:**
  - Outside the top 50, both entry and exit use 1-minute bars and the mcap proxy (today's supply), so every 2025 non-top-50 result is suspect. The WILLY and LITE 2025 losses are mostly there.
  - The four 2025 WILLY crash fills (−65 percentage points) exited on 1-minute bars; a live stop-market fill price in such a crash is unknown.
- **Survivorship:** the alt-breadth panel uses `backtest_cache/k1d`, coins still listed at fetch time. It is only valid from 2025-01-31, the first date with 30 days of history.
- **Not modelled:**
  - second-order fills (a skipped LITE stretch frees the slot or pair for later entries);
  - the replay's entries come from the 01419d4 config;
  - LONG / WIDE holding times under +1.5 are shorter than replayed, so they blocked LITE less.
- **Not directly measurable:** LONG-blocks-LITE from stamps (no per-event refusal log; counters only).
- **Master real fills:** FRENZY has 16–20 kept fills per sleeve over ≤ 5 days → a sign check only, uninformative.
- **Seeds:** FRENZY seeds are almost one path, so seeds 2+3 are a computational cross-check, not independent evidence. The real second computation is the stdlib-csv recount in `study_yr6wl_frenzy_verify.py`; every headline N / WR / avg in §1, §5 Q1–Q3 matched to 3 decimals.

## 8. Exact commands

```
venv/bin/python -I scripts/study_yr6wl_frenzy_summary.py     # §1 table, top-50 split, monthly, coverage, low-coverage own read
venv/bin/python -I scripts/study_yr6wl_frenzy_screen.py --nperm 500   # exhaustive 1D+2D screen, day-block null, discover→confirm (~4 s)
venv/bin/python -I scripts/study_yr6wl_frenzy_regime.py      # ex-ante daily panel switch test, fill-level stamp gates, monthly ρ (~40 s)
venv/bin/python -I scripts/study_yr6wl_frenzy_willy.py       # WILLY / LITE 2025-vs-2026 decomposition, month × slope, master check
venv/bin/python -I scripts/study_yr6wl_frenzy_litegate.py    # LITE G1 / G2 gate, two ways (needs regime panel csv)
venv/bin/python -I scripts/study_yr6wl_frenzy_lw.py          # LONG / WIDE universe split, +1.5 read, LONG-blocks-LITE bound
venv/bin/python -I scripts/study_yr6wl_frenzy_verify.py      # independent stdlib recount of the headline numbers + LONG top-50 P
```
Shared helpers: `scripts/study_yr6wl_frenzy_common.py`. The replay meta files under `reports/backtest_cache/replay/` were only read (the refusal counters).
