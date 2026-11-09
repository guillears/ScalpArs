# WILLY (FRENZY_WILLY) sleeve review — 2026-11-09

Research only. Nothing in the bot was changed. WILLY_HT is a separate sleeve and is not included anywhere below.

**Data used**
- **Current batch B25:** export `scalpars_orders_paper_2026-11-09_12-45-05.csv`. These are the closed FRENZY_WILLY fills opened on or after 2026-10-23 18:30 UTC: 33 fills, booked %.
- **Master:** `reports/MASTER_POOL_stacked.csv`, rows with `stack_keep` and without `is_probe`, priced at `stack_pct`: 38 fills in batches B19 to B24.
- **Daily compound (DCR):** every fill at ONE size, X = 0.61 (the same X as the best case), using `one_size.dcr`. For master, the DCR is the mean of the per-batch DCRs.
- **Check against the best case:** this setup reproduces `BEST_CASE_2026-11-09_12-45` exactly. Before: −1.52 (B25) / +1.73 (master). With WILLY_GAP: −0.26 / +1.83.
- **Luck test:** `luck_test.block_line` with 1,000 day-block shuffles on the pooled 71 fills. These are in-sample numbers.
- **Breakeven WR:** 70.6 % on today's stack. The frozen gate value is 69.0 %.
- **Scripts:** `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/willy/` (`lines.py`, `cands.py`, `vwap.py`, `path_master.py`, `path_sum.py`, `macro.py`, `null_run.py`).

**Baseline**
- **B25:** 33 · 48 % · −1.33 avg · Σ −43.90 · DCR **−1.52 %/day**
- **Master:** 38 · 82 % · +0.71 avg · Σ +26.98 · DCR **+1.73 %/day**

## 1. Registered watch lines (block lines) — before → after

Each cell reads N · WR · avg · Σ %.

| Line (registered) | Exit era | B25 zone | B25 kept | B25 DCR | Master zone | Master kept | Master DCR | Luck (search) |
|---|---|---|---|---|---|---|---|---|
| WILLY_GAP (DL 414) — ADX > 55 or gap8-13 ≤ 0.90 | V2b ✓ | 19 · 37 % · −1.91 · −36.4 | 14 · 64 % · −0.54 · −7.5 | −1.52 → **−0.26** | 17 · 71 % · −0.24 · −4.0 | 21 · 90 % · +1.48 | +1.73 → **+1.83** | NOT ABOVE LUCK (p_single 0.005, p_search 0.99) |
| WILLY_ADX55 (DL 412) | V2b ✓ | 5 · 0 % · −4.00 | 28 · 57 % · −0.85 | → −0.83 | 6 · 67 % · −0.34 | 32 · 84 % · +0.91 | → +1.62 | NOT ABOVE LUCK |
| MKT_HOT (Market-hot V2) | V2b ✓ | 15 · 47 % · −1.68 | 18 · 50 % · −1.04 | → −0.65 | 21 · 81 % · **+0.82** | 17 · 82 % · +0.57 | → **+0.24** | NOT ABOVE LUCK (zone wins in master) |
| WILLY_BTC1H (DL 275/409) — block BTC 1h slope ≤ 0 | V2b ✓ (re-based) | 21 · 62 % · −0.60 | 12 · **25 %** · −2.61 | → −1.08 | 15 · 73 % · +0.28 | 23 · 87 % · +0.99 | → +1.15 | NOT ABOVE LUCK (B25 points the wrong way) |
| DIP_SUB (DL 279) — keep only BTC rising ∧ bar ≤ −1.16 | V2b ✓ | 25 · 56 % · −0.86 | 8 · 25 % · −2.81 | → −0.77 | 27 · 81 % · +0.50 | 11 · 82 % · +1.23 | → +0.88 | NOT ABOVE LUCK |
| gap-only (half of WILLY_GAP) | V2b ✓ | 14 · 50 % · −1.17 | 19 · 47 % · −1.45 | → −0.95 | 11 · 73 % · −0.18 | 27 · 85 % · +1.07 | → +1.93 | NOT ABOVE LUCK |

**Leave-one-out (LOO)**
- **WILLY_GAP:** in B25 the result is robust (+36.4 → +32.4 without TUSDT). In master its whole gain is ONE fill (OGNUSDT +4.03, 100 % of the Δ; without it Δ is +0.01). Master gives WILLY_GAP essentially no support; B25 alone carries it.
- **ADX55, gap-only, BTC1H, DIP_SUB:** in master, one fill carries 50 % or more of the Δ (OGN, STRK, INIT).

**Lines not re-read here (not block lines)**
- **Exit lines, ⚠ STALE (they walk the pre-V2b +1.25 exit):** TIMECAP, TP125, the 30-min and higher-TP cells, STOP, AVGDOWN, MKTDIP. MKTDIP is a "+2 after +1.25" exit line that also needs a breadth input that is not stamped.
- **Add lines (missed entries, not blocks):** LATEJOIN, NOMISS.
- **Sizing lines:** WILLY_5X and HALF SIZE. The WILLY_5X gate rule is "first 15 fills avg < 0 → SLEEVE REVIEW". By the stamped numbers this sleeve review is consistent with that rule having triggered.

### Pairs of lines (2D of lines)

Two ways to combine a pair:
- **OR** = both lines switched on (block the union of their zones).
- **AND** = block only the fills that sit in both zones.

| Pair | B25 kept | B25 DCR | Master kept | Master DCR |
|---|---|---|---|---|
| GAP OR MKT_HOT (best-case runner-up) | 7 · 57 % · −0.70 | −0.17 | 11 · 82 % · +0.67 | **+0.16** |
| GAP OR BTC1H | 5 · 60 % · −1.05 | −0.18 | 13 · 92 % · +1.50 | +0.92 |
| GAP OR DIP_SUB | 3 · 67 % · −0.83 | −0.09 | 5 · 100 % · +2.85 | +0.84 |
| MKT_HOT OR DIP_SUB | 1 · 0 % | −0.14 | 3 · 100 % | +0.20 |
| GAP AND MKT_HOT | 25 · 56 % · −0.85 | −0.74 | 27 · 89 % · +1.23 | +1.91 |
| GAP AND BTC1H | 21 · 43 % · −1.60 | −1.16 | 31 · 87 % · +1.11 | +2.06 |
| BTC1H AND gap-only | 23 · 39 % · −1.81 | −1.43 | 34 · 85 % · +0.97 | +2.18 |
| (the other 29 pairs) | none is better than −0.26 on B25 while keeping master ≥ +1.7 | | | |

**Result:** no line and no pair of lines turns B25 positive while keeping master. Every OR pair that gets B25 close to 0 does it by blocking 28 to 32 of the 33 fills, and costs master 0.8 to 1.6 %/day.

## 2. Price-path decomposition (`path_labels.py`; stop Y = 3.9 % gross; % at one size)

| Group | Master (38) | B25 (33) |
|---|---|---|
| Entry WORKED / NEVER | 31 / 7 (**82 %**) | 21 / 12 (**64 %**) |
| NEVER_WORKED | 5 · −20.05 | 6 · −24.03 |
| STOPPED_RECOVERED | 1 · −4.06 | 5 · −20.02 |
| GAVE_BACK (exit side) | 1 · −4.00 | 6 · −19.29 |
| LEFT_ON_TABLE | 21 · +30.60 | 13 · +17.50 |
| CAPTURED | 10 · +24.50 | 3 · +1.94 |
| Entry-side vs exit-side loss | −24.1 vs −4.0 | **−44.1 vs −19.3** |

**Verdict: WILLY's loss is an ENTRY problem.** Entry-side losses are about 2.3 times the exit-side losses in B25 and 6 times in master. The share of entries that reach +1 % before the stop fell from 82 % to 64 %.

**Exit cross-check.** This is a rough flat-TP estimate, an upper bound because reach is measured over 24 h, not inside the 90-min hold:
- **B25, all fills:** −43.9 → at best −19.4 (TP +1.25). The batch still loses.
- **Master:** +27.0 → +10.3 (TP +1.25). Master loses most of its gain.

No exit change fixes the sleeve, and an exit change flips sign between master and B25. Under the joint-check rule that means no exit ship.

## 3. Sweep (`sweep_separators.py FRENZY_WILLY`, 71 fills, era split 10-20)

**Coverage.** 65 numeric and 14 level columns were tested.

**Blind spots (below 80 % coverage, not tested):**
- alt_beta24 / alt_corr24 / funding / gap_5_* signed (35 %)
- book_* / fill_vs_book / order_lag (66 %)
- btc_4h_ema50_200 / pair_1d_ndi / pair_age (45 %)
- frenzy_gvol 44 %, willy_spike_high 48 %, willy_nomiss 49 %
- mkt_up4h 8 % (rebuilt for MKT_HOT above)
- pair_alpha 14 %
- 33 columns never stamped on this sleeve

**Shuffled null (day blocks, 150 shuffles for 1D, 200 for 2D):**

| Label | 1D survivors vs null | 2D survivors vs null | Best split vs luck | Verdict |
|---|---|---|---|---|
| path (primary) | 48 vs 49.4 (p 0.51) | 874 vs 1,052 (p 0.73) | 1D p 0.33 · 2D p 0.51 · 3D 0 | not above luck |
| reach | 82 vs 81.0 (p 0.52) | 2,916 vs 3,321 (p 0.85) | 1D p 0.39 · 2D p 0.82 · 3D 0 | not above luck |
| pnl (booked) | 51 vs 76.0 (p 0.97) | 2,067 vs 2,892 (p 0.99) | 1D p 0.44 · 2D p 0.52 · 3D 0 | not above luck |

**Expected by luck:** about 49 of 48 path 1D survivors, about 1,050 of 874 path 2D survivors, and about 2,900 of 2,067 pnl 2D survivors. The whole sweep is at luck level.

### Best candidates that are NOT registered (screen only)

| Candidate | B25 kept | B25 DCR | Master kept | Master DCR | Luck (single / 1D / search) |
|---|---|---|---|---|---|
| **WILLY_FAILED_SPIKE: block `entry_frenzy_vs_vwap_pct` ≤ 0** (red signal bar closed at or below the spike-anchored VWAP; identical to above_streak = 0) | 11 · 82 % · +0.21 | −1.52 → **+0.07** | 15 · 93 % · +1.87 | +1.73 → **+2.00** | 0.001 / 0.14 / 0.70 → NOT ABOVE LUCK |
| 2D BTC off-30d-high ≤ median ∧ streak ≤ median | 11 · 82 % · +0.21 | +0.07 | 26 · 88 % · +1.30 | +1.86 | 0.001 / **0.02** / **0.14 → UNCLEAR** (best luck read of the review) |
| GAP OR WILLY_FAILED_SPIKE | 5 · 100 % · +1.05 | +0.18 | 11 · 100 % · +2.40 | +1.77 | 0.001 / 0.04 / 0.30 |
| above_share ≤ median | 17 · 76 % · −0.14 | −0.09 | 18 · 89 % · +1.50 | +2.02 | search 0.98 |
| 2D NOT(streak > median ∧ stop_atr ≤ median) | 10 · 80 % · +0.10 | +0.03 | 6 · 100 % · +3.64 | +1.85 | search 0.99 |
| 2D neg_di ≤ median ∧ vol/mcap > median (top path 2D) | 26 · 58 % · −0.82 | −0.74 | 30 · 87 % · +0.98 | +1.28 | search 1.0 |
| 2D btc_atr ≤ median ∧ stop_atr > median | 28 · 54 % · −1.08 | −1.04 | 32 · 88 % · +1.12 | +2.29 | search 1.0 |

### WILLY_FAILED_SPIKE zone measured against the expectancy filter bar

| Check | Pooled (master + B25) | Master only | B25 only |
|---|---|---|---|
| N | 45 | 23 | 22 |
| WR (bar: below 70.6 %) | 53 % | 74 % | 32 % |
| Avg % | −1.05 | −0.05 | −2.10 |
| P(mean < 0), day bootstrap | 0.99 | 0.53 | 1.00 |
| Days | 26 | 12 | 14 |
| Worst day / worst pair share of the loss | 12 % / 9 % | 25 % / 17 % | 17 % / 11 % |

**What this means**
- **Pooled:** all four parts of the expectancy filter bar pass.
- **Master alone:** fails.
- **Per batch:** the zone does worse than the kept fills in 5 of 6 batches with both sides (B19 is the reverse).
- **Theory:** the dip has already taken price below the average price paid since the spike, so the spike has failed and the buyers since then are underwater.

**Why WILLY_FAILED_SPIKE is not ready**
- It was found post-hoc in this sweep, and the search p is 0.70.
- The threshold is a knife edge in B25. The kept side at −0.25 / 0 / +0.25 averages −0.14 / **+0.21** / −0.67.
- DL 303 refuted a related cut on the old exit (vs_vwap ≤ 0.10 ∧ run ≤ 10). That was a different definition, but it was a similar idea.

**Ruling:** OBSERVE-ONLY candidate, frozen at ≤ 0. Do not arm it.

## 4. Sleeve-kill checklist

① **Pair-level dimensions: done.** 50 pair-level numeric columns and 14 level columns, each at 3 granularities, plus every 2D pair (8,320 cells) and 3D-from-2D (630). All three labels are at luck level.

② **Macro / regime dimensions at sign granularity, then buckets: done.** 24 macro columns, 55 tests per label.
- Only `btc_r72_pct` (pnl) and `btc_dist_from_ema13_pct` (path) are consistent at sign granularity. Both are weak: +0.18 / +0.24 %, and −0.05 / −0.26 WR points.
- Buckets: btc_atr (median) and btc_above72 (median) are consistent on pnl. Nothing is above the null.

③ **Uniform degradation: YES. Every cohort degraded together.**

| Cohort | Master WR / avg | B25 WR / avg |
|---|---|---|
| GAP zone | 71 % / −0.24 | 37 % / −1.91 |
| GAP kept | 90 % / +1.48 | 64 % / −0.54 |
| BTC1H, both sides | 73–87 % | 25–62 % |
| MKT_HOT, both sides | ≈ 81 % | 47–50 % |
| VWAP kept | 93 % / +1.87 | 82 % / +0.21 |

By the checklist this points to an **unmeasured regime variable, not to WILLY's entry rule**.

The master edge is also concentrated in time:

| Batch | WILLY fills | Σ % | Avg % |
|---|---|---|---|
| B19 (10-09 to 10-11) | 14 | +11.2 | — |
| B20 | 3 | +5.0 | — |
| B21 | 2 | +2.4 | — |
| B22 | 8 | — | +0.01 |
| B23 | 9 | — | −0.05 |
| B24 | 2 | +8.7 | — |
| B25 | 33 | — | −1.33 |

Apart from the 2 fills of B24, WILLY has been at about 0 or negative since 10-15.

④ **Tape context** (median of the entry stamps, master vs B25; BTC daily close):
- **BTC price:** about 82–84k during B19–B21, then a slide from 81k (10-23) to 74.8k (11-08).
- **BTC off its 30-day high:** −5.4 % → −10.3 %.
- **BTC 72h return:** −0.4 % → −1.9 %.
- **BTC 1h RSI:** 54 → 45.
- **BTC 1h slope:** + → −.
- **Pair ATR:** 1.62 → 2.07 %.
- **Pair ADX at entry:** 45 → 39.

WILLY buys the first red bar after a coin spike. In a BTC down-leg those dips keep falling: STOPPED_RECOVERED 1 → 5, GAVE_BACK 1 → 6. The best-luck item (off-30d × streak, UNCLEAR) is consistent with this. But B25 sits almost entirely in the deep-drawdown regime, so the regime and the batch cannot be separated yet.

## 5. Joint entry × exit check

**WILLY_FAILED_SPIKE-kept fills under today's V2b exit:**
- Master: 6 CAPTURED + 8 LEFT_ON_TABLE, 1 NEVER_WORKED.
- B25: 8 LEFT_ON_TABLE + 1 CAPTURED, 1 GAVE_BACK, 1 STOPPED_RECOVERED.

**Flat-TP estimate on those kept fills:**
- B25: +2.3 → +4.4 / +6.7 / +13.4 (TP 1 / 1.25 / 2).
- Master: +28.0 → +8.7 / +10.4 / +18.8.

The exit idea flips sign between master and B25. Under the joint-check rule that means no exit ship; keep V2b. The entry candidate was judged under V2b, the exit that will run.

## 6. Recommendation

1. **Is removal justified today? NOT YET by the locked checklist.** Item ③ (every cohort degrading together) points to a regime cause. The checklist makes a switch-off the last resort. But the evidence is plain:
   - No line, pair of lines or sweep cell gives WILLY a B25 kept set above the 70.6 % breakeven with a positive average that holds both ways.
   - The best one, WILLY_FAILED_SPIKE, gives +0.07 %/day on 11 fills, which is in-sample and needs the 30–50 % haircut.
   - Master's support for WILLY_GAP is a single fill.
2. **WILLY_GAP:** keep it as registered. Its frozen checkpoint (≈ Nov 15 / Dec 03) still runs. It is NOT ABOVE LUCK, and on master it adds about nothing (Δ +0.01 % without OGNUSDT).
3. **Register WILLY_FAILED_SPIKE as a new observe line** (operator approval needed; nothing was built):
   - Zone: `entry_frenzy_vs_vwap_pct` ≤ 0, frozen. It is a pair-level variable, so evidence counts in trade units.
   - Gate, using the WILLY_GAP template: zone N ≥ 15 on ≥ 8 days of clean forward fills.
   - RETIRE if zone avg ≥ kept avg or zone avg ≥ 0.
   - FILTER CANDIDATE if WR < 69 % ∧ P(mean < 0) ≥ 0.95 ∧ no day / pair ≥ 50 %.
   - Do not re-fit the threshold.
4. **Regime observe line (window units), operator call:** WILLY results split by BTC off-30d-high (for example ≤ −8 %), counted in DAY / window units. This is a sleeve on/off switch, not a trade filter.
5. **Removal trigger to pre-commit:**
   - At the WILLY_GAP checkpoint, the forward GAP-kept AND FAILED_SPIKE-kept fills are both below 69 % WR with avg < 0 → WILLY OFF.
   - If WILLY earns money again outside the BTC drawdown regime, that refutes the regime story and the question is revisited.
   - Until then, WILLY stays at today's half size (no step up, DL 503).

## 7. BTC macro switch (operator follow-up: "BTC was falling, the market dumping")

This section asks whether WILLY should be switched OFF during a BTC regime. A regime variable is market-wide, so it counts in DAY units: all fills on the same day are one observation. Scripts are in the scratchpad folder: `macro2.py`, `switches.py`, `fetch_btc1h.py` (the fetch used 13 BTC 1h requests, paced).

**Coverage.** Every BTC / market-wide stamp is 100 % scored on master (38) and B25 (33), with one exception:
- `entry_btc_4h_ema50_200_gap_pct` is stamped on 0 % of master and 97 % of B25.
- `mkt_up4h` is 0 % / 18 %; MKT_HOT is rebuilt for every fill in `zones.csv`.

To cover the gaps, these were rebuilt from BTC 1h klines (closed bars at entry, 2024-11 to now) for EVERY fill: off-30d-high, returns over 24h / 72h / 7d, 1h EMA20 slope, 1h RSI, 1h EMA13/50 gap, 4h EMA50/200 gap, 4h RSI, daily close vs daily EMA20. The rebuilt values agree with the stamps: correlation 0.92 to 1.00, same sign on 93 to 100 % of fills.

There are no ETH trend stamps; only `eth_5m_ret1` exists. Breadth bull / bear %, global volume ratio, market-hot, macro_trend and btc_regime were included.

### Scan

35 variables, 70 tests: a sign split (or median split when the variable has no natural zero), then terciles. Each test compares master and B25 separately. A test counts as consistent when the difference points the same way in both periods and is larger than 0.1 % in each.

- **Consistent tests: 14 of 70.** Day-block shuffles produce **26.2 on average (p 0.96)**.
- **Best strength: 1.39**, against the null 95 % of 2.04 (p 0.34).
- **Result: no macro variable is above luck.**

The consistent ones, as a screen only (low side vs high side, average %):

| Variable | Master low / high | B25 low / high |
|---|---|---|
| BTC ATR % (quiet BTC is worse), tercile | +0.44 / +1.83 | −2.56 / −0.78 |
| BTC below its EMA13 (better in both) | +1.06 / +0.51 | −0.36 / −2.36 |
| BTC 1h EMA13/50 gap < 0 (worse in both) | +0.40 / +1.19 | −1.45 / −0.89 |
| BTC 4h RSI, median (lower is worse) | −0.06 / +1.21 | −1.42 / −1.16 |
| BTC 72h return < 0 (worse in both) | +0.65 / +0.80 | −1.41 / −1.11 |
| Global volume ratio, BTC eff72, BTC above72, bull %, BTC RSI closed | small, same direction | small, same direction |

### Pre-declared drawdown switches

WILLY is switched OFF while the condition holds. All values are rebuilt. "In" = fills taken while the condition held.

| Switch | Master in | Master out | B25 in | B25 out | DCR B25 | DCR master | Day units: in / out (avg of daily means) | Day-shuffle p | LOO by day |
|---|---|---|---|---|---|---|---|---|---|
| (a) BTC ≤ −8 % below its 30-day high | **0 fills** | 38 · 82 % · +0.71 · 12 d | 20 · 55 % · −1.06 · 11 d | 13 · 38 % · −1.74 · 6 d | −1.52 → −0.78 | +1.73 → +1.73 | 10 d −0.91 / 17 d −0.07 | 0.82 | Δ +0.85, min +0.50 |
| (b) BTC 4h EMA50 < EMA200 | **0 fills** | 38 · 82 % · +0.71 | 31 · 48 % · −1.41 · 14 d | 2 · −0.16 | −1.52 → **−0.01** | +1.73 → +1.73 | 14 d −1.30 / 13 d +0.62 | 0.40 | Δ +1.92, min +1.61 |
| (c) BTC 1d return < 0 AND 72h return < 0 | 12 · 83 % · +0.55 · 4 d | 26 · 81 % · +0.78 | 20 · 50 % · −1.21 · 10 d | 13 · 46 % · −1.52 | −1.52 → −0.68 | +1.73 → +1.56 | 12 d −0.88 / 15 d +0.02 | 0.37 | Δ +0.90, min +0.59 |
| (d) BTC 1h EMA20 slope < 0 | 17 · 76 % · +0.05 · 6 d | 21 · 86 % · +1.24 | 22 · 59 % · −0.75 | 11 · 27 % · **−2.48** | −1.52 → −0.94 | +1.73 → +1.70 | 18 d −0.38 / 9 d −0.37 | 0.79 | Δ +0.01, min −0.30 |

**Reading the table**
- **(a) and (b) are the batch itself.** No master fill was ever in these regimes. B25 is the only regime episode in the real fills, so the window count is one and cross-period consistency cannot be tested.
- **(c) is weak.** It helps B25, and the out side still loses in B25 (−1.52), so the switch does not separate good from bad.
- **(d) goes the wrong way in B25.** Fills taken while BTC's 1h slope was rising lost more in B25.

### Replay cross-check (secondary)

Source: `PREP_B25_BT_WILLY_rows.csv`, arm A_replay. That is the yr6a 2025 + yr6 2026 engine replay, seeds 1 to 3, warm-up trimmed, re-priced on today's V2b exit: 4,129 WILLY fills.
- **WILLY overall by period:** 2025 −0.37 avg (63 % WR) · 2026-H1 +0.33 · 2026-H2 +0.03.

Avg % in the regime minus avg % out of it, for each of the 3 seeds:

| Switch | 2025 | 2026-H1 | 2026-H2 |
|---|---|---|---|
| (a) | −0.52 / −0.38 / −0.36 (agrees) | **+0.37 / +0.37 / +0.34 (contrary)** | +1.48 (contrary; only 9 fills on 2 days) |
| (b) | −0.35 / −0.22 / −0.20 (agrees) | **+0.27 / +0.26 / +0.24 (contrary)** | **+0.42 / +0.40 / +0.38 (contrary)** |
| (c) | −0.31 / −0.36 / −0.40 (agrees) | **+0.59 / +0.59 / +0.56 (contrary)** | −0.54 / −0.51 / −0.54 (agrees) |
| (d) | −0.54 / −0.42 / −0.42 (agrees) | +0.11 / +0.11 / +0.08 (contrary) | −0.89 / −0.87 / −0.91 (agrees) |

In the 2026-H1 bear half, WILLY did BETTER inside every drawdown regime. No switch holds in all three periods.

### Verdict

**No macro switch was found.** 35 variables plus 4 pre-declared switches were tested. The scan is at luck level (14 consistent tests vs 26 expected by luck). Every drawdown switch is either one regime episode (B25) or flips across replay periods.

The B25 drawdown explains the batch only in the sense that B25 IS that regime. Nothing shows WILLY reliably loses in BTC drawdowns: the 2026-H1 bear half refutes that in the replay.

**Optional line (one episode = watchlist only).** If the operator wants a line, register **WILLY_BTC4H_BEAR** observe-only:
- **Zone:** rebuilt BTC 4h EMA50 < EMA200 at entry. The stamp exists now (97 % on B25).
- **Gate:** counted in DAY units from a NEW regime episode (B25 days excluded).
- **OFF candidate (operator decides)** iff, at ≥ 8 in-regime days and ≥ 15 fills, the in-regime average is below 0 with a day-bootstrap P ≥ 0.95, AND the out-regime average is above 0 over the same span.
- **RETIRE** if in-regime ≥ out-regime.
