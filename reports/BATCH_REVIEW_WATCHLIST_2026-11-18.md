# Batch-review watchlist — active sleeves (written 2026-11-18, for the next batch review)

Every gate is **ALERT-ONLY** (it flags, the operator decides) and is **read once** at its count, from the deploy of its DECISION_LOG entry + 10 min.
Run the scouts through `scripts/opportunity_scout.py` (or each script directly). Compare every sleeve to its baseline in
`reports/SLEEVE_BASELINES_2026-11-18.md` first, then read the lines below. Each sleeve gets its own rows (never pool sleeves).

**Paper sleeve set (11-18):**
- **ON:**
  - EXHAUSTION_SHORT (+ WIDE universe);
  - DISTRIBUTION_SHORT;
  - FLATBASE_SHORT;
  - FRENZY_WILLY_HT.
- **OFF:**
  - FRENZY_LONG and plain WILLY (DL 587);
  - LITE, LITE_RETRY and WIDE (Frenzy);
  - momentum long / short;
  - SURGE, spike fade / chase / bounce, flips, Bull-Run and Bear-Run (DL 578).

**Deploys that split this batch (read each sleeve before / after):**

| Commit | Time (UTC) | What changed |
|---|---|---|
| 2c7e78d2 | 11-18 04:45 | EXHAUSTION_SHORT sleeve + paper sleeve set, momentum off (DL 578) |
| 7587c2b2 | 11-18 14:08 | EXH exit TP≈10 / SL≤6 (DL 579); WILLY / HT 0.5×5× (DL 581); WILLY/HT watch lines (DL 580) |
| 702aeac4 | 11-18 15:06 | EXH_SSC_OIVOL filter (DL 582) |
| 52f2079e | 11-18 17:04 | depth guard min_frac 0.25 (DL 583) |
| eb7b5fd8 | 11-18 18:23 | impact-mode depth sizing + capacity stamps (DL 584) — **⚠ paper era break: paper entry % now includes spread + walk** |
| c1ff13cf | 11-18 20:39 | EXH WIDE universe (DL 585), DISTRIBUTION_SHORT (DL 586) |
| 0896f6f6 | 11-18 22:56 | HT own exit +5/−4/24 h + plain WILLY off + FRENZY_LONG off (DL 587); $200 depth floor (DL 588); 1000-coin spot map (DL 589) |

---

## 1. 🪫 EXHAUSTION_SHORT (FRENZY universe)
**Baseline (backtest):** 2.7 trades/day · WR 42 % · +8.7 / −4.8 · +0.68…0.85 %/trade · TP 34 % / SL 57 % / time 9 %.

**Gates (frozen once, alert only):**

| # | Line | Read at | Flag when | Scout |
|---|---|---|---|---|
| E1 | DL 578 sleeve gate (override-era fills) | first 20 fills | re-priced avg ≤ 0 → OFF · mean modelled slip > 0.25 % → OFF | `scout_exhaustion_short.py` |
| E2 | DL 579 exit override vs old exit (3/5/1.5/5) | 40 · 80 · 120 fills, ≥ 10 days | 90 % upper bound of paired Δ < 0 → REVERT exit | `scout_exh_exit_override.py` (replica check → UNRELIABLE) |
| E3 | DL 582 EXH_SSC_OIVOL filter | ≥ 30 blocked signals, ≥ 10 days | 80 % lower bound of the blocked mean > 0 → REVERT filter | `scout_exh_ssc_oivol.py` |

**By hand:**
- **Spot mapping (DL 589):**
  - EXH_SPOT_MAPPED count (1000PEPE / 1000BONK …);
  - any EXH_SPOT_MISMATCH blocks (a mapping fault — investigate each);
  - `entry_exh_spot_symbol` on fills.
- **Block tally:** EXH_NO_SPOT, EXH_SSC_OIVOL(_UNREAD) and EXH_READ_CAP. Read cap hits mean coins went unread.
- **Stop clustering:** stops per day and per hour; the worst day at today's size (EXH_EXIT2: worst backtest day −34 %; 2025-11-22 had 12 EXH stops). It feeds the pre-live size review.
- **Pending observation:** none. The entry W/L study (ENTRY_WL) found nothing at entry, and every exit alternative lost (EXH_EXIT2).

## 2. 🌐 EXHAUSTION WIDE universe (DL 585) — read apart from FRENZY-universe EXH
**Baseline:** about +1.0 trades/day · +0.55 %/trade (in-sample, 3 of 7 half-years negative).

| # | Line | Read at | Flag when | Scout |
|---|---|---|---|---|
| W1 | WIDE sleeve gate | ≥ 30 WIDE fills, ≥ 10 days | 80 % upper bound of mean net (re-priced) < 0 → OFF | `scout_exh_wide_585.py` |

**By hand:**
- **WIDE fills are excluded from E1–E3.** Check that the tagging (`entry_exh_universe`) is right.
- **Read-cap counters:** EXH_WIDE_READ_CAP (> 30 extra coins) and EXH_READ_CAP.
- **Thin books:** WIDE coins ($5M floor) are thinner. Read the capacity stamps (shrinks / skips, entry slip) for WIDE separately.

## 3. 📉 DISTRIBUTION_SHORT (DL 586)
**Baseline (RSI-kept, 1m — source DIST_EXIT_2026-11-18; the SLEEVE_BASELINES row is the pre-filter L1):** about 1.3 trades/day · WR 42.5 % · +0.93 %/trade · 7/7 half-years positive.

| # | Line | Read at | Flag when | Scout |
|---|---|---|---|---|
| D1 | sleeve gate | first ≥ 30 fills, ≥ 10 days | 80 % upper bound of mean net < 0 → OFF | `scout_distribution_586.py` |
| D2 | RSI1h ≥ 74.07 filter — kept side | ≥ 60 kept fills, ≥ 10 days | mean net ≤ +0.30 → disarm / review | same |
| D3 | RSI1h filter — blocked side (priced with Exit B) | ≥ 30 blocked signals, ≥ 10 days (freezes only after ≥ 1 kept fill) | 80 % lower bound > +0.30 → "filter blocks winners" | same |

**Observe lines (no gate yet):**
- **Thin coin (24 h vol ≤ $42M):** a down-size candidate only, never a filter (ENTRY_WL). Re-price it at the pre-live size review: Kelly 0.012 thin vs 0.098 rest.
  - **Thin-coin cost (DIST_THIN_DEEPDIVE_2026-11-19):** per DIST fill, entry drift + fill-vs-book + exit slippage, thin vs rest, vs the backtest's flat 0.10 %. Running count (paper export 2026-11-19 12:48): 2 fills, both thin, both losers (NAORIS never worked, peak +0.07 %; SENT gave back after +3.57 %).
- **Stamps** `entry_dist_di_diff1h` / `entry_dist_gap15` (the DIST_WL strength factor) and `entry_dist_rsi1h`.
- **Position-ratio lag stamps** `entry_dist_pos_lag_min` / `_span_min`. If the lag is often > 5 min, live reads a later row than the study did.

**By hand:**
- **DIST_LATE count:** signals refused for running out of the 300 s window on busy closes.
- **DIST_RSI1H_UNREAD:** should be rare.
- **DIST_BOOK_FULL**, and coin-taken-by-EXHAUSTION skips (the joint sim saw 14 % of DIST signals lost this way).

## 4. 🩳 FLATBASE_SHORT
**Baseline:** 0.63 trades/day · WR 74 % · +4.7 / −9.4 · +0.97…1.04 %/trade.

| # | Line | Read at | Flag when | Scout |
|---|---|---|---|---|
| F1 | NEAR_BASE skip (DL 577), skips priced as if traded | any time / first 10 | Σ > +15 pts → set to 0 · first 10 Σ ≥ 0 → 0, else KEEP | `scout_flatbase_confirm_watch.py` (nb_lines) |
| F2 | DL 577 live slippage | first 20 fills | avg > 0.6 % → halve the size | same |
| F3 | HIGHER_HIGH skip (DL 540) | any time / first 5 | Σ > +10 pts → OFF · first 5 Σ ≥ 0 → OFF, else KEEP | same |
| F4 | OBS-FB2 low-turnover (DL 407) | ≥ 30 fresh setups, ≥ 10 days, per cell | CANDIDATE (switch the turnover side) iff avg > 0 ∧ day-boot P ≥ 0.95 ∧ ≥ +1.0 pt over traded ∧ no day / pair ≥ 50 % · RETIRE if avg ≤ traded | scout FLATBASE section |

- **By hand:** size recomputed monthly (DL 577); the capacity stamps (Flatbase is a CLAUDE.md capacity-rule priority).

## 5. 🎲 FRENZY_WILLY_HT (DL 587: own exit TP +5 net / SL −4 / 24 h, 0.5 × 5×; plain WILLY OFF)
**Baseline:** full-history backtest −0.19 %/trade at the old +2 exit (no baseline yet for +5 / 24 h). Master 15 · 80 % · +1.78 (filters fitted on those fills). New breakeven WR 44.4 % (4 ÷ 9).

| # | Line | Read at | Flag when | Scout |
|---|---|---|---|---|
| H1 | DL 580 lines (re-based to the DL 587 exit): CROWDED_LONGS · NEAR_SPIKE_HIGH · REANCHORED | zone N ≥ 15, ≥ 8 days | RETIRED / FILTER CANDIDATE / KEEP | `scout_willy_ht_lines.py` |

**⚠ Stale lines, to re-base or retire at the review (rule ④, DL 406):**
- WILLY_ADX55 (DL 412) and WILLY_GAP (DL 414) are pinned to the old V2b exit keys. The HT exit changed in DL 587, so these lines read ⚠ STALE for HT.
- Decide: retire them, or re-register them on the +5 / −4 / 24 h exit.

**Before the first DL 580 verdict:** verify the long/short ratio timestamp alignment (fill − 1 min cut) on the live API.

**By hand:**
- WILLY_PLAIN_OFF counter: plain setups refused.
- HT exit mix (TP +5 vs SL −4 vs 24 h time).
- Hold time and slot use: an HT now holds up to 24 h.

---

## 6. Cross-cutting (every sleeve)
- **📖 Capacity (CLAUDE.md rule; EXHAUSTION and FLATBASE first):**
  - planned vs filled $;
  - % shrunk / skipped;
  - binding side;
  - modelled walk vs realised entry fill;
  - book $ at 0.2 / 0.5 / 1 %.

  Scouts `scout_depth_impact_584.py` and `scout_depth_band_583.py`, plus the 📖 Capacity table.
- **Depth-guard gates:**

| # | Line | Read at | Flag when |
|---|---|---|---|
| C1 | DL 583 band (fills at < 0.5 of plan) | first 10 band fills | Σ$ < 0 or mean live |entry slip| or mean live |exit slip| > 0.15 % → revert min_frac to 0.5 |
| C2 | DL 588 $ floor (fills at entry_cap_frac < 0.25) | first 20 such fills (probes excluded) | mean booked net % < 0 or mean live |entry slip| > 0.15 % → revert min_usd to 0 |
| C3 | DL 584 impact mode (`scout_depth_impact_584.py`) | first 20 LIVE impact taker fills | mean (fill-vs-book − modelled walk) > +0.05 % or mean fill-vs-book > 0.20 % → revert to sweep |
| C4 | DL 551 / 565 depth guard (by hand, LIVE only — no pricer) | 15 guarded live entries | Σ$ at full size vs the guarded result → guard off if full size made more |

- **⚠ Paper era break (DL 584):** paper entry % after eb7b5fd8 includes the modelled walk. Split before / after it.
- **Block-reason column:** spot-check that the per-sleeve lines match the journals for 2–3 coins.
- **The joint view:**
  - %/day per sleeve at one size, and the combined %/day;
  - worst day and same-day stop count;
  - compare to `reports/JOINT_SIM_2026-11-18.md`. The backtest's worst days were −35…−42 % with 17–19 stops at today's size.
- **💸 Funding (FUNDING_RULE_2026-11-19, descriptive, no gate):** average `funding_fee_usd` ÷ notional per EXH / DIST fill and its share of the winners' gross, vs the backtest (−0.20 % EXH / −0.09 % DIST per fill). Flag if live runs far above it.

## 7. Decisions parked for the pre-live review (do not re-raise before then)
- **Size of EXHAUSTION / WIDE / DISTRIBUTION / Flatbase / HT.** Today all are 0.5 × 5× ≈ 0.61 × equity.
  - Joint sim: about full Kelly in-sample, 1.2–1.8× Kelly after haircut / slip.
  - Edge-sized alternative: 0.18 / 0.31 × equity → about +0.54–0.66 %/day, −31 % max DD.
  - Include the DIST thin-coin down-size note.

## 8. Closed today (don't reopen without a new mechanism)
- **No-spot shorts:** entries (NOSPOT_SLEEVE), exits incl. TP 1 / SL 3 (NOSPOT_EXIT), and BTC-bear switch (NOSPOT_BTCBEAR) → nothing. The one frozen watch: R4 × 5m-bear, re-read only after ≥ 30 new bear days.
- **Exits** of EXHAUSTION (EXH_EXIT2) and DISTRIBUTION (DIST_EXIT): keep today's.
- **Entry runner-vs-give-back separation (ENTRY_WL):** nothing on EXH; only the DIST thin-coin observe.
- **Funding rule for EXH / DIST (FUNDING_RULE_2026-11-19):** skip / half-size at negative funding, 20 rules → none pass (HOLDOUT negative except the EXH ≤ −1.0 pair, which flips sign vs DESIGN). Backtests already include real funding.
- **CRASH_DIP_LONG:** dropped (too few trades, one window = 77 % of profit).
