# EXH EXIT GRID: should EXHAUSTION_SHORT take profit later, stop wider, or trail? (2026-11-18)

**Pre-registered:** `EXH_EXIT_GRID_2026-11-18_PREREG.md`, frozen 12:40 UTC before any exit result. The operator's stop addendum (−5 % / −6 % fixed in every family, plus min(1.5 ATR, 6 %)) went in before results. The selection was frozen in a second addendum before holdout E was simulated. E was read once.

Research only. Nothing in the bot changed. Nothing was committed.

## Question
EXHAUSTION_SHORT (paper) exits at:
- TP = min(3 × ATR15, 5 %), a resting maker order;
- SL = min(1.5 × ATR15, 5 %);
- a 24 h time exit.

On ZKPUSDT today the TP took +5 % and the coin went on to fall ~12 %. The operator asks four things:
1. Why 3 × ATR and not 2, 1.5 or 1 × ATR?
2. Why not a fixed +6 / +7 / +8 % TP with the −5 % stop?
3. Would a trailing exit do better?
4. Would a −6 % stop do better than −5 %? (ZKP went +3.96 % against us before it fell.)

## Plain answer
1. **Smaller ATR take-profits are worse.** With the 5 % cap, 1 / 1.5 / 2 / 2.5 / 3 × ATR earn +0.32 / +0.39 / +0.44 / +0.45 / +0.46 % a trade. 3 × ATR was the best of them. The binding constraint is the **5 % cap**, not the multiplier.
2. **Larger fixed take-profits earn more on average, on every block we have, including the new clean holdout.**
   - With a 5 % stop, TP 4 / 5 / 6 / 7 / 8 / 10 % earn +0.37 / +0.47 / +0.53 / +0.58 / +0.59 / +0.59 % a trade.
   - With a 6 % stop: +0.39 / +0.49 / +0.56 / +0.63 / +0.66 / +0.69.
   - The shipped exit earns +0.46.
   - **However:** the gain lives in the big winners. Remove the best 5 % of trades and the wide TPs lose their edge over the shipped exit. The family-wise test also cannot rule out luck (p_fw 0.37 for the pick).
3. **Trailing exits are worse than the shipped exit in every version tested** (288 cells): at best +0.41 a trade, and only +0.12 – +0.30 when armed below +5 %. These coins bounce 3 % or more on the way down, so a trail gets shaken out. Each trail exit also pays taker fee plus slippage.
4. **A 6 % stop alone changes nothing with the 5 % TP:** Δ +0.04 %/trade, CI −0.03 … +0.11. The fills it saves (191) are paid for by the fills that just lose 1 % more (1,589).
   - The wider stop only pays together with a wider TP. With TP 2.5 × ATR, 6 % beats 5 % by +0.13 (CI +0.03 … +0.22).
   - On ZKP the stop never mattered: the worst move against us was +3.96 %, inside both stops.
5. **The shipped exit leaves a lot on the table.** Of 2,089 TP fills, the price kept falling a median **+7.1 % further** within 24 h of entry. 64.6 % of TP fills reached ≥ +10 % in total, and 64.6 % were still below the TP price 24 h later.
6. **Verdict by the pre-registered rule: OBSERVE, not CHANGE.**
   - The pick (TP 2.5 × ATR uncapped, SL 6 %) is +0.71 vs +0.46 a trade on seen data and +0.29 vs +0.11 on the clean holdout.
   - It fails gate 1 (family-wise null) and the best-5 %-removed part of gate 3.

## Method
- **Entries (frozen):** the shipped rule, realisable version (2): the `sl` mask (entry ≥ first shortlist, ≤ flag end), vd ≥ 0.6366, spot_share_chg < −0.0066, 15m RSI14 cross 70 → ≤ 70, entry at the next 5m open, one position per pair (re-applied per cell).
  - **Parity:** N 4,015 · 3.25/day · +0.458 %/trade at $500 slip, halves +0.59 / +0.47 / +0.46 / +0.21 / +0.55 / +0.13 / +0.84. This matches EXHAUSTION_EXPAND (2).
- **Price path:** 1m futures klines from data.binance.vision daily zips, 8,490 + 543 pair-days, 0 API weight. 100 % of seen trades and 100 % of E trades ran on 1m (the D-hold run 99.9 %).
  - **Replica check:** the shipped cell on 1m reproduces the 5m study's exit kind for 4,138 of 4,140 candidates (mean Δ 0.00005 %).
- **Fill rules:**
  - The stop as it stood at the bar's open is checked first, on the bar high.
  - The maker TP is checked next, on the bar low, and must trade through by ≥ 1 tick.
  - If a bar touches both, the stop wins.
  - Stops fill at max(stop, bar open), so gaps are charged.
  - The trailing best low updates after the checks; the new trail level applies from the next bar.
- **Costs:** taker 0.05 % + modelled slip at **$5k notional** (the shipped size, X = 0.5 of a $10k account, see assumptions) on entry and on every stop, trail or time exit. Maker TP 0.02 %, no slip. Real funding to each cell's own exit time. Stress columns: $1k model and flat 0.05 / 0.10 / 0.20 %.
- **Cells:** 378 grid cells + 18 hold variants = 396.
  - A. 60 cells: k × ATR, k ∈ {1, 1.5, 2, 2.5, 3}, cap none / 5 / 8 %.
  - B. 30 cells: TP 4 – 10 %.
  - C / C2. 288 trailing cells: arm at a ∈ {2, 3, 4, 5} %, trail b ∈ {1, 1.5, 2, 3} % or {0.5, 1} × ATR, TP cap none / 10 / 15 %.
  - D. Hold 12 h / 48 h for the top-3 cells of A, B and C ∪ C2.
  - **Stops:** S1 = min(1.5 ATR, 5) (shipped) · S2 = min(1.5 ATR, 6) · S3 = 5 % · S4 = 6 % (+ F4 = 4 % in B).
- **Selection:** on DESIGN (2025-01 → 2026-06) only.
  - Metric: the one-size daily compound return (X = 0.5), smoothed over each cell's grid neighbours.
  - Family-wise max-T: sign-flip on DESIGN day blocks, 2,000 draws, all 378 cells.
- **Holdout E (new, clean):** 2023-01 → 2023-06.
  - Built with the same walk and builder (`exh_expand_build.work`, unchanged; T0 2022-12-01).
  - 567 episodes on 150 pairs → 248 candidates on 100 pairs.
  - E is a slow half: only 1.3 trades a day.
- **Scripts:**
  - `scripts/exh_exit_grid_data.py` (1m + E data)
  - `scripts/exh_exit_grid_E_build.py`
  - `scripts/exh_exit_grid_sim.py`
  - `scripts/exh_exit_grid_eval.py`
  - `scripts/exh_exit_grid_report.py` (selection + null)
  - `scripts/exh_exit_grid_final.py` (tables + charts)
  - `scripts/exh_exit_grid_zkp.py`

## Main table (seen 2023-07 → 2026-11, $5k model slip; E = clean holdout)

| exit | N | /day | WR / BE | avg win / loss | **avg net %** (day-CI) | best-5 % out | @0.20 | Kelly f (floor) | **%/day one size** | max DD | **E** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **shipped** A k3 c5 S1 | 4,015 | 3.25 | 52.5 / 47.4 | +4.78 / −4.31 | **+0.46** (+0.32 … +0.61) | +0.22 | +0.16 | 0.05 | **+0.66** | −62 % | **+0.11** |
| **pick** A k2.5 uncapped, SL 6 % | 3,875 | 3.14 | 50.9 / 45.6 | +7.31 / −6.14 | **+0.71** (+0.47 … +0.95) | +0.00 | +0.40 | 0.05 | **+0.89** | −67 % | **+0.29** |
| B TP 10 %, SL min(1.5 ATR, 6) | 3,998 | 3.23 | 40.2 / 35.3 | +8.52 / −4.64 | +0.65 (+0.43 … +0.86) | +0.16 | +0.31 | 0.04 | +0.84 | −53 % | +0.46 |
| B TP 8 %, SL 6 % | 3,873 | 3.13 | 51.3 / 46.2 | +7.04 / −6.05 | +0.66 (+0.43 … +0.87) | **+0.27** | +0.35 | 0.05 | +0.84 | −61 % | +0.41 |
| B TP 6 %, SL min(1.5 ATR, 6) | 4,003 | 3.24 | 49.7 / 44.5 | +5.70 / −4.56 | +0.54 (+0.37 … +0.70) | +0.25 | +0.24 | 0.05 | +0.75 | −54 % | +0.33 |
| A k3 c5, SL min(1.5 ATR, 6) | 4,009 | 3.24 | 53.8 / 48.7 | +4.77 / −4.53 | +0.48 | +0.24 | +0.18 | 0.05 | +0.68 | −59 % | +0.14 |
| C best trail (a5, b3 %, S2) | 4,004 | 3.24 | 53.8 / 49.6 | +4.60 / −4.52 | +0.39 | −0.15 | −0.01 | 0.03 | +0.52 | −63 % | +0.13 |
| C2 best (a5, b1 ATR, cap 10) | 4,005 | 3.24 | 52.9 / 48.6 | +4.71 / −4.46 | +0.39 | −0.11 | +0.01 | 0.03 | +0.52 | −67 % | +0.15 |

**Per half-year (avg net %/trade):**

| exit | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | **E 23-H1** |
|---|---|---|---|---|---|---|---|---|
| shipped | +0.59 | +0.47 | +0.46 | +0.21 | +0.54 | +0.15 | +0.84 | +0.11 |
| pick A k2.5 uncapped S6 | +0.82 | +0.47 | +0.83 | **−0.20** | +0.88 | +0.80 | +0.97 | +0.29 |
| B TP 10, S2 | +0.83 | +0.66 | +0.75 | +0.17 | +0.76 | +0.54 | +0.79 | +0.46 |
| B TP 8, S 6 % | +0.72 | +0.51 | +0.93 | −0.08 | +0.70 | +0.55 | +1.10 | +0.41 |

**Paired Δ vs shipped** (same candidates, day-block 95 % CI; a robustness read, not pre-registered):

| | pick A k2.5 / SL 6 | B TP 10 / S2 | B TP 8 / SL 6 |
|---|---|---|---|
| all seen | **+0.22 (+0.05 … +0.39)** | +0.19 (+0.03 … +0.33) | +0.19 (+0.04 … +0.34) |
| DESIGN | +0.27 (+0.02 … +0.53) | +0.22 (+0.01 … +0.44) | +0.14 (−0.06 … +0.36) |
| E (clean) | +0.10 (−0.42 … +0.61) | +0.36 (−0.14 … +0.79) | +0.24 (−0.39 … +0.96) |
| 25-H1 (the weak half) | −0.43 (−0.92 … +0.05) | −0.06 | −0.28 |

## Pre-registered gate for the pick (A k2.5 uncapped, SL 6 %, 24 h)

| # | gate | result | |
|---|---|---|---|
| 1 | DESIGN Δ one-size %/day > 0 **and** family-wise p < 0.05 | +0.39 %/day, **p_fw 0.37** | ✗ |
| 2 | Δ > 0 in ≥ 6 of 7 halves and in E | 6/7 (25-H1 −0.43), E +0.18 | ✓ |
| 3 | Δ > 0 at 0.20 flat slip **and** with the best 5 % out | @0.20 +0.40 vs +0.16 ✓ · best-5 % out +0.00 vs +0.22 | ✗ |
| 4 | plateau: ≥ 75 % of neighbours better than shipped on DESIGN | 100 % | ✓ |
| 5 | Δ survives the 50 % haircut | +0.39 → +0.20 %/day | ✓ |

No cell in the whole grid reaches p_fw < 0.05; the best is 0.31. So the rule says **OBSERVE**.

## Family tables (avg net %/trade; full grids in `_tables.md` and `_cells.csv`)

**A: ATR take-profit** (columns = stop; S1 is shipped):

| k × ATR | cap 5 % S1 | cap 5 % S4 | cap 8 % S1 | cap 8 % S4 | uncapped S1 | uncapped S2 | uncapped S4 |
|---|---|---|---|---|---|---|---|
| 1 | +0.32 | +0.34 | +0.32 | +0.34 | +0.31 | +0.33 | +0.34 |
| 1.5 | +0.39 | +0.43 | +0.40 | +0.47 | +0.40 | +0.45 | +0.47 |
| 2 | +0.44 | +0.48 | +0.50 | +0.60 | +0.50 | +0.59 | +0.62 |
| 2.5 | +0.45 | +0.48 | +0.52 | +0.62 | +0.57 | +0.66 | **+0.71** |
| 3 | **+0.46 (shipped)** | +0.49 | +0.55 | +0.64 | +0.54 | +0.66 | +0.69 |

- **Worst half-year turns negative** for k ≥ 2.5 once the cap is 8 % or none (−0.06 to −0.39, in 25-H1).
- With the 5 % cap every k keeps every half positive. k = 2 uncapped keeps every half ≥ +0.13 (S3) to +0.25 (S2).

**B: fixed TP × stop:**

| TP | stop 4 % | **stop 5 %** | **stop 6 %** | min(1.5 ATR, 5) | min(1.5 ATR, 6) |
|---|---|---|---|---|---|
| 4 % | +0.24 | +0.37 | +0.39 | +0.35 | +0.37 |
| 5 % | +0.32 | +0.47 | +0.49 | +0.46 | +0.48 |
| 6 % | +0.36 | +0.53 | +0.56 | +0.51 | +0.54 |
| 7 % | +0.41 | +0.58 | +0.63 | +0.56 | +0.60 |
| 8 % | +0.44 | +0.59 | +0.66 | +0.57 | +0.63 |
| 10 % | +0.42 | +0.59 | **+0.69** | +0.57 | +0.65 |

- A 4 % stop is worse everywhere: −0.13 to −0.17 vs 5 %.
- **E (clean)** with the 5 % stop: TP 5 −0.04 · 6 +0.10 · 7 +0.24 · 8 +0.30 · 10 +0.25. TP ≥ 7 % beats TP 5 % in E as well.

**C: trailing** (no TP cap, shipped stop; rows = arm level, columns = trail distance):

| arm at | 0.5 ATR | 1 % | 1.5 % | 1 ATR | 2 % | 3 % |
|---|---|---|---|---|---|---|
| +2 % | +0.09 | +0.16 | +0.13 | +0.10 | +0.13 | +0.10 |
| +3 % | +0.18 | +0.21 | +0.19 | +0.19 | +0.20 | +0.20 |
| +4 % | +0.26 | +0.27 | +0.25 | +0.26 | +0.27 | +0.25 |
| +5 % | +0.38 | +0.38 | +0.38 | +0.36 | +0.38 | +0.37 |

- The earlier the trail arms, the worse it does. Adding a 10 / 15 % maker cap (C2) or a 6 % stop moves these numbers by −0.01 to +0.04.
- No trailing cell beats the shipped exit on the seen data (best +0.41). All fail best-5 %-out and @0.20.

**D: hold time:** 48 h adds a little to the wide TPs and 12 h costs:
- B TP 10 / S2: 12 h +0.57 · 24 h +0.65 · 48 h +0.71.
- Pick: 12 h +0.59 · 24 h +0.71 · 48 h +0.70.
- The trailing cells hardly change, because they exit early.

## Stop dimension (operator addendum; CLAUDE.md live-stopped rule)
The "live-stopped" set is the 1m replica's stop: paper fills are too few to read live. Δ = wider − narrower stop, on the same candidates. Every fill not stopped at the narrower level shows Δ = 0 (0 exceptions in every comparison and period), so the replica is consistent.

| comparison (same TP) | stopped narrow | **saved** (→ TP) · Σ Δ | **deeper** · Σ Δ | Δ avg %/trade (95 % CI), all seen | DESIGN | E |
|---|---|---|---|---|---|---|
| TP shipped: 5 % → 6 % | 1,780 | 191 (164) · +1,773 % | 1,589 · −1,620 % | **+0.04 (−0.03 … +0.11)** | +0.03 (−0.06 … +0.12) | +0.04 |
| TP shipped: S1 → min(1.5 ATR, 6) | 1,964 | 57 (57) · +564 % | 1,907 · −477 % | +0.02 (−0.01 … +0.06) | +0.03 | +0.03 |
| TP shipped: S1 → 6 % fixed | 1,964 | 375 (305) · +2,915 % | 1,589 · −2,787 % | +0.03 (−0.05 … +0.11) | +0.01 | −0.19 |
| TP 2.5 ATR uncapped: 5 % → 6 % | 2,200 | 217 (160) · +2,560 % | 1,983 · −2,030 % | **+0.13 (+0.03 … +0.22)** | +0.18 (+0.04 … +0.34) | +0.03 |

Per half (shipped TP, 5 → 6 %): +0.26 · −0.02 · −0.06 · −0.03 · +0.06 · +0.03 · +0.06. The sign flips, so this is noise.

**Reading:**
- With a 5 % TP, a wider stop is a wash. About 1 fill in 9 is saved, and the other 8 lose about 1 % more each.
- With a wide TP, each saved fill is worth more (it runs to a bigger TP), and the 6 % stop starts to pay.

## Left on the table (shipped exit, 2,089 TP fills, seen pool)

| further move after the TP fill, within 24 h of entry | p25 | median | p75 | p90 | reached ≥ +10 % in total | price still below the TP at 24 h |
|---|---|---|---|---|---|---|
| price % | +3.7 | **+7.1** | +11.7 | +17.7 | **64.6 %** | 64.6 % (median 24 h P&L +7.5 %) |

Chart: `EXH_EXIT_GRID_2026-11-18_left_on_table.png`.
- The typical TP fill is not the bottom; the coin usually keeps bleeding.
- But the path is choppy: a 1 – 3 % trail rarely survives it (family C). Only a wider resting TP captures the drift.

## ZKPUSDT 2026-11-18 08:30 UTC replayed (1m from fapi, 12 API weight; path up to 12:39 UTC)
- Entry 0.03843, ATR15 3.91 %, so the shipped TP = 5 % (cap) and SL = 5 % (cap).
- Worst move against us: +3.96 % at 09:01. That is inside both the 5 % and the 6 % stop.
- Best: +12.23 % at 12:28.

| exit | result | exit at |
|---|---|---|
| shipped (3 ATR cap 5, SL 5) | +5.00 TP | 09:26 |
| pick (2.5 ATR = 9.77 %, SL 6) | +9.77 TP | 11:10 |
| B TP 10 / S2 | +10.00 TP | 11:44 |
| B TP 8 / SL 6 | +8.00 TP | 10:02 |
| 3 ATR uncapped (11.73 %) | +11.73 TP | 12:07 |
| trail a5 b3 % | +5.74 TRAIL | 10:51 |
| shipped TP with SL 6 % | +5.00 TP | 09:26 (the stop changes nothing here) |

Chart: `EXH_EXIT_GRID_2026-11-18_zkp.png`.

ZKP is one trade, i.e. one observation. It is the kind of trade the wider TPs win on, and it is in no block of this study.

## Charts
- **`_frontier.png` (left):** avg vs worst half-year for all 378 cells.
  - No cell beats the shipped exit by a lot on both axes.
  - The ones that come close are B TP 10 / S1–S2 (+0.57 – 0.65, worst half +0.17) and A k2 uncapped S2 (+0.59, worst +0.25).
- **`_frontier.png` (right):** DESIGN Δ vs E Δ.
  - The wide-TP cells (A, B) are positive on both.
  - The trailing cells are negative on DESIGN; the arm-early ones are mildly positive on E, a quiet half.
- **`_left_on_table.png`:** the move after the TP, and the 24 h P&L of the same fills.

## What it means
- The edge of this sleeve is a **slow bleed after the exhaustion** (see EXHAUSTION_EXPAND: "the state is the edge"). The shipped 5 % cap books the first leg and gives up the median +7 % that follows.
- Wider resting TPs keep the maker fee and catch the bleed. Trailing stops do not: the bleed is choppy, and each shake-out costs taker fee plus slippage.
- **But the extra profit comes from a minority of big runners.** The win rate falls (52 → 40–51 %), and with the best 5 % of trades removed the uncapped / 10 % TPs are no better than shipped.
  - That is the classic shape of a real fat-tail edge, but also of luck.
  - The family-wise test across 378 cells cannot tell the two apart on DESIGN (p 0.37).
- The 6 % stop is not a fix by itself. It only helps when paired with a wider TP.
- **Side finding (sizing, not exits):** at the assumed one size X = 0.5 (≈ $5k notional on $10k), the 3.4-year replay draws down **−62 %** with the shipped exit, and −53 % to −67 % with the alternatives.
  - The EXHAUSTION_EXPAND Kelly-after-daily-clustering size was 0.031×.
  - If X = 0.5 is the real paper size, the size is about 16 × that Kelly. Please check before any live step (CLAUDE.md "size by edge").

## Verdict: **OBSERVE** (pre-registered rule; gates 1 and 3 fail)
- **Do not change the exit yet.** No cell passes the family-wise null, and the pick's gain disappears without its best 5 % of trades.
- **Observe line (frozen now):** the scout shadow-prices every EXHAUSTION_SHORT fill under two exits, from the 1m path after the fill (paired Δ vs the shipped exit, modelled slip):
  - **L1 = the pick:** TP 2.5 × ATR15 uncapped (resting maker), SL 6 % fixed, 24 h.
  - **L2 = B:** TP 10 % fixed (maker), SL min(1.5 × ATR15, 6 %), 24 h. This is the most robust alternative: worst half +0.17, E +0.46, max DD −53 %.
  - **Frozen gate:** after ≥ 40 fills on ≥ 10 distinct days, switch to L2 (or L1) if its cumulative paired Δ > 0 **and** > 0 with the top 5 % of fills removed. A negative paired Δ at 40 fills retires the line. No re-fitting of the TP after the fact.
- **If the operator prefers a declared override now** (the sleeve is paper, so the cost is attribution, not money), these are the exact config numbers. Both pass `exhaustion_config_errors` limits (tp_atr ≤ 20, caps < 50):
  - **L2:** `exhaustion_tp_atr_mult 20`, `exhaustion_tp_cap_pct 10` (= a fixed 10 % for any ATR ≥ 0.5 %) · `exhaustion_sl_atr_mult 1.5`, `exhaustion_sl_cap_pct 6` · `exhaustion_max_hold_hours 24`.
  - **L1:** `exhaustion_tp_atr_mult 2.5`, `exhaustion_tp_cap_pct 49` (effectively uncapped; the 99th pct of 2.5 × ATR is 21 %) · `exhaustion_sl_atr_mult 20`, `exhaustion_sl_cap_pct 6` (= a fixed 6 %, since ATR is never < 0.3 % here).
  - Planning Δ after the 50 % haircut: about +0.10 – 0.12 %/trade, ≈ +0.2 %/day at X = 0.5.
  - Check the exchange backstop gap for the 6 % stop at 5× (the validator does this).
- **KEEP the stop at the shipped min(1.5 ATR, 5 %) as long as the TP stays at 5 %.** The 6 % stop has no edge on its own: +0.04, CI −0.03 … +0.11.
- **Trailing: refuted** for this sleeve (288 cells, none better than shipped, all fail best-5 %-out and @0.20).

## Assumptions and caveats
1. **Seen data:** every block except E was read before for the entry rule. For the exits these are robustness reads, apart from E (2023-H1, clean, 237–248 trades, a slow half where the shipped exit made only +0.11).
2. **The new clean holdout E is weak for the entry rule itself:** +0.11 %/trade (CI −0.40 … +0.63) vs +0.46 on seen data. It is positive, but well below the pooled average. Worth noting for the sleeve.
3. **Size X = 0.5** of equity as notional ($5k on $10k) is my reading of "0.5 inv × 5×" (Flatbase's "≈ 0.45–0.5× notional"). The slip model at $5k is ≈ 0 on these high-volume coins (median 0, max 0.12 %), so the flat 0.10 / 0.20 % columns are the real stress test. Live Frenzy fills had a median slip of +0.06 %. At $1k the averages are the same to 2 decimals.
4. **1m order inside a bar is unknown;** the rule is stop first. The trail updates on the next bar, which is slightly optimistic for trailing (no intrabar re-tightening) and slightly pessimistic where the low came first. Trailing still loses.
5. **Maker TP fills need a 1-tick trade-through.** Queue position is not modelled: a large resting order on a thin coin may not fill at the first touch.
6. **Funding:** real, to each cell's own exit time. Missing months would be charged −0.05 %/8 h; none were missing.
7. **The one-position rule is re-applied per cell,** so N differs a little (3,869 – 4,015). The paired Δ table uses identical candidates.

## Files
- **Report and data:** `reports/EXH_EXIT_GRID_2026-11-18.md`, `_PREREG.md`, `_cells.csv` (396 cells, every column), `_tables.md`, `_frontier.png`, `_left_on_table.png`, `_zkp.png`.
- **Data:** `reports/EXH_EXIT_GRID_data/` (cand_ref.pkl, cand_E.pkl, k1m/, E/, sim_*.pkl, eval_*.pkl, report_grid.pkl, zkp/).
- **Scripts:** `scripts/exh_exit_grid_{data,E_build,sim,eval,report,final,zkp}.py`.
