# TP_GRID: is EXHAUSTION_SHORT's ≈ 10 % take-profit too ambitious? (2026-11-20)

Research only. No bot code, config or test changed. Committed with DECISION_LOG 603.

## Question
EXHAUSTION_SHORT exits at:
- TP = min(20 × ATR15, 10 %), a resting maker order (≈ 10 % in practice);
- SL = min(1.5 × ATR15, 6 %);
- a 24 h time exit.

The operator asks: is 10 % too ambitious? Would a 6, 7 or 8 % TP raise the win rate and improve results? What is the best TP level?

## Verdict: **KEEP the 10 % TP.**
- **Every lower TP raises the win rate, but none raises the result.** The breakeven win rate rises just as fast, because each win is smaller.
- **TP ≤ 6 % is clearly worse.**
  - TP 6 %: −0.15 %/trade (CI −0.27 … −0.04). TP 5 %: −0.21. TP 4 %: −0.32.
  - These are negative in 6 or 7 of the 7 half-years.
  - The family-wise test says they are worse than today's exit beyond luck (p 0.00 – 0.04).
- **TP 7 / 8 / 9 % are a statistical tie with 10 %.**
  - Δ −0.06 / −0.03 / −0.03 %/trade, and every CI includes 0.
  - No TP level beats 10 % on all data, on DESIGN or on HOLDOUT.
  - The best cell of the whole grid is min(5 × ATR, 10 %). It is +0.01 %/trade, family-wise p 0.27, so luck.
- **The 5 real fills agree.** 4 of the 5 reached +10 %, so any lower TP only cut them. The fifth was a stop 7 minutes after entry, which no TP could save. The win rate is 80 % at every TP level, and the sum falls from +30.2 % (TP 10) to +22.8 % (TP 8) and +16.7 % (TP 6). N = 5, so this is illustrative only.
- **The one honest argument for 8 % is risk, not return.**
  - Same return: daily compound +0.97 %/day at both 8 % and 10 %.
  - Smaller worst day: −27 % vs −40 %, at one size 0.61.
  - Smaller max drawdown: −56 % vs −72 %.
  - But this comes from two stop-cluster days (2025-11-22 and 2026-01-24). Those are 2 observations, and the ~16× ¼-Kelly size is the real driver of the drawdown, not the TP.
- **Recency watch (not a ship):** in 2026-H2 alone, TP 7 / 8 beat 10 % by +0.19 %/trade. In the 6 earlier half-years they lose or tie (2 of 7 halves are positive for TP 8). On the WIDE universe in 2026, TP 8 is worse: −0.18 (CI −0.35 … 0.00). That is one half against six, so it is not evidence.

## Method
**Step 1: the real fills.**
- The 5 closed EXHAUSTION_SHORT fills of the current batch export `exports_archive/scalpars_orders_paper_2026-11-20_16-01-39.csv`: ZKP, ZKC, 1000LUNC, YB, KERNEL. None is a probe. The master pool holds 0 EXH fills.
- Data: 1m futures klines and the funding history from the Binance public futures API.
  - 5 + 5 calls, peak used weight 50 / min; the script stops on HTTP 418 / 429.
  - Cached in `reports/TP_GRID_2026-11-20_EXH_data/real/`.
- Each fill is walked from its real fill price for 24 h:
  - the stop is checked first on the bar high, and fills at max(stop, bar open);
  - the maker TP fills on the bar low only when the price trades ≥ 1 tick through it (stress: ≥ 0.05 % through);
  - a bar that touches both counts as a stop.
- Costs:
  - entry taker 0.05 %;
  - TP maker 0.02 %;
  - stop or time exit: taker 0.05 % + 0.10 % slip;
  - real funding settlements up to each cell's own exit.
- Every cell uses today's stop, min(1.5 × ATR, 6 %).

**Step 2: the backtest.**
- Engine: the EXH_EXIT2 / NOSPOT_EXIT 1m engine, with the same fill rules as step 1.
  - Costs: slip = max(0.10 %, the size model at 0.61 × $10k) on each taker leg, entry included; stress at a flat 0.20 %.
  - The script is `scripts/TP_GRID_2026-11-20_EXH_sim.py`, which re-uses `nospot_exit_sim.run_rows`. Only the TP moves; the stop and the 24 h hold are today's.
- **Parity:** today's exit reproduces the baseline exactly.
  - FRENZY: 3,368 / 3,368 same exit kind vs both EXH_EXIT2 and BASELINE_TRADES_EXHAUSTION; max |Δ gross| 5e-7.
  - WIDE: 99.9 % same exit kind vs the L4 file.
- **Cohort = today's EXH stack, in two universes:**
  - **FRENZY**: the 3,368 kept fills of `BASELINE_TRADES_EXHAUSTION_2026-11-18.csv` (dl582_block False). This is the FRENZY universe at ≥ $20M with a spot market, the RSI cross, vol_decay and spot-share rules, the DL 582 EXH_SSC_OIVOL filter and one position per pair.
  - **WIDE** (DL 585): 939 L4 ADDED fills kept by DL 582. This is the joint_short_sim cohort; 2,130 missing 1m pair-days came from data.binance.vision.
  - **TODAY** = FRENZY ∪ WIDE, 4,307 fills, 2023-07 → 2026-11.
  - **E** = the clean 2023-H1 block: 248 candidates → 24 blocked by DL 582 → 215 after one position per pair.
- **Gaps vs the live sleeve (stated, small):**
  1. The study applied one position per pair *before* the filters; live applies it after (DL 578), so the live cohort is a superset.
  2. The L4 ADDED set includes FRENZY episodes at $5–20 M, which live blocks (DL 585).
  3. The FRENZY ∪ WIDE pooling does not apply one position per pair across the two universes.
  4. A lower TP frees a coin sooner, so it could take a later signal. This is not modelled; it would add a few trades to the low-TP cells.
  5. The depth guard / capacity shrink (DL 583 / 584) is not modelled.
  6. The DL 589 spot map changes nothing: the backtest always counted those coins as spot.
- **Cells (21):**
  - REF = min(20 ATR, 10);
  - fixed TP 4 – 9 %;
  - min(k × ATR, X) for k ∈ {1.5, 2, 3, 5} and X ∈ {6, 8, 10};
  - queue stress (TP must trade 0.05 % through) on 6 %, 8 % and REF.
- **Stats:**
  - The paired Δ vs today uses identical entries, with a day-block bootstrap 95 % CI (10-day blocks).
  - DCR = one size 0.61 × equity per trade, every calendar day of the block, 0 on no-fill days, P&L booked on the exit day.
  - Max drawdown and worst day are taken on that one-size equity path.
  - Family-wise null: a day-level sign flip of the paired Δ, with the same flips across all 21 cells, 4,000 draws, max-T (better) and min-T (worse).
- **Splits:** DESIGN 2023-07 → 2025-06 · HOLDOUT 2025-07 → 2026-11-18 · 2026 · each half-year · E. Every block except E was already seen by the earlier EXH studies; E is clean.
- **Scripts:**
  - `scripts/TP_GRID_2026-11-20_EXH_real.py`
  - `scripts/TP_GRID_2026-11-20_EXH_sim.py` (steps: rows, k1m, sim)
  - `scripts/TP_GRID_2026-11-20_EXH_eval.py`

## Step 1: the real fills (current batch, N = 5, ILLUSTRATIVE)

**Parity: each booked exit re-walked with its own stamped TP / SL.**

| pair | opened (UTC) | booked TP / SL | booked → model kind | booked → model exit time | booked net % | model net % | note |
|---|---|---|---|---|---|---|---|
| ZKP | 11-18 08:30 | 5 / 5 (pre-DL 579) | TP → TP | 09:25 → 09:26 | +4.94 | +4.93 | |
| ZKC | 11-19 00:30 | 10 / 6 | TP → TP | 05:39 → 05:40 | +6.83 | +6.64 | funding −3.12 booked / −3.29 model (hourly −0.4 … −0.8 %) |
| 1000LUNC | 11-19 07:00 | 10 / 3.82 | TP → TP | 07:46 → 07:47 | +9.96 | +9.93 | |
| YB | 11-19 16:15 | 10 / 6 | SL → SL | 16:22 → 16:23 | −6.58 | −6.20 | real stop slip 0.46 % vs 0.10 modelled |
| KERNEL | 11-20 14:15 | 10 / 6 | TP → TP | 15:45 → 15:46 | +9.95 | +9.93 | |

- All 5 reproduce the booked exit kind and price.
- The net differences have known causes:
  - booked fees are 0.045 / 0.018 % against the model's 0.05 / 0.02 %;
  - YB's real stop slip was 0.46 %;
  - the ZKC funding difference is one partial hourly settlement.

**The TP grid on the same 5 fills** (today's stop min(1.5 ATR, 6) for every cell; calendar days 11-18 → 11-20):

| TP | N | WR | avg net % | Σ % | DCR %/day (0.61) | max DD | ZKP | ZKC | LUNC | YB | KERNEL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 % | 5 | 80 % | +1.90 | +9.5 | +1.93 | 0 | +3.93 | +3.93 | +3.93 | −6.20 | +3.93 |
| 5 % | 5 | 80 % | +2.54 | +12.7 | +2.58 | 0 | +4.93 | +4.10 | +4.93 | −6.20 | +4.93 |
| 6 % | 5 | 80 % | +3.34 | +16.7 | +3.39 | 0 | +5.93 | +5.10 | +5.93 | −6.20 | +5.93 |
| 7 % | 5 | 80 % | +4.14 | +20.7 | +4.21 | 0 | +6.93 | +6.10 | +6.93 | −6.20 | +6.93 |
| 8 % | 5 | 80 % | +4.55 | +22.8 | +4.63 | 0 | +7.93 | +5.17 | +7.93 | −6.20 | +7.93 |
| 9 % | 5 | 80 % | +5.25 | +26.2 | +5.33 | 0 | +8.93 | +5.64 | +8.93 | −6.20 | +8.93 |
| **10 % (today)** | 5 | 80 % | **+6.05** | **+30.2** | **+6.15** | 0 | +9.93 | +6.64 | +9.93 | −6.20 | +9.93 |
| min(3 ATR, 10) | 5 | 80 % | +5.57 | +27.9 | +5.66 | 0 | | | | | |
| min(2 ATR, 10) | 5 | 80 % | +4.32 | +21.6 | +4.38 | 0 | | | | | |

- In 24 h the five coins fell a further +11.5 to +20.6 % (MFE) after entry. YB also went +6.9 % against us within 7 minutes, so a lower TP cannot rescue it.
- The 0.05 %-through fill stress changes nothing here.
- ZKC is not monotonic in the TP because of hourly funding: it pays about −0.8 %/h, so the exit time matters.

## Step 2: the backtest, TODAY = FRENZY ∪ WIDE (4,307 fills)

**ALL 2023-07 → 2026-11** (Δ = paired vs today, 95 % day-block CI; DCR at one size 0.61):

| TP | N | WR | BE WR | avg win / loss | **avg net %** | Σ % | Δ (CI) | Δ @0.20 slip | DCR %/day | max DD | worst day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **10 % (today)** | 4,307 | 41.1 | 36.3 | +8.26 / −4.71 | **+0.62** | +2,668 | — | — | **+0.97** | −72 % | −40.5 % |
| 9 % | 4,307 | 42.6 | 37.8 | +7.71 / −4.69 | +0.59 | +2,543 | −0.03 (−0.08 … +0.03) | −0.03 | +0.94 | −63 % | −40.7 % |
| 8 % | 4,307 | 44.7 | 39.7 | +7.11 / −4.68 | +0.59 | +2,539 | −0.03 (−0.11 … +0.04) | −0.02 | +0.97 | −56 % | −27.2 % |
| 7 % | 4,307 | 47.3 | 42.2 | +6.36 / −4.65 | +0.56 | +2,396 | −0.06 (−0.16 … +0.03) | −0.05 | +0.95 | −58 % | −28.4 % |
| 6 % | 4,307 | 50.0 | 45.4 | +5.55 / −4.63 | +0.47 | +2,005 | **−0.15 (−0.27 … −0.04)** | −0.14 | +0.79 | −59 % | −29.6 % |
| 5 % | 4,307 | 54.1 | 49.6 | +4.66 / −4.59 | +0.41 | +1,767 | **−0.21 (−0.35 … −0.07)** | −0.19 | +0.72 | −64 % | −24.5 % |
| 4 % | 4,307 | 58.6 | 55.0 | +3.74 / −4.57 | +0.29 | +1,269 | **−0.32 (−0.49 … −0.18)** | −0.30 | +0.51 | −65 % | −25.5 % |
| min(5 ATR, 10) | 4,307 | 41.2 | 36.4 | +8.25 / −4.71 | +0.63 | +2,722 | +0.01 (−0.00 … +0.03) | +0.01 | +0.99 | −72 % | −40.9 % |
| min(3 ATR, 10) | 4,307 | 43.7 | 38.7 | +7.53 / −4.76 | +0.60 | +2,603 | −0.02 (−0.08 … +0.05) | −0.01 | +0.97 | −64 % | −35.3 % |
| min(3 ATR, 8) | 4,307 | 45.8 | 41.0 | +6.77 / −4.71 | +0.55 | +2,378 | −0.07 (−0.15 … +0.01) | −0.06 | +0.91 | −60 % | −28.6 % |
| min(2 ATR, 10) | 4,307 | 50.1 | 45.4 | +5.79 / −4.82 | +0.50 | +2,144 | −0.12 (−0.22 … −0.02) | −0.10 | +0.83 | −62 % | −34.2 % |
| min(1.5 ATR, 10) | 4,307 | 55.8 | 52.0 | +4.44 / −4.82 | +0.35 | +1,512 | −0.27 (−0.40 … −0.14) | −0.24 | +0.59 | −64 % | −27.7 % |
| 8 %, TP 0.05 % through | 4,307 | 44.6 | 39.7 | | +0.57 | +2,466 | −0.05 (−0.12 … +0.03) | | +0.94 | −59 % | −27.2 % |
| 10 %, TP 0.05 % through | 4,307 | 41.1 | 36.4 | | +0.61 | +2,623 | −0.01 (−0.02 … −0.00) | | +0.94 | −72 % | −40.5 % |

**WR vs average, the whole story in one line:** each 1 % lower TP adds about 2–4 points of win rate, but the breakeven win rate rises by the same amount.
- The margin over breakeven stays ≈ +4.7 points from TP 10 down to TP 6 (+4.8 / +5.0 / +5.0 / +4.6).
- It shrinks at 5 % (+4.4) and 4 % (+3.6).
- So the win rate goes up without the edge going up, and each trade's average falls.

**Per block (avg net %/trade · paired Δ vs today · DCR %/day):**

| TP | DESIGN 23-07 → 25-06 (1,683) | HOLDOUT 25-07 → 26-11 (2,624) | 2026 (1,673) | E 2023-H1 clean (215) |
|---|---|---|---|---|
| **10 % (today)** | +0.47 · — · 0.47 | +0.71 · — · 1.69 | +0.73 · — · 1.73 | +0.36 · — · 0.18 |
| 9 % | +0.43 · −0.04 (−0.11 … +0.04) · 0.42 | +0.69 · −0.02 (−0.09 … +0.06) · 1.69 | +0.69 · −0.04 · 1.68 | +0.47 · +0.11 (−0.05 … +0.34) · 0.26 |
| 8 % | +0.43 · −0.05 (−0.14 … +0.06) · 0.43 | +0.69 · −0.02 (−0.12 … +0.09) · 1.76 | +0.74 · +0.01 (−0.14 … +0.16) · 1.90 | +0.48 · +0.12 (−0.11 … +0.42) · 0.28 |
| 7 % | +0.37 · −0.10 (−0.23 … +0.03) · 0.37 | +0.68 · −0.04 (−0.19 … +0.10) · 1.79 | +0.73 · −0.00 (−0.19 … +0.19) · 1.95 | +0.42 · +0.06 (−0.26 … +0.47) · 0.24 |
| 6 % | +0.30 · −0.18 (−0.33 … −0.05) · 0.29 | +0.57 · −0.14 (−0.30 … +0.04) · 1.53 | +0.61 · −0.11 · 1.64 | +0.23 · −0.13 (−0.47 … +0.25) · 0.11 |
| 5 % | +0.28 · −0.20 · 0.30 | +0.50 · −0.22 · 1.33 | +0.49 · −0.23 · 1.31 | +0.04 · −0.31 · −0.01 |
| 4 % | +0.17 · −0.31 · 0.17 | +0.38 · −0.34 · 1.01 | +0.36 · −0.37 · 0.96 | +0.14 · −0.22 · 0.06 |

**Per half-year, paired Δ vs today (%/trade):**

| TP | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | halves > 0 |
|---|---|---|---|---|---|---|---|---|
| 9 % | −0.10 | −0.06 | +0.03 | −0.05 | +0.01 | −0.11 | +0.06 | 3 / 7 |
| 8 % | −0.06 | −0.02 | +0.03 | −0.14 | −0.07 | −0.13 | **+0.19** | 2 / 7 |
| 7 % | 0.00 | −0.22 | +0.04 | −0.18 | −0.11 | −0.14 | **+0.19** | 2 / 7 |
| 6 % | −0.19 | −0.22 | −0.10 | −0.19 | −0.18 | −0.30 | +0.13 | 1 / 7 |
| 5 % | −0.24 | −0.28 | −0.22 | −0.06 | −0.19 | −0.42 | +0.01 | 1 / 7 |
| 4 % | −0.44 | −0.39 | −0.32 | −0.11 | −0.29 | −0.51 | −0.18 | 0 / 7 |
| min(5 ATR, 10) | +0.01 | +0.08 | +0.03 | −0.01 | +0.02 | 0.00 | −0.01 | 4 / 7 |

**Each universe on its own (ALL · 2026, avg net % / Δ):**

| TP | FRENZY ALL (3,368) | FRENZY 2026 (1,263) | WIDE ALL (939) | WIDE 2026 (410) |
|---|---|---|---|---|
| **10 %** | +0.68 / — | +0.67 / — | +0.40 / — | +0.89 / — |
| 9 % | +0.65 / −0.03 | +0.65 / −0.02 | +0.38 / −0.01 | +0.80 / −0.09 |
| 8 % | +0.65 / −0.03 | +0.74 / +0.07 (−0.10 … +0.24) | +0.36 / −0.04 | +0.71 / −0.18 (−0.35 … 0.00) |
| 7 % | +0.61 / −0.07 | +0.75 / +0.08 (−0.16 … +0.28) | +0.37 / −0.03 | +0.66 / −0.24 |
| 6 % | +0.52 / −0.16 | +0.62 / −0.05 | +0.28 / −0.12 | +0.58 / −0.32 |

## Who gains, who pays (TODAY, 4,307 fills, Σ paired Δ in %-points; $ at $6.1k notional)

| TP | REF +10 % winners cut to X (pays) | REF stopped → hits X first (gains) | REF 24 h time exit → hits X (gains) | stopped / timed out in both | **net Σ Δ** |
|---|---|---|---|---|---|
| 6 % | 1,324 fills · −5,121 (−$312k) | 362 · +4,024 (+$245k) | 383 · +435 (+$27k) | 2,238 · 0 | **−662** |
| 7 % | 1,324 · −3,836 (−$234k) | 248 · +3,016 (+$184k) | 314 · +548 (+$33k) | 2,421 · 0 | **−272** |
| 8 % | 1,324 · −2,549 (−$155k) | 143 · +1,880 (+$115k) | 220 · +539 (+$33k) | 2,620 · 0 | **−129** |
| 9 % | 1,324 · −1,275 (−$78k) | 58 · +835 (+$51k) | 107 · +315 (+$19k) | 2,818 · 0 | **−125** |

- **Why the low TPs lose:** 1,324 fills (31 %) reach +10 %, and every one of them pays (10 − X) %.
  - Their median 24 h MFE is +15 %. These coins usually keep falling past 10 %; they do not stall at 6–8 %.
- **The stops a lower TP could rescue are few.** Of the 2,494 stopped fills, only 15 % went +6 % our way before the stop, 10 % went +7 % and 6 % went +8 %.
  - The 85 % that never reached +6 % lose the same under every TP.
- **The trade-off at 6 %:** it rescues 362 stops (each worth about +11 %, from −5 % to +6 %), but cuts 1,324 winners by 4 % each. The cuts win.

## Luck and robustness checks
- **Family-wise null (TODAY, 21 cells, max-T):**
  - The best cell is min(5 ATR, 10), t +1.62, **p_fw 0.27** (DESIGN 0.22). No level beats today's exit beyond luck.
  - The worse-than-today side is significant for TP 4 / 5 / 6 (p 0.000 / 0.015 / 0.042) and for every k ≤ 2 ATR cell.
  - TP 7 / 8 / 9: p 0.52 / 0.77 / 0.60, i.e. not distinguishable from 10 % in either direction.
- **Best-5 %-out** (each cell's own average without its best 5 % of trades): 10 % +0.13 · 9 % +0.15 · 8 % +0.21 · 7 % +0.22 · 6 % +0.18.
  - ⚠ This favours low TPs by construction. A capped cell's best trades are its +X % TPs, so removing them costs a low TP less. Here it is not a fat-tail test: no cell can book more than its TP.
  - The relevant luck read is the paired CI and p_fw above.
- **Per-pair concentration:** no pair carries the Δ. The top pair is ≤ 27 % of |Σ Δ| for TP 8 (1000FLOKI, same sign). Leave-top-pair-out Δ: TP 8 −0.04 · TP 7 −0.07 · TP 6 −0.16 (unchanged).
- **Live costs:**
  - 0.20 % stop slip: the Δ moves by ≤ 0.02.
  - TP queue (must trade 0.05 % through): today −0.01, TP 8 −0.02. Wider TPs are a little more exposed to queue misses, but by too little to change the order.
  - Funding: a shorter hold pays less. This is in the nets (avg hold 6.8 h at TP 10 vs 5.2 h at 8 % and 3.5 h at 6 %).
- **Kelly at the Wilson floor (per trade):** 10 % 0.056 · 8 % 0.062 · 7 % 0.066 · 6 % 0.061. These are all ≈ equal: the TP does not change the edge per unit of risk.
- **In-sample haircut:** not needed. No lower TP has a positive Δ to haircut on the pooled data.

## What it means
- A higher win rate is not the goal; expectancy is.
  - At TP 6 % the sleeve wins 1 trade in 2 instead of 2 in 5, but it banks +5.6 per win instead of +8.3, and still loses about −4.6 on each stop.
  - The edge margin over breakeven stays the same, ≈ +4.7 points. So the extra wins are bought, not earned.
- The sleeve's edge is the slow bleed after the exhaustion signal. The coins that reach +10 % usually go on to +15 %, so the 10 % resting maker order is already a conservative capture.
- The stops do not fail because the TP is too far. 85 % of them never went +6 % our way: the entry was wrong, not the exit.
- **Where 8 % is defensible:** it ties 10 % on return (DCR 0.97 vs 0.97) and cut the worst day from −40 % to −27 % (max drawdown −72 → −56 %) at the 0.61 size.
  - That is a risk-smoothing choice, built on two stop-cluster days. It is not an edge.
  - Size is the right lever for that risk: the EXH size is ≈ 16 × ¼-Kelly (DL 578 / EXH_EXIT_GRID).
- **Optimal TP, with honest confidence:**
  - **10 % (today), or anything from 8 to 10 % (a statistical tie).**
  - **Not 6 % or lower.** That is refuted at about 95 % family-wise confidence on the pooled data, and wrong in 6–7 of 7 half-years.
  - The 2026-H2 tilt toward 7–8 % (+0.19) is one half-year, and the WIDE 2026 block contradicts it. If wanted, it could be registered as an observe line on the live fills (the scout already re-prices fills): shadow TP 8 % vs 10 %, read at ≥ 40 fills over ≥ 10 days. It is not a change now.

## Equal-drawdown read (added 2026-11-20, scripts/TP_GRID_2026-11-20_EXH_risk.py)

Method (operator 2026-11-20 "what is safer"): each cell is sized so ITS backtest max drawdown is −30 %, then its daily compound return at that size is compared with today's exit; 10-day-block bootstrap (500 resamples, size re-solved in each), paired by day. HOLDOUT = entries ≥ 2025-07-01. Caveats: the drawdown comes from one history (bootstrapped) and assumes every fill at full size with NO shared-book slot cap — the cluster days that set it are partly out of reach for the live book; P(better) is not a p-value and the cells were not pre-registered (best-of-few).

| block | TP | size for −30 % DD | daily compound at that size | worst day | P(better than today) | gain vs today (90 % range) |
|---|---|---|---|---|---|---|
| ALL | today (10 %) | 0.19 × | +0.38 %/day | -12.8 % | — | — |
| ALL | TP 9 % | 0.24 × | +0.44 %/day | -15.8 % | 0.56 | +0.01 (-0.07 … +0.10) |
| ALL | TP 8 % | 0.28 × | +0.53 %/day | -12.7 % | 0.82 | +0.06 (-0.06 … +0.19) |
| ALL | TP 7 % | 0.27 × | +0.47 %/day | -12.5 % | 0.75 | +0.07 (-0.11 … +0.25) |
| ALL | TP 6 % | 0.28 × | +0.41 %/day | -13.4 % | 0.46 | -0.01 (-0.16 … +0.15) |
| ALL | TP 5 % | 0.25 × | +0.33 %/day | -9.9 % | 0.51 | +0.00 (-0.18 … +0.20) |
| HOLDOUT | today (10 %) | 0.19 × | +0.66 %/day | -12.8 % | — | — |
| HOLDOUT | TP 9 % | 0.24 × | +0.78 %/day | -15.8 % | 0.68 | +0.06 (-0.11 … +0.27) |
| HOLDOUT | TP 8 % | 0.30 × | +0.97 %/day | -13.2 % | 0.86 | +0.16 (-0.09 … +0.47) |
| HOLDOUT | TP 7 % | 0.36 × | +1.15 %/day | -16.8 % | 0.91 | +0.30 (-0.05 … +0.75) |
| HOLDOUT | TP 6 % | 0.29 × | +0.81 %/day | -12.4 % | 0.75 | +0.13 (-0.17 … +0.56) |
| HOLDOUT | TP 5 % | 0.25 × | +0.60 %/day | -9.9 % | 0.69 | +0.11 (-0.35 … +0.60) |

Read: at equal drawdown TP 7–8 lean better than 10 % (P 0.75–0.91), all ranges include 0; TP 7 and 8 cannot be told apart. The gain comes from the larger size their smaller drawdown allows. Decide TP with the EXH size at the batch review.

## Files
- **Report:** `reports/TP_GRID_2026-11-20_EXH.md`.
- **Tables:**
  - `reports/TP_GRID_2026-11-20_EXH_grid.csv` (every cohort × block × cell, all columns)
  - `_null.csv`
  - `_decomp.csv`
  - `_E.csv`
  - `_real.csv`
  - `_real_parity.csv`
  - `_real_summary.csv`
- **Data:** `reports/TP_GRID_2026-11-20_EXH_data/`
  - `real/` (1m klines + funding of the 5 fills)
  - `k1m/` (2,130 WIDE pair-days)
  - `rows_*.pkl`
  - `sim_*.npz`
  - `eval.pkl`
  - `eval.log`
  - Two empty cache files from a first run with a timestamp bug can be deleted: `real/ZKPUSDT_1794960000_1m.csv` and `real/ZKPUSDT_1794990611_funding.json`.
- **Scripts:** `scripts/TP_GRID_2026-11-20_EXH_{real,sim,eval}.py`.
