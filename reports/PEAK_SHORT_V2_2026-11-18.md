# PEAK SHORT V2: double top, first-peak failure, wait-and-see, funding, all coins (2026-11-18)

**Pre-registered:** `PEAK_SHORT_V2_2026-11-18_PREREG.md`, frozen 13:58 UTC, before any result. The never-read block H0 (2023-H1) was read once, after the cell table was frozen (`_data/cells_frozen_before_H0.csv`).
Research only. Nothing in the bot changed. Nothing was committed.

Files: `_cells.csv` (40 cells, every metric) · charts `_dropleft.png`, `_halves.png`, `_event.png`, `_replay_<COIN>.png` · data `_data/` (trades.pkl, desc.csv, freq.csv, replay/trades.csv) · scripts `scripts/peak_short_v2_{build,eval,charts,replay,fetch_h0,fetch_replay}.py`.

## Question
Operator: "Deep dive into the double-peak theory … double peak, or even the first peak. We are still missing the sleeve that gives us many trades per day."
So: is there a short on pumped coins, at the first or second peak, that trades **8–10 times a day** and makes money **after all costs**, consistently?

## Plain answer
**No. Nothing passes. The frequency is there, the edge is not.**
- **Frequency is easy.** On every USDT perp, pumps (+15 % in ≤ 6 h on ≥ 10× volume) give **7.9 episodes a day**. The survival rules fire **8–14 times a day** (16–18/day in 2026).
- **Every high-frequency cell loses money.** The best 8+/day cell (S4X5 · X3) makes **−0.11 %/trade**. The operator-style "short 1 h after the top, 5 % below it" (S1X5 · X1) makes **−0.31 %** at 12/day.
- **The best cell overall is the 1 h RSI divergence (DV1h)**: −0.01 %/trade (X1) and +0.05 % (X2), at **3/day**. It is positive in 2025-H2 → now, but negative before. It loses on the never-read 2023-H1 block (−0.35 / −1.18).
- **Every cell beats random timing** (random shorts in the same state lose −0.4 to −1.0 %/trade). So these rules pick better moments than random, again. But "better than random" is still below zero.
- **Waiting does not fix squeezers.** Waiting 8 h after the top cuts the squeezer share only from 46 % to 35 %. By then, 10 % of the drop is gone. The mean 24 h short return stays at about 0 for every wait time.
- **The true double top is rare and bad**: 1.4–2.6/day, −0.22 to −0.64 %/trade.
- **Funding filter does not help.** Skipping coins already on 4 h / 1 h funding removes about 70 % of the trades and does not improve the rest.

**Verdict: NO** for all four ideas. No observe line meets the prereg bar. Details and the one honest lead below.

## Theory (written before the test)
1. Most pumps top at the first peak. The losses come from squeezers. Squeezers show themselves in the first ~2 h. So **wait W hours** after the top and short only if it held. Cost: part of the drop is gone.
2. A **true double top** (second high within ±3 % of the first, then a close under the neckline) means buyers failed twice at the same price. The stop sits above a level that held twice.
3. A **lower high on weaker RSI** (bearish divergence) means the buying is fading.
4. **Funding** costs a lot on these coins. Skip or exit around it.
5. Only the **whole market** (not just Frenzy coins) can give 8–10 trades a day.

## Method (short)
**Universe U-A (primary).**
- Every USDT perp in the 5m cache (658 pairs; BTC/ETH and blacklist out). 2023-07-01 → 2026-11-17.
- Pump trigger: close ≥ +15 % above the 6 h low ∧ 1 h volume ≥ 10× normal (engine `normal_hour_usd`) ∧ 24 h volume ≥ $20 M.
- Episode = first trigger → 24 h after the last trigger, capped at 72 h. **9,753 episodes on 638 pairs = 7.9/day.**
- **U-F** (subset) = U-A trades where the engine Frenzy flag was ON at the signal. 69 % of episodes touch a Frenzy flag.

**Entries (9) × exits (4) = 36 cells, + 4 funding cells = 40 cells.**
(The prereg said "10 entries, 44 cells". That was an arithmetic slip: 6 + 1 + 2 = 9 entries. The rules are exactly as frozen.)
- **S{W}X{X} survival** (6): the episode high has held W ∈ {1, 2, 4} h, and the 15m close is ≥ X ∈ {5, 10} % under it. Once per high.
- **DT double top** (1): 8 % pullback from P1; second top within ±3 %; first 15m close under the neckline. Stop above the higher top.
- **DV15 / DV1h divergence** (2): pivot high = episode high (P1), then a lower pivot high with RSI14 ≥ 5 points lower.
- **Exits:** X1 TP 5 % / stop peak + 0.5 % (skip if > 15 %) / 24 h · X2 TP 10 % / peak stop / 48 h · X3 TP 5 % / stop 12 % / 24 h · X4 TP 3 % / stop 8 % / 12 h.
- **F4 funding** on the two best DESIGN cells (both DV1h): F4a skip if funding interval < 8 h or last rate ≤ −0.10 %; F4b also exit 5 min before a settlement whose last rate ≤ −0.10 %.

**Realism.** Closed 15m bars; entry next 5m open. Taker 0.05 %, maker 0.02 %, slippage 0.10 % per taker leg (0.05 / 0.20 also). Maker TP fills only if price trades 1 tick through. Gap-aware stops; stop wins a same-bar tie. Real funding at every settlement (hourly when switched). One position per pair per cell.

**Nulls.** 20 random entry bars per trade, same episode, gate on, close ≥ 5 % under the episode high ("same state"), same exit. Family-wise max-z over all 40 cells.

## Pass bar — the 8 cells that matter (all 40 in `_cells.csv`)
Net %/trade at 0.10 % slippage. ① every half > 0 · ② best 5 % out > 0 · ③ @0.20 > 0 · ④ beats random (p) · ⑤ family-wise · ⑥ Kelly at Wilson floor (daily) · ⑦ holdouts A · B · ⑧ H0 never-read.

| cell | N | /day | WR · BE WR | avg | 95 % CI (days) | ② | ③ | ④ p (random avg) | ⑤ | ⑥ | ⑦ A · B | ⑧ H0 | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1X5 · X1 ("1 h after the top, 5 % under") | 14,785 | 12.0 | 60 · 62 % | **−0.31** | −0.53 … −0.12 | −0.59 | −0.46 | ✓ 0.000 (−0.97) | ✓ | ✗ −0.13 | ✗ −0.70 · −0.21 | ✗ −0.38 | 2/8 |
| S2X5 · X3 | 14,111 | 11.4 | 68 · 69 % | −0.20 | −0.48 … +0.07 | −0.47 | −0.34 | ✓ 0.000 (−0.88) | ✓ | ✗ −0.12 | ✗ −0.71 · +0.01 | ✗ −0.39 | 2/8 |
| S4X5 · X3 (best ≥ 8/day) | 12,541 | 10.2 | 67 · 68 % | **−0.11** | −0.33 … +0.11 | −0.38 | −0.25 | ✓ 0.000 (−0.70) | ✓ | ✗ −0.10 | ✗ −0.44 · +0.00 | ✗ −0.46 | 2/8 |
| DV15 · X3 | 12,635 | 10.2 | 68 · 69 % | −0.15 | −0.35 … +0.03 | −0.42 | −0.29 | ✓ 0.000 (−0.85) | ✓ | ✗ −0.11 | ✗ −0.33 · −0.14 | ✗ −0.22 | 2/8 |
| DT · X4 (best double top) | 3,159 | 2.6 | 70 · 72 % | −0.22 | −0.52 … +0.02 | −0.38 | −0.35 | ✗ 0.73 (−0.18) | ✗ | ✗ −0.18 | ✗ +0.01 · −0.29 | ✗ −0.52 | 0/8 |
| **DV1h · X1** (best avg ≈ 0) | 3,715 | 3.0 | 60 · 61 % | **−0.01** | −0.24 … +0.22 | −0.27 | −0.16 | ✓ 0.000 (−0.40) | ✓ | ✗ −0.07 | ✗ −0.26 · +0.24 | ✗ −0.35 | 2/8 |
| DV1h · X2 | 3,705 | 3.0 | 50 · 50 % | +0.05 | −0.30 … +0.40 | −0.47 | −0.11 | ✓ 0.000 (−0.60) | ✓ | ✗ −0.04 | ✗ −0.27 · +0.39 | ✗ −1.18 | 2/8 |
| DV1h · X1 · F4a (funding skip) | 1,174 | 1.0 | 58 · 58 % | +0.01 | −0.42 … +0.42 | −0.25 | −0.14 | ✓ 0.000 (−0.42) | ✓ | ✗ −0.08 | ✗ −0.12 · +0.88 | ✗ −0.40 | 2/8 |

**Across all 40 cells:**
- 3 of 40 have avg > 0 (DV1h · X2 +0.05, DV1h · X1 · F4a / F4b +0.01). **No cell passes more than 2 of 8.**
- 22 of 40 beat the random null. Not one passes ①, ②, ③, ⑥ or ⑧.
- Win rate sits 1–3 points **under** the break-even win rate in almost every cell. The wins are capped (TP 3–10 %); the losses are not (squeezes gap through stops: worst days −15 % to −240 % of equity at one size, see below).

![per half-year](PEAK_SHORT_V2_2026-11-18_halves.png)

**Per half-year** (avg net %, n in `_cells.csv`):

| cell | H0 23-H1 | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 |
|---|---|---|---|---|---|---|---|---|
| S1X5 · X1 | −0.38 | −0.27 | −0.90 | +0.13 | −0.52 | −0.20 | −0.20 | −0.21 |
| S4X5 · X3 | −0.46 | +0.06 | −0.69 | +0.48 | −0.17 | −0.32 | +0.07 | +0.00 |
| DV15 · X3 | −0.22 | −0.16 | −0.41 | −0.25 | −0.28 | −0.09 | +0.08 | −0.14 |
| DT · X4 | −0.52 | −0.10 | +0.07 | −1.67 | −0.08 | +0.16 | −0.19 | −0.29 |
| DV1h · X1 | −0.35 | −0.24 | −0.28 | −0.15 | −0.08 | +0.29 | +0.09 | +0.24 |
| DV1h · X2 | −1.18 | −0.02 | −0.41 | −0.95 | +0.47 | +0.65 | +0.23 | +0.39 |

DV1h is the only family that is positive in the last 3 halves in a row. It was negative in the 5 halves before. That is a regime pattern, not a stable edge (it is the same "works since 2025-H2" shape that the operator's live lessons warn about).

## How much of the drop is left after W hours (the survival idea)
Every pump high in U-A that held W hours (no new high), measured from the close at W. Hindsight-free at W.

![drop left](PEAK_SHORT_V2_2026-11-18_dropleft.png)

| W (h held) | n | drop already done | further drop in 24 h (median) | 24 h short, gross MEDIAN | 24 h short, gross MEAN | squeezers (≥ +10 % within 24 h) |
|---|---|---|---|---|---|---|
| 0.25 | 39,761 | −2.8 % | −9.4 % | +2.6 % | −0.1 % | 46 % |
| 1 | 24,645 | −4.8 % | −8.7 % | +2.2 % | −0.0 % | 44 % |
| 2 | 19,406 | −6.1 % | −8.3 % | +2.2 % | +0.1 % | 41 % |
| 4 | 15,343 | −7.9 % | −7.8 % | +1.9 % | +0.2 % | 39 % |
| 8 | 11,825 | −10.2 % | −7.3 % | +1.7 % | +0.3 % | 35 % |

What it says:
- **Waiting does remove some squeezers, but slowly**: 46 % → 35 % over 8 hours.
- **It costs the drop**: after 4 h, half of the total median drop is already gone (left share 78 % at 15 min → 52 % at 4 h → 45 % at 8 h).
- **The typical trade wins (median +2 %), the average trade does not (mean ≈ 0, gross).** After fees, slippage and funding (≈ 0.3–0.4 % per trade) the average is negative at every W. The "obvious" dump in the charts is the median; the money is decided by the tail.

## Event study (price after entry)
![event study](PEAK_SHORT_V2_2026-11-18_event.png)
*Median path of price vs entry, −6 h → +48 h. Winners drift down to −6 %. Losers rise +3 to +4 % and their upper band reaches +10 to +14 %. The tradable entries (blue dashed) drift lower than random same-state bars (grey), which is the "beats random" result. The gap is 1–2 %, smaller than the losers' tail.*

## Frequency (trades per day)
| entry | exit | U-A (all pumps) | U-F (Frenzy flag on) | share of days with ≥ 8 trades | U-A in 2026 |
|---|---|---|---|---|---|
| S1X5 | X3 | 12.4 | 8.7 | 59 % | 18.1 |
| S2X5 | X3 | 11.4 | 8.0 | 57 % | 16.4 |
| S4X5 | X3 | 10.2 | 7.0 | 55 % | 14.3 |
| S4X10 | X3 | 7.9 | 5.7 | 45 % | 12.0 |
| DT | X3 | 2.5 | 1.8 | 6 % | 4.5 |
| DV15 | X3 | 10.2 | 7.1 | 55 % | 14.2 |
| DV1h | X1 · X3 | 3.0 · 6.0 | 1.8 · 4.0 | 6 · 33 % | 3.2 · 8.4 |
| any entry (union) | X3 | 55 | 39 | 89 % | 80 |

**Frequency is not the problem.** These coins give plenty of short setups. None is profitable.

**Frenzy subset (U-F) is less bad than the rest** in almost every cell (e.g. S1X10 · X3: U-F −0.02 vs non-Frenzy −0.69; DV1h · X1: U-F **+0.16** at 1.8/day vs −0.27). This was not a pre-registered cell. It is a post-hoc split, so it is a watchlist note only.

## Costs, overlap and sizing
- **Funding** averages −0.04 to −0.19 %/trade here (shorter holds and a broader universe than the SQUEEZER study's −1 to −1.8 %).
- **F4 funding filter:** 68 % of DV1h trades are on coins already on 4 h or 1 h funding. Skipping them leaves 1/day and does not raise the average (DV1h · X1 −0.01 → +0.01; DV1h · X2 +0.05 → −0.30). The forced exit before settlements changes almost nothing.
- **Overlap with EXHAUSTION_SHORT:** 2–12 % of trades. The non-overlapping trades average −0.02 to −0.64. **Nothing here adds value on top of EXH.**
- **DCR at one size (0.183 × equity per trade, every calendar day):** every ≥ 8/day cell **ruins the account** (max drawdown −100 %). The reason is clustering as much as the negative average: on alt-pump days up to 130–180 positions are open at once (≈ 24–32× equity notional) and one squeeze day costs −75 % to −240 %. The best cell, DV1h · X1, is −0.03 %/day, worst day −15 %, max DD −68 %, up to 18 positions at once.

## Replay: the operator's coins, 11-16 → now (fresh klines to 11-18 13:55 UTC)
Net % at 0.10 % slip. X3 exit shown because the peak-stop exits skip most BREV entries (stop > 15 % away). "(open)" = marked to the last close.

| coin | S1X5 · X3 | S2X5 · X3 | S4X5 · X3 | DT · X3 | DV15 · X3 | DV1h · X3 |
|---|---|---|---|---|---|---|
| **BREV** | 11-17 16:30 +4.8 · 11-18 02:45 **−12.3** | 17:30 −13.7 · 03:45 −12.9 | 19:30 −12.6 · 05:45 +4.8 | — | 18:15 −13.0 · 03:45 −12.9 | 11-18 08:00 +4.8 |
| **HUMA** | 11-16 17:30 **−12.7** · 22:30 +4.5 | 18:30 −12.6 · 23:45 +4.7 | 11-17 01:15 +4.7 | — | 11-17 01:30 +4.7 | 11-17 09:00 +4.4 |
| **BLESS** | 11-16 19:15 +4.8 | 19:15 +4.8 | 21:15 +4.9 | 11-17 03:30 +4.8 | 11-16 19:00 +4.8 | 11-16 00:00 −12.3 · 11-17 04:00 +4.8 |
| **PIPPIN** | 11-18 11:30 +4.9 | 12:30 +4.8 | — | — | — | — |
| **ZKC** | 11-18 11:00 +2.8 | 12:00 +4.8 | — | — | 12:30 −6.8 (open) | — |
| **ZKP** | 11-18 09:15 +4.8 | 10:15 +3.6 | 12:15 +0.8 (open) | — | 09:45 +4.8 | — |
| **sum** | **+1.7 (8 trades)** | **−16.5 (8)** | **+2.5 (5)** | +4.8 (1) | −18.5 (6) | +1.7 (4) |

![BREV](PEAK_SHORT_V2_2026-11-18_replay_BREVUSDT.png)
![HUMA](PEAK_SHORT_V2_2026-11-18_replay_HUMAUSDT.png)

Also `_replay_BLESSUSDT.png`, `_replay_PIPPINUSDT.png`, `_replay_ZKCUSDT.png`, `_replay_ZKPUSDT.png`.

**Reading the replay:**
- The survival rule shorted BREV at 16:30 (+4.8, TP) — and again at 02:45, **after** the 01:00 spike to 0.136. BREV went back up and took −12 %. HUMA's first survival short (17:30) was stopped before its real top.
- So this week shows the same picture as 3 years: **most trades win 3–5 %, one squeeze per coin takes −12 to −14 %.** The win rate needed at TP 5 / stop 12 is 69 %; these rules win 67–68 %.
- These coins were picked because they dumped; they flatter every short rule.

## What it means
1. **The dump after a pump is real in the median, not in the mean.** The median short wins about +2 % in 24 h at every wait time. The mean is about 0 before costs. A few squeezes (+30 % to +100 %) cancel many small wins. This is the third study to find the same shape (SECOND_PEAK, SQUEEZER, this one).
2. **Timing rules help, but not enough.** Every reasonable trigger (held high, divergence, survival) is 0.4–1.0 %/trade better than shorting at a random moment. None gets above zero after costs.
3. **More trades do not fix a negative average; they multiply it.** At 10+/day the clustering makes it worse: pumps come in waves, and so do squeezes.
4. **A double top is rare and not better.** Requiring the neckline break puts the entry ≥ 8 % under the top, so the stop is wide and most of the drop is gone.
5. **The money is in the exit/tail, not the entry.** Every winning cell would need a way to cut the squeeze tail without stopping out the winners. Tight stops fail (the SQUEEZER study: 75 % go ≥ 5 % against first). That problem is unsolved.

## Verdict
**NO** — for all four families (survival, double top, divergence, funding-aware) and for the all-coins frequency goal.
- No cell passes more than 2 of 8 items. None passes the never-read H0 block.
- Nothing adds value on top of EXHAUSTION_SHORT.
- Not good enough for an observe line under the prereg rule (≥ 6 items incl. holdouts).
- EXHAUSTION_SHORT stays the only post-pump short (≈ 3.3/day). There is still no ≥ 8/day short sleeve with an edge.

## Next step
- **Nothing to build.**
- **One watchlist note (no arm, post-hoc):** DV1h (1 h bearish RSI divergence at the episode high) on **Frenzy-flagged** coins: +0.16 %/trade at 1.8/day (X1), positive in 25-H2 → 26-H2 only. Read it again on forward data at the next batch review. Bar to move it: the full prereg pass bar on data after 2026-11-18.
- **If the operator wants to keep hunting the high-frequency short:** the open problem is the squeeze tail, not the entry. A study of *position-splitting* (half off at +3 %, rest with a trailing stop) or *scale-in on strength* would attack that directly. Cross-coin clustering (cap of positions per pump wave) must be part of any design.

## Assumptions and limits
- Pump universe and episode definition are my pre-registered choice (+15 % / 6 h, 10× volume, $20 M). Other thresholds were not tried (by design, to keep the family small).
- The pair blacklist is today's list applied to history (same as all earlier studies).
- Random-timing null for the F4 cells uses the base exit (not the forced pre-settlement exit).
- H0 (2023-H1) covers 147 pairs (fewer coins were listed), 509 episodes; funding real for all its pair-months fetched.
- Months with no funding file are charged 0.05 %/8 h against the short (same rule as earlier studies).
- Order-book depth, latency and live slippage on thin coins (0.3–0.6 % seen live) are not modelled beyond the 0.10 / 0.20 % slip lines; they would make every result worse.
- HOLDOUT-A (2023-H2 part) was read by earlier studies; H0 was not.
