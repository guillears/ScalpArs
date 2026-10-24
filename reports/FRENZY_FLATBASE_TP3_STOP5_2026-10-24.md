# FLATBASE_SHORT: would a −3 % take-profit or a 5 % stop do better?

Date: 2026-10-24. Research only. No bot code or config changed, nothing was staged, and no network calls were made.

- Rules, frozen before any result: `FRENZY_FLATBASE_TP3_STOP5_2026-10-24_PREREG.md`.
- Script: `scripts/study_flatbase_tp3_stop5.py`. It reuses the exit study's simulator. `scripts/study_flatbase_short_exits.py` got one change: `simulate()` and `worker()` now take an optional exit grid. With no grid passed they behave exactly as before, and that was checked.
- Outputs:
  - `_tables.md`: the full printed output;
  - `_cells.csv`: every cell and every period;
  - `_trades.csv`: every trade under the 72 h cells;
  - `_wick1m.csv`: the 1-minute re-check of the 5 % stops.
- Runtime: about 19 s.

## Verdict: keep the live exit (TP −8 % / stop 15 %). None of the 8 alternatives qualifies.

Over **2,250+ simulated FLATBASE shorts** (2025-01 → 2026-10), no tighter exit beats today's.

**Every alternative made less money than today's exit across 2026.** On the full 2026 sample, each of the 8 cells is below TP −8 / stop 15.
- In 7 of them the 95 % range of the difference sits entirely below zero.
- The closest is TP −8 / stop 10: −0.17 %/trade, range −0.45 to +0.13.
- No cell passes the pre-registered bar. Each one fails "better in every half-year" and "2026 lower bound > 0".

**Why the 6-fill batch looked so different.** On the 10-23/24 batch, TP −5 / stop 5 made +1.60 %/fill against −3.57 for the live exit. That batch had 3 of 6 trades hit the −15 % stop. In the 2026 backtest only about 21 % of trades do that.
- A tight stop does save every one of those big losers.
- But it also stops out many trades that would have gone on to hit −8 %.
- Over 1,115 trades, the winners it kills cost more than the losers it saves. Example in 2026, TP −5 / stop 5:
  - saved: all 234 live −15 % stops, worth +2,679 points;
  - lost: 193 live winners stopped out, plus 454 winners capped at −5 %, costing −3,725 points;
  - net: about −1,200 points.

**−3 % take-profit:** it wins often, 63–83 % of trades. But each win is tiny, so it does not cover the 5–15 % stops. Its 2026 average is about 0 (−0.17 to +0.15 %/trade), against +0.93 for the live exit.

**5 % stop:**
- It is hit on 35–56 % of trades (47–56 % with TP −5 / −8).
- About 4 in 10 of those hits look like noise. The price spent less than a minute above the stop, then came back (see the wick section).
- With TP −5 it averages −0.14 %/trade in 2026.
- At the same risk per trade, its drawdown is far larger, because a tighter stop means a bigger position.

## Before → after (72 h hold; avg net % per trade; Δ = cell − live exit)

| exit | ① 6 real fills 10-23/24 (DL 367 scout read, no funding) | ① study live block 10-14→10-22 (13 real episodes) | ② master 10-02→10-13 (38 real episodes) | 2026 backtest, H1+H2 (≈1,118) · Δ | full year 2025-01→2026-10 (≈2,260) | max drawdown at the study's size, 2026 |
|---|---|---|---|---|---|---|
| **live TP −8 / stop 15** | −3.57 | +1.64 | +0.32 | **+0.93** | +0.72 | −48 % of $3,200 |
| TP −3 / stop 5 | not computable offline | +0.87 | −1.26 | −0.17 · **−1.10** | −0.10 | −270 % |
| TP −3 / stop 10 | not computable offline | +0.69 | −0.71 | +0.08 · **−0.85** | +0.05 | −79 % |
| TP −3 / stop 15 | −3.07 | −0.11 | +0.53 | +0.15 · **−0.79** | +0.06 | −77 % |
| TP −5 / stop 5 | **+1.60** | +0.11 | −1.34 | −0.14 · **−1.08** | +0.01 | −232 % |
| TP −5 / stop 10 | −0.07 | +1.02 | −1.28 | +0.37 · **−0.56** | +0.38 | −56 % |
| TP −5 / stop 15 | −1.73 | −0.16 | +0.24 | +0.53 · **−0.40** | +0.43 | −48 % |
| TP −8 / stop 5 | not computable offline | +1.45 | −1.24 | +0.09 · **−0.85** | +0.20 | −192 % |
| TP −8 / stop 10 | −7.07 | +2.83 | −1.18 | +0.77 · **−0.17** | +0.63 | −58 % |

**How the drawdown is measured:** this is the exit study's sizing. Each trade is sized so that the 1st-percentile loss equals 5 % of $3,200, with at most 5 shorts open and daily mark-to-market. A 5 % stop shrinks that 1st-percentile loss, so the position is about 2.5× bigger, and the frequent small stops then add up to drawdowns over 100 %.

**About block ① (the 6 real fills):** these are the DL 367 scout numbers. They do not include funding. The 5m path after 10-22 16:15 is not in any local cache, so the 3 cells the scout does not track could not be re-priced without a Binance fetch.

## Backtest per half-year (72 h): N · WR · avg % (funding included) · Δ vs live [95 % day-block bootstrap]

⚠ The 2025 halves are less reliable data (delisted coins and a reconstructed universe), as explained in the exit study.

| exit | ⚠ 2025-H1 | ⚠ 2025-H2 | 2026-H1 | 2026-H2 | 2026 Bonferroni LB of Δ | 2026 vs fair random entry: null · p |
|---|---|---|---|---|---|---|
| **live −8 / 15** | 357 · 66 % · +1.00 | 780 · 64 % · +0.30 | 730 · 68 % · +1.08 | 385 · 68 % · +0.66 | baseline | +0.39 · 0.050 |
| −3 / 5 | 359 · 64 % · −0.03 · Δ −1.03 [−2.05, +0.07] | 785 · 65 % · −0.04 · Δ −0.33 [−1.00, +0.34] | 735 · 64 % · −0.12 · Δ −1.20 [−1.89, −0.52] | 385 · 62 % · −0.25 · Δ −0.91 [−1.77, −0.02] | −1.84 | −0.07 · 0.78 |
| −3 / 10 | 359 · 78 % · +0.17 · Δ −0.83 [−1.77, +0.16] | 784 · 77 % · −0.06 · Δ −0.35 [−0.95, +0.27] | 734 · 79 % · +0.18 · Δ −0.89 [−1.53, −0.27] | 385 · 77 % · −0.12 · Δ −0.78 [−1.52, −0.04] | −1.50 | −0.01 · 0.31 |
| −3 / 15 | 359 · 82 % · +0.24 · Δ −0.76 [−1.57, +0.13] | 784 · 81 % · −0.14 · Δ −0.43 [−0.96, +0.13] | 734 · 85 % · +0.42 · Δ −0.66 [−1.20, −0.11] | 385 · 81 % · −0.37 · Δ −1.03 [−1.64, −0.41] | −1.35 | +0.13 · 0.48 |
| −5 / 5 | 359 · 54 % · +0.26 · Δ −0.74 [−1.64, +0.23] | 785 · 54 % · +0.10 · Δ −0.20 [−0.82, +0.43] | 734 · 52 % · −0.16 · Δ −1.24 [−1.88, −0.59] | 385 · 52 % · −0.11 · Δ −0.76 [−1.61, +0.10] | −1.80 | −0.03 · 0.75 |
| −5 / 10 | 359 · 68 % · +0.39 · Δ −0.61 [−1.34, +0.15] | 784 · 69 % · +0.39 · Δ +0.09 [−0.42, +0.63] | 733 · 71 % · +0.49 · Δ −0.59 [−1.13, −0.04] | 385 · 69 % · +0.15 · Δ −0.51 [−1.08, +0.07] | −1.11 | +0.12 · 0.18 |
| −5 / 15 | 358 · 72 % · +0.41 · Δ −0.60 [−1.09, −0.02] | 784 · 74 % · +0.31 · Δ +0.01 [−0.40, +0.45] | 733 · 77 % · +0.78 · Δ −0.29 [−0.68, +0.10] | 385 · 74 % · +0.04 · Δ −0.61 [−1.02, −0.19] | −0.83 | +0.28 · 0.21 |
| −8 / 5 | 359 · 47 % · +0.76 · Δ −0.24 [−1.00, +0.57] | 785 · 43 % · +0.10 · Δ −0.19 [−0.75, +0.35] | 734 · 43 % · +0.06 · Δ −1.02 [−1.61, −0.44] | 385 · 43 % · +0.13 · Δ −0.52 [−1.35, +0.29] | −1.51 | +0.08 · 0.53 |
| −8 / 10 | 359 · 62 % · +1.17 · Δ +0.17 [−0.31, +0.65] | 783 · 57 % · +0.19 · Δ −0.11 [−0.46, +0.26] | 733 · 62 % · +0.77 · Δ −0.31 [−0.69, +0.06] | 385 · 62 % · +0.76 · Δ +0.10 [−0.37, +0.57] | −0.56 | +0.24 · 0.040 |

Pre-funding averages, worst trades, stop and TP rates, $ per short and drawdown for each cell and half are in `_tables.md`.
- Worst trade: about −7 % with a 5 % stop, about −16 % with 10 %, about −21 % with 15 %.
- The 48 h hold, shown as context only, gives the same picture. Every cell is below the live exit in 2026.

**Sanity check:** the live-exit row and the TP −5 / −8 × stop 10 / 15 cells reproduce the 10-22 exit study exactly, period by period (N, average and null p).

## Verdict against the pre-registered bar (72 h)

| exit | Δ > 0 in every half | 2026 Bonferroni LB > 0 | beats fair random p < 0.05 | no pair or day ≥ 50 % of the gain | > 0 after the 30–50 % haircut | CANDIDATE |
|---|---|---|---|---|---|---|
| −3 / 5 | ❌ | ❌ (−1.84) | ❌ 0.78 | ❌ no gain | ❌ | **no** |
| −3 / 10 | ❌ | ❌ (−1.50) | ❌ 0.31 | ❌ no gain | ❌ | **no** |
| −3 / 15 | ❌ | ❌ (−1.35) | ❌ 0.48 | ❌ no gain | ❌ | **no** |
| −5 / 5 | ❌ | ❌ (−1.80) | ❌ 0.75 | ❌ no gain | ❌ | **no** |
| −5 / 10 | ❌ | ❌ (−1.11) | ❌ 0.18 | ❌ no gain | ❌ | **no** |
| −5 / 15 | ❌ | ❌ (−0.83) | ❌ 0.21 | ❌ no gain | ❌ | **no** |
| −8 / 5 | ❌ | ❌ (−1.51) | ❌ 0.53 | ❌ no gain | ❌ | **no** |
| −8 / 10 | ❌ (2025-H2, 2026-H1 < 0) | ❌ (−0.56) | ✅ 0.040 | ❌ no gain | ❌ | **no** |

**Recommendation:** keep TP −8 / stop 15. The DL 367 scout shadows keep running on real fills and stay under their frozen 30-fill gate. This study is now their prior:
- every shadow they track is below the live exit in 2026;
- TP −5 / stop 5 is −1.08 %/trade below the live exit.

## Saved vs cut (72 h, 2026; the trades both the cell and the live exit take, in % points)

| exit | live −15 stops: n · saved · ΣΔ | live −8 winners: n · cut by the stop / capped at the lower TP · ΣΔ | live time exits ΣΔ | net ΣΔ |
|---|---|---|---|---|
| −3 / 5 | 234 · 234 · +2,966 | 647 · 150 / 495 · −4,255 | +57 | −1,232 |
| −3 / 10 | 234 · 234 · +2,281 | 647 · 30 / 615 · −3,465 | +233 | −950 |
| −3 / 15 | 234 · 97 · +1,769 | 647 · 0 / 645 · −3,080 | +441 | −869 |
| −5 / 5 | 234 · 234 · +2,679 | 647 · 193 / 454 · −3,725 | −152 | −1,199 |
| −5 / 10 | 234 · 234 · +1,794 | 647 · 45 / 601 · −2,520 | +120 | −606 |
| −5 / 15 | 234 · 49 · +991 | 647 · 0 / 646 · −1,854 | +421 | −442 |
| −8 / 5 | 234 · 234 · +2,388 | 647 · 225 / 0 · −2,826 | −510 | −949 |
| −8 / 10 | 234 · 234 · +1,186 | 647 · 56 / 0 · −988 | −368 | −170 |

The pattern holds in every row: the tight exits rescue all of the live big losers, but they give back more on the winners. The full-year rows in `_tables.md` show the same.

## 5 % stop: wick risk

**On 5m bars** (all 5 % stop-outs on the taken trades, all periods; TP −8 / stop 5 has 1,282):
- **0 gap fills.** No stop bar opened above the stop.
- **Strict "wick stop"** (the bar's high crossed the stop, the 5m close was back below the entry): only 5 of 1,282 (0.4 %). It is rare, because the bar that hits +5 % rarely falls all the way back to the entry within 5 minutes.
- **Milder version** (the 5m close was back below the stop level): **57 %**. The 10 % and 15 % stops show the same 53–57 %, so this is how these coins move, not something special to 5 %.

**1-minute re-check:** 1m bars are cached locally in `reports/WILLY_LATE_JOIN_2026-10-22_data/k1m/`, and they cover 515 of the 1,282 stop bars (40 %; the rest could not be checked).
- In every covered case, a 1m high did reach the stop, so the data are consistent.
- **39 % (202) were pure wicks:** no 1m close at or above the stop. 154 of those touched the stop in only one 1m bar.
- The other 61 % had a 1m close at or above the stop.
- The pure-wick share is rising: 33–35 % in 2025, 41 % in 2026-H1, 48 % in 2026-H2, and 56 % on ② master (16 checked).

**What that means:**
- A resting stop, or the exchange backstop, fills on those wicks. So the backtest is right to count them, and the 5 % stop result is not too pessimistic for a resting stop.
- If the live stop is only checked by polling, some of those under-a-minute wicks might not fire. That was not modelled.
- The 5 % stop sits inside the normal one-minute noise of these coins. Roughly 4 in 10 of its stop-outs are noise.
- **The stop-first convention barely matters here.** A bar that touches both the stop and the TP happened only 2–3 times per cell.

## Blind spots / what could not be done
- **The 6 real fills under TP −3 / stop 5, TP −3 / stop 10 and TP −8 / stop 5.** These need the 5m path from 10-23 04:00 to 10-24 11:30. It is not cached locally (the study data end on 10-22 16:15, and the scout's kline cache is empty), and the brief said no network. Six fills could not change the verdict either way.
- **Funding** is charged on the entry notional, as in the exit study.
- **Margin** is modelled as isolated.
- **Timing within a 5m bar** is checked only through the 40 % 1m sample above. Ticks are not used.
- **Same family as the exit study:** this test adds 6 new cells to a family that has already been searched. This only matters for a cell that passes, and none did.


_Re-run 2026-10-24 after review: the sizing / drawdown pass now orders trades deterministically (time, then pair); drawdowns above are from the deterministic run. Every other number is unchanged (no-funding averages differ by < 1e-6)._
