# WILLY / WILLY_HT EXIT V2 (2026-11-18)

**Question (operator):** "Can it be that WILLY HT and WILLY do not work in the backtest due to the exit? Maybe try different exits?"

**Pre-registered:** `WILLY_EXIT_V2_2026-11-18_PREREG.md`, frozen 12:38 UTC, plus Addendum 1 (the full TP × SL grid) at 12:39 UTC. Both were frozen before any cell was read. Research only. Nothing in the bot changed and nothing was committed.

## Plain answer: no, the exit is not what breaks them
- **The "stopped then recovered" bucket is mostly an illusion of a −4 % stop.** It covers 81 HT trades that were back at or above entry within 4 h.
  - Before coming back, 64 of these 81 dipped below −6 %. 43 dipped below −8 %, 29 below −10 % and 21 below −12 %, all within 4 h. The median deepest dip was −8.2 %.
  - Within 24 h, 53 of the 81 went below −10 %.
  - The recovery is a short touch. On the median path even these trades sit at −4 to −6 % after 24 h (chart below).
- **Widening the stop splits the stopped trades about in half.**
  - Headline cell: HT, SL −10 %, today's TP, 24 h.
  - Of the 141 HT trades stopped at −4 % today, 75 are saved, for +373 pts. 66 of those 75 are saved by the +0.25 % late TP, and only 9 reach the full +2 % TP.
  - The other 66 now lose −10 % instead of −4 %, for −393 pts.
  - Net: −7 pts over 417 trades.
- **Win rate goes up but the bar goes up faster.** WR rises from 62 % to 86 %. The breakeven WR rises from 69 % to 87 %.
- **No exit fixes either sleeve.** The cells cover 180 exits per sleeve (wide stop × long hold × TP grid, structural stops, break-even lock, trailing, time grace).
  - No exit is positive on all three blocks: DESIGN, SEEN and HOLDOUT-D.
  - No exit beats the family-wise null. The best DESIGN gain is p = 0.25 for HT and p = 0.43 for WILLY.
  - Every DESIGN "winner" turns negative on SEEN (2026-H2).
- **Verdict:**
  - **HT: KEEP today's exit** (+2 % maker TP, −4 %, 90 m). Also start one OBSERVE line: shadow-price the −10 % / 24 h stop on real HT stops. The master disagrees with the backtest, see below.
  - **WILLY: KEEP.**
  - The losses are on the entry side: most entries are a fade that a wider stop only makes more expensive.

## Headline row (operator request): HT, SL −10 %, today's TP (+2 % floor, high target, late +0.25 % after 30 m)

| hold | stopped today | → reach TP (of which full TP / late TP) · pts | → still stopped at −10 % · pts | → time out · avg · pts | not-stopped trades changed · pts | total Δ pts | avg/trade before → after (all) |
|---|---|---|---|---|---|---|---|
| 4 h | 141 | 61 (9 / 52) · +309 | 55 · −329 | 25 · −4.78 · −19 | 10 · +16 | **−23** | −0.27 → −0.32 |
| 12 h | 141 | 72 (9 / 63) · +358 | 63 · −375 | 6 · −4.98 · −7 | 10 · +13 | **−11** | −0.27 → −0.30 |
| 24 h | 141 | 75 (9 / 66) · +373 | 66 · −393 | 0 | 10 · +13 | **−7** | −0.27 → −0.29 |

**Per period, 24 h:**

| | DESIGN 2025-01 → 06 | SEEN 2026-07 → 11 |
|---|---|---|
| total Δ | **+47 pts** (88 stopped: 52 saved, 36 to −10) | **−54 pts** (53 stopped: 23 saved, 30 to −10) |
| avg/trade | −0.37 → −0.19 | −0.11 → −0.45 |

- **HOLDOUT-D (2023-H2):** −0.21 → −0.32.
- **WR / breakeven WR:** 86 / 87 % on DESIGN.
- **Kelly at the daily Wilson floor:** −0.50.
- **Max loss per trade:** −10.2 %.

**A pure +2 % fixed TP with SL −10 % / 24 h** (no late TP; this is the literal "+2 % TP") is worse:
- 72 stopped trades reach +2 %, for +441 pts;
- 67 still hit −10 %, for −397 pts;
- dropping the late TP and the 90-min cap costs the not-stopped trades −113 pts;
- total −74 pts; DESIGN −0.50, SEEN −0.36, D −0.19.

**WILLY, same cell (24 h):**
- 80 saved (+382), 84 to −10 (−490). Total −127 pts.
- DESIGN −0.39 → −0.70, SEEN +1.02 → +1.07, D −0.18 → −0.15.

**Deepest dip of the "stopped then back ≥ entry within 4 h" trades:**

| sleeve | N | below −6 % | below −8 % | below −10 % | below −12 % | median deepest dip (4 h) | below −10 % within 24 h |
|---|---|---|---|---|---|---|---|
| HT | 81 | 64 | 43 | 29 | 21 | −8.2 % | 53 |
| WILLY | 96 | 72 | 54 | 37 | 24 | −8.8 % | 69 |

The dips are counted from entry, within 4 h, on 1m lows. Of ALL 141 HT trades stopped today, 65 went below −10 % within 4 h and 47 below −12 %.

**HT trades stopped today, by entry month in SEEN (24 h, SL −10 %), saved → TP vs still hit −10 %:**

| month (2026) | saved → TP | still hit −10 % |
|---|---|---|
| Jul | 3 | 6 |
| Aug | 7 | 7 |
| Sep | 3 | 8 |
| Oct | 2 | 6 |
| Nov | **8** | **3** |

- Only November looks like "stops recover".
- The 3 real master HT stops are all in that window: ICNT 11-08, EVAA 11-14 and INIT 11-15. All three recovered.

![event study](WILLY_EXIT_V2_2026-11-18_event_study.png)

**How to read the event study.** For the trades stopped today, the chart shows the median close vs the fill over 24 h, with an IQR band.
- **Blue:** stopped, then back ≥ entry within 4 h. These hover around −2 to −4 % and then slide to about −6 %.
- **Orange:** stopped, never back. These fall to about −8 % within 3 h and about −10 % by 24 h.
- **Green:** not stopped, as a reference.
- There is no "deeper dip, then rally" shape that a wider stop could harvest. Even the green reference trends down after 2 h, which is why today's short 90-min cap is right for this entry.

## Method
- **Trades.** The WILLY_DEEP base trades (today's entry rules, real-time flags):
  - WILLY: 545. HT: 417.
  - DESIGN = 2025-01-13 → 2026-06-30. SEEN = 2026-07 → 11-17.
  - HOLDOUT-D (2023-H2): WILLY 99, HT 201. It was **already read once** by WILLY_DEEP, so this is a declared second read.
- **Choice.** Made on DESIGN only, per sleeve; the sleeves are never pooled.
- **Simulator** (`scripts/willy_exit_v2_sim.py`).
  - With today's parameters it reproduces all 962 base trades: median |Δ| 0.000, 100 % within 0.05.
  - On the master fills it reproduces `my_reprice` on 36 of 39. The 3 differences are WILLY runner edge cases, and every variant is compared against my own today-exit.
- **Realism.**
  - 1m bars for the whole hold, up to 48 h. I downloaded 1,739 extra daily 1m files from data.binance.vision (0 API weight). For the five coins traded on 11-18, today's partial day came from REST (≈ 50 weight in total).
  - Maker TP fills only when the high trades through the limit by ≥ 1 tick (0.02 %).
  - Stops, trails and time exits are taker 0.05 % + 0.10 % slippage. They are also read at 0.20 %.
  - Stops are gap-aware. Inside a bar the stop is checked before the TP.
  - Trail and BE lines move only on closed-bar highs.
  - Real funding is included (hourly coins included).
- **Live-stopped split.** Every cell is read on three groups: trades stopped at −4 % today (saved vs deeper) and the trades not stopped.
  - For TP- and hold-unchanged stop variants, the not-stopped Δ is 0.00. Examples: HT `TPtoday/SL4/720m` vs 90 m, and WILLY `TPtoday/SL4/720m` total Δ 0. So the replica is not lying.
- **Family-wise null.** Day-block sign flips of every cell's per-trade Δ vs today (one sign per day for all 180 cells jointly), 2,000 draws. The statistic is the max DESIGN mean Δ per entry.
- **Plateau.** One step in SL / hold / TP from the chosen cell.
- **Pass bar (prereg).** All of: DESIGN Δ > 0 with null p < 0.05 · SEEN Δ > 0 · D Δ > 0 · plateau · Δ > 0 at 0.20 % · master Δ ≥ 0 with no joint-check flip.
- **Assumptions / limits.**
  - Entries are fixed, so the portfolio is not re-walked for longer holds. Same-pair overlaps the bot would refuse: 1 trade (SL10 / 24 h), 9 trades (TP6 / SL10 / 48 h). Peak concurrent book, both sleeves: 8 (today) → 13–15. That is under the 10 + 10 sleeve slots but not under the shared book.
  - In D, routing is not rebuildable: each sleeve's rule runs on every entry.
  - The 2025–26 universe is today's listings (survivorship).
  - DCR is at X = 1 (fill % per unit of equity). It scales linearly with the sleeve's real ticket.
  - The 11-18 paper EVAA fill is priced as +2.00 only where its exit could not change (a TP ≤ 2 hit at minute 3). Elsewhere it is marked n/p.

## Verdict per family (what the live engine supports)

| family | best cell on DESIGN, HT | best cell on DESIGN, WILLY | passes? | live engine |
|---|---|---|---|---|
| 1 wide stop + long hold, TODAY-TP | SL12/4 h: Δ +0.20 · SEEN −0.53 · D −0.08 | SL4/12 h ≈ today (Δ 0.00) | ✗ | config only (`frenzy_willy_stop_pct`, `frenzy_willy_max_hold_minutes`). The stop is SHARED by WILLY and HT, so an HT-only stop needs a new field. |
| 1 fixed TP × SL grid | TP6/SL10/48 h: Δ +0.64 · SEEN −0.24 · D −0.35 · p 0.25 | TP5/SL4/24 h: Δ +0.37 · SEEN −0.75 · D −0.05 · p 0.43 | ✗ | config mostly (`frenzy_willy_ht_tp_pct` / `tp_pct`, late TP off). Disabling the high-target raise is shared with WILLY. |
| 2 structural stop | spike-bar low / 4 h: Δ +0.19 · SEEN +0.10 · D +0.01 (sleeve still −0.25 / −0.02 / −0.24) | base / 4 h: Δ +0.20 · SEEN −0.31 · D −0.23 | ✗ (HT: positive Δ everywhere but not the DESIGN pick; the gain is mostly the 29 % of entries it skips, and on D the traded part worsens −0.19 → −0.24) | needs code |
| 3 break-even lock | +0.75: Δ +0.04 · SEEN −0.02 · D +0.13 | +1: Δ +0.04 · SEEN −0.18 | ✗ | needs code (the runner floor exists only for WILLY runner trades) |
| 4 trailing | +3/3 % / 90 m: Δ +0.12 · SEEN −0.06 · D −0.03 | +3/3 % / 24 h: Δ +0.24 · SEEN −0.76 · D +0.33 | ✗ | the ATR runner exists for WILLY; a %-trail needs code |
| 5 time grace (−15 % hard) | all Δ < 0 on DESIGN | all Δ < 0 on DESIGN | ✗ | needs code |

**Pass bar for the chosen cell (prereg):**

| sleeve · cell | DESIGN Δ & null p < .05 | SEEN Δ > 0 | D Δ > 0 | plateau | Δ > 0 @0.20 | master | sleeve > 0 & Kelly > 0 |
|---|---|---|---|---|---|---|---|
| HT · TP6/SL10/48 h | ✗ +0.64, p 0.25 | ✗ −0.24 | ✗ −0.35 | ✓ (SL12 +0.52 · 24 h +0.58 · TP5 +0.25 · SL8 +0.09) | DESIGN ✓ / SEEN ✗ (+0.24 / −0.39 avg) | ✓ 15 · +1.78 → +5.07 (14 priced) | ✗ Kelly −0.11, SEEN −0.35 |
| WILLY · TP5/SL4/24 h | ✗ +0.37, p 0.43 | ✗ −0.75 | ✗ −0.05 | ✓ (48 h +0.37 · 12 h +0.36 · TP4 +0.33 · SL6 +0.03) | ✗ −0.16 / +0.14 avg | ✗ 24 · +2.75 → −0.66 | ✗ Kelly −0.09 |

**After a 30–50 % in-sample haircut**, the chosen cells' DESIGN gains shrink:
- HT: +0.64 → +0.32 … +0.45;
- WILLY: +0.37 → +0.19 … +0.26.

They are still negative on SEEN.

## Master check: the one place the backtest and the real fills disagree (HT)
**HT, master kept fills (15, B25–B27, plus the 11-18 paper EVAA):**

| exit | avg net % | Σ pts | why |
|---|---|---|---|
| today | +1.78 | +26.7 | – |
| SL −10 % / today's TP | +3.28 | +49.3 | all 3 real HT stops recovered: ICNT −4.10 → +0.25, EVAA −4.10 → +8.00, INIT −4.10 → +2.00 |
| TP6 / SL10 / 48 h | +5.07 | – | – |

**What the rebuild shows about these fills:**
- It matches these fills one by one (same exit on 13 of 14 matched).
- It also shows that in the same months (Jul–Oct 2026) 27 of 42 HT stops went on to −10 % anyway.
- The master is 3 stops in one month (November). 3/3 recoveries at the backtest's 43 % recovery rate happens about 8 % of the time.
- Per the MASTER-OVER-BACKTEST rule, the master leads. But 3 stops are 3 observations (about 2 windows), not evidence of a rule.

**→ OBSERVE, not a change.**

**WILLY master (24 kept):** the wide-stop cells do not help.
- SL −10 / 24 h: +2.75 → +2.68.
- The fixed-TP cells hurt: TP5/SL4: +2.75 → −0.66.

**Joint check (HT, the trades kept by the observe-line filters):**
- On `top_pos ≤ 1`, TP6/SL10/48 h improves both blocks: DESIGN 33 trades +1.19 → +2.32, SEEN 14 trades +0.36 → +1.48. The N is tiny, and WILLY_DEEP already found that cut negative on D.
- On `4 h slope ≤ 0`, it flips: DESIGN −0.28 → +0.89, SEEN +0.38 → −0.10.
- The headline SL −10 flips on both filters' SEEN: top_pos ≤ 1 goes +0.36 → −0.18, and 4 h slope ≤ 0 goes +0.38 → −0.06.
- **Flip = no ship.**

## OBSERVE line (proposed; nothing armed)
**`HT_STOP_WIDE10_SHADOW`:** for every real HT and WILLY fill that books STOP_LOSS, price the same fill under SL −10 % with today's TP and late TP, held 24 h, from 1m klines after the close. Each sleeve gets its own row.

**Frozen gate (alert-only):**
- **When:** at ≥ 20 real HT stops.
- **Re-open the exit question if both hold:**
  - Σ (shadow − booked) > 0, and
  - the backtest rebuild over the same dates also shows Δ > 0.
- **Close the line instead if:** Σ ≤ 0, or the backtest disagrees.
- **Exit era:** today's HT exit (+2 / −4 / 90 m).

At ≈ 0.3 real HT stops/day this needs ~2 months.

## Next step
- No exit change.
- The ~−0.3 %/trade gap is entry-side. The trades that never work (orange line) fall 8–10 % in 3 h, and no stop width or hold changes their sign.
- Any further HT / WILLY work should go back to the entry. The W/L under the wide stop (below) found no separator beyond the null.

## Tables per sleeve (DESIGN / SEEN / HOLDOUT-D; all cells in WILLY_EXIT_V2_2026-11-18_cells.csv)

### HT — heat maps (fixed maker TP × SL; cell = avg net %/trade · WR % / breakeven WR %; DESIGN 2025-01→2026-06 | SEEN 2026-07→11 | HOLDOUT-D 2023-H2)


**hold 4 h** (today's exit for reference: DESIGN -0.37 · 62/69 \| SEEN -0.11 \| D -0.21)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.34 · 64/70 \| -0.10 \| -0.22 | -0.43 · 73/78 \| -0.43 \| -0.15 | -0.30 · 79/82 \| -0.57 \| -0.22 | -0.19 · 81/83 \| -0.54 \| -0.30 | -0.16 · 82/84 \| -0.64 \| -0.30 |
| +2 % | -0.52 · 57/66 \| -0.05 \| -0.21 | -0.64 · 64/72 \| -0.22 \| -0.12 | -0.50 · 69/75 \| -0.40 \| -0.19 | -0.46 · 71/76 \| -0.38 \| -0.31 | -0.49 · 71/77 \| -0.46 \| -0.31 |
| +3 % | -0.35 · 52/57 \| -0.22 \| -0.01 | -0.51 · 57/63 \| -0.54 \| +0.06 | -0.34 · 62/66 \| -0.80 \| -0.01 | -0.38 · 63/67 \| -0.72 \| -0.13 | -0.44 · 64/68 \| -0.86 \| -0.13 |
| +4 % | -0.34 · 45/50 \| -0.25 \| -0.27 | -0.46 · 52/57 \| -0.61 \| -0.21 | -0.23 · 57/60 \| -0.91 \| -0.21 | -0.19 · 59/61 \| -0.80 \| -0.34 | -0.26 · 59/62 \| -0.88 \| -0.35 |
| +5 % | -0.52 · 39/45 \| -0.09 \| -0.22 | -0.68 · 45/52 \| -0.52 \| -0.21 | -0.43 · 51/55 \| -0.95 \| -0.22 | -0.38 · 53/57 \| -0.77 \| -0.37 | -0.46 · 54/58 \| -0.81 \| -0.38 |
| +6 % | -0.46 · 36/41 \| -0.23 \| -0.28 | -0.59 · 43/48 \| -0.63 \| -0.26 | -0.27 · 49/51 \| -0.95 \| -0.22 | -0.16 · 51/53 \| -0.78 \| -0.35 | -0.23 · 52/53 \| -0.85 \| -0.37 |

**hold 12 h** (today's exit for reference: DESIGN -0.37 · 62/69 \| SEEN -0.11 \| D -0.21)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.34 · 64/70 \| -0.10 \| -0.19 | -0.39 · 74/79 \| -0.45 \| -0.12 | -0.29 · 81/84 \| -0.57 \| -0.20 | -0.17 · 85/87 \| -0.50 \| -0.30 | -0.23 · 86/88 \| -0.61 \| -0.28 |
| +2 % | -0.49 · 59/67 \| -0.02 \| -0.03 | -0.61 · 67/75 \| -0.21 \| -0.04 | -0.56 · 73/79 \| -0.47 \| -0.17 | -0.42 · 78/82 \| -0.38 \| -0.25 | -0.53 · 79/83 \| -0.56 \| -0.16 |
| +3 % | -0.32 · 53/57 \| -0.22 \| +0.26 | -0.55 · 60/66 \| -0.57 \| +0.11 | -0.47 · 67/71 \| -0.92 \| +0.01 | -0.41 · 71/74 \| -0.62 \| -0.08 | -0.60 · 71/76 \| -0.92 \| -0.02 |
| +4 % | -0.30 · 46/50 \| -0.21 \| -0.07 | -0.54 · 54/60 \| -0.56 \| -0.23 | -0.37 · 62/65 \| -0.92 \| -0.21 | -0.23 · 67/69 \| -0.57 \| -0.29 | -0.38 · 68/70 \| -0.81 \| -0.26 |
| +5 % | -0.47 · 39/45 \| -0.08 \| -0.08 | -0.73 · 48/54 \| -0.49 \| -0.31 | -0.52 · 56/60 \| -1.00 \| -0.36 | -0.29 · 61/63 \| -0.54 \| -0.51 | -0.50 · 62/66 \| -0.78 \| -0.54 |
| +6 % | -0.39 · 37/41 \| -0.28 \| -0.13 | -0.62 · 45/50 \| -0.68 \| -0.44 | -0.32 · 54/56 \| -1.12 \| -0.47 | +0.03 · 60/60 \| -0.61 \| -0.63 | -0.18 · 61/62 \| -0.86 \| -0.62 |

**hold 24 h** (today's exit for reference: DESIGN -0.37 · 62/69 \| SEEN -0.11 \| D -0.21)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.34 · 64/70 \| -0.10 \| -0.19 | -0.39 · 74/79 \| -0.45 \| -0.13 | -0.29 · 81/84 \| -0.55 \| -0.22 | -0.19 · 86/87 \| -0.45 \| -0.32 | -0.19 · 88/89 \| -0.59 \| -0.26 |
| +2 % | -0.49 · 59/67 \| -0.02 \| -0.05 | -0.61 · 67/75 \| -0.19 \| -0.10 | -0.59 · 74/80 \| -0.41 \| -0.12 | -0.50 · 79/83 \| -0.36 \| -0.19 | -0.49 · 80/84 \| -0.57 \| -0.00 |
| +3 % | -0.32 · 53/57 \| -0.27 \| +0.24 | -0.54 · 61/66 \| -0.52 \| +0.07 | -0.49 · 68/72 \| -0.87 \| +0.08 | -0.41 · 73/76 \| -0.63 \| +0.02 | -0.53 · 74/78 \| -0.87 \| +0.18 |
| +4 % | -0.30 · 46/50 \| -0.22 \| -0.08 | -0.52 · 55/60 \| -0.47 \| -0.25 | -0.35 · 63/66 \| -0.81 \| -0.17 | -0.19 · 69/70 \| -0.52 \| -0.30 | -0.30 · 71/73 \| -0.71 \| -0.22 |
| +5 % | -0.49 · 39/45 \| -0.09 \| -0.08 | -0.72 · 48/55 \| -0.41 \| -0.27 | -0.50 · 57/61 \| -0.86 \| -0.25 | -0.14 · 64/65 \| -0.49 \| -0.49 | -0.30 · 66/68 \| -0.63 \| -0.45 |
| +6 % | -0.39 · 37/41 \| -0.27 \| -0.11 | -0.58 · 45/50 \| -0.58 \| -0.43 | -0.27 · 55/57 \| -0.81 \| -0.36 | +0.22 · 62/61 \| -0.32 \| -0.63 | +0.10 · 64/63 \| -0.37 \| -0.63 |

### HT — full reading of the top / family-best cells (DESIGN unless marked)

| cell | N (D/S) | /day | WR | avg W / L | BE WR | DESIGN avg [day CI] | Δ/entry DES · SEEN · D | SEEN avg | D avg | best-5 % out | @0.20 DES/SEEN | Kelly floor | max loss | DCR X=1 %/day DES · SEEN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 256/161 | 0.48 | 62 | +1.79 / -3.90 | 69 | -0.37 [-0.74, +0.01] | +0.00 · +0.00 · +0.00 | -0.11 | -0.21 | -0.66 | -0.48 / -0.20 | -0.40 | -4.2 | -0.199 · -0.188 |
| F1 TPtoday/SL10/240m | 256/161 | 0.48 | 81 | +1.54 / -7.48 | 83 | -0.19 [-0.65, +0.28] | +0.18 · -0.43 · -0.09 | -0.54 | -0.30 | -0.48 | -0.33 / -0.62 | -0.43 | -10.1 | -0.129 · -0.773 |
| F1 TPtoday/SL10/720m | 256/161 | 0.48 | 85 | +1.48 / -9.61 | 87 | -0.17 [-0.71, +0.33] | +0.20 · -0.39 · -0.08 | -0.50 | -0.30 | -0.46 | -0.30 / -0.59 | -0.47 | -10.1 | -0.125 · -0.726 |
| F1 TPtoday/SL10/1440m | 256/161 | 0.48 | 86 | +1.47 / -10.00 | 87 | -0.19 [-0.72, +0.33] | +0.18 · -0.34 · -0.11 | -0.45 | -0.32 | -0.48 | -0.31 / -0.53 | -0.50 | -10.2 | -0.134 · -0.665 |
| F1 TP2/SL10/1440m | 256/161 | 0.48 | 79 | +2.04 / -9.79 | 83 | -0.50 [-1.08, +0.09] | -0.13 · -0.25 · +0.02 | -0.36 | -0.19 | -0.66 | -0.57 / -0.45 | -0.55 | -10.4 | -0.299 · -0.550 |
| F1 TP6/SL10/2880m | 256/161 | 0.48 | 63 | +6.05 / -9.68 | 62 | +0.27 [-0.69, +1.21] | +0.64 · -0.24 · -0.35 | -0.35 | -0.56 | -0.10 | +0.24 / -0.39 | -0.11 | -10.4 | -0.023 · -0.785 |
| F1 TP6/SL10/1440m | 256/161 | 0.48 | 62 | +6.00 / -9.26 | 61 | +0.22 [-0.79, +1.14] | +0.58 · -0.21 · -0.42 | -0.32 | -0.63 | -0.16 | +0.17 / -0.36 | -0.12 | -10.4 | -0.045 · -0.750 |
| F1 TP6/SL12/2880m | 256/161 | 0.48 | 66 | +6.05 / -11.32 | 65 | +0.15 [-0.92, +1.16] | +0.52 · -0.33 · -0.49 | -0.44 | -0.70 | -0.24 | +0.11 / -0.47 | -0.15 | -12.4 | -0.116 · -0.919 |
| F1 TPtoday/SL12/240m | 256/161 | 0.48 | 82 | +1.52 / -7.85 | 84 | -0.16 [-0.69, +0.31] | +0.20 · -0.53 · -0.08 | -0.64 | -0.30 | -0.46 | -0.27 / -0.72 | -0.42 | -12.1 | -0.122 · -0.892 |
| F2 struct-base/1440m | 153/85 | 0.29 | 84 | +1.26 / -7.57 | 86 | -0.18 [-0.71, +0.33] | +0.26 · -0.23 · +0.15 | -0.65 | -0.08 | -0.39 | -0.29 / -0.70 | -0.57 | -11.6 | -0.069 · -0.439 |
| F2 struct-spikelow/240m | 181/100 | 0.34 | 68 | +1.56 / -4.10 | 72 | -0.25 [-0.70, +0.17] | +0.19 · +0.10 · +0.01 | -0.02 | -0.24 | -0.49 | -0.36 / -0.09 | -0.41 | -10.8 | -0.102 · -0.044 |
| F3 BE+0.75 | 256/161 | 0.48 | 32 | +2.02 / -1.43 | 41 | -0.32 [-0.61, -0.05] | +0.04 · -0.02 · +0.13 | -0.13 | -0.08 | -0.59 | -0.58 / -0.36 | -0.25 | -4.1 | -0.169 · -0.171 |
| F4 trail+3/3/90m | 256/161 | 0.48 | 47 | +3.21 / -3.29 | 51 | -0.24 [-0.74, +0.28] | +0.12 · -0.06 · -0.03 | -0.17 | -0.25 | -0.77 | -0.44 / -0.37 | -0.20 | -4.2 | -0.156 · -0.287 |
| F5 grace15 | 256/161 | 0.48 | 67 | +1.78 / -4.81 | 73 | -0.41 [-0.84, -0.00] | -0.04 · -0.22 · +0.10 | -0.33 | -0.11 | -0.72 | -0.52 / -0.40 | -0.45 | -15.1 | -0.226 · -0.493 |

### HT — per half-year (N · avg net %)

| cell | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D |
|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 52·+0.14 | 91·-0.11 | 113·-0.81 | 161·-0.11 | 201·-0.21 |
| F1 TPtoday/SL10/240m | 52·+0.46 | 91·-0.11 | 113·-0.55 | 161·-0.54 | 201·-0.30 |
| F1 TPtoday/SL10/720m | 52·+0.18 | 91·+0.10 | 113·-0.55 | 161·-0.50 | 201·-0.30 |
| F1 TPtoday/SL10/1440m | 52·+0.09 | 91·+0.13 | 113·-0.57 | 161·-0.45 | 201·-0.32 |
| F1 TP2/SL10/1440m | 52·-0.37 | 91·-0.41 | 113·-0.62 | 161·-0.36 | 201·-0.19 |
| F1 TP6/SL10/2880m | 52·+0.61 | 91·+0.89 | 113·-0.38 | 161·-0.35 | 201·-0.56 |
| F1 TP6/SL10/1440m | 52·+0.25 | 91·+0.87 | 113·-0.33 | 161·-0.32 | 201·-0.63 |
| F1 TP6/SL12/2880m | 52·+1.14 | 91·+0.41 | 113·-0.52 | 161·-0.44 | 201·-0.70 |
| F1 TPtoday/SL12/240m | 52·+0.42 | 91·-0.13 | 113·-0.46 | 161·-0.64 | 201·-0.30 |
| F2 struct-base/1440m | 40·+0.35 | 42·-0.16 | 71·-0.50 | 85·-0.65 | 154·-0.08 |
| F2 struct-spikelow/240m | 43·+0.45 | 56·-0.37 | 82·-0.55 | 100·-0.02 | 176·-0.24 |
| F3 BE+0.75 | 52·+0.20 | 91·-0.20 | 113·-0.66 | 161·-0.13 | 201·-0.08 |
| F4 trail+3/3/90m | 52·-0.17 | 91·+0.53 | 113·-0.90 | 161·-0.17 | 201·-0.25 |
| F5 grace15 | 52·+0.19 | 91·-0.24 | 113·-0.82 | 161·-0.33 | 201·-0.11 |

### HT — live-stopped split (DESIGN + SEEN; Δ vs today per trade, summed)

| cell | stopped today | saved N · pts | deeper N · pts | not-stopped changed N · pts | total Δ pts |
|---|---|---|---|---|---|
| F1 TPtoday/SL10/240m | 141 | 70 · +318 | 71 · -357 | 10 · +16 | -23 |
| F1 TPtoday/SL10/720m | 141 | 74 · +360 | 67 · -384 | 10 · +13 | -11 |
| F1 TPtoday/SL10/1440m | 141 | 75 · +373 | 66 · -393 | 10 · +13 | -7 |
| F1 TP2/SL10/1440m | 141 | 72 · +441 | 69 · -402 | 159 · -113 | -74 |
| F1 TP6/SL10/2880m | 141 | 53 · +541 | 88 · -517 | 276 · +102 | +126 |
| F1 TP6/SL10/1440m | 141 | 54 · +529 | 87 · -502 | 276 · +88 | +116 |
| F1 TP6/SL12/2880m | 141 | 58 · +594 | 83 · -649 | 276 · +134 | +79 |
| F1 TPtoday/SL12/240m | 141 | 75 · +340 | 66 · -389 | 10 · +16 | -33 |
| F2 struct-base/1440m | 141 | 97 · +409 | 44 · -154 | 118 · -227 | +29 |
| F2 struct-spikelow/240m | 141 | 106 · +367 | 35 · -81 | 108 · -222 | +64 |
| F3 BE+0.75 | 141 | 69 · +252 | 0 · +0 | 147 · -243 | +8 |
| F4 trail+3/3/90m | 141 | 6 · +28 | 0 · +0 | 264 · -7 | +22 |
| F5 grace15 | 141 | 18 · +104 | 46 · -149 | 0 · +0 | -46 |

### HT — master per batch, kept fills re-priced (N · WR · avg net %; PAPER1118 = the 11-18 export, EVAA +2.00 at 3 min, n/p = not priceable yet)

| cell | B25 | B26 | B27 | PAPER1118 | all kept |
|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 8·88%·+2.93 | 6·83%·+1.22 | 1·0%·-4.10 | 1·100%·+2.00 | 15·+1.78·Σ+26.7 |
| F1 TPtoday/SL10/240m | 8·100%·+3.48 | 6·100%·+3.24 | 1·100%·+2.00 | 1·100%·+2.00 | 15·+3.28·Σ+49.3 |
| F1 TPtoday/SL10/720m | 8·100%·+3.48 | 6·100%·+3.24 | 1·100%·+2.00 | 1·100%·+2.00 | 15·+3.28·Σ+49.3 |
| F1 TPtoday/SL10/1440m | 8·100%·+3.48 | 6·100%·+3.24 | 1·100%·+2.00 | 1·100%·+2.00 | 15·+3.28·Σ+49.3 |
| F1 TP2/SL10/1440m | 8·100%·+2.02 | 6·100%·+2.01 | 1·100%·+2.00 | 1·100%·+2.00 | 15·+2.02·Σ+30.3 |
| F1 TP6/SL10/2880m | 8·100%·+6.21 | 6·83%·+3.39 | 1·100%·+6.00 | n/p | 15·+5.07·Σ+76.0 |
| F1 TP6/SL10/1440m | 8·100%·+6.21 | 6·83%·+3.39 | 1·100%·+6.00 | n/p | 15·+5.07·Σ+76.0 |
| F1 TP6/SL12/2880m | 8·100%·+6.21 | 6·83%·+3.06 | 1·100%·+6.00 | n/p | 15·+4.93·Σ+74.0 |
| F1 TPtoday/SL12/240m | 8·100%·+3.48 | 6·100%·+3.24 | 1·100%·+2.00 | 1·100%·+2.00 | 15·+3.28·Σ+49.3 |
| F2 struct-base/1440m | 4·100%·+2.46 | 4·100%·+2.01 | n/p | n/p | 8·+2.24·Σ+17.9 |
| F2 struct-spikelow/240m | 5·100%·+1.67 | 4·100%·+2.01 | 1·100%·+2.00 | n/p | 10·+1.84·Σ+18.4 |
| F3 BE+0.75 | 8·25%·-0.24 | 6·83%·+1.89 | 1·0%·-2.20 | n/p | 15·+0.48·Σ+7.2 |
| F4 trail+3/3/90m | 8·88%·+3.02 | 6·83%·+1.80 | 1·0%·-4.10 | n/p | 15·+2.06·Σ+30.9 |
| F5 grace15 | 8·88%·+2.93 | 6·83%·+1.22 | 1·100%·+2.00 | n/p | 15·+2.19·Σ+32.8 |

### HT — joint check: the top exits on the trades the observe-line filters keep (N: today → new, avg net %)

| trades | TP6/SL10/2880m DES | TP6/SL10/2880m SEEN | TPtoday/SL10/1440m DES | TPtoday/SL10/1440m SEEN | TPtoday/SL12/240m DES | TPtoday/SL12/240m SEEN |
|---|---|---|---|---|---|---|
| all | 256: -0.37→+0.27 | 161: -0.11→-0.35 | 256: -0.37→-0.19 | 161: -0.11→-0.45 | 256: -0.37→-0.16 | 161: -0.11→-0.64 |
| kept by 4 h slope ≤ 0 | 82: -0.28→+0.89 | 65: +0.38→-0.10 | 82: -0.28→+0.38 | 65: +0.38→-0.06 | 82: -0.28→+0.28 | 65: +0.38→-0.30 |
| kept by top_pos ≤ 1 | 33: +1.19→+2.32 | 14: +0.36→+1.48 | 33: +1.19→+1.02 | 14: +0.36→-0.18 | 33: +1.19→+1.29 | 14: +0.36→-0.47 |

### WILLY — heat maps (fixed maker TP × SL; cell = avg net %/trade · WR % / breakeven WR %; DESIGN 2025-01→2026-06 | SEEN 2026-07→11 | HOLDOUT-D 2023-H2)


**hold 4 h** (today's exit for reference: DESIGN -0.39 · 56/64 \| SEEN +1.02 \| D -0.18)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.40 · 57/64 \| +1.02 \| -0.19 | -0.59 · 63/72 \| +1.00 \| -0.06 | -0.67 · 67/76 \| +1.08 \| -0.12 | -0.67 · 69/78 \| +0.97 \| -0.08 | -0.62 · 70/78 \| +0.98 \| -0.13 |
| +2 % | -0.34 · 62/67 \| +0.07 \| -0.22 | -0.54 · 67/74 \| +0.10 \| -0.14 | -0.66 · 70/78 \| +0.27 \| -0.29 | -0.68 · 71/79 \| +0.19 \| -0.40 | -0.61 · 73/79 \| +0.31 \| -0.37 |
| +3 % | -0.16 · 55/58 \| +0.10 \| -0.03 | -0.42 · 61/65 \| +0.35 \| +0.02 | -0.53 · 64/69 \| +0.40 \| -0.14 | -0.55 · 65/71 \| +0.28 \| -0.23 | -0.50 · 66/71 \| +0.36 \| -0.20 |
| +4 % | -0.07 · 50/51 \| +0.20 \| -0.40 | -0.35 · 55/59 \| +0.61 \| -0.38 | -0.47 · 59/63 \| +0.65 \| -0.43 | -0.52 · 60/65 \| +0.48 \| -0.52 | -0.42 · 62/65 \| +0.48 \| -0.51 |
| +5 % | -0.04 · 45/45 \| +0.26 \| -0.21 | -0.35 · 50/54 \| +0.84 \| -0.27 | -0.47 · 54/58 \| +0.78 \| -0.33 | -0.53 · 55/60 \| +0.65 \| -0.44 | -0.36 · 57/60 \| +0.64 \| -0.43 |
| +6 % | -0.17 · 39/41 \| +0.18 \| -0.21 | -0.42 · 45/49 \| +0.76 \| -0.29 | -0.52 · 49/53 \| +0.83 \| -0.32 | -0.55 · 50/55 \| +0.71 \| -0.40 | -0.39 · 52/55 \| +0.71 \| -0.39 |

**hold 12 h** (today's exit for reference: DESIGN -0.39 · 56/64 \| SEEN +1.02 \| D -0.18)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.39 · 57/64 \| +1.02 \| -0.19 | -0.58 · 64/73 \| +1.02 \| -0.02 | -0.70 · 68/78 \| +1.11 \| -0.02 | -0.71 · 71/80 \| +1.02 \| -0.07 | -0.73 · 73/81 \| +1.02 \| -0.05 |
| +2 % | -0.33 · 62/67 \| +0.06 \| -0.20 | -0.52 · 68/75 \| +0.18 \| -0.10 | -0.67 · 72/79 \| +0.35 \| -0.25 | -0.72 · 74/81 \| +0.23 \| -0.39 | -0.74 · 76/82 \| +0.41 \| -0.34 |
| +3 % | -0.17 · 55/57 \| +0.09 \| +0.00 | -0.48 · 61/66 \| +0.28 \| -0.06 | -0.60 · 65/71 \| +0.34 \| -0.15 | -0.64 · 68/73 \| +0.15 \| -0.28 | -0.68 · 69/75 \| +0.27 \| -0.27 |
| +4 % | -0.06 · 49/50 \| +0.20 \| -0.40 | -0.38 · 56/59 \| +0.56 \| -0.43 | -0.53 · 60/65 \| +0.58 \| -0.36 | -0.58 · 63/67 \| +0.34 \| -0.53 | -0.57 · 65/69 \| +0.43 \| -0.53 |
| +5 % | -0.03 · 44/44 \| +0.27 \| -0.20 | -0.41 · 50/54 \| +0.67 \| -0.39 | -0.55 · 55/59 \| +0.56 \| -0.38 | -0.62 · 57/62 \| +0.37 \| -0.62 | -0.55 · 59/63 \| +0.42 \| -0.65 |
| +6 % | -0.19 · 38/40 \| +0.16 \| -0.11 | -0.50 · 45/50 \| +0.44 \| -0.42 | -0.63 · 50/55 \| +0.41 \| -0.44 | -0.68 · 53/58 \| +0.22 \| -0.67 | -0.57 · 55/59 \| +0.25 \| -0.66 |

**hold 24 h** (today's exit for reference: DESIGN -0.39 · 56/64 \| SEEN +1.02 \| D -0.18)

| TP | SL −4 % | SL −6 % | SL −8 % | SL −10 % | SL −12 % |
|---|---|---|---|---|---|
| TODAY-TP | -0.39 · 57/64 \| +1.02 \| -0.19 | -0.58 · 64/73 \| +1.02 \| -0.02 | -0.72 · 69/78 \| +1.10 \| -0.06 | -0.70 · 73/81 \| +1.07 \| -0.15 | -0.74 · 74/83 \| +1.10 \| -0.03 |
| +2 % | -0.32 · 62/67 \| +0.06 \| -0.21 | -0.51 · 69/75 \| +0.15 \| -0.18 | -0.73 · 72/80 \| +0.23 \| -0.21 | -0.78 · 76/82 \| +0.22 \| -0.37 | -0.85 · 77/84 \| +0.43 \| -0.19 |
| +3 % | -0.17 · 55/58 \| +0.09 \| -0.00 | -0.46 · 62/67 \| +0.24 \| -0.09 | -0.63 · 66/72 \| +0.12 \| -0.08 | -0.68 · 70/75 \| -0.01 \| -0.18 | -0.79 · 72/77 \| +0.20 \| -0.04 |
| +4 % | -0.06 · 50/50 \| +0.20 \| -0.41 | -0.35 · 57/60 \| +0.50 \| -0.46 | -0.57 · 61/66 \| +0.34 \| -0.42 | -0.61 · 65/70 \| +0.15 \| -0.68 | -0.69 · 67/72 \| +0.30 \| -0.60 |
| +5 % | -0.02 · 44/45 \| +0.27 \| -0.23 | -0.36 · 51/54 \| +0.60 \| -0.31 | -0.58 · 56/61 \| +0.28 \| -0.28 | -0.70 · 60/65 \| +0.13 \| -0.71 | -0.69 · 63/67 \| +0.30 \| -0.68 |
| +6 % | -0.18 · 38/40 \| +0.16 \| -0.21 | -0.45 · 46/50 \| +0.39 \| -0.43 | -0.70 · 51/57 \| +0.23 \| -0.44 | -0.78 · 55/61 \| +0.06 \| -0.87 | -0.78 · 58/63 \| +0.18 \| -0.85 |

### WILLY — full reading of the top / family-best cells (DESIGN unless marked)

| cell | N (D/S) | /day | WR | avg W / L | BE WR | DESIGN avg [day CI] | Δ/entry DES · SEEN · D | SEEN avg | D avg | best-5 % out | @0.20 DES/SEEN | Kelly floor | max loss | DCR X=1 %/day DES · SEEN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 426/119 | 0.80 | 56 | +1.86 / -3.29 | 64 | -0.39 [-0.74, -0.05] | +0.00 · +0.00 · +0.00 | +1.02 | -0.18 | -0.79 | -0.52 / +0.90 | -0.35 | -4.2 | -0.380 · +0.737 |
| F1 TPtoday/SL10/1440m | 426/119 | 0.80 | 73 | +1.60 / -6.87 | 81 | -0.70 [-1.30, -0.12] | -0.31 · +0.06 · +0.04 | +1.07 | -0.15 | -1.14 | -0.80 / +0.92 | -0.68 | -10.2 | -1.210 · +0.764 |
| F1 TP2/SL10/1440m | 426/119 | 0.80 | 76 | +2.02 / -9.43 | 82 | -0.78 [-1.27, -0.30] | -0.39 · -0.80 · -0.18 | +0.22 | -0.37 | -0.94 | -0.83 / +0.20 | -0.64 | -10.2 | -0.772 · +0.097 |
| F1 TP5/SL4/1440m | 426/119 | 0.80 | 44 | +5.03 / -4.04 | 45 | -0.02 [-0.45, +0.45] | +0.37 · -0.75 · -0.05 | +0.27 | -0.23 | -0.30 | -0.16 / +0.14 | -0.09 | -4.2 | -0.104 · +0.135 |
| F1 TP4/SL4/1440m | 426/119 | 0.80 | 50 | +4.02 / -4.06 | 50 | -0.06 [-0.44, +0.38] | +0.33 · -0.82 · -0.23 | +0.20 | -0.41 | -0.29 | -0.16 / +0.02 | -0.11 | -4.2 | -0.128 · +0.083 |
| F1 TP6/SL12/90m | 426/119 | 0.80 | 50 | +5.06 / -5.62 | 53 | -0.28 [-0.87, +0.27] | +0.11 · -0.53 · -0.14 | +0.49 | -0.32 | -0.64 | -0.46 / +0.38 | -0.16 | -12.2 | -0.384 · +0.283 |
| F1 TPtoday/SL4/720m | 426/119 | 0.80 | 57 | +1.85 / -3.33 | 64 | -0.39 [-0.73, -0.03] | +0.00 · +0.00 · -0.00 | +1.02 | -0.19 | -0.79 | -0.52 / +0.90 | -0.35 | -4.2 | -0.380 · +0.737 |
| F2 struct-base/240m | 265/86 | 0.50 | 71 | +1.45 / -4.67 | 76 | -0.31 [-0.75, +0.14] | +0.20 · -0.31 · -0.23 | +0.98 | -0.55 | -0.66 | -0.42 / +0.83 | -0.44 | -11.0 | -0.187 · +0.553 |
| F2 struct-vwap/240m | 424/119 | 0.79 | 34 | +2.30 / -1.53 | 40 | -0.23 [-0.47, +0.04] | +0.16 · -0.55 · +0.02 | +0.46 | -0.16 | -0.62 | -0.39 / +0.32 | -0.17 | -8.6 | -0.212 · +0.282 |
| F3 BE+1 | 426/119 | 0.80 | 41 | +1.85 / -1.87 | 50 | -0.35 [-0.63, -0.03] | +0.04 · -0.18 · +0.03 | +0.83 | -0.16 | -0.69 | -0.54 / +0.72 | -0.28 | -4.1 | -0.325 · +0.589 |
| F4 trail+3/3/1440m | 426/119 | 0.80 | 51 | +3.22 / -3.65 | 53 | -0.15 [-0.59, +0.28] | +0.24 · -0.76 · +0.33 | +0.26 | +0.15 | -0.74 | -0.36 / +0.09 | -0.15 | -4.2 | -0.214 · +0.117 |
| F5 grace15 | 426/119 | 0.80 | 60 | +1.88 / -4.00 | 68 | -0.48 [-0.94, -0.05] | -0.09 · +0.17 · +0.29 | +1.19 | +0.11 | -0.90 | -0.61 / +1.03 | -0.41 | -15.1 | -0.530 · +0.881 |

### WILLY — per half-year (N · avg net %)

| cell | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D |
|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 92·-0.08 | 175·-0.69 | 159·-0.23 | 119·+1.02 | 99·-0.18 |
| F1 TPtoday/SL10/1440m | 92·-0.01 | 175·-1.27 | 159·-0.48 | 119·+1.07 | 99·-0.15 |
| F1 TP2/SL10/1440m | 92·-0.38 | 175·-0.98 | 159·-0.78 | 119·+0.22 | 99·-0.37 |
| F1 TP5/SL4/1440m | 92·-0.24 | 175·-0.13 | 159·+0.24 | 119·+0.27 | 99·-0.23 |
| F1 TP4/SL4/1440m | 92·+0.01 | 175·-0.10 | 159·-0.05 | 119·+0.20 | 99·-0.41 |
| F1 TP6/SL12/90m | 92·-0.08 | 175·-0.57 | 159·-0.09 | 119·+0.49 | 99·-0.32 |
| F1 TPtoday/SL4/720m | 92·-0.08 | 175·-0.69 | 159·-0.23 | 119·+1.02 | 99·-0.19 |
| F2 struct-base/240m | 51·-0.10 | 107·-0.34 | 107·-0.37 | 86·+0.98 | 75·-0.55 |
| F2 struct-vwap/240m | 92·-0.00 | 175·-0.43 | 157·-0.13 | 119·+0.46 | 98·-0.16 |
| F3 BE+1 | 92·-0.25 | 175·-0.44 | 159·-0.30 | 119·+0.83 | 99·-0.16 |
| F4 trail+3/3/1440m | 92·+0.17 | 175·-0.20 | 159·-0.28 | 119·+0.26 | 99·+0.15 |
| F5 grace15 | 92·-0.11 | 175·-0.87 | 159·-0.28 | 119·+1.19 | 99·+0.11 |

### WILLY — live-stopped split (DESIGN + SEEN; Δ vs today per trade, summed)

| cell | stopped today | saved N · pts | deeper N · pts | not-stopped changed N · pts | total Δ pts |
|---|---|---|---|---|---|
| F1 TPtoday/SL10/1440m | 172 | 82 · +385 | 90 · -512 | 8 · +1 | -127 |
| F1 TP2/SL10/1440m | 172 | 79 · +467 | 93 · -519 | 373 · -208 | -260 |
| F1 TP5/SL4/1440m | 172 | 3 · +27 | 0 · +0 | 373 · +43 | +70 |
| F1 TP4/SL4/1440m | 172 | 4 · +32 | 0 · +0 | 373 · +12 | +44 |
| F1 TP6/SL12/90m | 172 | 76 · +328 | 95 · -401 | 365 · +56 | -17 |
| F1 TPtoday/SL4/720m | 172 | 0 · +0 | 0 · +0 | 8 · +0 | +0 |
| F2 struct-base/240m | 172 | 119 · +488 | 53 · -149 | 101 · -291 | +48 |
| F2 struct-vwap/240m | 172 | 159 · +399 | 13 · -22 | 160 · -373 | +4 |
| F3 BE+1 | 172 | 51 · +187 | 0 · +0 | 98 · -191 | -4 |
| F4 trail+3/3/1440m | 172 | 8 · +40 | 0 · +0 | 369 · -29 | +11 |
| F5 grace15 | 172 | 24 · +131 | 54 · -150 | 0 · +0 | -19 |

### WILLY — master per batch, kept fills re-priced (N · WR · avg net %; PAPER1118 = the 11-18 export, EVAA +2.00 at 3 min, n/p = not priceable yet)

| cell | B19 | B20 | B22 | B23 | B24 | B25 | B26 | B27 | all kept |
|---|---|---|---|---|---|---|---|---|---|
| F1 TPtoday/SL4/90m | 3·100%·+1.78 | 1·100%·+1.26 | 5·60%·+0.55 | 1·100%·+1.25 | 1·100%·+7.42 | 6·83%·+6.92 | 4·50%·+2.34 | 3·33%·-0.96 | 24·+2.75·Σ+66.0 |
| F1 TPtoday/SL10/1440m | 3·100%·+1.78 | 1·100%·+1.26 | 5·60%·+0.55 | 1·100%·+1.25 | 1·100%·+7.42 | 6·83%·+6.92 | 4·50%·+0.84 | 3·67%·+0.49 | 24·+2.68·Σ+64.4 |
| F1 TP2/SL10/1440m | 3·100%·+2.01 | 1·100%·+2.01 | 5·100%·+2.02 | 1·100%·+2.01 | 1·100%·+2.00 | 6·83%·-0.01 | 4·50%·-4.02 | 3·67%·+1.13 | 24·+0.39·Σ+9.4 |
| F1 TP5/SL4/1440m | 3·67%·+1.99 | 1·100%·+5.01 | 5·20%·-2.28 | 1·0%·-4.10 | 1·100%·+5.01 | 6·50%·+0.45 | 4·25%·-1.81 | 3·0%·-3.94 | 24·-0.66·Σ-15.8 |
| F1 TP4/SL4/1440m | 3·67%·+1.32 | 1·100%·+4.01 | 5·60%·+0.76 | 1·0%·-4.10 | 1·100%·+4.00 | 6·50%·-0.05 | 4·50%·-0.04 | 3·0%·-3.94 | 24·-0.02·Σ-0.6 |
| F1 TP6/SL12/90m | 3·67%·+3.21 | 1·100%·+6.00 | 5·60%·+2.84 | 1·0%·-0.15 | 1·100%·+6.00 | 6·67%·+1.20 | 4·50%·-3.21 | 3·0%·-4.04 | 24·+0.75·Σ+18.0 |
| F1 TPtoday/SL4/720m | 3·100%·+1.78 | 1·100%·+1.26 | 5·60%·+0.55 | 1·100%·+1.25 | 1·100%·+7.42 | 6·83%·+6.92 | 4·50%·+2.34 | 3·33%·-0.96 | 24·+2.75·Σ+66.0 |
| F2 struct-base/240m | 3·100%·+1.78 | n/p | 3·67%·+0.51 | 1·100%·+1.25 | n/p | 5·80%·+8.27 | 2·50%·+0.63 | 3·67%·+0.49 | 17·+3.07·Σ+52.2 |
| F2 struct-vwap/240m | 3·67%·+1.37 | 1·100%·+1.26 | 5·40%·+0.35 | 1·0%·-0.93 | 1·0%·-2.55 | 6·50%·+3.92 | 4·50%·+2.53 | 3·0%·-0.94 | 24·+1.44·Σ+34.5 |
| F3 BE+1 | 3·100%·+1.78 | 1·0%·-0.66 | 5·40%·+0.46 | 1·100%·+1.25 | 1·0%·-0.10 | 6·67%·+3.06 | 4·50%·+2.34 | 3·33%·-0.96 | 24·+1.38·Σ+33.0 |
| F4 trail+3/3/1440m | 3·67%·+0.11 | 1·100%·+3.41 | 5·60%·-0.22 | 1·0%·-4.10 | 1·100%·+0.26 | 6·50%·+5.24 | 4·50%·-1.25 | 3·33%·-2.31 | 24·+0.76·Σ+18.3 |
| F5 grace15 | 3·100%·+1.78 | 1·100%·+1.26 | 5·60%·+0.55 | 1·100%·+1.25 | 1·100%·+7.42 | 6·83%·+6.92 | 4·50%·+1.86 | 3·33%·-0.96 | 24·+2.67·Σ+64.1 |
## Winners vs losers under the wide-stop exit (operator request; CLAUDE.md "W/L = every variable")
**Labels.** Net per trade under:
- (a) the headline TODAY-TP / SL −10 % / 24 h;
- (b) the best wide-stop (SL ≥ 6) grid cell on DESIGN. That is HT TP6/SL10/48 h and WILLY TP6/SL12/90 m.

Each sleeve is screened separately (`scripts/willy_exit_v2_wl.py`).

**Coverage first** (table at the end of this section). 45 entry-time variables are tested.
- **Fully scored (≥ 97 %):** everything except the variables listed next.
- **Partly scored:**
  - spot share and spot share change: base 55 % (coins with a spot market only);
  - BTC 1d EMA50/200 and 1d return: base 86 %, D 61 % (EMA200 warm-up);
  - R (turnover ÷ market cap): D 70 %.
- **Unscored on the master:** `s4h_ema50`, `late`, `disloc`.

**Blind spots (not testable here):**
- order-book depth and imbalance at entry (no history);
- liquidations;
- news and listings;
- per-coin market cap in 2023 (D routing);
- the post-entry columns (peak, trough, duration), which are descriptive only and excluded by rule;
- the operator's dashboard W/L columns that come from the live stamp set only (quality score, pattern C/W flags, entry_book_*), which are not in the rebuild.

**Screen.**
- **1D:** every variable at its median, lower / upper tercile, sign (where signed) and theory value, both sides: 238 cuts.
- **2D:** every variable pair at each variable's best 1D cut, OR and AND: 1,980 masks.
- **Ranking:** consistency (DESIGN half-years where the blocked bucket < the rest), then DESIGN Δ.
- **Null:** 300 within-half label shuffles. Each shuffle re-runs the whole screen, including re-choosing the best cuts. The statistic is the max.

**Result: no separator beats the null.**

| label | HT p | WILLY p |
|---|---|---|
| (a) headline | 0.08 | 0.30 |
| (b) best wide-stop cell | 0.64 | 0.36 |

**Path-label version.** Among the trades stopped at −4 % today, does any variable separate those that recovered to the TP from those that hit −10 % anyway?
- HT: 75 recovered vs 66 hit −10 %. Max |AUC − 0.5| = 0.157, null 95th 0.165, p = 0.08.
- WILLY: 80 vs 84. Max |AUC − 0.5| = 0.140, p = 0.10.
- So no variable tells, at entry, which stop will recover.

**Theory check of the top rows (plain English)** — none is shippable, and each fails somewhere:
- **HT (a): `trend7d > 0` blocked.** This keeps only coins that were DOWN over the prior 7 days.
  - The theory said uptrend = buyers arriving, so this is the opposite of the theory.
  - D contradicts it (kept −0.48 vs blocked −0.31).
  - It leaves 10–42 trades per half.
  - **No theory, no ship.**
- **HT (a/b): `top_pos > 1` blocked (top traders net long → fade).** This one has a theory: a crowded long book fades.
  - It is the same line as WILLY_DEEP's HT observe line.
  - It is consistent in 2025–26 under both exits.
  - D contradicts it again (kept −0.63 / −1.00 vs blocked −0.15 / −0.31).
  - On the master it blocks 13 of 15 real fills, which averaged +3.35.
  - **Stays OBSERVE, unchanged.**
- **HT (b): `glob_acc > 1.075` / `top_acc > 1.333` blocked.** Same crowded-long theory.
  - Here D agrees: kept +1.30 / +0.88 vs blocked −1.59 / −2.60.
  - But this is the 4th–5th row of a screen whose best is at p = 0.64.
  - On the master the blocked fills are positive (+4.05 / +2.87).
  - **Watchlist only.** It is the same family as `top_pos`, so it is not a new independent line.
- **HT (a): `g1h > 0` blocked** (keep coins below their 1 h EMA200 gap). This is against the theory. D agrees (kept +0.44 vs −0.43), but the master's blocked fills are better (+4.06). No.
- **WILLY (a): `top_pos_chg < 0` blocked** (block when top traders were reducing longs in the last hour). No clear theory side.
  - It flips in 2026-H2: the blocked bucket +1.40 beats the rest +0.60.
  - On the master the blocked fills are the winners (+3.41).
  - No.
- **WILLY (a): `premium_chg60 < 0` / `spot_share_chg < 0.004` blocked** (premium falling / perp-led = fade). These are on the theory side.
  - Both flip in 2026-H2 (the blocked bucket wins).
  - D is flat or contradicts.
  - The master's blocked fills are better (+2.32 / +8.01).
  - No.
- **WILLY (b): `adx < 44` blocked** (block weak-trend entries). This is against the theory side.
  - D contradicts it (blocked −0.16 vs kept −0.52).
  - No.

#### HT (headline TODAY-TP / SL −10 % / 24 h: `F1 TPtoday/SL10/1440m`) — 238 1D cuts + 1980 2D masks; real best Δ +1.89 vs null 95th +1.94 → **family-wise p = 0.08**

Top 1D (B = blocked bucket, R = rest; N·WR·avg per half-year; cons = DESIGN halves where B < R):

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| trend7d > 0 (sign) | no | 3 | 1.351 | B 45·89%·+0.02 / R 7·100%·+0.56 | B 76·84%·-0.25 / R 15·100%·+2.06 | B 93·80%·-0.84 / R 20·95%·+0.70 | B 119·78%·-0.94 / R 42·90%·+0.95 | B 191·-0.31 / R 10·-0.48 | 15·+3.28 → 9·+4.33 (blocked 6·+1.72) |
| top_pos > 1 (theory) | yes | 3 | 1.206 | B 38·89%·-0.07 / R 14·93%·+0.53 | B 77·84%·-0.20 / R 14·100%·+1.93 | B 108·82%·-0.58 / R 5·80%·-0.15 | B 145·81%·-0.51 / R 16·88%·+0.12 | B 128·-0.15 / R 73·-0.63 | 15·+3.28 → 2·+2.88 (blocked 13·+3.35) |
| g1h > 0 (sign) | no | 3 | 1.152 | B 42·88%·-0.12 / R 10·100%·+0.96 | B 77·87%·+0.01 / R 14·86%·+0.78 | B 84·77%·-1.13 / R 29·97%·+1.06 | B 105·80%·-0.68 / R 56·84%·-0.01 | B 175·-0.43 / R 26·+0.44 | 15·+3.28 → 10·+2.90 (blocked 5·+4.06) |
| glob_acc > 1.075 (med) | yes | 3 | 0.582 | B 21·90%·-0.19 / R 31·90%·+0.28 | B 47·81%·-0.59 / R 44·93%·+0.89 | B 60·78%·-1.11 / R 53·87%·+0.05 | B 73·78%·-0.79 / R 88·84%·-0.16 | B 129·-0.20 / R 72·-0.54 | 15·+3.28 → 7·+3.72 (blocked 8·+2.90) |
| premium < -0.02703 (med) | yes | 3 | 0.55 | B 33·88%·-0.08 / R 19·95%·+0.38 | B 43·81%·-0.66 / R 48·92%·+0.84 | B 52·77%·-1.21 / R 61·87%·-0.01 | B 66·83%·-0.31 / R 95·80%·-0.54 | B 110·-0.52 / R 91·-0.09 | 15·+3.28 → 8·+3.87 (blocked 7·+2.61) |
| s4h_ema20 > 1.388 (t2) | no | 3 | 0.432 | B 16·88%·-0.26 / R 36·92%·+0.24 | B 33·85%·-0.11 / R 58·88%·+0.26 | B 36·69%·-2.28 / R 77·88%·+0.24 | B 41·76%·-1.47 / R 120·83%·-0.10 | B 31·-1.05 / R 170·-0.19 | 15·+3.28 → 13·+3.02 (blocked 2·+5.00) |

Top 2D:

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| g1h > 0 (sign) OR breadth > 0.6309 (t2) | no | 3 | 1.895 | B 46·89%·-0.01 / R 6·100%·+0.84 | B 83·86%·-0.13 / R 8·100%·+2.78 | B 90·78%·-1.11 / R 23·100%·+1.56 | B 114·81%·-0.53 / R 47·83%·-0.25 | B 183·-0.41 / R 18·+0.58 | 15·+3.28 → 7·+2.46 (blocked 8·+4.00) |
| gap813 < 2.09 (theory) OR btc_1d_ret < -0.1767 (med) | no | 3 | 1.708 | B 45·89%·-0.13 / R 7·100%·+1.50 | B 76·86%·-0.28 / R 15·93%·+2.21 | B 102·82%·-0.69 / R 11·82%·+0.60 | B 130·84%·-0.27 / R 31·71%·-1.18 | B 167·-0.18 / R 34·-1.03 | 15·+3.28 → 3·+5.32 (blocked 12·+2.77) |
| s4h_ema20 > 1.388 (t2) OR trend7d > 0 (sign) | no | 3 | 1.653 | B 45·89%·+0.02 / R 7·100%·+0.56 | B 78·85%·-0.21 / R 13·100%·+2.15 | B 95·79%·-0.92 / R 18·100%·+1.33 | B 124·79%·-0.87 / R 37·89%·+0.97 | B 191·-0.31 / R 10·-0.48 | 15·+3.28 → 8·+3.87 (blocked 7·+2.62) |
| gap813 < 2.09 (theory) OR premium_chg15 < -0.1023 (t1) | no | 3 | 1.637 | B 47·89%·-0.04 / R 5·100%·+1.30 | B 79·87%·-0.04 / R 12·83%·+1.26 | B 96·80%·-0.95 / R 17·94%·+1.63 | B 134·84%·-0.22 / R 27·67%·-1.57 | B 176·-0.16 / R 25·-1.44 | 15·+3.28 → 3·+3.91 (blocked 12·+3.13) |

#### HT (best wide-stop cell: `F1 TP6/SL10/2880m`) — 238 1D cuts + 1980 2D masks; real best Δ +2.82 vs null 95th +3.85 → **family-wise p = 0.64**

Top 1D (B = blocked bucket, R = rest; N·WR·avg per half-year; cons = DESIGN halves where B < R):

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| top_pos > 1 (theory) | yes | 3 | 2.046 | B 38·63%·+0.20 / R 14·71%·+1.73 | B 77·64%·+0.37 / R 14·86%·+3.74 | B 108·59%·-0.40 / R 5·60%·-0.01 | B 145·57%·-0.61 / R 16·75%·+2.05 | B 128·-0.31 / R 73·-1.00 | 15·+5.07 → 2·+6.02 (blocked 13·+4.92) |
| run_pct < 13.13 (med) | no | 3 | 1.073 | B 36·61%·+0.13 / R 16·75%·+1.71 | B 33·55%·-0.84 / R 58·74%·+1.88 | B 59·53%·-1.34 / R 54·67%·+0.67 | B 74·57%·-0.70 / R 87·61%·-0.04 | B 121·-0.44 / R 80·-0.74 | 15·+5.07 → 9·+6.18 (blocked 6·+3.39) |
| vs_vwap < 0.889 (med) | yes | 3 | 0.935 | B 35·63%·+0.39 / R 17·71%·+1.08 | B 40·55%·-0.87 / R 51·76%·+2.27 | B 53·55%·-1.20 / R 60·63%·+0.34 | B 80·57%·-0.51 / R 81·60%·-0.18 | B 110·-0.75 / R 91·-0.32 | 15·+5.07 → 9·+4.43 (blocked 6·+6.02) |
| glob_acc > 1.075 (med) | yes | 3 | 0.917 | B 21·62%·-0.51 / R 31·68%·+1.38 | B 47·64%·+0.30 / R 44·70%·+1.52 | B 60·53%·-1.43 / R 53·66%·+0.81 | B 73·51%·-1.76 / R 88·66%·+0.83 | B 129·-1.59 / R 72·+1.30 | 15·+5.07 → 7·+6.23 (blocked 8·+4.05) |
| top_acc > 1.333 (t2) | yes | 3 | 0.89 | B 15·67%·+0.42 / R 37·65%·+0.69 | B 33·58%·-0.71 / R 58·72%·+1.80 | B 37·43%·-3.02 / R 76·67%·+0.91 | B 58·50%·-1.84 / R 103·64%·+0.50 | B 83·-2.60 / R 118·+0.88 | 15·+5.07 → 10·+6.16 (blocked 5·+2.87) |
| premium < -0.02703 (med) | yes | 3 | 0.856 | B 33·61%·+0.12 / R 19·74%·+1.48 | B 43·60%·-0.18 / R 48·73%·+1.85 | B 52·52%·-1.36 / R 61·66%·+0.46 | B 66·58%·-0.28 / R 95·60%·-0.39 | B 110·-0.12 / R 91·-1.08 | 15·+5.07 → 8·+6.01 (blocked 7·+3.98) |

Top 2D:

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| g1h > 11.95 (med) OR run_pct < 13.13 (med) | no | 3 | 2.821 | B 45·60%·-0.13 / R 7·100%·+5.41 | B 60·60%·-0.18 / R 31·81%·+2.96 | B 90·54%·-1.13 / R 23·78%·+2.57 | B 119·60%·-0.27 / R 42·57%·-0.56 | B 147·-0.36 / R 54·-1.10 | 15·+5.07 → 7·+6.23 (blocked 8·+4.05) |
| breadth > 0.4613 (med) OR btc_1h_slope < 0 (sign) | no | 3 | 2.738 | B 44·64%·+0.22 / R 8·75%·+2.80 | B 81·65%·+0.65 / R 10·80%·+2.83 | B 97·56%·-0.97 / R 16·81%·+3.23 | B 131·63%·+0.37 / R 30·40%·-3.47 | B 174·-0.70 / R 27·+0.36 | 15·+5.07 → 0·+nan (blocked 15·+5.07) |
| s4h_ema50 > 0.6178 (med) OR run_pct < 13.13 (med) | no | 3 | 2.615 | B 47·62%·+0.13 / R 5·100%·+5.17 | B 64·62%·+0.21 / R 27·78%·+2.51 | B 88·53%·-1.30 / R 25·80%·+2.84 | B 113·60%·-0.23 / R 48·56%·-0.63 | B 143·-0.20 / R 58·-1.45 | 15·+5.07 → 9·+6.18 (blocked 6·+3.39) |
| top_acc_chg < 0 (sign) OR vol_decay < 1 (med) | no | 3 | 2.609 | B 45·64%·+0.39 / R 7·71%·+2.06 | B 77·64%·+0.37 / R 14·86%·+3.77 | B 93·56%·-1.01 / R 20·75%·+2.55 | B 132·59%·-0.30 / R 29·59%·-0.55 | B 194·-0.63 / R 7·+1.43 | 15·+5.07 → 4·+2.47 (blocked 11·+6.01) |

#### HT path label — stopped at −4 % today: recovered to the TP (75) vs hit −10 % anyway (66) under TODAY-TP / SL −10 / 24 h. max |AUC − 0.5| = 0.157, null 95th 0.165 → **p = 0.08**

| var | theory_fade_side | cov | rec_med | dead_med | AUC | dirs_25H1_25H2_26H1_26H2 |
|---|---|---|---|---|---|---|
| spot_share_chg | lo | 0.51 | 0.0116 | -0.0115 | 0.657 | ·+++ |
| top_acc_chg | hi | 0.99 | -0.0272 | -0.0683 | 0.625 | −+++ |
| vol_decay | lo | 1.0 | 1.0 | 1.0 | 0.403 | −−−− |
| adx | hi | 1.0 | 36.5304 | 42.1388 | 0.404 | −−−− |
| trend7d | lo | 1.0 | 25.0272 | 32.5433 | 0.408 | +−−− |
| glob_acc_chg | hi | 0.99 | 0.0001 | -0.0231 | 0.587 | −+++ |
| bar_ret | nan | 1.0 | -0.5432 | -0.8505 | 0.582 | −+++ |
| s4h_ema50 | lo | 1.0 | 0.3514 | 0.6728 | 0.425 | −−−− |
| gap813 | hi | 1.0 | 1.3186 | 1.4394 | 0.426 | +−−+ |
| vol_mult | lo | 1.0 | 39.5038 | 45.6066 | 0.427 | +−−− |

#### WILLY (headline TODAY-TP / SL −10 % / 24 h: `F1 TPtoday/SL10/1440m`) — 238 1D cuts + 1980 2D masks; real best Δ +1.83 vs null 95th +2.25 → **family-wise p = 0.30**

Top 1D (B = blocked bucket, R = rest; N·WR·avg per half-year; cons = DESIGN halves where B < R):

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| top_pos_chg < 0 (sign) | no | 3 | 0.876 | B 70·79%·-0.39 / R 22·91%·+1.19 | B 137·65%·-1.59 / R 38·71%·-0.13 | B 91·74%·-0.84 / R 68·76%·+0.01 | B 71·80%·+1.40 / R 48·85%·+0.60 | B 76·-0.24 / R 23·+0.17 | 24·+2.68 → 8·+1.23 (blocked 16·+3.41) |
| top_pos_chg < -0.05188 (med) | no | 3 | 0.589 | B 49·78%·-0.59 / R 43·86%·+0.64 | B 105·62%·-1.81 / R 70·73%·-0.47 | B 58·71%·-0.97 / R 101·77%·-0.19 | B 45·84%·+1.77 / R 74·81%·+0.65 | B 31·+0.60 / R 68·-0.48 | 24·+2.68 → 15·+1.72 (blocked 9·+4.28) |
| top_acc_chg < -0.1153 (med) | no | 3 | 0.516 | B 64·78%·-0.45 / R 28·89%·+0.99 | B 95·61%·-1.82 / R 80·72%·-0.62 | B 53·72%·-1.09 / R 106·76%·-0.17 | B 48·77%·+0.11 / R 71·86%·+1.72 | B 83·-0.19 / R 16·+0.06 | 24·+2.68 → 16·+3.56 (blocked 8·+0.92) |
| premium_chg60 < 0 (sign) | yes | 3 | 0.488 | B 70·80%·-0.33 / R 22·86%·+0.99 | B 124·65%·-1.37 / R 51·69%·-1.04 | B 105·73%·-0.76 / R 54·78%·+0.07 | B 56·86%·+1.84 / R 63·79%·+0.39 | B 68·-0.14 / R 31·-0.16 | 24·+2.68 → 9·+3.29 (blocked 15·+2.32) |
| spot_share_chg < 0.003913 (med) | yes | 3 | 0.454 | B 38·82%·-0.10 / R 54·81%·+0.05 | B 63·52%·-2.71 / R 112·74%·-0.46 | B 51·73%·-1.10 / R 108·76%·-0.18 | B 35·89%·+1.66 / R 84·80%·+0.83 | B 41·+0.45 / R 58·-0.56 | 24·+2.68 → 17·+0.49 (blocked 7·+8.01) |
| top_acc_chg < -0.2397 (t1) | no | 3 | 0.4 | B 54·78%·-0.38 / R 38·87%·+0.50 | B 56·54%·-2.56 / R 119·72%·-0.66 | B 31·65%·-1.59 / R 128·77%·-0.21 | B 21·76%·-0.25 / R 98·84%·+1.36 | B 52·+0.10 / R 47·-0.42 | 24·+2.68 → 21·+2.95 (blocked 3·+0.84) |

Top 2D:

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| bar_ret > -0.7321 (med) OR top_pos_chg < 0 (sign) | no | 3 | 1.833 | B 81·80%·-0.24 / R 11·91%·+1.67 | B 162·66%·-1.38 / R 13·69%·+0.14 | B 126·72%·-0.95 / R 33·85%·+1.34 | B 99·84%·+1.20 / R 20·75%·+0.43 | B 86·-0.31 / R 13·+0.97 | 24·+2.68 → 4·+1.75 (blocked 20·+2.87) |
| atr < 1.966 (med) OR top_pos_chg < 0 (sign) | no | 3 | 1.538 | B 85·80%·-0.17 / R 7·100%·+1.92 | B 156·66%·-1.48 / R 19·68%·+0.45 | B 119·71%·-0.91 / R 40·85%·+0.83 | B 97·85%·+1.34 / R 22·73%·-0.09 | B 93·-0.25 / R 6·+1.54 | 24·+2.68 → 5·+1.43 (blocked 19·+3.01) |
| gap813 > 1.725 (med) OR premium_chg60 < 0 (sign) | yes | 3 | 1.495 | B 81·80%·-0.10 / R 11·91%·+0.63 | B 154·64%·-1.49 / R 21·81%·+0.31 | B 129·72%·-0.86 / R 30·87%·+1.18 | B 79·84%·+1.34 / R 40·80%·+0.55 | B 81·-0.06 / R 18·-0.54 | 24·+2.68 → 8·+3.54 (blocked 16·+2.25) |
| premium_chg15 < -0.08357 (med) OR top_pos_chg < 0 (sign) | no | 3 | 1.46 | B 82·80%·-0.20 / R 10·90%·+1.49 | B 156·63%·-1.53 / R 19·89%·+0.82 | B 121·72%·-0.79 / R 38·84%·+0.53 | B 80·80%·+1.32 / R 39·87%·+0.57 | B 84·-0.40 / R 15·+1.26 | 24·+2.68 → 5·+3.74 (blocked 19·+2.40) |

#### WILLY (best wide-stop cell: `F1 TP6/SL12/90m`) — 238 1D cuts + 1980 2D masks; real best Δ +2.33 vs null 95th +3.12 → **family-wise p = 0.36**

Top 1D (B = blocked bucket, R = rest; N·WR·avg per half-year; cons = DESIGN halves where B < R):

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| adx < 44.42 (med) | no | 3 | 0.637 | B 45·33%·-1.05 / R 47·51%·+0.85 | B 82·48%·-1.26 / R 93·58%·+0.05 | B 86·49%·-0.52 / R 73·53%·+0.42 | B 76·49%·+0.09 / R 43·56%·+1.18 | B 54·-0.16 / R 45·-0.52 | 24·+0.75 → 12·+2.76 (blocked 12·-1.26) |
| top_pos_chg < 0 (sign) | no | 3 | 0.525 | B 70·40%·-0.30 / R 22·50%·+0.62 | B 137·52%·-0.73 / R 38·58%·+0.02 | B 91·48%·-0.34 / R 68·54%·+0.25 | B 71·54%·+0.78 / R 48·48%·+0.05 | B 76·-0.71 / R 23·+0.96 | 24·+0.75 → 8·-3.19 (blocked 16·+2.72) |
| spot_share_chg < 0.003913 (med) | yes | 3 | 0.394 | B 38·39%·-0.26 / R 54·44%·+0.05 | B 63·46%·-1.49 / R 112·57%·-0.04 | B 51·43%·-0.92 / R 108·55%·+0.30 | B 35·54%·-0.26 / R 84·50%·+0.80 | B 41·-0.26 / R 58·-0.36 | 24·+0.75 → 17·-0.53 (blocked 7·+3.85) |
| spot_share_chg < 0 (sign) | yes | 3 | 0.379 | B 36·36%·-0.56 / R 56·46%·+0.23 | B 59·47%·-1.19 / R 116·56%·-0.25 | B 47·40%·-1.21 / R 112·55%·+0.38 | B 33·58%·-0.04 / R 86·49%·+0.69 | B 38·-0.57 / R 61·-0.17 | 24·+0.75 → 18·-0.66 (blocked 6·+4.97) |
| spike_age_h < 0.1667 (med) | no | 3 | 0.367 | B 24·33%·-1.73 / R 68·46%·+0.51 | B 63·48%·-1.50 / R 112·56%·-0.04 | B 41·46%·-0.23 / R 118·53%·-0.04 | B 41·56%·+0.91 / R 78·49%·+0.26 | B 29·-0.10 / R 70·-0.41 | 24·+0.75 → 17·+0.81 (blocked 7·+0.60) |
| spike_age_h < 0.1667 (t1) | no | 3 | 0.367 | B 24·33%·-1.73 / R 68·46%·+0.51 | B 63·48%·-1.50 / R 112·56%·-0.04 | B 41·46%·-0.23 / R 118·53%·-0.04 | B 41·56%·+0.91 / R 78·49%·+0.26 | B 29·-0.10 / R 70·-0.41 | 24·+0.75 → 17·+0.81 (blocked 7·+0.60) |

Top 2D:

| candidate | theory | cons | delta | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | HOLDOUT-D | master before→after |
|---|---|---|---|---|---|---|---|---|---|
| premium_chg15 > -0.08357 (med) OR premium_chg60 < -0.1557 (med) | no | 3 | 2.333 | B 80·40%·-0.31 / R 12·58%·+1.45 | B 146·49%·-1.26 / R 29·76%·+2.91 | B 138·50%·-0.28 / R 21·57%·+1.21 | B 102·51%·+0.60 / R 17·53%·-0.18 | B 77·+0.17 / R 22·-2.04 | 24·+0.75 → 4·+1.01 (blocked 20·+0.70) |
| top_pos_chg < 0 (sign) OR flat14 > 71.86 (med) | no | 3 | 1.943 | B 81·38%·-0.51 / R 11·73%·+3.07 | B 163·52%·-0.74 / R 12·75%·+1.75 | B 140·50%·-0.21 / R 19·58%·+0.79 | B 99·53%·+0.65 / R 20·45%·-0.33 | B 82·-0.55 / R 17·+0.77 | 24·+0.75 → 1·-2.90 (blocked 23·+0.91) |
| hour < 11 (med) OR taker_30 < 1.035 (med) | no | 3 | 1.764 | B 72·42%·-0.35 / R 20·45%·+0.91 | B 127·46%·-1.20 / R 48·71%·+1.12 | B 122·46%·-0.80 / R 37·68%·+2.27 | B 85·52%·+0.47 / R 34·50%·+0.53 | B 70·-0.41 / R 29·-0.10 | 24·+0.75 → 8·+2.96 (blocked 16·-0.36) |
| hour < 11 (med) OR breadth < 0.5259 (med) | no | 3 | 1.684 | B 66·42%·-0.32 / R 26·42%·+0.55 | B 134·49%·-1.15 / R 41·66%·+1.35 | B 122·46%·-0.74 / R 37·68%·+2.06 | B 95·54%·+0.73 / R 24·42%·-0.47 | B 70·-0.28 / R 29·-0.43 | 24·+0.75 → 3·+1.14 (blocked 21·+0.69) |

#### WILLY path label — stopped at −4 % today: recovered to the TP (80) vs hit −10 % anyway (84) under TODAY-TP / SL −10 / 24 h. max |AUC − 0.5| = 0.140, null 95th 0.153 → **p = 0.10**

| var | theory_fade_side | cov | rec_med | dead_med | AUC | dirs_25H1_25H2_26H1_26H2 |
|---|---|---|---|---|---|---|
| spot_share | lo | 0.73 | 0.209 | 0.1824 | 0.64 | ++++ |
| run_pct | hi | 1.0 | 14.6334 | 19.2193 | 0.379 | +−−− |
| top_pos_chg | hi | 0.99 | -0.0313 | -0.0824 | 0.605 | ++−− |
| gap813 | hi | 1.0 | 1.7002 | 2.1684 | 0.396 | +−−− |
| s4h_ema20 | lo | 1.0 | 0.6691 | 0.2924 | 0.599 | −+++ |
| top_acc_chg | hi | 0.99 | -0.1295 | -0.193 | 0.596 | ++++ |
| glob_acc_chg | hi | 0.99 | -0.0778 | -0.1418 | 0.592 | −+++ |
| atr | hi | 1.0 | 2.1146 | 2.3896 | 0.419 | +−−− |
| s4h_ema50 | lo | 1.0 | 0.338 | 0.1482 | 0.581 | −+++ |
| premium_chg60 | lo | 0.99 | -0.0738 | -0.2687 | 0.58 | +++− |

#### Coverage (share of trades scored)

| var | base_HT | D_HT | master_HT | base_WILLY | D_WILLY | master_WILLY |
|---|---|---|---|---|---|---|
| gap813 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| adx | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| vs_vwap | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| vol_mult | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| atr | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| g1h | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| s4h_ema20 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| s4h_ema50 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| bar_ret | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| off_peak | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.58 |
| run_pct | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| late | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| wait_bars | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| disloc | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| vol24 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| R | 1.0 | 0.7 | 0.94 | 1.0 | 0.65 | 1.0 |
| hour | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| premium | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| premium_chg15 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| premium_chg60 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| spot_share | 0.55 | 0.86 | 0.31 | 0.7 | 0.8 | 0.71 |
| spot_share_chg | 0.55 | 0.86 | 0.31 | 0.7 | 0.8 | 0.71 |
| oi_chg_spike | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| oi_chg_1h | 0.99 | 1.0 | 0.94 | 0.99 | 1.0 | 0.88 |
| taker_15 | 0.97 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| taker_30 | 0.97 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| top_acc | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| top_pos | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| glob_acc | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| top_acc_chg | 0.99 | 1.0 | 0.94 | 0.99 | 1.0 | 0.96 |
| top_pos_chg | 0.99 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| glob_acc_chg | 0.99 | 1.0 | 0.94 | 1.0 | 1.0 | 0.96 |
| funding_rate | 0.99 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| vol_decay | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| flat14 | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| range3d | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| trend7d | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| age_days | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| mkt_frenzy_24h | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| breadth | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| btc_1d_ema50_200 | 0.86 | 0.61 | 0.94 | 0.81 | 0.65 | 1.0 |
| btc_1d_ret | 0.86 | 0.61 | 0.94 | 0.81 | 0.65 | 1.0 |
| btc_4h_slope | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| btc_1h_slope | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
| spike_age_h | 1.0 | 1.0 | 0.94 | 1.0 | 1.0 | 1.0 |
## Files
- **Pre-registration:** `reports/WILLY_EXIT_V2_2026-11-18_PREREG.md`
- **Every cell × sleeve, with all metrics:** `reports/WILLY_EXIT_V2_2026-11-18_cells.csv`
- **Event-study chart:** `reports/WILLY_EXIT_V2_2026-11-18_event_study.png`
- **Scripts:**
  - `scripts/willy_exit_v2_fetch.py` (1m data);
  - `scripts/willy_exit_v2_sim.py` (simulator) and `willy_exit_v2_parity.py`;
  - `willy_exit_v2_run.py` (all cells: base / D / master);
  - `willy_exit_v2_analyze.py`;
  - `willy_exit_v2_tables.py`;
  - `willy_exit_v2_wl.py` and `willy_exit_v2_wl_tables.py`;
  - `willy_exit_v2_chart.py`.
- **Intermediate data:** scratchpad `exitv2/*.pkl`.
