# DIST_PEAKSTOP — "stop at the pump top + 1 %" for DISTRIBUTION_SHORT: what it is, does it make sense, does it pay? (2026-11-21, BACKTEST + 6 real fills)

**Verdict: NO — keep today's stop (min(1.5 × ATR15, 6 %)). No observe line.**
The idea makes sense as a story, but the market does not treat the pump top as a line. Once price trades through the pump top + 1 %, **20 %** of those
shorts still go on to hit the take profit, versus **17 %** after a **random** level the same distance away. So the top carries no extra information.
In practice a pump-top stop is just a **wider stop**: the median stop goes from about 5 % to 8–9 %. It wins more often, but each loss is bigger, and the
two cancel. With the hard caps the operator asked for (8 / 10 / 12 / 15 %), **all 8 pump-top cells lose money against today's stop on the
HOLDOUT** (−0.02 … −0.15 %/fill). **The pick fails all 8 gates**, and the drawdown gets worse in every cell (−55 % → −65 … −88 %).

Pre-registration (frozen before the grid ran): `reports/DIST_PEAKSTOP_2026-11-21_PREREG.md`.
Scripts: `scripts/DIST_PEAKSTOP_2026-11-21_{sim,eval,real,extras}.py`, built on the DIST_THEORY_EXIT engine (imported, not edited).
Data: `reports/DIST_PEAKSTOP_2026-11-21_data/` (sim.npz · eval.csv · gates.csv · dist_to_top.csv · kelly.csv · halves.csv · toptest.csv ·
perfill.csv.gz · real_fills.csv · real_summary.csv · *.log). Chart: `reports/DIST_PEAKSTOP_2026-11-21.png`.

---

## 1. What DIST trades, and what a stop at the pump top means (theory)
**The trade (services/distribution.py, DECISION_LOG 586):** a coin had a FRENZY spike (a huge pump) within the last 72 h. Then a **closed red 15-minute
candle** prints **≥ 5 × the normal volume** while **top traders cut their long positions** (the top-trader L/S ratio falls by ≥ 0.011 over 4 h) and the
1h RSI is still ≥ 74 (the coin is still hot). The bot shorts right after that candle. The story is that a big holder is **selling (distributing) into
the pump**: the heavy red candle is that supply hitting the bids, and the top-trader ratio shows who is selling.

**Where would the story be proven wrong?**
- **The pump top (PK)** is the highest price since the spike. If a big holder is unloading, buyers should not be able to take out the high: the
  seller is capping the price. **A new high above the pump top means the selling was absorbed** — somebody bought everything the distributor sold,
  and more. That is the natural "thesis is dead" level, and +1 % is a buffer against a single wick.
- **The selling candle's high (DH)** is a tighter version: price back above the bar where the supply hit means that bar's sellers are already
  under water. It sits lower (median 6.3 % above entry vs 7.0 % for the top).
- **Today's stop** (1.5 × ATR15, max 6 %) is a volatility stop. It knows nothing about the pump's structure; on 94 % of fills it sits **below** the
  pump top + 1 %, so it can stop us out while the thesis is still alive.

**The prediction this makes, and how it can be checked:** if the top is a real line, then (a) after a break of the top the short should rarely
recover — fewer TPs than after breaking an arbitrary level of the same width; (b) a stop at the top should beat a fixed stop of the same width and a
"shuffled top" (another fill's distance). If neither holds, the pump-top stop is just a wider stop.

## 2. The exact rules tested
- **E** = entry price (backtest: the entry minute's 1m open; real: the real fill). **S0** = today's stop = E × (1 + min(1.5 × ATR15 %, 6 %)).
  TP = E × (1 − min(20 × ATR15 %, 10 %)), resting maker; time exit 24 h — unchanged in every cell.
- **PK** = highest price traded **from the FRENZY spike** (episode start: backtest `ep_on_ms` on 5m bars; real `entry_frenzy_spike_at` − 5 min on 1m bars)
  **up to the last bar closed before entry**. It includes the distribution candle and is known at entry.
- **Capped top** TOP(CAP) = min(PK × 1.01, E × (1 + CAP)) — the **hard cap** CAP ∈ {8, 10, 12, 15 %} (no −25 % trades).

| cell | stop | can it be tighter than today? |
|---|---|---|
| **P{CAP}** (replace) | TOP(CAP) if PK ≥ E × 1.01, else S0 | yes (top within ~4 % of entry) |
| **Q{CAP}** (wider-of) | max(S0, TOP(CAP)) | never — fills today did not stop are untouched (asserted Δ = 0) |
| W{CAP} control | fixed E × (1 + CAP) | — |
| S{CAP} control | as Q, but the distance comes from a random OTHER fill's top (5 seeded shuffles) | — |
| DQ10 / DQ15 (descriptive) | max(S0, min(DH × 1.01, cap)) — the selling candle's high | — |

(For reference: the first study's C_PK1 = P15 here — **parity 1,547 / 1,547**. Two corrections to the brief: its cap was **15 %**, not 25 %, and on the
real fills C_PK1 did **not** turn CYBER into a loss — C_PK0, with no +1 % buffer, did. With the buffer CYBER's stop sits 3.7 % above entry and it still
TPs.)

### Where the pump top sits relative to entry (known at entry, no outcomes)
| block | N | p10 | p25 | **median** | p75 | p90 | top + 1 % above today's stop | top + 1 % beyond cap 8 / 10 / 12 / 15 |
|---|---|---|---|---|---|---|---|---|
| DESIGN (< 2025-07) | 559 | 2.7 % | 3.8 % | **5.8 %** | 8.4 % | 12.8 % | 95 % | 37 / 21 / 14 / 8 % |
| HOLDOUT (≥ 2025-07) | 988 | 3.4 % | 5.1 % | **8.1 %** | 15.0 % | 26.0 % | 93 % | 59 / 46 / 37 / 27 % |
| ALL | 1,547 | 3.1 % | 4.5 % | **7.0 %** | 12.1 % | 22.9 % | 94 % | 51 / 37 / 29 / 21 % |
| real fills (6) | 6 | — | 4.7 % | **7.0 %** | 12.5 % | 50 % (BLESS) | 6 / 6 except CYBER 2.7 % | — |

Today's stop: median 4.7 % (p25 3.5, p75 6.0). In HOLDOUT more than a third of fills have a pump top beyond 12 %: there the cap, not the top, decides
the stop. A pump-top rule with a hard cap is a **wider stop with a cap** for most fills.

## 3. Method
- Cohort = the 1,547 DIST_EXIT REF_B backtest fills (entry rule incl. RSI1h ≥ 74.07, one position per pair). Engine = DIST_THEORY_EXIT `sim_one`
  (1m path, stop checked first on the high and filled at max(stop, open) so gaps count, maker TP one tick through, both in one bar → stop, 24 h).
  Costs: entry slip + 0.05 taker; exit 0.05 taker + slip (model, floor 0.10) or 0.02 maker; real funding. Parity: REF 1,547 / 1,547 and P15 = C_PK1
  1,547 / 1,547.
- DESIGN < 2025-07-01 picks one cell (best mean Δ among the 8 P / Q cells); HOLDOUT read once against 8 gates (prereg). $ at one size = $3,050 per fill;
  DCR = one size at 0.61 × equity per fill over every calendar day.
- Exit counterfactual read on the **live-stopped cohort** (fills today's stop actually stopped): saved vs deeper; Q / W / S non-stopped fills Δ = 0 (asserted).
- Real fills: newest export `scalpars_orders_paper_2026-11-21_19-22-18.csv` → 6 closed DISTRIBUTION_SHORT (same 6 keys as the all-export dedup).
  Re-walked from the real fill price on cached 1m. **There is no master batch for DIST:** `reports/MASTER_POOL_stacked.csv` (1,341 rows) has **0**
  DISTRIBUTION_SHORT rows — the sleeve has only been live since DL 586 (11-18), so the current batch is the whole real history.

## 4. Before → after — backtest (one size $3,050 per fill)
| block | cell | N | WR | avg net % | total $ | avg win | avg loss | worst trade | DCR %/day | max DD | worst day | duration avg / median (h) | TP / STOP / TIME | Δ %/fill [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DESIGN | **today (REF)** | 559 | 42.0 % | **+1.03** | $17,497 | +8.13 | −4.13 | −9.13 | 0.41 | −34.5 | −19.2 | **6.45 / 1.70** | 168 / 319 / 72 | — |
| DESIGN | P12 (pick) | 559 | 53.1 % | +0.97 | $16,554 | +7.69 | −6.64 | −13.80 | 0.35 | −57.7 | −23.0 | 9.66 / 5.03 | 192 / 237 / 130 | −0.055 [−0.43, +0.31] |
| DESIGN | Q8 | 559 | 51.2 % | +0.88 | $15,062 | +7.81 | −6.37 | −11.13 | 0.32 | −55.2 | −19.5 | 8.91 / 4.32 | 189 / 261 / 109 | −0.143 [−0.45, +0.15] |
| DESIGN | Q12 | 559 | 53.1 % | +0.96 | $16,297 | +7.69 | −6.68 | −13.80 | 0.35 | −57.7 | −23.0 | 9.66 / 5.03 | 192 / 237 / 130 | −0.070 [−0.44, +0.30] |
| DESIGN | Q15 | 559 | 53.5 % | +0.93 | $15,905 | +7.67 | −6.82 | −18.81 | 0.33 | −57.9 | −23.0 | 9.77 / 5.22 | 193 / 234 / 132 | −0.093 [−0.48, +0.30] |
| DESIGN | W12 (fixed) | 559 | 64.4 % | +1.18 | $20,191 | +7.42 | −10.10 | −16.76 | 0.42 | −57.0 | −30.0 | 12.73 / 11.57 | 218 / 150 / 191 | +0.158 [−0.32, +0.62] |
| DESIGN | S12 shuffled top | 559 | 53.5 % | +1.02 | $17,388 | +7.68 | −6.65 | −14.78 | 0.37 | −47.9 | −30.2 | 9.71 / 5.27 | 192 / 235 / 132 | −0.006 |
| DESIGN | DQ15 candle high | 559 | 51.9 % | +1.01 | $17,295 | +7.79 | −6.29 | −18.81 | 0.38 | −53.0 | −22.7 | 9.08 / 4.25 | 191 / 248 / 120 | −0.012 [−0.36, +0.36] |
| HOLDOUT | **today (REF)** | 988 | 42.7 % | **+0.88** | $26,441 | +9.11 | −5.26 | −7.82 | **0.85** | **−54.9** | **−21.8** | **3.51 / 0.63** | 372 / 561 / 55 | — |
| HOLDOUT | **P12 (pick)** | 988 | 51.8 % | +0.75 | $22,588 | +8.97 | −8.09 | −14.30 | 0.58 | −80.8 | −36.0 | 5.24 / 1.33 | 440 / 449 / 99 | **−0.128 [−0.45, +0.20]** |
| HOLDOUT | Q8 | 988 | 49.4 % | +0.86 | $25,951 | +8.97 | −7.05 | −10.82 | 0.78 | −65.2 | −28.1 | 4.80 / 1.11 | 419 / 482 / 87 | −0.016 [−0.27, +0.22] |
| HOLDOUT | Q12 | 988 | 52.4 % | +0.80 | $23,976 | +8.97 | −8.22 | −14.30 | 0.64 | −79.9 | −37.1 | 5.26 / 1.38 | 446 / 443 / 99 | −0.082 [−0.43, +0.27] |
| HOLDOUT | Q15 | 988 | 53.5 % | +0.78 | $23,459 | +8.96 | −8.65 | −17.87 | 0.58 | −87.1 | −42.5 | 5.44 / 1.57 | 455 / 428 / 105 | −0.099 [−0.48, +0.27] |
| HOLDOUT | W12 (fixed) | 988 | 61.1 % | +0.95 | $28,497 | +8.80 | −11.41 | −20.38 | 0.74 | −83.4 | −31.8 | 6.78 / 2.35 | 510 / 341 / 137 | +0.068 [−0.35, +0.52] |
| HOLDOUT | S12 shuffled top | 988 | 53.8 % | +0.82 | $24,654 | +8.88 | −8.56 | −17.46 | 0.67 | −76.2 | −29.7 | 5.57 / 1.53 | 454 / 429 / 105 | −0.059 |
| HOLDOUT | DQ15 candle high | 988 | 52.9 % | +0.85 | $25,550 | +8.96 | −8.28 | −17.87 | 0.68 | −84.4 | −42.5 | 5.30 / 1.41 | 450 / 437 / 101 | −0.030 [−0.38, +0.33] |

All 8 P / Q cells, both blocks: HOLDOUT Δ P8 −0.06 · P10 −0.13 · P12 −0.13 · P15 −0.14 · Q8 −0.02 · Q10 −0.08 · Q12 −0.08 · Q15 −0.10; DESIGN −0.06 … −0.17
(every one negative in both blocks). Full table: `eval.csv`. Worst trades exceed the cap (Q12 −14.3 %) because a market stop fills at the next
open when price gaps through it, plus costs. **A cap is not a guaranteed maximum loss.**

**Duration:** a wider stop keeps losers open longer. The HOLDOUT median goes from 0.63 h to 1.1–1.6 h and the average from 3.5 h to 4.8–5.4 h
(DESIGN 6.5 h → 8.9–9.8 h). More trades also run out the clock at 24 h (55 → 87–105).

### Live-stopped split (HOLDOUT, the 561 fills today's stop actually stopped)
| cell | median stop | saved n · $ | deeper n · $ | non-stopped changed · $ | net $ |
|---|---|---|---|---|---|
| P12 | 9.2 % | 151 · +$43,669 | 409 · −$45,091 | 6 · −$2,430 (tighter stop hit winners) | **−$3,852** |
| Q8 | 8.0 % | 76 · +$28,092 | 445 · −$28,582 | 0 | −$490 |
| Q12 | 9.2 % | 111 · +$42,635 | 410 · −$45,100 | 0 | −$2,465 |
| Q15 | 9.2 % | 123 · +$47,728 | 398 · −$50,709 | 0 | −$2,981 |
| W12 | 12 % | 206 · +$76,127 | 355 · −$74,071 | 0 | +$2,056 |
| S12 (shuffled, mean of 5) | 9.2 % | 124 · +$44,988 | 352 · −$46,775 | 0 | −$1,787 |

Roughly 1 trade is rescued for every 3–6 that are stopped deeper. The rescued ones are big (they flip to +TP), the deeper ones are smaller each, but
there are more of them, and the two cancel.

### The theory test: does the market respect the pump top?
Fills whose price traded **through** the stop level (above today's stop, below 15 %), and what the same fill did next with the stop moved out to 15 %:

| level broken | block | N broke | median level | went on to hit TP anyway | Δ if held to a 15 % stop |
|---|---|---|---|---|---|
| **pump top + 1 %** | HOLDOUT | 329 | 7.1 % | **24.3 %** | +0.77 |
| shuffled level, same widths | HOLDOUT | 257 | 7.8 % | 21.1 % | +0.40 |
| selling-candle high + 1 % | HOLDOUT | 330 | 6.8 % | 23.9 % | +0.57 |
| **pump top + 1 %** | ALL | 535 | 6.8 % | **19.6 %** | +0.56 |
| shuffled level, same widths | ALL | 444 | 7.2 % | 17.3 % | +0.36 |

If the top were where the distribution thesis dies, a short that breaks it should recover **less** often than one that breaks a random level. It
recovers **slightly more** often. A new high above the pump top is not a stronger "get out" signal than any other move of the same size. That is why
gate ⑦ (beat the shuffled top) fails. The selling candle's high behaves the same way.

### HOLDOUT gates (pick = P12; everything else descriptive)
| cell | HOLDOUT Δ [CI] | halves + | Δ @ 0.20 slip | saved > deeper | DCR before → after | max DD before → after | vs W same cap / shuffled | gates ①–⑧ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **P12** | −0.128 [−0.45, +0.20] | 1/3 | −0.12 | ✗ | 0.85 → 0.58 | −54.9 → −80.8 | −0.13 vs +0.07 / −0.06 | ✗✗✗✗✗✗✗✗ | **FAIL** |
| Q8 (best HOLDOUT P/Q) | −0.016 [−0.27, +0.22] | 1/3 | −0.01 | ✗ | 0.85 → 0.78 | −54.9 → −65.2 | beats W8 −0.09 / S8 −0.09 | ✗✗✗✗✗✗✓✗ | FAIL |
| W15 (best control) | +0.207 [−0.26, +0.67] | 2/3 | +0.23 | ✓ | 0.85 → 0.83 | −54.9 → −84.1 | — | ✗✓✓✓✓✗✓✗ | FAIL (and −25 % worst trade) |

Per half-year (Δ %/fill): Q12 −0.34 · −0.32 · +0.32 · −0.04 · +0.20 · −0.35 · −0.29 (2 / 7 positive); P12 the same shape; W12 5 / 7 positive
(`halves.csv`).

## 5. Before → after — current batch (all 6 real DIST fills, the sleeve's whole live history)
One size $3,050 per fill; DCR at 0.61 × equity over 11-18 … 11-21 (4 calendar days).

| cell | N | WR | avg net % | total $ | avg win | avg loss | worst | DCR %/day | duration avg / median (h) | TP / STOP / TIME | Δ $ | leave-one-out Δ range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| booked | 6 | 17 % | −2.83 | — | +9.95 | −5.38 | −6.11 | — | 1.05 / 0.43 | 1 / 5 / 0 | — | — |
| **today (REF re-walk)** | 6 | 17 % | **−3.06** | **−$560** | +9.83 | −5.64 | −6.30 | **−2.90** | **1.06 / 0.44** | 1 / 5 / 0 | — | — |
| P8 = Q8 | 6 | 33 % | −1.85 | −$339 | +9.83 | −7.70 | −8.30 | −1.98 | 1.95 / 0.60 | 2 / 4 / 0 | +$221 | −1.78 … +2.23 |
| P12 = Q12 (pick) | 6 | 33 % | −2.55 | −$467 | +9.83 | −8.74 | −12.30 | −2.71 | 1.98 / 0.68 | 2 / 4 / 0 | +$93 | −2.62 … +2.19 |
| P15 = Q15 (= C_PK1) | 6 | 50 % | +0.77 | +$141 | +9.09 | −7.56 | −8.41 | +0.52 | 5.86 / 1.48 | 2 / 3 / 1 | +$700 | +1.37 … +5.01 |
| W12 (fixed) | 6 | 33 % | −4.89 | −$895 | +9.83 | −12.25 | −12.30 | −4.81 | 2.55 / 0.73 | 2 / 4 / 0 | −$335 | — |

Per fill (net % · exit · stop used · hours held):

| fill | booked | pump top above entry | candle high above | today | P12 (pick) | Q8 | Q12 | Q15 | W12 |
|---|---|---|---|---|---|---|---|---|---|
| SENT 11-19 04:15 | −4.41 SL 0.25 h | 12.5 % | 12.5 % | −4.40 SL · 4.1 · 0.26 h | **−12.30 SL** · 12.0 · 0.68 h | −8.30 SL · 8.0 · 0.53 h | −12.30 SL · 12.0 · 0.68 h | **+7.62 TIME** · 13.6 · 24 h | −12.30 SL · 12.0 · 0.68 h |
| NAORIS 11-19 09:15 | −5.97 SL 0.04 h | 7.0 % | 7.0 % | −6.30 SL · 6.0 · 0.05 h | −8.38 SL · 8.1 · 0.13 h | −8.30 SL · 8.0 · 0.13 h | −8.38 SL · 8.1 · 0.13 h | −8.38 SL · 8.1 · 0.13 h | −12.30 SL · 12.0 · 0.15 h |
| EVAA 11-19 14:31 | −4.38 SL 2.97 h | 4.7 % | 4.7 % | −4.88 SL · 4.7 · 2.98 h | −5.88 SL · 5.7 · 7.69 h | −5.88 SL · 5.7 · 7.69 h | −5.88 SL · 5.7 · 7.69 h | −5.88 SL · 5.7 · 7.69 h | −12.09 SL · 12.0 · 11.0 h |
| CYBER 11-20 09:00 | +9.95 TP 2.29 h | 2.7 % | 2.2 % | +9.83 TP · 4.5 · 2.29 h | +9.83 TP · 3.7 · 2.29 h | +9.83 TP · 4.5 · 2.29 h | +9.83 TP · 4.5 · 2.29 h | +9.83 TP · 4.5 · 2.29 h | +9.83 TP · 12.0 · 2.29 h |
| KERNEL 11-20 12:30 | −6.11 SL 0.61 h | 7.0 % | 7.0 % | −6.30 SL · 6.0 · 0.63 h | −8.41 SL · 8.1 · 0.68 h | −8.30 SL · 8.0 · 0.68 h | −8.41 SL · 8.1 · 0.68 h | −8.41 SL · 8.1 · 0.68 h | −12.30 SL · 12.0 · 0.78 h |
| BLESS 11-21 09:30 | −6.03 SL 0.12 h | 50.2 % | 36.6 % | −6.30 SL · 6.0 · 0.13 h | **+9.83 TP** · 12.0 · 0.38 h | +9.83 TP · 8.0 · 0.38 h | +9.83 TP · 12.0 · 0.38 h | +9.83 TP · 15.0 · 0.38 h | +9.83 TP · 12.0 · 0.38 h |

What the six show: NAORIS, KERNEL and EVAA all ran **straight through** their pump tops, so the top stop only made those losses deeper
(−6.3 → −8.4). BLESS is rescued by **any** stop of 8 % or wider: its pump top was 50 % above entry, so this is the cap acting as a fixed stop, not the
top. SENT is a knife edge. It ran 12–13.6 % against the short and then fell to +7.6 %. The 15 % cap (stop 13.6 %) survives it; the 12 % cap books
−12.3 %. The whole "+$141" of C_PK1 / Q15 rests on SENT falling into that 1.6-point window. With N = 6 (about 3–4 market windows), the real fills
can neither confirm nor refute the backtest.

## 6. Size and per-stop risk
DIST size = `distribution_invest_mult` 0.5 × `distribution_lev_mult` 0.25 (5×) ≈ **0.61 × equity notional** per fill (config.py / trading_config.json,
DL 586). The exchange backstop sits at 6.5 %; any wider stop would also need the backstop moved past it.

| (ALL backtest) | WR | 95 % Wilson floor | breakeven WR | avg win / loss | Kelly at observed / at floor | avg loss per stop, % equity | worst trade, % equity | max stops in one day | worst day | max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| **today** | 42.5 % | 40.4 % | 35.6 % | +8.76 / −4.85 | 0.106 / **0.074** | **3.0 %** | −5.6 % | 8 | −21.8 % | −54.9 % |
| Q8 | 50.0 % | 47.9 % | 44.4 % | +8.54 / −6.81 | 0.102 / 0.064 | 4.3 % | −6.8 % | 7 | −28.1 % | −65.2 % |
| Q12 | 52.7 % | 50.6 % | 47.4 % | +8.51 / −7.67 | 0.100 / 0.061 | 4.9 % | −8.7 % | 7 | −37.1 % | −79.9 % |
| Q15 | 53.5 % | 51.4 % | 48.5 % | +8.50 / −7.99 | 0.098 / 0.058 | 5.1 % | −11.5 % | 7 | −42.5 % | −87.1 % |
| W12 | 62.3 % | 60.3 % | 57.0 % | +8.29 / −10.96 | 0.125 / 0.077 | 7.6 % | −12.4 % | 5 | −31.8 % | −83.4 % |

The win rate goes up, but the breakeven win rate goes up more. The cushion between the floor and breakeven shrinks from +4.8 points to +3.2
(Q12) or +2.9 (Q15), and Kelly at the floor falls. Each stop costs 1.4–1.7 × more equity at today's size. DIST stops cluster (up to 7–8 on one day in
the backtest), so a wider stop turns a −22 % worst day into −28 … −42 %. Making DIST's stop wider makes the sleeve riskier per dollar of edge.

## 7. Recommendation (plain English)
**Do not change DISTRIBUTION_SHORT's stop. Keep 1.5 × the 15-minute ATR, max 6 %.**
- *What the rule is:* "stop at the pump top + 1 %" puts the stop 1 % above the highest price since the coin's spike. The idea: if a big holder is
  selling into the pump, price should not make a new high, so a new high means the selling was absorbed and the trade is wrong.
- *Does it make sense?* As a story, yes. In the data, no. Once price makes that new high, the short recovers to its take profit about as often as
  after any other move of the same size (20 % vs 17 %). The market does not treat the pump top as a line.
- *What it really does:* it is a wider stop (8–9 % instead of 5 %). It turns about 1 in 5 stopped trades into winners, but makes most of the rest lose
  more. Net: −0.02 to −0.15 % per trade on the 16 months it was not tuned on. Drawdown goes from −55 % to −65 … −88 %, and trades stay open about
  twice as long.
- *The 6 real trades* look better with the 15 % cap (+$141 vs −$560). That depends on one trade (SENT) landing in a 1.6-point window, and it turns
  into −$467 with a 12 % cap. Six trades over three days cannot outweigh 1,547 backtest fills.
- **No observe line:** the pick failed gate ⑦ (it lost to the shuffled-top control), so there is no thesis left to watch. The real fills are already
  tracked by the DIST revert gate (DL 586, 30 fills over ≥ 10 days) and the DIST_TP_SHADOW line (DL 605). The open DIST levers stay **size** (the TP × size
  question at the batch review, DL 603) and the Route B squeeze data (DL 602).

## 8. Caveats / blind spots
- This is a follow-up on the same cohort: C_PK* and W_SL* were already seen on DESIGN in the first study. The HOLDOUT gate set and the shuffled control
  guard against that. Nothing passed, so no haircut is needed.
- The entry set is fixed (one position per pair). A longer-held losing trade blocks a later DIST signal on the same coin; that is not modelled and would
  only make the wider stops look worse.
- The backtest models slippage (floor 0.10 %); live thin-coin stop slippage is about 0.22 %. Stops wider than 6.5 % also need the exchange backstop moved.
- Drawdown sums every fill with no slot cap. That overstates clustering, but equally for every cell.
- Real-fill landmarks come from 1m bars, backtest landmarks from 5m bars (same definitions). BLESS has 10 h of data after entry (it TPs within 0.4 h in
  every wider cell, so nothing is censored).
