# STATE_EXIT 2026-11-21 — can a "time to sell" STATE signal beat today's exit? (FLATBASE_SHORT · FRENZY_WILLY_HT · EXHAUSTION_SHORT)

Research only — no engine / config / template change, nothing committed. Design frozen BEFORE outcomes in
`reports/STATE_EXIT_2026-11-21_PREREG.md` (cohorts, today's exit, gates, the thesis variables with their thresholds, the broad screen,
the nulls and the verdict rule; master-decides revision included). Full auto tables: `reports/STATE_EXIT_2026-11-21_tables.md`; every
rule × gate × period: `reports/STATE_EXIT_2026-11-21_pivot.md`.

**Short answer: no state exit found that robustly beats today's exit on any of the three sleeves.** One FLATBASE rule passes the
pre-committed master test on the letter (2 firings) but fails its own null and both backtest blocks → at most an OBSERVE line, not a ship.
The give-back signals the operator saw on ENSO are real, but they fire AFTER the turn and fire just as often on trades that went on to hit
the take profit.

---

## 0 · Method in one page

- **Cohorts.** REAL = every archived export (paper + live), CLOSED, no probes, operator-closed out (0), dedup (opened_at, pair, direction),
  back to each sleeve's first fill. MASTER = opened before 2026-11-17 17:00 UTC, BATCH = after. Every real fill is RE-PRICED under TODAY's
  exit from its real fill price (minute after the fill). BACKTEST: FB 545 (DL 601 stack) · HT 300 (DL 594 stack) · EXH 1,011 = a 30 %
  stratified sample of 3,368 (fetch time) — DESIGN < 2025-07-01 ≤ HOLDOUT.
- **Today's exit replayed on 1m:** FB TP −5 % maker / stop +10 % incl. funding PAID (DL 607) / 72 h · HT TP +5 net maker / −4 net / 24 h ·
  EXH row TP / SL / 24 h. Live costs (taker 0.05, maker 0.02, slip FB 0.25 / others 0.10 or the row's), real funding settlements.
- **Decision point** = every minute close while open; gates **G1** P&L now ≥ +1 % · **G2** ≥ +2 % · **G3** in-trade peak ≥ +2 % (any P&L
  now — the operator's "it was up 2–8 % and gave it back"). **Label** = taker exit at the next minute open (slip + fee + funding so far) −
  today's result on the same path. A rule = first minute (gate ∧ condition) → exit; one FILL = one unit.
- **Features** from data closed at or before the decision (1m rolling windows, complete clock 5m / 15m / 1h / 4h bars, metrics
  create_time ≤ bar open, last settled funding). **Lag check:** features recomputed on paths truncated at the decision minute for 208 fills
  × 3 random minutes → 0 real mismatches (3 flagged = inf vs inf on `oi_chg_in` when entry OI was 0; set to NaN in the evaluation).
- **Layer 1** = the pre-registered thesis family (entry thesis re-read in the trade + the operator's ENSO 5m signs, mirrored for shorts);
  **Layer 2** = broad screen, 51–55 variables × (sign/median, terciles, quintiles; thresholds frozen on the BACKTEST DESIGN) + every 2D
  sign quadrant (≈ 5,600–6,500 tests per sleeve), gate G1. Both with a within-fill circular-shift null (family-wise max).
- **Verdict rule (pre-committed, master decides):** CANDIDATE iff on MASTER Δ ≥ 0 ∧ SAVED $ ≥ CHOPPED $ ∧ day-block bootstrap P(Δ > 0) ≥ 0.80
  ∧ Δ > the best fixed-TP alternative; backtest = cross-check (caveat, not veto); a broad-screen rule must also beat its master null; EXH
  (0 master fills) cannot produce a candidate. $ at one size: $3,050 notional per fill (0.61 × $5,000); DCR at 0.61 of equity per fill.

### Parity (replay of today's exit vs booked / recorded)

| sleeve | cohort | N comparable | mean \|gap\| | median \|gap\| | same exit kind | note |
|---|---|---|---|---|---|---|
| FB | master | 23 (booked under −5 / +10) | 0.68 | 0.02 | 22 / 23 | 1 outlier XAN 10-28: replay TP −5 at +114 min, bot booked the +10 stop (likely a different TP in force that day); without it ≈ 0.02 |
| FB | batch | 4 | 0.10 | 0.09 | 4 / 4 | booked funding included |
| FB | backtest D / H | 94 / 451 | 0.04 / 0.06 | 0.01 / 0.01 | 94 / 451 all | vs the 5m recorded result |
| HT | batch (TP +5 era) | 10 | 0.14 | 0.01 | 10 / 10 | master HT rows all ran OLDER exits → no parity possible, re-priced |
| HT | backtest D / H | 33 / 267 | — | — | 33 / 33 · 266 / 267 | exit kind vs the DL 600 replica |
| EXH | batch | 5 | 0.11 | 0.02 | 5 / 5 | ZKC booked 6.83 = TP 10 minus 3.1 % funding paid — matches |
| EXH | backtest D / H | 406 / 605 | 0.003 / 0.035 | 0.000 | all | |

---

## 1 · FLATBASE_SHORT

**Thesis (services/flatbase.py).** A coin that was FLAT for 14 days pumps (Frenzy episode); the pump fades back toward the old base. Entry =
the first 4-h close ≥ 4 h after the spike that sits BELOW the spike-anchored VWAP, confirmed by a further 4-h close below it (FB_CONFIRM),
not too close to the base (> 2.09 %), no funding above base (DL 601). **Exit hypotheses (thesis exhausted / invalidated):** the 1h close
reclaims the spike VWAP (FB_VWAP) · a 4h close back above the entry (FB_4H) · price within 1 % of the pre-pump base = job done (FB_BASE) ·
hourly volume back ≥ 50 % of the spike hour = a second pump (FB_VOL) · plus the common ENSO signs mirrored for a short and funding.

**Cohort under today's exit**

| period | N | days | WR % | avg % | TP | STOP | TIME | DCR %/day (one size) |
|---|---|---|---|---|---|---|---|---|
| MASTER | 36 | 19 | 66.7 | +0.04 | 24 | 11 | 1 | −0.13 |
| BATCH | 4 | 2 | 25.0 | −6.55 | 1 | 3 | 0 | −5.33 |
| DESIGN | 94 | 76 | 76.6 | +1.31 | 71 | 19 | 4 | +0.18 |
| HOLDOUT | 451 | 263 | 75.4 | +1.05 | 330 | 100 | 21 | +0.50 |

**Give-back decomposition** ($ at $3,050 per fill)

| period | REACHED_TP | GAVE_BACK (peak ≥ +2, net < 0) | of which peak ≥ +4 | NEVER +2, loss |
|---|---|---|---|---|
| MASTER | 24 · +116.9 pts · +$3,566 | **5 · −51.9 pts · −$1,582** (median peak +3.0) | 0 | 7 · −63.7 pts · −$1,943 |
| BATCH | 1 · +4.9 | 1 · −10.4 (PHAROS, peak +4.9) | 1 | 2 · −20.8 |
| DESIGN | 71 · +331.2 | 8 · −75.9 | 5 · −34.8 | 13 · −134.2 |
| HOLDOUT | 330 · +1,535.5 | 62 · −588.3 | 19 · −172.9 | 49 · −487.1 |

On master the entry side (never worked) costs more than the give-back; on the backtest they are about equal.

**Layer 1 — thesis rules, G3 rows + fixed alternatives** (master first; every gate in the pivot file)

| rule | M fired | M Δ %/fill | M 95 % CI | M P(Δ>0) | M saved $ | M chopped $ | M DCR before→after | B Δ (P, n) | D Δ (P, n) | H Δ (P, n) |
|---|---|---|---|---|---|---|---|---|---|---|
| **FB_VWAP·G3** | 2 | **+0.41** | 0.00…+0.88 | **0.88** | +447 | 0 | −0.13→+0.26 | 0 (—, 0) | −0.27 (0.06, 9) | −0.11 (0.10, 39) |
| CLIMAX·G3 | 8 | +0.01 | −0.89…+0.80 | 0.50 | +617 | +606 | −0.13→−0.10 | +2.64 (2) | −0.83 (0.00, 41) | −0.34 (0.01, 148) |
| RSI_ROLL·G3 | 5 | −0.07 | −0.84…+0.63 | 0.39 | +357 | +434 | −0.13→−0.15 | +3.18 (1) | −0.47 (0.06, 32) | −0.22 (0.05, 129) |
| FB_VOL·G3 | 3 | −0.22 | −0.46…0.00 | 0.00 | 0 | +240 | −0.13→−0.35 | 0 | −0.14 (2) | +0.05 (0.94, 6) |
| LH2·G3 | 22 | −0.40 | −2.09…+1.18 | 0.30 | +1,489 | +1,925 | −0.13→−0.38 | +1.88 (2) | −1.17 (0.01, 71) | −0.77 (0.00, 338) |
| FB_BASE·G3 | 12 | −0.44 | −1.18…+0.53 | 0.16 | +372 | +858 | −0.13→−0.53 | 0 | −0.09 (0.25, 9) | −0.17 (0.00, 75) |
| FB_4H·G3 | 13 | −0.52 | −1.53…+0.76 | 0.20 | +719 | +1,287 | −0.13→−0.62 | +2.99 (1) | −1.26 (0.00, 70) | −0.53 (0.01, 342) |
| FUND_PAY·G3 | 12 | −0.63 | −1.34…+0.33 | 0.08 | +355 | +1,042 | −0.13→−0.70 | 0 | −0.82 (0.01, 38) | −0.36 (0.02, 206) |
| EMA_X·G3 | 14 | −0.65 | −2.17…+0.67 | 0.18 | +988 | +1,704 | −0.13→−0.62 | +2.45 (1) | −0.74 (0.05, 44) | −0.71 (0.00, 231) |
| ANY·G3 (exit at the first +2 %-peak minute) | 29 | −0.61 | −2.17…+0.93 | 0.22 | +1,762 | +2,428 | −0.13→−0.56 | | −1.35 | −0.75 |
| FIXED TP 6 % (best fixed on master) | 24 | −0.25 | −1.88…+0.70 | 0.38 | 0 | +934 | −0.13→−0.36 | +0.25 | **+0.45 (0.97)** | **+0.21 (0.95)** |
| FIXED TP 4 % | 24 | −0.66 | −0.81…−0.52 | 0.00 | 0 | +728 | −0.13→−0.69 | | −0.25 | −0.19 |
| FIXED TP 8 % | 24 | −1.58 | −3.48…+0.50 | 0.08 | 0 | +3,268 | −0.13→−1.80 | | +0.85 (0.97) | +0.06 (0.61) |

Thesis-family null (family-wise q95 of per-fill mean Δ): master **+0.64**, DESIGN +0.20, HOLDOUT +0.29 → **no thesis rule beats random timing
on any block** (FB_VWAP·G3 +0.41 < +0.64, family-wise p ≈ 1).

**Timing vs the final in-trade peak (G3, master + batch: 6 give-backs, 25 TP fills)** — LH2 fired on 6 / 6 give-backs but also on 18 / 25
fills BEFORE they hit TP; FB_4H 3 / 6 vs 11 / 25; FB_BASE 1 / 6 vs 11 / 25; FB_VWAP 2 / 6 vs 0 / 25 (median 170 min AFTER the peak, at −2.7 %);
FB_VOL 0 / 6. Backtest (70 give-backs / 401 TP): LH2 69 vs 328, FB_4H 68 vs 333, FB_VWAP 26 vs 20 (median +264 min after the peak at −5.9 %).

**Layer 2 — broad screen (master, 6,264 tests: 540 1D + 5,724 2D, gate G1):** best t +1.03 (dbase in the middle tercile, 7 fills) vs the
family-wise null q95 **1.08** → **0 survivors**. The top 15 are all "funding received > 0 ∧ X" cells — every one NEGATIVE on DESIGN
(t −1.3…−2.6) and HOLDOUT (t −1.2…−3.9).

**Verdict (pre-committed rule).** FB_VWAP·G3 meets the four master conditions **on the letter** (Δ +0.41 ≥ 0, saved $447 ≥ chopped $0,
P 0.88 ≥ 0.80, beats the best fixed TP 6 at −0.25) — but on **2 firings** (LIGHT 10-24: exit at +0.3 % after a +3.5 % peak; BANK 10-26:
exit at −5.7 % after a +3.0 % peak — both instead of the −10.4 % stop), 70 % of the Δ from one pair, below its own family null, and the
backtest DISAGREES (DESIGN −0.27, P 0.06; HOLDOUT −0.11, P 0.10, 39 firings, chopped $3,544 > saved $2,301). Mechanically it is a
conditional tighter stop, not a profit-taking signal, and tighter FB stops are already refuted (DL 603). → **formal CANDIDATE → OBSERVE only
(if the operator wants the line), NOT a ship; quant recommendation: do not even register it unless the operator wants it watched.** The
fixed-TP read is unchanged from DL 603/604 (TP 6 better on the backtest, worse on master — FB_TP6_SHADOW keeps deciding).

![FB](STATE_EXIT_2026-11-21_data/chart_FB.png)
*Event study: real + backtest fills that reached ≥ +2 %, aligned at the in-trade peak. Top: gross P&L (hindsight path, also after the
exit). Bottom: hourly volume ÷ the episode's peak hour. The give-backs show a volume re-surge only AFTER the peak (the second pump is visible
once it is already against us); before the peak the two groups look the same.*

---

## 2 · FRENZY_WILLY_HT (long)

**Thesis (services/frenzy.py).** Inside a live Frenzy burst (volume spike, price above the spike-anchored VWAP), buy the first red 5m dip
≥ 1.4 % under the spike high (DL 594) on a setup the turnover filter refused (HT) — continuation back to the spike high. Entry stamps: DI
spread, ADX delta, 5m volume trend, spike high H, VWAP. **Exit hypotheses:** the burst decays — 5m DI spread flips (HT_DI), ADX turns down
(HT_ADX), volume trend < 0.5 (HT_VOLFADE), 5m EMA5 ≤ EMA20 (HT_EMAGAP), price loses the spike VWAP (HT_VWAP) — or completes: price back at
the spike high (HT_NEWHIGH); plus the operator's ENSO signs (RSI rollover from > 75, two lower highs, EMA5/8 cross, red climax candle,
funding).

**Cohort under today's exit (TP +5 net / −4 / 24 h)**

| period | N | days | WR % | avg % | TP | STOP | DCR %/day |
|---|---|---|---|---|---|---|---|
| MASTER | 49 | 19 | 51.0 | +0.58 | 25 | 24 | +0.67 |
| BATCH | 11 | 3 | 54.5 | +0.86 | 6 | 5 | +1.85 |
| DESIGN | 33 | 31 | 36.4 | −0.71 | 12 | 21 | −0.11 |
| HOLDOUT | 267 | 189 | 44.2 | −0.05 | 118 | 149 | −0.04 |

**Give-back decomposition**

| period | REACHED_TP | GAVE_BACK | of which peak ≥ +4 | NEVER +2, loss |
|---|---|---|---|---|
| MASTER | 25 · +126.6 pts · +$3,861 | **9 · −36.9 pts · −$1,124** (median peak +2.7) | 2 · −8.2 | 15 · −61.5 pts · −$1,876 |
| BATCH | 6 · +30.0 | 4 · −16.4 | 1 · −4.1 | 1 · −4.1 |
| DESIGN | 12 · +60.0 | 9 · −36.2 | 5 · −19.9 | 12 · −47.3 |
| HOLDOUT | 118 · +594.0 | 56 · −227.9 | 17 · −69.3 | 93 · −380.6 |

Again the entry side (never reached +2 %) is the larger loss on every block.

**Layer 1 — G3 rows + fixed alternatives (incl. the TP grid around today's and the late-TP switch)**

| rule | M fired | M Δ | M 95 % CI | M P(Δ>0) | M saved $ | M chopped $ | M DCR b→a | B Δ (P, n) | D Δ (P, n) | H Δ (P, n) |
|---|---|---|---|---|---|---|---|---|---|---|
| CLIMAX·G3 | 10 | +0.13 | −0.36…+0.71 | 0.68 | +683 | +491 | +0.67→+0.82 | 0 (0) | −0.00 (0.49, 4) | −0.06 (0.30, 53) |
| RSI_ROLL·G3 | 4 | +0.11 | −0.27…+0.42 | 0.69 | +332 | +170 | +0.67→+0.80 | +0.32 (1) | +0.21 (0.70, 9) | −0.04 (0.30, 28) |
| HT_NEWHIGH·G3 | 14 | +0.03 | −0.40…+0.57 | 0.54 | +728 | +685 | +0.67→+0.71 | +1.39 (1.00, 8) | +0.65 (0.91, 15) | −0.04 (0.36, 68) |
| HT_EMAGAP·G3 | 7 | −0.05 | −0.40…+0.35 | 0.39 | +354 | +431 | +0.67→+0.59 | −1.05 (4) | +0.34 (7) | −0.03 (39) |
| FUND_RECV·G3 | 12 | −0.08 | −0.43…+0.27 | 0.32 | +685 | +802 | +0.67→+0.58 | −0.08 (3) | +0.30 (5) | +0.04 (37) |
| HT_VOLFADE·G3 | 7 | −0.08 | −0.35…+0.20 | 0.27 | +217 | +337 | +0.67→+0.56 | −0.81 (2) | −0.03 (6) | +0.02 (17) |
| HT_DI·G3 | 6 | −0.09 | −0.46…+0.30 | 0.31 | +297 | +437 | +0.67→+0.54 | −0.90 (4) | +0.32 (6) | +0.01 (36) |
| LH2·G3 | 14 | −0.11 | −0.73…+0.39 | 0.33 | +850 | +1,018 | +0.67→+0.53 | +0.59 (5) | +0.33 (16) | −0.05 (81) |
| HT_VWAP·G3 | 19 | −0.21 | −0.80…+0.29 | 0.21 | +999 | +1,319 | +0.67→+0.41 | +0.58 (5) | +0.30 (12) | −0.07 (63) |
| HT_ADX·G3 | 21 | −0.84 | −1.50…−0.26 | 0.00 | +608 | +1,861 | +0.67→−0.37 | −0.02 (6) | +0.48 (11) | −0.14 (78) |
| EMA_X·G3 | 3 | −0.01 | −0.29…+0.22 | 0.46 | +158 | +169 | +0.67→+0.65 | −1.33 (2) | +0.22 (4) | +0.00 (22) |
| **FIXED TP +4.5 net (best fixed on master)** | 26 | −0.08 | −0.30…+0.42 | 0.28 | +262 | +381 | +0.67→+0.58 | −0.27 | +0.08 | −0.03 |
| FIXED TP +4 | 27 | −0.18 | −0.55…+0.30 | 0.20 | +494 | +764 | +0.67→+0.44 | −0.55 | +0.37 | −0.05 |
| FIXED TP +3 | 27 | −0.73 | −1.11…−0.26 | 0.00 | +433 | +1,526 | +0.67→−0.23 | −0.45 | +0.56 | −0.02 |
| FIXED TP +6 | 25 | −0.29 | −1.18…+0.36 | 0.21 | 0 | +1,086 | +0.67→+0.26 | +0.55 (0.96) | −0.20 | +0.07 |
| FIXED late TP +0.25 from 30 min (WILLY's; HT runs it OFF) | 14 | −0.27 | −0.81…+0.17 | 0.11 | +797 | +1,207 | +0.67→+0.34 | −0.90 | +0.41 | −0.37 (0.01) |

Thesis-family null q95: master **+0.35** (best observed +0.13), HOLDOUT +0.38, DESIGN +1.11 → nothing beats random timing.
**No rule reaches P(Δ > 0) ≥ 0.80 on master** (best 0.69). No fixed TP around +5 beats today on master either (TP 4.5 −0.08, 4 −0.18,
3 −0.73, 6 −0.29, late TP −0.27) — today's +5 is the best fixed level on master.

**Timing vs the peak (G3, master + batch: 13 give-backs, 31 TP fills) — "how often does the SAME sign fire on trades that went on to hit TP":**

| sign | fired on give-backs | median min after the final peak | median P&L when it fired | fired on TP-bound fills BEFORE the TP |
|---|---|---|---|---|
| RSI_ROLL (5m RSI max ≥ 75, now −10) | 4 / 13 | +5 | −0.2 % | **1 / 31** |
| LH2 (2 lower highs) | 10 / 13 | +10 | +0.7 % | 9 / 31 |
| EMA_X (EMA5 under EMA8 & 13) | 2 / 13 | +54 | −1.3 % | 3 / 31 |
| CLIMAX (red 5m bar, 2 × volume, near the top) | 6 / 13 | +7 | +0.1 % | 4 / 31 |
| FUND_RECV (long receives ≥ 0.05 %) | 5 / 13 | 0 | +1.9 % | 10 / 31 |
| HT_ADX | 9 / 13 | +15 | −1.0 % | 18 / 31 |
| HT_NEWHIGH | 8 / 13 | −0.5 | +2.2 % | 14 / 31 |
| HT_VWAP | 13 / 13 | +4 | 0.0 % | 11 / 31 |

The RSI rollover is the cleanest (rarely chops a TP trade) but catches only 4 of 13 give-backs and fires ~5 min after the top, at ≈ 0 %
— master Δ +0.11 (P 0.69), HOLDOUT −0.04 over 28 firings. Everything else either fires late (after most of the move is gone) or chops TP
trades as often as it saves give-backs.

**Layer 2 — broad screen (master, 6,490 tests, G1):** best t +1.58 (1h RSI > 63 ∧ OI not up since entry, 7 fills) vs the family-wise null
q95 **1.77** → **0 survivors**. The most consistent-looking cell (5m ADX > 26.8 ∧ BTC 60-min return ≤ 0: master +0.52 / batch +1.37 /
DESIGN +0.50 / HOLDOUT +0.03, t 1.37 / 1.38 / 1.12 / 0.18) is below the null and fades to zero on the 267-fill HOLDOUT — not a candidate.

**Verdict: no state exit found for WILLY_HT.** Keep today's +5 / −4 / 24 h. (DL 600's result — HT leaks on the exit side, but no exit grid
cell is positive on both DESIGN and HOLDOUT — stands; a state signal does not change it.) Sensitivity: HT DESIGN is only 33 fills under
the brief's split; with DL 600's split (2025 / 2026) the HOLDOUT is still the 2026 block where every rule sits at ≈ 0.

![HT](STATE_EXIT_2026-11-21_data/chart_HT.png)
*At the peak the 5m RSI is the SAME (~71–73) for trades that went on to give back and for trades that hit TP; they separate only after the
turn (give-backs' RSI falls faster). The signal exists, but it arrives when the give-back has already started.*

---

## 3 · EXHAUSTION_SHORT

**Thesis (services/exhaustion.py).** An overextended Frenzy pump runs out of buyers: 15m RSI falls back through 70 while the last hour's
volume is still ≥ 64 % of the episode's peak hour and the spot share of volume is falling (+ the DL 582 OI / spot filter). **Exit
hypotheses:** the exhaustion has played out (15m RSI ≤ 30 — EXH_RSI_OS; RSI turning up two bars below 50 — EXH_RSI_UP) or buyers return
(hourly volume ≥ 80 % of the peak with the hour up — EXH_VOLBACK; OI +5 % in an hour — EXH_OI_UP; taker-buy share ≥ 55 % — EXH_TB_UP);
plus the mirrored ENSO signs and funding. (In-trade spot share was NOT available — blind spot.)

**Cohorts:** MASTER 0 (EXH started 11-18, all real fills are in the current batch) · BATCH 5 · DESIGN 406 · HOLDOUT 605 (30 % stratified
sample). **By the pre-committed rule EXH cannot produce a candidate; the backtest read below is exploratory.**

| period | N | WR % | avg % | TP | STOP | TIME | DCR %/day |
|---|---|---|---|---|---|---|---|
| BATCH | 5 | 80.0 | +5.08 | 4 | 1 | 0 | +5.16 |
| DESIGN | 406 | 40.2 | +0.60 | 114 | 239 | 53 | +0.16 |
| HOLDOUT | 605 | 43.1 | +0.71 | 224 | 341 | 40 | +0.39 |

**Give-back decomposition:** EXH is where the give-back is the largest loss — DESIGN 123 fills · −563 pts (58 of them first up ≥ +4 %),
HOLDOUT 210 · −1,181 pts (120 up ≥ +4 %) vs never-worked 120 · −516 / 134 · −699. Batch: 0 give-backs (4 TP, 1 never worked).

**Layer 1 — G3 rows + fixed TP (backtest; batch N = 5 illustrative)**

| rule | B Δ (n) | D Δ (P, n) | H Δ (P, n) |
|---|---|---|---|
| RSI_ROLL·G3 (5m RSI in-trade min ≤ 25, now +10) | 0 (0) | −0.16 (0.01, 53) | +0.05 (0.81, 50) |
| CLIMAX·G3 | 0 (0) | −0.24 (0.02, 97) | +0.03 (0.65, 67) |
| EXH_RSI_OS·G3 | 0 (0) | −0.04 (0.18, 35) | +0.02 (0.72, 26) |
| FUND_RECV·G3 | 0 (0) | −0.13 (0.05, 19) | +0.01 (0.61, 3) |
| EXH_OI_UP·G3 | −2.75 (2) | −0.40 (0.03, 196) | −0.04 (0.43, 330) |
| EXH_RSI_UP·G3 | −0.51 (1) | −0.44 (0.00, 134) | −0.13 (0.11, 147) |
| EXH_VOLBACK·G3 | 0 (0) | −0.13 (0.16, 60) | −0.27 (0.04, 170) |
| EMA_X·G3 | −0.93 (1) | −0.57 (0.00, 202) | −0.22 (0.10, 284) |
| FUND_PAY·G3 | −3.11 (3) | −0.24 (0.07, 107) | −0.38 (0.03, 284) |
| EXH_TB_UP·G3 | −0.40 (1) | −0.83 (0.00, 256) | −0.57 (0.00, 326) |
| LH2·G3 | −2.82 (2) | −0.68 (0.01, 272) | −0.71 (0.00, 431) |
| ANY·G3 | −4.75 (4) | −0.73 (0.01, 285) | −0.65 (0.01, 470) |
| FIXED TP 9 / 8 / 7 % | −0.60 / −1.10 / −1.34 | −0.10 / −0.21 / −0.39 | +0.03 / **+0.06** / +0.05 |
| FIXED TP 6 / 5 % | −1.94 / −2.54 | −0.47 / −0.45 | −0.13 / −0.29 |

Thesis-family null q95: DESIGN +0.38, HOLDOUT +0.80 — no rule is positive on DESIGN at all; the best HOLDOUT (+0.05) is noise. The lower
fixed TP (DL 603's 7–8 % lead) is again a tie with today's 10 % at the same size (HOLDOUT +0.05 / +0.06, DESIGN −0.39 / −0.21).
**Timing (backtest, 333 give-backs / 338 TP):** LH2 fires on 314 give-backs but also 303 TP-bound fills; EXH_TB_UP 246 vs 250; EXH_OI_UP
275 vs 186; FUND_PAY fires 22 min BEFORE the peak (it is an entry-time state, not a turn signal).

**Layer 2 — broad screen (deciding block DESIGN, 5,610 tests, G1):** best t +1.33 vs null q95 **3.38** → **0 survivors**; the top cell
flips sign on HOLDOUT.

**Verdict: no state exit found for EXH** (and no candidate possible until EXH has master fills). The real lever stays TP × size (DL 603).

![EXH](STATE_EXIT_2026-11-21_data/chart_EXH.png)
*The RSI gap before the peak between the two groups mirrors the P&L level (TP trades are already deep in profit), not a turn signal; after
the peak the give-backs' RSI rebounds — again only after the turn.*

---

## 4 · Named live cases — illustration only (in-sample, picked because they gave back; never counted)

First firing of each thesis sign once the trade had been up ≥ +2 % (minutes after entry @ gross P&L then):

| case | today's exit (replay) | peak | RSI_ROLL | LH2 | EMA_X | CLIMAX | FUND | HT_ADX / NEWHIGH / VWAP |
|---|---|---|---|---|---|---|---|---|
| **ENSO** HT long, 11-21 02:30:14 UTC @ 0.5034 | STOP −4.10 at +50 min (live: 03:21:47, −4.01) | **+5.05 gross at +12 min** (TP needs +5.07 → missed by a hair) | +23m @ +1.01 | +23m @ +1.01 | — | **+1m @ +2.17** (before the peak) | receive 0.03 % (< 0.05) | ADX +33m @ +2.05 · NEWHIGH +11m @ +3.73 (H approximated) · VWAP — |
| **EVAA** HT long, 11-20 23:20:12 @ 6.20 | open at data end (+2.60 marked) | +3.94 at +146 min | — | +143m @ +2.14 (before the peak) | — | +178m @ +0.78 | — | ADX / NEWHIGH +121m @ +1.24 |
| **RARE** EXH short, 11-20 20:00:39 @ 0.01386 | open at data end (+4.43 marked) | +7.94 at +44 min | — | +58m @ +6.06 | +138m @ +4.47 | — | **PAY from entry** (−1.38 % and −1.76 % per settlement on 11-20) | TB_UP +70m @ +4.83 |

ENSO, as the operator described: the RSI rollover and the second lower high came 11 minutes after the top, at about +1 % (an exit there
would have booked ≈ +0.8 % instead of −4.1 %). The climax candle fired at minute 1 — before the run to +5 %, so it would have cut the
winning part too. The funding the operator saw (−0.746 % per 4 h) is not in the settled history (00:00 UTC settled −0.027 %, 16:00 11-20
−0.16 %): it was probably the live PREDICTED rate, which no history keeps — blind spot. The real test is the master table above: the
RSI rollover rarely chops a TP-bound trade (1 / 31) but saves only 4 / 13 give-backs and does not pass the bar; the lower-highs sign
saves 10 / 13 but chops 9 / 31.

---

## 5 · Coverage and blind spots

**Tested (every variable × sign-or-median / terciles / quintiles + every 2D sign pair, and the thesis family at G1 / G2 / G3):** coin 1m
returns 5 / 15 / 60 min and their acceleration · 1m volume ratios (5/60, 15/240, last hour vs the pre-entry day) · taker-buy share 5 / 15 / 60
min and its change since entry · range expansion 15/60 · 5m RSI and its 3-bar turn · 5m +DI−−DI, ADX, ADX change, ATR % · close vs 5m
EMA5 / 13 / 20 / 50 · EMA5–8 and 5–20 gaps · 5m volume trend · consecutive lower highs · 15m RSI and its 1- and 2-bar changes · close vs 15m
EMA20 / 50 · close vs 1h EMA20, 1h RSI · hourly volume vs the episode peak (vol decay) · spike-VWAP distance (1m and 1h close), base
distance (FB), spike-high distance (HT) · last 4h close vs entry · BTC 5 / 15 / 60-min return, BTC 5m RSI, BTC vs 1h EMA20 · open interest
1h change and change since entry, top-trader position ratio change, taker long/short volume ratio (5m metrics) · funding at t (received /
paid) and accrued · in-trade P&L, peak, drawdown from peak, minutes in trade, minutes since the peak. Coverage 100 % of decision minutes for
every variable kept (FB 54, HT 55, EXH 51 — EXH has no spike-anchored variables).

**Not tested (blind spots):** order-book / tape (Route B — too few trades yet; for the ~12-10 weekly scan) · in-trade SPOT share (the EXH
entry variable) · the live PREDICTED funding rate (only settlements are historical) · liquidations · news · market-wide breadth during
the trade · EXH master fills (0) · EXH backtest is a 30 % sample · HT DESIGN only 33 fills under the brief's split · stop-first ordering on
1m bars, fixed exit slip (FB 0.25 / others 0.10) — a fast state exit in a falling market could slip more · trades re-priced under today's
exit from real entries (master HT / FB fills booked older exits).

---

## 6 · Verdicts

| sleeve | thesis family (master) | broad screen | best fixed alternative (master) | verdict |
|---|---|---|---|---|
| FLATBASE_SHORT | FB_VWAP·G3 passes the letter on 2 firings (Δ +0.41, P 0.88, DCR −0.13 → +0.26 %/day); below its own null; backtest disagrees (D −0.27, H −0.11) | 0 survivors (best t 1.03 < null 1.08) | TP 6 −0.25 (today's TP 5 best on master; TP 6 best on backtest — FB_TP6_SHADOW decides) | **No robust state exit.** FB_VWAP·G3 = formal candidate → OBSERVE only if the operator wants it; recommendation: don't ship |
| FRENZY_WILLY_HT | nothing reaches P ≥ 0.80 (best RSI_ROLL·G3 +0.11, P 0.69; CLIMAX·G3 +0.13, P 0.68) | 0 survivors (best 1.58 < 1.77) | TP +4.5 −0.08 · late TP −0.27 · TP +3 −0.73 · TP +6 −0.29 → today's +5 is best | **No state exit found.** Keep +5 / −4 / 24 h |
| EXHAUSTION_SHORT | no master fills → no candidate; backtest: nothing positive on DESIGN | 0 survivors (best 1.33 < 3.38) | (backtest) TP 8 ties TP 10 on HOLDOUT (+0.06), loses on DESIGN (−0.21) | **No state exit found.** TP × size stays the lever (DL 603) |

---

## 7 · Plain-English summary for the operator

- We asked: while a trade is up 2–8 %, is there something in the coin or the market that says "sell now, it is about to give it all back"?
  We tested the reasons each sleeve enters (re-read during the trade), the five signs you saw on ENSO, and about 6,000 other combinations,
  on every real trade since each sleeve started and on 1,800 backtest trades.
- **The signs are real, but they come too late and too often.** At the top, a trade that is about to give everything back looks exactly
  like a trade that is about to hit the take profit. The warning signs (RSI rolling over, lower highs, volume coming back) appear 5–15
  minutes AFTER the top — and they appear just as often on trades that then recover and hit the target. Selling on them gives away about
  as much from the winners as it saves on the losers.
- **ENSO:** the RSI rollover and the second lower high fired 11 minutes after the top, at about +1 %; selling there would have saved this
  trade. But across all WILLY_HT trades, that same rule would have saved only 4 of 13 give-backs, and on the backtest it is flat.
- **Lower fixed take profits do not help either** on the real trades (WILLY_HT +5 is the best level; FLATBASE −5 beats −4 and −6 on the
  real trades; EXH 7–8 % only ties 10 %).
- **The bigger loss is still the entry**, not the give-back, for FLATBASE and WILLY_HT (trades that never got to +2 %). For EXHAUSTION the
  give-back is the larger loss, but nothing measurable during the trade separates it in time.
- One FLATBASE rule ("after being up 2 %, the hourly close climbs back above the spike's average price → get out") passed our pre-set
  test on paper, but it only happened twice, it is not better than random timing, and on 545 backtest trades it loses. At most we watch it
  (observe only); we should not switch it on.
- Not yet testable: the order book / trade tape (Route B, about 3 more weeks of data) and the live funding prediction — those are the next
  places to look.

## Files

- Report: `reports/STATE_EXIT_2026-11-21.md` · pre-registration `reports/STATE_EXIT_2026-11-21_PREREG.md` · tables
  `reports/STATE_EXIT_2026-11-21_tables.md` · pivot `reports/STATE_EXIT_2026-11-21_pivot.md`
- Charts: `reports/STATE_EXIT_2026-11-21_data/chart_{FB,HT,EXH}.png`
- Data: `reports/STATE_EXIT_2026-11-21_data/` (real_cohort.csv, fills.csv, fills_out.csv, dec_*.pkl, eval_*.pkl, paths.pkl, k1m/ met/ fund/
  caches ≈ 7,200 kline pair-days + 4,500 metrics pair-days from data.binance.vision at 0 weight; fapi only for 11-20/11-21 days and the
  current month's funding, used weight ≤ 55 / min)
- Scripts: `scripts/STATE_EXIT_2026-11-21_{real_cohort,cohorts,common,fetch,build,eval,report,pivot}.py`
  (re-run: real_cohort → cohorts → fetch → build [--lagcheck] → eval → report → pivot)
