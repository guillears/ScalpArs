# FRENZY ATR cap: should frenzy_max_atr_pct move from 2.5 to 3.0, 3.2 or 3.5? (2026-10-08)

Research only. No code, config, template, test or script in the repo was touched, and nothing was committed. This is an **unreviewed backtest** (feedback_no_arm_before_review): it does not support any ship or arm until the caveman and deep reviews have checked it.

## Plain-English answer

**Keep the cap at 2.5.** Every higher cap fails the pre-registered rule.

- The setups a higher cap would let back in (ATR 2.5 → cap) **lose money on average** at every cap from 3.0 up. Each cap makes the year book smaller than 2.5 does: 3.0 gives −$3.6k, 3.2 gives −$4.7k and 3.5 gives −$4.0k, against $10.3k at 2.5.
- The only cap that is not negative is **2.75**, and it is break-even at best: +0.025 %/fill, CI −0.76…+0.87. It wins Jan–May (+0.46) and loses Jun–Sep (−0.41), and its book still ends **$241 below** 2.5's. It is not an edge.
- The out-of-sample check fails both ways:
  - Choosing on Jan–May picks 2.75, which then loses in Jun–Sep (−0.41 %/fill).
  - Choosing on Jun–Sep picks 3.5, which then loses in Jan–May (−0.51 %/fill).
- The ATR dose-response slopes down: higher ATR, worse trades. Trade-level Spearman is −0.18. The 3–3.5 band is a flat bump (+0.05) between two losing bands, the same non-monotone bump seen in the 10-06 WIDE study.
  - The drop from ≤2.5 to 2.5–3.5 is −0.52 %/fill. A shuffled-ATR null puts that at p ≈ 0.05.
- Even so, the re-admitted cohort is **not a proven loser** under the expectancy bar. At caps 3.0–4.0 its mean is negative with only 74–88 % confidence, not the 95 % required. Only "no cap at all" is a proven losing cohort (P(mean<0) = 0.997).
  - So 2.5 is not a proven filter edge either. But nothing is proven in favour of moving it, and the locked rule is "keep 2.5 unless the marginal band is non-negative out-of-sample both ways". It is not.
- **OGN 10-08 10:45 is one fill.**
  - Its ATR was 2.97 %, rebuilt exactly here, and the candle was red. So at a cap of 3.0, FRENZY_LONG would have entered.
  - On 1m klines, the live lock (+3 arm, 2-point trail) would probably have exited around **+3.7 % at 10:48**. It would not have ridden the +20–39 % run. The lock, not the ATR cap, decides how much of a runner is captured.
  - The year shows the 2.5–3 band losing on average (−0.33 %/fill, 95 fills). One big winner does not change that.
- **Watch item, not a ship.** Over Jul–Sep, the FRENZY_LONG (red candle) part of the marginal band is mildly positive: cap 3.0 gives +0.23 on 28 fills, and cap 3.5 gives +0.46 on 47. Jan–May was strongly negative (−0.80 / −0.77).
  - This could be regime drift. If the operator wants to follow it, the pre-registered way is an **observe-only** tracker: FRENZY_ATR_HIGH red refusals with 2.5 < ATR ≤ 3.5, re-priced with the live lock, frozen at 3.5 and never re-fit. Revisit it only if it clears the expectancy bar on fresh fills.

## 0. Method and parity

| item | detail |
|---|---|
| Pre-registration | `scratchpad/atr_cap/PREREG_ATR_CAP.txt`, written 2026-10-08T12:58:08Z before any P&L (sha256 0fc1853d…). Thresholds {2.5, 2.75, 3.0, 3.2, 3.5, 4.0, off}, bands, rulers, OOS, null and decision rule are as specified. |
| Cohort | Engine-parity fresh-ON cohort `reports/FRENZY_ENGINE_COHORT_2026-10-05.csv` (via `wrev/wlib.load`), **all 1,819 signals incl. 927 FRENZY_ATR_HIGH**, Jan 10 – Sep 27 2026. |
| ATR parity | The stamped `atr` equals the engine's `wilder_atr_pct(closed[-300:])` rebuilt from `k5m_full` on **1,819 / 1,819** signals (max error 4e-15, 0 side flips at any cap). Live checks on fresh klines: **OGN 10-08 bar 10:40 (close 10:45) = 2.972 %**, red (−0.62 %). **GRIFFAIN 10-06 close 17:40 = 3.630 %**, red. Both match the live stamps. |
| Re-decision per cap c | ATR > c gives ATR_HIGH (no LONG, and WIDE refuses ATR_HIGH under hold-green). Otherwise a green candle (bar_ret > 0) gives GREEN_BAR, which goes to WIDE only if above_streak > 12. Otherwise (red) FRENZY_LONG enters. **So raising the cap re-admits red setups to FRENZY_LONG and hold-green setups to WIDE**, because both read the same field. |
| Live gates | live-eligible pair, engine-parity market volume U2 < 1.0, dislocation > 1 % re-decided at the 8 s entry (30 refused), and sequencing: 2 slots per sleeve, one position per pair, 3 per pair per day per sleeve. Gated per-signal pool: 595. |
| Pricing (primary) | Ticks, entry on the first print ≥ close + 8 s, live LOCK (−3 until +3, then max(+2, peak − 2)), fees 0.09, exit slip 0.10, 12 h cap. Taken from `latency/priced_all.pkl` d = 8. **Parity: that pricer at d = 12 reproduces the cohort's LOCK2 on 1,608 / 1,608 tick fills exactly.** 15 gated fills priced only on 1m fall back to the 12 s LOCK2 price. |
| Book | $3,000, shared equity, FRENZY_LONG lev 0.5 when strong (ADX Δ > 0 ∧ DI > 0) else 0.32, WIDE 0.2. Same BASEF sizing as the 10-06 studies. |
| CIs | Day-block bootstrap (4,000 draws). Expectancy-bar confidence also clustered by episode (pair + spike). |

## 1. ATR bands (per signal, gated, unsequenced; primary ruler; 1× %)

Stop-out = exit ≤ −3 %. "Halves" = Jan–May / Jun–Sep.

### LONG (red) + WIDE hold-green combined
| band | N | days | WR | avg % | day 95% CI | Jan–May / Jun–Sep | worst | stop-out | sum % |
|---|---|---|---|---|---|---|---|---|---|
| 0–1.5 | 52 | 42 | 65% | **+1.006** | [+0.14, +1.89] | +1.12 (43) / +0.47 (9) | -3.18 | 35% | +52.3 |
| 1.5–2 | 117 | 83 | 54% | **+0.245** | [-0.34, +0.79] | +0.38 (77) / -0.01 (40) | -3.33 | 46% | +28.7 |
| 2–2.5 | 164 | 122 | 54% | **+0.219** | [-0.34, +0.74] | +0.18 (98) / +0.27 (66) | -3.34 | 46% | +35.9 |
| 2.5–3 | 95 | 77 | 49% | **-0.328** | [-0.87, +0.27] | -0.75 (49) / +0.12 (46) | -3.18 | 51% | -31.1 |
| 3–3.5 | 68 | 56 | 51% | **+0.045** | [-0.78, +0.91] | -0.24 (35) / +0.35 (33) | -3.18 | 49% | +3.1 |
| 3.5–4 | 38 | 34 | 39% | **-0.809** | [-1.69, +0.13] | -0.95 (19) / -0.67 (19) | -3.16 | 61% | -30.7 |
| >4 | 61 | 42 | 33% | **-1.352** | [-1.98, -0.72] | -1.27 (34) / -1.45 (27) | -3.18 | 66% | -82.5 |
| ≤2.5 (live) | 333 | 173 | 56% | **+0.351** | [-0.02, +0.70] | +0.43 (218) / +0.19 (115) | -3.34 | 44% | +116.8 |
| >2.5 (all refused) | 262 | 145 | 45% | **-0.539** | [-0.91, -0.15] | -0.78 (137) / -0.28 (125) | -3.18 | 55% | -141.3 |

trade-level Spearman(ATR, P&L) = -0.178

### FRENZY_LONG only (red candle)
| band | N | days | WR | avg % | day 95% CI | Jan–May / Jun–Sep | worst | stop-out | sum % |
|---|---|---|---|---|---|---|---|---|---|
| 0–1.5 | 28 | 25 | 54% | **+0.106** | [-1.08, +1.27] | +0.51 (22) / -1.36 (6) | -3.18 | 46% | +3.0 |
| 1.5–2 | 75 | 60 | 57% | **+0.507** | [-0.20, +1.19] | +0.44 (51) / +0.65 (24) | -3.33 | 43% | +38.0 |
| 2–2.5 | 105 | 88 | 52% | **+0.295** | [-0.41, +1.00] | +0.27 (65) / +0.33 (40) | -3.34 | 48% | +31.0 |
| 2.5–3 | 63 | 53 | 52% | **-0.223** | [-0.88, +0.41] | -0.80 (28) / +0.24 (35) | -3.18 | 48% | -14.1 |
| 3–3.5 | 47 | 40 | 47% | **-0.229** | [-1.20, +0.86] | -0.84 (19) / +0.19 (28) | -3.18 | 53% | -10.8 |
| 3.5–4 | 31 | 29 | 45% | **-0.614** | [-1.57, +0.29] | -0.37 (15) / -0.84 (16) | -3.16 | 55% | -19.0 |
| >4 | 45 | 32 | 36% | **-1.234** | [-1.95, -0.57] | -1.17 (27) / -1.33 (18) | -3.18 | 64% | -55.5 |
| ≤2.5 (live) | 208 | 135 | 54% | **+0.346** | [-0.10, +0.79] | +0.37 (138) / +0.30 (70) | -3.34 | 46% | +71.9 |
| >2.5 (all refused) | 186 | 118 | 46% | **-0.534** | [-0.94, -0.11] | -0.85 (89) / -0.25 (97) | -3.18 | 54% | -99.4 |

trade-level Spearman(ATR, P&L) = -0.154

### WIDE hold-green only
| band | N | days | WR | avg % | day 95% CI | Jan–May / Jun–Sep | worst | stop-out | sum % |
|---|---|---|---|---|---|---|---|---|---|
| 0–1.5 | 24 | 21 | 79% | **+2.055** | [+0.82, +3.53] | +1.76 (21) / +4.13 (3) | -3.12 | 21% | +49.3 |
| 1.5–2 | 42 | 37 | 48% | **-0.222** | [-1.12, +0.70] | +0.25 (26) / -0.99 (16) | -3.12 | 52% | -9.3 |
| 2–2.5 | 59 | 49 | 56% | **+0.083** | [-0.68, +0.86] | +0.00 (33) / +0.18 (26) | -3.13 | 44% | +4.9 |
| 2.5–3 | 32 | 29 | 44% | **-0.533** | [-1.65, +0.77] | -0.68 (21) / -0.25 (11) | -3.16 | 56% | -17.1 |
| 3–3.5 | 21 | 19 | 62% | **+0.658** | [-0.66, +2.07] | +0.46 (16) / +1.28 (5) | -3.15 | 38% | +13.8 |
| 3.5–4 | 7 | 7 | 14% | **-1.673** | [-3.13, +1.22] | -3.12 (4) / +0.26 (3) | -3.15 | 86% | -11.7 |
| >4 | 16 | 15 | 25% | **-1.685** | [-2.68, -0.43] | -1.69 (7) / -1.68 (9) | -3.14 | 69% | -27.0 |
| ≤2.5 (live) | 125 | 93 | 58% | **+0.359** | [-0.21, +0.94] | +0.55 (80) / +0.03 (45) | -3.13 | 42% | +44.9 |
| >2.5 (all refused) | 76 | 60 | 42% | **-0.552** | [-1.25, +0.20] | -0.65 (48) / -0.38 (28) | -3.16 | 57% | -41.9 |

trade-level Spearman(ATR, P&L) = -0.224

How to read the bands:
- **The stop-out share rises with ATR**: 35 % at <1.5, ~46 % at 1.5–2.5, 51 % at 2.5–3, 61–66 % above 3.5. This fits a −3 % stop sitting inside normal noise once ATR passes ~2.5–3.
- 3–3.5 is a flat bump (+0.05) between 2.5–3 (−0.33) and 3.5–4 (−0.81). As in the 10-06 WIDE study, it is not monotone above 2.5. Under the locked rule a mid-range bump between losing flanks is a confound, not a door.

## 2. Thresholds (sequenced live book, primary ruler)

Marginal = fills in that cap's book with ATR > 2.5, which is exactly what the cap change re-admits. Δ is measured against the 2.5 book, the span is 261 days, and Δ book is the compounded $ difference at live sizing.

| cap | fills (L/W) | avg % | sum % | book end / maxDD | marginal N (L/W) | marg days | marg WR | marg avg | marg day CI | marg Jan–May / Jun–Sep | Δ sum % vs 2.5 | Δ %/day | Δ book vs 2.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.5 | 331 (207/124) | +0.322 | +106.7 | $10,254 / -48% | 0 | | | | |  | +0.0 | +0.000 | $+0 |
| 2.75 | 381 (242/139) | +0.283 | +108.0 | $10,013 / -47% | 50 (35/15) | 47 | 56% | **+0.025** | [-0.76, +0.87] | +0.46 (25) / -0.41 (25) | +1.3 | +0.005 | $-241 |
| 3 | 426 (270/156) | +0.177 | +75.6 | $6,695 / -61% | 95 (63/32) | 77 | 49% | **-0.328** | [-0.89, +0.24] | -0.75 (49) / +0.12 (46) | -31.1 | -0.119 | $-3,559 |
| 3.2 | 454 (292/162) | +0.149 | +67.5 | $5,555 / -65% | 123 (85/38) | 92 | 49% | **-0.319** | [-0.86, +0.23] | -0.57 (63) / -0.06 (60) | -39.2 | -0.150 | $-4,699 |
| 3.5 | 493 (316/177) | +0.166 | +81.8 | $6,303 / -69% | 162 (109/53) | 116 | 51% | **-0.154** | [-0.64, +0.36] | -0.51 (83) / +0.22 (79) | -24.9 | -0.095 | $-3,951 |
| 4 | 530 (346/184) | +0.102 | +54.2 | $4,546 / -73% | 200 (140/60) | 129 | 48% | **-0.278** | [-0.73, +0.20] | -0.59 (102) / +0.05 (98) | -52.5 | -0.201 | $-5,708 |
| off | 589 (389/200) | -0.055 | -32.2 | $1,343 / -89% | 261 (185/76) | 145 | 45% | **-0.529** | [-0.89, -0.13] | -0.76 (136) / -0.28 (125) | -139.0 | -0.532 | $-8,911 |

By role (marginal, sequenced): mean · day CI · Jan–May / Jun–Sep · Jul–Sep:

| cap | FRENZY_LONG (red) | WIDE hold-green |
|---|---|---|
| 2.75 | 35 · −0.064 [−1.00, +0.79] · +0.16 / −0.23 · −0.19 (16) | 15 · +0.233 [−1.46, +2.23] · +0.90 / −1.11 |
| 3.0 | 63 · −0.223 [−0.88, +0.42] · −0.80 / +0.24 · +0.23 (28) | 32 · −0.533 [−1.67, +0.70] · −0.68 / −0.25 |
| 3.2 | 85 · −0.266 [−0.89, +0.35] · −0.67 / +0.05 · +0.12 (38) | 38 · −0.437 [−1.46, +0.68] · −0.41 / −0.49 |
| 3.5 | 109 · −0.199 [−0.78, +0.41] · −0.77 / +0.22 · +0.46 (47) | 53 · −0.061 [−0.88, +0.87] · −0.19 / +0.23 |
| 4.0 | 140 · −0.291 [−0.80, +0.24] · −0.67 / +0.00 · +0.15 (61) | 60 · −0.249 [−1.07, +0.66] · −0.47 / +0.23 |
| off | 185 · −0.520 [−0.93, −0.10] · −0.82 / −0.25 · −0.14 (76) | 76 · −0.552 [−1.25, +0.20] · −0.65 / −0.38 |

Concentration and months:
- **Cap 3.0:** 4 of 9 months positive. The top 5 fills carry +28.5 of a −31.1 net, and without them the mean is −0.66.
- **Cap 3.5:** 4 of 9 months positive. Without its top 5 fills the mean is −0.42.

## 3. Expectancy bar on the marginal cohort (is it a LOSING cohort?)

Breakeven WR on the 2.5 kept sequenced fills: **FRENZY_LONG 49.1 %** (207 fills, +0.317) and **WIDE-HG 51.7 %** (124 fills, +0.331).

| cap | N | WR vs BE (mix-weighted) | mean | P(mean<0) day-cluster | P(mean<0) episode-cluster | windows (days) | worst day share of loss | worst pair share | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2.75 | 50 | 56% vs 50% | +0.025 | 0.476 | 0.482 | 47 | n/a (net > 0) | n/a | not a losing cohort |
| 3 | 95 | 49% vs 50% | -0.328 | 0.872 | 0.861 | 77 | 20% | 20% | negative but not proven |
| 3.2 | 123 | 49% vs 50% | -0.319 | 0.880 | 0.893 | 92 | 24% | 16% | negative but not proven |
| 3.5 | 162 | 51% vs 50% | -0.154 | 0.736 | 0.738 | 116 | 38% | 37% | negative but not proven |
| 4 | 200 | 48% vs 50% | -0.278 | 0.883 | 0.895 | 129 | 22% | 22% | negative but not proven |
| off | 261 | 45% vs 50% | -0.529 | 0.997 | 0.996 | 145 | 11% | 9% | LOSING cohort (all 4 legs) |

Reading the expectancy table:
- At caps 2.75–4.0 the marginal cohort is **negative but not a proven loser**. Its confidence is 74–88 %, short of 95 %, and its WR sits right on breakeven.
- Only "off" passes all four legs.
- So the case for 2.5 is a weak-evidence keep, not a proven filter. But the case for moving is weaker still. Nothing re-admitted earns money in-sample except 2.75, and that is ≈ 0.

## 4. Out-of-sample, null, haircut

**OOS selection** (choose the cap with the best marginal sum on one half, confirm on the other):
- Train Jan–May picks 2.75 (+11.5); its Jun–Sep marginal is **−0.41** (25 fills, CI −1.45…+0.59). **Fail.**
- Train Jun–Sep picks 3.5 (+17.2); its Jan–May marginal is **−0.51** (83 fills, CI −1.11…+0.14). **Fail.**

**Shuffled null** (ATR permuted within role, 2,000 times): the observed mean of (2.5–3.5] minus ≤2.5 is **−0.523**. The null 95 % range is −0.62…+0.60, and P(null ≤ obs) = 0.049. The ATR ordering carries real information, and it points against higher caps.

**Haircut:** the only positive in-sample Δ is 2.75: +1.3 %-points of summed P&L over 261 days, about +0.005 %/day. After a 30–50 % haircut that is ≈ 0, and the compounded book is already −$241. Nothing remains to haircut at higher caps, since all are negative.

## 5. Declared secondaries (cannot trigger a ship)

| variant | cap 2.75 marginal | cap 3.0 | cap 3.2 | cap 3.5 | OOS (A→B / B→A) |
|---|---|---|---|---|---|
| **ATR-scaled stop** = −max(3, 1.2×ATR), same lock (fills ≤ 2.5 unchanged by construction) | −0.033 · book −$670 | −0.357 · −$3,916 | −0.378 · −$5,264 | −0.159 · −$3,995 | 2.75 → −0.47 / 3.5 → −0.46: fail |
| **Fixed +3 / −3** (all fills on this exit; 2.5 book $6,352) | +0.222 · book **+$1,140** | −0.154 · −$903 | −0.192 · −$1,899 | −0.078 · −$1,411 | 2.75 → +0.02 / 3.5 → −0.46: fail (reverse) |

What the secondaries show:
- A wider, ATR-scaled stop **does not rescue** the high-ATR band. Stop-outs fall, but the deeper losses offset them; above 4 % ATR the worst fill is −9.9 %.
- Under a fixed +3/−3 exit, 2.75 looks mildly positive. That exit is not live, the result fails the reverse OOS, and it is a secondary.

## 6. OGN 10-08 (the trigger case)

- ATR was 2.972 % on the 10:40 bar (closed 10:45), and the bar was red (−0.62 %). FRENZY_LONG would have entered at a cap of 3.0 or higher.
- On public 1m klines (ticks not cached), entry was ≈ 0.0306. The trade reached +3 within ~2 min, and the peak was +39 % at 12:58.
- A 1m walk of the live lock (prior-bar peak, low checked first) exits **≈ +3.7 % at 10:48**: the 2-point trail is hit on the first pullback after +5.7.
- So the bigger cost on runner days is the lock's tight trail, which earlier studies found is net-positive on the year. The ATR cap cost little here.
- One fill is not evidence. The 2.5–3 band is −0.33 %/fill over 95 sequenced fills.

## 7. Limitations / blind spots

- The cohort ends Sep 27. Oct 1–8 live fills are not in it, apart from the OGN and GRIFFAIN ATR parity checks.
- 15 gated fills were priced on 1m at 12 s, not on ticks at 8 s (both rulers carry 0.10 slip).
- The FRENZY catch-up path, LITE and the max-open-positions interaction with the momentum sleeves are not modelled. Only FRENZY_LONG and WIDE are sequenced.
- The band table is per signal (unsequenced). The threshold table is the sequenced book.
- The expectancy "window" is the calendar day, with episode clustering as a sensitivity check. Both agree.
- Binance requests: **3** public kline calls (OGN 5m, GRIFFAIN 5m, OGN 1m). The 1-minute weight peaked at 91 (89 already in use on the IP before the first call), with no 418 or 429. Everything else came from local caches.

Scripts and outputs are in `scratchpad/atr_cap/`: `PREREG_ATR_CAP.txt`, `atr_parity.py`, `price_sec.py`, `core.py`, `analyze.py`, `out_PRI.md`, `out_ASTOP8.md`, `out_FIX38.md` and `fetch/`.
