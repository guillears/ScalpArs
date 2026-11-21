# DIST_BUCKET_EXIT — should DIST use a different TP / SL depending on how far the coin fell before entry? (2026-11-21)

Research only. No bot code, config or engine touched.
Pre-registration: `reports/DIST_BUCKET_EXIT_2026-11-21_PREREG.md` (written before the grid was simulated).
Scripts: `scripts/DIST_BUCKET_EXIT_2026-11-21_sim.py` → `_eval.py` → `_real.py`.
Data and full tables: `reports/DIST_BUCKET_EXIT_2026-11-21_data/` (`tables.md` = every grid, pick, read and control; `grid_by_bucket.csv`, `picks.csv`, `control.csv`, `stop_read.csv`, `real_fills.csv`, `sim_grid.npz`).

## Verdict: **KEEP ONE EXIT (today's Exit B for every DIST fill).**
- The bucket-specific exit set loses on HOLDOUT: −0.020 %/fill (95 % CI −0.25 … +0.24), DCR +0.853 → +0.802 %/day, max drawdown −55 % → −76 %.
- It does not beat pure curve-fitting: shuffled buckets give a 95th-percentile HOLDOUT gain of +0.073; the real buckets give −0.020 (80th percentile — in the noise).
- Gates: CI ✗ · beats control ✗ · halves 2/3 ✓ · 0.20 % slip ✗ · DCR ✗ · concentration n/a (no gain) → 1 of 6.

## 1. Question
After BLESS (−22 % in the 15 min before entry, stopped −6 %) and the DIST_AFTERCRASH result (deep pre-entry falls do BETTER under today's exit, with more TP hits), the operator asked: should those fills get a different take-profit or stop than the rest?

## 2. Prior work and what is new
- **DL 602 (DIST_LOSERS2):** a LOWER per-cohort TP (3–8 %) lost for the whole cohort and every sub-cohort on HOLDOUT; Exit B kept for every fill. **DL 603:** TP grids, no change. **DIST_THEORY_EXIT:** fixed wider stops (8–15 %) fail on HOLDOUT.
- **New here:** cohorts defined by the pre-entry fall (`r15`), TPs ABOVE today's 10 % (12 / 15 %), a joint TP × SL grid per cohort, and a shuffled-bucket control that measures how much "gain" pure fitting produces.
- **Entry × exit joint check:** no DIST entry change is pending (DIST_AFTERCRASH refuted the after-crash filter), so the exit is read under today's entry stack.

## 3. Method
- Cohort: the same 1,547 DIST fills (today's stack, RSI1h ≥ 74.07, one position per pair, 2023-07 → 2026-11). DESIGN < 2025-07 (559) · HOLDOUT ≥ 2025-07 (988).
- Engine: the 1m-path engine of DIST_THEORY_EXIT (stop-first on the high, maker TP one tick through, 24 h, real funding). Parity: the (today, today) cell reproduces that sim's REF on **1,547 / 1,547** fills.
- Nets: TP maker 0.02 %, SL / time taker 0.05 % + modelled slip (≈ 0.10 %) on both sides; stress at flat 0.20 % slip.
- Buckets (frozen): `r15` ≤ −15 · −15 … −10 · −10 … −5 · > −5 %. DESIGN / HOLDOUT fills: 5 / 70 · 18 / 100 · 86 / 200 · 450 / 618.
- Grid per bucket: TP {today = min(20×ATR, 10 %) · 5 · 8 · 10 · 12 · 15 %} × SL {today = min(1.5×ATR, 6 %) · 4 · 6 · 8 · 10 %} = 30 exits. Today's TP is 10 % on 1,546 / 1,547 fills.
- Pick on DESIGN only (best avg net %/fill; must beat today). **Thin-bucket rule:** a bucket with < 30 DESIGN fills cannot be fitted and keeps today's exit — this covers both crash buckets (5 and 18 DESIGN fills).
- HOLDOUT read once; shuffled control = same procedure with `r15` permuted within each period, 100×.

## 4. DESIGN picks
| bucket (15-min fall) | DESIGN fills | pick | DESIGN avg today → pick |
|---|---|---|---|
| ≤ −15 % | 5 | today (too thin to fit) | +6.58 |
| −15 … −10 % | 18 | today (too thin to fit) | +3.40 |
| −10 … −5 % | 86 | **TP 15 % / SL 10 %** | +0.37 → +1.04 (+0.67) |
| > −5 % | 450 | today (no cell better) | +1.00 |

## 5. HOLDOUT — bucket set vs today's one exit
| bucket | fills | changed | WR before → after | avg %/fill before → after | Δ | 95 % CI | Δ at 0.20 % slip |
|---|---|---|---|---|---|---|---|
| ≤ −15 % | 70 | 0 | 51.4 → 51.4 % | +2.03 → +2.03 | 0 | — | 0 |
| −15 … −10 % | 100 | 0 | 55.0 → 55.0 % | +2.51 → +2.51 | 0 | — | 0 |
| −10 … −5 % | 200 | 184 | 47.5 → 51.5 % | +1.12 → +1.02 | **−0.099** | −1.21 … +1.21 | −0.107 |
| > −5 % | 618 | 0 | 38.2 → 38.2 % | +0.40 → +0.40 | 0 | — | 0 |
| **TOTAL** | 988 | 184 | 42.7 → 43.5 % | +0.877 → +0.857 | **−0.020** | −0.25 … +0.24 | −0.022 |

- By half (TOTAL Δ): 25-H2 −0.147 · 26-H1 +0.150 · 26-H2 +0.013.
- One size (0.61 × equity per fill, every calendar day): **DCR +0.853 → +0.802 %/day** · max DD −54.9 → −76.3 % · worst day −21.8 → −22.1 %. Δ ≈ −$606 at $3,050 per fill.
- The wider 10 % stop wins more trades but the 184 changed fills lose more on the ones it does not save — and the drawdown grows by 21 points.
- The HOLDOUT best cell for −10 … −5 is TP 8 / SL 8 (+1.74), not the DESIGN pick: the best cell moves between periods = no stable exit per bucket.
- **Single re-tuned exit for all fills** (descriptive): the DESIGN best is also TP 15 / SL 10 → HOLDOUT −0.152 %/fill, DCR 0.853 → 0.409. Re-tuning one exit is worse still.

## 6. Overfit control (shuffled buckets, 100×)
- HOLDOUT Δ from fitting random buckets: median −0.164 · 95th percentile +0.073 · max +0.213 · 14 % of shuffles positive.
- Real buckets: −0.020 → 80th percentile. **Does not beat the control** (needed > +0.073).
- Sensitivity (descriptive, not the pre-registered read): letting the thin crash buckets fit on their 5 / 18 DESIGN fills → picks TP 15 / today · TP 12 / SL 8 · TP 15 / SL 10 · today → HOLDOUT total **−0.001** (≤ −15 +0.58 · −15 … −10 −0.22 · −10 … −5 −0.10); its own control 95th pct +0.037. Still nothing.

## 7. Two-sided stop read (backtest; "stopped" = stopped under today's exit, TP kept at today)
| bucket | stop | stopped today | saved (N · $) | deeper (N · $) | smaller loss (N · $) | newly stopped winners (N · $) | total $ |
|---|---|---|---|---|---|---|---|
| ≤ −15 | 4 % | 33 | 0 | 0 | 33 · +2,013 | 11 · −3,814 | −1,801 |
| ≤ −15 | 8 % | 33 | 4 · +1,163 | 29 · −1,830 | 0 | 0 | −666 |
| ≤ −15 | 10 % | 33 | 6 · +2,059 | 27 · −3,355 | 0 | 0 | −1,295 |
| −15 … −10 | 8 % | 52 | 9 · +3,448 | 43 · −2,694 | 0 | 0 | +754 |
| −15 … −10 | 10 % | 52 | 14 · +5,743 | 38 · −4,844 | 0 | 0 | +899 |
| −10 … −5 | 8 % | 157 | 34 · +12,214 | 123 · −10,552 | 0 | 0 | +1,662 |
| −10 … −5 | 10 % | 157 | 50 · +15,051 | 107 · −15,386 | 0 | 0 | −335 |
| > −5 | 4 % | 638 | 25 · +4,959 | 274 · −7,234 | 339 · +12,394 | 42 · −17,275 | −7,155 |
| > −5 | 10 % | 638 | 242 · +69,359 | 396 · −68,449 | 0 | 0 | +911 |
(6 % rows and the rest in `tables.md`.) Fills not stopped today show $0 for every wider stop (replica check passes).
- Wider stops: what the saved fills gain, the deeper fills give back — every bucket nets near zero over 3.4 years ($ ±1.7k on $3,050 tickets), and the 10 % stop adds the drawdown above.
- Tighter 4 % stop: loses in every bucket (it stops winners that went on to TP).
- **BLESS-type bucket (≤ −15):** wider stops LOSE (−$666 / −$1,295). The bounce that stopped BLESS usually keeps going; widening would not have been a fix on average.

## 8. Real fills (6 DIST, 11-19 → 11-21, re-priced from the real fill price; direction only)
| pair | opened | booked % | r15 % | bucket | today (sim) | bucket exit | TP 15 / SL 10 |
|---|---|---|---|---|---|---|---|
| SENT | 11-19 04:15 | −4.41 | −1.4 | > −5 | −4.40 SL | same (today) | −10.30 SL |
| NAORIS | 11-19 09:15 | −5.97 | −0.8 | > −5 | −6.30 SL | same | −10.30 SL |
| EVAA | 11-19 14:31 | −4.38 | −2.0 | > −5 | −4.88 SL | same | −10.09 SL |
| CYBER | 11-20 09:00 | +9.95 | −1.1 | > −5 | +9.83 TP | same | +13.92 TP |
| KERNEL | 11-20 12:30 | −6.11 | −1.7 | > −5 | −6.30 SL | same | −10.30 SL |
| BLESS | 11-21 09:30 | −6.03 | −22.0 | ≤ −15 | −6.30 SL | same (thin bucket) | +14.83 TP |
- The bucket set changes none of the 6 real fills (five are in the > −5 bucket, BLESS's bucket keeps today's exit).
- BLESS would have been a +14.8 % winner with TP 15 / SL 10 — **that is hindsight on 1 fill**. On the 75 backtest fills of its bucket a 10 % stop loses −$1,295 vs today, and the same exit would have turned the other four losers into −10 % losses.

## 9. Verdict and recommendation
**KEEP ONE EXIT.** The bucket exit set fails 5 of 6 pre-committed gates on HOLDOUT, and its result is indistinguishable from fitting random buckets. No ship, no observe line. No positive Δ, so no haircut applies.

What it adds to DL 602: DL 602 found no per-cohort LOWER TP; this study finds no per-cohort HIGHER TP or wider / tighter stop by the pre-entry fall either.

Watch idea only (not a line): in the deepest-fall bucket the larger TP (12–15 %) looked better on HOLDOUT (+2.03 → +2.46 / +2.60, 70 fills), but it could not be fitted on DESIGN (5 fills) and the next bucket (−15 … −10) goes the other way, so the pooled read is zero.

Caveats:
- Backtest only: 0 master DIST fills; the 6 real fills are a direction check.
- DESIGN holds only 23 crash fills, so the crash buckets could never be fitted out of sample — the honest result for them is "untestable on DESIGN, flat on pooled sensitivity".
- Slip modelled at 0.10 %; crash coins may be thinner (the 0.20 % stress changes nothing).
- HOLDOUT for this cohort has now been read many times (DL 602, 603, AFTERCRASH, this study): any later cut on it is in-sample.

**Plain English for the operator.**
- We tested 30 combinations of take-profit and stop for each "how hard did it fall before entry" group, chose the best on 2023–2025 data, then checked on the next 16 months.
- The chosen set made slightly less money (−0.02 % per trade), lowered the daily compound (0.85 → 0.80 %/day) and made the worst drawdown much deeper (−55 % → −76 %).
- Picking exits for random groups produced the same kind of result — so any group-by-group exit is just fitting noise.
- A wider stop would have saved BLESS, but across the 75 BLESS-like trades it costs money: the bounce usually keeps going.
- **Recommendation: change nothing. One exit for every DIST trade. The open lever stays the size per stop (already on the pre-live list).**
