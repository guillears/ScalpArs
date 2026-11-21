# DIST_AFTERCRASH — do DIST shorts taken after the coin already crashed lose? (2026-11-21)

Research only. No bot code, config or engine touched.
Pre-registration: `reports/DIST_AFTERCRASH_2026-11-21_PREREG.md` (written before any outcome was read).
Scripts: `scripts/DIST_AFTERCRASH_2026-11-21_features.py` → `scripts/DIST_AFTERCRASH_2026-11-21_eval.py`.
Data and full tables: `reports/DIST_AFTERCRASH_2026-11-21_data/` (`tables.md`, `gates.csv`, `real_fills.csv`, `features.csv.gz`, `paths.npy`). Chart: `reports/DIST_AFTERCRASH_2026-11-21.png`.

## Verdict: **REFUTED, in the opposite direction.**
DIST shorts taken after a hard crash do **better** than the rest of the sleeve, not worse.
- Signal candle ≤ −15 % (BLESS-type): 75 fills · WR 53 % · **+2.33 %/trade** vs +0.86 % for the rest. HOLDOUT +2.03 (70 fills).
- Signal candle ≤ −20 %: 34 fills · WR 62 % · **+3.67 %**.
- The blocked cohort passes 3 of the 5 legs of the filter bar. It fails the two that matter: its WR is above breakeven, and its average is positive with near certainty.
- Blocking it would **cut** the daily compound: HOLDOUT +0.86 → +0.71 %/day at one size.
- BLESS was a loser from a cohort that wins more often than the sleeve does. **No filter, no observe line.**

## 1. Question
Operator, on BLESS (2026-11-21 09:30 UTC):
- The coin fell −21 % in the 15 min before entry (−24 % in 60 min), with $352k of long liquidations on a ~$7k book.
- The short was stopped at −6 % 7 min later, at the top of a bounce. Price was back at entry ~15 min after.
- Operator's ask: "check it in backtest, maybe that's something, it makes theoretical sense."

## 2. Theory (stated in the PREREG before the read)
- DIST shorts a big holder distributing into a pump. The edge is the supply still to come.
- If the dump already happened (capitulation, long-liquidation cascade done), the forced sellers are spent.
- The short then sells the low into a reflex bounce, and the stop sits on that bounce.
- Prediction: the deeper the fall before entry, the worse the short (a monotone dose), worst at the extreme tail.

## 3. Prior work (cited, not redone)
- **DIST_WL (2026-11-18, pre-RSI L1 cohort).** "Already dumped" was a theory branch (d). Far below peak / VWAP, late in the episode and a negative 1 h change all pointed the right way, but weakly (|t| ≤ 1.9 on net, under the null).
- **DIST_LOSERS2 (2026-11-20, today's 1,547-fill kept cohort, 112 variables, quintiles frozen on DESIGN).** Each of these already did **better** on HOLDOUT, not worse:
  - `bar_ret_pct` Q1 (signal candle ≤ −4.8 %): +1.70 %, t +2.7.
  - `dd_peak_pct` Q1 (≥ 8.4 % below the episode peak): +1.74 %, t +3.3.
  - `ret1h_pct` Q1: +1.60 %.
  - DL 602 concluded "no entry filter".
- **EXH analogue.** EXH_AFTER_DROP (EXH shorts after a −5…−10 % hour) is non-monotonic and observe-only. EXH_BADDAY_2026-11-18 is a different sleeve and not evidence here.
- **Route B** (real liquidations, ROUTE_B_REPORT §2a, the BLESS row) only starts 2026-11-19, so it cannot be used on the backtest. Here volume is the proxy for a cascade.
- **What is new here:**
  - The extreme tail (≤ −10 / −15 / −20 %, the BLESS signature), which the quintile screens averaged into a −4.8 % bucket.
  - The 30-min window and the fall from the 60-min high.
  - A liquidation-cascade proxy: signal-candle volume vs its 24 h norm.
  - The event-study path.

## 4. Method
- **Cohort.**
  - 1,547 DIST fills on today's stack: RSI1h ≥ 74.07 (DL 586), one position per pair, 2023-07 → 2026-11.
  - Exit B: TP min(20×ATR, 10) maker / SL min(1.5×ATR, 6) / 24 h, 0.10 % slip, real funding.
  - P&L = `REF` net % from DIST_THEORY_EXIT `perfill.csv.gz`.
  - Split: DESIGN < 2025-07 (559 fills) · HOLDOUT ≥ 2025-07 (988 fills).
- **Variables** (5m bars closed at or before the signal; no look-ahead):
  - `r15` / `r30` / `r60` / `r240`: return over the 15 / 30 / 60 / 240 min before entry.
  - `drop60` / `dd24`: how far price sits below the 60-min / 24 h high.
  - `volx15`: signal-15m quote volume ÷ the median 15m volume of the prior 24 h.
- **Lag check.**
  - `r60` equals the stored `ret1h_pct` to 6e-6.
  - `r240` equals `ret4h_pct`.
  - corr(`r15`, signal `bar_ret_pct`) = 0.99996.
- **Thresholds fixed by theory, not fitted.**
  - Only 5 DESIGN fills fall ≤ −15 %, too few to pick a cut on DESIGN.
  - So the read is on all fills, with DESIGN and HOLDOUT shown side by side.
  - Primary candidate: `r15 ≤ −15`. 14 family cuts plus the liquidation composite.
- **Gates.**
  - The locked Pattern-C expectancy bar, using a 4h-window-clustered bootstrap (5,000 resamples).
  - A family-wise shuffled null: the variable is shuffled within half-years, 2,000×, and the max |t| is taken over the 15 cuts.
  - Monotonicity, LOO by window, and DCR before → after at one size (0.61 × equity per fill).

## 5. Dose-response: primary `r15` (the signal candle)
Sleeve: 1,547 fills · WR 42.5 % · avg +0.93 % · breakeven WR 35.6 %.

| r15 bucket | ALL N · WR · avg | DESIGN N · avg | HOLDOUT N · WR · avg | HOL BTC bull N · avg | HOL BTC bear N · avg |
|---|---|---|---|---|---|
| ≤ −20 % | 34 · 62 % · **+3.67** | 2 · +1.77 | 32 · 62 % · +3.79 | 12 · +4.47 | 20 · +3.38 |
| −20 … −15 | 41 · 46 % · +1.22 | 3 · +9.78 | 38 · 42 % · +0.55 | 19 · +1.34 | 19 · −0.25 |
| −15 … −10 | 118 · 56 % · **+2.65** | 18 · +3.40 | 100 · 55 % · +2.51 | 44 · +0.70 | 56 · +3.93 |
| −10 … −5 | 286 · 45 % · +0.90 | 86 · +0.37 | 200 · 48 % · +1.12 | 94 · +0.38 | 105 · +1.70 |
| > −5 (most fills) | 1,068 · 40 % · +0.65 | 450 · +1.00 | 618 · 38 % · +0.40 | 287 · +0.35 | 331 · +0.45 |

How to read it:
- **The dose runs the wrong way for the theory.** The harder the crash, the better the short.
- The one red cell is HOLDOUT BTC-bear −20…−15 (19 fills, −0.25). It is flanked by +3.38 and +3.93, so it is a non-monotonic hole: a confound under our rules, and 19 fills is noise.
- BTC daily regime at sign granularity: positive in both bull and bear. Nothing flips.
- **Secondary variables, same picture** (full tables in `data/tables.md`):

| variable | worst-fall bucket (ALL) | rest of the sleeve |
|---|---|---|
| `r30` ≤ −20 % | 34 fills · +2.71 | +0.68 (> −5 % bucket) |
| `r60` ≤ −20 % | 18 fills · +3.57 | |
| `drop60` ≤ −20 % | 114 fills · +1.93 | +0.83 (> −5 % bucket) |
| `dd24` ≤ −30 % | 40 fills · +2.97 | +0.60 (> −10 % bucket) |
| `r240` < 0 | 77 fills · +2.67 | +0.84 (≥ 0) |

- **Liquidation proxy.** A crash with a volume spike (`r15` ≤ −10 ∧ `volx15` top tercile ≥ 157×): 90 fills · +1.53.
  - The same crash without the spike: 103 fills · +3.39.
  - So the cascade-like crashes are weaker than other crashes, but still above the no-crash rest (+0.60 … +0.94).
  - Not negative, so not a filter.
- **Exit mix of the ≤ −15 % cohort:** 52 % hit TP (rest 34 %), 44 % stopped (rest 58 %), 4 % timed out.

## 6. Gate table: locked Pattern-C → FILTER expectancy bar (blocked = the after-crash cohort)
Legs: ① WR < breakeven 35.6 % · ② avg < 0 with ≥ 95 % confidence (4h-window bootstrap upper bound < 0) · ③ ≥ 8 windows · ④ no window or pair ≥ 50 % of the loss · ⑤ N ≥ 15.

| would block | N (DES/HOL) | WR | avg % | rest avg % | DES / HOL avg | 95 % upper | t vs rest | legs |
|---|---|---|---|---|---|---|---|---|
| **r15 ≤ −15 (primary)** | 75 (5/70) | 53 % | **+2.33** | +0.86 | +6.58 / +2.03 | +3.89 | +1.57 | 3/5: ①✗ ②✗ |
| r15 ≤ −10 | 193 (23/170) | 55 % | +2.53 | +0.70 | +4.09 / +2.31 | +3.50 | +3.02 | 3/5: ①✗ ②✗ |
| r15 ≤ −20 | 34 (2/32) | 62 % | +3.67 | +0.87 | +1.77 / +3.79 | +6.03 | +2.03 | 3/5: ①✗ ②✗ |
| r30 ≤ −15 | 60 (5/55) | 57 % | +2.86 | +0.85 | +6.61 / +2.52 | +4.48 | +1.94 | 3/5: ①✗ ②✗ |
| r60 ≤ −15 | 37 (5/32) | 54 % | +2.46 | +0.89 | +2.81 / +2.40 | +4.66 | +1.16 | 3/5: ①✗ ②✗ |
| drop60 ≤ −15 | 217 (28/189) | 47 % | +1.34 | +0.86 | +1.07 / +1.38 | +2.22 | +0.82 | 3/5: ①✗ ②✗ |
| dd24 ≤ −30 | 40 (4/36) | 57 % | +2.97 | +0.88 | +5.80 / +2.66 | +5.16 | +1.63 | 3/5: ①✗ ②✗ |
| liquidation proxy | 90 (10/80) | 49 % | +1.53 | +0.89 | +6.32 / +0.93 | +2.93 | +0.74 | 3/5: ①✗ ②✗ |

All 15 cuts are in `data/gates.csv`. **None** has WR below breakeven or a negative average, in DESIGN or HOLDOUT.
- **Shuffled null.** The family-wise 95 % max |t| is 2.97. Only `r15 ≤ −10` crosses it (+3.02), and it crosses in the **positive** direction: the after-crash fills did better. That is a refutation of the theory, not a filter.
- **LOO (primary).** Removing any one 4h window leaves the average between +2.13 and +2.57. No single window drives it.
- **Concentration (primary).** Top window share of the loss 6 % · top pair 6 % (QUSDT 4 fills).

**DCR before → after at one size** (0.61 × equity per fill), skipping `r15 ≤ −15`:

| period | days | fills | DCR before | DCR after | Δ | Δ after 50 % haircut |
|---|---|---|---|---|---|---|
| ALL | 1,236 | 1,547 → 1,472 | +0.595 % | +0.520 % | −0.075 pp | −0.038 pp |
| DESIGN | 731 | 559 → 554 | +0.411 % | +0.387 % | −0.024 pp | −0.012 pp |
| HOLDOUT | 505 | 988 → 918 | +0.862 % | +0.712 % | −0.150 pp | −0.075 pp |

## 7. Real fills (master read; N = 6, all 11-19 → 11-21, direction only)
| opened (UTC) | pair | P&L % | exit | r15 % (stamp) | r30 | r60 | below 60-min high | below 24 h high | volx15 | ≤ −15 block? |
|---|---|---|---|---|---|---|---|---|---|---|
| 11-19 04:15 | SENT | −4.41 | stop | −1.4 (−1.4) | −0.1 | −0.2 | −10.7 | −10.7 | 90× | no |
| 11-19 09:15 | NAORIS | −5.97 | stop | −0.8 (−0.7) | +3.3 | +13.1 | −5.8 | −5.8 | 353× | no |
| 11-19 14:31 | EVAA | −4.38 | stop | −2.0 (−2.0) | +1.3 | +6.5 | −5.4 | −5.4 | 13× | no |
| 11-20 09:00 | CYBER | +9.95 | TP | −1.1 (−1.1) | +3.7 | +31.8 | −2.6 | −2.6 | 173× | no |
| 11-20 12:30 | KERNEL | −6.11 | stop | −1.7 (−1.7) | +5.1 | +2.7 | −6.2 | −6.2 | 484× | no |
| 11-21 09:30 | **BLESS** | −6.03 | stop | **−22.0** (−22.0) | −27.9 | −20.3 | −29.5 | −29.5 | 50× | **YES** |

- BLESS is the only real after-crash fill. The other four losers had **no** crash before entry (signal candle −0.7 … −2.0 %).
- So the batch's losses are not an after-crash pattern either. The stamped `entry_dist_bar_ret_pct` matches the kline read, so the variable is already recorded on every live fill.
- BLESS's volume ratio (50×) is in the *bottom* tercile of the backtest (the edge is 157×). On this proxy it was not a cascade-like signal candle. The $352k liquidations are visible only in Route B.

## 8. Event-study chart (`reports/DIST_AFTERCRASH_2026-11-21.png`)
Median price vs entry (5m closes), with IQR bands. Winners and losers are shown separately, for the after-crash cohort (≤ −15 %, N = 75) and the rest (N = 1,472).
- **After-crash coins pumped first.** The median after-crash coin was **+28 % above entry 20 min before**, then lost it all in the signal candle. The rest rose only +3 %.
  - So "already crashed" in this sleeve usually means "a vertical pump began unwinding". It is the start of the distribution, not its end.
  - The median after-crash short keeps falling: −1.5 % at +15 min, −3.9 % at +2 h. The rest: −0.4 % and −1.8 %.
- **Winners and losers look the same before entry.** Inside the after-crash cohort their median pre-entry paths overlap (same pump, same crash). They split only after entry: losers bounce +4–5 % and sit there; winners keep falling to about −12 %.
- So BLESS's reflex bounce cannot be told apart at entry from the 53 % of identical-looking setups that kept falling.

## 9. Verdict and recommendation
**REFUTED.** The pre-registered prediction (deeper fall → worse short) is contradicted in DESIGN, in HOLDOUT, and in both BTC regimes.
- Every after-crash cut earns 1.5–4× the sleeve average.
- Blocking BLESS-type entries would have removed some of the sleeve's best trades.
- It would have cut HOLDOUT DCR by ~0.15 pp/day (−0.075 after the haircut).
- Per the decision rule: no filter candidate and no observe line. The already-refuted quintile reads (DIST_WL, DIST_LOSERS2, DL 602) stand, and now cover the extreme tail too.

Not proposed:
- The positive side (after-crash = better) is **not** proposed as a multiplier.
- It barely clears the null at one cut only (≤ −10 %), the dose is not clean (−20…−15 is weaker than its neighbours), and DESIGN holds only 23 such fills.
- Per the rules it is at most a watch idea, and it would need the Pattern-W multiplier gates on real fills.

Caveats:
- The backtest carries the read: there is 1 real after-crash fill.
- Liquidations are proxied by volume. Real liquidation data (Route B) starts 2026-11-19, and a 'liquidation cascade done' signal can only be tested once it has piled up (the weekly Route B scan already queued).
- 1,547 one-per-pair fills; slip modelled at 0.10 %. After-crash coins may be thinner, so real slippage could shave part of the edge. It would not flip +2.3 % to negative.

**Plain English for the operator.** It made sense on paper, but the data says the opposite.
- When a DIST short is taken right after a coin has already crashed 15–20 % in 15 minutes, it has historically won **more** often (53–62 % vs 42 %) and earned about **2.5× more per trade** than a normal DIST short.
- Those coins had usually just gone up vertically (+28 % in the previous half hour), so the crash was the start of the unwind, not the end.
- BLESS was one of the ~45 % that bounce. Nothing visible at entry separated it from the ones that kept falling.
- **Recommendation:** change nothing. Don't add a filter for this; it would remove profitable trades. The open lever for DIST losses is still the size per stop already on the pre-live list.
