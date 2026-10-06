# 🪢 FRENZY / WIDE split-position runner (EMA200 / EMA100 / 1h EMA50), 2026-10-06

Research only. No bot file, config, test or script under `scripts/` was touched. Nothing was committed. The scripts live in the scratchpad (`split/price.py`, `analyze.py`, `extra.py`, `slots.py`, `bk2.py`). The results are **unreviewed**: no caveman or deep review has run on them yet.

## Engine parity (read first)

- **Cohort.** `reports/FRENZY_ENGINE_COHORT_2026-10-05.csv` holds the engine's own `frenzy_walk` fresh_on bars. The FRENZY_REVALIDATION live check found all 8 live FRENZY / WIDE fills from Oct 3–5 on exactly their signal bar. The tradeable flags (`live_elig`, `gvol_live_U2`) come from `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv`.
- **Primary cohort "B, tradeable":** priced ∧ live_elig ∧ live-like gvol < 1. That gives **FRENZY 205 fills on 133 days and WIDE 570 on 227 days**.
- **Secondary cohort "A, published":** gate_ok ∧ priced. That gives FRENZY 249 and WIDE 740.
- **Lock leg = live LOCK2.** It is re-priced here with the cohort's own 12 s pricer and matches the file's LOCK2 to 1.8e-15, its exit time exactly, and its entry price to 9e-13.
- **The runner leg is new logic, not engine code.** No engine path exists for it, so its parity is "by specification" only (spec below).

**Runner spec (pre-declared before any result was read)**
- **The split.** The position is split at entry. A share (1 − f) exits on the live lock. The runner share f ∈ {20, 30, 50 %} exits only on a closed-bar close below its line.
- **Line (3 options):**
  - **E200:** 5m EMA200.
  - **E100:** 5m EMA100.
  - **H50:** a 1h close below the 1h EMA50.
  - All EMAs are built on the full `k5m_full` history from Dec 30, which gives at least 3,217 5m bars and 268 1h bars before every entry. They use adjust=False and closed bars only, so there is no look-ahead.
- **Runner stop (3 options):**
  - **NONE:** no stop at all, only liquidation at the fill's leverage (FRENZY 6× normal / 10× strong, WIDE 4×).
  - **BE:** −3 until the lock arms, then breakeven (0 net).
  - **P2:** −3 until the lock arms, then +2 net.
- **Hold cap:** 24 / 48 / **72 (primary)** / 120 h.
- **Fees:** taker 0.09 round trip, plus a −0.10 slippage on every exit.
- **Pricing:**
  - **Within 12 h:** ticks, or the 1m fallback. Stops fill at the crossing print, and a line exit fills at the first print at or after the bar close.
  - **After 12 h:** 5m bars. A stop gapped at the open fills at the open. A line exit fills at the 5m close.
- **Fractions only scale Δ.** Blended Δ = f × (runner − lock), so the selection family is really **9 runner rules** (× 4 caps).

---

## 1. RLC 10-05: the trade behind the question (out of cohort; the cohort ends Sep 27)

Entry 0.4648 at 12:00:12, WIDE, 4×. The **12 s pricer's lock gives +5.30** (out at 12:08). The live fill closed at about +3 %; the gap comes from live exit timing.

| runner rule | runner % | exit | blended f = 20 / 30 / 50 % |
|---|---|---|---|
| **E200** (any stop) | **+118.9 (still OPEN, marked 10-06 12:05; peak +131.8)** | never closed below the 5m EMA200 | +28.0 / +39.4 / +62.1 |
| E100 (any stop) | +48.5 | 10-06 02:55, a 5m close below EMA100 (peak +69.6) | +13.9 / +18.3 / +26.9 |
| H50 (any stop) | +118.9 (OPEN, marked) | never closed below the 1h EMA50 | +28.0 / +39.4 / +62.1 |

- RLC is exactly what an EMA200 runner is built to catch, and it would be holding now.
- **It is one trade.** It is censored: the result is a mark, not an exit.

---

## 2. FRENZY_LONG (cohort B, 205 fills, Jan 10 → Sep 27)

Live lock alone: **+0.387 % per fill**, WR 54 %. Book **$3,000 → $9,741, max DD −43 %**.

All rows below are at the primary cap of 72 h with a **30 % runner**. Columns:
- "Δ" is per fill at 1×, blended vs the lock, with its day-block 95 % CI.
- "−top5 / −top10" is the Δ after dropping the 5 / 10 largest-Δ fills.
- The book is from $3k at live sizing (strong 0.5 / normal 0.32), with the runner **holding its slot**.

| runner | mean / median / WR | Δ [95 % CI] | −top5 / −top10 | top pair · top day share of Δ | months Δ > 0 | LOMO Δ | halves Jan–Apr / May–Sep | runner hold avg / max (h) | book end / DD |
|---|---|---|---|---|---|---|---|---|---|
| **E200 · NONE** | +0.343 / −0.42 / 47 % | −0.044 [−0.60, +0.58] | −0.49 / −0.79 | – (Δ < 0) | 4 / 9 | −0.15…+0.10 | −0.33 / +0.26 | 8.5 / 32 | $2,864 / **−69 %** |
| **E200 · BE** | +0.538 / +1.28 / 54 % | +0.150 [−0.21, +0.55] | −0.23 / −0.38 | ENSO 103 % · 01-24 87 % | 5 / 9 | +0.04…+0.21 | +0.19 / +0.11 | 2.9 / 32 | $12,241 / −44 % |
| **E200 · P2** | +0.464 / +1.87 / 54 % | +0.077 [−0.14, +0.40] | −0.17 / −0.21 | ENSO 202 % · 174 % | 3 / 9 | −0.08…+0.14 | +0.17 / −0.03 | 1.7 / 32 | $11,556 / −45 % |
| E100 · NONE | +0.791 / +0.17 / 51 % | +0.403 [−0.17, +1.09] | −0.12 / −0.40 | 44 % · 36 % | 5 / 9 | +0.06…+0.76 | +0.06 / +0.76 | 4.5 / 30 | $23,468 / −43 % |
| **E100 · BE** (best t) | +0.739 / +1.27 / 54 % | **+0.351 [−0.06, +0.81]** | **−0.05 / −0.24** | **ENSO 52 % · 01-24 43 %** | **7 / 9** | **+0.23…+0.41** | **+0.43 / +0.27** | 2.3 / 30 | $21,190 / −44 % |
| E100 · P2 | +0.571 / +1.84 / 54 % | +0.183 [−0.07, +0.55] | −0.10 / −0.17 | 100 % · 84 % | 5 / 9 | +0.01…+0.23 | +0.32 / +0.04 | 1.5 / 30 | $15,921 / −44 % |
| H50 · NONE | +0.209 / −1.46 / 39 % | −0.179 [−1.17, +0.92] | −1.09 / −1.54 | – | 3 / 9 | −0.60…+0.06 | −0.18 / −0.18 | 23.8 / 72 | $1,332 / −79 % (78 liquidations) |
| H50 · BE | +0.470 / +1.28 / 54 % | +0.082 [−0.32, +0.61] | −0.39 / −0.50 | PORTAL 192 % | 4 / 9 | −0.07…+0.24 | −0.22 / +0.40 | 4.5 / 72 | $6,286 / −48 % |
| H50 · P2 | +0.340 / +1.88 / 54 % | −0.047 [−0.21, +0.17] | −0.21 / −0.22 | – | 2 / 9 | −0.11…+0.01 | −0.05 / −0.05 | 2.3 / 56 | $8,502 / −48 % |

**Other runner sizes**
- **20 % / 50 % runner:** Δ scales linearly. For E100·BE it is +0.234 [−0.03, +0.54] at 20 % and +0.586 [−0.07, +1.38] at 50 %; for E200·BE, +0.100 / +0.250.
- **Books at 20 % / 50 % runner:**
  - E100·BE: $16.7k / $30.4k, DD −43 / −45 %.
  - E200·BE: $11.4k / $12.7k.
  - E200·NONE: $3.9k / $1.3k, DD −54 / −87 %.

**Remove the rides from the book** (E100·BE at 30 %; the top-k Δ fills are set back to their lock result):

| top rides removed | 0 | 1 (ENSO 01-24) | 5 | 10 |
|---|---|---|---|---|
| E100·BE | $21,190 | $14,171 | **$8,904** (below the lock's $9,741) | $4,965 (DD −60 %) |
| E200·BE | $12,241 | $8,874 | $4,210 | $3,490 |
| E200·P2 | $11,556 | $8,462 | $6,166 | $5,321 |

**Hold-cap sensitivity** (Δ at a 100 % runner): 24 h / 48 h / 72 h / 120 h

| rule | 24 h | 48 h | 72 h | 120 h |
|---|---|---|---|---|
| E100·BE | +1.29 | +1.17 | +1.17 | +1.17 |
| E200·BE | +0.68 | +0.50 | +0.50 | +0.50 |
| E200·NONE | +0.49 | −0.15 | −0.15 | −0.15 |

- No 5m runner lived beyond about 32 h, so caps of 72 h and above never bind.

**Per-month Δ** (100 % runner, cohort B):

| month | N | E100·BE | E200·BE | E200·P2 | E200·NONE |
|---|---|---|---|---|---|
| Jan | 25 | +4.10 | +3.08 | +3.92 | +2.36 |
| Feb | 13 | −0.76 | −1.43 | −0.67 | −4.23 |
| Mar | 45 | +0.65 | −0.16 | −0.44 | −1.91 |
| Apr | 22 | +1.31 | +0.63 | −0.40 | −1.59 |
| May | 30 | +0.02 | −0.66 | −0.20 | +1.75 |
| Jun | 12 | −1.94 | −1.83 | −0.83 | −4.30 |
| Jul | 19 | +2.94 | +3.30 | −0.31 | +3.19 |
| Aug | 21 | +1.40 | +0.11 | +0.34 | −1.79 |
| Sep | 18 | +1.50 | +0.79 | +0.36 | +3.47 |

**Cohort A** (published, 249 fills; lock +0.311, book $9,953 / −48 %), 30 % runner:

| rule | Δ [CI] | months > 0 | book |
|---|---|---|---|
| E200·NONE | −0.326 [−0.77, +0.17] | | $883 / −84 % |
| E200·BE | +0.053 [−0.24, +0.40] | | $8,332 |
| E200·P2 | +0.042 | | $9,728 |
| **E100·BE** | **+0.231 [−0.09, +0.62]** | 7 / 9 | $15,196 / −50 % |

- Same ordering as cohort B, all weaker.

### Two-sided: where the runner hurts (FRENZY B, 100 % runner)

| rule | fills where the runner < lock | mean hurt | sum hurt / sum helped (pts, at 30 %) | liquidations | worst single Δ |
|---|---|---|---|---|---|
| E200·NONE | 149 of 205 | −5.9 | −264 / +255 | 20 | GMT 05-23: lock +10.1 → runner −9.2; ACX / BTR / HIPPO liquidated at −9.6 |
| E200·BE | 93 | −3.3 | −92 / +123 | 0 | GMT 05-23 +10.1 → −0.1; ENJ 03-18 +8.3 → −0.1 (it peaked +27 first) |
| E200·P2 | 59 | −2.4 | −42 / +58 | 0 | GMT 05-23 +10.1 → +1.8 |
| E100·BE | 91 | −3.1 | −84 / +156 | 0 | GMT 05-23 −10.3; BERA 01-17 −8.4; PHA 05-26 −7.3 |

- **Typical failure:** the lock banks +6…+10 at peak − 2, and the runner then rides it back to its BE stop.
- **On rides that peaked ≥ +15**, the runner still gave back **24–28 pts from its peak** before the 5m close crossed the line (E100 / E200 respectively).
- **With no stop**, the runner loses −2.4 pts per fill on the trades the lock lost (E200·NONE). That is what kills its book.

---

## 3. FRENZY_WIDE (cohort B, 570 fills; lock −0.316 %/fill, book $3k → $446, DD −92 %)

30 % runner, 72 h, with and without RLC:

| rule | Δ [CI], no RLC | Δ [CI], + RLC | months > 0 | book no RLC / + RLC (lock $446 / $468) |
|---|---|---|---|---|
| E200·NONE | −0.328 [−0.99, +0.47] | −0.267 | 3 / 9 | $102 / $140 (−98 %) |
| E200·BE | −0.127 [−0.30, +0.11] | −0.067 | 3 / 9 | $210 / $287 |
| **E200·P2** | +0.054 [−0.09, +0.27] | **+0.114 [−0.08, +0.37]** | 2 / 9 → 3 / 10 | $530 / $727 |
| E100·NONE | −0.067 | −0.045 | | $321 / $376 |
| E100·BE | **−0.183 [−0.28, −0.07]** | **−0.160 [−0.27, −0.03]** | 2 / 9 | $172 / $201 |
| E100·P2 | −0.036 | −0.013 | | $359 / $421 |
| H50·NONE | −0.782 | −0.720 | | $178 / $244 (93 liquidations) |
| H50·BE | **−0.289 [−0.38, −0.19]** | **−0.229 [−0.36, −0.05]** | 0 / 9 | $94 / $129 |
| H50·P2 | −0.045 | +0.014 | | $328 / $450 |

- **The only positive cell is E200·P2.**
  - **TAIKO 07-01 alone is 159 % of its Δ.** TAIKO peaked +411 and the runner made +172.
  - With RLC, RLC becomes the second-largest contributor.
  - Months positive: 2 of 9. It is negative after dropping the top 5.
- The BE rules are significantly **worse** than the lock. Most WIDE fills that arm fall back below the lock's +2 floor.
- Cohort A agrees. The best cell is again E200·P2: +0.021 without RLC, +0.067 with RLC. E100·BE −0.169 and H50·BE −0.254 both have CIs below 0.
- **A runner cannot rescue a sleeve with a negative base edge.** Even E200·P2 leaves WIDE at −91 % DD.

---

## 4. Selection control (few variants, but it is still a pick)

**Pick-best test.** Day-level sign-flip null, joint across the 9 rules at 72 h, using the studentized max-t statistic:

| set | best rule (mean Δ, 100 % runner) | raw p | family-adjusted p |
|---|---|---|---|
| FRENZY B | E100·BE (+1.17, t 1.54) | 0.063 | **0.21** |
| FRENZY B, pre-declared primary E200 | E200·BE (+0.50) | 0.24 | 0.61 |
| FRENZY A | E100·BE (+0.77, t 1.24) | 0.13 | 0.39 |
| WIDE B (no RLC / + RLC) | E200·P2 (+0.18 / +0.38) | 0.34 / 0.17 | 0.76 / 0.51 |

- The max-mean statistic over all 36 rule × cap cells gives FRENZY p 0.33 and WIDE p 0.80.

**Out-of-sample.** Pick on one half, test on the other (100 % runner Δ):

| set | pick on Jan–Apr → test May–Sep | pick on May–Sep → test Jan–Apr |
|---|---|---|
| FRENZY B | E100·BE: in +1.44 → out **+0.89**, CI [−0.58, +2.66] | E100·NONE: in +2.55 → out +0.20, CI [−2.1, +3.0] |
| FRENZY A | E100·BE: in +1.12 → out +0.41, CI [−0.77, +1.77] | H50·BE: in +0.75 → out **−0.78** |
| WIDE B | E100·NONE: in +0.09 → out **−0.49** | E200·P2: in +0.27 → out +0.07 |

- E100·BE survives out-of-sample on sign in the forward direction. Every OOS CI spans 0.
- Haircut 30–50 %: E100·BE at a 30 % runner is +0.35 → **+0.18…+0.25 per fill**, with a CI that already touches 0 before the haircut.

---

## 5. Blind spot: slot occupancy

Live: FRENZY and WIDE have 2 slots each (`frenzy_max_slots`, `frenzy_wide_max_slots`). Every FRENZY / WIDE open also passes the bot-wide `max_open_positions = 4` check in `open_position` (non-manual rows count). A runner that remains open keeps both kinds of slot.

| rule (FRENZY only / FRENZY + WIDE) | own-sleeve signals lost to a held runner (of 205 / 769) | time with ≥ 1 runner-only leg open | live master fills (Jun 17 → Sep 27, 704) that would hit max_open 4 because of a runner |
|---|---|---|---|
| E100·BE | 5 / 20 | 5.0 % / 10.5 % | 0 / 7 |
| E200·BE | 8 / 26 | 6.4 % / 13.4 % | 0 / 7 |
| E200·P2 | 5 / 12 | 3.1 % / 5.8 % | 0 / 1 |
| E200·NONE | 27 / 226 | 20.7 % / 57.9 % | 8 / 59 |
| H50·NONE | 58 / 401 | 43 % / 86 % | 26 / 123 |

**How the books treat it**
- The books above already drop the own-sleeve signals a held runner blocks.
- The global-slot column is an estimate:
  - The master pool is the stacked live batches, with max concurrency 5.
  - Year fills from the other sleeves before Jun 17 are not modelled.
  - What those blocked momentum / flip trades would have earned is not counted.
- **For the BE / P2 rules on FRENZY alone, the slot cost is negligible.** For any no-stop or 1h rule it is large.

## Other blind spots

1. **The runner pricing after 12 h uses 5m OHLC.** Inside a 5m bar the stop is checked before the close, and peak and arming update only from the next bar. Live polls price every few seconds.
2. **Isolated margin on a partly closed position.** The model applies the same liquidation % to the runner leg. On Binance the margin left on the runner could differ.
3. **RLC is censored.** It is still open at the data end (10-06 12:05), so its E200 / H50 value is a mark.
4. **The FRENZY strong / normal split** comes from `strong.pkl`: di_spread > 0 ∧ adx_delta > 0, recomputed with `services.frenzy` functions.
5. **The tradeable universe** uses the Sep-16 and Oct-5 Alpha / listing snapshots (as in the overnight review).
6. **Engine parity is for the entries and the lock only.** No live code exists for a split exit.
7. **Not tested:**
   - partial-runner variants with a trailing ATR stop;
   - a runner taken only on strong fills;
   - a runner that arms only after a post-entry condition. That family was refuted in `FRENZY_REDO_EXITS` §6, so it was not re-run.

---

## Verdict

- **FRENZY_WIDE: refuted.**
  - No rule beats the lock with any confidence. BE and H50 are significantly worse.
  - The one positive cell (E200·P2) is one TAIKO day, plus RLC.
- **FRENZY_LONG, EMA200 (the operator's line): not supported.**
  - Δ is +0.08…+0.15 per fill at a 30 % runner, with the CI spanning 0.
  - It is negative after dropping the top 5. ENSO 01-24 is ≥ 100 % of the gain. Family p is 0.61.
  - The no-stop version loses money: book −69 % DD and 20 liquidations.
- **FRENZY_LONG, EMA100 with a breakeven stop: the best cell, but it is not a ship.**
  - It is +0.35 per fill at a 30 % runner, CI [−0.06, +0.81].
  - It is positive in 7 of 9 months and in both halves, and every leave-one-month-out run is positive. It holds out-of-sample in the forward direction (+0.89).
  - **It fails:**
    - the CI includes 0;
    - Δ is negative without the top 5 (−0.05) and top 10 (−0.24);
    - the book falls below the lock once the top 5 rides are removed;
    - ENSO carries 52 % of the gain;
    - the selection-adjusted p is 0.21.
  - It was also one of 9 rules, and not the pre-declared primary line.

**Recommendation: observe-only scout line (FRENZY_LONG only), no ship.**

The line costs no capital: it is a counterfactual per real FRENZY_LONG fill, computed from public 5m klines.

**Pre-registered, frozen:** `FRENZY_RUNNER_E100BE` = 30 % runner. Exit on the first closed 5m close below the 5m EMA100. Runner stop: −3 until the lock arms, then breakeven (0 net). 72 h cap.
- **Companion shadows**, tracked for context only, never promoted from this read: `E200BE` and `E200P2` (the operator's EMA200 line).
- **Promotion bar.** All five must hold:
  - ≥ 40 forward FRENZY_LONG fills on ≥ 20 distinct days;
  - forward Δ (blended) > 0 with the day-block 95 % CI low > 0;
  - still > 0 after dropping the top 3 forward fills;
  - no pair > 50 % of the forward Δ;
  - the pooled year + forward Δ > 0 without its top 10.
- **Retire** at 40 fills if forward Δ ≤ 0. Never re-fit the line, stop or fraction on the data that failed it.

WIDE gets no runner line.
