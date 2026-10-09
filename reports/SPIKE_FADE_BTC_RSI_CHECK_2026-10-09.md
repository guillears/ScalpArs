# SPIKE_FADE BTC-RSI band check: is the 45–50 band costing money? (2026-10-09)

**Question.** `spike_fade_max_btc_rsi` went 45 → 50 on 2026-09-24 (DECISION_LOG 112). The recall trace (2026-10-08) saw the yr5 replay's fades in the 45–50 band lose about −0.26 %/trade between Aug-5 and Sep-24. Is that band costing money?

**Answer.** No, not on any evidence that can decide it.
- **Live fades since the raise** (the only true out-of-sample test): 6 fills in the band, 5 won, +0.11 %/trade, +$234 on the $3k ruler.
- **Replay, full year:** the band is no worse than the rest (−0.00 % vs −0.01 %).
- **Replay, Aug-5 → Sep-24:** the −0.26 % was a bad stretch for every fade, not for this band. In that same replay window, fades below 45 lost just as much (−0.17 % vs −0.17 %).
- **Rules check:** the band fails every leg of the locked filter bar. Its own pre-committed revert gate is at 6 of 10 and currently passing.

**Recommendation: keep 50.** Let the existing DECISION_LOG 112 gate decide at N = 10. Do not move it.

Validation: `scripts/validate_against_master.py`: ALL CHECKS PASS (run first).
Script: `scripts/study_fade_brsi_check.py`.
Tables: `reports/SPIKE_FADE_BTC_RSI_CHECK_2026-10-09.csv` (long form) and `reports/SPIKE_FADE_BTC_RSI_CHECK_2026-10-09_forward_fills.csv`.

---

## 0 · Exact definitions (from the code and the log)

**Engine gate** (`services/trading_engine.py` 9340 scanner, 17266 top-50 hook):
- A fade is blocked iff `_current_btc_rsi > spike_fade_max_btc_rsi`. The test is **strict**, so a reading of exactly 50 passes.
- `_current_btc_rsi` is the BTC **5 m RSI including the forming candle** (`get_ohlcv('BTC/USDT:USDT','5m',100)`). It is not the closed-bar value (`_current_btc_rsi_closed` is a separate global).
- If the reading is missing, the gate fails open.
- The stamp `entry_btc_rsi` is the same reading. Validator F2: r = 0.968 at shift 0 against a rebuilt 5 m RSI.
- So the 45 rule admits fades with bRSI ≤ 45, the 50 rule admits bRSI ≤ 50, and **the band the raise added is (45, 50]**. All cohorts below use that band.

**Aug-5 tighten (50 → 45)**, DECISION_LOG 2026-08-05 addendum 5, commit 249ce1f (16:16 UTC):
- It was an operator call over the quant's "watch" recommendation.
- Its evidence was one batch: 8 fills, 38 % WR, −0.62 %, −$472, with no cross-pool support. EVAA alone was 54 % of the loss.
- Pre-committed revert: "at N≥6 blocked-in-[45,50), price-replay under the fade exit stack — WR≥55 % ∨ Σ>0 → ceiling back to 50".

**Sep-24 raise (45 → 50)**, DECISION_LOG 112, commit c0d7fed (2026-09-24 21:36 UTC). That revert gate fired on both legs.
- **🔒 Its own tight re-revert:** "fresh fades opened at BTC RSI [45,50) (`entry_btc_rsi` stamp) at N≥10 — WR<55 % ∨ Σ<0 → back to 45; same-minute fires count once."
- Read below as the engine's (45, 50]. No forward fill sits at exactly 50, so the two readings agree.

**Ceiling eras used for the live fills:**

| Era | Ceiling |
|---|---|
| before Jul-31 00:14 UTC | none |
| Jul-31 → Aug-5 16:16 | 50 |
| Aug-5 → Sep-24 21:36 | 45 |
| since Sep-24 21:36 | 50 |

**Sleeve breakeven WR** (expectancy bar, current-stack kept master fades, N = 68):
- Average win +0.590 %, average loss −1.343 %, so **breakeven WR = 69.5 %**.
- The replay's own breakeven is 76.8 % (average win +0.438, average loss −1.453).

**$ ruler.** Today's fade ticket on a $3k book = min($29,250, 0.5 % × 24 h volume, $500k), the H1-review rule. % = P&L % net of fees, which is leverage-invariant.

**Windows.** Fills chained while consecutive opens are ≤ 60 min apart (pooled over seeds). Bootstraps resample **days**; the window-clustered P is also in the CSV and agrees to within 0.02 everywhere.

---

## (a) yr5 replay: 1,183 fades, 3 seeds, warm-up trimmed, code 181131e (ceiling 50)

The highest admitted bRSI is 49.97, so the replay obeyed the ceiling.

| Segment | Band | N/seed | WR | avg % | $/seed ($3k) | days | day-CI 95 % | P(mean<0) |
|---|---|---|---|---|---|---|---|---|
| Full year | ≤45 | 290.7 | 76.6 | −0.012 | −809 | 173 | −0.09…+0.07 | 0.61 |
| Full year | **(45,50]** | 103.7 | 75.6 | **−0.001** | **+226** | 111 | −0.13…+0.12 | 0.51 |
| H1 Jan-04→May-20 | ≤45 | 118.3 | 71.3 | −0.108 | −3,095 | 82 | −0.23…+0.01 | 0.97 |
| H1 | (45,50] | 37.7 | 74.3 | −0.040 | −204 | 50 | −0.25…+0.14 | 0.65 |
| H2 May-20→Oct-4 | ≤45 | 172.3 | 80.3 | +0.053 | +2,286 | 91 | −0.06…+0.15 | 0.17 |
| H2 | (45,50] | 66.0 | 76.3 | +0.021 | +430 | 61 | −0.15…+0.19 | 0.41 |
| 45-rule era Aug-5→Sep-24 | ≤45 | 36.3 | 68.8 | **−0.166** | −1,285 | 31 | −0.43…+0.11 | 0.87 |
| 45-rule era | (45,50] | 13.0 | 66.7 | **−0.172** | −350 | 18 | −0.51…+0.26 | 0.80 |
| Post-raise Sep-24→Oct-4 | ≤45 | 2.3 | 100 | +0.463 | +132 | 2 | — | — |
| Post-raise | (45,50] | 3.3 | 50.0 | −0.607 | −330 | **3** | −1.67…+0.44 | 0.84 |

**Band minus rest** (day-paired bootstrap):

| Segment | Δ avg % | 95 % CI |
|---|---|---|
| Full year | +0.011 | −0.14…+0.16 |
| H1 | +0.068 | |
| H2 | −0.033 | −0.24…+0.17 |
| Aug-5→Sep-24 | −0.006 | −0.41…+0.48 |
| Post-raise | −1.07 | 3 days only; SOLV is 75 % of the loss |

**Dose-response** (full year):

| bRSI | avg % |
|---|---|
| ≤40 | −0.029 |
| 40–45 | +0.007 |
| 45–47.5 | +0.031 |
| 47.5–50 | −0.038 |

- This is flat and non-monotonic: a confound or noise, not a gradient.
- The top sliver (47.5, 50] is the weakest cell in H2 (−0.054), in the 45-rule era (−0.384, 5.3/seed, 10 days) and post-raise (−0.705, 3 days).
- It is positive in H1 (−0.012, about flat).
- No sub-band clears any confidence bar. The best is P(mean<0) = 0.65.

**Per seed and per month.**
- Per seed, the band averages +0.025 / −0.015 / −0.015, against −0.027…+0.004 for the rest.
- By month, the band beats the rest in 6 of 10 months. The two bands move together (both negative in Mar, Apr and Sep; both positive in Jan and Jun). That is the uniform-degradation signature of a regime variable, not of this band.

**What the trace's "−0.26 %" was.**
- It counted only the trace's `ERA_CONFIG_BRSI` class: the 22 replay fills in live-up stretches that live refused because live then ran 45.
- On the whole Aug-5→Sep-24 replay window, the band and the rest lost the same.
- The trace window (Jul-28→Oct-4) also takes in the 3 bad post-raise days, which pulls the band to −0.274 there.

**Stale 1 m candle bias by band** (trace window, trace signal file). The replay reads fades about 0.08 %/trade too low overall.

| Band | stale-view share | stale artefacts | artefact avg | replay avg |
|---|---|---|---|---|
| ≤45 | 72 % | 30 % | +0.19 (winners) | −0.09 |
| (45,50] | 41 % | 16 % | −0.35 | −0.27 |

- The 20 live fades the replay missed through stale data (classes A + B) are **all in ≤45**. That is by construction, because live ran 45 in that window, so the bias cannot be measured for the band from live misses.
- **The band is less exposed to the stale-view error.** The +0.08 correction belongs mostly to the ≤45 band. Calibrating would lift ≤45 slightly more than the band.
- That gap does not reach the full year, where the bands are equal before any correction.
- Band assignment is also slightly noisy. On 78 matched signals the replay bRSI reads +0.15 above live (median |Δ| 0.35), and 6 of 78 cross the 45 line.

**Replay verdict:** it does **not** refute the 50 rule. Under the cross-period rule a replay may only refute, and it finds no band effect at any granularity.

---

## (b) Live master fades, STACK 2026-10-08c

**Kept fills (today's-stack pct):**

| Band | N | WR | avg % | $ ($3k ruler) | days | day-CI | P(<0) | Ceiling when opened |
|---|---|---|---|---|---|---|---|---|
| ≤45 | 61 | 88.5 | +0.381 | +5,284 | 36 | +0.19…+0.60 | 0.00 | mixed |
| **(45,50]** | **7** | 85.7 | +0.195 | +417 | 5 | −0.23…+0.56 | 0.21 | 1 pre-ceiling (MET +0.70) · 6 under the 50 rule (all post-raise) |
| 45–47.5 | 3 | 66.7 | −0.178 | −4 | 3 | | | ORDI W, KOMA L, MET W |
| 47.5–50 | 4 | 100 | +0.476 | +421 | 3 | | | |

- **45 era:** 38 kept fills, all ≤45. No kept fill landed in the band, so stamp and gate agree.
- **The Aug-5 losers are already gone.** The 9 band fills of the first 50 era (Jul-28→Aug-5) were 33 % WR, −0.63 % as traded, −$1,275. They are **all stack-blocked today**:
  - 7 by FADE_FRESHBREAK (PIPPIN, EVAA, KSM, SNX, ERA, SXT, STORJ);
  - 2 by FADE_BD13 (FRAX, ICNT).
- So the evidence that drove the Aug-5 tighten is absorbed by later gates (consistent with DECISION_LOG 112). Reverting to 45 would remove none of today's losers.

---

## (c) Forward cohort of the 50 rule: live fills opened after 2026-09-24 21:36 UTC

**Sources:** B13–B18 plus every Downloads export to 2026-10-09 01:46, de-duplicated on (opened_at, pair, direction). That gives 16 fades, all present in master. The last fade opened 2026-10-08 02:26.

| Band | N | WR | avg % | Σ % | $ ($3k ruler) | master stack $ | days / windows |
|---|---|---|---|---|---|---|---|
| ≤45 | 10 | 100 | +0.375 | +3.75 | +778 | +632 | 7 / 9 |
| **(45,50]** | **6** | **83.3** | **+0.112** | **+0.67** | **+234** | +105 | 4 / 6 |

**The 6 band fills:**

| Pair | Date | bRSI | Result |
|---|---|---|---|
| 2Z | Sep-26 | 48.0 | +0.52 |
| BEAMX | Sep-26 | 48.3 | +0.32 |
| INIT | Oct-3 | 47.9 | +0.57 |
| ORDI | Oct-6 | 45.2 | +0.27 |
| KOMA | Oct-7 | 45.1 | **−1.50 stop** |
| BRETT | Oct-7 | 48.6 | +0.50 |

- As-traded and today's-stack pct are identical here (KOMA −1.497 vs −1.500).
- The same post-raise days the replay priced at −0.61 % (3 days, SOLV-driven) were **positive live**. The replay and live disagree on these exact days, and live decides.
- Fine bands: 45–47.5 gives 2 fills at 50 % (KOMA L, ORDI W), −0.62 %; 47.5–50 gives 4 fills at 100 %, +0.48 %. This is the opposite of the replay's weak top sliver.

---

## (d) The Sep-24 evidence (context, in-sample; DECISION_LOG 112, no saved rows)

- On Sep-24 logs, 41 spike triggers were priced on ticks under the live fade stack.
- The bRSI 45–50 group was 9 fills, 89 % WR, Σ +6.42 %, over 7 windows and 3 days. Excluding DRIFT it was 8 fills, 88 %, +1.67.
- 6 of the 9 also clear BD13 + FRESHBREAK: 6 fills, 100 %, +6.73.
- The same day, the bRSI > 50 group was 17 fills, 65 % WR, Σ −4.69, with 6 full stops. That is the reason the ceiling stops at 50 and not higher.

---

## Verdict against the rules

**(i) Locked expectancy filter bar on the (45, 50] band**

| Leg | Replay full year | Replay Aug-5→Sep-24 | Live master kept | Live forward |
|---|---|---|---|---|
| ① WR < breakeven 69.5 % (replay's own 76.8 %) | 75.6 ✗ (below its own BE, but leg ② fails) | 66.7 ✓ | 85.7 ✗ | 83.3 ✗ |
| ② P(mean<0) ≥ 0.95, day-clustered | 0.51 ✗ | 0.80 ✗ | 0.21 ✗ | 0.27 ✗ |
| ③ ≥ 8 windows, no window/pair ≥ 50 % of loss | ✓ | ✓ | 7 windows ✗ | 6 ✗ (KOMA = 100 % of the loss) |
| ④ N ≥ 15 | ✓ | ✓ | ✗ | ✗ |

**Result: FAILS on every cohort.** It is not even a frozen observe candidate, because the cohort that comes nearest (replay, 45-rule era) has the rest of the fades losing equally (Δ −0.006). That is regime, not band.

**(ii) The 50 change's own pre-committed gate** (DECISION_LOG 112): fresh (45, 50] fades, same-minute fires counted once, N ≥ 10 → WR < 55 % ∨ Σ < 0 → back to 45.
- **Status: 6 / 10** (no same-minute duplicates).
- **WR 83 %** (≥ 55 ✓), **Σ +0.67 %** (> 0 ✓). Currently **passing, not yet decidable.**
- At about 6 band fades per 2 weeks of live time, it reads in roughly 1–2 weeks.
- ⚠ **Fragility, flagged and not changed:** Σ has only +0.67 % of headroom. **One more −1.5 % stop in the next 4 fires fires the gate** (Σ < 0) even at 80 % WR. That is what was pre-committed; per the rules it stands as written and is not re-fitted now.

**(iii) Cross-period = refute-only.** The replay does not refute the 50 rule: the band equals the rest over the full year, both halves and the 45-rule era. The forward live fills, the deciding evidence, are positive.

**Recommendation: KEEP 50.**
- No config change, no new observe line.
- **Evidence level:** forward live N = 6 (thin, positive); replay N = 311 fills / 111 days (null band effect).
- **Gate:** the existing DECISION_LOG 112 gate stays exactly as written. At each batch review, also print the 45–47.5 / 47.5–50 split (read-only tally; the replay's weak cell is the top sliver, live's weak cell is the bottom one). This is no new gate.
- **No haircut needed:** nothing is shipped. If a revert were made today, the replay's Δ would be ≈ 0 and live would lose +$234 (on the $3k ruler) of forward fills.

## Blind spots (what this could not test)
- **No blocked (> 50) or band signals were priced on ticks for Sep-25 → Oct-8.** Only live fills, so the gate side above 50 is not re-checked.
- **No real-tick re-pricing of the replay band.** The stale-view bias is inferred from the trace's signal file (trace window only).
- **The Sep-24 nine are quoted, not re-derived.** Their rows were not saved.
- **The forward cohort is 4 days / 6 windows.** It is one regime stretch. Market-wide variable, so judge it in day units: effectively about 4 observations.
- **Windows use a 60 min chaining rule.** The day-clustered and window-clustered P agree within 0.02, so the choice does not change any conclusion.
