# STATE_EXIT_AUDIT 2026-11-21 — independent adversarial audit of `STATE_EXIT_2026-11-21`

This is an audit only. No engine, config or template file was changed and nothing was committed. I rebuilt the study's numbers from its own
data (`reports/STATE_EXIT_2026-11-21_data/`) and from the raw 1m klines, without trusting its tables. Helper scripts:
`scripts/STATE_EXIT_AUDIT_2026-11-21_{cohort,nullaudit,liversi}.py`. The power simulation and the give-back-zone rules were run inline;
their numbers are below.

## Verdict

| sleeve | study verdict | audit verdict | why |
|---|---|---|---|
| FRENZY_WILLY_HT | no state exit | **HOLDS, but the report gives the wrong reason** | The signs do not fail because they come "too late". RSI_ROLL fires around 0 % P&L and still saves about 3.6 pts per give-back, because give-backs ride all the way to the −4.1 stop. They fail on **precision**. On the 267-fill backtest HOLDOUT, every turn sign is right on 40–50 % of the fills it fires on, and breaking even needs about 55 % when firing near 0 %. The master 4 : 1 RSI asymmetry rests on 5 firings, and 3 of the 4 master firings are on entries today's DL 594 dip gate would block. |
| FLATBASE_SHORT | no robust state exit (FB_VWAP·G3 = formal candidate, don't ship) | **HOLDS, and FB_VWAP is weaker than reported** | LIGHT 10-24 carries about 70 % of FB_VWAP's master Δ, and it is an entry DL 601 would block today (entry funding 0.026 %). On today's stack FB_VWAP has 1 master firing. The backtest refutes it on its own (HOLDOUT −0.11, 39 firings). |
| EXHAUSTION_SHORT | no state exit | **HOLDS (inconclusive by construction)** | There are 0 master fills, so no candidate was possible. On the backtest, RSI_ROLL·G3 is +0.05 on HOLDOUT (P 0.80, 50 firings) but −0.16 on DESIGN (53 firings): it flips sign, so it is noise. |

**Bottom line: the null holds for all three sleeves. None is overturned or materially weakened.** Two parts of the report's argument are
flawed and should be corrected before anyone relies on them:
- **S1:** the "beats random timing" null is structurally biased.
- **S2:** the "fires after the turn" explanation is mostly wrong. The real constraint is precision (selection), not timing.

Neither flaw hides an edge. I re-ran the operator's sign at faster timings (live 5m bar, 1m RSI), and on the holdout it does as badly or worse.

---

## Issues found

| # | severity | issue | effect on the verdict |
|---|---|---|---|
| S1 | **HIGH (method)** | Thesis-family null (circular shift within each fill's gated minutes, family-wise max of unstandardised mean Δ) is biased and does not test what the report says | none on the HT / EXH verdicts (they fail the P ≥ 0.80 bar anyway); it overstates the FB case against FB_VWAP and the report's "nothing beats random timing" sentence |
| S2 | **HIGH (interpretation)** | "Signs fire after the turn → too late" is mostly not the mechanism; the mechanism is precision vs the save / chop payoff asymmetry | the conclusion stands, but the reason (and what to research next) changes |
| M1 | MEDIUM | Master HT cohort includes 6 / 49 entries that today's DL 594 dip ≥ 1.4 % gate would block. 3 of the 4 master RSI_ROLL firings are among them (AIN 11-05, KGEN 11-11, ONE 11-12). Master FB includes 5 / 36 DL 601-blocked entries, including LIGHT, the main FB_VWAP firing | it weakens the two positive-looking master reads (RSI_ROLL, FB_VWAP); the report's blind-spot list does not mention it |
| M2 | MEDIUM | Power: the HT master (49 fills, 9 give-backs, 19 days) can detect only strong rules. The family-wise null bar (+0.355 %/fill) needs a near-oracle rule | a moderate real edge (about +0.1 %/fill) cannot show on master. The backtest HOLDOUT is where the power lives, and it excludes effects ≳ +0.25 %/fill |
| M3 | MEDIUM (coverage) | Layer 2 (broad screen) is gated G1 (P&L now ≥ +1 %). The give-back zone, where most turn signs fire (median P&L at fire ≈ 0 %), was never broad-screened. This is not in the blind-spot list | I closed it here: simple give-back-zone rules are ≤ 0 on every large holdout (table C) |
| L1 | LOW | Parity flag `par_ok = k ≥ "2026-10-28"` is a date string, not the DL 408 deploy time. That flag explains the XAN outlier (below) | cosmetic: the re-price is correct |
| L2 | LOW | `rsi_max_in` and `lh_run_5m` can include the last pre-entry 5m bar (e.g. ENSO k = 1 reads the 02:25 bar's RSI 75.7) | minor: it can arm RSI_ROLL or LH2 on pre-entry structure |
| L3 | LOW | Fills walk from the minute after the fill, so the fill minute's remaining seconds are skipped (documented) | negligible |
| L4 | LOW | Report wording: "ENSO TP needs +5.07". Under the study's own maker model the TP needs a high of +5.12 % gross (5.07 + 0.05 % trade-through). ENSO's high was +5.046 % | none |
| ✓ | — | No look-ahead and no extra lag. Bar-close alignment, the exit walker, fees, slip, funding and the gates are all correct (section 1–2) | — |

---

## 1 · Look-ahead and lag (checked both ways)

- **Alignment** (`build.py:56-73`). `bars()` keeps only complete clock-aligned bars, and `avail` is the bar's last minute. `to_min` gives
  minute k the latest bar with `avail ≤ k`, and the decision is at the close of minute k. So a 5m bar enters the decision at the exact
  second it closes: no leak, and no extra lag.
- **Hand check.** I resampled ENSO 1m klines to 5m myself and ran Wilder RSI14. The 02:40 bar gives RSI 80.05 and the 02:50 bar gives
  69.80, both identical to `dec_HT`. The 02:50 bar first appears at k = 23 (minute 02:54, which closes 02:55:00). This matches the
  study's first RSI_ROLL fire at "+23 m @ +1.01".
- **Other inputs.** Metrics use `create_time ≤ bar open`, one minute conservative. Funding uses the last settlement ≤ t_k. 1h / 4h bars
  are complete bars. CLIMAX's extreme is compared with the running 1m extreme up to k. None of these leak.
- **Lag in the other direction.** Closed-5m RSI updates only every 5 min, while the operator watches the forming candle, so I tested
  faster variants (table A). All of them are no better on the holdout. Faster firing adds chops on TP-bound fills.

**Table A — HT RSI rollover at different speeds (G3, exploratory, post-hoc thresholds = the pre-registered 75 / −10)**

| variant | MASTER Δ (saves / chops, P) | BATCH | DESIGN | HOLDOUT (267) |
|---|---|---|---|---|
| pre-registered closed 5m | +0.108 (3 / 1, 0.69) | +0.32 (1 / 0) | +0.21 (6 / 3) | **−0.04 (14 / 14, P 0.30)** |
| live forming 5m bar | −0.14 (4 / 3, 0.30) | +0.60 (2 / 0) | +0.17 (6 / 4) | −0.08 (21 / 20) |
| closed-bar max, live drop | +0.06 (3 / 2) | +0.60 | +0.15 | −0.06 (17 / 18) |
| 1m RSI 75 / −10 | −0.04 (1 / 3) | +0.74 | +0.38 | −0.10 (10 / 23) |
| 1m RSI 80 / −20 | +0.08 (1 / 1) | 0 | +0.30 | +0.05 (5 / 4, 9 firings) |

On ENSO the live-bar variant fires 2 minutes earlier, at +1.23 % instead of +1.01 %, which is worth +0.06 pts. Lag is not what kills it.

## 2 · Labels and simulation — 3 trades recomputed by hand from the raw 1m bars

I wrote my own loop from the raw 1m bars, independent of `common.walk`.

| trade | my result | study `base_net` | notes |
|---|---|---|---|
| **ENSO** HT long 11-21 02:30:14 @ 0.5034 | Peak high +5.046 % gross at 02:43 (k 12). The TP limit 0.52893 (+5.071 %) needs a high of 0.52919 (+5.12 %): not touched. Stop line 0.48376 (−3.90 % gross = −4 net). Bar 03:21: low 0.4743 hits it, open 0.4908 is above the line, X = line × 0.999 → gross −3.998, **net −4.096** at k 50 | −4.096 STOP, hold 50 | Live booked −4.01 at 03:21:47. The trade sat at +1…+3.9 % for 35 minutes after the peak; the whole give-back is one candle (03:19, −3.6 %). RSI_ROLL (k 23, close +1.01 %) → exit at the 02:55 open 0.5082 = **+0.75 net, Δ +4.85** |
| **KGEN** HT 11-11 23:50:07 @ 0.08719 | TP limit E × 105.05 / 99.98. First high ≥ limit × 1.0005 at bar 00:26 (k 35) → **+5.000 net** | +4.995 TP, 35 | RSI_ROLL fired at k 18 (close −0.33 %) → exit −0.59 → **Δ −5.59** (the one master chop) |
| **TUSDT** FB short 10-25 08:00:10 @ 0.005979 | Funding −1.05 %/h … paid hourly. Cumulative paid −1.078 % by 09:50, so the DL 607 stop line moves to E × 1.0892. Stop at 09:50, X = max(line, open) × 1.0025 → **net −10.377** | −10.377 STOP | Booked +7.95 under the old +8 / −15 exit. Today's exit turns it into a stop because of the funding-paid stop, so the re-price is correct |

- **Label.** `now` = taker exit at `o[k+1]` × (1 ∓ slip) − entry fee − taker fee + funding to k (`build.py:285-286`), and label = now −
  today's result on the same path. This is correct. Decision rows exist only for k < exit bar, so the rule cannot fire after the exit.
- **Gates.** G1 = close P&L ≥ +1. G2 ≥ +2. G3 = running peak including bar k's high ≥ +2. All are applied as stated (`eval.py:64`).
- **Exit model.**
  - The maker TP with 0.05 % trade-through matches the engine's resting TP (`services/frenzy.py:815`, `flatbase.py:431`).
  - Stop-first on a bar that touches both is conservative.
  - The state exit pays the same taker fee and slip as the stop, so the costs are symmetric.

## 3 · Master cohort completeness and the XAN outlier

- **Completeness.** I rebuilt the cohort from all 117 `exports_archive/` order exports **plus** the 32 in `reports/.live_exports/`
  (which the study did not read): CLOSED, no probes, operator-closed out (0), dedup (opened_at[:19], pair, direction). The result is
  identical: HT 60 (49 master + 11 batch, first fill 10-23 20:30), FB 40 (36 + 4, first 10-23 04:00), EXH 5 (0 + 5, first 11-18). No
  fill is missing or extra, and no near-duplicate (same pair < 120 s apart) exists.
- **Re-pricing.** Correct: verified on KGEN, ENSO and TUSDT above. Every HT stop books −4.096, or −4.10 to −4.14 after a gap open; every
  TP books +5.00 when the limit fills, or more when the bar opens above it (BLUR +6.42).
- **⚠ Entries under older rules (M1).** Master HT entries predate the DL 594 dip ≥ 1.4 % gate. Using the stamped spike high H against the
  fill, 6 / 49 are < 1.4 % (AIN 11-05 1.22, AIN 11-08 0.71, UAI 11-09 1.11, KGEN 11-11 1.18, ONE 11-12 0.69, INIT 11-15 1.19). Excluding
  them changes:
  - RSI_ROLL·G3 from 4 firings / +0.108 to **1 firing / +0.037 %/fill**.
  - CLIMAX·G3 from +0.129 to **+0.005**.
  Master FB includes 6 / 40 entries that DL 601 now blocks (BTR 10-23, **LIGHT 10-24**, EVAA 10-29, DODOX 11-13, TAC 11-14, PHAROS
  11-19).
- **XAN 10-28 (FB parity outlier), explained.**
  1. XAN filled 10-28 12:00:09 UTC under the old **+8 / −15** exit. DL 408 (→ −5 / +10) was committed at 21:27 UTC.
  2. The path hit −5.57 % at +114 min (13:54 UTC). That is a TP under today's −5 %, but not under the −8 % in force then.
  3. After the deploy the +10 stop applied, and the bot booked STOP −10.09 at 22:38 UTC.
  4. The replay is right. The comparison was wrong: `cohorts.py:46` sets `par_ok = r.k >= "2026-10-28"` (a date string), so a pre-deploy
     fill was treated as parity-comparable. Without it the FB master parity gap is ≈ 0.02 (as the report says).

## 4 · Power, and whether the null was the right one

- **Which null.** The report compared Layer 1 against its own thesis-family null, not the broad-screen one. The HT family is 12 rules ×
  3 gates = 36 tests; the FB family is 30. The pre-registered verdict rule did **not** require thesis rules to beat that null, only
  broad-screen rules. So the null is not what failed HT. HT failed P(Δ > 0) ≥ 0.80 (best 0.69).
- **Why the null is biased (S1).** The circular shift keeps the same fills firing and only moves when they fire. That has two
  consequences:
  1. **It hands the null the rule's selection information for free.** The null can only test timing, but a give-back rule's value is
     mostly which fills it exits.
  2. **Shifting a condition that can only be true after a peak (RSI rollover, VWAP reclaim, lower highs) moves its first fire earlier.**
     Earlier means higher P&L, so the null is biased upward for turn signs. For signs that fire at high P&L (NEWHIGH, FB_BASE), it is
     biased downward.
- **Evidence for the bias.** Per-rule shift-null p-values come out as 0.000 or 1.000 in almost every block, instead of spread out as a
  fair null would give (`STATE_EXIT_AUDIT_2026-11-21_nullaudit.py`, 300–1,000 shifts). Some examples:
  - HT master RSI_ROLL·G3: observed +0.108 vs its own shifted null median **+0.266** (p = 1.00).
  - HT_NEWHIGH·G3: null median −0.235 (p = 0.00).
  - FB master FB_VWAP·G3: +0.407 vs null median +0.615.
  - EXH HOLDOUT EXH_OI_UP·G3: null median **+0.726** vs observed −0.04. This rule alone sets the EXH family-wise q95 of 0.80.
  Family-wise p = 1.0 for every rule in every block.
- **Conclusion on the null.** It does not measure "beats random timing" in a usable sense. Drop it from the argument, or replace it with
  a null that randomises **which fills fire**: permute condition sequences across fills with matched gate exposure, or compare against a
  random subset of the same size exiting at its first G3 minute.
- **Power (M2), simulated on the real master fills and day clusters.** I placed synthetic rules on the 49 HT master fills (19 days,
  9 give-backs, 25 TP): a save ≈ +3.8 ± 1.3 pts and a chop ≈ −4.6 ± 1.2 pts, the empirical sizes. Then I asked how often the rule
  clears the verdict bar.

| give-backs caught / TP chopped | precision | true Δ %/fill | P(pass P ≥ 0.80) | P(pass AND beat FW null 0.355) |
|---|---|---|---|---|
| 4 / 1 | 0.80 | +0.22 | 0.90 | **0.01** |
| 4 / 2 | 0.67 | +0.12 | 0.24 | 0.00 |
| 6 / 1 | 0.86 | +0.37 | 1.00 | 0.55 |
| 6 / 2 | 0.75 | +0.27 | 0.91 | 0.13 |
| 9 / 2 | 0.82 | +0.51 | 1.00 | 0.96 |

  The verdict bar (P ≥ 0.80) is reasonable: an 80 %-precision rule catching 4 / 9 give-backs passes it 90 % of the time. The FW-null bar
  needs about 6 / 9 caught at ≤ 1 chop (Δ ≳ +0.37 %/fill), which is half of what a perfect oracle would get (+0.70). A real but modest
  edge of about +0.1–0.2 %/fill is **not detectable on master**.
- **Backtest power.** The 267-fill HT HOLDOUT has the power the master lacks. On it, every turn sign's 95 % CI upper bound is ≤ +0.23
  %/fill (ANY·G3 ≤ +0.32), and their point estimates are −0.07…+0.04. So effects larger than about +0.25 %/fill are excluded there.
  Smaller ones are not.

## 5 · Why the HT asymmetries (RSI 4 / 13 vs 1 / 31 · CLIMAX 6 / 13 vs 4 / 31 · LH2 10 / 13 vs 9 / 31) don't make a rule

The "x / 13 vs y / 31" framing reads like precision, but it isn't: there are 2.4× more TP-bound fills than give-backs. The economics are
per firing:
- A save is worth about 4.1 + P&L at fire.
- A chop costs about 5 − P&L at fire.
- So at P&L ≈ 0 the break-even **precision is about 55 %**, and at +1 % it is about 44 %.

**Table B — every master + batch firing, G3** (pnl = gross P&L at the decision close; Δ = exit-now − today's result)

| rule | fill | outcome | min after final peak | pnl at fire | exit-now | today | Δ |
|---|---|---|---|---|---|---|---|
| RSI_ROLL | IOST 10-25 | give-back | +5 | −2.28 | −2.49 | −4.10 | +1.60 |
| RSI_ROLL | AIN 11-05 (dip 1.22) | give-back | −60 (before its +4.4 peak) | +1.48 | +1.30 | −4.10 | +5.40 |
| RSI_ROLL | KGEN 11-11 (dip 1.18) | **TP** | −17 | −0.33 | −0.59 | +4.99 | **−5.59** |
| RSI_ROLL | ONE 11-12 (dip 0.69) | give-back | +17 | −0.02 | −0.21 | −4.10 | +3.88 |
| RSI_ROLL | KAVA 11-19 (batch) | give-back | +5 | −0.33 | −0.53 | −4.10 | +3.57 |
| RSI_ROLL | *ENSO 11-21 (case)* | *give-back* | *+11* | *+1.01* | *+0.75* | *−4.10* | *+4.85* |

- **RSI_ROLL.** On master + batch it is 4 saves (+14.45) against 1 chop (−5.59), net +8.9 pts over 60 fills = +0.15 %/fill. The fires
  come late (median P&L −0.2 %), yet each still saves about 3.6 pts because the give-backs ran to the stop. **Timing is not the binding
  constraint; N and precision are.** On the backtest:
  - HOLDOUT: 14 saves (+48.5) vs **14 chops (−59.9)** → precision 50 % < 55 % → −0.04.
  - DESIGN: 6 / 3 → +0.21.
- **CLIMAX.**
  - Master: 6 saves (+22.4) vs 4 chops (−16.1) → +0.13.
  - HOLDOUT: 21 / 32 (precision 40 %) → −0.06.
  - Its chops include BLUR 10-30, where it fired 69 min before that trade's TP.
- **LH2.**
  - Master + batch: 10 saves (+42.3) vs 9 chops (−41.4) → 0.
  - HOLDOUT: 36 / 45 → −0.05.
- **HT_NEWHIGH** works like a "TP at the spike high". It fires at +2.5 %, so a save is worth about 6.4 and a chop costs about 2.3; it
  breaks even at about 26 % precision.
  - HOLDOUT: 17 / 68 = 25 % → −0.04.
  - DESIGN +0.65, master +0.03.

**Decomposition.**
- **Timing:** small. Faster variants (table A) don't improve anything.
- **Cost:** symmetric and small (taker 0.05 + slip 0.10 on both sides).
- **What decides it:** on large samples, the trades a turn sign fires on recover to TP about as often as they stop out. The master
  asymmetry is 5 firings, and 3 of the 4 master ones are on entries today's stack would not take.

## 6 · Other possible artefacts checked

**Table C — give-back-zone rules the G1-gated broad screen could not see (exploratory, Δ %/fill)**

| rule | HT master | HT HOLDOUT | FB master | FB HOLDOUT | EXH DESIGN / HOLDOUT |
|---|---|---|---|---|---|
| exit at break-even after +2 (G3 ∧ pnl ≤ 0) | −0.07 | −0.13 | −0.89 | −0.67 | −0.54 / −0.50 |
| G3 ∧ drawdown ≥ 2 | +0.02 | −0.05 | −0.59 | −0.82 | −0.63 / −0.69 |
| G3 ∧ drawdown ≥ 3 | −0.17 | −0.14 | −0.71 | −0.66 | −0.67 / −0.71 |
| peak ≥ 3 ∧ pnl ≤ +0.5 | +0.12 (4) | −0.13 | −0.45 | −0.44 | −0.51 / −0.42 |
| peak ≥ 4 ∧ pnl ≤ +1 | +0.03 (3) | −0.02 | −0.65 | −0.15 | −0.39 / −0.21 |
| 5m RSI −10 from in-trade max, no 75 floor | −0.27 | −0.14 | −0.97 | −0.91 | −0.68 / −0.39 |

- **Arming gate.** Raising the arming gate to +3 % or +4 % doesn't help; neither does dropping the RSI 75 floor. The gate is not the
  artefact.
- **1m vs 5m RSI:** see table A. 1m RSI is noisier and does worse on the holdout.
- **HT backtest cohort.** It is on the DL 594 stack (300 fills), and its exit replay agrees with the DL 600 replica on 299 / 300 exit
  kinds. It is the better-matched cohort for HT, and it is the one that shows ~0.
- **Not auditable, already listed as blind spots:** the predicted funding rate (the operator's −0.746 % on ENSO), order book / tape,
  in-trade spot share.

## What should be re-run / changed

1. **Fix the report's argument, not the verdict.**
   - Delete "no thesis rule beats random timing", or re-run the Layer-1 null as a **selection null**: permute condition sequences across
     fills within the same day-block with matched gate exposure, or compare against a same-size random subset exiting at its first G3
     minute, with a standardised (t) statistic.
   - Replace "the signs come too late" with the precision explanation (section 5).
2. **Re-read the master on today's entry stack:** HT with dip ≥ 1.4 (43 fills) and FB without the DL 601 entries (31 master). Report
   RSI_ROLL, CLIMAX and FB_VWAP there: 1 firing each for RSI_ROLL and FB_VWAP.
3. **Fix `par_ok` for FB** (`cohorts.py:46`): use the DL 408 deploy time (2026-10-28 ≈ 21:27 UTC), not the date.
4. **If the operator wants the RSI rollover watched,** register it as an OBSERVE line with a **precision gate**, not a Δ gate. Frozen
   definition (closed 5m, max ≥ 75, −10, G3), on today's HT stack. It needs ≥ 60 % of firings to be stop-bound over ≥ 15 firings on
   ≥ 8 windows before any arm. At roughly 1 firing per 12 HT fills, that is a long wait. Today's evidence (holdout 50 %) says it will
   likely fail.
5. **Where an edge could still be.** Effects ≲ +0.2 %/fill can't be excluded, and give-backs on HT are often a single crash candle after
   a long flat stretch (ENSO: 35 min flat at +1…+4 %, then −3.6 % in one bar). These are tape-speed events, so the next real test is the
   Route B order-book / trade-tape data and the predicted funding rate, as the study already says. No more kline-based state screens.
