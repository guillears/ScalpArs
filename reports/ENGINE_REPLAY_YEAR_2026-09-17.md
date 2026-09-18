# SCALPARS — Year-to-date ENGINE-CODE replay of today's live stack (Jan-01 → Sep-15 2026)

Operator brief: "Performance by Sleeve of what would have happened since January under today's live
filters per sleeve, N · WR · net $ · daily compound, $5,000 compounding; guarantee the methodology on
the periods we did run; audit that today's filters + multipliers are really in; double-click winners
vs losers per sleeve; recommendations with their impact on the current batch and on the pinned
live table." Method, fidelity and the six calibration passes: `reports/ENGINE_REPLAY_METHOD.md`.
Fills: `reports/backtest_cache/replay/ALL_fills.csv` (1,154 closed, every Order column).




## ⚠ CLOSING NOTE 2026-09-18 — tick data, era-config and exit-rule isolation tests

* **Real ticks (Binance aggTrades) for exits + the 20 s maker window**: matched-trade gap unchanged
  (live +0.354 vs replay +0.101 on 51 identical entries). The synthetic price path was NOT the cause.
* **Era-config replay (June rules on June, July rules on July, ticks)**: gap narrows to −0.15 (BASE)
  and −0.07 (B1) → the harness reproduces live within ~0.1–0.15%/trade when config matches; the
  residual is entry timing (replay fills ~14 bps late, one scan step after the live trigger).
* **Exit-rule isolation (today's entries, June's arm 0.70 / no lock / tp 0.45)**: +0.026%/trade on 235
  identical replay entries, WR unchanged, matched gap −0.254 → −0.212. The exit-parameter changes
  since July are worth ≈ +0.03%/trade at most — a minor lever, NOT the cause of the gap. Live
  cross-era data agrees: runners captured 58% of peak in every era; winners' peaks shrank
  (0.96 → 0.72) with the tape, not the rules. The +0.10 lock: 20 live fires, post-exit regret ≈ 0
  (60% ran again, 45% would have been stopped; no-lock CF −0.02/trade).
* **Standing verdict on the year table**: the replay carries a systematic exit-capture deficit of
  ≈0.15–0.25%/trade vs live that available data cannot remove (60 s scan on 1 m candles). Absolute
  year P&L is bracketed by "as replayed" and "+0.20/trade" (§ADDENDUM); relative reads (cohorts,
  gates, cells, populations) stand. Nothing from the exit side is recommended for ship.

## ⚠ RETRACTION 2026-09-18 — R4 (fade stop −1.5 → −0.7) WITHDRAWN before shipping

The Aug-10 stop widening carries a locked revert clause (CURRENT_STATE 41e): fills SAVED by the wider
stop (adverse ∈ (−0.70, −1.5], not full-stopped) at N≥6 → WR<50% ∨ Σ<0 → back to −0.70. Checked on
the fresh live cohort before shipping: saved-band fills 5 (RATS, XAN, ARC, AR, MERL) = 5 winners,
+$152; full −1.5 stops 3 = −$302 (vs −$158 at −0.70). Fresh kept cohort actual 22·86%·+$524 vs
+$156 under the old stop (no reversal credit) → the wide stop is +$368 BETTER live. Clause NOT met.

The replay counterfactual in §4/§5 was ONE-SIDED: it clipped losers at −0.7 but never stopped the
winners that dipped past −0.7 on the way to profit (the trades the change was built for). Corrected
reading: replay evidence on the stop is void; live evidence favours −1.5. R4 withdrawn. Fade sleeve:
entry gates working (blocked 45% / kept 79%), stop working, HEALTHY_BEAR regime = observe-only
(shared separator, real 13·54%·+0.07 vs 40·85%·+0.42; replay diff −0.28, 6/8 months) with gate
N≥30 real ∧ (WR≤55% ∨ Σ<0).

## ⚠ ADDENDUM 2026-09-18 — exit-capture sensitivity SUPERSEDES the absolute numbers in §0-§1

The operator challenged the year result against the live batches. The challenge was right. On the
35-51 trades where live and replay took the IDENTICAL entry (same pair, same minute), live earned
+0.35 to +0.50% per trade and the replay +0.11 to +0.31%. Three exit emulations were tested on the
June-July windows (stepped adverse-first, favourable-first, wick-stop): none reproduces live
(+0.31 / +0.22 / +0.30 vs +0.50); the replay's peaks are systematically lower (0.70 vs 1.00) because
its exits fire earlier, and 1-minute data cannot resolve why. **The gap is ≈0.2% per trade and it
decides the year:**

| shared $5,000 book, today's rules | as replayed | +0.20%/trade (live-calibrated) |
|---|---|---|
| Jan → Apr | 1,453 → 357 → 218 → 114 (ruin) | 7,835 → 11,153 → 21,561 → 25,604 |
| Sep-15 | $114 (−1.45%/day) | $244,443 (+1.52%/day; August bull window ×3) |

So the replay CANNOT determine the sign of the out-of-sample year. What it can determine is what
is robust to that bias:

* **Robust (holds in both views):** the fade stop (R4): fade P&L improves in 9 of 9 months either
  way (as replayed −$10.1k → +$2.6k; calibrated −$5.7k → +$8.6k), and live confirms the mechanism
  (post-Aug-10 fades: WR unchanged, avg +0.47 → +0.11, losers −0.63 → −1.40). The payoff-structure
  diagnosis (losers at the stop, winners keeping ~40% of peak) is a live fact as much as a replay fact.
* **NOT robust:** R1 (BTC RSI ≥ 64 block) — helps as replayed (+$10.6k), HURTS in the calibrated
  view (the blocked cohort is break-even once exits are credited) and the live zone is positive
  (40·70%·+$653). Downgraded to OBSERVE. R2/R3 (2× cells): small either way → OBSERVE, standard
  cell verdicts. R5 (drawdown brake): bias-dependent → sizing discussion, not a ship.
* **The strategic fact that survives:** at 20× with four slots, ±0.2% of exit quality per trade
  is the difference between ruin and +1.5%/day. The live edge (+0.26%/trade on the pinned window) is
  thin against that sensitivity. Exit capture is the lever, and the replay is not precise enough to
  tune it; the live BE-lock shadow (24b) and the runner giveback are where that work must happen.

Everything below this line is kept as the audit trail of how the numbers were produced.

## 0. The one-paragraph answer (as replayed — read the addendum first)

Under today's exact rules and sizing, a $5,000 book started on Jan-01 is **ruined by April ($114 on
Sep-15, max drawdown −98%)**. The out-of-sample half (Jan-01 → Jun-16, the tape the filters were
never tuned on) loses in every month and in every sleeve except BearRun; the in-sample half
(Jun-17 → Sep-15, the live batches the filters were tuned on) is +$23.5k, of which **+$22.7k is one
six-day bull-run window (Aug 19-24)**. Two sleeves destroy the book: **Spike-Fade** (own ledger
$5,000 → $256) and **Momentum-long** ($5,000 → $349). Without those two, the same year is
$5,000 → $20,303 (+0.55%/day, max DD −59%). The live record and the replay agree wherever the
rules were the same (June baseline: momentum-long 39·85%·+$5,094 replay vs 41·85%·+$3,506 live);
they diverge exactly where today's rules are LOOSER than the rules that produced the live fills
(slope cap disabled Aug-21, CALM3D door added Jul-27, 2× cells) and where a stop was widened
(fade −0.7 → −1.5 on Aug-10). Four rules are recommended; their live cost and their replay gain are
both stated. **No entry filter makes momentum-long positive out of sample.** Its edge is
regime-dependent, and the sizing (20×, 4 slots, 2× cells) turns a flat sleeve into ruin.

## 1. Performance by sleeve — today's rules, $5,000, compounding (engine sizing at the running balance)

### 1a. Full period Jan-01 → Sep-15 (258 days)

| sleeve | N | WR | avg P&L % | net $ (fixed $5k book) | own ledger $5,000 → | daily compound | max DD |
|---|---|---|---|---|---|---|---|
| BearRun-Short | 17 | 71% | +0.13 | +$690 | $5,546 | +0.040% | −10% |
| BullRun-Long | 226 | 51% | +0.22 | +$21,073 | $17,519 | +0.488% | −46% |
| FLIP-short | 99 | 71% | +0.03 | +$1,425 | $6,560 | +0.105% | −31% |
| MOM-long | 437 | 62% | −0.05 | −$11,420 | $349 | −1.028% | −93% |
| MOM-short | 151 | 64% | −0.00 | +$641 | $4,577 | −0.034% | −44% |
| Spike-Fade | 224 | 66% | −0.18 | −$10,120 | $256 | −1.146% | −95% |
| **TOTAL (shared book, 4 slots)** | **1,154** | **62%** | **−0.01** | **+$2,289** | **$114** | **−1.455%** | **−98%** |

Shared book chained: Jan $5,000 → $1,453 (−71%) · Feb → $357 (−75%) · Mar → $218 · Apr → $114 · flat
after (below minimum size). The fixed-book column is positive only because the August bull window
is counted at full size after the book was already dead in the chained view.

### 1b. Out-of-sample (Jan-01 → Jun-16, 167 days) vs in-sample (Jun-17 → Sep-15, 91 days)

| sleeve | OOS N·WR·avg·net | IS N·WR·avg·net |
|---|---|---|
| BearRun-Short | 10·90%·+0.33·+$843 | 7·43%·−0.15·−$153 |
| BullRun-Long | 55·49%·−0.10·−$1,635 (2 windows) | 171·52%·+0.32·+$22,708 (1 window) |
| FLIP-short | 62·66%·−0.03·−$732 | 37·78%·+0.12·+$2,157 |
| MOM-long | 308·59%·−0.07·−$9,948 | 129·68%·+0.02·−$1,472 |
| MOM-short | 123·62%·−0.04·−$1,476 | 28·75%·+0.19·+$2,117 |
| Spike-Fade | 144·62%·−0.20·−$8,280 | 80·72%·−0.15·−$1,840 |
| TOTAL | 702·61%·−0.08·−$21,228 | 452·64%·+0.12·+$23,517 |

### 1c. Daily compound per sleeve per month (own $5k ledger each; N in brackets)

| month | BearRun | BullRun | FLIP | MOM-long | MOM-short | Spike-Fade | TOTAL |
|---|---|---|---|---|---|---|---|
| Jan | +0.05% (1) | −0.71% (21) | −0.09% (12) | −2.58% (69) | +0.07% (10) | −0.99% (24) | −3.94% (137) |
| Feb | – | – | −0.12% (11) | −3.06% (56) | −0.04% (32) | −1.47% (14) | −4.86% (113) |
| Mar | +0.28% (6) | – | −0.03% (10) | −0.52% (55) | −0.16% (26) | −0.39% (23) | −0.77% (120) |
| Apr | – | −0.40% (20) | +0.20% (10) | −0.61% (31) | −1.06% (14) | −1.01% (28) | −2.90% (103) |
| May | – | – | −0.04% (16) | −0.26% (63) | −0.01% (18) | −2.46% (26) | −2.87% (123) |
| Jun | +0.26% (4) | +0.09% (14) | +0.40% (17) | −0.49% (56) | +0.30% (28) | −1.94% (57) | −0.93% (176) |
| Jul | – | – | +0.07% (8) | −0.12% (50) | +0.10% (4) | +0.95% (26) | +0.75% (88) |
| Aug | – | +4.80% (171) | +0.20% (7) | +0.13% (30) | +0.40% (14) | −0.85% (18) | +4.85% (240) |
| Sep (15d) | −0.49% (6) | – | +0.62% (8) | −3.09% (27) | +0.21% (5) | −1.13% (8) | −4.02% (54) |

Reading: momentum-long is negative in 8 of 9 months; Spike-Fade in 8 of 9; FLIP and MOM-short are
coin-flips out of sample and positive in the tuned months; BullRun is one event.

## 2. Audit — is the replay really running today's filters and multipliers?

* Config: `trading_config.json` loaded by the replay is byte-identical (MD5) to the deployed HEAD.
  Engine code = HEAD + the 2-line signal-expired fix (§7).
* Gates: all 19 gate tags the live bot logged in its last 14-hour server log appear in the replay's
  filter-block counters, same headline blockers on top.
* Cells: identical cell tags and sizes to live (UNMATCHED 2×, CALM3D 2×, SPIKE_FADE 2×, NEGDI15 2×,
  QS flip cell, W/C 1×, BULLRUN, BEARRUN).
* Hard-gate compliance on 1,090 fills: zero violations of every gate the pool builder encodes (BTC
  ADX strictly 18.1–39.9 on momentum fills, no fan flip past the bearish-BTC / 1h-slope / trend-depth
  gates, no matched-pattern long outside the door, no CALM3D under its DMI legs, no fade over $20M
  or above EMA13, no blacklisted pair).
* Fade entries: 214 of 222 replay fades pass the LIVE mid-candle trigger legs rebuilt minute by
  minute from 1m bars (the "volume-leg" explanation was tested and dropped).
* Batch-by-batch: BASE window (Jun-17 → Jul-8) live kept 76·87%·+$4,919 vs replay 116·80%·+$7,867;
  momentum-long 35·91%·+$3,482 vs 39·85%·+$5,094 — same trades. B1 (Jul-8 → 31) live +$213 vs
  replay −$1,286: the 16 extra replay longs are the July trades that the rules of the time blocked
  (10 above the 0.15 slope cap, 10 through the CALM3D door that did not exist yet).

**Why the master pool is positive and the replay is not:** "kept by today's rules" can only REMOVE
trades from what the old rules took; it can never ADD the trades the old, stricter rules blocked.
Every loosening since June (slope cap off Aug-21, CALM3D door Jul-27, NEGDI15 re-arm, UNMATCHED 2×,
STRONG_BEAR exemption) is invisible in the pool and visible in the replay. That is not a harness
error; it is the thing the pool structurally cannot show.

## 3. The pinned live table, three ways (Jun-17 → Sep-7)

| sleeve | LIVE pinned (as reported Sep-8) | LIVE, same fills, raw-P&L basis | LIVE with recommendations R1-R3 | REPLAY same window, today's rules |
|---|---|---|---|---|
| MOMENTUM LONG | 109·80%·+0.22·+$5,511 | 113·80%·+0.20·+$5,397 | 74·85%·+0.28·+$3,339 | 118·68%·+0.02·−$1,210 |
| MOMENTUM SHORT | 39·74%·+0.20·+$812 | 39·74%·+0.20·+$812 | unchanged | 25·72%·+0.17·+$1,931 |
| FLIP short | 42·74%·+0.17·+$1,704 | 29·83%·+0.31·+$971 | unchanged | 32·78%·+0.11·+$1,950 |
| SPIKE_FADE | 48·77%·+0.32·+$1,273 | 47·77%·+0.31·+$1,210 | unchanged (R4 acts on 4 post-Aug-10 losers: ≈ +$225) | 78·72%·−0.16·−$1,940 |
| BULLRUN_LONG | 22·55%·+0.62·+$1,560 (v2 cohort) | 49·33%·−0.14·−$843 (all fills) | unchanged | 171·52%·+0.32·+$22,708 |
| TOTAL | 260·75%·+0.26·+$10,860 · +3.75%/day (65 active days) | 286·70%·+0.16·+$7,138 · +$117/active day | 247·70%·+0.18·+$5,079 · +$83/active day | 425·63%·+0.12·+$23,540 |

Basis note: the pinned table uses the pool's counterfactual-adjusted P&L (fade SL re-price, sprint
de-mux) and the bull-run v2 cohort; columns 2-3 use raw closed P&L on the same kept set so that
before/after is apples to apples. **The recommendations cost about $1.8k of the pinned window's
past profit** (they block 39 live longs that were the weaker half of the sleeve, and halve two 2×
cells) **and remove about $9k of the replay's out-of-sample bleed** (§5). That trade-off is the
decision.

## 4. Double-click per sleeve — winners vs losers

Sweep: every stamped entry dimension × decile cuts × both sides; a candidate must block ≤50% of the
sleeve, be net-negative overall AND out of sample on its own, hold in ≥60% of months, agree with its
neighbouring cuts, not be 2-pair concentrated, and leave the kept side better out of sample.

**Momentum-long (437·62%·−$11,420).** Exit anatomy: winners +0.41% giving back 63% of a 0.66 median
peak; losers −0.79%, 99% stopped, 46% never positive. At 62% WR the payoff ratio 0.52 gives
−0.05%/trade; break-even needs 66% WR. Separators that survive:
* BTC RSI (prev bar) ≥ 64: blocks 176·57%·−$9,844 (OOS 114·51%·−$7,088), **9/9 months negative, 110
  days**; kept 261·66%·−$1,576. Live: the same zone is the weaker half (40·70%·+$653 vs <64
  78·83%·+$3,982). → R1.
* Entry distance from EMA13 < 0.33% (entering before the pair is extended): blocks 218·58%·−$10,759
  (OOS −$8,320), 9/9 months; kept 219·66%·−$660. Blocks half the sleeve → sleeve-verdict class,
  OBSERVE (window units) not a filter.
* BTC ATR < 0.18% (dead tape), BTC within 0.54% of its 24h high, BTC ADX ≥ 33: all 9/9 negative, all
  overlap the two above. Coherent story: **momentum longs lose when BTC is already extended
  (overbought RSI, at the 24h high, high ADX) and win as pullback-buys**. This is the climax logic
  gate 51 band ③ encodes at RSI 70+; the data says the line is 64.
* The relaxations: slope>0.15 zone 124·60%·−$3,604 (6/9 months negative; live since the disable
  14·93%·+$613 — not decisive either way → HOLD, read at N=25 live); CALM3D door 68·57%·−$5,037
  (7/9 negative; OOS 41·54%·−$2,267; live 13·92%·+$1,076 in-sample) → R2; UNMATCHED 2× 208·63%·
  −$5,153 (OOS the 2× cost $3,468 extra; IS it earned +$891; live 84·77%·+$3,651) → R3.
* **No combination makes the sleeve positive out of sample** (after R1-R3: OOS 184·65%·−$824, IS
  64·69%·+$1,904). The sleeve is flat-to-slightly-negative OOS and profitable only in the tuned tape.

**Momentum-short (151·64%·+$641).** OOS −$1,476, IS +$2,117. Winners +0.39 (give back 35%), losers
−0.71, 46% stopped, 23 EMA13-cross exits all losers (−$3,371, by construction). Candidates: pair
volume ratio < 0.70 (blocks 45·49%·−$2,212, 5/6 months, kept OOS +$443) and pair EMA13-50 gap
< −0.74 (already deep below trend = chasing: 60·60%·−$1,546, 6/7 months). Live disagrees on the
first (14·86%·+$453 in the zone) → OBSERVE both with a forward gate, no ship.

**FLIP-short (99·71%·+$1,425).** OOS −$732, IS +$2,157. Winners +0.46 (give back 37%), losers
−1.00 all stopped. One weak candidate (BTC trend gap ≥ −0.08: 30·67%·−$825, 4/6 months) — it
straddles the TG_SHALLOW band and contradicts the cell; not shippable. NEGDI15 2× is the sleeve's
best cohort (49·76%·+$2,180; OOS flat, IS +$2,181); TG_SHALLOW 21·81%·+$594; plain 1× flips
40·65%·−$759; STRONG_BEAR exemption 16·62%·+$242 (OOS −$391, IS +$633). Verdict: sleeve OK in the
tuned tape, no edge OOS, no filter.

**Spike-Fade (224·66%·−$10,120).** The book-killer. Winners +0.57 (give back 37%), **losers
−1.65, 97% stopped, 67% never positive**. No entry separator leaves the kept side positive
(128 "candidates" all fail that test). The mechanism is the stop: at −1.5 a 66% WR needs +0.78 per
winner to break even; the sleeve makes +0.57. Live confirms after the Aug-10 override: fades before
Aug-10 (−0.7 stop) 30·77%·+$1,188 avg +0.47, losers −0.63; since Aug-10 (−1.5 stop) 20·80%·+$254
avg +0.11, losers −1.40 — WR unchanged, payoff collapsed; the 4 live losers all ran to −1.5 and were
still −0.5 to −3.4% thirty minutes later (the wide stop saved nothing). Replay counterfactual by
stop width (year): −1.5 → −$8,005 · −1.0 → −$1,659 · **−0.7 → +$2,552 (OOS +$588, IS +$1,965)**.
Fidelity caveat: replay fades enter at candle close (live mid-candle) and the sleeve is the least
faithful in calibration; the direction of the stop finding is supported by live. → R4.

**BullRun-Long (226·51%·+$21,073).** One window (Aug 19-24, 171 fills, +$22.7k, 20× with the
re-arm door); the two OOS windows lost (Jan 21·43%·−$1,118, Apr 20·55%·−$599). Separator: weak
72h run (r72 < 9.2%) 88·40%·−$11,683 — but it is a window variable with 3 windows in total →
OBSERVE; the manual kill bar stands. Exits: 110 stops −$44.8k vs 59 trails +$33.3k + 17 ladder
+$29.8k: a right-tail sleeve, live with it or not at all.

**BearRun-Short (17·71%·+$690).** OOS 10·90%·+$843 (Mar/Jun windows), IS 7·43%·−$153 (the Sep-15
evening: 6 fills 2W/4L, three EMA13-cross exits). Too small; the probe/kill-bar machinery stands.

## 5. Recommendations (each with live cost and replay gain)

| # | change | replay year Δ | replay OOS Δ | live pinned-window Δ (past) | current batch Δ |
|---|---|---|---|---|---|
| **R1** | Momentum-LONG: block when BTC RSI(prev 5m bar) ≥ 64 (`btc_rsi_prev` climax line; sleeve switch, window units) | +$10,636 | +$7,657 | −$1.5k (39 fills·70% blocked; the weaker half) | OP −$188 avoided → batch −$303 → −$128 |
| **R2** | CALM3D door 2× → 1× | +$2,518 | +$1,134 | −$538 | OP loss halved (moot with R1) |
| **R3** | UNMATCHED 2× → 1.5× (the staging step it skipped) | ≈ +$1,300 | ≈ +$1,700 | ≈ −$900 | BTW/DASH −$13 |
| **R4** | Spike-Fade stop −1.5 → −0.7 (revert the Aug-10 override) | +$10,557 (CF) | +$8,134 (CF) | +$225 on the 4 post-Aug-10 losers | none (3 fade wins) |
| **R5** | Book-level monthly drawdown pause: no new entries for the rest of the month once the book is −20% from month start (new feature, D11) | $114 → $1,837 with R1-R3; $20,978 → $18,987 once fades are fixed (cheaper DD: −75% → −59%) | – | none (no month reached −20%) | none |

Year ledger, $5,000 → : today's rules **$114** · R1-R3 **$109** (fade still kills it) · R1-R3 + R5
**$1,837** · R1-R3 with fades OFF **$20,978 (+0.56%/day, DD −75%)** · same + R5 **$18,987 (DD −59%)**.
R4 (fade at −0.7) is expected to land between the "fades OFF" and "fades ON" cases; its ledger
needs a re-run at −0.7 (queued).

Pre-committed gates: R1 — PASS phantoms on the blocked signals; revert if blocked cohort ≥65% WR and
Σ>0 on N≥20 (≥6 dates). R2/R3 — standard multiplier verdict on the next 5 fresh fires each. R4 —
fade losers avg ≤ −0.9% on N≥10 losers → widen back to −1.0. R5 — none (risk control).

**Observe-only (registered, no ship):** slope-cap zone (G56: replay −$3.6k/yr vs live +$613 since
the disable; read at N=25 live) · MOM-long "too-early" longs (dist-EMA13 < 0.33) · MOM-short pair
volume ratio < 0.70 and pair gap < −0.74 · BullRun weak-run r72 < 9 · NEGDI15 / TG_SHALLOW /
exemption unchanged (their year reads are positive or too small).

**Not recommended:** any new momentum-long entry filter beyond R1 (none makes the sleeve positive
OOS; adding more only fits the year); switching momentum-long off (the tuned-tape edge is real and
R1-R3 make it flat OOS — the decision is sizing, not existence).

## 6. What the year says strategically

1. The system's profit is regime-conditional: it made money in the Jun-Sep tape it was tuned on
   and lost in Jan-May. Every live batch being positive under "today's rules" is a survivorship
   artefact of the pool, not evidence of an all-weather edge.
2. Two mechanics, not entries, explain most of the loss: fade stop width (R4) and 2× sizing on
   sleeves that are flat out of sample (R2/R3). Momentum-long's remaining gap is the giveback: winners
   keep 37% of their peak. The BE-lock shadow (watch 24b) is the next study, on 437 replay fills.
3. Compounding at 20× with 4 slots and no book-level brake turns a −0.05%/trade sleeve into ruin in
   two months. R5 is cheap insurance regardless of anything else.

## 7. Side findings

* Live bug (fix prepared, uncommitted, suite green): `_record_signal_expired_order()` rejects
  `entry_btc_trend_gap_pct` (passed since commit 1792492) → every maker-window expiry raises and
  the SIGNAL_EXPIRED row is never written. No money effect.
* Pool builder: fan-flip bearish-BTC gate added (stack v2026-09-16a) — see DECISION_LOG 65.
* Fidelity: individual live fills reproduced 27-39% (sub-minute door legs); populations match at
  sleeve level for MOM-long / MOM-short / FLIP; Spike-Fade low fidelity; BullRun one window.
