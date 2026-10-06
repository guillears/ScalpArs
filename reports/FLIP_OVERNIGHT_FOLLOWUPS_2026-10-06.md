# FAN flip-short overnight follow-ups (2026-10-06)

Follow-ups to `reports/FAN_FLIP_QNT_LOSS_REVIEW_2026-10-06.md`. Research only: nothing in services/, config, tests, templates or the replay scripts was touched, and nothing was committed.
- Patch proposal for the replay agent: `reports/REPLAY_FLIP_PARITY_PATCH_PROPOSAL_2026-10-06.md`.
- Scripts and CSVs: scratchpad `…/c066304e-…/scratchpad/flipfu/` (listed at the end).

---

## 1. Plain-language answers

### A. Is the yr5 flip loss real, or a replay artefact?
**Mostly real for today's settings. About 40 % of the loss the review reported was artefact, and none of that artefact is in the replay's exit.**

1. **The replay's exit is right.** I rebuilt the flip exit independently and ran it on the real trade-by-trade price feed. It covers the ATR-widened stop, the runner trail, the take-profit ladder and fees. It reproduces:
   - the replay's result on all 411 yr5 flips, to within 0.013 points on average;
   - live's result on all 49 live flips, to within 0.019 points.

   Stop distance, ATR, trail arming, tick vs 1-minute data, and price-check speed explain **none** of the gap. Every pair-day on both sides had real trades, so no 1-minute fallback was used.
2. **The review over-counted.** The replay runs each half-month chunk with a 3-day warm-up, and fills opened during the warm-up are saved too. They repeat time the previous chunk already covers. 55 of the review's 466 flips were these duplicates, and they averaged −0.34 %. Removing them gives **411 flips (137 per seed) at −0.137 % per trade, not −0.161 %.**
3. **The review's dollar figure was in the wrong units.** −$9,959 per seed adds up raw replay dollars from chunk books that each started at $5k and compounded. On the standard fixed $3,000 book at live sizing, the flip sleeve is **−$4,616 per seed** at today's 2× cells, or −$2,738 at 1×.
4. **The replay's entries come out slightly worse than they should.** It mostly lands on the unlucky side of its own entry moment: 0.039 points worse than entering at random within ±1–2 minutes (CI −0.071 … −0.007). Live shows no such effect (−0.007, ≈ 0). The gap sits in the maker (limit-order) fills. Taking it out entirely: **−0.082 % per trade, −$2,850 per seed, still negative at P = 0.969**, in both halves of the year and in 8 of 9 months.
5. **The big live-vs-replay gap on matching trades is mostly today's settings, not the replay.** On the 26 live flips that have a replay twin, live made +0.282 % and the replay +0.024 %. That gap of −0.26 splits into:
   - **−0.06 from today's exit.** The Aug-14 give-back cap cuts the big June runners short.
   - **−0.20 from entry timing.** Today's extra flip gates veto the exact moment live entered and admit the trade a few minutes later, at a lower and worse price. Three of those later entries got stopped out on a knife-edge.
   - **0.00 from replay mechanics.**
6. **The live record does not support today's flip setup.** All 49 live flips, re-run through today's exit, average +0.041 %, not the +0.31 % headline (that one is the 31 fills today's filters would still keep, at the old exits). Inside the dates live actually covers, the replay with timing corrected is −0.008 %. So live and the replay roughly agree. The yr5 loss sits mostly in months live never traded (Jan–May, mid-Jul to Aug-10, late Aug to Sep-10).

**Bottom line for A:** the flip sleeve under today's settings loses about **0.08–0.14 % per trade** in the replay. The true value is probably about −0.10: the trimmed figure, minus the replay's extra entry penalty measured against live. That is smaller than the review said, but the sign holds. It is still a "fix sizing and watch" finding, not a kill. The review's recommendations stand: NEGDI15 and TG_SHALLOW go to 1×, plus the sleeve-level bar.

### B. Does "pair rising while BTC falls" separate the losers?
**No.**
- **It describes almost every FAN flip, not the losing ones.** "Pair up against BTC over 30 minutes while BTC falls" is true for 93 % of yr5 flips (382 / 411) and 43 of 51 live ones. A FAN flip fires when a pumping pair's long is blocked, and the flip gates require BTC to be bearish. The group "passes" the expectancy bar on yr5 only because the whole sleeve is negative there. It is no worse than the rest: the shuffled-label null gives p = 0.67, and its "rest" is worse than the group itself.
- **The pre-registered QNT signature does not even match QNT.** The signature is pair 30-min return − BTC ≥ +1.0, BTC falling, and 30-min volume ≥ 2× the prior 2 h.
  - It is rare: 28 yr5 fills and 2 live.
  - It is not worse than the rest: −0.122 vs −0.138, null p 0.55.
  - QNT itself misses it. Its pre-entry volume ratio was only 0.76. The 2–4× volume came **after** the entry, so it was not knowable at the signal.
- **No continuous version works in both datasets either:**
  - rs30, rs60, the pair's own 30/60-min move, the volume ratios and the share of the drop recovered, read at sign then terciles: nothing beats chance, and the master does not agree with yr5.
  - A post-hoc "pair already up > 1.1 % in 30 min" cut leans the right way on yr5 (null p 0.06–0.10), but the master does not support it (17 · −0.01 vs +0.09 for the rest). Not proposed.
- **By-product:** the review's observe candidate (pair EMA13−50 gap > 0.5) **survives the timing correction**: yr5 144 fills · −0.157 vs −0.041 for the rest, shuffled null p 0.03. It stays observe-only, threshold frozen.

---

## 2. A — replay flip parity in detail

### 2a. Exit replica validation (the key negative result: no exit bug)
| Check | N | mean abs diff | within 0.05 / 0.10 pp | notes |
|---|---|---|---|---|
| Replica (era's exit config, live entry, real ticks) vs **live** | 49 | 0.019 | — / 98 % | one outlier: BICO 08-12 (live −0.06 vs replica +0.72); live closed on an early negative-floor trail before the Aug-12 negfloor ride existed |
| Replica (today's exit, replay entry, real ticks) vs **replay** | 411 | 0.013 | 99 % / 99 % | 4 outliers (CRV, CHZ, 2× SAHARA in May), all with ladder or trail timing at the bar edge |

Data: all 49 live and all 411 replay pair-days have quantity-bearing trade archives (`ticks_q`), so there is no 1m fallback anywhere. The replay ran with `tick_max_pts=0`: every trade goes to the realtime exit path, the same as live's `@trade` WebSocket. The "1 Hz monitor" touches only the slow monitor loop; flip exits fire in the per-tick path in both.

**Stop computation**, identical in live, replay and replica: SL = −max(0.70, min(1.5 × entry ATR %, 1.20)). Entry ATR % is the 5m ATR stamped at the signal scan.
- On twins, the replay's ATR is within ±0.07 of live's.
- Stop fill overshoot: live 0.00–0.02; replay 0.00–0.04, plus 0.027 on taker-entry fees (e.g. FET: −0.975 vs live −0.948 is just a taker vs maker entry fee).

### 2b. Exit-config eras (git history of `trading_config.json`)
| Era | Runner arm | Give-back | HARD_TP ladder | Live flips |
|---|---|---|---|---|
| Jun-17 → Jul-22 | 0.45 | 0.5 × ATR (no cap) | none | 31 (BASE) |
| Jul-22 → Aug-5 | 0.45 | 0.5 × ATR | short ladder 1.0:0.25 … | 0 (sleeve dead Jul-8 → Aug-11) |
| Aug-5 → Aug-14 13:35 UTC | 0.40 | 0.5 × ATR | yes | 4 |
| Aug-14 → now (= yr5 frozen config) | 0.40 | **min(0.5 × ATR, 0.35 × peak)** | yes | 14 |

All 49 live flips re-priced with **today's** exit (replica, same entries):

| Era | N | As-lived | Replica (era exit) | **Today's exit** | WR as-lived → today |
|---|---|---|---|---|---|
| Jun-17 → Aug-5 | 31 | +0.223 | +0.222 | **+0.154** | 81 % → 84 % |
| Aug-5 → Aug-14 | 4 | +0.160 | +0.356 (BICO) | +0.228 | 50 % → 75 % |
| Aug-14 → | 14 | −0.253 | −0.262 | −0.262 | 43 % |
| **All** | 49 | **+0.082** | | **+0.041** | |

The give-back cap trims the big June runners: BIO +1.23 → +0.72, JTO +0.72 → +0.27, FARTCOIN +0.96 → +0.24, BR +0.69 → +0.26, H +1.28 → +0.58. **On yr5 the cap is neutral:** −0.138 with it, −0.123 without, −0.134 under the full June exit. **So the cap is not why yr5 loses.**

### 2c. Per-trade side by side: the 26 live flips with a replay twin (same pair, ±30 min; 61 replay rows over 3 seeds)
Decomposition per row:
- **exit-cfg** = live entry at today's exit − live as-lived;
- **entry** = replay entry at today's exit − live entry at today's exit;
- **mech** = replay actual − replica on the replay entry.

The columns are averages over the seeds that matched. **journal** is what each seed's engine logged for that pair at the live moment (−4 / +2 min):
- **open** = it opened too;
- **veto** = flip gates refused it;
- **pre-empt** = a LONG-side gate fired before the FAN gate, so no flip candidate existed.

| live at (UTC) | pair | era | seeds | dt min | entry Δpx % | live | live@today exit | replay | exit-cfg | entry | mech | journal | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 06-17 06:46 | SENT | BASE | 1 | +3.2 | −0.41 | +0.69 | +0.53 | +0.27 | −0.16 | −0.26 | 0 | pre-empt ×3 | exit-cfg + gates moved entry |
| 06-17 07:26 | JTO | BASE | 1 | −0.6 | −0.09 | +0.15 | +0.37 | +0.29 | +0.22 | −0.08 | 0 | open, veto, veto | exit-cfg (helps) |
| 06-17 07:40 | XMR | BASE | 3 | +0.4 | +0.10 | +0.26 | +0.29 | +0.35 | +0.03 | +0.06 | 0 | open, open, pre-empt | match |
| 06-17 07:41 | JTO | BASE | 1 | −15.6 | −0.67 | +0.72 | +0.27 | +0.29 | −0.45 | +0.03 | 0 | pre-empt ×3 | exit-cfg |
| 06-18 03:42 | BIO | BASE | 1 | +3.9 | +0.06 | +1.23 | +0.72 | +0.30 | −0.50 | −0.42 | 0 | pre-empt ×3 | exit-cfg + entry timing |
| 06-18 09:01 | **STG** | BASE | 3 | +11.5 | −0.68 | +0.19 | +0.39 | **−1.20** | +0.21 | **−1.63** | +0.03 | veto ×3 (RSI_MIN, BEAR_MIN, ADXD, WEAK_BOUNCE, QUALITY) | **gates moved entry → knife-edge stop** |
| 06-18 22:18 | BR | BASE | 2 | +0.3 | −0.02 | +0.69 | +0.26 | +0.27 | −0.43 | +0.01 | 0 | open, veto, open | exit-cfg |
| 06-18 22:18 | INJ | BASE | 3 | +0.6 | −0.06 | +0.31 | +0.27 | +0.30 | −0.04 | +0.03 | 0 | open ×3 | match |
| 06-24 10:31 | DEXE | BASE | 3 | −0.1 | +0.02 | +0.26 | +0.30 | +0.27 | +0.04 | −0.03 | 0 | open ×3 | match |
| 06-24 12:55 | **DEXE** | BASE | 3 | +0.9 | −0.09 | +0.68 | +0.29 | **−1.08** | −0.39 | **−1.37** | 0 | open ×3 | **same moment, knife-edge stop** (live trough −1.01 vs stop −1.08; replay entered 0.05–0.18 % lower / taker) |
| 06-25 01:10 | DEXE | BASE | 3 | −0.0 | −0.06 | +0.16 | +0.30 | +0.33 | +0.13 | +0.04 | 0 | open ×3 | exit-cfg (helps) |
| 06-27 22:05 | SKYAI | BASE | 3 | +5.7 | −0.20 | +0.30 | +0.26 | +0.27 | −0.04 | +0.01 | 0 | pre-empt ×3 | match |
| 07-01 10:23 | **TAC** | BASE | 2 | +8.1 | −0.43 | +0.17 | +0.49 | **−0.44** | +0.32 | −0.93 | 0 | pre-empt ×3 | **gates moved entry → stop (s3)** |
| 07-01 11:02 | XPL | BASE | 1 | +0.3 | 0.00 | +0.48 | +0.33 | +0.25 | −0.15 | −0.07 | 0 | open, pre-empt ×2 | exit-cfg |
| 07-01 11:49 | LIT | BASE | 1 | +0.4 | +0.01 | +0.34 | +0.29 | +0.32 | −0.05 | +0.02 | 0 | veto ×2, open | match |
| 07-07 08:41 | RIF | BASE | 2 | +3.4 | −0.18 | −1.19 | −1.20 | −1.20 | −0.01 | 0 | +0.01 | veto, pre-empt ×2 | match |
| 08-12 14:16 | PUMP | B3 | 3 | +1.6 | −0.23 | +1.27 | +1.27 | +1.02 | 0 | −0.25 | 0 | open, veto, open | entry timing |
| 08-12 17:26 | HOME | B3 | 3 | +3.0 | −0.03 | +0.42 | +0.26 | +0.28 | −0.16 | +0.02 | 0 | veto, pre-empt, open | exit-cfg |
| 08-14 13:57 | ETHFI | B3 | 3 | +2.4 | −0.19 | +0.86 | +0.86 | +0.66 | 0 | −0.20 | 0 | open, pre-empt ×2 | entry timing |
| 08-17 00:33 | WAL | B3 | 3 | −0.4 | −0.06 | +0.23 | +0.23 | +0.38 | 0 | +0.15 | 0 | open ×3 | entry timing (helps) |
| 09-11 22:47 | SAGA | B6 | 3 | +2.2 | −0.27 | +0.95 | +0.95 | +0.71 | 0 | −0.25 | 0 | open, veto ×2 | entry timing |
| 09-16 13:16 | PEPE | B8 | 1 | +19.2 | +0.19 | −0.69 | −0.70 | +0.32 | −0.01 | **+1.02** | 0 | veto ×3 (weak-bounce, EMA13) | gates moved entry (helped) |
| 09-16 13:22 | **ENA** | B8 | 3 | −0.2 | −0.19 | +0.30 | +0.30 | **−0.88** | 0 | **−1.19** | +0.01 | open, open, veto | **same moment, knife-edge stop** (entries 0.10–0.24 % lower) |
| 09-25 04:13 | FET | B12 | 3 | −0.8 | 0 | −0.95 | −0.99 | −0.97 | −0.04 | +0.02 | 0 | open ×3 | match |
| 09-27 22:54 | BR | B13 | 3 | +0.6 | −0.06 | −0.76 | −0.77 | −0.75 | −0.01 | +0.01 | +0.01 | open, open, fan-only | match |
| 09-30 01:07 | ONDO | B15 | 3 | +0.7 | −0.09 | +0.27 | +0.27 | +0.29 | 0 | +0.02 | 0 | open ×3 | match |
| **Mean (26 fills)** | | | | +0.9 median | −0.09 median | **+0.282** | | **+0.024** | **−0.057** | **−0.202** | **+0.002** | | |

Classification summary:
- **Replay bug:** 0 in exits. A small maker-fill entry bias exists (section 2d).
- **Data granularity:** 0.
- **Config (today's gates or exit):** most of the gap. Exit-cfg −0.06; gate-moved entries (STG, TAC, PEPE, SENT, BIO).
- **Legit timing variance:** DEXE 06-24 and ENA. Same-moment entries a few hundredths of a percent apart flip a +0.3 / +0.7 winner into a stop. Section 2d shows live was not systematically lucky.

Entry-effect bootstrap over the 26 fills: mean −0.20, CI −0.41 … −0.005; median +0.003. The mean is carried by 4 knife-edge stops.

Live flips with **no** twin: 23, averaging −0.144 as-lived. At the live moment, today's engine logged:
- **veto** (EMA13 bearish-BTC, weak-bounce, BTC30-rise, LOATR, …) for 13 of them;
- **pre-empt** for 9;
- mixed for 1.

Over all 147 seed-moments:

| At the live entry moment, today's engine… | seed-moments | live avg of those fills |
|---|---|---|
| opened too | 37 | +0.259 |
| vetoed the flip (gates shipped since) | 57 | −0.056 |
| no flip candidate (LONG gate fired before the FAN gate) | 52 | +0.123 |
| FAN block, no flip record | 1 | −0.760 |

Gate counts in the vetoes:

| Gate | Count |
|---|---|
| FLIP_FAN_BTC_EMA13 | 38 |
| FLIP_FAN_WEAK_BOUNCE | 29 |
| BTC30_RISE | 11 |
| QUALITY | 11 |
| RSI_MIN | 11 |
| LOATR | 9 |
| STRETCH | 6 |
| ADXD | 6 |
| BEAR_MIN | 5 |
| other | 8 |

So today's flip vetoes removed a group that was slightly negative live. The "pre-empt" group, longs now stopped *before* the FAN gate, was positive live (+0.12). **Note for whoever adds long-side gates: every LONG gate placed ahead of the FAN gate silently shrinks the flip population.** That is an untested side effect.

### 2d. Entry-timing jitter (was live lucky? is the replay unlucky?)
Each fill is re-entered at its own time + offsets from −60 s to +120 s (10 s steps), at the last traded price, taker fee, with today's exit on real ticks. "Neutral" is the average over offsets, with the fee matched to the fill's actual order type.

| | N | Realized, today's exit | Timing-neutral | Realized − neutral | 95 % CI |
|---|---|---|---|---|---|
| Live | 49 | +0.041 | ≈ +0.06 | −0.02 (maker −0.011, taker −0.030) | −0.077 … +0.061 (taker-fee basis −0.007) |
| yr5 | 411 | −0.137 | −0.082 | −0.055 (maker −0.072, taker −0.029) | −0.071 … −0.007 (taker-fee basis −0.039) |

- **Live was not lucky on timing.**
- The replay's own fills are slightly worse than their neighbourhood, mostly through maker fills: share 61 % vs live 51 %, and adversely selected. Mechanism candidates and the diagnostic are in the patch proposal.
- The offset profile is flat: yr5 ranges −0.07 … −0.12 across all offsets and live +0.02 … +0.11. **No entry delay turns yr5 positive.**

### 2e. How much of the yr5 flip loss survives
| Step | N / seed | avg % / trade | P(avg < 0), window bootstrap (147 windows) | $/seed, fixed $3k book, live cells | $/seed at 1× |
|---|---|---|---|---|---|
| 10-06 review (untrimmed; raw chunk-book $) | 155 | −0.161 | 0.998 | −$9,959 (not comparable) | |
| **Fix 1: trim warm-up duplicates** | 137 | −0.137 | 0.998 (CI −0.24 … −0.03) | −$4,616 | −$2,738 |
| Exit parity check (replica = replay) | 137 | −0.138 | | | |
| Today's exit swapped for no-cap / June exit (for information) | 137 | −0.123 / −0.134 | | | |
| **Fix 2 upper bound: timing-neutral entries** | 137 | **−0.082** (H1 −0.081 · H2 −0.082) | **0.969** (CI −0.17 … +0.004) | **−$2,850** | ≈ −$1.7k |
| Inside live-covered windows only (31 / seed) | 31 | replay −0.063 · neutral −0.008 | | | live, today's exit: +0.041 (49) |
| Outside live-covered windows (106 / seed) | 106 | replay −0.158 · neutral −0.103 | | | |

Per month (timing-neutral avg %):

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| Avg % | −0.11 | −0.07 | −0.09 | −0.03 | −0.18 | −0.10 | −0.12 | **+0.12** | −0.07 |

8 of 9 months are negative; only August is positive.

**Verdict:**
- **Loss surviving parity:** about −0.08 to −0.10 % per trade, about −$2.9k to −$3.4k per seed per year at today's 2× cells, about −$1.6k to −$2.0k at 1×.
- **Confidence:** P ≈ 0.97, which only just clears the 95 % bar once the harness entry bias is removed.
- **What it is:** a weak but consistent loss for today's flip configuration, not a replay artefact. It is also not strong enough on its own to kill the sleeve. The sleeve-kill checklist from the review stands, item ③: uniform degradation.

---

## 3. B — "pair rising while BTC falls"

### 3a. Pre-registration (frozen in the `sig_rs.py` docstring before any cohort statistic was computed)
5m klines, **completed bars only** (≤ 5 min stale, declared). Sources: `k5m_full` + `btc_5m.csv`; the QNT review's REST pulls for Oct-5/6. Coverage: 51 / 51 master, 411 / 411 yr5.

| Name | Rule |
|---|---|
| rs30 | pair 30-min return − BTC 30-min return (pp) |
| rs60 | same over 60 min |
| vr30 / vr60 | quote volume of the last 30 / 60 min ÷ the prior 2 h |
| rec | share of the prior drop recovered (3 h low; high before it) |
| **PRIMARY QNT_RS** | rs30 ≥ +1.0 ∧ BTC30 < 0 ∧ vr30 ≥ 2.0 |
| S1 | rs30 > 0 ∧ BTC30 < 0 |
| S2 | rs60 ≥ 1 ∧ BTC60 < 0 |
| S3 | vr30 ≥ 2 |
| S4 | rec ≥ 0.5 ∧ BTC30 < 0 |
| S5 | 60-min version of the primary |

QNT itself reads rs30 +2.19 · BTC30 −0.34 · **vr30 0.76** · rs60 +0.84 · rec 1.35, so it fails the primary on volume. PUMP (a winner) reads rs30 +1.86 · BTC30 −0.24 · vr30 1.18.

### 3b. Locked FILTER expectancy bar, window-clustered bootstrap, 2,000-shuffle null
Breakeven WR per dataset:

| Dataset | Breakeven WR |
|---|---|
| master as-lived (51) | 62.3 % |
| master today-stack kept (31) | 60.7 % |
| yr5 replay | 72.0 % |
| yr5 timing-neutral | 66.4 % |

| Set | Rule | N | Windows | WR | Avg | Rest | P(avg < 0) | Null p | Bar |
|---|---|---|---|---|---|---|---|---|---|
| master as-lived | QNT_RS | 2 | 2 | 50 % | −0.27 | +0.07 | 0.75 | 0.30 | fail (N) |
| master as-lived | S1 | 43 | 29 | 65 % | +0.06 | +0.05 | 0.30 | 0.52 | fail |
| master as-lived | S2 | 21 | 18 | 67 % | +0.12 | +0.01 | 0.25 | 0.71 | fail |
| master as-lived | S4 | 34 | 26 | 68 % | +0.09 | −0.00 | 0.23 | 0.66 | fail |
| master kept (31) | S1 / S2 / S4 | 28 / 16 / 25 | | 82 / 81 / 80 % | +0.28 / +0.40 / +0.27 | | 0.01 / 0.00 / 0.02 | | fail (positive) |
| yr5 replay | QNT_RS | 28 | 11 | 61 % | −0.12 | −0.14 | 0.77 | 0.55 | fail |
| yr5 replay | S1 | 382 | 138 | 62 % | −0.13 | **−0.19** | 0.99 | **0.67** | "pass", but the rest is worse; it is the whole sleeve |
| yr5 replay | S2 | 186 | 78 | 61 % | −0.17 | −0.11 | 0.99 | 0.20 | "pass"; null not beaten |
| yr5 replay | S4 | 326 | 114 | 61 % | −0.13 | −0.15 | 0.99 | 0.58 | "pass"; rest is worse |
| yr5 replay | S3 / S5 | 38 / 14 | 16 / 6 | 61 / 79 % | −0.14 / +0.08 | | 0.84 / 0.27 | 0.48 / 0.89 | fail |
| yr5 neutral | S1 / S2 / S4 | 382 / 186 / 326 | | 58 / 54 / 57 % | −0.09 / −0.11 / −0.08 | −0.04 / −0.06 / −0.07 | 0.97 / 0.96 / 0.96 | 0.36 / 0.15 / 0.45 | "pass" on the bar, null not beaten |
| yr5 neutral | QNT_RS | 28 | 11 | 61 % | −0.07 | −0.08 | 0.67 | 0.55 | fail |

Walk-forward halves:

| Set | Rule | H1 (master BASE / yr5 Jan–May 14) | H2 (master post-BASE / yr5 May 15–Oct) |
|---|---|---|---|
| master as-lived | S1 | 23 · 83 % · **+0.28** | 20 · 45 % · −0.20 |
| yr5 replay | QNT_RS | 11 · **+0.04** vs −0.17 for the rest | 17 · −0.23 vs −0.10 for the rest |

S1 tracks the era (BASE good, later bad), not the signature. QNT_RS flips sign between halves.

Single variables (sign, then terciles) on master, yr5 and yr5-neutral: rs30, rs60, btc30, btc60, p30, p60, vr30, vr60, rec. None is monotone in the same direction on master and yr5. The closest is the pair's own 30-min move, top tercile > 1.1 %:
- yr5: −0.20 (neutral −0.15) vs −0.11 for the rest; null p 0.10 (neutral 0.06);
- master: 17 · −0.01 vs +0.09 for the rest.

It is post-hoc and does not beat the null. **Not proposed.**

### 3c. Before / after
Master: the current-stack ledger with all sleeves, $ as traded, B17 = the two fresh flips, DCR on a flat $3,000 over active days. Only eras with flips move.

| Era | Before | QNT_RS block | S1 block (≈ sleeve off) | Pair gap > 0.5 block (review's observe item) |
|---|---|---|---|---|
| BASE | +4,223 (4.27 %/d) | +4,223 | +3,578 (4.00) | +4,062 (4.16) |
| B3 | +3,833 (7.10) | +3,777 (7.03) | +3,373 (6.48) | +3,833 |
| B6 | +320 (5.19) | +320 | +105 (3.49) | +320 |
| B8 | +356 (3.81) | +356 | +268 (2.89) | +356 |
| B13 | +302 (4.92) | +302 | +405 (6.53) | +302 |
| B15 | −243 (−4.14) | −68 (−1.13) | −145 (−2.44) | −243 |
| B17 (flips) | −392 | −392 | 0 | +47 |
| **TOTAL (272)** | **+15,422 · 2.32 %/d** | +15,542 · 2.33 | +14,606 · 2.36 | +15,699 · 2.34 |
| Flip sleeve | 31 · 84 % · +0.306 · +$996 | 29 · 86 % · +0.345 | 3 · 100 % · +0.51 | 24 · 88 % · +0.372 |

yr5: trimmed fills, $ per seed on a fixed $3k book at live cells. The "neutral" column is the timing-corrected pnl.

| Month | Before | QNT_RS | S1 | Pair gap > 0.5 |
|---|---|---|---|---|
| Jan | −443 | −559 | −166 | −293 |
| Feb | −941 | −854 | −299 | −59 |
| Mar | −577 | −577 | −19 | −382 |
| Apr | −306 | −306 | −172 | −242 |
| May | −1,087 | −946 | +49 | −183 |
| Jun | −909 | −593 | +124 | −917 |
| Jul | −54 | −77 | +122 | +159 |
| Aug | +220 | +110 | −68 | +521 |
| Sep | −519 | −565 | −58 | −406 |
| **Year** | −4,616 · 137 / seed · 62 % · −0.137 (neutral −0.082 · −$2,850) | −4,367 · 128 · −0.138 (neutral −0.083) | −486 · 10 · −0.191 | −1,802 · 89 · 66 % · −0.072 (neutral −0.041 · −$1,222) |
| **Book, all sleeves $/seed** | **−12,946** | −12,697 | −8,816 | −10,132 |

S1 is not a filter: it removes 93 % of the sleeve, so its row is a "sleeve off" counterfactual. That is out of bounds under the sleeve-kill checklist.

Pair gap > 0.5 is the review's observe candidate, cut point found on yr5. Apply the 30–50 % haircut to its yr5 Δ: +$2.8k per seed raw, about +$1.4–2.0k after the haircut. The master Δ is +$277: BASE −$161, B17 +$439.

---

## 4. Blind spots
- **Gate-vs-data split of the "pre-empt" group.** The 52 pre-empt moments say a LONG-side gate fired before the FAN gate in today's engine. I did not run the engine under June's config, so I cannot say how many are config (gates added later) and how many are indicator or breadth differences. Separating them needs an as-was replay (`--config-history`) over Jun 17 → Jul 8.
- **Maker-fill mechanism not isolated.** The −0.035 to −0.06 pp harness bias is measured, not explained. The patch proposal gives the diagnostic.
- **Timing-neutral is an approximation.** It averages taker entries over −60 … +120 s and adds back the fee advantage for maker fills. It assumes live's timing is centred in that window. Live's own fills sit at −0.02 against the same yardstick, so the "relative to live" correction is ≈ −0.035, and the corrected yr5 range is −0.08 to −0.10.
- **Replica scope.** It covers stop, runner trail, HARD_TP ladder and fees, but not the sub-arm tight trail or EMA exits. Its 99 % agreement says those did not bind on these fills, but they could on other configs.
- **Master N.** 49 as-lived or 31 kept, BASE-heavy. Every B test on the master has fewer than 15 fills in the cohort or a positive average, so the master can only refute here.
- **B features use completed 5m bars only.** QNT's post-entry volume surge (02:12–02:17) is deliberately not visible. A forming-bar version (tick-rebuilt) was not tested.
- **The B17 fills (QNT, PUMP) have no tick archive yet,** so they are not in the replica or jitter work. Their 5m features came from the review's REST pulls.
- **The review's 21-twin figure vs my 26.** The review matched 30 pre-Oct-4 flips against untrimmed yr5. I matched all 49 as-lived master flips against trimmed yr5. Same conclusion.

## 5. Recommendations (nothing armed; operator decisions)
1. **Retire the review's yr5 flip figures** (−0.161 %/trade, −$9,959 per seed, book −$21,935). Restate them as:
   - **−0.137 %/trade (trimmed) / ≈ −0.08 to −0.10 (parity-corrected)**;
   - −$4,616 per seed at live cells on the fixed $3k book (−$2.9k to −$3.4k corrected);
   - book −$12,946 per seed.

   Every future yr5 read uses the window-trimmed loader (Fix 1 in the patch proposal).
2. **The review's actions stand unchanged.** NEGDI15 → 1.0× because its own locked gate fired. TG_SHALLOW → 1.0×. Keep the sleeve-level bar. Corrected yr5 still shows no edge for either 2× cell, and the live record re-priced at today's exit (+0.041) does not support 2× sizing.
3. **Do not demote or disable the sleeve on yr5.** The surviving loss only just clears 95 % (P 0.969) once the harness bias is removed. Inside live-covered windows the replay and live roughly agree (−0.01 vs +0.04). Keep the review's pre-registered sleeve bar: at 1×, the next 15 fresh FAN fills over ≥ 8 windows with WR < 60.7 % ∧ avg < 0 at 95 % → re-run the full checklist.
4. **Replay agent: run the maker-fill diagnostic** in `REPLAY_FLIP_PARITY_PATCH_PROPOSAL_2026-10-06.md` before yr5 is used as kill evidence for any maker-heavy sleeve, not only flips.
5. **Pair EMA13−50 gap > 0.5: keep it OBSERVE-only, threshold frozen.** Its yr5 separation survives the timing correction (neutral −0.157 vs −0.041, null p 0.03). The master is still 8 fills, −0.08. The review's bar and revert gate are unchanged.
6. **QNT relative-strength signature: not registered.**
   - The pre-registered form fails every leg of the test and misses QNT itself.
   - The sign version (S1) is the definition of the sleeve.
   - No continuous cut separates on both datasets.
   - Recorded as tested, refuted.
7. **Review candidate, not a proposal:** add a "flip candidates pre-empted by LONG gates" counter. A journal tally of FAN-eligible signals killed by LONG-side gates placed ahead of the FAN gate would make the side effect visible. Today it is invisible on every surface. That would be an engine logging change, and it needs the operator's go.

## 6. Artifacts (scratchpad `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/flipfu/`)
Scripts:

| Script | What it does |
|---|---|
| `replica.py` | Flip exit replica + tick loader + era configs |
| `yr5load.py` | Window-trimmed yr5 loader |
| `match.py` | Raw twin match |
| `val_live.py` | Replica validation on live + twin decomposition |
| `yr5_exit_variants.py` | Replica vs replay on 411 fills + exit variants |
| `journal_class.py` | What today's engine did at each live moment |
| `jitter.py` | Timing neighbourhood |
| `parity_summary.py` | Corrected yr5 stats, live-covered windows |
| `yr5_dollars.py` | Fixed-book $ via `engine_replay_report.py` outputs (`fills_yr5_s*.csv`) |
| `sig_rs.py` | B pre-registration + features |
| `sig_test.py` | Bar, null, halves, single-variable reads |
| `impact_b.py` | Before/after tables |

CSVs:

| CSV | Contents |
|---|---|
| `val_live.csv` | Replica validation on the live flips |
| `twins_decomp.csv` | Per-twin decomposition |
| `twins_table.csv` | Side-by-side table (section 2c) |
| `journal_class.csv` | Today's engine decision at each live moment |
| `yr5_exit_variants.csv` | yr5 fills under each exit variant |
| `jitter.csv` | Timing-neighbourhood re-entries |
| `yr5_parity_neutral.csv` | Timing-corrected yr5 pnl |
| `rs_master.csv` | RS features, master |
| `rs_yr5.csv` | RS features, yr5 |
| `sig_results.csv` | Bar test results |
| `impact_b_master.csv` | Master before/after |
| `impact_b_yr5.csv` | yr5 before/after |
