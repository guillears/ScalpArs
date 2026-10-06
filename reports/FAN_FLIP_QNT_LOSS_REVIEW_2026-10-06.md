# FAN flip-short review: the QNT loss (Oct 6) + full flip-short sleeve review

Batch: `~/Downloads/scalpars_orders_paper_2026-10-06_02-34-45.csv` (B17, 16 closed since the Oct-3 reset, +$548 as traded).
Validation gate: `scripts/validate_against_master.py` → **ALL CHECKS PASS** (run before any number below).
Research only. Nothing in services/, config, tests or the replay scripts was touched. No commit.

---

## 1. Short answer (plain language)

**Why QNT lost.** The trade went through every gate correctly. It lost for two reasons at once:
1. **QNT was climbing back from a dump, not fading after a pump.** It fell 3.2 % (266.9 → 258.3, 00:30–01:35 UTC). By the time we shorted it at 265.09 it had recovered +2.6 %. It kept climbing on rising volume, up to 269.09 at 02:17. BTC was falling over the same stretch, so QNT was getting stronger than the market.
2. **BTC hit its low right at our entry.** BTC's 5m RSI was 31 and BTC bottomed at 85,406 at 02:15. It then bounced to 85,713 by 02:23. The bounce added to QNT's climb.

The trade was never in profit (peak 0.00 %). It hit its stop after 5.7 minutes. The stop was the ATR-widened one: 1.5 × ATR 0.77 = −1.16 %, so the fill did not slip past the stop. The worst point was −1.51 % at 02:17, and after we were out QNT fell back to −0.72 % below our entry. The **2× NEGDI15 ticket doubled the loss: −$438 instead of −$219.**

PUMP (+0.12 %, +$47) was a normal small winner. It reached +0.61 % and closed early on the runner trail. Five minutes after the exit it was at +0.70 %.

**Watch items.** Several of these tripped on the as-lived count. "As-lived" means the fills that actually traded. "Today's stack" means only the fills that today's filters would still let through.
- **NEGDI15 2× revert gate: FIRED.**
  - Counted the Sep-25 way (from Aug-11): 9 fires · 67 % WR · −$159 at 2×.
  - Counted from the Aug-23 re-arm: 7 · 57 % · −$402.
  - Even keeping only today's-stack fills since the re-arm: Σ −$11, which is still ≤ 0.
  - The locked bar is WR < 70 % or Σ ≤ 0 → back to 1.0×. The cell verdict says ✗ HARMFUL.
  - The only count that does not trip is today's stack counted from Aug-11 (7 · 86 % · +$232). That count drops FET and AVAX using the weak-bounce filter, which was itself built on those two trades, so it is in-sample.
- **FAN bearish-BTC (EMA13) filter revert: FIRED as-lived.**
  - Bar: kept FAN shorts below 60 % WR on N ≥ 10 fresh fills → filter off.
  - As-lived since Aug-23: **10 fills · 40 %** (SAGA, ENA, ONDO and PUMP won; PEPE, FET, AVAX, BR, AAVE and QNT lost).
  - On today's stack it is 7 · 57 %, which is below the N ≥ 10 the bar needs.
  - **But turning the filter off would bring back a losing group.** On the master pool, the fills it blocks are 12 · 50 % · −0.23 %/trade · −$735. The gate measures the wrong thing: it checks the win rate of the trades the filter keeps, not the trades it blocks. Section 6 has my recommendation.
- **TG_SHALLOW 2× cell: ✗ HARMFUL as-lived.**
  - 5 fresh fires · 60 % · −$164. All of the extra loss from the 2× sits in PEPE and AAVE, the two fires with −DI < 15, which is exactly what the 48d watch tracks.
  - On today's stack PEPE is blocked by the weak-bounce filter, leaving N = 4 and +$36, below the bar.
- **Pair-ADX floor revert (pADX ≥ 21 flips ≤ 60 % on N ≥ 15 → floor off): one loser from firing.** QNT brought it to 33 fills · 60.6 % · −$844. The same "wrong thing measured" problem applies.
- **Did not move:** STRONG_BEAR exemption (QNT and PUMP were HEALTHY_BEAR, pair ADX 25.7 and 23.4; still 4/10) · FAN −DI < 15 observe (both had −DI ≥ 15; still 1 fresh, AAVE) · weak-sellers-in-chop (0 fresh) · 48d TG_SHALLOW with −DI < 15 (PUMP had −DI 17.5; still 1/5) · weak-bounce revert (the blocked flips are not in exports; needs the veto log).

**Separators.** The obvious idea from QNT, "don't short when BTC is oversold (5m RSI ≤ 35)", is **refuted**:
- In yr5, results get steadily worse as BTC RSI rises: ≤ 30 = −0.03 %, 30–35 = −0.10 %, 35–40 = −0.18 %, 40–45 = −0.22 %, ≥ 45 = −0.26 %.
- On the master pool, 5 of the 6 flips taken with BTC RSI ≤ 35 won.
- Blocking it makes yr5 worse per trade (the kept trades average −0.206 % against −0.161 % today).

Nothing passes the locked filter bar on both the master and yr5. The best near-miss is "block when the pair's EMA13 is > 0.5 % above its EMA50", i.e. the pair is already in an uptrend. QNT was at +0.82.
- It passes every leg of the bar on yr5: 163 fills · 61 windows · 53 % WR · −0.29 % (P 0.996). It holds in both halves, all 3 seeds and 7 of 9 months, and gets worse as the gap rises.
- The master does not support it: 7 · 71 % · +0.08 %, including five BASE winners at 0.86–0.97.
- That makes it an observe item, not a ship.

**The real finding is about the whole sleeve, not one filter.**
- In the yr5 replay (today's exact config, Jan-4 → Oct-4), FAN flip shorts are **155 per seed · 61 % WR · −0.16 % per trade**. That is negative with P = 0.998 over 151 windows, in both halves, in all 3 seeds and in 8 of 9 months. The only positive month is August, and live August was good too.
- Every cell is negative in the replay. NEGDI15 is no better than plain 1× (−0.156 vs −0.156) and TG + NEGDI is the worst (−0.28).
- The live master (+0.31 %, 84 %) comes almost entirely from BASE (June) and B3 (August).
- Live fills since Aug-23: 10 · 40 % · −0.34 % as-lived, and 7 · 57 % · −0.15 % on today's stack.
- **Caveat:** the replay's flip results are about 0.36 points per trade harsher than live on the same trades. On 21 fills both took, live averaged +0.45 % and the replay +0.09 %, and 4 live winners were stopped out in the replay. The yr5 loss is likely overstated, but the sign holds in every cut.

**Recommendation.**
- **(1) Put NEGDI15 back to 1.0× now.** Its own locked gate fired and yr5 shows no edge for the cell.
- **(2) Put TG_SHALLOW to 1.0× as well (all flips 1×).** Its as-lived verdict is HARMFUL and yr5 shows the shallow BTC-trend-gap zone is the worst one.
- **(3) Keep the EMA13 filter and pair-ADX floor ON as a stated override, and fix both revert gates.** Their gates measure the wrong thing (details in section 6).
- **(4) Register "pair EMA13−50 gap > 0.5" as observe-only**, threshold frozen.
- **(5) Do NOT disable or demote the sleeve.** The sleeve-kill checklist found uniform degradation across cells, and the rule says that points to an unmeasured regime cause or a replay artefact, not the entries. Register a sleeve-level bar instead (section 6).

Before/after for (1) and (2), at live sizing:
| Where | Before | (1) NEGDI15 → 1× | (2) all flips 1× |
|---|---|---|---|
| Master, all sleeves (272 fills) | +$16,231 · +2.38 %/day | +$15,646 · +2.34 %/day (−$585) | +$15,373 · +2.32 %/day (−$858) |
| B17 (this batch, all sleeves) | +$548 | +$767 | +$791 |
| yr5 flips, $ per seed | −$9,959 | −$7,712 (+$2,247) | −$5,893 (+$4,066) |

On the master, the cost of going to 1× is concentrated in BASE: −$576 for (1) and −$830 for (2), because the cells were designed on BASE. So that "cost" is the in-sample edge giving way. yr5's whole book is −$21,935 per seed.

---

## 2. QNT / PUMP anatomy (1m klines, Binance futures public REST)

| | QNT (id 16) | PUMP (id 15) |
|---|---|---|
| Opened / closed | 02:11:25 → 02:17:07 (5.7 min) | 02:06:02 → 02:10:35 (4.6 min) |
| Cell (size) | QS cell (1×, inert) + **[NEGDI15]×2** | QS + [TG_SHALLOW] + [NEGDI15]×2 |
| Close reason | FLIP_STOP_LOSS L1 | FLIP_RUNNER_TRAIL |
| P&L | −1.15 % · **−$438** (−$219 at 1×) | +0.12 % · +$47 (+$23 at 1×) |
| MFE / MAE (engine) | 0.00 / −1.15 (intrabar 1m high = −1.51 % at 02:17) | +0.61 / −0.27 |
| Stop | ATR-widened: −(1.5 × 0.77) = −1.157 %, above the −1.2 floor → the fill did not slip past the stop | — |
| After the exit | +1 min −0.76 · +5 min −0.98 · +15 min −0.07; fell to −0.72 % below entry at 02:33 (in our favour) | +5 min +0.70 (the trail exit gave up about 0.5) |
| −DI / +DI | 16.0 / 28.0 (inside NEGDI15: ≥ 15) | 17.5 / 31.3 |
| Pair RSI · ADX · range pos | 60.5 · 25.7 · 81.5 | 58.6 · 23.4 · 72.8 |
| Pair EMA13−50 gap · EMA50 slope | **+0.82 · +0.37** (uptrend) | −0.19 · −0.24 |
| BTC 5m RSI (prev6) · d-EMA13 · trend gap | 31.1 (37.3) · −0.22 · −0.129 | 35.7 (42.9) · −0.16 · −0.092 (TG_SHALLOW) |
| Bear breadth · regime · BTC off-24h-low | 80.0 · HEALTHY_BEAR · +0.63 % | 77.8 · HEALTHY_BEAR · +0.84 % |

Timeline (UTC):
- 00:30–01:35: QNT 266.85 → 258.32 (−3.2 %).
- 01:35–02:11: V-shaped recovery to 265 while BTC slid 85,940 → 85,423. QNT was outperforming BTC.
- 02:11: short at 265.09. The 02:12–02:17 1m bars climbed on 2–4× normal volume (02:15 volume 11,620 on 5m).
- 02:15: BTC 5m low 85,406 at RSI 31.
- 02:16–02:23: BTC bounced to 85,713. The QNT stop was taken at 02:17:07 (267.97). QNT reached 269.09 at 02:17 and again 269.00 at 02:25, then slid to 263.18 by 02:33.

In one sentence: we shorted a pair that was rebuilding after a flush and was stronger than BTC, at the moment BTC bottomed out oversold.

Can a wider stop be checked? Not from one fill. The locked rule (DECISION_LOG 110) says exit and stop counterfactuals are read only on the live-stopped group, two-sided. Earlier flip stop-grid studies (Jul-07, Jun-29) rejected both tighter and wider stops. Not proposed here.

Data note: `low_price_since_entry` for QNT is 258.32, which is a price from before entry (01:35). The field looks seeded from a pre-entry window on shorts. It is a logging quirk, not a P&L error.

---

## 3. Watchlist tally (every open flip / FAN / NEGDI15 / TG_SHALLOW gate)

Live union = every `~/Downloads/scalpars_orders_paper_*.csv` + `reports/BASELINE*_orders*.csv` + `MASTER_POOL_stacked.csv`, deduped on opened_at + pair, CLOSED only. No flip probes exist.

| Gate (source) | Bar | Before this batch | With QNT + PUMP | Status |
|---|---|---|---|---|
| **NEGDI15 2× demux** (DL 27, re-read rule DL 113) | N ≥ 5 fresh ×2 fires under EMA13: WR < 70 % ∨ Σ ≤ 0 → 1.0×; cell verdict HARMFUL = Σ < 0 on N ≥ 5 | 7 · 71 % · +$232 (DL-113 counting incl. ETHFI/WAL) | **9 · 67 % · −$159** (1× Δ −$79); since re-arm 7 · 57 % · −$402; today-stack since re-arm 5 · 80 % · **−$11** | **🔴 FIRED** (every as-lived count; today-stack since re-arm too) |
| **FAN EMA13 filter revert** (DL 21) | kept FAN shorts < 60 % WR on N ≥ 10 fresh → off | 8 · 38 % | **10 · 40 % · −$881** as traded | **🔴 FIRED as-lived** · today-stack 7 · 57 % · −$289 (N < 10). Instrument flawed: blocked cohort (master) 12 · 50 % · −0.23 % · −$735 (P 0.87, 10 windows) |
| **TG_SHALLOW 2× cell** (Jul-08 ship) | HARMFUL: net-negative on N ≥ 5 fresh → 1.0× | 4 · 50 % · −$211 | **5 · 60 % · −$164** (2× increment −$188, all PEPE + AAVE) | **🔴 HARMFUL as-lived** · today-stack 4 · +$36 (N < 5) |
| 48d TG_SHALLOW ∧ −DI < 15 → 1× (DL 68/69) | N ≥ 5 fresh: WR < 70 % ∨ Σ ≤ 0 | 1/5 (AAVE −$176) | 1/5 (PUMP −DI 17.5 = not a fire) | unchanged |
| Pair-ADX floor 21 revert (Jun-23) | pADX ≥ 21 flips ≤ 60 % WR on N ≥ 15 → 0 | 32 · 62.5 % | **33 · 60.6 % · −$844** | ⚠ one loser from firing |
| STRONG_BEAR pADX < 21 exemption (DL 113) | F1-admitted, net-negative at N ≥ 10 → clear | 4/10 · +$254 | 4/10 (both HEALTHY_BEAR, pADX ≥ 21) | unchanged |
| Weak-bounce revert (DL 113) | first 8 sole-blocked re-priced, WR ≥ 60 % ∨ Σ > 0 → off | 0/8 | not visible in exports (needs the [FLIP_FILTER] veto log) | unknown |
| FAN −DI < 15 observe (DL 120) | fresh N ≥ 15 / ≥ 8 windows, expectancy bar | 1 (AAVE, L) | 1 | unchanged |
| Weak-sellers-in-chop (DL 166) | fresh after 10-01 14:06, N ≥ 15 / 8 days… | 0 | 0 | unchanged |
| QS winner cell (1×, track) | re-2× only on N ≥ 30 forward | — | +PUMP (W), +QNT (L) | track |
| "bear 70–80" multiplier candidate (Jun-28) | forward ≥ 80 % WR on N ≥ 30 | — | PUMP 77.8 (W); QNT 80.0 sits exactly on the 70–80 / ≥ 80 boundary | track |
| GREEN-monitor flip observe | ≥ 5 fires / ≥ 2 GREEN episodes ≤ 40 % | 1 (TRB) | not stamped on the orders CSV | unknown |

Breakeven WR for the flip sleeve, from today's kept fills (31 incl. QNT and PUMP): avg win +0.519 / avg loss −0.803 → **60.7 %**. It was pinned at 57 % on 29 fills; QNT moves it up.

---

## 4. Full flip-short sleeve review

All flips in every dataset are `FLIP:FAN_RATIO_GATE` shorts. The other flip sources are off.

### 4a. Live / recent batches
| Cohort | N | WR | avg % (1×) | $ as traded | $ at 1× | never positive | MFE med / MAE med |
|---|---|---|---|---|---|---|---|
| Since Aug-23 (EMA13 filter + NEGDI15 re-arm), as-lived | 10 | 40 % | −0.336 | −$881 | −$492 | 4 | +0.15 / −0.70 |
| Same, today's stack (minus PEPE/FET/AVAX weak-bounce) | 7 | 57 % | −0.146 | −$289 | −$196 | 3 | +0.43 / −0.27 |
| Since Sep-25 (last flip change: weak-bounce) | 5 | 40 % | −0.456 | −$593 | −$347 | 3 | 0.00 / −0.76 |

By cell (since Aug-23, as-lived):
- NEGDI15 only: SAGA +$215, FET −$225, AVAX −$166, QNT −$438.
- TG + NEGDI: ENA +$88, ONDO +$77, PUMP +$47 (3/3 wins).
- TG only: PEPE −$201, AAVE −$176 (0/2).
- Plain 1×: BR −$102.

By close reason: every loser is FLIP_STOP_LOSS L1 (6/6) and every winner is FLIP_RUNNER_TRAIL.

### 4b. Master pool (current-stack ledger, 29 kept) + B17 fresh (2) = 31
| Cut | N | WR | avg % | $ as traded | $ at 1× | $ at today's cells |
|---|---|---|---|---|---|---|
| All | 31 | 84 % | +0.306 | +$996 | +$948 | +$1,805 |
| NEGDI15 only | 13 | 92 % | +0.505 | | +$586 | |
| TG + NEGDI | 7 | 100 % | +0.327 | | +$282 | |
| TG only | 2 | 50 % | −0.037 | | −$10 | |
| Plain 1× | 9 | 67 % | +0.078 | | +$90 | |
| RUNNER_TRAIL / STOP_LOSS | 27 / 4 | 96 % / 0 % | +0.495 / −0.971 | | | |
| By era: BASE 20 · 90 % · +0.386 · B3 4 · 100 % · +0.695 · B6 1 W · B8 1 W · B13 1 L · B15 1W/1L · B17 1W/1L | | | | | | |
| Washed-out window (Jun-18 → Jul-2) | 14 | 93 % | +0.478 | | | |
| Rest | 17 | 77 % | +0.164 | | | |

Never-positive share 13 % (all 4 were losers). MFE median +0.83, MAE median −0.49; the losers' MFE median is 0.00.

### 4c. yr5 engine replay
Source: `reports/backtest_cache/replay/yr5_<chunk>_s{1,2,3}_orders.csv`, 57 runs, Jan-4 → Oct-4. Config is HEAD 181131e frozen (`frozen_config_yr5_181131e.json`), run with the fixed harness and real ticks. 466 FAN shorts (162 / 137 / 167 per seed).

| Cut | N (3 seeds) | WR | avg % 1× | $ per seed at live cells |
|---|---|---|---|---|
| YEAR | 466 | 60.5 % | **−0.161** (window bootstrap P(avg < 0) = 0.998, CI [−0.28, −0.05], 151 windows; every seed P ≥ 0.993) | −$9,959 |
| H1 Jan–Apr / H2 May–Oct | 200 / 266 | 59.5 / 61.3 % | −0.194 / −0.136 | |
| Cells: NEGDI15 / TG + NEGDI / TG / plain | 183 / 58 / 44 / 181 | 59 / 55 / 68 / 62 % | −0.156 / −0.277 / −0.051 / −0.156 | |
| RUNNER_TRAIL / STOP_LOSS / TP ladder | 274 / 184 / 8 | 100 / 0 / 100 % | +0.362 / −0.991 / +1.018 | |
| Never positive | 15.7 % (all losers) | | | |

MFE median +0.42, MAE median −0.60; losers' MFE median +0.09. Avg win +0.38 vs avg loss −0.99, so the replay breakeven WR is 72 %.

Per month (avg % 1×, fills per seed):

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| Avg % | −0.11 | −0.28 | −0.31 | −0.05 | −0.25 | −0.19 | −0.08 | **+0.14** | −0.15 |
| Fills/seed | 14 | 19 | 17 | 16 | 21 | 22 | 16 | 11 | 19 |

**How far to trust the replay for flips.**
- Of 30 live flips before Oct-4, 21 have a replay twin (same pair, ±30 min). On those, live was +0.453 % / 95 % WR and the replay +0.092 % / 79 %, with the same outcome 81 % of the time. Four live winners (STG, DEXE 06-24, ENA, TAC s3) were stopped out in the replay. The replay entries land 0–12 minutes away from the live ones, and fade shorts are sensitive to that timing. → **the yr5 flip loss is probably overstated by up to ~0.3 points per trade.**
- Inside the live-covered periods, yr5 also takes **96 fills (≈ 32 per seed) that live never took**, at 62 % · −0.17 %. These include B1/B2 (Jul-11 → Aug-10, the flip-starvation weeks: 14 per seed at −0.18 / −0.53 %) and 4 B12 losers. So today's config admits more flips than the config that produced the good live record, and the extra ones lose.

---

## 5. Separator search (filter expectancy bar, window units, null)

The bar applied: WR below the sleeve breakeven (60.7 %) · avg < 0 at 95 % by window-clustered bootstrap · ≥ 8 windows · no window or pair carrying ≥ 50 % of the loss · N ≥ 15. Windows = fills chained within 60 min; on yr5 the window timeline is the union of the 3 seeds, so the same moment across seeds counts once. Per-pair concentration was checked first: no yr5 cohort has a pair above 38 % of its loss.

**Named candidates** (full table: scratchpad `named_candidates.csv`):
| Candidate | Master + fresh | yr5 | Verdict |
|---|---|---|---|
| BTC 5m RSI ≤ 35 (QNT idea) | 6 · 83 % · +0.23 | 166 · 68 % · −0.08 vs rest −0.21 (**better than rest**) | ✗ refuted (wrong sign; monotone the other way in yr5) |
| BTC RSI ≤ 35 ∧ ticking up (QNT exact) | 1 (QNT) | 40 · 65 % · −0.15 vs −0.16; H1 −0.33 / H2 +0.04 | ✗ noise, flips sign |
| BTC RSI falling vs prev6 | 31 / 31 (all) | 465 / 466 (all) | n/a — the gate stack already requires it |
| Bear breadth ≥ 80 / ≥ 76 | 5 · 60 % · +0.06 / 9 · 56 % · −0.04 | 91 · −0.07 / 135 · −0.06 (both **better** than rest) | ✗ refuted in yr5 (H1 only) |
| Range position ≥ 80 | 9 · 67 % · +0.03 | 154 · 58 % · −0.18 vs rest −0.15 | ✗ no separation |
| −DI 15–17 (QNT's band) | 13 · 92 % · +0.45 | 150 · 58 % · −0.19 vs −0.15 | ✗ master opposite |
| −DI < 15 | 11 · 64 % · +0.06 | 225 · 63 % · −0.14 vs −0.19 | ✗ (still observe, DL 120) |
| BTC off-24h-low ≤ 1 % | 5 · 40 % · −0.46 | 320 · 62 % · −0.13 vs −0.24 (better than rest) | ✗ refuted |

**Exhaustive 1D sweep** (`sweep.py`): 38 stamped entry columns + BTC-RSI Δ1/Δ6 + DI spread, cut at sign, median and outer-tercile, both sides. That is 166 tests across yr5 H1, yr5 H2 and the master.
- Block-side cuts that point the same way in all three: **23**. The shuffled-label null (60 shuffles) gives a median of 19.5 and a p95 of 28. **That is within chance.**
- Keep-side cuts positive in all three: 1 (pair ATR < 0.535, +0.02 %, break-even). No edge there.

**Exhaustive 2D sweep** (`sweep2d.py`): every pair of tercile tails (~3,000 pairs), discovered on yr5 H1, confirmed on H2 and the master.
- The best H1 Δ is −0.85, beyond the null p5 of −0.63.
- But the top 20 confirm out-of-sample on H2 only **45 %** of the time; the null gives 50 % median, 75 % p95. **It does not survive out-of-sample.**
- The one recurring cluster is "BTC RSI-prev6 high", i.e. BTC RSI had just dropped sharply. It holds in H2 but the master is the other way (+0.2..+0.5 on 3–7 fills).

**Regime axes** (sleeve-kill ②, read in windows):
- Regime: BEAR_EXHAUSTED 13 · 5 windows · +0.23 · HEALTHY_BEAR 242 · −0.18 · STRONG_BEAR 171 · −0.13 · CHOPPY_FLAT 38 · −0.38.
- BTC 1h slope: always ≤ 0 (gated).
- BTC trend gap, monotone in yr5: ≤ −0.2 +0.07 · −0.2..−0.1 −0.12 · −0.1..0 −0.18 · > 0 −0.36. **This runs against the TG_SHALLOW 2× cell.** The master is flat across these buckets.
- BTC off-30d-high: near the high (−4..0 %) −0.04 is the best bucket.
- eff72: no clean shape.
- BTC 1h RSI: ≤ 35 is the best bucket (−0.04).

**Best near-miss: pair EMA13−50 gap > 0.5** (column `entry_pair_ema20_ema50_gap_pct`, which actually holds EMA13−50):
| | N | windows | WR | avg | rest | P(avg < 0) | bar |
|---|---|---|---|---|---|---|---|
| yr5 | 163 | 61 | 52.8 % | −0.294 | −0.089 | 0.996 | **PASS** (top window 9 %, top pair 10 %) |
| yr5 H1 / H2 | 73 / 90 | 23 / 38 | 52 / 53 % | −0.30 / −0.29 | −0.13 / −0.06 | 0.95 / 0.99 | pass / pass |
| per seed | 61 / 47 / 55 | | | −0.27 / −0.30 / −0.32 | −0.11 / −0.10 / −0.06 | | consistent |
| **master + fresh** | 7 | 7 | 71 % | +0.081 | +0.372 | 0.39 | **fail; does not support** (BIO +1.23, ONDO +0.63, INJ +0.31, POWR +0.23, CHZ +0.54 vs DEXE −1.21, QNT −1.15) |

yr5 gradient by gap bucket: (0, 0.25] +0.02 · (0.25, 0.5] −0.11 · (0.5, 0.75] −0.21 · (0.75, 1.0] −0.44 / 43 % WR. The existing ceiling at ≥ 1.0 already blocks the top. It is worse than the rest in 7 of 9 months.

Theory: a fade sells a pump. A pair whose EMA13 is well above its EMA50 is in an uptrend, and its "pump" is the trend continuing. That fits QNT. But the live master has five winning BASE flips in exactly this zone, and the 1D family count is within chance. **→ OBSERVE ONLY, threshold frozen at 0.5** (see section 6). It does not pass the locked bar on both datasets.

---

## 6. Before / after impact

Master = the current-stack ledger (all sleeves) with flips repriced at today's cell multipliers, plus B17 (this export's 2 flips). DCR = flat $3,000 over active days (the ledger convention). The 1× $ = $ as traded ÷ cell multiplier as traded.

| Era | Before (NEGDI15 2× + TG 2×) | A: NEGDI15 → 1× | B: all flips 1× | C: block BTC RSI ≤ 35 (refuted) | D: block BTC trend gap > −0.119 (master-refuted) | E: block pair gap > 0.5 (near-miss) |
|---|---|---|---|---|---|---|
| BASE | +5,033 (4.80 %/d) | +4,457 (4.43) | +4,203 (4.26) | +4,578 | +4,106 | +4,647 |
| B3 | +3,833 (7.10) | +3,711 (6.94) | +3,711 | +3,833 | +3,777 | +3,833 |
| B6 | +320 (5.19) | +212 (3.48) | +212 | +320 | +320 | +320 |
| B8 | +356 (3.81) | +356 | +312 (3.35) | +356 | +268 | +356 |
| B13 | +302 | +302 | +302 | +302 | +302 | +302 |
| B15 | −243 (−4.14) | −243 | −194 (−3.29) | −243 | −145 | −243 |
| B17 (flips only) | −392 | −172 | −196 | +47 | −438 | +47 |
| B17 whole batch (all sleeves) | +548 (4.28 %/d) | +767 (5.86) | +791 (6.02) | — | — | +987 (7.37) |
| **TOTAL (272 fills)** | **+16,231 · 82 % · 2.38 %/d** | +15,646 · 2.34 | +15,373 · 2.32 | +16,215 · 2.38 | +15,212 · 2.31 | +16,284 · 2.38 |
| Flip sleeve | 31 · 84 % · +0.306 · +$1,805 | +$1,220 | +$948 | 25 · 84 % · +0.323 · +$1,789 | 16 · 81 % · +0.307 · +$786 | 24 · 88 % · +0.372 · +$1,858 |

Other eras are unchanged by every scenario. Master changes from the 1× scenarios are **in-sample**: the NEGDI15 cell was built on those 17 BASE trades.

yr5 ($ per seed at each scenario's sizing; per-trade avg is 1× / sized points):

| Month | Before | A NEGDI15 1× | B all 1× | C RSI ≤ 35 block | D TG > −0.119 block | E pgap > 0.5 block |
|---|---|---|---|---|---|---|
| Jan | −646 | −622 | −408 | −903 | +442 | −455 |
| Feb | −1,828 | −1,083 | −1,079 | −1,818 | −441 | −901 |
| Mar | −1,214 | −812 | −877 | −164 | −980 | −995 |
| Apr | −172 | −296 | −53 | −616 | +156 | −114 |
| May | −2,277 | −1,482 | −1,342 | −1,427 | −439 | −130 |
| Jun | −2,645 | −1,588 | −1,478 | −2,970 | −1,028 | −2,529 |
| Jul | −979 | −1,215 | −430 | −1,037 | −4 | +143 |
| Aug | +639 | +210 | +379 | +562 | +929 | +1,368 |
| Sep | −838 | −824 | −604 | −618 | −302 | −673 |
| **YEAR** | **−9,959** · 155 fills/seed · 61 % · −0.161 / −0.262 pts | −7,712 · −0.161 / −0.200 | −5,893 · −0.161 / −0.161 | −8,991 · 100 fills · 57 % · **−0.206** | −1,667 · 79 fills · 66 % · −0.062 | −4,286 · 101 fills · 65 % · −0.089 |
| yr5 whole book (all sleeves) | −$21,935 per seed | −$19,688 | −$17,869 | −$20,967 | −$13,643 | −$16,262 |

Apply the 30–50 % in-sample haircut to any yr5 Δ for D and E: both cut points were found on yr5.

### NEGDI15 2× sizing question on its own
| | 2× (now) | 1× | Δ |
|---|---|---|---|
| Master (13 NEGDI15-only + 7 TG + NEGDI) | | | −$576 BASE, −$108 B6, −$122 B3, +$220 B17 → **−$585** (in-sample) |
| Fresh fires (DL-113 count, 9) | −$159 | −$79 | **+$79** |
| Fresh since re-arm (7) | −$402 | −$201 | **+$201** |
| yr5 | −$9,959 per seed | −$7,712 | **+$2,247 per seed** (cell avg −0.156 = plain −0.156; TG + NEGDI −0.277) |

### Recommendations (pre-registered; nothing armed by me)
1. **NEGDI15 → 1.0× (`flip_short_negdi_mult` 2.0 → 1.0).** Its locked demux gate fired. This is not a judgment call. Re-arm only through the standard multiplier gate: N ≥ 30 fresh 1× fires · WR ≥ 70 % · avg ≥ +0.10 · Σ > 0, staged 1.5× first.
2. **TG_SHALLOW → 1.0× (`flip_short_tg_shallow_mult` 2.0 → 1.0).** Its as-lived verdict is HARMFUL (5 · −$164), its whole 2× increment sits in the −DI < 15 sub-cell, and yr5 shows its BTC-trend-gap zone is the sleeve's worst. Same re-arm gate as (1). Once both are at 1×, the 48d watch is moot; retire it.
3. **EMA13 filter and pair-ADX floor revert gates.** The EMA13 one fired as-lived, and the ADX one is one loser away. Both measure the kept group, whose problem is sleeve-wide. Turning the EMA13 filter off brings back a group that loses on the master (12 · 50 % · −0.23 % · −$735). **Recommendation: a declared operator override.** Keep both filters, and **re-specify both reverts on the BLOCKED group**: filter off if the first 10 blocked FAN signals (veto-log price + 1m klines, flip exit) reach WR ≥ 60 % ∧ Σ > 0. This needs the operator's explicit sign-off, because it changes a pre-committed gate after it fired.
4. **Register OBSERVE: FAN flip with pair EMA13−50 gap > 0.5** (threshold frozen; `entry_pair_ema20_ema50_gap_pct` is already stamped on every fill).
   - Bar to arm: fresh fills after 2026-10-06 02:35 UTC, 1×, N ≥ 15 over ≥ 8 windows, WR < 60.7 % ∧ avg < 0 at 95 % by window bootstrap ∧ no window or pair ≥ 50 % of the loss.
   - Then a 30–50 % haircut, and revert if the first 8 blocked signals re-price to WR ≥ 60 % ∨ Σ > 0.
   - Expected Δ if it ever ships: master +$53 (in-sample; BASE −$386), yr5 +$5.7k per seed before the haircut, ≈ +$2.8–4.0k after.
5. **Sleeve-level: no disable or demote proposal** (checklist below). Register a **sleeve bar**: at 1×, the next 15 fresh FAN fills over ≥ 8 windows with WR < 60.7 % ∧ avg < 0 at 95 % → re-run the full checklist, with a fixed replay as the counterfactual, and only then consider demotion.
6. **Fix the replay's flip fidelity first.** The paired gap is 0.36 points per trade on 21 matched fills, with entries landing 0–12 minutes apart. Do it before yr5's flip number is used as kill evidence.

**Sleeve-kill checklist (run, for the record):**
- ① Pair-level dimensions: done (1D 166 tests + exhaustive 2D, shuffled null). Nothing beyond chance.
- ② Macro/regime at sign granularity, then buckets, in windows: done (table above). The only consistent shapes are the BTC trend gap and BTC RSI gradients in yr5, and the master does not confirm either.
- ③ Uniform degradation: **YES.** Every cell is negative in yr5 and degrades together in live fresh fills. Per the rule, the cause is unmeasured regime or replay, not the entries.
- ④ Tape context: live wins sit in the Jun-18 → Jul-2 washed-out window (BTC 15–24 % below its 30d high; master 14 · 93 %) and in August. yr5 agrees on August (+0.14) but not on June (−0.19; calibration gap).
- **Verdict: no grounds to disable. Size at 1× is the in-rules risk control.**

---

## 7. Blind spots
- The **weak-bounce revert** and any blocked-signal re-pricing need the `[FLIP_FILTER]` veto log; exports carry only taken fills.
- **GREEN-monitor state** is not stamped on order rows, so the GREEN flip observe item could not be tallied.
- The **yr5 flip fidelity** is not validated against the master: 21/30 matched, +0.36 points per trade harsher, entries 0–12 minutes off. yr5 numbers are a direction signal, not a calibrated level. The 96 extra replay flips in live-covered periods could be partly harness timing, not config.
- **Master N is tiny (31) and BASE-heavy** (20 of 31, 14 in the washed-out window). All master-side Δs are in-sample for the cells (NEGDI15 and TG were built on BASE).
- The **"as-lived vs today's-stack" counting** changes which revert gates fire. Both are reported. The convention needs an operator ruling: DL 113 used today's-stack counting for the STRONG_BEAR exemption and as-lived counting for NEGDI15.
- The **relative-strength signature** ("pair rising while BTC falls in the 30 min before entry") is not stamped. It was seen on QNT but not testable without rebuilding from klines for every fill. A candidate for the next deep dive, rebuilt on every fill, never a stamped subset.
- **Stop-width counterfactual** for QNT is not computed (one fill; locked two-sided, live-stopped-only rule).
- **1D shuffle null** permutes P&L at the fill level, which is slightly anti-conservative. The 2D null's out-of-sample rate is the binding check.
- yr5 ends Oct-4, so QNT and PUMP have no replay twin.

Artifacts (scratchpad, not repo): `live_flip_shorts.csv`, `yr5_flips_tagged.csv`, `named_candidates.csv`, `sweep1d.csv`, `sweep2d_top20.csv`, `impact_master.csv`, `impact_yr5.csv`, scripts `common.py`, `sweep.py`, `sweep2d.py`, `impact.py`, `impact_yr5.py` under `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/`.
