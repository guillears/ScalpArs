# 🪜 FRENZY staircase study: ORCA / US / API3 / PARTI / RLC, the year, and three candidate mechanisms (2026-10-06)

Research only. No bot, config, test or template file was touched, nothing was committed, and no scout state was written.

- **Scripts** (scratchpad `$S/stair/`, `$S` = this session's scratchpad):
  - `walk_live.py` / `cases.py`: the live cases, run on public klines.
  - `build50.py`: the engine-exact walk at the staircase level.
  - `part2.py` / `part2b.py`: the class study.
  - `px.py` / `part3a.py` / `priceB.py` / `part3_report.py`: the mechanisms.
- **Pre-registration:** every rule below was frozen in those docstrings before its numbers were computed.
- **Data written to reports/:**
  - `FRENZY_STAIRCASE_EPISODES_2026-10-06.csv`: 946 alerted episodes.
  - `FRENZY_STAIRCASE_TRIGGERS_2026-10-06.csv`: 600 (a) triggers.

## Answer in one paragraph

**The scout's staircase is the FRENZY setup at half the volume bar.** The scout looks for 50× normal volume; FRENZY needs 100×. Otherwise the rule is identical: at least 2 h after the spike, and every close of the last hour at or above the spike VWAP.

The misses have three causes:
- **ORCA, US and PARTI never reached FRENZY's 100× volume.** ORCA touched it on exactly one bar, and that is the bar FRENZY bought.
- **API3's one chance was refused by the market-volume gate.**
- **RLC's one chance was taken.** After that the setup stayed ON for 25 h with no new entry bar.

**Over the year, buying staircases does not pay:**
- From the moment the staircase alert fires, 69 % of the 946 alerted episodes fall 5 % before they gain 10 %.
- 95 % close a 5m bar below the spike VWAP within 24 h.
- The median 24 h return from the alert is −7.4 %.

The giant staircases (RLC, ORCA, SAND, MOVR) are real, but **they are only visible in hindsight.**

**All three mechanisms fail:**
- (a) "enter when the staircase turns on": −0.38 to −0.49 %/trade, with the day-CI entirely below 0.
- (b) VWAP-anchored stop: on the live-stopped fills, the trades it saves and the trades it makes worse cancel out (+358 vs −378 points).
- (c) "exit when the staircase breaks": a higher mean, but a 20 % win rate, negative without its top 5 trades, and a book of +106 % against the lock's +314 %.

**Recommendation: no change to the bot.** One cheap observe-only line (§5) answers whether the VWAP stop would help FRENZY fills going forward.

---

## 0. Parity (done first)

| check | result |
|---|---|
| Staircase walk = the real `services.frenzy.frenzy_walk`, only `frenzy_state_vol_mult = 50`, on every bar ≥ 50× (the `frenzy_engine_cohort_build.py` construction) | engine code, not a re-implementation. 89,905 flagged state bars, 589 pairs, Jan 10 – Sep 27 |
| Its 100× read vs the engine cohort's own 100× bars (`mh/bars_mh.pkl`, min-hours 2) | 48,514 of the 52,989 100× state bars it sees are in the engine set. The gap is pairs the old build skipped (spot-checked: real `frenzy_walk` flags them too) |
| My exit pricer `px.exits` LOCK2 vs the cohort's LOCK2 (same tick prints) | max \|Δ\| = 2e-15 on 1,672 fills |
| Cohort `STAIR` column | **differs from its docstring.** Its −3 stop only applies before the peak reaches +3, so after a +3 peak a trade can ride down to −17. I report both versions: `STAIR_c` (cohort) and `STAIR` (−3 always) |
| Live ORCA 10-06 09:40 signal bar | live stamp 100.4× / VWAP 2.44558. My replica: 99.9× / VWAP 2.44558, i.e. a 0.5 % normal-volume drift (the live normal-hour cache is up to 8 h old). Same bar, same VWAP |

## 1. The five live cases (public klines, real engine functions, live thresholds)

### 1a. What the engine saw

| pair | scout said | FRENZY (100×) setup | every bar FRENZY / WIDE could enter | what happened and why |
|---|---|---|---|---|
| **ORCA** | +37 % over 37 h since the 10-04 18:30 spike, 12.8 % above VWAP 2.233, 58× | **Never ON in the 10-04 episode.** It peaked at 51–60× (`FRENZY_VOL_FADED`). At 100× that episode ended 24 h after its spike. A new 100× episode was anchored at the 10-05 22:20 spike, with VWAP about 2.44 | **One bar: 10-06 09:40.** Volume crossed 100× for that one bar (100.4× live, 99.9× replica). It was red, ATR 1.74 → `FRENZY_READY` | FRENZY_LONG bought 2.61 at 09:40:08 and stopped −3.00 % at 10:04 (2.534). The low after the stop was 2.506 (−4.0 %). The price then made 2.717 (+4.1 %) at 11:30 and is now 2.554 |
| **US** | +40 % since 10-05 16:45, 63× | **Never ON.** Volume peaked at about 54× (`FRENZY_VOL_FADED`) | none | No journal row. The journal logs fresh bars only, and US never had one |
| **API3** | +19 % since 10-05 16:30, 112× | ON at 10-06 06:35 (fresh, 103×, red, ATR 1.80 → READY) | 06:35 | **Refused by `FRENZY_GVOL_HIGH`** (journal 06:35:00), the market-volume gate, `frenzy_gvol_max` 1.0 |
| **PARTI** | +15 % since 10-05 07:20, 57× | **Never ON.** Volume peaked at about 61× (`FRENZY_VOL_FADED`) | none | No journal row |
| **RLC 10-05** | +43 % at 6.5 % above VWAP, 928× | ON at 12:00 (fresh, green, ATR 2.38) → `FRENZY_GREEN_BAR` → WIDE | **12:00 only.** The setup then stayed ON continuously to 10-06 13:35 (25 h+), so there was no second fresh bar | WIDE bought 0.4649 and took +3.0 % at 12:03 (the fixed TP at that time). Under today's lock (the replica, on ticks) it is +5.3. RLC then peaked at +132 % from the entry |

**Why ORCA stopped inside a staircase.** The staircase was built at about 50× volume, so FRENZY was off the whole time. FRENZY's only ON bar was the 100× crossing at 09:40, which came at the top of a 2-hour push (2.50 → 2.66). The normal pullback after that push went 4.0 % deep.
- The lowest price, 2.506, was still **+2.5 % above the engine's VWAP** (2.446) and **+9.4 % above the scout's VWAP** (2.29).
- The −3 % stop sits **inside** the staircase's normal breathing room. Here it was 0.6 % under a 5m-ATR-sized dip (ATR 1.74 %).

### 1b. Run-up from each possible entry, and what each exit got (1m public klines, entry = the 1m open after the signal close, fees 0.09 + slip 0.10, 12 h cap)

| case | entry | best run-up after entry | dip before that peak | live lock | (b) VWAP stop, k 0 / 0.5 | (c) staircase exit |
|---|---|---|---|---|---|---|
| ORCA 10-06 09:40 (the live fill) | 2.613 | +5.1 % (11:30) | −4.1 % | **−3.1** (10:04) | **+2.9 / +2.9** (lock armed at 11:30) | −3.1 |
| ORCA 10-05 00:50 (first 50× bar) | 2.087 | +31.6 % (10-06 11:30) | −6.7 % | −3.1 | −3.7 / −5.2 (VWAP broken 03:30) | −3.1 |
| ORCA 10-06 05:40 (first 50× bar after 32 h) | 2.461 | +11.6 % | −2.4 % | +1.9 | +1.9 / +1.9 | +3.2 (cap) |
| US 10-06 08:25 (first 50× bar) | 0.012811 | +20.1 % (12:09) | −5.2 % | −3.1 | +2.8 / +2.8 | −3.1 |
| API3 10-06 06:35 (READY, refused by GVOL) | 0.3457 | +14.3 % (08:45) | −2.5 % | **+3.9** | +3.9 / +3.9 | +3.4 |
| PARTI 10-06 03:10 (first 50× bar) | 0.03194 | +5.8 % | −2.1 % | +1.9 | +1.9 / +1.9 | +3.4 (cap) |
| RLC 10-05 12:00 (the live WIDE fill) | 0.4649 | +131.9 % (10-06 11:05) | −0.6 % | +5.3 (live took +3.0, fixed TP) | +5.3 / +5.3 | **+55.9** (12 h cap) |

Even inside these hand-picked winners, the alternative exits help on some and hurt on others:
- The VWAP stop rescues ORCA's live fill and US, and does worse on ORCA's early bar.
- The staircase exit only pays on RLC, the one that went straight up.

## 2. The year: staircases as a class (Jan 10 – Sep 27 2026)

**Definition (frozen, the scout's own rule, causal):**
- An **episode** is a 50× spike episode of the engine walk with at least one flagged state bar. Blacklisted pairs are out.
- The **alert** is the first state bar with 24 h volume ≥ $20M and 24 h change ≥ +15 % (the scout's shortlist), read at that bar's close.
- **ST32** is the first state bar at least 32 h after the spike (the scout's ⏳ note).
- **Fills** are FRENZY_ENGINE_COHORT fills under **today's** rules: gvol < 1, FRENZY, plus WIDE only on a green bar with a streak > 12. They are priced on ticks, and capture is the live lock at 1×.

### 2a. Per month

| month | alerted staircase episodes | episodes FRENZY / WIDE entered | fills | lock capture (sum of %, 1×) | median best move within 48 h of the alert | mean 24 h return from the alert | +10 % before −5 % | close below VWAP within 24 h | reached ST32 |
|---|---|---|---|---|---|---|---|---|---|
| Jan | 77 | 25 | 40 | +39.6 | +13.9 % | −0.7 % | 44 % | 90 % | 20 |
| Feb | 92 | 21 | 30 | +21.2 | +10.6 % | −5.0 % | 27 % | 95 % | 21 |
| Mar | 101 | 36 | 57 | +28.0 | +14.1 % | −1.0 % | 34 % | 95 % | 19 |
| Apr | 156 | 28 | 35 | −16.6 | +18.8 % | −2.2 % | 32 % | 93 % | 33 |
| May | 110 | 33 | 43 | +37.5 | +12.8 % | −1.1 % | 37 % | 98 % | 21 |
| Jun | 86 | 15 | 23 | +7.0 | +12.4 % | −4.1 % | 16 % | 95 % | 18 |
| Jul | 84 | 17 | 24 | +2.1 | +10.5 % | −7.8 % | 26 % | 99 % | 13 |
| Aug | 117 | 29 | 37 | +11.2 | +14.8 % | −3.3 % | 23 % | 97 % | 24 |
| Sep | 113 | 21 | 28 | +45.7 | +14.2 % | −2.6 % | 32 % | 94 % | 26 |
| **Total** | **946 (≈ 105 a month)** | **225 (24 %)** | **317** | **+0.55 %/fill** | **+13.8 %** (mean +31 %) | **−3.0 %** (median −7.4 %) | **30 %** (69 % fall −5 % first) | **95 %** | 197 |

### 2b. Why FRENZY / WIDE did not enter (946 alerted episodes)

| reason (first that applies) | episodes | mean / median 24 h return from the alert | median best move within 48 h | +10 % before −5 % | ran ≥ +30 % within 48 h |
|---|---|---|---|---|---|
| **entered** (FRENZY / WIDE, today's rules) | 225 | **+6.5 % / +0.8 %** | **+27.0 %** | 47 % | 101 |
| **volume never ≥ 100×** (`FRENZY_VOL_FADED` all episode: the ORCA / US / PARTI class) | **293** | **−9.9 % / −10.0 %** | +6.1 % | **11 %** | 24 |
| WIDE refused today (ATR > 2.5, or a green "reclaim" bar) | 256 | −5.6 % / −10.6 % | +14.2 % | 32 % | 73 |
| market-volume gate (`GVOL_HIGH`: the API3 class) | 79 | +2.6 % / −2.4 % | +15.7 % | 41 % | 27 |
| price dislocation > 1 % at the entry print | 61 | +1.6 % / −5.4 % | **+33.4 %** | 43 % | 35 |
| 100× state only on bars of another episode / unverified | 18 | −5.1 % | +13.3 % | 56 % | 4 |
| 24 h volume < $20M at the fresh bar | 14 | −13.6 % | +5.5 % | 7 % | 0 |

What the table shows:
- **FRENZY's 100× bar is doing its job.** The episodes it enters are by far the best class (+6.5 % in 24 h, 27 % median room).
- **The volume-faded staircases, the class the operator is pointing at today, are the worst class of the year.** They average −9.9 % over the next 24 h, and only 11 % reach +10 % before −5 %. ORCA, US and PARTI belong to the lucky 8 %.
- **"Setup ON with no fresh bar" is the dominant shape inside entered episodes:**
  - 161 of 225 entered episodes got exactly one fill.
  - After the last exit, the setup stayed ON for ≥ 6 h in 55 % of them.
  - A +15 % higher high followed within 24 h in 37 %.
  - **But the price 24 h later was below the exit price in 71 % (median −7 %).** That is why re-entry keeps failing (§3a, and 5 earlier refutations).
- **The largest untested pocket of big runs is the dislocation refusals:** 61 episodes, with a median +33 % within 48 h. They were not studied here (§6).

### 2c. Hindsight check: did the alert keep rising?

| read at the alert close | all alerts (946) | ST32 "still ON after 32 h" (197) |
|---|---|---|
| +10 % touched before −5 % | 30 % | 33 % |
| −5 % touched first | 69 % | 66 % |
| a 5m close below the spike VWAP within 24 h (staircase broken) | 95 % | 74 % |
| 24 h return, mean / median | −3.0 % / −7.4 % | −1.2 % / −5.0 % |
| best move within 48 h, median | +13.8 % | +17.9 % |

**The staircase is a description of the past. At the moment the alert fires, about 2 in 3 reverse first.**

## 3. Candidate mechanisms (pre-declared; live sizing FRENZY ×1.21 / WIDE ×0.94 of equity, 2 slots per sleeve, from $3k)

### 3a. Staircase-qualified ENTRY (enter once when the 50× state turns back ON; exit with the live lock)

**Rules:**
- **a1 "second chance"**: the first 50× fresh bar after the episode's first 100× FRENZY bar (entered or refused), with the pair flat.
- **a2 "volume-faded"**: the first 50× fresh bar while FRENZY's 100× setup is OFF (the ORCA / US / PARTI class).
- **For both:** gvol < 1, $20M volume, entry 12 s after the signal close, ticks or 1m prices.

| rule | N | mean %/trade | WR | day 95 % CI | Jan–Apr / May–Sep | drop top 5 / 10 | null (random state bar, same episode, same 5m pricer) | book from $3k (×0.94) | max DD |
|---|---|---|---|---|---|---|---|---|---|
| **a1 second chance** | 211 | **−0.49** | 45 % | **[−0.91, −0.08]** | −0.47 / −0.52 | −0.71 / −0.83 | p = 0.49 (timing no better than a random ON bar) | **−66 %** | −77 % |
| **a2 volume-faded** | 366 | **−0.38** | 46 % | **[−0.73, −0.04]** | −0.22 / −0.54 | −0.53 / −0.63 | p < 0.001 better than a random bar, but still below 0 | **−77 %** | −81 % |

**Detail:**
- **Per month:** a1 is positive in 3 of 9 months, and a2 in 3 of 9.
- **Red vs green signal candle:** neither helps (a1 red −0.69 / green −0.30; a2 red −0.43 / green −0.35).
- **Holding time and slot cost:** a1 holds 0.8 h on average and a2 0.5 h. If (a) shared WIDE's 2 slots, it would crowd out only 1–2 of today's fills, so slot cost is not the problem. The entries themselves lose.
- **Other exits on these entries:** the VWAP-stop and staircase-exit variants on the same entries are also negative (a1 −0.18…−0.41, a2 −0.68…−1.85), with every CI spanning or below 0.
- **Anecdotes inside the year:**
  - ORCA a1 on 04-28: +2.2.
  - MOVR a1 on 08-27: −3.1, after which it ran +14.
  - RLC / ORCA 10-06 are after the cache ends; see §1b.

**Verdict: REFUTED.** Both CIs are entirely below 0, both halves are negative, and a1's timing carries no information. This is the 6th refutation of the re-entry family. a2 is a new angle and fails on its own.

### 3b. VWAP-anchored stop (pre-arm −3 replaced, lock unchanged), on today's-rules fills (403: FRENZY 249 + WIDE hold-green 154; 395 on ticks)

**Variants:**
- **BP (primary, pure widening):** exit on a 5m close that is ≤ −3 net **and** below VWAP − k·ATR.
- **BF:** exit on any 5m close below VWAP − k·ATR.
- **Both** have a hard floor at −12 and keep the lock after the peak reaches +3.

**Read on the LIVE-STOPPED cohort only** (179 fills that the live lock's −3 stopped). Only one-point CIs/p-values were computed per variant, on the stopped cohort.

| variant | non-stopped fills, Δ | stopped: saved (Δ sum) | stopped: deeper (Δ sum) | Δ mean on stopped | day CI | sign-flip p | Jan–Apr / May–Sep | drop 5 / 10 | FRENZY / WIDE Δ on stopped | book (lock +314 %, DD −51 %) |
|---|---|---|---|---|---|---|---|---|---|---|
| **BP k 0** | **0 on 224** ✓ | 65 (+358) | 110 (−378) | −0.11 | [−0.90, +0.65] | 0.60 | +0.18 / −0.40 | −0.35 / −0.57 | +0.09 / −0.50 | +160 %, DD −67 % |
| **BP k 0.5** | **0 on 224** ✓ | 72 (+400) | 103 (−395) | +0.03 | [−0.79, +0.83] | 0.47 | +0.38 / −0.33 | −0.21 / −0.44 | +0.26 / −0.41 | +241 %, DD −59 % |
| BF k 0 | −45 on 10 | 92 (+370) | 86 (−355) | +0.08 | [−0.69, +0.84] | 0.44 | +0.39 / −0.23 | −0.15 / −0.36 | +0.41 / −0.54 | +140 %, DD −63 % |
| BF k 0.5 | −4 on 1 | 89 (+398) | 87 (−377) | +0.12 | [−0.70, +0.92] | 0.40 | +0.53 / −0.30 | −0.12 / −0.35 | +0.44 / −0.49 | — |

- On ORCA 10-06 itself, every variant saves the trade: −3.1 → +2.9.
- On the year, every −3 stop the VWAP rule saves is paid for by one that drops to −5…−12.
- **The halves flip sign, and the sleeves flip sign:** FRENZY is slightly positive and WIDE negative. The FRENZY-only reading is a post-hoc split, so it is not evidence.
- **The book is worse in every variant.** Holds grow from 0.6 h to 1.3–1.6 h, and the −12 tail deepens drawdowns.

**Verdict: NOT ESTABLISHED.** It fails the CI, both halves, drop-5 and the book. **Offer as an observe-only scout line for FRENZY fills (§5); do not arm.**

### 3c. Staircase exit (hold while above the spike VWAP, exit on the first 5m close below it) vs the lock, same 403 fills

| exit | N | mean %/trade | WR | day CI | Jan–Apr / May–Sep | drop top 5 / 10 | Δ vs lock (CI · sign-flip p) | book from $3k | max DD | mean hold |
|---|---|---|---|---|---|---|---|---|---|---|
| **LOCK2 (live)** | 403 | **+0.41** | 56 % | [+0.07, +0.72] | +0.45 / +0.37 | +0.27 / +0.17 | — | **+314 %** | −51 % | 0.6 h |
| STAIR (−3 stop always) | 403 | +0.68 | 20 % | [−0.35, +1.83] | **+0.03** / +1.34 | −0.11 / −0.69 | +0.27 ([−0.65, +1.38] · 0.32) | +106 % | −75 % | 3.1 h |
| STAIR_c (cohort: −3 only before +3) | 403 | +0.51 | 24 % | [−0.63, +1.73] | +0.05 / +0.98 | −0.28 / −0.87 | +0.10 ([−0.92, +1.25] · 0.42) | — | — | — |
| VWX (no −3, floor −12) | 403 | −0.07 | 29 % | [−1.37, +1.40] | −0.51 / +0.39 | −0.86 / −1.46 | −0.47 ([−1.76, +0.83] · 0.77) | **−95 %** | −99 % | 5.4 h |

- The staircase exit wins the RLC kind of day (+55.9 on RLC 10-05) and loses on four trades out of five.
- Its higher mean is carried by its top 5 trades (drop-5 turns it negative), and it is flat in Jan–Apr.
- The book is a third of the lock's, at a deeper drawdown.
- This repeats `FRENZY_RIDE_CAPTURE_SYNTHESIS` (STAIR: −2.3 on non-rides, 22 % WR).

**Verdict: REFUTED (third time).**

## 4. Verdict table

| mechanism | verdict | why |
|---|---|---|
| (a1) second-chance entry | **refuted** | −0.49, CI [−0.91, −0.08], both halves < 0, timing = random (p 0.49), book −66 % |
| (a2) volume-faded staircase entry (ORCA / US / PARTI class) | **refuted** | −0.38, CI [−0.73, −0.04], both halves < 0, book −77 %; the class itself averages −9.9 % over 24 h |
| (b) VWAP-anchored stop | **not established → observe-only** | saved ≈ deeper (+358 / −378), CI spans 0, halves and sleeves flip sign, book worse |
| (c) staircase exit | **refuted** | 20 % WR, negative without its top 5, Jan–Apr ≈ 0, book +106 % vs +314 % |

No haircut applies, because nothing is positive. No gate is pre-registered for arming, because nothing passed.

## 5. Cheapest forward test that answers the open question

**Scout line "🪜 VWAP-stop shadow" (observe-only).** For every FRENZY / WIDE fill that the live −3 stops, check whether a 5m close ≤ −3 net **and** below `entry_frenzy_vwap` happened within 12 h.
- `entry_frenzy_vwap` and the post-exit path (`post_exit_running_low`, ticks) are already stamped on every fill, so this needs no bot change.
- Price the BP k 0.5 variant on ticks: saved vs deeper.

**Frozen review gate (first 20 live-stopped FRENZY + WIDE fills):** to propose arming, all of these must hold:
- Δ sum > 0;
- saved > deeper;
- positive on FRENZY **and** WIDE separately;
- no single fill > 50 % of the gain.

Otherwise close the idea. Expected time: about 20 stopped fills ≈ 3–5 weeks at the current pace.

**Nothing to add for entries.** The scout's 🪜 alert already exists. Its own forward record will show whether the 2-in-3 reversal rate holds on new data.

## 6. Blind spots (what this study could NOT test)

1. **Year data ends 2026-09-27.** ORCA / US / API3 / PARTI / RLC (Oct 5–6) are anecdotes on public 1m klines, not part of any year statistic.
2. **Scout cadence is ignored.** Alerts are timed at the first qualifying 5m close. The real scout runs on a slower cadence, so real alerts come later. That would make (a) worse, not better, since alerts already sit at local tops.
3. **The scout's own implementation differs slightly.** It uses Binance quote volume, has no chained verification, and has a 96 h window. The class here uses the engine walk at 50×. On the live cases the two agree to about 0.2 % in volume (ORCA 100.09× vs 99.89×) but anchor episodes differently (ORCA: scout 10-04 18:30 vs engine-100× 10-05 22:20).
4. **The missed-entry breakdown does not model slots or the pair-day cap.** The books do model 2 slots per sleeve. They do not model the pair-day cap or FRENZY's strong-ADX 10× sizing.
5. **Funding is not included.** That matters for the multi-hour STAIR / VWX holds.
6. **The (a) null uses a 5m-bar pricer**, for both trigger and null, so it is the same statistic on both sides. P&L figures use ticks / 1m. a2 is −0.07 on 5m vs −0.38 on ticks, so granularity matters and the 1m/tick figure is the trustworthy one.
7. **(b)'s −12 floor and 12 h cap were pre-declared, not optimised.** Other k values or floors were not tried, on purpose.
8. **Dislocation-refused episodes were not studied.** These are 61 episodes, 35 of them ≥ +30 % within 48 h. They are the biggest pocket of big runs the bot structurally skips, and a candidate for a separate pre-registered study.
9. **No winner / loser separator screen was run on the alerts.** No exhaustive 2D scan with a shuffled null was done, so **no "nothing separates staircases that keep rising" claim is made.** Only the scout's own rule was tested.
10. **Logs:** the "why no entry" answers come from the replica plus the decisions journal. The journal logs only fresh bars, and US / PARTI had none. The EB server logs were not pulled. Code death on the FRENZY path is not ruled out by logs, although the API3 refusal and the ORCA / RLC fills show the path ran on 10-05 and 10-06.
