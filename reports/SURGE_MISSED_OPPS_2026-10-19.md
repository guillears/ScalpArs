# SURGE_LONG: are we missing opportunities? (2026-10-19)

Operator question: "SURGE trades: guarantee we are not missing many opportunities." The suspicion was that SURGE_LONG opens only 1 trade per trigger and misses the whole window.

## Plain-English answer

- **It is not a bug. Live, the bot's choices match the replica on every pair at every trigger.**
  - Since Oct-5 the bot judged 11 trigger bars, 20 coins per bar. The replica reproduces all 177 refusals in the decision journals exactly, plus the 3 coins that passed.
  - Each of those passes became a live fill: STRK 10-15, AGT 10-18, GTC 10-19.
  - No coin that passed was ever refused by the slot cap, the WILLY hold or the 5-minute window.
  - No trigger was blocked by the 4 h spacing.
- **"1 trade per trigger" is how the filters work. There is no rule that picks only one leader.**
  - The rule is *every* top-20 coin with ATR ≥ 1.5 % whose own 30-minute move beats BTC's, up to 4 coins.
  - On a typical trigger about 17–18 of the 20 coins are refused for low ATR. These are the majors (SOL, XRP, DOGE, BNB…), whose 5-minute ATR is about 0.2–1 %.
  - Another 1–3 coins are refused because they did not outrun BTC.
  - In the two-year backtest, about 1 filled trigger in 4 took 2–4 coins: 159 took one, 40 took two, 10 took three, 2 took four.
- **The coins we skip lose money.** Over two years, with today's exit, on ticks:
  - Coins that did not outrun BTC ("not leader"): 586 fills, **−0.30 % per fill**, 95 % range [−0.46, −0.13].
  - Leaders whose ATR was 1.0–1.5 %: 415 fills, **−0.26 %**, range [−0.42, −0.10].
  - The 2nd and 3rd best movers taken *next to* a leader: 13 fills, 1 winner, **−0.72 %**, range [−1.13, −0.17].
- **Spacing, a longer window and dropping the 24 h high do not pay.**
  - The would-be picks of the 299 trigger bars that the 4 h spacing blocked averaged **−0.14 % per fill**. They were negative in all three periods.
  - Spacing 1 h and 0 h lose money. 2 h is +$205 over two years, but 95 % range is −$430…+$991 and one pair carries the whole gain.
  - A 10- or 15-minute window gives Δ −$232 / −$129.
  - Dropping the BTC 24 h-high leg gives Δ −$1,560.
- **Recommendation: arm nothing.** Reject (b) top-2/3 movers and (c) no 24 h high outright. Keep (a) spacing at 4 h. Only (d) "re-judge the next 5-minute bar *when the trigger bought nothing*" earns an optional observe-only line, with a gate frozen in advance. The evidence for it is thin: it was found by a post-hoc split and fails the confidence bar.

**Units:** P&L is % per fill, net of fees, under today's SURGE exit (peak − 1×ATR once +1 %, initial stop −0.7…−1.2, no lock, 240 min). $ is at today's live size: notional ≈ $2,214, so **$22.1 per 1 %**. Evidence is counted in **windows** (trigger bars that filled) and **days** (the bootstrap resamples UTC days).

**Blocks:**
- ① BATCH / LIVE: Oct-5 → Oct-19 03:40 UTC. Decision journals plus the replica on public klines, walked on 1m bars.
- ② MASTER: the real SURGE fills.
- ③ BACKTEST: replica 2025-01-01 → 2026-10-04 on aggTrade ticks (1m fallback), plus the yr6 real-engine replay. Blocks ① and ③ do not overlap.

---

## 1. Per trigger: how many coins passed, and why the others were refused

### ① Live and paper decision journals vs the replica (Oct-5 → Oct-19)

| trigger (bar close UTC) | judged | ATR_LOW | NOT_LEADER | EMA_GAP | PASS | what happened |
|---|---|---|---|---|---|---|
| 10-09 11:35 / 11:40 / 11:50 (one window) | 20 ×3 | 18 | 2 | 0 | 0 | no pick (STRK ATR 1.31–1.41) |
| 10-11 15:45 (+15:50, 16:00) | 20 | 14 | 5 | 0 | **1 (STRK)** | **PAIR_HELD**: STRK was already open in another sleeve. Replica +1.41 % |
| 10-12 05:05 | 20 | 15 | 5 | 0 | 0 | no pick |
| 10-15 15:05 | 20 | 17 | 2 | 0 | **1 (STRK)** | filled, +1.53 % · +$74.64 |
| 10-16 11:25 | 20 | 17 | 3 | 0 | 0 | no pick |
| 10-18 01:45 | 20 | 15 | 4 | 0 | **1 (AGT)** | filled, +7.53 % · +$159.98 |
| 10-18 09:35 | 20 | 17 | 3 | 0 | 0 | no pick |
| 10-19 00:45 | 20 | 19 | 1 | 0 | 0 | no pick (replica only; the journal export ends 10-18 16:45) |
| 10-19 02:50 | 20 | 17 | 1 | 1 (AGT) | **1 (GTC)** | filled, +4.15 % · +$95.72 |
| 10-05 01:30 · 10-06 15:00 · 10-11 17:40 | — | | | | | market volume < 1 (GVOL_LOW), correctly refused. 10-05 was also found late (WINDOW_MISSED) but was below the volume gate anyway |

- **Parity:**
  - The universe matches the journal 20/20 at every trigger.
  - Pair verdicts agree **177 / 177**.
  - The replica's PASS coins are exactly the live fills.
  - Detail: `reports/SURGE_MISSED_OPPS_live_journal_parity.csv`.
- **Spacing:** no live trigger fell inside a 4 h spacing window. The next trigger after STRK came 20 h later, after AGT 7 h 50 later, and after GTC none yet.
- **Other refusals of a passing coin:** SURGE_MAX_SLOTS, WILLY_HOLD and OPEN_REFUSED never fired. The only such refusal was the STRK 10-11 PAIR_HELD.
- **Live averages:** 16.8 ATR_LOW · 2.7 NOT_LEADER · 0.1 EMA_GAP · **0.36 PASS per trigger bar**.

### ② Master (real fills)

- The only archived SURGE_LONG fill is STRK 10-15 (B22): +1.53 % · +$74.64.
- AGT +7.53 % ($159.98) and GTC +4.15 % ($95.72) are in the 10-19 live export and not yet archived.
- All three were PASS coins. The master holds fills only, so it cannot show refused candidates. Block ① is that check.

### ③ Backtest

**Replica, 581 trigger bars that met today's rule, 2025-01 → 2026-10-04.** Per-bar average:
- 18.2 ATR_LOW
- 1.0 NOT_LEADER
- 0.28 EMA_GAP
- **0.48 PASS**

Number of coins passing per bar:

| coins passing | trigger bars |
|---|---|
| 0 | 370 |
| 1 | 159 |
| 2 | 40 |
| 3 | 10 |
| 4 | 2 |

**Replica parity:** universe and status agree 4,220 / 4,220 with the bot-rule review replica (`surge_review_m0.3_h0_v5_f_g1.0_s4`). The walk agrees 211 / 211 exactly (±0.00) with its BR_NO_BE_LOCK walker. The trail-cap study validated that walker against the engine replay at 556 / 556.

**yr6 real-engine replay (3 seeds, gap ≥ 0):** fills per seed-trigger:

| fills per trigger | seed-triggers |
|---|---|
| 1 | 408 |
| 2 | 81 |
| 3 | 13 |
| 4 | 2 |

So 19 % of filled triggers take 2 or more coins.

**Verdict on Q1:** "1 per trigger" is the selection working. The limiting filter is the ATR ≥ 1.5 % floor, which leaves only 2–4 eligible alts per trigger in the top-20. The leader test then removes most of the rest. There is no bug.

---

## 2. Spacing: triggers inside the 4 h after a filled trigger

| block | spaced-out trigger bars | parent windows | would-be fills | WR | avg %/fill | 95 % CI (day-block) | Σ$ |
|---|---|---|---|---|---|---|---|
| ① live Oct-5→19 | **0** | 0 | 0 | — | — | — | — |
| ③ backtest | 262 | 96 of 211 filled triggers | 299 | 27 % | **−0.135** | [−0.38, +0.12] | −$889 |

- **Per period:** 2025 −0.08 · 26-H1 −0.20 · 26-H2 −0.22.
- **By time since the fill:**

| hours since the fill | fills | avg %/fill |
|---|---|---|
| 0–1 h | 203 | −0.08 |
| 1–2 h | 48 | −0.23 |
| 2–3 h | 38 | −0.35 |
| 3–4 h | 13 | −0.08 |

- **Same pair re-bought:** −0.02 (186 fills).
- **A different pair:** −0.31 (116 fills).

---

## 3. Other coins that also surged in the same window

### Theory first: why should the leader beat #2 and #3?

A BTC spike of ≥ 0.3 % in 30 minutes on 5× volume lifts almost the whole market through beta, and part of that move reverts within minutes to hours.

- A coin that **outruns BTC** shows its own demand: a catalyst, a short squeeze or rotating attention. Its move is not only borrowed from BTC, so a trailing exit can follow through.
- A coin that only keeps pace has nothing of its own once BTC's impulse fades.
- The more coins clear the bar together, the more the move looks like plain beta. Such triggers should follow through less.
- A coin that is still below its own 24 h high, or that leads only on the next bar, is chasing a move that is already being priced in.

The data agrees on every point.

### ③ Backtest: refused or late coins at the same trigger, priced with today's exit

| cohort | N | windows | WR | breakeven WR | avg %/fill | 95 % CI | 2025 / 26-H1 / 26-H2 |
|---|---|---|---|---|---|---|---|
| NOT_LEADER (ATR ≥ 1.5) | 586 | 327 | 25 % | 34 % | **−0.300** | [−0.46, −0.13] | −0.39 / +0.00 / −0.47 |
| NOT_LEADER but moving up (pmove > 0) and gap ≥ 0 | 64 | 57 | 28 % | 36 % | −0.209 | [−0.55, +0.16] | +0.06 / −0.29 / −0.50 |
| **2nd / 3rd mover next to a leader** (TOP2/3 adds at already-filled triggers) | 13 / 20 | 13 / 18 | 8 % / 10 % | 32 % | **−0.72 / −0.70** | [−1.13, −0.17] / [−1.11, −0.19] | negative in all 3 |
| Leader with ATR 1.0–1.5, gap ≥ 0 | 415 | 268 | 36 % | 48 % | **−0.263** | [−0.42, −0.10] | −0.32 / −0.30 / +0.13 |
| Leader with EMA13 < EMA50 (today's armed gap filter) | 161 | 137 | 24 % | 29 % | −0.203 | [−0.59, +0.34] | −0.57 / −0.32 / +0.57 |
| Late leader on the next bar (+5 min), at triggers already filled | 31 | 28 | 26 % | 41 % | −0.427 | [−0.95, +0.17] | −0.44 / −0.09 / −1.20 |
| Late leader on the next bar, at triggers that **bought nothing** | 50 | 46 | 40 % | 32 % | **+0.236** | [−0.24, +0.74] | +0.26 / **−0.12** / +0.58 |

**TODAY's picks for reference:** 277 fills · 211 windows · WR 35 % · **+0.116 %/fill** [−0.17, +0.43] · +$712. Per period: 2025 −0.18 / 26-H1 +0.16 / 26-H2 +0.63.

Within today's picks:
- When **one** coin passes: **+0.27 %/fill** (159 fills).
- When 2–4 coins pass: −0.08 (118 fills). Broad moves follow through worse, which matches the theory.
- A pick that is already at its own 24 h high: −0.14 (N 35) vs +0.16 for the others. This is small and noisy.

### ① Live (Oct-5 → 19; 1m walker, slightly pessimistic on fast moves)

| cohort | N | windows | WR | avg %/fill | detail |
|---|---|---|---|---|---|
| NOT_LEADER | 30 | 9 | 23 % | **−0.91** | −$601 at today's size |
| ATR_LOW (1.0–1.5) | 17 | 7 | 41 % | −0.15 | −$56 |
| EMA_GAP | 1 | 1 | 100 % | +2.81 | AGT 10-19 |
| 2nd mover next to a leader | 2 | 2 | 100 % | +2.72 | ZEREBRO +4.84 (10-15), GTC +0.60 (10-18) |

- The 2nd-mover result is the opposite of the backtest, but N = 2 cannot outweigh 13–20 backtest fills.
- **1m-walker pessimism:** AGT walks at +2.90 in the replica vs +7.53 live, because a 1m candle fills the trail at the bar's low. STRK walks at +1.61 vs +1.53 live, and GTC at +3.93 vs +4.15.

---

## 4. Candidate relaxations, priced

The Δ column is the variant minus TODAY, both adds and removes. A relaxation that fills a trigger TODAY left empty also starts the 4 h spacing, so it can displace a later TODAY fill. The "removed" columns show that.

### ③ Backtest, 2025-01 → 2026-10-04 (ticks; 9 % of walks on the 1m fallback)

| variant | fills | windows | WR | avg %/fill | 95 % CI | **added: N · avg · CI** | removed: N · avg | **Δ$ vs TODAY [95 % CI]** | P(Δ>0) | Δ$ 2025 / 26-H1 / 26-H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| TODAY | 277 | 211 | 34.7 | +0.116 | [−0.17, +0.43] | — | — | — | — | — |
| (a) spacing 2 h | 304 | 229 | 34.2 | +0.136 | [−0.14, +0.43] | 27 · +0.34 · [−0.70, +1.76] | 0 | **+$205** [−430, +991] | 0.68 | −58 / +23 / +240 |
| (a) spacing 1 h | 335 | 249 | 32.5 | +0.052 | [−0.19, +0.31] | 58 · −0.26 · [−0.65, +0.21] | 0 | −$328 [−865, +259] | 0.12 | −196 / −129 / −2 |
| (a) spacing 0 | 505 | 362 | 30.7 | −0.012 | [−0.20, +0.18] | 228 · −0.17 · [−0.44, +0.10] | 0 | −$845 [−2,144, +527] | 0.11 | −159 / −614 / −73 |
| (b) top-2 movers | 314 | 235 | 32.8 | +0.053 | [−0.20, +0.34] | 48 · −0.16 · [−0.57, +0.30] | 11 · +0.71 | −$342 [−889, +175] | 0.10 | +26 / −135 / −232 |
| (b) top-3 movers | 321 | 235 | 32.4 | +0.038 | [−0.21, +0.32] | 55 · −0.23 · [−0.60, +0.19] | 11 · +0.71 | −$445 [−1,017, +118] | 0.06 | −25 / −189 / −232 |
| (c) no BTC 24 h-high leg | 642 | 494 | 29.8 | −0.060 | [−0.24, +0.14] | 453 · −0.12 · [−0.32, +0.10] | 88 · +0.18 | **−$1,560** [−3,695, +737] | 0.09 | −1,476 / +271 / −355 |
| (d) window 10 min (re-judge bar +1) | 340 | 245 | 34.7 | +0.064 | [−0.17, +0.31] | 81 · −0.02 · [−0.40, +0.39] | 18 · +0.51 | −$232 [−1,141, +603] | 0.30 | +143 / −132 / −243 |
| (d) window 15 min (bars +1, +2) | 379 | 261 | 34.8 | +0.070 | [−0.15, +0.29] | 123 · −0.00 · [−0.31, +0.35] | 21 · +0.28 | −$129 [−1,196, +865] | 0.41 | +157 / +44 / −330 |
| (info) ATR floor 1.0 | 517 | 306 | 38.1 | −0.028 | [−0.20, +0.15] | 275 · −0.21 · **[−0.38, −0.03]** | 35 · −0.31 | −$1,030 [−2,228, +204] | 0.05 | −875 / +68 / −223 |

- **(a) spacing 2 h:** its +$205 comes from 4 fills in 26-H2 at +2.7 %/fill. One pair carries 143 % of the added Σ, and the result is negative in 2025. This is luck-shaped, not an edge.
- **(c) no 24 h high:** this is BTC's 24 h-high leg of the trigger. There is no per-coin 24 h-high filter today.

### ① Live, Oct-5 → 19: 9 windows, 1m walker

TODAY = 4 fills (replica: STRK 10-11 PAIR_HELD +1.41, STRK 10-15 +1.61, AGT +2.90, GTC +3.93) · 100 % WR · +2.46 %/fill · +$218.

| variant | added fills (pair, %) | Δ$ |
|---|---|---|
| spacing 2 h / 1 h | none | 0 |
| spacing 0 | MAGIC −1.39, STRK −0.48 | −$42 |
| top-2 / top-3 movers | ZEREBRO +4.84, GTC +0.60, AGT −0.70 | +$105 |
| no 24 h high | 20 fills, avg −0.01 (STRK 10-10 +8.30, BR +4.08 … 12 losers) | −$42 |
| window 10 min | MAGIC −1.39, WAL −1.27, ZEREBRO +5.11, AGT −0.54, AGT −1.37 | +$12 |
| window 15 min | same as 10 min, plus GTC −1.23 | −$102 |
| (info) ATR 1.0 | STRK −1.74, NEAR +0.01, WAL +2.33, TIA +0.54, AVA +3.12 | +$94 |

The live block is 9 windows. It is far too small to support or refute anything. It is listed because the rule is three blocks in every table.

---

## 5. Recommendation and pre-committed gates

| candidate | verdict | why |
|---|---|---|
| (a) spacing 4 h → 2 h / 1 h / 0 | **REJECT; keep 4 h** | Spaced-out picks lose −0.14 %/fill, negative in all 3 periods. 1 h and 0 lose $. 2 h is +$205 at P 0.68, driven by one pair, negative in 2025. Live: 0 spaced triggers so far, so nothing is missed today. |
| (b) top-2 / top-3 movers | **REJECT** | 2nd/3rd movers next to a leader lose −0.72 %/fill, CI below zero, negative in all 3 periods. Non-leaders lose −0.30 over 586 fills. Theory and data agree. |
| (c) drop the 24 h-high leg | **REJECT** | Adds 453 fills at −0.12 %, Δ −$1,560, the 2025 bull carries −$1,476. |
| (d) 10 / 15-min window | **REJECT as a rule.** Optional **observe-only** line for the sub-case "re-judge bar +1 only when the trigger bought nothing" | Overall Δ −$232 / −$129. At already-filled triggers the late leaders lose −0.43. At empty triggers +0.24 (N 50, 46 windows, P 0.82, 26-H1 −0.12). This split was found after seeing the data, so it is in-sample (30–50 % haircut) and fails the confidence bar. |
| (info) ATR floor 1.0 | **REJECT** | The added leaders lose −0.21, CI [−0.38, −0.03]. |

**Optional observe-only line (needs code, so not built; only if the operator wants it): SURGE_LATE_LEADER.**
- **What it records:** on a LONG trigger that ends its window with **zero** fills, re-run `surge_pair_pick` on the top-20 at the **next** closed 5m bar. The test is the pair's 30-min move vs BTC's 30-min move at that bar, with ATR ≥ 1.5 and gap ≥ 0. It records a would-be fill at that bar's close + latency and never trades. Counter tag `SURGE_LATE_LEADER`.
- **🔒 Frozen gate:**
  - Read at **≥ 15 would-be fills over ≥ 8 windows**, priced with today's SURGE exit.
  - ARM only if all of these hold: avg > 0 with **window-bootstrap P ≥ 0.95** · WR ≥ the sleeve breakeven (≈ 32 % in this study) · no window or pair ≥ 50 % of Σ · both halves of the observation period positive.
  - Otherwise drop it.
  - Never re-fit "next bar / empty trigger" on the data that failed it.
- **Expected rate:** the backtest has 50 such fills in 46 windows over 21 months (about 2.4 fills a month), so the bar takes about 6 months.

**What would change these conclusions:** a live SURGE book with ≥ 30 fills over ≥ 8 windows in which the journal shows passing coins lost to PAIR_HELD, WILLY_HOLD or slots. The only case so far is STRK 10-11. That would be an execution problem, not a filter problem. The per-trigger ledger (`surge_triggers`) and the journal already record it.

## Limitations (stated honestly)

- **What the replica does not model:**
  - WILLY global hold, PAIR_HELD, max_open_positions, entry dislocation > 0.3 % and scan order.
  - The yr6 real-engine replay does model them, and takes fewer fills (617 / 3 seeds ≈ 206 vs 277).
  - Its levels differ: 2025 −0.01 / H1 −0.00 / H2 +0.91 %/fill vs this replica's −0.18 / +0.16 / +0.63.
  - Both say the sleeve's edge is the 26-H2 bull, and both see the same refusal structure.
- **Latency:** fixed at 60 s (the review calibration; live fills opened 27–45 s after the bar close).
- **ATR:** Wilder over the full series vs the bot's 100-bar seed. The difference is negligible (the 4,220 / 4,220 status parity confirms it).
- **EMA gap:** judged on closed bars up to the trigger bar. The bot judges it on its scan view.
- **Live block:**
  - Walked on 1m candles, which are pessimistic on spikes (AGT +2.90 vs +7.53 real).
  - Its universe comes from 97 pairs (top-90 today plus every pair in the journals). Parity with the journal is 20/20 at every judged trigger.
- **Multiple testing:** 9 variants were tested, so the best-of-9 is expected to look lucky. The only positive Δ (spacing 2 h, P 0.68) is well inside that luck.

## Files

**Scripts (read-only research):**
- `scripts/study_surge_missed_opps.py` (feat / sim)
- `scripts/study_surge_missed_opps_report.py`
- `scripts/study_surge_missed_live_fetch.py` (public klines, budget-guarded)
- `scripts/study_surge_missed_live.py`

**Backtest outputs:**
- `reports/SURGE_MISSED_OPPS_fills.csv`: every variant fill
- `reports/SURGE_MISSED_OPPS_walks.csv`
- `reports/SURGE_MISSED_OPPS_universe.csv`: each trigger × top-20 pair status
- `reports/SURGE_MISSED_OPPS_spaced.csv`
- `reports/SURGE_MISSED_OPPS_refused.csv` and `reports/SURGE_MISSED_OPPS_refused_summary.csv`
- `reports/SURGE_MISSED_OPPS_variants.csv`, `reports/SURGE_MISSED_OPPS_incremental.csv`, `reports/SURGE_MISSED_OPPS_incremental_split.csv`, `reports/SURGE_MISSED_OPPS_delta.csv`

**Live outputs:**
- `reports/SURGE_MISSED_OPPS_live_fills.csv`
- `reports/SURGE_MISSED_OPPS_live_universe.csv`
- `reports/SURGE_MISSED_OPPS_live_variants.csv`
- `reports/SURGE_MISSED_OPPS_live_journal_parity.csv`
