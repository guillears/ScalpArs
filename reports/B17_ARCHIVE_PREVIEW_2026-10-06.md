# B17 archive preview: before and after today's changes (2026-10-06)

**PREVIEW ONLY.** Nothing in `reports/` was archived, `MASTER_POOL_stacked.csv` was not touched, and there are no code or config edits and no commit. All builds ran in the scratchpad, from a copy of `build_master_pool.py` whose era discovery pointed at a scratch `reports/` folder (symlinks to the real archives plus a scratch B17 file).

**Sources:** `~/Downloads/scalpars_orders_paper_2026-10-06_13-24-14.csv` (25 rows) and `scalpars_decisions_paper_2026-10-06_13-24-25.csv`. The decisions file holds no FRENZY lines, so FRENZY refusals were rebuilt from Binance klines (section 4).

**Units:** P&L % is the price move and does not depend on leverage. $ depends on size. Compare batches on %.

---

## 1. Headline tables

### B17 per sleeve: as traded vs today's rules

"Today's rules" means the builder stack v2026-10-06b plus the WIDE hold-green rule applied by hand (section 3).

| sleeve | as traded N · WR · avg % · $ | today's rules N · WR · avg % · $ | Δ $ |
|---|---|---|---|
| MOM-long | 6 · 67 % · +0.65 · +$548.22 | 6 · 67 % · +0.65 · **+$725.00** | +176.78 (CALM3D 2× → 1×) |
| MOM-short | 2 · 50 % · −0.21 · −$67.52 | 2 · 50 % · −0.21 · −$67.52 | 0 (pair-vol 0.86 keeps both) |
| FLIP short | 4 · 50 % · −0.49 · −$458.40 | 4 · 50 % · −0.49 · **−$262.65** | +195.75 (NEGDI15 / TG_SHALLOW 2× → 1×) |
| SPIKE_FADE | 3 · 100 % · +0.31 · +$183.99 | 3 · 100 % · +0.31 · +$197.65 | +13.66 (builder 0.5 % ticket re-price) |
| FRENZY_LONG | 5 · 20 % · −1.71 · −$380.91 | 5 · 40 % · **−0.60** · −$73.70 | +307.21 (+3 TP pricing, strong 0.5 lev) |
| FRENZY_WIDE | 5 · 40 % · −0.61 · −$110.48 | **2** · 100 % · **+3.00** · +$171.96 | +282.44 (hold-green blocks 3) |
| **TOTAL** | **25 · 52 % · −0.36 · −$285.11** | **22 · 64 % · +0.25 · +$690.74** | **+975.85** |

- **Builder only** (no WIDE hand rule): 25 · 56 % · −0.14 % · **+$408.02**.
- **Lock-exit sensitivity:** SAND #2 at the live lock floor of +2 % instead of the builder's +3 TP (section 2) gives a total of 22 · 64 % · +0.20 % · **+$622.06**.

### Full master ledger (`current_stack_ledger.py`, all eras, every sleeve)

| | fills | WR | net $ |
|---|---|---|---|
| Before B17 (BASE → B16) | 270 | 82 % | +$14,685.89 |
| + B17, builder stack | 295 | 80 % | +$15,093.91 |
| + B17, builder stack + WIDE hold-green by hand | 292 | 81 % | +$15,376.63 |
| same, SAND #2 at lock +2 % | 292 | 80 % | +$15,307.95 |

- **Scratch build check:** the scratch build without B17 reproduces the committed `MASTER_POOL_stacked.csv` exactly: 735 rows, 0 keep differences, 0 $ differences.
- **Master pool, full-size rows** (`stack_pct` mean): before 384 kept · 67.7 % · +0.121 % · +$7,584. After 409 kept · 67.0 % · +0.105 % · +$7,992.
- **Pool vs ledger:** the pool counts BOUNCE/CHASE and other rows the ledger drops, so the ledger rows above are the pinned figure.

---

## 2. How the builder (stack v2026-10-06b) re-prices B17

- **Blocks:** none. All 25 fills are `stack_keep`. There are no MANUAL fills, no `*_PROBE` fills and no SURGE_SHORT fills.
- **Re-priced fills:** 12, listed below.

| fill | as traded | today (builder) | why |
|---|---|---|---|
| QNT FLIP 10-06 02:11 | −1.15 % · −$438.44 | −1.15 % · −$219.22 | NEGDI15 2× → 1× (10-06a) |
| PUMP FLIP 10-06 02:06 | +0.12 % · +$46.93 | +0.12 % · +$23.46 | TG_SHALLOW + NEGDI15 2× → 1× (10-06a) |
| TAO MOM-long 10-06 10:44 | −0.69 % · −$200.66 | −0.69 % · −$100.33 | NONEXP_CALM3D 2× → 1× (10-06b) |
| SAND MOM-long 10-06 11:37 | −0.56 % · −$152.90 | −0.56 % · −$76.45 | NONEXP_CALM3D 2× → 1× (10-06b) |
| SAND FRENZY_LONG 10-04 05:05 | −3.00 % · −$123.82 | **+3.00 % · +$206.03** | peak +3.26 % reached the +3 TP, and it is sized at the strong lev 0.5 (10×) because ADX Δ and DI spread are both > 0 |
| AIN FRENZY_LONG 10-04 11:15 | +3.47 % · +$136.72 | +3.00 % · +$197.24 | +3 TP cap; strong lev 0.5 |
| SAND FRENZY_LONG 10-04 14:05 | −3.01 % · −$124.73 | −3.01 % · −$207.89 | strong lev 0.5 (it opened 11 min before that deploy) |
| MOVR WIDE 10-05 09:15 | +3.00 % · +$84.80 | +3.00 % · +$84.70 | +3 TP rounding |
| RLC WIDE 10-05 12:00 | +3.01 % · +$87.43 | +3.00 % · +$87.26 | +3 TP rounding |
| ZRX fade 10-05 00:14 | +0.30 % · +$30.05 | +0.30 % · +$40.75 | CF_FADE_CAP05 (ticket at 0.5 % × volume; optimistic, no market impact) |
| VTHO fade 10-05 23:51 | +0.36 % · +$89.41 | +0.36 % · +$92.37 | CF_FADE_CAP05 |
| ORCA, UMA FRENZY_LONG | −3.00 % | unchanged | ADX Δ < 0, so 0.32 lev as lived |

**Caveat on the FRENZY re-pricing.** Live exits now use the lock-then-trail rule (205): lock +2 % once the trade reaches +3 %, then trail 2 points below the peak. The builder still prices the old +3 TP (documented as path-dependent). I checked it on 1-minute klines:

| fill | stamped peak | builder price | lock exit | how sure |
|---|---|---|---|---|
| SAND #2 | +3.26 % | +3.00 % | **+2.0 %** (line = max(2, 3.26 − 2)), i.e. $137.35 | Deterministic: its whole path is stamped. |
| AIN #3 | +5.15 % | +3.00 % | ≥ +3.15 % | The builder's +3.00 is conservative. |
| MOVR, RLC | ≈ +3.0 % | +3.00 % | +2.0 % / +5.4 % in the 1-minute replica | Both hit the peak and the line inside one 1-minute bar, so the order within the bar is unknown. |

- **Replica check:** the replica matched all 6 stopped FRENZY/WIDE fills exactly (Δ 0).
- **Validator note:** the builder's +3 TP re-price of SAND #2 flips its sign (−3 % lived → +$206), and that is what trips validator check M1 (section 5).

---

## 3. Changes not in the builder, applied by hand

### WIDE hold-green rule (live config `frenzy_wide_hold_green_streak` = 12, DECISION_LOG 228)

**The rule:** WIDE keeps only FRENZY_GREEN_BAR refusals (ATR ≤ 2.5 %) whose `above_streak` is > 12.

**Method:**
- Rebuilt per fill from public Binance 5m klines: 1,499 closed bars ending at the signal bar, plus the 744-bar 1h window for the normal hour.
- Ran `services.frenzy.frenzy_walk`, `frenzy_long_status` and `frenzy_wide_hold_green_block` on those bars.
- **Parity:** the rebuild matched every live stamp on all 10 FRENZY/WIDE fills. That covers ATR, bar return, spike time, hours, VWAP and above_share.

| WIDE fill | ATR % | bar ret % | code | above_streak | today | P&L |
|---|---|---|---|---|---|---|
| AIN 10-03 23:25 | 6.63 | −1.76 | ATR_HIGH | 12 | **blocked** (FRENZY_WIDE_ATR_HIGH) | −3.01 % · −$84.99 |
| MOVR 10-05 09:15 | 2.45 | +1.31 | GREEN_BAR | **35** | kept | +3.00 % · +$84.70 |
| RLC 10-05 12:00 | 2.38 | +0.52 | GREEN_BAR | **17** | kept | +3.00 % · +$87.26 |
| AIN 10-05 16:15 | 4.34 | −0.32 | ATR_HIGH | 12 | **blocked** (ATR_HIGH) | −3.02 % · −$96.85 |
| FLUID 10-06 02:30 | 1.90 | +0.05 | GREEN_BAR | **12** (not > 12) | **blocked** (FRENZY_WIDE_RECLAIM) | −3.04 % · −$100.88 |

**Result:**

| | fills | avg % | $ |
|---|---|---|---|
| WIDE before the rule | 5 | −0.61 % | −$110.48 |
| WIDE after the rule | 2 | +3.00 % | +$171.96 |
| Blocked by the rule | 3 | all −3 % stops | −$282.72 |

This is in-sample anecdote on 5 fills, and two of the three blocks are the same pair. It is not evidence for the rule; the rule rests on the year cohort and its revert gate.

**Side finding:** ORCA FRENZY_LONG (10-06 09:40) rebuilds as FRENZY_VOL_FADED. Its volume multiple is 99.8× with a fresh normal hour, against a bar of 100×. The live engine opened it because its normal-hour cache was a few hours old, which reads 100.2–101.3×. That is a parity quirk of the 6–8 h norm cache, not a bug in this preview.

### Momentum-short pair-vol 0.86 (unchanged)

FET PVR 0.61 and PEPE PVR 0.43 are both below 0.86, so both are kept. There is no change.

### 1500-bar FRENZY window (DECISION_LOG 227) — no decision would have changed

- **The 10 fills:** every one gives identical code, verified status, fresh_on and streak at 1,499 and at 999 closed bars. The oldest spike is SAND at 55 h; the 999-bar window loses verification only past about 58 h.
- **Whole-batch scan:**
  - Pairs: 151 USDT perps with ≥ $8M 24h volume, 47 of them with a spike in range.
  - Bars: every 5m bar from 10-03 23:00 to 10-06 13:25.
  - Result: 1,346 bar-level differences in flag or verified status, on 8 pairs (AIN, LYN, MAGMA, MOVR, NIGHT, QNT, SAND, US).
  - **Zero** of them change an entry decision (READY or a WIDE code). All are "flagged but no entry" vs "unflagged/none". Only 18 bars were entry-eligible in either window, and both windows agree on all of them.
- **What the scan leaves out:** it does not model the shortlist cap, the gvol gate, slots, or the 6–8 h norm cache.
- **Conclusion:** the window fix matters for which pairs show as flagged (AIN past 10-06 00:40, for example), but it did not change any B17 trade.

---

## 4. Batch id and how it would be saved

**Batch id: B17.**

- **Previous archive:** `reports/BASELINE16_batch1002-1003_orders.csv` is the latest. B16 ran 10-02 00:18 → 10-03 21:45 per DECISION_LOG; its last fill was 10-03 15:36.
- **This export:**
  - Ids 1–25 (they restarted after the reset).
  - First fill 10-03 23:25, last close 10-06 12:27 UTC.
  - Everything starts after B16, and the batch matches `B17_BATCH_REVIEW_2026-10-06.md` (ids 1–24).
- **New since the 12:17 export:** one fill only. Its P&L is identical to the earlier export on the 24 shared fills.
  - **id 25 ORDI SPIKE_FADE short**, opened 12:26:45, RUNNER_TRAIL, **+0.27 % · +$64.53**, 2× SPIKE_FADE.
  - 24h volume $11.7M, not liquidity-capped.
  - BTC RSI 45.2 live (48.3 closed); CHOPPY_FLAT; −DI 14.8.
- **Watchlist effect of ORDI:**
  - It is the **first fade in the $10–20M band since the 0.5 % cap** deploy, and a winner.
  - It also falls in the fade BTC-RSI [45, 50) re-revert zone, as a winner, so it cannot push toward the revert.

**Archive filename** (same pattern as `BASELINE16_batch1002-1003_orders.csv`: first-open MMDD to last-open MMDD):
`reports/BASELINE17_batch1003-1006_orders.csv`

**Rows that go in:** follow B15/B16. Those files are the export verbatim (all columns, export row order), filtered to `status == CLOSED` and bot fills only (MANUAL kept apart in its own file).

- B17 has 0 MANUAL and 0 open rows, so all **25 rows** go in.
- The new export carries 6 extra columns, such as `entry_frenzy_above_share`, which the builder unions harmlessly.
- `id` stays in the file; the builder drops it.

**Draft CURRENT_STATE line** (text only, not written):

> **📦 BATCH BOUNDARY 2026-10-06 — B17 (Oct-3 23:25 → Oct-6 12:27 UTC) ARCHIVED `reports/BASELINE17_batch1003-1006_orders.csv`, BOT FILLS ONLY (25 closed · 52% · −0.36% · −$285 as traded: ML 6·67%·+$548 · MS 2·50%·−$68 · FLIP 4·50%·−$458 (QNT/PUMP at 2×) · fades 3/3 +$184 · FRENZY_LONG 5·20%·−$381 · WIDE 5·40%·−$110; UNDER TODAY'S STACK (v2026-10-06b + WIDE hold-green 12) 22·64%·+0.25%·+$691 — flip cells & CALM3D at 1×, WIDE keeps MOVR/RLC only (AIN×2 ATR_HIGH, FLUID streak 12 = reclaim), FRENZY at the +3 TP (SAND 10-04 at the live lock = +2%, −$69)). Ledger TOTAL 292·81%·+$15,377 (builder-only 295·80%·+$15,094; before B17 270·82%·+$14,686). 1500-bar window: 0 B17 decisions changed. 0 MANUAL / 0 probe fills. B18 STARTS at the operator reset.**

Note: CURRENT_STATE has no B16 boundary line (B16 is recorded only in DECISION_LOG); worth adding with this one.

---

## 5. Validator (`scripts/validate_against_master.py`)

- **No path option.** The script has no path argument: it `chdir`s to its own root and reads `reports/MASTER_POOL_stacked.csv`. I ran an unmodified copy from a scratch root whose `reports/` symlinks every real file, except that the master is the scratch build.
- **Without B17:** ALL CHECKS PASS (270 ledger fills).
- **With B17:** **1 FAIL, M1 pct/net sign — SANDUSDT 10-04 05:05 (FRENZY_SLEEVE).**
  - **Cause:** M1 reads `pnl_percentage` (−3.00 %) while the ledger net is the builder's +3 TP re-price (+$206). It is a validator convention gap: for FRENZY +TP rows it should read `stack_pct`, which the builder already writes (+3.00). It is not a data error.
  - **Under the live lock:** the fill is +2.0 %, which is still positive.
  - **Before the real archive:** fix M1 to read `stack_pct`, or decide the FRENZY +TP convention.
- **Every other check passes on the B17 master:** M2, F1, F2 (r 0.914 / 0.897 at shift 0), C1, A1, Y1, X1, PS1 and CB1.

Scratch artefacts (not in the repo): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/b17prev/` holds `master_before.csv`, `master_after.csv`, `b17_per_fill.csv`, `wide_streak.json`, `window_scan.json`, `ledger_*.txt` and `validate_*.log`.
