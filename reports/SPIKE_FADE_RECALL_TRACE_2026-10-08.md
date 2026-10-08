# SPIKE_FADE recall trace: why the yr5 backtest misses live fades (2026-10-08)

**Question.** The H1 review (`reports/SPIKE_FADE_H1_REVIEW_2026-10-08.md`) found three things:
- The yr5 replay reproduces only 32 of 111 live fades.
- Live-only fades won (+0.46) and replay-only fades lost (−0.24).
- So the replay reads fades about 0.4 %/trade worse than live.

This report traces every signal to find why, says which causes can be fixed in the harness, and gives a calibrated fade %/trade for the portfolio study.

**Validation first.** `scripts/validate_against_master.py` → **ALL CHECKS PASS**.

**Scope.** Read-only on code and config. No bot API. No commit.

**New files:**
- `scripts/study_fade_trace_build.py` → `reports/study_fade_trace_matchset.csv`, `reports/study_fade_trace_pairdays.csv`
- `scripts/study_fade_trace_fetch.py` → `reports/study_fade_trace_ticks/` (127 pair-days of real trades, outside the replay cache)
- `scripts/study_fade_trace_eval.py` → `reports/study_fade_trace_signals_raw.csv` (one row per signal × seed)
- `scripts/study_fade_trace_report.py` → **`reports/study_fade_trace_signals.csv`** (one row per signal, final class)

**Binance usage.** Only the public daily aggTrades archives (data.binance.vision): static files, no API weight, fetched serially, cache checked first. No REST calls.

## Plain-English answer

1. **The backtest's fade trigger code is right.**
   - Rebuilt from the replay's own data, 178/178 replay fades fire again at their decision scan.
   - On real trades, 110/111 live fades fire again at the live decision second.
   - The replay's stamped 24 h volume, pair daily −DI and BTC 4h EMA gap match this rebuild.
2. **Problem 1 (biggest, fixable): stale data.** The backtest sees the spike candle up to a minute late.
   - For most small coins the replay cache has no trade-by-trade data. The replay then builds the forming 5-minute candle from finished 1-minute bars only, so it can be up to 59 s stale.
   - 89 of the 111 live fades were in that situation. 4 more had no 1-minute data at all.
   - The fade trigger needs a big RSI jump and a 5× volume burst. Many live spikes meet it for only about 40 s (median 38 s for the fades this missed). A view up to a minute stale misses them.
   - **17 kept live fades were lost this way (live +0.17 %).** Another 3 were lost because the replay's 24 h volume ignores the spike's own volume and fell under the scanner's $2M floor (live +0.61).
3. **Problem 2 (not a bug): different moments in the bar.** Live and backtest look at each coin at different seconds of each 5-minute bar, about every 112 s.
   - 9 kept live fades sat in a trigger window that none of the replay's scan instants hit. They happened to be big winners: +1.13 average, +0.78 without VANRY's +3.9.
   - The reverse also happens: 3 fades per seed that the replay caught and live's own scan clock (journal) missed. Those lost −0.32.
   - This is luck of timing, not an edge. It accounts for a large share of live's high average.
4. **Problem 3 (live was the odd one out, not the backtest): the "replay-only losers" were mostly fades live's rules blocked at the time.**
   - **BTC RSI 45–50 band:** 7.3 fades per seed, −0.26 %. From Aug-5 to Sep-24 live only faded with BTC RSI ≤ 45. Today's config (≤ 50) takes them. Earlier OOS work also found this band loses.
   - **Real-money days, Aug-24 → Aug-27:** 2.7 fades per seed, −0.80 %. Live ran a small real-money book with only 1 fade then.
   - So live paper's +0.37 is **not** today's rule set: it never traded the bRSI 45–50 fades that today's config allows.
5. **Small residuals:**
   - 6 "kept" live fades are actually blocked by today's rules. The master's stack screen keeps them by mistake (FRESHBREAK cannot be recomputed from the stamps before Aug-10; 龙虾 is blacklisted).
   - 1 fade where a gate read differs by a hair (HIVE: pair EMA gap −0.391 vs the −0.40 line).
   - 1 unexplained.
6. **Best estimate of a fade's real per-trade result, today's rules, paper fills: about +0.04 %/trade.**
   - **Window where both ran:** the replay's −0.03 becomes **+0.05** once the stale-data fix is applied (95 % range −0.12…+0.23).
   - **Whole year:** about +0.04 (−0.01 as-is).
   - **Live paper's +0.37** comes from a lucky, rule-different sample. Do not use it for planning.
   - **Real-money stop slippage** (about −0.3 % per stop, roughly 20 % of fades stopped) would take about 0.06 off, which puts real money near zero.
   - **For the 7-strategy study:** base **+0.04 %/trade**, stress **−0.05**, upside **+0.13**.

## 1 · Matched set (Jul-28 → Oct-4, the replay window)

| | count | note |
|---|---|---|
| Live SPIKE_FADE fills (master, CLOSED) | 111 | 66 kept by today's stack (`stack_keep`), 45 blocked |
| Live fills reproduced by ≥ 1 seed (±10 min, same pair) | 32 | 29 kept · live +0.207 vs replay +0.179 on the same signals |
| Replay fades (3 seeds) | 178 | 59.3/seed · 131 while live was up · 78 matched |
| Batch rows missing from the master | 3 | BATCH1 losers XPL Jul-30, ZEREBRO Jul-30, SNX Jul-29 (all −0.70, old stop). Not in the master; not traced. |

**Live-up periods** are the union of the batch spans. The real offline stretches are:
- Aug-10 11:30 → Aug-11 20:00
- Aug-27 16:10 → Sep-11 15:26

Aug-24 21:02 → Aug-27 16:35 was **real money** (B4/B5 `is_paper` False).

## 2 · How each signal was traced

For each signal, both views of the forming 5-minute candle are rebuilt exactly as `code_yr5_181131e/scripts/engine_replay.py::_ohlcv5` does:
- **Replay view:** `ticks_q` (only if the file existed when that chunk ran) → else the 1-minute rebuild from completed 1-minute bars → else the last closed candle.
- **Truth view:** real trades up to the second.

The trigger legs are those of `_spike_scanner_cycle`: RSI12 prev in [35, 55], jump ≥ 25, candle ≥ +0.5 %, volume ≥ 5×.

The router and gates run in engine order: regime/ADX router → MAXVOL → BRSI → BD13 → FRESHBREAK → LAGGARD. They use the BTC readings in the replay's own SCAN journal lines.

**Replay decision instant.** The yr5 fade fills give OPEN − SCAN = **25.3 s + 0.170 s × universe rank** (r = **0.997**, N 210). The scanner walks its candidates in volume-rank order. So the decision second is SCAN + 18.3 + 0.17 × rank (± 4 s), checked on a 3 s grid from 20 to 89 s.

**Other checks:**
- Capacity: the replay journal's pair-tagged BOOK_FULL lines and the replay's own open positions. Note that scanner-path PAIR_HELD / COOLDOWN / NO_BALANCE / GROSS_CAP blocks are journaled **without the pair**.
- Live side: the live decision journal (`~/Downloads/scalpars_decisions_paper_*.csv`, covers Sep-28 →), the live book (batch CSVs) and the live-era config (`config_history_v2`).

**Parity:**
- 178/178 replay fills re-trigger.
- Stamped vs recomputed values agree: 24 h volume 99.4 % within 0.1 %; pair 1d −DI 100 %; BTC 4h EMA50/200 gap 88 % within 0.1 % (median Δ 0).
- 110/111 live fades re-trigger on real ticks at opened_at − 2…40 s.

## 3 · Why the replay missed each kept live fade (37 fades, none reproduced in any seed)

| Family | Fades | WR | live avg % | median trigger window (s) | Fix in harness? |
|---|---|---|---|---|---|
| **A · stale 1-min candle view** (replay saw finished minutes only; real tape triggers at a replay scan second) | 17 | 71 % | +0.166 | 38 | **yes**: real ticks for spike candidates |
| **C · scan instant / phase** (no replay scan second falls inside the trigger window, or only at an off-centre second) | 9 | 100 % | **+1.129** (+0.78 ex-VANRY) | 55 | no: sampling noise; use more seeds |
| **E · today's rules block it** (pre-Aug-10 FRESHBREAK ×5, 龙虾 blacklisted) | 6 | 83 % | +0.226 | 108 | n/a: the replay is right; master screen gap |
| **B · 24 h volume lag** (replay 24 h volume excludes the forming bar; spike volume pushes live over the $2M floor) | 3 | 100 % | +0.607 | 144 | **yes**: live-parity rolling volume |
| D · gate read at a different second (HIVE: pair EMA gap −0.391 vs −0.40) | 1 | 100 % | +0.437 | 213 | no (knife-edge) |
| G · unexplained (RESOLV Oct-3: trigger and gates pass, no open, no journal line) | 1 | 100 % | +0.287 | 18 | — |
| *reproduced by ≥ 1 seed (for reference)* | 29 | 83 % | +0.249 | 124 | |

Notes on the table:
- **Trigger window width separates the families cleanly.**
  - Live fades the replay *matched* stay triggerable for a median 124 s of their bar.
  - The stale-view misses last only 38 s, and the timing misses 55 s.
  - Short, sharp spikes are exactly the ones a stale or discrete view loses.
- **Seed level (kept fades not reproduced in that seed, 127 seed-rows):**
  - stale candle 47, plus closed-bars-only 6
  - off-centre scan second 25
  - today's rules 18 (15 FRESHBREAK-era, 3 blacklist)
  - phase 9
  - volume floor 9
  - gate-input 4, plus gate at off-centre second 2
  - trigger-pass-no-open 6
  - replay already held the pair 1
- **The 45 stack-blocked live fades (−0.34 % live) are correctly absent today.** The replay's own recompute agrees: FRESHBREAK, LAGGARD, MAXVOL, BD13, and the pre-Aug-4 $1M scanner floor (SPELL, FRAX).

## 4 · Why live did not take the replay's extra fades

| Class | per seed | WR | replay avg % | share decided on stale 1-min view | Kind |
|---|---|---|---|---|---|
| matched | 26.0 | 85 % | +0.132 | 86 % | — |
| live offline (Aug-10/11, Aug-27 → Sep-11) | 15.0 | 53 % | −0.512 | 44 % | not comparable |
| **live ran stricter BTC-RSI ≤ 45** (Aug-5 → Sep-24); BTC RSI here 45.4–49.5 | **7.3** | 64 % | **−0.256** | 27 % | **genuine rule difference: today's config takes them** |
| **real-money days** Aug-24 → 27 (small live book, 1 fade) | 2.7 | 38 % | −0.801 | 25 % | genuine live difference |
| live's real scan clock missed the trigger window (journal), or window < 112 s | 3.0 | 44 % | −0.322 | 67 % | timing noise (mirror of family C) |
| unexplained (FF Aug-5 with BTC RSI 49.7; KOMA Aug-14; AVA Sep-23) | 3.0 | 67 % | −0.201 | 56 % | — |
| stale-view artefact (real tape does not trigger at the replay's second) | 2.3 | 86 % | +0.544 | 100 % | **harness**: would vanish with ticks |

- No replay-only fade was refused live for capacity. The live book held ≤ 2 bot positions at those moments; the redeploy ceiling is 10. Live never held the same pair.
- **Effect on the replay's live-up fade average (−0.031):**

  | Remove this class | Replay average becomes |
  |---|---|
  | bRSI 45–50 era fades | +0.014 |
  | Real-money-day fades | +0.019 |
  | Stale artefacts | −0.064 (they were winners) |

## 5 · Ranked causes of the "≈ 0.4 %/trade" gap (live kept +0.365 vs replay −0.031 while both ran)

| # | Cause | Size | Type |
|---|---|---|---|
| 1 | **Live's sample excludes today's bRSI 45–50 fades** (live gate was ≤ 45 for 7 weeks); the replay adds 7.3/seed at −0.26 | explains ≈ 0.05 of the replay side; live's +0.37 is **not** today's rules | genuine rule/era difference |
| 2 | **Stale 1-min candle view** loses 17 short-window live fades (+0.17) and adds 2.3/seed one-minute-stale winners | net fixable Δ ≈ **+0.08** on the replay's window average | **harness fidelity, fixable** |
| 3 | **Scan-phase luck**: live's phase caught 9 big winners (+1.13) that no replay second hit; the replay caught 3/seed live missed (−0.32) | ≈ 0.15 of live's +0.37 is these 9 fills | noise; more seeds, not a fix |
| 4 | Real-money days Aug-24 → 27 | 2.7/seed at −0.80 | genuine live difference |
| 5 | Same-signal pricing (matched 32: live +0.207 vs replay +0.179) | −0.03 to −0.045/trade | known (entry second, stop fill past the line) |
| 6 | 24 h-volume lag at the $2M floor | 3 fades (+0.61) | **harness, fixable** |
| 7 | Master keeps 6 fades today's rules block | small (+0.23 ≈ cohort mean) | master screen gap |

## 6 · Calibrated fade expectancy (today's rules, paper fills, per trade)

**Window where both ran (live-up, Jul-28 → Oct-4):**
- Replay as-is: −0.031 % (day-bootstrap 95 % CI −0.24…+0.22).
- The fix adds back the A + B fades the replay missed: 20.7 per seed, priced at live % minus the same-signal gap (0.045), giving +0.270.
- It drops the 2.3 stale-view artefacts per seed.
- Result: **+0.047 % (95 % CI −0.12…+0.23).**

**Year:**
- yr5 fades decided on the stale 1-min view: 40 % for the year, 70 % in the window.
- Scaling the window Δ (+0.079) by that share moves the year **−0.009 → ≈ +0.036**.
- Fades the replay decided on real ticks: +0.023 for the year (CI −0.06…+0.10), against −0.057 for stale-view-decided fades. In H1: −0.03 vs −0.21.

  ⚠ The tick files exist mainly where earlier replays already filled, so this split is confounded. Read it as direction only.

**Recommended numbers for the 7-strategy portfolio study:**

| Case | %/trade | Note |
|---|---|---|
| **Base** | **+0.04** | |
| Stress | −0.05 | |
| Upside | +0.13 | |
| Real money | ≈ −0.02 | Subtract ≈ 0.06 for stop slippage: scout STOP_SLIP −0.32 per stop × ~20 % stop share |

**Uncertainties:**
- The live sample is N 66, mostly B2 (August).
- The 9 phase-luck winners carry ≈ 40 % of live's edge.
- The "recovered" fades are priced from live outcomes. Only the replay's real-tick re-run can confirm them.
- Treat fades as **≈ break-even-to-slightly-positive, unproven**. They are neither +0.37 nor −0.09.

## 7 · Proposed harness fixes (diffs only, NOT applied; `engine_replay.py` is agent-owned)

### FIX-1: real-trade forming candle for every spike candidate (biggest)

**Option A, data only (no code change).** Fetch `ticks_q` for every pair-day that has a 5-minute bar passing the *necessary* condition for the pump trigger:
- volume ≥ 5× the prior-20 average
- high ≥ +0.5 % over the previous close
- RSI12 prev in [35, 55] and RSI12-at-high − prev ≥ 25
- 24 h volume in [$1.5M, $25M]

Price is maximal at the high and partial volume ≤ final volume, so no instant can trigger unless this holds. Year scan of `k5m_full`: **14,565 candles on 12,650 pair-days; 866 already cached** → about 11.8k archive files. These come from data.binance.vision (no API weight), about 10–15 GB, an overnight serial run with `scripts/backtest_fetch_ticks.py --qty` on a pair-day list.

**Option B, code.** Fetch on demand inside the replay when a candidate candle has no ticks:

```diff
@@ KlineServer._ohlcv5(self, pair, limit, t_ms)
         if A.ticks:
             fc = TS.candle(pair, cur_open, t_ms)
+            if fc is None and A.fetch_spike_ticks and self._spike_candidate(pair, cur_open):
+                TS.fetch_day(pair, (cur_open // DAY) * DAY)        # data.binance.vision daily aggTrades → ticks_q (atomic, cached)
+                fc = TS.candle(pair, cur_open, t_ms)
             if fc is not None:
@@
+    def _spike_candidate(self, pair, cur_open):
+        """necessary condition for a pump trigger in the bar opening at cur_open, read from the CLOSED 5m bar (look-ahead is only
+        used to decide WHICH data to load, never as a signal input): vol ≥ 5× prior-20, high ≥ +0.5 % over prev close."""
+        s = self.get5(pair); i = int(np.searchsorted(s.ts, cur_open))
+        if s is None or i < 21 or i >= len(s) or s.ts[i] != cur_open: return False
+        av = s.v[i - 20:i].mean()
+        return av > 0 and s.v[i] >= 5 * av and s.h[i] >= s.c[i - 1] * 1.005
```

Also count, per run, how many scanner evaluations used each data source (`TICKS` / `1M` / `CLOSED_ONLY`) in the meta JSON. Today this is invisible.

### FIX-2: 24 h volume with the forming bar (live ticker parity)

```diff
@@ Universe._vol24(self, pair, t_ms)
-                i = np.searchsorted(ts, t_ms - 300_000, side="right")   # completed bars only
-                if i >= 288:
-                    return float(cq[i] - cq[i - 288])
+                cur5 = (t_ms // 300_000) * 300_000
+                i = int(np.searchsorted(ts, cur5))                      # all closed bars
+                if i >= 288:
+                    frac = (t_ms - cur5) / 300_000                      # rolling 24 h ends at t: trim the elapsed part of the oldest bar
+                    v = cq[i] - cq[i - 288] - frac * (cq[i - 287] - cq[i - 288])
+                    lo, hi = 1.0e6, 2.6e6                               # only near the scanner floor (cost: tick read per pair)
+                    if lo <= v <= hi or 1.6e7 <= v <= 2.1e7:            # … and near the fade MAXVOL ceiling
+                        v += KS.forming_quote_volume(pair, cur5, t_ms)  # Σ p·q of ticks in [cur5, t) (or 1m qvol fallback)
+                    return float(v)
```

Evidence for FIX-2: MERL Sep-16 live stamp $3.77M vs replay $1.51M; PORTAL $2.00M vs $1.51M; API3 $2.11M vs $1.99M.

### FIX-3: phase noise (no bug)

- Run fade studies on ≥ 6 cadence seeds, or score each fade signal as the average over phases.
- About 15 % of identical signals flip outcome by entry second (H1 review), and 9 fades here decided ≈ 40 % of live's edge.

### Not harness, for the operator

- The master builder's stack screen keeps 5 pre-Aug-10 fades that today's FADE_FRESHBREAK blocks (GRIFFAIN, XAN, FHE, KSM, ZEN).
- It also keeps 1 fade on the now-blacklisted 龙虾USDT. That is 6 of 66 "kept" fades.
- Recompute FRESHBREAK from klines in `build_master_pool.py` (RSI12 prev from the 5-minute closes, pair EMA13/50 gap).

## 8 · What this study could NOT test (blind spots)

- **Live scan instants before Sep-28.** The live journal starts Sep-28, so live-side phase before then is inferred from the trigger-window width (< 112 s ⇒ phase-likely), not observed.
- **Replay scanner-path refusals.** PAIR_HELD, COOLDOWN, GROSS/LIQ cap and NO_BALANCE are journaled without the pair, so they cannot be attributed to one pair. They were checked by time window (none found) and by the replay's own open positions.
- **The exact replay decision second** is modelled (r = 0.997 rank fit), not logged. "Off-centre" (edge) classes carry about ±4 s of uncertainty, and they are timing either way.
- **Recovered fades' outcomes** are live outcomes minus the matched gap. The replay's own exit on them is unknown until FIX-1 is re-run.
- **Funding and real-money slippage** are not modelled. The 3 batch fades missing from the master were not traced.
