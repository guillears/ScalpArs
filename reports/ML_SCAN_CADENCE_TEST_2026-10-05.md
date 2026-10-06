# Does a faster scan cycle improve momentum longs? — yr5 engine-replay test (2026-10-05/06)

## Answer first (plain language)

**No — not worth engineering for momentum longs.** In the yr5 engine replay, a faster cycle makes the bot enter
momentum longs earlier (median 137 s → 104 s after the 5-minute close at a ~31 s cycle). On the *same* signals, though,
entering earlier makes **no measurable difference**: +0.007 %/trade, 95 % CI −0.034…+0.054. What a faster cycle mostly
does is **catch more signals**: +42 % more momentum-long fills at a 30 s cycle. Those extra fills lose money
(−0.14 %/trade). Overall P&L in the sample got slightly worse (−3.1 % Σ per seed over 62 days). A 60 s cycle came out at
zero overall, and its same-signal effect was slightly *negative*.

Both variants fail the pre-registered bar (below). FRENZY is different: its entries come from its own loop, and they did
benefit (reference only, see §5). The reason momentum longs don't: their stops/targets are ~0.7–1 % wide, so 20–30 s of
entry timing is noise next to them. FRENZY's are much tighter.

**Recommendation: do not engineer a faster scan for the momentum sleeves. A full-year run is not needed for this
decision.** The timing effect is small at best: the upper CI on matched fills is +0.05 %/trade. The sleeve sits at −0.07 %/trade
over the full yr5 year (−0.15 in this sample), so even the best case would not fix it. The composition effect points the wrong way. A full year would only make the
"≈0" more precise.

## Pre-registered bar (in `scripts/ml_scan_cadence_test.py` docstring, written before any variant result was read)

Worth engineering only if, for momentum LONG vs the re-run baseline:
① matched same-signal fills (same chunk, seed, pair, 5-min bucket) have mean Δ %/trade > 0 in **both** halves,
② the pooled day-block bootstrap 95 % CI of that matched Δ is entirely > 0,
③ the total effect (Σ % per seed, variant − baseline) is not negative.

| variant | ① matched Δ H1 / H2 | ② pooled matched Δ, 95 % day-CI | ③ total Σ%/seed | verdict |
|---|---|---|---|---|
| c60 (~60 s cycle) | −0.038 / −0.020 ✗ | −0.031 [−0.069, +0.008] ✗ | +0.19 ✓ | **FAIL** |
| c30 (~31 s cycle) | +0.001 / +0.014 ✓ | +0.007 [−0.034, +0.054] ✗ | −3.07 ✗ | **FAIL** |

## 1. What was run (exactly)

- **Code:** a copy of the yr5 snapshot at `reports/backtest_cache/replay/code_cadtest_181131e/`. The agent-owned
  `scripts/engine_replay*.py` and the yr5 snapshot were **not edited**. The copy has one patch, `--multi-scan`, with
  every line marked `CADTEST`:
  - it allows more than one scan per simulated minute;
  - each scan starts at max(scheduled time, previous scan end + 1 s), which is the live `scan_loop` back-to-back
    behaviour.
- **Config:** `frozen_config_yr5_181131e.json`.
- **Args:** all yr5 args unchanged: `--ticks --real-tick --tick-max-pts 0 --hist-days 45 --open-delay-s 6 --frenzy-loop
  --monitor-1hz --follow-hours 24`, warm 3 d, tick_order random, tick_mode stepped, tick_steps 6, `--journal`.
- **Runner and logs:** `reports/backtest_cache/replay/cadtest/run_cadtest.sh`; per-run logs are in `cadtest/`.
- **Sample:** 4 half-month chunks × seeds s1, s2, which is 62 days.

  | chunk | half | ML fills/seed (yr5) |
  |---|---|---|
  | Feb-01→16 | H1 | 46 |
  | Mar-16→Apr-01 | H1 | 37.5 |
  | Jun-01→16 | H2 | 41.5, the most |
  | Aug-16→Sep-01 | H2 | 32 |

- **How `--cadence` / `--scan-step` work in the snapshot:**
  - `--cadence "mean,std,seed"` pre-builds the scan clock. Gaps follow N(mean, std), clipped to [0.9, 1.8]×mean, with a
    random phase from the seed.
  - `--scan-step` is inert whenever `--cadence` is set. It only drives the fixed-step fallback outside a cadence clock and
    the non-tick maker-persistence stand-in, and neither is used here. It was still set to match each cadence.
- **Finding that changed the design (from smoke runs, before any sample result was read):** one replay scan takes
  **65–77 s of simulated time**. That is ≈337 kline calls × 160 ms latency plus 4 × 5 s batch sleeps. So "cadence only"
  cannot go below ~78 s: 60 s and 30 s both collapse to back-to-back ~78 s cycles. A real 60 s or 30 s cycle needs a
  faster scan, which is exactly what engineering it would mean. The fast variants therefore also scale per-call latency:

| variant | --cadence | --scan-step | --latency-ms | measured scan (p50 / p90) | effective cycle | scans per chunk-seed |
|---|---|---|---|---|---|---|
| cbl (baseline) | 111.6,5,seed | 120 | 160 | ~65 / 78 s | 111.6 s | ~14.7 k |
| c60 | 60,2.69,seed | 60 | 100 | 50 / 60 s | ~60 s (17 % back-to-back) | ~25.8 k |
| c30 | 30,1.34,seed | 30 | 25 | 29–31 / 32–34 s | ~31 s (mostly back-to-back) | ~48–53 k |

  The batch sleeps, the 6 s open delay and the 20 s maker window are unchanged.

- **Baseline check:**
  - cbl (patched loop) reproduces the original yr5 runs: 314/314 ML fills matched, Δ +0.0001 %/trade.
  - The back-to-back rule changed nothing at 111.6 s, so **cbl = yr5**.
  - The original harness is deterministic: two identical runs gave identical orders.

## 2. Momentum LONG — headline (per seed, 62 days; pct = fee-net P&L / notional = 1× read)

| variant | N/seed | WR | avg %/trade | Σ %/seed | entry delay after 5m close p25 / p50 / p90 (s) |
|---|---|---|---|---|---|
| cbl (= yr5) | 157.0 | 56.7 % | −0.149 | −23.38 | 75 / 137 / 257 |
| c60 | 183.0 | 60.4 % | −0.127 | −23.19 | 69 / 123 / 255 |
| c30 | 222.5 | 63.1 % | −0.119 | −26.45 | 58 / 104 / 246 |

- Live reference: 131 s median, p25 76, p90 260. The baseline replay's delay matches live.

**Differences vs baseline** (day-block bootstrap, 4,000 draws, days resampled with seeds pooled):

| | Δ avg %/trade [95 % CI] | Δ Σ % per day per seed [95 % CI] | H1 Δ Σ/seed | H2 Δ Σ/seed |
|---|---|---|---|---|
| c60 | +0.022 [−0.049, +0.092] | +0.003 [−0.189, +0.194] | +3.25 | −3.06 |
| c30 | +0.030 [−0.031, +0.096] | −0.050 [−0.255, +0.152] | +0.21 | −3.29 |

- The higher avg %/trade is a mix effect: the baseline's lost fills were worse than the variants' new fills.
- It is not a timing gain: see §3.

**Per seed (N · avg · Σ):**

| variant | s1 | s2 |
|---|---|---|
| cbl | 170 · −0.139 · −23.7 | 144 · −0.160 · −23.1 |
| c60 | 185 · −0.127 · −23.5 | 181 · −0.126 · −22.9 |
| c30 | 219 · −0.104 · −22.7 | 226 · −0.134 · −30.2 |

**Per chunk (Σ %/seed: cbl / c60 / c30):**

| chunk | cbl | c60 | c30 | better in this chunk |
|---|---|---|---|---|
| Feb-01 | −6.4 | −6.4 | −8.5 | baseline |
| Mar-16 | −6.0 | −2.8 | −3.7 | faster |
| Jun-01 | −6.8 | −12.2 | −13.9 | baseline |
| Aug-16 | −4.2 | −1.9 | −0.4 | faster |

- There is no consistent direction across chunks: 2 better, 2 worse.

## 3. Same signal vs new / lost (timing effect vs composition effect)

| | matched / seed | matched Δ %/trade [95 % CI] | matched entry earlier by (median) | new fills / seed (avg) | lost fills / seed (avg) | Σ/seed = timing + composition |
|---|---|---|---|---|---|---|
| c60 | 79 | −0.031 [−0.069, +0.008] | 10 s | 104 (−0.167) | 78 (−0.256) | +0.19 = −2.42 + 2.61 |
| c30 | 91 | +0.007 [−0.034, +0.054] | 24 s | 131.5 (−0.138) | 66 (−0.220) | −3.07 = +0.61 − 3.68 |

- **Matched** = same chunk, seed, pair and 5-min bucket.
- **Matched fills by seed:** c30 s1 −0.014, s2 +0.031; c60 s1 −0.055, s2 −0.004. 95–96 % of matched fills end with
  the same sign.
- **Loose match (same pair within ±5 min, catches a signal taken one bucket earlier):**
  - c30: +0.030 [−0.012, +0.087]. H1 +0.004; H2 +0.060 [+0.003, +0.142].
  - c60: −0.017 [−0.046, +0.016].
  - This is the most favourable read for c30, but it is positive in one half only. It is not the pre-registered match,
    and it would not survive the in-sample haircut.
- **Reading:**
  - Entering the same signal 10–24 s earlier is worth ≈0 for momentum longs.
  - A faster cycle mostly changes which signals get taken. More scans catch more short-lived ("knife-edge")
    forming-candle signals: +42 % fills at c30. That matches the earlier finding that timing-fragile setups lose
    (`ENGINE_REPLAY_YEAR_PLAN.md`: setups seen by 1 of 3 seeds −0.20 %/trade).
  - The new fills are not better than the sleeve: c30 new −0.138 vs sleeve −0.149.

## 4. Momentum SHORT (reference)

| variant | N/seed | WR | avg | Σ/seed | matched Δ [CI] (H1 / H2) | delay p50 |
|---|---|---|---|---|---|---|
| cbl | 59.0 | 58.5 % | −0.108 | −6.36 | — | 117 s |
| c60 | 68.5 | 57.7 % | −0.111 | −7.60 | −0.042 [−0.108, +0.011] (−0.031 / −0.049) | 102 s |
| c30 | 96.5 | 58.0 % | −0.089 | −8.57 | +0.022 [−0.027, +0.079] (−0.040 / +0.067) | 90 s |

- The pattern is the same as for longs: no robust timing gain, more fills, and a worse total.

## 5. FRENZY (reference only — confounded)

FRENZY entries come from the `--frenzy-loop` pass about 4 s after each 5-minute close, not from the scan. The fast
variants also speed up FRENZY's own API calls through the lower latency, so its delay moves from 14 s to 12 s (c60) and
to 11 s (c30).

| | N/seed | avg %/trade | Σ/seed | matched Δ [CI] |
|---|---|---|---|---|
| cbl | 133 | −0.504 | −67.0 | — |
| c60 | 139 | −0.487 | −67.7 | −0.001 [−0.205, +0.214] |
| c30 | 135.5 | −0.421 | −57.0 | **+0.109 [−0.001, +0.289]** |

- c30's matched gain is +0.106 in H1 and +0.112 in H2, and +0.109 on each seed.
- This fits "FRENZY entry delay matters": 3 s faster is worth about +0.1 %/trade there, with the CI just touching zero.
- It is a question about FRENZY-pass latency (orderbook and research-kline calls), not about the scan cycle. If it is
  wanted, test it separately: FRENZY pass with faster calls, scan unchanged.

## 6. Blind spots / what this sample cannot show

1. **Sample, not the full year.** 4 half-month chunks × 2 seeds is 62 days and ~157 ML fills per seed.
   - The sample is harsher than the year: ML −0.149 vs the yr5 year −0.067.
   - CIs on matched Δ are about ±0.04–0.05 %/trade. An effect smaller than that cannot be excluded either way.
   - Two seeds share the same market path, so the seeds are not independent evidence.
2. **The replay's scan is not live's scan.**
   - In the replay, one scan is 65–77 s of simulated work plus idle time up to 111.6 s.
   - Live is ~112 s of real work, back-to-back. Its slow parts (real REST latency, CPU, DB) are not modelled one-to-one.
   - The faster variants were produced by shrinking per-call latency. That is the replay's lever, not a specific
     engineering design (parallel fetches, WebSocket klines, fewer pairs).
   - Binance request-weight limits at 3.5× the REST rate were not modelled.
3. **Within-scan decision timing:** the replay decides after Phase-1 collection (~30 s into a scan). Live may decide at a
   different point in its 112 s.
4. **Path dependence.** Different fills change slots, cooldowns and balance, so the totals mix timing with composition.
   The matched-signal read is the clean timing measure. Matching on the 5-min bucket misses signals taken one bucket
   earlier; the loose match covers that and does not change the conclusion.
5. **Not tested:** a faster cycle plus a persistence gate (the signal must hold on 2 consecutive scans). In principle
   that could keep the earlier entry and drop the fragile extras. The matched timing gain is ≈0 anyway, so the upside is
   bounded.
6. **Live-vs-replay ML recall** is about 64 % (audit Oct-4). The extra replay fills a faster cadence adds are exactly the
   knife-edge cases the replay already handles with lower fidelity.

## 7. Files

- Analysis: `scripts/ml_scan_cadence_test.py` → `reports/ML_SCAN_CADENCE_TEST_fills.csv`. The fills cover yr5, cbl,
  c60 and c30 for the 8 chunk-seeds and the ML, MS and FRENZY sleeves.
- Runs:
  - `reports/backtest_cache/replay/{cadbl,cad60,cad30}_<chunk>_s<seed>_*` (24 runs, all rc=0, 0 tracebacks);
  - journals in `replay/year/journal/<tag>/`;
  - the runner, queue and logs in `replay/cadtest/`.
- Patched code copy: `replay/code_cadtest_181131e/scripts/engine_replay.py`. It differs from the yr5 snapshot only in
  the `CADTEST` lines.
- Discarded: the first launch had a patch bug (the fixed `--scan-step` fallback also fired, doubling scans). Those
  outputs are kept in `replay/cadtest/bug_double_scan/`, and none of them are used here.
- Analysis-script fix applied on the interim read (the bar was not changed): pandas 3 parses `opened_at` to
  microseconds, so the ms, bucket and delay fields are now unit-safe.
