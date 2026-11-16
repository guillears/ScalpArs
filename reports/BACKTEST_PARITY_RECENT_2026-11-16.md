# Backtest parity replay — recent period (2026-11-16)

Operator: "if you run backtest for real trades now, there is no match at all" → "yes run the parity replay".

**Question:** does the engine replay (the backtest) reproduce what the REAL bot did, trade by trade, on the recent period?
**Window:** 2026-10-23 → 2026-11-16 15:46 UTC — B24 LIVE (10-23) · B25 paper · B26 paper · B27 live. 377 real bot fills (MANUAL out, 0 probes).

## Setup
- **Code:** `reports/backtest_cache/replay/code_parity_1116` = commit 47338d16 (today's engine) + the yr6 harness. Today's engine needed
  harness stand-ins it never had: `public_exchange.markets` (Alpha shortlist, DL 464), `get_ohlcv_until` (Flatbase 14-day read, closed
  bars only), exchange ticks for paper resting TPs, funding (not modelled → the DL 566 guard fails open), the DL 492 write lock made a
  no-op + a commit every replay minute (side-session ledgers deadlocked the single-coroutine replay), and — the decisive one — the paper
  exit book (DL 539) priced at the current tick instead of the last 1m close.
- **Config:** trading_config.json at each of the 57 commits since 2026-10-15, switched at commit time (`--config-history`).
- **Data:** caches extended to 2026-11-16 16:00 UTC under the request budget (≈ 77k weight, max used-weight 957/min, no 418/429);
  1m paths for all 132 pairs the real bot traded; mcap proxy = the Oct-8 supply file + 11 pairs from live order stamps.
- **Run:** 4 parallel chunks (3-day warm-up, 24 h follow-through, seed 1, live-like cadence 111.6 ± 5 s, 160 ms latency, 6 s open delay).
- **Match:** same pair + direction, opened within ±10 min. Script: `scripts/parity_compare_1116.py`; rows
  `reports/BACKTEST_PARITY_RECENT_2026-11-16_rows.csv`, table `…_sleeves.csv`.

**First run, discarded:** with the stale-book exit, a WILLY stop triggered at −4 was booked at the previous minute's close (−1.2 … −1.4),
so matched Frenzy trades looked +0.3 … +0.7 pt better than real. After the fix, matched Frenzy P&L sits within ±0.02 pt (median 0).
That bug only affects replays of code from DL 539 (11-13) on; the Oct-13 yr6 snapshot predates it.

## Results (corrected run)
Matched Δ = replay − real booked % on the same trade (mean / median / share where replay is better).

| Sleeve | Real N · WR · avg % | Replay N · WR · avg % | Recall | Precision | Matched Δ mean / median / > 0 | Exit reason agrees | Verdict |
|---|---|---|---|---|---|---|---|
| FRENZY_WILLY | 49 · 55 % · −1.03 | 66 · 61 % · −0.60 | 98 % | 70 % | +0.13 / 0.00 / 35 % | 85 % | **VALID (borderline)** |
| FRENZY_LONG | 21 · 48 % · −1.34 | 31 · 58 % · −0.79 | 95 % | 68 % | −0.01 / 0.00 / 30 % | 90 % | **PARTIAL (≈ valid)** |
| FRENZY_LITE | 43 · 60 % · +0.42 | 63 · 57 % · +0.19 | 91 % | 60 % | −0.02 / 0.00 / 33 % | 100 % | **PARTIAL** — exits exact, too many extra entries |
| LITE_RETRY | 43 · 51 % · −0.07 | 59 · 59 % · +0.40 | 100 % | 73 % | +0.27 / 0.00 / 40 % | 95 % | **PARTIAL** — a few large exit gaps |
| WILLY_HT | 49 · 67 % · +0.19 | 86 · 58 % · −0.09 | 71 % | 43 % | +0.16 / 0.00 / 51 % | 86 % | **PARTIAL** — 49 extra entries |
| Flatbase short | 34 · 62 % · −0.81 | 39 · 72 % · +2.03 | 35 % | 31 % | −0.01 / −0.01 / 33 % | 100 % | **INVALID** (entries) |
| Spike fade | 34 · 76 % · −0.01 | 33 · 27 % · −0.21 | 44 % | 45 % | −0.38 / −0.06 / 33 % | 60 % | **INVALID** |
| Momentum long | 62 · 58 % · −0.04 | 18 · 56 % · −0.04 | 13 % | 44 % | −0.02 / 0.00 / 38 % | 88 % | **INVALID** (replay takes ⅓ the trades) |
| Momentum short | 12 · 67 % · −0.04 | 6 · 50 % · −0.05 | 25 % | 50 % | +0.12 (n = 3) | 67 % | **INVALID** |
| FAN flip | 22 · 64 % · −0.04 | 6 · 83 % · +0.33 | 14 % | 50 % | +0.43 (n = 3) | 67 % | **INVALID** |
| SURGE long | 5 · 0 % · −1.01 | 7 · 0 % · −1.17 | 80 % | 57 % | −0.39 (n = 4) | 75 % | too few |
| Bear-Run | 3 | 0 | 0 % | — | — | — | **INVALID** |
| **Total** | **377 · Σ −91.4 %** | **414 · Σ +28.9 %** | **61 %** | **56 %** | | | |

## What it means
- **Frenzy family (WILLY, LONG, LITE, RETRY):** the replay finds the same signals (recall 91–100 %) and, on the same trade, books the same
  result (median Δ 0.00). Their weakness is **extra entries** (precision 60–73 %; HT 43 %) — trades the live bot did not take, spread
  evenly over the days (not downtime), and they are **profitable** (HT extras +0.25, WILLY +0.34, RETRY +0.94 %/trade). So the
  population-level backtest of these sleeves is **optimistic**: it adds trades live never got. Likely live-only frictions the replay cannot
  model: open-lane lateness (FRENZY_LATE), deploy restarts (spikes joined late), WILLY's turnover read on a live mcap that was missing or
  different (the proxy differs → more HT), book / depth guards (DL 551/565 — the replay's book has no depth).
- **Flatbase, momentum (long and short), FAN flips, spike fades, Bear-Run:** the replay does **not** reproduce the real entries
  (recall 13–44 %). Momentum longs: the replay takes 18 where the bot took 62, and not because of timing (recall stays 16 % at ±6 h; 41 of
  62 real pairs traded by the replay at some point). Suspects, not proven: the BTC market gates (BTC_RSI_ADX_CROSS / BTC_ADX_GATE_LOW block
  ≈ 42k long signals in the replay), forming-candle / maker-window timing, the book filled by the extra Frenzy trades. Flatbase entries
  shift by hours (recall 65 % at ±6 h): its S4 / confirm timing differs. Where Flatbase trades do match, the P&L is exact (Δ −0.01).
- **Matched-trade P&L is not the problem anywhere except fades (−0.38).** The gap is which trades the replay opens.

## Verdict
- **Usable with a stated optimistic bias:** FRENZY_WILLY, FRENZY_LONG (entries + exits reproduce). LITE / RETRY / HT: the same trades
  reproduce exactly, but the backtest adds profitable trades live never took → read their backtest only on the matched cohort or with an
  extras haircut.
- **Not usable as evidence:** momentum long, momentum short, FAN flip, spike fade, Flatbase, Bear-Run — the replay does not reproduce the
  bot's real entries on this period.
- Caveats: one seed, 24 days, B25 (paper) dominates (321 of 377 fills); paper and live fills mixed (live costs are only partly modelled:
  no funding, no depth); B25 ran on code older than the replay's (code-only changes in the window are config-gated except DL 539 / 553,
  handled as above).
