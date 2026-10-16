# WILLY from first principles: what a WILLY trade is, and which of its assumptions hold (2026-10-16)

> **⚠ CORRECTED 2026-10-16 after the adversarial review (DECISION_LOG 304) — read the "Review corrections" section at the END first.**
> The headline "the counter-thesis wins / a WILLY entry is about as good as any bar" does NOT hold on the engine-matched fills (the
> pre-registered verdict cohort): there, WILLY's entry beats a random window bar by **+0.19 /fill** (2026 +0.23 / +0.27). The "2025 −0.32"
> and the TP +2 "+0.14 / +0.11 / +0.10" are driven by rebuilt signals the engine does not take and (for TP +2) by an unequal comparison.

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing was committed and the bot API was not
touched. `backtest_cache` and `scripts/engine_replay*.py` were only read.
- **Pre-registered** in `reports/WILLY_THEORY_PREREG.txt`, frozen **2026-10-16 15:23:32 UTC** with input and code hashes (sha256 of the
  file `ef57c977…`), **before** any rebuilt signal, alternative entry or cell was priced. Operator addendum W6b (pair-specific vs
  market-wide red candle + post-entry market path) and the master/live-side-by-side rule are inside that freeze.
- This study was reviewed adversarially on 2026-10-16 — see "Review corrections" at the end (they supersede the body where they differ).
- Every number is at 1× (net % of notional). The exit is today's: +1.25 resting maker TP, −4 stop, 90-min cap, late TP +0.25 after
  30 min. Stops are booked at min(print, −4) − 0.20 slip.
- Binance public data only: 4,958 + 1,458 kline requests (weight 1–2 each), self-paced, `X-MBX-USED-WEIGHT-1M` read after each call,
  paused at ≥ 900; no 418 / 429; the shared cooldown file was checked first.
- The parallel exhaustive 2D WILLY screen (`reports/WILLY_2D_*`) was not duplicated. This study tests the **mechanism**.

## Plain-English summary

**What WILLY is meant to be.**
1. A coin pumps: at least +5 % in 30 minutes on at least 20× normal volume (a new FRENZY flag).
2. WILLY waits up to 60 minutes for the first red 5-minute candle and buys its close.
3. It aims for +1.25 % and accepts −4 %.

The story is that the first red candle is profit-taking, not the top, and continuation buyers lift the price about 1 % more.

**What WILLY actually is (the data's answer).**
- WILLY is a **volatility bracket on freshly pumped coins**. It does not pick a special moment.
- On a price path with no drift, a +1.25 / −4 bracket wins 4 ÷ 5.25 = **76.2 %** of the time. WILLY wins **77 %**. Its breakeven WR
  after fees and slips is **77.4 %**.
- So a coin-flip path would already give WILLY's hit rate. The +1.25 is hit fast (median **5 minutes**) because these coins swing about
  ±3–4 % within 90 minutes:
  - median best point +3.83 %;
  - median worst point −3.17 %.
- **The stops eat the profits.** Take-profits make Σ +1,506 %-points (66 % of trades). The 19 % of trades that stop lose Σ −1,471.
- The result follows the **market regime, not the entry rule**:
  - 2025: **−0.32 %/trade**;
  - 2026-H1: **+0.05**;
  - 2026-H2: **+0.05**;
  - the bar-exact engine cohort: +0.04 overall.

**Which parts of the thesis hold:**

| # | Assumption | Verdict | Plain meaning |
|---|---|---|---|
| W0 | The selection adds something | **No.** | Against a random bar in the same 60-min window: **+0.02 %/trade (P 0.62)**. Buying the trigger bar with no colour rule: −0.06. Buying the first GREEN bar instead: **−0.09 (P 0.08)**, i.e. the green bar did a little better. Every window bar, on average, *loses* with this bracket (−0.15 / −0.07 / −0.08 by era). |
| W1 | The FIRST red candle is special | **No.** | The 2nd red gives exactly the same result (Δ 0.000). The *other* red bars of the window average **+0.12 vs WILLY's −0.03 (P 0.00 against WILLY)**, so the first red is, if anything, the worst red. Depth of the red bar does not separate either: deep −0.01, shallow +0.01. The DIP_SUB idea is not confirmed. |
| W2 | Who sells on the red bar matters | **No usable signal.** | Red-bar taker-buy share does not separate: < 45 % is +0.01, 50–55 % is −0.14. A red bar with more volume than the spike bar is mildly worse: −0.10 vs +0.08 for < 0.5×. The last-minute sell-off cell is −0.11 (P 0.87). All fail. |
| W3 | The shape of the spike matters | **Mostly no, and one result runs against the theory.** | One-candle spikes are *not* worse (+0.05). Big spikes are not worse. Blow-off volume (≥ 40×) is the **best** bucket (+0.12, positive in all three eras). The weakest qualifying spikes (20–25×) are the worst (−0.21). The upper-wick buckets are non-monotone, which points to a confound. |
| W4 | Timing matters | **Weakly.** | An entry on the trigger bar itself (wait 0) is the best bucket (+0.06); waiting gives −0.06 to −0.10. **EU morning (08–13 UTC) is negative in all three eras** (−0.25 / −0.16 / −0.19, P 0.04). This was a bucket-only read, not a pre-registered block, so it is a watch item only. |
| W5 | Location: holding above the spike VWAP | **Direction yes, gate no.** | Below VWAP −0.10 vs above +0.08. The cohort mean is below zero at P(mean < 0) 0.94, just under the 0.95 bar. It fails walk-forward: in 2026-H1, refusing it would cost −0.03. Retrace depth is non-monotone. |
| W6 | Market support (BTC / breadth), sign first | **No.** | BTC 1h slope ≤ 0: −0.07 vs +0.01, but 2026-H1 runs the opposite way. BTC 15-min, BTC 1h and top-50 breadth all fail. |
| W6b | Is the red candle pair-specific or market-wide? | **Market-wide reds are fine. Coin-specific reds are a little worse, but not significantly.** | FLAT market bar: +0.09 (stop rate 17 %). BETA (BTC red): −0.04 (19 %). IDIO (BTC green while the coin is red): **−0.09 (21 %)**, P(mean < 0) 0.89; fails the confidence bar, and 2026-H2 is flat. When the market is red on the same bar (≥ 60 % of the top-50 red), the result is +0.02, i.e. not worse. |
| W6b-post | What the market does AFTER entry | **Losers are coin-specific dumps, not market moves.** | 15 min after entry, a stopped trade's coin is −2.77 %, while BTC is −0.03 % and the top-50 is −0.10 %. When BTC falls ≥ 0.3 % in the next 15 min the stop rate rises to 26 % (vs 18 %), but that covers only **9 % of all stops**. A BTC-based exit idea could address at most about one stop in ten. |
| W7 | Coin identity | **The most consistent area.** | **Turnover 0.5–1 (inside the live < 1) is the worst bucket: −0.22, and the only cell that passes the locked gate.** Its caveats are below. The cohort the live filter refuses (R ≥ 1) averages −0.18 vs −0.03 kept, which supports the live filter overall (but not in 2025). **Large caps are worse in every era:** ≥ $500M averages −0.34 / −0.07 / −0.33. Previous-day top-50 coins are worse in 2026 (−0.12 / −0.26 vs +0.11 / +0.16). The large-cap side was not the pre-registered block side, so it is a watch item only. |
| W8 | Exit fit: is +1.25 / −4 / 90 near the frontier? | **No. The edge is negative-skew and the entry adds ~nothing over the bracket.** | 45 % of stops are fast crashes (≤ 10 min). 56 % of stopped trades are back above −4 by minute 90, but only 20 % get back to ≥ 0. Across all 140 exit cells, WILLY entries beat random window bars by only **+0.03**. Among deployable exits (stop ≤ 4), **TP +2 / −4 / 90 beats today in all three eras**: +0.14 / +0.11 / +0.10, P 0.98 / 0.97 / 0.91. On the engine cohort the gain is only +0.04. Wider stops look better but are **not deployable at 10×**: the per-order exchange backstop is 4.5 %, and liquidation is ≈ −10 %. |

**Bottom line: the counter-thesis wins.**
- A WILLY entry is about as good as any bar of the same pump window.
- The pump window itself loses with this bracket in every era.
- WILLY's ~77 % WR is the arithmetic of a +1.25 / −4 bracket on a volatile coin, not evidence of continuation buyers.
- Whether WILLY makes money depends on the regime: 2025 negative, 2026 slightly positive. It also depends on the 1-in-5 stop, which
  removes almost all the take-profit gains.
- The only places with consistent structure are **which coin** (turnover, size) and **the exit geometry**, not the red candle.

### Every headline, three blocks side by side (today's exit)

| block | N | WR | avg %/trade | Σ %-pts | $ |
|---|---|---|---|---|---|
| Backtest, rebuilt signals, tick-verdict rows, 2025 | 398 | 70.9 % | −0.318 | −126.6 | – |
| Backtest, 2026-H1 | 936 | 78.1 % | +0.054 | +51.0 | – |
| Backtest, 2026-H2 | 505 | 79.2 % | +0.046 | +23.4 | – |
| Backtest, bar-exact engine cohort (MATCH), all eras | 656 | 78.8 % | +0.041 | +26.9 | – |
| MASTER 19 (B19 14 · B20 3 · B21 2), as traded | 19 | 84 % | +0.491 | +9.33 | +$1,150 |
| MASTER 19, re-priced on today's exit | 19 | 89 % | +0.569 | +10.82 | +$1,203 |
| LIVE 8 (10-15/16), as traded | 8 | 62 % | −0.690 | −5.52 | −$139 |
| LIVE 8, 1m re-price on today's exit | 8 | 75 % | −0.238 | −1.90 | −$42 |

## Recommendation (the operator decides)

1. **Change nothing on WILLY's entry definition.** None of the red-candle, spike-shape, timing, VWAP or BTC cells passes.
   - 1 of 28 block cells passed. The shuffle null expects **1.18** passes by luck (max 4 in 100 runs).
2. **`W7_HIGHTURN` (turnover R ≥ 0.5): observe-only Scout line at most. Do NOT arm.**
   - **What passes.** It meets every locked gate on the pooled tick cohort:
     - 515 fills · WR 73.0 % (below the 77.4 % breakeven) · −0.223 %/fill;
     - P(mean < 0) 0.99 · 311 days;
     - largest day 10 % of the loss, largest pair 19 %;
     - Δ positive in 2025, 2026-H1, 2026-H2 and on the 1m ruler.
   - **Why it is still observe-only.**
     - Its Δ on the bar-exact engine cohort is **+0.004**, essentially zero.
     - 2026 alone is not significant: P 0.85 / 0.76 by half.
     - The null expects about one lucky pass.
   - **Size of the effect.** Refusing it is worth +0.062 %/signal. After the 30–50 % haircut that is **+0.03 to +0.04 %/signal**.
   - **Master:** today total +10.82 → +12.55 %-pts (OGN stop removed; MINA and W TPs lost). In $, +$1,204 → +$1,180, because B19's
     $/pt is larger.
   - **Live:** −1.90 → −0.20 %-pts (ICNT stop removed; RLC and BLUR TPs lost), −$42 → −$9.
   - **Pre-committed tracker gate:**
     - frozen cut R ≥ 0.5 on fills opened after this report;
     - CANDIDATE only at N ≥ 15 blocked fills on ≥ 8 days, with WR < breakeven and day-bootstrap P(mean < 0) ≥ 0.95;
     - no day or pair may carry ≥ 50 % of the loss;
     - the next backtest re-read must show the MATCH Δ > 0;
     - blocked average ≥ kept average → RETIRE.
   - **Revert gate, if it is ever armed:** the first 10 refused signals, re-priced, average > 0 → off.
3. **Watch only, no line yet.** These are post-hoc and each needs a fresh pre-registration:
   - large-cap WILLY (mcap ≥ $500M: −0.26, negative in every era);
   - EU-morning WILLY (08–13 UTC: −0.18, negative in every era);
   - the weakest qualifying spikes (20–25× volume: −0.21).
   - These match the W7 picture: WILLY works on small, high-attention coins and fails on large, liquid ones.
4. **Exit (an input for the existing exit-cell machinery, not a change):** **TP +2.0 / −4 / 90** beat today's +1.25 in all three eras
   on rebuilt signals (P ≥ 0.91 per era).
   - On the bar-exact engine cohort it was only +0.04 better, so apply the haircut.
   - DL 290 already tracks T130–T150 observe-only. A **T200C90** cell is the natural extension, judged under the same frozen bar.
   - This contradicts the yr5 prior that +1.50 < +1.25, so it needs the forward read.

## Parity / validation

- **Episode rebuild.** The engine's `frenzy_walk` + `normal_hour_usd` were run on 448 pairs: 1,991,228 flagged bars, 7,313 episodes.
  - The builder is `study_lite_theory_episodes.build_pair` verbatim, except one declared line: the spike bar itself is judged, because
    WILLY arms on it.
- **Emulator.** It mirrors `_frenzy_willy_eval`:
  - the FRENZY shortlist (|24 h chg| or range ≥ 15 % ∧ ≥ $20M), with the 25-pair cap not emulated;
  - warm / cold arming (cold arms only if ≤ 3 bars late);
  - the first red bar within 60 min;
  - turnover R < 1 on the replay's mcap proxy (bapi supply × price; ratio vs the replay stamp 1.000).
  - Output: 7,172 armed episodes and 4,755 entries.
- **Parity vs replay** (`ENGINE_REPLAY_YR6_2Y_TODAY`, seed 1, 1,518 WILLY fills):
  - **bar-exact 1,244 = 81.9 %** (2025 81.9 / 2026 82.0; seeds 2 and 3 identical). This is **below the 85 % bar-exact gate.**
  - Episode level is **85.8 %**: 59 more fills share the trigger but entered a later red bar, because of the global WILLY hold or a
    pair already held, which the signal level cannot emulate.
  - Remaining misses:
    - 73: the replay triggered 1–3 bars later (a cold shortlist join: the 25-cap / ticker universe);
    - 99: the replay triggered more than 3 bars later;
    - 14: the replay triggered earlier;
    - 29: the episode was not rebuilt.
  - **Declared consequence:** every verdict also had to hold on the bar-exact engine cohort (MATCH). That is why W7_HIGHTURN and the
    TP 2 / 3 exits are downgraded.
- **Feature parity on MATCH:**
  - spike timestamp equal on 99.4 %;
  - wait bars equal on 91 %;
  - BTC 1h slope sign agrees with the stamp on 94.8 % (r 0.98);
  - |Δ vs_vwap| median 0.0003;
  - mcap ratio 1.000.
- **Outcome parity on MATCH vs `WILLY_LOWVWAP_SHORTRUN_rows`** (same walker, the replay's own E):
  - same exit reason on 97.3 %;
  - |Δy| ≤ 0.05 on 95.7 %;
  - my E is 0.036 % above the replay's (the entry-slip ruler);
  - average −0.219 vs −0.195.
- **Master / live:** today's re-price reproduces the LOWVWAP study exactly: master +10.82 pts / +$1,203, live −1.90 / −$42.
- **`scripts/validate_against_master.py`: ALL CHECKS PASS (exit 0).**

## Could NOT test / caveats

- **Signal level, not portfolio.** The global hold, the slot and pair held are not emulated.
  - 2025 signals the engine did not take are clearly worse than the ones it took, on both rulers: tick −0.53 vs −0.03; 1m −0.79 vs −0.55.
  - So 2025 rebuilt-signal numbers overstate the engine's 2025 losses. MATCH is the engine view.
- **Tick coverage is not random.**
  - Only 1,839 of 4,755 entries (39 %) have a fully tick-covered 90-min window. 2025 is thin (398).
  - Within MATCH 2025, tick-covered fills are better on the 1m ruler than uncovered ones (−0.02 vs −0.55).
  - The verdict is on ticks, as required. The 1m ruler (R1) is refute-only and **biased low**: bar-both-lines = stop, entry minute
    skipped. On all rows it reads −0.45 vs −0.03 on ticks. It is used only for the sign of Δ within its own ruler.
- **10 % of window bars have no 1m data.** They are NOENTRY and are excluded on both sides of every comparison.
- **Top-50 definition.** The 2025 `universe_daily_top50.csv` puts every 2025 signal in the top-50, a file artefact. `W7_NOTTOP50` and
  the top-50 table are therefore valid for 2026 only. The breadth measures use the same lists.
- **mcap** is the replay's current-supply × price proxy, not historical circulating supply.
- **Master / live features** come from their stamps plus freshly fetched Binance klines. `spike_volx` is unscored there, and the
  top-50 list is the last available day (10-10) for the 10-12 → 10-16 fills.
- **W8 exit grid.** It is plain (no late TP) and on rebuilt signals. The no-stop / −6 cells ignore liquidation and the exchange
  backstop. No R1 cross-check was run for the grid.
- **Multiple testing.** 28 block cells, plus about 30 bucket tables read two-sided. The shuffle null (100 within-era permutations)
  expects 1.18 false passes. It breaks day-level correlation of outcomes, so it if anything over-counts luck passes.
- **No order-book or open-interest data.** The taker flow is 1m klines only, not trade-by-trade.

## Files / exact commands
```
JOBS=8 venv/bin/python scripts/study_willy_theory_episodes.py      # reports/willy_theory/bars_w.pkl (≈7 min)
venv/bin/python scripts/study_willy_theory_signals.py              # emulator + replay parity → signals.pkl / alt_entries.pkl / parity_rows.csv
JOBS=6 venv/bin/python scripts/study_willy_theory_market.py        # W6 / W6b / post-entry market → market.pkl
venv/bin/python scripts/study_willy_theory_fetch_taker.py          # Binance 1m taker klines (≈4,960 req, weight 1)
venv/bin/python scripts/study_willy_theory_fetch_ctx.py            # master + live context (≈1,460 req)
venv/bin/python scripts/study_willy_theory_prereg.py               # frozen (refuses overwrite)
JOBS=8 venv/bin/python scripts/study_willy_theory_price.py         # RH + R1 + W8 grid → priced.pkl (≈4 min)
venv/bin/python scripts/study_willy_theory_data.py                 # bt / alt / ms frames
NULL_N=100 venv/bin/python scripts/study_willy_theory_report.py    # reports/willy_theory/tables.md, gates.csv, ms_rules.csv, w8_grid.csv
venv/bin/python scripts/validate_against_master.py
```
Every table, including the per-cell master / live before→after in % and $ for all 28 block cells and the per-trade master / live
feature sheet, is in `reports/willy_theory/tables.md` (appended below).

---

# WILLY THEORY — tables (generated by scripts/study_willy_theory_report.py)

BT rebuilt WILLY entries 4,755 · priced (resolved RH) 4,153 · **tick-verdict rows 1,839** · MATCH (bar-exact replay fills, seed 1) 1,244 (tick 656) · R1-resolved 3,891
Entry-source: {'1m': 2313, 'tick': 1842, 'none': 600} · path src {'1m': 2313, 'ticks': 1832, 'ticks+1m': 10}
Sleeve breakeven WR (RH ticks, pooled) **77.4 %**; per era 2025 77.8% · 2026-H1 77.0% · 2026-H2 78.3%

Outcome parity on MATCH vs WILLY_LOWVWAP rows (replay E, same walker): 1244 joined · same exit reason 97.3% · |Δy| ≤ 0.05 on 95.7% · my E / replay E median 1.00036 · avg y this study -0.219 vs LOWVWAP -0.195

## Headline — today's WILLY on rebuilt signals (RH ticks)

| set | era | N | WR | avg % | Σ %-pts | stop share | P(avg>0) day-boot |
|---|---|---|---|---|---|---|---|
| ALL tick | 2025 | 398 | 70.9% | -0.318 | -126.6 | 21.9% | 0.01 |
| ALL tick | 2026-H1 | 936 | 78.1% | +0.054 | +51.0 | 18.2% | 0.80 |
| ALL tick | 2026-H2 | 505 | 79.2% | +0.046 | +23.4 | 18.6% | 0.69 |
| ALL tick | all | 1839 | 76.8% | -0.028 | -52.2 | 19.1% | 0.30 |
| MATCH tick | 2025 | 169 | 77.5% | -0.027 | -4.6 | 18.3% | 0.42 |
| MATCH tick | 2026-H1 | 314 | 78.7% | +0.058 | +18.2 | 19.1% | 0.70 |
| MATCH tick | 2026-H2 | 173 | 80.3% | +0.077 | +13.3 | 16.8% | 0.70 |
| MATCH tick | all | 656 | 78.8% | +0.041 | +26.9 | 18.3% | 0.69 |
| R1 all (1m) | 2025 | 2374 | 66.6% | -0.669 | -1588.4 | 27.9% | 0.00 |
| R1 all (1m) | 2026-H1 | 983 | 74.7% | -0.116 | -114.5 | 21.2% | 0.04 |
| R1 all (1m) | 2026-H2 | 534 | 76.6% | -0.088 | -47.2 | 20.8% | 0.18 |
| R1 all (1m) | all | 3891 | 70.0% | -0.450 | -1750.1 | 25.2% | 0.00 |
| MASTER 19 | as traded | 19 | 84% | +0.491 | +9.33 ($+1,150) | | |
| MASTER 19 | today re-priced | 19 | 89% | +0.569 | +10.82 ($+1,203) | | |
| LIVE 8 | as traded | 8 | 62% | -0.690 | -5.52 ($-139) | | |
| LIVE 8 | today 1m re-price | 8 | 75% | -0.238 | -1.90 ($-42) | | |

## W0 / W1 — does the selection add anything? (paired per episode, RH ticks both sides)

| baseline | era | N episodes | WILLY avg | baseline avg | Δ / episode | P(Δ>0) day-boot | 95 % CI | WILLY WR | baseline WR |
|---|---|---|---|---|---|---|---|---|---|
| TRIG (trigger bar close, no colour rule) | 2025 | 280 | -0.391 | -0.034 | -0.356 | 0.02 | [-0.699, -0.013] | 70% | 78% |
| TRIG (trigger bar close, no colour rule) | 2026-H1 | 601 | +0.043 | -0.071 | +0.114 | 0.84 | [-0.101, +0.319] | 79% | 77% |
| TRIG (trigger bar close, no colour rule) | 2026-H2 | 346 | -0.021 | +0.091 | -0.112 | 0.20 | [-0.377, +0.133] | 78% | 80% |
| TRIG (trigger bar close, no colour rule) | all | 1227 | -0.074 | -0.017 | -0.057 | 0.23 | [-0.218, +0.089] | 77% | 78% |
| FIRST_GREEN (first green bar) | 2025 | 398 | -0.318 | +0.024 | -0.342 | 0.01 | [-0.605, -0.073] | 71% | 78% |
| FIRST_GREEN (first green bar) | 2026-H1 | 936 | +0.054 | +0.009 | +0.045 | 0.71 | [-0.119, +0.210] | 78% | 78% |
| FIRST_GREEN (first green bar) | 2026-H2 | 505 | +0.046 | +0.180 | -0.133 | 0.12 | [-0.353, +0.088] | 79% | 82% |
| FIRST_GREEN (first green bar) | all | 1839 | -0.028 | +0.059 | -0.088 | 0.08 | [-0.211, +0.031] | 77% | 79% |
| RED2 (2nd red bar) | 2025 | 395 | -0.316 | -0.315 | -0.002 | 0.50 | [-0.258, +0.253] | 71% | 71% |
| RED2 (2nd red bar) | 2026-H1 | 933 | +0.051 | +0.069 | -0.018 | 0.43 | [-0.147, +0.107] | 78% | 79% |
| RED2 (2nd red bar) | 2026-H2 | 503 | +0.052 | +0.019 | +0.034 | 0.60 | [-0.209, +0.274] | 79% | 79% |
| RED2 (2nd red bar) | all | 1831 | -0.028 | -0.028 | -0.000 | 0.51 | [-0.107, +0.110] | 77% | 77% |
| ANYRED (mean of the other red bars) | 2025 | 397 | -0.322 | -0.002 | -0.320 | 0.00 | [-0.547, -0.087] | 71% | 77% |
| ANYRED (mean of the other red bars) | 2026-H1 | 933 | +0.051 | +0.174 | -0.123 | 0.03 | [-0.259, +0.007] | 78% | 81% |
| ANYRED (mean of the other red bars) | 2026-H2 | 504 | +0.044 | +0.110 | -0.066 | 0.27 | [-0.283, +0.154] | 79% | 80% |
| ANYRED (mean of the other red bars) | all | 1834 | -0.032 | +0.118 | -0.150 | 0.00 | [-0.250, -0.045] | 77% | 80% |
| RANDOM (mean of every other window bar) | 2025 | 398 | -0.318 | -0.145 | -0.173 | 0.07 | [-0.406, +0.056] | 71% | 75% |
| RANDOM (mean of every other window bar) | 2026-H1 | 936 | +0.054 | +0.007 | +0.048 | 0.79 | [-0.081, +0.171] | 78% | 78% |
| RANDOM (mean of every other window bar) | 2026-H2 | 505 | +0.046 | -0.068 | +0.115 | 0.85 | [-0.095, +0.322] | 79% | 77% |
| RANDOM (mean of every other window bar) | all | 1839 | -0.028 | -0.047 | +0.018 | 0.62 | [-0.079, +0.117] | 77% | 77% |
| LATE_RANDOM (mean of bars after the WILLY bar) | 2025 | 398 | -0.318 | -0.127 | -0.191 | 0.05 | [-0.430, +0.054] | 71% | 75% |
| LATE_RANDOM (mean of bars after the WILLY bar) | 2026-H1 | 935 | +0.053 | +0.006 | +0.047 | 0.78 | [-0.085, +0.174] | 78% | 78% |
| LATE_RANDOM (mean of bars after the WILLY bar) | 2026-H2 | 505 | +0.046 | -0.040 | +0.087 | 0.78 | [-0.131, +0.301] | 79% | 78% |
| LATE_RANDOM (mean of bars after the WILLY bar) | all | 1838 | -0.029 | -0.035 | +0.006 | 0.54 | [-0.096, +0.110] | 77% | 77% |

Unpaired — every window bar of every ARMED episode (incl. the 2,376 that never entered), RH ticks:

| set | era | N bars | avg | WR | stop share |
|---|---|---|---|---|---|
| all window bars | 2025 | 10839 | -0.149 | 75% | 19% |
| all window bars | 2026-H1 | 17723 | -0.068 | 77% | 19% |
| all window bars | 2026-H2 | 10490 | -0.079 | 77% | 20% |
| red bars | 2025 | 5267 | -0.178 | 74% | 19% |
| red bars | 2026-H1 | 8371 | -0.064 | 77% | 18% |
| red bars | 2026-H2 | 5016 | -0.075 | 77% | 19% |
| green bars | 2025 | 5572 | -0.121 | 76% | 19% |
| green bars | 2026-H1 | 9352 | -0.072 | 77% | 19% |
| green bars | 2026-H2 | 5474 | -0.083 | 77% | 21% |
| k = 0 trigger bars | 2025 | 835 | -0.157 | 75% | 21% |
| k = 0 trigger bars | 2026-H1 | 1362 | -0.112 | 75% | 21% |
| k = 0 trigger bars | 2026-H2 | 805 | +0.086 | 79% | 19% |
| WILLY entries | 2025 | 398 | -0.318 | 71% | 22% |
| WILLY entries | 2026-H1 | 936 | +0.054 | 78% | 18% |
| WILLY entries | 2026-H2 | 505 | +0.046 | 79% | 19% |

Zero-drift benchmark: for a driftless price path, P(+1.25 before −4) = 4 / 5.25 = **76.2 %**; the sleeve breakeven WR (fees, slips, the late TP, caps) is 77.4 %. A WR in the high 70s is what a coin-flip path produces with this bracket.

**W1 — depth of the red bar** (`ent_bar_ret`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| DEEP ≤ −2.0 | 70 · 71% · -0.307 | 271 · 77% · +0.041 | 124 · 79% · +0.045 | 465 · 77% · -0.010 | 0.47 | -0.034 | -0.857 |
| MID | 158 · 69% · -0.364 | 324 · 79% · +0.029 | 178 · 78% · -0.018 | 660 · 76% · -0.078 | 0.18 | +0.105 | -0.333 |
| SHALLOW > −0.75 | 170 · 72% · -0.280 | 341 · 78% · +0.089 | 203 · 81% · +0.104 | 714 · 77% · +0.005 | 0.54 | +0.029 | -0.220 |

unscored rows (never counted on either side): 0

**W4 — wait bars from arming to the red bar** (`wait_bars`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| 0 | 118 · 72% · -0.146 | 335 · 77% · +0.074 | 159 · 81% · +0.193 | 612 · 77% · +0.063 | 0.77 | +0.360 | -0.374 |
| 1 | 138 · 74% · -0.270 | 312 · 78% · -0.038 | 169 · 79% · -0.002 | 619 · 77% · -0.080 | 0.19 | -0.080 | -0.435 |
| 2–3 | 112 · 66% · -0.549 | 219 · 80% · +0.163 | 132 · 77% · -0.010 | 463 · 76% · -0.058 | 0.30 | +0.122 | -0.637 |
| ≥ 4 | 30 · 70% · -0.356 | 70 · 79% · +0.030 | 45 · 78% · -0.128 | 145 · 77% · -0.099 | 0.32 | -0.052 | -0.296 |

unscored rows (never counted on either side): 0

**W2 — red-bar volume ÷ spike-bar volume** (`red_vs_spike_q`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 0.5 | 87 · 69% · -0.248 | 204 · 78% · +0.155 | 103 · 82% · +0.204 | 394 · 77% · +0.079 | 0.81 | +0.230 | -0.301 |
| 0.5–1 | 218 · 73% · -0.273 | 388 · 79% · +0.114 | 213 · 78% · -0.016 | 819 · 77% · -0.023 | 0.39 | +0.027 | -0.362 |
| ≥ 1 | 93 · 68% · -0.488 | 344 · 76% · -0.072 | 189 · 79% · +0.031 | 626 · 76% · -0.103 | 0.14 | -0.017 | -0.650 |

unscored rows (never counted on either side): 0

**W2 — red-bar taker-buy share %** (`tbs_red`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 45 | 131 · 73% · -0.267 | 245 · 80% · +0.159 | 171 · 79% · +0.010 | 547 · 78% · +0.010 | 0.54 | +0.063 | -0.536 |
| 45–50 | 174 · 70% · -0.310 | 431 · 77% · +0.043 | 219 · 81% · +0.125 | 824 · 77% · -0.010 | 0.45 | +0.064 | -0.428 |
| 50–55 | 73 · 67% · -0.598 | 230 · 78% · -0.003 | 104 · 76% · -0.115 | 407 · 75% · -0.139 | 0.12 | -0.072 | -0.396 |
| ≥ 55 | 20 · 80% · +0.296 | 30 · 73% · -0.187 | 11 · 82% · +0.578 | 61 · 77% · +0.110 | 0.68 | +0.188 | -0.360 |

unscored rows (never counted on either side): 0

**W2 — last 1m of the red bar taker-buy share %** (`tbs_red_last1m`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 40 | 99 · 69% · -0.449 | 191 · 77% · +0.016 | 120 · 78% · -0.041 | 410 · 75% · -0.113 | 0.13 | -0.046 | -0.561 |
| 40–50 | 141 · 70% · -0.371 | 347 · 78% · +0.027 | 187 · 81% · +0.161 | 675 · 77% · -0.019 | 0.42 | +0.151 | -0.503 |
| 50–60 | 117 · 74% · -0.152 | 290 · 81% · +0.189 | 141 · 77% · -0.044 | 548 · 78% · +0.056 | 0.75 | -0.008 | -0.343 |
| ≥ 60 | 41 · 73% · -0.290 | 108 · 73% · -0.150 | 57 · 81% · +0.077 | 206 · 75% · -0.115 | 0.23 | -0.103 | -0.358 |

unscored rows (never counted on either side): 0

**W2 — spike-bar taker-buy share %** (`tbs_spk`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 50 | 117 · 74% · -0.130 | 277 · 78% · +0.076 | 167 · 80% · +0.059 | 561 · 78% · +0.028 | 0.63 | +0.026 | -0.519 |
| 50–55 | 152 · 76% · -0.111 | 417 · 79% · +0.102 | 213 · 77% · -0.075 | 782 · 78% · +0.012 | 0.58 | +0.028 | -0.381 |
| 55–60 | 92 · 58% · -0.935 | 180 · 76% · -0.056 | 97 · 85% · +0.353 | 369 · 74% · -0.168 | 0.07 | -0.007 | -0.455 |
| ≥ 60 | 37 · 70% · -0.227 | 62 · 74% · -0.042 | 28 · 79% · -0.169 | 127 · 74% · -0.124 | 0.24 | +0.294 | -0.532 |

unscored rows (never counted on either side): 0

**W3 — spike shape** (`spike_conc`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| staircase ≤ 0.4 | 145 · 70% · -0.401 | 312 · 75% · -0.066 | 158 · 82% · +0.158 | 615 · 75% · -0.087 | 0.16 | -0.060 | -0.402 |
| mixed | 184 · 72% · -0.252 | 364 · 80% · +0.103 | 222 · 77% · -0.045 | 770 · 77% · -0.024 | 0.38 | +0.187 | -0.456 |
| one-candle ≥ 0.6 | 69 · 70% · -0.320 | 260 · 79% · +0.131 | 125 · 80% · +0.068 | 454 · 78% · +0.045 | 0.65 | -0.069 | -0.490 |

unscored rows (never counted on either side): 0

**W3 — spike-bar upper wick share** (`spike_uwick`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 0.15 | 149 · 68% · -0.373 | 334 · 76% · -0.065 | 173 · 79% · +0.024 | 656 · 75% · -0.112 | 0.11 | -0.126 | -0.389 |
| 0.15–0.4 | 161 · 74% · -0.171 | 413 · 80% · +0.196 | 211 · 81% · +0.139 | 785 · 79% · +0.105 | 0.92 | +0.123 | -0.412 |
| ≥ 0.4 | 88 · 69% · -0.494 | 189 · 77% · -0.044 | 121 · 77% · -0.082 | 398 · 75% · -0.155 | 0.11 | +0.179 | -0.596 |

unscored rows (never counted on either side): 0

**W3 — spike size (30-min %)** (`spike_ret6`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| 5–6.5 | 202 · 73% · -0.181 | 403 · 79% · +0.130 | 233 · 82% · +0.172 | 838 · 78% · +0.067 | 0.83 | +0.156 | -0.158 |
| 6.5–10 | 140 · 70% · -0.351 | 308 · 75% · -0.116 | 166 · 77% · -0.077 | 614 · 74% · -0.159 | 0.03 | -0.078 | -0.446 |
| ≥ 10 | 56 · 64% · -0.732 | 225 · 80% · +0.153 | 106 · 77% · -0.035 | 387 · 77% · -0.027 | 0.46 | -0.117 | -0.893 |

unscored rows (never counted on either side): 0

**W3 — spike volume multiple** (`spike_volx`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| 20–25× | 193 · 66% · -0.565 | 325 · 77% · -0.057 | 150 · 78% · -0.078 | 668 · 74% · -0.208 | 0.01 | -0.175 | -0.343 |
| 25–40× | 142 · 73% · -0.168 | 299 · 78% · +0.067 | 149 · 81% · +0.137 | 590 · 77% · +0.028 | 0.63 | +0.133 | -0.561 |
| ≥ 40× | 63 · 81% · +0.099 | 312 · 80% · +0.158 | 206 · 79% · +0.072 | 581 · 80% · +0.121 | 0.91 | +0.234 | -0.453 |

unscored rows (never counted on either side): 0

**W3 — entry vs the spike high %** (`off_high`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ −5 | 73 · 63% · -0.584 | 208 · 75% · -0.012 | 91 · 84% · +0.284 | 372 · 75% · -0.052 | 0.34 | -0.043 | -1.004 |
| −5…−2.5 | 106 · 69% · -0.343 | 319 · 79% · +0.056 | 165 · 76% · -0.130 | 590 · 76% · -0.068 | 0.22 | +0.063 | -0.360 |
| > −2.5 | 219 · 74% · -0.217 | 409 · 79% · +0.087 | 249 · 80% · +0.077 | 877 · 78% · +0.008 | 0.54 | +0.048 | -0.191 |

unscored rows (never counted on either side): 0

**W4 — session (UTC hour of the red bar)** (`hour`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| Asia 0–8 | 132 · 74% · -0.110 | 418 · 79% · +0.092 | 176 · 84% · +0.285 | 726 · 79% · +0.102 | 0.89 | +0.175 | -0.210 |
| EU 8–13 | 82 · 74% · -0.248 | 215 · 74% · -0.155 | 139 · 75% · -0.190 | 436 · 75% · -0.184 | 0.04 | -0.052 | -0.350 |
| US 13–21 | 141 · 70% · -0.426 | 277 · 80% · +0.175 | 176 · 78% · -0.022 | 594 · 77% · -0.026 | 0.41 | -0.023 | -0.424 |
| late 21–24 | 43 · 58% · -0.735 | 26 · 73% · -0.100 | 14 · 86% · +0.256 | 83 · 67% · -0.369 | 0.13 | +0.025 | -1.460 |

unscored rows (never counted on either side): 0

**W5 — entry vs spike VWAP (sign)** (`vs_vwap`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 (below VWAP) | 261 · 68% · -0.424 | 574 · 78% · +0.048 | 275 · 76% · -0.100 | 1110 · 75% · -0.100 | 0.07 | -0.007 | -0.547 |
| > 0 | 137 · 76% · -0.116 | 362 · 78% · +0.065 | 230 · 83% · +0.221 | 729 · 79% · +0.080 | 0.86 | +0.116 | -0.276 |

unscored rows (never counted on either side): 0

**W5 — share of the pump retraced** (`retrace_frac`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 0.2 | 195 · 76% · -0.185 | 396 · 80% · +0.124 | 245 · 80% · +0.111 | 836 · 79% · +0.048 | 0.75 | +0.093 | -0.326 |
| 0.2–0.35 | 128 · 63% · -0.625 | 320 · 76% · -0.111 | 166 · 76% · -0.118 | 614 · 73% · -0.220 | 0.01 | -0.029 | -0.530 |
| ≥ 0.35 | 75 · 71% · -0.139 | 220 · 79% · +0.170 | 94 · 82% · +0.169 | 389 · 78% · +0.110 | 0.83 | +0.028 | -0.556 |

unscored rows (never counted on either side): 0

**W5 — entry vs pre-spike base %** (`vs_base_pct`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 8 | 150 · 72% · -0.238 | 310 · 77% · +0.109 | 135 · 83% · +0.204 | 595 · 77% · +0.043 | 0.70 | +0.087 | -0.334 |
| 8–12 | 124 · 69% · -0.410 | 278 · 80% · +0.107 | 192 · 76% · -0.098 | 594 · 76% · -0.067 | 0.21 | +0.145 | -0.266 |
| ≥ 12 | 124 · 72% · -0.324 | 348 · 77% · -0.036 | 178 · 80% · +0.083 | 650 · 77% · -0.058 | 0.28 | -0.123 | -0.688 |

unscored rows (never counted on either side): 0

**W6 — BTC 1h EMA20 slope (sign)** (`btc_1h_slope`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 173 · 69% · -0.393 | 494 · 79% · +0.090 | 206 · 75% · -0.190 | 873 · 76% · -0.072 | 0.17 | -0.075 | -0.694 |
| > 0 | 225 · 72% · -0.261 | 442 · 77% · +0.014 | 299 · 82% · +0.209 | 966 · 77% · +0.011 | 0.56 | +0.128 | -0.222 |

unscored rows (never counted on either side): 0

**W6 — BTC 15-min return (sign)** (`btc_15m`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 174 · 72% · -0.154 | 457 · 78% · +0.047 | 242 · 81% · +0.154 | 873 · 78% · +0.037 | 0.72 | +0.241 | -0.307 |
| > 0 | 224 · 70% · -0.446 | 479 · 78% · +0.061 | 263 · 78% · -0.053 | 966 · 76% · -0.087 | 0.12 | -0.138 | -0.574 |

unscored rows (never counted on either side): 0

**W6 — BTC 1-hour return (sign)** (`btc_1h`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 174 · 67% · -0.476 | 463 · 80% · +0.170 | 222 · 78% · -0.004 | 859 · 77% · -0.006 | 0.49 | +0.184 | -0.581 |
| > 0 | 224 · 74% · -0.195 | 473 · 76% · -0.058 | 283 · 80% · +0.086 | 980 · 77% · -0.048 | 0.24 | -0.072 | -0.329 |

unscored rows (never counted on either side): 0

**W6b — BTC SAME 5m bar (sign)** (`btc_bar`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 239 · 69% · -0.377 | 511 · 79% · +0.107 | 266 · 78% · -0.058 | 1016 · 77% · -0.050 | 0.23 | +0.051 | -0.537 |
| > 0 | 159 · 74% · -0.229 | 425 · 77% · -0.009 | 239 · 80% · +0.163 | 823 · 77% · -0.002 | 0.50 | +0.028 | -0.356 |

unscored rows (never counted on either side): 0

**W6b — ETH same bar (sign)** (`eth_bar`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 235 · 69% · -0.392 | 485 · 79% · +0.062 | 245 · 80% · +0.076 | 965 · 77% · -0.045 | 0.26 | +0.011 | -0.525 |
| > 0 | 163 · 74% · -0.211 | 451 · 77% · +0.047 | 260 · 78% · +0.018 | 874 · 77% · -0.010 | 0.47 | +0.075 | -0.372 |

unscored rows (never counted on either side): 0

**W6b — SOL same bar (sign)** (`sol_bar`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 0 | 241 · 68% · -0.456 | 503 · 80% · +0.141 | 251 · 82% · +0.136 | 995 · 77% · -0.005 | 0.48 | +0.025 | -0.551 |
| > 0 | 157 · 76% · -0.107 | 433 · 76% · -0.046 | 254 · 77% · -0.042 | 844 · 76% · -0.056 | 0.19 | +0.061 | -0.334 |

unscored rows (never counted on either side): 0

**W6b — class of the red candle** (`btc_bar`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| (ii) BETA: BTC red < −0.02 | 193 · 70% · -0.321 | 402 · 79% · +0.091 | 196 · 79% · -0.012 | 791 · 77% · -0.035 | 0.33 | -0.002 | -0.585 |
| (iii) FLAT | 81 · 70% · -0.224 | 194 · 79% · +0.166 | 138 · 81% · +0.153 | 413 · 78% · +0.085 | 0.81 | +0.238 | -0.201 |
| (i) IDIO: BTC green > +0.02 | 124 · 72% · -0.375 | 340 · 76% · -0.052 | 171 · 78% · +0.027 | 635 · 76% · -0.094 | 0.11 | -0.031 | -0.439 |

unscored rows (never counted on either side): 0

**W6b — top-50 share red on the same bar** (`brd_red`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| ≤ 40 % red (market green) | 141 · 71% · -0.341 | 376 · 78% · +0.040 | 197 · 78% · +0.002 | 714 · 76% · -0.046 | 0.28 | +0.004 | -0.412 |
| 40–60 | 66 · 73% · -0.202 | 242 · 75% · -0.099 | 140 · 80% · +0.053 | 448 · 76% · -0.067 | 0.25 | -0.008 | -0.256 |
| ≥ 60 % red (market red) | 191 · 70% · -0.341 | 318 · 81% · +0.189 | 168 · 80% · +0.094 | 677 · 78% · +0.016 | 0.58 | +0.109 | -0.592 |

unscored rows (never counted on either side): 0

**W6 — top-50 breadth over 1 h** (`brd_1h`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 50 % up 1h | 183 · 70% · -0.374 | 513 · 80% · +0.139 | 239 · 78% · -0.008 | 935 · 78% · +0.001 | 0.53 | +0.109 | -0.530 |
| ≥ 50 | 215 · 72% · -0.270 | 423 · 76% · -0.048 | 266 · 80% · +0.095 | 904 · 76% · -0.059 | 0.20 | -0.021 | -0.363 |

unscored rows (never counted on either side): 0

**W6b — market (BTC+ETH+SOL) 1m taker-buy share in the bar** (`mkt_tbs`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 48 | 207 · 69% · -0.376 | 424 · 79% · +0.076 | 221 · 80% · +0.083 | 852 · 77% · -0.032 | 0.32 | +0.029 | -0.540 |
| 48–52 | 59 · 64% · -0.574 | 122 · 76% · -0.019 | 68 · 79% · +0.091 | 249 · 74% · -0.121 | 0.22 | +0.252 | -0.428 |
| ≥ 52 | 132 · 77% · -0.113 | 390 · 78% · +0.054 | 216 · 78% · -0.006 | 738 · 78% · +0.007 | 0.53 | -0.026 | -0.348 |

unscored rows (never counted on either side): 0

**W7 — turnover vol/mcap (live < 1)** (`R_ent`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 0.2 | 116 · 72% · -0.190 | 342 · 76% · -0.026 | 181 · 79% · +0.051 | 639 · 76% · -0.034 | 0.37 | -0.094 | -0.488 |
| 0.2–0.5 | 154 · 73% · -0.281 | 345 · 83% · +0.281 | 186 · 82% · +0.164 | 685 · 80% · +0.123 | 0.94 | +0.184 | -0.401 |
| 0.5–1 | 128 · 67% · -0.479 | 249 · 74% · -0.149 | 138 · 76% · -0.118 | 515 · 73% · -0.223 | 0.01 | -0.012 | -0.447 |

unscored rows (never counted on either side): 0

**W7 — market cap (supply proxy × price)** (`mcap`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < $100M | 26 · 58% · -0.867 | 530 · 80% · +0.122 | 273 · 80% · +0.091 | 829 · 79% · +0.081 | 0.87 | +0.116 | -0.421 |
| $100–500M | 156 · 73% · -0.199 | 293 · 77% · -0.019 | 152 · 82% · +0.162 | 601 · 77% · -0.020 | 0.42 | +0.058 | -0.474 |
| ≥ $500M | 216 · 71% · -0.338 | 113 · 74% · -0.072 | 80 · 72% · -0.326 | 409 · 72% · -0.262 | 0.01 | -0.077 | -0.473 |

unscored rows (never counted on either side): 0

**W7 — previous-day top-50** (`t50`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| not top-50 | 0 | 698 · 79% · +0.113 | 371 · 81% · +0.156 | 1069 · 80% · +0.128 | 0.99 | +0.186 | -0.482 |
| top-50 | 398 · 71% · -0.318 | 238 · 74% · -0.117 | 134 · 74% · -0.258 | 770 · 72% · -0.245 | 0.00 | -0.076 | -0.311 |

unscored rows (never counted on either side): 0

**W7 — listing age** (`listing_days`; RH tick rows; per era N · WR · avg %; ALL day-boot P(avg>0))

| bucket | 2025 N · WR · avg | 2026-H1 N · WR · avg | 2026-H2 N · WR · avg | ALL N · WR · avg | P(avg>0) | MATCH avg | R1 avg (all rows) |
|---|---|---|---|---|---|---|---|
| < 180 d | 128 · 70% · -0.438 | 78 · 81% · +0.200 | 33 · 67% · -0.689 | 239 · 73% · -0.264 | 0.03 | +0.141 | -0.516 |
| 180–365 | 75 · 65% · -0.556 | 307 · 76% · -0.065 | 71 · 86% · +0.372 | 453 · 76% · -0.078 | 0.23 | -0.108 | -0.401 |
| ≥ 365 | 195 · 74% · -0.148 | 551 · 79% · +0.100 | 401 · 79% · +0.049 | 1147 · 78% · +0.040 | 0.74 | +0.089 | -0.447 |

unscored rows (never counted on either side): 0

### W6b — die rate per class (stop share) vs breakeven, RH ticks

| class | era | N | stop share | TP share | avg | WR vs BE |
|---|---|---|---|---|---|---|
| BETA | 2025 | 193 | 22.8% | 70.5% | -0.321 | 70.5% vs 77.4% |
| BETA | 2026-H1 | 402 | 17.2% | 79.4% | +0.091 | 79.4% vs 77.4% |
| BETA | 2026-H2 | 196 | 18.4% | 79.1% | -0.012 | 79.1% vs 77.4% |
| BETA | all | 791 | 18.8% | 77.1% | -0.035 | 77.1% vs 77.4% |
| FLAT | 2025 | 81 | 17.3% | 70.4% | -0.224 | 70.4% vs 77.4% |
| FLAT | 2026-H1 | 194 | 17.0% | 79.4% | +0.166 | 79.4% vs 77.4% |
| FLAT | 2026-H2 | 138 | 17.4% | 81.2% | +0.153 | 81.2% vs 77.4% |
| FLAT | all | 413 | 17.2% | 78.2% | +0.085 | 78.2% vs 77.4% |
| IDIO | 2025 | 124 | 23.4% | 71.8% | -0.375 | 71.8% vs 77.4% |
| IDIO | 2026-H1 | 340 | 20.0% | 75.9% | -0.052 | 75.9% vs 77.4% |
| IDIO | 2026-H2 | 171 | 19.9% | 77.8% | +0.027 | 77.8% vs 77.4% |
| IDIO | all | 635 | 20.6% | 75.6% | -0.094 | 75.6% vs 77.4% |

### W6b POST-ENTRY (descriptive — after entry, never a filter): market path by outcome, RH ticks

| outcome | N | BTC +5 min | BTC +15 | BTC +30 | top-50 +5 | +15 | +30 | coin +5 | coin +15 | coin +30 |
|---|---|---|---|---|---|---|---|---|---|---|
| TP | 1209 | +0.006 | +0.021 | +0.050 | +0.015 | +0.014 | +0.059 | +0.679 | +1.406 | +1.821 |
| LATE_TP | 204 | -0.003 | -0.007 | +0.025 | -0.006 | -0.035 | -0.012 | -0.557 | -0.818 | -0.679 |
| STOP | 351 | -0.027 | -0.025 | -0.095 | -0.058 | -0.102 | -0.364 | -1.926 | -2.767 | -3.317 |
| CAP | 75 | -0.021 | -0.015 | -0.039 | -0.029 | -0.017 | -0.061 | -0.582 | -1.039 | -1.471 |

Stop rate by the BTC move in the 15 min AFTER entry (descriptive):

| BTC next 15 min | N | stop share | avg | stops' share of all stops |
|---|---|---|---|---|
| ≤ −0.3 | 117 | 25.6% | -0.410 | 9% |
| −0.3…−0.1 | 316 | 20.3% | -0.120 | 18% |
| flat ±0.1 | 920 | 18.4% | +0.007 | 48% |
| 0.1…0.3 | 355 | 18.9% | +0.005 | 19% |
| ≥ 0.3 | 131 | 16.0% | +0.194 | 6% |

Stopped trades: coin −2.77 % at +15 min vs BTC -0.025 % and top-50 -0.102 % — share of stops with BTC down > 0.3 % in the next 30 min: 15% (all entries 10%).

## W7 — the live turnover filter: what it refuses (first red bar with R ≥ 1, priced as an entry), RH ticks

| set | era | N | WR | avg | P(avg>0) |
|---|---|---|---|---|---|
| refused R ≥ 1 (first red) | 2025 | 294 | 76% | -0.122 | 0.15 |
| refused R ≥ 1 (first red) | 2026-H1 | 292 | 69% | -0.368 | 0.00 |
| refused R ≥ 1 (first red) | 2026-H2 | 250 | 77% | -0.018 | 0.47 |
| refused R ≥ 1 (first red) | all | 836 | 74% | -0.177 | 0.01 |
| kept (WILLY entries) | 2025 | 398 | 71% | -0.318 | 0.01 |
| kept (WILLY entries) | 2026-H1 | 936 | 78% | +0.054 | 0.82 |
| kept (WILLY entries) | 2026-H2 | 505 | 79% | +0.046 | 0.68 |
| kept (WILLY entries) | all | 1839 | 77% | -0.028 | 0.30 |

## W8 — exit fit (WILLY entries; RH path fully tick-covered for 240 min)

N 1832 · MFE90 median +3.83 · MAE90 median -3.17 · reached +1.25 within 90 min 79% (median 5.2 min) · touched −4 within 90 40%
Stops: 349 · fast crash (≤ 10 min) 45% · 10–30 min 33% · > 30 min 22% · stopped trades' 90-min close (no stop) median -3.48 / mean -3.01 (recovered above −4: 56%, back to ≥ 0: 20%)

P&L decomposition (today's exit): TP 66% of trades, Σ +1506 · LATE_TP 11% of trades, Σ +51 · STOP 19% of trades, Σ -1471 · CAP 4% of trades, Σ -135

Exit grid — avg % per trade by era (plain TP / SL / cap, no late TP; 'none' = no stop); WILLY entries vs the RANDOM window bar (mean of every other window bar of the same episodes, tick-covered):

TODAY (+1.25 / −4 / 90 + late TP): all -0.027 · 2025 -0.310 · 2026-H1 +0.054 · 2026-H2 +0.044

Top 12 cells by the WORST era (robustness), and the 1.25 / 4 / 90 plain reference:

| TP | SL | cap | 2025 | 2026-H1 | 2026-H2 | all | worst era | paired WILLY | paired RANDOM bar | WILLY − RANDOM |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | none | 90 | +0.104 | +0.474 | +0.439 | +0.385 | +0.104 | +0.385 | +0.239 | +0.145 |
| 3 | none | 240 | +0.071 | +0.769 | +0.806 | +0.630 | +0.071 | +0.630 | +0.451 | +0.179 |
| 2 | none | 90 | +0.030 | +0.361 | +0.407 | +0.303 | +0.030 | +0.303 | +0.149 | +0.153 |
| 3 | none | 60 | +0.030 | +0.431 | +0.420 | +0.342 | +0.030 | +0.342 | +0.184 | +0.158 |
| 3 | 3 | 240 | -0.017 | +0.185 | +0.046 | +0.103 | -0.017 | +0.103 | -0.008 | +0.112 |
| 2 | none | 240 | -0.020 | +0.547 | +0.639 | +0.451 | -0.020 | +0.451 | +0.292 | +0.159 |
| 2 | none | 60 | -0.031 | +0.336 | +0.359 | +0.263 | -0.031 | +0.263 | +0.119 | +0.144 |
| 3 | 6 | 240 | -0.046 | +0.455 | +0.422 | +0.338 | -0.046 | +0.338 | +0.241 | +0.097 |
| 3 | 2 | 30 | -0.065 | -0.068 | -0.040 | -0.060 | -0.068 | -0.060 | -0.131 | +0.071 |
| 3 | 2 | 240 | -0.020 | -0.039 | -0.086 | -0.048 | -0.086 | -0.048 | -0.128 | +0.081 |
| 2 | 3 | 240 | -0.087 | +0.088 | +0.057 | +0.042 | -0.087 | +0.042 | -0.052 | +0.094 |
| 3 | 2 | 60 | +0.005 | -0.074 | -0.087 | -0.061 | -0.087 | -0.061 | -0.132 | +0.071 |
| 1.25 | 4 | 90 | -0.318 | +0.073 | +0.025 | -0.024 | -0.318 | -0.024 | -0.027 | +0.003 |

Cells beating TODAY in all three eras: 45 of 140: 0.5/0/240, 0.75/0/30, 0.75/0/60, 0.75/0/90, 0.75/0/240, 1/0/30, 1/0/60, 1/0/90, 1/0/240, 1.25/0/30, 1.25/0/60, 1.25/0/90, 1.25/0/240, 1.5/0/60, 1.5/0/90, 1.5/0/240, 2/3/60, 2/3/90, 2/3/240, 2/4/30, 2/4/60, 2/4/90, 2/4/240, 2/6/30, 2/6/60, 2/6/90, 2/6/240, 2/0/30, 2/0/60, 2/0/90, 2/0/240, 3/3/30, 3/3/240, 3/4/30, 3/4/60, 3/4/90, 3/4/240, 3/6/30, 3/6/60, 3/6/90, 3/6/240, 3/0/30, 3/0/60, 3/0/90, 3/0/240
Share of the 140 cells with a positive average in ALL three eras: 3% · WILLY − RANDOM averaged over all cells: +0.028

**Deployable cells only** (live per-order exchange backstop = stop + 0.5 ≤ 4.5 % at 10× and liquidation ≈ −10 % → SL 6 / none are NOT deployable at 10×). Paired Δ vs TODAY per trade (same entries), day-bootstrap P(Δ>0) per era; MATCH = bar-exact engine fills:

| TP | SL | cap | Δ 2025 (P) | Δ 2026-H1 (P) | Δ 2026-H2 (P) | Δ all | Δ MATCH | avg all |
|---|---|---|---|---|---|---|---|---|
| 3 | 4 | 240 | +0.210 (0.95) | +0.211 (0.99) | +0.090 (0.77) | +0.177 | -0.062 | +0.151 |
| 3 | 4 | 90 | +0.187 (0.96) | +0.163 (0.96) | +0.056 (0.73) | +0.138 | -0.077 | +0.112 |
| 3 | 3 | 240 | +0.293 (0.99) | +0.131 (0.93) | +0.002 (0.52) | +0.130 | -0.048 | +0.103 |
| 3 | 4 | 60 | +0.155 (0.95) | +0.128 (0.95) | +0.103 (0.85) | +0.127 | -0.061 | +0.100 |
| 2 | 4 | 240 | +0.159 (0.98) | +0.103 (0.95) | +0.126 (0.94) | +0.121 | +0.015 | +0.094 |
| 2 | 4 | 90 | +0.139 (0.98) | +0.108 (0.97) | +0.102 (0.91) | +0.113 | +0.042 | +0.086 |
| 3 | 4 | 30 | +0.133 (0.93) | +0.079 (0.87) | +0.152 (0.98) | +0.111 | -0.017 | +0.084 |
| 2 | 4 | 60 | +0.137 (0.98) | +0.093 (0.96) | +0.096 (0.92) | +0.103 | +0.013 | +0.076 |
| 3 | 3 | 90 | +0.219 (0.97) | +0.091 (0.87) | -0.037 (0.39) | +0.083 | -0.077 | +0.057 |
| 2 | 4 | 30 | +0.087 (0.93) | +0.060 (0.89) | +0.093 (0.97) | +0.075 | +0.022 | +0.048 |

## BLOCK cells — locked expectancy filter bar + walk-forward (RH ticks pooled; Δ = refuse change per signal)

| cell | scored | blocked N | WR | avg | P(avg<0) | days | max day / pair loss share | Δ 2025 | Δ 2026-H1 | Δ 2026-H2 | Δ MATCH | Δ R1 | gates ①②③④ WF | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W1_DEEP | 1839 | 465 | 77.0% | -0.010 | 0.54 | 247 | 3.42 / 2.52 | +0.054 | -0.012 | -0.011 | +0.008 | +0.254 | ✓✗✗✓✗ | no |
| W1_SHALLOW | 1839 | 714 | 77.5% | +0.005 | 0.47 | 364 | – / – | +0.120 | -0.032 | -0.042 | -0.011 | +0.075 | ✗✗✓✓✗ | no |
| W2_HEAVY | 1839 | 751 | 77.1% | -0.038 | 0.69 | 346 | 0.44 / 0.72 | +0.128 | +0.003 | -0.050 | -0.034 | +0.243 | ✓✗✗✓✗ | no |
| W2_SELLERS | 1839 | 547 | 77.9% | +0.010 | 0.45 | 311 | – / – | +0.088 | -0.042 | -0.003 | -0.019 | +0.158 | ✗✗✓✓✗ | no |
| W2_LAST1M_SELL | 1839 | 410 | 75.4% | -0.113 | 0.87 | 266 | 0.18 / 0.26 | +0.112 | -0.003 | +0.010 | +0.010 | +0.119 | ✓✗✓✓✓ | no |
| W3_ONECANDLE | 1839 | 454 | 78.0% | +0.045 | 0.33 | 269 | – / – | +0.055 | -0.036 | -0.017 | +0.015 | +0.140 | ✗✗✓✓✗ | no |
| W3_STAIR | 1839 | 615 | 75.4% | -0.087 | 0.83 | 326 | 0.27 / 0.46 | +0.146 | +0.022 | -0.049 | +0.022 | +0.118 | ✓✗✓✓✗ | no |
| W3_WICK | 1839 | 399 | 75.4% | -0.152 | 0.88 | 253 | 0.45 / 0.30 | +0.109 | +0.008 | +0.020 | -0.030 | +0.149 | ✓✗✓✓✗ | no |
| W3_BIG | 1839 | 387 | 77.3% | -0.027 | 0.57 | 223 | 2.18 / 1.43 | +0.103 | -0.037 | +0.007 | +0.017 | +0.242 | ✓✗✗✓✗ | no |
| W3_VOLX | 1839 | 581 | 79.7% | +0.121 | 0.09 | 288 | – / – | -0.016 | -0.053 | -0.029 | -0.060 | +0.113 | ✗✗✓✓✗ | no |
| W3_FARHIGH | 1839 | 372 | 75.0% | -0.052 | 0.65 | 219 | 1.29 / 0.59 | +0.107 | +0.003 | -0.051 | +0.005 | +0.249 | ✓✗✗✓✗ | no |
| W4_WAIT4 | 1839 | 145 | 76.6% | -0.099 | 0.70 | 121 | 0.59 / 0.76 | +0.027 | -0.002 | +0.011 | +0.006 | +0.019 | ✓✗✗✓✓ | no |
| W4_LATEFLAG | 1839 | 126 | 72.2% | -0.073 | 0.67 | 102 | 0.57 / 0.82 | +0.024 | +0.011 | -0.021 | -0.003 | +0.007 | ✓✗✗✓✗ | no |
| W5_BELOWVWAP | 1839 | 1110 | 75.3% | -0.100 | 0.94 | 434 | 0.23 / 0.14 | +0.278 | -0.029 | +0.054 | +0.004 | +0.351 | ✓✗✓✓✗ | no |
| W5_RETRACE | 1839 | 389 | 77.9% | +0.110 | 0.17 | 245 | – / – | +0.026 | -0.040 | -0.032 | -0.004 | +0.130 | ✗✗✓✓✗ | no |
| W5_SMALLKEEP | 1839 | 595 | 77.3% | +0.043 | 0.30 | 344 | – / – | +0.090 | -0.036 | -0.054 | -0.029 | +0.101 | ✓✗✓✓✗ | no |
| W6_BTC1H_SLOPE | 1839 | 873 | 76.3% | -0.072 | 0.81 | 325 | 0.47 / 0.29 | +0.171 | -0.048 | +0.078 | +0.032 | +0.335 | ✓✗✓✓✗ | no |
| W6_BTC15 | 1839 | 873 | 77.7% | +0.037 | 0.30 | 385 | – / – | +0.067 | -0.023 | -0.074 | -0.114 | +0.143 | ✗✗✓✓✗ | no |
| W6_BTC1H | 1839 | 859 | 76.9% | -0.006 | 0.52 | 364 | 5.66 / 3.93 | +0.208 | -0.084 | +0.002 | -0.081 | +0.279 | ✓✗✗✓✗ | no |
| W6_BREADTH | 1839 | 889 | 77.8% | +0.021 | 0.38 | 376 | – / – | +0.182 | -0.077 | -0.037 | -0.050 | +0.268 | ✗✗✓✓✗ | no |
| W6B_BETA | 1839 | 791 | 77.1% | -0.035 | 0.66 | 377 | 0.67 / 0.92 | +0.156 | -0.039 | +0.005 | +0.001 | +0.245 | ✓✗✗✓✗ | no |
| W6B_IDIO | 1839 | 635 | 75.6% | -0.094 | 0.89 | 321 | 0.14 / 0.19 | +0.117 | +0.019 | -0.009 | +0.010 | +0.162 | ✓✗✓✓✓ | no |
| W6B_MKTRED | 1839 | 728 | 77.2% | -0.010 | 0.55 | 370 | 1.34 / 2.62 | +0.180 | -0.054 | -0.026 | -0.006 | +0.230 | ✓✗✗✓✗ | no |
| W6B_MKTSELL | 1839 | 852 | 76.5% | -0.032 | 0.67 | 403 | 0.64 / 0.79 | +0.196 | -0.034 | -0.037 | -0.013 | +0.247 | ✓✗✗✓✗ | no |
| W7_HIGHTURN | 1839 | 515 | 73.0% | -0.223 | 0.99 | 311 | 0.10 / 0.19 | +0.154 | +0.040 | +0.032 | +0.004 | +0.089 | ✓✓✓✓✓ | **PASS** |
| W7_SMALLCAP | 1839 | 829 | 79.0% | +0.081 | 0.13 | 287 | – / – | +0.057 | -0.069 | -0.049 | -0.036 | +0.191 | ✗✗✓✓✗ | no |
| W7_NOTTOP50 | 1839 | 1069 | 80.0% | +0.128 | 0.02 | 276 | – / – | +0.000 | -0.084 | -0.115 | -0.083 | +0.390 | ✗✗✓✓✗ | no |
| W7_NEWLIST | 1839 | 239 | 72.8% | -0.264 | 0.97 | 182 | 0.13 / 0.30 | +0.141 | -0.017 | +0.045 | -0.017 | +0.103 | ✓✓✓✓✗ | no |

BE WR used for gate ①: 77.4 %. Δ in %-points per signal of the whole cohort (refusing the blocked set). Haircut 30–50 % applies to any PASS.

Shuffle null (100 within-era permutations of y; nboot 300): expected PASS cells by luck 1.18 (max 4); observed PASS 1 of 28.

## MASTER (19, per batch) and LIVE (8) — every BLOCK cell, before / after (as traded and today re-priced), % and $

| cell | block | batch | n | scored | blocked | as traded %-pts → rule | as traded $ → rule | today %-pts → rule | today $ → rule | blocked fills (as traded %) |
|---|---|---|---|---|---|---|---|---|---|---|
| W1_DEEP | MASTER | B19 | 14 | 14 | 2 | +8.31 → +6.28 | +1,063 → +895 | +10.05 → +7.55 | +1,132 → +926 | LPTUSDT +1.03, CAPUSDT +1.01 |
| W1_DEEP | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.77 | -90 → -180 | -1.73 → -2.98 | -105 → -193 | RSRUSDT +1.28 |
| W1_DEEP | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.26 | +177 → +90 | +2.50 → +1.25 | +177 → +89 | BTWUSDT +1.25 |
| W1_DEEP | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -6.64 | -139 → -167 | -1.90 → -3.15 | -42 → -73 | ZKUSDT +1.12 |
| W1_SHALLOW | MASTER | B19 | 14 | 14 | 5 | +8.31 → +6.71 | +1,063 → +735 | +10.05 → +4.80 | +1,132 → +400 | SUSDT +1.00, SENTUSDT +1.03, CHIPUSDT +1.00, API3USDT -2.44, WUSDT +1.00 |
| W1_SHALLOW | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.75 | -90 → -161 | -1.73 → -2.98 | -105 → -176 | CAPUSDT +1.26 |
| W1_SHALLOW | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W1_SHALLOW | LIVE | LIVE | 8 | 8 | 4 | -5.52 → +4.48 | -139 → +113 | -1.90 → +5.00 | -42 → +126 | ICNTUSDT -3.86, RLCUSDT +1.08, STRKUSDT -3.79, DUSKUSDT -3.43 |
| W2_HEAVY | MASTER | B19 | 14 | 14 | 5 | +8.31 → +9.10 | +1,063 → +1,185 | +10.05 → +11.25 | +1,132 → +1,466 | BATUSDT -1.38, CHIPUSDT +1.00, CAPUSDT +1.01, ZKUSDT +1.02, API3USDT -2.44 |
| W2_HEAVY | MASTER | B20 | 3 | 3 | 2 | -1.49 → -4.03 | -90 → -252 | -1.73 → -4.23 | -105 → -265 | CAPUSDT +1.26, RSRUSDT +1.28 |
| W2_HEAVY | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W2_HEAVY | LIVE | LIVE | 8 | 8 | 7 | -5.52 → -3.43 | -139 → -92 | -1.90 → +0.25 | -42 → +7 | ICNTUSDT -3.86, BATUSDT +1.18, PYTHUSDT +1.09, RLCUSDT +1.08, STRKUSDT -3.79, BLURUSDT +1.09, ZKUSDT +1.12 |
| W2_SELLERS | MASTER | B19 | 14 | 14 | 6 | +8.31 → +2.22 | +1,063 → +382 | +10.05 → +3.55 | +1,132 → +419 | WUSDT +1.00, SENTUSDT +1.03, LPTUSDT +1.03, MINAUSDT +1.00, CAPUSDT +1.01, ZKUSDT +1.02 |
| W2_SELLERS | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W2_SELLERS | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.25 | +177 → +87 | +2.50 → +1.25 | +177 → +87 | ZKUSDT +1.26 |
| W2_SELLERS | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -6.62 | -139 → -167 | -1.90 → -3.15 | -42 → -73 | PYTHUSDT +1.09 |
| W2_LAST1M_SELL | MASTER | B19 | 14 | 8 | 2 | +8.31 → +6.29 | +1,063 → +871 | +10.05 → +8.55 | +1,132 → +1,020 | CAPUSDT +1.01, ZKUSDT +1.02 |
| W2_LAST1M_SELL | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W2_LAST1M_SELL | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.25 | +177 → +87 | +2.50 → +1.25 | +177 → +87 | ZKUSDT +1.26 |
| W2_LAST1M_SELL | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -2.09 | -139 → -47 | -1.90 → -2.15 | -42 → -48 | DUSKUSDT -3.43 |
| W3_ONECANDLE | MASTER | B19 | 14 | 14 | 4 | +8.31 → +4.27 | +1,063 → +559 | +10.05 → +5.05 | +1,132 → +509 | CFXUSDT +1.01, LPTUSDT +1.03, WUSDT +1.00, KAIAUSDT +1.01 |
| W3_ONECANDLE | MASTER | B20 | 3 | 3 | 2 | -1.49 → +1.26 | -90 → +72 | -1.73 → +1.25 | -105 → +71 | OGNUSDT -4.03, RSRUSDT +1.28 |
| W3_ONECANDLE | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W3_ONECANDLE | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W3_STAIR | MASTER | B19 | 14 | 14 | 6 | +8.31 → +4.65 | +1,063 → +623 | +10.05 → +9.00 | +1,132 → +1,095 | WUSDT +1.00, BATUSDT -1.38, CHIPUSDT +1.00, ORCAUSDT +1.01, MINAUSDT +1.00, ZKUSDT +1.02 |
| W3_STAIR | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W3_STAIR | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.25 | +177 → +87 | +2.50 → +1.25 | +177 → +87 | ZKUSDT +1.26 |
| W3_STAIR | LIVE | LIVE | 8 | 8 | 2 | -5.52 → -2.78 | -139 → -81 | -1.90 → +1.05 | -42 → +21 | ICNTUSDT -3.86, ZKUSDT +1.12 |
| W3_WICK | MASTER | B19 | 14 | 14 | 5 | +8.31 → +3.24 | +1,063 → +441 | +10.05 → +4.80 | +1,132 → +491 | SENTUSDT +1.03, CHIPUSDT +1.00, ORCAUSDT +1.01, CAPUSDT +1.01, ZKUSDT +1.02 |
| W3_WICK | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W3_WICK | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.25 | +177 → +87 | +2.50 → +1.25 | +177 → +87 | ZKUSDT +1.26 |
| W3_WICK | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -1.73 | -139 → -40 | -1.90 → +2.30 | -42 → +69 | STRKUSDT -3.79 |
| W3_BIG | MASTER | B19 | 14 | 14 | 2 | +8.31 → +6.28 | +1,063 → +834 | +10.05 → +7.55 | +1,132 → +851 | LPTUSDT +1.03, KAIAUSDT +1.01 |
| W3_BIG | MASTER | B20 | 3 | 3 | 2 | -1.49 → -4.03 | -90 → -252 | -1.73 → -4.23 | -105 → -265 | CAPUSDT +1.26, RSRUSDT +1.28 |
| W3_BIG | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.26 | +177 → +90 | +2.50 → +1.25 | +177 → +89 | BTWUSDT +1.25 |
| W3_BIG | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W3_VOLX | MASTER | B19 | 14 | 0 | 0 | +8.31 → +8.31 | +1,063 → +1,063 | +10.05 → +10.05 | +1,132 → +1,132 |  |
| W3_VOLX | MASTER | B20 | 3 | 0 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W3_VOLX | MASTER | B21 | 2 | 0 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W3_VOLX | LIVE | LIVE | 8 | 0 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W3_FARHIGH | MASTER | B19 | 14 | 14 | 2 | +8.31 → +6.28 | +1,063 → +816 | +10.05 → +7.55 | +1,132 → +828 | CFXUSDT +1.01, LPTUSDT +1.03 |
| W3_FARHIGH | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.77 | -90 → -180 | -1.73 → -2.98 | -105 → -193 | RSRUSDT +1.28 |
| W3_FARHIGH | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.26 | +177 → +90 | +2.50 → +1.25 | +177 → +89 | BTWUSDT +1.25 |
| W3_FARHIGH | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W4_WAIT4 | MASTER | B19 | 14 | 14 | 4 | +8.31 → +6.65 | +1,063 → +957 | +10.05 → +11.50 | +1,132 → +1,511 | BATUSDT -1.38, CAPUSDT +1.01, ZKUSDT +1.02, KAIAUSDT +1.01 |
| W4_WAIT4 | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.75 | -90 → -161 | -1.73 → -2.98 | -105 → -176 | CAPUSDT +1.26 |
| W4_WAIT4 | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W4_WAIT4 | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -1.73 | -139 → -40 | -1.90 → +2.30 | -42 → +69 | STRKUSDT -3.79 |
| W4_LATEFLAG | MASTER | B19 | 14 | 14 | 0 | +8.31 → +8.31 | +1,063 → +1,063 | +10.05 → +10.05 | +1,132 → +1,132 |  |
| W4_LATEFLAG | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W4_LATEFLAG | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W4_LATEFLAG | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W5_BELOWVWAP | MASTER | B19 | 14 | 14 | 8 | +8.31 → +3.69 | +1,063 → +471 | +10.05 → +1.05 | +1,132 → +71 | WUSDT +1.00, CHIPUSDT +1.00, CFXUSDT +1.01, ORCAUSDT +1.01, LPTUSDT +1.03, MINAUSDT +1.00, CAPUSDT +1.01, API3USDT -2.44 |
| W5_BELOWVWAP | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W5_BELOWVWAP | MASTER | B21 | 2 | 2 | 2 | +2.51 → +0.00 | +177 → +0 | +2.50 → +0.00 | +177 → +0 | BTWUSDT +1.25, ZKUSDT +1.26 |
| W5_BELOWVWAP | LIVE | LIVE | 8 | 8 | 3 | -5.52 → +0.95 | -139 → +16 | -1.90 → +5.25 | -42 → +130 | ICNTUSDT -3.86, BATUSDT +1.18, STRKUSDT -3.79 |
| W5_RETRACE | MASTER | B19 | 14 | 14 | 3 | +8.31 → +5.25 | +1,063 → +670 | +10.05 → +6.30 | +1,132 → +651 | SENTUSDT +1.03, CFXUSDT +1.01, LPTUSDT +1.03 |
| W5_RETRACE | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W5_RETRACE | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W5_RETRACE | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W5_SMALLKEEP | MASTER | B19 | 14 | 14 | 6 | +8.31 → +5.67 | +1,063 → +640 | +10.05 → +3.55 | +1,132 → +286 | SENTUSDT +1.03, CFXUSDT +1.01, ORCAUSDT +1.01, LPTUSDT +1.03, API3USDT -2.44, WUSDT +1.00 |
| W5_SMALLKEEP | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W5_SMALLKEEP | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W5_SMALLKEEP | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |
| W6_BTC1H_SLOPE | MASTER | B19 | 14 | 14 | 0 | +8.31 → +8.31 | +1,063 → +1,063 | +10.05 → +10.05 | +1,132 → +1,132 |  |
| W6_BTC1H_SLOPE | MASTER | B20 | 3 | 3 | 2 | -1.49 → +1.28 | -90 → +91 | -1.73 → +1.25 | -105 → +88 | CAPUSDT +1.26, OGNUSDT -4.03 |
| W6_BTC1H_SLOPE | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W6_BTC1H_SLOPE | LIVE | LIVE | 8 | 8 | 5 | -5.52 → -1.59 | -139 → -28 | -1.90 → -1.70 | -42 → -30 | RLCUSDT +1.08, STRKUSDT -3.79, BLURUSDT +1.09, ZKUSDT +1.12, DUSKUSDT -3.43 |
| W6_BTC15 | MASTER | B19 | 14 | 14 | 6 | +8.31 → +2.26 | +1,063 → +313 | +10.05 → +3.55 | +1,132 → +327 | WUSDT +1.00, CFXUSDT +1.01, ORCAUSDT +1.01, CAPUSDT +1.01, ZKUSDT +1.02, WUSDT +1.00 |
| W6_BTC15 | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W6_BTC15 | MASTER | B21 | 2 | 2 | 2 | +2.51 → +0.00 | +177 → +0 | +2.50 → +0.00 | +177 → +0 | BTWUSDT +1.25, ZKUSDT +1.26 |
| W6_BTC15 | LIVE | LIVE | 8 | 8 | 3 | -5.52 → -3.90 | -139 → -93 | -1.90 → -0.20 | -42 → +8 | RLCUSDT +1.08, STRKUSDT -3.79, BLURUSDT +1.09 |
| W6_BTC1H | MASTER | B19 | 14 | 14 | 6 | +8.31 → +2.27 | +1,063 → +276 | +10.05 → +3.55 | +1,132 → +281 | CHIPUSDT +1.00, CFXUSDT +1.01, ORCAUSDT +1.01, MINAUSDT +1.00, ZKUSDT +1.02, WUSDT +1.00 |
| W6_BTC1H | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.75 | -90 → -161 | -1.73 → -2.98 | -105 → -176 | CAPUSDT +1.26 |
| W6_BTC1H | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W6_BTC1H | LIVE | LIVE | 8 | 8 | 3 | -5.52 → +1.04 | -139 → +20 | -1.90 → +5.25 | -42 → +133 | ICNTUSDT -3.86, RLCUSDT +1.08, STRKUSDT -3.79 |
| W6_BREADTH | MASTER | B19 | 14 | 14 | 8 | +8.31 → +3.66 | +1,063 → +393 | +10.05 → +2.05 | +1,132 → +105 | WUSDT +1.00, SENTUSDT +1.03, CHIPUSDT +1.00, LPTUSDT +1.03, ZKUSDT +1.02, API3USDT -2.44, WUSDT +1.00, KAIAUSDT +1.01 |
| W6_BREADTH | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.75 | -90 → -161 | -1.73 → -2.98 | -105 → -176 | CAPUSDT +1.26 |
| W6_BREADTH | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.26 | +177 → +90 | +2.50 → +1.25 | +177 → +89 | BTWUSDT +1.25 |
| W6_BREADTH | LIVE | LIVE | 8 | 8 | 4 | -5.52 → -0.47 | -139 → -1 | -1.90 → -0.45 | -42 → +1 | RLCUSDT +1.08, STRKUSDT -3.79, BLURUSDT +1.09, DUSKUSDT -3.43 |
| W6B_BETA | MASTER | B19 | 14 | 14 | 4 | +8.31 → +6.68 | +1,063 → +900 | +10.05 → +10.50 | +1,132 → +1,313 | BATUSDT -1.38, ORCAUSDT +1.01, MINAUSDT +1.00, WUSDT +1.00 |
| W6B_BETA | MASTER | B20 | 3 | 3 | 3 | -1.49 → +0.00 | -90 → +0 | -1.73 → +0.00 | -105 → +0 | CAPUSDT +1.26, OGNUSDT -4.03, RSRUSDT +1.28 |
| W6B_BETA | MASTER | B21 | 2 | 2 | 2 | +2.51 → +0.00 | +177 → +0 | +2.50 → +0.00 | +177 → +0 | BTWUSDT +1.25, ZKUSDT +1.26 |
| W6B_BETA | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -6.61 | -139 → -165 | -1.90 → -3.15 | -42 → -71 | RLCUSDT +1.08 |
| W6B_IDIO | MASTER | B19 | 14 | 14 | 4 | +8.31 → +7.70 | +1,063 → +936 | +10.05 → +7.05 | +1,132 → +776 | WUSDT +1.00, LPTUSDT +1.03, ZKUSDT +1.02, API3USDT -2.44 |
| W6B_IDIO | MASTER | B20 | 3 | 3 | 0 | -1.49 → -1.49 | -90 → -90 | -1.73 → -1.73 | -105 → -105 |  |
| W6B_IDIO | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W6B_IDIO | LIVE | LIVE | 8 | 8 | 5 | -5.52 → +3.35 | -139 → +83 | -1.90 → +3.75 | -42 → +92 | ICNTUSDT -3.86, STRKUSDT -3.79, BLURUSDT +1.09, ZKUSDT +1.12, DUSKUSDT -3.43 |
| W6B_MKTRED | MASTER | B19 | 14 | 14 | 3 | +8.31 → +7.68 | +1,063 → +1,079 | +10.05 → +11.75 | +1,132 → +1,534 | BATUSDT -1.38, CAPUSDT +1.01, WUSDT +1.00 |
| W6B_MKTRED | MASTER | B20 | 3 | 3 | 2 | -1.49 → -4.03 | -90 → -252 | -1.73 → -4.23 | -105 → -265 | CAPUSDT +1.26, RSRUSDT +1.28 |
| W6B_MKTRED | MASTER | B21 | 2 | 2 | 2 | +2.51 → +0.00 | +177 → +0 | +2.50 → +0.00 | +177 → +0 | BTWUSDT +1.25, ZKUSDT +1.26 |
| W6B_MKTRED | LIVE | LIVE | 8 | 8 | 1 | -5.52 → -6.61 | -139 → -165 | -1.90 → -3.15 | -42 → -71 | RLCUSDT +1.08 |
| W6B_MKTSELL | MASTER | B19 | 14 | 14 | 3 | +8.31 → +7.68 | +1,063 → +1,000 | +10.05 → +11.75 | +1,132 → +1,438 | BATUSDT -1.38, ORCAUSDT +1.01, WUSDT +1.00 |
| W6B_MKTSELL | MASTER | B20 | 3 | 3 | 3 | -1.49 → +0.00 | -90 → +0 | -1.73 → +0.00 | -105 → +0 | CAPUSDT +1.26, OGNUSDT -4.03, RSRUSDT +1.28 |
| W6B_MKTSELL | MASTER | B21 | 2 | 2 | 2 | +2.51 → +0.00 | +177 → +0 | +2.50 → +0.00 | +177 → +0 | BTWUSDT +1.25, ZKUSDT +1.26 |
| W6B_MKTSELL | LIVE | LIVE | 8 | 8 | 2 | -5.52 → -7.78 | -139 → -194 | -1.90 → -4.40 | -42 → -102 | BATUSDT +1.18, RLCUSDT +1.08 |
| W7_HIGHTURN | MASTER | B19 | 14 | 14 | 2 | +8.31 → +6.31 | +1,063 → +831 | +10.05 → +7.55 | +1,132 → +843 | MINAUSDT +1.00, WUSDT +1.00 |
| W7_HIGHTURN | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W7_HIGHTURN | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W7_HIGHTURN | LIVE | LIVE | 8 | 8 | 3 | -5.52 → -3.84 | -139 → -107 | -1.90 → -0.20 | -42 → -9 | ICNTUSDT -3.86, RLCUSDT +1.08, BLURUSDT +1.09 |
| W7_SMALLCAP | MASTER | B19 | 14 | 14 | 1 | +8.31 → +10.75 | +1,063 → +1,306 | +10.05 → +9.80 | +1,132 → +1,107 | API3USDT -2.44 |
| W7_SMALLCAP | MASTER | B20 | 3 | 3 | 1 | -1.49 → +2.54 | -90 → +162 | -1.73 → +2.50 | -105 → +160 | OGNUSDT -4.03 |
| W7_SMALLCAP | MASTER | B21 | 2 | 2 | 0 | +2.51 → +2.51 | +177 → +177 | +2.50 → +2.50 | +177 → +177 |  |
| W7_SMALLCAP | LIVE | LIVE | 8 | 8 | 4 | -5.52 → -0.41 | -139 → -15 | -1.90 → -0.45 | -42 → -16 | ICNTUSDT -3.86, RLCUSDT +1.08, BLURUSDT +1.09, DUSKUSDT -3.43 |
| W7_NOTTOP50 | MASTER | B19 | 14 | 14 | 11 | +8.31 → +0.64 | +1,063 → +31 | +10.05 → -1.70 | +1,132 → -345 | WUSDT +1.00, SUSDT +1.00, SENTUSDT +1.03, CHIPUSDT +1.00, CFXUSDT +1.01, LPTUSDT +1.03, CAPUSDT +1.01, ZKUSDT +1.02, API3USDT -2.44, WUSDT +1.00, KAIAUSDT +1.01 |
| W7_NOTTOP50 | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.77 | -90 → -180 | -1.73 → -2.98 | -105 → -193 | RSRUSDT +1.28 |
| W7_NOTTOP50 | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.25 | +177 → +87 | +2.50 → +1.25 | +177 → +87 | ZKUSDT +1.26 |
| W7_NOTTOP50 | LIVE | LIVE | 8 | 8 | 5 | -5.52 → -1.53 | -139 → -44 | -1.90 → -1.70 | -42 → -49 | ICNTUSDT -3.86, PYTHUSDT +1.09, BLURUSDT +1.09, ZKUSDT +1.12, DUSKUSDT -3.43 |
| W7_NEWLIST | MASTER | B19 | 14 | 14 | 2 | +8.31 → +6.30 | +1,063 → +857 | +10.05 → +7.55 | +1,132 → +876 | CHIPUSDT +1.00, CAPUSDT +1.01 |
| W7_NEWLIST | MASTER | B20 | 3 | 3 | 1 | -1.49 → -2.75 | -90 → -161 | -1.73 → -2.98 | -105 → -176 | CAPUSDT +1.26 |
| W7_NEWLIST | MASTER | B21 | 2 | 2 | 1 | +2.51 → +1.26 | +177 → +90 | +2.50 → +1.25 | +177 → +89 | BTWUSDT +1.25 |
| W7_NEWLIST | LIVE | LIVE | 8 | 8 | 0 | -5.52 → -5.52 | -139 → -139 | -1.90 → -1.90 | -42 → -42 |  |

Per-trade master / live feature sheet:

| block | batch | pair | opened_at | pct_traded | usd_traded | y | why | ent_bar_ret | wait_bars | red_vs_spike_q | tbs_red | spike_conc | spike_uwick | off_high | vs_vwap | retrace_frac | btc_1h_slope | btc_bar | brd_red | mkt_tbs | btc_f15 | R_ent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASTER | B19 | WUSDT | 2026-10-11T18:10:08 | 1.003 | 140.095 | 1.25 | TP | -1.496 | 2.0 | 0.804 | 41.815 | 0.27 | 0.17 | -3.568 | -1.936 | 0.293 | 0.235 | 0.065 | 32.0 | 63.511 | 0.006 | 0.167 |
| MASTER | B19 | BATUSDT | 2026-10-11T16:05:07 | -1.377 | -211.939 | -4.201 | STOP | -1.326 | 9.0 | 10.286 | 49.077 | 0.392 | 0.094 | -3.116 | 2.178 | 0.181 | 0.194 | -0.152 | 76.0 | 43.58 | -0.003 | 0.301 |
| MASTER | B19 | SUSDT | 2026-10-11T15:25:07 | 1.005 | 153.346 | 1.25 | TP | -0.554 | 1.0 | 0.516 | 45.406 | 0.476 | 0.18 | -1.212 | 0.037 | 0.133 | 0.034 | 0.018 | 24.0 | 48.206 | 0.192 | 0.256 |
| MASTER | B19 | SENTUSDT | 2026-10-11T11:25:09 | 1.031 | 145.527 | 1.25 | TP | -0.215 | 2.0 | 0.191 | 44.939 | 0.461 | 0.616 | -3.656 | 0.222 | 0.353 | 0.013 | 0.009 | 58.0 | 60.211 | 0.012 | 0.101 |
| MASTER | B19 | CHIPUSDT | 2026-10-10T18:40:09 | 1.004 | 140.996 | 1.25 | TP | -0.503 | 2.0 | 1.002 | 50.141 | 0.324 | 0.513 | -1.289 | -0.088 | 0.094 | 0.081 | 0.005 | 48.0 | 49.735 | 0.015 | 0.241 |
| MASTER | B19 | CFXUSDT | 2026-10-10T13:35:12 | 1.007 | 143.791 | 1.25 | TP | -1.51 | 1.0 | 0.426 | 50.571 | 0.881 | 0.031 | -6.755 | -2.25 | 0.525 | 0.045 | -0.013 | 32.0 | 53.415 | -0.0 | 0.08 |
| MASTER | B19 | ORCAUSDT | 2026-10-10T11:35:12 | 1.012 | 143.024 | 1.25 | TP | -1.586 | 1.0 | 0.838 | 49.034 | 0.304 | 0.411 | -3.011 | -0.97 | 0.335 | 0.054 | -0.021 | 16.0 | 37.523 | 0.115 | 0.264 |
| MASTER | B19 | LPTUSDT | 2026-10-10T06:45:07 | 1.029 | 102.852 | 1.25 | TP | -2.86 | 2.0 | 0.383 | 42.895 | 0.659 | 0.133 | -8.201 | -4.026 | 0.601 | 0.058 | 0.083 | 24.0 | 70.818 | -0.138 | 0.214 |
| MASTER | B19 | MINAUSDT | 2026-10-10T05:20:08 | 1.002 | 100.194 | 1.25 | TP | -0.913 | 1.0 | 0.867 | 43.855 | 0.316 | 0.142 | -1.613 | -0.432 | 0.153 | 0.023 | -0.023 | 42.0 | 53.794 | 0.042 | 0.576 |
| MASTER | B19 | CAPUSDT | 2026-10-10T04:30:07 | 1.009 | 64.762 | 1.25 | TP | -3.166 | 4.0 | 2.072 | 35.227 | 0.453 | 0.601 | -3.574 | -0.969 | 0.223 | 0.015 | -0.009 | 76.0 | 49.22 | 0.028 | 0.314 |
| MASTER | B19 | ZKUSDT | 2026-10-09T16:35:08 | 1.018 | 127.192 | 0.25 | LATE_TP | -0.81 | 4.0 | 1.512 | 41.633 | 0.213 | 0.602 | -1.864 | 0.279 | 0.2 | 0.125 | 0.066 | 52.0 | 62.16 | 0.035 | 0.173 |
| MASTER | B19 | API3USDT | 2026-10-09T14:25:07 | -2.439 | -243.905 | 0.25 | LATE_TP | -0.541 | 1.0 | 1.637 | 53.062 | 0.427 | 0.179 | -1.692 | -0.148 | 0.231 | 0.168 | 0.088 | 28.0 | 57.263 | -0.139 | 0.387 |
| MASTER | B19 | WUSDT | 2026-10-09T09:05:08 | 1.001 | 131.085 | 1.25 | TP | -0.109 | 1.0 | 0.673 | 56.178 | 0.617 | 0.248 | -0.646 | 0.25 | 0.094 | 0.143 | -0.141 | 84.0 | 29.384 | 0.012 | 0.56 |
| MASTER | B19 | KAIAUSDT | 2026-10-09T08:45:09 | 1.01 | 125.504 | 1.25 | TP | -1.628 | 4.0 | 0.526 | 47.742 | 0.973 | 0.04 | -3.224 | 4.674 | 0.139 | 0.146 | 0.012 | 48.0 | 50.605 | 0.145 | 0.114 |
| MASTER | B20 | CAPUSDT | 2026-10-12T12:55:08 | 1.255 | 71.724 | 1.25 | TP | -0.722 | 5.0 | 1.358 | 51.895 | 0.494 | 0.053 | -3.024 | 4.722 | 0.136 | -0.145 | -0.208 | 82.0 | 32.416 | -0.067 | 0.268 |
| MASTER | B20 | OGNUSDT | 2026-10-12T10:40:09 | -4.029 | -252.119 | -4.229 | STOP | -1.096 | 2.0 | 0.837 | 52.023 | 0.612 | 0.621 | -1.306 | -0.673 | 0.199 | -0.066 | -0.036 | 46.0 | 38.633 | 0.009 | 0.64 |
| MASTER | B20 | RSRUSDT | 2026-10-12T05:15:11 | 1.283 | 90.668 | 1.25 | TP | -3.268 | 2.0 | 2.195 | 48.935 | 0.971 | 0.013 | -6.257 | 2.684 | 0.172 | 0.148 | -0.04 | 76.0 | 40.843 | -0.26 | 0.332 |
| MASTER | B21 | BTWUSDT | 2026-10-14T03:55:07 | 1.252 | 87.496 | 1.25 | TP | -4.203 | 1.0 | 0.571 | 48.976 | 0.443 | 0.165 | -5.473 | -1.896 | 0.321 | 0.09 | -0.059 | 90.0 | 45.361 | -0.111 | 0.054 |
| MASTER | B21 | ZKUSDT | 2026-10-13T14:35:12 | 1.255 | 89.516 | 1.25 | TP | -1.679 | 1.0 | 0.883 | 43.114 | 0.388 | 0.547 | -2.95 | -1.349 | 0.281 | 0.067 | -0.105 | 80.0 | 37.568 | 0.217 | 0.159 |
| LIVE | LIVE | ICNTUSDT | 2026-10-16T00:05:09 | -3.856 | -85.565 | -4.2 | STOP | -0.534 | 3.0 | 1.912 | 45.327 | 0.311 | 0.161 | -1.975 | -0.065 | 0.203 | 0.143 | 0.053 | 48.0 | 62.337 | 0.121 | 0.66 |
| LIVE | LIVE | BATUSDT | 2026-10-15T16:00:10 | 1.175 | 29.668 | 1.25 | TP | -0.805 | 1.0 | 1.298 | 47.642 | 0.414 | 0.025 | -1.761 | -0.228 | 0.154 | 0.047 | -0.019 | 18.0 | 45.817 | 0.075 | 0.238 |
| LIVE | LIVE | PYTHUSDT | 2026-10-15T15:30:10 | 1.094 | 27.478 | 1.25 | TP | -0.98 | 2.0 | 1.051 | 42.426 | 0.401 | 0.076 | -1.791 | 0.509 | 0.101 | 0.032 | 0.012 | 30.0 | 51.162 | 0.045 | 0.132 |
| LIVE | LIVE | RLCUSDT | 2026-10-15T13:30:11 | 1.084 | 25.398 | 1.25 | TP | -0.295 | 1.0 | 2.235 | 55.295 | 0.507 | 0.007 | -1.181 | 0.398 | 0.118 | -0.327 | -0.273 | 70.0 | 28.455 | 0.608 | 0.592 |
| LIVE | LIVE | STRKUSDT | 2026-10-15T10:20:11 | -3.792 | -99.463 | -4.2 | STOP | -0.489 | 4.0 | 1.792 | 55.213 | 0.404 | 0.424 | -2.353 | -0.262 | 0.167 | -0.147 | 0.077 | 28.0 | 62.071 | -0.035 | 0.409 |
| LIVE | LIVE | BLURUSDT | 2026-10-15T09:30:12 | 1.091 | 27.733 | 1.25 | TP | -1.515 | 3.0 | 2.003 | 45.031 | 0.487 | 0.239 | -2.217 | 2.241 | 0.109 | -0.116 | 0.037 | 56.0 | 54.559 | -0.039 | 0.938 |
| LIVE | LIVE | ZKUSDT | 2026-10-15T04:40:13 | 1.116 | 27.776 | 1.25 | TP | -2.547 | 2.0 | 1.657 | 45.262 | 0.381 | 0.028 | -4.228 | 0.868 | 0.237 | -0.272 | 0.045 | 42.0 | 48.606 | 0.183 | 0.203 |
| LIVE | LIVE | DUSKUSDT | 2026-10-15T01:45:12 | -3.433 | -92.246 | 0.25 | LATE_TP | -0.391 | 2.0 | 0.369 | 47.711 | 0.577 | 0.035 | -1.801 | 0.074 | 0.173 | -0.207 | 0.095 | 32.0 | 50.329 | -0.254 | 0.423 |

---
## Review corrections (adversarial review 2026-10-16; scratch: scratchpad/willyrev/ reprice.py · an_a.py · ms.py · late_mkt.py)
1. **TP +2 headline compared two different exits.** The "Deployable" table took `FX.evaluate(2, 4, 90)` (NO late TP) minus today's
   `evaluate_late(1.25, …, 30, 0.25)` (WITH the late TP) → two changes at once. Like-for-like (late TP in both arms, tick verdict):
   all rebuilt +0.063 (P 0.98; eras +0.062 / +0.055 / +0.078, none ≥ P 0.91) · **engine-matched +0.026 (P 0.69); 2026-H2 −0.178**.
   Booking the late TP like the live engine (DL 297: market close at the print only when price is already ≥ +0.25 at minute 30, else the resting line): all +0.094 · engine-matched +0.053 (P 0.83), 2026-H2 −0.132.
   The gain sits in fills the engine does not take (2026-H2 not-matched +0.21 vs matched −0.18).
2. **Master + live cut TP +2** (with late TP): master 19 +10.82 → +7.91 pts (+$1,203 → +$893; B19 +10.05 → +4.14) · live 8 −1.90 → −6.10
   (−$42 → −$146). Three +1.25 TPs turn into −4.2 stops while waiting (LPT, W 10-09, BAT live). The earlier yr5 study (reports/B19_WILLY_TP15.txt,
   ~530 engine fills) found TP +2 ≈ 0.08 worse than +1.25. → **TP +2 NOT registered** (operator 2026-10-16).
3. **W0 flips on engine-matched fills:** WILLY entry − random window bar = **+0.191 (P 0.99)**, eras +0.040 / +0.229 / +0.270; vs first green
   +0.045. Not-engine-matched −0.078 drives the pooled +0.018. Likely cause: 172 replay triggers 1–3+ bars later than the emulator's.
   Honest verdict: **era-dependent; on engine fills WILLY's selection adds ≈ +0.2 in 2026.**
4. **2025 "regime" headline** is also the not-matched fills: engine-matched by era −0.027 / +0.058 / +0.077 (vs −0.318 / +0.054 / +0.046 all).
5. Survive on engine-matched fills (bucket reads, not gate-tested): large caps worse (−0.08 vs +0.12 under $100M) · 20–25× spikes worse
   (−0.18) · TP +3 worse than today (−0.06 to −0.08). EU-morning weakens to −0.05 (drop as a watch item). Turnover 0.5–1: refusing it on
   engine fills = +0.052 / −0.004 / −0.077 by era (band best in 2026-H2) → **dropped, not registered**.
6. Protocol drift: 100 shuffles (prereg 200), 300 bootstrap resamples (prereg 4,000); day-block wording inconsistent. No lookahead found.
7. Tracker bug found: the DL 290 higher-TP cells in scripts/scout_willy_timecap.py measured against the PRE-297 exit (no late TP) → fixed
   under DECISION_LOG 304.
