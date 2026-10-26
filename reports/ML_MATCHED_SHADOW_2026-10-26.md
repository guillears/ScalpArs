# ML MATCHED-LONG SHADOW — should refused C/W-matched momentum longs be allowed? (2026-10-26)

Study only: no engine, config, UI, DECISION_LOG or CURRENT_STATE change. Scripts: `scripts/study_ml_matched_{extract,fetch,build,parity,analyse,report}.py` (+ `study_ml_matched_common.py`). All P&L is % per fill at 1×, priced by the validated ML exit replica at **today's** exit (RP.BASE), taker entry fee for shadows and controls alike.

## 0. Logs check — are matched-long refusals still recorded, with enough to price them?

**Short answer: only partly.** Nothing records the specific pattern labels of a refusal today. The live engine still logs every refusal, but each surface is missing something:

| source | from → to | per refusal | pattern info | indicator state | usable? |
|---|---|---|---|---|---|
| (a) orders export `scalpars_orders_*` | – | fills only; a refused long never becomes an Order | – | – | no |
| (a) decisions export `/api/decisions/export.csv` (`reports/BASELINE22-24_*_LIVE_decisions.csv`) | last 1–7 days per download | BLOCK rows **aggregated per 5 min × pair × gate** (t bucket, pair, count) — no price, no seconds | none | none | count only |
| (a) phantoms export `scalpars_phantoms_*` | **2026-07-06 → 07-26** (seeding retired 07-26/07-30, `_seed_phantom_flip` is a no-op) | pair, second, entry price, full entry_* stamps, July-exit outcome | **yes — `entry_cohort` (e.g. `W2+W6`)** | yes | **yes (303 rows → 209 episodes)** |
| (b) `_record_filter_block('LONG_UNMATCHED_ONLY')` | live | in-memory counter + `BotState.filter_block_counts_json` (counts only; reset on paper reset) | none | none | count only |
| (b) decision journal `/opt/scalpars-data/journal/decisions-*.jsonl` (server, 45-day retention) | server ≈ 09-11 → now; **local copy only 09-20 → 09-24** (`replay/year/journal/calib_live`) | BLOCK gate=LONG_UNMATCHED_ONLY: pair, ms time, room flag, **pair indicator snapshot (ctx: price, RSI, ADX, EMAs, ATR, range, volume)**; the SCAN header has BTC RSI / ADX / ADX-prev / slope / off30d / bull | **none** (rebuildable: engine pattern functions on ctx + SCAN + rebuilt BTC RSI-prev / ATR / EMA13-50 gap) | yes | yes (88 rows → 33 episodes); the server copy 09-25 → 10-26 needs the EB log bundle, broken since 10-19 |
| (c) CloudWatch `web.stdout.log` (14-day retention) | **2026-10-12 16:29 → 10-26 05:57** | `[LONG_UNMATCHED_ONLY] PAIR: LONG blocked — matched a pattern (c_any=…, w_any=…)` + `[OPEN_FILTERED] … LONG <conf>`: pair, ms time, confidence | **family only (C / W / C+W)** | none (rebuilt from klines; price = last closed 1m close) | yes (117 rows → 76 episodes) |
| (d) `reports/` July phantom CSVs | 07-06 → 07-26 | as (a) phantoms | yes | yes | same rows as (a) |
| engine replay journals yr6 / yr6a (today's config at 01419d4) | 2025-01-01 → 2026-10-11, 3 seeds | same BLOCK lines as the live journal | rebuilt (99.8 % of refusals rebuild as matched) | yes | backtest, refute-only (32,919 rows → 15,789 episodes) |

AWS access worked (read-only `aws logs filter-log-events`); nothing was written.

**Scout-line proposal (observe-only, NOT built):** a `LONG_UNMATCHED_ONLY` refusal recorder. The block already computes everything at the refusal site (`services/trading_engine.py` ~L12889–12950: `_um_pats` / `_um_cohort` and the full `_um_ef` entry-field dict are built and then handed to the retired `_seed_phantom_flip` no-op). It needs: ① one journal line per refusal: `_djournal.note('MATCHED_LONG', pair, price=current_price, conf, labels=_um_pats, ef=_um_ef, room)` (file-only, no DB, no trading-path effect) — or a `matched_long_refusals` table if it must ride the CSV export (D12: real model columns); ② the decisions export carrying those rows un-aggregated (pair, second, price, labels, entry_* stamps); ③ a scout script pricing each row with the ML replica at today's exit, one per pair at a time, with the frozen cells below; ④ D11/D12 if it gets a UI table (both text exports + CSV columns). Cost: ~650 bytes per refusal (≈ 10–25 / day live).

## 1. Plain-language summary

- **Every refused matched long we can recover, priced at today's exit, loses money.** Live records (July phantoms, Sep journal, Oct CloudWatch): 318 shadow trades on 36 days · WR 48.4 % · **-0.214 %/fill**, versus the unmatched longs the bot actually took on the same days (-0.001 %/fill, N 58). The block is still doing its job: the matched longs are ~0.21 pp/fill worse.
- It holds in **each live period separately** (July -0.257, Sep +0.053 on only 33 trades / 5 days, Oct -0.214), for C, W and C+W alike, and in every BTC regime / 1h-slope / 24h / 30d-high / breadth split.
- The **backtest** (21 months, 3 seeds, 15,789 shadow trades) also has matched longs losing (-0.134 %/fill, WR 57.1 %) — but there the unmatched longs lose about as much (-0.158), so the backtest says the block neither helps nor hurts; it cannot support allowing anything (its absolute numbers are negative everywhere).
- **Least-bad cell: W2 without any C pattern** (BTC RSI 50–65 tailwind). Live 82 · 58.5 % · −0.05 %/fill, still below its control; it is the same cell whose July re-enable reverted (22 · 45 % · −$250). Its positive sub-slices (STRONG_BULL, near the 30-day high, BTC 1h rising) are 32–53 trades with one or two days carrying more than all of the profit.
- **No cell clears the bar; none is a probe candidate.** Recommendation: keep `long_unmatched_only = true`. If the operator wants the question kept alive, build the observe-only refusal recorder above (no trading change) and read two frozen cells at N ≥ 30 / 10 days.

## 2. Data, parity and pricing checks

| cohort | refusals | blocked by today's post-block gates | priced | one-per-pair episodes | tick / 1m paths |
|---|---|---|---|---|---|
| PJUL | 303 | LONG_RSI_MOM_LOADX 45, LONG_MEGACAP_BLOCK 25, LONG_HEAT_BLOCK 14, PAIR_BLACKLIST 1 | 295 | 209 | 267 / 27 |
| JSEP | 88 | LONG_HEAT_BLOCK 18, LONG_MEGACAP_BLOCK 18 | 88 | 33 | 88 / 0 |
| CWOCT | 117 | 0 | 117 | 76 | 11 / 105 |
| RPL | 32,919 | LONG_MEGACAP_BLOCK 4153, LONG_HEAT_BLOCK 3990 | 32,919 | 15,789 | 29,005 / 3,670 |

- **Cohort = engine events, not a proxy** (FATAL rule): every shadow is a refusal the engine itself logged / journalled / seeded (live) or journalled (replay) — the same bars by construction. Labels are the engine's own `_compute_pattern_c/w_match`.
- Label rebuild parity: replay refusals rebuilt as matched **99.8 %** (journal ctx + SCAN BTC RSI/ADX exact; BTC RSI-prev / ATR / EMA13-50 gap rebuilt — ADX median |Δ| 0, RSI-prev median |Δ| 0.03 vs replay fill stamps); Sep journal 100 %; CloudWatch family (C / W / C+W) rebuilt = logged on **90.6 %** (specific labels used only on those rows, the logged family elsewhere).
- Kline-only rebuild on the **real master fills** since 09-20 (the CloudWatch path): family parity 81.3 %; 12 of 51 unmatched fills rebuild as matched (BTC RSI from the forming bar, median |Δ| 0.6) → specific labels from klines are approximate; the logged family is authoritative.
- July phantoms keep the **live-recorded** label: re-labelling them with today's thresholds agrees on 53 % — the gap is the July stamps lacking `entry_btc_trend_gap_pct` (W2 / C2 need it), not a threshold change.
- Replica on the 167 master KEEP control fills (own fee): +0.132 %/fill vs master stack +0.126; base exit reason 88 % equal (mismatches = BASE/B1 fills that traded older exit stacks; DL 380's 100 % is the as-was replica). PJUL episodes: July phantom exit -0.066 vs today's replica -0.257 — today's exit does **not** rescue them.
- Probe routing: shadows never traded, so none is probe-routed; controls and MREAL exclude every `is_probe` fill (W2 lesson). Pattern labels are kept separate from routing.

## 3. Per-cell tables

Bar for an *allow* cell: N ≥ 30 on ≥ 10 days · WR ≥ 61.5 % · avg ≥ +0.10 %/fill · day-block bootstrap lower bound > 0 at the Bonferroni level over all cells tested (LIVE m = 111, RPL m = 160) · ≥ the same-days unmatched control · no day / pair ≥ 50 % of the net · then a 30–50 % haircut. LIVE = PJUL + JSEP + CWOCT (decisive). RPL = replay (refute-only; N per seed, CI clustered by day across the 3 seeds). Δ vs ctl = cell avg − the unmatched control's avg on the same days (same replica, same fee). 'top day / pair share' is shown only when the net is positive. Regime cells are market-wide: read the **days** column.

### 3a. By pattern (single labels, families, C-free W, exact combos ≥ 30)

| cell | LIVE N | days | WR % | avg %/fill | LB (Bonf.) | same-days ctl | Δ vs ctl | top day / pair share | RPL N/seed | RPL days | RPL WR | RPL avg | RPL Δ vs ctl | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL matched | 318 | 36 | 48.4 | -0.214 | -0.401 | -0.001 | -0.213 | – | 5251 | 622 | 57.1 | -0.133 | +0.022 | **DECIDED no** — LIVE avg -0.214 ≤ 0 |
| C-only (no W) | 97 | 27 | 50.5 | -0.231 | -0.486 | +0.122 | -0.353 | – | 1214 | 513 | 57.1 | -0.117 | +0.029 | **DECIDED no** — LIVE avg -0.231 ≤ 0 |
| W-only (C-free) | 164 | 34 | 48.8 | -0.173 | -0.383 | +0.041 | -0.214 | – | 3484 | 596 | 57.3 | -0.136 | +0.017 | **DECIDED no** — LIVE avg -0.173 ≤ 0 |
| C+W | 57 | 19 | 43.9 | -0.305 | -0.608 | +0.031 | -0.336 | – | 552 | 410 | 56.2 | -0.157 | +0.005 | **DECIDED no** — LIVE avg -0.305 ≤ 0 |
| has C2 | 54 | 17 | 50.0 | -0.197 | -0.461 | +0.005 | -0.202 | – | 387 | 278 | 56.9 | -0.135 | +0.022 | **DECIDED no** — LIVE avg -0.197 ≤ 0 |
| has C4 | 57 | 23 | 49.1 | -0.281 | -0.579 | -0.083 | -0.198 | – | 523 | 285 | 53.5 | -0.138 | +0.046 | **DECIDED no** — LIVE avg -0.281 ≤ 0 |
| has C5 | 7 | 7 | 57.1 | -0.114 | -0.741 | +0.589 | -0.703 | – | 9 | 15 | 55.6 | -0.038 | +0.031 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| has C6 | 27 | 10 | 48.1 | -0.254 | -0.613 | +0.240 | -0.494 | – | 670 | 289 | 60.0 | -0.095 | +0.025 | **DECIDED no** — LIVE avg -0.254 ≤ 0 |
| has C7 | 9 | 6 | 44.4 | -0.254 | -1.135 | +0.582 | -0.836 | – | 288 | 227 | 56.8 | -0.170 | -0.012 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| has C8 | 17 | 4 | 23.5 | -0.553 | -0.978 | +0.098 | -0.651 | – | 20 | 35 | 54.1 | -0.133 | +0.037 | **DECIDED no** — LIVE avg -0.553 ≤ 0 |
| has C9 | 14 | 10 | 42.9 | -0.345 | -0.770 | +0.006 | -0.351 | – | 115 | 96 | 53.8 | -0.144 | +0.080 | **DECIDED no** — LIVE avg -0.345 ≤ 0 |
| has W1 | 37 | 16 | 29.7 | -0.378 | -0.693 | +0.252 | -0.630 | – | 378 | 338 | 58.6 | -0.128 | +0.017 | **DECIDED no** — LIVE avg -0.378 ≤ 0 |
| has W2 | 86 | 24 | 59.3 | -0.027 | -0.353 | +0.043 | -0.070 | – | 2644 | 526 | 58.0 | -0.131 | +0.008 | **DECIDED no** — LIVE avg -0.027 ≤ 0 |
| has W4 | 68 | 23 | 47.1 | -0.230 | -0.464 | +0.068 | -0.298 | – | 974 | 480 | 59.3 | -0.120 | +0.021 | **DECIDED no** — LIVE avg -0.230 ≤ 0 |
| has W5 | 5 | 5 | 40.0 | -0.321 | -0.734 | -0.274 | -0.047 | – | 316 | 306 | 54.9 | -0.177 | -0.018 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| has W6 | 64 | 21 | 46.9 | -0.253 | -0.507 | +0.079 | -0.332 | – | 1005 | 407 | 53.1 | -0.185 | -0.006 | **DECIDED no** — LIVE avg -0.253 ≤ 0 |
| W1 C-free | 19 | 13 | 36.8 | -0.240 | -0.810 | +0.187 | -0.427 | – | 285 | 288 | 58.8 | -0.132 | +0.025 | **DECIDED no** — LIVE avg -0.240 ≤ 0 |
| W2 C-free | 82 | 24 | 58.5 | -0.046 | -0.365 | +0.043 | -0.089 | – | 2556 | 522 | 58.2 | -0.127 | +0.012 | **DECIDED no** — LIVE avg -0.046 ≤ 0 |
| W4 C-free | 42 | 21 | 42.9 | -0.220 | -0.519 | +0.223 | -0.443 | – | 786 | 441 | 59.5 | -0.126 | +0.011 | **DECIDED no** — LIVE avg -0.220 ≤ 0 |
| W6 C-free | 51 | 19 | 49.0 | -0.186 | -0.490 | +0.079 | -0.265 | – | 789 | 374 | 52.6 | -0.187 | -0.005 | **DECIDED no** — LIVE avg -0.186 ≤ 0 |
| W1 alone | 11 | 7 | 27.3 | -0.452 | -1.099 | +0.262 | -0.714 | – | 67 | 103 | 54.0 | -0.175 | -0.029 | **DECIDED no** — LIVE avg -0.452 ≤ 0 |
| W2 alone | 50 | 21 | 60.0 | -0.034 | -0.428 | +0.020 | -0.054 | – | 1598 | 481 | 58.4 | -0.118 | +0.016 | **DECIDED no** — LIVE avg -0.034 ≤ 0 |
| W4 alone | 31 | 18 | 38.7 | -0.281 | -0.640 | +0.463 | -0.744 | – | 251 | 265 | 61.4 | -0.066 | +0.069 | **DECIDED no** — LIVE avg -0.281 ≤ 0 |
| W6 alone | 34 | 16 | 47.1 | -0.208 | -0.586 | +0.273 | -0.481 | – | 473 | 295 | 51.4 | -0.209 | -0.013 | **DECIDED no** — LIVE avg -0.208 ≤ 0 |
| C2 alone | 22 | 10 | 54.5 | -0.175 | -0.600 | +0.092 | -0.267 | – | 158 | 177 | 54.5 | -0.143 | -0.007 | **DECIDED no** — LIVE avg -0.175 ≤ 0 |
| C4 alone | 26 | 17 | 53.8 | -0.149 | -0.614 | +0.056 | -0.205 | – | 310 | 233 | 50.7 | -0.172 | +0.012 | **DECIDED no** — LIVE avg -0.149 ≤ 0 |
| C6 alone | 17 | 9 | 58.8 | -0.174 | -0.605 | +0.285 | -0.459 | – | 472 | 258 | 60.6 | -0.087 | +0.032 | **DECIDED no** — LIVE avg -0.174 ≤ 0 |
| has C1 | – | – | – | – | – | – | – | – | 23 | 35 | 63.8 | -0.042 | +0.173 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| has W3 | – | – | – | – | – | – | – | – | 37 | 49 | 65.2 | +0.012 | +0.180 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W3 C-free | – | – | – | – | – | – | – | – | 27 | 44 | 70.4 | +0.120 | +0.285 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W5 C-free | – | – | – | – | – | – | – | – | 279 | 290 | 55.6 | -0.157 | -0.011 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W3 alone | – | – | – | – | – | – | – | – | 9 | 15 | 66.7 | +0.162 | +0.367 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W5 alone | – | – | – | – | – | – | – | – | 25 | 51 | 49.3 | -0.189 | -0.174 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C5 alone | – | – | – | – | – | – | – | – | 2 | 3 | 100.0 | +0.203 | +0.166 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C7 alone | – | – | – | – | – | – | – | – | 73 | 77 | 60.5 | -0.130 | +0.055 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C8 alone | – | – | – | – | – | – | – | – | 3 | 7 | 100.0 | +0.561 | +0.694 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W4 | – | – | – | – | – | – | – | – | 410 | 304 | 58.8 | -0.147 | -0.015 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W6 | – | – | – | – | – | – | – | – | 228 | 174 | 52.7 | -0.180 | -0.013 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C6+W4 | – | – | – | – | – | – | – | – | 116 | 135 | 56.3 | -0.122 | +0.024 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W5 | – | – | – | – | – | – | – | – | 95 | 131 | 57.4 | -0.112 | +0.043 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+W6 | – | – | – | – | – | – | – | – | 87 | 103 | 60.4 | -0.114 | +0.110 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C4+C9 | – | – | – | – | – | – | – | – | 85 | 80 | 52.8 | -0.148 | +0.085 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W2 | – | – | – | – | – | – | – | – | 73 | 106 | 65.9 | -0.036 | +0.070 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C7+W6 | – | – | – | – | – | – | – | – | 60 | 71 | 55.0 | -0.166 | +0.034 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W2+W5 | – | – | – | – | – | – | – | – | 46 | 78 | 57.2 | -0.162 | -0.038 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C6+W1 | – | – | – | – | – | – | – | – | 42 | 63 | 61.4 | -0.071 | +0.021 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C4+W4 | – | – | – | – | – | – | – | – | 38 | 60 | 62.8 | -0.002 | +0.106 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+C4 | – | – | – | – | – | – | – | – | 36 | 39 | 63.0 | -0.075 | +0.134 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C7+W2 | – | – | – | – | – | – | – | – | 35 | 41 | 49.5 | -0.436 | -0.297 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W5+W6 | – | – | – | – | – | – | – | – | 29 | 42 | 52.9 | -0.222 | +0.044 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W4 | – | – | – | – | – | – | – | – | 27 | 48 | 56.2 | -0.200 | -0.062 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W2+W4 | – | – | – | – | – | – | – | – | 23 | 48 | 48.6 | -0.308 | -0.274 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+C7 | – | – | – | – | – | – | – | – | 23 | 32 | 73.5 | +0.240 | +0.200 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W4+W5 | – | – | – | – | – | – | – | – | 22 | 36 | 54.5 | -0.202 | -0.048 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+C7+W6 | – | – | – | – | – | – | – | – | 20 | 25 | 48.3 | -0.293 | -0.143 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C7+W2+W6 | – | – | – | – | – | – | – | – | 17 | 23 | 51.0 | -0.143 | +0.171 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W5+W6 | – | – | – | – | – | – | – | – | 17 | 19 | 48.0 | -0.223 | -0.024 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+W4 | – | – | – | – | – | – | – | – | 16 | 31 | 64.6 | -0.183 | -0.150 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W2+W4+W5 | – | – | – | – | – | – | – | – | 15 | 29 | 76.1 | +0.014 | +0.022 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C4+C7+C9 | – | – | – | – | – | – | – | – | 15 | 25 | 68.2 | +0.044 | +0.361 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C6+C7 | – | – | – | – | – | – | – | – | 14 | 21 | 64.3 | -0.207 | +0.070 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C4+W1 | – | – | – | – | – | – | – | – | 14 | 26 | 56.1 | -0.058 | +0.202 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W2+W4+W6 | – | – | – | – | – | – | – | – | 13 | 23 | 70.0 | -0.010 | +0.088 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C7+W5+W6 | – | – | – | – | – | – | – | – | 12 | 16 | 50.0 | -0.300 | -0.054 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C1+W2 | – | – | – | – | – | – | – | – | 12 | 16 | 60.0 | -0.093 | +0.040 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = C2+C4+C9 | – | – | – | – | – | – | – | – | 12 | 17 | 54.3 | -0.201 | -0.077 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| = W1+W5 | – | – | – | – | – | – | – | – | 10 | 24 | 38.7 | -0.385 | -0.205 | **DECIDED no** — LIVE N<10 — no evidence to move the block |

### 3b. Pattern family × regime (BTC regime label · BTC 1h EMA20 slope sign · BTC 24h return sign · distance from 30-day high · breadth = share of top-50 above 1h EMA50; definitions = the ML_REGIME_WL study's `study_ml_regime_wl_build.build`)

| cell | LIVE N | days | WR % | avg %/fill | LB (Bonf.) | same-days ctl | Δ vs ctl | top day / pair share | RPL N/seed | RPL days | RPL WR | RPL avg | RPL Δ vs ctl | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL matched × regime_lbl=CHOPPY_FLAT | 35 | 6 | 34.3 | -0.419 | -0.830 | +0.998 | -1.417 | – | – | – | – | – | – | **DECIDED no** — LIVE avg -0.419 ≤ 0 |
| ALL matched × regime_lbl=HEALTHY_BEAR | 29 | 5 | 62.1 | -0.069 | -0.192 | +0.998 | -1.067 | – | – | – | – | – | – | **DECIDED no** — LIVE avg -0.069 ≤ 0 |
| ALL matched × regime_lbl=HEALTHY_BULL | 153 | 33 | 47.1 | -0.220 | -0.453 | -0.012 | -0.208 | – | 3100 | 591 | 55.4 | -0.158 | +0.000 | **DECIDED no** — LIVE avg -0.220 ≤ 0 |
| ALL matched × regime_lbl=STRONG_BULL | 90 | 26 | 56.7 | -0.122 | -0.377 | -0.055 | -0.067 | – | 2136 | 472 | 59.8 | -0.096 | +0.025 | **DECIDED no** — LIVE avg -0.122 ≤ 0 |
| ALL matched × z_1h=1h<=0 | 169 | 25 | 43.2 | -0.268 | -0.448 | +0.018 | -0.286 | – | 2314 | 426 | 56.1 | -0.157 | +0.021 | **DECIDED no** — LIVE avg -0.268 ≤ 0 |
| ALL matched × z_1h=1h>0 | 149 | 24 | 54.4 | -0.153 | -0.429 | -0.028 | -0.125 | – | 2937 | 469 | 57.9 | -0.115 | +0.021 | **DECIDED no** — LIVE avg -0.153 ≤ 0 |
| ALL matched × z_24h=24h<=0 | 180 | 26 | 42.8 | -0.242 | -0.440 | +0.009 | -0.251 | – | 2243 | 405 | 55.8 | -0.156 | +0.033 | **DECIDED no** — LIVE avg -0.242 ≤ 0 |
| ALL matched × z_24h=24h>0 | 138 | 26 | 55.8 | -0.178 | -0.410 | -0.022 | -0.156 | – | 3008 | 465 | 58.1 | -0.117 | +0.024 | **DECIDED no** — LIVE avg -0.178 ≤ 0 |
| ALL matched × z_30d=off30(-10,-3] | 217 | 26 | 43.8 | -0.299 | -0.473 | -0.050 | -0.249 | – | 1988 | 327 | 55.9 | -0.139 | +0.032 | **DECIDED no** — LIVE avg -0.299 ≤ 0 |
| ALL matched × z_30d=off30>-3 | 101 | 12 | 58.4 | -0.032 | -0.412 | +0.154 | -0.186 | – | 1001 | 169 | 57.0 | -0.139 | -0.017 | **DECIDED no** — LIVE avg -0.032 ≤ 0 |
| ALL matched × z_br=breadth<=50 | 210 | 29 | 43.8 | -0.248 | -0.455 | -0.016 | -0.232 | – | 2999 | 472 | 56.9 | -0.137 | +0.031 | **DECIDED no** — LIVE avg -0.248 ≤ 0 |
| ALL matched × z_br=breadth>50 | 106 | 18 | 57.5 | -0.150 | -0.427 | +0.053 | -0.203 | – | 2252 | 385 | 57.5 | -0.129 | +0.000 | **DECIDED no** — LIVE avg -0.150 ≤ 0 |
| W-only (C-free) × regime_lbl=HEALTHY_BULL | 100 | 29 | 42.0 | -0.252 | -0.496 | +0.032 | -0.284 | – | 2142 | 536 | 55.8 | -0.157 | +0.002 | **DECIDED no** — LIVE avg -0.252 ≤ 0 |
| W-only (C-free) × regime_lbl=STRONG_BULL | 45 | 19 | 66.7 | +0.013 | -0.349 | +0.039 | -0.026 | 4.08 / 2.76 | 1333 | 407 | 59.8 | -0.099 | +0.013 | **DECIDED no** — LIVE below same-days control (Δ -0.026) |
| W-only (C-free) × z_1h=1h<=0 | 79 | 21 | 43.0 | -0.247 | -0.492 | +0.104 | -0.351 | – | 1354 | 353 | 55.8 | -0.166 | +0.009 | **DECIDED no** — LIVE avg -0.247 ≤ 0 |
| W-only (C-free) × z_1h=1h>0 | 85 | 20 | 54.1 | -0.104 | -0.449 | -0.088 | -0.016 | – | 2131 | 450 | 58.2 | -0.116 | +0.021 | **DECIDED no** — LIVE avg -0.104 ≤ 0 |
| W-only (C-free) × z_24h=24h<=0 | 91 | 23 | 44.0 | -0.177 | -0.413 | +0.115 | -0.292 | – | 1350 | 357 | 55.6 | -0.158 | +0.026 | **DECIDED no** — LIVE avg -0.177 ≤ 0 |
| W-only (C-free) × z_24h=24h>0 | 73 | 20 | 54.8 | -0.168 | -0.506 | -0.083 | -0.085 | – | 2135 | 429 | 58.4 | -0.122 | +0.015 | **DECIDED no** — LIVE avg -0.168 ≤ 0 |
| W-only (C-free) × z_30d=off30(-10,-3] | 113 | 25 | 45.1 | -0.251 | -0.444 | +0.003 | -0.254 | – | 1337 | 295 | 56.2 | -0.140 | +0.034 | **DECIDED no** — LIVE avg -0.251 ≤ 0 |
| W-only (C-free) × z_30d=off30>-3 | 51 | 10 | 56.9 | -0.001 | -0.462 | +0.154 | -0.155 | – | 668 | 154 | 58.0 | -0.125 | -0.017 | **DECIDED no** — LIVE avg -0.001 ≤ 0 |
| W-only (C-free) × z_br=breadth<=50 | 110 | 27 | 46.4 | -0.180 | -0.413 | +0.039 | -0.219 | – | 1866 | 412 | 56.7 | -0.138 | +0.031 | **DECIDED no** — LIVE avg -0.180 ≤ 0 |
| W-only (C-free) × z_br=breadth>50 | 52 | 16 | 53.8 | -0.162 | -0.565 | +0.053 | -0.215 | – | 1618 | 365 | 58.0 | -0.132 | -0.011 | **DECIDED no** — LIVE avg -0.162 ≤ 0 |
| C-only (no W) × regime_lbl=CHOPPY_FLAT | 15 | 4 | 40.0 | -0.337 | -0.492 | +0.998 | -1.335 | – | – | – | – | – | – | **DECIDED no** — LIVE avg -0.337 ≤ 0 |
| C-only (no W) × regime_lbl=HEALTHY_BEAR | 13 | 3 | 61.5 | -0.123 | -0.163 | +0.998 | -1.121 | – | – | – | – | – | – | **DECIDED no** — LIVE avg -0.123 ≤ 0 |
| C-only (no W) × regime_lbl=HEALTHY_BULL | 37 | 15 | 56.8 | -0.107 | -0.514 | +0.013 | -0.120 | – | 645 | 384 | 53.6 | -0.164 | +0.006 | **DECIDED no** — LIVE avg -0.107 ≤ 0 |
| C-only (no W) × regime_lbl=STRONG_BULL | 26 | 14 | 53.8 | -0.249 | -0.736 | +0.192 | -0.441 | – | 565 | 309 | 61.1 | -0.062 | +0.057 | **DECIDED no** — LIVE avg -0.249 ≤ 0 |
| C-only (no W) × z_1h=1h<=0 | 57 | 16 | 49.1 | -0.191 | -0.490 | +0.068 | -0.259 | – | 640 | 309 | 56.6 | -0.124 | +0.042 | **DECIDED no** — LIVE avg -0.191 ≤ 0 |
| C-only (no W) × z_1h=1h>0 | 40 | 15 | 52.5 | -0.288 | -0.647 | +0.331 | -0.619 | – | 574 | 304 | 57.6 | -0.108 | +0.023 | **DECIDED no** — LIVE avg -0.288 ≤ 0 |
| C-only (no W) × z_24h=24h<=0 | 56 | 16 | 44.6 | -0.252 | -0.604 | +0.095 | -0.347 | – | 599 | 286 | 56.2 | -0.140 | +0.023 | **DECIDED no** — LIVE avg -0.252 ≤ 0 |
| C-only (no W) × z_24h=24h>0 | 41 | 17 | 58.5 | -0.202 | -0.528 | +0.313 | -0.515 | – | 615 | 327 | 58.0 | -0.094 | +0.033 | **DECIDED no** — LIVE avg -0.202 ≤ 0 |
| C-only (no W) × z_30d=off30(-10,-3] | 65 | 19 | 44.6 | -0.336 | -0.617 | +0.107 | -0.443 | – | 454 | 242 | 57.3 | -0.105 | +0.070 | **DECIDED no** — LIVE avg -0.336 ≤ 0 |
| C-only (no W) × z_30d=off30>-3 | 32 | 9 | 62.5 | -0.016 | -0.482 | +0.157 | -0.173 | – | 230 | 114 | 56.5 | -0.137 | -0.020 | **DECIDED no** — LIVE avg -0.016 ≤ 0 |
| C-only (no W) × z_br=breadth<=50 | 67 | 19 | 46.3 | -0.254 | -0.553 | +0.096 | -0.350 | – | 756 | 352 | 57.1 | -0.116 | +0.039 | **DECIDED no** — LIVE avg -0.254 ≤ 0 |
| C-only (no W) × z_br=breadth>50 | 30 | 12 | 60.0 | -0.179 | -0.515 | +0.242 | -0.421 | – | 458 | 247 | 57.2 | -0.118 | +0.008 | **DECIDED no** — LIVE avg -0.179 ≤ 0 |
| C+W × regime_lbl=CHOPPY_FLAT | 13 | 4 | 30.8 | -0.484 | -0.940 | +0.998 | -1.482 | – | – | – | – | – | – | **DECIDED no** — LIVE avg -0.484 ≤ 0 |
| C+W × regime_lbl=HEALTHY_BULL | 16 | 13 | 56.2 | -0.285 | -0.768 | -0.319 | +0.034 | – | 313 | 284 | 56.1 | -0.153 | +0.031 | **DECIDED no** — LIVE avg -0.285 ≤ 0 |
| C+W × regime_lbl=STRONG_BULL | 19 | 9 | 36.8 | -0.266 | -0.698 | +0.500 | -0.766 | – | 238 | 224 | 56.4 | -0.161 | -0.040 | **DECIDED no** — LIVE avg -0.266 ≤ 0 |
| C+W × z_1h=1h<=0 | 33 | 12 | 33.3 | -0.453 | -0.708 | +0.045 | -0.498 | – | 320 | 242 | 56.8 | -0.184 | -0.015 | **DECIDED no** — LIVE avg -0.453 ≤ 0 |
| C+W × z_1h=1h>0 | 24 | 11 | 58.3 | -0.101 | -0.612 | +0.131 | -0.232 | – | 232 | 230 | 55.5 | -0.119 | +0.009 | **DECIDED no** — LIVE avg -0.101 ≤ 0 |
| C+W × z_24h=24h<=0 | 33 | 12 | 36.4 | -0.403 | -0.703 | +0.109 | -0.512 | – | 294 | 222 | 56.1 | -0.181 | +0.010 | **DECIDED no** — LIVE avg -0.403 ≤ 0 |
| C+W × z_24h=24h>0 | 24 | 12 | 54.2 | -0.170 | -0.627 | +0.109 | -0.279 | – | 258 | 252 | 56.4 | -0.129 | +0.016 | **DECIDED no** — LIVE avg -0.170 ≤ 0 |
| C+W × z_30d=off30(-10,-3] | 39 | 12 | 38.5 | -0.376 | -0.706 | -0.141 | -0.235 | – | 197 | 183 | 51.0 | -0.210 | -0.024 | **DECIDED no** — LIVE avg -0.376 ≤ 0 |
| C+W × z_30d=off30>-3 | 18 | 7 | 55.6 | -0.150 | -0.578 | +0.667 | -0.817 | – | 103 | 89 | 51.5 | -0.233 | -0.116 | **DECIDED no** — LIVE avg -0.150 ≤ 0 |
| C+W × z_br=breadth<=50 | 33 | 12 | 30.3 | -0.462 | -0.798 | -0.066 | -0.396 | – | 377 | 273 | 57.4 | -0.169 | +0.009 | **DECIDED no** — LIVE avg -0.462 ≤ 0 |
| C+W × z_br=breadth>50 | 24 | 10 | 62.5 | -0.088 | -0.427 | +0.131 | -0.219 | – | 176 | 190 | 53.7 | -0.130 | -0.009 | **DECIDED no** — LIVE avg -0.088 ≤ 0 |
| W2 C-free × regime_lbl=HEALTHY_BULL | 50 | 16 | 48.0 | -0.134 | -0.528 | -0.009 | -0.125 | – | 1268 | 382 | 56.3 | -0.158 | -0.023 | **DECIDED no** — LIVE avg -0.134 ≤ 0 |
| W2 C-free × regime_lbl=STRONG_BULL | 32 | 13 | 75.0 | +0.092 | -0.227 | +0.107 | -0.015 | 0.80 / 0.38 | 1288 | 389 | 60.1 | -0.096 | +0.014 | **DECIDED no** — LIVE below same-days control (Δ -0.015) |
| W2 C-free × z_1h=1h<=0 | 29 | 11 | 48.3 | -0.200 | -0.506 | +0.127 | -0.327 | – | 797 | 239 | 56.9 | -0.154 | -0.005 | **DECIDED no** — LIVE avg -0.200 ≤ 0 |
| W2 C-free × z_1h=1h>0 | 53 | 16 | 64.2 | +0.039 | -0.386 | -0.129 | +0.168 | 2.00 / 1.14 | 1760 | 415 | 58.8 | -0.115 | +0.012 | **DECIDED no** — LIVE +0.039 fails [avg<+0.10;LB_bonf<=0;conc>=50%]; replay refutes (-0.115, Δ +0.012) |
| W2 C-free × z_24h=24h<=0 | 39 | 14 | 48.7 | -0.103 | -0.418 | +0.076 | -0.179 | – | 861 | 262 | 56.1 | -0.145 | +0.019 | **DECIDED no** — LIVE avg -0.103 ≤ 0 |
| W2 C-free × z_24h=24h>0 | 43 | 13 | 67.4 | +0.006 | -0.462 | -0.118 | +0.124 | 15.53 / 7.75 | 1695 | 388 | 59.3 | -0.118 | +0.007 | **DECIDED no** — LIVE +0.006 fails [avg<+0.10;LB_bonf<=0;conc>=50%]; replay refutes (-0.118, Δ +0.007) |
| W2 C-free × z_30d=off30(-10,-3] | 50 | 16 | 52.0 | -0.192 | -0.442 | -0.004 | -0.188 | – | 947 | 243 | 57.9 | -0.114 | +0.041 | **DECIDED no** — LIVE avg -0.192 ≤ 0 |
| W2 C-free × z_30d=off30>-3 | 32 | 8 | 68.8 | +0.182 | -0.421 | +0.154 | +0.028 | 0.71 / 0.40 | 508 | 135 | 58.8 | -0.118 | -0.022 | **DECIDED no** — LIVE +0.182 fails [days<10;LB_bonf<=0;conc>=50%]; replay refutes (-0.118, Δ -0.022) |
| W2 C-free × z_br=breadth<=50 | 44 | 16 | 56.8 | -0.017 | -0.430 | -0.059 | +0.042 | – | 1250 | 304 | 58.2 | -0.116 | +0.048 | **DECIDED no** — LIVE avg -0.017 ≤ 0 |
| W2 C-free × z_br=breadth>50 | 37 | 11 | 59.5 | -0.099 | -0.511 | +0.018 | -0.117 | – | 1306 | 331 | 58.2 | -0.137 | -0.035 | **DECIDED no** — LIVE avg -0.099 ≤ 0 |
| W4 C-free × regime_lbl=HEALTHY_BULL | 29 | 14 | 37.9 | -0.351 | -0.648 | +0.125 | -0.476 | – | 485 | 339 | 60.5 | -0.118 | +0.034 | **DECIDED no** — LIVE avg -0.351 ≤ 0 |
| W4 C-free × regime_lbl=STRONG_BULL | 10 | 8 | 60.0 | +0.111 | -0.630 | +0.403 | -0.292 | 1.50 / 1.43 | 295 | 245 | 58.4 | -0.130 | -0.042 | **DECIDED no** — LIVE below same-days control (Δ -0.292) |
| W4 C-free × z_1h=1h<=0 | 15 | 10 | 33.3 | -0.392 | -0.803 | +0.532 | -0.924 | – | 260 | 192 | 62.8 | -0.115 | +0.031 | **DECIDED no** — LIVE avg -0.392 ≤ 0 |
| W4 C-free × z_1h=1h>0 | 27 | 14 | 48.1 | -0.125 | -0.530 | +0.091 | -0.216 | – | 526 | 319 | 57.9 | -0.131 | -0.010 | **DECIDED no** — LIVE avg -0.125 ≤ 0 |
| W4 C-free × z_24h=24h<=0 | 21 | 12 | 42.9 | -0.220 | -0.595 | +0.417 | -0.637 | – | 267 | 203 | 62.6 | -0.085 | +0.075 | **DECIDED no** — LIVE avg -0.220 ≤ 0 |
| W4 C-free × z_24h=24h>0 | 21 | 11 | 42.9 | -0.219 | -0.606 | +0.028 | -0.247 | – | 519 | 308 | 57.9 | -0.147 | -0.026 | **DECIDED no** — LIVE avg -0.219 ≤ 0 |
| W4 C-free × z_30d=off30(-10,-3] | 26 | 13 | 38.5 | -0.314 | -0.654 | +0.214 | -0.528 | – | 272 | 199 | 60.8 | -0.114 | +0.025 | **DECIDED no** — LIVE avg -0.314 ≤ 0 |
| W4 C-free × z_30d=off30>-3 | 16 | 9 | 50.0 | -0.068 | -0.508 | +0.231 | -0.299 | – | 131 | 101 | 59.3 | -0.077 | +0.020 | **DECIDED no** — LIVE avg -0.068 ≤ 0 |
| W4 C-free × z_br=breadth<=50 | 27 | 15 | 33.3 | -0.344 | -0.687 | -0.010 | -0.334 | – | 382 | 255 | 61.5 | -0.127 | +0.013 | **DECIDED no** — LIVE avg -0.344 ≤ 0 |
| W4 C-free × z_br=breadth>50 | 15 | 10 | 60.0 | +0.003 | -0.721 | +0.380 | -0.377 | 41.19 / 41.19 | 404 | 251 | 57.6 | -0.125 | -0.008 | **DECIDED no** — LIVE below same-days control (Δ -0.377) |
| W6 C-free × regime_lbl=HEALTHY_BULL | 41 | 18 | 48.8 | -0.186 | -0.458 | +0.054 | -0.240 | – | 789 | 374 | 52.6 | -0.187 | -0.005 | **DECIDED no** — LIVE avg -0.186 ≤ 0 |
| W6 C-free × z_1h=1h<=0 | 31 | 12 | 45.2 | -0.192 | -0.571 | +0.045 | -0.237 | – | 504 | 239 | 51.7 | -0.209 | +0.005 | **DECIDED no** — LIVE avg -0.192 ≤ 0 |
| W6 C-free × z_1h=1h>0 | 20 | 9 | 55.0 | -0.176 | -0.762 | +0.102 | -0.278 | – | 285 | 178 | 54.3 | -0.149 | +0.010 | **DECIDED no** — LIVE avg -0.176 ≤ 0 |
| W6 C-free × z_24h=24h<=0 | 38 | 12 | 44.7 | -0.202 | -0.471 | +0.038 | -0.240 | – | 442 | 218 | 51.6 | -0.206 | +0.018 | **DECIDED no** — LIVE avg -0.202 ≤ 0 |
| W6 C-free × z_24h=24h>0 | 13 | 8 | 61.5 | -0.139 | -0.832 | +0.180 | -0.319 | – | 347 | 209 | 53.8 | -0.164 | -0.008 | **DECIDED no** — LIVE avg -0.139 ≤ 0 |
| W6 C-free × z_30d=off30(-10,-3] | 33 | 14 | 48.5 | -0.184 | -0.565 | +0.073 | -0.257 | – | 342 | 173 | 48.4 | -0.230 | -0.012 | **DECIDED no** — LIVE avg -0.184 ≤ 0 |
| W6 C-free × z_30d=off30>-3 | 18 | 5 | 50.0 | -0.190 | -0.601 | +0.099 | -0.289 | – | 139 | 72 | 55.7 | -0.171 | -0.075 | **DECIDED no** — LIVE avg -0.190 ≤ 0 |
| W6 C-free × z_br=breadth<=50 | 38 | 16 | 50.0 | -0.151 | -0.480 | +0.045 | -0.196 | – | 568 | 255 | 49.7 | -0.219 | +0.002 | **DECIDED no** — LIVE avg -0.151 ≤ 0 |
| W6 C-free × z_br=breadth>50 | 13 | 7 | 46.2 | -0.287 | -0.993 | +0.189 | -0.476 | – | 221 | 154 | 60.1 | -0.107 | +0.011 | **DECIDED no** — LIVE avg -0.287 ≤ 0 |
| ALL matched × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 2262 | 236 | 58.3 | -0.126 | +0.038 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W-only (C-free) × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 1479 | 212 | 58.0 | -0.136 | +0.019 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C-only (no W) × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 531 | 200 | 57.2 | -0.118 | +0.028 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C+W × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 252 | 158 | 62.3 | -0.084 | +0.073 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W2 C-free × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 1101 | 187 | 58.2 | -0.142 | +0.000 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W4 C-free × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 383 | 162 | 58.6 | -0.151 | -0.005 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W6 C-free × z_30d=off30<=-10 | – | – | – | – | – | – | – | – | 308 | 144 | 55.8 | -0.147 | +0.037 | **DECIDED no** — LIVE N<10 — no evidence to move the block |

### 3c. By era (LIVE is all 2026-H2; per live period see 3d)

| cell | LIVE N | days | WR % | avg %/fill | LB (Bonf.) | same-days ctl | Δ vs ctl | top day / pair share | RPL N/seed | RPL days | RPL WR | RPL avg | RPL Δ vs ctl | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL matched × era=2026-H2 | 318 | 36 | 48.4 | -0.214 | -0.401 | -0.001 | -0.213 | – | 522 | 95 | 59.2 | -0.089 | -0.071 | **DECIDED no** — LIVE avg -0.214 ≤ 0 |
| W-only (C-free) × era=2026-H2 | 164 | 34 | 48.8 | -0.173 | -0.383 | +0.041 | -0.214 | – | 322 | 89 | 58.0 | -0.117 | -0.101 | **DECIDED no** — LIVE avg -0.173 ≤ 0 |
| C-only (no W) × era=2026-H2 | 97 | 27 | 50.5 | -0.231 | -0.486 | +0.122 | -0.353 | – | 146 | 69 | 62.2 | -0.037 | -0.042 | **DECIDED no** — LIVE avg -0.231 ≤ 0 |
| C+W × era=2026-H2 | 57 | 19 | 43.9 | -0.305 | -0.608 | +0.031 | -0.336 | – | 53 | 56 | 58.1 | -0.057 | -0.013 | **DECIDED no** — LIVE avg -0.305 ≤ 0 |
| ALL matched × era=2025 | – | – | – | – | – | – | – | – | 3393 | 351 | 56.4 | -0.140 | +0.073 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W-only (C-free) × era=2025 | – | – | – | – | – | – | – | – | 2308 | 336 | 57.0 | -0.133 | +0.079 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C-only (no W) × era=2025 | – | – | – | – | – | – | – | – | 739 | 302 | 56.2 | -0.123 | +0.083 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C+W × era=2025 | – | – | – | – | – | – | – | – | 346 | 233 | 52.7 | -0.224 | -0.009 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| ALL matched × era=2026-H1 | – | – | – | – | – | – | – | – | 1336 | 176 | 58.1 | -0.135 | -0.047 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| W-only (C-free) × era=2026-H1 | – | – | – | – | – | – | – | – | 854 | 171 | 57.7 | -0.151 | -0.066 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C-only (no W) × era=2026-H1 | – | – | – | – | – | – | – | – | 329 | 142 | 56.7 | -0.137 | -0.066 | **DECIDED no** — LIVE N<10 — no evidence to move the block |
| C+W × era=2026-H1 | – | – | – | – | – | – | – | – | 153 | 121 | 63.5 | -0.040 | +0.055 | **DECIDED no** — LIVE N<10 — no evidence to move the block |

### 3d. Live periods separately

| period | cell | N | days | WR % | avg %/fill | same-days ctl N | ctl avg | Δ |
|---|---|---|---|---|---|---|---|---|
| Jul 06–26 (phantoms) | ALL matched | 209 | 19 | 43.5 | -0.257 | 14 | +0.310 | -0.567 |
| Jul 06–26 (phantoms) | C-only (no W) | 76 | 17 | 51.3 | -0.193 | 12 | +0.333 | -0.526 |
| Jul 06–26 (phantoms) | W-only (C-free) | 87 | 18 | 37.9 | -0.252 | 14 | +0.310 | -0.562 |
| Jul 06–26 (phantoms) | C+W | 46 | 13 | 41.3 | -0.372 | 11 | +0.206 | -0.578 |
| Jul 06–26 (phantoms) | W2 C-free | 31 | 9 | 48.4 | -0.078 | 6 | +0.686 | -0.764 |
| Sep 20–24 (journal) | ALL matched | 33 | 5 | 69.7 | +0.053 | 11 | +0.005 | +0.048 |
| Sep 20–24 (journal) | C-only (no W) | 8 | 3 | 50.0 | -0.321 | 9 | -0.015 | -0.306 |
| Sep 20–24 (journal) | W-only (C-free) | 20 | 5 | 75.0 | +0.144 | 11 | +0.005 | +0.139 |
| Sep 20–24 (journal) | C+W | 5 | 2 | 80.0 | +0.287 | 3 | +0.823 | -0.536 |
| Sep 20–24 (journal) | W2 C-free | 19 | 5 | 73.7 | +0.147 | 11 | +0.005 | +0.142 |
| Oct 12–26 (CloudWatch) | ALL matched | 76 | 12 | 52.6 | -0.214 | 33 | -0.134 | -0.080 |
| Oct 12–26 (CloudWatch) | C-only (no W) | 13 | 7 | 46.2 | -0.397 | 18 | +0.050 | -0.447 |
| Oct 12–26 (CloudWatch) | W-only (C-free) | 57 | 11 | 56.1 | -0.165 | 30 | -0.071 | -0.094 |
| Oct 12–26 (CloudWatch) | C+W | 6 | 4 | 33.3 | -0.281 | 19 | -0.196 | -0.085 |
| Oct 12–26 (CloudWatch) | W2 C-free | 32 | 10 | 59.4 | -0.129 | 30 | -0.071 | -0.058 |

### 3e. Matched momentum longs that really traded (MREAL — doors / bypasses, non-probe, master)

49 fills · as traded -0.078 %/fill · today's replica -0.119 · WR 53 % — C6 10·-0.03, C6+W4 7·+0.04, W2 14·-0.33, W2+W4 15·-0.05. Same direction as the shadows (descriptive; door-selected).

## 4. Verdicts

170 cells: **DECIDED no** 170. Full list with reasons: `reports/ML_MATCHED_SHADOW_2026-10-26_verdicts.csv`.

- **SMALL-SIZE LIVE PROBE: none.** No cell clears the bar on live data; the replay refutes every positive live slice. (Had one cleared: 0.05 lev = 1× at 0.2 invest, revert on N ≥ 10 with WR < 55 % or Σ < 0, read at N 30.)
- **DECIDED no:** every family (C-only, C-free W, C+W, all matched) and every single label — negative live, below the same-days control, in all three live periods.
- **OPERATOR CALL:** none.
- **Frozen observe candidates (if the recorder is built; pre-registered now, never re-fit):** ① `W2 C-free` (any regime) · ② `W2 C-free × BTC regime STRONG_BULL`. Gate per cell, forward fills only: N ≥ 30 on ≥ 10 days · WR ≥ 61.5 % · avg ≥ +0.10 %/fill · day-bootstrap LB > 0 at α = 0.025 (Bonferroni over 2) · ≥ same-days unmatched control · no day / pair ≥ 50 %. Pass → OPERATOR CALL on a 1× probe; fail → DECIDED no, recorder off.

## 5. Could NOT test

- **Server journal 09-25 → 10-26** (ctx snapshots for every refusal) — the EB log bundle has been broken since 10-19; CloudWatch carries only the C/W family, so October specific labels are kline-rebuilt (approximate).
- **Aug 1 → Sep 19 live refusals** — no phantom seeding (retired), journal not yet shipped (Sep-18) / not local, CloudWatch retention 14 days. A 7-week hole.
- **Slot competition / portfolio effects** — a matched long would take one of the 4 slots and can displace an unmatched fill; shadows ignore slots (the replay journal only lists refusals with room = true).
- **LONG_CHOP_BURST** (needs the live fill book) not applied to shadows; **LONG_HEAT_BLOCK** fails open on CloudWatch rows (no breadth reading) and the mega-cap gate uses the daily top-50 rank as a proxy outside the July stamps.
- **Entry order type / slippage** — every shadow priced at a taker entry at the logged / journalled price (CloudWatch: last closed 1m close, ≤ 60 s stale); no queue or maker model.
- **Sizing cells** — matched cells would trade at their pattern-cell size (all LONG cells 1× today); % is leverage-invariant, so $ effects are not modelled.
- **July phantoms ran under the July ENTRY stack**: gates removed since then (e.g. FAKE_BULL_GUARD) could have refused rows that today's stack would see; gates added since were re-applied (blacklist, heat, mega-cap, LOADX).
- **Funding / order-book state** at refusal time — not recorded on any surface.
- **Backtest absolute level** — the replay ML sleeve itself loses (unmatched control −0.16 %/fill vs +0.11 live), so the replay is read only relative to its own control and only to refute.

## Files

- `reports/ML_MATCHED_SHADOW_2026-10-26_cloudwatch_refusals.csv`
- `reports/ML_MATCHED_SHADOW_2026-10-26_replay_blocks.csv.gz`
- `reports/ML_MATCHED_SHADOW_2026-10-26_shadows.csv.gz`
- `reports/ML_MATCHED_SHADOW_2026-10-26_controls.csv.gz`
- `reports/ML_MATCHED_SHADOW_2026-10-26_episodes.csv`
- `reports/ML_MATCHED_SHADOW_2026-10-26_regime.csv`
- `reports/ML_MATCHED_SHADOW_2026-10-26_parity.csv`
- `reports/ML_MATCHED_SHADOW_2026-10-26_cells.csv`
- `reports/ML_MATCHED_SHADOW_2026-10-26_verdicts.csv`
- price-path top-up: `reports/ML_MATCHED_SHADOW_2026-10-26_fetch/` (public fapi klines, 300 calls, ≤ 400 weight/min)
