# Tick study: resting take-profit, maker entry and profit lock for WILLY / LITE (2026-10-15)

Research only. No engine, config, template or test was changed, and nothing was committed. Cache-only: no Binance call was made.
This study was pre-registered (reports/TICK_STUDY_2026-10-15_PREREG.txt, frozen 12:15:53Z before any variant was priced). It has not
been reviewed, so it supports no arm or ship.

## Plain-English summary

**Verdict: no candidate under the pre-registered gates.** 55 variants were judged (WILLY 30, LITE 25). Not one passes "better with
P ≥ 0.90 in both 2025 and 2026, master not negative, stable month by month, not concentrated". By luck alone about 0.5 false passes
were expected.

1. **Resting take-profit (a limit order waiting at the TP).** Priced on real trade prints, it is a wash.
   - **WILLY:** Δ between −0.004 and +0.014 %/fill at every latency, in both years.
   - **LITE:** positive in 2026 (+0.03 to +0.04, P 0.94–1.00) but slightly negative in 2025 (−0.005 to −0.015).
   - **Why it is a wash on prints:** the price a market order gets 0.8–2.6 s after the touch is almost the touch price (−0.01 on
     average). The limit also books exactly +1.25 / +3, while the market TP keeps the "jump-over" prints (the replay's WILLY TP averages +1.33).
   - **The missed fills are not the danger the design feared.** The limit fills on the same share of trades as today's TP (WILLY 74 %
     in 2026). Of the 57 WILLY trades where the market TP fired first, 56 still filled at the limit moments later and 1 stopped out.
   - **Where the money really is:** live take-profit fills gave back 0.13–0.22 % (thin order books), and trade prints cannot see that.
     If that live give-back is real (it is 3 fills), the resting TP wins +0.07 to +0.14 %/fill on WILLY and +0.04 to +0.14 on LITE, in both years (§G).
   - **Only the live shadow can measure that give-back.** That shadow is design Phase 2. Building the observe-only shadow is the
     justified next step; arming is not.
2. **Maker entry (a limit buy instead of a market buy).** On paper-style costs (0.036 % entry slip) it does not pass.
   - **The selection risk is real.** The limit fills on 40–69 % of signals. The signals it misses are the winners that ran away:
     WILLY's unfilled signals average +0.10…+0.20 under today's entry vs −0.03…−0.12 for the filled ones. LITE "skip" variants lose in 2026 (−0.04 to −0.12).
   - **The headline warning:** with the live-measured entry cost (+0.27 % average vs the decision price, 5 live fills), today's WILLY
     taker entry is negative per signal in BOTH years: 2025 −0.23, 2026 −0.18. LITE is −0.42 in 2025 and +0.01 in 2026.
   - **At that cost, every WILLY maker variant wins by +0.08 to +0.31 per signal.** That sensitivity was pre-registered as "not a second
     chance", so it is NOT a candidate. It does say that the live entry slip is the single most important number to measure next.
3. **Profit lock (move the stop up after a gain).** No.
   - **WILLY:** every one of the 9 cells is negative in 2025 (−0.010 to −0.047) and none is significant in 2026 (−0.041 to +0.018).
   - **Saves vs kills:** each lock does save DUSK-type trades (+3 to +4 points each). It cuts 1.3–4× as many winners that dip back and
     still reach +1.25. For example A +0.8 / L 0 in 2026 saves 49 fills (+155 points) and kills 118 (−145 points).
   - **LITE:** the lock is clearly worse in 2026 (−0.05 to −0.14, P 0.02 for A +1.5). Its 2025 gain lives in the 1m-only (provisional)
     fills.
4. **Cross-check passed (the STOP rule did not trigger).** The walker reproduces the three existing WILLY 30-min Scout cells exactly.
   - **2025 (899 fills):** close at 30 −0.0075, +0.25 after 30 −0.0156 (P 0.20, top-50 −0.0315), +0.30 after 30 −0.0198. These are the
     DL 291 refs.
   - **2026 (the same 606 fills as DL 285):** −0.0127 (P 0.36), +0.0184 (P 0.79), +0.0218 (P 0.83).
   - **On all 617 fills 2026 differs** only because 11 fills that DL 285 left unpriced now have 1m data.

**Recommended next step (operator decides):**
- Build only the Phase-2 observe-only shadow. On every live WILLY / LITE TP it logs what a resting limit would have done, and on every
  entry the book slip vs the decision price.
- Re-read at ≥ 15 live TPs and ≥ 15 live entries.
- Nothing else changes.

## Parity (every headline computed two independent ways)

| check | result |
|---|---|
| Q1 arm A at latency 0 vs `scout_frenzy_willy_exit.evaluate` on the same tick path | **1,750 / 1,750 identical** |
| Q3 base walker (lat 0) vs `evaluate`, every path | **100 %** identical (price and exit kind) |
| Walker vs the replay's recorded `pct_new` (same exit kind ∧ \|Δ\| < 0.10) | WILLY 2025 92.7 % (tick paths **100 %**), 2026 88.7 % (ticks 89.0 %); LITE 2025 99.4 % (ticks 100 %), 2026 90.1 % |
| Independent loop re-implementation (`study_tick_maker_lock_verify.py`, 20 random + 10 targeted fills, raw npz files, no shared walker code) | A / B / B+ at 3 latencies, lock base + 2 cells, Q2 today + maker: **every check 30 / 30 (lock cells 12 / 12, 18 / 18) identical** |
| Scout cells, two evaluators (this walker vs `study_willy_late_sl.ev`) | 100 % identical per fill; DL 291 / DL 285 refs reproduced to the 3rd decimal |

The 2026 replay-vs-walker gap (~11 %) is the known 1 Hz monitor vs first-print difference (the WILLY_LATE_SL study saw 89 %). To keep
it from biasing Q3, every lock cell was also gated on the parity-matched subset (§F, "Q3-PAR" rows).

## Deviations from the pre-registration (stated, none touches a variant or a gate)

- **1m loader fix, made after the freeze and before any variant was read.** `WT.load_1m` reads neither `k1m_spike` nor the 2025 cache;
  the replay harness reads both. Without the fix, every 2025 Jan–Mar fill was UNRES. Now 0 fills are unpriced on Q3.
- **Partial tick coverage accepted for Q1 / Q2** when every arm's exit resolves inside the covered prints (an arm that would need
  uncovered time is UNRES). This affects 13 fills: 12 LITE 2025, 1 LITE 2026-H2 and the live DEEP fill.
- **LOMO per year** is in TICK_STUDY_MAKER_LOCK_gates.csv (`lomo_2025_sign_stable` / `lomo_2026_sign_stable`). The table shows pooled
  LOMO.

## Could NOT be tested

- **Order-book slippage** (the live TP give-back of 0.13–0.22 and the live entry cost of +0.27): aggTrades carry trade prints, not the
  book. This is the dominant term for Q1 and Q2, so it is shown only as sensitivity (§D ruler TL, §G).
- **Queue position:** fills require a print ≥ 1 tick THROUGH the limit; B_touch / B+ worst bracket it. Partial fills are not modelled.
- **Historic tick sizes:** today's `exchange_ticks.json` is used. 294 tick-priced fills sat on a coarser grid, and the inferred grid was
  used for those (stricter for the limit).
- **2025 non-top-50 fills (669 WILLY, 564 LITE) have no ticks.** They are excluded from Q1 / Q2, so 2025 Q1 / Q2 = top-50 only.
  Q3 2025 includes them on 1m bars (provisional: the lock books the line). The ticks-only Δ is shown beside every Q3 row.
- **Live block:** the 4 WILLY fills of 2026-10-15 have no tick or 1m data in the cache (Binance archive not published yet, not fetched),
  so they are n/a for Q1 / Q2 / Q3.
  - The recorded peaks still bound Q3. DUSK (peak +1.15 → −3.43) is saved by any A ≤ +1.0. STRK (peak +0.77 → −3.79) is saved only by
    A +0.6.
  - BLUR (trough −2.07, peak +1.32) and ZK (trough −1.31, peak +1.25) would be KILLED if they touched A before their dip. The export
    does not record the order.
  - LIVE LITE DEEP: Q1 / Q2 priced, Q3 Δ 0.
- **Master WILLY is 19 fills; their recorded close reasons are the OLD exit.** `stack_pct` is today's re-walk, so B19 kind-parity reads
  86 %. Master LITE B18 has 2 RUNNER_TRAIL rows of the same kind.
- **Slot effects of earlier or later exits** (a maker TP frees the slot at a slightly different time) are not modelled.

## Exact commands

```
venv/bin/python scripts/study_tick_maker_lock_prereg.py        # freeze cohort + hashes + gates (refuses to overwrite)
venv/bin/python scripts/study_tick_maker_lock_walk.py --jobs 8 # → reports/TICK_STUDY_MAKER_LOCK_rows.csv (asserts the frozen cohort hash)
venv/bin/python scripts/study_tick_maker_lock_report.py        # → reports/TICK_STUDY_MAKER_LOCK_tables.md + _gates.csv
venv/bin/python scripts/study_tick_maker_lock_verify.py        # → reports/TICK_STUDY_MAKER_LOCK_verify.txt (independent re-implementation)
```
Shared code: scripts/study_tick_maker_lock_common.py (walkers, loader patches, bootstrap / LOMO / concentration).

---
# Tables (generated — reports/TICK_STUDY_MAKER_LOCK_tables.md)

Three blocks in every table: ① live · ② master per batch · ③ backtest per year (2025 split top-50 / not; 2026-H1, 2026-H2, 2026).
Δ = variant − today on the same fill, %/fill net (Q2: per signal). P = day-clustered bootstrap P(mean Δ > 0), 4,000 draws, seed 7.

## A. Cohort, coverage and parity

| block | sleeve | era | fills | Q3 path ticks / ticks+1m / 1m | Q1–Q2 tick-priced (A & B at L 1.5) | replay/recorded parity (walker base, lat 0) |
|---|---|---|---|---|---|---|
| BT | LITE | 2025 | 828 | 252 / 12 / 564 | 263 | 99 % |
| BT | LITE | 2026-H1 | 370 | 370 / 0 / 0 | 369 | 90 % |
| BT | LITE | 2026-H2 | 256 | 255 / 1 / 0 | 255 | 90 % |
| BT | WILLY | 2025 | 899 | 230 / 0 / 669 | 230 | 93 % |
| BT | WILLY | 2026-H1 | 411 | 391 / 0 / 20 | 391 | 87 % |
| BT | WILLY | 2026-H2 | 208 | 202 / 0 / 6 | 202 | 92 % |
| LIVE | LITE | LIVE | 1 | 0 / 1 / 0 | 1 | 0 % |
| LIVE | WILLY | LIVE | 4 | 0 / 0 / 4 | 0 | 0 % |
| MASTER | LITE | B18 | 2 | 2 / 0 / 0 | 2 | 0 % |
| MASTER | LITE | B19 | 13 | 13 / 0 / 0 | 13 | 100 % |
| MASTER | LITE | B20 | 1 | 1 / 0 / 0 | 1 | 100 % |
| MASTER | LITE | B21 | 4 | 4 / 0 / 0 | 4 | 100 % |
| MASTER | WILLY | B19 | 14 | 14 / 0 / 0 | 14 | 86 % |
| MASTER | WILLY | B20 | 3 | 3 / 0 / 0 | 3 | 100 % |
| MASTER | WILLY | B21 | 2 | 2 / 0 / 0 | 2 | 100 % |

Parity checks (rule 4):
- Q1 arm A at L = 0 vs FX.evaluate on the same tick path: 1750 / 1750 identical.
- Q3 base walker (lat 0) vs FX.evaluate on every path: 100.0 % identical (why identical 100.0 %).
- WILLY 2025: walker vs the replay's recorded pct_new (same exit kind ∧ |Δ| < 0.10): 92.7 % of 899 (tick paths 100.0 % of 230); mean walker − recorded -0.068 %/fill.
- WILLY 2026: walker vs the replay's recorded pct_new (same exit kind ∧ |Δ| < 0.10): 88.7 % of 619 (tick paths 89.0 % of 593); mean walker − recorded +0.002 %/fill.
- LITE 2025: walker vs the replay's recorded pct_new (same exit kind ∧ |Δ| < 0.10): 99.4 % of 828 (tick paths 100.0 % of 252); mean walker − recorded -0.014 %/fill.
- LITE 2026: walker vs the replay's recorded pct_new (same exit kind ∧ |Δ| < 0.10): 90.1 % of 626 (tick paths 90.1 % of 625); mean walker − recorded +0.050 %/fill.
- Tick size source on tick-priced fills: {'exchange': 1453, 'coarser': 294, 'inferred': 4}.

## B. Walker cross-check — the three WILLY 30-min Scout cells (zero-latency rules, as DL 285 / DL 291)

| cohort | cell | N | Δ this walker | Δ study_willy_late_sl.ev | per-fill identical | P(Δ>0) | 2025 top-50 Δ | reference |
|---|---|---|---|---|---|---|---|---|
| 2025 | T125C30 | 899 | -0.0075 | -0.0075 | 100.0 % | 0.38 | -0.0511 | −0.008 (DL 291) |
| 2025 | LT30_025 | 899 | -0.0156 | -0.0156 | 100.0 % | 0.20 | -0.0315 | −0.016, P 0.20, top-50 −0.031 (DL 291) |
| 2025 | LT30_03 | 899 | -0.0198 | -0.0198 | 100.0 % | 0.13 | -0.0362 | −0.020, P 0.14 (DL 291) |
| 2026 | T125C30 | 617 | -0.0011 | -0.0011 | 100.0 % | 0.50 | – | −0.013, P 0.35 (606 fills, DL 285) |
| 2026 | LT30_025 | 617 | +0.0249 | +0.0249 | 100.0 % | 0.85 | – | +0.018, P 0.79 (606) |
| 2026 | LT30_03 | 617 | +0.0284 | +0.0284 | 100.0 % | 0.88 | – | +0.022, P 0.83 (606) |
| 2026 (the DL 285 606 fills) | T125C30 | 606 | -0.0127 | -0.0127 | 100.0 % | 0.36 | – | −0.013, P 0.35 (606 fills, DL 285) |
| 2026 (the DL 285 606 fills) | LT30_025 | 606 | +0.0184 | +0.0184 | 100.0 % | 0.79 | – | +0.018, P 0.79 (606) |
| 2026 (the DL 285 606 fills) | LT30_03 | 606 | +0.0218 | +0.0218 | 100.0 % | 0.83 | – | +0.022, P 0.83 (606) |

The 2026 full-cohort numbers differ from the DL 285 refs only through 11 fills the DL 285 walk left UNRES (1m tail missing — this study also reads the k1m_spike caches the replay used); on the same 606 fills every ref reproduces to the 3rd decimal.

## C. Q1 — resting take-profit (maker exit) vs today's market TP


### WILLY (TP +1.25 / stop −4.0 / 90 min)


**WILLY · L = 0.8 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 0.8 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.412 | +0.437 | **+0.025** | +0.35 | [-0.016, +0.124] | 0.71 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 3 | 67 % → 67 % | -0.637 | -0.532 | **+0.104** | +0.31 | [–, –] | – | 67 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 2 | 100 % → 100 % | +1.389 | +1.252 | **-0.137** | -0.27 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 19 | 79 % → 79 % | +0.349 | +0.369 | **+0.020** | +0.39 | [-0.052, +0.089] | 0.72 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.006 | +0.020 | **+0.014** | +3.22 | [-0.008, +0.036] | 0.89 | 70 % | 17 (17 / 0 / 0) | +2.70 | 0 (+0.00) |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.006 | +0.020 | **+0.014** | +3.22 | [-0.008, +0.036] | 0.89 | 70 % | 17 (17 / 0 / 0) | +2.70 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 74 % | +0.116 | +0.105 | **-0.011** | -4.19 | [-0.046, +0.017] | 0.27 | 74 % | 43 (42 / 1 / 0) | -3.01 | 0 (+0.00) |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.055 | +0.076 | **+0.021** | +4.16 | [-0.016, +0.055] | 0.87 | 73 % | 14 (14 / 0 / 0) | +1.58 | 1 (+0.99) |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.095 | +0.095 | **-0.000** | -0.04 | [-0.025, +0.022] | 0.51 | 74 % | 57 (56 / 1 / 0) | -1.43 | 1 (+0.99) |

**WILLY · L = 0.8 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.412 | +0.437 | **+0.025** | +0.35 | [-0.016, +0.124] | 0.71 |
| ② master B20 | 3 | 67 % → 67 % | -0.637 | -0.532 | **+0.104** | +0.31 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.389 | +1.252 | **-0.137** | -0.27 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.349 | +0.369 | **+0.020** | +0.39 | [-0.052, +0.089] | 0.72 |
| ③ BT 2025 | 230 | 72 % → 73 % | +0.006 | +0.045 | **+0.038** | +8.78 | [-0.003, +0.099] | 0.96 |
| ③ BT 2025 top-50 | 230 | 72 % → 73 % | +0.006 | +0.045 | **+0.038** | +8.78 | [-0.003, +0.099] | 0.96 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 75 % | +0.116 | +0.129 | **+0.013** | +5.27 | [-0.010, +0.038] | 0.86 |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.055 | +0.101 | **+0.046** | +9.26 | [-0.008, +0.111] | 0.95 |
| ③ BT 2026 | 593 | 74 % → 75 % | +0.095 | +0.120 | **+0.025** | +14.54 | [+0.000, +0.052] | 0.98 |

**WILLY · L = 0.8 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.412 | +0.346 | **-0.066** | -0.92 | [-0.247, +0.029] | 0.26 |
| ② master B20 | 3 | 67 % → 67 % | -0.637 | -0.418 | **+0.218** | +0.65 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.389 | +1.264 | **-0.124** | -0.25 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.349 | +0.322 | **-0.027** | -0.51 | [-0.187, +0.123] | 0.36 |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.006 | +0.020 | **+0.014** | +3.12 | [-0.015, +0.045] | 0.82 |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.006 | +0.020 | **+0.014** | +3.12 | [-0.015, +0.045] | 0.82 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 73 % | +0.116 | +0.105 | **-0.011** | -4.21 | [-0.060, +0.032] | 0.34 |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.055 | +0.046 | **-0.009** | -1.90 | [-0.034, +0.015] | 0.22 |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.095 | +0.085 | **-0.010** | -6.11 | [-0.043, +0.019] | 0.26 |

**WILLY · L = 1.5 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 1.5 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.440 | +0.442 | **+0.001** | +0.02 | [-0.047, +0.118] | 0.60 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 3 | 67 % → 67 % | -0.567 | -0.573 | **-0.006** | -0.02 | [–, –] | – | 67 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 2 | 100 % → 100 % | +1.270 | +1.252 | **-0.018** | -0.04 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 19 | 79 % → 79 % | +0.369 | +0.367 | **-0.002** | -0.03 | [-0.101, +0.110] | 0.48 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.018 | +0.021 | **+0.003** | +0.62 | [-0.022, +0.027] | 0.59 | 70 % | 17 (17 / 0 / 0) | +1.83 | 0 (+0.00) |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.018 | +0.021 | **+0.003** | +0.62 | [-0.022, +0.027] | 0.59 | 70 % | 17 (17 / 0 / 0) | +1.83 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 74 % | +0.115 | +0.108 | **-0.008** | -2.95 | [-0.047, +0.027] | 0.36 | 74 % | 43 (42 / 1 / 0) | -3.78 | 0 (+0.00) |
| ③ BT 2026-H2 | 202 | 73 % → 74 % | +0.057 | +0.076 | **+0.020** | +4.01 | [-0.019, +0.056] | 0.84 | 73 % | 14 (14 / 0 / 0) | +1.34 | 1 (+0.95) |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.095 | +0.097 | **+0.002** | +1.06 | [-0.027, +0.028] | 0.57 | 74 % | 57 (56 / 1 / 0) | -2.44 | 1 (+0.95) |

**WILLY · L = 1.5 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.440 | +0.442 | **+0.001** | +0.02 | [-0.047, +0.118] | 0.60 |
| ② master B20 | 3 | 67 % → 67 % | -0.567 | -0.573 | **-0.006** | -0.02 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.270 | +1.252 | **-0.018** | -0.04 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.369 | +0.367 | **-0.002** | -0.03 | [-0.101, +0.110] | 0.48 |
| ③ BT 2025 | 230 | 72 % → 73 % | +0.018 | +0.045 | **+0.027** | +6.24 | [-0.016, +0.088] | 0.85 |
| ③ BT 2025 top-50 | 230 | 72 % → 73 % | +0.018 | +0.045 | **+0.027** | +6.24 | [-0.016, +0.088] | 0.85 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 75 % | +0.115 | +0.132 | **+0.017** | +6.55 | [-0.015, +0.048] | 0.87 |
| ③ BT 2026-H2 | 202 | 73 % → 74 % | +0.057 | +0.101 | **+0.045** | +9.03 | [-0.010, +0.113] | 0.93 |
| ③ BT 2026 | 593 | 74 % → 75 % | +0.095 | +0.122 | **+0.026** | +15.58 | [-0.001, +0.058] | 0.97 |

**WILLY · L = 1.5 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.440 | +0.361 | **-0.079** | -1.11 | [-0.272, +0.035] | 0.15 |
| ② master B20 | 3 | 67 % → 67 % | -0.567 | -0.429 | **+0.138** | +0.41 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.270 | +1.489 | **+0.219** | +0.44 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.369 | +0.355 | **-0.013** | -0.26 | [-0.144, +0.145] | 0.46 |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.018 | +0.015 | **-0.003** | -0.58 | [-0.036, +0.034] | 0.43 |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.018 | +0.015 | **-0.003** | -0.58 | [-0.036, +0.034] | 0.43 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 73 % | +0.115 | +0.094 | **-0.021** | -8.35 | [-0.066, +0.017] | 0.15 |
| ③ BT 2026-H2 | 202 | 73 % → 74 % | +0.057 | +0.057 | **+0.000** | +0.06 | [-0.029, +0.032] | 0.49 |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.095 | +0.081 | **-0.014** | -8.29 | [-0.045, +0.013] | 0.17 |

**WILLY · L = 2.6 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 2.6 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.390 | +0.456 | **+0.066** | +0.92 | [+0.025, +0.106] | 1.00 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 3 | 67 % → 67 % | -0.516 | -0.553 | **-0.037** | -0.11 | [–, –] | – | 67 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 2 | 100 % → 100 % | +1.248 | +1.252 | **+0.004** | +0.01 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 19 | 79 % → 79 % | +0.338 | +0.380 | **+0.043** | +0.82 | [-0.122, +0.198] | 0.75 | 79 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.016 | +0.013 | **-0.004** | -0.82 | [-0.034, +0.025] | 0.42 | 70 % | 17 (17 / 0 / 0) | +0.76 | 0 (+0.00) |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.016 | +0.013 | **-0.004** | -0.82 | [-0.034, +0.025] | 0.42 | 70 % | 17 (17 / 0 / 0) | +0.76 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 74 % | +0.114 | +0.108 | **-0.006** | -2.25 | [-0.052, +0.034] | 0.42 | 74 % | 43 (42 / 1 / 0) | -4.40 | 0 (+0.00) |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.034 | +0.076 | **+0.041** | +8.34 | [+0.002, +0.077] | 0.98 | 73 % | 14 (14 / 0 / 0) | +2.39 | 1 (+0.94) |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.087 | +0.097 | **+0.010** | +6.09 | [-0.022, +0.040] | 0.76 | 74 % | 57 (56 / 1 / 0) | -2.01 | 1 (+0.94) |

**WILLY · L = 2.6 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.390 | +0.456 | **+0.066** | +0.92 | [+0.025, +0.106] | 1.00 |
| ② master B20 | 3 | 67 % → 67 % | -0.516 | -0.553 | **-0.037** | -0.11 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.248 | +1.252 | **+0.004** | +0.01 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.338 | +0.380 | **+0.043** | +0.82 | [-0.122, +0.198] | 0.75 |
| ③ BT 2025 | 230 | 72 % → 73 % | +0.016 | +0.037 | **+0.020** | +4.69 | [-0.028, +0.085] | 0.75 |
| ③ BT 2025 top-50 | 230 | 72 % → 73 % | +0.016 | +0.037 | **+0.020** | +4.69 | [-0.028, +0.085] | 0.75 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 75 % | +0.114 | +0.133 | **+0.019** | +7.39 | [-0.019, +0.056] | 0.84 |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.034 | +0.099 | **+0.064** | +12.97 | [+0.010, +0.128] | 0.99 |
| ③ BT 2026 | 593 | 74 % → 75 % | +0.087 | +0.121 | **+0.034** | +20.36 | [+0.003, +0.067] | 0.99 |

**WILLY · L = 2.6 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.390 | +0.391 | **+0.001** | +0.01 | [-0.062, +0.031] | 0.63 |
| ② master B20 | 3 | 67 % → 67 % | -0.516 | -0.565 | **-0.049** | -0.15 | [–, –] | – |
| ② master B21 | 2 | 100 % → 100 % | +1.248 | +1.647 | **+0.399** | +0.80 | [–, –] | – |
| ② master ALL | 19 | 79 % → 79 % | +0.338 | +0.372 | **+0.035** | +0.66 | [-0.035, +0.179] | 0.80 |
| ③ BT 2025 | 230 | 72 % → 72 % | +0.016 | -0.003 | **-0.019** | -4.34 | [-0.052, +0.013] | 0.13 |
| ③ BT 2025 top-50 | 230 | 72 % → 72 % | +0.016 | -0.003 | **-0.019** | -4.34 | [-0.052, +0.013] | 0.13 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 391 | 74 % → 73 % | +0.114 | +0.084 | **-0.030** | -11.79 | [-0.074, +0.006] | 0.05 |
| ③ BT 2026-H2 | 202 | 74 % → 74 % | +0.034 | +0.057 | **+0.023** | +4.58 | [-0.004, +0.051] | 0.95 |
| ③ BT 2026 | 593 | 74 % → 74 % | +0.087 | +0.075 | **-0.012** | -7.21 | [-0.041, +0.014] | 0.19 |

Modelled market-TP give-back on A-TP fills (fill print − trigger print, net %), BT: L 0.8: -0.012 (median -0.014) · L 1.5: -0.010 (median +0.000) · L 2.6: -0.016 (median -0.023). Live TP fills: −0.13 / −0.22 / −0.22 vs the trigger (book slippage included — not modelled here).
Modelled stop give-back (fill − trigger) on A-stop fills, L 0.8 / 1.5 / 2.6: +0.051 / +0.061 / +0.050 (live stops +0.21 / +0.57 better than the trigger).
Limit price vs entry (median): +1.3198 % (A's trigger sits at +1.3406 %).

### LITE (TP +3.0 / stop −3.0 / 720 min)


**LITE · L = 0.8 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 0.8 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.896 | +3.006 | **+0.110** | +0.11 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B18 | 2 | 100 % → 100 % | +2.604 | +3.013 | **+0.409** | +0.82 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B19 | 13 | 54 % → 54 % | +0.084 | +0.140 | **+0.056** | +0.73 | [-0.024, +0.102] | 0.74 | 54 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 1 | 0 % → 0 % | -3.015 | -3.015 | **+0.000** | +0.00 | [–, –] | – | 0 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 4 | 50 % → 50 % | +0.042 | +0.085 | **+0.044** | +0.17 | [–, –] | – | 50 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 20 | 55 % → 55 % | +0.173 | +0.259 | **+0.086** | +1.72 | [-0.003, +0.176] | 0.97 | 55 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.033 | -0.038 | **-0.005** | -1.25 | [-0.069, +0.037] | 0.48 | 49 % | 19 (18 / 1 / 0) | -4.72 | 0 (+0.00) |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.033 | -0.038 | **-0.005** | -1.25 | [-0.069, +0.037] | 0.48 | 49 % | 19 (18 / 1 / 0) | -4.72 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.278 | +0.331 | **+0.053** | +19.45 | [+0.005, +0.110] | 0.99 | 54 % | 30 (30 / 0 / 0) | +0.48 | 2 (+11.97) |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.222 | +0.240 | **+0.018** | +4.47 | [-0.009, +0.045] | 0.89 | 53 % | 9 (9 / 0 / 0) | -0.14 | 0 (+0.00) |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.255 | +0.294 | **+0.038** | +23.92 | [+0.008, +0.073] | 1.00 | 54 % | 39 (39 / 0 / 0) | +0.34 | 2 (+11.97) |

**LITE · L = 0.8 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.896 | +3.006 | **+0.110** | +0.11 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.604 | +3.013 | **+0.409** | +0.82 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.084 | +0.140 | **+0.056** | +0.73 | [-0.024, +0.102] | 0.74 |
| ② master B20 | 1 | 0 % → 0 % | -3.015 | -3.015 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | +0.042 | +0.085 | **+0.044** | +0.17 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.173 | +0.259 | **+0.086** | +1.72 | [-0.003, +0.176] | 0.97 |
| ③ BT 2025 | 263 | 49 % → 50 % | -0.033 | +0.008 | **+0.041** | +10.84 | [+0.001, +0.098] | 0.98 |
| ③ BT 2025 top-50 | 263 | 49 % → 50 % | -0.033 | +0.008 | **+0.041** | +10.84 | [+0.001, +0.098] | 0.98 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.278 | +0.331 | **+0.053** | +19.45 | [+0.005, +0.110] | 0.99 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.222 | +0.240 | **+0.018** | +4.47 | [-0.009, +0.045] | 0.89 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.255 | +0.294 | **+0.038** | +23.92 | [+0.008, +0.073] | 1.00 |

**LITE · L = 0.8 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.896 | +2.937 | **+0.041** | +0.04 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.604 | +2.856 | **+0.252** | +0.50 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.084 | +0.084 | **-0.001** | -0.01 | [-0.047, +0.090] | 0.49 |
| ② master B20 | 1 | 0 % → 0 % | -3.015 | -3.015 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | +0.042 | +0.041 | **-0.001** | -0.00 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.173 | +0.197 | **+0.024** | +0.49 | [-0.045, +0.128] | 0.73 |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.033 | -0.073 | **-0.040** | -10.56 | [-0.115, +0.016] | 0.11 |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.033 | -0.073 | **-0.040** | -10.56 | [-0.115, +0.016] | 0.11 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.278 | +0.327 | **+0.048** | +17.88 | [+0.003, +0.106] | 0.98 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.222 | +0.247 | **+0.025** | +6.42 | [-0.006, +0.058] | 0.95 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.255 | +0.294 | **+0.039** | +24.29 | [+0.008, +0.074] | 0.99 |

**LITE · L = 1.5 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 1.5 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +3.006 | **+0.069** | +0.07 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B18 | 2 | 100 % → 100 % | +2.524 | +3.013 | **+0.489** | +0.98 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B19 | 13 | 54 % → 54 % | +0.063 | +0.110 | **+0.047** | +0.61 | [-0.117, +0.131] | 0.71 | 54 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 1 | 0 % → 0 % | -2.866 | -2.866 | **+0.000** | +0.00 | [–, –] | – | 0 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 4 | 50 % → 50 % | -0.014 | +0.085 | **+0.100** | +0.40 | [–, –] | – | 50 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 20 | 55 % → 55 % | +0.147 | +0.247 | **+0.099** | +1.99 | [-0.026, +0.223] | 0.94 | 55 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.028 | -0.042 | **-0.015** | -3.85 | [-0.079, +0.033] | 0.32 | 49 % | 19 (18 / 1 / 0) | -5.07 | 0 (+0.00) |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.028 | -0.042 | **-0.015** | -3.85 | [-0.079, +0.033] | 0.32 | 49 % | 19 (18 / 1 / 0) | -5.07 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.272 | +0.329 | **+0.057** | +20.89 | [+0.009, +0.115] | 0.99 | 54 % | 30 (30 / 0 / 0) | +1.74 | 2 (+12.08) |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.244 | +0.231 | **-0.013** | -3.27 | [-0.051, +0.024] | 0.25 | 53 % | 9 (9 / 0 / 0) | -0.70 | 0 (+0.00) |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.260 | +0.289 | **+0.028** | +17.63 | [-0.005, +0.065] | 0.94 | 54 % | 39 (39 / 0 / 0) | +1.05 | 2 (+12.08) |

**LITE · L = 1.5 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +3.006 | **+0.069** | +0.07 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.524 | +3.013 | **+0.489** | +0.98 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.063 | +0.110 | **+0.047** | +0.61 | [-0.117, +0.131] | 0.71 |
| ② master B20 | 1 | 0 % → 0 % | -2.866 | -2.866 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | -0.014 | +0.085 | **+0.100** | +0.40 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.147 | +0.247 | **+0.099** | +1.99 | [-0.026, +0.223] | 0.94 |
| ③ BT 2025 | 263 | 49 % → 50 % | -0.028 | +0.004 | **+0.032** | +8.42 | [-0.015, +0.095] | 0.89 |
| ③ BT 2025 top-50 | 263 | 49 % → 50 % | -0.028 | +0.004 | **+0.032** | +8.42 | [-0.015, +0.095] | 0.89 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.272 | +0.329 | **+0.057** | +20.89 | [+0.009, +0.115] | 0.99 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.244 | +0.231 | **-0.013** | -3.27 | [-0.051, +0.024] | 0.25 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.260 | +0.289 | **+0.028** | +17.63 | [-0.005, +0.065] | 0.94 |

**LITE · L = 1.5 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +2.937 | **+0.000** | +0.00 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.524 | +3.378 | **+0.854** | +1.71 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.063 | +0.036 | **-0.027** | -0.35 | [-0.069, +0.113] | 0.33 |
| ② master B20 | 1 | 0 % → 0 % | -2.866 | -2.866 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | -0.014 | +0.061 | **+0.075** | +0.30 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.147 | +0.230 | **+0.083** | +1.66 | [-0.047, +0.356] | 0.82 |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.028 | -0.075 | **-0.047** | -12.44 | [-0.120, +0.007] | 0.06 |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.028 | -0.075 | **-0.047** | -12.44 | [-0.120, +0.007] | 0.06 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.272 | +0.322 | **+0.051** | +18.65 | [+0.006, +0.108] | 0.99 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.244 | +0.238 | **-0.006** | -1.44 | [-0.040, +0.029] | 0.38 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.260 | +0.288 | **+0.028** | +17.21 | [-0.004, +0.064] | 0.95 |

**LITE · L = 2.6 s · B (limit, fills on a print ≥ 1 tick through) − A (market TP after 2.6 s)**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | B fill rate | A-TP missed by B (later fill / stop / cap) | missed-fill cost Σ(B−A) | B-only extra fills (Σ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +3.006 | **+0.069** | +0.07 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B18 | 2 | 100 % → 100 % | +2.651 | +3.013 | **+0.362** | +0.72 | [–, –] | – | 100 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B19 | 13 | 54 % → 54 % | +0.095 | +0.153 | **+0.058** | +0.75 | [-0.169, +0.168] | 0.67 | 54 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B20 | 1 | 0 % → 0 % | -2.810 | -2.810 | **+0.000** | +0.00 | [–, –] | – | 0 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master B21 | 4 | 50 % → 50 % | -0.016 | +0.064 | **+0.080** | +0.32 | [–, –] | – | 50 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ② master ALL | 20 | 55 % → 55 % | +0.183 | +0.273 | **+0.090** | +1.79 | [-0.068, +0.204] | 0.87 | 55 % | 0 (0 / 0 / 0) | +0.00 | 0 (+0.00) |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.020 | -0.034 | **-0.014** | -3.80 | [-0.080, +0.032] | 0.33 | 49 % | 19 (18 / 1 / 0) | -4.55 | 0 (+0.00) |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.020 | -0.034 | **-0.014** | -3.80 | [-0.080, +0.032] | 0.33 | 49 % | 19 (18 / 1 / 0) | -4.55 | 0 (+0.00) |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.259 | +0.330 | **+0.070** | +26.01 | [+0.017, +0.134] | 1.00 | 54 % | 30 (30 / 0 / 0) | +2.58 | 2 (+11.90) |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.231 | +0.223 | **-0.008** | -2.13 | [-0.045, +0.028] | 0.33 | 53 % | 9 (9 / 0 / 0) | -0.91 | 0 (+0.00) |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.248 | +0.286 | **+0.038** | +23.88 | [+0.003, +0.077] | 0.98 | 54 % | 39 (39 / 0 / 0) | +1.67 | 2 (+11.90) |

**LITE · L = 2.6 s · B_touch (optimistic queue: fills on a touch) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +3.006 | **+0.069** | +0.07 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.651 | +3.013 | **+0.362** | +0.72 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.095 | +0.153 | **+0.058** | +0.75 | [-0.169, +0.168] | 0.67 |
| ② master B20 | 1 | 0 % → 0 % | -2.810 | -2.810 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | -0.016 | +0.064 | **+0.080** | +0.32 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.183 | +0.273 | **+0.090** | +1.79 | [-0.068, +0.204] | 0.87 |
| ③ BT 2025 | 263 | 49 % → 50 % | -0.020 | +0.013 | **+0.033** | +8.60 | [-0.014, +0.093] | 0.90 |
| ③ BT 2025 top-50 | 263 | 49 % → 50 % | -0.020 | +0.013 | **+0.033** | +8.60 | [-0.014, +0.093] | 0.90 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.259 | +0.330 | **+0.070** | +26.01 | [+0.017, +0.134] | 1.00 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.231 | +0.223 | **-0.008** | -2.13 | [-0.045, +0.028] | 0.33 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.248 | +0.286 | **+0.038** | +23.88 | [+0.003, +0.077] | 0.98 |

**LITE · L = 2.6 s · B+ worst (limit never fills; safety net 2 ticks + 3 s → market) − A**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +2.937 | +2.937 | **+0.000** | +0.00 | [–, –] | – |
| ② master B18 | 2 | 100 % → 100 % | +2.651 | +3.201 | **+0.550** | +1.10 | [–, –] | – |
| ② master B19 | 13 | 54 % → 54 % | +0.095 | +0.129 | **+0.034** | +0.44 | [-0.014, +0.184] | 0.86 |
| ② master B20 | 1 | 0 % → 0 % | -2.810 | -2.810 | **+0.000** | +0.00 | [–, –] | – |
| ② master B21 | 4 | 50 % → 50 % | -0.016 | +0.033 | **+0.050** | +0.20 | [–, –] | – |
| ② master ALL | 20 | 55 % → 55 % | +0.183 | +0.270 | **+0.087** | +1.74 | [+0.005, +0.252] | 0.99 |
| ③ BT 2025 | 263 | 49 % → 49 % | -0.020 | -0.061 | **-0.041** | -10.76 | [-0.112, +0.013] | 0.09 |
| ③ BT 2025 top-50 | 263 | 49 % → 49 % | -0.020 | -0.061 | **-0.041** | -10.76 | [-0.112, +0.013] | 0.09 |
| ③ BT 2025 not top-50 | 0 | – | – | – | – | – | – | – |
| ③ BT 2026-H1 | 369 | 54 % → 55 % | +0.259 | +0.323 | **+0.064** | +23.71 | [+0.021, +0.120] | 1.00 |
| ③ BT 2026-H2 | 255 | 54 % → 54 % | +0.231 | +0.230 | **-0.002** | -0.39 | [-0.036, +0.033] | 0.45 |
| ③ BT 2026 | 624 | 54 % → 54 % | +0.248 | +0.285 | **+0.037** | +23.31 | [+0.007, +0.072] | 0.99 |

Modelled market-TP give-back on A-TP fills (fill print − trigger print, net %), BT: L 0.8: -0.037 (median -0.032) · L 1.5: -0.017 (median -0.033) · L 2.6: -0.032 (median -0.033). Live TP fills: −0.13 / −0.22 / −0.22 vs the trigger (book slippage included — not modelled here).
Modelled stop give-back (fill − trigger) on A-stop fills, L 0.8 / 1.5 / 2.6: +0.062 / +0.051 / +0.053 (live stops +0.21 / +0.57 better than the trigger).
Limit price vs entry (median): +3.0703 % (A's trigger sits at +3.0914 %).

## D. Q2 — maker entry (per SIGNAL; a skipped / refused signal counts 0)

Rulers: T0 = 0 · **T1 = 0.036 % (primary)** · TL = live mean entry slip vs decision (+0.267 %, N 5) − backtest mean drift decision → +8 s (-0.006 %) = +0.273 % → priced at the nearest grid point 0.25 %.
Backtest drift P0 → print at +8 s: mean -0.006 %, median +0.000 %, p90 +0.300 %.

### WILLY


**Ruler T1 (taker slip 0.036 %)** — today (taker at +8 s): per signal avg by block: ② master B19 +0.440 (N 14, refused 0) · ② master B20 -0.916 (N 3, refused 1) · ② master B21 +1.254 (N 2, refused 0) · ② master ALL +0.311 (N 19, refused 1) · ③ BT 2025 -0.072 (N 230, refused 1) · ③ BT 2025 top-50 -0.072 (N 230, refused 1) · ③ BT 2026-H1 +0.098 (N 391, refused 8) · ③ BT 2026-H2 +0.031 (N 202, refused 1) · ③ BT 2026 +0.075 (N 593, refused 9)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 50 % | -0.031 | +0.098 | – | +0.294 [1.00] n19 | +0.061 [0.70] n230 | -0.024 [0.37] n391 | -0.001 [0.49] n202 | -0.016 [0.39] n593 | +40.7 · 71 % |
| limit P0−0% · 5s · taker fallback | 50 % | -0.031 | +0.098 | – | +0.066 [0.82] n19 | +0.063 [0.94] n230 | +0.005 [0.57] n391 | -0.004 [0.46] n202 | +0.002 [0.54] n593 | – |
| limit P0−0% · 10s · skip | 59 % | -0.043 | +0.143 | – | +0.362 [1.00] n19 | +0.064 [0.76] n230 | -0.004 [0.47] n391 | +0.023 [0.59] n202 | +0.005 [0.52] n593 | +48.6 · 72 % |
| limit P0−0% · 10s · taker fallback | 59 % | -0.043 | +0.143 | – | +0.041 [0.64] n19 | +0.061 [0.89] n230 | -0.006 [0.44] n391 | -0.002 [0.51] n202 | -0.005 [0.45] n593 | – |
| limit P0−0% · 20s · skip | 69 % | -0.041 | +0.197 | – | +0.152 [0.86] n19 | +0.049 [0.70] n230 | +0.014 [0.58] n391 | +0.066 [0.77] n202 | +0.032 [0.72] n593 | +51.0 · 72 % |
| limit P0−0% · 20s · taker fallback | 69 % | -0.041 | +0.197 | – | +0.066 [0.82] n19 | +0.040 [0.82] n230 | +0.012 [0.58] n391 | -0.009 [0.47] n202 | +0.005 [0.54] n593 | – |
| limit P0−0.1% · 5s · skip | 40 % | -0.121 | +0.138 | – | +0.438 [0.96] n19 | +0.124 [0.85] n230 | -0.075 [0.17] n391 | -0.091 [0.21] n202 | -0.080 [0.11] n593 | +67.8 · 72 % |
| limit P0−0.1% · 5s · taker fallback | 40 % | -0.121 | +0.138 | – | +0.065 [0.82] n19 | +0.083 [0.97] n230 | +0.029 [0.76] n391 | -0.006 [0.45] n202 | +0.017 [0.70] n593 | – |
| limit P0−0.1% · 10s · skip | 49 % | -0.073 | +0.136 | – | +0.296 [1.00] n19 | +0.144 [0.91] n230 | -0.012 [0.43] n391 | -0.083 [0.21] n202 | -0.036 [0.27] n593 | +57.2 · 71 % |
| limit P0−0.1% · 10s · taker fallback | 49 % | -0.073 | +0.136 | – | +0.041 [0.63] n19 | +0.080 [0.93] n230 | -0.009 [0.44] n391 | -0.005 [0.50] n202 | -0.007 [0.44] n593 | – |
| limit P0−0.1% · 20s · skip | 60 % | -0.065 | +0.180 | – | +0.087 [0.82] n19 | +0.154 [0.94] n230 | +0.015 [0.57] n391 | -0.015 [0.44] n202 | +0.005 [0.52] n593 | +59.8 · 71 % |
| limit P0−0.1% · 20s · taker fallback | 60 % | -0.065 | +0.180 | – | +0.067 [0.82] n19 | +0.101 [0.97] n230 | +0.011 [0.57] n391 | -0.063 [0.25] n202 | -0.014 [0.38] n593 | – |

**Ruler T0 (taker slip 0.0 %)** — today (taker at +8 s): per signal avg by block: ② master B19 +0.450 (N 14, refused 0) · ② master B20 -0.925 (N 3, refused 1) · ② master B21 +1.253 (N 2, refused 0) · ② master ALL +0.317 (N 19, refused 1) · ③ BT 2025 -0.059 (N 230, refused 1) · ③ BT 2025 top-50 -0.059 (N 230, refused 1) · ③ BT 2026-H1 +0.122 (N 391, refused 8) · ③ BT 2026-H2 +0.088 (N 202, refused 1) · ③ BT 2026 +0.111 (N 593, refused 9)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 50 % | +0.001 | +0.125 | – | +0.289 [1.00] n19 | +0.048 [0.67] n230 | -0.048 [0.25] n391 | -0.058 [0.28] n202 | -0.052 [0.18] n593 | +51.8 · 72 % |
| limit P0−0% · 5s · taker fallback | 50 % | +0.001 | +0.125 | – | +0.065 [0.80] n19 | +0.053 [0.91] n230 | +0.014 [0.71] n391 | -0.059 [0.15] n202 | -0.010 [0.36] n593 | – |
| limit P0−0% · 10s · skip | 59 % | -0.011 | +0.170 | – | +0.356 [1.00] n19 | +0.051 [0.71] n230 | -0.029 [0.33] n391 | -0.034 [0.33] n202 | -0.031 [0.27] n593 | +57.5 · 72 % |
| limit P0−0% · 10s · taker fallback | 59 % | -0.011 | +0.170 | – | +0.038 [0.63] n19 | +0.051 [0.85] n230 | -0.010 [0.41] n391 | -0.057 [0.25] n202 | -0.026 [0.26] n593 | – |
| limit P0−0% · 20s · skip | 69 % | -0.004 | +0.210 | – | +0.147 [0.86] n19 | +0.036 [0.65] n230 | -0.011 [0.42] n391 | +0.009 [0.53] n202 | -0.004 [0.45] n593 | +54.3 · 72 % |
| limit P0−0% · 20s · taker fallback | 69 % | -0.004 | +0.210 | – | +0.064 [0.78] n19 | +0.029 [0.74] n230 | +0.002 [0.51] n391 | -0.064 [0.22] n202 | -0.020 [0.31] n593 | – |
| limit P0−0.1% · 5s · skip | 40 % | -0.082 | +0.160 | – | +0.432 [0.96] n19 | +0.111 [0.83] n230 | -0.100 [0.11] n391 | -0.148 [0.08] n202 | -0.116 [0.03] n593 | +79.1 · 72 % |
| limit P0−0.1% · 5s · taker fallback | 40 % | -0.082 | +0.160 | – | +0.064 [0.78] n19 | +0.081 [0.97] n230 | +0.039 [0.87] n391 | -0.060 [0.14] n202 | +0.005 [0.58] n593 | – |
| limit P0−0.1% · 10s · skip | 49 % | -0.036 | +0.158 | – | +0.291 [1.00] n19 | +0.131 [0.89] n230 | -0.036 [0.31] n391 | -0.140 [0.07] n202 | -0.072 [0.12] n593 | +66.5 · 72 % |
| limit P0−0.1% · 10s · taker fallback | 49 % | -0.036 | +0.158 | – | +0.039 [0.63] n19 | +0.071 [0.90] n230 | -0.011 [0.42] n391 | -0.060 [0.24] n202 | -0.028 [0.27] n593 | – |
| limit P0−0.1% · 20s · skip | 60 % | -0.023 | +0.191 | – | +0.081 [0.81] n19 | +0.141 [0.92] n230 | -0.009 [0.42] n391 | -0.072 [0.21] n202 | -0.031 [0.27] n593 | +63.5 · 71 % |
| limit P0−0.1% · 20s · taker fallback | 60 % | -0.023 | +0.191 | – | +0.065 [0.79] n19 | +0.092 [0.96] n230 | +0.002 [0.51] n391 | -0.092 [0.14] n202 | -0.030 [0.26] n593 | – |

**Ruler TL (taker slip 0.25 %)** — today (taker at +8 s): per signal avg by block: ② master B19 +0.299 (N 14, refused 0) · ② master B20 -0.922 (N 3, refused 1) · ② master B21 +1.254 (N 2, refused 0) · ② master ALL +0.207 (N 19, refused 1) · ③ BT 2025 -0.231 (N 230, refused 1) · ③ BT 2025 top-50 -0.231 (N 230, refused 1) · ③ BT 2026-H1 -0.154 (N 391, refused 8) · ③ BT 2026-H2 -0.225 (N 202, refused 1) · ③ BT 2026 -0.178 (N 593, refused 9)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 50 % | -0.246 | -0.141 | – | +0.399 [1.00] n19 | +0.220 [0.97] n230 | +0.228 [1.00] n391 | +0.255 [0.98] n202 | +0.237 [1.00] n593 | -58.4 · 66 % |
| limit P0−0% · 5s · taker fallback | 50 % | -0.246 | -0.141 | – | +0.067 [0.82] n19 | +0.122 [0.98] n230 | +0.210 [1.00] n391 | +0.084 [0.91] n202 | +0.167 [1.00] n593 | – |
| limit P0−0% · 10s · skip | 59 % | -0.266 | -0.089 | – | +0.467 [1.00] n19 | +0.224 [0.98] n230 | +0.248 [1.00] n391 | +0.279 [0.99] n202 | +0.258 [1.00] n593 | -30.2 · 67 % |
| limit P0−0% · 10s · taker fallback | 59 % | -0.266 | -0.089 | – | -0.033 [0.43] n19 | +0.093 [0.87] n230 | +0.199 [1.00] n391 | +0.135 [0.96] n202 | +0.177 [1.00] n593 | – |
| limit P0−0% · 20s · skip | 69 % | -0.262 | -0.042 | – | +0.257 [0.88] n19 | +0.208 [0.98] n230 | +0.266 [1.00] n391 | +0.322 [1.00] n202 | +0.285 [1.00] n593 | -10.8 · 67 % |
| limit P0−0% · 20s · taker fallback | 69 % | -0.262 | -0.042 | – | +0.149 [1.00] n19 | +0.120 [0.94] n230 | +0.199 [1.00] n391 | +0.163 [0.97] n202 | +0.187 [1.00] n593 | – |
| limit P0−0.1% · 5s · skip | 40 % | -0.338 | -0.096 | – | +0.543 [0.98] n19 | +0.283 [0.99] n230 | +0.177 [0.98] n391 | +0.165 [0.92] n202 | +0.173 [0.99] n593 | -47.5 · 67 % |
| limit P0−0.1% · 5s · taker fallback | 40 % | -0.338 | -0.096 | – | +0.065 [0.81] n19 | +0.109 [0.97] n230 | +0.172 [1.00] n391 | +0.055 [0.81] n202 | +0.132 [1.00] n593 | – |
| limit P0−0.1% · 10s · skip | 49 % | -0.316 | -0.075 | – | +0.401 [1.00] n19 | +0.303 [1.00] n230 | +0.240 [1.00] n391 | +0.173 [0.93] n202 | +0.217 [1.00] n593 | -31.7 · 67 % |
| limit P0−0.1% · 10s · taker fallback | 49 % | -0.316 | -0.075 | – | -0.033 [0.43] n19 | +0.077 [0.82] n230 | +0.178 [1.00] n391 | +0.045 [0.73] n202 | +0.133 [0.99] n593 | – |
| limit P0−0.1% · 20s · skip | 60 % | -0.299 | -0.037 | – | +0.192 [0.87] n19 | +0.313 [1.00] n230 | +0.267 [1.00] n391 | +0.241 [0.98] n202 | +0.258 [1.00] n593 | -12.2 · 67 % |
| limit P0−0.1% · 20s · taker fallback | 60 % | -0.299 | -0.037 | – | +0.150 [1.00] n19 | +0.169 [0.98] n230 | +0.192 [1.00] n391 | +0.057 [0.70] n202 | +0.146 [1.00] n593 | – |

### LITE


**Ruler T1 (taker slip 0.036 %)** — today (taker at +8 s): per signal avg by block: ① LIVE +3.035 (N 1, refused 0) · ② master B18 +3.005 (N 2, refused 0) · ② master B19 +0.230 (N 13, refused 0) · ② master B20 -3.001 (N 1, refused 0) · ② master B21 -0.002 (N 4, refused 0) · ② master ALL +0.299 (N 20, refused 0) · ③ BT 2025 -0.094 (N 263, refused 0) · ③ BT 2025 top-50 -0.094 (N 263, refused 0) · ③ BT 2026-H1 +0.240 (N 369, refused 1) · ③ BT 2026-H2 +0.107 (N 255, refused 0) · ③ BT 2026 +0.186 (N 624, refused 1)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 39 % | -0.063 | +0.211 | -3.035 [–] n1 | +0.164 [0.82] n20 | -0.012 [0.46] n263 | -0.213 [0.03] n369 | +0.010 [0.53] n255 | -0.122 [0.10] n624 | +113.3 · 53 % |
| limit P0−0% · 5s · taker fallback | 39 % | -0.063 | +0.211 | -0.003 [–] n1 | +0.013 [1.00] n20 | -0.036 [0.30] n263 | +0.005 [0.57] n369 | +0.034 [0.81] n255 | +0.017 [0.76] n624 | – |
| limit P0−0% · 10s · skip | 49 % | -0.041 | +0.244 | -3.035 [–] n1 | -0.133 [0.39] n20 | +0.070 [0.69] n263 | -0.215 [0.02] n369 | +0.049 [0.64] n255 | -0.107 [0.11] n624 | +109.2 · 53 % |
| limit P0−0% · 10s · taker fallback | 49 % | -0.041 | +0.244 | +0.000 [–] n1 | +0.019 [1.00] n20 | +0.012 [0.61] n263 | +0.014 [0.64] n369 | +0.049 [0.84] n255 | +0.029 [0.82] n624 | – |
| limit P0−0% · 20s · skip | 60 % | +0.049 | +0.183 | -3.035 [–] n1 | -0.279 [0.10] n20 | +0.095 [0.78] n263 | -0.175 [0.03] n369 | +0.147 [0.86] n255 | -0.044 [0.29] n624 | +65.1 · 53 % |
| limit P0−0% · 20s · taker fallback | 60 % | +0.049 | +0.183 | +0.004 [–] n1 | -0.129 [0.34] n20 | -0.042 [0.27] n263 | -0.000 [0.50] n369 | +0.076 [0.91] n255 | +0.031 [0.81] n624 | – |
| limit P0−0.1% · 5s · skip | 30 % | +0.030 | +0.134 | -3.035 [–] n1 | +0.164 [0.82] n20 | +0.065 [0.64] n263 | -0.232 [0.04] n369 | +0.067 [0.65] n255 | -0.110 [0.14] n624 | +83.2 · 52 % |
| limit P0−0.1% · 5s · taker fallback | 30 % | +0.030 | +0.134 | -0.003 [–] n1 | +0.013 [1.00] n20 | -0.016 [0.46] n263 | +0.002 [0.55] n369 | +0.032 [0.81] n255 | +0.014 [0.69] n624 | – |
| limit P0−0.1% · 10s · skip | 38 % | +0.005 | +0.162 | -3.035 [–] n1 | +0.019 [0.65] n20 | +0.147 [0.83] n263 | -0.231 [0.03] n369 | +0.057 [0.64] n255 | -0.113 [0.13] n624 | +89.7 · 52 % |
| limit P0−0.1% · 10s · taker fallback | 38 % | +0.005 | +0.162 | +0.000 [–] n1 | +0.020 [1.00] n20 | +0.010 [0.59] n263 | +0.010 [0.60] n369 | +0.023 [0.65] n255 | +0.015 [0.68] n624 | – |
| limit P0−0.1% · 20s · skip | 47 % | +0.032 | +0.166 | -3.035 [–] n1 | +0.170 [0.82] n20 | +0.104 [0.76] n263 | -0.196 [0.04] n369 | +0.131 [0.82] n255 | -0.063 [0.25] n624 | +77.9 · 52 % |
| limit P0−0.1% · 20s · taker fallback | 47 % | +0.032 | +0.166 | +0.004 [–] n1 | -0.130 [0.34] n20 | -0.034 [0.34] n263 | -0.004 [0.48] n369 | +0.049 [0.80] n255 | +0.018 [0.67] n624 | – |

**Ruler T0 (taker slip 0.0 %)** — today (taker at +8 s): per signal avg by block: ① LIVE +3.031 (N 1, refused 0) · ② master B18 +3.015 (N 2, refused 0) · ② master B19 +0.230 (N 13, refused 0) · ② master B20 -3.003 (N 1, refused 0) · ② master B21 +0.001 (N 4, refused 0) · ② master ALL +0.301 (N 20, refused 0) · ③ BT 2025 -0.094 (N 263, refused 0) · ③ BT 2025 top-50 -0.094 (N 263, refused 0) · ③ BT 2026-H1 +0.290 (N 369, refused 1) · ③ BT 2026-H2 +0.107 (N 255, refused 0) · ③ BT 2026 +0.215 (N 624, refused 1)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 39 % | -0.061 | +0.244 | -3.031 [–] n1 | +0.162 [0.82] n20 | -0.012 [0.46] n263 | -0.263 [0.01] n369 | +0.010 [0.52] n255 | -0.151 [0.06] n624 | +131.2 · 53 % |
| limit P0−0% · 5s · taker fallback | 39 % | -0.061 | +0.244 | -0.003 [–] n1 | +0.011 [0.99] n20 | -0.014 [0.41] n263 | -0.031 [0.22] n369 | +0.082 [1.00] n255 | +0.015 [0.73] n624 | – |
| limit P0−0% · 10s · skip | 49 % | -0.025 | +0.270 | -3.031 [–] n1 | -0.135 [0.39] n20 | +0.070 [0.69] n263 | -0.265 [0.01] n369 | +0.049 [0.64] n255 | -0.137 [0.06] n624 | +120.9 · 54 % |
| limit P0−0% · 10s · taker fallback | 49 % | -0.025 | +0.270 | +0.000 [–] n1 | +0.016 [1.00] n20 | +0.057 [0.91] n263 | -0.020 [0.30] n369 | +0.049 [0.84] n255 | +0.008 [0.61] n624 | – |
| limit P0−0% · 20s · skip | 60 % | +0.061 | +0.217 | -3.031 [–] n1 | -0.281 [0.10] n20 | +0.095 [0.78] n263 | -0.225 [0.01] n369 | +0.146 [0.86] n255 | -0.073 [0.18] n624 | +77.1 · 53 % |
| limit P0−0% · 20s · taker fallback | 60 % | +0.061 | +0.217 | +0.004 [–] n1 | -0.131 [0.34] n20 | -0.042 [0.27] n263 | -0.050 [0.15] n369 | +0.075 [0.91] n255 | +0.001 [0.53] n624 | – |
| limit P0−0.1% · 5s · skip | 30 % | +0.032 | +0.163 | -3.031 [–] n1 | +0.163 [0.82] n20 | +0.065 [0.64] n263 | -0.282 [0.02] n369 | +0.067 [0.65] n255 | -0.139 [0.09] n624 | +101.1 · 52 % |
| limit P0−0.1% · 5s · taker fallback | 30 % | +0.032 | +0.163 | -0.003 [–] n1 | +0.011 [0.99] n20 | +0.006 [0.61] n263 | -0.017 [0.33] n369 | +0.080 [1.00] n255 | +0.023 [0.78] n624 | – |
| limit P0−0.1% · 10s · skip | 38 % | +0.006 | +0.194 | -3.031 [–] n1 | +0.017 [0.65] n20 | +0.147 [0.83] n263 | -0.280 [0.02] n369 | +0.057 [0.64] n255 | -0.143 [0.08] n624 | +107.6 · 53 % |
| limit P0−0.1% · 10s · taker fallback | 38 % | +0.006 | +0.194 | +0.000 [–] n1 | +0.017 [1.00] n20 | +0.055 [0.85] n263 | -0.024 [0.31] n369 | +0.023 [0.65] n255 | -0.005 [0.47] n624 | – |
| limit P0−0.1% · 20s · skip | 47 % | +0.034 | +0.204 | -3.031 [–] n1 | +0.169 [0.82] n20 | +0.104 [0.76] n263 | -0.246 [0.02] n369 | +0.130 [0.81] n255 | -0.092 [0.16] n624 | +95.7 · 53 % |
| limit P0−0.1% · 20s · taker fallback | 47 % | +0.034 | +0.204 | +0.004 [–] n1 | -0.132 [0.34] n20 | -0.034 [0.34] n263 | -0.054 [0.16] n369 | +0.049 [0.80] n255 | -0.012 [0.39] n624 | – |

**Ruler TL (taker slip 0.25 %)** — today (taker at +8 s): per signal avg by block: ① LIVE +3.020 (N 1, refused 0) · ② master B18 +3.013 (N 2, refused 0) · ② master B19 +0.229 (N 13, refused 0) · ② master B20 -3.004 (N 1, refused 0) · ② master B21 -0.000 (N 4, refused 0) · ② master ALL +0.300 (N 20, refused 0) · ③ BT 2025 -0.422 (N 263, refused 0) · ③ BT 2025 top-50 -0.422 (N 263, refused 0) · ③ BT 2026-H1 +0.042 (N 369, refused 1) · ③ BT 2026-H2 -0.037 (N 255, refused 0) · ③ BT 2026 +0.010 (N 624, refused 1)

| variant | fill rate (BT) | today avg on FILLED signals | today avg on UNFILLED | ① LIVE Δ/signal [P] | ② master ALL Δ/signal [P] | ③ 2025 Δ/signal [P] | ③ 2026-H1 Δ/signal [P] | ③ 2026-H2 Δ/signal [P] | ③ 2026 Δ/signal [P] | missed cost (skip, BT): today Σ on skipped · TP share |
|---|---|---|---|---|---|---|---|---|---|---|
| limit P0−0% · 5s · skip | 39 % | -0.310 | +0.007 | -3.020 [–] n1 | +0.164 [0.82] n20 | +0.317 [0.98] n263 | -0.015 [0.44] n369 | +0.154 [0.82] n255 | +0.054 [0.71] n624 | +3.5 · 50 % |
| limit P0−0% · 5s · taker fallback | 39 % | -0.310 | +0.007 | -0.003 [–] n1 | +0.012 [1.00] n20 | +0.179 [1.00] n263 | +0.084 [0.99] n369 | +0.083 [0.89] n255 | +0.083 [0.99] n624 | – |
| limit P0−0% · 10s · skip | 49 % | -0.237 | -0.001 | -3.020 [–] n1 | -0.134 [0.39] n20 | +0.398 [1.00] n263 | -0.017 [0.42] n369 | +0.193 [0.90] n255 | +0.069 [0.78] n624 | -0.6 · 49 % |
| limit P0−0% · 10s · taker fallback | 49 % | -0.237 | -0.001 | +0.000 [–] n1 | +0.017 [1.00] n20 | +0.204 [1.00] n263 | +0.128 [1.00] n369 | +0.146 [0.98] n255 | +0.135 [1.00] n624 | – |
| limit P0−0% · 20s · skip | 60 % | -0.113 | -0.126 | -3.020 [–] n1 | -0.280 [0.10] n20 | +0.424 [1.00] n263 | +0.023 [0.58] n369 | +0.291 [0.98] n255 | +0.132 [0.94] n624 | -44.6 · 47 % |
| limit P0−0% · 20s · taker fallback | 60 % | -0.113 | -0.126 | +0.004 [–] n1 | -0.129 [0.34] n20 | +0.172 [0.99] n263 | +0.065 [0.87] n369 | +0.195 [1.00] n255 | +0.118 [0.99] n624 | – |
| limit P0−0.1% · 5s · skip | 30 % | -0.180 | -0.091 | -3.020 [–] n1 | +0.164 [0.82] n20 | +0.394 [0.99] n263 | -0.034 [0.40] n369 | +0.211 [0.88] n255 | +0.066 [0.73] n624 | -56.7 · 48 % |
| limit P0−0.1% · 5s · taker fallback | 30 % | -0.180 | -0.091 | -0.003 [–] n1 | +0.013 [1.00] n20 | +0.153 [0.99] n263 | +0.033 [0.87] n369 | +0.057 [0.82] n255 | +0.042 [0.91] n624 | – |
| limit P0−0.1% · 10s · skip | 38 % | -0.199 | -0.069 | -3.020 [–] n1 | +0.018 [0.65] n20 | +0.476 [1.00] n263 | -0.033 [0.39] n369 | +0.201 [0.89] n255 | +0.063 [0.73] n624 | -38.5 · 48 % |
| limit P0−0.1% · 10s · taker fallback | 38 % | -0.199 | -0.069 | +0.000 [–] n1 | +0.019 [1.00] n20 | +0.202 [1.00] n263 | +0.091 [1.00] n369 | +0.118 [0.94] n255 | +0.102 [1.00] n624 | – |
| limit P0−0.1% · 20s · skip | 47 % | -0.131 | -0.107 | -3.020 [–] n1 | +0.170 [0.82] n20 | +0.433 [1.00] n263 | +0.001 [0.50] n369 | +0.274 [0.97] n255 | +0.113 [0.88] n624 | -50.3 · 48 % |
| limit P0−0.1% · 20s · taker fallback | 47 % | -0.131 | -0.107 | +0.004 [–] n1 | -0.130 [0.34] n20 | +0.182 [0.99] n263 | +0.027 [0.71] n369 | +0.167 [0.99] n255 | +0.085 [0.97] n624 | – |

## E. Q3 — profit lock (arm at A, stop moves to L; lock exit = L-touch + 1.5 s at the prevailing print)

Base = today's exit with the same execution (stop −sl at touch + 1.5 s; TP first print; cap close). 1m-bar segments book the line (provisional). Fills the lock never arms before the base exit carry Δ = 0 (asserted in the walk).

### WILLY


**WILLY · lock A = +0.6 → L = -1**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.447 | +0.592 | **+0.145** | +2.03 | [+0.000, +0.507] | 0.71 | 2 · +2.03 | 0 · +0.00 | 0 (+0.0) / 0 | +0.060 [0.71] | +0.145 | +0.114 |
| ② master B20 | 3 | 67 % → 33 % | -0.569 | -1.387 | **-0.818** | -2.45 | [–, –] | – | 0 · +0.00 | 1 · -2.45 | 0 (+0.0) / 0 | -0.818 [–] | -0.818 | -0.763 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +1.253 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master ALL | 19 | 79 % → 74 % | +0.372 | +0.349 | **-0.023** | -0.43 | [-0.409, +0.289] | 0.37 | 2 · +2.03 | 1 · -2.45 | 0 (+0.0) / 0 | -0.102 [0.26] | -0.023 | -0.036 |
| ③ BT 2025 | 899 | 69 % → 59 % | -0.303 | -0.350 | **-0.047** | -41.99 | [-0.110, +0.016] | 0.07 | 58 · +154.07 | 82 · -185.23 | 46 (+136.7) / 0 | -0.053 [0.06] | -0.148 | -0.045 |
| ③ BT 2025 top-50 | 230 | 72 % → 60 % | +0.021 | -0.127 | **-0.148** | -34.05 | [-0.270, -0.025] | 0.01 | 14 · +32.03 | 26 · -59.03 | 10 (+28.7) / 0 | -0.148 [0.01] | -0.148 | -0.140 |
| ③ BT 2025 not top-50 | 669 | 67 % → 59 % | -0.415 | -0.427 | **-0.012** | -7.94 | [-0.086, +0.062] | 0.37 | 44 · +122.04 | 56 · -126.19 | 36 (+108.0) / 0 | -0.017 [0.35] | – | -0.012 |
| ③ BT 2026-H1 | 410 | 75 % → 65 % | +0.125 | +0.113 | **-0.012** | -5.01 | [-0.116, +0.088] | 0.42 | 36 · +93.60 | 42 · -95.36 | 29 (+84.1) / 0 | -0.019 [0.37] | -0.017 | -0.005 |
| ③ BT 2026-H2 | 207 | 73 % → 57 % | +0.059 | -0.040 | **-0.099** | -20.42 | [-0.278, +0.081] | 0.15 | 22 · +53.81 | 32 · -71.26 | 16 (+45.5) / 0 | -0.093 [0.17] | -0.090 | -0.081 |
| ③ BT 2026 | 617 | 74 % → 62 % | +0.103 | +0.062 | **-0.041** | -25.43 | [-0.136, +0.048] | 0.19 | 58 · +147.41 | 74 · -166.61 | 45 (+129.6) / 0 | -0.045 [0.20] | -0.042 | -0.030 |

**WILLY · lock A = +0.6 → L = +0**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 71 % | +0.447 | +0.468 | **+0.021** | +0.29 | [-0.209, +0.386] | 0.60 | 2 · +3.75 | 3 · -3.46 | 0 (+0.0) / 0 | -0.168 [0.00] | +0.021 | -0.014 |
| ② master B20 | 3 | 67 % → 33 % | -0.569 | -1.068 | **-0.499** | -1.50 | [–, –] | – | 0 · +0.00 | 1 · -1.50 | 0 (+0.0) / 0 | -0.499 [–] | -0.499 | -0.430 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +0.638 | **-0.615** | -1.23 | [–, –] | – | 0 · +0.00 | 1 · -1.23 | 0 (+0.0) / 0 | -0.615 [–] | -0.615 | -0.630 |
| ② master ALL | 19 | 79 % → 68 % | +0.372 | +0.243 | **-0.128** | -2.43 | [-0.464, +0.157] | 0.20 | 2 · +3.75 | 5 · -6.19 | 0 (+0.0) / 0 | -0.279 [0.00] | -0.128 | -0.144 |
| ③ BT 2025 | 899 | 69 % → 52 % | -0.303 | -0.343 | **-0.040** | -35.63 | [-0.111, +0.032] | 0.13 | 71 · +215.02 | 187 · -246.58 | 46 (+178.4) / 0 | -0.050 [0.09] | -0.162 | -0.040 |
| ③ BT 2025 top-50 | 230 | 72 % → 61 % | +0.021 | -0.141 | **-0.162** | -37.34 | [-0.296, -0.016] | 0.01 | 18 · +48.87 | 66 · -83.22 | 10 (+38.8) / 0 | -0.162 [0.01] | -0.162 | -0.164 |
| ③ BT 2025 not top-50 | 669 | 67 % → 49 % | -0.415 | -0.412 | **+0.003** | +1.71 | [-0.082, +0.089] | 0.52 | 53 · +166.15 | 121 · -163.36 | 36 (+139.6) / 0 | -0.007 [0.43] | – | +0.003 |
| ③ BT 2026-H1 | 410 | 75 % → 67 % | +0.125 | +0.120 | **-0.005** | -2.02 | [-0.121, +0.113] | 0.46 | 44 · +132.73 | 108 · -134.75 | 29 (+112.5) / 0 | +0.015 [0.58] | -0.010 | -0.008 |
| ③ BT 2026-H2 | 207 | 73 % → 61 % | +0.059 | +0.027 | **-0.032** | -6.56 | [-0.215, +0.173] | 0.38 | 25 · +79.28 | 68 · -84.70 | 17 (+64.8) / 0 | -0.045 [0.33] | -0.026 | -0.024 |
| ③ BT 2026 | 617 | 74 % → 65 % | +0.103 | +0.089 | **-0.014** | -8.58 | [-0.114, +0.089] | 0.40 | 69 · +212.01 | 176 · -219.45 | 46 (+177.3) / 0 | -0.006 [0.46] | -0.016 | -0.013 |

**WILLY · lock A = +0.6 → L = +0.3**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 93 % | +0.447 | +0.461 | **+0.014** | +0.19 | [-0.227, +0.522] | 0.71 | 2 · +4.02 | 4 · -3.83 | 0 (+0.0) / 0 | -0.176 [0.00] | +0.014 | +0.024 |
| ② master B20 | 3 | 67 % → 33 % | -0.569 | -1.068 | **-0.499** | -1.50 | [–, –] | – | 0 · +0.00 | 1 · -1.50 | 0 (+0.0) / 0 | -0.499 [–] | -0.499 | -0.333 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +0.869 | **-0.385** | -0.77 | [–, –] | – | 0 · +0.00 | 1 · -0.77 | 0 (+0.0) / 0 | -0.385 [–] | -0.385 | -0.518 |
| ② master ALL | 19 | 79 % → 84 % | +0.372 | +0.262 | **-0.109** | -2.08 | [-0.419, +0.220] | 0.24 | 2 · +4.02 | 6 · -6.10 | 0 (+0.0) / 0 | -0.258 [0.00] | -0.109 | -0.090 |
| ③ BT 2025 | 899 | 69 % → 70 % | -0.303 | -0.336 | **-0.033** | -29.42 | [-0.106, +0.040] | 0.17 | 72 · +229.07 | 236 · -255.60 | 46 (+185.4) / 0 | -0.044 [0.12] | -0.194 | -0.030 |
| ③ BT 2025 top-50 | 230 | 72 % → 78 % | +0.021 | -0.172 | **-0.194** | -44.51 | [-0.331, -0.044] | 0.00 | 18 · +52.72 | 96 · -95.09 | 10 (+40.2) / 0 | -0.194 [0.00] | -0.194 | -0.182 |
| ③ BT 2025 not top-50 | 669 | 67 % → 68 % | -0.415 | -0.392 | **+0.023** | +15.08 | [-0.062, +0.109] | 0.70 | 54 · +176.34 | 140 · -160.52 | 36 (+145.2) / 0 | +0.013 [0.59] | – | +0.023 |
| ③ BT 2026-H1 | 410 | 75 % → 83 % | +0.125 | +0.086 | **-0.039** | -16.06 | [-0.161, +0.085] | 0.27 | 44 · +145.37 | 170 · -162.77 | 29 (+120.1) / 0 | -0.028 [0.33] | -0.046 | -0.054 |
| ③ BT 2026-H2 | 207 | 73 % → 83 % | +0.059 | +0.067 | **+0.008** | +1.67 | [-0.184, +0.225] | 0.52 | 26 · +86.64 | 88 · -84.81 | 17 (+69.5) / 0 | -0.008 [0.47] | +0.013 | +0.015 |
| ③ BT 2026 | 617 | 74 % → 83 % | +0.103 | +0.080 | **-0.023** | -14.40 | [-0.130, +0.085] | 0.34 | 70 · +232.00 | 258 · -247.59 | 46 (+189.7) / 0 | -0.021 [0.37] | -0.026 | -0.031 |

**WILLY · lock A = +0.8 → L = -1**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.447 | +0.540 | **+0.093** | +1.30 | [+0.000, +0.326] | 0.71 | 1 · +1.30 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [0.00] | +0.093 | +0.082 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.569 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +1.253 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master ALL | 19 | 79 % → 79 % | +0.372 | +0.440 | **+0.069** | +1.30 | [+0.000, +0.217] | 0.67 | 1 · +1.30 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [0.00] | +0.069 | +0.060 |
| ③ BT 2025 | 899 | 69 % → 63 % | -0.303 | -0.333 | **-0.030** | -26.62 | [-0.084, +0.023] | 0.13 | 39 · +98.12 | 51 · -115.16 | 28 (+82.4) / 0 | -0.035 [0.11] | -0.089 | -0.027 |
| ③ BT 2025 top-50 | 230 | 72 % → 64 % | +0.021 | -0.068 | **-0.089** | -20.43 | [-0.202, +0.022] | 0.06 | 11 · +22.97 | 16 · -36.36 | 7 (+19.8) / 0 | -0.089 [0.06] | -0.089 | -0.080 |
| ③ BT 2025 not top-50 | 669 | 67 % → 62 % | -0.415 | -0.424 | **-0.009** | -6.19 | [-0.068, +0.052] | 0.37 | 28 · +75.15 | 35 · -78.80 | 21 (+62.6) / 0 | -0.014 [0.34] | – | -0.009 |
| ③ BT 2026-H1 | 410 | 75 % → 68 % | +0.125 | +0.128 | **+0.003** | +1.03 | [-0.091, +0.093] | 0.52 | 28 · +72.66 | 30 · -69.38 | 22 (+63.9) / 0 | -0.009 [0.44] | -0.007 | +0.011 |
| ③ BT 2026-H2 | 207 | 73 % → 61 % | +0.059 | -0.047 | **-0.106** | -21.89 | [-0.253, +0.044] | 0.08 | 13 · +36.22 | 24 · -55.14 | 11 (+31.5) / 0 | -0.094 [0.12] | -0.108 | -0.083 |
| ③ BT 2026 | 617 | 74 % → 65 % | +0.103 | +0.069 | **-0.034** | -20.85 | [-0.114, +0.043] | 0.20 | 41 · +108.88 | 54 · -124.52 | 33 (+95.4) / 0 | -0.039 [0.19] | -0.041 | -0.020 |

**WILLY · lock A = +0.8 → L = +0**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.447 | +0.444 | **-0.003** | -0.05 | [-0.183, +0.263] | 0.38 | 1 · +2.31 | 2 · -2.35 | 0 (+0.0) / 0 | -0.196 [0.00] | -0.003 | -0.026 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.569 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +0.638 | **-0.615** | -1.23 | [–, –] | – | 0 · +0.00 | 1 · -1.23 | 0 (+0.0) / 0 | -0.615 [–] | -0.615 | -0.630 |
| ② master ALL | 19 | 79 % → 79 % | +0.372 | +0.304 | **-0.067** | -1.28 | [-0.327, +0.128] | 0.19 | 1 · +2.31 | 3 · -3.58 | 0 (+0.0) / 0 | -0.211 [0.00] | -0.067 | -0.086 |
| ③ BT 2025 | 899 | 69 % → 58 % | -0.303 | -0.334 | **-0.031** | -27.53 | [-0.090, +0.031] | 0.15 | 48 · +138.64 | 123 · -162.24 | 28 (+108.3) / 0 | -0.040 [0.10] | -0.107 | -0.030 |
| ③ BT 2025 top-50 | 230 | 72 % → 65 % | +0.021 | -0.086 | **-0.107** | -24.56 | [-0.226, +0.020] | 0.05 | 14 · +35.85 | 45 · -57.38 | 7 (+27.1) / 0 | -0.107 [0.05] | -0.107 | -0.104 |
| ③ BT 2025 not top-50 | 669 | 67 % → 55 % | -0.415 | -0.419 | **-0.004** | -2.97 | [-0.074, +0.068] | 0.44 | 34 · +102.79 | 78 · -104.86 | 21 (+81.2) / 0 | -0.015 [0.34] | – | -0.004 |
| ③ BT 2026-H1 | 410 | 75 % → 70 % | +0.125 | +0.151 | **+0.026** | +10.51 | [-0.077, +0.134] | 0.68 | 33 · +101.97 | 74 · -91.45 | 22 (+85.3) / 0 | +0.044 [0.77] | +0.019 | +0.022 |
| ③ BT 2026-H2 | 207 | 73 % → 69 % | +0.059 | +0.051 | **-0.008** | -1.68 | [-0.153, +0.150] | 0.46 | 16 · +52.62 | 44 · -53.26 | 12 (+45.3) / 0 | -0.008 [0.44] | -0.008 | -0.009 |
| ③ BT 2026 | 617 | 74 % → 70 % | +0.103 | +0.117 | **+0.014** | +8.84 | [-0.072, +0.103] | 0.62 | 49 · +154.59 | 118 · -144.72 | 34 (+130.6) / 0 | +0.026 [0.71] | +0.009 | +0.011 |

**WILLY · lock A = +0.8 → L = +0.3**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 86 % | +0.447 | +0.473 | **+0.026** | +0.37 | [-0.159, +0.331] | 0.60 | 1 · +2.31 | 2 · -1.94 | 0 (+0.0) / 0 | -0.162 [0.00] | +0.026 | +0.037 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.846 | **-0.277** | -0.83 | [–, –] | – | 0 · +0.00 | 1 · -0.83 | 0 (+0.0) / 0 | -0.277 [–] | -0.277 | -0.333 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +0.766 | **-0.488** | -0.98 | [–, –] | – | 0 · +0.00 | 1 · -0.98 | 0 (+0.0) / 0 | -0.488 [–] | -0.488 | -0.479 |
| ② master ALL | 19 | 79 % → 84 % | +0.372 | +0.296 | **-0.076** | -1.44 | [-0.325, +0.143] | 0.22 | 1 · +2.31 | 4 · -3.75 | 0 (+0.0) / 0 | -0.220 [0.00] | -0.076 | -0.075 |
| ③ BT 2025 | 899 | 69 % → 70 % | -0.303 | -0.329 | **-0.026** | -22.95 | [-0.087, +0.036] | 0.19 | 48 · +150.33 | 159 · -170.58 | 28 (+114.3) / 0 | -0.036 [0.14] | -0.122 | -0.025 |
| ③ BT 2025 top-50 | 230 | 72 % → 77 % | +0.021 | -0.101 | **-0.122** | -27.95 | [-0.244, +0.011] | 0.04 | 14 · +40.11 | 67 · -65.95 | 7 (+28.9) / 0 | -0.122 [0.04] | -0.122 | -0.119 |
| ③ BT 2025 not top-50 | 669 | 67 % → 68 % | -0.415 | -0.407 | **+0.007** | +5.00 | [-0.065, +0.080] | 0.57 | 34 · +110.23 | 92 · -104.63 | 21 (+85.5) / 0 | -0.003 [0.46] | – | +0.007 |
| ③ BT 2026-H1 | 410 | 75 % → 81 % | +0.125 | +0.146 | **+0.021** | +8.58 | [-0.090, +0.139] | 0.64 | 33 · +111.08 | 110 · -102.51 | 22 (+90.9) / 0 | +0.026 [0.65] | +0.013 | +0.011 |
| ③ BT 2026-H2 | 207 | 73 % → 80 % | +0.059 | +0.070 | **+0.011** | +2.22 | [-0.138, +0.176] | 0.55 | 17 · +56.22 | 56 · -53.85 | 12 (+47.6) / 0 | +0.008 [0.51] | +0.011 | +0.021 |
| ③ BT 2026 | 617 | 74 % → 80 % | +0.103 | +0.120 | **+0.018** | +10.80 | [-0.074, +0.113] | 0.65 | 50 · +167.31 | 166 · -156.36 | 34 (+138.4) / 0 | +0.020 [0.66] | +0.012 | +0.014 |

**WILLY · lock A = +1.0 → L = -1**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.447 | +0.540 | **+0.093** | +1.30 | [+0.000, +0.326] | 0.71 | 1 · +1.30 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [0.00] | +0.093 | +0.082 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.569 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +1.253 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master ALL | 19 | 79 % → 79 % | +0.372 | +0.440 | **+0.069** | +1.30 | [+0.000, +0.217] | 0.67 | 1 · +1.30 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [0.00] | +0.069 | +0.060 |
| ③ BT 2025 | 899 | 69 % → 66 % | -0.303 | -0.315 | **-0.012** | -10.49 | [-0.047, +0.024] | 0.24 | 20 · +49.07 | 25 · -56.05 | 14 (+40.6) / 0 | -0.015 [0.21] | -0.018 | -0.010 |
| ③ BT 2025 top-50 | 230 | 72 % → 68 % | +0.021 | +0.003 | **-0.018** | -4.14 | [-0.107, +0.071] | 0.34 | 8 · +17.97 | 9 · -20.05 | 6 (+16.6) / 0 | -0.018 [0.34] | -0.018 | -0.013 |
| ③ BT 2025 not top-50 | 669 | 67 % → 65 % | -0.415 | -0.424 | **-0.009** | -6.35 | [-0.044, +0.030] | 0.30 | 12 · +31.10 | 16 · -36.00 | 8 (+24.0) / 0 | -0.013 [0.26] | – | -0.009 |
| ③ BT 2026-H1 | 410 | 75 % → 71 % | +0.125 | +0.133 | **+0.008** | +3.16 | [-0.056, +0.072] | 0.59 | 16 · +41.52 | 18 · -38.36 | 12 (+35.2) / 0 | +0.001 [0.51] | +0.000 | +0.002 |
| ③ BT 2026-H2 | 207 | 73 % → 66 % | +0.059 | -0.031 | **-0.090** | -18.66 | [-0.200, +0.018] | 0.05 | 5 · +14.86 | 14 · -32.29 | 4 (+11.6) / 0 | -0.115 [0.02] | -0.092 | -0.072 |
| ③ BT 2026 | 617 | 74 % → 69 % | +0.103 | +0.078 | **-0.025** | -15.49 | [-0.080, +0.030] | 0.19 | 21 · +56.38 | 32 · -70.65 | 16 (+46.8) / 0 | -0.039 [0.10] | -0.031 | -0.023 |

**WILLY · lock A = +1.0 → L = +0**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 79 % | +0.447 | +0.444 | **-0.003** | -0.05 | [-0.183, +0.263] | 0.38 | 1 · +2.31 | 2 · -2.35 | 0 (+0.0) / 0 | -0.196 [0.00] | -0.003 | -0.026 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.569 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +1.253 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master ALL | 19 | 79 % → 79 % | +0.372 | +0.369 | **-0.002** | -0.05 | [-0.127, +0.166] | 0.36 | 1 · +2.31 | 2 · -2.35 | 0 (+0.0) / 0 | -0.138 [0.00] | -0.002 | -0.019 |
| ③ BT 2025 | 899 | 69 % → 64 % | -0.303 | -0.314 | **-0.011** | -9.84 | [-0.052, +0.031] | 0.29 | 22 · +69.43 | 60 · -78.21 | 14 (+54.7) / 0 | -0.016 [0.22] | -0.017 | -0.010 |
| ③ BT 2025 top-50 | 230 | 72 % → 69 % | +0.021 | +0.004 | **-0.017** | -3.88 | [-0.115, +0.091] | 0.36 | 8 · +26.59 | 23 · -29.41 | 6 (+23.2) / 0 | -0.017 [0.36] | -0.017 | -0.013 |
| ③ BT 2025 not top-50 | 669 | 67 % → 62 % | -0.415 | -0.424 | **-0.009** | -5.96 | [-0.052, +0.038] | 0.35 | 14 · +42.84 | 37 · -48.80 | 8 (+31.5) / 0 | -0.016 [0.26] | – | -0.009 |
| ③ BT 2026-H1 | 410 | 75 % → 73 % | +0.125 | +0.136 | **+0.011** | +4.62 | [-0.061, +0.093] | 0.60 | 17 · +58.26 | 44 · -53.64 | 12 (+48.0) / 0 | +0.018 [0.66] | +0.008 | +0.005 |
| ③ BT 2026-H2 | 207 | 73 % → 67 % | +0.059 | +0.006 | **-0.053** | -11.00 | [-0.150, +0.057] | 0.16 | 7 · +21.24 | 25 · -31.66 | 5 (+17.0) / 0 | -0.072 [0.08] | -0.054 | -0.037 |
| ③ BT 2026 | 617 | 74 % → 71 % | +0.103 | +0.093 | **-0.010** | -6.37 | [-0.070, +0.055] | 0.38 | 24 · +79.50 | 69 · -85.30 | 17 (+65.0) / 0 | -0.013 [0.34] | -0.013 | -0.009 |

**WILLY · lock A = +1.0 → L = +0.3**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 0 | – | – | – | – | – | – | – | – | – | – | – | – | – |
| ② master B19 | 14 | 79 % → 86 % | +0.447 | +0.476 | **+0.029** | +0.40 | [-0.154, +0.331] | 0.60 | 1 · +2.31 | 2 · -1.91 | 0 (+0.0) / 0 | -0.159 [0.00] | +0.029 | +0.037 |
| ② master B20 | 3 | 67 % → 67 % | -0.569 | -0.569 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 2 | 100 % → 100 % | +1.253 | +1.253 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master ALL | 19 | 79 % → 84 % | +0.372 | +0.393 | **+0.021** | +0.40 | [-0.106, +0.209] | 0.56 | 1 · +2.31 | 2 · -1.91 | 0 (+0.0) / 0 | -0.112 [0.00] | +0.021 | +0.028 |
| ③ BT 2025 | 899 | 69 % → 70 % | -0.303 | -0.313 | **-0.010** | -8.56 | [-0.050, +0.033] | 0.31 | 22 · +74.53 | 79 · -81.67 | 14 (+57.6) / 0 | -0.015 [0.23] | -0.014 | -0.009 |
| ③ BT 2025 top-50 | 230 | 72 % → 75 % | +0.021 | +0.007 | **-0.014** | -3.13 | [-0.113, +0.096] | 0.38 | 8 · +28.45 | 32 · -30.77 | 6 (+24.3) / 0 | -0.014 [0.38] | -0.014 | -0.010 |
| ③ BT 2025 not top-50 | 669 | 67 % → 68 % | -0.415 | -0.423 | **-0.008** | -5.43 | [-0.050, +0.039] | 0.36 | 14 · +46.08 | 47 · -50.91 | 8 (+33.3) / 0 | -0.016 [0.26] | – | -0.008 |
| ③ BT 2026-H1 | 410 | 75 % → 77 % | +0.125 | +0.126 | **+0.001** | +0.46 | [-0.075, +0.088] | 0.50 | 17 · +62.73 | 64 · -62.27 | 12 (+50.9) / 0 | +0.004 [0.52] | -0.003 | +0.000 |
| ③ BT 2026-H2 | 207 | 73 % → 76 % | +0.059 | +0.009 | **-0.050** | -10.40 | [-0.149, +0.064] | 0.18 | 8 · +23.44 | 35 · -33.85 | 5 (+18.2) / 0 | -0.072 [0.09] | -0.051 | -0.037 |
| ③ BT 2026 | 617 | 74 % → 77 % | +0.103 | +0.087 | **-0.016** | -9.95 | [-0.078, +0.053] | 0.32 | 25 · +86.17 | 99 · -96.12 | 17 (+69.1) / 0 | -0.023 [0.25] | -0.019 | -0.012 |

### LITE


**LITE · lock A = +1.5 → L = +0**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +3.019 | +3.019 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | – [–] | – | +0.000 |
| ② master B18 | 2 | 100 % → 100 % | +3.012 | +1.596 | **-1.416** | -2.83 | [–, –] | – | 0 · +0.00 | 1 · -2.83 | 0 (+0.0) / 0 | – [–] | -1.416 | -1.519 |
| ② master B19 | 13 | 54 % → 54 % | +0.113 | -0.333 | **-0.446** | -5.80 | [-0.781, +0.000] | 0.00 | 0 · +0.00 | 2 · -5.80 | 0 (+0.0) / 0 | -0.446 [0.00] | -0.446 | -0.464 |
| ② master B20 | 1 | 0 % → 100 % | -2.866 | +0.164 | **+3.030** | +3.03 | [–, –] | – | 1 · +3.03 | 0 · +0.00 | 1 (+3.0) / 0 | +3.030 [–] | +3.030 | +2.993 |
| ② master B21 | 4 | 50 % → 75 % | +0.089 | +0.944 | **+0.854** | +3.42 | [–, –] | – | 1 · +3.42 | 0 · +0.00 | 1 (+3.4) / 0 | +0.854 [–] | +0.854 | +0.750 |
| ② master ALL | 20 | 55 % → 65 % | +0.249 | +0.140 | **-0.109** | -2.18 | [-0.658, +0.782] | 0.42 | 2 · +6.45 | 3 · -8.63 | 2 (+6.4) / 0 | +0.036 [0.61] | -0.109 | -0.154 |
| ③ BT 2025 | 828 | 47 % → 35 % | -0.154 | -0.094 | **+0.060** | +49.72 | [-0.054, +0.179] | 0.85 | 157 · +459.31 | 135 · -406.93 | 153 (+454.9) / 0 | +0.057 [0.84] | -0.061 | +0.064 |
| ③ BT 2025 top-50 | 264 | 50 % → 47 % | -0.009 | -0.047 | **-0.038** | -10.12 | [-0.230, +0.149] | 0.33 | 43 · +126.82 | 45 · -136.93 | 43 (+126.8) / 0 | -0.038 [0.33] | -0.061 | -0.026 |
| ③ BT 2025 not top-50 | 564 | 46 % → 30 % | -0.222 | -0.115 | **+0.106** | +59.83 | [-0.046, +0.261] | 0.92 | 114 · +332.49 | 90 · -270.00 | 110 (+328.1) / 0 | +0.102 [0.91] | – | +0.106 |
| ③ BT 2026-H1 | 369 | 54 % → 52 % | +0.296 | +0.117 | **-0.180** | -66.27 | [-0.349, -0.011] | 0.02 | 55 · +154.14 | 73 · -218.68 | 48 (+136.1) / 1 | -0.225 [0.01] | -0.180 | -0.163 |
| ③ BT 2026-H2 | 255 | 54 % → 52 % | +0.232 | +0.147 | **-0.085** | -21.56 | [-0.306, +0.144] | 0.23 | 40 · +119.22 | 47 · -140.39 | 36 (+106.5) / 0 | -0.056 [0.31] | -0.085 | -0.089 |
| ③ BT 2026 | 624 | 54 % → 52 % | +0.270 | +0.129 | **-0.141** | -87.83 | [-0.265, -0.010] | 0.02 | 95 · +273.36 | 120 · -359.07 | 84 (+242.7) / 1 | -0.155 [0.01] | -0.141 | -0.133 |

**LITE · lock A = +1.5 → L = +0.5**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +3.019 | +3.019 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | – [–] | – | +0.000 |
| ② master B18 | 2 | 100 % → 100 % | +3.012 | +0.480 | **-2.531** | -5.06 | [–, –] | – | 0 · +0.00 | 2 · -5.06 | 0 (+0.0) / 0 | – [–] | -2.531 | -2.520 |
| ② master B19 | 13 | 54 % → 54 % | +0.113 | -0.486 | **-0.599** | -7.79 | [-1.505, -0.231] | 0.00 | 0 · +0.00 | 3 · -7.79 | 0 (+0.0) / 0 | -0.599 [0.00] | -0.599 | -0.580 |
| ② master B20 | 1 | 0 % → 100 % | -2.866 | +0.561 | **+3.426** | +3.43 | [–, –] | – | 1 · +3.43 | 0 · +0.00 | 1 (+3.4) / 0 | +3.426 [–] | +3.426 | +3.501 |
| ② master B21 | 4 | 50 % → 75 % | +0.089 | +0.930 | **+0.841** | +3.37 | [–, –] | – | 1 · +3.37 | 0 · +0.00 | 1 (+3.4) / 0 | +0.841 [–] | +0.841 | +0.875 |
| ② master ALL | 20 | 55 % → 65 % | +0.249 | -0.054 | **-0.303** | -6.06 | [-1.149, +0.631] | 0.22 | 2 · +6.79 | 5 · -12.85 | 2 (+6.8) / 0 | -0.055 [0.42] | -0.303 | -0.279 |
| ③ BT 2025 | 828 | 47 % → 66 % | -0.154 | -0.034 | **+0.120** | +99.28 | [-0.006, +0.251] | 0.97 | 159 · +540.48 | 175 · -440.08 | 153 (+530.2) / 0 | +0.113 [0.96] | +0.022 | +0.122 |
| ③ BT 2025 top-50 | 264 | 50 % → 66 % | -0.009 | +0.040 | **+0.049** | +12.91 | [-0.149, +0.249] | 0.67 | 43 · +149.15 | 54 · -136.24 | 43 (+149.2) / 0 | +0.049 [0.67] | +0.022 | +0.055 |
| ③ BT 2025 not top-50 | 564 | 46 % → 66 % | -0.222 | -0.068 | **+0.153** | +86.37 | [-0.015, +0.329] | 0.96 | 116 · +391.33 | 121 · -303.84 | 110 (+381.0) / 0 | +0.143 [0.95] | – | +0.153 |
| ③ BT 2026-H1 | 369 | 54 % → 68 % | +0.296 | +0.116 | **-0.180** | -66.51 | [-0.352, -0.004] | 0.02 | 55 · +182.40 | 98 · -247.67 | 48 (+161.8) / 1 | -0.230 [0.01] | -0.180 | -0.161 |
| ③ BT 2026-H2 | 255 | 54 % → 69 % | +0.232 | +0.159 | **-0.073** | -18.63 | [-0.286, +0.150] | 0.26 | 42 · +139.84 | 63 · -158.48 | 36 (+125.3) / 1 | -0.046 [0.34] | -0.073 | -0.073 |
| ③ BT 2026 | 624 | 54 % → 69 % | +0.270 | +0.134 | **-0.136** | -85.15 | [-0.266, -0.002] | 0.02 | 97 · +322.25 | 161 · -406.15 | 84 (+287.1) / 2 | -0.154 [0.02] | -0.137 | -0.125 |

**LITE · lock A = +2.0 → L = +0**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +3.019 | +3.019 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | – [–] | – | +0.000 |
| ② master B18 | 2 | 100 % → 100 % | +3.012 | +1.596 | **-1.416** | -2.83 | [–, –] | – | 0 · +0.00 | 1 · -2.83 | 0 (+0.0) / 0 | – [–] | -1.416 | -1.519 |
| ② master B19 | 13 | 54 % → 54 % | +0.113 | -0.336 | **-0.449** | -5.83 | [-0.781, +0.000] | 0.00 | 0 · +0.00 | 2 · -5.83 | 0 (+0.0) / 0 | -0.449 [0.00] | -0.449 | -0.463 |
| ② master B20 | 1 | 0 % → 0 % | -2.866 | -2.866 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 4 | 50 % → 75 % | +0.089 | +0.944 | **+0.854** | +3.42 | [–, –] | – | 1 · +3.42 | 0 · +0.00 | 1 (+3.4) / 0 | +0.854 [–] | +0.854 | +0.750 |
| ② master ALL | 20 | 55 % → 60 % | +0.249 | -0.013 | **-0.262** | -5.25 | [-0.716, +0.456] | 0.22 | 1 · +3.42 | 3 · -8.67 | 1 (+3.4) / 0 | -0.134 [0.37] | -0.262 | -0.303 |
| ③ BT 2025 | 828 | 47 % → 40 % | -0.154 | -0.131 | **+0.023** | +19.04 | [-0.067, +0.115] | 0.68 | 92 · +268.25 | 82 · -247.08 | 89 (+264.7) / 0 | +0.020 [0.66] | -0.191 | +0.026 |
| ③ BT 2025 top-50 | 264 | 50 % → 48 % | -0.009 | -0.180 | **-0.172** | -45.34 | [-0.318, -0.022] | 0.01 | 19 · +54.74 | 33 · -100.08 | 19 (+54.7) / 0 | -0.172 [0.01] | -0.191 | -0.161 |
| ③ BT 2025 not top-50 | 564 | 46 % → 37 % | -0.222 | -0.107 | **+0.114** | +64.38 | [-0.005, +0.241] | 0.97 | 73 · +213.51 | 49 · -147.00 | 70 (+210.0) / 0 | +0.110 [0.96] | – | +0.114 |
| ③ BT 2026-H1 | 369 | 54 % → 53 % | +0.296 | +0.202 | **-0.094** | -34.72 | [-0.237, +0.043] | 0.09 | 39 · +109.96 | 48 · -142.95 | 32 (+91.8) / 1 | -0.146 [0.03] | -0.094 | -0.088 |
| ③ BT 2026-H2 | 255 | 54 % → 53 % | +0.232 | +0.239 | **+0.007** | +1.86 | [-0.169, +0.192] | 0.54 | 28 · +84.95 | 28 · -83.09 | 26 (+78.7) / 0 | +0.021 [0.59] | +0.007 | -0.002 |
| ③ BT 2026 | 624 | 54 % → 53 % | +0.270 | +0.217 | **-0.053** | -32.86 | [-0.161, +0.056] | 0.17 | 67 · +194.92 | 76 · -226.04 | 58 (+170.4) / 1 | -0.077 [0.09] | -0.053 | -0.053 |

**LITE · lock A = +2.0 → L = +0.5**

| block | N | WR base → var | avg base | avg var | Δ %/fill | ΣΔ | day CI 95 % | P(Δ>0) | SAVED n · Σ (base STOP/CAP → better) | KILLED n · Σ (base TP cut) | Δ on recorded-STOP fills: saved / deeper | Δ on parity-matched fills [P] | ticks-only Δ | lat 0 Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ① LIVE | 1 | 100 % → 100 % | +3.019 | +3.019 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | – [–] | – | +0.000 |
| ② master B18 | 2 | 100 % → 100 % | +3.012 | +0.480 | **-2.531** | -5.06 | [–, –] | – | 0 · +0.00 | 2 · -5.06 | 0 (+0.0) / 0 | – [–] | -2.531 | -2.520 |
| ② master B19 | 13 | 54 % → 54 % | +0.113 | -0.274 | **-0.387** | -5.04 | [-0.712, +0.000] | 0.00 | 0 · +0.00 | 2 · -5.04 | 0 (+0.0) / 0 | -0.387 [0.00] | -0.387 | -0.386 |
| ② master B20 | 1 | 0 % → 0 % | -2.866 | -2.866 | **+0.000** | +0.00 | [–, –] | – | 0 · +0.00 | 0 · +0.00 | 0 (+0.0) / 0 | +0.000 [–] | +0.000 | +0.000 |
| ② master B21 | 4 | 50 % → 75 % | +0.089 | +0.930 | **+0.841** | +3.37 | [–, –] | – | 1 · +3.37 | 0 · +0.00 | 1 (+3.4) / 0 | +0.841 [–] | +0.841 | +0.875 |
| ② master ALL | 20 | 55 % → 60 % | +0.249 | -0.088 | **-0.337** | -6.73 | [-1.023, +0.383] | 0.16 | 1 · +3.37 | 4 · -10.10 | 1 (+3.4) / 0 | -0.093 [0.37] | -0.337 | -0.328 |
| ③ BT 2025 | 828 | 47 % → 58 % | -0.154 | -0.076 | **+0.078** | +64.40 | [-0.015, +0.179] | 0.94 | 94 · +319.03 | 101 · -253.54 | 89 (+310.1) / 0 | +0.070 [0.92] | -0.133 | +0.079 |
| ③ BT 2025 top-50 | 264 | 50 % → 56 % | -0.009 | -0.123 | **-0.115** | -30.32 | [-0.269, +0.039] | 0.07 | 19 · +65.24 | 38 · -95.56 | 19 (+65.2) / 0 | -0.115 [0.07] | -0.133 | -0.110 |
| ③ BT 2025 not top-50 | 564 | 46 % → 59 % | -0.222 | -0.054 | **+0.168** | +94.72 | [+0.037, +0.308] | 1.00 | 75 · +253.79 | 63 · -157.97 | 70 (+244.8) / 0 | +0.158 [0.99] | – | +0.168 |
| ③ BT 2026-H1 | 369 | 54 % → 64 % | +0.296 | +0.216 | **-0.080** | -29.56 | [-0.232, +0.067] | 0.15 | 39 · +132.23 | 64 · -160.55 | 32 (+111.6) / 1 | -0.122 [0.07] | -0.080 | -0.080 |
| ③ BT 2026-H2 | 255 | 54 % → 65 % | +0.232 | +0.229 | **-0.003** | -0.70 | [-0.192, +0.201] | 0.49 | 28 · +98.04 | 40 · -98.73 | 26 (+91.3) / 0 | -0.001 [0.50] | -0.003 | -0.011 |
| ③ BT 2026 | 624 | 54 % → 65 % | +0.270 | +0.222 | **-0.048** | -30.26 | [-0.164, +0.069] | 0.20 | 67 · +230.27 | 104 · -259.28 | 58 (+202.9) / 1 | -0.072 [0.12] | -0.049 | -0.052 |

## F. Pre-registered gates — every variant

G1 Δ > 0 with P ≥ 0.90 in 2025 AND 2026 · G2 master mean Δ ≥ 0 · G3 leave-one-month-out sign-stable · G4 no day / pair ≥ 50 % of ΣΔ · 2025 top-50 same sign. Q2 judged on ruler T1 (TL rows = sensitivity). Q3 must pass on all fills AND the parity-matched subset.

| sleeve | q | variant | Δ 2025 (n) [P] | Δ 2026 (n) [P] | master Δ (n) | LOMO min…max | top day / pair share | G1 | G2 | G3 | G4 | top-50 | CANDIDATE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WILLY | Q1 | Q1 B vs A L0.8 | +0.014 (230) [0.89] | -0.000 (593) [0.51] | +0.020 (19) | -0.001 … +0.012 | 0.46 / 0.84 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| WILLY | Q1 | Q1 B_touch vs A L0.8 | +0.038 (230) [0.96] | +0.025 (593) [0.98] | +0.020 (19) | +0.023 … +0.032 | 0.24 / 0.22 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q1 | Q1 B+worst vs A L0.8 | +0.014 (230) [0.82] | -0.010 (593) [0.26] | -0.027 (19) | -0.008 … +0.006 | 1.64 / 1.64 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| WILLY | Q1 | Q1 B vs A L1.5 | +0.003 (230) [0.59] | +0.002 (593) [0.57] | -0.002 (19) | -0.004 … +0.006 | 1.16 / 1.53 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| WILLY | Q1 | Q1 B_touch vs A L1.5 | +0.027 (230) [0.85] | +0.026 (593) [0.97] | -0.002 (19) | +0.022 … +0.030 | 0.25 / 0.23 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q1 | Q1 B+worst vs A L1.5 | -0.003 (230) [0.43] | -0.014 (593) [0.17] | -0.013 (19) | -0.013 … -0.005 | 0.56 / 0.56 | ✗ | ✗ | ✓ | ✗ | ✓ | no |
| WILLY | Q1 | Q1 B vs A L2.6 | -0.004 (230) [0.42] | +0.010 (593) [0.76] | +0.043 (19) | -0.002 … +0.010 | 0.43 / 0.55 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| WILLY | Q1 | Q1 B_touch vs A L2.6 | +0.020 (230) [0.75] | +0.034 (593) [0.99] | +0.043 (19) | +0.024 … +0.037 | 0.22 / 0.20 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q1 | Q1 B+worst vs A L2.6 | -0.019 (230) [0.13] | -0.012 (593) [0.19] | +0.035 (19) | -0.019 … -0.007 | 0.44 / 0.43 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B vs A L0.8 | -0.005 (263) [0.48] | +0.038 (624) [1.00] | +0.086 (20) | +0.017 … +0.033 | 0.27 / 0.27 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B_touch vs A L0.8 | +0.041 (263) [0.98] | +0.038 (624) [1.00] | +0.086 (20) | +0.031 … +0.045 | 0.18 / 0.18 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q1 | Q1 B+worst vs A L0.8 | -0.040 (263) [0.11] | +0.039 (624) [0.99] | +0.024 (20) | +0.009 … +0.023 | 0.43 / 0.43 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B vs A L1.5 | -0.015 (263) [0.32] | +0.028 (624) [0.94] | +0.099 (20) | +0.006 … +0.024 | 0.46 / 0.47 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B_touch vs A L1.5 | +0.032 (263) [0.89] | +0.028 (624) [0.94] | +0.099 (20) | +0.020 … +0.039 | 0.24 / 0.25 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B+worst vs A L1.5 | -0.047 (263) [0.06] | +0.028 (624) [0.95] | +0.083 (20) | -0.002 … +0.013 | 1.26 / 1.25 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| LITE | Q1 | Q1 B vs A L2.6 | -0.014 (263) [0.33] | +0.038 (624) [0.98] | +0.090 (20) | +0.012 … +0.031 | 0.32 / 0.32 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B_touch vs A L2.6 | +0.033 (263) [0.90] | +0.038 (624) [0.98] | +0.090 (20) | +0.027 … +0.046 | 0.20 / 0.20 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q1 | Q1 B+worst vs A L2.6 | -0.041 (263) [0.09] | +0.037 (624) [0.99] | +0.087 (20) | +0.006 … +0.022 | 0.48 / 0.47 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 5s · skip | +0.061 (230) [0.70] | -0.016 (593) [0.39] | +0.294 (19) | -0.010 … +0.026 | 1.94 / 3.26 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 5s · taker fallback | +0.063 (230) [0.94] | +0.002 (593) [0.54] | +0.066 (19) | +0.004 … +0.026 | 0.35 / 0.42 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 10s · skip | +0.064 (230) [0.76] | +0.005 (593) [0.52] | +0.362 (19) | +0.002 … +0.038 | 0.60 / 0.61 | ✗ | ✓ | ✓ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 10s · taker fallback | +0.061 (230) [0.89] | -0.005 (593) [0.45] | +0.041 (19) | +0.005 … +0.027 | 0.97 / 0.59 | ✗ | ✓ | ✓ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 20s · skip | +0.049 (230) [0.70] | +0.032 (593) [0.72] | +0.152 (19) | +0.024 … +0.048 | 0.36 / 0.36 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0% · 20s · taker fallback | +0.040 (230) [0.82] | +0.005 (593) [0.54] | +0.066 (19) | -0.001 … +0.035 | 0.89 / 0.55 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 5s · skip | +0.124 (230) [0.85] | -0.080 (593) [0.11] | +0.438 (19) | -0.039 … +0.005 | 0.33 / 0.40 | ✗ | ✓ | ✗ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 5s · taker fallback | +0.083 (230) [0.97] | +0.017 (593) [0.70] | +0.065 (19) | +0.023 … +0.044 | 0.19 / 0.22 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 10s · skip | +0.144 (230) [0.91] | -0.036 (593) [0.27] | +0.296 (19) | -0.004 … +0.035 | 0.80 / 1.14 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 10s · taker fallback | +0.080 (230) [0.93] | -0.007 (593) [0.44] | +0.041 (19) | +0.006 … +0.032 | 0.76 / 0.48 | ✗ | ✓ | ✓ | ✗ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 20s · skip | +0.154 (230) [0.94] | +0.005 (593) [0.52] | +0.087 (19) | +0.022 … +0.063 | 0.25 / 0.35 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2 | Q2 limit P0−0.1% · 20s · taker fallback | +0.101 (230) [0.97] | -0.014 (593) [0.38] | +0.067 (19) | +0.007 … +0.034 | 0.37 / 0.45 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q2-TL | Q2 limit P0−0% · 5s · skip [TL] | +0.220 (230) [0.97] | +0.237 (593) [1.00] | +0.399 (19) | +0.212 … +0.265 | 0.06 / 0.09 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0% · 5s · taker fallback [TL] | +0.122 (230) [0.98] | +0.167 (593) [1.00] | +0.067 (19) | +0.143 … +0.168 | 0.05 / 0.09 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0% · 10s · skip [TL] | +0.224 (230) [0.98] | +0.258 (593) [1.00] | +0.467 (19) | +0.221 … +0.271 | 0.05 / 0.07 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0% · 10s · taker fallback [TL] | +0.093 (230) [0.87] | +0.177 (593) [1.00] | -0.033 (19) | +0.143 … +0.166 | 0.08 / 0.10 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q2-TL | Q2 limit P0−0% · 20s · skip [TL] | +0.208 (230) [0.98] | +0.285 (593) [1.00] | +0.257 (19) | +0.238 … +0.280 | 0.05 / 0.07 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0% · 20s · taker fallback [TL] | +0.120 (230) [0.94] | +0.187 (593) [1.00] | +0.149 (19) | +0.157 … +0.181 | 0.08 / 0.08 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 5s · skip [TL] | +0.283 (230) [0.99] | +0.173 (593) [0.99] | +0.543 (19) | +0.187 … +0.244 | 0.06 / 0.11 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 5s · taker fallback [TL] | +0.109 (230) [0.97] | +0.132 (593) [1.00] | +0.065 (19) | +0.110 … +0.143 | 0.07 / 0.12 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 10s · skip [TL] | +0.303 (230) [1.00] | +0.217 (593) [1.00] | +0.401 (19) | +0.220 … +0.275 | 0.05 / 0.10 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 10s · taker fallback [TL] | +0.077 (230) [0.82] | +0.133 (593) [0.99] | -0.033 (19) | +0.098 … +0.133 | 0.07 / 0.18 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 20s · skip [TL] | +0.313 (230) [1.00] | +0.258 (593) [1.00] | +0.192 (19) | +0.246 … +0.301 | 0.05 / 0.09 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| WILLY | Q2-TL | Q2 limit P0−0.1% · 20s · taker fallback [TL] | +0.169 (230) [0.98] | +0.146 (593) [1.00] | +0.150 (19) | +0.134 … +0.172 | 0.05 / 0.13 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q2 | Q2 limit P0−0% · 5s · skip | -0.012 (263) [0.46] | -0.122 (624) [0.10] | +0.164 (20) | -0.131 … -0.058 | 0.15 / 0.20 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0% · 5s · taker fallback | -0.036 (263) [0.30] | +0.017 (624) [0.76] | +0.013 (20) | -0.007 … +0.015 | 6.24 / 6.22 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0% · 10s · skip | +0.070 (263) [0.69] | -0.107 (624) [0.11] | -0.133 (20) | -0.093 … -0.021 | 0.20 / 0.31 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0% · 10s · taker fallback | +0.012 (263) [0.61] | +0.029 (624) [0.82] | +0.019 (20) | +0.010 … +0.040 | 0.29 / 0.29 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0% · 20s · skip | +0.095 (263) [0.78] | -0.044 (624) [0.29] | -0.279 (20) | -0.030 … +0.023 | 4.10 / 5.36 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0% · 20s · taker fallback | -0.042 (263) [0.27] | +0.031 (624) [0.81] | -0.129 (20) | -0.014 … +0.027 | 0.77 / 0.77 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 5s · skip | +0.065 (263) [0.64] | -0.110 (624) [0.14] | +0.164 (20) | -0.097 … -0.019 | 0.35 / 0.41 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 5s · taker fallback | -0.016 (263) [0.46] | +0.014 (624) [0.69] | +0.013 (20) | -0.002 … +0.020 | 1.28 / 1.30 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 10s · skip | +0.147 (263) [0.83] | -0.113 (624) [0.13] | +0.019 (20) | -0.068 … -0.000 | 0.47 / 0.56 | ✗ | ✓ | ✓ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 10s · taker fallback | +0.010 (263) [0.59] | +0.015 (624) [0.68] | +0.020 (20) | -0.001 … +0.029 | 0.51 / 0.51 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 20s · skip | +0.104 (263) [0.76] | -0.063 (624) [0.25] | +0.170 (20) | -0.038 … +0.018 | 0.80 / 1.26 | ✗ | ✓ | ✗ | ✗ | ✓ | no |
| LITE | Q2 | Q2 limit P0−0.1% · 20s · taker fallback | -0.034 (263) [0.34] | +0.018 (624) [0.67] | -0.130 (20) | -0.021 … +0.020 | 2.80 / 2.84 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0% · 5s · skip [TL] | +0.317 (263) [0.98] | +0.054 (624) [0.71] | +0.164 (20) | +0.086 … +0.170 | 0.13 / 0.21 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0% · 5s · taker fallback [TL] | +0.179 (263) [1.00] | +0.083 (624) [0.99] | +0.012 (20) | +0.088 … +0.122 | 0.06 / 0.12 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q2-TL | Q2 limit P0−0% · 10s · skip [TL] | +0.398 (263) [1.00] | +0.069 (624) [0.78] | -0.134 (20) | +0.124 … +0.191 | 0.08 / 0.14 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0% · 10s · taker fallback [TL] | +0.204 (263) [1.00] | +0.135 (624) [1.00] | +0.017 (20) | +0.134 … +0.169 | 0.07 / 0.09 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q2-TL | Q2 limit P0−0% · 20s · skip [TL] | +0.424 (263) [1.00] | +0.132 (624) [0.94] | -0.280 (20) | +0.189 … +0.248 | 0.06 / 0.09 | ✓ | ✗ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0% · 20s · taker fallback [TL] | +0.172 (263) [0.99] | +0.118 (624) [0.99] | -0.129 (20) | +0.109 … +0.155 | 0.08 / 0.10 | ✓ | ✗ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0.1% · 5s · skip [TL] | +0.394 (263) [0.99] | +0.066 (624) [0.73] | +0.164 (20) | +0.123 … +0.209 | 0.10 / 0.17 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0.1% · 5s · taker fallback [TL] | +0.153 (263) [0.99] | +0.042 (624) [0.91] | +0.013 (20) | +0.057 … +0.085 | 0.09 / 0.13 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q2-TL | Q2 limit P0−0.1% · 10s · skip [TL] | +0.476 (263) [1.00] | +0.063 (624) [0.73] | +0.018 (20) | +0.146 … +0.220 | 0.09 / 0.11 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0.1% · 10s · taker fallback [TL] | +0.202 (263) [1.00] | +0.102 (624) [1.00] | +0.019 (20) | +0.116 … +0.150 | 0.08 / 0.07 | ✓ | ✓ | ✓ | ✓ | ✓ | **YES** |
| LITE | Q2-TL | Q2 limit P0−0.1% · 20s · skip [TL] | +0.433 (263) [1.00] | +0.113 (624) [0.88] | +0.170 (20) | +0.179 … +0.243 | 0.08 / 0.10 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| LITE | Q2-TL | Q2 limit P0−0.1% · 20s · taker fallback [TL] | +0.182 (263) [0.99] | +0.085 (624) [0.97] | -0.130 (20) | +0.086 … +0.133 | 0.09 / 0.12 | ✓ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.6 L-1 | -0.047 (899) [0.07] | -0.041 (617) [0.19] | -0.023 (19) | -0.054 … -0.032 | 0.13 / 0.11 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.6 L-1 [parity-matched] | -0.053 (833) [0.06] | -0.045 (549) [0.20] | -0.102 (17) | -0.062 … -0.037 | 0.13 / 0.13 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.6 L+0 | -0.040 (899) [0.13] | -0.014 (617) [0.40] | -0.128 (19) | -0.041 … -0.016 | 0.16 / 0.15 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.6 L+0 [parity-matched] | -0.050 (833) [0.09] | -0.006 (549) [0.46] | -0.279 (17) | -0.047 … -0.016 | 0.16 / 0.17 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.6 L+0.3 | -0.033 (899) [0.17] | -0.023 (617) [0.34] | -0.109 (19) | -0.042 … -0.012 | 0.17 / 0.21 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.6 L+0.3 [parity-matched] | -0.044 (833) [0.12] | -0.021 (549) [0.37] | -0.258 (17) | -0.052 … -0.016 | 0.16 / 0.19 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.8 L-1 | -0.030 (899) [0.13] | -0.034 (617) [0.20] | +0.069 (19) | -0.043 … -0.023 | 0.18 / 0.12 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.8 L-1 [parity-matched] | -0.035 (833) [0.11] | -0.039 (549) [0.19] | +0.000 (17) | -0.051 … -0.027 | 0.17 / 0.11 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.8 L+0 | -0.031 (899) [0.15] | +0.014 (617) [0.62] | -0.067 (19) | -0.028 … -0.001 | 0.31 / 0.34 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.8 L+0 [parity-matched] | -0.040 (833) [0.10] | +0.026 (549) [0.71] | -0.211 (17) | -0.032 … -0.002 | 0.30 / 0.33 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A0.8 L+0.3 | -0.026 (899) [0.19] | +0.018 (617) [0.65] | -0.076 (19) | -0.025 … +0.002 | 0.40 / 0.52 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A0.8 L+0.3 [parity-matched] | -0.036 (833) [0.14] | +0.020 (549) [0.66] | -0.220 (17) | -0.033 … -0.004 | 0.26 / 0.33 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A1 L-1 | -0.012 (899) [0.24] | -0.025 (617) [0.19] | +0.069 (19) | -0.023 … -0.012 | 0.18 / 0.20 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A1 L-1 [parity-matched] | -0.015 (833) [0.21] | -0.039 (549) [0.10] | +0.000 (17) | -0.033 … -0.018 | 0.14 / 0.15 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A1 L+0 | -0.011 (899) [0.29] | -0.010 (617) [0.38] | -0.002 (19) | -0.019 … -0.003 | 0.27 / 0.23 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A1 L+0 [parity-matched] | -0.016 (833) [0.22] | -0.013 (549) [0.34] | -0.138 (17) | -0.025 … -0.007 | 0.21 / 0.18 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| WILLY | Q3 | Q3 lock A1 L+0.3 | -0.010 (899) [0.31] | -0.016 (617) [0.32] | +0.021 (19) | -0.021 … -0.006 | 0.21 / 0.20 | ✗ | ✓ | ✓ | ✓ | ✓ | no |
| WILLY | Q3-PAR | Q3 lock A1 L+0.3 [parity-matched] | -0.015 (833) [0.23] | -0.023 (549) [0.25] | -0.112 (17) | -0.030 … -0.012 | 0.15 / 0.14 | ✗ | ✗ | ✓ | ✓ | ✓ | no |
| LITE | Q3 | Q3 lock A1.5 L+0 | +0.060 (828) [0.85] | -0.141 (624) [0.02] | -0.109 (20) | -0.042 … +0.009 | 0.24 / 0.29 | ✗ | ✗ | ✗ | ✓ | ✗ | no |
| LITE | Q3-PAR | Q3 lock A1.5 L+0 [parity-matched] | +0.057 (823) [0.84] | -0.155 (564) [0.01] | +0.036 (18) | -0.046 … +0.010 | 0.22 / 0.30 | ✗ | ✓ | ✗ | ✓ | ✗ | no |
| LITE | Q3 | Q3 lock A1.5 L+0.5 | +0.120 (828) [0.97] | -0.136 (624) [0.02] | -0.303 (20) | -0.009 … +0.042 | 0.76 / 1.01 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| LITE | Q3-PAR | Q3 lock A1.5 L+0.5 [parity-matched] | +0.113 (823) [0.96] | -0.154 (564) [0.02] | -0.055 (18) | -0.016 … +0.040 | 1.94 / 2.59 | ✗ | ✗ | ✗ | ✗ | ✓ | no |
| LITE | Q3 | Q3 lock A2 L+0 | +0.023 (828) [0.68] | -0.053 (624) [0.17] | -0.262 (20) | -0.022 … +0.017 | 0.64 / 0.84 | ✗ | ✗ | ✗ | ✗ | ✗ | no |
| LITE | Q3-PAR | Q3 lock A2 L+0 [parity-matched] | +0.020 (823) [0.66] | -0.077 (564) [0.09] | -0.134 (18) | -0.033 … +0.008 | 0.32 / 0.33 | ✗ | ✗ | ✗ | ✓ | ✗ | no |
| LITE | Q3 | Q3 lock A2 L+0.5 | +0.078 (828) [0.94] | -0.048 (624) [0.20] | -0.337 (20) | +0.009 … +0.052 | 0.32 / 0.32 | ✗ | ✗ | ✓ | ✓ | ✗ | no |
| LITE | Q3-PAR | Q3 lock A2 L+0.5 [parity-matched] | +0.070 (823) [0.92] | -0.072 (564) [0.12] | -0.093 (18) | -0.004 … +0.042 | 0.65 / 0.65 | ✗ | ✗ | ✗ | ✗ | ✗ | no |

**Candidates under the pre-registered rules: NONE** · variants judged: 55 (+ 24 TL sensitivity rows, + 13 parity re-reads).
Single-L / context passes (NOT candidates — Q1 needs every L, B_touch is the optimistic bound): WILLY Q1 B_touch vs A L0.8, LITE Q1 B_touch vs A L0.8.
Sensitivity-only passes (ruler TL, N = 5 live entries — not a second chance): 14 rows: WILLY limit P0−0% · 5s · skip [TL], WILLY limit P0−0% · 5s · taker fallback [TL], WILLY limit P0−0% · 10s · skip [TL], WILLY limit P0−0% · 20s · skip [TL], WILLY limit P0−0% · 20s · taker fallback [TL], WILLY limit P0−0.1% · 5s · skip [TL], WILLY limit P0−0.1% · 5s · taker fallback [TL], WILLY limit P0−0.1% · 10s · skip [TL], WILLY limit P0−0.1% · 20s · skip [TL], WILLY limit P0−0.1% · 20s · taker fallback [TL], LITE limit P0−0% · 5s · taker fallback [TL], LITE limit P0−0% · 10s · taker fallback [TL], LITE limit P0−0.1% · 5s · taker fallback [TL], LITE limit P0−0.1% · 10s · taker fallback [TL].

## G. Q1 sensitivity — if today's market TP also pays the LIVE book slippage

Live TP fills gave back −0.13 / −0.22 / −0.22 vs the trigger (mean −0.19); the trade-print model gives ≈ −0.01 → an unmodelled book cost of ≈ 0.18 per market TP. Below: Δ(B − A) when every A-TP fill pays an extra s_b (B's maker fills pay none; stops / caps identical in both arms). Not a gate — the live shadow (design Phase 2) is what measures s_b.

| sleeve | L | s_b | 2025 Δ [P] | 2026 Δ [P] | master Δ | B+worst 2026 Δ [P] (its market net exits pay s_b too) |
|---|---|---|---|---|---|---|
| WILLY | 0.8 | 0.1 | +0.084 [1.00] | +0.074 [1.00] | +0.099 | -0.010 [0.26] |
| WILLY | 0.8 | 0.18 | +0.141 [1.00] | +0.133 [1.00] | +0.162 | -0.010 [0.26] |
| WILLY | 1.5 | 0.1 | +0.073 [1.00] | +0.076 [1.00] | +0.077 | -0.014 [0.18] |
| WILLY | 1.5 | 0.18 | +0.129 [1.00] | +0.135 [1.00] | +0.140 | -0.014 [0.17] |
| WILLY | 2.6 | 0.1 | +0.067 [1.00] | +0.084 [1.00] | +0.122 | -0.012 [0.19] |
| WILLY | 2.6 | 0.18 | +0.123 [1.00] | +0.143 [1.00] | +0.185 | -0.012 [0.19] |
| LITE | 0.8 | 0.1 | +0.045 [0.93] | +0.092 [1.00] | +0.141 | +0.039 [0.99] |
| LITE | 0.8 | 0.18 | +0.084 [0.99] | +0.135 [1.00] | +0.185 | +0.038 [0.99] |
| LITE | 1.5 | 0.1 | +0.035 [0.88] | +0.082 [1.00] | +0.154 | +0.027 [0.95] |
| LITE | 1.5 | 0.18 | +0.074 [0.99] | +0.125 [1.00] | +0.198 | +0.027 [0.95] |
| LITE | 2.6 | 0.1 | +0.035 [0.88] | +0.092 [1.00] | +0.145 | +0.037 [0.99] |
| LITE | 2.6 | 0.18 | +0.075 [0.98] | +0.135 [1.00] | +0.189 | +0.037 [0.99] |


## Review corrections (deep review, 2026-10-15) — conclusions HOLD, 0 candidates
- **Q2 limit placement bug (fixed in scripts/study_tick_maker_lock_walk.py):** float32 prices floored without snapping to the tick grid
  put the maker limit 1 tick too low ~half the time. Re-run of all 1,750 Q2 signals with grid-snapped prices (reviewer's
  scratchpad q2fix): fill rates +1–3 pts (m0 T5 44.5 → 47.6 %), Δ up to +0.07; best 2026 P under the primary ruler 0.82 → still
  **0 Q2 candidates**. The tables above are the pre-fix numbers.
- **Entry-cost headline restated:** the slip sensitivity was priced at **0.25 %** (nearest grid point), not 0.27. Today's taker entry,
  avg %/signal at slip 0 / 0.036 / 0.15 / 0.25: WILLY 2025 (230 top-50 tick signals) −0.059 / −0.072 / −0.168 / −0.231 — already
  negative at ZERO slip on this subset (the replay records +0.011 on the same 230 signals: entry timing +8 s vs the replay's ~+14 s —
  these signals are extremely entry-sensitive); WILLY 2026 (593) +0.111 / +0.075 / −0.034 / −0.178 → **break-even entry slip ≈ 0.11 %**;
  LITE 2025 −0.093 / −0.094 / −0.300 / −0.421; LITE 2026 +0.215 / +0.186 / +0.099 / +0.010. Live entry slip so far: median 0.22,
  mean 0.267 (5 fills, one 0.73).
- **Q3 wording:** "kills 1.3–4× more winners" is a COUNT; in points the two sides are close (A +0.8 / L 0, 2026: 49 saved +155 vs
  118 killed −145) and two 2026 cells are net positive (0.8/0 +0.014, 0.8/0.3 +0.018), neither significant. 2025: all 9 negative.
- Scout 30-min cells on all 617 2026 fills: −0.0011 / +0.0249 (P 0.85) / +0.0284 (P 0.88) — stronger than the 606-fill refs, not judged
  here (they keep their own frozen live gate).
