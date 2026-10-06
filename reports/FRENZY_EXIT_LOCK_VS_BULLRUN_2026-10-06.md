# FRENZY exit: the live lock vs the BULLRUN exit and break-even hybrids (2026-10-06)

Research only. No bot file, config or test was touched. Nothing was committed. Variants frozen first in `reports/FRENZY_EXIT_LOCK_VS_BULLRUN_PREREG_2026-10-06.txt`.

## Verdict: keep the lock

**Not one variant beats the live lock on any split.**
- **Split by split:** every Δ vs the lock is negative, on all fills, strong, normal and WIDE hold-green.
- **Selection-adjusted p = 1.00 everywhere.** Nothing even reaches the observe-candidate bar (Δ > 0 in both halves).
- **The operator is right about ORCA 18:05.** It peaked at +1.38 %. The BULLRUN exit, or the +1 → +0.2 hybrid (V3), would have closed it at about +0.2 % instead of −3.01 %. That is roughly +$14 instead of −$213.
- **Over the year, that same rule costs more than it saves.** For every trade like ORCA that it rescues, it cuts a lock winner of the same size:
  - V3 on all 205 FRENZY fills: **45 trades saved, +144 % in total.**
  - **38 lock wins cut** (they finished ≥ +2 under the lock): **−144 %.**
  - Another 21 lock wins, the ones that ended exactly at the +1.9 lock floor, are also cut to +0.1: −38 %. That leaves V3 at Δ −0.18 per fill.
  - FRENZY winners usually dip back through break-even before they run. 57 % of the lock wins (38 of 67) touched +1 and then fell back under +0.2 before reaching +3. The median low after the break-even exit was **−0.88 %**, a real dip that live price polling would also catch.

| (fills, % per fill at 1×, 0.10 slippage) | ALL 205 | STRONG 115 | NORMAL 90 | WIDE-HG 125 |
|---|---|---|---|---|
| **V0 live lock** | **+0.387** | **+0.672** | +0.024 | **+0.457** |
| V1 BULLRUN as live (SL −1.2 after ATR widening) | +0.010 (Δ −0.38) | +0.175 (Δ −0.50) | −0.200 | −0.101 (Δ −0.56, CI < 0) |
| V1b BULLRUN with a flat −0.7 SL | −0.095 (Δ −0.48, CI < 0) | −0.009 | −0.206 | −0.140 (CI < 0) |
| V2 BULLRUN with the −3 stop | +0.128 (Δ −0.26) | +0.386 | −0.201 | −0.077 (CI < 0) |
| V3 −3 · +1 → +0.2 · lock from +3 | +0.204 (Δ −0.18) | +0.487 (Δ −0.19) | −0.157 | +0.120 (Δ −0.34) |
| V4 as V3, break-even arm +1.5 | +0.113 (Δ −0.27, CI < 0) | +0.399 | −0.252 | +0.037 (CI < 0) |
| V5 as V3, floor +0.5 | +0.095 (Δ −0.29) | +0.303 | −0.171 | +0.003 |
| **$3k FRENZY book (strong 10× / normal 6×)** | V0 **$9,741** (DD −43 %) · V3 $6,208 (−37 %) · V1 $3,273 (−26 %) · V2 $4,904 · V4 $4,391 (−54 %) · V5 $4,167 | | | |

**Win rate rises from 54 % to 76 % under the hybrids, but the average win falls from +3.4 to about +1.2.**
- That is the whole story.
- A 76 % win rate feels better, but the lock earns its money from the minority of trades that go on to +3 and beyond, after a dip.

**The strong split does not change this.**
- V3 on strong fills is Δ −0.19, CI [−0.61, +0.26], with halves −0.49 / +0.07.
- Strong fills are the lock's best cohort (+0.67), so they have the most winners to cut.

**What was asked about earlier tests.**

| test | cohort | ruler | baseline | BE / BR-like variants | result |
|---|---|---|---|---|---|
| `FRENZY_EXIT_BULLRUN_TEST_2026-10-03` | **pre-parity** hand-rolled ON rule of `frenzy_reentry_while_on_test` (1h above-average closes, not the engine `frenzy_walk`), 487 trades | 1m bars | the OLD +5 / 1.5 trail | BR stack, BE + ladder, ATR trails | BR −0.43 / trade vs the old trail |
| `FRENZY_EXIT_PROTECTION_TEST_2026-10-02` | pre-parity, 1,121 entries | 1m | the old +5 / 1.5 trail | BE at +1.5 / +2 / +3 | all ≤ baseline |
| `FRENZY_REDO_EXITS_2026-10-05` (#1, 75 variants) | **engine cohort** | 12 s ticks | fixed +3 and the lock | lock / frac / ATR / step / cap; **every one arms at ≥ +3** | lock best; **no break-even below +3 tested** |
| `FRENZY_REDO_EXITS` #6 / `FRENZY_SPLIT_RUNNER_EMA200` | engine cohort | ticks | the lock | +0.2 / 0 break-even on the **runner half only, after the lock arms** | not a first-position break-even |

**So this is the first engine-cohort, tick-priced test of a break-even that arms before +3.**
- It confirms the pre-parity result.
- The size of the cost is smaller than the old −0.43 because the baseline is now the lock, not the +5 trail. The direction is the same.

### Decision against the frozen bar
- **Arm proposal:** none. Every variant fails the first leg (Δ day-CI low > 0), on every split.
- **Observe candidate (Δ > 0 and both halves > 0):** none. The best cell, V3 STRONG, has a negative mean Δ and a negative first half.
- **Keep the lock** (DECISION_LOG 205). Its existing revert gate stays as it is.
- **No new scout shadow is proposed.** 205 year fills already answer the question, and a 40-fill forward shadow could not overturn it.
  - If the operator still wants to watch it, add a V3 line to the existing per-fill FRENZY exit shadows (decision 214).
  - Frozen bar for that line: re-open only if, after ≥ 40 live FRENZY_LONG fills on ≥ 20 days, V3 − lock is > +0.30 per fill with a day CI low > 0 and is positive without its top 5 fills.

**Why the live run feels different** (anecdotes below). The last 12 bot FRENZY / WIDE fills favour the hybrids: Σ V0 −12.0 % vs V3 −1.7 %.
- Those fills cover 4 days, in a stretch where the lock is losing.
- The year shows the same mechanical pattern by month. The hybrids gain mainly in the months where the lock loses: ALL Jul, lock −0.97, V3 +0.15; WIDE-HG Apr.
- They lose in every month where the lock wins.
- That is not something an exit rule can switch on at entry. The engine-cohort regime study (FRENZY_REGIME_2026-10-05) found no FRENZY regime variable.

---

## 1. Per variant × split (full read-out)

Δ = variant − V0 on the same fill, mean % per fill. The CI is a day-block bootstrap (2,000 draws). "w/o top 5 / 10 Δ" drops the fills where the variant gained most.

p raw / sel-adj is one-sided P(Δ > 0) from a joint per-day sign flip (5,000 draws). The selection adjustment is the max-t over the family of 6 variants × 4 splits = 24 cells.

### ALL — 205 fills, 133 days

| exit | mean % | WR | avg win / loss | Δ vs V0 [day CI] | Δ Jan–Apr / May–Sep | Δ w/o top 5 / 10 Δ | months Δ ≥ 0 | mean w/o own top 5 / 10 | p raw / sel-adj |
|---|---|---|---|---|---|---|---|---|---|
| V0 live lock | **+0.387** | 54% | +3.42 / -3.12 | — | +0.00 / +0.00 | +0.000 / +0.000 | — | +0.151 / -0.018 | — |
| V1 BULLRUN (live fn, SL −1.2) | **+0.010** | 53% | +1.20 / -1.31 | **-0.377** [-0.80, +0.05] | -0.67 / -0.07 | -0.469 / -0.563 | 3/9 | -0.119 / -0.234 | 0.95 / 1.00 |
| V1b BULLRUN, flat −0.7 SL | **-0.095** | 39% | +1.05 / -0.81 | **-0.483** [-0.93, -0.05] | -0.69 / -0.27 | -0.578 / -0.675 | 1/9 | -0.217 / -0.310 | 0.98 / 1.00 |
| V2 BULLRUN, −3 stop | **+0.128** | 76% | +1.18 / -3.12 | **-0.259** [-0.56, +0.03] | -0.52 / +0.02 | -0.349 / -0.440 | 3/9 | -0.003 / -0.125 | 0.94 / 1.00 |
| V3 hybrid +1→+0.2, lock ≥+3 | **+0.204** | 76% | +1.28 / -3.12 | **-0.183** [-0.47, +0.11] | -0.39 / +0.03 | -0.270 / -0.359 | 4/9 | -0.022 / -0.171 | 0.89 / 1.00 |
| V4 hybrid +1.5→+0.2 | **+0.113** | 68% | +1.65 / -3.12 | **-0.274** [-0.53, -0.01] | -0.43 / -0.11 | -0.363 / -0.454 | 3/9 | -0.116 / -0.273 | 0.98 / 1.00 |
| V5 hybrid +1→+0.5 | **+0.095** | 76% | +1.13 / -3.12 | **-0.293** [-0.60, +0.01] | -0.42 / -0.16 | -0.389 / -0.489 | 1/9 | -0.122 / -0.250 | 0.96 / 1.00 |

### STRONG — 115 fills, 94 days

| exit | mean % | WR | avg win / loss | Δ vs V0 [day CI] | Δ Jan–Apr / May–Sep | Δ w/o top 5 / 10 Δ | months Δ ≥ 0 | mean w/o own top 5 / 10 | p raw / sel-adj |
|---|---|---|---|---|---|---|---|---|---|
| V0 live lock | **+0.672** | 55% | +3.80 / -3.12 | — | +0.00 / +0.00 | +0.000 / +0.000 | — | +0.281 / -0.017 | — |
| V1 BULLRUN (live fn, SL −1.2) | **+0.175** | 55% | +1.41 / -1.32 | **-0.497** [-1.16, +0.14] | -0.84 / -0.21 | -0.667 / -0.852 | 2/9 | -0.043 / -0.235 | 0.92 / 1.00 |
| V1b BULLRUN, flat −0.7 SL | **-0.009** | 40% | +1.20 / -0.81 | **-0.681** [-1.41, +0.01] | -1.04 / -0.38 | -0.860 / -1.054 | 2/9 | -0.208 / -0.380 | 0.96 / 1.00 |
| V2 BULLRUN, −3 stop | **+0.386** | 77% | +1.46 / -3.12 | **-0.286** [-0.75, +0.18] | -0.69 / +0.05 | -0.448 / -0.622 | 5/9 | +0.159 / -0.051 | 0.88 / 1.00 |
| V3 hybrid +1→+0.2, lock ≥+3 | **+0.487** | 77% | +1.59 / -3.12 | **-0.185** [-0.61, +0.26] | -0.49 / +0.07 | -0.340 / -0.509 | 5/9 | +0.088 / -0.164 | 0.80 / 1.00 |
| V4 hybrid +1.5→+0.2 | **+0.399** | 70% | +1.94 / -3.12 | **-0.273** [-0.66, +0.09] | -0.48 / -0.10 | -0.432 / -0.605 | 5/9 | -0.004 / -0.275 | 0.92 / 1.00 |
| V5 hybrid +1→+0.5 | **+0.303** | 77% | +1.35 / -3.12 | **-0.369** [-0.80, +0.11] | -0.62 / -0.16 | -0.545 / -0.738 | 4/9 | -0.075 / -0.282 | 0.93 / 1.00 |

### NORMAL — 90 fills, 71 days

| exit | mean % | WR | avg win / loss | Δ vs V0 [day CI] | Δ Jan–Apr / May–Sep | Δ w/o top 5 / 10 Δ | months Δ ≥ 0 | mean w/o own top 5 / 10 | p raw / sel-adj |
|---|---|---|---|---|---|---|---|---|---|
| V0 live lock | **+0.024** | 52% | +2.90 / -3.12 | — | +0.00 / +0.00 | +0.000 / +0.000 | — | -0.368 / -0.671 | — |
| V1 BULLRUN (live fn, SL −1.2) | **-0.200** | 50% | +0.91 / -1.31 | **-0.224** [-0.84, +0.35] | -0.51 / +0.18 | -0.429 / -0.656 | 3/9 | -0.457 / -0.698 | 0.76 / 1.00 |
| V1b BULLRUN, flat −0.7 SL | **-0.206** | 37% | +0.85 / -0.82 | **-0.230** [-0.86, +0.43] | -0.34 / -0.08 | -0.435 / -0.662 | 3/9 | -0.441 / -0.556 | 0.75 / 1.00 |
| V2 BULLRUN, −3 stop | **-0.201** | 74% | +0.80 / -3.12 | **-0.225** [-0.73, +0.30] | -0.35 / -0.04 | -0.430 / -0.657 | 4/9 | -0.470 / -0.711 | 0.81 / 1.00 |
| V3 hybrid +1→+0.2, lock ≥+3 | **-0.157** | 74% | +0.86 / -3.12 | **-0.181** [-0.65, +0.30] | -0.28 / -0.04 | -0.383 / -0.607 | 4/9 | -0.467 / -0.655 | 0.77 / 1.00 |
| V4 hybrid +1.5→+0.2 | **-0.252** | 66% | +1.25 / -3.12 | **-0.276** [-0.69, +0.12] | -0.37 / -0.14 | -0.484 / -0.714 | 3/9 | -0.568 / -0.787 | 0.90 / 1.00 |
| V5 hybrid +1→+0.5 | **-0.171** | 74% | +0.84 / -3.12 | **-0.196** [-0.72, +0.31] | -0.22 / -0.17 | -0.416 / -0.661 | 4/9 | -0.457 / -0.606 | 0.77 / 1.00 |

### WIDE-HG — 125 fills, 93 days

| exit | mean % | WR | avg win / loss | Δ vs V0 [day CI] | Δ Jan–Apr / May–Sep | Δ w/o top 5 / 10 Δ | months Δ ≥ 0 | mean w/o own top 5 / 10 | p raw / sel-adj |
|---|---|---|---|---|---|---|---|---|---|
| V0 live lock | **+0.457** | 59% | +2.91 / -3.11 | — | +0.00 / +0.00 | +0.000 / +0.000 | — | +0.147 / -0.061 | — |
| V1 BULLRUN (live fn, SL −1.2) | **-0.101** | 58% | +0.79 / -1.31 | **-0.558** [-1.04, -0.09] | -0.77 / -0.32 | -0.715 / -0.885 | 2/9 | -0.289 / -0.457 | 0.99 / 1.00 |
| V1b BULLRUN, flat −0.7 SL | **-0.140** | 41% | +0.83 / -0.81 | **-0.597** [-1.12, -0.07] | -0.89 / -0.27 | -0.755 / -0.926 | 2/9 | -0.328 / -0.484 | 0.99 / 1.00 |
| V2 BULLRUN, −3 stop | **-0.077** | 78% | +0.76 / -3.11 | **-0.534** [-0.97, -0.07] | -0.39 / -0.69 | -0.690 / -0.859 | 1/9 | -0.272 / -0.448 | 0.99 / 1.00 |
| V3 hybrid +1→+0.2, lock ≥+3 | **+0.120** | 78% | +1.01 / -3.11 | **-0.337** [-0.70, +0.04] | -0.24 / -0.44 | -0.485 / -0.646 | 1/9 | -0.185 / -0.347 | 0.95 / 1.00 |
| V4 hybrid +1.5→+0.2 | **+0.037** | 71% | +1.31 / -3.11 | **-0.420** [-0.75, -0.08] | -0.37 / -0.48 | -0.572 / -0.735 | 1/9 | -0.271 / -0.440 | 0.99 / 1.00 |
| V5 hybrid +1→+0.5 | **+0.003** | 78% | +0.86 / -3.11 | **-0.453** [-0.90, +0.01] | -0.24 / -0.69 | -0.618 / -0.797 | 1/9 | -0.204 / -0.306 | 0.97 / 1.00 |

## 1b. Two-sided anatomy (Σ of Δ in % points; SAVED = the ORCA-type fill, CUT = a lock winner exited at the early floor)


**ALL**

| exit | SAVED: V0 lost, variant ≥ 0 (n · ΣΔ) | CUT: variant out at its early floor, V0 lock win ≥ +2 (n · ΣΔ) | CUT incl. +1.9 floor wins (n · ΣΔ) | rest (n · ΣΔ) | ΣΔ on the V0-stopped fills / on the rest | exits stop · early floor · lock/trail |
|---|---|---|---|---|---|---|
| V1 BULLRUN (live fn, SL −1.2) | 29 · **+93.1** | 25 · **-97.3** | 43 · -129.8 | 151 · -73.1 | +212.2 (95) / -289.5 | 97 · 72 · 36 |
| V1b BULLRUN, flat −0.7 SL | 24 · **+77.1** | 20 · **-74.4** | 32 · -96.1 | 161 · -101.7 | +240.6 (95) / -339.6 | 126 · 56 · 23 |
| V2 BULLRUN, −3 stop | 45 · **+144.3** | 38 · **-144.0** | 59 · -181.9 | 122 · -53.5 | +144.3 (95) / -197.5 | 50 · 104 · 51 |
| V3 hybrid +1→+0.2, lock ≥+3 | 45 · **+144.3** | 38 · **-144.0** | 59 · -181.9 | 122 · -37.9 | +144.3 (95) / -181.9 | 50 · 104 · 51 |
| V4 hybrid +1.5→+0.2 | 29 · **+93.1** | 31 · **-124.0** | 45 · -149.3 | 145 · -25.3 | +93.1 (95) / -149.3 | 66 · 74 · 65 |
| V5 hybrid +1→+0.5 | 45 · **+157.7** | 48 · **-177.0** | 75 · -217.7 | 112 · -40.7 | +157.7 (95) / -217.7 | 50 · 120 · 35 |

**STRONG**

| exit | SAVED: V0 lost, variant ≥ 0 (n · ΣΔ) | CUT: variant out at its early floor, V0 lock win ≥ +2 (n · ΣΔ) | CUT incl. +1.9 floor wins (n · ΣΔ) | rest (n · ΣΔ) | ΣΔ on the V0-stopped fills / on the rest | exits stop · early floor · lock/trail |
|---|---|---|---|---|---|---|
| V1 BULLRUN (live fn, SL −1.2) | 17 · **+54.4** | 15 · **-62.7** | 23 · -77.2 | 83 · -48.8 | +117.3 (52) / -174.5 | 52 · 40 · 23 |
| V1b BULLRUN, flat −0.7 SL | 14 · **+44.9** | 12 · **-47.0** | 18 · -57.8 | 89 · -76.2 | +132.4 (52) / -210.8 | 69 · 32 · 14 |
| V2 BULLRUN, −3 stop | 25 · **+80.1** | 20 · **-83.3** | 30 · -101.4 | 70 · -29.7 | +80.1 (52) / -113.0 | 27 · 55 · 33 |
| V3 hybrid +1→+0.2, lock ≥+3 | 25 · **+80.1** | 20 · **-83.3** | 30 · -101.4 | 70 · -18.1 | +80.1 (52) / -101.4 | 27 · 55 · 33 |
| V4 hybrid +1.5→+0.2 | 17 · **+54.5** | 18 · **-75.1** | 24 · -85.9 | 80 · -10.8 | +54.5 (52) / -85.9 | 35 · 41 · 39 |
| V5 hybrid +1→+0.5 | 25 · **+87.5** | 28 · **-113.4** | 39 · -129.9 | 62 · -16.5 | +87.5 (52) / -129.9 | 27 · 64 · 24 |

**NORMAL**

| exit | SAVED: V0 lost, variant ≥ 0 (n · ΣΔ) | CUT: variant out at its early floor, V0 lock win ≥ +2 (n · ΣΔ) | CUT incl. +1.9 floor wins (n · ΣΔ) | rest (n · ΣΔ) | ΣΔ on the V0-stopped fills / on the rest | exits stop · early floor · lock/trail |
|---|---|---|---|---|---|---|
| V1 BULLRUN (live fn, SL −1.2) | 12 · **+38.6** | 10 · **-34.6** | 20 · -52.6 | 68 · -24.3 | +94.8 (43) / -115.0 | 45 · 32 · 13 |
| V1b BULLRUN, flat −0.7 SL | 10 · **+32.3** | 8 · **-27.4** | 14 · -38.3 | 72 · -25.5 | +108.1 (43) / -128.8 | 57 · 24 · 9 |
| V2 BULLRUN, −3 stop | 20 · **+64.2** | 18 · **-60.7** | 29 · -80.5 | 52 · -23.8 | +64.2 (43) / -84.5 | 23 · 49 · 18 |
| V3 hybrid +1→+0.2, lock ≥+3 | 20 · **+64.2** | 18 · **-60.7** | 29 · -80.5 | 52 · -19.8 | +64.2 (43) / -80.5 | 23 · 49 · 18 |
| V4 hybrid +1.5→+0.2 | 12 · **+38.6** | 13 · **-49.0** | 21 · -63.5 | 65 · -14.5 | +38.6 (43) / -63.5 | 31 · 33 · 26 |
| V5 hybrid +1→+0.5 | 20 · **+70.2** | 20 · **-63.7** | 36 · -87.8 | 50 · -24.2 | +70.2 (43) / -87.8 | 23 · 56 · 11 |

**WIDE-HG**

| exit | SAVED: V0 lost, variant ≥ 0 (n · ΣΔ) | CUT: variant out at its early floor, V0 lock win ≥ +2 (n · ΣΔ) | CUT incl. +1.9 floor wins (n · ΣΔ) | rest (n · ΣΔ) | ΣΔ on the V0-stopped fills / on the rest | exits stop · early floor · lock/trail |
|---|---|---|---|---|---|---|
| V1 BULLRUN (live fn, SL −1.2) | 16 · **+51.1** | 17 · **-59.0** | 32 · -85.9 | 92 · -61.9 | +114.1 (51) / -183.9 | 53 · 48 · 24 |
| V1b BULLRUN, flat −0.7 SL | 11 · **+35.1** | 13 · **-41.0** | 23 · -58.9 | 101 · -68.7 | +127.1 (51) / -201.7 | 74 · 34 · 17 |
| V2 BULLRUN, −3 stop | 24 · **+76.7** | 26 · **-84.6** | 45 · -118.7 | 75 · -58.9 | +76.7 (51) / -143.5 | 27 · 69 · 29 |
| V3 hybrid +1→+0.2, lock ≥+3 | 24 · **+76.7** | 26 · **-84.6** | 45 · -118.9 | 75 · -34.3 | +76.7 (51) / -118.9 | 27 · 69 · 29 |
| V4 hybrid +1.5→+0.2 | 15 · **+48.0** | 21 · **-71.6** | 37 · -100.5 | 89 · -28.8 | +48.0 (51) / -100.5 | 36 · 52 · 37 |
| V5 hybrid +1→+0.5 | 24 · **+83.9** | 32 · **-107.3** | 54 · -140.6 | 69 · -33.3 | +83.9 (51) / -140.6 | 27 · 78 · 20 |

## 1c. Books from $3,000 at live sizing (each variant re-sequenced on its own exit times)

| exit | FRENZY book (strong 10× / normal 6×) final $ · max DD | strong-only book | normal-only book | WIDE-HG book (4×) | FRENZY + WIDE-HG together |
|---|---|---|---|---|---|
| V0 live lock | **$9,741 · -43% (205)** | $10,153 · -37% (115) | $2,881 · -23% (90) | $4,831 · -24% (125) | $13,474 · -54% (328) |
| V1 BULLRUN (live fn, SL −1.2) | **$3,273 · -26% (205)** | $4,137 · -23% (115) | $2,374 · -36% (90) | $2,631 · -18% (125) | $2,873 · -37% (330) |
| V1b BULLRUN, flat −0.7 SL | **$2,215 · -42% (205)** | $2,798 · -18% (115) | $2,375 · -36% (90) | $2,525 · -22% (125) | $1,865 · -51% (330) |
| V2 BULLRUN, −3 stop | **$4,904 · -32% (205)** | $6,289 · -30% (115) | $2,339 · -38% (90) | $2,678 · -25% (125) | $4,100 · -41% (329) |
| V3 hybrid +1→+0.2, lock ≥+3 | **$6,208 · -37% (205)** | $7,592 · -31% (115) | $2,453 · -33% (90) | $3,348 · -20% (125) | $6,171 · -42% (329) |
| V4 hybrid +1.5→+0.2 | **$4,391 · -54% (205)** | $6,023 · -37% (115) | $2,193 · -40% (90) | $3,015 · -24% (125) | $3,926 · -66% (328) |
| V5 hybrid +1→+0.5 | **$4,167 · -39% (205)** | $5,156 · -32% (115) | $2,424 · -31% (90) | $2,947 · -26% (125) | $4,068 · -53% (329) |

## 1d. Months (Δ vs V0, mean % per fill; the V0 column is V0's own mean)


**ALL**

| month | N | V0 mean | Δ V1 | Δ V1b | Δ V2 | Δ V3 | Δ V4 | Δ V5 |
|---|---|---|---|---|---|---|---|---|
| 01 | 25 | +1.87 | -1.88 | -2.09 | -1.64 | -1.12 | -1.30 | -1.05 |
| 02 | 13 | -0.38 | -0.51 | -0.30 | -0.53 | -0.39 | -0.05 | -0.21 |
| 03 | 45 | +0.27 | -0.26 | -0.29 | -0.02 | +0.06 | +0.11 | -0.18 |
| 04 | 22 | +0.01 | -0.24 | -0.11 | -0.26 | -0.46 | -0.75 | -0.30 |
| 05 | 30 | +0.42 | -0.25 | -0.55 | -0.19 | -0.04 | -0.41 | -0.09 |
| 06 | 12 | +1.01 | -0.91 | -1.08 | -0.47 | -0.15 | +0.02 | -0.95 |
| 07 | 19 | -0.97 | +0.39 | +0.66 | +0.20 | +0.15 | +0.07 | +0.29 |
| 08 | 21 | -0.15 | +0.21 | -0.16 | +0.25 | +0.01 | -0.02 | -0.09 |
| 09 | 18 | +1.23 | +0.01 | -0.38 | +0.21 | +0.16 | -0.02 | -0.33 |

**STRONG**

| month | N | V0 mean | Δ V1 | Δ V1b | Δ V2 | Δ V3 | Δ V4 | Δ V5 |
|---|---|---|---|---|---|---|---|---|
| 01 | 15 | +2.98 | -2.74 | -3.28 | -2.24 | -1.52 | -1.39 | -1.40 |
| 02 | 2 | -3.11 | +1.80 | +2.31 | +1.60 | +1.60 | +1.60 | +1.75 |
| 03 | 23 | +0.36 | -0.17 | -0.39 | +0.25 | +0.45 | +0.38 | +0.01 |
| 04 | 12 | +0.41 | -0.20 | -0.06 | -0.94 | -1.35 | -1.35 | -1.23 |
| 05 | 17 | +0.71 | -0.23 | -0.52 | +0.14 | +0.28 | +0.01 | +0.06 |
| 06 | 5 | +0.87 | -0.48 | -0.28 | +0.74 | +0.80 | +0.80 | -0.18 |
| 07 | 16 | -0.87 | +0.35 | +0.67 | +0.16 | +0.09 | +0.00 | +0.22 |
| 08 | 13 | +0.49 | -0.39 | -0.71 | -0.15 | -0.43 | -0.53 | -0.59 |
| 09 | 12 | +1.40 | -0.63 | -1.29 | -0.30 | -0.03 | -0.30 | -0.53 |

**WIDE-HG**

| month | N | V0 mean | Δ V1 | Δ V1b | Δ V2 | Δ V3 | Δ V4 | Δ V5 |
|---|---|---|---|---|---|---|---|---|
| 01 | 16 | +0.82 | -0.71 | -0.78 | -0.24 | -0.35 | -0.38 | -0.20 |
| 02 | 12 | +1.42 | -1.52 | -1.85 | -1.59 | -0.99 | -0.82 | -0.78 |
| 03 | 20 | +1.16 | -1.28 | -1.58 | -0.87 | -0.60 | -0.70 | -0.83 |
| 04 | 18 | -0.61 | +0.24 | +0.42 | +0.79 | +0.74 | +0.31 | +0.72 |
| 05 | 13 | +0.12 | -0.23 | -0.26 | -0.92 | -0.84 | -0.84 | -0.96 |
| 06 | 9 | -0.88 | +0.55 | +0.83 | -0.09 | -0.05 | -0.05 | -0.11 |
| 07 | 11 | +1.24 | -0.96 | -0.86 | -0.74 | -0.39 | -0.52 | -0.68 |
| 08 | 14 | +0.47 | -0.71 | -0.59 | -0.92 | -0.54 | -0.55 | -0.54 |
| 09 | 12 | +0.07 | -0.05 | -0.19 | -0.56 | -0.24 | -0.30 | -1.01 |

**How to read the months.**
- The hybrids' clear gains come in the months where the lock itself is negative: ALL Jul, Aug; WIDE-HG Apr.
- Their big losses come in the lock's strong months: ALL Jan V3 −1.12; strong Apr −1.35; WIDE-HG Feb / Mar.
- Mar and Sep (ALL) are small positives, so V3 is ≥ 0 in only 4 of 9 months.

## 2. Anecdotes: the live bot fills (not evidence)

**Setup:**
- Every bot `FRENZY_LONG` / `FRENZY_WIDE` fill in the B16, B17 and B18 exports.
- Priced on Binance aggTrades at the real fill second and price:
  - data.binance.vision daily files for 10-03 → 10-05;
  - the REST API for 10-06.
- Paper convention: 0.09 fees, 0 slippage.
- The operator's manual "FRENZY" trades are left out (hand exits).
- **Parity:** V0 reproduces ORCA 18:05 exactly (−3.01 at 12.07 min; live closed 18:17:12) and UMA (−3.00).
- Fills opened before the lock went live (SAND 05:05, AIN 11:15, MOVR, RLC) closed on the exit live used at the time. V0 shows what the lock would have done.

| fill | lev · strong | live result | peak ≤ 12 h | V0 lock | V1 BR | V1b | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ORCA 10-06 18:05 LONG** | 10× · strong | **−3.01** (pk +1.38) | +1.39 (data to 18:38) | −3.01 (12 min) | **+0.20** (1 min) | +0.20 | +0.20 | **+0.20** | −3.01 | +0.49 |
| ORCA 10-06 09:40 LONG | 6× · normal | −3.00 | **+22.4** (later) | −3.00 | −1.20 | −0.70 | −3.00 | −3.00 | −3.00 | −3.00 |
| UMA 10-06 10:10 LONG | 6× · normal | −3.00 | +1.87 | −3.00 | −1.21 | −0.71 | +0.20 | +0.20 | +0.20 | +0.48 |
| SAND 10-04 14:05 LONG | 6× · strong | −3.01 | +1.02 | −3.01 | +0.20 | −0.70 | +0.20 | +0.20 | −3.01 | +0.50 |
| SAND 10-04 05:05 LONG | 6× · strong | −3.00 (old exit) | +3.67 | **+1.99** | −1.21 | −0.71 | +0.62 | +1.99 | +1.99 | +1.99 |
| AIN 10-04 11:15 LONG | 6× · strong | +3.47 (old exit) | **+11.5** | **+2.70** | +0.20 | +0.20 | +0.20 | +0.20 | +0.20 | +0.50 |
| ENJ 10-03 03:36 LONG | 10× · n/a | −3.02 | +0.40 | −3.02 | −1.21 | −0.72 | −3.02 | −3.02 | −3.02 | −3.02 |
| AIN 10-03 23:25 WIDE | 4× | −3.01 | **+74.5** (later) | −3.01 | −1.21 | −0.70 | −3.01 | −3.01 | −3.01 | −3.01 |
| MOVR 10-05 09:15 WIDE | 4× | +3.00 (old TP) | +4.85 | +2.00 | +0.19 | +0.19 | +0.19 | +2.00 | +2.00 | +0.50 |
| RLC 10-05 12:00 WIDE | 4× | +3.01 (old TP) | +66.9 | **+5.37** | +5.48 | +5.48 | +5.48 | +5.37 | +5.37 | +5.37 |
| AIN 10-05 16:15 WIDE | 4× | −3.02 | +4.51 | −3.00 | −1.21 | −0.70 | +0.20 | +0.20 | +0.20 | +0.49 |
| FLUID 10-06 02:30 WIDE | 4× | −3.04 | +0.23 | −3.04 | −1.24 | −0.73 | −3.04 | −3.04 | −3.04 | −3.04 |
| **Σ (12 fills)** | | | | **−12.0** | −2.2 | +0.4 | −4.8 | **−1.7** | −8.1 | −1.8 |

**ORCA 18:05 in dollars.** Notional ≈ $7,069, so the result would have been:
- V0 −$213
- V1 / V2 / V3 +$14
- V4 −$213 (its peak, +1.38, never reached the +1.5 arm)
- V5 +$34

**These 12 fills are four days of one tape.** Three of the losers later ran very far: ORCA 09:40 to +22 %, AIN 10-03 to +75 %, RLC to +67 %. None of the variants catches that kind of move either.

Over 205 year fills, the same rule loses (§1). Recent pain is the reason to look, not evidence to switch.

## 3. Method

**Cohort:**
- `wrev/cohort_plus.pkl`: the same signals as `reports/FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv`, i.e. the engine's own `frenzy_walk` fresh_on bars.
- Filters: FRENZY ∧ priced ∧ U2 < 1 ∧ live-eligible.
- Sequenced by `frenzy_engine_cohort_report.sequence`. Result: **205 fills on 133 days**, the FRENZY_LONG_LEVERAGE_TRADEABLE cohort (+0.387 reproduced). 199 are tick-priced and 6 use the 1m fallback (flagged); tick-only Δ is shown below.
- **Strong flag:** `services.frenzy.frenzy_adx_delta > 0 ∧ frenzy_di_spread > 0` on `closed[-300:]` 5m bars. 115 strong, 90 normal.
- **WIDE-HG:** sleeve WIDE ∧ FRENZY_GREEN_BAR ∧ above_streak > 12 (the engine's `frenzy_wide_hold_green_block`) ∧ priced ∧ U2 < 1 ∧ live-eligible. That gives 125 signals, all tick-priced, sequenced alone; none are dropped. The live rule's year count was 124.

**Pricing:**
- The unchanged cohort pricer (`frenzy_engine_cohort_price.prints`): first print ≥ signal close + 12 s, 1 % dislocation guard, net = gross − 0.09.
- Lines are checked against the prior-print peak. Crossing-print fills; 0.10 slippage on every exit; 12 h cap.
- **Parity:** V0 = the cohort's LOCK2 to 1.8e-15, exit times identical, entry price to 6e-13.
- **V1 parity:** V1 lines were checked against `services.trading_engine._bullrun_exit_for` with the live config on sampled prints. Max error 0.
- **The live BULLRUN stop is not −0.7 on FRENZY fills.** `sl_atr_multiplier` 1.5 widens it to −1.2 (the `sl_atr_widen_floor_pct`) on 329 of 330 fills, because FRENZY ATRs are ≥ 0.8 %. V1b shows the flat −0.7 the operator quoted.
- ATR = the cohort's signal-bar ATR, used as the entry ATR.

**Tick-only Δ (ALL):** V1 −0.41 · V1b −0.50 · V2 −0.29 · V3 −0.21 · V4 −0.32 · V5 −0.31. These are the same signs as the full table.

**Replica vs live polling:**
- Live polls price, while the replica sees every print. So live could miss a brief dip under +0.2 and cut fewer winners than the replica does.
- On the 59 lock wins V3 cuts, the low between the break-even exit and the lock's own exit has a median of **−0.88 %**. 64 % went below −0.5 and 44 % below −1.0.
- These are real dips, not wicks. Polling would catch them, so the CUT side is not an artefact.

**Books:**
- $3,000 start; strong 0.5 → 10×, normal 0.32 → 6×, notional 1.21 × L / 6 × equity; WIDE 0.2 → 4× at 0.94 × equity.
- 2 slots per sleeve, pair flat, 3 per pair per day, size at entry, compounding in exit order, liquidation cap.
- Each variant is re-sequenced on its own exit times (the per-trade tables use the V0 fill set).

**Not tested:**
- Funding.
- Slippage above 0.10 on the break-even exits. Paper charges 0, which shifts every variant by +0.10 and leaves the Δ unchanged.
- The bars' rolling ATR for the BR trail (live uses entry ATR too).
- Re-entry after a break-even exit. The re-entry family is separate (FRENZY_REDO_REENTRY).

**Scripts (scratchpad, research only):**
- `…/scratchpad/lbr/price.py`: pricing plus V1 parity.
- `an.py`: stats, bootstrap, sign-flip null, books.
- `tables.py`
- `anec.py`: live fills on aggTrades.
- Outputs: `priced.pkl`, `splits.pkl`, `res.pkl`, `anec.pkl`.
