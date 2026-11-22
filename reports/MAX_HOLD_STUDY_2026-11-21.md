# MAX HOLD STUDY — every sleeve, funding charged (2026-11-21)

**Question (operator):** "Should we review the max duration per sleeve, considering funding? (24 h in EXH). Run the study, show me your recommendation."

**Verdict: keep every max hold as it is.** Shorter caps lose on EXH, DIST and FLATBASE. Longer caps are statistical ties. Funding does not decide the cap: on EXH / DIST at every cap and FB up to 48 h, a fill still open at the cap is worth +0.2 … +1.0 % more held than closed (FB at 60 h, HT and WILLY lean to closing — the caps there are ties anyway), and the extra funding costs only 0.01 … 0.09 %. Funding moves no ALL / HOLDOUT result by more than 0.02 % (DIST DESIGN 6 h: 0.03). Nothing to ship → nothing waits for the one-week freeze.

Full tables: `reports/MAX_HOLD_STUDY_2026-11-21_tables.md` · chart `reports/MAX_HOLD_STUDY_2026-11-21_chart.png` · scripts `scripts/MAX_HOLD_STUDY_2026-11-21_{bt,willy,report}.py` · data `reports/MAX_HOLD_STUDY_2026-11-21_data/`.

## Parity (the walk reproduces earlier studies exactly)
- At today's caps: DL 616 EXH 4,307/4,307 · DIST 1,547/1,547 · FB 545/545 (max |Δ| 0, exit type 100 %); WILLY_HT = the DL 608 walk on 362 fills; WILLY = the HT_THEORY_EXIT sim on 630 fills.
- Fills exiting by TP / stop before a shorter cap show Δ = 0 exactly in every block. No backtest fill runs out of data before 48 h / 120 h.

## Summary
% of notional per fill at one size, net of fees, slippage and real Binance funding. Δ = cap − today's cap over all fills. CI = 95 % day-block bootstrap.

| sleeve | today | best cap | Δ ALL (CI) | HOLDOUT Δ (CI) | funding effect | real fills | verdict |
|---|---|---|---|---|---|---|---|
| EXHAUSTION_SHORT | 24 h | 48 h | +0.012 (−0.011…+0.035) | +0.009 (−0.019…+0.038) | −0.001 | 0 master · 8 batch, Δ ≡ 0 | **Keep 24 h** (48 h a tie; negative in 26-H2). 6 h −0.087 (−0.148…−0.023) |
| DISTRIBUTION_SHORT | 24 h | 48 h | +0.038 (−0.027…+0.103) | −0.023 (−0.088…+0.037) | 0.000 | 0 · 6, Δ ≡ 0 | **Keep 24 h** (48 h gain is old data only; flips in HOLDOUT). 20 h −0.044 in HOLDOUT |
| FLATBASE_SHORT | 72 h | 60 h | +0.006 (−0.052…+0.067) | +0.014 (−0.050…+0.078) | 0.000 | 17 · 3, Δ ≡ 0 | **Keep 72 h** (tie). 24 h −0.166 · 48 h −0.059 · 120 h −0.030 |
| FRENZY_WILLY_HT | 24 h | 8 h | +0.028 (−0.051…+0.116), 7 fills | +0.013 (−0.072…+0.102) | 0.000 (HT receives) | 12 · 12 | **Keep 24 h** — no fill in any block is open at 24 h |
| FRENZY_WILLY | 90 min | 45 min | +0.023 (−0.020…+0.067) | +0.031 (−0.013…+0.076) | 0.000 | 23 master (45 min −0.136, one fill) | **Keep 90 min** (CI incl. 0, master disagrees) |
| SURGE_LONG | 240 min | — | — | — | — | longest real hold 25 min | Keep (never binds) |
| MOMENTUM (OFF) | 180 min | — | — | — | — | — | Not reviewed |

**Why real fills show no change:** under today's exits no real EXH / DIST / FB fill comes near any cap (EXH ≤ 1.9 h closed, ENSO 2.5 h; DIST ≤ 3.0 h; FB ≤ 17.8 h). The batch EXH fills held 5–15 h (ZKC, the two RAREs, ENSO ~24 h) were TP-10 era; under TP 7 (DL 609) they all exit within ~2.5 h. The backtest carries the evidence; there are no master EXH / DIST fills.

## EXH (24 h) — backtest ALL (N 4,227 · WR 46.8 % · +0.500 %/fill at 24 h)
| cap | Δ ALL (CI) | HOLDOUT | DESIGN | DCR %/day |
|---|---|---|---|---|
| 6 h | −0.087 (−0.148…−0.023) | −0.016 | −0.197 | 0.678 |
| 8 h | −0.070 (−0.123…−0.019) | −0.023 | −0.144 | 0.697 |
| 12 h | −0.039 (−0.076…−0.003) | −0.007 | −0.089 | 0.745 |
| 16 h | −0.030 (−0.056…−0.003) | −0.012 | −0.057 | 0.756 |
| 20 h | −0.011 (−0.028…+0.006) | −0.002 | −0.025 | 0.793 |
| **24 h** | — | — | — | **0.813** |
| 30 h | −0.002 | −0.010 | +0.009 | 0.807 |
| 36 h | −0.005 | −0.005 | −0.006 | 0.802 |
| 48 h | +0.012 (−0.011…+0.035) | +0.009 | +0.016 | 0.831 |
- 6 h cuts 438 winners (−1,708 pts) to save 287 stops (+1,380). 48 h: of 163 time exits, 70 reach TP (+234), 39 stop deeper (−187), 54 drift (+4); ARPA 35 % of the sum; 5 of 7 half-years positive but −0.020 in 2026-H2.
- Close vs hold: open at 6 h (21 % of fills) +1.82 closed vs +2.24 held (49 % later TP, 32 % stop; extra funding −0.07); open at 20 h +2.35 vs +2.56 (extra funding −0.02).

## DIST (24 h) — backtest (N 1,545 · WR 42.3 % · +0.909 %/fill)
6 h −0.145 · 12 h −0.082 · 16 h −0.070 · 20 h −0.053 (HOLDOUT −0.044, CI −0.087…−0.005) · 30 h −0.015 · 36 h −0.006 · 48 h +0.038 (HOLDOUT −0.023; DESIGN +0.146). 48 h by half-year +0.175, +0.176, +0.012, +0.215, then −0.052 (25-H2), −0.030 (26-H1), +0.040 (26-H2) → the gain is old. Close vs hold at 6 h: +3.35 vs +4.00.

## FLATBASE (72 h) — backtest (WR 75.3 % · +1.122 %/fill)
24 h −0.166 (P(better) 0.04) · 36 h −0.094 · 48 h −0.059 · 60 h +0.006 · 96 h 0.000 · 120 h −0.030. Close vs hold: at 24 h −0.53 vs +0.42 (hold wins); 48 h −0.98 vs −0.27; 60 h −0.92 vs −1.01 (edge exhausted ≈ 60 h). Heaviest funding of any sleeve (−0.71 % on 16–24 h holds), yet no cap's Δ moves > 0.006 from it.

## WILLY_HT (24 h)
No real (62, any filter) or backtest (300) fill is open at 24 h; longest hold 16.8 h. 4 h −0.032 (HOLDOUT −0.069) · 6 h +0.013 · 8 h +0.028 (7 fills) · 12 h −0.016. Batch 4 h +0.45 = one fill (EVAA 11-20 23:20).

## WILLY (90 min)
45 min +0.023 ALL / +0.031 HOLDOUT / −0.014 DESIGN; master −0.136 (one fill, ZK 10-09 late TP). Fills open at 45 min: −0.86 held vs −0.48 closed (33 fills). Least-bad candidate if the operator wants a change after the freeze — but its revert gate needs ~250 WILLY fills (months), which argues against it.

## Theory
- **EXH:** the fade of a crowded pump happens within hours (median winner 2.5 h, p90 11 h) or the squeeze stops it fast (median loser 1.2 h); late fills are profitable drifters. Funding grows linearly but stays below the remaining expectancy.
- **DIST:** fast stops and slow winners (128 time exits average +3.99 %); whether the drift continues past 24 h depends on the regime, which turned in 2025-H2.
- **FB:** slow reversion to the base → a long cap is right; past ≈ 60 h only base-stuck fills remain.
- **HT:** fast TP / stop, funding received → the cap never binds.
- **WILLY:** a quick scalp; fills alive at 45 min have missed the late TP (slight, unproven edge to cutting).

## Re-open only with
Master EXH / DIST fills holding ≥ 12 h under TP 7 · an FB fill reaching 48 h · an HT fill open at 16 h · any TP / stop change on a sleeve (the cap must be re-walked under the new exit, DL 406).

## Blind spots
Real fills cannot test the cap (Δ ≡ 0 except HT 4 h and WILLY 45 min, one fill each) · time-exit slippage ≥ 0.10 % (0.02 / 0.30 % sensitivity changes no verdict) · FB (and 1 EXH / 4 DIST fills) on 5m bars · 80 EXH / 2 DIST / 6 FB backtest rows dropped for a missing funding month (N 4,227 / 1,545 / 539, not the parity counts 4,307 / 1,547 / 545) · SURGE / MOMENTUM not walked · DCR on an assumed $10k one size · 5–8 caps per grid; no positive ALL / HOLDOUT cell clears its CI (DIST 48 h DESIGN does, +0.146, CI +0.010…+0.291, but it flips in HOLDOUT) — no family-wise correction needed to reject.
