# SURGE_LONG trailing stop — should the trail distance have an ATR-size cap? (2026-10-15)

Question (operator, watching live STRK SURGE_LONG: peak +3.54 %, dip to +1.80 %, trail line +1.59 %): is "peak − 1× ATR" OK,
or should the trail distance be capped based on ATR size?

## Plain-English summary

- **Keep today's trail. None of the 10 cap variants passes the pre-registered bar, and most lose money.**
- **What a cap does:** it pulls the stop line closer to the peak. You win more often: with a 1 % cap, 55 % of fills close green instead of 36 %. But each win is much smaller. Over two years, every fixed cap (1.0 / 1.25 / 1.5 / 2.0 %) made LESS money than today: −0.07 to −0.17 % per fill.
- **Where the money is lost:** in the 2026-H2 bull run, where SURGE earns all its profit. Today's exit made +0.91 % per fill there. A 1.25 % cap made +0.26. That is −0.65 % per fill, and the 95 % range is entirely below zero.
  - The big runners pay for everything: PLAY +17.1 %, ONG +13.1 %, ACE +8.9 %, TLM +7.7 %, ONE +8.3 %.
  - Under a 1.25 % cap they would have closed at +0.7, +3.6, +2.8, +0.5 and +1.3 %.
  - This matches the house rule "capping a winner chops the runner".
- **"Tighten after a big run" also loses.** Halving the trail once the peak reaches +3 % or +5 % costs −0.13 and −0.09 % per fill. On fills that reached +3 % (the STRK situation), today's exit averaged +3.29 %. The tightened version averaged +2.79 %, and a 1.25 % cap averaged +1.84 %. Most trades that pull back 2 % from a +3.5 % peak go on to make new highs. The tight line sells those trades too early.
- **Tighten only the very wide-ATR pairs: close, but no.** The best of these keeps 1× ATR when ATR is below 2.5 % and uses 0.75× above it. It earns +0.03 % per fill (P 0.77). It wins in the 2026-H1 bear market and loses in the H2 bull, so it depends on the market regime. It fails the gate in 2025 (P 0.895 < 0.90) and in 2026 (P 0.73).
- **STRK today:** its ATR is about 1.95 % (+3.54 peak − 1.59 line). The ATR-tier variants would not change it. A 1.25 % cap would have put its line near +2.29 %, so the dip to +1.80 % would have closed it there. The year data says that trade-off loses money on average. The trade is still open and not yet in any export, so it is described only.

## Method

- **Pre-registration:** `reports/SURGE_TRAIL_CAP_PREREG.txt`, written before any variant was priced and refusing overwrite.
  - 10 cells: CAP 1.0 / 1.25 / 1.5 / 2.0, four ATR-tier cells (A 2.0 / 2.5 × k_hi 0.6 / 0.75), and TIGHT at P 3 / 5.
  - Multiplicity is 10, read as best-of-10.
- **Cohort:**
  - SURGE_LONG CLOSED fills from `reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv` with EMA13−EMA50 gap ≥ 0 (today's armed filter, DL 287).
  - 617 fills with the 3 seeds pooled.
  - 556 were walked on aggTrade ticks: 220 in 2025 (ticks from `reports/backtest_cache_2025/`, the yr6a cache), 181 in 2026-H1 and 155 in 2026-H2.
  - 61 fills are 2025 non-top-50 pairs with no ticks. They use the 1m fallback, are flagged and are not in the primary read.
- **Walker:**
  - `scripts/surge_bearrun_review.py`, reused: `sec_path` reads every price-changing print, and `run_exit` takes the first print whose net P&L is at or below the line, with peak starting at 0 and a 240-min hold.
  - Each fill starts from the replay's own entry time and price.
  - Today's line was checked against `services.trading_engine._bullrun_exit_for(no_lock=True)` at 28 (ATR, peak) points: exact match.
- **Parity (gate ≥ 85 % within ±0.10):**
  - Tick-walked fills: **556 / 556 = 100 %**, median |diff| 0.0005. Passed.
  - 1m-fallback fills: 31 % (19 / 61). That is why they are excluded from the primary read and shown as a sensitivity line.
- **Exit-CF rule:** all 281 never-armed fills (peak < +1.0) show Δ = 0 under every cell (asserted). Every cell's line is at or above today's, so Δ > 0 = SAVED and Δ < 0 = KILLED.
- **Slip:** every exit is a market order (−0.17 on the exit, +0.03 on SURGE entries), so the slip is the same per fill under every rule. Δ does not change with slip. Levels are shown both ways.
- **Statistics:** day-clustered bootstrap with 5,000 draws.

## Gate (frozen; tick sample)

| cell | Δ all | P(Δ>0) all | Δ 2025 | P 2025 | Δ 2026 | P 2026 | LOMO sign-stable | top day / pair share | CANDIDATE |
|---|---|---|---|---|---|---|---|---|---|
| CAP_1.00 | −0.173 | 0.05 | +0.007 | 0.55 | −0.291 | 0.03 | (negative) | n/a (ΣΔ<0) | no |
| CAP_1.25 | −0.173 | 0.05 | +0.072 | 0.84 | −0.334 | 0.01 | (negative) | n/a | no |
| CAP_1.50 | −0.115 | 0.10 | +0.022 | 0.66 | −0.205 | 0.08 | (negative) | n/a | no |
| CAP_2.00 | −0.065 | 0.17 | +0.009 | 0.65 | −0.114 | 0.16 | (negative) | n/a | no |
| TIER_A2.0_K0.60 | −0.029 | 0.36 | −0.031 | 0.34 | −0.027 | 0.44 | no | n/a | no |
| TIER_A2.0_K0.75 | +0.014 | 0.62 | +0.032 | **0.90** | +0.002 | 0.53 | no | 0.79 / **0.97** | no |
| TIER_A2.5_K0.60 | −0.003 | 0.51 | +0.025 | 0.83 | −0.022 | 0.44 | no | n/a | no |
| TIER_A2.5_K0.75 | +0.032 | 0.77 | +0.019 | 0.895 | +0.040 | 0.73 | yes | 0.34 / 0.42 | no |
| TIGHT_P3 | −0.131 | 0.05 | −0.098 | 0.19 | −0.152 | 0.11 | (negative) | n/a | no |
| TIGHT_P5 | −0.086 | 0.06 | −0.059 | 0.26 | −0.103 | 0.11 | (negative) | n/a | no |

**Candidates: none.** TIER_A2.0_K0.75 passes 2025 only. One PAIR carries 97 % of its gain, and it is ≈ 0 in 2026. TIER_A2.5_K0.75 fails confidence in both years.

## Per period (tick; avg % per fill, net of fees, zero slip; live-slip level = avg − 0.14)

| period | cell | N | days | WR % | avg % | avg live-slip | Δ vs today | Δ 95 % CI | P(Δ>0) | saved n | killed n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 BULL | TODAY | 220 | 57 | 31.4 | −0.006 | −0.146 | — | — | — | — | — |
| 2025 BULL | CAP_1.25 | 220 | 57 | 43.6 | +0.066 | −0.074 | +0.072 | −0.10…+0.21 | 0.84 | 89 | 22 |
| 2025 BULL | TIER_A2.5_K0.75 | 220 | 57 | 31.8 | +0.013 | −0.127 | +0.019 | −0.01…+0.05 | 0.895 | 12 | 1 |
| 2026-H1 BEAR | TODAY | 181 | 45 | 32.6 | −0.004 | −0.144 | — | — | — | — | — |
| 2026-H1 BEAR | CAP_1.25 | 181 | 45 | 44.2 | −0.070 | −0.210 | −0.066 | −0.39…+0.23 | 0.37 | 64 | 31 |
| 2026-H1 BEAR | TIER_A2.0_K0.75 | 181 | 45 | 35.4 | +0.183 | +0.043 | +0.187 | +0.10…+0.28 | 1.00 | 58 | 5 |
| 2026-H1 BEAR | TIER_A2.5_K0.75 | 181 | 45 | 35.4 | +0.126 | −0.014 | +0.130 | +0.06…+0.20 | 1.00 | 32 | 1 |
| 2026-H2 BULL | TODAY | 155 | 29 | 45.8 | **+0.909** | +0.769 | — | — | — | — | — |
| 2026-H2 BULL | CAP_1.00 | 155 | 29 | 65.2 | +0.296 | +0.156 | **−0.613** | −1.21…−0.09 | 0.01 | 56 | 48 |
| 2026-H2 BULL | CAP_1.25 | 155 | 29 | 53.5 | +0.263 | +0.123 | **−0.646** | −1.23…−0.13 | 0.01 | 59 | 45 |
| 2026-H2 BULL | CAP_1.50 | 155 | 29 | 49.0 | +0.446 | +0.306 | −0.463 | −1.00…+0.01 | 0.03 | 68 | 34 |
| 2026-H2 BULL | CAP_2.00 | 155 | 29 | 49.7 | +0.663 | +0.523 | −0.246 | −0.72…+0.10 | 0.10 | 43 | 15 |
| 2026-H2 BULL | TIER_A2.0_K0.75 | 155 | 29 | 46.5 | +0.695 | +0.555 | −0.214 | −0.66…+0.12 | 0.13 | 42 | 14 |
| 2026-H2 BULL | TIER_A2.5_K0.75 | 155 | 29 | 47.7 | +0.844 | +0.704 | −0.065 | −0.39…+0.17 | 0.34 | 32 | 6 |
| 2026-H2 BULL | TIGHT_P3 | 155 | 29 | 47.1 | +0.590 | +0.450 | −0.319 | −0.84…+0.08 | 0.06 | 34 | 27 |

All 11 rows × 5 periods are in `reports/SURGE_TRAIL_CAP_cells.csv` and `reports/SURGE_TRAIL_CAP_tables.md`.

**Two-sided ledger (all periods; % points per seed):**

| cell | saved n | saved points | killed n | killed points |
|---|---|---|---|---|
| CAP_1.00 | 197 | +64.6 | 113 | −96.7 |
| CAP_1.25 | 212 | +56.9 | 98 | −89.0 |
| CAP_1.50 | 231 | +45.7 | 71 | −67.1 |
| CAP_2.00 | 119 | +26.0 | 30 | −38.1 |
| TIGHT_P3 | 85 | +30.0 | 61 | −54.2 |
| TIER_A2.5_K0.75 | 76 | +17.2 | 8 | −11.4 |

Caps save many small givebacks and kill fewer but much larger runners.

**Top-10 runners today (unique trades), exit % under each rule:**

| pair | opened (UTC) | ATR | peak | TODAY | CAP 1.0 | CAP 1.25 | CAP 1.5 | CAP 2.0 | TIER 2.5/0.75 | TIGHT P3 | TIGHT P5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAY | 2026-04-15 19:36 | 4.29 | 21.38 | 17.08 | 0.93 | 0.68 | 0.44 | 2.34 | 18.10 | 2.19 | 4.29 |
| ONG | 2026-08-21 01:20 | 4.92 | 17.98 | 13.07 | 2.73 | 3.63 | 3.38 | 2.88 | 12.05 | 2.42 | 7.77 |
| MOVE | 2025-06-25 09:10 | 1.79 | 12.33 | 10.52 | 6.33 | 11.06 | 10.82 | 10.52 | 10.52 | 3.36 | 5.43 |
| ACE | 2026-08-17 02:56 | 2.00 | 10.87 | 8.86 | 3.07 | 2.83 | 4.74 | 8.86 | 8.86 | 3.07 | 5.24 |
| USELESS | 2026-09-11 12:41 | 1.87 | 10.14 | 8.27 | 5.84 | 5.59 | 5.34 | 8.27 | 8.27 | 5.91 | 5.91 |
| ONE | 2026-09-21 00:16 | 3.00 | 11.27 | 8.26 | 0.42 | 1.26 | 1.01 | 0.51 | 0.26 | 5.97 | 5.97 |
| EVAA | 2026-06-15 15:50 | 3.93 | 12.02 | 8.09 | 0.63 | 0.39 | 3.26 | 3.72 | 9.05 | 3.76 | 3.76 |
| TLM | 2026-07-03 20:21 | 4.77 | 12.45 | 7.65 | 0.70 | 0.45 | 0.20 | 0.31 | 8.84 | 0.74 | 7.51 |
| TST | 2026-05-04 04:05 | 2.95 | 10.24 | 7.28 | 0.60 | 0.35 | 0.10 | 3.97 | 8.01 | 4.49 | 4.49 |
| ARC | 2026-02-08 08:26 | 2.08 | 8.19 | 6.10 | 3.79 | 4.92 | 4.68 | 6.19 | 6.10 | 3.74 | 5.14 |

**Armed fills today:**

- 310 of 556 fills armed. Their average peak was +3.70 % and their average exit +1.41 %, so they gave back 2.29 points on average.
- 36 % of armed fills ended at or below 0.
- The giveback is real, but it is what pays for the runners.

**Big-run fills (peak ≥ +3 %, the STRK case):**

- 147 fills, 71 unique trades. Their average peak was +5.83 %.
- Average exit: today +3.29 · CAP 1.0 +1.66 · CAP 1.25 +1.84 · CAP 1.5 +2.31 · TIGHT P3 +2.79.

**Δ by ATR bucket (armed fills):**

- Every cap loses in every ATR bucket except the 2.5–3 % bucket (N 27), where it is flat.
- The ≥ 3 % ATR bucket loses most: CAP 1.25 −0.39, TIGHT P3 −0.67. Wide-ATR pairs are exactly where the big runners come from.

**Concentration:**

- n/a for every cell with negative ΣΔ.
- TIER_A2.0_K0.75: top pair 97 % of ΣΔ, which fails.
- TIER_A2.5_K0.75: top day 34 %, top pair 42 %.

**Leave-one-month-out (best pooled cell, TIER_A2.5_K0.75):** every month left out stays positive, from +0.007 to +0.066. It still fails the per-year confidence gate.

**Sensitivity, with the 61 flagged 1m 2025 fills added:**

- The 2025 caps look better: CAP 1.25 +0.127, P 0.95.
- Those fills reproduce the replay only 31 % of the time. The 1m O-H-L-C path gives wide trails a pessimistic exit, which favours tighter lines. That is an artefact, so this line is not evidence.
- Even with them, the pooled two-year Δ stays negative for every CAP (−0.04…−0.12) and +0.006…+0.028 for the tiers. No candidate.

## Recommendation

- **No change to the SURGE_LONG trail. No Scout line:** no cell met the frozen gate.
- **Mechanism:**
  - SURGE earns its money from a few runners on high-ATR pairs, with WR around 36 %.
  - A cap or a post-run tightening converts those runners into small wins.
  - The higher WR is real but does not pay for the lost tail.
- **The tier idea is worth remembering only as a regime note.** Tightening wide-ATR pairs helped in the 2026-H1 bear market (+0.13…+0.19, P 1.0) and hurt in the H2 bull. That is a regime switch, which would need window-unit evidence under the WINDOW-UNITS rule. It is not a trade rule.
- **Re-read with real fills:** at the DL 202/203 review (first 15 filled SURGE_LONG triggers), re-walk the same fills under CAP_1.25 and TIER_A2.5_K0.75 alongside the lock check. This is an observation only and moves no gate.

## Could not test / limits

- **Live / master:**
  - Master `MASTER_POOL_stacked.csv` has **0 SURGE_LONG fills**; its one SURGE row is the SAND SURGE_SHORT of 10-02.
  - Live exports (export_glob, 4 files to 14:50 UTC 10-15) hold no SURGE_LONG row.
  - The open STRK SURGE_LONG is not exported yet and is described only from the operator's numbers.
  - There is no real-fill check of this exit.
- **2025 non-top-50 pairs (61 fills)** have no ticks. The 1m fallback reproduces the replay poorly, so they are excluded from the primary read.
- **Second-order effects are not modelled.** An earlier exit frees the slot and the pair sooner, so a re-entry or another SURGE fill could follow. The replay's entries are fixed.
- **Seeds:** the 3 seeds are near-copies of the same trades. The day-clustered bootstrap handles the dependence; unique-trade counts are given where relevant.
- **ATR:** the trail uses ENTRY ATR, as live does. A rolling-ATR trail was not tested.
- **Slip:** modelled as a uniform per-fill shift, so it does not change Δ. Slip that depends on exit speed in fast dumps was not tested.

## Exact commands

```
venv/bin/python scripts/study_surge_trail_cap_prereg.py          # pre-registration (refuses overwrite) → reports/SURGE_TRAIL_CAP_PREREG.txt
venv/bin/python scripts/study_surge_trail_cap_walk.py --parity   # TODAY vs replay pct_new (tick 556/556 = 100 % within ±0.10)
venv/bin/python scripts/study_surge_trail_cap_walk.py            # walk TODAY + 10 cells → reports/SURGE_TRAIL_CAP_fills.csv (~60 s, cache-only)
venv/bin/python scripts/study_surge_trail_cap_report.py          # tables + gate → reports/SURGE_TRAIL_CAP_tables.md, reports/SURGE_TRAIL_CAP_cells.csv
```
