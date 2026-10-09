# FRENZY_LITE entry signs — does the SKLUSDT loser's look separate winners from losers? (2026-10-08)

Research only: no code, config or bot changes. Scripts: `scripts/study_lite_signs_main.py` (pre-registration in its docstring) and `scripts/study_lite_signs_fetch_skl.py` (two SKL kline calls, used weight 4). Per-fill data is in `reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv` and every test row is in `reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08_tests.csv`. `validate_against_master.py` reported ALL CHECKS PASS before this run.

## Verdict
**None of the three signs passes the block bar. No filter.** Each sign makes LITE look a bit worse, and they all point the same way, but no flagged group is reliably below zero. The best result, at most 0.82, falls short of the 0.95 needed for P(mean<0). When the whole family is shuffled by day, the strongest result has a family-wise p of 0.47, so it is about what chance alone would produce. When bearish days are added back (the 724 robustness cohort), the volume sign (S3) flips to positive. **Proposed watch line, observe-only:** "bought ≥ 3 % below the 30-min high" (S1 dd30 ≥ 3). It is the only sign that is negative in both halves of the year and in every leave-one-month-out run. Its effect is too small and too uncertain to arm.

## Cohort and ruler
- **MAIN**: the LITE study cohort (`lite_streak/kept.pkl` K[12]) with bearish-day fills removed. It has 548 fills over 214 days, priced on ticks at today's exit (+3 / −3 / 12 h, live ruler of 8 s plus 0.10 % slip, bot fees; `tp34 walk_c fix3`). This reproduces the +0.277 %/fill from FRENZY_TP3_VS_TP4_TICKS.
- MAIN baseline: WR 54.6 %, avg **+0.277** [+0.007, +0.536], Σ +152 %. **Breakeven WR is 49.9 %** (avg win +3.00, avg loss −2.99).
- **ROBUST**: all 724 fills over 244 days. WR 54.3 %, avg +0.257.
- Parity checks: the above-streak join with the year walk matches 100 %, and my vol_trend recompute equals the year-walk value (max difference 3e-14). That value is the engine's `frenzy_vol_trend`: Σ volume·close of the last 12 closed 5m bars ÷ Σ of the 12 before them.

## SKL's values and where it sits (percentile within MAIN)
| sign | SKL | percentile |
|---|---|---|
| S1 dd30, below the 30-min high | 3.93 % | 88th |
| S1 dd60 | 3.93 % | 82nd |
| S1 burst60, largest 15-min up-burst in the last hour | 9.0 % | 95th |
| S2 above_streak | 12 (the minimum) | shared by 78 % of MAIN |
| S2 vs VWAP | +3.83 % | 81st (top tercile) |
| S3 vol_trend | 3.88 (stamp 3.882, rebuilt 3.882) | 98th |
| S3 1h volume ÷ 24h-average hour | 2.82 | 78th |

SKL is an extreme case on S1 and S3. On S2, being at the minimum streak is normal, since 78 % of fills share it.

## Results on MAIN at today's exit (N · WR · avg % · day-clustered CI · P(mean<0) · Δ vs rest · null p / family p · H1 / H2)
| test | N | WR | avg | CI | P<0 | Δ | p / fam | H1 / H2 | LOMO Δ range |
|---|---|---|---|---|---|---|---|---|---|
| S1 dd30 ≥ 2 | 245 | 52.7 | +0.163 | [−0.20, +0.55] | 0.19 | −0.21 | 0.43 / 1.00 | +0.04 / +0.26 | −0.38…−0.06 |
| **S1 dd30 ≥ 3** | 120 | 48.3 | **−0.097** | [−0.63, +0.43] | 0.64 | −0.48 | 0.13 / 0.89 | **−0.10 / −0.09** | **−0.67…−0.31** |
| S1 dd30 ≥ 4 | 63 | 47.6 | −0.143 | [−0.88, +0.55] | 0.65 | −0.48 | 0.26 / 0.98 | −0.51 / +0.34 | −0.80…−0.26 |
| S1 dd60 ≥ 2 / 3 / 4 | 300 / 163 / 97 | 54.0 / 49.7 / 47.4 | +0.244 / −0.016 / −0.153 | — | 0.08 / 0.52 / 0.69 | −0.08 / −0.42 / −0.52 | ≥ 0.13 / ≥ 0.88 | mixed | all ≤ +0.06 |
| S1 burst60 ≥ 3 | 330 | 53.0 | +0.183 | [−0.15, +0.54] | 0.14 | −0.24 | 0.39 / 1.00 | −0.00 / +0.30 | −0.37…−0.09 |
| S2 streak == 12 | 427 | 53.9 | +0.235 | [−0.07, +0.53] | 0.07 | −0.19 | 0.56 / 1.00 | +0.29 / +0.20 | −0.39…−0.06 |
| S2 streak 12–14 | 441 | 53.7 | +0.227 | [−0.07, +0.52] | 0.06 | −0.26 | 0.44 / 1.00 | +0.28 / +0.19 | |
| S2 vs-VWAP T1 / T2 / T3 (cuts 1.28 / 2.73 %) | 183 / 182 / 183 | 55.2 / 53.3 / 55.2 | +0.320 / +0.197 / +0.314 | | ≤ 0.22 | +0.06 / −0.12 / +0.06 | ≥ 0.69 | | |
| S3 vol_trend > 1 (sign) | 293 | 52.9 | +0.182 | [−0.19, +0.57] | 0.18 | −0.21 | 0.44 / 1.00 | +0.17 / +0.19 | −0.34…−0.10 |
| S3 vol_trend ≥ 2 | 93 | 48.4 | −0.054 | [−0.67, +0.55] | 0.56 | −0.40 | 0.26 / 0.98 | −0.35 / +0.17 | −0.48…−0.15 |
| S3 vol_trend ≥ 3 | 28 | 46.4 | −0.154 | [−1.23, +1.01] | 0.60 | −0.45 | 0.46 / 1.00 | −0.74 / +0.43 | |
| S3 1h/24h > 1 (sign) | 383 | 53.0 | +0.183 | [−0.12, +0.48] | 0.13 | −0.31 | 0.29 / 0.99 | +0.22 / +0.16 | |
| S3 1h/24h ≥ 2 | 193 | 50.8 | +0.051 | [−0.39, +0.48] | 0.39 | −0.35 | 0.23 / 0.96 | +0.08 / +0.03 | |
| S3 1h/24h ≥ 3 | 106 | 45.3 | **−0.283** | [−0.87, +0.29] | **0.82** | −0.70 | **0.04** / 0.47 | −0.27 / −0.29 | −0.87…−0.46 |
| S1p ∧ S2p | 73 | 45.2 | −0.286 | [−1.00, +0.44] | 0.77 | −0.65 | 0.12 / 0.79 | −0.46 / −0.15 | |
| S1p ∧ S3p | 32 | 43.8 | −0.372 | [−1.42, +0.69] | 0.74 | −0.69 | 0.24 / 0.97 | −0.90 / +0.51 | |
| S2p ∧ S3p | 83 | 49.4 | +0.012 | | 0.49 | −0.31 | 0.41 / 1.00 | −0.24 / +0.18 | |
| S1p ∧ S2p ∧ S3p (SKL's cell) | 26 | 46.2 | −0.227 | [−1.39, +0.94] | 0.62 | −0.53 | 0.41 / 1.00 | −0.85 / +0.51 | |

Primary flags: S1p = dd30 ≥ 3, S2p = streak == 12, S3p = vol_trend ≥ 2. Pair and day concentration is never an issue: the top pair or day carries ≤ 14 % of any flagged group's loss.

Buckets on MAIN:
- **dd30**: 0–1 +0.39 · 1–2 +0.36 · 2–3 +0.41 · 3–4 −0.05 · ≥ 4 −0.14. The pattern is monotone in shape, with a step at 3 %. The 3–4 bucket's halves are +0.56 / −0.43, so even the step is not stable.
- **streak**: 12 +0.24 · 13–14 +0.00 (N 14) · 15–24 +0.11 · **≥ 25 +0.93 (N 49, CI +0.07…+1.69)**. Long stretches whose first signal was refused for vol ≥ 100× or age < 2 h did best. This is a side observation, not a test.
- **vol_trend**: < 0.7 +0.56 · 0.7–1 +0.19 · 1–2 +0.29 · 2–3 −0.01 · ≥ 3 −0.15.
- **1h/24h quartiles**: +0.55 · +0.20 · +0.42 · −0.06. This is not monotone.

## Locked expectancy bar (block candidates, MAIN)
No test passes. Every flagged group with WR below 49.9 % fails the confidence leg (P(mean<0) ≤ 0.82, needed ≥ 0.95). All of them pass N ≥ 15, ≥ 8 days and concentration. The closest is **S3 1h/24h ≥ 3**: avg −0.283, P 0.82, single-test null p 0.04, family p 0.47.
- That candidate **would not have caught SKL**, whose ratio was 2.82.
- In ROBUST it is only +0.023, and its quartile is not monotone.
- If it were armed anyway, the in-sample gain is +30 %-points over the year, about +0.05 %/fill. After the 30–50 % haircut that leaves about +0.03, on a sleeve designed on this same year (in-sample twice). This is noise-sized.

## Robustness (724, bearish days included)
The S1 effects shrink: dd30 ≥ 3 goes to +0.039 (H1 −0.07 / H2 +0.14). S3 vol_trend ≥ 2 flips to **+0.259** and ≥ 3 to +0.52. S2 is unchanged. The volume sign is therefore not stable across cohort definitions. S1 holds direction but not size.

## Stops: flush vs gradual (MAIN, 246 stops, all with ticks)
A "flush" stop is one where the largest 60-s drop in the 5 minutes before the stop print is ≥ 2 %. SKL qualifies: its 21:47 1m bar fell 2.6 %, and the stop printed at 21:49.

| stop type | share | median minutes, entry → stop | stopped ≤ 15 min after entry | back to entry price within 60 min after the stop |
|---|---|---|---|---|
| flush | **31 %** (77) | 16 | 47 % | **56 %** (median 17 min) |
| gradual | 69 % (169) | 62 | 14 % | 28 % (median 31 min) |

- Flush stops are twice as common after a mini-pump: 52 % of dd30 ≥ 3 stops are flushes, against 24 % for the rest.
- Flush stops cluster in the first 15 minutes after entry.
- More than half of flush stops return to the entry price within the hour.
- These are descriptive figures only. A stop-width counterfactual was not run and must follow the live-stopped / two-sided rule.

## Blind spots (what this could NOT test)
- The features come from 5m bars. The 1m shape of the mini-pump (SKL's 3-minute burst) is only approximated by 5m highs, and burst60 ≥ 3 % fires on 60 % of fills, so it is too common to separate anything.
- This is the study cohort, not an engine replay. About 21 shortlist misses are excluded.
- The year that designed LITE is the year being tested.
- The 1h/24h ratio uses Binance quote volume. The engine does not stamp it.
- There are 5 live LITE fills to date (SKL, MOVR, MET, HEMI, SAND). With that N, no live read is possible.

## Watch line proposal (observe-only, not armed)
Pre-registered, with the threshold frozen now: **LITE_OFF30H3**. A LITE fill is flagged when the signal close is ≥ 3 % below the max 5m high of the 6 bars ending at the signal bar.
- Track it on live fills.
- **Review** when ≥ 30 flagged fills fall on ≥ 15 days.
- Propose a block only if the flagged group meets the locked expectancy bar on live fills alone: WR < LITE breakeven, P(mean<0) ≥ 0.95 day-clustered, no pair or day ≥ 50 % of the loss.
- **Retire** it if the flagged avg is ≥ the unflagged avg at that review.

Engine stamping would need a new entry column (for example `entry_frenzy_off30_high_pct`). That is a D11/D12 change for the operator to approve.
