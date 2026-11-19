# Flatbase: enter at the signal (S4) or wait for confirmation, on TODAY's filters (2026-11-19)

Research only. Rows: OOS_REAL_FLATBASE_2026-11-17 episodes (route TRADE), priced on today's exit (+5 / −10 / 72 h, 0.10 % slip, real funding). Filters: NEAR_BASE (≤ 2.09 % above the pre-pump base → skip; on the signal path it is read at the signal close) and HIGHER_HIGH (confirm path). Base = the 5m close 30 min before the spike bar's close (engine `c[on − 6]`; the OOS `spike` column is the spike bar's close), from the 5m cache (1,219 of 1,223 episodes). Review fix: a first version read the base one 5m bar late (25 min); corrected here — the verdict did not change, two side-claims flipped (see notes). One position / 4 slots not modelled. Script `scripts/flatbase_s4_vs_confirm_nb.py`.

| block | policy | trades | WR | avg / trade (day-block 95 % CI) | Σ points |
|---|---|---|---|---|---|
| H2024 | wait for confirm (today) | 50 | 68% | +0.05 (-1.88 … +2.39) | +2 |
| H2024 | enter at signal | 79 | 66% | -0.07 (-1.67 … +1.72) | -5 |
| DESIGN | wait for confirm (today) | 570 | 74% | +0.90 (+0.30 … +1.53) | +514 |
| DESIGN | enter at signal | 875 | 70% | +0.32 (-0.13 … +0.72) | +283 |
| FWD | wait for confirm (today) | 4 | 100% | +4.94 (+nan … +nan) | +20 |
| FWD | enter at signal | 4 | 75% | +1.13 (+nan … +nan) | +5 |
| ALL | wait for confirm (today) | 624 | 74% | +0.86 (+0.27 … +1.46) | +536 |
| ALL | enter at signal | 958 | 70% | +0.29 (-0.13 … +0.71) | +282 |

| half (spike) | confirm: N · avg · Σ | signal: N · avg · Σ |
|---|---|---|
| 24-H1 | 2 · +2.57 · +5 | 3 · -2.35 · -7 |
| 24-H2 | 58 · +0.21 · +12 | 90 · +0.01 · +1 |
| 25-H1 | 59 · +1.54 · +91 | 95 · -0.24 · -23 |
| 25-H2 | 157 · -0.17 · -26 | 241 · +0.32 · +76 |
| 26-H1 | 202 · +0.69 · +140 | 277 · +0.82 · +226 |
| 26-H2 | 146 · +2.15 · +314 | 252 · +0.03 · +9 |

### Where the difference comes from (signal-entry trades, all blocks, by what the confirm path did with them)

| what confirm did | N | signal-entry avg | signal-entry Σ | confirm Σ on the same episodes |
|---|---|---|---|---|
| confirmed and traded | 624 | +2.19 | +1367 | +536 |
| confirmed but NEAR_BASE at confirm | 107 | +2.13 | +227 | +0 |
| never confirmed (SKIP_CONFIRM) | 147 | -8.15 | -1199 | +0 |
| higher high (SKIP_HH) | 80 | -1.42 | -114 | +0 |
| confirm-only (NEAR_BASE at the signal, not at confirm) | 0 | — | 0 | +0 |

## What it means
- **Waiting wins overall:** +0.86 / trade on 624 trades (Σ +536) vs +0.29 on 958 (Σ +282). Close to the DL 577 independent rebuild (619 · 74 % · +0.97).
- **Why:** 147 episodes never confirm. Shorted at the signal they lose −8.15 each (−1,199 points), and the confirmation step avoids all of them. On episodes that do confirm, the signal entry earns more (+2.19 vs +0.86), because it gets a higher price. That gain does not cover the unconfirmed losses.
- **Not uniform:** the signal entry was better in 25-H2 and 26-H1; the confirmation in 24-H2, 25-H1 and 26-H2. The small 2024 holdout is a tie (+0.05 vs −0.07, CIs ±2). Real fills (DL 428 study: +0.09 vs +3.51) favour the confirmation.
- **Verdict:** keep the confirmation step.
- **Base-offset fix (review):** with the base read one bar late, the numbers were 605 · +0.88 (confirm) vs 935 · +0.33 (signal); the 2024 holdout leaned to the signal (−0.37 vs +0.16) and 24-H2 favoured the signal. With the correct base both flip to the confirm side or a tie. This was a bug in this research script only; the engine reads the base correctly.
