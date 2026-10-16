# LITE from first principles: what a LITE trade is, and which of its assumptions hold (2026-10-15)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing was committed, the bot API was not called, and **no Binance call** was made (cache only).
- Pre-registered in `reports/LITE_THEORY_PREREG.txt` (frozen 20:24 UTC with input hashes) **before** any bar, signal or variant was priced. One correction was made after the freeze; it is declared in A0.
- This study has not been reviewed, so it cannot support arming or shipping anything.
- Every number is at 1× (net % of notional). The exit is today's: +3 resting maker TP, −3 stop, 12 h cap. Live slip is applied: −0.20 on every non-TP exit and +0.036 % on entry.

## Plain-English summary

**What a LITE trade is.**
1. A coin has a big volume spike (a FRENZY flag).
2. Between 2 and 16¾ hours later, the price has closed above the spike's average price (its "VWAP") for 12 five-minute candles in a row, while volume has calmed down.
3. LITE buys the first such moment, but only if that candle is red (a small dip).

The idea is that buyers defended the average, so the pump was "accepted" and a dip-buy catches the next leg.

**What we did.**
- We rebuilt every FRENZY episode on 398 coins (Jan 2025 → Oct 2026) using the engine's own code: 6,841 episodes, 1.85 M five-minute bars.
- Our rebuilt LITE signals reproduce **96.6 %** of the backtest's LITE trades at the same coin and candle (gate: 85 %).
- We priced 1.15 M possible entry moments and 53 pre-registered changes to the definition.

**Which assumptions hold, and which don't:**

| # | Assumption | Verdict | Plain meaning |
|---|---|---|---|
| A0 | The LITE selection adds something | **Partly.** | Better than "buy any time it is above the average": +0.47 to +0.53 %/trade in both years. But in 2025 it is **no better than a random entry** in the same episodes. In 2026 it is about +0.2 to +0.3 better. **Most of LITE's result is the market regime:** 2025 −0.31 %/trade, 2026 +0.15. |
| A1 | "Held above VWAP" predicts the next leg | **Weakly, and not at 12.** | Above-average bars beat below-average bars by about 0.1–0.25 %/trade. Nothing changes at 12 closes: 6–11 closes look the same. Very long streaks (36+) are the worst bucket in 2025. There is no clean "the longer it held, the better". |
| A2 | Buy the red candle | **Not stable.** | In 2025, green first-candles did *better* than red. In 2026, worse. Entering on any colour, or waiting for a later red, does not help. |
| A3 | Need 12 closes | **Arbitrary, and nothing beats it.** | 6, 24 and 36 all lose to 12 in 2026, and on master. |
| A4 | Window 2–16.75 h | **The last hours are the weakest.** | 12–16.75 h is negative in both years (−0.44 / −0.21). Stopping at 12 h improves both years, but fails 2026 confidence (P 0.80) and master (−11.8). Not shippable. |
| A5 | Volume must have cooled (< 100×) | **No useful cut.** | < 20× or < 50× hurt 2026. The lead's own test (keep vol ≥ 20) fails the same way: it helps master, hurts 2025/2026. |
| A6 | Buy near support (small stretch above VWAP) | **No.** | Results do not improve as entries get closer to the average: 0–1 % and 1–2 % are as bad as 3–5 %. Caps on the stretch fail 2026. The lead's test (stretch ≥ 1 %) also fails. A **limit buy at VWAP + 0.5 %** is strong in 2025 (+0.18 %/trade, P 0.99) but flat in 2026 (P 0.65) and costs master −11 (it misses quick winners). |
| A7 | A close below VWAP means the thesis is broken | **Refuted as an exit.** | All 10 VWAP-based exits or stops lose in 2025, and none passes in 2026. Trades that dip under the average often recover. Cutting them costs more than it saves. |
| A8 | Spike features: size, volume, distance from the high, stretch number | **No robust cut.** | Each looks good in one year and bad in the other. The 3rd+ stretch in an episode is weak in both years (−0.48 / −0.21). Dropping it fails 2026 confidence (0.75) and master (−2.8). |
| A9 | Price broke above the spike high = an accepted pump | **The most theory-consistent near-miss.** | When the coin never printed above its spike-candle high, LITE lost in both years (−0.47 / −0.31) and on master (5 fills, −0.72). It is small (170 of 1,983 fills) and fails the 2026 confidence bar (P 0.76). |

**What this implies.**
- LITE's story ("acceptance above VWAP → next leg") is only partly true in the data:
  - holding above the average beats being below it;
  - **but none of the specific choices are where the edge is:** 12 closes, a red candle, buying near the average, exiting under it.
- The sleeve's P&L follows the regime: entries in the same episodes lose in 2025 and win in 2026 whatever rule picks them.
- This agrees with the 10 earlier LITE studies: there is no separator, because the definition's details don't carry the edge.

**Recommendation (the operator decides):**
- **Change nothing on LITE's definition.** No cell of 53 passes the gates. Expected false passes by luck were about 0.5.
- **One new observe-only Scout line is worth adding: `LITE_NO_NEWHI`.** It marks a LITE entry where no high above the spike-candle high printed between the spike and the signal.
  - It is the only cell that is direction-consistent in every block: 2025, 2026 and master. It also has a theory behind it.
  - **Tracker / revert gate (pre-committed):** as a filter it ships only if the observed live + master cohort reaches N ≥ 15 on ≥ 8 days with avg < 0 at day-bootstrap P ≥ 0.95. Its WR must also be below the sleeve's 51.5 % breakeven, and the 2026 backtest re-read must hold P ≥ 0.90.
  - The engine already has the data: the episode's spike-bar high, and the max high since the spike, both from the frenzy_walk window. It would need one new stamp column, e.g. `entry_frenzy_new_high` (bool) or `entry_frenzy_vs_spike_high_pct`, plus the D11/D12 plumbing.
- **Watch list only** (strong in 2025, fails 2026 and/or master; no new line):
  - stop LITE entries after 12 h;
  - skip the 3rd+ stretch;
  - a VWAP + 0.5 % limit entry.

### Every headline two ways
- **Today's rebuilt LITE signals (RH tick ruler):**
  - 2025: 1,275 signals, WR 46.4 %, **−0.31 %/trade** (Σ −391 %-pts).
  - 2026: 708 resolved, WR 54 %, **+0.14 %/trade** (Σ +99).
  - Master 20: **+0.21 %/trade**, WR 55 %.
- **A0, all flagged episodes (no look-ahead):**
  - 2025: LITE −0.31 vs random bar −0.32, i.e. the same.
  - 2026: LITE +0.15 vs random −0.19, i.e. **+0.33 better**.
- **A0, paired against bars after the first LITE signal in the same episode:**
  - vs any bar: 2025 −0.09 (P 0.14), 2026 +0.21 (P 0.96).
  - vs any later above-VWAP bar: +0.46 / +0.66 (P 1.00).
- **A9 `NO_NEWHI` fills:**
  - 2025: 125 fills, −0.47 avg (−0.16 below today's average).
  - 2026: 45 fills, −0.31 avg (−0.45 below).
  - Master: 5 fills, −0.72.
  - Dropping them gives +58.8 / +13.9 %-pts (P 0.94 / 0.76). After a 30–50 % haircut that is about +0.02 %/trade overall.

## Parity / validation
- **Episode rebuild:** `services.frenzy.frenzy_walk` on the engine's 1,499-bar window. The normal hour comes from `normal_hour_usd` on 743 completed 1h bars, refreshed hourly; the engine refreshes it every 6–8 h, which is a declared difference.
- **Signal emulator:** asserted equal to `services.frenzy.frenzy_lite_status` on all 1,852,059 bars.
- **Replay fills reproduced: 1,405 / 1,454 = 96.6 %** (2025 97.9 %, 2026 94.9 %).
  - On matched bars: streak equal on 98.3 %, spike equal on 97.4 %.
  - Median |Δ vs_vwap| 0.0002, vol_mult ratio 0.999.
  - The bearish flag was wrong on 3 / 1,447 fills.
- **Walker:** X.eval_v reproduces the frozen LITE_PROTECT `rm_base_pct` on all 1,472 cohort fills (max |Δ| 4e-16). This study's walker equals X.eval_v on every priced signal (asserted).
- **Entry-price rule.** This study's E is the next 1m open × (1 + 0.036 %). On matched fills it sits about 0.04–0.07 % above the replay's E, and the exit reason agrees on 94 % (2025) and 85 % (2026).
  - 2025 matched average: −0.36 vs the cohort's −0.28.
  - 2026: +0.13 vs +0.16.
  - The same rule is used for every set, so the comparisons are fair.
- **`validate_against_master.py`: ALL CHECKS PASS (exit 0).**

## Could NOT test / caveats
- **Signal level, not portfolio.** Slots, pair-day cap, the WILLY hold and chunk warm-up are ignored. 2,106 rebuilt signals vs 1,454 replay fills.
- **Missing 1m data.** 14 % of bulk bars and 647 / 11,663 signal bars have no 1m entry bar (gaps in the 2025 cache). They are excluded on both sides of every comparison.
- **A0 correction after the freeze.**
  - The pre-registered paired baseline used the WHOLE 2–16.75 h window. That window includes bars *before* the LITE signal, and those bars are conditioned on the future: the episode later rose 12 closes above the average.
  - That look-ahead made "every bar" look like +0.14 in 2025 episodes. Both versions are shown; the conclusions use the corrected one and the unpaired all-episode table.
- **A2_WAITRED:** the next red bar was searched only within 6 bars (as registered).
- **A6 limit entries:** the TP trigger keeps the taker-entry formula, and only the +0.027 fee saving is added, which is slightly conservative. An unfilled limit counts as 0.
- **A7 VWAP exits:** booked at the 5m or 1m close that broke the average, plus −0.20 slip.
- **Master block:**
  - Replacement cells use rebuilt signals in the 2026-10-06 → 10-12 window, labelled REBUILT. They are not real fills.
  - Filter cells: 15 / 20 master fills could be scored on rebuilt spike features.
  - A7 C5/C1 cells on master: only fills with a rebuilt VWAP path count.
- **Live:** ZK, DUSK, EPIC and DEEP are listed in the bucket tables and are never gated.
- **Multiple testing:** 53 cells. Several cells pass 2025 at P ≥ 0.90 because any filter that trades less "wins" in a losing year. That is why the per-signal average change (in `gates.csv`) matters.
- **No order-book or open-interest data.**

## Exact commands
```
JOBS=8 venv/bin/python scripts/study_lite_theory_episodes.py      # bars.pkl (≈8 min)
venv/bin/python scripts/study_lite_theory_signals.py              # emulator parity + replay-fill parity
venv/bin/python scripts/study_lite_theory_prereg.py               # frozen (refuses overwrite)
JOBS=8 venv/bin/python scripts/study_lite_theory_price.py         # walker parity, R1 bulk, RH cells, A6/A7 (≈5 min)
venv/bin/python scripts/study_lite_theory_report.py               # reports/lite_theory/tables.md + gates.csv + a0.csv
venv/bin/python scripts/validate_against_master.py
```

---

# LITE THEORY — tables (generated by scripts/study_lite_theory_report.py)

Today's LITE signals (rebuilt, non-bearish): 2106 · RH-resolved 1983 · R1-resolved 1980 · RH source {'ticks': 1088, '1m': 881, 'ticks+1m': 17}
- 2025: N 1275 · RH avg -0.307 · WR 46.4% · Σ -391.0 · R1 avg -0.305
- 2026: N 831 · RH avg +0.140 · WR 46.0% · Σ +99.0 · R1 avg +0.146
- MASTER 20 fills (LS): avg +0.207 · WR 55.0% · Σ +4.15
- LIVE 4: ZKUSDT -2.93, DUSKUSDT -2.54, EPICUSDT -3.43, DEEPUSDT +2.80

## A0 — does the LITE selection add anything over entering anywhere in the same episodes? (R1 1m ruler, LS slip)

Paired per LITE signal: Y(LITE) − mean Y of the same episode's bars in the set (window 2–16.75 h, not in state, vol < 100, non-bearish).

| year | baseline set | N signals | LITE avg | baseline avg | Δ/signal | P(Δ>0) day-boot | 95% CI |
|---|---|---|---|---|---|---|---|
| 2025 | (i) every bar [as pre-registered: whole window — LOOK-AHEAD biased] | 1275 | -0.305 | +0.135 | -0.439 | 0.00 | [-0.595, -0.278] |
| 2026 | (i) every bar [as pre-registered: whole window — LOOK-AHEAD biased] | 705 | +0.146 | +0.251 | -0.105 | 0.18 | [-0.337, +0.122] |
| all | (i) every bar [as pre-registered: whole window — LOOK-AHEAD biased] | 1980 | -0.144 | +0.176 | -0.320 | 0.00 | [-0.449, -0.186] |
| 2025 | (i) every bar [corrected: bars at/after the episode's first LITE signal] | 1275 | -0.305 | -0.213 | -0.091 | 0.14 | [-0.257, +0.078] |
| 2026 | (i) every bar [corrected: bars at/after the episode's first LITE signal] | 705 | +0.146 | -0.061 | +0.207 | 0.96 | [-0.025, +0.442] |
| all | (i) every bar [corrected: bars at/after the episode's first LITE signal] | 1980 | -0.144 | -0.159 | +0.015 | 0.58 | [-0.115, +0.152] |
| 2025 | (ii) every close ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 1275 | -0.305 | -0.525 | +0.220 | 1.00 | [+0.072, +0.366] |
| 2026 | (ii) every close ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 705 | +0.146 | -0.293 | +0.439 | 1.00 | [+0.229, +0.643] |
| all | (ii) every close ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 1980 | -0.144 | -0.442 | +0.298 | 1.00 | [+0.178, +0.422] |
| 2025 | (ii) every close ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 1275 | -0.305 | -0.764 | +0.459 | 1.00 | [+0.308, +0.608] |
| 2026 | (ii) every close ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 705 | +0.146 | -0.511 | +0.656 | 1.00 | [+0.440, +0.866] |
| all | (ii) every close ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 1980 | -0.144 | -0.674 | +0.529 | 1.00 | [+0.410, +0.654] |
| 2025 | (iii) every RED bar ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 1275 | -0.305 | -0.513 | +0.208 | 1.00 | [+0.060, +0.355] |
| 2026 | (iii) every RED bar ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 705 | +0.146 | -0.276 | +0.421 | 1.00 | [+0.211, +0.619] |
| all | (iii) every RED bar ≥ VWAP [as pre-registered: whole window — LOOK-AHEAD biased] | 1980 | -0.144 | -0.428 | +0.284 | 1.00 | [+0.163, +0.407] |
| 2025 | (iii) every RED bar ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 1275 | -0.305 | -0.708 | +0.403 | 1.00 | [+0.255, +0.550] |
| 2026 | (iii) every RED bar ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 705 | +0.146 | -0.446 | +0.592 | 1.00 | [+0.382, +0.798] |
| all | (iii) every RED bar ≥ VWAP [corrected: bars at/after the episode's first LITE signal] | 1980 | -0.144 | -0.615 | +0.470 | 1.00 | [+0.353, +0.593] |

Unpaired, ALL flagged episodes (not only those holding a LITE signal), same window — mean Y per entry, TP-first share:

| year | set | N bars | episodes | avg Y | TP-first % |
|---|---|---|---|---|---|
| 2025 | (i) every bar | 430,034 | 3,550 | -0.324 | 43.2 |
| 2025 | (ii) every close ≥ VWAP | 139,627 | 2,704 | -0.320 | 45.0 |
| 2025 | (iii) every RED bar ≥ VWAP | 65,347 | 2,552 | -0.312 | 45.0 |
| 2025 | **today's LITE signals** | 1,275 | 995 | -0.305 | 45.5 |
| 2026 | (i) every bar | 222,424 | 2,197 | -0.185 | 46.0 |
| 2026 | (ii) every close ≥ VWAP | 69,419 | 1,538 | -0.055 | 49.8 |
| 2026 | (iii) every RED bar ≥ VWAP | 31,373 | 1,401 | -0.047 | 49.8 |
| 2026 | **today's LITE signals** | 705 | 566 | +0.146 | 53.2 |

## A1 — does 'held above VWAP' predict continuation? (all bars, window 2–16.75 h, R1)

**by above_streak**

| year | bucket | N bars | episodes | avg Y (+3/−3/12h) | day-boot 95% CI | TP-first % | fwd 1h % | fwd 4h % | fwd 12h % |
|---|---|---|---|---|---|---|---|---|---|
| 2025 | 0 | 290,407 | 3,216 | -0.326 | [-0.421, -0.237] | 42.3 | -0.04 | -0.11 | -0.37 |
| 2025 | 1–5 | 27,254 | 2,284 | -0.237 | [-0.354, -0.127] | 46.2 | -0.09 | -0.08 | -0.17 |
| 2025 | 6–11 | 17,046 | 1,887 | -0.173 | [-0.290, -0.057] | 47.3 | -0.15 | -0.06 | +0.04 |
| 2025 | 12–23 | 22,462 | 1,724 | -0.253 | [-0.359, -0.140] | 46.0 | -0.08 | +0.05 | +0.19 |
| 2025 | 24–35 | 15,251 | 1,320 | -0.161 | [-0.299, -0.012] | 47.4 | +0.14 | +0.12 | +0.37 |
| 2025 | 36+ | 57,614 | 1,293 | -0.472 | [-0.941, +0.028] | 42.6 | -0.06 | +0.30 | -0.24 |
| 2026 | 0 | 153,005 | 2,054 | -0.244 | [-0.343, -0.148] | 44.3 | -0.02 | +0.02 | +0.37 |
| 2026 | 1–5 | 14,111 | 1,328 | -0.038 | [-0.167, +0.081] | 49.9 | +0.12 | +0.65 | +1.33 |
| 2026 | 6–11 | 9,211 | 1,065 | -0.012 | [-0.158, +0.126] | 50.5 | +0.23 | +0.71 | +1.39 |
| 2026 | 12–23 | 12,593 | 999 | +0.006 | [-0.137, +0.156] | 51.0 | +0.18 | +0.70 | +0.97 |
| 2026 | 24–35 | 8,742 | 754 | -0.121 | [-0.284, +0.037] | 49.1 | +0.05 | +0.32 | +0.48 |
| 2026 | 36+ | 24,762 | 689 | -0.087 | [-0.284, +0.107] | 49.1 | +0.06 | -0.07 | -0.12 |

**by above_share (time above VWAP since the spike)**

| year | bucket | N bars | episodes | avg Y (+3/−3/12h) | day-boot 95% CI | TP-first % | fwd 1h % | fwd 4h % | fwd 12h % |
|---|---|---|---|---|---|---|---|---|---|
| 2025 | <50% | 297,823 | 2,894 | -0.317 | [-0.408, -0.233] | 42.5 | -0.04 | -0.09 | -0.27 |
| 2025 | 50–70% | 55,943 | 1,527 | -0.292 | [-0.418, -0.174] | 45.3 | -0.12 | -0.07 | -0.38 |
| 2025 | 70–85% | 34,914 | 1,058 | -0.220 | [-0.380, -0.063] | 46.8 | -0.02 | -0.01 | -0.35 |
| 2025 | ≥85% | 41,354 | 760 | -0.511 | [-1.069, +0.162] | 42.0 | -0.04 | +0.42 | -0.05 |
| 2026 | <50% | 152,841 | 1,826 | -0.218 | [-0.319, -0.114] | 44.7 | +0.01 | +0.15 | +0.70 |
| 2026 | 50–70% | 32,709 | 944 | -0.184 | [-0.338, -0.031] | 47.7 | +0.03 | +0.15 | -0.01 |
| 2026 | 70–85% | 19,800 | 625 | +0.032 | [-0.128, +0.200] | 51.5 | +0.13 | +0.18 | -0.23 |
| 2026 | ≥85% | 17,074 | 355 | -0.136 | [-0.335, +0.061] | 48.3 | -0.04 | -0.17 | +0.00 |

## Bucket tables on today's signals (RH, LS) — three blocks: backtest per year · master (stamps) · live

**A4 hours since spike**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| 2–4 | 338 · 48% · -0.206 | 186 · 57% · +0.319 | 4 · +1.456 | EPIC -3.43 |
| 4–8 | 382 · 45% · -0.419 | 214 · 56% · +0.246 | 3 · +0.919 | — |
| 8–12 | 278 · 49% · -0.147 | 142 · 54% · +0.156 | 4 · -1.654 | — |
| 12–16.75 | 277 · 44% · -0.435 | 166 · 48% · -0.212 | 4 · +1.450 | ZK -2.93, DUSK -2.54, DEEP +2.80 |
| >16.75 | 0 | 0 | 0 · — | — |
| (master unscored) | | | 5 | |

**A5 vol_mult**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| <20 | 624 · 50% · -0.084 | 290 · 53% · +0.064 | 7 · -2.320 | ZK -2.93, EPIC -3.43, DEEP +2.80 |
| 20–50 | 370 · 44% · -0.455 | 240 · 53% · +0.099 | 4 · +1.453 | — |
| 50–100 | 281 · 42% · -0.606 | 178 · 57% · +0.318 | 9 · +1.620 | DUSK -2.54 |
| (master unscored) | | | 0 | |

**A6 entry stretch above VWAP %**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| 0–1 | 369 · 46% · -0.300 | 171 · 55% · +0.189 | 3 · -3.223 | ZK -2.93 |
| 1–2 | 370 · 46% · -0.350 | 183 · 56% · +0.254 | 8 · +0.675 | DEEP +2.80 |
| 2–3 | 195 · 50% · -0.098 | 125 · 50% · -0.077 | 2 · -0.100 | — |
| 3–5 | 204 · 45% · -0.415 | 145 · 52% · +0.007 | 6 · +0.937 | DUSK -2.54, EPIC -3.43 |
| 5+ | 137 · 46% · -0.345 | 84 · 57% · +0.345 | 1 · +3.002 | — |
| (master unscored) | | | 0 | |

**A8 spike run_pct %**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| <10 | 235 · 46% · -0.244 | 136 · 52% · -0.004 | 2 · -3.229 | DEEP +2.80 |
| 10–20 | 546 · 50% · -0.126 | 297 · 53% · +0.073 | 9 · -1.137 | ZK -2.93, DUSK -2.54 |
| ≥20 | 494 · 43% · -0.536 | 275 · 56% · +0.283 | 9 · +2.316 | EPIC -3.43 |
| (master unscored) | | | 0 | |

**A8 spike volume multiple**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| <50 | 1167 · 47% · -0.284 | 579 · 53% · +0.070 | 13 · +0.135 | — |
| 50–100 | 92 · 41% · -0.525 | 98 · 59% · +0.471 | 1 · +3.002 | — |
| ≥100 | 16 · 38% · -0.725 | 31 · 58% · +0.400 | 1 · +3.001 | — |
| (master unscored) | | | 5 | |

**A8 close vs spike-bar high %**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| <−8 | 36 · 39% · -0.779 | 22 · 64% · +0.754 | 0 · — | — |
| −8…−3 | 150 · 45% · -0.466 | 66 · 48% · -0.194 | 1 · -3.250 | — |
| ≥−3 | 1089 · 47% · -0.269 | 620 · 54% · +0.154 | 14 · +0.787 | — |
| (master unscored) | | | 5 | |

**A8 prior judged stretches**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| 0 | 760 · 46% · -0.338 | 436 · 57% · +0.305 | 8 · -0.881 | — |
| 1 | 346 · 48% · -0.151 | 189 · 50% · -0.087 | 4 · +3.002 | — |
| ≥2 | 169 · 44% · -0.483 | 83 · 48% · -0.213 | 3 · +0.933 | — |
| (master unscored) | | | 5 | |

**A9 new high above the spike-bar high since the spike**

| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |
|---|---|---|---|---|
| yes | 1150 · 47% · -0.289 | 663 · 54% · +0.170 | 15 · +0.518 | — |
| no | 125 · 45% · -0.470 | 45 · 47% · -0.309 | 5 · -0.723 | — |
| (master unscored) | | | 0 | |

## Gated cells (53 pre-registered) — RH ruler, LS slip; Δ = Σ variant − Σ today (%-points at 1×)

Columns: ΣΔ and Δ per today-signal per year · day-bootstrap P(Δ>0) · N signals variant / today · avg per signal variant / today · R1 (1m) P as sensitivity · master Δ · LOMO · largest day / pair share · neighbours · CANDIDATE

| cell | 2025 ΣΔ (per sig) · P | 2026 ΣΔ (per sig) · P | N v/t 25 | N v/t 26 | avg v/t 25 | avg v/t 26 | R1 P 25/26 | master Δ | LOMO | day / pair share | nbrs | CAND |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2_GREEN | +360.3 (+0.283) · 0.99 | -180.6 (-0.255) · 0.06 | 1137/1275 | 683/708 | -0.027/-0.307 | -0.119/+0.140 | 0.99/0.06 | -5.80 (REBUILT master window (36 signal rows)) | ✓ | 0.15 / 0.16 | — | no |
| A2_ANY | -30.7 (-0.024) · 0.39 | -81.6 (-0.115) · 0.14 | 2412/1275 | 1391/708 | -0.175/-0.307 | +0.013/+0.140 | 0.43/0.17 | -1.59 (REBUILT master window (36 signal rows)) | ✓ | 0.24 / 0.26 | — | no |
| A2_WAITRED | -102.5 (-0.080) · 0.16 | -95.2 (-0.134) · 0.11 | 2338/1275 | 1344/708 | -0.211/-0.307 | +0.003/+0.140 | 0.17/0.14 | +10.84 (REBUILT master window (36 signal rows)) | ✓ | 0.12 / 0.12 | — | no |
| A3_N6 | +159.9 (+0.125) · 0.87 | -41.1 (-0.058) · 0.34 | 1665/1275 | 933/708 | -0.139/-0.307 | +0.062/+0.140 | 0.87/0.40 | -22.87 (REBUILT master window (39 signal rows)) | ✓ | 0.28 / 0.23 | — | no |
| A3_N24 | +110.3 (+0.087) · 0.80 | -135.7 (-0.192) · 0.08 | 722/1275 | 451/708 | -0.389/-0.307 | -0.081/+0.140 | 0.79/0.09 | -18.03 (REBUILT master window (33 signal rows)) | ✗ | 1.23 / 0.97 | ✗ | no |
| A3_N36 | +391.0 (+0.307) · 1.00 | -82.5 (-0.116) · 0.21 | 511/1275 | 317/708 | +0.000/-0.307 | +0.052/+0.140 | 1.00/0.20 | -7.64 (REBUILT master window (23 signal rows)) | ✓ | 0.07 / 0.07 | ✗ | no |
| A4_H2_8 | +160.8 (+0.126) · 0.99 | +21.7 (+0.031) · 0.64 | 726/1275 | 407/708 | -0.317/-0.307 | +0.297/+0.140 | 0.99/0.66 | +1.18 (REBUILT master window (20 signal rows)) | ✓ | 0.11 / 0.07 | ✓ | no |
| A4_H2_12 | +123.4 (+0.097) · 0.99 | +34.8 (+0.049) · 0.80 | 999/1275 | 546/708 | -0.268/-0.307 | +0.245/+0.140 | 0.99/0.82 | -11.83 (REBUILT master window (20 signal rows)) | ✓ | 0.10 / 0.08 | ✓ | no |
| A4_H4_1675 | +37.2 (+0.029) · 0.69 | -75.5 (-0.107) · 0.07 | 1114/1275 | 620/708 | -0.318/-0.307 | +0.038/+0.140 | 0.68/0.07 | -2.82 (REBUILT master window (22 signal rows)) | ✓ | 0.48 / 0.40 | ✗ | no |
| A4_H8_1675 | +164.7 (+0.129) · 0.95 | -118.1 (-0.167) · 0.05 | 773/1275 | 439/708 | -0.293/-0.307 | -0.043/+0.140 | 0.95/0.05 | -5.60 (REBUILT master window (22 signal rows)) | ✗ | 0.55 / 0.41 | ✗ | no |
| A4_H2_24 | -84.7 (-0.066) · 0.07 | +8.8 (+0.012) · 0.57 | 1616/1275 | 895/708 | -0.294/-0.307 | +0.120/+0.140 | 0.07/0.57 | -0.20 (REBUILT master window (22 signal rows)) | ✓ | 0.17 / 0.17 | — | no |
| A5_V20 | +282.3 (+0.221) · 1.00 | -99.2 (-0.140) · 0.07 | 804/1275 | 385/708 | -0.135/-0.307 | -0.000/+0.140 | 1.00/0.06 | -14.22 (REBUILT master window (22 signal rows)) | ✓ | 0.10 / 0.12 | ✗ | no |
| A5_V50 | +120.9 (+0.095) · 0.98 | -76.9 (-0.109) · 0.04 | 1115/1275 | 600/708 | -0.242/-0.307 | +0.037/+0.140 | 0.98/0.03 | -8.60 (REBUILT master window (21 signal rows)) | ✓ | 0.29 / 0.43 | ✗ | no |
| A2_ORD1 | +134.0 (+0.105) · 0.98 | +34.1 (+0.048) · 0.76 | 760/1275 | 436/708 | -0.338/-0.307 | +0.305/+0.140 | 0.98/0.77 | -14.81 (master stamps scored 15/20) | ✓ | 0.10 / 0.12 | — | no |
| A2_ORD2P | +257.0 (+0.202) · 1.00 | -133.1 (-0.188) · 0.01 | 515/1275 | 272/708 | -0.260/-0.307 | -0.125/+0.140 | 1.00/0.01 | +7.04 (master stamps scored 15/20) | ✓ | 0.17 / 0.13 | — | no |
| A5_DROPLT20 | +52.4 (+0.041) · 0.75 | -18.6 (-0.026) · 0.38 | 651/1275 | 418/708 | -0.520/-0.307 | +0.192/+0.140 | 0.74/0.37 | +16.24 (master stamps scored 20/20) | ✗ | 0.60 / 0.49 | — | no |
| A6_CAP2 | +150.9 (+0.118) · 0.97 | -20.3 (-0.029) · 0.36 | 739/1275 | 354/708 | -0.325/-0.307 | +0.222/+0.140 | 0.97/0.35 | -8.42 (master stamps scored 20/20) | ✓ | 0.15 / 0.12 | ✗ | no |
| A6_CAP3 | +131.9 (+0.103) · 0.99 | -30.0 (-0.042) · 0.25 | 934/1275 | 479/708 | -0.277/-0.307 | +0.144/+0.140 | 0.99/0.23 | -8.62 (master stamps scored 20/20) | ✓ | 0.13 / 0.13 | ✗ | no |
| A6_CAP5 | +47.2 (+0.037) · 0.92 | -29.0 (-0.041) · 0.17 | 1138/1275 | 624/708 | -0.302/-0.307 | +0.112/+0.140 | 0.92/0.17 | -3.00 (master stamps scored 20/20) | ✗ | 0.36 / 0.70 | ✗ | no |
| A6_FLOOR1 | +110.7 (+0.087) · 0.96 | -32.3 (-0.046) · 0.23 | 906/1275 | 537/708 | -0.309/-0.307 | +0.124/+0.140 | 0.96/0.22 | +9.67 (master stamps scored 20/20) | ✓ | 0.19 / 0.16 | — | no |
| A8_run_lt10 | +57.3 (+0.045) · 0.86 | +0.5 (+0.001) · 0.51 | 1040/1275 | 572/708 | -0.321/-0.307 | +0.174/+0.140 | 0.86/0.47 | +6.46 (master stamps scored 20/20) | ✓ | 0.28 / 0.22 | ✗ | no |
| A8_run_10_20 | +69.1 (+0.054) · 0.81 | -21.7 (-0.031) · 0.35 | 729/1275 | 411/708 | -0.442/-0.307 | +0.188/+0.140 | 0.81/0.36 | +10.23 (master stamps scored 20/20) | ✗ | 0.38 / 0.34 | ✗ | no |
| A8_run_ge20 | +264.7 (+0.208) · 1.00 | -77.9 (-0.110) · 0.09 | 781/1275 | 433/708 | -0.162/-0.307 | +0.049/+0.140 | 1.00/0.09 | -20.84 (master stamps scored 20/20) | ✓ | 0.11 / 0.07 | ✗ | no |
| A8_svx_lt50 | +331.1 (+0.260) · 1.00 | -40.5 (-0.057) · 0.31 | 108/1275 | 129/708 | -0.554/-0.307 | +0.454/+0.140 | 1.00/0.27 | -1.76 (master stamps scored 15/20) | ✓ | 0.08 / 0.08 | ✗ | no |
| A8_svx_50_100 | +48.3 (+0.038) · 0.93 | -46.2 (-0.065) · 0.09 | 1183/1275 | 610/708 | -0.290/-0.307 | +0.087/+0.140 | 0.93/0.11 | -3.00 (master stamps scored 15/20) | ✗ | 3.08 / 4.79 | ✗ | no |
| A8_svx_ge100 | +11.6 (+0.009) · 0.83 | -12.4 (-0.018) · 0.24 | 1259/1275 | 677/708 | -0.301/-0.307 | +0.128/+0.140 | 0.83/0.25 | -3.00 (master stamps scored 15/20) | ✗ | 11.34 / 11.34 | ✗ | no |
| A8_dhi_ge-3 | +293.0 (+0.230) · 1.00 | -95.3 (-0.135) · 0.11 | 186/1275 | 88/708 | -0.527/-0.307 | +0.043/+0.140 | 1.00/0.10 | -11.02 (master stamps scored 15/20) | ✓ | 0.11 / 0.10 | ✓ | no |
| A8_dhi_-3_-8 | +69.9 (+0.055) · 0.96 | +12.8 (+0.018) · 0.69 | 1125/1275 | 642/708 | -0.285/-0.307 | +0.174/+0.140 | 0.96/0.68 | +3.25 (master stamps scored 15/20) | ✓ | 0.12 / 0.12 | ✗ | no |
| A8_dhi_lt-8 | +28.0 (+0.022) · 0.94 | -16.6 (-0.023) · 0.12 | 1239/1275 | 686/708 | -0.293/-0.307 | +0.120/+0.140 | 0.94/0.12 | -0.00 (master stamps scored 15/20) | ✓ | 0.56 / 0.56 | ✓ | no |
| A8_prior_0 | +257.0 (+0.202) · 1.00 | -133.1 (-0.188) · 0.01 | 515/1275 | 272/708 | -0.260/-0.307 | -0.125/+0.140 | 1.00/0.01 | +7.04 (master stamps scored 15/20) | ✓ | 0.17 / 0.13 | ✓ | no |
| A8_prior_1 | +52.4 (+0.041) · 0.82 | +16.4 (+0.023) · 0.65 | 929/1275 | 519/708 | -0.364/-0.307 | +0.223/+0.140 | 0.82/0.67 | -12.01 (master stamps scored 15/20) | ✓ | 0.19 / 0.28 | ✗ | no |
| A8_prior_ge2 | +81.6 (+0.064) · 0.98 | +17.7 (+0.025) · 0.75 | 1106/1275 | 625/708 | -0.280/-0.307 | +0.187/+0.140 | 0.98/0.75 | -2.80 (master stamps scored 15/20) | ✓ | 0.10 / 0.06 | ✓ | no |
| A9_NEWHI | +58.8 (+0.046) · 0.94 | +13.9 (+0.020) · 0.76 | 1150/1275 | 663/708 | -0.289/-0.307 | +0.170/+0.140 | 0.94/0.76 | +3.62 (master stamps scored 20/20) | ✓ | 0.14 / 0.09 | — | no |
| A9_NONEWHI | +332.2 (+0.261) · 1.00 | -112.9 (-0.160) · 0.09 | 125/1275 | 45/708 | -0.470/-0.307 | -0.309/+0.140 | 1.00/0.08 | -7.76 (master stamps scored 20/20) | ✓ | 0.10 / 0.09 | — | no |
| A6_L0.5_W30 | +234.8 (+0.184) · 0.99 | +26.4 (+0.037) · 0.65 | 1275/1275 | 708/708 | -0.122/-0.307 | +0.177/+0.140 | 0.99/0.54 | -11.05 (master priced 20/20) | ✓ | 0.12 / 0.11 | ✗ | no |
| A6_L0.5_W60 | +215.3 (+0.169) · 0.98 | +7.2 (+0.010) · 0.55 | 1275/1275 | 708/708 | -0.138/-0.307 | +0.150/+0.140 | 0.98/0.50 | -14.38 (master priced 20/20) | ✓ | 0.16 / 0.13 | ✗ | no |
| A6_L0.5_W120 | +191.4 (+0.150) · 0.96 | +13.1 (+0.018) · 0.58 | 1275/1275 | 708/708 | -0.157/-0.307 | +0.158/+0.140 | 0.96/0.54 | -20.74 (master priced 20/20) | ✓ | 0.17 / 0.14 | ✓ | no |
| A6_L1_W30 | +195.0 (+0.153) · 0.98 | -0.5 (-0.001) · 0.51 | 1275/1275 | 708/708 | -0.154/-0.307 | +0.139/+0.140 | 0.98/0.45 | -14.43 (master priced 20/20) | ✓ | 0.15 / 0.15 | ✗ | no |
| A6_L1_W60 | +209.5 (+0.164) · 0.98 | -13.5 (-0.019) · 0.44 | 1275/1275 | 708/708 | -0.142/-0.307 | +0.121/+0.140 | 0.98/0.34 | -20.78 (master priced 20/20) | ✓ | 0.16 / 0.15 | ✗ | no |
| A6_L1_W120 | +191.3 (+0.150) · 0.97 | +13.0 (+0.018) · 0.58 | 1275/1275 | 708/708 | -0.157/-0.307 | +0.158/+0.140 | 0.97/0.42 | -20.78 (master priced 20/20) | ✓ | 0.15 / 0.14 | ✗ | no |
| A6_L1.5_W30 | +119.9 (+0.094) · 0.94 | +2.9 (+0.004) · 0.52 | 1275/1275 | 708/708 | -0.213/-0.307 | +0.144/+0.140 | 0.91/0.33 | -17.73 (master priced 20/20) | ✓ | 0.18 / 0.18 | ✗ | no |
| A6_L1.5_W60 | +127.7 (+0.100) · 0.95 | -3.3 (-0.005) · 0.48 | 1275/1275 | 708/708 | -0.207/-0.307 | +0.135/+0.140 | 0.94/0.30 | -20.91 (master priced 20/20) | ✓ | 0.15 / 0.15 | ✗ | no |
| A6_L1.5_W120 | +83.2 (+0.065) · 0.86 | +11.5 (+0.016) · 0.57 | 1275/1275 | 708/708 | -0.241/-0.307 | +0.156/+0.140 | 0.83/0.32 | -14.85 (master priced 20/20) | ✓ | 0.23 / 0.20 | ✗ | no |
| A7_C5_b0 | -58.9 (-0.046) · 0.18 | -53.3 (-0.075) · 0.12 | 1275/1275 | 708/708 | -0.353/-0.307 | +0.065/+0.140 | 0.18/0.11 | -0.12 (master priced 15/20) | ✓ | 0.16 / 0.17 | ✗ | no |
| A7_C5_b0.5 | -98.9 (-0.078) · 0.04 | -26.5 (-0.037) · 0.24 | 1275/1275 | 708/708 | -0.384/-0.307 | +0.102/+0.140 | 0.04/0.25 | +1.47 (master priced 15/20) | ✓ | 0.12 / 0.15 | ✗ | no |
| A7_C5_b1 | -50.4 (-0.040) · 0.11 | -20.4 (-0.029) · 0.24 | 1275/1275 | 708/708 | -0.346/-0.307 | +0.111/+0.140 | 0.10/0.29 | +1.47 (master priced 15/20) | ✓ | 0.21 / 0.28 | ✗ | no |
| A7_C1_b0 | -69.9 (-0.055) · 0.15 | -102.0 (-0.144) · 0.02 | 1275/1275 | 708/708 | -0.361/-0.307 | -0.004/+0.140 | 0.15/0.02 | +1.77 (master priced 15/20) | ✓ | 0.11 / 0.10 | ✗ | no |
| A7_C1_b0.5 | -102.2 (-0.080) · 0.04 | -56.2 (-0.079) · 0.08 | 1275/1275 | 708/708 | -0.387/-0.307 | +0.060/+0.140 | 0.04/0.09 | -0.37 (master priced 15/20) | ✓ | 0.11 / 0.12 | ✗ | no |
| A7_C1_b1 | -67.3 (-0.053) · 0.07 | -10.9 (-0.015) · 0.37 | 1275/1275 | 708/708 | -0.359/-0.307 | +0.125/+0.140 | 0.06/0.37 | +2.36 (master priced 15/20) | ✓ | 0.22 / 0.26 | ✗ | no |
| A7_R5 | -103.6 (-0.081) · 0.12 | +15.1 (+0.021) · 0.60 | 1275/1275 | 708/708 | -0.388/-0.307 | +0.161/+0.140 | 0.11/0.60 | -16.46 (master priced 20/20) | ✓ | 0.21 / 0.28 | ✗ | no |
| A7_VS_b0 | -38.1 (-0.030) · 0.34 | -71.6 (-0.101) · 0.14 | 1275/1275 | 708/708 | -0.337/-0.307 | +0.039/+0.140 | 0.34/0.19 | +2.70 (master priced 20/20) | ✓ | 0.16 / 0.21 | ✗ | no |
| A7_VS_b0.5 | -100.9 (-0.079) · 0.11 | +2.9 (+0.004) · 0.52 | 1275/1275 | 708/708 | -0.386/-0.307 | +0.144/+0.140 | 0.11/0.59 | +2.75 (master priced 20/20) | ✓ | 0.21 / 0.27 | ✗ | no |
| A7_VS_b1 | -127.5 (-0.100) · 0.06 | +28.9 (+0.041) · 0.70 | 1275/1275 | 708/708 | -0.407/-0.307 | +0.181/+0.140 | 0.06/0.76 | +3.45 (master priced 20/20) | ✓ | 0.18 / 0.27 | ✗ | no |

Cells with P ≥ 0.90 (Δ > 0): 2025 31/53 · 2026 0/53 · BOTH 0 · full candidates 0.

**Filter cells — the dropped cohort at 1× (CLAUDE.md expectancy bar read)**

Sleeve breakeven WR (today's signals, RH LS): 51.5%

| cell | dropped N | WR | avg | day-boot P(avg<0) | days |
|---|---|---|---|---|---|
| A2_ORD1 | 787 | 47.8% | -0.214 | 0.98 | 384 |
| A2_ORD2P | 1196 | 50.0% | -0.104 | 0.87 | 507 |
| A5_DROPLT20 | 914 | 50.9% | -0.037 | 0.65 | 413 |
| A6_CAP2 | 890 | 49.2% | -0.147 | 0.91 | 430 |
| A6_CAP3 | 570 | 48.8% | -0.179 | 0.91 | 332 |
| A6_CAP5 | 221 | 50.2% | -0.082 | 0.64 | 180 |
| A6_FLOOR1 | 540 | 48.9% | -0.145 | 0.86 | 325 |
| A8_run_lt10 | 371 | 48.5% | -0.156 | 0.81 | 234 |
| A8_run_10_20 | 843 | 50.8% | -0.056 | 0.70 | 406 |
| A8_run_ge20 | 769 | 47.6% | -0.243 | 0.98 | 363 |
| A8_svx_lt50 | 1746 | 48.9% | -0.166 | 0.98 | 541 |
| A8_svx_50_100 | 190 | 50.5% | -0.011 | 0.52 | 143 |
| A8_svx_ge100 | 47 | 51.1% | +0.017 | 0.48 | 37 |
| A8_dhi_ge-3 | 1709 | 49.6% | -0.116 | 0.93 | 543 |
| A8_dhi_-3_-8 | 216 | 45.8% | -0.383 | 0.95 | 158 |
| A8_dhi_lt-8 | 58 | 48.3% | -0.197 | 0.68 | 52 |
| A8_prior_0 | 1196 | 50.0% | -0.104 | 0.87 | 507 |
| A8_prior_1 | 535 | 49.0% | -0.129 | 0.83 | 328 |
| A8_prior_ge2 | 252 | 45.2% | -0.394 | 0.99 | 183 |
| A9_NEWHI | 170 | 45.3% | -0.428 | 0.95 | 125 |
| A9_NONEWHI | 1813 | 49.5% | -0.121 | 0.94 | 551 |
