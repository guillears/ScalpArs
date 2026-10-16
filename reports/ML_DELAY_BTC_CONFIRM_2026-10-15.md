# Momentum LONG: wait for BTC to confirm before entering? (2026-10-15)

- **Question (operator):** winners are the trades where BTC rises in the 15 min after entry. So wait D minutes and enter only if BTC rose ≥ X % (and the pair's signal is still on).
- **Pre-registration:** `reports/ML_DELAY_BTC_PREREG.txt`, written before any delayed entry was priced. The script refuses to overwrite it.
- **Safety:** nothing shipped. No engine, config, template or test edits, no git, no bot API.
- **Data:** cache only, plus one small Binance top-up for 6 late master fills (B19–B21): 35 requests, used weight ≤ 603, written to `reports/ml_delay_fetch/`, outside `backtest_cache`.

---

## Plain-English summary

1. **The BTC condition does pick better trades. But waiting for it costs more than it saves.**
   - Signals where BTC then rose had a good result *if entered on time*: +0.27 to +0.56 % on average.
   - By the time BTC has confirmed (15 min), the coin has usually already run: **+0.57 to +0.74 %** on the signals we would take.
   - Entering that late, those same trades **lose**: −0.09 to −0.18 % per fill in every block (backtest 2025, 2026, real master fills).
2. **On real fills the rule loses money vs today.** For the operator's version (wait 15 min, BTC ≥ +0.10 %, signal still on):
   - **Master as-traded:** −0.067 %/signal vs today (P 0.12).
   - **Master on today's stack:** −0.171 %/signal (P 0.01).
   - It skips about 80 % of signals. The trades it does take earn −0.12 / −0.09 % each.
3. **The backtest "improvement" is not an edge. It is mostly the rule trading less in a sleeve that loses in the backtest.**
   - In 2025 the rule beats today by +0.17 %/signal, but its own fills lose (−0.165 %).
   - Simply **not trading at all** would beat today by more: +0.199 (2025) and +0.063 (2026).
   - In the current bull half (2026-H2) the rule is flat to negative (−0.008).
4. **Shorter waits (5 or 10 min) do not fix it.** The coin moves less during a shorter wait, but the rule's fills are still at or below zero in nearly every block. The master check fails for all 18 rule cells.
5. **Waiting alone (no BTC condition) is clearly harmful on real fills:** −0.197 %/signal on master and −0.124 in 2026-H2. That is the pure cost of entering late.
6. **Verdict: NO.** 0 of 18 pre-registered rule cells pass. Every one fails the "master not negative" leg, and none has profitable fills. No Scout line is proposed.
   - **What this adds to the earlier study:** the earlier study found the BTC move after entry explains winners, but nothing at entry predicts it. This one shows that *waiting to see it* is too late: the coin prices the move before we can confirm it.

---

## 0. Fidelity (checked before any delayed pricing)

**Exit replica.**
- No ready-made "price an entry at a later time" tool existed. The closest faithful one is `scripts/ml_exit_optimize_yr5.py` (tick-level replica of today's live momentum-long exit stack: stop / signal-wide stop / ATR widening, recovery hold, hard-TP ladder, runner trail, 180-min no-expansion). It is imported read-only.
- The yr6 frozen config's exit keys are identical to yr5's (verified key by key). Its data accessors were re-pointed, in this process only, at both the 2025 and 2026 caches.
- The "91 % sign agreement" in `YR6_WL_MOMENTUM.md` is replay vs real fills (the engine replay itself), not a re-pricing tool.

| block | priced | within ±0.05 | sign agreement | corr | engine avg | replica avg |
|---|---|---|---|---|---|---|
| replay 2025 | 2617/2617 | 85.1 % | 97.7 % | 0.989 | −0.187 | −0.199 |
| replay 2026-H1 | 1171/1171 | 94.7 % | 99.6 % | 0.995 | −0.085 | −0.085 |
| replay 2026-H2 | 521/521 | 97.3 % | 99.8 % | 0.996 | −0.015 | −0.013 |
| **replay all** | 4309/4309 | **89.2 %** | **98.4 %** | 0.992 | −0.139 | −0.146 |
| master stack_keep (vs stack_pct) | 139/139 | 81.3 % | 98.6 % | 0.940 | +0.141 | +0.153 |
| master as-traded (historical exit stacks) | 191/191 | 82.7 % | 97.9 % | 0.947 | +0.027 | +0.045 |

- **Parity gate (≥ 85 % on the replay): PASS.**
- 2025 sits right at the bar (85.1 %): 21 % of 2025 fills have no ticks and are priced on 1m bars.
- Master within-0.05 is 81–83 %, but sign agreement is 98 %. Master is reported as a check only.
- To keep the comparison like-for-like, **Δ is primary against the replica at D = 0 ("model")** and repeated against the engine's own number ("engine"). Both give the same signs everywhere.

**Signal still valid at t0 + D.**
- Recomputed with the engine's own `calculate_indicators` + `get_signal` (read-only), using the frozen yr6 thresholds, on the 5m read the engine would see (99 closed bars + the forming bar from ticks / 1m).
- Fidelity at the original entry: get_signal says LONG on **76.6 %** of replay fills at t0, 85.9 % at t0 − 25 s (the scan precedes a maker fill), and 91.3 % at either. On master it is 67.5 %.
- That is below the pre-registered 80 %, so **the proxy is primary** (EMA5 > EMA8 > EMA13 > EMA20 and price > EMA5). The proxy says yes on 85.6 % of replay fills at t0.
- The get_signal version is run as a sensitivity (the "Sg" rows). Same conclusion.
- Engine-chain gates after get_signal (BTC macro gates, LOADX, heat, rank, slots) are **not** recomputed.
- ATR recompute vs the stamp: corr 0.998.

## 1. Gate: all 18 pre-registered rule cells

- **A** = BTC ≥ X and signal still valid (proxy). **B** = BTC ≥ X only. **C** = wait only. **CV** = wait + valid. **Sg** = A with get_signal validity (sensitivity).
- Δ is per signal (skip = 0) vs today (model), day-bootstrap P(Δ > 0).
- Gate legs: ① P ≥ 0.90 in 2025 and 2026 separately, engine view same sign · ② master Δ ≥ 0 on as-traded and stack_keep · ③ leave-one-month-out sign-stable · ④ no day/pair ≥ 50 % of ΣΔ.

| cell | Δ 2025 (P) | Δ 2026 (P) | master Δ as-traded / keep | ① | ② | ③ | ④ | verdict |
|---|---|---|---|---|---|---|---|---|
| A D5 X0.05 | +0.145 (1.00) | +0.060 (0.99) | −0.042 / −0.151 | ✓ | ✗ | ✓ | ✓ | fail |
| A D5 X0.10 | +0.165 (1.00) | +0.059 (0.99) | −0.053 / −0.161 | ✓ | ✗ | ✓ | ✓ | fail |
| A D5 X0.15 | +0.172 (1.00) | +0.060 (0.99) | −0.076 / −0.184 | ✓ | ✗ | ✓ | ✓ | fail |
| B D5 X0.05 | +0.143 (1.00) | +0.056 (0.98) | −0.041 / −0.143 | ✓ | ✗ | ✓ | ✓ | fail |
| B D5 X0.10 | +0.167 (1.00) | +0.059 (0.98) | −0.051 / −0.153 | ✓ | ✗ | ✓ | ✓ | fail |
| B D5 X0.15 | +0.172 (1.00) | +0.061 (0.99) | −0.072 / −0.179 | ✓ | ✗ | ✓ | ✓ | fail |
| A D10 X0.05 | +0.159 (1.00) | +0.040 (0.93) | −0.047 / −0.154 | ✓ | ✗ | ✓ | ✓ | fail |
| A D10 X0.10 | +0.174 (1.00) | +0.055 (0.98) | −0.048 / −0.162 | ✓ | ✗ | ✓ | ✓ | fail |
| A D10 X0.15 | +0.192 (1.00) | +0.059 (0.98) | −0.049 / −0.159 | ✓ | ✗ | ✓ | ✓ | fail |
| B D10 X0.05 | +0.147 (1.00) | +0.030 (0.87) | −0.066 / −0.182 | ✗ | ✗ | ✓ | ✓ | one-year only |
| B D10 X0.10 | +0.169 (1.00) | +0.048 (0.95) | −0.053 / −0.169 | ✓ | ✗ | ✓ | ✓ | fail |
| B D10 X0.15 | +0.189 (1.00) | +0.050 (0.96) | −0.053 / −0.165 | ✓ | ✗ | ✓ | ✓ | fail |
| A D15 X0.05 | +0.155 (1.00) | +0.037 (0.92) | −0.081 / −0.176 | ✓ | ✗ | ✓ | ✓ | fail |
| **A D15 X0.10 (operator)** | +0.168 (1.00) | +0.041 (0.93) | **−0.067 / −0.171** | ✓ | ✗ | ✓ | ✓ | fail |
| A D15 X0.15 | +0.177 (1.00) | +0.050 (0.96) | −0.060 / −0.165 | ✓ | ✗ | ✓ | ✓ | fail |
| B D15 X0.05 | +0.141 (1.00) | +0.021 (0.80) | −0.089 / −0.194 | ✗ | ✗ | ✓ | ✓ | one-year only |
| B D15 X0.10 | +0.162 (1.00) | +0.028 (0.85) | −0.084 / −0.193 | ✗ | ✗ | ✓ | ✓ | one-year only |
| B D15 X0.15 | +0.174 (1.00) | +0.042 (0.93) | −0.077 / −0.193 | ✓ | ✗ | ✓ | ✓ | fail |
| *C D5 / D10 / D15 (wait only)* | +0.007 / +0.041 / +0.052 | +0.002 / −0.020 / −0.032 | −0.098 / −0.109 / **−0.197** (as-traded) | | | | | control |
| *Sg D15 X0.10 (get_signal validity)* | +0.186 (1.00) | +0.056 (0.98) | −0.051 / −0.161 | ✓ | ✗ | ✓ | ✓ | fails (sensitivity) |

**Result: 0 of 18 pass.** Every cell fails ② on real fills.
- Multiplicity: 18 cells were tested. With that many, a stray backtest pass would be expected anyway, which is why the master leg exists.
- The engine-view Δ has the same sign as the model view on every cell: within 0.012 on the replay, ~0.02 on master (`ML_DELAY_tables.txt` §1).

**Why ① passes in the backtest but means nothing.** The rule's own fills lose in every block (post-hoc diagnostic, labelled):

| cell | master as-traded fill avg (N) | master keep | 2025 | 2026-H1 | 2026-H2 |
|---|---|---|---|---|---|
| A D15 X0.10 | −0.121 (35) | −0.086 (29) | −0.165 (487) | −0.119 (218) | −0.102 (110) |
| B D15 X0.10 | −0.149 (50) | −0.135 (41) | −0.151 (634) | −0.114 (286) | −0.176 (148) |
| A D5 X0.10 | −0.054 (27) | −0.053 (20) | −0.229 (392) | −0.006 (150) | −0.065 (80) |
| A D10 X0.10 | −0.016 (38) | −0.047 (28) | −0.143 (457) | −0.110 (215) | +0.092 (113) |
| **sleeve OFF (Δ vs today)** | −0.045 | −0.153 | **+0.199** | +0.085 | +0.013 |

- Per signal (vs sleeve off = 0), no rule cell is positive in 2025 or in 2026 pooled. A few are marginally positive on master or 2026-H2 only (≤ +0.02; full grid in `ML_DELAY_tables.txt`).
- In the backtest, "skip everything" beats every rule cell. The positive Δ is the skip effect on a sleeve that loses there. It is **not** a delayed-entry edge.
- This is the sleeve on/off question already pre-committed in CURRENT_STATE (30-fill gate at 1×), not a filter.

## 2. The operator's rule: A D15 X0.10, three blocks

All numbers are % at 1×. Δ is per signal (skip = 0), with day-bootstrap 95 % CI.

| block | signals · days | fill rate | today (model) | rule avg/signal | Δ/sig model [CI] P | Δ/sig engine | rule fills: N · WR · avg | coin move during the wait: all / taken | today's result of taken / skipped | skipped: saved losers / missed winners |
|---|---|---|---|---|---|---|---|---|---|---|
| ① live | 1 · 1 | 0 % | +1.098 | 0 | −1.098 | −1.157 | 0 | +0.445 / — | — / +1.098 | 0 / −1.10 (1) |
| ② master as-traded | 191 · 77 | 18 % | +0.045 | −0.022 | **−0.067** [−0.187, +0.050] 0.12 | −0.049 | 35 · 51 % · −0.121 | +0.078 / **+0.734** | +0.471 / −0.051 | +48.7 (61) / −40.8 (95) |
| ② master stack_keep | 139 · 66 | 21 % | +0.153 | −0.018 | **−0.171** [−0.315, −0.029] 0.01 | −0.159 | 29 · 52 % · −0.086 | +0.128 / +0.736 | +0.558 / +0.046 | +30.6 (38) / −35.7 (72) |
| ③ 2025 BULL | 2617 · 301 | 19 % | −0.199 | −0.031 | +0.168 [+0.124, +0.211] 1.00 | +0.156 | 487 · 54 % · −0.165 | −0.040 / +0.574 | +0.269 / −0.306 | +983 (1204) / −332 (926) |
| ③ 2026-H1 BEAR | 1171 · 154 | 19 % | −0.085 | −0.022 | +0.063 [−0.004, +0.134] 0.96 | +0.063 | 218 · 60 % · −0.119 | +0.000 / +0.635 | +0.340 / −0.182 | +373 (448) / −200 (505) |
| ③ 2026-H2 BULL | 521 · 79 | 21 % | −0.013 | −0.022 | −0.008 [−0.099, +0.083] 0.44 | −0.006 | 110 · 60 % · −0.102 | +0.064 / +0.633 | +0.474 / −0.144 | +145 (184) / −86 (227) |

- **Two-sided read, master as-traded:**
  - Skipping saves +48.7 % of losers but gives up −40.8 % of winners.
  - The 35 trades it keeps lose −20.7 % in total versus entering them on time.
  - Net −12.8 % over 191 signals.
- **Live (N = 1, descriptive only):**
  - The PEPE winner had BTC +0.46 % over the 15 min, but the coin had already run +0.44 % and the proxy stack had broken, so A skipped it.
  - B (BTC only) would have taken it at +1.35 vs +1.10. That is one trade, not evidence.
- **Master per batch** (`ML_DELAY_tables.txt`):
  - The rule helps only in the losing batches, where it mostly sits out: B14 / B15 / B18 / B20 / B21 come out +0.6 to +0.9 because fill rate there is 0–12 %.
  - It hurts where the sleeve earned: BASE −0.56, B1 −0.15, B12 −0.17, B16 −0.48, B17 −0.27.
  - Again, that is an on/off effect, not selection.
- **Leave-one-month-out:** sign-stable in both backtest years (2025: +0.152 … +0.183; 2026: +0.031 … +0.057). But it is stable in the backtest only.
- **Concentration:** the top day / pair carries 5–19 % of ΣΔ. Not concentrated.

## 3. Selection vs price: where the idea breaks

| block | cell | taken: today → when entered late | skipped: today |
|---|---|---|---|
| master | B D5 X0.10 | +0.166 → −0.037 | +0.021 |
| master | B D10 X0.10 | +0.322 → −0.037 | −0.028 |
| master | B D15 X0.10 | +0.354 → −0.149 | −0.065 |
| 2025 | B D15 X0.10 | +0.118 → −0.151 | −0.300 |
| 2026-H1 | B D15 X0.10 | +0.155 → −0.114 | −0.163 |
| 2026-H2 | B D15 X0.10 | +0.323 → −0.176 | −0.147 |

- The operator's observation is confirmed: BTC-up signals are the better ones if entered on time.
- The confirmation arrives together with the coin's own move, which is the move the exit stack is built to harvest.
- Entering after it turns +0.12 … +0.35 into −0.04 … −0.18. The pure-wait control shows the same cost: −0.197 %/signal on master.

## 4. Recommendation

- **No.** No candidate. No observe line, no config, no code.
- Do not re-run this grid with other D / X values. The failure is structural (the price moves during the wait), not a threshold choice.
- The real lever remains the CURRENT_STATE momentum-long on/off gate (30 fills at 1×).

## 5. Could not test

- **Signal-validity fidelity:**
  - get_signal reproduces the engine's LONG on 77 % of replay fills at t0 (91 % allowing the ~25 s scan lead), so the proxy is primary.
  - Engine-chain gates after get_signal (BTC macro gates, LOADX, heat, rank / megacap, blacklist, chop-burst) are not re-checked at t0 + D. If anything, they would block more late entries.
- **Slot effects:** a delayed entry holds or frees a slot at different times. Other sleeves' fills and same-pair re-signals inside the wait are not modelled.
- **Only signals today's filters admit:** the replay only contains signals that today's filters let through. Signals blocked at t0 but valid at t0 + D are absent.
- **Execution model:** the delayed fill keeps the original fill's maker/taker edge (E0 / p(t0)). A late chase after a +0.6 % move would more likely be taker with ≥ 0.10 slip (live ruler), which makes every cell worse.
- **Timing resolution:** BTC confirmation uses 1m closes, up to 1 min stale, and the wait is measured on closed 1m bars.
- **Coverage gaps:** 21 % of 2025 fills are priced on 1m paths (no ticks). Master parity is 81–83 % within ±0.05 (sign 98 %).
- **Live:** N = 1.

## 6. Exact commands (cache-only except the top-up; about 2 min on 6 cores)

```
venv/bin/python scripts/study_ml_delay_prereg.py                      # write-once pre-registration (refuses overwrite)
venv/bin/python -W ignore scripts/study_ml_delay_price.py --jobs 6 --blocks master,live --out /tmp/ml.csv
venv/bin/python -W ignore scripts/study_ml_delay_fetch.py /tmp/ml.csv  # top-up of the 6 uncached late master fills → reports/ml_delay_fetch/
venv/bin/python -W ignore scripts/study_ml_delay_price.py --jobs 6     # → reports/ML_DELAY_rows.csv (4,501 signals × D ∈ {0,5,10,15})
venv/bin/python -W ignore scripts/study_ml_delay_analyse.py > reports/ML_DELAY_tables.txt   # all tables, csv-module recount, → reports/ML_DELAY_cells.csv
```

**Files:**
- `reports/ML_DELAY_BTC_PREREG.txt`
- `reports/ML_DELAY_rows.csv`
- `reports/ML_DELAY_cells.csv`
- `reports/ML_DELAY_tables.txt`
- `scripts/study_ml_delay_{common,prereg,price,fetch,analyse}.py`
