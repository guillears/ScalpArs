# 📈 Capacity path — days to $50k … $1M at 8 % and 5 %/day with per-sleeve caps (2026-11-09, DECISION_LOG 513)

Operator ask: "Lets run the case to 50k, 100k, 200k, 500k and 1M — how many days, considering 8 % per day" → "you are not considering the capacity limit per sleeve" → "add the 5 % row".

**Rebuild:** `venv/bin/python -I scripts/best_case/capacity_path.py <orders export.csv> [--rates 8,5] [--start 10000] [--no-reserve]` (method + approximations in the script docstring; sleeve shares read from the newest BEST_CASE table ①). For a later batch, change `--since` / `--start-equity` to that batch's size boundary.

**Inputs of this run:** export 2026-11-09 16-20 · BEST_CASE 2026-11-09 12-45 · B25 fills since the 10-26 13:50 top-up · reserve schedule + leverage_balance_schedule (20× → 15× above $25k) from trading_config.json.

**Rates:** 8 % = the in-sample best case (B25 +8.43 lines only … +11.58 with re-sizing) · 5 % = the luck-checked best case (+5.02, lines NOT ABOVE LUCK switched off). Both use the same sleeve mix (the 8 % mix) — an approximation for 5 %.

**Read:** coin-volume caps, not the reserve or the leverage step, bend the path (reserve off at 8 %: $1M in 184 days vs 192). Spike Fade and LITE cap out first (≈ half size by $25k); Flatbase and Momentum Long hold best. The rate falls to ≈ 2.5 %/day (8 % start) / 1.6 % (5 % start) at $200k and ≈ 0.7 / 0.4 % at $1M — the next step up needs deep-coin sleeves. Approximations: coin volumes move with the market; cap profiles use all of a sleeve's fills, its share only the best-line fills; Momentum Short (3 fills) and flips (6) are thin.

```
best case: BEST_CASE_2026-11-09_12-45.md · export scalpars_orders_paper_2026-11-09_16-20-40.csv · fills since 2026-10-26 13:50: 221 · reserve schedule · leverage schedule '0:20, 25000:15' · 8 / 5 %/day at $10,000
sleeve shares of the rate: FRENZY_LITE_RETRY 24% (34 fills) · FRENZY_WILLY 4% (25 fills) · FLATBASE_SHORT 13% (15 fills) · FLIP:FAN_RATIO_GATE 12% (6 fills ⚠ few) · SPIKE_FADE 10% (20 fills) · FRENZY_LITE 25% (29 fills) · FRENZY_WILLY_HT 4% (28 fills) · MOMENTUM_LONG 1% (38 fills) · MOMENTUM_SHORT 6% (3 fills ⚠ few)

| Balance | Rate at 8 % start | Rate at 5 % start | FRENZY_LITE_RETRY | FRENZY_WILLY | FLATBASE_SHORT | FLIP:FAN_RATIO_GATE | SPIKE_FADE | FRENZY_LITE | FRENZY_WILLY_HT | MOMENTUM_LONG | MOMENTUM_SHORT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $10,000 | 8.00 | 5.00 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| $25,000 | 6.23 | 3.90 | 100% | 68% | 100% | 100% | 40% | 52% | 93% | 100% | 65% |
| $50,000 | 5.09 | 3.18 | 92% | 51% | 100% | 91% | 21% | 29% | 82% | 100% | 45% |
| $100,000 | 3.84 | 2.40 | 72% | 31% | 100% | 74% | 11% | 15% | 73% | 100% | 31% |
| $200,000 | 2.50 | 1.56 | 50% | 22% | 99% | 58% | 7% | 9% | 67% | 98% | 19% |
| $500,000 | 1.28 | 0.80 | 26% | 13% | 75% | 37% | 3% | 5% | 41% | 88% | 9% |
| $1,000,000 | 0.66 | 0.42 | 17% | 8% | 45% | 22% | 2% | 3% | 25% | 72% | 6% |
| $2,000,000 | 0.34 | 0.21 | 11% | 5% | 28% | 14% | 1% | 2% | 15% | 50% | 3% |

**8 %/day start** (day 1 = Nov 10, 2026)

| Target | No caps, no reserve | With caps + reserve + leverage schedule | ≈ Date reached |
|---|---|---|---|
| $50,000 | 21 | **26** | Dec 5, 2026 |
| $100,000 | 30 | **42** | Dec 21, 2026 |
| $200,000 | 39 | **64** | Jan 12, 2027 |
| $500,000 | 51 | **115** | Mar 4, 2027 |
| $1,000,000 | 60 | **192** | May 20, 2027 |

**5 %/day start** (day 1 = Nov 10, 2026)

| Target | No caps, no reserve | With caps + reserve + leverage schedule | ≈ Date reached |
|---|---|---|---|
| $50,000 | 33 | **41** | Dec 20, 2026 |
| $100,000 | 48 | **67** | Jan 15, 2027 |
| $200,000 | 62 | **102** | Feb 19, 2027 |
| $500,000 | 81 | **184** | May 12, 2027 |
| $1,000,000 | 95 | **306** | Sep 11, 2027 |
```

---

## Re-run 2026-11-09 evening — export 14-33, BEST_CASE 14-33, WILLY + WILLY_HT re-sized (DECISION_LOG 514)

Operator: "commit and push, and re-run the capacity path" after the best-case C5 row (WILLY ×2.05 · WILLY_HT ×4.09 = one −4 stop costs 5 % of the account; after each fill's coin cap the traded size grows only ×1.46 / ×3.27 — the caps already bind at today's balance, deep review 514 — so C5 = +11.97 %/day vs C4 +11.21, a +0.76 lift).

```
venv/bin/python -I scripts/best_case/capacity_path.py ~/Downloads/scalpars_orders_paper_2026-11-09_14-33-26.csv
venv/bin/python -I scripts/best_case/capacity_path.py ~/Downloads/scalpars_orders_paper_2026-11-09_14-33-26.csv --resize FRENZY_WILLY=2.05,FRENZY_WILLY_HT=4.09
venv/bin/python -I scripts/best_case/capacity_path.py ~/Downloads/scalpars_orders_paper_2026-11-09_14-33-26.csv --rates 8.76 --resize FRENZY_WILLY=2.05,FRENZY_WILLY_HT=4.09
```

`--resize` takes the NOMINAL factor: the sleeve's wanted size × f (its coin cap does not move) and its share × f × (what its caps let through at the start balance, new ÷ today).

Days from $10k, with caps + reserve + leverage schedule:

| Target | 8 %, today's sizes | 8 %, WILLY / HT re-sized | 5 %, today's sizes | 5 %, WILLY / HT re-sized | 8.76 %, WILLY / HT re-sized (8 % + the C5 lift) |
|---|---|---|---|---|---|
| $50,000 | 27 (26) | 27 | 42 (41) | 42 | 24 (Dec 3) |
| $100,000 | 43 (42) | 43 | 67 (67) | 68 | 40 (Dec 19) |
| $200,000 | 65 (64) | 67 | 103 (102) | 107 | 62 (Jan 10) |
| $500,000 | 117 (115) | 125 | 186 (184) | 199 | 114 (Mar 3) |
| $1,000,000 | 195 (192) | 211 | 311 (306) | 337 | 193 (May 21, 2027) |

(brackets = the 12-45 run above; the small move comes from the flips line losing share after INJ / PEPE / UNI.)

**Reading.** At the SAME start rate a bigger WILLY / HT is slower above $100k: their share of the rate grows (WILLY 4 → 6 %, HT 4 → 11 %) and their coins cap early (at $200k WILLY keeps 17 %, HT 24 % of its wanted size, vs 22 % / 67 % at today's size). The re-size pays only through a higher start rate: the C5 lift (+0.76 %/day in-sample) on 8 % gives 8.76 % → $1M in 193 days instead of 195 — about the same, because the lift is spent in coins that cap out. The lift is in-sample (HT on 6 fills) and both WILLY lines are NOT ABOVE LUCK (luck-checked lift = 0), so the luck-checked 5 % row stays at today's sizes.

---

## Re-run 2026-11-09 late — rates from the newest best case (BEST_CASE 14-33, table ②), each with its own re-sizes

Operator: "re-run the capacity path with the new best case". The start rates now come from best-case table ② instead of the fixed 8 / 5:

| Start rate | Best-case row | Re-sizes passed (`--resize`, nominal factors) |
|---|---|---|
| **11.97 %/day** | in-sample, C5 (watch lines + every re-size) | RETRY ×1.70 · LITE ×0.86 · Flatbase ×2.05 · Momentum Long ×10 · WILLY ×2.05 · WILLY_HT ×4.09 |
| **8.08 %/day** | in-sample, watch lines only | none |
| **4.86 %/day** | luck-checked, C5 (= C1: only re-sizes whose lines are not NOT ABOVE LUCK, plus the step-down) | Flatbase ×2.05 · LITE ×0.86 |

```
E=~/Downloads/scalpars_orders_paper_2026-11-09_14-33-26.csv; B=reports/best_case/BEST_CASE_2026-11-09_14-33.md
venv/bin/python -I scripts/best_case/capacity_path.py $E --best-case $B --rates 11.97 --resize FRENZY_LITE_RETRY=1.70,FRENZY_LITE=0.86,FLATBASE_SHORT=2.05,MOMENTUM_LONG=10,FRENZY_WILLY=2.05,FRENZY_WILLY_HT=4.09
venv/bin/python -I scripts/best_case/capacity_path.py $E --best-case $B --rates 8.08
venv/bin/python -I scripts/best_case/capacity_path.py $E --best-case $B --rates 4.86 --resize FLATBASE_SHORT=2.05,FRENZY_LITE=0.86
```

Days from $10,000 (day 1 = Nov 10, 2026), with coin caps + reserve + leverage schedule (no caps in brackets):

| Target | 11.97 % (in-sample, all re-sizes) | 8.08 % (lines only) | 4.86 % (luck-checked) |
|---|---|---|---|
| $50,000 | **17** (15) · Nov 26 | **26** (21) · Dec 5 | **41** (34) · Dec 20 |
| $100,000 | **28** (21) · Dec 7 | **42** (30) · Dec 21 | **65** (49) · Jan 13 |
| $200,000 | **45** (27) · Dec 24 | **64** (39) · Jan 12 | **98** (64) · Feb 15 |
| $500,000 | **92** (35) · Feb 9 | **116** (51) · Mar 5 | **189** (83) · May 17 |
| $1,000,000 | **169** (41) · Apr 27 | **193** (60) · May 21 | **332** (98) · Oct 7 |

Rate left after the caps: 11.97 % start → 7.90 at $50k · 5.40 at $100k · 3.07 at $200k · 1.30 at $500k · 0.65 at $1M. 8.08 % → 5.06 · 3.82 · 2.48 · 1.28 · 0.66. 4.86 % → 3.33 · 2.60 · 1.57 · 0.69 · 0.36.

**Reading.** The higher in-sample start (11.97 vs 8.08) saves 9 days to $50k but only 24 days to $1M: above $200k both paths converge on ≈ 1.3 %/day at $500k and ≈ 0.65 % at $1M, because the re-sized sleeves (Momentum Long ×10, RETRY ×1.7, Flatbase ×2.05, WILLY / HT) hit their coin caps sooner (at $200k Momentum Long keeps 30 % of its wanted size re-sized vs 98 % at today's size; Flatbase 74 % vs 99 %). The re-sizes buy speed below $100k, not capacity. The luck-checked path is the one to plan on: $100k mid-January, $1M ≈ 11 months. Approximations: the luck-checked run keeps the in-sample sleeve mix (sleeves whose lines are switched off still hold their share, as before); Momentum Short (3 fills) and flips (6) are thin; coin volumes move with the market. In-sample rates carry the 30–50 % cut — 0 of 12 lines have passed their checkpoint.
