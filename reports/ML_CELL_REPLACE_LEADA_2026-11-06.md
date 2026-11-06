# Momentum long — today's 1.5× UNMATCHED cell vs Lead A, with the Market-hot filter (read 2026-11-06, export 11-06 13:03)

Operator 11-06: "today we have some momentum longs with 1.5x … they fit in or out the mkt hot … i want to see if this 1.5x is complementary,
or we have to replace the current" → "in the replace, + mkt hot filter … before and after" → "vs just market hot" → "same in master batch" →
"save this to the B25 review plan". DECISION_LOG 481 (+ its same-day correction). NOTHING CHANGED IN THE BOT.

**Definitions.** 1.5× = the UNMATCHED long cell (inv 1.5, DL 206), de-muxed to 1× when pair volume ratio ≥ 0.90 or crowd sprint
(global volume > 0.74 ∧ BTC EMA20 slope > 0.07); other routes (NONEXP_CALM3D, ADX_SURGE_OPEN) 1×. Rule replicates B25 exactly (19 / 19);
master = today's rule on the master columns. Market-hot = ML_MKT_HOT zone (DL 416, best-case zones.csv). Lead A = not hot ∧
`entry_ema20_slope` > 0.1481 ∧ `entry_rsi_prev` ≤ 60.9 (DL 481). Full-size only (master 158 on today's stack, B25 39).
Daily compound: B25 = the real account path (review_math.dcr_account, 2,900 → +7,100 on 10-26), master = per-batch calendar days at the
1× intended notional (B25 median notional ÷ equity ÷ size), averaged over batches.

## Overlap — size × Market-hot × Lead A (avg P&L %)

| | All | Market-hot (blocked) | Not hot (kept) | of which Lead A |
|---|---|---|---|---|
| B25 1.5× | 19 · 79 % · +0.32 | 3 · 100 % · +0.14 | 16 · 75 % · +0.36 | 3 · 100 % · +1.86 |
| B25 1× | 20 · 65 % · +0.08 | 15 · 53 % · −0.10 | 5 · 100 % · +0.62 | 0 |
| Master 1.5× | 73 · 75 % · +0.22 | 24 · 75 % · +0.08 | 49 · 76 % · +0.29 | 19 · 89 % · +0.69 |
| Master 1× | 85 · 67 % · +0.04 | 49 · 61 % · −0.04 | 36 · 75 % · +0.16 | 8 · 88 % · +0.68 |

22 of the 30 Lead A fills already ride the 1.5× cell (B25 3 / 3). Inside the not-hot 1.5× fills, Lead A 22 · +0.85 vs the other 43 ≈ +0.02:
the cell's edge is almost all Lead A.

## Current batch (B25, 13.8 days)

| Scenario | Trades | WR | Avg % | Size-weighted avg % | Total $ | Trades/day | Avg hold | Daily compound %/day |
|---|---|---|---|---|---|---|---|---|
| Before (today: UNMATCHED 1.5×, no Market-hot) | 39 | 72 % | +0.20 | +0.22 | +187 | 2.83 | 51 min | +0.183 |
| Market-hot only (1.5× cell kept) | 21 | 81 % | +0.42 | +0.40 | +216 | 1.53 | 34 min | +0.194 |
| Market-hot + replace, Lead A ×1.5 | 21 | 81 % | +0.42 | +0.51 | +214 | 1.53 | 34 min | +0.193 |
| Market-hot + replace, Lead A ×2 | 21 | 81 % | +0.42 | +0.60 | +266 | 1.53 | 34 min | +0.238 |
| (ref) Market-hot, everything 1× | 21 | 81 % | +0.42 | +0.42 | +163 | 1.53 | 34 min | +0.147 |

Replace ×1.5 moves: blocked 18 · −0.06 (−$29.77 → 0, +$29.77) · 1.5× → 1× 13 · +0.01 (−$2.08) · Lead A stays 1.5× 3 · +1.86 · 1× unchanged 5 · +0.62.
Per fill: 1.5× → 1× = ENA 10-24 −0.89, STRK, UNI ×2, LINK ×2, DOT, XLM, ENA / PUMP 11-02 −0.69, XMR 11-03 −0.71, SUI, SHIB · Lead A = LIGHT +0.82, ARB +4.66, PUMP 11-05 +0.09.

## Master (158 fills, 23 batches, 123 calendar days) — money = Σ % × size (1× units)

| Scenario | Trades | WR | Avg % | Size-weighted avg % | Σ % × size | Trades/day | Avg hold | Daily compound %/day |
|---|---|---|---|---|---|---|---|---|
| Before (today: UNMATCHED 1.5×, no Market-hot) | 158 | 71 % | +0.12 | +0.14 | +27.7 | 1.28 | 29 min | −0.016 |
| Market-hot only (1.5× cell kept) | 85 | 75 % | +0.23 | +0.24 | +26.6 | 0.69 | 34 min | +0.015 |
| Market-hot + replace, Lead A ×1.5 | 85 | 75 % | +0.23 | +0.29 | +28.9 | 0.69 | 34 min | +0.022 |
| Market-hot + replace, Lead A ×2 | 85 | 75 % | +0.23 | +0.34 | +38.3 | 0.69 | 34 min | +0.032 |
| (ref) Market-hot, everything 1× | 85 | 75 % | +0.23 | +0.23 | +19.6 | 0.69 | 34 min | +0.013 |

Replace ×1.5 moves: blocked 73 · +0.00 (Σ +1.03 → 0, −1.03) · 1.5× → 1× 30 · +0.03 (−0.43) · 1× unchanged 28 · +0.00 · Lead A 1× → 1.5× 8 · +0.68 (+2.73) ·
Lead A stays 1.5× 19 · +0.69. Complement (take-the-max: keep the cell, add Lead A 1.5×) master +0.020 / B25 +0.194 ≈ replace; stacked 2.25×
master +0.027 / B25 +0.262 (ARB-driven).

Per batch, replace ×1.5 vs Market-hot only (Σ % × size): better in B6, B12, B16, B17, B18, B22, B23 · worse in BASE (17.22 → 14.90), B1, B3, B8,
B9, B13, B21 · equal elsewhere → 7 better / 7 worse: the in-sample total is NOT batch-consistent at ×1.5.

## Best-case view with the replace (export 11-06 13:03, %/day; scripts: the best-case pipeline + an ad-hoc replace layer)

Momentum Long row of table ① (sleeve alone): before +0.16 / master −0.02 → Market-hot, cell kept +0.17 / +0.015 · replace Lead A 1.5× +0.17 / +0.022
· replace Lead A 2× +0.21 / +0.032.

| Scenario (table ②) | ML sizing | Total, current batch | Total, master | Momentum Long, current batch | Momentum Long, master |
|---|---|---|---|---|---|
| Watch lines only | today (UNMATCHED 1.5×) | +4.90 | +1.56 | +0.109 | +0.015 |
| | replace → Lead A 1.5× | +4.90 | +1.57 | +0.108 | +0.022 |
| | replace → Lead A 2× | +4.92 | +1.58 | +0.133 | +0.032 |
| + WILLY 10×, FRENZY_LONG ×5, Flatbase 0.5 | today | +5.68 | +2.43 | +0.103 | +0.015 |
| | replace → Lead A 1.5× | +5.68 | +2.44 | +0.103 | +0.022 |
| | replace → Lead A 2× | +5.71 | +2.45 | +0.127 | +0.032 |
| + … Momentum Long ×5 | today | +6.08 | +2.49 | +0.505 | +0.072 |
| | replace → Lead A 1.5× | +6.07 | +2.52 | +0.504 | +0.108 |
| | replace → Lead A 2× | +6.19 | +2.56 | +0.616 | +0.154 |
| + … Momentum Long ×10 | today | +6.55 | +2.55 | +0.984 | +0.136 |
| | replace → Lead A 1.5× | +6.54 | +2.61 | +0.983 | +0.208 |
| | replace → Lead A 2× | +6.75 | +2.69 | +1.195 | +0.296 |

Clean only = today (no line past its checkpoint); luck-checked switches the replace off (Market-hot NOT ABOVE LUCK, Lead A from a search not
above luck). Master ML on the 1× notional ruler (B25 notional ÷ equity ÷ its cell size); "Momentum Long, current batch" is the sleeve inside the
full scenario (other sleeves' gains grow the account), so it reads lower than table ①.

## Recommendation (for the B25 review, ≈ Nov 12)

1. Decide **Market-hot first**, on its clean fills then. It is the base under both options. Weaknesses: master money it blocks nets ≈ 0
   (73 · +0.00 — its +0.03 %/day mostly comes from zeroing small all-hot batches B14 / B19 / B20), it blocked winners since its 10-30
   freeze, NOT ABOVE LUCK.
2. If it holds: **REPLACE** the UNMATCHED 1.5× with Lead A 1.5× — the same risk on the fills that carry the edge, ≈ neutral in money
   (master +0.022 vs +0.015 / +0.020 complement; B25 equal). Read the cell's own gate first (DL 206 / 227 / 233, CURRENT_STATE cell registry). The simulation sizes Lead A at 1.5× EVEN where the pair-vol (≥ 0.90) or crowd-sprint de-mux would cut it to 1× (the 8 master "1× → 1.5×" fills, +2.73 of the gain) — the engine rule must do the same (Lead A overrides both de-mux legs), or the read is re-run with the de-mux kept.
3. Lead A **×2 only after its new fills confirm** (scripts/scout_ml_lead_a_watch.py) — that step is where the money is (master +0.032,
   B25 +0.238) and where the in-sample / not-above-luck risk bites.
4. Any of it needs an engine build: live market-breadth scoring (108 coins vs 4 h earlier) + the Lead A size rule (config field + UI + revert
   gate, D11), first step 1.5×, 30–50 % haircut, tight pre-committed revert.
5. If replaced — the Open Orders + Closed Orders STRATEGY badge (operator 11-06: "consider the batch that appears in open trades and closed trades … the new 1.5x should be in the lead A ones"): today `doorBadges()` (templates/index.html ~L9269) tags UNMATCHED longs QUIET / DE-MUX / UNM by pair-vol ratio + the row's `cell_multiplier`. After the replace a Lead A fill must read LEAD A 1.5× (then ×2 if stepped) and every other UNMATCHED long its real size (UNM 1×), on BOTH tables, from the row's RECORDED size source (a new `cell_multiplier_source` value such as `LEAD_A` and/or new stamps `entry_mkt_breadth` / `entry_ml_lead_a` with their STAMP_* class, D13), never recomputed in the browser; the dashboard Multiplier-Cell table and both text exports carry the new cell (D12); grep-verify the badge on an open and a closed Lead A row before push.
