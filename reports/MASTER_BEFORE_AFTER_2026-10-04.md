# MASTER before / after — 2026-10-04 (stack 2026-10-03a → 2026-10-04b)

BEFORE = `reports/MASTER_POOL_stacked.csv.pre1004_bak` (stack **2026-10-03a**, this morning). AFTER = `reports/MASTER_POOL_stacked.csv` (stack **2026-10-04b**): SURGE_SHORT_OFF + sleeve fills re-priced at today's size + FRENZY +3 % TP (10-04a, DECISION_LOG 200) and LONG_CHOP_BURST (10-04b, DECISION_LOG 201). Intermediate 10-04a kept at `reports/MASTER_POOL_stacked.csv.pre1004b_bak`.

Population per era: stack_keep ∧ non-probe ∧ non-MANUAL ∧ CLOSED. N · WR (stack_pnl > 0) · net $ (Σ stack_pnl) · days · DAILY COMPOUND RETURN = (1 + net / start)^(1/days) − 1, days = last closed_at − first opened_at of the era's kept fills (min 0.5).

⚠ **Start balance = $3,000 ASSUMED for every batch** — no per-batch starting equity is recorded in CLAUDE_CURRENT_STATE, DECISION_LOG, the batch archives or the order columns; $3,000 is the ledger's flat convention (scripts/current_stack_ledger.py EQ). Today's config paper_balance is $2,900. A batch that started on a different balance scales its DCR accordingly; the BEFORE→AFTER Δ is unaffected in sign.
⚠ This is the MASTER stack only (stack_keep). The pinned ledger (current_stack_ledger.py) applies further live gates (pair blacklist, gate-51 bands, sleeves switched off) and is not what this table shows.

| batch | BEFORE (10-03a): N · WR · net · days · DCR | AFTER (10-04b): N · WR · net · days · DCR | Δ net |
|---|---|---|---|
| BASE | 67 · 90% · $+4,917 · 23.5 d · +4.22%/d | 66 · 91% · $+5,029 · 23.5 d · +4.28%/d | $+112 |
| B1 | 25 · 76% · $+1,197 · 19.8 d · +1.71%/d | 25 · 76% · $+1,197 · 19.8 d · +1.71%/d | $+0 |
| B2 | 40 · 82% · $+3,938 · 9.5 d · +9.25%/d | 40 · 82% · $+3,938 · 9.5 d · +9.25%/d | $+0 |
| B3 | 84 · 58% · $+1,041 · 12.5 d · +2.41%/d | 84 · 58% · $+1,041 · 12.5 d · +2.41%/d | $+0 |
| B4 | 12 · 58% · $-98 · 0.8 d · -4.25%/d | 12 · 58% · $-98 · 0.8 d · -4.25%/d | $+0 |
| B5 | 7 · 100% · $+308 · 1.1 d · +9.41%/d | 7 · 100% · $+308 · 1.1 d · +9.41%/d | $+0 |
| B6 | 4 · 75% · $-125 · 1.7 d · -2.52%/d | 4 · 75% · $-125 · 1.7 d · -2.52%/d | $+0 |
| B7 | 3 · 100% · $+212 · 0.5 d · +14.62%/d | 3 · 100% · $+212 · 0.5 d · +14.62%/d | $+0 |
| B8 | 5 · 100% · $+460 · 1.7 d · +8.69%/d | 5 · 100% · $+460 · 1.7 d · +8.69%/d | $+0 |
| B9 | 39 · 46% · $-666 · 1.1 d · -20.73%/d | 39 · 46% · $-666 · 1.1 d · -20.73%/d | $+0 |
| B10 | 32 · 59% · $-229 · 1.4 d · -5.56%/d | 32 · 59% · $-229 · 1.4 d · -5.56%/d | $+0 |
| B11 | 10 · 30% · $-643 · 0.5 d · -38.26%/d | 10 · 30% · $-643 · 0.5 d · -38.26%/d | $+0 |
| B12 | 15 · 80% · $+792 · 3.9 d · +6.22%/d | 15 · 80% · $+792 · 3.9 d · +6.22%/d | $+0 |
| B13 | 10 · 70% · $+362 · 1.7 d · +6.81%/d | 10 · 70% · $+362 · 1.7 d · +6.81%/d | $+0 |
| B14 | 12 · 42% · $-504 · 0.6 d · -26.73%/d | 12 · 42% · $-504 · 0.6 d · -26.73%/d | $+0 |
| B15 | 6 · 33% · $-596 · 1.5 d · -13.38%/d | 5 · 40% · $-314 · 1.5 d · -6.91%/d | $+283 |
| B16 | 11 · 64% · $-1,101 · 1.6 d · -24.36%/d | 10 · 70% · $-323 · 1.6 d · -6.72%/d | $+778 |
| **TOTAL** | 382 · 68% · $+9,265 · 83.3 d · +1.70%/d | 379 · 68% · $+10,438 · 83.3 d · +1.82%/d | $+1,173 |

TOTAL DCR = (1 + Σ net / $3,000)^(1 / Σ era days) − 1 — one $3,000 book compounding over the summed active spans (descriptive; eras are separate paper resets).

## Every row that changed (4)

| era | opened (UTC) | pair | dir | strategy | probe | BEFORE keep · $ | AFTER keep · $ | Δ $ | why |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 2026-07-10T17:02:16 | LITUSDT | LONG | MOMENTUM |  | kept · $-112.22 | blocked · $+0.00 | $+112.22 | LONG_CHOP_BURST (10-04b): BTC eff72 ≤ 0.007 ∧ 2nd+ fill of a burst → refused |
| B15 | 2026-10-01T01:25:05 | WLDUSDT | LONG | MOMENTUM |  | kept · $-282.58 | blocked · $+0.00 | $+282.58 | LONG_CHOP_BURST (10-04b): BTC eff72 ≤ 0.007 ∧ 2nd+ fill of a burst → refused |
| B16 | 2026-10-02T14:55:48 | SANDUSDT | SHORT | SURGE_SHORT |  | kept · $-222.54 | blocked · $+0.00 | $+222.54 | SURGE_SHORT_OFF (10-04a): surge_short_enabled = false → never opened |
| B16 | 2026-10-03T03:36:19 | ENJUSDT | LONG | FRENZY_LONG |  | kept · $-793.60 | kept · $-238.08 | $+555.52 | sleeve re-priced at today's size (10-04a) |

By cause (all rows incl. probes; the per-batch table above counts non-probe only):

- LONG_CHOP_BURST (10-04b): BTC eff72 ≤ 0.007 ∧ 2nd+ fill of a burst → refused: 2 rows · Δ $+394.80
- SURGE_SHORT_OFF (10-04a): surge_short_enabled = false → never opened: 1 rows · Δ $+222.54
- sleeve re-priced at today's size (10-04a): 1 rows · Δ $+555.52

CHOP_BURST rows (the only 10-04b change vs 10-04a; verified by a column-by-column diff of 10-04a vs 10-04b — no other row or column moved):
- LITUSDT 2026-07-10 17:02:16 (BASE) — eff72 0.001 (rebuilt from BTC 5m), ADAUSDT momentum long opened 22 s earlier → −$112.22 removed.
- WLDUSDT 2026-10-01 01:25:05 (B15) — eff72 0.006 (live stamp), ENAUSDT momentum long opened 2 s earlier → −$282.58 removed.
- The first fill of each burst (ADA Jul-10, ENA Oct-1) stays kept — sub-line A refuses only the 2nd+ fill.


**Assumption (independent review):** ENJ B16 (FRENZY_LONG, opened 2026-10-03 03:36) predates the ADX/DI stamps, so its "strong signal" state is unknown and it is priced at the normal 6× (−$238.08). If its ADX was rising (its DI spread was +13.2), today's 10× would price it at −$396.80: B16 AFTER −$481 instead of −$323, TOTAL Δ +$1,015 instead of +$1,173. Start balance $3,000 per batch is an assumption (no per-batch starting equity is recorded).
