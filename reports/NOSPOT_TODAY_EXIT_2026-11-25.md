# No-spot coins at TODAY's EXHAUSTION exit (2026-11-25, research only, DECISION_LOG 642)

**Question (operator):** the 11-18 no-spot verdict ("2,795 trades −0.09 %/trade vs +0.43 with spot") was priced on the OLD exit
(TP min(3 × ATR, 5) / SL min(1.5 × ATR, 5)). Does it hold at today's exit (DL 609: TP min(20 × ATR15, 7 %) / SL min(1.5 × ATR15, 6 %) / 24 h)?
**Prior-study check:** NOSPOT_MCAP (old exit) · NOSPOT_EXIT (DL 579 exit TP ≤ 10 / SL ≤ 6 + a TP 2–5 / 10 grid) · TP_GRID 2026-11-20 (TP 7 on
WITH-spot fills only). New here = TP 7 / SL 6 on the no-spot rows. Nothing else re-run.

**Method:** the NOSPOT_SLEEVE NS0 rows (EXH trigger on no-spot coins, 2023-07 → 2026-11), the NOSPOT_EXIT 1m engine + the TP_GRID TP-only
kernel, real funding, modelled slip, maker TP / taker stop + time, one position per pair per cell. DESIGN < 2025-07, HOLDOUT ≥ 2025-07.
Parity: REF (the DL 579 exit) reproduces NOSPOT_EXIT exactly — −0.050 %/trade on 2,696. With-spot side = TP_GRID FRENZY rows (same engine and costs).
Code: scripts/NOSPOT_TODAY_EXIT_2026-11-25_{sim,eval}.py · data: NOSPOT_TODAY_EXIT_2026-11-25{,_halves}.csv.

## Result (avg net %/trade)
| exit | no-spot ALL: N · WR · avg | no-spot HOLDOUT (≥ 2025-07): N · avg | with spot ALL: N · avg | with spot HOLDOUT: N · avg |
|---|---|---|---|---|
| TP ≤ 10 / SL ≤ 6 (DL 579) | 2,696 · 36 % · −0.05 | 2,290 · +0.04 | 3,368 · +0.68 | 2,019 · +0.74 |
| TP 5 / SL 6 | 2,704 · 51 % · −0.15 | 2,298 · −0.09 | 3,368 · +0.46 | 2,019 · +0.51 |
| **TP 7 / SL 6 (today)** | **2,699 · 44 % · −0.12** | **2,293 · −0.04** | **3,368 · +0.61** | **2,019 · +0.72** |
| today, TP filled only ≥ 0.05 % through (live rule) | 2,699 · 43 % · −0.14 | 2,293 · −0.06 | – | – |

TODAY (TP min(20 × ATR, 7)) ≡ fixed TP 7 on these rows (lowest ATR 1.19 % → 20 × ATR > 7 always).
No-spot at today's exit — ALL: day-block 95 % CI −0.35 … +0.12 · Kelly at the Wilson floor −0.05 · leave-top-pair-out −0.15 ·
positive half-years 2 of 7 (24-H1 +0.84, 26-H1 +0.12) · latest half 26-H2 (625) −0.18. HOLDOUT: WR 44.9 % vs breakeven 45.2 % ·
CI −0.26 … +0.22 · Kelly@floor −0.04 · LOO −0.07 — about breakeven, no edge. TP 7 is slightly WORSE than TP 10 on no-spot coins
(HOLDOUT −0.04 vs +0.04); with spot it is about equal (+0.72 vs +0.74). Adding no-spot to the sleeve at today's exit ≈ +0.61 → +0.28 %/trade (ALL, trade-weighted).

**Caveats:** the with-spot side also carries the DL 582 filter (no-spot rows cannot be scored on it — it needs spot volume) · the
with-spot rows fixed one position per pair once under the REF exit (N 3,368 in every cell); the no-spot rows re-apply it per cell's own exit
(small difference) · every cell but TODAY_T05 fills the TP one tick through · backtest — the forward watch (scout_exh_nospot_630, frozen
gate 30 setups / 10 days) is the live read.

## Verdict
No change. No edge on no-spot coins at today's exit either (Kelly < 0 in every block; HOLDOUT ≈ breakeven, ALL negative); EXH_NO_SPOT stays a block; the DL 630 forward watch keeps its frozen gate.
