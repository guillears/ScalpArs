# Live batch B27 — before / after every rule shipped since go-live (2026-11-16)
Data: live export `scalpars_orders_live_2026-11-16_12-09-51.csv` (20 bot fills, all CLOSED, 0 MANUAL) + decisions `…_12-09-56.csv`. Go-live 2026-11-14 23:45 UTC, baseline $10,076.78.
Rules applied with the engine's own functions (`services/watch_filters.py`) on each fill's stamps: DL 554 LITE_BTC_RSI_HOT · DL 555 ML_WEAK_RESCUE · DL 556 LITE_WEAK_HOLD · DL 559 ML_WEAK_MARKET + ML_FADING_QUIET · DL 560 momentum-short 15× · Rule A (momentum short blocked when BTC daily EMA50 > EMA200 ∧ quality score ≤ 2 or ≥ 5 — BTC daily gap +1.34 on 11-16, uptrend). Every fill in the export traded before the rule that would block it was deployed.

## Per sleeve
| Sleeve | Before (as traded) | After (all rules from the start) | Removed |
|---|---|---|---|
| Momentum LONG | 8 · 25 % · −$1,310 | 2 · 50 % · −$235 (XLM +$68, AAVE −$303) | ML_WEAK_MARKET: AVAX −$246, INJ +$10, ARB −$188 · ML_FADING_QUIET: XRP −$223, AVAX −$380, DOGE −$47 |
| Momentum SHORT | 3 · 33 % · −$638 | 0 · $0 | Rule A: QNT −$391, LTC −$278, ONE +$32 (15× alone would have made them −$470) |
| FRENZY_LITE | 3 · 67 % · −$78 | 2 · 100 % · +$474 | LITE_BTC_RSI_HOT: SHELL −$552 |
| FRENZY_WILLY | 2 · 100 % · +$124 | unchanged | — |
| FRENZY_WILLY_HT | 1 · 0 % · −$439 | unchanged | — |
| FRENZY_LONG | 1 · 0 % · −$14 | unchanged | — |
| FLIP short | 1 · 100 % · +$54 | unchanged | — |
| SPIKE_FADE | 1 · 0 % · −$101 | unchanged | — |
| **Total** | **20 · 40 % · −$2,401** | **11 · 55 % · −$136** | 9 fills: 7 losers −$2,306, 2 winners +$41 |

## Account
- As traded: $10,076.78 − $2,401 ≈ **$7,675 (−23.8 %)** (live dashboard ≈ $7,681 with funding).
- All rules from the start: $10,076.78 − $136 ≈ **$9,941 (−1.4 %)**.

## Already blocked live (not in the export; their effect is already in "as traded")
Depth guard skipped H (WILLY, 11-15 04:20), TAC (HT, 11-15 13:10), SOPH (HT, 11-16 04:30) — all crashed after, ≈ +$2,200 in its favour. LITE_WEAK_HOLD blocked SOPH LITE (11-16 07:05; would have won ≈ +3.66 %), ML_WEAK_RESCUE blocked SUI (11-16 07:56), LITE_BTC_EMA20_DOWN + LITE_WEAK_HOLD blocked BULLA (11-16 09:15) — not priced here.

## Caveats
- **In-sample:** every rule except the depth guard was found or confirmed partly on these same live fills → apply the 30–50 % haircut: realistic "after" ≈ −$1,250 … −$800 → account ≈ $8,850 … $9,300.
- Blocked winners are counted (INJ, ONE short); the blocked SOPH LITE winner (live) is not priced.
- No second-order effects: freed slots / capital could have opened other trades (good or bad); capped-entry slippage savings not modelled.
- Not addressed by any rule: WILLY_HT INIT −$439, AAVE momentum long −$303, AVA fade −$101.
