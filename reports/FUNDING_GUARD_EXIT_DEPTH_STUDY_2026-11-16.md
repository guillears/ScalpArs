# Funding guard (A) + exit-side depth (E) — study (2026-11-16)
Operator: "something we could do to avoid high funding pairs … or to avoid the high slippage at exit?" → "yes run the studies for A and E".
Data: master MASTER_POOL_stacked kept full-size (477 fills, every sleeve) + live archives B22 / B23 / B24 / B27 (111 bot fills); funding rebuilt from Binance /fapi/v1/fundingRate for all 199 pairs (cache: scratchpad/funding/), 99.5 % of fills matched. For each fill: the NEXT settled payment after entry, minutes to it, its rate, whether the trade was still open then. Rate "against us" = long with rate > 0 / short with rate < 0. Funding interval = gap between payments (4 h 353 fills · 8 h 195 · **1 h 13**).

## A — Funding guard
- **Live funding so far:** −$30.39 net over 111 live trades (0.7 % of the −$4,394 gross live losses). INIT FAN flip alone: −$33.46 (rate −0.183 %, hourly coin, paid 1 min 36 s after the entry).
- **Extreme-funding trades are mostly WINNERS:** 58 fills where the next payment charges us ≥ 0.03 % → 86 % WR · +2.35 % avg (Flatbase squeeze shorts, LITE / RETRY pumps). A broad funding block would cut the best Frenzy / Flatbase trades.

| Cut (charges us ≥ X %, due ≤ M min) | Blocked (master + live) | Live blocked |
|---|---|---|
| X 0.03 · M 15 | 2 · 50 % · +1.45 (CAP RETRY +3.00 winner) | INIT |
| **X 0.05 · M 15** | **1 · 0 % · −0.10** | **INIT** |
| X 0.05 · M 30 | 3 · 67 % · +2.27 | INIT, BLESS LITE (+$207 winner) |
| X 0.05 · M 60 | 9 · 67 % · +1.25 | INIT, BLESS |
| X 0.10 · M 15 | 1 · 0 % · −0.10 | INIT |

**Recommended rule:** skip an entry when the NEXT funding payment charges the trade ≥ 0.05 % of notional (5 × normal) and is due within 15 minutes. Blocks only live INIT in 564 fills; master unchanged (no master fill blocked → master N · WR · DCR identical); live FAN flip 2 · 50 % · +$36 → 1 · 100 % · +$54. It is a COST guard, not an edge filter: N = 1, cannot be validated statistically; its cost when wrong is near zero because it fires ≈ 2 in 1,000 entries. Live input: Binance premiumIndex `lastFundingRate` (the predicted rate charged at `nextFundingTime`) — the study used the settled next rate as its proxy.

## E — Exit-side depth
- **Live exit slippage** (86 bot exits, + = worse than the decision price): adverse total $462, net $330. By type: trail median +0.02 % · stop +0.01 % · TP (resting limit since DL 296) 0.00 %. Outliers: KMNO stop +0.63 % ($74, 24 h vol $18 M) · EPIC stop +0.42 % · **INIT trail +0.375 % ($68, vol $113 M)** · **SHELL stop +0.30 % ($50, vol $174 M)**.
- **By 24 h volume:** < $20 M avg +0.35 % (3) · $20–100 M −0.01 % (55) · $100–500 M +0.04 % (25) · > $500 M −0.80 % (3). Correlation slip ~ notional 0.14, ~ notional / 24 h volume 0.18 → **weak: 24 h volume does not predict it**; INIT and SHELL were liquid by volume but one-sided at that minute (squeeze).
- **Coverage gap found:** the DL 551 depth guard reads the book only on the FRENZY / SURGE / FLATBASE taker path — **momentum and flips (maker entries) are never depth-checked**; INIT's $18.3 k flip had no book read. No exit-side (opposite side) depth is recorded anywhere → the effect of sizing to min(entry-side, exit-side) **cannot be measured on past fills** (no snapshots; reconstructing from today's book would be fiction).
- **Recommended:** phase 1 (observe) — read the book on EVERY bot entry (momentum and flips too) and stamp both sides' depth within 0.15 % (`entry_depth_bid_usd` / `entry_depth_ask_usd`); phase 2 — at ~30 live fills, size to min(entry side, exit side) if the exit-side depth predicts exit slippage. Biggest saver already in place: resting maker TPs (0.00 %), now also for FAN flips (+0.5 TP).

## Caveats
Funding guard = 1 live case; the rule is chosen for zero collateral, not for statistical proof. Slippage sample = 86 live exits, 4 outliers carry 54 % of the adverse $; exit-side depth unmeasured historically.
