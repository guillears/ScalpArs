# 💧 Capacity first sketch — when the liquidity cap starts cutting each sleeve (2026-11-09, DECISION_LOG 505)

Operator 2026-11-09: "for how many days could the 10 % be valid considering the max cap per trade of 0.1 % or 0.5 % volume … starting balance
10,000" → "save this table". Source: export `scalpars_orders_paper_2026-11-09_01-43-55.csv`, current batch B25; each sleeve at today's size X
(× account, one size) and its coins' 24 h volume stamp (`entry_pair_volume_24h_usd`). The engine caps a position at 0.1 % of that volume
(Spike Fade 0.5 %, `spike_lowvol_liq_cap_pct`). **Median trade only, no price impact** — the full per-sleeve study with impact is
`scripts/study_capacity_caps.py` → `reports/CAPACITY_CAPS_<export>.md`.

At 10 %/day from $10,000: day 10 ≈ $26k · day 20 ≈ $67k · day 30 ≈ $174k · day 40 ≈ $453k · day 60 ≈ $3.0M · day 90 ≈ $53M.

**Account size at which the cap starts cutting a typical trade** (median trade, current-batch coins):

| Sleeve | Coins' 24 h volume (median) | Cap at 0.1 % | Cap at 0.5 % | Position kept at $250k: 0.1 % → 0.5 % (mean across trades) |
|---|---|---|---|---|
| Spike Fade | $4M | — | ≈ $9k (already biting: 29 % of fades capped) | 1 % → 5 % |
| Momentum Short | $69M | ≈ $14k | ≈ $70k | 10 % → 51 % |
| FAN flips | $61M | ≈ $25k | ≈ $125k | 22 % → 65 % |
| **LITE** | $52M | **≈ $27k (≈ day 11)** | **≈ $135k (≈ day 27)** | 13 % → 58 % |
| WILLY | $31M | ≈ $51k | ≈ $255k | 30 % → 91 % |
| **LITE RETRY** | $55M | ≈ $56k (≈ day 18) | ≈ $282k (≈ day 35) | 39 % → 88 % |
| WILLY_HT | $66M | ≈ $213k | ≈ $1.1M | 76 % → 100 % |
| Momentum Long | $84M | ≈ $336k | ≈ $1.7M | 93 % → 100 % |
| **Flatbase** | $88M | ≈ $364k (≈ day 38) | ≈ $1.8M | 91 % → 100 % |
| FRENZY_LONG | $122M | ≈ $2.4M | ≈ $12M | 100 % |

Rough read: at 0.1 % the 10 %/day holds ≈ 2–3 weeks (LITE caps first, ≈ day 11; RETRY ≈ day 18); at 0.5 % ≈ 4–5 weeks (LITE ≈ day 27,
RETRY ≈ day 35); beyond ≈ $1M only Flatbase, Momentum Long, WILLY_HT and FRENZY_LONG still scale. A higher cap costs price impact
(paper books every fill at the line) — the trade-off the full study measures.

These account sizes use today's sizes; with the ② C3 re-sizes they come earlier (Momentum Long ≈ $67k, Flatbase ≈ $179k, RETRY ≈ $33k). The impact side — whether a higher cap pays — is unresolved: see reports/CAPACITY_CAPS_<export>.md ④ (DL 505).
