# FLATBASE theory check — is the trigger shorting a coin that was really FLAT? (2026-11-27, research only)

**Question (operator):** Flatbase is supposed to short a coin that pumped after being FLAT. Is the trigger's "flat" right, and does flatness
split winners from losers (master, current batch, backtest)?

**Theory.** A coin with no trend and little day-to-day noise that pumps on a one-off event has no structural buyer → once the pump fails it
drifts back toward the base. Today's FLAT = the 14-day close range ≤ 44.4 % (flat14). A 44 % range is wide: a coin can slide, climb or swing
every day inside it.

**Prior:** flat14 as a W/L variable (OOS_REAL 11-17, inconsistent) · DL 649 falling base (no edge) · entry timing (keep). **New:** six
"flat" measures, direction pre-registered before any result (scripts/FLATBASE_THEORY_2026-11-27.py). Backtest 545 (today's stack + exit),
real 21 (master 15 + batch 4 kept + MOVE / LAYER open; G / CLANKER / MMT not mappable to a rebuild episode).

| Measure (theory) | Best third (N · WR · avg) | Worst third | Gap before / after mid-2025 | Half-years agreeing | Real: theory side vs other |
|---|---|---|---|---|---|
| 14-day range (today's flat14), lower better | 182 · 75 % · +1.01 | 182 · 77 % · **+1.44** | +2.25 / **−0.89** | 3 / 5 | 5 · 40 % · −2.57 vs 16 · 88 % · +3.00 (**opposite**) |
| **Base daily volatility, lower better** | **182 · 81 % · +1.79** | 182 · 71 % · +0.48 | **+2.05 / +1.11** | **5 / 5** | **11 · 82 % · +2.37 vs 10 · 70 % · +0.90** |
| \|base trend\|, lower better | 182 · 78 % · +1.52 | 182 · 77 % · +1.41 | +1.49 / −0.08 | 2 / 5 | 10 · +2.22 vs 11 · +1.17 |
| Quiet-day share, higher better | 140 · 78 % · +1.37 | 208 · 75 % · +1.10 | +4.11 / −0.19 | 2 / 5 | 11 · +0.11 vs 10 · +3.39 (opposite) |
| Trend R², lower better | 182 · 71 % · +0.45 | 182 · 79 % · +1.59 | −2.42 / −0.98 | 1 / 5 | opposite |
| Pump size ÷ base volatility, higher better | 182 · 80 % · +1.72 | 182 · 72 % · +0.65 | +0.22 / +1.33 | 4 / 5 | 14 · 86 % · +2.74 vs 7 · 57 % · −0.46 |

Base daily volatility tercile middle 181 · +1.13 → monotone (+1.79 / +1.13 / +0.48). Luck (all 6 measures, labels shuffled 2,000×): best
gap +1.31 → family-wise p **0.12** (pump ÷ noise p 0.23).

**As a filter (block noisy bases):** the noisy side still earns (vol > 5.0 %: 165 · 71 % · +0.40 > breakeven 67.9 %); blocking it lowers the
daily compound (0.375 → 0.357 %/day at > 5.0; 0.344 at the tercile) while max drawdown improves (−41.3 → −31.4 / −28.3 %). Real fills at the
tercile cut disagree (the 6 "noisy" real fills averaged +2.40). → **not a loser filter.**

**The batch losers on these measures:** BIGTIME vol 3.87 / pump÷noise 19.9 · SLX 5.69 / 4.46 · KERNEL 4.60 / 9.07 · LAYER 3.51 / 3.72 · MOVE
4.76 / 7.56 — mixed; flatness does not explain this batch either.

## Meaning
The theory is half right: what predicts a Flatbase win is a **quiet base** (low day-to-day volatility, and a pump that is large against that
noise), not a **narrow range**. Today's range test (≤ 44.4 %) does not measure flatness — wider-range bases did no worse. But the noisy bases
still make money, so the lever is conviction (size), not a block.

## Verdict
- Keep the trigger; no filter (the blocked side is profitable and the daily compound drops).
- **Candidate: a size multiplier for quiet bases** (base daily volatility in the low third ≤ ≈ 3.7 %): backtest 182 · 81 % · +1.79 passes the
  Pattern-W shape (N ≥ 30 · WR ≥ 70 % · avg ≥ +0.10 · Σ > 0) but only 11 real fills support it and the luck p is 0.12 → OBSERVE line first
  (frozen cut, real fills), multiplier only after it meets the bar on real fills.
