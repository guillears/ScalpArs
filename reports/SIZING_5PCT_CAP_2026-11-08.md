# FRENZY + Flatbase sizing at the 5 % cap — analysis 2026-11-08 (DECISION_LOG 502 / 503)

Operator, 2026-11-08: "why Flatbase at 0.5 and not 1? why FRENZY_LONG ×5 and not ×8? we should re think all frenzy leverage levels" →
"risk limits ok" → "save this analysis on sizing and add it as watchlist item for each sleeve". **Nothing in the bot changed.** These are
provisional targets for the B25 review (plan ②y3); each step waits for its sleeve's own line gate and the joint review study (all sleeves
together, 30 % drawdown check).

Source: export `scalpars_orders_paper_2026-11-09_01-43-55.csv`, current batch B25 (opened ≥ 2026-10-23 18:30 UTC), best case
`reports/best_case/BEST_CASE_2026-11-09_01-43.md` (one size, DL 503). In-sample: the 30–50 % cut applies; master history is added at the review.

## 🔒 Operator risk limits (confirmed 2026-11-08)
- One trade's loss ≤ **5 %** of the account.
- Deepest peak-to-trough < **30 %**.
- Whether they also bring TODAY's sizes down is judged per sleeve on its whole history (WR vs breakeven, payoff, ½ Kelly, losing streaks,
  worst day, share of the drawdown) — never on one stop.

## How a FRENZY / Flatbase position is sized
position = margin slot (working capital ÷ 4 positions — ≈ 0.244 of equity, ≈ $1,635 at ≈ $6.7k) × stake (`*_invest_mult`) × leverage (20 × `*_lev_mult`).
The slot follows the account, so a stop costs the same % of the account as it grows — **below $10k**: above it the `reserve_schedule` (10000:9750 …) holds part of the account back and `leverage_balance_schedule` drops the base 20× to 15× at $25k, so the factors must be re-read then.
**Binding caps not in the factors:** Binance leverage brackets and the per-pair notional cap (`entry_bracket_cap_notional` / `entry_liquidity_cap_notional`) already cut ≈ 35 % of LITE fills (6 to 5× by the bracket, 5 at a $9.5k notional cap); its X 1.94 is the uncapped size (mean ≈ 1.75).

| Sleeve | Stake | Leverage setting → actual | Typical position | Today's size X (× account) | One stop costs |
|---|---|---|---|---|---|
| LITE | 1.0 | 0.4 → 8× | ≈ $11.9k | 1.94 | **≈ −5.8 %** (over the cap) |
| LITE RETRY | 1.0 | 0.2 → 4× | ≈ $6.2k | 0.98 | ≈ −2.9 % |
| WILLY | 0.5 | 0.25 → 5× | ≈ $4k (0.61 × equity) | 0.61 | ≈ −2.4 % |
| Flatbase | 0.2 | 0.25 → 5× | ≈ $1.7k | 0.24 | ≈ −2.4 % |
| WILLY_HT | — | → 5× | — | 0.31 | ≈ −1.2 % |
| FRENZY_LONG | 0.2 | 0.05 → 1× | ≈ $0.3–0.5k | 0.05 | ≈ −0.2 % |

X = median notional ÷ the account equity when each fill opened (scripts/best_case/one_size.py sleeve_x). One stop = X × stop %.

## Stake vs leverage — the rule
1. The 5 % cap sets the POSITION: position ≤ 5 % × account ÷ stop % (≈ $11k for a −3 % stop, ≈ $3.3k for a −10 % stop at ≈ $6.7k).
2. Liquidation must sit ≥ 2× the stop distance away → leverage ceiling ≈ 15× (−3 % stop) · ≈ 12× (−4 %) · **≈ 5× (−10 %, Flatbase)**.
3. Below the ceiling, prefer more stake and less leverage; the stake can't exceed 1.0 (one slot).

## Per-sleeve read (current batch, best-case line on, today's exits)

| Sleeve | Best-case line | Trades · WR · avg % | Breakeven WR | Cautious WR (90 % lower bound) | Kelly at cautious WR | ½ Kelly | Verdict |
|---|---|---|---|---|---|---|---|
| RETRY | RETRY_BTC1H | 17 · 88 % · +2.30 | 50 % | 70 % | +0.39 | +20 % | the 5 % cap binds → **step up ×1.7** |
| Flatbase | FB_CONFIRM | 20 · 95 % · +4.19 | 67 % | 80 % | +0.41 | +21 % | the 5 % cap binds → **step up ×2.05** |
| LITE | LITE_LATE_BTCFALL + LITE_BTC1H_RISING | 15 · 73 % · +1.40 | 50 % | 52 % | +0.04 | +2 % | the 5 % cap binds the other way: one stop is 5.8 % → **cap down ×0.86** (both lines NOT ABOVE LUCK) |
| WILLY_HT | none helps | 34 · 68 % · +0.38 | 58 % (actual payoff) | 48 % | negative | — | hold, no step up |
| WILLY | WILLY_GAP | 13 · 62 % · −0.60 (−$798) | 72 % | 40 % | negative | — | **below breakeven — no step up** |
| FRENZY_LONG | LONG_ADX35 | 9 · 56 % · −0.95 (−$34) | 73 % | 31 % | negative | — | **below breakeven — no step up** |

Kelly f = WR − (1 − WR) ÷ R, with **R = target ÷ stop (nominal: LITE / RETRY 3 ÷ 3, Flatbase 5 ÷ 10, WILLY / LONG ≈ 1.5 ÷ 4)**; WILLY_HT uses its actual avg win ÷ avg loss (its V2b target varies). f = the share of the account one stop may cost. Small samples → the cautious WR, then ½.

## Targets that fit the 5 % cap

| Sleeve | Today | Target | Why |
|---|---|---|---|
| LITE RETRY | 1.0 × 4× | **1.0 × 7× (lev setting ≈ 0.35)** | ×1.7 the position; 7× below the 15× liquidation ceiling — but Binance brackets / the notional cap will trim some fills (as on LITE), so ×1.7 is an upper bound |
| LITE | 1.0 × 8× | **1.0 × 7× (lev setting ≈ 0.35)** | ×0.86, one stop under 5 % |
| Flatbase | 0.2 × 5× | **0.4 × 5× (stake ×2)** | the −10 % stop allows no more than 5×; 0.2 × 10× would put liquidation at the stop |
| WILLY · WILLY_HT · FRENZY_LONG | — | **no change** | below breakeven (WILLY, LONG) / not proven (HT) — sizing up can only lose more |

Operator example checked: RETRY 1.0 × 10× (setting 0.5) → position ≈ $15.5k → one stop ≈ −7 % → over the cap.

## Effect on the current batch (daily compound %/day, one size, best-case lines on)

| Sleeve | Factor | Best line, today's size → 5 % cap size |
|---|---|---|
| LITE RETRY | ×1.70 | +2.08 → **+3.48** |
| Flatbase | ×2.05 | +1.13 → **+2.30** |
| LITE | ×0.86 | +2.12 → **+1.84** |
| Total (② "watch lines only" → "+ 5 % cap") | | +7.74 → **+10.08** (luck-checked +4.33 → +5.26) |

Without the lines the step-ups barely help or hurt: Flatbase on ALL trades (today's size −0.06 %/day) loses more at ×2.05; RETRY on ALL trades
(+0.27) gains only a little at ×1.7. So every step UP waits for its line's gate; the LITE cap DOWN is a risk limit — the best case applies it at once,
the live change is an operator call at the review. Momentum Long's X 0.25 is its 1× fills (21 of 45 ran at the 1.5× cell) — sizes are de-multiplied.

## Gates (in CLAUDE_CURRENT_STATE.md, one watch item per sleeve)
RETRY ×1.7 ← RETRY_BTC1H frozen read (≈ Nov 13) · Flatbase stake 0.4 ← FB_CONFIRM gate (≈ Nov 17–19) · LITE ×0.86 ← operator call at the
review (a size-down) · WILLY / FRENZY_LONG ← above breakeven first (WILLY_GAP ≈ Nov 14, LONG_ADX35 ≈ Nov 22) · WILLY_HT ← a line that helps.
Then the ②y3 joint study (all sleeves together, 30 % drawdown).

---

# Non-FRENZY sleeves at the same premise (2026-11-09, DECISION_LOG 504)

Operator: "with the same sizing premise … re check the other sleeves (non frenzy ones), except Surge … and Bear-Run" → "save it and add watch
lines for each sleeve" → "in ② only add Momentum Long". Same export and method as above (one size; cautious WR = 90 % Wilson lower bound;
Kelly with R = actual avg win ÷ avg loss — these sleeves have variable exits; the 5 % cap on a typical stop = the median booked STOP loss).

## Today's size and risk

| Sleeve | Sized today | Median traded notional (batch) | X (× account) | Typical stop | One stop costs |
|---|---|---|---|---|---|
| Momentum Long | stake 0.2 × 5× | ≈ $2.3k | 0.25 | −0.70 % | ≈ 0.17 % |
| Momentum Short | full cell, 20× | ≈ $43k | 4.88 | −0.75 % ¹ | ≈ 3.7 % |
| Spike Fade | stake 2.0 × 15× | ≈ $16k | 2.33 | −1.49 % | ≈ 3.5 % |
| FAN flips | full cell, 10× | ≈ $6.8k (pre-top-up fills; ≈ $16k at today's ≈ $6.6k) | 2.44 | −0.95 % | ≈ 2.3 % |

## WR vs breakeven and Kelly

| Sleeve (best line) | Current batch best case: N · WR · avg % | Breakeven | Cautious WR | ½ Kelly (cautious) | Master: N · WR · avg % (BE) | Master ½ Kelly |
|---|---|---|---|---|---|---|
| Momentum Long (ML_MKT_HOT) | 27 · 74 % · +0.31 | 53 % | 58 % | +6.3 % | 158 · 71 % · +0.12 (61 %) | +4 % |
| Momentum Short (no line) | 5 · 100 % · +0.39 | – | – | – | 41 · 76 % · +0.19 (59 %) | ≈ +6 % |
| Spike Fade (OBS-F1) | 16 · 94 % · +0.41 | 74 % | 76 % | +4.9 % | 76 · 87 % · +0.31 (71 %) | +14 % |
| FAN flips (exit TP +2) | 7 · 71 % · +1.15 | 33 % | 41 % | +5.9 % | 35 · 51 % · +0.50 (TP +2; BE 35 %) | +2 % |

## Steps and current-batch effect (daily compound %/day, one size, best line on)

| Sleeve | What binds | Step | Best case today → step |
|---|---|---|---|
| Momentum Long | Kelly + clusters (cap allows ×29) | **×5 (stake 0.2 → 1.0 at 5×), then ×10** — in ② (C3 / C4) | +0.11 → +0.54 (×5) → +1.04 (×10) |
| Spike Fade | the 5 % cap = ½ Kelly (×1.4) | ×1.4 (stake ≈ 2.8, or 15× → ≈ 20×) — watch only | +0.84 → +1.17 |
| Momentum Short | the 5 % cap (×1.37); 5 current trades | hold — watch only | +0.52 → +0.71 at ×1.36 |
| FAN flips | thin TP +2 evidence (master ½ Kelly ≈ 2 %) | hold — watch only | +1.06 |

¹ Momentum Short had no stop in the current batch: −0.75 % is the median of only 3 master stops.

**Momentum Long — how the step is declared (deep review, DL 504):**
- It is a **sleeve re-size (a part-reversal of DL 289, which cut it from 10× to 1× on 10-14), not a Pattern W cell multiplier** — the Pattern W
  staging (1.5× first, 2.0× after +50) does not apply; it is staged on its own: ×5 = stake 1.0 at 5× (the full slot margin) → ×10 = stake 1.0
  at 10× (`momentum_long_lev_mult` 0.25 → 0.5, the pre-DL 289 setting). Stake can't exceed 1.0 (DL 298); 4 open longs at stake 1.0 use ≈ the
  whole book's margin (4 slots × ≈ 0.244).
- **Gate:** ML_MKT_HOT must PASS its own test (checkpoint ≈ Nov 20, deciding test "blocked < kept" ≈ Dec 26) AND stop reading NOT ABOVE LUCK.
  Without the line the current batch is 45 · 62 % · +0.06 — cautious WR 50 % vs 57 % breakeven, cautious Kelly negative — so the current batch
  supports **no** step up unless the line passes. Master alone (158 · 71 % vs 61 %) supports a modest one.
- **🔒 Pre-committed revert gate:** the first 15 ×5 fills — WR < 61 % (master breakeven) or avg < 0 → back to stake 0.2. ×10 only if those 15
  hold WR ≥ 61 % ∧ avg > 0, and only after the ②y3 joint study (30 % drawdown).
- **Clusters:** up to 4 longs opened within 10 min this batch (10-27, 11-04 — not all stopped); the clusters that ALL stopped (11-06, 11-09) were
  3 longs each — at ×10 the 11-06 one alone (−0.76 / −0.94 / −1.00) ≈ 6.8 % of the account, above master's cautious ½ Kelly (≈ 4.1 %); ×5 ≈ 3.4 %
  stays inside. One stop: today ≈ 0.17 % · ×5 ≈ 0.87 % · ×10 ≈ 1.75 %.
All in-sample (30–50 % cut); the joint study at the review (②y3) decides with the 30 % drawdown check.
