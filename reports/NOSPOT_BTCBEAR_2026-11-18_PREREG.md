# PRE-REGISTRATION — No-spot Frenzy short, ON only when BTC is bearish (2026-11-18)

**Frozen: 2026-11-18 23:05 UTC**, before any new result was computed. Addendum E (operator: exits + W/L) frozen in the same file at
the same time, also before any new result. Research only. No engine / config / test change. No engine / config change (research only; working data in the *_data folders stays local).

## Question
Is there a real, tradable short sleeve on Frenzy coins WITHOUT a Binance spot market that is switched ON only when BTC is bearish?
It must ADD to the combined short book (EXHAUSTION + Flatbase + L4 + DISTRIBUTION), not only be positive alone.

## What has ALREADY been seen (honest list — this hypothesis is born in-sample)
- NOSPOT_EXIT (`reports/NOSPOT_EXIT_2026-11-18_regime.csv`), full span 2023-07 → 2026-11, split by BTC daily R1 (EMA50 vs EMA200) and
  BTC 5m R4 (`services/regime.py` labels, bear set −1 / bull set +1 / else 0):
  - **R4 TOP_DISTRIB · TP 4 / SL 10: 5m-bear +0.37 (693) · chop +0.05 · bull −0.24 (700); daily-bear +0.16 · daily-bull −0.01.** (The source.)
  - R4 · REF (EXHAUSTION exit): 5m-bear −0.03 · bull −0.26; daily-bear +0.07 · bull −0.18.
  - R4 · time 4 h: 5m-bear +0.25 · bull −0.38. R1 · TP 4 / SL 12: 5m-bear −0.26 · bull +0.10 (opposite sign).
  - NS0 · REF: 5m-bear −0.04 · chop +0.07 · bull −0.26; daily-bear −0.02 · bull −0.08. R1 · REF: 5m-bear −0.50 · bull −0.21.
  - Half-year detail of all the above was shown in that csv.
- NOSPOT_SLEEVE: NS0 · XA by daily R1 (−0.13 bear / −0.38 bull) and 5m R4 (−0.10 / −0.25 / −0.39); DESIGN is 100 % daily-bull.
- **Contaminated cells** (their HOLDOUT is NOT fresh evidence): regime (a) 5m and (d) daily × cohorts R4, NS0, R1 × exits REF,
  TP4/SL10 (R4), TP4/SL12 / T 4 h. They are reported, but they can be at most OBSERVE, never SHIP-candidate, on this data.
- Fresh evidence must come from: regime definitions (b) (c) (e) (f) (g), cohorts R2 R3 R5–R10, the window-level statistics, the
  rotation null, leave-one-window-out, the random-entry control, the spot-coin comparison and the joint book.

## Data (reuse, nothing new except BTC 5m after 2026-11-16 16:00)
- Rows: `reports/NOSPOT_SLEEVE_data/X.pkl` (476,231 no-spot Frenzy bars, 215 pairs). Trades: the 1m sim of NOSPOT_EXIT
  (`reports/NOSPOT_EXIT_data/sim/*.npz`, every eligible bar in the 2,081 episodes where NS0 / R1 / R4 fire × 46 exits + REF).
  Rule rows outside those episodes are dropped (coverage: R3 92 %, R5 93 %, R6 94 %, R7 93 %, R9 92 %, others ≥ 97 %; reported).
- Costs exactly as NOSPOT_EXIT (taker 0.05, maker 0.02, slip max(0.10, model) per taker leg, real funding). Stress: flat 0.20.
- One position per pair, re-applied on the ON-gated set (the sleeve only trades while ON).
- BTC 5m: `leads_deep_regimes.btc5m()` + the Binance fapi for the missing tail (≤ 1,200 weight/min, header read, stop on 418/429).

## Regime definitions (frozen; all measured AT ENTRY from closed bars only; ON = bearish)
| id | ON (bearish) when | OFF-bull | note |
|---|---|---|---|
| a | `classify_btc_regime` on BTC 5m (ADX14, RSI14, EMA20 slope 3 bars) ∈ {HEALTHY_BEAR, STRONG_BEAR, BEAR_EXHAUSTED} | ∈ {HEALTHY_BULL, STRONG_BULL, BULL_EXHAUSTED} | CHOPPY_* / UNKNOWN = neither (reported, OFF) |
| b | BTC 1 h EMA20 now < EMA20 three closed 1 h bars ago | ≥ | 1 h bars from 5m, valid from the 1 h close |
| c | BTC 4 h return (last closed 5m close vs 48 bars before) < 0 | ≥ 0 | |
| d | BTC daily EMA50 < EMA200 | ≥ | daily closes, valid from the next day |
| e | BTC last closed 1 h close < 1 h EMA20 AND < 1 h EMA50 | otherwise | |
| f | BTC 24 h return < 0 | ≥ 0 | |
| g | breadth: share of USDT perps with daily close > daily EMA50 < 50 % (regimes.pkl R5) | ≥ 50 % | ≥ 30 pairs |
Sign granularity only. No threshold is tuned.

## Cohorts
NS0 (EXH trigger on no-spot rows), R1 … R10 (NOSPOT_SLEEVE primary masks, frozen). **Control:** RND = every eligible bar in the same
episodes (no one-position), split by the same switch. **Comparison:** the spot-coin EXHAUSTION trades (BASELINE_TRADES_EXHAUSTION, dl582
kept, 1m, same costs) split by the same switch.

## Exits (main family)
Primary: REF = shipped EXHAUSTION exit TP min(20 × ATR15, 10 %) / SL min(1.5 × ATR15, 6 %) / 24 h. Secondary: A TP 4 / SL 10 / 24 h.

## Windows
**Window = UTC calendar day of entry on which the cell has ≥ 1 ON trade.** Contiguous ON episodes are also counted (descriptive).

## Splits
DESIGN 2023-07-01 → 2025-07-01 · HOLDOUT 2025-07-01 → 2026-11-18, HOLDOUT read once per cell. Bear-window counts per split reported;
(d) has ~0 DESIGN windows (DESIGN 100 % daily-bull) — it is judged by HOLDOUT + leave-one-window-out + the null only.

## Ship bar (all on HOLDOUT ON trades unless stated)
1. ≥ 15 distinct windows. 2. Window-clustered bootstrap (resample ON-days, 2,000) 95 % lower bound of avg > 0.
3. No window ≥ 25 % of profit; no pair ≥ 50 %. 4. Positive in ≥ 2/3 of half-years (full span) with ≥ 5 ON trades.
5. avg > 0 at flat 0.20 % slip. 6. Kelly at the 95 % Wilson WR floor > 0 (trade level).
7. Family-wise rotation null p_fw < 0.05 (see below). 8. Joint-book add > 0 (HOLDOUT DCR of all four + candidate > all four) with no
worse worst day. 9. DESIGN ON avg > 0 where DESIGN has ≥ 10 ON windows (else "untestable": then leave-one-window-out min > 0 required).
10. Entry edge vs market effect: cohort-ON avg − RND-ON avg > 0 (same exit). Item 10 decides whether a pass is an ENTRY edge or a
market effect; a pure market effect is still reported but is not credited to the rule.

**Null:** circular rotation of the BTC regime timeline vs the trades (offset uniform in [14 d, span − 14 d], wrapped within HOLDOUT),
1,000 draws, keeps the regime's autocorrelation and duty cycle. Statistic per cell = ON avg − cohort avg (HOLDOUT). Family = every
definition × cohort × exit of the main family; p_fw = share of draws whose family max ≥ the cell's observed statistic.
**Leave-one-window-out:** min over the top-10 profit days of the avg with that day removed; plus drop-top-pair.

## Verdict rule
- SHIP-candidate: a NON-contaminated cell passes all 10 items (then 30–50 % haircut, revert gate).
- OBSERVE: a cell passes 1, 2, 3 and 7, but fails others or is contaminated → frozen rule + exit + a forward read at 30 new ON windows.
- NOTHING otherwise.

## Analyses (all reported regardless of result)
Per definition × cohort × exit: N, windows, trades/day while ON, WR, avg, per-window distribution; path decomposition ON vs OFF (squeeze
share); W/L inside the bear-ON cohort; per-pair / per-window concentration; LOWO; RND control; spot comparison; duty cycle and trades/day
added; one-size DCR at 0.61 alone; the joint add via `scripts/joint_short_sim.py` (in-sample, haircut 30 %, 50 %); the worst day and
stop clustering vs EXHAUSTION on ON days; this week's replays with the BTC regime at each signal.

---

## ADDENDUM E — exit family and W/L for the bear-ON cohort (operator, frozen 2026-11-18 23:05 UTC, before any result)

**E1. Which (definition, cohort) gets the exit study — mechanical, decided before any exit result is seen.**
Rank every (definition × cohort) pair by the **DESIGN window-level t** of its ON trades under REF: t = mean of the per-day mean net ÷
(sd of per-day means ÷ √days), DESIGN ON-days only, requiring ≥ 10 DESIGN ON windows. Take the top 2 pairs (at most one per
definition). If none has ≥ 10 DESIGN windows, take the top 2 by HOLDOUT-free full-DESIGN t anyway and say so.

**E2. Exit family:** all 46 selectable NOSPOT_EXIT exits (A fixed TP 2–5 × SL 6/10/12; B break-even 2/3/4; C profit lock 3→1, 4→2, 5→3;
D trails arm 2/3/4 × give-back 1.5/3 and arm 3 × 1/2 ATR; T time 4/12/24/48 h; S TP 1/1.5 × SL 2/3/4/6 × 4/24 h) + REF.
Reused from `scripts/nospot_exit_sim.py` output — nothing re-simulated.

**E3. Selection on DESIGN ON trades only:** N ≥ 30 · avg > 0 · avg > REF · ≥ 2/3 DESIGN half-years (≥ 5 ON trades) positive · ≥ 2/3 grid
neighbours positive (NOSPOT_EXIT neighbour map). Pick = the highest DESIGN avg survivor per chosen pair; HOLDOUT read once.

**E4. Rigour for the picks:** the full ship bar above; family-wise rotation null over regime × cohort × exit (all 46 + REF for the chosen
pairs); the random-entry control under the same regime and exit; saved vs deeper vs REF; Δ = 0 on fills where a BE / lock / trail never
armed; window units; flat 0.20 % slip.

**E5. W/L (joint check).** Inside the bear-ON cohort of each chosen pair: every numeric entry-time column of X.pkl (coverage table first),
3 granularities (sign/median split, terciles, quintiles), statistic = best kept-bin avg − cohort avg; family-wise label-shuffle null (500)
on DESIGN. A separator survives at p_fw < 0.05 on DESIGN with the same sign under BOTH REF and the picked exit. Then HOLDOUT once.
A sign flip between the REF read and the picked-exit read, or between DESIGN and HOLDOUT = no ship.

**E6. Final combined candidate** = switch + entry + (surviving W/L filter, if any) + picked exit: trades/day while ON and averaged, DCR
at 0.61 alone, the joint-book add (in-sample / 30 % / 50 % haircut), the worst day, Kelly at the floor, then the verdict by the rule above.
