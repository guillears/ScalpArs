# ROUTE_B_PEAK vs REAL EXIT — 2026-11-22 (follow-up to ROUTE_B_PEAK_STUDY_2026-11-21)

**Question (operator):** "We should test it vs our current real exit for each trade for each sleeve" · then: "WR before and after per sleeve, net $ and avg P&L, how would the rule be defined, could it live together with the current exit?" · "avg duration before and after" · "review again per sleeve the best market exit signal and build the table vs today's exit".

**Prior-study check:** ROUTE_B_PEAK_STUDY (11-21) priced the exit only for screen cells ≥ 60 % precision. STATE_EXIT (11-21) priced the candle signals and "close at +3.03 %" on the 49-fill HT master (TP +3: −0.73 %/fill). **New here:** every pre-registered order-flow and candle variable, plus the price-only controls and a plain "close at ½ TP" benchmark, priced as an add-on exit against each fill's exit, every sleeve, REAL and SHADOW.

**Answer:**
- **EXH:** on the 4 REAL fills re-priced under **today's TP 7**, today's exit (+3.94 %/fill) beats every signal; the best signals tie it or cut winners.
- **DIST:** on REAL fills, every signal that fired cut the CYBER +9.95 TP winner.
- **HT:** the best signal is the plain "close at ½ TP" (= TP +3) in disguise, and that exit loses −0.73 %/fill on the 49-fill master.
- **SHADOW (priced blocked signals, today's exits already):** some signals beat both today and ½-TP on EXH. BOOK_FLIP +0.24 and CAPITULATION_LIQ +0.11, vs −0.10 booked and −0.26 for ½-TP. The evidence is thin: 2–3 days, 16 rules tried, and the CI rests on 2–3 day blocks.

**No exit change. No observe line in code.** One pre-registered re-read at the end of the freeze (below). All numbers UNVERIFIED.

**Correction log (same day):**
- The first version compared signals against the **booked** REAL exits. Those include older TP levels (EXH TP 10, HT TP 5), and it said EXH REAL signals beat ½-TP and today.
- The deep review caught it. The REAL EXH / HT fills are re-priced below under today's TP 7 / TP 6.063.
- Result: today's exit wins on EXH REAL. The HT REAL ½-TP gap shrinks from +2.36 to +2.12 (CAPITULATION_LIQ +2.38 → +2.14).

## Method
- **Add-on rule:** TP, stop and max hold stay as today. Once the trade is ≥ ½ its TP in profit (EXH 3.5 % · HT ≈ 3.03 % · DIST ½ the row's ATR TP), close at market the first closed minute the signal fires.
- **Cost:** 2 × taker + measured book slippage (0.10 % when no book was recorded).
- **Benchmark:** plain "close at ½ TP" (C_HALF_TP). This is the yardstick for "does the order flow add anything beyond taking profit earlier?".
- **REAL fills re-priced to today's exit** (`route_b_peak_1122_reprice_today.py`):
  - EXH TP 7 % (cap) and HT TP 6.063 % gross; a TP is reached on the bar's favourable extreme, net = TP − 0.09.
  - Stops as booked.
  - HT fills booked at TP 5 keep walking up to 30 bars after their booked exit; all 4 reached TP 6.
  - A signal counts only if it fires before today's TP / stop minute.
  - DIST REAL ran today's TP (10 % cap), so no re-price was needed.
- **SHADOW** = blocked signals priced by the signal pricer under today's exits.
- **CI** = 95 % day-block bootstrap (2–3 days: not meaningful). LOO = leave-one-fill-out.

## REAL fills, re-priced under today's TP (WR · avg %/fill · net $ at each fill's own notional · avg hold)
| sleeve | exit | WR | avg % | net $ | hold |
|---|---|---|---|---|---|
| EXH (4) | booked (TP 10 era) | 50 % | +0.72 | +$126 | 6.0 h |
| | **today (TP 7)** | **75 %** | **+3.94** | **+$993** | 1.1 h |
| | LH2 | 75 % | +3.34 | +$865 | 0.7 h |
| | TAKER_FADE | 75 % | +3.14 | +$823 | 0.7 h |
| | CAPITULATION_LIQ | 75 % | +2.74 | +$679 | 0.9 h |
| | plain ½-TP | 75 % | +1.78 | +$470 | 0.5 h |
| DIST (3) | today (= booked) | 33 % | −0.73 | −$170 | 1.0 h |
| | every signal that fired (1 fill: CYBER) | 33 % | −1.6 … −2.4 | −$460 | 0.6 h |
| HT (8) | booked (TP 5 era) | 50 % | +0.38 | +$191 | 1.5 h |
| | **today (TP 6)** | 50 % | +0.87 | +$434 | 1.6 h |
| | CAPITULATION_LIQ | 88 % | +3.00 | +$1,515 | 1.1 h |
| | plain ½-TP | 88 % | +2.98 | +$1,506 | 1.1 h |
| FLATBASE (1) | today | 0 % | −11.69 | −$792 | 5.1 h — never armed |

Per fill (today → signal):
- **EXH:**
  - RARE (booked TP +5.31, TP level at close unclear): today +6.91; CAPITULATION_LIQ +6.91 (it fires only after TP 7) · ½-TP +3.70.
  - RARE (stop −7.42): today **+6.91** — under TP 7 it is a TP win; CAPITULATION_LIQ +4.38.
  - KERNEL (+9.95): today +6.91; CAPITULATION_LIQ +4.62.
  - RARE (stop −4.95): unchanged.
- **HT:** the 3 stops that first reached +3 % (ENSO, EVAA, BLESS) are rescued to +3.07 … +3.55, and 2 TP-6 winners are cut to +3.0. The liquidation signal fires at the same minute as ½-TP on 4 of 5 fills (BLESS 1 min later).

## SHADOW (priced blocked signals, today's exits) — best signal per sleeve, all 16 rules (picked in-sample → 30–50 % haircut)
| sleeve · N | today: WR · avg · $ · hold | best signal (fills changed) | best: WR · avg · $ · hold | Δ avg (CI) · LOO | ½-TP Δ |
|---|---|---|---|---|---|
| EXH · 62 | 37 % · −0.10 · −$350 · 3.6 h | BOOK_FLIP (6) | 42 % · +0.24 · +$872 · 2.8 h | +0.34 (+0.25 … +0.41) · +0.16 … +0.39 | −0.16 |
| | | CAPITULATION_LIQ (17), pre-registered | 44 % · +0.11 · +$391 · 3.1 h | +0.20 (−0.07 … +0.45) · +0.03 … +0.27 | |
| DIST · 77 | 26 % · −0.74 · −$3,409 · 5.9 h | TAKER_FADE (23) | 34 % · −0.15 · −$672 · 3.2 h | +0.59 (0 … +0.65) · +0.38 … +0.67 | +0.48 |
| HT · 7 | 29 % · −1.42 · −$615 · 1.2 h | OI_NO_NEW_MONEY (2) | 57 % · +0.73 · +$317 · 0.7 h | +2.16 · +1.25 … +2.51 | +1.34 |
| FB · 1 | never armed | — | — | 0 | 0 |

SHADOW $ uses the sleeve's median REAL notional: EXH $5.8k · DIST $6.0k · HT $6.2k. The DIST ½-TP +0.48 is over all 77 signals; the 11-21 report's "+0.63" is the same exit over its 59 eligible signals.

## What it means
- **Win rate always rises with an early exit** (stops that first reached ½ TP are rescued), but big TP winners are cut. Judge by avg %, not WR.
- **EXH:** under today's TP 7, the real fills already take profit at +6.9, and every signal either ties or exits earlier for less. The SHADOW gain (BOOK_FLIP / CAPITULATION_LIQ) is the only hint that order flow adds something, and it rests on 2–3 days.
- **HT:** the signals equal "close at TP +3". This batch had 3 stops that first reached +3 %. On the 49-fill master the same exit loses −0.73 %/fill (CI −1.11 … −0.26). This is a TP-level question (DL 608), not an order-flow finding. There is no liquidation data on the master (recorder from 11-19), so only the +3 % exit could be tested there.
- **DIST:** the early exit cuts its rare big winners on REAL fills. On SHADOW the gain is mostly "take profit at ½ TP", and the result stays negative.

## Verdict
- No exit change, no observe line in code.
- **Pre-registered for the end of the freeze, REAL fills only, frozen definitions:**
  1. **EXH + CAPITULATION_LIQ add-on exit** (≥ $5k long liquidations on the coin summed over the last 3 minutes, armed at ≥ +3.5 %). It must beat both today's exit (TP 7) and the plain ½-TP exit on avg %/fill; below that it is retired.
  2. **WILLY_HT:** does CAPITULATION_LIQ let the fills that go on to TP 6 run, or cut them like the plain TP +3?
- BOOK_FLIP / LH2 stay secondary reads, not new frozen tests: freezing this batch's winners is the in-sample trap.
- Run `route_b_peak_1121_*` and then `route_b_peak_1122_{vs_real_exit,wr_table,reprice_today}.py`.

## Blind spots
- 2–3 days, one regime; ≤ 8 REAL fills per sleeve.
- EXH TP is taken as the 7 % cap (an ATR TP below 7 % is possible). Intrabar TP-before-stop order is assumed from bar extremes.
- HT TP-5 fills were walked ≤ 30 bars past their booked exit, and signals there are not evaluated.
- In-trade funding is not charged to the new exit. 0.10 % slippage is assumed where no book was recorded.
- SHADOW = priced blocked signals.
- The liquidation stream is a sample. The full book (from 11-21 22:00) and sub-minute data are not tested.
- $ joins the newest orders export (a later export can change the $ column; % is unaffected).
- The booked REAL RARE TP fill (+5.31 at 408 min) does not match its path (TP 7 reached at 108 min under the stamped TP). Its TP era / level at close is unclear; the re-price uses today's TP 7.

Files:
- `scripts/route_b_peak_1122_vs_real_exit.py` → `reports/ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22.csv` (per sleeve × variable vs the booked exit) · `_trades.csv` (per fill × variable, with hold minutes)
- `scripts/route_b_peak_1122_wr_table.py` → `_wr.csv` (WR · avg · $ · hold, booked basis)
- `scripts/route_b_peak_1122_reprice_today.py` → `_today.csv` · `_today_summary.csv` (REAL EXH / HT under today's TP)

Decision: DECISION_LOG 626.
