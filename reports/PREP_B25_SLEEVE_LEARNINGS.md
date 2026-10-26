# B25 review prep — learnings box per sleeve (plan §0a)

Built 2026-10-26 from CLAUDE.md · CLAUDE_CURRENT_STATE.md (CS) · reports/NEXT_BATCH_REVIEW_PLAN.md · scripts/batch_review_checklist.py ·
DECISION_LOG (DL, grepped per sleeve) · memory notes · trading_config.json (live values checked) · reports/MASTER_POOL_stacked.csv
(STACK 2026-10-26a, stack-kept, non-probe, stack_pct) · scout gate state reports/SCOUT_REVERT_GATES.json (10-26 12:41 UTC) and
reports/SCOUT_REPORT_latest.md (10-26 11:18 UTC; order exports only up to 10-24 10:06 UTC, so the 10-26 B25 export is NOT in these counts).
Counts are a starting point. Re-run every scout on the B25 export before you quote any of them.
Order = the operator's stake order (DL 376). Master figures are IN-SAMPLE (today's filters were fitted on those fills) and do not count
as evidence for a gate. **No persisted verdict file exists yet** (.willy_exit_v2b_gate / .scout_fade_watch / .flatbase_tp_shadow_gate /
.willy_latejoin_gate are absent), so every frozen read below is still open unless it says "decided".

---

## 1. SPIKE_FADE (short the 5m RSI explosion, pair ADX > 30 route)
- **Live config.** Entry gates:
  - BTC 5m RSI ≤ 50 (DL 112)
  - BTC ≤ 5m EMA13, called BD13 (08-04)
  - FRESHBREAK: block when rsi_prev < 44 ∧ pgap > −0.40 (41c)
  - 24 h volume < $20M (DL 55)
  - LAGGARD: block when daily −DI > 16.1 ∧ BTC 4h EMA50 > EMA200 (DL 128)
- **Live exit:** SL −1.5 (DL 11), tripwire −2.5 alert-only, no EMA13-cross exit (DL 54), exempt from the spike lock (41f), runner arm +0.40, late-arm at +0.30 after 15 min (DL 111).
- **Size:** invest 2.0 × lev 1.0. Liquidity cap 0.5 % of 24 h volume for every fade (DL 119 / 165).
- **Master:** 76 · 86.8 % · +0.307 %/fill · +$4,237.
  - B25: 7 fades · −$396 (RAYSOL −$304, 1000LUNC −$308).
  - Breakeven WR ≈ 72 % (avg win +0.55 vs avg loss −1.39).
- **Open gates** (counts at registration):
  - DL 374 watch lines, read forward on the expectancy bar at N ≥ 15 on ≥ 8 days. All at 0 (cohort starts at the DL 374 commit):
    - 41k ADX Δ < 2.359
    - 41l pair ADX ≤ 19.2
    - 41o HB ∧ ADX Δ ≤ 2.4341
    - 41p HB ∧ BTC ADX > 24.5874
    - 41n HB size shadow: ≥ 15 fades on 8 days, avg < 0 ∧ P ≥ 0.80 → SIZE_CUT_CANDIDATE.
  - ⚠ **DL 311 / 41i mcap gate TRIGGERED 10-25** (1000LUNC, $249M, −10.6 % of the account). The size question is reopened:
    - 41m 1.5× cap: decided at the first 10 fades ≥ $200M on avg % < 0.
    - DL 312 rule "1× if ≥ $300M": did not fire.
  - 41h HEALTHY_BEAR ∧ trend-gap ≥ −0.13: SHIP at N ≥ 20 ∧ WR ≤ 55 ∧ Σ < 0. Last tally 13 · 62 % · −$244 (09-20).
  - LAGGARD money revert (DL 334): next 10 blocked fades from 10-21 02:14 with Σ > 0 → off. Read before CloudWatch's 14-day expiry.
  - Late-arm (DL 332): first 8 LIVE late exits, Σ < 0 ∨ cuts > rescues ∨ any cut > $100 → off. Now 1/8.
  - Live execution (DL 327): first 10 live fades avg < −0.5 % or stop slip above the assumption → review. Live 0/2.
- **Decided:**
  - BTC-RSI 50 re-revert (DL 112 / 260): FROZEN 10/10 · 90 % · Σ +1.79 → **ceiling 50 holds**.
- **Refuted — do not re-run:**
  - From the 09-14 investigation (DL 55): hour-of-day; BTC macro at entry (51 tests); exhaustive 1D / 2D sweep (sweep_fades.py, shuffle-null); ATR ceiling; wider flat stop; ATR-scaled exits; whole-sleeve de-mux.
  - 20 s entry delay and 20 s no-SL grace (DL 11).
  - Entry delay or limit, and exit widening on real 1m paths (DL 93).
  - Time stop (32 variants) and an earlier arm at 0.35 / 0.30 (DL 111).
  - Fade fixed TP (DL 265).
  - Mcap SIZE rule (FADE_SIZE_MCAP §8; master fades ≥ $200M won 9/9).
  - Winners-vs-losers screen: 0 of 38 columns beyond chance (DL 374).
  - Entry-RSI level, RSI-jump size, BTC-ADX direction / regime, 3D ADX-rising cut (41c).
  - Spike lock on fades: 0 saves, 4 kills (41f).
- **Shipped / reverted:**
  - SL −0.70 → −1.5.
  - Size 2× → 1× (08-05 gate), then restored to 2× by the operator.
  - BRSI 50 → 45 (08-05) → 50 (DL 112).
  - Liquidity cap 0.2 → 0.3 → 0.5. The $1T threshold: set → reverted (DL 80) → set again (DL 165).
  - LAGGARD and late-arm both KEPT after their reverts fired (DL 334 / 332, declared overrides).
- **Pitfalls:**
  - Fade breakeven is 72–79 %, so a WR-only gate misleads (DL 334).
  - Price exits on 1m / ticks only. Post-exit checkpoints only nominate candidates (a +$3k mirage became −$356).
  - The replay reproduces about 29 % of live fades (DL 256).
  - Re-enabling ANY global exit toggle re-arms it on fades too (41g).
  - The mcap stamp exists only since 10-19.
- **Operator questions:**
  - Stop-size question: a −1.5 stop at 2× ≈ $305 ≈ 10–12 % of $2.9k. No size cap alone meets the 4 % bar, so the decision is stop width × size (plan §5).
  - Scale-up / more-fills levers.
  - SPIKE_FADE vs FRENZY pair-occupancy contest.
  - ⚠ Check whether 1000LUNC (−$308) trips the DL 165 tight revert (one $10–20M-band fill losing > $300 → $10M threshold).

## 2. FRENZY_WILLY
- **Live config:**
  - LONG. Trigger A = a new FRENZY flag (a cold read is allowed ≤ 3 bars late; R_NOMISS past that, DL 350). Trigger B is OFF (DL 256).
  - Entry on the first strict red 5m bar within 60 min.
  - Filters: turnover (24 h vol / mcap) < 1.0, missing mcap = no entry, R ≥ 1 routes to HT; $20M volume floor; 1 slot; pair held; late; dislocation. Global hold OFF (DL 340).
  - Size: invest 1 × lev 0.5 = 10× (DL 280).
  - **Exit V2b** (DL 358): resting maker TP at T = max(+1.25, min(+8, net at spike high × 0.99)); stop −4 (DL 271); late TP +0.25 after 30 min (DL 297); cap 90 (DL 281); no floor.
- **Master** (re-priced at V2b, DL 377): 38 · 82 % · +0.71 · +$1,538. INIT + RSR carry 77 % of the gain, so this is a pricing fix, not evidence for V2b.
- **Open gates:**
  - V2B_EARLY 4/10 · MAIN 4/15 on 2/6 days · FINAL 4/25 runner fills. Shadows (FLOOR10, STEPS, TRAIL10/15, HALF, NOLATE, TGT15/20, LATE45/60) at 4/15.
  - ⚠ **DL 281 revert gate reads "REVERT CANDIDATE" in the scout:** N 23 · 9 days · new −0.159 vs old −0.086 · new avg < 0. But parity is 61 % (bar 90 %), and CS does not record this — decide it or fix parity.
  - Turnover v2 (DL 314 / 336): 27/30 blocked walked · 81 % · +0.238 vs let-through. v1 fired 10-18 and the operator kept the filter.
  - WILLY_BTC1H (DL 275): blocked 17/15 but only 6/8 days. DIP_SUB 4/15 and 5/15.
  - Higher-TP cells T130–T150 (DL 290): 18/20 on 6/8 days. LT30 cells (DL 285): 9/20. SL4 vs no stop (DL 268): 36/20 · dipped 7/8.
  - Late TP rollback (DL 297). Resting-TP revert (DL 296, 15 armed TPs). R_NOMISS 0/15 (DL 350). Late-join L 4–12 first 30 on 10 days (DL 366). MKTDIP (DL 305). AVGDOWN (DL 313).
  - Hold-off (DL 340): (a) a day ≤ −10 % of the account; (b) first 8 freed fills, now 0/8.
  - DL 251 revert flag holds: first 20 avg +0.295.
- **Refuted:**
  - Entry B: −0.24 / −0.33 (DL 256).
  - Flag-moment entries, the X study, 5,376 flag rules, the seconds study (DL 251).
  - Stops: −2 at any cap, tighter late stops, +0.3 @ 20 min (DL 285); −3 / −2.5 / −3.5 vs −4 worse in every era (DL 303).
  - Exhaustive 2D, 27,984 tests: nothing (DL 303). Low-VWAP ∧ short-run is noise (DL 303).
  - TP +2 and turnover 0.5 not registered (DL 304). Always holding for +2 ≈ 0 (DL 305).
  - V2 floor at F 1.0 and F 0.5, and the BTC-falling −2.5 stop (DL 356 / 358).
  - TP +0.75, +0.3 from entry, CAP15 / CAP20, +1.50 vs +1.25 (yr5).
- **Shipped:** DL 251 → 271 → 280 / 281 → 296 → 297 → 340 → 341 → 350 → 356 → 358.
- **Pitfalls:**
  - Never mix exit eras: runner / floor fills stay out of old-exit gates (willy_exit_era.py).
  - Turnover reads are walker vs walker (DL 336: booked −0.084 vs walker +0.104).
  - Ticks first (OGN: a −19.65 1m artefact was really +1.00).
  - exit_price_vs_ema5_pct is corrupt on WILLY rows.
  - Losers are coin-specific dumps (DL 304).
- **Operator questions:**
  - Split / runner exit; EMA-cross exit (6 pre-registered variants); price vs EMA13 / 20 / 50, winners vs losers.
  - One final trigger-B re-test (plan §3).
  - 60 vs 90 cap on yr6.

## 3. FRENZY_WILLY_HT (quarter-size WILLY on turnover-only refusals, DL 341)
- **Live config:**
  - A WILLY setup refused only by a measured turnover R ≥ 1.0 opens at 0.25 × WILLY's margin at WILLY's leverage.
  - Same V2b exit, own slot (1). Never counts as a WILLY for the slot or the hold.
  - Size stamp entry_willy_ht_size_frac (DL 365).
- **Master:** 0 HT rows (PROBE_EXEMPT line). B25 holds 7 HT fills, which get priced from WILLY_EXIT_CF.csv once archived.
- **Gates (DL 341):**
  - Kill: Σ ≤ −8 pts → review. First 10 avg < 0 → review.
  - MAIN: frozen first 30 on ≥ 10 days vs same-period WILLY (≥ 10 comparators):
    - HT ≥ WILLY → turnover ratio 0 + HT off
    - HT ≤ WILLY − 0.20 → filter confirmed, HT off
    - else re-read at 60
  - Status: old-exit cohort 0/30. V2b-era run: 2 fills · avg +1.706 (exports to 10-24 only).
- **Re-size ladder (DL 365):** 0.25 → 0.5 → 1.0. Needs 30 fills on 10 days passing Pattern W ∧ HT ≥ WILLY − 0.10. At 1.0, re-decide frenzy_willy_ht_max_slots.
- **Evidence prior:**
  - Backtest, turnover-blocked: 477 · 72.5 % · −0.24 vs kept 1,093 · 79 % · +0.08. Every mcap bucket < $200M is negative (WILLY_TURNOVER_REPRICE_2026-10-19).
  - Forward v2 blocked: 27 · 81 % · +0.238 (contradicts the backtest so far).
- **Per-sleeve V2b split (DL 373, observe-only):** HT n 2 — NOLATE Δ +1.24; every floor / ladder Δ −0.75. Read the "same days only" line.
- **Refuted / closed:** turnover 0.5 (DL 304). Vol/mcap rejected as a FRENZY / WIDE / LITE filter.
- **Pitfalls:** HT N will stay small, so expect a watchlist item, not a filter. The V2b shadows pool WILLY + HT, but the HT gate compares HT against full-size WILLY.
- **Operator questions (10-25), each in three blocks:**
  - Q1: should HT exist at all? The frozen DL 341 gate decides; any earlier answer is labelled below-gate.
  - Q2: should HT have its own exit / protection? (DL 373 split)
  - Q3: should HT winners vs losers be filtered? Run sweep_separators + exhaustive 2D + null + OOS; ship only at the expectancy bar, else observe-only.

## 4. FLATBASE_SHORT (DL 348, operator DISCIPLINE OVERRIDE — the study FAILED its ship bars)
- **Live config:**
  - Episode = a FRENZY pump on a coin whose 14-day 5m close range ≤ 0.444.
  - Entry (S4) = the first 4-h boundary close below the spike VWAP, 4–96 h after the spike, 24 h volume ≥ $20M.
  - Turnover > 1 → trade; ≤ 1 → SHADOW ledger only; unknown mcap → trade.
  - Exit: resting TP −8 % / stop +15 % (backstop 15.5) / 72 h.
  - Size 0.2 × 0.25 (5×). No slot; max_open 10. An open short holds its pair up to 72 h (blocks FRENZY longs on ≈ 4–5 % of episodes).
- **Master:** 1 fill +7.95 (B24).
- **Forward** (scout, exports to 10-24): 6 · 50 % · −3.58 %/fill · Σ −21.48 pts · −$187 on 2 days. FLAT shadow: 1 · +7.95.
- **Gates (alert-only):**
  - 3 stop-outs in a row, or Σ ≤ −30 → review. ⚠ **Σ is −21.5, so one more −15 stop crosses −30.**
  - First 10 avg < 0 → review.
  - Main: 30 fills on ≥ 10 days → ≥ +0.5 step up / < 0 off / else re-read at 60.
  - Shadow vs real after 30 fills.
  - Re-size ladder (DL 363): 0.2 → 0.5 → 1.0, each step re-decides max_open.
  - Exit shadows (DL 367 / 370), 8 cells: CANDIDATE needs avg Δ ≥ +0.10 ∧ Bonferroni/8 lower bound > 0 ∧ no day/fill ≥ 50 % ∧ parity. Now 6/30 on 2/10 days. The early fills are in-sample (the 10-24 batch was included by override), so apply the haircut.
- **Refuted / tested:**
  - Study: S4 · −8 · 72 h · stop 15 = +0.93 in 2026, not fully OOS. Fair-random p = 0.050. No exit passes OOS.
  - Turnover > 1 fails the expectancy bar as a filter; it is part of the definition only.
  - 15m closes for FLAT: rejected.
  - DL 368 grid (TP −3 / −5 / −8 × stop 5 / 10 / 15): live −8 / 15 beats all 8 → KEEP.
  - DL 370 TP +1: WR 82–92 %, but avg −0.05 / −0.07 / −0.19 vs +0.93.
- **Pitfalls:**
  - Units are % of notional (operator).
  - Gates read P&L net of funding; paper has no funding.
  - Generic PAIR_HELD blocks are unattributed. LITE is unpriced while LITE C is on.
  - The memory note project_flatbase_short_sleeve.md still says "build on hold" — stale.
- **Operator questions:**
  - Lower TP / narrower stop: the 6-fill shadows lean strongly positive for TP5_SL5 / TP5_SL10, but that is in-sample at N 6, and the backtest prior says the opposite.
  - Re-size if the sleeve proves good.
  - The real-money probe intent.

## 5. FRENZY_LITE · LITE_RETRY · FRENZY_LONG (WIDE OFF since DL 291)
- **FRENZY_LONG live:**
  - Flag = spike, 30m ≥ +5 % ∧ vol ≥ 20×. Entry on the ON bar (≥ 2 h, 1 h of closes ≥ VWAP, vol ≥ 100×).
  - Filters: 24 h vol ≥ $20M, ATR ≤ 3.0 (DL 250), red / flat signal candle (DL 180), bearish-day block (DL 250), ≤ 3 per pair-day, ≤ 2 open, gvol gate off (DL 263).
  - Exit: TP +1.5 resting (DL 283 / 310), stop **−4** (DL 317), 12 h.
  - Size: **blocker** invest 0.2 × lev 0.05 = 1× (DL 289). Its job is to block bad LITE entries on the same coin.
- **LITE live:**
  - Entry: ≥ 12 closes ≥ spike VWAP, vol < 100×, age 2–16.75 h, each stretch judged once, no ATR cap.
  - Exit LITE C (DL 301): +3 if vol mult ≥ 20 ∧ BTC 1h slope > 0, else +2; stop −3; 12 h.
  - Size: 8× (DL 309).
- **LITE_RETRY live (DL 316):** one retry after a green-bar refusal, ≤ 15 min, at 4×.
- **Master:**
  - LITE 31 · 58 % · +0.293 · +$409
  - RETRY 3 · +1.34
  - FRENZY_LONG 29 · 69 % · +0.126 · −$1
- **Gates — LITE:**
  - LITE watch (DL 243): ⚠ the scout FLAGS avg < 0 at ≥ 20 fills (32/40 · 13/15 days · WR 53 % · −0.001). Review at 40 fills on 15 days.
  - 8× revert (DL 309): first 20 fills avg < 0, or a day < −25 % of the book.
  - TP gate re-based (DL 333): 8/20 · LITE C +0.794 vs the +1.5 counterfactual +0.412.
  - LITE C revert (DL 301): EARLY / FIRST-10 / MAIN-20 reads.
  - RETRY (DL 316): Σ ≤ −6 → off; 10 avg < 0 → off; 20 on 10 days → 8×.
- **Gates — FRENZY_LONG:**
  - −4 stop (DL 317): 0 live −4 fills yet. Re-size read at 20 fills on 8 days (operator waiting until ≈ 11-02).
  - TP revert (DL 283): pooled 11/20.
- **Gates — shared FRENZY:**
  - BEARISH_BLOCKED (DL 250): 16/15 but 7/8 days · mean −1.98 (the block is holding). LITE-only sub-read per DL 272.
  - ATR_RAISE (DL 250): 7/15 · −1.44 vs ≤ 2.5 at −0.81 (leaning back to 2.5).
  - FRENZY_STRONG (DL 197 / 262): 16/20 · −1.23 vs −0.88. The strong bump is off since DL 280 anyway.
  - WIDE re-enable (DL 291): ≥ 15 LITE / SURGE fills on WIDE coins over 8 windows averaging ≤ −1.0.
  - DL 289 blocker: if a LONG setup is ever refused for size → back to 2×.
- **Observe lines:**
  - LITE_BTC1H_FALLING 7/20 · LONG_NONTOP50 4/20 · LITE_LATE_BTCFALL 3/15 (master + forward pooled) · LITE_STRETCH 0/15 · OFF30H3 7/30 · GVOL24_LOW 16/30.
  - Also tracked: GVOL_SPLIT, REENTRY / STRETCHED, LITE 4-h split, GREEN_RETRY, OB_THIN, LONG_STOP4, touch snapshots, FRENZY_WILLY_TP grid (LITE 14/15 · 7/8 days).
- **Refuted:**
  - Entries (Oct-6 → 8 memory note): pre-ON / rising-volume / will-turn-ON / manual-derived / buy-every-flag / 5,376 flag rules.
  - Scalp program (DL 188 / 189).
  - Re-entry while ON; wider / ATR trails; EMA / staircase exits; conditional re-entry (RLC memory note).
  - A later ON bar: −0.49.
  - LITE: 4 / 6 / 8 closes; late / falling ADX; 2D in every regime (P 0.46); cap 1–8 h (DL 327); late-TP grid; Z3 (DL 300); +1.5 halves LITE winners (DL 284); +1 / −4 (P 0.00).
  - LONG → SHORT mirror; adaptive small TP driven by recent P&L.
  - The lock exit (TP3_VS_LOCK holds, Δ −2.15).
  - VWAP_STOP: ❌ closed.
  - Entry speed (latency memory note); Bull-Run exit on FRENZY; vol/mcap filter.
- **Shipped / reverted:**
  - LONG size: 2× → 0.32 → strong 10× → 2× → 10× → 2× → blocker.
  - TP: +4 → +3 → lock → +3 → +1.5.
  - LITE TP: +1.5 → +3 → LITE C. LITE leverage: 0.2 → 0.32 → 0.4 → 0.5 → 0.4.
  - ATR cap: 2.0 → 2.5 → 3.0 (an override of the study's 2.5).
  - Gvol gate: on → off. WIDE: on → hold-green → off.
- **Pitfalls:**
  - Never switch LONG off. It keeps out 19 LITE trades at −1.77 %/fill.
  - Live ruler = 8 s entry + ≥ 0.10 slip (0.18 on stops).
  - Never quote the retracted +0.084 / +0.088 / +0.257 (DL 270).
  - Normalize B19-era fills.
- **Operator questions:**
  - "Is LITE_RETRY ready for 8×?" (⭐①)
  - Remove the bearish block for LITE only? (blocked LITE +0.19 on the year)
  - Register LITE_WEAK_HOLD (above_share ≤ 43.91)? Not registered yet.
  - Split SL from the touch snapshots ("close at −2 / −3 if X").
  - First vs second red candle; one timeline per episode; when a small FRENZY TP wins (path shapes, yr6).

## 6. MOMENTUM LONG (ML)
- **Live config:**
  - long_unmatched_only.
  - Gates: heat block on the original 3-leg rule (slope 0.07 ∧ RSI 64 ∧ bull 80, DL 208); mega-cap rank ≤ 10 (DL 110); LOADX (DL 126); CHOP_BURST (DL 201); gate-51 bands; 1h dead-band 0.05 / 0.025; BTC RSI×ADX cross; ATR×GAP re-opened (DL 133); CROSS_OB narrow OFF (DL 121); ADX-surge waiver (DL 123).
  - Doors: CALM3D 1× (DL 225), RSIADX breadth release, upper dead-band.
  - Exit: SL −0.70 dead / −1.00 active, ×1.5 ATR, floor −1.2; quiet-pair −2.0 stop (gate 53); runner arm **0.40**; HARD_TP ladder; recovery hold (DL 172).
  - Size: 1× notional at 5× with 0.2 margin (DL 298). UNMATCHED cell 1.5× (DL 206). BTC RSI×ADX long cells 60-65 at 2.0×.
- **Master:** 158 · 70.9 % · +0.124 · +$140. This is IN-SAMPLE. Real as-traded ≈ +0.05 (DL 204).
- **Breakeven:** 61 % pinned Sep-25, but live scout reads 66.5–66.9 %. Use the live figure per line.
- **Open gates:**
  - **DL 289 kill gate:** after 30 fills at 1× with avg < 0 at day-boot 90 %, and no ML line at its candidate bar → ML OFF. If a line passes → arm it and restore 10×. The current count needs a run.
  - BTC24H_DOWN and ADX_RISING_IN_REGIME (DL 273): formal re-read due — consistent in all three eras, pooled p 0.001 / 0.05 (DL 304).
  - B1H_NEGFLANK 13/15 · 5/8 days · NEG_DAILYUP 3/15.
  - PAIR_1H_DOWNTREND frozen **KEEP OBSERVING** (N 15 · 33 % · −0.150 · P 0.74; re-read at 30).
  - TREND_ALIGNED keep 2/15. MARKET_TURN (DL 299, 6–8 weeks). STOP_COOLDOWN 1/15. BURST / CHOP fresh-only (DL 160 / 161). BTC-drop + rollover (DL 125). Zones C / D (DL 127). REBOUND (DL 204).
- **Revert gates:**
  - CHOP_BURST 0/6.
  - Heat original 6/15 windows · Σ −1.22. Heat revert check 3/10.
  - MEGACAP 11/8 · 5 won · Σ −4.18 (still collecting).
  - LOADX ✅ holds (decided).
  - Recovery-hold kill.
  - ATR×GAP tripwire / N ≥ 10 verdict.
  - UNMATCHED 15-fill read (DL 206).
- **Refuted:**
  - 1D scan over ~40 columns = chance, except breadth.
  - BTC 1h slope < 0, including 2D (DL 159).
  - bull ≥ 85 ∧ X (pair ADX / BTC RSI / BTC slope).
  - Pair rank ≥ 28 (a fit). Wider stop (DL 110). QS4 2D / 3D. C1 regime.
  - SL −0.70 is a local optimum in both directions (gate 40). ATR-specific vol scaling (06-29). 3 big-loser levers closed (09-18).
  - Blanket 30-min cooldown (DL 248). BTC near its 24 h low (DL 171). BTC impulse (DL 144). Scan cadence (DL 237).
  - Recovery-hold variants (DL 172).
  - DOWNTREND × X, 1,622 cells (DL 303 / 304). Regime switch, 50 day variables (DL 304).
  - ADX-falling ∧ volume — skipped: the shown +$2,540 used the wrong baseline (DL 305).
  - Two-year replay: no separator; sleeve-kill ①–④ run (DL 289).
  - FAKE_BULL_GUARD and the heat re-scope, both reverted.
- **Pitfalls:**
  - Judge on as-traded / forward fills.
  - Cohort completeness: add back fills an older gate screened out.
  - entry_pair_ema20_ema50_gap_pct = EMA13 − EMA50.
  - Show results with and without the washed-out window.
  - Compare in % only ($ is 1/10 before DL 289).
  - Market-wide variables count in DAY / window units.
- **Operator questions:**
  - "ML must become profitable now": winners vs losers on the FULL master first (DL 376).
  - The arm decision on the DL 273 lines was deferred by the operator.
  - Per-cell size via the cell registry (plan §7) — filters first, then sizing.

## 7. MOMENTUM SHORT (MS)
- **Live config:**
  - Pattern-matched only: UNMATCHED SHORT and W6+!W1 are blocked. C1 SHORT has a fixed SL −0.70.
  - C1 blocked in STRONG_BEAR (DL 114). Pair-vol ceiling 0.86 (gate 68, KEPT by DL 226).
  - Weakcap block. Deep-gap floor −1.0. BTC ATR ≥ 0.12. 1h RSI ≥ 35. 1h slope ≥ −0.60. Pair ADX rising.
  - Exit: short stack — ATR stop, arm 0.40, giveback cap 0.35 (46c), NEGFLOOR ride (46b), EMA13 cross, short HARD_TP ladder.
  - Size: pattern cells all 1.0 (DL 227 / 233).
- **Master:** 41 · 75.6 % · +0.190 · +$661. SCREENED_BASELINE v23 MS: 14 · 86 % · +$672.
- **Open gates:**
  - MS_PVR_BLOCKED Cohort A: 3/15 · 33 % · −0.367. Revert to 1.0 if A mean ≥ the kept mean.
  - Kept-side review if WR < ~59 % at 20 fills (DL 226; the kept side was 11 · 36 % at 10-06).
  - C1 STRONG_BEAR revert: first 5 blocked with WR ≥ 60 %. Now 0/5.
  - C1 dist-EMA13 ≤ −0.75 tally: 4L / 0W, null P 0.62.
- **Refuted:**
  - BTC "bull week" H1 — wrong sign (DL 226).
  - High PVR is not BTC macro in disguise (DL 226).
  - C1 at 2× (cut Jun-29, 1W / 4L). W2+W1 2× (de-muxed 07-30).
  - Caps-for-losers (combined-pool memory note).
  - max_open 4 → 3 (DL 149, not shipped).
- **Pitfalls:**
  - Use SCREENED_BASELINE with C1 de-muxed, never the raw pool.
  - The slump is day-clustered (09-29 = 41 % of the kept loss).
  - ⚠ trading_config.json still carries BTC RSI×ADX SHORT cells at 2.0× (35-40:33-36, 25-30:28-30), while the master prices "multiplied momentum SHORT → 1×". Confirm what the engine actually does.

## 8. FAN flip-shorts
- **Live config:**
  - FAN_RATIO_GATE at 1.0 inv × 0.5 lev = **10×** (DL 252).
  - Gates: BTC 1h slope ≤ 0; BTC trend-gap ≥ −0.22; BTC vs EMA13 ≤ −0.08; weak-bounce block (DL 113); pair ADX ≥ 21 (exempt in STRONG_BEAR); QS ≥ 2; RSI ≥ 55; HEALTHY_BULL regime block; ATR block in bear regimes.
  - NEGDI15 / TG_SHALLOW cells reverted to 1× (DL 220). Exit: STRPK runner.
- **Master:** 35 · 80 % · +0.233 · +$366. SCREENED v23: FLIP 31 · 81 % · +$346. Breakeven ≈ 57 % (DL 166).
- **Open gates:**
  - FAN_10X: 5/15 · 80 % · +0.001. Restore 20× at WR ≥ 63 % ∧ avg ≥ +0.20. avg < 0 → sleeve review, kill checklist first.
  - FLIP_EMA13_BLOCKED 4/10 windows (−0.34). FLIP_PADX_BLOCKED 6/10 (−0.11).
  - Weak-bounce revert 0/8. STRONG_BEAR pADX exemption 4/10.
  - −DI<15 (DL 120, tally only). Weak-sellers-in-chop (DL 166). Arm-0.35 watch (gate 48).
- **Refuted:**
  - Jul-1 entry-separator search (31 dimensions).
  - −DI<15 in the backtest (09-29). bear ≥ 76 has the wrong sign.
  - QS = 2 (positive in every regime). 41-dimension shuffle scan, P 0.69 (gate 68).
  - The 1.70–3.0 fan band collapses after the later gates (DL 112).
  - 2× cells (DL 220). Entries inseparable — blacklist only. Caps-for-losers.
- **Pitfalls:**
  - SCREENED pool only: a flip N ≫ 34 or a negative total means you used the raw pool.
  - Judge flip filters on what they block (DL 221).
  - Use live eff72 stamps over rebuilds near thresholds.

## 9. BEAR-RUN (BEARRUN_SHORT)
- **Live config:**
  - Monitor: r24 ≤ −4 ∧ below-EMA20 ≥ 50 ∧ eff ≥ 0.15 (stay 3 / 47 / 0.15); squeeze latch; BTC ≤ 2 % above its 24 h low; breadth ≥ 40.
  - While ON, momentum shorts bypass 5 BTC gates. BTC / ETH blacklisted. 2 h spacing. Shared slots. Momentum-short exit, including the EMA13 cross.
  - Size: invest 0.2 × lev 0.25 ≈ $500 notional (DL 360).
- **Master:** 2 · +0.54 (10-07). Live 10-22 window: 3 shorts −$69.
- **Open gates:**
  - (DL 200) arm bar to full size: 1/5 windows.
  - Kill bar: first 10 fills WR ≤ 45 ∨ Σ < 0, or 2 negative windows in a row.
  - Breadth revert. EMA13-exit watch (DL 148).
  - V2 shadow (DL 359): ARM at 20 windows on 6 weeks (walk-forward ⑤ fails today at −253). DROP at 30.
  - r24 −2.5 shadow (DL 369): 20 windows on 6 weeks.
  - BEARRUN_5X is superseded (DL 360). The scout row is stale — ignore it.
- **Refuted:**
  - 72 h mirror. Bull-run exits for shorts. PVR ≤ 1.2 (hurts). No rank cap.
  - Replay −0.24 / window (DL 200).
  - Sleeve-kill checklist: no separator. Trigger timing: no earlier trigger helps. V2 replay fails vs T0 (0/4 walk-forward). V2 screen, 2,041 features: none (all DL 360).
- **Operator question:** the v2 redesign (early 1h −DI trigger, coin exhaustion guard, NO EMA13 exit — memory note) is pending the studies and the shadow windows.

## 10. SURGE_LONG (SURGE_SHORT OFF since DL 200)
- **Live config:**
  - Trigger: BTC 30-min move ≥ 0.3 % ∧ bar vol ≥ 5× ∧ 24 h-high close ∧ market vol ≥ 1 (DL 202).
  - Picks: top-20, leader, ATR ≥ 1.5 %. 5-min window. Spacing 4 h, counted only after a fill.
  - Filter: EMA13−50 gap ≥ 0 (DL 287).
  - Exit: Bull-Run trail with no lock (DL 203); 240 min cap.
  - Size 1 × 1 (targets 20×; Binance brackets cap thin pairs).
- **Master / scout:** 3 filled triggers · 100 % · +4.40 · +$330.
- **Gates:**
  - Option B revert: first 15 filled triggers with mean ≤ 0 → lev 0.05. Now 3/15. At that review, also re-walk with the lock.
  - EMA-gap revert: ≥ 10 blocked on 5 windows with WR ≥ 31.5 ∧ avg > 0.
  - Option A comparison at 30 A triggers. EMA-stack watch (DL 148). Bracket watch (DL 335, Pattern W only).
- **Refuted:**
  - SURGE_SHORT: 146 triggers · −0.094.
  - SHORT −2 % stop cap (DL 175, declined).
  - 1 % trigger; longer windows; 1 h cooldown (DL 202).
  - The design's +0.2 was a 1m-candle artefact — tick-level only (DL 200).
- **Pitfalls:** count fills from 09-30 15:30 UTC only (cohort floor). auto_kill is off (DL 202).

## 11. BULL-RUN (BULLRUN_LONG — enabled, but no GREEN since 09-19 / 20)
- **Live config:**
  - GREEN gate: r72 10 / 8, above 56 / 53, eff 0.10. Top-10 universe. Dip 0.3 ATR. 4 slots.
  - SL −0.7. BE +1.0 / +0.2. Trail 2× ATR (REARM 1×). Ladder.
  - Filters: PVR ≤ 1.2; blacklist ONG / ETH / BTC / ONE / ZEC; breadth ≥ 40; dislocation 0.3.
  - REARM door with a 60-min entry-age cap (gate 57l).
- **Master:** 102 · 48 % · −0.033 · −$532 (v2 floor 08-21 19:16).
- **Gates:**
  - 57l age cap: first 2 REARM episodes with net-negative kept fills → cap 0. 0/2 — never tested (DL 274).
  - "2 net-negative GREEN episodes → OFF": not run.
  - 57c / 57e / 57j / 57k watches.
- **Refuted:** rank 11–20 (−$1,933); arm 0.7 live (DL 93); every peak-giveback floor (57j).
- **Pitfall:** no GREEN / REARM = a logs question first.

**Inactive (confirm still off):** FRENZY_WIDE (DL 291) · SURGE_SHORT (DL 200) · SPIKE_CHASE (08-21) · SPIKE_BOUNCE (08-10) · probe fleet (08-11). MANUAL is the operator's sleeve and is excluded from every gate.

---

## Contradictions found (CURRENT_STATE vs DECISION_LOG / config)

> **Status 10-26 (DECISION_LOG 378):** items 1, 2, 3 and 6 were CORRECTED in place in CURRENT_STATE against trading_config.json; the rest are AUDIT FLAG rows in NEXT_BATCH_REVIEW_PLAN §0b (read at their sleeve's step).
1. **CS "Active Pattern Cell Ship rules"** still lists C1 SHORT 2.0× inv, W2+W1 SHORT 2.0× inv and UNMATCHED LONG 2.0×. The JSON has C1 / W2+W1 at 1.0 and UNMATCHED at 1.5. DL 206 cut UNMATCHED to 1.5×; DL 227 / 233 put every SHORT pattern cell at 1×.
2. **CS "Active multipliers"** says SPIKE_FADE SHORT = 1.0× / 1.0×, but the JSON has spike_fade_invest_mult 2.0, and DL 374 / 41h / 41m all assume 2×.
3. **CS line "SURGE_LONG + SURGE_SHORT — LIVE, normal size 1×/1×"** contradicts DL 200 (SURGE_SHORT OFF; the JSON has surge_short_enabled false). The CS DL 202 line agrees with DL 200.
4. **The ML analysis pin and the A3 line still say "heat block = bull breadth ≥ 85"**, but DL 208 reverted to the 3-leg rule (the JSON has bull_pct_min 80). The pin also says UNMATCHED "stays 2×" (now 1.5×).
5. **The CS sizing-observations line headlines LITE 10× (DL 283).** DL 309 put LITE back to 8× (the JSON has 0.4). The FRENZY_LITE line is correct. That same LITE line also says "FRENZY's exit (fixed +1.5 / −3)", which contradicts its own +3 / LITE C text.
6. **CS "Mode & capital"** says 5 max positions and $2,500 + $500 BNB. The JSON has max_open_positions 4 (gate 55, 08-20) and paper 10,000 (DL 375). The long trailing line says "arms at peak ≥ 0.45"; the JSON has runner_trail_arm_peak 0.40.
7. **The DL 55 pre-committed REOPEN (2026-10-14: fade volume ceiling → 0 + a new large-cap 0.5× probe) is overdue.** No DL entry executes or defers it. The JSON still has spike_fade_max_vol_24h_usd 20M, yet CS C-⑪ says "the Oct-14 probe collects it" — no such probe field exists.
8. **The DL 281 WILLY revert gate** shows "REVERT CANDIDATE" in the scout (new avg < 0, parity 61 %). CS does not mention it.
9. **Rule conflict:** CS "Do not change for now" says "exit-side + sizing is the active lever, no more entry filters". That conflicts with the operator's 10-09 rule "filters, not sizing" (memory note) and with plan §7 ("sizing LAST, after the filters").
10. **Stale memory notes:** project_flatbase_short_sleeve.md says "build on hold", but DL 348 shipped. project_run_sleeve.md says FRENZY_LONG is "ARMED at Inv 2x", but it is blocker size since DL 289.
