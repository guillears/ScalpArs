# Averaging down (in-trade martingale) by sleeve — 2026-10-19

**Operator question:** "Martingale: which of our active sleeves would it make sense for?" The question was narrowed during the study to **averaging down**: adding to a losing position while it is still open. Re-entry after an exit is covered only in the short appendix at the end.

**Short answer: no sleeve qualifies for an armed rule. Two-times "true martingale" (AD2) is rejected everywhere, and so is sequence martingale (double the next trade after a loss).** One sleeve, FRENZY_WILLY, has a weak sign that agrees between master and backtest for a single add of 1× at −2 %. Live goes the other way on 6 adds. That is enough for an **observe-only shadow line with a frozen gate**, and nothing more.

Scripts: `scripts/study_reentry_after_loss_avgdown.py` (walker) and `scripts/study_reentry_after_loss_avgdown_report.py` (tables).
CSVs: `reports/AVERAGE_DOWN_2026-10-19_{rows,summary,margin,seqmart,concurrency,master_per_batch,reentry_appendix}.csv`.
Validation: `scripts/validate_against_master.py` reported ALL CHECKS PASS. The study's walker matches the master `stack_pct` on 96–100 % of FRENZY fills, with the same sign on 100 %. On trail sleeves, a fill with no add equals the master row exactly by construction.

## What was tested (pre-registered in the script docstring before any result was read)

- **Trigger.** One add, made while the original position is open, when the original leg reaches net −X %. X is ⅓, ½ or ⅔ of the sleeve stop S.

  | Sleeve | Stop S (%) |
  |---|---|
  | MOM long / short | 0.70 |
  | FLIP | 1.20 |
  | FADE | 1.50 |
  | BULLRUN | 0.90 |
  | SURGE | 1.20 |
  | WILLY | 4 |
  | LITE | 3 |
  | FRENZY_LONG | 3 |

- **The three variants:**
  - **AD1:** add 1× the original notional. Fixed-TP sleeves (WILLY, LITE, LONG) take their own TP measured from the **new average price**. The **original stop price** and time cap stay unchanged, so a failed add doubles the loss. WILLY's late TP (+0.25 from minute 30) is also measured from the new average. Trail sleeves have no TP, so the add **rides the original exit**: the trail stays keyed to the original entry and both legs leave at the recorded exit price.
  - **AD2:** the same with a 2× add.
  - **AD3:** add 1×, then close everything at break-even + fees from the new average. If that level is never reached, the original exit applies.
- **Costs.** Taker fee 0.045 % on both sides of the add. Add slippage 0.03 % on entry and on any non-TP exit. Fade market impact (KMNO-type) is **not** modelled, so fade results are optimistic.
- **Price path.** Ticks where the cache has them (9.8k of 10k rows), otherwise 1m bars with stop-first ordering:
  - a bar that touches both the add line and the stop: add, then stop;
  - a bar that touches both TP and stop: stop;
  - no TP on the add bar itself;
  - a bar that opens through a line fills at its open;
  - if the baseline exit bar also touches the add line: no add.
- **Trail-sleeve trigger.** The bot's live-polled trough stamp (`trough_pnl ≤ −X`) must confirm the add. A 1m wick the bot never polled does not count as an add.
- **Sizing in $.** Today's notional at the $1,225 account, taken from the B22/B23 live sizes:

  | Sleeve | Notional |
  |---|---|
  | ML | $250 (5×) |
  | MS | $4,800 (20×) |
  | FLIP | $2,400 (10×) |
  | FADE | $11,870 (20×, live KMNO) |
  | BULLRUN | $4,800 (20×) |
  | SURGE | $2,300 (10×) |
  | WILLY | $2,300 (10×) |
  | LITE | $1,950 (8×) |
  | FRENZY_LONG | $100 (blocker) |

- **Blocks:**
  - ① **LIVE:** B22 + B23 live exports, 42 bot fills.
  - ② **MASTER:** `MASTER_POOL_stacked.csv`, STACK 2026-10-19a, kept and non-probe, 475 fills in active sleeves.
  - ③ **BACKTEST:** yr6 2026 replay (`ENGINE_REPLAY_YR6_fills.csv`), seeds 1/2/3. For the FRENZY sleeves the three seeds hold almost the same fills, so they are not independent evidence.
- **Statistics.** Δ$ = (variant − today) × today's notional. 95 % confidence intervals come from a day-block bootstrap with 4,000 draws. Fixed-TP sleeves are compared against the walker's own baseline.

## Theory test: whose dips are these?

Averaging down pays only if the dip to −X is **noise inside future winners**. In that case the add leg, roughly (exit + X), is positive. If the fills that reach −X are mostly future stop-outs, the add only doubles the loss.

In the table below, **add leg** is the mean extra P&L of the added unit (AD1) per trigger, as % of the original notional.

| Sleeve | X | Winners reaching −X (master / BT s1) | Losers reaching −X (master / BT s1) | Add leg, master | Add leg, BT s1 / s2 / s3 | Reading |
|---|---|---|---|---|---|---|
| MOM_LONG | 0.35 | 39 % / 41 % | 100 % / 100 % | +0.04 | −0.16 / −0.14 / −0.12 | continuation (BT CIs < 0) |
| MOM_SHORT | 0.35 | 38 % / 31 % | 100 % / 98 % | +0.12 | −0.16 / −0.14 / −0.11 | continuation |
| FLIP_SHORT | 0.60 | 29 % / 19 % | 86 % / 100 % | +0.33 (7 losers) | −0.24 / −0.31 / −0.23 | regime break (BT CIs < 0 in every seed) |
| SPIKE_FADE | 0.75 | 15 % / 18 % | 62 % / 97 % | +0.42 | −0.19 / −0.12 / −0.07 | continuation + gap tail |
| BULLRUN_LONG | 0.45 | 45 % / 41 % | 98 % / 100 % | +0.01 | −0.11 / −0.12 / +0.16 | coin-flip |
| SURGE_LONG | 0.60 | (N = 1) / 37 % | — / 94 % | — | −0.04 / +0.02 / −0.24 | coin-flip |
| FRENZY_LITE | 1.50 | 21 % / 34 % | 100 % / 100 % | −0.67 | +0.06 / +0.06 / +0.06 | split (master −, BT +) |
| FRENZY_LONG | 1.50 | 28 % / 31 % | 100 % / 100 % | −0.45 | +0.17 (×3) | split |
| **FRENZY_WILLY** | **2.00** | **30 % / 23 %** | **100 % / 95 %** | **+0.79** | **+0.19 / +0.19 / +0.20** | **only sign agreement** |

On every sleeve, essentially **all losers reach −X and only a minority of winners do**. A dip to −X is therefore mostly a signal that the trade is going to lose. The momentum, flip, fade and bull-run sleeves are continuation sleeves: their stop-outs are regime breaks, not shake-outs.

WILLY is the one exception in kind. It buys a dip inside a frenzy and has a wide stop (−4) and a short clock (90 min). Within that structure, a −2 % dip is close to a coin-flip that sometimes reverts. A new average about 1 % lower then makes the +1.25 TP, or the +0.25 late TP, reachable.

## Results per sleeve (AD1 / AD2 / AD3, X = ½ stop unless stated)

"Triggered" is the number of fills that added. The breakdown is:
- **R** = original losers rescued to ≥ 0
- **L+** = original losers made bigger losers
- **W+** = original winners made bigger
- **W−** = original winners made smaller

Δ$ is the total versus today, with its 95 % day-block CI. Worst = the worst single trade as % of the account. Full grids for all three X values are in `_summary.csv`.

### ② MASTER (in-sample to today's filters; per-batch Δ in `_master_per_batch.csv`)

| Sleeve (N, today avg %) | Variant | Triggered (losers/winners) | R / L+ / W+ / W− | avg % → | Δ$ [95 % CI] | worst % acct (today) | batches +/triggered |
|---|---|---|---|---|---|---|---|
| MOM_LONG (145, +0.142) | AD1 | 82 (42/40) | 0/42/40/0 | +0.166 | +9 [−27, +47] | −0.8 (−0.4) | 7/20 |
| | AD2 | 82 | 0/42/40/0 | +0.190 | +17 [−55, +92] | −1.1 | 7/20 |
| | AD3 | 82 | 0/25/0/40 | −0.005 | **−53 [−87, −21]** | −0.5 | 4/20 |
| MOM_SHORT (39, +0.179) | AD1 | 21 (10/11) | 0/10/11/0 | +0.246 | +125 [−154, +435] | −5.4 (−3.1) | 6/9 |
| | AD2 | 21 | 0/10/11/0 | +0.312 | +249 [−323, +877] | −7.8 | 6/9 |
| | AD3 | 21 | 4/6/0/11 | +0.042 | **−256 [−509, −15]** | −5.1 | 1/9 |
| FLIP_SHORT (35, +0.233) | AD1 | 14 (6/8) | 0/6/8/0 | +0.366 | +111 [−79, +333] | −3.9 (−2.4) | 1/5 (BASE only) |
| | AD2 | 14 | 0/6/8/0 | +0.498 | +222 [−154, +676] | −5.3 | 1/5 |
| | AD3 | 14 | 0/4/0/8 | +0.110 | −104 [−228, +6] | −3.9 | 2/5 |
| SPIKE_FADE (74, +0.356) | AD1 X0.50 | 26 (7/19) | 0/6/19/0 | +0.483 | +1,112 [+265, +1,941] | **−25.6 (−14.5)** | 9/11 |
| | AD1 | 15 (5/10) | 0/5/10/0 | +0.442 | +752 [−138, +1,635] | −23.2 | 7/9 |
| | AD2 | 15 | 0/5/10/0 | +0.527 | +1,503 [−207, +3,308] | **−31.9** | 7/9 |
| | AD3 | 15 | 0/5/1/9 | +0.260 | **−842 [−1,443, −264]** | −23.2 | 1/9 |
| BULLRUN_LONG (102, −0.033) | AD1 | 74 (52/22) | 0/52/22/0 | −0.028 | +23 [−454, +482] | −13.0 (−13.0) | 3/5 |
| | AD2 | 74 | 0/52/22/0 | −0.024 | +46 [−922, +1,000] | −13.0 | 3/5 |
| | AD3 | 74 | 25/27/0/22 | −0.236 | **−994 [−1,767, −246]** | −13.0 | 2/5 |
| FRENZY_WILLY (27, +0.370) | AD1 | 11 (4/7) | 1/3/7/0 | +0.693 | +200 [−144, +568] | −11.8 (−7.6) | **1/3 (B19 +$295; B20 −$50, B22 −$44)** |
| | AD2 | 11 | 2/2/7/0 | +1.268 | +557 [−9, +1,081] | **−15.9** | 2/3 |
| | AD3 | 11 | 2/2/0/7 | +0.319 | −32 [−437, +424] | −11.8 | 1/3 |
| FRENZY_LITE (24, +0.294) | AD1 | 13 (10/3) | 0/10/3/0 | −0.069 | −170 [−420, +110] | −7.5 (−4.9) | 1/5 |
| | AD2 | 13 | 0/10/3/0 | −0.432 | −340 [−861, +196] | −10.2 | 1/5 |
| | AD3 | 13 | 4/6/0/3 | +0.051 | −114 [−351, +185] | −7.5 | 1/5 |
| FRENZY_LONG (26, +0.120) | AD1 | 13 (8/5) | 0/8/5/0 | −0.105 | −6 [−15, +4] | −0.4 | 1/6 |
| | AD2 | 13 | 0/8/5/0 | −0.333 | −12 [−30, +10] | −0.5 | 1/6 |
| | AD3 | 13 | 4/4/0/5 | +0.056 | −2 [−16, +13] | −0.4 | 2/6 |

SURGE_LONG (N = 1) and BEARRUN_SHORT (N = 2, no trigger) are too thin to read.

**Per-pair concentration.** The top pair carries ≤ 20 % of |Δ| on every sleeve except:
- WILLY: BAT 35 % at AD1, 47 % at the −2.67 X;
- FLIP: BIO 32 % at the −0.80 X;
- FADE AD3: ENSO 25 %.

### ① LIVE (B22 + B23, 42 bot fills; FRENZY rows re-walked)

| Sleeve (N) | AD1 Δ$ | AD2 Δ$ | AD3 Δ$ | Note |
|---|---|---|---|---|
| FRENZY_WILLY (12) | **−145** (6 adds: 4 L+, 2 W+) | **−288** | +27 | live WILLY losers all ran on to −4; worst trade −11.8 % / −15.9 % of the account vs −7.6 % today |
| FRENZY_LITE (7) | −98 | −196 | −6 | 3 adds, all three original losers made bigger |
| FRENZY_LONG (4) | −3 | −7 | +1 | blocker size |
| MOM_LONG (15) | −8 | −16 | −5 | |
| SPIKE_FADE (1: KMNO) | **−180** | **−360** | −180 | one fill: **−20.6 % of the account today → −35 % (AD1) / −50 % (AD2)**, before any extra impact |
| SURGE_LONG (3) | +152 | +304 | −131 | 2 winners that dipped first (AGT, GTC); N = 3 |

### ③ BACKTEST (yr6 2026 replay; Δ$ per seed s1 / s2 / s3, a full year at flat $1,225 sizing)

| Sleeve (n/seed) | AD1 | AD2 | AD3 | Read |
|---|---|---|---|---|
| MOM_LONG (~565) | −148 / −130 / −110, **all CIs < 0** | −295 / −261 / −219, **all < 0** | −128 / −171 / −156, **all < 0** | reject |
| MOM_SHORT (~155) | −646 / −616 / −490, **all < 0** | −1,292 / −1,233 / −980, **all < 0** | −427 / −730 / −606 | reject |
| FLIP_SHORT (~140) | −417 / −435 / −396, **all < 0** | −834 / −870 / −792, **all < 0** | −147 / −83 / −119 | reject |
| SPIKE_FADE (~390) | −3,293 / −2,274 / −1,244 | −6,586 / −4,548 / −2,488 | −7,191 / −7,282 / −7,541, **all < 0** | reject; **worst single trade −41 % / −67 % / −123 % of the account (TAC −6.7 % gap through the −1.5 stop, doubled)** |
| BULLRUN_LONG (~130) | −474 / −511 / +728 | −949 / −1,023 / +1,457 | −758 / −1,236 / −1,898 | reject (sign flips by seed) |
| SURGE_LONG (~177) | −118 / +54 / −764 | −237 / +107 / −1,529 | −342 / −974 / −68 | reject |
| FRENZY_WILLY (~608) | **+1,008 / +1,008 / +1,059**, CI [−548, +2,556] | +1,634 / +1,634 / +1,735, CI [−1,044, +4,201] | −113 / −113 / −63 | **only consistent +; CI spans 0** |
| FRENZY_LITE (~626) | +416 / +441 / +455, CI [−1,297, +2,241] | +1,972 / +2,023 / +2,051, CI [−1,315, +5,282] | −2,456 / −2,639 / −2,586, **all < 0** | master and live both −; CI spans 0 |
| FRENZY_LONG (~350) | +32 / +34 / +31 | +60 / +63 / +57 | **+46**; at ⅓ X **+67 [+4, +131]** | $ immaterial at blocker size; master −2, live +1 |

**AD3 (break-even rescue) loses in every block on every trail sleeve.** It converts dipped winners into scratches (W−) far more often than it rescues losers (R). This is the clearest result in the study.

## Margin, liquidation, feasibility at $1,225 (X = ½ stop; cross margin; 1 % maintenance assumed; others = other slots of ~$240 margin)

| Sleeve | Add margin AD1 / AD2 | Liq. distance from add price, AD1 / AD2 (this position alone) | Stop distance from add | Loss at stop, % of account: today → AD1 / AD2 | Max other slots open for the add to fit, AD1 / AD2 |
|---|---|---|---|---|---|
| MOM_LONG (5×, $250) | $50 / $100 | 244 % / 162 % | 0.35 % | 0.1 → 0.2 / 0.3 | 4 / 4 |
| MOM_SHORT (20×) | $240 / $480 | 11.6 % / 7.4 % | 0.35 % | 2.7 → 4.6 / 6.4 | 3 / 2 |
| FLIP (10×) | $240 / $480 | 24 % / 16 % | 0.60 % | 2.4 → 3.8 / 5.2 | 3 / 2 |
| **SPIKE_FADE (20×, $11,870)** | **$594 / $1,187** | **3.8 % / 2.2 %** | 0.75 % | **14.5 → 23.0 / 31.4** | **impossible even with 0 other slots** (free margin ≈ $570) |
| BULLRUN (20×) | $240 / $480 | 11.5 % / 7.4 % | 0.45 % | 3.5 → 5.8 / 8.0 | 3 / 2 |
| WILLY (10×) | $230 / $460 | 24.6 % / 16.1 % | 2.0 % | 7.5 → 11.5 / 15.5 | 2 / 2 |
| LITE (8×) | $244 / $488 | 29.7 % / 19.4 % | 1.5 % | 4.8 → 7.4 / 9.9 | 2 / 1 |
| FRENZY_LONG (1×) | $100 / $200 | > 400 % | 1.5 % | 0.2 → 0.4 / 0.5 | 4 / 3 |

**Other positions open at the master add times (AD1, a lower bound):**
- WILLY: mean 0.09, max 1, so the add would have fit every time.
- BULLRUN: mean 1.9, max 3; 36 % of adds would have hit the slot / margin wall.
- FADE: the add cannot be funded at this account size.

The backtest gap tail (TAC −6.7 % on a −1.5 stop) is beyond the AD2 liquidation distance of 2.2 %. On that fill, a doubled fade would have been liquidated.

## Sequence martingale (2× the next trade of the same sleeve after a loss, capped at 2 doublings: 1 → 2 → 4)

Doubling only helps if the trade after a loss is better than the trade after a win.

| Block · sleeve | Next trade after a loss | Next trade after a win | Diff [95 % day CI] | Max loss streak | Max DD, % of account: flat → doubling |
|---|---|---|---|---|---|
| Master MOM_LONG | −0.042 (n 42) | +0.213 | −0.26 [−0.56, +0.13] | 7 | −1.8 → −7.6 |
| Master MOM_SHORT | −0.050 | +0.282 | −0.33 [−0.66, +0.21] | 5 | −15 → −38 |
| Master BULLRUN | −0.228 | +0.194 | −0.42 [−1.39, +0.23] | 8 | −88 → **−202** |
| Master FLIP | +0.184 | +0.230 | −0.05 [−0.42, +0.67] | 2 | −7.7 → −8.4 |
| Master FADE | +0.579 (n 8) | +0.316 | +0.26 [+0.00, +0.51] | 1 | −30 → −23 |
| Master WILLY | +1.250 (n 3) | +0.205 | +1.05 [+0.42, +2.10] | 1 | −10 → −8 |
| Master LITE / LONG | +0.89 / +0.94 (n 9 / 8) | −0.19 / −0.05 | CIs span 0 | 2 | — |
| BT s1 MOM_LONG | −0.167 | −0.006 | −0.16 **[−0.27, −0.04]** | 7 | −10 → −24 |
| BT s2 MOM_LONG | −0.193 | +0.047 | −0.24 **[−0.35, −0.13]** | 11 | −8 → −27 |
| BT s1 FADE / s2 / s3 | +0.12 / −0.13 / −0.05 | — | +0.12 / −0.16 / −0.04 (all CIs span 0) | 3–5 | **−217 → −278 / −207 → −363 / −306 → −415** |
| BT s1 WILLY | +0.063 | +0.102 | −0.04 [−0.39, +0.28] | 6 | −73 → **−134** |
| BT s1 LITE | +0.135 | +0.235 | −0.10 [−0.58, +0.36] | 7 | −84 → **−250** |
| BT s1 SURGE | +0.268 | +0.361 | −0.09 | 11 | −33 → −95 |

The master's "after a loss is better" readings on WILLY, FADE, LITE and LONG come from 3–9 trades. The backtest flips them to roughly zero or negative. On momentum long, the backtest points to the opposite: the next trade after a loss is reliably worse.

Doubling multiplies the maximum drawdown by 1.5–3× on every sleeve and creates no edge. **Rejected outright.** It is also ruin-shaped: BT loss streaks run 6–13 on WILLY, LITE, SURGE, BULLRUN and ML.

## Recommendation per sleeve

**Every Δ above needs the 30–50 % in-sample haircut. No backtest here has been reviewed for arming (no-arm-before-review rule).**

| Sleeve | Verdict | Why (numbers) |
|---|---|---|
| MOM_LONG | **Reject** | BT add leg −0.12 to −0.16 on ~370 adds, all 3 seeds with CI < 0; 100 % of losers vs 39 % of winners reach −0.35 |
| MOM_SHORT | **Reject** | BT CIs < 0 in all seeds at X ≥ ½ stop; master + is in-sample and spans 0 |
| FLIP_SHORT | **Reject** | BT −$400/yr per 1× add with every CI < 0; master + comes from BASE alone (1 of 5 batches) |
| SPIKE_FADE | **Reject, hard** | add infeasible at $1,225 (margin); worst trade −23 to −31 % of the account in master, up to −123 % in BT; live KMNO −20.6 % → −35/−50 %; BT AD1 −$1.2k to −$3.3k |
| BULLRUN_LONG | **Reject** | add leg ≈ 0, sign flips by seed; AD3 CI < 0 in master; 36 % of adds would hit the slot wall |
| SURGE_LONG | **Reject (unproven)** | BT mixed by seed (−$764 to +$54); master N = 1; live + on N = 3 |
| FRENZY_LITE | **Reject** | master and live negative, BT + with CI spanning 0; AD3 BT CI < 0 |
| FRENZY_LONG | **Reject (immaterial)** | blocker size $100: even the one BT CI > 0 cell (AD3 ⅓) is +$67/yr, and master / live disagree |
| **FRENZY_WILLY** | **Observe-only shadow (not armed)** | the only sleeve whose add leg is + in master (+0.79, 11 adds) and BT (+0.19, ~230 adds/seed). But live is −$145 on 6 adds, master + sits in one batch (B19), CI spans 0, and BAT carries 35 %. Risk: the worst trade goes from −7.6 % to −11.8 % of the account (AD1). AD2 rejected (−15.9 %/trade). |
| Sequence martingale (all) | **Reject** | no sleeve shows a reliable "after a loss is better" effect in BT; doubling multiplies DD 1.5–3× |

**Proposed observe-only line (only if the operator wants it; it changes no trading).** Stamp a WILLY shadow add:
- **When:** the first time a WILLY fill touches net −2.00 %.
- **What is recorded:** the add price, and the shadow position's exit (TP +1.25 / late +0.25 from the new average, original −4 stop price, 90 min cap).
- **🔒 Frozen gate (pre-registered now, never re-fit):** at N ≥ 30 shadow adds over ≥ 8 distinct days, the add-leg mean must be > 0 with day-bootstrap P ≥ 0.95. In addition:
  - no pair carries ≥ 50 % of ΣΔ;
  - the haircut Δ (×0.5) is reported for the discussion (as a pass test it equals ΣΔ > 0, already required);
  - no shadow trade is worse than −12 % of the account, measured sizing-invariantly (merged % × $2,300 ÷ $1,225; DL 313 deep review).
- **Then:** discuss AD1 only, never AD2. If it fails, the line is dropped.

Expected time to N = 30: about 4–6 weeks. The add rate is about 0.8 per day in the backtest (230 per year) and 1.2 per day live (6 in B22/B23).

## Caveats

- **Trail sleeves use co-exit (both legs leave at the original exit).** A real engine with one merged position would key its trail and stop to the new average. That design is not tested here.
- **WILLY's resting maker TP** is booked exactly at the line. Paths are mostly ticks; on 1m bars the order is stop-first.
- **Fade results are optimistic.** Liquidity caps and impact on a doubled order are not modelled. The fade ledger is known to be about $900 high (CURRENT_STATE note).
- **Backtest $ use flat sizing** at $1,225 over a full year with no compounding. BT FRENZY_LONG `pct` reflects the replay's own exit, while the walker uses today's +1.5. Seeds 1–3 of the FRENZY sleeves are nearly the same fills.
- **Master is in-sample** to today's filters, so the master positives on momentum, flip and fade are exactly what the cross-period refute-only rule warns about.

## Appendix: re-entry after the exit (the original question, kept short)

Earlier work already refuted re-entry on the FRENZY family over the year:
- `FRENZY_REENTRY_STOP_GVOL_2026-10-05`: re-entry while ON loses in 12/12 cells, −0.08 to −0.32 %/trade;
- `WILLY_REENTRY.txt`: re-entry after a STOP is +0.10 immediately but −0.12 to −0.20 after a 5–15 min wait;
- `SCOUT_LITE_STRETCH_REENTRY`: LITE retry −0.005 %/trade.

From the master's stamped post-exit path, re-buying at the exit after one more round trip of fees (`post_exit_pnl_at_t − pnl − 0.09`, net %; full table in `_reentry_appendix.csv`) gives:

| Sleeve | After a loss, +15 / +30 min | After a win, +15 / +30 min |
|---|---|---|
| MOM_LONG | −0.06 / +0.09 (n 41) | −0.06 / +0.03 |
| MOM_SHORT | −0.13 / −0.18 | −0.24 / −0.19 |
| BULLRUN | −0.24 / −0.03 | −0.18 / −0.31 |
| SPIKE_FADE | **−1.46 / −1.88** (n 7) | +0.16 / −0.27 |
| FLIP | +0.46 / +0.62 (n 7, CI spans 0) | +0.10 / +0.15 |
| FRENZY_LONG | +1.47 / +1.18 (n 8, CI > 0) | +0.08 / −0.21 |
| LITE | +0.27 / +1.08 (n 9) | +1.18 / +1.25 |
| WILLY | +0.35 / −1.64 (n 3) | −0.03 / −1.00 |

FRENZY_LONG losers bouncing after the stop (8 fills) is the one positive post-loss read. It contradicts the year-long refutation of FRENZY re-entry and sits on a $100 blocker sleeve, so at most it is a watch note. Fade stop-outs keep going against the fade (a squeeze), which is the opposite of a re-entry case.
