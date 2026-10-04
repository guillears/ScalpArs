# ⚡🐻 SURGE (long + short) and BEAR-RUN — deep review of entries and exits (2026-10-04)

Operator ask: *"Re-evaluate deeply SURGE entries and exits to see if both make sense, long and short, or just longs maybe. Same with
BEAR RUN. Deep review and your recommendation."*

Tool: `scripts/surge_bearrun_review.py` (stages `live · events · ctrl · needticks · walk · features · candle · robust · sweep2d · bear ·
bearsweep · validate · report`). Every generated table: `reports/SURGE_BEARRUN_REVIEW_2026-10-04_tables.md`; per-fill data:
`reports/backtest_cache/surge_review/` (`walk.csv`, `events_picks.csv`, `ctrl_picks.csv`, `bear_fills.csv`, `variants_*.csv`, `sweep_*.csv`, `sweep2d.csv`).
Nothing in the bot was changed. Every recommendation below needs your approval before any config change.

---

## 1. Verdicts

| Side | Live record (counted) | Year replica on ticks (live rules) | Does the entry signal work? | Does the exit work? | Recommendation |
|---|---|---|---|---|---|
| **SURGE_SHORT** | 1 fill: SAND −1.21 % (−$223), stop after 3.6 min. 4 earlier fills excluded (bugged first trigger). | 146 trigger windows · 394 fills · WR 70 % · **−0.094 % per trigger, 95 % CI [−0.18, −0.00]** · Jan–Apr −0.10 / May–Oct −0.09 · ≈ −$6.2k over 9 months at the live ticket | **No.** It is no better than the same selection at the same clock time on days with no dump: +0.05 per trigger, CI [−0.06, +0.16]. The picked pairs are no better than the pairs refused for low ATR (−0.07). | **No.** Wins average +0.38 % and stops average −1.21 %, so it needs a 75 % win rate to break even and gets 70 %. None of the 37 exits tested has a CI above 0. | **Turn the side OFF** (`surge_short_enabled` = false). The four sleeve-kill checks were run (§5). |
| **SURGE_LONG** | 0 fills. Its one live trigger (10-02 04:20) found 0 of 20 pairs with ATR ≥ 1.5 %. | 67 triggers (43 with a pick) · 68 fills · WR 49 % · **−0.031 % per trigger, CI [−0.41, +0.38]** · −0.185 % per fill · ≈ −$2.3k over 9 months | **No edge as built.** Same selection on no-trigger days: +0.17 per trigger, CI [−0.29, +0.65]. One conditional looks good (BTC up ≤ +2.7 % over 3 days: 22 triggers, +0.51, CI [+0.01, +1.12]), but a split that large shows up by chance 21 % of the time. | The 0.5×ATR trail beats the live 1×ATR trail by +0.18 (CI [+0.06, +0.31]), but it does worse on no-trigger days. No exit gives a per-trigger CI above 0. | **Cut to probe size** (`surge_long_lev_mult` 0.05) as a forward test of the BTC 3-day-return hypothesis, frozen below. Second choice: OFF. "Just longs" at normal size is **not** supported. |
| **BEARRUN_SHORT** | 0 fills since it shipped. No Bear monitor window has opened since Sep-15 (yr5 replay ledger for Sep-16 → Oct-4 is empty). | yr4 + yr5 engine replays: 7 windows with fills in 9 months (24 monitor windows, most short flickers) · **−0.24 % per window, 3 of 7 positive** · about 15 fills per seed per year | **Not shown.** The design study (+0.28 per fill on a 15-alt proxy) is not reproduced by the engine. Its own kill bar trips in all 3 yr4 seeds (2 net-negative windows in a row). | No variant is positive in window units. Without the EMA13 exit: −0.23 vs −0.24 as replayed; the EMA13 exit only hurts in the Sep-15 window. | **De-arm to probe** (`bearrun_lev_mult` 0.05, the sleeve's documented de-arm). Keep the monitor and ledger. Checklist ran but is under-powered at 7 windows, so this is **not OFF**. EMA13 watch unchanged. |

**Money (live ticket ≈ $18.4k notional per fill, 1 % ≈ $184).**
- Replica as the bot runs now: SHORT about −$700/month (43 fills/month), LONG about −$250/month (7.5 fills/month).
- Bear-Run is immaterial: about 1 window/month, about −$45 per window.
- Master per batch, before → after the proposal: B16 −$1,129 → −$906 (the SAND SURGE_SHORT fill, −$223, removed). Every other batch is unchanged (no SURGE or BEARRUN fills in them).

**The automatic kill bars would not reliably catch this.**
- On the replica distribution, SURGE_SHORT's 10-fill bar (≤ 4 winners or mean ≤ −0.20 %) trips only 31 % of the time, because the win rate is 70 % while expectancy is negative.
- LONG's bar trips 47 % of the time.
- A payoff-asymmetric sleeve passes a win-count bar while losing money. That is why this needs a decision rather than waiting for the auto-kill.

---

## 2. Why the design grids said +0.32 (long) / +0.22 (short) and the replica says ≈ 0 / −0.09

Same 68 long and 394 short picks, three simulators (tables file, "Why the design grids looked positive"):

| side | design simulator (1m candle path, entry at window open, stop filled AT the level) | ticks, no latency, fill at the crossing print | ticks, entry +60 s (live-like) |
|---|---|---|---|
| LONG per trigger | **+0.213** | −0.060 | −0.031 (per fill −0.185) |
| SHORT per trigger | **+0.191** | −0.009 | −0.094 |

- **About +0.25–0.28 % per trade is a simulator artefact.**
  - A 1m candle walks open → high → low → close. The trail arms at the extreme and then exits exactly at its line.
  - On real prints, the noise inside each minute on ≥ 1.5 %-ATR alts hits the tight short trail (peak × 0.65) at lower peaks.
  - Stops fill past the level.
- **Entry latency costs about 0.08 % on shorts.**
- **The long grid's fat tail came from pairs the live sleeve cannot trade.**
  - 56 of its 117 fills were Alpha-subtype (47) or < 90-day listings (9) (SIREN, BEAT, RIVER, early BTW, …), the "top-3 events = 113 %" pumps.
  - The live universe is COIN, non-Alpha, listed ≥ 90 days, top-20 by volume.
- **Lesson for any future sleeve with exits shorter than about 5 minutes:** a 1m-candle backtest is not admissible evidence. Walk ticks.

---

## 3. Entries

**Replica rules: the live functions are called directly.**
- `services.surge.surge_trigger` runs on the BTC 5m bars. Every replica trigger was re-confirmed by the bot function.
- Universe = top-20 tradeable by rolling 24 h volume: COIN, non-Alpha, listed ≥ 90 days, global blacklist / no-trade / side blacklist skipped.
- `surge_pair_pick` runs on 100 closed 5m bars. Picks are taken in rank order, ≤ 4 per trigger, 4 h spacing.
- Entry is the last print at window open + 60 s.

**Window units:** one trigger = one observation.

| cohort | LONG per trigger | SHORT per trigger |
|---|---|---|
| PICK (live rule) | −0.031 [−0.41, +0.38] (43 t) | −0.094 [−0.18, −0.00] (146 t) |
| CTRL: fresh no-trigger selection at the same clock time ±1/±2 days | −0.112 [−0.29, +0.09] | −0.117 [−0.18, −0.06] |
| PICK − CTRL, paired | +0.17 [−0.29, +0.65] | +0.05 [−0.06, +0.16] |
| REFUSED_ATR_LOW (same trigger, ATR < 1.5) | −0.033 | −0.073 |
| REFUSED_NOT_LEADER (long) / MAX_SLOTS (short) | −0.195 | −0.061 |
| CONTROL_SAMEPAIR (the design's control, same pair 24 h earlier) — biased, picked with hindsight | +0.064 | −0.181 |

- **Live funnel (decision journals): 3 live triggers.**
  - 09-30 SHORT: bugged (excluded).
  - 10-02 04:25 LONG: 19 ATR_LOW + MOVR NOT_LEADER, 0 opens.
  - 10-02 14:55 SHORT: 19 ATR_LOW, SAND opened.
  - In today's calm tape the top-20 is mostly majors with 5m ATR 0.2–1.2 %, so the ATR ≥ 1.5 % floor admits about 0–1 pair per trigger.
- **Long vs short.** Neither side has an entry edge. Longs are fat-tailed:
  - 49 % WR, wins +0.90 vs stops −1.21; three triggers carry most of any positive subset.
  - Shorts are thin-tailed and consistently slightly negative: 70 % WR, wins +0.38 vs stops −1.21.
- **Month by month (per trigger).**
  - SHORT is negative in 8 of 10 months, positive only in Jun (+0.08) and Aug (+0.27).
  - LONG is positive in 4 of 9 months, but on 1–9 triggers each.
- **yr5 corrected engine replay (Jul-16 → Oct-4, 3 seeds; crowding with every other sleeve included) agrees.**
  - SURGE_SHORT: −0.084 per window [−0.25, +0.06], 11 of 21 windows positive, −0.029 per fill.
  - SURGE_LONG: −0.24 per fill; +0.13 per window [−0.82, +1.46], 4 of 10 windows positive, carried by one Aug-21 ladder fill of +5.7 %.
  - The replica and yr5 pick the same pairs on every shared trigger. 240 of 248 yr5 fills matched; sign agreement 87 %.

---

## 4. Exits (same fills; Δ vs live, paired per trigger; full tables in the tables file)

### SURGE_SHORT

**Live exit:** `surge_short_exit_for`.
- Stop: −0.70 widened to −1.5×ATR, capped −1.2. On these ≥ 1.5 %-ATR pairs it is always −1.2.
- Trail arms at +0.40, floor peak − min(0.5×ATR, 0.35×peak), plus the ladder.
- Result: 279 trails × +0.38 and 115 stops × −1.21.

| variant | per trigger | CI | Δ vs live |
|---|---|---|---|
| live | −0.094 | [−0.18, −0.00] | — |
| best: fixed TP 5 / SL −3 | +0.139 | [−0.28, +0.56] | +0.23 [−0.18, +0.65] |
| fixed TP 3 / SL −0.7 | +0.029 | [−0.14, +0.20] | +0.12 [−0.04, +0.29] |
| design QUICK exit (−0.5 stop, arm +0.3, 30 min) | −0.117 | [−0.16, −0.07] | −0.02 |
| arm +0.8 / +1.2 · give-back 0.5 · trail 0.5×ATR | −0.09 … −0.16 | | ≤ 0 |
| hold to 240 min, no stop | +0.315 | [−0.63, +1.23] | Jan–Apr +0.96 / May–Oct −0.89 (regime, not an exit) |

**Stop width, read on the live-stopped cohort only, two-sided.** Fills that were not stopped stay unchanged (Δ = 0).
- −2.0: 35 saved / 80 deeper, Δ −0.07 on the stopped fills.
- −3.0: 52 saved / 63 deeper, Δ −0.28.
- −0.7: Δ +0.01 overall (+0.50 on stopped fills, −0.21 on fills it now stops).
- This is consistent with DECISION_LOG 175. **No exit change rescues the side.**

### SURGE_LONG

**Live exit:** the Bull-Run exit with a 1×ATR trail.
- BE arm +1.0, lock +0.2, ladder, stop −1.2.
- Result: 33 trails × +0.90 and 35 stops × −1.21.

| variant | per trigger | CI | Δ vs live |
|---|---|---|---|
| live | −0.031 | [−0.41, +0.38] | — |
| trail 0.5×ATR | +0.148 | [−0.26, +0.58] | **+0.18 [+0.06, +0.31]**, both halves; but on CTRL −0.14 vs −0.11 (not exit-generic) |
| fixed TP 5 / SL −0.7 | +0.445 | [−0.15, +1.11] | +0.48 [−0.01, +0.99]; Jan–Apr −0.10 / May–Oct +1.02 (inconsistent) |
| no BE lock | +0.289 | [−0.36, +1.08] | +0.32 [−0.17, +1.00]; Jan–Apr −0.20 |
| wider stops −2 / −3 / −5 (real exit, only the pre-arm stop moved) | −0.07 … −0.13 | | −0.04 / −0.08 / −0.10; live-stopped: deeper 27 / 23 / 15 vs saved 8 / 12 / 20; not-stopped fills Δ = 0 (two-sided check holds) |
| momentum-long runner exit | +0.001 | | +0.03 |

- After a 30–50 % haircut, the best paired exit gain (+0.09–0.12) still leaves LONG's per-trigger CI spanning 0.
- **An exit change is not enough to justify normal size.**

---

## 5. Sleeve-kill checklist (all four items, before the SURGE_SHORT OFF and the two de-sizes)

`scripts/sweep_separators.py` cannot run on these sleeves: it is hard-wired to `SCREENED_BASELINE.csv`, which holds 0–1 SURGE / BEARRUN fills. The equivalent was run on the replica fills.

**① Every pair-level entry dimension.** ATR, rank, pair 30-min move, EMA stack (rebuilt for every fill), vs EMA50, pair 24 h / 4 h return and volume ratio, at sign / median / outer terciles, with trigger-clustered CIs.
- SHORT: one survivor, pair volume ratio top tercile, Δ +0.22 [+0.02, +0.42]. Its good side is only +0.03 per trigger.
- LONG: none with CI excluding 0.

**② Every macro / regime dimension, sign first.** BTC 1 h EMA20 slope, BTC vs 1 h EMA50, BTC 24 h and 72 h returns, eff24, BTC ATR, BTC off 30-day high, top-50 breadth, trigger size, volume multiple, hour, weekend.
- SHORT, at sign: BTC 1 h slope > 0, Δ +0.21 [+0.04, +0.39]; BTC above its 1 h EMA50, Δ +0.21 [+0.02, +0.38]. Both good sides are only +0.03 / +0.02 per trigger, and the May–Oct Δ is ≈ 0.
- LONG: BTC 72 h return ≤ median, Δ −1.11 [−1.83, −0.43], consistent across halves.
- Shuffled null over all macro splits: LONG p = 0.21, SHORT p = 0.26. Neither best split beats chance.
- Exhaustive 2D: every pair of the 19 variables, 684 quadrants. Best LONG quadrant +0.67 (null p 0.42). Best SHORT +0.19 (null p 0.49; May–Oct −0.17).
- **No separator survives.**

**③ Uniform-degradation test.** There is no degradation to explain. SHORT is flat at about −0.1 per trigger in every quarter and every cohort (ATR terciles, rank halves, stack states). The "good period" in the design was the 1m-candle simulator, not a regime.

**④ Tape context.**
- SHORT loses in falling months (Jan −10 % BTC: −0.05; Feb −15 %: −0.15) and in rising months (Apr, Jul, Sep). It is positive only in Jun (−21 % BTC, +0.08) and Aug (+25 %, +0.27). No tape where it pays repeatedly.
- LONG is mixed, with too few triggers per month.

**Bear-Run.** ① + ②: 49 stamped `entry_*` columns, 82 splits, in replicate-window units. With 7 windows, every split has 3–6 windows a side, so nothing can be judged (17 splits show a side above +0.10, about the chance level). ③ / ④: the windows run from −0.80 to +0.45 with no order by depth (Jan-29, r24 −8 %: −0.66; Jun-02, −7.5 %: +0.45). **Under-powered, so the recommendation stops at de-arm, not OFF.**

---

## 6. Watch items touched

- **SURGE EMA-stack watch** (live gate: N ≥ 30 per side over ≥ 8 windows; live N = 0 / 1, so it stays open). Replica, stack rebuilt for every fill:
  - LONG: stacked +0.002 (51 fills); not stacked −0.51 (12); opposite −0.07 (5).
  - SHORT: bear-stacked −0.075 (176); not stacked −0.078 (100); bull-stacked −0.158 (118).
  - No cohort passes the expectancy filter bar. A stack filter would not rescue either side.
- **Bear-Run EMA13-exit watch** (live N = 0, so it stays open). Replica: 20 EMA13 exits in 3 windows.
  - Without the EMA13 exit: Sep-15 +0.64 per fill (17 fills, 6 replicates of one window); Jan-29 −0.17; Mar-27 −0.14.
  - This is the same one-window pattern as DECISION_LOG 148. Unchanged.
- **The SURGE automatic kill bars** stay as they are. They are a coin flip on this payoff shape (§1), so the decision should not wait for them.

## 7. Proposed config (not applied — needs your OK; D11 fields already exist, so no code change)

| field | now | proposed | pre-committed gate |
|---|---|---|---|
| `surge_short_enabled` | true | **false** | Reopen only for a new entry/exit design that passes on THIS tick replica: ≥ 20 triggers, per-trigger CI > 0, both halves > 0, beats the fresh CTRL in both halves. Then observe-first. 1m-candle evidence is not admissible. |
| `surge_long_lev_mult` | 1.0 | **0.05** (probe) | Frozen hypothesis: BTC 72 h return ≤ +2.7 % at the trigger bar (rebuild from klines at review; do not re-fit the cut). After ≥ 8 trigger windows with fills: arm 1× with that gate only if that cohort's per-trigger mean > 0 with window-bootstrap CI > 0, and the > +2.7 % cohort ≤ 0. If the ≤ +2.7 % cohort's mean ≤ 0 at 8 windows → `surge_long_enabled` false. Expect about 2–4 windows/month in volatile tapes, about 0 in calm ones. |
| `bearrun_lev_mult` | 1.0 | **0.05** (the documented de-arm) | Re-arm 1× only after ≥ 5 live windows with fills: ≥ 3 positive and Σ > 0. The existing kill bar (2 consecutive net-negative windows → `bearrun_sleeve_enabled` false) stays. |

**Haircut.** The only positive projection, LONG under the r72 gate (+0.51 per trigger), is in-sample from a sweep, so expect +0.25–0.36 at best. It stays a probe hypothesis. Nothing is armed on it.

**Shared slots.** Probe-sized fills still occupy one of the 4 global slots (median SURGE hold is minutes; Bear-Run about 30 min). That is accepted as the cost of keeping a forward test.

## 8. Fidelity, and what this review could NOT test

**Checks run (`stage validate`).**
- **Live triggers match.** The live 10-02 LONG trigger was reproduced 20/20 (pairs and refusal reasons). The 10-02 SHORT trigger was 19/20; the rank-20 pair differs (replica CT, live LINK).
- **The live SAND fill is reproduced exactly.** Re-walked from its real open time and price: −1.206 STOP after 3.6 min, vs live −1.207 STOP after 3.6 min.
- **yr5 agreement.** 240 of 248 yr5 SURGE fills matched to replica picks, sign agreement 87 %, mean yr5 −0.060 vs replica −0.081.
- **Bear-Run re-walk.** Replay fills that closed on stop / trail / ladder re-walk to Δ −0.022 (median |Δ| 0.027).
- **Generic suite.** `scripts/validate_against_master.py`: ALL CHECKS PASS. It was not extended (another agent has uncommitted edits in it); the tool's own `validate` stage covers this review.

**Limits.**
- The replica ignores the 4 shared global slots, so its fill count is an upper bound. yr5 includes crowding, and it agrees.
- Universe Alpha flags are today's `exchange_info` snapshot.
- Entry latency is fixed at 60 s (live 30–120 s). Zero latency would still be ≈ 0 for SHORT (−0.01).
- Exits are checked on every price-changing aggTrade. The live WebSocket cadence can only add stop slippage.
- **yr5 was used only for Jul-16 → Oct-4** (6 half-month chunks × 3 seeds). Jan → Jul-15 was still running (it runs backwards, about 2.4 h per chunk round). `venv/bin/python scripts/surge_bearrun_review.py bear && … validate && … report` refreshes these tables when it finishes. SURGE in yr4 is invalid (BUGHUNT_D B2) and was not used.
- Bear-Run momentum-only exits (EMA13, signal-lost, …) are taken from the engine as replayed, not re-simulated. Re-walk variants exclude them.
- The replica, yr4 and yr5 all run on the same tape, so they are not independent. The live forward test is the only out-of-sample evidence, and it currently has N = 1 (SURGE) / 0 (Bear-Run).
