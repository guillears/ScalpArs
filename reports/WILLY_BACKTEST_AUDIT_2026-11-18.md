# WILLY / WILLY_HT backtest audit — 2026-11-18

Research only. No code, config, test or CLAUDE file was changed.
Files: this report · `WILLY_BACKTEST_AUDIT_2026-11-18_trades.csv` (every rebuild trade 10-08 → 11-17, classified) · `WILLY_BACKTEST_AUDIT_2026-11-18_beforeafter.csv` (every variant × sleeve × half-year) · `WILLY_BACKTEST_AUDIT_2026-11-18.png` (chart).

## 1. Question

The real master fills are strongly positive (WILLY 24 kept fills · 71 % · +2.75 %/trade; HT 15 kept · 80 % · +1.78). The full-history rebuild is negative (WILLY 545 · −0.08; HT 417 · −0.27). In the live window the rebuild's HT trades that the real bot did not take were 22 · 55 % · −0.45, and the matched ones 13 · 69 % · +0.85.

Is that because the rebuild is missing a real guard, so its extra trades are trades the bot would block? If so, would adding the guard flip the full-history verdict?

## 2. Plain answer

**No. The backtest is not missing a guard that would rescue either sleeve.**

- Where the real bot and the rebuild overlap, they agree. The rebuild finds **36 of the 39 master-kept fills**, at the same 5-minute bar (median 10 s apart). The 94 real fills that today's filters block are all absent from the rebuild, as they should be.
- On those 36 shared trades the rebuild is **more generous** than the master: Σ +78.9 % vs master Σ +42.3 %. Most of the gap is 3 runner exits (HUMA, INIT, BARD). So the rebuild's exits are not too harsh.
- The rebuild's extra trades are **not** trades a guard would block. Most were not taken because of **when** they happened: HT was not live yet (before 10-21), Alpha coins were not read by Frenzy yet (before 11-04), WILLY had only 1 slot (before 11-13), or the bot restarted. Today's config would take all of these.
- Only 5 extras were blocked by a real guard: the depth guard (2 losers), the dislocation guard (2 big winners) and an open-path refusal (1 winner). Together they **remove more profit than loss**.
- The one guard the rebuild really lacks and that can be rebuilt exactly (Frenzy's 25-coin shortlist cap) helps a little: WILLY −0.08 → −0.05, HT −0.27 → −0.22 per trade, 2025 onward. **No verdict flips.**
- The master looks good because it covers only **Oct 9 → Nov 17, 2026**. The rebuild says the same about that window: WILLY 2026-H2 is **+1.02 %/trade** (CI +0.14 to +2.27). The negative full history comes from **2025-H2 and 2026-H1**, which the master never saw.
- **Neither sleeve passes** any of the four proof tests in any variant.

## 3. Method

1. **Match.** Every rebuild trade from 2026-10-08 12:13 (WILLY switched on) to 11-17 (87 trades) was matched to a real WILLY or HT fill on the same pair within ±15 min. The real fills come from every export: BASELINE19–27 plus `exports_archive/` and the newest Downloads files (136 fills, deduplicated).
2. **Find the cause.** For each unmatched trade, I read:
   - every WILLY journal row (909 rows, 10-14 → 11-18, from all decisions exports);
   - CloudWatch `[FRENZY_WILLY…]` and `DEPTH_GUARD` lines (retention 14 days, so 11-04 onward);
   - the 113 "Application startup complete" restart lines;
   - the open book at that minute (any strategy);
   - the `git log -p trading_config.json` timeline for every WILLY switch;
   - the Binance Alpha subtype.
   Every trade got exactly one cause, with evidence quoted in the CSV.
3. **Reverse match.** Every real fill was checked against the rebuild, with the master's `stack_keep` and block reason.
4. **Guards.** A guard counts as rebuildable only if the engine code defines it and history has the data for it. I rebuilt one exactly: the shortlist cap (G1). Three more are run as sensitivities (S1–S3) because they are either not guards or not rebuildable exactly. Each variant was re-run on the full history (2025-01 → now, and 2023-H2 holdout), per sleeve and per half-year.
5. **Stats.** All at 0.10 % slippage unless stated.
   - Day-clustered bootstrap CI: 2,000 draws.
   - Kelly: at the observed WR and at the one-sided 95 % Wilson floor, with n = distinct trading days (the daily-clustering assumption).
   - Best-5 %-out: drop the best 5 % of trades.
   - Slippage test: every trade re-priced at 0.20 %.
   - DCR: one size, X = 1 (notional = 100 % of equity per trade). Scale it linearly for today's X.

## 4. Cause table — every rebuild trade, 2026-10-08 → 11-17 (87 trades)

Net % is the rebuild's own price for that trade (today's exits, 0.10 % slip).

| cause | what it means | N | WR | avg net % | Σ % |
|---|---|---:|---:|---:|---:|
| MATCHED | a real fill on the same pair ±15 min | 38 | 68 % | +2.00 | +76.1 |
| A · HT not live | R ≥ 1 before 2026-10-21 01:49 → turnover-refused (HT did not exist yet) | 12 | 83 % | +0.56 | +6.7 |
| B · Alpha excluded | Alpha coin; Frenzy read no Alpha coin before 2026-11-04 19:32 (commit a6d9cdb2) | 14 | 57 % | +2.48 | +34.8 |
| C · one WILLY slot | `frenzy_willy_max_slots` = 1 until 11-13; the slot was taken | 2 | 100 % | +1.26 | +2.5 |
| D · restart | cold pair after a restart → "already flagged … > 3-bar late window" / NOMISS_MISSED | 2 | 50 % | −1.36 | −2.7 |
| J · bot down | no journal row of any kind 10-21 15:00 → 10-22 15:00 | 1 | 100 % | +1.25 | +1.3 |
| E · re-anchored episode | the rebuild fired entry A on an episode that was 9–59 h old after its spike time shifted by one bar; the real bot saw no new flag (or was cold) | 6 | 67 % | −0.90 | −5.4 |
| E · not read / not flagged | the real bot never read or flagged the pair at that bar (signal not reproduced) | 4 | 100 % | +1.68 | +6.7 |
| G · filter divergence | real: HT FAILED_SPIKE block (live VWAP stamp); the rebuild's VWAP kept it | 1 | 100 % | +2.01 | +2.0 |
| F · DISLOC guard | live price moved 1.35 % / 1.90 % within ~1 s of the decision (CYBER, ARIA) | 2 | 100 % | +6.26 | +12.5 |
| H · DEPTH guard | live only: book held $2k / $5k within 0.15 % vs $17.5k planned (TAC, SOPH) | 2 | 0 % | −4.07 | −8.1 |
| I · open path refused | live: "slots / balance / cooldown / price moved" (KAVA) | 1 | 100 % | +2.03 | +2.0 |
| K · unknown | before the journal window (CKB 10-11) / no WILLY row (STX 10-13) | 2 | 50 % | −1.42 | −2.8 |

By sleeve, the HT extras after 10-22 (the operator's 22 · −0.45) break down as:

- Alpha-era exclusion: 7 HT · 29 % · −1.75;
- re-anchored episodes: 6 · −0.90;
- depth guard: 2 · −4.07;
- restart: 1 · −4.10;
- the rest (disloc, not read, filter divergence, open refused): 6 · all winners.

**Real guards block 2 losers and 3 winners (Σ +6.4 % removed).** So the hypothesis "the extras are trades a guard would block" is refuted. Most extras are trades today's config **would** take.

**Reverse: real kept fills the rebuild missed (3, all winners).**

| fill | why the rebuild missed it |
|---|---|
| AKE 11-07 01:55 (+1.25) | The rebuild entered the same episode at 03:25 instead (−4.1). |
| JCT 11-08 17:50 (+8.0) | The real bot took a re-anchored entry A on a 7.7 h-old episode. The rebuild did not. |
| IDOL 11-13 16:30 (+2.0) | The rebuild spent the episode's one entry A at 15:35, the bar the real bot blocked with FAILED_SPIKE. |

Filter-stamp disagreements run both ways: ARK 10-23 and UAI 11-08 were kept by the rebuild but blocked by the master stack; IDOL 15:35 is the opposite. That is 3 of ~135.

## 5. Guards the rebuild lacks, and before → after

| guard | engine cite | rebuildable? | effect |
|---|---|---|---|
| **G1 shortlist cap**: non-Alpha top 25 + Alpha top 25, \|chg\| qualifiers first, then biggest move | `services/trading_engine.py:3333` `frenzy_shortlist`, `:3372` `frenzy_shortlist_parts`, read order `:3352` | **Yes, exactly.** The pair's rank among the full 922-pair universe at each pass from the spike bar to spike + 3 bars (the cold window). | Cuts 74 WILLY (−0.34 avg) and 31 HT (−0.59 avg). In the live window it cuts only MAGIC 10-15, which is consistent with real logs (the cap bound on 178 of 3,909 passes, 4.5 %). |
| **DISLOC** (live price vs decision price > 1 %) | `frenzy_max_entry_dislocation_pct`, WILLY open path | **No.** It is a ~1-second move. A 1-minute proxy (first-minute high ≥ 2.41 %, tuned to the live ~6 % refusal rate) is S2, low fidelity. | The proxy's blocked trades are winners (WILLY 41 · +2.21; HT 35 · +2.19). Missing it makes the backtest **optimistic**. |
| **DEPTH guard** (book within 0.15 % ≥ 0.5 × planned notional) | `trading_engine.py` ~15940–15990 | **No.** There is no order-book history, it depends on size, and it applies live only. | Both live cases were losers. The bias direction is unknown (N = 2). |
| **Pair held by any strategy** (any open position on the pair) | `trading_engine.py` `_frenzy_open`, the "any open position" check | No. It needs the other Frenzy sleeves simulated. It delays (stays armed) rather than blocks. | Real journal: 2 PAIR_HELD rows in 35 days. Negligible. |
| **Turnover R** from live mcap | `_willy_turnover` → `mcap_service` | Partly. The rebuild uses today's circulating supply. | Matched R values agree (AKE 4.27 vs 4.60, INIT 9.73 vs 10.01, ONE 1.82 vs 1.82). For 2025, today's larger supply makes past R too small, so some past HT trades are filed as WILLY. |

Sensitivities. These are **not** missing guards:

- **S1, no re-anchored A** (spike > 1 h old). The engine does fire entry A on a re-anchored spike when the pair is warm: 9 of 135 real fills were > 1 h after the spike. The real bot avoids most of them only because it restarts ~6 times a day. This is engine behaviour, so the rebuild is right to include it. HT has 114 of these at −0.39.
- **S2, the DISLOC proxy.** Described above.
- **S3, no Alpha coins.** This was the rule before 11-04. Today Frenzy reads Alpha. The labels used are today's.

### Before → after, per half-year (avg net %/trade, N, 95 % day-CI)

| sleeve | variant | 2023-H2 (holdout) | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 |
|---|---|---|---|---|---|---|
| WILLY | BASE | −0.18 (99) [−0.60, +0.26] | −0.08 (92) | **−0.70** (175) [−1.44, −0.11] | −0.23 (159) | **+1.02** (119) [+0.14, +2.27] |
| WILLY | +G1 cap | −0.18 (91) | −0.25 (79) | −0.68 (136) | −0.18 (151) | +1.04 (113) |
| WILLY | S1 | −0.18 (97) | −0.08 (92) | −0.68 (174) | −0.23 (156) | +1.02 (118) |
| WILLY | G1+S1+S2 | −0.29 (86) | −0.30 (76) | −1.00 (121) | −0.35 (139) | +0.84 (105) |
| HT | BASE | −0.21 (201) | +0.14 (52) | −0.11 (91) | **−0.81** (113) [−1.36, −0.26] | −0.11 (161) |
| HT | +G1 cap | −0.27 (184) | +0.21 (48) | −0.00 (85) | −0.76 (111) | −0.09 (159) |
| HT | S1 | −0.30 (179) | +0.30 (42) | +0.04 (69) | −0.75 (90) | −0.03 (124) |
| HT | G1+S1+S2 | −0.51 (158) | +0.28 (37) | −0.30 (54) | −0.84 (83) | −0.18 (111) |

2023-H2 has no market caps, so every entry runs under each sleeve's rule. WILLY and HT there are the same entries with different exits.

### Totals, 2025-01 → now

| sleeve | variant | N | WR | avg | 95 % day-CI | Kelly @ floor | best-5 %-out | avg @ 0.20 % slip | DCR %/day (X = 1) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| WILLY | BASE | 545 | 59.6 % | −0.083 | [−0.47, +0.30] | −0.15 | −0.58 | −0.21 | −0.149 |
| WILLY | +G1 cap | 479 | 60.3 % | −0.045 | [−0.40, +0.37] | −0.14 | −0.56 | −0.17 | −0.089 |
| WILLY | S1 | 540 | 59.8 % | −0.073 | [−0.46, +0.33] | −0.15 | −0.58 | −0.20 | −0.141 |
| WILLY | S2 | 508 | 58.9 % | −0.256 | [−0.62, +0.10] | −0.26 | −0.71 | −0.39 | −0.262 |
| WILLY | S3 | 482 | 59.8 % | −0.130 | [−0.49, +0.21] | −0.19 | −0.59 | −0.27 | −0.155 |
| WILLY | G1+S1+S2 | 441 | 59.2 % | −0.238 | [−0.60, +0.15] | −0.25 | −0.70 | −0.37 | −0.205 |
| HT | BASE | 417 | 63.8 % | −0.268 | [−0.55, +0.04] | −0.30 | −0.57 | −0.37 | −0.203 |
| HT | +G1 cap | 403 | 64.3 % | −0.217 | [−0.52, +0.10] | −0.27 | −0.52 | −0.32 | −0.166 |
| HT | S1 | 325 | 63.4 % | −0.172 | [−0.51, +0.19] | −0.24 | −0.49 | −0.27 | −0.111 |
| HT | S2 | 387 | 62.8 % | −0.445 | [−0.74, −0.16] | −0.44 | −0.68 | −0.55 | −0.289 |
| HT | S3 | 345 | 65.5 % | −0.183 | [−0.50, +0.14] | −0.27 | −0.46 | −0.28 | −0.121 |
| HT | G1+S1+S2 | 285 | 62.8 % | −0.334 | [−0.67, +0.01] | −0.36 | −0.58 | −0.44 | −0.165 |

**Pass tests:** every half-year > 0, best-5 %-out > 0, Kelly floor > 0, > 0 at 0.20 % slip. Every variant **fails all four** for both sleeves.

The breakeven win rate is around 63 % for WILLY (win:loss ratio b ≈ 0.6) and around 69 % for HT (b ≈ 0.45). Observed WRs sit below breakeven in most half-years.

![per half-year](WILLY_BACKTEST_AUDIT_2026-11-18.png)

## 6. Matched-trade divergence (38 matched, 36 master-kept)

| metric | value |
|---|---|
| entry time, real − rebuild | median +10 s, p90 +13 s (two at +5 min: ARK, UAI armed one bar later) |
| entry price, real vs rebuild's next-bar open | median +0.08 %, p90 \|diff\| 0.55 % (the rebuild adds 0.10 % slip on top) |
| net %, rebuild − master `stack_pct` | median +0.00, p90 \|diff\| 5.3 % (runner trades) |
| Σ net on the 36 kept | rebuild +78.9 % · master stack +42.3 % · as traded +23.5 % |
| win/loss agreement (rebuild vs master stack) | 83 % |
| stop agreement | 89 % |

Where they disagree:

- **Runners.** HUMA +15.9 vs +1.25, INIT 11-09 +23.0 vs +1.25, BARD +12.6 vs +1.51. The rebuild applies today's runner (on since 11-13) to older fills. The master builder kept those fills at TP.
- **The rebuild stopped, the real bot took TP.** RLC 10-15 (real entry 0.49 % lower, TP in 53 s) and H 11-12. The rebuild assumes the stop is hit first when both are inside one 1-minute bar.
- **The opposite.** DUSK 10-15 hit its stop as traded but reads as TP in the rebuild. The master re-price gives it +0.25.

Net: on the trades both took, the rebuild is optimistic, not pessimistic.

## 7. What it means

1. The gap between master and backtest is mostly **time, not modelling**.
   - The master is 5 weeks of 2026-H2. In those weeks the rebuild agrees on WILLY: +2.96 (39 trades) vs master +2.75 (24 kept).
   - The full history is negative because of 2025-H2 (WILLY −0.70) and 2026-H1 (HT −0.81). Both are periods with no real fills.
2. HT's master edge (+1.78) is **partly a selection effect from config history**:
   - Before 11-04 Frenzy read no Alpha coin. Rebuild HT trades on Alpha coins in that window were 7 · 29 % · −1.75. Over the full history, Alpha HT is 72 · 56 % · −0.68 vs non-Alpha HT at −0.18.
   - Many 10-26 → 11-17 re-anchored entries were skipped only because the bot was cold after restarts.
   - The two depth-guard skips were both losers.
   - Today's config trades Alpha and has 10 slots, so the master's HT number overstates what today's config would earn.
3. The rebuild has one real gap, the shortlist cap. Adding it moves WILLY by +0.04 and HT by +0.05 %/trade (2025 onward). It does not flip any half-year.
4. The DISLOC guard is not modelled, and the evidence says that flatters the backtest: it blocks fast bounces, which are winners.

## 8. Verdict

**Is the WILLY / HT backtest wrong?** Not in a way that changes the decision.

- It reproduces 92 % of the master's kept fills at the same bar, and agrees on the 94 fills today's filters block.
- On shared trades it is about +1.0 %/trade **more** generous than the master re-price (Σ +78.9 vs +42.3 over 36).
- Its extra trades are explained by config eras, restarts and episode re-anchoring. None is a guard-blocked loser.

Adding the one engine-exact missing guard changes the full-history average by about +0.04 (WILLY) and +0.05 (HT) %/trade.

**Does fixing it prove either sleeve?** No. After every fix and sensitivity:

| sleeve | full-history avg (2025 →) | negative half-years | Kelly at floor | best-5 %-out | at 0.20 % slip |
|---|---|---|---|---|---|
| WILLY | −0.05 to −0.26 | 2025-H2 (CI below 0) | < 0 | < 0 | < 0 |
| HT | −0.17 to −0.45 | 2026-H1 (CI below 0) | < 0 | < 0 | < 0 |

Each sleeve also has its 2023-H2 holdout negative.

## 9. Assumptions and limits

- The real bot's journal starts 2026-10-13 18:00 and CloudWatch keeps 14 days (11-04 onward). Causes for 10-09 → 10-13 rest on the slot ledger and config history only (2 left as unknown).
- Alpha status is today's `underlyingSubType`. Coins move in and out of Alpha, which affects S3 and the G1 groups for 2025.
- G1 uses closed-bar 24 h stats from 5m klines (not the live ticker) over the 922-pair vision universe. "Read" means rank ≤ 25 at any pass from the spike bar to spike + 3 bars. That is lenient: it ignores R_NOMISS late reads beyond 3 bars, so G1 may under-block slightly.
- The S2 DISLOC proxy is low fidelity. The 1-minute high does not see the ~1-second move live measures. On live-window matched trades it blocks fills the real bot took. Use it only for direction.
- The depth guard cannot be rebuilt. The real cases are N = 2.
- Re-pricing the 1-minute path assumes the stop fills first when the stop and TP are inside the same bar.
- DCR is at X = 1 notional per trade. It scales roughly linearly with today's X.
- Kelly "floor" uses the one-sided 95 % Wilson bound with n = distinct trading days (daily clustering) and the pooled win/loss ratio.
- All figures are in-sample descriptive reads. Nothing here is a ship candidate. S1 and S3 would need the observe-first route.

## 10. Next step

1. Stop treating the master as proof of edge for HT. Five weeks of 2026-H2, an Alpha-free first half and restart-suppressed re-anchors make it a biased sample of today's config.
2. Engine question for the operator: should entry A fire on a re-anchored spike of an episode that is already running?
   - Today this depends on whether the pair happens to be warm (restarts suppress it). It is not a design rule.
   - Over the full history these entries average −0.39 (HT, 114 trades).
   - Observe-only line, not an arm.
3. If WILLY stays on, it is a 2026-H2 regime bet (+1.02, CI > 0) on a sleeve whose other three half-years and its holdout are negative. Per the profitable-machine rule that is paper, not live.
4. Optional fidelity work, which would not flip the verdict:
   - add G1 to `wsignals.py`;
   - price DISLOC from futures aggTrades for the 2026 trades only;
   - stamp every restart into the journal, so cold-pair misses become countable without CloudWatch.
