# HT FAILED_SPIKE parity — live engine vs the WILLY rebuild, all 49 real HT fills — 2026-11-19

Research only. No engine code, config, test or CLAUDE file was changed. The two scripts need `WILLY_REBUILD_DIR` = a local copy of the WILLY rebuild (its `code/` and `sig_all.pkl`); that rebuild is local research data, not in the repo, so the per-fill CSVs here are the reproducible record.

Files:
- this report
- `HT_FAILED_SPIKE_PARITY_2026-11-19.csv` — one row per real HT fill: both spike anchors, VWAPs, bar closes, vs-VWAP values, the three decisions, the norm (normal hour) each side used
- `HT_FAILED_SPIKE_PARITY_2026-11-19_fullhist.csv` — every unfiltered rebuild HT entry (1,142), its flip share under the live norm spread, and the +5 / −4 / 24 h price of every entry that can flip
- scripts: `scripts/ht_failed_spike_parity.py` (per fill), `scripts/ht_failed_spike_parity_fullhist.py` (full history). Both need `WILLY_REBUILD_DIR` = the folder holding the rebuild's `code/` and `sig_all.pkl`.

## 1. Question

Does the WILLY rebuild compute FAILED_SPIKE (`watch_block_willy_ht_failed_spike`, DL 511 / 527) differently from the live engine? The 11-18 audit compared only 38 matched fills. This check covers all 49 real FRENZY_WILLY_HT fills, including the 34 the filter blocks today.

## 2. Plain answer

**No. The rule is the same in both. The decisions agree on all 47 fills where the rebuild sees the episode.**

- **Same code.** The engine and the rebuild call the same function, `services/watch_filters.willy_blocks(ht=True)`, with the same input: `frenzy_walk(...)['vs_vwap_pct']` rounded to 3 dp.
  - That value is the close of the last closed 5m bar (the red signal bar) vs the VWAP anchored at the spike lead bar.
  - The rule is block when the value is ≤ 0 %.
  - Engine: `services/trading_engine.py` ~13445. Rebuild: `wsim.run_portfolio` → `wsignals.red_rows`.
  - The two also use the same bar window (1,499 closed bars) and refuse in the same place in the gate order (after DISLOC, and the refusal ends the entry).
- **`today_keep` comes from the live stamps.** It equals the live decision on 49 of 49 fills, and also equals `stack_keep` in MASTER_POOL. It was not computed from the rebuild.
- **Decisions:**
  - live = today_keep on 49 / 49;
  - live = rebuild on 47 / 49;
  - the other 2 are not a different decision: the rebuild has **no episode at all** on that bar, so it would never have entered.
- **Values:** 43 of the 47 are identical to 0.001 %. In 4 the anchor sits one bar off or on another spike, but the sign is the same (INIT 11-11, ICNT 11-08 16:15, JCT 11-06, IDOL 11-13).

## 3. Method

1. **Fills.** Took the 49 rows of `WILLY_BACKTEST_WHY_2026-11-18_fills.csv` with sleeve FRENZY_WILLY_HT and matched each one to MASTER_POOL on pair + opened_at (±2 s): 49 / 49 matched, no catch-up entries.
2. **LIVE decision.** `willy_blocks` on `entry_frenzy_vs_vwap_pct`.
   - The live bar close is computed as `entry_frenzy_vwap × (1 + vs / 100)`. This is the walk's `price`, the signal-bar close, not the fill price.
   - Fills opened before the 11-11 arm have no journal block row, because the filter was off then.
3. **REBUILD decision.** `frenzy_walk` run on the rebuild's own 5m data (`wdata.load_5m`) at the closed bar before the live pass, with the rebuild's normal-hour model (copied from `wsignals.py`).
   - The rebuild's stored `sig_all.pkl` red row is quoted where it exists (40 / 49). Where it exists, it equals the re-walk exactly.
4. **Cause test.** The live normal hour is implied by the stamp: `entry_frenzy_vol_mult` = last-hour quote volume / norm. I re-walked each fill with that norm.
5. **Full history.** Re-walked all 1,142 unfiltered rebuild HT entries (the H7 OFF run) with the norm scaled by each of the 49 measured live / rebuild norm ratios.
   - I counted how often the decision flips.
   - Every entry that could flip was priced at +5 / −4 / 24 h on 5m bars: stop checked first, 0.10 slip, fees 0.07 / 0.10.

## 4. Per-fill table

Spikes are the episode anchor (spike bar close, UTC). Δ VWAP = rebuild / live − 1.

| # | pair | opened (UTC) | live spike | rebuild spike | live VWAP | rebuild VWAP | Δ VWAP % | live bar close | rebuild bar close | live vs VWAP % | rebuild vs VWAP % | live | rebuild | today_keep | agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | JCTUSDT | 2026-11-11 13:05 | 11-11 13:00 | 11-11 13:00 | 0.00748786 | 0.00748786 | 0.00 | 0.00735099 | 0.007351 | -1.828 | -1.828 | BLOCK | BLOCK | BLOCK | yes |
| 2 | INITUSDT | 2026-11-11 11:00 | 11-09 19:15 | 11-09 19:10 | 0.226516 | 0.226467 | -0.02 | 0.181861 | 0.18186 | -19.714 | -19.697 | BLOCK | BLOCK | BLOCK | yes |
| 3 | JCTUSDT | 2026-11-10 21:35 | 11-10 08:25 | — | 0.00622093 | — | — | 0.00528001 | — | -15.125 | — | BLOCK | NO_EPISODE | BLOCK | no episode in rebuild |
| 4 | ONUSDT | 2026-11-10 15:55 | 11-10 15:50 | 11-10 15:50 | 0.106997 | 0.106997 | 0.00 | 0.1072 | 0.1072 | 0.190 | 0.190 | KEEP | KEEP | KEEP | yes |
| 5 | UAIUSDT | 2026-11-10 08:30 | 11-09 10:25 | 11-09 10:25 | 0.0408665 | 0.0408665 | 0.00 | 0.0377001 | 0.0377 | -7.748 | -7.748 | BLOCK | BLOCK | BLOCK | yes |
| 6 | ONEUSDT | 2026-11-10 07:45 | 11-10 07:30 | 11-10 07:30 | 0.00155586 | 0.00155586 | 0.00 | 0.00156889 | 0.0015689 | 0.838 | 0.838 | KEEP | KEEP | KEEP | yes |
| 7 | AKEUSDT | 2026-11-10 00:50 | 11-10 00:45 | 11-10 00:45 | 0.00258039 | 0.00258039 | 0.00 | 0.002514 | 0.002514 | -2.573 | -2.573 | BLOCK | BLOCK | BLOCK | yes |
| 8 | UAIUSDT | 2026-11-09 10:20 | 11-09 10:15 | 11-09 10:15 | 0.0446005 | 0.0446005 | 0.00 | 0.0443998 | 0.0444 | -0.450 | -0.450 | BLOCK | BLOCK | BLOCK | yes |
| 9 | JCTUSDT | 2026-11-08 17:50 | 11-08 10:10 | 11-08 10:10 | 0.0104132 | 0.0104132 | 0.00 | 0.011967 | 0.011967 | 14.922 | 14.922 | KEEP | KEEP | KEEP | yes |
| 10 | ICNTUSDT | 2026-11-08 16:15 | 11-08 07:40 | 11-08 07:35 | 0.365164 | 0.364816 | -0.10 | 0.359098 | 0.3591 | -1.661 | -1.567 | BLOCK | BLOCK | BLOCK | yes |
| 11 | ICNTUSDT | 2026-11-08 07:55 | 11-08 07:35 | 11-08 07:35 | 0.362972 | 0.362972 | 0.00 | 0.3725 | 0.3725 | 2.625 | 2.625 | KEEP | KEEP | KEEP | yes |
| 12 | CYBERUSDT | 2026-11-08 05:55 | 11-08 05:40 | 11-08 05:40 | 0.321277 | 0.321277 | 0.00 | 0.3086 | 0.3086 | -3.946 | -3.946 | BLOCK | BLOCK | BLOCK | yes |
| 13 | UAIUSDT | 2026-11-08 04:30 | 11-08 04:25 | 11-08 04:25 | 0.0818697 | 0.0818697 | 0.00 | 0.0816003 | 0.0816 | -0.329 | -0.329 | BLOCK | BLOCK | BLOCK | yes |
| 14 | AINUSDT | 2026-11-08 04:10 | 11-08 03:55 | 11-08 03:55 | 0.139534 | 0.139534 | 0.00 | 0.13929 | 0.13929 | -0.175 | -0.175 | BLOCK | BLOCK | BLOCK | yes |
| 15 | ACTUSDT | 2026-11-07 08:45 | 11-07 08:35 | 11-07 08:35 | 0.0116984 | 0.0116984 | 0.00 | 0.0116081 | 0.011608 | -0.772 | -0.772 | BLOCK | BLOCK | BLOCK | yes |
| 16 | AKEUSDT | 2026-11-07 01:55 | 11-07 01:15 | 11-07 01:15 | 0.00527489 | 0.00527489 | 0.00 | 0.00555699 | 0.005557 | 5.348 | 5.348 | KEEP | KEEP | KEEP | yes |
| 17 | AINUSDT | 2026-11-07 01:00 | 11-07 00:55 | 11-07 00:55 | 0.126651 | 0.126651 | 0.00 | 0.12419 | 0.12419 | -1.943 | -1.943 | BLOCK | BLOCK | BLOCK | yes |
| 18 | ONUSDT | 2026-11-06 13:45 | 11-06 13:30 | 11-06 13:30 | 0.199447 | 0.199447 | 0.00 | 0.1984 | 0.1984 | -0.525 | -0.525 | BLOCK | BLOCK | BLOCK | yes |
| 19 | JCTUSDT | 2026-11-06 05:10 | 11-05 05:55 | 11-05 05:50 | 0.00482695 | 0.00484171 | 0.31 | 0.00302399 | 0.003024 | -37.352 | -37.543 | BLOCK | BLOCK | BLOCK | yes |
| 20 | UAIUSDT | 2026-11-06 03:30 | 11-06 02:50 | 11-06 02:50 | 0.146083 | 0.146083 | 0.00 | 0.1422 | 0.1422 | -2.658 | -2.658 | BLOCK | BLOCK | BLOCK | yes |
| 21 | EVAAUSDT | 2026-11-06 03:05 | 11-06 02:55 | 11-06 02:55 | 1.36661 | 1.36661 | 0.00 | 1.3539 | 1.3539 | -0.930 | -0.930 | BLOCK | BLOCK | BLOCK | yes |
| 22 | CLANKERUSDT | 2026-11-05 23:55 | 11-05 23:50 | 11-05 23:50 | 19.1699 | 19.1699 | -0.00 | 19.0001 | 19 | -0.886 | -0.886 | BLOCK | BLOCK | BLOCK | yes |
| 23 | AINUSDT | 2026-11-05 20:20 | 11-05 20:15 | 11-05 20:15 | 0.279547 | 0.279547 | 0.00 | 0.279351 | 0.27935 | -0.070 | -0.070 | BLOCK | BLOCK | BLOCK | yes |
| 24 | AKEUSDT | 2026-11-05 18:55 | 11-05 18:40 | 11-05 18:40 | 0.00224601 | 0.00224601 | 0.00 | 0.00231599 | 0.002316 | 3.116 | 3.116 | KEEP | KEEP | KEEP | yes |
| 25 | FOGOUSDT | 2026-11-05 14:35 | 11-05 14:30 | 11-05 14:30 | 0.00280982 | 0.00280982 | 0.00 | 0.00280799 | 0.002808 | -0.065 | -0.065 | BLOCK | BLOCK | BLOCK | yes |
| 26 | AIOUSDT | 2026-11-05 10:20 | 11-05 10:15 | 11-05 10:15 | 0.0431329 | 0.0431329 | 0.00 | 0.0429798 | 0.04298 | -0.355 | -0.355 | BLOCK | BLOCK | BLOCK | yes |
| 27 | JCTUSDT | 2026-11-05 05:55 | 11-05 05:45 | 11-05 05:45 | 0.00673611 | 0.00673611 | 0.00 | 0.00670701 | 0.006707 | -0.432 | -0.432 | BLOCK | BLOCK | BLOCK | yes |
| 28 | UAIUSDT | 2026-11-05 02:55 | 11-05 02:45 | 11-05 02:45 | 0.09387 | 0.09387 | 0.00 | 0.0946998 | 0.0947 | 0.884 | 0.884 | KEEP | KEEP | KEEP | yes |
| 29 | SOPHUSDT | 2026-11-01 15:00 | 11-01 14:10 | 11-01 14:10 | 0.00146246 | 0.00146246 | 0.00 | 0.00142001 | 0.00142 | -2.903 | -2.903 | BLOCK | BLOCK | BLOCK | yes |
| 30 | BLURUSDT | 2026-10-30 18:55 | 10-30 18:40 | 10-30 18:40 | 0.00916034 | 0.00916034 | 0.00 | 0.00926999 | 0.00927 | 1.197 | 1.197 | KEEP | KEEP | KEEP | yes |
| 31 | BLURUSDT | 2026-10-30 10:15 | 10-30 10:10 | — | 0.00880568 | — | — | 0.00879996 | — | -0.065 | — | BLOCK | NO_EPISODE | BLOCK | no episode in rebuild |
| 32 | AINUSDT | 2026-10-30 06:15 | 10-30 06:10 | 10-30 06:10 | 0.179822 | 0.179822 | 0.00 | 0.17293 | 0.17293 | -3.833 | -3.833 | BLOCK | BLOCK | BLOCK | yes |
| 33 | INITUSDT | 2026-10-29 15:45 | 10-29 15:30 | 10-29 15:30 | 0.23043 | 0.23043 | 0.00 | 0.227169 | 0.22717 | -1.415 | -1.415 | BLOCK | BLOCK | BLOCK | yes |
| 34 | CAPUSDT | 2026-10-27 07:40 | 10-27 01:00 | 10-27 01:00 | 0.0966641 | 0.0966641 | 0.00 | 0.0955099 | 0.09551 | -1.194 | -1.194 | BLOCK | BLOCK | BLOCK | yes |
| 35 | ORDERUSDT | 2026-10-26 17:50 | 10-26 17:45 | 10-26 17:45 | 0.0413203 | 0.0413203 | 0.00 | 0.0403001 | 0.0403 | -2.469 | -2.469 | BLOCK | BLOCK | BLOCK | yes |
| 36 | BANKUSDT | 2026-10-26 13:45 | 10-26 12:45 | 10-26 12:45 | 0.0270731 | 0.0270731 | 0.00 | 0.0267 | 0.0267 | -1.378 | -1.378 | BLOCK | BLOCK | BLOCK | yes |
| 37 | CAPUSDT | 2026-10-26 09:30 | 10-26 09:30 | 10-26 09:30 | 0.08473 | 0.08473 | 0.00 | 0.0842903 | 0.08429 | -0.519 | -0.519 | BLOCK | BLOCK | BLOCK | yes |
| 38 | ERAUSDT | 2026-10-26 00:50 | 10-26 00:45 | 10-26 00:45 | 0.0842725 | 0.0842725 | 0.00 | 0.0838604 | 0.08386 | -0.489 | -0.489 | BLOCK | BLOCK | BLOCK | yes |
| 39 | IOSTUSDT | 2026-10-25 05:15 | 10-25 05:10 | 10-25 05:10 | 0.000831592 | 0.000831592 | 0.00 | 0.000831202 | 0.0008312 | -0.047 | -0.047 | BLOCK | BLOCK | BLOCK | yes |
| 40 | ARKUSDT | 2026-10-25 01:10 | 10-25 00:55 | 10-25 00:55 | 0.193048 | 0.193048 | 0.00 | 0.183 | 0.183 | -5.205 | -5.205 | BLOCK | BLOCK | BLOCK | yes |
| 41 | LAUSDT | 2026-10-24 10:25 | 10-24 10:10 | 10-24 10:10 | 0.0685566 | 0.0685566 | 0.00 | 0.0672602 | 0.06726 | -1.891 | -1.891 | BLOCK | BLOCK | BLOCK | yes |
| 42 | ERAUSDT | 2026-10-23 20:30 | 10-23 20:25 | 10-23 20:25 | 0.0773756 | 0.0773756 | 0.00 | 0.0769501 | 0.07695 | -0.550 | -0.550 | BLOCK | BLOCK | BLOCK | yes |
| 43 | EVAAUSDT | 2026-11-14 15:05 | 11-14 14:20 | 11-14 14:20 | 1.6938 | 1.6938 | 0.00 | 1.77301 | 1.773 | 4.676 | 4.676 | KEEP | KEEP | KEEP | yes |
| 44 | IDOLUSDT | 2026-11-13 16:30 | 11-13 16:20 | 11-13 15:25 | 0.0255263 | 0.0249369 | -2.31 | 0.0256189 | 0.025619 | 0.363 | 2.735 | KEEP | KEEP | KEEP | yes |
| 45 | AKEUSDT | 2026-11-13 07:15 | 11-13 06:45 | 11-13 06:45 | 0.00429189 | 0.00429189 | 0.00 | 0.00463602 | 0.004636 | 8.018 | 8.018 | KEEP | KEEP | KEEP | yes |
| 46 | INITUSDT | 2026-11-12 09:00 | 11-12 08:55 | 11-12 08:55 | 0.128683 | 0.128683 | 0.00 | 0.12872 | 0.12872 | 0.029 | 0.029 | KEEP | KEEP | KEEP | yes |
| 47 | ONEUSDT | 2026-11-12 08:20 | 11-12 08:05 | 11-12 08:05 | 0.00168112 | 0.00168112 | 0.00 | 0.0016999 | 0.0016999 | 1.117 | 1.117 | KEEP | KEEP | KEEP | yes |
| 48 | KGENUSDT | 2026-11-11 23:50 | 11-11 23:40 | 11-11 23:40 | 0.0866257 | 0.0866257 | 0.00 | 0.08685 | 0.08685 | 0.259 | 0.259 | KEEP | KEEP | KEEP | yes |
| 49 | INITUSDT | 2026-11-15 08:20 | 11-15 08:05 | 11-15 08:05 | 0.124527 | 0.124527 | 0.00 | 0.12721 | 0.12721 | 2.154 | 2.154 | KEEP | KEEP | KEEP | yes |

## 5. Result

| check | result |
|---|---|
| live vs today_keep | 49 / 49 agree (today_keep = the live stamp rule; same as `stack_keep`) |
| live vs rebuild, decision | 47 / 49 agree. 2 have no rebuild episode: JCT 11-10 21:35 and BLUR 10-30 10:15, both BLOCK live |
| live vs rebuild, value | 43 / 47 identical (≤ 0.001). 4 have another anchor, same sign |
| rebuild generated that red row itself (`sig_all`) | 40 / 49. The 9 missing are entry-existence gaps (11-18 report H3), not FAILED_SPIKE gaps. 6 are "new flags" 6.7 – 39.8 h after the spike (live re-flags after restarts / re-anchors); the other 3 have no episode or another anchor in the rebuild (JCT 11-10, BLUR 10-30, IDOL 11-13) |
| norm ratio live / rebuild | median 1.001, p10 – p90 0.994 – 1.031. One outlier: IDOL 1.56 |

**Disagreement causes.** Every value gap comes from one place: **the normal hour (norm)**, the volume baseline that decides which bar counts as the spike lead.

- **Why the norms differ.** The rebuild models the engine's per-pair norm refresh on a grid anchored at the epoch. Live, the grid is anchored at process start and restarts. The other causes on the list (bar choice, close vs price, ≤ vs <, rounding, data source) were each checked, and each matched.
- **BLUR 10-30.** With the live-implied norm, the rebuild finds live's exact episode: 10:10 anchor, −0.065 → BLOCK, agree.
- **IDOL 11-13.** With the live-implied norm, the rebuild finds live's exact 16:20 anchor and +0.363.
- **JCT 11-10.** It needs a norm about 10–20 % lower than either side's to see an episode. So the live 1,499-bar window or its live klines differed slightly (residual, unexplained).
- **CAP 10-27.** Switching to the live norm moves the rebuild's anchor **away** from live's: KEEP +4.07 vs live's BLOCK −1.19. So the implied norm (from a 1-dp stamp) is not a perfect reconstruction either.

**Full-history effect of the norm spread** (1,142 unfiltered rebuild HT entries, 416 kept by FAILED_SPIKE):

| | value |
|---|---|
| expected share of entries whose decision flips | **0.78 %** (132 entries can flip at ≥ 1 of 49 ratios; most at only 1 / 49, often the IDOL 1.56× outlier, so this is an upper-ish bound) |
| expected share that lose the episode | 1.6 % |
| KEEP → BLOCK (expected N) | 3.6, avg −0.79 % at +5 / −4 / 24 h |
| BLOCK → KEEP (expected N) | 5.3, avg +0.30 % |
| net effect on the kept set | ≈ +4.4 %-points over 416 kept ≈ **+0.01 % per kept trade**. Noise. |

## 6. Verdict

- **No definition mismatch.** The rebuild runs the engine's own FAILED_SPIKE function on the engine's own walk value, the same bar and the same rounding.
- On the 49 real HT fills, live = today_keep 49 / 49 and live = rebuild 47 / 49. The 2 misses are episodes the rebuild does not see, not a different filter verdict.
- The only modelling gap is the normal-hour refresh phase (already logged as 11-18 H3 / H8). Its full-history effect on the HT FAILED_SPIKE decision is ~0.8 % of entries and ~+0.01 %/trade. It cannot explain the master vs backtest gap.
- **The 11-18 conclusion stands:** the HT filter separates kept from blocked only in the live window it was fitted on.

## 7. What could not be checked

- **The live normal hour is not stamped.** It is implied from the 1-dp `entry_frenzy_vol_mult`, which is good to about 0.1 / vol_mult (a 1–5 % error at 2×–10×).
- **Live 5m klines as fetched at the time are not stored.** The rebuild data is the archived klines. That is the likely cause of JCT 11-10.
- **The full-history pricer uses 5m bars, not 1m.** The +5 / −4 / 24 h price is only used for the ~9 expected flips, so the effect on the verdict is negligible.
