# HT_THEORY_EXIT 2026-11-21 — pre-registration (frozen before any cell is simulated)

Research only. No engine / config / test change, nothing committed. Scripts `scripts/HT_THEORY_EXIT_2026-11-21_*.py`, data
`reports/HT_THEORY_EXIT_2026-11-21_data/`.

## Why
Recent FRENZY_WILLY_HT losers peaked +2.5 … +5 % and were then stopped at −4 % (ENSO, BLESS, EVAA, NAORIS, KAVA). Operator: "build all
studies, and bring a solid exit strategy per sleeve".

## Theory of the trade (HT)
HT buys the first red 5m bar of a live Frenzy burst once price is ≥ 1.4 % below the spike high H (DL 594). The bet: the burst is a pause,
not the top, so price goes back to H and maybe beyond. What follows from that:
1. **The setup has a natural target: H itself.** A dip-buy is paid when the retest happens; going *through* H is a second, harder bet (a
   breakout). With a median dip of ~2–3 %, today's +5 % net TP sits *above* H on most fills — it needs the breakout. A trade that peaks
   at about H and then turns (+2.5 … +4 %) is a "double top at H": the retest happened, the breakout failed. Prediction: a TP at H (or
   just under it) converts many give-backs into wins; it costs the runners that would have gone from H to +5.
2. **The thesis is wrong below the dip low.** If price breaks the low of the dip it bought (or loses the episode VWAP, the average
   buyer's cost), the burst is over. A flat −4 % stop is blind to that level. Prediction: a structure stop cuts losers earlier, unless the
   level is so close that noise hits it (then it only becomes a tighter stop — hence fixed-width controls).
3. **Continuation is fast or not at all.** A burst that stops making new highs is fading (WILLY_HT_THEORY 11-19: the median fill drifts to
   −6 % by 24 h). Prediction: after a +X % move, N minutes without a new high = take what is there.
4. **After the retest, the breakout leg is optional.** Once H has been touched the setup's job is done; the rest is a free option that
   should not be allowed to turn into a −4 % loss. Prediction: locking the stop at break-even after an H touch, or banking half at H, keeps
   the runners' upside and removes the double-top losses.
5. **Funding** — longs in squeezed coins usually *receive* funding (negative rates); a funding budget is a short-side problem. Reported
   descriptively only (no rule).

## Already tested — cited, not redone
STATE_EXIT 2026-11-21 (+ audit): in-trade state signs (RSI rollover, lower highs, climax, DI, ADX, VWAP *as a profit-taking exit under
gates G1–G3*, EMA gap, NEWHIGH = taker exit at the minute close when price ≥ H while ≥ +2 %), fixed TP 3 / 4 / 4.5 / 6, WILLY late TP — all
null on master. HT_WILLY 11-20: 40-cell TP × SL × hold grid (wider stops lose), entry vs random. WILLY_EXIT_V2 11-18: structural stops
(spike-bar low / base / VWAP), BE +0.75 / 1 / 1.5, trails, time grace — on the OLD HT exit (+2-era, 90 min) and on WILLY's V2b; none passed.
WILLY_HT_THEORY 11-19: entry × exit combos. **New here:** a *resting maker* TP anchored to H (fires intrabar, any P&L level, no slip) ·
nearer-of(H, +5) · structure stops measured from the *dip low* and a 5m-close VWAP stop under today's +5 / 24 h · stall exits · H-touch
locks · exact half-at-H scale-outs — all against today's REF (+5 / −4 / 24 h, dip ≥ 1.4 stack).

## Cohorts
- **REAL** — every `exports_archive/scalpars_orders_*.csv` (newest `…paper_2026-11-21_18-44-02.csv`), FRENZY_WILLY_HT, CLOSED, no probes,
  operator-closed out, dedup (opened_at[:19], pair, direction) newest export wins (`HT_THEORY_EXIT_2026-11-21_cohort.py`).
  - **MASTER** = opened < 2026-11-17T17:00 (49 fills). "As taken" = all 49; **"today's filters" = dip ≥ 1.4 %** (dip = stamped
    `entry_frenzy_willy_dip_pct`, else (H − E) / H) = 43 fills. **The deciding cohort is MASTER · today's filters.**
  - **BATCH** = opened ≥ 2026-11-17T17:00 (13 fills, read separately; in-sample → 30–50 % haircut on any number derived from it).
  - Every real fill is RE-PRICED under each exit from its real fill price, the walk starting the minute after the fill.
- **BACKTEST** = `reports/HT_WILLY_2026-11-20_cohort_HT.csv` (300 fills, DL 594 stack), entry bar = `ts`, E = the row's E (entry slip
  already inside). DESIGN < 2025-07-01 ≤ HOLDOUT.
- **Second section — plain FRENZY_WILLY** (85 real fills ≥ 15 → included): REAL same rules (all master; "today's filters" =
  `reports/MASTER_POOL_stacked.csv` stack_keep), BACKTEST `reports/HT_WILLY_2026-11-20_cohort_WILLY.csv` (545).

## Engine (both sleeves, 1m bars)
- 1m bars: `reports/STATE_EXIT_2026-11-21_data/k1m` (read-only, has volume) → `fast_tp_data.get1m` (read-only) → own vision fetch
  into this study's data folder (0 Binance weight; fapi only for days not yet on vision, weight header read, stop on 418 / 429).
- **HT REF** = today's exit: resting maker TP at net +5 (price rounded up to the tick, fills when the high trades ≥ 1 tick through), stop
  at net −4 (fill at min(stop, bar open) − 0.10 % slip, taker), stop checked first in a bar, 24 h time exit at the open of the first bar
  ≥ entry + 24 h (taker + slip). Costs: entry taker 0.05, maker TP 0.02, stop / time / rule exits taker 0.05 + 0.10 slip. Real funding
  at settlements (STATE_EXIT `fund/` read-only, else own fetch); a long pays a positive rate.
- **WILLY REF** = V2b as `fast_tp_sim.willy_ref` with today's config (TP = max(1.25, min(8, net at H × 0.99)) maker, stop −4, late TP
  +0.25 from 30 min, 90 min cap, ATR runner where the row carries one).
- **Parity gate (stop if failed):** REF exit kind = booked kind on the real HT fills booked under +5 / −4 (`frenzy_willy_tp_target_pct` = 5)
  and = `ref_kind` on the HT backtest cohort, on ≥ 99 % of fills. WILLY: REF kind = `ref_kind` on ≥ 99 % of the backtest; real WILLY
  booked kinds reported (eras differ; not a gate).
- Anchors, all known at entry: **H** (stamp / cohort); **dip low L** = lowest 1m low from the minute of H (argmax high in [spike, entry))
  to the last 1m bar closed before the fill, capped at E; **episode VWAP** = Σ typical × quote volume ÷ Σ quote volume from the spike bar,
  updated each minute (needs volume; unavailable → the fill keeps REF in that cell, counted); **base** = 1m close 30 min before the spike bar.

## HT cells (23) — frozen
| family | cell | rule (everything not named = REF) |
|---|---|---|
| A target (replace +5) | A_H0 · A_Hm05 · A_Hp05 · A_H75 | maker TP at H · H × 0.995 · H × 1.005 · E + 0.75 × (H − E); net floor +1.0, cap +10 |
| N nearer-of | N_H0 · N_Hm05 | maker TP at the nearer of +5 net and the A level |
| B stall (overlay) | B_X{2,3}_N{30,60,120} | once the in-trade high has reached +X % gross, if N minutes pass without a new 1m high and the 1m close is net > 0 (exit now) → market exit at the next minute open |
| C structure stop (replace −4) | C_DIPLOW · C_VWAP5 · C_BASE | stop at L × 0.997 (distance clamped 1.5–8 % below E) · exit at the next open when a CLOSED 5m bar closes < VWAP × 0.995, hard stop −8 net · stop at base × 0.995 (clamped 1.5–10 %) |
| C controls | S3 · S5 · S6 | fixed net stop −3 / −5 / −6 |
| D H-touch lock (overlay) | D_HLOCK0 · D_HLOCK1 | after the first 1m bar whose high ≥ H, an extra stop at net ≈ 0 (E × 1.0025) / net +1 from the next bar |
| E scale-out (exact, two half legs) | E_H0_5 · E_Hm05_5 · E_H0_LOCK | ½ A_H0 + ½ REF · ½ A_Hm05 + ½ REF · ½ A_H0 + ½ D_HLOCK0 |
**Fixed-TP comparison set (not candidates):** TP net 2 · 2.5 · 3 · 3.5 · 4 · 4.5 · 6 with −4 / 24 h. A candidate must beat the BEST of these on master.

## WILLY cells (7) — frozen (structure stops / BE / trails were tested on V2b 11-18 → not repeated)
WA_H0 (target from H, k = 0) · WA_Hp05 (k = −0.5) · WE_HALF (½ V2b + ½ WA_Hp10, k = −1) · WB_X{1,2}_N{15,30} (stall overlay, as B).
Fixed comparison: flat TP 1.25 / 2 / 3 with V2b's late TP / stop / cap.

## Reads and the decision rule (pre-committed)
Per cell, per cohort: N · WR · avg net % · Δ vs REF (%/fill) · saved $ (Σ positive Δ) / cut $ (Σ negative Δ) at **$3,050 per fill**
(0.61 × $5,000) · day-block bootstrap (3,000 draws, resample UTC entry days) P(Δ > 0) and 95 % CI · one-size DCR before → after
(r_d = Σ net % × 0.61 / 100 over fills exiting on day d, every calendar day of the block, compounded) · avg loss · worst day.
**CANDIDATE iff on MASTER · today's filters:** ① Δ ≥ 0 · ② saved $ ≥ cut $ · ③ P(Δ > 0) ≥ 0.80 · ④ Δ > the best fixed-TP alternative's Δ.
Then the read-outs (caveats, not vetoes — master decides): MASTER as taken same sign · BATCH (haircut 30–50 %) · backtest DESIGN and
HOLDOUT with CI · 0.20 % slip stress (Δ still ≥ 0 on master) · concentration (no pair / day ≥ 50 % of the positive Δ) · leave-one-pair-out
(min Δ) · risk (avg loss, worst day). 23 HT cells (+ 7 fixed-TP comparisons) and 7 WILLY cells are tested: P ≥ 0.80 on one cell is weak on its own — a candidate whose
backtest HOLDOUT Δ CI is fully below 0 is reported as "master-only, contradicted" and at most an OBSERVE line. No ship from this study;
a candidate becomes a proposal to the operator with a priced revert gate.
