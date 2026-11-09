# FRENZY_LONG review — 2026-11-09 (research only, nothing changed)

Operator: "LONG_ADX35 is not enough to make the sleeve a winner. Re-check all variables, and theory, to understand winners vs losers."
Data: master `MASTER_POOL_stacked.csv` stack_keep, probes out, −3-era stops re-priced at today's −4 (`master_exits.reprice_long_stop4`) = **29 · 86 % · +0.74 %**; B25 = export `scalpars_orders_paper_2026-11-09_12-45-05.csv`, opened ≥ 10-23 18:30 = **17 · 47 % · −1.41 % · Σ −24.0** (no MANUAL / operator-closed FRENZY_LONG fills in either). Exit today: TP +1.5 (resting maker) / stop −4 / 12 h → breakeven WR **72.7 %**. One size X = 0.05 (notional ÷ equity), so the whole sleeve moves the account by < 0.1 %/day either way.
Scratch code + per-fill tables: `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/flong/`.

## 0 · Theory first (written before the confirmatory read)

**The bet.** A coin spiked (≥ 5 %/h on ≥ 20× volume). ≥ 2 h later it has closed 12 × 5m bars at or above the spike-anchored VWAP (the frenzy buyers' average cost) on state volume ≥ 100×, and we buy a RED 5m bar: a dip into a level the crowd defends. Continuation of +1.5 % before −4 % is the payoff; it needs 3 wins per loss.
**It should fail when the defence is absent or spent.** Pre-declared tests (cut = the frozen MASTER median, so B25 is out-of-sample for the cut; one test each; "worse" = zone below rest):

| # | Hypothesis (predicted: zone worse) | Variable · zone | Master zone \| rest (N · WORKED-rate · avg %) | B25 zone \| rest | Same way both? | P(zone ≥ rest), day-bootstrap pooled |
|---|---|---|---|---|---|---|
| H1 | stretched far above support | vs_vwap > 3.42 | 14·79%·+0.71 \| 15·87%·+0.77 | 12·58%·−1.25 \| 5·60%·−1.80 | split | 0.34 |
| H2 | run exhausted | run_pct > 37.4 | 13·85%·+0.65 \| 16·81%·+0.81 | 8·75%·−1.25 \| 9·44%·−1.56 | opposite | 0.50 |
| H3 | late in the episode | frenzy_hours > 13.8 | 14·79%·+0.32 \| 15·87%·+1.13 | 8·62%·−1.94 \| 9·56%·−0.94 | avg yes, WORKED no | 0.07 |
| H4a | volume fading | vol_trend ≤ 1.63 | 14·93%·+1.11 \| 14·79%·+0.71 | 5·40%·−1.80 \| 12·67%·−1.25 | split | 0.86 |
| H4b | volume low | vol_mult ≤ 141 | 15·80%·+0.77 \| 14·86%·+0.71 | 13·54%·−1.89 \| 4·75%·+0.13 | split (master flat) | 0.08 |
| H5 | BTC falling | btc_1h_slope ≤ 0 | 11·73%·+0.50 \| 18·89%·+0.89 | 9·67%·−0.94 \| 8·50%·−1.94 | split | 0.45 |
| H6 | weak breadth | bull_pct ≤ 41.3 | 15·87%·+0.77 \| 14·79%·+0.71 | 10·60%·−1.25 \| 7·57%·−1.64 | opposite | 0.51 |
| H7 | trend late / crowded | adx > 29.5 | 14·64%·−0.07 \| 15·100%·+1.50 | 12·67%·−0.79 \| 5·40%·−2.91 | split | 0.10 |
| H8 | red bar = real selling | bar_ret ≤ −0.54 | 15·87%·+0.77 \| 14·79%·+0.71 | 13·62%·−1.46 \| 4·50%·−1.25 | opposite | 0.29 |
| H9a | repeat on the pair (48 h) | repeat | 11·82%·+0.50 \| 18·83%·+0.89 | 5·60%·−2.91 \| 12·58%·−0.79 | avg yes, WORKED no | 0.19 |
| H9b | right after a loss on the pair | after_loss | 1 \| 28 | 3·67%·−2.18 \| 14·57%·−1.25 | too few | 0.28 |
| H10 | thin coin | vol24h ≤ $249M | 15·80%·+0.77 \| 14·86%·+0.71 | 14·57%·−1.25 \| 3·67%·−2.17 | opposite | 0.30 |
| H11 | high ATR → stop reachable | atr > 1.94 | 14·86%·+0.71 \| 15·80%·+0.77 | 10·80%·+0.41 \| 7·29%·−4.01 | **opposite** (low ATR worse in B25) | 0.95 |
| H12 | Asian hours | 00–08 UTC | 12·75%·+0.58 \| 17·88%·+0.85 | 3·67%·−2.17 \| 14·57%·−1.25 | avg yes, WORKED no | 0.57 |
| H13 | pair overbought | rsi > 58.9 | 14·64%·−0.07 \| 15·100%·+1.50 | 9·78%·−0.33 \| 8·38%·−2.63 | split | 0.37 |
| H14 | VWAP not really held | above_share ≤ 63.8 | 13·92%·+1.08 \| 12·75%·+1.04 | 3·0%·−4.01 \| 14·71%·−0.86 | split | 0.65 |
| H15 | BTC overheated | btc_rsi > 51.8 | 14·71%·+0.71 \| 15·93%·+0.77 | 8·62%·−1.94 \| 9·56%·−0.95 | avg yes, WORKED no | 0.30 |

WORKED = +1 % (gross) before the −4 stop line on the 1m path (exit-independent). H2 / H7 / H11 / H15 had been glimpsed in a load-test sweep printout before writing — treat them as contaminated.
**Result: 0 of 17 theory hypotheses point the same way in master AND B25 on both WORKED-rate and booked avg.** On booked avg alone 4 of 17 do (H3, H9a, H12, H15) — what chance gives (~4 of 17). The theory as written does not separate winners from losers on 46 fills.

## 1 · Watch lines (before → after = the line switched on; ONE size; Σ in % points)

| Line (registered) | B25 zone | B25 kept | Master zone | Master kept | DCR %/day B25 | DCR master | B25 ΔΣ (leave-one-out) |
|---|---|---|---|---|---|---|---|
| LONG_ADX35 (DL 418) adx > 35 | 8·38%·−1.94 | 9·56%·−0.95 | 11·64%·−0.50 | 18·100%·+1.50 | −0.067 → −0.024 | +0.062 → +0.077 | +15.5 (+11.5…+17.0) |
| FRENZY_STRETCHED (DL 261) vwap ≥ 13 | 4·50%·−1.25 | 13·46%·−1.46 | 3·100%·+1.50 | 26·85%·+0.65 | → −0.053 | → +0.055 | +5.0 (+1.0…+6.5) |
| REENTRY_AFTER_WIN (DL 261) | 5·40%·−1.80 | 12·50%·−1.25 | 13·85%·+0.65 | 16·88%·+0.81 | → −0.042 | → +0.027 | +9.0 (+5.0…+10.5) |
| LONG_NONTOP50 (DL 288) | 11·45%·−1.50 | 6·50%·−1.26 | 14·93%·+1.11 | 15·80%·+0.40 | → −0.021 | → +0.004 | +16.5 (+12.5…+18.0) |
| ADX35 + STRETCHED | 10·40%·−1.80 | 7·57%·−0.86 | 14·71%·−0.07 | 15·100%·+1.50 | → −0.017 | → +0.070 | +18.0 |
| ADX35 + REENTRY | 11·36%·−2.00 | 6·67%·−0.33 | 18·78%·+0.28 | 11·100%·+1.50 | → −0.006 | → +0.038 | +22.0 |
| ADX35 + NONTOP50 | 12·42%·−1.71 | 5·60%·−0.70 | 22·82%·+0.50 | 7·100%·+1.50 | → −0.010 | → +0.017 | +20.5 |
| STRETCHED + REENTRY | 7·29%·−2.43 | 10·60%·−0.70 | 14·86%·+0.71 | 15·87%·+0.77 | → −0.019 | → +0.024 | +17.0 |
| STRETCHED + NONTOP50 | 13·38%·−1.89 | 4·75%·+0.12 | 17·94%·+1.18 | 12·75%·+0.12 | → +0.001 | → −0.003 | +24.5 |
| REENTRY + NONTOP50 | 11·45%·−1.50 | 6·50%·−1.26 | 20·90%·+0.95 | 9·78%·+0.28 | → −0.021 | → −0.001 | +16.5 |

Format N · WR · avg %. Matches best case 12-45 (LONG_ADX35 −0.065 → −0.023 / +0.061 → +0.075; tiny span rounding).
- **LONG_ADX35 is the only single line that helps both blocks** (the ADX35 + STRETCHED pair also helps both, but is worse on master than ADX35 alone: +0.070 vs +0.077 — STRETCHED's master zone is 3 winners), and its B25 zone is genuinely out-of-sample** (registered 10-30, B25 had 0 zone fills then; all 8 came after) — direction held: 8·38%·−1.94 vs 9·56%·−0.95, single-test day-shuffle p 0.017. But its kept set is still **56 % vs 72.7 % breakeven, −0.95 %/fill** in B25 — the operator is right: it cannot make the sleeve a winner. Gate state: zone 8 on 5 days of the needed 15 on 8.
- Every other single line and every pair without ADX35 HURTS master (NONTOP50 kills the master winners: zone 93 % WR). Exit lines (LONG_STOP4, FRENZY_V2B_SHADOW TP2/3/4, FRENZY_WILLY_TP grid) are read in §2. LONG blocks LITE on the same coin: only 1 LITE pair-held block in B25 (BMT 11-03, an ADX35 zone fill), so any LONG filter frees ≈ nothing for LITE this batch.

## 2 · Decomposition from the price path — entry or exit?

| Block | Losers | NEVER_WORKED | STOPPED_RECOVERED | GAVE_BACK | Winners CAPTURED / LEFT_ON_TABLE | Reach median W / L |
|---|---|---|---|---|---|---|
| B25 (17) | 9 (all −4 stops) | **7** (BIRB, BMT×2, HOT, HAEDAL, POWR, + BMT 11-03) | 1 (EVAA, +11.7 after) | 1 (CYBER, reached +1.14) | 3 / 5 | +5.9 / +0.68 |
| Master (29, −4) | 4 | 2 (ENJ, RLC) | 1 (CFX) | 1 (SAND, +1.11) | 25 winners | +5.3 / +0.51 |

**B25 is an ENTRY problem: 8 of 9 losers never went +1 % our way** (7 NEVER_WORKED + EVAA, stopped then recovered without reaching +1 %) (median reach +0.68 %). Exit side = 1 fill. Study order per DL 406: entry first.
Exit counterfactuals anyway (1m replica of +1.5 / −4 reproduces 42 / 46 booked results; the 4 misses are master −3-era fills that the 10-19 TICK study saved at −4, left out below):

| Exit | Master (25) | Δ on live-stopped (saved) | Δ on NOT stopped | B25 (17) | Δ live-stopped (saved) | Δ not stopped |
|---|---|---|---|---|---|---|
| today +1.5 / −4 / 12 h | 84%·+0.62 | – | – | 47%·−1.42 | – | – |
| +1.5 / −3 | 80%·+0.60 | +4.0 (0) | −4.5 | 41%·−1.15 | +9.0 (0) | −4.5 |
| +1.5 / −5 | 84%·+0.46 | −4.0 (0) | 0 | 47%·−1.95 | −9.0 (0) | 0 |
| +1 / −4 | 88%·+0.40 | +5.0 (1) | −10.5 | 59%·−1.06 | +10.0 (2) | −4.0 |
| +2 / −4 | 76%·+0.56 | 0 | −1.5 | 47%·−1.18 | 0 | +4.0 |
| +3 / −4 | 56%·−0.08 | 0 | −17.6 | 41%·−1.12 | 0 | +5.0 |
| +1.5 / −4 / **60 min cap** | 84%·+0.79 | +4.4 (0) | **0** | 47%·−1.02 | +6.7 (0) | **0** |

- **No TP / stop / time variant makes B25 positive** (best −0.89 %/fill at +1 / −3). Stop width is era-dependent: B25 says −3 (+4.5 pts; −4 saved 1 winner, deepened 9), master's tick study said −4 (5 of 9 saved) — owned by the DL 317 revert gate / LONG_STOP4 line, not re-decided here.
- **60-min cap** is the one exit that helps both with Δ = 0 on fills live did not stop (no winner in either block took > 60 min to TP). Caveats: the 4 tick-saved master fills (BAT touched −3 at 35 min) are exactly the ones a cap could hurt and are outside the replica; best case ①c: "too few" for the luck test. → a study / observe line, not a ship.

## 3 · Winners vs losers — every variable (exploratory)

**Coverage.** 147 entry_* columns; 83 numeric/flag tested at ≥ 80 % coverage in my master-vs-B25 screen, 68 + 17 in `sweep_separators.py`. **Blind spots (< 80 %, NOT tested):** book drift / spread / fill-vs-book / order lag (21 % master), gvol (24–38 %), funding, gap_5_20 / gap_5_8 signed, gap_expand, pair_1d_ndi, pair_age, btc_4h_ema50_200, alt_corr / beta, pair_alpha, mkt_up4h (B25-only stamps), catchup bars / move; pre-entry 1m features (return 60 / 240 min, distance from 24 h high, 60-min range — the 1m cache lacks pre-entry history for many fills); never stamped on this sleeve: 32 (bear/br/flatbase/surge/willy families, pair_rank). Market-wide variables: 13 master days + 9 B25 days only.

**Sweeps** (`sweep_separators.py FRENZY_LONG`, path → reach → booked, 1D × 3 granularities, every 2D pair, 3D from the top 2D cells; day-block shuffled null, `scout_sweep_all`):

| Label | 1D survivors vs null | best 1D vs luck's best (p) | 2D survivors vs null | best 2D (p) | 3D |
|---|---|---|---|---|---|
| path (WORKED) | 53 vs 60 | 0.24 vs 0.43 (0.94) | 543 vs 758 | 0.46 vs 0.55 (0.55) | 0 consistent |
| reach | 54 vs 74 | 7.6 vs 12.7 (0.71) | 1461 vs 2096 | 12.4 vs 15.3 (0.48) | 0 |
| booked | 70 vs 70 | 1.40 vs 2.15 (0.52) | 1744 vs 1933 | 2.66 vs 2.86 (0.17) | 0 |

**Not above luck at any level, on any label.** (The sweep's booked label reads master at the traded −3; my own screen below uses the −4 re-pricing.)
**Master-vs-B25 screen** (era = master vs B25, cut from master; 131 splits): 22 splits point the same way in both with |Δavg| ≥ 0.5 vs a shuffled null of 30 (95 % 53); best 1.38 vs luck 2.08 (p 0.69). **Fewer consistent splits than chance.**

Strongest individual reads (each single-test; family-wise none survives):

| Candidate | Master zone \| rest | B25 zone \| rest | DCR B25 / master | ΔΣ B25 (LOO) / master (LOO) | single p | Backtest yr6W (+3/−3, 623) |
|---|---|---|---|---|---|---|
| LONG_ADX35 (ref) | 11·64%·−0.50 \| 18·100%·+1.50 | 8·38%·−1.94 \| 9·56%·−0.95 | −0.067→−0.024 / +0.062→+0.077 | +15.5 (+11.5…+17.0) / +5.5 (+1.5…+7.0) | 0.017 | no difference (−0.17 vs −0.18) |
| SLIP_UP slippage > 0 | 10·70%·−0.15 \| 19·95%·+1.21 | 9·33%·−2.17 \| 8·62%·−0.56 | −0.067→−0.012 / +0.062→+0.068 | +19.6 (+15.5…+21.1) / +1.5 (−2.5…+3.0) | 0.026 | weak (−0.21 vs −0.15; slippage is modelled there) |
| LOW_ATR atr ≤ 1.94 | 15·87%·+0.77 \| 14·86%·+0.71 | **7·0%·−4.01** \| 10·80%·+0.41 | −0.067→+0.011 / +0.062→+0.050 | +28.1 (+24.1…+28.1) / −11.5 | 0.003 | worse 2 of 3 regimes (−0.25 vs −0.12; 2025 opposite) |
| H4b vol_mult ≤ 141 | 15·80%·+0.77 \| 14·86%·+0.71 | 13·38%·−1.89 \| 4·75%·+0.13 | → +0.001 / → +0.048 | +24.6 / −11.5 | 0.13 | **worse in all 3 regimes** (−0.25 vs −0.03) |
| H3 hours > 13.8 | 14·79%·+0.32 \| 15·93%·+1.13 | 8·38%·−1.94 \| 9·56%·−0.94 | → −0.024 / → +0.064 | +15.5 / −4.5 | 0.08 | mixed |
| BTC_HOT btc_rsi > 55.4 | 9·89%·+0.89 \| 20·85%·+0.67 | 3·0%·−4.01 \| 14·57%·−0.86 | → −0.033 / → +0.047 | +12.0 / −8.0 | 0.15 | worse overall, 26-H2 only |

Joint entry × exit (1m replica): every candidate's zone stays worse than its kept set under +1.5/−4, +1/−3, +1/−4, +2/−4 and +3/−4 in both blocks — no flip between reads. But **no kept set is above breakeven in B25 under any exit** except LOW_ATR / VOL_LOW, and those two cost master −11.5 pts (their master zone is the master's winners).
Per-pair concentration (B25): BMT 3 fills = −12.1 (33 % of the gross loss), below the 60 % pair-blacklist rule.
Post-hoc theory for the two B25-only reads: a **"dying frenzy"** — the second fill on a pair after a win came with lower ATR and volume (POWR ATR 2.49 → 1.60, CYBER 2.61 → 1.90, both losers; EVAA the reverse, loser → winner). Low absolute ATR / vol_mult in master was not bad because master's coins were bigger; the relative fade inside an episode (ATR now ÷ ATR at the spike) is **not stamped** — a blind spot, not a finding.

## 4 · Master vs B25 consistency and tape

Uniform degradation: both sides of 13 of the 17 pre-declared splits fell from positive (master) to negative (B25); of the other 4, two (adx > 29.5, rsi > 58.9) were already negative in master on the zone side, and two leave one positive B25 cell. The only B25 cells that stayed positive are atr > 1.94 (10·80%·+0.41) and vol_mult > 141 (4·75%·+0.13) — master shows no edge for them.
Tape (medians at entry, master → B25): BTC below its 30-day high 5.2 % → **10.5 %** · BTC 72 h return −0.05 → **−1.02 %** · BTC 1h slope +0.04 → **−0.06** · alt breadth (bull %) 41 → **33** · global volume ratio 0.88 → 0.79 · frenzy volume ×141 → **×112** · pair 24 h volume $249M → **$122M** · ATR 1.94 → 2.11 · ADX 29 → 35. B25 = a weaker market with smaller, thinner frenzies.

## 5 · Sleeve-kill checklist (no kill proposed)

① Every pair-level entry dimension tested — yes: 83-column master-vs-B25 screen + sweep 68 numeric + 17 level × 3 granularities, path / reach / booked; blind spots listed in §3. Nothing above luck.
② Macro / regime at SIGN first, then buckets — yes: BTC 1h slope, ema20 slope, dist-from-EMA13, 1d return, trend gap, r72 at sign (sweep) then median / tercile; BTC-hot family points the same way in both blocks (B25 3 fills on 3 days) but not above luck. Only 22 market-days in total — a regime switch cannot be judged in day units yet.
③ Uniform degradation — **YES**: 13 of 17 pre-declared splits (both sides) degraded together master → B25 ⇒ the cause is most likely an unmeasured regime variable, not one entry dimension. Per the rule: keep hunting, do not kill.
④ Tape context — done (§4): B25 was a weaker, thinner market than the Oct 3–19 master window.
Also: LONG's "blocks bad LITE" role (DL 280 / 291) barely fired in B25 (1 block).

## 6 · Recommendation

**Nothing found that ships.** No theory hypothesis, no registered line and no exploratory separator passes master-and-B25 consistency AND the null; B25's losses are entry-side (8 of 9 losers never worked) in a weaker tape. The sleeve's stake is tiny (X 0.05: ±0.07 %/day). No size step (DL 503 — below breakeven); keep observation size.

1. **Keep LONG_ADX35 observing on its frozen gate** (zone 8 on 5 days of 15 on 8; now OOS-confirmed in direction). It is necessary-not-sufficient.
2. **Proposed observe lines (NOT registered — operator call; frozen definitions below)** — cohort from the registering commit + 10 min, booked % at +1.5 / −4 / 12 h (exit change → STALE):
   - **LONG_SLIP_UP**: zone = `entry_slippage_pct` > 0 (frozen). Theory: filling above the red bar's close means the dip was already being bought — we pay the bounce. Only candidate pointing the same way in master (−0.15 vs +1.21) and B25 (−2.17 vs −0.56); master LOO fragile; family-wise not above luck.
   - **LONG_LOW_ATR**: zone = `entry_atr_pct` ≤ 1.94 (frozen master median). B25-only (master flat), backtest worse in 2 of 3 regimes; post-hoc.
   - Gate for both (locked expectancy bar): at zone N ≥ 15 on ≥ 8 windows, once — zone avg ≥ rest → RETIRE; zone WR < 72.7 % ∧ window-bootstrap P(mean < 0) ≥ 0.95 ∧ no window / pair ≥ 50 % of the zone loss → FILTER CANDIDATE (30–50 % haircut, tight revert gate, operator decides); else keep observing. Evidence needed: the post-registration fills only — the 46 fills here never count.
3. **Exit to study: 60-min time cap** for FRENZY_LONG — helps both blocks on the live-stopped cohort with Δ = 0 elsewhere (master +0.62 → +0.79, B25 −1.42 → −1.02). Before any read: tick re-walk of the 4 master −3-era fills the cap could hurt; then the live-stopped two-sided read per DL 110 at ≥ 15 stopped fills on ≥ 8 days, joint-checked on the fills LONG_ADX35 keeps.
4. **Stamp idea (D11 / D13, ask before building):** `entry_frenzy_atr_vs_spike` (ATR now ÷ ATR at the spike) and the episode's fill count — the "dying frenzy" theory cannot be tested on today's stamps.
   ⚠ *Superseded by DL 512 (reports/DYING_FRENZY_REBUILD_TEST_2026-11-09.md): both measures rebuild from `entry_frenzy_spike_at` + klines + order history — no stamps needed; the theory was not confirmed beyond B25.*
5. Whether to keep running the sleeve in this tape is an operator call below every locked bar: the kill checklist says keep hunting (③ uniform degradation), and the sleeve's cost at today's size is small.
