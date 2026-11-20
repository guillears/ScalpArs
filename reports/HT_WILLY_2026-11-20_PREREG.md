# HT_WILLY 2026-11-20 — PRE-REGISTRATION (frozen before any grid / random-entry result was computed)

Research only. No engine / config / test change. Nothing committed. Written 2026-11-20 before running `scripts/ht_willy_*.py`.
Already seen before this file (stated honestly): FAST_TP_2026-11-20 (HT REF at TP 5 / SL 4 / 24 h = −0.33 %/fill full span, −0.56 HOLDOUT;
yesterday's ±12 h random control WILLY +0.20 vs −0.08, HT +0.58 vs −0.33). Nothing else about the cells / controls below.

## Data
- Backtest fills: `reports/BASELINE_TRADES_WILLY_HT_2026-11-18.csv` (2025-01 → 2026-11-17). WILLY 545 (today's DL 527 filters already applied),
  HT 417 (FAILED_SPIKE applied; built under HT's OLD +2 exit).
- 1m bars: `scripts/fast_tp_data.py` caches + data.binance.vision zips (0 weight); fapi only for 2026-11-20 (≤ 1,200 weight/min, header read, stop 418/429).
- **Split = the sleeve's own (history starts 2025-01): DESIGN = 2025 (25-H1, 25-H2) · HOLDOUT = 2026-01-01 → (26-H1, 26-H2), read once.**
  The 2023-07 → 2025-06 split is impossible here (no WILLY/HT signals before 2025).
- Real fills: master `MASTER_POOL_stacked.csv` stack_keep, full size (is_probe False, no *_PROBE): WILLY B19–B27; HT master = all probes → none.
  Current batch `~/Downloads/scalpars_orders_paper_2026-11-20_13-22-26.csv`: HT 10 fills (EVAA 11-18 12:30 predates the DL 587 exit → priced, flagged), WILLY 0.
- One size: X = 0.61 of a $10k account → $6,100 per fill (same as FAST_TP). DCR-cal via `scripts/best_case/one_size.py` (`dcr`).

## Today's stack
- HT cohort = baseline HT rows with dip = (H − live) / H × 100 ≥ 1.4 (DL 594; H = spike high `peak`, live = decision price) → then de-overlapped
  per pair: an entry is kept only if the previous KEPT HT trade in that pair has exited (REF exit). = trading possibilities.
- WILLY cohort = the 545 rows (V2b exit, 90 min: overlap per pair is checked the same way).
- REF exits: HT = TP +5 net resting maker (1 tick through) / SL −4 net (taker + 0.10 slip) / 24 h at the bar open, no late TP, no high target, funding.
  WILLY = V2b (TP 1.25 · high target H × 0.99 capped 8 · late +0.25 @ 30 · runner 2 × ATR on EMA20 < EMA200 coins · SL −4 · 90 min), `fast_tp_sim.W_CUR`.
- Replica parity: HT REF vs current-batch real HT fills (exit kind, exit minute ±1, net within 0.15).

## PART 1 — HT exit grid (40 + 1 cells)
- Cells T{3,4,5,6,8} × S{3,4,5,6} × H{12 h, 24 h} (T = net TP, maker resting; S = net stop) + OLD2 (the old +2 exit: WILLY logic, target ≥ 2, high target,
  late TP, 90 min). REF = T5_S4_H24 (excluded from the pick).
- Entries fixed = the REF-de-overlapped cohort (paired Δ = cell − REF per fill). Sensitivity (not for the verdict): each cell de-overlapped under its own exit.
- Pick: the cell with the highest DESIGN mean Δ, among cells with ≥ 30 changed DESIGN fills; none > 0 → NO PICK.
- Null: family-wise day sign-flip (flip the sign of each DESIGN day's summed Δ jointly across all 40 cells), max t over cells, 2,000 draws → pick's t > 95th.
- HOLDOUT read once for the pick. CI = day-block bootstrap (resample entry days, 5,000), 95 %.
- PASS = HOLDOUT Δ > 0 ∧ CI low > 0 ∧ DESIGN Δ > 0 ∧ beats null ∧ ≥ 3 of 4 half-years Δ > 0 ∧ top pair < 50 % of +Δ ∧ leave-one-pair-out min Δ > 0
  ∧ current-batch real fills not contradicting (contradict = Σ Δ < 0 on ≥ 3 changed fills). Then 30–50 % haircut.
  Below the bar but DESIGN ∧ HOLDOUT Δ > 0 → OBSERVE-line candidate (labelled).
- Live-stopped rule: stop-only cells T5_S{3,5,6}_H24 split into saved / deeper (REF-stopped) and new stops (REF not stopped); wider stops must show Δ = 0 on REF-not-stopped fills (asserted).
- Path decomposition of the REF cohort (`scripts/path_labels.py` definitions if importable, else the same rule set re-implemented): NEVER_WORKED · STOPPED_RECOVERED ·
  GAVE_BACK · DRIFT_LOSS · LEFT_ON_TABLE · CAPTURED, with $ at one size.

## PART 2 — real entry vs random entry (WILLY and HT separately, never pooled)
- K = 50 random entries per real fill per control, same coin, long, same exit (same H, same runner ATR), 0.10 slip, entry = the previous 1m close, funding same method:
  - **C_LOC**: real entry minute + U{−30 … +30} min (≠ 0), never before the spike time.
  - **C_EP**: uniform minute in [spike_ms, spike_ms + 6 h) — any time inside the episode after the spike.
  - **C_EPD (HT only)**: C_EP restricted to minutes whose previous close ≤ H × (1 − 1.4 %) (today's dip rule).
  - **C_OLD12 (diagnostic only)**: yesterday's ±12 h 5m grid, reported split into pre-spike vs post-spike draws.
- Statistic per fill d = real net − mean of its 50 random nets; mean d, day-block bootstrap CI, per half-year, DESIGN / HOLDOUT; mean rank percentile of the real fill.
- Verdict per sleeve × control: REAL BEATS RANDOM iff full-span mean d > 0 with CI low > 0 ∧ HOLDOUT d > 0. RANDOM BEATS REAL iff mean d < 0 with CI high < 0.
  Else NO DIFFERENCE. If REAL BEATS RANDOM on C_EP and C_LOC → stop (no timing rules).
- Real fills: master WILLY (24) and current-batch HT (10) get the same controls.
- If not: EXPLORATORY timing map on DESIGN C_EP draws only (minutes since spike, % below the running episode high, pullback from the running high, ret since spike);
  then ≤ 3 timing rules frozen in an ADDENDUM below BEFORE any HOLDOUT read of them; tested with the Part-1 bar (DESIGN Δ > 0, HOLDOUT Δ > 0 with CI low > 0,
  family-wise null over the ≤ 3 rules, ≥ 3/4 half-years, top pair < 50 %).

## PART 3 — joint
- Any Part-1 exit pick re-read on the entries a Part-2 timing rule keeps, and any timing rule re-read under the Part-1 pick; a sign flip between reads = no ship.

## ADDENDUM (timing rules) — frozen 2026-11-20 after the DESIGN-only map (`HT_WILLY_2026-11-20_timing_map_DESIGN.csv`), BEFORE any HOLDOUT read of a rule
Part-2 outcome that triggered this: neither sleeve's real entry beats random-in-episode (WILLY C_EP / C_LOC: no difference; HT C_EP / C_LOC: no difference,
C_EPD: random beats real). Already seen at freeze: the Part-2 HOLDOUT real-vs-random rows (a test, not a rule pick) and the Part-1 grid.
DESIGN map reading used for the rules (HT, C_EP draws): entries 1.4–4 % below the spike high H average −1.38 (779 draws / 85 fills), 4–8 % below +0.48,
8–15 % below +0.90; real HT fills 1.4–4 % below +0.13 (61), 4–8 % below +1.45 (23). Mechanism (theory first): HT's +5 TP from a shallow dip needs a NEW high
above H (a breakout); from ≥ 4 % below H the +5 sits at or under H (a retest of the high) — a retest is the more common post-spike path.
WILLY: no bucket where random draws beat the real fills robustly (all within ±0.3, and the real-fill "since" pattern runs opposite to the random map = confound) → NO WILLY rule.
Rules (HT only; family of 3; exit = HT REF TP 5 / SL 4 / 24 h; one trade per coin per holding period; a skipped signal = 0):
- **HT_R1 DELAY_DIP4**: after the HT signal (already dip ≥ 1.4), enter at the first minute whose previous 1m close ≤ H × 0.96, within 6 h of the signal; else skip.
- **HT_R2 FILTER_DIP4**: take the HT signal only if its own dip ≥ 4 % (no delay); else skip.
- **HT_R3 DELAY_DIP8**: as R1 with H × 0.92.
Δ per SIGNAL = rule net (0 if skipped) − REF net; also per-fill avg. Bar: DESIGN Δ > 0 ∧ HOLDOUT Δ > 0 with day-block CI low > 0 ∧ HOLDOUT t > the family
max-t 95th (day sign-flip over the 3 rules) ∧ ≥ 3/4 half-years ∧ top pair < 50 % of +Δ ∧ current-batch real fills not contradicting. Joint: each rule also read under
the Part-1 pick exit (T8_S3_H12, failed) and the pick re-read on each rule's kept entries; a sign flip = no ship.
