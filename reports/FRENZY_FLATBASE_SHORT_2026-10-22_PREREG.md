# FRENZY flat-base short — PRE-REGISTRATION (frozen 2026-10-22 ~17:00 UTC, before any outcome of this study was computed)

Operator hypothesis (2026-10-22): "FRENZY pairs that were FLAT / almost dead for days or weeks before the frenzy end up going down strong
afterwards. Shorting them with longer holds would give massive wins." Example IOSTUSDT: flat ≈ 0.0007 for weeks, +150 % 10-19 → 10-20,
−47 % from the peak on 10-22. Funding on IOST at that time ≈ −0.047 %/h (the short PAYS ≈ 1.1 %/day).

## Theory (first principles, written before any number)
A coin with no market for weeks has no natural holders above the base: the frenzy buyers are momentum / wash / listing-pump flow. When
that flow stops, nothing supports the price above the old base, so it should retrace toward it over DAYS (not hours — the prior study
held 4–48 h). A coin that was already trending / volatile before the spike has a real market and a reason (news, sector rotation) and is
more likely to hold or continue. Counter-theory (why it may fail): (1) the post-pump move may already be priced when any entry fires
(the prior study: below-VWAP shorts were worse than random bars); (2) dead coins pumped once tend to be pumped AGAIN (same operators,
thin books) → squeezes; (3) crowded shorts on these coins pay large negative funding for days — over a 7–14 d hold this can exceed
the edge; (4) thin books gap through stops.

## Relation to the frozen re-open gate (FRENZY_SWING_SHORT_2026-10-20)
That gate re-opens the swing-short question only with a NEW information source (OI, funding at entry, book depth). Pre-spike flatness is
NOT a new information source: it is a new CONDITIONING VARIABLE computed from the same 5m klines. Therefore this study does NOT re-open
the gate by itself. It is admitted only because (a) the flatness state, (b) multi-day holds, and (c) exhaustion-timed entries were never
tested, and it must clear the gate's statistical bars ON ITS OWN, pre-registered, with no re-fitting. A pass here would be a new sleeve
candidate; a fail leaves the 10-20 gate exactly as frozen.

## Episodes (cohort = the live engine's own FRENZY episodes)
`scripts/study_frenzy_swing_short_walk.py` (the prior study's re-implementation of services/frenzy.py frenzy_walk, 96.5 % bar parity),
UNCHANGED, on every USDT perp, incl. pairs delisted since 2024-06 where data.binance.vision has their 5m history (survivorship fix), klines
extended to 2026-10-22 (public data only: data.binance.vision zips = 0 API weight; fapi klines for 10-22, ≤ 1,200 weight/min, header read,
stop on 418/429, scripts/binance_cooldown.py honoured). Parity re-checked on a fresh sample incl. the live-block episodes and against every
real FRENZY fill. Episode = verified ∧ ever on the shortlist while flagged (as before).
Blocks: ③ year = spike 2025-01-01 → 2026-10-01 (periods 2025 / 2026-H1 / 2026-H2) · ② master = 10-02 → 10-13 · ① live = 10-14 → now.
①② count REAL episodes only (journal FRENZY rows or FRENZY fills inside the episode, as in the 10-20 study); every walk episode of ①② is
also listed.

## Flatness (measured strictly before the pump: window ends at the BASE bar = the close 30 min before the spike bar)
- **PRIMARY FLAT14** = max(close) / min(close) − 1 over the 14 days (4,032 5m closes) ending at the base bar. FLAT = bottom tertile; the
  threshold is the 33.3rd percentile of FLAT14 over year-block episodes with spike < 2025-07-01 ONLY, frozen numerically in the addendum
  below before any outcome is computed, then applied unchanged to every block. Unscored (< 90 % real bars in the window, i.e. listed < 14 d)
  = its own line, never "non-flat".
- Secondary (reported, never a ship basis): FLAT30 (same on 30 d) · RV14 (std of 1 h log returns over 14 d) · DEADVOL (mean daily quote
  volume over the 7 d before the base / median daily quote volume over the 60 d before those 7 d) · DIST90 (base / 90-d high − 1).
  Each split at its own pre-2025-07 bottom tertile.
- Dose-response: the primary cell across quintiles of FLAT14 (quintile edges from the pre-2025-07 episodes too).

## Entries (signal on a 5m close; fill at close − 0.05 % slippage; search window spike → spike + 96 h; ONE trade per episode per entry;
no new short on a pair while the same cell's short on that pair is still open; universe gates as before: the pair was on the shortlist
while flagged at some bar from the spike to min(entry, flag-off), 24 h quote volume ≥ $20M at entry, not BTC / ETH)
- S1 flag drop (the 10-20 E1, for comparison).
- S2 exhaustion: the first 5m close ≤ 0.85 × running peak (max high since the spike bar) while that peak is ≥ 6 h old (72 bars).
- S3 time: the close 24 h after the spike bar.
- S4 the first UTC-aligned 4 h candle (fully after the spike bar) that closes below the spike-anchored VWAP (VWAP continued past the
  episode end with the engine formula).

## Exits (18 per entry per leverage)
hold 3 d / 7 d / 14 d (exit at the close) · TP: none / BASE (price ≤ base close; "none" when the entry is already ≤ 1.01 × base) ·
stop: none / 25 % / 50 % above entry, filled at max(level, bar open) + 0.05 % (gap-aware), stop assumed before TP in a shared bar ·
leverage 1× / 2× / 3×: liquidation when the high reaches entry × (1 + 1/L − 0.01) → loss = −100/L % of notional (the whole isolated
margin; under CROSS margin the account could lose more — stated, not modelled). The earliest of stop / liquidation / TP / time wins.
Pair data ending inside the hold (delisting): settled at the last close, flagged. Data ending at "now": marked to market, flagged OPEN.
→ 4 entries × 54 = 216 cells. Results are % of notional at 1× size (leverage only changes the liquidation level).

## Costs
0.20 % round trip (taker 0.05 % × 2 + 0.05 % slippage × 2) + REAL funding summed over every settlement inside the hold (short receives
+rate, pays −rate; monthly data.binance.vision zips, fapi fundingRate for 2026-10). Missing funding history: −0.10 %/8 h assumed, counted
and reported. Every result is reported before AND after funding.

## Primary / secondary (pre-declared)
PRIMARY = S2 · 7 d · TP BASE · stop 25 % · 1× on the FLAT cohort. SECONDARY = S3 · 7 d · TP BASE · stop 25 % · 1× on FLAT.
All other cells exploratory, judged family-wise.

## Nulls
1. Random bars, same episodes: a uniform bar inside [spike + 2 h, spike + 96 h] passing the same universe gates; same exit; funding
   included; 200 shuffles; own p = share of shuffles ≥ real; family-wise p = max-z over all 216 cells.
2. Non-flat contrast: same cell on non-flat (scored) episodes; Δ = flat − non-flat mean.
3. Shuffled flatness labels: permute the FLAT label across scored year episodes (1,000 permutations, label count preserved); p of Δ; family-wise
   by max-z over all cells.

## Robustness (reported for primary + any cell that passes the family-wise tests)
day-clustered bootstrap 95 % CI (window = UTC entry day; a market-wide squeeze day such as 2025-10-11 is ONE window by construction) ·
2025 / 2026-H1 / 2026-H2 · leave-one-month-out · best 5 % removed · per-pair and per-day share of profit · survivorship: pairs no longer
trading (delisted) — their count and their cohort result vs listed pairs.

## Sizing / ruin (account $3,200)
size so the 1st-percentile trade loss = 5 % of the account ($160); ≤ 5 concurrent shorts, one per pair, first-come; report sum, worst trade,
worst day, max drawdown (realised and daily mark-to-market), funding $ over the holds.

## SHIP bars (all required, primary flat cohort, year block ③, after costs + funding)
1. N ≥ 300 trades on ≥ 30 distinct days. 2. Day-bootstrap 95 % lower bound > 0. 3. Beats non-flat episodes (shuffled-label p) AND random
bars (own null) at p < 0.05 — family-wise p < 0.05 for any non-primary cell. 4. Mean > 0 in 2025, 2026-H1 and 2026-H2. 5. ≥ 8 live/master
windows (UTC days with a real flat-episode primary trade, open ones marked to market), mean of window means > 0 and ≥ half the windows
positive. 6. Mean × 0.6 (40 % haircut) ≥ +0.50 %/trade. Plus the 10-20 robustness bars: every leave-one-month-out mean > 0, mean > 0 with the
best 5 % removed, no pair and no day > 25 % of the total profit.
- OBSERVE-ONLY (at most): bars fail but the primary's year CI lower bound > 0, OR (mean > 0 in all three periods AND own non-flat-contrast
  p < 0.05). Then a frozen gate is written; no money.
- REJECT otherwise. Thresholds above are not moved after results.

## Addendum (frozen numbers, filled after the episode walk and BEFORE any trade outcome is computed)
(see below)
Frozen 2026-10-22 after the walk (7,114 episodes, 613 pairs, incl. the delisted pairs' history), before any signal / exit / outcome was run:
- FLAT14 bottom-tertile threshold = **44.39 %** (1,139 year episodes with a spike 2025-01-01 → 2025-06-30; 1,134 scored). FLAT = 14-d close
  range ≤ 44.39 %.
- FLAT14 quintile edges = 34.81 / 49.44 / 69.24 / 103.66 %.
- Secondary bottom-tertile thresholds: FLAT30 75.84 % · RV14 1.281 % · DEADVOL 0.986 · DIST90 −61.97 %.
- Note (no change to any rule): FRENZY pairs are volatile even when "flat" — the bottom tertile still allows a 44 % 14-d range. The
  operator's "dead coin" picture is closer to the bottom quintile (≤ 34.8 %), which the dose-response reports.
