# NEWSLEEVE 2026-11-21: four new-sleeve ideas on 3.4 years of Binance history (H5 · H6 · H1 · H2)

**Verdict: no sleeve. All six idea-directions are REFUTED on HOLDOUT. Nothing goes to paper and no new observe line comes out of this study.**
One result passed all 8 gates at first: H2 squeeze-fade, HOLDOUT 743 · 72 % · +1.51 %. It was a **data artefact**. From 2024-05 on, Binance's
public `metrics` files label each open-interest snapshot 5 minutes before it was taken, so the OI leg was reading 5 minutes into the future.
With the timestamps fixed, H2 is −1.45 % (value OI) or −1.20 % (contracts OI) on HOLDOUT. See §2. This matters for other studies in the repo too.

Research only. No engine, config or AWS change. Pre-registration (frozen before any outcome was computed): `NEWSLEEVE_2026-11-21_PREREG.md`.
Generated tables: `NEWSLEEVE_2026-11-21_tables.md` (OI value, as pre-registered) · `NEWSLEEVE_2026-11-21_data/NEWSLEEVE_2026-11-21_tables_contracts.md`
(OI contracts, the theory-correct variant, disclosed below).

## 1 · Summary (HOLDOUT 2025-07 → 2026-11-17, read once at the DESIGN pick, net of all costs, 5m walk)

| idea | prior work it overlaps | DESIGN pick (2023-07 → 2025-06) | DESIGN avg | null 95th | HOLDOUT N · days | WR | **avg net %** | day-boot 90 % CI | random control | paired Δ lower 95 % | half-years > 0 | gates | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H5B buy-climax → short | SPIKE_FADE · EXH · DL 164 | ratio ≥ 10 · share ≥ 0.65 · TP1/SL4/6h | −0.27 | −0.32 | 5,526 · 504 | 74 % | **−0.32** | −0.37 … −0.28 | −0.43 | +0.04 | 0/3 | 2/7 | REFUTED |
| H5S sell-climax → long | SPIKE_BOUNCE (dead 08-10) | ratio ≥ 10 · share ≤ 0.35 · TP4/SL4/1h | −0.19 | −0.30 | 11,406 · 504 | 41 % | **−0.34** | −0.39 … −0.29 | −0.38 | −0.02 | 0/3 | 1/7 | REFUTED |
| H1 cascade bounce (OI value) | SPIKE_BOUNCE · DL 164 | r15 ≤ −5 % · dOI ≤ −3 % · TP4/SL4/6h | +0.16 | −0.13 | 10,442 · 468 | 48 % | **−0.36** | −0.60 … −0.13 | −0.09 | −0.49 | 0/3 | 2/8 | REFUTED |
| H1 (OI contracts) | " | r15 ≤ −8 % · dOI ≤ −3 % · TP4/SL4/6h | +0.73 | −0.10 | 2,347 · 392 | 45 % | **−0.67** | −0.80 … −0.53 | −0.01 | −0.87 | 0/3 | 2/8 | REFUTED (sign flip) |
| H2 squeeze fade (OI value) | SPIKE_FADE · EXH · DIST | r15 ≥ +8 % · dOI ≤ −1 % · TP1/SL4/1h | −0.04 | +0.67 | 175 · 93 | 56 % | **−1.45** | −1.74 … −0.91 | −0.62 | −1.15 | 0/3 | 1/8 | REFUTED |
| H2 (OI contracts) | " | r15 ≥ +8 % · dOI ≤ −1 % · TP1/SL4/1h | −0.08 | +0.26 | 1,751 · 379 | 60 % | **−1.20** | −1.63 … −0.77 | −0.47 | −1.11 | 0/3 | 1/8 | REFUTED |
| H6L funding ≤ −0.5 % → long | FUNDING_RULE 11-19 · DIST_FUNDING_EXIT | frozen · TP3/SL6/8h | −0.06 | +1.11 | 1,158 · 402 | 60 % | **−0.36** | −0.58 … −0.15 | −0.15 | −0.49 | 1/3 | 1/8 | REFUTED |
| H6S funding ≥ +0.1 % → short | none found | frozen · TP6/SL6/24h | **+1.58** | +0.64 | 598 · 291 | 51 % | **−0.23** | −0.64 … +0.20 | −0.26 | −0.44 | 1/3 | 2/8 | REFUTED (era effect) |

UNVERIFIED-by-design caveats apply to every number: the walk uses 5m bars (stop checked first when a bar touches both; a resting TP needs
0.05 % through), entry is the next bar open with a modelled slip (0.02 / 0.05 / 0.10 % by 24 h volume, doubled on entry and on stops), and there
is no order book. None of these assumptions can turn a −0.3 % … −1.4 % HOLDOUT into a positive one, and the 1m re-walk was not needed because
nothing reached CANDIDATE. The slip stress makes it worse by a further −0.20 %.

## 2 · The open-interest timestamp trap (the most important finding here)
- **What happened.** The first full run passed H2 on every gate (HOLDOUT 743 · 72 % · +1.51 %, Kelly at the floor +0.34, 3/3 half-years,
  beat the control by +1.5 %). The pass came almost entirely from 2025–26. Without the OI leg the same trigger was −0.4 … −0.8 %. That pattern
  pointed to look-ahead, so I tested the timestamps.
- **Test.** `sum_open_interest_value / sum_open_interest` gives the mark price at the snapshot, and I compared it with the 5m closes. For
  SOL 2025-10-10 it matches the close **5 min after** `create_time` (median error 0.006 %, vs 0.17 % at `create_time`). For DOGE 2024-03-05
  it matches at `create_time`. Every pair-day was then aligned the same way
  (`NEWSLEEVE_2026-11-21_data/oi_alignment_by_month.csv`): **through 2024-04 the label is the snapshot time. From 2024-05 to now, 95–100 % of
  files label a snapshot taken 5 min later.**
- **Effect.** Read naively, "OI fell in the last 15 min" in 2024-05+ includes the first 5 minutes AFTER the trigger, which is exactly when a
  squeeze unwinds. Aligned, H2 drops to −1.45 % and fails the null on DESIGN.
- **Consequence for the repo (outside this study's scope, flagged):** 20 scripts read vision `daily/metrics` (e.g. `STATE_EXIT_2026-11-21_common.py`,
  `high_atr_data.py`, `exh_exit2_data.py`, `leads_deep_fetch.py`, `frenzy_early_data.py`, the `oos_real_*` / `wl_deep_*` feature builders,
  `verify_1117_lib.py`). Any backtest feature taken at `create_time ≤ t` from 2024-05 onward reads 5 min ahead. The DISTRIBUTION_SHORT trigger
  leg (top-trader position L/S 4 h change) and the EXH spot-share / OI features come from this file family. A 5-min lead on a 4 h change is
  probably small, but it is untested. Recommended: one audit that re-aligns those features with the implied-price method in
  `scripts/NEWSLEEVE_2026-11-21_eval.py` (`oi_at`) and re-reads the shipped thresholds. Live engine reads come from fapi in real time and are
  not affected.

## 3 · Second disclosed correction: OI value vs contracts
The pre-registration said "OI value". Value = contracts × price, so an 8 % price drop by itself "drops OI" by ~8 %. H1's OI leg was therefore
almost always met (dOI ≤ −1 % kept 7,741 of the possibilities, ≤ −3 % kept 7,567), while H2 needed contracts to fall about 11 %. The theory
("positions closed") is about contracts. Both versions are shown, and both are refuted, so the error does not change any verdict. H1 on contracts
is the classic overfit shape: DESIGN +0.73 % (null passed) → HOLDOUT −0.67 % on every half-year.

## 4 · Method (full detail in the PREREG)
- **Data:** 5m USDT-M klines with taker-buy quote volume for 913 symbols listed on data.binance.vision, delisted ones included (0 Binance
  weight, 5.2 GB, gitignored). 5m `metrics` for 30,745 pair-days touched by H1/H2 candidates. Real funding settlements (reused
  `HIGH_ATR_2026-11-18_data/funding*`). BTC regime = spot daily EMA50 vs EMA200 since 2017 (warm-up), state on the previous day.
- **Universe:** alts only, trailing 24 h volume ≥ $10M, ≥ 7 days of history.
- **Triggers:** on the 5m bar close, using the last 15 min. Entry at the next bar open (no look-ahead, apart from the metrics issue in §2,
  now fixed).
- **Exit grid:** H5/H1/H2 use TP {1, 2, 4} % × SL {2, 4} % × {1 h, 6 h}. H6 uses TP {3, 6} × SL {3, 6} × {8 h, 24 h}.
- **Costs:** taker 0.05 / maker 0.02, slip by tier, real funding signed by side.
- **Possibilities, not events:** one per coin per holding period (6 h, or 24 h for H6). Counts are also given in 15-min windows and in days,
  and the bootstrap resamples UTC DAYS.
- **Controls:** a random entry on the same coin, same UTC day (≥ 1 h away), same direction and exit, paired.
- **Null:** 100 replicates of the full grid with random entries. The DESIGN pick must beat the 95th percentile of the replicate max.
- **Promotion bar:** Kelly > 0 at the one-sided 95 % Wilson floor · day-bootstrap lower bound > 0 · ≥ 2/3 half-years · +0.20 % slip ·
  no pair/day ≥ 50 % · beats the control · null · ≥ 30 days.

## 5 · Per idea: what the data say

### H5 taker-climax fade (`NEWSLEEVE_2026-11-21_H5B.png`, `_H5S.png`)
- **Frequency:** very common. At ratio ≥ 5 / share 60 %: 41.6k buy and 70.8k sell possibilities on DESIGN (27k / 30k distinct 15-min windows,
  all 731 days). The strict variant (10× / 65 %) has 3.7k / 8.3k.
- **Edge:** the fade is a hair better than shorting the same coin at a random time the same day (paired Δ ≈ +0.10 %, lower bound +0.04 on
  H5B). That edge is about one fifth of the round-trip cost on these mostly $10–50M coins. This repeats Route A's "ρ −0.027 ≈ 2 bps" on 160k
  possibilities.
- **Splits:** negative in every half-year (−0.29 … −0.35), both BTC regimes and every volume tier. The ≥ $200M tier comes closest (−0.11 …
  −0.22), but it is still negative.
- **Sell side:** the sell-climax long is SPIKE_BOUNCE's failure again, now on 11k fills.
- **Verdict:** REFUTED. Retire Route B line H5 as a sleeve idea. It can stay as a descriptive count only.

### H6 funding extremes (`_H6L.png`, `_H6S.png`)
- **H6L (≤ −0.5 %, long, receives funding):** −0.36 % on HOLDOUT (1,158 · 402 days) and −0.06 % on DESIGN. Collecting the funding does not
  pay for the drift. Deeply negative funding sits on coins that keep falling, which agrees with FUNDING_RULE 11-19, where shorts at ≤ −0.5 % still
  earned money.
- **H6S (≥ +0.1 %, short):** +1.58 % on DESIGN (869 fills, 2023-H2 → 2024 alt bull, all BTC-bull days), then −0.23 % on HOLDOUT, positive in
  only 1 of 3 half-years and losing to the control.
- **Meaning:** the DESIGN number was the 2024 crowded-long unwind episode, not a repeatable rule, and the BTC-regime split only shows that it
  belonged to that era.
- **Verdict:** both REFUTED.

### H1 long-liquidation cascade bounce (OI proxy) (`_H1.png`)
- **Frequency:** sharp drops with heavy volume are frequent: ~9k (DESIGN) and ~15k (HOLDOUT) possibilities over 450–600 days, clustered in
  dump days.
- **Edge:** HOLDOUT is −0.36 % (value) or −0.67 % (contracts), it loses to its own random control, and it is negative in every half-year and
  both regimes.
- **Does the OI leg add anything?** Without it the same trigger is −0.40 % at TP2/SL4/6h. With it, −0.35 % (value) or −0.39 … −0.77 %
  (contracts). It adds nothing.
- **Verdict:** REFUTED as an OI-proxy sleeve. Route B's H1 line uses REAL liquidation prints (forceOrder ≥ the coin's 99th percentile, ≥ $50k),
  which is a different variable. Keep it as registered (DAY / dump-episode units), but with a lower prior: the price + volume + OI footprint of
  a cascade does not bounce profitably at 5m.

### H2 short-squeeze fade (OI proxy) (`_H2.png`)
- **Edge:** aligned, HOLDOUT is −1.45 % (value) or −1.20 % (contracts), with 0/3 half-years positive. The DESIGN pick failed the shuffled null
  in both versions.
- **Why it loses:** fading a +8 % / 15-min squeeze with a 4 % stop gets run over. Losers average −4.3 % against winners of +0.8 % at TP1, and
  the breakeven WR is 84 %.
- **Compared with the bot's fades:** EXH / DIST / SPIKE_FADE wait for exhaustion signals (RSI roll-over, volume decay, L/S shift) and use wider
  stops. That is what this naive fade lacks.
- **Verdict:** REFUTED. Route B's H2 line (real short-liquidation prints) is worth keeping only as the EXH / DIST **timing** question it was
  registered for, not as a sleeve.

## 6 · Event-study charts
Each `NEWSLEEVE_2026-11-21_<idea>.png` shows the MEDIAN price path in trade direction from −1 h to +6 h (H6: +24 h) around the entry. It splits
DESIGN from HOLDOUT, and winners from losers at the pick, against the random control. **H5B is the one chart with a visible effect:** the median
buy-climax coin drifts +0.3 % (DESIGN) / +0.5 % (HOLDOUT) in the short's favour over 6 h, against ~0 for the control, in both blocks. The fade is
real as a median tendency, but a fat adverse tail (losers −3.35 % at SL4) and ~0.4 % round-trip cost on mostly $10–50M coins make the
expectancy −0.32 %. The other charts show no favourable drift beyond the control. In all of them winners and losers part company only after
entry, so nothing at the trigger separates them.

## 7 · Blind spots / not tested
- There is no order book, so H3 and H4 are left to Route B.
- Real liquidation prints are not used. Binance publishes no history of them, so OI was only a proxy.
- The walk is 5m, not 1m, which matters little at these losses.
- Survivorship: symbols missing from vision entirely are not covered.
- Thresholds were frozen from theory. Other X / Y values were not searched, on purpose.
- The BTC regime is confounded with the period: DESIGN has only ~9 bear days, while HOLDOUT 2025-11 → 2026-09 is mostly bear.

## Files
`reports/NEWSLEEVE_2026-11-21_PREREG.md` · `reports/NEWSLEEVE_2026-11-21.md` · `reports/NEWSLEEVE_2026-11-21_tables.md` ·
`reports/NEWSLEEVE_2026-11-21_{H5B,H5S,H1,H2,H6L,H6S}.png` · `reports/NEWSLEEVE_2026-11-21_data/` (small CSVs: counts, design grids,
holdout halves, regime, capacity, summary, `oi_alignment_by_month.csv`, `H2_naive_lookahead_artefact.csv`, the `_contracts` variant) ·
`scripts/NEWSLEEVE_2026-11-21_fetch.py` · `_build.py` · `_eval.py`. Big caches (k5m 5.2 GB, met, pkl) are gitignored.
