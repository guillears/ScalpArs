# NO-SPOT coins and coin SIZE for the exhaustion short (2026-11-18)

**Pre-registered:** `NOSPOT_MCAP_2026-11-18_PREREG.md`, frozen 12:55 UTC before any result. Every 2023-07 → 2026-11 number below is a
**robustness read** (those blocks were read by EXHAUSTION_EXPAND / SPOT_SHARE_DEEP). **HOLDOUT-E (2023-01 → 06)** was built new for this
study and read once, in the final run.

Research only. Nothing in the bot changed. No commit.

## Questions
- **A.** Does the EXHAUSTION_SHORT trigger work on coins with **no Binance spot market** (today they are blocked as `EXH_NO_SPOT`)?
  Would adding them raise trades/day without lowering quality?
- **B.** Does coin **size** (market cap, FDV, 24 h volume, OI, turnover, listing age, Alpha listing) separate winners from losers —
  on EXHAUSTION_SHORT's own trades and on the no-spot coins? Operator hypothesis: coins under ~$50 M dump harder after the Frenzy peak.

## Plain answer
1. **A — NO.** On no-spot coins the same trigger **loses money**: 2,795 trades, **−0.09 %/trade**, 5 of 7 half-years negative,
   Kelly < 0. Adding them to EXHAUSTION_SHORT would cut its average in half (+0.43 → +0.22 %/trade) and turn its Kelly to zero.
   - A perp-only replacement for the spot filter (OI falling over the last hour) lifts no-spot to +0.08 %/trade, still failing 4 items.
   - The operator's coins are the worst no-spot names in history: **PIPPIN −55 %, BLESS −51 %, JELLYJELLY −66 %** (summed P&L of all their shorts).
2. **B — size does not separate the exhaustion short.**
   - EXHAUSTION_SHORT trades: **≤ $50 M +0.31 %/trade (985) vs > $50 M +0.35 % (2,526)**. Small caps are not better; family-wise p = 1.0.
   - No other size proxy passes the null (best p = 0.72 on EXHAUSTION_SHORT).
   - No-spot coins: small caps lose less (−0.07 vs −0.20) but **both sides lose**.
   - The clean 2023-H1 holdout points the other way again (≤ $50 M: 19 trades, −0.46).
3. **The coins really did dump** after the peak this week (BLESS −52 %, AIOT −57 %, BREV −36 %, HUMA −33 %, SPACE −30 %).
   - The exhaustion trigger mostly fires **during** the pump. A 5 % stop gets hit there.
   - Or it does not fire at all: no 15m RSI cross with vol_decay ≥ 0.64 on AIOT, PIPPIN, ZKC or BREV.
   - "Small coins dump" is true as a description. It is not a tradable edge for this entry.
4. **New, and not good:** on the never-read 2023-H1 holdout, **EXHAUSTION_SHORT itself** made only +0.10 %/trade (259 trades).
   - It is negative at 0.20 % slippage and after removing the best 5 %, Kelly is < 0, and one coin (GAL) carries 57 % of the profit.
   - It is the first clean read of the shipped rule outside 2023-07 → now.

## Method
- **Rows:** the EXHAUSTION_EXPAND all-spike build (`allspk/all.pkl`, every bar-rule spike, no later-flag selection).
  - **Parity:** REF reproduces exactly: **N 4,132 · 3.34/day · +0.425 %/trade · 23-H2 +0.50** (EXPAND §0 (3): 3.34/day, +0.43, +0.50).
- **Trigger, all cohorts:** Frenzy window, 15m RSI14 cross 70 → ≤ 70, vol_decay ≥ 0.6366, vol24 ≥ $20 M, entry at the next 5m open.
- **Exit EA:**
  - TP = min(3 × ATR15, 5 %), resting maker, filled only through ≥ 1 tick.
  - SL = min(1.5 × ATR15, 5 %), gap-aware.
  - 24 h time exit.
  - One position per pair. Real funding, incl. hourly coins.
  - Fees: taker 0.05 %, maker 0.02 %.
- **Slippage:** live model slip = max(0, −0.042 + 0.276·√(6000 ÷ Q5)) at the shipped ≈ $6k notional.
  - On these coins Q5 is large (median $1.3–1.6 M per 5 min), so the model gives 0 on 96–98 % of trades.
  - That is why flat 0.10 / 0.20 % are shown too.
  - The engine depth guard needs the live order book, so it **cannot be rebuilt** historically.
- **has_spot** per (pair, month): a spot 15m zip exists on data.binance.vision for that month. 2,387 pair-months were "missing" (no spot).
  - 101 candidate rows (1 %) had unknown status and were dropped.
  - Spot status changes over time: many coins list the perp first and spot later.
- **NS-P proxy, pre-registered:** OI 1 h change ≤ 0. Source: Binance metrics, create_time = period start, last row ≤ entry − 5 min. 100 % scorable.
- **HOLDOUT-E:**
  - Built from data.binance.vision only (5m klines 2022-12 → 2023-07, all 916 symbols incl. delisted).
  - Same all-spike rule: 541 spikes, 156 pairs. Same builder, spot and funding.
  - 6 % of rows had no funding month and got the conservative penalty.
  - API weight 0.

### How market cap was obtained (the blind spot)
- **Source:** the bot's own source, the Binance "Info panel" endpoint in `services/mcap_service.py`. It is undocumented and serves CoinMarketCap data.
  - It gives today's circulating supply (`cs`), total supply (`ts`), mc and FDV.
  - 421 pairs were already cached (Oct-8). 226 more were fetched today at ≥ 1.6 s spacing, with no 418/429.
  - Entry mcap = circulating supply × entry price ÷ contract multiplier. FDV = total supply × entry price ÷ multiplier.
- **No full history exists.** The endpoint has no history. CoinGecko's free API serves 365 days only. No account or key was used, and no new CoinGecko calls were made.
- **Constant-supply assumption, checked three ways:**
  - **vs the bot's own `entry_mcap_usd` stamps:** 594 fills on 169 pairs (Sep-28 → Nov-18). Median ratio 0.998 (IQR 0.96–1.01). The ≤ $50 M side agrees on **91 %** of fills.
  - **CoinGecko supply drift** (365-day cache, 904 rows): supply at entry ÷ latest supply median 0.999. 8 % of rows show ≥ 25 % supply growth since then.
    - So older mcaps are somewhat **overstated** for coins with unlocks. 2023–24 drift cannot be checked.
  - **vs the Binance Alpha list's own circulating supply** (188 pairs): median ratio 1.03.
- **Coverage:** mcap is known for **95 %** of rows on coins listed today, but only **9 %** on delisted coins.
  - Scored share: REF 85 %, no-spot 71 %, HOLDOUT-E 14–64 %.
  - Unscored trades are shown apart, never as "rest". **They are the best shorts** (REF unscored +0.91 %/trade, 621 trades): coins that later died dumped hardest.
  - So every size read covers survivors only.
- **Proxies used over the full span:**
  - vol24, Q5, OI $ (≈ 100 % coverage), turnover (vol24 ÷ mcap), OI ÷ mcap, listing age (futures onboardDate), and the Alpha flag.
  - The Alpha flag is today's list: look-ahead, descriptive only.
  - Agreement with mcap (Spearman): OI $ +0.55, vol24 +0.41, Q5 +0.39, listing age +0.12, FDV +0.95.
  - **The proxies are weak stand-ins for mcap.**

## A. No-spot cohort per half-year (avg net %/trade; EA; slippage model; N in brackets)

| cohort | 23-H1 (E, clean) | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 |
|---|---|---|---|---|---|---|---|---|
| **REF** = EXHAUSTION_SHORT | **+0.10 (259)** | +0.50 (313) | +0.50 (475) | +0.40 (472) | +0.09 (446) | +0.49 (885) | +0.15 (851) | +0.82 (690) |
| **NS0** no-spot, no filter | +0.45 (12) | −1.31 (28) | +1.16 (49) | −0.18 (124) | **−0.63 (242)** | **−0.18 (873)** | +0.17 (843) | **−0.14 (636)** |
| **NS-P** no-spot ∧ OI 1 h ≤ 0 | −2.33 (2) | −0.52 (5) | +3.49 (7) | −0.35 (12) | **−1.11 (79)** | +0.04 (303) | +0.21 (288) | +0.30 (230) |

## A. Union vs EXHAUSTION_SHORT alone (2023-07 → 2026-11, one size)

| cohort | /day | WR / BE | avg | @0.10 · @0.20 | best-5 % out | Kelly f (WR floor, daily) | DCR %/day at 0.6× | max loss · worst 1 % | hourly-funding trades (avg) | items 1–6 |
|---|---|---|---|---|---|---|---|---|---|---|
| **REF alone** | 3.34 | 51.8 / 47.1 | **+0.43** | +0.28 · +0.13 | +0.19 | **+0.040** (49.3 %) | +0.72 | −7.2 · −6.2 | 365 (+0.05) | pass |
| NS0 alone | 2.26 | 48.0 / 49.0 | −0.09 | −0.24 · −0.40 | −0.36 | −0.079 | −0.22 | −6.5 · −5.5 | 296 (**−0.58**) | fail 1, 2, 3, 4, 6 |
| NS-P alone | 0.75 | 50.5 / 49.8 | +0.08 | −0.07 · −0.22 | −0.19 | −0.061 | +0.005 | −6.4 · −5.5 | 125 (−0.22) | fail 1, 2, 3, 4, 6 (IN 48 % of $) |
| **U0 = REF ∪ NS0** | 5.60 | 50.3 / 48.0 | +0.22 | +0.07 · −0.08 | −0.03 | −0.003 | +0.50 | −7.2 · −6.0 | 661 (−0.23) | fail 1, 2, 3, 4, **5 (avg −49 %)** |
| UP = REF ∪ NS-P | 4.09 | 51.6 / 47.6 | +0.36 | +0.21 · +0.06 | +0.12 | +0.027 | +0.73 | −7.2 · −6.1 | 490 (−0.02) | fail 1 (25-H1 −0.09), **5 (Kelly −33 %)** |

- **DCR** = daily compound return over every calendar day. Each fill is sized at 0.6 × equity (≈ $6k on $10k), with no slot cap. It is descriptive only.
- **UP's DCR (+0.73) is level with REF's (+0.72):** the 924 extra trades add almost nothing per day and cost a third of the Kelly.
- **HOLDOUT-E (2023-H1, read once):**
  - REF +0.10 (1.4/day), U0 +0.12, UP +0.08.
  - NS0: 12 trades on 8 days, +0.45. That is too few to read (item 7 = "no read").

## B. Size W/L (2023-07 → 2026-11; avg net %/trade; low side vs high side; family-wise p from 2,000 P&L shuffles over every var × cut)

| cohort | variable | cut | coverage | low side N · WR · avg | high side N · WR · avg | halves low > high | p_fw | block bar |
|---|---|---|---|---|---|---|---|---|
| REF | **mcap** | **≤ $50 M vs > $50 M** | 85 % | 985 · 52.3 % · +0.31 | 2,526 · 50.4 % · +0.35 | 3/7 | 1.00 | no |
| REF | mcap | median $119 M | 85 % | 1,756 · +0.30 | 1,755 · +0.38 | 3/7 | 1.00 | no |
| REF | mcap | T1 ≤ $62 M vs T3 > $230 M | 85 % | 1,171 · +0.35 | 1,170 · +0.34 | 4/7 | 1.00 | no |
| REF | FDV | T1 vs T3 | 85 % | +0.30 | +0.41 | 3/7 | 1.00 | no |
| REF | vol24 | median $85 M | 100 % | +0.37 | +0.48 | 2/7 | 1.00 | no |
| REF | Q5 | T1 vs T3 | 100 % | +0.27 | +0.40 | 3/7 | 0.99 | no |
| REF | OI $ | T1 ≤ $6.6 M vs T3 | 100 % | +0.55 | +0.41 | 3/7 | 0.99 | no |
| REF | turnover | median 0.74 | 85 % | +0.37 | +0.31 | 5/7 | 1.00 | no |
| REF | listing age | T1 ≤ 339 d vs T3 > 838 d | 98 % | +0.29 | +0.54 | 3/7 | 0.72 | no |
| REF | Alpha (today) | listed vs not | 100 % | 732 · +0.36 | 3,400 · +0.44 | 2/5 | 1.00 | no |
| NS0 | **mcap** | **≤ $50 M vs > $50 M** | 71 % | 889 · 49.2 % · −0.07 | 1,106 · 46.4 % · −0.20 | 3/4 | 1.00 | no |
| NS0 | vol24 | T1 ≤ $50 M vs T3 | 100 % | 932 · +0.06 | 932 · −0.25 | 5/7 | 0.80 | no |
| NS0 | **OI $** | **T1 ≤ $6.7 M vs T3 > $13.7 M** | 100 % | 932 · 49.7 % · **+0.07** | 932 · 45.0 % · **−0.40** | **7/7** | **0.34** | items pass for blocking high-OI, **p_fw fails** |
| NS0 | FDV | T1 vs T3 | 71 % | −0.28 | −0.08 | 1/4 | 0.99 | no |
| U0 | listing age | T1 ≤ 223 d vs T3 | 98 % | −0.05 | +0.44 | 0/7 | **0.008** | no (composition: no-spot coins are young) |
| U0 | Alpha (today) | listed vs not | 100 % | 3,072 · +0.02 | 3,855 · +0.37 | 1/6 | 0.11 | no (composition) |

- **Coverage table:** `_coverage.csv`. **Full 1D table:** `_sizewl.csv`. **2D table** (every size var × vd, ATR %, ssc, OI 1 h change, pre-entry 1 h / 2 h move, hour; 416 cells): `_2d.csv`.
- **Blind spots:** mcap before 2025-10 assumes today's supply. Delisted coins are unscored (they are the best shorts). There is no historical Alpha flag and no order-book depth.
- **HOLDOUT-E size read** (REF, 178 scored): ≤ $50 M 19 trades −0.46 vs > $50 M 159 trades −0.02. That is the opposite of the hypothesis, at tiny N.
- **Per-window:** no market-wide variable is involved. Size is per pair, and no size cell has one day or pair ≥ 50 % of its loss.
- **Not a size effect, seen in the 2D screen (post-hoc, 416 cells, NOT a finding):**
  - On REF, OI rising over the last hour (above the median of +8.3 %) is +0.69 vs +0.17 at or below it, higher in 7/7 half-years.
  - On NS0 the sign flips.
  - This is a candidate for its own pre-registered study, nothing more.

## Event study
`NOSPOT_MCAP_2026-11-18_events.png` shows the mean gross short path from entry to +24 h.

![events](NOSPOT_MCAP_2026-11-18_events.png)

- **By mcap (REF ∪ NS0):**
  - The ≤ $50 M bucket drifts the **least** after entry (+0.5 % at 24 h).
  - $50–250 M and unscored (delisted) coins drift the most (+1.9 %).
- **Spot vs no-spot:** spot coins +1.4 % at 24 h vs no-spot +1.0 %. The no-spot band includes 0 up to +4 h.
- The coins rose ≈ 6 % in the 2 h before the signal: the entry is inside the late pump.

## Replays 2026-11-16 → 11-18 (fresh public klines; EA; slippage model; funding from fundingRate)

| coin | spot? | trades (study rule) | result | post-peak drop (peak → low so far) |
|---|---|---|---|---|
| BLESS | no | NS0 11-16 09:45 SL · 17:45 SL · 11-18 04:45 TP · 07:45 TP | −5.10 −5.10 +4.93 +4.93 = **−0.34 %** | −51.5 % (peak 11-16 17:05) |
| SPACE | no | NS0 11-16 14:00 SL · 16:00 TP | −3.52 +4.91 = +1.39 % | −29.6 % |
| AIOT | no | none: no 15m RSI cross inside the window | – | −56.9 % (peak 11-13) |
| PIPPIN | no | none: one cross 11-18 12:15 at vol_decay 0.49 (< 0.64) | – | −15.6 % (peak 11-18 10:25) |
| HUMA | yes | REF 11-16 17:30 SL (hourly funding −0.18) | −5.28 % | −32.5 % (peak 11-16 21:05, after the stop) |
| ZKC | yes | none: one cross 11-18 10:45 at vd 0.63 (just < 0.64) | – | −16.5 % |
| BREV | yes | none: one cross 11-18 02:00 at vd 0.63 | – | −36.2 % |
| ZKP (live) | yes | REF 11-18 07:30 SL (−5 % + −1.05 % funding at 08:00) · **08:30 TP** | −6.15, **+4.93** | −18.0 % |

- **ZKP live paper fill:** 11-18 08:30, +4.94 %, mcap stamp $12.9 M. It matches the replay's 08:30 TP.
- The 07:30 replay stop was not traded live.
  - Live tracks episodes from the engine's own flag. The replay uses the all-spike list, which can start an episode earlier.
  - This is worth a check by the builder, not a bug claim.
- Live N = 1, which cannot inform a size or no-spot rule.

## What it means
- **The spot-share filter is not just a "can't read it" gap. The no-spot coins are a worse population for this short.**
  - They are younger, Alpha-listed, and more often on hourly funding. Their 296 hourly-funding trades average −0.58 %/trade.
  - They squeeze through a 5 % stop more often.
  - The HOLDOUT-C "+0.66" in SPOT_SHARE_DEEP was the base trigger without the vol_decay cut. With the shipped trigger, no-spot is negative in 5 of 7 halves.
- **Size is not the lever.**
  - Within EXHAUSTION_SHORT, ≤ $50 M and > $50 M earn the same.
  - Small coins do fall far after a Frenzy peak, but they also squeeze far before it. This entry sits in that squeeze zone.
  - The only consistent size-like read is "lower OI does better" among no-spot coins (7/7 halves). It fails the family-wise null, and its good side is only +0.07 %/trade, too small to add.
- **The shipped rule's first clean out-of-sample read (2023-H1) is weak** (+0.10, Kelly < 0, GAL-dominated).
  - 2023-H1 had few Frenzy coins (1.4/day). It sits next to six positive half-years.
  - It does not refute the rule, but it widens the honest range of expectations.

## Verdict
- **A: NO. Do not add no-spot coins to EXHAUSTION_SHORT,** with or without the OI proxy.
  - NS0 and NS-P fail items 1–4. Both unions fail item 5.
  - Keep `EXH_NO_SPOT` blocking.
- **B: NO size filter and no size multiplier.**
  - mcap ≤ $50 M does not separate (p_fw 1.0, 3/7 halves, holdout opposite).
  - Nothing passes the expectancy bar together with the family-wise null.
- **OBSERVE only** (no build needed; the scout can read them from stamps):
  - entry_mcap_usd ≤ $50 M on EXHAUSTION_SHORT fills;
  - entry OI 1 h change > its median on EXHAUSTION_SHORT fills (post-hoc 2D, needs its own pre-registration).

## Side findings for the builder (no change made)
1. **Engine and study disagree on 1000-prefixed coins.**
   - The engine checks `pair in spot exchangeInfo symbols` (exact). The study fetched spot for the stripped base (1000PEPE → PEPEUSDT).
   - So 87 REF study trades on 1000-prefixed pairs can never trade live. They averaged −0.27, so the engine's behaviour is the safer one.
   - LUNA2USDT is the reverse case: its spot symbol is LUNAUSDT. It is counted as no-spot in both.
2. **HOLDOUT-E REF +0.10:** worth adding to the EXHAUSTION_SHORT revert-gate context as the realistic low end.

## Assumptions and caveats
1. Mcap = today's circulating supply × the past price. Unlocks overstate older mcaps. This is checked only for the last 365 days (8 % of rows show ≥ 25 % drift).
2. Delisted coins have no mcap (9 % coverage), and they were the best shorts. Size reads are survivor-only.
3. Alpha listing and the CMC rank are today's values (look-ahead), so they are descriptive only.
4. The slippage model is ≈ 0 at $6k on these coins. Live Frenzy fills had a median of +0.06 %, so read the @0.10 column as the realistic one. The depth guard is not rebuildable.
5. One position per pair inside each cohort. No book-level slot cap, so DCR overstates what a shared book can take.
6. All 2023-07+ reads are robustness reads. Only HOLDOUT-E is clean, and its no-spot N is 12.

## Next step
1. Keep EXHAUSTION_SHORT as is (spot-only). Add the two OBSERVE lines above to its scout, with frozen definitions.
2. If the operator wants to trade the post-peak dump on small / no-spot coins, that needs a **different entry**: after the peak is confirmed, not at the RSI cross during the squeeze. It would be a new pre-registered study (e.g. the CONFIRMED_TOP / SECOND_PEAK work), not a change to this sleeve.

## Files
- **Report:** `reports/NOSPOT_MCAP_2026-11-18.md`, `_PREREG.md`, `_cells.csv`, `_sizewl.csv`, `_2d.csv`, `_coverage.csv`, `_mcap_checks.csv`, `_replays.csv`, `_events.png`.
- **Scripts:** `scripts/nospot_mcap_common.py`, `_data_e.py`, `_build_e.py`, `_cands.py`, `_fetch.py`, `_features.py`, `_eval.py`, `_validate.py`, `_event.py`, `_replay_fetch.py`, `_replay.py`.
- **Data:** `reports/NOSPOT_MCAP_data/` (E/ holdout build, feat.pkl, eval.pkl, bapi/, metrics/, alpha.json, xinfo.json, k5m_live/, replay/).
