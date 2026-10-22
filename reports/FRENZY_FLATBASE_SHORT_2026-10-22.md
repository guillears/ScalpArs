# FRENZY flat-base short: short the "dead coin" pumps and hold for days?

Date: 2026-10-22. Research only. No bot code or config was changed, nothing was staged or committed.
The rules were frozen before any result: `reports/FRENZY_FLATBASE_SHORT_2026-10-22_PREREG.md` (the flatness threshold was added there after the episode walk and before any trade was simulated).

## Verdict: REJECT. This does not add a new sleeve, and it does not earn an observe-only line.

> **Correction (2026-10-22, after a hostile review, which is credited here; no other number in this report was changed).**
> - **The random-bar benchmark below looked into the future.** It drew evenly from bars that pass the $20M / 24 h volume gate over the whole 96 h. Whether a bar passes depends on when the coin's volume later dies. Episodes that go quiet early average +5.6 %; those that stay liquid average −3.9 %.
>   - Wherever this report says "random bars +2.0–2.3, p = 1.00", read it as "**an anticipating benchmark**".
>   - The fair, non-anticipating benchmark draws a delay first, then enters at the first eligible bar. It gives **+1.27 %/trade [+0.32, +2.19]** (2026-H2 −0.24, best 5 % removed −0.36), against the primary's **+0.94**. The primary still does not beat random timing, so the verdict stands.
> - **Funding is charged on the entry notional, not the live position value.** This mildly favours the short (it under-charges funding during squeezes), so the rejection is conservative.
> - **Exit follow-up:** see `FRENZY_FLATBASE_SHORT_2026-10-22_EXITS.md`. It covers shorter exits, a walk-forward test and the fair benchmark. Verdict: still no.

Your theory is half right, and the useful part is worth keeping:

- **Flatness really does separate the pumps.** Coins that were flat for 14 days before the frenzy fall back much more than coins that were already volatile. The same short makes about **+1 %/trade on flat coins and −0.5 %/trade on non-flat ones**. That gap is not luck: shuffling the flat / non-flat labels at random almost never produces it (p ≈ 0.01). The dose-response points the same way: the flatter the base, the better the short.
- **It still cannot be traded at a size that matters.**
  - The best cell's edge is about +1 %/trade, but the 95 % range touches zero for the pre-registered primary.
  - It loses in 2026-H2, which is the most recent period, the live block included.
  - It turns negative once the best 5 % of trades are removed, so a handful of big collapses carry it.
  - Shorting at a random moment in the same flat episodes does better than every entry rule. The original p = 1.00 came from an anticipating benchmark; the fair one gives +1.27 against the primary's +0.94 (see the correction).
  - Funding and squeezes take most of the edge.
  - At a 5 %-risk size, the drawdown is larger than the account.

The "massive wins" are real on single coins. IOST this week made +12 % to +38 % depending on the entry. But the year shows that for every IOST there is a GLM, SAHARA or ONT: flat coins that pumped again and charged 10–15 % in funding on the way to the stop.

## Before → after, per block

The primary is pre-registered: **S2**, the first close 15 % below the peak once the peak is at least 6 h old. It holds 7 days, takes profit back at the pre-pump price, uses a 25 % stop at 1×, and only trades flat coins.
- Results are % of position after 0.20 % costs and real funding.
- $ figures use the size where a 1st-percentile loss is 5 % of $3,200, which is about $548 notional per short.
- "Before" is today's book: FRENZY is long-only, so there are no shorts and the figure is $0.

| block | before | after: N · win rate · avg %/trade [95 % day range] | avg before funding | after in $ |
|---|---|---|---|---|
| ① live 10-14 → now (real episodes, 3 still open and marked to market) | $0 | 10 · 60 % · **+4.89** [−9.4, +12.8] | +5.45 | ≈ +$268 |
| ② master 10-02 → 10-13 (real episodes) | $0 | 30 · 63 % · **−2.98** [−9.0, +0.9] | −2.27 | ≈ −$489 |
| ③ year 2025-01 → 2026-10-01 | $0 | 1,605 · 67 % · **+0.94** [−0.07, +1.97] | +1.60 | **−$449** over the year with at most 5 open at once (663 taken). Max drawdown **−$3,273 (−102 % of the account)**. Funding paid: −$2,374 |

The best exploratory cell is **S4**: the first 4-hour candle that closes below the spike's average price (VWAP), with the same exit.
- Flat coins: ① 13 trades +0.88 · ② 36 trades +2.60 · ③ 2,207 trades **+1.16 [+0.42, +1.88]**.
- Sized the same way, it makes **+$144 over the year, with a −$2,452 drawdown (−77 %)** and −$3,394 of funding.
- It fails anyway: random bars beat it (family-wise p = 1.00 against the anticipating benchmark; see the correction), 2026-H2 is −0.32, and removing the best 5 % of trades gives −0.15.

## Flat vs non-flat (the key contrast)
Year block, every entry at the primary exit (7 d · take profit at the pre-pump price · 25 % stop · 1×):

| entry | FLAT coins (14-day range ≤ 44 %) | NON-flat coins | shuffled-label p (own / family-wise) |
|---|---|---|---|
| S1 flag drops | +0.37 [−0.48, +1.21] · 1,965 | −0.52 · 3,039 | 0.04 / 0.48 |
| **S2 exhaustion (primary)** | **+0.94 [−0.07, +1.97] · 1,605** | −0.64 · 2,742 | **0.01** / 0.12 |
| S3 24 h after the spike | +0.31 [−0.50, +1.14] · 2,043 | −0.49 · 3,228 | 0.05 / 0.56 |
| S4 4 h close below VWAP | +1.16 [+0.42, +1.88] · 2,207 | −0.40 · 3,401 | 0.00 / 0.02 |

**Dose-response** (S4, quintiles of the 14-day range, with edges frozen before the results):

| 14-day range | ≤ 35 % | 35–49 % | 49–69 % | 69–104 % | > 104 % |
|---|---|---|---|---|---|
| avg %/trade | **+1.60** [+0.78, +2.41] | +0.27 | −0.34 | −0.95 | −0.32 |

S2 has the same shape: +1.47 / +0.34 / −0.54 / −1.56 / −0.68. Your "dead coin" picture is the bottom quintile, and that is where the edge lives.

**Secondary flatness measures** (exploratory, S2, low tertile vs the rest):
- **Agree with the main measure:**
  - 30-day range: +0.93 vs −0.66.
  - Low 14-day volatility: +1.22 vs −0.79.
- **Do not agree:**
  - "Dead volume" (the last week's volume vs its 60-day median) does not separate the coins: −0.16 vs +0.02.
  - Coins already more than 62 % below their 90-day high are **worse** shorts: −1.26 vs +0.65.

So what matters is a quiet *price*, not a quiet volume. Coins that are already crushed tend to bounce.

## Is flatness the separator, or is it timing? Random bars win (anticipating benchmark; see the correction)
In the same flat episodes, a short at a random moment between 2 h and 96 h after the spike averages about **+2.0 to +2.3 %/trade**. Every entry rule averages +0.3 to +1.2. Every one of the 216 cells loses to random timing (family-wise p = 1.00). Note that this benchmark looked into the future: the fair one is +1.27 against the primary's +0.94.

A post-hoc diagnostic shorts flat coins at fixed delays after the spike. It is not pre-registered, so it can only refute, never support.
- At 2 h, 6 h, 12 h, 24 h, 48 h, 72 h and 96 h the averages are +1.03, +1.03, +0.38, +0.31, +1.27, +1.63 and +2.43 %/trade.
- Every delay is negative in 2026-H2.
- Almost every delay goes negative once the best 5 % of trades are removed.

The edge is "short a flat coin that pumped", not any particular entry. It is a thin edge carried by a fat tail of collapses, and it has not worked in the last three months.

## Funding: the hidden cost
- Over the full hold, flat-coin shorts pay on average −0.69 % (3 d), −0.98 % (7 d) and −1.35 % (14 d).
  - 5 % of 14-day holds pay more than **8 %**.
  - About half of the trades pay, and the other half receive a little.
- In the primary cell funding takes −0.66 of a +1.60 gross edge.
- The squeezes are where funding bites hardest. The 8 worst primary trades each paid 7–15 % in funding *before* the stop hit (GLM −15 %, SAHARA −11 %, ONT −11 %). A squeezed short pays twice.
- IOST shows how extreme it gets:
  - Funding at 10-21 20:00 was **−2.0 % for one settlement**.
  - From 21:00 it settled hourly at −0.6 % to −1.25 % per hour.
  - The 10-21 14:10 short paid **−8.2 % in about one day**. It still made +38 % because the coin fell 47 %.

## Tails and ruin
- **No stop, 14-day hold, 1×** (the "just hold it" version):
  - 87 of 1,509 flat S2 shorts (5.8 %) were **liquidated**, losing the whole margin plus funding. The worst was −134 %.
  - The 1st-percentile trade is −103 %.
  - Year average +0.51 %/trade, with a 95 % range of [−1.5, +2.6].
- **At 3× with no stop**, liquidation comes at about +32 % against. 372 trades lost at least 30 %.
- **Sizing at $3,200 with a 25 % stop:**
  - The 1st-percentile loss is −29 %, so each short can only be about $550 notional.
  - With 5 open at once, the year path loses money and draws down more than the whole account.
- **No-stop 14 d:** a 1st-percentile loss of −103 % means only $155 per short. That is too small to matter, and it still loses −$876 for the year.
- **One window:** 2025-04-20 (−118 % summed) and 2026-09-11 (−98 %) are each counted as a single window in the day bootstrap. 2025-10-11 does not dominate this study (it dominated the 10-20 study), because flat coins were few on that day.

## Pre-registered SHIP bars (primary, flat, year)

| bar | result | pass? |
|---|---|---|
| ≥ 300 trades on ≥ 30 days | 1,605 trades on 503 days | ✅ |
| day-bootstrap 95 % lower bound > 0 | −0.07 | ❌ |
| beats non-flat (shuffled labels) **and** random bars | non-flat: p = 0.01 ✅ · random bars: p = 1.00 against the anticipating benchmark; fair benchmark +1.27 vs +0.94 ❌ | ❌ |
| > 0 in 2025 / 2026-H1 / 2026-H2 | +0.62 / +2.42 / **−0.62** | ❌ |
| ≥ 8 live/master windows directionally consistent | 15 windows, mean of the window means −3.88, 8 of 15 positive | ❌ |
| × 0.6 haircut ≥ +0.50 %/trade | +0.56 | ✅ |
| every leave-one-month-out > 0 | lowest +0.72 | ✅ |
| best 5 % removed > 0 | −0.67 | ❌ |
| no pair and no day > 25 % of the profit | top pair 6 % (SIREN) · top day **26 %** (2026-01-25) | ❌ |

The observe-only bars were: CI lower bound > 0, or all three periods > 0 together with a non-flat p < 0.05. **Neither holds, so the verdict is REJECT.**

## Live / master episodes (blocks ① ②)
`reports/FRENZY_FLATBASE_SHORT_2026-10-22_live_master.csv` lists every walk episode since 10-01: 236 episodes, 114 of them real, meaning the bot logged FRENZY rows or made fills on them. Of the real ones, 51 are flat. For each episode it gives every entry's result at the primary exit and the 14-day no-stop result, with open trades marked to market.

Highlights:
- **IOST 10-20 14:10** (flat: 14-day range 25 %, real): S2 +16.8 · S3 +38.5 · S4 +11.5 (TP at the base on 10-22). The 14-day no-stop versions are still open at +11 to +38.
- **Flat winners:** WAL +28, OGN +27, CFX +13, CKB +18.
- **Flat coins that squeezed through the 25 % stop:** SAND 10-02, AIN, RLC 10-05, BAT 10-09, AVA 10-13, ALT 10-17, BMT. Some of these were back in profit by 14 days (AIN +24, RLC +7), the same pattern the 10-20 study found.

## Survivorship
- 14.2 % of year episodes, on 112 pairs, are on coins that no longer trade. All of them are in the data:
  - the 10-20 caches already held the settling pairs;
  - this study fetched the full history of all 164 delisted or settling USDT perps from data.binance.vision.
- Only 31 symbols are fully gone from Binance. They contribute 36 episodes, all old.
- Delisted flat coins are no better shorts (+0.75) than listed ones (+0.97). Survivorship is not hiding a winner.
- One data trap was found and fixed: a settled contract keeps publishing flat, zero-volume bars after delisting. These are now cut.

## Relation to the frozen 10-20 re-open gate
Pre-spike flatness is a new *conditioning variable* built from the same 5m klines. It is **not** the new information source that gate requires (open interest, funding at entry, order-book depth).

This study was therefore run under the gate's statistical bars on its own, and it fails them. **The 10-20 gate stays exactly as frozen.**

If the question is re-opened with one of those new sources, the frozen FLAT14 ≤ 44.4 % label should be carried as a pre-registered covariate:
- it is the one thing that separated the coins here;
- the threshold must not be re-fitted.

## Validation
- **Engine parity** (`scripts/study_flatbase_short_validate.py` against the real `services.frenzy.frenzy_walk`, 376 episodes / 4,429 bars):
  - the flag agrees on **95.3 %** of bars;
  - the spike is the same in 96.3 % of cases;
  - the VWAP is identical.
- **Real fills:** all **117 of 117** real FRENZY fills since 10-02 fall inside a walk-flagged window. The cohort uses the engine's signal bars.
- `scripts/validate_against_master.py`: **ALL CHECKS PASS**.

## What this could NOT test (blind spots)
- **Open interest, funding at entry, order-book depth:** not in the data. Note that funding *during* the hold was the main killer.
- **Liquidation:** modelled as isolated margin, losing the margin at about 1/L adverse. You run CROSS margin, where a squeeze can take more than one position's margin. Real results would be worse.
- **Stops:** stops and exits are checked on 5m bars. Gaps inside a bar beyond the stop, and borrow or availability limits on thin coins, are not modelled.
- **Interaction with the FRENZY longs:** one position per pair would block some longs. Not measured, because the study was rejected first.
- **Live block ①:** only 9 days long. 14-day holds there are still open and marked to market.
- **The random-bar null:** it does not apply the one-open-short-per-pair rule.
- **Funding gaps:** funding history is missing for 0–2 % of trades. Those were charged an assumed −0.10 %/8 h.

## Files
- **Scripts:**
  - `scripts/study_flatbase_short_data.py` (klines, delisted pairs, funding; public endpoints, weight-checked);
  - `_common.py`;
  - `_walk.py` (episodes and features);
  - `_validate.py` (parity);
  - `_sim.py` (entries, 54 exits, random-bar null);
  - `_report.py` (all tables);
  - `_posthoc.py` (entry-delay diagnostic).
- **CSVs:**
  - `_cells.csv`: every cell × cohort × block, with the nulls;
  - `_trades.csv`: every signal, with the net of all 54 exits;
  - `_episodes.csv`;
  - `_live_master.csv`;
  - `_posthoc.csv`;
  - `_parity.csv`.
- **Tables:** `_tables.md`, with the complete printed output.
- **Data:** `reports/FRENZY_FLATBASE_SHORT_2026-10-22_data/` (about 2 GB: klines, funding, signals, null). Not for git, and nothing has been staged.
