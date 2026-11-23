# 📡 Route B — what the recorder holds after 87 hours, and what it can do for each sleeve (2026-11-23)

**Short version.** The recorder has run almost without a break since 2026-11-19 23:23 UTC: 86.6 hours, 361 coins with the order book and every trade, and mark price, funding and liquidations for all 912 USDT coins. Since 11-21 22:02 UTC it also keeps the **full order book on the bot's own coins** (30 at a time, 40 so far) and **open interest and crowd-positioning ratios** on ~120–150 hot coins. Every bot fill opened since recording began has its order book recorded at entry and for the whole trade (18 of 18). **That is still far too little to decide anything.** There are 18 closed fills over four sleeves, and **no new fill since 2026-11-21 15:15 UTC** (a known 47-hour dry spell, audited as no bug in DL 630). The three EXH positions that were open in the 11-21 edition have closed (1 stop, 1 TP, 1 time exit). All fills fall in one BTC daily-trend sign. Everything below is **descriptive**. Nothing in it is a filter, an exit or a ship input. The new full book also answers the 11-21 data-quality gap: on fine-tick coins the 20-level book understated the depth within 0.5 % by **up to 11–13×** (EVAA, DODOX, PIEVERSE). Where the 20 levels did reach far enough, the two books agree.

Research only: no engine, config or recorder change. S3 was only read. Built with local research scripts and data (not in the repo). **Format (operator 11-23): summaries only — per sleeve, never per trade; coverage rows carry the 11-20 colour marks.**

### What changed since the 11-21 edition

| Item | 11-21 edition | Now (2026-11-23) |
|---|---|---|
| Data | 43.6 h, to 11-21 18:59 UTC | **86.6 h**, to 11-23 13:59 UTC (the last complete hourly summary). 11-22 is the first full day with no gap at all |
| Bot fills | 15 closed + 3 open | **18 closed, 0 open.** No new entry since 11-21 15:15 UTC. The 3 open EXH shorts closed: 1 stop (half of its loss was funding), 1 TP, 1 at the 24-hour max hold |
| Full order book (DL 611, recorder only) | not recorded; the 20-level book was the only depth | Since 11-21 22:02 UTC: a local full book (Binance diff-depth, resynced by sequence number) for up to **30 priority coins** (the bot's open, flagged and tracked coins). 40 distinct coins so far. Depth within 0.1 / 0.2 / 0.5 / 1 / 2 / 5 %, and the cost of a $1k / 3k / 6k / 10k / 25k market order, every second. Health check: 30 / 30 synced, 0 gaps, 0 resyncs, 100 % coverage of the coin-seconds the recorder wanted |
| Open interest + positioning (DL 611) | not recorded | OI every 60 s (210 coins seen) and the top-trader / global long-short ratios + taker buy/sell ratio every 5 min (209 coins), hot coins only. 98 % coverage, no 429 / 418 |
| Exits the fills ran | HT TP +5 net · EXH TP cap 10 % | **DL 608 / 609 (11-21 ~21:00 UTC): HT TP +6 net · EXH TP cap 7 %.** DL 619 (11-22): HT burst-rejection entry filter armed. Fills before ~21:00 on 11-21 ran the old exits. 2 EXH fills were open across the change |
| Related research since | – | DL 612: the **signal-level loader** (`scripts/route_b_signals/`, 332 journal signals by 11-22 01:00) · NEWSLEEVE 11-21: H1 / H2 / H5 / H6 **refuted on 3.4 years of history**, and the Binance `metrics` open-interest timestamp trap found · DL 626: the Route B peak study and an order-flow exit vs today's exit: **no change** · DL 627: DIST "squeeze fuel" not supported |
| Market | BTC 61.9k → 55.4k → 60.1k | BTC → **63.8k** on 11-23 (the window high). The BTC daily EMA50 crossed **below** the EMA200 on 11-20, so every fill sits in the same daily-trend sign |

---

## 1 · Data inventory

| Item | Value |
|---|---|
| Recording window | 2026-11-19 23:23 → 2026-11-23 13:59 UTC (the last closed hour; the recorder is live) = **86.6 h** |
| Outages | Two restarts during setup (11-19 23:32 / 23:36, ~1 s each). One restart for the DL 598 deploy (11-20 02:59, 2 s) and one for the DL 611 deploy (11-21 22:01, 3.5 s). Four websocket drops on 11-20 (14:43 ×3, 17:45), **13 s in total**. None since. Gap seconds on 11-22 and 11-23: 0 on every stream, the full book included. Health now: ✅ (running since 11-21 22:01, 0 failed uploads, 29.6 GB disk free) |
| Order book (top 20 levels every 0.5 s) + every trade | **361 coins**, 536,564 coin-minutes, 51.1 M book snapshots, 130.2 M trades (11-19: 84 coins · 11-20: 263 · 11-21: 289 · 11-22: 172 · 11-23 to 14:00: 162, ≤ 120 at any one time) |
| **Full order book** (new, DL 611) | **40 coins** (≤ 30 at once), 71,829 coin-minutes since 11-21 22:02. The bot wanted up to 38 priority coins at the start, so the 30-coin cap left 1–8 out at times on 11-21 / 11-22. None were left out after 11-23 11:26 |
| **Open interest / positioning ratios** (new, DL 611) | OI on **210 coins** (220,853 coin-minutes) · ratios on **209 coins**, since 11-21 22:02 (1.7 days) |
| Mark / index / funding every 1 s + liquidations | 912 USDT coins (coin-margined contracts are excluded: their size is in contracts, not $) |
| Liquidations (a SAMPLE: Binance sends at most one per coin per second) | longs force-sold **$1,016 M**, shorts force-bought **$401 M** (USDT coins only). By day, longs / shorts: 11-20 $689 M / $58 M · 11-21 $264 M / $165 M · 11-22 $31 M / $66 M · 11-23 (to 14:00) $24 M / $111 M. The long side came mostly in the two BTC dumps (11-20 and 11-21 early). The short side leads during the 11-22 / 11-23 rally |
| Hourly per-minute summaries (built on the server) | **87 of 87 hours**, 0 failures, ≈ 1.3 MB each (now with the full-book, OI and ratio columns) |
| Raw data on S3 | **23.3 GB** compressed: book 13.8 · mark 5.4 · trades 3.3 · full book 0.32 + 30-s snapshots 0.36 · liquidations 0.009 · OI / ratios 0.004 GB (≈ 6.4 GB/day; health check: ≈ $0.53/month at today's size) |
| Bot's coin list (open / flagged / tracked) | Published since 11-20 02:59 (**981 lists**). The last one (11-23 13:55): 0 open positions, 9 flagged, 29 tracked; sleeves on: EXHAUSTION, DISTRIBUTION, FLATBASE, WILLY_HT, WILLY (all others off), as expected |
| Market context | BTC 61.9k → **low 55.4k** (11-21) → 60.2k → **63.8k** (11-23, the high of the window). Daily EMA50 < EMA200 from 11-20 on. **One daily-trend sign for every fill**: a dump, a recovery, then a rally with no bot fills in it |

### Coverage of the bot's real fills

The fills come from every November export in `exports_archive/` plus the newest dashboard export (`scalpars_orders_paper_2026-11-22_20-32-36.csv`, also archived). They are deduplicated on (opened_at, pair, direction) and CLOSED only. There are no probes in this window. No position is open (the bot's last published list). The newest export already contains every close, including ORCA's at 11-22 15:15.

Colour marks (agreed 11-20, data quality only — never a finding): 🟢 ≥ 90 % of the sleeve's fills with the book recorded at entry ·
🟡 50–90 % or the recorder degraded · 🔴 < 50 % or the recorder down · ⚪ no fills.

| | Sleeve | Fills since recording began | Book recorded at entry | Recorded through exit | Note |
|---|---|---|---|---|---|
| 🟢 | FRENZY_WILLY_HT | 8 closed (4 W · 4 L) | 8 / 8 | 8 / 8 | all on 11-20 / 11-21, under the old TP +5 |
| ⚪ | FRENZY_WILLY (plain) | 0 | – | – | |
| 🟢 | EXHAUSTION_SHORT | 6 closed (3 W · 3 L) | 6 / 6 | 6 / 6 | the full book + OI cover only the end of the last holds |
| 🟢 | DISTRIBUTION_SHORT | 3 closed (1 W · 2 L) | 3 / 3 | 3 / 3 | |
| 🟢 | FLATBASE_SHORT | 1 closed (L) | 1 / 1 | 1 / 1 | a second fill opened before recording began (only its last 17 % recorded) is left out |
| 🟢 | **Total** | **18 closed** | **18 / 18** | **18 / 18** | **Full book at entry: 0 / 18** (every entry predates 11-21 22:02) |
| 🟢 | Hourly summaries | 87 of 87 hours | | | 0 failures |

**Data-quality gap: now answered for the bot's coins.** The 20-level book still only reaches a short way on fine-tick coins. Across all 361 recorded coins, the median reach of the 20 levels is **< 0.5 % on 43 %**, and on **22 %** the cost of a $10k market order cannot be computed in most minutes. On the 40 full-book coins, the full book reaches past 0.5 % on **every** coin, and the $10k cost is computable in **100 %** of minutes. Like for like (the same minutes), the two books agree wherever the 20 levels reach: median $10k cost 14.7 vs 14.8 bps over 54,453 coin-minutes, and the $ within 0.5 % is the same (ratio 1.0) on 23 of 40 coins; on 12 coins the 20 levels showed 1.5× or more too little. Where the 20 levels fall short, they understated the depth badly. Among the fill coins (full-book numbers are medians since 11-21 22:02, **not at those fills' entries**):

| Coin | Reach of the 20 levels | $10k order not computable (20 levels) | Full book: reach | Full book: $ within 0.5 % bid / ask | Top-20's 0.5 % depth vs the full book (same minutes) | Full book: cost of a $6k order, bps (buy / sell) |
|---|---|---|---|---|---|---|
| EVAA | 0.05 % | 97 % of minutes | ≥ 4.6 % | $36k / $34k | **11× too low** ($3.3k) | 8.5 / 8.4 |
| BLESS | 0.17 % | 78 % | ≥ 7.6 % | $27k / $30k | **3.3× too low** | 8.2 / 8.2 |
| PHAROS | 0.37 % | 90 % | whole side | $9k / $9k | 1.4× too low | 21.4 / 21.1 |
| KERNEL, ENSO | 0.42–0.46 % | 0 % | whole side | $36–48k / $38–50k | 1.0–1.2× | 7.2–9.9 |
| *(non-fill bot coins)* DODOX · PIEVERSE | – | – | – | $30k · $50k | **12.6× · 12.1× too low** | 10.2 · 6.0 |

So the 11-21 edition's depths marked * on those coins were **lower bounds that understated the real book several-fold**. A cost marked "n/a" was not "too thin to fill". The two coins that really are thin at our size are PHAROS and NAORIS (≈ 21 and 17–20 bps for $6k on the full book). This is fixed **only for the ≤ 30 priority coins**. All other coins, and every entry before 11-21 22:02, still have only the 20-level book.

---

## 2 · Per-sleeve first look (DESCRIPTIVE · N tiny · one regime)

How to read the tables (every cell is a sleeve aggregate):
- "At entry" = the last full minute before the fill, or the 5 / 15 minutes before it.
- "Against us" = the side a move against the trade would hit. For a short, that means short liquidations, taker buying and asks above the price. For a long, it is the mirror (long liquidations, taker selling — the sell share — and bids below).
- Funding is shown per settlement. Negative funding means shorts pay.
- Result % = the export's P&L % on notional, funding included.

### 2a · State at entry, winners and losers side by side

No fill has the full book, open interest or the positioning ratios at entry (all entries predate 11-21 22:02). Each cell is the
**median** of the winners / the losers of that sleeve.

| Sleeve | Fills (W · L) | Avg result % | Spread bps | $ within 0.5 % against us | Cost to enter, bps | Taker share against us, 15 min | 15-min traded $ vs usual | Liquidations against us, 15 min $ | Funding % | Move 15 min % |
|---|---|---|---|---|---|---|---|---|---|---|
| HT | 8 (4 · 4) | +0.4 | 1.5 / 4.4 | 4.9k* / 16.3k* | 4.8² / 8.6 | 0.47 / 0.49 | 2× / 11× | 0.8k / 2.0k | +0.03 / +0.01 | +1.0 / +1.9 |
| EXH | 6 (3 · 3) | −0.6 | 6.4 / 6.7 | 41.5k / 30.3k | 9.6 / 11.1 | 0.48 / 0.47 | 24× / 14× | 0 / 0 | −1.04 / −1.31 | −1.9 / −3.0 |
| DIST | 3 (1 · 2) | −0.7 | 4.9 / 6.7 | 43.3k / 13.3k* | 8.9 / 13.6² | 0.50 / 0.48 | 55× / 43× | 32.6k / 62.9k | −1.49 / −0.77 | +1.2 / −13.0 |
| FB | 1 (0 · 1) | −11.7 | – / 3.0 | – / 37.4k | – / 7.7 | – / 0.52 | – / 5× | – / 0 | – / −0.19 | – / −0.3 |

Each cell = winners / losers. With 1–4 fills per side these medians are anecdotes, not evidence.

\* The 20 levels reach less than 0.5 % on most of these fills, so the depth is a lower bound (the full book showed the 20 levels
understating such coins 3–11×). ² The $6k cost is measurable on only part of the side (HT winners 2 / 4, DIST losers 1 / 2): the
others were too thin for 20 levels. "Usual" = the coin's median 15-min traded $ over the whole recording (it moves as the window grows).

**What it shows (descriptive, not a finding):**
- **No entry column separates winners from losers in any sleeve.** That is what 1–4 fills per side should give.
- **Crowded shorts (negative funding) at entry** sit on both sides: EXH 3 / 3 winners and 3 / 3 losers, DIST 1 / 1 and 1 / 2. This
  matches DL 602 and DL 627 ("squeeze fuel" not supported).
- **Cost at our size, where measurable, is 3–16 bps per side** (0.1–0.3 % for a round trip with fees). The losers lost 4–12 %, so
  cost is not the source of the losses.
- **Funding paid while held is a real cost on the long EXH holds:** median $83, at most $307 (5 % of notional, half of that fill's
  loss); FB $108 (1.6 %); HT and DIST ≈ 0. DL 616 / 622 already tested funding rules on every sleeve and kept today's exits; this is
  more data for their re-open trigger, not a reason to change anything.
- **DIST "already crashed" question stays open:** one DIST loser entered after a > 20 % 15-min fall with a long-liquidation cascade.
  DL 612's after-crash study found no change on history; the scan tests it as a pre-registered line (S2-a, HOLDOUT-only).

### 2b · During the trade (summary)

Counts per sleeve, winners / losers. "Against us" after the peak = taker flow tilted to the side that hurts the trade.

| Sleeve | Taker flow against us after the peak | Liquidation burst above the coin's p99 against us | Funding paid while held |
|---|---|---|---|
| HT | W 0 of 4 · L 4 of 4 | W 0 of 4 · L 1 of 4 | ≈ 0 |
| EXH | W 2 of 3 · L 2 of 3 | W 1 of 3 · L 1 of 3 | median $83, max $307 |
| DIST | W 0 of 1 · L 0 of 2 | W 1 of 1 · L 0 of 2 | ≈ 0 |
| FB | L 1 of 1 | L 0 of 1 | $108 |

**What it shows (descriptive):**
- **Taker flow turning against us after the peak** marks every HT loser but also 2 of 3 EXH winners, so on its own it is a false
  alarm as often as a warning. DL 626 tested order-flow exits on shadow paths and found none that beats today's exit.
- **Liquidation bursts come with the sharp stops, not before them**, and a winner survived a burst above p99, so a burst alone is
  not an exit signal.
- **Two loss shapes:** a sharp leg (liquidations + book thinning, seconds to minutes) and a slow drift (no burst; the short pays
  funding all along). Only the first is something a book / tape exit could catch.
- **Open interest** exists on only 2 partial holds so far, too few to say anything.
- An aggregate winners-vs-losers event-study chart comes with the first scan that has enough fills; per-trade charts are not shown.

---

## 3 · What Route B can add, sleeve by sleeve

**Fill rate in this window.** Over all 3.6 days: HT ≈ 2.2 / day, EXH ≈ 1.7, DIST ≈ 0.8, FB ≈ 0.3. That is lower than the 11-21 edition's rates (≈ 5 / 4 / 2 / 0.7 per day over the ~36 h with fills), because of the 47-hour dry spell. The short sleeves still cluster on the same coins and days (3 coins carry 8 of the 18 fills), so in window units (the locked rule) they accumulate more slowly still.

**The locked gates on fills alone** need ≥ 15 blocked fills and ≥ 8 windows. A W/L read on 30 fills of a ~60 % WR sleeve has ~12 losers, which detects only very large effects. **The realistic route is SIGNAL-level evidence, and its loader now exists** (DL 612, `scripts/route_b_signals/`). Every journal signal on a recorded coin (taken, skipped or blocked) is priced on its own 1-minute path under today's exit, with book / tape state taken strictly before the signal: 332 signals, 236 possibilities, 137 windows by 11-22 01:00. That multiplies N by 3–10× and keeps the master-over-backtest rule intact. There is no backtest equivalent: book data cannot be downloaded for the past, so Route B evidence is forward-only. The new full-book, OI and ratio columns exist only from 11-21 22:02. They are not yet in the signal loader's state columns, and they need adding before the scan.

| Sleeve | Open questions Route B can answer that klines could not | Already visible (N tiny, descriptive only) | Earliest valid read |
|---|---|---|---|
| **FLATBASE_SHORT** | ① **Squeeze build at entry**: ask-wall thinning, short-liquidation pressure, taker-buy share, OI build-up and the long/short ratios in the 4 h before the FB_CONFIRM bar · ② **real funding paid while held** against the DL 601 funding stamp · ③ **real stop cost** of a −10 % market stop on a thin coin, now measurable on the full book · ④ capacity: full-book $ at 0.1–5 % against the planned size | 1 recorded fill (a loser). Nothing | Fills: ~0.3 / day → ~3 months for 30 (≈ Feb). **Signal route at the 12-10 scan** |
| **EXHAUSTION_SHORT** | ① Entry: taker-flow exhaustion, liquidation state and **OI direction** at the signal minute · ② **exit**: the DL 626 pre-registered CAPITULATION_LIQ re-read, judged two-sided on the live-stopped cohort · ③ the **slow-drift + funding** loss: already tested (DL 616 / 622, keep) | 6 fills, 3 W · 3 L. Funding negative at entry on all 6. Drift losers paid the most funding | Fills: 30 at ~1.7 / day ≈ **12-07** (pushed out by the dry spell); window gate ≈ late Dec. **Signal route at the 12-10 scan** |
| **DISTRIBUTION_SHORT** | ① **"Already crashed" entries**: 15-min fall + long-liquidation cascade at entry · ② the post-entry short-liquidation-burst exit (DL 602; DL 626 found every signal cut the winner) · ③ real entry cost vs DL 599's fast-entry lag | 3 fills, 1 W · 2 L. One loser entered right after a crash; the winner survived a burst | Fills: 30 at ~0.8 / day ≈ **late Dec**. **Signal route at the 12-10 scan** |
| **FRENZY_WILLY_HT** | ① **Give-back exit**: after-peak taker reversal, bid thinning, long-liquidation burst (DL 626: the best one is "close at ½ TP" in disguise, −0.73 %/fill on the master) · ② entry: book depth / flow at the signal, with the full book for flagged coins · ③ real exit slip on −4 % stops · ④ the **burst-rejection filter** (DL 619): Route B can price what it refuses | 8 fills (4 W · 4 L), all under the old TP +5. After-peak taker flow against us on 4 / 4 losers and 0 / 4 winners | Fills under today's TP +6: **0 so far.** 30 at ~2.2 / day ≈ **12-03**. **Shadow paths at the 12-10 scan** |
| **All sleeves: cost / capacity** | The real book cost at entry and stop for every fill, against the paper drift and the model slip (`entry_exh_model_slip_pct`, `entry_dist_model_slip_pct`); capacity per signal (CLAUDE.md capacity rule) | 7–16 bps per side where measurable. **Now measurable on fine-tick priority coins too**: full-book $6k cost 6–23 bps (median of the 40 coins ≈ 12) | Usable now for the priority coins; non-priority coins and every pre-11-21 22:02 entry stay on the 20 levels |

**Pre-register before the data arrives: done.** `reports/ROUTE_B_WEEKLY_SCAN_PREREG.md`, frozen 11-23 15:01 UTC, plus Addendum 1
(implementation pins and stated deviations, DL 631) and Addendum 2 (verdict colours, summaries only, DL 632). The signal loader
now carries the full-book, OI and ratio columns (build items 1–7, DL 631).

---

## 4 · New-sleeve ideas the data can test (counts only, no outcomes looked at)

I counted how often each candidate trigger fires. Counting rules: alts only (BTC / ETH excluded), coins with the book recorded, **one possibility per coin per 60 min** (24 h for funding), and distinct 15-min windows next to it. Rates are per day over the full 3.61 days (the OI rows: over the 1.66 days with OI). They fell for H1 and H6 because 11-22 / 11-23 were quieter than the dump days. **I deliberately did not compute the forward moves.** DL 612's loader stores H1–H6 outcomes but never prints them until the PREREG is frozen. With 3.6 days in one daily-trend sign, any return would be a story rather than evidence (operator rule: no performance claims from a few days).

**History since the 11-21 edition:** NEWSLEEVE 11-21 tested kline / metrics proxies of H1, H2, H5 and H6 on 3.4 years of Binance history. **All were REFUTED on HOLDOUT** (−0.23 to −1.45 % avg net). The one apparent pass (H2) was the Binance `metrics` open-interest timestamp artefact. So H1 / H2 / H5 / H6 now have a **low prior**. Route B can only add a *book- or OI-conditioned* variant that history could not see. H3 and H4 need the book and were never testable on history.

| # | Hypothesis | Trigger as counted | Possibilities / day | 15-min windows / day | The test it needs | Earliest read |
|---|---|---|---|---|---|---|
| H1 | **Long-liquidation cascade bounce** (alt longs flushed, then a bounce) | 5-min long liquidations ≥ the coin's 99th percentile and ≥ $50k | 24 (was 37) | 10 | Long at the next minute's ask, fixed TP/SL grid frozen on DESIGN, cost from the recorded book; random-minute same-coin control; **clustered by market moment** (count DAYS / dump episodes, not events). History proxy REFUTED: only a book / OI-conditioned variant is new | 12-10 scan, but it needs several separate dump episodes; realistically 4–6 weeks |
| H2 | **Short-squeeze fade** (shorts flushed, then a fade); doubles as an EXH / DIST timing filter | 5-min short liquidations ≥ the coin's 99th percentile and ≥ $50k | 16 (was 14) | 10 | Same as H1, short side; also checked against EXH / DIST signals on the same coins. History proxy REFUTED (after the timestamp fix) | 12-10 (as an EXH / DIST timing question), 4–6 weeks as a sleeve |
| H3 | **Thin-book imbalance breakout** | 0.5 % imbalance ≥ +0.5 (or ≤ −0.5) for 3 minutes, book < $40k | 35 + 31 | 29 + 27 | Follow the heavy side, 5–15 min. **Low prior**: the 11-20 / 11-22 first views find imbalance → next minutes ρ ≈ +0.01 to +0.025, ~1–3 bps against 10–30 bps cost. On fine-tick coins the imbalance should come from the **full book** (the 20 levels mis-state depth there) | 12-10 |
| H4 | **Book vacuum** (one side < 35 % of its usual depth) | ask- or bid-side 0.5 % depth < 0.35 × the coin median | 64 (ask) / 81 (bid) | 42 / 48 | Does price move toward the vacuum? Also tested as an **exit warning** for open positions (it led 1 of 4 sharp stops). Use the full book where it exists | 12-10 |
| H5 | **Taker-climax fade** | 15-min traded $ ≥ 5× usual and ≥ 60 % one-sided | 97 (buy) / 159 (sell) | 55 / 64 | Fade the climax. The 11-22 first view: taker-buy share → 15-min underperformance (ρ −0.031, 24 / 24 hours), ~2 bps against 10–30 bps cost. History proxy REFUTED (H5B −0.32 %, H5S −0.34 %) | 12-10 |
| H6 | **Funding extremes** | funding ≤ −0.5 % per settlement (crowded shorts) · ≥ +0.1 % (crowded longs) | 11 · 37 (one per coin per day) | 9 · 21 | Squeeze-long / crowded-long fade, *net of the funding received or paid* (1 s funding recorded); also as an avoid-filter for the short sleeves (section 2a: on both W and L; DL 627 not supported). History REFUTED (H6L −0.36 %, H6S −0.23 %) | 12-10, in DAY units |
| H7 *(new data)* | **Open-interest flush / build** (the cleaner cascade signal the liquidation sample cannot give) | OI −3 % in 5 min with price −2 % (flush, down) · OI −3 % with price +2 % (short covering) · OI +5 % with price +2 % (build) | 4.8 · 2.4 · 4.2 | same (every possibility its own window) | As H1 / H2 but on OI; read with the timestamp care NEWSLEEVE found (the recorder's OI is its own live sweep, so there is no 5-min label shift). Only 1.7 days and ~120–150 hot coins so far | 12-10 at the earliest; realistically with H1 / H2 |

All of these go to the 12-10 scan as pre-registered lines. Any survivor faces the locked promotion gates, theory first, the in-sample haircut and the capacity read, then operator OK. A live trigger is phase 3 (DL 597).

---

## 5 · The weekly research scan

- **Status:** **pre-registered and built (DL 631 / 632); the weekly task is not created yet** (needs the operator's OK). The daily report's countdown still targets **≈ 2026-12-10** (day 21 since 11-19; at 11-22 it read "4 of 21 days"), which matches DL 597 and memory ("set up the WEEKLY research scan at ~3 weeks of data").
- **To prepare before 12-10** (progress since 11-21 in **bold**; the rest needs the operator's OK):
  1. **The PREREG file: done** (frozen 11-23 15:01 UTC + Addenda 1–2). Open before 12-10: the H6 control-exclusion choice (operator).
  2. A signal-level dataset: **done (DL 612, `scripts/route_b_signals/build.py`, incremental).** Its last report is `ROUTE_B_SIGNALS_2026-11-22.md` (data to 11-22 01:00), so it should be run daily. **The DL 611 columns are now in its state block, with the no-look-ahead tests (DL 631).**
  3. A decision on the deeper book: **done for the bot's coins (DL 611, ≤ 30 priority coins).** It works (0 gaps, 0 resyncs) and confirms the 20-level book where both reach. Open point: the 30-coin cap left 1–8 wanted coins out on 11-21 / 11-22 (none since 11-23 11:26). If the bot's list grows again, raising the cap is a recorder change, so it is the operator's call.
  4. **A regime caveat in every read**: every fill so far is in one BTC daily-trend sign (EMA50 < EMA200 since 11-20). The scan should report per BTC daily-trend sign (locked live-mode lesson) and say so if only one sign has been seen.
  5. **Create the scheduled weekly task** (Thursdays from 12-10 — 12-10 is a Thursday, PREREG Addendum 1), running the pre-registered scan and writing one organised report (summaries only, verdict colours).

## Blind spots

- **N = 18 closed fills** across 4 sleeves; per sleeve ≤ 8. Every observation above is anecdotal. **No new fill for the last 47 hours** of the window.
- **One BTC daily-trend sign**: 3.6 days, one −11 % / +15 % swing. No fill was taken during the 11-22 / 11-23 rally.
- **Two exit eras.** The HT fills ran TP +5 (today +6). The EXH fills ran TP 10 (today 7), and two of them crossed the change. Per-fill results are as booked; DL 626 holds the today's-TP re-pricing.
- **The full book, OI and ratios start at 11-21 22:02**: 0 of 18 entries and only 2 partial trades have them. They cover only the ≤ 30 priority coins (full book) and ~120–150 hot coins (OI / ratios).
- **20-level book depth** is still a lower bound on fine-tick coins outside the priority set, and for every pre-11-21 22:02 minute. On priority coins the full book showed it can be off 3–13×.
- **Liquidations are a 1-per-second sample.**
- **Paper fills**: the entry/exit "slippage" in the export is price drift in the paper model, not a real book walk.
- **Two fills are not covered**: one opened before recording began, one on a coin the recorder never held.
- **The per-minute summaries have no sub-minute order**: "before vs same minute" is at 1-minute resolution. The raw 0.5 s / 1 s files and the 30 s full-book snapshots on S3 can refine it.
- **The H-counts and the "usual" baselines move with the window**: per-day rates and the "vs coin's usual" column are averages over a dump day and quieter days, so they will keep moving until several weeks are in.
