# SECOND PEAK or TIME AFTER THE FLAG? Shorting Frenzy coins (2026-11-18)

**Pre-registered:** `SECOND_PEAK_2026-11-18_PREREG.md`, frozen 12:15 UTC, before any result (descriptive or tradable).
Research only. Nothing in the bot changed. Nothing was committed.

Files: `_cells.csv` (42 cells, every metric) · `_trades.csv` (173 k cell-trades with entry features) · `_replay.csv` · charts `_event.png`, `_hours.png`, `_replay_<COIN>.png` · scripts `scripts/second_peak_{build,eval,wl,charts,fetch,replay}.py`.

## Question (operator)
"BREVUSDT had the same behaviour as the others: after the second peak, a huge dump. Same as HUMA, BLESS and all other frenzys. See if the second peak is the pattern, or X time after its frenzy ON."

So: on Frenzy coins, is the tradable short trigger (a) the **second peak**, or (b) a **fixed time X after the flag turns ON**?

## Plain answer
**Neither is a tradable edge on its own.**
- In hindsight, the second peak is NOT where the big dump usually starts. In most episodes the **first** peak is the top (58–73 %, depending on how a "peak" is measured). The second peak is the top in only 8–19 % of episodes.
- When an episode does have a second peak, the dump after it is about the same size as the dump after the first peak (median −16 % in 24 h after either).
- The main top has **no stable time**. 30 % of tops print within 30 minutes of the flag. The median is 3.4 h. The other tops spread evenly from 4 h to 24 h. There is no "X".

**What the tests say:**
- **Time-after-flag (T) is dead.** All 18 T cells lose or sit at zero (best: +0.10 %/trade, T 12 h with a 5 % TP). None of them beats a random entry time in the same window.
- **The second-peak trigger (P2) is the more interesting finding.** It beats random timing very strongly in every one of the 24 cells (family-wise p < 0.001).
  - Why: shorting a Frenzy coin at a random moment after a pullback loses about −0.9 %/trade, because many of those coins keep pumping.
  - Waiting for the second top cuts that loss to about 0.
  - But "about 0" is not a profit. The best P2 cell makes **+0.03 %/trade**. It was positive from 2023-H2 to 2025-H2 and negative in 2026-H1. It fails 4 of the 7 pass items.
  - It adds nothing on top of EXHAUSTION_SHORT. The P2 trades that EXH does not already take average −0.04 %.
- **Your coins did fit the pattern this week.** The plain "new high, then a 15m close 3 % under it" cell (P2 · P8 · R0 · K) fired 7 times (on 6 of the 8 coins) and won 5. That includes BREV: short at 01:45, +4.8 % in 5 minutes.
  - But you picked these coins because they dumped.
  - Over 3 years the same rule wins 47 % of 3,573 trades and averages −0.21 %.
  - The coins where the second top was not the top (BLESS 11-16 before its +102 % run, PIPPIN today) are the ones that pay for the wins.

**Verdict: NO** for both families. The cheap follow-up is in "Next step" below.

## Method (short)
**Universe.**
- Every **verified** Frenzy episode: the engine `frenzy_walk` rule re-run bar by bar (spike = 6-bar return ≥ 5 % ∧ 1h volume ≥ 20× normal ∧ ≥ $2 M; verified at the spike; flag = verified ∧ ≤ 96 h).
- **All verified episodes are kept.** There is no later-shortlist selection, which removes the EXHAUSTION_EXPAND §0 survivorship bias.
- A signal trades only if the short-list gate (24 h move ≥ 15 % ∧ 24 h volume ≥ $20 M; blacklist out) is true **at the signal bar**.
- Coverage: 658 pairs; 11,657 episodes from 2023-07 to 2026-11-17 (9,723 of them pass the gate at some point).

**Families.** 42 cells.
- **P2** (24 cells) = pullback ≥ P % from the running peak (P 8 / 15), then a return to the peak (R 0 = equal or new high; R −3 = within 3 %).
  - Trigger: K = first 15m close ≥ 3 % under the second high, or RSI = 15m RSI14 crosses 70 → ≤ 70 on that leg.
  - One trade per episode.
- **T** (18 cells) = short at flag + 2 / 4 / 6 / 8 / 12 / 24 h.
- **Exits:**
  - EA = the EXHAUSTION exit: TP min(3×ATR15, 5 %) maker, SL min(1.5×ATR15, 5 %), 24 h.
  - S5 / S10 = TP 5 / 10 % maker, stop at the second high (T: the running high) + 0.5 %. The trade is skipped if that stop is > 15 % away.

**Realism.**
- Closed 15m bars only. Entry at the next 5m open.
- Costs: taker 0.05 %, maker 0.02 %, slippage 0.10 % per taker leg (0.05 / 0.20 also shown).
- A maker TP fills only if price trades through it by 1 tick. Stops are gap-aware, and a stop wins a same-bar tie with the TP.
- Real funding at every settlement (hourly when Binance switched the coin). Missing pair-months were charged 0.05 %/8 h; none occurred.
- One position per pair.

**Nulls.**
- Random timing: 40 random entry bars per trade, in the same episode, flag ∧ gate on, drawn from the cell's eligible start (P2: the pullback-confirm bar; T: the spike). Same exit, and the S stops use the trade's own stop distance. 2,000 bootstrap means.
- Family-wise: a max-z statistic over all 42 cells.

**Implementation fix disclosed.**
- The first run counted a wide pump bar's own low as the "pullback" from that same bar's high. A wide bar also cancelled the leg that it had just started.
- That is not the prereg's intent, which is a zigzag-style pullback after the peak. It was found on the BREV replay and fixed before the final read.
- The buggy grid is kept in `_data/cells_v0_samebar_bug.csv`. The conclusion is the same in both runs (best P2 +0.12 there, +0.03 now; nothing passes).

## Step 0 — what the peaks look like (HINDSIGHT, not tradable)
Peaks are 15m zigzag highs inside the flag window, confirmed by an X % reversal. Universe: 9,723 gate-eligible episodes.

| | zigzag 8 % | zigzag 15 % |
|---|---|---|
| episodes with 1 / 2 / 3 / 4+ peaks | 31 % / 25 % / 15 % / 28 % | 61 % / 14 % / 5 % / 6 % (15 % have no 15 % swing) |
| **episode max = peak 1 / peak 2 / peak 3+** | **58 % / 19 % / 21 %** | **73 % / 8 % / 5 %** |
| biggest 24 h dump starts after peak 1 / peak 2 | 61 % / 20 % | 74 % / 7 % |
| *only episodes with ≥ 2 peaks:* max = peak 1 / peak 2 | 41 % / 28 % (n = 6,478) | 49 % / 32 % (n = 2,332) |
| *only ≥ 2 peaks:* peak 2 above peak 1 · within 3 % of it | 48 % · 66 % | 45 % · 57 % |
| *only ≥ 2 peaks:* median drawdown 24 h after peak 1 vs peak 2 | −16.6 % vs −15.9 % | −24.9 % vs −23.2 % |
| hours flag → peak 1 (median, IQR) | 0.3 h (−0.1 … 1.9) | 1.2 h (0.0 … 6.9) |
| hours flag → peak 2 (median, IQR) | 6.2 h (2.4 … 12.9) | 12.6 h (5.6 … 19.9) |

(Drawdowns after a confirmed zigzag peak are partly true by definition: a 15 % peak needs a 15 % drop. That is why the 15 % column shows bigger dumps.)

**Hours from the flag to the episode's highest high:**
- p10 0.0 · p25 0.2 · **median 3.4** · p75 14.1 · p90 22.2.
- 30 % fall within 30 min, 13 % at 0.5–2 h, 9 % at 2–4 h, 10 % at 4–8 h, 9 % at 8–12 h, 24 % at 12–24 h, and 5 % after 24 h.
- The 12–24 h mass is partly the window edge: 78 % of flags last only 24 h, so a coin still rising is cut off there.
- Per half-year the median is stable at 2.9–4.1 h, but the spread is huge. **There is no single X.**
- Only 22 % of episodes ever reach the long-side "FRENZY ON" state. Measured from that state, the top is a median 3.0 h later (IQR 0 … 14.6 h).

**Stable across time:** "max = peak 2" (8 % zigzag) is 15–20 % in every half-year from 2023-H2 to 2026-H2. "Biggest dump after peak 1" is 52–71 %.

![event study](SECOND_PEAK_2026-11-18_event.png)
*Median path (band = IQR) of the 15m close vs the peak's high, −24 h → +72 h. When a peak turns out to be the top, price is −15 % three days later, whether it was peak 1 or peak 2. When it is not the top, the loss shrinks to −6 % (peak 1) or −10 % (peak 2), and the IQR spans well above 0. At the moment of the peak, you cannot know which line you are on.*

![hours to top](SECOND_PEAK_2026-11-18_hours.png)

## Step 1–2 — tradable results (42 cells; the best and representative ones below; all in `_cells.csv`)
Net %/trade at 0.10 % slippage. Pass bar ①–⑦ as frozen. Cell names use "·" for "|".

| cell | N | /day | WR | BE WR | avg net % | 95 % CI (day-clustered) | ① every half > 0 | ② best-5 % out | ③ @0.20 | ④ vs random (p, random avg) | ⑤ family-wise p | ⑥ Kelly at Wilson floor | ⑦ holdouts A · B | passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2·P15·R0·RSI·EA (best P2) | 947 | 0.77 | 52 % | 52 % | **+0.03** | −0.30 … +0.35 | ✗ | ✗ −0.23 | ✗ −0.12 | ✓ 0.000 (−0.80) | ✓ 0.000 | ✗ −0.070 | ✓ +0.74 · +0.04 | 3/7 |
| P2·P15·R−3·RSI·EA | 1,101 | 0.89 | 51 % | 51 % | −0.04 | −0.32 … +0.25 | ✗ | ✗ −0.30 | ✗ −0.19 | ✓ 0.000 (−0.80) | ✓ 0.000 | ✗ −0.080 | ✓ +0.39 · +0.08 | 3/7 |
| P2·P8·R0·RSI·S10 | 2,057 | 1.67 | 41 % | 41 % | +0.02 | −0.33 … +0.36 | ✗ | ✗ −0.51 | ✗ −0.15 | ✓ 0.000 (−1.13) | ✓ 0.000 | ✗ −0.045 | ✗ +0.50 · −0.51 | 2/7 |
| P2·P8·R0·RSI·EA | 2,222 | 1.80 | 49 % | 49 % | −0.07 | −0.27 … +0.13 | ✗ | ✗ −0.33 | ✗ −0.22 | ✓ 0.000 (−0.67) | ✓ 0.000 | ✗ −0.070 | ✗ +0.63 · −0.27 | 2/7 |
| P2·P8·R0·K·EA ("the operator's rule") | 3,573 | 2.89 | 47 % | 50 % | **−0.21** | −0.40 … −0.03 | ✗ | ✗ −0.48 | ✗ −0.37 | ✓ 0.000 (−0.56) | ✓ 0.000 | ✗ −0.098 | ✗ +0.11 · −0.16 | 2/7 |
| P2·P8·R−3·K·EA | 4,541 | 3.68 | 45 % | 49 % | −0.32 | −0.48 … −0.17 | ✗ | ✗ −0.59 | ✗ −0.48 | ✓ 0.001 (−0.49) | ✗ 0.075 | ✗ −0.118 | ✗ −0.42 · −0.17 | 1/7 |
| T 12 h · S5 (best T) | 5,758 | 4.66 | 57 % | 56 % | **+0.10** | −0.16 … +0.38 | ✗ | ✗ −0.16 | ✗ −0.05 | ✗ 0.272 (+0.06) | ✗ 1.000 | ✗ −0.034 | ✗ +0.43 · −0.19 | 0/7 |
| T 12 h · S10 | 5,758 | 4.66 | 47 % | 47 % | −0.01 | −0.31 … +0.32 | ✗ | ✗ −0.53 | ✗ −0.19 | ✓ 0.031 (−0.14) | ✗ 0.730 | ✗ −0.048 | ✗ +0.12 · −0.37 | 1/7 |
| T 8 h · S5 | 6,317 | 5.11 | 54 % | 55 % | −0.13 | −0.31 … +0.06 | ✗ | ✗ −0.40 | ✗ −0.28 | ✗ 1.000 (+0.10) | ✗ 1.000 | ✗ −0.084 | ✗ −0.03 · −0.26 | 0/7 |
| T 2 h · S5 | 7,026 | 5.69 | 50 % | 52 % | −0.21 | −0.35 … −0.07 | ✗ | ✗ −0.48 | ✗ −0.37 | ✗ 1.000 (+0.04) | ✗ 1.000 | ✗ −0.097 | ✗ −0.05 · −0.22 | 0/7 |
| T 4 h · EA | 8,552 | 6.92 | 45 % | 48 % | −0.30 | −0.56 … −0.11 | ✗ | ✗ −0.57 | ✗ −0.46 | ✗ 1.000 (−0.14) | ✗ 1.000 | ✗ −0.111 | ✗ +0.01 · −0.15 | 0/7 |

**Across the grid:**
- 2 of 24 P2 cells and 1 of 18 T cells have avg > 0. **No cell passes more than 3 of 7.**
- Every P2 cell beats its random-timing null. Only 3 T cells do, and none family-wise.
- Random timing in the P2 window averages −0.89 %/trade. In the T window it is −0.16 %.

**Per half-year and costs** (n in brackets; DESIGN = 2024-07 → 2026-06):

| cell | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | DESIGN | @0.05 | @0.10 | @0.20 | EXH overlap | avg of non-EXH trades |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P2·P15·R0·RSI·EA | +0.62 (30) | +0.80 (61) | +1.02 (37) | +0.39 (64) | +0.16 (272) | −0.58 (272) | +0.04 (211) | −0.08 | +0.10 | +0.03 | −0.12 | 19 % | −0.04 |
| P2·P8·R0·RSI·S10 | +0.71 (111) | +0.39 (217) | +1.07 (181) | +0.42 (247) | −0.21 (456) | −0.30 (469) | −0.51 (376) | +0.04 | +0.10 | +0.02 | −0.15 | 26 % | −0.22 |
| P2·P8·R0·K·EA | −0.04 (175) | +0.19 (305) | +0.74 (249) | +0.02 (390) | −0.66 (975) | −0.30 (833) | −0.16 (646) | −0.29 | −0.14 | −0.21 | −0.37 | 10 % | −0.33 |
| T 12 h · S5 | +0.30 (486) | +0.50 (789) | −0.16 (635) | −0.53 (746) | +0.69 (1257) | −0.20 (1095) | −0.19 (750) | +0.04 | +0.17 | +0.10 | −0.05 | 4 % | +0.10 |
| T 4 h · EA | −0.11 (614) | +0.08 (968) | +0.12 (793) | −0.55 (1034) | −0.67 (2090) | −0.29 (1784) | −0.15 (1269) | −0.42 | −0.22 | −0.30 | −0.46 | 10 % | −0.37 |

**The P2 decay is consistent.** Every RSI-on-the-second-leg cell was clearly positive in 2023-H2 to 2024-H2, when Frenzy episodes were few (30–250 per half). It turned negative from 2025-H2, when episodes multiplied by 4–7×. More, smaller frenzies now have second legs that keep running.

**EXHAUSTION_SHORT comparison** (REF = R4 × EA on the all-spike list, as in EXHAUSTION_EXPAND §0):
- EXH earns ≈ +0.43 %/trade at ≈ 3.3/day.
- The best P2 cell overlaps EXH on 19 % of trades. The rest average −0.04 %. **P2 adds no value on top of EXH.**

## W/L — best P2 cell (P2·P15·R0·RSI·EA, N 947) and the T reference (T 24 h · S10, N 3,027)
**Coverage.**
- 20 entry-time variables at 99–100 %: vol decay, 15m RSI, ATR %, hours since flag, volume ×normal, 24 h volume, 24 h move, distance to VWAP, run since base, off-peak, gain, in-state, last funding, pullback depth, peak-2/peak-1 height ratio, BTC 1h / 4h / 24h return, BTC vs 1h EMA20, BTC daily EMA50 vs 200.
- Spot-share change: 50 % (coins with no spot market).
- **Blind spots:** taker buy/sell ratio (the cached klines carry no taker volume) · order book · open interest · long/short ratio · market cap · news.

**Best P2 cell:**
- **No 1D separator beats the shuffled null.** The largest tercile spread is the spot-share change at 1.22 %, against a null 95th percentile of 1.38.
- **2D:** the top pair is last funding (middle tercile) × spot-share change (middle tercile): n 46, +2.53 %, WR 76 %, max-stat null p 0.037.
  - It is a middle-tercile × middle-tercile hole on a 50 %-coverage variable. That is the "non-monotonic = confound" pattern, so it is watchlist only.
- **BTC daily regime (sign):** EMA50 > EMA200 gives +0.34 (477) vs −0.27 (464) below. That fits the operator's live lesson that entry rules flip with the BTC daily trend. Null p 0.99, so it is not significant here. Note only.
- **Peak-2/peak-1 ratio:** the second top being much higher or only a re-test does not separate (terciles +0.42 / −0.30 / −0.04, p 0.95).

**T 24 h · S10 (the reference losing cell):** several 1D separators pass the null:
- vol decay (p 0.007), volume ×normal (p 0.003), BTC 1h return (p 0.003), BTC vs EMA20 (p 0.007), BTC 24h (p 0.03), run since base (p 0.02).
- **Every one only separates "very bad" from "≈ 0".** The low-volume tercile is −1.35 vs +0.21 in the top tercile. None creates a positive cell.

Full tables: `_data/wl_P2_P15_R0_RSI_EA.json`, `_data/wl_T_24h_S10.json`.

## Step 3 — the operator's coins (data fetched to 11-18 ≈ 12:30 UTC)
Peaks are zigzag 8 % (15 % in brackets where different). Net % at 0.10 % slippage. The EXH column is my approximation of the shipped rule; it is not the engine's log.

| coin | flag ON (UTC) | peak 1 → peak 2 (→ main top) | P2·P8·R0·K·EA | P2·P15·R0·RSI·EA (best P2) | T 2 h · 4 h · 8 h (EA) · T 12 h S5 | EXH (approx) |
|---|---|---|---|---|---|---|
| **BREV** 11-17 | 12:50 | P1 15:15 (+2.4 h) → small P2 16:00 → **main top 11-18 01:30** (+12.7 h, 15 % zigzag P2, 17 % above P1, then −36 % in 24 h) | **01:45 → +4.8 %** (TP in 5 min) | 02:00 → +4.8 % | −5.3 · −6.4 · −5.7 · −9.4 (stop gapped) | no trade: vol decay 0.63 < 0.6366 at the 01:45 cross, spot share rising |
| **HUMA** 11-16 | 16:25 | P1 16:15 → **P2 21:00 = top** (+4.6 h, then −30 %) | 21:45 → +4.4 % | — (no 15 % pullback) | −5.4 · −5.3 · +4.7 · — | 17:30 → −5.0 % gross (stopped before the top) |
| **BLESS** 11-16 | 09:20 | P1 09:30 → P2 15:15 → **top 17:05** (+102 % vs base; −51 % in 24 h) | 13:45 → **−5.3 %** (entered a lower leg; the coin ran another +35 %) | 17:45 → −5.3 % (S10 version +10.0 %) | −5.3 · −5.3 · +4.8 · +4.9 | no trade (no spot market = EXH_NO_SPOT) |
| BLESS 11-18 (in progress) | 03:55 | P1 04:00 → P2 07:15 = top so far | 07:45 → +4.8 % | — | −5.3 · +4.8 · −1.3 (open) · — | no spot |
| AIOT 11-16 | 16:20 | P1 16:30 → P2 18:15 → top 11-17 01:30 | 11-17 01:45 → +4.8 % | — | +4.8 · −5.3 · −5.3 · — | no spot |
| SPACE 11-16 | 13:50 | P1 14:45 → P2 15:30 → top 22:45 | 16:00 → +4.8 % (TP before the later top) | — | +4.8 · −5.3 · +4.8 · +4.8 | no spot |
| ZKC 11-18 (in progress) | 08:20 | single peak 09:45 (top so far; now −15 %) | — | — | +4.8 · +1.7 (open) | blocked: vol decay 0.63 |
| ZKP 11-18 (in progress) | 06:00 | single peak 08:00 (now −17 %) | — | — | −5.3 · +4.8 | 07:30 −5.0 · 08:30 +5.0 (gross) |
| **PIPPIN** 11-18 (**in progress**) | 07:40 | P1 08:15 → P2 08:45 → P3 10:15 → top so far 10:25 (+98 % vs base); **now −12 % off the top**, flag still on | 09:00 → **−5.3 %** (P2 was not the top) | — | −5.3 · +4.9 | blocked (no spot) |

![BREV](SECOND_PEAK_2026-11-18_replay_BREVUSDT.png)
![HUMA](SECOND_PEAK_2026-11-18_replay_HUMAUSDT.png)
![BLESS](SECOND_PEAK_2026-11-18_replay_BLESSUSDT.png)
![PIPPIN](SECOND_PEAK_2026-11-18_replay_PIPPINUSDT.png)

Also: `_replay_AIOTUSDT.png`, `_replay_SPACEUSDT.png`, `_replay_ZKCUSDT.png`, `_replay_ZKPUSDT.png`.

**Reading the replay:**
- **The simple second-peak rule won 5 of 7 this week. BREV is exactly the operator's picture.**
- But BREV's "second peak" that dumped was its **main** top, 12.7 h after the flag, after a 17 % higher second leg. A rule cannot know at 01:15 that this leg is the last one.
- BLESS 11-16 and PIPPIN show the other side. The second top was followed by a third leg up, and the short was stopped.
- On 3 years of data, those two cases are as common as the winners (47 % WR).
- A fixed time after the flag lost on BREV at every X (2 / 4 / 8 / 12 h). BREV squeezed between peaks.

## What it means
1. **The pattern the operator sees is real in hindsight but not stable enough to trade.**
   - Frenzy coins do dump hard after their top: median −15 % three days after the top, whichever peak it is.
   - But whether *this* top is the last one is close to a coin flip at the moment you would short.
   - The second peak is the top in only 8–19 % of all episodes, and in 28–32 % of episodes that have a second peak at all.
2. **Waiting for a second top is much better than shorting a Frenzy coin at random.** The random baseline is ≈ −0.9 %/trade.
   - So if the bot ever shorts these coins, it should not short them early or at fixed times. This supports the existing EXH design (it waits for volume decay and an RSI rollover).
3. **Time X after the flag carries no information.** The tops are spread from minute 0 to hour 24.

## Verdict
- **T (time after flag): NO.** 0–1 of 7 items in every cell, and no cell beats random timing family-wise.
- **P2 (second peak): NO.**
  - The best cell is +0.03 %/trade with 3 of 7 items. It decays from 2025-H2 and goes negative at 0.20 % slippage.
  - It adds nothing on top of EXHAUSTION_SHORT (the non-overlapping trades average −0.04 %).
  - It is not good enough for an observe line either: the prereg OBSERVE bar is ≥ 5 items including holdouts.

## Next step
- **Nothing to build.** EXHAUSTION_SHORT stays the Frenzy short.
- **Watchlist only** (one line, no arm):
  - P2·P15·R0·RSI·EA restricted to BTC daily EMA50 > EMA200 (+0.34, n 477, null p 0.99). This is the regime flip from the live lessons.
  - Re-read it at the next batch review on new episodes only, with a frozen definition.
- **Operator hypothesis for a follow-up (not tested here):** "low market-cap coins (< ~$50 M) dump harder after the Frenzy top."
  - Market cap is a blind spot in every Frenzy study so far.
  - A test needs circulating-supply history per coin, e.g. a CoinGecko/CMC snapshot at each spike.
  - Pre-register it as a size/filter split of EXHAUSTION_SHORT and of the Step-0 drawdown, both at sign and tercile granularity.
  - The proxy available today (24 h quote volume) does not separate in the P2 W/L.

## Assumptions and limits
- Episodes come from the engine rule re-implemented on 5m klines, with a per-bar normal-hour approximation (as in all earlier Frenzy studies). 658 pairs, including delisted ones present in the caches.
- The short-list gate is approximated as 24 h move ≥ 15 % ∧ 24 h volume ≥ $20 M at the signal bar. It does not model the live depth guard.
- Fixed 0.10 % slippage per taker leg. Thin coins can slip 0.3–0.6 % live (INIT). At 0.20 % every cell is ≤ 0.
- Funding: real settlements. Missing months would be charged 0.05 %/8 h (none were missing).
- Hindsight peaks (Step 0) use a 15m zigzag. Confirmed peaks imply a drop by construction.
- **Holdouts:**
  - HOLDOUT-A 2023-07 → 2024-06: 2023-H2 was read by earlier Frenzy studies, so it is not virgin.
  - HOLDOUT-B 2026-07 → 11-17: cut at the cache end. The replay coins use fresh data, but no replay trade enters the cell statistics.
- EXH replay is an approximation from the published definitions (spot-share, vol decay since the spike bar). The engine's own logs decide what it actually did.
- One implementation fix (same-bar pullback / cancel) was applied after the first run. It is disclosed above, and both grids are saved.
