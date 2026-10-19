# SPIKE_FADE size: by market cap, or a lower cap? (2026-10-19)

Research only. No engine, config, `trading_config.json`, template or test change. Nothing committed, and the bot API was not touched.
Binance public data only: one KMNO daily trade file from data.binance.vision, the 12 worst back-test stop days, one order-book snapshot of
281 fade-universe pairs (peak weight 706/min, paused at 700, no 418/429) and the Binance Info-panel supply for 77 pairs.
`scripts/validate_against_master.py` → **ALL CHECKS PASS**. Master fades reconcile with STACK 2026-10-16a and 2026-10-19a (same 74 kept fades · 89 % · +0.356 % · Σ +26.35 pts · stack_pnl $4,531).

> **⚖ OPERATOR DECISION 2026-10-19 (DECISION_LOG 312) — supersedes the §6 / §8 recommendations and the §8.8 gates below:** NO change to
> spike-fade sizing (operator: "spike fade is one of our best performing sleeves"). The candidate "1× if mcap ≥ $300M, 2× below" is
> WATCHLISTED with its own frozen gate in CLAUDE_CURRENT_STATE 41i — threshold sweep in §9. A cut on HIGH-volume fades was also rejected.

> **UPDATE 2026-10-19 (§8, operator follow-up on higher-mcap risk). Final recommendation = `min(0.6× ticket, 0.05 % of 24 h volume)`**, i.e. the
> §6 rule plus a hard cap of ≈ 3× the account (`spike_fade_invest_mult` 2.0 → **0.6**, `spike_lowvol_liq_cap_pct` 0.5 → 0.05; no new plumbing).
> Higher mcap is NOT where the tail lives (master ≥ $200M 9/9 wins; backtest ≥ $200M has the smallest stop gaps); KMNO lost on ticket size vs a
> book that is bottom-3 % thin for its mcap. **Correction:** 3 master fades had a zero bracket-cap stamp that the first pass read as a cap
> (KOMA −1.5 %, RESOLV, FLOW sized at $0). Fixed; all master $ / P30 / P50 below are the corrected values (rankings unchanged;
> e.g. today −$561 → −$888, the §6 rule +$512 → +$481). Backtest and live figures were not affected.

## Plain English first

**Recommendation: cut every fade to `min(1× ticket, 0.05 % of the pair's 24 h volume)`.** In config terms that is
`spike_fade_invest_mult` 2.0 → 1.0 and `spike_lowvol_liq_cap_pct` 0.5 → 0.05. Both fields already exist, so there is no new plumbing.
On the $1,225 account the fade ticket today is ≈ $11.9k. Under the rule it would be at most $6.0k, and the median would be ≈ $1.7–2.5k
(1.4–2.1× the account).

**The operator's idea that "lower mcaps work fine at 2×" is not supported. The data points the other way.**
- **Edge.** Low-mcap fades had the weakest edge in both blocks:
  - master: < $50M +0.20 % vs +0.38 % (50–200M) and +0.45 % (200M–1B);
  - backtest: +0.01 % vs +0.05 %, +0.05 % and +0.11 %.
- **Order books.** Low-mcap coins have the thinnest books. A $12k market order moves price a median 0.19 % on a < $50M coin vs 0.105 % on a
  $200M–1B coin. Mcap explains book depth better than 24 h volume does (R² 0.42 vs 0.23).
- **The mcap rules.** 2× on the low-mcap third is the **worst** sizing rule I priced, in every block. 2× only on the top-mcap third beats
  today but loses to plain size or liquidity caps. It would also have left KMNO itself ($332M mcap) at full 2×.
- **Plumbing.** Mcap is not available at fade time for most fade pairs. The cache covers only the Top-N pairs plus FRENZY flags, which is why
  KMNO's stamp is NaN. An mcap rule would need new plumbing plus a fail-closed default (missing mcap → small size).

**What actually hurt on KMNO: our own order eating the book, not a price gap.** The tick-by-tick read of the two KMNO orders:
- The entry sell swept 17 price levels: **−0.24 %**.
- The stop buy-back swept 24 levels, 0.6 s after a $107k squeeze had thinned the book: **−0.345 %**.
- Gap past the stop line: only ≈ 0.04 %.
- So −2.13 % ≈ −1.5 stop − 0.09 fees − **0.58 % own market impact**.
- At half the size the same books cost 0.33 % instead of 0.58 %. At a quarter size, 0.19 %.

**Why this matters beyond KMNO.**
- **Paper fills carry zero market impact.** The master fades earned their +0.36 % on paper tickets of ≈ $9k.
- **What live tickets will cost.** An impact model fitted on the 281-book snapshot (cost grows with size^0.55 and falls with volume and mcap)
  and stress-calibrated on KMNO says a $12k fade pays ≈ 0.44 % round trip on a typical fade pair. That is more than the whole master edge.
- **Today's 2× on master fades priced live:** −0.10 % per trade and **−$888** on the live account, vs +$2,972 if fills were free.
- **The recommended rule on the same fades:** +0.17 % per trade and **+$481**.

**Which liquidity variable predicts what.**
- The **path gap** past the stop (backtest, 144 stops, 12 worst re-priced on ticks) is **not** predicted by mcap, volume or turnover
  (|Spearman| ≤ 0.09). Those tails are random with respect to liquidity, and **only a smaller ticket bounds them**.
- **Own impact** is predicted by ticket ÷ book depth. Depth tracks **mcap best**, then 24 h volume.
- 24 h volume at fade time is inflated by the spike itself: median 1.8× the pair's normal volume, KMNO 3×. A volume-% cap therefore
  over-states liquidity, which is why it needs a tight setting and a second leg.

**Why both legs.**
- **The volume cap alone does not protect the account.** KMNO traded on $17.8M of volume, so even a 0.05 % cap would have allowed $8.9k
  (7.3× the account) → −$179.
- **The 1× leg bounds the account.** No fade is ever above ≈ 4.9× the account (worst stop ≈ −9 % of the account).
- **The volume leg bounds market impact** and shrinks the thin-pair fades, the weakest cohort in both blocks.

**Evidence strength: moderate for the direction, weak for the exact numbers.**
- Smaller is better for ruin in every block and under every impact assumption. Low-volume fades are weaker in master and backtest.
- The 0.05 % value was picked from the brief's 3-value grid on the same blocks. The impact model rests on one live fill plus a calm snapshot.

**Effect on profit and ruin, at today's $1,225 account (central impact model):**

| block | $ today → rule | worst single fade, % of account | P(drawdown ≥ 50 % in a year) |
|---|---|---|---|
| master (74 fades) | −$888 → **+$481** (after a 30–50 % haircut ≈ +$240–340) | −23.7 % → −3.1 % | 100 % → 0 % |
| backtest (625 fades) | −$28.4k → −$2.0k | −53 % → −15 % | 100 % → 93 % (no-impact world 90 % → 14 %) |
| KMNO | −$252 → −$112 (−20.7 % → −9.1 % of the account) | | |

**What it gives up.** If live impact turned out to be as small as paper's (it was not on KMNO), the rule would leave money on the table:
master $2,972 → $839 (−72 %), backtest $4,512 → $2,109 (−53 %).

**Sleeve-level flag (not a kill proposal).** With realistic impact the backtest fade is negative at every size tested. The best case is
−0.04 %/trade at half impact, and the rule's $ total is +$47 at half impact. The master edge after impact is +0.17 % at the new size.
The first live fades at the new size (with their slippage stamps) are the deciding evidence.

---

## 1. How today's live fade size is computed (code + config, read 2026-10-19)

| step | value | source |
|---|---|---|
| equal-split slot | (equity − 2.5 % fee reserve) ÷ 4 slots ≈ $298.6 on $1,225 | `max_open_positions` 4, `fee_reserve_pct` 2.5 |
| fade multipliers | invest **2.0** × lev **1.0** (20× base) → margin ≈ $597, notional ≈ **$11.9k = 9.75× the account** | `spike_fade_invest_mult` 2.0, `spike_fade_lev_mult` 1.0; engine `trading_engine.py` ~12201 ("liquidity cap governs") |
| liquidity cap ① | notional ≤ **0.5 %** of 24 h volume for every spike (fade, chase-probe, bounce) | `spike_lowvol_liq_cap_pct` 0.5 with `spike_lowvol_threshold_usd` $1T (= all pairs); engine ~12330; global `max_notional_pct_of_pair_volume` 0.1 for non-spikes |
| fade volume ceiling | fades refused at ≥ $20M 24 h volume | `spike_fade_max_vol_24h_usd` 2e7 |
| bracket cap | notional ≤ exchange bracket at the leverage (KMNO 50× → $25k) | `leverage_bracket_cap_enabled` |
| gross cap | Σ open notional ≤ 25× balance | `max_gross_leverage` 25 |
| stop | −1.5 % net, market close; tripwire −2.5 | `spike_fade_sl_pct`, `spike_fade_tripwire_pct` |

**On a ~$1.2k account the 0.5 % cap binds only when the 24 h volume is below $2.4M.** Fade pairs run $2–20M, so practically every live fade
opens at the full $11.9k.

**Mcap at fade time.** Every bot order calls `mcap_service.get(pair)`, which reads the cache only. The cache is refreshed for the Top-N scan
pairs plus FRENZY-flagged pairs (engine ~17259). FRENZY_WILLY's turnover filter additionally asks for a one-pair fetch when it arms (~9700).
The fade path does neither, so most fade pairs carry `entry_mcap_usd` = NaN. KMNO (rank 95 by volume) is NaN.

**Mcap coverage.**
- master stamps: 0 / 74
- backtest replay proxy: 494 / 625
- live: 0 / 1
- **Rebuilt for this study** (Info-panel circulating supply today × entry price, the replay's own proxy): master 69 / 74, backtest
  537 / 625, KMNO 1 / 1 ($332M).
- Unscored fills are shown as their own row, never as "rest".

## 2. KMNO tick forensics (aggTrades 2026-10-17, data.binance.vision)

| leg | time (UTC) | our qty | levels swept | best → VWAP | own impact | impact if the ticket had been ½ / ¼ |
|---|---|---|---|---|---|---|
| entry SELL | 00:29:06.544 | 306,154 | 17 | 0.03887 → 0.038777 | **0.239 %** | 0.133 % / 0.076 % |
| stop BUY | 00:32:20.205 | 306,154 | 24 | 0.03943 → 0.039566 | **0.345 %** | 0.200 % / 0.109 % |
| context | 00:32:19.623 | – | – | a third-party $107k sweep spiked 0.03903 → 0.04005 in 2 ms | stop line ≈ 0.03932 | – |
| **TOTAL** | | | | | **0.584 %** | 0.333 % / 0.185 % |

−2.125 % = −1.44 % price (best ask vs best bid) − 0.09 % fees − 0.584 % own impact. The gap past the stop line was ≈ 0.04 %.
Today's calm-book snapshot for KMNO prices a $12k order at 0.168 %, so the live entry paid ≈ 1.4× calm and the squeeze exit ≈ 2.05× calm.
**These two stress multipliers drive the impact model below.**

## 3. Order-book snapshot: which liquidity measure predicts impact (281 pairs, 2026-10-19 03:05 UTC; 165 in the $1.5–30M band)

| fit, log impact of a market order | coefficients | R² |
|---|---|---|
| size + 24 h volume | size^0.546 · vol^−0.336 | 0.469 |
| size + mcap | size^0.545 · mcap^−0.238 | 0.525 |
| size + volume + mcap | size^0.545 · vol^−0.189 · mcap^−0.188 | **0.557** |
| depth within 1 %: vs volume / mcap / both | – | 0.23 / **0.42** / 0.47 |

| bucket (snapshot) | N | impact $6k | impact $12k | depth ±1 % |
|---|---|---|---|---|
| mcap < $50M | 68 | 0.125 % | 0.188 % | – |
| mcap $50–200M | 58 | 0.095 % | 0.129 % | – |
| mcap $200M–1B | 29 | 0.078 % | 0.105 % | – |
| mcap > $1B | 6 | 0.060 % | 0.079 % | – |
| vol < $3M | 65 | 0.130 % | 0.184 % | $52k |
| vol $3–6M | 50 | 0.098 % | 0.142 % | $73k |
| vol $6–12M | 30 | 0.069 % | 0.097 % | $160k |
| vol $12–20M | 10 | 0.077 % | 0.104 % | $148k |
| **TOTAL (1.5–30M band)** | **165** | – | median ≈ 0.14 % | – |

**Impact model used below ("central").** For each fill:
- calm cost = the size + volume + mcap fit (volume-only fit when mcap is unscored), with 24 h volume ÷ 1.8 for spike inflation;
- live cost = calm × (1.4 entry + 2.05 stop exit, or 1.0 other exit).

For KMNO the model predicts 0.34 % vs the measured 0.58 %, so it is not harsh. Every rule is also priced at ½ impact and at zero impact
(= paper).

## 4. Outcome by MCAP, by 24 h VOLUME, by turnover (three blocks)

Columns:
- `avg` / `Σ` = net % per fill / sum, as priced (master = today's stack `stack_pct`, paper; backtest = yr6W at today's rules, the 12 worst
  stops re-priced on ticks; live as traded). `CI` = 95 % day-clustered bootstrap.
- `gap` = backtest stop fill beyond −1.5 (mean / p90). Master is paper, so it has no gap.
- `imp2×` = modelled round-trip impact at today's $11.9k ticket.
- `live2×` / `live1×` = avg after impact at 2× / 1×.

### By MCAP

| block | mcap | N | WR | avg | CI | Σ % | worst | stops | gap mean / p90 | ≤ −2 % | imp2× | live2× | live1× |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASTER | < $50M | 33 | 82 % | +0.201 | [−0.06, +0.44] | +6.6 | −1.50 | 5 | – | 0 | 0.55 | −0.35 | −0.20 |
| MASTER | $50–200M | 27 | 93 % | +0.375 | [+0.14, +0.59] | +10.1 | −1.50 | 2 | – | 0 | 0.40 | −0.03 | +0.09 |
| MASTER | $200M–1B | 8 | 100 % | +0.454 | [+0.34, +0.57] | +3.6 | +0.27 | 0 | – | 0 | 0.31 | +0.14 | +0.24 |
| MASTER | > $1B | 1 | 100 % | +0.542 | – | +0.5 | +0.54 | 0 | – | 0 | 0.17 | +0.37 | +0.42 |
| MASTER | unscored | 5 | 100 % | +1.085 | – | +5.4 | +0.24 | 0 | – | 0 | 0.46 | +0.63 | +0.77 |
| **MASTER** | **TOTAL** | **74** | **89 %** | **+0.356** | | **+26.3** | −1.50 | 7 | – | 0 | 0.46 | **−0.10** | **+0.03** |
| BT | < $50M | 217 | 73 % | +0.008 | [−0.13, +0.15] | +1.7 | −2.98 | 52 | 0.07 / 0.09 | 1 | 0.54 | −0.54 | −0.37 |
| BT | $50–200M | 210 | 73 % | +0.050 | [−0.12, +0.23] | +10.5 | −4.87 | 47 | 0.25 / 0.59 | 8 | 0.42 | −0.37 | −0.24 |
| BT | $200M–1B | 92 | 73 % | +0.050 | [−0.16, +0.26] | +4.6 | −2.27 | 22 | 0.06 / 0.06 | 1 | 0.32 | −0.27 | −0.17 |
| BT | > $1B | 18 | 83 % | +0.110 | [−0.27, +0.45] | +2.0 | −1.50 | 3 | 0.00 / 0.00 | 0 | 0.20 | −0.09 | −0.03 |
| BT | unscored | 88 | 70 % | +0.213 | [−0.12, +0.54] | +18.8 | −3.57 | 20 | 0.32 / 0.67 | 6 | 0.45 | −0.24 | −0.10 |
| **BT** | **TOTAL** | **625** | **73 %** | **+0.060** | | **+37.6** | −4.87 | 144 | 0.16 / 0.52 | 16 | 0.45 | **−0.39** | **−0.25** |
| LIVE | $200M–1B (KMNO) | 1 | 0 % | −2.125 | – | −2.1 | −2.13 | 1 | 0.63 (exit_slippage_pct) | 1 | 0.58 (measured) | −2.13 | −1.88 |
| **LIVE** | **TOTAL** | **1** | 0 % | −2.125 | | −2.1 | | | | | | | |

### By 24 h VOLUME (bucket = ticket ÷ volume at today's $11.9k ticket)

| block | 24 h volume (ticket ÷ vol) | N | WR | avg | CI | Σ % | worst | gap mean / p90 | ≤ −2 % | imp2× | live2× | live1× |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MASTER | < $3M (> 0.40 %) | 27 | 85 % | +0.238 | [−0.03, +0.47] | +6.4 | −1.50 | – | 0 | 0.50 | −0.26 | −0.12 |
| MASTER | $3–6M (0.20–0.40 %) | 27 | 85 % | +0.219 | [−0.08, +0.48] | +5.9 | −1.50 | – | 0 | 0.48 | −0.26 | −0.12 |
| MASTER | $6–12M (0.10–0.20 %) | 17 | 100 % | +0.686 | [+0.39, +1.15] | +11.7 | +0.22 | – | 0 | 0.39 | +0.29 | +0.41 |
| MASTER | ≥ $12M (< 0.10 %) | 3 | 100 % | +0.785 | – | +2.4 | +0.54 | – | 0 | 0.33 | +0.46 | +0.56 |
| **MASTER** | **TOTAL** | **74** | **89 %** | **+0.356** | | **+26.3** | | | 0 | 0.46 | −0.10 | +0.03 |
| BT | < $3M | 146 | 70 % | +0.040 | [−0.15, +0.24] | +5.8 | −4.87 | 0.21 / 0.41 | 3 | 0.53 | −0.49 | −0.33 |
| BT | $3–6M | 209 | 72 % | −0.046 | [−0.18, +0.09] | −9.6 | −3.57 | 0.13 / 0.41 | 5 | 0.48 | −0.52 | −0.37 |
| BT | $6–12M | 177 | 74 % | +0.123 | [−0.05, +0.31] | +21.8 | −4.34 | 0.14 / 0.14 | 4 | 0.40 | −0.28 | −0.15 |
| BT | ≥ $12M | 93 | 80 % | +0.209 | [−0.06, +0.51] | +19.5 | −2.72 | 0.20 / 0.59 | 4 | 0.33 | −0.12 | −0.02 |
| **BT** | **TOTAL** | **625** | **73 %** | **+0.060** | | **+37.6** | | 0.16 / 0.52 | 16 | 0.45 | −0.39 | −0.25 |
| LIVE | ≥ $12M (KMNO 0.067 %) | 1 | 0 % | −2.125 | – | −2.1 | | 0.63 | 1 | 0.58 | – | – |
| **LIVE** | **TOTAL** | **1** | | −2.125 | | −2.1 | | | | | | |

KMNO's ticket ÷ volume was 0.067 %, 7× larger than any other live fill of any sleeve (B22 + this export, 42 fills, all ≤ 0.011 %).

### By turnover (24 h volume ÷ mcap) — TOTAL rows as above

| block | < 0.02 | 0.02–0.05 | 0.05–0.15 | ≥ 0.15 | unscored |
|---|---|---|---|---|---|
| MASTER N · avg | 9 · +0.43 | 16 · +0.43 | 22 · +0.20 | 22 · +0.26 | 5 · +1.09 |
| BT N · avg | 77 · +0.05 | 124 · −0.12 | 192 · +0.02 | 144 · +0.18 | 88 · +0.21 |

There is no consistent shape across the two blocks, so turnover is not a sizing variable.

### What predicts the loss tail (backtest, Spearman on log variable)

| target | mcap | 24 h volume | turnover |
|---|---|---|---|
| stop gap past −1.5 (144 stops) | −0.08 | +0.06 | +0.09 |
| tail fill ≤ −2 % (625) | +0.04 | +0.04 | −0.05 |
| avg % (625) | +0.03 | +0.09 | +0.04 |
| master avg % (74) | +0.16 | +0.11 | −0.09 |

**Reading.**
- **Gap tails** (FUSDT −4.4 % on ticks, DAM −3.4 %, PARTI −4.3 %) land on any liquidity bucket. No mcap or volume rule can avoid them, so
  only the ticket size bounds what they cost the account.
- **Own impact** is a pure size ÷ depth effect (§2–3).
- **Edge** rises with both mcap and volume, most clearly with volume (master ≥ $6M 20/20 wins). It is direction-consistent in master and
  backtest, and small in both.

## 5. Candidate rules priced — at today's live account ($1,225), central impact

Column definitions:
- `x acct` = median notional ÷ account.
- `$` = sum over the block, no compounding.
- `worst` = worst single fade as % of the account.
- `P30` / `P50` = probability of a ≥ 30 % / ≥ 50 % drawdown within a year. These come from 2,000 one-year paths built from 7-day blocks of the
  block's real calendar (backtest 2025-01-01 → 2026-10-11; master Jul-28 → Oct-13), compounding, with the ticket re-computed from equity
  every day.
- `½imp` = the same $ with half the impact. `paper` = zero impact.
- `paper-era $` = master at the $2,900 paper book, zero impact (the master's own `stack_pnl` total = **$4,531**, for reference).
- `shrunk` = master / backtest fills smaller than today's ticket.

| rule | shrunk | x acct | MASTER avg | MASTER $ | worst | P30 | P50 | ½imp $ | paper $ | paper-era $ | BT avg | BT $ | worst | P30 | P50 | ½imp $ | paper $ | P50 paper | KMNO $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **(a) today 2×, liq 0.5 %** | 0 / 0 | 9.75 | −0.10 | −888 | −23.7 | 100 % | 100 % | +1,042 | +2,972 | 5,522 | −0.38 | −28,443 | −53.4 | 100 % | 100 % | −11,966 | +4,512 | 90 % | −254 |
| (b) 1× all | 70 / 625 | 4.88 | +0.03 | +132 | −10.4 | 100 % | 69 % | +847 | +1,562 | 3,398 | −0.25 | −9,185 | −25.8 | 100 % | 100 % | −3,471 | +2,244 | 71 % | −112 |
| (b2) 0.5× all | 74 / 625 | 2.44 | +0.13 | +289 | −4.7 | 22 % | 0 % | +538 | +787 | 1,838 | −0.15 | −2,794 | −12.6 | 100 % | 95 % | −836 | +1,122 | 11 % | −52 |
| (c1) 2× if mcap < $40M else 1× (operator's idea) | 41 / 446 | 4.88 | −0.04 | −841 | −23.7 | 100 % | 100 % | +484 | +1,809 | 4,002 | −0.30 | −16,762 | −36.2 | 100 % | 100 % | −7,188 | +2,387 | 84 % | −112 |
| (c2) 2× if mcap > $102M else 1× | 49 / 446 | 4.88 | −0.00 | +179 | −10.4 | 100 % | 71 % | +1,170 | +2,161 | 3,933 | −0.28 | −13,018 | −47.2 | 100 % | 100 % | −5,045 | +2,929 | 81 % | −254 |
| (d) 2×, liq 0.05 % | 74 / 625 | 1.43 / 2.09 | +0.17 | +499 | −3.1 | 0 % | 0 % | +687 | +874 | 874 | −0.14 | −2,522 | −24.3 | 100 % | 95 % | −157 | +2,207 | 20 % | −179 |
| (d) 2×, liq 0.1 % | 71 / 530 | 2.86 / 4.18 | +0.09 | +632 | −6.8 | 10 % | 1 % | +1,155 | +1,679 | 1,748 | −0.23 | −7,816 | −30.3 | 100 % | 100 % | −1,799 | +4,218 | 47 % | −254 |
| (d) 2×, liq 0.2 % | 51 / 354 | 5.52 / 8.35 | −0.00 | +301 | −15.1 | 93 % | 70 % | +1,351 | +2,401 | 3,380 | −0.31 | −17,260 | −47.2 | 100 % | 100 % | −6,437 | +4,385 | 69 % | −254 |
| (e) 2×, ≤ 3× account | 74 / 625 | 3.00 | +0.10 | +283 | −6.0 | 59 % | 4 % | +625 | +968 | 2,248 | −0.17 | −4,017 | −15.6 | 100 % | 99 % | −1,318 | +1,381 | 25 % | −65 |
| (e) 2×, ≤ 4× account | 74 / 625 | 4.00 | +0.06 | +222 | −8.3 | 95 % | 34 % | +756 | +1,291 | 2,903 | −0.21 | −6,577 | −21.0 | 100 % | 100 % | −2,368 | +1,841 | 51 % | −89 |
| (e) 2×, ≤ 5× account | 70 / 625 | 5.00 | +0.03 | +116 | −10.7 | 100 % | 75 % | +858 | +1,600 | 3,464 | −0.25 | −9,583 | −26.5 | 100 % | 100 % | −3,641 | +2,301 | 70 % | −115 |
| (e) 2×, ≤ 6× account | 70 / 625 | 6.00 | −0.01 | −37 | −13.3 | 100 % | 93 % | +935 | +1,908 | 3,974 | −0.28 | −12,990 | −32.0 | 100 % | 100 % | −5,114 | +2,761 | 82 % | −143 |
| (f) 2×, ≤ 5× acct, ≤ 0.1 % vol | 74 / 625 | 2.86 / 4.18 | +0.10 | +483 | −6.8 | 9 % | 1 % | +864 | +1,245 | 1,734 | −0.20 | −5,316 | −23.1 | 100 % | 100 % | −1,526 | +2,264 | 35 % | −115 |
| (f) 1×, ≤ 0.1 % vol | 74 / 625 | 2.86 / 4.18 | +0.11 | +474 | −6.8 | 10 % | 1 % | +849 | +1,224 | 1,730 | −0.19 | −5,224 | −22.8 | 100 % | 100 % | −1,516 | +2,193 | 35 % | −112 |
| **(f) 1×, ≤ 0.05 % vol — first recommendation (§6)** | 74 / 625 | 1.43 / 2.09 | +0.17 | +481 | −3.1 | 0 % | 0 % | +660 | +839 | 874 | −0.14 | −2,015 | −14.6 | 100 % | 93 % | +47 | +2,109 | 14 % | −112 |
| (f) 0.5×, ≤ 0.1 % vol | 74 / 625 | 2.44 | +0.14 | +304 | −4.7 | 4 % | 0 % | +525 | +747 | 1,367 | −0.14 | −2,567 | −12.2 | 100 % | 95 % | −733 | +1,101 | 9 % | −52 |
| (f) 2×, ≤ 3× acct, ≤ 0.1 % vol | 74 / 625 | 2.86 / 3.00 | +0.13 | +338 | −5.8 | 9 % | 0 % | +606 | +874 | 1,544 | −0.16 | −3,326 | −13.8 | 100 % | 98 % | −999 | +1,327 | 19 % | −65 |
| (f) 2× if mcap > $102M else 1×, ≤ 5× acct | 70 / 625 | 4.88 | +0.03 | +135 | −10.4 | 100 % | 69 % | +856 | +1,577 | 3,423 | −0.25 | −9,263 | −25.8 | 100 % | 100 % | −3,501 | +2,261 | 70 % | −115 |

The mcap cut points ($40M / $102M) are the backtest's own terciles, set before looking at outcomes and frozen. A fill with unscored mcap gets 1×.

**Master per batch, $ at the live account, central impact** (TOTAL rows; the backtest split by regime):

| batch | N | (a) today | (b) 1× | (b2) 0.5× | (c1) mcap<$40M 2× | (c2) mcap>$102M 2× | (d) liq 0.05 % | (e) ≤ 5× acct | **(f) 1× ≤ 0.05 %** | (f) 1× ≤ 0.1 % |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 6 | +139 | +112 | +73 | +134 | +105 | +125 | +114 | +117 | +136 |
| B2 | 17 | +136 | +217 | +162 | −150 | +278 | +249 | +218 | +238 | +273 |
| B3 | 17 | −797 | −240 | −63 | −473 | −241 | −18 | −252 | −18 | −86 |
| B5–B9 | 7 | +146 | +131 | +85 | +95 | +164 | +51 | +133 | +51 | +80 |
| B10 | 2 | −240 | −100 | −41 | −240 | −100 | −10 | −103 | −10 | −26 |
| B12–B13 | 7 | +80 | +88 | +60 | +88 | +54 | +61 | +89 | +61 | +92 |
| B15–B17 | 8 | +9 | +58 | +51 | +32 | +69 | +55 | +57 | +55 | +65 |
| B18 | 4 | −304 | −123 | −47 | −285 | −123 | −24 | −127 | −24 | −62 |
| B19–B21 | 6 | −57 | −12 | +10 | −42 | −27 | +10 | −13 | +10 | +1 |
| **MASTER TOTAL** | **74** | **−888** | **+132** | **+289** | **−841** | **+179** | **+499** | **+116** | **+481** | **+474** |
| BT 2025 BULL | 223 | −7,874 | −2,238 | −516 | −4,107 | −3,727 | −151 | −2,350 | +101 | −1,078 |
| BT 2026-H1 BEAR | 282 | −14,722 | −5,048 | −1,703 | −8,557 | −6,676 | −1,848 | −5,252 | −1,610 | −3,065 |
| BT 2026-H2 BULL | 120 | −5,847 | −1,898 | −575 | −4,097 | −2,615 | −523 | −1,981 | −506 | −1,081 |
| **BT TOTAL** | **625** | **−28,443** | **−9,185** | **−2,794** | **−16,762** | **−13,018** | **−2,522** | **−9,583** | **−2,015** | **−5,224** |
| **LIVE (KMNO)** | **1** | −254 | −112 | −52 | −112 | −254 | −179 | −115 | −112 | −112 |

The §6 rule beats today in 12 of 18 master batches. The other 6 are ≈ ties of a few dollars, all small winners.
It beats today in all 3 backtest regimes and on live.

**Under zero impact (paper) today's 2× makes the most money in every block.** The whole case for shrinking rests on two things: (i) impact is
real (KMNO measured it), and (ii) ruin. Under zero impact, today's 2× still carries backtest P(DD ≥ 50 %) of 90 % vs 14 % for the
recommended rule.

## 6. Recommendation (one rule) — superseded by §8 (adds a ≈ 3× account cap)

**Rule: a fade's notional = min(1× ticket, 0.05 % of the pair's 24 h volume).**
- Config: `spike_fade_invest_mult` 2.0 → 1.0 and `spike_lowvol_liq_cap_pct` 0.5 → 0.05.
- Side effect: the spike cap also sizes the SPIKE_CHASE probe, which is enabled and probe-sized. Bounce and full chase are off.
- This is one sizing rule with two legs:
  - the **account leg** bounds ruin: a fade is never above ≈ 4.9× the account;
  - the **volume leg** bounds market impact and shrinks the thin-pair fades, the weakest cohort in both blocks.
- If the operator wants a single knob instead:
  - (d) `spike_lowvol_liq_cap_pct` 0.05 alone is the runner-up on profit (master +$499), but it leaves a KMNO-type fade at 7.3× the account;
  - (b) invest 1.0 alone is clearly weaker (master +$132, backtest P50 100 %).

**Evidence strength.**
- Moderate for "smaller is better": every block, every impact assumption on ruin, and both blocks on low-volume weakness.
- Weak for the exact 0.05: chosen from a 3-value grid on these same fills, and the impact model is calibrated on one live fill plus one calm
  snapshot.
- The mcap variants are refuted for sizing: they are worse than plain volume or account caps in every block, and need new plumbing.

**Expected effect, at $1,225.**
- Master: −$888 → +$481 over 74 fades (≈ 11 weeks). After the 30–50 % in-sample haircut, ≈ +$240–340.
- Backtest loss: cut by 93 %.
- Worst single fade: −23.7 % → −3.1 % of the account (master), −53 % → −15 % (backtest).
- P(DD ≥ 50 %/yr): master 100 % → 0 %; backtest 100 % → 93 % with central impact, 90 % → 14 % with zero impact.
- Cost if impact were zero: −72 % of master $, −53 % of backtest $.
- Self-scaling: as the account grows, the volume leg binds more, so the fade's share of the account keeps shrinking.

**🔒 Pre-committed gates (frozen now; cohort = SPIKE_FADE opened after the deploy, live, with `entry_slippage_pct` / `exit_slippage_pct`):**
- **Step-up (never straight back to 2×).** Raise the cap one step to 0.1 % (not the 1× leg) only when all three hold at N ≥ 20 new-size
  fades:
  - live as-traded avg ≥ +0.25 % net;
  - median round-trip |entry_slippage| + |exit_slippage| ≤ 0.15 %;
  - no single fade worse than −5 % of the account.
- **Model check at N ≥ 10.** If the measured round-trip slippage is ≤ ½ of the model's prediction for those fills, re-price this table at
  ½ impact. The recommended rule still wins on ruin there: backtest P50 56 %, $ +47.
- **Sleeve flag.** Rolling 15 new-size fades with Σ P&L % < 0, OR any fade worse than −5 % of the account → open the SPIKE_FADE sleeve
  review under the sleeve-kill checklist. That route is a sleeve review, not a size-up and not a further size cut by reflex.
  (Backtest after impact is negative at every size tested.)

## 7. Inputs, method, blind spots

- **Master.** `reports/MASTER_POOL_stacked.csv` STACK 2026-10-16a (= 2026-10-19a for fades), SPIKE_FADE ∧ `stack_keep`: 74 fills, `stack_pct` / `stack_pnl`. Paper,
  so zero impact and no stop gap; stops are counted where `stack_pct` ≤ −1.49.
- **Backtest.** `reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv` (seed 1, today's fade rules, 1-minute stepped path). The 12 worst stops were
  re-priced on Binance aggTrades at first-tick-through + 1 s:
  - FUSDT −7.71 → −4.87, DAM −5.63 → −3.57, PUMPBTC −4.68 → −2.72;
  - PARTI −3.84 → −4.34, WAXP −2.68 → −2.72, MAV −2.65 → −2.98;
  - backtest avg +0.043 → +0.060.
  - The rest stays on 1-minute pricing (not tick-verified).
- **Live.** `~/Downloads/scalpars_orders_live_2026-10-19_02-59-03.csv` + B22 archive. SPIKE_FADE = KMNO only (N = 1). Its rule
  counterfactuals use its own measured order-book ladder.
- **Mcap rebuild.** Uses today's circulating supply, so 2025 mcaps are overstated for coins whose supply has grown since. The impact model
  uses mcap only where scored.
- **Impact model limits.**
  - The snapshot is one calm Monday-03:00 book. The stress multipliers come from one fill.
  - The volume-inflation factor 1.8 is the median stamp ÷ today's volume.
  - The model under-predicts KMNO (0.34 vs 0.58), so the "central" case is not pessimistic for pumped-volume coins.
  - Not modelled: maker or limit entries, partial fills, impact on the trailing exits' timing, funding, and the gross cap with other sleeves.
- **Bootstrap.** Fade sleeve alone, 2,000 paths. Master's 78-day calendar resampled to a year is optimistic (74 near-flawless paper fills);
  read it as a ranking of rules, not a forecast.
- **Files.**
  - `reports/FADE_SIZE_MCAP_2026-10-19_rows.csv` (every fill with mcap, volume, impact and per-rule inputs)
  - `_rules.csv`
  - `_perbatch.csv`
  - `_depth_snapshot.csv`
  - `_tail_ticks.csv`
  - scripts + Info-panel supply in `reports/fade_size_mcap_2026-10-19/` (run from that folder: `build.py` → `analyse.py` → `rules.py` /
    `perbatch.py`).


---

## 8. Follow-up: is the risk in HIGHER-mcap fades? (operator, 2026-10-19) — final recommendation

Operator: *"spike fade will need to come with recommendations … keep risk under control on HIGHER mcap trades, check deeply master batch."*
Same rules as above: research only, today's stack (`MASTER_POOL_stacked.csv` STACK **2026-10-19a** — the 74 kept fades are identical to
2026-10-16a), $ at the $1,225 live account with the central impact model unless marked paper, mcap rebuilt (unscored kept apart).

### 8.0 Plain English

- **The data says the opposite of "higher mcap = tail risk".**
  - Master: every fade on a coin ≥ $200M won: **9 / 9, worst +0.27 %, avg +0.46 %**. All 8 master losers sat below $100M mcap **and** below
    $4.2M of 24 h volume (PORTAL, SXT, KOMA, BEL, ENSO, FORM, PLUME, AIN).
  - Backtest: ≥ $200M fades (110) have the **smallest** stop gaps (0.05 % vs 0.25 % at $50–200M), 1 fill below −2 % (vs 8 of 210 at
    $50–200M) and the best average at ≥ $500M (+0.16 %).
  - Order books: the higher the mcap, the deeper the book (median depth ±1 %: $50k < $50M → $246k ≥ $500M).
- **KMNO was the exception, not the rule.** At $332M it should have had a deep book. In the snapshot its depth is **in the bottom 3 % of
  all ≥ $200M fade coins** ($55k within ±1 %, the depth of a median < $50M coin). What hurt was a $11.9k ticket (9.75× the account) pushed
  through a thin book right after a squeeze. That is ticket size vs book depth, which a mcap label does not see.
- **Why it can still *feel* like high-mcap risk.** High-mcap coins trade more volume, so the 0.5 % volume cap never shrinks them: they always
  get the full ticket, and a stop there is the biggest $ loss. The fix is to cap the ticket against the **account**, which is exactly what
  bites on those trades. In the backtest a 3× account cap binds on 54 % of ≥ $200M fades vs 30 % of the rest, with no mcap data needed.
- **"1× above $200M, 2× below" is the wrong way round.** It halves the proven winners and keeps the losers (all small-cap) at 2×. On the
  master it is *worse than today* (−$1,079 vs −$888). An mcap ladder (2 → 0.5× as mcap rises) is also worse than today on master.

**FINAL RECOMMENDATION: fade ticket = `min(0.6× ticket, 0.05 % of 24 h volume)`.**
- That equals the §6 rule plus a hard cap of ≈ 3× the account (0.6 × 4.875 = 2.93×). It uses only fields that exist and needs no mcap:
  `spike_fade_invest_mult` 2.0 → **0.6**, `spike_lowvol_liq_cap_pct` 0.5 → **0.05**. The cap also sizes the enabled SPIKE_CHASE probe.
- On the $1,225 account: max fade ≈ $3.6k, median ≈ $1.8–2.6k (1.4–2.1× the account).
- vs today:
  - master **−$888 → +$406** (≈ +$200–280 after a 30–50 % haircut);
  - backtest −$28.4k → −$1.6k;
  - KMNO −$254 (−20.7 % of the account) → **−$65 (−5.3 %)**.
- Worst single fade: master −23.7 % → −3.1 % of the account; backtest −53 % → −11 %.
- P(drawdown ≥ 50 % in a year): master 100 % → 0 %; backtest 100 % → 85 % with central impact, 89 % → 6 % in a paper (zero-impact) world.
- vs the §6 rule it gives up $75 on master (and $139 of paper $), and gains $404 on the backtest, $47 on KMNO and a backtest P50 of 6 % vs
  13 % in the paper world.
- **The more conservative option** is (e′), ≤ 2× the account (`invest_mult` 0.4): master +$307, KMNO −$41, backtest paper P50 2 %. Use it if
  the operator prefers the tightest ruin control over the last ~$100 of master profit.

### 8.1 Rules side by side (master + backtest + KMNO)

Unscored mcap → the smaller size (fail-closed).

**Plumbing cost for any mcap rule.** The engine has no mcap at fade time for most fade pairs. You would need:
- a one-pair `mcap_service.request_pair` fetch at the spike trigger, plus a wait or a fail-closed size;
- a stamp, config field, UI field and tests (D11);
- the Info-panel endpoint is undocumented (DL 124).

| rule | needs mcap at fade time? | MASTER $ (central / ½ / paper) | MASTER worst % acct | MASTER P30 / P50 | BT $ (central / ½ / paper) | BT worst % acct | BT P50 central / paper | KMNO $ (% acct) | max notional × acct |
|---|---|---|---|---|---|---|---|---|---|
| today: 2×, liq 0.5 % | no | −888 / +1,042 / +2,972 | −23.7 | 100 % / 100 % | −28,443 / −11,966 / +4,512 | −53.4 | 100 % / 89 % | −254 (−20.7) | 9.75 |
| (a) §6 rule: min(1×, 0.05 % vol) | no | +481 / +660 / +839 | −3.1 | 0 % / 0 % | −2,015 / +47 / +2,109 | −14.6 | 94 % / 13 % | −112 (−9.1) | 4.88 |
| (b) 1× if mcap ≥ $200M else 2× (unscored → 1×) | yes (new plumbing + fail-closed) | −1,079 / +662 / +2,403 | −23.7 | 100 % / 100 % | −24,406 / −10,728 / +2,950 | −53.4 | 100 % / 88 % | −112 (−9.1) | 9.75 |
| (b′) 1× if mcap ≥ $300M else 2× (unscored → 1×) | yes (new plumbing + fail-closed) | −1,067 / +726 / +2,519 | −23.7 | 100 % / 100 % | −25,506 / −11,289 / +2,928 | −53.4 | 100 % / 90 % | −112 (−9.1) | 9.75 |
| (c) 1× flat | no | +132 / +847 / +1,562 | −10.4 | 100 % / 67 % | −9,185 / −3,471 / +2,244 | −25.8 | 100 % / 69 % | −112 (−9.1) | 4.88 |
| (d) mcap ladder 2 / 1.5 / 1 / 0.5× (< 50M / < 200M / < 500M / ≥ 500M; unscored 1×) | yes (new plumbing + fail-closed) | −940 / +585 / +2,111 | −23.7 | 100 % / 100 % | −21,017 / −9,282 / +2,453 | −39.4 | 100 % / 88 % | −112 (−9.1) | 9.75 |
| (d′) inverse ladder 0.5 / 1 / 1.5× (< 50M / < 200M / ≥ 200M; unscored 0.5×) | yes (new plumbing + fail-closed) | +372 / +853 / +1,334 | −9.4 | 85 % / 18 % | −6,498 / −2,336 / +1,827 | −25.8 | 100 % / 67 % | −180 (−14.7) | 7.31 |
| **(e) §6 rule + notional ≤ 3× account = FINAL** | no | +406 / +553 / +700 | −3.1 | 0 % / 0 % | −1,611 / −101 / +1,409 | −11.3 | 85 % / 6 % | −65 (−5.3) | 3.00 |
| (e′) §6 rule + notional ≤ 2× account | no | +307 / +420 / +534 | −3.1 | 0 % / 0 % | −1,247 / −185 / +877 | −9.1 | 64 % / 2 % | −41 (−3.4) | 2.00 |
| (f) §6 rule + 0.5× if mcap ≥ $200M (unscored 1×) | yes (new plumbing + fail-closed) | +466 / +641 / +815 | −3.1 | 0 % / 0 % | −2,018 / −104 / +1,811 | −14.6 | 94 % / 11 % | −52 (−4.2) | 4.88 |

Reading:
- **Every mcap-based rule that puts 2× anywhere** (b, b′, d) keeps a ≥ 9.75× account ticket on some fades. That gives a −24 % to −53 %
  single-fade worst case and P50 ≈ 100 %.
- **(f)**, §6 + half size on ≥ $200M, is about the same as (a) on master and backtest. It is better than (a) on KMNO, but worse than the
  mcap-free account cap (e) on KMNO, the backtest and ruin. It also costs the plumbing.
- **The inverse ladder (d′)** confirms the depth physics: big on deep books beats big on thin books. But it still ranks below (e) on every
  ruin metric.

### 8.2 Master by mcap and by volume ($ at $1,225; final = 8.0 rule)

`gap` is not measurable on paper fills (master stops fill exactly at −1.5).

| mcap | N | WR | avg % | worst % | stops | gap mean | ≤ −2 % | today 2× $ | §6 rule $ | (b) 1× ≥ $200M $ | (c) 1× flat $ | **final rule $** | today paper $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < $50M | 33 | 82 % | +0.20 | −1.50 | 5 | – | 0 | −1,336 | +77 | −1,336 | −382 | **+51** | +682 |
| $50–200M | 27 | 93 % | +0.38 | −1.50 | 2 | – | 0 | −111 | +140 | −111 | +145 | **+134** | +1,148 |
| $200–300M | 4 | 100 % | +0.48 | +0.27 | 0 | – | 0 | +73 | +24 | +61 | +61 | **+24** | +231 |
| $300–500M | 3 | 100 % | +0.37 | +0.28 | 0 | – | 0 | +29 | +23 | +31 | +31 | **+22** | +133 |
| ≥ $500M | 2 | 100 % | +0.56 | +0.54 | 0 | – | 0 | +78 | +32 | +48 | +48 | **+23** | +134 |
| cum ≥ $200M | 9 | 100 % | +0.46 | +0.27 | 0 | – | 0 | +181 | +79 | +140 | +140 | **+69** | +498 |
| cum ≥ $300M | 5 | 100 % | +0.45 | +0.28 | 0 | – | 0 | +108 | +54 | +79 | +79 | **+45** | +267 |
| cum ≥ $500M | 2 | 100 % | +0.56 | +0.54 | 0 | – | 0 | +78 | +32 | +48 | +48 | **+23** | +134 |
| unscored | 5 | 100 % | +1.08 | +0.24 | 0 | – | 0 | +379 | +184 | +229 | +229 | **+151** | +643 |
| **TOTAL** | **74** | 89 % | +0.36 | −1.50 | 7 | – | 0 | −888 | +481 | −1,079 | +132 | **+406** | +2,972 |
| *by 24 h volume* | | | | | | | | | | | | | |
| vol < $3M | 27 | 85 % | +0.24 | −1.50 | 3 | – | 0 | −819 | +26 | −836 | −195 | **+26** | +675 |
| vol $3–6M | 27 | 85 % | +0.22 | −1.50 | 4 | – | 0 | −829 | +18 | −780 | −194 | **+18** | +639 |
| vol $6–12M | 17 | 100 % | +0.69 | +0.22 | 0 | – | 0 | +597 | +335 | +392 | +421 | **+293** | +1,376 |
| vol ≥ $12M | 3 | 100 % | +0.79 | +0.54 | 0 | – | 0 | +164 | +100 | +145 | +100 | **+68** | +281 |
| **TOTAL** | **74** | 89 % | +0.36 | −1.50 | 7 | – | 0 | −888 | +481 | −1,079 | +132 | **+406** | +2,972 |

### 8.3 Backtest by mcap and by volume (same columns)

| mcap | N | WR | avg % | worst % | stops | gap mean | ≤ −2 % | today 2× $ | §6 rule $ | (b) 1× ≥ $200M $ | (c) 1× flat $ | **final rule $** | today paper $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < $50M | 217 | 73 % | +0.01 | −2.98 | 52 | 0.07 | 1 | −13,824 | −1,640 | −13,824 | −4,754 | **−1,208** | +172 |
| $50–200M | 210 | 73 % | +0.05 | −4.87 | 47 | 0.25 | 8 | −9,123 | −46 | −9,123 | −2,972 | **−298** | +1,265 |
| $200–300M | 39 | 67 % | −0.03 | −2.27 | 11 | 0.08 | 1 | −1,733 | −30 | −633 | −633 | **−84** | −80 |
| $300–500M | 34 | 76 % | +0.05 | −1.70 | 7 | 0.04 | 0 | −972 | −156 | −307 | −307 | **−66** | +222 |
| ≥ $500M | 37 | 81 % | +0.16 | −1.52 | 7 | -0.00 | 0 | −335 | +109 | −6 | −6 | **+75** | +691 |
| cum ≥ $200M | 110 | 75 % | +0.06 | −2.27 | 25 | 0.05 | 1 | −3,040 | −77 | −945 | −945 | **−75** | +832 |
| cum ≥ $300M | 71 | 79 % | +0.11 | −1.70 | 14 | 0.02 | 0 | −1,307 | −47 | −313 | −313 | **+9** | +912 |
| cum ≥ $500M | 37 | 81 % | +0.16 | −1.52 | 7 | -0.00 | 0 | −335 | +109 | −6 | −6 | **+75** | +691 |
| unscored | 88 | 70 % | +0.21 | −3.57 | 20 | 0.32 | 6 | −2,456 | −252 | −513 | −513 | **−30** | +2,242 |
| **TOTAL** | **625** | 73 % | +0.06 | −4.87 | 144 | 0.16 | 16 | −28,443 | −2,015 | −24,406 | −9,185 | **−1,611** | +4,512 |
| *by 24 h volume* | | | | | | | | | | | | | |
| vol < $3M | 146 | 70 % | +0.04 | −4.87 | 34 | 0.21 | 3 | −8,272 | −216 | −7,672 | −2,869 | **−216** | +723 |
| vol $3–6M | 209 | 72 % | −0.05 | −3.57 | 49 | 0.13 | 5 | −13,045 | −1,047 | −10,913 | −4,650 | **−1,047** | −1,143 |
| vol $6–12M | 177 | 74 % | +0.12 | −4.34 | 42 | 0.14 | 4 | −5,804 | −665 | −5,467 | −1,579 | **−475** | +2,605 |
| vol ≥ $12M | 93 | 80 % | +0.21 | −2.72 | 19 | 0.20 | 4 | −1,322 | −87 | −354 | −87 | **+125** | +2,327 |
| **TOTAL** | **625** | 73 % | +0.06 | −4.87 | 144 | 0.16 | 16 | −28,443 | −2,015 | −24,406 | −9,185 | **−1,611** | +4,512 |

### 8.4 Live (KMNO)

| mcap | N | WR | avg % | worst % | stops | gap mean | ≤ −2 % | today 2× $ | §6 rule $ | (b) 1× ≥ $200M $ | (c) 1× flat $ | **final rule $** | today paper $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| < $50M | 0 | | | | | | | | | | | | |
| $50–200M | 0 | | | | | | | | | | | | |
| $200–300M | 0 | | | | | | | | | | | | |
| $300–500M | 1 | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |
| ≥ $500M | 0 | | | | | | | | | | | | |
| cum ≥ $200M | 1 | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |
| cum ≥ $300M | 1 | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |
| cum ≥ $500M | 0 | | | | | | | | | | | | |
| unscored | 0 | | | | | | | | | | | | |
| **TOTAL** | **1** | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |
| *by 24 h volume* | | | | | | | | | | | | | |
| vol < $3M | 0 | | | | | | | | | | | | |
| vol $3–6M | 0 | | | | | | | | | | | | |
| vol $6–12M | 0 | | | | | | | | | | | | |
| vol ≥ $12M | 1 | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |
| **TOTAL** | **1** | 0 % | −2.13 | −2.13 | 1 | 0.63 | 1 | −254 | −112 | −112 | −112 | **−65** | −184 |

### 8.5 Master per batch

| batch | N | wins | avg % | worst % | median mcap $M | today 2× $ | final rule $ | today paper $ |
|---|---|---|---|---|---|---|---|---|
| B1 | 6 | 5 | +0.61 | −0.24 | 38 | +139 | +93 | +439 |
| B2 | 17 | 15 | +0.55 | −1.50 | 64 | +136 | +191 | +1,088 |
| B3 | 17 | 14 | +0.12 | −1.50 | 55 | −797 | −18 | +239 |
| B5 | 1 | 1 | +0.53 | +0.53 | 208 | +19 | +6 | +63 |
| B6 | 1 | 1 | +0.79 | +0.79 | 280 | +53 | +9 | +95 |
| B7 | 2 | 2 | +0.39 | +0.34 | 30 | −14 | +11 | +92 |
| B8 | 1 | 1 | +0.43 | +0.43 | 35 | −6 | +5 | +51 |
| B9 | 2 | 2 | +0.86 | +0.69 | 75 | +94 | +19 | +192 |
| B10 | 2 | 1 | −0.38 | −1.50 | 26 | −240 | −10 | −87 |
| B12 | 3 | 3 | +0.57 | +0.22 | 118 | +81 | +39 | +204 |
| B13 | 4 | 4 | +0.38 | +0.28 | 311 | −1 | +22 | +181 |
| B15 | 1 | 1 | +0.75 | +0.75 | 121 | +52 | +21 | +90 |
| B16 | 4 | 4 | +0.41 | +0.24 | 19 | −24 | +20 | +171 |
| B17 | 3 | 3 | +0.31 | +0.27 | 99 | −19 | +11 | +105 |
| B18 | 4 | 3 | −0.11 | −1.50 | 33 | −304 | −24 | −80 |
| B19 | 3 | 3 | +0.32 | +0.25 | 59 | −33 | +9 | +79 |
| B20 | 2 | 2 | +0.22 | +0.17 | 16 | −11 | +0 | +22 |
| B21 | 1 | 1 | +0.29 | +0.29 | 23 | −12 | +2 | +29 |
| **TOTAL** | **74** | **66** | **+0.36** | −1.50 | 53 | **−888** | **+406** | **+2,972** |

Losses and cost are concentrated in three batches:
- **B3** (Aug 12–21): 3 stops, PLUME / FORM / PORTAL;
- **B18** (Oct 7): KOMA;
- **B10** (Sep 20): ENSO.

Under today's 2× those three batches cost −$1,341; under the final rule −$52. No batch is negative by more than $24 under the final rule.

### 8.6 Master per-pair concentration (8 worst under today's 2×)

| pair | fills | Σ % | mcap $M | vol $M | today 2× $ | today paper $ | final rule $ |
|---|---|---|---|---|---|---|---|
| PORTALUSDT | 2 | −1.14 | 10 | 2.5 | −316 | −143 | −26 |
| SXTUSDT | 2 | −1.16 | 17 | 2.4 | −307 | −141 | −20 |
| KOMAUSDT | 1 | −1.50 | 8 | 3.5 | −290 | −179 | −32 |
| BELUSDT | 1 | −1.50 | 11 | 4.2 | −280 | −179 | −38 |
| ENSOUSDT | 1 | −1.50 | 28 | 2.3 | −266 | −175 | −20 |
| FORMUSDT | 1 | −1.50 | 82 | 2.8 | −254 | −179 | −24 |
| PLUMEUSDT | 1 | −1.50 | 79 | 4.2 | −249 | −179 | −36 |
| AINUSDT | 1 | −0.24 | 31 | 2.2 | −83 | −26 | −4 |
| **TOTAL (all 63 pairs)** | **74** | **+26.35** | | | **−888** | **+2,972** | **+406** |

- The 8 losing fades come from 8 pairs, and no pair has two stops. PORTAL and SXT each had a winner plus a loser.
- The top 5 pairs by paper profit (VANRY, PROM, TLM, XEC, ZIL) hold 38 % of the master's paper $.
- No single-pair blacklist explains the losses. The losing cohort is "small cap ∧ thin volume", already the shrink target of the 0.05 %
  volume leg.

### 8.7 Hypothesis test: high mcap vs ticket size vs book depth

| question | evidence | verdict |
|---|---|---|
| Do high-mcap fades lose more often? | master ≥ $200M 9/9 wins vs 82–93 % below; backtest ≥ $200M WR 75 % vs 73 % | **no** |
| Do they have deeper stop gaps or tails? | backtest gap mean ≥ $200M 0.05 % vs 0.25 % (50–200M) / 0.07 % (< 50M); fills ≤ −2 %: 1/110 vs 9/427 | **no — smaller** |
| Do they have thinner books (more own impact)? | depth ±1 % median $124k ($200–300M) / $246k (≥ $500M) vs $50k (< $50M); $12k impact 0.11 % / 0.07 % vs 0.19 % | **no — deeper** |
| Was KMNO a high-mcap problem? | KMNO depth $55k = bottom 3 % of ≥ $200M coins; own impact 0.58 % on a 9.75× account ticket | **no — a thin-book coin at a full ticket** |
| What does bound the KMNO-type loss? | ticket ÷ account (KMNO −20.7 % → −5.3 % at ≤ 3×) and ticket ÷ depth (impact falls as size^0.55) | **ticket size** |

A live pre-trade **book-depth** check would be the physically right gate, e.g. ticket ≤ x % of the ±1 % depth. It needs a depth read
(weight 5–10) on the open path, i.e. new plumbing. It is noted as a candidate only, to be considered once live fades at the new size show
whether impact still matters.

### 8.8 🔒 Pre-committed gates for the final rule (frozen now)

Cohort = live SPIKE_FADE opened after the deploy. These gates replace the §6 gates; the §6 model check is kept.

- **Step-up (one step only, never straight back to 2×).** At N ≥ 20 new-size fades, if all three hold, raise to `invest_mult` 1.0
  (≤ 4.9× the account) and keep the 0.05 % volume leg:
  - live as-traded avg ≥ +0.25 % net;
  - median round-trip |entry_slippage| + |exit_slippage| ≤ 0.15 %;
  - no fade worse than −4 % of the account.
- **High-mcap watch (the operator's concern, tracked, no rule).** At every batch review report new-size fades split by rebuilt mcap
  ≥ $200M vs < $200M: N, WR, avg %, worst % of the account, round-trip slippage. If ≥ $200M fades reach N ≥ 10 with avg < 0 or any one
  worse than −4 % of the account, re-open an mcap or depth rule. That is the trigger for the plumbing cost; never before it.
- **Model check at N ≥ 10.** If the measured round-trip slippage is ≤ ½ of the model's prediction, re-price this section at ½ impact.
  The rule still ranks first or second there: master +$553, backtest −$101, P50 central 0 % / 85 %.
- **Sleeve flag.** Rolling 15 new-size fades with Σ P&L % < 0, or any fade worse than −4 % of the account → SPIKE_FADE sleeve review under
  the sleeve-kill checklist. That is a sleeve review, not a reflex further size cut.

### 8.9 Every master fade (today's stack; $ at the $1,225 account, central impact)

The final rule's ticket is shown for each fade. Mcap is rebuilt; peak / trough are the master's own columns.

| batch | opened (UTC) | pair | mcap $M | 24h vol $M | peak % | trough % | exit | stack_pct | today 2× $ | final rule notional | final rule $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 07-28 08:14 | PROMUSDT | 39 | 14.1 | +1.56 | −0.09 | HARD_TP_LADDER L2 | +1.16 | +94 | $3,675 | +35 |
| B1 | 07-29 11:50 | METUSDT | 99 | 5.3 | +1.09 | −0.65 | HARD_TP_LADDER L1 | +0.70 | +39 | $2,628 | +14 |
| B1 | 07-29 15:35 | PROMUSDT | 38 | 7.7 | +1.89 | −0.24 | HARD_TP_LADDER L2 | +0.52 | +13 | $3,675 | +11 |
| B1 | 07-30 06:55 | TWTUSDT | 161 | 3.4 | +0.55 | −0.09 | RUNNER_TRAIL | +0.36 | −1 | $1,693 | +4 |
| B1 | 07-31 09:27 | TLMUSDT | 11 | 8.9 | +1.67 | −0.09 | HARD_TP_LADDER L2 | +1.15 | +76 | $3,675 | +32 |
| B1 | 07-31 12:44 | AINUSDT | 31 | 2.2 | +0.14 | −0.71 | STOP_LOSS L1 | −0.24 | −83 | $1,085 | −4 |
| B2 | 08-01 00:00 | EGLDUSDT | 83 | 2.2 | +1.03 | −0.09 | RUNNER_TRAIL | +0.88 | +49 | $1,089 | +8 |
| B2 | 08-01 14:01 | EGLDUSDT | 85 | 2.7 | +0.53 | −0.09 | RUNNER_TRAIL | +0.39 | −6 | $1,329 | +3 |
| B2 | 08-02 13:07 | SXTUSDT | 17 | 2.3 | +0.46 | −0.68 | RUNNER_TRAIL | +0.34 | −28 | $1,126 | +2 |
| B2 | 08-03 02:05 | HOLOUSDT | 53 | 3.4 | +0.39 | −0.70 | STOP_LOSS L1 | +0.79 | +40 | $1,689 | +11 |
| B2 | 08-03 02:36 | PIPPINUSDT | 16 | 2.2 | +0.66 | −0.09 | RUNNER_TRAIL | +0.47 | −13 | $1,089 | +3 |
| B2 | 08-03 03:35 | VANRYUSDT | unscored | 9.2 | +4.83 | −0.09 | RUNNER_TRAIL | +3.92 | +426 | $3,675 | +137 |
| B2 | 08-04 06:04 | SNXUSDT | 75 | 4.5 | +0.74 | −0.60 | RUNNER_TRAIL | +0.60 | +23 | $2,249 | +10 |
| B2 | 08-04 07:14 | 1000LUNCUSDT | 278 | 5.1 | +0.47 | −0.41 | RUNNER_TRAIL | +0.35 | +5 | $2,571 | +5 |
| B2 | 08-05 22:28 | WLFIUSDT | 1,711 | 17.2 | +0.65 | −0.44 | RUNNER_TRAIL | +0.54 | +44 | $3,675 | +17 |
| B2 | 08-06 00:07 | 1000XECUSDT | 130 | 3.6 | +1.52 | −0.09 | RUNNER_TRAIL | +1.28 | +108 | $1,788 | +21 |
| B2 | 08-06 00:17 | RESOLVUSDT | 9 | 8.1 | +0.65 | −0.09 | RUNNER_TRAIL | +0.25 | −34 | $3,675 | −1 |
| B2 | 08-06 19:39 | SXTUSDT | 18 | 2.6 | +0.23 | −0.26 | SPIKE_LOCK L1 | −1.50 | −280 | $1,280 | −22 |
| B2 | 08-07 00:14 | MIRAUSDT | 15 | 14.4 | +0.85 | −0.09 | RUNNER_TRAIL | +0.65 | +26 | $3,675 | +16 |
| B2 | 08-07 02:31 | QNTUSDT | 720 | 2.5 | +0.68 | −0.49 | RUNNER_TRAIL | +0.58 | +34 | $1,275 | +6 |
| B2 | 08-07 22:58 | SENTUSDT | 123 | 2.2 | +0.36 | −0.46 | SPIKE_LOCK L1 | +0.27 | −14 | $1,079 | +2 |
| B2 | 08-08 21:54 | SWARMSUSDT | 8 | 3.4 | +1.28 | −0.09 | RUNNER_TRAIL | +0.95 | +36 | $1,700 | +12 |
| B2 | 08-09 09:44 | BELUSDT | 11 | 4.2 | +0.00 | −0.69 | STOP_LOSS L1 | −1.50 | −280 | $2,100 | −38 |
| B3 | 08-12 13:44 | RECALLUSDT | 13 | 3.3 | +0.83 | −0.44 | RUNNER_TRAIL | +0.59 | −1 | $1,653 | +6 |
| B3 | 08-12 23:48 | TAKEUSDT | 17 | 2.7 | +1.57 | −0.09 | HARD_TP_LADDER L2 | +1.19 | +71 | $1,334 | +13 |
| B3 | 08-13 09:00 | PLUMEUSDT | 79 | 4.2 | +0.08 | −1.51 | STOP_LOSS L1 | −1.50 | −249 | $2,083 | −36 |
| B3 | 08-13 09:42 | SQDUSDT | 55 | 8.5 | +0.95 | −0.24 | RUNNER_TRAIL | +0.39 | +1 | $3,675 | +7 |
| B3 | 08-13 18:38 | TWTUSDT | 176 | 2.3 | +0.47 | −0.29 | RUNNER_TRAIL | +0.30 | −9 | $1,137 | +2 |
| B3 | 08-14 09:04 | XPLUSDT | 336 | 8.8 | +0.44 | −0.52 | RUNNER_TRAIL | +0.31 | +5 | $3,675 | +6 |
| B3 | 08-14 12:14 | FORMUSDT | 82 | 2.8 | +0.10 | −1.49 | STOP_LOSS L1 | −1.50 | −254 | $1,393 | −24 |
| B3 | 08-14 22:18 | PORTALUSDT | 10 | 2.0 | +0.63 | −0.09 | RUNNER_TRAIL | +0.36 | −27 | $1,001 | +2 |
| B3 | 08-15 05:49 | PORTALUSDT | 10 | 3.1 | +0.18 | −1.53 | STOP_LOSS L1 | −1.50 | −289 | $1,532 | −28 |
| B3 | 08-15 08:09 | ARCUSDT | 75 | 2.7 | +0.40 | −1.17 | RUNNER_TRAIL | +0.26 | −23 | $1,332 | +2 |
| B3 | 08-15 11:52 | TLMUSDT | 11 | 8.4 | +0.41 | −0.53 | RUNNER_TRAIL | +0.22 | −35 | $3,675 | −2 |
| B3 | 08-15 17:49 | STORJUSDT | unscored | 4.4 | +0.61 | −0.09 | RUNNER_TRAIL | +0.45 | −0 | $2,189 | +6 |
| B3 | 08-16 22:19 | XANUSDT | 26 | 2.2 | +0.58 | −0.81 | RUNNER_TRAIL | +0.37 | −19 | $1,089 | +2 |
| B3 | 08-17 22:59 | 1000RATSUSDT | unscored | 3.1 | +0.75 | −1.05 | RUNNER_TRAIL | +0.43 | −9 | $1,537 | +4 |
| B3 | 08-17 23:01 | CFGUSDT | 113 | 3.2 | +1.27 | −0.09 | RUNNER_TRAIL | +0.90 | +60 | $1,618 | +12 |
| B3 | 08-19 06:12 | SOONUSDT | 118 | 2.7 | +0.58 | −0.50 | RUNNER_TRAIL | +0.37 | −4 | $1,371 | +3 |
| B3 | 08-21 17:24 | CHILLGUYUSDT | 13 | 2.9 | +0.74 | −0.09 | RUNNER_TRAIL | +0.48 | −15 | $1,443 | +4 |
| B5 | 08-27 04:39 | KAIAUSDT | 208 | 2.7 | +0.64 | −0.45 | RUNNER_TRAIL | +0.53 | +19 | $1,344 | +6 |
| B6 | 09-13 07:36 | SYRUPUSDT | 280 | 2.6 | +1.05 | −0.34 | RUNNER_TRAIL | +0.79 | +53 | $1,323 | +9 |
| B7 | 09-15 00:00 | HIVEUSDT | 31 | 10.9 | +0.68 | −0.09 | RUNNER_TRAIL | +0.44 | +4 | $3,675 | +8 |
| B7 | 09-15 00:02 | KNCUSDT | 30 | 4.4 | +0.55 | −0.09 | RUNNER_TRAIL | +0.34 | −18 | $2,181 | +3 |
| B8 | 09-16 22:49 | MERLUSDT | 35 | 3.8 | +0.71 | −1.41 | RUNNER_TRAIL | +0.43 | −6 | $1,885 | +5 |
| B9 | 09-19 00:00 | QTUMUSDT | 103 | 3.5 | +0.89 | −0.09 | RUNNER_TRAIL | +0.69 | +35 | $1,735 | +10 |
| B9 | 09-19 06:23 | API3USDT | 47 | 2.1 | +1.27 | −0.77 | RUNNER_TRAIL | +1.03 | +58 | $1,053 | +9 |
| B10 | 09-20 04:14 | BTRUSDT | 24 | 3.6 | +1.08 | −0.65 | HARD_TP_LADDER L1 | +0.73 | +25 | $1,789 | +10 |
| B10 | 09-20 10:22 | ENSOUSDT | 28 | 2.3 | +0.00 | −1.50 | STOP_LOSS L1 | −1.50 | −266 | $1,163 | −20 |
| B12 | 09-23 09:04 | ZILUSDT | 81 | 6.5 | +1.52 | −0.09 | RUNNER_TRAIL | +1.21 | +101 | $3,247 | +33 |
| B12 | 09-23 23:53 | PROVEUSDT | 118 | 5.9 | +0.35 | −1.53 | STOP_LOSS L1 | +0.22 | −15 | $2,951 | +2 |
| B12 | 09-24 00:00 | COMPUSDT | 225 | 7.5 | +0.44 | −1.20 | RUNNER_TRAIL | +0.27 | −4 | $3,675 | +4 |
| B13 | 09-26 05:23 | 2ZUSDT | 311 | 6.6 | +0.79 | −0.98 | RUNNER_TRAIL | +0.52 | +27 | $3,301 | +12 |
| B13 | 09-26 17:44 | BEAMXUSDT | 113 | 3.1 | +0.50 | −0.50 | RUNNER_TRAIL | +0.32 | −10 | $1,548 | +3 |
| B13 | 09-27 03:31 | 1000RATSUSDT | unscored | 2.7 | +0.63 | −0.09 | RUNNER_TRAIL | +0.40 | −15 | $1,344 | +3 |
| B13 | 09-27 05:03 | 1000LUNCUSDT | 312 | 4.4 | +0.44 | −0.11 | RUNNER_TRAIL | +0.28 | −3 | $2,204 | +3 |
| B15 | 10-01 13:03 | DYDXUSDT | 121 | 10.2 | +1.03 | −0.23 | RUNNER_TRAIL | +0.75 | +52 | $3,675 | +21 |
| B16 | 10-02 05:59 | PARTIUSDT | 19 | 2.6 | +0.84 | −0.95 | RUNNER_TRAIL | +0.56 | +3 | $1,279 | +5 |
| B16 | 10-02 06:44 | AZTECUSDT | unscored | 3.0 | +0.40 | −0.47 | RUNNER_TRAIL | +0.24 | −22 | $1,520 | +1 |
| B16 | 10-03 04:19 | RESOLVUSDT | 11 | 11.6 | +0.48 | −0.23 | RUNNER_TRAIL | +0.29 | −24 | $3,675 | +1 |
| B16 | 10-03 15:36 | INITUSDT | 35 | 7.9 | +0.89 | −0.29 | RUNNER_TRAIL | +0.57 | +19 | $3,675 | +13 |
| B17 | 10-05 00:14 | ZRXUSDT | 109 | 2.7 | +0.46 | −0.17 | RUNNER_TRAIL | +0.30 | −8 | $1,356 | +2 |
| B17 | 10-05 23:51 | VTHOUSDT | 69 | 5.2 | +0.54 | −0.46 | RUNNER_TRAIL | +0.36 | −5 | $2,583 | +5 |
| B17 | 10-06 12:26 | ORDIUSDT | 99 | 11.7 | +0.44 | −0.43 | RUNNER_TRAIL | +0.27 | −6 | $3,675 | +4 |
| B18 | 10-07 11:06 | KOMAUSDT | 8 | 3.5 | +0.08 | −1.50 | STOP_LOSS L1 | −1.50 | −290 | $1,774 | −32 |
| B18 | 10-07 13:33 | FLOWUSDT | 53 | 6.7 | +0.45 | −0.54 | RUNNER_TRAIL | +0.29 | −13 | $3,355 | +3 |
| B18 | 10-07 21:24 | BRETTUSDT | 55 | 2.2 | +0.72 | −0.09 | RUNNER_TRAIL | +0.50 | +5 | $1,075 | +4 |
| B18 | 10-08 02:26 | TURTLEUSDT | 12 | 2.3 | +0.46 | −0.19 | RUNNER_TRAIL | +0.29 | −6 | $1,140 | +1 |
| B19 | 10-09 18:59 | ARXUSDT | 59 | 3.7 | +0.75 | −0.09 | RUNNER_TRAIL | +0.46 | +9 | $1,874 | +6 |
| B19 | 10-10 20:14 | SUPERUSDT | 145 | 2.7 | +0.38 | −0.23 | RUNNER_TRAIL_LATE | +0.26 | −15 | $1,365 | +2 |
| B19 | 10-11 14:23 | AEVOUSDT | 27 | 2.0 | +0.40 | −1.40 | RUNNER_TRAIL | +0.25 | −27 | $1,000 | +1 |
| B20 | 10-12 07:49 | GRIFFAINUSDT | 16 | 4.7 | +0.40 | −0.09 | RUNNER_TRAIL | +0.17 | −8 | $2,336 | −1 |
| B20 | 10-12 09:41 | GRIFFAINUSDT | 16 | 4.5 | +0.43 | −0.23 | RUNNER_TRAIL | +0.28 | −3 | $2,251 | +1 |
| B21 | 10-13 10:07 | PARTIUSDT | 23 | 7.8 | +0.47 | −0.12 | RUNNER_TRAIL | +0.29 | −12 | $3,675 | +2 |
| **TOTAL** | | **74** | | | | | 8 losers (as traded) · 7 stops on today's stack | **+26.35 Σ** | **−888** | median $1,754 | **+406** |

Files for this section, in `reports/`:
- `FADE_SIZE_MCAP_2026-10-19_master_ledger.csv`
- `FADE_SIZE_MCAP_2026-10-19_rules_followup.csv`
- scripts `deep.py`, `deep2.py`, `sec8.py` in `reports/fade_size_mcap_2026-10-19/`


## 9. Operator follow-up: halve only HIGH-mcap fades — threshold sweep (2026-10-19, DECISION_LOG 312)

Same rows as §8 (`FADE_SIZE_MCAP_2026-10-19_rows.csv`), $1,225 account, 2× notional = 9.75 × account, 1× = half, notional also capped at
0.5 % × 24 h volume; impact = (pct0 − pct_live2x) × (notional / 2× notional)^0.55 (simplified form of the §4 model, ±$1–6 vs it);
unknown mcap → 2× (5 master / 88 backtest unscored). Rule: 1× if mcap ≥ X, 2× below. $ totals.

| X | master halved (all won) | master paper / ½ / full | BT halved · WR · avg | BT paper / ½ / full | KMNO (full) |
|---|---|---|---|---|---|
| none (today) | – | +3,095 / +1,115 / −865 | – | +4,512 / −11,902 / −28,316 | −254 |
| $100M | 21 | +2,490 / +790 / −911 | 184 · 77 % · +0.08 | +3,614 / −10,467 / −24,548 | −112 |
| $150M | 11 | +2,808 / +960 / −887 | 130 · 75 % · +0.06 | +3,992 / −10,878 / −25,748 | −112 |
| $200M | 9 | +2,846 / +971 / −905 | 110 · 75 % · +0.06 | +4,070 / −11,076 / −26,223 | −112 |
| $250M | 7 | +2,893 / +992 / −910 | 91 · 77 % · +0.06 | +4,129 / −11,286 / −26,702 | −112 |
| **$300M** | **5** | **+2,961 / +1,034 / −893** | **71 · 79 % · +0.11** | **+4,049 / −11,636 / −27,321** | **−112** |
| $400M | 2 | +3,028 / +1,066 / −895 | 51 · 78 % · +0.11 | +4,174 / −11,744 / −27,663 | −254 |
| $500M | 2 | +3,028 / +1,066 / −895 | 37 · 81 % · +0.16 | +4,166 / −11,910 / −27,986 | −254 |
| $1B | 1 | +3,062 / +1,090 / −883 | 18 · 83 % · +0.11 | +4,394 / −11,877 / −28,148 | −254 |

- Backtest by year: ≥ $200M 2025 +0.18 / 2026 −0.01 · ≥ $300M +0.18 / +0.07 · ≥ $500M +0.63 / −0.04 (%/fade).
- With unknown mcap → 1× at $300M: master −453 / −316 / −180 vs today, backtest −1,583 / +677 / +2,937 (rules_followup.csv row b2).
- Halving vs a tighter cap: 0.05 % volume cap at ≥ $200M master −371 / −232 / −93 (KMNO −179); at ≥ $300M −176 / −110 / −44.
- No threshold changes the worst single fade (−23.7 % of the account, a low-volume coin). $300M is the cheapest line that still
  covers KMNO ($328–332M) — chosen just under the one loser, so it is an operator line, not evidence.
- High-VOLUME fades (operator's first wording) are the best and cheapest to fill (master ≥ $5M 24/24 won; backtest WR 73 → 80 % as volume
  rises; own impact 0.53 → 0.33 %): cutting them is worse in every scenario and leaves the worst case unchanged — rejected.
