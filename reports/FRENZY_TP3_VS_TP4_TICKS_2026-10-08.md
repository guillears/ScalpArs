# FRENZY exit: +3 vs +4 (and the lock) re-walked on real ticks — 2026-10-08

**Question (operator):** "+3 vs +4 tick check." The live FRENZY_LONG / WIDE / LITE exit is fixed TP +3 % net, stop −3 %, 12 h cap (DECISION_LOG 250). Two earlier tick studies found +3 ≈ +4. This morning's halves report said fixed +3 cuts FRENZY_LONG from +0.174 to +0.011 %/trade. That figure came from a peak-based shortcut. This study settles the conflict on aggTrades ticks.

**Validation first:** `scripts/validate_against_master.py` → **ALL CHECKS PASS**.

**Walker checks (both exact):**
- **Live fills (d):** 19 / 19 live FRENZY-family fills reproduced with the same exit reason. The largest P&L difference is 0.0000 pts, across all four exit eras (old trail, +4, +3, lock).
- **Earlier study (b):** using the earlier study's own timing, the walker reproduces it to the third decimal: +3/−3 +0.192 · +4/−3 +0.187 · +6/−3 +0.263 · lock +0.234 · trail +5/1.5 +0.142.

**Ruler (live):**
- Entry = first print ≥ signal close + 8 s. Entry price = that print + 0.10 % slip. The bot puts its levels on that slipped price.
- Fees: taker 0.045 % on each side. Net % = (p/E − 1)·100 − 0.045 − 0.045·p/E, which reproduces live `pnl_percentage`.
- Exits fill at the print that crosses the line.
- If the TP and the stop are hit in the same millisecond, the stop counts first.
- 12 h cap.

**Data and scripts:**
- Data: the local tick cache. 2 OXT daily archives were downloaded from data.binance.vision. About 4.5k API weight was used for today's 4 live-fill windows (sequential, used-weight peaked at 380, no 418/429).
- Scripts: `scripts/study_tp34_{common,cohorts,walk,report,willy,portfolio,fetch_api}.py`.
- Per-fill CSVs:
  - `reports/FRENZY_TP3_VS_TP4_TICKS_2026-10-08_fills.csv` (cohorts a/b/c, every exit)
  - `…_live.csv` (d)
  - `…_willy.csv`

## Plain-English answer

1. **Keep +3/−3.** On today's entry stack, +4/−3 does not beat +3/−3 on ticks:
   - **Engine-replay fills (a):** −0.084 %/fill [−0.31, +0.14]. Better in only 3 of 10 months; all 3 seeds −0.08.
   - **Today's slice of the 850-fill cohort (b):** −0.028 [−0.22, +0.15].
   - **Live fills (d):** Σ −23.1 vs −15.1 on 17 fills.
   - **Only LITE prefers +4:** +0.107 [−0.05, +0.25], 7 of 10 months, CI spans 0.
   - **Whole family, weighted:** about +0.04/fill, inside noise.
   - **Verdict on the DECISION_LOG 199 rule:** "+4 beats +3 on ticks → revert" is **NOT met**.
2. **Don't go back to the lock either.** Lock minus +3 per fill:

   | Cohort | Lock − +3 |
   |---|---|
   | (a) | +0.014 [−0.13, +0.17] |
   | (b) today's stack | −0.027 |
   | (c) LITE | −0.025 |
   | Family, weighted | −0.012 |
   | (d) live, Σ | −16.4 vs −15.1 |

   - The lock's +0.042 edge in the Oct-5 study came from 1-min-late entries with no slip, and was chosen on that same data.
   - At the live ruler it is +0.001 on that same cohort. After the 30–50 % haircut, nothing is left.
3. **The halves report's "+3 costs FRENZY_LONG most of its year" was arithmetically right, but it was measured on the wrong cohort.**
   - It was the "TP +4 → +3" step of the waterfall, measured on the ORIGINAL yr5 FRENZY_LONG fills: ATR ≤ 2.5, no bearish block.
   - Ticks confirm it there: +4 +0.175 vs +3 −0.010 (Δ +0.185 [−0.02, +0.37]). The replay agrees with ticks on 98 % of outcomes, and 45.4 % reach +4 on both.
   - Two later steps of today's stack remove that edge:
     - the bearish-day block takes out the fills where +4 was strongest (Δ +0.35 [+0.05, +0.56]);
     - the ATR 2.5–3.0 red fills it adds prefer +3 (Δ −0.28).
   - Hold-green WIDE also prefers +3 (Δ −0.28).
   - On the fills today's stack actually keeps, the shortcut gave +0.075 (+4) vs +0.076 (+3), i.e. no cost. Ticks give +0.113 vs +0.084 for FRENZY_LONG and −0.196 vs +0.088 for WIDE.
4. **Sub-groups flip sign between cohorts, so a per-sleeve TP split is not supported.** Example: ATR ≤ 2.5 kept LONG prefers +4 in (a) (+0.12) but +3 in (b) (−0.10). That is noise.
5. **The bigger lever is the entry slip, not the TP.** At 8 s with no slip:
   - (a) +3/−3 = +0.180 instead of +0.085;
   - WIDE +0.238 instead of +0.088.

   The 0.10 % entry ruler cuts about half the family's edge and moves WIDE the most. Measuring real FRENZY entry slip on live fills is worth more than any TP tweak.

### Key table — Δ per fill vs the live +3/−3 (live ruler, day-clustered 95 % CI)

| Cohort | N | +3/−3 avg | +4/−3 − +3 | lock − +3 | +5/−3 − +3 | +6/−3 − +3 | months +4 better |
|---|---|---|---|---|---|---|---|
| (a) yr5 replay, today's stack, FRENZY_LONG + WIDE (3 seeds pooled) | 883 | +0.085 | **−0.084** [−0.31, +0.14] | +0.014 [−0.13, +0.17] | −0.022 | −0.010 | 3 / 10 |
| ↳ FRENZY_LONG | 564 | +0.084 | +0.029 [−0.22, +0.26] | +0.054 | +0.200 | +0.175 | 6 / 10 |
| ↳ FRENZY_WIDE | 319 | +0.088 | −0.284 [−0.73, +0.11] | −0.057 | −0.414 | −0.336 | 3 / 9 |
| (b) Oct-4 850 cohort, all | 850 | +0.143 | +0.020 [−0.10, +0.13] | +0.001 | −0.034 | −0.013 | 6 / 9 |
| (b) filtered to today's stack | 316 | +0.174 | −0.028 [−0.22, +0.15] | −0.027 | +0.056 | +0.096 | 5 / 9 |
| (c) LITE, bearish block (= live LITE) | 548 | +0.277 | +0.107 [−0.05, +0.25] | −0.025 | +0.104 | −0.092 | 7 / 10 |
| (d) live master fills (complete) | 17 | Σ −15.1 | Σ −8.0 | Σ −1.3 | – | – | – |
| (a0) ORIGINAL yr5 FRENZY_LONG (pre-stack, the halves "+0.174 → +0.011" cohort) | 606 | −0.010 | +0.185 [−0.02, +0.37] | +0.187 | – | +0.355 | – |

Without the 5 fills with the largest Δ, (+4 − +3) is:

| Cohort | Δ (+4 − +3) without top 5 |
|---|---|
| (a) | −0.091 |
| (b) today's stack | −0.045 |
| (c) | +0.098 |

So none of the results depends on a handful of runners. The lock without its top 5 is negative in every cohort (−0.04 … −0.12), which shows it is a runner-only rule.

Δ(+4 − +3) by half-year:

| Cohort | H1 | H2 |
|---|---|---|
| (a) | −0.142 | −0.012 |
| (b) today's stack | +0.046 | −0.129 |
| (c) | +0.175 | +0.043 |

## Verdict vs the pre-registered gates

- **DECISION_LOG 199** — "+4/−3 beats +3/−3 on ticks → revert to +4": **not met. Keep +3.**
  - The engine-parity cohort (a) and the live fills (d) both favour +3.
  - Only the LITE study cohort favours +4. That cohort was designed on this same year, so a 30–50 % haircut leaves about +0.05–0.07, with the CI already spanning 0.
  - No cohort has a CI above zero.
- **DECISION_LOG 250 TP3_VS_LOCK scout gate** — first 20 live fills, Σ lock − Σ fixed > +3 pts → review: **leave it running unchanged.**
  - On ticks the year says lock ≈ +3 (family −0.012/fill). For 20 fills, the expected Σ difference is ≈ −0.2 pts. The gate firing would need runners, as it is designed to.
  - The 19 live fills so far: lock Σ −16.4 vs +3 Σ −15.1, so no fire.
- **What the operator should watch live:**
  1. The TP3_VS_LOCK row as is.
  2. Add +4/−3 to that row's per-fill detail. The scout already prices +4 in `price_fixed` for FRENZY_LOCK; show it for the record, not as a gate.
  3. Real entry slip on FRENZY fills (entry_price vs the first print ≥ close + 8 s). This is the variable that moves the result most. On the live fills where both exist, the ruler's slipped entry sits about 0.10 % above the print.
  4. LITE separately. It is the only cohort leaning +4. Pre-register now, frozen: at 40 live LITE fills, if Σ(+4 − +3) on ticks > +4 pts and it is positive in both 20-fill halves → propose +4 for LITE only. Otherwise drop it.

## Halves-table rows replaced (live exit +3/−3/12 h on ticks, flat $3k book, today's sizing as `study_yr5_halves_today.py`)

| Strategy | H1 N · WR · avg % [CI] · $ | H2 N · WR · avg % [CI] · $ | Year N · WR · avg % [CI] · $ |
|---|---|---|---|
| FRENZY_LONG | 103 · 49.8 % · **−0.014** [−0.48, +0.45] · +$549 | 85 · 53.3 % · **+0.202** [−0.38, +0.75] · +$1,408 | 188 · 51.4 % · **+0.084** [−0.26, +0.44] · **+$1,957** |
| FRENZY_WIDE | 60 · 51.7 % · **+0.137** [−0.64, +0.91] · +$240 | 46 · 50.4 % · **+0.024** [−1.04, +0.98] · +$33 | 106 · 51.1 % · **+0.088** [−0.52, +0.71] · **+$273** |
| FRENZY_LITE | 265 · 55.5 % · **+0.333** [−0.10, +0.74] · +$3,877 | 283 · 53.7 % · **+0.225** [−0.13, +0.57] · +$2,792 | 548 · 54.6 % · **+0.277** [−0.01, +0.54] · **+$6,669** |

How these compare with the old shortcut (year):

| Strategy | Old shortcut | Ticks, live ruler | Comment |
|---|---|---|---|
| FRENZY_LONG | +0.076 · +$1,870 | +0.084 · +$1,957 | confirmed |
| FRENZY_WIDE | +0.288 · +$895 | +0.088 · +$273 | almost all of the drop is the 0.10 % entry slip; at no slip, +0.238 · ≈ +$739 |
| FRENZY_LITE | +0.278 · +$6,678 | +0.277 · +$6,669 | confirmed (its shortcut already charged 0.10) |

N is per seed (mean of 3) for LONG / WIDE. LITE is one study cohort.

Caveat: the replay's slot occupancy reflects its own +4 exit. Under +3 the trades are shorter, so a few more signals would be taken. This is not modelled; at ≤ 2 slots per sleeve the effect is small.

## FRENZY_WILLY on ticks, current design

The triggers, the turnover R < 1 rule (unknown → refused), the first red 5m bar within 60 min, entry at the first print ≥ close + 8 s with +0.035 % slip, the dislocation guard and one-WILLY-at-a-time are all exactly as in `scripts/study_yr5_halves_willy.py` (imported).

That script was already tick-based for exits; only red-bar detection uses 5m klines. It is not "1m-based".

**CONV row** = that script's conventions re-run here: TP books exactly +1.0, the cap pays −0.05 extra slip, flat 0.09 fees. It reproduces the script exactly: 1,263 fills, +0.053, worst −25.8.

**NOSTOP row** = bot accounting: TP fills at the crossing print, which averages a little above +1.0; no extra cap slip; exact fees. It comes out +0.023/fill above CONV.

**BACKSTOP row** = the exchange backstop treated as a −2.2 % net stop, with the slots re-sequenced. A stopped trade frees the slot, so there are 63 more fills.

| variant · trigger | H1 N · WR · avg % [CI] · $ | H2 | Year | worst | dipped ≤ −4.5 % | TP / cap / stop |
|---|---|---|---|---|---|---|
| NOSTOP · A | 572 · 83.6 % · **+0.121** [−0.07, +0.31] · +$10,148 | 495 · 86.3 % · **+0.151** · +$10,901 | 1,067 · 84.8 % · **+0.135** [−0.02, +0.28] · +$21,049 | −25.76 | 14.2 % | 902 / 165 / 0 |
| NOSTOP · B | 126 · 78.6 % · **−0.210** · −$3,862 | 70 · 81.4 % · **−0.305** · −$3,127 | 196 · 79.6 % · **−0.244** [−0.68, +0.16] · −$6,988 | −14.69 | 20.9 % | 156 / 40 / 0 |
| NOSTOP · all | 698 · **+0.062** · +$6,287 | 565 · **+0.094** · +$7,774 | 1,263 · 84.0 % · **+0.076** [−0.07, +0.21] · +$14,061 | −25.76 | 15.2 % | 1,058 / 205 / 0 |
| BACKSTOP · A | 605 · **−0.025** · −$2,212 | 516 · **+0.038** · +$2,880 | 1,121 · 68.4 % · **+0.004** [−0.08, +0.09] · +$668 | −2.81 | 0 % | 764 / 13 / 344 |
| BACKSTOP · B | 130 · **−0.353** [−0.69, −0.02] · −$6,706 | 75 · **−0.277** · −$3,040 | 205 · 58.5 % · **−0.325** [−0.59, −0.07] · −$9,746 | −2.39 | 0 % | 120 / 0 / 85 |
| BACKSTOP · all | 735 · **−0.083** · −$8,918 | 591 · **−0.002** · −$160 | 1,326 · 66.9 % · **−0.047** [−0.13, +0.04] · −$9,078 | −2.81 | 0 % | 884 / 13 / 429 |

$ = flat $3k book × 24.375 % × 20× (WILLY lev 1.0).

What the WILLY rows say:
- **Paper, no stop:** WILLY is positive, but every CI spans 0.
- **Trigger B loses in every variant.** With the backstop, B's CI is below 0.
- **Tail risk without a stop:** 15 % of trades dip through −4.5 % (the 20× liquidation zone), and the worst trade is −25.8 %.
- **With the real backstop:** A ≈ 0 and the total is −0.047. The halves report's backstop approximation (−0.105, CI below 0) was too pessimistic: it did not re-sequence the freed slots, and it priced the stop differently.
- **Plainly:** trigger B is a loser on ticks under both designs, and A alone is ≈ 0 once the backstop is real.

## 5-sleeve portfolio (operator study) — one shared compounding book

**Setup:**
- **Book:** $3,000 on 2026-01-04 → 2026-10-04, compounding. Only these five sleeves are active: BULLRUN_LONG, BEARRUN_SHORT (5×), FRENZY_LONG (6× / 10× when strong), FRENZY_WIDE (4×) and FRENZY_LITE (6×).
- **Sizing (today's live rules):**
  - Equal split = (equity − reserve-schedule reserve − fee reserve, max($15, 2.5 %)) / 4.
  - Each trade is also capped at the free tradeable balance.
  - Leverage-balance schedule: 20× below $25k, 15× above.
- **Capacity:**
  - 4 global slots shared by the five sleeves.
  - Per-sleeve slot caps: FRENZY 2 · WIDE 2 · LITE 2 · BULLRUN 4.
  - One position per pair.
  - ≤ 3 entries per pair per UTC day per FRENZY sleeve.
  - Fills are processed in entry-time order; any fill that cannot open is skipped and counted.
- **Prices:**
  - FRENZY_LONG / WIDE / LITE use this study's tick prices (live exit, live ruler).
  - BULLRUN / BEARRUN use the yr5 engine replay's own fills.
- **Equity:** marked at closes (realized).
- **Seeds:** 3 seeds for the replay-sourced sleeves. LITE is identical in all three.

| Scenario (mean of 3 seeds [range]) | End balance | Total return | Daily compound (273 cal. days) | Daily compound (per trading day) | Max DD | Worst day | H1 → H2 return |
|---|---|---|---|---|---|---|---|
| **ALL 5** | **$29,857** [19,324 … 38,397] | +895 % [+544 … +1,180] | **+0.83 %** [+0.68 … +0.94] | +0.96 % (236 days) | **−48.6 %** | −20.5 % (05-14) | H1 +91 % · H2 +421 % [+236 … +560] |
| without BULLRUN | $17,749 [17,117 … 18,530] | +492 % [+471 … +518] | +0.65 % [+0.64 … +0.67] | +0.75 % (236 days) | −48.6 % | −20.5 % | H1 +91 % · H2 +210 % |
| without LITE | $15,182 [6,090 … 22,207] | +406 % [+103 … +640] | +0.55 % [+0.26 … +0.74] | +0.91 % (164 days) | −53.9 % [−65.4 … −47.0] | −21.0 % [−33.1 … −14.0] | H1 +5 % · H2 +382 % [+93 … +596] |
| without BULLRUN and LITE | $4,303 [4,099 … 4,652] | +43 % [+37 … +55] | +0.13 % [+0.11 … +0.16] | +0.22 % (160 days) | −46.3 % | −14.0 % | H1 +5 % · H2 +37 % |

Per sleeve inside the ALL-5 portfolio (mean of 3 seeds):

| Sleeve | Offered / taken | WR | avg % | Net $ [range] | H1 $ · H2 $ |
|---|---|---|---|---|---|
| FRENZY_LONG | 188 / 185 | 51.2 % | +0.069 | +$4,062 [+2,483 … +5,778] | −870 · +4,932 |
| FRENZY_WIDE | 106 / 106 | 51.1 % | +0.088 | −$453 [−634 … −254] | −234 · −218 |
| FRENZY_LITE | 548 / 518 | 54.1 % | +0.254 | +$10,861 [+9,015 … +12,466] | +3,864 · +6,997 |
| BULLRUN | 147 / 136 | 49.4 % | +0.215 | +$12,263 [+2,979 … +19,506] | 0 · +12,263 |
| BEARRUN | 13 / 13 | 67.0 % | +0.035 | +$123 [+13 … +221] | −26 · +149 |

Skipped for capacity, per seed:

| Sleeve | Reason | Fills skipped |
|---|---|---|
| FRENZY_LITE | sleeve slots | 15.0 |
| FRENZY_LITE | pair held | 13.0 |
| FRENZY_LITE | global slots | 2.3 |
| BULLRUN | global slots | 11.0 |
| FRENZY_LONG | pair held | 3.0 |

WIDE has a positive average per trade but loses $ inside the compounding book: its losers landed when the book was larger.

**Caveats (read before quoting any number):**
- **In-sample.** The FRENZY bearish block, ATR 3.0, WIDE hold-green and the LITE design were all chosen on this same year. Haircut the FRENZY-family gains by 30–50 %.
- **LITE is a study cohort, not the engine replay.** It provides about half of the non-BULLRUN growth and all of H1's (+91 % with LITE vs +5 % without).
- **BULLRUN is one 6-day stretch in August.** It is about +$12k of the result, with a seed range of +$3k … +$19.5k.
- **The max drawdown is about −49 % to −54 % in every scenario.** At 4 slots × 24 % of equity × 6× × −3 %, four simultaneous FRENZY-family stops cost about 17 % of the book. The 05-14 worst day was five LITE stops.
- **Costs:** FRENZY-family fills carry taker 0.045 % each side plus 0.10 % entry slip. They carry no exit slip; live stops land ≈ 0.18 % below the last print, which would cost ≈ −0.08 %/fill across the board. BULL/BEAR use the replay's own fills.
- **Not modelled:** funding, liquidation (paper) and the WILLY global hold. WILLY is not in this book; in live trading it would block these sleeves while it is open.
- **"Per trading day"** compounding counts days with at least one entry.

---

# Detail tables (generated by `scripts/study_tp34_report.py`)
## (a) yr5 replay FRENZY_LONG + FRENZY_WIDE, today's entry stack (3 seeds pooled, day clusters)

**(a) all kept fills, live ruler (8 s + 0.10 % entry slip)** — N 883 fills · 158 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 51 % | **+0.085** | +75 | +0.042 (489) | +0.139 (394) | – | – | – |
| +4/−3 | 43 % | **+0.001** | +1 | -0.100 (489) | +0.127 (394) | -0.084 [-0.312, +0.142] | 3 of 10 | -0.091 |
| +5/−3 | 38 % | **+0.063** | +56 | +0.045 (489) | +0.086 (394) | -0.022 [-0.301, +0.262] | 5 of 10 | -0.034 |
| +6/−3 | 34 % | **+0.076** | +67 | +0.088 (489) | +0.060 (394) | -0.010 [-0.379, +0.341] | 4 of 10 | -0.027 |
| lock +2@+3 trail 2 | 51 % | **+0.099** | +87 | +0.123 (489) | +0.069 (394) | +0.014 [-0.128, +0.172] | 3 of 10 | -0.037 |
| old trail +5/1.5 | 38 % | **+0.069** | +61 | +0.109 (489) | +0.019 (394) | -0.017 [-0.321, +0.291] | 4 of 10 | -0.064 |

**(a) FRENZY_LONG** — N 564 fills · 124 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 51 % | **+0.084** | +47 | -0.014 (309) | +0.202 (255) | – | – | – |
| +4/−3 | 45 % | **+0.113** | +64 | +0.008 (309) | +0.241 (255) | +0.029 [-0.222, +0.256] | 6 of 10 | +0.020 |
| +5/−3 | 41 % | **+0.284** | +160 | +0.248 (309) | +0.326 (255) | +0.200 [-0.128, +0.497] | 5 of 10 | +0.183 |
| +6/−3 | 37 % | **+0.259** | +146 | +0.299 (309) | +0.211 (255) | +0.175 [-0.242, +0.559] | 5 of 10 | +0.149 |
| lock +2@+3 trail 2 | 51 % | **+0.138** | +78 | +0.073 (309) | +0.217 (255) | +0.054 [-0.123, +0.237] | 5 of 10 | -0.020 |
| old trail +5/1.5 | 41 % | **+0.278** | +157 | +0.296 (309) | +0.255 (255) | +0.194 [-0.163, +0.537] | 5 of 10 | +0.129 |

**(a) FRENZY_WIDE** — N 319 fills · 82 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 51 % | **+0.088** | +28 | +0.137 (180) | +0.024 (139) | – | – | – |
| +4/−3 | 40 % | **-0.196** | -63 | -0.285 (180) | -0.081 (139) | -0.284 [-0.734, +0.111] | 3 of 9 | -0.305 |
| +5/−3 | 33 % | **-0.326** | -104 | -0.303 (180) | -0.355 (139) | -0.414 [-0.980, +0.115] | 3 of 9 | -0.453 |
| +6/−3 | 30 % | **-0.249** | -79 | -0.273 (180) | -0.218 (139) | -0.336 [-0.996, +0.298] | 2 of 9 | -0.390 |
| lock +2@+3 trail 2 | 51 % | **+0.031** | +10 | +0.210 (180) | -0.202 (139) | -0.057 [-0.280, +0.215] | 2 of 9 | -0.179 |
| old trail +5/1.5 | 33 % | **-0.301** | -96 | -0.213 (180) | -0.414 (139) | -0.389 [-0.990, +0.173] | 3 of 9 | -0.521 |

Per seed (replicates; seeds differ only in replay tick ordering → slot occupancy):

| seed | N | +3/−3 | +4/−3 | Δ +4−+3 | lock | Δ lock−+3 | +6/−3 |
|---|---|---|---|---|---|---|---|
| 1 | 294 | +0.089 | +0.005 | -0.084 | +0.103 | +0.014 | +0.079 |
| 2 | 296 | +0.088 | +0.008 | -0.080 | +0.099 | +0.011 | +0.089 |
| 3 | 293 | +0.079 | -0.009 | -0.088 | +0.096 | +0.017 | +0.059 |

**(a) sensitivity: same fills, 8 s, NO entry slip** — N 883 fills · 158 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 53 % | **+0.180** | +159 | +0.140 (489) | +0.229 (394) | – | – | – |
| +4/−3 | 43 % | **+0.048** | +42 | -0.015 (489) | +0.126 (394) | -0.132 [-0.363, +0.095] | 1 of 10 | -0.138 |
| +5/−3 | 38 % | **+0.062** | +55 | +0.044 (489) | +0.085 (394) | -0.117 [-0.395, +0.177] | 3 of 10 | -0.130 |
| +6/−3 | 36 % | **+0.197** | +174 | +0.255 (489) | +0.126 (394) | +0.018 [-0.322, +0.368] | 6 of 10 | +0.000 |
| lock +2@+3 trail 2 | 53 % | **+0.215** | +190 | +0.226 (489) | +0.203 (394) | +0.036 [-0.096, +0.181] | 4 of 10 | -0.016 |
| old trail +5/1.5 | 38 % | **+0.076** | +68 | +0.106 (489) | +0.040 (394) | -0.103 [-0.391, +0.203] | 4 of 10 | -0.152 |

Per-month Δ (a, live ruler):

| month | N | Δ(+4 − +3) | Δ(lock − +3) |
|---|---|---|---|
| 2026-01 | 83 | +0.291 | +0.341 |
| 2026-02 | 72 | -0.090 | -0.196 |
| 2026-03 | 129 | -0.284 | +0.362 |
| 2026-04 | 142 | -0.272 | -0.088 |
| 2026-05 | 119 | -0.152 | +0.190 |
| 2026-06 | 69 | -0.353 | -0.441 |
| 2026-07 | 78 | +0.229 | -0.025 |
| 2026-08 | 79 | +0.127 | -0.190 |
| 2026-09 | 109 | -0.019 | -0.105 |
| 2026-10 | 3 | +0.000 | +0.000 |

### Why the peak-based approximation disagreed (cohort a, same fills)

| pricing | avg %/fill |
|---|---|
| replay as run (its exit = fixed +4/−3) | +0.049 |
| peak approximation of +3 (replay peak ≥ 3 → +3, else replay result) — the halves table | +0.152 |
| ticks +4/−3 (live ruler) | +0.001 |
| ticks +3/−3 (live ruler) | +0.085 |

- replay +4 TP rate 43.5 % · ticks +4 TP rate 42.8 % · ticks +3 TP rate 51.3 %
- replay peak ≥ +3 52.7 % · ticks peak-before-stop ≥ +3 51.3 % · ≥ +4 42.8 %
- replay vs ticks +4/−3 outcome agreement (TP/SL/CAP): 97.3 %
- the +3-only 'rescue' band (peak before the stop in [+3, +4)): ticks 8.5 % · replay 8.8 %
- entry: replay E vs tick-ruler E median +0.100 % (replay opens 13.7 s after the close, ruler 8 s + 0.10 % slip)

## (b) the Oct-4/5 850-fill tick cohort

Parity (old ruler, entry = 1-min close after the signal, flat 0.09 fees): +3/−3 +0.192 · +4/−3 +0.187 · +6/−3 +0.263 · lock +0.234 · trail +5/1.5 +0.142 (published: +0.192 · +0.187 · +0.263 · +0.234 · +0.142)

Old ruler Δ(+4 − +3) -0.005 [-0.137, +0.121] · Δ(lock − +3) +0.042 [-0.047, +0.141]

**(b) all 850, live ruler** — N 850 fills · 245 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 52 % | **+0.143** | +121 | +0.160 (439) | +0.124 (411) | – | – | – |
| +4/−3 | 45 % | **+0.163** | +138 | +0.159 (439) | +0.167 (411) | +0.020 [-0.099, +0.131] | 6 of 9 | +0.013 |
| +5/−3 | 39 % | **+0.109** | +93 | +0.051 (439) | +0.170 (411) | -0.034 [-0.199, +0.127] | 5 of 9 | -0.046 |
| +6/−3 | 35 % | **+0.129** | +110 | +0.186 (439) | +0.069 (411) | -0.013 [-0.220, +0.184] | 3 of 9 | -0.032 |
| lock +2@+3 trail 2 | 52 % | **+0.144** | +122 | +0.141 (439) | +0.147 (411) | +0.001 [-0.083, +0.097] | 4 of 9 | -0.050 |
| old trail +5/1.5 | 39 % | **+0.055** | +47 | -0.012 (439) | +0.127 (411) | -0.088 [-0.261, +0.085] | 5 of 9 | -0.131 |

**(b) today's entry stack (LONG red ATR ≤ 3 + WIDE hold-green, bearish block): 217 LONG-type · 99 WIDE-type** — N 316 fills · 172 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 53 % | **+0.174** | +55 | +0.285 (183) | +0.021 (133) | – | – | – |
| +4/−3 | 45 % | **+0.146** | +46 | +0.331 (183) | -0.109 (133) | -0.028 [-0.219, +0.153] | 5 of 9 | -0.045 |
| +5/−3 | 41 % | **+0.229** | +73 | +0.437 (183) | -0.056 (133) | +0.056 [-0.208, +0.307] | 6 of 9 | +0.023 |
| +6/−3 | 37 % | **+0.270** | +85 | +0.568 (183) | -0.140 (133) | +0.096 [-0.217, +0.398] | 5 of 9 | +0.048 |
| lock +2@+3 trail 2 | 53 % | **+0.147** | +47 | +0.321 (183) | -0.092 (133) | -0.027 [-0.158, +0.110] | 5 of 9 | -0.116 |
| old trail +5/1.5 | 41 % | **+0.165** | +52 | +0.375 (183) | -0.125 (133) | -0.009 [-0.269, +0.241] | 6 of 9 | -0.107 |

Per-month Δ (b today's stack, live ruler):

| month | N | Δ(+4 − +3) | Δ(lock − +3) |
|---|---|---|---|
| 2026-01 | 39 | +0.076 | +0.177 |
| 2026-02 | 25 | -0.540 | -0.433 |
| 2026-03 | 49 | +0.369 | +0.218 |
| 2026-04 | 44 | +0.044 | +0.086 |
| 2026-05 | 37 | -0.166 | -0.133 |
| 2026-06 | 20 | -0.903 | -0.547 |
| 2026-07 | 29 | +0.274 | +0.123 |
| 2026-08 | 43 | -0.257 | +0.006 |
| 2026-09 | 30 | +0.301 | -0.230 |

## (c) FRENZY_LITE 724-fill study cohort

**(c) all 724, live ruler** — N 724 fills · 244 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 54 % | **+0.257** | +186 | +0.293 (365) | +0.220 (359) | – | – | – |
| +4/−3 | 48 % | **+0.346** | +250 | +0.430 (365) | +0.259 (359) | +0.089 [-0.050, +0.211] | 6 of 10 | +0.081 |
| +5/−3 | 42 % | **+0.325** | +236 | +0.415 (365) | +0.234 (359) | +0.069 [-0.132, +0.265] | 6 of 10 | +0.055 |
| +6/−3 | 36 % | **+0.202** | +146 | +0.205 (365) | +0.198 (359) | -0.055 [-0.326, +0.207] | 4 of 10 | -0.077 |
| lock +2@+3 trail 2 | 54 % | **+0.216** | +156 | +0.235 (365) | +0.197 (359) | -0.041 [-0.135, +0.062] | 5 of 10 | -0.107 |
| old trail +5/1.5 | 42 % | **+0.257** | +186 | +0.311 (365) | +0.203 (359) | +0.000 [-0.216, +0.216] | 6 of 10 | -0.053 |

**(c) bearish-day block applied (= live LITE)** — N 548 fills · 214 days

| Exit | WR | avg %/fill | Σ % | H1 avg (N) | H2 avg (N) | Δ vs +3/−3 per fill [95 % day-CI] | months Δ > 0 | Δ w/o top-5 Δ fills |
|---|---|---|---|---|---|---|---|---|
| +3/−3 (live) | 55 % | **+0.277** | +152 | +0.333 (265) | +0.225 (283) | – | – | – |
| +4/−3 | 48 % | **+0.384** | +211 | +0.509 (265) | +0.268 (283) | +0.107 [-0.048, +0.248] | 7 of 10 | +0.098 |
| +5/−3 | 43 % | **+0.381** | +209 | +0.512 (265) | +0.259 (283) | +0.104 [-0.122, +0.321] | 8 of 10 | +0.086 |
| +6/−3 | 36 % | **+0.185** | +101 | +0.197 (265) | +0.174 (283) | -0.092 [-0.371, +0.194] | 4 of 10 | -0.122 |
| lock +2@+3 trail 2 | 55 % | **+0.252** | +138 | +0.273 (265) | +0.233 (283) | -0.025 [-0.136, +0.098] | 4 of 10 | -0.113 |
| old trail +5/1.5 | 43 % | **+0.335** | +184 | +0.440 (265) | +0.237 (283) | +0.058 [-0.192, +0.310] | 7 of 10 | -0.013 |

Per-month Δ (c kept):

| month | N | Δ(+4 − +3) | Δ(lock − +3) |
|---|---|---|---|
| 2026-01 | 32 | -0.127 | -0.002 |
| 2026-02 | 62 | +0.465 | -0.035 |
| 2026-03 | 52 | +0.016 | -0.087 |
| 2026-04 | 72 | +0.109 | +0.008 |
| 2026-05 | 75 | +0.130 | -0.188 |
| 2026-06 | 53 | +0.434 | -0.071 |
| 2026-07 | 52 | -0.159 | -0.084 |
| 2026-08 | 73 | -0.250 | +0.082 |
| 2026-09 | 71 | +0.224 | +0.065 |
| 2026-10 | 6 | +0.501 | +0.662 |

## (d) live master FRENZY-family fills (walker validation)

As filled (live opened_at + entry_price, the exit era in force): **19 / 19 same exit reason**, |Δ P&L| max 0.0000 pts, mean 0.0000.

| opened (UTC) | pair | sleeve | era exit | live | walker (as filled) | ruler +3/−3 | ruler +4/−3 | ruler lock |
|---|---|---|---|---|---|---|---|---|
| 10-03 03:36 | ENJ | LONG | old trail +5/1.5 | -3.02 STOP_LOSS | -3.02 SL | -3.00 | -3.00 | -3.00 |
| 10-03 23:25 | AIN | WIDE | old trail +5/1.5 | -3.01 STOP_LOSS | -3.01 SL | -3.02 | -3.02 | -3.02 |
| 10-04 05:05 | SAND | LONG | old trail +5/1.5 | -3.00 STOP_LOSS | -3.00 SL | +3.01 | -3.00 | +1.99 |
| 10-04 11:15 | AIN | LONG | old trail +5/1.5 | +3.47 RUNNER_TRAIL | +3.47 TRAIL | +3.01 | +4.01 | +2.67 |
| 10-04 14:05 | SAND | LONG | +4/−3 | -3.01 STOP_LOSS | -3.01 SL | -3.01 | -3.01 | -3.01 |
| 10-05 09:15 | MOVR | WIDE | +3/−3 (live) | +3.00 FRENZY_TP | +3.00 TP | +3.00 | +4.00 | +2.00 |
| 10-05 12:00 | RLC | WIDE | +3/−3 (live) | +3.01 FRENZY_TP | +3.01 TP | +3.00 | +4.01 | +5.04 |
| 10-05 16:15 | AIN | WIDE | lock +2@+3 trail 2 | -3.02 STOP_LOSS | -3.02 SL | -3.01 | -3.01 | -3.01 |
| 10-06 02:30 | FLUID | WIDE | lock +2@+3 trail 2 | -3.04 STOP_LOSS | -3.04 SL | -3.04 | -3.04 | -3.04 |
| 10-06 09:40 | ORCA | LONG | lock +2@+3 trail 2 | -3.00 STOP_LOSS | -3.00 SL | -3.02 | -3.02 | -3.02 |
| 10-06 10:10 | UMA | LONG | lock +2@+3 trail 2 | -3.00 STOP_LOSS | -3.00 SL | -3.01 | -3.01 | -3.01 |
| 10-06 18:05 | ORCA | LONG | lock +2@+3 trail 2 | -3.01 STOP_LOSS | -3.01 SL | -3.01 | -3.01 | -3.01 |
| 10-07 05:45 | SAND | LITE | lock +2@+3 trail 2 | +1.99 RUNNER_TRAIL | +1.99 TRAIL | +3.00 | -3.01 | +2.00 |
| 10-07 19:10 | HEMI | LITE | lock +2@+3 trail 2 | +1.72 RUNNER_TRAIL | +1.72 TRAIL | open | open | open |
| 10-07 19:55 | MET | LITE | lock +2@+3 trail 2 | +3.04 RUNNER_TRAIL | +3.04 TRAIL | +3.01 | +4.00 | +3.06 |
| 10-07 22:00 | MOVR | LITE | lock +2@+3 trail 2 | -3.00 STOP_LOSS | -3.00 SL | -3.00 | -3.00 | -3.00 |
| 10-08 05:50 | W | LONG | lock +2@+3 trail 2 | -3.00 STOP_LOSS | -3.00 SL | -3.00 | -3.00 | -3.00 |
| 10-08 06:50 | MET | WIDE | lock +2@+3 trail 2 | +1.99 RUNNER_TRAIL | +1.99 TRAIL | +3.02 | open | +1.99 |
| 10-08 11:30 | ERA | LONG | lock +2@+3 trail 2 | -3.01 STOP_LOSS | -3.01 SL | -3.01 | -3.01 | -3.01 |

Live ruler on the 17 complete live fills: +3/−3 Σ -15.11 · +4/−3 Σ -23.13 · lock Σ -16.38 · as traded Σ -21.61

## Halves-table rows at the live exit (+3/−3/12 h) on ticks

| Strategy | H1 N · WR · avg % [CI] · $ | H2 N · WR · avg % [CI] · $ | Year N · WR · avg % [CI] · $ |
|---|---|---|---|
| FRENZY_LONG | 103 · 49.8 % · **-0.014** [-0.48, +0.45] · +549 $ | 85 · 53.3 % · **+0.202** [-0.38, +0.75] · +1,408 $ | 188 · 51.4 % · **+0.084** [-0.26, +0.44] · +1,957 $ |
| FRENZY_WIDE | 60 · 51.7 % · **+0.137** [-0.64, +0.91] · +240 $ | 46 · 50.4 % · **+0.024** [-1.04, +0.98] · +33 $ | 106 · 51.1 % · **+0.088** [-0.52, +0.71] · +273 $ |
| FRENZY_LITE | 265 · 55.5 % · **+0.333** [-0.10, +0.74] · +3,877 $ | 283 · 53.7 % · **+0.225** [-0.13, +0.57] · +2,792 $ | 548 · 54.6 % · **+0.277** [-0.01, +0.54] · +6,669 $ |

Same rows under the old peak approximation (from YR5_HALVES_TODAY_STACK): FRENZY_LONG FY +0.076 · +$1,870 · FRENZY_WIDE FY +0.288 · +$895 · FRENZY_LITE FY +0.278 · +$6,678.

## FRENZY_WILLY on ticks (bot accounting)

| variant · trigger | H1 N · WR · avg % [CI] · $ | H2 N · WR · avg % [CI] · $ | Year N · WR · avg % [CI] · $ | worst trade | dipped ≤ −4.5 % | TP / cap / stop |
|---|---|---|---|---|---|---|
| CONV · A | 572 · 83.4 % · **+0.096** [-0.10, +0.28] · +8,056 $ | 495 · 86.3 % · **+0.131** [-0.11, +0.35] · +9,495 $ | 1067 · 84.7 % · **+0.112** [-0.04, +0.26] · +17,551 $ | -25.82 | 14.2 % | 902 / 165 / 0 |
| CONV · B | 126 · 78.6 % · **-0.235** [-0.78, +0.25] · -4,325 $ | 70 · 81.4 % · **-0.327** [-1.06, +0.35] · -3,351 $ | 196 · 79.6 % · **-0.268** [-0.71, +0.13] · -7,677 $ | -14.75 | 20.9 % | 156 / 40 / 0 |
| CONV · all | 698 · 82.5 % · **+0.037** [-0.16, +0.22] · +3,730 $ | 565 · 85.7 % · **+0.074** [-0.16, +0.28] · +6,143 $ | 1263 · 83.9 % · **+0.053** [-0.09, +0.19] · +9,874 $ | -25.82 | 15.2 % | 1058 / 205 / 0 |
| NOSTOP · A | 572 · 83.6 % · **+0.121** [-0.07, +0.31] · +10,148 $ | 495 · 86.3 % · **+0.151** [-0.09, +0.36] · +10,901 $ | 1067 · 84.8 % · **+0.135** [-0.02, +0.28] · +21,049 $ | -25.76 | 14.2 % | 902 / 165 / 0 |
| NOSTOP · B | 126 · 78.6 % · **-0.210** [-0.76, +0.27] · -3,862 $ | 70 · 81.4 % · **-0.305** [-1.03, +0.37] · -3,127 $ | 196 · 79.6 % · **-0.244** [-0.68, +0.16] · -6,988 $ | -14.69 | 20.9 % | 156 / 40 / 0 |
| NOSTOP · all | 698 · 82.7 % · **+0.062** [-0.13, +0.24] · +6,287 $ | 565 · 85.7 % · **+0.094** [-0.14, +0.30] · +7,774 $ | 1263 · 84.0 % · **+0.076** [-0.07, +0.21] · +14,061 $ | -25.76 | 15.2 % | 1058 / 205 / 0 |
| BACKSTOP · A | 605 · 67.6 % · **-0.025** [-0.14, +0.09] · -2,212 $ | 516 · 69.4 % · **+0.038** [-0.09, +0.16] · +2,880 $ | 1121 · 68.4 % · **+0.004** [-0.08, +0.09] · +668 $ | -2.81 | 0.0 % | 764 / 13 / 344 |
| BACKSTOP · B | 130 · 57.7 % · **-0.353** [-0.69, -0.02] · -6,706 $ | 75 · 60.0 % · **-0.277** [-0.69, +0.12] · -3,040 $ | 205 · 58.5 % · **-0.325** [-0.59, -0.07] · -9,746 $ | -2.39 | 0.0 % | 120 / 0 / 85 |
| BACKSTOP · all | 735 · 65.9 % · **-0.083** [-0.19, +0.02] · -8,918 $ | 591 · 68.2 % · **-0.002** [-0.13, +0.12] · -160 $ | 1326 · 66.9 % · **-0.047** [-0.13, +0.04] · -9,078 $ | -2.81 | 0.0 % | 884 / 13 / 429 |


### (a0) / (a) sub-cohort decomposition (live ruler) — where the halves report's "+0.174 → +0.011" went

| sub-cohort | N | +3/−3 | +4/−3 | Δ +4−+3 [CI] | lock | +6/−3 |
|---|---|---|---|---|---|---|
| a0 ORIGINAL yr5 FRENZY_LONG (ATR ≤ 2.5 red, replay label, no stack) | 606 | −0.010 | +0.175 | +0.185 [−0.02, +0.37] | +0.177 | +0.345 |
| ↳ kept by today's stack | 435 | +0.048 | +0.168 | +0.120 [−0.14, +0.36] | +0.197 | +0.421 |
| ↳ refused today (bearish day) | 171 | −0.156 | +0.195 | +0.351 [+0.05, +0.56] | +0.127 | +0.154 |
| a FRENZY_LONG moved in (red ATR 2.5–3.0) | 129 | +0.206 | −0.073 | −0.279 [−0.90, +0.31] | −0.063 | −0.286 |
| a FRENZY_WIDE (hold-green) | 319 | +0.088 | −0.196 | −0.284 [−0.73, +0.11] | +0.031 | −0.249 |
| b FRENZY (ATR ≤ 2.5 red), all | 193 | +0.352 | +0.364 | +0.012 [−0.24, +0.24] | – | – |
| b today's LONG kept, ATR ≤ 2.5 | 152 | +0.269 | +0.166 | −0.103 [−0.41, +0.18] | – | – |
| b today's WIDE kept | 99 | +0.356 | +0.414 | +0.058 [−0.34, +0.40] | – | – |

On a0 the replay agrees with the ticks: same +4 outcome 98 %, 45.4 % reach +4 on both, and 4.5 % (ticks) vs 5.0 % (replay) fall in the +3…+4 rescue band. The shortcut's arithmetic was exact.

The conflict is cohort composition, not pricing:
- the bearish block removes the +4-friendly fills;
- the ATR 2.5–3.0 fills and hold-green WIDE fills prefer +3;
- the matching sub-groups of (b) do not reproduce (a)'s signs.
