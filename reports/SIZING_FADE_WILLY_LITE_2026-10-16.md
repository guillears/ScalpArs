# Are SPIKE_FADE, FRENZY_WILLY and FRENZY_LITE under-sized? (2026-10-16)

Research only. No engine, config, `trading_config.json`, template or test change; nothing committed; no bot API; backtest_cache and the
replay scripts untouched (only their saved outputs in `reports/` were read). Script: scratchpad `sizing_study.py` (inputs listed in §6).
`scripts/validate_against_master.py` → **ALL CHECKS PASS** (run after the study). Master figures reconcile with STACK 2026-10-16a
(FADE 74 · 89 % · +0.356 · WILLY 27 · 85 % · +0.360 · LITE 24 · 58 % · +0.309); WILLY backtest reconciles with WILLY_THEORY (MATCH tick 656 · 78.8 % · +0.041).

## Plain English first

**The honest answer is "no, none of the three is under-sized — don't size any of them up."** The ruin risk at today's live size is already
the binding constraint, and for two of the three the backtest edge is too small to pay for the size they already carry.

- **What "today's size" means on the live book.** Live margins are ≈ $250, which on equal-split over 4 slots (minus the 2.5 % fee reserve)
  means a book of about **$1,025**. WILLY and LITE run at 10×, so each trade's notional is **≈ 2.4× the whole book**: a −4 % WILLY stop costs
  ≈ −9.8 % of the book, a −3 % LITE stop ≈ −7.3 %. The fade runs 2× margin at 20×, so its notional is **≈ 9.75× the book** (≈ $10k):
  its −1.5 % stop costs ≈ −14.6 % of the book, its −2.5 % tripwire ≈ −24 %.
- **New finding: the live fade is already ~3× bigger, relative to the book, than any fade the evidence was earned on.** On the $2.9k paper
  book the 0.5 %-of-24 h-volume liquidity cap cut almost every fade (median paper fade ≈ $9.0k notional ≈ **3.1× the book**). On a $1k book
  the same cap ($10k at the smallest $2M pair) no longer binds, so every live fade is ≈ **9.75× the book**. So the fade has, in effect,
  *already been sized up ~3×* by going live on a small account. This is the most important sizing fact in this study.
- **Edges (per trade, net % of notional, 1×):**
  - FADE — backtest two years **+0.043 %** (95 % CI −0.056 … +0.139; without the washed-out Jun-18→Jul-2 window only +0.019) vs master
    **+0.356 %** (74 fills, 89 % WR, CI +0.19 … +0.54, positive in Aug, Sep and Oct). Today's fade filters remove 50 master fills that
    averaged −0.26 %, so part of the master number is in-sample. No live fade yet.
  - WILLY — engine-matched backtest on today's exit **+0.041 %** (656 fills, P(>0) 0.68; 2025 −0.03, 2026 +0.06 / +0.08); master 19
    **+0.59 %** (17 % of that is the in-sample late-TP / resting-TP re-pricing); **live 8 fills −0.69 % as traded** (−0.19 % on today's stack).
  - LITE — backtest with today's LITE-C exit **−0.01 %** all pricing / **+0.07 %** tick-priced only (2025 −0.20, 2026 +0.18 / +0.18);
    master 20 **+0.45 %** (**47 % of it is the in-sample LITE-C re-pricing**); live 4 fills −1.53 % as traded.
- **Kelly reference (not a target).** Growth-optimal exposure per trade is μ/σ². On the backtest edges, today's size is already
  **≈ 3.4× full Kelly for the fade, ≈ 2.6× for WILLY and ≈ 2.6× for LITE** — past the point where extra size *lowers* long-run growth (the
  growth per trade is negative at today's size: FADE −0.30 %, WILLY −0.03 %, LITE −0.06 % of the book). Only on the master edges, halved for
  in-sample bias, does today's size sit at ⅓ Kelly (FADE), ½ Kelly (WILLY) and 1.3× Kelly (LITE).
- **Ruin.** Even granting the master edges in full ("master world", tails excluded), going from 1× to 1.5× moves P(a 50 % drawdown within a
  year) **FADE 5 % → 41 %, WILLY 31 % → 86 %, LITE 95 % → 100 %**. With the master edge halved: FADE 41 % → 88 %, WILLY 76 % → 99 %. That is
  a material rise in every case, so by the brief's own rule: **don't**.
- **Tails make it worse.** The backtest has one fill per sleeve that by itself would gut the live book: WILLY SAHARA −40.3 % (≈ −98 % of the
  book at today's 2.4× exposure — under CROSS margin that is the account), FADE FUSDT −7.7 % (≈ −75 %), LITE XANUSDT −9.3 % (≈ −23 %).
- **Can the fade even be raised?** Mechanically, a little: at $1,025 the volume cap starts to bind at 1.5× (23 % of fills), 2× (38 %),
  3× (57 %), and 3× (≈ 29× book) would hit `max_gross_leverage` 25 and, on 10×-bracket coins, need more margin than the book has.
  At a $10k book the cap binds on 99 % of fades already (realised ≈ 3.4× book) — the fade's size self-limits as the account grows.

**Verdicts**

| sleeve | verdict | evidence strength | proposal |
|---|---|---|---|
| SPIKE_FADE | **NOT under-sized — over-exposed at the live book** (≈ 9.75× book vs ≈ 3× when the evidence was earned) | master strong & stable (74 · 89 %), backtest weak (+0.04, CI spans 0); in-sample part real | No size-up. **Option for the operator (not a ship request):** `spike_fade_invest_mult` 2.0 → 1.0 (≈ 4.9× book, the paper-evidence exposure range); that flips the backtest growth per trade from −0.30 % to ≈ +0.03 % while keeping most of the master-world growth (P(DD≥50 %) 5 % → 0 % unhaircut). |
| FRENZY_WILLY | **NOT under-sized — at or above the growth optimum** | weak: backtest ≈ 0 (P 0.68), master 19 good, live 8 negative | No size-up; never raise WILLY leverage (backstop gap, DL 271/292). |
| FRENZY_LITE | **NOT under-sized — above the growth optimum** | weak: backtest ≈ 0 (tick +0.07, 2025 negative), master half in-sample, live 4 negative | No size-up. The existing 283 gate (→ 8× if the first 20 fills at 10× average < 0) stays the live instrument. |

A size-up would be re-opened only through the gates in §5.

---

## 1. Sizing facts (today, read from `trading_config.json` / `config.py` / engine)

| item | value |
|---|---|
| live account (sizing equity) | ≈ **$1,025** (B22 margins $222–268 ≈ (E − 2.5 % fee reserve) / 4 slots; equal_split, `max_open_positions` 4) |
| base leverage | 20× (VERY_STRONG / STRONG_BUY), `leverage_balance_schedule` 0:20 |
| SPIKE_FADE | `spike_fade_invest_mult` 2.0 · `spike_fade_lev_mult` 1.0 → 20× · notional ≈ **9.75× book** · SL −1.5, tripwire −2.5 · liq cap `spike_lowvol_liq_cap_pct` 0.5 % of 24 h vol (all pairs; fade pairs ≤ $20M) |
| FRENZY_WILLY | invest 1.0 · lev 0.5 → 10× · notional ≈ **2.44× book** · TP +1.25 (resting maker) · late TP +0.25 after 30 min · stop −4 · 90 min · 1 slot · **global hold** (no other automated open while a WILLY is open) |
| FRENZY_LITE | invest 1.0 · lev 0.5 → 10× · notional ≈ **2.44× book** · LITE-C TP +2 / +3 · stop −3 · 2 slots |
| exchange backstop | per order = own stop + 0.5 (≥ 2.5): FADE 2.5 % · LITE 3.5 % · WILLY 4.5 % |
| margin mode | CROSS (the whole account backs every position) |
| gross cap | `max_gross_leverage` 25 (Σ open notional ≤ 25× balance) |

Liquidation distance (CROSS, the position alone, maintenance 1–2.5 %): move ≈ (1 − x·mmr)/x for exposure x = notional/book.

| sleeve | x at 1× | liq. move 1× | 1.5× | 2× | 3× |
|---|---|---|---|---|---|
| FADE | 9.75 | ≈ 7.8–9.3 % | ≈ 4.3–5.8 % | ≈ 2.6–4.1 % (≈ the −2.5 tripwire) | ≈ 0.9–2.4 % (and over the gross cap) |
| WILLY / LITE | 2.44 | ≈ 38–40 % | ≈ 25–27 % | ≈ 18–20 % | ≈ 11–13 % |

Under CROSS a gap through the backstop is paid by the whole book: SAHARA's −40 % at 2.44× ≈ −98 %; FUSDT's −7.7 % at 9.75× ≈ −75 %.

Fade liquidity cap vs book size (backtest fade pairs, 24 h vol median $5.1M, p10 $2.5M):

| book | size | desired notional | fills capped | realised notional / book |
|---|---|---|---|---|
| $1,025 | 1× | $9,994 | 0 % | 9.75 |
| $1,025 | 1.5× | $14,991 | 23 % | 14.09 |
| $1,025 | 2× | $19,988 | 38 % | 17.49 |
| $1,025 | 3× | $29,981 | 57 % | 22.49 |
| $2,900 (paper) | 1× | $28,275 | 54 % | 7.68 |
| $10,000 | 1× | $97,500 | 99 % | 3.38 |
| $30,000 | 1× | $292,500 | 100 % | 1.13 |

Master paper fades as filled: median notional $8,964 ≈ **3.1× the ~$2.9k book** (median filled/desired 0.36); bracket max leverage stamped
50× on 6, 25× on 4, 20× on 4, **10× on 3** of 17 → on a 10×-bracket coin a $10k fade needs ≈ $1,000 margin = the whole live book.

## 2. Edge — three blocks (net % per trade at 1×; CI = day-clustered bootstrap)

### SPIKE_FADE (backtest = yr6W engine replay re-priced to today, 2025-01 → 2026-10-11; no fade-config change since the replay commit 01419d4)

| set | N | WR | avg %/trade | SD | avg win / avg loss | breakeven WR | 95 % day-boot CI | P(avg>0) | Σ %-pts |
|---|---|---|---|---|---|---|---|---|---|
| BT 2025 BULL | 223 | 67 % | +0.072 | 1.64 | +0.88 / -1.55 | 64 % | [-0.145, +0.295] | 0.74 | +16.0 |
| BT 2026-H1 BEAR | 282 | 77 % | +0.010 | 0.92 | +0.47 / -1.49 | 76 % | [-0.100, +0.108] | 0.57 | +2.8 |
| BT 2026-H2 BULL | 120 | 77 % | +0.067 | 0.94 | +0.53 / -1.45 | 73 % | [-0.111, +0.237] | 0.78 | +8.0 |
| **BT TOTAL** | **625** | **73 %** | **+0.043** | 1.23 | +0.61 / -1.51 | 71 % | [-0.056, +0.139] | 0.80 | **+26.9** |
| BT TOTAL ex washed-out Jun-18→Jul-2 | 563 | 71 % | +0.019 | 1.28 | +0.64 / -1.51 | 70 % | [-0.085, +0.118] | 0.63 | +10.6 |
| BT washed-out window only | 62 | 92 % | +0.263 | 0.58 | +0.42 / -1.51 | 78 % | [+0.116, +0.383] | 1.00 | +16.3 |
| MASTER Aug B1–B3 | 40 | 85 % | +0.376 | 0.94 | +0.67 / -1.29 | 66 % | [+0.101, +0.694] | 1.00 | +15.0 |
| MASTER Sep B4–B13 | 16 | 94 % | +0.419 | 0.58 | +0.55 / -1.50 | 73 % | [+0.154, +0.629] | 1.00 | +6.7 |
| MASTER Oct B14–B21 | 18 | 94 % | +0.257 | 0.46 | +0.36 / -1.50 | 81 % | [+0.087, +0.412] | 1.00 | +4.6 |
| **MASTER TOTAL today's stack** | **74** | **89 %** | **+0.356** | 0.77 | +0.56 / -1.34 | 70 % | [+0.192, +0.538] | 1.00 | **+26.3** |
| MASTER as traded, ALL fade fills (kept + today-refused) | 124 | 69 % | +0.091 | 0.84 | +0.53 / -0.91 | 63 % | [-0.048, +0.251] | 0.90 | +11.3 |
| MASTER today-refused fills (as traded) | 50 | 46 % | -0.261 | 0.82 | +0.44 / -0.85 | 66 % | [-0.496, -0.033] | 0.01 | -13.1 |
| LIVE B22 | 0 | – | – | – | – | – | – | – | – |

In-sample share: the kept-vs-all gap (+0.356 vs +0.091) is the fade filters' doing, and those filters were fitted on the same fills → treat
≥ half of the master edge as in-sample (30–50 % haircut ⇒ +0.18 … +0.25). Oct B14–B21 (mostly after the filters shipped) is the cleanest
forward slice: 18 · 94 % · +0.26. The backtest (which already applies today's fade rules) gives +0.04 — the master/backtest gap is unexplained
(thin-pair paper fills? regime?) and is the main uncertainty.

### FRENZY_WILLY (backtest = WILLY_THEORY bt.pkl, RH ticks, today's exit; MATCH = bar-exact engine fills)

| set | N | WR | avg %/trade | SD | avg win / avg loss | breakeven WR | 95 % day-boot CI | P(avg>0) | Σ %-pts |
|---|---|---|---|---|---|---|---|---|---|
| BT MATCH tick 2025 | 169 | 78 % | -0.027 | 2.09 | +1.04 / -3.69 | 78 % | [-0.337, +0.273] | 0.44 | -4.6 |
| BT MATCH tick 2026-H1 | 314 | 79 % | +0.058 | 2.15 | +1.15 / -3.98 | 78 % | [-0.183, +0.287] | 0.69 | +18.2 |
| BT MATCH tick 2026-H2 | 173 | 80 % | +0.077 | 2.03 | +1.04 / -3.87 | 79 % | [-0.224, +0.364] | 0.70 | +13.3 |
| **BT MATCH TOTAL** | **656** | **79 %** | **+0.041** | 2.10 | +1.09 / -3.87 | 78 % | [-0.114, +0.194] | 0.68 | **+26.9** |
| BT MATCH ex washed-out | 641 | 79 % | +0.030 | 2.10 | +1.09 / -3.87 | 78 % | [-0.131, +0.188] | 0.66 | +19.1 |
| BT all rebuilt signals, tick | 1839 | 77 % | -0.028 | 2.14 | +1.11 / -3.79 | 77 % | [-0.131, +0.073] | 0.30 | -52.2 |
| BT yr6W engine replay (old +1.25/−4/90, 1m+tick, gap tails) | 1519 | 68 % | -0.104 | 2.48 | +1.32 / -3.12 | 70 % | [-0.227, +0.014] | 0.04 | -158.2 |
| MASTER as traded (B19–B21) | 19 | 84 % | +0.491 | 1.46 | +1.07 / -2.62 | 71 % | [-0.054, +0.924] | 0.97 | +9.3 |
| MASTER today's stack (B19–B21) | 19 | 89 % | +0.591 | 1.65 | +1.13 / -4.01 | 78 % | [-0.066, +1.167] | 0.96 | +11.2 |
| LIVE B22 as traded | 8 | 62 % | -0.690 | 2.49 | +1.11 / -3.69 | 77 % | – | – | -5.5 |
| LIVE B22 today's stack | 8 | 75 % | -0.188 | 2.38 | +1.08 / -4.00 | 79 % | – | – | -1.5 |
| **MASTER + LIVE TOTAL today's stack** | **27** | **85 %** | **+0.360** | 1.88 | +1.12 / -4.01 | 78 % | [-0.353, +0.875] | 0.86 | **+9.7** |

In-sample share: the late TP (297) and resting TP (296) were fitted on these fills: +1.89 of the master's +11.22 pts (17 %) and +4.02 pts
on B22 are re-pricing, not evidence. WR 79 % vs breakeven 78 % — the sleeve lives one percentage point above breakeven.

### FRENZY_LITE (backtest = `reports/lite_wlsweep` paths, today's LITE-C exit: +3 if vol ≥ 20× ∧ BTC 1 h slope > 0, else +2; −3 stop; −0.2 stop-slip ruler)

| set | N | WR | avg %/trade | SD | avg win / avg loss | breakeven WR | 95 % day-boot CI | P(avg>0) | Σ %-pts |
|---|---|---|---|---|---|---|---|---|---|
| BT 2025 (all pricing) | 828 | 55 % | -0.156 | 2.77 | +2.34 / -3.18 | 58 % | [-0.340, +0.027] | 0.05 | -129.3 |
| BT 2025 tick-priced only | 264 | 54 % | -0.197 | 2.81 | +2.38 / -3.19 | 57 % | [-0.513, +0.120] | 0.11 | -52.1 |
| BT 2026-H1 | 369 | 61 % | +0.186 | 2.74 | +2.34 / -3.22 | 58 % | [-0.084, +0.454] | 0.91 | +68.5 |
| BT 2026-H2 | 255 | 60 % | +0.182 | 2.77 | +2.40 / -3.15 | 57 % | [-0.217, +0.557] | 0.82 | +46.4 |
| **BT TOTAL (all pricing)** | **1452** | **57 %** | **-0.010** | 2.77 | +2.35 / -3.18 | 58 % | [-0.154, +0.129] | 0.45 | **-14.4** |
| BT TOTAL tick-priced | 888 | 59 % | +0.071 | 2.77 | +2.37 / -3.19 | 57 % | [-0.114, +0.255] | 0.75 | +62.7 |
| BT TOTAL ex washed-out | 1426 | 57 % | -0.011 | 2.77 | +2.35 / -3.18 | 58 % | [-0.154, +0.132] | 0.45 | -15.1 |
| BT old +3/−3 (reference) | 1452 | 50 % | -0.084 | 3.10 | +3.00 / -3.17 | 51 % | [-0.250, +0.088] | 0.16 | -122.3 |
| MASTER as traded (B18–B21) | 20 | 55 % | +0.237 | 3.03 | +2.89 / -3.01 | 51 % | [-1.001, +1.801] | 0.65 | +4.7 |
| MASTER today's stack (B18–B21) | 20 | 60 % | +0.445 | 2.92 | +2.75 / -3.01 | 52 % | [-0.787, +2.066] | 0.74 | +8.9 |
| LIVE B22 as traded | 4 | 25 % | -1.526 | 2.91 | +2.80 / -2.97 | 51 % | – | – | -6.1 |
| LIVE B22 today's stack | 4 | 50 % | -0.368 | 2.74 | +2.00 / -2.74 | 58 % | – | – | -1.5 |
| **MASTER + LIVE TOTAL today's stack** | **24** | **58 %** | **+0.309** | 2.85 | +2.64 / -2.96 | 53 % | [-0.865, +1.704] | 0.67 | **+7.4** |

In-sample share: LITE-C (301) was built on these fills: +4.16 of the master's +8.90 pts (**47 %**) and +4.63 pts on B22 are re-pricing.
2025's "1m" rows carry a stop-first artefact (DL 301) — the tick-only total (+0.07) is the fairer backtest read.

## 3. Risk at today's size and 1.5× / 2× / 3× (% of a $1,025 book)

Method: fills sequenced in time; per fill, book % = net % × exposure (FADE 9.75 with the live liquidity cap, WILLY / LITE 2.44); summed per UTC
day, compounded across days (a day can lose at most 100 %). "Path" = the actual 2025-01 → 2026-10 sequence. 1-year probabilities = 4,000
paths of 365 days built from 7-day blocks of the real day sequence (keeps clustering). "Master world" = the backtest's fill frequency with
trades drawn from the master's today's-stack fills (no gap tails — an optimistic upper bound). Ruin = equity ever ≤ 10 % of start.

### SPIKE_FADE

| source | size | worst trade | worst day | worst week | max DD (path) | P(DD≥30 %) 1 y | P(DD≥50 %) 1 y | P(ruin) | median end × 1 y [p10, p90] |
|---|---|---|---|---|---|---|---|---|---|
| BT, raw (FUSDT −7.7 % kept) | 1× | -75.1 % | -72.3 % | -72.3 % | -100 % | 100 % | 100 % | 48.0 % | 0.28 [0.01, 5.61] |
| BT, raw | 1.5× | -109.2 % | -105.0 % | -105.0 % | -100 % | 100 % | 100 % | 85.4 % | 0.00 [0.00, 1.17] |
| BT, raw | 2× | -109.8 % | -126.0 % | -181.4 % | -100 % | 100 % | 100 % | 96.0 % | 0.00 [0.00, 0.05] |
| BT, raw | 3× | -142.7 % | -174.0 % | -413.5 % | -100 % | 100 % | 100 % | 99.4 % | 0.00 |
| BT, tails clipped at −3.0 (backstop holds) | 0.5× | -14.6 % | -21.3 % | -29.5 % | -84 % | 99 % | 73 % | 0.7 % | 1.57 [0.42, 5.61] |
| BT, tails clipped | **1× (today)** | -29.2 % | -42.5 % | -52.7 % | -99 % | 100 % | 100 % | 26.7 % | 0.77 [0.05, 11.14] |
| BT, tails clipped | 1.5× | -43.9 % | -63.8 % | -67.9 % | -100 % | 100 % | 100 % | 62.2 % | 0.17 [0.00, 8.61] |
| BT, tails clipped | 2× | -58.5 % | -85.0 % | -86.6 % | -100 % | 100 % | 100 % | 84.6 % | 0.01 [0.00, 2.13] |
| BT, tails clipped | 3× | -87.8 % | -127.5 % | -130.9 % | -100 % | 100 % | 100 % | 98.5 % | 0.00 |
| Master world, no haircut | 0.5× | -7.3 % | – | – | – | 2 % | 0 % | 0.0 % | 290 |
| Master world, no haircut | **1× (today)** | -14.6 % | – | – | – | 61 % | 5 % | 0.0 % | 4.5e4 |
| Master world, no haircut | 1.5× | -21.9 % | – | – | – | 97 % | 41 % | 0.1 % | 3.4e6 |
| Master world, no haircut | 2× | -29.3 % | – | – | – | 100 % | 81 % | 0.8 % | 2.2e8 |
| Master world, no haircut | 3× | -43.9 % | – | – | – | 100 % | 99 % | 10.1 % | 6.1e10 |
| Master world, edge halved | 0.5× | – | – | – | – | 26 % | 1 % | 0.0 % | 15.9 |
| Master world, edge halved | **1× (today)** | – | – | – | – | 96 % | 41 % | 0.0 % | 162 |
| Master world, edge halved | 1.5× | – | – | – | – | 100 % | 88 % | 0.5 % | 854 |

(The master-world end multiples are fantasy-scale because 74 near-flawless paper fills are resampled for a year; read them only as "how
much ruin risk rises with size even if the master edge is real".)

### FRENZY_WILLY

| source | size | worst trade | worst day | worst week | max DD (path) | P(DD≥30 %) 1 y | P(DD≥50 %) 1 y | P(ruin) | median end × 1 y [p10, p90] |
|---|---|---|---|---|---|---|---|---|---|
| BT yr6W, raw (SAHARA −40 % kept) | **1× (today)** | -98.2 % | -95.1 % | -96.6 % | -100 % | 100 % | 100 % | 82.5 % | 0.02 [0.00, 0.30] |
| BT yr6W, raw | 1.5× | -147.2 % | -142.6 % | -124.2 % | -100 % | 100 % | 100 % | 95.6 % | 0.00 |
| BT yr6W, raw | 2× | -196.3 % | -190.2 % | -140.0 % | -100 % | 100 % | 100 % | 98.9 % | 0.00 |
| BT yr6W, raw | 3× | -294.5 % | -285.3 % | -143.3 % | -100 % | 100 % | 100 % | 100 % | 0.00 |
| BT yr6W, tails clipped at −5.0 | 0.5× | -6.1 % | -19.6 % | -27.3 % | -93 % | 100 % | 92 % | 6.3 % | 0.37 |
| BT yr6W, tails clipped | 1× | -12.2 % | -39.3 % | -52.0 % | -100 % | 100 % | 100 % | 67.7 % | 0.07 |
| BT MATCH tick, today's exit | 0.25× | -2.7 % | -6.0 % | -8.5 % | -27 % | 20 % | 0 % | 0.0 % | 1.03 [0.77, 1.40] |
| BT MATCH tick, today's exit | 0.5× | -5.4 % | -12.0 % | -16.4 % | -50 % | 80 % | 26 % | 0.0 % | 1.00 [0.54, 1.89] |
| BT MATCH tick, today's exit | **1× (today)** | -10.8 % | -24.1 % | -30.9 % | -81 % | 100 % | 88 % | 3.5 % | 0.75 [0.21, 2.69] |
| BT MATCH tick, today's exit | 1.5× | -16.2 % | -36.1 % | -43.5 % | -94 % | 100 % | 99 % | 25.8 % | 0.42 [0.06, 3.14] |
| BT MATCH tick, today's exit | 2× | -21.6 % | -48.2 % | -54.4 % | -99 % | 100 % | 100 % | 52.7 % | 0.18 [0.01, 2.71] |
| BT MATCH tick, today's exit | 3× | -32.4 % | -72.2 % | -76.9 % | -100 % | 100 % | 100 % | 86.2 % | 0.01 |
| Master world, no haircut | **1× (today)** | -9.8 % | – | – | – | 96 % | 31 % | 0.0 % | 664 |
| Master world, no haircut | 1.5× | -14.7 % | – | – | – | 100 % | 86 % | 0.1 % | 7.8e3 |
| Master world, no haircut | 2× | -19.6 % | – | – | – | 100 % | 99 % | 0.8 % | 5.6e4 |
| Master world, edge halved | **1× (today)** | – | – | – | – | 100 % | 76 % | 0.1 % | 16.3 |
| Master world, edge halved | 1.5× | – | – | – | – | 100 % | 99 % | 2.5 % | 34.1 |

### FRENZY_LITE

| source | size | worst trade | worst day | worst week | max DD (path) | P(DD≥30 %) 1 y | P(DD≥50 %) 1 y | P(ruin) | median end × 1 y [p10, p90] |
|---|---|---|---|---|---|---|---|---|---|
| BT yr6W (old +3 exit) | **1× (today)** | -22.6 % | -44.8 % | -79.3 % | -100 % | 100 % | 100 % | 68.9 % | 0.08 [0.00, 1.58] |
| BT yr6W | 1.5× | -33.9 % | -67.2 % | -94.0 % | -100 % | 100 % | 100 % | 92.2 % | 0.00 |
| BT yr6W | 2× | -45.2 % | -89.6 % | -99.5 % | -100 % | 100 % | 100 % | 98.8 % | 0.00 |
| BT yr6W | 3× | -67.8 % | -134.3 % | -124.8 % | -100 % | 100 % | 100 % | 100 % | 0.00 |
| BT LITE-C tick-priced | 0.25× | -2.2 % | -9.9 % | -12.3 % | -38 % | 49 % | 5 % | 0.0 % | 1.12 [0.70, 1.85] |
| BT LITE-C tick-priced | 0.5× | -4.4 % | -19.7 % | -23.6 % | -64 % | 98 % | 58 % | 0.1 % | 1.08 [0.43, 2.92] |
| BT LITE-C tick-priced | **1× (today)** | -8.7 % | -39.5 % | -43.2 % | -89 % | 100 % | 99 % | 16.5 % | 0.66 [0.11, 4.28] |
| BT LITE-C tick-priced | 1.5× | -13.1 % | -59.2 % | -59.2 % | -98 % | 100 % | 100 % | 53.2 % | 0.21 [0.01, 4.28] |
| BT LITE-C tick-priced | 2× | -17.5 % | -79.0 % | -72.0 % | -100 % | 100 % | 100 % | 80.5 % | 0.03 [0.00, 1.59] |
| Master world, no haircut | **1× (today)** | -7.4 % | – | – | – | 100 % | 95 % | 0.8 % | 70.6 |
| Master world, no haircut | 1.5× | -11.2 % | – | – | – | 100 % | 100 % | 6.3 % | 149 |
| Master world, no haircut | 2× | -14.9 % | – | – | – | 100 % | 100 % | 26.6 % | 71 |
| Master world, edge halved | **1× (today)** | – | – | – | – | 100 % | 100 % | 10.7 % | 3.4 |
| Master world, edge halved | 1.5× | – | – | – | – | 100 % | 100 % | 39.8 % | 1.4 |

LITE's problem is variance, not tails: ±3 % brackets × 2.44× = ±7 % of the book per fill, up to 2 at once, ~2 fills a day in the backtest
(191 days with ≥ 2 LITE stops, worst day 6 stops).

### TOTAL — the three sleeves together (yr6W engine replay, one portfolio, live exposures, tails clipped at the backstop)

| size (all three) | worst trade | worst day | worst week | max DD (path) | P(DD≥30 %) 1 y | P(DD≥50 %) 1 y | P(ruin) | median end × 1 y [p10, p90] |
|---|---|---|---|---|---|---|---|---|
| 0.25× | -7.3 % | -16.7 % | -29.0 % | -90 % | 100 % | 83 % | 2.3 % | 0.79 [0.24, 2.44] |
| 0.5× | -14.6 % | -33.4 % | -51.3 % | -99 % | 100 % | 100 % | 40.8 % | 0.29 [0.03, 3.15] |
| **1× (today)** | -29.2 % | -66.9 % | -82.5 % | -100 % | 100 % | 100 % | 90.0 % | 0.00 [0.00, 0.61] |
| 1.5× | -43.9 % | -100.3 % | -100.1 % | -100 % | 100 % | 100 % | 99.2 % | 0.00 |
| raw (tails kept) 1× | -98.2 % | -108.4 % | -104.8 % | -100 % | 100 % | 100 % | 96.6 % | 0.00 [0.00, 0.04] |

(The yr6W portfolio uses the pre-297/301 WILLY / LITE exits and 1m pricing for part of 2025, so it is the pessimistic block; the master-world
blocks above are the optimistic one. Today's size is ruinous on the first and already drawdown-heavy on the second.)

### Concurrency, correlation, live

- Concurrency (backtest): FADE up to 3 open at once (33 of 625 fills opened beside another fade; 17 days with ≥ 2 fade stops); WILLY 1 by
  design (global hold — while a WILLY is open nothing else automated opens); LITE 2 slots, 295 of 1,455 fills opened beside another LITE.
- Daily P&L correlation between sleeves ≈ 0 (FADE/WILLY −0.00, FADE/LITE −0.00, WILLY/LITE 0.01; 459 days with both WILLY and LITE
  trading, 114 of them both negative). Diversification is real, but uncorrelated variance still adds: three sleeves each near or past
  full Kelly make a combined book far past it.
- **Live B22 (first two live days, $1,025 book):** 2026-10-15 LITE 3 fills −$216 (−21.1 % of the book: EPIC −3.43, DUSK −2.54 within
  10 minutes, ZK −2.93) and WILLY 7 fills −$54 (−5.2 %); 2026-10-16 WILLY ICNT −$86 (−8.3 %). Worst live fill STRK WILLY −$99.46 (−9.7 %);
  worst live day (all bot) −$189 (−18.5 %). Two concurrent LITE stops in one window already cost ≈ 15 % of the book — exactly the
  variance the tables model.

## 4. Fractional-Kelly reference (a reference, NOT a target)

f* = μ / σ² per trade, as notional ÷ book. Concurrency (2 LITE slots, up to 3 fades) would lower the per-position optimum further.

| sleeve | edge source | μ %/trade | haircut | μ used | σ % | full Kelly × book | ½ Kelly | ¼ Kelly | **today × book** | today ÷ ½-Kelly |
|---|---|---|---|---|---|---|---|---|---|---|
| FADE | BT engine replay | +0.043 | 0 % | +0.043 | 1.23 | 2.85 | 1.43 | 0.71 | **9.75** | 6.8 |
| FADE | MASTER today's stack | +0.356 | 50 % | +0.178 | 0.77 | 29.7 | 14.9 | 7.4 | **9.75** | 0.7 |
| FADE | MASTER today's stack | +0.356 | 30 % | +0.249 | 0.77 | 41.6 | 20.8 | 10.4 | **9.75** | 0.5 |
| WILLY | BT MATCH tick | +0.041 | 0 % | +0.041 | 2.10 | 0.93 | 0.47 | 0.23 | **2.44** | 5.2 |
| WILLY | BT yr6W (tails) | -0.104 | 0 % | -0.104 | 2.48 | < 0 | < 0 | < 0 | **2.44** | ∞ |
| WILLY | MASTER + LIVE | +0.360 | 50 % | +0.180 | 1.88 | 5.08 | 2.54 | 1.27 | **2.44** | 1.0 |
| LITE | BT LITE-C all | -0.010 | 0 % | -0.010 | 2.77 | < 0 | < 0 | < 0 | **2.44** | ∞ |
| LITE | BT LITE-C tick | +0.071 | 0 % | +0.071 | 2.77 | 0.92 | 0.46 | 0.23 | **2.44** | 5.3 |
| LITE | MASTER + LIVE | +0.309 | 50 % | +0.155 | 2.85 | 1.91 | 0.95 | 0.48 | **2.44** | 2.6 |

Growth per trade at today's exposure (μx − ½σ²x², % of the book): FADE BT **−0.297** · FADE master-halved +1.451 · WILLY BT MATCH
**−0.031** · WILLY master-halved +0.333 · LITE BT tick **−0.056** · LITE master-halved +0.136. FADE at invest 1.0 (x 4.9): BT ≈ +0.03.

Reading: the only sleeve where *any* defensible edge estimate puts today's size below half-Kelly is the fade on halved master evidence — and
that estimate is the one the backtest contradicts (+0.04). For WILLY and LITE, today's size is at ½-Kelly only on halved master evidence and
2.6–5× full Kelly on the backtest. Nothing here argues for more size.

## 5. Verdict per sleeve, and the pre-committed re-open gates

**SPIKE_FADE — not under-sized; over-exposed at the live book (moderate evidence).** Master 74 · 89 % · +0.36 is stable across Aug / Sep /
Oct, but ≥ half is in-sample and the two-year backtest on the same rules says +0.04 (CI spans 0). Live, the fade carries ≈ 9.75× the book
vs ≈ 3× on paper. Pattern W (N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 ∧ Σ$ > 0) passes on master, but the Pattern-W multiplier ladder assumes the
sizing base is unchanged; going live already tripled it. **Recommendation: no size-up. Operator option (not requested, not shipped):**
`spike_fade_invest_mult` 2.0 → 1.0 brings the live exposure back into the range the evidence was earned at (≈ 4.9× vs paper median 3.1×,
mean 4.1×). 🔒 Re-open a size-up only when ALL hold: ≥ 30 LIVE fades (at the live, uncapped exposure) on ≥ 15 UTC days · WR ≥ 80 % ·
avg ≥ +0.20 % · window-bootstrap P(avg > 0) ≥ 0.95 · no fill ≤ −3.0 % (gap through the backstop) · the master-vs-backtest gap explained;
then at most 1.25×, still inside `max_gross_leverage` 25, with revert to 1× if the next 20 fades average < +0.10 % or any fade day
loses > 20 % of the book.

**FRENZY_WILLY — not under-sized; at or above the growth optimum (weak evidence).** Engine-matched backtest +0.04 (P 0.68), 79 % WR vs
78 % breakeven, 2025 negative; master good but small and partly re-priced in-sample; live 8 fills negative. One gap-through (SAHARA −40 %)
equals the live book at today's size. **Recommendation: no size-up; keep 10× (never raise WILLY leverage — DL 271/292).** 🔒 Re-open only
when: ≥ 50 live WILLY fills on ≥ 20 UTC days under today's exit · avg ≥ +0.25 % with window-bootstrap P ≥ 0.95 · WR ≥ breakeven + 5 pts ·
no fill beyond −5 % · AND the backtest era read (2025 / 2026-H1 / 2026-H2) all ≥ 0; then 1.25× invest (not leverage), revert at 20 fills
if avg < 0 or one day loses > 15 % of the book.

**FRENZY_LITE — not under-sized; above the growth optimum (weak evidence).** Backtest ≈ 0 (tick +0.07, 2025 −0.20), half the master edge is
the in-sample LITE-C re-pricing, live 4 fills negative; even on master evidence 1× already gives P(DD ≥ 50 %) ≈ 95 % in a year because a
±3 % bracket at 2.44× book is ±7 % of the book per fill with two slots. Pattern W cannot apply (WR ≈ 58 %). The DL 276 note already
found 8× ≈ the growth optimum. **Recommendation: no size-up; the live 283 gate (→ 8× if the first 20 fills at 10× average < 0, or a UTC
day's FRENZY fills lose > 25 % of the book) stays the instrument.** 🔒 Re-open a size-up only when: ≥ 50 live LITE fills on ≥ 20 days ·
avg ≥ +0.30 % with window-bootstrap P ≥ 0.95 · the LITE-C revert gate (DL 301) passed its MAIN READ · worst LITE day > −15 % of the book.

**Dream bigger — the evidence path, not incredulity.** The road to $1M runs through *proving* edge, not adding exposure to unproven edge: at
today's sizes a single bad week can halve a $1k book, and compounding cannot recover from ruin. The scale lever is the account (the fade's
liquidity cap and the reserve schedule de-lever automatically as it grows) plus fresh live fills that confirm the master edges; when the
§5 gates pass, the Kelly table says there is room (FADE ⅓-Kelly, WILLY ½-Kelly on halved master edge).

## 6. Inputs, method notes, blind spots

- Inputs: `trading_config.json`, `services/trading_engine.py` (equal-split, liquidity / gross caps); `reports/MASTER_POOL_stacked.csv`
  (STACK 2026-10-16a, `stack_keep` / `stack_pct`); `~/Downloads/scalpars_orders_live_2026-10-16_19-07-25.csv` (B22, 22 live fills);
  `reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv` (6,947 trimmed fills, seed 1, 2025-01-01 → 2026-10-11; fade = today's fade rules);
  `reports/willy_theory/bt.pkl` (WILLY today's exit, RH ticks; MATCH = bar-exact engine fills); `reports/lite_wlsweep/matrix.csv` +
  `menu.csv` (LITE paths, +3/−3 and +2/−3 with −0.2 stop slip; LITE-C rule applied per fill).
- Book: $1,025 assumed (from live margins); exposure ratios are book-independent except the fade's liquidity cap (table in §1).
- Not modelled: funding, the fee-reserve / BNB legs, partial resting-TP fills, the WILLY global hold's effect on the other sleeves' fill
  counts in the master world, exchange bracket re-tiering at larger notionals (only flagged), slippage at 1.5–3× size on $2–20M fade
  pairs (would make larger fades worse, not better).
- Blind spots: WILLY tick windows exclude the gap tails (shown separately via yr6W); LITE 2025 partly 1m-priced (tick-only rows shown);
  master WILLY / LITE spans only 8–9 days (fill frequency in the master world borrowed from the backtest); live = 2 days.
- Washed-out window (Jun-18 → Jul-2 2026) shown apart for every backtest; it contains no master fills (fade started Jul-27).
