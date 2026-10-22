# WILLY late joiners — should WILLY wait for pairs that join the shortlist after their spike? (2026-10-22)

## Verdict: keep today's rule (R0). Don't widen the late window.

**Plain version.** You were right about how often it happens. About 1 FRENZY episode in 8 has a pump coin that only joins the shortlist
20–60 minutes after its spike, and in 86 % of these cases the reason is the one you saw on BTR: the 24 h volume was below $20M until
the pump itself pushed it over. These are also the bigger pumps (median +20 % in the first hour, against +13 % for pairs that were
already on the list).

**But buying them with WILLY loses money.** Under today's WILLY rules (first red 5m bar, TP +1.25 resting, −4 stop, 90-min cap,
late TP +0.25 after 30 min, turnover routing), the late joiners average **−0.27 % per trade over 720 trades on 394 days (2025 → today)**.
That is negative in every period, and the 95 % range of the average is fully below zero.

They are not worse than WILLY's normal trades. Over the same period the on-time trades average −0.46 %, or −0.36 % without the
2025-10-10 crash day, and both are negative with this walker (see the calibration note). They just aren't good enough to add.
Each added trade loses about a quarter of a percent on average.

**Today's BTR would have won** (+1.25 TP, 40 min after the spike). The other 12 engine refusals of the last 14 days add up to about
−2 points, so the live sample tells the same story: one memorable winner, no edge.

**Pre-registered bar (frozen before any result, `WILLY_LATE_JOIN_2026-10-22_PREREG.md`). All three candidate rules fail:**

| condition | R1-30 (joined ≤ 30 min) | R1-60 (≤ 60 min) | R2 (≤ 60 min, wait from spike) |
|---|---|---|---|
| ① avg ≥ on-time avg − 0.20 (on-time −0.456) | −0.256 ✅ | −0.267 ✅ | −0.285 ✅ |
| ② avg > 0 in 2026-H1 **and** 2026-H2 | −0.32 / −0.51 ❌ | −0.30 / −0.44 ❌ | −0.28 / −0.46 ❌ |
| ③ day-bootstrap 95 % lower bound > 0 | −0.49 ❌ | −0.43 ❌ | −0.46 ❌ |
| ④ N ≥ 30 on ≥ 10 days | 350 / 250 ✅ | 720 / 394 ✅ | 671 / 380 ✅ |
| ⑤ not 1–2 pairs (avg > 0 without top 2) | −0.30 ❌ (top-2 share 5 %) | −0.29 ❌ | −0.31 ❌ |

The rules pass ① and ④ but fail ②, ③ and ⑤. Under the pre-registration that allows an **observe-only line at most** and rules out
arming. My recommendation: no engine change. If you want the live record anyway, add a zero-cost observe line to the scout. The
engine already logs every refusal with its spike time and age ("late window — no entry A"), so the scout could price the L ≤ 12
refusals on WILLY's exit. Frozen gate: at 30 priced refusals on ≥ 10 days, re-open only if avg > 0 with a day-bootstrap lower bound
> 0. It would take about 4 months at the live rate of ~2 a week.

**Would it add trades?** Yes. R1-30 adds 3.7 trades a week (3.6 once WILLY's single slot and HT's single slot are respected).
R1-60 adds 7.7 a week (7.2 after slots). Over the 94 weeks studied, R1-60 would have added about −170 points of position P&L, and
R1-30 about −66.

## Q1 — How often? (engine-parity FRENZY episodes, every USDT perp incl. delisted, 2025-01-01 → 2026-10-22 10:05)

L = how many 5m bars after the spike bar the pair first appears on the shortlist while flagged. This is exactly the "N bars old" the
engine logs.

| L (bars) | 2025 | 2026-H1 | 2026-H2 | All | share |
|---|---|---|---|---|---|
| 0 (already on the list at the spike) | 1,975 | 992 | 602 | 3,569 | 51.6 % |
| 1–3 (today's late window catches it) | 316 | 149 | 57 | 522 | 7.5 % |
| **4–6 (joined 20–30 min after)** | 218 | 140 | 78 | **436** | **6.3 %** |
| **7–12 (35–60 min after)** | 205 | 159 | 98 | **462** | **6.7 %** |
| 13–36 (1–3 h after) | 354 | 282 | 170 | 806 | 11.6 % |
| 37–288 (3–24 h after) | 438 | 436 | 248 | 1,122 | 16.2 % |
| All | 3,507 | 2,158 | 1,255 | 6,920 | |

- **About 9.5 late joiners a week** (L 4–12): 13 % of all episodes that ever reach the shortlist.
- Why they were late (L 4–12): the 24 h volume was under $20M at the spike in 776 cases (86 %), the 24 h move was under 15 % in 50,
  and both legs were short in 72. **The hypothesis is confirmed: the volume threshold is what holds them back.**
- Late joiners are the bigger pumps: median first-hour pump +19.9 % against +12.6 % for on-time pairs.
- **Parity with the real engine:** in the 10-08 → 10-22 window the walk's late joiners match 12 of the 13 CloudWatch refusals with
  L ≤ 12, and the lateness is identical on 10 of the 12 matches. The walk misses KAIA 10-12. It has 3 extras the engine never logged
  (ARIA, BLUAI, MASK), most likely pairs the engine already followed, the 25-pair cap, or the listing pre-filters.

## Q2 — The outcome under each rule (WILLY's exact current entry and exit, routed by turnover)

Routing: R < 1 → full WILLY; R ≥ 1 → WILLY_HT (¼ size, DL 341); unknown supply → refused live, not counted. All numbers are net %
per position.

**Routing surprise:** most late joiners are **not** high-turnover. A pair that has only just reached $20M of 24 h volume rarely has
volume above its market cap. In R1-60: 610 full WILLY (68 %) · 110 HT (12 %) · 178 unknown supply (20 %). That compares with
29 % HT among on-time trades. The HT share is the worst part: −0.57 % per trade (R1-60) and −0.99 % (R1-30), against −0.21 % /
−0.14 % on full WILLY.

### ③ Backtest (spikes 2025-01-01 → WILLY deploy 2026-10-08) — before → after

| rule | before (R0 on-time) | added late joiners | after |
|---|---|---|---|
| R1-30 | 3,349 · −0.459 % · Σ −1,537 | 343 · 72 % WR · −0.269 % · Σ −92 | 3,692 · −0.441 % · Σ −1,629 |
| R1-60 | 3,349 · −0.459 % · Σ −1,537 | 706 · 72 % WR · −0.270 % · Σ −191 | 4,055 · −0.426 % · Σ −1,728 |
| R2 | 3,349 · −0.459 % · Σ −1,537 | 659 · 71 % WR · −0.284 % · Σ −187 | 4,008 · −0.430 % · Σ −1,724 |

The "after" average looks better only because the added trades are less bad than the base. The total gets worse.

### ② Master (10-08 → 10-14, paper) — the engine's OWN refusals, priced · before = the real WILLY fills (master `stack_pct`)

| rule | before (real WILLY) | added (refusals priced) | after |
|---|---|---|---|
| R1-30 | 19 · +0.591 % · Σ +11.2 | 3 · LUMIA +1.25, WAL +1.25, METIS +1.25 · Σ +3.75 | 22 · +0.680 % · Σ +15.0 |
| R1-60 | 19 · +0.591 % · Σ +11.2 | 8 · + KAIA −3.42 (cap), RED +1.25, S +1.25, AVA(HT) +1.25, WAL −4.00 · Σ +0.08 | 27 · +0.419 % · Σ +11.3 |
| R2 | 19 · +0.591 % · Σ +11.2 | 7 (RED expires) · Σ −1.17 | 26 · +0.387 % · Σ +10.1 |

### ① Live (10-14 18:36 → now) — the engine's OWN refusals, priced · before = the real live WILLY fills

| rule | before (real WILLY) | added (refusals priced) | after |
|---|---|---|---|
| R1-30 | 17 · −0.102 % · Σ −1.7 | 2 · GOAT(HT) −4.00, AVA +0.25 · Σ −3.75 | 19 · −0.289 % · Σ −5.5 |
| R1-60 | 17 · −0.102 % · Σ −1.7 | 5 · + ANIME +0.25, IOST +1.25, **BTR +1.25** · Σ −1.00 | 22 · −0.125 % · Σ −2.7 |
| R2 | 17 · −0.102 % · Σ −1.7 | 4 (IOST expires) · Σ −2.25 | 21 · −0.190 % · Σ −4.0 |

Across ①+② under R1-60 that is 13 refusals, about 1 a day: 10 winners, 3 losers, Σ −0.9 points. BTR is one of the winners.

## Q3 — Are late joiners a different cohort? (all walk periods, routed)

| | on-time (R0) | late R1-30 | late R1-60 |
|---|---|---|---|
| trades · days · pairs | 3,436 · 643 · 440 | 350 · 250 · 223 | 720 · 394 · 316 |
| win rate | 69.7 % | 72.3 % | 71.8 % |
| avg net % (day-bootstrap 95 %) | −0.456 (−0.66, −0.30) | −0.256 (−0.49, −0.02) | −0.267 (−0.43, −0.11) |
| avg without the 2025-10-10 crash day | −0.362 | −0.258 | −0.278 |
| 2025 · 2026-H1 · 2026-H2 | −0.61 · −0.28 · −0.27 | −0.09 · −0.32 · −0.51 | −0.15 · −0.30 · −0.44 |
| stop rate · worst trade | 27.8 % · −40.3 (crash gap) | 24.9 % · −9.1 | 24.7 % · −9.1 |
| avg win · avg loss | +1.13 · −4.11 | +1.16 · −3.96 | +1.13 · −3.83 |
| minutes spike → entry (median, IQR) | 5 (5–10) | 30 (25–35) | 40 (30–50) |
| price at entry vs pre-spike base (median) | +6.8 % | +12.2 % | +12.4 % |
| full WILLY · HT | −0.51 · −0.32 | −0.12 · −1.05 | −0.21 · −0.57 |
| $-weighted (HT × ¼) | −0.494 | −0.161 | −0.228 |
| top-2 pairs' share of the positive Σ | 5 % | 5 % | 4 % |

- **Win rate is too low for this exit.** With wins of about +1.13 and losses of about −3.9, WILLY needs a ~77 % win rate to break
  even. Late joiners win 72 %.
- **Entering later is not the problem on its own.** Fair null: the on-time episodes re-entered at the same 20–60 min lag average
  **−0.273**, and the real late joiners at the same lag mix average **−0.267**. So a late joiner behaves like any FRENZY coin bought
  30–40 minutes after its spike. It is not a special "dead-coin" edge, and not a special trap either.
- No timing bucket works: entries ≤ 30 min −0.29 · 30–45 min −0.24 · 45–60 min −0.34 · 60–80 min −0.02 (N 45).
- Nothing is concentrated in one or two pairs. The negative result is broad.

## Calibration and blind spots (read before trusting the absolute level)

- **This walker is about 0.14 %/trade more pessimistic than the tick walker.** It uses 1m bars; a bar touching both the TP and the
  stop counts as a stop; slip is 0.10 % against the 0.035 % in the tick study, which accounts for 0.065 of the gap. On the 832
  on-time fills both walkers priced, it gives −0.094 against +0.042 with ticks, with the same exit on 92 %. The **comparison**
  between late and on-time uses the same walker, so it is fair. Adding the full +0.14 back still leaves the late cohort negative
  (R1-60 ≈ −0.13) with an upper bound near zero. It still fails ② and ③.
- On the 36 real master WILLY fills, the walker re-walked from the booked entry gives the same exit and the same % (±0.10) on 36/36.
- Turnover R uses today's circulating supply × the price at the time (bot source `bapi_supply.json`). Older dates and changes in
  supply are approximate, and 20 % of late joiners have no supply (live would refuse them).
- Not modelled: the shortlist's 25-pair cap; the follow list (a pair the engine already follows is "warm" and arms on time); the
  dislocation and lateness guards inside the open path; the pair already being held by another sleeve; the episode ending inside
  the wait.
- The slot model is simple: an entry that finds the slot full is dropped. Live, it would stay armed and might fill later.
- The live comparator is small: 17 live and 19 master WILLY fills. WILLY / HT fills after the 10-21 02:14 export are not included.
- Late joiners that join more than an hour after the spike (1,931 episodes, L > 12) were not priced. They are out of scope for any
  rule here, and the engine's 12-bar clamp keeps them out too.

## Files
- Pre-registration: `reports/WILLY_LATE_JOIN_2026-10-22_PREREG.md`
- Tables (script output): `reports/WILLY_LATE_JOIN_2026-10-22_tables.md`
- CSVs: `reports/WILLY_LATE_JOIN_2026-10-22_entries_priced.csv` (every episode × rule incl. the null, priced) ·
  `_cohort_stats.csv` · `_q1_lateness.csv` · `_live.csv` (engine refusals priced) · `_onTime_live.csv` (real WILLY fills re-walked)
- Data: `reports/WILLY_LATE_JOIN_2026-10-22_data/` (episodes.csv, entries.csv, CloudWatch refusals, 1m klines)
- Scripts: `scripts/study_willy_latejoin_{common,build,fetch,price,live,report}.py`
- `scripts/validate_against_master.py`: ALL CHECKS PASS (2026-10-22).
