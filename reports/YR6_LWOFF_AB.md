# YR6 A/B: FRENZY_LONG + FRENZY_WIDE switched OFF (yr6x / yr6ax)

2026-10-13 · operator-approved · question from `reports/YR6_WL_FRENZY.md` Q3 ("LONG-blocks-LITE cannot be measured from stamps; the decisive test is a replay with LONG / WIDE switched off").

- **A** = yr6 / yr6a, seed 1, LONG / WIDE **ON**.
- **B** = yr6x / yr6ax, seed 1, LONG / WIDE **OFF**.

Both arms ran the same snapshot (`code_yr6_01419d4`), the same flags, chunks and cadence seed, and the same data caches. Fills are trimmed to each chunk window and priced at **today's config (HEAD 443a7b5)**: LONG / WIDE +1.5 at 2× / 1×, LITE +3 at 10×, WILLY 10×, momentum long × 0.5. $ is on a **fixed $2,900 book** unless a row says "compounding". Regimes: **2025 BULL · 2026-H1 BEAR (to 07-01) · 2026-H2 BULL (to 10-11)**.

Seed 1 only. FRENZY seeds follow nearly one path, so seeds 2 and 3 would only have been a computation check, not independent evidence.

---

## Verdict (plain English)

**Once their blocking effect is counted, LONG / WIDE neither clearly help nor clearly hurt the book. Keep them as they are (no config change). The blocking is real and roughly pays for their own losses. They are still not a money-maker.**

1. **Their own trades lose money: −$1,783 over the two years** at today's 2× / 1× (LONG −$1,290, WIDE −$493). This is the same figure as the yr6 study, so arm A reproduces exactly.
2. **The LITE trades they keep out would have lost money.** With LONG / WIDE off, 21 extra LITE trades open on coins that a LONG / WIDE held in A.
   - Those 21 trades: **29 % win rate, −1.46 % per trade, −$2,161**. Day-bootstrap 95 % CI of the average is [−3.08, +0.006].
   - The mechanism is simple. **When the LONG on a coin fails, the LITE on that same coin fails too.** The 13 blocked LITE trades whose LONG / WIDE never reached +1.5 lost every time: 13 / 13 losers, −3.29 % each, −$3,025, CI [−4.98, −2.09].
   - The blocked LITE trades whose holder went on to win also won.
   - At today's +1.5 take-profit, 3 of the 8 "winning-holder" cases would no longer be blocked (the holder takes +1.5 before the LITE entry). Today's blocking benefit is therefore **≈ +$1,950 (18 trades)**, not +$2,161.
   - One trade carries a lot of the loss: SKL 2026-09-20 at −11.48 %, −$812. Without it the 21 trades are −0.76 % per trade, −$1,132.
3. **Whole book, B − A.** Each step below adds one correction to the previous line:
   - **Raw: +$540** (OFF better), CI [−5.0k, +7.0k], P(OFF worse) 0.45.
   - **Cleaned of the replay's own exit noise: +$2,697**, CI [−2.1k, +8.8k], P(OFF worse) 0.17. The noise is −$2,156 and sits entirely in 2025; see §4.
   - **Mechanism-only: about +$160 to +$370 for OFF.** Of the cleaned +$2.7k, **+$2,325 is a BullRun-Long re-sequencing cascade**: 27 BullRun trades exist only in B and 21 only in A, and none of them is on a LONG / WIDE coin. That is path luck, not something LONG / WIDE do. Counting only what LONG / WIDE mechanically cause (their own trades, the LITE they block, and the 9 SURGE winners they block, +$511) gives this range. That is zero within noise.
4. **FRENZY-only compounding book.** ON finishes higher in 2026: **A $20,787 vs B $18,293** (H1 $14.8k vs $12.9k, H2 $3.5k vs $3.1k).
   - The reason is size. A blocked LITE loser costs a 10× position; the LONG / WIDE trade that took its place is 2× / 1×.
   - In 2025 both arms are ruined: $85 vs $113 from $2,900.
   - The FULL compounding book is ruined in both arms in 2025 and 2026 ($95–115). That reflects the rest of the stack (MOM-long etc. at today's size), not this A/B. It says nothing about LONG / WIDE.
5. **No other sleeve moves beyond noise:**
   - **WILLY:** 1,518 → 1,521 fills. Structural effect −$90; the −$514 raw difference is 2025 exit noise. WILLY_HOLD refusals 941 → 865.
   - **MOM-long:** structural −$440. 26 B-only fills at −0.20 %, none on a LONG / WIDE coin, so this is slot / path.
   - **Fades, flips and MOM-short:** within ±$55 structurally. Spike-Fade's raw −$1,224 is entirely 2025 exit noise.

**What this means for the open questions:**
- The operator rule "keep LONG / WIDE ON, they block bad LITE entries on the same coin" (memory feedback_keep_long_wide_on) is **supported in direction**: the blocked LITE trades are losers. It is **not proven at N = 21** (CI touches 0, and one trade is 38 % of the loss).
- DL 280's "blocked LITE lost 8 of 10" read is consistent with this result: 15 of 21 lost here.
- The yr6 Q3 proxy (LITE after a LONG / WIDE in the same episode, +0.49 %) measured a different cohort, the fills after the holder closed. This replay is the direct measurement.
- **Not a ship signal for anything.** The "LITE on a coin whose FRENZY_LONG is failing" signature (13 / 13 losers) was found after looking, so it is **observe-only**. It is already enforced indirectly by the one-position-per-coin rule while LONG / WIDE are on.

---

## 1. Per regime, per sleeve: A (ON) vs B (OFF)

Each cell is N · WR · avg % · $ (fixed book, today's size). Full table incl. shared / A-only / B-only / shifted splits: `reports/YR6_LWOFF_AB_tables.txt` and `YR6_LWOFF_AB_sleeves.csv`.

| regime | sleeve | A (ON) | B (OFF) | B − A $ |
|---|---|---|---|---|
| 2025 BULL | FRENZY_LITE | 828 · 48 % · −0.142 · −$8,320 | 836 · 47 % · −0.148 · −$8,737 | −416 |
| | FRENZY_LONG / WIDE | 270 · −0.061 · −$233 / 205 · −0.168 · −$243 | – | +475 |
| | FRENZY_WILLY | 899 · 69 % · −0.238 · −$15,097 | 901 · 68 % · −0.247 · −$15,706 | −608 (≈ all shared-fill re-exit noise; structural −$90 over 2 years) |
| | MOM-long | 884 · −0.171 · −$13,259 | 892 · −0.175 · −$13,605 | −347 |
| | SURGE_LONG | 127 · −0.153 · −$2,755 | 131 · −0.125 · −$2,321 | +434 |
| | Spike-Fade | 223 · +0.109 · +$6,960 | 223 · +0.082 · +$5,906 | −1,054 (all shared-fill noise) |
| | **TOTAL** | 3,950 · −$37,703 | 3,500 · −$39,227 | **−1,524 raw / +631 cleaned** |
| 2026-H1 BEAR | FRENZY_LITE | 370 · 51 % · +0.226 · +$5,901 | 380 · 50 % · +0.185 · +$4,973 | −928 |
| | FRENZY_LONG / WIDE | 229 · −0.179 · −$579 / 157 · −0.080 · −$89 | – | +668 |
| | FRENZY_WILLY | 411 · +0.101 · +$2,946 | 412 · +0.104 · +$3,040 | +94 |
| | SURGE_LONG | 92 · +0.084 · +$1,096 | 96 · +0.101 · +$1,370 | +274 |
| | **TOTAL** | 2,152 · +$5,464 | 1,789 · +$5,745 | **+281** |
| 2026-H2 BULL | FRENZY_LITE | 256 · 49 % · +0.137 · +$2,485 | 258 · 49 % · +0.080 · +$1,462 | −1,023 |
| | FRENZY_LONG / WIDE | 124 · −0.273 · −$479 / 81 · −0.282 · −$161 | – | +640 |
| | BullRun-Long | 126 · 47 % · +0.051 · +$915 | 129 · 52 % · +0.177 · +$3,225 | +2,309 (re-sequencing) |
| | MOM-long | 189 · −0.026 · −$314 | 193 · −0.037 · −$495 | −181 |
| | **TOTAL** | 1,273 · +$10,735 | 1,080 · +$12,518 | **+1,783** |
| 2 years | **TOTAL** | 7,375 · −0.069 · −$21,503 | 6,369 · −0.059 · −$20,963 | **+540 raw / +2,697 cleaned** |

Matching: **6,244 shared** fills (identical pair / side / strategy / open timestamp). **29 shifted**: the same trade within ±120 s. **1,102 A-only**, of which 1,066 are LONG / WIDE. **96 B-only**.

## 2. FRENZY_LITE: the fills LONG / WIDE kept out

Each row: N · WR · avg % [95 % day-bootstrap CI] · $ [CI].

| regime | B-only LITE, a LONG / WIDE held the coin in A | other B-only (path) | A-only LITE (lost when OFF) |
|---|---|---|---|
| 2025 BULL | 8 · 50 % · +0.006 [−2.17, +2.18] · +$4 | 1 · +3.07 · +$217 | 1 · +3.05 · +$216 |
| 2026-H1 BEAR | 10 · 20 % · −1.313 [−2.84, +0.43] · −$928 | 0 | 0 |
| 2026-H2 BULL | 3 · 0 % · −5.829 [−11.5, −3.0] · −$1,236 | 0 | 1 · −3.02 · −$213 |
| **all** | **21 · 29 % · −1.456 [−3.08, +0.006] · −$2,161 [−$4,619, +$7]** | 1 · +$217 | 2 · +$3 |

- Holders in A: FRENZY_LONG 19, FRENZY_WIDE 2. Exit mix of the B-only LITE: 13 stops, 7 TPs, 2 time caps.
- **At today's +1.5 LONG / WIDE TP.** The replay held the holders to +3. Whether a holder touched +1.5 before the LITE entry was checked on real ticks, with 1m bars as fallback; the replay does not stamp `peak_reached_at`.
  - **13 still blocked (holder never reached +1.5): 0 % WR, −3.29 %, −$3,025**, CI [−4.98, −2.09]. Spread: 2025 3 · H1 7 · H2 3.
  - **5 still blocked (holder hit +1.5 only after the LITE entry): 100 % WR, +3.04 %, +$1,073.**
  - **3 freed today (holder hit +1.5 first): −0.99 %, −$209.** These would trade in A today.
  - Today's blocked set is therefore 18 trades, −$1,952 that ON avoids.
- Engine refusal counters, seed 1, warm-up included (an upper bound): FRENZY_LITE_PAIR_HELD **92 → 66**. The 66 that remain are holds by other sleeves. The reverse direction (FRENZY / WIDE refused because LITE held the coin) was 12 in A.

## 3. Whole-book decomposition of B − A (fixed book, day-bootstrap 95 % CI)

① = LONG / WIDE's own trades removed · ② = structural rest (B-only − A-only + shifted) · ③ = shared-fill re-exit noise.

| regime | ① | ② | ③ (shared re-exited) | **cleaned net ① + ②** | P(OFF worse) |
|---|---|---|---|---|---|
| 2025 BULL | +475 | +156 [−1.9k, +2.2k] | −2,156 (511 / 3,449) | **+631** [−1.5k, +2.9k] | 0.29 |
| 2026-H1 BEAR | +668 | −386 [−1.9k, +1.2k] | −1 (2 / 1,754) | **+282** [−1.4k, +1.9k] | 0.37 |
| 2026-H2 BULL | +640 | +1,143 [−2.9k, +6.5k] | 0 (0 / 1,041) | **+1,783** [−2.2k, +7.2k] | 0.25 |
| **2 years** | **+1,783** | **+913** [−3.9k, +6.8k] | −2,156 | **+2,697** [−2.1k, +8.8k] | 0.17 |

② by sleeve, both years ("coin-held" = a LONG / WIDE held that coin in A at that moment):

| sleeve | coin-held B-only | other B-only | A-only | ② total |
|---|---|---|---|---|
| FRENZY_LITE | 21 · −1.46 % · −$2,161 | 1 · +$217 | 2 · +$3 | −$1,940 |
| SURGE_LONG | 9 · +0.40 % · +$511 | 4 · +$14 | 4 · −1.08 % · −$609 | +$1,135 |
| BullRun-Long | 0 | 27 · +0.51 % · +$1,939 | 21 · −0.07 % · −$196 | +$2,325 (+$190 shifted) |
| MOM-long | 0 | 26 · −0.20 % · −$413 | 8 · +$27 | −$440 |
| FRENZY_WILLY | 0 | 3 · −$97 | 0 | −$90 |
| FLIP / MOM-short / Spike-Fade | 1 · +$25 | 4 | 1 | +$40 / +$53 / −$170 |

**BullRun-Long cascade.** At the B-only BullRun entry times, A already had 2–4 BullRun positions open and almost never a LONG / WIDE. A different BullRun sequence follows from an early divergence, not from LONG / WIDE blocking the trade. I count it as path luck.

**Mechanism-attributable.** Adding ① to the LONG / WIDE-caused parts of ② (the LITE and SURGE / FLIP coin-held trades):
- replay: +1,783 − 2,161 + 511 + 25 = **+$158** for OFF;
- at today's +1.5 TP: +1,783 − 1,952 + 536 ≈ **+$367** for OFF;
- a sign flip on the LITE count is within its CI.

## 4. The replay's own noise floor (important for every A/B on this harness)

- 513 of the 6,244 shared fills exit differently between arms: same entry, exit seconds apart. The split is symmetric: 255 better, 258 worse.
- **All 511 of the 2025 ones are on pair-days without tick data** (crosstab: tick-covered shared fills changed 0 / 1,713; 1m-path fills changed 511 / 1,736).
- 2026 (ticks on 99 % of position-minutes) has 2 changed fills in H1 and 0 in H2.
- The cause is the stepped 1m path with random intra-minute ordering. Its seeded RNG stream diverges as soon as the fill lists differ.
- The sum of this noise is −$2,156. That is as large as the effect being measured. **Any A/B on the 2025 run must compare structurally (only-in fills) or hold shared fills fixed, never raw totals.**

## 5. Verification (two independent computations plus checks)

1. **Stdlib recount.** I recounted from the raw replay `*_orders.csv` and meta windows with csv + datetime only: my own trim, key and re-price code, no pandas, no yr6_tables.
   - Fill counts: A 7,375 / B 6,369, identical to the pipeline.
   - Every strategy's N and avg % matched to 4 decimals (all FRENZY sleeves printed in the tables file; no MISMATCH line for any other).
   - B-only + shifted LITE matched: N 30, avg −0.8530.
2. **A arm re-priced from scratch at HEAD 443a7b5.** `ENGINE_REPLAY_YR6LWA_TODAY` reproduces the published `ENGINE_REPLAY_YR6_2Y_TODAY` exactly: every sleeve row, and LONG −$1,290 / WIDE −$493 as in YR6_WL_FRENZY.
3. **Identity.** B − A $+540.24 = B-only − A-only + shifted Δ + shared Δ ($+540.24).
4. **Noise attribution.** The tick-coverage crosstab in §4 explains every 2025 re-exited fill.
5. **Data parity.**
   - **10-11 chunk.** It ran on a shadow cache without the 12 `ticks/<pair>/2026-10-12.npz` files that another agent wrote on 10-13, which yr6 never saw. Its shared fills are identical to A, including W's −0.564 exit at 10-12 11:01.
   - **10-01 chunk.** B has no position open across 10-11 00:00. (yr6 ran that chunk before the 10-11 data existed.)
   - **2025 cache.** Unchanged since yr6a.
   - **Clean runs.** 45 runs, rc = 0, no tracebacks; ondemand 1m fetches 0 (286 blocked misses, as in the parent runs).

## 6. Limits (state with every number)

- **Seed 1 only.** One path.
- **Entries are from config 01419d4**, where LONG / WIDE TP = +3. The +1.5 re-price is exact for each fill's own %. Second-order effects of today's earlier LONG / WIDE exits are only partly modelled:
  - §2 handles the 3 freed LITE trades;
  - re-entries after a +1.5 exit are not modelled.
- **WILLY mcap.** WILLY uses the declared "mcap always readable" proxy, as in yr6.
- **2025 data.** Non-top-50 pair-days run on 1m bars (yr6a fidelity limit).
- **Small numbers.** The LITE blocked cohort is N = 21, about 21 distinct days. It is a measurement, not a gate pass.

## 7. Files and commands

- Launch / resume: `scripts/yr6x_launch.sh`.
  - Config: `reports/backtest_cache/replay/frozen_config_yr6x_lwoff.json`, sha256[:12] **33b5784891bd** (A: 83ecb8787a5b).
  - Outputs: `replay/yr6x_*_s1_*`, `replay/yr6ax_*_s1_*`.
  - Logs: `replay/year/yr6x_orchestrator.log`, `replay/year/yr6ax_orchestrator.log`.
- Tables, from the snapshot dir: `yr6_tables.py --prefix yr6ax,yr6x --cmp "" --out-tag YR6X_2Y` → `reports/ENGINE_REPLAY_YR6X_2Y_*`.
- Re-price (git-archive of 443a7b5 + working-tree `yr6_reprice_today.py`):
  - B: `--fills ENGINE_REPLAY_YR6X_2Y_fills.csv --years 2025,2026 --tag YR6X_2Y_TODAY --no-book`
  - A: `--fills ENGINE_REPLAY_YR6_combined_fills.csv --years 2025,2026 --tag YR6LWA_TODAY --no-book`
- Study, run from the 443a7b5 code dir: `venv/bin/python scripts/study_yr6_lwoff_ab.py`.
  - Outputs: `reports/YR6_LWOFF_AB_{tables.txt, sleeves, decomp, decomposition, net, lite_bonly, books}.csv`.
