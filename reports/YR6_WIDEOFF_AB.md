# YR6 A/B: FRENZY_WIDE switched off ALONE (yr6w / yr6aw)

2026-10-14 · operator-approved · follow-up to `reports/YR6_LWOFF_AB.md` (LONG + WIDE both off).

Three arms, all seed 1:

| arm | runs | FRENZY_LONG | FRENZY_WIDE | frozen config (sha256[:12]) |
|---|---|---|---|---|
| **A** | yr6 / yr6a | ON | ON | frozen_config_yr6_01419d4.json (83ecb8787a5b) |
| **B** | yr6x / yr6ax | OFF | OFF | frozen_config_yr6x_lwoff.json (33b5784891bd) |
| **C** | yr6w / yr6aw (this run) | ON | **OFF** | frozen_config_yr6w_wideoff.json (**d11dec3bb4e1**) |

All three arms share these settings:
- **Code and inputs:** the same snapshot (`code_yr6_01419d4`), flags, chunks, cadence seed and data caches. Both shadow-cache runs of the 10-11 chunk (B and C) exclude every tick file dated 2026-10-12 (§6).
- **Re-pricing:** fills are re-priced to **today's config** with the YR6_LWOFF_AB pricing (HEAD 443a7b5): LONG / WIDE +1.5 at 2× / 1×, LITE +3 at 10×, WILLY 10×, momentum long × 0.5.
- **Money:** $ is on a **fixed $2,900 book** unless a row says "compounding".
- **Regimes:** 2025 BULL · 2026-H1 BEAR (to 07-01) · 2026-H2 BULL (to 10-11).

**HEAD 2338871 (Oct-13) added a SURGE_LONG EMA13−EMA50 gap gate** (DECISION_LOG 287). That gate is an entry filter, so re-pricing cannot model it. Every headline below therefore comes in two versions: the replay number and the number "with today's SURGE gap gate" (SURGE fills that exist in only one arm and have gap < 0 are dropped).

---

## Verdict (plain English)

**Switch FRENZY_WIDE off. Keep FRENZY_LONG on.** This is a recommendation; the config change is the operator's call.

1. **WIDE loses money on its own trades in every regime.** Over 2 years it made 443 trades: 63 % WR, −0.157 % per trade, **−$493**.
   - By regime: 2025 −0.168 % · H1 −0.080 % · H2 −0.282 %.
   - At +1.5 / −3, WIDE needs a **67 % breakeven WR**. It wins 63 % (60–65 % per regime).
   - Day-bootstrap CI of the 2-year average is [−0.37, +0.05]. On its own trades alone, that is not proven at 95 %.
2. **The "keep WIDE on, it blocks bad LITE" argument does not hold for WIDE. That protection comes from LONG.**
   - Of the 21 LITE trades that LONG + WIDE kept out in the last A/B, **19 were held by a LONG and only 2 by a WIDE**.
   - With WIDE off, exactly those 2 LITE trades appear:
     - ATH 2025-09-08: +3.06 %, +$216.
     - STEEM 2026-03-01: −0.09 %, −$6.
   - So WIDE's LITE-blocking was worth **−$210 to the book**: it blocked a winner.
   - All 19 LONG-blocked LITE trades stay blocked in C, still held by the same LONG.
   - The LONG fills are **identical in A and C: 623 / 623 exact keys**.
3. **WIDE also blocks SURGE_LONG winners on the same coin.**
   - 5 SURGE trades appear in C on a coin WIDE held in A: +1.29 % average, **+$911**.
   - 3 of them would be refused today by the new SURGE gap gate (BNX +$183, ZEC −$170, TAKE +$395). That leaves ZK +$313 and STRK +$191, so **+$504** at today's HEAD.
4. **Whole book, C − A.** Each line adds one correction to the one before:
   - **Raw: −$1,265**, P(C worse) 0.78. This is entirely the replay's own 2025 exit noise (point 6).
   - **Cleaned** (shared fills held at their A outcome): **+$1,661**, CI [+$69, +$3,386], **P(C worse) 0.02**.
   - **Cleaned, with today's SURGE gap gate: +$1,254**, CI [−$119, +$2,729], **P(C worse) 0.04**.
   - The cleaned result is **positive in all three regimes**:

     | regime | cleaned C − A | with today's gap gate |
     |---|---|---|
     | 2025 BULL | +$816 (P 0.11) | +$804 |
     | 2026-H1 BEAR | +$610 (P 0.05) | +$215 |
     | 2026-H2 BULL | +$235 (P 0.18) | +$235 |

   - Mechanism-attributable part:
     - Replay: +$1,614.
     - At today's HEAD: **≈ +$1,207** = WIDE's own +$493, plus the blocked LITE +$210, plus the blocked SURGE +$504.
     - Path noise in the remainder: MOM-long −$99, BullRun −$44, WILLY +$93, FLIP +$15, one FARTCOIN SURGE re-timed +$292.
   - Unlike the LONG + WIDE A/B, **there is no BullRun re-sequencing cascade here**: BullRun moves only −$44. The C − A result is mechanical, not path luck.
   - In-sample haircut of 30–50 %: **≈ +$600–850 over 21 months** on a fixed $2,900 book. A small but consistent gain.
5. **FRENZY-only compounding book, 2026, from $2,900:**

   | period | A (both on) | B (both off) | C (WIDE off) |
   |---|---|---|---|
   | 2026 full | $20,787 | $18,293 | **$21,866** |
   | H1 | $14,789 | $12,853 | **$15,121** |
   | H2 | $3,466 | $3,060 | **$3,672** |

   - In 2025, all three FRENZY-only books are ruined ($85 / $113 / $91).
   - The FULL book is ruined in all three arms in 2025 and 2026 (rest of the stack, not this A/B). The exception is the H2-only FULL book: A $24.3k · B $24.4k · C $25.0k.
6. **Noise floor, the lesson from the last A/B, reproduced.**
   - 536 of the 6,915 shared C / A fills re-exit differently: 267 better, 269 worse, **−$2,926 in total**.
   - **Every one of them is a 2025 fill on a pair-day without tick data**: 536 of 1,877 such fills changed, and 0 of 1,857 tick-covered fills.
   - 2026 has **0** re-exits.
   - This noise accounts for the whole raw 2025 gap: Spike-Fade −$2,019 and LITE −$860 on shared fills. No mechanism behind it.

**What LONG alone adds (C vs B).** LONG **mechanically pays for itself and more**.
- Its own trades: **−$1,292** (623 trades, 64 % WR, −0.147 %).
- The LITE trades it keeps out: **19 trades, 26 % WR, −1.77 %, −$2,371**. Day-CI of the average [−3.41, −0.19]; $ CI [−$4,970, −$217]. This is now significant at 95 %, which it was not in the pooled LONG + WIDE read.
- **At today's +1.5 TP: 15 trades still blocked, −$2,162.**
  - 12 trades where the holding LONG never reached +1.5: all 12 lost, −3.56 % each, −$3,019.
  - 4 trades where the holding LONG reached +1.5 only after the LITE entry: all 4 won, +$857.
  - 3 more trades are freed today: the holding LONG hits +1.5 before the LITE entry, so it no longer blocks them.
- It also keeps out 4 SURGE trades: −0.71 %, −$400.
- **Mechanism: switching LONG off costs ≈ $1,245 (today) to $1,454 (replay).**
- The cleaned whole-book B − C is +$1,038 (CI [−$3.6k, +$6.9k], P 0.38). That figure is carried by the same **+$2,369 BullRun-Long re-sequencing cascade** as last time: 26 B-only BullRun trades, none on a coin LONG held. Without the cascade it is ≈ −$1,330, consistent with the mechanism.
- The operator rule "keep LONG on, it blocks bad LITE entries" (feedback_keep_long_wide_on) is **supported for LONG and refuted for WIDE**.

**Gate status.** This is an A/B measurement, not a filter promotion. The decision is "turn a losing sleeve off whose claimed side-benefit (LITE blocking) is ≈ 0".
- The evidence base is 443 WIDE trades plus a whole-book cleaned P(C worse) of 0.02 / 0.04.
- It is one seed and one path, and the SURGE part is N = 5.
- **Proposed pre-committed revert gate** (to be locked if shipped): after WIDE is off, a scout counts the LITE / SURGE fills that open on a coin where the WIDE entry rule would have fired (observe-only stamp). **Revert if those fills reach N ≥ 15 over ≥ 8 windows with avg ≤ −1.0 %**. In the replay that cohort is +$210.

---

## 1. Per regime, per sleeve: A (both on) vs C (WIDE off)

Each cell is N · WR · avg % · $ (fixed book, today's size). The full tables, including the shared / only-in / shifted splits for C vs A, B vs C and B vs A, are in `reports/YR6_WIDEOFF_AB_tables.txt` and `YR6_WIDEOFF_AB_sleeves.csv`.

| regime | sleeve | A (ON) | C (WIDE OFF) | C − A $ |
|---|---|---|---|---|
| 2025 BULL | FRENZY_WIDE | 205 · 63 % · −0.168 · −$243 | – | +243 |
| | FRENZY_LONG | 270 · 66 % · −0.061 · −$233 | 270 · 66 % · −0.061 · −$234 | −2 (1 fill exit noise) |
| | FRENZY_LITE | 828 · 48 % · −0.142 · −$8,320 | 828 · 47 % · −0.157 · −$9,180 | −860 (shared-fill noise) |
| | FRENZY_WILLY | 899 · −0.238 · −$15,097 | 900 · −0.237 · −$15,069 | +28 |
| | SURGE_LONG | 127 · 28 % · −0.153 · −$2,755 | 130 · 30 % · −0.116 · −$2,137 | +618 |
| | Spike-Fade | 223 · +0.109 · +$6,960 | 223 · +0.072 · +$4,941 | −2,019 (all shared-fill noise) |
| | MOM-long | 884 · −0.171 · −$13,259 | 887 · −0.172 · −$13,330 | −71 |
| | **TOTAL** | 3,950 · −0.145 · −$37,703 | 3,752 · −0.148 · −$39,812 | **−2,110 raw / +816 cleaned** |
| 2026-H1 BEAR | FRENZY_WIDE | 157 · 65 % · −0.080 · −$89 | – | +89 |
| | FRENZY_LONG | 229 · 63 % · −0.179 · −$579 | identical | 0 |
| | FRENZY_LITE | 370 · 51 % · +0.226 · +$5,901 | 371 · 50 % · +0.225 · +$5,895 | −6 |
| | SURGE_LONG | 92 · 36 % · +0.084 · +$1,096 | 93 · 37 % · +0.113 · +$1,491 | +395 (TAKE; gap-gated today) |
| | MOM-long | 385 · −0.090 · −$3,090 | 387 · −0.086 · −$2,972 | +117 |
| | **TOTAL** | 2,152 · +0.011 · +$5,464 | 2,000 · +0.021 · +$6,074 | **+610** |
| 2026-H2 BULL | FRENZY_WIDE | 81 · 60 % · −0.282 · −$161 | – | +161 |
| | FRENZY_LONG / LITE / WILLY / BullRun | identical | identical | 0 |
| | SURGE_LONG | 85 · +0.547 · +$6,569 | 86 · +0.556 · +$6,759 | +191 (STRK) |
| | MOM-long | 189 · −0.026 · −$314 | 191 · −0.032 · −$431 | −117 |
| | **TOTAL** | 1,273 · +0.032 · +$10,735 | 1,195 · +0.053 · +$10,970 | **+235** |
| 2 years | **TOTAL** | 7,375 · −0.069 · −$21,503 | 6,947 · −0.065 · −$22,768 | **−1,265 raw / +1,661 cleaned** |

**Matching, C vs A:**
- **6,915 shared** fills: identical pair / side / strategy / open timestamp.
- **11 shifted**: the same trade within ±120 s.
- **449 A-only**, of which 443 are WIDE.
- **21 C-only.**

## 2. FRENZY_WIDE's own trades (A, today's +1.5 TP, 1×)

| regime | N · WR · avg % · $ | day-CI avg | exits |
|---|---|---|---|
| 2025 BULL | 205 · 63 % · −0.168 · −$243 | [−0.47, +0.14] | 102 TP / 102 stop / 1 cap |
| 2026-H1 BEAR | 157 · 65 % · −0.080 · −$89 | [−0.45, +0.28] | 79 / 76 / 2 |
| 2026-H2 BULL | 81 · 60 % · −0.282 · −$161 | [−0.79, +0.20] | 41 / 40 / 0 |
| **all** | **443 · 63 % · −0.157 · −$493** ($ CI [−$1,149, +$161]) | [−0.37, +0.05] | 222 / 218 / 3 |

Average win +1.49 %, average loss −3.02 %, so breakeven WR = 66.9 %. The sleeve runs at 63.6 %.

## 3. FRENZY_LITE: the fills that differ

**C vs A (the LITE trades WIDE kept out).** All C-only and A-only LITE fills, exact key:

| fill | outcome | holder of the coin in the other arm |
|---|---|---|
| C-only ATH 2025-09-08 17:35 | +3.06 % · +$216 TP | A: FRENZY_WIDE (its holder hit +1.5 only after the LITE entry → still blocked at today's TP) |
| C-only STEEM 2026-03-01 02:20 | −0.09 % · −$6 cap | A: FRENZY_WIDE (holder never reached +1.5) |
| A-only STRK 2025-11-15 02:15 | +3.05 % · +$216 TP | C: none (path) |
| shifted ±6 s: THE 09-27, MINA 11-02, STRK 11-17, ZK 2026-10-10 | Σ Δ +$5 | — |

WIDE-held LITE blocks total 2 fills, +$210 kept out. The engine's FRENZY_LITE_PAIR_HELD counter reads 92 (A) → 90 (C) → 66 (B).

**C vs B (the LITE trades LONG keeps out)**, by regime:

| regime | N · WR · avg % · $ |
|---|---|
| 2025 | 7 · 43 % · −0.43 · −$212 |
| H1 | 9 · 22 % · −1.45 · −$922 |
| H2 | 3 · 0 % · −5.83 · −$1,236 (SKL −11.48 % is one of them) |
| **all** | **19 · 26 % · −1.77 · −$2,371**, avg CI [−3.41, −0.19] |

Each listed fill and its today's-TP class is in the tables file, and in `YR6_WIDEOFF_AB_lite_BvsC.csv` / `_CvsA.csv`.

**LONG-holder blocks match A.** All 19 LITE trades that a LONG kept out in A are absent in C, and C's LONG holds each of those coins at the same moment. The 2 WIDE-held ones are present in C. FRENZY_LONG is identical across arms: 623 / 623 shared exact keys, 0 only-in, 2-year $ −$1,290 vs −$1,292. The $2 is one 2025 non-tick fill's exit noise.

## 4. Whole-book decomposition (fixed book, day-bootstrap 95 % CI)

① = removed sleeve's own trades · ② = structural rest (only-in + shifted) · ③ = shared-fill re-exit noise · cleaned = ① + ②.

**C − A (WIDE off):**

| regime | ① | ② | ③ noise (re-exited / shared) | **cleaned** [CI] · P(C worse) | cleaned with today's SURGE gap gate |
|---|---|---|---|---|---|
| 2025 BULL | +243 | +573 | −2,926 (536 / 3,734) | **+816** [−472, +2,231] · 0.11 | +804 [−401, +2,117] · 0.10 |
| 2026-H1 BEAR | +89 | +521 | 0 (0 / 1,990) | **+610** [−65, +1,522] · 0.05 | +215 [−211, +660] · 0.17 |
| 2026-H2 BULL | +161 | +74 | 0 (0 / 1,191) | **+235** [−273, +798] · 0.18 | +235 [−273, +798] · 0.18 |
| **2 years** | **+493** | **+1,168** [−310, +2,833] | −2,926 | **+1,661** [+69, +3,386] · **0.02** | **+1,254** [−119, +2,729] · **0.04** |

② by sleeve, both years:
- SURGE_LONG **+1,202**: 5 on a WIDE-held coin, +$911 (BNX, ZEC, ZK, TAKE, STRK); FARTCOIN re-timed by 6 min, +$122 vs −$170.
- MOM-long −99
- BullRun −44
- WILLY +93
- FLIP +15
- LITE −1

**B − C (LONG off, WIDE already off):**
- ① +1,292
- ② −254, of which BullRun cascade +2,369, LITE −1,938, SURGE −67, MOM-long −342
- ③ +767 (459 2025 non-tick fills)
- **cleaned +1,038** [−3,620, +6,889], P(B worse) 0.38
- Mechanism-attributable: **−1,454** (replay) / **−1,245** (today's TP). Switching LONG off loses that much.

**B − A (code check):** reproduces `YR6_LWOFF_AB.md` exactly: raw +$540, cleaned +$2,697, 513 shared re-exited, 21 coin-held LITE −$2,161, mechanism +$158 / +$367.

## 5. Replay noise floor (tick coverage of shared fills)

| comparison | 2025 non-tick shared: re-exited · Σ Δ$ | 2025 tick-covered | 2026 (all) |
|---|---|---|---|
| C vs A | 536 / 1,877 · −$2,926 (267 better / 269 worse) | 0 / 1,857 | 0 / 3,181 |
| B vs C | 459 / 1,742 · +$768 | 0 / 1,724 | 2 / 2,809 (−$1) |
| B vs A | 511 / 1,737 · −$2,156 | 0 / 1,712 | 2 / 2,795 |

The stepped-1m path's intra-minute ordering RNG diverges as soon as two arms' fill lists differ. That noise is symmetric in count and **±$3k in sum**: as large as anything measured here. Any A/B on the 2025 run is read structurally only.

## 6. Verification (two independent computations plus checks)

1. **Stdlib recount** from the raw replay `*_orders.csv` and meta windows (own trim / key / re-price code, csv + datetime only):
   - A 7,375 / B 6,369 / C 6,947 fills.
   - **Key sets equal to the pipeline in all three arms.**
   - Every strategy's N and avg % match to 1e-6: 0 mismatches.
2. **Code check.** The generalised study script reproduces every B − A number of the previous A/B.
3. **Identities.** C − A −$1,264.77 = only-in + shifted Δ + shared Δ (−$1,264.77); B − C and B − A identities also hold.
4. **Structural invariants.**
   - LONG fills identical A ↔ C (623 / 623).
   - All 19 LONG-held LITE blocks persist in C; both WIDE-held ones appear.
   - The 10-11 chunk equals yr6's fill list minus the 6 WIDE fills, plus 1 SURGE_LONG on STRK (WIDE held STRK in A), with the ZK LITE 10-10 entry 6 s earlier (as in B). That includes W 10-11 23:00 LITE −0.5646 closing at 10-12 11:01, identical to A and B.
5. **Data parity.**
   - **10-11 chunk.** C read a new shadow cache (`scratchpad/cache_yr6w_1011`). It is the yr6x shadow plus OGNUSDT / CAPUSDT without their `2026-10-12.npz`, which another agent wrote on 10-13 12:26 local, after the yr6x shadow was built. 0 tick files dated 10-12 are visible in it. The k1m_spike files touched on 10-12 07:18 local predate yr6's 10-11 run (07:20).
   - **10-01 chunk.** C has no position open across 10-11 00:00.
   - **2025 cache.** Unchanged.
   - **Clean runs.** 45 runs (25 yr6aw + 19 yr6w + the 10-11 shadow run), all rc = 0, 0 tracebacks. ondemand 1m fetches 0 (291 blocked misses; B 286).

## 7. Limits (state with every number)

- **Seed 1 only.** One path.
- **Entries are from config 01419d4.** Today's +1.5 exits are re-priced exactly per fill, but re-entries after an earlier +1.5 exit are not modelled.
- **SURGE gap gate (HEAD 2338871)** is applied only to only-in SURGE fills (shared ones cancel).
- **WILLY** uses the "mcap always readable" proxy.
- **2025 data:** non-top-50 pair-days run on 1m bars, which is the noise source in §5.
- **Small N:** the SURGE part is N = 5 (3 gap-gated today) and the WIDE-held LITE part is N = 2. The headline rests mostly on WIDE's own 443 trades, which are negative in every regime.

## 8. Files and commands

- **Launch / resume:** `scripts/yr6w_launch.sh`.
  - Config: `reports/backtest_cache/replay/frozen_config_yr6w_wideoff.json`, sha256[:12] **d11dec3bb4e1**. It is a 1-line diff vs yr6: `frenzy_wide_enabled` true → false.
  - Outputs: `replay/yr6w_*_s1_*`, `replay/yr6aw_*_s1_*`.
  - Logs: `replay/year/yr6w_orchestrator.log` (marker YR6W_ALL_DONE), `replay/year/yr6aw_orchestrator.log`.
- **Tables** (from the snapshot dir): `yr6_tables.py --prefix yr6aw,yr6w --cmp "" --out-tag YR6W_2Y` → `reports/ENGINE_REPLAY_YR6W_2Y_*`.
- **Re-price** (scratchpad `code_443a7b5`): `scripts/yr6_reprice_today.py --fills ENGINE_REPLAY_YR6W_2Y_fills.csv --years 2025,2026 --tag YR6W_2Y_TODAY --no-book`. Arms A and B reuse `ENGINE_REPLAY_YR6LWA_TODAY` / `ENGINE_REPLAY_YR6X_2Y_TODAY`.
- **Study** (run from `code_443a7b5`): `venv/bin/python scripts/study_yr6_wideoff_ab.py`.
  - Outputs: `reports/YR6_WIDEOFF_AB_{tables.txt, sleeves, decomposition, lite, lite_CvsA, lite_BvsC, lite_BvsA, books}.csv`.
