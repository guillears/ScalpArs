# Review: arm FRENZY_WIDE as HOLD_GREEN at lev 0.05 (2026-10-06)

This is a read-only adversarial review. No code, config or test was touched, and nothing was committed. Every number here was re-derived with my own scripts: load, sequencer, books, bootstraps and nulls (`…/scratchpad/rhg/base.py`, `a1.py`–`a6.py`). The only inputs shared with the studies are the two cohort CSVs (`FRENZY_ENGINE_COHORT_2026-10-05.csv`, `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv`) and `gw/strong.pkl`, which holds the strong-sizing flag.

**Proposal under review.**
- WIDE takes a FRENZY_LONG refusal only when all of these hold:
  - the code is `FRENZY_GREEN_BAR` (ATR ≤ 2.5 and a green signal bar);
  - `above_streak > 12` at the signal bar.
- It drops ATR_HIGH and the green "reclaim" bars (streak = 12).
- Leverage goes from 0.2 to 0.05.
- Gates: switch WIDE off if the mean is ≤ 0 at 40 fills. Size up to 0.2 only at ≥ +0.30 with CI low > 0 on ≥ 80 fills and ≥ 8 windows.

## Verdict

**Do not arm this as an edge. Arming it is acceptable only as a damage-limiting replacement for today's as-is WIDE, and only with the four fixes in §7.**

**Evidence-preferred order: WIDE OFF ≈ HOLD_GREEN @ 0.05 (fixed) ≫ keep as-is.**
- **HOLD_GREEN @ 0.05 is in-sample money-neutral against OFF.**
  - Book: $9,784 / −45 % vs $9,741 / −43 %.
  - Additive Δ: +$3, day CI [−$1,044, +$782].
  - The tie is set by one crowd-out event. Its information value mostly duplicates the zero-cost scout line FRENZY_GREEN_CLOCK V2, which prices the same signals with the same lock exit.
- **Keep-as-is (0.2, every code) costs about $8.5k a year against OFF.** It is the one option with no support. Whichever way the operator goes, as-is should not stay live.

---

## 1. Definition and engine parity: PASS, with one caveat (Minor)

| check | result |
|---|---|
| HOLD_GREEN in the studies | `wfq/p3.py`: `hold_green = (code == "FRENZY_GREEN_BAR") & (above_streak > 12)`. GREEN_BAR implies ATR ≤ 2.5 and a green bar, because `frenzy_long_status` judges ATR_HIGH first and then `not bar_red`. **Exactly the proposal.** |
| `above_streak` source | `scripts/frenzy_engine_cohort_trigger.py` re-runs `services.frenzy.frenzy_walk` at each fresh bar, on the same 1,499-bar window as the cohort build. The re-walk is identical on every row (check `chk`, above_share equal). |
| Scout FRENZY_GREEN_CLOCK V2 | `scout_frenzy_exits.py:105` sets `GC_STREAK = 12`, and `v2 = above_streak > 12` on a replayed `FRENZY_GREEN_BAR` (parity required). **The same signal set.** Two differences: V2 counts the **first refusal per (pair, spike)** (124 HG fills = 100 episodes), and it prices the trade as a LONG at 0.32. The %/fill is the same, because the exit is the same live lock. |
| Can the engine compute it live? | **Yes, at no extra cost.** `frenzy_walk` already returns `above_streak` (`services/frenzy.py:110`), and the engine copies the whole `ep` into `flag` (`trading_engine.py:7184`) before `frenzy_wide_ready` / `_frenzy_open(wide=True)` is called. A gate is a one-line read of `flag['above_streak']`. Streak ≥ 12 always holds on a fresh bar, so "= 12" and "> 12" split every signal. |
| Window caveat | Live 5m reads return **1,000 bars**, not 1,499 (DECISION_LOG 224). The streak is anchored at the spike, so it is identical whenever the episode is identical. Only 1 of 124 HG fills is more than 80 h after its spike. **Negligible.** |
| Stamp | **`above_streak` is NOT stamped on orders** (no column in `models.py`). The proposed gates could not be audited from the orders CSV. See Important 4. |

## 2. The HOLD_GREEN WIDE subset (TRADE universe, sequenced with LONG and HG-only WIDE, LOCK2, 1×)

My sequencer reproduces the studies exactly: LONG 204 · +0.359 and WIDE 565 · −0.331 as live; OFF has LONG 205 · +0.387.

| metric | value |
|---|---|
| N / days / episodes / pairs | **124 / 93 / 100 / 79** |
| WR / breakeven WR | 58.9 % / 51.7 % (avg win +2.90, avg loss −3.11) |
| mean / median | **+0.430** / +1.89 (bimodal: stop −3.1 or lock floor ≈ +1.9) |
| 95 % CI, day blocks | **[−0.15, +1.01]**, P(mean ≤ 0) = 0.075 |
| episode / ISO week / pair blocks | [−0.14, +1.01] · [−0.10, +0.99] · [−0.13, +0.98] |
| per month (N, mean) | Jan +0.82 (16) · Feb +1.20 (11) · Mar +1.16 (20) · **Apr −0.61 (18)** · May +0.12 (13) · **Jun −0.88 (9)** · Jul +1.24 (11) · Aug +0.47 (14) · Sep +0.07 (12) → 7 of 9 positive |
| leave one month out | +0.29 … +0.61, all positive |
| halves | Jan–Apr +0.59 [−0.20, +1.42] · May–Sep +0.25 [−0.60, +1.07] · Jan–Jun +0.37 · Jul–Sep +0.57 |
| concentration | top pair SAHARA 21 % of the net · top 3 pairs 60 % · top day 23 % · **top 5 days 96 %** · top episode (STEEM) 20 % |
| drop the top 1 / 3 / 5 / 10 fills | +0.35 / +0.21 / **+0.12** / **−0.10** |
| **ATR inside HG** | **≤ 1.5: +1.96 (23)** · 1.5–2.0: −0.08 (43) · 2.0–2.5: +0.20 (58). **Without the ATR ≤ 1.5 pocket, HG is ≈ +0.08 on 101 fills.** |

**Same-statistic nulls inside the green WIDE pool** (204 sequenced green fills; this is the hold vs reclaim split):

| test | observed | p |
|---|---|---|
| random 123-of-204 subset mean ≥ HG mean | +0.458 | 0.008 |
| hold − reclaim gap, circular shift of the label in time | +1.12 | 0.005 |
| same gap, permutation within day (24 mixed days, 64 fills) | +1.61 | 0.007 |
| same gap, day-block bootstrap | +1.12 | CI [+0.18, +1.99] |
| **selection-adjusted over the ~1,900-cell search (overnight review)** | rank 13 of 1,892 | **0.87** |

**How to read this:**
- Inside the green pool, "reclaim versus not" is a real-looking split.
- But the split was found post hoc on this same year, the Oct-5 trigger read, inside a ~1,900-cell search. Corrected for that search it is at chance.

**Replication in independent strata (hold − reclaim):**

| stratum | hold − reclaim |
|---|---|
| ATR_HIGH, green | **−0.04** (76 / 102) |
| FRENZY_LONG, red | **−0.08** (101 / 104) |
| ATR_HIGH, red | +0.60 (78 / 105) |

- **The loser side replicates:** a green reclaim bar loses in both ATR strata (−0.66 and −0.59).
- **The winner side does not:** a green hold bar is +0.46 at ATR ≤ 2.5 but −0.63 above 2.5.

**Streak dose-response** (green fills):

| streak | mean (N) |
|---|---|
| 12 | −0.66 (81) |
| 13–15 | +1.02 (12) |
| 16–20 | −2.28 (6) |
| 21–36 | +1.15 (24) |
| 37–80 | −0.26 (33) |
| > 80 | +0.81 (48) |

Every threshold from 12 to 60 gives +0.37 to +0.58. The information is binary (reclaim or not), not a dose. That fits the mechanism, but there is no gradient to lean on.

## 3. Books (TRADE, $3,000 start, shared equity, live sequencing)

Sizing for these books:
- **Sequencing:** LONG regains every pair and slot WIDE no longer takes.
- **LONG:** 0.32, or 0.5 when strong.
- **Leverage:** `max(1, round(20 × mult))`, as `calculate_position_size`.
- **Notional per fill:** calibrated as in the reviewed `watr/core.py` (WIDE at 4× = 0.94 of equity).

| variant | year end | max DD | Jan–Apr | May–Sep | Jan–Jun | Jul–Sep |
|---|---|---|---|---|---|---|
| **OFF** | **$9,741** | **−43 %** | $7,081 | **$4,127** | **$10,128** | $2,885 |
| as-is 0.2 | $1,188 | −94 % | $3,723 | $957 | $1,689 | $2,110 |
| as-is 0.05 | $5,511 | −66 % | $6,160 | $2,684 | $6,107 | $2,707 |
| green-only 0.05 | $8,667 | −50 % | $7,384 | $3,522 | $8,960 | $2,902 |
| **HG 0.05** | **$9,784** | **−45 %** | $7,732 | $3,796 | $9,691 | $3,029 |
| HG 0.1 | $10,965 | −48 % | $8,408 | $3,912 | $10,369 | $3,172 |
| HG 0.2 | $13,474 | −54 % | $9,819 | $4,117 | $11,690 | $3,458 |

All figures reproduce the studies exactly.

- **HG @ 0.05 minus OFF:**
  - Additive at fixed $3k equity: **+$3**, day-block CI [−$1,044, +$782], P(≤ 0) = 0.46.
  - OFF wins May–Sep and Jan–Jun. HG wins Jan–Apr and Jul–Sep.
  - By month: −$428 in June (the crowd-out), +$92 to +$163 in Jan–Mar.
- **HG @ 0.2 minus OFF:** +$1,130, CI [−$1,072, +$3,321], P(≤ 0) = 0.16, with 11 more points of DD. Applying the 30–50 % haircut to a post-hoc pocket leaves about +$0.6–0.8k, inside the noise.
- **Crowd-out (the STG lesson), checked directly:**
  - Only **one** LONG signal in the whole year falls inside a WIDE hold on the same pair: STG 06-11. WIDE −3.11 blocked a strong LONG +6.16.
  - It is the same event for HG, green-only and as-is. HG holds are short: median 16 min, p90 64 min.
  - At 0.05 that single event (≈ −$370 at $3k) cancels HG's whole year (124 × 0.235 × 0.43 % × $3k ≈ +$376).
  - So the crowd-out is rare (≈ 1 a year), but at 0.05 one event is the same size as the sleeve's whole expected yearly gain.
- **Not modelled (Important 3):**
  - **Margin lock-up.** Sizing is `equal_split`, and a lev-only cut keeps the margin. Live WIDE fills on Oct 3–6 locked $706–$830 of margin each.
  - **The global position count:** 4, or 10 with redeploy.
  - **Pair crowd-out of the momentum, FLIP and other sleeves.**
  - All of these only cost.

## 4. Is dropping ATR_HIGH still justified inside streak > 12?

Yes, but on weaker evidence than the whole-WIDE read.

| hold fills (streak > 12) | N / days | WR | mean | day CI | Jan–Apr / May–Sep |
|---|---|---|---|---|---|
| HG (green, ATR ≤ 2.5) | 123 / 93 | 59 % | +0.46 | [−0.13, +1.04] | +0.59 / +0.31 |
| ATR_HIGH, all | 154 / 108 | 47 % | −0.36 | [−0.85, +0.15] | −0.11 / −0.59 |
| ATR_HIGH, green | 76 / 61 | 41 % | −0.63 | [−1.30, +0.10] | −0.29 / −1.01 |
| ATR_HIGH, red | 78 / 64 | 53 % | −0.10 | [−0.76, +0.57] | +0.09 / −0.25 |

- The gap between HG and hold ATR_HIGH is **+0.82, day CI [+0.04, +1.60]**.
- Hold ATR_HIGH by ATR band:

  | ATR band | mean (N) |
  |---|---|
  | 2.5–3 | −0.42 (65) |
  | 3–3.5 | +0.07 (40) |
  | 3.5–4.5 | −0.17 (27) |
  | > 4.5 | −1.19 (22) |

  The 3–3.5 flat band matches the review's non-monotone dose-response.
- **Hold ATR_HIGH on its own would not pass the expectancy bar** (CI spans 0).
- **The red hold ATR_HIGH part is about flat (−0.10).** Removing it costs nothing, but there is no evidence that it hurts.
- **The removal is justified by the whole-WIDE ATR_HIGH result:** 361 fills, −0.53, CI [−0.83, −0.21]. That result already passed its review as damage limitation, and it points the same way here.

## 5. Risk at 0.05

**Per fill.** Leverage becomes `max(1, round(20 × 0.05)) = 1×`. The margin stays at about $700–830, because `frenzy_wide_invest_mult = 1.0` and sizing is `equal_split`.
- **Worst fill in the year: −3.13 %.** Every HG loss is a lock stop. Dislocated entries are refused by the pricer and guard.
- **In dollars: ≈ −$25 a stop, ≈ 0.9 % of a $2.65k book.** At 0.2 it is ≈ −$100 (3.8 %).

**Per day.**
- The worst days were 2 stops: −6.2 %, three times. That is **≈ −$50, 1.9 % of the book.**
- The most fills in one day was 5.
- The longest losing run was 6, ≈ −$150 (5.7 %). At 0.2 the same run would be −$600 (23 %).

**Tail.**
- A gap through the stop at 1× costs only notional × gap: a −10 % print is ≈ −$80.
- Liquidation is not a risk at 1×.

**The real risk is not the stop. It is the unmodelled costs** (the margin, the slot, and the one-in-a-year LONG crowd-out, which is worth about −$490 at today's strong LONG size).

## 6. RLC 10-05

- **RLC was a HOLD_GREEN signal:**
  - 11:55 bar, streak 17, ATR 2.3775 (the live stamp equals the recompute), green +0.52 %, gvol 0.684, code GREEN_BAR.
  - The scout's engine replay (`GC_dry2.csv`) gives the same: `above_streak` 17, `v2` True.
- **The other live WIDE fills on Oct 3–6:**

  | fill | streak / code | under HG | result |
  |---|---|---|---|
  | MOVR | 35 | kept | +3.00 |
  | FLUID | 12 (a reclaim) | dropped | −3.04 |
  | AIN ×2 | ATR_HIGH | dropped | −3.01, −3.02 |

  Under HG: 2 wins kept, 3 losses dropped.
- **This is an anecdote.** It is 3 days, RLC is the case that motivated V2, and it carries no statistical weight.
- At 0.05 the RLC lock gain (+5.3 %) is worth about **$40**. The ride (+130 %) is not captured by any exit, so the "don't miss RLC" motive buys almost nothing at this size.

## 7. Findings

### Critical
1. **HG has no established edge, so the arm must not be described as evidence-backed.**
   - Mean +0.43, CI [−0.15, +1.01].
   - Selection-adjusted p 0.87.
   - Top 5 days = 96 % of the net. Without the top 10 fills it is −0.10.
   - Without the ATR ≤ 1.5 pocket (23 fills) it is ≈ +0.08.
   - The winner side does not replicate above ATR 2.5 or on red LONG bars.

   Under CLAUDE.md this is a **discipline-override ship**: it must be acknowledged as such and carry a tighter revert gate than standard (Important 2).
2. **Against WIDE OFF, HG @ 0.05 is a coin flip in money:** +$3 additive, CI ±$900, OFF ahead in 2 of 4 half-periods, and that is before the unmodelled costs.
   - Its forward information is already being collected free by FRENZY_GREEN_CLOCK V2 (same signals, same lock).
   - Real fills add only live-execution fidelity.
   - **OFF stays the evidence-preferred option. HG @ 0.05 is defensible only as the replacement for as-is.**

### Important
3. **The 0.05 cut via leverage keeps the full margin.** About $800 a fill (~30 % of the book) is locked for about 1/20th of a normal slot's P&L. Two WIDE slots can tie up about 60 % of margin under `equal_split` and shrink or skip the momentum/FLIP engines. None of the books model this.
   - **Fix:** get the same notional with `frenzy_wide_invest_mult 0.25` at `lev_mult 0.2`. That is 4× leverage on ¼ the margin; the leverage floor 0.05 and invest floor 0.1 both allow it. The stop at −3 % is far from liquidation at 4×.
4. **No `above_streak` stamp exists on orders.** D11/D12 needs:
   - an `entry_frenzy_above_streak` model column;
   - a config field such as `frenzy_wide_hold_only` (or `frenzy_wide_min_above_streak`), with a UI toggle and load/save handlers;
   - `_record_filter_block("FRENZY_WIDE_RECLAIM")` and `("FRENZY_WIDE_ATR_HIGH")` counters.

   Without the stamp the gates cannot be audited from the CSV.
5. **The gates are weak and slow.** Power by day-block bootstrap from the HG fills, SD 3.25, 0.48 fills a day:

   | true mean | P(off-gate fires at 40) | P(size-up passes at 80) |
   |---|---|---|
   | +0.43 | 21 % | 24 % |
   | +0.22 (haircut) | 35 % | 11 % |
   | 0 | 51 % | 3 % |
   | −0.33 | 73 % | 0 % |

   - At 40 fills, a true-zero sleeve survives the off-gate half the time.
   - 40 fills take about **84 days**; 80 take about **168 days**.
   - "≥ 8 windows" is meaningless at 80 fills, which span about 60+ days.

   **Tighten** (discipline-override rule):
   - **Off** if the mean ≤ 0 at 40 fills, **or** the mean ≤ −1.0 at 20 fills (live-bleed stop).
   - **Off** if the mean without the top 5 fills is ≤ 0 at 60.
   - **Size-up** to 0.2 only if all hold: ≥ 80 fills on ≥ 40 distinct days; mean ≥ +0.30; day CI low > 0; mean without the top 5 > 0; no pair > 25 % and no day > 25 % of the net; mean after a 50 % haircut ≥ +0.15.
   - Count only post-deploy fills. Judge on actual fill P&L, and show LOCK2 alongside.
6. **Do not double-count the evidence.** WIDE-HG fills and the scout's V2 line are the same signals. They must not later be cited as two confirmations. If V2 is promoted to FRENZY_LONG (its own frozen bar), WIDE-HG becomes redundant and should be retired.

### Minor
7. HOLD_GREEN is 124 fills in the HG book vs 123 in the as-is book; the extra one is a fill that other WIDE fills had crowded out. Both are quoted; the numbers are consistent.
8. The green reclaim block (the other half of the change) has: N 81, 62 days, WR 41 % vs breakeven 51.7 %, mean −0.66, day CI [−1.30, +0.04], 8 of 9 months negative, both halves negative, worst pair 17 %.
   - It **fails only the 95 % leg (marginally)**, and it replicates in the ATR > 2.5 green stratum (−0.59).
   - It fits DECISION_LOG 180's "green reclaim = chase" mechanism. As a WIDE block it is better supported than the HG winner side.
9. The 1,000-bar live window vs the 1,499-bar cohort window affects 1 of 124 HG fills (spike > 80 h old).
10. There is no seed dimension (deterministic replay), so the per-seed check is not applicable.
11. Not tested:
    - other sleeves' slot and margin interaction;
    - paper-vs-pricer slippage (paper runs about +0.10/fill better);
    - order-book stamps;
    - Oct live fills in any statistic.

## 8. Recommendation

1. **Preferred: `frenzy_wide_enabled = false`.** The scout's V2 and WIDE_BY_CODE lines keep the forward read at zero cost.
2. **Acceptable if the operator wants real WIDE fills:** HG-only, labelled as a discipline-override probe, with:
   - (a) the same notional reached via `frenzy_wide_invest_mult 0.25` at `lev 0.2`, rather than lev 0.05 at full margin;
   - (b) the `above_streak` stamp, config field, UI, handlers and counters (D11/D12);
   - (c) the tightened gates in Important 5;
   - (d) no size-up before ~6 months of fills.
3. **Not supported:** keeping as-is at any size, or HG at 0.2 now. The +$3.7k book is a post-hoc pocket that loses its significance after the haircut, and it carries +11 points of DD.
