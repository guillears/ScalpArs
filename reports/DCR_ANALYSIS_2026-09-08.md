# Daily Compound Return — per-batch analysis under the CURRENT stack (2026-09-08)
**Question (operator):** what is the expected daily compound return? Answered iteratively — four methodology errors caught by the operator and fixed; final version below reconciles with every pinned ledger figure.

## Methodology (final, after corrections)
- **Pool:** MASTER_POOL_stacked.csv (563 rows, eras BASE/B1/B2/B3/B4) + BASELINE5 CSV (9), CLOSED only.
- **🔒 Full-size only** (locked rule): is_probe/is_door rows EXCLUDED (B1 was 197 probes vs 38 real trades — probe fires are deliberately-admitted blocked-zone experiments, the opposite of the current stack).
- **Current-stack screen:** scripts/screen_pool.py `sleeve()` + `pnl_current()` run TODAY (reads live config → today's thresholds/blacklist/de-muxes).
- **Both pricings** (locked rule): 1× anchor AND live-sized (script's exact cell-mult logic: flip TG_SHALLOW/NEGDI15 max-not-stacked; ML/MS de-muxes). VALIDATION: BASE live-sized reproduces the v14 pin **+$6,187 exactly**.
- **Equity denominators DERIVED from data** (base investment × slots + reserve), not guessed: BASE $3,330 · B1 $3,305 · B2 $2,760 · B3 $3,157 · **B4 $1,108** (not $2.7k!) · B5 $2,689 (recorded).
- **Running-equity compounding**, P&L attributed to close date.
- **Active-day rate** (machine production, matches the dashboard DCR concept) vs calendar rate (capital experience — dilutes by operator downtime; B5 ran 2 of its 9 span days).

## Final table — geo % per ACTIVE day, live-sized, current stack
| era | eq0 | kept | WR | Σ$ 1x | Σ$ LIVE | act.days | geo%/act.day | geo%/cal.day |
|---|---|---|---|---|---|---|---|---|
| BASE (master, paper Jun17–Jul10) | 3,330 | 91 | 82% | +5,037 | **+6,187** ✓pin | 21 | **+5.13** | +4.47 |
| B1 (paper Jul11–31) | 3,305 | 26 | 57% | −132 | −132 | 13 | −0.31 | −0.19 |
| B2 (paper Jul31–Aug10) | 2,760 | 4 | 75% | +113 | +113 | 3 | +1.35 | +0.37 |
| B3 (live Aug11–24) | 3,157 | 36 | 72% | +576 | +236 | 11 | +0.66 | +0.52 |
| B4 (live Aug24–26) | 1,108 | 12 | 75% | +53 | +53 | 2 | +2.34 | +1.56 |
| B5 (live CLEAN Aug26–Sep3) | 2,689 | 8 | 100% | +287 | +287 | 2 | **+5.21** | +1.13 |
RAW (unscreened full-size) records for contrast: B2 −1.44%/cal.day (worst day −14.6) · B3 −0.83 (−11.9) · B4 −7.76 (−26.6 on the $1.1k book) — the filter stack's retro-delta.

## Findings
1. **Master batch = +5.1%/active day (+4.5%/calendar) live-sized** — operator's ">2%" recollection confirmed, $3,330→$9,517 in 24 days. In-sample era (filters derived on it) + trending tape.
2. **B5 (only bug-free live sample) = +5.2%/active day** — the clean stack live-delivers the master-era rate. Strongest single fact. N=2 active days.
3. **Off-regime active days:** −0.3 to +1.4%/day (B1/B2/B3) — filters hold ~breakeven where the raw book bled −1 to −8%/day.
4. **Multiplier caveat:** live sizing CUT B3 (+576→+236) — NEGDI15 2× amplified that era's flip losers; multiplier cells cut both ways off-regime.
5. **Expectation:** +2–5%/day favorable tapes · ~0–1.5%/day chop · **+1–2%/day through-the-cycle** at current size, decaying with scale (liquidity caps). Regime mix ≈ 1/3 favorable in the Jun–Sep record.
6. **Caveats:** BASE in-sample (×0.6 haircut view exists); retro-screened live eras support direction not precision (old exits — cross-period rule); B3's rehabilitation partly circular (its losers motivated gates).

## ⚠ CORRECTION (operator-caught, same day): the "off-regime" floor is BIASED LOW
Data check: **B1+B2 contain ZERO flip trades** (34-day flip-kill bug, dead code until Aug-11 18:32 — master flips were +$589/31tr, a winning chop-condition sleeve entirely absent and UNREPLAYABLE); **bull-run sleeve didn't exist** (shipped Aug-21; B3's 42 sleeve fills = the v1 biased cohort, excluded); **B2 = 72/77 spike-experiment trades under the wrong-pair trigger bug** (Phase-3 leak) — barely a momentum sample. So B1's screened −$132 is the momentum-core-ALONE floor in chop; the current bot adds flip-short + bullrun v2 in exactly those windows. **The +1–2%/day through-cycle band is a conservative floor with known upward bias; magnitude unquantifiable pre-B6** (dead code leaves no counterfactual — the 'absence is a logs question' class).

## 🔒 Adjudication
**B6 (fixed stack, uninterrupted, N≥50) replaces this estimate with a measurement.** Judge B6 vs: through-cycle band +1–2%/day; favorable-regime windows vs the +2–5 band. Methodology errors log (for future self): guessed equities → derive from data; calendar vs active days; probe exclusion; both pricings; running compounding.
