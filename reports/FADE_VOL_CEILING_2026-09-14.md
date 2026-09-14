# SPIKE_FADE 24h-volume ceiling — ship record and the investigation behind it (2026-09-14)

**Ship:** `spike_fade_max_vol_24h_usd = 20,000,000` (block a SPIKE_FADE when the pair's 24h USD volume at fire ≥ $20M). Operator DISCIPLINE-OVERRIDE, acknowledged — evidence is below every locked block gate (see §4). Counter `SPIKE_FADE_MAXVOL`; every block logs a reopen-read row. Pool stack `2026-09-14a` encodes it as `FADE_MAXVOL`.

**Trigger:** VTHOUSDT (−$188, EMA13 tick-1 bug — fixed same day, DECISION_LOG 54) and 龙虾USDT (−$381, −1.5 stop in 8 s, then a −5.8% collapse) on Sep-14 — the two largest-volume fades ever taken.

## 1. Per-batch results — before vs after the ceiling (screened by today's fade filters; master pool + B6 export)

| Batch | Before N · WR · net | After N · WR · net | Removed |
|---|---|---|---|
| B1 | 6 · 83% · +$261 | 6 · 83% · +$261 | — |
| B2 | 26 · 69% · +$775 | 25 · 72% · +$776 | DEXE −$1 (EMA13 bug) |
| B3 | 18 · 83% · +$296 | 16 · 81% · +$174 | HEMI +$61, TAC +$62 (both WON) |
| B4 | 1 · 0% · −$111 | 0 | ZRO −$111 |
| B5 | 0 fades | 0 | — |
| B6 (current, to Sep-14 12:53) | 3 · 33% · −$506 | 1 · 100% · +$63 | VTHO −$188 (bug), 龙虾 −$381 |
| **TOTAL** | **54 · 72% · +$715** | **48 · 77% · +$1,273** | **6 · 2W/4L · −$558** (ex-bug 4 · 2W/2L · −$369) |

Raw (unscreened, all 88 fades ever): 88 · 62% · −$38 → the same 6 removed. Window units: 3 (Aug-23 +$122, Aug-25 −$111, Sep-14 −$569). Take Sep-14 out and the ceiling removes 4 fills · 2W/2L · +$11.

## 2. What the removed cohort looks like vs the kept one

| | Micro-cap (<$20M) | Large-cap (≥$20M) |
|---|---|---|
| N · WR · net | 48 · 77% · +$1,273 | 6 · 33% · −$558 |
| Entry ATR (median) | 0.43–0.90% | 1.1–1.5% |
| Stop (−1.5) in ATR units | 1.7–3.5 | 1.0–1.4 |
| Distance above EMA13 at entry | 1.3% | 4.3–4.9% |
| Time to exit (median) | 1.8–3.3 min | 8–12 s |
| Squeeze in the 45 min after entry | +0.25–1.0% | +3.0% |
| Collapse in the 45 min after entry | −1.1–1.8% | −4.1% |

Large caps resolve inside 30 s win or lose, at 3–4× the excursion amplitude. Five of six collapsed 1.4–7.8% after entry (the fade thesis was right); the losers met the squeeze first.

## 3. Volume × ATR cross (the sharpest view)

| Cell | N | W/L | Net | Stop in ATR | Dist>EMA13 | Time to exit | Squeeze | Collapse |
|---|---|---|---|---|---|---|---|---|
| ATR<1.0 × vol<$20M | 43 | 32/11 | +$576 | 3.2 | 1.3% | 222 s | +0.25% | −1.1% |
| ATR≥1.0 × vol<$20M | 5 | 5/0 | +$697 | 1.1 | 6.0% | 48 s | +2.5% | −4.4% |
| ATR≥1.0 × vol≥$20M | 5 | 2/3 | −$557 | 1.0 | 4.9% | 12 s | +3.9% | −5.8% |
| ATR<1.0 × vol≥$20M | 1 | 0/1 | −$1 | 2.7 | 1.5% | 1 s (bug) | | |

Every geometry variable is set by ATR, not volume. The two high-ATR cells are twins in geometry and opposite in outcome (5/5 vs 2/5; exact P ≈ 0.08). What decided them is sequencing — collapse-before-squeeze wins (VANRY: 7.2% squeeze in-window, still +3.92%), squeeze-first loses (ZRO: 13% squeeze first) — not stamped at entry.

## 4. Every lead tested and refuted on the way (so none is re-hunted)

| Lead | Test | Result |
|---|---|---|
| Hour of day | 6 buckets | 04–15 UTC −$340 → ex-large +$152 · refuted (it was the large caps) |
| BTC stamped state (5m RSI/ADX/regime, 1h RSI/slope, trend gap, ATR) | sign + buckets | none; 1h-RSI 50–60 / gap>0 cells → ex-large +$147 · refuted |
| BTC macro rebuilt from candles: prior 15m/1h/4h/24h/72h/7d returns, forward 15/30/60m, dist EMA20/50 1h, off-24h-high, realized vol, bars-above-EMA20, EMA20 slope | 17 features × 3 cohorts, sign + terciles, shuffle p | 51 tests, one at p=0.047 (raw 72h return) vanishing under today's filters; forward-BTC correlation with fade P&L ≈ 0 (no beta squeeze); zero fades ever entered in bull-run conditions; Aug-17 week (BTC +23.5%) fades 7/7 |
| Exhaustive separator sweep (`scripts/sweep_fades.py`): 48 dims (all stamped entry_* + 13 BTC features) × 3 granularities, cross-era; all 2D quadrants; 150–200 label shuffles | screened & raw | 1D survivors 32 vs chance median 42 (P=0.88); 2D 462 vs 584 (P=0.79); top strengths below the chance 95th pct. **No separator, 1D or 2D.** 3D not run (≈7 fills/cell) |
| ATR ceiling on fade entry (1.0) | bucket + sensitivity | blocks 10 fades · 70% · +$140 (first 7 high-ATR fades all won: PROM/TLM/VANRY/EVAA/SQD/TAC/HEMI) · **refuted** |
| Wider flat stop (−2.0/−2.5/−3.0) | post-exit paths of the 4 −1.5 stops | 1 near-miss saved, 2 squeezes no width holds · kept −1.5 |
| ATR-scaled fade exits (`scripts/fade_exit_resim.py`, real ticks first 10 min + 1m bars, entry-aligned, adverse-first; live replica 91% sign agreement) | 9 variants × 54 fades | ATR-scaled STOPS −$419…−$512 (large-cap squeezes exceed a 3% cap); take-side variants +$159…+$237 on micro-caps but −$215…−$343 on large caps; **no variant beats the live stack on the whole sleeve** |
| Whole-sleeve de-mux 2×→1× | per-batch actual vs 1× | halves a proven micro-cap edge (+$1,273 → +$637) to save $279 · rejected; large-cap-only 1× staging was the quant recommendation |
| Fade WR history per batch | | B1 83 / B2 69 / B3 83 / B4 0 (1 fill) / B6 33 (3 fills) screened; micro-cap-only 83 / 72 / 81 / — / 100 |

## 5. Governance

Below the locked gates on every axis: N=6 (gate 30), one batch positive of three, no population-level signal, half the losses a fixed bug. Shipped on the operator's explicit call as a discipline override. 🔒 **REOPEN (pre-committed, does not move):** 30 days after ship (→ 2026-10-14) the ceiling goes to 0 and large-cap fades return as a 0.5× probe via a NEW large-cap-only sizing field (the sleeve-wide `spike_fade_invest_mult` must not be halved) until N≥10 clean fills adjudicate at the standard gates; OR sooner if ≥3 of the next 10 micro-cap fades (`entry_pair_volume_24h_usd` < 20M) show the large-cap failure shape: `closed_at−opened_at` ≤ 45 s AND `(entry_price − post_exit_running_low)/entry_price` ≥ 1.5% (the post-exit window is 45 min from close), which would mean the ceiling is blocking the wrong thing. Blocked large caps exist only as `[SPIKE_FADE_MAXVOL]` log rows, so the 30-day counterfactual is a log replay. Quant recommendation on record was 1× staging, not a block.

Sources: `reports/MASTER_POOL_stacked.csv` (stack 2026-09-14a), `~/Downloads/scalpars_orders_paper_2026-09-14_12-53-05.csv` (B6, to be archived as BASELINE6 at reset), tick/bar paths and `resim_results.csv` in the session scratchpad (regenerable via the two scripts).
