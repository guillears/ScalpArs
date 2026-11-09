# 🧪 FAILED_SPIKE theory test on the other Frenzy sleeves — result of the DL 511 pre-registration (2026-11-09)

Pre-registered before any result was read (DECISION_LOG 511). Frozen cut: zone = `entry_frenzy_vs_vwap_pct ≤ 0` (signal bar closed at or below the
spike-anchored VWAP). One test per sleeve, never pooled. Predictions: WILLY_HT / LITE / LITE_RETRY / LONG → zone WORSE; FLATBASE_SHORT → zone BETTER.
Rule: CONFIRMED only if zone − rest has the predicted sign on master (today's stack) AND on B25; wrong sign or flip → REFUTED; < 5 zone or rest fills
on either side → TOO FEW (no call). Nothing was re-cut or searched.

## Result table (avg % per fill · Σ % · daily compound at ONE size, %/day)

| Sleeve | Block | Zone N · WR · avg · Σ | Rest N · WR · avg · Σ | DCR before → block zone | Verdict |
|---|---|---|---|---|---|
| WILLY_HT | master | 0 | 0 | no master trades | **TOO FEW** |
| WILLY_HT | B25 | 29 · 62 % · −0.31 · −9.09 (13 days) | 6 · 83 % · +3.00 · +17.99 (4 days) | +0.144 → +0.302 | (sign as predicted, rest N 6) |
| LITE | master | 0 | 30 · 53 % · +0.21 · +6.29 | −1.023 → −1.023 | **TOO FEW** (zone empty by design) |
| LITE | B25 | 0 | 31 · 61 % · +0.64 · +19.75 | +1.910 → +1.910 | |
| LITE_RETRY | master | 0 | 2 · 100 % · +3.00 · +6.00 | +1.147 → +1.147 | **TOO FEW** (zone empty by design) |
| LITE_RETRY | B25 | 0 | 37 · 51 % · +0.08 · +2.91 | +0.099 → +0.099 | |
| FRENZY_LONG | master | 0 | 29 · 86 % · +0.74 · +21.47 | +0.061 → +0.061 | **TOO FEW** (zone empty by design) |
| FRENZY_LONG | B25 | 0 | 17 · 47 % · −1.41 · −24.03 | −0.065 → −0.065 | |
| FLATBASE_SHORT | master | 1 · 100 % · +7.95 | 0 | +1.943 → +0.000 (keep-zone-only +1.943) | **TOO FEW** (rest empty by design) |
| FLATBASE_SHORT | B25 | 25 · 64 % · −0.11 · −2.87 (14 days) | 0 | −0.064 → +0.000 (keep-zone-only −0.064) | |

**All five sleeves: TOO FEW → no call, no line (per the pre-registration).**

## Why four of the five can never be tested on this cut

The zone is empty (or the whole sleeve) by the entry rule itself, not by chance:
- **LITE, LITE_RETRY, FRENZY_LONG** open only after a full hour of closes at or above the spike VWAP — FRENZY_LONG via `services/frenzy.py`
  `frenzy_long_status` (block `FRENZY_BELOW_AVG`), LITE / RETRY via `frenzy_lite_status` (12 closes in a row, `frenzy_lite_min_above_closes`,
  block `FRENZY_LITE_NOT_ABOVE`). Observed stamp ranges: LITE +0.06 … +9.80, RETRY +0.16 … +27.59, LONG +0.37 … +41.00 — never ≤ 0, master or B25.
- **FLATBASE_SHORT** opens only when the 4-h close is below the spike VWAP (`services/flatbase.py`, `c[w] < vw`). Every fill is in the zone
  (B25 −13.24 … −1.12). "Short only below the VWAP" is already the live rule; the mirror line adds nothing.
- The theory still holds as a description of *why* those entry rules exist, but this data cannot confirm or refute it for these sleeves.

**WILLY_HT** is the only sleeve where the cut splits fills (stamp −37.35 … +14.92). B25 points the predicted way (zone −0.31 vs rest +3.00 %/fill,
Δ −3.31 pts; leave-one-out range −4.71 … −2.31, never flips; blocking the zone +0.144 → +0.302 %/day; LOO on the Σ blocked: +9.09 pts, without the
biggest (UAIUSDT 11-09 10:20) +4.96). But master has no WILLY_HT fills at all and the B25 rest is only 6 fills on 4 days, so the confirmation rule
cannot be met. Zone by day (B25, n · Σ %): 10-23 1·+3.15 · 10-24 1·+0.26 · 10-25 2·−0.24 · 10-26 4·+3.56 · 10-27 1·+4.35 · 10-29 1·−4.00 ·
10-30 2·+1.51 · 11-01 1·+0.29 · 11-05 5·−0.01 · 11-06 4·+0.27 · 11-07 2·−2.37 · 11-08 4·−11.74 · 11-09 1·−4.13 — the zone's loss sits on 11-07 → 11-09.
Note the zone is 29 of 35 HT fills: blocking it would leave ~1 trade every 3 days, so even a forward confirmation would be a near-shutdown, not a filter.

## Data checks

- Before DCR matches the best case ① (reports/best_case/BEST_CASE_2026-11-09_12-45.md) on the best_dcr fill set for every sleeve: WILLY_HT +0.144
  (+0.14), LITE +1.910 / −1.060 (+1.91 / −1.06), RETRY +0.099 / +1.152 (+0.10 / +1.15), LONG −0.065 / +0.061 (−0.07 / +0.06), Flatbase −0.064 /
  +1.943 (−0.06 / +1.94). Exits re-priced exactly as best_dcr: B25 LITE C +2 → +3 (8 LITE, 6 RETRY), 13 Flatbase pre-DL 408 fills → +5 / −10,
  9 master FRENZY_LONG −3 stops → −4. One-size X: HT 0.306 · LITE 1.942 · RETRY 0.978 · LONG 0.049 · Flatbase 0.244.
- Operator-closed fills excluded (scout_gatekit.split_operator_closed): master LITE ICNTUSDT 10-20 22:10 EXTERNAL_CLOSE, master RETRY AUSDT
  10-20 14:50 MANUAL. This moves master LITE −1.060 → −1.023 and RETRY +1.152 → +1.147 (the table uses the excluded set). B25: none.
- `cell_multiplier_source` *_PROBE rows in these sleeves' master fills: none.
- Stamp coverage: 100 % on every sleeve, both blocks (HT 35/35 · LITE 30/30 + 31/31 · RETRY 2/2 + 37/37 · LONG 29/29 + 17/17 · Flatbase 1/1 + 25/25).
  No unscored fills.

## What this means

No new watch line from DL 511. The FAILED_SPIKE idea already lives inside the entry rules of LITE / RETRY / LONG (only above the VWAP) and
Flatbase (only below it). WILLY_HT is the one open question; a forward read on HT would need master data that does not exist yet, and its zone is
most of the sleeve. WILLY's own line (DL 509) is unaffected.

Script (scratch, read-only): /private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/frenzy_fs/fs_test.py
