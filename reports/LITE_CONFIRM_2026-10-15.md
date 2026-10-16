# LITE confirmation entry (buy-stop above the signal): study of 2026-10-15

Pre-registration: `reports/LITE_CONFIRM_PREREG.txt` (frozen 2026-10-15T19:39:23Z, before any cell was priced).
Generated tables: `reports/LITE_CONFIRM_tables.md`. Gates: `reports/LITE_CONFIRM_gates.csv`. Per-fill rows: `reports/LITE_CONFIRM_rows.csv`
and `reports/LITE_CONFIRM_live_rows.csv`. WAY-2 log: `reports/LITE_CONFIRM_verify.txt`.
Research only. No engine, config, template or test file was edited. The bot API was not called, nothing was committed, and no Binance call was made (everything came from the cache).

---

## Plain-English summary

**The question:** if the coin has to prove itself before we buy, do the dead-on-arrival trades go away, and what does that cost?

**What happens:** yes, the dead-on-arrival trades go away. The cost is higher than the saving, though, so LITE makes less money. This holds in every year, at every trigger level and window, and for every way of placing the exits.

- **The good part is real.** A buy-stop +0.75 % above the signal price, valid for 15 minutes, never triggers on **98 %** of the trades that peaked below +0.5 % (the "dead on arrival" ones). It also skips about half of the trades that peaked between +0.5 and +1 %. In 2025 that avoids **226 losers worth +672 points**, and in 2026 it avoids 155 losers worth +466 points.
- **It is paid for three ways:**
  1. **Real winners never trigger.** Many LITE winners dip first and only then run to +3 %. For example, 102 winners in 2025 (−308 points) and 58 in 2026 (−175 points) never rise +0.75 % within 15 minutes.
  2. **The winners we do take are bought higher.** Each one gives up the trigger distance plus the slip, which costs 244 to 308 points per year.
  3. **The losers we take lose more.** We bought higher but the stop is in the same place, which costs another 126 to 189 points.
- **The net result**, with a realistic +0.25 % slip on the triggered buy and today's replay as the comparison, is negative in every one of the 48 confirmation variants. In 2025 it ranges from −0.02 to −0.32 % per signal, and in 2026 from −0.12 to −0.39 %. With zero slip on both sides it comes out about flat to negative (2025 −0.13…+0.05, 2026 −0.29…+0.06). So the trigger does not pick better trades; it only trades less.
- **The one way it can look positive is a slip effect, not a better entry.** If today's live entry also pays +0.25 % slip (live says it does, about +0.27 % on average over 5 fills), then every skipped signal "saves" that slip. Under that assumption a few variants look positive. The best is +0.75 % trigger, 15 minutes, TP +3 % from the new entry and stop at the original −3 %: **+0.10 [P 0.88] in 2025 and +0.12 [P 0.88] in 2026**, which still misses the 0.90 bar. But skipping the **same share of trades at random** would have done **better in 2025 in all 96 such readings**: 0 of 96 show any selection value in 2025 (65 of 96 do in 2026). That gain is "trade less while LITE pays 0.25 % to get in", not "the coin proved itself".
- **The exit-side twin also fails.** It enters as today and sells at minute 3, 5 or 10 if the trade hasn't reached +0.3 or +0.5 %. It cuts 131 to 405 future winners for every 220 to 502 losers it saves. With the −0.20 % market-exit slip it loses in both years (2025 −0.01 to −0.11, 2026 −0.04 to −0.25 per trade). Without slip, its best variant (minute 10, +0.5 %) is +0.07 [0.89] in 2025 and +0.01 [0.55] in 2026.
- **The operator's instinct is half right.** 25–35 % of LITE entries are dead on arrival, and a confirmation trigger finds almost all of them. The problem is LITE's own shape: a +3 / −3 trade whose winners often start by going down. Any rule that waits for proof either throws away those winners or pays for the proof on every trade it takes. That matches LITE_EPIC_ENTRY: what separates the losers only becomes visible after the entry, and seeing it costs about the same as it saves.
- **Recommendation: none of the 54 pre-registered cells is a candidate.** Not under the primary ruler, and not under any of the 4 other rulers. Nothing to arm or scout. The "symmetric" near-miss (c +0.75 / W 15 / anchoring C) is not a Scout line either: it fails the random-skip null in 2025 and its W 30 neighbour is ≤ 0 in 2025.
- **A cheaper lever the data points to:** the live +0.27 % entry cost is the whole reason the symmetric rows look good. Cutting entry slip directly (TICK_STUDY_MAKER_LOCK Q2-TL already showed a passive-limit entry wins under that ruler for WILLY) is the honest version of this idea. It is not re-tested here.

---

## Method (short)

- **Cohort.** This is the frozen TICK_STUDY_2026-10-15 LITE cohort as priced in `LITE_PROTECT_rows.csv`:
  - BT (backtest, yr6 2Y_TODAY seed 1): 2025 has 828 fills (264 top-50 on ticks, 564 on 1m bars), 2026-H1 has 369 and 2026-H2 has 255.
  - Master: B18 2, B19 13, B20 1, B21 4 fills.
  - Paths come from `FX.load_path`, extended to 780 minutes. On that horizon there are 875 tick paths, 564 1m paths and 13 ticks+1m paths.
- **Decision price** = the row's E (the replay fill's decision price; the recorded entry for master). Paths hold prints **strictly after** the decision time (`te`), so nothing before the decision is used.
- **Buy-stop.**
  - Entry is the first print ≥ E·(1 + c/100) within W minutes. On a 1m bar the entry is max(trigger, bar open), and that is flagged: 38 % of triggers fall on 1m bars, almost all in 2025 outside the top-50.
  - The entry-slip ruler is s = +0.10 or +0.25 %.
  - The exit walk starts at the event **after** the entry print, and the 12 h cap runs from the entry.
- **Exit anchorings:**
  - A = TP +3 % and stop −3 %, both from the new entry.
  - B = TP stays at today's resting limit (decision +3.06 % gross) and the stop stays at decision −3 %.
  - C = TP +3 % from the new entry, stop at decision −3 %.
- **TP ruler.** The primary ruler RM is today's resting maker limit (DL 296); S0 (taker at the line) is the sensitivity check. Stops book the print through the line.
- **Two headline rulers:**
  - **CONSERVATIVE** (the pre-registered primary): the variant pays s and today pays 0, as in the replay.
  - **SYMMETRIC**: today pays the same s on its entry.
- **Follow-through twin (FT).** At minute k, if today's exit has not happened and the peak so far (net % vs E) is below +p, sell at the last print ≤ te + k minutes, with exit slip 0 or −0.20.
- **Statistics.** Every Δ is per signal (not triggered = 0) against today on the same fill. P is the day-clustered bootstrap P(Δ > 0) with 4,000 resamples, seed 7. Gates G1–G5 are as pre-registered.

### Parity and verification
- **c = 0 / W = ∞** (entry at the decision, no slip) reproduces today on **every** fill (1,472 fills, RM and S0, |Δ| < 1e-9 against the frozen `LITE_PROTECT_rows.csv` today). This is asserted inside the walk.
- **WAY 2** (an independent pure event loop, no shared walker code) covered 160 fills: a random 150 plus targeted 1m-bar triggers and CAP exits. That is 34,560 cell × ruler × slip checks, every CF cell (RM/S0 × s 0/+0.25) and every FT cell, with **0 mismatches**.

---

## Block ① — LIVE (replica paths)

| fill | booked live | today (replica, RM) | minutes to +0.25 / +0.5 / +0.75 / +1.0 | +0.5 W15 A / B / C | +0.75 W15 A / B / C | FT k5 p0.5 |
|---|---|---|---|---|---|---|
| DEEP 10-14 21:40 (ticks) | +2.80 | +3.01 | 1.3 / 1.6 / 4.6 / 4.9 | +3.03 / +2.21 / +3.03 | **−3.02** / +1.96 / unres. | +3.01 (no fire) |
| EPIC 10-15 14:55 (1m) | −3.43 | −3.00 | 1.9 / 2.9 / 3.9 / 5.9 | −3.00 / −3.72 / −3.72 | −3.00 / −3.96 / −3.96 | −3.00 (no fire) |
| DUSK 10-15 15:05 (1m) | −2.54 | unresolved on the 1m replica | 1.9 / 1.9 / 5.9 / 5.9 | −3.00 / – / – | −3.00 / – / – | – |

All three live fills triggered every buy-stop within 6 minutes, so confirmation would not have kept the bot out of EPIC or DUSK. EPIC was never a dead-on-arrival trade: it peaked at +2.6. On DEEP, the higher entry with the stop moved up (anchoring A, c ≥ +0.75) turns a +3 winner into a −3 stop.
**ZK** (live LITE) has no export row and no path in the cache, so it is not priced. If ZK is still open, its outcome cannot be described from data here.

## Block ② — MASTER (20 fills, B18–B21)

Primary ruler, Δ per signal (triggered / N). The full grid is in tables §5.

| cell | master ALL | B18 | B19 | B20 | B21 |
|---|---|---|---|---|---|
| +0.5 W15 A | +0.30 · 14/20 | −3.02 · 2/2 | +0.70 · 8/13 | +3.00 · 0/1 | +0.01 · 4/4 |
| +0.75 W15 C | +1.01 · 11/20 | −0.00 · 2/2 | +1.16 · 6/13 | +3.00 · 0/1 | +0.51 · 3/4 |
| +1.0 W15 A | +0.15 · 7/20 | −4.52 · 1/2 | +0.47 · 3/13 | +3.00 · 0/1 | +0.76 · 3/4 |
| FT k5 p0.5 (−0.20) | +0.17 | −2.18 | +0.40 | +2.50 | 0.00 |

Master is positive for most cells. Its 25 % dead-on-arrival rate is higher than the backtest's 15 %, and the B20 −3 fill alone is worth +3.00 whenever it is skipped. With N = 20 and one or two fills moving each batch, this is not evidence on its own. G2 only requires master not to be negative, and that is the gate this block passes.

## Block ③ — BACKTEST, the headline grid (W = 15; all 48 cells in tables §2)

Δ % per signal vs today [day-bootstrap P(Δ > 0)], RM TP ruler, entry slip +0.25. "cons" = today pays no slip (pre-registered primary); "sym" = today pays the same +0.25.

| cell | ruler | triggered (BT) | 2025 | 2026-H1 | 2026-H2 | BT tick-only | master |
|---|---|---|---|---|---|---|---|
| +0.25 W15 A | cons | 87 % | −0.28 [0.00] | −0.21 [0.01] | −0.14 [0.12] | −0.22 [0.00] | +0.45 |
| +0.25 W15 A | sym | 87 % | −0.02 [0.39] | +0.07 [0.85] | +0.07 [0.74] | +0.06 [0.87] | +0.45 |
| +0.5 W15 A | cons | 73 % | −0.26 [0.00] | −0.18 [0.07] | −0.24 [0.08] | −0.21 [0.00] | +0.30 |
| +0.5 W15 B | cons | 73 % | −0.19 [0.00] | −0.25 [0.00] | −0.14 [0.12] | −0.19 [0.00] | +0.37 |
| +0.5 W15 C | sym | 73 % | +0.06 [0.81] | +0.12 [0.87] | +0.13 [0.80] | +0.12 [0.95] | +0.79 |
| +0.75 W15 A | cons | 62 % | −0.21 [0.02] | −0.19 [0.07] | −0.44 [0.01] | −0.26 [0.00] | +0.15 |
| +0.75 W15 B | cons | 62 % | −0.16 [0.01] | −0.16 [0.06] | −0.24 [0.04] | −0.17 [0.01] | +0.49 |
| +0.75 W15 C | cons | 62 % | −0.16 [0.03] | −0.07 [0.29] | −0.22 [0.10] | −0.15 [0.04] | +1.01 |
| **+0.75 W15 C** | **sym** (best of 240 CF readings) | 62 % | +0.10 [0.88] | +0.21 [0.96] | −0.01 [0.47] | +0.14 [0.94] | +1.00 |
| +1.0 W15 A | cons | 51 % | −0.12 [0.12] | −0.31 [0.01] | −0.38 [0.02] | −0.29 [0.00] | +0.15 |
| +1.0 W15 C | sym | 51 % | +0.16 [0.97] | +0.15 [0.90] | −0.09 [0.31] | +0.11 [0.90] | +0.39 |

Today, per signal (RM): 2025 −0.165 · 2026-H1 +0.297 · 2026-H2 +0.201 · master +0.297. With the symmetric +0.25 entry cost these become 2025 −0.431 · 2026 +0.005.

**Pure path effect** (no slip on either side; tables §2b) by c, for W = 15, 2025 / 2026:

| c | A | B | C |
|---|---|---|---|
| +0.25 | −0.070 / −0.014 | −0.045 / −0.017 | −0.047 / +0.010 |
| +0.5 | −0.033 / +0.019 | −0.007 / −0.020 | −0.005 / +0.061 |
| +0.75 | −0.064 / −0.049 | −0.012 / −0.030 | −0.036 / +0.022 |
| +1.0 | −0.036 / −0.195 | −0.006 / −0.116 | −0.021 / −0.104 |

Even with free entries, confirmation adds nothing.

**Random-skip null (symmetric rulers).** Skipping the same share of signals at random scores, in 2025, **higher than every one of the 48 cells at both slips (0 / 96)**. Selection value ranges from −0.15 to −0.03 in 2025. In 2026 it ranges from −0.13 to +0.16 and is positive in 65 of 96 readings (anchoring C, whose 2026 credit is partly just a wider stop measured from the higher entry). One year positive and the other negative = **regime-dependent → no ship**, even before G1.

### Two-sided accounting (primary ruler, tables §3); example +0.75 W15 C

| block | N | dead-on-arrival losers avoided | winners never triggered | winners bought higher | triggered losers | ΣΔ |
|---|---|---|---|---|---|---|
| 2025 | 828 | 226 · +671.8 | 102 · −307.6 | 283 · −307.7 | 214 · −189.3 | **−135.6** |
| 2026 | 624 | 155 · +466.3 | 58 · −175.0 | 278 · −244.2 | 129 · −125.8 | **−81.4** |
| master | 20 | 8 · +24.1 | 1 · −3.0 | 10 · 0.0 | 1 · −1.0 | +20.2 |

Who gets filtered (tables §1, BT trigger rate by today's peak): +0.75 / W 15 triggers on 2 % of the < +0.5 trades, 46 % of +0.5…1, 75 % of +1…2 and 77 % of ≥ +2. That means 22 % of today's TP winners are skipped. The trigger is a sharp dead-on-arrival detector; the trouble is that it also skips 1 in 5 winners.

### Follow-through exit twin (tables §4)

| cell | ruler | 2025 | 2026 | master | fired (BT) | winners cut · Σ | losers saved · Σ |
|---|---|---|---|---|---|---|---|
| k3 p0.3 | RM −0.20 | −0.080 [0.13] | −0.185 [0.01] | +0.18 | 49 % | 300 · −1080 | 399 · +909 |
| k5 p0.5 | RM −0.20 | −0.083 [0.14] | −0.228 [0.00] | +0.17 | 53 % | 327 · −1165 | 435 · +966 |
| k10 p0.5 | RM −0.20 | −0.007 [0.47] | −0.053 [0.22] | +0.62 | 35 % | 184 · −677 | 322 · +649 |
| k10 p0.5 | RM 0 slip | +0.071 [0.89] | +0.008 [0.55] | +0.68 | 35 % | – | – |

### Gates (all 54 cells × 5 CF rulers / 3 FT rulers; `LITE_CONFIRM_gates.csv`)
- **G1** (Δ > 0, P ≥ 0.90 in 2025 AND 2026) passes in **0 of 258** cell-rulers.
- **G4** (concentration, which also requires ΣΔ > 0) fails everywhere under the conservative rulers.
- **G5** (positive neighbours) holds only in some symmetric 2026 readings.
- **Candidates: 0.** With no candidate there is no haircut, LOMO or concentration read to report.

---

## Could NOT be tested
- **Order-book slip of a stop-triggered market buy.** Prints only, so the slip is ruled by s. A buy-stop fires into a rising print, so its real slip is likely ≥ today's, which makes the symmetric reading optimistic.
- **Trigger latency.** A software trigger on the price stream adds roughly the live 8 s, and a Binance STOP_MARKET buy adds less. Both are priced at the trigger print. Earlier work found delay flat over 1–13 s (FRENZY_ENTRY_LATENCY_STUDY).
- **1m-bar triggers.** 38 % of triggers (2025 outside the top-50) fall on 1m bars: the fill is max(trigger, open) and the rest of that minute is not walked. The tick-only subset gives the same verdict (primary Δ −0.15…−0.29 at W15).
- **Slot effects.** A skipped signal frees one of LITE's 2 slots; per-signal accounting does not model the next signal.
- **Live.** EPIC and DUSK are on 1m replicas (DUSK today is unresolved there). ZK has no data. Queue position of the resting TP is not modelled (RM = ≥ 1 tick through).

## Exact commands
```
venv/bin/python scripts/study_lite_confirm_prereg.py      # freezes reports/LITE_CONFIRM_PREREG.txt (refuses overwrite)
venv/bin/python scripts/study_lite_confirm_cells.py --jobs 9   # rows + live rows + parity asserts + WAY-2 verify (~2.5 min)
venv/bin/python scripts/study_lite_confirm_report.py      # tables + gates
```
Code: `scripts/study_lite_confirm_common.py` (walkers, levels, WAY 2 loop), `_prereg.py`, `_cells.py`, `_report.py`.
