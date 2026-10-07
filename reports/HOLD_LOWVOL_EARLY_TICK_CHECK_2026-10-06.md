# HOLD_LOWVOL_EARLY: tick check (2026-10-06)

## Resumen en lenguaje simple

- **Pregunta.** El +0.215 %/trade del candidato HOLD_LOWVOL_EARLY, ¿es real o es un artefacto del orden de precios dentro de cada vela de 1 minuto?
- **Respuesta: en buena parte es un artefacto.** Re-precié con ticks reales (aggTrades) los mismos 2,239 trades, congelados antes de mirar ningún resultado. Quedaron 2,236 con datos.
  - El resultado baja de **+0.215 a +0.115 %/trade**.
  - El intervalo de confianza por días es **[−0.06, +0.30]**. Incluye el cero.
  - Con el timing real del bot (entrada 12 s después del cierre, 0.10 % de slippage) da **+0.045**.
- **Por qué baja.** Casi toda la diferencia viene de **59 trades (2.6 %)** con picos violentos dentro de un minuto. Explican el 83 % de la diferencia.
  - La vela de 1m asumía que el precio llegaba primero al máximo y después retrocedía.
  - Los ticks muestran que el trailing se tocaba antes del máximo. Ejemplo: ZORA, +32.5 % en 1m contra +11.1 % en ticks.
- **Entrar en cualquier vela al azar en el mismo estado rinde igual.** Esas velas al azar dan +0.117 en ticks (p = 0.51). No hay ventaja de timing: es el estado del mercado, no el momento de entrada.
- **No es un "WIDE más laxo".** Solo el 3 % de los trades coincide con una posición FRENZY/WIDE abierta, y solo el 9 % cae a menos de 1 h de una entrada del motor.
  - Lo que sí pasa: los trades en episodios que **más tarde** se vuelven FRENZY ganan +0.86. Los demás pierden −0.10. Eso no se puede saber al momento de entrar.
- **Veredicto (barra declarada de antemano): NO PASA.** La media en ticks (+0.115) supera +0.10, pero el límite inferior del intervalo (−0.06) no es mayor que cero. **Se cierra la línea HOLD_LOWVOL_EARLY.** No se propone ninguna línea observe ni ningún arm.
- **Estudio secundario de salidas (pedido del operador, pre-registrado).** El mejor trailing elegido en Ene–Abr (stop −4, se activa en +3, trail 3) gana +0.24 sobre el lock en Ene–Abr.
  - En May–Oct, sin tocar, la ventaja es **+0.007** con IC [−0.14, +0.16]. **NO PASA.**
  - La salida BULLRUN pierde en todos los cortes (−0.128).
  - Con un tope de 4–5 posiciones abiertas, el lock queda en +0.055…+0.066. Los pocos trades que se saltan son justamente los días de pump masivo, que ganan +2.2…+2.6 de media.

## 1. Freeze (before any tick pricing)

| item | value |
|---|---|
| fill list | `scratchpad/holdlow_ticks/FREEZE_HOLD_LOWVOL_EARLY_fills.csv` = `manualall/trades_C.pkl` with 2 ≤ hours ≤ 16.75 (cut unchanged), **2,239 rows**, 362 pairs, 2026-01-10 → 2026-10-03 |
| sha256 | `4b2ae2d40c81b7aca4381a3c2ecabecc4f544234ee9407907d4d5dac648d73dd` |
| frozen at | **2026-10-06T22:46:45Z** |
| null control | `FREEZE_null_same_state.csv`: `null_priced.pkl["C"]` rows (random non-state, above-VWAP bars, entry at a random 1m inside the next 5m) with 2 ≤ hours ≤ 16.75 and an episode (pair, spike_ts) that is in the fill list. 1,908 rows. sha256 `0b33a37e…db19` |
| 1m reproduction at freeze | LOCK +0.2151 (path order) / −0.0137 (pessimistic) / +3/−3 +0.1250 (matches the study) |
| secondary-exit pre-registration | `PREREG_SECONDARY_EXITS.txt`, **2026-10-06T22:48:14Z**, before any secondary cell was priced |
| primary verdict written | **2026-10-06T23:06:56Z** (`PRIMARY_VERDICT_WRITTEN_AT.txt`), before the secondary study ran |

**Primary tick ruler**, declared before pricing:
- Entry is the first aggTrade print ≥ the signal-bar close (= the next 5m open), +0.02 % slip. It is identical to the 1m `pe` (median rel. diff 0.0000 %).
- Fees 0.09 % round trip. Levels are on net %.
- The lock arms on the **prior-print** peak, and the exit is at the crossing print −0.02 %. 12 h cap.
- Sensitivity ruler (the live ruler of the engine-cohort studies): first print ≥ close + 12 s, no entry slip, exit −0.10 %, 1 % dislocation guard.

## 2. Coverage

- **Full coverage, no sampling.**
  - 1,777 of the 2,190 needed pair-days were already in `reports/backtest_cache/ticks{,_q}/`.
  - The other 413 were fetched from data.binance.vision daily aggTrades. I used my own scratch fetcher (`fetch_ticks.py`): same `ticks/<PAIR>/<date>.npz` format, 4 threads, 0.2 s pacing.
  - 2 pair-days returned 404 (EPTUSDT 2026-01-31, DAMUSDT 2026-04-30).
- **Priced:** fills **2,236 / 2,239** (3 have no ticks); null 1,906 / 1,908.
- **Live ruler:** 38 fills refused by the 1 % dislocation guard.

## 3. Pricer validation

| check | result |
|---|---|
| `FRENZY_ENGINE_COHORT_2026-10-05.csv`, every gate_ok row with tick outcome (958 = FRENZY 241 + WIDE 717), live ruler | **958/958 exact** on LOCK2 and FIX3 (max abs diff 0.0). Entry price max rel diff 9e-13. FRENZY mean +0.3588 = cohort |
| Live-window activations in `FRENZY_GVOL_GATE_REVALIDATION_2026-10-06.md` with daily tick files | SAND 10-04 05:05 +1.90 · AIN 10-04 11:15 +2.45 · SAND 10-04 14:05 −3.11 · MOVR 10-05 09:15 +1.90: **4/4 exact**. The other six (10-05 12:00 onward) have no daily archive yet, so they were not checkable |
| secondary-study LOCK cell vs primary LOCK_t | max abs diff 0.0 |

## 4. Primary result on ticks (frozen LOCK exit)

| set · exit · ruler | N | days | WR | BE WR | avg %/trade | day-CI 95 % | Jan–Apr / May–Oct | LOMO min…max | months > 0 | top pair / day share of gain |
|---|---|---|---|---|---|---|---|---|---|---|
| **fills · LOCK · ticks (primary)** | 2236 | 267 | 52.2 | 50.3 | **+0.115** | **[−0.061, +0.300]** | +0.051 / +0.159 | +0.08…+0.15 | 6/10 | 2.0 % / 4.6 % |
| fills · LOCK · 1m path order (same 2,236) | 2236 | 267 | 52.3 | 48.8 | +0.216 | [+0.032, +0.406] | +0.160 / +0.254 | | 8/10 | |
| fills · LOCK · 1m pessimistic | 2236 | | 52.3 | 52.5 | −0.014 | [−0.19, +0.17] | −0.057 / +0.017 | | 5/10 | |
| fills · LOCK · ticks, live ruler (12 s, 0.10) | 2198 | 267 | 52.6 | 51.9 | +0.045 | [−0.129, +0.229] | −0.024 / +0.093 | | 5/10 | |
| fills · +3/−3 · ticks | 2236 | | 52.3 | 50.4 | +0.116 | [−0.028, +0.250] | +0.080 / +0.140 | +0.10…+0.14 | 7/10 | 1.9 % / 3.3 % |
| fills · +3/−3 · ticks, live ruler | 2198 | | 52.7 | | +0.055 | [−0.086, +0.193] | −0.012 / +0.102 | | 7/10 | |
| **null (random bars, same state, same episodes) · LOCK · ticks** | 1906 | 250 | 52.4 | 50.4 | **+0.117** | [−0.166, +0.447] | +0.329 / −0.030 | | 4/10 | 2.3 % / 9.9 % |
| null · LOCK · 1m path | 1906 | | 52.9 | | +0.294 | | | | | |
| null · +3/−3 · ticks | 1906 | | 52.8 | | +0.145 | [−0.078, +0.372] | +0.253 / +0.071 | | 6/10 | |

**Comparisons and robustness**
- **Fill vs null on ticks.** LOCK: p **0.51** (null 95 % quantile +0.227). +3/−3: p 0.69.
- **The fill set has no timing edge over random bars in the same state.**
- **Haircut 30–50 %:** +0.057…+0.080.
- **Drop the best days:**

| best days dropped | mean %/trade |
|---|---|
| 1 | +0.063 |
| 5 | +0.005 |
| 10 | −0.041 |

**Per month (LOCK):**

| month | N | ticks | 1m path | 1m pess | +3/−3 ticks | null ticks |
|---|---|---|---|---|---|---|
| 2026-01 | 170 | +0.111 | +0.162 | −0.015 | +0.227 | +0.880 |
| 2026-02 | 241 | +0.387 | +0.558 | +0.216 | +0.228 | +0.799 |
| 2026-03 | 226 | −0.106 | +0.026 | −0.136 | −0.014 | −0.120 |
| 2026-04 | 277 | −0.151 | −0.079 | −0.257 | −0.061 | −0.112 |
| 2026-05 | 271 | +0.192 | +0.313 | +0.112 | +0.146 | −0.252 |
| 2026-06 | 257 | +0.172 | +0.275 | +0.048 | +0.173 | −0.127 |
| 2026-07 | 187 | −0.014 | +0.102 | −0.241 | +0.028 | −0.171 |
| 2026-08 | 285 | +0.282 | +0.328 | +0.114 | +0.214 | +0.347 |
| 2026-09 | 292 | +0.218 | +0.316 | +0.068 | +0.204 | +0.010 |
| 2026-10 (partial) | 30 | −0.921 | −0.813 | −0.916 | −0.822 | −0.246 |

### Per-fill tick − 1m (path-order) difference

| quantile | 1 % | 5 % | 10 % | 25 % | 50 % | 75 % | 90 % | 99 % |
|---|---|---|---|---|---|---|---|---|
| Δ % | −2.19 | −0.085 | −0.044 | −0.019 | −0.008 | −0.003 | −0.001 | 0.000 |

**Size of the change**
- Mean Δ −0.101. 91 % of fills move by less than 0.05 and 97 % by less than 0.25.
- Only 1 fill improves by more than 0.05.
- Against the pessimistic 1m, ticks are +0.128 better. The truth sits about halfway between the two 1m orderings.

| 1m exit → tick exit | n | 1m mean | tick mean | share of total Δ |
|---|---|---|---|---|
| LOCK → LOCK | 1149 | +3.144 | +2.964 | **92 %** |
| SL → SL | 1057 | −3.020 | −3.035 | 7 % (crossing-print slippage past −3) |
| CAP → CAP | 29 | +2.04 | +2.04 | 0 |
| LOCK → SL | 1 | +1.98 | −0.22 | 1 % |

**Why the fills changed**
- **The tail drives it.** 59 fills with Δ < −0.5 carry **83 %** of the total Δ, at a mean of −3.2 each.
- **They are pump minutes.** On a 400-fill sample of LOCK exits, the tick exit line sits on average 0.23 below the 1m line, while the crossing-print gap is only −0.02.
- **The mechanism.** On a huge green spike minute, the colour order (O→L→H→C) credits the full bar high as the trail's peak. On ticks, the price retraced 2 points before it reached that high. So the lock fired lower: ZORA +32.5 → +11.1, TAC +18.3 → +5.2, PLAY +15.3 → +2.6.
- **What it means.** The 1m "edge" was largely a few fat right-tail minutes credited at their extreme.

## 5. Overlap with FRENZY_LONG / FRENZY_WIDE

**What was compared.** The engine cohort (`FRENZY_ENGINE_COHORT_2026-10-05.csv`, fresh FRENZY / WIDE entries, through 2026-09-27) against the 2,177 fills inside its window.

| relation to engine entries | share of fills | LOCK tick mean (in / out) |
|---|---|---|
| same episode (pair + spike) has any engine entry | 31 % | +0.52 / −0.04 |
| … a gate-passing one | 23 % | +0.50 / +0.02 |
| … engine entry **before** the fill | 15 % | −0.21 / +0.19 |
| … engine entry **after** the fill (episode later turns ON) | 24 % | **+0.86** / −0.10 |
| engine entry within ±1 h, same pair | 8.7 % | +1.01 / +0.05 |
| an engine position still open at the fill time | **3.0 %** | −0.46 / +0.15 |

**Live fills.** All 11 live FRENZY_LONG / WIDE fills in the master pool (10-03 → 10-06) have no frozen fill within 6 h. The window barely overlaps: fills end 10-03 22:50.

**Reading**
- **It is not a looser WIDE.** It almost never trades alongside a WIDE / FRENZY position.
- **Where the year's profit comes from.** It is the 24 % of fills whose episode **later** turns FRENZY-ON (+0.86). Those are pre-run holds. That is outcome knowledge, unknowable at entry.
- **Everything else is ≤ 0.** Episodes that never go ON: −0.04. Fills after an engine entry: −0.21.

## 6. Verdict (pre-declared bars)

**Bar:** tick mean ≥ +0.10 ∧ day-CI lower bound > 0 ∧ May–Oct > 0.

| leg | value | pass? |
|---|---|---|
| tick mean | +0.115 | ✓ |
| day-CI lower bound | −0.061 | **✗** |
| May–Oct | +0.159 | ✓ |

**→ FAIL. Close the line HOLD_LOWVOL_EARLY.**
- No scout observe line and no arm proposal.
- Also failing outside the bar:
  - On the live timing ruler it is +0.045.
  - Under a 4-position cap it is +0.055.
  - It does not beat random bars in the same state (p 0.51).

**State-not-timing caveat.** Whatever is positive here belongs to the **state** ("a flagged pair still above its spike VWAP, low volume, 2–17 h after the spike"), not to the entry timing. Random bars in that state earn the same on ticks. And the positive part is concentrated in episodes that later re-ignite, which is not observable at entry.

## 7. Secondary exit study (operator addendum; pre-registered at 22:48:14Z, run after the primary verdict)

### 7.1 Design

- **Family, 32 cells.** All on the same 2,236 tick paths and the same entry:
  - LOCK baseline;
  - 24 trailing cells: stop S ∈ {−2, −3, −4}; (activation, trail) ∈ {(1,1), (2,1), (2,1.5), (2,2), (3,1), (3,1.5), (3,2), (3,3)}; declared prune trail ≤ activation; line = max(S, peak − trail);
  - ATR lock k ∈ {1.5, 2, 3} × 5m ATR (stop −3, arm +3, floor +2);
  - BE15_T2 (stop −3; at +1.5 the line becomes max(0, peak − 2));
  - 4 h / 8 h time overlays on the best Jan–Apr cell;
  - BULLRUN as live (`_bullrun_exit_for`, GREEN door): SL min(−0.7, max(−1.5·ATR, −1.2)), arm +1.0 → max(+0.2, peak − 2·ATR, ladder).
- **Selection.** Top 3 by Jan–Apr mean. PASS = the #1 finalist's May–Oct paired Δ vs LOCK (day-block CI) has lower bound > 0 and its May–Oct mean > 0.

### 7.2 Results

**Jan–Apr ranking:**

| rank | cell | Jan–Apr mean |
|---|---|---|
| 1 | TRAIL_s−4_a3_t3 | +0.287 |
| 2 | TRAIL_s−3_a3_t3 | +0.211 |
| 3 | TRAIL_s−4_a3_t3 + T8h | +0.202 |
| baseline | LOCK | +0.051 |

| cell | N | WR | BE WR | avg (all) | day-CI | Jan–Apr / May–Oct | months > 0 | top pair / day |
|---|---|---|---|---|---|---|---|---|
| LOCK | 2236 | 52.2 | 50.3 | +0.115 | [−0.06, +0.30] | +0.051 / +0.159 | 6/10 | 2 % / 5 % |
| **TRAIL_s−4_a3_t3** | 2236 | 59.5 | 56.4 | +0.216 | [−0.04, +0.48] | +0.287 / +0.166 | 8/10 | 2 % / 6 % |
| TRAIL_s−3_a3_t3 | 2236 | 51.7 | 49.3 | +0.142 | [−0.10, +0.40] | +0.211 / +0.094 | 7/10 | 2 % / 7 % |
| TRAIL_s−4_a3_t3+T8h | 2236 | 58.9 | 56.4 | +0.166 | [−0.06, +0.39] | +0.202 / +0.142 | 8/10 | 2 % / 5 % |
| ATRk_k2 | 2236 | 52.2 | 50.7 | +0.089 | [−0.08, +0.26] | +0.075 / +0.100 | 8/10 | |
| BE15_T2 | 2236 | 49.6 | 49.2 | +0.013 | [−0.12, +0.15] | −0.033 / +0.045 | 4/10 | |
| BULLRUN | 2236 | 49.4 | 55.3 | **−0.128** | [−0.20, −0.06] | −0.074 / −0.166 | 2/10 | |

**Paired vs LOCK, two-sided.** Improved / worsened count fills that move by more than 0.05.

| cell | period | Δ mean | paired day-CI | improved / worsened | saved losers / killed winners | Σ gains / Σ losses |
|---|---|---|---|---|---|---|
| TRAIL_s−4_a3_t3 | Jan–Apr (discovery) | +0.237 | [+0.01, +0.46] | 221 / 673 | 71 / 6 | +924 / −708 |
| **TRAIL_s−4_a3_t3** | **May–Oct (confirm)** | **+0.007** | **[−0.14, +0.16]** | 282 / 1013 | 104 / 7 | +1087 / −1078 |
| TRAIL_s−3_a3_t3 | May–Oct | −0.065 | [−0.16, +0.04] | 171 / 505 | 0 / 7 | +484 / −570 |
| TRAIL_s−4_a3_t3+T8h | May–Oct | −0.017 | [−0.15, +0.12] | 302 / 1009 | 106 / 15 | +1066 / −1089 |
| BULLRUN | May–Oct | −0.325 | [−0.52, −0.13] | 718 / 599 | 207 / 261 | +1527 / −1957 |

**Per month, the top cell vs LOCK:**

| | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|---|
| TRAIL_s−4_a3_t3 | +0.30 | +0.72 | +0.25 | −0.07 | +0.14 | +0.15 | +0.10 | +0.18 | +0.32 | −0.62 |
| LOCK | +0.11 | +0.39 | −0.11 | −0.15 | +0.19 | +0.17 | −0.01 | +0.28 | +0.22 | −0.92 |

**What TRAIL_s−4_a3_t3 trades off against the lock**
- The −4 stop saves losers that the −3 stop would have cut: 175 over the year.
- It costs about 1 pt on most stopped fills, and the trail-3 gives back more on lock wins.
- Its gains and losses roughly cancel out of sample.

**Secondary verdict: FAIL.**
- The best-of-family's out-of-sample Δ is +0.007, CI spanning 0.
- No exit change is recommended.
- The haircut is moot: the 30–50 % haircut on the all-period Δ (+0.101) would give +0.05…+0.07, but it is not confirmed.
- BULLRUN is clearly worse than the lock on this cohort. Its tight ATR stop (≈ −0.7…−1.2) gets shaken out.

### 7.3 Hold time, concurrency, position cap (chronological, first-come)

| exit | hold mean / median (min) | daily max concurrent: mean / p90 / max | cap 4: skipped · taken mean [day-CI] · May–Oct · skipped mean | cap 5: skipped · taken mean · skipped mean |
|---|---|---|---|---|
| LOCK | 88 / 43 | 2.5 / 4 / 27 | 63 (2.8 %) · **+0.055** [−0.10, +0.20] · +0.145 · +2.17 | 43 (1.9 %) · +0.066 · +2.60 |
| TRAIL_s−4_a3_t3 | 130 / 69 | | 113 (5.1 %) · +0.102 [−0.10, +0.30] · +0.119 · +2.35 | 65 (2.9 %) · +0.121 · +3.38 |
| TRAIL_s−3_a3_t3 | 101 / 52 | | 79 (3.5 %) · +0.043 · +0.063 · +2.86 | 49 (2.2 %) · +0.050 · +4.27 |
| TRAIL_s−4_a3_t3+T8h | 120 / 69 | | 103 (4.6 %) · +0.093 · +0.111 · +1.70 | 60 (2.7 %) · +0.102 · +2.49 |

- **What gets skipped.** The cap skips exactly the market-wide pump days: 2026-02-06 had 52 signals at +2.27 average, and 2026-08-21 had 24 at +2.56. Those days carry a large share of the year's mean.
- **What it means for a real book.** A real book capped at 4–5 positions, sharing slots with the other sleeves, would realise less than the uncapped figure.

## Files

Report: `reports/HOLD_LOWVOL_EARLY_TICK_CHECK_2026-10-06.md`. All other files are in `scratchpad/holdlow_ticks/`.

| purpose | files |
|---|---|
| freeze | `FREEZE_*.csv`, `freeze.py` |
| pre-registration and verdict timestamp | `PREREG_SECONDARY_EXITS.txt`, `PRIMARY_VERDICT_WRITTEN_AT.txt` |
| tick pricer and fetcher | `ticklib.py`, `fetch_ticks.py` |
| validation | `validate.py`, `validate_out.csv` |
| primary pricing and results | `price_primary.py`, `tick_*.pkl`, `analyze_primary.py`, `primary.out`, `primary_*.csv`, `diag.py` |
| overlap | `overlap.py`, `overlap.out`, `overlap.csv` |
| secondary study | `price_secondary.py`, `sec_*.pkl`, `analyze_secondary.py`, `secondary.out`, `secondary_*.csv` |

413 new pair-day tick files were added to `reports/backtest_cache/ticks/` (same format, cache only). No bot code, config, template or test was touched.
