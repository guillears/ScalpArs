# FRENZY market-volume gate: which definition of "market volume"? (2026-10-06)

Research only. No code, config or test was touched, and nothing was committed. This is an **unreviewed backtest**. Under the no-arm-before-review rule it supports no ship until the caveman and deep reviews have checked it.

## Resumen en lenguaje simple

**Recomendación: mantener la fórmula actual de FRENZY (V1). No cambiarla por la del dashboard ("Vol") ni por ninguna otra.**

- **Por qué no es la misma cifra que el dashboard.** El "Vol" del dashboard y el filtro de FRENZY miden cosas parecidas, pero no iguales:
  - El dashboard **suaviza** las últimas ~5 velas (EMA5).
  - El dashboard incluye la vela **todavía abierta**, a medio formar.
  - El dashboard **saca los pares de la blacklist**.
  - FRENZY mira **solo la vela cerrada** de la señal.

  Por eso hoy, en la vela de las 21:00, FRENZY vio **1,61** (PUMP solo aportó el 55 % del volumen), mientras el dashboard mostraba **≈0,85**.
- **Probé 5 fórmulas sobre las mismas 588 señales del año**, con el mismo precio de salida (lock en vivo):
  - V1 actual;
  - V2 dashboard, tal como FRENZY lo vería en su momento de decidir;
  - V3 = V1 en dólares;
  - V4 = mediana por par;
  - V5 = dashboard en dólares.
- **V1 es la que mejor separa las operaciones buenas de las malas:**
  - Deja pasar operaciones que promedian **+0,41 %/trade** y bloquea las que promedian **−0,28 %**, una diferencia de +0,69.
  - El libro del año queda en **+349 %**, contra **+73 %** sin filtro.
- **La del dashboard (V2) queda un poco peor, pero estadísticamente no se puede distinguir de V1:**
  - Diferencia +0,57 contra +0,69. La diferencia entre ambas tiene un intervalo de [−0,66, +0,45].
  - Libro +239 % contra +349 %.
  - Va peor justo en el período reciente: en Jul–Sep su diferencia es −0,51, contra +0,09 de V1.
- **Las versiones en dólares (V3, V5) y la mediana (V4) son claramente peores.**
  - V5 directamente no separa nada: la diferencia es −0,01.
  - Usar dólares, que suena "más justo", **empeora** el filtro.
  - El volumen en monedas baratas (PEPE, PUMP, PENGU…) es justamente la señal que funciona.
- **Ninguna alternativa supera a V1 por más de lo que da la suerte** (prueba de barajado: p = 0,94).
- **¿Unificar con el dashboard es solo cosmético? No del todo.**
  - Las dos son estadísticamente equivalentes.
  - Pero todas las estimaciones puntuales favorecen a V1.
  - Además, el número del dashboard **salta según el segundo en que se lea**: justo después del cierre de vela cae ~30 % (hoy a las 21:05 pasó de 0,85 a 0,63). Eso es un defecto para un filtro.
- **Si querés coherencia visual**, lo más limpio es **mostrar en el dashboard el número de FRENZY (V1) como una cifra aparte** ("Vol FRENZY"), no cambiar el filtro. Eso es una decisión de pantalla, no de estrategia, y queda en tus manos.
- **Advertencia ya conocida:** la ventaja del filtro (de cualquier versión) viene de Ene–Jun. En Jul–Sep ninguna versión separa bien. Esto no cambia la comparación, pero sí el peso de la evidencia del filtro en sí (ver `FRENZY_GVOL_GATE_REVALIDATION_2026-10-06.md`).

## 1. The five definitions, and how each was rebuilt

| | definition | universe | bar | rebuilt from |
|---|---|---|---|---|
| **V1** (live FRENZY) | Σ base vol on the signal bar ÷ Σ each pair's 48-bar mean (ending at and including the signal bar) | top-50 by 24 h quote vol (COIN, ≥ 90 d, not Alpha), **no blacklist** | **closed** signal bar | `wrev/gvol_live.py` U2 logic. It reproduces the cohort's U2 exactly (MAE 0.0000 on 1,801 bars) and live stamps to ±0.005 (revalidation §1). |
| **V2** (dashboard "Vol") | Σ EMA5(base vol) ÷ Σ SMA48(base vol), over the scan's 100 klines whose **last row is the forming bar** | V1's top-50 **minus `pair_blacklist`** (the scan removes it after the cut) | forming bar at the **last scan's read time** | `services/indicators.py` L54–89 and `trading_engine.py` L14835–14914, with 1m klines for the forming bar |
| V2c | V2's formula on **closed** bars ending at the signal bar | as V2 | closed | 5m cache |
| **V3** | V1 in quote $ (Σ quote vol ÷ Σ 48-bar mean quote vol) | as V1 | closed | 5m cache |
| **V4** | median over pairs of (bar vol ÷ own 48-bar mean) | as V1 | closed | 5m cache |
| **V5** | V2 in quote $ | as V2 | forming, same read time | 1m + 5m |
| V5c | V2c in quote $ | as V2 | closed | 5m cache |

**Forming bar or closed bar? (V2.)** `get_ohlcv` returns Binance klines, which include the still-open bar. So `indicators['volume']` is EMA5 including a **partial** bar, and `avg_volume_global` is the SMA48 including it. The scan loop runs continuously (≥ 30 s apart, ~30–60 s per scan). The global is only overwritten at the end of each scan's Phase 2. At FRENZY's judge time (close + 4–12 s) the dashboard therefore shows the value of the **last completed scan**, read some tens of seconds before the close.

**V2 reconstruction, validated against the real stamps.** `entry_global_volume_ratio` on fills **is** the scan value (FRENZY copies `_global_volume_ratio`; it is not its own gvol). I modelled the read time as `open − offset`, with offsets of 0–300 s and the current minute pro-rated from 1m klines, and the universe and blacklist taken from `trading_config.json` at that date:

| fills | N | median abs error (best read time) | ≤ 0.01 | ≤ 0.03 | same side of 1.0 |
|---|---|---|---|---|---|
| Jun 17 – Sep 24 (all sleeves) | 548 | 0.0022 | 94 % | 97 % | 99 % |
| Oct 2 – 6 (incl. 12 FRENZY) | 27 | 0.0022 | 100 % | 100 % | 100 % |

- **The formula is right.** The read time is per-fill unknown, so it is fitted, but the error surface is smooth with one minimum.
- **FRENZY read time:** on the 12 FRENZY fills, the best read is a median **69 s before the signal close** (range 21–119 s).
- **Pre-registered for the year:** V2 = V5 read at **close − 70 s**. This was fixed on stamps, never on P&L.
- At that fixed read the 12 FRENZY stamps are matched with MAE 0.014 (max 0.047), all 12 on the same side of 1.0.
- Sensitivity across 15–120 s is in §4.

**The sawtooth.** If the last scan read just after the close, the forming bar is the new, empty one: EMA5 loses one third and SMA48 barely moves. The dashboard value then drops by ~30 % for a minute.
- Examples: AIN 10-03, 0.93 → 0.65; SAND 10-04, 0.99 → 0.69; the 21:00 bar today, 0.85 → 0.63.
- The dashboard number depends on **when** the scan happened to run. That is a property worth knowing before wiring it into any gate.

## 2. Sanity case, 2026-10-06 20:15–21:20 (5m signal bars; "close" = when FRENZY judges)

| signal bar | close | **V1** | V3 $ | V4 median | V2c closed | V5c closed $ | **V2 dashboard @ close−70 s** | V5 @ −70 s | V2 @ close+4 s | PUMP share of Σ base vol | top pair (share) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20:15 | 20:20 | 0.49 | 0.59 | 0.58 | 0.85 | 0.56 | 0.82 | 0.51 | 0.58 | 23 % | 1000PEPE 28 % |
| 20:20 | 20:25 | 0.64 | 0.76 | 0.67 | 0.76 | 0.63 | 0.71 | 0.58 | 0.52 | 9 % | PENGU 31 % |
| 20:25 | 20:30 | 0.55 | 0.66 | 0.57 | 0.67 | 0.64 | 0.62 | 0.60 | 0.46 | 21 % | 1000PEPE 30 % |
| 20:30 | 20:35 | 0.58 | 0.60 | 0.60 | 0.62 | 0.63 | 0.56 | 0.58 | 0.43 | 20 % | PENGU 26 % |
| 20:35 | 20:40 | 0.81 | 0.56 | 0.62 | 0.70 | 0.61 | 0.65 | 0.55 | 0.48 | 23 % | PENGU 26 % |
| 20:40 | 20:45 | 0.94 | 0.51 | 0.62 | 0.76 | 0.57 | 0.71 | 0.55 | 0.52 | 14 % | VTHO 29 % |
| 20:45 | 20:50 | 0.58 | 0.61 | 0.59 | 0.68 | 0.59 | 0.66 | 0.53 | 0.47 | 14 % | VTHO 27 % |
| 20:50 | 20:55 | 0.61 | 0.44 | 0.51 | 0.64 | 0.53 | 0.59 | 0.51 | 0.45 | 23 % | PUMP 23 % |
| 20:55 | 21:00 | 0.60 | 0.45 | 0.44 | 0.63 | 0.51 | 0.58 | 0.48 | 0.43 | 53 % | PUMP 53 % |
| **21:00** | **21:05** | **1.61** | 0.87 | **0.89** | 0.91 | 0.61 | **0.85** | 0.54 | 0.63 | **55 %** | PUMP 55 % |
| 21:05 | 21:10 | 0.99 | 0.56 | 0.74 | 0.90 | 0.58 | 0.81 | 0.53 | 0.62 | 42 % | PUMP 42 % |
| 21:10 | 21:15 | 1.07 | 0.58 | 0.63 | 0.94 | 0.57 | 0.90 | 0.53 | 0.65 | 62 % | PUMP 62 % |
| 21:15 | 21:20 | 0.99 | 0.58 | 0.58 | 0.92 | 0.55 | 0.87 | 0.52 | 0.63 | 35 % | PUMP 35 % |

- These reproduce the operator's reading: V1 1.61–1.62, median 0.88–0.89, PUMP 55 %.
- On the 21:00 bar, V1 alone crosses 1.0. Every other definition reads 0.5–0.9.
- That bar is exactly what V1 is built to react to: one low-priced meme coin doing 2–3× its 4-hour norm.

## 3. Year test: same cohort, same pricing

**Data:**
- **Cohort:** the revalidation's engine-bar cohort (`wrev/cohort_plus.pkl`): FRENZY_LONG READY 352 plus WIDE hold-green 236, giving **588 eligible signals**, Jan 10 → Sep 27, live-tradeable pairs, priced.
- **Engine parity:** V1 here equals the cohort's U2 bit-for-bit. The revalidation verified U2 against 18 live reads, including ORCA 10-06 18:05 (stamp 0.7802, rebuilt 0.780).
- **Exits:** `LOCK2` = live lock (stop −3, arm +3, floor +2, trail 2) on ticks, entry at close + 12 s.
- **Costs:** 0.09 % fees plus **0.10 %** slip. This is what the revalidation actually used, not 0.02 %. At 0.02 % slip every trade gains +0.08, which leaves every gap unchanged; the books are in §4.
- **Books:** step2's sequencing (slots, pair-day cap, gate judged last) and live sizing.

**Two thresholds per version, pre-registered:**
- (a) keep **1.0**;
- (b) the threshold that blocks the **same share as V1 at 1.0** (43.9 %).

Neither was fitted on outcome.

| version | rule | thr | taken N · WR · avg | blocked N · WR · avg | **gap T−B** [day CI] | taken avg [day CI] | Δ vs no gate | book / DD | Jan–Jun gap | Jul–Sep gap | 1st / 2nd half gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **V1** | 1.0 | 1.00 | 330 · 56 % · **+0.414** | 258 · 46 % · **−0.277** | **+0.69** [+0.15, +1.21] | [+0.05, +0.78] | +0.303 | **+349 % / −54 %** | +0.92 | +0.09 | +1.04 / +0.33 |
| **V2** dashboard | 1.0 | 1.00 | 374 · 55 % · +0.317 | 214 · 46 % · −0.250 | +0.57 [−0.04, +1.17] | [−0.03, +0.66] | +0.207 | +239 % / −55 % | +1.00 | **−0.51** | +1.12 / −0.01 |
| V2 | rate | 0.94 | 330 · 54 % · +0.317 | 258 · 48 % · −0.153 | +0.47 [−0.06, +1.00] | [−0.05, +0.67] | +0.206 | +195 % / −50 % | +0.86 | −0.52 | +0.97 / −0.05 |
| V2c closed | 1.0 | 1.00 | 315 · 55 % · +0.370 | 273 · 48 % · −0.188 | +0.56 [+0.01, +1.10] | [−0.03, +0.74] | +0.259 | +270 % / −51 % | +0.96 | −0.45 | +1.01 / +0.09 |
| V2c closed | rate | 1.015 | 330 · 56 % · +0.442 | 258 · 46 % · −0.312 | +0.75 [+0.18, +1.30] | [+0.05, +0.80] | +0.331 | +399 % / −50 % | +1.14 | −0.22 | +1.35 / +0.14 |
| V3 $ | 1.0 | 1.00 | 370 · 53 % · +0.214 | 218 · 49 % · −0.064 | +0.28 [−0.23, +0.80] | [−0.13, +0.56] | +0.103 | +135 % / −70 % | +0.57 | −0.56 | +0.87 / −0.36 |
| V3 $ | rate | 0.91 | 330 · 55 % · +0.265 | 258 · 48 % · −0.086 | +0.35 [−0.14, +0.85] | [−0.09, +0.62] | +0.154 | +137 % / −57 % | +0.57 | −0.28 | +0.67 / +0.01 |
| V4 median | 1.0 | 1.00 | 458 · 53 % · +0.209 | 130 · 46 % · −0.235 | +0.44 [−0.16, +1.05] | [−0.10, +0.53] | +0.098 | +199 % / −64 % | +0.72 | −0.55 | +0.84 / −0.08 |
| V4 median | rate | 0.80 | 330 · 51 % · +0.084 | 258 · 52 % · +0.145 | −0.06 [−0.59, +0.50] | [−0.28, +0.45] | −0.027 | +27 % / −66 % | +0.12 | −0.60 | +0.16 / −0.30 |
| V5 dash $ | 1.0 | 1.00 | 379 · 52 % · +0.109 | 209 · 51 % · +0.114 | −0.01 [−0.57, +0.55] | [−0.24, +0.45] | −0.002 | +39 % / −65 % | +0.43 | −1.18 | +0.61 / −0.61 |
| V5 dash $ | rate | 0.93 | 330 · 52 % · +0.152 | 258 · 50 % · +0.058 | +0.09 [−0.42, +0.61] | [−0.20, +0.50] | +0.041 | +66 % / −55 % | +0.40 | −0.72 | +0.57 / −0.38 |
| V5c closed $ | 1.0 | 1.00 | 332 · 52 % · +0.105 | 256 · 51 % · +0.118 | −0.01 [−0.53, +0.52] | [−0.25, +0.45] | −0.005 | +45 % / −61 % | +0.31 | −0.86 | +0.48 / −0.51 |

**Reference rows:**
- No gate: 588 signals, +0.111 %/trade, book +73 %.
- The chronological half cut is 2026-04-24.
- At matched rate V1 is unchanged (its threshold is 0.998).

### 3a. Head-to-head vs V1 (paired, same signals, day-block bootstrap)

| version | rule | gap − V1 gap | 95 % CI | P(≤ 0) | book − V1 | beats V1 in Jan–Jun / Jul–Sep / 1st / 2nd half |
|---|---|---|---|---|---|---|
| V2 | 1.0 | −0.12 | [−0.66, +0.45] | 0.67 | −110 pp | Y / n / Y / n |
| V2 | rate | −0.22 | [−0.71, +0.29] | 0.81 | −154 pp | n / n / n / n |
| V2c | 1.0 | −0.13 | [−0.62, +0.36] | 0.71 | −79 pp | Y / n / n / n |
| V2c | rate | **+0.06** | [−0.40, +0.55] | 0.39 | +50 pp | Y / n / Y / n |
| V3 | 1.0 / rate | −0.41 / −0.34 | [−0.99, +0.18] / [−0.95, +0.26] | 0.91 / 0.86 | −214 / −212 | none |
| V4 | 1.0 / rate | −0.25 / **−0.75** | [−0.84, +0.40] / [−1.36, −0.16] | 0.77 / 0.99 | −150 / −322 | none |
| V5 | 1.0 / rate | −0.70 / −0.60 | [−1.43, +0.06] / [−1.24, +0.09] | 0.97 / 0.96 | −311 / −283 | none |
| V5c | 1.0 / rate | **−0.70** / −0.66 | [−1.33, −0.02] / [−1.30, +0.03] | 0.98 / 0.97 | −304 / −295 | none |

**Selection-adjusted null.** The version columns were permuted jointly across signals within month, 2,000 times. The statistic is the best non-V1 gap minus V1's gap.
- **Rule 1.0:** the best alternative (V2) sits at −0.12 against a null 95th percentile of +0.73, **p = 0.94**.
- **Matched rate:** the best alternative (V2c) sits at +0.06 against +0.67, **p = 0.76**.
- With the 30–50 % haircut, V2c's +0.06 becomes +0.03.
- **No version beats V1 out of sample in both halves.** No alternative clears the null.

**Where V1 and V2 disagree (year):**
- V1 alone blocks 88 signals averaging −0.09 %.
- V2 alone blocks 44 signals averaging **+0.23 %**: V2 throws away winners.
- In Jan–Jun V2's extra blocks were right (−0.37, N 28).
- In Jul–Sep they were badly wrong (+1.29, N 16; September alone 7 at +3.51).
- These are small window counts. Read them as noise, not as a regime finding.

### 3b. Per month (gap taken − blocked; blocked N in brackets), rule 1.0

| month | V1 | V2 | V2c | V3 | V4 | V5 | V5c |
|---|---|---|---|---|---|---|---|
| Jan | +2.21 (39) | +1.40 (29) | +1.51 (34) | +0.91 (31) | +2.02 (20) | +0.78 (29) | +0.61 (40) |
| Feb | +0.64 (35) | +1.01 (28) | +1.30 (36) | +2.16 (28) | +1.69 (19) | +1.25 (19) | +1.13 (29) |
| Mar | +1.52 (50) | +1.84 (41) | +1.25 (55) | +1.21 (44) | +0.30 (27) | +0.54 (40) | +0.68 (46) |
| Apr | −0.68 (25) | −0.65 (20) | −0.16 (26) | −1.17 (25) | −1.17 (15) | −0.08 (24) | −0.77 (29) |
| May | +0.51 (30) | +1.02 (25) | +0.98 (32) | −0.45 (32) | +0.57 (19) | +0.39 (34) | +0.47 (36) |
| Jun | +0.57 (8) | +0.09 (9) | +0.09 (9) | +0.26 (6) | +1.43 (3) | −1.33 (8) | −1.54 (9) |
| **Jul** | −0.00 (19) | +0.10 (18) | −0.11 (25) | −1.24 (10) | +0.00 (5) | −1.37 (14) | −1.04 (15) |
| **Aug** | +0.17 (24) | +1.05 (21) | +1.15 (25) | +0.38 (20) | −0.18 (8) | −0.87 (19) | −0.03 (24) |
| **Sep** | +0.24 (28) | **−2.46** (23) | −2.24 (31) | −0.88 (22) | −0.73 (14) | −1.21 (22) | −1.33 (28) |
| leave-one-month-out gap | **+0.46 … +0.86** | +0.26 … +0.92 | +0.38 … +0.88 | +0.04 … +0.46 | +0.18 … +0.65 | −0.14 … +0.14 | −0.17 … +0.15 |
| months gap > 0 | **7/9** | 7/9 | 6/9 | 5/9 | 6/9 | 4/9 | 4/9 |

- V1 is the only version that is non-negative in every month from Jul to Sep, though it is barely positive.
- Matched-rate months tell the same story. The full table is in the scratch `eval_out.md`.

### 3c. Blocked side vs the locked expectancy bar

The sleeve breakeven WR is 49.8 % (all eligible signals, 1×). P(mean < 0) uses a bootstrap clustered on 4-hour windows.

| version | rule | blocked N | windows | WR (< BE?) | mean | P(mean < 0) | worst pair / worst day / top-3 days share of blocked loss |
|---|---|---|---|---|---|---|---|
| **V1** | 1.0 | 258 | 222 | 46 % (yes) | −0.277 | **0.91** | B 15 % / 01-14 17 % / 41 % |
| V2 | 1.0 | 214 | 183 | 46 % (yes) | −0.250 | 0.85 | TAC 20 % / 03-11 24 % / 64 % |
| V2c | rate | 258 | 217 | 46 % (yes) | −0.312 | **0.94** | TAC 13 % / 03-11 16 % / 43 % |
| V3 | 1.0 | 218 | 186 | 49 % (yes) | −0.064 | 0.64 | STEEM 67 % / 08-14 90 % (concentrated) |
| V4 | 1.0 | 130 | 120 | 46 % (yes) | −0.235 | 0.76 | AXS 30 % / 01-14 20 % / 61 % |
| V5 / V5c | 1.0 | 209 / 256 | 178 / 212 | 51 % (no) | +0.11 / +0.12 | 0.32 / 0.30 | blocked side is net positive |

- **No version meets the 95 % bar.** V1 stays the declared operator override that it already was (DECISION_LOG 194 and the revalidation).
- V2c at matched rate (0.94) is statistically the same as V1 (0.91).

### 3d. Distribution and correlation (588 eligible signals)

| version | mean | p10 | p25 | median | p75 | p90 | max | share ≥ 1.0 |
|---|---|---|---|---|---|---|---|---|
| V1 | 1.14 | 0.52 | 0.67 | 0.94 | 1.28 | 1.78 | 22.6 | 44 % |
| V2 | 1.00 | 0.58 | 0.73 | 0.90 | 1.13 | 1.48 | 8.2 | 36 % |
| V2c | 1.07 | 0.62 | 0.78 | 0.97 | 1.21 | 1.63 | 8.7 | 46 % |
| V3 | 1.11 | 0.47 | 0.63 | 0.86 | 1.19 | 1.81 | 18.1 | 37 % |
| V4 | 0.89 | 0.49 | 0.60 | 0.75 | 0.93 | 1.26 | 19.2 | 22 % |
| V5 | 0.98 | 0.53 | 0.67 | 0.87 | 1.12 | 1.51 | 7.3 | 36 % |
| V5c | 1.06 | 0.57 | 0.72 | 0.93 | 1.19 | 1.63 | 7.7 | 44 % |

**Spearman correlation with V1:** V2 0.78 · V2c 0.83 · V3 0.53 · V4 0.61 · V5 0.43 · V5c 0.45. V2 and V2c correlate at 0.99, and V3 and V4 at 0.87.

**Same block decision as V1 at 1.0:** V2 78 % · V2c 81 % · V3 70 % · V4 72 % · V5 64 %.

- **Dollar weighting is the largest departure from V1.** It turns "the busy low-priced coins" into "BTC/ETH-dominated turnover", and that read carries no FRENZY edge.
- **Smoothing (EMA5) and dropping the blacklist matter much less.**

## 4. Robustness

**V2 read time** (pre-registered 70 s; the gap is at 1.0):

| s before close | 15 | 30 | 45 | 60 | **70** | 90 | 120 |
|---|---|---|---|---|---|---|---|
| block % | 44 | 42 | 39 | 38 | 36 | 35 | 31 |
| gap | +0.71 | +0.60 | +0.58 | +0.54 | +0.57 | +0.59 | +0.70 |
| Jul–Sep gap | −0.30 | −0.32 | −0.53 | −0.46 | −0.51 | −0.15 | +0.11 |
| book | +357 % | +284 % | +254 % | +243 % | +239 % | +279 % | +326 % |

- At no read time is V2 clearly better than V1 (+0.69, +349 %).
- The spread across read times (+0.54 … +0.71) is the same size as the V1-vs-V2 difference. That is the sawtooth problem in numbers.

**Slip at 0.02 % instead of 0.10 %:** gaps are unchanged.

| book | off | V1 @ 1.0 | V2 @ 1.0 | V2c @ 1.0 |
|---|---|---|---|---|
| year | +228 % | **+546 %** | +412 % | +424 % |

**1m coverage for V2/V5 forming bars:** 95 % of the denominator on average. The remainder, mostly the few signals after Sep 25 when the 1m cache ends, uses the 5m bar pro-rated.

## 5. Verdict

**Keep V1.** A switch needed an alternative that beats V1 in both halves, with a CI excluding 0, and that clears the selection null.
- No alternative does: best edge +0.06 (V2c at matched rate), CI [−0.40, +0.55], null p 0.76, 2nd half not better.
- The dollar versions (V3, V5) and the median (V4) are **worse**, with V4 at matched rate and V5c significantly so.

**V2 (dashboard) is statistically equivalent to V1, but every point estimate is lower:**
- gap −0.12;
- book −110 pp;
- worse in Jul–Sep and in the 2nd half;
- its live value depends on the scan's timing, about ±30 % around a bar close.

**Unifying the gate with the dashboard is therefore allowed as a clarity choice, but it is not free.** The expected cost, after haircut, is about 0.06–0.08 %/trade on the gap.
- If the operator wants one number, the cheaper path is to **display V1 on the dashboard as its own "FRENZY Vol"** next to the scan's Vol. That is a UI change and needs the usual D11/D12 wiring.
- If the gate were moved to the dashboard formula, use **V2c**: closed bars, EMA5/SMA48, blacklist-free universe, threshold ≈ 1.0. Do not use the live scan number.

**Not changed by this study:**
- The gate's own weakness in Jul–Sep: every version is ≤ +0.24 in Jul–Sep.
- The pending revert gate (revalidation §0).
- Which definition to use does not decide whether to keep the gate.

## Files (scratch `…/scratchpad/gvolformula/`; none in the repo except this report)

| file | role |
|---|---|
| `yr_arrays.py`, `yr_versions.py` | per-bar arrays (k5m_full) → V1 (= U2), V3, V4, V2c, V5c |
| `v2live.py`, `yr_live.py` | dashboard forming-bar model → V2 / V5 at read times of 15–120 s |
| `val.py`, `valrep.py`, `val_old.pkl`, `val_oct.pkl` | V2 validation vs `entry_global_volume_ratio` stamps (548 + 27 fills) |
| `fetch5.py`, `fetchwin.py` | public 5m (Oct 1 → now) and targeted 1m klines |
| `evalv.py` → `eval_out.md` | all cohort tables, paired CIs, null |
| `extra.py` | read-time sensitivity, disagreement cohort, slip sensitivity |
| `today.py` → `today.csv` | §2 |

Base cohort: `wrev/cohort_plus.pkl` + `flev/strong.pkl` via `gvg/base.py` and `gvg/step2.py` (books).
