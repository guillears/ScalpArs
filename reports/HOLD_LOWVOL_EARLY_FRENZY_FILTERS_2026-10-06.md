# HOLD_LOWVOL_EARLY: los filtros vivos de FRENZY_LONG aplicados a sus entradas (2026-10-06)

## Resumen en lenguaje simple

**Veredicto: CERRADO. Ningún filtro pasa la barra pre-declarada.**

- **Qué se probó.** Se tomaron las 2.236 entradas congeladas de HOLD_LOWVOL_EARLY (las que tienen ticks) y se les aplicaron los filtros que FRENZY_LONG usa hoy en vivo, con las mismas definiciones del motor y los mismos umbrales de `trading_config.json`: tope de ATR 2,5 %, saltear la vela verde, volumen de mercado < 1,0×, precio que se movió > 1 % al momento de entrar y máximo 3 entradas por par por día. Además se probó la regla "hold-green" de WIDE y, como corte informativo (no como filtro), el "strong" (ADX subiendo y +DI > −DI). Son 13 lecturas, todas registradas antes de mirar resultados y sin ajustar ningún umbral.
- **Resultado con timing real** (entrada 12 s tarde, slippage 0,10 %, comisiones 0,09 %). El mejor conjunto es el stack de FRENZY con la regla de WIDE en lugar de "saltear verde": 728 entradas, **+0,21 % por trade**, pero el intervalo de confianza por días va de **−0,05 a +0,47** (cruza cero). Además Ene–Abr da solo +0,06. El stack exacto de FRENZY (F1–F5) da +0,19 %, IC [−0,10, +0,46]. Ninguna lectura tiene el límite inferior del IC por encima de cero.
- **¿Los filtros ayudan algo?** Sí, un poco: el stack sube la media de +0,04 a +0,19/+0,21 por trade. Pero comparado con etiquetas de filtro mezcladas al azar dentro de cada mes, la mejora del stack no supera al azar con el rigor necesario: p = 0,11 para F1–F5 y p = 0,044 para el stack con WIDE, cuando 13 lecturas exigen p < 0,004. **Con el ajuste de 30–50 % por sesgo in-sample, el mejor queda en +0,11 a +0,15 por trade, con un IC que ya cruzaba cero.**
- **Qué bloquean los filtros.** Ningún grupo bloqueado cumple la barra de expectativa (pérdida con 95 % de confianza). El que más se acerca es el de ATR alto: 339 trades, −0,20 % por trade y 87 % de probabilidad de media negativa, que no llega al 95 %. Saltear la vela verde **no hace nada** en este universo: los verdes bloqueados ganan +0,045, lo mismo que los rojos que se quedan.
- **Conclusión.** Igual que en el estudio principal, cualquier ventaja que haya parece venir del *estado* del mercado y no de la selección de entradas. Los filtros de FRENZY no la convierten en una línea demostrable. HOLD_LOWVOL_EARLY queda cerrado por el lado de los filtros. No se propone armar nada.
- **Lectura secundaria con salida.** El informe de salidas ronda 2 (`HOLD_LOWVOL_EARLY_EXITS_ROUND2_2026-10-06.md`) todavía no tenía resultados al cerrar este informe (solo el pre-registro). No hay salida aprobada para combinar, así que esa lectura no se hizo.

---

## Pre-registration (verbatim, `scratchpad/holdlow_filters/PREREG_FILTERS.txt`, written 2026-10-06T23:18:22Z before any split was computed)

```
Universe: frozen FREEZE_HOLD_LOWVOL_EARLY_fills.csv (2239; 2236 tick-covered). Exit: frozen FRENZY LOCK (arm 3 / floor 2 / trail 2 / stop -3, 12 h cap).
PRIMARY = live timing (entry first print >= close+12 s, no entry slip, exit -0.10, fees 0.09). SECONDARY = next open +0.02, exit -0.02.
 F1 ATR: services.surge.wilder_atr_pct(last 300 closed 5m bars ending at the signal bar) <= 2.5 (None -> block).
 F2 green skip: block when signal close > open (engine bar_red = close <= open).
 F3 gvol: engine V1 (global_volume_ratio, top-50 by 24h, lookback 48) on the signal bar < 1.0; unreadable -> block.
 F4 dislocation: block if |first print at close+12 s / last print before close - 1| > 1.0 %.
 F5 pair-day cap 3: per pair per UTC day at most 3 KEPT entries (alone: over all fills; in the stack: over fills passing the other filters,
    as the engine counts opened orders).
 F6 WIDE hold-green: block a green signal bar unless above_streak > 12. Alone, and as F2's replacement in the stack (FRENZY+WIDE union).
Family: F1..F6 alone (6) · STACK F1..F5 (1) · leave-one-out of STACK (5) · STACK_WIDE (13th read).
Split (not a filter): strong = frenzy_adx_delta > 0 and frenzy_di_spread > 0 on the same 300 bars.
VERDICT: kept cohort on PRIMARY >= +0.10 %/trade, day-CI lo > 0, May-Oct > 0, >= 30 fills on >= 8 days, top pair/day share < 50 %,
shuffled-null p < 0.05 (robust only if p < 0.05/13). Otherwise close. No arm proposal.
```

## Engine parity of the filter inputs
- **ATR:** the engine's `wilder_atr_pct` runs on the last 300 closed 5m bars (k5m_full). It matches the frozen `atr5` column exactly (corr 1.000, median |Δ| 0.0000).
- **Green bar:** the engine's `bar_red` is close ≤ open on the signal bar.
- **Market volume (gvol):** `manualall/gvol_year.pkl` `gvol_engine` is the engine's V1. It is identical to the validated `gvolformula/yr_versions.pkl` V1 on all 1,801 shared bars (max |Δ| 1e-6, block side 100 % equal). Coverage on the fills is 100 %.
- **Dislocation:** computed on ticks at the live entry time. 38 fills are > 1 %, the same 38 the holdlow_ticks live ruler had dropped. They were re-priced with the guard off, so F4 can be read as a label.
- **Strong split:** `frenzy_adx_delta` and `frenzy_di_spread` run on the same 300 bars.

## Universe (LOCK exit)
| ruler | N | WR | avg %/trade | day CI | Jan–Apr | May–Oct | breakeven WR |
|---|---|---|---|---|---|---|---|
| PRIMARY (live 12 s / 0.10) | 2,236 | 52.6 % | +0.043 | −0.128 … +0.228 | −0.031 | +0.094 | 51.9 % |
| SECONDARY (next open) | 2,236 | 52.2 % | +0.115 | −0.061 … +0.300 | +0.051 | +0.159 | 50.3 % |

## Kept cohorts: PRIMARY ruler (the verdict ruler)
| read | N | days | keep | WR | avg | day CI | Jan–Apr | May–Oct | +months | top pair / day share | Δ vs all | shuffled p | haircut 50/70 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 ATR ≤ 2.5 | 1,897 | 265 | 85 % | 53.1 | +0.087 | −0.102 … +0.276 | +0.030 | +0.127 | 7/10 | 2 % / 5 % | +0.044 | 0.070 | +0.04 / +0.06 |
| F2 skip green | 1,144 | 261 | 51 % | 53.3 | +0.041 | −0.157 … +0.259 | −0.017 | +0.084 | 5/10 | 3 % / 4 % | −0.002 | 0.516 | +0.02 / +0.03 |
| F3 gvol < 1 | 1,469 | 265 | 66 % | 53.6 | +0.083 | −0.106 … +0.264 | −0.109 | +0.211 | 8/10 | 2 % / 3 % | +0.040 | 0.216 | +0.04 / +0.06 |
| F4 disloc ≤ 1 % | 2,198 | 267 | 98 % | 52.6 | +0.045 | −0.129 … +0.229 | −0.024 | +0.093 | 5/10 | 2 % / 5 % | +0.002 | 0.414 | +0.02 / +0.03 |
| F5 pair-day ≤ 3 | 2,188 | 267 | 98 % | 52.7 | +0.052 | −0.117 … +0.240 | −0.001 | +0.089 | 6/10 | 2 % / 5 % | +0.009 | 0.178 | +0.03 / +0.04 |
| F6 WIDE hold-green | 1,382 | 265 | 62 % | 53.0 | +0.045 | −0.143 … +0.253 | −0.036 | +0.105 | 4/10 | 2 % / 3 % | +0.002 | 0.472 | +0.02 / +0.03 |
| **STACK F1–F5 (live FRENZY)** | 615 | 229 | 28 % | 55.9 | **+0.191** | **−0.102 … +0.464** | +0.200 | +0.184 | 7/10 | 3 % / 5 % | +0.148 | 0.106 | +0.10 / +0.13 |
| STACK − F1 | 724 | 244 | 32 % | 55.7 | +0.163 | −0.107 … +0.421 | +0.156 | +0.168 | 7/10 | 3 % / 4 % | +0.120 | 0.126 | |
| STACK − F2 | 1,228 | 259 | 55 % | 54.3 | +0.131 | −0.083 … +0.349 | −0.025 | +0.238 | 8/10 | 2 % / 4 % | +0.088 | 0.090 | |
| STACK − F3 | 954 | 252 | 43 % | 53.8 | +0.087 | −0.134 … +0.329 | −0.017 | +0.164 | 5/10 | 3 % / 5 % | +0.044 | 0.289 | |
| STACK − F4 | 616 | 229 | 28 % | 55.8 | +0.185 | −0.108 … +0.457 | +0.188 | +0.184 | 7/10 | 3 % / 5 % | +0.142 | 0.113 | |
| STACK − F5 | 616 | 229 | 28 % | 55.8 | +0.185 | −0.109 … +0.463 | +0.200 | +0.174 | 7/10 | 3 % / 5 % | +0.142 | 0.114 | |
| **STACK_WIDE (F6 for F2)** | 728 | 243 | 33 % | 55.5 | **+0.211** | **−0.046 … +0.473** | +0.055 | +0.333 | 8/10 | 2 % / 4 % | +0.168 | 0.044 | +0.11 / +0.15 |

**PASS reads: none.** Every read fails on day-CI lo > 0. The best read, STACK_WIDE, also misses the family-adjusted null (0.044 > 0.0038), and its Jan–Apr half is near zero (+0.055).

## Kept cohorts: SECONDARY ruler (next open +0.02)
| read | N | avg | day CI | Jan–Apr | May–Oct | shuffled p |
|---|---|---|---|---|---|---|
| F1 | 1,897 | +0.166 | −0.026 … +0.364 | +0.107 | +0.208 | 0.042 |
| F2 | 1,144 | +0.131 | −0.065 … +0.345 | +0.118 | +0.141 | 0.397 |
| F3 | 1,469 | +0.178 | −0.013 … +0.361 | −0.006 | +0.301 | 0.108 |
| F4 | 2,198 | +0.123 | −0.055 … +0.311 | +0.075 | +0.156 | 0.190 |
| F5 | 2,188 | +0.121 | −0.057 … +0.308 | +0.089 | +0.144 | 0.264 |
| F6 | 1,382 | +0.154 | −0.033 … +0.362 | +0.079 | +0.209 | 0.202 |
| STACK F1–F5 | 615 | +0.297 | +0.010 … +0.573 | +0.331 | +0.273 | 0.060 |
| STACK_WIDE | 728 | +0.338 | +0.077 … +0.593 | +0.185 | +0.458 | 0.012 |

On the next-open ruler, two stack reads clear CI > 0. That ruler buys 12 s earlier with 0.08 % less exit slippage than live can. The ~0.1 %/trade gap between the two rulers costs the same on every read, and it is what pushes all of them below zero on the CI. Even here, no p-value clears 0.0038.

## Blocked cohorts vs the expectancy bar (primary; sleeve breakeven WR 51.9 %)
| read | blocked N | days | WR | avg | day CI | P(avg<0), day-clustered | worst day / pair share of loss | bar |
|---|---|---|---|---|---|---|---|---|
| F1 ATR > 2.5 | 339 | 166 | 50.1 | −0.202 | −0.571 … +0.163 | 0.87 | 15 % / 28 % | fail (confidence) |
| F2 green | 1,092 | 257 | 51.9 | +0.045 | −0.189 … +0.317 | 0.39 | n/a (net +) | fail |
| F3 gvol ≥ 1 | 767 | 243 | 50.7 | −0.033 | −0.312 … +0.246 | 0.60 | 87 % / 78 % | fail |
| F4 disloc | 38 | 35 | 52.6 | −0.091 | −0.97 … +0.85 | 0.59 | 91 % / >100 % | fail |
| F5 4th+ entry | 48 | 36 | 47.9 | −0.384 | −1.11 … +0.34 | 0.86 | 26 % / 26 % | fail (confidence) |
| F6 green reclaim | 854 | 241 | 52.1 | +0.040 | −0.248 … +0.340 | 0.42 | n/a (net +) | fail |
| STACK blocked | 1,621 | 265 | 51.4 | −0.013 | −0.204 … +0.173 | 0.56 | >100 % | fail |
| STACK_WIDE blocked | 1,508 | 264 | 51.3 | −0.038 | −0.223 … +0.155 | 0.66 | 40 % / 50 % | fail |

No blocked cohort reaches 95 % confidence of a negative average. Two are directionally negative: ATR > 2.5 at −0.20 (both halves negative) and the 4th+ entry of the pair-day at −0.38 (N 48, H2 positive).

## Splits and capacity (STACK F1–F5, primary)
- **By month:** Jan +0.34 · Feb +0.55 · Mar +0.09 · Apr −0.17 · May +0.43 · Jun −0.22 · Jul −0.18 · Aug +0.72 · Sep +0.03 · Oct (6 fills) +0.06.
- **Strong (ADX Δ>0 ∧ +DI>−DI):** N 315, +0.274, CI −0.11 … +0.66, H1 +0.24, H2 +0.30.
- **Not strong:** N 300, +0.103, CI −0.26 … +0.46.
- The strong/not-strong difference is in the same direction as FRENZY's sizing, but both CIs straddle zero. It is informational only.
- **5-position cap** (chronological; slot held until the secondary-ruler exit time):
  - STACK: N 610, +0.171, CI −0.116 … +0.435.
  - STACK_WIDE: N 723, +0.195, CI −0.056 … +0.445.
  - The cap barely binds, because these setups rarely overlap.

## Verdict
- **CLOSED.** No pre-registered read meets the free-scout bar on the live-timing ruler.
- The best read is STACK_WIDE at +0.211 %/trade, which falls to +0.11 … +0.15 after the 30–50 % in-sample haircut. It fails on the day-CI (−0.046), on the family-adjusted shuffled null, and on Jan–Apr (≈ 0).
- The live FRENZY filters do lift the average from +0.04 to about +0.19 by removing high-ATR and high-market-volume entries. Against shuffled labels, that lift is not distinguishable from luck at family scale.
- This agrees with the primary study: whatever edge exists is a state effect and is not separable by these entry filters.
- **Secondary exit read: not run.** `reports/HOLD_LOWVOL_EARLY_EXITS_ROUND2_2026-10-06.md` held only its pre-registration when this report closed. No exit had passed.
- No arm proposal.

## Files (scratch: `scratchpad/holdlow_filters/`)
`PREREG_FILTERS.txt` · `features.py` (engine-exact ATR / bar_red / gvol / DI / ADX-Δ) · `features.pkl` · `price_disloc.py` + `live_all.pkl` (live ruler with the guard as a label) · `analyze.py` · `analyze.out` · `kept.csv` · `blocked.csv` · `A.pkl` (per-fill table).
