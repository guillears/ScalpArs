# FRENZY: "volume rising toward 100× + distance above the average rising" (2026-10-06)

Research only. Nothing in `services/`, `config.py`, `trading_config.json`, `templates/` or `tests/` was touched. Nothing was committed, and the bot's API was not called. **Unreviewed:** no caveman or deep review has run on these numbers, so this report makes no arm recommendation from them (feedback_no_arm_before_review).

## Resumen en lenguaje simple (español)

**Pregunta del operador:** en GRIFFAIN (20:31) y EDU (20:50 y 21:09) el volumen estaba por debajo de 100× pero **subiendo**, y la distancia del precio sobre su promedio (el VWAP del spike que usa FRENZY) también **subía**. ¿Es una señal de entrada que FRENZY se pierde?

1. **Hoy, sólo GRIFFAIN cumplía las dos cosas.**
   - **GRIFFAIN 20:31:** sí. Volumen 60× → 72× en la última media hora (×1,21 en 3 velas). Distancia sobre el VWAP de −2,2 % a +1,2 % (+3,4 puntos en 3 velas).
   - **EDU 20:50:** no. El volumen estaba **bajando** (42× → 28×, ×0,94 en 3 velas, ×0,66 en 6). Sólo la distancia subía, de −3,4 % a +0,1 %.
   - **EDU 21:09:** no. Volumen 24× y bajando (×0,86). La distancia subía poco (+0,6 %).
   - **GRIFFAIN 17:40 (la vela ON del bot):** el volumen se disparó (37× → 155×). La distancia había tocado techo una vela antes (+13,4 %) y ya caía (+5,9 %). Después vino el flash dump.
2. **Probé la idea en el año completo, con las funciones reales de FRENZY.** Fueron 20 variantes, congeladas antes de ver resultados:
   - volumen entre 25× o 50× y 100×;
   - volumen subiendo ×1,2 o ×1,5 en 3 o 6 velas;
   - distancia sobre el VWAP subiendo;
   - una entrada por episodio, del 10-ene al 3-oct.
   - **Ninguna gana dinero.** Las 20 pierden por trade, entre **−0,02 % y −0,38 %**, con 223 a 1.106 entradas cada una. La salida +3/−3 también pierde en las 20.
   - La mejor (volumen 50–100×, ×1,5 en 3 velas) da **−0,02 %/trade**, IC −0,37…+0,35. Pierde en la primera mitad del año (−0,20) y gana en la segunda (+0,15).
   - El listón exige las dos mitades positivas e IC > 0. **No pasa.**
3. **Lo que sí es real (pero no paga):** comparada con una vela al azar del mismo episodio, la señal es mejor, unos +0,3 puntos por trade (p ajustada 0,002 en la mejor). Pero esas velas al azar pierden −0,35 a −0,9 %/trade. "Mejor que el azar" sigue siendo ≈ 0 o negativo. Y es claramente peor que las entradas reales de FRENZY_LONG: +0,43 %/trade con el mismo método de precios.
4. **Por qué parece funcionar a ojo:** cuando el episodio **termina** poniéndose ON, la entrada temprana gana mucho: +0,83 a +0,89 %/trade, y bate a la propia vela ON (−0,2). Pero en el momento de la señal **no se sabe** si va a ponerse ON. Los episodios que nunca llegan a ON pierden −0,9 a −1,8 %/trade, y son la mitad o más de las señales. Eso es mirar el resultado, no una regla.
5. **¿Ayuda "volumen subiendo" a filtrar las propias entradas ON de FRENZY?** No. Separé las ON por volumen subiendo, plano o bajando. El resultado no es monótono: en los trades reales, subiendo da +0,65, plano −0,02 y bajando +0,67. Es ruido.
6. **Veredicto: refutado.** Ni la mejor variante pasa el listón bloqueado. No propongo ni línea de observación: el pre-registro sólo la permitía con las dos mitades positivas, y no las hay. Tus clics de hoy siguen el patrón de los días anteriores: lo que suma es **tu salida y tu timing**, no el estado del volumen.

---

## 1. Today, bar by bar (5m closed bars, real `services.frenzy` walk)

Source: `scratchpad/mvb/pairs.pkl`, the prior agent's public-klines rebuild, through `replay.state_at`, the real `frenzy_walk`, `frenzy_long_status` and `wilder_atr_pct`. The bar label is the bar's **open** (UTC); its close is 5 minutes later. "vm/k" = vol_mult(t) ÷ vol_mult(t−k). "Δd k" = vs_vwap(t) − vs_vwap(t−k), in points. EMA columns are price vs that EMA, in %.

**GRIFFAINUSDT** (spike 14:50, flagged from the 17:35 pass)

| bar | vol× | vm/3 | vm/6 | vm/12 | vs VWAP % | Δd 3 | Δd 6 | streak | state | vs EMA5/8/20 % | RSI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16:55 | 22.1 | 0.95 | 0.83 | 0.56 | +1.70 | +1.57 | +1.40 | 4 | below_avg | +0.9/+1.3/+2.5 | 66 |
| 17:05 | 33.0 | **1.47** | 1.31 | 1.01 | +3.88 | +3.00 | +4.67 | 6 | below_avg (24h vol $16.2M) | +2.0/+2.6/+4.3 | 73 |
| 17:15 | 36.1 | 1.52 | 1.58 | 1.26 | +3.34 | +1.35 | +2.82 | 8 | (24h vol $16.5M) | +0.7/+1.4/+3.2 | 70 |
| 17:25 | 47.6 | 1.36 | 2.16 | 1.79 | +8.66 | +5.43 | +6.96 | 10 | (24h vol $17.3M) | +3.9/+5.1/+7.8 | 81 |
| 17:30 | 105.9 | 2.94 | 4.45 | 4.20 | **+13.38** | +10.04 | +11.38 | 11 | below_avg, 11 of 12 | +8.3/+10.8/+15.3 | 89 |
| **17:35 = ON** | 155.3 | 4.19 | 4.70 | 6.16 | +5.91 (−7.5 vs the bar before) | +1.99 | +2.03 | 12 | **fresh ON, ATR 3.63 % → refused** | +2.2/+4.4/+9.0 | 71 |
| 18:50 | 100.2 | 0.72 | 0.43 | 0.46 | −0.93 | −4.76 | −3.21 | 0 | ON ended | −1.9/−2.3/−1.5 | 49 |
| 19:40 | 79.0 | 1.19 | 0.85 | 0.69 | −7.50 | −8.23 | −8.77 | 0 | dump | −4.5/−5.7/−7.1 | 33 |
| 20:10 | 59.6 | 0.94 | 0.75 | 0.64 | −2.19 | +2.56 | +5.31 | 0 | below | +0.6/+0.6/−0.5 | 48 |
| 20:20 | 67.2 | 1.10 | 1.04 | 0.84 | −0.39 | +1.30 | +4.86 | 0 | below | +1.3/+1.7/+1.2 | 53 |
| **20:25 (click 20:31)** | **72.1** | **1.21** | **1.14** | **1.08** | **+1.17** | **+3.36** | **+5.91** | 1 | below_avg, 1 of 12 | +1.9/+2.5/+2.5 | 57 |
| 20:35 | 67.7 | 1.01 | 1.11 | 1.04 | +2.70 | +3.09 | +4.39 | 3 | | +1.8/+2.6/+3.4 | 61 |
| 20:40 | 51.1 | 0.71 | 0.86 | 0.65 | +3.07 | +1.90 | +5.25 | 4 | (volume already fading) | +1.4/+2.3/+3.4 | 62 |

**EDUUSDT** (spike 17:55, Oct-6 episode)

| bar | vol× | vm/3 | vm/6 | vm/12 | vs VWAP % | Δd 3 | Δd 6 | streak | vs EMA5/8/20 % | RSI |
|---|---|---|---|---|---|---|---|---|---|---|
| 19:00 | 51.3 | 0.86 | 0.88 | 1.12 | +0.21 | +0.75 | +1.01 | 2 | +0.5/+0.7/+1.5 | 59 |
| 19:20 | 42.3 | 0.81 | 0.77 | 0.76 | −3.64 | −2.97 | −2.97 | 0 | −2.6/−2.9/−2.8 | 42 |
| 20:00 | 41.3 | 1.07 | 0.95 | 0.80 | −5.55 | −1.80 | −2.81 | 0 | −1.4/−2.2/−3.7 | 33 |
| 20:15 | 41.6 | 1.01 | 1.08 | 1.15 | −3.18 | +2.37 | +0.58 | 0 | 0.0/−0.1/−1.2 | 45 |
| 20:30 | 29.3 | 0.70 | 0.71 | 0.67 | −2.04 | +1.14 | +3.52 | 0 | +0.6/+0.7/+0.1 | 49 |
| **20:45 (click 20:50)** | **27.6** | **0.94** | **0.66** | **0.72** | **+0.08** | **+2.12** | **+3.26** | 1 | +1.2/+1.7/+1.8 | 58 |
| 20:50 | 28.2 | 1.01 | 0.82 | 0.72 | +0.86 | +2.15 | +4.22 | 2 | +1.4/+1.9/+2.4 | 61 |
| **21:00 (click 21:09)** | **23.6** | **0.86** | **0.80** | **0.57** | **+0.64** | **+0.56** | **+2.67** | 4 | +0.6/+1.1/+1.8 | 59 |

**Was each click "volume rising and distance rising"?**

| click | volume multiple rising? | distance above VWAP rising? | verdict |
|---|---|---|---|
| GRIFFAIN 20:31 | **Yes.** 59.6 → 72.1× over 3 bars (×1.21; ×1.14 over 6). It peaked on this bar and fell to 51× by 20:40. | **Yes.** −2.2 → +1.2 % (+3.4 pts over 3, +5.9 over 6) | **Both rising: confirmed** |
| EDU 20:50 | **No.** It fell, 41.6 → 27.6× over 6 bars (×0.66); ×0.94 over 3 | **Yes.** −3.2 → +0.1 % (+2.1 / +3.3 pts) | Only the distance was rising |
| EDU 21:09 | **No.** 23.6×, ×0.86 / ×0.80, and under the L = 25 floor | Weakly (+0.6 pt over 3, +2.7 over 6) | Only the distance was rising |
| GRIFFAIN 17:40 ON bar | **Yes, explosively.** 37 → 155× (×4.2 over 3) | Over 3 bars yes (+2.0). On the last bar **no**: it peaked one bar earlier at +13.4 % and the ON close was +5.9 % | A flash dump followed (−5.8 % at +20 s, prior report) |

GRIFFAIN 17:05–17:25 is the textbook case of the hypothesis. Volume went 33 → 48×, rising at ×1.4–2.2. Distance went +3.9 → +8.7 %, also rising. The price then ran +23 % in the next hour. **But the bot could not see it:** GRIFFAIN's 24 h volume was $16–17M (< $20M). So the pair was not on the FRENZY shortlist, and it is not in the pre-registered universe. It is an anecdote (N = 1), not evidence. The rule's first trigger in the frozen family on today's GRIFFAIN episode is the **20:25 bar**, which is the operator's click. On 1m klines with the live lock, that entry gives +1.98 %. EDU fires **no** cell today.

## 2. Pre-registration (frozen 2026-10-06 21:40:19 UTC, before any year outcome)

Verbatim in `scratchpad/risingvol/PREREG.txt`. Summary:
- **Base, every cell:**
  - the real `frenzy_walk` on the engine window (same construction as `scripts/frenzy_engine_cohort_build.py`);
  - `frenzy_flagged`, hours ≥ 2, **not in state**, vs_vwap > 0;
  - 24 h vol ≥ $20M, shortlisted at least once since the spike;
  - live universe: ASCII, not Alpha, not TradFi, listed ≥ 90 d, not blacklisted;
  - signal window Jan-10 → Oct-3 2026.
- **20 cells:**
  - L ∈ {25, 50} (vol_mult in [L, 100)) × k ∈ {3, 6} × volume ratio r ∈ {1.2, 1.5} × streak {any, 1–6}, with distance rising (> 0 pt over the same k) = 16 cells;
  - plus 4 cells at L25 / streak any with distance rising ≥ +1.0 pt.
- **One entry per episode** (the first trigger).
- **Entry:** next 5m open + 0.02 % slip. Fees 0.09 %.
- **Exits:**
  - **primary:** the live lock (−3 stop; at +3 the line = max(+2, peak − 2); 720 min), on public 1m klines with pessimistic intrabar ordering;
  - **secondary:** fixed +3 / −3.
- **Statistics:** N, WR, mean, day-block bootstrap 95 % CI, Jan–Apr / May–Oct halves, concentration.
- **Null:**
  - 1,000 draws, each replacing every cell's trigger with a random base-eligible bar **of the same episode**;
  - p_adj = the share of draws whose **best** cell ≥ the real cell.
- **Pass:** both halves > 0 ∧ day-CI > 0 ∧ ≥ 30 fills on ≥ 8 days ∧ no pair or day ≥ 50 % of the gain ∧ p_adj < 0.05, then a 30–50 % haircut.
- **Borderline** (both halves > 0, fails only CI or p) → a free scout observe line at most.

## 3. Engine-cohort parity

| check | result |
|---|---|
| Walk / flag / state | `scan.py` copies the cohort builder's window, normal-hour and shortlist code, and calls the real `frenzy_walk` + `frenzy_flagged` on every bar with 25× ≤ vol < 100× and 24 h vol ≥ $20M (110,829 flagged, not-in-state, above-VWAP bars; 76,108 in the live universe, 1,348 episodes, 345 pairs, 267 days). |
| Volume multiple | Array recompute (rolling-hour quote vol ÷ normal hour) vs the walk's `vol_mult`: max relative difference **8.5e-14**. On the 1,819 cohort ON bars: **2.2e-14**. |
| Anchored VWAP (for the lagged distance) | Cumulative recompute from the walk's `spike_ts` vs the walk's `vwap`: max relative difference **2.6e-11** on all 110,829 bars. So vs_vwap(t − k) is the walk's own line. |
| Live events today | The `mvb` replay (real functions) reproduces the GRIFFAIN 17:35 fresh-ON bar, **ATR 3.63 %** ("ATR 3.63% > 2.5%" in the log). The 24 h vol crosses $20M on the 17:30 → 17:35 bars (the flag first appears at the 17:35 pass). It also reproduces the EDU 17:55 re-flag. ORCA 18:05 gvol 0.7802 was matched by the prior agent; **I did not recompute gvol.** The gvol split below uses `gvr_year.pkl`, a different (year) definition, as a control only. |
| Live FRENZY fills | Not re-run. The engine cohort's 8/8 live-fill parity (FRENZY_GVOL_GATE_REVALIDATION / ON-scalp reports) carries over, because the ON comparator below is that cohort. |
| Data limit | `k5m_full` ends 2026-10-04 00:00 UTC. Today's events are described from the `mvb` rebuild, not from the year set. |
| Pricer calibration | My 1m pricer (entry at the next 5m open) vs the cohort's tick LOCK2 (entry at ON close + 12 s) on the same ON bars: **all fresh-ON −0.021 (1m) vs −0.192 (tick)**; FRENZY_LONG trades +0.434 vs +0.387. **The 1m pricer is the kinder one**, so the negative cell results below are, if anything, flattered. |

## 4. Year results (live lock, 1m, % of position at 1×)

| cell | N | days | WR | mean %/trade | day-CI 95 % | Jan–Apr | May–Oct | +3/−3 mean | null mean (same episodes) | p vs own null | p_adj (best-of-family) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L25 k3 r1.2 stany d>0.0 | 1040 | 262 | 46 % | -0.184 | -0.37 … +0.01 | -0.380 | -0.032 | -0.265 | -0.493 | 0.000 | 0.199 |
| L25 k3 r1.2 st1-6 d>0.0 | 665 | 248 | 44 % | -0.258 | -0.49 … -0.02 | -0.342 | -0.187 | -0.359 | -0.630 | 0.001 | 0.543 |
| L25 k3 r1.5 stany d>0.0 | 730 | 249 | 43 % | -0.194 | -0.44 … +0.06 | -0.322 | -0.085 | -0.406 | -0.408 | 0.023 | 0.227 |
| L25 k3 r1.5 st1-6 d>0.0 | 396 | 208 | 41 % | -0.376 | -0.71 … -0.02 | -0.542 | -0.228 | -0.572 | -0.770 | 0.001 | 0.979 |
| L25 k6 r1.2 stany d>0.0 | 1106 | 263 | 46 % | -0.197 | -0.39 … -0.01 | -0.461 | +0.009 | -0.233 | -0.599 | 0.000 | 0.240 |
| L25 k6 r1.2 st1-6 d>0.0 | 733 | 251 | 46 % | -0.213 | -0.45 … +0.03 | -0.393 | -0.069 | -0.244 | -0.710 | 0.000 | 0.309 |
| L25 k6 r1.5 stany d>0.0 | 950 | 258 | 45 % | -0.205 | -0.42 … +0.02 | -0.456 | +0.000 | -0.314 | -0.535 | 0.000 | 0.273 |
| L25 k6 r1.5 st1-6 d>0.0 | 531 | 231 | 45 % | -0.212 | -0.50 … +0.08 | -0.307 | -0.130 | -0.333 | -0.763 | 0.000 | 0.302 |
| L50 k3 r1.2 stany d>0.0 | 735 | 248 | 47 % | -0.113 | -0.37 … +0.14 | -0.107 | -0.119 | -0.211 | -0.349 | 0.008 | 0.046 |
| L50 k3 r1.2 st1-6 d>0.0 | 377 | 196 | 45 % | -0.233 | -0.55 … +0.09 | -0.358 | -0.111 | -0.305 | -0.640 | 0.004 | 0.419 |
| **L50 k3 r1.5 stany d>0.0 (best)** | **481** | **224** | **46 %** | **-0.023** | **-0.37 … +0.35** | **-0.198** | **+0.150** | -0.242 | -0.363 | 0.001 | **0.002** |
| L50 k3 r1.5 st1-6 d>0.0 | 223 | 152 | 43 % | -0.308 | -0.75 … +0.15 | -0.537 | -0.072 | -0.455 | -0.920 | 0.000 | 0.808 |
| L50 k6 r1.2 stany d>0.0 | 796 | 255 | 47 % | -0.136 | -0.37 … +0.10 | -0.302 | +0.006 | -0.192 | -0.469 | 0.000 | 0.075 |
| L50 k6 r1.2 st1-6 d>0.0 | 423 | 207 | 47 % | -0.176 | -0.48 … +0.13 | -0.357 | -0.016 | -0.216 | -0.734 | 0.000 | 0.178 |
| L50 k6 r1.5 stany d>0.0 | 675 | 246 | 45 % | -0.208 | -0.47 … +0.06 | -0.327 | -0.100 | -0.282 | -0.437 | 0.026 | 0.288 |
| L50 k6 r1.5 st1-6 d>0.0 | 295 | 173 | 45 % | -0.225 | -0.60 … +0.16 | -0.349 | -0.096 | -0.306 | -0.914 | 0.000 | 0.370 |
| L25 k3 r1.2 stany d>1.0 | 1005 | 261 | 45 % | -0.247 | -0.43 … -0.06 | -0.435 | -0.102 | -0.295 | -0.446 | 0.020 | 0.473 |
| L25 k3 r1.5 stany d>1.0 | 708 | 248 | 43 % | -0.213 | -0.48 … +0.05 | -0.385 | -0.069 | -0.435 | -0.382 | 0.070 | 0.309 |
| L25 k6 r1.2 stany d>1.0 | 1079 | 263 | 46 % | -0.200 | -0.38 … -0.01 | -0.389 | -0.054 | -0.236 | -0.571 | 0.000 | 0.245 |
| L25 k6 r1.5 stany d>1.0 | 925 | 255 | 44 % | -0.278 | -0.50 … -0.05 | -0.450 | -0.137 | -0.358 | -0.519 | 0.005 | 0.657 |

- **0 of 20 cells have a positive mean.** 0 of 20 are positive in both halves. 6 of 20 have the whole day-CI below 0. The secondary +3/−3 exit is negative in all 20 cells.
- **Concentration:** not applicable. No cell has a positive total, so there is no gain to concentrate.
- **Haircut:** not applicable (nothing passed).
- **Null:**
  - Best-of-family on random same-episode bars: 50th / 95th / 99th pct = −0.25 / −0.12 / −0.04 %/trade.
  - The best cell (−0.023) beats it (p_adj 0.002). So the rising-volume timing **is better than a random above-VWAP bar of the same episode, by about +0.3 points**. But the random bars lose −0.35 to −0.9 %/trade, so "better than random" still lands at ≈ 0 or below.
  - Caveat: the null bar is uniform over the episode, while the trigger is the first, earliest bar. Part of the gap is "early in the episode" rather than "rising".
- **Bar-weighted base (all eligible bars, not one per episode):** L25 −0.017 (H1 +0.067, H2 −0.086), L50 −0.028. Long-lived above-VWAP episodes get more weight there. Even that pool is flat.

## 5. Comparators

| set (live universe, Jan-10 → Oct-3) | N | WR | mean (1m, same pricer) | tick LOCK2 (cohort) | Jan–Apr / May–Oct (1m) |
|---|---|---|---|---|---|
| Best RVP cell (L50 k3 r1.5) | 481 | 46 % | −0.023 | — | −0.198 / +0.150 |
| All fresh-ON bars (any refusal code) | 1,443 | 48 % | −0.021 | −0.192 | −0.048 / +0.002 |
| **FRENZY_LONG trades (READY ∧ gvol < 1)** | 210 | 53 % | **+0.434** (CI −0.03 … +0.90) | +0.387 | +0.286 / +0.591 |
| Random same-episode bars (null mean, best cell's episodes) | — | — | −0.363 | — | — |

The best rising-volume cell is no better than buying every fresh-ON bar, and about 0.45 points/trade worse than what FRENZY_LONG actually trades (READY + gvol gate).

## 6. Does "volume rising" add anything to FRENZY's own ON entries?

The split is by vol_mult(ON bar) ÷ vol_mult(k bars earlier), using the cohort's tick LOCK2.

| set | split | N | mean (tick) | mean (1m) |
|---|---|---|---|---|
| All ON | k3 ≥ 1.2 | 613 | +0.024 | +0.170 |
| | k3 1.0–1.2 | 496 | −0.432 | −0.182 |
| | k3 < 1 (falling) | 231 | −0.247 | −0.213 |
| | k6 ≥ 1.2 | 893 | −0.143 | +0.027 |
| | k6 1.0–1.2 | 227 | −0.078 | −0.017 |
| | k6 < 1 | 220 | −0.505 | −0.230 |
| **FRENZY_LONG trades** | k3 ≥ 1.2 | 87 | +0.646 (CI −0.18 … +1.43) | +1.003 |
| | k3 1.0–1.2 | 81 | −0.019 | +0.017 |
| | k3 < 1 | 37 | **+0.671** | −0.011 |
| | k6 ≥ 1.2 | 138 | +0.333 | +0.595 |
| | k6 1.0–1.2 | 36 | **+0.829** | +0.318 |
| | k6 < 1 | 31 | +0.116 | −0.156 |

- **The response is non-monotone** on the traded population. On tick, rising ≈ falling at k3, and the middle bucket is best at k6. Every CI spans 0.
- Under the locked rules, a non-monotone single-variable pattern is a confound, not a filter. "Volume rising" adds nothing reliable on top of the ON entry.
- On all-ON bars, "falling volume" is the worst bucket. That bucket is mostly refused bars (ATR / green), so it is not a live lever.

## 7. Overlap: does the signal get into the same move earlier?

| cell | signal position in its episode | N | WR | mean (lock) | median lead to ON | median price move entry → ON close |
|---|---|---|---|---|---|---|
| L50 k3 r1.5 (best) | **pre-ON** (episode turns ON later) | 236 | 56 % | **+0.88** | 35 min | +2.5 % |
| | between two ON stretches | 93 | 57 % | +0.41 | 45 min | +2.9 % |
| | after the last ON | 27 | 30 % | −1.36 | — | — |
| | **never turns ON** | 125 | 23 % | **−1.76** | — | — |
| L25 k3 r1.2 | pre-ON | 334 | 58 % | +0.83 | 83 min | +6.1 % |
| | never ON | 523 | 37 % | −0.90 | — | — |
| L25 k6 r1.2 | pre-ON | 338 | 60 % | +0.89 | 100 min | +6.2 % |
| | never ON | 592 | 38 % | −0.86 | — | — |

- On the pre-ON episodes, the early signal beats the same episode's later ON bar by a wide margin: **+0.83…+0.89 vs −0.21…−0.28** (1m lock). It gets in 35–100 min earlier, 2.5–6 % lower.
- **This is look-ahead.** "Turns ON later" requires the price to stay above the VWAP for an hour with volume ≥ 100×. That is the outcome the signal is trying to predict.
- At signal time the signal cannot tell a pre-ON episode from a never-ON one. 26–54 % of signals are never-ON or after-ON, and they lose −0.9 to −1.8 %/trade. They cancel the pre-ON gains.
- An ex-ante separator for the two groups would be a new hypothesis outside this frozen family. It was not tested here, and testing it on this data would be fitting on the result.

## 8. Controls (reported, not used to choose)

| control | best cell (L50 k3 r1.5) | L25 k3 r1.2 | L25 k6 r1.2 |
|---|---|---|---|
| ATR ≤ 2.5 % / > 2.5 % | −0.10 (397) / +0.35 (84) | −0.13 / −0.52 | −0.21 / −0.15 |
| gvr < 1 / ≥ 1 (year gvr, not the live U2) | −0.06 / −0.02 | −0.15 / −0.27 | −0.28 / −0.12 |
| vol band 25–50 / 50–75 / 75–100 | — / −0.11 / +0.23 | −0.16 / −0.37 / −0.03 | −0.21 / −0.27 / −0.03 |
| month (best cell) | Jan +0.49, Feb −0.04, Mar −0.83, Apr −0.23, May −0.10, Jun +0.35, Jul −0.21, Aug −0.17, Sep +0.81 | | |

- No control flips the family positive in a consistent direction. ATR is non-monotone across cells.
- The 75–100× band is the least bad (≈ 0). That is the band closest to FRENZY's own 100× line, where the episode is about to be ON anyway.

## 9. Verdict (locked bars)

| criterion | best cell L50 k3 r1.5 | pass? |
|---|---|---|
| mean > 0 in both halves | −0.198 / +0.150 | ✗ |
| day-CI lower > 0 | −0.37 | ✗ |
| ≥ 30 fills on ≥ 8 days | 481 on 224 | ✓ |
| no pair or day ≥ 50 % of the gain | total < 0 | n/a |
| beats the selection-adjusted null | p_adj 0.002 | ✓ (relative only) |

- **FAIL. "Volume rising toward 100× + distance above the spike VWAP rising" is refuted as an entry** on the year, the live universe, the engine's own walk and the live lock exit. The best of 20 frozen cells is break-even, and the family average is about −0.2 %/trade. A borderline observe line is not warranted, because no cell is positive in both halves (the pre-registered condition).
- **What is true:**
  - the timing is better than a random bar of the same episode;
  - when an episode later turns ON, an early entry beats FRENZY's ON entry.
  - Neither is tradeable without knowing the future. The never-ON episodes eat the gain.
- **Today:**
  - GRIFFAIN 20:31 fits the hypothesis (both rising), and the rule would have fired on that bar (+1.98 on the lock).
  - EDU 20:50 and 21:09 do **not**: volume was falling.
  - So the operator's three clicks are not one "rising volume" pattern. As the earlier report found, the edge is in the hand exits and the timing.
- **Not tested (blind spots):**
  - pairs under the $20M / shortlist gate (GRIFFAIN 17:05–17:25 lived there; the bot cannot see them without a shortlist change);
  - tick-level entry (the 1m pricer was shown to be kinder than ticks on ON bars);
  - an ex-ante separator of pre-ON vs never-ON episodes (would be a new pre-registered study);
  - the live gvol U2 gate on these signals.

**Scratch:** `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/risingvol/`:
- `PREREG.txt` and `PREREG.stamp`
- `today.py` and `today_*.csv`
- `today_cells.py`
- `scan.py`, with `sc/` holding the year walk shards and `bars_all.pkl`
- `fetch1m.py`, with `m1/` holding the public 1m klines for 354 pairs
- `pricer.py`
- `analyze.py`, `cells.csv`, `trig.pkl` and `nullmeans.npy`
- `on.py` and `on_live.pkl`
- `overlap.py`
