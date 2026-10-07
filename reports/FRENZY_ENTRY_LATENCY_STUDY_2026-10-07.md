# FRENZY: ¿vale la pena entrar más rápido? (2026-10-07)

Investigación nada más. No se tocó código, config, tests ni templates, y no hubo commits. Es un **backtest sin revisar**: según la regla de no armar nada antes de la revisión, no respalda ningún cambio hasta que pase por la caveman review y la deep review.

## Resumen en lenguaje simple

**Veredicto: no conviene invertir ingeniería en entrar más rápido. La velocidad actual ya alcanza.**

- **Qué se probó.** Se tomaron las mismas 332 señales de FRENZY_LONG + WIDE hold-green del año, con los filtros que el bot usa hoy. Cada una se re-preció con ticks reales, entrando 0, 1, 2 … 15, 20 y 30 segundos después del cierre de la vela. La salida fue siempre la misma (el lock en vivo).
- **Resultado.** Entre ~1 y ~13 s la ganancia por trade es **plana**: entre +0,32 y +0,43 % por trade, y las diferencias son ruido.
  - Entrar justo al cierre (0 s) es un poco *peor* (+0,29).
  - En el universo grande (1.755 señales, sin filtros) la curva es plana de 0 a 30 s.
  - La regla pre-registrada pedía que "cada segundo ahorrado" valiera algo con IC > 0. Dio **−0,015 % por segundo, IC [−0,031, +0,001]**. Si algo, ir más rápido sale algo peor, pero tampoco eso es distinguible del azar. **No pasa.**
- **El precio no "se escapa" en esos segundos.** A los 12 s el precio está en promedio −0,03 % *por debajo* del cierre, no por encima. Lo que cambia entre un retardo y otro no es el precio de entrada. Es que la salida (stop −3 / armado +3) cae de un lado u otro por centésimas, y eso es azar.
- **¿Qué tan rápido entra hoy el bot?** Medido en los logs y en los fills reales:
  - Desde el speed-up del 5-oct (835620d) la orden sale **~7,5–8 s** después del cierre.
  - Antes eran 11–12 s, y el 3-oct por el camino viejo del scan, 79 s.
  - Solo 1,6 % de las pasadas terminan a ≥ 14 s.
  - Los 8 s de hoy ya están en la zona plana.
- **La brecha de HOLD_LOWVOL (+0,30 "al cierre" vs +0,19 "en vivo") NO es la velocidad.**
  - ~75 % de esa brecha es la **hipótesis de slippage de salida**: 0,02 contra 0,10.
  - El resto es ruido del retardo: entrar a 5–9 s rinde lo mismo que a 0 s.
  - Además, los stops reales en paper muestran una salida ~0,18 % peor que el último print (9 stops). La hipótesis de 0,10 no es pesimista; la de 0,02 es optimista.
  - **HOLD_LOWVOL sigue cerrado:** ningún retardo alcanzable lo vuelve viable de forma robusta.
- **Qué construir.** Nada por velocidad.
  - Opcional, de muy bajo valor: darle prioridad a FRENZY en la cola de órdenes del bot, para que una apertura del scan no la demore a > 14 s. Eso pasa en ~1,6 % de las barras.
  - Sí conviene actualizar el "ruler en vivo" de los backtests de 12 s a **8 s** (lo medido). Esto casi no cambia los números.

---

## 0. Freeze and pre-registration

- **Freeze:** `scratchpad/latency/FREEZE_SHA.txt`, frozen 2026-10-06T23:28:51Z.
  - `FREEZE_cohort.csv`: 332 fills, sha256 `3ba25d39…85bcd63`.
  - `FREEZE_holdlow_stack.csv`: 731 rows, sha256 `5a6eb0ba…f4de5a45`.
- **Pre-registration:** `PREREG_LATENCY.txt`, written 23:29:05Z, before any delay was priced.
- **Cohort:**
  - Engine-bar fresh-ON `FRENZY_READY` (204), plus `FRENZY_GREEN_BAR` with above_streak > 12 (WIDE hold-green, 128).
  - Live-eligible pairs, engine-parity market volume U2 < 1.0, Jan 10 → Sep 27.
  - Same cohort as `FRENZY_GVOL_GATE_REVALIDATION_2026-10-06.md` §2 (205 / 125 taken). The 6 fills priced on 1m are excluded because they cannot be re-timed.
- **Pricer:**
  - `holdlow_ticks/ticklib.py`. Entry = first print ≥ close + d. Live LOCK exit: −3 until the prior-print peak reaches +3, then max(+2, peak − 2). Fees 0.09, crossing-print fill, 12 h cap.
  - The dislocation guard (> 1 % from the last print before the close → refused) is re-decided at each delay, as live.
- **Parity:** at d = 12 s with exit slip 0.10, the pricer reproduces the frozen LOCK2 on **324 / 324 tick fills exactly** (max |Δ| 2e-15). The 8 frozen disloc refusals are refused here too.
- **Rulers:**
  - (i) Fixed slip: entry 0, exit 0.10 (also reported at exit 0.02).
  - (ii) Spread slip: entry × (1 + half-spread). Half-spread = max(½ tick, ½ Roll estimate on the prints from close to close + 60 s); median 0.012 %, mean 0.018 %.
- **Limitation:** unsequenced. Slots and the pair-day cap are not re-applied; they barely bind on this per-signal cohort (gvol revalidation §3).

## PART A: the value of speed

### A1. Delay curve: live cohort (FRENZY + WIDE-HG, ruler i, exit slip 0.10)

| d (s) | N | disloc refused | WR | avg %/trade | day CI | Jan–Apr | May–Sep | months > 0 | exit slip 0.02 | spread entry slip | mean drift vs close % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 332 | 0 | 54.5 | **+0.285** | [−0.07, +0.63] | +0.332 | +0.235 | 6/9 | +0.365 | +0.273 | +0.004 |
| 1 | 331 | 1 | 55.9 | +0.357 | [+0.00, +0.70] | +0.439 | +0.267 | 6/9 | +0.437 | +0.349 | −0.011 |
| 2 | 331 | 1 | 55.6 | +0.381 | [+0.01, +0.74] | +0.538 | +0.212 | 6/9 | +0.461 | +0.370 | −0.011 |
| 3 | 329 | 3 | 55.9 | +0.321 | [−0.05, +0.68] | +0.444 | +0.188 | 6/9 | +0.401 | +0.310 | −0.018 |
| 5 | 330 | 2 | 56.7 | +0.364 | [+0.00, +0.71] | +0.467 | +0.252 | 6/9 | +0.444 | +0.368 | −0.026 |
| 6 | 326 | 6 | 56.1 | +0.358 | [−0.01, +0.71] | +0.466 | +0.241 | 6/9 | +0.438 | +0.330 | −0.014 |
| 9 | 325 | 7 | 55.7 | +0.365 | [+0.01, +0.69] | +0.509 | +0.206 | 7/9 | +0.445 | +0.350 | −0.019 |
| 12 (old live ruler) | 324 | 8 | 56.2 | **+0.430** | [+0.05, +0.79] | +0.573 | +0.277 | 7/9 | +0.510 | +0.406 | −0.028 |
| 15 | 319 | 13 | 54.2 | +0.241 | [−0.13, +0.58] | +0.473 | −0.004 | 5/9 | +0.321 | +0.228 | −0.013 |
| 20 | 313 | 19 | 54.0 | +0.223 | [−0.13, +0.57] | +0.288 | +0.154 | 6/9 | +0.303 | +0.221 | −0.020 |
| 30 | 306 | 26 | 53.6 | +0.217 | [−0.15, +0.59] | +0.327 | +0.097 | 6/9 | +0.297 | +0.163 | −0.019 |

**Fine grid** (avg %/trade, ruler i):

| | 4 s | 7 s | 8 s | 10 s | 11 s | 11.5 s | 12.5 s | 13 s | 14 s |
|---|---|---|---|---|---|---|---|---|---|
| avg | +0.331 | +0.392 | +0.366 | +0.362 | +0.350 | +0.414 | +0.396 | +0.412 | +0.302 |

**Shifting the entry by 1 s or less** moves an individual fill by a mean |Δ| of about 0.10 % (12 → 13 s: mean |Δ| 0.103, 1.2 % of fills move by more than 1 %). That is the knife-edge of the −3 / +3 lock. The curve's wiggles (±0.05) are this noise. **12 s sits on a local high**, inside an 11.5–13 s bump, and it is the delay all of the exit and gate research was done at. Quoted "live" numbers are therefore probably ~0.03–0.05 optimistic against the 5–9 s band; that is not material.

### A2. Paired Δ vs d = 12 (same fills), with decomposition

| d | pairs | Δ %/trade | day CI | entry-price term | exit-path residual | better / worse / same | intent-to-trade Δ per signal (refused = 0) |
|---|---|---|---|---|---|---|---|
| 0 | 324 | −0.169 | [−0.347, +0.001] | −0.032 | −0.137 | 153 / 157 / 14 | −0.135 [−0.31, +0.03] |
| 1 | 324 | −0.099 | [−0.247, +0.042] | −0.018 | −0.080 | 151 / 156 / 17 | −0.064 |
| 2 | 324 | −0.072 | [−0.209, +0.058] | −0.016 | −0.056 | 154 / 156 / 14 | −0.040 |
| 3 | 323 | −0.141 | [−0.297, +0.006] | −0.003 | −0.138 | 148 / 158 / 17 | −0.102 |
| 5 | 324 | −0.111 | [−0.250, +0.011] | +0.003 | −0.115 | 148 / 147 / 29 | −0.058 |
| 6 | 322 | −0.088 | [−0.215, +0.027] | −0.001 | −0.087 | 140 / 159 / 23 | −0.068 |
| 9 | 322 | −0.061 | [−0.169, +0.037] | +0.001 | −0.062 | 138 / 153 / 31 | −0.063 |
| 15 | 318 | −0.167 | [−0.280, −0.070] | −0.015 | −0.153 | 128 / 157 / 33 | −0.188 |
| 20 | 312 | −0.159 | [−0.305, −0.023] | −0.005 | −0.155 | 133 / 158 / 21 | −0.210 |
| 30 | 303 | −0.175 | [−0.353, −0.001] | −0.001 | −0.174 | 131 / 159 / 13 | −0.220 |

**Reading the columns:**
- **Entry-price term** is the "price already ran" share: the Δ that buying at e_d instead of e_12 would cause with the same exit price.
- **Exit-path residual** is everything else.
- **"Price already ran" explains almost nothing.** The entry term is between −0.03 and +0.003 at every delay. Each delay's Δ is ~85–100 % exit-path residual.
- **Better / worse are balanced at every delay (two-sided).** Speed does not consistently help one type of fill. The shift reshuffles which fills reach +3 before −3.

### A3. Pre-registered decision statistic: gain per second saved (per-fill OLS slope, day-block CI)

| range | N | gain per second saved | day CI | P(≤ 0) |
|---|---|---|---|---|
| **d ∈ [3, 12] (primary)** | 321 | **−0.0148 %/trade/s** | **[−0.0312, +0.0013]** | 0.97 |
| d ∈ [0, 15] | 315 | −0.0024 | [−0.0138, +0.0086] | 0.67 |
| d ∈ [0, 3] | 329 | −0.0093 | [−0.0592, +0.0365] | 0.67 |
| FRENZY only [3, 12] | 198 | −0.0181 | [−0.0435, +0.0051] | 0.93 |
| WIDE-HG only [3, 12] | 123 | −0.0094 | [−0.0294, +0.0104] | 0.82 |

**Rule** ("propose engineering only if positive with CI lower bound > 0"): **FAILS.** The point estimate has the wrong sign.

### A4. Bands, and the universe that is not selected at 12 s

Band means (per-fill average over the delays in the band) and Δ vs the 5–9 s band, day CI:

| universe | N | d = 0 | 1–4 s | 5–9 s | 10–13 s | 14–15 s | 20–30 s |
|---|---|---|---|---|---|---|---|
| live cohort | 332 | +0.285 (Δ −0.09 [−0.23, +0.04]) | +0.346 (−0.04) | **+0.395** [+0.03, +0.73] | +0.386 (+0.02) | +0.269 (−0.09 [−0.23, +0.04]) | +0.234 (−0.11 [−0.24, +0.02]) |
| FRENZY only | 204 | +0.287 | +0.328 | +0.384 | +0.386 | +0.232 | +0.135 (Δ −0.18 [−0.36, +0.01]) |
| WIDE-HG only | 128 | +0.283 | +0.376 | +0.412 | +0.420 | +0.329 | +0.397 (Δ +0.01) |
| **ALL engine fresh-ON, ungated** | 1,755 | −0.179 (Δ −0.05 [−0.11, +0.01]) | −0.161 | −0.138 | −0.163 (Δ +0.00) | −0.166 (Δ −0.00) | −0.156 (Δ +0.00) |
| gvol ≥ 1 (blocked side) | 717 | −0.294 | −0.302 | −0.267 | −0.295 | −0.300 | −0.244 |

- **The large universe is flat from 0 to 30 s** (all band Δ within ±0.05).
- **The live cohort's late dip** (14–30 s, FRENZY −0.18) is directional only. Its CI touches 0, WIDE does not show it, and the 1,755-signal universe does not replicate it. **Treat it as unproven, not as a cliff.**
- **The only consistent sign is that d = 0 is slightly worse** (−0.03 to −0.09 in every universe, no CI clear of 0). The first prints after a close carry a small adverse micro-bounce. Racing to the close is not a free win.
- **The queued memory note's "+0.109 %/trade for 14 → 11 s" (ML scan-cadence) does not reproduce on FRENZY ticks.** Here 14 → 11 s is +0.048, and the ungated universe gives +0.011.

### A5. Slippage, separated

- **Exit slip** is a constant shift. 0.02 vs 0.10 is worth exactly +0.080 %/trade at every delay (columns "exit slip 0.02" vs "avg").
- **Entry spread (ruler ii)** costs 0.01–0.03 %/trade at every delay (e.g. 8 s: +0.366 → spread ruler ~+0.35). It does not depend on the delay.
- **Measured on the live paper fills (§B1):**
  - Entry: the paper fill sits on average **+0.036 %** above the last print at the order (it fills at the best ask). That is about twice the model's half-spread, still small.
  - **Stop exits: 9 paper stops sit on average ~0.18 % below the last print at `closed_at`** (0.00 … 0.37). The exit-slip assumption of 0.10 is therefore not pessimistic, and 0.02 is optimistic. With stops at 0.18 (stops are 44 % of exits), every number above drops by ~0.035.
  - **Caveat:** `closed_at` may lag the trigger by the monitor's cadence, so part of that 0.18 is monitor latency. That is an exit-side latency question, not an entry one.

### A6. Secondary (labelled): HOLD_LOWVOL STACK fills

Frozen membership; F4 dislocation re-decided at each delay; ruler i.

| d | STACK F1–F5: avg [day CI] · May–Sep | STACK_WIDE: avg [day CI] · Jan–Apr / May–Sep |
|---|---|---|
| 0 | +0.220 [−0.06, +0.51] · +0.184 | +0.256 [−0.00, +0.51] · +0.111 / +0.370 |
| 1 | +0.226 [−0.05, +0.52] | **+0.273 [+0.01, +0.53]** · +0.163 / +0.359 |
| 3 | +0.223 [−0.05, +0.51] | +0.261 [+0.00, +0.52] |
| 5 | +0.228 [−0.07, +0.52] | +0.250 [−0.02, +0.51] |
| 6 | +0.242 [−0.04, +0.53] | **+0.279 [+0.01, +0.55]** · +0.098 / +0.421 |
| 9 | +0.222 [−0.06, +0.52] | +0.251 [−0.01, +0.53] |
| 12 (old ruler) | +0.191 [−0.09, +0.47] | +0.211 [−0.06, +0.48] |
| 15 | +0.249 [−0.04, +0.54] | +0.260 [−0.01, +0.54] |
| 20 | +0.262 [−0.03, +0.56] | +0.273 [−0.01, +0.54] |
| 30 | +0.207 [−0.08, +0.50] | +0.206 [−0.06, +0.46] |

- **Gain per second saved over [3, 12]:** STACK +0.0039 [−0.004, +0.012]; STACK_WIDE +0.0062 [−0.002, +0.015]. Both fail the rule.
- **On this universe 12 s is the *worst* delay** (a local dip). On FRENZY it was a local high. That is the clearest sign that single-delay differences of ±0.05 are noise.
- **Where the gap comes from** (STACK F1–F5): close/0.02 (+0.300) vs 12 s/0.10 (+0.191) = 0.109.
  - **0.080 (~75 %) is the exit-slip assumption.**
  - 0.029 is the 0 → 12 s delay. Its paired CI spans 0, and the 5–9 s band equals d = 0.
- **STACK_WIDE gap:** 0.125, of which 0.080 is exit slip; 12 s is the dip.
- **Pre-declared bar** (avg ≥ +0.10, day-CI lo > 0, May–Oct > 0):
  - STACK_WIDE technically clears it at d = 1, 3 and 6 s (CI lo +0.01 / +0.00 / +0.01). Neighbouring delays fail (2, 5 and 9 s).
  - The family-adjusted null from PREREG_FILTERS (p 0.044 vs the 0.0038 needed) does not depend on the delay.
  - With the measured ~0.18 stop slip the band average falls to ~+0.22, and the CI goes below 0.
  - **Flagged "possible luck". HOLD_LOWVOL stays CLOSED.** Speed does not rescue it.

## PART B: where the live seconds go

### B1. Measured live latency (every live FRENZY / WIDE fill; master pool + 10-06 21:11 export; logs from L12/all.log + ebl2)

| fill | path | order (s after close) | drift to order % | fill vs close % | fill vs last print % | fill vs last taker-buy % |
|---|---|---|---|---|---|---|
| ENJ 10-03 03:35 | old scan path | **79** | +0.22 | +0.14 | −0.08 | – |
| AIN 10-03 23:25 W | frenzy_loop | 12 | +0.89 | +0.89 | 0.00 | – |
| SAND 10-04 05:05 | frenzy_loop | 16 | −0.24 | −0.19 | +0.05 | – |
| AIN 10-04 11:15 | frenzy_loop | 11 | −1.05 | −0.92 | +0.13 | – |
| SAND 10-04 14:05 | frenzy_loop | 12 | +0.09 | +0.09 | 0.00 | – |
| MOVR 10-05 09:15 W | frenzy_loop | 11.4 | −0.03 | +0.01 | +0.04 | 0.00 |
| RLC 10-05 12:00 W | frenzy_loop | 11.1 | −0.04 | +0.02 | +0.07 | 0.00 |
| AIN 10-05 16:15 W | frenzy_loop | 11.2 | +0.10 | +0.09 | −0.02 | −0.02 |
| FLUID 10-06 02:30 W | after speed-up 835620d | **7.5** | +0.09 | +0.14 | +0.05 | 0.00 |
| ORCA 10-06 09:40 | after speed-up | **7.5** | −0.12 | −0.12 | 0.00 | 0.00 |
| UMA 10-06 10:10 | after speed-up | **7.6** | −0.22 | −0.20 | +0.02 | +0.02 |
| ORCA 10-06 18:05 | after speed-up | **7.5** | +0.35 | +0.42 | +0.06 | +0.06 |

- **Today's live entry is ~7.5 s to the order** (`opened_at` stamps 8 s). The 12 s live ruler is obsolete.
- **Pass-end distribution** (FRENZY_LOOP "s into the bar", 479 passes):
  - Before the speed-up: median 12 s, p95 16 s.
  - **After it: median 9 s, p90 12 s, p95 13 s, max 26 s.** 4 of 248 passes (1.6 %) ended at ≥ 14 s and 2 (0.8 %) at ≥ 20 s.
  - No FRENZY_LATE refusal in the logs.
- **Drift to the order is noise** (−1.05 … +0.89, mean ≈ 0). The paper fill is the best ask: +0.036 % mean over the last print, which matches the last taker-buy print.

### B2. Timeline from the bar close to the order (code + Oct-6 log timestamps)

| step | code | when (s after close) | cost |
|---|---|---|---|
| deliberate settle sleep | `main.frenzy_loop` sleeps to close + 4 s; `_update_frenzy_pass` also waits for `now − bar_open ≥ 4000` ms | 0 → 4.0 | **4.0 s (fixed)** |
| tickers + universe filters | `get_top_futures_pairs(5000, …)` (crypto / new-listing / Alpha filters) | 4.0 → 4.35 | ~0.35 s |
| market-volume read started | `_frenzy_gvol_start` → `_market_gvol_bars`: 50 pairs × 60 bars, research client, Semaphore(10), 6 s timeout each | 4.35 → **~7.0–7.7** | ~3 s (5 waves of 10) |
| shortlist klines + judge as each lands | `_fz_b5` (5-bar incremental tail; full 1500-bar raw read every 12 bars per pair), 1h every 6–8 h, Semaphore(6), `as_completed` → `frenzy_walk`, ATR, indicators | 4.35 → ~6.7–7.0 for the ON pair | ~2.5 s for ~22–25 pairs |
| `_frenzy_open`: slots / pair-day DB checks, then **await gvol** | `_frenzy_gvol_value(wait=True)` | refusals log at ~7.0, "→ opening" at ~7.5–7.7 | **gvol is the binding wait (~0.5–0.8 s)** |
| `open_position`: BOT_OPEN_LANE, sizing, `fetch_orderbook` (paper fill = best ask), DB writes | lane can wait on a concurrent scan open | 7.5 → 8.1 ("ORDER CREATED") | ~0.6 s (tail: the lane) |
| WS subscribe of the new pair | after the order (240/241 moved it off the open path) | 8.1+ | does not affect the fill |

### B3. Engineering options (none recommended for speed)

The backtest value of every option is ~0, because the curve is flat between 1 and 13 s.

| # | option | new latency | complexity / risk | backtest value at that delay vs today's ~8 s |
|---|---|---|---|---|
| 1 | Settle sleep 4 s → ~1.5 s (retry is already built in: "window missing or not up to the last closed bar") | ~5–5.5 s | low code; **parity risk**: gvol pairs whose closed bar is not yet served are dropped (min_pairs 30), which changes the gvol value the gate was validated on | 5–9 s band = same as 8 s (Δ 0); per-second slope −0.015 (wrong sign) |
| 2 | Pre-compute gvol: fetch the 47 prior bars before the close, read only the closed bar after | removes the ~0.5–0.8 s gvol wait → ~7 s | medium; **parity risk**: the top-50 must still be ranked at read time (the API3 / VTHO lesson) | 0 |
| 3 | Event-driven: WS kline-close (x = true) for the shortlist + the 50 gvol pairs; judge on the event | ~1–2 s | high (≈ 75 streams, gap / reconnect handling, a second data path to keep in parity) | d = 1–4 s band +0.346 vs 5–9 s +0.395: **Δ −0.04 [−0.13, +0.05]**, d = 0 −0.09 |
| 4 | FRENZY priority in BOT_OPEN_LANE (a scan open never delays a FRENZY open) | cuts the > 14 s tail (1.6 % of passes) | low–medium | ≤ 0.016 × ~0.1 ≈ 0.002 %/trade, and the late penalty itself is unproven (A4) |
| 5 | More parallel fetches | already done (`as_completed`, Semaphore 6 / 10, incremental windows) | – | – |

## VERDICT

**(a) Is speed worth it? No.**
- Gain per second saved over [3, 12] s = **−0.015 %/trade/s, day CI [−0.031, +0.001]** (FRENZY + WIDE-HG, 321 fills).
- Applied to the live volume (~330 FRENZY + WIDE fills a year at the gate, ~$4.5k notional each, 5 s saved by options 1–2):
  - point estimate ≈ **−0.07 %/trade ≈ −$1k a year** (noise);
  - CI upper bound ≈ +0.0065 %/trade ≈ **+$100 a year**.
- No positive case exists. Prices do not run away in the first 15 s (entry-drift term ≈ 0). Per-delay differences are the lock exit's knife-edge noise. The 1,755-signal universe is flat from 0 to 30 s.

**(b) What to build first: nothing for speed.**
- **Do (research hygiene, no code):** re-base the backtests' "live ruler" from 12 s to **8 s**, the measured latency. Keep the 0.10 exit slip; the paper stops measure ~0.18. The 12 s numbers sit on a local high for FRENZY: quoted +0.43 vs the 5–9 s band +0.395.
- **Optional, only if the operator wants robustness:** option 4 (FRENZY lane priority). Its value is negligible.
- **Do not do:** option 3 (high complexity, no value), and options 1–2 unless gvol parity is re-proven first.
- **The open question worth more than entry speed is the exit side.** Paper stop fills sit ~0.18 % below the last print at `closed_at` (N 9). An exit-latency / stop-fill study is the better next use of effort.

**(c) Does HOLD_LOWVOL become viable at achievable latency? No (secondary).**
- Its gap is ~75 % the exit-slip assumption, not speed.
- At 1, 3 and 6 s STACK_WIDE grazes CI lo ≥ 0, while neighbouring delays fail. With the family null unchanged (p 0.044 vs 0.0038) and the measured stop slip, it stays **CLOSED**. Flagged as possible luck.

## Files (scratch: `scratchpad/latency/`)

| file | contents |
|---|---|
| `FREEZE_cohort.csv`, `FREEZE_holdlow_stack.csv`, `FREEZE_SHA.txt`, `PREREG_LATENCY.txt` | freeze and pre-registration |
| `freeze.py`, `price.py` (ticklib, Pool 8; `DEL` / `TAG` env), `analyze.py`, `bands.py` | scripts |
| `priced_cohort.pkl` (+ `_fine`), `priced_holdlow.pkl`, `priced_all.pkl` | priced fills |
| `analyze_cohort.out`, `analyze_holdlow.out`, `bands_all.out`, `delay_table_*.csv` | outputs |
| `live.py` → `live_fills_latency.csv`, `live_exit.py` → `live_exit_slip.csv` | live entry and exit measurements (aggTrades with side via public REST for Oct 5–6; archives for Oct 3–4) |
| `passes.txt` | FRENZY_LOOP pass durations |
