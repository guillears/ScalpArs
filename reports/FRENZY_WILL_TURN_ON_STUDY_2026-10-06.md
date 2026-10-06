# FRENZY: can we tell, at signal time, which "rising volume" episodes will turn ON? (2026-10-06)

Research only. Nothing in `services/`, `config.py`, `trading_config.json`, `templates/` or `tests/` was touched. Nothing was committed, and the bot's API was not called (only public Binance klines were used). **Unreviewed:** no caveman or deep review has run on these numbers, so this report makes no arm recommendation (feedback_no_arm_before_review).

## Resumen en lenguaje simple (español)

**La pregunta.** El estudio anterior (volumen subiendo hacia 100× y precio alejándose del VWAP) mostró dos cosas:
- Si el episodio **después** se ponía ON, entrar temprano ganaba unos +0,85 %/trade.
- Si **nunca** se ponía ON, perdía entre −0,9 y −1,8 %.

Hoy busqué si hay algo visible **en el momento de la señal**, sin mirar el futuro, que separe un caso del otro. Lo hice de forma 100 % automática y con reglas fijadas antes de mirar los resultados.

**Qué hice.**
- Tomé las 1.121 señales del año, una por episodio, del 10-ene al 3-oct.
- Medí **116 variables** en la vela de la señal. Volumen y su pendiente, distancia al VWAP, tamaño del spike, velas, RSI/ADX/DI y sus cambios, todas las brechas entre EMAs y sus giros, la firma "gap girando" de tus trades manuales, el flujo comprador del último minuto, BTC, el volumen del mercado, la amplitud y la hora.
- Busqué reglas **solo en la primera mitad** (hasta el 22-may), probando todas las combinaciones de a dos: 46.637 pruebas.
- Congelé las 3 mejores y un modelo simple. Después los probé en la segunda mitad, que no había tocado.

**Resultado.**
1. **No hay separador de "ganancia" que sobreviva.**
   - Las 3 mejores reglas de la primera mitad ganaban +1,1 a +1,5 %/trade.
   - En la segunda mitad las tres **pierden**: −0,17, −0,42 y −0,42 %/trade.
   - Ninguna superaba siquiera la barra de "suerte" de la primera mitad. Con 46.000 pruebas, el azar puro da combinaciones así de buenas.
2. **Sí se puede predecir bastante bien quién se va a poner ON.** El modelo de 5 variables acierta fuera de muestra con un AUC de 0,72; el azar da 0,54. La clave es simple: cuanto más cerca está el volumen de 100×, más probable es que se ponga ON.
3. **Pero predecir "ON" no predice "ganancia"** (AUC 0,49 sobre ganar/perder). La razón:
   - **Los ON fáciles de predecir** tienen el volumen ya cerca de 100×. Están a pocos minutos del ON y ya subieron, así que queda poco recorrido. Ganan poco: +0,1 a +0,8.
   - **Los fallos de ese mismo grupo** pierden fuerte: −1,4 a −1,9.
   - **Los ON que más pagan** son los que arrancan con volumen bajo, unos 26×, y tardan 2–3 h en ponerse ON. Esos son justamente los **imposibles de predecir**.
   - Las dos cosas se cancelan.
4. **Tu firma "gap 5-20 negativo pero subiendo + gap 5-8 positivo"** casi no aparece en este grupo: 22 veces en el año, porque aquí el precio ya está arriba de su promedio.
   - Da +0,65 %/trade, pero con un IC de −0,68 a +2,18.
   - Un solo par explica el 68 % de la ganancia.
   - Es ruido por ahora, y no se pudo testear con el rigor pedido.
5. **Bajar el piso de volumen 24 h de $20M a $10M** (el caso GRIFFAIN) **empeora**. Las 317 señales nuevas, de pares entre $10M y $20M, pierden −0,23 % en la primera mitad y −0,55 % en la segunda.
6. **Comparación.** En el mismo período, las entradas reales de FRENZY_LONG ganan +0,82 %/trade (83 trades). Ninguna regla de este estudio se le acerca.

**Veredicto: FALLA.** No hay regla automática que sepa de antemano qué episodio se va a poner ON **y además** gane plata. No propongo línea de observación ni cambio alguno. Lo que te funciona a mano sigue siendo tu timing y tu salida, no una condición medible en la vela de la señal.

---

## 1. Pre-registration and process (timestamps)

- `scratchpad/willon/PREREG.txt`, stamped **2026-10-06 22:13:08 UTC**. It covers the cohort, labels, time split, the 65-feature family, granularities, the score, the null, the freeze rule, the model, the confirmation bars, the secondary exits and the $10M secondary read.
  - The only things looked at before it were the cohort N and feature distributions. No P&L or label was looked at by feature.
- **Addendum A (operator momentum dynamics)** was added to the same file before any outcome split was computed. It holds 51 more features:
  - RSI Δ1/Δ3/Δ6 and its 50/55/60 cross-ups;
  - ADX Δ1/Δ3, +DI Δ3, −DI Δ3, spread Δ1/Δ3/Δ6, and fresh +DI>−DI crosses in the last 3/6 bars;
  - the six EMA gaps 5-8 / 5-13 / 5-20 / 8-20 / 13-50 / 20-50 as % of price: level, Δ1/Δ3/Δ6, cross-up in 3 bars and slope turn-up in 3 bars;
  - the EMA5 slope;
  - the operator signatures TG (gap5-20 < 0 ∧ Δ3 gap5-20 > 0 ∧ gap5-8 > 0) and TG+ (TG ∧ ADX Δ3 > 0 ∧ RSI Δ3 > 0).
  - Because no outcome had been seen, they joined the **same** family and the **same** shuffled null. That raises the bar; nothing was nulled separately in their favour.
- `FROZEN.json` / `model_frozen.pkl`, stamped **22:18:58 UTC**, were frozen after discovery and before any confirmation-half outcome.

**Cohort (declared).** The union of the 20 frozen rising-volume cells: the first bar per (pair, spike_ts) episode with 25 ≤ vol_mult < 100 ∧ [(vm/vm₋₃ ≥ 1.2 ∧ Δdist₃ > 0) ∨ (vm/vm₋₆ ≥ 1.2 ∧ Δdist₆ > 0)]. The base is risingvol's verified engine-walk set:
- flagged, ≥ 2 h after the spike, not in state, above the spike VWAP;
- vol24 ≥ $20M and shortlisted;
- live universe.

That gives **1,121 signals, 263 days, 319 pairs**.

**Entry, exit and target.** Entry, exit and pricer are risingvol's unchanged: next 5m open + 0.02 %, fees 0.09 %, the live lock on public 1m klines with pessimistic intrabar order. **Target** = lock P&L % at 1×.

**Labels.** Labels come from the engine cohort's fresh-ON bars of the same episode (`engcoh/bars_all.pkl`, the real `frenzy_walk`):
- TURNS_ON = an ON later in the episode;
- ALREADY_PAST = an ON before the signal;
- NEVER_ON = neither.

**Split.** Discovery (D) = signal before 2026-05-23 00:00 UTC (the calendar midpoint), 589 signals. Confirm (C) = 532 signals, untouched until the freeze.

**Engine parity.** The cohort and its vol-multiple and VWAP lines are risingvol's. Its walk matched to 8.5e-14 and its VWAP to 2.6e-11. Its eligibility reconstruction for the $10M read reproduced risingvol's `elig` column exactly (110,829 / 110,829 bars). Indicators use the bot's own `ta` classes (RSI 12, ADX 14, EMA 5/8/13/20/50) on the last 300 closed 5m bars. gvol = `global_volume_ratio` V1 on the U2 universe (rolling-24h top-50, ≥ 90 d, not Alpha, COIN), as in `wrev/gvol_live.py`. Taker-buy share and the last-1m features come from public 1m klines fetched per signal, with 100 % coverage.

## 2. The outcome split (what we are trying to predict)

| half | label | N | WR | mean %/trade (lock) | day-CI 95 % |
|---|---|---|---|---|---|
| D | all | 589 | 43 % | **−0.41** | −0.66 … −0.17 |
| D | TURNS_ON | 189 | 58 % | +0.69 | +0.21 … +1.20 |
| D | NEVER_ON | 300 | 33 % | −1.15 | −1.45 … −0.84 |
| D | ALREADY_PAST | 100 | 42 % | −0.31 | −0.95 … +0.38 |
| C | all | 532 | 50 % | **+0.09** | −0.18 … +0.36 (EVAA = 62 % of the net total) |
| C | TURNS_ON | 155 | 59 % | +0.96 | +0.35 … +1.58 |
| C | NEVER_ON | 296 | 44 % | −0.49 | −0.81 … −0.15 |
| C | ALREADY_PAST | 81 | 58 % | +0.54 | −0.17 … +1.26 |

This reproduces risingvol's finding. The base cohort is about −0.4 in D and about 0 in C, which is a regime difference, not a rule.

## 3. Discovery (half D only)

- **Coverage:** all 116 features are scored on 100 % of D and C fills, so there are no "unscored = rest" problems.
- **Tests:**
  - 988 one-dimensional buckets: sign / natural split, terciles and quintiles, with cut points from D.
  - **45,649** two-dimensional cells: every feature pair × tercile × tercile.
  - Minimum N per cell = 30.
- **Score:** t of the cell's mean lock P&L.
- **Selection-adjusted threshold:** the 95th percentile of the max-t over all tests under 200 within-day outcome permutations = **3.23**. A global permutation gives 3.19, and the 1D-only family gives 1.79.

| best of | cell | N | WR | mean | t | p_adj |
|---|---|---|---|---|---|---|
| 1D | d_di3 top quintile (> 0.88) | 118 | 50 % | +0.27 | 0.79 | 1.00 |
| 1D | vm ÷ peak-vm 2nd quintile | 117 | 52 % | +0.25 | 0.76 | 1.00 |
| 2D #1 | lvm1 ≤ 0.079 (volume flat over the last bar) × taker-buy 5 min > 54.4 % | 57 | 68 % | +1.49 | 3.06 | 0.10 |
| 2D #2 | ADX Δ3 ∈ (1.89, 4.25] × EMA5 slope ≤ 1.21 % | 57 | 65 % | +1.24 | 2.59 | 0.58 |
| 2D #3 | −DI > 13.2 × RSI Δ1 ∈ (1.06, 5.64] | 51 | 69 % | +1.08 | 2.52 | 0.70 |
| best operator-momentum cell | gap5-20 slope turn-up × gap8-20 Δ6 top tercile | 40 | 70 % | +1.15 | 2.37 | 0.86 |

- **Nothing clears the selection-adjusted threshold.**
- **No single feature is significantly positive at any granularity.** The best 1D bucket has t = 0.79.
- The 2D "winners" sit inside what 46k random tests produce. As pre-registered, the top 3 distinct-feature cells were frozen anyway and flagged as "did not clear the discovery null".

**Not testable at the locked N (blind spots):**
- **TG**: 10 fills in D, 22 in the year. **TG+**: 3 in D, 8 in the year. The cohort requires the price to be rising above the VWAP, so the "gap5-20 still negative" signature is structurally rare here.
- Order book, funding and open interest: not in the data.
- Tick-level entry: the 1m pricer is the kinder one (risingvol §3).
- The live forming-bar gvol: the closed-bar V1 was used.

## 4. Frozen rules on the untouched half C

| rule (frozen) | D: N / mean | **C: N** | days | WR | **mean** | day-CI 95 % | top pair / day share | leave-one-month-out (min…max) | random same-episode bars | p_adj |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 lvm1-low × taker5-high | 57 / +1.49 | 55 | 42 | 49 % | **−0.17** | −0.87 … +0.56 | n/a (net < 0) | −0.75 … +0.22 | −0.93 | 0.43 |
| R2 ΔADX3-mid × EMA5-slope-low | 58 / +1.16 | 41 | 37 | 46 % | **−0.42** | −1.27 … +0.46 | n/a | −0.83 … +0.05 | −0.87 | 0.87 |
| R3 −DI-high × ΔRSI1-mid | 51 / +1.08 | 69 | 52 | 46 % | **−0.42** | −1.04 … +0.20 | n/a | −0.70 … +0.16 | −0.51 | 0.85 |
| M model top tercile | 196 / −0.42 | 159 | 100 | 51 % | **+0.12** | −0.42 … +0.64 | BR 84 % / one day 78 % | −0.07 … +0.26 | −0.30 | 0.066 |

**Comparators in the C period** (same 1m pricer, live universe):

| set | N | WR | mean | day-CI |
|---|---|---|---|---|
| All FRENZY fresh-ON bars | 659 | 50 % | +0.07 | −0.19 … +0.31 |
| FRENZY_LONG trades (READY ∧ gvol < 1) | 83 | 57 % | **+0.82** | −0.07 … +1.67 |

- **All three rules flip sign out of sample.** That is the textbook signature of selection noise.
- The model rule is the only one positive in C, at +0.12. But:
  - it was **−0.42 in D**;
  - its CI spans 0;
  - one pair carries 84 % of its net and one day carries 78 %;
  - it does not beat the random-bar null after selection (p_adj 0.066).
- **The haircut** (30–50 %) is not applicable as a pass. For reference, M would be +0.06 to +0.08 %/trade after the haircut.

**Secondary exits (sensitivity only, C):**

| rule | fixed +3/−3 | time stop 60 min | time stop 240 min |
|---|---|---|---|
| R1 | −0.07 (CI −0.74 … +0.63) | −0.08 | +0.96 (CI −1.26 … +3.64) |
| R2 | −0.23 | −0.32 | −0.51 |
| R3 | −0.23 | −0.43 | −0.39 |
| M | +0.05 | −0.59 | −0.41 |
| base | 0.00 | −0.23 | −0.09 |

None is distinguishable from 0. R1's 240-min figure is one of 12 exit × rule reads and has a 5-point-wide CI.

## 5. The model: "will turn ON" is predictable, profit is not

- **Logistic model.** Five features were picked on D by |rank-t| against TURNS_ON:
  - vol_mult;
  - peak vol_mult since the spike;
  - log vm/vm₋₁₂;
  - log vm/vm₋₆;
  - gap20-50 Δ6.

  It was fit on D with rank-standardised features and ridge 1.

| score on C | AUC vs TURNS_ON | null 95th | p | AUC vs win (lock > 0) | null 95th | p | rank-corr with lock |
|---|---|---|---|---|---|---|---|
| logistic (5 features) | **0.716** | 0.543 | < 0.001 | 0.493 | 0.541 | 0.61 | −0.004 |
| depth-2 tree (vol_mult → lvm6 / lvm12) | **0.655** | 0.537 | < 0.001 | 0.515 | 0.539 | 0.28 | +0.04 |

**Why predicting ON does not pay.** The table splits by model score tercile (cuts from D) and label.

| half | score tercile | label | N | mean | median lead to ON (min) | median vol× at signal |
|---|---|---|---|---|---|---|
| D | low | TURNS_ON | 45 | **+1.28** | 130 | 27 |
| D | low | NEVER_ON | 150 | −1.06 | — | 27 |
| D | high | TURNS_ON | 81 | +0.11 | 55 | 72 |
| D | high | NEVER_ON | 40 | **−1.86** | — | 57 |
| C | low | TURNS_ON | 42 | **+1.55** | 183 | 27 |
| C | low | NEVER_ON | 144 | −0.28 | — | 26 |
| C | high | TURNS_ON | 57 | +0.83 | 65 | 65 |
| C | high | NEVER_ON | 44 | **−1.37** | — | 57 |

- **Episodes the model can call** have volume already near 100×. ON is minutes away and most of the move is already in the price, so they earn little. Their misses, the high-volume failures, are the worst losers in the cohort.
- **The rich turn-ONs** start at ~26× volume and take 2–3 h to turn ON. These are the ones the features cannot see.
- The two effects cancel, so the score carries no P&L information (rank-corr ≈ 0 on C).

## 6. Secondary read: vol24 floor lowered to $10M (declared up front; not in the verdict)

- **Method:** a re-scan with the same code, VMIN = $10M for both the vol24 gate and the shortlist. Eligibility was reconstructed (exact parity on the $20M set). **1,271 signals**, 317 of them on pairs with vol24 of $10–20M, which needed 1m paths fetched.
- **GRIFFAIN anecdote:** GRIFFAIN 2026-10-06 17:05 (+23 % in the next hour, best pre-ON move) had 24 h vol of $16–17M, below the $20M shortlist. Today is beyond the year data (ends 2026-10-04), so it is an N = 1 anecdote only.

| set | D N / mean (CI) | C N / mean (CI) |
|---|---|---|
| added pairs, vol24 $10–20M | 171 / **−0.23** (−0.70 … +0.26) | 146 / **−0.55** (−1.08 … +0.01) |
| ≥ $20M part of the $10M cohort | 500 / −0.26 | 454 / +0.11 |
| frozen R1 / R2 / R3 / M on the $10M cohort (C) | — | +0.08 / +0.23 / −0.45 / +0.05. For R1, R2 and M, one pair holds more than 100 % of the net. |

Lowering the floor adds net losers. GRIFFAIN 10-06 is not representative of the $10–20M band.

## 7. Relation to earlier refutations (not re-run)

- **BELOW_AVERAGE_TEST (10-02):** bought *below* the VWAP and was refuted. This study is above the VWAP with rising volume, a different population.
- **STAIRCASE_STUDY:** the scout staircase is FRENZY at 50× and does not pay. This cohort overlaps it (25–100×), and the result agrees: half-way-to-ON volume does not pay as a class.
- **ON_SCALP_STUDY:** buying at ON fails on every exit grid, and ADX/DI "strong" does not create the edge. Here ADX/DI dynamics were re-tested *before* ON, as entry separators, and also fail out of sample.
- **MINHOURS_AND_WAITRED:** moving the 2 h clock or waiting for a red bar changes ~10–18 fills and is noise. This study is a different entry point (pre-ON) and fails too.
- **REENTRY_NEW_ANGLES:** re-entries after ON. In this cohort that is the ALREADY_PAST label. It is not a separator: D −0.31 and C +0.54 flip sign.
- **REGIME_REVIEW:** no regime split separates FRENZY_LONG. The BTC, gvol and breadth features were included here. The best such cells (BTC RSI, BTC trend gap, gvol × EMA gap) reach t ≈ 1.95, far below the 3.23 threshold. Per the WINDOW-UNITS rule they would be sleeve switches anyway.

## 8. Verdict vs locked bars

| criterion | R1 | R2 | R3 | M |
|---|---|---|---|---|
| C mean > 0 | ✗ −0.17 | ✗ −0.42 | ✗ −0.42 | ✓ +0.12 |
| day-CI lower > 0 | ✗ | ✗ | ✗ | ✗ −0.42 |
| ≥ 30 fills on ≥ 8 days | ✓ | ✓ | ✓ | ✓ |
| no pair or day ≥ 50 % of the gain | n/a | n/a | n/a | ✗ (84 % / 78 %) |
| beats the random-bar null after selection | ✗ 0.43 | ✗ 0.87 | ✗ 0.85 | ✗ 0.066 |
| cleared the discovery null | ✗ | ✗ | ✗ | — (was −0.42 in D) |

**FAIL. No pre-registered signal-time rule separates profitable "will turn ON" episodes from losers. That covers levels, slopes, accelerations, turning flags, the EMA-gap family, the operator's momentum dynamics, 1m flow, market state and time, plus all 45k pairwise combinations and a 5-feature model.**
- Turning ON **is** predictable out of sample (AUC 0.72), but the predictable part is the low-value part.
- No observe line is proposed. No candidate is borderline: R1–R3 are negative in C, and M fails concentration and was negative in D.
- Nothing to arm.
- The only unfinished thread is the operator's TG signature: N = 22 in this cohort, +0.65 %/trade, CI −0.68 … +2.18, one pair = 68 % of the gain. It would need its own pre-registered study on a cohort where gap5-20 < 0 is common (for example, FRENZY-flagged bars around the VWAP cross, not this rising-above-VWAP set). It is a watchlist note, not evidence.

**Scratch:** `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/willon/`:
- `PREREG.txt` + `.stamp`, `FROZEN.json` + `.stamp`
- `cohort.py`, `features.py`, `feats_def.py`, `disc.py` (with `disc_tests.csv` and `disc_meta.json`), `model.py`, `common.py`, `confirm.py` (with `confirm_main.csv` and `confirm_10m.csv`)
- the $10M read: `scan10.py`, `elig.py`, `cohort10.py`, `price10.py`
- fetched klines: `tk/` (1m taker) and `m1x/` (1m paths)
