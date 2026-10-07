# Manual trades → automated rules: full study (2026-10-06)

Research only. Nothing in `services/`, `config.py`, `trading_config.json`, `templates/` or `tests/` was touched. Nothing was committed. No auth API was called (public Binance klines only).
**Unreviewed:** no caveman or deep review has run on these numbers yet. Per `feedback_no_arm_before_review`, nothing here is an arm proposal.

Scripts, pre-registrations and raw outputs are in the scratchpad `…/scratchpad/manualall/`:
- `PREREG_RECLAIM_DIP.txt` and `PREREG_MANUAL_CANDIDATES.txt`, with addenda 1 and 2. Every rule and threshold was frozen there **before** any year outcome was computed.
- Data: `census.py`, `feat.py`, `momfeat.py`, `yearwalk.py`, `yenrich.py`, `gvolyear.py`.
- Backtest: `signals.py`, `resolve.py`, `price.py`, `evalall.py`.
- Screens: `manscreen*.py`, `sweep_run.py`.

Files saved in `reports/`:
- `…_manual_features.csv`: every manual fill with its rebuilt context.
- `…_eval.csv`: every candidate × exit.
- `…_reclaim_grid.csv`: the 48-cell RECLAIM_DIP family.

---

## Resumen en castellano (para el operador)

**La pregunta.** ¿Hay alguna regla **automática** que capture la plata que aparece en tus trades manuales y que el bot no toma, y qué filtros la ayudan? Tus trades se usaron solo como **fuente de hipótesis**, no como una habilidad a copiar.

**Respuesta corta: no encontramos ninguna regla automática con ventaja real que se pueda activar. Hay una sola pista que vale la pena observar gratis.**

1. **Tu historial completo.** Son **126 trades manuales cerrados**, del 29-sep al 6-oct (6 días, 28 pares): 95 largos y 31 cortos.
   - Ganaste el 65 % de las veces, con **+0,11 % por trade**.
   - El estudio anterior tenía estos mismos 126 trades en sus datos, pero **solo analizó los 95 largos**: le faltaron tus 31 cortos. No hay trades manuales antes del 29-sep (ahí nació la función manual) ni el 4 y 5-oct.
   - Casi todo es **un solo día y una sola moneda**: 57 trades en MOVR (53 el 1-oct), 17 en SAND y 13 en AIN.
2. **¿La ganancia vino de la entrada o de la salida?** De ninguna de las dos, en promedio.
   - Si a tus mismas entradas les ponemos una salida fija de +1 / −3 (tu FLOOR), dan **+0,12 % por trade**, casi lo mismo que tu resultado real (+0,11 %).
   - Con +1 / −3 hay que ganar el 75 % de las veces solo para empatar. Tus entradas ganan el 77 %, prácticamente en el punto de empate.
   - En 23 de tus 44 pérdidas, el precio después tocó +1 % antes que −3 %: cerraste antes de tiempo.
3. **Probamos 5 reglas automáticas sacadas de tu historial, más la tuya de hoy** (comprar el primer pullback tras recuperar el VWAP). Las probamos sobre **un año entero** (enero → 3-oct, ~400 pares, el mismo motor FRENZY del bot) y **todas fallan**:
   - **Comprar dips con FRENZY encendido (como MOVR):** −0,14 % por trade (5.506 trades).
   - **Shortear el giro con FRENZY encendido:** −0,25 %.
   - **Tu patrón de hoy, con tus bandas literales:** en todo el año aparece solo **19 veces**, y pierde −1,31 % por trade. Ni siquiera tus 3 trades de hoy cumplen las bandas literales: les falta por redondeo (11 velas debajo en vez de 12, volumen 72,5× en vez de ≤ 72×, etc.).
   - **La versión amplia (RECLAIM_DIP v1):** −0,13 % (4.433 trades). Es **peor que entrar en un momento al azar** del mismo episodio.
   - **48 variantes de tu patrón:** 19 dan positivo, pero ninguna supera lo que da el azar al probar tantas.
   - **Tu firma de "gap 5-20 negativo pero subiendo + ADX y RSI subiendo":** −0,20 %. Ninguno de tus trades la cumplía.
4. **Filtros.** Ningún filtro sobrevive: ni el "mercado con poco volumen" que salía muy fuerte en tus trades (29 de 29 ganados), ni ningún otro de ~25.000 combinaciones probadas, ni el volumen de mercado en sus 4 versiones, ni el momentum (RSI, ADX y DI subiendo, cruces de EMAs).
5. **La única pista.** Pares FRENZY que **se sostienen arriba del VWAP una hora completa, pero con volumen menor a 100×** (el bot no los compra porque FRENZY pide 100×), entre 2 y ~17 horas después del spike. Es lo que hiciste en SAND, STRK y ATH.
   - En el año: **+0,22 % por trade** (2.239 trades, positivo en las dos mitades del año y en 8 de 10 meses).
   - **Pero** no le gana a entrar en un momento cualquiera de ese mismo estado. El corte de horas lo encontramos mirando los datos. Y con el supuesto más pesimista dentro de cada vela de 1 minuto cae a −0,01 %.
   - **Recomendación:** anotarla como **línea de observación gratis** (sin plata), con los umbrales congelados. No activarla.
6. **Qué NO recomendamos:** seguir juntando clicks manuales como camino. El objetivo es automático, y el camino es la línea de observación del punto 5, o nada.

---

## 1. Census of manual fills

**Tagging.** `open_manual_position` (`services/trading_engine.py:10816`) labels every manual open `entry_strategy = "MANUAL"`. The sleeve exists since 2026-09-29. Older files contain `close_reason = MANUAL`, but those are manual **closes of bot trades**, not manual entries.

**Sources scanned:**
- all 89 `Downloads/scalpars_orders_paper_*.csv`;
- every `reports/*.csv` except `*.superseded_bak` / `*_bak`;
- `MASTER_POOL_stacked.csv`, which holds no MANUAL rows;
- the local DBs, which have no `entry_strategy`.

972 MANUAL rows were found in 44 files. After dedup on `(opened_at, pair, direction)`, CLOSED only (never `id`), **126 fills** remain. No near-duplicates (same pair and side within 5 s). Timestamps are UTC.

| | N | WR | avg P&L % (net) | $ |
|---|---|---|---|---|
| All | 126 | 65.1 % | +0.11 | +2,462 |
| LONG | 95 | 63.2 % | +0.12 | −1,616 |
| SHORT | 31 | 71.0 % | +0.09 | +4,078 |
| 2026-09 (09-29, 09-30) | 25 | 52 % | −0.12 | |
| 2026-10 (10-01, 02, 03, 06) | 101 | 68 % | +0.17 | |
| by day: 09-29 · 09-30 · 10-01 · 10-02 · 10-03 · 10-06 | 15 · 10 · 53 · 23 · 22 · 3 | 47 · 60 · 72 · 78 · 45 · 100 % | −0.19 · −0.01 · +0.19 · +0.19 · −0.07 · +1.45 | |

- The $ and % disagree in sign for longs because leverage and size varied. Per CLAUDE.md, % is the comparable measure.
- **The earlier study** (`MANUAL_VS_BOT_FRENZY_2026-10-06.md`) had 123 of these rows plus today's 3, so it missed no fill. But it analysed only the **95 longs**; the **31 shorts were never characterised**.
- There are no manual fills on 10-04 or 10-05; none of those exports holds a MANUAL row.
- **Concentration:** MOVR 57 fills (53 on 10-01, in a 3 h session), SAND 17, AIN 13, QNT 6. That is effectively about 4 market windows.

## 2. Context rebuild and parity

**What was rebuilt for every fill** (last CLOSED 5m bar before the click; no look-ahead):
- **The real `services.frenzy` functions:** `frenzy_walk`, `frenzy_flagged`, `frenzy_long_status`, `frenzy_di_spread`, `frenzy_adx_delta`, `frenzy_vol_trend`, `global_volume_ratio`, plus `services.surge.wilder_atr_pct`. Thresholds come from the live `trading_config.json`.
- **Walk history** over the 72 bars before the click: vol-multiple slopes over 3, 6 and 12 bars; VWAP-distance slope; below-run before the current streak; bars since the last state.
- **Levels:** EMA 5/8/13/20/50 levels, gaps and slopes; RSI14; ADX, ±DI and their deltas; returns over 1 h, 4 h and 24 h; distance off the 1 h and 4 h highs; 24 h quote volume.
- **BTC:** 1 h, 4 h and 24 h returns; gap 20-50; RSI.
- **The bot's own `entry_*` stamps**, which 86 % of manual fills carry.

**Four market-volume versions per fill:**
- **Engine FRENZY:** base-unit, top-50 by 24 h quote volume, 48 bars.
- **Quote-dollar.**
- **Median-of-pairs.**
- **Dashboard "Vol":** Σ EMA5(base vol) ÷ Σ SMA48, closed-bar reconstruction.

**Outcomes per fill:** MFE and MAE at 30, 60 and 120 min from 1m bars, plus every fill re-priced on standard exits: LOCK, +3/−3, +1/−3, and +3/−3 with a 60-min time stop.

**Parity and validation**

| check | result |
|---|---|
| Bot FRENZY fills, 10-03 → 10-06 (12 fills), recomputed at the signal bar vs the stamped `entry_frenzy_*` | **12/12 exact** on hours, VWAP, vs-VWAP, run %, bar return, DI spread, ADX Δ, vol trend and above-share (where stamped). Vol multiple within 0.1–1.6 % (ccxt vs native kline volume). Engine gvol exact where computable (AIN 0.8055; ORCA 0.7802 in the earlier study). |
| Today's 3 manual fills vs the earlier study | identical (GRIFFAIN 72.5×, +0.79 % vs VWAP, below-run 11; EDU 27.6× / 23.6×; gvol 0.545 / 0.462 / 1.621) |
| Year walk vs `FRENZY_ENGINE_COHORT_2026-10-05.csv` | the year walk is the real `frenzy_walk` on **every** bar that can belong to a flagged episode: 1,372,274 flagged bars on 589 pairs, 792,679 in the eligible universe. **1,819 / 1,819** cohort fresh bars are present; hours, vol_mult, vs_vwap and above_share are identical. |
| Engine gvol, year | my rolling top-50 version equals the per-fill values (123/123 within 1 %). `gvr_year.pkl` (the older "day's top-50" variant) agrees only at Spearman 0.86. |
| Dashboard gvol reconstruction vs the stamped `entry_global_volume_ratio` | Spearman **0.84**, level ×1.17. The live stamp reads the forming bar at scan time. The stamp is used directly where present (86 %). |
| 1m pricer vs real ticks | on the 249 tick-priced FRENZY_LONG fills of the engine cohort: LOCK by ticks **+0.311**. 1m bars with path order (green bar O→L→H→C, red bar O→H→L→C) **+0.330**, corr 0.87. 1m fully pessimistic: **−0.113**, because it cuts every runner at the arming minute. FIX +3/−3: ticks +0.081, 1m +0.099 under both orders. **Primary = path order; pessimistic shown as a sensitivity.** |

**Could not be tested / low fidelity:**
- **aggTrades for 09-29 → 10-03.** Binance fapi limits time-window aggTrades search to the last 2 days, so the 60 s move and taker share use 1m proxies: the previous full minute's move and its taker share.
- **Order-book stamps.** They cover only 4 % of fills.
- **`on_age_min`** (56 % coverage) and **`bars_since_below`** (28 %) were not screened.
- **Taker share of the RECLAIM_DIP trigger minute** is not in the cached 1m data.

## 3. Winners vs losers

**Entry or exit?**

| | winners (82) | losers (44) |
|---|---|---|
| median actual P&L % | +0.60 | −0.83 |
| median MFE / MAE 60 min | +3.39 / −3.12 | +1.81 / −5.30 |
| mean P&L % if held on +1/−3 | +0.60 | −0.78 |
| mean P&L % if held on LOCK | +0.51 | −1.63 |

- All 126 on +1/−3 average **+0.12 %**, against **+0.11 %** actual.
- WR on +1/−3 is 77 %, against a **75.1 %** breakeven after costs.
- 23 of the 44 losers would have won on +1/−3: the hand close came before +1. Eight winners would have lost.
- **Neither the entries nor the exits carry an edge in aggregate.** The bracket and the hand produce the same mean.

**Styles** (rule-based classification of the context at the click):

| style | N | pairs | days | WR | actual % | +1/−3 % | +3/−3 % | LOCK % | MFE60 / MAE60 (median) |
|---|---|---|---|---|---|---|---|---|---|
| ON_CHASE long (in FRENZY state) | 38 | 3 | 3 | 68 % | +0.28 | +0.26 | +0.32 | −0.15 | +3.7 / −7.7 |
| ON_CHASE short | 27 | 2 | 2 | 78 % | +0.25 | +0.70 | +0.11 | −0.31 | +3.2 / −3.3 |
| UNFLAGGED long (momentum-sleeve territory) | 18 | 10 | 2 | 50 % | −0.13 | −0.50 | −0.81 | −0.98 | +0.8 / −1.4 |
| BELOW_VWAP_DIP long | 14 | 8 | 5 | 71 % | +0.14 | +0.14 | 0.00 | −0.36 | +1.9 / −2.4 |
| EARLY_ABOVE_VWAP (AIN 10-03, < 1.5 h after the spike) | 13 | 1 | 1 | 46 % | −0.20 | −0.54 | −0.69 | −0.39 | +19.6 / −9.0 |
| **ABOVE_VWAP_NOT_ON** (held above, vol < 100× or < 2 h) | **9** | 6 | 3 | 78 % | +0.38 | +0.56 | **+2.33** | **+2.17** | +5.8 / −2.5 |
| VWAP_RECLAIM (streak 1–4 after ≥ 6 below) | 3 | 3 | 2 | 67 % | +0.12 | −0.33 | −1.00 | −1.33 | +1.4 / −4.2 |
| UNFLAGGED short | 3 | 3 | 1 | 33 % | −0.23 | | | | |
| BELOW_VWAP short | 1 | 1 | 1 | 0 % | −3.01 | | | | |

**Separator screen** (`manscreen.py`, `manscreen2.py`):
- **Setup:** Welch t of bucket vs rest; family-wise shuffled-label null (max |t|, 300 permutations); same sign in both time halves required.
- **Outcomes:** win, P&L %, and +1/−3 entry quality. Run on all 126 fills and on the 95 longs.
- **Family 1** (82 features: 1D at sign / tercile / quintile = ~700 tests; exhaustive 2D tercile pairs = ~20,600 tests): **0 survivors in every 1D screen**. In 2D, only the entry-quality outcome produced survivors:
  - longs: **gvol_quote T1 ∧ gvol_median T1** (low market volume), 27 fills all +1, p_fw 0.027;
  - longs: gvol_quote T1 ∧ BTC 1 d T3, p 0.037;
  - all fills: px_vs_ema13 T1 ∧ off_4h_high T1 (dip buys), p 0.05.
  - The low-volume cell spans 6 days and 12 pairs (MOVR 15). It was frozen as filter **LOWMKT** (gvol_quote ≤ 0.63 ∧ gvol_median ≤ 0.56) **before** the year test (§6).
- **Family 2, momentum dynamics** (operator addendum, 43 features, separately nulled: RSI Δ1/3/6 and up-crosses of 50/55/60; ADX Δ; ±DI Δ; DI cross within 3/6 bars; gaps 5-8, 5-13, 5-20, 8-20, 13-50 and 20-50 with level, slope, sign and turn flag; EMA slopes; price vs EMA20): **0 survivors** on every outcome. The "turning-gap" signature (gap5-20 < 0 and rising, gap5-8 > 0, RSI and ADX rising) matched **0 of 95** manual longs.
- **Honest read:** N = 126 over about 4 market windows. The screens are mostly noise. LOWMKT is the one thing worth carrying to the year.

## 4. Frozen candidate specs (before the year was read)

The older half is the first 63 fills (09-29 21:38 → 10-01 17:36). It holds:
- 20 MOVR ON-state longs;
- 18 MOVR ON-state shorts;
- 21 unflagged momentum-style fills (the momentum sleeve's territory, so no new candidate);
- 4 others.

Full text is in the PREREG files.

| candidate | bar rule (closed 5m, flagged, eligible universe¹) | entry | primary exit |
|---|---|---|---|
| **A ON_DIP_LONG** | in state | first 1m red close ≥ 1 % under the 5-min high, in the 5 min after the bar; ≤ 1 per pair per 60 min (first-per-stretch also reported) | +1/−3 |
| **B ON_FADE_SHORT** | in state ∧ vs VWAP ≥ +15 % ∧ RSI14 < 50 ∧ close < EMA8 | short at the next 1m open; ≤ 1 per pair per 60 min | +1/−3 |
| **C HOLD_LOWVOL_LONG** (exploratory, suggested by the newer half) | not in state ∧ ≥ 12 closes above the spike VWAP ∧ vol < 100× ∧ hours ≥ 2 | next 1m open, first bar per above-stretch | LOCK |
| **D TURNING_GAP_LONG** (operator addendum) | gap5-20 < 0 and rising ∧ gap5-8 > 0 ∧ RSI Δ1 > 0 ∧ ADX Δ1 > 0 | next 1m open; ≤ 1 per pair per 60 min | LOCK |
| **RECLAIM_DIP_OP** (operator's literal bands from today's 3 fills; in-sample only, 10-06 excluded from the year) | not in state ∧ 3–6 h ∧ below-run 12–18 ∧ streak 1–4 ∧ vol 24–72× ∧ EMA5 > EMA8 ∧ RSI 54–61 | first 1m bar with a −0.7…−0.3 % body in the next 5 min → next 1m open | LOCK |
| **RECLAIM_DIP_V1** (`MANUAL_VS_BOT` spec, unchanged) | not in state ∧ ≥ 2 h ∧ streak 1–4 ∧ below-run ≥ 6; first bar per below-stretch | limit at close × 0.997 for 10 min (primary); LAG0 / LAG1 market | LOCK |
| RECLAIM grid (declared family) | hours {3–6, 2–8} × below-run {≥ 6, ≥ 12, 12–18} × vol {24–72, 20–100} × RSI {54–61, 50–65} × dip {−0.7…−0.3, −1.0…−0.2} = 48 cells | as OP | LOCK |

¹ Eligible universe: coin-underlying, listed ≥ 90 d, non-Alpha, 24 h quote volume ≥ $20M, live blacklists.

**Costs:** 0.09 % fees and 0.02 % slippage on market entries and stop-type exits. Year = 2026-01 → 2026-10-03, the `k5m_full` extent.

**Recall and precision**, newer half inside the cache (10-01 17:40 → 10-03):

| candidate | manual fills matching | rule's own signals in that window | LOCK on those signals |
|---|---|---|---|
| A | 18 of 63 newer-half fills match its bar state | 53 signals, 51 on pairs he traded | −0.25 |
| B | 2 | 9 | −0.72 |
| C | 3 (older half: 1) | 33, 17 on pairs he traded | −0.38 |
| D | 0 | 25 | −1.76 |
| RECLAIM_DIP_OP | **0 of his fills, including today's 3** (below-run 11, vol 72.5×, hours 2.92, vol 23.6×) | | |
| RECLAIM_DIP_V1 | today's 3 only (actual +3.11 / +1.00 / +0.24; LOCK +2 / +2 / −3) | | |

## 5. Year backtest (1m, path-order primary; % of position at 1×)

| candidate · exit | N | days | WR | BE WR | avg % | day-CI 95 % | Jan–Apr / May–Oct | LOMO min…max | months > 0 | null mean · p | pessimistic avg | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A ON_DIP_LONG · +1/−3 | 5506 | 266 | 71.7 | 75.1 | **−0.138** | [−0.19, −0.09] | −0.15 / −0.13 | −0.16…−0.12 | 1/10 | −0.10 · 1.00 | −0.15 | fail |
| A · LOCK | 5506 | 266 | 48.1 | 48.4 | −0.021 | [−0.11, +0.08] | −0.07 / +0.01 | | 4/10 | −0.04 · 0.11 | −0.34 | fail |
| A first-per-stretch · LOCK | 1638 | 265 | 48.6 | 47.9 | +0.046 | [−0.14, +0.22] | +0.08 / +0.02 | −0.01…+0.11 | 6/10 | −0.04 · 0.13 | −0.34 | fail |
| B ON_FADE_SHORT · +1/−3 | 746 | 130 | 68.9 | 75.1 | **−0.250** | [−0.37, −0.13] | −0.21 / −0.27 | | 0/10 | −0.17 · 0.93 | −0.29 | fail |
| B · LOCK | 746 | 130 | 47.3 | 46.7 | +0.041 | [−0.22, +0.31] | −0.20 / +0.13 | | 5/10 | −0.27 · 0.002 | −0.28 | fail (H1 < 0, CI spans 0) |
| **C HOLD_LOWVOL_LONG · LOCK** | 3278 | 267 | 50.0 | 49.1 | **+0.054** | [−0.08, +0.20] | +0.07 / +0.04 | +0.03…+0.08 | 8/10 | **+0.08 · 0.86** | −0.17 | fail (CI, null) |
| C · +3/−3 | 3278 | 267 | 50.0 | 50.1 | −0.006 | [−0.12, +0.10] | +0.03 / −0.04 | | 5/10 | | −0.01 | fail |
| D TURNING_GAP_LONG · LOCK | 2364 | 268 | 46.3 | 49.7 | **−0.197** | [−0.33, −0.05] | −0.28 / −0.12 | | 0/10 | −0.15 · 0.83 | −0.37 | fail |
| RECLAIM_DIP_V1 LIMIT · LOCK | 3659 | 267 | 47.4 | 49.5 | **−0.129** | [−0.24, −0.02] | −0.29 / −0.00 | | 2/10 | −0.01 · 1.00 | −0.39 | fail |
| V1 LAG0 / LAG1 · LOCK | 4433 | 267 | 47 | 49 | −0.133 / −0.155 | [−0.25, −0.01] / [−0.27, −0.03] | | | 3 / 2 of 10 | ≈ 0 · 1.00 | | fail |
| V1 LIMIT · +1/−3 | 3659 | 267 | 72.5 | 75.1 | −0.104 | [−0.16, −0.04] | −0.19 / −0.04 | | | | | fail |
| **RECLAIM_DIP_OP · LOCK** | **19** | 19 | 31.6 | 55.9 | **−1.31** | [−2.44, −0.15] | −0.59 / −1.97 | | 3/9 | 0.00 · 0.96 | −1.31 | fail (N < 30, negative) |
| RECLAIM grid, 48 cells (695 distinct trades) | 19–483 | | | | best cell +0.308 (N 234; Jan–Apr +0.79, May–Oct −0.02) | best CI [−0.07, +0.71] | | | 19 of 48 cells > 0 | best-of-family null 95 % = +1.63, **p 1.0** | | fail |

**How to read the null column.** Random 1m entries on bars of the same flagged episodes, in the same state class, with the same fill model and exits. RECLAIM_DIP_V1 is **worse than a random entry** on the same episodes (random ≈ 0.00, V1 −0.13): the reclaim timing loses money.

**How this differs from the earlier refutations:**
- `FRENZY_BELOW_AVERAGE_TEST` RECLAIM: −0.17…−0.31 under 0.21 % costs, hand-rolled walk. This study uses the engine walk on every bar, 0.11 % costs, the micro-dip trigger, the lock exit, a 6+ below-run, and limit / LAG0 / LAG1 entries. **It is still negative, so the cost and trigger changes did not rescue it.**
- `FRENZY_REENTRY_WHILE_ON` (re-entry while ON −0.09…−0.20): A adds a 1 % micro-dip trigger and +1/−3 exits. Still negative.
- `FRENZY_ON_SCALP`: fresh ON bar only.
- `FRENZY_SHORT_TEST`: any-minute shorts. B adds a VWAP-stretch plus RSI/EMA8 roll-over. Still negative.
- Hot-scalp memory: the pullback-depth separator is A's trigger, and it does not pay.
- The parallel rising-volume pre-ON study's pre-registered test was **not** duplicated. Its features (vol-multiple slopes, VWAP-distance slope) are only carried as screen features.

## 6. Filters

**Pre-registered LOWMKT** (gvol_quote ≤ 0.63 ∧ gvol_median ≤ 0.56, frozen from the manual screen): it **fails on every candidate**.

| candidate · exit | kept N / days | kept avg | Jan–Apr / May–Oct | blocked avg | p vs random subset |
|---|---|---|---|---|---|
| C · LOCK | 702 / 235 | +0.183 | +0.45 / **−0.02** | +0.02 | 0.15 |
| A first · LOCK | 245 / 163 | −0.04 | +0.16 / −0.18 | +0.06 | 0.66 |
| B · LOCK | 146 / 70 | −0.54 | | +0.18 | 0.99 |
| D · LOCK | 444 / 211 | −0.18 | | −0.20 | 0.43 |
| V1 LIMIT · LOCK | 650 / 227 | −0.29 | | −0.10 | 0.93 |

The 29/29 manual cell was a property of that week's tape (mostly MOVR 10-01), not of low market volume.

**Full sweep** on the three near-zero candidates (C, A first-per-stretch and B, all on LOCK). Method:
- ~77 features each: every engine walk feature, the 4 gvol versions, the momentum family, BTC 1 h / 4 h / 24 h return and trend gap;
- 1D at sign / tercile / quintile (~710 tests) plus exhaustive 2D tercile pairs (~24,000 tests);
- 200-permutation shuffled-label family null;
- a survivor must also be same-sign in both halves and span ≥ 8 days;
- blocked cohorts are judged against the sleeve breakeven WR with a day-clustered bootstrap.

**Result: 0 survivors for all three candidates.**
- An earlier pass without the BTC features had one survivor in C: a 2D cell "late hours ∧ middle 1 h return", |t| 4.83 vs null95 4.79. It disappears in the larger family, and it is non-monotone, so it is a confound by the CLAUDE.md rule.
- The nearest 1D read is C by hours since the spike: last tercile −0.27 vs +0.22, p_fw 0.055, same sign in both halves.

**Not tested in the year sweep:**
- order-book and aggTrades features (no year data);
- taker share at trigger minutes (not in the cached 1m data);
- funding;
- any `entry_*` live stamp that has no kline equivalent: `bull_pct` breadth, mcap and CMC rank.

## 7. The one lead: C restricted by hours (post-hoc, observe only)

| C HOLD_LOWVOL_LONG with hours ≤ 16.75 (cut = the upper tercile of the Jan–Apr fills only) | value |
|---|---|
| N · days · pairs | 2239 · 267 · 362 |
| LOCK avg · day-CI | **+0.215** · [+0.03, +0.41] |
| Jan–Apr (cut fitted here) / May–Oct (out of sample) | +0.16 / **+0.25**; May–Oct with hours > cut: −0.44 |
| LOMO · months > 0 | +0.17…+0.26 · 8/10 (Apr −0.08, partial Oct −0.81) |
| top pair / top day share of gain | 2 % / 4 % |
| +3/−3 · +1/−3 · T60 | +0.125 [−0.02, +0.26] · −0.02 · +0.05 |
| **pessimistic 1m ordering** | **−0.014** (LOCK); +0.125 (+3/−3) |
| null (random above-VWAP non-state bars, same hours, same episodes) | 95 % quantile +0.40, **p 0.81** |
| 30–50 % in-sample haircut | +0.11…+0.15 |

Context: by hours bucket, 2–4 h +0.03, 4–8 h +0.17, **8–12 h +0.47**, 12–18 h −0.09, 18–24 h −0.39. Low ATR (≤ 1 %) +0.75. vol < 20× +0.35.

**Reading:**
- This is a **state** effect: "a flagged pair holding above its spike VWAP without FRENZY-level volume, roughly 4–16 h after the spike". It is not a timing edge. Random bars in the same state earn as much.
- It fails the pre-committed bar on two counts: it does not beat the null, and the hours cut is post-hoc.
- It is also fragile to intrabar ordering on the LOCK exit. +3/−3 is ordering-robust at +0.125, but its CI spans 0.
- It is the only place where the operator's newer-half "above VWAP, not ON" fills (SAND, STRK, ATH, PUMPBTC: 9 fills, +2.17 LOCK) line up with a year-positive cohort.

## Verdict

| candidate | ≥ 30 fills, ≥ 8 days | both halves > 0 | day-CI > 0 | no concentration | beats null | result |
|---|---|---|---|---|---|---|
| A ON_DIP_LONG | ✓ | ✗ | ✗ | ✓ | ✗ | **fail** |
| B ON_FADE_SHORT | ✓ | ✗ | ✗ | ✓ | (LOCK p 0.002, but itself ≈ 0) | **fail** |
| C HOLD_LOWVOL_LONG | ✓ | ✓ | ✗ | ✓ | ✗ | **fail** (near-zero) |
| D TURNING_GAP_LONG | ✓ | ✗ | ✗ | ✓ | ✗ | **fail** |
| RECLAIM_DIP_OP | ✗ (19) | ✗ | ✗ | ✓ (25 % top pair/day) | ✗ | **fail**, family closed |
| RECLAIM_DIP_V1 (limit / LAG0 / LAG1) | ✓ | ✗ | ✗ | ✓ | ✗ (worse than random) | **fail**, family closed |
| RECLAIM grid (48) | — | — | — | — | best-of-family p 1.0 | **fail** |
| Filters (LOWMKT + ~25k swept rules × 3 candidates) | | | | | | **0 survive** |

**What to do next:**
1. **Arm nothing.** No candidate passes. The RECLAIM_DIP family (operator literal, v1, 48-cell grid) is closed on this data: no v2 on the same year.
2. **One free scout observe line (proposal; needs operator approval and a review before it is built): `HOLD_LOWVOL_EARLY`.**
   - **Bar:** a flagged, eligible pair, not in state, with `above_streak ≥ 12`, `vol_mult < 100` and `2 ≤ hours ≤ 16.75`.
   - **Entry:** first bar of each above-stretch, next-open long.
   - **Exits:** paper LOCK and +3/−3, both logged.
   - **Thresholds are frozen as written.** Promote-review needs **≥ 40 forward fills on ≥ 15 days with mean LOCK ≥ +0.10 %**, and it must still be positive on tick re-pricing. Retire if the mean is < 0 at 40.
   - **Before any build:** re-price the 2,239 year fills on real ticks. 1m path vs pessimistic ordering moves LOCK by 0.23 pts.
3. **Do not use manual clicks as the research path.** Per the operator, the goal is automated rules. The manual record was a hypothesis source, and its main lesson is that the profitable-looking manual cohorts (MOVR ON-chase, LOWMKT) came from one week's tape.
4. **Data caveat for future gvol work.** The engine's base-unit gvol and the quote / median / dashboard versions rank-correlate only 0.5–0.7 on these fills. The largest single pair holds ~49 % of top-50 base volume on an average bar (`gvol_top1_share`). Any gvol filter should be specified on the quote or median version. See the separate formula-comparison study.
