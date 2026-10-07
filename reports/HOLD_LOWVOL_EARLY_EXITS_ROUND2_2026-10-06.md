# HOLD_LOWVOL_EARLY: exit study round 2 (2026-10-06)

## Resumen en lenguaje simple

- **Pregunta del operador.** Con los mismos 2,239 trades de HOLD_LOWVOL_EARLY (sin cambiar ninguna entrada), ¿alguna de estas salidas los vuelve rentables?
  1. el lock actual de FRENZY;
  2. un lock parecido pero que se arma en +1 % (en vez de +3);
  3. la salida de BULLRUN, con los parámetros reales del bot.
- **Cómo lo medí.** Con ticks reales y el timing real del bot: entrada 12 s después del cierre de la vela, 0.10 % de slippage en la salida, 0.09 % de comisiones. Son 2,198 trades.
  - Antes de mirar ningún resultado dejé escrita la lista de 15 salidas y las reglas para aprobar.
  - Elegí las mejores solo con Ene–Abr y las confirmé en May–Oct, que no se tocó al elegir.
- **Respuesta: ninguna salida lo arregla. Todas las salidas nuevas quedan PEOR que el lock actual.**

| salida (timing real) | ganancia media por trade | trades ganadores | ¿mejor que el lock? |
|---|---|---|---|
| **Lock FRENZY actual** (−3, se arma en +3, piso +2, trail 2) | **+0.045 %** | 53 % | — |
| Armado en +1 (6 variantes) | **−0.08 … −0.11 %** | 35–75 % | no; las 6 pierden plata |
| Armado en +1.5 / +2 (4 variantes) | −0.05 … +0.01 % | 49–63 % | no |
| BULLRUN real (stop −0.7…−1.2, se arma en +1, trail 2×ATR) | **−0.18 %** | 50 % | no; la peor, 0 de 10 meses positivos |
| BULLRUN con stop −3 (modo manual "BULLRUN_SL") | −0.08 % | 74 % | no |

- **Por qué el "armado en +1" gana más seguido pero pierde plata.**
  - Gana 3 de cada 4 trades, pero cada ganancia es chica: unos **+0.9 %**.
  - Las pérdidas siguen siendo de **−3 %**: 1 de cada 4 trades nunca llega a +1 y cae hasta el stop.
  - Con el lock actual las ganancias promedian **+2.9 %**, porque deja correr a los que suben.
  - Armar temprano corta justo esos ganadores grandes. El universo depende de unos pocos días de pump fuerte.
- **La salida BULLRUN no sirve acá.** Su stop chico (−0.7 a −1.2 %) se toca enseguida en estas monedas, que se mueven mucho. Promedio −0.18 %, peor en todos los meses.
- **Confirmación May–Oct.** Las 3 finalistas (armado en +2) quedan por debajo del lock: −0.04 a −0.06 %/trade. Ninguna pasa.
- **Veredicto (barra declarada de antemano): NO PASA. Se cierra HOLD_LOWVOL_EARLY definitivamente.** No hay línea observe ni propuesta de arm.
  - La barra pedía +0.10 %/trade con el límite inferior del intervalo por encima de cero.
  - El lock actual da +0.045 con intervalo [−0.13, +0.23].
  - Con un máximo de 5 posiciones abiertas da −0.01.
  - Si se quitan los 5 mejores días da −0.06.
- **Una advertencia honesta.** Mirando las 47 salidas probadas en las dos rondas, una del round 1 (stop −4, se arma en +3, trail 2) da +0.16 con el timing real.
  - Nadie la eligió de antemano: es la mejor de 47 elegida después de ver los resultados.
  - Su ventaja sobre el lock en May–Oct (+0.08) queda muy por debajo de lo que la suerte sola produce eligiendo la mejor de 47 (+0.20).
  - **No es evidencia.** No se propone.

---

_(Summary added after pricing. The pre-registration below was written into this file at 2026-10-06T23:16:40Z, before any round-2 cell was priced.)_

## Pre-registration (verbatim, `scratchpad/holdlow_exits2/PREREG_ROUND2.txt`)

```
PRE-REGISTRATION — HOLD_LOWVOL_EARLY exit study ROUND 2 (operator request). Written 2026-10-06T23:16:12Z, BEFORE any round-2 cell is priced.
This is ROUND 2 on the SAME fills: round 1 (PREREG_SECONDARY_EXITS.txt, 32 cells) already looked at these fills and failed OOS.
Multiplicity is therefore 32 + 15 = 47 exit cells on one fill list; the confirmation bar below includes a best-of-47 null.

FILLS (fixed, no entry change): FREEZE_HOLD_LOWVOL_EARLY_fills.csv, sha256 4b2ae2d40c81b7aca4381a3c2ecabecc4f544234ee9407907d4d5dac648d73dd
(2,239 rows; 2,236 have ticks). Tick pricer = holdlow_ticks/ticklib.py (validated 958/958 exact vs the engine cohort).

RULERS
 PRIMARY (live timing, = round-1 "live ruler"): entry = first aggTrade print >= signal close + 12 s, no entry slip, exit at the crossing
   print - 0.10 %, fees 0.09 round trip, levels on NET %, arm on the PRIOR-print peak, 12 h cap, 1 % dislocation guard (fills refused by
   the guard are dropped for every cell alike; round 1: 38 refused -> N 2,198).
 SECONDARY (next open): entry = first print >= signal close, +0.02 % entry slip, exit -0.02 %, same fees/levels/cap.

FAMILY (15 cells). Notation A{arm}_F{floor}_D{trail}: stop -3 until the prior-print peak >= arm, then line = max(F, peak - D). All NET %.
  1  LOCK            = live FRENZY lock (services/frenzy.py frenzy_exit_for, trading_config: stop 3, lock_arm 3, floor 2, trail 2) = A3_F2_D2. BASELINE.
  6  A1_F0_D1, A1_F0_D1.5, A1_F0_D2, A1_F0.5_D1, A1_F0.5_D1.5, A1_F0.5_D2          (armed at +1; F=0 = break-even incl. fees)
  4  A1.5_F0_D1.5, A1.5_F0_D2, A2_F0_D1.5, A2_F0_D2                              (armed at +1.5 / +2, break-even floor)
  1  BULLRUN         = live _bullrun_exit_for (GREEN door) with trading_config: SL = min(-0.7, max(-1.5 x ATR, -1.2)) [bullrun_base_sl -0.7,
                       sl_atr_multiplier 1.5, sl_atr_widen_floor -1.2]; at peak >= +1.0 (be_arm) line = max(+0.2 (be_lock), peak - 2.0 x ATR,
                       ladder floor 4:3.5,5:4.5,6:5.5,8:7,10:9,12:11,15:13.5,20:18,25:22.5,30:27). ATR = entry 5m ATR14 % (atr5; missing -> 2.0).
                       12 h cap like every cell (live sleeve MAX_HOLD differs; declared simplification).
  3  NEIGHBOURS declared up front:
     N1 BULLRUN_SL3  = the live MANUAL "BULLRUN_SL" mode with the operator's -3 stop: -3 until peak >= +1.0, then the BULLRUN profit side unchanged.
     N2 A2_F1_D2     = the FRENZY lock geometry shifted one notch down (arm 2, floor = arm - 1, trail 2).
     N3 A1.5_F0.5_D1.5
 For the best-of-47 null the 32 round-1 cells (24 TRAIL_s{S}_a{A}_t{T}, 3 ATRk, BE15_T2, BULLRUN [same as cell above], 2 TIME overlays
 [on round-1's #1 TRAIL_s-4_a3_t3: 4 h and 8 h]) are re-priced on the PRIMARY ruler too (31 distinct + BULLRUN shared = 46 distinct cells).

SELECTION / CONFIRMATION (primary ruler)
 Discovery = fills with entry < 2026-05-01 (Jan-Apr). Rank the 14 non-baseline round-2 cells by Jan-Apr mean %/trade; finalists = top 3.
 Confirmation on May-Oct (not used for ranking): finalist - LOCK per-fill paired difference; day-block bootstrap 95 % CI (2,000, seed 0).
 PASS (exit change) = #1 finalist has ALL of: (a) May-Oct paired delta CI lower bound > 0; (b) May-Oct mean of the cell > 0;
   (c) May-Oct paired delta > the 95th percentile of the best-of-46 shuffled null: per draw, flip the sign of every fill's (cell - LOCK)
   difference by DAY (one random sign per day, same flips for all cells), take the max May-Oct mean delta over all 46 non-baseline cells
   (round 1 + round 2); 2,000 draws, seed 0.
 Finalists #2/#3 are reported but cannot rescue a #1 failure. Haircut 30-50 % on any delta.

PER CELL REPORT: N, WR, avg %/trade, day-block 95 % CI, Jan-Apr / May-Oct, months positive, top pair / day share of gain, hold time
 (mean / median minutes), two-sided vs LOCK (improved / worsened >0.05, saved losers LOCK<0 & cell>0, killed winners LOCK>0 & cell<=0,
 sum of gains vs sum of losses), and mean under a 5-position cap (chronological, first-come, slot held until that cell's exit).

VERDICT BAR (universe + exit, free scout observe line): on the PRIMARY ruler, avg >= +0.10 %/trade AND day-CI lower bound > 0 AND May-Oct > 0.
 It is applied to (i) LOCK and (ii) the #1 finalist ONLY if that finalist passed the confirmation gate above (a full-period number of a cell
 picked best-of-15 is not admissible otherwise). If neither meets the bar -> close HOLD_LOWVOL_EARLY for good. No arm proposal in any case.
```

## 1. Setup and parity

- **Fills:** the frozen list, unchanged (sha256 `4b2ae2d4…d73dd`). Primary (live) ruler: 2,198 priced; 38 refused by the 1 % dislocation guard (same as round 1). Secondary (next-open): 2,236.
- **Pricer:** `holdlow_ticks/ticklib.py`, which was validated 958/958 exact against the engine cohort. Script: `scratchpad/holdlow_exits2/price2.py`, output `priced2.pkl`. Analysis: `analyze2.py` → `analyze2.out`, `eval2_{P,S}.csv`, `months2_{P,S}.csv`.
- **Parity, max abs diff 0.0 on every check:**
  - round-2 LOCK on the live ruler = round-1 `LOCK_live` (2,198 fills);
  - next-open LOCK, BULLRUN, TRAIL_s-4_a3_t3, ATRk_k2, BE15_T2, TRAIL_s-2_a1_t1 and the 4 h overlay = round-1 `sec_base` / `sec_overlay`.
- **Live params verified in code and config:**
  - `services/frenzy.py:frenzy_exit_for` (use_tp branch) with `frenzy_stop_pct` 3, `frenzy_lock_arm_pct` 3, `frenzy_lock_floor_pct` 2, `frenzy_lock_trail_pct` 2.
  - `services/trading_engine.py:_bullrun_exit_for` (GREEN door) with `bullrun_base_sl_pct` −0.7, `sl_atr_multiplier` 1.5, `sl_atr_widen_floor_pct` −1.2, `bullrun_be_arm_pct` 1.0, `bullrun_be_lock_pct` 0.2, `bullrun_trail_atr_mult` 2.0 and `bullrun_ladder` as in the pre-registration.
  - BULLRUN_SL3 = `manual_bullrun_exit_for` with custom_sl −3.
- **Declared simplification:** BULLRUN uses the 5m entry ATR and the 12 h cap. The live sleeve's MAX_HOLD is not applied.

## 2. Primary results: live timing (12 s, exit −0.10, fees 0.09), N = 2,198

Columns:
- Δ = cell − LOCK, paired per fill, with a day-block CI.
- Cap 5 = mean per trade when at most 5 positions are open (chronological, first-come); "skip" = share of trades not taken and their mean.
- Hold = mean / median minutes.

| cell | WR | avg | day-CI 95 % | Jan–Apr | May–Oct | months > 0 | top pair / day share | hold | cap 5 (skip %, skipped mean) | Δ all [CI] | Δ May–Oct [CI] | improved / worsened | saved losers / killed winners | Σgain / Σloss vs LOCK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LOCK (baseline)** | 52.6 | **+0.045** | [−0.129, +0.229] | −0.024 | +0.093 | 5/10 | 1.8 % / 4.7 % | 90 / 44 | −0.011 (1.8 %, +3.07) | — | — | — | — | — |
| A1_F0_D1 | 67.3 | −0.099 | [−0.188, −0.017] | −0.150 | −0.064 | 2/10 | 2.0 / 2.5 | 36 / 15 | −0.105 | −0.145 [−0.30, −0.01] | −0.158 [−0.32, −0.01] | 649 / 979 | 417 / 95 | +1852 / −2170 |
| A1_F0_D1.5 | 46.1 | −0.100 | [−0.201, −0.002] | −0.164 | −0.055 | 3/10 | 2.3 / 3.9 | 41 / 18 | −0.114 | −0.145 [−0.28, −0.02] | −0.149 [−0.30, −0.00] | 738 / 893 | 207 / 351 | +1735 / −2054 |
| A1_F0_D2 | 35.3 | −0.082 | [−0.191, +0.030] | −0.138 | −0.044 | 2/10 | 2.5 / 5.0 | 45 / 21 | −0.102 | −0.128 [−0.25, −0.02] | −0.137 [−0.27, −0.01] | 564 / 824 | 103 / 484 | +1658 / −1939 |
| A1_F0.5_D1 | 74.8 | −0.110 | [−0.191, −0.030] | −0.160 | −0.076 | 2/10 | 2.2 / 2.7 | 35 / 14 | −0.113 | −0.155 [−0.30, −0.02] | −0.169 [−0.33, −0.01] | 624 / 1010 | 494 / 7 | +1893 / −2235 |
| A1_F0.5_D1.5 | 74.7 | −0.108 | [−0.197, −0.027] | −0.178 | −0.060 | 2/10 | 2.6 / 2.6 | 36 / 15 | −0.112 | −0.154 [−0.30, −0.02] | −0.154 [−0.31, +0.00] | 667 / 970 | 494 / 9 | +1844 / −2182 |
| A1_F0.5_D2 | 74.7 | −0.096 | [−0.191, −0.004] | −0.174 | −0.043 | 2/10 | 2.7 / 3.3 | 39 / 16 | −0.102 | −0.141 [−0.28, −0.01] | −0.136 [−0.27, +0.01] | 540 / 942 | 494 / 9 | +1807 / −2118 |
| A1.5_F0_D1.5 | 62.8 | −0.051 | [−0.169, +0.064] | −0.131 | +0.005 | 3/10 | 1.9 / 3.8 | 54 / 25 | −0.065 | −0.096 [−0.20, +0.01] | −0.089 [−0.21, +0.04] | 655 / 798 | 286 / 62 | +1335 / −1545 |
| A1.5_F0_D2 | 48.5 | −0.037 | [−0.170, +0.097] | −0.086 | −0.004 | 3/10 | 2.0 / 5.0 | 61 / 29 | −0.059 | −0.083 [−0.17, +0.01] | −0.098 [−0.21, +0.01] | 426 / 708 | 137 / 228 | +1222 / −1404 |
| A2_F0_D1.5 (#3) | 61.9 | −0.010 | [−0.148, +0.118] | −0.070 | +0.031 | 4/10 | 1.9 / 3.6 | 64 / 31 | −0.033 | −0.056 [−0.15, +0.04] | −0.062 [−0.17, +0.04] | 609 / 703 | 207 / 3 | +998 / −1120 |
| A2_F0_D2 (#1) | 58.2 | +0.011 | [−0.142, +0.163] | −0.042 | +0.047 | 5/10 | 1.8 / 4.6 | 71 / 35 | −0.024 | −0.035 [−0.11, +0.04] | −0.046 [−0.14, +0.04] | 315 / 620 | 173 / 51 | +910 / −987 |
| BULLRUN | 50.5 | **−0.182** | [−0.249, −0.116] | −0.138 | −0.212 | 0/10 | 2.4 / 3.5 | 17 / 9 | −0.183 | −0.227 [−0.39, −0.07] | −0.305 [−0.50, −0.11] | 1214 / 975 | 334 / 382 | +2555 / −3054 |
| N1 BULLRUN_SL3 | 74.2 | −0.083 | [−0.183, +0.014] | −0.136 | −0.047 | 3/10 | 2.4 / 3.0 | 41 / 20 | −0.088 | −0.129 [−0.28, +0.01] | −0.140 [−0.30, +0.01] | 754 / 886 | 492 / 18 | +1903 / −2186 |
| N2 A2_F1_D2 (#2) | 61.9 | +0.012 | [−0.132, +0.151] | −0.052 | +0.056 | 5/10 | 1.8 / 3.9 | 66 / 32 | −0.024 | −0.033 [−0.12, +0.05] | −0.038 [−0.13, +0.06] | 303 / 709 | 207 / 3 | +992 / −1066 |
| N3 A1.5_F0.5_D1.5 | 67.6 | −0.043 | [−0.157, +0.064] | −0.145 | +0.027 | 2/10 | 2.0 / 3.1 | 53 / 24 | −0.054 | −0.088 [−0.21, +0.02] | −0.066 [−0.19, +0.06] | 643 / 813 | 332 / 4 | +1371 / −1566 |

**Win and loss sizes, live ruler.** This is the mechanism.

| cell | avg win | avg loss | share ≤ −2.5 % | share ≥ +2 % |
|---|---|---|---|---|
| LOCK | +2.87 | −3.10 | 46.9 % | 24.4 % |
| A1_F0_D1 | +1.00 | −2.36 | 24.5 % | 8.2 % |
| A1_F0.5_D1 | +0.88 | −3.04 | 24.5 % | 6.6 % |
| A1_F0_D2 | +2.06 | −1.25 | 24.5 % | 13.9 % |
| A2_F0_D2 | +2.05 | −2.82 | 37.7 % | 22.3 % |
| BULLRUN | +0.90 | −1.28 | 0 % | 10.2 % |
| BULLRUN_SL3 | +0.92 | −2.98 | 24.5 % | 15.5 % |

- About 75 % of fills touch +1 net. Arming there turns those fills into small wins, which caps the winners.
- The 24.5 % that never touch +1 still lose the full −3.
- **Every armed-at-+1 cell has a paired Δ vs LOCK below zero, with the all-fills day-CI upper bound below zero (−0.008 to −0.017).**
  - On May–Oct alone the upper bound is ≤ +0.005 for all six.
  - Jan–Apr means are −0.15 to −0.18 vs LOCK's −0.024.
- The early lock is not a near miss: it is significantly worse.

**Per month, live ruler (mean %/trade):**

| month | N | LOCK | A1_F0_D1 | A1_F0.5_D1 | A2_F0_D2 | A2_F1_D2 | BULLRUN | BULLRUN_SL3 |
|---|---|---|---|---|---|---|---|---|
| 2026-01 | 165 | −0.060 | −0.088 | −0.064 | +0.026 | +0.038 | −0.058 | +0.007 |
| 2026-02 | 238 | +0.344 | −0.193 | −0.203 | +0.302 | +0.193 | −0.243 | −0.145 |
| 2026-03 | 224 | −0.088 | −0.216 | −0.247 | −0.145 | −0.048 | −0.102 | −0.282 |
| 2026-04 | 270 | −0.274 | −0.096 | −0.109 | −0.302 | −0.326 | −0.125 | −0.095 |
| 2026-05 | 268 | +0.119 | −0.137 | −0.145 | −0.004 | −0.036 | −0.227 | −0.166 |
| 2026-06 | 254 | +0.066 | −0.112 | −0.116 | +0.094 | +0.106 | −0.300 | −0.022 |
| 2026-07 | 183 | −0.015 | +0.042 | +0.049 | +0.038 | +0.031 | −0.149 | −0.017 |
| 2026-08 | 279 | +0.277 | +0.038 | +0.019 | +0.209 | +0.251 | −0.295 | +0.016 |
| 2026-09 | 287 | +0.085 | −0.064 | −0.097 | −0.034 | −0.025 | −0.083 | +0.001 |
| 2026-10 | 30 | −0.879 | −0.623 | −0.559 | −0.574 | −0.449 | −0.164 | −0.413 |

The full per-month table for all 15 cells is in `months2_P.csv`.

## 3. Selection and confirmation, as pre-registered

- **Jan–Apr ranking (14 non-baseline cells, live ruler):** A2_F0_D2 −0.042 > A2_F1_D2 −0.052 > A2_F0_D1.5 −0.070 > A1.5_F0_D2 −0.086 > … > A1_F0.5_D1.5 −0.178.
- **Every cell is negative in discovery**, and the LOCK itself is −0.024 there. Finalists: A2_F0_D2, A2_F1_D2, A2_F0_D1.5.
- **Best-of-46 null:** day-level sign flips of the per-fill (cell − LOCK) difference on May–Oct, max over all 46 non-baseline cells (round 1 + round 2, all re-priced on the live ruler), 2,000 draws. The 95th percentile is **+0.197**.

| finalist | May–Oct Δ vs LOCK | paired day-CI | cell May–Oct mean | > null q95 (+0.197)? | verdict |
|---|---|---|---|---|---|
| #1 A2_F0_D2 | **−0.046** | [−0.135, +0.040] | +0.047 | no | **FAIL** |
| #2 A2_F1_D2 | −0.038 | [−0.128, +0.055] | +0.056 | no | FAIL (cannot rescue) |
| #3 A2_F0_D1.5 | −0.062 | [−0.169, +0.039] | +0.031 | no | FAIL (cannot rescue) |

**Exit-change verdict: FAIL.** No round-2 exit beats the FRENZY lock out of sample.

**Multiplicity disclosure (post-hoc, not admissible).** Re-pricing round 1's 32 cells on the live ruler shows these top cells:

| cell (round 1) | all | Jan–Apr | May–Oct |
|---|---|---|---|
| TRAIL_s-4_a3_t2 | +0.157 | +0.132 | +0.175 |
| TRAIL_s-4_a3_t3 (round 1's own #1) | +0.127 | +0.231 | +0.055 |
| TRAIL_s-4_a3_t1.5 | +0.107 | +0.068 | +0.134 |

- TRAIL_s-4_a3_t2 is the best of 47 by an after-the-fact look.
- Its May–Oct Δ vs LOCK is +0.081, well under the best-of-46 null q95 of +0.197.
- No pre-registered rule would have selected it: round 1 picked t3 on next-open Jan–Apr.
- It is recorded only so nobody "discovers" it later. It is not a candidate.

## 4. Secondary ruler (next open, +0.02 slip), N = 2,236

- **Same ordering.** LOCK is +0.115 [−0.061, +0.300]. Every round-2 cell is lower:

| cell | avg | note |
|---|---|---|
| A2_F0_D2 | +0.050 | best round-2 cell |
| A2_F1_D2 | +0.035 | |
| armed-at-+1 cells | −0.054 … +0.007 | |
| BULLRUN | −0.128 | |
| BULLRUN_SL3 | −0.022 | |

- **The same three finalists are picked on Jan–Apr.** Their May–Oct Δ vs LOCK:

| finalist | May–Oct Δ | CI |
|---|---|---|
| A2_F0_D2 | −0.092 | [−0.18, −0.00] |
| A2_F1_D2 | −0.107 | [−0.20, −0.01] |
| A2_F0_D1.5 | −0.102 | [−0.22, +0.00] |

- The ruler choice does not change any conclusion.

## 5. Verdict against the pre-declared bar

The bar for a free scout observe line, on the live ruler: avg ≥ +0.10 AND day-CI lower bound > 0 AND May–Oct > 0. It is applied to the LOCK, and to the #1 finalist only if that finalist passed confirmation.

- **LOCK:** +0.045, CI [−0.129, +0.229], May–Oct +0.093.
  - It fails on the average and on the CI.
  - Under a 5-position cap it is −0.011: the 1.8 % of skipped trades are the mass-pump days, worth +3.07 each.
  - Dropping the best 1 / 5 / 10 days gives −0.006 / −0.063 / −0.113.
- **#1 finalist A2_F0_D2:** it did not pass confirmation, so it is not eligible. For the record it would also fail the bar: +0.011, CI [−0.142, +0.163].

**→ HOLD_LOWVOL_EARLY is CLOSED for good** (universe + any of 47 exits). No observe line, no arm proposal.

**Haircut:** not applicable, because there is no positive Δ to haircut.
