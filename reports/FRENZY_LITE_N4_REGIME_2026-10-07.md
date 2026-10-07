# FRENZY_LITE N=4: ¿una variable macro explica su caída? (2026-10-07)

## Resumen en lenguaje simple

**Veredicto: ninguna variable macro explica la caída de N=4. No hay ninguna regla del tipo "usar N=4 cuando el mercado esté X". Se mantiene N=12.**

- **Primero, la "caída" de N=4 no es segura.** N=4 pasa de +0,218 %/op en enero–abril a +0,028 % en mayo–octubre, una diferencia de −0,19. El intervalo por bloques de día es [−0,66, +0,27], así que podría ser ruido. N=12 queda plano: +0,156 → +0,168.
- **Lo que sí empeora es el grupo de "tramos nuevos"**, los que llegan a 4 cierres pero nunca a 12:
  - pasan de −0,85 a −1,20 %/op;
  - la tendencia mes a mes va en contra, unos −0,11 puntos cada 30 días, con un intervalo que no llega a cero;
  - pierden más porque tocan más el stop (58 % → 66 %), no porque las pérdidas sean más grandes.
  - La parte de "entrar antes en los mismos tramos" no cambia: +0,606 en las dos mitades. La proporción de tramos nuevos tampoco cambia: 46,7 % en las dos mitades.
- **Se probaron 23 variables macro**, una por una y en todas las parejas posibles (~1.000 combinaciones). Entre ellas:
  - BTC: retornos a 1/3/7/30 días, distancia al máximo de 30 días, pendiente 1 h, tendencia 4 h y diaria, volatilidad, ADX;
  - volumen global del mercado;
  - amplitud del mercado y fuerza de las alts frente a BTC;
  - cantidad de pumps y calidad de los episodios.
  - **Ninguna pasa** con las mismas reglas de siempre: igual en las dos mitades del año, mejor que el azar barajado, ≥ 8 días por lado y sin depender de un solo día.
- **Los tramos nuevos pierden en TODOS los estados del mercado.** En ningún grupo de ninguna variable salen positivos (el mejor grupo está en −0,02 con N pequeño). No hay un "buen momento" para N=4.
- **No se mueven junto con N=12.** Semana a semana, los tramos nuevos y N=12 apenas van juntos (correlación +0,18). Por eso no es un régimen de todo el sleeve: lo que se degrada son justamente esos tramos que no se confirman, de forma lenta y sin causa macro visible.
- **En ningún estado del mercado N=4 le gana a N=12 de forma fiable**: la ventaja cambia de signo entre las dos mitades del año en todas las variables.
- **Hallazgo aparte (no sobre N=4, sino sobre N=12):** cuando el volumen medio del mercado de las 24 h anteriores está **alto** (> 0,996, medido al inicio de la ventana de 4 h), N=12 gana **+0,53 %/op** (373 ops). Cuando está bajo, **−0,23 %** (351 ops).
  - Va en la misma dirección en las dos mitades (+0,82 / +0,72), en 9 de 10 meses, y supera al azar (p = 0,02).
  - **Pero** con la medición principal pre-registrada (a las 00:00) falla en enero–abril.
  - Por eso queda **solo como línea de observación gratuita, sin armar**.

---

## 1. Pre-registro
Archivo: `scratchpad/lite_regime/PREREG_N4_REGIME.txt`, escrito a las 23:50:19Z. Para entonces solo se había visto la descomposición por mitades, todavía sin datos macro.

- **Medición principal:** el valor macro a las 00:00 UTC del día de la operación. Unidad = día.
- **Medición secundaria:** el valor al inicio de la ventana de 4 h.
- **Para sobrevivir, una puerta tiene que cumplir todo esto:**
  - mismo signo en ene–abr y en may–oct;
  - |z| ≥ 2 en ambas mitades (descubrir en una y confirmar en la otra, en los dos sentidos);
  - ≥ 8 ventanas dentro y fuera en cada mitad;
  - ninguna ventana con ≥ 50 % del resultado neto;
  - superar el percentil 95 del máximo de la familia con un barajado dentro del mismo mes.

Cohortes (de `lite_streak/kept.pkl`, PRI, convención de bloqueo del estudio):

| | qué es | ops |
|---|---|---|
| **A** | N=4, todas | 1.198 |
| **B** | N=4, tramos nuevos (nunca llegan a 12) | 560 |
| **C** | N=12, control | 724 |

## 2. Descomposición del cambio entre mitades (`decomp_half.out`)

**N=4:** el cambio es de −0,190 = +0,050 por cambio de mezcla + −0,240 dentro de cada grupo.

| grupo | proporción ene–abr / may–oct | media ene–abr → may–oct | aporte |
|---|---|---|---|
| a, mismo tramo que 12 ("entrada antes") | 28,5 % / 23,7 % | +0,606 → +0,606 | 0,000 |
| **b, tramos nuevos** | 46,7 % / 46,7 % | **−0,848 → −1,195** | **−0,163** |
| c, llega a 12 pero 12 bloqueado | 24,8 % / 29,6 % | +1,780 → +1,497 | −0,077 |

**N=12:** cambio de +0,012. El grupo emparejado con N=4 pasa de −0,05 a +0,40. Las entradas solo de N=12 pasan de +0,33 a +0,02. Las dos cosas se compensan.

Diferencia entre mitades (bootstrap por bloques de día):

| cohorte | diferencia may–oct − ene–abr | IC 95 % |
|---|---|---|
| A | −0,19 | [−0,66, +0,27] |
| B | −0,35 | [−0,97, +0,22] |
| C | +0,01 | [−0,52, +0,53] |

Tendencia lineal (cada 30 días):

| cohorte | pendiente | IC 95 % |
|---|---|---|
| A | −0,089 | [−0,179, +0,001] |
| **B** | **−0,109** | **[−0,215, −0,006]** |
| C | −0,015 | [−0,125, +0,088] |

Medias mensuales de B, de enero a octubre: −0,89 · −0,45 · −0,86 · −1,19 · −0,64 · −1,03 · −1,07 · −1,40 · −1,54 · −1,66.

Estructura de B:

| | WR | stop | ganancia media | pérdida media |
|---|---|---|---|---|
| ene–abr | 39 % | 58 % | +2,57 | −3,04 |
| may–oct | 33 % | 66 % | +2,62 | −3,05 |

Al repesar B por variables propias de cada par (atr5, vol_mult, horas, gvol 5m, move12), el cambio de mezcla explica **≤ 0,04 de los −0,35**.

## 3. Tabla macro (`macro_hourly.pkl`, `macro_day.pkl`, `macro_4h.pkl`)
Todo se calcula "a fecha": solo cuentan las barras ya cerradas.

- **Origen:**
  - BTC: `btc_1h.csv`.
  - Universo: `k5m_full` → horario (509 pares COIN, sin Alpha).
  - Volumen: `gvol_year.pkl`.
  - Episodios: `yw_all.pkl`.
  - Señales LITE: universo de `lite_streak`.
- **Variables (23):**
  - BTC: `r1d`, `r3d`, `r7d`, `r30d`, `off30d`, `slope1h` (EMA20 1 h frente a 3 barras antes, como el motor), `gap4h`, `gap1d` (EMA13−EMA50 %), `rv1d`, `rv7d`, `adx4h`, `adx1d`;
  - volumen de mercado: `gvol24` (V1 del motor, media de 24 h) y `gvoldash24` (estilo dashboard);
  - amplitud del top-50 por volumen de 24 h: `breadth_ema20` (% sobre la EMA20 1 h) y `breadth_pos24` (% con 24 h positivo);
  - fuerza relativa: `alt_rs7d` (mediana de las alts del top-50 a 7 días − BTC a 7 días);
  - densidad de pumps: `pump_eps24` y `pump_eps7d` (episodios FRENZY por día);
  - calidad: `ep_quality` (% de episodios de hace 24–72 h que llegaron a ON);
  - señales: `lite4_sig24` y `lite12_sig24`;
  - `fail4_72h`: % de tramos ≥ 4 terminados en las últimas 72 h que no llegaron a 12.
- **Cobertura de los días con operaciones:** 100 % en todas las variables, excepto `ep_quality` y `fail4_72h`, que tienen 99,6 % (les falta el primer día, el 10 de enero).

## 4. Barrido de separadores (`screen_day.out`, `screen_4h.out`, `trend_null.out`)

Puntuación = min(|z| ene–abr, |z| may–oct), con el mismo signo en las dos mitades. "Nulo 95" es el percentil 95 del máximo de la familia con datos barajados.

| familia | A: obs / nulo 95 | B: obs / nulo 95 | C: obs / nulo 95 |
|---|---|---|---|
| 1D día (69 puertas) | 1,67 / 2,17 | 1,32 / 2,03 | 1,27 / 1,85 |
| 2D día (1.012 puertas) | 2,60 / 2,73 | 2,12 / 2,77 | 2,28 / 2,72 |
| 1D 4 h, agrupado por día | 1,39 / 1,98 | 1,57 / 1,87 | **2,06 / 1,85** (p = 0,02) |
| 2D 4 h, agrupado por 4 h | 2,60 / 2,69 | 2,11 / 2,77 | 2,45 / 2,39 (mismo tema gvol) |

- **Para A y B no sobrevive nada**, con ninguna medición ni granularidad (primero por signo, luego por terciles, luego 2D).
- El mejor candidato 1D de A es `btc_adx4h > mediana`:
  - z = 1,67 en ene–abr y 3,02 en may–oct;
  - pero con 4 h agrupado por día baja a 1,34;
  - y C se mueve en la misma dirección (z 1,22), así que sería un efecto de régimen de todo el sleeve, no algo propio de N=4.
- **B en todos los grupos de todas las variables:** la media dentro del grupo es negativa en ambas mitades. El mejor grupo 2D es `btc_r7d− & alt_rs7d−`, con ene–abr −0,02 (n pequeño) y may–oct −0,45.

**Cambio de mezcla entre mitades (repesando may–oct a la mezcla de ene–abr):**
- may–oct tuvo mucha menos volatilidad de BTC: el tercil alto de `rv7d` pasó del 53 % al 19 % de las ops;
- y más fuerza de las alts frente a BTC: el tercil alto de `alt_rs7d` pasó del 14 % al 46 %;
- con eso se explica como mucho −0,09 de los −0,19 de A (por `rv7d`) y −0,20 de los −0,35 de B;
- pero `rv7d` no pasa como separador (z 1,2–1,8), así que es solo una pista.

## 5. Degradación uniforme (punto ③ del checklist)

Correlación entre cohortes, por semana (38 semanas) y por mes (10 meses):

| | semanal | mensual |
|---|---|---|
| A con C | +0,33 | +0,32 |
| **B con C** | **+0,18** (rango +0,08) | +0,28 |
| A con B | +0,76 | +0,90 |

- La pendiente de C es plana (−0,015 cada 30 días); B baja (−0,109).
- **La caída no es uniforme.** Lo que pierde y empeora son los tramos que no se confirman. N=12 no acompaña esa caída.
- No es un régimen macro medible con estas variables.

**"Usar N=4 cuando X, si no 12"** (`n4_vs_12_by_state.out`). En ningún estado sale una ventaja estable de N=4 sobre N=12 (cifras: dif. N=4 − N=12):

| estado | ene–abr | may–oct | año, IC 95 % |
|---|---|---|---|
| gvol24 bajo | +0,27 | −0,22 | — |
| alt_rs7d ≤ 0 | — | — | −0,27 [−0,69, +0,13] |
| btc_rv7d alto | — | — | +0,09 [−0,28, +0,44] |

**No hay regla candidata para N=4.**

## 6. Hallazgo aparte: N=12 y el volumen de mercado de las 24 h anteriores (solo observar)

**Definición:** `gvoldash24` (media de 24 h del ratio estilo dashboard), tomado al inicio de la ventana de 4 h. Umbral **0,996** (la mediana de las ventanas con operaciones), **congelado**.

| volumen 24 h | ops | media | WR |
|---|---|---|---|
| alto (> 0,996) | 373 | **+0,532** | 59,5 % |
| bajo (≤ 0,996) | 351 | **−0,230** | 51,6 % |

Cómo se sostiene:

| prueba | resultado |
|---|---|
| ene–abr | diferencia +0,82, z(día) 2,12 |
| may–oct | diferencia +0,72, z(día) 2,17 |
| meses donde alto > bajo | 9 de 10 |
| días por lado | 171 |
| concentración | ningún día > 5,3 % de la pérdida; ningún par > 2,8 % |
| frente al azar (barajado dentro del mes, agrupado por día) | p = 0,02 |

**Por qué no se arma:**
- Con la medición principal (00:00), en ene–abr la diferencia es solo +0,09: +0,202 con volumen alto frente a +0,115 con bajo. Falla.
- Viene de una familia de 69 puertas × 2 momentos de medición.
- Es una variable de mercado: según la regla de unidades de ventana, funcionaría como interruptor de encendido/apagado de todo el sleeve, no como filtro de operaciones.
- Hay que aplicar el recorte del 30–50 % (in-sample). Δ esperado si se bloquea el lado bajo: unas +0,38 ×(0,5–0,7) ≈ **+0,19 a +0,27 puntos por op**, a cambio de perder ~48 % de las ops.

**Propuesta (no armar):** línea de scout gratuita **LITE_GVOL24_LOW**.
- Marcar las entradas FRENZY_LITE N=12 con `gvoldash24` de 24 h ≤ 0,996 al inicio de la ventana de 4 h.
- Barras congeladas:
  - **promover a revisión** si, en datos nuevos, el lado bajo tiene ≥ 30 ops en ≥ 15 días, una media por op ≥ 0,4 por debajo del lado alto y un IC de bloques de día por debajo de 0;
  - **retirar** si la diferencia ≤ 0 tras 30 ops.
- La decisión es del coordinador o del operador.

## 7. Qué NO se pudo probar
- Funding, interés abierto (OI), long/short ratio, liquidaciones y profundidad del libro: no hay histórico público en la caché. Funding y OI de Binance tienen límites de histórico.
- BTC dominance (como índice), flujos de stablecoins, calendario de listados y noticias.
- Variables de cada par en el momento de la entrada para el grupo B: no son macro y aquí solo se repesaron 5. Un barrido completo de las columnas `entry_*` de B queda pendiente (`sweep_separators`-style).
- La deriva de B (−0,11 por cada 30 días) no tiene causa identificada. Puede ser estructural (competencia en los primeros minutos del tramo, por ejemplo).
- El 2D de la medición de 4 h con agrupación por día (en C) no tiene un nulo propio. Solo se tomó el 1D.

## Archivos (scratch `scratchpad/lite_regime/`)
- Pre-registro: `PREREG_N4_REGIME.txt`.
- Construcción de datos: `decomp_half.py`, `hourly_univ.py`, `macro.py`.
- Análisis: `screen.py` (day / 4h), `follow.py`, `follow2.py`, `trend_null.py`, `n4_vs_12_by_state.py`. Cada uno tiene su `.out`.
- Datos: `macro_{hourly,day,4h}.pkl`, `univ_1h_{close,qvol}.pkl`, `screen_{day,4h}.pkl`.
