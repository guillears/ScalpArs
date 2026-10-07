# FRENZY_LITE — entrar tras 4 / 6 / 8 cierres sobre el VWAP en vez de 12 (2026-10-07)

## Resumen en lenguaje simple

**Veredicto: dejar 12. No pasar a 6 (ni a 4 ni a 8).**

- Con 12 cierres seguidos sobre la media (lo actual) cada operación ganó de media **+0,163 %** (724 operaciones en el año, timing real).
  Con 6 cierres: **+0,036 %** (1.010 ops). Con 4: +0,106 %. Con 8: −0,008 %. Ninguna variante supera a 12, ni en el año completo
  ni en mayo–octubre (12: +0,168 % · 6: +0,015 %).
- **Por qué entrar antes no sirve:** cuando el precio llega a 12 cierres, entrar antes en ese mismo movimiento sí es más barato
  (con N=6, +0,15 puntos de media en el mismo tramo). Pero de antemano no se sabe qué tramos llegarán a 12. Los que llegan a 6
  y se caen antes de 12 son **328 operaciones nuevas que pierden −1,27 % cada una** (pierden en enero–abril y en mayo–octubre).
  Eso se come todo lo ganado con la entrada más barata.
- La regla quedó fijada antes de ver resultados. Pedía que 6 ganara a 12 en el año y en mayo–octubre, con un intervalo
  emparejado no claramente negativo y con los tramos nuevos no negativos. **No cumple ninguna de las cuatro condiciones.**
- **⚠️ Hallazgo importante y aparte (paridad con el motor):** el código de FRENZY_LITE que se está escribiendo **no hace lo
  mismo que el estudio de +0,163 %**:
  - **En el estudio:** si la primera vela que cumple es verde (o la frena el volumen global, la dislocación o el tope de 3 por día), **se descarta todo el tramo**.
  - **En el código:** el tramo solo se marca como "hecho" cuando hay una COMPRA, así que **reintenta en la vela siguiente**.
  - **Resultado con esa lógica:** aun con 12 cierres, el resultado es **−0,005 %/op en 1.992 ops** (casi 3 veces más operaciones). Octubre sale en −0,71 %.

  **Para que el motor opere lo que se probó**, el tramo tiene que quedar "consumido" en la primera vela que cumple (≥ N cierres, fuera de estado, volumen < 100×, entre 2 y 16,75 h), aunque luego la frene un filtro.
- **2026-10-06** (datos públicos de Binance; cálculo aproximado con velas de 1 minuto, sin el filtro de volumen global):
  - **GRIFFAIN (tramo desde las 20:25).** Con la lógica del estudio, ninguna N entra, porque la primera vela que cumple es verde en los cuatro casos. Con la lógica del código actual:
    - N=4 y N=6: entran a las 21:00 y tocan el stop a las 21:28 (**−3,1 %**).
    - N=8: entra a las 21:10 y sale con ganancia asegurada (**+1,9 %**).
    - N=12: entra a las 21:30 y sale con ganancia asegurada (**+3,0 %**).
  - **EDU (tramo desde las 20:45).** Solo llega a 6 cierres; a las 21:15 vuelve a caer bajo la media.
    - N=6 entra a las 21:15 y toca el stop a las 21:18 (**−3,1 %**). Luego el precio cayó hasta −8 %.
    - N=4 también entra ahí con la lógica del código actual. Con la lógica del estudio queda bloqueado, porque su primera vela que cumple (21:00) es verde.
    - N=8 y N=12 nunca entran.

  O sea: el día de ayer **es un ejemplo de por qué 12 es mejor**. Entrar antes compra justo los tramos que se caen.

---

## 1. Pre-registro (antes de calcular P&L)
Archivo: `scratchpad/lite_streak/PREREG_STREAK_N.txt`.
- **23:34:17Z:** primario 6 vs 12; vecinos 3 y 9; regla de decisión fijada.
- **Enmienda 1, 23:34:41Z:** el operador cambió la rejilla a **N ∈ {4, 6, 8, 12}** antes de ver ningún resultado (todavía no había cohortes ni precios). Familia de 4 celdas.
- **Enmienda 2, 23:39:46Z:** se añade la lectura de **SENSIBILIDAD** del estudio de latencia: entrada a +8 s, deslizamiento de salida 0,10 en salidas lock/cap y 0,18 en el stop. Hasta ese momento solo se habían visto los conteos de las cohortes; no se había calculado ningún P&L.
- **Enmienda 3, 23:42:08Z:** se añade la lectura secundaria **DEFER** (paridad con el motor; ver §6). Para entonces ya se conocían los resultados primarios con la convención de bloqueo. Esta lectura no cambia la regla.

Método:
- Cohortes con el **frenzy_walk real** del recorrido anual (`manualall/yw_all.pkl`): exactamente la máscara de `signals.py` CAND_C, cambiando `above_hour` por `above_streak ≥ N`. Primera barra por tramo, después el corte 2–16,75 h, después el stack F2 verde + F3 gvol V1<1 + F4 dislocación 1 % + F5 tope de 3 por par-día. Sin ATR.
- Precios sobre ticks con la librería `holdlow_ticks/ticklib.py` y el mismo lock (−3 hasta +3, después max(+2, pico−2)), con tope de 12 h.
- Se bajaron 143 + 1 pares-día de data.binance.vision.

## 2. Paridad (la cohorte de 12 reproduce el estudio)
- Universo N=12: **2.239 barras = FREEZE_HOLD_LOWVOL_EARLY, coincidencia 2.239/2.239** (0 sobran, 0 faltan).
- Stack sin F1: **724 = 724**, las 724 son las mismas, con PRI medio +0,1628 = +0,1628.
- Repricing de las 2.236 barras ya calculadas: diferencia máxima en LOCK_t, PRI, move12 y tiempo de entrada = **0,0**.
- bar_red del motor frente a velas de 5 m: 100 % de acuerdo.

## 3. Resultados por N (convención del estudio = pre-registrada)
- **PRI** = timing real primario: +12 s, salida −0,10, comisiones 0,09.
- **SENS** = +8 s, 0,10 en lock/cap y 0,18 en stop.
- **NEXT** = siguiente apertura +0,02.
- IC = bootstrap por bloques de día al 95 %.

| N | ops | días | WR | **PRI media** | IC PRI | ene–abr | may–oct | meses + | suma PRI | **SENS** | NEXT | top par / día |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1198 | 255 | 53,9 | +0,106 | [−0,101, +0,337] | +0,218 | +0,028 | 5/10 | +126,6 | +0,098 | +0,186 | 1,5 % / 5,5 % |
| **6** | 1010 | 259 | 53,6 | **+0,036** | [−0,178, +0,271] | +0,063 | **+0,015** | 4/10 | +36,0 | +0,035 | +0,132 | 1,8 % / 5,6 % |
| 8 | 873 | 251 | 51,4 | −0,008 | [−0,249, +0,246] | −0,219 | +0,150 | 5/10 | −6,5 | −0,006 | +0,138 | 2,2 % / 4,8 % |
| **12** | 724 | 244 | 55,7 | **+0,163** | [−0,107, +0,421] | +0,156 | **+0,168** | 7/10 | +117,9 | +0,143 | +0,271 | 2,5 % / 4,1 % |

**Dosis-respuesta 4 → 6 → 8 → 12** (PRI): +0,106 → +0,036 → −0,008 → +0,163.
- No es monótona, y 12 es el máximo.
- N=4 obtiene una suma similar (+126,6 frente a +117,9), pero con un 65 % más de operaciones, la mitad de rentabilidad por operación y mayo–octubre en +0,03.
- La proporción de stops es de 44–48 % en todas las N.

Con límite de posiciones (PRI):

| N | 2 slots: ops / media / may–oct | 5 posiciones: ops / media / may–oct |
|---|---|---|
| 4 | 1074 / +0,035 / +0,016 | 1175 / +0,068 / +0,023 |
| 6 | 936 / **−0,016** / +0,009 | 997 / **−0,003** / +0,021 |
| 8 | 809 / −0,063 / +0,152 | 863 / −0,052 / +0,150 |
| 12 | 689 / +0,092 / +0,116 | 719 / +0,146 / +0,168 |

## 4. Descomposición frente a 12 (PRI)
- **(a) Emparejados:** tramo con entrada en N y también en 12 dentro del stack.
- **(b) Nunca llegan a 12:** tramos nuevos.
- **(c) Llegan a 12, pero la entrada de 12 se bloqueó:** en ~93 % de los casos la frenó un filtro (sobre todo la vela verde).
- Además, una parte de los tramos de 12 **desaparece** en N, porque su barra N estaba bloqueada.

| N | (a) n · N-entrada vs 12-entrada · **dif emparejada [IC]** · may–oct dif [IC] | (b) nuevos: n · media [IC] | (c) n · media | 12 perdidos: n · media |
|---|---|---|---|---|
| 4 | 307 · +0,606 vs +0,194 · **+0,412 [+0,15, +0,67]** · +0,209 [−0,18, +0,59] | 560 · **−1,053** [−1,32, −0,78] | 331 · +1,601 | 417 · +0,140 |
| 6 | 330 · +0,205 vs +0,054 · **+0,151 [−0,07, +0,35]** · +0,376 [+0,14, +0,63] | 328 · **−1,269** [−1,59, −0,92] | 352 · +1,092 | 394 · +0,254 |
| 8 | 325 · +0,228 vs +0,313 · **−0,085 [−0,27, +0,10]** · −0,025 [−0,27, +0,22] | 205 · **−1,442** [−1,82, −1,07] | 343 · +0,627 | 399 · +0,041 |

Lectura:
- **(a)** Entrar antes en el mismo tramo es algo más barato, salvo con N=8.
- **(c)** Es positivo, pero está definido con información futura: el tramo "llegó a 12". No se puede elegir en tiempo real.
- **(b)** Es lo único que se puede saber al entrar, y pierde mucho y de forma consistente en las dos mitades del año (N=6: ene–abr −0,96, may–oct −1,48).
- (a) y (c) también dependen del futuro. La comparación honesta es la de cohorte contra cohorte (§5).

## 5. Comparación 6 vs 12 (cohorte contra cohorte, bootstrap conjunto por bloques de día)

| | dif. media/op [IC 95 %] | P(dif<0) | dif. suma [IC] |
|---|---|---|---|
| 6−12 año, PRI | −0,127 [−0,382, +0,144] | 0,82 | −81,9 [−303, +154] |
| **6−12 may–oct, PRI** | **−0,153 [−0,512, +0,192]** | 0,80 | −61,7 [−241, +107] |
| 6−12 año, SENS | −0,108 [−0,374, +0,172] | 0,78 | −68,3 |
| 6−12 may–oct, SENS | −0,126 [−0,496, +0,240] | 0,75 | −50,1 |
| 4−12 may–oct, PRI | −0,140 [−0,470, +0,191] | 0,80 | −50,9 |
| 8−12 may–oct, PRI | −0,018 [−0,373, +0,344] | 0,53 | +4,6 |

**Nota de selección:**
- N=6 es una variante propuesta después de ver los resultados de la familia FRENZY.
- La familia tiene 4 celdas; se aplica el recorte del 30–50 % a cualquier mejora.
- Ninguna variante muestra mejora, así que el recorte no cambia nada: ninguna supera a 12.

## 6. Paridad con el motor: convención DEFER (secundaria, enmienda 3)
- **Código en curso** (`frenzy_lite_status` + `_frenzy_lite_eval`, sin commit): el tramo se marca "hecho" **solo al haber una compra**. Si la vela es verde, si la frena gvol o la dislocación, o si se alcanza el tope por día, el motor reintenta en la vela siguiente del mismo tramo.
- **Estudio de 724:** descarta el tramo entero.

DEFER (primera barra del tramo que pasa todo), PRI:

| N | ops | media | IC | ene–abr | may–oct | meses + | 2 slots | 5 pos |
|---|---|---|---|---|---|---|---|---|
| 4 | 3072 | −0,093 | [−0,24, +0,07] | −0,057 | −0,118 | 3/10 | −0,135 | −0,140 |
| 6 | 2663 | −0,051 | [−0,21, +0,13] | −0,044 | −0,057 | 4/10 | −0,110 | −0,095 |
| 8 | 2373 | −0,028 | [−0,21, +0,17] | −0,069 | +0,002 | 4/10 | −0,104 | −0,070 |
| **12** | **1992** | **−0,005** | [−0,19, +0,20] | +0,016 | −0,020 | 4/10 | −0,073 | −0,058 |

- Con DEFER y N=12, 720 de las 724 operaciones del estudio siguen ahí. Las **1.272 entradas extra** (diferidas una mediana de 3 velas) dan una media de **−0,093 %**.
- SENS con DEFER y N=12: −0,043. NEXT: +0,066.
- **Con la lógica del código actual, FRENZY_LITE no tiene la ventaja probada.**
- Arreglo propuesto: guardar el id del tramo como "juzgado" en la primera barra que cumple las condiciones de señal (≥ N cierres, fuera de estado, vol < 100×, 2 ≤ h ≤ 16,75), aunque la rechace el filtro de vela verde, gvol, dislocación o tope de 3.
- En el estudio, el filtro de volumen 24 h actúa como diferimiento: esa barra queda fuera del universo elegible. Mantenerlo como diferimiento es consistente con la evidencia.
- La decisión de cambiar el código corresponde al coordinador o al operador.
- Incluso con DEFER, 6 no le gana a 12: −0,051 frente a −0,005 en el año y −0,057 frente a −0,020 en mayo–octubre. Los tramos nuevos dan −1,49.

## 7. 2026-10-06 en vivo
Velas públicas de 5 m de fapi y frenzy_walk real con la configuración en vivo. La normal-hour sale de las velas de 1 h.

**GRIFFAIN** (spike 14:50, tramo desde la vela de las 20:25):
- La racha llega a 4 en la vela de las 20:40, a 6 en la de las 20:50, a 8 en la de las 21:00 y a 12 en la de las 21:20. **Las cuatro son verdes.**
- El volumen ronda 37–51×, nunca entra en estado, y lleva unas 6 h desde el spike.

**EDU** (spike 17:55, tramo desde la vela de las 20:45):
- Racha 4 en la vela de las 21:00 (verde), 6 en la de las 21:10 (roja).
- En la vela de las 21:15 vuelve a 0 y nunca llega a 8.

P&L aproximado con velas de 1 m: entrada en la apertura del minuto del cierre de señal, orden pesimista (primero el mínimo), sin chequear gvol.

| caso | estudio (bloqueo) | código actual (defer) |
|---|---|---|
| GRIFFAIN N=4 | sin entrada (verde) | entrada 21:00 → stop 21:28 **−3,1 %** |
| GRIFFAIN N=6 | sin entrada (verde) | entrada 21:00 → stop 21:28 **−3,1 %** |
| GRIFFAIN N=8 | sin entrada (verde) | entrada 21:10 → lock 22:23 **+1,9 %** |
| GRIFFAIN N=12 | sin entrada (verde) | entrada 21:30 → lock 22:23 **+3,0 %** (pico +5,2) |
| EDU N=4 | sin entrada (verde a las 21:00) | entrada 21:15 → stop 21:18 **−3,1 %** |
| EDU N=6 | entrada 21:15 → stop 21:18 **−3,1 %** | igual |
| EDU N=8 / N=12 | nunca llega | nunca llega |

## Archivos
- Scratch: `scratchpad/lite_streak/`. Contiene:
  - `PREREG_STREAK_N.txt`
  - `build.py`, `price.py`, `price2.py`, `analyze.py`, `decomp.py` (salida en `decomp.out`)
  - `defer.py`, `adefer.py` (salida en `adefer.out`)
  - `live1006.py`, `live_px.py`
  - datos: `priced*.pkl`, `kept.pkl`, `defer_kept.pkl`, `summary.csv`
- Ticks nuevos: `reports/backtest_cache/ticks/` (mismo formato).
