# Refused signals vs taken trades — months ['08', '09'], journal seed 1

Calibration (live kept momentum fills, same trades): N 77 · actual avg +0.175 / WR 77% · sim avg +0.099 / WR 65% · sign agreement 81% · corr 0.64

- momentum LONGS taken (live kept): N   64 · WR  61% · avg +0.042 · full stops  39%
- momentum SHORTS taken (live kept): N   13 · WR  85% · avg +0.378 · full stops  15%
- REFUSED by ATR_GAP_LONG (LONG): N  426 · WR  66% · avg +0.060 · full stops  34% · days 58 · positive days 57% · by month 07 -0.10 (n16) / 08 +0.11 (n261) / 09 -0.02 (n149)
- REFUSED by FAN_RATIO_GATE (LONG): N 1500 · WR  54% · avg -0.064 · full stops  45% · days 58 · positive days 33% · by month 07 -0.28 (n81) / 08 -0.04 (n751) / 09 -0.07 (n668)
- REFUSED by PAIR_ADX_DIR (SHORT): N 1500 · WR  52% · avg -0.099 · full stops  48% · days 58 · positive days 24% · by month 07 -0.05 (n71) / 08 -0.11 (n757) / 09 -0.09 (n672)
- REFUSED by MOMENTUM_SHORT_LOATR (SHORT): N 1368 · WR  59% · avg -0.032 · full stops  41% · days 49 · positive days 29% · by month 07 -0.23 (n46) / 08 -0.02 (n778) / 09 -0.03 (n544)
