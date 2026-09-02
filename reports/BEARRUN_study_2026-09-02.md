# BEARRUN study — mirrored bull-run sleeve for SHORTS (2026-09-02)
**Question (operator):** would a bear-side mirror of the 🌊 bull-run sleeve have paid? Define the bearish filter, check the last 72h, then evaluate across history.

## Filter definition (exact mirror of GREEN, replayed with hysteresis + latch)
- RED turn-ON (BTC 5m, 864 closed bars): r72 ≤ −5.0% ∧ below-EMA21 ≥ 56% ∧ efficiency ≥ 0.10
- Squeeze-latch (mirror of crash-latch): r6h ≥ +3% OR price > 1h EMA50 → instant OFF
- Stay band: −4.0 / 53 / 0.10 · 30-min stay-band re-fires merged (engine reopen rule)
- Entries: mirrored dip-reclaim on per-window top-10 by prior-day quote volume (blacklist ONG/ETH/BTC kept): rally ≥0.3×ATR above 5m EMA20 (6h expiry) → closed bar rejects below EMA20, pair < its 1h EMA50, BTC not ≥2% above its 24h low (bounce-phase gate), 2h/pair spacing, 4 slots
- Exits: the sleeve stack sign-mirrored — SL min(−0.7, −1.5×ATR) floored −1.2 · BE arm +1.0 → lock +0.2 · 2.0×ATR trail · ladder floors · 180-min max-hold · fees 0.08% RT

## Last 72h: filter NEVER armed (0 windows)
Most bearish moment Aug-31 01:25Z: r72 −4.16 / below 51.7 / eff 0.074 — all three legs short. Chop regime, hostile to both sleeves. → Bear sleeve result last 72h = $0, 0 trades.

## 6.5-month replay (Feb-16 → Sep-02): 12 RED episodes, ~66h armed
| cohort | N | WR | Σ pnl% |
|---|---|---|---|
| ALL (COIN-only) | 55 | 44% | −4.36 |
| Jun-2→4 sustained window (43.7h, latch-ended) | 19 | 63% | **+7.36** |
| 11 flicker windows (0.6–5.3h) | 36 | 33% | **−11.72** |
Positive windows: 3/12. Raw incl. non-coin perps (XAU/XAG/MRVL/CL/SPCX/MU — real scan is COIN-only): 62 tr · 40% · −8.93.

## Verdict: symmetric clone REFUTED — regime asymmetry
Bull regimes persist for days; bear moves on this tape are violent mean-reverting flushes. 11/12 RED windows were flickers that bounced immediately after arming (stop-cascade at −0.7); ALL the concept's profit lives in the one sustained episode. The GREEN gate set is insufficient on the short side because a bear-regime birth is indistinguishable from a capitulation low.
**Surviving candidate: AGE GATE — arm the sleeve only once RED is ≥6h old.** Would have skipped ~all 11 flickers and kept the Jun-2 payer — but that is fit on N=1 sustained window → WATCHLIST, not a ship (WINDOW-UNITS rule).

## Caveats on record
Per-window top-10 drawn from the CURRENT top-45 universe (mild survivorship early months) · 5m-bar exit granularity vs live ticks · flat 0.08% fee · no funding. None flip the flicker cohort's sign.

Files: BEARRUN_study_trades_2026-09-02.csv (62 trades) · BEARRUN_study_sim_2026-09-02.py · BEARRUN_study_windows_2026-09-02.py (reproducible).
