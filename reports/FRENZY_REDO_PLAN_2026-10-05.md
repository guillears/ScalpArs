# FRENZY / WIDE studies — redo plan on the ENGINE cohort (2026-10-05)

Cause: the P2 moments cohort's fresh-ON bar lagged services.frenzy.frenzy_walk (streak never reset after a down run) → every FRENZY/WIDE year
study of 2026-10-04/05 studied partly the wrong bars. Each study below is re-run ONE BY ONE on reports/FRENZY_ENGINE_COHORT_2026-10-05.csv
(built by the re-validation task with the real frenzy_walk, checked against live fills), same pre-registered variants and bars, old vs new
verdict side by side. Nothing in the bot changes until the operator sees each result.

| # | Study (original report) | Decision touched | Status |
|---|---|---|---|
| 0 | Engine cohort rebuild + core re-validation (FRENZY_REVALIDATION_ENGINE_BARS) | 205, 207/216, 215, 216 | running |
| 1 | Lock vs fixed / trail family (FRENZY_TRAIL_V2, V2B) | 205 lock exit | queued |
| 2 | Re-entry while ON · ATR stop · gvol gate × ATR (FRENZY_REENTRY_STOP_GVOL) | 194 gvol gate | queued |
| 3 | Audit items: entry delay, re-entry caps 1–2, conditional re-entry a–d, structure exits, combos (FRENZY_REENTRY_AUDIT) | — | queued |
| 4 | Exit selector + selector-typed re-entry (FRENZY_EXIT_SELECTOR) | — | queued |
| 5 | ATR direction + pure EMA20/EMA50 no-stop + ATR-below-entry exit (FRENZY_ATR_AND_PURE_EMA_EXIT) | 214 ATRE shadow | queued |
| 6 | Post-entry info, split position, add-on (FRENZY_RIDE_NEW_ANGLES) | — | queued |
| 7 | Ride-signature re-entry (FRENZY_RIDE_SIGNATURE_REENTRY) | — | queued |
| 8 | Signature re-entry + runner exit (FRENZY_SIGNATURE_RUNNER_REENTRY) | — | queued |
| 9 | Loser separator + leverage + brakes (FRENZY_LOSER_SEPARATOR_LEVERAGE) | 216 leverage, 215 filter | queued |
| 10 | Regime study (FRENZY_REGIME) | 216 tracker | queued |
| 11 | RLC 10-05 under every variant (illustration — already on the engine's own bars; unchanged) | — | valid |
| 12 | Re-entry ONLY AFTER A WINNING first trade — with the lock, and with each runner exit (EMA20 / EMA50 / 5-pt / 2×ATR / state-off) (operator) | — | queued |
| 13 | RLC-TYPE CAPTURE synthesis (operator's top priority): on the engine cohort, how many RLC-like rides per month, what each surviving exit / re-entry / sizing rule captures of them, and the best combined rule in $ at real sizing | all | queued |

Reporting: each study → old vs corrected verdict + recommendation; leverage recommendation (FRENZY / WIDE) restated on the corrected cohort; regime impact; RLC case in every report.
