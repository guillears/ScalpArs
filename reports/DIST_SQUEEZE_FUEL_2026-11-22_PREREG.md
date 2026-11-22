# DIST_SQUEEZE_FUEL — pre-registration (written 2026-11-22, before any outcome was read)

**Question (operator 11-22):** "why not testing it now with the info we have and adding it to watchlist?" — theory (e) for DISTRIBUTION_SHORT losers.

**Theory (e) squeeze fuel:** the short loses when the coin is already crowded with shorts. Late shorts are trapped, the first up-tick forces short covering / liquidations, and the stop is hit within minutes. The batch losers (NAORIS 2.4 min, BLESS 7, SENT 15, KERNEL 37) motivated it, so the batch is **not** evidence — only the backtest DESIGN → HOLDOUT read and the Route B descriptive read count.

**Prior-study check:** DIST_WL 11-18 P8 tested `fund_last` (flat) and `prem_last` (high premium better, weak); P3 found rising OI into the red candle BETTER (it contradicts the "new shorts piling in" form of (e)); P4 `top_pos_ls` no pattern. New here: the squeeze-specific outcome (fast stop), the crowded-short composite, the account / global L/S and taker variables under that outcome, and the Route B pre-entry read.

## Cohort
`reports/DIST_WL_2026-11-18_trades.csv.gz`, today's armed rule = L1 ∧ `rsi1h ≥ 74.07`, LIVE exit (Exit B), `in_LIVE`, exit kind ≠ 3. Blocks: DESIGN = `block == DESIGN`; HOLDOUT = H-OLD ∪ H-NEW (read once). Bins (terciles) frozen on DESIGN.

## Outcomes
- **Primary — FAST_STOP:** LIVE_k == 1 (stop) ∧ (LIVE_x − entry_ms) ≤ 60 min.
- **Secondary — net:** `net_LIVE` % per trade.

## Variables (entry-time only) and the predicted "crowded short" side (worse)
| # | variable | crowded side |
|---|---|---|
| E1 | `fund_last` | LOW tercile (most negative funding = shorts pay) |
| E2 | `prem_last` | LOW tercile (perp below index) |
| E3 | `top_pos_ls` | LOW tercile (top traders' positions net short) |
| E4 | `top_acct_ls` | LOW tercile |
| E5 | `glob_ls` | LOW tercile (all accounts short-heavy) |
| E6 | `taker_1h` | LOW tercile (aggressive sellers already in) |
| E7 | `oi_chg1h` | HIGH tercile (new positions opened into the drop) — known-opposite from P3, kept for completeness |
| C | crowd score = count of E1–E5 in the crowded tercile (0–5) | score ≥ 3 (and score as a dose) |

Sign granularity too where natural: `fund_last` < 0 vs ≥ 0 · `prem_last` < 0 vs ≥ 0 · `top_pos_ls` / `glob_ls` < 1 vs ≥ 1.

## Pass rule for an OBSERVE line (frozen)
A variable / the composite becomes a DIST observe line only if ALL hold:
1. DESIGN: the crowded side has a higher FAST_STOP rate AND a lower net % than the rest, |t| (net) beyond the theory-family null (95 % of max |t| over E1–E7 + C on 200 label shuffles).
2. HOLDOUT (read once): same sign on BOTH outcomes.
3. Dose: terciles monotone on net in DESIGN (no mid-hole).
4. No single pair ≥ 50 % of the crowded side's loss.
The observe line's cut = the frozen DESIGN cut. Arming needs the locked EXPECTANCY bar later (and not inside the freeze).

## Route B (descriptive only, 2–3 days — cannot pass or fail anything)
DIST REAL + SHADOW signals since 11-19 23:00: short liquidations $ in the 15 min before entry, recorded funding, taker buy share 15 min before → FAST_STOP rate by "above / below median".

## Reported per the rules
Batch before → after (6 real fills, each tagged by the frozen cut) and backtest before → after (N · WR · avg % · DCR at 0.61 × equity), DESIGN / HOLDOUT, per-pair concentration, blind spots.
