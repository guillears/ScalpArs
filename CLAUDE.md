# SCALPARS - Automated Crypto Futures Trading Platform

## Role & Responsibility

You are the technical owner of development and analysis for this project. The user is not a developer and will not review code at a technical level. You should act as the coding expert, make sound engineering decisions, write clean and production-ready code, and proactively identify the best implementation approach without depending on the user for low-level technical validation.

Likewise, the user is not the quantitative analyst. You should act as the quant expert when analyzing results, performance, filters, strategy behavior, and proposed parameter changes. Your role is to interpret the data rigorously, avoid overfitting, distinguish between weak signals and strong conclusions, and recommend the most robust next steps based on evidence.

When responding:
- Do not assume the user will catch technical mistakes
- Do not delegate key engineering or analytical judgment back to the user
- Explain trade-offs clearly in simple language when needed
- Make the strongest recommendation you can based on best practices, data, and logic
- Be proactive, precise, and accountable for both implementation quality and analytical quality

Your job is not just to execute instructions, but to act as the project's technical and quantitative expert, helping drive the best possible decisions.

## Project Overview
Python 3 / FastAPI trading bot for Binance Futures. Uses EMA, RSI, ADX indicators for signal generation with configurable break-even levels and stop-loss management.

**Run locally:** `python3 run.py` (auto-creates venv, installs deps, starts on http://localhost:8000)

**Key files:**
- `main.py` — FastAPI app, API endpoints, background tasks
- `services/trading_engine.py` — Core trading logic, position management
- `services/indicators.py` — Technical analysis (EMA, RSI, ADX)
- `services/binance_service.py` — Binance Futures API wrapper (CCXT)
- `trading_config.json` — Strategy parameters
- `config.py` — Environment and app configuration

## Deploy Flow
When the user says **"commit and push"**, commit all changes and push to `main`. AWS auto-deploys from `main`.
- Do NOT push unless the user explicitly says "push" or "commit and push".
- If changes are risky (new logic, parameter changes in live mode), warn the user and suggest testing on paper mode before pushing.
- A simple "commit" means commit only, do not push.

## Trading Strategy Analysis Context (188 trades, March 2026)

Reference this when working on trading_engine.py, main.py, or trading_config.json.

### Exit Optimization
- **RSI Momentum Exit is the strongest exit signal**: Replacing all BE exits with RSI Momentum would swing P&L from -$1,409 to ~+$536.
- **Signal Lost is the dominant loss driver**: 38 trades at avg -$65.70 = -$2,496 total. RSI fading precedes full signal reversal.
- **BE L1/L2**: reasonable as safety nets. **L3/L4/L5**: leaving money on the table vs RSI exit — consider raising offsets.
- **Levers to test (300-400 trades)**: Lower RSI Momentum `min_profit` to catch losers earlier; adjust BE L3-L5 offsets upward.

### Entry Optimization
- **EMA5-EMA8 Gap is the strongest entry quality predictor**: SHORT 0.10-0.20% gap = 67.5% WR, only profitable bucket. Weak momentum entries (0.02-0.05%) are the primary Signal Lost source.
- **ADX sweet spot is 20-25**: Only profitable range. ADX 30+ underperforms in choppy conditions.
- **EMA20 Slope**: moderate slopes (0.08-0.12%) best for shorts. Very steep (>0.20%) may be overextended.
- **Levers to test (300-400 trades)**: Raise EMA5-EMA8 min gap from 0.02% to 0.05%+; cap ADX max at 30.

### April 8, 2026 — Pre-Exit Optimization Baseline (35L + 3S)

Baseline saved at `reports/report_2026-04-08_pre_exit_optimization.txt`.

#### Key Findings (35 LONG trades in Bullish regime)
- **Total P&L**: -$36.75 (-0.23% avg), 45.7% WR, expectancy -$1.05/trade
- **FL_SIGNAL_LOST**: 12 trades, -$43.79 — biggest loss driver, avg peak only +0.08%
- **TICK_MOMENTUM_EXIT**: 11 trades, +$6.65 — cutting winners at +0.14% when post-peak was +1.03%
- **TRAILING_STOP**: 3 trades, +$7.65 — best exit type, +0.65% avg
- **9 "Positive, No BE" SL trades**: peaked at +0.12% avg then dropped to -0.87%, lost $33.91
- **EMA5 Stretch**: 0.16-0.20% = 100% WR (+$6.60). Below 0.16% = 22% WR. Phase 2 entry filter candidate.
- **All flagged trades had positive peaks** — entries work, exits fail to capture profit

#### New Exit Structure (deployed April 8, collecting data)
- **BE L1**: trigger at 0.15% peak, floor at 0.10% (protects weak winners that peaked 0.10-0.25%)
- **Trailing Stop**: TP at 0.40%, pullback 0.08% (rides strong winners)
- **Tick Momentum**: DISABLED (was net negative — cutting winners short)
- **BE L2-L5**: disabled (99)
- **Signal Lost Flag + Security Gap**: kept
- **SL**: -0.9% unchanged

#### Comparison Table (fill at 40 trades)
| Metric | Before (35L) | After (40L target) |
|--------|-------------|-------------------|
| Win Rate | 45.7% | ? |
| Avg P&L % | -0.23% | ? |
| Expectancy/trade | -$1.05 | ? |
| Profit Factor | 0.30 | ? |
| Trailing Stop avg close % | +0.65% (3 trades) | ? |
| BE L1 saves | 0 (was off) | ? |
| FL_SIGNAL_LOST trades | 12 (-$43.79) | ? |
| Post-peak captured | +0.14% (TM cut early) | ? |
| EXTERNAL exits | 34.3% | ? |

#### Infrastructure Fixes (April 8)
- **P&L accuracy**: Exit now uses real Binance fill price, not WebSocket price
- **Slippage tracking**: `[EXIT_SLIPPAGE]` logged on every close, `exit_slippage_pct` stored per order
- **DB lock fix**: WAL mode + 30s busy timeout + same-session commit retry (5 attempts)
- **Position check before retry**: Prevents ReduceOnly rejected spam when close succeeded but CCXT lost response
- **EXTERNAL exits**: Should drop to ~0% with these fixes

#### Current Config (April 8)
- Both confidence levels: `both` mode, 1x leverage
- Tick Momentum: OFF | RSI Momentum: OFF
- BE L1: 0.15%/0.10% | BE L2-L5: OFF (99)
- Trailing: TP 0.40%, pullback 0.08%
- Breadth filter: OFF | Short RSI min: 25
- Signal Lost Flag + Security Gap: ON

#### Phase 2 (after 40-trade validation)
- **Entry tuning**: EMA5 Stretch minimum threshold (0.16%?)
- **Pair blacklisting**: Based on Slip% data per pair
- **BE L1 fine-tuning**: Adjust trigger/offset based on new data

### Caveats
- 188 trades in a single choppy bearish regime — patterns may shift in trending/bullish markets.
- Small samples for higher BE levels (L3=23, L4=10, L5=5) and longs (46 trades).
- RSI range observations may be regime-specific; momentum strength (EMA gap, slope) is likely more universal.

## April 11, 2026 — DB-Loss Incident & AWS Hardening

### What happened
At ~01:39 UTC-3 on Apr 11, the bot "crashed" and stopped working while leaving ~40+ orphaned open positions on Binance for ~8 hours. Root cause diagnosed:
1. **AWS Managed Platform Update fired** on its default schedule (Saturday night maintenance window)
2. Deployment policy was **Immutable** — AWS launched a brand-new EC2 instance (`i-0c68bf886974420d6`) to apply the update
3. Old instance (`i-0606409df26cd905f`) was terminated at 01:48:53
4. Since SQLite `scalpars.db` lived on the old instance's **ephemeral local disk**, it was destroyed with the instance
5. New instance started with an empty DB — `bot_state` defaulted to `is_running=False, is_paper_mode=True`
6. Background loops (scan, monitor, realtime callback) silently returned early because `is_running=False`
7. User lost ~40 trades of data from the overnight window and had to manually close orphaned Binance positions

### AWS hardening applied (all via console, no code changes)
| Change | Path | Why |
|---|---|---|
| Managed Platform Updates **disabled** | Config → Updates, monitoring, logging → Managed updates | The exact trigger of the incident. Saturday auto-upgrade is now off. |
| Deployment policy = **All at once** | Config → Rolling updates and deployments | Code deploys now apply in-place on the existing instance; SQLite survives. |
| Rolling config updates = **Disabled** | Same page | VPC/config changes no longer spawn new instances. |
| Capacity verified = **Single instance + On-demand** | Config → Capacity | No load balancer, no Spot reclaim, no ASG health-replacement. |
| CloudWatch Logs streaming **enabled** (14-day retention) | Config → Updates, monitoring, logging → Log streaming | Logs survive instance death for post-mortem investigation. |
| EB native email notifications **enabled** | Config → Updates, monitoring, logging → Email notifications | Alerts on deployment/instance events. |
| EC2 termination protection **enabled** on `i-0c68bf886974420d6` | EC2 → Instances → Actions → Instance settings | Blocks accidental termination via EC2 API/console. |
| CloudWatch alarm `SCALPARS-EB-Health-Severe` | CloudWatch → Alarms | Fires when `AWS/ElasticBeanstalk / EnvironmentHealth > 10` for 2 of 3 datapoints (1-min period), notifies SNS topic `scalpars-alerts` → user email. |

### What each defense blocks
- Managed updates off → blocks the specific trigger from Apr 11
- All-at-once → blocks any future config/code change from replacing the instance
- Single instance + on-demand → no ALB, no Spot, no ASG replacement
- Termination protection → human error / some AWS ops
- CloudWatch Logs → preserves forensic evidence even if instance dies
- CloudWatch alarm → pages user within ~3 min if environment health degrades

### What is NOT covered (known gaps)
1. **EC2 hardware failure** — rare but possible; would still destroy the ephemeral SQLite file. Mitigation = DB durability work (see below).
2. **Application-level crashes** where OS stays up — EB doesn't detect, environment health stays OK, no alarm fires. Primary defense = code-level fixes (already applied: FL1/FL2 race, DB contention, autoflush, greenlet).
3. **Manual platform upgrades** — when eventually needed, they DO replace the instance. Must back up `scalpars.db` first and restore after.
4. **Binance-side outages** — not infra solvable.

### Trade-off accepted
Managed Platform Updates are now permanently off. No weekly auto-applied OS/runtime security patches. This is **low urgency** because the bot is not a public-facing web app — attack surface is narrow (outbound Binance API + self-hosted UI). Recommended manual update cadence: every 3-6 months, **only after** DB durability is fixed so instance replacement is safe.

### DB Durability — Pending Decision (Recommended Path)

The fundamental remaining problem: `scalpars.db` lives on the EC2 instance's ephemeral local disk. Any instance replacement for any reason still destroys it. The Apr 11 hardening blocks the most common triggers but does not eliminate this class of risk.

Three real options were presented to the user on Apr 11, decision deferred:

#### C1 — S3 snapshot on cron (recommended FIRST step)
- **What:** background task copies `scalpars.db` to `s3://scalpars-db-backups/latest/scalpars.db` every 5 min using SQLite's `.backup` / `VACUUM INTO` for atomic consistency. Rotates daily snapshots to `s3://scalpars-db-backups/daily/YYYY-MM-DD.db`. On app startup, if local DB missing OR older than S3 latest, download from S3 before opening SQLAlchemy.
- **Data loss window:** up to 5 min (last backup interval)
- **Effort:** ~1.5 hours (S3 bucket + IAM role + ~50 lines of Python + test + deploy)
- **Cost:** ~$0.10/mo
- **Risk:** near zero — self-contained, SQLite stays SQLite
- **Verdict:** fastest real protection. Would have cut Apr 11 blast radius from 8 hours to 5 minutes.

#### C2 — EFS mounted filesystem (SKIP)
- **What:** mount EFS at `/mnt/efs/scalpars/`, put SQLite there, DB survives instance replacement.
- **Effort:** ~2-3 hours (can go sideways)
- **Risk:** SQLite-on-NFS is a known sharp edge; file locking can be unreliable → WAL mode corruption risk under contention.
- **Verdict:** worst of both worlds — not as fast as C1, not as clean as C3. Do not pursue.

#### C3 — RDS Postgres migration (recommended LONG-TERM)
- **What:** replace SQLite with managed Postgres on RDS. Full backups, point-in-time recovery (7 days), no WAL mode / busy_timeout hacks, proper concurrency.
- **Data loss window:** zero
- **Effort:** 1-2 full days focused work (audit SQLite-isms in `database.py`/`models.py`, fix type hacks, build one-time migration script, test locally against Postgres container, cut over production)
- **Cost:** ~$13-20/mo RDS `db.t4g.micro` (Free Tier first 12 months if unused)
- **Risk:** medium — requires porting and focused testing time
- **Verdict:** the real answer. Do it after bot is validated and strategy is working, not while still iterating.

#### Recommended staged path
1. **This week / next:** ship C1 (S3 snapshot) as safety net — ~1.5 hours
2. **1-3 months out:** migrate to C3 (RDS) as real fix — 1-2 focused days on a quiet weekend

**Status:** user deferred decision on Apr 11 — "lets wait". Revisit when ready. Skip C2 entirely. Do NOT re-enable Managed Platform Updates under any circumstances until DB durability is resolved (C1 or C3).

## April 14, 2026 — Locked Baseline for Fresh Validation

### Context
Two reports now exist as reference data:
- `reports/report_2026-04-13_full_117trades.txt` — 117 trades (66L + 51S), runtime 1.86 days, 1x leverage. Best full-detail sample available.
- Today's 21L sample (in-memory) — 21 trades, runtime ~1 day, mixed 10x/20x leverage, contaminated by 6+ config changes (log below).

### Apr 13 117-trade findings (THE REFERENCE dataset)
- **Total: Longs -$7.77 (65.2% WR, PF 0.83), Shorts +$0.33 (60.8% WR, PF 1.01). Net ~$-7.50 over 117 trades at 1x.**
- **VERY_STRONG is broken in both directions**: 15L 40% WR -$12.83 + 19S 47% WR -$5.33 = 34 trades 44% WR -$18.16. STRONG_BUY meanwhile: 51L 72.6% WR +$5.06 + 32S 68.8% WR +$5.66. The stricter VERY_STRONG filters are producing WORSE trades.
- **Long ADX 22-25 is a disaster bucket**: 18 trades 38.9% WR -$16.60. But the disagreement (today's 21L shows ADX 22-25 at 80% WR) is not resolvable yet.
- **Long EMA5-EMA8 gap 0.02-0.04% is the BEST bucket** (83.3% WR +$3.12). DO NOT raise `ema_gap_threshold_long` above 0.02.
- **FL_DEEP_STOP L1 is still the dominant loss path**: 14L -$33, 10S -$23. Peak only +0.16% avg before SL hit. Same problem as April 8 baseline.
- **Short Gap 0.08%+ works, <0.08% loses**: current `ema_gap_threshold_short = 0.08` is correctly set.
- **Maker vs Taker fee split**: TAKER_FALLBACK longs 59% WR -$10.92 vs MAKER 70.6% WR +$3.15. Taker fallback trades are systematically worse — consider blocking entry when maker doesn't fill.

### Today's 21-trade contradictions (DO NOT TRUST for filter tuning)
- Config churn: 6 changes on Apr 14 alone (leverage 10↔20x, EMA gap, BTC ADX min, flat thresholds, regime_change_exit toggle)
- Small sample inverted several Apr 13 signals (ADX 22-25 looked good, BTC ADX 20-25 looked bad, range position 75%+ looked bad)
- Treat as illustrative, not reliable

### LOCKED Apr 14 baseline (do NOT change until 40+ fresh trades collected)
- Leverage: **1x both levels** (fresh test, leverage-invariant statistics)
- Trade mode: both, max 5 positions, equal_split, $100 fixed
- `ema_gap_threshold_long = 0.02` (0.02-0.04% is the best bucket per Apr 13)
- `ema_gap_threshold_short = 0.08`
- `momentum_adx_max_long = 25` (unchanged — Apr 13 vs today contradict, leave for data to decide)
- `btc_adx_min_long = 25`, `btc_adx_min_short = 20`
- VERY_STRONG + STRONG_BUY: both enabled, identical params, 1x lev
- Exits: BE L1 0.15/0.10, Trailing TP 0.40%/pullback 0.08%, Tick Momentum OFF, RSI Momentum OFF, Signal Lost Flag + Security Gap ON, FL1/FL2 ON, Regime Change Exit ON
- Market Breadth filter ON (30% bull L, 45% bear S, flat threshold 0.02)
- EMA Gap Expanding Filter ON, RSI Momentum Filter ON

### Validation plan
1. Reset (clears Orders/Transactions/BnbSwapLog, keeps config)
2. Let 40+ trades accumulate with zero config changes
3. Compare fresh sample to Apr 13 117-trade reference
4. THEN make filter decisions with 3 samples to triangulate

### Known issues to investigate AFTER 40-trade fresh baseline
- **VERY_STRONG underperformance**: 34-trade signal. Either disable or relax to match STRONG_BUY.
- **FL_DEEP_STOP L1 path**: Peak-to-exit gap too wide. The flag fires correctly but exit lets it ride to SL. Worth a code review of the FL1/FL2 branches in `trading_engine.py`.
- **Long ADX max**: 25 vs 22. Needs more data to resolve Apr 13 / today contradiction.
- **TAKER_FALLBACK longs losing**: Consider blocking entries when maker doesn't fill within timeout.

### Rule for the next 40 trades
**NO CONFIG CHANGES.** Every config tweak invalidates the sample. If a problem seems urgent, pause trading and document the issue for post-sample review rather than flipping parameters mid-stream.
