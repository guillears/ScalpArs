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

## Core Operating Principles

These principles govern every engineering and analytical decision on this project. They override shortcut instincts — if a faster path conflicts with one of these, pause and flag it rather than silently taking the shortcut.

- **Build everything to scale from day one. Never implement patchwork or temporary solutions. Every system, component, and decision should be designed to handle growth and be ready to scale to thousands of transactions reliably and efficiently.**

- **Act as a top-tier crypto quant analyst whenever you receive a trading report. Your job is to identify the optimal combination of indicators, filters, and thresholds to maximize expectancy and therefore maximize long-term profitability, always prioritizing data-driven decisions over intuition.**

- **When comparing results across different reports or batches, always use Avg P&L % instead of absolute P&L, because the invested amount may differ between batches. Percentage performance is the correct metric for apples-to-apples comparison.**

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

## April 14, 2026 — Locked Baseline for 100-Trade Fine-Tuning Sample

### Filter design principle (how we make filter decisions)
**Use raw BTC dimension tables, not regime labels.** The BTC regime classifier
(`services/regime.py`) is a lossy summary of BTC ADX + RSI + Slope with arbitrary
thresholds (ADX cutoff 28, slope flat 0.02%, RSI exhaustion 30/70). Filter decisions
should be made on the raw dimension tables (`Performance by BTC ADX`,
`Performance by BTC Entry RSI`, `Performance by BTC EMA20 Slope`) and cross-tabs
(`BTC RSI × BTC ADX`, `BTC Slope × BTC ADX`) directly. Regime labels are retained
for reporting/intuition only, not for gating logic.

**Implementation target (Phase 2 code work):** Build a "BTC RSI × BTC ADX Cross-Filter"
UI section at BTC level, mirroring the existing pair-level `RSI x ADX Cross-Filter`
(see UI: "Restrict which ADX ranges are allowed per RSI range. Empty = allow all
combinations."). Separate LONG rules and SHORT rules. Each rule says: "for BTC RSI
range [min, max], require BTC ADX ≥ X" or "require BTC ADX ≤ X" or "block entirely."
Empty = allow all combinations. This replaces the need for hard-coded conditional
logic with flexible per-direction rules. Stored as `btc_rsi_adx_filter_long` and
`btc_rsi_adx_filter_short` strings, same format as existing pair-level strings.

### Target sample
**100 trades: ~50 longs + ~50 shorts** collected at 1x leverage under frozen config. This is the decision point for the first round of data-driven filter tuning. Paper mode.

### Reference datasets available
- `reports/report_2026-04-13_full_117trades.txt` — 117 trades (66L + 51S), runtime 1.86 days, 1x leverage, loose filters (BTC ADX min L=15, regime_change_exit OFF). **The primary reference.**
- `reports/report_2026-04-12_partial_53trades.txt` — 53 trades, full detail
- `reports/report_2026-04-08_pre_exit_optimization.txt` — 38 trades summary, pre-TM-disable baseline
- Multiple earlier reports (Apr 6, 7, 9, 10) for cross-sample pattern confirmation

Each historical sample ran under a different config. **Do NOT pool raw trades across samples** — the bot was a different strategy in each run. Use cross-sample patterns only (findings that replicate across different configs = robust signal).

### Apr 13 117-trade findings (primary reference)
- **Total: Longs -$7.77 (65.2% WR, PF 0.83), Shorts +$0.33 (60.8% WR, PF 1.01). Net ~$-7.50 at 1x.**
- **VERY_STRONG underperforms STRONG_BUY in both directions**: combined 34 trades 44% WR -$18.16 vs STRONG_BUY 83 trades 71% WR +$10.72. Paradoxical — the "stricter" filter produces worse trades. Likely because its ADX>22 requirement pushes entries into the ADX 22-25 loser zone.
- **FL_DEEP_STOP L1 is the biggest absolute-loss sub-bucket** (14L -$33, 10S -$23), BUT the FL system as a whole is NET POSITIVE via the `NetRecover` column (+$11.76 total across 33 FL trades). FL is working — it saves money vs passive SL. The sub-issue is specifically that flagged trades with weak peaks (+0.1-0.2%) still ride to deep_stop at -1.0%. Regime Change Exit was added Apr 14 to catch some of these earlier.
- **Long EMA5-EMA8 gap 0.02-0.04% is the BEST bucket** (83.3% WR +$3.12, 6 trades). 0.04-0.08% is the loss zone. DO NOT raise `ema_gap_threshold_long` above 0.02.
- **Short EMA5-EMA8 gap**: <0.08% loses, 0.08-0.12% wins, 0.18-0.20% crushes (100% WR +$9.26). Current `ema_gap_threshold_short=0.08` correctly set.
- **Maker vs Taker Fallback**: MAKER 70.6% WR longs / 65.6% WR shorts, TAKER_FALLBACK 59.4% WR longs / 52.6% WR shorts. 10-13% WR gap. Taker fallback trades are chasing fast-moving price that often exhausts. Consider blocking entry entirely when maker doesn't fill within timeout.

### Cross-sample CONFIRMED findings — LONGS (Apr 13 + Today 21-trade)

These are directional patterns that replicate across both samples despite different leverage/config. **Use these for tuning decisions.**

| Pattern | Apr 13 | Today | Interpretation |
|---|---|---|---|
| EMA5-EMA8 gap 0.04-0.08% loses | 56.2% WR -$7 (32 trades) | 55.6% WR -$130 (9 trades) | Below 0.08% gap is weak entry |
| Range Position 50-75% > 75-100% | 69% vs 63% WR | 77% vs 37.5% WR | More room to run is better |
| ADX Delta 0.1-0.3 is the sweet spot | 81.2% WR (16 trades) | 77.8% WR (9 trades) | Building trend > peaking trend |
| Breadth 50-65% > 70%+ | 70-86% WR vs 69% WR | 83-100% WR vs 20% WR | Moderate breadth beats extreme |
| EMA5 Stretch 0.20-0.25% works | 66.7% WR +$0.76 | 80% WR +$24.21 | Only consistently positive bucket |
| EMA5 Stretch >0.25% fades | 64.3% WR -$1.43 (14 trades) | 66.7% WR -$17.62 (3 trades) | Mean reversion territory |
| "Positive, No BE" SL still a problem | 15 trades peak +0.18% → close -1.00% | 5 trades same pattern | Exit fails weak-peak trades |

### Cross-sample DIVERGE — do NOT tune these yet (need 3rd sample)

| Pattern | Apr 13 | Today | Why divergent |
|---|---|---|---|
| Long ADX 22-25 | disaster 38.9% WR | best 80% WR | Confidence-level confound |
| BTC ADX 20-25 | OK 67.7% WR | disaster 28.6% WR | Regime/sample confound |
| Entry Gap 0.25-0.30% | disaster 33% WR | OK 66.7% WR | Small-N confound |

### Shorts — NO cross-sample yet
Today's 21-trade sample has **zero shorts**. All short findings below are Apr 13 1-sample only and must be confirmed by the new 100-trade sample.

**Apr 13 short patterns to validate**:
- EMA5 Stretch: 0.25-0.30% best (83.3% WR +$11.88, 12 trades), 0.16-0.20% worst (50% WR -$9.46, 18 trades). **Opposite direction from longs** (shorts want HIGH stretch = confirmed downtrend).
- EMA5-EMA8 gap sweet spots: 0.08-0.12% (64-83% WR) OR 0.18-0.20% (100% WR)
- Short ADX: 25-30 better than 30-35
- Breadth 70%+: 72% WR (+$10.55) — acceptable

### FL system reality check (important correction)
Earlier claim "FL is broken" was wrong. The `NetRecover` column in the Flagged Exits table shows FL trades net +$11.76 across 33 trades — the system saves money vs passive SL. The specific sub-issue is FL_DEEP_STOP L1 (weak-peak trades riding to -1.0% stop). Regime Change Exit (added Apr 14) should catch macro rollovers that FL misses.

### Exit system status
- **Trailing Stop**: working across all samples. Apr 8 (3 trades +0.65% avg), Apr 13 (56+ trades +0.35-0.51% avg). Dominant winner-capture.
- **TP min raised 0.40 → 0.50, Pullback raised 0.08 → 0.20 (Apr 14)**: data-backed. Apr 13 winners had post-peak +1.36% to +3.77% — bot was exiting too early on the 0.08 pullback. New params allow tail-winners to run. Small risk of marginal-winner give-back; watch for "Positive, No BE" regression at 40-trade checkpoint.
- **Tick Momentum Exit**: permanently OFF. 2-sample confirmed (Apr 8 cut winners at +0.14% when post-peak was +1.03%; Apr 13 without TM captured 3-5x more).
- **Signal Lost Flag + Security Gap**: ON. Working as confirmed by NetRecover.
- **FL1 wide SL + FL2**: ON.
- **Regime Change Exit**: newly ON. Tests whether catching macro reversal improves weak-peak FL outcomes.
- **RSI Momentum Exit**: OFF.

### Phase 1a (CLOSED) — 19 trades collected at tight config (Apr 14 baseline)
- Collected ~19 trades (1L + 18S) from Apr 14 deploy → Apr 15 ~13:00
- 1 long trade only: BTC filters + gap min starving long side
- 18 shorts: acceptable rate, confirmed HEALTHY_BEAR winning / STRONG_BEAR losing 2-sample pattern
- Archived as reference for comparison

### Phase 1b (CURRENT) — Looser config for data collection (Apr 15 onwards, amended Apr 16)
Rationale: Phase 1a was starving the long side. Four filter changes applied on Apr 15 to unblock long entries and collect data on currently-unexplored buckets. **Amended Apr 16** with two additional BTC ADX min changes to widen the entry surface further. The strategic decision: use the BTC RSI × BTC ADX cross-tab (Phase 2 code work) as the fine-grained filter, rather than multiple coarse per-variable mins that may be cutting good trades along with bad.

**Changes from Phase 1a → Phase 1b:**
| Config | Apr 14 (Phase 1a) | Apr 15 (1b original) | Apr 16 (1b amended) | Rationale |
|---|---|---|---|---|
| `btc_adx_min_long` | 25 | 20 | **18** | Apr 15: historical BTC ADX 20-25 longs 41 trades, 71% WR, +$124 across 4 samples. Apr 16 further lowered to 18 to explore BTC ADX 18-20 long bucket (zero historical data in that range). |
| `btc_adx_min_short` | 20 | 20 | **18** | Apr 16: cross-sample confirmed winning short zone is BTC ADX 18-27 + slope falling (27 trades, 81% WR, +$19.93). 18 is exactly the lower bound of that winning zone; keeps <18 CHOPPY_WEAK shorts (43% WR -$2.54) blocked. |
| `macro_trend_flat_threshold_long` | 0.06 | **0.02** | 0.02 | Match shorts (which were firing fine at 0.02). Expected to unblock longs in sideways BTC conditions |
| `ema_gap_5_20_min_long` | 0.10 | **0.05** | 0.05 | Gap 5-20 is non-monotonic per 4-sample data. 0.12-0.15% was OK (17 trades, +$11.84), 0.15-0.20% was worst (40 trades, -$79). Lowering min explores 0.05-0.10% range (zero historical data) |
| `ema_gap_5_20_min_short` | 0.15 | **0.05** | 0.05 | Historical short data only for 0.15+ bucket. Lowering explores uncharted 0.05-0.15% range |

**Unchanged filters (still locked for Phase 1b):**
- Leverage: 1x both VERY_STRONG and STRONG_BUY
- Trade mode: both, max 5 positions, equal_split, $100 fixed
- `ema_gap_threshold_long = 0.02`, `ema_gap_threshold_short = 0.08` (EMA5-EMA8 gap, separate from EMA5-EMA20)
- `momentum_adx_max_long = 25` (not changed; let data decide)
- `momentum_ema20_slope_min_long = 0.0`, `momentum_ema20_slope_min_short = 0.04`
- Exits: TP 0.50 / pullback 0.20, BE L1 0.15/0.10, Signal Lost Flag ON, FL1/FL2 ON, Regime Change Exit ON, Tick Momentum OFF, RSI Momentum OFF
- Market Breadth ON (30 bull L, 45 bear S, flat 0.02)
- EMA Gap Expanding ON, RSI Momentum Filter ON
- Spike Guard ON (3x vol, 1.5% price)

### Rule for Phase 1b (the remainder of the 100-trade window)
**NO FURTHER CONFIG CHANGES from here.** Trade counting continues — the Apr 16 amendment does NOT reset the Phase 1b counter. Pre-Apr-16 trades (collected at `btc_adx_min=20`) and post-Apr-16 trades (at `btc_adx_min=18`) accumulate toward the same 100-trade target.

**Methodological caveat for bucket analysis at 100-trade review:** the sample will have a mixed entry-criteria distribution. Any bucket with BTC ADX ≥ 20 has data from the full Phase 1b run; the BTC ADX 18-20 sub-bucket (both LONG and SHORT) has data only from Apr 16 onwards, so its N will be smaller than its share of the population would suggest. Bucket-level WR / Avg P&L % comparisons remain valid for BTC ADX ≥ 20 buckets; the 18-20 bucket should be flagged as "partial coverage" when reported.

If long trade rate is still starved after this amendment, the issue is elsewhere (code bug, indicator calc, or truly no market opportunity).

### How to analyze Phase 1b data (amended Apr 16)
At 100-trade checkpoint (post Apr 16 amendment), in addition to the existing 22-question checklist:
1. **Did long trade rate increase meaningfully?** Compare Phase 1a (~1 long/day) to Phase 1b-amended rate.
2. **What's the 0.05-0.10% gap 5-20 bucket performance (longs)?** First-ever data in this range.
3. **What's the 0.05-0.15% gap 5-20 bucket performance (shorts)?** First-ever data in this range.
4. **Did BTC ADX 20-25 long bucket replicate historical 71% WR?** 5th-sample confirmation.
5. **Did BTC slope-flat longs perform OK now that threshold was lowered to 0.02%?** Previously blocked entirely.
6. **NEW (Apr 16): What is BTC ADX 18-20 LONG bucket performance?** Zero historical data in this range before Apr 16. Treat as pure exploration — at least 10 trades needed before drawing any conclusion. If WR < 50% or avg P&L % clearly negative → `btc_adx_min_long` should return to 20 in Phase 2.
7. **NEW (Apr 16): What is BTC ADX 18-20 SHORT bucket performance?** Expected to confirm the 81% WR winning zone extends down to 18 (per 2-sample BTC ADX 18-27 + slope falling pattern). If WR < 60% with ≥10 trades → the 18-20 sub-bucket was weaker than the 20-27 zone, raise `btc_adx_min_short` back to 20.
8. **Full BTC RSI × BTC ADX cross-tab** — the fine-grained filter that will eventually replace these blunt mins. Cross-tab now needs explicit BTC ADX 18-20 row.

### Pooling rule (amended Apr 16)
**Do NOT pool Phase 1a (19 trades) with Phase 1b raw data.** They were different configs. Use Phase 1a as a "pre" reference and Phase 1b as the "post" measurement.

**Phase 1b pre- and post-Apr-16 trades are pooled** into the same 100-trade target — the counter keeps going through the amendment. When analyzing bucket-level results, be aware that BTC ADX ≥ 20 buckets cover the full Phase 1b window while BTC ADX 18-20 buckets only have post-Apr-16 coverage. Report N per bucket to expose this.

Per Core Operating Principles: when comparing Phase 1b to earlier samples (Apr 6, 12, 13, Phase 1a) always use **Avg P&L %** — invested-amount-invariant, safe across batches with different position sizing.

### Checklist for the 100-trade report review
When the fresh report arrives, answer these questions in order:

**Go/No-Go (first check)**
1. Profit Factor? >1 continue, <0.5 stop and rethink, 0.5-1 likely fee drag on real edge.
2. Win Rate in 55-70% range? Outside = something broken.
3. Avg Loss % vs Avg Peak %? Are losers still bleeding past weak peaks (FL_DEEP_STOP regression)?

**2-sample confirmation tests (replicate Apr 13 findings)**
4. Does VERY_STRONG still underperform STRONG_BUY? If yes after 100 trades → **disable VERY_STRONG** (or relax to match STRONG_BUY entry rules).
5. Is MAKER still 10%+ WR higher than TAKER_FALLBACK? If yes → **block taker entries** (save fees + WR).
6. FL_DEEP_STOP count dropped (because regime_change_exit is now cutting them)? Measure absolute count and NetRecover$.
7. Long EMA5-EMA8 0.02-0.04% still best bucket? If yes → structural finding, document.
8. Long ADX 22-25 bucket — tiebreaker vs Apr 13 (disaster) + today (best). Decides whether to cap long ADX max.

**3-sample confirmation tests (patterns from this run + Apr 13)**
9. EMA5 Stretch >0.25% longs fade? (2-sample already says yes.)
10. Range Position 75-100% longs underperform? (2-sample already says yes.)
11. ADX Delta 0.1-0.3 long sweet spot? (2-sample already says yes.)
12. Breadth 70%+ longs underperform vs 50-65%? (2-sample already says yes.)

**BTC Entry RSI — 4-sample check (HIGHEST PRIORITY entry filter finding)**
13. **Long BTC RSI 50-55 still losing?** Historical data (Mar 30 + Apr 6 + Apr 12 + Apr 13 = 22 combined trades, 50% WR, negative $ in ALL 4 samples). If 5th sample confirms → **strongest cross-sample filter signal in the whole dataset.** Action: Option B code refactor (move BTC RSI / BTC ADX Dir into "BTC Independent Filters" section, independent of btc_global toggle) and set `btc_rsi_min_long: 55`.
14. **Long BTC RSI 60-65 still winning?** Historical: 44 combined trades, 73% WR, best zone. If confirmed → keep `btc_rsi_max_long: 65` or relax to 70.
15. **Short BTC RSI <35 still winning?** Historical: 48 combined trades, 77% WR across Mar 30 + Apr 6 + Apr 12 + Apr 13. If confirmed → this is the golden short zone.
16. **Short BTC RSI 45-50 still losing?** Historical: 14 combined trades, 36% WR. If confirmed → set `btc_rsi_max_short: 45`.
17. **Short BTC RSI <30 × BTC ADX cross-tab — IS IT CONDITIONAL?** Early Apr 15 data (4 trades) showed <30 winning only when BTC ADX 20-25 (1/1 win), losing when BTC ADX 25+ (0/3 wins). Historical data had BTC RSI <30 mostly at BTC ADX 20-25 (Apr 13 had 6/7 of <30 shorts in that bucket, 83% WR). Hypothesis: the <30 short edge may be conditional on BTC being in moderate trend (ADX 20-25), not strong trend. At 100 trades, build the full BTC RSI <30 × BTC ADX cross-tab. Decision logic:
    - If BTC RSI <30 + BTC ADX 20-25 wins (≥10 trades, ≥65% WR) AND BTC RSI <30 + BTC ADX 25+ loses (≥10 trades, ≤40% WR) → **conditional filter**: allow `btc_rsi < 30` entries ONLY when `btc_adx < 25`. Requires code change (not just config).
    - If BTC RSI <30 loses across ALL BTC ADX buckets (≥15 trades combined, <45% WR) → **raise `btc_rsi_min_short: 30`** (remove <30 entirely).
    - If BTC RSI <30 wins across all BTC ADX buckets → keep current setting, historical pattern confirmed.
    - If N <10 → inconclusive, wait for 200-trade sample.

Note: these filters currently only activate when `btc_global_filter_enabled = true` (which is OFF). Phase 2 code work: move them into "BTC Independent Filters" section (like BTC ADX range already is) so they can be applied without bundling BTC regime alignment.

**Pre-committed BTC RSI × BTC ADX rule validation (5th-sample check)**
17c. **Validate each pre-committed BTC RSI × BTC ADX rule against fresh data.** For each of the 4 HARD BLOCK rules (L-B1, S-B1, S-B2, S-B3) and 4 PREMIUM ZONE rules (L-P1, L-P2, S-P1, S-P2) documented in the Pre-committed Phase 2 section: count trades in the bucket, compute WR. Apply gates:
    - HARD BLOCK rule shows ≥55% WR with ≥5 trades → drop the block (pattern broke in 5th sample)
    - PREMIUM ZONE rule shows ≤55% WR with ≥5 trades → demote from VERY_STRONG (no leverage boost)
    - <4 trades in bucket → insufficient data, ship rule unchanged, re-validate at 200 trades
    
    Special attention: `SHORT + <30 RSI + ADX 25-30` and `+ ADX 30-35` — currently PENDING. If Phase 1 confirms losing (≤50% WR with ≥5 trades), add them to HARD BLOCKS. If still winning (≥65% WR with ≥5 trades), add to PREMIUM ZONES.

**BTC ADX × BTC Slope — 2-sample confirmed shorts finding (HIGHEST PRIORITY after Q13)**
17b. **BTC ADX 18-27 + slope falling WINS; BTC ADX ≥28 + slope falling LOSES — 3rd-sample confirmation.** Apr 13 + Apr 15 combined:
    - BTC ADX 18-27 + slope falling: **27 shorts, 81% WR, +$19.93** (equivalent to what the classifier calls HEALTHY_BEAR)
    - BTC ADX ≥28 + slope falling: **27 shorts, 52% WR, -$10.71** (classifier: STRONG_BEAR + BEAR_EXHAUSTED)
    - Same sample size, opposite outcomes. If 3rd sample replicates → **strongest short filter in the dataset.**
    
    Decision logic at 100 trades (TBD Phase 2 filter):
    - If shorts in BTC ADX ≥28 + slope falling still lose (≥10 trades, ≤50% WR) AND shorts in BTC ADX 18-27 + slope falling still win (≥10 trades, ≥65% WR) → **implement raw-dimension filter**: block shorts when `btc_adx >= 28 AND btc_slope < 0`. Expressed in the proposed "BTC RSI × BTC ADX Cross-Filter" UI as: no direct RSI rule needed; a separate "BTC ADX max for shorts in downtrend" cap, or a rule that disallows the specific (BTC ADX ≥28, slope < 0) combination.
    - If patterns don't replicate → reopen analysis.
    - This overlaps Q17 (BTC RSI <30 × BTC ADX). Raw-dimension filters handle both: the conditional "BTC RSI <30 only when BTC ADX <25" and the "block ADX ≥28 + slope falling" are both expressible as cross-filter rules without needing a regime classifier.

**New SHORT questions (Apr 13 was 1-sample)**
18. Short EMA5 Stretch: 0.25-0.30% best, 0.16-0.20% worst? Replicates = add `short_min_ema5_stretch` filter.
19. Short EMA5-EMA8 gap 0.18-0.20% best bucket replicate?
20. Short ADX 25-30 > 30-35 replicate?

**New exit validation (Apr 14 changes)**
21. Did TP 0.50 / pullback 0.20 actually capture more of the post-peak move? Compare Apr 13 winners avg close +0.39% vs new sample winners avg close %.
22. Any "Positive, No BE" regression (trades that used to exit profitably now hitting SL)?

### Known issues flagged for investigation AFTER 100-trade review
- **VERY_STRONG**: disable if 2nd sample confirms underperformance
- **TAKER_FALLBACK entries**: block entirely if 2nd sample confirms WR gap
- **FL_DEEP_STOP weak-peak path**: if regime_change_exit didn't help, code-level review of FL2 logic in `services/trading_engine.py`. Candidate rule: "if flagged trade's peak <0.3% AND current P&L negative, exit immediately rather than waiting for deep_stop."
- **Long ADX max**: cap at 22 vs leave at 25 — decide from 3rd sample
- **Short-specific filters**: `short_min_ema5_stretch = 0.20` candidate if Apr 13 short pattern replicates
- **BTC RSI independent filters** (Option B code refactor): Move `btc_rsi_min/max_long/short` and `btc_adx_dir_long/short` OUT of the `if btc_global_enabled:` block in `services/trading_engine.py:3067+` and INTO the independent BTC filter section (alongside BTC ADX range, which is already independent per code comment). Add UI toggles in "BTC Independent Filters" section. If Phase 1 confirms the 4-sample BTC RSI patterns, apply new ranges simultaneously: `btc_rsi_min_long: 55`, `btc_rsi_max_short: 45`. Ship the refactor + new ranges in one Phase 2 deploy.
- **BTC RSI × BTC ADX Cross-Filter (Phase 2 code work — preferred implementation for all BTC-level conditional filters)**: Build a BTC-level RSI×ADX cross-filter section in the UI, mirroring the existing pair-level `RSI x ADX Cross-Filter`. Separate LONG rules and SHORT rules tables. Each rule: "for BTC RSI range [min, max], require BTC ADX in range [min, max] (or ≥ X, or ≤ X)." Empty = allow all combinations. Stored as config strings `btc_rsi_adx_filter_long` and `btc_rsi_adx_filter_short` (same string format as existing pair-level `rsi_adx_filter_long/short`). This single mechanism replaces:
  - Hard-coded conditional "allow BTC RSI <30 only when BTC ADX <25" logic
  - Regime-based blocking ("skip STRONG_BEAR shorts")
  - Static `btc_rsi_min/max_long/short` caps (which the cross-filter can also express)

  Advantages: no classifier to maintain, no arbitrary thresholds baked in, any future BTC conditional filter becomes a UI rule rather than code work. File locations to modify: `services/trading_engine.py` (filter evaluation), `config.py` (schema), `templates/index.html` (UI section).

### Cross-sample confirmed entry findings — BTC RSI (4 samples: Mar 30 + Apr 6 + Apr 12 + Apr 13)

These are the strongest cross-sample patterns in the entire dataset. Each bucket result replicates across 4 different configs over ~5 weeks. NEXT REPORT is the 5th sample to confirm.

**LONGS:**
| BTC RSI | Combined n | Combined WR | Pattern | Current config | Target if confirmed |
|---|---|---|---|---|---|
| 35-40 | 6 | 17% | **Consistent loser** (negative $ in 3 of 4 samples) | min: 40 | keep |
| 40-45 | 12 | 66.7% | Mixed → acceptable | min: 40 | keep |
| 45-50 | 12 | 75% | Winning | — | — |
| **50-55** | **22** | **50%** | **Negative $ in ALL 4 samples — strongest historical signal** | — | **min: 55** |
| 55-60 | 35 | 63% | Winning (positive $ in 2 of 4) | — | — |
| **60-65** | **44** | **73%** | **Best long zone** | max: 65 | keep or relax to 70 |
| 65-70 | 6 | 67% | Small sample, positive | — | — |
| 70+ | 13 | 62% | Mixed — watch for exhaustion | — | watch |

**SHORTS:**
| BTC RSI | Combined n | Combined WR | Pattern | Current config | Target if confirmed |
|---|---|---|---|---|---|
| **<30** | **23 (+4 Apr 15)** | **83% historical, 25% Apr 15** | **⚠️ Possibly conditional on BTC ADX** — Apr 15 early data (4 trades) flipped to losing. Cross-tab hint: wins at BTC ADX 20-25, loses at BTC ADX 25+. Historical samples had BTC in moderate ADX; current sample has BTC in strong trend. | min: 25 | **Check Q17 at 100 trades** |
| 30-35 | 25 | 72% | Strong winner (positive in 2 of 3) | — | — |
| 35-40 | 48 | 60% | Mixed — largest bucket but volatile | — | — |
| 40-45 | 19 | 63% | Mixed — regime-dependent | — | — |
| **45-50** | **14** | **36%** | **Consistent loser** (negative in both samples tested) | max: 60 | **max: 45** |
| 50+ | 3 | 33% | Small sample, losing | — | — |

### Cross-sample confirmed entry findings — BTC ADX × Slope for SHORTS (Apr 13 + Apr 15)

**2-sample confirmed — one of the strongest findings in the dataset.** Below is expressed in raw dimensions (BTC ADX bucket + BTC slope direction) rather than regime labels. Source tables archived at `reports/btc_regime_tables_not_in_main_reports.md` (Apr 13 per-regime breakdown was from a user-provided screenshot — retained for historical reference only, not for filter-design input).

| Raw dimensions | Apr 13 | Apr 15 | **Combined (69 shorts)** | Verdict |
|---|---|---|---|---|
| **BTC ADX 18-27, slope falling** | 18 / 83% / +$14.17 | 9 / 78% / +$5.76 | **27 / 81% / +$19.93** | ★ BEST — winning zone, consistent both samples |
| **BTC ADX ≥28, slope falling** | 20 / 50% / -$10.52 | 7 / 57% / -$0.19 | **27 / 52% / -$10.71** | ★ LOSER — same sample size, opposite result |
| BTC ADX ≥28, slope falling, RSI ≤30 | 1 / 100% / +$0.81 | 2 / 0% / -$1.07 | 3 / 33% / -$0.26 | Small N, directionally negative (≡ classifier's BEAR_EXHAUSTED) |
| BTC ADX <18 (any slope) | 7 / 43% / -$2.54 | — | 7 / 43% / -$2.54 | Losing (Apr 13 only) (≡ classifier's CHOPPY_WEAK) |
| BTC ADX ≥18, `|slope|` <0.02% | 4 / 25% / -$2.69 | — | 4 / 25% / -$2.69 | Losing (Apr 13 only) (≡ classifier's CHOPPY_FLAT) |

**Key insight:** The winning bucket and losing bucket differ ONLY by BTC ADX cutoff (27 vs 28 with slope falling). 27 trades each, opposite outcomes. This signal lives in raw dimensions — no regime label needed.

**Phase 2 filter candidate (TBD — not yet committed):** Block short entries when BTC ADX ≥28 AND slope falling. Keep BTC ADX 18-27 + slope falling (the winning zone). Estimated impact on Apr 13 + Apr 15 combined: cut 41 losing trades (-$16), keep 27 winning trades (+$19.93), PF jumps from current ~1.1 to ~3. **Implementation mechanism:** the proposed BTC RSI × BTC ADX Cross-Filter UI (see "Filter design principle" section above). **Exact filter definition to be decided at 100-trade review based on confirmation in fresh data.**

### Quant methodology notes (avoid repeating mistakes)
- **Never pool raw trades across different configs.** Each run = different strategy. Use cross-sample patterns only.
- **Non-monotonic single-variable patterns are usually confounds.** If a mid-range bucket loses while both flanks win, something ELSE (ADX, gap, breadth, regime) is varying with that bucket. Don't filter on non-monotonic holes.
- **Small N (<10 trades per bucket) is noise, not signal.** Apr 8's "EMA5 Stretch 0.16-0.20% = 100% WR" didn't replicate because it was 4-5 trades.
- **2-sample confirmation raises confidence from "hypothesis" to "likely real."** 3-sample is "structural finding." 1-sample findings are hypotheses until replicated.
- **Absolute P&L is leverage-dependent; WR and Avg% are leverage-invariant.** Compare samples using WR/Avg% to normalize across leverage changes.
- **"More aligned filters" ≠ "higher edge."** Apr 13's Entry Quality Score table showed Score 3 (long) and Score 2-4 (short) were best, while Score 4-5 (long) and Score 5 (short) UNDERPERFORMED. Over-aligned conditions often signal exhaustion/overextension. Don't assume stricter = better.

## April 16, 2026 — SUIUSDT Reconciler Race Guard (EXTERNAL_CLOSE mislabeling bug)

### What happened
SUIUSDT LONG closed cleanly at 13:02:11.077 UTC on a trailing-stop L1 trigger (high=0.9839, fill=0.9789, pnl=+$0.9095, slippage 0%). The bot's trailing-stop path committed the correct exit reason (`TRAILING_STOP L1`) to the DB at 13:02:12.444. **55 ms later** the monitor reconciler overwrote the same row as `EXTERNAL_CLOSE @ 0.9789` because from its own session's view it saw "DB status=OPEN, Binance shows no position = orphan."

The trade executed perfectly. Only the label was wrong. But that label is what the post-trade report uses to attribute exits, so trailing-stop effectiveness and the `EXTERNAL_CLOSE` rate in reports were both being silently distorted.

### Root cause
Two state-update paths write to the same Order row without coordination:
1. **Trailing-stop close path** (`services/trading_engine.py::_close_position_locked`) — fires close, waits for Binance fill, commits `status=CLOSED, close_reason=TRAILING_STOP L1`.
2. **Monitor reconciler** (`main.py::_reconcile_open_orders`, runs every 60 s in live mode) — polls Binance open positions, re-reads DB open orders, and for each DB row not found on Binance writes `close_reason=EXTERNAL_CLOSE`.

When the reconciler tick lands inside the narrow window between "Binance filled the close" and "DB commit from close path settled," the reconciler's own SELECT sees the row as still OPEN (its session's snapshot was taken before the close path committed) and issues a conflicting UPDATE. Last write wins. The trailing-stop label loses.

This is not a new code path — it became visible because the Apr 8 work made the bot correctly check Binance before retrying. The reconciler existed before but had enough other bugs masking this specific race.

### Fix (Option 2: intent-to-close flag)

Chose Option 2 over Option 1 (read-before-write `exit_reason` check) because Option 1's race window is only shrunk, not closed, and Option 2's pattern generalizes to every future close path without re-introducing the same race each time. Aligns with the "build to scale from day one" core principle — Option 1 would have held up at today's 20 trades/day but broken under the concurrency profile of thousands of trades/day. Option 2 is the same pattern that survives the Postgres migration, the multi-tenant rebuild, and the WebSocket-user-data-stream refactor without rework.

**Mechanism:**
- New columns on `Order`: `closing_in_progress` (bool, default False) and `close_initiated_at` (datetime, nullable).
- Bot publishes intent **before** sending the Binance close order via `trading_engine._mark_close_in_progress(db, order_id)` — sets the flag and timestamp in its own committed transaction so other sessions can see it immediately.
- Reconciler skips rows where `closing_in_progress=True AND close_initiated_at > now − CLOSE_INTENT_STALE_SECONDS` (120 s).
- Stale flags (≥ 120 s, indicating a crashed close path) are ignored so orphan rows still get reconciled eventually.
- Fails open: if the flag commit fails under lock contention (5-attempt retry), the close proceeds anyway and the guard is simply inactive for that one close. No regression vs pre-fix behaviour.

**Files changed:**
- `models.py` — added `closing_in_progress`, `close_initiated_at` columns with the incident attribution in the comment block.
- `database.py` — auto-migrate stanza adds the columns to existing DBs.
- `services/trading_engine.py` — `_mark_close_in_progress` helper + call site at top of `_close_position_locked` (live mode only).
- `main.py` — `CLOSE_INTENT_STALE_SECONDS = 120` constant + flag check at top of `_close_orphan_orders` loop, logs `[RECONCILE_SKIP]` (fresh) or `[RECONCILE_STALE_INTENT]` (stale).
- `tests/test_reconciler_race_guard.py` — 4 scenarios (baseline unset, fresh intent skipped, stale intent reconciled, null-timestamp safety). All pass.

### Sizing of the 120 s stale threshold
Worst-case bot close flow:
- 3 exit retries × 2 s sleep = 6 s
- 3 exit retries × 5 s DB busy_timeout = 15 s
- 1 s post-close verify sleep + API = ~1.5 s
- 5 DB commit retries × ~5 s each = ~25 s

Realistic ceiling ~45-50 s. 120 s gives safety margin. Beyond that we assume the close path crashed and let the reconciler recover the row.

### How to diagnose a recurrence
Signs the race is back (or a new close path is bypassing the guard):
1. **Report shows `EXTERNAL_CLOSE` trades** with non-zero P&L and exit_order_type `TAKER` (not `EXTERNAL`). Pre-fix Apr 16 SUIUSDT had these. A truly external close (user manually closed on Binance UI, liquidation, etc.) typically has `exit_order_type=EXTERNAL`.
2. **No `[RECONCILE_SKIP]` log lines when `EXTERNAL_CLOSE` fires** inside a 60 s window where the bot also logged `[CLOSE_COMMITTED]` for the same pair. Either the flag wasn't set (bug in the close path that skipped `_mark_close_in_progress`) or was stale (close path took longer than 120 s — investigate lock contention).
3. **`[CLOSE_INTENT_FAIL]` appearing frequently** — the intent commit is losing to lock contention. Root cause is SQLite write contention; real fix is the Postgres migration (C3).

### Future analysis hooks
- The `closing_in_progress` flag + `close_initiated_at` columns are stored even for successfully-closed orders (nothing clears them). They can be used for future audits:
  - "How often did the guard activate?" → count orders with `close_initiated_at IS NOT NULL AND status='CLOSED'`
  - "How long does the bot's close path actually take?" → `closed_at - close_initiated_at` per trade, bucket the distribution, confirm the 120 s ceiling is comfortably above p99.
- If `[RECONCILE_STALE_INTENT]` ever fires in production, treat it as a first-class incident: the bot's own close path took > 120 s, which at current scale should never happen. Likely signal of lock starvation, greenlet deadlock, or a fresh infrastructure regression.

### Impact on Phase 1b data integrity
Any `EXTERNAL_CLOSE` trades in pre-Apr-16 samples may be mislabeled trailing-stop (or other bot-initiated) exits. When comparing new (post-fix) samples against historical data, do **not** treat pre-fix `EXTERNAL_CLOSE` counts as ground truth — they over-count external closes and under-count whichever exit type (likely TRAILING_STOP L1) lost the race. For 5th-sample and later report comparisons, expect `EXTERNAL_CLOSE` to drop toward ~0% and the corresponding gained trades to appear under their real bot-initiated reason.

## April 17, 2026 — Broker-Side Protective Stops: REMOVED after failed rollout

**Feature status: REMOVED.**  Originally shipped in commit `edb6970` and iterated through 4 hotfixes, the broker-side protective stops feature could not be made functional for this Binance account.  All code, UI, config, and tests related to the feature were removed in a single clean commit.  The `Order.protective_sl_order_id` / `protective_tp_order_id` columns were kept (all NULL) to avoid a schema migration and to permit future re-attempt without a column re-add.

### Forensic trail — what we tried and why each failed

| Hotfix | Commit | Change | Binance error |
|---|---|---|---|
| Initial | `edb6970` | `reduceOnly=true + closePosition=true` + `type='STOP_MARKET'` | `-1106` "Parameter 'reduceonly' sent when not required" |
| #1 | `20c4e41` | Removed `reduceOnly`, kept `closePosition=true` | `-4120` "Order type not supported for this endpoint. Please use the Algo Order API endpoints instead." |
| #2 | `bf1ceee` | Switched to `reduceOnly=true + amount=quantity`, dropped `closePosition` | `-4120` (same error) |
| #3 | `ed920e0` | Added `portfolioMargin=True` (theory: account in PM mode) | `-2015` "Invalid API-key, IP, or permissions" — account confirmed NOT in Portfolio Margin |
| #4 | `7a49a56` | Reverted PM, switched to CCXT unified syntax (`type='market' + stopLossPrice/takeProfitPrice`), dropped `GTE_GTC` | `-4120` (SAME error as hotfixes #1 and #2) |

### Why we stopped

After hotfix #4 produced the same `-4120` error as hotfixes #1 and #2 on the standard `/fapi/v1/order` endpoint, the root cause was determined to be something about this specific Binance account + CCXT version combination that does not yield to parameter-level fixes.  The Portfolio Margin routing alternative (`/papi/v1/um/conditional/order`) was ruled out (account not PM-enrolled).  Without empirical diagnostic against real Binance (would have required handing API keys to a diagnostic script), further blind hotfix iteration was deemed unsafe: each deploy risks an in-flight maker-order orphan like the DOTUSDT incident during the hotfix #4 deploy window.

### Known production impact during the failed rollout

One orphan position (DOTUSDT) was created on Binance without a corresponding DB Order row when the hotfix #2 deploy (bf1ceee) killed the bot process mid-way through a 20-second MAKER_ENTRY wait.  Binance filled the maker order during the restart gap.  The orphan was detected when Binance showed 5 open positions vs the bot's UI showing 4.  Resolution: user manually closed DOTUSDT on Binance UI.  The error-spam from `-4120` and `-2015` during the rollout did not cause any direct P&L impact — the bot's internal exits continued to work correctly throughout, and no positions were lost to the failed protective-stops logic.

### Current risk management posture

The bot relies exclusively on its **internal, in-process exits** for risk management.  These are unchanged from what has been running successfully since the Apr 14 baseline:
- Main SL at `-0.9%`
- Trailing stop with `TP 0.50 / pullback 0.20`
- Signal Lost Flag + Security Gap at `-0.8% to -0.9%`
- FL2 deep stop at `-1.0%`
- FL_EMERGENCY_SL backstop at `-1.2%`
- Regime Change Exit on opposite BTC regime flip

These are well-tested and have been handling all closes during the rollout period.  **No code change is needed to restore normal behavior** — the removal is additive (takes away the non-functional broker layer, everything else continues unchanged).

### What would need to be investigated before any re-attempt

Do NOT re-attempt this feature without first:

1. **Empirical diagnostic** — run a script that places STOP_MARKET with various parameter combinations against live Binance (requires API keys).  The `tests/diag_stop_market_binance.py` file was removed with this commit but the script design is documented in CLAUDE.md (git blame this section) if useful.
2. **Check Binance account capabilities** — call `/fapi/v1/account` and `/fapi/v2/account` to inspect account type, margin mode, and feature flags.  Also check `exchange.load_markets()` for any per-symbol restrictions.
3. **Check Binance API documentation for STOP_MARKET changes** — Binance may have deprecated `/fapi/v1/order` for conditional orders on some account tiers.  Verify the current documented endpoint for STOP_MARKET on standard USDⓈ-M Futures.
4. **Verify CCXT 4.4.100+ behavior** — check if CCXT has a newer method specifically for Futures Algo Orders (separate from the Portfolio Margin `/papi/*` routing).
5. **If all else fails, consider support ticket** — submit the exact request + error to Binance API support.

### Alternative protection layers to consider instead of a re-attempt

Since the broker-side stops primarily existed as **insurance for the Apr 11 failure mode** (instance replacement / bot outage leaving positions orphaned), other protections can address the same risk without requiring broker-side stop orders:

- **C1 (S3 DB snapshot)** — pending from CLAUDE.md Apr 11 hardening.  If the bot restarts after an outage, it can recover the DB state and resume managing open positions normally.  Does not protect during the outage itself but reduces the orphan window from "unbounded" to "time-to-restart."
- **CloudWatch alarms → auto-close via Binance Web API** — a Lambda function triggered by bot-health alarms that calls Binance directly to close all open positions.  Heavier infrastructure but achieves the same insurance goal.
- **User-side manual monitoring** — a separate lightweight script/phone alert that notifies if the bot hasn't placed any API calls in N minutes.  User manually closes positions if alert fires.

The broker-side protective stops approach is the simplest in theory but has proven technically incompatible with this Binance account's current configuration.  Alternative approaches above should be considered before any re-attempt of broker-side stops.

## April 17, 2026 — Broker-Side Protective Stops (OLD — original design, kept for reference only)

### The gap this closes
The bot's exit logic is in-process — trailing stops, BE levels, FL flags, regime change exits all run inside the monitor loop. If the bot dies, stalls, gets rate-limited, or the EC2 instance is replaced (Apr 11 scenario), **open positions drift unmanaged until recovery.** In the worst case that happened Apr 11, ~40 positions stayed open for 8 hours.

Broker-side protective stops fix this by placing `STOP_MARKET` + `TAKE_PROFIT_MARKET` orders on Binance immediately after each position opens. Binance's matching engine enforces them — they fire regardless of bot state.

### Mechanism

**On every live-mode position open** (`services/trading_engine.py::open_position`, right after the entry commit), the bot calls `binance_service.place_protective_stops(symbol, direction, entry_price, sl_pct, tp_pct)` which places two Binance Futures orders:

| Order | Type | Side | Trigger (LONG) | Trigger (SHORT) | Flags |
|---|---|---|---|---|---|
| Protective SL | `STOP_MARKET` | Opposite of position | entry × (1 − sl_pct/100) | entry × (1 + sl_pct/100) | `reduceOnly=true`, `closePosition=true`, `workingType=MARK_PRICE`, `timeInForce=GTE_GTC` |
| Protective TP | `TAKE_PROFIT_MARKET` | Opposite of position | entry × (1 + tp_pct/100) | entry × (1 − tp_pct/100) | same |

### The `closePosition=true` flag — why it works so cleanly

Binance auto-cancels a `closePosition=true` order as soon as the position it references is closed by any other means. This means:
- When the bot's trailing stop fires → Binance automatically cancels our protective SL and TP. **No manual cleanup logic needed.**
- When the bot's emergency backstop at -1.2% fires first → Binance auto-cancels the protective orders.
- Only way for the protective orders to fill: the position is still open AND the price hits the trigger AND the bot hasn't closed it via other means.

This is the exact behaviour we want for "insurance-only" stops. No coordination, no orphan cleanup, no race conditions with the bot's own exits.

### Why `workingType=MARK_PRICE`

Binance offers `MARK_PRICE` (smoothed index price) vs `CONTRACT_PRICE` (last traded price) as trigger reference. MARK_PRICE is safer for safety stops because:
- Single-candle wick hunts don't fire false stops
- Flash crashes on low-liquidity pairs don't cascade our SLs
- Matches Binance's liquidation engine behaviour (consistency)

Trade-off: a genuine panic-dump scenario will fire MARK_PRICE slightly later than CONTRACT_PRICE would. For bot-is-dead insurance, later is fine; false triggers while bot is alive are the bigger concern.

### Default levels — 1.5% SL / 5% TP

| Level | Value | Rationale |
|---|---|---|
| `protective_sl_pct` | **1.5%** | Sits 0.3% below bot's deepest in-process stop (FL_EMERGENCY_SL at -1.2%). In normal operation the bot always hits its own stops first. Only fires when bot is dead. |
| `protective_tp_pct` | **5.0%** | Well above typical trailing exits (max observed in Apr 17 sample: +1.33%). Only fires on extreme spikes that the bot missed due to unresponsiveness. |

**Both configurable in `trading_config.json` + UI.** Tighter TP (e.g. 3%) accepted as a trade-off if the user wants broker-side to occasionally capture extreme winners before the bot's trailing can react.

### Reconciler integration — BROKER_SL / BROKER_TP close reasons

When the monitor reconciler detects a closed position on Binance that the bot didn't initiate, it now checks the fill price against the expected SL and TP triggers (within 0.15% tolerance for MARK_PRICE slippage + tick rounding):

- Fill within tolerance of expected SL → `close_reason = "BROKER_SL"` (vs generic `EXTERNAL_CLOSE`)
- Fill within tolerance of expected TP → `close_reason = "BROKER_TP"`
- Outside both windows → `EXTERNAL_CLOSE` (true user-initiated or liquidation)

This preserves analytical clarity: a BROKER_SL/TP entry in the report means **"bot was unresponsive, broker-side insurance activated"** — a first-class incident worth investigating, not a routine external close. Logged at CRITICAL level.

### Cost

- **Unfilled orders: free** (Binance doesn't charge for resting STOP/TP orders).
- **Filled orders: taker fee only** — same cost as the bot's own emergency exits. No additional cost.
- **API cost:** +2 calls per position open, 0 on close (auto-cancel). At 20 trades/day = 40 extra calls/day — negligible vs 1200/min rate limit.
- **Net steady-state cost: $0** unless the bot is actually unresponsive AND price hits a trigger.

### Configuration

```json
"protective_stops_enabled": true,
"protective_sl_pct": 1.5,
"protective_tp_pct": 5.0
```

UI: "Broker-Side Protective Stops" panel in the config section (amber-bordered, next to Pair Blacklist / New-Listing Filter).

### Fail-open safety

If `place_protective_stops` fails for any reason (Binance API error, network timeout, invalid price rounding):
- The main trade is already committed — NOT rolled back
- `protective_sl_order_id` / `protective_tp_order_id` stay NULL on the Order row
- Error is logged at ERROR level ("position is NOT protected on broker side")
- Bot continues normal operation; this position is exposed to the Apr 11 failure mode only

This is intentional: we never want protective-order failures to block trading or cause position state corruption.

### What this does NOT replace — still-pending C1 S3 snapshot

| Protection | Scope |
|---|---|
| **Protective stops (this)** | **Open positions** during bot outage — bounded loss, bounded gain |
| **C1 S3 snapshot** (pending) | **Trade history + bot state** during DB loss — enables clean recovery to correct mode |

Both are needed. Protective stops protect capital immediately. C1 protects analytical history and state integrity on infrastructure failure. Revisit C1 before moving off 1x leverage.

### Files changed

- `models.py` — `protective_sl_order_id`, `protective_tp_order_id` columns on Order
- `database.py` — auto-migrate stanza for new columns
- `config.py` — `protective_stops_enabled`, `protective_sl_pct`, `protective_tp_pct` fields
- `trading_config.json` — default values
- `services/binance_service.py` — `place_protective_stops`, `cancel_protective_stops` methods
- `services/trading_engine.py` — wire-up in `open_position` after DB commit
- `main.py` — reconciler BROKER_SL / BROKER_TP label detection; ConfigUpdate schema additions (also backfilled previously-dropped fields: `pair_blacklist`, `trading_pairs_limit`, `new_listing_filter_days`, BNB toggles)
- `templates/index.html` — UI panel + JS load/save handlers
- `tests/test_protective_stops.py` — 10 scenarios (price math, partial failure, cancel-is-success-when-gone, reconciler label detection). All pass.

### ConfigUpdate Pydantic bug fix (side-effect of this work)

While wiring the UI, I discovered that the `ConfigUpdate` Pydantic model was missing top-level fields that the UI had been sending for a while: `pair_blacklist`, `trading_pairs_limit`, `bnb_swap_enabled`, `paper_bnb_initial_usd`, `bnb_check_interval_hours`, `bnb_runway_hours`. Pydantic v2's default behaviour is to silently drop extras, so UI saves to these fields have been no-ops since introduction. All backfilled in the same commit. Future UI edits to blacklist / pair-limit / BNB settings will now actually persist.

## April 17, 2026 — Phase 1c Amendment (81-trade sample analysis + filter tightening)

### Sample that triggered Phase 1c
81 trades (40 LONG BULLISH + 41 SHORT BEARISH) collected Apr 15-17, archived at `reports/report_2026-04-17_phase1b_81trades.txt`.

Headline numbers (Avg P&L % per Core Operating Principles):
- LONG: 62.5% WR, +0.01% Avg, PF 1.08 — marginally profitable, don't break it
- SHORT: 46.3% WR, -0.09% Avg, PF 0.72 — drag on the system, needs fix
- Combined: -$4.83 over 2.9 days, roughly -0.04% Avg/trade

**Critical observations:**
1. **Short regime-flip dominance:** 17 of 41 shorts (41%) closed on REGIME_CHANGE or FL_REGIME_CHANGE. SAME_REGIME shorts: 73.3% WR, +0.07% Avg (consistent with prior 4-sample short edge). REGIME_SHIFT shorts: 30.8% WR. **The short edge is intact — it's being obscured by regime chop in this window.**
2. **Many prior multi-sample patterns broke** in this 5th sample (BTC RSI <30 shorts, BTC ADX 18-27+slope-falling shorts, EMA5-EMA8 0.08-0.12% shorts, Range Position 75-100% longs). These "broken" patterns do not justify deploying their opposites as filters — 1-sample break ≠ pattern reversal.
3. **Pair-level loss concentration:** RAVEUSDT = 3L, 0% WR, -$6.17 (biggest single-pair drag on longs). All 3 trades at RSI 50-55. Classic confound: the "RSI 50-55 losing" bucket was entirely RAVEUSDT.
4. **Pair ADX 18-22 is the single strongest winner bucket in the data:** 18 LONG trades, 77.8% WR, +0.30% Avg, +$9.94. Cleanest structural signal.
5. **VERY_STRONG tier underperforms STRONG_BUY** (2-sample confirmed): LONG 58% vs 64% WR, SHORT 37% vs 48% WR.
6. **Current short RSI×ADX cross-filter rule (`30-35:25,35-50:30`) funnels entries into toxic high-ADX zones** (RSI 30-35 × ADX 28-33 = 22% WR, -$5.85). The rule was inverting good logic.

### Phase 1c config changes (deployed Apr 17)

Every config change has ≥2-sample evidence OR is a targeted pair-specific kill. 1-sample-driven changes were considered and then walked back after cross-sample validation (see "Changes considered and rejected" below). Non-monotonic / broken-from-prior patterns were deliberately NOT touched.

**LONG side:**

| # | Change | Rationale |
|---|---|---|
| 1 | `pair_blacklist`: add `RAVEUSDT` | 3L, 0% WR, -$6.17 — surgical kill, preserves RSI 50-55 for other pairs |
| 2 | `adx_strong_long`: 15 → 18 | ADX 18-22 = 77.8% WR biggest winner; 15-18 = 40% WR. Net +$5.16 on retained subsample |
| 3 | `btc_adx_max_long`: 40 → 35 | BTC ADX 35-40 = 40% WR -$5.54 (loss zone); 25-35 = 73% WR |

**SHORT side:**

| # | Change | Rationale |
|---|---|---|
| 4 | `adx_strong` (short): 25 → 22 | Lowers SHORT STRONG_BUY floor. Explores previously-blocked ADX 22-25 zone. User decision: pure exploration, let the data validate. |
| 5 | `momentum_adx_max` (short): 33 → 28 | 2-sample confirmed (Apr 13 + Apr 17): ADX 28+ shorts = weak zone. Blocks regime-flip-prone high-ADX entry profile |
| 6 | `adx_very_strong` (short): 30 → 28 | Combined with max=28, VERY_STRONG short tier is effectively disabled (no ADX value can qualify for >28 and be ≤28) |
| 7 | `btc_adx_dir_short`: `"both"` → `"rising"` | **3-sample structural** (Apr 6 + Apr 13 + Apr 17): Rising BTC ADX shorts > Falling BTC ADX shorts every sample. Apr 17 Falling bucket was 25% WR -$5.89 (92% of total short losses). Required the Option B refactor below to actually take effect. |

**Pre-filter:**

| # | Change | Rationale |
|---|---|---|
| 8 | New config field `new_listing_filter_days: 180` | Age-based proxy for Binance's Seed Tag. Drops pairs whose `onboardDate` is within last 180 days BEFORE the top-N-by-volume cut, so "top 50" stays "top 50 of eligible pairs." Known limitation: age ≠ Seed Tag perfectly; pairs like RAVEUSDT (124d old) are caught at 180d, but older Seed-tagged pairs like HYPE (322d) are not. Manual `pair_blacklist` still needed for known-risky older pairs. |

**Code refactor (Phase 2 Option B):**

| # | Change | Rationale |
|---|---|---|
| 9 | BTC ADX Direction filter moved OUT of `if btc_global_enabled:` block in `services/trading_engine.py` | Pre-refactor: `btc_adx_dir_long/short` only fired when Macro Trend / BTC Global toggle was ON. Since user's config has `btc_global_filter_enabled: false`, the filter was effectively dead. Now runs independently (like BTC ADX range already did). Required for the change #7 to actually apply. |
| 10 | UI: BTC ADX Direction dropdowns moved from "Macro Trend Regime" section to "BTC Independent Filters" section (`templates/index.html`) | Matches the backend — the UI now correctly represents that these filters run standalone. |

### Changes considered and rejected during Phase 1c design (anti-overfit discipline)

**1-sample-driven changes walked back after cross-sample check:**

| Change I proposed | Why I walked it back |
|---|---|
| `ema_gap_threshold_short: 0.08 → 0.04` | 1-sample only (Apr 17). Apr 13 data showed the OPPOSITE pattern (0.08+ wins). Non-monotonic and unstable. Kept at 0.08. |
| `rsi_adx_filter_short: "30-35:25,35-50:30"` → remove | 1-sample-driven. Apr 9 data showed this rule's target cell had 84% WR. With new `momentum_adx_max: 28` the rule becomes a benign additional gate that blocks mediocre RSI 35-50 shorts. Kept the rule. |

**Never proposed (rejected on evidence):**

| Tempting change | Why rejected |
|---|---|
| Blacklist ETHUSDT, AAVEUSDT shorts | User decision: do not blacklist short pairs — not enough evidence, could be regime-specific |
| Raise LONG RSI min above 40 | RAVEUSDT-specific finding; blacklist is surgical, pair-RSI-threshold would over-cut |
| Cap LONG breadth max at 70% | 3-sample confirmed but signal weak (~-0.05%); drops 37% of long trades for marginal gain |
| Tighten BTC RSI ranges (longs or shorts) | 4-sample patterns broke in 5th sample — unstable, wait for more data |
| Deploy BTC ADX + slope conditional short rule | 2-sample pattern did NOT replicate in 5th sample |
| Add volume filter for shorts | 1-sample only. Apr 13 data showed the Low/Low short bucket at 66.7% WR; Apr 17 flipped to 40% WR. Magnitudes differ too much across samples to trust as filter. |
| Touch EMA5 Stretch buckets | Non-monotonic; over-cutting risk |
| Touch Range Position filter | 2-sample pattern just reversed in 5th sample |
| Re-admit ADX 28-30 zone for shorts via VERY_STRONG tier | User considered, then chose to skip. The 28-30 zone has 2-sample confirmed weakness (Apr 13 + Apr 17). Exploring only the unknown 22-25 zone is asymmetrically data-backed. |
| Leverage increase (e.g. STRONG_BUY 20x / VERY_STRONG 10x) | User proposed, flagged strongly as premature given PF 0.72 on shorts + 1-sample validation gap. User confirmed: stay at 1x until Phase 1c data validates. |

### About the `new_listing_filter_days` — known limitation

The filter uses `onboardDate` from Binance's USDT-M futures `exchangeInfo`. This is an AGE-based proxy for Binance's **Seed Tag / Monitoring Tag** — not the tag itself. The two correlate but aren't identical:

- Binance's Seed Tag list is published in announcements, NOT exposed via the standard futures API
- `onboardDate` catches literally-recent listings (0-180d old), which is a subset of Seed-tagged pairs
- Some Seed-tagged pairs (e.g. RAVEUSDT at 124d) ARE caught by the 180d filter
- Some older pairs (HYPE 322d, FARTCOIN 483d) with Seed-tag-like characteristics are NOT caught
- Pairs with missing `onboardDate` metadata are kept (fail-open, conservative)

**Expected behavior:** the filter is a defensive first line. It does NOT replace `pair_blacklist` — it's a broader net that catches brand-new listings before they ever reach the top-50. Known-risky older pairs must still go in `pair_blacklist` manually (as RAVEUSDT does).

**If the threshold proves too tight/loose:** adjustable in `trading_config.json` without code change. 180d was chosen as a balance between "catches most recent Seed Tags" (like RAVEUSDT) and "doesn't block established mid-age pairs." Revisit if the 200-trade review shows pairs sneaking through.

### Exit criteria for Phase 1c → Phase 2

Advance to Phase 2 when:
1. ≥160-180 total trades collected under Phase 1c config (pre-Apr-17 data stays separate per pooling rule)
2. LONG PF ≥ 1.2, SHORT PF ≥ 1.0 (both sides profitable, not just longs)
3. Pair ADX 18-22 long bucket replicates ≥70% WR at N ≥ 30 (3-sample structural confirmation)
4. Short side shows SAME_REGIME WR ≥ 70% on ≥30 same-regime trades (confirm short edge is real outside regime chop)

If Phase 1c fails (LONG PF < 1 or SHORT PF < 0.8): code-level review of regime change exit logic + short entry filters, not more filter tuning.

### Pooling rule update (Apr 17)

Phase 1b data (81 trades pre-Apr-17) and Phase 1c data (post-Apr-17) are SEPARATE sub-samples. Different configs. Do NOT pool raw trades. Compare across sub-samples using Avg P&L % only (per Core Operating Principles).

Phase 1c is now the active sample. Pre-Apr-17 data (Phase 1a, Phase 1b) becomes historical reference for cross-sample pattern validation.

### Exit Quality watchlist — LONG "early flameout" pattern (Apr 17, 1-sample finding, validate next batch)

Observed in the Phase 1b 81-trade sample (`reports/report_2026-04-17_phase1b_81trades.txt`) via the Exit Quality table: **LONG losers peak absurdly fast at absurdly low magnitude, then ride directly to the stop.**

**The signature:**

| Exit cell (LONG) | # | Peak % | PkMin (time to peak) | Close % |
|---|---|---|---|---|
| FL_DEEP_STOP L1 | 5 | **+0.04%** | **0.1 min (≈6 sec)** | −1.04% |
| FL_REGIME_CHANGE L1 | 6 | **+0.07%** | **1.7 min** | −0.56% |
| FL_EMERGENCY_SL L1 | 2 | +0.20% | 1.6 min | −1.32% |

Compare to LONG winners:

| Exit cell (LONG) | # | Peak % | PkMin | Close % |
|---|---|---|---|---|
| TRAILING_STOP L1 | 11 | +0.51% | 21 min | +0.46% |
| TRAILING_STOP L2 | 10 | +0.67% | 39 min | +0.45% |

**13 of 15 LONG losses in the sample follow the "flameout" path:** peak in under 2 minutes at under +0.10%, then ride to SL. Winners peak in 21–39 minutes at +0.5% to +0.7%.

**Importantly, the flameout pattern is NOT evident on SHORTs.** `FL_RECOVERED L1` shorts peaked at +0.02% in 6.9 min (similar low-magnitude early peak) but 100% recovered to close at only −0.39%. Early flat peaks on SHORTs are NOT predictive of loss — they often reverse and recover. On LONGs, they predict the trade will die.

**Hypothesis:** in choppy/mid-range markets, bullish signals fire at local intrabar tops. Price briefly confirms (0.04–0.10% bump within first candle) then sellers hit, trend fails, bot rides the full SL distance. This is structural "bought the top" behavior, not a filter-side issue (entry conditions for these flameouts vs winners are remarkably similar on single-variable axes).

**What to check in next batch (Phase 1c, target ≥15 LONG losses):**

1. **PkMin distribution of LONG losers vs winners.**
   - If the next batch's LONG losses also cluster at PkMin < 2 min AND Peak < 0.15% → pattern is 2-sample structural.
   - If losses are spread across PkMin ranges → pattern was 1-sample regime-specific noise.

2. **What fraction of LONG winners had Peak ≥ 0.15% by minute 10?**
   - This determines whether tightening `no_expansion_minutes` from 240 → 15 would cut legitimate winners.
   - Cross-reference with Never-Positive Deep Dive + per-trade data if available.
   - Want to see: winners hit +0.15% by minute 10 at ≥80% rate.

3. **Does the pattern hold across BTC regimes or only BULLISH?**
   - Phase 1b was pure-BULLISH LONG sample. Regime-specific pattern is weaker evidence.

**Candidate fixes (ranked, if pattern confirmed):**

| Option | What | Effort | Risk |
|---|---|---|---|
| 1 (⭐ preferred) | Tighten `no_expansion_minutes`: 240 → 15 | Config only, zero code | Low (some slow-start winners possibly cut) |
| 2 | Tight initial SL (−0.3% for first 2–3 min, widen to −0.9% after) | ~15 lines in `_check_exit_conditions` | Medium (whipsaw risk) |
| 3 | Entry momentum confirmation delay (wait 10–30s after signal, re-check) | ~50 lines; new tunable param | Medium, fits trend-following profile but adds complexity |

**Analytical note on Option 3 — to evaluate empirically in the next batch:**

> Option 3 (entry momentum confirmation delay) would change entry price, not entry frequency.  Could be better (entering at the reversal bottom — i.e., buying after the micro-spike that triggered the signal has exhausted) or worse (entering after the reversal has already started — i.e., buying into a fade).  **Unclear without simulation.**

Because the delay doesn't filter signals but just changes WHEN the order is placed, it would require either (a) a backtest against historical intra-bar tick data, or (b) a shadow-mode implementation that logs "what would have happened with delay X" on live trades without actually trading.  **Do not ship Option 3 blind.**  If Options 1 and 2 prove insufficient after 2-sample validation, the next step for Option 3 is shadow-simulation, NOT a live deploy.

**Default action at 2-sample confirmation: Option 1 (config-only) ships first.** Option 2 only if Option 1 insufficient.  Option 3 only after shadow-simulation justifies it.

**Do NOT act on 1-sample evidence.** Wait for Phase 1c data.

**Why this is specifically a LONG-side finding (asymmetry note):**
The Exit Quality table shows clear SL recovery on SHORTs (`FL_RECOVERED L1` at 100% recovery, `FL_TRAILING_STOP L1/L2/L3` all 100% recovery) that is completely absent on LONGs (all FL_* LONG buckets at 0% recovery). This is consistent with bearish regimes having more frequent bounces (pullbacks that recover) while bullish regimes showing more failed rallies (pullbacks that break through). If confirmed in 2nd sample, it justifies asymmetric exit handling (`signal_lost_flag_long_enabled` vs `..._short_enabled`).

## Three-Phase Plan to Make the Bot Profitable

### Phase 1 — Validate the baseline (CURRENT: 0-100 trades at 1x)

**Goal:** Build a clean sample at frozen Apr 14 config. Identify which setups have real positive expectancy vs which are noise.

**Config:** Apr 14 locked baseline (see above). 1x leverage both confidence levels. NO CONFIG CHANGES.

**Sample target:** ~100 trades, ideally ~50L + 50S for balance.

**Exit criteria to advance to Phase 2:**
1. PF > 1 overall (or clear on-track trajectory; PF 0.8-1.0 with correctable loss patterns is acceptable)
2. At least 40 longs + 40 shorts collected (below this, bucket analysis is too noisy)
3. Core 2-sample-confirmed patterns either replicate or clearly break

**If Phase 1 fails** (PF < 0.5 or WR < 45%): stop and diagnose before Phase 2. Likely requires deeper exit-logic fix (FL2 weak-peak path) before filter changes can help.

### Phase 2 — Redefine VERY_STRONG around confirmed edges (100-200 trades at 1x)

**Goal:** Rebuild the two confidence levels around what ACTUALLY works. VERY_STRONG becomes "high-conviction setup" (confirmed edge), STRONG_BUY stays as catch-all default. Still 1x leverage — proving the edge before leveraging it.

**IMPORTANT: BTC-level vs Pair-level filters are at different levels and must not be conflated:**
- **BTC-level filters** (HARD BLOCKS, PREMIUM ZONES from pre-committed rules): macro entry gating based on BTC RSI × BTC ADX. Implemented via `btc_rsi_adx_filter_long/short` config (the proposed BTC-level Cross-Filter UI).
- **Pair-level tier** (VERY_STRONG vs STRONG_BUY): per-pair setup quality based on pair ema_gap, range_position, breadth. Implemented via `confidence_levels.*` config.
- A single trade has BOTH: BTC filter decides "allowed at all?", pair-level tier decides "STRONG_BUY or VERY_STRONG?".
- PREMIUM ZONE rules (L-P1, L-P2, S-P1, S-P2) are BTC-level. They do NOT define VERY_STRONG — that's pair-level.

**Required code work before Phase 2 starts:**
1. **Build BTC RSI × BTC ADX Cross-Filter UI** (mirrors existing pair-level `RSI x ADX Cross-Filter`). Stored as `btc_rsi_adx_filter_long` and `btc_rsi_adx_filter_short` strings. Separate LONG rules and SHORT rules tables.
2. **Option B refactor**: move BTC RSI and BTC ADX Dir filters OUT of `if btc_global_enabled:` block in `services/trading_engine.py:3067+` into independent "BTC Independent Filters" section (like BTC ADX range already is).
3. **Implement the 8 pre-committed BTC-level rules** (4 HARD BLOCKS + 4 PREMIUM ZONES from the Phase 2 pre-committed section above) as the default filter config that ships with Phase 2 code.
4. **Redefine VERY_STRONG at PAIR level** using the documented pair-level rules:
   - OLD VERY_STRONG rule (stricter ADX >22) → REMOVE (pushes into the loser zone per Apr 13)
   - NEW VERY_STRONG rule: AND-combination of the 2-sample-confirmed pair-level winning conditions (see definition below)
   - STRONG_BUY unchanged: current default filters
   - Modify confidence-level logic in `services/indicators.py` (or wherever signal generation assigns confidence)
5. **Logging is NOT needed as a separate work item** — existing logged dimensions (`btc_rsi_at_entry`, `btc_adx_at_entry`, `ema_gap_5_8_at_entry`, `range_position_at_entry`, `market_breadth_at_entry`, etc.) already capture everything needed to derive rule matches at report time. A post-sample report query/script can compute which rule each trade matched — no schema change needed unless a specific dimension is found missing.

**Pre-committed VERY_STRONG definition (lock this BEFORE seeing Phase 1 data to prevent overfitting):**

*Long VERY_STRONG requires ALL of:*
- `ema_gap_5_8 ≥ 0.08%` (2-sample confirmed: below is loss zone)
- `range_position ≤ 75%` (2-sample confirmed: 75-100% underperforms)
- `market_breadth in [50%, 65%]` (2-sample confirmed: 70%+ overextension loses)

*Short VERY_STRONG requires ALL of (PENDING Phase 1 replication of short patterns):*
- `ema_gap_5_8 ≥ 0.10%`
- `ema5_stretch ≥ 0.20%` (Apr 13 showed 0.25-0.30% crushes)
- `ADX in [25, 30]` (avoid 30+ overextension)

*Rule:* Only activate short VERY_STRONG in Phase 2 if Phase 1 shorts replicate the Apr 13 stretch/gap/ADX patterns. If not replicated, short VERY_STRONG stays identical to STRONG_BUY at 1x (no gating).

**Pre-committed BTC RSI × BTC ADX Cross-Filter rules (locked Apr 15, based on 4-5 sample cross-tab aggregation):**

These rules are derived from aggregating the BTC RSI × BTC ADX cross-tab across Mar 30 + Apr 6 + Apr 12 + Apr 13 + Apr 15 samples. Each has N ≥ 4 trades combined, with consistent pattern across multiple samples under different configs. **Locked TODAY, before Phase 1 data arrives, to prevent overfitting.** Will be implemented via the BTC RSI × BTC ADX Cross-Filter UI (see Filter design principle above). At 100-trade review, each rule is re-validated against fresh sample — if 5th sample flips the sign meaningfully, the rule is dropped. Otherwise it ships as the default Phase 2 filter set.

**HARD BLOCKS (never trade these conditions):**

| Rule ID | Direction | BTC RSI | BTC ADX | Historical n | WR | Total $ | Samples |
|---|---|---|---|---|---|---|---|
| L-B1 | LONG | 50-55 | 25-30 | 6 | **17%** | **-$501** | 4 samples, negative in each |
| S-B1 | SHORT | 35-40 | 15-20 | 9 | 44% | -$103 | 4 samples |
| S-B2 | SHORT | 35-40 | 30-35 | 13 | 54% | -$130 | 3 samples |
| S-B3 | SHORT | 45-50 | 15-20 | 8 | 37.5% | -$2 | 2 samples |

**PREMIUM ZONES (candidates for VERY_STRONG tier if Phase 2 code supports per-bucket leverage):**

| Rule ID | Direction | BTC RSI | BTC ADX | Historical n | WR | Total $ | Samples |
|---|---|---|---|---|---|---|---|
| L-P1 | LONG | 60-65 | 20-25 | **19** | **74%** | +$126 | 4 samples — largest N long bucket |
| L-P2 | LONG | 60-65 | 30-35 | 4 | 100% | +$20 | 3 samples, 4/4 winners |
| S-P1 | SHORT | <30 | 20-25 | 13 | 77% | +$32 (excl Mar 30 outlier) | 4 samples |
| S-P2 | SHORT | 30-35 | 25-30 | 12 | 83% | +$1052 | 4 samples |

**Rules NOT committed (need Phase 1 data before locking):**
- `SHORT + <30 + ADX 25-30` and `SHORT + <30 + ADX 30-35`: historically winners but Apr 15 (4 trades) flipped sign. Hypothesis: <30 RSI edge may be conditional on ADX 20-25 only. Needs 5th sample to confirm.
- `LONG + 50-55 + ADX 20-25` (-$8, 7 trades, 43% WR): directional loser but N small. Watch at 100 trades.

**5th-sample validation gates at 100-trade review:**
- If any HARD BLOCK rule shows ≥55% WR with ≥5 trades → drop the block (pattern broke)
- If any PREMIUM ZONE rule shows ≤55% WR with ≥5 trades → demote to neutral (no VERY_STRONG boost)
- If a rule gets <4 trades at 100 → insufficient 5th-sample data, ship rule unchanged, re-validate at 200 trades

**What to expect at Phase 2 with proper definition:**
- VERY_STRONG should be a RARE signal (~15-25% of all entries). If it fires on >40% of entries, the rule is too loose.
- VERY_STRONG WR should be 75%+ if the rule is capturing real edge
- VERY_STRONG avg $/trade should be higher than STRONG_BUY (not just WR — expectancy matters)
- If VERY_STRONG looks identical to STRONG_BUY → the rules aren't capturing real edge. Redesign or abandon the tier system.

**Exit criteria to advance to Phase 3:**
1. VERY_STRONG at 1x shows WR ≥ 70% AND avg $/trade > STRONG_BUY avg $/trade by meaningful margin (>30% better expectancy)
2. VERY_STRONG sample size ≥ 20 trades per direction (40+ total) — enough to trust the edge
3. STRONG_BUY still near-breakeven or better (not gutted by VERY_STRONG siphoning the winners)

**If Phase 2 fails** (VERY_STRONG ≈ STRONG_BUY): tier system doesn't work for this strategy. Stay at single-tier 1x, focus on filter tightening instead.

### Phase 3 — Data-driven per-bucket leverage (100+ trades with variable leverage)

**Goal:** Apply leverage based on per-bucket expectancy data from Phase 2, NOT based on a crude tier system. Different BTC × pair setup combinations get different leverage based on their proven edge.

**Why NOT simple tier-based leverage (VERY_STRONG=2x, STRONG_BUY=1x):**
- A VERY_STRONG long in a neutral BTC zone has different edge than a VERY_STRONG long in a PREMIUM ZONE (L-P1 or L-P2)
- A STRONG_BUY short matching S-P2 (BTC RSI 30-35 × ADX 25-30, 83% WR historically) might have more edge than a VERY_STRONG long in a marginal zone
- Leverage should follow $EDGE per trade, not tier labels

**Phase 2 close deliverable (required INPUT for Phase 3):**

At end of Phase 2, produce a per-bucket expectancy table from the fresh data:

| Bucket | Direction | WR | Avg $ / trade | Expectancy | Variance | N | Recommended lev |
|---|---|---|---|---|---|---|---|
| S-P2 (RSI 30-35 × ADX 25-30) | SHORT | ? | ? | ? | ? | ? | ? |
| S-P1 (RSI <30 × ADX 20-25) | SHORT | ? | ? | ? | ? | ? | ? |
| L-P1 (RSI 60-65 × ADX 20-25) | LONG | ? | ? | ? | ? | ? | ? |
| L-P2 (RSI 60-65 × ADX 30-35) | LONG | ? | ? | ? | ? | ? | ? |
| VERY_STRONG + PREMIUM ZONE match | both | ? | ? | ? | ? | ? | ? |
| VERY_STRONG alone (not in PZ) | both | ? | ? | ? | ? | ? | ? |
| STRONG_BUY + PREMIUM ZONE match | both | ? | ? | ? | ? | ? | ? |
| STRONG_BUY alone (not in PZ) | both | ? | ? | ? | ? | ? | ? |

**Leverage assignment rules (derive from the above):**
- Expectancy > +$0.50/trade AND variance acceptable AND N ≥ 20 → **2.5x-3x**
- Expectancy +$0.20 to +$0.50/trade AND N ≥ 20 → **2x**
- Expectancy 0 to +$0.20/trade OR 10 ≤ N < 20 → **1.5x**
- Expectancy < 0 OR N < 10 → **1x** (default, conservative)
- Expectancy strongly negative across 2+ samples → **add to HARD BLOCKS, 0x (skip)**

**Implementation:**
- Leverage lookup table stored in config (JSON), keyed by (BTC RSI bucket, BTC ADX bucket, direction, pair tier)
- Engine computes applicable leverage at entry time
- If a trade matches multiple buckets, use the highest applicable leverage (user safeguard: enforce a hard cap, e.g., 3x max)

**Initial conservative starting point (Phase 3 entry):**
- NO bucket gets more than 2x in first Phase 3 run, regardless of Phase 2 data
- Step up to 2.5x / 3x only after 50-trade validation at 2x confirms edge survives leverage

**What to measure in Phase 3:**
- Per-bucket: leveraged P&L / unleveraged Phase 2 P&L. Should be ≈ leverage multiplier. Materially lower = leverage eroding edge.
- Variance per bucket: did drawdown widen disproportionately?
- Correlation of losses: if multiple leveraged positions open simultaneously and all SL together, you get a drawdown cluster. Cap total leveraged position count.

**Exit criteria to advance to Phase 4 (higher leverage on winning buckets):**
1. Each leveraged bucket shows actual $ return ≥ 1.7× its Phase 2 unleveraged $ return
2. Max drawdown across portfolio stays within acceptable range
3. Sample size per leveraged bucket ≥ 20 trades

**If Phase 3 fails** (leveraged P&L < 1.5× unleveraged for a bucket): drop that bucket back to 1x. Other buckets may still be fine — don't abandon the whole system.

### Phase 4+ — Scale up confirmed buckets, add new ones (ongoing)

Once Phase 3 validates per-bucket leverage:
- Buckets that showed ≥1.7× scaling at 2x → consider 2.5x, then 3x (each a separate 50-trade validation)
- Discover new buckets over time as sample sizes grow in currently-marginal zones
- Cap maximum per-bucket leverage at 3x-5x (beyond this, slippage and liquidity become dominant risks on 5m timeframe)
- Never increase a bucket's leverage faster than one step per validated 50-trade phase

### Anti-overfit rules (MUST follow at every phase)

1. **Lock the VERY_STRONG definition BEFORE seeing the data that would inform it.** Today (Apr 14) is the lock date. The definition above uses only 2-sample-confirmed criteria. Don't modify it based on Phase 1 data to "optimize" — that's textbook overfitting.

2. **Minimum sample sizes are hard gates.** Don't advance a phase with fewer trades than required. 20 VERY_STRONG trades for tier validation, 50 for leverage validation.

3. **If the rule produces near-zero signals, it's over-constrained.** Loosen one condition at a time until signal rate is 15-25% of entries. But loosen only within the 2-sample-confirmed range (don't introduce new criteria).

4. **Never leverage a 1-sample finding.** Short VERY_STRONG must wait for Phase 1 replication. Leveraging something that appeared once is coin-flip gambling.

5. **Symmetry check**: if Long VERY_STRONG has 3 criteria, Short VERY_STRONG should also have ~3 criteria of similar strictness. Asymmetric rules usually indicate one side is overfit.

### Rough timeline (at 20 trades/day paper rate)
- Phase 1 (100 trades): ~5 days
- Phase 2 (100-200 trades): ~5-10 days
- Phase 3 (100 trades): ~5 days
- **Total: ~3 weeks minimum** to get to validated 2x leverage

This is deliberately slow. Compressing the timeline is the most common way to blow up a new strategy.
