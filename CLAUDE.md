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
- **Paper vs live: comparable for strategy, NOT for fill mechanics.** All historical reports through Apr 17 were captured in LIVE mode despite some misleadingly displaying "Paper Trading: ON" (template bug fixed Apr 17 — `cfg.paper_trading` was being read instead of runtime `is_paper_mode`). Apr 18 was the first report with the corrected "Paper Trading: OFF (runtime) — LIVE mode" label. **When reading historical samples, treat them as live regardless of the displayed paper/live flag.** When running paper after Apr 18: filter/regime/bucket findings (BTC ADX cells, BTC RSI cells, pair ADX patterns, regime-change clusters, signal/exit logic) are directly comparable to live history because they share identical code paths. **Fill-mechanics findings are NOT comparable**: paper's `_simulate_maker_entry_paper` uses a "did WS price touch the limit?" check which over-fills relative to real orderbook competition. So paper systematically reports higher MAKER fill rate, near-zero slippage, and inflated MAKER-vs-TAKER WR gaps. **Do not use paper data to evaluate Amendment #6 (timeout), #8 (offset ticks), or any future fill-mechanics experiments — those need live samples.** Amendment #7 (signal re-validation) IS paper-equivalent because it's pure indicator math, not order placement. When the next paper batch arrives, label conclusions explicitly: "filter-layer (paper-OK)" vs "fill-mechanics (paper-biased, hold for live retest)".

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
| 3 | `btc_adx_max_long`: 40 → 35 → **25** (amended Apr 17 PM) | Initial change to 35 was 1-sample driven. 3-sample re-analysis (Apr 17 PM, see "Phase 1c amendment #2" below) showed the clean breakpoint is at 25, not 35. Kept as row 3 here with the full history; actual deployed value is 25. |

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

### Phase 1c amendment #2 (deployed Apr 17 PM) — BTC ADX caps from 3-sample cross-analysis

After the Apr 17 AM filter-tightening deploy, validation of the "late-cycle BTC ADX" SHORT watchlist hypothesis against the Apr 13 117-trade Entry Conditions by Close Reason table falsified the original pair-level signature (ADXΔ / RngPos / Breadth were identical across SHORT winners and losers). Re-aggregation on **raw BTC ADX magnitude** across 3 independent samples (Apr 6 + Apr 13 + Apr 17 Phase 1b = 259 trades) produced clean breakpoints that had been invisible in single-sample analysis. Mar 30 sample excluded (too old, different config). Per Core Operating Principle: Avg P&L % used throughout (invest amounts varied across batches).

**3-sample BTC ADX bucket performance:**

LONGs (126 trades, ex-MANUAL/EXTERNAL):

| BTC ADX | N | Avg P&L % |
|---|---|---|
| <20 | 8 | +0.19% |
| 20-25 | 52 | **+0.23%** |
| 25-30 | 70 | **−0.17%** |
| 30-35 | 5 | +0.52% (tiny N) |
| 35-40 | 5 | −1.11% (tiny N) |

SHORTs (133 trades):

| BTC ADX | N | Avg P&L % |
|---|---|---|
| <20 | 12 | −0.53% |
| 20-25 | 23 | +0.20% |
| 25-30 | 48 | +0.05% (flat) |
| 30-35 | 16 | **−0.63%** |
| 35-40 | 8 | **−0.36%** |

**Changes deployed Apr 17 PM:**

| # | Change | Rationale |
|---|---|---|
| 11 | `btc_adx_max_long`: 35 → **25** | **Strongest finding in the entire dataset.** LONG 20-25 = +0.23% on N=52 across 3 samples; LONG 25-30 = −0.17% on N=70 across 3 samples. N=122 informs the 25-line. 3 independent samples, 3 independent configs, all consistent. Config-invariant macro mechanism (BTC over-stretched = late-cycle entry). |
| 12 | `btc_adx_max_short`: 40 → **30** | Robust. SHORT 30-35 = −0.63% on N=16 across 3 samples (all negative direction); SHORT 35-40 = −0.36% on N=8 Phase 1b alone. Apr 13 FL_DEEP_STOP (10 @ −1.02%) = biggest single loss bucket in the entire dataset, exactly the trades this cap kills. |

**HELD back (not deployed, N too thin):**

| Change | Why held |
|---|---|
| `btc_adx_min_short`: 18 → 20 | SHORT <20 = −0.53% but only N=12 across 3 samples. Per CLAUDE.md's own anti-overfit rule ("Small N <10 per bucket is noise"), N=12 is marginal. Current min=18 already blocks the worst BTC ADX values seen (16.2, 16.9, 17.1). Re-evaluate at 200-trade sample. |

**Ex-post what-if on the 259-trade 3-sample dataset (these filters applied):**

| Metric | Before | After | Delta |
|---|---|---|---|
| Total trades | 259 | 131 | **−49%** |
| LONG Avg P&L % | −0.01% | **+0.23%** | +0.24 pct-pt/trade |
| SHORT Avg P&L % | −0.09% | **+0.10%** | +0.19 pct-pt/trade |
| Combined Avg P&L % | −0.05% | **+0.16%** | +0.21 pct-pt/trade |
| Projected PF | 0.8-1.0 | **1.5-1.7** | regime flip |

**Important caveats:**
1. The ex-post projection is not a forecast — it assumes the same market regimes and pair mixes persist. Forward P&L will differ.
2. Apr 13 LONG 25-30 (N=44 @ −0.44%) dominates the LONG improvement. If Apr 13 was regime-specific, forward improvement will be smaller.
3. Trade count halves (~5/day → ~2.5/day at same entry rate). The 100-trade Phase 1c sample will take ~2x longer to collect.
4. The two big caps (LONG max 25, SHORT max 30) are the 95% of projected improvement. The held-back SHORT min ≥20 is marginal (5%).

**Pooling rule (amended Apr 17 PM):** Pre-Apr-17-PM Phase 1c data (the Apr 17 AM 13-trade peek) and post-Apr-17-PM data should be **separated** at analysis time. The Apr 17 AM data is 1-config, 13-trade. The Apr 17 PM data is the real Phase 1c sample against which the 100-trade checkpoint should be run. If you want to reason about cumulative Phase 1c bucket counts, add the two sub-samples but flag N per sub-sample.

**What to measure at 100-trade Phase 1c checkpoint (in addition to the existing 22-question checklist):**
- **Did the caps work?** Count trades with `entry_btc_adx >= 25` (LONG) and `entry_btc_adx >= 30` (SHORT). Should be **zero**. Any non-zero count = filter not firing.
- **Did the kept buckets perform?** LONG 20-25 at 3-sample avg +0.23% — does fresh sample show ≥+0.15% Avg on ≥30 trades? SHORT 25-30 at 3-sample avg +0.05% — does fresh sample show ≥0 Avg on ≥30 trades?
- **Did the held-back <20 SHORT bucket get enough data to decide?** Target ≥10 SHORT trades in BTC ADX <20 in the fresh sample. If yes, decide on `btc_adx_min_short: 18 → 20` at the checkpoint rather than waiting for 200 trades.

### Phase 1c amendment #3 (deployed Apr 17 PM, same day as #2) — Revert pair-level ADX changes potentially confounded with BTC ADX

After shipping the BTC ADX caps (Amendment #2), a second pass through Amendment #1 revealed that three pair-level ADX changes made earlier the same day were likely doing work that the new macro filter now handles. Specifically: the Apr 13 + Phase 1b evidence that drove those pair-level caps could have been driven by correlated BTC ADX (pair ADX and BTC ADX correlate ~0.5), meaning the "loser buckets" at pair-level 28-33 (shorts) and 15-18 (longs) may have been loser buckets *because* their BTC ADX was high, not because their pair ADX was. With `btc_adx_max_long: 25` and `btc_adx_max_short: 30` now enforcing the macro slice, the pair-level caps became candidates for over-restriction.

**Reverted:**

| # | Revert | Amendment #1 value | Restored to | Confidence |
|---|---|---|---|---|
| 2 | `adx_strong_long`: 18 → **15** | 18 (up from 15 in #1) | 15 (Apr 14 baseline) | HIGH — was 1-sample-driven, contradicted by Apr 13 data. Low conviction originally. |
| 5 | `momentum_adx_max` (short): 28 → **33** | 28 (down from 33 in #1) | 33 (Apr 14 baseline) | MEDIUM — 2-sample evidence exists but likely BTC-ADX-confounded. Falsifiable at checkpoint. |
| 6 | `adx_very_strong` (short): 28 → **30** | 28 (down from 30 in #1) | 30 (Apr 14 baseline) | MEDIUM — gated with #5, reverts together. |

**Effective entry windows now:**

| Side | Before (Amendment #2 only) | After Amendment #3 |
|---|---|---|
| LONG | Pair ADX [18, 25] AND BTC ADX [18, 25] | **Pair ADX [15, 25]** AND BTC ADX [18, 25] |
| SHORT | Pair ADX [22, 28] AND BTC ADX [18, 30] rising | **Pair ADX [22, 33]** AND BTC ADX [18, 30] rising |

**Falsification condition (what would overturn this revert):**

If fresh Phase 1c data shows:
- **LONG** pair ADX 15-18 bucket ≤0% Avg P&L on ≥10 trades, AND these trades are NOT concentrated at BTC ADX ≥23 (i.e., the loss isn't explained by the macro slice being near the cap) → re-tighten `adx_strong_long` to 18.
- **SHORT** pair ADX 28-33 bucket ≤0% Avg P&L on ≥10 trades, AND these trades are NOT concentrated at BTC ADX ≥28 → re-tighten `momentum_adx_max` to 28.

If the losing trades in those pair-level buckets DO concentrate at the high-BTC-ADX edge of our current macro filter, the revert is validated (BTC ADX filter is doing the real work; pair-level filter was confounded).

**Rationale for reverting all three now (rather than waiting for checkpoint):**

1. Over-restriction isn't free. Trade rate at Amendment #2 alone was projected to drop ~50% (5/day → 2.5/day). Adding redundant pair-level filters on top compounds the slowdown. Slower data = slower validation of the BTC ADX caps themselves.
2. Under-restriction is bounded and correctable. SL = −0.9%. Worst case from re-admitted losers is ~0.5% per trade × N trades, observable within a week.
3. Falsification is better than assumption. Keeping #5/#6 on the theory they're non-redundant meant never actually testing it. Revert + measure produces evidence.

**What this does NOT revert:**
- Pair blacklist (#1) — independent
- `adx_strong` short 25 → 22 (#4) — exploration, independent
- `btc_adx_dir_short` "both" → "rising" (#7) — 3-sample structural, non-redundant with magnitude cap
- `new_listing_filter_days: 180` (#8) — independent
- Option B refactor (#9/10) — infrastructure
- BTC ADX caps from Amendment #2 — the core finding

**Net config state after Amendments #1 + #2 + #3:**
- All Apr 14 baseline pair-ADX values (`adx_strong_long=15`, `adx_very_strong=30`, `momentum_adx_max=33`)
- Plus new BTC ADX caps: `btc_adx_max_long=25`, `btc_adx_max_short=30`
- Plus `btc_adx_dir_short: rising`
- Plus `adx_strong` short 22 (exploration)
- Plus pair blacklist addition + new-listing filter + Option B refactor

### Phase 1c amendment #4 (deployed Apr 17 PM, same day as #2/#3) — Soften LONG BTC ADX cap to preserve pre-committed PREMIUM ZONE L-P2

**Methodological miss that triggered this amendment:** Amendment #2 used the raw `Performance by BTC ADX` column to set BTC ADX magnitude caps. This aggregates across all BTC RSI bands, mixing winning cells with losing cells inside the same magnitude bucket. The CLAUDE.md "Filter design principle" section (Apr 14) explicitly states the cross-tab is the correct tool. I didn't apply it.

**What the raw-column aggregation hid:**

Inside the LONG BTC ADX 25-30 bucket that Amendment #2 blocked (at `btc_adx_max_long: 25`):
- BTC RSI 50-55 × BTC ADX 25-30: L-B1 HARD BLOCK (6 trades, 17% WR, loser in all 4 samples) — correctly blocked
- BTC RSI 60-65 × BTC ADX 25-30: part of the 73% WR L-P1-adjacent winner zone — incorrectly blocked

At LONG BTC ADX 30-35 (also blocked by Amendment #2):
- **BTC RSI 60-65 × BTC ADX 30-35: L-P2 PREMIUM ZONE (4 trades, 100% WR across 3 samples) — pre-committed Apr 15, incorrectly blocked**

Phase 1b data also showed LONG BTC ADX 25-30 at +0.27% on N=17 (winning in the most recent config), contradicting the 3-sample aggregate of −0.17%. The older Apr 13 + Apr 6 samples (different config, different regime) drove the negative aggregate.

**Change:** `btc_adx_max_long`: 25 → **35** (restored to Amendment #1 value).

**Why 35, not back to 40:** Phase 1b showed LONG BTC ADX 35-40 at N=5 @ −1.11%; no PREMIUM ZONE cells above 35; 35 preserves L-P2 (30-35) without re-admitting the clearly-bad 35-40 zone.

**SHORT side kept at `btc_adx_max_short: 30`** — cross-tab confirms no PREMIUM ZONE cells above 30 for SHORTs (S-P2 sits at 25-30, inside the kept range; S-B2 HARD BLOCK sits at 30-35).

**New methodological rule added to the filter design principle:**

> **When a pre-committed PREMIUM ZONE or HARD BLOCK exists in the cross-tab, raw-dimension caps MUST NOT block a PREMIUM ZONE or leave a HARD BLOCK unblocked. Check the cross-tab first. Raw-magnitude caps are blunt instruments — only defensible when no cross-tab evidence exists for cells inside the capped range, OR the cross-tab evidence across cells inside the range is uniformly negative.**

This rule is now part of the Apr 14 "Filter design principle" covenant and applies to all future BTC-level filter decisions.

**Net config state after Amendments #1 + #2 + #3 + #4:**
- All Apr 14 baseline pair-ADX values (`adx_strong_long=15`, `adx_very_strong=30`, `momentum_adx_max=33`)
- `btc_adx_max_long`: **35** (restored from 25 — L-P2 preservation)
- `btc_adx_max_short`: **30** (kept — cross-tab consistent)
- `btc_adx_dir_short: rising`
- `adx_strong` short 22 (exploration)
- Pair blacklist addition, new-listing filter, Option B refactor

**What this costs in expected P&L (vs Amendment #2's projection):**

The Amendment #2 ex-post projection assumed −0.05% → +0.16% with `btc_adx_max_long: 25`. Relaxing to 35 re-admits LONG BTC ADX 25-35 trades (N=75 in the 3-sample data: 70 at 25-30 + 5 at 30-35). Their 3-sample Avg P&L % was −0.13%. So the headline projection softens from +0.16% to roughly +0.05% combined (directionally still positive, materially weaker).

However: the projection was itself built on mixed-config data (Apr 6/13 high leverage, tick momentum ON, different exits). Under current config (Apr 14+ baseline exits), Phase 1b LONG 25-30 was +0.27%. So the "cost" of re-admitting 25-35 may be ≤0, not −0.13%. Phase 1c data will tell.

**Trade-off accepted:** smaller headline P&L improvement from BTC ADX caps alone, but preserves cross-tab integrity and the pre-committed Phase 2 PREMIUM ZONE rule L-P2. The proper way to recover the extra P&L is Phase 2 cross-filter code work (blocks L-B1 specifically while keeping L-P2 open), not via blunter raw-magnitude caps.

**What to measure at 100-trade Phase 1c checkpoint (updated):**

In addition to everything already documented:
- **BTC RSI × BTC ADX cross-tab** — compute Avg P&L % per cell on fresh Phase 1c data. Validate L-P2 (60-65 × 30-35) still wins ≥65% WR on ≥5 trades → keep open. Validate L-B1 (50-55 × 25-30) still loses ≤40% WR on ≥5 trades → build the Phase 2 cross-filter and block this cell specifically.
- **LONG BTC ADX 25-30 Avg P&L %** — if ≥+0.15% on ≥20 trades, the macro cap was correctly relaxed. If ≤−0.15%, re-consider a softer mechanism (e.g., allow 25-30 only for BTC RSI 60-65).
- **LONG BTC ADX 30-35 N and performance** — L-P2 is thin (N=4 pre-committed). Want ≥5 Phase 1c trades in this specific cell to validate.

## April 18, 2026 — Phase 1c amendment #5 (33-trade fresh data) — SHORT overhaul

Fresh 33-trade Phase 1c sample (12L + 21S, runtime 0.98 days) archived at
`reports/report_2026-04-18_phase1c_33trades.txt`. SHORT side PF 0.30, WR 23.8%,
Avg −0.29% — the core loss driver. Analysis showed:

### Critical falsification: Amendment #3's revert of `momentum_adx_max` was wrong

Yesterday I reverted `momentum_adx_max: 28 → 33` under the hypothesis that
Phase 1b's "pair ADX 28+ = loser" pattern was BTC-ADX-confounded. **Falsified.**
Fresh sample: pair ADX ≥28 SHORTs = 7 trades, 0% WR, avg −0.62%. 2-sample
confirmed now (Phase 1b + Apr 18). The pair-level pattern is real and
independent of the BTC ADX filter.

### Critical regime shift: BTC ADX pattern inverted vs 3-sample pool

| BTC ADX | 3-sample pool | Apr 18 sample |
|---|---|---|
| 20-25 | +0.20% (N=23) | **−0.50% (N=14, 7.1% WR)** |
| 25-30 | +0.05% (N=48) | **+0.13% (N=7, 57.1% WR)** |
| 30-35 | −0.53% (N=18) | no data (was blocked) |

The winning zone 25-30 is stable, but what surrounds it inverted. The Apr 17
Amendment #2 decision to cut 30-35 was based on the 3-sample pool; the new
sample shows 20-25 is now the bigger loser and we don't know about 30-35
since it's been blocked.

### Changes deployed Apr 18

| # | Change | Confidence | Evidence |
|---|---|---|---|
| 1 | `momentum_adx_max` (short): 33 → **28** | HIGH (2-sample) | Pair ADX ≥28 SHORT: 0% WR across Phase 1b + Apr 18 combined |
| 2 | `btc_adx_min_short`: 18 → **25** | HIGH in current regime | Apr 18: BTC ADX 20-25 = 7.1% WR on N=14. Cuts the new loser zone. |
| 3 | `adx_very_strong` (short): 30 → **28** | HIGH (follows from #1) | VERY_STRONG shorts 0/5 in this sample; #1 makes tier functionally inactive |
| 4 | `btc_adx_max_short`: 30 → **35** | EXPLORATION | User decision. Re-opens 30-35 to collect fresh-config data. Historical pool at 30-35 was -0.53% but under old configs. |

### New SHORT entry window (net of #1, #2, #4)
- Pair ADX: [22, 28] (was [22, 33])
- BTC ADX: [25, 35] (was [18, 30])
- BTC ADX direction: rising (unchanged)
- BTC RSI: [25, 60] (unchanged — BTC RSI 30-35 pattern inverted in this sample but insufficient evidence to act)

### What this sacrifices
- SHORTs at BTC ADX 20-25 (cut — Apr 18 showed this as the loser zone; 3-sample pool said winner)
- SHORTs with pair ADX 28-33 (cut — 2-sample loser)

**Risk:** if the Apr 18 BTC ADX inversion is 1-sample noise and the 3-sample pool is
actually right about 20-25, we just cut the winning zone. The trade-off: ship a
filter that worked in the most recent sample (regime-current) OR ship a filter that
worked across older samples (regime-stale). Given the loss velocity is −$11 in 24h,
ship the regime-current filter and accept the risk.

### ADX delta < 2.0 watchlist (NEW — do not deploy yet)

Apr 18 sample showed:
- SHORTs with ADXΔ < 2.0: 13 trades, 1 win, −0.50% avg
- SHORTs with ADXΔ ≥ 2.0: 8 trades, 4 wins, +0.06% avg

Clean breakpoint at 2.0. But 1-sample only — must replicate in next batch before becoming
a filter. If next batch shows SHORTs with ADXΔ < 2.0 at ≤30% WR on ≥10 trades, add as
`short_min_adx_delta: 2.0` filter.

### BTC RSI 30-35 × BTC ADX — conflicting signal (do NOT tighten yet)

Apr 18 sample:
- BTC RSI 30-35 × 20-25: 8 trades, 0% WR, −0.52%
- BTC RSI 30-35 × 25-30: 4 trades, 25% WR, −0.13%
- BTC RSI <30 × 25-30: 3 trades, 100% WR, +0.48% ← S-P1 at new ADX range
- BTC RSI <30 × 20-25: 4 trades, 25% WR, −0.33%

4-sample pool had 30-35 × 25-30 at 83% WR (S-P2 PREMIUM). This sample shows 25% WR.
The pre-commit rules are under stress. **Do NOT deploy `btc_rsi_max_short: 35`**
(which would be attractive on Apr 18 data alone) — wait for next batch. If 2-sample
confirms 30-35 as loser, tighten then.

### Phase 1c amendment #6 (deployed Apr 18) — Maker entry timeout experiment (20s → 40s)

Hypothesis-driven experiment, not a structural change. Framed as a
bounded test with explicit falsification criteria.

**Hypothesis:** extending the maker entry timeout from 20s to 40s captures
more pullback entries (higher maker fill rate) without degrading taker
fallback quality enough to offset the fee savings and the pullback-entry
structural advantage.

**What the maker entry logic does today:**
1. Signal fires, bot places LIMIT order at EMA5 ± 1 tick (offset configurable)
2. Waits up to `maker_timeout_seconds` for fill
3. If not filled: cancels limit, places MARKET order (TAKER_FALLBACK)

**Effect of the change:** price that retraces to our limit within 20-40s
now fills as MAKER instead of TAKER_FALLBACK. On fast-moving momentum
trades (price runs away), the maker still never fills and the TAKER_FALLBACK
fires at t=40s instead of t=20s — potentially worse slip.

**Cross-sample maker-vs-taker data (inconsistent):**

| Sample | MAKER LONG WR | TAKER LONG WR | MAKER SHORT WR | TAKER SHORT WR | Gap |
|---|---|---|---|---|---|
| Apr 13 (117 tr) | 70.6% | 59.4% | 65.6% | 52.6% | **+10-13% MAKER** |
| Apr 17 Phase 1b (81 tr) | — | — | 46.4% | 46.2% | **None** |
| Apr 18 (33 tr) | 57.1% | 60% | **8.3%** | **44.4%** | **MAKER lost by 36%** |

The MAKER edge is regime-dependent. Apr 18 (current regime) showed MAKER
SHORTS badly broken — possibly the fill mechanics themselves are the issue
(entering on pullbacks into an already-reversing move), possibly regime-specific
noise. Extending timeout is a clean test: if MAKER is structurally good, more
maker fills = better overall WR. If MAKER is structurally broken in current
regime, more maker fills = worse overall WR → revert.

**Why NOT the "winners take time to peak" logic:**

This is a documented conflation in the LONG flameout watchlist.
- HOLD duration (entry to peak) = 20-40min for winners vs <2min for losers
- ENTRY timing (signal to fill) = the 20-40s window

The two are orthogonal. A 40s maker delay does not prevent a trade from
flaming out at +0.04% in the next 2 minutes. The real argument for extending
the timeout is the pullback-entry structural advantage, not the hold duration.

**Falsification criteria at 30-trade checkpoint (post Apr 18 reset):**

| Metric | Threshold to keep 40s | Threshold to revert |
|---|---|---|
| Maker fill rate | ≥65% (up from ~55% baseline) | <60% (timeout wasn't the constraint) |
| MAKER vs TAKER WR gap | ≥+8% MAKER edge | ≤0% or MAKER worse |
| TAKER_FALLBACK entry slippage | similar to baseline | widens by >0.02% (fast-runner penalty) |

**Decision rule:**
- Fill rate ↑ AND MAKER WR edge ≥8% → keep 40s
- Fill rate ↑ but WR flat or reversed → revert, timeout wasn't the solution
- Fill rate unchanged → revert, timeout wasn't the constraint
- TAKER slippage widens materially → revert, cost of late fallback exceeds fee savings

**Expected fee impact:**

Fee savings per shifted entry: 0.045% (taker) − 0.018% (maker) = 0.027%.
At 20 trades/day and ~10% fill-rate uplift (pure estimate), ~2 additional
maker entries/day = ~0.054% savings/day. Material but small.

**Risk assessment:**

Low config risk (single integer change, easily revertable). Low code risk
(mechanism already exists, just a parameter tweak). The cost of being wrong
is taker slippage on ~2 additional trades/day × 1-2 days of observation =
bounded downside.

### Phase 1c amendment #7 (deployed Apr 18) — Signal re-validation before taker fallback

Infrastructure fix enabling Amendment #6's timeout extension to be safe.

**Problem discovered by user:** current maker-entry flow at timeout unconditionally places a taker market order without re-checking that the signal is still valid. With the 20s → 40s timeout change (Amendment #6), the window for signal-staleness doubled — BTC regime can flip, pair ADX direction can reverse, RSI momentum can shift, any of which invalidate the original signal. Bot was potentially entering on stale signals after long waits.

**Fix implemented (Option A from user decision tree):**

1. **New method `_revalidate_entry_signal(symbol, pair, direction, confidence)`** in `services/trading_engine.py`:
   - Re-fetches fresh 5m OHLCV for the pair and BTC
   - Re-computes indicators
   - Re-runs `get_signal(...)` — checks if (direction, confidence) still match
   - Re-checks BTC-level filters: `btc_adx_dir_*` (rising/falling), BTC ADX range, BTC RSI range
   - Returns `(is_valid, reason)` — FAILS OPEN on fetch/compute errors (defers to taker)

2. **`_try_maker_entry` and `_simulate_maker_entry_paper` signatures extended** to accept `confidence`. If provided, at timeout (before taker fallback) they call `_revalidate_entry_signal`. If re-validation fails:
   - Return `{'entry_order_type': 'SIGNAL_EXPIRED', 'skipped': True, 'reason': ...}`
   - Caller `open_position` detects the skipped flag and:
     - Calls `_record_signal_expired_order(...)` to persist a minimal Order row with `status='SIGNAL_EXPIRED'`, `entry_order_type='SIGNAL_EXPIRED'`, zero PnL/investment/quantity
     - Logs `[SIGNAL_EXPIRED] pair direction confidence: reason — taker fallback aborted`
     - Returns None (no position opened)

3. **Reporting changes in `main.py`**:
   - `_compute_performance` loads BOTH `status='CLOSED'` and `status='SIGNAL_EXPIRED'` orders
   - `_compute_entry_type_stats(orders, signal_expired_orders=...)` now accepts the SIGNAL_EXPIRED list and adds a synthetic row to Entry Type Performance
   - SIGNAL_EXPIRED rows have WR=0, Avg P&L=0 by construction (no trade happened) — the value is the **count** of aborted entries, broken down by confidence level
   - All other aggregations (by-regime, by-RSI, by-ADX, etc.) only see `CLOSED` orders — SIGNAL_EXPIRED rows are excluded to avoid polluting PnL/WR analytics

4. **What the operator sees:**
   - Daily report's Entry Type Performance table now shows a SIGNAL_EXPIRED row with count + confidence breakdown
   - Logs: `[SIGNAL_EXPIRED] DOGEUSDT SHORT STRONG_BUY: btc_adx_direction_not_rising — taker fallback aborted`
   - No PnL contamination; no phantom trades in WR calculations

**Re-validation reason codes logged:**
- `signal_flipped_<from>_to_<to>` — core get_signal changed direction (e.g. SHORT→LONG, SHORT→NO_TRADE)
- `confidence_lost` — signal still has direction but confidence is None/NO_TRADE
- `btc_adx_direction_not_rising` / `not_falling` — BTC ADX direction requirement no longer met
- `btc_adx_out_of_range_<value>` — BTC ADX moved outside the configured [min, max] window
- `btc_rsi_out_of_range_<value>` — BTC RSI moved outside the configured [min, max] window
- `fetch_failed_defer` / `indicators_failed_defer` / `error_defer` — infrastructure failure; FAILS OPEN (taker fallback still proceeds) to not block trading on transient issues

**Fail-open design:** any exception or fetch failure in re-validation defers to the original taker fallback behavior. Rationale: better to occasionally enter on a stale signal than to systematically block entries when Binance rate-limits or the API hiccups. The cost of a false positive (aborting valid trade) is greater than the cost of a false negative (entering on stale signal) at current trade volume.

**Files changed:**
- `services/trading_engine.py` — `_revalidate_entry_signal`, `_record_signal_expired`, `_record_signal_expired_order`; `_try_maker_entry` and `_simulate_maker_entry_paper` signature + body updates; two caller sites in `open_position` (live + paper) updated to pass `confidence` and handle skipped result
- `main.py` — `_compute_performance` queries SIGNAL_EXPIRED status; `_compute_entry_type_stats` accepts optional `signal_expired_orders` and adds synthetic row
- `models.py` — **no schema change**. Uses existing `status` (String 15) and `entry_order_type` (String 15); "SIGNAL_EXPIRED" is 14 chars, fits.

**Measurement at next checkpoint:**
- SIGNAL_EXPIRED count per direction per day (tells us the rate of stale-signal rejections)
- Reason breakdown (log-based, `grep SIGNAL_EXPIRED /var/log/web.stdout.log | sort | uniq -c`)
- MAKER vs TAKER_FALLBACK WR gap (the original Amendment #6 falsification criterion — but now TAKER_FALLBACK trades are ONLY those with re-validated signals, cleaner signal)

If SIGNAL_EXPIRED count is >20% of entry attempts: the 40s timeout is systematically causing signal staleness → revert Amendment #6 to 20s. If <5%: signal staleness isn't the dominant issue and Amendment #6 can stay. If 5-20%: middle ground, operator decides.

### Phase 1c amendment #8 (deployed Apr 18) — Maker offset 1 → 2 ticks

Builds on Amendment #6 (40s timeout) and Amendment #7 (signal re-validation). Now that stale-signal risk on taker fallbacks is mitigated, safe to deepen the maker offset.

**Change:** `maker_offset_ticks: 1 → 2`

Places the maker limit order 2 ticks deeper into the book (LONG bids 2 ticks below best_bid, SHORTs ask 2 ticks above best_ask). Acts as an implicit pullback-entry filter: trades that can't get 2 ticks of retracement within 40s are over-extended momentum → go to re-validated taker fallback or SIGNAL_EXPIRED.

**Pair-variance note:** 2 ticks has very different meaning by pair:
- BTCUSDT/ETHUSDT (large caps): 0.0003-0.0007% deeper = near-zero functional impact
- SOLUSDT-tier (mid caps): ~0.013% deeper
- DOGEUSDT/small caps: 0.05-0.15% deeper = meaningful pullback requirement

The change primarily tightens entries on mid/small caps where momentum fakes are more frequent.

**Expected effects (to measure at 30-trade checkpoint):**
- MAKER fill rate: likely drops (offset harder to reach)
- MAKER entry price quality: improves (filled closer to EMA5 pullback bottom)
- TAKER_FALLBACK count: rises (but now re-validated per Amendment #7)
- SIGNAL_EXPIRED count: may rise (more trades reach timeout + re-validation stage)

**Decision rule at checkpoint:**
- MAKER WR improves ≥8% vs TAKER_FALLBACK → keep offset 2 (quality filter working)
- MAKER fill rate drops >30% AND WR unchanged → revert to offset 1 (over-restrictive)
- SIGNAL_EXPIRED rate >25% → Amendment #6 timeout is too long (revert timeout to 20s first, keep offset 2)

Config-only change, single integer, instant revert if needed.

## April 28, 2026 — Phase 1c-Explore (sub-phase) — Loosen restrictions for ablation testing

### Strategic shift: validation → exploration

Switching from live to **paper trading** with the **Exploration Analytics** indicators just added. Previous Phase 1c amendments (#1-#8) were progressively tightening filters under live mode to validate edge. Now the goal flips: collect data in **previously-blocked zones** to test whether the recent restrictions captured the **real driver** or were **proxy filters** for something else (e.g., EMA50 alignment, DI spread, funding rate).

**The methodological argument:** an active filter blocks data in its target zone, so we can't ablation-test it. To answer "was `btc_adx_min_short: 25` a real driver or a proxy for EMA50 alignment?", we need data on BTC ADX 18-25 SHORTs trades **with the new dimensions captured**. Paper mode + fresh batch is the ideal experimental environment — exploration is free.

### Config changes (deployed Apr 28)

**SHORT side (the loss leader — needs the most exploration):**

| # | Change | Restored to | Hypothesis to test |
|---|---|---|---|
| 1 | `btc_adx_min_short`: 25 → **18** | Apr 16 baseline | BTC ADX 18-25 SHORTs may only fail when EMA50 slope is rising (not because of BTC ADX). |
| 2 | `momentum_adx_max` (short): 28 → **33** | Apr 14 baseline | Pair ADX 28-33 SHORT failures may concentrate at compressed DI spread (not high ADX per se). |
| 3 | `adx_very_strong` (short): 28 → **30** | Apr 14 baseline | Gates with #2; restores VERY_STRONG SHORT tier so we can analyze it under new dimensions. |
| 4 | `btc_adx_max_short`: 35 → **40** | Pre-Apr-14 | Re-admits BTC ADX 35-40 zone (zero current-config data). Tests S-B2 HARD BLOCK validity at high ADX. |

**LONG side:**

| # | Change | Restored to | Hypothesis to test |
|---|---|---|---|
| 5 | `btc_adx_max_long`: 35 → **40** | Pre-Apr-14 | Tests L-P2 PREMIUM ZONE (60-65 × 30-35) directly + admits BTC ADX 35-40 zone for fresh data. |

**Paper-mode setup:**

| Setting | Value | Purpose |
|---|---|---|
| `paper_balance` | $1,000 | Realistic small account size matching potential live deploy. |
| `paper_bnb_initial_usd` | $50 | Realistic BNB fee runway. |

### What is NOT loosened (kept tight)

| Filter | Rationale to keep |
|---|---|
| `btc_adx_dir_short: rising` | **3-sample structural finding** (Apr 6 + Apr 13 + Apr 17). Strongest evidence in dataset; orthogonal axis to magnitude, not redundant with the loosened caps. |
| Pair blacklist (RAVEUSDT) | Surgical pair-quality kill, validated. |
| `new_listing_filter_days: 180` | Independent pair-quality filter. |
| All EMA gap thresholds, RSI ranges, breadth filter | Untouched by recent amendments — not over-tightened. |

### Effective new entry windows

| Side | Pair ADX | BTC ADX | BTC ADX dir | BTC RSI |
|---|---|---|---|---|
| LONG | [15, 25] | [18, 40] | both | [40, 65] |
| SHORT | [22, 33] | [18, 40] | rising | [25, 60] |

### Sample size target — 150 trades (200 ideal), with multi-checkpoint structure

The originally-documented 100-trade target was set when we had only single-dimension tables. **Cross-tabs split N across multiple cells, so the cell-level requirement (N≥10 per critical cell) drives a higher total-sample target.**

**Bottleneck: BTC ADX × EMA50 Alignment SHORT cross-tab.** Critical cells that must populate to N≥10:
- BTC ADX 18-25 × Aligned (re-admitted zone, primary ablation test)
- BTC ADX 18-25 × Opposite (re-admitted zone, primary ablation test)
- BTC ADX 25-30 × Aligned (baseline comparison)
- BTC ADX 30-35 × Opposite (re-admitted zone)
- BTC ADX 35-40 × any (re-admitted zone, S-B2 test)

If these 5 cells together account for ~60-70% of total SHORTs (rest scatter to less-critical cells), need **~70 SHORTs** to populate them robustly. Same logic on LONG side. **~140 trades minimum, 150 target, 200 ideal.**

**Three-checkpoint sequence:**

| Checkpoint | Purpose | What's allowed at this point |
|---|---|---|
| **~50 trades — health check** | Verify all new fields populate (not NULL), tables render, no bot errors, zones starting to fill | Zero config decisions. Only fix data-pipeline bugs. |
| **~100 trades — qualitative read** | First analytical pass; identify directional patterns; check zone population per Priority 1 below | Note hypotheses, do NOT promote to filter changes yet. |
| **~150-200 trades — decision checkpoint** | Full ablation analysis; filter swaps decided; new filters from Tier 1 dimensions promoted | Ship config changes per locked decision rules below. |

**Hard floor:** do NOT stop the batch before 100 trades. Below that, the cross-tabs won't deliver what they're designed for and we're back to single-dim analysis we already have.

**Early-decision exception:** if a single cross-tab cell shows extreme signal (e.g., 10+ trades at 100% loss in a clearly-defined zone) at 100 trades, we can decide early on that specific filter while letting the rest of the analysis wait for 150-200.

### What to measure at the Phase 1c-Explore decision checkpoint

**Priority 1 — zone population.** Did we get ≥10 SHORT trades in EACH of: BTC ADX 18-25, pair ADX 28-33, BTC ADX 35-40? If any zone is empty, other filters are constraining it and we need to re-evaluate. ≥10 LONG trades at BTC ADX 35-40 also required for L-P2 test.

**Priority 2 — ablation tests** (the core point of this sub-phase):

For each loosened filter, build a 2D cross-tab against the most relevant new dimension:

| Loosened filter | Cross-tab against | Decision rule |
|---|---|---|
| `btc_adx_min_short: 18` (re-admitted 18-25 zone) | EMA50 alignment | If 18-25 SHORTs only fail at EMA50 rising → ship `btc_adx_min_short: 18 + new filter "block SHORT if entry_ema50_slope > +0.04%"`. Loosen by replacement, not removal. |
| `momentum_adx_max: 33` (re-admitted 28-33 zone) | DI spread | If 28-33 SHORTs only fail at DI spread <2 → ship `momentum_adx_max: 33 + new filter "block SHORT if DI spread <2"`. |
| `btc_adx_max_short: 40` (re-admitted 35-40 zone) | EMA50 alignment + DI spread | Likely will be a loser zone regardless — confirms the prior cap. If so, restore `btc_adx_max_short: 35`. |
| `btc_adx_max_long: 40` (re-admitted 35-40 zone) | EMA50 alignment + DI direction | Tests L-P2; if L-P2 cell still wins on N≥4, preserve. If it loses, restore `btc_adx_max_long: 35`. |

**Priority 3 — Net Avg P&L %.** Sample-level Avg P&L % vs the prior 33-trade Apr 18 sample. Should be similar (±0.20%) or better. Materially worse → at least one of the loosened filters was a real driver, revert it specifically.

**Priority 4 — Volume.** Expect ~+50-80% more entry attempts vs prior config. If volume drops or stays flat → other (untouched) filters are the binding constraint.

**Priority 5 — SIGNAL_EXPIRED rate.** Amendment #7 will fire more often with looser entry windows. Watch the rate. If >25% of attempted entries expire, the macro filters are flickering on/off too fast — revert one of the looser ones.

### Promotion rules at the Explore checkpoint

- **Filter swap (recommended outcome):** loosened filter X stays loose AND new filter Y from Tier 1 dimensions is added. Net trade volume goes up, edge goes up.
- **Revert (if ablation shows the original was right):** the loosened filter is restored to its prior tight value. New filter Y is NOT added (no evidence it discriminates).
- **Keep loose, no Y filter (rare):** the re-admitted zone performs at parity. Zone is just neutral, no filter needed there.

### Pooling rule for Phase 1c-Explore

This sub-phase's data is **separate** from prior Phase 1c amendments #1-#8 data. Different config, different mode (paper vs live), different goal (explore vs validate). Do NOT pool raw trades across the boundary. Compare bucket-level WR / Avg P&L %, treating Phase 1c-Explore as its own sample.

When eventually returning to live, **retest the winning config under live conditions** before treating any Phase 1c-Explore finding as live-deployable. Per the Apr 18 quant methodology note: paper data is comparable to live for filter/regime/bucket findings, NOT for fill-mechanics findings (Amendments #6/#8). The exploration here is filter-layer, so paper-OK.

## April 28, 2026 — Exploration Analytics (Tier 1 indicators added, observation-only)

### Provisional status (added Apr 30)

**The entire Exploration Analytics section may turn out to make no sense and is on probation.** The Tier 1 dimensions (EMA50 alignment, DI spread, ATR%, funding rate, TtP ratio) and the cross-tabs (BTC ADX × EMA50 Align, Pair ADX × DI Spread, BTC RSI × Funding, Direction × EMA50 Align) were added as exploratory hypotheses, not validated edge sources. At the 200-trade Phase 1c-Explore decision checkpoint:

- If NO Tier 1 dimension meets the 6-criterion promotion bar (N≥20 per bucket, ≥15pp WR gap, ≥0.20pp Avg P&L gap, direction-consistent, TtP≤0.45 sanity check, cross-tab confirmation), AND none of the cross-tabs show meaningful discrimination → **remove the entire Exploration Analytics UI section, the 6 single-dim tables, the 4 cross-tabs, and the TtP table.** Keep the underlying `entry_*` DB columns (cheap, no harm in retaining captured data) but drop the rendering and the report-export entries.
- If ONE OR TWO dimensions show signal → keep only those tables/cross-tabs, remove the rest.
- If most/all dimensions show signal → keep the section as-is and proceed to filter promotion per the locked Phase 1c-Explore plan.

**Do NOT remove anything before the 200-trade checkpoint** — the value of these analytics is precisely in being able to falsify them with real data, not in pre-emptive removal. The removal is conditional on the analysis confirming no signal.

Files that would be touched on removal: `templates/index.html` (UI section + 6 table renderers + 4 cross-tab renderers + 2 text-export sites), `main.py` (`_compute_performance` payload sections, bucket-perf functions). Underlying capture (`models.py`, `services/trading_engine.py`, `services/binance_service.py::fetch_funding_rate`) stays — capture is cheap, removing only the analysis surface is the right granularity.

### What changed

5 new `entry_*` columns on the Order model, captured at signal time. **Zero filter logic changes** — pure observation. Purpose: at the next 100-trade checkpoint, bucket-analyze the new dimensions and identify which discriminate winners from losers, before promoting any to filters.

| Field | Source | What it captures |
|---|---|---|
| `entry_pos_di` | TA `ADXIndicator.adx_pos()` | +DI: directional component of ADX measuring upward pressure |
| `entry_neg_di` | TA `ADXIndicator.adx_neg()` | -DI: directional component measuring downward pressure |
| `entry_atr_pct` | TA `AverageTrueRange(14)` / price × 100 | ATR as % of price; volatility regime per pair |
| `entry_ema50_slope` | (ema50 − ema50_prev12) / ema50_prev12 × 100 | 5m EMA50 slope ≈ 4 hours of higher-timeframe context |
| `entry_funding_rate` | Binance `fetch_funding_rate(symbol)` cached 8h | Positioning context (negative = market heavily short) |

### Why each was selected

**+DI / −DI:** ADX magnitude alone tells you trend STRENGTH but not DIRECTION CONVICTION. ADX can rise in a choppy market when both +DI and −DI are jumping. Trades where the gap between +DI/−DI is compressing (low conviction) probably underperform same-ADX-magnitude trades with wide DI spread. Currently a known blind spot in our ADX usage.

**ATR(14):** Our SL is fixed at −0.9% across all pairs. BTC/ETH (calm pairs) at −0.9% means hours of price action (SL too wide, losers ride too long). Small caps at −0.9% means single-candle noise (SL too tight, stops on noise). Even without changing SL logic yet, capturing ATR enables volatility-bucket analysis to confirm whether this hypothesis holds.

**5m EMA50 slope:** Cheap proxy for higher-timeframe context. EMA50 on the 5m chart spans 50 candles ≈ 4 hours of trend — much longer than what we filter on currently (5m EMA Gap Expanding looks at 1 candle, RSI Momentum at 3 candles ≈ 15 minutes). The LONG flameout pattern (winners take 20-40min to peak, losers peak <2min and fail) and SHORT regime-change losses both look like "5m signal valid but fighting higher timeframe." If EMA50 slope alignment discriminates, the 4-hour TF context matters and the 5m momentum filters are missing it. Chosen instead of 15m candle fetch because the data is **already in our pipeline** — calculate_indicators already computes EMA50 and ema50_prev12; we just weren't storing the slope on Order.

**Funding rate:** Free positioning data. SHORTS into very negative funding (e.g., < −0.05%) = market heavily short already, paying carry to hold, and over-positioning often resolves in violent squeezes against shorts. Particularly relevant given our SHORT side has been the loss leader. LONGS into very high positive funding (> +0.05%) = late to a crowded trade.

### Why these specifically (vs other candidates)

Held for later (after Tier 1 shows value):
- **MACD histogram + divergence** — momentum confirmation, but partly redundant with EMA Gap Expanding + RSI Momentum
- **Open Interest** — would add positioning depth but requires per-pair API call (already adding 1 call/pair for funding)
- **VWAP distance** — institutional reference, useful but lower discrimination expected vs Tier 1

Rejected:
- **Bollinger %B / BB width** — overlaps with EMA stretch + ATR
- **Stochastic** — overlaps with RSI
- **Candle patterns** — statistically weak signal in crypto futures
- **15m candle fetch** — escalation only if 5m EMA50 slope analysis shows higher-TF context matters

### Dashboards added

UI section: **"Exploration Analytics — Observation Only"** below Pair Performance. Six tables, each with bucket × direction × N × WR × Avg P&L %:

1. **Performance by ATR** — buckets <0.3%, 0.3-0.6%, 0.6-1.0%, 1.0-1.5%, >1.5%
2. **Performance by EMA50 Slope (raw)** — buckets <-0.10%, -0.10 to -0.04, -0.04 to +0.04 (flat), +0.04 to +0.10, >+0.10
3. **Performance by EMA50 Alignment** — Aligned / Opposite / Flat (relative to trade direction)
4. **Performance by Funding Rate** — buckets <-0.05%, -0.05 to -0.02, -0.02 to +0.02 (neutral), +0.02 to +0.05, >+0.05
5. **Performance by DI Direction** — +DI > −DI / −DI > +DI / Tight (gap < 2)
6. **Performance by DI Spread** — <2 / 2-5 / 5-10 / >10

All tables also appear in text-exported reports (both export functions — clipboard copy + auto-saved file).

### Cross-tabs added Apr 28 (for ablation testing at next checkpoint)

Four 2D bucket × bucket cross-tabs added under Exploration Analytics ("Cross-tabs for ablation testing" sub-section). Each directly answers one ablation hypothesis from Phase 1c-Explore:

| Cross-tab | Hypothesis it tests | Decision rule |
|---|---|---|
| **BTC ADX × EMA50 Alignment** | Were BTC ADX 18-25 SHORTs only failing because of misaligned EMA50, not BTC ADX itself? | If 18-25 + Aligned cell ≥55% WR on N≥10 AND 18-25 + Opposite ≤30% WR → swap: keep `btc_adx_min_short: 18` + add EMA50-alignment filter. |
| **Pair ADX × DI Spread** | Are pair ADX 28-33 SHORT failures concentrated at compressed DI spread (<2)? | If 28-33 + spread≥5 cell ≥50% WR on N≥10 AND 28-33 + spread<2 ≤30% WR → swap: keep `momentum_adx_max: 33` + add DI-spread filter. |
| **BTC RSI × Funding Rate** | Are BTC RSI extremes confounded by positioning rather than RSI value? | If a particular RSI band is winning at neutral funding and losing at extreme funding → swap RSI cap for funding cap. |
| **Direction × EMA50 Alignment** | Does EMA50 alignment matter equally for LONGs and SHORTs? | If gap between Aligned and Opposite is ≥15% WR on N≥10 each → ship asymmetric filter (e.g., LONG-only or SHORT-only EMA50 alignment requirement). |

All cross-tabs follow same shape as existing BTC RSI × BTC ADX table: row × column with N / WR / Avg$ / Avg% / Total$ / Conf per cell. Cells with no data are dropped (no zero-fill rows).

Cell N≥10 required before drawing any conclusion. Below that, cell content is a hypothesis at best.

### Trade Quality Metric — Time-to-Peak Ratio (added Apr 28)

Added in response to user observation that trades surviving 2+ hours through regime chop and ending barely-positive look like edge but are actually survival luck. Pure WR doesn't distinguish "fast directional winners" from "slow-grind survivors."

**Definition:** `(peak_reached_at − opened_at) / (closed_at − opened_at)` — where in the hold did the trade reach its peak P&L? Range 0.0 to 1.0.

| Ratio | Interpretation |
|---|---|
| 0.0 - 0.2 | Peaked in first 20% — fast directional, real edge |
| 0.2 - 0.4 | Peaked early-mid — directional with some lag |
| 0.4 - 0.6 | Peaked mid-hold — mediocre |
| 0.6 - 0.8 | Peaked late grind — survival, not skill |
| 0.8 - 1.0 | Peaked at close — either pure skill or got lucky on exit timing |

Computed on-the-fly from existing `peak_reached_at`, `opened_at`, `closed_at` fields. Returns None for trades that never went positive, are still open, or have zero duration. **No schema change.**

**What this enables at the next checkpoint:**

1. **Quality-adjust the WR signal.** When two setups have similar WR, the one with lower mean TtP (faster peaks) has the real edge. The other got lucky in current regime and won't generalize.

2. **Cross-validate EMA50 Alignment.** If Aligned trades cluster at TtP < 0.4 → Alignment captures genuine directional edge → ship as entry filter. If Aligned trades cluster at TtP > 0.6 → Alignment is just regime correlation, not edge → don't promote.

3. **Justify a time-based exit.** If TtP > 0.5 AND peak < +0.20% systematically results in negative or barely-positive close, evidence supports an exit rule: "close trade at 30 min if no peak above +0.20%."

**Where it appears:**
- New table: **Performance by Time-to-Peak Ratio** under Exploration Analytics — 5 buckets × direction
- New column **TtPRatio** on Entry Conditions by Close Reason (avg per group)
- Both text-export sites

**What this does NOT do:** TtP is a quality lens, not a direct entry filter. It's computed at exit, so you can't filter on it at entry time. Its value is in interpreting WR honestly and validating which other dimensions are real edge vs regime correlation.

**Promotion rule for using TtP at checkpoint:**
- A setup is "real edge" only if it has WR ≥ baseline AND mean TtP ≤ 0.45 AND mean peak ≥ +0.30%
- Setups failing the TtP test are flagged "survival wins, hold for cross-sample replication" — don't promote to filter from a single batch

### What to deeply analyze at next 100-trade checkpoint

Treat this section as the FIRST thing to look at when the next batch lands.

**Priority 1 — does EMA50 alignment matter?** This is the highest-expected-value test. If "Aligned" trades have ≥ +15% WR over "Opposite" trades on N ≥ 20 each, the LONG flameout pattern hypothesis is confirmed and we ship a filter: only enter when 5m EMA50 slope agrees with trade direction. Promotes to filter at next-batch deploy.

**Priority 2 — does ATR explain pair-level variance?** Build "Pair × ATR bucket" cross-tab if N permits. If high-ATR pairs systematically lose (because −0.9% is too tight on noisy pairs) and low-ATR pairs systematically win, that's a strong case for volatility-aware stops (Phase 3 work) and possibly a coarser filter "skip pairs with ATR > 1.5%" in the interim.

**Priority 3 — DI spread filter potential?** If "Tight (gap < 2)" trades are ≥ −0.2% Avg below "+DI > -DI" or "-DI > +DI" trades on N ≥ 15, the ADX-magnitude-only filter has a real gap. Filter candidate: require |+DI − -DI| ≥ 2 at entry, in addition to ADX-rising.

**Priority 4 — funding rate filter for shorts?** If SHORTS at funding < −0.05% have meaningfully lower WR than SHORTS at neutral funding (≥ 5% gap on N ≥ 10), confirms positioning-fights-shorts hypothesis. Filter candidate: skip SHORT entry when funding < −0.05%.

**Priority 5 — does +DI/−DI direction agree with trade?** Especially for LONG entries: if pair signal is LONG but +DI < −DI at entry (downward pressure dominant), that's contradiction. Cross-tab "Trade direction × DI dominance" would expose it.

### Promotion rules (anti-overfit discipline)

- **N ≥ 20 per bucket** for any filter promotion. Below that, single-sample noise.
- **Cross-direction sanity check.** If a dimension matters for LONGS but not SHORTS (or vice versa), that's a real asymmetry to respect — don't promote symmetrically.
- **2-sample replication required** before locking a filter. First batch = hypothesis, second batch confirms.
- **Order of escalation:** ATR cap (coarse pair filter) → EMA50 alignment (entry filter) → DI spread (entry filter) → funding rate cap → ATR-normalized stops (Phase 3 SL change).

### What this does NOT do

- ❌ No filter logic changes — only observation
- ❌ No SL changes — current −0.9% kept; ATR data informs future SL design
- ❌ No 15m candle fetch — escalate only if EMA50 slope shows real signal
- ❌ No cross-tabs in this build — single-dimension tables first; cross-tabs come at the next-next checkpoint when N is high enough per cell

### Files changed

- `models.py` — 5 nullable columns on Order
- `database.py` — auto-migrate ADD COLUMN for existing DBs
- `services/indicators.py` — expose `pos_di`, `neg_di`, `atr` in indicators dict (TA library already computed them)
- `services/binance_service.py` — `fetch_funding_rate(symbol)` with 8h in-memory cache
- `services/trading_engine.py` — capture path: compute slope from existing data, fetch funding, pass through to `open_position`, persist on Order
- `main.py` — 6 new bucket-perf functions + payload integration in `_compute_performance`
- `templates/index.html` — new "Exploration Analytics" UI section + 6 table renderers + text-export entries in both export sites

## April 28, 2026 — LOCKED Phase 1c-Explore Plan (200-trade frozen exploration batch)

### Why this is locked

Past 6 weeks of iteration produced break-even-ish results (Avg P&L % oscillating
in ~[-0.30%, +0.05%] band) across multiple amendment cycles. The honest read on
that data is **confounded** — every sample was collected during active config
change, so we have no clean test of "what does ANY frozen config produce."
This plan fixes that. Exploration batch (this 200) → key-variable identification →
validation batch (next 200 with adjustments). Decoupling exploration from
validation is the discipline we've been missing.

**Goal of THIS batch:** identify the key variables to adjust for the next batch.
Not profitability. Profitability is the goal of the NEXT batch.

### Locked config (frozen as of Apr 28)

The Phase 1c-Explore config deployed Apr 28 (paper mode, looser filters than Apr 18).
Specifically: `btc_adx_min_short: 18`, `momentum_adx_max_short: 33`, `adx_very_strong_short: 30`, `btc_adx_max_short: 40`, `btc_adx_max_long: 40`, plus the Tier 1 exploration analytics columns.

**No amendments during the 200 trades. Period.** Even if the first 50 look catastrophic,
even if a "clear pattern" emerges at 100 trades, even if user requests "small tweaks" —
the answer is no. The whole analytical value of this batch depends on it being
collected under one frozen config. Mid-batch amendments invalidate the entire sample.

**The only exceptions** are operational (not strategic):
- Bug fixes (data not persisting, capture pipeline broken, bot crash)
- Infrastructure issues (DB durability, race conditions, etc.)
- Pair blacklist additions for EMERGENT operational reasons (e.g., Binance delisting)

Strategic config changes (filter values, thresholds, ADX caps, RSI ranges, leverage,
exit parameters, maker timeout, offset ticks): **none, until 200 trades collected.**

### Pre-committed analytical dimensions (locked BEFORE the data arrives)

At 200 trades, analyze these dimensions and ONLY these dimensions. Looking at
"everything" is multiple-comparisons hacking. Pre-commit list:

**Tier 1 — new exploration dimensions (highest priority — never tested before):**
1. **EMA50 alignment** (Aligned / Opposite / Flat vs trade direction) — best-EV hypothesis
2. **DI spread** (|+DI − -DI|) — conviction-strength filter candidate
3. **ATR %** (volatility regime per pair)
4. **Funding rate** (positioning context, especially for shorts)
5. **TtP ratio** (Time-to-Peak — distinguishes real edge from survival luck)

**Tier 2 — cross-tabs for ablation testing (locked Apr 28):**
6. **BTC ADX × EMA50 Alignment** (does Apr 18's BTC ADX cap need to be a real macro filter, or was it proxying for EMA50 misalignment?)
7. **Pair ADX × DI Spread** (was momentum_adx_max really about high ADX, or about compressed DI spread within high ADX?)
8. **BTC RSI × Funding Rate** (does positioning explain the BTC RSI patterns better than RSI value itself?)

**Tier 3 — pre-existing dimensions for cross-validation only:**
9. BTC RSI × BTC ADX cross-tab (validates 8 pre-committed Phase 2 rules at 5th sample)
10. Pair ADX bucket × direction (does Apr 18's pair ADX 28+ SHORT loser pattern hold?)

**Dimensions explicitly NOT to look at (avoid p-hacking):**
- EMA5 stretch buckets (already extensively analyzed, non-monotonic)
- Range Position (broke in 5th sample, unstable)
- Market Breadth (broke in 5th sample, unstable)
- Hour-of-day, day-of-week (no theoretical basis, pure data dredging)
- Pair-level performance (informs blacklist, not strategy edge)
- Anything not on this list

### Promotion criteria — what makes a variable "the key adjustment"

A variable qualifies for promotion to filter/threshold change in the validation
batch ONLY if ALL of these are true:

1. **N ≥ 20 trades per bucket** in the discriminating range (not just total N — the
   specific buckets being compared each need ≥20)
2. **WR gap between best and worst bucket ≥ 15 percentage points**
3. **Avg P&L % gap between best and worst bucket ≥ 0.20 percentage points**
4. **Pattern is direction-consistent OR has a documented theoretical reason for
   asymmetry** (e.g., "longs need rising EMA50, shorts need falling" is symmetric and
   theoretically grounded; "longs use criterion X but shorts use unrelated criterion Y"
   is asymmetric and suspect)
5. **TtP ratio sanity check**: the winning bucket should also have mean TtP ≤ 0.45
   (peak in first half of hold). If winners peak late, it's survival luck not edge —
   demote.
6. **Cross-tab confirmation**: if the variable is in a pre-committed cross-tab, it
   must hold up in the cross-tab too (not just the single-dimension table).

A variable that meets bars 1-3 but fails 4-6 is flagged as **"hypothesis, hold for
replication"** — not promoted to filter, watched in next batch.

A variable that meets all 6 bars is promoted to filter for validation batch.

### Stop rule for the validation batch (locked NOW)

After identified key variables are applied to the next 200-trade batch:

| Validation result | Verdict | Action |
|---|---|---|
| Avg P&L % combined ≥ +0.10% | Strategy has plausible edge | Advance to Phase 2 (tier system + leverage) |
| Avg P&L % combined +0.00% to +0.10% | Marginal — could be edge or noise | One more 200-trade cycle allowed at validated config |
| Avg P&L % combined -0.05% to +0.00% | Statistically indistinguishable from break-even | Strategy class likely lacks edge on 5m; structural pivot conversation |
| Avg P&L % combined < -0.05% | Negative edge confirmed | Strategy class doesn't work; abandon current architecture |

**These thresholds are LOCKED.** No moving the goalposts after the data arrives.
If this batch produces -0.04% Avg, that is a "marginal/structural pivot" verdict,
not a "let me explain why this time was special" verdict.

### Paper-mode analytical scope (CRITICAL)

This batch is in PAPER MODE. Paper's `_simulate_maker_entry_paper` over-fills
relative to real orderbook competition (CLAUDE.md Apr 18). Therefore:

**Paper-VALID for this batch (filter-layer findings):**
- All Tier 1, 2, 3 dimensions above (pure indicator math, identical code path)
- Bucket-level WR / Avg P&L %
- Regime/cross-tab analysis
- TtP ratio analysis

**Paper-INVALID — DO NOT analyze in this batch:**
- MAKER vs TAKER_FALLBACK WR gap (paper over-fills MAKER artificially)
- Maker timeout effectiveness (Amendment #6 — paper biased)
- Maker offset ticks effectiveness (Amendment #8 — paper biased)
- Entry slippage distributions
- Anything about fill mechanics

Fill-mechanics questions require a separate LIVE batch later. Do not attempt to
draw fill-mechanics conclusions from this paper batch.

### Checkpoint structure

| Checkpoint | Trades | Purpose | Decisions allowed |
|---|---|---|---|
| Health check | ~30-50 | Verify Tier 1 fields populating, no bot errors, data pipeline working | NONE strategic. Bug fixes only. |
| Mid-batch sanity | ~100 | Confirm batch is on-track operationally | NONE strategic. |
| Decision checkpoint | 200 | Full analysis per pre-committed dimensions | Identify key variables; commit to validation batch config |

### Mutual commitments during the batch

**Claude commits to:**
1. Not propose strategic config changes during data collection. If user asks "should
   we adjust X" mid-batch, the answer is always "no, we wait for 200."
2. Not interpret partial-batch data as preliminary findings. Sub-200 checkpoints
   are operational only.
3. At 200 trades, perform the analysis with discipline: pre-committed dimensions only,
   N ≥ 20 bucket bar enforced, no chasing of post-hoc patterns, no expanding the
   dimension list because something interesting appeared.
4. Be willing to declare "no clear key variable found" if the data doesn't support
   one. Honest negative result is better than manufactured positive result.

**User commits to:**
1. Not request mid-batch config changes. The discipline is the whole point.
2. Flag operational issues (bugs, capture failures) immediately — those ARE
   actionable mid-batch.
3. Accept the locked stop rule above. If validation batch hits -0.05% Avg, the
   verdict is what the locked rule says, not a re-litigation of thresholds.

### What success of THIS batch looks like

Not "the bot was profitable." The bot is expected to be break-even-ish.

Success = **at 200 trades, we can name 1-3 variables that meet the 6-criterion
promotion bar, with documented evidence**, AND we have a defensible config for
the validation batch built on those variables. That's it.

If we end at 200 trades with zero variables meeting the promotion bar, that
is ALSO a real result — it means filter-layer adjustments aren't the answer
and we move to structural conversation (timeframe, strategy class, fee structure).

### Why this plan is being locked in writing now

Past amendments were each individually defensible but collectively undisciplined —
the meta-question "is this exercise converging?" was never asked because each step
felt small. Writing the rules down before the data arrives prevents the same
failure mode this time. When mid-batch temptation arrives ("let's just tighten
this one filter"), this section is the answer.

### Pooling rule for Phase 1c amendments
Amendment #1 (Apr 17 AM, 13 trades), #2/3/4 (Apr 17 PM, same trades continuing +
later), #5 (Apr 18, 33 trades). Each amendment is a config inflection point. Do NOT
pool raw trades across amendments. When analyzing Phase 1c aggregate, treat each
amendment window as a sub-sample, flag N per sub-sample, compare Avg P&L % per
Core Operating Principles.

### Pre-commit rule re-validation against 3-sample pool (Apr 17 PM cross-tab audit)

Built the pooled BTC RSI × BTC ADX cross-tab from Apr 6 + Apr 13 + Apr 17 Phase 1b (Mar 30 excluded). This is a proper 3-sample validation of the pre-committed Phase 2 rules that were locked Apr 15 based on 4-sample aggregation (the original aggregation had Mar 30 carrying more weight than we realized). WR is the primary metric (leverage-invariant across the samples; Apr 6 used 20x-30x leverage so $ amounts aren't directly poolable with Apr 13/17 at 1x).

Pool methodology: for each cell, sum trades across all 3 samples, sum wins, compute pool WR = total_wins / total_N. Cells with pooled N ≥ 5 are listed. Cells with N < 5 pooled are flagged as insufficient data.

**LONG pool (cells with N ≥ 5):**

| BTC RSI | BTC ADX | N pool | Pool WR | Apr 13 | Apr 6 | Apr 17 | Pre-commit | 3-sample verdict |
|---|---|---|---|---|---|---|---|---|
| 45-50 | 15-20 | 5 | 60% | 5 @ 60% | — | — | — | OK |
| 50-55 | 20-25 | 6 | 50% | 5 @ 60% | 1 @ 0% | — | — | Inconclusive |
| 55-60 | 15-20 | 7 | 71% | — | 7 @ 71% | — | — | WIN ZONE (undocumented) |
| **55-60** | **20-25** | **9** | **78%** | 7 @ 71% | 2 @ 100% | — | — | **STRONG WIN (undocumented)** |
| 60-65 | 15-20 | 5 | 80% | 1 @ 100% | 4 @ 75% | — | — | WIN ZONE |
| **60-65** | **20-25** | **19** | **63%** | 9 @ 67% | 4 @ 75% | 6 @ 50% | **L-P1 PREMIUM** | **Confirmed winner, weaker than pre-commit** (pool 63% vs pre-commit 74%) |
| 60-65 | 25-30 | 5 | 60% | 2 @ 100% | 3 @ 33% | — | — | Mixed / sample-divergent |

**L-P2 (60-65 × 30-35): N=2 pooled without Mar 30.** The pre-commit's "4 trades, 100% WR across 3 samples" was heavily Mar 30-weighted. Without Mar 30, L-P2 essentially has no data — it's a hypothesis, not a validated rule.

**SHORT pool (cells with N ≥ 5):**

| BTC RSI | BTC ADX | N pool | Pool WR | Apr 13 | Apr 6 | Apr 17 | Pre-commit | 3-sample verdict |
|---|---|---|---|---|---|---|---|---|
| **<30** | **20-25** | **12** | **75%** | 6 @ 83% | 1 @ 100% | 5 @ 60% | **S-P1 PREMIUM** | **Validated** (pool 75% ≥ pre-commit 77%, consistent) |
| 30-35 | 25-30 | 7 | **57%** | 3 @ 100% | 1 @ 0% | 3 @ 33% | S-P2 PREMIUM | **Weakened** (pre-commit 83% → pool 57%; Apr 17 33% the concerning divergence) |
| 30-35 | 35+ | 7 | 43% | 7 @ 43% | — | — | — | LOSER (not pre-committed; already blocked by `btc_adx_max_short=30`) |
| **35-40** | **15-20** | **7** | **43%** | 1 @ 0% | 3 @ 33% | 3 @ 67% | **S-B1 HARD BLOCK** | **Validated** (43% < 55% threshold) |
| 35-40 | 20-25 | 10 | 50% | 7 @ 71% | 3 @ 0% | — | — | Sample-divergent (Apr 13 strong winner, Apr 6 strong loser) |
| **35-40** | **30-35** | **5** | **40%** | 3 @ 33% | — | 2 @ 50% | **S-B2 HARD BLOCK** | **Validated** (40% < 55%) |
| **45-50** | **15-20** | **6** | **33%** | 5 @ 40% | — | 1 @ 0% | **S-B3 HARD BLOCK** | **Validated** (33% clearly loser) |

**Summary of pre-commit re-validation:**

| Rule | Pre-commit label | Pre-commit WR | Pool WR (ex-Mar 30) | Pool N | Verdict |
|---|---|---|---|---|---|
| L-B1 (50-55 × 25-30) | HARD BLOCK | 17% | Only 3 trades pooled | 3 | **Insufficient data post-Mar-30** |
| L-P1 (60-65 × 20-25) | PREMIUM | 74% | **63%** | 19 | Winner but weaker |
| L-P2 (60-65 × 30-35) | PREMIUM | 100% | Only 2 trades pooled | 2 | **Insufficient data, barely exists** |
| S-B1 (35-40 × 15-20) | HARD BLOCK | 44% | **43%** | 7 | Validated |
| S-B2 (35-40 × 30-35) | HARD BLOCK | 54% | **40%** | 5 | Validated (stronger than pre-commit) |
| S-B3 (45-50 × 15-20) | HARD BLOCK | 37.5% | **33%** | 6 | Validated |
| S-P1 (<30 × 20-25) | PREMIUM | 77% | **75%** | 12 | Validated |
| S-P2 (30-35 × 25-30) | PREMIUM | 83% | **57%** | 7 | **Weakened — Apr 17 contributed 33% WR** |

**New WIN ZONE discovered in the pool (not pre-committed):**

**LONG 55-60 × 20-25: N=9 pool, 78% WR.** Strongest N-to-WR ratio in the LONG cross-tab. Add as L-P3 candidate for Phase 2 pre-commit list, pending 100-trade validation.

**What this means for Amendment #4's L-P2 preservation rationale:**

Amendment #4 relaxed `btc_adx_max_long` from 25 → 35 specifically to preserve L-P2 at BTC ADX 30-35 × BTC RSI 60-65. Post-audit: L-P2 has N=2 pooled (ex-Mar 30), so preserving it is based on a rule that barely has data under current-era configs. The decision is NOT reverted (option c below), but the confidence is explicitly weaker than Amendment #4's original commit claimed. If Phase 1c data populates the L-P2 cell with losing trades, we revisit.

**Rationale for keeping current config (Amendment #4's `btc_adx_max_long: 35`):**

- Option (a) Keep at 35 + document weakness: current. Preserves L-P2 (thin) and allows 100-trade checkpoint to produce real data.
- Option (b) Tighten to 30: would sacrifice L-P2 AND the thin-but-directional LONG 25-30 winners in Phase 1b (+0.27% N=17). More restrictive without strong evidence.
- Option (c) Keep 35 AND explicitly flag L-P2 as weakest pre-commit: chosen.

**Broader methodological takeaway:**

The Apr 15 pre-commit rules were aggregated across 4 samples. When we exclude Mar 30 (which had unusual regime and different config), the N per cell shrinks dramatically — L-P2 and L-B1 both drop below the 3-trade threshold. This means: **much of the pre-committed Phase 2 rule set rests on Mar 30's weight.** At 100-trade checkpoint, fresh Phase 1c cross-tab data will re-test each of these rules with current-config trades, and several may be retired or redefined.

**Rule for future decisions:**

> Before deploying or reverting a BTC-level raw-magnitude cap, build the cross-tab from all samples considered relevant (not just the raw-magnitude column), confirm which cells inside the capped range have pool N ≥ 5 and WR signal, and only cap if the evidence across cells inside the range is uniformly negative OR there are no PREMIUM ZONE cells inside the range. This rule is now part of the Apr 14 Filter design principle.

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

### Entry watchlist — SHORT "late-cycle BTC ADX" bad entry (Apr 17, cross-sample directional, needs Phase 1c confirmation)

**Corrected framing (Apr 17, after cross-sample validation).** Earlier version of this entry claimed a pair-level "capitulation combo-filter" (rising BTC ADX + ADXΔ 0.84-1.41 + RangePos <25% + Breadth ≥85%) separated Phase 1c's 5/5 losing SHORTs from winners. **Validation against Apr 13's 117-trade Entry Conditions by Close Reason table showed those pair-level dimensions do NOT discriminate winners from losers** — they describe the bot's general short-entry profile. The real discriminator sits one layer upstream: macro BTC ADX magnitude.

**What Phase 1c (Apr 17, 5S) actually tells us:** all 5 SHORTs lost via regime change during the trade. The new `btc_adx_dir_short: rising` filter was respected (100% had rising BTC ADX at entry) — but the filter only checks *direction*, not *absolute level*. A rising BTC ADX going 30→32 is a macro-trend reversal waiting to happen; a rising BTC ADX going 20→22 has room to run. Both pass our current filter.

**The actual discriminator (Apr 13 117-trade Entry Conditions by Close Reason table):**

| Bucket | # | BTC ADX | ADXΔ | RngPos | Breadth | Outcome |
|---|---|---|---|---|---|---|
| FL_DEEP_STOP L1 (S) | 10 | **31.7** | +1.10 | 12% | 72% | LOSS -$23.19 |
| FL_RECOVERED L1 (S) | 7 | 28.5 | +0.57 | 13% | 68% | LOSS -$6.66 |
| FL_STOP_LOSS L1 (S) | 2 | 28.1 | +0.91 | 6% | 80% | LOSS -$4.07 |
| TRAILING_STOP L1 (S) | 12 | **28.2** | +1.12 | 13% | 73% | WIN +$10.06 |
| TRAILING_STOP L2 (S) | 8 | 28.3 | +1.19 | 19% | 67% | WIN +$7.58 |
| FL_TRAILING_STOP L1 (S) | 6 | 26.6 | +0.34 | 14% | 73% | WIN +$6.97 |
| TRAILING_STOP L3/L4/L6+ (S) | 3 | **23.6-25.6** | +2.28 to +2.95 | 8-21% | 79-85% | BIG WIN |

Pair-level ADXΔ, RngPos, Breadth: **identical across winners and losers** (ADXΔ +1.10 vs +1.12, RngPos 12 vs 13, Breadth 72 vs 73). BTC ADX magnitude: **cleanly separated** (weighted-avg losers ~29.5 vs winners ~26.8, with biggest winners clustering at 23-26 and biggest loser at 31.7).

**Cross-sample status:** already 2-sample confirmed (Apr 13 + Apr 15, see "Cross-sample confirmed entry findings — BTC ADX × Slope for SHORTs" section). The 2-sample-confirmed rule: BTC ADX **18-27** with slope falling → WIN zone (27 trades, 81% WR, +$19.93); BTC ADX **≥28** with slope falling → LOSS zone (27 trades, 52% WR, -$10.71). Apr 17 Phase 1c regime-change cluster is directionally consistent with this but needs the full Entry Conditions by Close Reason table to confirm per-trade BTC ADX values.

**What to check in next batch (Phase 1c full sample, target ≥15 SHORT losses):**

1. **BTC ADX magnitude split.** For each SHORT loss bucket (REGIME_CHANGE, FL_REGIME_CHANGE, FL_DEEP_STOP, FL_STOP_LOSS) vs winner buckets (TRAILING_STOP L1/L2, FL_TRAILING_STOP), tabulate the AvgBTCADX column. Expect losers >29, winners <28.
2. **3-sample structural check.** If Apr 17 Phase 1c data also shows losers at BTC ADX ~29-32 and winners at ~24-27 → **3-sample structural** finding, ship the filter at Phase 1c → Phase 2 transition.
3. **Do NOT re-investigate the pair-level ADXΔ/RngPos/Breadth combo** — already falsified against Apr 13.

**Candidate fix (already pre-committed elsewhere in this doc — no new mechanism needed):**

Cap `btc_adx_max_short` at 28 (currently 40). This is the already-documented raw-dimension filter. Delivers via the pre-committed Phase 2 BTC RSI × BTC ADX cross-filter UI work, not a new combo-filter. If 3-sample confirms, raise from "candidate" to "ship in Phase 2 default config."

**Do NOT ship the filter on Apr 17 1-sample evidence.** The 2-sample Apr 13 + Apr 15 confirmation already exists for the "BTC ADX 18-27 vs ≥28" raw-dimension pattern; what's pending is whether Apr 17 Phase 1c makes it 3-sample structural. Wait for Phase 1c to accumulate ≥15 SHORT losses before committing.

**Methodological note — why this entry was corrected:**

The original "pair-level combo-filter" framing was a cautionary lesson in **not validating a 1-sample pattern against the right comparison.** I noted the 5/5 losers shared a pair-level profile and treated it as discriminative without first asking "do winners share this same profile?" The correct procedure — which the Apr 13 Entry Conditions by Close Reason table immediately showed — is to compare losers' bucket averages against winners' bucket averages on every dimension. If winners and losers share a dimension's value, that dimension is not the discriminator, regardless of how clean the losers' profile looks in isolation. The real discriminator was one layer up (macro BTC ADX level), which only became visible by cross-sample + winner-vs-loser comparison. **Future 1-sample pattern claims must include this winner-vs-loser cross-check before being added to any watchlist.**

## April 29, 2026 — Peak/Trough P&L Invariant Bug + Option A Fix (forward guard + diagnostic logs)

### What was observed
A single closed FL_TRAILING_STOP L1 order in paper-mode Phase 1c-Explore showed:
- Close P&L: $0.70 (+0.35%)
- Peak P&L %: +0.03%

This is logically impossible — peak P&L is *defined* as the maximum P&L reached during
the trade, and the close is by construction a P&L the trade reached. Peak must be ≥ close.

### Root cause
Three-tier P&L tracking pipeline in `services/trading_engine.py`:
1. **Realtime callback** (`_realtime_callback`, ~line 3897) updates `cache.peak_pnl` on every
   WebSocket price tick. Discrete tick stream — can miss intra-tick spikes if the WS
   callback isn't reached for that price level.
2. **Monitor loop** (~line 3042) writes `order.peak_pnl = exit_result.get("peak_pnl")` from
   the cache value at poll time.
3. **Close path** (`_close_position_locked`, ~line 2046) sets `order.pnl_percentage` from
   the actual exit price — but **never updates `order.peak_pnl`**.

If the close price produces a P&L higher than any tick the realtime callback observed,
`pnl_percentage` gets the correct value while `peak_pnl` stays stuck at the older cached
value. Result: `peak < close` in DB.

### Why this matters beyond cosmetics
- Peak/trough columns feed downstream analytics: TtP ratio, peak-by-bucket tables,
  flagged-exits NetRecover columns, post-exit regret deep dive
- Silently wrong peak data undermines the 6-criterion promotion bar in the locked
  Phase 1c-Explore plan
- We only *noticed* this case because the peak-vs-close gap was large (+0.03% vs +0.35%).
  Trades where close was +0.18% and peak should have been +0.20% would look subtly wrong
  and we'd never spot them. **One observed = lower bound, not accurate count.**

### Option A fix applied (deployed Apr 29, commit `1265e12`)

**Forward invariant guard** in `_close_position_locked`:
```python
_close_pct = pnl_data['pnl_percentage']
if order.peak_pnl is None or order.peak_pnl < _close_pct:
    order.peak_pnl = _close_pct
if order.trough_pnl is None or order.trough_pnl > _close_pct:
    order.trough_pnl = _close_pct
```

Tautological invariant — peak ≥ close, trough ≤ close. Risk near zero (no P&L change,
no close behavior change, no new branches; idempotent on already-correct rows).

**Diagnostic logs** when the guard activates:
- `[PEAK_INVARIANT_FIX] {pair} {direction}: peak_pnl was {old}% but close was {close}% — corrected (likely realtime-callback cache lag, reason={reason})`
- `[TROUGH_INVARIANT_FIX] {pair} {direction}: trough_pnl was {old}% but close was {close}% — corrected (...)` (only fires when `_close_pct < 0` to avoid noise on winners)

### What was deliberately NOT done
- **No historical backfill.** A migration to fix existing `peak_pnl < pnl_percentage`
  rows was considered (Option C) and rejected (Option A chosen instead). Backfilling
  on a 1-observation hunch risked papering over an upstream cache-lag issue we don't
  yet understand. Historical rows stay as captured. **The original FL_TRAILING_STOP
  order will continue to show peak +0.03% / close +0.35% in reports** — known data
  defect, not corrected.
- **No upstream investigation yet.** The realtime callback might be missing ticks
  systematically (worth investigating) or this might be a 1-in-200 fluke (not worth
  the engineering time). The diagnostic logs are designed to tell us which.

### Decision rule for follow-up at next checkpoint

At the 200-trade Phase 1c-Explore decision checkpoint, count `[PEAK_INVARIANT_FIX]` and
`[TROUGH_INVARIANT_FIX]` log lines:

| Frequency | Verdict | Action |
|---|---|---|
| 0 lines in 200 trades | 1-in-200 fluke confirmed | Guard is dormant insurance, no upstream work |
| 1-3 lines | Rare edge case (e.g., specific FL paths) | Guard is doing its job, no action |
| 4-10 lines | Notable but bounded | Optionally investigate FL_TRAILING_STOP path specifically |
| > 10 lines | Real upstream issue | Investigate `_realtime_callback` tick coverage, monitor poll cadence, cache contention |

**To pull the count from logs:**
```
grep -c "PEAK_INVARIANT_FIX" /var/log/web.stdout.log
grep -c "TROUGH_INVARIANT_FIX" /var/log/web.stdout.log
```

### What an upstream investigation would cover (if triggered)
1. **WebSocket tick coverage** — measure tick-arrival cadence per pair vs the actual
   price path on Binance. Missing ticks during high-volume moments = cache lag.
2. **Monitor poll cadence vs price velocity** — if the monitor loop runs every 60s but
   trailing-stop fires intra-poll on a fast move, the cache snapshot at the previous
   poll is what gets persisted to DB.
3. **Cache-write race** — `_realtime_callback` updates `order_info['peak_pnl']` without
   locking; possible (but unlikely at single-bot scale) for two callbacks to race.
4. **FL flow specifics** — does the FL system reset peak tracking on flag transition?
   If FL2 entry resets peak_pnl in cache, post-FL trailing-stop trades would all show
   incorrect peaks.

### Why this entry exists in CLAUDE.md
Forward guard masks the symptom by design. Without explicit follow-up tracking, we'd
forget to check whether the underlying cache-lag is real and frequent. This section is
the followup checklist — at the 200-trade checkpoint, run the grep, apply the decision
rule, document the verdict here.

## April 30, 2026 — BTC RSI Re-Validation Filter Mismatch Bug (Phase 1c-Explore data partially contaminated)

### What was observed
67-trade Phase 1c-Explore sanity check showed SIGNAL_EXPIRED breakdown with **18 LONG
entries blocked** under reason category "BTC RSI Out of Range" (avg BTC RSI ~69.3,
range 65.0-78.6). The user spotted the obvious anomaly: **BTC Global filter is OFF
in the current config (`btc_global_filter_enabled: false`)** — so BTC RSI shouldn't
be blocking anything.

### Root cause
Filter logic mismatch between entry path and re-validation path:

**At entry** (`services/trading_engine.py` ~line 3439):
```python
if signal in ["LONG", "SHORT"] and btc_global_enabled:
    ...
    if btc_rsi is not None:
        # BTC RSI check is GATED inside this block
        ...
```
With `btc_global_enabled = False`, the BTC RSI check is skipped at entry. Correct
behavior — matches the UI which shows BTC RSI under "Macro Trend Regime" (disabled).

**At re-validation** (`_revalidate_entry_signal`, `services/trading_engine.py` ~line 754):
```python
# BTC RSI range  ← UNCONDITIONAL, ignores btc_global_enabled
if original_direction == 'LONG':
    btc_rsi_min = getattr(th, 'btc_rsi_min_long', 0)
    ...
if new_btc_rsi is not None and (new_btc_rsi < btc_rsi_min or new_btc_rsi > btc_rsi_max):
    return False, f'btc_rsi_out_of_range_{round(new_btc_rsi, 1)}'
```
The re-validation step ran the BTC RSI check unconditionally, ignoring the
`btc_global_enabled` toggle. Result: signals that passed all entry filters then
got rejected from taker fallback by a phantom filter that didn't apply at entry.

### Why this slipped through analytical review for multiple sessions
The SIGNAL_EXPIRED breakdown was added Apr 29 and the BTC RSI Out of Range row was
visible from the first display. **Two sessions of analysis discussed "why is
SIGNAL_EXPIRED blocking entries" without anyone (Claude or user) immediately asking
"wait — is BTC RSI even an active filter in the current config?"** The user caught
it Apr 30 by spotting the contradiction between the SIGNAL_EXPIRED data and the
BTC Global toggle being off. **Lesson for future analysis: when a filter shows
activity in a report, the first sanity check is whether the filter is actually
enabled — before any interpretation of the data.**

### Fix applied (deployed Apr 30, commit `605b29b`)

Re-validation BTC RSI check now mirrors entry-time gating:
```python
btc_global = getattr(th, 'btc_global_filter_enabled', False)
if btc_global:
    # ... existing BTC RSI range check ...
```

The other re-validation checks (BTC ADX direction, BTC ADX range) are unchanged
because those filters DO live in the BTC Independent Filters section (per Apr 17
Option B refactor) — they correctly run unconditionally at both entry and
re-validation.

### Impact on Phase 1c-Explore current 67-trade dataset

**What stays clean and valid (do NOT reset):**
- All 67 closed trade rows: P&L, peak/trough, fees, all bucket assignments — unaffected
- All filter-layer bucket analyses (RSI, ADX, gap, BTC slope, BTC RSI buckets at entry, etc.)
- All cross-tabs (BTC ADX × EMA50 Align, Pair ADX × DI Spread, etc.)
- Signal Flipped category in SIGNAL_EXPIRED breakdown (43 entries — legitimate, signals genuinely decayed)
- BTC ADX Out of Range category (1 entry — BTC ADX filter is genuinely active in BTC Independent Filters)

**What is contaminated:**
- The 18 BTC RSI Out of Range entries in the SIGNAL_EXPIRED breakdown — these
  represent phantom blocks, not real filter rejections. They should NOT be
  interpreted as "filter caught a bad entry."
- Any conclusion that "SIGNAL_EXPIRED is acting as a quality filter" should be
  re-evaluated by mentally subtracting these 18.

**What was lost (opportunity, not data):**
- The 18 LONG signals that got blocked from taker fallback would have entered as
  TAKER_FALLBACK trades. We have no P&L data for what they would have done.
  The expectancy of "post-maker-timeout taker fallbacks under current config" is
  measured on a smaller-than-it-should-be sample.

### Decision: NOT resetting the batch counter

User decision Apr 30: 67 valid trades is genuine data; resetting throws away
clean P&L for a contamination that's isolated to one row of one breakdown table
and trivially subtracted mentally. **Continue toward 200 trades. Treat the
SIGNAL_EXPIRED breakdown's BTC RSI Out of Range row as "ignore this row in
analysis" until it stops appearing post-fix.**

### Follow-up at the 200-trade checkpoint

1. **Verify the fix worked**: Count BTC RSI Out of Range entries in the
   SIGNAL_EXPIRED breakdown for trades captured AFTER Apr 30 commit `605b29b`.
   With `btc_global_filter_enabled: false`, this count should be **zero**. Any
   non-zero count = fix didn't work or there's another path with the same bug.

2. **Re-evaluate "SIGNAL_EXPIRED as quality filter" hypothesis**: Now that the
   BTC RSI category will be empty (or non-spurious if BTC Global is later enabled),
   the breakdown is clean. The remaining categories (Signal Flipped dominant,
   BTC ADX Out of Range, BTC ADX Direction Flipped) are all legitimate per-trade
   filter rejections. The hypothesis can now be tested without the noise.

3. **Sub-batch annotation**: When analyzing the 200-trade aggregate, flag that
   trades 1-67 ran under buggy re-validation, trades 68+ ran under fixed logic.
   The actual P&L data is comparable across the boundary because the bug only
   affected which trades happened, not how they were executed. So pooling is
   acceptable — but if doing fine-grained SIGNAL_EXPIRED analysis specifically,
   exclude trades 1-67 from the SIGNAL_EXPIRED count.

### Phase 2 work this surfaces

The proper long-term fix isn't this gating patch — it's the previously-documented
Option B refactor: move `btc_rsi_min/max_long/short` OUT of the
`if btc_global_enabled:` block at entry and INTO the BTC Independent Filters
section (alongside BTC ADX range and BTC ADX direction, which were moved Apr 17).
Once that ships, the BTC RSI filter runs independently of the BTC Global toggle
and the entry-vs-revalidation paths converge naturally without conditional gating.

This patch (gating re-validation on `btc_global_enabled`) is a stop-gap that
keeps the two paths consistent under current code. When Phase 2 ships the
refactor, this patch should be removed and replaced with an unconditional check
on `btc_independent_rsi_enabled` (or whatever the new flag is named).

### Why this entry exists in CLAUDE.md

To follow up at the 200-trade checkpoint that the fix actually eliminated the
spurious entries, AND to remember that this stop-gap needs to be replaced
properly when the Option B refactor ships.

## April 30, 2026 — Winner Exit Optimization Plan (200-trade counterfactual analysis)

### The problem this addresses

Apr 14 raised TP min 0.40 → 0.50 and pullback 0.08 → 0.20 to capture more of
post-peak winners. At 67-trade Phase 1c-Explore checkpoint:
- TRAILING_STOP L2 BULLISH: avg close +0.50%, post-peak runway +0.95%
- TRAILING_STOP L2 BEARISH: avg close +0.52%, post-peak runway +1.77%

The trailing stop is still exiting on mean-reversion mid-trend pullbacks while
the trend continues. Bigger-peak trades have disproportionately bigger tails:
+0.71% peak → +0.95% post-peak; +0.72% peak → +1.77% post-peak. **Fixed pullback
treats all peaks identically, which is structurally wrong for this distribution.**

### Why naive solutions don't work (sanity-checked)

Three options were debated:

1. **Wider fixed pullback (0.20 → 0.30)** — same mechanism, just shifted. Trades
   small-winner capture for marginal big-winner improvement. Doesn't address the
   peak-vs-tail relationship.

2. **RSI exhaustion exit at peak ≥ +0.50%** — fails because the trailing stop's
   0.20% pullback fires before RSI exhaustion has time to develop. The post-peak
   RSIEx data we keep observing is *post-exit*, meaning the bot was already gone
   when RSI exhausted. Adding RSI exit at the same peak threshold without
   loosening trailing = same outcomes as today. (User caught this via correct
   reasoning Apr 30; original Claude framing was wrong.)

3. **Tier-aware pullback formula** — pullback widens with peak. Keeps tight
   exit on small winners, gives big winners room. The mechanism that actually
   addresses the peak-vs-tail data signature.

### Proposed analysis at 200-trade checkpoint

This is **counterfactual analysis on existing trade data — no config change**.
Per locked Phase 1c-Explore plan, no strategic changes during the batch. This
is identification of the candidate rule for the next validation batch.

**Step 1: Run counterfactual on each pullback rule using captured peak/trough/
close data:**

| Rule | Definition |
|---|---|
| A. Current (baseline) | Fixed 0.20% pullback |
| B. Wider fixed | Fixed 0.30% pullback |
| C. Slope 0.30 | `pullback = 0.20% + 0.30 × max(0, peak - 0.50%)` |
| D. Slope 0.50 | `pullback = 0.20% + 0.50 × max(0, peak - 0.50%)` |
| E. Hard threshold | < +0.80% peak: 0.20% pullback; ≥ +0.80%: trailing OFF, hold to signal/regime exit |

**Step 2: Compute the comparison table:**

| Rule | All winners Avg% | Tail (peak ≥ +0.80%) Avg% | Small (peak +0.50-0.70%) Avg% | N tail | N small |
|---|---|---|---|---|---|
| A: 0.20 fixed | ? | ? | ? | ? | ? |
| B: 0.30 fixed | ? | ? | ? | ? | ? |
| C: slope 0.30 | ? | ? | ? | ? | ? |
| D: slope 0.50 | ? | ? | ? | ? | ? |
| E: cutoff @ 0.80 | ? | ? | ? | ? | ? |

**Step 3: Apply decision rule.** Winning rule must satisfy BOTH:
- Tail capture improves ≥ +0.20 pp on big-winner Avg P&L %
- Small-winner give-back ≤ -0.05 pp degradation

If multiple rules pass, pick the simplest (slope formula > tier table > hard cutoff)
to minimize parameters and overfitting risk.

If NO rule passes both gates, conclusion is **"current trailing is approximately
right; no exit change in next batch."** That's a valid negative result.

### What gets tested in the validation batch (next 200 trades)

The single winning rule from Step 3 gets deployed as the only strategic change
for the next 200-trade batch. Pre-committed decision rule at end of validation:

| Validation result | Verdict | Action |
|---|---|---|
| Combined Avg P&L % improves ≥ +0.10 pp vs current | Rule confirmed | Ship to live |
| Combined Avg P&L % flat (±0.05 pp) | Counterfactual didn't predict live correctly | Revert; sample noise was driving counterfactual |
| Combined Avg P&L % worse | Rule actively harmful | Revert; deeper investigation needed |

### What is explicitly NOT proposed

- **Not a multi-rule live experiment.** One change per validation batch — clean attribution.
- **Not RSI exhaustion exit alone.** It can't fire during the trade with current
  trailing in place. RSI overlay is at most a Layer 3 follow-up after wider
  pullback validates.
- **Not data-fitted tier breakpoints.** Slope formula has one design parameter;
  tier breakpoints are multiple. Slope generalizes better, less overfitting.
- **Not changes deployed mid-Phase-1c-Explore batch.** Per locked plan, this
  analysis happens AT the 200-trade checkpoint, not before.

### Counterfactual implementation notes (for whoever runs the analysis)

The counterfactual is approximate — we have peak%, trough%, close%, peak_min,
trough_min per trade, but not moment-by-moment tick data. Approximations:

- For trades that closed via TRAILING_STOP, the actual exit was "first 0.20%
  retrace from peak." Under rule X with pullback Y, the exit would be "first
  Y% retrace from peak." Since trough < close ≤ peak, we know retrace size at
  close was ≥ X%. Under wider Y > X, we'd hold longer.
- For "hold longer" scenarios, terminal P&L is bounded by:
  - Upper bound: peak + post-peak runway (if available; from PostPeak% column)
  - Lower bound: trough (if widening pullback would let trade hit trough first)
- Conservative simulation: assume trade hits peak + post-peak (continued trend),
  then retraces by Y% to compute counterfactual close.

This is directionally correct but not exact. If the counterfactual analysis at
200 trades produces clear signal (>20 bp improvement on tail captures), the
validation batch will tell us whether the directional read translates. If
counterfactual is borderline (<10 bp), don't deploy — likely sample noise.

### Caveats and follow-up

1. **Whole framework assumes peak-vs-post-peak correlation persists.** Current
   67-trade data shows bigger peaks → bigger tails. If 200-trade data breaks
   this relationship (correlation drops below 0.3 across all winners), Option 3's
   logic collapses. The counterfactual analysis verifies this implicitly — if
   wider pullback doesn't outperform fixed pullback, the underlying assumption
   was wrong.

2. **Regime sensitivity.** Current data is two regimes (BULLISH choppy,
   BEARISH trending). Big-tail data is concentrated in BEARISH SHORTs. If next
   200 trades are pure BULLISH choppy, the big-tail population we're optimizing
   for may not reappear.

3. **RSI exhaustion overlay (Layer 3) is deferred** until Step 2 winning rule
   is validated. Cannot test RSI overlay first — requires loosened trailing
   to give RSI time to fire during trades.

### Why this entry exists in CLAUDE.md

The exit problem (post-peak runway loss) is real and recurring across multiple
samples. The counterfactual approach is the highest-EV use of 200-trade data
because it tests multiple rules without committing to any. This section is the
analysis blueprint to run at the 200-trade checkpoint — exact rules, exact
tables to compute, pre-committed decision criteria.

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
