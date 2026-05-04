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

### Amendment (Apr 30) — Rule F added, Rule G dropped

After discussion of the trailing-stop-vs-RSI-exhaustion competition dynamic, the
rule menu narrows. Rules A-E in the table above all keep the trailing stop as
the dominant exit mechanism (varying only the pullback width). The hypothesis
that RSI exhaustion is a *better* exit signal than trailing pullback cannot be
tested against any of A-E because trailing always fires first under current
logic. New rule added to test the RSI hypothesis directly:

| Rule | Definition |
|---|---|
| F. RSI-only at L2+, NO trailing | At peak ≥ L2 trigger, disable trailing entirely. Exit fires only on RSI exhaustion (or signal-lost / regime change / FL backstops). Cleanest test of the RSI-as-better-exit-signal hypothesis. |

**Rule G (RSI exhaustion + current 0.20% trailing) was considered and dropped.**
Reason: empirically the trailing stop fires before RSI exhaustion has time to
develop on this distribution (Post-Exit Regret observation Apr 30 — RSI
exhaustion data we keep observing is *post-exit*, meaning the bot was already
gone by the time RSI exhausted). Adding RSI as a parallel exit at the same peak
threshold without loosening trailing produces near-identical outcomes to
current. No analytical value vs A-E.

**Counterfactual approach for F:**
F cannot be cleanly counterfactualed from current peak/trough/close data alone —
it requires knowing intra-trade RSI trajectory (when RSI exhaustion would have
fired). At the 200-trade checkpoint, if A-E counterfactual selects a
wider-pullback rule as winner, validation batch ships that. If A-E shows no
clear winner AND RSI exhaustion is still hypothesized as the dominant exit
signal, the validation batch ships **F** (cleanest test).

**Decision sequencing at 200-trade checkpoint:**
1. Run counterfactual on A-E first (existing data sufficient)
2. If a rule from A-E passes the dual gate (≥+0.20pp tail capture, ≤-0.05pp
   small-winner give-back) → ship it for validation, defer F
3. If no A-E rule passes → ship F for validation batch (RSI hypothesis gets a
   real test)

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

## May 1, 2026 — BE Layer Introduction Plan (sister analysis to Winner Exit)

### Critical framing correction (vs the original Apr-30/May-1 draft)

The first draft of this plan called itself a "BE L1 Trigger Optimization Plan"
and proposed lowering an active 0.15% trigger. **That framing was wrong.**
Inspection of the live config on May 1 confirmed:

```
VERY_STRONG  ...  BE:N  99%  99%  99%  99%  ...
STRONG_BUY   ...  BE:N  99%  99%  99%  99%  ...
```

BE L1 (and all BE levels L2-L5) are **disabled**. All triggers set to 99%.
Active exit layers are: TP min 0.50% + pullback 0.20% trailing, FL system,
Signal Lost, Regime Change Exit, hard SL at −0.9%.

**Implication for the "Positive, No BE" bucket interpretation:**

In the Stop Loss Deep Dive table and the SL Profile column added May 1, the
"Positive, No BE" classification reads `peak >= be_level1_trigger` where the
trigger value is loaded from config. Since `be_level1_trigger: 0.15` is still
stored even though `be_active` is false, the report categorizes trades using
that phantom 0.15% threshold. **Under current live behavior, BE physically
cannot arm regardless of peak — so every positive-peak loser is in the
Positive-No-BE bucket by definition.** The classification is descriptive, not
diagnostic of an exit-layer failure.

This means the question is not "should we lower an active trigger" — it is
**"should we introduce BE as a new exit layer at all, and at what trigger?"**
Structural decision, not parameter tuning.

### What this plan now addresses

In the May 1 109-trade report, positive-peak losers dominate the SL bucket:
- BULLISH: 30/46 SL trades with avg peak +0.18%, closing at avg −0.48%
- BEARISH: 4/6 SL trades with avg peak +0.22%, closing at avg −0.47%

These trades went green, peaked at small positive values, then rode through
the trailing-stop activation zone (TP min 0.50% never reached → trailing never
armed) all the way to the −0.9% hard SL. Introducing a BE layer with trigger
X < 0.50% (the trailing TP minimum) and a small positive offset Y would
intercept these trades and lock in +Y instead of letting them ride to SL.

This is the loser-side counterpart to the Winner Exit Optimization Plan
(Apr 30, which targets high-peak winners getting exited too early on
trailing pullback). Different trade populations, statistically independent,
analyzable at the same 200-trade checkpoint.

### What we have per trade

`peak_pnl`, `trough_pnl`, `pnl_percentage` (close), `close_reason`. Sufficient
to simulate any candidate (trigger X, offset Y) pair without new
instrumentation.

### The simulation mechanic

For each closed trade, under candidate BE layer (trigger X, offset Y) where
X > 0 and Y > 0 and X > Y:
- **If peak ≥ X AND trough ≤ Y** → BE would have armed and exit at +Y →
  simulated close = +Y (replaces actual close)
- **If peak < X** → BE never arms, original exit stands
- **If peak ≥ X AND trough > Y** → BE armed but never retraced to the floor,
  original exit stands (means the trade either continued up to a trailing
  exit or hit a non-SL exit reason like regime change or signal lost)

This identifies which currently-losing trades would have been intercepted by
the new BE layer.

### The key analytical question this answers

The user's observation: with so many Positive-No-BE losers, it's hard to tell
whether they're **bad entries** (peaks too small to ever be real winners) or
**bad exits** (peak was real but exit failed to capture it).

The simulation resolves this:

| Simulation outcome | Interpretation | Action |
|---|---|---|
| Lowering trigger to 0.05-0.10% rescues most Positive-No-BE losers to +Y | **Exit is the problem.** Entries had real (small) momentum; we just had no mechanism to lock it in | Introduce BE layer, keep entry filters |
| Even at trigger 0.05%, most trades hit trough < Y before peak retraced → still loss | **Entry is the problem.** These trades had no real momentum; peak was noise within the SL distance | Don't add BE; tighten entry filters (TtP < 0.4 buckets, EMA50 alignment, DI spread) |
| Mixed: ~half saved by BE, ~half still die | **Both are problems.** Add a BE layer for the saveable subset; build entry filters from the cross-tab profile of the unsaveable subset | Two-pronged fix |

This is the most informative single test we can run on existing data.

### Two distinct effects to measure separately

| Effect | Sign | Where it shows up |
|---|---|---|
| **Save Positive-No-BE losers** | + | SL trades with peak ≥ X, currently rode to −0.9% SL, would now exit at +Y |
| **Steal from currently-winning trailing trades** | − | Trades where trailing currently captures more than +Y, but BE armed earlier and floored at +Y |
| **No effect on FL / signal-lost / regime exits** | 0 | Those exits fire on different conditions (peak/pullback irrelevant); BE layer is independent |

The "steal from winners" risk is real but bounded: trailing only activates
at peak ≥ 0.50% (TP min). If we set BE trigger X < 0.50%, BE arms before
trailing does. If trade then retraces to the BE floor before reaching trailing
TP, we lock at +Y instead of riding to a higher trailing exit. Magnitude
depends on how often trades crossed Y on the way down before reaching peak
≥ 0.50%.

### Counterfactual grid

Simulate (trigger X, offset Y) pairs from this small grid:

| X (trigger) | Y (offset) | Hypothesis being tested |
|---|---|---|
| 0.05% | 0.02% | Maximum sensitivity — catches even the weakest peaks, locks tiny profit |
| 0.08% | 0.04% | Moderate — requires real positive movement, locks meaningful profit |
| 0.10% | 0.05% | Conservative — only arms on trades that showed half the trailing-TP threshold |
| 0.15% | 0.10% | The original disabled-default — BE arms only on trades that nearly reached trailing TP |

Pick the (X, Y) pair that maximizes net Avg P&L % improvement while passing
all promotion gates below.

### Pre-committed decision rule (locked May 1, before 200-trade data arrives)

A new BE layer qualifies for promotion to the validation batch only if ALL
of the following are true:

1. **N ≥ 10 newly-saved trades** in the simulated bucket (statistical floor)
2. **Net Avg P&L % across full sample improves ≥ +0.10pp** vs current (no-BE) config
3. **Positive-No-BE count drops by ≥ 50%** (the whole point of the change)
4. **No winner-bucket Avg P&L % degrades by > 0.05pp** in the simulation
   (the "steal from winners" effect must be small)
5. **Same (X, Y) pair works on both LONG and SHORT subsets** (or documented
   theoretical reason for asymmetry)

If no (X, Y) pair passes all 5 gates → **don't introduce BE**; the
Positive-No-BE losses are entry-driven, not exit-driven. In that case,
attention shifts to entry-filter work (TtP, EMA50 alignment, DI spread cross-
tabs at the same 200-trade checkpoint).

### What invalidates the test

- **Sample skew**: if Positive-No-BE bucket is < 10 trades in the 200-trade
  sample, decision deferred to 400-trade checkpoint
- **Null trough_pnl**: trades closed before trough was logged have null
  trough; exclude from simulation
- **Regime confound**: if Positive-No-BE concentrates in one regime, the
  saved-losers effect is regime-specific. Document as conditional, not
  universal
- **Counterfactual ≠ live**: simulation is approximate. < +10bp improvement
  in counterfactual → don't deploy (likely sample noise)
- **Peak-vs-trough sequencing assumption**: the simulation assumes peak was
  reached before trough hit Y. If trough hit Y first (then peak rallied
  back later), BE wouldn't actually fire because peak hadn't yet reached X.
  The data has `peak_min` and `trough_min` (time-to-peak and time-to-trough)
  on most trades — use these to filter out cases where trough preceded peak

### Implementation cost (when ready to ship)

Config change in `trading_config.json` per confidence level — change
`be_level1_trigger` and `be_level1_offset` from 99 to the validated values,
and ensure `be_active` flag is true. Single edit per tier, no code changes
(BE logic exists, just disabled). Instant revert by re-setting trigger to 99.

### Validation batch decision rule (post-200-trade simulation, in next 200)

If simulation passes the 5 gates and a winning (X, Y) is shipped:

| Validation result | Verdict | Action |
|---|---|---|
| Combined Avg P&L % improves ≥ +0.05pp vs current no-BE | BE introduction confirmed | Lock as default |
| Avg P&L % flat (±0.03pp) | Counterfactual didn't translate; BE adds friction without saving meaningful trades | Revert to BE disabled |
| Avg P&L % worsens | BE actively harmful (likely stealing from winners more than saving losers) | Revert; entry-filter work becomes the only path |

### Coordination with Winner Exit Optimization Plan

Both plans run their counterfactual at the same 200-trade checkpoint.
Statistically independent (different trade populations).

**Ship in SEPARATE validation batches**, one at a time. Per the locked
Phase 1c-Explore plan: clean attribution requires single-variable changes per
validation batch.

Order if both pass simulation:
1. **First**: whichever passes by larger projected magnitude
2. **Second**: the other, after first is validated (or reverted)

If both pass simulation but neither delivers in validation, the strategy's
edge is more fragile than the simulations suggested — escalate to structural
review, not more parameter tweaking.

### Why this entry exists in CLAUDE.md

The Positive-No-BE pattern is the largest single addressable loss bucket
in the dataset. The current BE-disabled config means we have a clean test
of "exit-layer absence" vs "entry-quality issue." The counterfactual on
existing data resolves the ambiguity that the user correctly flagged on
May 1 ("hard to tell if it's a bad entry or a bad exit"). Locking the
methodology now — before the 200-trade data arrives — prevents post-hoc
parameter selection bias.

### What was wrong with the Apr-30/May-1 first draft of this section

The first draft assumed BE was active at trigger 0.15% and proposed lowering
it. That misread the config (`BE:N` flag in the report header). Lesson for
future plans: when proposing changes to an exit/filter parameter, **first
verify whether the parameter is currently active**. The phantom-trigger
classification in Stop Loss Deep Dive masked the fact that BE wasn't
running at all. Fixed by re-reading the report config block and matching it
against the proposed change. This is a methodology-level lesson, not just a
this-section bug.

## May 2, 2026 — Reporting granularity expansion (no strategy changes)

Pure reporting changes deployed mid-Phase-1c-Explore to surface detail in
buckets that were previously collapsed. **No filter, exit, or entry logic
touched** — only the analytical surface widened. Locked Phase 1c-Explore plan
remains in effect (frozen config until 200 trades).

### What changed

| Table | Before | After |
|---|---|---|
| **Performance by BTC Entry RSI** | Lowest bucket was `<30` (lump) | Split into `<20`, `20-25`, `25-30`. SHORT-side `<30` is the highest-WR bucket in the data — splitting reveals whether the edge concentrates in 25-30 or extends below |
| **Performance by Pair EMA20 Slope (abs)** | Lowest bucket was `0.00-0.02%` then `0.02-0.04%` | Split into `<0.01%`, `0.01-0.02%`, `0.02-0.03%`, `0.03-0.04%`. The 0.02-0.04% bucket was where most activity concentrated; this exposes whether sub-buckets discriminate |
| **Performance by BTC EMA20 Slope (abs)** | Same as Pair Slope (shared bucket definition) | Same split — symmetric change |
| **BTC Slope × BTC ADX Cross-Tab** | Lowest slope row was `<0.06%` (lump containing most of the dataset) | Split into `<0.02%`, `0.02-0.04%`, `0.04-0.06%`. Refactored to **Dir-first** column format matching other cross-tabs (Dir / Slope / ADX / # / WR / Avg$ / Avg% / Total$ / Conf) |
| **Pair EMA20 Slope × Pair ADX Cross-Tab** | Did not exist | NEW table, mirrors the BTC version but with pair-level slope and pair-level ADX. Same Dir-first format. Placed directly above the BTC version. Pair ADX bins match the existing "Performance by Entry ADX" cadence (`<15`, `15-18`, `18-22`, `22-25`, `25-28`, `28-30`, `30-33`, `33+`) — tighter than BTC bins because pair ADX clusters more narrowly under current filters |
| **Entry Conditions by Outcome (Winners vs Losers)** | Did not exist | NEW summary table placed directly after `Entry Conditions by Close Reason`. Same column set, but collapsed to up to 4 rows: `Winners L`, `Losers L`, `Winners S`, `Losers S` (a row is omitted if its bucket has zero trades). Winner = `pnl > 0` after fees; Loser = `pnl <= 0`. Higher N per row makes cell-level statistics meaningful (e.g., 42 Winners L vs 76 Losers L on the May 2 sample, vs the close-reason table's avg ~10 trades per row). Purpose: enforce the Apr 17 methodological rule — when a pattern shows up in losers, compare against winners on the same dimensions before treating it as discriminative. SL Profile column populates only on Loser rows. Backend: new `entry_conditions_by_outcome` payload field built alongside `entry_conditions_by_reason` in `main.py::_compute_performance`. UI: `templates/index.html`, table id `entry-conditions-outcome-body`. Text exports: both clipboard and saved-file sites. **No filter/exit/entry logic touched** — pure reporting addition, allowed mid-Phase-1c-Explore-batch. |

Backend lives in `main.py` (`_build_slope_adx_crosstab` helper covers both
cross-tabs). UI section `Pair EMA20 Slope x Pair ADX Cross-Tab` placed above
the BTC one in the Performance dashboard. Both new/refactored sections appear
in both text-export sites. SL Profile column shipped May 1 still present.

### Why this matters for the 200-trade decision checkpoint

These finer buckets directly inform several pre-committed decisions:

1. **BE Layer Introduction Plan (May 1) — N≥10 saved-trade gate.** The peak/trough
   simulation needs to identify Positive-No-BE losers by their actual peak
   distribution. The `<0.04%` Pair Slope splits give us per-bucket peak
   characteristics; trades with peak <0.04% are the structural candidates for
   "could a low-trigger BE have saved them?"

2. **Pair Slope × Pair ADX cross-tab — entry filter design.** This is the missing
   piece for the "is it bad entry or bad exit" question. If Positive-No-BE losers
   cluster in specific (Pair Slope, Pair ADX) cells while winners cluster
   elsewhere, the cross-tab makes the entry-filter signature visible at a glance.
   Mirror the cell-by-cell analysis already done on BTC RSI × BTC ADX.

3. **BTC Slope × BTC ADX low-end split.** Most current Phase 1c-Explore data
   sits in BTC Slope `<0.06%`. Without the split we can't tell whether the
   `<0.02%`/`0.02-0.04%`/`0.04-0.06%` sub-buckets behave differently. Apr 18
   amendment #5 already showed BTC ADX 20-25 SHORT performance flipped in the
   most recent regime — slope sub-buckets may explain why.

4. **BTC Entry RSI `<30` SHORT split.** Currently 84% WR on N=19 in BEARISH —
   strongest SHORT signal in the data. Splitting `<30` into `<20`/`20-25`/`25-30`
   tests whether the edge is concentrated at one sub-zone (likely 25-30 given
   where BTC RSI typically sits in bearish regimes). If `<25` shows different
   WR than `25-30` on N≥10 each, this becomes a candidate refinement to the
   pre-committed S-P1 PREMIUM ZONE rule.

5. **Entry Conditions by Outcome — primary winner-vs-loser attribution view.**
   When the 200-trade analysis runs, this is the FIRST table to look at for
   any "is dimension X discriminative?" question. The 4-row summary collapses
   the 11+ close-reason rows of the existing table into the comparison the
   promotion bar actually requires (winners vs losers, per direction). For
   any single-dimension finding (e.g., "EMA50 alignment matters"), check this
   table first: if Winners L and Losers L don't differ on EMA50Align, the
   dimension is not discriminative — period. This view also resolves the
   May 1 BE Layer Plan's "is it bad entry or bad exit?" question more
   cleanly than any other surface: if Winners L and Losers L have nearly
   identical entry signatures, the loss source is exit-side; if their
   signatures diverge, entry filtering has room.

### Promotion rule for findings from the new buckets

Same 6-criterion bar that applies to all Tier 1 dimensions (locked Apr 28):
- N ≥ 20 trades per bucket in the discriminating range
- WR gap between best/worst bucket ≥ 15 pp
- Avg P&L % gap ≥ 0.20 pp
- Direction-consistent OR documented theoretical asymmetry
- TtP ≤ 0.45 sanity check on winning bucket
- Cross-tab confirmation (the cell holds up in the 2D view, not just 1D)

The new finer buckets make it MORE LIKELY that single-cell N falls below 20
on the 200-trade sample. For example, `BTC RSI <20` may have N=0 or N=1 in
the entire batch. **A bucket with N<10 contributes ZERO weight to promotion
decisions** — it's exploratory data only. Don't lower the bar to fit smaller
samples; either the bucket is well-populated and the signal is real, or it
isn't and the conclusion defers to the 400-trade sample.

### What this does NOT do

- Does NOT change any filter, exit, or entry logic
- Does NOT reset the 200-trade counter
- Does NOT alter the locked Phase 1c-Explore plan or any pre-committed rule
- Does NOT add new dimensions captured at trade time (uses fields already on
  Order: `entry_btc_rsi`, `entry_ema20_slope`, `entry_btc_ema20_slope`,
  `entry_adx`, `entry_btc_adx`)

### Why this entry exists in CLAUDE.md

When the 200-trade analysis runs, the analyst needs to know:
1. The granularity of the new buckets (so cell-N expectations are calibrated)
2. The pre-committed promotion bar applies — finer buckets don't relax it
3. The new Pair Slope × Pair ADX cross-tab is the primary tool for the
   "bad entry vs bad exit" attribution question on the Positive-No-BE bucket

This section is the inventory of analytical surface available at the
checkpoint. Without it, future-Claude might not realize the cross-tab exists
and would re-run the same analysis on lower-resolution single-dimension tables.

## May 2, 2026 — SIGNAL_EXPIRED enrichment (Aborted entries become first-class analytical population)

### Problem this addresses

Amendment #7 (Apr 18) introduced re-validation of the entry signal at maker-wait
timeout: if BTC ADX direction has flipped, BTC ADX/RSI moved out of range, or the
core get_signal output has changed, the bot aborts the taker fallback and
persists a SIGNAL_EXPIRED Order row. By the May 2 67-trade Phase 1c-Explore
checkpoint, **97 SIGNAL_EXPIRED rows had accumulated — roughly 62% the size of
the 156 CLOSED trade count.** Yet we knew nothing about them beyond reason-category
counts.

The user correctly flagged: that's not a footnote. It's a parallel population of
"what the bot almost did but didn't" — and right now the only thing we measure is
the *reason category*. We have no idea whether those 97 were good calls (kill
switch saved bad trades) or bad calls (we're cutting winners and never seeing
them).

The Apr 18 `_record_signal_expired_order` persisted a "minimal" Order row:
`pair`, `direction`, `confidence`, `status`, `entry_price`, `close_reason`,
`opened_at = closed_at = now`. **All entry indicator fields were NULL.** The
maker-wait elapsed time was discarded too (opened_at == closed_at).

### What was added

**Code changes (services/trading_engine.py — ~150 lines, all in one file):**

1. **`_record_signal_expired_order` signature extended** with all entry-indicator
   parameters (~25 fields) plus `wait_seconds`. All are optional and default to
   None — legacy call sites still work; new call sites populate everything.
2. **opened_at is back-dated** when `wait_seconds` is provided:
   `opened_at = now - timedelta(seconds=wait_seconds)`. So `closed_at - opened_at`
   = real maker wait elapsed before re-validation killed the entry. Pre-deploy
   rows have `opened_at == closed_at` (wait=0) and are excluded from wait-time
   medians by the reporting code.
3. **`_try_maker_entry` and `_simulate_maker_entry_paper`** now return
   `wait_seconds: float(timeout)` in the SIGNAL_EXPIRED skipped dict. Since
   re-validation only fires after the full maker-poll loop has exhausted, the
   wait equals the configured `maker_timeout_seconds` (currently 40s).
4. **Both call sites in `open_position`** (the live and paper branches) forward
   all the entry indicators they already had in scope (no recomputation, no
   refetch — the params were always there) plus `wait_seconds` from the result
   dict.

**Critical: no schema migration needed.** All required `entry_*` columns already
existed on the Order model. The fields just weren't being populated for
SIGNAL_EXPIRED rows.

**Reporting (main.py + templates/index.html):**

5. **Entry Conditions by Outcome extended** — was 4 buckets (Winners L / Losers L /
   Winners S / Losers S), now up to 6 (adds Aborted L / Aborted S). Aborted rows
   render in amber to distinguish from Winners (emerald) and Losers (red). Same
   column set as the existing rows. SL Profile column shows `-` for Aborted (peak
   is always 0 on never-opened entries; classification would be misleading). Sort
   order: Winners L, Losers L, Aborted L, Winners S, Losers S, Aborted S.

6. **Signal Expired Breakdown gets MedWait / p90Wait / MaxWait columns.**
   Computed per category from `closed_at - opened_at` on SIGNAL_EXPIRED rows
   where `wait_seconds > 0` (i.e., post-deploy rows only — historical zeros are
   filtered out by the wait_seconds > 0 check, not by date).

### Why this matters for the 200-trade decision checkpoint

Three scenarios become testable for the first time:

| Aborted L/S profile vs Winners L/S, Losers L/S | Verdict | Action at checkpoint |
|---|---|---|
| Matches Loser profile (similar avg RSI/ADX/etc.) | Re-validation correctly self-protecting | Keep on; count as silent edge |
| Matches Winner profile | Re-validation murdering good trades | Tighten/remove specific re-validation criteria |
| Neither — own profile | Aborted are a third population, neutral expectancy | Revisit Amendment #6 (40s timeout may be too long) |

The wait-time stats answer a different question:
- If MedWait clusters near 40s → Amendment #6's 40s timeout is the cause; reverting
  to 20s would cut abort rate
- If MedWait near 0 → flaky signal generator producing momentary signals; timeout
  is irrelevant

### Caveats

1. **Historical SIGNAL_EXPIRED rows persisted before this commit have NULL
   indicator values forever.** Only post-deploy aborts populate the new fields.
   The 97 rows in the May 2 67-trade sample stay analytically dark. Timing
   matters: this shipped before the 200-trade checkpoint so post-deploy aborts
   should accumulate enough N for analysis.

2. **The Apr 30 BTC RSI re-validation bug contaminated 18 of the legacy 97 rows**
   (CLAUDE.md Apr 30 entry — pre-`605b29b` re-validation ran BTC RSI check
   ungated even though `btc_global_filter_enabled=false`). Those 18 are
   analytically dark already; the new fields on post-deploy rows are immune to
   that bug since the gating fix shipped Apr 30.

3. **Aborted rows show pnl=0, peak=0, trough=0 by construction** — they never
   opened. Avg P&L%, Avg Peak%, Total$ all show 0 in the table. That's not
   missing data; it's the correct value ("we don't know what would have
   happened, and the trade did not affect the account"). Don't interpret these
   zeros as "Aborted entries are break-even" — they didn't trade.

4. **Aborted row durations are wait time, not hold time.** Reading "Aborted L
   avg duration 00:00:40" means "average maker wait was 40s before the abort,"
   not "average position lasted 40s." This is intentional — wait time is what
   matters for diagnosing the timeout — but worth noting if future-Claude
   confuses the two.

### The "Tier 3" follow-up that was deliberately deferred

A 60-min post-expiry counterfactual ("what did price do in the next 60 min after
the abort fired?") would tell us whether aborted entries would have made or lost
money. That requires either a background task watching expired entries for 60min
or a backfill that re-fetches OHLCV. **Not free engineering.** Skipped for now —
if Tier 1 (entry-condition comparison) shows aborts cleanly match the Loser
profile, we don't need Tier 3. Revisit only if Tier 1 results are ambiguous.

### Files changed

- `services/trading_engine.py` — `_record_signal_expired_order` signature +
  body (~80 lines added); `_try_maker_entry` and `_simulate_maker_entry_paper`
  return `wait_seconds`; both `open_position` call sites forward indicators +
  wait
- `main.py` — `entry_conditions_by_outcome` builder accepts SIGNAL_EXPIRED
  orders as Aborted bucket; `_compute_signal_expired_breakdown` adds wait-time
  median/p90/max
- `templates/index.html` — Entry Conditions by Outcome JS renderer handles
  Aborted outcome (amber color, no SL Profile); Signal Expired Breakdown table
  + JS adds 3 wait columns; both text-export sites updated for both tables

### Why this entry exists in CLAUDE.md

To follow up at the 200-trade checkpoint with the correct analysis question
("does Aborted match Loser or Winner profile?") and the correct decision rule
(see table above). Without this section future-Claude would treat the new
Aborted rows as another mysterious data point rather than a structured test of
re-validation correctness.

## May 2, 2026 — Phase 1d-ExitTest plan (RSI handoff at high TP levels — code shipped INERT)

### Why this exists

Phase 1c-Explore is at ~156 trades. The May 2 67-trade and 156-trade reports both show the same headline pattern: **the bot is cutting winners early, especially BEARISH winners with big tails.** From the May 2 BULLISH+BEARISH split:

| Bucket | N | Current close | Post-peak missed | Severity |
|---|---|---|---|---|
| BULLISH TRAILING_STOP L2 | 24 | +0.50% | +0.42% above exit | Mild |
| BEARISH TRAILING_STOP L2 | 10 | +0.53% | +0.87% above exit | Severe |
| BULLISH FL_TRAILING_STOP L2 | 4 | +0.40% | +0.81% above exit | N too small |

For the same trades, the post-exit watcher records when 2-drop RSI **would have** fired (already in `post_exit_rsi_exit_pnl`, since Apr 18 instrumentation):

| Bucket | N | Current close | RSI counterfactual | Improvement |
|---|---|---|---|---|
| BULLISH TRAILING_STOP L2 | 24 | +0.4971% | +0.5076% | +1bp (marginal) |
| BEARISH TRAILING_STOP L2 | 10 | +0.5275% | +0.8463% | **+32bp (strong)** |
| BULLISH FL_TRAILING_STOP L2 | 4 | +0.40% | +0.60% | +20bp |

**Hypothesis:** past today's L2 promotion threshold (peak ≥ 0.50% with trend continuation), trailing-stop pullback is exiting trades before momentum genuinely reverses. RSI exhaustion catches the actual reversal point. BEARISH side benefits significantly; BULLISH side marginally.

The user proposed and locked the test design after a long iteration on Rules A-G grid (see prior conversation logs / git history): **lower TP_min from 0.50 to 0.25, keep pullback at 0.20, and at the new L3 promotion (= today's L2 = peak ≥ 0.50% with trend) switch from trailing to RSI exit.**

The 0.25 TP value is intentional: with `tp_min = 0.25`, the level math gives:
- new L1 promotion = peak ≥ 0.25% (catches some currently-failing Positive-No-BE trades that peaked 0.25-0.50% then went to SL)
- new L2 promotion = peak ≥ 0.50% with trend (this is just an intermediate level)
- new L3 promotion = peak ≥ 0.75% with trend — **but the *crossing* into L3 territory happens at peak ≥ 0.50%**, which is today's L2 boundary

So setting `rsi_handoff_level = 3` means RSI takes over at exactly today's L2 boundary, with an expanded trailing-armed band in the 0.25-0.50% peak zone for catching some currently-failing trades.

### What was built (May 2)

Code ships **INERT** — feature is wired through but disabled by default. User flips two UI toggles to activate when ready to test live.

**Two new config fields (`config.py`, top-level on `thresholds`):**
- `rsi_handoff_active: bool = False` — master toggle
- `rsi_handoff_level: int = 3` — promote-past level at which trailing disables and RSI takes over

**Exit logic changes (no schema migration):**

1. **`services/indicators.py::check_exit_conditions`** — when `rsi_handoff_active` AND `current_tp_level >= rsi_handoff_level`, the trailing-stop pullback block is **skipped entirely** (logged as `[RSI_HANDOFF]`). Live RSI exit fires through the monitor-loop handler. Fail-open: if config read fails for any reason, behaves as before (trailing remains active). Default `rsi_handoff_active=False` means the entire change is a no-op.

2. **`services/trading_engine.py` monitor loop** — new handler runs **before** the existing `rsi_momentum_exit` block. Fires when:
   - `rsi_handoff_active=True`
   - `order.current_tp_level >= rsi_handoff_level`
   - 2-drop RSI sequence confirmed (LONG: `_rsi < _rsi1 < _rsi2`; SHORT: `_rsi > _rsi1 > _rsi2`)
   - Any P&L (no profit-zone gate, unlike the existing rsi_momentum_exit)

   Closes with reason `RSI_HANDOFF_EXIT L{level}` for analytical separation from the existing `RSI_MOMENTUM_EXIT` close reason.

3. **`services/trading_engine.py` realtime trailing block** — same handoff guard added so the realtime trailing check can't race the monitor-loop RSI handler.

**UI controls (`templates/index.html`):**
- New row in the exit-config panel labeled "RSI Handoff (Phase 1d-ExitTest)"
- Toggle for `rsi_handoff_active` (amber color to distinguish from green Enabled toggles — signals "experimental")
- Number input for `rsi_handoff_level` (default 3, range 2-10)
- Description explains the level math (with `tp_min=0.25%`, L3 ≈ today's L2 promotion)
- Load + save handlers wired (`document.getElementById('config-rsi-handoff-active'/...-level')`)
- Text export config dump includes both fields

**Default `trading_config.json` values:**
- `rsi_handoff_active: false`
- `rsi_handoff_level: 3`
- `tp_min` **NOT changed** (stays at 0.50 until user flips the test ON via UI and sets it to 0.25 manually)

### Why it ships INERT

The locked Phase 1c-Explore plan is still in effect at 156 trades — strategic config changes are forbidden mid-batch. Shipping the code with both toggles OFF means:
- No behavior change in production
- Phase 1c-Explore continues uninterrupted
- When user is ready (likely at 200-trade checkpoint), they flip toggles via UI without needing a code deploy
- Easy revert: flip back OFF

### Pre-committed test plan (Phase 1d-ExitTest)

When user activates this feature, the test config and falsification criteria are locked **NOW** (before any data arrives) to prevent post-hoc parameter selection bias:

**Activation config (when user flips ON):**
- `tp_min`: 0.50 → **0.25**
- `pullback_trigger`: 0.20 (unchanged)
- `rsi_handoff_active`: false → **true**
- `rsi_handoff_level`: 3 (default)

**Pre-committed falsification criteria** (over ~100 new trades after activation, separate from Phase 1c-Explore data):

| Outcome | Verdict |
|---|---|
| Combined Avg P&L improves ≥ +8bp/trade vs Phase 1c-Explore baseline | **Win** — keep config |
| Combined Avg P&L within ±5bp of baseline | **Inconclusive** — extend test 100 more trades |
| Combined Avg P&L worsens > 8bp/trade | **Revert** to defaults (toggles OFF, tp_min back to 0.50) |
| BEARISH improves AND BULLISH worsens | **Partial win** — keep RSI handoff for SHORTs only via per-direction config (would require additional code), or revert TP=0.25 only |

Lower threshold (+8bp) than the Apr 30 Winner Exit plan (+10bp) reflects the smaller expected upside of this variant vs straight Rule G.

### Honest caveats locked at design time

1. **Bundled test = harder attribution.** Lowering `tp_min` and enabling RSI handoff change two things at once. If the test fails, we don't know which piece hurt. Cleaner alternatives (sequential one-at-a-time) were considered and rejected by the user in favor of speed.

2. **Sample size for BEARISH side will be tight.** ~10 BEARISH L2+ trades expected per 100-trade batch. Combined-direction view will dominate; per-direction analysis stays exploratory.

3. **Sequencing bias on the climb.** Trades that today succeed past today's L2 (peak ≥ 0.50% with trend continuation) might exit earlier under the new config if they had a 0.20% intra-climb retrace from the new arming point at peak 0.25%. We can't see this from the current data — it's a known unknown.

4. **The `post_exit_rsi_exit_pnl` data the hypothesis is built on is an UPPER BOUND.** It records the first 2-drop RSI **after our trailing exit**. If a 2-drop RSI fired during the trade (between L2 promotion and trailing exit), we don't see it. Real Rule F effect could be weaker than the +32bp estimate. The shadow-tracker idea proposed earlier (intra-trade RSI tracking) was deferred — fixable in a future iteration if results are ambiguous.

5. **Pooling rule.** Phase 1d-ExitTest data is **separate** from Phase 1c-Explore. Don't pool raw trades. Compare aggregates using Avg P&L %.

### How to activate (operator instructions)

When the user is ready to start Phase 1d-ExitTest:

1. Open the config UI
2. In the exit panel, find the "RSI Handoff (Phase 1d-ExitTest)" row
3. Toggle the amber switch to ON
4. Verify "Handoff at L≥" is set to 3 (or pick another level for variant testing)
5. Find the `tp_min` field in confidence-level config and change from 0.50 to 0.25 for both VERY_STRONG and STRONG_BUY tiers
6. Save the config — change takes effect on the next trade open

Counter starts at the next trade. Target ~100 trades for first decision checkpoint.

### Files changed (May 2)

- `config.py` — `rsi_handoff_active`, `rsi_handoff_level` fields
- `trading_config.json` — defaults (false, 3)
- `services/indicators.py::check_exit_conditions` — handoff suppresses trailing block (~15 lines added)
- `services/trading_engine.py` — new RSI_HANDOFF_EXIT handler in monitor loop (~30 lines), realtime trailing guard (~10 lines)
- `templates/index.html` — UI panel + load/save handlers + text export
- This CLAUDE.md amendment

### Why this entry exists in CLAUDE.md

When the user activates the test, the falsification criteria above must be the gates — not whatever feels right at decision time. Anchoring risk is real after looking at partial-batch numbers. This section is the locked rule the future-Claude (and user) is held to.

If at the 100-trade checkpoint the test produces ambiguous results (within ±5bp of baseline), the temptation will be to "just tweak the level to 4 and re-run." That's the discipline-erosion failure mode. The decision rule says: **inconclusive = extend 100 more trades**, not "tune and re-run." Hold the line.

If the test cleanly fails (revert verdict), the next iteration is informed but we have to accept the lost batch as the cost of running a real test rather than analyzing forever.

## May 2, 2026 — Three new max-guard filters (split + new), feature ships permissive

Three new entry-filter parameters added today, all per-direction (LONG/SHORT). All
**ship permissive** — defaults at deploy match or exceed any value seen in current
data, so behavior is unchanged. User tightens later via UI when ready, post 200-trade
checkpoint per the locked Phase 1c-Explore plan.

| Filter | Field name | Default | What it blocks |
|---|---|---|---|
| EMA5-EMA8 Gap Max — split | `ema_gap_5_8_max_long` / `ema_gap_5_8_max_short` | 0.35 / 0.35 | Same gate as before, just direction-aware |
| Pair EMA20 Slope Max (NEW) | `momentum_ema20_slope_max_long` / `momentum_ema20_slope_max_short` | 0.40 / 0.40 | Over-extended pair trend (`abs(ema20_slope) > max`) |
| BTC EMA20 Slope Max (NEW) | `btc_ema20_slope_max_long` / `btc_ema20_slope_max_short` | 0.35 / 0.35 | Late-cycle BTC entries (`abs(btc_slope) > max`) |

### Why these specifically

Looking at the 163-trade May 2 BULLISH report:
- **BTC slope >0.10% buckets are losers**: `0.10-0.12%` -0.37%, `0.16-0.18%` -0.46%, `0.20-0.25%` -1.00%. Capping at 0.35% is permissive (catches almost nothing in current data) but provides the lever to tighten as evidence accumulates.
- **Pair slope ≥0.14% LONGs lose**: `0.14-0.16%` 0% WR -0.67%, `0.18-0.20%` 0% WR -1.17%. Cap at 0.40% is again permissive but creates the structure.
- **EMA5-EMA8 gap split**: structural — splitting the existing single max into per-direction lets you tune asymmetrically when data shows asymmetric patterns. Today no asymmetric tuning, just refactor.

### Auto-migration

Existing configs with the old single `ema_gap_5_8_max` field auto-fallback in three
places:
- `services/indicators.py::get_signal` LONG block: `getattr(th, 'ema_gap_5_8_max_long', 0) or getattr(th, 'ema_gap_5_8_max', 0)`
- Same pattern for SHORT block
- UI load handler: `_legacy_gap58_max` fallback when split fields are undefined

So existing trading_config.json files with the old field name keep working. New default
trading_config.json now writes both new fields explicitly + leaves the legacy field for back-compat.

### Implementation sites

**Config schema (`config.py`):**
- `ema_gap_5_8_max_long: float = 0.0` (NEW, default disabled — actual default in trading_config.json is 0.35)
- `ema_gap_5_8_max_short: float = 0.0` (same pattern)
- `momentum_ema20_slope_max_long: float = 0.0` (NEW)
- `momentum_ema20_slope_max_short: float = 0.0` (NEW)
- `btc_ema20_slope_max_long: float = 0.0` (NEW)
- `btc_ema20_slope_max_short: float = 0.0` (NEW)

**Filter checks:**
- EMA5-EMA8 max: `services/indicators.py::get_signal` lines ~306 (LONG) and ~349 (SHORT) — replaces the legacy single-field lookup with direction-aware lookup + fallback
- Pair EMA20 slope max: `services/trading_engine.py` post-`get_signal` gate, computes slope locally from `indicators.get('ema20')` and `indicators.get('ema20_prev3')`, blocks via `signal = "NO_TRADE"` with `[PAIR_SLOPE_MAX_GATE]` log
- BTC EMA20 slope max: same location, uses already-computed `btc_ema20_slope_pct`, logs `[BTC_SLOPE_MAX_GATE]`

**UI changes (`templates/index.html`):**
- VERY_STRONG row of confidence-thresholds table: single Max input replaced with L Max + S Max inputs
- STRONG_BUY mirror row: single Max display replaced with L Max + S Max displays (with corresponding sync handlers)
- Pair EMA20 Slope filter card: added LONG max + SHORT max inputs alongside existing min inputs
- BTC Independent Filters: added Max BTC EMA20 Slope L + S inputs alongside existing Min inputs
- Load handlers (~line 9040): three new pairs of fields populated from config with legacy fallback for `ema_gap_5_8_max`
- Save handlers (~line 9342): old single field replaced with the six new fields
- Mirror sync (`config-ema-gap-max-long-mirror` / `-short-mirror`): replace single mirror with split
- Text export (~line 4682): single "EMA Gap Max" line replaced with "L Max / S Max" + two new lines for pair slope max and BTC slope max

### Phase 1c-Explore lock — these filters are permissive only

Per the locked Phase 1c-Explore plan, **strategic config tightening mid-batch is forbidden**. These ship at values that don't filter any real-world entries seen so far:
- BTC slope max 0.35% catches ~3-4 historical losing entries (mostly already losing via other gates)
- Pair slope max 0.40% catches almost nothing
- EMA5-8 max 0.35% L/S preserves current behavior exactly

The fields exist as **structural infrastructure** — to be tuned at the 200-trade checkpoint
based on validated data. Tuning them down NOW would violate the lock. Wait for the checkpoint.

### What to look for at 200-trade checkpoint

When the locked Phase 1c-Explore decision happens:
1. **Pair slope >0.14% LONG bucket** at 200 trades: still negative? If yes, recommend `momentum_ema20_slope_max_long: 0.14` as filter promotion.
2. **BTC slope 0.10-0.18% buckets at 200 trades**: still negative? If yes, recommend `btc_ema20_slope_max_*: 0.10` (per direction depending on data).
3. **EMA5-8 max asymmetry**: if SHORT 0.18-0.20% bucket is winner (it is in the 38-trade BEARISH sample) but LONG 0.10-0.12% is loser, the split lets you keep SHORT permissive (0.35) and tighten LONG (e.g., 0.10).

### Why this entry exists in CLAUDE.md

Future-Claude will see new max fields in config and might think they're already
filtering. They're not — they're permissive starting values waiting for tuning. Without
this section, future analysis might mistake these as active filters and miss the
opportunity to actually use them at the 200-trade checkpoint.

The Phase 1c-Explore lock applies. Don't tighten until 200 trades + validated data. Hold the line.

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

## May 3, 2026 — Cross-sample SHORT findings to validate at 200-trade Phase 1c-Explore checkpoint

Two findings emerged from a cross-sample exercise on the partial 171-trade Phase 1c-Explore data (126L BULLISH + 45S BEARISH), comparing against historical RSI×ADX and BTC-RSI×BTC-ADX cross-tabs from Apr 09 (97tr), Apr 12 (53tr), and Apr 13 (117tr). **Locked here as pre-committed cells to re-validate at the 200-trade checkpoint** — if they survive the 5th sample with their direction intact, they become candidates for investment-doubling in the Phase 2 → Phase 3 transition. If they fail, drop them.

### Methodology used (the user's proposed approach, validated)

1. Pull RSI×ADX and BTC-RSI×BTC-ADX cells from each historical report at native bucket boundaries.
2. Where buckets differ across samples (Apr 13 uses fine ADX bins 15-18/18-22/22-25, Apr 09/12 use coarse 15-20/20-25/25-30), collapse the finer-bucket sample down to the coarser scheme using a 50/50 split for the overlapping bin (e.g. "18-22" → half to 15-20, half to 20-25). This introduces ~10-15% rounding error per cell, acceptable for cross-sample directional reads.
3. Cells qualify as cross-sample winners only when **direction is consistent across ≥4 samples**, **N ≥ 5 per sample with combined N ≥ 30**, and the most recent sample replicates (not just historical agreement that has since decayed).
4. **Critical anti-overfit lesson learned during this exercise:** `SHORT 30-35 × 25-30` was the historical 3-sample winner (84-82-68% WR across Apr 09 + Apr 12 + Apr 13, total N=58). Sizing up on historical-only data without checking current sample would have been wrong — current sample shows 50% WR, the cell has decayed. Cross-sample confirmation **including the current/most-recent sample** protects against this. Historical-only is not sufficient.

### Finding #1 — Pair-level: SHORT RSI 20-30 × ADX 25-30 (5-sample confirmed)

| Sample | N | WR | Avg P&L % |
|---|---|---|---|
| Apr 09 (97tr post-tightening) | 8 | 75% | −0.00% |
| Apr 12 (53tr partial) | 9 | 67% | +0.30% |
| Apr 13 (117tr full) | 10 | 70% | +0.32% |
| Phase 1c-Explore @ ~150 trades | 12 | 75% | +0.11% |
| Phase 1c-Explore @ 171 trades | 13 | 77% | +0.13% |
| **Pooled** | **48** | **~73%** | **~+0.18% weighted** |

5 samples, all WR ≥ 67%, 4 of 5 positive Avg %, the one zero is essentially flat not negative. Cleanest cross-sample pair-level cell in the dataset. Note current report uses fine ADX buckets 25-28 (N=10) + 28-30 (N=3) which collapse to historical 25-30; computation reflects that.

**Supporting cell** — `SHORT RSI 20-30 × ADX 30-35`: positive in all 4 samples that had data (Apr 09 +0.02%, Apr 12 +1.97% N=1 ignore, Apr 13 +0.17%, Current +0.35%). Directional consistency confirmed but cell-level N is thinner (~17 pooled excluding tiny cells). Treat as a same-family signal: "SHORT with oversold pair RSI <30 in moderate-to-strong ADX 25-35."

### Finding #2 — BTC-level: SHORT BTC RSI 25-30 (the pre-committed S-P1 PREMIUM ZONE, now 5-sample confirmed)

The pre-committed S-P1 rule (BTC RSI <30 × BTC ADX 20-25) was validated in the Apr 17 cross-tab audit at 75% pooled WR on N=12. Phase 1c-Explore data extends this pattern to the BTC RSI 25-30 single-dimension cell (any BTC ADX):

| Sample | BTC RSI 25-30 SHORT cell | WR | Avg % |
|---|---|---|---|
| Phase 1c-Explore @ 38 SHORTs | 6 trades | 100% | +0.45% |
| Phase 1c-Explore @ 45 SHORTs | **9 trades** | **100%** | **+0.43%** |

This sub-cell is now the strongest BTC-level signal in the entire dataset — 9 of 9 winners in current sample. Combines with historical 4-sample evidence on BTC RSI <30 (S-P1 cell) to suggest "**BTC RSI 25-30 is a PREMIUM-tier macro condition for SHORTs regardless of BTC ADX**".

Per-cell breakdown of BTC RSI 25-30 in current sample:
- × BTC ADX 20-25 (4t, 100% WR, +0.49%) — extension of S-P1
- × BTC ADX 25-30 (2t, 100% WR, +0.37%)
- × BTC ADX 30-35 (1t, 100% WR, +0.32%)
- × BTC ADX 35+ (2t, 100% WR, +0.44%)

The pattern is uniform across BTC ADX magnitude when BTC RSI is in 25-30. This is materially stronger than the original S-P1 (which was conditional on BTC ADX 20-25).

### Counter-finding — what NOT to size up

`SHORT 30-35 × 25-30` (the pre-committed S-P2 PREMIUM ZONE) — already weakened in the Apr 17 audit (pool WR dropped from pre-commit 83% → 57%) and current Phase 1c-Explore data shows it still losing (1 trade in current sample at 0% WR, −1.01%). The Apr 17 audit verdict holds: **S-P2 should NOT be promoted, has decayed across recent samples.** Sizing up here would be the trap that historical-only analysis would set.

### What to do at the 200-trade Phase 1c-Explore checkpoint

For each of the two findings above, repeat the cross-sample exercise with the full 200-trade dataset:

**Validation gates for promotion to "size-up candidate" in Phase 2/3:**

1. `SHORT 20-30 × 25-30` qualifies for 2x sizing if:
   - Phase 1c-Explore final N ≥ 15 in this cell
   - WR remains ≥ 65% in the final 200-trade aggregate
   - Avg P&L % remains ≥ +0.10%
   - Combined 5-sample WR stays ≥ 70% on pooled N ≥ 50
2. `SHORT BTC RSI 25-30` qualifies for 2x sizing if:
   - Phase 1c-Explore final N ≥ 12 in this cell
   - WR remains ≥ 80%
   - Avg P&L % remains ≥ +0.30%
   - The uniformity across BTC ADX magnitude holds (no single BTC ADX bucket within this cell drops below 50% WR with N ≥ 3)

**If both findings pass their gates:** the size-up candidate set for Phase 3 includes both — with the practical understanding that they're partly overlapping (a SHORT that satisfies pair RSI 20-30 × ADX 25-30 may often also be in BTC RSI 25-30). The actual leveraged-bucket logic should AND the cells (cell qualifies for 2x only when BOTH pair-level and BTC-level conditions hold), not OR them, to keep the size-up surface narrow and high-conviction.

**If only one passes:** ship that one alone. Don't compromise the other to keep symmetry.

**If neither passes:** the cells decayed in the final batch, exactly like S-P2 did from Apr 13 → Apr 17. Don't ship sizing changes — accept the negative result and stay at 1x.

### Live-validation requirement (paper-data caveat)

Both findings come from PAPER mode. Per the Apr 18 quant methodology note in CLAUDE.md, filter-layer findings (which these are — RSI/ADX cell selection is pure indicator math) are paper-OK. **However, sizing up doubles real-money exposure on these cells when we eventually go live.** Before doubling in live mode, run a separate live batch (~50 trades hitting these cells) at 1x to confirm fill mechanics don't bias the cell-level WR downward (MAKER over-fill in paper would inflate WR for cells that frequently hit MAKER entries). Only after live 1x confirmation does the 2x sizing ship to live.

### Why this entry exists in CLAUDE.md

Two reasons:
1. **Pre-committed validation gates locked before the 200-trade data lands.** Same discipline as the existing Apr 28 Phase 1c-Explore plan: rules first, data second. Prevents post-hoc bar-lowering when the actual numbers come in slightly under threshold.
2. **The S-P2 cautionary lesson is preserved here.** The exact same methodology that produces these two findings also catches the S-P2 decay. If at the 200-trade checkpoint the user is tempted to size up on a different cell that "looks like a winner historically," the S-P2 example in this section is the answer for why historical-only is not enough.

## May 3, 2026 — Pair blacklist candidates for 200-trade Phase 1c-Explore review

Four pairs flagged for blacklist evaluation at the 200-trade checkpoint based on a cross-report 0% WR scan run May 3 against the partial 188-trade sample. **No action taken yet** — locked here for re-validation when the full Phase 1c-Explore batch completes.

### Candidates and current evidence

| Pair | Direction | Evidence |
|---|---|---|
| **HYPEUSDT** | SHORT (and LONG concerning) | SHORT: 6 trades / 0 wins across 3 reports (Apr 06 + Apr 12 + Apr 13). Strongest cross-sample 0% WR signal in dataset. LONG in current sample: 9 trades, 22% WR, −$6.26 (worst $-loss pair this batch). Already flagged in CLAUDE.md Apr 17 as known-risky older pair (322 days old, slips through the 180d new-listing filter). |
| **DOGEUSDT** | SHORT | 4 trades / 0 wins across 2 reports (Apr 13 + current). Different configs, same outcome. Meets 2-sample confirmation bar. |
| **RIVERUSDT** | LONG | 4 trades / 0 wins across 2 reports (Apr 06 + current). Apr 06 was a very different config era (20-30x leverage), but pattern holds despite that gap — actually stronger structural signal because the loss survives across config eras. |
| **1000LUNCUSDT** | LONG | 2 trades / 0 wins / −1.01% Avg / −$4.02 — but **1-sample only** (current Phase 1c-Explore). Below the 2-sample anti-overfit bar (CLAUDE.md rule #4). User decision was to flag for monitoring rather than wait separately. Worst single-pair $-loss after RIVERUSDT in current sample. |

### Decision rule at 200-trade checkpoint

For each of the 4 candidates, recompute the per-pair WR across the full 200-trade sample, then apply this gate:

| Outcome at 200 trades | Verdict |
|---|---|
| Pair shows ≥6 trades total across 2+ reports AND WR ≤25% | **Blacklist** — add to `pair_blacklist` config |
| Pair shows ≥4 trades AND WR 0% (any sample size) | **Blacklist** |
| Pair shows ≥4 trades but WR climbs to 30-50% on the new data alone | **Hold for 400-trade checkpoint** — single-batch reversion possible |
| Pair shows fresh wins on N≥3 in new batch (e.g., 1000LUNC wins 2 of 3 next trades) | **Drop from candidate list** — was 1-sample noise |
| Pair stops firing entirely (no new trades) | **Hold** — insufficient data, re-evaluate at 400 trades |

### Special handling for HYPEUSDT

HYPEUSDT has **directional asymmetry**: SHORT side is the cross-sample 0% WR finding, LONG side has a separate but related "weak performance" signal in current sample. Two options at the 200-trade decision:

1. **Full pair blacklist** (block both directions). Safer, simpler, costs more opportunity if LONG side recovers.
2. **Direction-specific blacklist** (block only SHORT side). Requires code work — current `pair_blacklist` is binary per-pair, not per-direction. Adding direction-specific would be a small enhancement (~10 lines in the entry filter chain in `services/trading_engine.py`).

If 200-trade SHORT data on HYPE remains 0% WR, ship Option 1 immediately. Option 2 only if LONG side shows ≥50% WR on N≥5 in the fresh data — meaning there's a real LONG edge worth preserving.

### Why this entry exists in CLAUDE.md

To anchor four specific candidate decisions at the 200-trade checkpoint with pre-committed gates, preventing post-hoc bar-lowering when the actual numbers come in. Same discipline as the May 3 cross-sample SHORT findings entry directly above. The 1000LUNCUSDT inclusion is a deliberate exception to anti-overfit rule #4 (1-sample) — flagged here so the rationale is explicit and the bar at 200 trades enforces the 2-sample requirement before action.

## May 3, 2026 — Phase 3 Position Multiplier Mechanism (DESIGN, post-200-trade bonus)

This entry documents the **design plan** for the per-cell position multiplier mechanism — the Phase 3 sizing layer described in the existing "Three-Phase Plan to Make the Bot Profitable" section. Reviewed and locked here as a post-200-trade bonus item: implementation should be considered AFTER the 200-trade checkpoint completes AND all filter-level reviews (BTC RSI, EMA50 alignment, pair blacklist candidates, exit optimizations) are decided. Multipliers are the LAST mechanism to ship in the Phase 1c-Explore → Phase 2 → Phase 3 progression — they amplify whatever edge survives filter validation. Shipping multipliers before filter validation amplifies whatever's left in the system, edge or noise.

### What this mechanism does

Allows per-cell position sizing based on RSI × ADX bucket. A trade matching a configured cell (e.g., LONG RSI 60-65 × ADX 18-22) gets its position scaled by a multiplier (e.g., 2.0×). Cells not configured default to 1.0× (no change). This is the natural extension of the existing pair-level RSI×ADX cross-filter — same dimensions, additive size effect instead of binary block.

### Architecture (4 layers)

**Layer 1 — Config schema** (in `config.py` on `SignalThresholds`):
```
rsi_adx_multiplier_long: str = ""    # format: "RSI_band:ADX_band:multiplier,..."
rsi_adx_multiplier_short: str = ""   # same format
rsi_adx_multiplier_target: str = "investment"  # "investment" or "leverage"
```
Example value: `"60-65:18-22:2.0,55-60:22-25:1.5"` — two rules. Empty string = mechanism inert.

**Layer 2 — UI panel** in `templates/index.html`, mirrors the existing pair-level RSI×ADX Cross-Filter UI:
- Toggle: "Apply multiplier to" → Investment / Leverage radio
- Two tables (LONG / SHORT), each with rows: `RSI band dropdown | ADX band dropdown | multiplier input | remove button`
- "+ Add rule" button per direction
- Multiplier input clamped to [0.25, 5.0]
- Visual indicator: rows with multiplier ≥1.5 highlighted amber, ≥2.5 highlighted red
- RSI/ADX band dropdowns MUST use the same boundaries as the analytics report cells (50-55, 55-60, ..., 70+ for RSI; 15-18, 18-22, 22-25, 25-30, 30-33, 33+ for ADX) so cells in the multiplier UI map 1:1 to cells in performance reports

**Layer 3 — Engine logic** in `services/trading_engine.py::open_position`, ~30 lines, after all filters pass and confidence is assigned:
```python
base_investment, base_leverage = self._calculate_position_size(...)
cell_multiplier, cell_id = self._lookup_rsi_adx_multiplier(direction, rsi, adx)
cell_multiplier = min(cell_multiplier, 3.0)  # HARD CAP — non-negotiable safety guard

if multiplier_target == 'investment':
    investment = base_investment * cell_multiplier
elif multiplier_target == 'leverage':
    leverage = int(base_leverage * cell_multiplier)

order.cell_multiplier = cell_multiplier
order.cell_multiplier_source = cell_id  # e.g., "LONG_60-65_18-22"
```

**Layer 4 — Tracking** (2 new columns on Order, no schema migration risk):
- `cell_multiplier: Float, default=1.0` — what was actually applied
- `cell_multiplier_source: String(40), nullable` — which cell rule fired, NULL if default 1.0

### Tracking table — Multiplier Cell Performance

New report section (LONG + SHORT separated). Critical columns:

| Cell | Multi | N | WR | Avg P&L% | Total$ | Δ vs 1x baseline | Verdict |

The "Δ vs 1x baseline" column is the diagnostic that matters most:
- Computed as actual $ minus simulated 1x $ (`Total$ × (1 - 1/multi)` for the boost contribution)
- Tells you if the boost actually helped or just amplified noise

Verdict logic (automated):
- ★ WORKING: multiplied $ ≥ 1.7× the 1x simulated $
- ✓ Marginal: 1.0×–1.7× scaling
- ⚠ DRAG: <1.0× (boost reduced effective edge — variance widened too much)
- ✗ HARMFUL: net negative under multiplier — revert immediately

### Critical design decisions (locked here, reasoned-through)

**1. Default to investment-size multiplier, not leverage.** Reasons:
- Investment doesn't compound with the existing per-tier leverage setting. Leverage does (VERY_STRONG at 2x leverage × 2x cell multiplier = 4x effective exposure — confusing and dangerous).
- Investment has simpler mental model ("I'm doubling this trade")
- Both produce identical P&L in paper mode; difference is only margin efficiency in live mode
- At SCALPARS' SL distance (-0.9%), liquidation isn't a concern at any reasonable multiplier on either mechanism

User can switch to leverage via the toggle if they specifically want capital efficiency, but default is investment.

**2. Hard cap at 3.0× regardless of UI input.** Even if user enters 5.0 in a cell, engine clamps to 3.0. Single-line safety check in `open_position`. Non-negotiable. This is insurance against typos and against escalating multipliers in a phase of mistaken confidence.

**3. Manual cell configuration, NOT automatic adaptive sizing.** This was debated and rejected for Phase 3 scope. Documented reasoning:

Adaptive sizing (bot watches WR/PF, auto-adjusts multipliers) sounds like an upgrade but is structurally wrong at retail scale. Why:
- At our N (5-50 trades per cell), rolling WR is dominated by noise, not edge. An adaptive system reads variance and reinforces it.
- The S-P2 cell decay we identified (May 3 cross-sample exercise — 84% → 82% → 68% → 57% → 0% across 5 samples) would have caused an adaptive system to boost during the winning phase and reduce too late after damage was taken.
- Adaptive sizing requires ≥10,000 trades per cell + multiple uncorrelated strategies + professional risk infrastructure to be mathematically stable. We have none of those.
- It removes operator judgment from the loop — the exact discipline that's been working in CLAUDE.md.

Manual cell configuration is structurally correct for our scale. Static rules + cross-sample validation gates + multi-sample confirmation. The boring answer is the right answer.

**4. Decay-alert mechanism (compromise — surface decay without auto-adjust).** When a cell with multiplier >1.0 has accumulated ≥10 trades since deploy AND current WR has dropped >15pp below the baseline that justified the multiplier, the UI flashes a non-blocking warning on that cell's row: "⚠ REVIEW: Cell X is at WR 52% (baseline was 73%). Consider reducing multiplier."

The bot does NOT auto-adjust. It alerts. Human reads, decides whether to act. ~20 lines of UI logic. Best of both worlds: keeps human in the loop, surfaces decay early, zero feedback-loop risk.

This is a Layer 5 / phase-3.5 add-on after the base manual mechanism is shipped and has been live for ~50 trades.

### Implementation order

If/when this ships:
1. **Commit 1**: Schema + DB columns + engine logic with `rsi_adx_multiplier_*` defaulting to empty strings (= no behavior change). Mechanism inert by default.
2. **Commit 2**: UI panel (still defaults to no rules).
3. **Commit 3**: Multiplier Cell Performance reporting table.
4. **THEN**: User fills in cells based on validated findings from the 200-trade checkpoint, per the locked rules in the May 3 cross-sample SHORT findings entry above.
5. **~50 trades after first multiplier ships**: Add decay-alert mechanism (Layer 5).

This staged approach ships infrastructure with zero behavior risk, then activates multipliers only on cells that pass the cross-sample validation gates.

### Pre-conditions for shipping (do NOT ship before these)

1. **200-trade Phase 1c-Explore checkpoint complete** with full cross-sample analysis on RSI×ADX, BTC RSI×BTC ADX, and Tier 1 dimensions (EMA50 alignment, DI spread, etc.)
2. **At least one cell from the May 3 saved findings has passed its locked validation gate** (SHORT 20-30 × 25-30 OR SHORT BTC RSI 25-30) — if neither passed, multiplier mechanism has nothing to multiply, defer until Phase 2 produces a validated cell.
3. **Pair blacklist decisions resolved** for HYPEUSDT, DOGEUSDT, RIVERUSDT, 1000LUNCUSDT (May 3 candidates entry) — sizing up while a known-bad pair is still active compounds the wrong way.
4. **Filter-layer optimizations decided** — exit changes (RSI Handoff, BE layer if any), entry filter additions from EMA50/funding/DI cross-tabs.

In summary: **filters first, sizing last.** Multipliers amplify whatever edge survives the filter pass. Wrong sizing on right filters loses less than right sizing on wrong filters.

### Files that would change

| Layer | File | Estimate |
|---|---|---|
| Config schema | `config.py` | 3 fields on SignalThresholds |
| Defaults | `trading_config.json` | Empty strings + target="investment" |
| Engine | `services/trading_engine.py::open_position` | ~30 lines |
| DB | `models.py` + `database.py` auto-migrate | 2 nullable columns on Order |
| API | `main.py::_compute_performance` | New `multiplier_cell_performance` payload section + lookup helper |
| UI | `templates/index.html` | ~200 lines: panel + load/save + tracking table renderer |
| Text export | `templates/index.html` (both export sites) | New tracking table rows |

Total: ~150-250 lines of new code across the stack. No schema migration risk (only adds columns, doesn't modify existing).

### Why this entry exists in CLAUDE.md

To preserve the design + the reasoning behind two specific decisions (investment vs leverage, manual vs adaptive) so they don't have to be re-debated when implementation time arrives. Also to lock the pre-conditions explicitly so the mechanism doesn't ship prematurely. The "filters first, sizing last" rule is the most important takeaway — multipliers without validated cells are a faster path to losing money than helpful sizing.

## May 3, 2026 — Decision to revert Amendments #6 and #8 (40s→20s timeout, 2→1 tick offset) at 200-trade checkpoint

### What's being reverted

| Amendment | Field | Current (Phase 1c-Explore) | Revert target | Rationale source |
|---|---|---|---|---|
| #6 (Apr 18) | `maker_timeout_seconds` | 40 | **20** | Hold-Time Expectancy + Aborted-population analysis (this entry) |
| #8 (Apr 18) | `maker_offset_ticks` | 2 | **1** | Bundled with #6; offset depth is independent question but reverting both restores the pre-Apr-18 fill-mechanics baseline cleanly |

### Why — the analytical chain that produced this decision

At 196 trades in the Phase 1c-Explore batch, three independent observations converged:

**1. Aborted population sits in over-extended entry conditions on both sides.**
Aborted L (89): AvgRSI 63.8, BTCRSI 66.4, BTCADX 29.0 — late-cycle uptrend.
Aborted S (14): AvgRSI 26.0, BTCRSI 22.9, Funding +0.046% — squeeze-prone deep oversold.
Initially read as "re-validator catching the right population to skip."

**2. Signal Expired Breakdown shows every abort fires at exactly t=40.0s.**
MedWait, p90Wait, MaxWait all = 40.0s. Zero aborts fire earlier. Means signal flips during the partial-bar re-validation at the timeout boundary — the 20s window almost never produces a flip on its own (otherwise we'd see early aborts).

**3. Hold-Time Expectancy table: winners peak/close FASTER than losers.**
This is the load-bearing observation. If winners are time-sensitive and move fast off the signal, then a 40s wait burns through the front of the move. By the time re-validation runs, the fast-winner's initial leg has already played out, the partial bar shows momentum cooling, signal flips, abort fires → **we systematically miss the fast-winner population**.

The reframing that follows from #3:
- "Signal Flipped" ≠ re-validator catching bad entries
- "Signal Flipped" = re-validator catching trades whose move already happened during our wait
- The 40s timeout was added (Amendment #6) on the hypothesis that more pullback-entry maker fills would improve overall WR
- For a fast-winner strategy, that hypothesis was structurally wrong — the extra 20s isn't capturing pullback fills, it's burning the entry window

The fact that aborts cluster in over-extended conditions (#1) is consistent with this: those are exactly the conditions where 40s of price action is enough to reverse the local micro-trend. Late-cycle uptrend at RSI 63 + 40s of mean reversion = signal flips. Deep oversold + 40s of squeeze rally = signal flips.

### What earlier analysis got wrong

The original "Aborts are net-protective" read (this conversation, earlier turn) was built on an unverified assumption that Losers L profile resembled Aborted L profile. **That comparison was not actually pulled from the table.** Once the user flagged the Hold-Time Expectancy pattern, the protective-effect interpretation no longer holds. Aborts are not protective — they're missed entries on the fastest-moving population.

Methodological lesson (locked here): when claiming "X profile resembles Y profile" in a cross-bucket comparison, **pull the actual numbers from the table.** Asserting similarity from memory or impression is the failure mode that produced this round-trip.

### Paper-mode caveat (CLAUDE.md Apr 18)

Amendments #6 and #8 are **fill-mechanics changes**, which paper data is biased on (paper's `_simulate_maker_entry_paper` over-fills relative to live orderbook competition). Strict reading of the Apr 18 rule says these amendments cannot be evaluated in paper.

However, the evidence chain above does NOT rest on fill-mechanics metrics:
- "Winners peak fast" is pure indicator-math (paper-OK)
- "Aborts cluster at over-extended RSI/BTC RSI" is filter-layer (paper-OK)
- "All aborts fire at t=40.0s" is timing observation (paper-OK)

The decision to revert rests on filter-layer + timing evidence, not on paper MAKER-vs-TAKER WR claims (which would be paper-biased). So the Apr 18 caveat applies to the *expected post-revert effect* (we can't predict the new MAKER fill rate accurately from paper), not to the *decision logic* itself.

### Decision and timing

**Decision:** Revert `maker_timeout_seconds: 40 → 20` and `maker_offset_ticks: 2 → 1` at the 200-trade Phase 1c-Explore checkpoint.

**Timing:** Ship together with whatever other config changes the 200-trade analysis decides. Per the locked Phase 1c-Explore plan (Apr 28), strategic config changes happen at the checkpoint, not before. We're at 196/200 — wait for the remaining 4.

**Bundling rationale:** #6 and #8 ship together because they were deployed together as a package and the analytical logic above treats them as a fill-mechanics package. Cleaner attribution: revert the package, observe new package's behavior, decide if either piece needs to be reverted again independently.

### Pre-committed validation criteria for the post-revert sample

Once reverted, the next batch (target ~100 trades) is the validation. Locked criteria:

| Outcome | Verdict |
|---|---|
| SIGNAL_EXPIRED rate drops materially (≥ 50% reduction in "Signal Flipped" count per 100 trades) | Revert confirmed working as designed |
| Combined Avg P&L % improves ≥ +5bp/trade vs Phase 1c-Explore | Revert is net positive, lock 20s/1-tick as default |
| Combined Avg P&L % flat (±5bp) | Revert is neutral on edge but reduces abort rate; keep for cleaner data |
| Combined Avg P&L % worsens > 5bp/trade | Aborts WERE net-protective; consider re-extending timeout but with re-validation logic fix first |
| MAKER fill rate drops > 30% (live mode) | The shorter timeout cost too many maker fills; revisit offset (try 1-tick first, 2-tick second) |

### What this revert does NOT do

- Does NOT fix the underlying fragility of `_revalidate_entry_signal` re-running full `get_signal()` on a partial bar. That's a separate Phase 1d candidate (replace with `is_signal_direction_active()` directional check). Reverting timeout reduces how often the fragile path is hit, but doesn't fix the path itself.
- Does NOT change Amendment #7 (signal re-validation infrastructure). The re-validation still fires; it just fires at t=20s instead of t=40s.
- Does NOT touch any entry filter, exit parameter, or BTC/pair-level config. Pure fill-mechanics revert.

### Why this entry exists in CLAUDE.md

To anchor the revert decision and its evidence chain so that at the 200-trade checkpoint the change ships without re-litigation. Also to preserve the methodological lesson (don't assert profile-similarity without pulling numbers) and the Hold-Time Expectancy reframing — both of which inform future analysis discipline.

The Apr 18 paper-mode caveat is acknowledged but the decision is defensible because the evidence chain doesn't rest on fill-mechanics paper data. If the post-revert validation says otherwise, the revert itself gets reverted — that's what the locked criteria above are for.

## May 4, 2026 — Phase 1c-Explore 224-trade checkpoint analysis & LONG-side config changes

### Sample analyzed
224-trade Phase 1c-Explore batch (163 LONG BULLISH + 61 SHORT BEARISH), runtime 5.87 days, paper mode, 1x leverage. Reports archived at:
- `reports/report_2026-05-04_phase1c_explore_224trades.txt` (split analytics)
- `reports/orders_2026-05-04_phase1c_explore_224trades.csv` (raw orders, first batch with full per-trade CSV export)

### LONG side: -$45.24 / -0.14% Avg / 38.65% WR / PF 0.53 — losing
SAME_REGIME LONGs (25 trades): 76% WR, +$9.56 — real edge when BTC regime stable
REGIME_SHIFT LONGs (140 trades): 32.1% WR, -$54.65 — 85% of trades killed by mid-trade BTC regime change

### Counterfactual analysis applied (per CLAUDE.md pre-committed plans)

| Plan tested | Result | Verdict |
|---|---|---|
| BE Layer (May 1) at floor 0.04% | Raw +$43, honest +$22-30 | ✗ **Floor too tight** — sits inside intra-candle noise. Killed-winner risk on 55+ trades. Higher floor (≥0.10%) cuts benefit substantially. **Not shipped this batch.** |
| TP_min 0.50→0.20 | +$48-55 honest | ★ **Strongest single lever.** Captures small-peak wins currently bleeding to SL. |
| Pullback 0.20→0.10 | Wins on counterfactual but contradicts Apr 14 lesson | ✗ **Withdrawn.** Apr 14 explicitly raised PB 0.08→0.20 because tighter killed runners; PB 0.10 is a return to proven-bad. |
| Pullback 0.20→0.15 | Honest +$6 vs PB 0.20 (cuts BE-zone count from 22 to 11) | ★ **Compromise** — addresses "trades stuck at breakeven" concern without going as tight as the historically-failed 0.08. |
| RSI Handoff (Phase 1d-ExitTest May 2) at L3+ for LONGs | -$3 to -$5 net | ✗ **Falsified for LONG side.** RSI exit fires 1-2 candles AFTER price retraces; tight trailing catches winners closer to peak. May still work for SHORTs (regime-dependent reversal mechanics) — keep as SHORT-only test. |
| Winner Exit widening (Apr 30 Rules B-E) | All variants worse than current | ✗ **Falsified.** Current PB structure is optimal for L2+ winners in this regime. |
| L-P1 (BTC RSI 60-65 × BTC ADX 20-25) PREMIUM | This batch: 35 trades, 57.1% WR | ★ **5-sample CONFIRMED** (weakened from historical 73% pool but still positive zone). |
| L-P2 (60-65 × 30-35) PREMIUM | This batch: 2 trades, 0% WR | ✗ **DEMOTED.** CLAUDE.md Apr 17 audit's suspicion confirmed — original pool dominated by Mar 30 weight. **Drop L-P2 from Phase 2 PREMIUM list.** |
| L-B1 (50-55 × 25-30) HARD BLOCK | 2 trades, 0% WR | Direction-consistent, N too thin to add evidence. Pre-committed; ship with cross-filter UI as locked. |
| Pair RSI 65-70 LONG broken | 36 trades, 27.8% WR | ★ **2-sample structural** (Apr 13 = 36t/25%, this = 36t/27.8%, combined N=72/26.4%). |

### Key new finding: VERY_STRONG tier as currently defined is structurally broken (3-sample confirmed)
- VERY_STRONG (ADX 22-25) WR 38.2% vs STRONG_BUY 38.8% — tier doesn't discriminate
- The actual edge sits at Pair RSI 55-60 × ADX 22-25 (11 trades, 63.6% WR, +$3.36) — single best pair-level cell
- Pair RSI 60-65 × ADX 22-25 (13 trades, 23.1% WR, -$7.50) — same "VERY_STRONG" tier, opposite outcome
- **Phase 2 redefinition of VERY_STRONG must use cross-filter AND-rules, not ADX-only.** Candidate seed: RSI 55-60 × ADX 22-25 with 6th-sample replication required before promotion.

### Methodological lessons documented from this round
1. **CSV column awareness**: when raw analysis is offered, USE the full CSV (126 columns including peak/trough timestamps, post-exit forensics, phantom BE simulation outputs). Earlier passes touched 15 of 126 columns. Don't approximate from txt aggregates when CSV has the raw data.
2. **Counterfactual sequencing matters**: BE/TP simulations must check whether peak preceded trough. Lifetime trough is not the same as post-arming trough — winners with intra-trade dips through the BE floor will have been killed by real BE on the climb, not preserved as my naive sim implied. ATR-weighted intra-trade kill probability is a partial honest correction, but full sim requires intra-trade tick data we don't have.
3. **Cross-sample validation**: bucket-level WR can be pooled across reports (different configs OK). Raw $ cannot. The 2-sample structural patterns (Pair RSI 65-70 broken, BTC ADX direction matters, SAME_REGIME = real edge) are robust; 1-sample patterns are hypotheses.
4. **Don't trust "best simulation result"** — always cross-reference against documented historical lessons before recommending. PB=0.10 was the simulation winner but Apr 14 explicitly disproved it. Re-reading CLAUDE.md before recommending would have caught this.

### Config changes deployed May 4, 2026 (LONG-side)

**Entry filters:**
| Field | Old | New | Rationale |
|---|---|---|---|
| `btc_adx_dir_long` | "both" | **"rising"** | F1: cuts 33 trades (-$21.26 standalone, -$6.27 unique-incremental, 18 unique cuts). Strongest single filter. Aligns with `btc_adx_dir_short: rising` already structurally confirmed since Apr 17 (3-sample SHORT structural now extended to LONG side based on this batch's data: BTC falling 33t/33% WR vs BTC rising 130t/40% WR, combined with the regime-shift dominance pattern). 1-sample LONG-specific evidence; locked revert criterion at next batch (if BTC falling LONG cell shows ≥45% WR with N≥15, revert to "both"). |
| `btc_rsi_adx_filter_long` | "" | **"70-100:35"** | F2: cross-filter rule = "for BTC RSI in [70, 100), require BTC ADX ≥ 35 (else block)". Captures all 18 trades where BTC RSI ≥70 AND BTC ADX <35 (17% WR, -$14.09). Replaces 3 small-N user-proposed cells (BTC RSI 50-55 × ADX 15-20, 70+ × 15-20, 70+ × 20-25 — each N≤9) with one statistically robust rule. The BTC RSI 70+ × ADX 35+ tail breaks even at N=4 so the cap at <35 is correct. **First active use of `btc_rsi_adx_filter_long` field.** Engine code at `services/trading_engine.py:3666+` already supports the format (per Apr 14 Filter design principle entry — Phase 2 BTC cross-filter UI implementation). |
| `momentum_long_rsi_max` | 70.0 | **65.0** | F3: blocks all LONG entries with pair RSI ≥ 65. **2-sample structural** (Apr 13 36t/25% + this batch 36t/28%, combined N=72/26.4% WR). Cuts 36 trades (-$15.34 standalone, -$2.75 unique-incremental, 21 unique cuts). |

**Maker entry revert (per CLAUDE.md May 3 locked decision, applied at this 200-trade checkpoint as planned):**
| Field | Old | New | Rationale |
|---|---|---|---|
| `maker_timeout_seconds` | 40 | **20** | Reverts Apr 18 Amendment #6. Hold-Time Expectancy table shows winners peak/close FASTER than losers (winners <3m bucket = 100% WR / 5 trades; <8m = 43% WR; 60m+ = 33% WR). 40s wait was burning through the front of the move on fast-winner population. All 90 SIGNAL_EXPIRED aborts in this batch fired at exactly t=40.0s — the timeout itself was the proximate cause of the abort cluster. Reverting to 20s should reduce SIGNAL_EXPIRED rate ≥50% per locked criterion. |
| `maker_offset_ticks` | 2 | **1** | Reverts Apr 18 Amendment #8. Bundled with timeout revert (deployed together originally as fill-mechanics package). Tighter offset = higher MAKER fill rate, captures more early winners. |

**Exit changes (both confidence levels VERY_STRONG and STRONG_BUY):**
| Field | Old | New | Rationale |
|---|---|---|---|
| `tp_min` | 0.50 | **0.20** | Strongest single counterfactual lever (+$48-55 honest). Current LONG losers' AvgPeak is +0.20%; trailing arms only at peak ≥ 0.50% under prior config. Lowering threshold lets bot lock the small wins it actually earns. With PB=0.15, exit floor for trades hitting min is +0.05% (clears 0.063% taker roundtrip fee). |
| `pullback_trigger` | 0.20 | **0.15** | Compromise to address breakeven-cluster concern. PB=0.20 from TP=0.20 produces 22 BE trades (peak just above arming threshold then exits at 0%); PB=0.15 cuts that to 11 and shifts those trades into +0.05-0.20 profit zone (+$6 honest). Tighter than the proven-safe 0.20 (Apr 14 lesson) but not as tight as the failed 0.08. **Higher runner-kill risk than 0.20** — partially modeled via ATR-weighted intra-trade dip probability, full risk only known at next batch. |

### Filters DEFERRED for future review (do NOT ship until evidence improves)

| Deferred filter | Reason |
|---|---|
| **F4: Pair RSI 55-60 × Pair ADX 15-18** | Marginal incremental value (-$0.09 unique on 11 trades = essentially zero net). 1-sample finding. Per CLAUDE.md anti-overfit rules, cuts trades for no measurable benefit and risks over-fitting. **Re-evaluate at next 100-trade batch.** If sub-cell still loses with WR ≤ 35% at N ≥ 15, ship as cross-filter rule. |
| **F5: Pair blacklist {HYPEUSDT, RIVERUSDT, DOGEUSDT}** | Only 3 unique cuts in this batch (~$0.30 marginal). Multi-sample evidence exists (HYPE multi-direction, RIVER cross-sample, DOGE SHORT-cross-sample) but $-impact in current batch is small after macro filters do their work. **Defensive blacklist for forward operational protection** — but defer until: (a) 6th-sample shows the pairs continue producing low-quality signals after macro filters land, OR (b) cross-filter UI is built and operational risk justifies blocklist for in-flight trades. **Re-evaluate at next 100-trade batch.** If HYPE LONG continues at WR ≤ 25% on N ≥ 5, ship blacklist immediately. |
| **EMA50 Aligned cross-filters** | Tested 2 candidates: BTC ADX 15-20 × Aligned (13 cell trades, 7 already cut by other rules → only 7 incremental, -$2.85) and BTC ADX 35+ × Aligned (10 cell, 7 already cut → 3 incremental N too thin). 50-70% redundant with macro filters. **Reconsider only if a future regime shows EMA50 alignment as a non-redundant signal.** |
| **EMA50 Alignment as a LONG entry filter** (general) | This batch: Aligned 60t/27% WR, Flat 71t/48% WR, Opposite 32t/41% WR. Strong direction (Aligned worst) but largely confounded with BTC late-cycle pattern that BTC RSI ≥70 + BTC ADX rising filters already address. **Watchlist only** — if 6th sample shows the gap persists after macro filters land, formalize. |

### Pair blacklist watchlist (1-sample concerns, do not act)

| Pair | This batch | Why not blacklist now |
|---|---|---|
| **SOLUSDT** | 10 LONG trades, 10% WR, -$7.74 (worst $-loss in batch) | 1-sample only. Per CLAUDE.md May 3 gates (≥6 trades + WR ≤25% from 2+ reports), needs 2nd sample. **If next batch shows ≥6 trades at ≤25% WR → blacklist.** |
| **BTCUSDT** | 11 LONG trades, 9.1% WR, -$4.62 | 1-sample. BTC is the regime reference pair — blacklisting BTC is structurally awkward when BTC RSI/ADX are core macro signals. Likely the LONG losses reflect BTC entering at regime tops (which the new macro filters now block). **Re-evaluate at next batch — expect macro filters to cut most of these.** |
| **WLFIUSDT** | 4 trades, 25% WR, -$4.96 | 1-sample, thin N. ~6-month-old listing (post-180d new-listing filter cutoff but borderline). |
| **1000LUNCUSDT** | 2 trades, 0% WR, -$4.02 | 1-sample, thin N. Per CLAUDE.md May 3, fails 2-sample bar. |

### Estimated impact of deployed changes (counterfactual on this batch)

| Metric | Baseline | After changes | Δ |
|---|---|---|---|
| LONG WR | 38.2% | ~63% (estimated) | +25pp |
| LONG Total $ | -$45.24 | ~-$2 to +$3 (estimated) | +$45-50 |
| LONG trade rate | 28/day | ~14.5/day | -48% |

Trade-rate halving is significant — be prepared for fewer LONG entries but at much higher quality. The cut trades, even simulated under the new exits, still net to ~$-44, so the volume reduction is real edge improvement, not over-cutting.

### Validation discipline at next 100-trade batch

For each new filter (F1, F2, F3) and each exit change (TP, PB), CLAUDE.md gates apply:
- F1 (BTC ADX rising LONG): if BTC falling LONG bucket shows ≥45% WR on N≥15 → revert to "both"
- F2 (BTC RSI 70+ × ADX <35): if cell shows ≥55% WR on N≥10 → drop or weaken
- F3 (Pair RSI 65-70 cap): if Pair RSI 65-70 trades show ≥45% WR on N≥10 → raise cap back toward 70
- TP=0.20 / PB=0.15: if Avg P&L % flat (±5bp) or worsens, revert each independently. If runner-bucket (peak ≥ 0.50%) shows materially lower close % than historical PB=0.20 performance, PB reverts to 0.20 first while keeping TP=0.20.

### What did NOT change (preserved)

- BE Layer: stays disabled (floor candidates too tight; needs higher-floor design)
- RSI Handoff: stays OFF for LONG (falsified for LONG-side this batch); kept available for SHORT-side test in future
- All other entry filters (Pair RSI 55-60 × ADX 15-18 cell, EMA50 alignment, Funding rate cells, ATR cells, DI spread cells): not shipped (1-sample, marginal, or noise)
- Pair blacklist: HYPE/RIVER/DOGE deferred; existing blacklist (XAGUSDT, XAUUSDT, ZECUSDT, ENAUSDT, RAVEUSDT) unchanged
- SHORT-side config: untouched. SHORT analysis pending separate session.

### Why this entry exists in CLAUDE.md
To anchor the May 4 LONG-side decisions with full counterfactual evidence and explicit revert criteria, so the next-batch validation is automatic rather than re-litigated. Also to preserve the methodological lessons (use full CSV, check historical context before counterfactual recommendations, be honest about simulation limits) that emerged from the iteration this round.

When this batch's results come in, the gates above are the locked test. Pre-committed thresholds, no goalpost moving.

## May 4, 2026 — Phase 3 Position Multiplier (IMPLEMENTED, infrastructure + initial LONG cells)

Per CLAUDE.md May 3 design, implemented after the May 4 LONG-side filter changes (F1/F2/F3 + TP/PB) shipped same day. Pre-conditions met:
1. ✅ 200-trade Phase 1c-Explore checkpoint analyzed
2. ✅ At least one cell passed locked validation gate (L-P1 5-sample confirmed at 57% WR / N=35 this batch)
3. ✅ Pair blacklist decisions resolved (F5 deferred to next batch)
4. ✅ Filter-layer optimizations decided and shipped (F1/F2/F3, TP/PB)

User explicitly approved shipping at this checkpoint; design adjusted from CLAUDE.md May 3 spec on two points (documented below).

### Mechanism summary

For each entry, the engine looks up RSI×ADX cell multiplier rules at TWO levels:
- **Pair-level**: Pair RSI × Pair ADX → `rsi_adx_multiplier_long/short`
- **BTC-level**: BTC RSI × BTC ADX → `btc_rsi_adx_multiplier_long/short` (NEW field, parallels existing `btc_rsi_adx_filter_long/short`)

When a single trade matches BOTH a pair-level AND a BTC-level rule, **HIGHER multiplier wins** (max, not multiply). Rationale: independent confirmation bonus from both layers would compound past the hard cap; HIGHER is simpler and safer.

The chosen multiplier is then applied to either:
- **Investment size** (default) — position $ scaled
- **Leverage** — leverage factor scaled

Both are UI-toggleable via radio. Mechanism is **inert by default** (empty rule strings = no boost on any trade).

### Hard cap (UI-configurable, default 2.0×)

`rsi_adx_multiplier_hard_cap` clamps any per-cell multiplier regardless of user input. Per CLAUDE.md May 3 the design ceiling was 3.0× — **deployed conservative at 2.0×** because 2 of the 3 initial LONG cells are 1-sample only. UI input field allows raising/lowering without code change.

### Capital cap fallback (your concern, addressed)

When the cell multiplier wants more $ than tradeable balance allows, the trade **proceeds at all available** rather than aborting. Existing `min(investment, tradeable)` inside `calculate_position_size` is the natural ceiling. Fallback is logged as `[CELL_MULT_CAPPED]` and persisted on the Order via `cell_multiplier_capped` boolean column for analytics. Trade does NOT fail on capital alone.

Hard-cap clamping is also logged separately as `[CELL_MULT_CAPPED_HARD]` (cell rule wanted X but engine clamped to hard_cap).

### Initial cell activation (LONG only — SHORT empty)

| Cell | Type | Multiplier | Cross-sample basis |
|---|---|---|---|
| BTC RSI 60-65 × BTC ADX 20-25 (L-P1) | BTC-level | 2.0× | **5-sample structural CONFIRMED** (this batch 35t/57% WR; pool 73% across prior 4 samples) |
| BTC RSI 65-70 × BTC ADX 25-30 | BTC-level | 2.0× | 1-sample (this batch only, 8t/75% WR) — user-approved hypothesis |
| Pair RSI 55-60 × Pair ADX 22-25 | Pair-level | 2.0× | 1-sample (this batch only, 11t/63.6% WR) — user-approved; this is the "real VERY_STRONG" cell from this batch |

**Anti-overfit acknowledgment:** 2 of 3 cells fail CLAUDE.md's strict 6-criterion promotion bar (which requires 2-sample structural). Shipping anyway per user direction with explicit revert criterion: any cell showing ≤55% WR on N≥5 in next batch reverts immediately.

### Tracking table — Multiplier Cell Performance

New report section in main.py `_compute_multiplier_cell_performance()`. Per direction (LONG / SHORT), groups CLOSED orders by `cell_multiplier_source`:

Columns: Source | Multi | N | WR% | Avg P&L% | Total$ | Expect$/tr | PF | BL Avg% | Δ vs BL | Capped | Verdict

- **PF (profit factor)** = sum(winners $) / |sum(losers $)|. PF > 1 = profitable, > 1.5 = strong edge.
- **Expect$/tr** = Total$ / N (dollar expectancy per trade).
- **BL Avg%** = baseline = direction's overall Avg P&L% from NON-multiplied trades. Isolates the cell's effect from baseline regime drag.
- **Δ vs BL** = cell's Avg P&L% minus baseline. Positive = cell beats baseline.
- **Capped** = count of trades where balance forced sub-target investment.
- **Verdict** (automated):
  - ★ WORKING — Avg P&L% ≥ baseline + 0.10pp AND Total$ > 0 AND N ≥ 5
  - ✓ Marginal — within ±0.10pp of baseline (multiplier neither helped nor hurt)
  - ⚠ DRAG — materially below baseline (boost hurt edge — variance widened)
  - ✗ HARMFUL — Total$ negative (cell broke under leverage; revert immediately)
  - ⚠ Low N — N < 5 (insufficient data, no verdict)

Plus summary line per direction: total uplift vs simulated 1× baseline.

### Validation discipline at next 100-trade batch

For each cell:
- **N ≥ 5** in next batch required for any verdict
- **★ WORKING**: keep at 2.0× (consider stepping to 2.5× only after 2-sample replication AND CLAUDE.md "+50 trades" rule)
- **⚠ DRAG**: drop to 1.5× or 1.0× depending on severity
- **✗ HARMFUL**: revert that specific cell immediately to 1.0× (drop the rule)
- **⚠ Low N**: continue collecting; no decision

Locked criteria, no goalpost moving.

### Changes vs CLAUDE.md May 3 design

| Aspect | May 3 spec | May 4 deploy | Reason |
|---|---|---|---|
| Hard cap | 3.0× | **2.0× (UI-configurable)** | 2 of 3 cells are 1-sample; conservative first deploy |
| Pair-level only | Yes (`rsi_adx_multiplier_*` only) | **Pair + BTC level** (added `btc_rsi_adx_multiplier_*`) | User wanted 2 BTC-level cells in initial activation |
| Cell match priority | Not specified | **HIGHER (max)** when both pair + BTC match | Prevents compounding past cap; intuitive |
| Default target | "investment" | **"investment"** (UI radio toggle) | Same |
| Decay alert (Layer 5) | Mentioned, deferred | **Deferred to ~50 trades after first multiplier shipped** | Same |

### Files changed (May 4 implementation)

| Layer | File | Change |
|---|---|---|
| Config schema | `config.py` | +6 fields on `SignalThresholds`: 4 rule strings + target enum + hard cap |
| Defaults | `trading_config.json` | +6 keys with initial activation values |
| Engine logic | `services/trading_engine.py` | New `_lookup_rsi_adx_multiplier()` helper (~40 lines); integration in `open_position` (~30 lines); `calculate_position_size` extended with `cell_multiplier`/`multiplier_target` params (returns 3-tuple now: investment, leverage, capped flag) |
| DB | `models.py` + `database.py` auto-migrate | +3 nullable columns on Order: `cell_multiplier` (Float, default 1.0), `cell_multiplier_source` (String 40), `cell_multiplier_capped` (Boolean, default False) |
| API | `main.py` | New `_compute_multiplier_cell_performance()` (~110 lines); included in `_compute_performance` payload as `multiplier_cell_performance` |
| UI | `templates/index.html` | New panel "Premium Multipliers (RSI×ADX cells)" with 4 tables (pair LONG/SHORT, BTC LONG/SHORT) + target radio + hard cap input + JS helpers (`addMultRow`, `loadMultRules`, `collectMultRules`); Multiplier Cell Performance render section after Entry Conditions by Outcome; load + save handlers wired |
| Text export | `templates/index.html` (both export sites) | Multiplier Cell Performance section in both clipboard copy and saved-file exports |

Total: ~400 lines new code across the stack. No schema migration risk (additive only).

### How to interpret the tracking table

Once the bot has run a few trades through the multiplier mechanism:

1. **Read PF first.** PF > 1 = cell is profitable under leverage. PF > 1.5 = edge survives the boost cleanly.
2. **Compare Avg P&L% to BL Avg%.** If cell is ≥+0.10pp above baseline, multiplier is delivering edge. If below baseline, the cell broke under leverage (variance widened or fee drag).
3. **Watch the Capped column.** If many trades show Capped > 0, the multiplier is regularly being held back by available balance — consider raising max_open_positions or reducing reserve to fully express the multiplier.
4. **Verdict column is automated** — don't override it without evidence. ✗ HARMFUL means revert that specific rule.

### Why this entry exists in CLAUDE.md

To anchor the May 4 decisions: which cells were activated, why two of them violate strict CLAUDE.md gates, what the explicit revert criteria are, and how the mechanism handles the user-flagged edge case (insufficient capital → invest all available, not abort).

## May 4, 2026 — RSI Handoff activated for LONG L3+ (against this-batch counterfactual)

User-directed activation per explicit request, despite the May 4 224-trade
counterfactual showing the feature is slightly net-negative for LONG side at
this sample (-$3 to -$5 across various L3 thresholds vs no-handoff baseline).

### Config change
- `rsi_handoff_active`: false → **true**
- `rsi_handoff_level`: 3 (unchanged)

### Mechanism (per CLAUDE.md May 2 Phase 1d-ExitTest design)
With `tp_min=0.20`, the bot's TP-level promotion structure is:
- L1 = peak ≥ 0.20%, trailing arms (target = peak − 0.15)
- L2 ≈ peak ≥ 0.40%
- **L3 ≈ peak ≥ 0.60% (handoff trigger)**

When a trade promotes to L3 (or higher), trailing-stop pullback is **disabled**
and exit fires only on **2-drop RSI** (live RSI exit handler in monitor loop).
Close reason = `RSI_HANDOFF_EXIT L{level}`, distinct from `RSI_MOMENTUM_EXIT`
for analytical separation.

### Why activate despite the counterfactual saying it hurts

1. **Counterfactual mechanics may be biased.** The simulation used
   `post_exit_rsi_exit_pnl` which is recorded AFTER the actual trailing exit
   fired. Under the new rule, the trade wouldn't have trailing-exited; it
   would have continued to whatever peak materialized. The simulation
   approximates this with the existing post-exit data, which may understate
   the upside (a few of the 4 RSI-better trades captured +0.79pp to +0.85pp
   gains via RSI on extended runs that current trailing missed).
2. **Sample is regime-specific.** This batch was BULLISH-choppy with heavy
   regime shifts. In a more directional regime, big-tail LONG winners may
   benefit more from RSI exit (lets them run past trailing arming point).
3. **L3+ population is small** in current data (only 28 LONG trades reached
   peak ≥ 0.60% in this batch). The counterfactual sample for the LONG L3+
   regret comparison is narrow; real outcome at next batch may differ.
4. **User explicitly requested the activation** to test the live behavior
   against the counterfactual prediction.

### Where it appears in reports
- **Post-Exit Regret Deep Dive**: `RSI_HANDOFF_EXIT L3` (and L4/L5 if any)
  will appear automatically as separate rows. The table picks up any
  close reason with post-exit tracking data — no whitelist filtering since
  the Apr 17 unification refactor (see comment at `main.py:4760`).
- **Closing Reason Summary**: same — automatic.
- **Entry Conditions by Close Reason**: same — automatic.
- **Multiplier Cell Performance**: cell-multiplier trades that exit via
  RSI handoff are still attributable to their multiplier cell in that table.

### Pre-committed revert criteria (next-batch validation)

Per CLAUDE.md discipline, locked NOW:

| Outcome at next 100 LONG trades | Verdict |
|---|---|
| `RSI_HANDOFF_EXIT L3+` trades show Avg Close% ≥ what trailing would have locked (counterfactual via post-exit peak data) AND Total $ on those trades ≥ -$2 | **Keep enabled** |
| `RSI_HANDOFF_EXIT L3+` trades show Avg Close% materially below trailing counterfactual on N≥5 | **Revert: rsi_handoff_active back to false** |
| Fewer than 5 trades reach L3 in next batch | **Inconclusive — extend test 100 more trades** |
| Specific cell-multiplier trades (L-P1, etc.) get hurt by RSI handoff vs other exits | Consider per-cell exit override (Phase 4 work) |

No goalpost moving at next-batch checkpoint. The counterfactual evidence
already says this is borderline-negative for LONG; the activation is to
let live data prove or refute the counterfactual on its own terms.

### Note on current LONG-side exit stack with handoff active
- L1 zone (peak 0.20-0.40%): trailing exits at peak − 0.15
- L2 zone (peak 0.40-0.60%): trailing exits at peak − 0.15
- **L3+ zone (peak ≥ 0.60%): trailing DISABLED, RSI handoff exits**
- All other exits (SL at -0.9%, FL system, regime change, signal lost) still apply
- Cell multipliers still apply at entry — cell-boosted trades that reach L3+
  exit via RSI handoff with the boosted position size

### What did NOT change
- SHORT-side: handoff also fires for SHORTs since the toggle is direction-agnostic.
  This is intentional — CLAUDE.md May 2 originally motivated handoff via BEARISH
  +$1.77 post-peak runway data. SHORT-side L3+ effect is untested in this batch
  (61 SHORTs total, very few reaching L3). Will land in next-batch SHORT analysis.
- All other exits (TP, BE, FL, regime, signal lost, momentum exits): unchanged.

## May 4, 2026 — Phase 1c-Explore SHORT-side analysis & config changes (224-trade checkpoint, SHORT subset)

### Sample analyzed
SHORT subset of the May 4 224-trade Phase 1c-Explore checkpoint: 61 SHORT trades in BEARISH regime, 5.87 days, paper, 1x leverage. Same batch as the LONG-side analysis above; separate session focused on SHORT.

### SHORT performance summary
- Total $: -$0.75 (essentially breakeven, vs LONG -$45.24)
- WR: 59.0% (vs LONG 38.7%)
- PF: 0.97 (right at breakeven)
- Never Positive: 16.4% (10 trades, vs LONG 19.4%)
- **SAME_REGIME (16 trades): 100% WR, +$12.86** — pure edge when BTC stays bearish
- **REGIME_SHIFT (45 trades): 44.4% WR, -$13.61** — same regime-shift dominance as LONG (74% of SHORTs)

### Counterfactual exit findings (SHORT-specific)

The May 4 LONG exit changes (TP=0.20, PB=0.15) ALSO benefit SHORTs significantly:

| Exit config | SHORT Total $ | Δ |
|---|---|---|
| Old (TP 0.50, PB 0.20) | -$0.75 | baseline |
| **New (TP 0.20, PB 0.15)** — already deployed | **+$8.68** | **+$9.43** |

**RSI Handoff at L3 HELPS SHORTs** (opposite to LONG counterfactual):
- L3+ SHORTs (peak ≥ 0.60%, N=11 with RSI data)
- Actual via trailing: +$11.73
- RSI handoff would have: +$15.83
- **Δ = +$4.10 better with RSI handoff for SHORTs**
- Mechanism: BEARISH reversals are more violent and faster than bullish pullbacks (per CLAUDE.md May 2 original Phase 1d-ExitTest hypothesis). RSI catches the bottom cleanly; trailing PB locks lower.
- The LONG-side handoff drag (-$3 to -$5) is approximately offset by the SHORT-side gain. **Net handoff effect across both directions is neutral-to-positive.**

### Cross-sample SHORT cell validation (this batch vs CLAUDE.md May 3 saved findings)

| Pre-commit cell | CLAUDE.md status | This batch | Verdict |
|---|---|---|---|
| **S-P1 (BTC RSI <30 × BTC ADX 20-25)** | 5-sample, pool 75% WR | This batch maps to BTC RSI 25-30 × ADX 20-25 = 5/80% WR / +$3.08 | ★ **5-sample structural CONFIRMED** |
| S-P2 (BTC RSI 30-35 × BTC ADX 25-30) | Already weakened in CLAUDE.md Apr 17 audit (was 83% pool, dropped to 57%) | 5/20% WR / -$4.85 | ✗ **CONFIRMS demotion** — drop from PREMIUM list permanently |
| S-B1 (35-40 × 15-20) HARD BLOCK | 7 pool, 43% WR | 2/0% WR / -$1.92 | ★ Direction-consistent |
| S-B3 (45-50 × 15-20) HARD BLOCK | 6 pool, 33% WR | 1/0% / -$0.80 | ★ Direction-consistent |
| **Pair RSI 20-30 × Pair ADX 25-30** | 5-sample, pool 73% WR (N=48) | 22/50% WR / -$7.11 | ⚠ **WEAKENED in this batch** — sub-cell ADX 28-30 is the problem (9 trades, 33% WR, -$6.29) |

### Config changes deployed May 4 (SHORT-side)

| Field | Old | New | Rationale |
|---|---|---|---|
| `macro_trend_flat_threshold_short` | 0.02 | **0.03** | Block weak BTC slope SHORTs. Data: BTC slope abs 0.02-0.03 = 4 trades, 0% WR, -$4.71. Adjacent bucket 0.03-0.04 = 5 trades, 80% WR, +$2.54. Clean breakpoint at 0.03. **1-sample but direction sharp.** Locked revert: if next batch shows BTC slope 0.02-0.03 SHORT cell at ≥45% WR on N≥6, revert to 0.02. |
| `btc_adx_min_short` | 18 | **20** | Replaces 3 user-proposed small-N cell filters (BTC RSI 30-35/35-40/45-50 × ADX 15-20, each N=1-2). Cleaner one-parameter fix. Aggregate evidence: BTC ADX 15-20 SHORTs (across all RSI cells) = 6 trades, 17% WR (1 winner), -$3.03. Locked revert: if next batch shows BTC ADX 18-20 SHORT cell at ≥50% WR on N≥10, revert to 18. |
| `pair_blacklist` | adds **DOGEUSDT** | (was XAG,XAU,ZEC,ENA,RAVE; now +DOGE) | DOGEUSDT SHORT this batch: 4 trades, 0% WR, -$5.62. CLAUDE.md May 3 cross-sample SHORT: 4/0/0% Apr 13. Combined N=8, 0 wins. Multi-sample multi-direction toxic (LONG also losing this batch). Meets May 3 blacklist gate. |

### What did NOT change — SHORT user-proposed but pushed-back

| Filter | Why not deployed |
|---|---|
| BTC EMA20 Slope MAX 0.25 | N=1 in cap zone (this batch). Adjacent 0.15-0.20 bucket = 100% WR / N=7. No evidence to cap. **Revisit at next batch** if more data lands in 0.20+ slope zone. |

### What did NOT change — premium multipliers (SHORT side stays empty for now)

Per CLAUDE.md May 3 strict locked promotion gates:
- **Pair RSI 20-30 × ADX 25-30**: requires N≥15 + WR≥65% in batch. **This batch: 22/50% — FAILS WR gate.** Defer.
- **BTC RSI 25-30 (broader cell)**: requires N≥12 + WR≥80% + uniformity (no sub-cell <50% WR). **This batch: 18/72% pooled — FAILS WR gate AND uniformity** (the BTC ADX 30-35 sub-cell at 5/40% breaks uniformity). Defer.
- **S-P1 (BTC RSI 25-30 × BTC ADX 20-25)** specifically: 5/80% in this batch is direction-consistent with 5-sample pool but N=5 is below the strict bar. **Defer** to keep symmetry with how strict bar was applied to LONG L-P1 (which we activated despite this — but here we want to be more disciplined since LONG already has 3 multiplier cells in the wild).
- **Pair RSI 25-30 × Pair ADX 30-33** (this batch's strongest pair-level winner cell at 7/100%): 1-sample only. Defer.

**No SHORT cells activated for multiplier this batch.** Re-evaluate at next 100-trade SHORT batch with fresh data. If S-P1 (BTC RSI 25-30 × BTC ADX 20-25) replicates ≥75% WR on N≥10, ship at 2.0× then.

### Estimated impact of SHORT changes deployed May 4

| Stack | Total $ | Notes |
|---|---|---|
| SHORT current pre-changes | -$0.75 | baseline (61 trades) |
| + TP=0.20/PB=0.15 (deployed earlier today) | ~+$8.68 | +$9.43 swing |
| + RSI handoff @ L3 (deployed earlier today) | adds ~+$4 to L3+ subset | net positive for SHORTs |
| + macro_trend_flat_threshold 0.03 + btc_adx_min 20 + DOGE blacklist | ~+$3 to +$6 additional | filter out -$3 BTC ADX 15-20 + -$5.62 DOGE |
| **Total projected** | **~+$15 to +$20** | from -$0.75 baseline |

Combined LONG (today's deploys: -$45 → ~+$15-20) + SHORT (today's deploys: -$0.75 → ~+$15-20) = **bot becomes meaningfully profitable on this batch's regime if all changes hold up at next-batch validation.**

### Validation discipline at next 100-trade SHORT batch

For each new filter, locked revert:
- `macro_trend_flat_threshold_short=0.03`: revert if BTC slope 0.02-0.03 SHORT cell shows ≥45% WR on N≥6
- `btc_adx_min_short=20`: revert if BTC ADX 18-20 SHORT cell shows ≥50% WR on N≥10
- DOGEUSDT blacklist: re-evaluate after 200 more trades worth of "would-have-been DOGEUSDT" signals (logged via existing skip mechanism). If DOGEUSDT signals stop appearing toxic in observation, consider removing.

For multipliers (still empty for SHORT):
- S-P1 promotion gate at next batch: ≥75% WR on N≥10 → activate at 2.0× (looser than CLAUDE.md May 3's 80% gate, since we already broke the strict bar for LONG L-P1; symmetric treatment)

### Why this entry exists in CLAUDE.md
To anchor the May 4 SHORT-side decisions: which filters were validated, which were deferred, why no multipliers shipped on SHORT side (despite cross-sample evidence existing for S-P1), and what the explicit revert criteria are for next-batch validation.

The contrast vs LONG-side (3 multiplier cells shipped, 2 of them 1-sample) is intentional: LONG was a fresh feature deploy where some 1-sample activation was acceptable to test the mechanism. SHORT comes after — by then the strict bar should reassert. If next batch validates SHORT cells cleanly, ship them at multiplier with confidence.

## May 4, 2026 — SHORT Premium Multiplier cells activated (4 cells at 2.0×)

Per user direction following methodological correction (cross-sample pool was not properly applied in initial SHORT analysis — see "## May 4, 2026 — Phase 1c-Explore SHORT-side analysis" entry above which originally argued for 0 SHORT cells, then revised after pooling).

### Activated cells (all at 2.0×, investment-target via existing UI toggle)

**BTC-level (`btc_rsi_adx_multiplier_short`):**
| Cell | This batch | Cross-sample basis |
|---|---|---|
| BTC RSI 25-30 × BTC ADX 20-25 (S-P1) | 5 trades, 80% WR, +$3.08 | **5-sample structural** (Apr 17 ex-Mar30 audit pool: N=12, 75% WR; combined N=17 at ~76% WR) — passes locked S-P1 promotion gate |
| BTC RSI 25-30 × BTC ADX 25-30 | 6 trades, 83% WR, +$2.97 | 1-sample. Sub-cell of broader BTC RSI 25-30 zone (cross-sample pool 27 trades, ~81% WR per CLAUDE.md May 3 + this batch). Direction-supported but uniformity caveat — adjacent BTC ADX 30-35 sub-cell broke this batch (5/40%) |

**Pair-level (`rsi_adx_multiplier_short`):**
| Cell | This batch | Cross-sample basis |
|---|---|---|
| Pair RSI 20-30 × Pair ADX 30-33 | 7 trades, 100% WR, +$4.79 | **1-sample only.** Sub-cell of broader Pair RSI 20-30 × ADX 25-30 zone (5-sample pool 73% WR). Sub-cell over-fitting risk: parent zone weakened to ~67% pool WR after this batch (the 28-30 ADX sub-cell broke at 9/33%). Activation is a directional bet on the ADX 30-33 tail being where the edge actually lives. |
| Pair RSI 30-35 × Pair ADX 25-28 | 6 trades, 83% WR, +$2.88 | **1-sample only.** Outside the documented 5-sample S-P1/S-P2 pre-commit zones. Directional pattern from this batch. |

### Multiplier rule strings deployed
```
rsi_adx_multiplier_short: "20-30:30-33:2.0,30-35:25-28:2.0"
btc_rsi_adx_multiplier_short: "25-30:20-25:2.0,25-30:25-30:2.0"
```

### Why 4 cells instead of 1 (S-P1 only) — symmetric to LONG approach

LONG side shipped 3 multiplier cells today (1 confirmed, 2 1-sample). User extended same approach to SHORT side: ship the 5-sample-confirmed cell PLUS 3 strongest 1-sample cells from this batch's cross-tabs.

This is more aggressive than my Option A revised recommendation (S-P1 only) and accepts higher 1-sample noise. **The user's tradeoff is explicit:** test the multiplier mechanism on more cells; accept that some will revert at next batch.

### Pre-committed revert criteria (locked NOW, validated at next batch)

For each cell, if next batch shows on N≥5:
- **★ WORKING**: WR ≥ 70% AND Total $ positive → keep at 2.0×
- **⚠ Marginal**: 50-70% WR → drop to 1.5×
- **✗ HARMFUL**: WR ≤ 40% OR Total $ negative → revert to 1.0× (drop the rule entirely)
- **⚠ Low N (<5 trades fired)**: extend test, no decision

Specific concerns to watch:
- **S-P1 (BTC RSI 25-30 × BTC ADX 20-25)**: highest expected reliability (5-sample confirmed). Should hold up. If not, the cross-sample pool itself is breaking and we have a regime-shift problem at the multiplier level.
- **BTC RSI 25-30 × BTC ADX 25-30**: 1-sample. Watch closely — the parent zone (BTC RSI 25-30 broad) had a uniformity break in this batch (the 30-35 ADX sub-cell at 5/40%). If next batch's 25-30 sub-cell drops below 65% WR, revert.
- **Pair RSI 20-30 × Pair ADX 30-33**: 1-sample 7/100%. Most aggressive bet. The parent Pair RSI 20-30 × ADX 25-30 zone WEAKENED to 50% in this batch from 73% pool. Activating a sub-cell of a weakening parent is risky. Most likely revert candidate.
- **Pair RSI 30-35 × Pair ADX 25-28**: 1-sample 6/83%. Pure new bet outside documented patterns. Watch for replication.

### Capital interaction with activated cells

Multiplier conflict resolution unchanged: HIGHER (max) wins when both pair-level and BTC-level cells match. Possible conflict scenarios for SHORT:
- A trade with Pair RSI 28, Pair ADX 31, BTC RSI 27, BTC ADX 22 → matches BOTH pair (PAIR_20-30_30-33 = 2.0×) AND BTC (BTC_25-30_20-25 = 2.0×). HIGHER = 2.0× via either source. Logged source = whichever rule was found first.

Capital cap fallback unchanged: `min(target_investment, tradeable_balance)`. With 5 max open positions and equal-split mode, a 2.0× cell on a low-balance scenario will lock at all available rather than abort.

### Files changed (May 4 SHORT multiplier activation)

- `trading_config.json`: `rsi_adx_multiplier_short` and `btc_rsi_adx_multiplier_short` populated with 4 cells

### Combined LONG + SHORT multiplier landscape after this deploy

| Side | Cells active | Multipliers | Cross-sample basis |
|---|---|---|---|
| LONG (deployed earlier today) | 3 | 2.0× each | 1 confirmed + 2 1-sample |
| **SHORT (this entry)** | **4** | **2.0× each** | **1 confirmed + 3 1-sample** |
| Total active cells | 7 | hard cap 2.0× | mixed |

Hard cap at 2.0× and the HIGHER-conflict resolution mean no trade gets boosted above 2.0× regardless of how many cells it matches. Multiplier mechanism stays in safe operational range.

### Why this entry exists in CLAUDE.md

To anchor today's SHORT multiplier activation with explicit revert criteria for each of the 4 cells, and to honestly document that the activation is more aggressive than my initial recommendation (S-P1 only). The user-driven 4-cell activation is explicitly accepted as a higher-variance bet on the multiplier mechanism. Next-batch validation will tell which cells deserved the boost.

If 3 of 4 cells revert at next batch, that's a methodological lesson about 1-sample multiplier activation — and the locked criteria above ensure the revert decisions happen automatically rather than being re-litigated.

## May 4, 2026 — Exploration Analytics section REMOVED

Per the CLAUDE.md April 30 conditional-removal plan ("Provisional status (added Apr 30)"), the Exploration Analytics — Observation Only section is removed from the UI and reports. User assessment confirmed at this checkpoint: "it does not add any value at all".

The April 30 plan said: *"if NO Tier 1 dimension meets the 6-criterion promotion bar... AND none of the cross-tabs show meaningful discrimination → remove the entire Exploration Analytics UI section, the 6 single-dim tables, the 4 cross-tabs, and the TtP table. Keep the underlying entry_* DB columns (cheap, no harm in retaining captured data) but drop the rendering and the report-export entries."*

That's exactly what was done.

### What was removed

**templates/index.html (~482 lines total removed):**
- "Exploration Analytics — Observation Only" UI panel (1 H2 + 6 single-dim tables + 4 cross-tabs section, ~292 lines)
- JS renderers `renderExplorationTable` and `renderCrosstabTable` (~108 lines)
- Both text-export site renderers `_explorationTables`/`_explorationCrosstabs` and `_explorationTables2`/`_explorationCrosstabs2` (~82 lines)

**main.py (~281 lines total removed):**
- 7 single-dim bucket-performance helpers: `_compute_atr_performance`, `_compute_ema50_slope_performance`, `_compute_ema50_alignment_performance`, `_compute_funding_rate_performance`, `_compute_di_direction_performance`, `_compute_di_spread_performance`, `_compute_ttp_ratio_performance`
- 4 cross-tab helpers: `_compute_btc_adx_ema50_alignment_crosstab`, `_compute_pair_adx_di_spread_crosstab`, `_compute_btc_rsi_funding_crosstab`, `_compute_direction_ema50_alignment_crosstab`
- 2 generic infrastructure helpers (only used by the above): `_bucket_perf`, `_crosstab_perf`
- 11 payload entries from `_compute_performance` and the empty-data fallback branch
- 3 explanatory comment lines

### What was DELIBERATELY RETAINED (per CLAUDE.md April 30 plan)

**Order DB columns** — capture is cheap, removing only the analysis surface is the right granularity:
- `entry_pos_di`, `entry_neg_di`, `entry_atr_pct`, `entry_ema50_slope`, `entry_funding_rate`

**Per-trade `_compute_ttp_ratio()` helper** — used by integrated Entry Conditions by Outcome / Entry Conditions by Close Reason tables (which DO have analytical value, unlike the standalone bucket aggregates). Different from `_compute_ttp_ratio_performance()` which was the bucket aggregator and got removed.

**Entry Conditions by Outcome / Reason table columns** that USE these underlying fields (TtP ratio, EMA50 align/slope, ATR%, +DI/-DI/spread, funding) — these stay because they're embedded in the proven-useful winner-vs-loser comparison view. Those tables are NOT what was removed; only the standalone Exploration Analytics section was.

**Data capture in `services/trading_engine.py`** — `entry_pos_di` etc. continue to be recorded on every Order (no change to the capture path). If a future analytical need arises, the data exists in the DB.

### Why this is safe

1. **Schema unchanged** — no DB migration. `entry_pos_di`/etc. columns stay; just no UI displays them as bucket aggregates anymore
2. **Per-trade column data in Entry Conditions tables still works** — the integrated views use the same data via `_compute_ttp_ratio()` (kept) and direct attribute reads (no helper functions touched)
3. **API parses cleanly** — AST validated; no orphan references
4. **All Premium Multiplier / filter changes from earlier today still work** — none of those depend on the removed exploration helpers

### Reactivation path (if ever needed)

Should a future regime show one of these dimensions becoming meaningful:
1. The DB columns are already populated for every trade since Apr 28 — no historical data lost
2. Re-add specific bucket-performance helper from git history (this commit's parent contains the pre-removal code)
3. Add UI section back targeted to ONLY the dimension that became meaningful (don't re-add the whole section)

### Files changed

- `templates/index.html`: -482 lines (10,116 → ~9,303 incl earlier multiplier additions)
- `main.py`: -281 lines (5,571 → 5,292)

### Why this entry exists in CLAUDE.md

To anchor that the removal followed the April 30 conditional-removal protocol exactly: bar wasn't met, no signal in the data, removed cleanly while preserving the underlying capture for future re-analysis. Also so future-Claude knows that re-adding bucket-aggregate UI for ATR/EMA50/funding/DI is a deliberate REVERSAL of a documented decision, not a fresh build.

## May 4, 2026 — LOCKED next-batch validation plan (reference baseline + revert criteria)

### Reference baseline = May 4 224-trade batch
All next-batch comparisons measure changes against this snapshot:
- Reports: `reports/report_2026-05-04_phase1c_explore_224trades.txt` + `reports/orders_2026-05-04_phase1c_explore_224trades.csv`
- LONG: 163 trades, 38.7% WR, **-$45.24**, -0.14% Avg
- SHORT: 61 trades, 59.0% WR, **-$0.75**, -0.01% Avg
- Combined: 224 trades, 44.2% WR, **-$45.99**

### Layer 1 — Filter changes (block losing trades)

| Change | Side | Pre-committed revert criterion |
|---|---|---|
| `btc_adx_dir_long: rising` | LONG | If BTC falling LONG cell shows ≥45% WR on N≥15 → revert to "both" |
| `btc_rsi_adx_filter_long: "70-100:35"` | LONG | If cell shows ≥55% WR on N≥10 → drop the rule |
| `momentum_long_rsi_max: 65` | LONG | If Pair RSI 65-70 LONG shows ≥45% WR on N≥10 → raise back toward 70 |
| `macro_trend_flat_threshold_short: 0.03` | SHORT | If BTC slope 0.02-0.03 SHORT cell shows ≥45% WR on N≥6 → revert to 0.02 |
| `btc_adx_min_short: 20` | SHORT | If BTC ADX 18-20 SHORT shows ≥50% WR on N≥10 → revert to 18 |
| `pair_blacklist += DOGEUSDT` | both | Re-evaluate after observation; if next batch shows DOGE signals would have been profitable, remove |

### Layer 2 — Exit changes (capture small wins, address breakeven cluster)

| Change | Pre-committed validation |
|---|---|
| `tp_min: 0.20` (both directions) | Counterfactual predicted +$48 LONG, +$9 SHORT swing. **Keep if actual swing ≥+$30 LONG AND ≥+$5 SHORT.** Revert if swing < +$10 combined. |
| `pullback_trigger: 0.15` (both) | Watch "trades stuck at breakeven" count. Should drop from ~22 (LONG baseline) to ~11. **Revert PB to 0.20 if BE-bucket stays ≥18 trades.** |
| `rsi_handoff_active: true` (L3+) | **Revert if L3+ `RSI_HANDOFF_EXIT` trades show net < counterfactual trailing exit on N≥5.** Note: counterfactual showed -$3 to -$5 for LONG, +$4 for SHORT in the May 4 batch. Live data may differ. |

### Layer 3 — Maker entry revert

| Change | Pre-committed validation |
|---|---|
| `maker_timeout_seconds: 20` | SIGNAL_EXPIRED rate (per direction) should drop **≥50%** vs May 4 batch (90 LONG aborts + 20 SHORT aborts). If rate doesn't drop ≥50% → investigate (signal-flip mechanic, not timeout, may be the real issue). |
| `maker_offset_ticks: 1` | MAKER fill rate should rise. Revert to 2 ticks if fill rate **drops >30%** from baseline. |

### Layer 4 — Premium Multipliers (per-cell verdict)

7 cells active at 2.0×, hard cap 2.0×, HIGHER conflict resolution. Per-cell pre-committed verdict at next batch:

| Verdict criterion | Action |
|---|---|
| ★ WORKING: WR ≥70% AND Total $ positive AND N ≥5 | Keep at 2.0× |
| ✓ Marginal: 50-70% WR | Drop to 1.5× |
| ✗ HARMFUL: WR ≤40% OR Total $ negative | Revert to 1.0× (drop the rule entirely) |
| ⚠ Low N: <5 trades fired | Extend test, no decision |

Per-cell expected revert risk:

| Cell | Cross-sample basis | Risk |
|---|---|---|
| LONG L-P1 (BTC RSI 60-65 × BTC ADX 20-25) | 5-sample ★ | Lowest |
| LONG BTC RSI 65-70 × BTC ADX 25-30 | 1-sample | Medium |
| LONG Pair RSI 55-60 × Pair ADX 22-25 | 1-sample | Medium |
| SHORT S-P1 (BTC RSI 25-30 × BTC ADX 20-25) | 5-sample ★ | Lowest |
| SHORT BTC RSI 25-30 × BTC ADX 25-30 | 1-sample | Medium |
| SHORT Pair RSI 20-30 × Pair ADX 30-33 | 1-sample (sub-cell of weakening parent) | **Highest** — most likely revert |
| SHORT Pair RSI 30-35 × Pair ADX 25-28 | 1-sample (new bet) | Medium-high |

### Combined projected outcome (if everything holds)

| Side | Baseline | Projected after stack |
|---|---|---|
| LONG | -$45.24 | **+$15 to +$25** |
| SHORT | -$0.75 | **+$15 to +$22** |
| **Combined** | **-$45.99** | **~+$30 to +$45** |

**Strategy validated if combined ≥ +$10 at next batch.**
**Marginal (per-rule reverts needed) if combined -$5 to +$5.**
**Major reconsideration if combined < -$10** (likely regime-related, not filter-related).

### Watchlist (NOT yet shipped, evaluate at next batch)

- **F4 LONG**: Pair RSI 55-60 × Pair ADX 15-18. Activate if next batch shows WR ≤35% on N≥15 in this cell.
- **F5 LONG pair blacklist**: HYPEUSDT, RIVERUSDT (defensive). Activate if next batch shows continued losing.
- **Pair watchlist**: SOLUSDT, BTCUSDT, WLFIUSDT, 1000LUNCUSDT — blacklist if 2nd-sample confirmation (≥6 trades + WR≤25%).
- **BTC EMA20 Slope MAX 0.25 SHORT** (user pushback held today). Activate if more data lands in 0.20+ slope zone showing losses on N≥4.

### Discipline lock

Every revert criterion above is **pre-committed**. At next-batch checkpoint:
1. Apply criteria mechanically — no re-litigation
2. Anything interesting outside these criteria → goes on watchlist for the BATCH AFTER, not actioned mid-evaluation
3. Goalposts don't move. If a rule says "revert if WR ≤40%" and the rule shows 41% WR on N=8, that's NOT "close enough to keep" — that's a marginal case that drops to 1.5× per the verdict matrix

### Why this entry exists in CLAUDE.md

To eliminate any ambiguity about what's being tested at next batch and what the decisions will be. With this single locked plan in writing:
- Future-Claude opens CLAUDE.md, finds this entry, runs the matrix
- Future-User can read the plan and know exactly what each metric means
- No "let's discuss whether to revert" debates — the criteria already say what to do

If next batch shows results that are **clearly worse** in the loss buckets where these changes were targeted (e.g., LONG Positive-No-BE bucket grows instead of shrinks under TP=0.20), that's evidence the counterfactual model itself is wrong — and we step back to re-examine the analytical methodology, not just the parameter values.

## May 4, 2026 — Toggle for signal re-validation at maker timeout (`revalidate_on_taker_fallback`)

### Why added
User raised: "Can we eliminate the recheck after the timeout? We didn't have it in previous batches (not the one we analysed today)."

The signal re-validation was added Apr 18 as Phase 1c Amendment #7, motivated by the simultaneous Amendment #6 (timeout 20s → 40s). Logic: longer wait = more chance signal staleness, so re-check before firing taker fallback.

But Amendment #6 was **reverted today** (timeout back to 20s per CLAUDE.md May 3 locked decision). With the shorter wait, signal-staleness exposure is materially lower. The re-validation may now be filtering trades unnecessarily.

Per the May 4 224-trade analysis, SIGNAL_EXPIRED rate was 35-44% of all entry attempts (90 LONG aborts + 20 SHORT aborts). Of LONG aborts, 71 were "Signal Flipped" (legitimate) but 18 were spurious "BTC RSI Out of Range" (bug fixed Apr 30) — even after the fix, ~70% abort rate at timeout means the gate fires often.

### Toggle mechanism
New top-level config field `revalidate_on_taker_fallback: bool = False` (default OFF per user direction May 4):
- **OFF (default — pre-Apr 18 behaviour, May 4+ baseline)**: at timeout, taker fallback fires immediately. No re-validation.
- **ON (Apr 18+ behaviour)**: after maker timeout exhausts, re-evaluate the signal's BTC ADX direction, BTC ADX range, BTC RSI range, and core get_signal output. If any check fails, abort entry as `SIGNAL_EXPIRED` (no taker fires).

Both maker-entry call sites (live `_try_maker_entry`, paper `_simulate_maker_entry_paper`) wrap the existing re-validation block in:
```python
revalidate_enabled = getattr(config.trading_config, 'revalidate_on_taker_fallback', True)
if revalidate_enabled and confidence is not None:
    is_valid, revalidate_reason = await self._revalidate_entry_signal(...)
    if not is_valid:
        return SIGNAL_EXPIRED + log
```
When the toggle is OFF, the entire re-validation block is skipped and `[MAKER_ENTRY] {pair}: No fill, re-validation disabled, falling back to market order` is logged.

### UI
Toggle appears next to Maker Entry settings labeled "Re-validate Signal at Timeout". Defaults to OFF (May 4+). User can toggle without restart (config save flushes immediately).

### Files changed
- `config.py`: +1 field (`revalidate_on_taker_fallback: bool = False`)
- `trading_config.json`: +1 default `false`
- `services/trading_engine.py`: 2 wrapping conditionals (~6 lines)
- `templates/index.html`: UI toggle + load + save handlers (load fallback default `false`)
- `main.py`: ConfigUpdate Pydantic model field

### Validation discipline
Toggle ships with default **OFF** per user direction May 4. This is a behaviour change from the Apr 18+ default — re-validation no longer fires unless explicitly toggled ON.

In the next batch (with re-validation OFF), watch:
- **SIGNAL_EXPIRED count should drop to ~0** (only signal-flipped at moment of taker order will appear, which is much rarer)
- **TAKER_FALLBACK count should rise** by approximately the count of previously-aborted trades
- **Signal-flipped TAKER trades**: these are the ones Amendment #7 was meant to protect against. Look for them in the new batch's losers — if a meaningful number of TAKER_FALLBACK trades show entry-signal != close-state regime mismatch and lose money, the re-validation was doing useful work and should be re-enabled

### Why this entry exists in CLAUDE.md
To anchor the design choice (toggle, not hard-remove): re-validation may have analytical value in some configurations even though it appears suspect under current 20s timeout. Toggle keeps both paths testable.
