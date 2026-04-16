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

### Phase 1b (CURRENT) — Looser config for data collection (Apr 15 onwards)
Rationale: Phase 1a was starving the long side. Four filter changes applied to unblock long entries and collect data on currently-unexplored buckets. The strategic decision: use the BTC RSI × BTC ADX cross-tab (Phase 2 code work) as the fine-grained filter, rather than multiple coarse per-variable mins that may be cutting good trades along with bad.

**Changes from Phase 1a → Phase 1b:**
| Config | Apr 14 (Phase 1a) | Apr 15 (Phase 1b) | Rationale |
|---|---|---|---|
| `btc_adx_min_long` | 25 | **20** | Historical BTC ADX 20-25 longs: 41 trades, 71% WR, +$124 across 4 samples — not a losing bucket |
| `macro_trend_flat_threshold_long` | 0.06 | **0.02** | Match shorts (which were firing fine at 0.02). Expected to unblock longs in sideways BTC conditions |
| `ema_gap_5_20_min_long` | 0.10 | **0.05** | Gap 5-20 is non-monotonic per 4-sample data. 0.12-0.15% was OK (17 trades, +$11.84), 0.15-0.20% was worst (40 trades, -$79). Lowering min explores 0.05-0.10% range (zero historical data) |
| `ema_gap_5_20_min_short` | 0.15 | **0.05** | Historical short data only for 0.15+ bucket. Lowering explores uncharted 0.05-0.15% range |

**Unchanged filters (still locked for Phase 1b):**
- Leverage: 1x both VERY_STRONG and STRONG_BUY
- Trade mode: both, max 5 positions, equal_split, $100 fixed
- `ema_gap_threshold_long = 0.02`, `ema_gap_threshold_short = 0.08` (EMA5-EMA8 gap, separate from EMA5-EMA20)
- `momentum_adx_max_long = 25` (not changed; let data decide)
- `btc_adx_min_short = 20` (unchanged)
- `momentum_ema20_slope_min_long = 0.0`, `momentum_ema20_slope_min_short = 0.04`
- Exits: TP 0.50 / pullback 0.20, BE L1 0.15/0.10, Signal Lost Flag ON, FL1/FL2 ON, Regime Change Exit ON, Tick Momentum OFF, RSI Momentum OFF
- Market Breadth ON (30 bull L, 45 bear S, flat 0.02)
- EMA Gap Expanding ON, RSI Momentum Filter ON
- Spike Guard ON (3x vol, 1.5% price)

### Rule for Phase 1b (the next 100 trades)
**NO FURTHER CONFIG CHANGES.** Phase 1b starts fresh with the 4 changes above. From here to 100 trades: no tweaks. If long trade rate is still starved after Phase 1b, the issue is elsewhere (code bug, indicator calc, or truly no market opportunity).

### How to analyze Phase 1b data
At 100-trade checkpoint, in addition to the existing 22-question checklist:
1. **Did long trade rate increase meaningfully?** Compare Phase 1a (~1 long/day) to Phase 1b rate.
2. **What's the 0.05-0.10% gap 5-20 bucket performance (longs)?** First-ever data in this range.
3. **What's the 0.05-0.15% gap 5-20 bucket performance (shorts)?** First-ever data in this range.
4. **Did BTC ADX 20-25 long bucket replicate historical 71% WR?** 5th-sample confirmation.
5. **Did BTC slope-flat longs perform OK now that threshold was lowered to 0.02%?** Previously blocked entirely.
6. **Full BTC RSI × BTC ADX cross-tab** — the fine-grained filter that will eventually replace these blunt mins.

### Pooling rule
**Do NOT pool Phase 1a (19 trades) with Phase 1b raw data.** They were different configs. Use Phase 1a as a "pre" reference and Phase 1b as the "post" measurement.

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
