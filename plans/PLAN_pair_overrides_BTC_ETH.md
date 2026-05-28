# PLAN — Per-Pair Override Mechanism for BTCUSDT + ETHUSDT

**Status:** NOT BUILT — design + values locked, awaiting go-ahead.
**Created:** 2026-05-28
**Author:** Claude analysis session (deduped pool 965 trades, May 4 → May 27)

---

## Goal

Enable per-pair overrides so BTC and ETH use their own entry filters, exit
thresholds, and sizing — independent of alt-tuned globals. Unblock the
83-84% of historical BTC/ETH trades that current filters reject.

User intent: **more trades on BTCUSDT (and ETHUSDT), both LONG and SHORT
enabled**, while alts keep their existing filters.

---

## KEY DATA FINDING (the reason this plan exists)

Applying the CURRENT filter stack to historical BTC/ETH trades:

| | Historical | Survives today | Cut |
|---|---|---|---|
| BTC | 23 | **4 (17%)** | 19 (83%) |
| ETH | 25 | **4 (16%)** | 21 (84%) |

### What's blocking them (top blockers):

| Blocker | BTC cuts | ETH cuts |
|---|---|---|
| BTC_RSI_ADX_X_L (LONG macro cross-filter) | 12 | 8 |
| EMA_GAP_MIN_L (ema_gap_threshold_long 0.06) | 4 | 7 |
| BTC_RSI_ADX_X_S (SHORT macro cross-filter) | 2 | 5 |
| PAIR_ADX_MAX_S | 0 | 1 |
| GLOBAL_VOL_MAX_S | 1 | 0 |

**Two filters do ~90% of the blocking: the BTC RSI × BTC ADX cross-filter
and the EMA gap minimum.**

### Survivor performance (after current filters):

- **BTC survivors (N=4, ALL SHORT — LONG side 100% blocked):** 75% WR,
  Total -$20.61. 3 tiny wins (+$0.61/+$0.86/+$5.02), 1 loss -$27.10.
- **ETH survivors (N=4, 3 SHORT + 1 LONG):** 75% WR, Total +$153.93.
  SHORTs +$41/+$7/+$110. The 1 LONG lost -$4.

### Winners vs Losers profile (within survivors):

**BTC** (3W vs 1L): winners had BTC ABOVE 4hr trend (BTCTGap +0.15 vs -0.35),
reached peak +0.45% vs loser +0.05%.

**ETH** (3W vs 1L): winners ALL SHORTs in bearish setups (PairRSI 31 vs 65,
RngPos 20 vs 78, EMA20slope -0.15 vs +0.14, BTCRSI 37 vs 66). The lone LONG
loser was the chase-the-top profile (high RSI + high RngPos + bullish macro).

### Cross-pair pattern:
- SHORTs dominate survivor pool (BTC 4/4, ETH 3/4)
- Both LONG sides essentially blocked by BTC_RSI_ADX_X_L
- The one LONG that survived (ETH) LOST
- → if loosening filters, SHORT side has evidence; LONG side carries higher
  uncertainty

---

## BASELINE — actual current config (from trading_config.json, NOT config.py defaults)

**Exit params live in `confidence_levels.<tier>`** (both VERY_STRONG and
STRONG_BUY identical today):

| Param | Actual current |
|---|---|
| tp_min | 0.45 |
| pullback_trigger | 0.25 |
| stop_loss / signal_active_sl | -0.70 |
| be_levels_enabled | false (all BE triggers 99 = off) |
| leverage | 20× |
| investment_multiplier | 1.0 |

**Globally OFF mechanisms:** fast_exit_enabled=false, fast_exit_l2_enabled=false,
signal_lost_exit_enabled=false, fl1_for_wide_sl_enabled=false,
pair_trend_filter_enabled=false, rsi_handoff_active=false.

**Key thresholds:** ema_gap_threshold_long=0.06, ema_gap_threshold_short=0.08,
ema_gap_5_8_max_long/short=0.35, momentum_long_rsi_max=65, momentum_short_rsi_max=50,
momentum_adx_max_long=30, momentum_adx_max(short)=35, adx_dir_long/short="rising",
min_adx_delta_long/short=0.0, momentum_ema20_slope_min_short=0.06,
momentum_ema20_slope_max_long/short=0.40, entry_dist_from_ema13_min_long=0.20,
entry_quality_score_filter_enabled=true, btc_adx_min_long=15, btc_adx_max_long=40,
btc_adx_min_short=18, btc_adx_max_short=40, global_volume_threshold_long=0.7,
global_volume_threshold_short=0.5, global_volume_max_short=1.1,
pair_volume_usd_rescue_long=$50M.

btc_rsi_adx_filter_long="70-100:40,60-65:22-25,60-65:27-30,55-60:20-25,50-55:22"
btc_rsi_adx_filter_short="30-35:30,35-40:20-26,45-50:25,0-30:25-30"
btc_rsi_band_atr_block_long="65-70:<0.10"

---

## ARCHITECTURE

### Lookup order at runtime
```
pair_overrides[pair].<field>      → if present, use this
confidence_levels[tier].<field>   → else fallback (exits, sizing)
thresholds.<field>                → else fallback (filters)
```

### Config schema (new top-level field in trading_config.json)
```json
"pair_overrides": {
  "BTCUSDT": { "enabled": true, "exits": {...}, "filters": {...},
               "sizing": {...}, "directions": {...}, "macro_filters": {...},
               "multipliers": {...} },
  "ETHUSDT": { ... }
}
```
Every field optional — omit = inherit global.

### Engineering scope (~566 LOC, single commit)
| File | Change | LOC |
|---|---|---|
| config.py | `pair_overrides: Dict = {}` field | 5 |
| trading_config.json | initial BTC + ETH blocks | 80 |
| services/trading_engine.py | `_get_pair_param(pair, field, default, tier)` helper + ~40 lookup replacements | 100 |
| services/indicators.py | same helper at ~12 filter sites | 30 |
| templates/index.html | Pair Overrides UI panel (collapsible BTC+ETH, "Show effective config" preview, load/save) | 250 |
| main.py | ConfigUpdate Pydantic field | 1 |
| CLAUDE.md | doc entry + values + revert criteria | 100 |

---

## BTCUSDT — COMPLETE OVERRIDE VALUES

### exits (overrides confidence_levels)
| Variable | Global | BTC |
|---|---|---|
| tp_min | 0.45 | **0.08** |
| pullback_trigger | 0.25 | **0.05** |
| pullback_widening_per_level | 0.10 | **0.04** |
| stop_loss | -0.70 | **-0.35** |
| signal_active_sl | -0.70 | **-0.35** |
| be_levels_enabled | false | **true** |
| be_level1_trigger | 99 | **0.10** |
| be_level1_offset | 99 | **0.04** |
| fast_exit_enabled | false | **true** |
| fast_exit_threshold_pct | 0.20 | **0.08** |
| fast_exit_window_minutes | 2 | **4** |
| fast_exit_l1_atr_multiplier | 0.50 | **0.50** |
| fast_exit_l1_atr_floor_cap_pct | 0.60 | **0.20** |
| fast_exit_l2_enabled | false | **true** |
| fast_exit_l2_threshold_pct | 0.40 | **0.15** |
| fast_exit_l2_window_minutes | 5 | **6** |
| fast_exit_l2_atr_floor_cap_pct | 0.80 | **0.25** |
| signal_lost_exit_enabled | false | **false** |
| signal_lost_flag_security_min | -0.9 | **-0.35** |
| signal_lost_flag_security_max | -0.8 | **-0.25** |
| fl1_for_wide_sl_enabled | false | **false** |
| fl1_wide_sl_backstop | -0.95 | **-0.55** |
| ema13_cross_exit_enabled | true | **true** |
| rsi_handoff_active | false | **false** |
| trailing_atr_multiplier | 0.50 | **0.50** |

### filters — LONG
| Variable | Global | BTC |
|---|---|---|
| ema_gap_threshold_long | 0.06 | **0.005** |
| ema_gap_5_8_max_long | 0.35 | **0.20** |
| momentum_long_rsi_max | 65 | **65** |
| momentum_adx_max_long | 30 | **30** |
| adx_dir_long | "rising" | **"both"** |
| min_adx_delta_long | 0.0 | **0.0** |
| momentum_ema20_slope_min_long | 0.0 | **0.0** |
| momentum_ema20_slope_max_long | 0.40 | **0.30** |
| entry_dist_from_ema13_min_long | 0.20 | **0.05** |
| entry_dist_from_ema13_filter_enabled | true | **true** |
| ema5_stretch_min_long | 0.0 | **0.0** |
| pair_trend_filter_enabled | false | **false** |
| entry_quality_score_filter_enabled | true | **false** |

### filters — SHORT
| Variable | Global | BTC |
|---|---|---|
| ema_gap_threshold_short | 0.08 | **0.010** |
| ema_gap_5_8_max_short | 0.35 | **0.20** |
| momentum_short_rsi_max | 50 | **50** |
| momentum_adx_max (short) | 35 | **35** |
| adx_dir_short | "rising" | **"both"** |
| min_adx_delta_short | 0.0 | **0.0** |
| momentum_ema20_slope_min_short | 0.06 | **0.02** |
| momentum_ema20_slope_max_short | 0.40 | **0.30** |
| ema5_stretch_min_short | 0.0 | **0.0** |

### macro_filters
| Variable | Global | BTC |
|---|---|---|
| btc_adx_min_long | 15 | **15** |
| btc_adx_max_long | 40 | **40** |
| btc_adx_min_short | 18 | **18** |
| btc_adx_max_short | 40 | **40** |
| btc_rsi_adx_filter_long | (active) | **""** (DISABLE — self-referential, blocks 12/19 BTC trades) |
| btc_rsi_adx_filter_short | (active) | **""** (DISABLE) |
| btc_rsi_band_atr_block_long | "65-70:<0.10" | **""** (DISABLE) |
| btc_rsi_band_atr_block_short | "" | **""** |
| btc_trend_filter_enabled | false | **false** |
| global_volume_filter_enabled | true | **true** |
| global_volume_threshold_long | 0.7 | **0.7** |
| global_volume_threshold_short | 0.5 | **0.5** |
| global_volume_max_short | 1.1 | **1.1** |

### sizing
| Variable | Global | BTC |
|---|---|---|
| investment_multiplier | 1.0 | **0.5** |
| leverage | 20× | **10×** |

### directions
| Variable | Global | BTC |
|---|---|---|
| long_enabled | true | **true** |
| short_enabled | true | **true** |

### multipliers
| Variable | Global | BTC |
|---|---|---|
| Pattern Cell rules (C4/C8/W1/W2/W4/W6/C1/UNMATCHED) | active | **disabled for pair** |
| RSI×ADX multiplier cells | active | **disabled for pair** |

---

## ETHUSDT — COMPLETE OVERRIDE VALUES

### exits
| Variable | Global | ETH |
|---|---|---|
| tp_min | 0.45 | **0.25** |
| pullback_trigger | 0.25 | **0.18** |
| pullback_widening_per_level | 0.10 | **0.08** |
| stop_loss | -0.70 | **-0.50** |
| signal_active_sl | -0.70 | **-0.50** |
| be_levels_enabled | false | **true** |
| be_level1_trigger | 99 | **0.15** |
| be_level1_offset | 99 | **0.07** |
| fast_exit_enabled | false | **true** |
| fast_exit_threshold_pct | 0.20 | **0.18** |
| fast_exit_window_minutes | 2 | **3** |
| fast_exit_l1_atr_multiplier | 0.50 | **0.50** |
| fast_exit_l1_atr_floor_cap_pct | 0.60 | **0.40** |
| fast_exit_l2_enabled | false | **true** |
| fast_exit_l2_threshold_pct | 0.40 | **0.35** |
| fast_exit_l2_window_minutes | 5 | **5** |
| fast_exit_l2_atr_floor_cap_pct | 0.80 | **0.55** |
| signal_lost_exit_enabled | false | **false** |
| signal_lost_flag_security_min | -0.9 | **-0.50** |
| signal_lost_flag_security_max | -0.8 | **-0.40** |
| fl1_for_wide_sl_enabled | false | **false** |
| fl1_wide_sl_backstop | -0.95 | **-0.75** |
| ema13_cross_exit_enabled | true | **true** |
| rsi_handoff_active | false | **true** |
| rsi_handoff_level | 2 | **3** |
| trailing_atr_multiplier | 0.50 | **0.50** |

### filters — LONG
| Variable | Global | ETH |
|---|---|---|
| ema_gap_threshold_long | 0.06 | **0.020** |
| ema_gap_5_8_max_long | 0.35 | **0.30** |
| momentum_long_rsi_max | 65 | **65** |
| momentum_adx_max_long | 30 | **30** |
| adx_dir_long | "rising" | **"both"** |
| min_adx_delta_long | 0.0 | **0.0** |
| momentum_ema20_slope_min_long | 0.0 | **0.0** |
| momentum_ema20_slope_max_long | 0.40 | **0.35** |
| entry_dist_from_ema13_min_long | 0.20 | **0.10** |
| entry_dist_from_ema13_filter_enabled | true | **true** |
| ema5_stretch_min_long | 0.0 | **0.0** |
| pair_trend_filter_enabled | false | **false** |
| entry_quality_score_filter_enabled | true | **true** |

### filters — SHORT
| Variable | Global | ETH |
|---|---|---|
| ema_gap_threshold_short | 0.08 | **0.030** |
| ema_gap_5_8_max_short | 0.35 | **0.30** |
| momentum_short_rsi_max | 50 | **50** |
| momentum_adx_max (short) | 35 | **35** |
| adx_dir_short | "rising" | **"both"** |
| min_adx_delta_short | 0.0 | **0.0** |
| momentum_ema20_slope_min_short | 0.06 | **0.04** |
| momentum_ema20_slope_max_short | 0.40 | **0.35** |
| ema5_stretch_min_short | 0.0 | **0.0** |

### macro_filters
| Variable | Global | ETH |
|---|---|---|
| btc_adx_min_long | 15 | **15** |
| btc_adx_max_long | 40 | **40** |
| btc_adx_min_short | 18 | **18** |
| btc_adx_max_short | 40 | **40** |
| btc_rsi_adx_filter_long | (active) | **""** (DISABLE — blocks 8 ETH) |
| btc_rsi_adx_filter_short | (active) | **""** (DISABLE — blocks 5 ETH) |
| btc_rsi_band_atr_block_long | "65-70:<0.10" | **(keep global)** |
| btc_rsi_band_atr_block_short | "" | (keep global) |
| btc_trend_filter_enabled | false | **false** |
| global_volume_filter_enabled | true | **true** |
| global_volume_threshold_long | 0.7 | **0.7** |
| global_volume_threshold_short | 0.5 | **0.5** |
| global_volume_max_short | 1.1 | **1.1** |

### sizing
| Variable | Global | ETH |
|---|---|---|
| investment_multiplier | 1.0 | **1.0** |
| leverage | 20× | **15×** |

### directions
| Variable | Global | ETH |
|---|---|---|
| long_enabled | true | **true** |
| short_enabled | true | **true** |

### multipliers
| Variable | Global | ETH |
|---|---|---|
| Pattern Cell rules | active | **keep global** |
| RSI×ADX multiplier cells | active | **keep global** |

---

## SIDE-BY-SIDE SUMMARY

| Parameter | Global | BTC | ETH |
|---|---|---|---|
| tp_min | 0.45 | 0.08 | 0.25 |
| pullback_trigger | 0.25 | 0.05 | 0.18 |
| stop_loss | -0.70 | -0.35 | -0.50 |
| be_levels_enabled | false | true | true |
| be_level1_trigger/offset | 99/99 | 0.10/0.04 | 0.15/0.07 |
| fast_exit_enabled | false | true | true |
| fast_exit_threshold_pct | 0.20 | 0.08 | 0.18 |
| fast_exit_l2_enabled | false | true | true |
| fast_exit_l2_threshold_pct | 0.40 | 0.15 | 0.35 |
| rsi_handoff_active | false | false | true |
| leverage | 20× | 10× | 15× |
| investment_multiplier | 1.0 | 0.5 | 1.0 |
| ema_gap_threshold_long/short | 0.06/0.08 | 0.005/0.010 | 0.020/0.030 |
| ema_gap_5_8_max_long/short | 0.35/0.35 | 0.20/0.20 | 0.30/0.30 |
| momentum_ema20_slope_min_short | 0.06 | 0.02 | 0.04 |
| momentum_ema20_slope_max_long/short | 0.40 | 0.30 | 0.35 |
| entry_dist_from_ema13_min_long | 0.20 | 0.05 | 0.10 |
| adx_dir_long/short | "rising" | "both" | "both" |
| entry_quality_score_filter | true | false | true |
| btc_rsi_adx_filter_long/short | active | DISABLED | DISABLED |
| btc_rsi_band_atr_block_long | "65-70:<0.10" | DISABLED | keep |
| Pattern Cell rules | active | DISABLED | keep |
| long/short direction | both | both | both |

---

## BUILD PHASING

### Phase 1A — Infrastructure (single commit, ~566 LOC)
Ship the override mechanism + UI panel + initial BTC/ETH values above.

### Phase 1B — Observe (~30 fresh trades per pair, no code change)
| Metric | BTC pass | ETH pass |
|---|---|---|
| Trades/day | ≥ 3 (up from 1.8) | ≥ 4 |
| WR | ≥ 50% | ≥ 55% |
| Avg P&L % | ≥ 0% | ≥ +0.05% |
| Total $ | ≥ -$50 | ≥ +$100 |
| BE fire rate | ≥ 25% | ≥ 30% |
| FAST_EXIT fire rate | ≥ 15% | ≥ 15% |
| EMA13 cross loss rate | < 25% | < 20% |
| LONG side WR | ≥ 35% (loose) | ≥ 50% |

### Phase 1C — Locked revert criteria (per parameter)
| Override | Revert trigger |
|---|---|
| BTC tp_min 0.08 | TP fires ≥30% AND avg close <+0.06% AND post-exit peak >+0.20% → raise to 0.12 |
| BTC pullback 0.05 | trailing exits avg +0.05% AND post-exit peak >+0.30% → raise to 0.08 |
| BTC stop_loss -0.35 | SL fires ≥3 AND any single loss <-$50 full size → revert -0.50 |
| BTC btc_rsi_adx_filter disabled | BTC trades ≤30% WR on N≥10 → re-enable for BTC |
| BTC entry_quality_score disabled | Score 1 BTC trades ≤25% WR on N≥5 → re-enable |
| BTC LONG enabled | BTC LONG WR ≤30% on N≥10 → disable LONG (per-pair direction lock) |
| ETH analogues | Same gates, more lenient (ETH baseline better) |

### Phase 1D — Expand per-cell after validation
Validated params → consider Stage 2 loosening. BTC LONG fails → SHORT-only.

---

## OPEN QUESTIONS (resolve before build)

1. **btc_rsi_band_atr_block_long for ETH** — proposed keep global. Data:
   0 ETH trades blocked by it historically, so keep vs disable is moot.
   DEFAULT: keep global.

2. **leverage override (10× BTC / 15× ETH vs 20× global)** — changes margin.
   If you want 20× same as alts, only cushion left is investment_multiplier.
   DECISION NEEDED: reduced leverage on these pairs, or keep 20×?

3. **Pattern Cell rules disabled for BTC** — currently moot (zero historical
   BTC trades matched active cells). DEFAULT: disable for BTC safety.

---

## CRITICAL CAVEATS (do not lose these)

1. **BTC LONG has ZERO survivors under current filters** — opening BTC LONG
   via override means trading on zero recent comparable data. Only the
   historical 16-trade / 19% WR baseline applies (under looser filters).
   Forward edge UNKNOWN. The locked revert (LONG WR ≤30% on N≥10 → disable)
   is the safety net.

2. **The 1 ETH LONG that survives LOSES** — it's the chase-the-top profile
   (high RSI + high RngPos + bullish macro), a documented failure mode.
   More ETH LONGs via override could compound losses, not generate edge.

3. **Both pairs' winners share a SHORT profile.** If forced to phase,
   SHORT side has evidence (16 of 19 matching historical SHORTs won);
   LONG side does not.

4. **Anti-overfit:** BTC N=23, ETH N=25 cross-batch. Small samples.
   Survivor subsets are N=4 each. All forward projections need 30-50%
   in-sample-bias haircut.

5. **User explicitly directed: both LONG and SHORT enabled for both pairs.**
   Accepted as deliberate override of the data signal (which favors SHORT-only
   start). Locked revert criteria above are the mitigation.

---

## NOTHING SHIPPED YET. This is the design + locked initial values only.
When ready: build Phase 1A as a single commit, then observe per Phase 1B.
