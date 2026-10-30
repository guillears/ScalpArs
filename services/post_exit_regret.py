"""🧭 Post-Exit Regret Deep Dive (restored 2026-10-30, DECISION_LOG 419, operator: "one of the most important tables"; removed by DL 211).
The row computation is the pre-DL 211 main.py code, unchanged. One row per close_reason with tracked post-exit data, so a new exit name appears by itself;
post_exit_untracked lists the closed reasons without data so nothing is left out silently."""
import logging
import traceback

logger = logging.getLogger(__name__)


def post_exit_regret_rows(orders):
    post_exit_regret_deep_dive = []
    try:
        # Include every close reason that has post-exit tracking data.
        # The old hand-picked whitelist (BREAKEVEN_EXIT / SIGNAL_LOST / STOP_LOSS
        # / TICK_MOMENTUM_EXIT / RSI_MOMENTUM_EXIT / REGIME_CHANGE) was silently
        # dropping the majority of trades, including TRAILING_STOP (the single
        # most useful regret signal — "did we exit too early?"), the new FL2
        # family (FL_RECOVERED, FL_DEEP_STOP), FL1[WIDE_SL]'s FL_EMERGENCY_SL,
        # and FL_NO_EXPANSION.  Every row that has post_exit_peak_pnl tracked
        # is a candidate for regret analysis.
        pe_orders = [o for o in orders if o.post_exit_peak_pnl is not None and o.close_reason]
        if pe_orders:
            reason_groups = {}
            for o in pe_orders:
                reason_groups.setdefault(o.close_reason, []).append(o)

            for reason in sorted(reason_groups.keys()):
                group = reason_groups[reason]
                count = len(group)
                pe_longs = sum(1 for o in group if o.direction == "LONG")
                pe_shorts = count - pe_longs

                avg_duration_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in group if o.closed_at) / count if count > 0 else 0
                dur_h, dur_m, dur_s = int(avg_duration_secs // 3600), int((avg_duration_secs % 3600) // 60), int(avg_duration_secs % 60)

                avg_close_pnl = sum(o.pnl_percentage or 0 for o in group) / count
                avg_post_peak = sum(o.post_exit_peak_pnl or 0 for o in group) / count
                avg_peak_min = sum(o.post_exit_peak_minutes or 0 for o in group) / count
                avg_post_trough = sum(o.post_exit_trough_pnl or 0 for o in group) / count
                avg_trough_min = sum(o.post_exit_trough_minutes or 0 for o in group) / count
                avg_final = sum(o.post_exit_final_pnl or 0 for o in group) / count

                peak_first_count = sum(1 for o in group if (o.post_exit_peak_minutes or 0) < (o.post_exit_trough_minutes or 0))
                peak_first_pct = round(peak_first_count / count * 100, 1) if count > 0 else 0

                sig_lost_orders = [o for o in group if o.post_exit_signal_lost_minutes is not None]
                sig_lost_pct = round(len(sig_lost_orders) / count * 100, 1) if count > 0 else 0
                avg_sig_lost_min = sum(o.post_exit_signal_lost_minutes for o in sig_lost_orders) / len(sig_lost_orders) if sig_lost_orders else None
                avg_pnl_at_sig_lost = sum(o.post_exit_pnl_at_signal_lost or 0 for o in sig_lost_orders) / len(sig_lost_orders) if sig_lost_orders else None
                reachable_peak_orders = [o for o in group if o.post_exit_peak_before_signal_lost is not None]
                avg_reachable_peak = sum(o.post_exit_peak_before_signal_lost for o in reachable_peak_orders) / len(reachable_peak_orders) if reachable_peak_orders else None

                rsi_exit_orders = [o for o in group if o.post_exit_rsi_exit_minutes is not None]
                rsi_exit_pct = round(len(rsi_exit_orders) / count * 100, 1) if count > 0 else 0
                avg_rsi_exit_min = sum(o.post_exit_rsi_exit_minutes for o in rsi_exit_orders) / len(rsi_exit_orders) if rsi_exit_orders else None
                avg_rsi_exit_pnl = sum(o.post_exit_rsi_exit_pnl or 0 for o in rsi_exit_orders) / len(rsi_exit_orders) if rsi_exit_orders else None

                ema13_cross_orders = [o for o in group if getattr(o, 'post_exit_ema13_cross_minutes', None) is not None]
                ema13_cross_pct = round(len(ema13_cross_orders) / count * 100, 1) if count > 0 else 0
                avg_ema13_cross_min = sum(o.post_exit_ema13_cross_minutes for o in ema13_cross_orders) / len(ema13_cross_orders) if ema13_cross_orders else None
                avg_ema13_cross_pnl = sum(o.post_exit_ema13_cross_pnl or 0 for o in ema13_cross_orders) / len(ema13_cross_orders) if ema13_cross_orders else None

                # May 23: post-exit regime-flip aggregation. RegFlipMin =
                # minutes from close until BTC regime first stopped supporting
                # trade direction. P&L@RegFlip = running P&L at that moment.
                # NULL/dash = regime never flipped in tracking window (= original
                # exit was structurally correct; holding would have ridden into
                # the post-exit trough). The comparison column for the
                # "EMA13 vs regime exit" question.
                regime_flip_orders = []
                for o in group:
                    _rf_at = getattr(o, 'post_exit_regime_flip_at', None)
                    _exit_at = getattr(o, 'closed_at', None)
                    if _rf_at is not None and _exit_at is not None:
                        regime_flip_orders.append(o)
                regime_flip_pct = round(len(regime_flip_orders) / count * 100, 1) if count > 0 else 0
                if regime_flip_orders:
                    _rf_minutes = []
                    _rf_pnls = []
                    for o in regime_flip_orders:
                        try:
                            _dt = (o.post_exit_regime_flip_at - o.closed_at).total_seconds() / 60.0
                            _rf_minutes.append(_dt)
                        except Exception:
                            pass
                        _pnl = getattr(o, 'post_exit_regime_flip_pnl_pct', None)
                        if _pnl is not None:
                            _rf_pnls.append(_pnl)
                    avg_regime_flip_min = sum(_rf_minutes) / len(_rf_minutes) if _rf_minutes else None
                    avg_regime_flip_pnl = sum(_rf_pnls) / len(_rf_pnls) if _rf_pnls else None
                else:
                    avg_regime_flip_min = None
                    avg_regime_flip_pnl = None

                rec_neg020 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.20)
                rec_neg010 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.10)
                rec_neg005 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.05)
                rec_005 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.05)
                rec_010 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.10)

                sig_regained_orders = [o for o in group if o.post_exit_signal_regained_minutes is not None]
                sig_regained_pct = round(len(sig_regained_orders) / count * 100, 1) if count > 0 else 0
                avg_sig_regained_min = sum(o.post_exit_signal_regained_minutes for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None
                avg_pnl_at_sig_regained = sum(o.post_exit_pnl_at_signal_regained or 0 for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None

                floor_orders = [o for o in sig_regained_orders if o.post_exit_floor_before_signal_regain is not None]
                avg_floor_before_sig_regain = sum(o.post_exit_floor_before_signal_regain for o in floor_orders) / len(floor_orders) if floor_orders else None

                ema5_dists_pe = [o.exit_price_vs_ema5_pct for o in group if o.exit_price_vs_ema5_pct is not None]
                ema5_slopes_pe = [o.exit_ema5_slope_pct for o in group if o.exit_ema5_slope_pct is not None]
                ema5_crossed_pe = sum(1 for o in group if o.exit_ema5_crossed is True)

                # May 22: ATR aggregation for sl_atr_multiplier diagnostic.
                # AvgATR% shows volatility character of the bucket.
                # ATR-SL@1.5× shows what wider SL would have been; compare against AvgClose
                # to gauge whether ATR widening would have rescued the trade.
                atrs_pe = [o.entry_atr_pct for o in group if o.entry_atr_pct is not None and o.entry_atr_pct > 0]
                avg_atr_pe = sum(atrs_pe) / len(atrs_pe) if atrs_pe else None
                atr_sl_widened_pe = -(avg_atr_pe * 1.5) if avg_atr_pe is not None else None

                # May 31: stretch-fade recoverable-regret band (Leash Shadow strpk/stren).
                # strpk = stretch-trail (aggressive ceiling), stren = stretch-to-entry
                # (conservative floor). Both run the ACTUAL post-exit path (respect the
                # sequence-trap, unlike PostPeak%). Read Close% → [stren, strpk] → PostPeak%.
                # Only armed post-deploy trades have shadow data; NULL/dash otherwise.
                strpk_pe = [o.shadow_strpk_pnl for o in group if getattr(o, 'shadow_strpk_pnl', None) is not None]
                stren_pe = [o.shadow_stren_pnl for o in group if getattr(o, 'shadow_stren_pnl', None) is not None]
                avg_strpk_pe = sum(strpk_pe) / len(strpk_pe) if strpk_pe else None
                avg_stren_pe = sum(stren_pe) / len(stren_pe) if stren_pe else None
                # May 31: looser-K stretch-trail variants (0.4 / 0.3) — holds runners longer
                strpk04_pe = [o.shadow_strpk04_pnl for o in group if getattr(o, 'shadow_strpk04_pnl', None) is not None]
                strpk03_pe = [o.shadow_strpk03_pnl for o in group if getattr(o, 'shadow_strpk03_pnl', None) is not None]
                avg_strpk04_pe = sum(strpk04_pe) / len(strpk04_pe) if strpk04_pe else None
                avg_strpk03_pe = sum(strpk03_pe) / len(strpk03_pe) if strpk03_pe else None
                # fire-minute (from open); compare to Duration → pre/post-close
                strpk_min_pe = [o.shadow_strpk_min for o in group if getattr(o, 'shadow_strpk_min', None) is not None]
                stren_min_pe = [o.shadow_stren_min for o in group if getattr(o, 'shadow_stren_min', None) is not None]
                avg_strpk_min = sum(strpk_min_pe) / len(strpk_min_pe) if strpk_min_pe else None
                avg_stren_min = sum(stren_min_pe) / len(stren_min_pe) if stren_min_pe else None

                no_regain = [o for o in group if o.post_exit_signal_regained_minutes is None]
                nr_count = len(no_regain)
                nr_rec_neg020 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.20) if nr_count else 0
                nr_rec_neg010 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.10) if nr_count else 0
                nr_rec_neg005 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.05) if nr_count else 0

                post_exit_regret_deep_dive.append({
                    "reason": reason,
                    "count": count,
                    "longs": pe_longs,
                    "shorts": pe_shorts,
                    "avg_duration": f"{dur_h:02d}:{dur_m:02d}:{dur_s:02d}",
                    "avg_close_pnl": round(avg_close_pnl, 4),
                    "avg_post_peak": round(avg_post_peak, 4),
                    "avg_peak_min": round(avg_peak_min, 1),
                    "avg_post_trough": round(avg_post_trough, 4),
                    "avg_trough_min": round(avg_trough_min, 1),
                    "peak_first_pct": peak_first_pct,
                    "avg_final_pnl": round(avg_final, 4),
                    "sig_lost_pct": sig_lost_pct,
                    "avg_sig_lost_min": round(avg_sig_lost_min, 1) if avg_sig_lost_min is not None else None,
                    "avg_pnl_at_sig_lost": round(avg_pnl_at_sig_lost, 4) if avg_pnl_at_sig_lost is not None else None,
                    "avg_reachable_peak": round(avg_reachable_peak, 4) if avg_reachable_peak is not None else None,
                    "rsi_exit_pct": rsi_exit_pct,
                    "avg_rsi_exit_min": round(avg_rsi_exit_min, 1) if avg_rsi_exit_min is not None else None,
                    "avg_rsi_exit_pnl": round(avg_rsi_exit_pnl, 4) if avg_rsi_exit_pnl is not None else None,
                    "ema13_cross_pct": ema13_cross_pct,
                    "avg_ema13_cross_min": round(avg_ema13_cross_min, 1) if avg_ema13_cross_min is not None else None,
                    "avg_ema13_cross_pnl": round(avg_ema13_cross_pnl, 4) if avg_ema13_cross_pnl is not None else None,
                    # May 23: post-exit regime-flip diagnostic (EMA13 vs regime exit comparison)
                    "regime_flip_pct": regime_flip_pct,
                    "avg_regime_flip_min": round(avg_regime_flip_min, 1) if avg_regime_flip_min is not None else None,
                    "avg_regime_flip_pnl": round(avg_regime_flip_pnl, 4) if avg_regime_flip_pnl is not None else None,
                    "recovery_neg020_pct": round(rec_neg020 / count * 100, 1),
                    "recovery_neg010_pct": round(rec_neg010 / count * 100, 1),
                    "recovery_neg005_pct": round(rec_neg005 / count * 100, 1),
                    "recovery_005_pct": round(rec_005 / count * 100, 1),
                    "recovery_010_pct": round(rec_010 / count * 100, 1),
                    "sig_regained_pct": sig_regained_pct,
                    "avg_sig_regained_min": round(avg_sig_regained_min, 1) if avg_sig_regained_min is not None else None,
                    "avg_pnl_at_sig_regained": round(avg_pnl_at_sig_regained, 4) if avg_pnl_at_sig_regained is not None else None,
                    "avg_floor_before_sig_regain": round(avg_floor_before_sig_regain, 4) if avg_floor_before_sig_regain is not None else None,
                    "nr_count": nr_count,
                    "nr_rec_neg020_pct": round(nr_rec_neg020 / nr_count * 100, 1) if nr_count > 0 else None,
                    "nr_rec_neg010_pct": round(nr_rec_neg010 / nr_count * 100, 1) if nr_count > 0 else None,
                    "nr_rec_neg005_pct": round(nr_rec_neg005 / nr_count * 100, 1) if nr_count > 0 else None,
                    "avg_exit_ema5_dist": round(sum(ema5_dists_pe) / len(ema5_dists_pe), 4) if ema5_dists_pe else None,
                    "avg_exit_ema5_slope": round(sum(ema5_slopes_pe) / len(ema5_slopes_pe), 4) if ema5_slopes_pe else None,
                    "exit_ema5_crossed_pct": round(ema5_crossed_pe / count * 100, 1) if count > 0 else None,
                    "avg_atr_pct": round(avg_atr_pe, 4) if avg_atr_pe is not None else None,
                    "atr_sl_15x": round(atr_sl_widened_pe, 4) if atr_sl_widened_pe is not None else None,
                    # May 31: stretch-fade recoverable-regret band (observation-only)
                    "avg_strpk_pct": round(avg_strpk_pe, 4) if avg_strpk_pe is not None else None,
                    "avg_strpk04_pct": round(avg_strpk04_pe, 4) if avg_strpk04_pe is not None else None,
                    "avg_strpk03_pct": round(avg_strpk03_pe, 4) if avg_strpk03_pe is not None else None,
                    "avg_stren_pct": round(avg_stren_pe, 4) if avg_stren_pe is not None else None,
                    "avg_strpk_min": round(avg_strpk_min, 1) if avg_strpk_min is not None else None,
                    "avg_stren_min": round(avg_stren_min, 1) if avg_stren_min is not None else None,
                })
    except Exception as e:
        logger.error(f"[PERF] Error computing Post-Exit Regret deep dive: {e}\n{traceback.format_exc()}")

    return post_exit_regret_deep_dive


def post_exit_untracked(orders):
    """closed fills with a close_reason and no post-exit data yet, per reason (tracking window still running, or closed before this exit was tracked)."""
    n = {}
    for o in orders:
        if o.close_reason and getattr(o, 'status', 'CLOSED') == 'CLOSED' and o.post_exit_peak_pnl is None:
            n[o.close_reason] = n.get(o.close_reason, 0) + 1
    return [{"reason": k, "n": n[k]} for k in sorted(n)]
