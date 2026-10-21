"""🔥 FRENZY sleeve (Oct-2, DECISION_LOG 176) — pure rules shared by the engine and the tests. No I/O.

A pair is FLAGGED after a volume spike and followed for up to 96 h. While flagged, a LONG opens when the "staircase" state turns
on (the research rule of scripts/staircase_long_tight_test.py / frenzy_long_followup.py) and the pair's 5m ATR is small enough
for the fixed stop; every close below the 5m EMA50 / EMA200 of a flagged pair is recorded as a SHORT observation (no trade).

  spike     a 5m close with 30-min return ≥ frenzy_spike_ret_pct ∧ last-hour quote volume ≥ frenzy_spike_vol_mult × the pair's
            normal hour ∧ ≥ frenzy_spike_min_hour_usd. The spike-anchored VWAP starts at that bar.
  episode   starts at a spike bar; ends when 24 h pass without a state bar — the next spike bar after that starts a new one.
  state     ≥ frenzy_min_hours after the spike ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥
            frenzy_state_vol_mult × normal.
  entry     the state turns ON after ≥ 1 h off (the first state bar of a stretch) ∧ 24 h volume ≥ frenzy_min_volume_usd ∧
            5m ATR(14) ≤ frenzy_max_atr_pct ∧ (frenzy_long_skip_green_bar) that signal bar closed at or below its open — judged on
            that bar only (DECISION_LOG 180: after a red / flat bar +0.47 %/trade, after a green bar −0.11).
  exit      stop −frenzy_stop_pct · once the peak reaches +frenzy_trail_arm_pct, close frenzy_trail_giveback_pct below the best
            price · frenzy_max_hold_minutes. P&L is the caller's net-of-fees % of the position.
Everything is recomputed from a fresh kline window on every evaluation, so a deploy needs no warm-up; the engine persists only
the NAMES of the flagged pairs (table frenzy_flags) so a cooled pair is re-checked after a restart.
Quote volume is taken as volume × typical price (the exchange OHLCV rows carry base volume)."""
import math
from statistics import median
from typing import List, Optional, Tuple

BAR_MS = 300_000
HOUR_MS = 3_600_000
GAP_MS = 24 * HOUR_MS          # an episode ends after this long without a state bar
VERIFY_MS = GAP_MS + 12 * BAR_MS   # history needed before a spike to know it is the first one of its episode


def _f(th, name, default):
    try:
        v = getattr(th, name, default)
        return float(default) if v is None else float(v)
    except (TypeError, ValueError):
        return float(default)


def normal_hour_usd(hour_bars, last_closed_ms: int) -> Optional[float]:
    """The pair's normal hourly quote volume: the median of full 1h bars that ENDED at least 24 h before the last closed 5m bar
    (so the frenzy itself never inflates it). None when the pair has < 240 such hours (listed < ~10 days) or the data is bad."""
    try:
        q = [float(r[5]) * (float(r[2]) + float(r[3]) + float(r[4])) / 3.0 for r in (hour_bars or [])
             if r and len(r) >= 6 and int(r[0]) + HOUR_MS <= int(last_closed_ms) - GAP_MS]
        if len(q) < 240:
            return None
        m = median(q[-720:])
        return m if m > 0 else None
    except (TypeError, ValueError, IndexError):
        return None


def frenzy_walk(bars, norm_hour, th) -> Optional[dict]:
    """bars = CLOSED 5m OHLCV rows [open_ms, o, h, l, c, v], oldest first. Returns the LIVE episode (the one still running at the
    last bar) or None. Keys: spike_ts (close time of the spike bar), base (price 30 min before it), vwap, price, peak, hours,
    vol_mult, above_hour, above_streak (closes in a row at or above the average price, ending at the last bar), in_state,
    fresh_on (the last bar is the first state bar after ≥ 1 h off), verified, last_bar_ts, on_bar_ts (⏪ Oct-6: open ms of the bar the
    current state stretch turned on — == last_bar_ts exactly when fresh_on; None when not in state),
    vs_vwap_pct, run_pct, gain_pct, off_peak_pct, bar_ret_pct (the last bar's close vs its open, %) and bar_red (close ≤ open). Fail-closed.
    verified = the spike is known to be its episode's FIRST: either the previous episode is fully inside the window and was itself
    verified, or the 25 h before the spike are visible and hold no bar at the state volume (a state bar needs that volume, so an
    older episode whose own spike already left the window cannot still be alive — review: a 5-day frenzy re-anchored on a
    mid-episode spike once its first spike scrolled out)."""
    try:
        if not bars or len(bars) < 40 or not norm_hour or not norm_hour > 0:
            return None
        t = [int(r[0]) for r in bars]; h = [float(r[2]) for r in bars]; l = [float(r[3]) for r in bars]; c = [float(r[4]) for r in bars]
        try:   # an unreadable open only refuses the ENTRY (fail-closed below) — it must never drop the pair's flag
            o_last = float(bars[-1][1])
        except (TypeError, ValueError, IndexError):
            o_last = 0.0
        q = [float(r[5]) * (h[i] + l[i] + c[i]) / 3.0 for i, r in enumerate(bars)]
        n = len(c)
        ret_min, vol_min, hour_min = _f(th, 'frenzy_spike_ret_pct', 5.0), _f(th, 'frenzy_spike_vol_mult', 20.0), _f(th, 'frenzy_spike_min_hour_usd', 2e6)
        state_vol, min_bars = _f(th, 'frenzy_state_vol_mult', 100.0), max(12, int(round(_f(th, 'frenzy_min_hours', 2.0) * 12)))
        if ret_min <= 0 or vol_min <= 0 or state_vol <= 0:
            return None
        q1h = [None] * n; run = 0.0
        for i in range(n):
            run += q[i]
            if i >= 12:
                run -= q[i - 12]
            if i >= 11:
                q1h[i] = run
        volx = [(x / norm_hour) if x is not None else None for x in q1h]
        lead = [i for i in range(11, n) if c[i - 6] > 0 and (c[i] / c[i - 6] - 1) * 100 >= ret_min and volx[i] >= vol_min and q1h[i] >= hour_min]
        i = 0; chained = False                              # chained = the previous in-window episode was verified and has ended
        for on in lead:
            if on < i:
                continue                                    # inside the previous episode
            verified = chained or bool(t[on] - t[0] >= VERIFY_MS and not any(
                volx[k] is not None and volx[k] >= state_vol for k in range(max(0, on - VERIFY_MS // BAR_MS), on)))
            pv = qq = 0.0; vw = []; above = []; state = []; streak = 0; n_up = 0
            for j in range(on, n):
                pv += (h[j] + l[j] + c[j]) / 3.0 * q[j]; qq += q[j]
                v = pv / qq if qq > 0 else c[j]; vw.append(v)
                streak = streak + 1 if c[j] >= v else 0
                n_up += 1 if c[j] >= v else 0
                ab = streak >= 12; above.append(ab)
                state.append(bool(ab and volx[j] is not None and volx[j] >= state_vol and (j - on) >= min_bars))
            last = on; end = None
            for j in range(on + 1, n):
                if t[j] - t[last] > GAP_MS:
                    end = j; break
                if state[j - on]:
                    last = j
            if end is not None:
                i = end; chained = verified; continue
            k = n - 1 - on
            fresh = bool(state[k] and not any(state[max(0, k - 12):k]))
            on_ts = None   # ⏪ Oct-6 catch-up: the bar the CURRENT state stretch turned on (its first state bar after ≥ 1 h off) — None when not in state
            if state[k]:
                for jj in range(k, -1, -1):
                    if state[jj] and not any(state[max(0, jj - 12):jj]):
                        on_ts = t[on + jj]; break
            base = c[on - 6]; peak = max(h[on:]); price = c[-1]
            return dict(spike_ts=t[on] + BAR_MS, base=base, vwap=vw[-1], price=price, peak=peak,
                        hours=(t[-1] - t[on]) / HOUR_MS, vol_mult=volx[-1], above_hour=bool(above[-1]), above_streak=streak, in_state=bool(state[-1]),
                        fresh_on=fresh, verified=verified, last_bar_ts=t[-1], on_bar_ts=on_ts,
                        vs_vwap_pct=(price / vw[-1] - 1) * 100 if vw[-1] > 0 else None,
                        run_pct=(peak / base - 1) * 100, gain_pct=(price / base - 1) * 100,
                        off_peak_pct=(price / peak - 1) * 100 if peak > 0 else None,
                        bar_ret_pct=((price / o_last - 1) * 100 if o_last > 0 else None), bar_red=bool(o_last > 0 and price <= o_last),
                        above_share=100.0 * n_up / (n - on))   # 🌀 Oct-5 (215): % of the episode's 5m closes at / above the spike-anchored VWAP
        return None
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def frenzy_di_spread(bars) -> Optional[float]:
    """🔬 Oct-2 OBSERVE-ONLY stamp (DECISION_LOG 187): +DI − −DI on the LAST row of CLOSED 5m bars [open_ms, o, h, l, c, v] — the bot's own
    ADX(14) (ta ADXIndicator, as services/indicators.py). Positive = up-moves dominate. Backtest (FRENZY_LONG_INDICATORS_2026-10-02.md):
    on red / flat signal candles the trades with the highest spread did best (+0.17 → +0.93 %/trade, lowest → highest fifth) but the read
    sits at the 75th percentile of luck → watch item for the 40-fill review. 💪 Oct-4 (DECISION_LOG 197, operator override): > 0 together with
    frenzy_adx_delta > 0 sizes a FRENZY_LONG at frenzy_long_lev_mult_strong. None when unreadable."""
    try:
        import math
        import pandas as pd
        from ta.trend import ADXIndicator
        if not bars or len(bars) < 60:
            return None
        h, l, c = (pd.Series([float(r[k]) for r in bars]) for k in (2, 3, 4))
        a = ADXIndicator(high=h, low=l, close=c, window=14)
        v = float(a.adx_pos().iloc[-1] - a.adx_neg().iloc[-1])
        return round(v, 3) if math.isfinite(v) else None
    except Exception:
        return None


def frenzy_adx_delta(bars) -> Optional[float]:
    """🔬 Oct-3 OBSERVE-ONLY stamp (DECISION_LOG 193): ADX(14) on the last CLOSED 5m bar minus ADX three bars earlier — the backtest's d_adx
    ("ADX rising", the operator's manual-trade read). On FRENZY-WIDE's year: rising +0.108 %/trade vs +0.054 all, both halves positive, but
    random subsets of the same size beat it 28 % of the time → a 40-fill watch item. 💪 Oct-4 (DECISION_LOG 197, operator override): > 0
    together with frenzy_di_spread > 0 sizes a FRENZY_LONG at frenzy_long_lev_mult_strong. None when unreadable."""
    try:
        import math
        import pandas as pd
        from ta.trend import ADXIndicator
        if not bars or len(bars) < 60:
            return None
        h, l, c = (pd.Series([float(r[k]) for r in bars]) for k in (2, 3, 4))
        a = ADXIndicator(high=h, low=l, close=c, window=14).adx()
        v = float(a.iloc[-1] - a.iloc[-4])
        return round(v, 3) if math.isfinite(v) else None
    except Exception:
        return None


def frenzy_vol_trend(bars) -> Optional[float]:
    """🔬 Oct-3 OBSERVE-ONLY stamp (DECISION_LOG 193): quote volume (≈ base volume × close — ccxt bars carry no quote column; the backtest
    used Binance's own quote volume) of the last 12 CLOSED 5m bars ÷ the 12 before them (> 1 = volume rising,
    the backtest's vol_trend). FRENZY-WIDE's year: rising +0.099 %/trade, random-subset luck 21 % → watch item only. None when unreadable."""
    try:
        if not bars or len(bars) < 24:
            return None
        q = [float(r[5]) * float(r[4]) for r in bars[-24:]]
        prev, last = sum(q[:12]), sum(q[12:])
        return round(last / prev, 3) if prev > 0 else None
    except Exception:
        return None


def global_volume_ratio(bars_by_pair, bar_open_ms, lookback: int = 48, min_pairs: int = 30) -> Optional[float]:
    """🌊 Oct-3 (DECISION_LOG 194): the market's volume on ONE closed 5m bar vs normal — Σ base volume of that bar ÷ Σ each pair's mean volume
    over the `lookback` bars ending at it (the engine's global volume ratio, but on the CLOSED signal bar, as the year test). bars_by_pair =
    {pair: [[open_ms, o, h, l, c, v], …]}. A pair counts only when it has that exact bar and the full lookback. None below min_pairs (never raises).
    Year (scripts/frenzy_global_volume_test.py): FRENZY + WIDE first candles at < 1.0 +0.225 %/trade, at ≥ 1.0 −0.185; a one-bar-older reading
    halves the edge (+0.126) → the signal bar itself, read at the close."""
    try:
        sv = sa = 0.0; n = 0
        for rows in (bars_by_pair or {}).values():
            try:
                ts = [int(r[0]) for r in rows]
                if bar_open_ms not in ts:
                    continue
                j = ts.index(bar_open_ms)
                if j < lookback - 1:
                    continue
                vols = [float(r[5]) for r in rows[j - lookback + 1:j + 1]]
                a = sum(vols) / lookback
                if a > 0:
                    sv += vols[-1]; sa += a; n += 1
            except (TypeError, ValueError, IndexError):
                continue
        return round(sv / sa, 4) if n >= min_pairs and sa > 0 else None
    except Exception:
        return None


FRENZY_WIDE_CODES = ("FRENZY_ATR_HIGH", "FRENZY_GREEN_BAR")   # the only FRENZY refusals FRENZY-WIDE takes


def frenzy_wide_ready(ep, code, th, atr_pct=None) -> bool:
    """🔥🌐 Oct-3 FRENZY-WIDE: True when FRENZY refused this fresh setup ONLY for its ATR cap or a green signal candle (frenzy_long_status
    judges those two LAST, so every earlier gate — setup ON, fresh bar, 24 h volume — already passed) and the WIDE switch is on. An UNREADABLE
    ATR (FRENZY_ATR_HIGH "ATR unreadable") is refused: the backtest always had one — fail closed (review)."""
    return (bool(getattr(th, 'frenzy_wide_enabled', False)) and bool(ep and ep.get('fresh_on')) and code in FRENZY_WIDE_CODES
            and atr_pct is not None)


def frenzy_wide_shadow(ep, code, th, atr_pct=None) -> bool:
    """👁 2026-10-14 (DECISION_LOG 291, WIDE switched OFF): True when frenzy_wide_enabled is OFF and the setup is one frenzy_wide_ready would
    have taken with the switch ON (same fresh / code / readable-ATR test). OBSERVE-ONLY — the engine only counts (FRENZY_WIDE_OFF) and logs it
    for the DL 291 revert gate; nothing opens. The later in-open checks (slots, day cap, market volume, choppy / hold-green, dislocation)
    are NOT applied, so this is the setup cohort, an upper bound on the fills WIDE would have made."""
    return (not bool(getattr(th, 'frenzy_wide_enabled', False)) and bool(ep and ep.get('fresh_on')) and code in FRENZY_WIDE_CODES
            and atr_pct is not None)


def frenzy_wide_choppy(ep, th) -> bool:
    """🌀 Oct-5 (DECISION_LOG 215, operator ARMED override): True = FRENZY_WIDE refuses this fresh setup because the pump is CHOPPY / FADING —
    at most frenzy_wide_above_share_min % of the episode's 5m closes (spike bar → signal bar) held at / above the spike-anchored VWAP, i.e. most
    spike buyers are under water. Year (617 live-gated WIDE first entries, live lock, 12 s entry): blocked 124 · 40 % WR · −0.87 %/trade, all 9
    months negative; picked independently in both halves; out-of-sample −0.40 % (day CI −0.99…+0.23 → fails only the 95 % leg). 0 = off.
    FAIL-OPEN: an unreadable share never blocks."""
    try:
        mn = _f(th, 'frenzy_wide_above_share_min', 0.0)
        a = (ep or {}).get('above_share')
        return bool(mn > 0 and a is not None and float(a) <= mn)
    except (TypeError, ValueError):
        return False


def frenzy_wide_hold_green_block(ep, code, th) -> Optional[str]:
    """🟢 Oct-6 (DECISION_LOG 231, operator discipline-override probe): when frenzy_wide_hold_green_streak > 0, FRENZY_WIDE takes ONLY the
    "hold-green" refusals — FRENZY_GREEN_BAR (ATR within the cap) on a setup whose price had ALREADY closed above the spike VWAP for MORE than
    that many 5m closes in a row at the signal bar (RLC 10-05: streak 17). → the block counter name, or None (= WIDE may open).
    FRENZY_ATR_HIGH → "FRENZY_WIDE_ATR_HIGH" (year: −0.53 %/fill, day CI below 0) · a green "reclaim" bar (streak ≤ min: the 12th close lands
    on the signal bar) → "FRENZY_WIDE_RECLAIM" (year: −0.66 %, 8/9 months negative). Kept hold-green: 124 · +0.43 %, day CI −0.15…+1.01,
    top 5 days = 96 % of the net — NOT a proven edge. FAIL-CLOSED: an unreadable streak blocks. 0 = off (WIDE as before)."""
    try:
        mn = _f(th, 'frenzy_wide_hold_green_streak', 0.0)
        if mn <= 0:
            return None
        if code != "FRENZY_GREEN_BAR":
            return "FRENZY_WIDE_ATR_HIGH"
        if (ep or {}).get('bar_ret_pct') is None:   # FRENZY_GREEN_BAR also covers "signal candle unreadable" — fail closed (review)
            return "FRENZY_WIDE_RECLAIM"
        s = (ep or {}).get('above_streak')
        return None if (s is not None and float(s) > mn) else "FRENZY_WIDE_RECLAIM"
    except (TypeError, ValueError):
        return "FRENZY_WIDE_RECLAIM"


def frenzy_gvol_block(gv, th) -> Optional[str]:
    """🌊 The frenzy_gvol_max gate as a pure rule (DECISION_LOG 194; shared by FRENZY_LONG / WIDE / LITE since 243): None = pass ·
    "GVOL_UNREAD" (gate on, reading missing / not a number / NaN / ±inf — fail-closed) · "GVOL_HIGH" (reading ≥ frenzy_gvol_max).
    Gate off (≤ 0) → None (the reading is then only stamped)."""
    import math
    try:
        mx = _f(th, 'frenzy_gvol_max', 0.0)
        if mx <= 0:
            return None
        if gv is None or not math.isfinite(float(gv)):
            return "GVOL_UNREAD"
        return "GVOL_HIGH" if float(gv) >= mx else None
    except (TypeError, ValueError):
        return "GVOL_UNREAD"


def _finite_or_none(v) -> Optional[float]:
    import math
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def frenzy_bearish_day(btc_1d_ret_pct, btc_trend_gap_pct) -> Optional[bool]:
    """🐻 Oct-8 (DECISION_LOG 250; definition frozen at its registration, DECISION_LOG 239): a BEARISH DAY = BTC's last closed daily return
    < 0 ∧ BTC's 5m trend gap (EMA13 − EMA50, %) < 0 — the two values the fill would be stamped with (entry_btc_1d_ret_pct /
    entry_btc_trend_gap_pct). Three-valued: True (both legs < 0) · False (a READABLE leg is ≥ 0 — decided whatever the other says) · None
    (not decidable: a leg unreadable / NaN / ±inf and every readable leg < 0)."""
    r, g = _finite_or_none(btc_1d_ret_pct), _finite_or_none(btc_trend_gap_pct)
    if (r is not None and r >= 0) or (g is not None and g >= 0):
        return False
    if r is None or g is None:
        return None
    return True


def frenzy_bearish_block(btc_1d_ret_pct, btc_trend_gap_pct, th) -> Optional[str]:
    """🐻 The frenzy_bearish_day_block gate as a pure rule (FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE): None = pass · "BEARISH_DAY" (refuse)
    · "BEARISH_UNREAD" (not decidable → FAIL-OPEN: the entry proceeds; the caller counts it). Gate off → None."""
    if not bool(getattr(th, 'frenzy_bearish_day_block', False)):
        return None
    b = frenzy_bearish_day(btc_1d_ret_pct, btc_trend_gap_pct)
    return "BEARISH_DAY" if b is True else ("BEARISH_UNREAD" if b is None else None)


# ── 🔥🪶 Oct-7 FRENZY_LITE (DECISION_LOG 243 — operator ARMED as a DECLARED EXCEPTION below the locked gates) ──────────────────────────
# FRENZY without the ≥ frenzy_state_vol_mult setup volume, limited to the first frenzy_lite_max_hours of the episode: a verified, flagged
# episode whose price has closed at / above the spike VWAP for ≥ frenzy_lite_min_above_closes 5m closes in a row while the last-hour
# volume is BELOW frenzy_state_vol_mult (FRENZY would say FRENZY_VOL_FADED), frenzy_min_hours ≤ hours ≤ frenzy_lite_max_hours. ONE entry
# per above-VWAP stretch (stretch id = open ms of the first bar of the current above streak), JUDGED ONCE: the first just-closed bar that
# meets the SIGNAL conditions (flagged, not in state, streak, volume < setup ×, hours window, 24 h volume) is the stretch's only chance —
# refused by ANY later filter (green candle, market volume, dislocation, pair-day cap) or open-path refusal (slots, pair held, late, open
# refused / failed) = the stretch is done (engine parity with the backtest cohort, reports/FRENZY_LITE_ABOVE_CLOSES_6_VS_12_2026-10-07.md:
# retrying later bars until a fill turns +0.163 %/trade into −0.005 on 1,992 fills). Only a 24 h volume refusal (a pre-signal condition) and
# a crash in the LITE path itself leave the stretch open. NO ATR filter (ATR only stamped — the scout's LITE_ATR tracker reads it). Evidence: reports/HOLD_LOWVOL_EARLY_FRENZY_FILTERS_2026-10-06.md
# "STACK minus F1": 724 fills, +0.163 %/trade at live timing, day CI [−0.107, +0.421], halves +0.156 / +0.168 → UNPROVEN; FRENZY lock exit
# (HOLD_LOWVOL_EARLY_EXITS_ROUND2_2026-10-06.md). REACHABLE expectation (243 review): the engine's FRENZY shortlist (|24 h change| or range ≥
# frenzy_shortlist_change_pct, ≤ 25 pairs) cannot reach 21 of the 724 study fills (avg +1.49 %, ~26 % of the P&L) and the engine-pair-level
# stretch id refuses 8 re-anchor fills (avg −1.23 %) → reachable ≈ 696 fills at +0.134 %/trade BEFORE the 30–50 % in-sample haircut. No automatic off — review at ≥ 40 fills on ≥ 15 days; avg < 0 at ≥ 20 → operator review.
FRENZY_LITE_READY = "FRENZY_LITE_READY"
FRENZY_LITE_NEED_DEFAULT, FRENZY_LITE_HMAX_DEFAULT = 12, 16.75


def frenzy_lite_need(th) -> int:
    """🪶 the closes-above requirement actually used: frenzy_lite_min_above_closes, ≤ 0 / unreadable → the shipped 12 (243 review)."""
    try:
        n = int(round(float(getattr(th, 'frenzy_lite_min_above_closes', FRENZY_LITE_NEED_DEFAULT))))
        return n if n >= 1 else FRENZY_LITE_NEED_DEFAULT
    except (TypeError, ValueError):
        return FRENZY_LITE_NEED_DEFAULT


def frenzy_lite_hmax(th) -> float:
    """🪶 the episode-age window end actually used: frenzy_lite_max_hours, ≤ frenzy_min_hours / unreadable → the shipped 16.75 (243 review)."""
    try:
        h = float(getattr(th, 'frenzy_lite_max_hours', FRENZY_LITE_HMAX_DEFAULT))
        return h if h > _f(th, 'frenzy_min_hours', 2.0) else FRENZY_LITE_HMAX_DEFAULT
    except (TypeError, ValueError):
        return FRENZY_LITE_HMAX_DEFAULT
FRENZY_LITE_COUNTED = ("FRENZY_LITE_VOL24_LOW", "FRENZY_LITE_GREEN_BAR")   # signal-level refusals of a LITE candidate bar → filter-block counters
FRENZY_LITE_JUDGES = (FRENZY_LITE_READY, "FRENZY_LITE_GREEN_BAR")   # the bar met the SIGNAL → the stretch is judged NOW, whatever follows (VOL24_LOW is not one)
FRENZY_LITE_SHOWN = (FRENZY_LITE_READY, "FRENZY_LITE_STRETCH_DONE", "FRENZY_LITE_TOO_EARLY", "FRENZY_LITE_NO_DATA") + FRENZY_LITE_COUNTED   # codes whose text replaces FRENZY's on the monitor
# 🔁 2026-10-19 FRENZY_LITE_RETRY (operator DECLARED OVERRIDE, DECISION_LOG 316): a LITE stretch refused ONLY for a green signal candle gets ONE
# pending retry — the first red / flat closed 5m bar within frenzy_lite_retry_wait_minutes of the green bar's close on which the SAME stretch is
# still LITE-READY opens FRENZY_LITE_RETRY (own sleeve, own size, LITE's exit, no sleeve slot limit). LITE itself stays judged once.
FRENZY_LITE_RETRY = "FRENZY_LITE_RETRY"
FRENZY_LITE_FAMILY = ("FRENZY_LITE", FRENZY_LITE_RETRY)   # LITE's exit / TP (LITE C) / resting TP / pair-day cap / PAIR_HELD_BY_LITE scope


def frenzy_lite_stretch_id(ep) -> Optional[int]:
    """🪶 The current above-VWAP stretch's id = open ms of its FIRST bar: last_bar_ts − (above_streak − 1) × 5 min. None when the price is
    not above the VWAP on the last bar (streak < 1) or the episode is unreadable."""
    try:
        s = int((ep or {}).get('above_streak') or 0); last = (ep or {}).get('last_bar_ts')
        if s < 1 or last is None:
            return None
        return int(last) - (s - 1) * BAR_MS
    except (TypeError, ValueError):
        return None


def frenzy_lite_status(ep, th, volume_24h, judged_stretch_ms=None, judged_note=None) -> Tuple[bool, str, str]:
    """🪶 (ready, code, text) for FRENZY_LITE on the just-closed bar of a flagged pair. judged_stretch_ms = the last stretch id this pair
    already JUDGED (its first signal bar was evaluated — filled or refused; memory + BotState + DB; None = none); judged_note = how that bar
    ended ("13:40 refused: GREEN_BAR"), shown in the STRETCH_DONE text. Codes in FRENZY_LITE_JUDGES (READY, GREEN_BAR) mean the bar met the
    signal → the caller marks the stretch judged BEFORE any later filter or the open runs. Fail-closed: any unreadable input refuses. The
    market-volume gate, slots, the pair-day cap, an open position on the pair and the dislocation guard are judged by the engine at the open
    (same order as FRENZY). NO ATR check — by design (DECISION_LOG 243)."""
    try:
        if not bool(getattr(th, 'frenzy_lite_enabled', False)):
            return False, "FRENZY_LITE_OFF", "LITE off"
        if not frenzy_flagged(ep, th):
            return False, "FRENZY_LITE_NOT_FLAGGED", "LITE: not a verified flag"
        if ep.get('in_state'):
            return False, "FRENZY_LITE_IN_STATE", "LITE: FRENZY setup ON (FRENZY / WIDE territory)"
        need = frenzy_lite_need(th)
        streak = int(ep.get('above_streak') or 0)
        vm, hrs = ep.get('vol_mult'), ep.get('hours')
        tail = f"vol {float(vm):.0f}× · {float(hrs or 0):.1f} h" if vm is not None else f"vol ? · {float(hrs or 0):.1f} h"
        if streak < need:
            return False, "FRENZY_LITE_NOT_ABOVE", f"LITE: {streak} of {need} closes above its average"
        vneed = _f(th, 'frenzy_state_vol_mult', 100.0)
        if vm is None or float(vm) >= vneed:
            return False, "FRENZY_LITE_VOL_HIGH", (f"LITE: volume {float(vm):.0f}× ≥ {vneed:.0f}× (FRENZY territory)" if vm is not None else "LITE: volume unreadable")
        if hrs is None:
            return False, "FRENZY_LITE_NO_DATA", "LITE: episode age unreadable"
        hmin, hmax = _f(th, 'frenzy_min_hours', 2.0), frenzy_lite_hmax(th)
        if float(hrs) < hmin:
            return False, "FRENZY_LITE_TOO_EARLY", f"LITE: {float(hrs):.1f} h after the spike < {hmin:g} h · vol {float(vm):.0f}×"
        if float(hrs) > hmax:
            return False, "FRENZY_LITE_TOO_LATE", f"LITE: window over ({float(hrs):.1f} h > {hmax:g} h)"
        sid = frenzy_lite_stretch_id(ep)
        if sid is None:
            return False, "FRENZY_LITE_NO_DATA", "LITE: stretch unreadable"
        if judged_stretch_ms is not None and int(judged_stretch_ms) >= sid:
            return False, "FRENZY_LITE_STRETCH_DONE", (f"LITE: stretch judged at {judged_note} · {tail}" if judged_note else f"LITE: stretch judged · {tail}")
        vmin = _f(th, 'frenzy_min_volume_usd', 20e6)
        if volume_24h is None or float(volume_24h) < vmin:
            return False, "FRENZY_LITE_VOL24_LOW", f"LITE: 24 h volume ${(volume_24h or 0) / 1e6:.0f}M < ${vmin / 1e6:.0f}M"
        if bool(getattr(th, 'frenzy_long_skip_green_bar', True)) and not ep.get('bar_red'):   # fail-closed: an unreadable bar is not red
            _br = ep.get('bar_ret_pct')
            return False, "FRENZY_LITE_GREEN_BAR", (f"LITE: green candle ({_br:+.3f}%) · {tail}" if _br is not None else "LITE: signal candle unreadable")
        return True, FRENZY_LITE_READY, f"LITE: held above · {tail}"
    except (TypeError, ValueError, AttributeError):
        return False, "FRENZY_LITE_NO_DATA", "LITE: unreadable"


LITE_RETRY_WAIT_DEF = 15   # frenzy_lite_retry_wait_minutes default — the deep dive's S15r (frozen; never re-fit, DECISION_LOG 316)
LITE_RETRY_OPEN, LITE_RETRY_WAIT, LITE_RETRY_DROP = "OPEN", "WAIT", "DROP"


def frenzy_lite_retry_enabled(th) -> bool:
    """🔁 the retry switch — strictly True (a missing / non-bool value = off)."""
    return getattr(th, 'frenzy_lite_retry_enabled', False) is True


def frenzy_lite_retry_wait_ms(th) -> int:
    """🔁 the retry window after the green bar's close, ms: frenzy_lite_retry_wait_minutes; ≤ 0 / blank / unreadable / non-finite → 15."""
    w = _fin(getattr(th, 'frenzy_lite_retry_wait_minutes', LITE_RETRY_WAIT_DEF))
    if w is None or w <= 0:
        w = LITE_RETRY_WAIT_DEF
    return int(round(w * 60_000))


def frenzy_lite_retry_pending(sid, spike_ms, green_close_ms, green_ret_pct, th) -> Optional[dict]:
    """🔁 the pending retry a GREEN_BAR refusal records: stretch id, episode spike, the green bar's CLOSE (= the pass bar, UTC ms), the expiry
    (green close + the wait) and the green bar's return. None when an id is unreadable (no retry — fail closed)."""
    try:
        g = int(green_close_ms)
        return dict(sid=int(sid), spike=int(spike_ms), green=g, exp=g + frenzy_lite_retry_wait_ms(th),
                    ret=(round(float(green_ret_pct), 4) if _fin(green_ret_pct) is not None else None))
    except (TypeError, ValueError):
        return None


def frenzy_lite_retry_pending_clean(v) -> Optional[dict]:
    """🔁 a stored pending retry (BotState JSON 'lite_retry') → its validated dict, or None (dropped)."""
    try:
        if not isinstance(v, dict):
            return None
        r = _fin(v.get('ret'))
        return dict(sid=int(v['sid']), spike=int(v['spike']), green=int(v['green']), exp=int(v['exp']), ret=r)
    except (TypeError, ValueError, KeyError):
        return None


LONG4_FIRST_N, LONG4_FIRST_MIN_REC, LONG4_MAIN_N, LONG4_MAIN_DAYS = 10, 3, 20, 8          # 🔒 frozen revert gate (DECISION_LOG 317)
LONG4_SIZE_N, LONG4_SIZE_DAYS, LONG4_SIZE_P, LONG4_SIZE_WR, LONG4_SIZE_SHARE = 20, 8, 0.80, 73.0, 0.50   # 🔒 frozen re-sizing read (317)


def frenzy_long_stop4_gate(fills, stop4_min=3.5, boot_n=2000, seed=7):
    """🔒 (DECISION_LOG 317) FRENZY_LONG −4 LIVE: the frozen revert gate + the re-sizing read → (verdict, text). Display only (operator acts).
    fills = iterables of (opened_at, pair, pnl_percentage, entry_frenzy_stop_pct, touch3_at, touch3_recovered_tp) of CLOSED FRENZY_LONG fills.
    Live −4 fills = stamp ≥ stop4_min. REVERT (→ frenzy_long_stop_pct 0): among the first 10 that touched −3 (by touch time) with a KNOWN
    outcome (touch3_recovered_tp None = unknown → excluded, shown) fewer than 3 recovered to the TP · or on the FROZEN main sample = the
    shortest prefix of touches with ≥ 20 touches AND ≥ 8 UTC days, Σ(booked − (−3)) < 0 (the −3 stop would have booked ≈ −3 on each; the
    exact −3 walk is scripts/scout_long_stop4.py). RE-SIZE read (only while the revert gate holds): ≥ 20 live −4 fills on ≥ 8 days, avg > 0 with
    day-bootstrap P ≥ 0.80 ∧ WR ≥ 73 % (+1.5 / −4 breakeven) ∧ no day / pair ≥ 50 % of the gain → PROPOSE one size step. Never raises."""
    try:
        f = [(a, str(p), float(x), t, r) for a, p, x, st, t, r in (fills or [])
             if a is not None and _fin(x) is not None and _fin(st) is not None and float(st) >= stop4_min]
        n = len(f)
        tch = sorted([z for z in f if z[3] is not None], key=lambda z: z[3])
        unk = [z for z in tch if z[4] is None]   # (dual review) recovered unknown (outcome not recorded) — never counted either way
        first = [z for z in tch if z[4] is not None][:LONG4_FIRST_N]
        rec = sum(1 for z in first if z[4] is True)
        head = f"−4 live: {n} fills · {len(tch)} touched −3" + (f" ({len(unk)} unknown outcome)" if unk else "")
        if len(first) >= LONG4_FIRST_N and rec < LONG4_FIRST_MIN_REC:
            return "REVERT", f"🛑 REVERT to −3 — {rec}/{LONG4_FIRST_N} first −3 touches recovered to the TP (< {LONG4_FIRST_MIN_REC}) · {head}"
        main, _dset = None, set()   # (dual review) FROZEN main sample = the SHORTEST prefix of touches (by touch time) with ≥ 20 AND ≥ 8 UTC days
        for k, z in enumerate(tch):
            _dset.add(z[3].date())
            if k + 1 >= LONG4_MAIN_N and len(_dset) >= LONG4_MAIN_DAYS:
                main = tch[:k + 1]
                break
        if main is not None:
            edge = sum(z[2] + 3.0 for z in main)
            if edge < 0:
                return "REVERT", f"🛑 REVERT to −3 — first {len(main)} touches on {len(_dset)} days: Σ(live − (−3)) {edge:+.2f} pts < 0 · {head}"
        txt = (f"⏳ {head} · first-10 recovered {rec}/{len(first)} (revert if < {LONG4_FIRST_MIN_REC} of 10) · touches {len(tch)}/{LONG4_MAIN_N} on "
               f"{len(_dset)}/{LONG4_MAIN_DAYS} days" + (" · main read passed" if main is not None else ""))
        days = sorted({z[0].date() for z in f})
        if n < LONG4_SIZE_N or len(days) < LONG4_SIZE_DAYS:
            return "COLLECTING", txt + f" · re-size read at {LONG4_SIZE_N} fills on {LONG4_SIZE_DAYS} days"
        pn = [z[2] for z in f]
        avg = sum(pn) / n
        wr = 100.0 * sum(1 for x in pn if x > 0) / n
        import random
        rnd = random.Random(seed); byd = {}
        for z in f:
            byd.setdefault(z[0].date(), []).append(z[2])
        dl = list(byd.values()); pos = 0
        for _ in range(boot_n):
            smp = [x for _ in dl for x in dl[rnd.randrange(len(dl))]]
            pos += (sum(smp) / len(smp)) > 0
        pboot = pos / boot_n
        gain = sum(x for x in pn if x > 0)
        dsh = max(sum(x for x in v if x > 0) for v in dl) / gain if gain > 0 else 1.0
        byp = {}
        for z in f:
            byp[z[1]] = byp.get(z[1], 0.0) + max(z[2], 0.0)
        psh = (max(byp.values()) / gain) if gain > 0 else 1.0
        ok = avg > 0 and pboot >= LONG4_SIZE_P and wr >= LONG4_SIZE_WR and dsh < LONG4_SIZE_SHARE and psh < LONG4_SIZE_SHARE
        return (("RESIZE" if ok else "HOLD"), txt + f" · re-size read: avg {avg:+.3f}% P {pboot:.2f} WR {wr:.0f}% day share {dsh:.0%} pair share {psh:.0%} → "
                + ("PROPOSE one size step (operator decides; revert if the next 15 average < 0 or a day loses > 10 % of the book)" if ok else "hold blocker size"))
    except Exception:
        return "COLLECTING", "gate unreadable"


def frenzy_long_stop_config_errors(th, base_leverage=20.0) -> list:
    """🛑 (DECISION_LOG 317) PUT /api/config refusals: frenzy_long_stop_pct must be 0 (= the shared frenzy_stop_pct) or 1–10 % · (dual review)
    and the LONG exchange backstop at LONG's MAX leverage (base_leverage × max(frenzy_long_lev_mult, frenzy_long_lev_mult_strong) — the
    sleeves open at the STRONG_BUY level) must stay ≥ MANUAL_BACKSTOP_MIN_GAP_PCT beyond the stop (a liquidation-capped backstop inside the
    stop would fire first)."""
    v = _fin(getattr(th, 'frenzy_long_stop_pct', 0.0))
    if v is None or not (v == 0 or 1.0 <= v <= 10.0):
        return ["frenzy_long_stop_pct must be 0 (= the shared FRENZY stop) or a number from 1 to 10"]
    try:
        from services.trading_engine import order_backstop_pct, MANUAL_BACKSTOP_MIN_GAP_PCT
        mults = [m for m in (_fin(getattr(th, 'frenzy_long_lev_mult', 0.05)), _fin(getattr(th, 'frenzy_long_lev_mult_strong', 0.0))) if m and m > 0]
        lev = max(1, int(round(float(base_leverage or 20.0) * (max(mults) if mults else 1.0))))
        stop = frenzy_stop_for(th, "FRENZY_LONG")
        if order_backstop_pct(th, "FRENZY_LONG", leverage=lev) - stop < MANUAL_BACKSTOP_MIN_GAP_PCT - 1e-9:
            return [f"frenzy_long_stop_pct {stop:g} % is too wide for FRENZY_LONG at {lev}×: its exchange backstop "
                    f"({order_backstop_pct(th, 'FRENZY_LONG', leverage=lev):g} %, capped by liquidation) must sit ≥ {MANUAL_BACKSTOP_MIN_GAP_PCT:g} % beyond it"]
    except Exception:
        return ["frenzy_long_stop_pct: the backstop distance could not be checked"]
    return []


def frenzy_lite_retry_config_errors(th) -> list:
    """🔁 PUT /api/config refusals for the FRENZY_LITE_RETRY fields (DECISION_LOG 316)."""
    errs = []
    if not isinstance(getattr(th, 'frenzy_lite_retry_enabled', False), bool):
        errs.append("frenzy_lite_retry_enabled must be true or false")
    w = _fin(getattr(th, 'frenzy_lite_retry_wait_minutes', LITE_RETRY_WAIT_DEF))
    if w is None or not (5 <= w <= 60):
        errs.append("frenzy_lite_retry_wait_minutes must be a number from 5 to 60 (the frozen rule is 15)")
    for k in ('frenzy_lite_retry_invest_mult', 'frenzy_lite_retry_lev_mult'):
        v = getattr(th, k, None)
        if v is not None and (_fin(v) is None or _fin(v) < 0):
            errs.append(f"{k} must be a number ≥ 0 (blank = 1.0)")
    return errs


LITE_RETRY_KILL_SUM, LITE_RETRY_FIRST_N, LITE_RETRY_MAIN_N, LITE_RETRY_MAIN_DAYS = -6.0, 10, 20, 10   # 🔒 frozen gates (DECISION_LOG 316)


def frenzy_lite_retry_gate(retry_fills, lite_fills) -> Tuple[str, str]:
    """🔒 (DECISION_LOG 316) the FRENZY_LITE_RETRY frozen gates, read on CLOSED fills → (verdict, text). retry_fills / lite_fills = iterables of
    (opened_at datetime, pnl_percentage) of this mode's closed FRENZY_LITE_RETRY / FRENZY_LITE fills. Verdicts: COLLECTING · OFF (early kill:
    Σ pnl% ≤ −6 pts at any N · first read: the first 10 fills average < 0 · main read failed) · STEP_UP (main read at 20 fills on ≥ 10 UTC days:
    avg > 0 ∧ ≥ the same-period live FRENZY_LITE avg = LITE fills opened at / after the first retry fill → lev 0.2 → 0.4; the main sample is
    FROZEN = the shortest prefix with ≥ 20 fills AND ≥ 10 distinct UTC days). Display only —
    the operator switches (no automatic off). Never raises."""
    try:
        rf = sorted([(a, float(p)) for a, p in (retry_fills or []) if a is not None and _fin(p) is not None], key=lambda x: x[0])
        n = len(rf)
        if n == 0:
            return "COLLECTING", "⏳ 0/20 closed — gates: kill Σ ≤ −6 pts · first read 10 (avg < 0 → off) · main read 20 on ≥ 10 days (vs live LITE → 8×)"
        ssum = sum(p for _, p in rf)
        if ssum <= LITE_RETRY_KILL_SUM:
            return "OFF", f"🛑 EARLY KILL — Σ {ssum:+.2f} pts ≤ {LITE_RETRY_KILL_SUM:g} on {n} fills → switch FRENZY_LITE_RETRY OFF (operator)"
        first = rf[:LITE_RETRY_FIRST_N]
        if n >= LITE_RETRY_FIRST_N and sum(p for _, p in first) / len(first) < 0:
            return "OFF", f"🛑 FIRST READ — first {LITE_RETRY_FIRST_N} avg {sum(p for _, p in first) / len(first):+.3f}% < 0 → switch OFF (operator)"
        main, _dset = None, set()   # (dual review) the FROZEN sample = the SHORTEST prefix with ≥ 20 fills AND ≥ 10 distinct UTC days
        for k, (a, _) in enumerate(rf):
            _dset.add(a.date())
            if k + 1 >= LITE_RETRY_MAIN_N and len(_dset) >= LITE_RETRY_MAIN_DAYS:
                main = rf[:k + 1]
                break
        if main is None:
            return "COLLECTING", (f"⏳ {n}/{LITE_RETRY_MAIN_N} closed on {len(_dset)}/{LITE_RETRY_MAIN_DAYS} days · Σ {ssum:+.2f} pts"
                                  f" · avg {ssum / n:+.3f}% (kill Σ ≤ −6 · first read at 10 · declared override)")
        t0 = main[0][0]
        lt = [float(p) for a, p in (lite_fills or []) if a is not None and a >= t0 and _fin(p) is not None]
        mavg = sum(p for _, p in main) / len(main)
        lavg = (sum(lt) / len(lt)) if lt else None
        ok = mavg > 0 and lavg is not None and mavg >= lavg
        return (("STEP_UP" if ok else "OFF"),
                f"📋 MAIN READ — first {len(main)} fills (≥ {LITE_RETRY_MAIN_N} on ≥ {LITE_RETRY_MAIN_DAYS} days) avg {mavg:+.3f}% vs live LITE same period "
                + (f"{lavg:+.3f}% ({len(lt)} fills)" if lavg is not None else "n/a (no LITE fills)")
                + (" → STEP UP lev 0.2 → 0.4 (8×, operator)" if ok else " → switch OFF (operator)"))
    except Exception:
        return "COLLECTING", "gate unreadable"


def frenzy_lite_retry_step(ep, th, volume_24h, pending, bar_close_ms) -> Tuple[str, str, str]:
    """🔁 (action, code, text) for a pending FRENZY_LITE_RETRY on the closed 5m bar whose CLOSE is bar_close_ms (the pass bar). Pure, never raises.
    DROP  FRENZY_LITE_RETRY_EXPIRED       the bar closed after the window (green close + frenzy_lite_retry_wait_minutes)
          FRENZY_LITE_RETRY_ENDED         the episode is no longer flagged / a different spike
          FRENZY_LITE_RETRY_BROKEN        the stretch broke (a close below the spike VWAP → a different stretch id, or none)
          FRENZY_LITE_RETRY_WINDOW_OVER   LITE's episode-age window passed (frenzy_lite_max_hours)
          FRENZY_LITE_RETRY_OFF           LITE switched off (the retry re-reads LITE's own status)
    OPEN  the SAME stretch is FRENZY_LITE_READY on this bar (red / flat candle ∧ every frenzy_lite_status check, read as if unjudged)
    WAIT  anything else (a green candle, volume ≥ the setup ×, FRENZY in state, 24 h volume low, unreadable) — skipped; the next bar may open.
    A bar at / before the green bar's close never acts (WAIT)."""
    try:
        p = pending or {}
        b = int(bar_close_ms)
        if b <= int(p['green']):
            return LITE_RETRY_WAIT, "FRENZY_LITE_RETRY_SAME_BAR", "LITE retry: the green bar itself"
        if b > int(p['exp']):
            return LITE_RETRY_DROP, "FRENZY_LITE_RETRY_EXPIRED", "LITE retry: no red / flat READY bar within the window"
        if not frenzy_flagged(ep, th) or (ep or {}).get('spike_ts') is None or int(ep['spike_ts']) != int(p['spike']):
            return LITE_RETRY_DROP, "FRENZY_LITE_RETRY_ENDED", "LITE retry: the episode ended"
        sid = frenzy_lite_stretch_id(ep)
        if sid is None or int(sid) != int(p['sid']):
            return LITE_RETRY_DROP, "FRENZY_LITE_RETRY_BROKEN", "LITE retry: the stretch broke (a close below the spike average)"
        ok, code, text = frenzy_lite_status(ep, th, volume_24h, None)
        if ok:
            return LITE_RETRY_OPEN, code, text
        if code == "FRENZY_LITE_TOO_LATE":
            return LITE_RETRY_DROP, "FRENZY_LITE_RETRY_WINDOW_OVER", text
        if code == "FRENZY_LITE_OFF":
            return LITE_RETRY_DROP, "FRENZY_LITE_RETRY_OFF", text
        return LITE_RETRY_WAIT, code, text
    except (TypeError, ValueError, KeyError, AttributeError):
        return LITE_RETRY_WAIT, "FRENZY_LITE_RETRY_NO_DATA", "LITE retry: unreadable"


FRENZY_CATCHUP_OK, FRENZY_CATCHUP_STALE = "FRENZY_CATCHUP", "FRENZY_CATCHUP_STALE"


def frenzy_catchup_check(ep, prev_judged_ms, max_bars) -> Tuple[Optional[str], Optional[int]]:
    """⏪ Oct-6 FRENZY catch-up (NMRUSDT 10-06: the fresh bar closed while the bot was paused → "ON (entry bar passed)", never entered).
    → (status, age_bars). status None = nothing to recover: catch-up off (max_bars ≤ 0) · no stored last-judged bar (cold start — never
    guess) · not in state · fresh_on (the normal path judges it) · the ON bar is at / before prev_judged_ms (a completed pass judged it).
    FRENZY_CATCHUP = the ON bar fell inside an unjudged window and is ≤ max_bars bars older than the last closed bar → judge it as if fresh.
    FRENZY_CATCHUP_STALE = unjudged but older than that. prev_judged_ms = open ms of the last 5m bar a COMPLETED pass judged BEFORE this one.
    Entering on a later ON bar is refuted (−0.49 %/trade, FRENZY_STAIRCASE_STUDY_2026-10-06) — this recovers only the fresh bar itself."""
    try:
        mb = int(max_bars or 0)
        if mb <= 0 or prev_judged_ms is None or not ep or not ep.get('in_state') or ep.get('fresh_on'):
            return None, None
        on, last = ep.get('on_bar_ts'), ep.get('last_bar_ts')
        if on is None or last is None or int(on) <= int(prev_judged_ms) or int(on) >= int(last):
            return None, None
        age = int((int(last) - int(on)) // BAR_MS)
        return (FRENZY_CATCHUP_OK if age <= mb else FRENZY_CATCHUP_STALE), age
    except (TypeError, ValueError):
        return None, None


def frenzy_catchup_moved(on_close, live_price, max_pct) -> bool:
    """⏪ True = REFUSE the catch-up: the live price is more than max_pct % from the ON bar's close (not the same trade any more). Fail-closed:
    a guard switched off (≤ 0) or an unreadable price refuses — a catch-up without the economics check is never taken."""
    try:
        mx, c, p = float(max_pct or 0), float(on_close or 0), float(live_price or 0)
        return not (mx > 0 and c > 0 and p > 0 and abs(p / c - 1) * 100 <= mx)
    except (TypeError, ValueError):
        return True


def frenzy_vol24_at(bars) -> Optional[float]:
    """⏪ 24 h quote volume (volume × typical price, the walk's convention) of the 288 closed 5m bars ending at bars[-1] — the catch-up's
    24 h volume ON the signal bar (the ticker's figure is today's). None when < 288 bars or unreadable."""
    try:
        if not bars or len(bars) < 288:
            return None
        return sum(float(r[5]) * (float(r[2]) + float(r[3]) + float(r[4])) / 3.0 for r in bars[-288:])
    except (TypeError, ValueError, IndexError):
        return None


def frenzy_flagged(ep, th) -> bool:
    """A live episode counts as a FRENZY flag while its spike is verifiable and ≤ frenzy_max_hours old."""
    return bool(ep and ep.get('verified') and ep.get('hours') is not None and ep['hours'] <= _f(th, 'frenzy_max_hours', 96.0))


def frenzy_long_status(ep, atr_pct, volume_24h, th) -> Tuple[bool, str, str]:
    """(ready, code, text) for a flagged pair. ready = the LONG may open on this bar. Codes double as filter-block counters."""
    vneed = _f(th, 'frenzy_state_vol_mult', 100.0)
    if not ep.get('in_state'):
        if not ep.get('above_hour'):
            # Oct-2 (operator: hover "+1.2 % vs its average" while this said "below"): the check is a full HOUR of closes at or
            # above the line, so a price already back above it says so. Same code either way (one block counter).
            # Short: the Block Reason cell does not wrap. Within ±0.05 % two decimals, so the sign shown is the real one.
            vs, n_up = ep.get('vs_vwap_pct'), int(ep.get('above_streak') or 0)
            pct = (lambda v: f"{v:+.2f}%" if abs(v) < 0.05 else f"{v:+.1f}%")
            if vs is not None and vs >= 0 and n_up > 0:
                return False, "FRENZY_BELOW_AVG", f"back above its average ({pct(vs)}) · {min(n_up, 11)} of 12 closes"
            return False, "FRENZY_BELOW_AVG", "below its average price" + (f" ({pct(vs)})" if vs is not None else "")
        if ep.get('vol_mult') is None or ep['vol_mult'] < vneed:
            return False, "FRENZY_VOL_FADED", f"held above 1 h · volume {ep.get('vol_mult') or 0:.0f}× < {vneed:.0f}×"
        return False, "FRENZY_TOO_EARLY", f"{ep.get('hours') or 0:.1f} h after the spike < {_f(th, 'frenzy_min_hours', 2.0):g} h"
    if not ep.get('fresh_on'):
        return False, "FRENZY_ON", "ON (entry bar passed)"
    vmin = _f(th, 'frenzy_min_volume_usd', 20e6)
    if volume_24h is None or float(volume_24h) < vmin:
        return False, "FRENZY_VOL24_LOW", f"24 h volume ${(volume_24h or 0) / 1e6:.0f}M < ${vmin / 1e6:.0f}M"
    amax = _f(th, 'frenzy_max_atr_pct', 2.5)
    if amax > 0 and (atr_pct is None or float(atr_pct) > amax):
        return False, "FRENZY_ATR_HIGH", (f"ATR {atr_pct:.2f}% > {amax:g}%" if atr_pct is not None else "ATR unreadable")
    if bool(getattr(th, 'frenzy_long_skip_green_bar', True)) and not ep.get('bar_red'):   # fail-closed: an unreadable bar is not red
        _br = ep.get('bar_ret_pct')
        return False, "FRENZY_GREEN_BAR", (f"signal candle green ({_br:+.3f}%)" if _br is not None else "signal candle unreadable")
    return True, "FRENZY_READY", "READY"


def _fin(v) -> Optional[float]:
    """float(v) when finite, else None (None / NaN / ±inf / unreadable)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def frenzy_lite_tp_override(v) -> Optional[float]:
    """🎯 (DECISION_LOG 301) an order's stored LITE take profit (Order.frenzy_lite_tp_used) when usable — finite and > 0 — else None
    (a legacy / pre-301 order, LITE C off at its entry, or a corrupt value → the config TP, exactly as before 301)."""
    x = _fin(v)
    return x if (x is not None and x > 0) else None


def frenzy_lite_c_enabled(th) -> bool:
    return getattr(th, 'frenzy_lite_c_enabled', False) is True


def frenzy_lite_c_tp(th, vol_mult, btc_1h_slope) -> Tuple[float, bool]:
    """🎯 "LITE C" (operator 2026-10-15, DECISION_LOG 301) → (the take profit a NEW FRENZY_LITE fill runs, in_zone). Decided ONCE at entry
    from the SAME values stamped on the order (entry_frenzy_vol_mult, entry_btc_1h_slope):
    ZONE = vol_mult ≥ frenzy_lite_c_vol_mult ∧ btc_1h_slope > frenzy_lite_c_btc_slope_min → the normal LITE TP (frenzy_tp_for);
    otherwise — either value missing / NaN / unreadable included (fail to the LOWER TP) — frenzy_lite_c_tp_pct.
    Switch off → the normal LITE TP for every fill (in_zone still reported for the tally). A frenzy_lite_c_tp_pct ≤ 0 / unreadable never
    becomes "TP off": the normal LITE TP is used instead. Never raises."""
    base = frenzy_tp_for(th, "FRENZY_LITE")
    vm, sl = _fin(vol_mult), _fin(btc_1h_slope)
    vcut = _fin(getattr(th, 'frenzy_lite_c_vol_mult', 20.0))
    scut = _fin(getattr(th, 'frenzy_lite_c_btc_slope_min', 0.0))
    vcut = 20.0 if vcut is None else vcut
    scut = 0.0 if scut is None else scut
    zone = bool(vm is not None and sl is not None and vm >= vcut and sl > scut)
    if not frenzy_lite_c_enabled(th) or zone:
        return base, zone
    ctp = _fin(getattr(th, 'frenzy_lite_c_tp_pct', 2.0))
    if base <= 0 or ctp is None or ctp <= 0:              # LITE TP off stays off; an unusable C value → the normal LITE TP
        return base, zone
    return min(ctp, base), zone                          # never ABOVE the zone's TP (a LITE TP set below 2 is not inverted)


def frenzy_stop_for(th, sleeve=None) -> float:
    """🛑 (DECISION_LOG 317, operator DECLARED OVERRIDE 2026-10-19) the loss stop a FRENZY exit uses, positive %: FRENZY_LONG runs
    frenzy_long_stop_pct when > 0 (live 4.0); every other caller — FRENZY_WIDE / FRENZY_LITE / FRENZY_LITE_RETRY / the MANUAL "FRENZY" exit
    (sleeve None) — and LONG with 0 / blank / unreadable runs the shared frenzy_stop_pct (0 / missing → 3). Never raises."""
    base = abs(_f(th, 'frenzy_stop_pct', 3.0)) or 3.0
    if sleeve == "FRENZY_LONG":
        v = _fin(getattr(th, 'frenzy_long_stop_pct', 0.0))
        if v is not None and v > 0:
            return v
    return base


def frenzy_tp_for(th, sleeve=None, tp_override=None) -> float:
    """🎯 the fixed FRENZY take profit for a sleeve (DECISION_LOG 284, operator): FRENZY_LITE runs frenzy_lite_tp_pct when > 0, every other
    caller (FRENZY_LONG / FRENZY_WIDE / the manual FRENZY exit, or LITE with frenzy_lite_tp_pct ≤ 0) the shared frenzy_tp_pct. ≤ 0 = off.
    tp_override (🎯 DECISION_LOG 301 LITE C): the order's stored LITE TP (Order.frenzy_lite_tp_used) — FRENZY_LITE only, used when finite
    and > 0; None (legacy order / LITE C off at entry) → the config TP above. Ignored for every other sleeve."""
    if sleeve in FRENZY_LITE_FAMILY:   # 🔁 (316) FRENZY_LITE_RETRY runs LITE's exit exactly (its own stored LITE C TP included)
        ov = frenzy_lite_tp_override(tp_override)
        if ov is not None:
            return ov
        lt = _f(th, 'frenzy_lite_tp_pct', 0.0)
        if lt > 0:
            return lt
    return _f(th, 'frenzy_tp_pct', 0.0)


def frenzy_exit_for(pnl, peak_pnl, th, stop_floor=None, short=False, use_tp=False, sleeve=None, resting_tp=False, tp_override=None) -> Tuple[bool, str, float]:
    """FRENZY_LONG's own exit → (close, reason, line). use_tp (the FRENZY_LONG / FRENZY_WIDE sleeve AND the MANUAL "FRENZY" exit — operator):
    🎯 Oct-5 (DECISION_LOG 205) when frenzy_lock_arm_pct > 0: lock +floor at +arm, then trail points below the peak (no fixed TP) — below.
    🎯 Oct-4 (operator, DECISION_LOG 196) a FIXED take-profit at +frenzy_tp_pct net (0 = off) → reason FRENZY_TP, checked before the trail
    (the trail is only the fallback). Real-tick year (scripts/frenzy_exit_ticks.py, 850 quiet-market fills): fixed +4/−3 +0.211 %/trade
    vs the +5/1.5 trail +0.169 (5 of 9 months better, CI spans 0); bot-exact re-run (DECISION_LOG 199): +3/−3 +0.192 · +4/−3 +0.187 → operator +3. A peak that already reached the TP while the close was missed (failed
    close, feed gap, restart) closes at once (review: never ride a +tp back down to the stop); the trail is only the fallback when tp = 0.
    Stop at −frenzy_stop_pct; once the peak reaches +frenzy_trail_arm_pct the
    line trails frenzy_trail_giveback_pct of PRICE below the best point (peak − giveback × (1 + peak/100)). Reasons are the
    momentum stack's own (STOP_LOSS / RUNNER_TRAIL) so every matcher knows them; entry_strategy tells the sleeve apart.
    sleeve also picks the STOP (🛑 DECISION_LOG 317): FRENZY_LONG → frenzy_long_stop_pct when > 0, else frenzy_stop_pct (frenzy_stop_for).
    stop_floor: optional cap (never wider than this). The engine passes None since DECISION_LOG 271 — the sleeve's own stop runs in
    paper AND live; the per-order exchange backstop rests just beyond it (order_backstop_pct).
    short=True (a MANUAL short on this exit): the give-back is measured from the LOWEST price → peak − giveback × (1 − peak/100).
    sleeve: the order's entry_strategy — picks the take profit via frenzy_tp_for (FRENZY_LITE's own TP since DECISION_LOG 284).
    resting_tp=True (🎯 DECISION_LOG 296): a resting exchange TP limit is alive for this order — the fixed TP is NOT fired here (the limit
    books it; the engine runs the through + grace safety net); the stop / trail below are judged exactly as before.
    tp_override (🎯 DECISION_LOG 301 LITE C): the order's stored LITE TP — passed through to frenzy_tp_for (FRENZY_LITE only)."""
    try:
        pnl = float(pnl); pk = float(peak_pnl or 0.0)
        stop = -frenzy_stop_for(th, sleeve)   # 🛑 (317) FRENZY_LONG: frenzy_long_stop_pct when > 0 · everyone else (manual FRENZY included): frenzy_stop_pct
        if stop_floor is not None:
            stop = max(stop, float(stop_floor))
        # 🎯 Oct-5 (operator, DECISION_LOG 205): LOCK-then-trail — once the peak reaches +frenzy_lock_arm_pct the line jumps to
        # +frenzy_lock_floor_pct and then trails frenzy_lock_trail_pct POINTS below the peak (never below the floor); no fixed TP. Replaces
        # the fixed TP while armed (> 0). Real-tick year, bot accounting (scripts/frenzy_trail_v2.py, 850 fills): lock +2 at +3 / trail 2
        # +0.234 %/trade vs fixed +3/−3 +0.192 (Δ +0.042, CI −0.05…+0.15, 5 of 9 months; −0.011 without its 5 best trades = a runner rule).
        # Same lines for a MANUAL short (pnl is direction-aware net P&L, the trail is in points).
        la = _f(th, 'frenzy_lock_arm_pct', 0.0) if use_tp else 0.0
        if la > 0:
            if pk >= la:
                lf = min(_f(th, 'frenzy_lock_floor_pct', 2.0), la)
                lt = abs(_f(th, 'frenzy_lock_trail_pct', 2.0))
                line = max(lf, pk - lt)
                return (pnl <= line), "RUNNER_TRAIL", line
            return (pnl <= stop), "STOP_LOSS", stop
        tp = frenzy_tp_for(th, sleeve, tp_override) if (use_tp and not resting_tp) else 0.0   # ≤ 0 = off (never abs(): a negative value must not become an active TP) · 🎯 296: a resting TP owns it
        if tp > 0 and (pnl >= tp or pk >= tp):
            return True, "FRENZY_TP", tp
        arm, give = abs(_f(th, 'frenzy_trail_arm_pct', 5.0)), abs(_f(th, 'frenzy_trail_giveback_pct', 1.5))
        if arm > 0 and give > 0 and pk >= arm:
            line = max(stop, pk - give * ((1 - pk / 100.0) if short else (1 + pk / 100.0)))
            return (pnl <= line), "RUNNER_TRAIL", line
        return (pnl <= stop), "STOP_LOSS", stop
    except (TypeError, ValueError):
        try:
            return (float(pnl) <= -3.0), "STOP_LOSS", -3.0
        except (TypeError, ValueError):
            return False, "STOP_LOSS", -3.0


# ── 🎲 Oct-8 FRENZY_WILLY (operator DECLARED EXCEPTION, DECISION_LOG 251) ─────────────────────────────────────────────────────────────
# LONG only, no filters (no gvol / ATR / bearish gates): entry A = the bar a pair becomes FRENZY-flagged for an episode (engine view),
# entry B = an episode's fresh ON bar that FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE did not take. Its OWN exit (never FRENZY's TP3):
# fixed TP +frenzy_willy_tp_pct net · NO stop (operator Oct-8; frenzy_willy_stop_pct 0 = off) · time cap frenzy_willy_max_hold_minutes
# (120). While a WILLY is open NO other automated trade opens (the global hold, TradingEngine._willy_hold_block). All research says ≈ 0 or
# negative (reports/FRENZY_NEW_FLAG_X_STUDY_2026-10-07.md, FRENZY_FLAG_TRADE_MATH_2026-10-08.md, FRENZY_SECONDS_DELAY_STUDY_2026-10-08.md).
FRENZY_WILLY = "FRENZY_WILLY"
# 🎲¼ 2026-10-21 FRENZY_WILLY_HT (operator option B, DECISION_LOG 341): a QUARTER-size sleeve for the WILLY setups the turnover filter alone
# refuses (24 h volume / mcap ≥ frenzy_willy_max_vol_mcap_ratio, a MEASURED ratio — never "unreadable"). Same entry and guards as WILLY, the
# SAME exit (WILLY_EXIT_CLASS: TP / resting maker TP / stop / time cap / late TP / backstop), its OWN slot (frenzy_willy_ht_max_slots) and
# size (frenzy_willy_ht_size_frac × the margin WILLY would have opened, same leverage → exactly ¼ of the notional). NOT a WILLY for the
# WILLY slot count, the hard single-WILLY guard or the global hold (HT is never a hold trigger; as a blocked sleeve it follows the generic
# rule). Purpose: real-fill evidence on the turnover filter instead of shadow-only.
FRENZY_WILLY_HT = "FRENZY_WILLY_HT"
WILLY_EXIT_CLASS = (FRENZY_WILLY, FRENZY_WILLY_HT)   # every exit / protection site: run WILLY's exit
WILLY_HT_FRAC_DEF = 0.25
WILLY_HT_KILL_SUM, WILLY_HT_FIRST_N, WILLY_HT_MAIN_N, WILLY_HT_MAIN_DAYS, WILLY_HT_REREAD_N, WILLY_HT_CONFIRM_GAP = -8.0, 10, 30, 10, 60, 0.20
WILLY_HT_MIN_WILLY = 10   # (341 review) the main read needs ≥ 10 full-size WILLY comparator fills in the same period, else REREAD


def is_willy_exit_class(es) -> bool:
    """True for a row that runs the WILLY exit (FRENZY_WILLY · FRENZY_WILLY_HT). Never raises."""
    try:
        return str(es or "") in WILLY_EXIT_CLASS
    except Exception:
        return False


def frenzy_willy_ht_frac(th) -> float:
    """the HT size fraction the engine USES: frenzy_willy_ht_size_frac in (0, 1]; blank / ≤ 0 / > 1 / unreadable → 0.25. Never raises."""
    try:
        v = float(getattr(th, 'frenzy_willy_ht_size_frac', WILLY_HT_FRAC_DEF))
        return v if (math.isfinite(v) and 0 < v <= 1.0) else WILLY_HT_FRAC_DEF
    except Exception:
        return WILLY_HT_FRAC_DEF


def frenzy_willy_ht_config_errors(th) -> list:
    """🎲¼ PUT /api/config refusals for the FRENZY_WILLY_HT fields (DECISION_LOG 341). The UI clamps the size fraction to the default
    0.25 (with a note) before it sends; anything else that reaches the server out of range is refused."""
    errs = []
    if not isinstance(getattr(th, 'frenzy_willy_ht_enabled', True), bool):
        errs.append("frenzy_willy_ht_enabled must be true or false")
    v = _fin(getattr(th, 'frenzy_willy_ht_size_frac', WILLY_HT_FRAC_DEF))
    if v is None or not (0 < v <= 1.0):
        errs.append("frenzy_willy_ht_size_frac must be > 0 and ≤ 1 (0.25 = a quarter of WILLY)")
    s_ = _fin(getattr(th, 'frenzy_willy_ht_max_slots', 1))
    if s_ is None or s_ < 1:
        errs.append("frenzy_willy_ht_max_slots must be ≥ 1")
    return errs


def frenzy_willy_ht_gate(ht_fills, willy_fills) -> Tuple[str, str]:
    """🔒 (DECISION_LOG 341) the FRENZY_WILLY_HT gates — ALL alert-only (the operator switches; nothing automatic). ht_fills / willy_fills =
    iterables of (opened_at datetime, pnl_percentage) of CLOSED fills (operator-closed already excluded by the caller). Verdicts:
    KILL (Σ ≤ −8 pts at any N → review) · FIRST (first 10 avg < 0 → review) — both still show the main-read status after them ·
    COLLECTING · MAIN at the FROZEN sample = the shortest prefix with ≥ 30 fills on ≥ 10 UTC days, vs full-size WILLY fills opened in the same
    period (≥ the first HT fill, ≤ the last sampled one; ≥ WILLY_HT_MIN_WILLY of them, else REREAD): HT avg ≥ WILLY avg → REMOVE_FILTER
    (recommend turnover ratio 0 + HT off) · HT avg ≤ WILLY avg − 0.20 → CONFIRMED (filter kept, HT off) · else REREAD, judged again on the
    first 60 HT fills once ≥ 60 exist (INCONCLUSIVE if still between). → (verdict, text). Never raises."""
    try:
        hf = sorted([(a, float(p)) for a, p in (ht_fills or []) if a is not None and _fin(p) is not None], key=lambda x: x[0])
        n = len(hf)
        tail = " (alert-only · operator switches · DECISION_LOG 341)"
        if n == 0:
            return "COLLECTING", (f"⏳ 0/{WILLY_HT_MAIN_N} closed — gates: kill Σ ≤ {WILLY_HT_KILL_SUM:g} pts · first read {WILLY_HT_FIRST_N} (avg < 0) · "
                                  f"main read {WILLY_HT_MAIN_N} on ≥ {WILLY_HT_MAIN_DAYS} days vs WILLY" + tail)

        def _read(sample, label):
            t0, t1 = sample[0][0], sample[-1][0]
            wv = [float(p) for a, p in (willy_fills or []) if a is not None and t0 <= a <= t1 and _fin(p) is not None]
            mavg = sum(p for _, p in sample) / len(sample)
            wavg = (sum(wv) / len(wv)) if wv else None
            head = (f"📋 {label} — first {len(sample)} HT fills avg {mavg:+.3f}% vs full-size WILLY same period "
                    + (f"{wavg:+.3f}% ({len(wv)} fills)" if wavg is not None else "n/a (no WILLY fills)"))
            if len(wv) < WILLY_HT_MIN_WILLY:
                return "REREAD", head + f" · comparator too thin ({len(wv)} < {WILLY_HT_MIN_WILLY} WILLY fills)"
            if mavg >= wavg:
                return "REMOVE_FILTER", head + " → recommend removing the turnover filter (ratio 0) and HT off"
            if mavg <= wavg - WILLY_HT_CONFIRM_GAP:
                return "CONFIRMED", head + f" → turnover filter confirmed (HT ≤ WILLY − {WILLY_HT_CONFIRM_GAP:.2f}), HT off"
            return "REREAD", head

        ssum = sum(p for _, p in hf)
        main, ds = None, set()
        for k, (a, _) in enumerate(hf):
            ds.add(a.date())
            if k + 1 >= WILLY_HT_MAIN_N and len(ds) >= WILLY_HT_MAIN_DAYS:
                main = hf[:k + 1]
                break
        if main is None:
            mv, mt = "COLLECTING", f"⏳ {n}/{WILLY_HT_MAIN_N} closed on {len(ds)}/{WILLY_HT_MAIN_DAYS} days · Σ {ssum:+.2f} pts · avg {ssum / n:+.3f}%"
        else:
            mv, mt = _read(main, "MAIN READ")
            if mv == "REREAD":
                if n < WILLY_HT_REREAD_N:
                    mt += f" → inconclusive, re-read at {WILLY_HT_REREAD_N} HT fills ({n}/{WILLY_HT_REREAD_N})"
                else:
                    v2, t2 = _read(hf[:WILLY_HT_REREAD_N], f"RE-READ at {WILLY_HT_REREAD_N}")
                    mv, mt = (v2 if v2 != "REREAD" else "INCONCLUSIVE"), t2 + (" → still inconclusive: operator decides" if v2 == "REREAD" else "")
        if ssum <= WILLY_HT_KILL_SUM:
            return "KILL", f"🛑 EARLY KILL — Σ {ssum:+.2f} pts ≤ {WILLY_HT_KILL_SUM:g} on {n} fills → ALERT: review FRENZY_WILLY_HT · main read: {mt}" + tail
        first = hf[:WILLY_HT_FIRST_N]
        if n >= WILLY_HT_FIRST_N and sum(p for _, p in first) / len(first) < 0:
            return "FIRST", (f"⚠ FIRST READ — first {WILLY_HT_FIRST_N} avg {sum(p for _, p in first) / len(first):+.3f}% < 0 → ALERT: review "
                             f"FRENZY_WILLY_HT · main read: {mt}" + tail)
        return mv, mt + tail
    except Exception:
        return "COLLECTING", "gate unreadable"
WILLY_TP_DEF, WILLY_HOLD_DEF = 1.0, 120   # 🎯 operator Oct-8 originals — ONLY the blank / 0 fallbacks of frenzy_willy_levels; the LIVE exit is trading_config.json (+1.25 / −4 / 90 since DECISION_LOG 281)


def frenzy_willy_levels(th) -> Tuple[float, Optional[float], int]:
    """(tp %, stop % as a NEGATIVE number or None = NO stop, max hold minutes) the WILLY exit USES. TP blank / ≤ 0 → +1; hold blank / < 1 →
    120 (the take profit and the time cap ARE the exit — a zero never removes them). Stop: frenzy_willy_stop_pct > 0 → −that; blank / 0 /
    unreadable → None — unreachable from the config since DECISION_LOG 292 ⑤ (save refuses ≤ 0, load resets it to 4.0)."""
    tp = _f(th, 'frenzy_willy_tp_pct', WILLY_TP_DEF)
    st = abs(_f(th, 'frenzy_willy_stop_pct', 0.0))
    mh = _f(th, 'frenzy_willy_max_hold_minutes', WILLY_HOLD_DEF)
    return ((tp if tp > 0 else WILLY_TP_DEF), (-st if st > 0 else None), (int(round(mh)) if mh >= 1 else WILLY_HOLD_DEF))


def frenzy_willy_late_tp(th) -> Optional[Tuple[float, float]]:
    """⏰ (DECISION_LOG 297, operator-armed DECLARED EXCEPTION) the WILLY LATE take profit → (minutes, net %) or None = off. From the order's
    age ≥ frenzy_willy_late_tp_minutes its take profit is frenzy_willy_late_tp_pct instead of the WILLY TP (stop and time cap unchanged).
    Off when either is ≤ 0 / blank / unreadable, or when the late % is not below the WILLY TP (it would not be a lower target)."""
    try:
        mins = float(getattr(th, 'frenzy_willy_late_tp_minutes', 0) or 0)
        pct = float(getattr(th, 'frenzy_willy_late_tp_pct', 0) or 0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(mins) and math.isfinite(pct)) or mins <= 0 or pct <= 0 or pct >= frenzy_willy_levels(th)[0]:
        return None
    return mins, pct


def frenzy_willy_tp_at(th, held_minutes) -> Tuple[float, str]:
    """(the WILLY take profit in force at this age, its close reason): the late TP (WILLY_LATE_TP) once the order is ≥ the late minutes old,
    else the WILLY TP (FRENZY_TP). Unknown age → the WILLY TP. ONE rule for the exit, the resting limit and the sweeps."""
    late = frenzy_willy_late_tp(th)
    try:
        if late is not None and held_minutes is not None and float(held_minutes) >= late[0]:
            return late[1], "WILLY_LATE_TP"
    except (TypeError, ValueError):
        pass
    return frenzy_willy_levels(th)[0], "FRENZY_TP"


def frenzy_willy_exit_for(pnl, peak_pnl, th, held_minutes=None, stop_floor=None, resting_tp=False) -> Tuple[bool, str, Optional[float]]:
    """FRENZY_WILLY's own exit → (close, reason, line). Order: TP — pnl ≥ +tp → FRENZY_TP; a peak that already reached +tp while the close was
    missed (feed gap / failed close / restart) closes at once → FRENZY_TP if the exit pnl ≥ 0 else FRENZY_TP_LATE (never labelled a TP at a
    loss) · a configured stop (frenzy_willy_stop_pct > 0; 4.0 since DECISION_LOG 271) → STOP_LOSS · held ≥ the cap → MAX_HOLD_TIME (line = pnl).
    stop_floor: optional cap. The engine passes None since DECISION_LOG 271 — WILLY's own stop runs in paper AND live; the per-order
    exchange backstop rests just beyond it (order_backstop_pct).
    P&L = the caller's net-of-fees % of the position.
    ⏰ (DECISION_LOG 297) from age ≥ frenzy_willy_late_tp_minutes the take profit is frenzy_willy_late_tp_pct → WILLY_LATE_TP (current P&L
    only); the +tp branches below stay (a jump straight past +tp still books FRENZY_TP).
    resting_tp=True (🎯 DECISION_LOG 296): a resting exchange TP limit is alive — neither TP branch fires here (the limit books the TP; the
    engine runs the through + grace safety net); stop and time cap are judged exactly as before."""
    tp, stop, mh = frenzy_willy_levels(th)
    try:
        p = float(pnl); pk = float(peak_pnl or 0.0)
        if stop_floor is not None:
            stop = float(stop_floor) if stop is None else max(stop, float(stop_floor))
        if p >= tp and not resting_tp:
            return True, "FRENZY_TP", tp
        late_tp, late_reason = frenzy_willy_tp_at(th, held_minutes)
        if late_reason == "WILLY_LATE_TP" and p >= late_tp and not resting_tp:   # ⏰ (297) late TP: the CURRENT P&L only (a peak before
            return True, "WILLY_LATE_TP", late_tp                                  # the late minute is no fill at the late target)
        if pk >= tp and not resting_tp:
            return True, ("FRENZY_TP" if p >= 0 else "FRENZY_TP_LATE"), p
        if stop is not None and p <= stop:
            return True, "STOP_LOSS", stop
        if held_minutes is not None and float(held_minutes) >= mh:
            return True, "MAX_HOLD_TIME", p
        return False, "MAX_HOLD_TIME", stop
    except (TypeError, ValueError):
        return False, "MAX_HOLD_TIME", stop


WILLY_REVERT_N = 20                       # FROZEN: the first 20 fills (by OPEN time) — all closed — average < 0 → REVERT (operator decision)
WILLY_CAPLOSS_N, WILLY_CAPLOSS_SHARE = 10, 50.0   # FROZEN: the first 10 fills — all closed — share closed at the time cap AT A LOSS > 50 % → REVIEW


def frenzy_willy_reads(fills) -> dict:
    """🎲 The two frozen FRENZY_WILLY reads — ONE definition for the dashboard (main.py) and the scout (scout_frenzy_exits tracker 15).
    fills = iterable of (opened_key, pnl_pct or None while open, close_reason); the cohorts are the FIRST fills by opened_key (open ones
    included — a cohort is fixed, never re-picked), each read judged only once all its fills closed.
    → dict(revert='revert'|'holds'|'collecting', revert_avg, revert_closed, review='review'|'ok'|'collecting', capped_loss_share, review_closed)."""
    rows = sorted(((str(k), p, str(r or '')) for k, p, r in (fills or [])), key=lambda x: x[0])

    def _num(v):
        try:
            v = float(v)
            return v if v == v else None
        except (TypeError, ValueError):
            return None
    f20 = [(_num(p), r) for _, p, r in rows[:WILLY_REVERT_N]]
    n20 = sum(1 for p, _ in f20 if p is not None)
    out = dict(revert="collecting", revert_avg=None, revert_closed=n20, review="collecting", capped_loss_share=None, review_closed=0)
    if len(f20) >= WILLY_REVERT_N and n20 == WILLY_REVERT_N:
        out["revert_avg"] = sum(p for p, _ in f20) / WILLY_REVERT_N
        out["revert"] = "revert" if out["revert_avg"] < 0 else "holds"
    f10 = [(_num(p), r) for _, p, r in rows[:WILLY_CAPLOSS_N]]
    n10 = sum(1 for p, _ in f10 if p is not None)
    out["review_closed"] = n10
    if len(f10) >= WILLY_CAPLOSS_N and n10 == WILLY_CAPLOSS_N:
        out["capped_loss_share"] = 100.0 * sum(1 for p, r in f10 if r.startswith("MAX_HOLD_TIME") and p < 0) / WILLY_CAPLOSS_N
        out["review"] = "review" if out["capped_loss_share"] > WILLY_CAPLOSS_SHARE else "ok"
    return out


def frenzy_willy_reads_text(rd, hold_min=WILLY_HOLD_DEF) -> str:
    """the dashboard / scout wording of frenzy_willy_reads (both reads, always)."""
    if rd["revert"] == "revert":
        a = f"🛑 REVERT (operator decision): turn FRENZY_WILLY off — the first {WILLY_REVERT_N} fills average {rd['revert_avg']:+.3f}% < 0"
    elif rd["revert"] == "holds":
        a = f"✅ revert read holds — the first {WILLY_REVERT_N} fills average {rd['revert_avg']:+.3f}% ≥ 0"
    else:
        a = f"⏳ revert read {min(rd['revert_closed'], WILLY_REVERT_N)}/{WILLY_REVERT_N} closed"
    if rd["review"] == "review":
        b = (f"⚠ REVIEW — {rd['capped_loss_share']:.0f}% of the first {WILLY_CAPLOSS_N} fills closed at the {hold_min}-min cap at a loss "
             f"(> {WILLY_CAPLOSS_SHARE:g}%)")
    elif rd["review"] == "ok":
        b = f"✅ {rd['capped_loss_share']:.0f}% of the first {WILLY_CAPLOSS_N} closed at the {hold_min}-min cap at a loss (≤ {WILLY_CAPLOSS_SHARE:g}%)"
    else:
        b = f"⏳ cap-loss read {min(rd['review_closed'], WILLY_CAPLOSS_N)}/{WILLY_CAPLOSS_N} closed"
    return f"{a} · {b} · no automatic off"


WILLY_RED_WAIT_DEF = 60   # frenzy_willy_red_max_wait_minutes default (operator Oct-8: enter after the FIRST RED 5m candle, ≤ 60 min after the trigger)


def frenzy_willy_wait_ms(th) -> int:
    """the red-candle wait window in ms (blank / < 0 / unreadable → 60 min; 0 = only the trigger bar itself)."""
    w = _f(th, 'frenzy_willy_red_max_wait_minutes', WILLY_RED_WAIT_DEF)
    return int(round((w if w >= 0 else WILLY_RED_WAIT_DEF) * 60_000))


def frenzy_willy_red(ep) -> Optional[bool]:
    """the closed 5m bar the pass judged is RED (close < open — strictly; a flat bar is not red). None = unreadable (fail-closed: no entry)."""
    try:
        v = (ep or {}).get('bar_ret_pct')
        if v is None:
            return None
        v = float(v)
        return None if v != v else (v < 0)
    except (TypeError, ValueError):
        return None


def frenzy_willy_pending_clean(v) -> Optional[dict]:
    """a stored pending entry → its validated dict (trig 'A' / 'B', spike / armed / exp ms ints, why text, blocked flag) or None (dropped)."""
    try:
        if not isinstance(v, dict) or v.get('trig') not in ("A", "B"):
            return None
        _tw = v.get('turnover')
        return dict(trig=v['trig'], spike=int(v['spike']), armed=int(v['armed']), exp=int(v['exp']), why=str(v.get('why') or '')[:160],
                    blocked=bool(v.get('blocked')),
                    turnover=(_tw if _tw in ("FRENZY_WILLY_TURNOVER", "FRENZY_WILLY_TURNOVER_UNREAD") else None),
                    tkinds=[k for k in (v.get('tkinds') or []) if k in ("FRENZY_WILLY_TURNOVER", "FRENZY_WILLY_TURNOVER_UNREAD")])
    except (TypeError, ValueError, KeyError):
        return None


def vol_mcap_ratio(vol24, mcap) -> Optional[float]:
    """🔄 turnover R = 24 h quote volume / market cap (both USD). None when either is missing / ≤ 0 / not finite. Never raises."""
    try:
        v, m = float(vol24), float(mcap)
        if v != v or m != m or v in (float('inf'),) or m in (float('inf'),) or v <= 0 or m <= 0:
            return None
        return round(v / m, 6)
    except (TypeError, ValueError):
        return None


def frenzy_willy_new_flag(prev_spike_ts, seen_spike_ts, spike_ts) -> bool:
    """Entry A: True = this pass is the first on which the pair is flagged for THIS episode (spike_ts) — it was not in the engine's flag set
    with the same spike on the previous judged bar (prev_spike_ts) and no earlier pass / process recorded it (seen_spike_ts: the newest
    episode this pair was ever seen flagged with — persisted, so a restart or a toggle never fires A for an episode already flagged)."""
    try:
        if spike_ts is None:
            return False
        s = int(spike_ts)
        if prev_spike_ts is not None and int(prev_spike_ts) == s:
            return False
        return seen_spike_ts is None or s > int(seen_spike_ts)
    except (TypeError, ValueError):
        return False


WILLY_A_COLD_BARS_MAX = 12  # hard ceiling (one hour of 5m bars)
WILLY_A_COLD_BARS_DEF = 3   # frenzy_willy_a_cold_max_bars default (Oct-8 GTCUSDT miss: a pair that joins FRENZY's read late may still fire A ≤ 3 bars = 15 min after its spike bar)


def frenzy_willy_a_cold_bars(th) -> int:
    """entry A's late window for a COLD pair (not judged by this process on the previous bar), in closed 5m bars between the spike bar's close
    and the close of the bar just judged. blank / < 0 / unreadable → 3; 0 = the old rule (only a spike on the bar just closed); clamped to
    ≤ 12 (one hour) so a restart can never fire A for an episode older than an hour that no process saw."""
    w = _f(th, 'frenzy_willy_a_cold_max_bars', WILLY_A_COLD_BARS_DEF)
    if w != w or w < 0:
        return WILLY_A_COLD_BARS_DEF
    return int(min(w, WILLY_A_COLD_BARS_MAX))


def frenzy_willy_late_bars(spike_ts, bar_open) -> Optional[int]:
    """how many closed 5m bars the spike bar's close (spike_ts) lies before the close of the bar just judged (bar_open = the forming bar's
    open): 0 = the spike bar IS the bar just closed. None = unreadable (the caller treats it as too late — fail-closed)."""
    try:
        d = int(bar_open) - int(spike_ts)
        return max(0, d // BAR_MS)
    except (TypeError, ValueError):
        return None


def frenzy_breaks(bars) -> List[int]:
    """SHORT observation: which lines (50 / 200) did the LAST closed 5m bar break — a close below the EMA of 5m closes with the
    previous 12 closes all at/above it (the research trigger of scripts/break_short_review.py). [] when none / too few bars."""
    out = []
    try:
        c = [float(r[4]) for r in (bars or [])]
        if len(c) < 260:
            return out
        for span in (50, 200):
            a = 2.0 / (span + 1); e = c[0]; ema = []
            for x in c:
                e = e + a * (x - e); ema.append(e)
            if c[-1] < ema[-1] and all(c[k] >= ema[k] for k in range(len(c) - 13, len(c) - 1)):
                out.append(span)
    except (TypeError, ValueError, IndexError):
        return []
    return out


def merge_klines(cached, fresh, keep: int):
    """⚡ Oct-5 (DECISION_LOG 213): the window a full `keep`-bar fetch would return, rebuilt from the previous pass's window (`cached`) and a
    short fetch of the newest bars (`fresh`, the exchange's own rows — the forming bar included). Fresh rows replace cached rows from their
    first open on (so the previous pass's forming bar and any late revision are overwritten), then the last `keep` rows are kept.
    None (→ the caller does a full fetch) when the two do not join: empty input, bad rows, fresh not ascending / not contiguous, or fresh
    rows that do not cover the cache's last bar (that bar was still forming when it was cached). Pure; never raises."""
    try:
        if not cached or not fresh or keep <= 0:
            return None
        fo = [int(r[0]) for r in fresh]
        if any(b - a != BAR_MS for a, b in zip(fo, fo[1:])):
            return None
        last_c = int(cached[-1][0])
        if fo[0] > last_c or fo[-1] < last_c:   # the fresh rows must COVER the cache's last row — it was the previous pass's forming bar
            return None                           # (review: a tail starting right after it would keep that half-built bar)
        head = [r for r in cached if int(r[0]) < fo[0]]
        if head and fo[0] - int(head[-1][0]) != BAR_MS:
            return None
        out = head + [list(r) for r in fresh]
        return out[-keep:]
    except (TypeError, ValueError, IndexError):
        return None
