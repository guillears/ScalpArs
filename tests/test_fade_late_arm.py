"""⏱ Sep-24 FADE LATE-ARM — a spike fade still unarmed after X minutes arms at a lower peak.

Operator-directed ship (declared override at N=2: PROVE +$310, SENT +$19 vs up to −$147 of exposure on
seven late winners; DECISION_LOG 111). Thesis: a fade is decided in its first minutes; a fade that has
not faded by X minutes and then only manages a small bounce is banked instead of waiting for the 0.40 arm.

Invariants pinned here:
  · SPIKE_FADE only — every other strategy keeps the plain arm.
  · 0 / missing X or level = off. Before X the rule does not exist.
  · The late arm uses the peak reached AFTER minute X, never the peak since entry: an early pop that has
    since collapsed must NOT arm at minute X (that would silently become a time stop — refuted Sep-24).
  · The normal arm always wins when reached; late-arm never lowers a normally-armed floor.
  · The history counterfactual only rescues fades that never armed, stayed open past X, and peaked in
    [late level, base arm) AFTER X — the exposed late winners are left as lived (their path isn't stamped).
  · The base-stack shadow replicates today's fade exit (stop, 0.40 arm, trail, HARD_TP ladder) after a
    late exit, so every late decision is priced against what would have happened.
"""
from types import SimpleNamespace

import pytest

from services.trading_engine import (fade_late_arm_level, next_fade_late_peak, short_runner_arm,
                                     short_trail_floor, fade_late_arm_cf, fade_base_shadow_step,
                                     fade_base_shadow_seed, fade_shadow_timeout_reason)


def _th(**kw):
    d = dict(spike_fade_late_arm_after_min=15.0, spike_fade_late_arm_peak=0.30,
             runner_trail_short_arm_peak=0.40, runner_trail_short_atr_mult=0.5,
             runner_trail_short_giveback_frac=0.35, runner_trail_short_be_ratchet_enabled=False,
             runner_trail_short_be_lock_pct=0.10, spike_fade_sl_pct=-1.5,
             hard_tp_enabled=True, hard_tp_pct=1.0, runner_trail_short_use_atr=True,
             hard_tp_ladder_short="1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80")
    d.update(kw)
    return SimpleNamespace(**d)


# ───────────────────────────────────────────── level
def test_level_only_for_fades_after_x():
    assert fade_late_arm_level(_th(), "SPIKE_FADE", 15.0) == 0.30
    assert fade_late_arm_level(_th(), "SPIKE_FADE", 14.99) is None
    for s in ("MOMENTUM", "SPIKE_BOUNCE", "FLIP:FAN_RATIO_GATE", "BEARRUN_SHORT", None, ""):
        assert fade_late_arm_level(_th(), s, 60.0) is None


def test_level_off_and_garbage():
    for kw in (dict(spike_fade_late_arm_after_min=0), dict(spike_fade_late_arm_peak=0),
               dict(spike_fade_late_arm_after_min=None), dict(spike_fade_late_arm_after_min="x")):
        assert fade_late_arm_level(_th(**kw), "SPIKE_FADE", 60.0) is None
    assert fade_late_arm_level(SimpleNamespace(), "SPIKE_FADE", 60.0) is None
    for age in (None, "x", float("nan")):
        assert fade_late_arm_level(_th(), "SPIKE_FADE", age) is None


def test_level_at_or_above_base_arm_is_meaningless_and_off():
    assert fade_late_arm_level(_th(spike_fade_late_arm_peak=0.40), "SPIKE_FADE", 60.0) is None
    assert fade_late_arm_level(_th(spike_fade_late_arm_peak=0.55), "SPIKE_FADE", 60.0) is None


# ───────────────────────────────────────────── late peak = peak since X
def test_late_peak_starts_at_x_and_ratchets():
    th = _th()
    p = next_fade_late_peak(th, "SPIKE_FADE", 10.0, None, 0.38)
    assert p is None                                          # before X: nothing tracked (early pop ignored)
    p = next_fade_late_peak(th, "SPIKE_FADE", 15.0, p, -0.60)
    assert p == pytest.approx(-0.60)                          # first reading after X seeds it, even negative
    p = next_fade_late_peak(th, "SPIKE_FADE", 16.0, p, 0.345)
    p = next_fade_late_peak(th, "SPIKE_FADE", 17.0, p, 0.10)
    assert p == pytest.approx(0.345)
    assert next_fade_late_peak(th, "MOMENTUM", 30.0, None, 0.5) is None


# ───────────────────────────────────────────── arm decision
def test_base_arm_wins_and_uses_the_all_time_peak():
    assert short_runner_arm(_th(), "SPIKE_FADE", 60.0, 0.55, 0.31) == (True, 0.55, False)
    assert short_runner_arm(_th(), "SPIKE_FADE", 5.0, 0.3955, None) == (True, 0.3955, False)   # −0.005 tolerance kept


def test_prove_arms_late_on_its_post_x_peak():
    armed, pk, late = short_runner_arm(_th(), "SPIKE_FADE", 15.9, 0.345, 0.345)
    assert (armed, late) == (True, True) and pk == pytest.approx(0.345)


def test_early_pop_that_collapsed_does_not_arm_at_x():
    """HOLO-class: +0.389 at minute 5, −0.5 at minute 15. Since-entry peak would arm and close at a loss."""
    assert short_runner_arm(_th(), "SPIKE_FADE", 15.0, 0.389, -0.50) == (False, 0.389, False)


def test_non_fade_or_off_never_late_arms():
    assert short_runner_arm(_th(), "MOMENTUM", 60.0, 0.35, 0.35)[0] is False
    assert short_runner_arm(_th(spike_fade_late_arm_after_min=0), "SPIKE_FADE", 60.0, 0.35, 0.35)[0] is False
    assert short_runner_arm(_th(), "SPIKE_FADE", 10.0, 0.35, 0.35)[0] is False             # before X


# ───────────────────────────────────────────── trail floor (shared formula)
def test_floor_matches_the_live_short_formula():
    f, raw, capped = short_trail_floor(_th(), 0.345, 0.30)
    assert capped and f == pytest.approx(0.345 - 0.35 * 0.345)                            # 0.224
    f, raw, capped = short_trail_floor(_th(), 2.0, 0.30)
    assert not capped and f == pytest.approx(2.0 - 0.15)
    f, _, _ = short_trail_floor(_th(runner_trail_short_be_ratchet_enabled=True), 0.30, 2.0)
    assert f == pytest.approx(max(0.30 - 0.105, 0.10))


# ───────────────────────────────────────────── history counterfactual (stamps only)
def test_cf_rescues_prove_and_sent():
    assert fade_late_arm_cf(_th(), -1.527, 0.345, 15.95, 102.3, 0.30) == pytest.approx(0.345 * 0.65, abs=1e-6)
    assert fade_late_arm_cf(_th(), -0.164, 0.356, 104.7, 123.9, 0.177) == pytest.approx(0.356 - 0.5 * 0.177, abs=1e-6)   # low ATR: 0.5×ATR < 0.35×peak


def test_cf_leaves_everything_else_alone():
    th = _th()
    assert fade_late_arm_cf(th, -0.703, 0.389, 5.38, 17.3, 0.42) is None      # HOLO: peak came BEFORE X
    assert fade_late_arm_cf(th, 0.54, 0.65, 32.4, 32.4, 0.3) is None          # armed normally (peak ≥ 0.40)
    assert fade_late_arm_cf(th, -1.50, 0.09, 0.7, 1.7, 0.5) is None           # closed before X
    assert fade_late_arm_cf(th, -1.49, 0.10, 0.64, 15.8, 0.57) is None        # never reached the late level
    assert fade_late_arm_cf(th, -1.5, 0.35, None, 60.0, 0.3) is None          # unstamped peak time: no guess
    assert fade_late_arm_cf(_th(spike_fade_late_arm_after_min=0), -1.527, 0.345, 15.95, 102.3, 0.3) is None


# ───────────────────────────────────────────── base-stack shadow
def _run(th, path, peak=0.345, atr=0.30):
    st = fade_base_shadow_seed(th, peak, atr)
    for x in path:
        r = fade_base_shadow_step(st, x)
        if r is not None:
            return r
    return None


def test_shadow_prove_class_rides_to_the_stop():
    # engine stop epsilon: fires at pnl <= sl + 0.01 (realtime path), so −1.48 holds and −1.495 stops
    assert _run(_th(), [0.2, 0.0, -0.8, -1.2, -1.48, -1.495]) == (pytest.approx(-1.495), "STOP")


def test_shadow_late_winner_arms_at_040_then_trails():
    # arms once the peak reaches 0.40 (0.55); at peak 0.62 the floor is 0.62 − min(0.5×0.30, 0.35×0.62) = 0.47
    assert _run(_th(), [0.2, 0.39, 0.55, 0.62, 0.45]) == (pytest.approx(0.45), "TRAIL")


def test_shadow_ladder_backstop_on_big_runs():
    # at peak 1.6 the trail floor is 1.6 − 0.35×1.6 = 1.04, the 1.5-rung ladder floor is 1.20 → ladder binds
    assert _run(_th(), [0.8, 1.6, 1.25], atr=5.0) is None
    assert _run(_th(), [0.8, 1.6, 1.19], atr=5.0) == (pytest.approx(1.19), "LADDER")


def test_shadow_unresolved_returns_none():
    assert _run(_th(), [0.1, 0.0, 0.2]) is None


# ───────────────────────────────────────────── engine wiring parity (source level)
# The late arm lives in ONE engine site (the realtime non-flip SHORT runner). These asserts fail if the site
# stops routing through the helper, builds its floor from the since-entry peak, loses the LATE close reason,
# or if the cache rebuild stops carrying the post-X peak (it would be wiped every ~1 s and never fire).
import inspect  # noqa: E402
import services.trading_engine as te  # noqa: E402


def _src():
    return inspect.getsource(te)


def test_realtime_short_runner_routes_through_the_helper():
    s = _src()
    blk = s[s.index("REALTIME NON-FLIP SHORT RUNNER ATR-FLOOR (3rd appearance"):s.index("REALTIME NON-FLIP SHORT RUNNER ATR-FLOOR END")]
    assert "short_runner_arm(" in blk
    assert "current_peak >= _rs_arm" not in blk                  # the old hard-wired arm must be gone
    assert "_rs_raw_floor = _rs_pk - _rs_gb" in blk              # floor from the ARMING peak
    assert '"RUNNER_TRAIL_LATE" if _rs_late else "RUNNER_TRAIL"' in blk


def test_post_x_peak_is_tracked_and_survives_the_cache_rebuild():
    s = _src()
    assert "order_info['fade_late_peak'] = next_fade_late_peak(" in s
    keep = s[s.index("for _key in ('_htp_persisted_lvl', '_sp_lock_peak_persisted',"):][:400]
    assert "'fade_late_peak'" in keep and "'fade_late_armed_at'" in keep


def test_late_exit_seeds_the_base_stack_shadow():
    s = _src()
    i = s.index("self._register_post_exit_tracking(order, reason)\n")      # seeded AFTER phase 2, next to post-exit
    assert "self._seed_fade_late_shadow(order, resumed=False)" in s[i:i + 600]
    assert "await self._update_fade_late_shadow()" in s
    assert "await self._recover_fade_late_shadow(db)" in s


# ───────────────────────────────────────────── deep-review guards (2026-09-24)
def test_cf_never_reprices_a_fade_that_closed_above_the_late_floor():
    """I3: peak 0.35 after X, closed +0.30 by another exit — the trail never bound; must NOT be dragged to 0.2275."""
    assert fade_late_arm_cf(_th(), 0.30, 0.35, 20.0, 40.0, 0.30) is None
    assert fade_late_arm_cf(_th(), 0.10, 0.35, 20.0, 40.0, 0.30) == pytest.approx(0.35 * 0.65, abs=1e-6)


def test_shadow_uses_the_orders_own_stop():
    st = fade_base_shadow_seed(_th(), 0.2, 0.3, stop=-0.70)
    assert fade_base_shadow_step(st, -0.695) == (pytest.approx(-0.695), "STOP")


def test_shadow_hard_tp_snapshot_off_ladder_and_flat_cap():
    assert fade_base_shadow_seed(_th(hard_tp_enabled=False), 0.2, 0.3)['rungs'] is None
    st = fade_base_shadow_seed(_th(hard_tp_ladder_short=""), 0.2, 0.3)
    assert st['rungs'] is None and st['flat_cap'] == pytest.approx(1.0)
    assert fade_base_shadow_step(st, 1.01) == (pytest.approx(1.01), "HARD_TP")


def test_shadow_ladder_does_not_need_atr():
    st = fade_base_shadow_seed(_th(), 1.6, None)
    assert fade_base_shadow_step(st, 1.19) == (pytest.approx(1.19), "LADDER")


def test_timeout_reason_distinguishes_a_dead_feed():
    assert fade_shadow_timeout_reason(0, False) == "NO_FEED"
    assert fade_shadow_timeout_reason(12, False) == "TIMEOUT"
    assert fade_shadow_timeout_reason(0, True) == "NO_FEED_RESUMED"


def test_seed_is_prefix_safe_and_protected_from_the_ws_prune():
    s = _src()
    assert '_strip_reason_prefixes(reason).startswith("RUNNER_TRAIL_LATE")' in s
    assert "like('%RUNNER_TRAIL_LATE%')" in s
    assert "self._fade_late_shadow.values()}" in s[s.index("_ws_keep = ("):][:600]
