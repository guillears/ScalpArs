#!/usr/bin/env python3
"""
Cross-batch trajectory-based counterfactual.

For each closed trade across all archived batches + the latest live CSV,
build a chronological list of all known (time, P&L) points and simulate
the first-fire mechanism between candidate exit rules:

    - TP threshold (e.g., 0.60% net)
    - BE Layer 1 (trigger 0.20% net -> floor 0.10% net once armed)
    - SL (e.g., -0.70% net)
    - Fallback: actual close (if no candidate exit triggers within our horizon)

P&L columns in CSV are net-of-fees (per engine: pnl_percentage subtracts
roundtrip ~0.063% taker). We treat the snapshot values as-is — no double
subtraction.

Trajectory points used (all in minutes from open):
    t=0                     -> 0 P&L
    t=peak_reached_at       -> peak_pnl
    t=trough_reached_at     -> trough_pnl
    t=closed_at             -> pnl_percentage (actual exit)
    t=closed + 1m..30m      -> post_exit_pnl_at_{1,2,5,15,30}min
    t=closed + post_exit_peak_minutes  -> post_exit_peak_pnl
    t=closed + post_exit_trough_minutes -> post_exit_trough_pnl

The simulator walks chronologically. At each step, between adjacent
known points (t_prev, pnl_prev) and (t_next, pnl_next), it linearly
interpolates and checks whether any exit threshold is crossed. The
FIRST crossing wins.

If TP crossed first -> exit at TP (close_pnl = TP threshold)
If BE first armed (peak >= BE trigger) and price retraces through floor -> exit at floor
If SL crossed first -> exit at SL
If trade naturally closed before any candidate fired -> use actual close

Dedup pool: (opened_at, pair, direction) per CLAUDE.md May 11 methodology.
Status filter: CLOSED only. Date filter: opened_at >= 2026-05-04.

Output:
    - Total trades simulated
    - Per close_reason breakdown (actual exit + counterfactual exit)
    - Aggregate $ comparison: actual vs CF
    - Per-direction split
    - Edge cases flagged
"""

import csv
import glob
import os
from collections import defaultdict
from datetime import datetime, timedelta

# Counterfactual parameters
TP_PCT = 0.60          # take profit threshold (net%)
BE_TRIGGER = 0.20      # BE arms when peak >= this
BE_FLOOR = 0.10        # BE exits when armed AND P&L retraces to this
SL_PCT = -0.70         # stop loss threshold (net%)

# Pool selection
START_DATE = "2026-05-04"

REPORTS_DIR = "/Users/guillearslanian/Downloads/NOFA AI/reports"
LIVE_CSV_DIR = "/Users/guillearslanian/Downloads"
LIVE_GLOB = "scalpars_orders_paper_*.csv"


def parse_dt(s):
    if not s or s == "":
        return None
    s = s.strip()
    # Try multiple formats
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def to_int_minutes(s):
    v = to_float(s)
    if v is None:
        return None
    return v


def build_trajectory(row):
    """
    Return chronologically-sorted list of (minutes_from_open, pnl_pct) points.
    Filters out None values.
    """
    opened_at = parse_dt(row.get("opened_at"))
    closed_at = parse_dt(row.get("closed_at"))
    if opened_at is None or closed_at is None:
        return None

    trade_dur_min = (closed_at - opened_at).total_seconds() / 60.0
    if trade_dur_min < 0:
        return None

    points = [(0.0, 0.0)]  # entry

    peak_pnl = to_float(row.get("peak_pnl"))
    trough_pnl = to_float(row.get("trough_pnl"))
    peak_at = parse_dt(row.get("peak_reached_at"))
    trough_at = parse_dt(row.get("trough_reached_at"))

    if peak_pnl is not None and peak_at is not None:
        t = (peak_at - opened_at).total_seconds() / 60.0
        if 0 <= t <= trade_dur_min:
            points.append((t, peak_pnl))

    if trough_pnl is not None and trough_at is not None:
        t = (trough_at - opened_at).total_seconds() / 60.0
        if 0 <= t <= trade_dur_min:
            points.append((t, trough_pnl))

    # Actual close
    close_pct = to_float(row.get("pnl_percentage"))
    if close_pct is not None:
        points.append((trade_dur_min, close_pct))

    # Post-exit snapshots
    snap_min_pnl = [
        (1, to_float(row.get("post_exit_pnl_at_1min"))),
        (2, to_float(row.get("post_exit_pnl_at_2min"))),
        (5, to_float(row.get("post_exit_pnl_at_5min"))),
        (15, to_float(row.get("post_exit_pnl_at_15min"))),
        (30, to_float(row.get("post_exit_pnl_at_30min"))),
    ]
    for offset, pnl in snap_min_pnl:
        if pnl is not None:
            points.append((trade_dur_min + offset, pnl))

    # Post-exit peak / trough
    pe_peak = to_float(row.get("post_exit_peak_pnl"))
    pe_peak_min = to_int_minutes(row.get("post_exit_peak_minutes"))
    if pe_peak is not None and pe_peak_min is not None:
        points.append((trade_dur_min + pe_peak_min, pe_peak))

    pe_trough = to_float(row.get("post_exit_trough_pnl"))
    pe_trough_min = to_int_minutes(row.get("post_exit_trough_minutes"))
    if pe_trough is not None and pe_trough_min is not None:
        points.append((trade_dur_min + pe_trough_min, pe_trough))

    points.sort(key=lambda x: x[0])
    return points, trade_dur_min


def simulate_cf(trajectory, tp, be_trigger, be_floor, sl):
    """
    Walk chronologically through trajectory. Track:
      - be_armed: True once peak hits be_trigger
      - First-fire check between adjacent points.

    Returns (cf_exit_reason, cf_exit_pct, cf_exit_min).
    If no candidate fires within the trajectory horizon, return (None, None, None).
    """
    if not trajectory or len(trajectory) < 2:
        return None, None, None

    be_armed = False

    for i in range(len(trajectory) - 1):
        t0, p0 = trajectory[i]
        t1, p1 = trajectory[i + 1]

        # First, check if at the start of this segment (the point itself) any rule triggers.
        # Arm BE if needed at point i (peak so far)
        if p0 >= be_trigger:
            be_armed = True

        # Check exits AT point i exactly
        if p0 >= tp:
            return ("TP", tp, t0)
        if be_armed and p0 <= be_floor:
            return ("BE", be_floor, t0)
        if p0 <= sl:
            return ("SL", sl, t0)

        # Now check the segment (t0, t1] — linear interpolation
        # Find any crossings
        crossings = []  # (time, kind, exit_pct)

        # TP crossing: when does p cross above tp?
        if p0 < tp <= p1:
            # Linear interp time
            if p1 != p0:
                frac = (tp - p0) / (p1 - p0)
                t_cross = t0 + frac * (t1 - t0)
                crossings.append((t_cross, "TP", tp))

        # SL crossing: when does p cross below sl?
        if p0 > sl >= p1:
            if p1 != p0:
                frac = (sl - p0) / (p1 - p0)
                t_cross = t0 + frac * (t1 - t0)
                crossings.append((t_cross, "SL", sl))

        # BE: must consider arming in segment
        if not be_armed:
            # Check if peak crosses be_trigger within segment
            if p0 < be_trigger <= p1:
                # BE arms at this crossing point. After arming, check if p1 retraces to floor.
                if p1 != p0:
                    frac = (be_trigger - p0) / (p1 - p0)
                    t_arm = t0 + frac * (t1 - t0)
                else:
                    t_arm = t0
                # BE just armed at t_arm with p = be_trigger. Won't fire in same segment unless we model the post-trigger trajectory.
                # If p1 < be_floor (we're going down after arming), we'd fire later. But within this segment we only see linear.
                # If p1 > be_trigger, BE armed but no retrace -> still armed for next segment.
                # Mark armed for next iteration.
                # Edge case: if p1 < be_floor, BE arms at t_arm then floor crossed later in same segment. Compute when:
                if p1 < be_floor:
                    # From (t_arm, be_trigger) to (t1, p1), find when crosses floor
                    if p1 != be_trigger:
                        frac2 = (be_floor - be_trigger) / (p1 - be_trigger)
                        t_floor = t_arm + frac2 * (t1 - t_arm)
                        crossings.append((t_floor, "BE", be_floor))
                be_armed = True  # will be true for next iter regardless
        else:
            # Already armed. Check floor crossing in segment.
            if p0 > be_floor >= p1:
                if p1 != p0:
                    frac = (be_floor - p0) / (p1 - p0)
                    t_cross = t0 + frac * (t1 - t0)
                    crossings.append((t_cross, "BE", be_floor))

        # Take earliest crossing in segment
        if crossings:
            crossings.sort(key=lambda x: x[0])
            t_first, kind, exit_pct = crossings[0]
            return (kind, exit_pct, t_first)

        # Also: track if peak in segment (p1 if higher) arms BE for next iter
        if p1 >= be_trigger:
            be_armed = True

    return (None, None, None)


def collect_orders():
    """Load all CSVs, dedup by (opened_at, pair, direction), filter CLOSED + date."""
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "orders_*.csv")))
    live_files = sorted(glob.glob(os.path.join(LIVE_CSV_DIR, LIVE_GLOB)))
    if live_files:
        files.append(live_files[-1])  # most recent live

    seen = set()
    rows = []
    for path in files:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") != "CLOSED":
                    continue
                opened = row.get("opened_at", "")
                if not opened or opened < START_DATE:
                    continue
                key = (opened, row.get("pair"), row.get("direction"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def main():
    print(f"Counterfactual: TP={TP_PCT}% / BE {BE_TRIGGER}/{BE_FLOOR} / SL={SL_PCT}%")
    print(f"Pool: opened_at >= {START_DATE}, dedup by (opened_at, pair, direction), CLOSED only")
    print()

    rows = collect_orders()
    print(f"Total deduped CLOSED trades: {len(rows)}")

    # Aggregates
    actual_total_usd = 0.0
    cf_total_usd = 0.0
    cf_reason_counts = defaultdict(int)
    cf_reason_dollars = defaultdict(float)
    actual_reason_counts = defaultdict(int)
    actual_reason_dollars = defaultdict(float)

    per_dir = {
        "LONG": {"n": 0, "actual_$": 0.0, "cf_$": 0.0},
        "SHORT": {"n": 0, "actual_$": 0.0, "cf_$": 0.0},
    }

    no_traj_count = 0
    no_cf_fired_count = 0  # CF couldn't determine exit -> use actual

    # Track per close_reason what CF would have done
    by_reason = defaultdict(lambda: {"n": 0, "actual_$": 0.0, "cf_$": 0.0, "cf_breakdown": defaultdict(int)})

    for row in rows:
        traj_result = build_trajectory(row)
        if traj_result is None:
            no_traj_count += 1
            continue
        trajectory, trade_dur = traj_result

        actual_pnl_pct = to_float(row.get("pnl_percentage"))
        notional = to_float(row.get("notional_value"))
        if notional is None or notional <= 0:
            # fallback: investment * leverage
            inv = to_float(row.get("investment"))
            lev = to_float(row.get("leverage"))
            if inv is not None and lev is not None:
                notional = inv * lev

        if actual_pnl_pct is None or notional is None:
            no_traj_count += 1
            continue

        actual_usd = actual_pnl_pct / 100.0 * notional
        actual_total_usd += actual_usd

        direction = row.get("direction")
        if direction in per_dir:
            per_dir[direction]["n"] += 1
            per_dir[direction]["actual_$"] += actual_usd

        actual_reason = row.get("close_reason") or "UNKNOWN"
        actual_reason_counts[actual_reason] += 1
        actual_reason_dollars[actual_reason] += actual_usd

        # Simulate CF
        cf_kind, cf_pct, cf_time = simulate_cf(
            trajectory, TP_PCT, BE_TRIGGER, BE_FLOOR, SL_PCT
        )

        if cf_kind is None:
            # No CF rule fired within trajectory horizon -> use actual
            cf_kind = "FALLBACK_ACTUAL"
            cf_pct = actual_pnl_pct
            no_cf_fired_count += 1

        cf_usd = cf_pct / 100.0 * notional
        cf_total_usd += cf_usd
        cf_reason_counts[cf_kind] += 1
        cf_reason_dollars[cf_kind] += cf_usd

        if direction in per_dir:
            per_dir[direction]["cf_$"] += cf_usd

        by_reason[actual_reason]["n"] += 1
        by_reason[actual_reason]["actual_$"] += actual_usd
        by_reason[actual_reason]["cf_$"] += cf_usd
        by_reason[actual_reason]["cf_breakdown"][cf_kind] += 1

    print(f"Trades with insufficient data (skipped): {no_traj_count}")
    print(f"Trades where no CF rule fired (used actual fallback): {no_cf_fired_count}")
    print()

    print("=" * 78)
    print("AGGREGATE COMPARISON")
    print("=" * 78)
    n_total = sum(d["n"] for d in per_dir.values())
    print(f"  N total:     {n_total}")
    print(f"  Actual $:    {actual_total_usd:+.2f}")
    print(f"  CF $:        {cf_total_usd:+.2f}")
    print(f"  Delta $:     {cf_total_usd - actual_total_usd:+.2f}")
    print()

    for d, vals in per_dir.items():
        if vals["n"] > 0:
            delta = vals["cf_$"] - vals["actual_$"]
            print(f"  {d:6} N={vals['n']:3}  actual={vals['actual_$']:+9.2f}  cf={vals['cf_$']:+9.2f}  delta={delta:+9.2f}")

    print()
    print("=" * 78)
    print("CF EXIT DISTRIBUTION")
    print("=" * 78)
    for kind in ["TP", "BE", "SL", "FALLBACK_ACTUAL"]:
        n = cf_reason_counts.get(kind, 0)
        d = cf_reason_dollars.get(kind, 0.0)
        avg = d / n if n > 0 else 0.0
        print(f"  {kind:18} N={n:4}  total $={d:+9.2f}  avg $={avg:+7.2f}")

    print()
    print("=" * 78)
    print("BY ACTUAL CLOSE REASON  (actual_$ vs cf_$)")
    print("=" * 78)
    sorted_reasons = sorted(by_reason.items(), key=lambda x: -x[1]["n"])
    for reason, vals in sorted_reasons:
        if vals["n"] < 3:
            continue
        delta = vals["cf_$"] - vals["actual_$"]
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(vals["cf_breakdown"].items(), key=lambda x: -x[1]))
        print(f"  {reason:25} N={vals['n']:3}  actual={vals['actual_$']:+9.2f}  cf={vals['cf_$']:+9.2f}  delta={delta:+9.2f}")
        print(f"      CF dist: {breakdown}")


if __name__ == "__main__":
    main()
