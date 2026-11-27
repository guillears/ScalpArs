"""🤫 DECISION_LOG 654 — FLATBASE quiet-base observe stamp + scout (pure parts, no network)."""
import math
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    p in sys.path or sys.path.insert(0, p)

from services.flatbase import base_vol14_from_bars, QUIET_BASE_VOL_MAX_PCT, DAY_MS, BAR_MS  # noqa: E402
import scout_fb_quiet_base_654 as S  # noqa: E402

BC = 1_800_000_000_000 // BAR_MS * BAR_MS


def _rows(day_closes, start_pad_days=1):
    """5m rows whose daily last close follows day_closes (14 values), plus a pad before the window."""
    lo = BC - 14 * DAY_MS
    rows = []
    t = lo - start_pad_days * DAY_MS
    while t + BAR_MS <= BC:
        d = (t - lo) // DAY_MS
        c = day_closes[max(0, min(13, d))] if t >= lo else day_closes[0]
        rows.append([t, c, c, c, c, 1.0])
        t += BAR_MS
    return rows


def test_vol_matches_population_std_of_daily_log_returns():
    cs = [1.0 * (1.03 if i % 2 else 0.98) ** i for i in range(14)]
    lr = [math.log(cs[i] / cs[i - 1]) for i in range(1, 14)]
    m = sum(lr) / 13
    want = math.sqrt(sum((x - m) ** 2 for x in lr) / 13) * 100
    assert base_vol14_from_bars(_rows(cs), BC) == pytest.approx(want, rel=1e-9)


def test_vol_unscorable_young_listing_and_thin_window():
    cs = [1.0] * 14
    assert base_vol14_from_bars(_rows(cs, start_pad_days=0)[5:], BC) is None        # history starts inside the window
    r = _rows(cs)
    assert base_vol14_from_bars(r[::2], BC) is None                                  # < 90 % of bars
    assert base_vol14_from_bars(None, BC) is None
    assert base_vol14_from_bars([[0, 1, 1, 1, "x", 1]], BC) is None


def test_flat_base_is_zero_and_quiet():
    v = base_vol14_from_bars(_rows([2.0] * 14), BC)
    assert v == 0.0 and S.side_of(v) == "quiet"


def test_side_cut_frozen():
    assert QUIET_BASE_VOL_MAX_PCT == 3.732
    assert S.side_of(3.732) == "quiet" and S.side_of(3.7321) == "other" and S.side_of(None) == "unscored" and S.side_of("nan") == "unscored"


def _fr(pcts, side):
    return pd.DataFrame(dict(pct=pcts, usd=[p * 10 for p in pcts], side=[side] * len(pcts)))


def test_retire_and_promote_reads():
    q = _fr([4.9] * 14 + [-10.2], "quiet")
    o = _fr([4.9] * 5, "other")
    assert S.retire_verdict(q.iloc[:14], o)[0] == "COLLECTING"
    assert S.retire_verdict(q, o)[0] == "RETIRED"                                    # quiet avg ≤ other avg
    assert S.retire_verdict(q, o.iloc[:4])[0] == "COLLECTING"                        # too few other fills → waits
    qq = _fr([4.9] * 27 + [-10.2] * 3, "quiet")
    allc = pd.concat([qq, _fr([4.9] * 10, "other")])
    st, det = S.promote_verdict(qq, allc)
    assert st.startswith("MULTIPLIER CANDIDATE"), det                               # 90 % WR, floor 79.6 % ≥ breakeven 67.6 %
    weak = pd.concat([qq, _fr([-10.2] * 8, "other")])
    assert S.promote_verdict(qq, weak)[0] == "KEEP OBSERVING"                        # the sleeve's floor below its breakeven


def test_wilson_and_breakeven():
    assert S.wilson_floor(18, 22) == pytest.approx(0.651, abs=0.002)                 # the DL 652 real floor
    assert S.breakeven_wr([4.93, -10.21]) == pytest.approx(10.21 / 15.14)
    assert S.breakeven_wr([1.0]) is None


def test_run_reads_stamp_and_fallback(tmp_path):
    f = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(dict(opened_at=["2026-12-01 00:00:00", "2026-12-01 04:00:00", "2026-12-01 08:00:00"], pair=["AUSDT", "BUSDT", "CUSDT"],
                      direction="SHORT", entry_strategy="FLATBASE_SHORT", status="CLOSED", pnl_percentage=[4.9, -10.2, 4.9], pnl=[5, -10, 5],
                      entry_frenzy_spike_at=["2026-11-30 20:00:00"] * 3, entry_flatbase_base_vol_pct=[2.0, 5.0, None])).to_csv(f, index=False)
    calls = []
    out = "\n".join(S.run(now_ms=1, paths=[str(f)], state_path=None, since_ms=0,
                          fetch=lambda *a: calls.append(a) or [], cache_dir=str(tmp_path / "c")))
    assert "| quiet (base daily vol ≤ 3.732 %) | 1 |" in out and "| other | 1 |" in out
    assert "| UNSCORED — never counted (1 pending a kline read) | 1 |" in out          # an empty read is never cached → retried
    assert len(calls) == 1                                                           # only the unstamped fill fetched
    assert not os.path.exists(tmp_path / "c")


def test_pending_fill_blocks_the_freeze(tmp_path):
    n = 16
    f = tmp_path / "scalpars_orders_paper_x.csv"
    o = pd.date_range("2026-01-01", periods=n + 6, freq="6h").strftime("%Y-%m-%d %H:%M:%S")
    vol = [None] + [2.0] * n + [5.0] * 5                                              # the first fill needs a kline read
    pd.DataFrame(dict(opened_at=o, pair=[f"P{i}USDT" for i in range(n + 6)], direction="SHORT", entry_strategy="FLATBASE_SHORT",
                      status="CLOSED", pnl_percentage=4.9, pnl=5, entry_frenzy_spike_at="2026-11-30 20:00:00",
                      entry_flatbase_base_vol_pct=vol)).to_csv(f, index=False)
    st = tmp_path / "st.json"

    def boom(*a):
        raise OSError("down")
    S.run(now_ms=1, paths=[str(f)], state_path=str(st), since_ms=0, fetch=boom, cache_dir=str(tmp_path / "c"))
    assert not st.exists() or "FB_QUIET_BASE@retire" not in open(st).read()          # a pending earlier fill → nothing frozen
    S.run(now_ms=1, paths=[str(f)], state_path=str(st), since_ms=0, fetch=lambda *a: [[0, 1, 1, 1, 1, 1]], cache_dir=str(tmp_path / "c"))
    assert "FB_QUIET_BASE@retire" in open(st).read()                                  # read once it resolved (unscorable → cached None)


def test_registered_everywhere():
    assert "scout_fb_quiet_base_654" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "scout_fb_quiet_base_654.py" in open(os.path.join(ROOT, "scripts", "observe_registry.py")).read()
    import models
    assert "entry_flatbase_base_vol_pct" in models.Order.__table__.columns
    assert "base_vol_pct" in models.FlatbaseEpisode.__table__.columns


def test_crossing_needs_other_fills_and_open_fill_blocks():
    sc = pd.DataFrame(dict(side=["quiet"] * 15 + ["other"] * 5))
    assert len(S.crossing_prefix(sc, 15, 5)) == 20 and len(S.crossing_prefix(sc, 15)) == 15
    t = pd.Timestamp("2026-01-02")
    assert S.opened_before([pd.Timestamp("2026-01-01")], t) and not S.opened_before([pd.Timestamp("2026-01-03")], t)


def test_vol_needs_12_daily_closes():
    lo = BC - 14 * DAY_MS
    rows = [[t, 1, 1, 1, 1.0 + ((t - lo) // DAY_MS) * 0.01, 1] for t in range(lo - DAY_MS, BC, BAR_MS)]
    assert base_vol14_from_bars(rows, BC) is not None
    holes = [r for r in rows if not (lo <= r[0] < lo + 3 * DAY_MS and r[0] % (2 * BAR_MS))]   # 3 days thinned: coverage still ≥ 90 %? no → None
    assert base_vol14_from_bars(holes, BC) is None
