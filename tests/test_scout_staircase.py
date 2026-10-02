"""🪜 Scout staircase watch — the pure state rule, the scan and the notes (alert only)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_staircase as S  # noqa: E402

T0 = 1_790_000_000_000 // 300_000 * 300_000
NORM = 12_000.0                                                          # normal hour = 12 bars × $1,000
QUIET, HOT = 1000.0, 400_000.0


def frame(closes, quotes):
    c = np.array(closes, float)
    return pd.DataFrame({"h": c * 1.001, "l": c * 0.999, "c": c, "q": np.array(quotes, float)}, index=T0 + np.arange(len(c)) * 300_000)


def climb(n, a=1.06, b=1.60):
    return list(np.linspace(a, b, n))


def test_no_spike_no_episode():
    assert S.staircase_state(frame([1.0] * 400, [QUIET] * 400), NORM) is None
    assert S.staircase_state(None, 1.0) is None and S.staircase_state(frame([1.0] * 100, [1.0] * 100), 0) is None
    assert S.staircase_state(frame([1.0] * 100, [1.0] * 100), float("nan")) is None


def test_state_needs_two_hours_an_hour_above_and_volume():
    pre = 300                                                            # 25 h of quiet history → the onset can be verified
    c = [1.0] * pre + climb(40); q = [QUIET] * pre + [HOT] * 40
    s = S.staircase_state(frame(c, q), NORM)
    assert s["onset_ts"] == T0 + (pre + 4) * 300_000                     # first bar whose last-hour volume reaches $2M (5 hot bars) with +5 % in 30 min
    assert s["in_state"] and s["above_hour"] and s["volx"] >= 100 and s["onset_verified"] and s["base"] == 1.0 and s["price"] > s["vwap"]
    on = pre + 4
    for bars_after, want in ((23, False), (24, True)):                   # the tested rule first evaluates the state 24 bars after the onset
        k = on + bars_after + 1
        assert S.staircase_state(frame(c[:k], q[:k]), NORM)["in_state"] is want
    faded = S.staircase_state(frame(c + [1.10] * 3, q + [HOT] * 3), NORM)             # back through the average price
    assert not faded["in_state"] and not faded["above_hour"]
    thin = S.staircase_state(frame(c + climb(12, 1.61, 1.70), q + [20_000.0] * 12), NORM)   # still above, volume gone (20×)
    assert thin["above_hour"] and not thin["in_state"] and thin["volx"] < S.STATE_VOLX


def test_onset_follows_the_research_episode_walk():
    pre = 300
    # (a) a live staircase that makes a fresh +5 % push 30 h in keeps its ORIGINAL onset (no re-anchor, no lost "still on" clock)
    c = [1.0] * pre + climb(360, 1.06, 3.0); q = [QUIET] * pre + [HOT] * 360
    c += list(np.linspace(3.0, 3.4, 8)); q += [HOT] * 8
    s = S.staircase_state(frame(c, q), NORM)
    assert s["onset_ts"] == T0 + (pre + 4) * 300_000 and s["hours"] > 30 and s["in_state"]
    # (b) a spike that never reached the state, then > 24 h of nothing, then a new spike → the NEW spike is the onset
    c2 = [1.0] * pre + [1.08] * 6 + [1.0] * 320 + climb(40); q2 = [QUIET] * pre + [HOT] * 6 + [QUIET] * 320 + [HOT] * 40
    s2 = S.staircase_state(frame(c2, q2), NORM)
    assert s2["onset_ts"] > T0 + (pre + 300) * 300_000 and s2["hours"] < 4 and s2["in_state"]
    # (c) an old spike that died more than 24 h ago and nothing since → no live episode
    c3 = [1.0] * pre + [1.08] * 6 + [1.0] * 320; q3 = [QUIET] * pre + [HOT] * 6 + [QUIET] * 320
    assert S.staircase_state(frame(c3, q3), NORM) is None
    # (d) an onset in the first 24 h of the window cannot be verified
    s4 = S.staircase_state(frame([1.0] * 60 + climb(40), [QUIET] * 60 + [HOT] * 40), NORM)
    assert s4 is not None and not s4["onset_verified"]


class _Ex:
    """Fake exchange: one staircase pair, one flat pair, one that cannot be read, one too new."""
    def __init__(self, last_closed):
        self.lc = last_closed

    def fapiPublicGetTicker24hr(self):
        return [{"symbol": "AAAUSDT", "priceChangePercent": "44", "quoteVolume": "5e8"}, {"symbol": "BBBUSDT", "priceChangePercent": "20", "quoteVolume": "5e7"},
                {"symbol": "CCCUSDT", "priceChangePercent": "30", "quoteVolume": "5e7"}, {"symbol": "NEWUSDT", "priceChangePercent": "90", "quoteVolume": "5e7"},
                {"symbol": "BTCUSDT", "priceChangePercent": "50", "quoteVolume": "1e10"}, {"symbol": "DDDUSDT", "priceChangePercent": "3", "quoteVolume": "5e9"},
                {"symbol": "EEEUSDT", "priceChangePercent": "40", "quoteVolume": "1e6"}]

    def fapiPublicGetKlines(self, p):
        sym, iv = p["symbol"], p["interval"]
        if sym == "CCCUSDT":
            raise RuntimeError("down")
        if iv == "1h":
            n = 100 if sym == "NEWUSDT" else 744
            return [[self.lc - (n - i) * 3600_000, 0, 0, 0, 0, 0, 0, str(NORM)] for i in range(n)]
        c = ([1.0] * 300 + climb(40)) if sym == "AAAUSDT" else [1.0] * 340; q = ([QUIET] * 300 + [HOT] * 40) if sym == "AAAUSDT" else [QUIET] * 340
        rows = [[self.lc - (len(c) - 1 - i) * 300_000, 0, c[i] * 1.001, c[i] * 0.999, c[i], 0, 0, str(q[i])] for i in range(len(c))]
        return rows + [[self.lc + 300_000, 0, 9.0, 0.1, 0.1, 0, 0, "1"]]           # the forming bar must be ignored


def test_scan_shortlist_columns_forming_bar_and_failures():
    rows, n, bad = S.scan(_Ex(T0), T0)
    assert [r["pair"] for r in rows] == ["AAAUSDT"] and rows[0]["in_state"] and rows[0]["price"] > 1.5 and rows[0]["chg24"] == 44.0
    assert n == 4 and bad == 1                                            # AAA, CCC (unreadable), BBB, NEW (too new: skipped, not a failure); BTC / +3 % / $1M never shortlisted
    assert S.scan(_Ex(T0), T0, extra=["DDDUSDT", "ZZZUSDT"])[1] == 5       # a followed pair is checked even when its 24 h change has cooled; an unknown one is ignored

    class _Boom:
        def fapiPublicGetTicker24hr(self): raise RuntimeError("down")
    assert S.scan(_Boom(), T0) is None and any("Unavailable" in x for x in S.lines(None))


def test_lines_notes_and_followed():
    assert any("No pair" in x for x in S.lines(([], 7, 0))) and any("2 of 7" in x for x in S.lines(([], 7, 2)))
    row = dict(pair="SANDUSDT", chg24=44.0, onset_ts=T0, base=0.047, price=0.068, vwap=0.064, hours=6.5, volx=156.0, above_hour=True, above_share=1.0, in_state=True, onset_verified=True)
    res = ([row, {**row, "pair": "NIGHTUSDT", "hours": 40.0, "volx": 64.0}, {**row, "pair": "GTCUSDT", "in_state": False, "above_hour": False, "volx": 12.0}], 9, 0)
    out = "\n".join(S.lines(res))
    assert "| ★ ON |" in out and "| ON · ⏳ |" in out and "below average" in out and "NOT established" in out and "never a signal" in out
    items = S.note_items(res, {}, T0)
    assert [k for k, _ in items] == [f"ST|SANDUSDT|{T0}", f"ST|NIGHTUSDT|{T0}", f"ST32|NIGHTUSDT|{T0}"]
    noted = {f"ST|SANDUSDT|{T0 - 3 * 3600_000}": T0, f"ST|NIGHTUSDT|{T0}": T0}          # same episode even if the onset shifted by a few hours
    assert [k for k, _ in S.note_items(res, noted, T0)] == [f"ST32|NIGHTUSDT|{T0}"]
    assert S.note_items(None, {}, T0) == [] and S.note_items(([], 0, 0), {}, T0) == []
    assert S.followed({f"ST|SANDUSDT|{T0}": T0, "S|TREND|XUSDT|UP|1": T0, f"ST|OLDUSDT|{T0}": T0 - 200 * 3600_000}, T0) == ["SANDUSDT"]
