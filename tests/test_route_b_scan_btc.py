"""Route B weekly-scan PREREG build item 4 — the BTC daily-trend stamp (1 d EMA50 vs EMA200 on the last CLOSED day). No network."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import btc_trend as BT   # noqa: E402


def _days(closes, start=0):
    return pd.DataFrame(dict(open_ms=start + np.arange(len(closes), dtype=np.int64) * BT.DAY_MS, c=closes))


def test_trend_sign_and_warmup():
    up = np.linspace(100, 300, 400)
    tb = BT.trend_table(_days(up))
    assert np.isnan(tb.trend.iloc[BT.MIN_DAYS - 2]) and tb.trend.iloc[-1] == 1.0
    dn = BT.trend_table(_days(up[::-1]))
    assert dn.trend.iloc[-1] == -1.0


def test_stamp_reads_only_closed_days():
    c = np.r_[np.linspace(100, 300, 300), np.linspace(300, 10, 100)]
    tb = BT.trend_table(_days(c))
    T = int(tb.close_ms.iloc[299]) + 5 * 3600_000                     # 5 h into day 300 (open, not read)
    st = BT.stamp(tb, T)
    assert st["btc_d_ema50"] == pytest.approx(tb.ema50.iloc[299])
    edge = BT.stamp(tb, int(tb.close_ms.iloc[299]))                     # exactly at the close: that day is closed
    assert edge["btc_d_ema50"] == pytest.approx(tb.ema50.iloc[299])
    before = BT.stamp(tb, int(tb.close_ms.iloc[299]) - 1)
    assert before["btc_d_ema50"] == pytest.approx(tb.ema50.iloc[298])
    assert math.isnan(BT.stamp(tb, -1)["btc_d_trend"])


def test_stamp_causal_on_future_changes():
    c = np.linspace(100, 300, 400)
    tb = BT.trend_table(_days(c))
    T = int(tb.close_ms.iloc[350]) + 1000
    c2 = c.copy()
    c2[351:] = 1.0
    tb2 = BT.trend_table(_days(c2))
    assert BT.stamp(tb, T) == BT.stamp(tb2, T)


def test_load_uses_cache_then_fetches(tmp_path):
    class Lim:
        calls = 0

        def get(self, path, params, w):
            Lim.calls += 1
            return [[int(i * BT.DAY_MS), "0", "0", "0", str(100 + i)] for i in range(400)]
    tb = BT.load(Lim(), need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path))
    assert Lim.calls == 1 and int(tb.close_ms.max()) == 300 * BT.DAY_MS          # days still open at need are not kept
    BT.load(Lim(), need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path))
    assert Lim.calls == 1                                                            # cache covers it


def test_stale_table_is_unscored():
    tb = BT.trend_table(_days(np.linspace(100, 300, 400)))
    last = int(tb.close_ms.iloc[-1])
    assert BT.stamp(tb, last + BT.DAY_MS - 1)["btc_d_trend"] == 1.0
    assert math.isnan(BT.stamp(tb, last + BT.DAY_MS)["btc_d_trend"])               # one missed daily close → unscored


def test_mid_day_need_is_floored_and_open_day_never_cached(tmp_path):
    class Lim:
        calls = 0

        def get(self, path, params, w):
            Lim.calls += 1
            return [[int(i * BT.DAY_MS), "0", "0", "0", str(100 + i)] for i in range(400)]
    need = 300 * BT.DAY_MS + 5 * 3600_000                                   # mid-day
    tb = BT.load(Lim(), need_close_ms=need, data_dir=str(tmp_path))
    assert int(tb.close_ms.max()) == 300 * BT.DAY_MS
    BT.load(Lim(), need_close_ms=need + 3600_000, data_dir=str(tmp_path))
    assert Lim.calls == 1                                                   # floored: the cache covers it, no refetch
    os.makedirs(tmp_path / "x")
    tb2 = BT.load(Lim(), need_close_ms=350 * BT.DAY_MS, data_dir=str(tmp_path / "x"), now_ms=320 * BT.DAY_MS + 7)
    assert int(tb2.close_ms.max()) == 320 * BT.DAY_MS                        # never past now


def test_golden_ema_and_warmup_index():
    c = np.linspace(100, 300, 400)
    tb = BT.trend_table(_days(c))
    s = pd.Series(c)
    assert tb.ema50.iloc[-1] == pytest.approx(s.ewm(span=50, adjust=False).mean().iloc[-1])
    assert tb.ema200.iloc[-1] == pytest.approx(s.ewm(span=200, adjust=False).mean().iloc[-1])
    assert np.isnan(tb.trend.iloc[BT.MIN_DAYS - 2]) and not np.isnan(tb.trend.iloc[BT.MIN_DAYS - 1])
    assert BT.MIN_DAYS == 250


class _Lim:
    def __init__(self, days=400, fail=False):
        self.calls, self.days, self.fail = 0, days, fail

    def get(self, path, params, w):
        self.calls += 1
        return None if self.fail else [[int(i * BT.DAY_MS), "0", "0", "0", str(100 + i)] for i in range(self.days)]


def test_cache_ending_early_is_refetched_and_merged(tmp_path):
    BT.load(_Lim(), need_close_ms=290 * BT.DAY_MS, data_dir=str(tmp_path))
    lim = _Lim()
    tb = BT.load(lim, need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path))
    assert lim.calls == 1 and int(tb.close_ms.max()) == 300 * BT.DAY_MS and int(tb.close_ms.min()) == BT.DAY_MS


def test_fetch_failure_falls_back_to_the_cache_then_stamps_nan_past_it(tmp_path):
    BT.load(_Lim(), need_close_ms=290 * BT.DAY_MS, data_dir=str(tmp_path))
    tb = BT.load(_Lim(fail=True), need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path))
    assert int(tb.close_ms.max()) == 290 * BT.DAY_MS
    assert math.isnan(BT.stamp(tb, 300 * BT.DAY_MS)["btc_d_trend"])
    assert BT.load(_Lim(fail=True), need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path / "none")) is None


def test_corrupt_cache_is_ignored(tmp_path):
    (tmp_path / BT.CACHE).write_bytes(b"not a pickle")
    tb = BT.load(_Lim(), need_close_ms=300 * BT.DAY_MS, data_dir=str(tmp_path))
    assert int(tb.close_ms.max()) == 300 * BT.DAY_MS


def test_stamp_key_set_and_day_label():
    tb = BT.trend_table(_days(np.linspace(100, 300, 400)))
    keys = {"btc_d_trend", "btc_d_ema50", "btc_d_ema200", "btc_d_day"}
    for T in (None, -1, int(tb.close_ms.iloc[-1]) + 5 * BT.DAY_MS, int(tb.close_ms.iloc[299]) + 1):
        st = BT.stamp(None if T is None else tb, 0 if T is None else T)
        assert set(st) == keys
    st = BT.stamp(tb, int(tb.close_ms.iloc[299]) + 1)
    assert st["btc_d_day"] == pd.Timestamp(int(tb.close_ms.iloc[299]) - BT.DAY_MS, unit="ms", tz="UTC").strftime("%Y-%m-%d")


def test_adjust_false_on_a_jumpy_series():
    c = np.r_[np.full(10, 100.0), np.full(30, 200.0)]
    tb = BT.trend_table(_days(c))
    assert tb.ema50.iloc[20] == pytest.approx(pd.Series(c).ewm(span=50, adjust=False).mean().iloc[20], rel=1e-12)
