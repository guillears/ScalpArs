"""🏷 DECISION_LOG 314: the WILLY_TURNOVER_BLOCKED watch line is split by market cap (observe-only, never gated). A blocked row's mcap is
rebuilt as its 24 h futures quote volume at the refusal ÷ R (the bot's own two inputs); let-through fills use entry_mcap_usd."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_frenzy_exits as SF  # noqa: E402


def test_bucket_edges():
    assert SF.tb_mcap_bucket(49e6) == "< $50M" and SF.tb_mcap_bucket(50e6) == "$50–100M"
    assert SF.tb_mcap_bucket(199.9e6) == "$100–200M" and SF.tb_mcap_bucket(271e6) == "≥ $200M"
    assert SF.tb_mcap_bucket(None) == "mcap unknown" and SF.tb_mcap_bucket(np.nan) == "mcap unknown" and SF.tb_mcap_bucket(0) == "mcap unknown"


def _rows():
    return pd.DataFrame([
        dict(t="2026-10-20T01:00:00", pair="AGTUSDT", gate="FRENZY_WILLY_TURNOVER", trig="A", R=1.11, price=0.08, pct=1.25, how="take profit",
             tries=0, src="ticks", final=True, entry=0.08, err=np.nan),
        dict(t="2026-10-20T02:00:00", pair="XUSDT", gate="FRENZY_WILLY_TURNOVER_UNREAD", trig="A", R=np.nan, price=1.0, pct=np.nan, how=np.nan,
             tries=0, src=np.nan, final=np.nan, entry=np.nan, err=np.nan),
    ])


def test_fill_mcap_is_vol_over_R_and_never_refetched(monkeypatch):
    calls = []
    monkeypatch.setattr(SF, "tb_vol24", lambda pair, t, used=None: calls.append(pair) or 300e6)
    monkeypatch.setattr(SF.time, "sleep", lambda s: None)
    r = SF.tb_fill_mcap(_rows())
    assert calls == ["AGTUSDT"]                                       # the UNREAD row (no R) is never fetched
    assert abs(r.loc[0, "mcap"] - 300e6 / 1.11) < 1 and np.isnan(pd.to_numeric(r.loc[1, "mcap"]))
    SF.tb_fill_mcap(r)                                                # stored → no second request
    assert calls == ["AGTUSDT"]


def test_fill_mcap_failure_leaves_unknown(monkeypatch):
    monkeypatch.setattr(SF, "tb_vol24", lambda pair, t, used=None: None)
    monkeypatch.setattr(SF.time, "sleep", lambda s: None)
    r = SF.tb_fill_mcap(_rows())
    assert np.isnan(pd.to_numeric(r.loc[0, "mcap"]))


def test_lines_carry_the_split_and_the_gate_is_unchanged():
    rows = _rows().assign(vol24=[300e6, np.nan], mcap=[300e6 / 1.11, np.nan])
    w = pd.DataFrame([dict(k="2026-10-20T03:00:00", pair="SMLUSDT", actual=1.2, mcap=30e6)])
    txt = "\n".join(SF.tb_lines(rows, w, {}))
    assert "🏷 ≥ $200M · blocked | 1 |" in txt and "🏷 < $50M · let through | 1 |" in txt
    assert SF.TB_N == 15 and SF.TB_DAYS == 8 and SF.TB_CONFIRM == -0.20      # the revert gate stays pooled across mcaps


def _blk(n, pct, start_day=20):
    return pd.DataFrame([dict(t=f"2026-10-{start_day + (i % 10):02d}T17:00:{i % 60:02d}", pair=f"P{i}USDT", gate="FRENZY_WILLY_TURNOVER",
                              pct=pct, how="take profit", final=True, wpct=pct, wfinal=True) for i in range(n)])


def _lt(n, pct):
    return pd.DataFrame([dict(t=f"2026-10-{20 + (i % 10):02d}T18:00:00", pct=pct, wpct=pct, wfinal=True) for i in range(n)])


def test_v2_gate_counts_the_v1_rows_and_waits():
    assert SF.TB2_FROM <= "2026-10-08T17:30:00"                        # operator: the already-rejected entries count (first refusal 10-08 17:30)
    old = _blk(15, 1.0, start_day=10)                                  # 15 rows on 10 days, like the v1 sample
    assert SF.tb2_step({}, old, _lt(20, 0.0)) == {}                    # counted, but < 30 → waits
    both = pd.concat([old, _blk(15, 1.0)], ignore_index=True)          # + 15 newer → 30 → judged
    assert SF.tb2_step({}, both, _lt(20, 0.0))["verdict"].startswith("REVERT")


def test_v2_gate_verdicts():
    assert SF.tb2_step({}, _blk(30, 0.5), _lt(40, 0.1))["verdict"].startswith("REVERT")      # blocked ≥ let-through
    assert SF.tb2_step({}, _blk(30, -0.3), _lt(40, 0.1))["verdict"] == "filter confirmed"    # ≤ let-through − 0.20
    s = SF.tb2_step({}, _blk(30, 0.0), _lt(40, 0.1))                                          # in between → hold, re-read at 60
    assert s.get("held_at_30") and not s.get("verdict")
    assert SF.tb2_step(s, _blk(45, 0.0), _lt(40, 0.1)) == s                                   # still < 60
    assert SF.tb2_step(s, _blk(60, 0.0), _lt(40, 0.1))["verdict"] == "holds (operator decides)"
    frozen = SF.tb2_step({}, _blk(30, 0.5), _lt(40, 0.1))
    assert SF.tb2_step(frozen, _blk(60, -5.0), _lt(40, 0.1)) == frozen                       # frozen at the first verdict


def test_tb_vol24_window_column_and_scaling(monkeypatch):
    """deep review C1: exactly the 288 5m bars CLOSED before the refusal's bar (no after-refusal bar), quote volume = kline index 7."""
    seen = {}

    class _R:
        def __init__(self, rows): self.rows, self.headers = rows, {"X-MBX-USED-WEIGHT-1M": "10"}
        def read(self): return __import__("json").dumps(self.rows).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(url, timeout=10):
        seen["url"] = url
        return _R([[0, "1", "1", "1", "1", "0", 0, "100.0"]] * 260)   # 260 rows × 100 quote → scaled to 288
    monkeypatch.setattr(SF.urllib.request, "urlopen", fake)
    monkeypatch.setattr(SF, "TICK_CACHE", "/nonexistent-cooldown-dir")
    t = 1_792_000_006_000                                              # 6 s after a 5m boundary (the refusal timing)
    end = (t // 300_000) * 300_000
    v = SF.tb_vol24("XUSDT", t)
    assert f"startTime={end - 288 * 300_000}" in seen["url"] and f"endTime={end - 1}" in seen["url"] and "interval=5m" in seen["url"]
    assert abs(v - 100.0 * 288) < 1e-6


def test_v2_gate_frozen_first_sample():
    """deep review G1: the read takes the FIRST 30 by time; a later provisional row never holds it; let-through cut at the sample's end."""
    b = pd.concat([_blk(30, 0.5), pd.DataFrame([dict(t="2026-10-31T23:00:00", pair="LATEUSDT", gate="FRENZY_WILLY_TURNOVER", pct=np.nan,
                                                       how=np.nan, final=False, wpct=np.nan, wfinal=False)])], ignore_index=True)
    lt = pd.concat([_lt(40, 0.1), pd.DataFrame([dict(t="2026-11-05T00:00:00", pct=99.0, wpct=99.0, wfinal=True)])], ignore_index=True)   # after the sample → excluded
    s = SF.tb2_step({}, b, lt)
    assert s["verdict"].startswith("REVERT") and s["n"] == 30 and abs(s["let_through"] - 0.1) < 1e-9
