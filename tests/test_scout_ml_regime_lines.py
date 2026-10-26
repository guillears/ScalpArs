"""🌐 tests for scripts/scout_ml_regime_lines.py (DECISION_LOG 382 — frozen momentum-long regime watch lines OBS-R1 / R1b / R2;
observe-only). Synthetic exports / regime, no network, state in tmp."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_regime_lines as S  # noqa: E402

MIN, DAY = 60_000, 86_400_000
T0 = int(pd.Timestamp("2026-11-02 00:00:00").value // 1_000_000)


def _iso(ms):
    return pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")


def _row(te, pair="AAAUSDT", pnl=0.5, peak=0.6, status="CLOSED", strat="MOMENTUM", src="UNMATCHED", reason="TRAIL", **kw):
    r = dict(opened_at=_iso(te), closed_at=_iso(te + 30 * MIN), pair=pair, direction="LONG", status=status, entry_strategy=strat,
             cell_multiplier_source=src, pnl_percentage=pnl, peak_pnl=peak, close_reason=reason,
             _m=1.0, _src="scalpars_orders_paper_2026-11-20_10-00-00.csv")
    r.update(kw)
    return r


def _regime(corr=0.40, beta=1.5, rv=0.10):
    def f(T):
        T = np.asarray(T, dtype=np.int64)
        return pd.DataFrame({"t": T, "corr24": corr, "beta24": beta, "rv7": rv, "n_corr": 20})
    return f


def _cohort(n, y=-0.5, days=10, t0=T0 + DAY):
    return [_row(t0 + (i % days) * DAY + i * MIN, pair=f"P{i:02d}USDT", pnl=y, peak=0.05) for i in range(n)]


def _run(rows, tmp_path, regime=None, master=None):
    return "\n".join(S.run(now_ms=T0 + 60 * DAY, raw=pd.DataFrame(rows), master=master if master is not None else pd.DataFrame(columns=S.COLS),
                           fetch=False, state_path=str(tmp_path / "g.json"), csv_path=None, regime_fn=regime or _regime()))


@pytest.fixture(autouse=True)
def _reports_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "REPORTS", str(tmp_path))


def test_pinned_constants():
    assert (S.DL, S.BE_WR, S.P_MIN) == (382, 61.5, 0.95)
    assert (S.R1_THR, S.R1B_THR, S.R2_THR, S.C1_THR) == (0.3164, 1.372, 0.1122, 0.365)
    assert (S.OBS_N1, S.OBS_N2, S.OBS_DAYS, S.EPISODES_MIN, S.EPISODE_GAP_DAYS) == (15, 30, 8, 3, 5)
    assert S.OBS_FLOOR == "2026-10-27T00:00:00"
    assert S.MAX_REQ <= 200 and S.WEIGHT_WAIT <= 400 and S.KLINES_LIMIT <= 1000 and 5 * 60 / S.PACE_S <= 400
    for p in (S.STATE, S.CSV_OUT, S.WORK, S.TOP50_OWN, S.FETCH5):
        assert p.startswith(os.path.join(ROOT, "reports"))


def test_keep_observing_single_episode_then_frozen(tmp_path):
    out = _run(_cohort(16), tmp_path)
    assert "OBS-R1 @ N=15: KEEP OBSERVING" in out and "1 episode(s)" in out        # corr never leaves the zone → 1 episode
    assert "OBS-R1b @ N=15: KEEP OBSERVING" in out and "OBS-R2 @ N=15: KEEP OBSERVING" in out and "OBS-R1 @ N=30: WAIT" in out
    st = json.load(open(tmp_path / "g.json"))
    assert st["OBS-R1@15"]["verdict"] == "KEEP OBSERVING" and "OBS-R1@30" not in st
    assert "OBS-R1 @ N=15: KEEP OBSERVING" in _run(_cohort(16, y=0.5), tmp_path)     # later data never flips a frozen read


def test_retire_when_zone_avg_non_negative(tmp_path):
    out = _run(_cohort(16, y=0.5), tmp_path)
    assert "OBS-R1 @ N=15: RETIRE" in out and "OBS-R2 @ N=15: RETIRE" in out


def test_outside_zone_is_not_counted(tmp_path):
    out = _run(_cohort(16), tmp_path, regime=_regime(corr=0.20, beta=1.0, rv=0.20))
    assert "OBS-R1 @ N=15: WAIT" in out and "OBS-R2 @ N=15: WAIT" in out and "OBS-R1b @ N=15: WAIT" in out


def test_switch_candidate_needs_three_episodes(tmp_path):
    # 18 losing fills on 3 separated zone stretches; noon corr low for 7 days between them
    rows = []
    for k, start in enumerate((0, 14, 28)):
        for i in range(6):
            rows.append(_row(T0 + (start + i) * DAY + (k * 10 + i) * MIN, pair=f"P{k}{i}USDT", pnl=-0.5 - 0.01 * i, peak=0.05))

    def reg(T):
        T = np.asarray(T, dtype=np.int64)
        d = (T - T0) // DAY
        corr = np.where(((d >= 6) & (d < 14)) | ((d >= 20) & (d < 28)), 0.20, 0.40)
        return pd.DataFrame({"t": T, "corr24": corr, "beta24": 1.5, "rv7": 0.10, "n_corr": 20})
    out = _run(rows, tmp_path, regime=reg)
    assert "OBS-R1 @ N=15: SWITCH-CANDIDATE" in out and "3 episode(s)" in out and "OPERATOR CALL" in out
    assert "OBS-R2 @ N=15: KEEP OBSERVING" in out                                     # rv7 never leaves its zone → 1 episode


def test_episodes_rule():
    days = pd.date_range("2026-11-01", "2026-12-10").strftime("%Y-%m-%d")
    daily = {d: 0.45 for d in days}
    for d in pd.date_range("2026-11-05", "2026-11-11").strftime("%Y-%m-%d"):
        daily[d] = 0.20
    for d in pd.date_range("2026-11-20", "2026-11-23").strftime("%Y-%m-%d"):
        daily[d] = 0.20
    assert S.episodes(["2026-11-02", "2026-11-15", "2026-11-25"], daily, 0.3164, gt=True) == 2
    assert S.episodes([], daily, 0.3164) == 0
    assert S.episodes(["2026-11-01", "2026-11-30"], {d: 0.2 for d in days}, 0.1122, gt=False) == 2


def test_b25_seen_not_counted_and_filters(tmp_path):
    master = pd.DataFrame([dict(opened_at="2026-10-20T00:00:00", pair="OLD", direction="LONG", status="CLOSED", entry_strategy="MOMENTUM",
                                cell_multiplier_source="UNMATCHED", stack_keep=True, stack_pct=0.3, is_probe=False, era="B24", peak_pnl=0.4)])
    rows = [_row(S._ms("2026-10-24T17:00:00"), pair="B25USDT", pnl=0.7),             # after the master, before the floor → SEEN
            _row(S._ms("2026-10-19T00:00:00"), pair="STALE"),                         # before the master's last fill → never counted
            _row(T0, pair="PRB", src="GAPFLAT_PROBE"), _row(T0 + MIN, pair="MAN", strat="MANUAL"),
            _row(T0 + 2 * MIN, pair="OPC", reason="MANUAL"), _row(T0 + 3 * MIN, pair="OK1USDT", pnl=-0.2)]
    out = _run(rows, tmp_path, master=master)
    assert "B25 SEEN (not counted): 1·100%·+0.700" in out and "cohort: 1·0%·-0.200" in out and "Operator-closed (excluded): 1" in out


def test_unreadable_state_never_overwritten_and_reports_only(tmp_path, monkeypatch):
    (tmp_path / "g.json").write_text("{not json")
    out = _run(_cohort(16), tmp_path)
    assert "state file unreadable" in out and (tmp_path / "g.json").read_text() == "{not json"
    monkeypatch.setattr(S, "REPORTS", os.path.join(ROOT, "reports"))
    with pytest.raises(ValueError):
        S.write_csv(pd.DataFrame({"_k": ["x"]}), str(tmp_path / "x.csv"))


def test_day_boot_and_loss_share():
    assert S.day_boot_neg(np.full(10, -0.2), np.arange(10)) == 1.0
    assert np.isnan(S.day_boot_neg([-0.2], [1]))
    assert S.loss_share([-1, -1, 2], ["a", "b", "a"]) == 0.5 and np.isnan(S.loss_share([1, 2], ["a", "b"]))


# ─────────────────────────── universe parity / completeness (DL 382 dual review) ───────────────────────────
@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "reports", "backtest_cache", "universe_daily_top50.csv")),
                    reason="backtest cache not present")
@pytest.mark.parametrize("day", ["2026-10-10", "2026-10-11"])
def test_own_list_equals_cache_list(day):
    """the scout's rebuild (1d volumes of the full universe, list dated L = day L − 1 volume) equals the study's cached list."""
    by = S._read_daily([S.DAILY_CACHE])
    d = S._ms(day + "T00:00:00")
    ref = pd.read_csv(S.TOP50_CACHE)
    assert [s for _, s, _ in S.rank_list(by, d)] == ref[ref.date_ms == d].sort_values("rank").pair.tolist()


def test_rank_list_uses_previous_day_volume():
    by = {T0 - DAY: {"AUSDT": 1.0, "BUSDT": 3.0, "CUSDT": 2.0}, T0: {"AUSDT": 99.0}}
    assert [s for _, s, _ in S.rank_list(by, T0, n=2)] == ["BUSDT", "CUSDT"]
    assert S.rank_list(by, T0 + 5 * DAY) is None


@pytest.fixture
def uni(tmp_path, monkeypatch):
    w = tmp_path / "w"
    for k, v in dict(WORK=w, DAILY_CACHE=tmp_path / "dc", DAILY_OWN=w / "daily", SYMBOLS=w / "sym.json", CHECKED=w / "chk.json",
                     INCOMPLETE=w / "inc.json", TOP50_CACHE=tmp_path / "top50.csv", TOP50_OWN=w / "own.csv").items():
        monkeypatch.setattr(S, k, str(v))
    (tmp_path / "dc").mkdir()
    pd.DataFrame([dict(date_ms=T0 - 5 * DAY, rank=1, pair="XUSDT", quote_vol=1)]).to_csv(tmp_path / "top50.csv", index=False)
    return tmp_path


def _fake_get(fail_after=None, weight=10):
    calls = {"n": 0}

    def get(url, bud, **k):
        calls["n"] += 1
        if fail_after is not None and calls["n"] > fail_after:
            bud.stop = "test stop"
            return None
        if "exchangeInfo" in url:
            return {"symbols": [{"symbol": s, "contractType": "PERPETUAL", "status": "TRADING"} for s in ("AUSDT", "BUSDT", "CUSDT")]}
        sym = url.split("symbol=")[1].split("&")[0]
        vol = {"AUSDT": 5, "BUSDT": 9, "CUSDT": 7}[sym]
        return [[T0 - i * DAY, 0, 0, 0, 0, 0, T0 - i * DAY + DAY - 1, vol] for i in range(3, 0, -1)]
    return get


def test_incomplete_daily_fetch_never_builds_a_list_then_resumes(uni, monkeypatch):
    monkeypatch.setattr(S, "_get", _fake_get(fail_after=2))
    assert S.ensure_daily(T0, S.Budget(), T0 + DAY) is False and S.universe_incomplete()
    assert T0 not in S.build_own_lists([T0])                                     # the marker blocks any own list
    assert not os.path.exists(S.TOP50_OWN)
    monkeypatch.setattr(S, "_get", _fake_get())
    assert S.ensure_daily(T0, S.Budget(), T0 + DAY) is True and not S.universe_incomplete()
    assert T0 in S.build_own_lists([T0])
    assert S._lists()[T0] == ["BUSDT", "CUSDT", "AUSDT"]                          # ranked on day T0 − 1, full universe


def test_list_not_built_when_a_perp_lacks_the_day(uni, monkeypatch):
    monkeypatch.setattr(S, "_get", _fake_get())
    S.ensure_daily(T0, S.Budget(), T0 + DAY)
    json.dump(["AUSDT", "BUSDT", "CUSDT", "DUSDT"], open(S.SYMBOLS, "w"))          # a perp never asked about
    assert T0 not in S.build_own_lists([T0])
    json.dump({"DUSDT": T0}, open(S.CHECKED, "w"))                               # asked, no bar → not listed yet → complete
    json.dump(["AUSDT", "BUSDT", "CUSDT", "DUSDT"], open(S.SYMBOLS, "w"))
    assert T0 in S.build_own_lists([T0])


def test_budget_waits_are_capped_per_run(monkeypatch):
    class R:
        headers = {"X-MBX-USED-WEIGHT-1M": "390"}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"[]"
    monkeypatch.setattr(S._CD, "active", lambda *a, **k: False)
    slept = []
    bud = S.Budget(max_req=100)
    for _ in range(10):
        S._get("http://x", bud, sleep=slept.append, opener=lambda *a, **k: R())
    waits = [s for s in slept if s != S.PACE_S]
    assert bud.waits == S.MAX_WAITS == len(waits) and bud.stop and "waits" in bud.stop


def test_hold_on_unscored_or_open_before_the_nth_zone_fill(tmp_path):
    rows = _cohort(16)

    def reg_nan_first(T):
        T = np.asarray(T, dtype=np.int64)
        corr = np.where(T == T.min(), np.nan, 0.40)
        return pd.DataFrame({"t": T, "corr24": corr, "beta24": 1.5, "rv7": 0.10, "n_corr": 20})
    out = _run(rows, tmp_path, regime=reg_nan_first)
    assert "OBS-R1 @ N=15: HOLD" in out and "1 unscored / 0 open" in out
    assert "OBS-R1@15" not in json.load(open(tmp_path / "g.json"))                # never frozen on incomplete data
    rows = _cohort(16) + [_row(T0 + DAY + 30, pair="OPENUSDT", status="OPEN", pnl=None)]
    out = _run(rows, tmp_path)
    assert "OBS-R1 @ N=15: HOLD" in out and "0 unscored / 1 open" in out
    assert "OBS-R1@15" not in json.load(open(tmp_path / "g.json"))


def test_dl383_stamp_preferred_over_rebuild(tmp_path):
    rows = _cohort(3)
    rows[0]["entry_alt_corr24"] = 0.20
    rows[0]["entry_alt_beta24"] = 1.0
    out = _run(rows, tmp_path)
    assert "DL 383 stamp vs rebuild: 1 fills with both · max |Δ corr| 0.2000" in out
    f = S.add_regime(pd.DataFrame({"t_ms": [T0, T0 + 1], "entry_alt_corr24": [0.2, np.nan], "entry_alt_beta24": [1.0, np.nan]}), _regime())
    assert list(f.corr24) == [0.2, 0.4] and list(f.r1) == [False, True] and list(f.corr24_rebuild) == [0.4, 0.4]


def test_regime_at_has_no_stale_list_fallback(monkeypatch):
    monkeypatch.setattr(S, "_lists", lambda: {T0 - 10 * DAY: ["AUSDT"] * 20})
    r = S.regime_at([T0 + 3600_000])
    assert r.corr24.isna().all() and r.beta24.isna().all()
