"""🎲² DECISION_LOG 356 (WILLY exit V2): exit-era guards in scripts/scout_willy_timecap.py and scripts/scout_frenzy_willy_exit.py.
Synthetic, hermetic (temp caches / state, network blocked, git floors stubbed): the era line prints; a V2 runner fill (stored target
+3.5, WILLY_FLOOR or FRENZY_TP at +3.5) is kept OUT of every old-exit parity / verdict cohort and listed with a count; a V2 fill with
target +1.25 stays in; the pinned exits stay pinned and the V2 config fields are never read."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
import willy_exit_era as WEra  # noqa: E402

E = 100.0
TH = {"frenzy_willy_tp_pct": 1.25, "frenzy_willy_stop_pct": 4.0, "frenzy_willy_max_hold_minutes": 90,
      "frenzy_willy_high_target_enabled": True, "frenzy_willy_high_k_pct": 50, "frenzy_willy_high_tp_max_pct": 8.0}


def _px(net):
    return E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setattr(WT, "_NET_BLOCKED", True)
    monkeypatch.setattr(WT, "_CACHE", str(cache))
    monkeypatch.setattr(WT, "MY_CACHE", str(cache / "scout_willy_timecap"))
    monkeypatch.setattr(WT, "EXPORT_GLOB", str(tmp_path / "none" / "scalpars_orders_paper_*.csv"))
    monkeypatch.setattr(WT, "STATE", str(tmp_path / "S.json"))
    monkeypatch.setattr(WT.NF, "_CACHE", str(cache))
    monkeypatch.setattr(WT.NF, "SHARED_COOLDOWN", str(cache / ".binance_cooldown_until"))
    monkeypatch.setattr(WT.NF, "DU_COOLDOWN", str(cache / "scout_dailyup" / ".ratelimited_until"))
    monkeypatch.setattr(WT, "_git_ct", lambda g: pd.Timestamp("2026-10-01"))          # every DL floor before the fixture day
    monkeypatch.setattr(FX, "deploy_ts", lambda: pd.Timestamp("2026-10-01"))
    WT._K1.clear()
    yield tmp_path
    WT._K1.clear()


def _ticks(pair, opened, nets):
    """a full-day tick file: prints at +1, +5 … min after the entry with the given net %."""
    te = pd.Timestamp(opened).value // 10**6
    t = te + np.array([1, 5, 10, 20][:len(nets)]) * WT.MIN
    d = os.path.join(WT._CACHE, "ticks_q", pair)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{opened[:10]}.npz"), t=t.astype(np.int64), p=np.array([_px(x) for x in nets]))


def _orders():
    base = dict(direction="LONG", status="CLOSED", notional_value=1000.0, entry_price=E, pnl=1.0, closed_at="x",
                entry_strategy="FRENZY_WILLY", entry_frenzy_willy_trigger="A", entry_frenzy_willy_spike_high=np.nan,
                willy_floor_armed_at=np.nan)
    rows = [dict(base, opened_at="2026-10-20T10:00:00", pair="PREUSDT", pnl_percentage=1.25, close_reason="FRENZY_TP",
                 frenzy_willy_tp_target_pct=np.nan),                                              # pre-V2
            dict(base, opened_at="2026-10-20T11:00:00", pair="V2OLDUSDT", pnl_percentage=1.25, close_reason="FRENZY_TP",
                 frenzy_willy_tp_target_pct=1.25),                                                # V2, target +1.25 = the old exit
            dict(base, opened_at="2026-10-20T12:00:00", pair="FLOORUSDT", pnl_percentage=1.0, close_reason="WILLY_FLOOR",
                 frenzy_willy_tp_target_pct=3.5, willy_floor_armed_at="2026-10-20T12:05:00"),     # V2 runner, floor exit
            dict(base, opened_at="2026-10-20T13:00:00", pair="RUNUSDT", pnl_percentage=3.5, close_reason="FRENZY_TP",
                 frenzy_willy_tp_target_pct=3.5)]                                                 # V2 runner, TP at +3.5
    for r in rows:
        _ticks(r["pair"], r["opened_at"], [0.5, 1.27, 1.0, 3.5])
    return pd.DataFrame(rows)


def test_reader_carries_v2_stamps():
    assert set(WEra.V2_COLS) <= set(WT.COLS) and "close_reason" in WT.COLS
    o, _ = WT.load_orders(_orders_nofiles())
    assert WEra.TP_COL in o.columns and o[WEra.TP_COL].notna().sum() == 3


def _orders_nofiles():
    return pd.DataFrame([dict(opened_at=f"2026-10-20T1{i}:00:00", pair=f"P{i}USDT", direction="LONG", status="CLOSED", entry_price=E,
                              notional_value=1.0, pnl=0.0, pnl_percentage=1.0, close_reason="FRENZY_TP", closed_at="x",
                              entry_strategy="FRENZY_WILLY", entry_frenzy_willy_trigger="A", frenzy_willy_tp_target_pct=t)
                         for i, t in enumerate([np.nan, 1.25, 3.5, 2.0])])


def test_v2_split_unit():
    o, _ = WT.load_orders(_orders_nofiles().assign(close_reason=["FRENZY_TP", "FRENZY_TP", "WILLY_FLOOR", "FRENZY_TP"]))
    keep, out = WT.v2_split(o)
    assert sorted(keep.pair) == ["P0USDT", "P1USDT"] and sorted(out.pair) == ["P2USDT", "P3USDT"]   # target 1.25 stays in
    k2, o2 = WT.v2_split(keep)
    assert len(k2) == 2 and len(o2) == 0                                                            # idempotent
    e, x = WT.v2_split(o.iloc[:0])
    assert len(e) == 0 and len(x) == 0


def test_pinned_exits_and_no_v2_config_read():
    assert (WEra.OLD_TP, WEra.OLD_STOP, WEra.OLD_HOLD, WEra.OLD_LATE) == (1.25, 4.0, 90, (30.0, 0.25))
    assert WT.OLD_EXIT == (1.0, 4.0, 120)
    assert WT.S4_CELLS == (("CAP60", 1.0, 60), ("CAP90", 1.0, 90), ("T125C60", 1.25, 60), ("T125C90", 1.25, 90))
    assert WT.levels(TH) == (1.25, 4.0, 90)                                   # V2 fields present → ignored
    assert WT.levels({**TH, "frenzy_willy_high_tp_max_pct": 99}) == WT.levels(TH)
    assert FX.GRID[-1] == ("T125_S4", 1.25, 4.0) and FX.TP125 == 1.25


def test_timecap_run_splits_v2_runners(sandbox):
    out = "\n".join(WT.run(pd.Timestamp("2026-10-22 00:00").value // 10**6, orders=_orders(), fetch=False,
                           state_path=str(sandbox / "S.json"), open_ts=[], th=TH, study=False))
    assert out.count("🎲² exit era (WILLY_TIMECAP): pre-V2 1 · V2 3 (runner target / floor 2)") == 1, out
    assert "V2 runner / floor fills kept OUT of every cohort, parity line and verdict below" in out and ": 2 — " in out
    assert "FLOORUSDT target +3.5 WILLY_FLOOR +1.00 %" in out and "RUNUSDT target +3.5 FRENZY_TP +3.50 %" in out
    par = [ln for ln in out.split("\n") if ln.startswith("Walker parity")]
    assert len(par) == 4, out                                                 # CAP15 / CAP20 · S4 (DL 281) · DL 285 · DL 290
    for ln in par:
        assert "2/2" in ln and "PREUSDT" not in ln.split("—")[0] and "FLOORUSDT" not in ln and "RUNUSDT" not in ln, ln
    assert "V2OLDUSDT live FRENZY_TP" in par[0] and "PREUSDT live FRENZY_TP" in par[0]   # both old-exit fills in, parity ✓
    live = [ln for ln in out.split("\n") if ln.startswith("Live so far")][0]
    assert "2 scored WILLY fills" in live, live
    for ln in out.split("\n"):                                                # the runners appear only on the split-out line
        if "FLOORUSDT" in ln or "RUNUSDT" in ln:
            assert ln.startswith("V2 runner / floor fills kept OUT"), ln


def test_s4_section_direct_call_splits(sandbox):
    o, _ = WT.load_orders(_orders())
    st = {}
    L, _, _ = WT.s4_section(o, [], pd.Timestamp("2026-10-22").value // 10**6, (1.25, 4.0, 90), {}, st, True, None)
    out = "\n".join(L)
    par = [ln for ln in L if ln.startswith("Walker parity")][0]
    assert "2/2" in par and "FLOORUSDT" not in par and "RUNUSDT" not in par, out
    assert "V2 runner / floor fills kept OUT" in out and ": 2 — " in out


def test_frenzy_willy_exit_prints_era_line(sandbox):
    rows = _orders()
    rows = pd.concat([rows, rows.iloc[:1].assign(pair="FZUSDT", entry_strategy="FRENZY_LONG", opened_at="2026-10-20T14:00:00",
                                                 pnl_percentage=1.5, frenzy_willy_tp_target_pct=np.nan)], ignore_index=True)
    _ticks("FZUSDT", "2026-10-20T14:00:00", [0.5, 1.6])
    th = {**TH, "frenzy_tp_pct": 1.5, "frenzy_stop_pct": 3.0, "frenzy_max_hold_minutes": 720}
    out = "\n".join(FX.run(pd.Timestamp("2026-10-22").value // 10**6, orders=rows, fetch=False, state_path=str(sandbox / "F.json"),
                           open_ts=[], th=th))
    assert out.count("🎲² exit era (FRENZY_WILLY_TP): pre-V2 1 · V2 0 (runner target / floor 0) — cohort = FRENZY_LONG / WIDE / LITE "
                     "only") == 1, out
    assert "RUNUSDT" not in out and "FLOORUSDT" not in out and "V2OLDUSDT" not in out


def test_module_selftests_still_pass(capsys):
    WT.selftest()
    FX.selftest()
    o = capsys.readouterr().out
    assert "selftest WILLY_TIMECAP OK" in o and "selftest FRENZY_WILLY_EXIT OK" in o
