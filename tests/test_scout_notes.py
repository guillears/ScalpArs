"""🗒 Sep-30 — the scout writes its own notes (DECISION_LOG 156): a scheduled run is ONE fixed command. Pinned: the first run marks
the backlog as seen and writes only recent items; every item is written once; a failed run leaves one line."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import opportunity_scout as S  # noqa: E402

BAR, H = S.BAR, 3_600_000
NOW = 1_790_000_000_000 // 86_400_000 * 86_400_000 + 10 * 60_000     # 00:10 UTC: every step below stays on one UTC day


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "NOTES_MD", str(tmp_path / "notes.md"))
    monkeypatch.setattr(S, "NOTES_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(S, "REPORTS", str(tmp_path))
    monkeypatch.setattr(S.glob, "glob", lambda pat: [])            # no Decisions CSV → the stale warning fires once per day


def _ev(t, typ="BTC_MOVE", pair="BTCUSDT", side="DOWN", missed=False, **kw):
    return dict(type=typ, pair=pair, side=side, bar_ts=t, move_first=-1.6, move_max=-1.6, held_first=True, bot="none", why="",
                missed=missed, in_universe=True, **kw)


def test_first_run_backlog_then_new_items_once(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    old = _ev(NOW - 20 * H, typ="TREND", pair="ARKUSDT", side="UP", missed=True, gate_sets="PAIR_ADX_MAX×3", miss_class="FILTER_NEAR")
    recent = _ev(NOW - 1 * H)
    L1 = S.write_notes(pd.DataFrame([old, recent]), None, [], NOW)
    assert any("BTC_MOVE DOWN" in x and "SURGE trigger held" in x for x in L1)
    assert not any("ARKUSDT" in x for x in L1)                    # backlog: marked seen, not written
    assert any("Decisions CSV" in x for x in L1)
    L2 = S.write_notes(pd.DataFrame([old, recent]), None, [], NOW + 4 * H)
    assert L2 == []                                                # nothing new, warning already given today
    new_star = _ev(NOW + 3 * H, typ="TREND", pair="MOVRUSDT", side="UP", missed=True, gate_sets="PAIR_ADX_MAX×11", miss_class="FILTER_NEAR")
    L3 = S.write_notes(pd.DataFrame([old, recent, new_star]), None, [], NOW + 8 * H)
    assert len([x for x in L3 if "MOVRUSDT" in x]) == 1 and "PAIR_ADX_MAX×11" in L3[0]
    txt = open(tmp_path / "notes.md").read()
    assert txt.count("## ") == 2                                   # two runs wrote, the silent one did not


def test_failure_line(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    S.note_failure("BTC 5m fetch failed")
    assert "scout run failed: BTC 5m fetch failed" in open(tmp_path / "notes.md").read()


def _mv(start, end, move=-50.0, pair="ARKUSDT", side="DOWN"):
    fmt = lambda ms: pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d %H:%M")
    return dict(pair=pair, side=side, start_ts=start, end_ts=end, move_4h=move, bot="none", why="X×1", start_utc=fmt(start), end_utc=fmt(end))


def test_growing_mover_and_old_items_never_renoted(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    S.write_notes(pd.DataFrame([_ev(NOW - H)]), None, [], NOW)                                   # bootstrap run
    L = S.write_notes(pd.DataFrame([_ev(NOW - H)]), pd.DataFrame([_mv(NOW + H, NOW + 5 * H)]), [], NOW + 6 * H)
    assert len([x for x in L if "ARKUSDT" in x]) == 1
    grown = pd.DataFrame([_mv(NOW + 3 * H, NOW + 7 * H, move=-55.0)])                          # the episode's window slid
    assert S.write_notes(pd.DataFrame([_ev(NOW - H)]), grown, [], NOW + 8 * H) == []
    star = _ev(NOW + 2 * H, typ="TREND", pair="ZECUSDT", side="UP", missed=True, miss_class="FILTER_NEAR")
    S.write_notes(pd.DataFrame([star]), None, [], NOW + 9 * H)
    later = NOW + 9 * H + 9 * 86_400_000                                                          # 9 days on, still in the frame
    L2 = S.write_notes(pd.DataFrame([star]), grown, [], later)
    assert not any("ZECUSDT" in x or "ARKUSDT" in x for x in L2)                                 # too old to be 'new' again


def test_cap_keeps_the_rest_for_next_run(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    S.write_notes(pd.DataFrame([_ev(NOW - 5 * H)]), None, [], NOW)
    stars = [_ev(NOW + i * BAR, typ="TREND", pair=f"P{i}USDT", side="UP", missed=True) for i in range(1, 16)]
    L1 = S.write_notes(pd.DataFrame(stars), None, [], NOW + 2 * H)
    assert len([x for x in L1 if x.startswith("⭐")]) == S.NOTE_MAX_LINES and L1[-1].startswith("…")
    L2 = S.write_notes(pd.DataFrame(stars), None, [], NOW + 3 * H)
    assert len([x for x in L2 if x.startswith("⭐")]) == 15 - S.NOTE_MAX_LINES                  # the capped ones, not lost


def test_failure_line_once_per_6h(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    S.note_failure("net down"); S.note_failure("net down")
    assert open(tmp_path / "notes.md").read().count("net down") == 1
