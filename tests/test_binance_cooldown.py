"""⏳ DECISION_LOG 306: the shared Binance cooldown file is written in SECONDS by some study fetchers and in MILLISECONDS by the scouts —
every reader must honour both (a ban armed in one unit was silently ignored by the scouts, and permanently blocked scout_frenzy_exits'
gvol fetch in the other). Pins scripts/binance_cooldown.py, the scout readers that delegate to it, and the scout_frenzy_exits fix."""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import binance_cooldown as BC  # noqa: E402
import scout_b1h_negflank as NF  # noqa: E402

NOW_MS = 1_792_000_000_000


def test_units():
    assert BC.to_ms("1792000600") == 1_792_000_600_000            # integer seconds
    assert BC.to_ms("1792000600.75") == 1_792_000_600_750          # float seconds (study fetchers: str(time.time() + ra))
    assert BC.to_ms("1792000600000") == 1_792_000_600_000         # milliseconds (scouts)
    assert BC.to_ms("") == 0 and BC.to_ms("x") == 0 and BC.to_ms(None) == 0 and BC.to_ms("-5") == 0


def test_active_both_units(tmp_path):
    f = tmp_path / "cd"
    for raw in (str((NOW_MS + 60_000) / 1000), str(NOW_MS + 60_000)):          # a live ban in seconds and in ms
        f.write_text(raw)
        assert BC.active(str(f), now_ms=NOW_MS) is True
    for raw in (str((NOW_MS - 60_000) / 1000), str(NOW_MS - 60_000), "", "garbage"):   # expired / empty / unreadable
        f.write_text(raw)
        assert BC.active(str(f), now_ms=NOW_MS) is False
    assert BC.active(str(tmp_path / "missing"), now_ms=NOW_MS) is False


def test_negflank_reader_honours_seconds(tmp_path, monkeypatch):
    shared = tmp_path / ".binance_cooldown_until"
    monkeypatch.setattr(NF, "SHARED_COOLDOWN", str(shared))
    monkeypatch.setattr(NF, "DU_COOLDOWN", str(tmp_path / "du"))
    monkeypatch.setattr(NF, "_CACHE", str(tmp_path))
    shared.write_text(str(time.time() + 600))                      # a study wrote float SECONDS
    assert NF._cooldown_until() > int(time.time() * 1000)            # → seen as a live ban, in ms
    shared.write_text(str(int(time.time() * 1000) + 600_000))         # a scout wrote ms
    assert NF._cooldown_until() > int(time.time() * 1000)
    shared.unlink()
    assert NF._cooldown_until() == 0


def test_no_raw_unit_assuming_readers_left():
    """frenzy_exits compares via the helper (it compared time.time() SECONDS with an ms file → blocked forever once a scout armed it)."""
    src = open(os.path.join(HERE, "scripts", "scout_frenzy_exits.py")).read()
    assert "_BC.active(os.path.join(TICK_CACHE, \".binance_cooldown_until\"))" in src
    assert not re.search(r"time\.time\(\) >= cd", src)
    for name in ("scout_b1h_negflank.py", "scout_ml_trend_lines.py"):
        s = open(os.path.join(HERE, "scripts", name)).read()
        assert "binance_cooldown" in s


def test_every_script_reader_uses_the_helper():
    """no script may parse the shared cooldown file with a fixed unit (a new study fetcher would bring the bug back)."""
    import glob
    bad = []
    for f in glob.glob(os.path.join(HERE, "scripts", "*.py")):
        if f.endswith("binance_cooldown.py"):
            continue
        s = open(f).read()
        if ".binance_cooldown_until" not in s and "ratelimited_until" not in s:
            continue
        for m in re.finditer(r"(float|int)\(open\([^)]*\)\.read\(\)", s):
            bad.append(f"{os.path.basename(f)}: {m.group(0)}")
    assert not bad, bad
    assert BC.to_ms("inf") == 0 and BC.to_ms("nan") == 0
