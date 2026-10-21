"""scripts/archive_batch.py — one command to add a batch. Everything runs in tmp dirs: no network (the klines fetch is mocked), no real
build (subprocess.run is monkeypatched), never the real reports/ or the shared Binance cooldown file."""
import io, os, urllib.error
import pandas as pd
import pytest

from scripts import archive_batch as A

COLS = ["opened_at", "pair", "direction", "status", "entry_strategy", "close_reason", "pnl", "pnl_percentage", "cell_multiplier_source",
        "pattern_cell_source", "leverage", "bracket_capped", "entry_bracket_max_leverage"]


def _orders(rows):
    return pd.DataFrame([dict(zip(COLS, r)) for r in rows], columns=COLS)


BATCH = [
    ("2026-10-21T08:00:00", "AAAUSDT", "LONG", "CLOSED", "MOMENTUM", "RUNNER_TRAIL", 10.0, 0.5, "UNMATCHED", "UNMATCHED", 20, 0, 50),
    ("2026-10-21T09:00:00", "BBBUSDT", "LONG", "CLOSED", "MOMENTUM", "STOP_LOSS L1", -5.0, -0.25, "UNMATCHED", "UNMATCHED", 20, 0, 50),
    ("2026-10-21T10:00:00", "CCCUSDT", "LONG", "CLOSED", "SURGE_LONG", "BR_TRAILING_STOP", 50.0, 2.0, "SURGE_LONG", "", 10, 1, 10),
    ("2026-10-21T11:00:00", "DDDUSDT", "LONG", "CLOSED", "FRENZY_LITE", "MANUAL", 1.0, 0.1, "FRENZY_LITE", "", 8, 0, 25),
    ("2026-10-21T12:00:00", "EEEUSDT", "LONG", "CLOSED", "FRENZY_LITE", "EXTERNAL_CLOSE", -2.0, -0.2, "FRENZY_LITE", "", 8, 0, 25),
    ("2026-10-21T13:00:00", "FFFUSDT", "SHORT", "CLOSED", "MANUAL", "MANUAL_TP", 7.0, 0.7, "", "", 10, 0, 25),
    ("2026-10-22T14:00:00", "GGGUSDT", "LONG", "CLOSED", "MOMENTUM", "STOP_LOSS", -3.0, -0.3, "GAPFLAT_PROBE", "", 1, 0, 50),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    rep = tmp_path / "reports"; rep.mkdir()
    _orders([("2026-10-19T08:00:00", "OLDUSDT", "LONG", "CLOSED", "MOMENTUM", "STOP_LOSS", 1, 0.1, "UNMATCHED", "", 20, 0, 50),
             ("2026-10-20T22:00:00", "OLDUSDT", "LONG", "CLOSED", "MOMENTUM", "STOP_LOSS", 1, 0.1, "UNMATCHED", "", 20, 0, 50)]
            ).to_csv(rep / "BASELINE23_batch1019-1020_LIVE_orders.csv", index=False)
    (rep / "BASELINE23_batch1019-1020_LIVE_decisions.csv").write_text("t,e\n")
    cache = rep / "BTCUSDT.csv"
    cache.write_text("open_time,o,h,l,c,vol,qvol\n1000,1,1,1,1,1,1\n")
    builder = tmp_path / "build_master_pool.py"
    builder.write_text('X = 1\nSTACK_VERSION = "2026-10-21b"  # 10-21b (bracket cap). Prior 10-21a — 10-21a (+ B23 archived)\n')
    dl = tmp_path / "DECISION_LOG.md"; dl.write_text("## (331) x\n## (333) y\n")
    cool = tmp_path / ".binance_cooldown_until"
    import binance_cooldown as BC
    monkeypatch.setattr(BC, "SHARED", str(cool))
    monkeypatch.setattr(A, "REPORTS", str(rep)); monkeypatch.setattr(A, "BTC_CACHE", str(cache)); monkeypatch.setattr(A, "BUILDER", str(builder))
    monkeypatch.setattr(A, "MASTER", str(tmp_path / "MASTER_POOL_stacked.csv")); monkeypatch.setattr(A, "DECISION_LOG", str(dl))
    dl_dir = tmp_path / "dl"; dl_dir.mkdir()
    o = dl_dir / "scalpars_orders_live_2026-10-22_15-00-00.csv"; _orders(BATCH).to_csv(o, index=False)
    d = dl_dir / "scalpars_decisions_live_2026-10-22_15-00-00.csv"; d.write_text("t,e\n2026-10-21T08:00:00,OPEN\n")
    calls = []

    class R:
        returncode, stdout, stderr = 0, "ok\n", ""

    def fake_run(args, **kw):
        calls.append([os.path.basename(x) for x in args[1:]])
        if args[-1].endswith("build_master_pool.py"):      # the fake builder writes a master holding the new era
            m = _orders(BATCH); m["era"] = "B24"
            m["stack_keep"] = [True, False, True, True, True, False, True]
            m["stack_pct"] = [0.5, None, 2.0, 0.1, -0.2, None, -0.3]
            m["stack_pnl"] = [1.0, 0.0, 50.0, 1.0, -2.0, 0.0, -0.15]
            m.to_csv(A.MASTER, index=False)
        return R()
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    fetched = []

    def fake_fetch(url):
        fetched.append(url)
        start = int(url.split("startTime=")[1].split("&")[0])
        return [[t, "2", "3", "1", "2", "9", 0, "18"] for t in range(start, start + 3 * A.BAR_MS, A.BAR_MS)], {"x-mbx-used-weight-1m": "12"}
    monkeypatch.setattr(A, "_http_get", fake_fetch)
    monkeypatch.setattr(A, "_now_ms", lambda: 1000 + 4 * A.BAR_MS + 10)   # bars 1000+300k … +900k closed
    return dict(rep=rep, orders=str(o), decisions=str(d), cache=cache, builder=builder, calls=calls, fetched=fetched, cool=cool)


def test_plan_number_window_names(env):
    p = A.plan(env["orders"])
    assert p["n"] == 24 and p["first"] == "2026-10-21T08:00:00" and p["last"] == "2026-10-22T14:00:00" and p["mode"] == "LIVE"
    assert os.path.basename(p["orders_dest"]) == "BASELINE24_batch1021-1022_LIVE_orders.csv"
    assert os.path.basename(p["decisions_dest"]) == "BASELINE24_batch1021-1022_LIVE_decisions.csv"
    assert A.archive_names(7, "2026-10-12T05:00:00", "2026-10-12T15:00:00", "PAPER") == ("BASELINE7_batch1012_orders.csv", "BASELINE7_batch1012_decisions.csv")
    assert A.export_mode("scalpars_orders_paper_2026-10-14.csv") == "PAPER" and A.export_mode("x.csv") is None
    assert A.plan(env["orders"], n=30)["n"] == 30


def test_refuses_empty_overlap_and_taken_number(env, tmp_path):
    e = tmp_path / "empty.csv"; _orders([]).to_csv(e, index=False)
    with pytest.raises(A.ArchiveError, match="empty"):
        A.plan(str(e))
    ov = tmp_path / "scalpars_orders_live_ov.csv"
    _orders([("2026-10-20T12:00:00", "ZZZUSDT", "LONG", "CLOSED", "MOMENTUM", "X", 1, 0.1, "", "", 20, 0, 50)]).to_csv(ov, index=False)
    with pytest.raises(A.ArchiveError, match="overlaps BASELINE23"):
        A.plan(str(ov))
    with pytest.raises(A.ArchiveError, match="BASELINE23 already has"):
        A.plan(env["orders"], n=23)


def test_dry_run_touches_nothing(env, capsys):
    before = sorted(os.listdir(env["rep"])); cache0 = env["cache"].read_text()
    assert A.main(["--orders", env["orders"], "--decisions", env["decisions"], "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert sorted(os.listdir(env["rep"])) == before and env["cache"].read_text() == cache0
    assert not env["calls"] and not env["fetched"]
    assert "① B24" in out and "⑦ B24" in out and "master does not hold this era" in out and "⑧" not in out
    assert "MOM-long" in out and "MANUAL fills: 1" in out and "*_PROBE fills (excluded): 1" in out


def test_stack_bump_required_then_full_run(env, capsys):
    rc = A.main(["--orders", env["orders"], "--decisions", env["decisions"]])
    out = capsys.readouterr().out
    assert rc == 3
    assert 'STACK_VERSION = "2026-10-' in out and "(+ B24 archived: fills 10-21 08:00 → 10-22 14:00 UTC, 6 bot fills + 1 MANUAL" in out
    assert "Prior 10-21b — " in out
    # ②③④ ran, the build chain did not; the source was not edited
    assert (env["rep"] / "BASELINE24_batch1021-1022_LIVE_orders.csv").exists()
    assert (env["rep"] / "BASELINE24_batch1021-1022_LIVE_decisions.csv").exists()
    assert env["calls"] == [["-I", "build_willy_exit_cf.py"]]
    assert "B24" not in env["builder"].read_text()
    assert env["cache"].read_text().strip().splitlines()[-1].startswith(str(1000 + 3 * A.BAR_MS))   # only CLOSED bars appended
    # operator bumps the stack → the same command resumes
    env["builder"].write_text('STACK_VERSION = "2026-10-22a"  # 10-22a (+ B24 archived). Prior 10-21b — x\n')
    env["calls"].clear()
    assert A.main(["--orders", env["orders"], "--decisions", env["decisions"]]) == 0
    out = capsys.readouterr().out
    assert "RESUMING" in out and "already archived (identical)" in out
    assert env["calls"] == [["-I", "build_willy_exit_cf.py"], ["build_master_pool.py"], ["screen_pool.py"], ["validate_against_master.py"]]
    assert "TOTAL (bot)" in out and "+50.00" in out                          # kept $ from the (fake) master, bot only: 1 + 50 + 1 − 2
    assert "EXTERNAL_CLOSE rows: 1 — EEEUSDT" in out
    assert "closed by the operator (MANUAL / MANUAL_* / EXTERNAL_CLOSE — scout_gatekit): 2 — DDDUSDT" in out and "EEEUSDT 10-21 12:00 FRENZY_LITE (EXTERNAL_CLOSE)" in out
    assert "bracket-capped rows: 1 — CCCUSDT" in out
    assert "⑧ DRAFT — CLAUDE_CURRENT_STATE.md" in out and "BATCH BOUNDARY" in out and "## (334)" in out


def test_allow_same_stack(env, capsys):
    assert A.main(["--orders", env["orders"], "--decisions", env["decisions"], "--allow-same-stack"]) == 0
    assert "--allow-same-stack" in capsys.readouterr().out


def test_failed_step_stops(env, monkeypatch, capsys):
    env["builder"].write_text('STACK_VERSION = "2026-10-22a"  # (+ B24 archived)\n')

    class Bad:
        returncode, stdout, stderr = 1, "", "boom"
    seen = []
    monkeypatch.setattr(A.subprocess, "run", lambda args, **kw: (seen.append(os.path.basename(args[-1])), Bad())[1])
    assert A.main(["--orders", env["orders"], "--decisions", env["decisions"]]) == 2
    assert seen == ["build_willy_exit_cf.py"] and "WILLY exit CF failed" in capsys.readouterr().out


def test_existing_different_archive_refused(env):
    p = A.plan(env["orders"])
    open(p["orders_dest"], "w").write("different\n")
    with pytest.raises(A.ArchiveError, match="BASELINE24 already has"):
        A.plan(env["orders"], n=24)


def test_stack_mentions_word_boundary():
    assert A.stack_mentions('STACK_VERSION = "x"  # + B24 archived', 24)
    assert not A.stack_mentions('STACK_VERSION = "x"  # + B241 archived, B2 era', 24)
    assert A.suggest_stack("2026-10-22a", 24, "2026-10-21T08:00:00", "2026-10-22T14:00:00", 6, 1, "exp", "/r/B.csv", today="2026-10-22").startswith(
        'STACK_VERSION = "2026-10-22b"  # 10-22b (+ B24 archived')
    assert A.suggest_stack("2026-10-21b", 24, "2026-10-21T08:00:00", "2026-10-22T14:00:00", 6, 1, "exp", "/r/B.csv", today="2026-10-22").startswith(
        'STACK_VERSION = "2026-10-22a"')


def test_fetch_budget_cooldown_and_ban(env, monkeypatch):
    import binance_cooldown as BC
    # an active shared cooldown → no request at all
    env["cool"].write_text(str(10 ** 13))
    with pytest.raises(A.ArchiveError, match="cooldown active"):
        A.refresh_btc_cache(str(env["cache"]), now_ms=1000 + 10 * A.BAR_MS)
    assert not env["fetched"]
    env["cool"].unlink()
    # 429 → stop and arm the shared cooldown (ms, Retry-After honoured)

    def ban(url):
        raise urllib.error.HTTPError(url, 429, "too many", {"Retry-After": "120"}, io.BytesIO(b""))
    with pytest.raises(A.ArchiveError, match="429"):
        A.refresh_btc_cache(str(env["cache"]), now_ms=1_792_000_000_000, fetch=ban)
    assert BC.read_until_ms(str(env["cool"])) == 1_792_000_000_000 + 120_000
    env["cool"].unlink()
    # weight above the pause line → sleeps before the next request
    slept = []

    def heavy(url):
        start = int(url.split("startTime=")[1].split("&")[0])
        return [[start, "1", "1", "1", "1", "1", 0, "1"]], {"x-mbx-used-weight-1m": "1100"}
    n = A.refresh_btc_cache(str(env["cache"]), now_ms=1000 + 4 * A.BAR_MS + 10, fetch=heavy, sleep=slept.append)
    assert n == 3 and len(slept) == 2


def test_operator_closed_uses_the_gatekit_rule():
    d = _orders([
        ("2026-10-21T08:00:00", "A1USDT", "LONG", "CLOSED", "FRENZY_LITE", "MANUAL", 1, 0.1, "", "", 8, 0, 25),
        ("2026-10-21T08:05:00", "A2USDT", "LONG", "CLOSED", "FRENZY_LITE", "MANUAL_CLOSE", 1, 0.1, "", "", 8, 0, 25),
        ("2026-10-21T08:10:00", "A3USDT", "LONG", "CLOSED", "MOMENTUM", "EXTERNAL_CLOSE", 1, 0.1, "UNMATCHED", "", 20, 0, 50),
        ("2026-10-21T08:15:00", "A4USDT", "LONG", "CLOSED", "MOMENTUM", "OVERRIDE_TP", 1, 0.1, "UNMATCHED", "", 20, 0, 50),   # bot-executed
        ("2026-10-21T08:20:00", "A5USDT", "LONG", "CLOSED", "MANUAL", "EXTERNAL_CLOSE", 1, 0.1, "", "", 10, 0, 25),           # manual entry
        ("2026-10-21T08:25:00", "A6USDT", "LONG", "CLOSED", "MOMENTUM", "STOP_LOSS L1", 1, 0.1, "UNMATCHED", "", 20, 0, 50)])
    text, st = A.batch_report(d, "B24")
    assert st["_opclose"] == 3 and set(st["_op_rows"].pair) == {"A1USDT", "A2USDT", "A3USDT"}
    assert st["_ext"] == 2 and set(st["_ext_rows"].pair) == {"A3USDT", "A5USDT"}             # EXTERNAL_CLOSE still listed apart (any sleeve)
