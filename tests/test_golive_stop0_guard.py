"""🛡 Go-live ⑤ (DECISION_LOG 292): a FRENZY sleeve never runs without a bot stop. frenzy_stop_pct / frenzy_willy_stop_pct ≤ 0 are refused
by the config save (server + UI); a hand-edited trading_config.json carrying ≤ 0 loads as 3.0 / 4.0 with a CRITICAL log."""
import asyncio, json, logging, os, sys
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import bot_stop_errors, sanitize_bot_stops


class _TH:
    def __init__(self, fz, wl): self.frenzy_stop_pct, self.frenzy_willy_stop_pct = fz, wl


def test_validation_pure():
    assert bot_stop_errors(_TH(3.0, 4.0)) == []
    for fz, wl, n in ((0, 4.0, 1), (3.0, 0, 1), (-3, -4, 2), (None, "x", 2), (float("nan"), float("inf"), 2), (0.1, 0.1, 0)):
        assert len(bot_stop_errors(_TH(fz, wl))) == n, (fz, wl)


def test_sanitize_in_place():
    t = _TH(0, -1)
    assert sanitize_bot_stops(t) == [("frenzy_stop_pct", 0, 3.0), ("frenzy_willy_stop_pct", -1, 4.0)]
    assert t.frenzy_stop_pct == 3.0 and t.frenzy_willy_stop_pct == 4.0
    t = _TH(2.5, 3.5)
    assert sanitize_bot_stops(t) == [] and (t.frenzy_stop_pct, t.frenzy_willy_stop_pct) == (2.5, 3.5)


def test_a_loaded_zero_runs_the_safe_default_and_logs_critical(monkeypatch, tmp_path, caplog):
    import config
    p = tmp_path / "trading_config.json"
    p.write_text(json.dumps({"thresholds": {"frenzy_willy_stop_pct": 0.0, "frenzy_stop_pct": 0, "broker_backstop_enabled": True}}), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", str(p))
    with caplog.at_level(logging.CRITICAL):
        cfg = config.load_trading_config()
    assert cfg.thresholds.frenzy_willy_stop_pct == 4.0 and cfg.thresholds.frenzy_stop_pct == 3.0
    assert sum("[STOP0_GUARD]" in r.message for r in caplog.records) == 2
    p.write_text(json.dumps({"thresholds": {"frenzy_willy_stop_pct": 3.5, "frenzy_stop_pct": 2.0}}), encoding="utf-8")
    cfg = config.load_trading_config()
    assert (cfg.thresholds.frenzy_willy_stop_pct, cfg.thresholds.frenzy_stop_pct) == (3.5, 2.0)     # valid values untouched
    # the live file carries real stops
    th = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))["thresholds"]
    assert th["frenzy_willy_stop_pct"] > 0 and th["frenzy_stop_pct"] > 0


@pytest.mark.parametrize("upd,needle", [
    ({"frenzy_willy_stop_pct": 0}, "frenzy_willy_stop_pct must be > 0"),
    ({"frenzy_stop_pct": -3}, "frenzy_stop_pct must be > 0"),
    ({"untracked_position_action": "close"}, "untracked_position_action must be one of"),
])
def test_config_save_refuses(monkeypatch, upd, needle):
    import main
    from fastapi import HTTPException
    saved = []
    monkeypatch.setattr(main, "save_trading_config", lambda c: saved.append(c) or True)   # never touch the real file
    with pytest.raises(HTTPException) as ei:
        asyncio.run(main.update_config(main.ConfigUpdate(thresholds=upd)))
    assert ei.value.status_code == 400 and needle in ei.value.detail and saved == []


def test_ui_guard():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    j = html.index("async function saveConfig() {")
    blk = html[j:j + 1600]
    assert "[['config-fz-stop', 'FRENZY stop %'], ['config-fz-willy-stop', 'WILLY stop %']]" in blk and "_sx > 0" in blk
    assert 'min="0.1" id="config-fz-willy-stop"' in html and 'min="0.1" id="config-fz-stop"' in html
