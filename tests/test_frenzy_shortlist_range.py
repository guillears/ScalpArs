"""🔥 Oct-3 FRENZY shortlist: |24 h change| OR the 24 h low→high range (MOVR dumped then pumped back: change −0.6 %, range 28 %, never read)."""
from services.binance_service import _range_24h_pct
from services.trading_engine import frenzy_shortlist


def _p(pair, chg, rng, vol=50e6):
    return {"pair": pair, "change_24h": chg, "range_24h": rng, "volume_24h": vol}


def test_range_pct():
    assert abs(_range_24h_pct(1.28, 1.0) - 28.0) < 1e-9
    assert _range_24h_pct(None, 1.0) == 0.0 and _range_24h_pct(1.0, 0) == 0.0 and _range_24h_pct("x", 1) == 0.0
    assert _range_24h_pct(0.9, 1.0) == 0.0                      # high < low = bad data, never a negative swing


def test_range_admits_flat_day_pair():
    allp = [_p("MOVRUSDT", -0.6, 27.9), _p("SANDUSDT", 17.0, 37.9), _p("BTCUSDT", 1.0, 3.0), _p("DUMPUSDT", -22.0, 25.0)]
    assert frenzy_shortlist(allp, set(), 15.0, 20e6) == ["SANDUSDT", "DUMPUSDT", "MOVRUSDT"]   # change qualifiers first, then range-only; BTC never


def test_filters_still_apply():
    allp = [_p("MOVRUSDT", -0.6, 27.9), _p("LOWUSDT", 30.0, 40.0, vol=5e6), _p("龙虾USDT", 23.0, 57.0), _p("NORNGUSDT", 16.0, None)]
    assert frenzy_shortlist(allp, {"MOVRUSDT"}, 15.0, 20e6) == ["NORNGUSDT"]   # skip-list, volume floor, non-ASCII; missing range falls back to change


def test_cap():
    allp = [_p(f"P{i}USDT", 0.0, 20.0 + i) for i in range(30)]
    out = frenzy_shortlist(allp, set(), 15.0, 20e6)
    assert len(out) == 25 and out[0] == "P29USDT"


def test_change_qualifiers_never_dropped_by_cap():
    allp = [_p(f"R{i}USDT", 1.0, 40.0 + i) for i in range(30)] + [_p("CHGUSDT", 16.0, 19.0)]
    out = frenzy_shortlist(allp, set(), 15.0, 20e6)
    assert out[0] == "CHGUSDT" and len(out) == 25          # the old rule's pair is read even when 30 choppier pairs out-range it
