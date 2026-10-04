"""⚖️ Oct-4 Sleeve Sizing table (Liquidity & Risk Caps) — single source of truth for every sleeve's switch + Inv / Lev
multipliers. Pins: each moved input exists ONCE and lives inside the table (old locations keep only a note), the new
Bull-long / Bounce-long inputs are loaded AND saved (D11), the preview's engine mirrors stay in sync, and the balance
endpoint carries the sizing base the preview needs."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()

MOVED = [
    "config-long-unmatched-quiet-mult", "config-long-unmatched-quiet-lev-mult", "config-long-unmatched-demux-inv-mult",
    "config-long-unmatched-demux-lev-mult", "config-nonexp-calm3d-enabled", "config-nonexp-calm3d-invest-mult",
    "config-nonexp-calm3d-lev-mult", "config-long-cross-ob-narrow-enabled", "config-long-cross-ob-invest-mult",
    "config-long-cross-ob-lev-mult", "config-long-btc-adx-surge-enabled", "config-long-btc-adx-surge-invest-mult",
    "config-long-btc-adx-surge-lev-mult", "config-c1-demux-breadth-enabled", "config-flip-entry-enabled",
    "config-flip-long-enabled", "config-flip-negdi-mult", "config-flip-negdi-lev", "config-flip-tg-shallow-mult",
    "config-flip-tg-shallow-lev", "config-spike-fade-enabled", "config-spike-fade-invest-mult", "config-spike-fade-lev-mult",
    "config-spike-chase-enabled", "config-spike-invest-mult", "config-spike-lev-mult", "config-spike-bounce-enabled",
    "config-spike-bounce-invest-mult", "config-spike-bounce-lev-mult", "config-br-enabled", "config-br-invest-mult",
    "config-br-lev-mult", "config-be-enabled", "config-be-invest-mult", "config-be-lev-mult", "config-sg-long-enabled",
    "config-sg-short-enabled", "config-sg-long-invest-mult", "config-sg-long-lev-mult", "config-sg-short-invest-mult",
    "config-sg-short-lev-mult", "config-fz-long-enabled", "config-fz-invest-mult", "config-fz-lev-mult",
    "config-fz-lev-mult-strong", "config-fz-wide-enabled", "config-fz-wide-invest-mult", "config-fz-wide-lev-mult",
    "config-manual-max-open-positions",
]
NEW = {
    "config-bull-long-enabled": "bull_long_enabled", "config-bull-long-size-mult": "bull_long_size_mult",
    "config-bull-long-lev-mult": "bull_long_lev_mult", "config-bounce-long-enabled": "bounce_long_enabled",
    "config-bounce-long-size-mult": "bounce_long_size_mult", "config-bounce-long-lev-mult": "bounce_long_lev_mult",
}


def _table():
    a = HTML.index("<!-- ===== ⚖️ SLEEVE SIZING")
    b = HTML.index("<!-- BNB Fee Management -->")
    assert a < b and HTML.index("💧 Liquidity &amp; Risk Caps") < a          # inside Liquidity & Risk Caps, above BNB
    return HTML[a:b]


def test_every_sizing_input_lives_once_in_the_table():
    t = _table()
    for _id in MOVED + list(NEW):
        assert HTML.count(f'id="{_id}"') == 1, _id
        assert f'id="{_id}"' in t, f"{_id} not in the Sleeve Sizing table"
    assert HTML.count("→ set in ⚖️ Sleeve Sizing") >= 20                      # old locations keep a pointer note
    for key in ("mom_long", "unm", "quiet", "demux", "calm3d", "crossob", "adxs", "mom_short", "c1", "c1demux", "flip",
                "negdi", "tg", "fade", "chase", "bounce", "bullrun", "bearrun", "sg_long", "sg_short", "bull_long",
                "bounce_long", "fz", "fz_strong", "fzw", "manual"):
        assert t.count(f'data-ss="{key}"') == 1, key
        assert key == "manual" or re.search(rf"(\bS\.{key}\s*=|\['{key}', )", HTML), f"row {key} has no sizing spec"


def test_new_observation_sleeve_inputs_are_loaded_and_saved():
    import config as C
    th = type(C.trading_config.thresholds).model_fields
    import json
    cfgj = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for _id, key in NEW.items():
        assert key in th and key in cfgj, key                                   # config default + JSON value
        assert HTML.count(f"'{_id}'") >= 1, _id                                 # load / save walk the id
    assert "bull_long_enabled: document.getElementById('config-bull-long-enabled')?.checked === true" in HTML
    assert "bounce_long_enabled: document.getElementById('config-bounce-long-enabled')?.checked === true" in HTML
    assert HTML.count("['config-bounce-long-lev-mult', 'bounce_long_lev_mult', 0.05]") == 2   # load + save share the default


def test_preview_mirrors_engine_constants():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "flips now exit via the NORMAL realtime stack" in eng                        # → the preview prices flips at the momentum stop
    assert "target: 'both', stop: momSl, stopNote: momNote, note });" in HTML
    assert "leverage = max(1, int(round(leverage * cell_lev_multiplier)))" in eng      # the JS uses Python half-even rounding
    assert "function _ssPyRound(x)" in HTML and "renderSleeveSizing();" in HTML
    assert "window._ssEquity = Number(data.sizing_equity)" in HTML


def test_balance_endpoint_carries_the_sizing_base():
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert src.count('"sizing_equity":') == 2 and src.count('"fee_reserve_burn_usd": round(_fee_reserve_burn_leg(), 2)') == 2
    assert "_fee_res = max(_fee_res, _fee_reserve_burn_leg())" in src           # one burn-leg source for the card and the preview
