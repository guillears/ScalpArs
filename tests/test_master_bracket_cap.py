"""🪜 STACK 2026-10-21b — a master re-price never exceeds the exchange leverage bracket the fill traded under.

B23 SURGE_LONG AGT 10-18 / GTC 10-19 traded 10× (bracket_capped, entry_bracket_max_leverage 10) but the master priced them at today's 20×
(stack_pnl 319.95 / 191.44 vs as-traded 159.98 / 95.72). The cap lives in ONE place (build_master_pool.bracket_cap_scale, reached through
bracket_cap_row / today_size_rule_row) and screen_pool.pnl_current uses the same rule.
"""
import csv, inspect, os
import pytest

from scripts import build_master_pool as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the two B23 rows as archived (reports/BASELINE23_batch1016-1020_LIVE_orders.csv)
AGT = dict(entry_strategy="SURGE_LONG", direction="LONG", pair="AGTUSDT", opened_at="2026-10-18T01:45:32", leverage=10.0,
           investment=212.389944, notional_value=2123.899439, cell_multiplier=1.0, cell_lev_multiplier=1.0,
           bracket_capped=1.0, entry_bracket_max_leverage=10.0, entry_bracket_cap_notional=0.0, pnl=159.976629, pnl_percentage=7.532213)
GTC = dict(AGT, pair="GTCUSDT", opened_at="2026-10-19T02:50:45", investment=230.481058, notional_value=2304.810584,
           pnl=95.722098, pnl_percentage=4.153144)


def _sleeve_price(r, inv_mult=1.0, lev_mult=1.0):
    """the builder's sleeve re-price (SURGE_LONG today 1 × 1.0 = 20×) with the cap — mirrors main()'s sleeve branch."""
    lev = max(1, int(round(20 * lev_mult)))
    cm = float(r["cell_multiplier"])
    ratio = float(r["investment"]) / cm * inv_mult * lev / float(r["notional_value"])
    f, capped, lev_eff = B.bracket_cap_row(r, lev, ratio)
    return round(r["pnl_percentage"] / 100 * r["investment"] / cm * inv_mult * lev * f, 2), capped, lev_eff


@pytest.mark.parametrize("row,traded", [(AGT, 159.98), (GTC, 95.72)])
def test_agt_gtc_priced_at_the_traded_10x(row, traded):
    pnl, capped, lev_eff = _sleeve_price(row)
    assert capped and lev_eff == 10
    assert pnl == pytest.approx(traded, abs=0.01)          # was 319.95 / 191.44 at 20×


def test_uncapped_row_unchanged():
    # stamped, not capped, re-price stays under the pair max and the tier notional → factor 1
    r = dict(AGT, bracket_capped=0.0, entry_bracket_max_leverage=50.0, entry_bracket_cap_notional=25000.0, leverage=20.0, notional_value=4000.0)
    assert B.bracket_cap_row(r, 20, 1.0) == (1.0, False, 20)
    assert B.bracket_cap_row(r, 10, 0.5) == (1.0, False, 10)
    # pre-Oct-1 fill: no stamp, bracket_capped unknown → never capped, even at a bigger size
    old = dict(leverage=20.0, notional_value=10000.0)
    assert B.bracket_cap_row(old, 20, 3.0) == (1.0, False, 20)
    assert B.bracket_cap_scale(20, 3.0, 20, 10000, None, float("nan"), float("nan")) == (1.0, False, 20)


def test_leverage_and_notional_legs():
    # leverage above the pair max: margin kept, leverage dropped
    assert B.bracket_cap_scale(20, 2.0, 10, 1000, False, 10, 0) == (0.5, True, 10)
    # capped row: allowed ≤ the traded leverage even when the pair max is higher
    f, c, le = B.bracket_cap_scale(20, 2.0, 10, 1000, True, 50, 0)
    assert c and le == 10 and f == pytest.approx(0.5)
    # notional leg: a bigger ticket (fade CAP05) clipped at the tier notional, same leverage
    f, c, le = B.bracket_cap_scale(20, 1.5, 20, 10000, False, 25, 10000)
    assert c and le == 20 and f == pytest.approx(1 / 1.5)
    # capped row cut on NOTIONAL at an unchanged leverage (as traded ≤ stamp): the stamp is the cap at the priced leverage → clip
    f, c, _ = B.bracket_cap_scale(20, 1.5, 20, 4731.9, True, 20, 4750)
    assert c and f == pytest.approx(4750 / (4731.9 * 1.5))
    # capped row traded ABOVE its stamp (TURTLE B18: 10× on a lower tier, stamp = the cap at the requested 20×) → no notional clip
    assert B.bracket_cap_scale(10, 2.0, 10, 5699.6, True, 20, 5000) == (1.0, False, 10)
    # uncapped row priced at a LOWER leverage than it traded: the stamp is the traded tier's cap, a lower leverage may allow more → no clip
    assert B.bracket_cap_scale(10, 1.5, 20, 10000, False, 25, 10000) == (1.0, False, 10)
    # never below what the exchange accepted (stamp below the traded size is ignored)
    assert B.bracket_cap_scale(20, 1.0, 20, 10000, False, 10, 5000) == (1.0, False, 20)
    # a de-levering re-price never trips the cap
    assert B.bracket_cap_scale(1, 0.05, 10, 2124, True, 10, 0) == (1.0, False, 1)


def test_cell_rule_row_tags_bracket_cap():
    # a cell re-price that lowers the size never binds; the tag only appears when the cap does
    r = dict(entry_strategy="FLIP:FAN_RATIO_GATE", direction="SHORT", cell_multiplier_source="FLIP:FAN_RATIO_GATE", cell_multiplier=1.0,
             cell_lev_multiplier=1.0, leverage=10.0, notional_value=7765.4, bracket_capped=1.0, entry_bracket_max_leverage=10.0,
             entry_bracket_cap_notional=0.0)
    f, tag = B.today_size_rule_row(r)
    assert tag == "FLIP_FAN_LEV" and f == pytest.approx(1.0)    # the BTW 10-06 10×-bracket row — unchanged


def test_screen_pool_uses_the_same_rule():
    from scripts import screen_pool
    assert screen_pool.today_size_rule_row is B.today_size_rule_row
    assert "today_size_rule_row(r)" in inspect.getsource(screen_pool.pnl_current)
    # builder: cell site, fade CAP05 site and sleeve site all route through the shared cap
    src = inspect.getsource(B.main)
    assert "today_size_rule_row(r)" in src and src.count("bracket_cap_row(") >= 2


def test_stack_version_and_master_rows():
    assert B.STACK_VERSION == "2026-10-21b" or B.STACK_VERSION > "2026-10-21b"
    p = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
    if not os.path.exists(p):
        pytest.skip("master not built")
    rows = {(r["opened_at"][:19], r["pair"]): r for r in csv.DictReader(open(p, encoding="utf-8"))}
    for row, traded in ((AGT, "159.98"), (GTC, "95.72")):
        m = rows.get((row["opened_at"], row["pair"]))
        if m is None:
            pytest.skip("B23 not in the master")
        assert float(m["stack_pnl"]) == pytest.approx(float(traded), abs=0.011)
        assert "BRACKET_CAP" in m["size_note"]


def test_capped_loser_with_unknown_cap_notional_is_not_shrunk():
    """review fix: a stamp of 0 / NaN = the cut was on LEVERAGE (requested above the pair max) — the lower tier's notional is unknown, so a
    bigger re-priced ticket is NOT clipped (KOMA B18 fade: 10× on a 10×-max pair, CAP05 ticket 2× → stays −$266.13, never flattered)."""
    koma = dict(leverage=10.0, notional_value=8871.161189, bracket_capped=1.0, entry_bracket_max_leverage=10.0, entry_bracket_cap_notional=0.0)
    assert B.bracket_cap_row(koma, 10.0, 2.0) == (1.0, False, 10.0)
    assert B.bracket_cap_row(dict(koma, entry_bracket_cap_notional=float("nan")), 10.0, 2.0) == (1.0, False, 10.0)
    # the leverage leg still binds on the same stamps (AGT shape)
    f, c, le = B.bracket_cap_row(koma, 20.0, 2.0)
    assert c and le == 10.0 and f == pytest.approx(0.5)
    p = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
    if os.path.exists(p):
        m = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["pair"] == "KOMAUSDT" and r["opened_at"].startswith("2026-10-07T11:06:01")]
        if m:
            assert float(m[0]["stack_pnl"]) == pytest.approx(-266.13, abs=0.011) and "BRACKET_CAP" not in m[0]["size_note"]
