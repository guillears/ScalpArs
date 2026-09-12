"""Gate-51 cell classifier (Sep-11): the three relaxed BTC-gate bands overlap; the cells must be an
exact PARTITION of their union so the dashboard's attribution rows sum to the de-duplicated total.
Falsifiable: widening any cell boundary (e.g. `rsi < 55` → `<= 55`) or dropping a cell breaks it."""
import itertools
import main


def _in_b1(r, a): return 50 <= r < 55
def _in_b2(r, a): return 15 <= a < 18
def _in_b3(r, a): return 55 <= r < 60 and (15 <= a < 20 or 25 < a <= 30)


# Fine grid over the whole allowed long space (BTC RSI 40-65, ADX 15-40) incl. every boundary.
GRID = [(r / 10.0, a / 10.0) for r, a in itertools.product(range(380, 671, 1), range(140, 411, 1))]


def test_cells_partition_the_band_union():
    for r, a in GRID:
        cell = main._g51_cell(r, a)
        in_any = _in_b1(r, a) or _in_b2(r, a) or _in_b3(r, a)
        assert (cell is not None) == in_any, (r, a, cell)
        if cell is not None:
            assert cell in main._G51_CELL_ORDER


def test_each_band_is_the_union_of_its_cells():
    b1 = {"①∩②", "①only"}
    b2 = {"①∩②", "②∩③", "②only<50", "②only60+"}
    b3 = {"②∩③", "③only"}
    for r, a in GRID:
        cell = main._g51_cell(r, a)
        assert _in_b1(r, a) == (cell in b1), (r, a, cell)
        assert _in_b2(r, a) == (cell in b2), (r, a, cell)
        assert _in_b3(r, a) == (cell in b3), (r, a, cell)


def test_known_fills_land_in_the_expected_cell():
    # BCH Sep-11 (57.3 / 17.14) and DOGE Aug-24 (56.1 / 16.23) = ②∩③; ARB/HYPE Sep-11 (53.7 / 16.29) = ①∩②;
    # SHIB Aug-25 (64.4 / 16.04) = the history-flagged ② half; LIT Aug-27 (56.3 / 25.38) = ③ only.
    assert main._g51_cell(57.3, 17.1405) == "②∩③"
    assert main._g51_cell(56.1, 16.2307) == "②∩③"
    assert main._g51_cell(53.7, 16.2861) == "①∩②"
    assert main._g51_cell(64.4, 16.0362) == "②only60+"
    assert main._g51_cell(56.3, 25.3764) == "③only"
    assert main._g51_cell(52.6, 29.1147) == "①only"
    assert main._g51_cell(62.5, 21.05) is None          # kept zone, no band
    assert main._g51_cell(None, 20) is None and main._g51_cell("x", 20) is None


def test_null_handling_mirrors_band_rows():
    # Band ① needs only RSI, band ② only ADX (main.py _g51_b1/_g51_b2) — the classifier must not
    # drop a fill the band rows count, else the dedup row's residual would be non-zero.
    assert main._g51_cell(52, None) == "①only"       # in ① by RSI alone
    assert main._g51_cell(None, 16) == "②only60+"    # in ② by ADX alone (RSI unknown → not '<50')
    assert main._g51_cell(57, None) is None          # ③ needs both; ① and ② do not match
    assert main._g51_cell(None, None) is None


def test_dedup_row_counts_true_band_union():
    """The Σ row must be derived from the band lists, not from the cells (review I-1)."""
    from types import SimpleNamespace as O
    from datetime import datetime
    fills = [O(entry_btc_rsi=57.3, entry_btc_adx=17.1, pnl=-500.0, pnl_percentage=-1.99, peak_pnl=0.0, spike_armed=False, opened_at=datetime(2026, 9, 11)),   # ②∩③
             O(entry_btc_rsi=53.7, entry_btc_adx=16.3, pnl=55.0, pnl_percentage=0.19, peak_pnl=1.5, spike_armed=False, opened_at=datetime(2026, 9, 11)),      # ①∩②
             O(entry_btc_rsi=62.5, entry_btc_adx=21.0, pnl=250.0, pnl_percentage=0.9, peak_pnl=1.0, spike_armed=False, opened_at=datetime(2026, 8, 22))]      # no band
    b1 = [o for o in fills if 50 <= o.entry_btc_rsi < 55]
    b2 = [o for o in fills if 15 <= o.entry_btc_adx < 18]
    b3 = [o for o in fills if 55 <= o.entry_btc_rsi < 60 and (15 <= o.entry_btc_adx < 20 or 25 < o.entry_btc_adx <= 30)]
    naive = len(b1) + len(b2) + len(b3)                       # 1 + 2 + 1 = 4 (double counts)
    uniq = list({id(o): o for o in (b1 + b2 + b3)}.values())  # 2
    cells = [main._g51_cell(o.entry_btc_rsi, o.entry_btc_adx) for o in fills]
    assert naive == 4 and len(uniq) == 2
    assert sum(1 for c in cells if c is not None) == len(uniq)
    assert cells == ["②∩③", "①∩②", None]
