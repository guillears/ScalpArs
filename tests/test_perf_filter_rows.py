"""🗂 (DECISION_LOG 472) the Performance filter block is three rows: ① filters · ② Show switch + report / export buttons · ③ accounting."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows():
    src = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    a = src.index('id="regime-filter"')
    block = src[src.rfind('<div class="mb-', 0, a):src.index("</div>", src.index('id="auto-acct-toggle"'))]
    starts = [m.start() for m in re.finditer(r'\n\s*<div class="mb-[34] flex items-center', block)]
    return block, starts


def test_three_rows_in_order():
    block, starts = _rows()
    assert len(starts) == 2                                              # rows ② and ③ open inside the block (row ① opens it)
    r1, r2, r3 = block[:starts[0]], block[starts[0]:starts[1]], block[starts[1]:]
    for x in ('id="regime-filter"', 'id="window-filter"', 'id="flip-source-filter"', 'id="direction-filter"', 'id="alpha-filter"'):
        assert x in r1
    for x in ('id="origin-switch-filters"', 'id="copy-analytics-btn"', 'id="copy-split-btn"', 'id="download-orders-csv-btn"',
              'id="download-decisions-csv-btn"', 'id="auto-decisions-toggle"'):
        assert x in r2
    for x in ('id="acct-from"', 'id="acct-to"', 'id="download-accounting-xlsx-btn"', 'id="acct-statements-group"', 'id="auto-acct-toggle"'):
        assert x in r3
    assert 'id="origin-switch-filters"' not in r1 and 'id="acct-from"' not in r2
