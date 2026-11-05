"""🧾 (DECISION_LOG 476) synthetic investor for the statement tests + preview: joins Oct 23 with $3,000 at NAV 1.00, adds $7,100 on
Oct 26; NAV path shaped like the real October paper run, then a recovering November. Not real data."""
from datetime import date, datetime, timedelta

OCT = [1.000, 1.037, 0.874, 0.689, 0.693, 0.723, 0.736, 0.634, 0.761, 0.600]          # Oct 22 (pre-join) … Oct 31 closes
NOV = [0.600, 0.596, 0.603, 0.611, 0.605, 0.598, 0.555, 0.561, 0.575, 0.582, 0.590, 0.586, 0.594, 0.601, 0.598, 0.603, 0.611,
       0.648, 0.640, 0.632, 0.627, 0.633, 0.629, 0.622, 0.628, 0.631, 0.634, 0.629, 0.632, 0.634, 0.6338]  # Oct 31 close, Nov 1 … Nov 30


def snaps():
    out = {}
    d = date(2026, 10, 22)
    for v in OCT:
        out[d] = v
        d += timedelta(days=1)
    d = date(2026, 11, 1)
    for v in NOV[1:]:
        out[d] = v
        d += timedelta(days=1)
    return out


def ledger(extra=()):
    n26 = snaps()[date(2026, 10, 25)]
    rows = [dict(type="DEPOSIT", amount=3000.0, nav_at_time=1.0, shares_delta=3000.0, created_at=datetime(2026, 10, 23, 15, 0)),
            dict(type="DEPOSIT", amount=7100.0, nav_at_time=n26, shares_delta=7100.0 / n26, created_at=datetime(2026, 10, 26, 14, 0))]
    return rows + list(extra)


def orders():
    out = []
    t = datetime(2026, 10, 23, 12, 0)
    i = 0
    while t < datetime(2026, 12, 1, 3, 0):
        win = i % 3 != 0
        out.append(dict(closed_at=t, pnl=(4.0 if win else -6.0), pnl_percentage=(0.6 if win else -0.9), is_paper=True))
        t += timedelta(hours=1, minutes=47)
        i += 1
    return out
