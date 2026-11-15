"""📈 NAV and outside money (DECISION_LOG 422). NAV/share = equity ÷ shares, so money that reaches the account without shares reads as profit: the 10-26 paper top-up
of $7,100 showed as +399 %. Pure math: a flow is priced at the NAV just before it and becomes shares; a past history is rebuilt from the same rule."""

from datetime import datetime

REBUILD_MARK = "DL 422 NAV rebuild"
UNBOOKED_TOL_USD, UNBOOKED_TOL_PCT = 5.0, 0.005


def flow_shares(flow, nav):
    """shares a flow is worth at the NAV just before it (a negative flow burns)."""
    if nav <= 0:
        raise ValueError("NAV must be positive to price a flow")
    return float(flow) / float(nav)


def prorata_mint(holdings, flow, nav):
    """holdings {id: shares} → {id: (shares_delta, usd_delta)}: every holder keeps their ownership %, NAV does not move."""
    total = sum(holdings.values())
    if total <= 0:
        return {}
    new = flow_shares(flow, nav)
    if new < 0 and -new > total + 1e-9:
        raise ValueError("flow is larger than the fund")
    return {k: (new * v / total, float(flow) * v / total) for k, v in holdings.items()}


def rebuild_plan(start_capital, flows, closed):
    """start_capital at NAV 1.00, flows [(t, amount)], closed [(t, pnl)] (closed fills of the batch) → (share steps [(t, shares after, nav before, equity before)], final shares)."""
    if start_capital <= 0:
        raise ValueError("start capital must be positive")
    shares, flow_sum, steps = float(start_capital), 0.0, []
    for t, amt in sorted(flows, key=lambda x: x[0]):
        eq = float(start_capital) + flow_sum + sum(p for ct, p in closed if ct < t)
        if eq <= 0:
            raise ValueError("equity not positive before a flow")
        nav = eq / shares
        shares += flow_shares(amt, nav)
        flow_sum += float(amt)
        steps.append((t, shares, nav, eq))
    return steps, shares


def shares_at(t, start_capital, steps):
    """shares outstanding at time t on the rebuilt path."""
    s = float(start_capital)
    for ft, after, _, _ in steps:
        if t >= ft:
            s = after
    return s


def unbooked_flow(external_net, booked_net, reference_equity):
    """outside money (Binance transfers) vs what the Deposit / Withdraw buttons booked → the gap $ or None when they agree within max($5, 0.5 % of equity). A flag: a cash-out with no transfer also shows a gap; a fund founded on money already there is booked OPENING (DL 550) and does not."""
    gap = float(external_net) - float(booked_net)
    return round(gap, 2) if abs(gap) > max(UNBOOKED_TOL_USD, UNBOOKED_TOL_PCT * abs(float(reference_equity or 0))) else None


OPENING_TOL_USD, OPENING_TOL_PCT = 1.0, 0.005   # same tolerance as main.nav_start_warning: "the deposit IS the account"
OPENING_RELABEL_BEFORE = datetime(2026, 11, 15, 3, 0)   # (550) rows booked before the add-time check existed


def opening_on_account(equity_before, deposit):
    """(DECISION_LOG 550) the first deposit into a share-less fund whose money is ALREADY on the account (equity ≈ deposit) is opening
    capital, not a transfer still to arrive — booked as OPENING so the outside-money check never waits for a transfer that came before the NAV start."""
    e, d = float(equity_before or 0.0), float(deposit or 0.0)
    return d > 0 and abs(e - d) <= max(OPENING_TOL_USD, OPENING_TOL_PCT * d)


def gap_is_this_deposit(gap, amount):
    """(550) the whole unbooked gap is exactly `amount` booked with no transfer since the NAV start (money that was on the account before it)."""
    amt = float(amount or 0.0)
    return gap is not None and amt > 0 and abs(float(gap) + amt) <= max(UNBOOKED_TOL_USD, OPENING_TOL_PCT * amt)


def legacy_opening_row(first_row, gap):
    """(550) one-time relabel for a fund opened before the add-time check: first_row = (type, amount, created_at) of the epoch's first ledger row.
    True when it is a DEPOSIT booked before OPENING_RELABEL_BEFORE and the whole unbooked gap is exactly that deposit missing from the transfers."""
    if first_row is None or gap is None:
        return False
    t, amt, at = first_row
    return t == "DEPOSIT" and at is not None and at < OPENING_RELABEL_BEFORE and gap_is_this_deposit(gap, amt)
