"""📈 NAV and outside money (DECISION_LOG 422). NAV/share = equity ÷ shares, so money that reaches the account without shares reads as profit: the 10-26 paper top-up
of $7,100 showed as +399 %. Pure math: a flow is priced at the NAV just before it and becomes shares; a past history is rebuilt from the same rule."""

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
    """outside money (Binance transfers) vs what the Deposit / Withdraw buttons booked → the gap $ or None when they agree within max($5, 0.5 % of equity). A flag: a fund founded on money already there or a cash-out with no transfer also shows a gap."""
    gap = float(external_net) - float(booked_net)
    return round(gap, 2) if abs(gap) > max(UNBOOKED_TOL_USD, UNBOOKED_TOL_PCT * abs(float(reference_equity or 0))) else None
