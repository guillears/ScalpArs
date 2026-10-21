#!/usr/bin/env python3
"""🔓 Scout — WILLY global hold OFF revert-gate tracker (DECISION_LOG 340, operator 2026-10-21, override BELOW the evidence bar).

RULE (engine)   frenzy_willy_global_hold = false: while a FRENZY_WILLY is open, the other bot sleeves keep opening (never on the WILLY's
                pair — PAIR_HELD; never a second WILLY — its own slot cap + open_position's hard single-WILLY guard). Before DL 340 these
                fills were refused (FRENZY_WILLY_HOLD) — reports/SCOUT_WILLY_HOLD.csv: the 8 non-WILLY setups the hold blocked came to
                8 · 50 % · −0.56 %/trade · Σ −4.5 pts (small N, one window, inconclusive).
COHORT          bot fills (entry_strategy ≠ MANUAL) from the orders exports (paper + live via export_glob; dedup (opened_at, pair,
                direction), CLOSED wins), opened ≥ the DL 340 deploy (git commit + 10 min; not in git → NOT EVALUABLE, nothing counted).
                WILLY intervals = FRENZY_WILLY rows [opened_at, closed_at) of the SAME book (is_paper); a WILLY still open runs to now.
GATE (a)        every UTC day on which ≥ 1 FREED fill (a non-WILLY bot fill OPENED while a FRENZY_WILLY of the same book was open, ≥ the
                deploy) was opened — positions opened before the WILLY never count → that day's bot P&L (Σ pnl $ of bot fills CLOSED that
                day, operator-closed included — it is the account's money) as % of the account.
                🔒 a day ≤ −10 % → ALERT: switch the hold back ON and review. The exports carry no balance column → ACCOUNT (stated;
                --account overrides).
GATE (b)        the TIGHTER gate on the freed cohort: non-WILLY bot fills OPENED while a FRENZY_WILLY was open (same book), CLOSED,
                operator-closed excluded (scout_gatekit). FROZEN sample = the first 8 by opened_at (stored in STATE once reached, never
                re-fit). 🔒 avg pnl_percentage < 0 AND Σ ≤ −4 pts → ALERT: review turning the hold back ON. Alert-only — never changes config.
Usage           venv/bin/python scripts/scout_willy_hold_off.py [--account USD]      (batch review; no network, no bot API)
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402

DL = 340
ACCOUNT = 1225.0                                             # same stated account as scout_willy_avgdown (no balance column in the exports)
DAY_LOSS_PCT = -10.0                                         # gate (a)
B_N, B_AVG_MAX, B_SUM_MAX = 8, 0.0, -4.0                     # gate (b): first 8 fills · avg < 0 AND Σ ≤ −4 pts
STATE = os.path.join(os.path.dirname(_D), "reports", "SCOUT_WILLY_HOLD_OFF_STATE.json")
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper")


def _ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace(" ", "T"), errors="coerce", format="%Y-%m-%dT%H:%M:%S")


def load(paths=None):
    """every export row (paper + live), dedup (opened_at, pair, direction) — a CLOSED row wins, then the newest export."""
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS))
    a = pd.concat(fr, ignore_index=True)
    for c in COLS:
        if c not in a:
            a[c] = None
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a["o"] = _ts(a.opened_at).values
    a["c"] = _ts(a.closed_at).values
    a["paper"] = a.is_paper.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["usd"] = pd.to_numeric(a.pnl, errors="coerce")
    a["es"] = a.entry_strategy.astype(str)
    return a[a.o.notna()].drop(columns=["_c", "_i"]).sort_values("o").reset_index(drop=True)


def willy_intervals(a, now):
    w = a[a.es == "FRENZY_WILLY"]
    return [(bool(r.paper), r.o, (r.c if pd.notna(r.c) else now), r.pair) for r in w.itertuples()]


def during_willy(r, iv):
    """the WILLY interval (same book) this fill opened inside, or None."""
    for paper, o, c, p in iv:
        if paper == bool(r.paper) and o <= r.o < c:
            return p
    return None


def freed(a, since, now):
    """gate (b) cohort: non-WILLY bot fills opened ≥ since while a FRENZY_WILLY of the same book was open (all statuses)."""
    iv = willy_intervals(a, now)
    b = a[(a.es != "MANUAL") & (a.es != "FRENZY_WILLY") & (a.o >= since)].copy()
    b["willy_pair"] = [during_willy(r, iv) for r in b.itertuples()]
    return b[b.willy_pair.notna()].reset_index(drop=True)


def gate_a(a, since, now, account=ACCOUNT):
    """→ rows (day, book, n_bot_closed, usd, pct_of_account, alert) for the UTC days on which at least one FREED fill (a non-WILLY bot fill
    OPENED while a FRENZY_WILLY of the same book was open, ≥ since) was opened — what the switch actually changes; positions opened before
    the WILLY never make a day count. The day's bot P&L = Σ pnl $ of bot fills CLOSED that day (operator-closed included: account money)."""
    b = freed(a, since, now)
    bot = a[(a.es != "MANUAL") & (a.o >= since)]
    days = sorted({(x.o.strftime("%Y-%m-%d"), bool(x.paper)) for x in b.itertuples()})
    out = []
    for d, paper in days:
        cl = bot[(bot.paper == paper) & (bot.status.astype(str) == "CLOSED") & (bot.c.dt.strftime("%Y-%m-%d") == d)]
        usd = float(cl.usd.sum())
        pct = usd / account * 100.0
        out.append(dict(day=d, book=("paper" if paper else "live"), n=len(cl), n_freed=int(((b.paper == paper) & (b.o.dt.strftime("%Y-%m-%d") == d)).sum()),
                        usd=usd, pct=pct, alert=pct <= DAY_LOSS_PCT))
    return out


def gate_b(b, state_path=STATE):
    """→ (verdict text, frozen rows, kept closed rows, operator-closed rows). The first B_N closed fills freeze in state_path."""
    cl = b[b.status.astype(str) == "CLOSED"]
    kept, opx = _GK.split_operator_closed(cl) if len(cl) else (cl, cl)
    kept = kept.sort_values("o").reset_index(drop=True)
    st = {}
    try:
        with open(state_path) as f:
            st = json.load(f)
    except Exception:
        st = {}
    if st.get("frozen_keys"):
        keys = [tuple(k) for k in st["frozen_keys"]]
        idx = {(str(r.opened_at), r.pair, r.direction): i for i, r in enumerate(kept.itertuples())}
        fz = kept.iloc[[idx[k] for k in keys if k in idx]] if keys else kept.iloc[0:0]
    elif len(kept) >= B_N:
        fz = kept.head(B_N)
        st = dict(frozen_keys=[[str(r.opened_at), r.pair, r.direction] for r in fz.itertuples()], frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            with open(state_path, "w") as f:
                json.dump(st, f, indent=1)
        except Exception:
            pass
    else:
        fz = kept.iloc[0:0]
    if len(fz) < B_N:
        return f"collecting ({len(kept)}/{B_N} closed freed fills)", fz, kept, opx
    avg, tot = float(fz.pct.mean()), float(fz.pct.sum())
    if avg < B_AVG_MAX and tot <= B_SUM_MAX:
        return f"🔴 ALERT — first {B_N}: avg {avg:+.3f} % < 0 AND Σ {tot:+.2f} ≤ {B_SUM_MAX:g} pts → review turning the hold back ON", fz, kept, opx
    return f"🟢 first {B_N} frozen: avg {avg:+.3f} % · Σ {tot:+.2f} pts — bar not hit (keep OFF; gate (a) still runs)", fz, kept, opx


def _line(g):
    if not len(g):
        return "N 0"
    return f"N {len(g)} · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · Σ {g.pct.sum():+.2f} pts · ${g.usd.sum():+.2f}"


def run(now_ms=None, paths=None, since_ms=None, account=ACCOUNT, state_path=STATE):
    """→ markdown lines (opportunity_scout embeds them; never raises past its caller's try)."""
    now = pd.Timestamp(int(now_ms or time.time() * 1000), unit="ms")
    L = [f"## 🔓 WILLY_HOLD_OFF — the global hold switched OFF (DECISION_LOG {DL}, operator override below the evidence bar)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_HOLD_OFF")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = load(paths)
    b = freed(a, since, now)
    L.append(f"Cohort: bot fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (deploy + 10 min) while a FRENZY_WILLY of the same book was open.")
    L.append("")
    rows = gate_a(a, since, now, account)
    L.append(f"**Gate (a)** — UTC days on which ≥ 1 non-WILLY bot fill was OPENED while a FRENZY_WILLY was open (a freed fill): that day's bot "
             f"P&L % of the account · 🔒 ≤ {DAY_LOSS_PCT:g} % → switch the hold back ON + review · ASSUMED account ${account:,.0f} — pass --account")
    if rows:
        for r in rows:
            L.append(f"- {r['day']} {r['book']}: {r['n_freed']} freed opened · {r['n']} bot fills closed · ${r['usd']:+.2f} · {r['pct']:+.2f} % of "
                     f"ASSUMED account ${account:,.0f} — pass --account" + (" · 🔴 ALERT" if r["alert"] else ""))
        L.append(f"Gate (a) verdict: {'🔴 ALERT — a day breached −10 %' if any(r['alert'] for r in rows) else '🟢 no day breached'}")
    else:
        L.append("- no such day yet")
    L.append("")
    verdict, fz, kept, opx = gate_b(b, state_path)
    L.append(f"**Gate (b)** — freed cohort (non-WILLY bot fills opened while a WILLY was open), closed: {_line(kept)} · "
             f"{_GK.operator_closed_line(opx)}")
    for es, g in kept.groupby("es"):
        L.append(f"- {es}: {_line(g)}")
    L.append(f"Gate (b) verdict (first {B_N} frozen · avg < 0 AND Σ ≤ {B_SUM_MAX:g} pts → ALERT, alert-only): {verdict}")
    L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", type=float, default=ACCOUNT)
    args = ap.parse_args(argv)
    print("\n".join(run(account=args.account)))


if __name__ == "__main__":
    main()
