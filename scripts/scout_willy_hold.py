"""🔒 Scout — the cost of the FRENZY_WILLY GLOBAL HOLD (2026-10-08, DECISION_LOG 251; operator-directed declared exception, NO automatic off).

While a FRENZY_WILLY position is open no other automated trade opens. Every refusal is persisted (table willy_hold_blocks, one row per
(open WILLY, pair, direction, sleeve) with a repeat count) and rides the decisions export as e=WILLY_HOLD rows (wh_* columns). This tracker:
  · merges those rows from ~/Downloads/scalpars_decisions_paper_*.csv into reports/SCOUT_WILLY_HOLD.csv (a row survives its export leaving),
  · keeps the rows from the deploy floor = the "(DECISION_LOG 251)" commit time (git; no FRENZY_WILLY fill or hold row can predate it —
    the dashboard reads every WILLY fill, the same set), else NOW,
  · prices each blocked trade AS IF OPENED under ITS sleeve's current exit, entry = the first 1m open at / after the signal (+ FEE + SLIP):
      FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE  fixed +3 / −3, 12 h cap (the live FRENZY exit since DECISION_LOG 250)
      FRENZY_WILLY                              TP +1 net, NO stop, 120-min cap (its own exit)
      every other sleeve (MOMENTUM, FLIP:*, SPIKE_*, BULLRUN_LONG, SURGE_*, BEARRUN_SHORT …)  PROXY — no kline replica of the momentum /
          flip / spike exit stacks exists in the scout toolset (they need the live BE / trail / EMA ladder state): the row is priced at that
          sleeve's own mean realised P&L % over its closed fills ±3 days in the orders exports (method 'proxy'); none → unpriced.
  · $ = pct × the base notional (the median FRENZY_WILLY fill notional — WILLY trades at 1 × 1.0 = the base) × invest mult × lev mult,
  · reports per sleeve: N blocked, repeats, distinct days, priced N, Σ % and Σ $, and the COST OF THE HOLD = Σ$ of the blocked trades vs
    WILLY's own Σ$ over the same floor (positive cost = the hold gave away more than WILLY earned).
Klines: cached first (reports/cache_willy_hold/), ≤ WEIGHT_STOP used weight per minute read from the X-MBX-USED-WEIGHT-1M header, stop on
HTTP 418 / 429 (no more network this run; unpriced rows are retried next run). Self-test: venv/bin/python scripts/scout_willy_hold.py --selftest
"""
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "reports", "SCOUT_WILLY_HOLD.csv")
CACHE = os.path.join(ROOT, "reports", "cache_willy_hold")
MIN = 60_000
FEE, SLIP = 0.09, 0.10                 # % round-trip fees + entry slippage (the FRENZY scouts' live ruler)
WEIGHT_STOP = 1000                     # stop the network at this used weight per minute (Binance limit 2,400; operator budget ≤ 1,200)
FRENZY3 = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
COLS = ["t", "pair", "dir", "sleeve", "price", "signal_at", "last_at", "repeats", "inv", "lev", "willy_id", "willy_pair",
        "method", "pct", "exit_how", "reason", "exit_ms"]   # reason = the engine's wh_reason (OPEN / UNREAD) · exit_ms = the priced exit minute (266)


class RateLimited(Exception):
    pass


# ─────────────────────────── pure pricing (self-tested) ───────────────────────────
def _live_th():
    """the LIVE thresholds from trading_config.json (round-3 M4: never a hardcoded exit) — an empty namespace when unreadable (defaults)."""
    from types import SimpleNamespace
    try:
        return SimpleNamespace(**(json.load(open(os.path.join(ROOT, "trading_config.json"))).get("thresholds") or {}))
    except Exception:
        return SimpleNamespace()


def exit_rule(sleeve, th=None):
    """the sleeve's CURRENT exit as (tp %, stop % or None, cap minutes) from trading_config.json, or None (no kline replica → proxy).
    FRENZY_WILLY: services.frenzy.frenzy_willy_levels (TP / no stop / cap as the engine uses them) · FRENZY / WIDE / LITE: frenzy_tp_pct /
    frenzy_stop_pct / frenzy_max_hold_minutes (the fixed-TP exit since DECISION_LOG 250; a 0 TP → +3, 0 hold → 12 h)."""
    s = str(sleeve or "")
    th = _live_th() if th is None else th
    if s in FRENZY3:
        g = lambda k, d: (float(getattr(th, k, d)) if getattr(th, k, None) not in (None, "") else d)
        tp = g("frenzy_tp_pct", 3.0) or 3.0
        return tp, -abs(g("frenzy_stop_pct", 3.0) or 3.0), int(g("frenzy_max_hold_minutes", 720) or 720)
    if s == "FRENZY_WILLY":
        sys.path.insert(0, ROOT)
        from services.frenzy import frenzy_willy_levels
        return frenzy_willy_levels(th)
    return None


def walk_fixed(m1, direction, tp, stop, cap_min):
    """net % on 1m rows [open_ms, o, h, l, c] from the entry minute (entry = its open + SLIP; FEE out). Stop checked before TP inside a minute
    (conservative). → (pct, how) or (None, 'no data')."""
    pct, how, _ = walk_fixed_t(m1, direction, tp, stop, cap_min)
    return pct, how


def walk_fixed_t(m1, direction, tp, stop, cap_min):
    """walk_fixed + the exit time (ms: the end of the exit minute; the cap time on a cap; None without data)."""
    if not m1:
        return None, "no data", None
    lg = str(direction).upper() != "SHORT"
    e = float(m1[0][1]) * (1 + SLIP / 100 if lg else 1 - SLIP / 100)
    net = lambda p: ((p / e - 1) * 100 if lg else (1 - p / e) * 100) - FEE
    t_end = int(m1[0][0]) + int(cap_min) * MIN
    last = None
    for t, o, h, l, c in (r[:5] for r in m1):
        if t >= t_end:
            return net(last), "cap", int(t_end)
        worst, best = (net(l), net(h)) if lg else (net(h), net(l))
        if stop is not None and worst <= stop:
            return float(stop), "stop", int(t) + MIN
        if best >= tp:
            return float(tp), "take profit", int(t) + MIN
        last = c
    done = int(m1[-1][0]) + MIN >= t_end
    return net(last), ("cap" if done else "open"), (int(t_end) if done else None)


def cost_summary(rows, willy_usd):
    """per sleeve (N, repeats, days, priced, Σ%, Σ$) + the cost of the hold = Σ$ blocked − WILLY's own Σ$."""
    if not len(rows):
        return [], 0.0, -float(willy_usd or 0.0)
    out = []
    for sl, g in rows.groupby("sleeve"):
        p = pd.to_numeric(g.pct, errors="coerce")
        out.append(dict(sleeve=sl, n=len(g), repeats=int(pd.to_numeric(g.repeats, errors="coerce").fillna(1).sum()),
                        days=g.t.astype(str).str[:10].nunique(), priced=int(p.notna().sum()), sum_pct=float(p.sum()),
                        sum_usd=float(pd.to_numeric(g.get("usd"), errors="coerce").sum()) if "usd" in g else 0.0))
    out.sort(key=lambda d: -d["n"])
    blocked = sum(d["sum_usd"] for d in out)
    return out, blocked, blocked - float(willy_usd or 0.0)


def merge(old, new):
    """stored + new rows → one row per (t, pair, dir, sleeve, willy_id); the export's repeats / last_at win; a priced row keeps its price."""
    parts = [x for x in (old, new) if x is not None and len(x)]
    if not parts:
        return pd.DataFrame(columns=COLS)
    m = pd.concat(parts, ignore_index=True).reindex(columns=COLS)
    for c in ("method", "exit_how", "reason"):           # text columns: an all-NaN read is float64 and refuses "klines 1m" (Oct-9 crash)
        m[c] = m[c].astype(object)
    k = ["t", "pair", "dir", "sleeve", "willy_id"]
    m["willy_id"] = m.willy_id.astype(str)
    priced = m[m.pct.notna()].drop_duplicates(k, keep="first").set_index(k)[["method", "pct", "exit_how", "exit_ms"]]
    m = m.drop_duplicates(k, keep="last").set_index(k)
    for c in ("method", "pct", "exit_how", "exit_ms"):
        m.loc[priced.index, c] = priced[c]
    return m.reset_index().reindex(columns=COLS).sort_values("t").reset_index(drop=True)


def rate_check(status, used_weight):
    """→ raise RateLimited on HTTP 418 / 429 or a used weight at the stop line (pure)."""
    if status in (418, 429) or (used_weight is not None and int(used_weight) >= WEIGHT_STOP):
        raise RateLimited(f"status {status} · used weight {used_weight}")


# ─────────────────────────── I/O ───────────────────────────
def floor_ms():
    try:
        sd = os.path.join(ROOT, "scripts")
        if sd not in sys.path:
            sys.path.insert(0, sd)
        import scout_revert_gates as RG
        return int(RG.deploy_ms("FRENZY_WILLY")) - 10 * MIN   # the commit time itself (deploy_ms adds 10 min)
    except Exception:
        return int(time.time() * 1000)


def _ms(s):
    try:
        return int(pd.Timestamp(str(s)[:19], tz="UTC").value // 1_000_000)
    except Exception:
        return None


def export_rows():
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")):
        try:
            d = pd.read_csv(f, dtype=str, keep_default_na=False, low_memory=False)
        except Exception:
            continue
        if "e" in d and "wh_willy_id" in d:
            fr.append(d[d.e == "WILLY_HOLD"].assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=COLS)
    d = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    return pd.DataFrame(dict(t=d.t, pair=d.pair, dir=d.dir, sleeve=d.strategy, price=pd.to_numeric(d.price, errors="coerce"),
                             signal_at=d.wh_signal_at, last_at=d.wh_last_at, repeats=pd.to_numeric(d.wh_repeats, errors="coerce"),
                             inv=pd.to_numeric(d.wh_invest_mult, errors="coerce"), lev=pd.to_numeric(d.wh_lev_mult, errors="coerce"),
                             willy_id=d.wh_willy_id.astype(str), willy_pair=d.wh_willy_pair, method=np.nan, pct=np.nan, exit_how=np.nan,
                             reason=(d.wh_reason if "wh_reason" in d else np.nan), exit_ms=np.nan))


def orders():
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "entry_strategy", "status", "pnl_percentage",
                                                                        "pnl", "notional_value", "direction"))
            fr.append(d.assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=["opened_at", "pair", "entry_strategy", "status", "pnl_percentage", "pnl", "notional_value", "direction"])
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable").drop_duplicates(["opened_at", "pair"], keep="last")
    return o


def klines_1m(sym, start, end, state):
    """cached 1m klines [open_ms, o, h, l, c] for [start, end) — the cache first, else the public endpoint under the weight budget."""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{sym}_{int(start)}_{int(end)}.json")
    if os.path.exists(fp):
        try:
            return json.load(open(fp))
        except Exception:
            pass
    if state.get("stopped"):
        raise RateLimited("stopped earlier this run")
    out, s = [], int(start)
    while s < end:
        q = urllib.parse.urlencode(dict(symbol=sym, interval="1m", startTime=s, endTime=int(end), limit=1000))
        try:
            r = urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?{q}", timeout=20)
            rate_check(r.status, r.headers.get("X-MBX-USED-WEIGHT-1M"))
            data = json.loads(r.read())
        except urllib.error.HTTPError as ex:
            state["stopped"] = True
            rate_check(ex.code, None)
            raise
        except RateLimited:
            state["stopped"] = True
            raise
        if not data:
            break
        out += [[int(x[0])] + [float(v) for v in x[1:5]] for x in data]
        nxt = int(data[-1][0]) + MIN
        if nxt <= s:
            break
        s = nxt
        time.sleep(0.1)
    if out and int(out[-1][0]) + MIN >= end:   # complete windows only are cached
        json.dump(out, open(fp, "w"))
    return out


def price_rows(rows, o, now_ms):
    state = {}
    for i, r in rows.iterrows():
        if pd.notna(r.pct):
            continue
        rule = exit_rule(r.sleeve)
        sig = _ms(r.signal_at) or _ms(r.t)
        if rule is None:
            g = o[(o.entry_strategy.astype(str) == str(r.sleeve)) & (o.status.astype(str) == "CLOSED")] if len(o) else o
            if len(g) and sig:
                ts = g.opened_at.map(_ms)
                g = g[(ts - sig).abs() <= 3 * 86_400_000]
            p = pd.to_numeric(g.pnl_percentage, errors="coerce").dropna() if len(g) else pd.Series(dtype=float)
            if len(p):
                rows.at[i, "pct"] = float(p.mean()); rows.at[i, "method"] = "proxy"; rows.at[i, "exit_how"] = f"sleeve mean of {len(p)} fills ±3 d"
            continue
        tp, stop, cap = rule
        if sig is None or sig + cap * MIN + MIN > now_ms:
            continue   # not finished yet
        start = (sig // MIN + (1 if sig % MIN else 0)) * MIN
        try:
            m1 = klines_1m(str(r.pair), start, start + cap * MIN, state)
        except RateLimited:
            break
        except Exception:
            continue
        pct, how, xt = walk_fixed_t(m1, r.dir, tp, stop, cap)
        if pct is not None:
            rows.at[i, "pct"] = round(float(pct), 4); rows.at[i, "method"] = "klines 1m"; rows.at[i, "exit_how"] = how
            rows.at[i, "exit_ms"] = xt
    return rows, state.get("stopped", False)


def run(now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    fl = floor_ms()
    old = pd.read_csv(STORE, dtype={"willy_id": str}) if os.path.exists(STORE) else pd.DataFrame(columns=COLS)
    rows = merge(old, export_rows())
    rows = rows[rows.t.map(lambda x: (_ms(x) or 0) >= fl)].reset_index(drop=True)
    o = orders()
    rows, stopped = price_rows(rows, o, now_ms)
    if len(rows):
        tmp = f"{STORE}.{os.getpid()}.tmp"; rows.to_csv(tmp, index=False); os.replace(tmp, STORE)
    w = o[(o.entry_strategy.astype(str) == "FRENZY_WILLY")] if len(o) else o
    w = w[w.opened_at.map(lambda x: (_ms(x) or 0) >= fl)] if len(w) else w
    base = float(pd.to_numeric(w.notional_value, errors="coerce").median()) if len(w) else float("nan")
    wc = w[w.status.astype(str) == "CLOSED"] if len(w) else w
    w_usd = float(pd.to_numeric(wc.pnl, errors="coerce").sum()) if len(wc) else 0.0
    if len(rows):
        rows["usd"] = pd.to_numeric(rows.pct, errors="coerce") / 100 * base * pd.to_numeric(rows.inv, errors="coerce").fillna(1.0) \
            * pd.to_numeric(rows.lev, errors="coerce").fillna(1.0)
    per, blocked, cost = cost_summary(rows, w_usd)
    L = ["## 🔒 WILLY_HOLD — the cost of the FRENZY_WILLY global hold (DECISION_LOG 251, declared exception, no auto-off)", "",
         f"_Rows from {pd.Timestamp(fl, unit='ms'):%Y-%m-%d %H:%M} UTC (the (DECISION_LOG 251) commit). Blocked trades priced AS IF OPENED: "
         "FRENZY / WIDE / LITE at the fixed +3 / −3 / 12 h, WILLY at TP +1 / no stop / 120 min (1m klines, FEE 0.09 + SLIP 0.10); other sleeves "
         "at their own ±3-day mean realised % (PROXY — no kline replica of their exit stacks). $ = % × the base notional (median WILLY fill"
         f"{' $%.0f' % base if base == base else ' — none yet'}) × the blocked trade's invest × lev mult._" + (" ⚠ network stopped (418 / 429 / weight) — unpriced rows retry next run." if stopped else ""), ""]
    if not len(rows):
        return L + ["No refused trade recorded yet.", ""]
    L += ["| Sleeve | Setups | Tries | Days | Priced | Σ % | Σ $ |", "|---|---|---|---|---|---|---|"]
    for d in per:
        L.append(f"| {d['sleeve']} | {d['n']} | {d['repeats']} | {d['days']} | {d['priced']} | {d['sum_pct']:+.2f} | {d['sum_usd']:+.2f} |")
    L += ["", f"**Cost of the hold:** blocked Σ$ {blocked:+.2f} vs FRENZY_WILLY's own Σ$ {w_usd:+.2f} ({len(wc)} closed) → "
          f"{'the hold gave away' if cost > 0 else 'the hold saved'} ${abs(cost):.2f} (positive blocked Σ = trades the hold prevented that "
          "would have won).", ""]
    return L


def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    t0 = 1_790_000_000_000 // MIN * MIN
    m = lambda ps: [[t0 + i * MIN, p, p * 1.001, p * 0.999, p] for i, p in enumerate(ps)]
    chk(exit_rule("FRENZY_LITE") == (3.0, -3.0, 720) and exit_rule("FRENZY_WILLY") == (1.0, None, 120) and exit_rule("MOMENTUM") is None,
        "sleeve exits")
    from types import SimpleNamespace as _NS
    chk(exit_rule("FRENZY_WILLY", _NS(frenzy_willy_tp_pct=2.0, frenzy_willy_max_hold_minutes=90)) == (2.0, None, 90)
        and exit_rule("FRENZY_LONG", _NS(frenzy_tp_pct=4.0, frenzy_stop_pct=2.5, frenzy_max_hold_minutes=600)) == (4.0, -2.5, 600),
        "exits read from the live config (never hardcoded)")
    pct, how = walk_fixed(m([100, 100.5, 102]), "LONG", 1.0, None, 120)
    chk(pct == 1.0 and how == "take profit", "WILLY TP +1")
    pct, how = walk_fixed(m([100] + [90] * 130), "LONG", 1.0, None, 120)
    chk(how == "cap" and pct < -9, "WILLY: no stop — a −10 % trade rides to the 120-min cap")
    pct, how = walk_fixed(m([100, 96]), "LONG", 3.0, -3.0, 720)
    chk(pct == -3.0 and how == "stop", "FRENZY −3 stop")
    chk(walk_fixed_t(m([100, 96]), "LONG", 3.0, -3.0, 720)[2] == t0 + 2 * MIN and walk_fixed_t(m([100] * 3), "LONG", 3.0, -3.0, 2)[2] == t0 + 2 * MIN
        and walk_fixed_t(m([100] * 3), "LONG", 3.0, -3.0, 720)[2] is None, "exit time: the stop minute's end · the cap · None while open")
    pct, _ = walk_fixed(m([100, 96]), "SHORT", 3.0, -3.0, 720)
    chk(pct == 3.0, "a SHORT priced the other way")
    chk(walk_fixed([], "LONG", 1, None, 120) == (None, "no data"), "no data")
    r = pd.DataFrame(dict(t=["2026-10-09T01:00:00", "2026-10-09T01:00:00", "2026-10-10T02:00:00"], sleeve=["MOMENTUM", "MOMENTUM", "FRENZY_LONG"],
                          pct=[0.5, np.nan, -3.0], repeats=[3, 1, 2], usd=[10.0, np.nan, -60.0]))
    per, blocked, cost = cost_summary(r, 25.0)
    chk(per[0]["sleeve"] == "MOMENTUM" and per[0]["n"] == 2 and per[0]["repeats"] == 4 and per[0]["priced"] == 1 and blocked == -50.0 and cost == -75.0,
        "cost of the hold = Σ$ blocked − WILLY Σ$")
    a = pd.DataFrame(dict(t=["T1"], pair=["X"], dir=["LONG"], sleeve=["MOMENTUM"], willy_id=["7"], repeats=[1], pct=[0.4], method=["proxy"], exit_how=["m"]))
    b = pd.DataFrame(dict(t=["T1"], pair=["X"], dir=["LONG"], sleeve=["MOMENTUM"], willy_id=["7"], repeats=[5], pct=[np.nan], method=[np.nan], exit_how=[np.nan]))
    mm = merge(a, b)
    chk(len(mm) == 1 and mm.repeats.iloc[0] == 5 and mm.pct.iloc[0] == 0.4, "merge: one row per setup, the export's repeats, the stored price")
    fresh = merge(None, b.assign(t=["T2"]))              # every row unpriced → method / exit_how all NaN (the Oct-9 crash shape)
    fresh.at[0, "method"] = "klines 1m"; fresh.at[0, "exit_how"] = "take profit"; fresh.at[0, "pct"] = 3.0
    chk(fresh.method.iloc[0] == "klines 1m" and fresh.exit_how.iloc[0] == "take profit", "an all-unpriced merge accepts a text price label")
    for st, w, bad in ((429, None, True), (418, None, True), (200, "1000", True), (200, "999", False), (200, None, False)):
        try:
            rate_check(st, w); chk(not bad, f"rate {st} {w}")
        except RateLimited:
            chk(bad, f"rate {st} {w}")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
