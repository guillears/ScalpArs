#!/usr/bin/env python3
"""WILLY late joiners (2026-10-22) — LIVE / MASTER blocks (①②). READ-ONLY on the bot.
  --prep   the engine's own "late window — no entry A" refusals (CloudWatch, saved to data/cw_latewindow_refusals.json) with L ≤ 12 bars:
           5m klines from fapi (weight 2 per call, scripts/binance_cooldown.py honoured, used-weight header read, stop on 418 / 429) →
           the R1-30 / R1-60 / R2 red bar + R at the red bar (same rules as study_willy_latejoin_build.py) → data/live_entries.csv and
           data/need_days_live.csv (incl. the real WILLY fills' days for the walker-parity re-walk).
  --price  (after study_willy_latejoin_fetch.py --extra need_days_live.csv) walk every row → reports/WILLY_LATE_JOIN_2026-10-22_live.csv;
           real WILLY fills (MASTER_POOL_stacked.csv, FRENZY_WILLY) re-walked from their booked entry → _onTime_live.csv (parity).
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_latejoin_common as C  # noqa: E402
import binance_cooldown as BC  # noqa: E402

CW = os.path.join(C.D, "cw_latewindow_refusals.json")
RX = re.compile(r"\[FRENZY_WILLY\] (\w+): episode \(spike (\d\d)-(\d\d) (\d\d):(\d\d) UTC\).* and (\d+) bars old")


def fapi_5m(pair, start, end):
    if BC.active():
        raise SystemExit("cooldown active — stopping")
    u = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={start}&endTime={end}&limit=499"
    try:
        with urllib.request.urlopen(u, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0); x = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"fapi {e.code} — stopping")
        raise
    if w > 900:
        time.sleep(60)
    time.sleep(0.3)
    now = time.time() * 1000
    return pd.DataFrame([[int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[7])] for a in x if int(a[6]) < now],
                        columns=["t", "o", "h", "l", "c", "qv"])


def refusals():
    out = []
    for ts, m in json.load(open(CW)):
        g = RX.search(m)
        if not g:
            continue
        pair, mo, dd, hh, mi, L = g.groups()
        sp = pd.Timestamp(f"2026-{mo}-{dd} {hh}:{mi}")
        out.append(dict(pair=pair, spike_close=int(sp.value // 10**6), L=int(L), log_ms=int(ts)))
    return pd.DataFrame(out).drop_duplicates(["pair", "spike_close"])


def prep():
    rf = refusals()
    rf.to_csv(os.path.join(C.D, "cw_refusals_parsed.csv"), index=False)
    sup = C.supply(); rows = []; need = set()
    for r in rf[rf.L <= 12].itertuples():
        k = fapi_5m(r.pair, r.spike_close - 24 * 3600_000 - 40 * C.MIN, r.spike_close + 3 * 3600_000)
        if not len(k):
            continue
        t = k.t.values; o, c = k.o.values, k.c.values
        vol24 = pd.Series(k.qv.values).rolling(288, min_periods=288).sum().values
        on = int(np.searchsorted(t, r.spike_close - C.BAR)); j = on + r.L
        cs = sup.get(r.pair)
        for rule, a, e in (("R1-30", j, j + 12), ("R1-60", j, j + 12), ("R2", j, on + 12)):
            if rule == "R1-30" and r.L > 6:
                continue
            kk = next((x for x in range(a, min(e, len(t) - 1) + 1) if c[x] < o[x]), None) if a <= e else None
            d = dict(pair=r.pair, spike_close=r.spike_close, L=r.L, rule=rule, block=C.block(r.spike_close), log_ms=r.log_ms)
            if kk is None:
                d.update(st="red_expired"); rows.append(d); continue
            rc = int(t[kk] + C.BAR); Rv = vol24[kk] / (cs * c[kk]) if (cs and np.isfinite(vol24[kk])) else np.nan
            d.update(st="ok", red_close=rc, wait_bars=kk - a, R=Rv, vol24_red=vol24[kk],
                     route=("UNREAD" if not np.isfinite(Rv) else ("FULL" if Rv < C.RCUT else "HT")),
                     min_spike_to_entry=(rc - r.spike_close) / C.MIN, run_from_base=(c[kk] / c[on - 6] - 1) * 100,
                     vol24_spike=vol24[on])
            rows.append(d)
            for x in (rc, rc + (C.CAP + 2) * C.MIN):
                need.add((r.pair, pd.Timestamp(int(x), unit="ms").strftime("%Y-%m-%d")))
    pd.DataFrame(rows).to_csv(os.path.join(C.D, "live_entries.csv"), index=False)
    m = real_fills()
    for p, te in zip(m.pair, m.te):
        for x in (te, te + (C.CAP + 2) * C.MIN):
            need.add((p, pd.Timestamp(int(x), unit="ms").strftime("%Y-%m-%d")))
    pd.DataFrame(sorted(need), columns=["pair", "day"]).to_csv(os.path.join(C.D, "need_days_live.csv"), index=False)
    print(pd.DataFrame(rows).to_string())


def real_fills():
    m = pd.read_csv(os.path.join(C.R, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[(m.entry_strategy == "FRENZY_WILLY") & (m.status == "CLOSED")].copy()
    m["te"] = [int(pd.Timestamp(str(x)[:19]).value // 10**6) for x in m.opened_at]
    m["block"] = [C.block(x) for x in m.te]
    return m


def price():
    le = pd.read_csv(os.path.join(C.D, "live_entries.csv"))
    out = []
    for r in le.to_dict("records"):
        if r["st"] == "ok":
            r.update(C.walk(r["pair"], int(r["red_close"])))
        out.append(r)
    o = pd.DataFrame(out); o["rule_exit"] = C.RULE
    o.to_csv(C.OUTP + "_live.csv", index=False)
    print(o[[c for c in ["block", "pair", "L", "rule", "st", "wait_bars", "min_spike_to_entry", "R", "route", "why", "pct"] if c in o]].to_string())
    m = real_fills(); res = []
    for r in m.itertuples():
        P = C.path_1m(r.pair, int(r.te), float(r.entry_price))
        ev = C.WT.evaluate_late(P, C.TP, C.SL, C.CAP, C.LATE_M, C.LATE_P, book=C.BOOK) if len(P["t"]) else dict(why="UNRES", pct=np.nan)
        res.append(dict(block=r.block, pair=r.pair, opened_at=r.opened_at, booked_pct=r.pnl_percentage, stack_pct=r.stack_pct,
                        close_reason=r.close_reason, R=r.entry_vol_mcap_ratio, walk_why=ev["why"], walk_pct=ev["pct"]))
    f = pd.DataFrame(res); f.to_csv(C.OUTP + "_onTime_live.csv", index=False)
    print(f.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prep", action="store_true"); ap.add_argument("--price", action="store_true")
    A = ap.parse_args()
    if A.prep:
        prep()
    if A.price:
        price()
