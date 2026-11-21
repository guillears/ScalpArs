#!/usr/bin/env python3
"""📡 Route B — per-coin, per-minute FEATURE table from one recorded hour (DECISION_LOG 597).

Raw hour files (book / trade / liq / mark, ≈ 230 MB gzip per hour for ~100 coins) → one small CSV (≈ 1 row per coin-minute) that every
report reads in seconds. Runs anywhere the raw files are (the plan: on the recorder server every hour, uploaded to features/…).

Columns per (sym, minute = UTC minute start ms):
  book (top-20 snapshots, 500 ms):  n_book · mid (last) · mid_open · mid_hi · mid_lo · spread_bps (mean) ·
     bid_usd_02 / ask_usd_02 / bid_usd_05 / ask_usd_05 / bid_usd_1 / ask_usd_1  ($ resting within 0.2 / 0.5 / 1 % of mid, mean) ·
     imb_05 = (bid_usd_05 − ask_usd_05) ÷ (bid_usd_05 + ask_usd_05) (mean) · reach_bid_pct / reach_ask_pct (how far the 20 levels reach,
     mean — a band beyond it is a LOWER bound) · cost_buy_2500 / cost_sell_2500 / cost_buy_10000 / cost_sell_10000 (bps vs mid to fill
     that $ notional at market by walking the last snapshot of the minute; NaN when the 20 levels are not enough)
  trades (aggTrade): n_trades · buy_usd (taker buys) · sell_usd (taker sells) · max_trade_usd · vwap
  liquidations (forceOrder snapshot stream — a SAMPLE): liq_long_usd (longs force-sold) · liq_short_usd (shorts force-bought)
  mark (1 s): mark (last) · index (last) · funding (last)
  FULL book (fbook, 1 s, the bot's priority coins only — fullbook.py; every value the MEAN over the minute's synced seconds):
     n_fbook · fb_spread_bps · fb_reach_bid_pct / fb_reach_ask_pct (how far the trusted full book reaches; 100 = whole side) ·
     fb_bid_usd_<01|02|05|1|2|5> / fb_ask_usd_… ($ within 0.1 / 0.2 / 0.5 / 1 / 2 / 5 % of mid) ·
     fb_cost_buy_<1000|3000|6000|10000|25000> / fb_cost_sell_… (bps vs mid to fill that $ at market; mean of the seconds where the book
     within 5 % was enough — blank when never) · fb_cost_na_25000 (seconds where $25k did NOT fit within 5 %, either side)
  open interest (oi, every 60 s, hot coins — positioning.py): oi (contracts, last of the minute) · oi_chg_1m / oi_chg_5m (% vs the
     recorder's sweep 1 / 5 min earlier — computed live, so an hour boundary never blanks them)
  positioning (ratio, every 5 min, period 5m, the latest row; carried forward inside the hour from the minute it was read):
     top_ls_acct · top_ls_pos · global_ls_acct (long ÷ short) · taker_bs_ratio (taker buy ÷ sell vol) · ratio_ts (that row's period ms)
Usage: python3 features.py <out.csv[.gz]> --book F [F …] [--trade F …] [--liq F …] [--mark F …] [--fbook F …] [--oi F …] [--ratio F …]
       (several files per kind = one per recorder process that wrote that hour; a .gz output is gzip-compressed)"""
import argparse
import csv
import gzip
import json
import math
import sys
import os
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # fullbook.py beside this file (also under python -I)
try:                                          # the full-book record layout (same folder; a missing file only drops the fb_* columns)
    from fullbook import FB_BAND_TAGS, FB_SIZES
except Exception:                             # pragma: no cover
    FB_BAND_TAGS, FB_SIZES = ("01", "02", "05", "1", "2", "5"), (1000, 3000, 6000, 10000, 25000)

BANDS = ((0.002, "02"), (0.005, "05"), (0.01, "1"))
FILL_USD = (2500.0, 10000.0)
COLS = ["sym", "minute", "n_book", "mid", "mid_open", "mid_hi", "mid_lo", "spread_bps",
        "bid_usd_02", "ask_usd_02", "bid_usd_05", "ask_usd_05", "bid_usd_1", "ask_usd_1", "imb_05", "reach_bid_pct", "reach_ask_pct",
        "cost_buy_2500", "cost_sell_2500", "cost_buy_10000", "cost_sell_10000",
        "n_trades", "buy_usd", "sell_usd", "max_trade_usd", "vwap", "liq_long_usd", "liq_short_usd", "mark", "index", "funding"]
FB_COLS = (["n_fbook", "fb_spread_bps", "fb_reach_bid_pct", "fb_reach_ask_pct"]
           + [f"fb_{s}_usd_{t}" for t in FB_BAND_TAGS for s in ("bid", "ask")]
           + [f"fb_cost_{s}_{z}" for z in FB_SIZES for s in ("buy", "sell")] + ["fb_cost_na_25000"])
POS_COLS = ["oi", "oi_chg_1m", "oi_chg_5m", "top_ls_acct", "top_ls_pos", "global_ls_acct", "taker_bs_ratio", "ratio_ts"]
COLS = COLS + FB_COLS + POS_COLS              # appended: the old columns keep their names and order


def _lines(paths):
    """JSON objects of one or several gzip JSONL files; a truncated last member (power loss) ends that file's read quietly."""
    for path in ([paths] if isinstance(paths, str) else (paths or [])):
        try:
            with gzip.open(path, "rt") as f:
                for ln in f:
                    try:
                        yield json.loads(ln)
                    except ValueError:
                        continue
        except (EOFError, OSError, zlib.error, UnicodeDecodeError):
            continue


def walk_cost_bps(levels, usd, mid, side):
    """bps vs mid to fill `usd` by walking `levels` [[price, qty], …] (asks for a buy, bids for a sell); NaN if not enough. Pure."""
    left, cost_qty, spent = usd, 0.0, 0.0
    for p, q in levels:
        p, q = float(p), float(q)
        take = min(left, p * q)
        spent += take
        cost_qty += take / p
        left -= take
        if left <= 1e-9:
            avg = spent / cost_qty
            return (avg / mid - 1.0) * 1e4 if side == "buy" else (1.0 - avg / mid) * 1e4
    return float("nan")


def book_row(bids, asks):
    """one snapshot → (mid, spread_bps, {band: (bid$, ask$)}, reach_bid_pct, reach_ask_pct). None if unusable. Pure."""
    try:
        bb, ba = float(bids[0][0]), float(asks[0][0])
    except (IndexError, TypeError, ValueError):
        return None
    if not (bb > 0 and ba > 0 and ba >= bb):
        return None
    mid = (bb + ba) / 2.0
    try:
        bl = [(float(p), float(q)) for p, q in bids]
        al = [(float(p), float(q)) for p, q in asks]
    except (TypeError, ValueError):
        return None
    bands = {}
    for frac, tag in BANDS:
        lo, hi = mid * (1 - frac), mid * (1 + frac)
        bands[tag] = (sum(p * q for p, q in bl if p >= lo), sum(p * q for p, q in al if p <= hi))
    reach_b = (1 - bl[-1][0] / mid) * 100
    reach_a = (al[-1][0] / mid - 1) * 100
    return mid, (ba - bb) / mid * 1e4, bands, reach_b, reach_a


class Acc:
    __slots__ = ("n", "mids", "spread", "bands", "imb", "rb", "ra", "last", "nt", "buy", "sell", "mx", "pq", "q", "ll", "ls", "mark", "fb", "oi")

    def __init__(self):
        self.n, self.mids, self.spread, self.imb, self.rb, self.ra = 0, [], 0.0, 0.0, 0.0, 0.0
        self.bands = {t: [0.0, 0.0] for _, t in BANDS}
        self.last = None
        self.nt, self.buy, self.sell, self.mx, self.pq, self.q, self.ll, self.ls = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.mark = None
        self.fb = None                            # full book: column → [sum, count]
        self.oi = None                            # (r ms, oi, chg1, chg5) — the newest of the minute


def _fb_add(x, o):
    """one fbook line → the minute's sums. Pure on x."""
    fb = x.fb
    if fb is None:
        fb = x.fb = {"n": [0, 0]}

    def add(col, v):
        if v is None:
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isnan(v):
            return
        c = fb.setdefault(col, [0.0, 0])
        c[0] += v
        c[1] += 1
    fb["n"][0] += 1
    add("fb_spread_bps", o.get("sp"))
    add("fb_reach_bid_pct", o.get("rb"))
    add("fb_reach_ask_pct", o.get("ra"))
    b, a, cb, cs = o.get("b") or [], o.get("a") or [], o.get("cb") or [], o.get("cs") or []
    for i, t in enumerate(FB_BAND_TAGS):
        add(f"fb_bid_usd_{t}", b[i] if i < len(b) else None)
        add(f"fb_ask_usd_{t}", a[i] if i < len(a) else None)
    for i, z in enumerate(FB_SIZES):
        add(f"fb_cost_buy_{z}", cb[i] if i < len(cb) else None)
        add(f"fb_cost_sell_{z}", cs[i] if i < len(cs) else None)
    if 25000 in FB_SIZES:
        i = FB_SIZES.index(25000)
        if (i < len(cb) and cb[i] is None) or (i < len(cs) and cs[i] is None):
            fb["n"][1] += 1


def build(book, trade=None, liq=None, mark=None, fbook=None, oi=None, ratio=None):
    acc = {}

    def A(sym, ms):
        k = (sym, int(ms) // 60_000 * 60_000)
        a = acc.get(k)
        if a is None:
            a = acc[k] = Acc()
        return a
    for o in _lines(book):
        sym, t, b, a_ = o.get("s"), o.get("T") or o.get("E") or o.get("r"), o.get("b") or [], o.get("a") or []
        r = book_row(b, a_) if sym and t else None
        if r is None:
            continue
        mid, sp, bands, rb, ra = r
        x = A(sym, t)
        x.n += 1
        x.mids.append((int(t), mid))
        x.spread += sp
        x.rb += rb
        x.ra += ra
        for tag, (bv, av) in bands.items():
            x.bands[tag][0] += bv
            x.bands[tag][1] += av
        b5, a5 = bands["05"]
        x.imb += (b5 - a5) / (b5 + a5) if (b5 + a5) > 0 else 0.0
        if x.last is None or int(t) >= x.last[0]:                 # the newest snapshot by exchange time (several files per hour)
            x.last = (int(t), mid, b, a_)
    for o in _lines(trade):
        d = o.get("d") or {}
        try:
            p, q, t, sym = float(d["p"]), float(d["q"]), int(d["T"]), d["s"]
        except (KeyError, TypeError, ValueError):
            continue
        x = A(sym, t)
        usd = p * q
        x.nt += 1
        if d.get("m"):
            x.sell += usd                  # buyer was the maker → the TAKER sold
        else:
            x.buy += usd
        x.mx = max(x.mx, usd)
        x.pq += usd
        x.q += q
    for o in _lines(liq):
        od = (o.get("d") or {}).get("o") or {}
        try:
            usd = float(od.get("ap") or od.get("p")) * float(od.get("z") or od.get("q"))
            t, sym, side = int(od["T"]), od["s"], od["S"]
        except (KeyError, TypeError, ValueError):
            continue
        x = A(sym, t)
        if side == "SELL":
            x.ll += usd                    # a long position force-sold
        else:
            x.ls += usd
    for o in _lines(mark):
        t = o.get("E") or o.get("r")
        for row in o.get("m") or []:
            try:
                y = A(row[0], t)
                if y.mark is None or int(t) >= y.mark[0]:
                    y.mark = (int(t), row[1], row[2], row[3])
            except (IndexError, TypeError):
                continue
    for o in _lines(fbook):
        sym, t = o.get("s"), o.get("r")
        if not sym or not t:
            continue
        _fb_add(A(sym, t), o)
    for o in _lines(oi):
        t = o.get("r")
        for row in o.get("m") or []:
            try:
                x = A(row[0], t)
                if x.oi is None or int(t) >= x.oi[0]:
                    x.oi = (int(t), row[1], row[3], row[4])
            except (IndexError, TypeError, ValueError):
                continue
    rat = {}                                      # sym → [(minute, [ts, tla, tpa, gla, tkr])] — carried forward below
    for o in _lines(ratio):
        try:
            m0 = int(o.get("r")) // 60_000 * 60_000
        except (TypeError, ValueError):
            continue
        for row in o.get("m") or []:
            if isinstance(row, list) and len(row) >= 6 and row[0]:
                rat.setdefault(row[0], []).append((m0, row[1:6]))
                A(row[0], m0)
    for v in rat.values():
        v.sort(key=lambda z: z[0])
    rows = []
    for (sym, m), x in acc.items():
        r = {"sym": sym, "minute": m, "n_book": x.n, "n_trades": x.nt, "buy_usd": round(x.buy, 2), "sell_usd": round(x.sell, 2),
             "max_trade_usd": round(x.mx, 2), "vwap": (x.pq / x.q if x.q else ""), "liq_long_usd": round(x.ll, 2), "liq_short_usd": round(x.ls, 2)}
        if x.mark:
            r["mark"], r["index"], r["funding"] = x.mark[1:]
        if x.n:
            ms_ = sorted(x.mids)
            mv = [m_ for _, m_ in ms_]
            r.update(mid=mv[-1], mid_open=mv[0], mid_hi=max(mv), mid_lo=min(mv), spread_bps=round(x.spread / x.n, 3),
                     imb_05=round(x.imb / x.n, 4), reach_bid_pct=round(x.rb / x.n, 3), reach_ask_pct=round(x.ra / x.n, 3))
            for tag, (bv, av) in x.bands.items():
                r[f"bid_usd_{tag}"], r[f"ask_usd_{tag}"] = round(bv / x.n, 2), round(av / x.n, 2)
            _, mid, b, a_ = x.last
            for usd in FILL_USD:
                cb, cs = walk_cost_bps(a_, usd, mid, "buy"), walk_cost_bps(b, usd, mid, "sell")
                r[f"cost_buy_{int(usd)}"] = "" if math.isnan(cb) else round(cb, 2)
                r[f"cost_sell_{int(usd)}"] = "" if math.isnan(cs) else round(cs, 2)
        if x.oi:
            r["oi"], r["oi_chg_1m"], r["oi_chg_5m"] = (("" if v is None else v) for v in x.oi[1:])
        obs = [z for z in rat.get(sym, ()) if z[0] <= m]
        if obs:
            ts, tla, tpa, gla, tkr = obs[-1][1]
            r.update({k: ("" if v is None else v) for k, v in
                      (("ratio_ts", ts), ("top_ls_acct", tla), ("top_ls_pos", tpa), ("global_ls_acct", gla), ("taker_bs_ratio", tkr))})
        if x.fb:
            r["n_fbook"], r["fb_cost_na_25000"] = x.fb["n"]
            for col, (sm, c) in x.fb.items():
                if col != "n" and c:
                    r[col] = round(sm / c, 2 if "usd" in col else 3)
        rows.append(r)
    rows.sort(key=lambda r: (r["sym"], r["minute"]))
    return rows


def write(rows, out):
    op = (lambda: gzip.open(out, "wt", newline="")) if out.endswith(".gz") else (lambda: open(out, "w", newline=""))
    with op() as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    for k in ("book", "trade", "liq", "mark", "fbook", "oi", "ratio"):
        ap.add_argument(f"--{k}", nargs="*", default=[])
    a = ap.parse_args(argv)
    rows = build(a.book, a.trade, a.liq, a.mark, a.fbook, a.oi, a.ratio)
    write(rows, a.out)
    print(f"{len(rows)} coin-minutes → {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
