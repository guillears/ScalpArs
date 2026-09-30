#!/usr/bin/env python3
"""⚡ SURGE-SHORT design (operator, 2026-09-30 ~13:40 UTC: BTC 85,632 → 83,964 in minutes — "same thing for shorting").
Mirror of scripts/surge_sleeve_design.py, walked on 1m bars with the same validated candle-path simulator (short side).

PRE-DECLARED (before any result):
  trigger  DUMP       BTC 30-min return ≤ −1.0 % ∧ bar quote volume ≥ 3× 24 h median (≥ 4 h apart)   ← today's move qualifies
           BREAKDOWN  DUMP ∧ BTC close ≤ prior 24 h low                                        ← strict mirror of the long trigger
  universe top-20 tradeable alts by 24 h volume at the event
  select   ALL · HI_ATR (5m ATR ≥ 1.5 %) · LEADER (pair 30-min return < BTC's — falling faster) · HI_ATR_LEADER
  entry    0 / 10 / 20 min after the event bar closes
  exit     BOT_SHORT  −0.70 stop; runner arms +0.45; floor max(peak − 0.5×ATR, +0.10)   (live short runner, lock assumed on)
           KEEP60     −0.70 stop; arm +0.40; floor max(0.60 × peak, +0.10); 60 min
           QUICK      −0.50 stop; arm +0.30; floor max(0.50 × peak, +0.10); 30 min
           BULLRUN_1ATR / BULLRUN_2ATR  the live Bull-Run exit mirrored to the short side (REARM 1×ATR / GREEN 2×ATR trail)
  bar      same as the long grid: both halves > 0, beats the control (same cell, 1 day earlier) in both, 95 % CI > 0, ≥ 20 events.
Usage: venv/bin/python scripts/surge_short_design.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_1m as M
V2 = M.V2
MIN = M.MIN
SPLIT = 1777593600000
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:"); sys.path.insert(0, V2.ROOT)
import services.trading_engine as _TE

btc = V2.btc
lo24 = btc.l.shift(1).rolling(288, min_periods=200).min()


def events(breakdown):
    ok = (V2.r30 * 100 <= -1.0) & (btc.qvol >= 3 * V2.vmed)
    if breakdown:
        ok &= btc.c <= lo24
    out, last = [], -10**18
    for t in btc.index[ok.fillna(False).values]:
        if t - last >= 4 * 3600_000 and t >= 1767225600000:
            out.append(t); last = t
    return out


def walk_short(w, entry, stop_fn, hold, ema=None):
    """SHORT on 1m bars along the candle path (falling candle O→H→L→C, rising O→L→H→C); P&L = (entry/price − 1) − fee."""
    w = w.loc[entry:entry + hold * MIN - MIN]
    if len(w) < 5:
        return None
    e = float(w.o.iloc[0]); peak = 0.0
    for t, o, h, l, c in zip(w.index, w.o.values, w.h.values, w.l.values, w.c.values):
        for px in ((o, h, l, c) if c < o else (o, l, h, c)):
            v = (e / px - 1) * 100 - V2.FEE
            stop = stop_fn(peak)
            if v <= stop:
                return stop, peak
            peak = max(peak, v)
        if ema is not None:   # EMA13-cross exit on the last CLOSED 5m bar (short: close above EMA13 with EMA5 > EMA8)
            k = ema.index.searchsorted(t - 300_000, side="right") - 1
            if k >= 0:
                r = ema.iloc[k]
                if c > r.e13 and r.e5 > r.e8:
                    return (e / c - 1) * 100 - V2.FEE, peak
    return (e / w.c.iloc[-1] - 1) * 100 - V2.FEE, peak


LADDER_SHORT = _TE.parse_hard_tp_ladder("1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80")   # live hard_tp_ladder_short


def bear_run_stop(atr):
    """The Bear-Run sleeve's exit = the live momentum-SHORT stack (BEARRUN_SHORT fills have no dedicated exit): stop −0.70
    widened to −1.5×ATR (cap −1.2) · runner arms at +0.40 and trails peak − min(0.5×ATR, 0.35×peak) (BE lock OFF for shorts; a
    negative trail floor is suppressed = rides on the stop) · hard-TP ladder floors (peak 1.0→0.75, 1.5→1.2, 2.0→1.6, 3.0→2.4,
    4.0→3.2). The EMA13-cross exit (price above EMA13 with EMA5 > EMA8) is applied in walk_short via `ema`."""
    sl = min(-0.70, max(-1.5 * atr, -1.2))
    def f(pk):
        if pk < 0.395:
            return sl
        trail = pk - min(0.5 * atr, 0.35 * pk)
        floors = [x for x in (trail if trail >= 0 else None, _TE.hard_tp_ladder_floor(LADDER_SHORT, pk)[0]) if x is not None]
        return max(floors) if floors else sl
    return f


def ema_frame(pair):
    d = V2.D.get(pair)
    if d is None:
        return None
    return pd.DataFrame({"e5": d.c.ewm(span=5, adjust=False).mean(), "e8": d.c.ewm(span=8, adjust=False).mean(),
                         "e13": d.c.ewm(span=13, adjust=False).mean()})


def exits(atr):
    br = lambda door: (lambda pk: _TE._bullrun_exit_for(float("inf"), pk, atr, door)[2])
    return {
        "BOT_SHORT":    ((lambda pk: -0.70 if pk < 0.45 else max(pk - 0.5 * atr, 0.10)), 240),
        "KEEP60":       ((lambda pk: -0.70 if pk < 0.40 else max(0.60 * pk, 0.10)), 60),
        "QUICK":        ((lambda pk: -0.50 if pk < 0.30 else max(0.50 * pk, 0.10)), 30),
        "BULLRUN_1ATR": (br("REARM"), 240),
        "BULLRUN_2ATR": (br("GREEN"), 240),
        "BEARRUN":      (bear_run_stop(atr), 240),   # operator: the Bear-Run sleeve's exit (= momentum-short stack + EMA13 cross)
    }


def own30(pair, t):
    d = V2.D.get(pair)
    if d is None or t not in d.index:
        return None
    i = d.index.get_loc(t)
    return (d.c.iloc[i] / d.c.iloc[i - 6] - 1) * 100 if i >= 6 else None


def fills(evs, control=False):
    rows = []
    for t0 in evs:
        tc = t0 - 288 * V2.BAR if control else t0
        start = tc + V2.BAR; b30 = V2.r30.get(tc)
        for p in V2.universe(tc, 20):
            atr = M.atr_at(p, tc); o30 = own30(p, tc)
            w = M.bars_1m(p, start, start + 265 * MIN)
            if w is None or len(w) < 30:
                continue
            sel = {"ALL": True, "HI_ATR": atr >= 1.5, "LEADER": (o30 is not None and b30 is not None and o30 < b30 * 100)}
            sel["HI_ATR_LEADER"] = sel["HI_ATR"] and sel["LEADER"]
            em = ema_frame(p)
            for off in (0, 10, 20):
                for ex, (fn, hold) in exits(atr).items():
                    res = walk_short(w, start + off * MIN, fn, hold, ema=em if ex == "BEARRUN" else None)
                    if res is None:
                        continue
                    for s, ok in sel.items():
                        if ok:
                            rows.append((t0, p, s, off, ex, res[0], res[1]))
    return pd.DataFrame(rows, columns=["ev", "pair", "sel", "off", "exit", "r", "peak"])


def table(evs, tag):
    F = fills(evs); C = fills(evs, control=True); out = []
    for (s, off, ex), g in F.groupby(["sel", "off", "exit"]):
        e = g.groupby("ev").r.mean(); lo, hi = V2.ci(e.values)
        a, b = e[e.index < SPLIT].mean(), e[e.index >= SPLIT].mean()
        c = C[(C.sel == s) & (C.off == off) & (C.exit == ex)].groupby("ev").r.mean()
        ca, cb = c[c.index < SPLIT].mean(), c[c.index >= SPLIT].mean()
        ok = len(e) >= 20 and a > 0 and b > 0 and a > ca and b > cb and lo > 0
        top3 = e.sort_values(ascending=False).head(3).sum() / e.sum() if e.sum() > 0 else np.nan
        out.append(dict(trigger=tag, sel=s, entry=off, exit=ex, events=len(e), fills=len(g), per_event=round(e.mean(), 3),
                        ci=f"[{lo:+.2f},{hi:+.2f}]", train=round(a, 3), test=round(b, 3), ctrl_train=round(ca, 3), ctrl_test=round(cb, 3),
                        wr=round((g.r > 0).mean() * 100), top3_share=round(top3, 2) if top3 == top3 else None, PASS="✅" if ok else ""))
    return pd.DataFrame(out).sort_values("per_event", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    res = []
    for tag, bd in (("DUMP", False), ("BREAKDOWN", True)):
        ev = events(bd)
        print(f"{tag}: {len(ev)} events (train {sum(t < SPLIT for t in ev)}, test {sum(t >= SPLIT for t in ev)})")
        T = table(ev, tag); res.append(T)
        print(T.head(15).to_string(index=False)); print("passing:", (T.PASS == "✅").sum(), "of", len(T), "\n")
    pd.concat(res).to_csv(os.path.join(V2.ROOT, "reports", "SURGE_SHORT_GRID_2026-09-30.csv"), index=False)
