#!/usr/bin/env python3
"""⚡ SURGE sleeve design (operator, 2026-09-30: "something like BULLRUN, but short, quick-term, triggered by the BTC spike").
Candidate design space, walked on 1m bars with the candle-path simulator validated on the operator's real spike trades
(scripts/btc_spike_follow_1m.py). Trigger = the pre-registered BTC spike (BTC +1.0 %/30 min ∧ 24 h-high breakout ∧ volume ≥ 3×
median). Universe = top-20 tradeable alts by 24 h volume at the event.

Grid (declared before running — 4 selections × 3 entry delays × 6 exits = 72 cells; the two BULLRUN exits added at the operator's request):
  SELECTION  ALL · HI_ATR (pair 5m ATR ≥ 1.5 %) · LEADER (own 30-min return > BTC's) · HI_ATR_LEADER (both)
  ENTRY      0 / 10 / 20 min after the event bar closes
  EXIT       BOT      the live momentum exit (−0.70 stop; arm +0.40; floor max(peak − 1×ATR, +0.10); 4 h)
             KEEP60   −0.70 stop; arm +0.40; floor max(0.60 × peak, +0.10); 60 min max hold
             TP60     fixed +0.60 take-profit / −0.70 stop; 60 min max hold
             QUICK    −0.50 stop; arm +0.30; floor max(0.50 × peak, +0.10); 30 min max hold
SHIP BAR for a candidate (all must hold; one-sided, no re-fitting afterwards):
  ① per-event mean > 0 in BOTH halves (Jan–Apr train, May–Sep test) · ② beats the SAME cell on the control (same universe, same
  clock time 1 day earlier) in both halves · ③ 95 % bootstrap CI of the per-event mean (all events) above 0 · ④ ≥ 20 events.
48 cells ⇒ ~2–3 pass ① by luck alone; ③ is the real filter. A pass is a CANDIDATE for an observe-first probe, not a ship.
Usage: venv/bin/python scripts/surge_sleeve_design.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_1m as M
V2 = M.V2
MIN = M.MIN
SPLIT = 1777593600000   # 2026-05-01

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:"); sys.path.insert(0, V2.ROOT)
import services.trading_engine as _TE   # the LIVE Bull-Run exit rule, called as-is (no re-implementation)


def _br_stop(atr, door):
    """Stop line of the live Bull-Run exit for a given peak: SL = min(−0.70, −1.5×ATR capped −1.2) until the +1.0 arm, then
    max(+0.2 lock, peak − trail×ATR, ladder rung) — trail 2×ATR (GREEN door) or 1×ATR (REARM door)."""
    return lambda pk: _TE._bullrun_exit_for(float("inf"), pk, atr, door)[2]


EXITS = {
    "BOT":    (None, 240),
    "KEEP60": (lambda pk: -0.70 if pk < 0.40 else max(0.60 * pk, 0.10), 60),
    "TP60":   ("tp", 60),
    "QUICK":  (lambda pk: -0.50 if pk < 0.30 else max(0.50 * pk, 0.10), 30),
    "BULLRUN_2ATR": ("br:GREEN", 240),   # operator: "use the Bull-Run exit" — the live sleeve's exit, GREEN door trail
    "BULLRUN_1ATR": ("br:REARM", 240),   # same exit, REARM door trail (1×ATR)
}


def walk_tp(w, entry, tp=0.60, sl=-0.70, hold=60):
    w = w.loc[entry:entry + hold * MIN - MIN]
    if len(w) < 5:
        return None
    e = float(w.o.iloc[0]); peak = 0.0
    for o, h, l, c in zip(w.o.values, w.h.values, w.l.values, w.c.values):
        for px in ((o, h, l, c) if c < o else (o, l, h, c)):
            v = (px / e - 1) * 100 - V2.FEE
            if v <= sl:
                return sl, peak
            if v >= tp:
                return tp, max(peak, v)
            peak = max(peak, v)
    return (w.c.iloc[-1] / e - 1) * 100 - V2.FEE, peak


def own30(pair, t):
    d = V2.D.get(pair)
    if d is None or t not in d.index:
        return None
    i = d.index.get_loc(t)
    return (d.c.iloc[i] / d.c.iloc[i - 6] - 1) * 100 if i >= 6 else None


def fills(events, control=False):
    rows = []
    for t0 in events:
        tc = t0 - 288 * V2.BAR if control else t0
        start = tc + V2.BAR
        b30 = V2.r30.get(tc)
        for p in V2.universe(tc, 20):
            atr = M.atr_at(p, tc); o30 = own30(p, tc)
            w = M.bars_1m(p, start, start + (20 + 245) * MIN)
            if w is None or len(w) < 30:
                continue
            sel = {"ALL": True, "HI_ATR": atr >= 1.5,
                   "LEADER": (o30 is not None and b30 is not None and o30 > b30 * 100)}
            sel["HI_ATR_LEADER"] = sel["HI_ATR"] and sel["LEADER"]
            for off in (0, 10, 20):
                for ex, (fn, hold) in EXITS.items():
                    if fn == "tp":
                        res = walk_tp(w, start + off * MIN)
                    elif isinstance(fn, str) and fn.startswith("br:"):
                        res = M.walk(w, start + off * MIN, atr, hold_min=hold, exit_fn=_br_stop(atr, fn[3:]))
                    else:
                        res = M.walk(w, start + off * MIN, atr, hold_min=hold, exit_fn=fn)
                    if res is None:
                        continue
                    for s, ok in sel.items():
                        if ok:
                            rows.append((t0, p, s, off, ex, res[0], res[1]))
    return pd.DataFrame(rows, columns=["ev", "pair", "sel", "off", "exit", "r", "peak"])


def stats(R):
    e = R.groupby("ev").r.mean()
    a, b = e[e.index < SPLIT], e[e.index >= SPLIT]
    lo, hi = V2.ci(e.values)
    return dict(events=len(e), fills=len(R), mean=e.mean(), lo=lo, hi=hi, train=a.mean(), test=b.mean(), n_tr=len(a), n_te=len(b),
                pos=(e > 0).mean() * 100, wr=(R.r > 0).mean() * 100)


if __name__ == "__main__":
    P = V2.events(1.0, True, True)
    F = fills(P); C = fills(P, control=True)
    out = []
    for (s, off, ex), g in F.groupby(["sel", "off", "exit"]):
        st = stats(g); c = C[(C.sel == s) & (C.off == off) & (C.exit == ex)]
        cs = stats(c) if len(c) else dict(train=np.nan, test=np.nan, mean=np.nan)
        passed = (st["events"] >= 20 and st["train"] > 0 and st["test"] > 0 and st["train"] > cs["train"] and st["test"] > cs["test"]
                  and st["lo"] > 0)
        out.append(dict(sel=s, entry=off, exit=ex, events=st["events"], fills=st["fills"], per_event=round(st["mean"], 3),
                        ci=f"[{st['lo']:+.2f},{st['hi']:+.2f}]", train=round(st["train"], 3), test=round(st["test"], 3),
                        ctrl_train=round(cs["train"], 3), ctrl_test=round(cs["test"], 3), pos_events=round(st["pos"]), fill_wr=round(st["wr"]),
                        PASS="✅" if passed else ""))
    T = pd.DataFrame(out).sort_values("per_event", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(f"events: {len(P)} (train Jan–Apr {sum(t < SPLIT for t in P)}, test May–Sep {sum(t >= SPLIT for t in P)}) · top-20 universe · 1m walk\n")
    print(T.to_string(index=False))
    T.to_csv(os.path.join(V2.ROOT, "reports", "SURGE_SLEEVE_GRID_2026-09-30.csv"), index=False)
    F.to_csv(os.path.join(V2.ROOT, "reports", "backtest_cache", "surge_grid_fills.csv"), index=False)
