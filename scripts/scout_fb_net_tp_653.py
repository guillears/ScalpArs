#!/usr/bin/env python3
"""💸 Scout — FLATBASE net take profit revert gate (DECISION_LOG 653, 2026-11-27, operator "yes build the net take profit"; ALERT-only).

RULE (engine)   flatbase_tp_counts_funding: the take profit moves out by the funding PAID (resting limit at fill × (1 − (tp + paid) %), re-priced
                by the resting-TP sweep after each settlement; received funding never moves it). Stop: +10 % counting funding paid (DL 607).
EVIDENCE        backtest 545 +1.155 → +1.224 %/fill (156 better / 5 worse) · real kept fills 6 changed, all better (DL 653).
COHORT          closed FLATBASE_SHORT fills opened ≥ the DL 653 commit + 10 min (orders exports), dedup (opened_at, pair).
PRICING         each fill re-walked on its own 5m path from its fill price with REAL Binance funding settlements, under BOTH exits with the
                DL 616 walker (scripts/FUNDING_STUDY_ALL_2026-11-21_bt.walk_all): base = the old PRICE TP (−5 %) · NTP = the net TP (today).
                Δ = NTP − base per fill. "touched the price TP then stopped" = base TP and NTP not TP (STOP or TIME). A fill is priced only
                once its full 72 h path is published (else "path not ready" — the counted fills stay a prefix).
GATE (frozen)   at the first 20 priced fills: Σ Δ < 0, or ≥ 3 touched-then-stopped fills → REVERT (tight: the expected Σ over 20 is
                ≈ +1.4 pts, so ONE TP → STOP (≈ −14) flips it) (flag: switch flatbase_tp_counts_funding off —
                operator decides) · else KEEP. State reports/SCOUT_FB_NET_TP_653.json (gitignored).
Usage: venv/bin/python scripts/scout_fb_net_tp_653.py"""
import importlib.util
import json
import os
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

DL = 653
GATE_N, MAX_TOUCH_STOP = 20, 3
STATE = os.path.join(_R, "reports", "SCOUT_FB_NET_TP_653.json")
HR, BAR = 3_600_000, 300_000


_MODS = {}


def _imp(name, fn):
    """load a study module ONCE (cached) without leaking its sys.argv rewrite into the scout process (review)."""
    if name in _MODS:
        return _MODS[name]
    argv = list(sys.argv)
    try:
        sp = importlib.util.spec_from_file_location(name, os.path.join(_D, fn)); m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
    finally:
        sys.argv[:] = argv
    _MODS[name] = m
    return m


def gate(rows):
    """rows = [(delta, touched_then_stopped)] in open-time order → (verdict, text, n). Pure."""
    if len(rows) < GATE_N:
        s = sum(d for d, _ in rows)
        return "COLLECTING", f"⏳ {len(rows)}/{GATE_N} priced · Σ Δ so far {s:+.2f} pts · touched-then-stopped {sum(1 for _, t in rows if t)}", None
    cut = rows[:GATE_N]
    s = sum(d for d, _ in cut); ts = sum(1 for _, t in cut if t)
    info = f"first {GATE_N} Flatbase fills: Σ (net TP − price TP) {s:+.2f} pts · touched the price TP then stopped {ts}"
    if s < 0 or ts >= MAX_TOUCH_STOP:
        return "REVERT", f"🔴 {info} → flag: switch flatbase_tp_counts_funding off (operator decides)", GATE_N
    return "KEEP", f"🟢 {info} → keep the net take profit", GATE_N


_STOP = {"why": None}   # (656 review) set on a 418 / 429 / 403 — every later walk_fill of the run raises without fetching
_PATHS = {}   # (pair, first bar ms) → (t, o, h, l, c, funding times, funding rates) — one fetch per path per run, shared by every caller
FUND_CACHE = os.path.join(_R, "reports", "backtest_cache", "fb_net_walk")   # closed windows' funding (gitignored cache dir)


def _funding(RR, pair, a, b):
    """the funding settlements in [a, b] — from the disk cache when this closed window was read before, else fetched and cached (a 72 h
    window that has ended never changes). A fetch error raises (never cached)."""
    import numpy as np
    p = os.path.join(FUND_CACHE, f"{pair}_{int(a)}_{int(b)}.json")
    try:
        with open(p) as f:
            d = json.load(f)
        return np.array(d["t"], dtype="int64"), np.array(d["r"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    ft, fr = RR.fetch_funding(pair, int(a), int(b))
    try:
        os.makedirs(FUND_CACHE, exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(dict(t=[int(x) for x in ft], r=[float(x) for x in fr]), f)
        os.replace(tmp, p)
    except OSError:
        pass
    return ft, fr


def walk_arrays(t, o, h, l, c, i0, E, t_fill, fs_t, fs_r, tp, sl, end_t):
    """🔁 (DECISION_LOG 656) the ONE pricer of today's FLATBASE exit for every Flatbase observe line: a short at E from bar i0 on 5m arrays,
    with the REAL Binance funding settlements after t_fill, walked by the DL 616 walker (FUNDING_STUDY_ALL_2026-11-21_bt.walk_all,
    paid_stop True): stop first on a bar touching both, a resting TP needs a 0.05 % trade-through, 0.10 % slip on taker exits, entry 0.05,
    taker 0.05, maker 0.02, 72 h. → the walk dict or None (no bars from i0). Keys: NTP_* = TODAY (net TP 653 + funding stop 607) ·
    NTP_PSL_* = net TP + price-only stop · base_* = price TP + funding stop · PSL_* = price TP + price-only stop (net, why, hold). PURE."""
    import numpy as np
    BT = _imp("fsb", "FUNDING_STUDY_ALL_2026-11-21_bt.py")
    if not len(t) or i0 >= len(t) or not (E and E > 0):
        return None
    F = BT.C.settle_on_bars(t, o, E, t_fill, fs_t, fs_r, 5, markscale=True)
    return BT.walk_all(t, o, h, l, c, i0, E, float(tp), float(sl), min(int(end_t), int(t[-1]) + BAR), F, 0.10, 0.05, 0.05, 0.02, True, time_cells=False)


def walk_fill(pair, te, E=None, tp=5.0, sl=10.0, now_ms=None, entry="fill"):
    """🔁 (656) one Flatbase short priced under today's exit machinery from public Binance 5m klines + funding (fetched once per path).
    entry="fill": a REAL fill at te (ms) and price E — the path starts at the bar AFTER te's bar (its pre-fill high / low never fire).
    entry="open": a shadow short at the 5m open at te (on the grid; a refused setup) — E = that open, the path starts AT that bar.
    → walk dict (walk_arrays) · None (the 72 h path is not published yet) · "NO_DATA" (no bars). A fetch error RAISES (RuntimeError —
    the caller holds the read: transient, never dropped)."""
    import numpy as np
    now = (int(time.time() * 1000) if now_ms is None else int(now_ms)) // BAR * BAR - BAR
    te = int(te)
    b0 = te // BAR * BAR if entry == "open" else te // BAR * BAR + BAR
    if now < b0 + 72 * HR:
        return None                                   # both walks need the full 72 h path — a younger fill waits
    key = (str(pair), b0)
    need_end = (te if entry == "fill" else b0) + 72 * HR          # the walk's last bar must exist (a truncated path is never priced)
    if key not in _PATHS:
        if _STOP["why"]:                                          # (656 review) after a 418 / 429 nothing more is fetched this run
            raise RuntimeError(f"Binance fetch stopped earlier this run ({_STOP['why']})")
        RR = _imp("fbfr", "FLATBASE_FUNDING_RERUN_2026-11-27.py")
        try:
            try:
                t, o, h, l, c, _ = RR.ET.real_arrays(str(pair), te - 2 * HR, b0 + 73 * HR)
            except (IndexError, KeyError, ValueError):            # (656 review) no rows at all (delisted / invalid symbol): arrays_from on
                t = np.array([], "int64"); o = h = l = c = np.array([])   # an empty frame raises → NO_DATA, never a permanent hold
            if len(t) and int(t[-1]) < need_end - BAR:            # (656 review) the study cache skips a refetch at ≥ 98 % coverage → the tail
                ex = RR.ET.fetch(str(pair), int(t[-1]) + BAR, b0 + 73 * HR)   # may be missing: fetch it, never price a cut path as TIME
                if len(ex):
                    import pandas as pd
                    base = pd.DataFrame(dict(open_time=t, o=o, h=h, l=l, c=c, vol=np.zeros(len(t))))
                    t, o, h, l, c, _ = RR.ET.arrays_from(pd.concat([base, ex[["open_time", "o", "h", "l", "c", "vol"]]]))
            fs_t, fs_r = _funding(RR, str(pair), te - 3 * 86_400_000, b0 + 73 * HR) if len(t) else (np.array([], "int64"), np.array([]))
        except SystemExit as e:                       # the study fetcher stops on 418 / 429 with SystemExit — never kill the scout
            _STOP["why"] = str(e)[:80]
            raise RuntimeError(f"Binance fetch stopped ({e})")
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (418, 429, 403):               # (656 review) fundingRate's urllib HTTPError: stop every fetch for the rest of the run
                _STOP["why"] = f"HTTP {code} on fundingRate"
            raise RuntimeError(f"fetch failed ({type(e).__name__}: {str(e)[:60]})")
        _PATHS[key] = (np.asarray(t, "int64"), o, h, l, c, fs_t, fs_r)
    t, o, h, l, c, fs_t, fs_r = _PATHS[key]
    i0 = int(np.searchsorted(t, b0))
    if not len(t) or i0 >= len(t) or (entry == "open" and int(t[i0]) != b0):   # a shadow needs its entry bar's open; a fill only a path
        return "NO_DATA"
    E = float(o[i0]) if entry == "open" else float(E)
    if int(t[-1]) < need_end - BAR:
        if now < need_end + 24 * HR:
            return None                               # the tail is not published / fetched yet — wait (a later run re-reads it)
        return "NO_DATA"                              # a day later and still cut short → the data does not exist (out, never TIME)
    w = walk_arrays(t, o, h, l, c, i0, E, te + 10_000, fs_t, fs_r, tp, sl, end_t=need_end)
    return "NO_DATA" if w is None else w


def price_fill(pair, te, E):
    """→ (base_net, ntp_net, base_why, ntp_why) for one closed fill, or None (data not ready / unreadable)."""
    w = walk_fill(pair, te, E)
    if not isinstance(w, dict):
        return None
    return w["base_net"], w["NTP_net"], w["base_why"], w["NTP_why"]


def report(frame=None, state_path=STATE, floor_ms="auto", pricer=None):
    import pandas as pd
    out = [f"## 💸 FLATBASE net take profit (DECISION_LOG {DL}) — revert gate (alert only)",
           "net TP = the take profit moves out by the funding paid · evidence: backtest +1.155 → +1.224 %/fill (156 better / 5 worse)"]
    if floor_ms == "auto":
        import scout_gatekit as GK
        floor_ms = GK.dl_deploy_ms(DL, lag_min=10, label=f"FLATBASE net TP (DECISION_LOG {DL})")
    if floor_ms is None:
        out.append(f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.")
        return "\n".join(out)
    if frame is None:
        import export_glob as EG
        fs = sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")), key=os.path.getmtime)
        parts = []
        for f in fs:
            try:
                x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "direction", "status", "entry_strategy", "entry_price"))
                parts.append(x[(x.entry_strategy == "FLATBASE_SHORT") & (x.status == "CLOSED")])
            except Exception:
                continue
        frame = pd.concat(parts).drop_duplicates(["opened_at", "pair", "direction"], keep="last") if parts else pd.DataFrame()
    pricer = pricer or price_fill
    rows, lines = [], []
    if len(frame):
        frame = frame.assign(_t=[int(pd.Timestamp(str(v)[:19]).value // 10**6) for v in frame.opened_at])
        for r in frame[frame._t >= floor_ms].sort_values("_t").itertuples():
            try:
                p = pricer(r.pair, r._t // BAR * BAR, float(r.entry_price))
            except Exception as e:
                lines.append(f"  ⏸ {r.pair} {str(r.opened_at)[:16]}: not priced ({type(e).__name__}) — later fills wait")
                break
            if p is None:
                lines.append(f"  ⏸ {r.pair} {str(r.opened_at)[:16]}: path not ready — later fills wait")
                break
            touched = p[2] == "TP" and p[3] != "TP"            # (review) the price TP hit, the net TP did not (STOP or TIME)
            rows.append((p[1] - p[0], touched))
            lines.append(f"  {r.pair} {str(r.opened_at)[:16]}: price TP {p[0]:+.2f} ({p[2]}) → net TP {p[1]:+.2f} ({p[3]}) · Δ {p[1] - p[0]:+.2f}"
                         + (" · ⚠ touched the price TP then stopped" if touched else ""))
    v, txt, n = gate(rows)
    st = None
    if state_path:
        try:
            st = json.load(open(state_path))
        except FileNotFoundError:
            st = {}
        except Exception:
            st = None
        if st is not None and "verdict" not in st and n is not None:
            st = dict(verdict=v, detail=txt, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
            try:
                json.dump(st, open(state_path, "w"))
            except Exception:
                st = {}
                out.append("  ⚠ state not saved — not frozen, re-read next run")
    if st and st.get("verdict"):
        out.append(f"FROZEN ({st.get('run_at')}): {st['verdict']} — {st.get('detail')}")
    else:
        out.append(f"now: {v} — {txt}")
    out.extend(lines[-25:])
    return "\n".join(out)


def run():
    return [report(), ""]


if __name__ == "__main__":
    print(report())
