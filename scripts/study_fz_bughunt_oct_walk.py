#!/usr/bin/env python3
"""🐞 Oct-10 FRENZY bug-hunt — walk the backtest-logic Oct signal set (study_fz_bughunt_oct_cohort.py) with TODAY's exit and match it to live.

Entry = the backtest ruler: signal close + 8 s, price × 1.0010 (here: the 1m bar holding sig + 8 s — its OPEN, × 1.0010; the tick ruler is
used instead wherever the shared tick cache / scratch API windows cover the path). Exit fixed +3 / −3 NET (net = gross − 0.045 − 0.045·p/E),
12 h cap; on 1m bars a bar touching both lines = stop (conservative, the tick walker's tie rule). Bearish-day block = services.frenzy
semantics on BTC 5m (EMA13/EMA50 gap on the last CLOSED bar, the last closed UTC daily return) rebuilt from k5m_full + the Oct fetch and
checked against the live stamps. Market-volume gate OFF (today). Slots 2 per sleeve · pair-day cap 3 · LITE refused while the pair holds any
position · LONG/WIDE refused while the pair holds a LITE. 1m klines: fapi, sequential, weight 5/call, X-MBX-USED-WEIGHT-1M pause at ≥ 900,
418/429 → stop + shared cooldown; cached in scratch.
Usage: venv/bin/python scripts/study_fz_bughunt_oct_walk.py"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd
from urllib.parse import quote
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_tp34_common as C                                    # noqa: E402
SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
BH = os.path.join(SCR, "bh"); K1 = os.path.join(BH, "k1m"); os.makedirs(K1, exist_ok=True)
COOL = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")
NOW = int(time.time() * 1000)


def get(url):
    try:
        if _BC.active(COOL):
            print("cooldown live — stop"); sys.exit(1)
    except (FileNotFoundError, ValueError):
        pass
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0); x = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            ra = int(e.headers.get("Retry-After") or 120)
            open(COOL, "w").write(str(int(time.time() * 1000 + ra * 1000))); print("STOP", e.code); sys.exit(2)
        raise
    time.sleep(60 if w >= 900 else 0.4)
    return x


def k1m(pair, sig):
    fp = os.path.join(K1, f"{pair}_{sig}.json")
    end = min(sig + C.HOLD, NOW - 60_000)
    if os.path.exists(fp):
        x = json.load(open(fp))
        if x and (x[-1][0] >= end - 120_000 or len(x) >= 720):
            return x
    x = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(pair)}&interval=1m&startTime={sig}&endTime={sig + C.HOLD}&limit=1000")
    x = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in x]
    json.dump(x, open(fp, "w"))
    return x


def walk_1m(bars, E, t0):
    """first bar whose low crosses −3 (stop first on a both-lines bar) / high crosses +3; CAP at the last close ≤ t0 + 12 h."""
    for b in bars:
        if b[0] + 60_000 <= t0:
            continue
        lo = (b[3] / E - 1) * 100 - 0.045 - 0.045 * b[3] / E; hi = (b[2] / E - 1) * 100 - 0.045 - 0.045 * b[2] / E
        if lo <= -3:
            return -3.0 - 0.0, "SL", b[0]
        if hi >= 3:
            return 3.0, "TP", b[0]
    if bars and bars[-1][0] + 60_000 >= t0 + C.HOLD - 60_000:
        c = bars[-1][4]; return (c / E - 1) * 100 - 0.045 - 0.045 * c / E, "CAP", bars[-1][0]
    return np.nan, "OPEN", None


def btc_frame():
    d = pd.read_csv(os.path.join(ROOT, "reports", "backtest_cache", "k5m_full", "BTCUSDT.csv"))
    fp = os.path.join(BH, "k5m_oct", "BTCUSDT.json")
    if os.path.exists(fp):
        d = pd.concat([d, pd.DataFrame(json.load(open(fp)), columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])])
    d = d.astype({"open_time": "int64"}).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True); c = d.c.astype(float)
    d["gap"] = (c.ewm(span=13, adjust=False).mean() / c.ewm(span=50, adjust=False).mean() - 1) * 100
    d["day"] = pd.to_datetime(d.open_time, unit="ms").dt.floor("D"); dc = d.groupby("day").c.last().astype(float)
    return d, (dc / dc.shift(1) - 1) * 100


def bearish_at(sig, B):
    d, dret = B
    i = int(np.searchsorted(d.open_time.values, sig - 300_000, side="right")) - 1        # last CLOSED 5m bar at the signal close
    g = float(d.gap.iloc[i]); r = dret.get(pd.Timestamp(sig, unit="ms").floor("D") - pd.Timedelta(days=1), np.nan)
    return bool(g < 0 and r < 0), r, g


def main():
    X = pd.read_pickle(os.path.join(BH, "oct_cohort.pkl"))
    X = X[X.kind.isin(["FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"]) & ~X.blacklisted].copy()
    X["sig"] = X.t_signal_close.astype("int64"); B = btc_frame()
    bb = [bearish_at(s, B) for s in X.sig]; X["bear"] = [x[0] for x in bb]; X["btc_1d"] = [x[1] for x in bb]; X["btc_gap"] = [x[2] for x in bb]
    rows = []
    for r in X.sort_values("sig").itertuples():
        d = dict(r._asdict())
        tk = C.walk_entry(r.pair, r.sig, 8_000, 0.10, which=["fix3"])
        if tk is not None and tk["cap_complete"] or (tk is not None and tk["fix3"][1] in ("TP", "SL")):
            d.update(fix3=tk["fix3"][0], how=tk["fix3"][1], xms=tk["fix3"][2], E=tk["E"], src="ticks")
        else:
            bars = k1m(r.pair, r.sig)
            b8 = [b for b in bars if b[0] <= r.sig + 8000 < b[0] + 60_000]
            if not b8:
                d.update(src="no_data"); rows.append(d); continue
            E = b8[0][1] * 1.001
            v, how, x = walk_1m(bars, E, r.sig + 8000)
            d.update(fix3=v, how=how, xms=x, E=E, src="1m")
        rows.append(d)
    W = pd.DataFrame(rows)
    # portfolio rules (today): slots 2 per sleeve, pair-day cap 3, LITE needs the pair flat, LONG/WIDE refused if the pair holds a LITE
    W = W.sort_values("sig").reset_index(drop=True); openp = []; day = {}; st = []
    for r in W.itertuples():
        t = r.sig + 8000
        openp = [o for o in openp if (o[2] is None or o[2] > t)]
        why = None
        if r.bear:
            why = "BEARISH_DAY"
        elif not r.seen:
            why = "not shortlisted"
        elif r.kind == "FRENZY_LITE" and any(o[0] == r.pair for o in openp):
            why = "PAIR_HELD"
        elif r.kind != "FRENZY_LITE" and any(o[0] == r.pair and o[1] == "FRENZY_LITE" for o in openp):
            why = "PAIR_HELD_BY_LITE"
        elif sum(o[1] == r.kind for o in openp) >= 2:
            why = "MAX_SLOTS"
        elif day.get((r.pair, r.kind, pd.Timestamp(r.sig, unit="ms").date()), 0) >= 3:
            why = "PAIR_DAY_CAP"
        st.append(why or "kept")
        if not why and r.src != "no_data":
            openp.append((r.pair, r.kind, (r.xms if pd.notna(r.xms) else None)))
            k = (r.pair, r.kind, pd.Timestamp(r.sig, unit="ms").date()); day[k] = day.get(k, 0) + 1
    W["stack"] = st
    W.to_pickle(os.path.join(BH, "oct_walk.pkl"))
    W.drop(columns=["Index"], errors="ignore").to_csv(os.path.join(ROOT, "reports", "FRENZY_BACKTEST_BUGHUNT_2026-10-10_oct_signals.csv"), index=False)
    print(W.groupby(["kind", "stack"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
