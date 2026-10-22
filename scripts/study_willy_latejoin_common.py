"""WILLY late joiners (2026-10-22) — shared constants, 1m loader and the WILLY walker. READ-ONLY on the bot.
Pre-registration: reports/WILLY_LATE_JOIN_2026-10-22_PREREG.md.

Walker = scout_willy_timecap.evaluate_late(book="live") (the master / scout walker) on a 1m path built here from the study's own 1m cache
(reports/WILLY_LATE_JOIN_2026-10-22_data/k1m/<PAIR>/<YYYY-MM-DD>.npz: data.binance.vision daily zips, fapi for days not yet archived).
Exit levels are read from trading_config.json at run time.
"""
import json, os, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
import scout_willy_timecap as WT  # noqa: E402

D = os.path.join(R, "reports", "WILLY_LATE_JOIN_2026-10-22_data")
K1 = os.path.join(D, "k1m")
OUTP = os.path.join(R, "reports", "WILLY_LATE_JOIN_2026-10-22")
FB = os.path.join(R, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_data")
MIN, BAR, DAY = 60_000, 300_000, 86_400_000
DELAY_MS, SLIP_IN = 8_000, 0.10          # live ruler (FRENZY latency study): 8 s + ≥ 0.10 % slip
TH = json.load(open(os.path.join(R, "trading_config.json")))["thresholds"]
TP = float(TH["frenzy_willy_tp_pct"]); SL = abs(float(TH["frenzy_willy_stop_pct"])); CAP = int(TH["frenzy_willy_max_hold_minutes"])
LATE_M = float(TH["frenzy_willy_late_tp_minutes"]); LATE_P = float(TH["frenzy_willy_late_tp_pct"])
WAIT_BARS = int(round(float(TH["frenzy_willy_red_max_wait_minutes"]) / 5))       # 12
COLD = int(TH.get("frenzy_willy_a_cold_max_bars", 3))                           # 3
RCUT = float(TH["frenzy_willy_max_vol_mcap_ratio"])                              # 1.0
HT_FRAC = float(TH.get("frenzy_willy_ht_size_frac", 0.25))
BOOK = "live" if TH.get("resting_tp_willy_enabled") else "print"
RULE = f"+{TP:g}/-{SL:g}/{CAP}/late{LATE_M:g}@+{LATE_P:g}/{BOOK}"
T_DEPLOY = pd.Timestamp("2026-10-08 15:23")      # WILLY deploy (+10 min scout convention)
T_LIVE = pd.Timestamp("2026-10-14 18:36:30")     # go-live
os.makedirs(K1, exist_ok=True)


def supply():
    out = {}
    for f in (os.path.join(R, "reports", "cache_vol_mcap", "bapi_supply.json"), os.path.join(FB, "winloss_supply.json")):
        if os.path.exists(f):
            for k, v in json.load(open(f)).items():
                cs = v.get("cs") if isinstance(v, dict) else None
                if cs and cs > 0 and k not in out:
                    out[k] = float(cs) * float(v.get("mult") or 1)
    return out


def era(ts_ms):
    t = pd.Timestamp(int(ts_ms), unit="ms")
    if t < pd.Timestamp("2026-01-01"):
        return "2025"
    return "2026-H1" if t < pd.Timestamp("2026-07-01") else "2026-H2"


def block(ts_ms):
    t = pd.Timestamp(int(ts_ms), unit="ms")
    return "3_backtest" if t < T_DEPLOY else ("2_master" if t < T_LIVE else "1_live")


def k1_day(pair, day):
    f = os.path.join(K1, pair, f"{day}.npz")
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        return z["a"]          # columns: open_time, o, h, l, c


def path_1m(pair, te, E):
    """→ the evaluate_late path dict on 1m bars strictly after the entry minute, up to te + CAP + 1 min; gaps truncate the path."""
    lim = te + (CAP + 1) * MIN
    days = sorted({pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"), pd.Timestamp(lim, unit="ms").strftime("%Y-%m-%d")})
    parts = [k1_day(pair, d) for d in days]
    parts = [p for p in parts if p is not None and len(p)]
    first = -(-int(te) // MIN) * MIN
    if parts:
        a = np.concatenate(parts)
        a = a[np.argsort(a[:, 0], kind="stable")]
        a = a[(a[:, 0] >= first) & (a[:, 0] + MIN <= lim)]
    else:
        a = np.zeros((0, 5))
    if len(a):
        o = a[:, 0].astype("int64")
        if o[0] != first:
            a = a[:0]
        else:
            g = np.flatnonzero(np.diff(o) != MIN)
            if len(g):
                a = a[:g[0] + 1]
    o = a[:, 0].astype("int64") if len(a) else np.array([], dtype="int64")
    net = lambda x: WT.net_pct(x, E)  # noqa: E731
    return dict(t=o + MIN, hi=net(a[:, 2]) if len(a) else np.array([]), lo=net(a[:, 3]) if len(a) else np.array([]),
                cl=net(a[:, 4]) if len(a) else np.array([]), op=net(a[:, 1]) if len(a) else np.array([]),
                bar=np.ones(len(a), bool), end_ms=int(o[-1] + MIN) if len(a) else int(te), n_tick=0, n_bar=int(len(a)), te=int(te))


def entry_px(pair, red_close_ms):
    """the 1m bar opening at the red close → (E, te) with the 0.10 % slip; None when the minute is missing."""
    d = pd.Timestamp(red_close_ms, unit="ms").strftime("%Y-%m-%d")
    a = k1_day(pair, d)
    if a is None or not len(a):
        return None, None
    m = a[a[:, 0] == red_close_ms]
    if not len(m) or not np.isfinite(m[0, 1]) or m[0, 1] <= 0:
        return None, None
    return float(m[0, 1]) * (1 + SLIP_IN / 100), int(red_close_ms + DELAY_MS)


def walk(pair, red_close_ms):
    E, te = entry_px(pair, red_close_ms)
    if E is None:
        return dict(why="NOENTRY", pct=np.nan)
    P = path_1m(pair, te, E)
    if not len(P["t"]):
        return dict(why="UNRES", pct=np.nan, E=E)
    ev = WT.evaluate_late(P, TP, SL, CAP, LATE_M, LATE_P, book=BOOK)
    return dict(why=ev["why"], pct=float(ev["pct"]) if ev["why"] != "UNRES" else np.nan, xmin=ev["xmin"], worst=ev.get("worst"), E=E,
                peak=float(np.max(P["hi"][: int(np.searchsorted(P["t"], te + CAP * MIN, side="right"))])) if len(P["hi"]) else np.nan)


def day_boot(x, days, n=5000, seed=7):
    """day-clustered bootstrap of the mean → (lo, hi) 95 %."""
    x = np.asarray(x, float); days = np.asarray(days)
    if len(x) < 2:
        return np.nan, np.nan
    u = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in u}
    s = np.array([x[idx[d]].sum() for d in u]); c = np.array([len(idx[d]) for d in u])
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(u), size=(n, len(u)))
    m = s[k].sum(1) / c[k].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
