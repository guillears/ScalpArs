"""WILLY high-target study (2026-10-23) — shared: spike high H, variant targets, the walker on one path. READ-ONLY on the bot.
Pre-registration: reports/WILLY_HIGHTARGET_2026-10-23_PREREG.md.

Walker = scout_willy_timecap.evaluate_late(book="live") — the master / scout walker; net % = gross − 0.045 − 0.045 × exit/entry.
"""
import json, os, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
import scout_willy_timecap as WT  # noqa: E402

MIN, BAR = 60_000, 300_000
TH = json.load(open(os.path.join(R, "trading_config.json")))["thresholds"]
TP = float(TH["frenzy_willy_tp_pct"]); SL = abs(float(TH["frenzy_willy_stop_pct"])); CAP = int(TH["frenzy_willy_max_hold_minutes"])
LATE_M = float(TH["frenzy_willy_late_tp_minutes"]); LATE_P = float(TH["frenzy_willy_late_tp_pct"])
TPMAX = 8.0
HORIZON = 181                      # minutes of path needed (H4 cap 180)
KS = (0.5, 1.0, 2.0)
VARIANTS = ["TODAY"] + [f"H1_k{k:g}" for k in KS] + [f"H2_k{k:g}" for k in KS] + ["H3", "H4", "D_H1k1_noLate", "F100", "F075", "F050"]
SHIP = [v for v in VARIANTS if v not in ("TODAY", "D_H1k1_noLate", "F100", "F075", "F050")]
FIXED = {"F100": 1.0, "F075": 0.75, "F050": 0.5}   # Addendum 2: lower fixed targets (BTC-falling side)


def tgt_raw(H, E, k):
    """net % target of a TP resting at H × (1 − k %), on the walker's accounting."""
    return float(WT.net_pct(H * (1 - k / 100.0), E))


def _ev(P, tp, cap, late=True):
    """one walk: TP line tp, stop −SL, cap minutes; late TP +LATE_P from LATE_M unless off / not below tp."""
    if late and LATE_P < tp:
        return WT.evaluate_late(P, tp, SL, cap, LATE_M, LATE_P, book="live")
    return WT.evaluate_late(P, tp, SL, cap, cap + 1.0, tp, book="live")


def walk_all(P, H, E):
    """→ dict variant → (pct, why, target). Unresolved → NaN."""
    out = {}
    def rec(name, ev, tg):
        out[name] = (float(ev["pct"]) if ev["why"] != "UNRES" else np.nan, ev["why"], tg)
    t0 = _ev(P, TP, CAP); rec("TODAY", t0, TP)
    for k in KS:
        r = tgt_raw(H, E, k)
        g1 = min(max(r, TP), TPMAX)
        rec(f"H1_k{k:g}", t0 if g1 == TP else _ev(P, g1, CAP), g1)
        g2 = min(max(r, LATE_P), TPMAX)
        rec(f"H2_k{k:g}", t0 if g2 == TP else _ev(P, g2, CAP), g2)
    g = out["H1_k1"][2]
    eB = t0 if g == TP else _ev(P, g, CAP)
    pa, pb = out["TODAY"][0], (float(eB["pct"]) if eB["why"] != "UNRES" else np.nan)
    out["H3"] = (0.5 * pa + 0.5 * pb, f"{t0['why']}|{eB['why']}", g)
    rec("H4", t0 if g == TP else _ev(P, g, 180), g)
    rec("D_H1k1_noLate", t0 if g == TP else _ev(P, g, CAP, late=False), g)
    for name, tp in FIXED.items():
        rec(name, _ev(P, tp, CAP), tp)
    return out


def spike_high_ticks(pair, spike_close_ms, red_close_ms, tcache=None):
    """H from tick prints in [spike bar open, red close) — None unless every needed UTC day has a full tick file."""
    a, b = int(spike_close_ms) - BAR, int(red_close_ms)
    days = sorted({pd.Timestamp(a, unit="ms").strftime("%Y-%m-%d"), pd.Timestamp(b - 1, unit="ms").strftime("%Y-%m-%d")})
    hs = []
    for d in days:
        z = WT._load_day(pair, d, tcache)
        if z is None or z[2] is not None:
            return None
        t, p = z[0], z[1]
        m = (t >= a) & (t < b)
        if m.any():
            hs.append(float(p[m].max()))
    return max(hs) if hs else None


def spike_high_1m(k1, spike_close_ms, red_close_ms):
    """H from 1m highs (k1 DataFrame open_time/h) over [spike bar open, red close); None if any minute is missing."""
    a, b = int(spike_close_ms) - BAR, int(red_close_ms)
    s = k1[(k1.open_time >= a) & (k1.open_time < b)]
    if len(s) != (b - a) // MIN:
        return None
    return float(s.h.max())


SCRATCH = os.environ.get("WHT_SCRATCH", "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/"
                                        "c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/k1m")


def scratch_1m(pair, day, prev=False):
    """1m klines fetched from fapi for this study (json list) → DataFrame open_time/o/h/l/c; prev also reads the previous UTC day's file."""
    days = [day] + ([(pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")] if prev else [])
    fr = []
    for d in days:
        f = os.path.join(SCRATCH, f"{pair}_{d}.json")
        if os.path.exists(f):
            a = json.load(open(f))
            fr.append(pd.DataFrame(dict(open_time=[int(x[0]) for x in a], o=[float(x[1]) for x in a], h=[float(x[2]) for x in a],
                                        l=[float(x[3]) for x in a], c=[float(x[4]) for x in a])))
    if not fr:
        return pd.DataFrame(dict(open_time=pd.Series(dtype="int64"), o=[], h=[], l=[], c=[]))
    return pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def day_boot(x, days, n=5000, seed=7):
    x = np.asarray(x, float); days = np.asarray(days)
    ok = np.isfinite(x); x, days = x[ok], days[ok]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    u = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in u}
    s = np.array([x[idx[d]].sum() for d in u]); c = np.array([len(idx[d]) for d in u])
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(u), size=(n, len(u)))
    m = s[k].sum(1) / c[k].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def dist_bucket(d):
    return "≤2 %" if d <= 2 else ("2–5 %" if d <= 5 else ">5 %")
