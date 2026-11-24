#!/usr/bin/env python3
"""TP_HOLD_2026-11-24 — "when NOT to take the TP" study (research only; no bot code / config touched).

Operator 11-24 (PROM / ON / SAHARA ran +15 % past the TP): is there a variable AT THE TP TOUCH that says "cancel the TP, trail instead"?
Decision point = the minute the TP was hit. Cohort = every fill that closed at its TP (as traded). Outcome per fill = a trail started at
the TP price (width W % from the running best, 1m bars, stop checked before the bar's new extreme, 24 h cap then market) MINUS the TP,
in P&L-% points, after costs (trail exit = taker 0.05 + slip 0.10 vs the resting maker TP 0.02 → −0.13 pp). Funding not counted.
Features = bars CLOSED before the touch minute only (no look-ahead). Screen = every feature × {median split, terciles} × both sides
× every width; null = the same max statistic with each feature shuffled within sleeve (1,000×).

  phase real  : real fills (exports_archive, dedup, no probes, operator-closed out) + the 2 newest fills from the operator's 11-24 charts
  phase bt    : the backtest cohorts (run with --bt)
"""
import glob, os, sys, json, functools
print = functools.partial(print, flush=True)
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import importlib
C = importlib.import_module("STATE_EXIT_2026-11-21_common")
import scout_gatekit as GK  # noqa: E402

OUT = os.path.join(ROOT, "reports", "TP_HOLD_2026-11-24_data")
os.makedirs(OUT, exist_ok=True)
MIN, HR = 60_000, 3_600_000
SLV = ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT")
WIDTHS = (1.0, 2.0, 3.0, 5.0)
COST_PP = 0.05 + 0.10 - 0.02
HOLD_MAX = 24 * HR
# the two fills closed after the newest export (operator screenshots 11-24): opened, pair, entry, TP exit, touch time
EXTRA = [dict(entry_strategy="DISTRIBUTION_SHORT", pair="ONUSDT", direction="SHORT", opened_at="2026-11-24T03:00:10",
              closed_at="2026-11-24T05:58:16", entry_price=0.14472, exit_price=0.13020, close_reason="FRENZY_TP", src="screenshot"),
         dict(entry_strategy="EXHAUSTION_SHORT", pair="SAHARAUSDT", direction="SHORT", opened_at="2026-11-24T05:15:31",
              closed_at="2026-11-24T12:01:02", entry_price=0.0058781, exit_price=0.0054660, close_reason="FRENZY_TP", src="screenshot")]


def real_cohort():
    P = []
    for f in sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv"))):
        h = pd.read_csv(f, nrows=0).columns
        if "entry_strategy" not in h:
            continue
        x = pd.read_csv(f, low_memory=False)
        x = x[x.entry_strategy.isin(SLV) & (x.status == "CLOSED")]
        if len(x):
            x["src"] = os.path.basename(f); P.append(x)
    X = pd.concat(P, ignore_index=True)
    if "is_probe" in X:
        X = X[~X.is_probe.fillna(False).astype(str).str.lower().isin(["true", "1"])]
    X["k"] = X.opened_at.astype(str).str[:19]
    X = X.sort_values("src").drop_duplicates(["k", "pair", "direction"], keep="last")
    X, oc = GK.split_operator_closed(X)
    E = pd.DataFrame(EXTRA); E["k"] = E.opened_at.str[:19]
    E = E[~E.set_index(["k", "pair"]).index.isin(X.set_index(["k", "pair"]).index)]
    X = pd.concat([X, E], ignore_index=True)
    X = X[X.close_reason.astype(str).isin(["FRENZY_TP", "WILLY_LATE_TP"])].copy()
    _ms = lambda s: ((pd.to_datetime(s, utc=True, format="mixed") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")
    X["t_open"] = _ms(X.opened_at)
    X["t_touch"] = _ms(X.closed_at)
    X["tp_px"] = X.exit_price.astype(float)
    X["E"] = X.entry_price.astype(float)
    X["dirn"] = np.where(X.direction.astype(str).str.upper() == "SHORT", -1, 1)
    return X[["entry_strategy", "pair", "dirn", "t_open", "t_touch", "E", "tp_px", "src", "close_reason"]].reset_index(drop=True)


def fetch(X):
    need = set()
    for r in X.itertuples():
        for p in (r.pair, "BTCUSDT"):
            for ds in C.days_of(r.t_touch - 72 * HR, r.t_touch + HOLD_MAX):
                need.add((p, ds))
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d"); yday = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for p, ds in sorted(need):
        f = os.path.join(C.K1, p, ds + ".npz")
        if ds in (today, yday) and os.path.exists(f):
            os.remove(f)            # a recent day may have been cached partial — re-read it
        if pd.Timestamp(ds) > pd.Timestamp(today):
            continue
        C.fetch_kday(p, ds)
        if p != "BTCUSDT":
            C.fetch_mday(p, ds)


def trail(g, i0, tp_px, dirn, w, floor=None):
    """after the TP touch: best = most favourable price since; exit at the first bar reaching the stop = the tighter of the trail
    (W % back from best) and the 🛡 SAFETY FLOOR (operator 11-24: TP − F %, fixed at the touch; None = no floor). A bar OPENING beyond
    the stop fills at its open (gap). → (exit_px, minutes, censored)."""
    best = tp_px
    fl = None if floor is None else (tp_px * (1 + floor / 100) if dirn < 0 else tp_px * (1 - floor / 100))
    n = len(g["t"])
    for i in range(i0 + 1, n):
        hi, lo, op = g["h"][i], g["l"][i], g["o"][i]
        if dirn < 0:
            stop = best * (1 + w / 100)
            if fl is not None:
                stop = min(stop, fl)
            if hi >= stop:
                return max(stop, op), i - i0, False
            best = min(best, lo)
        else:
            stop = best * (1 - w / 100)
            if fl is not None:
                stop = max(stop, fl)
            if lo <= stop:
                return min(stop, op), i - i0, False
            best = max(best, hi)
    return g["c"][n - 1], n - 1 - i0, True


# outcome columns: plain trails d<W> + floor variants f<F>w<W>
# fixed-% (reference) + 📐 ATR-scaled (operator 11-24 "atr based trails, based on our past learnings": EXH / DIST exits are ATR-sized).
# A spec is a number (%) or ("atr", k) = k × the coin's 15m ATR14 % at the touch.
VARIANTS = ([(f"d{w:g}", w, None) for w in WIDTHS] + [(f"f{fl:g}w{w:g}", w, fl) for fl in (0.5, 1.0, 2.0) for w in (3.0, 5.0, 8.0)]
            + [(f"a{k:g}", ("atr", k), None) for k in (1.0, 1.5, 2.0, 3.0)]
            + [(f"af{fk:g}a{k:g}", ("atr", k), ("atr", fk)) for fk in (0.5, 1.0) for k in (2.0, 3.0)])


def _spec(x, atr):
    if x is None or not isinstance(x, tuple):
        return x
    return (x[1] * atr) if (atr is not None and np.isfinite(atr) and atr > 0) else None


def atr15_pct(pair, T):
    """the coin's 15m ATR14 as % of the last close, from the 1m bars closed before the touch minute."""
    a = C.k1m(pair, T - 5 * HR, T)
    if a is None or len(a["t"]) < 15 * 15:
        return np.nan
    t = a["t"]; b = (t // (15 * MIN))
    df = pd.DataFrame(dict(b=b, h=a["h"], l=a["l"], c=a["c"])).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pc = df.c.shift(1)
    tr = np.maximum(df.h - df.l, np.maximum((df.h - pc).abs(), (df.l - pc).abs())).dropna()
    if len(tr) < 14:
        return np.nan
    return float(tr.iloc[-14:].mean() / df.c.iloc[-1] * 100)
OUTS = [v[0] for v in VARIANTS]


def ret(g, a, b, dirn):
    if a < 0 or not (g["c"][a] > 0):
        return np.nan
    return dirn * (g["c"][b] / g["c"][a] - 1) * 100


def features(r, g, gb, i0):
    """bars closed BEFORE the touch minute: index i0 is the touch minute → use ≤ i0-1."""
    j = i0 - 1
    if j < 61:
        return None
    d = r.dirn
    qv = g["qv"]
    f = dict(
        mins_to_tp=(r.t_touch - r.t_open) / MIN,
        r5=ret(g, j - 5, j, d), r15=ret(g, j - 15, j, d), r60=ret(g, j - 60, j, d),
        vol5_60=(qv[j - 4:j + 1].sum() / 5) / max(qv[j - 59:j + 1].mean(), 1e-9),
        vol15_60=(qv[j - 14:j + 1].sum() / 15) / max(qv[j - 59:j + 1].mean(), 1e-9),
        taker_with15=(lambda s, q: (s / q if d > 0 else 1 - s / q) if q > 0 else np.nan)(g["tbq"][j - 14:j + 1].sum(), qv[j - 14:j + 1].sum()),
        range5_atr=((g["h"][j - 4:j + 1].max() - g["l"][j - 4:j + 1].min()) / max(np.mean(g["h"][j - 59:j + 1] - g["l"][j - 59:j + 1]) * 5, 1e-12)),
        fav5=float(np.sum(d * (g["c"][j - 4:j + 1] - g["o"][j - 4:j + 1]) > 0)),
        gap_to_tp=d * (r.tp_px / g["c"][j] - 1) * 100,            # how far the last close was from the TP (fast jump vs crawl)
        hour=float(pd.Timestamp(r.t_touch, unit="ms").hour),
    )
    # 🧾 operator 11-24 "no buyers, mostly sellers": taker flow WITH the trade (short → taker SELL share) over 1 / 3 / 5 / 60 min,
    # its acceleration, and the with-trade taker volume vs its own 60-min average
    tb = g["tbq"]
    def _with(a, b):
        q, s = qv[a:b + 1].sum(), tb[a:b + 1].sum()
        return np.nan if q <= 0 else (s / q if d > 0 else 1 - s / q)
    def _wvol(a, b):
        return (qv[a:b + 1] - tb[a:b + 1]).sum() if d < 0 else tb[a:b + 1].sum()
    f["taker_with1"] = _with(j, j); f["taker_with3"] = _with(j - 2, j); f["taker_with5"] = _with(j - 4, j)
    f["taker_with60"] = _with(j - 59, j)
    f["taker_accel5_60"] = (f["taker_with5"] - f["taker_with60"]) if np.isfinite(f["taker_with5"]) and np.isfinite(f["taker_with60"]) else np.nan
    f["with_vol5_60"] = (_wvol(j - 4, j) / 5) / max(_wvol(j - 59, j) / 60, 1e-9)
    # worst adverse excursion before the touch
    k0 = int(np.searchsorted(g["t"], r.t_open // MIN * MIN))
    if k0 < j:
        adv = (g["h"][k0:j + 1].max() / r.E - 1) * 100 if d < 0 else (1 - g["l"][k0:j + 1].min() / r.E) * 100
        f["mae_before"] = adv
    else:
        f["mae_before"] = np.nan
    if gb is not None:
        f["btc15"] = ret(gb, j - 15, j, d); f["btc60"] = ret(gb, j - 60, j, d)
    else:
        f["btc15"] = f["btc60"] = np.nan
    return f


def build(X):
    rows = []
    for r in X.itertuples():
        t0, t1 = r.t_touch // MIN * MIN - 2 * HR, r.t_touch // MIN * MIN + HOLD_MAX
        a = C.k1m(r.pair, t0, t1)
        if a is None or not len(a["t"]):
            continue
        now = int(pd.Timestamp.utcnow().value // 10**6) // MIN * MIN
        g = C.grid(a, t0, min(t1, int(a["t"][-1]) + MIN))
        b = C.k1m("BTCUSDT", t0, t1)
        gb = C.grid(b, t0, min(t1, int(b["t"][-1]) + MIN)) if b is not None and len(b["t"]) else None
        i0 = int((r.t_touch // MIN * MIN - t0) // MIN)
        if i0 >= len(g["t"]):
            continue
        f = features(r, g, gb, i0)
        if f is None:
            continue
        o = dict(sleeve=r.entry_strategy, pair=r.pair, t_touch=r.t_touch, src=r.src, E=r.E, tp_px=r.tp_px, dirn=r.dirn,
                 tp_pct=r.dirn * (r.tp_px / r.E - 1) * 100, bars_after=len(g["t"]) - 1 - i0)
        _atr = atr15_pct(r.pair, r.t_touch // MIN * MIN); o["atr15"] = _atr
        for nm, w, fl in VARIANTS:
            w_, fl_ = _spec(w, _atr), _spec(fl, _atr)
            if w_ is None:
                o[nm] = np.nan; continue
            px, m, cen = trail(g, i0, r.tp_px, r.dirn, w_, fl_)
            o[nm] = r.dirn * (px / r.tp_px - 1) * 100 * (r.tp_px / r.E) - COST_PP   # pp of entry
            o["m_" + nm] = m; o["c_" + nm] = cen
        # what the path offered: best favourable move within 24 h after the touch (hindsight ceiling)
        seg = slice(i0 + 1, len(g["t"]))
        if len(g["t"]) > i0 + 1:
            o["best_after"] = (r.dirn * ((g["l"][seg].min() if r.dirn < 0 else g["h"][seg].max()) / r.tp_px - 1) * 100) * (r.tp_px / r.E)
        o.update(f); o.update(theory_feats(r.pair, r.t_touch, r.tp_px, r.dirn))
        rows.append(o)
    return pd.DataFrame(rows)


# ─────────── 🧠 THEORY variables (operator 11-24 "focus on theory of what the trade is about") ───────────
# Thesis (pump-unwind shorts EXH / DIST / FB; HT = the long mirror): a FRENZY pump pulls in late leveraged longs; the short earns as they are
# forced out. After the TP the drop continues only while (T1) FUEL is left — open interest still high vs before the pump AND falling now
# (forced exits in progress) · (T2) ROOM is left — the price still far above the pre-pump base / small share of the pump unwound ·
# (T3) NO BUYERS — sellers dominate, closes at the lows, no absorbing lower wicks · (T4) ACCELERATION — the move speeding up.
def theory_feats(pair, t_touch, tp_px, d):
    f = {}
    T = t_touch // MIN * MIN
    a = C.k1m(pair, T - 72 * HR, T)
    if a is not None and len(a["t"]) > 600:
        c, h, l = a["c"], a["h"], a["l"]
        old = c[a["t"] < T - 48 * HR]
        base = np.median(old) if len(old) > 60 else np.nan
        rec = a["t"] >= T - 48 * HR
        ext = (h[rec].max() if d < 0 else l[rec].min()) if rec.any() else np.nan      # the pump extreme (short: the peak)
        if np.isfinite(base) and base > 0:
            f["room_to_base"] = (tp_px / base - 1) * 100 * (-d)   # short: +x % = the TP price still x % above the pre-pump base
            mv = ext - base
            f["unwound_frac"] = ((ext - tp_px) / mv) if (d < 0 and mv > 0) else (((tp_px - ext) / -mv) if (d > 0 and mv < 0) else np.nan)
        else:
            f["room_to_base"] = f["unwound_frac"] = np.nan
        j = len(c) - 1
        if j > 15:
            o = a["o"][j - 4:j + 1]; hh = h[j - 4:j + 1]; ll = l[j - 4:j + 1]; cc = c[j - 4:j + 1]
            rng = np.maximum(hh - ll, 1e-12)
            cl = (cc - ll) / rng                                     # 0 = closed at the low
            f["close_loc5"] = float(np.mean(1 - cl)) if d < 0 else float(np.mean(cl))         # 1 = closes WITH the trade
            wick = (np.minimum(o, cc) - ll) / rng if d < 0 else (hh - np.maximum(o, cc)) / rng  # absorbing wick against the trade
            f["absorb_wick5"] = float(np.mean(wick))
            r5 = d * (c[j] / c[j - 5] - 1) * 100; r15 = d * (c[j] / c[j - 15] - 1) * 100
            f["accel"] = r5 - r15 / 3
    m = C.metrics(pair, T - 26 * HR, T)
    if m is not None and len(m["t"]) > 10:
        oi = m["oi"]; t = m["t"]
        now = oi[t <= T][-1] if (t <= T).any() else np.nan
        h1 = oi[t <= T - HR][-1] if (t <= T - HR).any() else np.nan
        d24 = oi[t <= T - 24 * HR][-1] if (t <= T - 24 * HR).any() else np.nan
        f["oi_chg_1h"] = (now / h1 - 1) * 100 if h1 and h1 > 0 else np.nan
        f["oi_vs_24h"] = (now / d24 - 1) * 100 if d24 and d24 > 0 else np.nan
        tp_ = m["top_pos"][t <= T]
        f["top_pos"] = float(tp_[-1]) if len(tp_) else np.nan
    for k in ("room_to_base", "unwound_frac", "close_loc5", "absorb_wick5", "accel", "oi_chg_1h", "oi_vs_24h", "top_pos"):
        f.setdefault(k, np.nan)
    return f


THEORY = ["room_to_base", "unwound_frac", "close_loc5", "absorb_wick5", "accel", "oi_chg_1h", "oi_vs_24h", "top_pos"]


FEATS = ["mins_to_tp", "r5", "r15", "r60", "vol5_60", "vol15_60", "taker_with15", "range5_atr", "fav5", "gap_to_tp", "hour",
         "mae_before", "btc15", "btc60", "taker_with1", "taker_with3", "taker_with5", "taker_with60", "taker_accel5_60", "with_vol5_60", "atr15"] + THEORY


def zones(D, v):
    x = D[v]
    if x.notna().sum() < 8:
        return []
    out = []
    med = x.median(); q1, q2 = x.quantile([1 / 3, 2 / 3])
    out += [(f"{v} > {med:.3g}", x > med), (f"{v} ≤ {med:.3g}", x <= med)]
    out += [(f"{v} ≤ {q1:.3g} (low third)", x <= q1), (f"{v} {q1:.3g}…{q2:.3g} (mid third)", (x > q1) & (x <= q2)),
            (f"{v} > {q2:.3g} (top third)", x > q2)]
    return out


def screen(D, nshuf=1000, seed=7):
    """rule = hold (trail) inside the zone, take TP elsewhere → value = Σ Δ over the zone. stat = best total over every test."""
    res = []
    for col in OUTS:
        w = col
        for v in FEATS:
            for name, m in zones(D, v):
                m = m.fillna(False)
                n = int(m.sum())
                if n < 5:
                    continue
                z = D.loc[m, col]
                res.append(dict(exit=w, zone=name, n=n, sum_pp=z.sum(), avg_pp=z.mean(), win=(z > 0).mean(),
                                n_pairs=D.loc[m, "pair"].nunique(), top_pair_share=(D.loc[m].groupby("pair")[col].sum().clip(lower=0).max() / max(z.clip(lower=0).sum(), 1e-9))))
    R = pd.DataFrame(res).sort_values("sum_pp", ascending=False)
    rng = np.random.default_rng(seed)
    best_null = []
    for _ in range(nshuf):
        S = D.copy()
        for v in FEATS:
            S[v] = S.groupby("sleeve")[v].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        bb = -1e9
        for col in OUTS:
            for v in FEATS:
                for name, m in zones(S, v):
                    m = m.fillna(False)
                    if m.sum() >= 5:
                        bb = max(bb, S.loc[m, col].sum())
        best_null.append(bb)
    return R, np.array(best_null)


def main():
    X = real_cohort()
    print("TP-hit real fills:", X.groupby("entry_strategy").size().to_dict())
    fetch(X)
    D = build(X)
    D.to_csv(os.path.join(OUT, "real_tp_hits.csv"), index=False)
    print("built", len(D), D.groupby("sleeve").size().to_dict())
    # unconditional: always trail vs always take TP
    for s, g in list(D.groupby("sleeve")) + [("ALL", D)]:
        print(f"\n{s}  N={len(g)}  TP avg {g.tp_pct.mean():+.2f}  best-after (hindsight) median {g.best_after.median():+.2f} pp  mean {g.best_after.mean():+.2f}")
        for nm in OUTS:
            c = g[nm]
            print(f"   always {nm:8s}: Δ vs TP avg {c.mean():+.2f} pp · median {c.median():+.2f} · better {100*(c>0).mean():.0f}% · Σ {c.sum():+.1f} pp")
    R, null = screen(D, nshuf=int(os.environ.get("NSHUF", 500)))
    R.to_csv(os.path.join(OUT, "real_screen.csv"), index=False)
    p95 = np.quantile(null, 0.95)
    print(f"\nscreen: {len(R)} tests · null best Σ: median {np.median(null):+.1f} · 95th {p95:+.1f} pp")
    print(R.head(15).round(3).to_string(index=False))
    json.dump(dict(null_p95=float(p95), null_med=float(np.median(null)), n_tests=len(R), best=float(R.sum_pp.iloc[0]),
                   p_best=float((null >= R.sum_pp.iloc[0]).mean())), open(os.path.join(OUT, "real_null.json"), "w"))
    print("p(best real ≥ null) =", (null >= R.sum_pp.iloc[0]).mean())


if __name__ == "__main__" and "--bt" not in sys.argv:
    main()


# ───────────────────────────── phase bt: backtest cohorts on TODAY's exits ─────────────────────────────
SE = os.path.join(ROOT, "reports", "STATE_EXIT_2026-11-21_data", "fills_out.csv")
DIST_T = os.path.join(ROOT, "reports", "DIST_EXIT_2026-11-18_trades.csv.gz")
SPLIT = int(pd.Timestamp("2025-07-01").value // 10**6)
SL_MAP = {"EXH": "EXHAUSTION_SHORT", "FB": "FLATBASE_SHORT", "HT": "FRENZY_WILLY_HT", "DIST": "DISTRIBUTION_SHORT"}


def bt_entries():
    f = pd.read_csv(SE, low_memory=False)
    f = f[f.kind == "BT"].copy()
    rows = []
    for r in f.itertuples():
        E, d = float(r.E), int(r.dirn)
        if r.sleeve == "FB":
            tp, sl, hold = E * 0.95, float(r.sl_px), 72
        elif r.sleeve == "HT":
            tp, sl, hold = E * (1 + (6.0 + 0.07) / 100), float(r.sl_px), 24          # HT TP +6 net (11-21 20:59)
        else:
            tpp = min(abs(float(r.tp_px) / E - 1) * 100, 7.0)                          # EXH TP cap 7 (DL 609)
            tp, sl, hold = E * (1 - tpp / 100), float(r.sl_px), 24
        rows.append(dict(sleeve=r.sleeve, pair=r.pair, te=int(r.te), E=E, dirn=d, tp=tp, sl=sl, hold_h=hold))
    D = pd.read_csv(DIST_T)
    D = D[D.cell == "REF_B"]
    for r in D.itertuples():
        rows.append(dict(sleeve="DIST", pair=r.pair, te=int(r.entry_ms), E=np.nan, dirn=-1, tp_pct=min(float(r.tp_pct), 7.0),
                         sl_pct=float(r.sl_pct), hold_h=24))
    return pd.DataFrame(rows)


def _fetch_parallel(pairs_days):
    from concurrent.futures import ThreadPoolExecutor
    todo = [(p, ds) for p, ds in pairs_days if not os.path.exists(os.path.join(C.K1, p, ds + ".npz"))]
    print("fetching", len(todo), "pair-days from data.binance.vision")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda x: C.fetch_kday(*x), todo))
    mt = sorted({(p, ds) for p, ds in pairs_days if p != "BTCUSDT" and not os.path.exists(os.path.join(C.MET, p, ds + ".npz"))})
    print("fetching", len(mt), "metric days")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda x: C.fetch_mday(*x), mt))


def bt_build(B):
    need = set()
    for r in B.itertuples():
        for p in (r.pair, "BTCUSDT"):
            for ds in C.days_of(r.te - 72 * HR, r.te + (r.hold_h + 24) * HR):
                need.add((p, ds))
    _fetch_parallel(sorted(need))
    out = []
    for r in B.itertuples():
        t0 = r.te // MIN * MIN - 2 * HR
        t1 = r.te // MIN * MIN + int(r.hold_h) * HR + HOLD_MAX + MIN
        a = C.k1m(r.pair, t0, t1)
        if a is None or len(a["t"]) < 200:
            continue
        g = C.grid(a, t0, min(t1, int(a["t"][-1]) + MIN))
        ie = int((r.te // MIN * MIN - t0) // MIN)
        if ie + 1 >= len(g["t"]):
            continue
        E = r.E if np.isfinite(r.E) else g["o"][ie]                       # DIST: the entry minute's open
        if r.sleeve == "DIST":
            tp, sl = E * (1 - r.tp_pct / 100), E * (1 + r.sl_pct / 100)
        else:
            tp, sl = r.tp, r.sl
        d = r.dirn
        iend = min(len(g["t"]) - 1, ie + int(r.hold_h) * 60)
        touch = None
        for i in range(ie + 1, iend + 1):                               # the walk starts the minute after the fill
            hi, lo = g["h"][i], g["l"][i]
            if (d < 0 and hi >= sl) or (d > 0 and lo <= sl):
                break                                                   # stop first on a bar touching both
            if (d < 0 and lo <= tp) or (d > 0 and hi >= tp):
                touch = i; break
        if touch is None:
            continue
        b = C.k1m("BTCUSDT", t0, t1)
        gb = C.grid(b, t0, min(t1, int(b["t"][-1]) + MIN)) if b is not None and len(b["t"]) else None
        rr = pd.Series(dict(t_touch=int(g["t"][touch]), t_open=int(r.te), tp_px=tp, E=E, dirn=d))
        f = features(rr, g, gb, touch)
        if f is None:
            continue
        o = dict(sleeve=SL_MAP[r.sleeve], pair=r.pair, t_touch=int(g["t"][touch]), period=("DESIGN" if r.te < SPLIT else "HOLDOUT"),
                 E=E, tp_px=tp, dirn=d, tp_pct=d * (tp / E - 1) * 100, bars_after=len(g["t"]) - 1 - touch)
        gg = {k: v[: min(len(g["t"]), touch + 1 + HOLD_MAX // MIN)] for k, v in g.items()}
        _atr = atr15_pct(r.pair, int(g["t"][touch])); o["atr15"] = _atr
        for nm, w, fl in VARIANTS:
            w_, fl_ = _spec(w, _atr), _spec(fl, _atr)
            if w_ is None:
                o[nm] = np.nan; continue
            px, m, cen = trail(gg, touch, tp, d, w_, fl_)
            o[nm] = d * (px / tp - 1) * 100 * (tp / E) - COST_PP
            o["m_" + nm] = m; o["c_" + nm] = cen
        seg = slice(touch + 1, len(gg["t"]))
        o["best_after"] = (d * ((gg["l"][seg].min() if d < 0 else gg["h"][seg].max()) / tp - 1) * 100) * (tp / E)
        o.update(f); o.update(theory_feats(r.pair, int(g["t"][touch]), tp, d))
        out.append(o)
    return pd.DataFrame(out)


def summarize(D, label):
    print(f"\n===== {label} =====")
    for s, g in list(D.groupby("sleeve")) + [("ALL", D)]:
        print(f"{s}  N={len(g)}  TP avg {g.tp_pct.mean():+.2f}  hindsight best-after median {g.best_after.median():+.2f}  mean {g.best_after.mean():+.2f}")
        for nm in OUTS:
            c = g[nm]
            print(f"   always {nm:8s}: avg {c.mean():+.2f} · median {c.median():+.2f} · better {100*(c>0).mean():.0f}% · Σ {c.sum():+.1f}")


def bt_main():
    B = bt_entries()
    print("entries:", B.groupby("sleeve").size().to_dict())
    D = bt_build(B)
    D.to_csv(os.path.join(OUT, "bt_tp_hits.csv"), index=False)
    for per, g in D.groupby("period"):
        summarize(g, f"BACKTEST {per}")
    res = {}
    for s, g in list(D.groupby("sleeve")) + [("ALL", D)]:
        des, hol = g[g.period == "DESIGN"], g[g.period == "HOLDOUT"]
        if len(des) < 30:
            continue
        R, null = screen(des.reset_index(drop=True), nshuf=int(os.environ.get("NSHUF", 200)))
        p95 = np.quantile(null, 0.95)
        top = R[R.sum_pp > p95].head(10).copy()
        print(f"\n[{s}] DESIGN screen {len(R)} tests · null 95th Σ {p95:+.1f} · best {R.sum_pp.iloc[0]:+.1f} · survivors {len(top)}")
        print(R.head(8).round(3).to_string(index=False))
        # HOLDOUT confirmation of each survivor with the DESIGN-frozen cut
        for t in top.itertuples():
            v = t.zone.split(" ")[0]; col = t.exit
            m = re_mask(hol, t.zone, v)
            z = hol.loc[m, col]
            print(f"   HOLDOUT {t.zone} {t.exit}: n {len(z)} · avg {z.mean():+.2f} · Σ {z.sum():+.1f} · better {100*(z>0).mean():.0f}%")
        res[s] = dict(null95=float(p95), best=float(R.sum_pp.iloc[0]), survivors=len(top))
        R.to_csv(os.path.join(OUT, f"bt_screen_{s}.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "bt_null.json"), "w"), indent=1)


def re_mask(D, zone, v):
    import re
    x = D[v]
    m = re.search(r"≤ ([-\d.e+]+) \(low", zone)
    if m: return x <= float(m.group(1))
    m = re.search(r"([-\d.e+]+)…([-\d.e+]+)", zone)
    if m: return (x > float(m.group(1))) & (x <= float(m.group(2)))
    m = re.search(r"> ([-\d.e+]+)", zone)
    if m: return x > float(m.group(1))
    m = re.search(r"≤ ([-\d.e+]+)", zone)
    return x <= float(m.group(1))


if __name__ == "__main__" and "--bt" in sys.argv:
    bt_main()
