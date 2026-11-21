"""FB_THEORY_EXIT_2026-11-21 — shared cells + walker + statistics (research only; no bot code / config touched).
PREREG: reports/FB_THEORY_EXIT_2026-11-21_PREREG.md (frozen before any cell was simulated).

One walker for both cohorts (5m backtest bars, 1m real bars). Conventions = FUNDING_STOP_2026-11-21_common.walk arm 'Bp' (today's exit,
DL 607): stop checked first on the bar HIGH (line = E × (1 + (d + min(F, 0)) / 100), filled at max(line, open) × (1 + slip)), maker TP
filled at the line only when the low trades 0.05 % through it, time exit at the last close inside the hold × (1 + slip), rule exits =
taker at the open of the trigger bar × (1 + slip) (the trigger bar is the bar AFTER the condition became known). Funding F = per-bar
cumulative % to the short (positive = received), already assigned to bars by the caller."""
import numpy as np
import pandas as pd

MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
THROUGH = 0.05
CAP_TP, CAP_SL = 25.0, 25.0


def cells():
    C = [dict(name="REF", fam="REF"), dict(name="X_TP6", fam="X", tp=6.0)]
    for a in ("BASE", "HALF", "R618", "R786", "FARB"):
        C.append(dict(name=f"A_{a}", fam="A", anc=a))
    for nm, kind, b in (("C_PK0", "PEAK", 0.0), ("C_PK1", "PEAK", 1.0), ("C_TIGHT", "PEAKT", 0.5), ("C_VWAP", "VWAP", 0.5), ("C_CFHI", "CFHI", 0.5)):
        C.append(dict(name=nm, fam="C", lvl=kind, b=b))
    for s in (8, 12, 15, 20):
        C.append(dict(name=f"K_S{s}", fam="K", sl=float(s)))
    for nm, key in (("K_MPK", "PEAK"), ("K_MVWAP", "VWAP"), ("K_MCFHI", "CFHI")):
        C.append(dict(name=nm, fam="K", med=key))
    for x in (2, 3, 4):
        for n in (240, 480, 960):
            C.append(dict(name=f"B_X{x}_N{n}", fam="B", x=float(x), n=n))
    C += [dict(name="T_H12_Y1", fam="T", h=12, y=1.0), dict(name="T_H24_Y2", fam="T", h=24, y=2.0),
          dict(name="T_H24_UW", fam="T", h=24, y=None), dict(name="T_H48_UW", fam="T", h=48, y=None)]
    C += [dict(name="F_P1", fam="F", y=1.0, neg=False), dict(name="F_P2", fam="F", y=2.0, neg=False), dict(name="F_P1N", fam="F", y=1.0, neg=True)]
    return C


def struct_dist(kind, b, E, an):
    """stop distance % for a structure level (None → keep today's)."""
    if an is None:
        return None
    lv = {"PEAK": an.get("peak"), "PEAKT": an.get("peak"), "VWAP": an.get("vwap"), "CFHI": an.get("cfhi")}[kind]
    if lv is None or not np.isfinite(lv):
        return None
    lv = lv * (1 + b / 100)
    if lv < E * 1.01:
        return None
    return min((lv / E - 1) * 100, CAP_SL)


def tp_anchor(anc, E, an):
    if an is None:
        return None
    pk, bs = an.get("peak", np.nan), an.get("base", np.nan)
    if not np.isfinite(bs):
        return None
    if anc in ("BASE", "FARB"):
        ap = bs
    elif anc == "HALF":
        ap = (E + bs) / 2
    else:
        if not np.isfinite(pk):
            return None
        f = {"R618": 0.618, "R786": 0.786}[anc]
        ap = pk - f * (pk - bs)
    if not (ap <= E * 0.99):
        return None
    ap = max(ap, E * (1 - CAP_TP / 100))
    if anc == "FARB":
        ap = min(ap, E * 0.95)
    return ap


def run_cell(cl, t, o, h, l, c, i0, E, end_t, F, an, slip, entry_cost, taker, maker, barms, kmed, te):
    """one short from bar i0 (inclusive) to the last bar with t < end_t. → dict(net, why, j, fund, gross) or None."""
    last = int(np.searchsorted(t, end_t, "left")) - 1
    if last < i0:
        return None
    s = slice(i0, last + 1)
    tt, oo, hh, ll, cc, Fj = t[s], o[s], h[s], l[s], c[s], F[s]
    n = len(hh)
    d = 10.0; tp_px = E * 0.95; trig = None
    f = cl["fam"]
    if f == "X":
        tp_px = E * (1 - cl["tp"] / 100)
    elif f == "A":
        ap = tp_anchor(cl["anc"], E, an)
        if ap is not None:
            tp_px = ap
    elif f == "C":
        sd = struct_dist(cl["lvl"], cl["b"], E, an)
        if sd is not None:
            d = min(d, sd) if cl["lvl"] == "PEAKT" else sd
    elif f == "K":
        d = cl["sl"] if "sl" in cl else kmed[cl["med"]]
    elif f == "B":
        runmin = np.minimum.accumulate(ll)
        best = (1 - runmin / E) * 100
        newlow = np.r_[True, ll[1:] < runmin[:-1]]
        idx = np.arange(n)
        lastnew = np.maximum.accumulate(np.where(newlow, idx, 0))
        ok = (best >= cl["x"]) & ((idx - lastnew) * barms >= cl["n"]) & (cc < E)
        ii = np.nonzero(ok[:-1])[0]
        if len(ii):
            trig = int(ii[0]) + 1
    elif f == "T":
        H = int(np.searchsorted(tt, te + cl["h"] * HR, "left"))
        if 0 < H < n:
            if cl["y"] is None:
                cond = cc[H - 1] > E
            else:
                cond = (1 - ll[:H].min() / E) * 100 < cl["y"]
            if cond:
                trig = H
    elif f == "F":
        paid = -Fj
        hit = np.nonzero(paid >= cl["y"])[0]
        for j in hit:
            if cl["neg"]:
                pr = (1 - oo[j] / E) * 100
                if not (pr < paid[j]):
                    continue
            if j + 1 < n:
                trig = int(j) + 1
            break
    line = E * (1 + (d + np.minimum(Fj, 0.0)) / 100.0)
    tl = ll <= tp_px * (1 - THROUGH / 100.0)
    hs = hh >= line
    lim = n if trig is None else trig
    ks = int(np.argmax(hs[:lim])) if hs[:lim].any() else 10**9
    kt = int(np.argmax(tl[:lim])) if tl[:lim].any() else 10**9
    if ks < 10**9 and ks <= kt:
        k = ks; X = max(line[k], oo[k]) * (1 + slip / 100.0); why = "STOP"; fee = taker * X / E
    elif kt < 10**9:
        k = kt; X = tp_px; why = "TP"; fee = maker * X / E
    elif trig is not None:
        k = trig; X = oo[k] * (1 + slip / 100.0); why = "RULE"; fee = taker * X / E
    else:
        k = n - 1; X = cc[k] * (1 + slip / 100.0); fee = taker * X / E
        why = "TIME" if tt[-1] + barms * MIN >= end_t - MIN else "OPEN"
    gross = (1 - X / E) * 100.0
    fu = float(Fj[k])
    return dict(net=gross - entry_cost - fee + fu, why=why, j=i0 + k, fund=fu, gross=gross)


# ───────────────────────────── statistics ─────────────────────────────
def block_boot(delta, keys, B=4000, seed=7):
    """cluster bootstrap of the mean Δ: resample clusters (keys) → (mean, lo95, hi95, P(mean > 0))."""
    d = np.asarray(delta, float)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else np.nan), np.nan, np.nan, np.nan
    u, inv = np.unique(np.asarray(keys), return_inverse=True)
    sums = np.bincount(inv, d, len(u)); ns = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = sums[ix].sum(1) / ns[ix].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def dcr(pct, close_ms, d0, d1, x=0.61):
    """one-size daily compound return % over every calendar day [d0, d1] + max drawdown % + worst day %."""
    days = pd.date_range(pd.Timestamp(d0).normalize(), pd.Timestamp(d1).normalize(), freq="D")
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(np.asarray(close_ms), unit="ms").normalize()).sum()
    r = s.reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return (eq[-1] ** (1 / len(r)) - 1) * 100, min(dd, 0) * 100, r.min() * 100


def loo(delta):
    d = np.asarray(delta, float)
    if len(d) < 2:
        return np.nan, np.nan
    m = (d.sum() - d) / (len(d) - 1)
    return float(m.min()), float(m.max())


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:+.{fl}f}" if abs(v) < 10000 else f"{v:,.0f}"))
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)
