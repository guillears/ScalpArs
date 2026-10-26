"""B19 review §2 — FRENZY_LONG / WIDE / LITE exits: shared cohorts, path loader, exit evaluator, statistics (CACHE-ONLY).

Cohorts (explicit, NEXT_BATCH_REVIEW_PLAN §0c):
  ① B19   = CLOSED FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE bot fills in ~/Downloads/scalpars_orders_paper_2026-10-11_21-09-44.csv
            opened ≥ 2026-10-08 18:30 UTC (deploy 85e6a83, today's stack: +3 / −3 fixed, 12 h, bearish block, ATR 3.0; market-volume
            gate ON until the (263) deploy 10-09, OFF after — irrelevant to the exit), dedup (opened_at[:19], pair, entry_strategy),
            MANUAL closes none.  Entry = live opened_at + entry_price (live fills book at about the print — DL 270).
  ② MASTER = reports/MASTER_POOL_stacked.csv (STACK 2026-10-08c) FRENZY_LONG / WIDE / LITE rows with stack_keep (today's stack keeps
            them), split per archived batch by opened_at (B16 ≤ 10-03 12:00 · B17 10-03 23:25 → 10-06 12:27 · B18 10-06 13:55 → 10-08
            18:30).  Refused rows (bearish day / WIDE ATR / reclaim) are listed as context, never counted.  Their live exits were
            trail / lock / fixed-+3 eras → parity is checked only where the live exit is a −3 stop or a +3 TP.
  ③ BACKTEST = DL 270 corrected recipe (scratch fz_grid_final.py): reports/FRENZY_TP2_VS_TP3_TICKS_2026-10-09_fills.csv st ok:
            replay cohort a seed 1 (bearish block, live eligibility, slots) + a_off_addback (the market-volume-gate refusals, gate off
            since 263) live-eligible only and not already in the replay (pair, sig) → LONG / WIDE;  LITE = cohort c, bear False,
            eligible (no LITE slot / pair-held / day-cap sim — DL 270 puts that at ≈ −0.04 %/fill).  ONE seed; entry = the study's
            print at signal close + 8 s × 1.0010 (E, entry_ms).  Halves = the study's 'half' column.

Path: aggTrades ticks where a full UTC day (or a covering slice) is cached, then 1m klines (PROVISIONAL: the entry minute's bar is
skipped, a bar touching both lines = stop, a bar opening through the stop fills at its open, TP booked at the line).  All levels are
judged on the bot's NET % (scout_willy_timecap.net_pct: gross − 0.045 − 0.045 × r).  Ruler 'tick' books the first print through a
line (the yr5 study ruler); ruler 'line' books exactly ±level (the paper engine books about the line — CFX 10-10 15:25 −3.00 live vs
−3.93 first print).  A variant whose exit is not decided before the cached path ends is UNRESOLVED (listed, never priced).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT            # noqa: E402  (parity-proven walker + caches)

MIN = 60_000
DAY = 86_400_000
H12 = 720
SLEEVES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
B19_EXPORT = os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-10-11_21-09-44.csv")
B19_FROM = "2026-10-08T18:30"
STUDY = os.path.join(ROOT, "reports", "FRENZY_TP2_VS_TP3_TICKS_2026-10-09_fills.csv")
MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
OUT_ROWS = os.path.join(ROOT, "reports", "B19_REVIEW_FRENZY_EXITS_rows.csv")

# ── pre-registered variant families (frozen BEFORE any read; counted in the report) ──
BASE = ("B", 3.0, 3.0, H12)                                           # today's exit: +3 / −3 / 12 h
GRID = [("+1/-2", 1.0, 2.0, H12), ("+1/-2.5", 1.0, 2.5, H12), ("+1/-3", 1.0, 3.0, H12), ("+1.25/-2", 1.25, 2.0, H12),
        ("+1.25/-2.5", 1.25, 2.5, H12), ("+1.25/-3", 1.25, 3.0, H12), ("+1.25/-4", 1.25, 4.0, H12)]          # 7 (Scout 267/269)
CAPS = [(f"cap{c}", 3.0, 3.0, c) for c in (30, 60, 120, 240, 360, 480)]                                     # 6 (new, §2 duration)
STOPS = [(f"sl{s:g}", 3.0, s, H12) for s in (2.0, 2.5, 3.5, 4.0, 5.0)]                                      # 5 (new, §2 trough)
NOSTOP = ("nostop", 3.0, None, H12)                                   # descriptive only (winners' trough), never a candidate
VARIANTS = [BASE] + GRID + CAPS + STOPS + [NOSTOP]
FAMILIES = {"GRID": [v[0] for v in GRID], "CAPS": [v[0] for v in CAPS], "STOPS": [v[0] for v in STOPS]}


def ms(ts):
    return int(pd.Timestamp(str(ts)[:19]).tz_localize("UTC").value // 1_000_000)


def era_b(opened):
    o = str(opened)[:19]
    if o < "2026-10-03T23:00":
        return "B16"
    if o < "2026-10-06T13:00":
        return "B17"
    if o < "2026-10-08T18:30":
        return "B18"
    return "B19"


# ─────────────────────────── cohorts ───────────────────────────
def load_b19():
    o = pd.read_csv(B19_EXPORT, low_memory=False)
    o = o[o.entry_strategy.isin(SLEEVES) & (o.status == "CLOSED") & (o.opened_at.astype(str) >= B19_FROM)].copy()
    o["k"] = o.opened_at.astype(str).str[:19] + o.pair + o.entry_strategy
    o = o.drop_duplicates("k", keep="last")
    return pd.DataFrame(dict(block="B19", era="B19", sleeve=o.entry_strategy.values, pair=o.pair.values,
                             t=o.opened_at.astype(str).str[:19].values, te=[ms(x) for x in o.opened_at], E=o.entry_price.astype(float).values,
                             live_why=o.close_reason.values, live_pct=o.pnl_percentage.astype(float).values,
                             live_peak=o.peak_pnl.astype(float).values, live_trough=o.trough_pnl.astype(float).values,
                             live_xms=[ms(x) for x in o.closed_at], keep=True, why_refused=""))


def load_master():
    m = pd.read_csv(MASTER, low_memory=False)
    m = m[m.entry_strategy.isin(SLEEVES)].copy()
    return pd.DataFrame(dict(block="MASTER", era=[era_b(x) for x in m.opened_at], sleeve=m.entry_strategy.values, pair=m.pair.values,
                             t=m.opened_at.astype(str).str[:19].values, te=[ms(x) for x in m.opened_at], E=m.entry_price.astype(float).values,
                             live_why=m.close_reason.values, live_pct=m.pnl_percentage.astype(float).values,
                             live_peak=m.peak_pnl.astype(float).values, live_trough=m.trough_pnl.astype(float).values,
                             live_xms=[ms(x) for x in m.closed_at], keep=m.stack_keep.astype(bool).values,
                             why_refused=np.where(m.stack_keep.astype(bool), "", m.stack_block_reason.astype(str)),
                             stack_pct=m.stack_pct.values))


def load_backtest():
    import study_fz_bughunt_eligibility as EL
    D = pd.read_csv(STUDY)
    D = D[(D.st == "ok") & D.entry_ms.notna()]
    a = D[(D.cohort == "a") & (D.seed == 1)]
    ad = D[D.cohort == "a_off_addback"].copy()
    li = D[(D.cohort == "c") & (D.bear == False)].copy()           # noqa: E712
    ka = set(zip(a.pair, a.sig))
    ad = ad[[EL.why(p, int(s)) == "eligible" and (p, s) not in ka for p, s in zip(ad.pair, ad.sig)]]
    li = li[[EL.why(p, int(s)) == "eligible" for p, s in zip(li.pair, li.sig)]]
    C = pd.concat([a, ad, li])
    return pd.DataFrame(dict(block="BACKTEST", era=C.half.values, sleeve=C.sleeve.values, pair=C.pair.values, t=C.t.astype(str).str[:19].values,
                             te=C.entry_ms.astype("int64").values, E=C.E.astype(float).values, live_why="", live_pct=np.nan,
                             live_peak=np.nan, live_trough=np.nan, live_xms=np.nan, keep=True, why_refused="",
                             src_cohort=C.cohort.values, study_fix3=C.fix3.values, study_fix3_how=C.fix3_how.values))


# ─────────────────────────── path ───────────────────────────
def load_path(pair, te, E, horizon=H12, tcache=None):
    """→ dict(t, hi, lo, cl, op, bar, end_ms, n_tick, n_bar) — net % events strictly after te up to te + horizon (+1 min slack)."""
    lim = te + (horizon + 1) * MIN
    days = [f"{d:%Y-%m-%d}" for d in pd.date_range(pd.Timestamp(te, unit="ms").normalize(), pd.Timestamp(lim, unit="ms").normalize())]
    ts, ps, cov = [], [], te
    for d in days:
        z = WT._load_day(pair, d, tcache)
        d0 = int(pd.Timestamp(d).tz_localize("UTC").value // 1_000_000)
        if z is None:
            break
        if z[2] is not None:                                   # window slice: usable only if one window covers the rest of the need
            w0, w1 = z[2]
            ok = np.flatnonzero((w0 <= max(cov, d0)) & (w1 >= min(lim, d0 + DAY)))
            if not len(ok):
                break
            ts.append(z[0]); ps.append(z[1]); cov = min(lim, d0 + DAY)
            continue
        ts.append(z[0]); ps.append(z[1]); cov = d0 + DAY
    if ts:
        t = np.concatenate(ts); p = np.concatenate(ps)
        i, j = np.searchsorted(t, te, side="right"), np.searchsorted(t, min(cov, lim), side="right")
        t, p = t[i:j], p[i:j]
    else:
        t, p = np.array([], dtype="int64"), np.array([], dtype=float)
    cov = min(cov, lim) if ts else te
    net = WT.net_pct(p, E) if len(t) else np.array([], dtype=float)
    T, HI, LO, CL, OP, BAR = [t], [net], [net], [net], [np.full(len(t), np.nan)], [np.zeros(len(t), bool)]
    end = cov
    if cov < lim:                                              # 1m tail (provisional)
        k = WT.load_1m(pair)
        first = max(-(-int(te) // MIN) * MIN if not len(t) else cov, -(-int(cov) // MIN) * MIN)
        k = k[(k.open_time.values >= first) & (k.open_time.values + MIN <= lim)]
        if len(k):
            o = k.open_time.values.astype("int64")
            gaps = np.flatnonzero(np.diff(o) != MIN)
            if o[0] != first:
                k = k.iloc[:0]
            elif len(gaps):
                k = k.iloc[:gaps[0] + 1]
        if len(k):
            o = k.open_time.values.astype("int64")
            T.append(o + MIN); HI.append(WT.net_pct(k.h.values, E)); LO.append(WT.net_pct(k.l.values, E))
            CL.append(WT.net_pct(k.c.values, E))
            OP.append(WT.net_pct(k.o.values, E) if "o" in k else np.full(len(k), np.nan)); BAR.append(np.ones(len(k), bool))
            end = int(o[-1] + MIN)
    return dict(t=np.concatenate(T), hi=np.concatenate(HI), lo=np.concatenate(LO), cl=np.concatenate(CL), op=np.concatenate(OP),
                bar=np.concatenate(BAR), end_ms=int(end), n_tick=int(len(t)), n_bar=int(sum(len(x) for x in T[1:])), te=int(te))


def evaluate(P, tp, sl, cap):
    """→ dict(why TP/STOP/CAP/UNRES, pct (tick ruler), pct_line (paper ruler), xmin, worst) — first-touch on the net path."""
    te = P["te"]
    n = int(np.searchsorted(P["t"], te + cap * MIN, side="right"))
    hi, lo = P["hi"][:n], P["lo"][:n]
    itp = np.flatnonzero(hi >= tp)
    ist = np.flatnonzero(lo <= -sl) if sl else np.array([], dtype=int)
    a = int(itp[0]) if len(itp) else None
    b = int(ist[0]) if len(ist) else None
    if b is not None and (a is None or b <= a):
        if P["bar"][b]:
            op = P["op"][b]
            v = min(-sl, float(op)) if np.isfinite(op) else -sl
        else:
            v = float(lo[b])
        return dict(why="STOP", pct=v, pct_line=-float(sl), xmin=(P["t"][b] - te) / MIN, worst=float(lo[:b + 1].min()))
    if a is not None:
        v = float(tp) if P["bar"][a] else float(hi[a])
        return dict(why="TP", pct=v, pct_line=float(tp), xmin=(P["t"][a] - te) / MIN, worst=float(lo[:a + 1].min()))
    if P["end_ms"] < te + cap * MIN or n == 0:
        return dict(why="UNRES", pct=np.nan, pct_line=np.nan, xmin=np.nan, worst=float(lo.min()) if n else np.nan)
    return dict(why="CAP", pct=float(P["cl"][n - 1]), pct_line=float(P["cl"][n - 1]), xmin=float(cap), worst=float(lo.min()))


def trough_stats(P, tp=3.0, horizon=H12):
    """no-stop walk: did the fill reach +tp within the horizon; the deepest net before it (winners' trough); MFE / MAE."""
    n = int(np.searchsorted(P["t"], P["te"] + horizon * MIN, side="right"))
    hi, lo = P["hi"][:n], P["lo"][:n]
    i = np.flatnonzero(hi >= tp)
    if len(i):
        a = int(i[0])
        return dict(reach=True, trough_before=float(min(0.0, lo[:a + 1].min())), min_to_reach=(P["t"][a] - P["te"]) / MIN,
                    mae=float(lo.min()) if n else np.nan, mfe=float(hi.max()) if n else np.nan)
    known = P["end_ms"] >= P["te"] + horizon * MIN
    return dict(reach=False if known else None, trough_before=np.nan, min_to_reach=np.nan,
                mae=float(lo.min()) if n else np.nan, mfe=float(hi.max()) if n else np.nan)


# ─────────────────────────── statistics ───────────────────────────
def p_pos(delta, day, n=4000, seed=7):
    """day-clustered bootstrap P(mean Δ > 0) (resample UTC days, pooled mean of their fills)."""
    g = pd.DataFrame(dict(p=delta, d=day)).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return np.nan
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    return float(((s[idx].sum(1) / c[idx].sum(1)) > 0).mean())


def ci(x, day, n=4000, seed=7):
    g = pd.DataFrame(dict(p=x, d=day)).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return (np.nan, np.nan)
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def maxstat_p(D, day, n=4000, seed=11):
    """best-of-N guard: day-clustered max-statistic bootstrap over the K cells of a family.
    D = (n_fills, K) Δ matrix. Resample days, centre each cell's resampled mean at its observed mean, take the max over cells;
    p = share of resamples whose centred max ≥ the observed best mean Δ (Romano-Wolf style single-step)."""
    D = np.asarray(D, float)
    obs = D.mean(0)
    g = pd.DataFrame(D).assign(d=day).groupby("d")
    S, C = g.sum().values, g.size().values
    idx = np.random.default_rng(seed).integers(0, len(C), size=(n, len(C)))
    m = S[idx].sum(1) / C[idx].sum(1)[:, None]                 # n × K resampled means
    mx = (m - obs).max(1)
    return float((mx >= obs.max()).mean()), int(np.argmax(obs))
