#!/usr/bin/env python3
"""yr6 three-arm A/B (2026-10-14, operator-approved): does switching FRENZY_WIDE off ALONE help or hurt the book?

  A = yr6 / yr6a seed 1       LONG ON  · WIDE ON   (frozen_config_yr6_01419d4.json,   83ecb8787a5b)
  B = yr6x / yr6ax seed 1     LONG OFF · WIDE OFF  (frozen_config_yr6x_lwoff.json,    33b5784891bd)  — reports/YR6_LWOFF_AB.md
  C = yr6w / yr6aw seed 1     LONG ON  · WIDE OFF  (frozen_config_yr6w_wideoff.json,  d11dec3bb4e1)
All three: snapshot code_yr6_01419d4, same flags / chunks / cadence seed / caches (the 10-11 chunk of B and C on shadow caches without
the 2026-10-12 tick files written after yr6 ran — see ENGINE_REPLAY_YEAR_PLAN yr6x / yr6w). Fills trimmed by scripts/yr6_tables.py and
RE-PRICED to today's config by scripts/yr6_reprice_today.py run from a git-archive of HEAD 443a7b5 (LONG / WIDE +1.5 at 2× / 1×,
LITE +3 at 10×, WILLY 10×, momentum long × 0.5) — the YR6_LWOFF_AB pricing.

  cd <code_443a7b5 snapshot> && <venv>/bin/python "<repo>/scripts/study_yr6_wideoff_ab.py"

Method = scripts/study_yr6_lwoff_ab.py generalised to any pair of arms (X = more sleeves on, Y = the arm with sleeve set R removed):
  · matching: SHARED = identical (pair, direction, entry_strategy, opened_at); leftovers with the same (pair, direction, strategy) within
    ±--tol s = SHIFTED (one-to-one, nearest); rest X-only / Y-only.
  · decomposition of Y − X (fixed $2,900 book): ① R's own fills removed · ② structural rest (Y-only − X-only + shifted Δ) · ③ shared-fill
    re-exit noise (same entry, different exit — the stepped-1m intra-minute RNG divergence on non-tick pair-days) · CLEANED = ① + ②
    (every shared fill held at its X outcome). Day-block bootstrap 95 % CIs over the regime's calendar days.
  · coin-held attribution: a Y-only fill is "coin-held" when an R fill in X held the same pair at that moment.
Comparisons: C vs A (WIDE off alone; R = WIDE) · C vs B (what LONG alone adds; X = C, Y = B, R = LONG) · B vs A (R = LONG + WIDE —
must reproduce YR6_LWOFF_AB exactly: a code check).
Outputs reports/YR6_WIDEOFF_AB_{tables.txt, sleeves, decomposition, lite, books}.csv.
Second computation: stdlib-csv recount from the raw replay orders CSVs + meta windows (own trim / key / re-price code, no pandas).
"""
import argparse, csv, glob, json, os, sys, warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
REPO = "/Users/guillearslanian/Downloads/NOFA AI"
REP = os.path.join(REPO, "reports")
RAW = os.path.join(REP, "backtest_cache", "replay")
ap = argparse.ArgumentParser()
ap.add_argument("--a", default="ENGINE_REPLAY_YR6LWA_TODAY_fills.csv")
ap.add_argument("--b", default="ENGINE_REPLAY_YR6X_2Y_TODAY_fills.csv")
ap.add_argument("--c", default="ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv")
ap.add_argument("--book", type=float, default=2900.0)
ap.add_argument("--draws", type=int, default=4000)
ap.add_argument("--tol", type=float, default=120.0)
ap.add_argument("--no-book", action="store_true")
A = ap.parse_args()
CODE = os.getcwd()
sys.path.insert(0, CODE)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./_yr6_wideoff_dummy.db"; os.environ["DEBUG"] = "false"
import logging  # noqa: E402
logging.disable(logging.WARNING)
import config  # noqa: E402
from services import trading_engine as te  # noqa: E402
assert os.path.abspath(os.path.dirname(config.__file__)) == os.path.abspath(CODE), "run from the today's-config code dir"
ENG = te.trading_engine; ENG._bnb_burn_rate = 0.0
INV = config.trading_config.investment; TH = config.trading_config.thresholds
FZ_LEV = {"FRENZY_LONG": float(TH.frenzy_long_lev_mult), "FRENZY_WIDE": float(TH.frenzy_wide_lev_mult),
          "FRENZY_LITE": float(TH.frenzy_lite_lev_mult), "FRENZY_WILLY": float(TH.frenzy_willy_lev_mult)}
rng = np.random.default_rng(20261014)
REGIMES = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
RG_SPAN = {"2025 BULL": ("2025-01-01", "2026-01-04"), "2026-H1 BEAR": ("2026-01-04", "2026-07-01"), "2026-H2 BULL": ("2026-07-01", "2026-10-12")}
PREFIX = {"A": ("yr6a", "yr6"), "B": ("yr6ax", "yr6x"), "C": ("yr6aw", "yr6w")}
L = []
P = L.append


def load(fn):
    F = pd.read_csv(os.path.join(REP, fn), low_memory=False)
    F = F[F.seed == 1].copy()
    F["t"] = pd.to_datetime(F.t, format="mixed")
    F["tc"] = pd.to_datetime(F.closed_at.astype(str).str[:23].str.replace("T", " "), format="mixed")
    F["strat"] = F.entry_strategy.fillna("MOMENTUM")
    F["key"] = list(zip(F.pair, F.direction, F.strat, F.opened_at.astype(str)))
    assert not F.key.duplicated().any()
    return F.reset_index(drop=True)


ARM = {"A": load(A.a), "B": load(A.b), "C": load(A.c)}
assert not ARM["B"].sleeve.isin(["FRENZY_LONG", "FRENZY_WIDE"]).any(), "B arm has LONG / WIDE fills"
assert not ARM["C"].sleeve.isin(["FRENZY_WIDE"]).any(), "C arm has WIDE fills — wrong config?"
assert ARM["C"].sleeve.eq("FRENZY_LONG").any(), "C arm has no FRENZY_LONG fills — wrong config?"
P(f"YR6 WIDE-OFF A/B — seed 1 — A {A.a} ({len(ARM['A'])}) · B {A.b} ({len(ARM['B'])}) · C {A.c} ({len(ARM['C'])}) · priced at {CODE}")
P(f"lev mult {FZ_LEV} · momentum long × {float(TH.momentum_long_lev_mult):g} · $ = fixed ${A.book:,.0f} book (usd_new) unless 'book'")


def stats(g):
    if not len(g):
        return dict(N=0, WR=np.nan, avg=np.nan, usd=0.0)
    return dict(N=len(g), WR=(g.pct_new > 0).mean() * 100, avg=g.pct_new.mean(), usd=g.usd_new.sum())


def fmt(n, wr, avg, usd):
    return f"{n:5.0f} · {wr:4.0f}% · {avg:+.3f} · ${usd:+8,.0f}" if n else f"{0:5d} ·    – ·    –   · ${0:+8,.0f}"


def fs(g):
    s = stats(g); return fmt(s["N"], s["WR"], s["avg"], s["usd"])


def regime_days(rg):
    if rg == "ALL":
        return pd.DatetimeIndex(np.concatenate([regime_days(x) for x in REGIMES]))
    lo, hi = RG_SPAN[rg]
    return pd.date_range(lo, pd.Timestamp(hi) - pd.Timedelta(days=1), freq="D")


def sel_rg(F, rg):
    return F if rg == "ALL" else F[F.regime == rg]


def day_boot(g, days):
    d = g.groupby(g.t.dt.floor("D")).agg(s=("pct_new", "sum"), n=("pct_new", "size"), u=("usd_new", "sum")).reindex(days, fill_value=0)
    if d.n.sum() < 3:
        return (np.nan, np.nan), (np.nan, np.nan)
    idx = rng.integers(0, len(d), size=(A.draws, len(d)))
    s, n, u = d.s.values[idx].sum(1), d.n.values[idx].sum(1), d.u.values[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    return tuple(np.nanpercentile(m, [2.5, 97.5])), tuple(np.percentile(u, [2.5, 97.5]))


# ── tick coverage of a fill (every calendar day it was open has a tick file for the pair) ─────────────────────────────────────────
_tick_cache = {}


def tick_covered(pair, t0, t1):
    root = os.path.join(REP, "backtest_cache_2025" if t0 < pd.Timestamp("2026-01-04") else "backtest_cache")
    for d in pd.date_range(t0.floor("D"), t1.floor("D"), freq="D"):
        k = (root, pair, d)
        if k not in _tick_cache:
            _tick_cache[k] = any(os.path.exists(os.path.join(root, sub, pair, f"{d:%Y-%m-%d}.npz")) for sub in ("ticks_q", "ticks"))
        if not _tick_cache[k]:
            return False
    return True


# ── matching ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def match(X, Y):
    X = X.copy(); Y = Y.copy()
    kx, ky = set(X.key), set(Y.key)
    X["m"] = np.where(X.key.isin(ky), "shared", "X-only"); Y["m"] = np.where(Y.key.isin(kx), "shared", "Y-only")
    gy = {k: g for k, g in Y[Y.m == "Y-only"].groupby(["pair", "direction", "strat"])}
    used = set()
    for i, r in X[X.m == "X-only"].iterrows():
        g = gy.get((r.pair, r.direction, r.strat))
        if g is None:
            continue
        dt = (g.t - r.t).abs().dt.total_seconds()
        dt = dt[(dt <= A.tol) & ~dt.index.isin(used)]
        if len(dt):
            j = dt.idxmin(); used.add(j)
            X.at[i, "m"] = "shifted"; Y.at[j, "m"] = "shifted"
    SH = X[X.m == "shared"].merge(Y[Y.m == "shared"][["key", "pct_new", "usd_new", "closed_at"]], on="key", suffixes=("_x", "_y"))
    SH["changed"] = (SH.pct_new_x - SH.pct_new_y).abs() > 1e-6
    return X, Y, SH


def holder_fn(F, sleeves):
    H = F[F.sleeve.isin(sleeves)][["pair", "t", "tc", "sleeve", "peak_pnl", "entry_price", "exit_price", "pct"]]
    by = {p: g for p, g in H.groupby("pair")}

    def held(pair, t):
        g = by.get(pair)
        if g is None:
            return None
        h = g[(g.t <= t) & (g.tc > t)]
        return h.iloc[0] if len(h) else None
    return held


def any_holder_fn(F):
    """who held the coin in arm F at time t (any sleeve, any direction) — for the LITE fills one arm took and the other did not."""
    by = {p: g for p, g in F[["pair", "t", "tc", "sleeve"]].groupby("pair")}

    def held(pair, t, exclude_t=None):
        g = by.get(pair)
        if g is None:
            return ""
        h = g[(g.t <= t) & (g.tc > t)]
        if exclude_t is not None:
            h = h[h.t != exclude_t]
        return ",".join(sorted(set(h.sleeve))) if len(h) else ""
    return held


def _first_touch_before(pair, t0, t1, px):
    """did price print ≥ px in [t0, t1)? Real ticks where present, else 1m bars fully inside the window. None = no data. (=lwoff)"""
    root = os.path.join(REP, "backtest_cache_2025" if t0 < pd.Timestamp("2026-01-01") else "backtest_cache")
    a_ms, z_ms = int(t0.value // 1_000_000), int(t1.value // 1_000_000)
    ticks, seen_any = [], False
    for d in pd.date_range(t0.floor("D"), t1.floor("D"), freq="D"):
        for sub in ("ticks_q", "ticks"):
            f = os.path.join(root, sub, pair, f"{d:%Y-%m-%d}.npz")
            if os.path.exists(f):
                z = np.load(f); m = (z["t"] >= a_ms) & (z["t"] < z_ms); ticks.append(z["p"][m]); seen_any = True
                break
    if seen_any:
        return bool(any(len(x) and float(x.max()) >= px for x in ticks))
    for sub in ("k1m_spike", "k1m"):
        f = os.path.join(root, sub, f"{pair}.csv")
        if os.path.exists(f):
            k = pd.read_csv(f, usecols=["open_time", "h"])
            k = k[(k.open_time >= a_ms) & (k.open_time + 60_000 <= z_ms)]
            if len(k):
                return bool(k.h.max() >= px)
    return None


def today_class(h, r):
    """the holder h (a LONG / WIDE fill, replayed to +3) at today's +1.5 TP: does it still block the LITE fill r?"""
    if h is None:
        return ""
    if float(h.peak_pnl) < 1.5:
        return "still blocked today (holder never reached +1.5)"
    fee = (float(h.exit_price) / float(h.entry_price) - 1) * 100 - float(h.pct)
    hit = _first_touch_before(r.pair, h.t, r.t, float(h.entry_price) * (1 + (1.5 + fee) / 100))
    if hit is None:
        return "unknown (no price data)"
    return "freed today (holder hit +1.5 before the LITE entry)" if hit else "still blocked today (holder hit +1.5 only after)"


CMP_ROWS, DEC_ROWS, LITE_ROWS = [], [], []


def compare(xn, yn, R, label):
    """Y − X where Y = X with sleeve set R switched off."""
    X, Y, SH = match(ARM[xn], ARM[yn])
    held = holder_fn(X, R)
    Y["held_in_X"] = [(h.sleeve if (h := held(r.pair, r.t)) is not None else "") if r.m == "Y-only" else "" for r in Y.itertuples()]
    P(""); P("#" * 150)
    P(f"# {label}: {yn} − {xn}  ({yn} = {xn} with {'+'.join(R)} OFF)")
    P(f"# matching: shared {int((X.m == 'shared').sum())} · shifted (±{A.tol:.0f} s) {int((X.m == 'shifted').sum())} · {xn}-only {int((X.m == 'X-only').sum())} "
      f"(of which {'+'.join(R)} {int((X.m.eq('X-only') & X.sleeve.isin(R)).sum())}) · {yn}-only {int((Y.m == 'Y-only').sum())} · shared re-exited {int(SH.changed.sum())}")
    P("#" * 150)
    sleeves = sorted(set(X.sleeve) | set(Y.sleeve))
    for rg in REGIMES + ["ALL"]:
        P(""); P(f"=== {label} · {rg}: per sleeve  {xn} vs {yn} — N · WR · avg % · $ (fixed book, today's size) ===")
        P(f"{'sleeve':20} │ {xn:^34} │ {yn:^34} │ {yn + '−' + xn + ' $':>8} │ shared │ {xn}-only N/avg/$ │ {yn}-only N/avg/$ │ shifted $")
        for sl in sleeves + ["TOTAL", "FRENZY family"]:
            def s_(F):
                q = sel_rg(F, rg)
                return q if sl == "TOTAL" else (q[q.sleeve.str.startswith("FRENZY")] if sl == "FRENZY family" else q[q.sleeve == sl])
            x, y = s_(X), s_(Y)
            sx, sy = stats(x), stats(y)
            xo, yo = stats(x[x.m == "X-only"]), stats(y[y.m == "Y-only"])
            shd = y[y.m == "shifted"].usd_new.sum() - x[x.m == "shifted"].usd_new.sum()
            CMP_ROWS.append(dict(cmp=label, regime=rg, sleeve=sl, **{f"X_{k}": v for k, v in sx.items()}, **{f"Y_{k}": v for k, v in sy.items()},
                                 d_usd=sy["usd"] - sx["usd"], shared_N=int((x.m == "shared").sum()), Xonly_N=xo["N"], Xonly_avg=xo["avg"], Xonly_usd=xo["usd"],
                                 Yonly_N=yo["N"], Yonly_avg=yo["avg"], Yonly_usd=yo["usd"], shifted_d_usd=shd))
            P(f"{sl:20} │ {fmt(sx['N'], sx['WR'], sx['avg'], sx['usd'])} │ {fmt(sy['N'], sy['WR'], sy['avg'], sy['usd'])} │ {sy['usd'] - sx['usd']:+8,.0f} │ "
              f"{int((x.m == 'shared').sum()):6d} │ {xo['N']:4.0f} {xo['avg'] if xo['N'] else 0:+.2f} {xo['usd']:+7,.0f} │ {yo['N']:4.0f} {yo['avg'] if yo['N'] else 0:+.2f} "
              f"{yo['usd']:+7,.0f} │ {shd:+6,.0f}")

    # decomposition
    P(""); P(f"=== {label} DECOMPOSITION of {yn} − {xn} (fixed book $): ① {'+'.join(R)} own fills removed · ② structural rest ({yn}-only − {xn}-only + "
             f"shifted Δ) · ③ shared re-exit noise · CLEANED = ① + ② · day-bootstrap 95 % CI ===")
    CHG = set(SH[SH.changed].key)
    for rg in REGIMES + ["ALL"]:
        days = regime_days(rg); x = sel_rg(X, rg); y = sel_rg(Y, rg)

        def daily(g):
            return g.groupby(g.t.dt.floor("D")).usd_new.sum().reindex(days, fill_value=0).values
        own = -daily(x[x.sleeve.isin(R)])
        struct = daily(y[y.m.isin(["Y-only", "shifted"])]) - daily(x[x.m.isin(["X-only", "shifted"]) & ~x.sleeve.isin(R)])
        noise = daily(y[y.m == "shared"]) - daily(x[x.m == "shared"])
        idx = rng.integers(0, len(days), size=(A.draws, len(days)))
        ci = lambda v: tuple(np.percentile(v[idx].sum(1), [2.5, 97.5]))
        clean = own + struct; raw = clean + noise
        nsh = int((x.m == "shared").sum()); nchg = int(x[x.m == "shared"].key.isin(CHG).sum())
        r = dict(cmp=label, regime=rg, own_removed=own.sum(), structural=struct.sum(), struct_lo=ci(struct)[0], struct_hi=ci(struct)[1],
                 noise=noise.sum(), noise_lo=ci(noise)[0], noise_hi=ci(noise)[1], clean=clean.sum(), clean_lo=ci(clean)[0], clean_hi=ci(clean)[1],
                 p_clean_lt0=(clean[idx].sum(1) < 0).mean(), raw=raw.sum(), raw_lo=ci(raw)[0], raw_hi=ci(raw)[1], p_raw_lt0=(raw[idx].sum(1) < 0).mean(),
                 shared_n=nsh, shared_changed=nchg)
        DEC_ROWS.append(r)
        P(f"  {rg:13} ① {r['own_removed']:+,.0f} · ② {r['structural']:+,.0f} [{r['struct_lo']:+,.0f}, {r['struct_hi']:+,.0f}] · ③ noise {r['noise']:+,.0f} "
          f"({nchg}/{nsh} shared re-exited) · CLEANED {r['clean']:+,.0f} [{r['clean_lo']:+,.0f}, {r['clean_hi']:+,.0f}] P({yn}<{xn}) {r['p_clean_lt0']:.2f} "
          f"· raw {r['raw']:+,.0f} [{r['raw_lo']:+,.0f}, {r['raw_hi']:+,.0f}] P {r['p_raw_lt0']:.2f}")
        # variant: today's SURGE_LONG EMA13−EMA50 gap gate (HEAD 2338871) — drop SURGE only-in / shifted fills with gap < 0 (shared ones cancel)
        gb = lambda F: F.sleeve.eq("SURGE_LONG") & (pd.to_numeric(F.entry_pair_ema20_ema50_gap_pct, errors="coerce") < 0)
        struct_g = (daily(y[y.m.isin(["Y-only", "shifted"]) & ~gb(y)]) - daily(x[x.m.isin(["X-only", "shifted"]) & ~x.sleeve.isin(R) & ~gb(x)]))
        clean_g = own + struct_g
        r.update(clean_gap=clean_g.sum(), clean_gap_lo=ci(clean_g)[0], clean_gap_hi=ci(clean_g)[1], p_clean_gap_lt0=(clean_g[idx].sum(1) < 0).mean())
        P(f"  {'':13}   with today's SURGE gap gate: ② {struct_g.sum():+,.0f} · CLEANED {r['clean_gap']:+,.0f} [{r['clean_gap_lo']:+,.0f}, "
          f"{r['clean_gap_hi']:+,.0f}] P({yn}<{xn}) {r['p_clean_gap_lt0']:.2f}")
    # noise floor: tick coverage crosstab of the shared fills
    SH["tick"] = [tick_covered(r.pair, r.t, r.tc) for r in SH.itertuples()]
    SH["d_usd"] = SH.usd_new_y - SH.usd_new_x
    P(f"  ③ noise floor — shared fills re-exited, by tick coverage of the fill's pair-days (ticks_q / ticks file for every open day):")
    for rg in REGIMES:
        s = SH[SH.regime == rg]
        for tk, g in s.groupby("tick"):
            c = g[g.changed]
            P(f"    {rg:13} {'tick-covered' if tk else '1m path     '} shared {len(g):5d} · re-exited {len(c):4d} · better {int((c.d_usd > 0).sum())} / worse "
              f"{int((c.d_usd < 0).sum())} · Σ Δ$ {c.d_usd.sum():+,.0f}")
    # ② by sleeve and cause
    P(f"  ② by sleeve (ALL regimes): {yn}-only coin-held (a {'/'.join(R)} held the coin in {xn}) vs other · {xn}-only · shifted Δ")
    for sl in sleeves:
        if sl in R:
            continue
        yo = Y[(Y.sleeve == sl) & (Y.m == "Y-only")]; xo = X[(X.sleeve == sl) & (X.m == "X-only")]
        sd = Y[(Y.sleeve == sl) & (Y.m == "shifted")].usd_new.sum() - X[(X.sleeve == sl) & (X.m == "shifted")].usd_new.sum()
        if not (len(yo) or len(xo) or abs(sd) > 0.5):
            continue
        h = yo[yo.held_in_X != ""]; o = yo[yo.held_in_X == ""]
        P(f"    {sl:14} {yn}-only coin-held {len(h):3d} ({h.pct_new.mean() if len(h) else 0:+.2f} %, ${h.usd_new.sum():+,.0f}) · other {len(o):3d} "
          f"({o.pct_new.mean() if len(o) else 0:+.2f} %, ${o.usd_new.sum():+,.0f}) · {xn}-only {len(xo):3d} ({xo.pct_new.mean() if len(xo) else 0:+.2f} %, "
          f"${xo.usd_new.sum():+,.0f}) · shifted Δ ${sd:+,.0f} · ② ${yo.usd_new.sum() - xo.usd_new.sum() + sd:+,.0f}")
    # LITE detail
    hx_any, hy_any = any_holder_fn(X), any_holder_fn(Y)
    YL = Y[(Y.sleeve == "FRENZY_LITE") & (Y.m == "Y-only")].copy()
    XL = X[(X.sleeve == "FRENZY_LITE") & (X.m == "X-only")].copy()
    YL["holder_in_X"] = [hx_any(r.pair, r.t) for r in YL.itertuples()]
    XL["holder_in_Y"] = [hy_any(r.pair, r.t) for r in XL.itertuples()]
    YL["today_class"] = [today_class(held(r.pair, r.t), r) if r.held_in_X else "" for r in YL.itertuples()]
    P(""); P(f"=== {label} FRENZY_LITE: fills only in {yn} (kept out in {xn}) and only in {xn} — N · WR · avg % [95 % day CI] · $ [95 % day CI] ===")
    for rg in REGIMES + ["ALL"]:
        days = regime_days(rg)
        for sub, g0 in ((f"{yn}-only, {'/'.join(R)} held the coin in {xn}", YL[YL.held_in_X != ""]), (f"{yn}-only, other", YL[YL.held_in_X == ""]),
                        (f"{xn}-only (lost when OFF)", XL)):
            g = sel_rg(g0, rg); st = stats(g); (alo, ahi), (ulo, uhi) = day_boot(g, days)
            LITE_ROWS.append(dict(cmp=label, regime=rg, cohort=sub, **st, avg_lo=alo, avg_hi=ahi, usd_lo=ulo, usd_hi=uhi))
            P(f"  {rg:13} {sub:40} {fmt(st['N'], st['WR'], st['avg'], st['usd'])}   avg CI [{alo:+.3f}, {ahi:+.3f}]  $ CI [{ulo:+,.0f}, {uhi:+,.0f}]")
    ch = YL[YL.held_in_X != ""]
    if len(ch):
        P(f"  at TODAY's LONG / WIDE +1.5 TP (replay held the {xn} holders to +3):")
        for cl, g in ch.groupby("today_class"):
            (alo, ahi), (ulo, uhi) = day_boot(g, regime_days("ALL"))
            P(f"    {cl:52} {fs(g)}  avg CI [{alo:+.3f}, {ahi:+.3f}]  $ CI [{ulo:+,.0f}, {uhi:+,.0f}]  by regime {g.groupby('regime').size().to_dict()}")
    P(f"  every {yn}-only LITE fill (holder of the coin in {xn} at that moment):")
    for r in YL.sort_values("t").itertuples():
        P(f"      {str(r.opened_at)[:16]} {r.pair:14} {r.regime:13} LITE {r.pct_new:+6.2f} % ${r.usd_new:+5,.0f} {r.exit_kind:8} · held in {xn}: "
          f"{r.holder_in_X or '—'}{' · ' + r.today_class if r.today_class else ''}")
    P(f"  every {xn}-only LITE fill (holder of the coin in {yn} at that moment):")
    for r in XL.sort_values("t").itertuples():
        P(f"      {str(r.opened_at)[:16]} {r.pair:14} {r.regime:13} LITE {r.pct_new:+6.2f} % ${r.usd_new:+5,.0f} {r.exit_kind:8} · held in {yn}: {r.holder_in_Y or '—'}")
    out = pd.concat([YL.assign(side=f"{yn}-only"), XL.assign(side=f"{xn}-only")]).drop(columns=["key"])
    out.to_csv(os.path.join(REP, f"YR6_WIDEOFF_AB_lite_{label.replace(' ', '')}.csv"), index=False)
    # mechanism-attributable
    own = -X[X.sleeve.isin(R)].usd_new.sum()
    coin = Y[(Y.m == "Y-only") & (Y.held_in_X != "")]
    mech_replay = own + coin.usd_new.sum()
    lite_today = YL[YL.today_class.astype(str).str.startswith("still blocked")].usd_new.sum()
    mech_today = own + coin[coin.sleeve != "FRENZY_LITE"].usd_new.sum() + lite_today
    P(f"  MECHANISM-ATTRIBUTABLE {yn} − {xn}: ① {own:+,.0f} + coin-held {yn}-only ${coin.usd_new.sum():+,.0f} ({len(coin)} fills: "
      f"{coin.groupby('sleeve').usd_new.agg(['size', 'sum']).round(0).to_dict('index')}) = {mech_replay:+,.0f} (replay) · at today's +1.5 TP "
      f"(LITE still-blocked only ${lite_today:+,.0f}) = {mech_today:+,.0f}")
    # every non-LITE only-in fill (both sides), with the SURGE_LONG EMA13−EMA50 gap gate of HEAD 2338871 (DECISION_LOG 287: refuse when
    # entry_pair_ema20_ema50_gap_pct < 0 — misnamed column, holds EMA13−EMA50; unreadable = fail-open) — the gate is NOT in the 443a7b5
    # pricing, so a SURGE only-in fill with gap < 0 would not exist on today's HEAD.
    P(f"  every non-LITE only-in fill (SURGE rows: gap = entry EMA13−EMA50 %, 'GAP-BLOCKED' = refused by today's SURGE gap gate at 2338871):")
    gcol = "entry_pair_ema20_ema50_gap_pct"
    rows_ = [(r, f"{yn}-only", r.held_in_X or "—") for r in Y[(Y.m == "Y-only") & (Y.sleeve != "FRENZY_LITE")].itertuples()]
    rows_ += [(r, f"{xn}-only", hy_any(r.pair, r.t) or "—") for r in X[(X.m == "X-only") & ~X.sleeve.isin(R) & (X.sleeve != "FRENZY_LITE")].itertuples()]
    surge_gate = {f"{yn}-only": 0.0, f"{xn}-only": 0.0}
    for r, side, hold in sorted(rows_, key=lambda z: (z[0].sleeve, z[0].t)):
        g = pd.to_numeric(pd.Series([getattr(r, gcol, None)]), errors="coerce").iloc[0]
        gate = ""
        if r.sleeve == "SURGE_LONG":
            gate = f" · gap {g:+.2f}" + (" GAP-BLOCKED" if pd.notna(g) and g < 0 else "") if pd.notna(g) else " · gap n/a (fail-open)"
            if pd.notna(g) and g < 0:
                surge_gate[side] += r.usd_new
        if r.sleeve in ("SURGE_LONG", "FLIP-short", "FRENZY_WILLY", "Spike-Fade", "MOM-short") or len(rows_) <= 40:
            P(f"      {side:8} {r.sleeve:13} {str(r.opened_at)[:16]} {r.pair:14} {r.regime:13} {r.pct_new:+6.2f} % ${r.usd_new:+6,.0f} "
              f"· coin held in {'the other arm' if side.startswith(xn) else xn}: {hold}{gate}")
    P(f"  SURGE only-in $ that today's gap gate would remove: {yn}-only ${surge_gate[f'{yn}-only']:+,.0f} · {xn}-only ${surge_gate[f'{xn}-only']:+,.0f} "
      f"→ ② SURGE adjusted by {-(surge_gate[f'{yn}-only'] - surge_gate[f'{xn}-only']):+,.0f}")
    return X, Y, SH, YL, XL


XA_CA, YC_CA, SH_CA, LC_only, LA_only = compare("A", "C", ("FRENZY_WIDE",), "C vs A")
XC_BC, YB_BC, SH_BC, LB_only_vsC, LC_only_vsB = compare("C", "B", ("FRENZY_LONG",), "B vs C")
XA_BA, YB_BA, SH_BA, LB_only_vsA, _ = compare("A", "B", ("FRENZY_LONG", "FRENZY_WIDE"), "B vs A")

# ── LONG-holder blocks must match A: the B-vs-A coin-held LITE fills whose A holder was a LONG should be absent in C too; WIDE-held ones
#    should be present in C (unless C's own LONG took the coin) ─────────────────────────────────────────────────────────────────────
P(""); P("=== LONG-holder blocks in C vs A: the LITE fills B took on a coin a LONG / WIDE held in A — are they in C? ===")
kc = set(ARM["C"].key)
hc_any = any_holder_fn(ARM["C"])
held_A = holder_fn(ARM["A"], ("FRENZY_LONG", "FRENZY_WIDE"))
for r in LB_only_vsA[LB_only_vsA.held_in_X != ""].sort_values("t").itertuples():
    h = held_A(r.pair, r.t)
    inC = r.key in kc
    P(f"  {str(r.opened_at)[:16]} {r.pair:14} LITE {r.pct_new:+6.2f} % · A holder {r.held_in_X:12} · in C: {'YES' if inC else 'no '} · C holder now: "
      f"{hc_any(r.pair, r.t) or '—'}")
chk = LB_only_vsA[LB_only_vsA.held_in_X != ""].copy(); chk["inC"] = chk.key.isin(kc)
P(f"  summary: A holder LONG → in C {int(chk[chk.held_in_X == 'FRENZY_LONG'].inC.sum())}/{int((chk.held_in_X == 'FRENZY_LONG').sum())} (expected 0) · "
  f"A holder WIDE → in C {int(chk[chk.held_in_X == 'FRENZY_WIDE'].inC.sum())}/{int((chk.held_in_X == 'FRENZY_WIDE').sum())}")
la, lc = ARM["A"][ARM["A"].sleeve == "FRENZY_LONG"], ARM["C"][ARM["C"].sleeve == "FRENZY_LONG"]
P(f"  FRENZY_LONG fills A {len(la)} vs C {len(lc)} · shared (exact key) {len(set(la.key) & set(lc.key))} · A-only {len(set(la.key) - set(lc.key))} · "
  f"C-only {len(set(lc.key) - set(la.key))}")
for nm, F, other in (("C-only LONG", lc[~lc.key.isin(set(la.key))], ARM["A"]), ("A-only LONG", la[~la.key.isin(set(lc.key))], ARM["C"])):
    ho = any_holder_fn(other)
    P(f"    {nm}: {fs(F)} · coin held in the other arm by: {pd.Series([ho(r.pair, r.t) or '—' for r in F.itertuples()]).value_counts().to_dict() if len(F) else {}}")

# ── WIDE's own trades in A ────────────────────────────────────────────────────────────────────────────────────────────────────────
P(""); P("=== FRENZY_WIDE's own trades in A (today's pricing: +1.5 TP, 1×) and FRENZY_LONG in A vs C ===")
for rg in REGIMES + ["ALL"]:
    w = sel_rg(ARM["A"], rg); w = w[w.sleeve == "FRENZY_WIDE"]
    (alo, ahi), (ulo, uhi) = day_boot(w, regime_days(rg))
    la_ = sel_rg(ARM["A"], rg); lc_ = sel_rg(ARM["C"], rg)
    P(f"  {rg:13} WIDE (A) {fs(w)} avg CI [{alo:+.3f}, {ahi:+.3f}] $ CI [{ulo:+,.0f}, {uhi:+,.0f}] · exits {w.exit_kind.value_counts().to_dict()} │ "
      f"LONG A {fs(la_[la_.sleeve == 'FRENZY_LONG'])} │ LONG C {fs(lc_[lc_.sleeve == 'FRENZY_LONG'])}")

pd.DataFrame(CMP_ROWS).to_csv(os.path.join(REP, "YR6_WIDEOFF_AB_sleeves.csv"), index=False)
pd.DataFrame(DEC_ROWS).to_csv(os.path.join(REP, "YR6_WIDEOFF_AB_decomposition.csv"), index=False)
pd.DataFrame(LITE_ROWS).to_csv(os.path.join(REP, "YR6_WIDEOFF_AB_lite.csv"), index=False)

# ── engine refusal counters (meta filter_blocks, seed 1, incl. warm-up = upper bound) ──────────────────────────────────────────────
P(""); P("=== engine refusal counters, seed 1 (meta filter_blocks totals incl. warm-up = upper bound) — FRENZY rows ===")
for arm, prefs in PREFIX.items():
    c = defaultdict(int)
    for pre in prefs:
        for f in glob.glob(os.path.join(RAW, f"{pre}_2*_s1_meta.json")):
            if not os.path.basename(f).startswith(pre + "_2"):
                continue
            for row in (json.load(open(f)).get("filter_blocks") or {}).get("rows", []):
                nm = str(row.get("filter", ""))
                if "FRENZY" in nm and not row.get("is_variant"):
                    c[nm] += int(row.get("total") or 0)
    P(f"  {arm}: " + " · ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))


# ── compounding books (engine sizing at today's config) ───────────────────────────────────────────────────────────────────────────
def notional_at(equity, available, r, open_notional=0.0):
    s = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
    clm = FZ_LEV.get(s, float(r.cell_lev_multiplier or 1.0))
    inv, lev, _ = ENG.calculate_position_size(available, r.confidence, total_portfolio=equity, cell_multiplier=float(r.cell_multiplier or 1.0),
                                              cell_lev_multiplier=clm, multiplier_target="both",
                                              base_lev_multiplier=te.momentum_long_lev_factor(TH, s, r.direction))
    if inv <= 0 or lev <= 0:
        return 0.0, 0.0
    inv = ENG.momentum_long_resize("STUDY", inv, lev, available, r.confidence, equity, float(r.cell_multiplier or 1.0), clm, "both", s,
                                   r.direction, te.momentum_long_lev_factor(TH, s, r.direction))[0]   # ⚖️ DL 298: same notional, margin-scaled
    notional = inv * lev
    liq = float(INV.max_notional_pct_of_pair_volume or 0)
    vol = pd.to_numeric(pd.Series([r.get("entry_pair_volume_24h_usd")]), errors="coerce").iloc[0]
    if s.startswith("SPIKE") and INV.spike_lowvol_liq_cap_pct and vol and 0 < vol < INV.spike_lowvol_threshold_usd:
        liq = float(INV.spike_lowvol_liq_cap_pct)
    cap = liq / 100.0 * vol if (liq > 0 and vol and vol > 0) else None
    if INV.max_notional_hard_ceiling:
        cap = INV.max_notional_hard_ceiling if cap is None else min(cap, INV.max_notional_hard_ceiling)
    if cap is not None and notional > cap:
        notional = cap
    if INV.max_gross_leverage:
        notional = min(notional, max(0.0, equity * INV.max_gross_leverage - open_notional))
    inv = notional / lev
    if inv < INV.min_investment_size:
        return 0.0, 0.0
    return notional, inv


def book(F, bal0):
    F = F.sort_values("t").reset_index(drop=True)
    ev = []
    for i, r in F.iterrows():
        ev.append((r.t, 1, i)); ev.append((max(r.tc, r.t + pd.Timedelta(milliseconds=1)), 0, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    bal = bal0; openp = {}; curve = [(F.t.min(), bal0)]; skipped = 0
    for ts, kind, i in ev:
        r = F.loc[i]
        if kind == 1:
            om = sum(p[1] for p in openp.values()); on = sum(p[0] for p in openp.values())
            n, inv = notional_at(bal, bal - om, r, on)
            if n <= 0:
                skipped += 1; continue
            openp[i] = (n, inv)
        else:
            p = openp.pop(i, None)
            if p is not None:
                bal += float(r.pct_new) / 100.0 * p[0]; curve.append((ts, bal))
    c = pd.DataFrame(curve, columns=["ts", "bal"])
    mdd = ((c.bal / c.bal.cummax()) - 1).min() * 100
    return dict(end=round(bal, 2), mdd_pct=round(mdd, 1), skipped=skipped, n=len(F))


if not A.no_book:
    P(""); P(f"=== shared COMPOUNDING book from ${A.book:,.0f}, today's sizing (engine calculate_position_size, gross / liquidity caps, margin locked) ===")
    BK = []
    for per, lo, hi in (("2025", "2025-01-01", "2026-01-04"), ("2026", "2026-01-04", "2026-10-12"), ("2026-H1", "2026-01-04", "2026-07-01"),
                        ("2026-H2", "2026-07-01", "2026-10-12")):
        for scope in ("FRENZY only", "FULL"):
            out = {}
            for arm, F in ARM.items():
                G = F[(F.t >= lo) & (F.t < hi)]
                if scope == "FRENZY only":
                    G = G[G.sleeve.str.startswith("FRENZY")]
                out[arm] = book(G, A.book)
            BK.append(dict(period=per, scope=scope, **{f"{a}_{k}": v for a in out for k, v in out[a].items()}))
            P(f"  {per:8} {scope:12} " + " │ ".join(f"{a} ${out[a]['end']:>9,.0f} (mdd {out[a]['mdd_pct']:.0f} %, skip {out[a]['skipped']})" for a in out))
    pd.DataFrame(BK).to_csv(os.path.join(REP, "YR6_WIDEOFF_AB_books.csv"), index=False)

# ── VERIFICATION 2: stdlib recount from the raw replay orders (own trim / key / re-price code) ───────────────────────────────────
P(""); P("=== verification 2: stdlib recount from raw replay orders CSVs + meta windows (no pandas) vs the pipeline ===")


def raw_fills(prefixes):
    out = {}
    for pre in prefixes:
        for mf in glob.glob(os.path.join(RAW, f"{pre}_2*_s1_meta.json")):
            tag = os.path.basename(mf)[:-10]
            if not tag.startswith(pre + "_2"):
                continue
            m = json.load(open(mf)); a, z = m["start_ms"], m["end_ms"]
            for r in csv.DictReader(open(os.path.join(RAW, f"{tag}_orders.csv"), newline="")):
                if r["status"] != "CLOSED" or "PROBE" in (r.get("cell_multiplier_source") or "") or r.get("entry_strategy") == "MANUAL":
                    continue
                ts = r["opened_at"][:23].replace("T", " ")
                t = datetime.strptime(ts + (".000" if len(ts) == 19 else ""), "%Y-%m-%d %H:%M:%S.%f")
                ms = (t - datetime(1970, 1, 1)).total_seconds() * 1000
                if not (a <= ms < z):
                    continue
                k = (r["pair"], r["direction"], r.get("entry_strategy") or "MOMENTUM", r["opened_at"])
                assert k not in out
                out[k] = r
    return out


def pct_today(r):   # independent re-price rule: LONG / WIDE first touch of +1.5 on the replayed peak
    s = r.get("entry_strategy") or ""; p = float(r["pnl_percentage"])
    if s in ("FRENZY_LONG", "FRENZY_WIDE"):
        pk = r.get("peak_pnl")
        if pk not in (None, "") and float(pk) >= 1.5:
            return 1.5
    return p


RAWS = {arm: raw_fills(p) for arm, p in PREFIX.items()}
for arm in "ABC":
    R_, F = RAWS[arm], ARM[arm]
    bad = 0
    by = defaultdict(list)
    for k, r in R_.items():
        by[r.get("entry_strategy") or "MOMENTUM"].append(pct_today(r))
    for s in sorted(by):
        v = by[s]; g = F[F.strat == s].pct_new
        ok = len(v) == len(g) and abs(np.mean(v) - g.mean()) < 1e-6
        bad += (not ok)
        if not ok or s.startswith("FRENZY"):
            P(f"  {arm} {s:14} raw N {len(v):5d} avg {np.mean(v):+.4f} │ pipeline N {len(g):5d} avg {g.mean():+.4f} {'OK' if ok else 'MISMATCH'}")
    P(f"  {arm}: raw trimmed fills {len(R_)} (pipeline {len(F)}) · key sets equal: {set(R_) == set(F.key)} · strategies mismatching {bad}")
# raw-key recount of the C-vs-A LITE only-in sets
rc, ra = RAWS["C"], RAWS["A"]
c_only = [pct_today(rc[k]) for k in rc if k not in ra and rc[k].get("entry_strategy") == "FRENZY_LITE"]
a_only = [pct_today(ra[k]) for k in ra if k not in rc and ra[k].get("entry_strategy") == "FRENZY_LITE"]
P(f"  C-vs-A LITE exact-key only-in: raw C-only N {len(c_only)} avg {np.mean(c_only) if c_only else float('nan'):+.4f} · raw A-only N {len(a_only)} "
  f"avg {np.mean(a_only) if a_only else float('nan'):+.4f} │ pipeline C-only {len(LC_only)} + shifted {int((YC_CA.m.eq('shifted') & YC_CA.sleeve.eq('FRENZY_LITE')).sum())} "
  f"· A-only {len(LA_only)} + shifted {int((XA_CA.m.eq('shifted') & XA_CA.sleeve.eq('FRENZY_LITE')).sum())}")
for nm, X, Y in (("C − A", XA_CA, YC_CA), ("B − C", XC_BC, YB_BC), ("B − A", XA_BA, YB_BA)):
    tot = Y.usd_new.sum() - X.usd_new.sum()
    parts = (Y[Y.m == "Y-only"].usd_new.sum() - X[X.m == "X-only"].usd_new.sum() + Y[Y.m == "shifted"].usd_new.sum() - X[X.m == "shifted"].usd_new.sum()
             + Y[Y.m == "shared"].usd_new.sum() - X[X.m == "shared"].usd_new.sum())
    P(f"  identity {nm} ${tot:+,.2f} = only-in + shifted Δ + shared Δ ${parts:+,.2f}")
P("  (B − A must reproduce reports/YR6_LWOFF_AB.md: raw +$540, cleaned +$2,697, 511 / 513 shared re-exited, 21 coin-held LITE −$2,161)")
txt = "\n".join(L)
open(os.path.join(REP, "YR6_WIDEOFF_AB_tables.txt"), "w").write(txt + "\n")
print(txt)
