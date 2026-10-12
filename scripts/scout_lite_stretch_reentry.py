#!/usr/bin/env python3
"""📐🔁 Scout — LITE_STRETCH_REENTRY observe line (pre-registered 2026-10-12, DECISION_LOG 272; OBSERVE only — never changes config, no bot
API, NO network: the orders exports + the shared tick / 1m / 5m caches other lines fill).

ZONE (FROZEN)  a FRENZY_LITE fill whose pair EMA13 − EMA50 gap at entry is > GAP_CUT (1.87 %, strictly) AND which is NOT the first non-WILLY
               FRENZY entry of its episode (k_now ≥ 2).
               · gap = REBUILT FIRST (272 review I1 — the 1.87 cut was set on the study's rebuilt closed-bar gap): the engine's EMA
                 (services/indicators.py: ta EMAIndicator = ewm(span, adjust=False)) on the last 300 CLOSED 5m closes ending at the signal
                 bar = entry_close − 5m (catch-up bars honoured, study_b19_fzentry_common.features_at's gap13_50), all 300 present. Fallback
                 (labelled ˢ): the order column entry_pair_ema20_ema50_gap_pct — MISNAMED, it holds (EMA13 − EMA50) / EMA50 × 100 (engine
                 pair_entry_stamps; never renamed) and includes the FORMING bar (B19: SD 0.22, max +0.67 vs the rebuild). Neither → UNSCORED
                 (never counted as rest). Both values are stored; the report prints how many fills sit on different sides of the cut.
               · episode = (pair, entry_frenzy_spike_at); k_now = 1 + the earlier non-WILLY FRENZY fills (LONG / WIDE / LITE, any status,
                 any export, dedup (opened_at[:19], pair, entry_strategy)) of the SAME episode whose entry bar close is strictly earlier
                 (entry_close = ⌊opened / 5m⌋·5m − catchup bars·5m — scripts/study_b19_rerun_fzentry_data.py episode_cols / live()). A fill
                 without a spike stamp → UNSCORED.
EVIDENCE       reports/B19_RERUN_VERIFY_B.md "The LITE separator" (study scripts/study_b19_rerun_fzentry_screen.py, backtest year, stop
               regime P = −3): zone 26 fills · 25 days · 26 pairs · 19 % WR · −1.86 %/fill vs +0.37 outside. Threshold frozen on the first
               half: H1 13 · −2.10, H2 14 · −1.72. Welch |t| 4.52 vs the day-block null's 95th 4.41 — a borderline SCREEN ARTEFACT that
               fails the study's own floor (27 for LITE; pooled t −3.75 below the null) and sits near v1's refuted "k-th entry" theme →
               OBSERVE ONLY (CLAUDE.md OBSERVE-FIRST RULE).
COHORT         CLOSED FRENZY_LITE LONG fills opened ≥ the "(DECISION_LOG 250)" commit (the fixed +3 / −3 exit went live; git unavailable →
               the pinned 2026-10-08 13:37 UTC), dedup (opened_at[:19], pair, entry_strategy) — never id; CLOSED beats OPEN; MANUAL_* closes
               excluded (listed). DAY units.
PRICING        every fill (zone and rest) RE-WALKED under LITE's live exit (scout_willy_hold.exit_rule('FRENZY_LITE') = frenzy_tp_pct /
               frenzy_stop_pct / frenzy_max_hold_minutes: +1.5 / −3 / 12 h since DECISION_LOG 283 (+3 / −3 before), the same stop in paper and
               live since DECISION_LOG 271; fills opened before the DL 283 deploy traded +3 → their walker parity misses on TP exits — informational)
               from the fill's own entry with the ticks-first ruler (scout_frenzy_willy_exit.load_path / evaluate, DECISION_LOG 272: ticks,
               then the 1m tail; tick ruler = first print through the line; bot accounting). Walker parity vs the actual exit printed.
               A fill whose exit is not decided on the cached path is UNPRICED (listed; holds the freeze for as long as it is unpriced —
               272 review M2, the safer option). A fill with no gap rebuild yet (unscored, or on the stamp fallback) holds it for 3 days.
FROZEN BAR     (pre-registered 2026-10-12 — never changed later; written in the report output too)
               At the first prefix (by open time, pair) of priced, scored fills where the zone reaches N ≥ 15 on ≥ 8 distinct days, ONCE:
               ① zone avg ≥ rest avg → RETIRED;
               ② else FILTER CANDIDATE (operator decides — the 30–50 % in-sample haircut then applies; never armed automatically) iff
                  zone WR < LITE breakeven WR = |avg loss| / (avg win + |avg loss|) on the kept (priced) LITE fills ≤ the prefix end
                  ∧ day-clustered bootstrap P(zone mean < 0) ≥ 0.95 (4,000 resamples, seed 7)
                  ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss;
               ③ else KEEP OBSERVING.
               Deferred while a LITE fill opened ≤ the prefix end is OPEN, unpriced, or priced with a 1m part while its tick archive is
               still pending (NF.freeze_hold). State reports/SCOUT_LITE_STRETCH_REENTRY.json · rows reports/SCOUT_LITE_STRETCH_REENTRY.csv.
Usage: venv/bin/python scripts/scout_lite_stretch_reentry.py [--selftest]
"""
import glob
import os
import tempfile
import time

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, gross-loss share, breakeven, freeze hold, state I/O)
import scout_frenzy_willy_exit as FWX               # noqa: E402  (the 272 hybrid walker, deploy floor)
import scout_willy_hold as WH                       # noqa: E402  (LITE's live exit from trading_config.json)

WT = FWX.WT
# FROZEN 2026-10-12 (DECISION_LOG 272) — never re-fit
GAP_CUT, K_MIN = 1.87, 2                            # zone = gap > 1.87 % (strict) ∧ k_now ≥ 2
STAMP_BAND = 0.67                                  # (272 deep review I-A) a stamp-fallback gap within ±0.67 of the cut (the B19 max stamp − rebuild gap) → UNSCORED
N_MIN, DAYS_MIN = 15, 8
P_MIN, SHARE_MAX = 0.95, 0.50
BOOT_N, BOOT_SEED = 4000, 7
EMA_BARS = 300                                      # closes the gap rebuild reads (study features_at)
M5 = 300_000
STRATEGY = "FRENZY_LITE"
NON_WILLY = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "pnl", "notional_value",
        "close_reason", "entry_pair_ema20_ema50_gap_pct", "entry_frenzy_spike_at", "entry_frenzy_catchup_bars")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, "reports", "backtest_cache")
K5_DIRS = ("k5m_full", "b19_5m", os.path.join("negflank2d_ext", "5m"), os.path.join("scout_ml_trend", "5m"), os.path.join("scout_lite_off30", "5m"))
STATE = os.path.join(_ROOT, "reports", "SCOUT_LITE_STRETCH_REENTRY.json")
ROWS_CSV = os.path.join(_ROOT, "reports", "SCOUT_LITE_STRETCH_REENTRY.csv")
EVIDENCE = ("backtest year, stop −3 (reports/B19_RERUN_VERIFY_B.md 'The LITE separator'): zone 26 fills · 25 days · 26 pairs · 19 % WR · "
            "−1.86 %/fill vs +0.37 outside; frozen on H1: H1 13 · −2.10 / H2 14 · −1.72; Welch |t| 4.52 vs null 95th 4.41 — a borderline "
            "screen artefact that fails the study's own floor → observe only")
BAR = (f"FROZEN bar (pre-registered 2026-10-12, DECISION_LOG 272, never changed): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ "
       f"{DAYS_MIN} distinct days — zone avg ≥ rest avg → RETIRED; else FILTER CANDIDATE (operator decides; then the 30–50 % haircut; never "
       f"armed automatically) iff zone WR < LITE breakeven WR (|avg loss| / (avg win + |avg loss|) on the kept LITE fills) ∧ day-clustered "
       f"P(zone mean < 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no single day or pair ≥ {SHARE_MAX * 100:.0f} % of the zone loss; "
       f"else KEEP OBSERVING. No rest fill in the prefix → the RETIRED check is skipped (the candidate bar decides). Breakeven WR = all "
       f"priced LITE fills ≤ the prefix end (zone included). Gap = closed-bar rebuild; the engine stamp only as fallback and only when it "
       f"sits > {STAMP_BAND:g} pts from the cut, else UNSCORED. An unpriceable fill (1m page given up, no tick day due) is released and listed.")
REVERT = ("Pre-committed revert if ever armed: the first 10 blocked LITE re-entries re-priced with the live exit → WR ≥ LITE breakeven or "
          "Σ > 0 → switch it off.")


# ─────────────────────────── orders ───────────────────────────
def _read_exports():
    fr = []
    for f in glob.glob(EXPORT_GLOB):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    return fr


def raw_orders():
    fr = _read_exports()
    return pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable") if fr else pd.DataFrame(columns=list(COLS))


def family(o):
    """rows in precedence order (last wins; CLOSED beats OPEN) → dedup (opened_at[:19], pair, entry_strategy) → every non-WILLY FRENZY LONG
    fill, any status, prepared (ts, te, entry_close, ep, gap, E, pct, …)."""
    o = o.reindex(columns=list(dict.fromkeys(list(COLS) + [c for c in o.columns if c not in COLS]))).copy()
    if len(o):
        o = o.assign(_k=o.opened_at.astype(str).str[:19], _c=o.status.astype(str).str.upper().eq("CLOSED"))
        o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
        o = o[(o.direction.astype(str) == "LONG") & o.entry_strategy.astype(str).isin(NON_WILLY)].copy()
    o["ts"] = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    o = o[o.ts.notna()].copy()
    tf = pd.to_datetime(o.opened_at.astype(str), format="ISO8601", errors="coerce")
    o["te"] = np.where(tf.notna(), tf.values, o.ts.values).astype("datetime64[ms]").astype("int64") if len(o) else np.array([], dtype="int64")
    cu = pd.to_numeric(o.entry_frenzy_catchup_bars, errors="coerce").fillna(0).astype(int)
    o["entry_close"] = (o.ts.values.astype("datetime64[ms]").astype("int64") // M5) * M5 - cu.values * M5 if len(o) else np.array([], dtype="int64")
    sp = pd.to_datetime(o.entry_frenzy_spike_at.astype(str).str[:19].where(o.entry_frenzy_spike_at.notna(), None), format="ISO8601", errors="coerce")
    o["ep"] = np.where(sp.notna(), o.pair.astype(str) + "|" + sp.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna(""), None)
    o["pair"] = o.pair.astype(str)
    o["strat"] = o.entry_strategy.astype(str)
    o["closed"] = o.status.astype(str).str.upper().eq("CLOSED")
    o["E"] = pd.to_numeric(o.entry_price, errors="coerce")
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["usd"] = pd.to_numeric(o.pnl, errors="coerce")
    o["notional"] = pd.to_numeric(o.notional_value, errors="coerce")
    o["why"] = o.close_reason.astype(str)
    o["gap_stamp"] = pd.to_numeric(o.entry_pair_ema20_ema50_gap_pct, errors="coerce")
    o["day"] = o.ts.dt.strftime("%Y-%m-%d")
    return o.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True)


def k_now(fills, fam):
    """1 + earlier same-episode non-WILLY FRENZY fills (entry_close strictly earlier), None when the fill has no episode stamp."""
    by = {k: g.entry_close.values for k, g in fam[fam.ep.notna()].groupby("ep")}
    return [None if r.ep is None or (isinstance(r.ep, float) and np.isnan(r.ep)) else 1 + int((by.get(r.ep, np.array([])) < r.entry_close).sum())
            for r in fills.itertuples()]


def open_lite_ts():
    """OPEN FRENZY_LITE LONG positions in the NEWEST export (by mtime) → [(opened_at, 'OPEN <pair>')]."""
    fs = sorted(glob.glob(EXPORT_GLOB), key=os.path.getmtime)
    if not fs:
        return []
    try:
        d = pd.read_csv(fs[-1], low_memory=False, usecols=lambda c: c in COLS)
    except Exception:
        return []
    if not {"opened_at", "status", "direction", "entry_strategy", "pair"} <= set(d.columns):
        return []
    d = d[(d.status.astype(str).str.upper() == "OPEN") & (d.direction.astype(str) == "LONG") & (d.entry_strategy.astype(str) == STRATEGY)]
    t = pd.to_datetime(d.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    return [(a, f"OPEN {p}") for a, p in zip(t, d.pair.astype(str)) if pd.notna(a)]


# ─────────────────────────── gap (stamp, else the engine EMA on cached closed 5m bars) ───────────────────────────
_K5 = {}


def load_5m(pair):
    if pair in _K5:
        return _K5[pair]
    fr = []
    for d in K5_DIRS:
        p = os.path.join(_CACHE, d, f"{pair}.csv")
        if os.path.exists(p):
            try:
                x = pd.read_csv(p, usecols=["open_time", "c"]).apply(pd.to_numeric, errors="coerce").dropna()
                fr.append(x.assign(open_time=x.open_time.astype("int64")))
            except Exception:
                continue
    k = (pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True) if fr
         else pd.DataFrame({"open_time": pd.Series(dtype="int64"), "c": pd.Series(dtype=float)}))
    if len(_K5) > 24:
        _K5.clear()
    _K5[pair] = k
    return k


def gap_rebuild(k5, entry_close):
    """(EMA13 − EMA50) / EMA50 × 100 on the last EMA_BARS closed 5m closes ending at the signal bar — open = entry_close − 5m, the study's
    entry bar close with catch-up bars honoured (272 review M3); every bar of the window must be present, else None."""
    sig = int(entry_close) - M5
    w = k5[(k5.open_time.values > sig - EMA_BARS * M5) & (k5.open_time.values <= sig)]
    if len(w) != EMA_BARS or int(w.open_time.values[-1]) != sig:
        return None
    c = pd.Series(w.c.values.astype(float))
    e13, e50 = float(c.ewm(span=13, adjust=False).mean().iloc[-1]), float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    return (e13 - e50) / e50 * 100 if e50 else None


GAP_PENDING_MS = 3 * 86_400_000                                   # a lagging 5m cache may still score the fill

def pick_gap(rebuilt, stamp):
    """(272 review I1) the 1.87 cut was set on the study's REBUILT closed-bar gap; the engine stamp includes the forming bar (B19: SD 0.22,
    max +0.67 vs the rebuild). → (gap, source): the rebuild first, the stamp only as a labelled fallback, else NaN / 'none'."""
    g, src = [], []
    for a, b in zip(np.asarray(rebuilt, dtype=float), np.asarray(stamp, dtype=float)):
        if not np.isnan(a):
            g.append(a), src.append("rebuilt")
        elif not np.isnan(b) and abs(b - GAP_CUT) > STAMP_BAND:      # the forming-bar stamp only where it cannot sit on the wrong side
            g.append(b), src.append("stamp")
        else:
            g.append(np.nan), src.append("none")
    return g, src


def side_flips(rebuilt, stamp):
    """fills with both measures whose side of the cut differs (rebuild > 1.87 vs stamp > 1.87)."""
    a, b = np.asarray(rebuilt, dtype=float), np.asarray(stamp, dtype=float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return int(((a[m] > GAP_CUT) != (b[m] > GAP_CUT)).sum()), int(m.sum())


def group(gap, k):
    """→ 'zone' / 'rest' / 'unscored' (FROZEN: gap > 1.87 strictly ∧ k_now ≥ 2)."""
    if gap is None or k is None or pd.isna(gap) or pd.isna(k):     # caveman review: a missing spike stamp reads back as NaN in the
        return "unscored"                                          # float k_now column — never "rest"
    return "zone" if (gap > GAP_CUT and k >= K_MIN) else "rest"


# ─────────────────────────── pricing ───────────────────────────
def lite_rule(th=None):
    """LITE's live exit (tp %, stop % positive or None, cap min) from trading_config.json (scout_willy_hold.exit_rule)."""
    tp, st, cap = WH.exit_rule(STRATEGY, FWX._ns(th))
    return float(tp), (abs(float(st)) if st else None), int(cap)


def price(r, rule, tcache=None):
    """→ dict(walk, walk_why, walk_min, src, par_ok) — the 272 hybrid walk from the fill's own entry; walk NaN when undecided."""
    tp, st, cap = rule
    P = FWX.load_path(r.pair, int(r.te), float(r.E), cap, tcache)
    ev = FWX.evaluate(P, tp, st, cap)
    if ev["why"] == "UNRES":
        return dict(walk=np.nan, walk_why="UNRES", walk_min=np.nan, src=None, par_ok=False)
    ok, _ = WT.parity(r.why, r.pct, {"v": {"LIVE": ev}})
    src = "ticks" if (P["n_tick"] > 0 and not ev["on_bar"]) else FWX.path_src(P)   # (review M8) decided on a tick print = final
    return dict(walk=float(ev["pct"]), walk_why=ev["why"], walk_min=float(ev["xmin"]), src=src, par_ok=bool(ok))


def pending(src, ticks_pend, given_up=False):
    """(272 review M2) an UNPRICED fill (src None) holds the freeze for as long as it is unpriced (the safer option: an undecided exit
    could be a zone loss); a fill with a 1m part (provisional) holds while a tick archive can still arrive; decided on ticks = final."""
    if not isinstance(src, str):
        return not given_up                            # (deep review M-1) released once its 1m page is given up and no tick day is due
    return bool(src != "ticks" and ticks_pend)


# ─────────────────────────── verdict / freeze ───────────────────────────
def verdict(zone, rest, be):
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, wr = float(zone.walk.mean()), 100.0 * float((zone.walk > 0).mean())
    mr = float(rest.walk.mean()) if len(rest) else float("nan")
    p = NF.p_mean_neg(zone.walk.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.walk.values, zone.day.values), NF.gross_loss_share(zone.walk.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0f} % vs breakeven " + ("n/a" if be is None else f"{be:.1f} %") + f" · avg {mz:+.3f} % vs rest "
           f"{mr:+.3f} % (N {len(rest)}) · P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair "
           f"{sp * 100:.0f} % of the zone loss")
    if len(rest) and mz >= mr:
        return "RETIRED", det
    if be is not None and wr < be and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return "FILTER CANDIDATE (operator decides; 30–50 % haircut; never armed automatically)", det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """priced scored fills by (open time, pair) → the prefix ending at the zone fill where the zone first reaches N ≥ 15 on ≥ 8 days."""
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(z.grp.values, z.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, sc, kept, now_iso, hold=None, why=None):
    """ONCE at the first crossing; never recomputed. kept = every priced LITE fill (the breakeven pool). → (st, changed)."""
    if "first" in st:
        return st, False
    pre = crossing_prefix(sc)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    be = NF.breakeven_wr(kept[kept.ts <= last].walk.values)
    state, det = verdict(pre[pre.grp == "zone"], pre[pre.grp == "rest"], be)
    st["first"] = dict(state=state, detail=det, be=None if be is None else round(be, 2), at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                       n=int((pre.grp == "zone").sum()), days=int(pre[pre.grp == "zone"].day.nunique()),
                       keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair, pre.grp)])
    return st, True


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Group | N | days | pairs | WR | avg % (walked) | Σ % | Σ$ at the fill's size | worst |", "|---|---|---|---|---|---|---|---|---|"]


def _row(lab, g):
    g = g[g.walk.notna()] if len(g) else g
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {g.pair.nunique()} | {(g.walk > 0).mean() * 100:.0f} % | {g.walk.mean():+.3f} % | "
            f"{g.walk.sum():+.2f} % | {(g.walk * g.notional / 100).sum():+,.0f} | {g.walk.min():+.2f} % |")


def _store(rows, path):
    """merge into the CSV store (key opened_at|pair; this run's row wins) — a record of every fill scored (the verdict is built from the current exports, like the sibling lines)."""
    cols = ["k", "pair", "day", "gap", "gap_src", "gap_rebuilt", "gap_stamp", "ep", "k_now", "grp", "live_pct", "live_why", "walk", "walk_why", "walk_min", "src", "par_ok"]
    new = rows.reindex(columns=cols)
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            new = pd.concat([old.reindex(columns=cols), new], ignore_index=True).drop_duplicates(["k", "pair"], keep="last")
        except Exception:
            pass
    tmp = f"{path}.{os.getpid()}.tmp"
    new.sort_values(["k", "pair"]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def run(now_ms=None, orders=None, state_path=None, rows_path=None, open_ts=None, th=None, deploy=None):
    """→ markdown lines (the caller wraps it in its own try). Cache only — no network."""
    now_ms = now_ms or int(time.time() * 1000)
    state_path, rows_path = state_path or STATE, rows_path or ROWS_CSV
    WT._K1.clear()
    _K5.clear()
    fam = family(raw_orders() if orders is None else orders)
    floor = FWX.deploy_ts() if deploy is None else pd.Timestamp(deploy)
    if open_ts is None:
        try:
            open_ts = open_lite_ts() if orders is None else []
        except Exception:
            open_ts = []
    rule = lite_rule(th)
    c = fam[(fam.strat == STRATEGY) & fam.closed & (fam.ts >= floor) & fam.pct.notna() & (fam.E > 0)].copy()
    man = c[c.why.str.upper().str.startswith("MANUAL")]
    c = c[~c.why.str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    c["k_now"] = k_now(c, fam) if len(c) else []
    c["gap_rebuilt"] = [np.nan if (g := gap_rebuild(load_5m(r.pair), r.entry_close)) is None else g for r in c.itertuples()]
    c["gap"], c["gap_src"] = pick_gap(c.gap_rebuilt.values, c.gap_stamp.values)
    c["grp"] = [group(g, k) for g, k in zip(c.gap, c.k_now)]
    tc = {}
    pr = [price(r, rule, tc) for r in c.itertuples()]
    for k in ("walk", "walk_why", "walk_min", "src", "par_ok"):
        c[k] = [x[k] for x in pr]
    done = WT.load_attempts()["done"]
    _tp = [WT.ticks_pending(r, rule[2], done, now_ms, tc) for r in c.itertuples()]
    c["pend"] = [pending(s, t, given_up=(not t and WT._kk(r.pair, r.te, WT._span(rule[2])) in done))
                 for s, t, r in zip(c.src, _tp, c.itertuples())]
    priced = c[c.walk.notna()]
    sc = priced[priced.grp != "unscored"]
    zone, rest, uns = sc[sc.grp == "zone"], sc[sc.grp == "rest"], c[c.grp == "unscored"]
    be_now = NF.breakeven_wr(priced.walk.values) if len(priced) else None
    if len(c):
        _store(c.assign(k=c.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), live_pct=c.pct, live_why=c.why), rows_path)
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        hold = list(open_ts) + [(r.ts, f"{'unpriced' if not isinstance(r.src, str) else 'provisional'} {r.pair}") for r in c.itertuples() if r.pend]
        hold += [(r.ts, f"gap pending {r.pair}") for r in c.itertuples()          # caveman review: a fill unscored only because the 5m cache
                 if r.gap_src in ("none", "stamp") and pd.notna(r.k_now)                 # lags may still score (or replace the stamp
                 and now_ms - r.ts.value // 1_000_000 < GAP_PENDING_MS]                  # fallback with the rebuild — review I1)
        st, changed = freeze(st, sc, priced, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if changed:
            NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest, be_now)
    lab = f"+{rule[0]:g} / " + (f"−{rule[1]:g}" if rule[1] else "no stop") + f" / {rule[2] / 60:g} h"
    L = ["## 📐🔁 LITE_STRETCH_REENTRY — FRENZY_LITE stretched re-entry (pre-registered 2026-10-12, DECISION_LOG 272, OBSERVE only)", "",
         f"Zone (frozen): pair EMA13 − EMA50 gap > {GAP_CUT:g} % at entry — closed-bar rebuild from cached 5m bars FIRST, the engine stamp "
         f"entry_pair_ema20_ema50_gap_pct (misnamed, holds EMA13; ˢ) only as the fallback, neither → UNSCORED) ∧ k_now ≥ {K_MIN} (not the first non-WILLY FRENZY entry of its episode). "
         f"Cohort: CLOSED FRENZY_LITE fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC, every fill re-walked under LITE's live exit ({lab}) with "
         f"the ticks-first ruler (DECISION_LOG 272). DAY units.", "",
         *HDR, _row("**zone (stretched re-entry)**", zone), _row(f"rest (gap ≤ {GAP_CUT:g} % or first entry)", rest),
         _row("UNSCORED (no gap / no spike stamp) — priced, never counted", uns), ""]
    if len(c):
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} [{r.grp}] gap {('–' if np.isnan(r.gap) else f'{r.gap:+.2f}')}"
            f"{'ˢ' if r.gap_src == 'stamp' else ''} (stamp {('–' if np.isnan(r.gap_stamp) else f'{r.gap_stamp:+.2f}')}) k {int(r.k_now) if pd.notna(r.k_now) else '–'} · live {r.pct:+.2f} → walked "
            + (f"{r.walk:+.2f} ({r.walk_why} @ {r.walk_min:.0f} min) [{r.src}{'ᵖ' if r.src != 'ticks' else ''}]" if r.walk == r.walk else "UNPRICED")
            for r in c.itertuples()))
        pz = priced
        L.append(f"Walker parity (walked vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(pz.par_ok.sum())}/{len(pz)} · "
                 f"unpriced {int(c.walk.isna().sum())} · gap = the closed-bar rebuild, ˢ = engine-stamp fallback (no rebuild) · ᵖ = provisional "
                 f"1m part.")
        nf, nb = side_flips(c.gap_rebuilt.values, c.gap_stamp.values)
        L.append(f"Gap measure check (272 review I1): {nf} of {nb} fills with both measures sit on different sides of the {GAP_CUT:g} % cut "
                 f"(rebuilt closed-bar gap vs the engine stamp, which includes the forming bar) · {int((c.gap_src == 'stamp').sum())} on the "
                 f"stamp fallback · {int((c.gap_src == 'none').sum())} with neither.")
    if len(man):
        L.append(f"Excluded: {len(man)} manually closed fill(s): " + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} ({r.why})" for r in man.itertuples()) + ".")
    L.append(f"Evidence (not part of the verdict): {EVIDENCE}.")
    L.append(BAR)
    if "first" in st:
        f0 = st["first"]
        L.append(f"**Frozen verdict (crossing at fill {f0['at']}, frozen on the run of {f0.get('run_at', '?')}, zone N {f0['n']} · {f0['days']} d, "
                 f"breakeven {f0['be']} %): {f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    return L + [REVERT, ""]


# ─────────────────────────── self-test (hermetic, offline) ───────────────────────────
def selftest():
    global EXPORT_GLOB, _CACHE
    import socket
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (EXPORT_GLOB, _CACHE, WT._CACHE, WT.MY_CACHE, WT._NET_BLOCKED, socket.socket)

    class _NoNet(socket.socket):
        def connect(self, *a, **k):
            raise RuntimeError("network blocked (selftest)")
    socket.socket = _NoNet
    WT._NET_BLOCKED = True
    try:
        chk((GAP_CUT, K_MIN, N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED, EMA_BARS) == (1.87, 2, 15, 8, 0.95, 0.50, 4000, 7, 300),
            "frozen constants pinned")
        # ── zone / rest / unscored
        chk([group(1.88, 2), group(1.87, 2), group(2.5, 1), group(2.5, 3), group(np.nan, 2), group(None, 2), group(2.5, None), group(-1.0, 4)]
            == ["zone", "rest", "rest", "zone", "unscored", "unscored", "unscored", "rest"], "zone = gap > 1.87 (strict) ∧ k_now ≥ 2; missing → UNSCORED")
        # ── k_now: same episode, non-WILLY only, any status, strictly earlier entry bar close (catch-up bars shift it)
        b = dict(direction="LONG", status="CLOSED", entry_price=1.0, pnl_percentage=3.0, pnl=10.0, notional_value=1000.0, close_reason="FRENZY_TP",
                 entry_pair_ema20_ema50_gap_pct=2.5, entry_frenzy_catchup_bars=0)
        S, S2 = "2026-10-10T12:00:00", "2026-10-10T18:00:00"
        rows = [dict(b, opened_at="2026-10-10T13:00:08", pair="AAAUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S),
                dict(b, opened_at="2026-10-10T13:30:08", pair="AAAUSDT", entry_strategy="FRENZY_WILLY", entry_frenzy_spike_at=S),
                dict(b, opened_at="2026-10-10T14:00:08", pair="AAAUSDT", entry_strategy="FRENZY_LONG", entry_frenzy_spike_at=S, status="OPEN"),
                dict(b, opened_at="2026-10-10T15:00:08", pair="AAAUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S),
                dict(b, opened_at="2026-10-10T19:00:08", pair="AAAUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-10T16:00:08", pair="BBBUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=np.nan),
                # catch-up: opened 14:20 but its entry bar closed 2 bars earlier (14:10) — the WIDE fill opened 14:12 (bar 14:10) is NOT earlier
                dict(b, opened_at="2026-10-10T14:12:00", pair="CCCUSDT", entry_strategy="FRENZY_WIDE", entry_frenzy_spike_at=S),
                dict(b, opened_at="2026-10-10T14:20:08", pair="CCCUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S, entry_frenzy_catchup_bars=2)]
        fam = family(pd.DataFrame(rows))
        lt = fam[fam.strat == STRATEGY]
        K = dict(zip(zip(lt.pair, lt.ts.dt.strftime("%H:%M")), k_now(lt, fam)))
        chk(K[("AAAUSDT", "13:00")] == 1 and K[("AAAUSDT", "15:00")] == 3, f"k_now counts LITE + an OPEN LONG, never WILLY ({K})")
        chk(K[("AAAUSDT", "19:00")] == 1, "another episode (spike) on the same pair starts at 1")
        chk(K[("BBBUSDT", "16:00")] is None, "no spike stamp → k_now unknown (UNSCORED)")
        chk(K[("CCCUSDT", "14:20")] == 1, "entry bar close (catch-up bars) decides the order, strictly earlier only")
        chk(len(family(pd.DataFrame([rows[0], dict(rows[0], status="OPEN", opened_at="2026-10-10T13:00:08.4")]))) == 1,
            "dedup (opened_at[:19], pair, strategy): CLOSED beats OPEN")
        # ── gap rebuild = the engine EMA on the 300 closed bars ending at the signal bar; a hole → None
        t_e = pd.Timestamp("2026-10-10T13:00:08").value // 10**6
        sig = (t_e // M5) * M5 - M5
        ot = sig - np.arange(EMA_BARS + 10)[::-1] * M5
        cl = 1.0 + 0.001 * np.arange(len(ot))
        k5 = pd.DataFrame(dict(open_time=ot.astype("int64"), c=cl))
        c300 = pd.Series(cl[-EMA_BARS:])
        exp = (c300.ewm(span=13, adjust=False).mean().iloc[-1] / c300.ewm(span=50, adjust=False).mean().iloc[-1] - 1) * 100
        ec = (t_e // M5) * M5                            # entry bar close (no catch-up) → signal bar = ec − 5m
        chk(abs(gap_rebuild(k5, ec) - exp) < 1e-12 and gap_rebuild(k5.drop(index=100), ec) is None
            and gap_rebuild(k5.iloc[:-1], ec) is None, "gap rebuild: ewm(span, adjust=False) on 300 closes; a missing bar → unscored")
        c2 = pd.Series(cl[-EMA_BARS - 2:-2])                # catch-up 2 bars: the window ends 2 bars earlier (review M3)
        exp2 = (c2.ewm(span=13, adjust=False).mean().iloc[-1] / c2.ewm(span=50, adjust=False).mean().iloc[-1] - 1) * 100
        chk(abs(gap_rebuild(k5, ec - 2 * M5) - exp2) < 1e-12, "gap rebuild ends at entry_close − 5m (catch-up bars honoured)")
        fz = family(pd.DataFrame([dict(rows[0], entry_frenzy_catchup_bars=2)]))
        chk(int(fz.entry_close.iloc[0]) == ec - 2 * M5, "family: entry_close = ⌊t / 5m⌋·5m − catch-up · 5m")
        # ── (review I1) precedence: the closed-bar rebuild first, the engine stamp only as a labelled fallback; side flips counted
        g_, s_ = pick_gap([1.5, np.nan, np.nan], [2.6, 2.6, np.nan])
        chk(g_[0] == 1.5 and g_[1] == 2.6 and np.isnan(g_[2]) and s_ == ["rebuilt", "stamp", "none"],
            "gap: rebuild wins over the stamp (1.5 rebuilt vs 2.6 stamped → rest), stamp only when no rebuild, neither → none")
        chk(group(g_[0], 2) == "rest" and group(2.6, 2) == "zone" and side_flips([1.5, 2.0, np.nan], [2.61, 2.1, 3.0]) == (1, 2),
            "the rebuilt measure classifies; one of the two fills with both measures flips side")
        # ── (review M2) an UNPRICED fill holds the freeze while unpriced; decided on ticks = final
        chk(pending(None, False) and pending(None, True) and pending("ticks+1m", True) and not pending("ticks+1m", False)
            and not pending("ticks", True), "pending: unpriced always holds; a 1m part holds while ticks can come; ticks final")
        # ── verdicts
        def mk(z, r, zd=15):
            Z = pd.DataFrame(dict(walk=z, day=[f"D{i % zd}" for i in range(len(z))], pair=[f"P{i}" for i in range(len(z))]))
            R = pd.DataFrame(dict(walk=r, day=[f"D{i}" for i in range(len(r))], pair=[f"R{i}" for i in range(len(r))]))
            return Z, R
        z, r = mk([-3.0, -3.0, -3.0, 3.0] * 5, [3.0, -3.0] * 10)
        chk(verdict(z.iloc[:14], r, 50.0)[0] == "COLLECTING" and verdict(*mk([-3.0] * 15, [3.0] * 5, zd=7), 50.0)[0] == "COLLECTING",
            "N < 15 or < 8 days → collecting")
        chk(verdict(z, r, 50.0)[0].startswith("FILTER CANDIDATE"), f"bad zone → candidate ({verdict(z, r, 50.0)})")
        # empty rest group: the rest avg is undefined → never RETIRED by the comparison; the candidate bar alone decides
        er = r.iloc[:0]
        chk(verdict(z, er, 50.0)[0].startswith("FILTER CANDIDATE") and verdict(z.assign(walk=1.0), er, 50.0)[0] == "KEEP OBSERVING"
            and "rest +nan % (N 0)" in verdict(z, er, 50.0)[1], "empty rest: no RETIRED (undefined rest avg), the bar decides")
        # a day / pair carrying EXACTLY 50 % of the zone loss blocks the candidate (bar: '≥ 50 %')
        h = mk([-3.0] * 4 + [-1.0] * 12 + [0.5] * 4, [3.0] * 20, zd=20)[0]
        h["day"] = ["DX"] * 4 + [f"E{i}" for i in range(16)]
        chk(abs(NF.gross_loss_share(h.walk.values, h.day.values) - 0.5) < 1e-12 and verdict(h, r, 90.0)[0] == "KEEP OBSERVING"
            and verdict(h.assign(day=[f"F{i}" for i in range(20)]), r, 90.0)[0].startswith("FILTER CANDIDATE"),
            "a day at exactly 50 % of the zone loss → no candidate; spread → candidate")
        chk(verdict(z, r.assign(walk=-3.5), 50.0)[0] == "RETIRED", "zone avg ≥ rest avg → RETIRED (checked first)")
        chk(verdict(z, r, 20.0)[0] == "KEEP OBSERVING" and verdict(z, r, None)[0] == "KEEP OBSERVING", "WR ≥ breakeven / no breakeven → observe")
        chk(verdict(z.assign(walk=np.where(np.arange(20) == 0, -500.0, z.walk)), r, 50.0)[0] == "KEEP OBSERVING",
            "one day / pair ≥ 50 % of the zone loss → observe")
        zm, rm = mk([-1.0, 1.0] * 10, [3.0] * 20)
        chk(verdict(zm.assign(walk=[-1.0, 0.9] * 10), rm, 60.0)[0] == "KEEP OBSERVING", "P(mean < 0) < 0.95 → observe")
        # ── freeze once at the crossing, OPEN / provisional hold, never recomputed
        sc = pd.concat([z.assign(grp="zone"), r.assign(grp="rest")], ignore_index=True)
        sc["ts"] = pd.Timestamp("2026-10-10") + pd.to_timedelta(np.r_[np.arange(20) * 25, np.arange(20) * 25 + 1], "h")
        sc["day"] = sc.ts.dt.strftime("%Y-%m-%d")
        st, ch = freeze({}, sc, sc, "t1")
        chk(ch and st["first"]["n"] == 15 and st["first"]["state"].startswith("FILTER CANDIDATE"), f"frozen at the 15th zone fill ({st})")
        chk(freeze(dict(st), sc.assign(walk=3.0), sc, "t2") == (st, False), "a frozen verdict is never recomputed")
        why = []
        chk(freeze({}, sc, sc, "t", [(sc.ts.min(), "provisional X")], why) == ({}, False) and "provisional X" in why[0], "pending fill holds")
        # ── end-to-end (temp exports + tick files; LITE exit read from a config namespace)
        with tempfile.TemporaryDirectory() as td:
            WT._CACHE = td
            WT.MY_CACHE = os.path.join(td, "scout_willy_timecap")
            _CACHE = td
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            taker = WT.TAKER
            px = lambda net: 1.0 * (1 + (net + taker) / 100) / (1 - taker / 100)   # noqa: E731

            def ticks(pair, day, t0, nets):
                os.makedirs(os.path.join(td, "ticks_q", pair), exist_ok=True)
                np.savez_compressed(os.path.join(td, "ticks_q", pair, f"{day}.npz"),
                                    t=(t0 + np.arange(1, len(nets) + 1) * 60_000).astype(np.int64), p=np.array([px(x) for x in nets]))
            e2e = [dict(b, opened_at="2026-10-10T01:00:08", pair="ZZZUSDT", entry_strategy="FRENZY_LONG", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=3.0),
                   dict(b, opened_at="2026-10-10T02:00:08", pair="ZZZUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=3.0, pnl_percentage=-3.02, pnl=-30.0, close_reason="STOP_LOSS"),
                   dict(b, opened_at="2026-10-10T03:00:08", pair="YYYUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=3.0, pnl_percentage=3.01, close_reason="FRENZY_TP"),
                   dict(b, opened_at="2026-10-10T04:00:08", pair="XXXUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=np.nan, pnl_percentage=3.01, close_reason="FRENZY_TP"),
                   dict(b, opened_at="2026-10-10T05:00:08", pair="WWWUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=1.0, pnl_percentage=1.0, close_reason="MANUAL_CLOSE"),
                   dict(b, opened_at="2026-10-01T05:00:08", pair="OLDUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S),
                   # stamp 3.0 (forming bar) but the closed-bar rebuild ≈ 0 → REST (review I1); UUU has no path → UNPRICED (holds)
                   dict(b, opened_at="2026-10-10T06:00:08", pair="VVVUSDT", entry_strategy="FRENZY_LONG", entry_frenzy_spike_at=S),
                   dict(b, opened_at="2026-10-10T07:00:08", pair="VVVUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S,
                        entry_pair_ema20_ema50_gap_pct=3.0, pnl_percentage=3.01, close_reason="FRENZY_TP"),
                   dict(b, opened_at="2026-10-10T08:00:08", pair="UUUUSDT", entry_strategy="FRENZY_LITE", entry_frenzy_spike_at=S)]
            pd.DataFrame(e2e).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            ecv = pd.Timestamp("2026-10-10T07:00:00").value // 10**6
            os.makedirs(os.path.join(td, "k5m_full"))
            pd.DataFrame(dict(open_time=ecv - M5 - np.arange(EMA_BARS)[::-1] * M5, c=1.0)).to_csv(os.path.join(td, "k5m_full", "VVVUSDT.csv"),
                                                                                                    index=False)
            for p_, h_, nets in (("ZZZUSDT", 2, [-1.0, -3.02]), ("YYYUSDT", 3, [1.0, 3.01]), ("XXXUSDT", 4, [3.01]), ("VVVUSDT", 7, [3.01])):
                ticks(p_, "2026-10-10", pd.Timestamp(f"2026-10-10T{h_:02d}:00:08").value // 10**6, nets)
            th = {"frenzy_tp_pct": 3.0, "frenzy_stop_pct": 3.0, "frenzy_max_hold_minutes": 720}
            chk(lite_rule(th) == (3.0, 3.0, 720), "LITE's exit read from the config")
            sp, rp = os.path.join(td, "s.json"), os.path.join(td, "rows.csv")
            out = "\n".join(run(pd.Timestamp("2026-10-11T12:00").value // 10**6, state_path=sp, rows_path=rp, open_ts=[], th=th,
                                deploy="2026-10-08 13:37:25"))
            chk("| **zone (stretched re-entry)** | 1 | 1 | 1 | 0 % | -3.020 %" in out and "ZZZUSDT [zone] gap +3.00ˢ (stamp +3.00) k 2" in out,
                f"run(): ZZZ (LITE after a LONG in the same episode, gap 3.0) in the zone, tick-priced\n{out}")
            chk("YYYUSDT [rest] gap +3.00ˢ (stamp +3.00) k 1" in out and "XXXUSDT [unscored] gap – (stamp –) k 1" in out
                and "| rest (gap ≤ 1.87 % or first entry) | 2 |" in out and "VVVUSDT [rest] gap +0.00 (stamp +3.00) k 2" in out
                and "1 of 1 fills with both measures sit on different sides" in out and "UUUUSDT [unscored] gap – (stamp +2.50) k 1 · live +3.00 → walked UNPRICED" in out   # stamp 2.50 within 0.67 of the cut → UNSCORED (I-A)
                and pick_gap([np.nan], [2.6])[1] == ["stamp"] and pick_gap([np.nan], [2.5])[1] == ["none"],
                "first entry → rest; no gap stamp and no 5m cache → UNSCORED (priced, never counted as rest)\n" + out)
            chk("Excluded: 1 manually closed" in out and "OLDUSDT" not in out and ": 4/4 · unpriced 1" in out and "COLLECTING" in out and BAR in out,
                "manual close excluded, pre-floor fill out, parity printed, the frozen bar in the output")
            rs = pd.read_csv(rp)
            chk(len(rs) == 5 and not os.path.exists(sp) and {"gap_rebuilt", "gap_stamp", "gap_src"} <= set(rs.columns)
                and rs[rs.pair == "VVVUSDT"].gap_src.iloc[0] == "rebuilt", "rows stored with both gap measures, nothing frozen at N 1")
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1)
            chk(False, "network must be blocked")
        except RuntimeError:
            chk(True, "network blocked inside the selftest")
    finally:
        EXPORT_GLOB, _CACHE, WT._CACHE, WT.MY_CACHE, WT._NET_BLOCKED, socket.socket = saved
        WT._K1.clear()
        _K5.clear()
    print(f"selftest LITE_STRETCH_REENTRY OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
