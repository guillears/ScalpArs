#!/usr/bin/env python3
"""🧊 Scout — momentum-LONG cooldown observations (registered 2026-10-07; OBSERVE only — never changes config, never trades).

Source study: reports/ML_COOLDOWN_STUDY_2026-10-07.md (Rule A, the operator's 30-min open-cooldown, failed in both sources; Rule C — wait 30 min
after a momentum-long STOP-OUT — was the only pre-registered rule pointing the same way in the replay and on live fills, live 7 fills / 6 windows).
Definitions mirror the study's code (scratchpad ml_cooldown/analyze.py tag / simulate, analyze3.py cap).

FILLS  momentum-LONG bot fills from the orders exports (~/Downloads/scalpars_orders_paper_*.csv), newest export wins per (opened_at[:19], pair,
       direction) — the cross-batch dedup key. Momentum = entry_strategy MOMENTUM or empty (every sleeve, MANUAL and FLIP carry their own
       strategy name and are out); probes out like the engine: "PROBE" in cell_multiplier_source OR pattern_cell_source. Status OPEN / CLOSED
       are real fills (they extend chains / count as prior opens); only CLOSED fills carry P&L and enter the tallies. Timestamps are parsed as
       UTC, then made naive.
1×     pnl_percentage is the % of NOTIONAL (price move net of fees) — a cell size multiplier scales the notional, not the move, so the % is
       already the 1× value (the study used it as-is: master stack_pct / pnl_percentage). $ at 1× = pnl ÷ (cell_multiplier ×
       cell_lev_multiplier), each missing / ≤ 0 → 1; $ as-sized = pnl.
STOP   a STOP-LOSS close = close_reason whose base (a " L<n>" ladder suffix stripped) starts with STOP_LOSS (STOP_LOSS, STOP_LOSS_WIDE) — the
       study's written, frozen signature "STOP_LOSS*" (§8). (The study's code regex STOP|SL|HARD_STOP also caught profit-side TRAILING_STOP
       exits; on the live master cohort both readings give the same 7 fills.) Display-only alternative: STOP_LOSS* ∪ RH_HARD_STOP (the
       recovery hold's own stop close) — shown, never counted.
WINDOW the study's §8 unit: 60-min CHAINS of momentum-long fills by open time (a new chain when the gap to the previous fill's open is > 60 min);
       a cohort fill's window = the start of its chain. Used for the ≥ 8-windows leg AND the bootstrap clustering. Days shown too.

(A) ML_STOP_COOLDOWN (PRE-REGISTERED, frozen exactly as tested): a momentum long opened < 30.0 min (strict) after a momentum-long fill closed
    with a STOP-LOSS reason (close ≤ open). SEQUENTIAL like the study: a fill already in the cohort (it would be refused once armed) never
    starts or extends a cooldown — the sequential cohort is THE counted one. Counted only for fills opened ≥ 2026-10-08 00:00 UTC (operator
    choice; the in-sample fills are reference only). BAR (CLAUDE.md expectancy filter bar, frozen): N ≥ 15 ∧ ≥ 8 windows ∧ WR < 61.5 % (sleeve
    breakeven frozen from the study) ∧ window-clustered bootstrap P(mean < 0) ≥ 0.95 (seed 7, 4,000 resamples) ∧ no single window / pair
    ≥ 50 % of the cohort's gross loss → PROPOSE ARMING (operator decision; then the 30–50 % haircut and the revert gate below). Below N /
    windows → collecting; else not established. Never re-fit 30 min.
(B) CLUSTER2_120 (EXPLORATORY WATCHLIST — chosen AFTER seeing the study's data, NOT pre-registered): a momentum long opened when ≥ 2 ACCEPTED
    momentum longs (not themselves flagged — the study's sequential cap) opened < 120 min (strict) before it. Tally only; no bar.
STORE  reports/SCOUT_ML_COOLDOWN.csv, one row per CLOSED fill, WRITE-ONCE (a stored key is never rewritten: every flag depends only on EARLIER
       fills, so a later export cannot change it); unreadable → a timestamped .bad, start empty. No network. Cost: one sort + one linear pass
       with time-pruned queues per run (O(n)); the exports are re-read each run with usecols (a ~1.5k-row read).
"""
import glob
import os
import re
import time
from collections import deque

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "reports", "SCOUT_ML_COOLDOWN.csv")
EXPORT_GLOB = "~/Downloads/scalpars_orders_paper_*.csv"
VER = 2                                                # 2 = sequential cohort + 60-min chains (review 2026-10-07)
FRESH_FROM = pd.Timestamp("2026-10-08 00:00:00")       # counted floor (opened_at, UTC) — operator choice
CD_MIN, BE_WR, N_MIN, W_MIN, P_MIN, TOP_MAX = 30.0, 61.5, 15, 8, 0.95, 0.50   # FROZEN (study §8)
CHAIN_GAP_MIN = 60.0                                   # the study's window unit: gap > 60 min starts a new chain
C_K, C_WIN_MIN = 2, 120.0                              # CLUSTER2_120 (exploratory)
BOOT_N, BOOT_SEED = 4000, 7
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "close_reason", "pnl", "pnl_percentage",
        "cell_multiplier", "cell_lev_multiplier", "cell_multiplier_source", "pattern_cell_source")
STORE_COLS = ("key", "opened_at", "closed_at", "pair", "close_reason", "is_stop", "pnl_pct", "pnl_usd", "size_mult", "usd_1x",
              "chain", "cd_flag", "cd_stop_close", "cd_min_since_stop", "cd_window30", "cd_any", "cd_alt", "c120_prior", "c120_flag",
              "fresh", "ver", "recorded_at")
REVERT_TEXT = "revert if the first 10 refused signals re-priced with the live exit replica show WR ≥ 61 % or Σ > 0"
REF_A = ("study (2026-10-07): yr5 replay Rule C 52 fills · 42 % WR · −0.272 %/fill vs −0.061 for the rest (diff −0.21, 98 %; mostly Jan–Mar) · "
         "live master 7 fills / 6 windows (5 days) · 57 % · −0.077 % vs +0.21 rest")
REF_B = "study (exploratory, sequential cap): live master 15 · 60 % · −0.059 % · yr5 replay 194 · 54 % · −0.157 %"
_SUFFIX = re.compile(r"\s+L\d+$")


# ─────────────────────────── pure ───────────────────────────
def _base(reason):
    if reason is None or (isinstance(reason, float) and np.isnan(reason)):
        return ""
    return _SUFFIX.sub("", str(reason).strip())


def is_stop(reason):
    """STOP-LOSS family: base reason (a trailing ' L<n>' stripped) starts with STOP_LOSS."""
    return _base(reason).startswith("STOP_LOSS")


def is_stop_alt(reason):
    """display-only alternative: STOP_LOSS* ∪ RH_HARD_STOP."""
    return is_stop(reason) or _base(reason) == "RH_HARD_STOP"


def is_probe(cms, pcs):
    return any("PROBE" in str(x) for x in (cms, pcs) if x is not None and not (isinstance(x, float) and np.isnan(x)))


def to_ts(s):
    """UTC-parsed, then naive (an offset suffix is honoured; naive strings are read as UTC)."""
    t = pd.to_datetime(pd.Series(s).astype(str).str.strip().str.replace("T", " "), format="mixed", errors="coerce", utc=True)
    return t.dt.tz_convert(None)


def size_mult(cm, clm):
    def f(v):
        v = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) and np.isfinite(v) and v > 0 else 1.0
    return f(cm) * f(clm)


def demux_usd(pnl, cm, clm=None):
    """$ at 1× = pnl ÷ (cell_multiplier × cell_lev_multiplier); the % needs no de-mux (it is % of notional)."""
    p = pd.to_numeric(pd.Series([pnl]), errors="coerce").iloc[0]
    return None if pd.isna(p) else float(p) / size_mult(cm, clm)


def chains(opens, gap_min=CHAIN_GAP_MIN):
    """sorted open times → chain start per fill (the study: a new chain when the gap to the previous open is > gap_min, or the first)."""
    out, start, prev = [], None, None
    for t in opens:
        if prev is None or (t - prev) > pd.Timedelta(minutes=gap_min):
            start = t
        out.append(start); prev = t
    return out


def sequential(t_open, t_close, stop, mins=CD_MIN):
    """the study's Rule C, sequential: walk fills by open time; a fill is FLAGGED when an ACCEPTED earlier fill closed with a stop at or before
    its open and < mins before it; flagged fills are not accepted (they never start / extend a cooldown). → (flags, trigger close, minutes)."""
    lim = pd.Timedelta(minutes=mins)
    acc = []                                          # accepted stop closes (time-pruned below)
    flags, trig, mns = [], [], []
    for t, tc, s in zip(t_open, t_close, stop):
        acc = [c for c in acc if t - c < lim or c > t]  # drop expired stops (a stop closing after t stays for later fills)
        q = [c for c in acc if c <= t and (t - c) < lim]
        if q:
            c = max(q); flags.append(True); trig.append(c); mns.append((t - c).total_seconds() / 60.0)
            continue
        flags.append(False); trig.append(None); mns.append(None)
        if s and pd.notna(tc):
            acc.append(tc)
    return flags, trig, mns


def cluster_cap(t_open, k=C_K, mins=C_WIN_MIN):
    """the study's exploratory cap, sequential: flagged when ≥ k ACCEPTED fills opened < mins (strict) before; flagged fills not accepted.
    → (flags, prior accepted count)."""
    lim = pd.Timedelta(minutes=mins)
    acc = deque(); flags, prior = [], []
    for t in t_open:
        while acc and t - acc[0] >= lim:
            acc.popleft()
        n = len(acc); prior.append(n)
        if n >= k:
            flags.append(True)
        else:
            flags.append(False); acc.append(t)
    return flags, prior


def stop_windows(stop_closes, mins=CD_MIN):
    """merged [close, close + mins) intervals → sorted list of (start, end) (display only)."""
    iv = sorted((c, c + pd.Timedelta(minutes=mins)) for c in stop_closes if c is not None and pd.notna(c))
    out = []
    for s, e in iv:
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def window_of(t_open, windows):
    for s, e in windows:
        if s <= t_open < e:
            return s.strftime("%Y-%m-%dT%H:%M:%S")
    return None


def boot_p_neg(pct, win, n=BOOT_N, seed=BOOT_SEED):
    """window-clustered bootstrap: resample windows with replacement, pooled mean of their fills → P(mean < 0). < 2 windows → None."""
    g = pd.DataFrame(dict(p=np.asarray(pct, float), w=list(win))).groupby("w").p.agg(["sum", "count"])
    if len(g) < 2:
        return None
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    return float(((s[idx].sum(1) / c[idx].sum(1)) < 0).mean())


def top_loss_share(pct, key):
    """largest single key's share of the cohort's GROSS loss (losing fills' |pct| summed per key — the study's method). No loss → 0."""
    d = pd.DataFrame(dict(p=np.asarray(pct, float), k=list(key)))
    lo = d[d.p < 0]
    gl = -lo.p.sum()
    return float((-lo.groupby("k").p.sum()).max() / gl) if gl > 0 else 0.0


def decide(c):
    """c = DataFrame(pct, window, pair) of the COUNTED cooldown cohort → (state, detail): collecting / propose / not_established."""
    c = c.assign(pct=pd.to_numeric(c.pct, errors="coerce")).dropna(subset=["pct"])
    n = len(c)
    nw = c.window.nunique() if n else 0
    if n < N_MIN or nw < W_MIN:
        return "collecting", f"N {n}/{N_MIN} · windows {nw}/{W_MIN}"
    wr = 100.0 * float((c.pct > 0).mean())
    p = boot_p_neg(c.pct.values, c.window.values)
    tw, tp = top_loss_share(c.pct.values, c.window.values), top_loss_share(c.pct.values, c.pair.values)
    legs = [wr < BE_WR, p is not None and p >= P_MIN, tw < TOP_MAX, tp < TOP_MAX]
    det = (f"N {n} · windows {nw} · WR {wr:.0f} % vs breakeven {BE_WR:.1f} % · P(mean<0) {('–' if p is None else f'{p:.2f}')} · "
           f"top window {tw * 100:.0f} % / top pair {tp * 100:.0f} % of the loss")
    return ("propose" if all(legs) else "not_established"), det


def _iso(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def tag(fills):
    """ONE momentum-long fill set (OPEN + CLOSED) → DataFrame of the CLOSED fills with every store column, computed in open-time order."""
    f = fills.copy()
    f["t"], f["tc"] = to_ts(f.opened_at).values, to_ts(f.closed_at).values
    f = f[f.t.notna()].sort_values(["t", "pair"], kind="stable").reset_index(drop=True)
    if not len(f):
        return pd.DataFrame(columns=list(STORE_COLS))
    closed = (f.status.astype(str).str.upper() == "CLOSED").values
    f["is_stop"] = [bool(c) and is_stop(r) for c, r in zip(closed, f.close_reason)]
    alt = [bool(c) and is_stop_alt(r) for c, r in zip(closed, f.close_reason)]
    T, TC = list(f.t), [x if pd.notna(x) else None for x in f.tc]
    f["chain"] = chains(T)
    fl, trig, mns = sequential(T, TC, list(f.is_stop))
    fl_alt = sequential(T, TC, alt)[0]
    # non-sequential reading (any stop, accepted or not) — display only
    stops_all = [c for c, s in zip(TC, f.is_stop) if s and c is not None]
    lim = pd.Timedelta(minutes=CD_MIN)
    any_ = [any(c <= t and t - c < lim for c in stops_all) for t in T]
    acc_stops = [c for c, s, x in zip(TC, f.is_stop, fl) if s and not x and c is not None]
    w30 = stop_windows(acc_stops)
    cf, cp = cluster_cap(T)
    out = []
    for i, r in enumerate(f.itertuples()):
        if not closed[i]:
            continue
        out.append(dict(key=f"{str(r.opened_at)[:19]}|{r.pair}|LONG", opened_at=_iso(r.t), closed_at=_iso(r.tc), pair=r.pair,
                        close_reason=r.close_reason, is_stop=bool(r.is_stop),
                        pnl_pct=pd.to_numeric(pd.Series([r.pnl_percentage]), errors="coerce").iloc[0],
                        pnl_usd=pd.to_numeric(pd.Series([r.pnl]), errors="coerce").iloc[0],
                        size_mult=size_mult(r.cell_multiplier, getattr(r, "cell_lev_multiplier", None)),
                        usd_1x=demux_usd(r.pnl, r.cell_multiplier, getattr(r, "cell_lev_multiplier", None)),
                        chain=_iso(r.chain), cd_flag=bool(fl[i]), cd_stop_close=_iso(trig[i]),
                        cd_min_since_stop=(round(mns[i], 3) if fl[i] else None), cd_window30=(window_of(r.t, w30) if fl[i] else None),
                        cd_any=bool(any_[i]), cd_alt=bool(fl_alt[i]), c120_prior=int(cp[i]), c120_flag=bool(cf[i]),
                        fresh=bool(r.t >= FRESH_FROM), ver=VER))
    return pd.DataFrame(out, columns=[c for c in STORE_COLS if c != "recorded_at"])


def merge_store(old, new, now_iso):
    """WRITE-ONCE: rows of `new` whose key is already stored are dropped; the rest are appended with recorded_at."""
    if new is None or not len(new):
        return old if old is not None and len(old) else pd.DataFrame(columns=list(STORE_COLS))
    have = set(old.key.astype(str)) if old is not None and len(old) and "key" in old else set()
    add = new[~new.key.astype(str).isin(have)].assign(recorded_at=now_iso)
    if old is None or not len(old):
        return add.reset_index(drop=True)
    return pd.concat([old, add], ignore_index=True) if len(add) else old


# ─────────────────────────── data ───────────────────────────
def load_orders(paths=None):
    """momentum-LONG bot fills (OPEN / CLOSED, non-probe) from the exports, newest export per (opened_at[:19], pair, direction)."""
    fr = []
    for f in (paths if paths is not None else glob.glob(os.path.expanduser(EXPORT_GLOB))):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
            if len(d) and {"opened_at", "pair", "direction", "status", "pnl_percentage"} <= set(d.columns):
                fr.append(d.assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS))
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    o["_k"] = o.opened_at.astype(str).str[:19]
    o = o.drop_duplicates(["_k", "pair", "direction"], keep="last")
    for c in COLS:
        if c not in o:
            o[c] = np.nan
    o = o[(o.direction.astype(str) == "LONG") & o.status.astype(str).str.upper().isin(["OPEN", "CLOSED"])]
    o = o[o.entry_strategy.fillna("").astype(str).str.strip().isin(["MOMENTUM", ""])]
    o = o[[not is_probe(a, b) for a, b in zip(o.cell_multiplier_source, o.pattern_cell_source)]]
    return o.drop(columns=["_k", "_m"]).reset_index(drop=True)


def _bad_path(path):
    base = f"{path}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{os.getpid()}"
    p, i = base + ".bad", 1
    while os.path.exists(p):
        p, i = f"{base}.{i}.bad", i + 1
    return p


def load_store(path=None):
    path = path or CSV
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        d = pd.read_csv(path)
        if len(d) and not {"key", "pnl_pct", "cd_flag", "fresh", "chain"} <= set(d.columns):
            raise ValueError("columns missing")
        return d
    except Exception:
        try:
            os.replace(path, _bad_path(path))
        except OSError:
            pass
        return pd.DataFrame()


def save_store(df, path=None):
    path = path or CSV
    tmp = f"{path}.{os.getpid()}.tmp"; df.to_csv(tmp, index=False); os.replace(tmp, path)


# ─────────────────────────── render ───────────────────────────
_T = ("True", "1", "1.0", "true")


def _b(s):
    return s.astype(str).isin(_T)


def _line(g, wcol=None):
    if not len(g):
        return "0"
    x = pd.to_numeric(g.pnl_pct, errors="coerce")
    u, a = pd.to_numeric(g.usd_1x, errors="coerce"), pd.to_numeric(g.pnl_usd, errors="coerce")
    days = g.opened_at.astype(str).str[:10].nunique()
    w = f" · {g[wcol].nunique()} windows" if wcol else ""
    return (f"{len(g)}{w} · {days} d · WR {(x > 0).mean() * 100:.0f} % · avg {x.mean():+.3f} % · Σ {x.sum():+.2f} % · "
            f"Σ$ 1× {u.sum():+.0f} · Σ$ as-sized {a.sum():+.0f}")


def render(st):
    st = st.copy() if len(st) else pd.DataFrame(columns=list(STORE_COLS))
    fr, cd, cl = _b(st.fresh), _b(st.cd_flag), _b(st.c120_flag)
    coh = st[fr & cd]
    state, det = decide(pd.DataFrame(dict(pct=coh.pnl_pct.values, window=coh.chain.astype(str).values, pair=coh.pair.values)))
    lab = {"collecting": "⏳ collecting", "propose": "📋 PROPOSE ARMING — operator decision (then 30–50 % haircut + the revert gate below)",
           "not_established": "❌ not established"}[state]
    n30 = coh.cd_window30.nunique() if "cd_window30" in coh and len(coh) else 0
    alt_n = int((fr & _b(st.cd_alt)).sum()) if "cd_alt" in st else 0
    any_n = int((fr & _b(st.cd_any)).sum()) if "cd_any" in st else 0
    return ["## 🧊 ML cooldown — momentum longs after a STOP-OUT (pre-registered) · cluster watch (exploratory) — OBSERVE only", "",
            f"Momentum-LONG bot fills (MOMENTUM / empty strategy, probes and MANUAL + sleeves out) from the orders exports, newest export per "
            f"(opened_at, pair, direction). **Counted from {FRESH_FROM:%Y-%m-%d %H:%M} UTC (operator choice) — every earlier fill is in-sample "
            f"and shown as reference only.** % = pnl_percentage (% of notional — size-invariant, already 1×); Σ$ 1× = pnl ÷ (cell multiplier × "
            f"cell lev multiplier); Σ$ as-sized = pnl. STOP = close_reason STOP_LOSS* (' Ln' suffix stripped). Window = the study's 60-min chain "
            f"of momentum-long fills. Source: reports/ML_COOLDOWN_STUDY_2026-10-07.md.", "",
            f"**(A) ML_STOP_COOLDOWN (pre-registered, frozen):** opened < {CD_MIN:g} min after a momentum-long STOP-LOSS close — SEQUENTIAL "
            f"(the counted cohort): a fill already in the cohort would be refused once armed, so it never starts or extends a cooldown.", "",
            "| Group | N · windows · days · WR · avg % 1× · Σ % · Σ$ 1× · Σ$ as-sized |", "|---|---|",
            f"| **after-stop cohort (counted, sequential)** | {_line(coh, 'chain')} |",
            f"| rest (counted) | {_line(st[fr & ~cd])} |",
            f"| after-stop, before the floor (reference) | {_line(st[~fr & cd], 'chain')} |",
            f"| rest, before the floor (reference) | {_line(st[~fr & ~cd])} |", "",
            f"**Bar (frozen):** N ≥ {N_MIN} ∧ ≥ {W_MIN} windows (60-min chains) ∧ WR < {BE_WR:g} % (sleeve breakeven, frozen from the study) ∧ "
            f"chain-clustered bootstrap P(mean < 0) ≥ {P_MIN:.2f} (seed {BOOT_SEED}, {BOOT_N:,} resamples) ∧ no window / pair ≥ "
            f"{TOP_MAX * 100:.0f} % of the gross loss → **{lab}** ({det}).",
            f"- Pre-committed revert gate if it is ever armed: \"{REVERT_TEXT}\".",
            f"- Reference: {REF_A}.",
            f"- Display only, NOT counted: merged 30-min-after-stop windows in the counted cohort {n30} · non-sequential reading (any stop) "
            f"{any_n} counted fills · STOP_LOSS* ∪ RH_HARD_STOP (recovery-hold stop) sequential {alt_n} counted fills.", "",
            f"**(B) CLUSTER2_120 (EXPLORATORY WATCHLIST — chosen after seeing the study's data, NOT pre-registered):** opened when ≥ {C_K} "
            f"accepted momentum longs opened < {C_WIN_MIN:g} min before (the study's sequential cap). Tally only, no bar — needs fresh evidence.", "",
            "| Group | N · days · WR · avg % 1× · Σ % · Σ$ 1× · Σ$ as-sized |", "|---|---|",
            f"| **≥ {C_K} in prior {C_WIN_MIN:g} min (counted)** | {_line(st[fr & cl])} |",
            f"| rest (counted) | {_line(st[fr & ~cl])} |",
            f"| ≥ {C_K} in prior {C_WIN_MIN:g} min, before the floor (reference) | {_line(st[~fr & cl])} |", "",
            f"- Reference: {REF_B}.", ""]


def run(now_ms=None, paths=None, store=None):
    """→ markdown lines. Never raises past the caller's try."""
    now_iso = pd.Timestamp(int(now_ms or time.time() * 1000), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
    o = load_orders(paths)
    new = tag(o) if len(o) else pd.DataFrame()
    old = load_store(store)
    allr = merge_store(old, new, now_iso)
    if len(allr) and len(allr) != len(old):
        save_store(allr, store)
    return render(allr)


# ─────────────────────────── self-test ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    T = pd.Timestamp
    chk(is_stop("STOP_LOSS L1") and is_stop("STOP_LOSS_WIDE L1") and is_stop("STOP_LOSS"), "STOP_LOSS family incl the ' L1' suffix")
    chk(not is_stop("TRAILING_STOP L2") and not is_stop("RH_HARD_STOP") and is_stop_alt("RH_HARD_STOP") and not is_stop(None), "non-stops / alt")
    s = [T("2026-10-08 10:00:00")]
    f1 = sequential([T("2026-10-08 10:29:59"), T("2026-10-08 10:30:00")], [None, None], [False, False])
    chk(not any(f1[0]), "no stop → no flag")
    f2 = sequential([T("2026-10-08 09:50"), T("2026-10-08 10:29:59"), T("2026-10-08 10:30:00")], [s[0], None, None], [True, False, False])
    chk(f2[0] == [False, True, False], "< 30.0 min strict")
    f3 = sequential([T("2026-10-08 09:50"), T("2026-10-08 10:05"), T("2026-10-08 10:40")], [s[0], T("2026-10-08 10:15"), None], [True, True, False])
    chk(f3[0] == [False, True, False], "a cohort fill's own stop never starts a cooldown (sequential)")
    chk(chains([T("2026-10-08 10:00"), T("2026-10-08 11:00"), T("2026-10-08 12:01")]) ==
        [T("2026-10-08 10:00"), T("2026-10-08 10:00"), T("2026-10-08 12:01")], "60-min chains (gap 60 continues, > 60 starts)")
    cf, cp = cluster_cap([T("2026-10-08 10:00"), T("2026-10-08 11:00"), T("2026-10-08 11:30"), T("2026-10-08 11:40"), T("2026-10-08 12:00")])
    chk(cf == [False, False, True, True, False] and cp == [0, 1, 2, 2, 1], "sequential cap: flagged fills not accepted; 120 strict")
    chk(abs(demux_usd(-150.0, 1.5) + 100.0) < 1e-9 and abs(demux_usd(10.0, None, 2.0) - 5.0) < 1e-9, "$ de-mux")
    w = [f"w{i}" for i in range(10) for _ in range(2)]
    neg = np.array([-0.2, 0.1] * 10)
    chk(boot_p_neg(neg, w) == boot_p_neg(neg, w) and boot_p_neg(neg, w) > 0.95, "bootstrap deterministic, negative → P ≈ 1")
    c = pd.DataFrame(dict(pct=neg, window=w, pair=[f"P{i}" for i in range(20)]))
    chk(decide(c)[0] == "propose", "WR 50 < 61.5, negative, spread → propose")
    chk(decide(c.head(14))[0] == "collecting", "N 14 → collecting")
    chk(decide(c.assign(pct=-neg))[0] == "not_established", "winning cohort → not established")
    chk(decide(pd.concat([c.head(14), c.head(4).assign(pct=np.nan)]))[0] == "collecting", "NaN pct dropped before N")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
