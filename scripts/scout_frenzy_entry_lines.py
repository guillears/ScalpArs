#!/usr/bin/env python3
"""🔁📏 Scout — two FRENZY entry observe lines (pre-registered 2026-10-09; OBSERVE only — never changes config, no bot API, NO Binance data:
the orders exports' own stamps only).

COHORT (both)  CLOSED FRENZY_LONG + FRENZY_WIDE LONG fills in the ~/Downloads orders exports, dedup (opened_at[:19], pair, direction) —
               never id; a CLOSED row beats an OPEN one whatever the export order, else the newest export (mtime) wins — opened ≥ the
               DECISION_LOG 250 deploy (the current FRENZY stack: the commit whose message carries "(DECISION_LOG 250)" + 10 min, as
               scripts/scout_revert_gates.py resolves it; commit not in git → NOW). DAY units; Σ$ as sized = the fill's own pnl; avg = pnl %
               (leverage-invariant). FRENZY_LITE fills since the same deploy are scored the same way and shown apart (reference, never counted).

LINE 1 FRENZY_REENTRY_AFTER_WIN   zone = a fill on a pair that already had a FRENZY-family fill (LONG / WIDE / LITE / WILLY, any time,
               counted or not) CLOSED at a profit (pnl > 0) in the SAME FRENZY episode (same entry_frenzy_spike_at) with closed_at < this
               fill's opened_at. An earlier fill still open when this one opened does not count. UNSCORED: this fill has no spike stamp, or
               a same-pair same-episode winner opened earlier has no closed_at (can't tell whether it closed before).
               Live example: RLCUSDT 10-09 03:10 (−3 %) after RLCUSDT 10-08 23:00 (+3 %), same spike 2026-10-08T02:25.
LINE 2 FRENZY_STRETCHED          zone = entry_frenzy_vs_vwap_pct ≥ STRETCH_CUT; missing stamp → UNSCORED. The cut was FROZEN 2026-10-09
               from the DISTRIBUTION ONLY (pnl never loaded): 75th percentile (numpy linear) of entry_frenzy_vs_vwap_pct over ALL
               FRENZY_LONG + FRENZY_WIDE rows of reports/MASTER_POOL_stacked.csv (any stack_keep) = 12.984 on N = 15 (9 LONG + 6 WIDE,
               all stamped) → rounded to 0.1 = 13.0. Never re-fit.

FROZEN VERDICT (per line, NF.du_freeze pattern)  first prefix by (open time, pair) of the scored counted fills where the zone reaches
               N ≥ 15 on ≥ 8 days, frozen ONCE (st['first']); one re-read at zone N ≥ 30 (st['reread'], only on a later run). RETIRE if
               zone mean ≥ rest mean (checked first); FILTER CANDIDATE (operator decides) iff zone WR < FRENZY breakeven WR
               (|avg loss| / (avg win + |avg loss|) on the scored counted fills ≤ the prefix end; fallback 51.5 % until 30 fills) ∧
               day-clustered bootstrap P(mean < 0) ≥ 0.95 (4,000, seed 7) ∧ no day / pair ≥ 50 % of the gross loss; else KEEP OBSERVING.
               Deferred while an OPEN FRENZY-family LONG position opened ≤ the prefix end exists in the newest export (NF.freeze_hold).
"""
import glob
import json
import os
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, gross-loss share, breakeven, freeze hold, state I/O)

# FROZEN 2026-10-09 — distribution only: np.percentile(entry_frenzy_vs_vwap_pct, 75) over every FRENZY_LONG + FRENZY_WIDE row of
# reports/MASTER_POOL_stacked.csv (any stack_keep), N = 15 (all stamped; 1.155 … 14.792) = 12.984 → round(·, 1) = 13.0. pnl not read.
STRETCH_CUT, STRETCH_CUT_N = 13.0, 15
N_MIN, DAYS_MIN, REREAD_N, BE_REF, BE_MIN_FILLS = 15, 8, 30, 51.5, 30
BOOT_N, BOOT_SEED = 4000, 7
DEPLOY_GREP, MIN_MS = "(DECISION_LOG 250)", 60_000
GVOL_OFF_GREP = "(DECISION_LOG 263)"                # FRENZY market-volume gate switched OFF (operator 2026-10-09) — era split only, cohort unchanged
COUNTED = ("FRENZY_LONG", "FRENZY_WIDE")
FAMILY = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY")
REF_STRAT = "FRENZY_LITE"
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl",
        "entry_frenzy_spike_at", "entry_frenzy_vs_vwap_pct")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATES = {"REENTRY": os.path.join(_ROOT, "reports", "SCOUT_FRENZY_REENTRY.json"),
          "STRETCHED": os.path.join(_ROOT, "reports", "SCOUT_FRENZY_STRETCHED.json")}
REVERT = ("Pre-committed revert if ever armed: the first 10 blocked signals re-priced with the live exit → WR ≥ FRENZY breakeven or "
          "Σ > 0 → switch it off")


# ─────────────────────────── deploy floor ───────────────────────────
def deploy_ts(log=None):
    """push time of the first commit whose message carries '(DECISION_LOG 250)' + 10 min (scout_revert_gates.deploy_ms); git
    unavailable / commit absent → NOW (nothing counted until the commit exists)."""
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + DEPLOY_GREP, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        ct = int((out.stdout.strip().splitlines() or [""])[0])
        return pd.Timestamp(ct * 1000 + 10 * MIN_MS, unit="ms")
    except Exception:
        if log is not None:
            log.append(f"deploy commit {DEPLOY_GREP} not found in git — counting from NOW")
        return pd.Timestamp(int(time.time() * 1000), unit="ms")


def gvol_off_ts():
    """the market-volume-gate-OFF deploy = first commit naming '(DECISION_LOG 263)' + 10 min (scout_frenzy_exits.gvol_off_ms); not in git
    yet → None (every counted fill so far is gate-ON era)."""
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + GVOL_OFF_GREP, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000 + 10 * MIN_MS, unit="ms")
    except Exception:
        return None


def era_line(cnt, off):
    """one line: the counted zone / rest split at the gate-OFF deploy (information — the cohort and the verdict pool both eras)."""
    def part(g):
        if not len(g):
            return "0"
        return f"{len(g)} · {(g.pct > 0).mean() * 100:.0f} % · {g.pct.mean():+.3f} %"
    on = cnt if off is None else cnt[cnt.ts < off]
    of = cnt.iloc[:0] if off is None else cnt[cnt.ts >= off]
    when = (f"from {off:%Y-%m-%d %H:%M} UTC = the {GVOL_OFF_GREP} commit + 10 min" if off is not None
            else f"the {GVOL_OFF_GREP} commit is not in git yet — every fill so far is gate-ON era")
    return (f"Era note — market-volume gate OFF ({when}): gate-ON era zone {part(on[on.grp == 'zone'])} / rest {part(on[on.grp == 'rest'])} · "
            f"gate-OFF era zone {part(of[of.grp == 'zone'])} / rest {part(of[of.grp == 'rest'])} (N · WR · avg; both eras counted).")


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
    """rows in precedence order (last wins; CLOSED beats OPEN) → dedup (opened_at[:19], pair, direction) → CLOSED FRENZY-family LONG,
    prepared (ts, close_ts, pct, usd, day, spike, vwap)."""
    o = o.reindex(columns=list(COLS)).copy()                         # only the needed columns (a full 392-col export fragments)
    if len(o):
        o = o.assign(_k=o.opened_at.astype(str).str[:19], _c=o.status.astype(str).str.upper().eq("CLOSED"))
        o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
        o = o[o._c & (o.direction.astype(str) == "LONG") & o.entry_strategy.astype(str).isin(FAMILY)]
    o = o.copy()
    o["ts"] = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    o["close_ts"] = pd.to_datetime(o.closed_at.astype(str).str[:19].where(o.closed_at.notna(), None), format="ISO8601", errors="coerce")
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    o["usd"] = pd.to_numeric(o.pnl, errors="coerce")
    o = o[o.ts.notna() & o.pct.notna()].copy()
    o["pair"] = o.pair.astype(str)
    o["strat"] = o.entry_strategy.astype(str)
    o["day"] = o.ts.dt.strftime("%Y-%m-%d")
    sp = pd.to_datetime(o.entry_frenzy_spike_at.astype(str).str[:19].where(o.entry_frenzy_spike_at.notna(), None),
                        format="ISO8601", errors="coerce")
    o["spike"] = sp.dt.strftime("%Y-%m-%dT%H:%M:%S").where(sp.notna(), None)
    o["vwap"] = pd.to_numeric(o.entry_frenzy_vs_vwap_pct, errors="coerce")
    return o.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True)


def open_family_ts():
    """OPEN FRENZY-family LONG positions in the NEWEST export (by mtime) → [(opened_at, 'OPEN <strategy> <pair>')]."""
    fs = sorted(glob.glob(EXPORT_GLOB), key=os.path.getmtime)
    if not fs:
        return []
    try:
        d = pd.read_csv(fs[-1], low_memory=False, usecols=lambda c: c in COLS)
    except Exception:
        return []
    if not {"opened_at", "status", "direction", "entry_strategy", "pair"} <= set(d.columns):
        return []
    d = d[(d.status.astype(str).str.upper() == "OPEN") & (d.direction.astype(str) == "LONG") & d.entry_strategy.astype(str).isin(FAMILY)]
    t = pd.to_datetime(d.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    return [(a, f"OPEN {s} {p}") for a, s, p in zip(t, d.entry_strategy.astype(str), d.pair.astype(str)) if pd.notna(a)]


# ─────────────────────────── zone rules ───────────────────────────
def reentry_group(fills, fam):
    """fills = the fills to score; fam = every CLOSED family fill (the 'win before' pool) → (grp array, prior-win label list)."""
    grp, why = [], []
    wins = fam[(fam.usd > 0) | (fam.usd.isna() & (fam.pct > 0))]           # pnl > 0; a missing $ falls back to pnl % > 0
    for r in fills.itertuples():
        if pd.isna(r.spike):
            grp.append("unscored"), why.append("no spike stamp")
            continue
        c = wins[(wins.pair == r.pair) & (wins.spike == r.spike) & (wins.ts < r.ts)]
        if (c.close_ts.notna() & (c.close_ts < r.ts)).any():
            w = c[c.close_ts < r.ts].iloc[-1]
            grp.append("zone"), why.append(f"after {w.strat} {w.ts:%m-%d %H:%M} {w.pct:+.2f} %")
        elif c.close_ts.isna().any():
            grp.append("unscored"), why.append("earlier same-episode winner without closed_at")
        else:
            grp.append("rest"), why.append("")
    return np.array(grp, dtype=object), why


def stretched_group(vwap):
    v = np.asarray(vwap, dtype=float)
    return np.where(np.isnan(v), "unscored", np.where(v >= STRETCH_CUT, "zone", "rest"))


# ─────────────────────────── verdict / freeze ───────────────────────────
def _be(sc):
    be = NF.breakeven_wr(sc.pct) if len(sc) >= BE_MIN_FILLS else None
    return (be, "live") if be is not None else (BE_REF, "fallback")


def verdict(zone, rest, be):
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, wr = float(zone.pct.mean()), 100.0 * float((zone.pct > 0).mean())
    mr = float(rest.pct.mean()) if len(rest) else float("nan")
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0f} % vs breakeven {be:.1f} % · mean {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.2f} · top day {sd * 100:.0f} % / top pair {sp * 100:.0f} % of the gross loss")
    if len(rest) and mz >= mr:
        return "RETIRE", det
    if wr < be and p is not None and p >= 0.95 and sd < 0.5 and sp < 0.5:
        return "FILTER CANDIDATE (operator decides)", det
    return "KEEP OBSERVING", det


def crossing_prefix(sc, n_min):
    """scored counted fills sorted by (open time, pair) → the positional prefix ending at the zone fill where the zone first reaches
    N ≥ n_min on ≥ DAYS_MIN days, or None."""
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(z.grp.values, z.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, sc, now_iso, hold=None, why=None):
    """'first' at the N ≥ 15 crossing; 'reread' at N ≥ 30 on a LATER call. Frozen entries are never recomputed. → (st, changed)."""
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = crossing_prefix(sc, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    be, src = _be(pre)
    zone, rest = pre[pre.grp == "zone"], pre[pre.grp == "rest"]
    state, det = verdict(zone, rest, be)
    st[key] = dict(state=state, detail=det, be=round(be, 2), be_src=src, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso,
                   n=len(zone), days=int(zone.day.nunique()),
                   keys=[f"{a}|{b}|{g}" for a, b, g in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair, pre.grp)])
    return st, True


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Group | N | days | WR | avg % | Σ$ as-sized | worst |", "|---|---|---|---|---|---|---|"]


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean() * 100:.0f} % | {g.pct.mean():+.3f} % | "
            f"{g.usd.sum():+,.0f} | {g.pct.min():+.2f} % |")


LINES = {
    "REENTRY": dict(icon="🔁", name="FRENZY_REENTRY_AFTER_WIN",
                    rule="a FRENZY_LONG / WIDE fill on a pair that already had a FRENZY-family fill (LONG / WIDE / LITE / WILLY) CLOSED at a "
                         "profit (pnl > 0) in the SAME episode (same entry_frenzy_spike_at) before this fill opened",
                    zl="zone (re-entry after a same-episode win)", rl="rest", ul="UNSCORED (no spike stamp / winner without closed_at)"),
    "STRETCHED": dict(icon="📏", name="FRENZY_STRETCHED",
                      rule=f"entry_frenzy_vs_vwap_pct ≥ {STRETCH_CUT:g} % (frozen 2026-10-09 = P75 of the master-pool FRENZY_LONG + WIDE "
                           f"distribution, N {STRETCH_CUT_N}, pnl not read)",
                      zl=f"zone (vs VWAP ≥ {STRETCH_CUT:g} %)", rl=f"rest (< {STRETCH_CUT:g} %)", ul="UNSCORED (no vs-VWAP stamp)"),
}


def _section(key, cnt, ref, now_ms, state_path, hold, dep, off=None):
    m = LINES[key]
    sc = cnt[cnt.grp != "unscored"]
    be, src = _be(sc)
    zone, rest = sc[sc.grp == "zone"], sc[sc.grp == "rest"]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        st, changed = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), hold, why)
        if changed:
            NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest, be)
    L = [f"## {m['icon']} {m['name']} — FRENZY long entry observe line (pre-registered 2026-10-09, OBSERVE only, block side)", "",
         f"Zone (frozen): {m['rule']}. Counted: CLOSED FRENZY_LONG + FRENZY_WIDE LONG fills opened ≥ {dep:%Y-%m-%d %H:%M} UTC (DECISION_LOG 250 "
         f"deploy + 10 min = today's FRENZY stack); DAY units; avg = pnl % (leverage-invariant); FRENZY_LITE shown apart, never counted.", "",
         *HDR, _row(f"**{m['zl']}**", zone), _row(m["rl"], rest), _row(m["ul"], cnt[cnt.grp == "unscored"]),
         _row(f"FRENZY_LITE reference — zone (not counted)", ref[ref.grp == "zone"]),
         _row(f"FRENZY_LITE reference — rest / unscored (not counted)", ref[ref.grp != "zone"]), ""]

    def fmt(r):
        x = f"{r.ts:%m-%d %H:%M} {r.pair} {r.strat.replace('FRENZY_', '')} {r.pct:+.2f} % [{r.grp}]"
        if key == "STRETCHED":
            x += f" (vs VWAP {r.vwap:+.2f} %)" if not np.isnan(r.vwap) else ""
        elif r.why:
            x += f" ({r.why})"
        return x
    if len(cnt):
        L += ["Counted fills: " + " · ".join(fmt(r) for r in cnt.itertuples()), ""]
    if len(ref):
        L += ["FRENZY_LITE reference fills: " + " · ".join(fmt(r) for r in ref.itertuples()), ""]
    L.append(era_line(cnt, off))
    L.append(f"Live so far: {len(cnt)} counted fills, {len(zone)} in the zone — review bar N ≥ {N_MIN} zone fills on ≥ {DAYS_MIN} days; "
             f"information only below it.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    bar = (f"Bar (frozen once at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} days, one re-read at N ≥ {REREAD_N}, "
           f"never re-fit): zone mean ≥ rest mean → RETIRE; zone WR < FRENZY breakeven {be:.1f} % ({src}"
           f"{' until 30 scored fills' if src == 'fallback' else ''}; {len(sc)} scored fills) ∧ day-clustered "
           f"P(mean < 0) ≥ 0.95 ({BOOT_N:,} resamples, seed {BOOT_SEED}) ∧ no day / pair ≥ 50 % of the gross loss → FILTER CANDIDATE "
           f"(operator decides); else KEEP OBSERVING.")
    if "first" not in st:
        L.append(f"{bar} Now: ⏳ {live_state} ({live_det}).")
    else:
        for k in ("first", "reread"):
            if k in st:
                f0 = st[k]
                L.append(f"**Frozen verdict{' (re-read at 30)' if k == 'reread' else ''} (crossing at fill {f0['at']}, frozen on the run of "
                         f"{f0.get('run_at', '?')}, zone N {f0['n']} · {f0['days']} d, breakeven {f0['be']} % {f0['be_src']}): "
                         f"{f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    return L + [REVERT, ""]


def run(now_ms=None, orders=None, state_paths=None, open_ts=None, deploy=None):
    """→ markdown lines for both lines (the caller wraps it in its own try)."""
    now_ms = now_ms or int(time.time() * 1000)
    sp = dict(STATES, **(state_paths or {}))
    fam = family(raw_orders() if orders is None else orders)
    notes = []
    dep = deploy_ts(notes) if deploy is None else pd.Timestamp(deploy)
    if open_ts is None:
        try:
            open_ts = open_family_ts() if orders is None else []
        except Exception:
            open_ts = []
    off = gvol_off_ts()
    win = fam[fam.ts >= dep]
    cnt = win[win.strat.isin(COUNTED)].copy()
    ref = win[win.strat == REF_STRAT].copy()
    out = []
    for key in ("REENTRY", "STRETCHED"):
        c, r = cnt.copy(), ref.copy()
        if key == "REENTRY":
            c["grp"], c["why"] = reentry_group(c, fam) if len(c) else (np.array([], dtype=object), [])
            r["grp"], r["why"] = reentry_group(r, fam) if len(r) else (np.array([], dtype=object), [])
        else:
            c["grp"], r["grp"] = stretched_group(c.vwap), stretched_group(r.vwap)
            c["why"], r["why"] = "", ""
        out += _section(key, c, r, now_ms, sp[key], open_ts, dep, off)
    return out + ([f"Data: {' · '.join(notes)}.", ""] if notes else [])


# ─────────────────────────── cut derivation (distribution only) ───────────────────────────
def derive_cut(path=os.path.join(_ROOT, "reports", "MASTER_POOL_stacked.csv")):
    """re-derives the frozen cut from the master pool's FRENZY_LONG + WIDE vs-VWAP distribution (pnl columns are never loaded)
    → (p75 raw, rounded, N) or None when the pool is absent. Information only — the hard-coded STRETCH_CUT is what the line uses."""
    if not os.path.exists(path):
        return None
    m = pd.read_csv(path, low_memory=False, usecols=["entry_strategy", "entry_frenzy_vs_vwap_pct"])
    v = pd.to_numeric(m[m.entry_strategy.astype(str).isin(COUNTED)].entry_frenzy_vs_vwap_pct, errors="coerce").dropna()
    if not len(v):
        return None
    p = float(np.percentile(v, 75))
    return p, round(p, 1), len(v)


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global EXPORT_GLOB
    import socket
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved_glob, saved_sock = EXPORT_GLOB, socket.socket

    class _NoNet(socket.socket):
        def connect(self, *a, **k):
            raise RuntimeError("network blocked (selftest)")
    socket.socket = _NoNet
    try:
        chk((STRETCH_CUT, STRETCH_CUT_N, N_MIN, DAYS_MIN, REREAD_N, BE_REF, BE_MIN_FILLS, BOOT_N, BOOT_SEED)
            == (13.0, 15, 15, 8, 30, 51.5, 30, 4000, 7), "pre-registered constants pinned")
        chk(COUNTED == ("FRENZY_LONG", "FRENZY_WIDE") and set(FAMILY) == {"FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY"},
            "counted strategies + win-before family pinned")
        # ── the cut: zone ≥ 13.0 inclusive, NaN unscored; live examples
        chk(list(stretched_group([13.0, 12.9999, np.nan, 8.983, 0.815, 1.388, 14.0]))
            == ["zone", "rest", "unscored", "rest", "rest", "rest", "zone"], "stretch zone = vs VWAP ≥ 13.0 (inclusive), NaN unscored")
        try:                                             # info only — the live master grows; the frozen 13.0 is pinned above, never re-asserted
            dc = derive_cut()
            print(f"  info: live master re-derivation of the cut = {dc} (frozen {STRETCH_CUT} stays; not asserted)")
        except Exception as e:
            print(f"  info: cut re-derivation unavailable ({e})")
        # ── re-entry membership edges
        b = dict(direction="LONG", status="CLOSED", pnl_percentage=3.0, pnl=100.0)
        S1, S2 = "2026-10-08T02:25:00", "2026-10-08T13:10:00"
        rows = [dict(b, opened_at="2026-10-08T23:00:08", closed_at="2026-10-08T23:34:43.136661", pair="RLCUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S1, entry_frenzy_vs_vwap_pct=0.815),
                dict(b, opened_at="2026-10-09T03:10:08", closed_at="2026-10-09T03:12:47", pair="RLCUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S1, entry_frenzy_vs_vwap_pct=8.983, pnl_percentage=-3.0, pnl=-212.0),
                # different episode, same pair: an earlier win in S2 does not make an S1… fill zone (and vice versa)
                dict(b, opened_at="2026-10-08T15:00:00", closed_at="2026-10-08T15:30:00", pair="AAAUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T18:00:00", closed_at="2026-10-08T18:30:00", pair="AAAUSDT", entry_strategy="FRENZY_WIDE",
                     entry_frenzy_spike_at=S1, pnl_percentage=-1.0, pnl=-5.0),
                # loss before (same episode) → rest
                dict(b, opened_at="2026-10-08T19:00:00", closed_at="2026-10-08T19:30:00", pair="AAAUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S1, pnl_percentage=1.0, pnl=7.0),
                # LITE win before → zone; WILLY win before → zone
                dict(b, opened_at="2026-10-08T14:00:00", closed_at="2026-10-08T14:20:00", pair="LLLUSDT", entry_strategy="FRENZY_LITE",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T16:00:00", closed_at="2026-10-08T16:20:00", pair="LLLUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2, pnl_percentage=-3.0, pnl=-90.0),
                dict(b, opened_at="2026-10-08T14:00:00", closed_at="2026-10-08T14:20:00", pair="WWWUSDT", entry_strategy="FRENZY_WILLY",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T16:00:00", closed_at="2026-10-08T16:20:00", pair="WWWUSDT", entry_strategy="FRENZY_WIDE",
                     entry_frenzy_spike_at=S2),
                # earlier winner still OPEN when this fill opened (closed after) → does not count
                dict(b, opened_at="2026-10-08T14:00:00", closed_at="2026-10-08T17:00:00", pair="OOOUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T15:00:00", closed_at="2026-10-08T15:10:00", pair="OOOUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2, pnl_percentage=-3.0, pnl=-90.0),
                # missing spike stamp → unscored; MOMENTUM win same pair never counts as family
                dict(b, opened_at="2026-10-08T20:00:00", closed_at="2026-10-08T20:10:00", pair="NNNUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=np.nan, entry_frenzy_vs_vwap_pct=np.nan),
                dict(b, opened_at="2026-10-08T14:00:00", closed_at="2026-10-08T14:10:00", pair="MMMUSDT", entry_strategy="MOMENTUM",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T15:00:00", closed_at="2026-10-08T15:10:00", pair="MMMUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2),
                # a winner OPEN in the exports (status) never counts
                dict(b, opened_at="2026-10-08T14:00:00", closed_at=np.nan, pair="PPPUSDT", entry_strategy="FRENZY_LONG", status="OPEN",
                     entry_frenzy_spike_at=S2),
                dict(b, opened_at="2026-10-08T15:00:00", closed_at="2026-10-08T15:10:00", pair="PPPUSDT", entry_strategy="FRENZY_LONG",
                     entry_frenzy_spike_at=S2)]
        fam = family(pd.DataFrame(rows))
        g, w = reentry_group(fam, fam)
        G = {(r.pair, f"{r.ts:%m-%d %H:%M}"): x for r, x in zip(fam.itertuples(), g)}
        chk(G[("RLCUSDT", "10-09 03:10")] == "zone" and G[("RLCUSDT", "10-08 23:00")] == "rest", "RLC 03:10 after the RLC 23:00 win → zone")
        chk(G[("AAAUSDT", "10-08 18:00")] == "rest", "a win in a DIFFERENT episode on the same pair does not count")
        chk(G[("AAAUSDT", "10-08 19:00")] == "rest", "a same-episode LOSS before → rest")
        chk(G[("LLLUSDT", "10-08 16:00")] == "zone" and G[("WWWUSDT", "10-08 16:00")] == "zone", "LITE / WILLY wins count as 'win before'")
        chk(G[("OOOUSDT", "10-08 15:00")] == "rest", "an earlier win still open when this fill opened does not count")
        chk(G[("NNNUSDT", "10-08 20:00")] == "unscored", "no spike stamp → UNSCORED")
        chk(G[("MMMUSDT", "10-08 15:00")] == "rest" and "MMMUSDT" in set(fam.pair) and (fam.strat == "MOMENTUM").sum() == 0,
            "a non-FRENZY win never counts (not in the family)")
        chk(G[("PPPUSDT", "10-08 15:00")] == "rest", "an OPEN-status earlier fill never counts")
        nc = fam.copy()
        nc.loc[(nc.pair == "RLCUSDT") & (nc.pct > 0), "close_ts"] = pd.NaT
        g2, _ = reentry_group(nc[nc.pair == "RLCUSDT"], nc)
        chk(list(g2) == ["rest", "unscored"], "a same-episode winner without closed_at → UNSCORED (can't tell)")
        nu = fam.copy()
        nu.loc[(nu.pair == "RLCUSDT") & (nu.pct > 0), "usd"] = np.nan
        chk(list(reentry_group(nu[nu.pair == "RLCUSDT"], nu)[0]) == ["rest", "zone"], "missing pnl $ → pnl % > 0 decides the win")
        nl = nu.copy()
        nl.loc[(nl.pair == "RLCUSDT") & (nl.pct > 0), "pct"] = -1.0
        chk(list(reentry_group(nl[nl.pair == "RLCUSDT"], nl)[0]) == ["rest", "rest"], "missing pnl $ and pnl % ≤ 0 → not a win")
        eg = fam[fam.pair == "RLCUSDT"].assign(grp=["rest", "zone"])
        e1, e2 = era_line(eg, None), era_line(eg, pd.Timestamp("2026-10-09 00:00"))
        chk("not in git yet" in e1 and "gate-ON era zone 1 · 0 % · -3.000 % / rest 1" in e1 and "gate-OFF era zone 0 / rest 0" in e1
            and "gate-ON era zone 0 / rest 1" in e2 and "gate-OFF era zone 1" in e2, f"era split at the 263 deploy ({e1} | {e2})")
        # ── loader: CLOSED beats OPEN regardless of export order, dedup key opened_at[:19]
        dup = pd.DataFrame([dict(rows[0], status="CLOSED", opened_at="2026-10-08T23:00:08.5"),
                            dict(rows[0], status="OPEN", pnl_percentage=0.0, pnl=0.0)])
        f2 = family(dup)
        chk(len(f2) == 1 and float(f2.pct.iloc[0]) == 3.0, "dedup (opened_at[:19], pair, direction): CLOSED beats a later OPEN row")
        # ── verdicts: RETIRE precedence, candidate, collecting
        z = pd.DataFrame(dict(ts=[pd.Timestamp("2026-10-10 01:00") + pd.Timedelta(hours=12 * i) for i in range(20)],
                              pct=[-3.0, -3.0, 3.0, -3.0] * 5, day=[f"D{i}" for i in range(20)], pair=[f"P{i}" for i in range(20)],
                              usd=0.0, grp="zone"))
        r = z.assign(grp="rest", pct=3.0, ts=z.ts + pd.Timedelta(minutes=1), pair=[f"R{i}" for i in range(20)])
        chk(verdict(z.iloc[:14], r, 51.5)[0] == "COLLECTING" and verdict(z.iloc[:15].assign(day="D1"), r, 51.5)[0] == "COLLECTING",
            "N < 15 or < 8 days → collecting")
        chk(verdict(z, r, 51.5)[0] == "FILTER CANDIDATE (operator decides)", f"bad zone → candidate ({verdict(z, r, 51.5)})")
        chk(verdict(z, r.assign(pct=-5.0), 99.0)[0] == "RETIRE", "zone mean ≥ rest mean → RETIRE, checked before the candidate bar")
        chk(verdict(z, r, 20.0)[0] == "KEEP OBSERVING", "WR above breakeven → keep observing")
        one = z.assign(pct=np.where(np.arange(20) == 0, -500.0, z.pct))
        chk(verdict(one, r, 51.5)[0] == "KEEP OBSERVING", "one day / pair ≥ 50 % of the gross loss → no candidate")
        # ── bootstrap determinism
        p1 = NF.p_mean_neg(z.pct.values, z.day.values, BOOT_N, BOOT_SEED)
        p2 = NF.p_mean_neg(z.pct.values, z.day.values, BOOT_N, BOOT_SEED)
        chk(p1 == p2 and 0 <= p1 <= 1, f"day bootstrap deterministic (seed 7 · 4,000) ({p1})")
        # ── freeze at the first crossing; never recomputed; OPEN hold; re-read at 30 on a later call
        sc = pd.concat([z, r], ignore_index=True)
        st, ch = freeze({}, sc, "t1")
        last = z.ts.iloc[14]
        chk(ch and st["first"]["n"] == 15 and st["first"]["at"] == f"{last:%Y-%m-%d %H:%M} UTC" and st["first"]["be_src"] == "fallback"
            and st["first"]["be"] == 51.5 and len(st["first"]["keys"]) == len(sc[sc.ts <= last]),
            f"first crossing frozen on its prefix, fallback breakeven < 30 fills ({st['first']})")
        st2, ch2 = freeze(json.loads(json.dumps(st)), sc.assign(pct=-sc.pct).iloc[:29], "t2")
        chk(not ch2 and st2 == json.loads(json.dumps(st)), "a frozen first verdict is never recomputed (re-read needs N ≥ 30)")
        why = []
        sh, chh = freeze({}, sc, "th", [(sc.ts.min(), "OPEN FRENZY_LONG Q")], why)
        chk(not chh and sh == {} and "OPEN FRENZY_LONG Q" in why[0], "an OPEN position opened ≤ the prefix end defers the freeze")
        sh, chh = freeze({}, sc, "th", [(pd.Timestamp("2027-01-01"), "OPEN Z")], [])
        chk(chh, "… one opened after the prefix end does not")
        z30 = pd.concat([z, z.assign(ts=z.ts + pd.Timedelta(days=30), pair=[f"S{i}" for i in range(20)])], ignore_index=True)
        st3, ch3 = freeze(json.loads(json.dumps(st)), pd.concat([z30, r], ignore_index=True), "t3")
        chk(ch3 and st3["reread"]["n"] == 30 and st3["first"] == st["first"], "re-read frozen at zone N 30 on a later call")
        # ── end-to-end through run(): temp exports, temp state, deploy floor
        with tempfile.TemporaryDirectory() as td:
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            extra = [dict(b, opened_at="2026-10-08T21:40:08", closed_at="2026-10-08T21:49:51", pair="SKLUSDT", entry_strategy="FRENZY_LITE",
                          entry_frenzy_spike_at="2026-10-08T13:25:00", entry_frenzy_vs_vwap_pct=3.832, pnl_percentage=-3.0, pnl=-127.0),
                     dict(b, opened_at="2026-10-08T10:00:00", closed_at="2026-10-08T10:10:00", pair="PREUSDT", entry_strategy="FRENZY_LONG",
                          entry_frenzy_spike_at=S1, entry_frenzy_vs_vwap_pct=20.0),
                     dict(b, opened_at="2026-10-09T06:25:08", closed_at="2026-10-09T06:44:06", pair="CTSIUSDT", entry_strategy="FRENZY_LONG",
                          entry_frenzy_spike_at=S2, entry_frenzy_vs_vwap_pct=1.388, pnl_percentage=-3.0, pnl=-118.0),
                     dict(b, opened_at="2026-10-09T07:00:00", closed_at="2026-10-09T07:10:00", pair="BIGUSDT", entry_strategy="FRENZY_WIDE",
                          entry_frenzy_spike_at=S2, entry_frenzy_vs_vwap_pct=14.0, pnl_percentage=-2.0, pnl=-50.0)]
            pd.DataFrame(rows[:2] + extra).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            chk(open_family_ts() == [], "no OPEN family position in the newest export")
            sps = {k: os.path.join(td, os.path.basename(v)) for k, v in STATES.items()}
            out = "\n".join(run(1791490000000, state_paths=sps, open_ts=None, deploy="2026-10-08 13:47:25"))
            chk("## 🔁 FRENZY_REENTRY_AFTER_WIN" in out and "## 📏 FRENZY_STRETCHED" in out, "both sections render")
            chk("| **zone (re-entry after a same-episode win)** | 1 |" in out and "| rest | 3 |" in out
                and "10-09 03:10 RLCUSDT LONG -3.00 % [zone] (after FRENZY_LONG 10-08 23:00 +3.00 %)" in out,
                f"run(): RLC 03:10 in the re-entry zone, pre-deploy PRE fill excluded\n{out}")
            chk("| **zone (vs VWAP ≥ 13 %)** | 1 |" in out and "10-09 07:00 BIGUSDT WIDE -2.00 % [zone] (vs VWAP +14.00 %)" in out
                and "10-09 03:10 RLCUSDT LONG -3.00 % [rest] (vs VWAP +8.98 %)" in out and "PREUSDT" not in out,
                "run(): stretched zone uses the frozen 13.0 cut")
            chk("FRENZY_LITE reference fills: 10-08 21:40 SKLUSDT LITE -3.00 %" in out and "COLLECTING" in out
                and not any(os.path.exists(p) for p in sps.values()), "LITE shown apart; nothing frozen at N 1")
            empty = "\n".join(run(1791490000000, orders=pd.DataFrame(columns=list(COLS)), state_paths=sps, open_ts=[],
                                  deploy="2026-10-08 13:47:25"))
            chk(empty.count("COLLECTING") == 2, "empty orders → collecting, never raises")
            # freeze through run(): 15 zone fills on 15 days, persisted; a later run with flipped pnl keeps it
            zr = []
            for i in range(16):
                t = pd.Timestamp("2026-10-10 00:00:08") + pd.Timedelta(hours=25 * i)
                zr.append(dict(b, opened_at=f"{t:%Y-%m-%dT%H:%M:%S}", closed_at=f"{t + pd.Timedelta(minutes=5):%Y-%m-%dT%H:%M:%S}",
                               pair=f"Z{i}USDT", entry_strategy="FRENZY_LONG", entry_frenzy_spike_at=f"{t:%Y-%m-%dT00:00:00}",
                               entry_frenzy_vs_vwap_pct=15.0, pnl_percentage=[-3.0, -3.0, 3.0][i % 3], pnl=-10.0))
            zo = pd.DataFrame(zr)
            out = "\n".join(run(1791490000000, orders=zo, state_paths=sps, open_ts=[], deploy="2026-10-08 13:47:25"))
            s1 = json.load(open(sps["STRETCHED"]))
            chk("Frozen verdict" in out and s1["first"]["n"] == 15 and not os.path.exists(sps["REENTRY"]),
                f"run() freezes STRETCHED at the 15th zone fill on ≥ 8 days; REENTRY (0 zone) stays collecting ({s1['first']['state']})")
            run(1791490000000, orders=zo.assign(pnl_percentage=3.0), state_paths=sps, open_ts=[], deploy="2026-10-08 13:47:25")
            chk(json.load(open(sps["STRETCHED"])) == s1, "persisted verdict survives a later run unchanged")
            held = "\n".join(run(1791490000000, orders=zo, state_paths={"STRETCHED": os.path.join(td, "h.json"), "REENTRY": sps["REENTRY"]},
                                 open_ts=[(pd.Timestamp("2026-10-10"), "OPEN FRENZY_WIDE X")], deploy="2026-10-08 13:47:25"))
            chk("freezing deferred" in held and not os.path.exists(os.path.join(td, "h.json")), "OPEN hold through run()")
            open(os.path.join(td, "c.json"), "w").write("{bad")
            bad = "\n".join(run(1791490000000, orders=zo, state_paths={"STRETCHED": os.path.join(td, "c.json"), "REENTRY": sps["REENTRY"]},
                                open_ts=[], deploy="2026-10-08 13:47:25"))
            chk("frozen state corrupt" in bad and glob.glob(os.path.join(td, "c.json.*.bad")), "corrupt state → .bad, freezing skipped")
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1)
            chk(False, "network must be blocked")
        except RuntimeError:
            chk(True, "network blocked inside the selftest")
    finally:
        EXPORT_GLOB = saved_glob
        socket.socket = saved_sock
    print(f"selftest FRENZY entry lines OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    elif "--derive-cut" in sys.argv:
        print(derive_cut())
    else:
        print("\n".join(run()))
