#!/usr/bin/env python3
"""👁 Scout — ALPHA_SLEEVES watch lines (pre-registered 2026-11-04, DECISION_LOG 464, operator "go Alpha"; OBSERVE ONLY — never changes config,
no bot API, no network).

QUESTION   Binance re-tagged ~62 coins "Alpha" on Oct-30/31 (the universe filter's excluded count 69 → 131 of 513) and Frenzy's flags fell
           ~8 → 2-3 per pass. The Alpha block was shipped May-5 on MOMENTUM evidence only. In PAPER every group's block is OFF
           (alpha_filter_momentum / _frenzy / _spike): does each sleeve make money on Alpha coins, or does its group's block come back?
EVIDENCE   (NEVER counted) B25 fills on now-Alpha coins before the re-tag: LITE+RETRY 10 · +$1,144 · WILLY 5 · −$1,018 · Flatbase 5 · −$575 ·
           LONG+HT 7 · +$65 · Momentum 2 · +$6 — ~3 of 9.4 Frenzy / Flatbase fills a day were on those coins, 0 since Nov-1.
COHORT     CLOSED full-size fills (*_PROBE out, operator-closed out) from the orders exports (export_glob), dedup (opened_at[:19], pair) — CLOSED
           beats OPEN, newest export wins. COUNTED: opened ≥ the DECISION_LOG 464 commit + 10 min. ALPHA = entry_pair_alpha True (Binance's tag AT
           ENTRY — tags move, today's tag is never used) · CONTROL = False · missing → UNSCORED (never counted). Each sleeve (entry_strategy;
           blank = MOMENTUM) is its own line (FRENZY sleeves are never pooled).
FROZEN BAR per sleeve, decided ONCE at its 15th ALPHA fill (by open time), at 1× (pnl_percentage, leverage-invariant): RE-BLOCK (operator turns
           the group's switch back ON; never automatic) iff the 15 fills' Σ % < 0 ∧ their WR < the sleeve's breakeven WR = |avg loss| / (avg win +
           |avg loss|) of its CONTROL fills to that point (≥ 10 controls; with fewer, Σ % < 0 alone decides) · else KEEP OPEN. Deferred while any
           ALPHA fill of that sleeve opened before the 15th is still OPEN in the newest export. Before LIVE every group is re-decided from these lines.
LIMITS     the tag is Binance's as loaded at the bot's last process start (markets load once per process; a deploy reloads them) — the
           same view the universe filter used for that fill. entry_pair_alpha is NULL on every fill before this ship, so the W/L sweep tests it
           on the post-464 cohort only (it is "blind" on master / older batches). Replays carry NULL (the backfill is skipped there).
CONFOUND   (operator 11-04: "is the problem the Alpha pair, or the watchlist item?") a RE-BLOCK is a CANDIDATE only: before the switch goes back ON,
           the batch review's W/L sweep (entry_pair_alpha is an entry_* stamp → 1D and EVERY-pair 2D with the sleeve's other variables, incl. each
           of its live watch lines' own columns) must show ALPHA still losing OUTSIDE those lines' zones. If the loss sits inside another line's
           zone, that line is judged instead. 🧭 (DECISION_LOG 471) the run prints it: per sleeve, ALPHA fills inside any of the sleeve's block
           lines (scripts/watch_zones.py — the best-case set) vs outside every one, and each closed ALPHA fill with the lines it sits in. Every
           ALPHA fill also counts in its sleeve's own lines like any other fill — this table is an extra view, never a separate bucket.
           MKT_HOT needs Market-hot — % of the rolling top-100 up over 4 h (budgeted public 1h klines, cached — scout_ml_mkt_hot_watch): read only with run(market=True), as the
           hourly scout calls it; an incomplete read → NOT SCORED, never 'outside'. STOP_COOLDOWN (MOMENTUM_LONG) and OBS-F1 (SPIKE_FADE)
           are never scored here (listed under 'Lines not scored') — both groups keep Alpha blocked today.
STATE      reports/SCOUT_ALPHA_SLEEVES.json (gitignored).
Usage: venv/bin/python -I scripts/scout_alpha_sleeves_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (state I/O)
import watch_zones as WZ           # noqa: E402  (every sleeve's block-line zones, DL 471)

DL = 464
N_GATE = 15
MIN_CONTROL = 10
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_ALPHA_SLEEVES.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source", "entry_pair_alpha", "entry_mkt_up4h_pct") + WZ.ZONE_COLS   # 🌡 (DL 489) the V2 Market-hot stamp
REBLOCK = "RE-BLOCK CANDIDATE (operator decides after the 2D confound check; never automatic)"


def group_of(sleeve):
    """sleeve → the config switch that governs its universe. PURE."""
    s = str(sleeve)
    if s.startswith("FRENZY") or s.startswith("FLATBASE"):
        return "alpha_filter_frenzy"
    if s.startswith("SPIKE"):          # SURGE picks from the MAIN-SCAN list (surge_rank) → the momentum switch (464 review)
        return "alpha_filter_spike"
    return "alpha_filter_momentum"


def _flag(v):
    """entry_pair_alpha as exported (True / False / 'True' / 1 / NaN) → True / False / None. PURE."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    t = str(v).strip().lower()
    return True if t in ("true", "1", "1.0", "t") else False if t in ("false", "0", "0.0", "f") else None


def load(paths=None):
    """→ (rows: CLOSED and OPEN full-size fills, dedup, CLOSED beats OPEN, newest export wins; + o, sleeve, alpha, closed)."""
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "sleeve", "alpha", "closed", "pct"])
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["closed"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["closed", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")   # the project dedup key
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    d = d[d.o.notna()].copy()
    d["sleeve"] = d.entry_strategy.fillna("").astype(str).replace({"": "MOMENTUM", "nan": "MOMENTUM"})
    d["alpha"] = [_flag(v) for v in d.entry_pair_alpha.values]
    d["pct"] = pd.to_numeric(d.pnl_percentage, errors="coerce")
    d["zsleeve"] = [WZ.sleeve_name(a, b) for a, b in zip(d.entry_strategy.values, d.direction.values)]
    return d.sort_values(["o", "pair"]).reset_index(drop=True)


def breakeven_wr(control):
    """|avg loss| / (avg win + |avg loss|) of the control fills; None with < MIN_CONTROL fills or no win / no loss. PURE."""
    if len(control) < MIN_CONTROL:
        return None
    w, lo = control.pct[control.pct > 0], control.pct[control.pct <= 0]
    if not len(w) or not len(lo):
        return None
    aw, al = float(w.mean()), abs(float(lo.mean()))
    return al / (aw + al) if aw + al > 0 else None


def verdict(alpha15, control):
    """the first 15 ALPHA fills + the sleeve's control fills to that point → (state, detail). PURE."""
    s, wr = float(alpha15.pct.sum()), float((alpha15.pct > 0).mean())
    be = breakeven_wr(control)
    det = (f"ALPHA 15 · WR {wr:.0%} · Σ {s:+.2f} % · avg {alpha15.pct.mean():+.3f} % · control N {len(control)} "
           f"(avg {control.pct.mean() if len(control) else float('nan'):+.3f} %) · breakeven WR {('%.1f %%' % (be * 100)) if be is not None else 'n/a (< 10 controls → Σ alone)'}")
    bad = s < 0 and (be is None or wr < be)
    return (REBLOCK if bad else "KEEP OPEN"), det


def _row(sleeve, a, c, u, frozen):
    def cell(g):
        if not len(g):
            return "0"
        return f"{len(g)} · {(g.pct > 0).mean():.0%} · {g.pct.mean():+.2f} % · ${pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f}"
    return f"| {sleeve} | `{group_of(sleeve)}` | {cell(a)} | {cell(c)} | {len(u)} | {frozen} |"


HOT_SLEEVES = ("FRENZY_WILLY", "FRENZY_WILLY_HT", "MOMENTUM_LONG")   # sleeves with a Market-hot (MKT_HOT) line


def market_hot(o, now_ms, stamps=None):
    """entry times (naive UTC) [+ the fills' own V2 stamps entry_mkt_up4h_pct] → ML_MKT_HOT zone labels ('zone' / 'rest' / 'unscored'),
    or None on an incomplete Market-hot read. 🌡 V2 rolling top-100 (DL 489): a stamped fill reads its stamp; the others are rebuilt (budgeted,
    cached public 1h klines — the same read as scout_ml_mkt_hot_watch); never raises."""
    try:
        import scout_ml_mkt_hot_watch as MH
        if not len(o):
            return np.array([], dtype=object)
        if stamps is not None:
            st = pd.to_numeric(pd.Series(list(stamps)), errors="coerce")
            if st.notna().all():
                return MH.score(pd.DataFrame({"o": list(o), MH.STAMP: st.values}), None).zone.values
            miss = [x for x, v in zip(o, st) if pd.isna(v)]
            closes, complete = MH.fetch_closes(int(pd.Timestamp(min(miss)).value // 10 ** 6), int(now_ms),
                                               days=sorted({pd.Timestamp(x).normalize() for x in miss}))
            if not complete:
                return None
            return MH.score(pd.DataFrame({"o": list(o), MH.STAMP: st.values}), MH.breadth_series(closes)).zone.values
        closes, complete = MH.fetch_closes(int(pd.Timestamp(min(o)).value // 10 ** 6), int(now_ms),
                                           days=sorted({pd.Timestamp(x).normalize() for x in o}))   # 🌡 V2 rolling top-100 (DL 489)
        if not complete:
            return None
        return MH.score(pd.DataFrame({"o": list(o)}), MH.breadth_series(closes)).zone.values
    except Exception:
        return None


def zone_view(a, hot=None):
    """closed ALPHA fills of ONE sleeve (+ hot labels aligned to a, or None) → (in: rows inside ≥ 1 scored line, out: rows outside every
    scored line, lines per row {index: [names]}, not-scored line names, has_lines). PURE."""
    a = a.copy()
    sl = a.zsleeve.iloc[0] if len(a) else ""
    z = WZ.zones(a, hot=hot).get(sl, {})
    names = {i: [] for i in a.index}
    skipped = [ln for ln, m in z.items() if m is None]
    for ln, m in z.items():
        if m is None:
            continue
        for i in a.index[m.fillna(False).astype(bool).values]:
            names[i].append(ln)
    inn = a[[bool(names[i]) for i in a.index]]
    out = a[[not names[i] for i in a.index]]
    return inn, out, names, skipped, bool(z)


def confound_lines(keep, now_ms, market=False):
    """🧭 (DL 471) the 'Alpha or another watch line?' note → markdown lines."""
    L = ["### 🧭 Alpha or another watch line? — closed ALPHA fills inside the sleeve's own block lines (DECISION_LOG 471)", "",
         "Every ALPHA fill also counts in its sleeve's own lines like any other fill; this only shows where they overlap. A loss inside "
         "another line's zone is that line's evidence first — Alpha is judged on the fills OUTSIDE every line.", "",
         "| Sleeve | ALPHA inside a line's zone (N · Σ % · Σ$) | ALPHA outside every line (N · WR · avg · Σ$) | Lines not scored here |",
         "|---|---|---|---|"]
    cl = keep[keep.closed & keep.pct.notna() & (keep.alpha == True)] if len(keep) else keep   # noqa: E712
    detail, hot_all = [], None
    need = cl[cl.zsleeve.isin(HOT_SLEEVES)] if len(cl) else cl
    if market and len(need):   # ONE Market-hot read per run for every sleeve that needs it
        h = market_hot(need.o.values, now_ms, need["entry_mkt_up4h_pct"].values if "entry_mkt_up4h_pct" in need else None)
        hot_all = None if h is None else pd.Series(h, index=need.index)
    for sl in sorted(cl.zsleeve.unique()) if len(cl) else []:
        a = cl[cl.zsleeve == sl]
        hot = hot_all.reindex(a.index).values if (hot_all is not None and sl in HOT_SLEEVES) else None
        inn, out, names, skipped, has = zone_view(a, hot)
        if not has:
            L.append(f"| {sl} | no block line for this sleeve | {len(a)} (all) | – |")
            continue
        su = lambda g: pd.to_numeric(g.pnl, errors="coerce").sum()
        cin = f"{len(inn)} · {inn.pct.sum():+.2f} % · ${su(inn):+,.0f}" if len(inn) else "0"
        cout = f"{len(out)} · {(out.pct > 0).mean():.0%} · {out.pct.mean():+.2f} % · ${su(out):+,.0f}" if len(out) else "0"
        L.append(f"| {sl} | {cin} | {cout} | {', '.join(skipped) or '–'} |")
        for i, r in a.iterrows():
            detail.append(f"- {r.o:%m-%d %H:%M} {r.pair} {sl} {r.pct:+.2f} % → {', '.join(names[i]) or 'outside every line'}")
    if not len(cl):
        L.append("| – | no closed ALPHA fill yet | – | – |")
    return L + [""] + detail + ([""] if detail else [])


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, market=False):
    """→ markdown lines. since_ms None → the DECISION_LOG 464 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write.
    market True → MKT_HOT scored from the budgeted Market-hot read (the hourly scout); False → listed as not scored (tests, offline)."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁🅰️ ALPHA_SLEEVES — every sleeve on Binance \"Alpha\" coins (tag at entry), per-sleeve frozen re-block gate "
         f"(pre-registered 2026-11-04, DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms is not None
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ALPHA_SLEEVES watch lines")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    d = load(paths)
    d = d[d.o >= since] if len(d) else d
    keep, ox = _GK.split_operator_closed(d) if len(d) else (d, d)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    L += [f"Cohort: full-size fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'floor given' if given else f'DL {DL} commit + 10 min'}) · ALPHA = "
          f"entry_pair_alpha True at entry · {_GK.operator_closed_line(ox)}.", "",
          "| Sleeve | switch | ALPHA closed (N · WR · avg · Σ$) | CONTROL closed | UNSCORED | Frozen gate (15th ALPHA fill) |",
          "|---|---|---|---|---|---|"]
    notes, dirty = [], False
    for sleeve in sorted(keep.sleeve.unique()) if len(keep) else []:
        g = keep[keep.sleeve == sleeve]
        cl = g[g.closed & g.pct.notna()]
        a, c, u = cl[cl.alpha == True], cl[cl.alpha == False], cl[cl.alpha.isna()]   # noqa: E712 (object column)
        rec = st.get(sleeve)
        if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
            rec, ok = None, False
        unpriced = g[g.closed & (g.alpha == True) & g.pct.isna()]   # noqa: E712 — a CLOSED ALPHA fill without a P&L defers, never vanishes
        if not rec and ok and len(a) >= N_GATE:
            a15 = a.iloc[:N_GATE]
            cut = a15.o.max()
            still_open = g[(~g.closed) & (g.alpha == True) & (g.o <= cut)]   # noqa: E712
            if len(still_open) or len(unpriced[unpriced.o <= cut]):
                notes.append(f"{sleeve}: freezing deferred — {len(still_open)} ALPHA fill(s) opened before the 15th still OPEN · "
                             f"{len(unpriced[unpriced.o <= cut])} CLOSED without a P&L")
            else:
                state, det = verdict(a15, c[c.o <= cut])
                rec = dict(state=state, detail=det, at=f"{cut:%Y-%m-%d %H:%M} UTC", switch=group_of(sleeve),
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                           keys=[f"{x:%Y-%m-%dT%H:%M:%S}|{p}" for x, p in zip(a15.o, a15.pair)])
                st[sleeve] = rec
                dirty = True
        frozen = f"**{rec['state']}** ({rec['at']}) — {rec['detail']}" if rec else f"⏳ {min(len(a), N_GATE)}/{N_GATE}"
        L.append(_row(sleeve, a, c, u, frozen))
    if dirty and state_path:
        NF.du_save_state(st, state_path)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(str(state_path))}.*.bad); freezing skipped this run.")
    L += [""] + [f"Note: {n}" for n in notes]
    L += ["Evidence (NEVER counted): B25 fills on now-Alpha coins before the re-tag — LITE+RETRY 10 · +$1,144 · WILLY 5 · −$1,018 · Flatbase 5 · "
          "−$575 · LONG+HT 7 · +$65 · Momentum 2 · +$6.",
          f"FROZEN bar (DL {DL}) per sleeve at its 15th ALPHA fill, 1×: {REBLOCK} iff Σ % < 0 ∧ WR < the sleeve's control breakeven WR "
          f"(≥ {MIN_CONTROL} controls; fewer → Σ alone); else KEEP OPEN. Every group is re-decided from these lines before LIVE.",
          "Confound check before any RE-BLOCK: run `scripts/sweep_separators.py <SLEEVE>` — entry_pair_alpha must still separate losers outside the "
          "sleeve's other watch-line zones (2D cells); a loss that lives inside another line's zone is that line's, not Alpha's.", ""]
    L += confound_lines(keep, now_ms, market)
    return L


if __name__ == "__main__":
    print("\n".join(run()))
