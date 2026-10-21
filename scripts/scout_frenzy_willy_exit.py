#!/usr/bin/env python3
"""🎲➡🔥 Scout — FRENZY_WILLY_EXIT exit shadow (operator hypothesis 2026-10-10, DECISION_LOG 267; OBSERVE only — never changes config).

HYPOTHESIS  FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE should share FRENZY_WILLY-style small fixed targets — a 7-cell grid, operator 2026-10-10:
            TP +frenzy_willy_tp_pct (1.00) or +1.25 net × stop −2 / −2.5 / −3 net, plus +1.25 / −4 (added the same day, the WILLY
            insurance candidate) — instead of today's +frenzy_tp_pct / −frenzy_stop_pct
            (+3 / −3 until DECISION_LOG 283, +1.5 / −3 since); every cell keeps the live cap (frenzy_max_hold_minutes, 12 h). Levels read from trading_config.json. Judged PER SLEEVE.
COHORT      CLOSED FRENZY_LONG / WIDE / LITE fills in the ~/Downloads orders exports, dedup (opened_at[:19], pair, strategy) — never id; a
            CLOSED row beats an OPEN one — opened ≥ the "(DECISION_LOG 283)" commit + 10 min (RE-BASED 2026-10-12: today's exit became
            +1.5 / −3, so the cells are compared with the exit those fills really ran and the parity gate stays meaningful; the +3-era fills
            from the DL 250 floor are out — nothing had frozen; git unavailable → the PLACEHOLDER pin DEPLOY_PIN + 10 min).
            Manually closed fills (MANUAL_*) are excluded and listed — not the walked exit.
WALKER      ONE hybrid path per fill (DECISION_LOG 272, load_path / evaluate below — the B19 RE-RUN §9 fix, 28 / 28 live parity): aggTrades
            ticks for each consecutive covered UTC day (scripts/scout_willy_timecap.py's shared caches), then the 1m tail (provisional) after
            the covered time; every cell whose TP / stop is hit before the path ends is scored, an undecided cell stays unscored (per cell);
            tick ruler = the first print through a line. All exits on the SAME path with the bot's accounting (net = gross − 0.045 − 0.045 ×
            exit/entry). Data: WILLY_TIMECAP's ensure_data
            for a 12 h window (its attempts ledger with a 12 h key, ≤ 2 requests per call — 4 per scout run with WILLY_TIMECAP's own call,
            ≤ 1 archive each — shared cooldown, used-weight stop). Paused (nothing frozen) if the live exit is not a fixed TP. Parity: the LIVE walk must
            reproduce each fill's actual exit (reason + % within 0.05 pts) — printed per sleeve.
VERDICT     (FROZEN, pre-registered — PER SLEEVE, computed ONCE on the first prefix by (open time, pair) of that sleeve's scored fills reaching
            N ≥ 15 on ≥ 8 days; deferred while a fill of that sleeve opened ≤ the prefix end is OPEN, unscored-but-not-final or 1m-provisional
            with its tick archive pending; persisted in reports/SCOUT_FRENZY_WILLY_EXIT.json; never re-fit; one re-read at N ≥ 30).
            Per cell, Δ_i = cell_i − LIVE_i. Cell CANDIDATE (operator decides) iff mean Δ > 0 ∧ day-clustered bootstrap P(mean Δ > 0)
            ≥ 0.95 (4,000, seed 7 — 0.95, not 0.90, because the best of 7 cells is picked: multiple-comparison guard) ∧ no single fill
            > 50 % of Σ Δ ∧ walker parity ≥ 90 %; several qualify → the larger mean Δ (tie → grid order); every cell mean Δ ≤ 0 → KEEP today's exit;
            else KEEP OBSERVING.
CAVEATS     Δ is per fill on the SAME entries: a smaller TP frees FRENZY's slots sooner (median hold printed) — the extra trades that
            would let in are not priced. Tick paths can touch wicks the live poller rides through — read the parity line.
Usage: venv/bin/python scripts/scout_frenzy_willy_exit.py [--selftest]
"""
import os
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import os as _os_gk, sys as _sys_gk; _sys_gk.path.insert(0, _os_gk.path.dirname(_os_gk.path.abspath(__file__))) if _os_gk.path.dirname(_os_gk.path.abspath(__file__)) not in _sys_gk.path else None
import scout_gatekit as _GK   # noqa: E402  🧰 (DECISION_LOG 331) DL-number deploy lookup + operator-closed fills (MANUAL / MANUAL_* / EXTERNAL_CLOSE) out of every gate

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, freeze hold, state I/O)
import scout_willy_timecap as WT                    # noqa: E402  (walker, caches, data fetcher, accounting)
import scout_willy_hold as WH                       # noqa: E402  (the FRENZY live exit from trading_config.json)

SLEEVES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
PROVISIONAL = ("1m", "ticks+1m")                    # (272) a fill with a 1m segment: provisional while its tick archive is pending
TP125 = 1.25
GRID = (("T1_S2", None, 2.0), ("T1_S25", None, 2.5), ("T1_S3", None, 3.0),       # None = frenzy_willy_tp_pct (1.00) from the config
        ("T125_S2", TP125, 2.0), ("T125_S25", TP125, 2.5), ("T125_S3", TP125, 3.0), ("T125_S4", TP125, 4.0))
VARS = tuple(g[0] for g in GRID)
N_MIN, DAYS_MIN, REREAD_N = 15, 8, 30
P_MIN, SHARE_MAX, PARITY_MIN = 0.95, 0.50, 0.90
BOOT_N, BOOT_SEED = 4000, 7
DEPLOY_GREP = "(DECISION_LOG 283)"                   # RE-BASED 2026-10-12 (was "(DECISION_LOG 250)", the +3 / −3 era): today's exit = +1.5 / −3
DEPLOY_PIN = "2026-10-12 23:00:00"                   # PLACEHOLDER (UTC) — used only if git cannot find the DL 283 commit (was 6d3974f 2026-10-08 13:37:25)
DEPLOY_LAG_MIN = 10                                  # the cohort starts 10 min after the commit (deploy lag)
KEEP_LIVE = "KEEP today's exit"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_FRENZY_WILLY_EXIT.json")
STUDY_REF = ("yr5 tick study, CORRECTED 2026-10-10 (DECISION_LOG 270) to today's stack: LONG / WIDE = replay seed 1 (bearish block, live "
             "eligibility) + the setups the market-volume gate refused (gate off since 263), live-eligible pairs only, rows already in the "
             "replay not counted twice; LITE = bearish days removed, eligible pairs (no LITE slot limit — the bug hunt puts LITE at ≈ +0.23 "
             "with slots); one replay seed (not 3 pooled); same walker, +3 reproduced on 100 %; 12 h. %/fill, Δ vs +3 / −3 (P(better)): "
             "LONG 320 · +3/−3 49 % · −0.039 (H1 −0.198 / H2 +0.182) · +1: −2 −0.121 / −2.5 −0.100 / −3 −0.056 (Δ −0.017, P 0.44) · +1.25: "
             "−2 −0.200 / −2.5 −0.212 / −3 −0.153 / −4 −0.104 (Δ −0.065) · WIDE 204 · +3/−3 49 % · −0.047 (H1 +0.045 / H2 −0.176) · +1: "
             "−0.219 / −0.231 / −0.190 · +1.25: −0.166 / −0.198 / −0.160 / −0.210 (best Δ −0.113, P 0.23) · LITE 546 · +3/−3 54 % · +0.267 "
             "· +1: −0.084 / −0.058 / −0.032 · +1.25: −0.074 / −0.042 / −0.010 / −0.001 (every Δ ≤ −0.27, P ≤ 0.01) — no cell beats +3 / −3 "
             "significantly in any sleeve; LONG +1/−3 is a tie (Δ −0.017 here, +0.032 on a cohort without the de-dup). (The 2026-10-09 "
             "reference quoted LONG / WIDE +0.084 / +0.088: gate-on replay, 3 seeds pooled — superseded.) Winners' trough (fills reaching +3, "
             "no stop; from the OLD gate-on cohort, not re-run): LONG avg −2.82 / median −1.94, WIDE −3.14 / −1.78, LITE −2.65 / −1.75")
REVERT = ("Pre-committed revert if a sleeve ever takes one of these cells: its first 15 fills after the switch — Σ % below what the exit "
          "it replaced would have given on the same fills (this walker) → back to that exit. (Today's +1.5 / −3 has its own revert gate: "
          "Scout FRENZY_TP_REVERT, DECISION_LOG 283.)")


def deploy_ts():
    """the DECISION_LOG 283 commit + DEPLOY_LAG_MIN via scout_gatekit (the number anywhere in a 'DECISION_LOG …' list — DECISION_LOG 331);
    not found → the PLACEHOLDER pin + lag after a LOUD warning."""
    import scout_gatekit as _GK
    ct = _GK.tag_commit_ts(DEPLOY_GREP)
    if ct is None:
        if _GK.dl_from_tag(DEPLOY_GREP) is not None:
            _GK.warn_not_evaluable(f"FRENZY_WILLY_TP floor {DEPLOY_GREP} (placeholder pin {DEPLOY_PIN} used)", _GK.dl_from_tag(DEPLOY_GREP))
        return pd.Timestamp(DEPLOY_PIN) + pd.Timedelta(minutes=DEPLOY_LAG_MIN)
    return pd.Timestamp(int(ct) * 1000, unit="ms") + pd.Timedelta(minutes=DEPLOY_LAG_MIN)


def _ns(th=None):
    """one thresholds namespace: None → trading_config.json; a dict (flat or {"thresholds": …}) or a namespace as given."""
    from types import SimpleNamespace
    if th is None:
        return WH._live_th()
    if isinstance(th, dict):
        return SimpleNamespace(**(th["thresholds"] if isinstance(th.get("thresholds"), dict) else th))
    return th


def _num(ns, k, d):
    try:
        v = getattr(ns, k, None)
        return d if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return d


def frenzy_levels(th=None, sleeve="FRENZY_LONG"):
    """the FRENZY live exit (tp %, stop % positive or None, hold min) — scout_willy_hold.exit_rule on trading_config.json. sleeve: FRENZY_LITE
    runs its own TP (frenzy_lite_tp_pct, DECISION_LOG 284) — LONG / WIDE the shared frenzy_tp_pct."""
    tp, st, cap = WH.exit_rule(sleeve, _ns(th))
    return float(tp), (abs(float(st)) if st else None), int(cap)


def load_orders(orders=None):
    """→ (closed FRENZY-family fills, [(opened_at, 'OPEN <pair>')]) — the WILLY_TIMECAP loader's rules, FRENZY sleeves instead of WILLY."""
    if orders is None:
        fr = WT._read_exports()
        raw = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable") if fr else pd.DataFrame(columns=list(WT.COLS))
    else:
        raw = orders
    o = raw.reindex(columns=list(dict.fromkeys(list(WT.COLS) + list(raw.columns))))
    if not len(o):
        return _prep(o), []
    o["_k"] = o.opened_at.astype(str).str[:19]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values("_c", kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    o = o[(o.direction.astype(str) == "LONG") & o.entry_strategy.astype(str).isin(SLEEVES)]
    t = pd.to_datetime(o.opened_at.astype(str).str[:19], format="ISO8601", errors="coerce")
    op = o.status.astype(str).str.upper().eq("OPEN")
    opens = [(a, f"OPEN {p} {s}") for a, p, s in zip(t[op], o.pair[op].astype(str), o.entry_strategy[op].astype(str)) if pd.notna(a)]
    return _prep(o[o.status.astype(str).str.upper() == "CLOSED"]), opens


def _prep(o):
    if not len(o):
        return pd.DataFrame({c: pd.Series(dtype=object) for c in ["pair", "sleeve", "ts", "day", "te", "E", "pct", "notional", "why"]})
    d = WT._prep(o.assign(entry_frenzy_willy_trigger=o.entry_strategy.astype(str)))   # trig carries the sleeve through WT's prep
    return d.rename(columns={"trig": "sleeve"})


# ─────────────────────────── hybrid path (DECISION_LOG 272) ───────────────────────────
# B19 RE-RUN §9 (reports/B19_RERUN_FRENZY_EXITS.md): the old score_fill needed ONE source for the whole 12 h window — ticks only if every
# UTC day to te + 12 h had a full tick file, else 1m for the WHOLE fill → 20 / 28 B19 parity (CFX LONG 10-10 15:25 walked on 1m because
# the 10-11 tick day was missing: its first 1m bar opens through the stop, −3.93 vs live −3.00; 7 fills of 10-11 left unscored though the
# TP / stop was hit minutes after entry). load_path + evaluate below are the study's (scripts/study_b19_fzexit_common.py, untracked —
# copied here verbatim so the tracked scouts never import a study) with 28 / 28 B19 live parity: tick prints for each consecutive UTC day
# with a full-day file (or a covering window slice), then the 1m tail strictly after the covered time; a cell is scored as soon as its
# TP / stop is hit before the path ends, only a cell still undecided at the path end stays UNRES. Shared by scout_frenzy_exits
# (WILLY_TURNOVER_BLOCKED) and scout_lite_stretch_reentry. No stop-floor clamp: DECISION_LOG 271 removed the live-only floor (paper = live).
def load_path(pair, te, E, horizon, tcache=None, k1m=None):
    """→ dict(t, hi, lo, cl, op, bar, end_ms, n_tick, n_bar, tick_end, lim, te) — net % events strictly after te up to te + horizon (+1 min
    slack): ticks while consecutive covered days last, then 1m bars (k1m = DataFrame open_time / o / h / l / c, default WT.load_1m)."""
    MIN, DAY = WT.MIN, WT.DAY_MS
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
    if cov < lim:                                              # 1m tail (provisional: entry minute skipped, both lines = stop)
        k = WT.load_1m(pair) if k1m is None else k1m
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
                bar=np.concatenate(BAR), end_ms=int(end), n_tick=int(len(t)), n_bar=int(sum(len(x) for x in T[1:])),
                tick_end=int(cov), lim=int(lim), te=int(te))


def evaluate(P, tp, sl, cap):
    """→ dict(why TP / STOP / CAP / UNRES, pct (tick ruler: the first print through the line = the live replica, the engine checks every
    @trade print), pct_line, xmin, worst, on_bar) — first-touch on the net path. 1m bars: TP booked at the line, a bar touching both lines
    = stop, a bar opening through the stop fills at its open. Undecided before the path end (or no event in the cap) → UNRES."""
    MIN = WT.MIN
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
        return dict(why="STOP", pct=v, pct_line=-float(sl), xmin=(P["t"][b] - te) / MIN, worst=float(lo[:b + 1].min()), on_bar=bool(P["bar"][b]))
    if a is not None:
        v = float(tp) if P["bar"][a] else float(hi[a])
        return dict(why="TP", pct=v, pct_line=float(tp), xmin=(P["t"][a] - te) / MIN, worst=float(lo[:a + 1].min()), on_bar=bool(P["bar"][a]))
    if P["end_ms"] < te + cap * MIN or n == 0:
        return dict(why="UNRES", pct=np.nan, pct_line=np.nan, xmin=np.nan, worst=float(lo.min()) if n else np.nan, on_bar=False)
    return dict(why="CAP", pct=float(P["cl"][n - 1]), pct_line=float(P["cl"][n - 1]), xmin=float(cap), worst=float(lo.min()),
                on_bar=bool(P["bar"][n - 1]))


def path_src(P):
    """the fill's source label: 'ticks' (ticks cover the whole window) · 'ticks+1m' (ticks, then the 1m tail — or nothing yet — after the
    covered time) · '1m' (no tick print). Anything but 'ticks' is PROVISIONAL and keeps the freeze deferral while its archive is pending."""
    return "ticks" if P["tick_end"] >= P["lim"] else ("ticks+1m" if P["n_tick"] else "1m")


# ─────────────────────────── scoring ───────────────────────────
def score_fill(r, live, alts, tcache=None):
    """→ (dict or None, source). live = (tp, stop, hold); alts = {name: (tp, stop)} with live's hold. ONE hybrid path (ticks → 1m tail,
    DECISION_LOG 272); every cell decided before the path end is scored, an undecided cell is NaN (why UNRES). None when LIVE is UNRES.
    live may be {sleeve: (tp, stop, hold)} — each fill walks its own sleeve's live exit (FRENZY_LITE's own TP, DECISION_LOG 284)."""
    if isinstance(live, dict):
        live = live.get(getattr(r, "sleeve", None), live["FRENZY_LONG"])
    tp, st, hold = live
    P = load_path(r.pair, int(r.te), float(r.E), hold, tcache)
    la = evaluate(P, tp, st, hold)
    if la["why"] == "UNRES":
        return None, None
    ok, txt = WT.parity(r.why, r.pct, {"v": {"LIVE": la}})
    rec = dict(LIVE=la["pct"], why_LIVE=la["why"], min_LIVE=la["xmin"], par_ok=ok, par_txt=txt)
    evs = [la]
    for name, (x, y) in alts.items():
        b = evaluate(P, x, y, hold)
        evs.append(b)
        rec.update({name: b["pct"], f"why_{name}": b["why"], f"min_{name}": b["xmin"]})
    rec["n_unres"] = sum(e["why"] == "UNRES" for e in evs)      # (272 review I2) undecided cells → the fill stays pending
    on_ticks = P["n_tick"] > 0 and all(e["why"] != "UNRES" and not e["on_bar"] for e in evs)
    return rec, ("ticks" if on_ticks else path_src(P))          # (review M8) every cell decided on tick prints → 'ticks' (final)


def scored_frame(o, live, alts, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = score_fill(r, live, alts, tcache)
        rows.append(dict(src=src, **(w or {})))
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


def _pending(src, n_unres, ticks_pend, page_due):
    """a fill that can still change → holds the freeze: unscored, or any undecided cell (272 review I2), while a tick archive can arrive or
    its 1m page is still due; a fill with a 1m part (provisional) while a tick archive can arrive. Decided on ticks = final."""
    n_unres = 0 if n_unres is None or n_unres != n_unres else int(n_unres)
    if not isinstance(src, str) or n_unres > 0:
        return bool(ticks_pend or page_due)
    return bool(src in PROVISIONAL and ticks_pend)


# ─────────────────────────── verdict / freeze ───────────────────────────
def judge(sc, v):
    par = float(sc.par_ok.astype(bool).mean()) if len(sc) else 0.0     # parity on every scored fill
    sc = sc[pd.to_numeric(sc[v], errors="coerce").notna()]            # (272) a cell undecided at the path end is not scored for that cell
    d = sc[v].values.astype(float) - sc.LIVE.values.astype(float)
    s = float(d.sum())
    p = NF.p_mean_neg(-d, sc.day.values, BOOT_N, BOOT_SEED) if len(sc) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    m = float(d.mean()) if len(d) else float("nan")
    return dict(mean=m, sum=s, p=p, share=share, par=par, n=int(len(d)),
                q=bool(len(d) >= N_MIN and m > 0 and p is not None and p >= P_MIN and share <= SHARE_MAX and par >= PARITY_MIN))   # round-2: per-cell N gate


def verdict(sc):
    n, nd = len(sc), sc.day.nunique() if len(sc) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    J = {v: judge(sc, v) for v in VARS}
    det = f"N {n} · {nd} d · parity {J[VARS[0]]['par'] * 100:.0f} % · " + " · ".join(
        f"{v} (N {j['n']}): mean Δ {j['mean']:+.3f} pts · Σ Δ {j['sum']:+.2f} · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
        f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ" for v, j in J.items())
    qs = [v for v in VARS if J[v]["q"]]
    if qs:
        best = max(qs, key=lambda v: (round(J[v]["mean"], 12), -VARS.index(v)))
        return f"{best} CANDIDATE (operator decides)", det, J
    if all(J[v]["mean"] <= 0 for v in VARS):
        return KEEP_LIVE, det, J
    return "KEEP OBSERVING", det, J


def prefix(sc, n_min):
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, sleeve, sc, now_iso, cfg, hold=None, why=None):
    """per sleeve: 'first' (N ≥ 15 on ≥ 8 days) ONCE, 'reread' (N ≥ 30) on a later call. Never frozen below the parity floor."""
    s = st.setdefault(sleeve, {})
    key = "reread" if "first" in s else "first"
    if key in s:
        return st, False
    pre = prefix(sc, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    if pre.par_ok.astype(bool).mean() < PARITY_MIN:
        if why is not None:
            why.append(f"{sleeve}: WALKER PARITY below {PARITY_MIN * 100:.0f} % — not frozen (operator review)")
        return st, False
    state, det, _ = verdict(pre)
    s[key] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                  n_1m=int((pre.src != "ticks").sum()), config=cfg,
                  keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


# ─────────────────────────── run ───────────────────────────
HDR = ["| Sleeve | Exit | N | days | WR | avg % | Σ % | Σ$ at the fill's size | median min held |",
       "|---|---|---|---|---|---|---|---|---|"]


def _row(sl, lab, g, v):
    if not len(g):
        return f"| {sl} | {lab} | 0 | – | – | – | – | – | – |"
    g = g[pd.to_numeric(g[v], errors="coerce").notna()]                 # (272) per cell: only fills whose cell is decided
    if not len(g):
        return f"| {sl} | {lab} | 0 | – | – | – | – | – | – |"
    p = g[v].astype(float)
    return (f"| {sl} | {lab} | {len(g)} | {g.day.nunique()} | {(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % | "
            f"{(p * g.notional.astype(float) / 100).sum():+,.0f} | {g[f'min_{v}'].median():.0f} |")


def run(now_ms=None, orders=None, fetch=True, state_path=None, open_ts=None, th=None):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    WT._K1.clear()
    live = frenzy_levels(th)
    live_by = {s: frenzy_levels(th, s) for s in SLEEVES}   # DECISION_LOG 284: LITE walks its own TP
    _t = _ns(th)
    paused = _num(_t, "frenzy_lock_arm_pct", 0.0) > 0 or _num(_t, "frenzy_tp_pct", 3.0) <= 0
    alts = {n: (float(WT.levels(vars(_t))[0]) if tp_ is None else tp_, sl_) for n, tp_, sl_ in GRID}
    o, opens = load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    opens = [x for x in opens if pd.notna(x[0]) and x[0] >= floor]       # a stale pre-floor OPEN row never holds a freeze
    man = o[_GK.reason_mask(o.why)]       # operator closes: not the walked exit → out, listed
    o = o[~_GK.reason_mask(o.why)].reset_index(drop=True)
    span = live[2]
    tc = {}
    _pre = [score_fill(r, live_by, alts, tc) for r in o.itertuples()]
    o["src"] = [x[1] for x in _pre]
    o["need_1m"] = [bool(x[0] and x[0].get("n_unres", 0)) for x in _pre]   # (272 review I2) a scored fill with an undecided cell still
    notes = WT.ensure_data(o, span, now_ms, fetch=fetch)                    # needs its 1m page
    if notes:
        tc.clear()
        WT._K1.clear()
    sc_all = scored_frame(o.drop(columns=["src"]), live_by, alts, tc)
    done = WT.load_attempts()["done"]
    sc_all["pend"] = [_pending(s, getattr(r, "n_unres", 0), WT.ticks_pending(r, span, done, now_ms, tc),
                               WT._kk(r.pair, r.te, WT._span(span)) not in done) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = sc_all[sc_all.src.notna()].copy()
    uns = sc_all[sc_all.src.isna()]
    labs = {"LIVE": (f"+{live[0]:g} / −{live[1]:g} (live)" if live[1] else f"+{live[0]:g} (live)")
            + (f" · LITE +{live_by['FRENZY_LITE'][0]:g}" if live_by["FRENZY_LITE"][0] != live[0] else "")}
    labs.update({n: f"+{tp_:g} / −{sl_:g}" for n, (tp_, sl_) in alts.items()})
    L = [f"## 🎲➡🔥 FRENZY_WILLY_TP — should FRENZY / WIDE / LITE take a small fixed target (+{alts['T1_S2'][0]:g} or +{TP125:g}) with a "
         f"−2 / −2.5 / −3 stop, or +{TP125:g} / −4? (DECISION_LOG 267 · 269, OBSERVE only, exit shadow grid)", "",
         f"Cohort: CLOSED FRENZY_LONG / WIDE / LITE fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC (today's +{live[0]:g} / −{live[1] or 0:g} exit went live — {DEPLOY_GREP} + {DEPLOY_LAG_MIN} min; re-based 2026-10-12), re-walked from "
         f"each fill's own entry on the same hybrid path (ticks first, then the 1m tail — DECISION_LOG 272; WILLY_TIMECAP's caches; a 1m part = provisional; a cell undecided at the path end is not counted for that cell) with the bot's accounting. Every exit "
         f"keeps the live {live[2]}-min cap; TP and stop differ per cell. Judged per sleeve.", ""]
    if paused:
        L.append("⏸ PAUSED: the live FRENZY exit is not a fixed TP (frenzy_lock_arm_pct > 0 or frenzy_tp_pct ≤ 0) — the '(live)' row below is "
                 "NOT what live trades; no Δ is judged and nothing is frozen until the fixed TP is back.")
    L += [*HDR]
    for sl in SLEEVES:
        g = sc[sc.sleeve == sl]
        L += [_row(sl, f"**{labs[v]}**" if v == "LIVE" else labs[v], g, v) for v in ("LIVE",) + VARS]
    L.append("")
    if len(sc):
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.sleeve.replace('FRENZY_', '')} [{r.src}{', provisional' if r.src in PROVISIONAL else ''}] live "
            f"{r.LIVE:+.2f} ({r.why_LIVE} @ {r.min_LIVE:.0f} min) → " + " / ".join(
                f"{labs[v]} " + (f"{getattr(r, v):+.2f}" if getattr(r, v) == getattr(r, v) else "undecided") for v in VARS)
            for r in sc.itertuples()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (live walk vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(par.sum())}/{len(sc)} match"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.par_txt}" for r in sc[~par].itertuples())))
        if sc.src.isin(PROVISIONAL).any():
            L.append(f"⏳ {int(sc.src.isin(PROVISIONAL).sum())} fill(s) with a 1m segment (ticks first, then 1m klines — DECISION_LOG 272 "
                     "hybrid; PROVISIONAL on the 1m part: targets booked exactly, a bar touching both lines = stop, the entry minute's bar is "
                     "skipped) — re-walked on aggTrades once the daily archive is out.")
        L.append("")
    if len(man):
        L.append(f"Operator-closed: {len(man)} — excluded, manually closed fill(s) (not the walked exit): "
                 + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {str(r.sleeve).replace('FRENZY_', '')} ({r.why})" for r in man.itertuples()) + ".")
    if len(uns):
        L.append("UNSCORED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {str(r.sleeve).replace('FRENZY_', '')}"
                                                       + ("" if r.pend else " (final — no data)")
                                                       for r in uns.itertuples()))
    why = []
    if paused:
        why.append("the live FRENZY exit is not a fixed TP (frenzy_lock_arm_pct > 0 or frenzy_tp_pct ≤ 0) — the walker cannot model it, "
                   "line paused (nothing frozen)")
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok and not paused:
        changed = False
        for sl in SLEEVES:
            g = sc[sc.sleeve == sl]
            hold_l = [x for x in opens if x[1].endswith(sl)] + [(r.ts, f"{'unscored' if not isinstance(r.src, str) else 'undecided cell' if (getattr(r, 'n_unres', 0) or 0) > 0 else '1m-provisional'} {r.pair}")
                                                               for r in sc_all[sc_all.sleeve == sl].itertuples() if r.pend]
            st, ch = freeze(st, sl, g, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                            dict(live=(list(live_by[sl]) if sl in live_by else {k: list(v) for k, v in live_by.items()}), alts=alts), hold_l, why)
            changed |= ch
        st = {k: v for k, v in st.items() if v}
        if changed:
            NF.du_save_state(st, state_path)
    L.append(f"Bar (pre-registered, PER SLEEVE, frozen once at the first prefix of N ≥ {N_MIN} closed fills on ≥ {DAYS_MIN} days, never re-fit; "
             f"one re-read at N ≥ {REREAD_N}): per cell Δ = cell − live per fill; cell CANDIDATE (operator decides) iff mean Δ > 0 ∧ "
             f"day-clustered P(mean Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > {SHARE_MAX * 100:.0f} % of Σ Δ ∧ walker "
             f"parity ≥ {PARITY_MIN * 100:.0f} % (0.95: the best of {len(VARS)} cells is picked); several → the larger Δ; all ≤ 0 → {KEEP_LIVE}; "
             f"else keep observing.")
    for sl in SLEEVES:
        g = sc[sc.sleeve == sl]
        s0 = st.get(sl, {}) if ok else {}
        if "first" not in s0:
            L.append(f"{sl}: ⏳ {verdict(g)[0]} ({verdict(g)[1]}).")
        else:
            for k in ("first", "reread"):
                if k in s0:
                    f0 = s0[k]
                    L.append(f"{sl}: **Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {f0['at']}, N {f0['n']} · {f0['days']} d, "
                             f"{f0.get('n_1m', 0)} on 1m only): {f0['state']}** ({f0['detail']})")
            L.append(f"{sl} live (information only): {verdict(g)[0]} — {verdict(g)[1]}")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L.append("Δ is per fill on the SAME entries: a smaller TP frees FRENZY's slots sooner (median minutes held above); the extra trades that "
             "would let in are not priced.")
    L += [REVERT, f"Study reference, not part of the verdict — {STUDY_REF}."]
    if notes:
        L.append(f"Data: {' · '.join(notes)}.")
    return L + [""]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED)
    WT._NET_BLOCKED = True
    try:
        chk((SLEEVES, VARS, N_MIN, DAYS_MIN, REREAD_N, P_MIN, SHARE_MAX, PARITY_MIN, BOOT_N, BOOT_SEED) ==
            (("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"), ("T1_S2", "T1_S25", "T1_S3", "T125_S2", "T125_S25", "T125_S3", "T125_S4"),
             15, 8, 30, 0.95, 0.50, 0.90, 4000, 7), "pre-registered constants pinned")
        th = {"frenzy_tp_pct": 3.0, "frenzy_stop_pct": 3.0, "frenzy_max_hold_minutes": 720, "frenzy_willy_tp_pct": 1.0,
              "frenzy_willy_stop_pct": 0.0, "frenzy_willy_max_hold_minutes": 120}
        _two = {**th, "frenzy_tp_pct": 1.5, "frenzy_lite_tp_pct": 3.0}
        chk(frenzy_levels(_two, "FRENZY_LITE")[0] == 3.0 and frenzy_levels(_two)[0] == 1.5 and frenzy_levels(_two, "FRENZY_WIDE")[0] == 1.5,
            "LITE walks its own TP (DECISION_LOG 284)")
        chk(frenzy_levels(th) == (3.0, 3.0, 720) and WT.levels(th) == (1.0, None, 120), "both exits read from the config")
        E, te = 100.0, 1_800_000_000_000
        px = lambda net: E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)   # noqa: E731
        t = te + np.array([1, 5, 30, 200]) * WT.MIN
        a = WT.walk_ticks(t, [px(-1.0), px(1.1), px(-3.1), px(3.2)], te, E, 3.0, 3.0, 720)["v"]["LIVE"]
        b = WT.walk_ticks(t, [px(-1.0), px(1.1), px(-3.1), px(3.2)], te, E, 1.0, 3.0, 720)["v"]["LIVE"]
        c = WT.walk_ticks(t, [px(-1.0), px(1.1), px(-3.1), px(3.2)], te, E, TP125, 3.0, 720)["v"]["LIVE"]
        chk(a["why"] == "STOP" and abs(a["pct"] + 3.1) < 1e-9 and b["why"] == "TP" and abs(b["pct"] - 1.1) < 1e-9 and c["why"] == "STOP",
            "+3: −3 stop · TP1: +1.1 first · TP125: never reached, −3 stop")

        def coh(d, n=15):
            days = [f"2026-10-{10 + i % 8:02d}" for i in range(n)]
            return pd.DataFrame(dict(pair=[f"P{i}" for i in range(n)], ts=pd.to_datetime(days) + pd.to_timedelta(np.arange(n), "min"), day=days,
                                     LIVE=0.0, par_ok=True, src="ticks", notional=1000.0, **{v: d for v in VARS}))
        chk(verdict(coh([1.0] * 15))[0] == "T1_S2 CANDIDATE (operator decides)", "all better, tie → grid order")
        chk(verdict(coh([1.0] * 15).assign(T125_S25=2.0))[0] == "T125_S25 CANDIDATE (operator decides)", "the larger Δ wins")
        chk(verdict(coh([-1.0] * 15))[0] == KEEP_LIVE, "all worse → keep today's exit")
        chk(verdict(coh([0.0] * 14 + [5.0]))[0] == "KEEP OBSERVING", "one fill carries Σ → observe")
        chk(verdict(coh([1.0] * 15).assign(par_ok=[False] * 3 + [True] * 12))[0] == "KEEP OBSERVING", "parity < 90 % → no candidate")
        chk(verdict(coh([1.0] * 15).iloc[:5])[0] == "COLLECTING", "N < 15 → collecting")
        st, ch = freeze({}, "FRENZY_LONG", coh([1.0] * 15), "now", {})
        chk(ch and st["FRENZY_LONG"]["first"]["state"].startswith("T1_S2") and "FRENZY_WIDE" not in st, "frozen per sleeve")
        st2, ch2 = freeze(st, "FRENZY_LONG", coh([-1.0] * 15), "x", {})
        chk(not ch2 and st2["FRENZY_LONG"]["first"]["state"].startswith("T1_S2"), "never re-fit (N 15 < the 30 re-read)")
        chk(not freeze({}, "FRENZY_LONG", coh([1.0] * 15), "x", {}, [(pd.Timestamp("2026-10-10"), "OPEN X FRENZY_LONG")], [])[1], "open fill defers")
        with tempfile.TemporaryDirectory() as td:
            WT._CACHE = os.path.join(td, "cache")
            WT.MY_CACHE = os.path.join(WT._CACHE, "scout_willy_timecap")
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            WT.EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            base = dict(direction="LONG", status="CLOSED", notional_value=1000.0, entry_price=E, pnl=1.0, closed_at="x")
            pd.DataFrame([dict(base, opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", entry_strategy="FRENZY_LITE", pnl_percentage=-3.1,
                               close_reason="STOP_LOSS"),
                          dict(base, opened_at="2026-10-09T09:00:00", pair="NOKUSDT", entry_strategy="FRENZY_LONG", pnl_percentage=3.0,
                               close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-09T09:30:00", pair="WILUSDT", entry_strategy="FRENZY_WILLY", pnl_percentage=1.0,
                               close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-01T09:30:00", pair="OLDUSDT", entry_strategy="FRENZY_LONG", pnl_percentage=1.0,
                               close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-09T10:00:00", pair="MANUSDT", entry_strategy="FRENZY_WIDE", pnl_percentage=0.5,
                               close_reason="MANUAL_TP")]).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            tk = pd.Timestamp("2026-10-09 08:45:09").value // 10**6
            os.makedirs(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT"))
            np.savez_compressed(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT", "2026-10-09.npz"),
                                t=(tk + np.array([0, 60_000, 300_000, 1_800_000])).astype(np.int64),
                                p=np.array([E, px(-1.0), px(1.1), px(-3.1)], dtype=np.float64))
            # the shared tick ledger: a SLICED archive day merges windows and is never final for another line's uncovered fill
            chk(WT._arch_final({"k": 123}, "k") and not WT._arch_final({"k": "sliced 9"}, "k") and not WT._arch_final({}, "k"), "sliced ≠ final")
            WT._save_slice("ZZUSDT", "2026-10-09", [1, 2], [1.0, 2.0], [(0, 10)])
            WT._save_slice("ZZUSDT", "2026-10-09", [2, 50], [2.0, 5.0], [(40, 60)])
            with np.load(WT._slice_file("ZZUSDT", "2026-10-09")) as z:
                chk(z["t"].tolist() == [1, 2, 50] and sorted(zip(z["w0"].tolist(), z["w1"].tolist())) == [(0, 10), (40, 60)],
                    "a second line's slice is merged (prints + windows), never overwritten")
            chk(WT._have_day("ZZUSDT", "2026-10-09", (45, 55)) and not WT._have_day("ZZUSDT", "2026-10-09", (20, 30)), "coverage per window")
            # ── (272) hybrid walk: day 1 = a full tick file, day 2 = 1m klines only (the B19 10-11 situation)
            th0 = pd.Timestamp("2026-10-09 22:00:00").value // 10**6
            d2 = pd.Timestamp("2026-10-10").value // 10**6

            def hy(pair, nets, bars=(), bars_from=d2):
                os.makedirs(os.path.join(WT._CACHE, "ticks_q", pair), exist_ok=True)
                np.savez_compressed(os.path.join(WT._CACHE, "ticks_q", pair, "2026-10-09.npz"),
                                    t=(th0 + np.arange(1, len(nets) + 1) * 600_000).astype(np.int64), p=np.array([px(x) for x in nets]))
                if bars:
                    os.makedirs(os.path.join(WT.MY_CACHE, "1m"), exist_ok=True)
                    pd.DataFrame([dict(open_time=bars_from + i * WT.MIN, o=px(o), h=px(h), l=px(l), c=px(c)) for i, (o, h, l, c) in enumerate(bars)]
                                 ).to_csv(os.path.join(WT.MY_CACHE, "1m", f"{pair}.csv"), index=False)
                WT._K1.clear()
                return SimpleNamespace(pair=pair, te=th0, E=E, why="STOP_LOSS", pct=-3.0)
            from types import SimpleNamespace
            lv = (3.0, 3.0, 720)
            al = {"T1_S2": (1.0, 2.0), "T125_S4": (1.25, 4.0)}
            ra = hy("HYAUSDT", [-1.0, -3.07, 3.5], [(0.0, 0.5, -0.5, 0.0)] * 5)
            w, s_ = score_fill(SimpleNamespace(**{**vars(ra), "pct": -3.07}), lv, al, {})
            chk(s_ == "ticks" and w["why_LIVE"] == "STOP" and abs(w["LIVE"] + 3.07) < 1e-6 and w["par_ok"]
                and w["why_T125_S4"] == "TP" and abs(w["T125_S4"] - 3.5) < 1e-6, f"stop hit on the day-1 ticks → tick-priced; every cell on ticks → 'ticks' (review M8) ({w}, {s_})")
            rb = hy("HYBUSDT", [-1.0, 1.1, -0.5], [(0.0, 0.5, -0.5, 0.0), (0.2, 3.2, -0.4, 3.1)])
            w, s_ = score_fill(SimpleNamespace(**{**vars(rb), "why": "FRENZY_TP", "pct": 3.0}), lv, al, {})
            chk(s_ == "ticks+1m" and w["why_LIVE"] == "TP" and w["LIVE"] == 3.0 and abs(w["min_LIVE"] - (d2 + 2 * WT.MIN - th0) / WT.MIN) < 1e-9
                and abs(w["T1_S2"] - 1.1) < 1e-6 and w["par_ok"], f"TP in the 1m tail → line-priced; +1 on the ticks before it ({w})")
            # CFX 10-10 15:25 shape: the stop is crossed on the ticks at −3.02, the 1m bar of that minute OPENS through it (−3.93 on 1m)
            rc = hy("HYCUSDT", [-1.0, -3.02], [(-0.5, 0.0, -1.0, -0.5)] * 10 + [(-3.93, -3.0, -4.2, -3.5)] + [(-3.5, -3.0, -4.0, -3.5)] * 400,
                    bars_from=th0)
            w, s_ = score_fill(rc, lv, al, {})
            old = WT.walk_1m(WT.load_1m("HYCUSDT"), th0, E, 3.0, 3.0, 720)["v"]["LIVE"]
            chk(abs(w["LIVE"] + 3.02) < 1e-6 and w["par_ok"] and old["pct"] < -3.9,
                f"CFX-like: hybrid −3.02 on the ticks; a whole-window 1m walk books the gap open −3.93 ({w['LIVE']} vs {old['pct']})")
            rd = hy("HYDUSDT", [-1.0, 1.1])
            chk(score_fill(SimpleNamespace(**{**vars(rd), "why": "FRENZY_TP", "pct": 3.0}), lv, al, {}) == (None, None),
                "LIVE undecided when the path ends (no 1m tail yet) → unscored")
            re_ = hy("HYEUSDT", [-1.0, -3.05])
            w, s_ = score_fill(re_, lv, al, {})
            chk(s_ == "ticks+1m" and abs(w["LIVE"] + 3.05) < 1e-6 and abs(w["T1_S2"] + 3.05) < 1e-6 and w["why_T125_S4"] == "UNRES"
                and np.isnan(w["T125_S4"]), "per cell: LIVE / −2 decided on the ticks, −4 still undecided at the path end → NaN for that cell only")
            jj = pd.DataFrame(dict(day=["a", "b"], LIVE=[0.0, 0.0], par_ok=[True, False], T1_S2=[1.0, np.nan]))
            chk(judge(jj, "T1_S2")["mean"] == 1.0 and judge(jj, "T1_S2")["par"] == 0.5 and judge(jj, "T1_S2")["n"] == 1,
                "judge: per-cell NaN dropped (per-cell N), parity on every scored fill")
            chk(w["n_unres"] == 1 and s_ == "ticks+1m", "the HYE fill carries 1 undecided cell")
            # (272 review I2 / M8) pending: an undecided cell holds the freeze while data can still come; decided on ticks = final
            chk(_pending("ticks+1m", 1, False, True) and _pending("ticks+1m", 1, True, False) and not _pending("ticks+1m", 1, False, False)
                and _pending(None, 0, False, True) and _pending("1m", 0, True, False) and not _pending("1m", 0, False, True)
                and not _pending("ticks", 0, True, True), "pending: undecided cell / unscored hold while ticks or the 1m page can come")
            wa, sa = score_fill(SimpleNamespace(**{**vars(ra), "pct": -3.07}), lv, {"T1_S2": (1.0, 2.0)}, {})
            chk(sa == "ticks" and wa["n_unres"] == 0, f"every cell decided on tick prints → 'ticks' though the window has a 1m part ({sa})")
            Jd = coh([1.0] * 15).assign(T1_S2=[np.nan] + [1.0] * 14)
            chk("T1_S2 (N 14)" in verdict(Jd)[1] and "T1_S25 (N 15)" in verdict(Jd)[1], "per-cell N printed in the verdict detail")
            sp = os.path.join(td, "s.json")
            global deploy_ts
            _dts = deploy_ts
            deploy_ts = lambda: pd.Timestamp("2026-10-08 13:37:25")   # noqa: E731  (the fixture's fills predate the DL 283 floor)
            try:
                out = "\n".join(run(pd.Timestamp("2026-10-09 23:00").value // 10**6, fetch=False, state_path=sp, th=th))
            finally:
                deploy_ts = _dts
            chk(_dts() >= pd.Timestamp("2026-10-12") + pd.Timedelta(minutes=DEPLOY_LAG_MIN), "re-based floor: the DL 283 commit (or pin) + 10 min")
            chk("KAIAUSDT LITE [ticks] live -3.10 (STOP @ 30 min) → +1 / −2 +1.10 / +1 / −2.5 +1.10 / +1 / −3 +1.10 / +1.25 / −2 -3.10" in out
                and "UNSCORED (no path yet): 10-09 09:00 NOKUSDT LONG" in out and "Operator-closed: 1 — excluded, manually closed" in out and "MANUSDT WIDE (MANUAL_TP)" in out and "WILUSDT" not in out and "OLDUSDT" not in out
                and "1/1 match" in out and "FRENZY_LITE: ⏳ COLLECTING" in out and not os.path.exists(sp), f"end-to-end\n{out}")
    finally:
        WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED = saved
        WT._K1.clear()
    print(f"selftest FRENZY_WILLY_EXIT OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
