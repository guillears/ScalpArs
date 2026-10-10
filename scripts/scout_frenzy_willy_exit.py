#!/usr/bin/env python3
"""🎲➡🔥 Scout — FRENZY_WILLY_EXIT exit shadow (operator hypothesis 2026-10-10, DECISION_LOG 267; OBSERVE only — never changes config).

HYPOTHESIS  FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE should share FRENZY_WILLY-style small fixed targets — a 7-cell grid, operator 2026-10-10:
            TP +frenzy_willy_tp_pct (1.00) or +1.25 net × stop −2 / −2.5 / −3 net, plus +1.25 / −4 (added the same day, the WILLY
            insurance candidate) — instead of today's +frenzy_tp_pct / −frenzy_stop_pct
            (+3 / −3); every cell keeps the live cap (frenzy_max_hold_minutes, 12 h). Levels read from trading_config.json. Judged PER SLEEVE.
COHORT      CLOSED FRENZY_LONG / WIDE / LITE fills in the ~/Downloads orders exports, dedup (opened_at[:19], pair, strategy) — never id; a
            CLOSED row beats an OPEN one — opened ≥ the "(DECISION_LOG 250)" commit (the fixed +3 / −3 exit went live; git unavailable → the
            pinned 2026-10-08 13:37 UTC).
            Manually closed fills (MANUAL_*) are excluded and listed — not the walked exit.
WALKER      scripts/scout_willy_timecap.py's walker and shared caches (aggTrades ticks, else 1m = provisional) from each fill's own entry, both
            exits on the SAME path with the bot's accounting (net = gross − 0.045 − 0.045 × exit/entry). Data: WILLY_TIMECAP's ensure_data
            for a 12 h window (its attempts ledger with a 12 h key, ≤ 2 requests per call — 4 per scout run with WILLY_TIMECAP's own call,
            ≤ 1 archive each — shared cooldown, used-weight stop). Paused (nothing frozen) if the live exit is not a fixed TP. Parity: the LIVE walk must
            reproduce each fill's actual exit (reason + % within 0.05 pts) — printed per sleeve.
VERDICT     (FROZEN, pre-registered — PER SLEEVE, computed ONCE on the first prefix by (open time, pair) of that sleeve's scored fills reaching
            N ≥ 15 on ≥ 8 days; deferred while a fill of that sleeve opened ≤ the prefix end is OPEN, unscored-but-not-final or 1m-provisional
            with its tick archive pending; persisted in reports/SCOUT_FRENZY_WILLY_EXIT.json; never re-fit; one re-read at N ≥ 30).
            Per cell, Δ_i = cell_i − LIVE_i. Cell CANDIDATE (operator decides) iff mean Δ > 0 ∧ day-clustered bootstrap P(mean Δ > 0)
            ≥ 0.95 (4,000, seed 7 — 0.95, not 0.90, because the best of 7 cells is picked: multiple-comparison guard) ∧ no single fill
            > 50 % of Σ Δ ∧ walker parity ≥ 90 %; several qualify → the larger mean Δ (tie → grid order); every cell mean Δ ≤ 0 → KEEP +3 / −3;
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

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, freeze hold, state I/O)
import scout_willy_timecap as WT                    # noqa: E402  (walker, caches, data fetcher, accounting)
import scout_willy_hold as WH                       # noqa: E402  (the FRENZY live exit from trading_config.json)

SLEEVES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
TP125 = 1.25
GRID = (("T1_S2", None, 2.0), ("T1_S25", None, 2.5), ("T1_S3", None, 3.0),       # None = frenzy_willy_tp_pct (1.00) from the config
        ("T125_S2", TP125, 2.0), ("T125_S25", TP125, 2.5), ("T125_S3", TP125, 3.0), ("T125_S4", TP125, 4.0))
VARS = tuple(g[0] for g in GRID)
N_MIN, DAYS_MIN, REREAD_N = 15, 8, 30
P_MIN, SHARE_MAX, PARITY_MIN = 0.95, 0.50, 0.90
BOOT_N, BOOT_SEED = 4000, 7
DEPLOY_GREP = "(DECISION_LOG 250)"
DEPLOY_PIN = "2026-10-08 13:37:25"                   # 6d3974f commit time (UTC) — used only if git is unavailable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_FRENZY_WILLY_EXIT.json")
STUDY_REF = ("yr5 tick study (2026-10-10, the reports/FRENZY_TP2_VS_TP3_TICKS_2026-10-09_fills.csv entries re-walked by this walker — "
             "reproduces their +3 on 100 %; 12 h): every cell is below +3 / −3 in every sleeve except LONG +1/−3 (Δ +0.017, P 0.54). "
             "LONG 564 · +3/−3 +0.084 %/fill · +1: −2 −0.037 / −2.5 −0.011 / −3 +0.101 · +1.25: −2 −0.095 / −2.5 −0.129 / −3 −0.039 · "
             "WIDE 319 · +0.088 · +1: −0.136 / −0.214 / −0.160 · +1.25: −0.042 / −0.156 / −0.062 · LITE 724 · +0.257 · +1: −0.074 / −0.029 / "
             "−0.016 · +1.25: −0.066 / −0.025 / +0.000 (P(better) 0.00 in every other cell); +1.25/−4 (269): LONG +0.020 (Δ −0.063, P 0.35; "
             "H1 +0.155 / H2 −0.143) · WIDE −0.175 (Δ −0.262) · LITE +0.010 (Δ −0.247, P 0.00), 3.2 winners pay a loser. Winners' trough (fills reaching +3, no stop): LONG avg −2.82 / "
             "median −1.94, WIDE −3.14 / −1.78, LITE −2.65 / −1.75")
REVERT = ("Pre-committed revert if a sleeve ever takes the smaller TP: its first 15 fills after the switch — Σ % below what +3 would have "
          "given on the same fills (this walker) → back to +3.")


def deploy_ts():
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + DEPLOY_GREP, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000, unit="ms")
    except Exception:
        return pd.Timestamp(DEPLOY_PIN)


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


def frenzy_levels(th=None):
    """the FRENZY live exit (tp %, stop % positive or None, hold min) — scout_willy_hold.exit_rule on trading_config.json."""
    tp, st, cap = WH.exit_rule("FRENZY_LONG", _ns(th))
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


# ─────────────────────────── scoring ───────────────────────────
def score_fill(r, live, alts, tcache=None):
    """→ (dict or None, source). live = (tp, stop, hold); alts = {name: (tp, stop)} with live's hold. ONE path: ticks, else 1m."""
    tp, st, hold = live
    t, p = WT.load_ticks(r.pair, WT._days(r.te, hold), tcache, WT._need(r.te, hold))
    if t is not None:
        i, j = np.searchsorted(t, r.te, side="right"), np.searchsorted(t, r.te + (max(hold, 120) + 1) * WT.MIN, side="right")
        walk, src = (lambda x, y: WT.walk_ticks(t[i:j], p[i:j], r.te, r.E, x, y, hold)), "ticks"
    else:
        k = WT.load_1m(r.pair)
        need = WT._need_1m(r.te, hold)
        if not (len(k) and np.isin(need, k.open_time.values).all()):
            return None, None
        kk = k[(k.open_time >= need[0]) & (k.open_time <= need[-1])]
        walk, src = (lambda x, y: WT.walk_1m(kk, r.te, r.E, x, y, hold)), "1m"
    a = walk(tp, st)
    la = a["v"]["LIVE"]
    ok, txt = WT.parity(r.why, r.pct, a)
    rec = dict(LIVE=la["pct"], why_LIVE=la["why"], min_LIVE=la["xmin"], par_ok=ok, par_txt=txt)
    for name, (x, y) in alts.items():
        b = walk(x, y)["v"]["LIVE"]
        rec.update({name: b["pct"], f"why_{name}": b["why"], f"min_{name}": b["xmin"]})
    return rec, src


def scored_frame(o, live, alts, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = score_fill(r, live, alts, tcache)
        rows.append(dict(src=src, **(w or {})))
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


# ─────────────────────────── verdict / freeze ───────────────────────────
def judge(sc, v):
    d = sc[v].values.astype(float) - sc.LIVE.values.astype(float)
    s = float(d.sum())
    p = NF.p_mean_neg(-d, sc.day.values, BOOT_N, BOOT_SEED) if len(sc) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    par = float(sc.par_ok.astype(bool).mean()) if len(sc) else 0.0
    m = float(d.mean()) if len(d) else float("nan")
    return dict(mean=m, sum=s, p=p, share=share, par=par,
                q=bool(m > 0 and p is not None and p >= P_MIN and share <= SHARE_MAX and par >= PARITY_MIN))


def verdict(sc):
    n, nd = len(sc), sc.day.nunique() if len(sc) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    J = {v: judge(sc, v) for v in VARS}
    det = f"N {n} · {nd} d · parity {J[VARS[0]]['par'] * 100:.0f} % · " + " · ".join(
        f"{v}: mean Δ {j['mean']:+.3f} pts · Σ Δ {j['sum']:+.2f} · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
        f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ" for v, j in J.items())
    qs = [v for v in VARS if J[v]["q"]]
    if qs:
        best = max(qs, key=lambda v: (round(J[v]["mean"], 12), -VARS.index(v)))
        return f"{best} CANDIDATE (operator decides)", det, J
    if all(J[v]["mean"] <= 0 for v in VARS):
        return "KEEP +3 / −3", det, J
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
                  n_1m=int((pre.src == "1m").sum()), config=cfg,
                  keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


# ─────────────────────────── run ───────────────────────────
HDR = ["| Sleeve | Exit | N | days | WR | avg % | Σ % | Σ$ at the fill's size | median min held |",
       "|---|---|---|---|---|---|---|---|---|"]


def _row(sl, lab, g, v):
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
    _t = _ns(th)
    paused = _num(_t, "frenzy_lock_arm_pct", 0.0) > 0 or _num(_t, "frenzy_tp_pct", 3.0) <= 0
    alts = {n: (float(WT.levels(vars(_t))[0]) if tp_ is None else tp_, sl_) for n, tp_, sl_ in GRID}
    o, opens = load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    opens = [x for x in opens if pd.notna(x[0]) and x[0] >= floor]       # a stale pre-floor OPEN row never holds a freeze
    man = o[o.why.astype(str).str.upper().str.startswith("MANUAL")]       # operator closes: not the walked exit → out, listed
    o = o[~o.why.astype(str).str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    span = live[2]
    tc = {}
    o["src"] = [score_fill(r, live, alts, tc)[1] for r in o.itertuples()]
    notes = WT.ensure_data(o, span, now_ms, fetch=fetch)
    if notes:
        tc.clear()
        WT._K1.clear()
    sc_all = scored_frame(o.drop(columns=["src"]), live, alts, tc)
    done = WT.load_attempts()["done"]
    sc_all["pend"] = [(not isinstance(s, str) and not (WT._kk(r.pair, r.te, WT._span(span)) in done and not WT.ticks_pending(r, span, done, now_ms, tc)))
                      or (s == "1m" and WT.ticks_pending(r, span, done, now_ms, tc)) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = sc_all[sc_all.src.notna()].copy()
    uns = sc_all[sc_all.src.isna()]
    labs = {"LIVE": f"+{live[0]:g} / −{live[1]:g} (live)" if live[1] else f"+{live[0]:g} (live)"}
    labs.update({n: f"+{tp_:g} / −{sl_:g}" for n, (tp_, sl_) in alts.items()})
    L = [f"## 🎲➡🔥 FRENZY_WILLY_TP — should FRENZY / WIDE / LITE take a small fixed target (+{alts['T1_S2'][0]:g} or +{TP125:g}) with a "
         f"−2 / −2.5 / −3 stop, or +{TP125:g} / −4? (DECISION_LOG 267 · 269, OBSERVE only, exit shadow grid)", "",
         f"Cohort: CLOSED FRENZY_LONG / WIDE / LITE fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC (the fixed +3 / −3 exit went live), re-walked from "
         f"each fill's own entry on the same path (WILLY_TIMECAP's walker and cache; 1m = provisional) with the bot's accounting. Every exit "
         f"keeps the live {live[2]}-min cap; TP and stop differ per cell. Judged per sleeve.", ""]
    if paused:
        L.append("⏸ PAUSED: the live FRENZY exit is not a fixed TP (frenzy_lock_arm_pct > 0 or frenzy_tp_pct ≤ 0) — the '+3 / −3' row below is "
                 "NOT what live trades; no Δ is judged and nothing is frozen until the fixed TP is back.")
    L += [*HDR]
    for sl in SLEEVES:
        g = sc[sc.sleeve == sl]
        L += [_row(sl, f"**{labs[v]}**" if v == "LIVE" else labs[v], g, v) for v in ("LIVE",) + VARS]
    L.append("")
    if len(sc):
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.sleeve.replace('FRENZY_', '')} [{r.src}{', provisional' if r.src == '1m' else ''}] live "
            f"{r.LIVE:+.2f} ({r.why_LIVE} @ {r.min_LIVE:.0f} min) → " + " / ".join(f"{labs[v]} {getattr(r, v):+.2f}" for v in VARS)
            for r in sc.itertuples()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (live walk vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(par.sum())}/{len(sc)} match"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.par_txt}" for r in sc[~par].itertuples())))
        if (sc.src == "1m").any():
            L.append(f"⏳ {int((sc.src == '1m').sum())} fill(s) on 1m klines (PROVISIONAL: targets booked exactly; the entry minute's bar is "
                     "skipped) — re-walked on aggTrades once the daily archive is out.")
        L.append("")
    if len(man):
        L.append(f"Excluded: {len(man)} manually closed fill(s) (not the walked exit): "
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
            hold_l = [x for x in opens if x[1].endswith(sl)] + [(r.ts, f"{'1m-provisional' if r.src == '1m' else 'unscored'} {r.pair}")
                                                               for r in sc_all[sc_all.sleeve == sl].itertuples() if r.pend]
            st, ch = freeze(st, sl, g, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                            dict(live=list(live), alts=alts), hold_l, why)
            changed |= ch
        st = {k: v for k, v in st.items() if v}
        if changed:
            NF.du_save_state(st, state_path)
    L.append(f"Bar (pre-registered, PER SLEEVE, frozen once at the first prefix of N ≥ {N_MIN} closed fills on ≥ {DAYS_MIN} days, never re-fit; "
             f"one re-read at N ≥ {REREAD_N}): per cell Δ = cell − live per fill; cell CANDIDATE (operator decides) iff mean Δ > 0 ∧ "
             f"day-clustered P(mean Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > {SHARE_MAX * 100:.0f} % of Σ Δ ∧ walker "
             f"parity ≥ {PARITY_MIN * 100:.0f} % (0.95: the best of {len(VARS)} cells is picked); several → the larger Δ; all ≤ 0 → KEEP +3 / −3; "
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
        chk(verdict(coh([-1.0] * 15))[0] == "KEEP +3 / −3", "all worse → keep")
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
            sp = os.path.join(td, "s.json")
            out = "\n".join(run(pd.Timestamp("2026-10-09 23:00").value // 10**6, fetch=False, state_path=sp, th=th))
            chk("KAIAUSDT LITE [ticks] live -3.10 (STOP @ 30 min) → +1 / −2 +1.10 / +1 / −2.5 +1.10 / +1 / −3 +1.10 / +1.25 / −2 -3.10" in out
                and "UNSCORED (no path yet): 10-09 09:00 NOKUSDT LONG" in out and "Excluded: 1 manually closed" in out and "MANUSDT WIDE (MANUAL_TP)" in out and "WILUSDT" not in out and "OLDUSDT" not in out
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
