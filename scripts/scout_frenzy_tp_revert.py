#!/usr/bin/env python3
"""🎯↩ Scout — FRENZY_TP_REVERT: the revert gate of the FRENZY take-profit change (DECISION_LOG 283; OBSERVE only — never changes config).

CHANGE      2026-10-12 (operator declared override "Ship all as asked"): FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE take profit frenzy_tp_pct
            +3 → +1.5 (stop −3 and the 12 h cap unchanged; WILLY has its own exit). The manual "FRENZY" exit mode reads the same field (it is
            not part of this cohort: MANUAL rows are out).
COHORT      CLOSED non-MANUAL FRENZY_LONG / WIDE / LITE fills from the orders exports (scout_frenzy_willy_exit.load_orders: dedup
            (opened_at[:19], pair, strategy), never id; a CLOSED row beats an OPEN one) opened ≥ the "(DECISION_LOG 283)" commit time + 10 min
            (git %ct; git unavailable / not found → the PLACEHOLDER pin DEPLOY_PIN + 10 min — replace the pin with the real commit time).
WALKER      ONE hybrid path per fill (DECISION_LOG 272: aggTrades ticks per covered UTC day, then the 1m tail = PROVISIONAL; tick ruler = the
            first print through a line; scout_frenzy_willy_exit.load_path / evaluate, the bot's accounting). Per fill: NEW = today's live exit
            (frenzy_tp_pct / frenzy_stop_pct / frenzy_max_hold_minutes from trading_config.json) and OLD = the pre-registered old exit
            +3 / −3 / 720 min (OLD_EXIT). Δ = OLD − NEW (pts). Parity: the NEW walk must reproduce the fill's actual exit (reason + % within
            0.05 pts). Integrity: a fill whose NEW exit is the stop or the cap never printed +1.5 before it → OLD (same stop, same cap, a higher
            TP) must equal NEW there (Δ ≡ 0), else the walker is lying.
BAR         🔒 FROZEN, pre-registered 2026-10-12 (DECISION_LOG 283) — for POOLED (LONG + WIDE + LITE) and per sleeve, ONCE each, on the first
            prefix by (open time, pair) of counted fills reaching N ≥ 20 on ≥ 8 UTC days:
              REVERT the TP to +3  iff  the OLD exit beats the NEW one — mean paired Δ > 0 ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 0.90
                                        (4,000, seed 7) ∧ no single fill > 50 % of Σ Δ ∧ walker parity ≥ 90 % —  OR  the NEW exit's average < 0 while the OLD
                                        averages higher on the same fills (point estimate, no confidence leg — never revert into a worse exit);
              else KEEP +1.5.
            Never frozen on a failed parity (< 90 %) or integrity check; deferred while a fill of that group opened ≤ the prefix end is OPEN,
            unscored-but-not-final, has an undecided exit, or is 1m-provisional with its tick archive pending. State reports/SCOUT_FRENZY_TP_REVERT.json.
Usage: venv/bin/python scripts/scout_frenzy_tp_revert.py [--selftest]
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
import scout_willy_timecap as WT                    # noqa: E402  (caches, data fetcher, parity, accounting)
import scout_frenzy_willy_exit as FX                # noqa: E402  (loader, DECISION_LOG 272 hybrid walker, live FRENZY exit)

SLEEVES = FX.SLEEVES                                # FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE
GROUPS = ("POOLED",) + SLEEVES
OLD_EXIT = (3.0, 3.0, 720)                          # (TP %, stop %, cap min) = the FRENZY exit before DECISION_LOG 283
N_MIN, DAYS_MIN = 20, 8
P_MIN, SHARE_MAX, PARITY_MIN = 0.90, 0.50, 0.90
BOOT_N, BOOT_SEED = 4000, 7
DEPLOY_GREP = "(DECISION_LOG 283)"
DEPLOY_PIN = "2026-10-12 23:00:00"                  # PLACEHOLDER commit time (UTC) — used only if git cannot find the DL 283 commit
LAG_MIN = 10
REVERT, KEEP = "REVERT frenzy_tp_pct → 3 (operator acts)", "KEEP +1.5"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_FRENZY_TP_REVERT.json")


def _git_ct(grep):
    try:
        out = subprocess.run(["git", "-C", _ROOT, "log", "--format=%ct", "--grep=" + grep, "--fixed-strings", "--reverse"],
                             capture_output=True, text=True, timeout=10)
        return pd.Timestamp(int(out.stdout.strip().splitlines()[0]) * 1000, unit="ms")
    except Exception:
        return None


def deploy_ts():
    """the counted floor = the "(DECISION_LOG 283)" commit time + 10 min (deploy lag); not found → DEPLOY_PIN + 10 min."""
    t = _git_ct(DEPLOY_GREP)
    return (pd.Timestamp(DEPLOY_PIN) if t is None else t) + pd.Timedelta(minutes=LAG_MIN)


dl283_floor = deploy_ts


def score_fill(r, live, tcache=None, k1m=None):
    """→ (dict or None, source). ONE hybrid path to max(live cap, OLD cap); NEW = live, OLD = OLD_EXIT. None when NEW is undecided."""
    tp, st, hold = live
    P = FX.load_path(r.pair, int(r.te), float(r.E), max(int(hold), OLD_EXIT[2]), tcache, k1m)
    nw = FX.evaluate(P, tp, st, hold)
    if nw["why"] == "UNRES":
        return None, None
    ok, txt = WT.parity(r.why, r.pct, {"v": {"LIVE": nw}})
    od = FX.evaluate(P, OLD_EXIT[0], OLD_EXIT[1], OLD_EXIT[2])
    same_rule = (st is not None and abs(float(st) - OLD_EXIT[1]) < 1e-12 and int(hold) == OLD_EXIT[2] and float(tp) < OLD_EXIT[0])
    rec = dict(NEW=nw["pct"], why_NEW=nw["why"], min_NEW=nw["xmin"], OLD=od["pct"], why_OLD=od["why"], min_OLD=od["xmin"],
               par_ok=ok, par_txt=txt, n_unres=int(od["why"] == "UNRES"),
               untouched=bool(same_rule and nw["why"] in ("STOP", "CAP")))
    on_ticks = P["n_tick"] > 0 and all(e["why"] != "UNRES" and not e["on_bar"] for e in (nw, od))
    return rec, ("ticks" if on_ticks else FX.path_src(P))


def scored_frame(o, live, tcache=None):
    rows = [dict(src=s, **(w or {})) for w, s in (score_fill(r, live, tcache) for r in o.itertuples())]
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


def counted(g):
    """fills that count: NEW scored AND OLD decided."""
    if not len(g) or "NEW" not in g:
        return g.iloc[:0]
    return g[g.src.notna() & pd.to_numeric(g.OLD, errors="coerce").notna()]


def judge(g):
    d = g.OLD.values.astype(float) - g.NEW.values.astype(float)
    s = float(d.sum()) if len(d) else 0.0
    m = float(d.mean()) if len(d) else float("nan")
    p = NF.p_mean_neg(-d, g.day.values, BOOT_N, BOOT_SEED) if len(d) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    unt = g.untouched.astype(bool).values if len(g) else np.zeros(0, bool)
    integ = bool(np.all(np.abs(d[unt]) < 1e-9))
    par = float(g.par_ok.astype(bool).mean()) if len(g) else 0.0
    new_avg = float(g.NEW.astype(float).mean()) if len(g) else float("nan")
    old_avg = float(g.OLD.astype(float).mean()) if len(g) else float("nan")
    better = bool(len(d) and m > 0 and p is not None and p >= P_MIN and share <= SHARE_MAX and par >= PARITY_MIN)
    return dict(n=int(len(d)), days=int(g.day.nunique()) if len(g) else 0, mean=m, sum=s, p=p, share=share, integ=integ,
                n_untouched=int(unt.sum()), par=par, new_avg=new_avg, old_avg=old_avg, better=better)


def verdict(g):
    """the DECISION_LOG 283 bar on counted fills g → (state, detail, judge dict)."""
    n, nd = len(g), g.day.nunique() if len(g) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    j = judge(g)
    det = (f"N {n} · {nd} d · new avg {j['new_avg']:+.3f} % vs old {j['old_avg']:+.3f} % · mean Δ (old − new) {j['mean']:+.3f} pts · Σ Δ "
           f"{j['sum']:+.2f} · P(Δ>0) {(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
           f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ · untouched Δ = 0 on {j['n_untouched']} fill(s) "
           f"{'✓' if j['integ'] else '✗'} · parity {j['par'] * 100:.0f} %")
    # caveman review: "new avg < 0" alone could revert INTO a worse exit — it also needs the old exit to average better on the same fills
    neg = j["new_avg"] < 0 and j["old_avg"] > j["new_avg"]
    if j["better"] or neg:
        return REVERT, det + (" · old better" if j["better"] else "") + (" · new avg < 0 and old avg higher" if neg else ""), j
    return KEEP, det, j


def prefix(g, n_min):
    z = g.sort_values(["ts", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def freeze(st, group, g, now_iso, cfg, hold=None, why=None):
    """freeze `group` ONCE at its first prefix (N ≥ 20 counted on ≥ 8 days); deferred by hold; never on failed parity / integrity."""
    if group in st:
        return st, False
    pre = prefix(g, N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    w = []
    if NF.freeze_hold(last, hold, w):
        if why is not None:
            why += [f"{group} — {s}" for s in w]
        return st, False
    state, det, j = verdict(pre)
    if j and (j["par"] < PARITY_MIN or not j["integ"]):
        if why is not None:
            why.append(f"{group} — WALKER PARITY / INTEGRITY FAILED — not frozen (parity {j['par'] * 100:.0f} %, untouched Δ = 0 "
                       f"{'✓' if j['integ'] else '✗'})")
        return st, False
    st[group] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                     n_provisional=int((pre.src != "ticks").sum()), config=cfg,
                     keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


def _grp(sc, group):
    return sc if group == "POOLED" else sc[sc.sleeve == group]


def run(now_ms=None, orders=None, fetch=True, state_path=None, open_ts=None, th=None):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    WT._K1.clear()
    live = FX.frenzy_levels(th)
    _t = FX._ns(th)
    paused = FX._num(_t, "frenzy_lock_arm_pct", 0.0) > 0 or FX._num(_t, "frenzy_tp_pct", 0.0) <= 0
    o, opens = FX.load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    opens = [x for x in opens if pd.notna(x[0]) and x[0] >= floor]
    man = o[o.why.astype(str).str.upper().str.startswith("MANUAL")]
    o = o[~o.why.astype(str).str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    span = max(int(live[2]), OLD_EXIT[2])
    tc = {}
    _pre = [score_fill(r, live, tc) for r in o.itertuples()]
    o["src"] = [x[1] for x in _pre]
    o["need_1m"] = [bool(x[0] and x[0].get("n_unres", 0)) for x in _pre]
    notes = WT.ensure_data(o, span, now_ms, fetch=fetch)
    if notes:
        tc.clear()
        WT._K1.clear()
    sc_all = scored_frame(o.drop(columns=["src", "need_1m"]), live, tc)
    done = WT.load_attempts()["done"]
    sc_all["pend"] = [FX._pending(s, getattr(r, "n_unres", 0), WT.ticks_pending(r, span, done, now_ms, tc),
                                  WT._kk(r.pair, r.te, WT._span(span)) not in done) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = counted(sc_all)
    uns = sc_all[sc_all.src.isna()]
    new_lab = f"+{live[0]:g} / −{live[1]:g} / {live[2]} min" if live[1] else f"+{live[0]:g} / no stop / {live[2]} min"
    old_lab = f"+{OLD_EXIT[0]:g} / −{OLD_EXIT[1]:g} / {OLD_EXIT[2]} min"
    L = [f"## 🎯↩ FRENZY_TP_REVERT — does the old FRENZY exit ({old_lab}) beat today's ({new_lab})? (DECISION_LOG 283 revert gate, OBSERVE only)", "",
         f"Cohort: CLOSED non-MANUAL FRENZY_LONG / WIDE / LITE fills opened ≥ {floor:%Y-%m-%d %H:%M:%S} UTC (the \"{DEPLOY_GREP}\" commit + "
         f"{LAG_MIN} min" + ("" if _git_ct(DEPLOY_GREP) is not None else f"; commit not found → PLACEHOLDER pin {DEPLOY_PIN} UTC") +
         "), each re-walked from its own entry on ONE hybrid path (ticks first, then the 1m tail = provisional — DECISION_LOG 272) under "
         "today's exit (NEW, parity vs the actual exit) and the old one (OLD); Δ = OLD − NEW per fill. Untouched = NEW closed at the stop or "
         "the cap (never printed +1.5) → must show Δ = 0.", "",
         "| Group | Exit | N counted | days | WR | avg % | Σ % | Σ$ at the fill's size |", "|---|---|---|---|---|---|---|---|"]
    if paused:
        L.insert(2, "⏸ PAUSED: the live FRENZY exit is not a fixed TP (frenzy_lock_arm_pct > 0 or frenzy_tp_pct ≤ 0) — nothing frozen.")
    for gname in GROUPS:
        g = _grp(sc, gname)
        for lab, v in ((f"**{new_lab} (today)**", "NEW"), (old_lab + " (old)", "OLD")):
            if not len(g):
                L.append(f"| {gname} | {lab} | 0 | – | – | – | – | – |")
                continue
            p = g[v].astype(float)
            L.append(f"| {gname} | {lab} | {len(g)} | {g.day.nunique()} | {(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % | "
                     f"{(p * g.notional.astype(float) / 100).sum():+,.0f} |")
    L.append("")
    if len(sc):
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.sleeve.replace('FRENZY_', '')} [{r.src}{'' if r.src == 'ticks' else ', provisional'}] new "
            f"{r.NEW:+.2f} ({r.why_NEW} @ {r.min_NEW:.0f} min) · old {r.OLD:+.2f} ({r.why_OLD}) · Δ {r.OLD - r.NEW:+.2f}" for r in sc.itertuples()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (today's walk vs the actual exit): {int(par.sum())}/{len(sc)} match"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.par_txt}" for r in sc[~par].itertuples())))
    und = sc_all[sc_all.src.notna() & ~sc_all.index.isin(sc.index)]
    if len(und):
        L.append("Old exit still undecided (not counted yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" for r in und.itertuples()))
    if len(man):
        L.append(f"Excluded: {len(man)} manually closed fill(s): " + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} ({r.why})" for r in man.itertuples()) + ".")
    if len(uns):
        L.append("UNSCORED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair} {str(r.sleeve).replace('FRENZY_', '')}"
                                                       + ("" if r.pend else " (final — no data)") for r in uns.itertuples()))
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok and not paused:
        changed = False
        for gname in GROUPS:
            g = _grp(sc, gname)
            ga = _grp(sc_all, gname)
            hold = [x for x in opens if gname == "POOLED" or x[1].endswith(gname)] + [
                (r.ts, f"{'unscored' if not isinstance(r.src, str) else 'undecided exit' if (getattr(r, 'n_unres', 0) or 0) > 0 else '1m-provisional'} {r.pair}")
                for r in ga.itertuples() if r.pend]
            st, ch = freeze(st, gname, g, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                            dict(new=list(live), old=list(OLD_EXIT)), hold, why)
            changed |= ch
        if changed:
            NF.du_save_state(st, state_path)
    L.append(f"Bar (🔒 pre-registered 2026-10-12, DECISION_LOG 283 — POOLED and per sleeve, each frozen ONCE at its first prefix of N ≥ {N_MIN} "
             f"counted fills on ≥ {DAYS_MIN} UTC days, never re-fit): {REVERT} iff the old exit beats the new — mean Δ (old − new) > 0 ∧ "
             f"day-clustered P(Δ>0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > {SHARE_MAX * 100:.0f} % of Σ Δ ∧ parity ≥ "
             f"{PARITY_MIN * 100:.0f} % — OR the new exit's average < 0 with the old averaging higher on the same fills (point estimate); else {KEEP}. Never frozen on a failed parity / integrity check; "
             "open / unscored / undecided / provisional fills hold the freeze.")
    for gname in GROUPS:
        g = _grp(sc, gname)
        s0 = st.get(gname) if ok else None
        if s0:
            L.append(f"{gname}: **Frozen (prefix to {s0['at']}, N {s0['n']} · {s0['days']} d, {s0.get('n_provisional', 0)} provisional): "
                     f"{s0['state']}** ({s0['detail']})")
            L.append(f"{gname} live (information only): {verdict(g)[0]} — {verdict(g)[1]}")
        else:
            L.append(f"{gname}: ⏳ {verdict(g)[0]} ({verdict(g)[1]}).")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L.append("Δ is per fill on the SAME entries: +1.5 frees FRENZY's slots sooner — the extra trades that lets in are not priced.")
    if notes:
        L.append(f"Data: {' · '.join(notes)}.")
    return L + [""]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    from types import SimpleNamespace
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    chk((OLD_EXIT, N_MIN, DAYS_MIN, P_MIN, SHARE_MAX, PARITY_MIN, BOOT_N, BOOT_SEED, DEPLOY_GREP, LAG_MIN, GROUPS) ==
        ((3.0, 3.0, 720), 20, 8, 0.90, 0.50, 0.90, 4000, 7, "(DECISION_LOG 283)", 10,
         ("POOLED", "FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")), "pre-registered constants pinned")
    chk(deploy_ts() >= pd.Timestamp("2026-10-12") + pd.Timedelta(minutes=LAG_MIN), "floor = the DL 283 commit (or the pin) + 10 min")

    def coh(new, old, n=20, days=8, **kw):
        dd = [f"2026-10-{13 + i % days:02d}" for i in range(n)]
        return pd.DataFrame(dict(pair=[f"P{i}" for i in range(n)], ts=pd.to_datetime(dd) + pd.to_timedelta(np.arange(n), "min"), day=dd,
                                 NEW=new, OLD=old, par_ok=True, untouched=False, src="ticks", notional=1000.0, sleeve="FRENZY_LONG", **kw))
    chk(verdict(coh([0.5] * 20, [0.5] * 20))[0] == KEEP, "tie, new ≥ 0 → keep")
    chk(verdict(coh([0.5] * 20, [1.5] * 20))[0] == REVERT, "old better on every day → revert")
    chk(verdict(coh([-0.1] * 20, [-0.5] * 20))[0] == KEEP, "new average < 0 but old is worse → keep (never revert into a worse exit)")
    chk(verdict(coh([-0.3] * 20, [-0.1] * 20))[0] == REVERT, "new average < 0 and old higher → revert")
    chk(verdict(coh([0.5] * 20, [0.5] * 19 + [10.0]))[0] == KEEP, "one fill carries Σ Δ → not 'old better'")
    chk(verdict(coh([0.5] * 20, [1.5] * 20).assign(par_ok=[False] * 3 + [True] * 17))[0] == KEEP, "parity < 90 % → no 'old better'")
    chk(verdict(coh([0.5] * 19, [1.5] * 19, n=19))[0] == "COLLECTING" and verdict(coh([0.5] * 20, [1.5] * 20, days=7))[0] == "COLLECTING",
        "N < 20 or < 8 days → collecting")
    st, ch = freeze({}, "POOLED", coh([0.5] * 20, [1.5] * 20), "now", {})
    chk(ch and st["POOLED"]["state"] == REVERT and "FRENZY_LONG" not in st, "frozen per group")
    st2, ch2 = freeze(st, "POOLED", coh([0.5] * 25, [0.0] * 25, n=25), "x", {})
    chk(not ch2 and st2["POOLED"]["state"] == REVERT, "decided once, never re-fit")
    w = []
    chk(not freeze({}, "POOLED", coh([0.5] * 20, [1.5] * 20), "x", {}, [(pd.Timestamp("2026-10-13"), "OPEN X FRENZY_LITE")], w)[1] and w,
        "an open fill ≤ the prefix end holds the freeze")
    w = []
    chk(not freeze({}, "POOLED", coh([0.5] * 20, [1.5] * 20, ).assign(untouched=[True] + [False] * 19), "x", {}, None, w)[1]
        and "INTEGRITY" in w[0], "untouched fill with Δ ≠ 0 → never frozen")
    w = []
    chk(not freeze({}, "POOLED", coh([0.5] * 20, [0.5] * 20).assign(par_ok=[False] * 3 + [True] * 17), "x", {}, None, w)[1]
        and "PARITY" in w[0], "parity < 90 % → never frozen (any verdict)")
    # walker: NEW +1.5 / −3 and OLD +3 / −3 on one tick path
    saved = (WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED)
    WT._NET_BLOCKED = True
    try:
        with tempfile.TemporaryDirectory() as td:
            WT._CACHE = os.path.join(td, "cache")
            WT.MY_CACHE = os.path.join(WT._CACHE, "scout_willy_timecap")
            E = 100.0
            px = lambda net: E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)   # noqa: E731
            t0 = pd.Timestamp("2026-10-14 20:00:00").value // 10**6

            def path(pair, nets, why, pct):
                os.makedirs(os.path.join(WT._CACHE, "ticks_q", pair), exist_ok=True)
                np.savez_compressed(os.path.join(WT._CACHE, "ticks_q", pair, "2026-10-14.npz"),
                                    t=(t0 + np.arange(1, len(nets) + 1) * 600_000).astype(np.int64), p=np.array([px(x) for x in nets]))
                return SimpleNamespace(pair=pair, te=t0, E=E, why=why, pct=pct)
            lv = (1.5, 3.0, 720)
            w1, s1 = score_fill(path("AAUSDT", [-1.0, 1.6, 3.1], "FRENZY_TP", 1.6), lv, {})
            chk(s1 == "ticks" and w1["why_NEW"] == "TP" and abs(w1["NEW"] - 1.6) < 1e-6 and w1["why_OLD"] == "TP" and abs(w1["OLD"] - 3.1) < 1e-6
                and w1["par_ok"] and not w1["untouched"], f"TP +1.5 first, old rides to +3 ({w1})")
            w2, _ = score_fill(path("BBUSDT", [-1.0, 1.2, -3.05], "STOP_LOSS", -3.05), lv, {})
            chk(w2["why_NEW"] == "STOP" and w2["why_OLD"] == "STOP" and w2["untouched"] and abs(w2["OLD"] - w2["NEW"]) < 1e-12,
                f"never printed +1.5 → both stop, untouched Δ = 0 ({w2})")
            w3, _ = score_fill(path("CCUSDT", [-1.0, 1.7, -3.2], "FRENZY_TP", 1.7), lv, {})
            chk(w3["why_NEW"] == "TP" and w3["why_OLD"] == "STOP" and abs(w3["OLD"] - w3["NEW"] + 4.9) < 1e-6, f"the new TP saves a fill ({w3})")
            w4, s4 = score_fill(path("DDUSDT", [-1.0, 1.6], "FRENZY_TP", 1.6), lv, {})
            chk(w4["why_OLD"] == "UNRES" and w4["n_unres"] == 1 and np.isnan(w4["OLD"]) and s4 == "ticks+1m", f"old undecided at the path end ({w4}, {s4})")
            o = pd.DataFrame([dict(pair="AAUSDT", sleeve="FRENZY_LONG", ts=pd.Timestamp(t0, unit="ms"), day="2026-10-14", te=t0, E=E,
                                   pct=1.6, notional=1000.0, why="FRENZY_TP")])
            chk(len(counted(scored_frame(o, lv, {}))) == 1, "counted frame")
    finally:
        WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED = saved
        WT._K1.clear()
    print(f"selftest FRENZY_TP_REVERT OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
