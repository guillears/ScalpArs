#!/usr/bin/env python3
"""🎯 Scout — WILLY_TP125 exit shadow (2026-10-09, DECISION_LOG 265; OBSERVE only — never changes config, no bot API).

HYPOTHESIS  FRENZY_WILLY with a fixed TP of +1.25 % net instead of today's +frenzy_willy_tp_pct (+1.00) — same entries, same no-stop, same
            frenzy_willy_max_hold_minutes cap. Source: reports/WILLY_TP075_AND_TURNOVER_2026-10-09.md (yr5 WILLY A 1,067 fills: +1.25
            +0.191 %/fill vs +1.00 +0.135, P(better) 0.915, both halves; the only level above today's) — a side finding, not a ship.
COHORT      = WILLY_TIMECAP's (scripts/scout_willy_timecap.py): CLOSED FRENZY_WILLY LONG fills in the ~/Downloads orders exports, dedup
            (opened_at[:19], pair, direction) — never id, CLOSED beats OPEN — opened ≥ the sleeve's ship ("(DECISION_LOG 251)").
WALKER      WILLY_TIMECAP's walker and caches, run twice per fill from its own entry: LIVE (today's config TP) and TP125 (+1.25), both with
            today's stop (none) and today's cap, on aggTrades (1m klines = provisional: TP booked at exactly the target). CACHE ONLY — this
            line never contacts Binance; WILLY_TIMECAP's fetch pass (run just before it in the scout) fills the shared cache.
            Δ_i = TP125_i − LIVE_i. Fills with no LIVE TP inside the cap close identically under both targets → must show Δ = 0 (sanity).
VERDICT     (FROZEN, pre-registered — computed ONCE on the first prefix by (open time, pair) of the scored fills reaching N ≥ 20 on ≥ 8
            days; deferred while a WILLY fill opened ≤ the prefix end is OPEN, unscored-but-not-final or 1m-provisional with its tick
            archive still pending; persisted in reports/SCOUT_WILLY_TP125.json; never re-fit; one re-read at N ≥ 40 on a later run).
            TP125 CANDIDATE (operator decides) iff mean Δ > 0 ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 0.90 (4,000, seed 7) ∧ no single
            fill > 50 % of Σ Δ ∧ Δ = 0 on every no-LIVE-TP fill (else Δ=0 SANITY FAILED, nothing frozen); mean Δ ≤ 0 → KEEP LIVE TP; else
            KEEP OBSERVING. No freeze unless walker parity holds on the prefix: every fill's LIVE walk matches its actual exit (reason + %
            within 0.05 pts), or ≥ 90 % match AND every fill with a LIVE TP (the fills where the targets differ) matches.
ADOPTED     live target ≥ +1.25 → no new verdict / freeze; the switch time is stamped once in the state and the pre-committed revert
            read runs: first 15 final fills after it, Σ live % < Σ of the +1.00 walk on the same fills → REVERT FLAG (operator decides).
            FRENZY_TP_LATE exits (restart / feed gap, not replicable by the walker) leave the cohort and are listed.
CAVEAT      Δ is per fill on the SAME entries. A higher target keeps the single WILLY slot — and the global hold (no other automated open
            while a WILLY is open) — busy longer; the extra minutes held are printed, the trades they would block are not modelled.
Usage: venv/bin/python scripts/scout_willy_tp125.py [--selftest]
"""
import os
import tempfile
import time

import numpy as np
import pandas as pd

if os.path.dirname(os.path.abspath(__file__)) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_b1h_negflank as NF                     # noqa: E402  (bootstrap, freeze hold, state I/O)
import scout_willy_timecap as WT                    # noqa: E402  (cohort loader, walker, shared caches, attempts ledger)

TP_ALT = 1.25
BASE_TP = 1.00                                       # the +1.00 comparator the revert read needs once +1.25 is live
REVERT_N = 15
N_MIN, DAYS_MIN, REREAD_N = 20, 8, 40
P_MIN, SHARE_MAX = 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_TP125.json")
STUDY_VER = 1                                        # bump when the study re-walk changes (part of the study cache key)
MY_CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_willy_tp125")
REVERT = ("Pre-committed revert if +1.25 is ever adopted: the first 15 WILLY fills after the switch — Σ % below what the +1.00 target "
          "would have given on the same fills (this walker) → back to +1.00.")


# ─────────────────────────── scoring ───────────────────────────
def score_pair(r, hold, tp, stop, tcache=None):
    """→ (dict or None, source). LIVE and TP125 walked on the SAME path (ticks first, else 1m) by WILLY_TIMECAP's walker."""
    a, src = WT.score_fill(r, hold, tp, stop, tcache)
    if a is None:
        return None, None
    b, src2 = WT.score_fill(r, hold, TP_ALT, stop, tcache)
    c, src3 = WT.score_fill(r, hold, BASE_TP, stop, tcache)
    if b is None or c is None or src2 != src or src3 != src:         # cannot happen (same data, same order) — fail loudly, never drop
        raise RuntimeError(f"WILLY_TP125: {r.pair} {r.ts} walked on different paths ({src}/{src2}/{src3})")
    la, lb, lc = a["v"]["LIVE"], b["v"]["LIVE"], c["v"]["LIVE"]
    ok, txt = WT.parity(r.why, r.pct, a)
    return dict(LIVE=la["pct"], why_LIVE=la["why"], min_LIVE=la["xmin"], TP125=lb["pct"], why_TP125=lb["why"], min_TP125=lb["xmin"],
                worst_TP125=lb["worst"], B100=lc["pct"], par_ok=ok, par_txt=txt), src


def scored_frame(o, hold, tp, stop, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = score_pair(r, hold, tp, stop, tcache)
        rows.append(dict(src=src, **(w or {})))
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


# ─────────────────────────── verdict / freeze ───────────────────────────
def judge(sc):
    d = sc.TP125.values - sc.LIVE.values
    notp = sc.why_LIVE.values != "TP"
    s = float(d.sum())
    p = NF.p_mean_neg(-d, sc.day.values, BOOT_N, BOOT_SEED) if len(sc) else None
    share = float(d.max() / s) if s > 0 else float("nan")
    sane = bool(np.all(np.abs(d[notp]) < 1e-9))
    m = float(d.mean()) if len(d) else float("nan")
    q = sane and m > 0 and p is not None and p >= P_MIN and share <= SHARE_MAX
    return dict(mean=m, sum=s, p=p, share=share, sane=sane, n_notp=int(notp.sum()), q=bool(q))


def verdict(sc):
    n, nd = len(sc), sc.day.nunique() if len(sc) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days", {}
    j = judge(sc)
    det = (f"N {n} · {nd} d · mean Δ {j['mean']:+.3f} pts · Σ Δ {j['sum']:+.2f} · P(Δ>0) "
           f"{(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill "
           f"{(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % of ΣΔ · no-LIVE-TP fills ({j['n_notp']}) Δ = 0 "
           f"{'✓' if j['sane'] else '✗'}")
    if not j["sane"]:
        return "Δ=0 SANITY FAILED", det, j
    if j["q"]:
        return "TP125 CANDIDATE (operator decides)", det, j
    if j["mean"] <= 0:
        return "KEEP LIVE TP", det, j
    return "KEEP OBSERVING", det, j


def prefix(sc, n_min):
    z = sc.sort_values(["ts", "pair"], kind="stable")
    seen = set()
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        if k >= n_min and len(seen) >= DAYS_MIN:
            return z.iloc[:k]
    return None


def parity_gate(pre):
    """ok iff every LIVE walk matches its actual exit, or ≥ 90 % match AND every fill with a LIVE TP (where the targets differ) matches."""
    po = pre.par_ok.astype(bool).values
    tp_ = (pre.why_LIVE == "TP").values
    txt = f"walker parity {int(po.sum())}/{len(po)} · {int(po[tp_].sum())}/{int(tp_.sum())} on the LIVE-TP fills"
    return bool(po.all() or (po.mean() >= 0.90 and po[tp_].all())), txt


def freeze(st, sc, now_iso, cfg, hold=None, why=None, dropped=None):
    """'first' (N ≥ 20 on ≥ 8 days) ONCE; 'reread' (N ≥ 40) only on a later call. Never frozen: failed walker parity or Δ=0 SANITY FAILED."""
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    pre = prefix(sc, REREAD_N if key == "reread" else N_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    pok, ptxt = parity_gate(pre)
    if not pok:
        if why is not None:
            why.append(f"WALKER PARITY FAILED — not frozen ({ptxt}); the prefix is fixed, so this stays blocked until the walker is "
                       "fixed — operator review")
        return st, False
    state, det, _ = verdict(pre)
    if state == "Δ=0 SANITY FAILED":
        if why is not None:
            why.append("Δ=0 sanity failed (no-LIVE-TP fills show Δ ≠ 0) — fix the walker, nothing frozen")
        return st, False
    nd = int(sum(1 for t in (dropped or []) if pd.notna(t) and t <= last))
    st[key] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                   parity=ptxt, dropped_no_data=nd, n_1m=int((pre.src == "1m").sum()), config=dict(tp=cfg[0], stop=cfg[1], hold=cfg[2], alt=TP_ALT),
                   keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


# ─────────────────────────── rendering ───────────────────────────
HDR = ["| Exit | N | days | WR | avg % | Σ % | Σ$ at the fill's size | closed at TP | median min held |",
       "|---|---|---|---|---|---|---|---|---|"]


def _row(lab, g, v):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – | – | – |"
    p = g[v]
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % | "
            f"{(p * g.notional / 100).sum():+,.0f} | {(g[f'why_{v}'] == 'TP').mean() * 100:.0f} % | {g[f'min_{v}'].median():.1f} |")


def run(now_ms=None, orders=None, state_path=None, open_ts=None, th=None, study=True):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    WT._K1.clear()
    tp, stop, hold = WT.levels(th)
    o, opens = WT.load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = WT.deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    late = o[o.why.astype(str) == "FRENZY_TP_LATE"]                 # restart / feed-gap exits the walker cannot replicate → out, counted
    o = o[o.why.astype(str) != "FRENZY_TP_LATE"].reset_index(drop=True)
    tc = {}
    sc_all = scored_frame(o, hold, tp, stop, tc)
    done = WT.load_attempts()["done"]
    sc_all["pend"] = [(not isinstance(s, str) and not (WT._kk(r.pair, r.te, WT._span(hold)) in done and not WT.ticks_pending(r, hold, done, now_ms, tc)))
                      or (s == "1m" and WT.ticks_pending(r, hold, done, now_ms, tc)) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = sc_all[sc_all.src.notna()].copy()
    uns = sc_all[sc_all.src.isna()]
    live_lab = f"LIVE (TP +{tp:g} · {'no stop' if not stop else f'stop −{stop:g}'} · {hold} min)"
    L = [f"## 🎯 WILLY_TP125 — FRENZY_WILLY fixed TP +{TP_ALT:g} vs today's +{tp:g} (DECISION_LOG 265, OBSERVE only, exit shadow)", "",
         f"Cohort: CLOSED {WT.STRATEGY} fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC (the sleeve's ship), dedup (opened_at, pair, direction). "
         f"Both targets walked from each fill's own entry on the same path (WILLY_TIMECAP's walker and cache — no Binance call from this "
         f"line; 1m = provisional) with the bot's accounting. Same stop and same {hold}-min cap; only the target differs.", ""]
    if len(late):
        L.append(f"Excluded: {len(late)} FRENZY_TP_LATE fill(s) (restart / feed-gap exit the walker cannot replicate): "
                 + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" for r in late.itertuples()) + ".")
    if tp >= TP_ALT - 1e-9:                                         # +1.25 (or higher) is live → no verdict / freeze; the revert read runs
        return L + adopted_block(sc, sc_all, tp, now_ms, state_path) + [""]
    L += [*HDR, _row(f"**{live_lab}**", sc, "LIVE"), _row(f"**TP +{TP_ALT:g}**", sc, "TP125"), ""]
    if len(sc):
        d = sc.TP125 - sc.LIVE
        hit = sc.why_TP125 == "TP"
        lost = (sc.why_LIVE == "TP") & ~hit
        L.append(f"Reached +{TP_ALT:g} within {hold} min: {int(hit.sum())}/{len(sc)} (Σ Δ {d[hit].sum():+.2f} pts) · took +{tp:g} but not "
                 f"+{TP_ALT:g}: {int(lost.sum())} (closed at the cap / stop instead, Σ Δ {d[lost].sum():+.2f} pts) · no TP either way: "
                 f"{int((sc.why_LIVE != 'TP').sum())} (Δ must be 0).")
        ext = (sc.min_TP125 - sc.min_LIVE)[sc.why_LIVE == "TP"]
        if len(ext):
            L.append(f"Extra minutes the WILLY slot / global hold stays busy, on the fills that took +{tp:g}: median {ext.median():.1f} · "
                     f"Σ {ext.sum():.0f} min — the automated trades those minutes would block are NOT in Δ.")
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} {r.trig} [{r.src}{', provisional' if r.src == '1m' else ''}] LIVE {r.LIVE:+.2f} "
            f"({r.why_LIVE} @ {r.min_LIVE:.0f} min) → TP125 {r.TP125:+.2f} ({r.why_TP125} @ {r.min_TP125:.0f} min) · Δ {r.TP125 - r.LIVE:+.2f}"
            for r in sc.itertuples()))
        dd = sc.assign(d=d).groupby("day").d.agg(["count", "sum"])
        L.append("Day units (Σ Δ per day, pts): " + " · ".join(f"{k} n{int(v['count'])} {v['sum']:+.2f}" for k, v in dd.iterrows()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (LIVE walk vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(par.sum())}/{len(sc)} match"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.par_txt}" for r in sc[~par].itertuples())))
        L.append("")
    if len(uns):
        L.append("UNSCORED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" + ("" if r.pend else " (final — no data)")
                                                       for r in uns.itertuples()))
    hold_list = list(opens) + [(r.ts, f"{'1m-provisional' if r.src == '1m' else 'unscored'} {r.pair}") for r in sc_all.itertuples() if r.pend]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        changed_any = False
        for _ in range(2):                                     # 'first' then (a LATER run only) 'reread'
            had = set(st)
            st, ch = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), (tp, stop, hold), hold_list, why,
                            dropped=list(uns[~uns.pend.astype(bool)].ts))
            changed_any |= ch
            if not ch or "first" not in had:
                break
        if changed_any:
            NF.du_save_state(st, state_path)
    live_state, live_det, _ = verdict(sc)
    bar = (f"Bar (pre-registered, frozen once at the first prefix of N ≥ {N_MIN} closed WILLY fills on ≥ {DAYS_MIN} days, never re-fit; "
           f"one re-read at N ≥ {REREAD_N}): Δ = TP125 − LIVE per fill; TP125 CANDIDATE (operator decides) iff mean Δ > 0 ∧ day-clustered "
           f"P(mean Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > {SHARE_MAX * 100:.0f} % of Σ Δ (no-LIVE-TP fills must "
           f"show Δ = 0); mean Δ ≤ 0 → KEEP +{tp:g}; else keep observing.")
    n_drop = int((~uns.pend.astype(bool)).sum()) if len(uns) else 0
    L.append(f"Live so far: {len(sc)} scored WILLY fills on {sc.day.nunique() if len(sc) else 0} day(s)"
             + (f" ({parity_gate(sc)[1]})" if len(sc) else "") + f" · {n_drop} fill(s) final — no data, excluded"
             + f" — information only, no read below N ≥ {N_MIN} on ≥ {DAYS_MIN} days.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if "first" not in st:
        L.append(f"{bar} Now: ⏳ {live_state} ({live_det}).")
    else:
        for k in ("first", "reread"):
            if k in st:
                f0 = st[k]
                L.append(f"**Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {f0['at']}, frozen on the run of {f0['run_at']}, "
                         f"N {f0['n']} · {f0['days']} d, {f0.get('parity', '')}, {f0.get('dropped_no_data', 0)} final-no-data fill(s) "
                         f"excluded, {f0.get('n_1m', 0)} walked on 1m only (both targets booked exactly), live exit then TP "
                         f"+{f0['config']['tp']:g} / hold {f0['config']['hold']}): {f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    L.append(REVERT)
    if study:
        try:
            L += [""] + study_block()
        except Exception as e:
            L.append(f"Study reference unavailable ({str(e)[:100]}).")
    return L + [""]


def adopted_block(sc, sc_all, tp, now_ms, state_path):
    """the live target is ≥ +1.25: no new verdict / freeze. Revert read (pre-committed): the first 15 scored, final WILLY fills opened
    after the switch was first seen (stamped once in the state) — Σ live % < Σ +1.00 walk on the same fills → REVERT FLAG (operator)."""
    L = [f"ℹ today's config takes +{tp:g} (≥ +{TP_ALT:g}) — no new verdict or freeze; the pre-committed revert read runs instead."]
    st, ok = NF.du_load_state(state_path, now_ms)
    if not ok:
        return L + [f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); revert read skipped."]
    if "switch_seen" not in st:
        st["switch_seen"] = pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
        NF.du_save_state(st, state_path)
    sw = pd.Timestamp(st["switch_seen"])
    for k in ("first", "reread"):
        if k in st:
            L.append(f"Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {st[k]['at']}): {st[k]['state']} ({st[k]['detail']}).")
    after = sc_all[(sc_all.ts >= sw)].sort_values(["ts", "pair"], kind="stable")
    stop_at = after.pend.astype(bool).values.argmax() if after.pend.astype(bool).any() else len(after)
    g = after.iloc[:stop_at]
    g = g[g.src.notna()].iloc[:REVERT_N]
    if len(g) < REVERT_N:
        return L + [f"Revert read (switch first seen {sw:%Y-%m-%d %H:%M} UTC): {len(g)}/{REVERT_N} final fills — collecting."]
    a, b = float(g.LIVE.sum()), float(g.B100.sum())
    return L + [f"Revert read on the first {REVERT_N} fills after {sw:%Y-%m-%d %H:%M} UTC: live Σ {a:+.2f} % vs +{BASE_TP:g} walk Σ {b:+.2f} % → "
                + ("**REVERT FLAG → back to +1.00 (operator decides)**" if a < b else "keep +1.25")]


# ─────────────────────────── study reference (cached ticks only) ───────────────────────────
def _study_rows_path():
    return os.path.join(MY_CACHE, "study_ref_rows.csv")


def _study_key():
    return f"{WT._study_key()}|tp{TP_ALT:g}|v{STUDY_VER}"


def build_study_ref(hold=120):
    """the yr5 WILLY A NOSTOP cohort (WILLY_TIMECAP's study file) re-walked at +1.00 and +1.25 from CACHED ticks only."""
    D = pd.read_csv(WT.STUDY_CSV)
    S = D[(D.variant == "NOSTOP") & (D.st == "ok") & (D.trigger == "A")].reset_index(drop=True)
    cache, rows = {}, []
    for r in S.sort_values(["pair", "te"]).itertuples():
        te = int(r.te)
        t, p = WT.load_ticks(r.pair, WT._days(te, hold), cache)
        if len(cache) > 6:
            cache.clear()
        if t is None:
            rows.append(dict(pair=r.pair, te=te, day=r.day, study_pct=r.pct, ok=False))
            continue
        k = int(np.searchsorted(t, te, side="left"))                # the study's entry print t[k] == te; it walks t[k+1:]
        j = int(np.searchsorted(t, te + hold * WT.MIN, side="right"))
        a = WT.walk_ticks(t[k + 1:j], p[k + 1:j], te, float(r.E), 1.0, None, hold)["v"]["LIVE"]
        b = WT.walk_ticks(t[k + 1:j], p[k + 1:j], te, float(r.E), TP_ALT, None, hold)["v"]["LIVE"]
        rows.append(dict(pair=r.pair, te=te, day=r.day, study_pct=r.pct, ok=True, LIVE=a["pct"], why_LIVE=a["why"], TP125=b["pct"],
                         why_TP125=b["why"]))
    R = pd.DataFrame(rows)
    os.makedirs(MY_CACHE, exist_ok=True)
    R.assign(_key=_study_key()).to_csv(_study_rows_path(), index=False)
    return R


def study_block():
    head = ("**Study reference, not part of the verdict** — FRENZY_WILLY A, yr5 tick study (reports/FRENZY_TP3_VS_TP4_TICKS_2026-10-08_willy.csv, "
            "NOSTOP-sequenced entries at today's +1 / 120-min sequencing; a +1.25 target would hold the slot longer, not modelled")
    if not os.path.exists(WT.STUDY_CSV):
        return [head + "; file absent)."]
    p = _study_rows_path()
    try:
        R = pd.read_csv(p) if os.path.exists(p) else None
    except Exception:
        R = None
    if R is None or not len(R) or str(R["_key"].iloc[0]) != _study_key():
        R = build_study_ref()
    miss = int((~R.ok.astype(bool)).sum())
    R = R[R.ok.astype(bool)].copy()
    d = R.TP125 - R.LIVE
    pr = NF.p_mean_neg(-d.values, R.day.values, BOOT_N, BOOT_SEED)
    par = (np.abs(R.LIVE - R.study_pct) <= 1e-6).mean() * 100
    h = pd.to_datetime(R.day) < pd.Timestamp("2026-05-20")
    return [head + f"). N {len(R)} on {R.day.nunique()} days ({miss} rows without cached ticks); LIVE walk reproduces the study on {par:.1f} %.",
            f"+1.00: WR {(R.LIVE > 0).mean() * 100:.0f} % · {R.LIVE.mean():+.3f} %/fill · +{TP_ALT:g}: WR {(R.TP125 > 0).mean() * 100:.0f} % · "
            f"{R.TP125.mean():+.3f} %/fill · mean Δ {d.mean():+.3f} pts (H1 {d[h].mean():+.3f} / H2 {d[~h].mean():+.3f}), day-clustered "
            f"P(Δ>0) {pr if pr is not None else float('nan'):.2f}; no-LIVE-TP Δ = 0 on "
            f"{int((d[R.why_LIVE != 'TP'].abs() < 1e-9).sum())}/{int((R.why_LIVE != 'TP').sum())}."]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    global STATE, MY_CACHE
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (STATE, MY_CACHE, WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT.STUDY_CSV, WT._NET_BLOCKED)
    WT._NET_BLOCKED = True
    try:
        chk((TP_ALT, N_MIN, DAYS_MIN, REREAD_N, P_MIN, SHARE_MAX, BOOT_N, BOOT_SEED) == (1.25, 20, 8, 40, 0.90, 0.50, 4000, 7),
            "pre-registered constants pinned")
        E, te = 100.0, 1_800_000_000_000
        px = lambda net: E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)   # exit price giving that net %   # noqa: E731
        chk(abs(float(WT.net_pct(px(1.3), E)) - 1.3) < 1e-9, "price helper inverts the bot's accounting")
        # walker: +1.3 at 5 min → both TP; +1.1 then −2 at the cap → LIVE TP, TP125 cap; never +1 → identical cap close
        t = te + np.array([1, 5, 30, 119]) * WT.MIN
        a = WT.walk_ticks(t, [px(0.5), px(1.1), px(1.3), px(-2.0)], te, E, 1.0, None, 120)["v"]["LIVE"]
        b = WT.walk_ticks(t, [px(0.5), px(1.1), px(1.3), px(-2.0)], te, E, TP_ALT, None, 120)["v"]["LIVE"]
        chk(a["why"] == b["why"] == "TP" and abs(a["pct"] - 1.1) < 1e-9 and abs(b["pct"] - 1.3) < 1e-9 and a["xmin"] < b["xmin"],
            "TP +1 at 5 min vs +1.25 at 30 min")
        b2 = WT.walk_ticks(t[:2].tolist() + [t[3]], [px(0.5), px(1.1), px(-2.0)], te, E, TP_ALT, None, 120)["v"]["LIVE"]
        chk(b2["why"] == "CAP" and abs(b2["pct"] + 2.0) < 1e-9, "+1.1 then −2 → TP125 rides to the cap")
        # verdict / freeze on synthetic scored fills
        n = 20

        def fills(dl, why_live=None):
            days = [f"2026-10-{10 + i % 10:02d}" for i in range(n)]
            return pd.DataFrame(dict(pair=[f"P{i}USDT" for i in range(n)], ts=pd.to_datetime(days) + pd.to_timedelta(np.arange(n), "min"),
                                     day=days, LIVE=1.0, TP125=[1.0 + x for x in dl], why_LIVE=why_live or ["TP"] * n, par_ok=True, src="ticks",
                                     notional=1000.0))
        chk(verdict(fills([0.25] * n))[0] == "TP125 CANDIDATE (operator decides)", "all +0.25 → candidate")
        chk(verdict(fills([0.25] * 14 + [-2.0] * 6))[0] == "KEEP LIVE TP", "6 rides to −1 → keep the live TP")
        chk(verdict(fills([0.25] * 18 + [-1.0] * 2))[0] == "KEEP OBSERVING", "small positive, P below the bar → keep observing")
        chk(verdict(fills([0.0] * 19 + [5.0]))[0] == "KEEP OBSERVING", "one fill carrying Σ Δ → not a candidate")
        chk(verdict(fills([0.25] * 19 + [0.1], ["TP"] * 19 + ["CAP"]))[0] == "Δ=0 SANITY FAILED", "a no-LIVE-TP fill with Δ ≠ 0 → sanity")
        chk(verdict(fills([0.25] * n).iloc[:10])[0] == "COLLECTING", "N < 20 → collecting")
        st, ch = freeze({}, fills([0.25] * n), "now", (1.0, None, 120), [], [])
        chk(ch and st["first"]["state"].startswith("TP125 CANDIDATE") and st["first"]["config"]["alt"] == 1.25, "frozen once at N 20")
        chk(freeze(st, fills([-1.0] * n), "later", (1.0, None, 120))[0]["first"]["state"].startswith("TP125"), "first verdict never re-fit")
        why = []
        chk(not freeze({}, fills([0.25] * n).assign(par_ok=np.arange(n) != 3), "x", (1.0, None, 120), [], why)[1]
            and "WALKER PARITY FAILED" in why[0], "a LIVE-TP fill off parity → not frozen")
        chk(not freeze({}, fills([0.25] * n), "x", (1.0, None, 120), [(pd.Timestamp("2026-10-10"), "OPEN X")], [])[1],
            "an OPEN fill inside the prefix defers the freeze")
        # end-to-end run() on a temp tree: cache only, never a fetch
        with tempfile.TemporaryDirectory() as td:
            WT._CACHE = os.path.join(td, "cache")
            WT.MY_CACHE = os.path.join(WT._CACHE, "scout_willy_timecap")
            MY_CACHE = os.path.join(WT._CACHE, "scout_willy_tp125")
            STATE = os.path.join(td, "SCOUT_WILLY_TP125.json")
            WT.STUDY_CSV = os.path.join(td, "absent.csv")
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            WT.EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            base = dict(direction="LONG", entry_strategy=WT.STRATEGY, status="CLOSED", notional_value=10_000.0, entry_frenzy_willy_trigger="A",
                        entry_price=E, pnl=1.0, closed_at="x")
            pd.DataFrame([dict(base, opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", pnl_percentage=1.1, close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-09T12:00:00", pair="NOKUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP")]
                         ).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            tk = pd.Timestamp("2026-10-09 08:45:09").value // 10**6
            os.makedirs(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT"))
            np.savez_compressed(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT", "2026-10-09.npz"),
                                t=np.array([tk, tk + 60_000, tk + 300_000, tk + 1_800_000], dtype=np.int64),
                                p=np.array([E, px(0.5), px(1.1), px(1.3)], dtype=np.float64))
            out = "\n".join(run(pd.Timestamp("2026-10-09 16:00").value // 10**6, state_path=STATE, th={}, study=True))
            chk("KAIAUSDT A [ticks] LIVE +1.10 (TP @ 5 min) → TP125 +1.30 (TP @ 30 min) · Δ +0.20" in out
                and "UNSCORED (no path yet): 10-09 12:00 NOKUSDT" in out and "1/1 match" in out and "COLLECTING" in out
                and "Extra minutes" in out and "file absent" in out and not os.path.exists(STATE), f"end-to-end run()\n{out}")
            chk("TP125 rides" not in out and "| **TP +1.25** | 1 |" in out, "summary row")
            out = "\n".join(run(pd.Timestamp("2026-10-09 16:00").value // 10**6, state_path=STATE, th={"frenzy_willy_tp_pct": 1.25},
                                study=False))
            chk("revert read runs instead" in out and "0/15 final fills" in out and "first" not in NF.du_load_state(STATE)[0]
                and NF.du_load_state(STATE)[0]["switch_seen"] == "2026-10-09T16:00:00", f"live at +1.25 → revert read, switch stamped\n{out}")
            sp = os.path.join(td, "adopt.json")
            NF.du_save_state({"switch_seen": "2026-10-10T00:00:00"}, sp)
            sa = pd.DataFrame(dict(ts=pd.Timestamp("2026-10-10") + pd.to_timedelta(np.arange(16), "h"), pair="X", src="ticks", pend=False,
                                   LIVE=[1.25] * 10 + [-1.5] * 6, B100=1.0))
            chk("REVERT FLAG" in adopted_block(sa, sa, 1.25, 0, sp)[-1], "15 fills, live Σ below the +1.00 walk → revert flag")
            chk("keep +1.25" in adopted_block(sa.assign(LIVE=1.25), sa.assign(LIVE=1.25), 1.25, 0, sp)[-1], "live Σ above → keep")
            chk("collecting" in adopted_block(sa, sa.assign(pend=[False] * 3 + [True] * 13), 1.25, 0, sp)[-1],
                "a pending fill inside the window holds the revert read")
            # FRENZY_TP_LATE fills leave the cohort, counted
            pd.DataFrame([dict(base, opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", pnl_percentage=1.1, close_reason="FRENZY_TP_LATE")]
                         ).to_csv(os.path.join(dl, "scalpars_orders_paper_b.csv"), index=False)
            os.utime(os.path.join(dl, "scalpars_orders_paper_a.csv"), (1, 1))
            out = "\n".join(run(pd.Timestamp("2026-10-09 16:00").value // 10**6, state_path=os.path.join(td, "late.json"), th={}, study=False))
            chk("Excluded: 1 FRENZY_TP_LATE" in out and "KAIAUSDT A [ticks]" not in out, f"TP_LATE excluded and counted\n{out}")
    finally:
        STATE, MY_CACHE, WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT.STUDY_CSV, WT._NET_BLOCKED = saved
        WT._K1.clear()
    print(f"selftest WILLY_TP125 OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
