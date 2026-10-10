#!/usr/bin/env python3
"""🛑 Scout — WILLY_STOP exit shadow (operator 2026-10-10, DECISION_LOG 268; OBSERVE only — never changes config, no bot API).

HYPOTHESIS  FRENZY_WILLY should carry a −4 % net stop (SL4) instead of no stop — same TP (frenzy_willy_tp_pct), same cap
            (frenzy_willy_max_hold_minutes), read from trading_config.json. Operator 2026-10-10 chose −4 after reading −2 / −3 / −5: the
            stop is INSURANCE (caps one WILLY loss at ≈ −4 % ≈ a fifth of the book at 20×; no-stop worst yr5 −25.8 %) at a known cost.
            SL2 / SL3 / SL5 / SL8 are shown for context only (not part of the verdict).
COHORT      = WILLY_TIMECAP's: CLOSED FRENZY_WILLY LONG fills from the sleeve's ship, dedup (opened_at[:19], pair, direction) — never id;
            manually closed fills (MANUAL_*) are excluded and listed.
WALKER      WILLY_TIMECAP's walker and shared caches (aggTrades ticks, else 1m = provisional), every variant on the SAME path from each fill's
            own entry with the bot's accounting; a stop = the first print with net ≤ −stop (that print's net is booked). CACHE ONLY — no
            Binance call from this line (WILLY_TIMECAP's fetch pass, run just before it, fills the shared cache).
            NOSTOP = today's exit with no stop; LIVE = today's config as it stands (≡ NOSTOP while frenzy_willy_stop_pct is 0) — parity.
            Δ_i = SL4_i − NOSTOP_i. Sanity: a fill whose lowest point before its NOSTOP close stayed above −4 must show Δ = 0.
VERDICT     (FROZEN, pre-registered — computed ONCE on the first prefix by (open time, pair) reaching N ≥ 20 on ≥ 8 days AND ≥ 8 fills that
            dipped to −4 before their no-stop close (the only fills the stop changes — a −4 dip is rare, yr5 175 / 1,067 = 16 %, so the
            first verdict is expected around ~50 fills and the re-read around ~100); deferred while a WILLY
            fill opened ≤ the prefix end is OPEN, unscored-but-not-final or 1m-provisional with its tick archive pending; persisted in
            reports/SCOUT_WILLY_STOP.json; never re-fit; one re-read at N ≥ 40 with ≥ 16 dipped). SL4 CANDIDATE (operator decides) iff mean Δ > 0 ∧
            day-clustered bootstrap P(mean Δ > 0) ≥ 0.90 (4,000, seed 7) ∧ no single fill > 50 % of Σ Δ ∧ the Δ = 0 sanity holds ∧ walker
            parity (every LIVE walk matches its actual exit, or ≥ 90 % do and every fill that dipped below −4 does; a live STOP_LOSS matches
            within 0.5 pts — the poll price on a dump); mean Δ ≤ 0 → KEEP NO
            STOP; else KEEP OBSERVING.
CAVEAT      Tick prints can stop on wicks the live poller (~2 s) rides through, so a tick-walked stop fires a little more often than live would —
            read the stop rows as slightly pessimistic. On 1m (provisional) a stop books −4, or the bar's open when it opens through the
            stop (open known for pages fetched from Oct-10; older caches book −4), and a bar that both reaches the TP and dips to −4 counts as dipped and stops (conservative). A −4 dip is rare (yr5: 175 of 1,067 fills), so live evidence accrues slowly;
            the line also prints, every run, the worst no-stop loss so far (what the stop would have capped). Δ is per fill on the SAME entries (a stopped WILLY frees the slot / global hold sooner — not priced).
Usage: venv/bin/python scripts/scout_willy_stop.py [--selftest]
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

SL_MAIN = 4.0
SL_CONTEXT = (2.0, 3.0, 5.0, 8.0)
N_MIN, DAYS_MIN, REREAD_N = 20, 8, 40
HIT_MIN, HIT_REREAD = 8, 16                          # fills that dipped to the stop — the only ones where the stop changes anything
P_MIN, SHARE_MAX, PARITY_MIN = 0.90, 0.50, 0.90
BOOT_N, BOOT_SEED = 4000, 7
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_STOP.json")
STUDY_REF = ("yr5 tick study (2026-10-10, the WILLY_TIMECAP study cohort: 1,067 WILLY A fills, same walker): no stop 85 % WR · +0.135 %/fill "
             "(H1 +0.121 / H2 +0.151), worst fill −25.8 %, 26 fills ≤ −8 % · SL4 80 % · +0.085 (H1 +0.066 / H2 +0.106; Δ −0.050, P(better) "
             "0.18) — cuts 55 of 902 TP closes (−276 pts), catches 120 of 165 cap closes (+225 pts) · SL2 65 % · −0.027 (Δ −0.162, P 0.01) · SL3 75 % · "
             "+0.030 (Δ −0.105) · SL5 81 % · +0.062 (Δ −0.073) · SL6 +0.073 · SL8 84 % · +0.113 (Δ −0.022); the −4 … −8 costs are within "
             "noise (non-monotonic). Troughs: all −2.25 avg / −1.19 median, winners −1.42 / −0.84, cap closes −6.79 / −5.77")
REVERT = ("Pre-committed revert if the −4 stop is ever adopted: the first 15 WILLY fills after the switch — Σ % below what NO stop would have "
          "given on the same fills (this walker) → stop back to 0.")


def _name(s):
    return f"SL{s:g}"


def score_fill(r, hold, tp, stop, tcache=None):
    """→ (dict or None, source). NOSTOP / LIVE (config stop) / SL4 + the context stops on ONE path (ticks first, else 1m)."""
    t, p = WT.load_ticks(r.pair, WT._days(r.te, hold), tcache, WT._need(r.te, hold))
    if t is not None:
        i, j = np.searchsorted(t, r.te, side="right"), np.searchsorted(t, r.te + (WT._span(hold) + 1) * WT.MIN, side="right")
        walk, src = (lambda s: WT.walk_ticks(t[i:j], p[i:j], r.te, r.E, tp, s, hold)), "ticks"
    else:
        k = WT.load_1m(r.pair)
        need = WT._need_1m(r.te, hold)
        if not (len(k) and np.isin(need, k.open_time.values).all()):
            return None, None
        kk = k[(k.open_time >= need[0]) & (k.open_time <= need[-1])]
        walk, src = (lambda s: WT.walk_1m(kk, r.te, r.E, tp, s, hold)), "1m"
    live = walk(stop)
    ok, txt = WT.parity(r.why, r.pct, live)
    rec = dict(par_ok=ok, par_txt=txt)
    for name, s in [("LIVE", stop), ("NOSTOP", None)] + [(_name(x), x) for x in (SL_MAIN,) + SL_CONTEXT]:
        v = (live if name == "LIVE" else walk(s))["v"]["LIVE"]
        rec.update({name: v["pct"], f"why_{name}": v["why"], f"min_{name}": v["xmin"], f"worst_{name}": v["worst"]})
    return rec, src


def scored_frame(o, hold, tp, stop, tcache=None):
    rows = []
    for r in o.itertuples():
        w, src = score_fill(r, hold, tp, stop, tcache)
        rows.append(dict(src=src, **(w or {})))
    if not rows:
        return o.reset_index(drop=True).assign(src=pd.Series(dtype=object))
    return pd.concat([o.reset_index(drop=True), pd.DataFrame(rows, index=range(len(o)))], axis=1)


# ─────────────────────────── verdict / freeze ───────────────────────────
def judge(sc, v=_name(SL_MAIN), s=SL_MAIN):
    d = sc[v].values.astype(float) - sc.NOSTOP.values.astype(float)
    above = sc.worst_NOSTOP.values.astype(float) > -s                 # never dipped to the stop before the no-stop close → Δ must be 0
    tot = float(d.sum())
    p = NF.p_mean_neg(-d, sc.day.values, BOOT_N, BOOT_SEED) if len(sc) else None
    share = float(d.max() / tot) if tot > 0 else float("nan")
    sane = bool(np.all(np.abs(d[above]) < 1e-9))
    m = float(d.mean()) if len(d) else float("nan")
    return dict(mean=m, sum=tot, p=p, share=share, sane=sane, n_hit=int((~above).sum()),
                q=bool(sane and m > 0 and p is not None and p >= P_MIN and share <= SHARE_MAX))


def _hits(sc):
    return int((sc.worst_NOSTOP.astype(float) <= -SL_MAIN).sum()) if len(sc) else 0


def verdict(sc, hit_min=HIT_MIN):
    n, nd, nh = len(sc), sc.day.nunique() if len(sc) else 0, _hits(sc)
    if n < N_MIN or nd < DAYS_MIN or nh < hit_min:
        return "COLLECTING", f"N {n}/{N_MIN} · {nd}/{DAYS_MIN} days · dipped to −{SL_MAIN:g} {nh}/{hit_min}", {}
    j = judge(sc)
    det = (f"N {n} · {nd} d · {j['n_hit']} fill(s) dipped to −{SL_MAIN:g} · mean Δ {j['mean']:+.3f} pts · Σ Δ {j['sum']:+.2f} · P(Δ>0) "
           f"{(j['p'] if j['p'] is not None else float('nan')):.2f} · top fill {(j['share'] * 100 if j['share'] == j['share'] else float('nan')):.0f} % "
           f"of ΣΔ · untouched fills Δ = 0 {'✓' if j['sane'] else '✗'}")
    if not j["sane"]:
        return "Δ=0 SANITY FAILED", det, j
    if j["q"]:
        return f"{_name(SL_MAIN)} CANDIDATE (operator decides)", det, j
    if j["mean"] <= 0:
        return "KEEP NO STOP", det, j
    return "KEEP OBSERVING", det, j


def prefix(sc, n_min, hit_min=HIT_MIN):
    z = sc.sort_values(["ts", "pair"], kind="stable")
    hit = (z.worst_NOSTOP.astype(float) <= -SL_MAIN).values if len(z) else np.array([], dtype=bool)
    seen, nh = set(), 0
    for k, d in enumerate(z.day.values, 1):
        seen.add(d)
        nh += bool(hit[k - 1])
        if k >= n_min and len(seen) >= DAYS_MIN and nh >= hit_min:
            return z.iloc[:k]
    return None


STOP_TOL = 0.50                                      # a live STOP books the ~2 s poll price on a dump — reason must match, % within 0.5


def parity_gate(pre):
    if {"why_LIVE", "why", "LIVE", "pct"} <= set(pre.columns):
        stop_ok = ((pre.why_LIVE == "STOP") & (pre.why.astype(str) == "STOP_LOSS")
                   & ((pre.LIVE.astype(float) - pre.pct.astype(float)).abs() <= STOP_TOL)).values
    else:
        stop_ok = np.zeros(len(pre), dtype=bool)
    po = pre.par_ok.astype(bool).values | stop_ok
    hit = (pre.worst_NOSTOP.astype(float) <= -SL_MAIN).values
    txt = f"walker parity {int(po.sum())}/{len(po)} · {int(po[hit].sum())}/{int(hit.sum())} on the fills that dipped to −{SL_MAIN:g}"
    return bool(po.all() or (po.mean() >= PARITY_MIN and po[hit].all())), txt


def freeze(st, sc, now_iso, cfg, hold=None, why=None, dropped=None):
    key = "reread" if "first" in st else "first"
    if key in st:
        return st, False
    rr = key == "reread"
    pre = prefix(sc, REREAD_N if rr else N_MIN, HIT_REREAD if rr else HIT_MIN)
    if pre is None:
        return st, False
    last = pre.ts.max()
    if NF.freeze_hold(last, hold, why):
        return st, False
    pok, ptxt = parity_gate(pre)
    if not pok:
        if why is not None:
            why.append(f"WALKER PARITY FAILED — not frozen ({ptxt}); operator review")
        return st, False
    state, det, _ = verdict(pre, HIT_REREAD if rr else HIT_MIN)
    if state == "Δ=0 SANITY FAILED":
        if why is not None:
            why.append("Δ=0 sanity failed — fix the walker, nothing frozen")
        return st, False
    st[key] = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=now_iso, n=len(pre), days=int(pre.day.nunique()),
                   parity=ptxt, n_1m=int((pre.src == "1m").sum()), config=dict(tp=cfg[0], stop=cfg[1], hold=cfg[2]),
                   dropped_no_data=int(sum(1 for t in (dropped or []) if pd.notna(t) and t <= last)),
                   keys=[f"{a}|{b}" for a, b in zip(pre.ts.dt.strftime("%Y-%m-%dT%H:%M:%S"), pre.pair.astype(str))])
    return st, True


# ─────────────────────────── run ───────────────────────────
HDR = ["| Exit | N | days | WR | avg % | Σ % | Σ$ at the fill's size | stopped | median min held |", "|---|---|---|---|---|---|---|---|---|"]


def _row(lab, g, v):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – | – | – |"
    p = g[v].astype(float)
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(p > 0).mean() * 100:.0f} % | {p.mean():+.3f} % | {p.sum():+.2f} % | "
            f"{(p * g.notional.astype(float) / 100).sum():+,.0f} | {int((g[f'why_{v}'] == 'STOP').sum())} | {g[f'min_{v}'].median():.0f} |")


def run(now_ms=None, orders=None, state_path=None, open_ts=None, th=None):
    now_ms = now_ms or int(time.time() * 1000)
    state_path = state_path or STATE
    WT._K1.clear()
    tp, stop, hold = WT.levels(th)
    o, opens = WT.load_orders(orders)
    if open_ts is not None:
        opens = list(open_ts)
    floor = WT.deploy_ts()
    o = o[o.ts >= floor].reset_index(drop=True)
    man = o[o.why.astype(str).str.upper().str.startswith("MANUAL")]
    o = o[~o.why.astype(str).str.upper().str.startswith("MANUAL")].reset_index(drop=True)
    tc = {}
    sc_all = scored_frame(o, hold, tp, stop, tc)
    done = WT.load_attempts()["done"]
    sc_all["pend"] = [(not isinstance(s, str) and not (WT._kk(r.pair, r.te, WT._span(hold)) in done and not WT.ticks_pending(r, hold, done, now_ms, tc)))
                      or (s == "1m" and WT.ticks_pending(r, hold, done, now_ms, tc)) for s, r in zip(sc_all.src, sc_all.itertuples())]
    sc = sc_all[sc_all.src.notna()].copy()
    uns = sc_all[sc_all.src.isna()]
    L = [f"## 🛑 WILLY_STOP — FRENZY_WILLY with a −{SL_MAIN:g} % stop vs today's no stop (operator 2026-10-10, DECISION_LOG 268, OBSERVE only, "
         f"exit shadow)", "",
         f"Cohort: CLOSED {WT.STRATEGY} fills opened ≥ {floor:%Y-%m-%d %H:%M} UTC, every stop walked on the same path from each fill's own entry "
         f"(WILLY_TIMECAP's walker and cache — no Binance call from this line; 1m = provisional). TP +{tp:g} and the {hold}-min cap are today's "
         f"for every variant; only the stop differs.", ""]
    if stop:
        L.append(f"ℹ today's config already carries a −{stop:g} stop — LIVE ≠ NOSTOP; the verdict below still compares −{SL_MAIN:g} with NO stop.")
    L += [*HDR, _row("**NO stop (today)**" if not stop else "NO stop", sc, "NOSTOP"), _row(f"**−{SL_MAIN:g} stop**", sc, _name(SL_MAIN))]
    L += [_row(f"−{s:g} stop (context)", sc, _name(s)) for s in SL_CONTEXT] + [""]
    if len(sc):
        d = sc[_name(SL_MAIN)] - sc.NOSTOP
        hit = sc.worst_NOSTOP.astype(float) <= -SL_MAIN
        sane = bool((d[~hit].abs() < 1e-9).all())
        L.append(f"Dipped to −{SL_MAIN:g} before the no-stop close: {int(hit.sum())}/{len(sc)} — of them {int((hit & (sc.why_NOSTOP == 'TP')).sum())} "
                 f"still took the TP without a stop (the stop would cut them) · Σ Δ on those {d[hit].sum():+.2f} pts · untouched fills Δ = 0 "
                 f"{'✓' if sane else '✗ (walker check failed)'}.")
        L.append("Per fill: " + " · ".join(
            f"{r.ts:%m-%d %H:%M} {r.pair} [{r.src}{', provisional' if r.src == '1m' else ''}] no stop {r.NOSTOP:+.2f} ({r.why_NOSTOP} @ "
            f"{r.min_NOSTOP:.0f} min, low {r.worst_NOSTOP:+.2f}) → −{SL_MAIN:g}: {getattr(r, _name(SL_MAIN)):+.2f} "
            f"({getattr(r, 'why_' + _name(SL_MAIN))})" for r in sc.itertuples()))
        tail = sc.NOSTOP.astype(float)
        L.append(f"Insurance read (the reason the operator chose −{SL_MAIN:g}): worst no-stop result so far {tail.min():+.2f} % "
                 f"({sc.loc[tail.idxmin(), 'pair']}) · lowest point reached {sc.worst_NOSTOP.min():+.2f} % · no-stop losses beyond −{SL_MAIN:g}: "
                 f"{int((tail < -SL_MAIN).sum())} fill(s), Σ {float((tail[tail < -SL_MAIN] + SL_MAIN).sum()):+.2f} pts the stop would have cut off.")
        dd = sc.assign(d=d).groupby("day").d.agg(["count", "sum"])
        L.append("Day units (Σ Δ per day, pts): " + " · ".join(f"{k} n{int(v['count'])} {v['sum']:+.2f}" for k, v in dd.iterrows()))
        par = sc.par_ok.astype(bool)
        L.append(f"Walker parity (LIVE walk vs the actual exit, reason + % within {WT.PARITY_TOL} pts): {int(par.sum())}/{len(sc)} match"
                 + ("" if par.all() else " — off: " + " · ".join(f"{r.pair} {r.par_txt}" for r in sc[~par].itertuples())))
        L.append("")
    if len(man):
        L.append(f"Excluded: {len(man)} manually closed WILLY fill(s): " + ", ".join(f"{r.ts:%m-%d %H:%M} {r.pair} ({r.why})" for r in man.itertuples()))
    if len(uns):
        L.append("UNSCORED (no path yet): " + " · ".join(f"{r.ts:%m-%d %H:%M} {r.pair}" + ("" if r.pend else " (final — no data)")
                                                       for r in uns.itertuples()))
    hold_list = list(opens) + [(r.ts, f"{'1m-provisional' if r.src == '1m' else 'unscored'} {r.pair}") for r in sc_all.itertuples() if r.pend]
    why = []
    st, ok = NF.du_load_state(state_path, now_ms)
    if ok:
        st, ch = freeze(st, sc, pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), (tp, stop, hold), hold_list, why,
                        dropped=list(uns[~uns.pend.astype(bool)].ts) if len(uns) else [])
        if ch:
            NF.du_save_state(st, state_path)
    live_state, live_det, _ = verdict(sc)
    L.append(f"Bar (pre-registered, frozen once at the first prefix of N ≥ {N_MIN} closed WILLY fills on ≥ {DAYS_MIN} days, never re-fit; one "
             f"re-read at N ≥ {REREAD_N} on a later run; both also need ≥ {HIT_MIN} / {HIT_REREAD} fills that dipped to −{SL_MAIN:g}): Δ = −{SL_MAIN:g} stop − no stop per fill; {_name(SL_MAIN)} CANDIDATE (operator "
             f"decides) iff mean Δ > 0 ∧ day-clustered P(mean Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no fill > "
             f"{SHARE_MAX * 100:.0f} % of Σ Δ ∧ untouched fills Δ = 0 ∧ walker parity; mean Δ ≤ 0 → KEEP NO STOP; else keep observing.")
    if why:
        L.append("⏸ freezing deferred this run: " + " · ".join(why) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    if "first" not in st:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    else:
        for k in ("first", "reread"):
            if k in st:
                f0 = st[k]
                L.append(f"**Frozen {'verdict' if k == 'first' else 're-read'} (prefix to {f0['at']}, N {f0['n']} · {f0['days']} d, "
                         f"{f0.get('parity', '')}, {f0.get('n_1m', 0)} on 1m only): {f0['state']}** ({f0['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    L += ["Tick prints can stop on wicks the live poller rides through — the stop rows are slightly pessimistic.", REVERT,
          f"Study reference, not part of the verdict — {STUDY_REF}."]
    return L + [""]


# ─────────────────────────── self-test (hermetic) ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    saved = (WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED, WT.deploy_ts)
    WT._NET_BLOCKED = True
    WT.deploy_ts = lambda: pd.Timestamp(WT.DEPLOY_PIN)               # no git read inside the selftest
    try:
        chk((SL_MAIN, SL_CONTEXT, N_MIN, DAYS_MIN, REREAD_N, P_MIN, SHARE_MAX, PARITY_MIN, BOOT_N, BOOT_SEED) ==
            (4.0, (2.0, 3.0, 5.0, 8.0), 20, 8, 40, 0.90, 0.50, 0.90, 4000, 7), "pre-registered constants pinned")
        chk((HIT_MIN, HIT_REREAD, STOP_TOL) == (8, 16, 0.50), "dipped-fill floors + live-stop parity tolerance pinned")
        pp = pd.DataFrame(dict(par_ok=[False, True], why_LIVE=["STOP", "TP"], why=["STOP_LOSS", "FRENZY_TP"], LIVE=[-4.0, 1.0], pct=[-4.4, 1.0],
                               worst_NOSTOP=[-6.0, -1.0]))
        chk(parity_gate(pp)[0] and not parity_gate(pp.assign(pct=[-4.7, 1.0]))[0], "a live stop 0.4 below the walk matches; 0.7 does not")

        E, te = 100.0, 1_800_000_000_000
        px = lambda net: E * (1 + (net + WT.TAKER) / 100) / (1 - WT.TAKER / 100)   # noqa: E731
        t = te + np.array([1, 5, 30]) * WT.MIN
        a = WT.walk_ticks(t, [px(-4.5), px(-1.0), px(1.2)], te, E, 1.0, None, 120)["v"]["LIVE"]
        b = WT.walk_ticks(t, [px(-4.5), px(-1.0), px(1.2)], te, E, 1.0, 4.0, 120)["v"]["LIVE"]
        chk(a["why"] == "TP" and abs(a["pct"] - 1.2) < 1e-9 and b["why"] == "STOP" and abs(b["pct"] + 4.5) < 1e-9,
            "a winner that dipped −4.5 first: TP without a stop, −4.5 (the print) with −4")
        kb = pd.DataFrame(dict(open_time=te + np.arange(1, 4) * WT.MIN, o=[px(-0.1), px(-1.0), px(-1.2)],
                               h=[px(0.2), px(-0.5), px(2.0)], l=[px(-0.5), px(-4.5), px(-1.5)], c=[px(-1.0), px(-1.2), px(1.5)]))
        chk(abs(WT.walk_1m(kb, te, E, 1.0, 4.0, 120)["v"]["LIVE"]["pct"] + 4.0) < 1e-9, "1m: a bar opening above −4 books the stop at −4")
        kg = kb.assign(o=[px(-0.1), px(-8.8), px(-1.2)], l=[px(-0.5), px(-9.0), px(-1.5)], h=[px(0.2), px(-8.0), px(2.0)])
        chk(abs(WT.walk_1m(kg, te, E, 1.0, 4.0, 120)["v"]["LIVE"]["pct"] + 8.8) < 1e-6, "1m: a bar opening through −4 books its open (the gap)")
        chk(abs(WT.walk_1m(kg.drop(columns=["o"]), te, E, 1.0, 4.0, 120)["v"]["LIVE"]["pct"] + 4.0) < 1e-9, "1m: no open column → −4 (old caches)")

        def coh(d, n=20, worst=-5.0):
            days = [f"2026-10-{10 + i % 8:02d}" for i in range(n)]
            return pd.DataFrame(dict(pair=[f"P{i}" for i in range(n)], ts=pd.to_datetime(days) + pd.to_timedelta(np.arange(n), "min"), day=days,
                                     NOSTOP=0.0, SL4=d, worst_NOSTOP=worst, par_ok=True, src="ticks", notional=1000.0))
        chk(verdict(coh([0.5] * 20))[0] == "SL4 CANDIDATE (operator decides)", "stop better everywhere → candidate")
        chk(verdict(coh([-3.0] * 20))[0] == "KEEP NO STOP", "stop cuts winners → keep no stop")
        chk(verdict(coh([0.0] * 19 + [5.0]))[0] == "KEEP OBSERVING", "one fill carries Σ → observe")
        chk(verdict(coh([0.5] * 20).assign(worst_NOSTOP=[-5.0] * 10 + [-3.0] * 10))[0] == "Δ=0 SANITY FAILED",
            "untouched fills (low −3 > −4) with Δ ≠ 0 → sanity")
        chk(verdict(coh([0.5] * 20).iloc[:5])[0] == "COLLECTING", "N < 20 → collecting")
        few = coh([0.0] * 20, worst=-1.0).assign(worst_NOSTOP=[-5.0] * 3 + [-1.0] * 17, SL4=[0.5] * 3 + [0.0] * 17)
        chk(verdict(few)[0] == "COLLECTING" and "dipped to −4 3/8" in verdict(few)[1] and not freeze({}, few, "x", (1.0, None, 120))[1],
            "only 3 fills dipped to −4 → no verdict, nothing frozen (a −4 stop changes nothing on the rest)")
        st, ch = freeze({}, coh([0.5] * 20), "now", (1.0, None, 120), [], [])
        chk(ch and st["first"]["state"].startswith("SL4") and not freeze(st, coh([-3.0] * 20), "x", (1.0, None, 120))[1], "frozen once")
        why = []
        chk(not freeze({}, coh([0.5] * 20).assign(par_ok=np.arange(20) != 3), "x", (1.0, None, 120), [], why)[1]
            and "PARITY" in why[0], "a dipped fill off parity → not frozen")
        with tempfile.TemporaryDirectory() as td:
            WT._CACHE = os.path.join(td, "cache")
            WT.MY_CACHE = os.path.join(WT._CACHE, "scout_willy_timecap")
            dl = os.path.join(td, "dl")
            os.makedirs(dl)
            WT.EXPORT_GLOB = os.path.join(dl, "scalpars_orders_paper_*.csv")
            base = dict(direction="LONG", entry_strategy=WT.STRATEGY, status="CLOSED", notional_value=10_000.0, entry_frenzy_willy_trigger="A",
                        entry_price=E, pnl=1.0, closed_at="x")
            pd.DataFrame([dict(base, opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", pnl_percentage=1.2, close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-09T12:00:00", pair="NOKUSDT", pnl_percentage=1.0, close_reason="FRENZY_TP"),
                          dict(base, opened_at="2026-10-09T13:00:00", pair="MANUSDT", pnl_percentage=0.5, close_reason="MANUAL_TP")]
                         ).to_csv(os.path.join(dl, "scalpars_orders_paper_a.csv"), index=False)
            tk = pd.Timestamp("2026-10-09 08:45:09").value // 10**6
            os.makedirs(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT"))
            np.savez_compressed(os.path.join(WT._CACHE, "ticks_q", "KAIAUSDT", "2026-10-09.npz"),
                                t=np.array([tk, tk + 60_000, tk + 300_000, tk + 1_800_000], dtype=np.int64),
                                p=np.array([E, px(-4.5), px(-1.0), px(1.2)], dtype=np.float64))
            sp = os.path.join(td, "s.json")
            out = "\n".join(run(pd.Timestamp("2026-10-09 16:00").value // 10**6, state_path=sp, th={}))
            chk("KAIAUSDT [ticks] no stop +1.20 (TP @ 30 min, low -4.50) → −4: -4.50 (STOP)" in out and "worst no-stop result so far +1.20 %" in out and "UNSCORED (no path yet): 10-09 12:00 NOKUSDT"
                in out and "MANUSDT (MANUAL_TP)" in out and "1/1 match" in out and "COLLECTING" in out and not os.path.exists(sp), f"end-to-end\n{out}")
    finally:
        WT._CACHE, WT.MY_CACHE, WT.EXPORT_GLOB, WT._NET_BLOCKED, WT.deploy_ts = saved
        WT._K1.clear()
    print(f"selftest WILLY_STOP OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
