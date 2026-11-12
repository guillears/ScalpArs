#!/usr/bin/env python3
"""WILLY HT exit study (operator 2026-11-12, after INIT HT 11-12: TP +1.25 at 38 s, then +6.64 % post-exit peak).
Pre-registration: reports/WILLY_HT_EXIT_STUDY_2026-11-12_PREREG.md (grid + decision bar frozen before any result).

Cohorts (each sleeve apart — never pooled):
  MASTER  every archived FRENZY_WILLY_HT / FRENZY_WILLY fill (reports/BASELINE*_orders*.csv via build_willy_exit_cf.archived_fills),
          KEPT = master stack_keep ∧ non-probe (the decision cohort) · ALL = every archived fill incl. the ones today's filters block (context / power)
  B26     export fills (~/Downloads + exports_archive, export_glob, newest wins) opened ≥ 2026-11-11 16:00 UTC and not archived
  BT      2-year today's-stack replay reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv — FRENZY_WILLY only (no HT: every replay fill has turnover
          R < 1) → a WILLY cross-check, never HT evidence
Path: study_b19_fzexit_common.load_path (ticks, then 1m; net % = gross − 0.045 − 0.045 r) — B26 1m bars fetched once into
reports/backtest_cache/study_willy_ht_1112/1m (public klines). Target T per fill: build_willy_exit_cf.resolve_target (stamped T, else rebuilt
from the spike high); BASE = build_willy_exit_cf.walk_v2b (the CF builder's walker = the master's pricing).
Usage: venv/bin/python -I scripts/study_willy_ht_exit_1112.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
import build_willy_exit_cf as CF          # noqa: E402
import study_b19_fzexit_common as FX      # noqa: E402
import review_math as RM                  # noqa: E402
import export_glob as EG                  # noqa: E402

MIN = 60_000
STOP, CAP, LATE_MIN, LATE_TP, ARM_BE, BE = 4.0, 90, 30, 0.25, 1.25, 0.10
SLIP = float(os.environ.get("WHT_SLIP", "0.05"))   # % off every NEW market exit (trail / BE stop); sensitivity via env
ACC = 10_000.0
X_HT = 4.9 / 4 * 1.6       # today's HT one size: WILLY C5 X (4.9 % per −4 stop at 5×) × the DL 533 step to 8× (bracket keep holds the plan)
OUT = os.path.join(ROOT, "reports", "WILLY_HT_EXIT_STUDY_2026-11-12")
B26_FROM = "2026-11-11T16:00"
K1_EXTRA = os.path.join(ROOT, "reports", "backtest_cache", "study_willy_ht_1112", "1m")
VARS = ["BASE", "TP2", "TP3", "RUN05", "RUN10", "RUN15", "RUN10_BE", "HALF_RUN10", "HALF_TP3_BE", "TP3_BE"]

# B26 1m bars (fetched for this study) ride the walker's 1m loader
_orig_load_1m = FX.WT.load_1m


def _load_1m(pair):
    k = _orig_load_1m(pair)
    f = os.path.join(K1_EXTRA, f"{pair}.csv")
    if os.path.exists(f):
        x = FX.WT._read_k1(f)
        k = pd.concat([k, x]).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
        FX.WT._K1[pair] = k
    return k


FX.WT.load_1m = _load_1m


# ─────────────────────────── walker ───────────────────────────
def _ev(P, i):
    o = P["op"][i]
    return bool(P["bar"][i]), (float(o) if np.isfinite(o) else float("nan"))


def leg(P, tp, run_k=None, atr=None, be_after=None, be_floor=None):
    """one position leg on a net path → (pct or nan, why).
    pre-arm (until the first event with hi ≥ tp, or ≥ be_after for the BE leg): today's rules — stop −4, late TP +0.25 from minute 30
    (booked at the line / the open or print when the first late event is already through it), cap 90 at the close.
    run_k: at the first touch of tp → trail (peak of EARLIER events, starting at tp) − k × atr; floor ≥ be_floor when given.
    be_after (no run_k): after an EARLIER event touched be_after, the stop is +be_floor and the TP stays at tp."""
    te, t = int(P["te"]), P["t"]
    n = int(np.searchsorted(t, te + CAP * MIN, side="right"))
    first_late = None
    idx = np.flatnonzero(t[:n] > te + LATE_MIN * MIN)
    if len(idx):
        first_late = int(idx[0])
    armed, be_on, pk = False, False, float("nan")
    for i in range(n):
        bar, o = _ev(P, i)
        hi, lo = float(P["hi"][i]), float(P["lo"][i])
        if armed:                                                   # trailing runner
            fl = pk - run_k * atr
            if be_floor is not None:
                fl = max(fl, be_floor)
            fl = max(fl, -STOP)
            if lo <= fl:
                v = (min(fl, o) if np.isfinite(o) else fl) if bar else lo
                return v - SLIP, "TRAIL"
            pk = max(pk, hi)
            continue
        stop_line = be_floor if be_on else -STOP
        if lo <= stop_line:
            v = (min(stop_line, o) if np.isfinite(o) else stop_line) if bar else lo
            return (v - SLIP if be_on else v), ("BE_STOP" if be_on else "STOP")
        lt = (first_late is not None and i >= first_late and not be_on)
        line = LATE_TP if (lt and LATE_TP < tp) else tp
        if hi >= line:
            if run_k is not None and line == tp:
                armed, pk = True, float(tp)
                continue
            if lt and i == first_late and line == LATE_TP:
                v = (o if (np.isfinite(o) and o >= line) else line) if bar else hi
                return float(v), "LATE_TP"
            return float(line), ("LATE_TP" if line == LATE_TP else "TP")
        if be_after is not None and hi >= be_after:
            be_on = True                                            # from the NEXT event
    if P["end_ms"] < te + CAP * MIN or n == 0:
        return float("nan"), "UNRES"
    return float(P["cl"][n - 1]) - (SLIP if armed else 0.0), "CAP"


def price_variants(P, T, atr):
    """→ {var: (pct, why)}. BASE = the CF builder's V2b walker (parity with the master's pricing)."""
    b = CF.walk_v2b(P, T, TH)
    out = {"BASE": (float(b["pct"]) if b["why"] != "UNRES" else float("nan"), b["why"])}
    out["BASE_MINE"] = leg(P, T)                                    # my walker at T — must equal BASE (walker parity)
    out["TP2"] = leg(P, max(T, 2.0))
    out["TP3"] = leg(P, max(T, 3.0))
    if atr and atr > 0:
        for k, name in ((0.5, "RUN05"), (1.0, "RUN10"), (1.5, "RUN15")):
            out[name] = leg(P, T, run_k=k, atr=atr)
        out["RUN10_BE"] = leg(P, T, run_k=1.0, atr=atr, be_floor=BE)
        r = out["RUN10"]
        out["HALF_RUN10"] = (0.5 * out["BASE"][0] + 0.5 * r[0], f"½{out['BASE'][1]}+½{r[1]}")
    else:
        for name in ("RUN05", "RUN10", "RUN15", "RUN10_BE", "HALF_RUN10"):
            out[name] = (float("nan"), "NO_ATR")
    t3 = leg(P, max(T, 3.0), be_after=ARM_BE, be_floor=BE)
    out["TP3_BE"] = t3
    out["HALF_TP3_BE"] = (0.5 * out["BASE"][0] + 0.5 * t3[0], f"½{out['BASE'][1]}+½{t3[1]}")
    return out


# ─────────────────────────── cohorts ───────────────────────────
def _key(o, p):
    return str(o).replace(" ", "T")[:19] + "|" + str(p)


def master_flags():
    M = pd.read_csv(FX.MASTER, low_memory=False)
    M = M[M.entry_strategy.isin(CF.STRATS)].copy()
    M["k"] = [_key(o, p) for o, p in zip(M.opened_at, M.pair)]
    probe = M.cell_multiplier_source.astype(str).str.endswith("_PROBE")
    keep = (M.stack_keep == True) & ~probe                         # noqa: E712
    return dict(zip(M.k, keep)), dict(zip(M.k, M.era.astype(str))), dict(zip(M.k, pd.to_numeric(M.stack_pct, errors="coerce")))


def b26_fills(archived_keys):
    fr = []
    for f in EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in d:
            continue
        d = d[d.entry_strategy.astype(str).isin(CF.STRATS) & (d.status.astype(str) == "CLOSED")]
        if len(d):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True).sort_values("_m").drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")
    d = d[d.opened_at.astype(str) >= B26_FROM]
    d = d[~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")]
    d = d[~d.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "EXTERNAL"))]
    return d[[_key(o, p) not in archived_keys for o, p in zip(d.opened_at, d.pair)]]


def price_rows(D, block, keepmap=None, eramap=None, stackmap=None, booked_col="pnl_percentage"):
    rows, cache, tcache = [], {}, {}
    for r in D.to_dict("records"):
        te = CF._ms(r.get("opened_at")); E = CF._num(r.get("entry_price"))
        k = _key(r.get("opened_at"), r.get("pair"))
        base = dict(block=block, key=k, opened_at=str(r.get("opened_at"))[:19], pair=r.get("pair"), sleeve=r.get("entry_strategy"),
                    era=(eramap or {}).get(k, block), keep=(keepmap or {}).get(k, True), booked=CF._num(r.get(booked_col)),
                    stack_pct=(stackmap or {}).get(k), atr=CF._num(r.get("entry_atr_pct")), post_peak=CF._num(r.get("post_exit_peak_pnl")))
        if te is None or not E:
            rows.append(dict(base, why_unpriced="no entry")); continue
        P = FX.load_path(r.get("pair"), te, E, CAP, cache)
        if not len(P["t"]):
            rows.append(dict(base, why_unpriced="no path")); continue
        tg = CF.resolve_target(TH, r, tcache)
        T = tg["T"] if tg["T"] is not None else 1.25
        V = price_variants(P, T, base["atr"])
        rec = dict(base, T=T, t_src=tg["t_src"] if tg["T"] is not None else "assumed1.25", src=("ticks" if P["n_bar"] == 0 else "1m" if P["n_tick"] == 0 else "ticks+1m"),
                   path_peak=float(np.nanmax(P["hi"][: int(np.searchsorted(P["t"], te + CAP * MIN, side="right"))])) if len(P["hi"]) else np.nan)
        for v, (pct, why) in V.items():
            rec[v] = pct; rec[v + "_why"] = why
        rows.append(rec)
    return pd.DataFrame(rows)


def label(r):
    """path label from BASE + the 90-min path: CAPTURED (exit ≥ T and the path peak < T + 1) · LEFT_ON_TABLE (exit at T, path peak ≥ T + 1)
    · STOPPED / LATE / CAP (never reached T)."""
    w = str(r.get("BASE_why"))
    if w == "TP":
        return "LEFT_ON_TABLE" if (r.get("path_peak") or -9) >= r["T"] + 1.0 else "CAPTURED"
    return {"STOP": "NEVER_WORKED(stop)", "LATE_TP": "LATE(+0.25 line)", "CAP": "DRIFT(cap)"}.get(w, w)


# ─────────────────────────── stats ───────────────────────────
def boot_ci(d, days, n=4000, seed=7):
    d = np.asarray(d, float); days = np.asarray(days)
    u = np.unique(days)
    if len(u) < 2:
        return (float("nan"), float("nan"))
    idx = {x: np.flatnonzero(days == x) for x in u}
    rng = np.random.default_rng(seed)
    s = [d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].sum() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def luck_p(D, days, n=4000, seed=11):
    """each variant's Σ Δ vs the best Σ Δ of the grid under day sign-flips → {var: p}."""
    M = np.asarray(D, float); u = list(dict.fromkeys(days)); di = np.array([u.index(x) for x in days])
    S = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n, len(u)))[:, di]
    best = (S @ M).max(axis=1)
    return {c: (1 + int((best >= M[:, j].sum()).sum())) / (1 + n) for j, c in enumerate(D.columns)}


def dcr(g, col, days):
    r = (g[col] * X_HT / 100).groupby(pd.to_datetime(g.opened_at).dt.normalize()).sum().reindex(days).fillna(0.0)
    return ((1 + r).prod() ** (1 / len(r)) - 1) * 100 if len(r) else float("nan")


def days_of(g):
    if not len(g):
        return pd.DatetimeIndex([])
    o = pd.to_datetime(g.opened_at)
    return pd.date_range(o.min().normalize(), o.max().normalize(), freq="D")


def table(g, title, days=None):
    g = g[g.BASE.notna()]
    days = days if days is not None else days_of(g)
    L = [f"\n### {title} — N {len(g)} · {g.opened_at.str[:10].nunique()} trade days · {len(days)} calendar days",
         "| exit | N | WR | avg % | Σ % | Δ vs BASE | gained on (n) | gave back on (n) | %/day one size | LOO |", "|---|---|---|---|---|---|---|---|---|---|"]
    base_dcr = dcr(g, "BASE", days)
    for v in VARS:
        x = g[v]
        ok = x.notna()
        d = (x - g.BASE)[ok]
        up, dn = d[d > 1e-9], d[d < -1e-9]
        loo = RM.loo_text(d.values, list((g.pair + " " + g.opened_at.str[5:10])[ok])) if v != "BASE" else "—"
        L.append(f"| {v} | {int(ok.sum())} | {(x[ok] > 0).mean():.0%} | {x[ok].mean():+.2f} | {x[ok].sum():+.2f} | "
                 f"{'—' if v == 'BASE' else f'{d.sum():+.2f}'} | {up.sum():+.2f} ({len(up)}) | {dn.sum():+.2f} ({len(dn)}) | "
                 f"{base_dcr:+.2f}" + ("" if v == "BASE" else f" → {dcr(g.assign(**{v: x.fillna(g.BASE)}), v, days):+.2f}") + f" | {loo} |")
    return L


def main():
    global TH
    TH = CF.load_th()
    keepmap, eramap, stackmap = master_flags()
    A = CF.archived_fills()
    A = A[~A.cell_multiplier_source.astype(str).str.endswith("_PROBE")]
    A = A[~A.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "EXTERNAL"))]
    A = A.drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")
    R = price_rows(A, "MASTER", keepmap, eramap, stackmap)
    akeys = set(R.key)
    B = b26_fills(akeys)
    R26 = price_rows(B, "B26") if len(B) else pd.DataFrame()
    bt = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False)
    bt = bt[bt.entry_strategy == "FRENZY_WILLY"].copy()
    bt["opened_at"] = pd.to_datetime(bt.opened_at).dt.strftime("%Y-%m-%dT%H:%M:%S")
    RBT = price_rows(bt, "BT")
    ALL = pd.concat([R, R26, RBT], ignore_index=True)
    ALL["label"] = [label(r) for r in ALL.to_dict("records")]
    ALL["day"] = ALL.opened_at.str[:10]
    if abs(SLIP - 0.05) < 1e-9:
        ALL.to_csv(OUT + "_rows.csv", index=False)

    L = []
    # parity
    for blk in ("MASTER", "B26", "BT"):
        g = ALL[(ALL.block == blk) & ALL.BASE.notna()]
        if not len(g):
            continue
        mine = (g.BASE_MINE - g.BASE).abs() <= 0.02
        ref = g.stack_pct if blk == "MASTER" else g.booked
        ok = ref.notna()
        m2 = ((g.BASE - ref).abs() <= 0.10)[ok]
        L.append(f"- {blk}: priced {len(g)} of {int((ALL.block == blk).sum())} · my walker = the CF walker on {int(mine.sum())}/{len(g)} · "
                 f"BASE vs {'master stack_pct' if blk == 'MASTER' else 'booked %'} within 0.10 on {int(m2.sum())}/{int(ok.sum())}")
    par = L

    def blockset(sleeve):
        out = []
        m = ALL[(ALL.block == "MASTER") & (ALL.sleeve == sleeve)]
        out += table(m[m.keep == True], f"{sleeve} · MASTER KEPT (today's filters — the decision cohort)")   # noqa: E712
        out += table(m, f"{sleeve} · MASTER ALL archived (incl. fills today's filters block — context / power)")
        b = ALL[(ALL.block == "B26") & (ALL.sleeve == sleeve)]
        if len(b):
            out += table(b, f"{sleeve} · B26 forward")
        return out

    T_HT = blockset("FRENZY_WILLY_HT")
    T_W = blockset("FRENZY_WILLY")
    T_BT = table(ALL[ALL.block == "BT"], "FRENZY_WILLY · BACKTEST 2-year replay (WILLY cross-check — NOT HT evidence)")

    # decision stats on HT kept + HT all
    dec = []
    for nm, g in (("HT KEPT", ALL[(ALL.block == "MASTER") & (ALL.sleeve == "FRENZY_WILLY_HT") & (ALL.keep == True)]),   # noqa: E712
                  ("HT ALL", ALL[(ALL.block == "MASTER") & (ALL.sleeve == "FRENZY_WILLY_HT")]),
                  ("WILLY KEPT", ALL[(ALL.block == "MASTER") & (ALL.sleeve == "FRENZY_WILLY") & (ALL.keep == True)]),   # noqa: E712
                  ("BT WILLY", ALL[ALL.block == "BT"])):
        g = g[g.BASE.notna()]
        D = pd.DataFrame({v: (g[v] - g.BASE).fillna(0.0).values for v in VARS[1:]})
        if not len(g):
            continue
        ps = luck_p(D, list(g.day))
        dec.append(f"\n**{nm}** (N {len(g)}, {g.day.nunique()} days):")
        for v in VARS[1:]:
            lo, hi = boot_ci(D[v], g.day)
            top = g.assign(d=D[v].values).groupby("pair").d.sum()
            tp = top.abs().max() / abs(D[v].sum()) if D[v].sum() else float("nan")
            dec.append(f"- {v}: Σ Δ {D[v].sum():+.2f} · day-boot 95 % CI [{lo:+.2f}, {hi:+.2f}] · luck p (best of {len(VARS) - 1}) {ps[v]:.2f} · "
                       f"top pair {top.abs().idxmax() if len(top) else '–'} {tp:.0%} of Δ")
    labels = []
    for nm, g in (("HT master all", ALL[(ALL.block == "MASTER") & (ALL.sleeve == "FRENZY_WILLY_HT")]),
                  ("WILLY master all", ALL[(ALL.block == "MASTER") & (ALL.sleeve == "FRENZY_WILLY")]),
                  ("B26 HT + WILLY", ALL[ALL.block == "B26"]), ("BT WILLY", ALL[ALL.block == "BT"])):
        g = g[g.BASE.notna()]
        labels.append(f"- {nm}: " + " · ".join(f"{k} {n} ({g[g.label == k].BASE.sum():+.1f} %)" for k, n in g.label.value_counts().items()))
    detail = ["\n| block | opened | pair | sleeve | kept | T | label | BASE | " + " | ".join(VARS[1:]) + " |", "|" + "---|" * (6 + len(VARS))]
    for r in ALL[(ALL.block != "BT") & (ALL.sleeve == "FRENZY_WILLY_HT") & ALL.BASE.notna()].sort_values("opened_at").itertuples():
        detail.append(f"| {r.block} | {r.opened_at[5:16]} | {r.pair} | HT | {'✓' if r.keep else '–'} | {r.T:.2f} | {r.label} | {r.BASE:+.2f} | "
                      + " | ".join(f"{getattr(r, v):+.2f}" if pd.notna(getattr(r, v)) else "–" for v in VARS[1:]) + " |")
    txt = ["# WILLY HT exit study — 2026-11-12 (results)", "", f"Slip on new market exits: {SLIP:.2f} %. One size: HT X = {X_HT:.2f} (× $10,000).", "",
           "## Replica parity", *par, "", "## Path labels (BASE)", *labels, "", "## WILLY HT", *T_HT, "", "## FRENZY_WILLY (own row — same exit)", *T_W,
           "", "## Backtest cross-check", *T_BT, "", "## Decision statistics (Δ vs BASE)", *dec, "", "## Every HT fill", *detail]
    open(OUT + f"_results_slip{SLIP:.2f}.md", "w").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
