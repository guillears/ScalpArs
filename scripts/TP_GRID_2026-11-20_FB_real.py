#!/usr/bin/env python3
"""TP_GRID_2026-11-20_FB — STEP 1: the FLATBASE_SHORT TP × SL grid on REAL fills (master on today's stack + the current batch).

Cohort + data: scripts/TP_GRID_2026-11-20_FB_fetch.py → reports/TP_GRID_2026-11-20_FB_data/{cohort.csv, k1m/, funding/}.
Entry: REAL fills (opened after the FB_CONFIRM deploy) = the real fill price at the fill minute's 1m bar. Fills opened before it =
MODEL_CONFIRM: the master's confirm entry rebuilt on 1m (first 4-h close in (fill, fill + 24 h] below the fill price → the next bar's
open) with a 0.25 % entry drift (the real FB entry slip), so the TP / stop levels sit where a real fill would.
Walk: every 1m bar from the entry bar (its pre-fill seconds included — conservative for the stop) for the hold.
  primary (live costs): entry taker 0.05 % · TP resting maker 0.02 %, filled only when traded THROUGH the line by ≥ 0.05 %, at the line ·
                        stop / time = market: taker 0.05 % + 0.25 % slip from max(line, bar open) · funding from Binance history (short
                        receives + rate × 100 % per settlement inside the hold).
  parity (paper)     : TP through by any amount, net = TP − 0.06 · stop at max(line, open), net = gross − 0.09 · same for time — vs booked.
Same-bar TP + stop → stop first; counted. A cell still open at the data end is marked to the last close (counted).
Output: reports/TP_GRID_2026-11-20_FB_data/real_*.csv + real_tables.md
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_FB_real.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("G", os.path.join(ROOT, "scripts", "TP_GRID_2026-11-20_FB_common.py"))
G = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(G)
D = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data")
HR, H4, M1 = 3_600_000, 14_400_000, 60_000
DRIFT, TAKER, MAKER, SLIP, THROUGH = 0.25, 0.05, 0.02, 0.25, 0.05
FUND_THR = 0.00005
SPANS = {"B24": ("2026-10-23", "2026-10-23"), "B25": ("2026-10-23", "2026-11-11"), "B26": ("2026-11-11", "2026-11-14"),
         "B27": ("2026-11-14", "2026-11-17"), "B28": ("2026-11-18", "2026-11-20")}
OUTMD = []


def P(s=""):
    print(s)
    OUTMD.append(s)


def load_k(pair):
    k = pd.read_csv(os.path.join(D, "k1m", f"{pair}.csv"))
    return k.open_time.values.astype("int64"), k.o.values.astype(float), k.h.values.astype(float), k.l.values.astype(float), k.c.values.astype(float)


def load_f(pair):
    f = os.path.join(D, "funding", f"{pair}.csv")
    x = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["t", "rate", "mark"])
    return x.t.values.astype("int64"), x.rate.values.astype(float)


def entry_of(r, t, o, c):
    """→ (entry ms, entry price, entry bar idx, note)."""
    if r.kind == "REAL":
        i = int(np.searchsorted(t, r.te // M1 * M1))
        return r.te, float(r.entry_price), i, "real fill"
    B = (r.te // H4 + 1) * H4
    while B <= r.te + 24 * HR:
        j = np.searchsorted(t, B - M1)
        if j < len(t) and t[j] == B - M1 and c[j] < r.entry_price and j + 1 < len(t) and t[j + 1] == B:
            return B, float(o[j + 1]) * (1 - DRIFT / 100), j + 1, f"confirm {pd.to_datetime(B, unit='ms'):%m-%d %H:%M}"
        B += H4
    return None, None, None, "confirm never came"


def walk(t, o, h, l, c, i, E, tp, sl, hold_h, mode):
    """→ (net % pre-funding, reason, exit ms, same-bar flag)."""
    end_t = t[i] + hold_h * HR
    last = int(np.searchsorted(t, end_t)) - 1
    S = E * (1 + sl / 100); T = E * (1 - tp / 100)
    Tt = T * (1 - (THROUGH if mode == "primary" else 1e-9) / 100)
    hh, ll = h[i:last + 1], l[i:last + 1]
    ks, kt = G.first_hits(hh, ll, S, np.nextafter(Tt, 0) if mode == "parity" else Tt)
    m = len(hh)
    if ks < m and ks <= kt:
        j = i + ks
        if mode == "primary":
            X = max(S, o[j]) * (1 + SLIP / 100); net = (1 - X / E) * 100 - TAKER - TAKER * X / E
        else:
            X = max(S, o[j]); net = (1 - X / E) * 100 - 0.09
        return net, "STOP", int(t[j]) + M1, ks == kt
    if kt < m:
        j = i + kt
        net = tp - TAKER - MAKER * (1 - tp / 100) if mode == "primary" else tp - 0.06
        return net, "TP", int(t[j]) + M1, False
    complete = t[last] + M1 >= end_t
    if mode == "primary":
        X = c[last] * (1 + SLIP / 100); net = (1 - X / E) * 100 - TAKER - TAKER * X / E
    else:
        X = c[last]; net = (1 - X / E) * 100 - 0.09
    return net, ("TIME" if complete else "OPEN"), int(t[last]) + M1, False


def fund(ft, fr, a, b):
    m = (ft > a) & (ft <= b)
    return float(fr[m].sum() * 100)


def build():
    R = pd.read_csv(os.path.join(D, "cohort.csv"))
    rows = []
    for r in R.itertuples():
        t, o, h, l, c = load_k(r.pair)
        ft, fr = load_f(r.pair)
        te, E, i, note = entry_of(r, t, o, c)
        d = dict(k=r.k, pair=r.pair, batch=r.batch, kind=r.kind, booked=r.booked, fund_rate=r.entry_funding_rate, note=note,
                 fund_in_pnl=bool(r.batch in ("B27", "B28") or str(r.k) >= "2026-11-14"))
        if te is None:
            rows.append(d); continue
        d.update(te_eff=te, E=E, win=str(pd.to_datetime(te, unit="ms").floor("4h")), data_end=int(t[-1]) + M1)
        for hd in (24, 48, 72):
            for tp, sl in G.CELLS:
                net, why, xt, sb = walk(t, o, h, l, c, i, E, tp, sl, hd, "primary")
                fu = fund(ft, fr, te, xt)
                kk = f"{tp}_{sl}_{hd}"
                d[kk] = net + fu; d[kk + "_why"] = why; d[kk + "_x"] = xt; d[kk + "_sb"] = sb; d[kk + "_fund"] = fu
        net, why, xt, sb = walk(t, o, h, l, c, i, E, 5, 10, 72, "parity")
        d.update(par_net=net, par_why=why, par_fund=fund(ft, fr, te, xt), par_min=(xt - te) / M1)
        rows.append(d)
    X = pd.DataFrame(rows)
    X.to_pickle(os.path.join(D, "real_grid.pkl"))
    return X


def col(tp, sl, hd=72):
    return f"{tp}_{sl}_{hd}"


def main():
    X = build()
    P("# TP_GRID FB — real fills (master on today's stack + current batch B28)")
    bad = X[X.E.isna()]
    if len(bad):
        P(f"\n⚠ no entry rebuilt for {len(bad)}: " + ", ".join(f"{r.pair} {r.k} ({r.note})" for r in bad.itertuples()))
    X = X[X.E.notna()].copy()
    X["close_day"] = None
    # parity
    X["par_cmp"] = np.where(X.fund_in_pnl, X.par_net + X.par_fund, X.par_net)
    X["par_diff"] = X.par_cmp - X.booked
    P("\n## Parity — today's −5 / +10 / 72 h re-walked under PAPER costs vs the booked / stack net (REAL = booked; MODEL_CONFIRM = the "
      "master's stack_pct from its own 5m confirm walk)")
    pt = X[["batch", "k", "pair", "kind", "note", "booked", "par_cmp", "par_diff", "par_why", "par_min"]].copy()
    pt["live-cost net"] = X[col(5, 10)]
    pt["live why"] = X[col(5, 10) + "_why"]
    P(G.md(pt.round(3), 2))
    P(f"\n|Δ| ≤ 0.30 on {int((X.par_diff.abs() <= 0.30).sum())} of {len(X)} fills · same exit reason on "
      f"{int(((X.par_why == 'TP') == (X.booked > 0)).sum())} of {len(X)}")
    X.to_csv(os.path.join(D, "real_fills.csv"), index=False)
    for name, S in (("TODAY'S STACK (stack_keep + DL 601 funding filter)", X[~(X.fund_rate > FUND_THR)]), ("stack_keep only (pre-DL 601)", X)):
        report(S.copy(), name, primary=name.startswith("TODAY"))
    with open(os.path.join(D, "real_tables.md"), "w") as fh:
        fh.write("\n".join(OUTMD) + "\n")


def report(S, name, primary):
    P(f"\n## {name}: {len(S)} fills ({int((S.kind == 'REAL').sum())} real entries, {int((S.kind != 'REAL').sum())} model-confirm entries) · "
      f"{S.win.nunique()} market windows")
    sb = {c_: int(S[col(*c_) + "_sb"].sum()) for c_ in G.CELLS}
    op = {c_: int((S[col(*c_) + "_why"] == "OPEN").sum()) for c_ in G.CELLS}
    P(f"same-bar (1m) TP + stop, stop first: {sum(sb.values())} fill-cells over the grid (max {max(sb.values())} per cell) · still OPEN at the "
      f"data end, marked to the last close: max {max(op.values())} per cell ({[f'-{a}/+{b}' for (a, b), v in op.items() if v]})")
    blocks = [(b, S.batch == b) for b in ("B24", "B25", "B26", "B27", "B28")] + [("master B24–B27", S.batch != "B28"), ("ALL", S.batch.notna())]
    rows = []
    for bn, m in blocks:
        g = S[m]
        if not len(g):
            continue
        if bn in SPANS:
            d0, d1 = SPANS[bn]
        elif bn.startswith("master"):
            d0, d1 = "2026-10-23", "2026-11-17"
        else:
            d0, d1 = "2026-10-23", "2026-11-20"
        base = g[col(5, 10)].values
        for tp, sl in G.CELLS:
            v = g[col(tp, sl)].values
            cd = pd.to_datetime(g[col(tp, sl) + "_x"], unit="ms").dt.strftime("%Y-%m-%d")
            dd1 = max(d1, cd.max())
            rr = G.cell_row(v, cd, d0, dd1)
            dm, lo, hi, pg = G.cluster_boot(v - base, g.win.values) if len(g) > 1 else (np.nan,) * 4
            rows.append(dict(block=bn, tp=tp, sl=sl, **rr, d_sum=(v - base).sum(), d_avg=dm, d_lo=lo, d_hi=hi, P_gain=pg))
    T = pd.DataFrame(rows)
    tag = "today" if primary else "keep"
    T.to_csv(os.path.join(D, f"real_cells_{tag}.csv"), index=False)
    g = T[T.block == "ALL"]
    P("\n### ALL real — avg net % per fill (live costs + funding; TP rows × stop columns; today bold)")
    P(G.heat({(r.tp, r.sl): r.avg for r in g.itertuples()}))
    P("\n### ALL real — WR %")
    P(G.heat({(r.tp, r.sl): r.WR for r in g.itertuples()}, fl=0))
    P("\n### ALL real — Σ net % pts")
    P(G.heat({(r["tp"], r["sl"]): r["sum"] for r in g.to_dict("records")}, fl=1))
    P("\n### ALL real — one-size DCR %/day (X = 0.61, every day 10-23 → 11-20)")
    P(G.heat({(r.tp, r.sl): r.DCR for r in g.itertuples()}, fl=3))
    P("\n### ALL real — max drawdown % (one size)")
    P(G.heat({(r.tp, r.sl): r.maxDD for r in g.itertuples()}, fl=1))
    # family-wise null (window-clustered sign flips)
    D_ = np.column_stack([S[col(*c_)].values - S[col(5, 10)].values for c_ in G.CELLS])
    mx, p = G.fw_null(D_, S.win.values)
    obs = D_.mean(0)
    b = int(np.argmax(obs))
    P(f"\nfamily-wise null (random sign per 4-h market WINDOW on every cell's paired Δ, best of 49): best cell −{G.CELLS[b][0]} / +{G.CELLS[b][1]} "
      f"Δ {obs[b]:+.3f} %/fill · null 95th pct of the best {np.percentile(mx, 95):+.3f} · p_fw {p[b]:.3f}")
    allT = g.sort_values("avg", ascending=False)
    top = [tuple(x) for x in allT[["tp", "sl"]].values if tuple(x) != (5, 10)][:5]
    sel = T[[(a, b_) in ([(5, 10)] + top) for a, b_ in zip(T.tp, T.sl)]].copy()
    sel["cell"] = "−" + sel.tp.astype(str) + " / +" + sel.sl.astype(str)
    P("\n### Today + the top 5 cells (by ALL avg), per batch")
    P(G.md(sel[["block", "cell", "N", "WR", "avg", "sum", "DCR", "maxDD", "worst_day", "BE_WR", "d_sum", "d_avg", "d_lo", "d_hi", "P_gain"]].round(3), 2))
    P("\n### Concentration + leave-one-out for the top 5 (Δ vs today, ALL)")
    rr = []
    for tp, sl in top:
        d = (S[col(tp, sl)] - S[col(5, 10)]).values
        lo, hi, worst = G.loo_range(d)
        big = S.iloc[int(np.argmax(np.abs(d)))]
        rr.append(dict(cell=f"−{tp} / +{sl}", d_sum=d.sum(), d_avg=d.mean(), loo_min=lo, loo_max=hi, drops_most=f"{S.iloc[worst].pair} {S.iloc[worst].k[:10]}",
                       biggest_fill=f"{big.pair} {big.k[:10]} {d[int(np.argmax(np.abs(d)))]:+.2f}", top_pair_pct=G.top_share(d, S.pair.values),
                       top_window_pct=G.top_share(d, S.win.values), fills_changed=int((np.abs(d) > 1e-9).sum()), p_fw=p[G.CELLS.index((tp, sl))]))
    P(G.md(pd.DataFrame(rr).round(3), 2))
    # two-sided stop split
    P("\n### Two-sided stop split on real fills — stop-only changes at TP −5 (vs +10)")
    bw = S[col(5, 10) + "_why"]
    sp = []
    for sl in G.SLS:
        if sl == 10:
            continue
        d = S[col(5, sl)] - S[col(5, 10)]
        w2 = S[col(5, sl) + "_why"]
        if sl > 10:
            grp = {"saved": (bw == "STOP") & (w2 != "STOP"), "deeper": (bw == "STOP") & (w2 == "STOP"), "unchanged": bw != "STOP"}
        else:
            grp = {"new stops": (bw != "STOP") & (w2 == "STOP"), "cheaper": (bw == "STOP") & (w2 == "STOP"), "unchanged": (bw != "STOP") & (w2 != "STOP")}
        r = dict(stop=f"+{sl}")
        for k_, mm in grp.items():
            r[f"{k_} N"] = int(mm.sum()); r[f"{k_} ΣΔ"] = d[mm].sum()
            if k_ == "new stops":
                r["new-stop pairs"] = ", ".join(S[mm].pair.str.replace("USDT", ""))
        r["total ΣΔ"] = d.sum()
        sp.append(r)
    P(G.md(pd.DataFrame(sp).round(2), 2))
    # per-fill view of today's stopped fills
    P("\n### Today's stopped / losing fills on the grid (net % at each stop width, TP −5)")
    lz = S[S[col(5, 10)] <= 0]
    P(G.md(pd.DataFrame([dict(fill=f"{r.pair} {r.k[:16]}", **{f"+{sl}": S.loc[r.Index, col(5, sl)] for sl in G.SLS})
                         for r in lz.itertuples()]).round(2), 2))
    # hold variants
    P("\n### Hold 24 h / 48 h / 72 h for today + the top 3 cells (ALL real)")
    hr_ = []
    for tp, sl in [(5, 10)] + top[:3]:
        for hd in (24, 48, 72):
            v = S[col(tp, sl, hd)]
            hr_.append(dict(cell=f"−{tp} / +{sl}", hold=hd, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum()))
    P(G.md(pd.DataFrame(hr_).round(2), 2))


if __name__ == "__main__":
    main()
