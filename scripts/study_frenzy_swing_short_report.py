#!/usr/bin/env python3
"""FRENZY swing short — REPORT tables from study_frenzy_swing_short.py outputs (cells.pkl, _cells_year.csv). Prints markdown sections
and writes reports/FRENZY_SWING_SHORT_2026-10-20_trades.csv (every trade of the primary / secondary / best cells, all blocks) and
_episodes_real.csv (per-episode table, blocks ① ②). The narrative .md is written by hand from these tables.
Usage: venv/bin/python scripts/study_frenzy_swing_short_report.py [best_cell]"""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
_a, sys.argv = sys.argv, [sys.argv[0]]
import study_frenzy_swing_short as S  # noqa: E402  (argparse defaults only; main guarded)
sys.argv = _a
D, OUT = S.D, S.OUT
X = pd.read_pickle(os.path.join(D, "cells.pkl"))
CS = pd.read_csv(OUT + "_cells_year.csv").set_index("cell")
eps = pickle.load(open(os.environ.get("EPS", os.path.join(D, "episodes.pkl")), "rb"))
PRI, SEC = S.cell_name(*S.PRIMARY), S.cell_name(*S.SECONDARY)
best = sys.argv[1] if len(sys.argv) > 1 else CS[CS.N >= 100].sort_values("lo", ascending=False).index[0]
P = lambda *a: print(*a)


def md(df, floatfmt="+.2f", index=True):
    d = df.reset_index() if index else df
    f = lambda v: (format(v, floatfmt) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("–" if isinstance(v, (float, np.floating)) else str(v)))
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def block_rows(cell, real_only=True):
    out = []
    for b in ("1_live", "2_master", "3_year"):
        g = X[(X.cell == cell) & (X.block == b)]
        if b != "3_year" and real_only:
            g = g[g.real_ep]
        out.append((b, g))
    return out


def three_blocks(cell):
    P(f"\n#### {cell}\n")
    P("| block | N | episodes | days | WR | avg % | median | 95 % day-boot | Σ % | worst | stopped | funding avg |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    allg = []
    for b, g in block_rows(cell):
        s = S.summ(g); allg.append(g)
        if s.get("N", 0):
            P(f"| {b} | {s['N']} | {s['eps']} | {s['days']} | {s['WR']:.0f}% | {s['avg']:+.2f} | {s['med']:+.2f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | {s['tot']:+.0f} | {s['worst']:+.1f} | {s['stop_rate']:.0f}% | {s['fund']:+.2f} |")
        else:
            P(f"| {b} | 0 | | | | | | | | | | |")
    g = pd.concat(allg); s = S.summ(g)
    P(f"| **total** | {s['N']} | {s['eps']} | {s['days']} | {s['WR']:.0f}% | {s['avg']:+.2f} | {s['med']:+.2f} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | {s['tot']:+.0f} | {s['worst']:+.1f} | {s['stop_rate']:.0f}% | {s['fund']:+.2f} |")


def robustness(cell):
    g = X[(X.cell == cell) & (X.block == "3_year")]
    P(f"\n#### robustness — {cell} (year block)\n")
    P("| cut | N | WR | avg % |"); P("|---|---|---|---|")
    for p in ("2025", "2026H1", "2026H2"):
        x = g[g.period == p]; P(f"| {p} | {len(x)} | {(x.net > 0).mean() * 100:.0f}% | {x.net.mean():+.2f} |")
    for m in sorted(g.month.unique()):
        x = g[g.month == m]; P(f"| {m} | {len(x)} | {(x.net > 0).mean() * 100:.0f}% | {x.net.mean():+.2f} · without: {g[g.month != m].net.mean():+.2f} |")
    s = S.summ(g)
    P(f"| best 5 % removed | | | {s['ex_top5']:+.2f} |")
    P(f"| before funding | | | {g.net_nofund.mean():+.2f} |")
    P(f"| funding missing (assumed −0.10 %/8 h) | {int(g.fund_missing.sum())} | | |")
    pc = g.groupby("pair").net.sum().sort_values(); dc = g.groupby("day").net.sum().sort_values()
    P(f"\ntop pair {pc.index[-1]} {pc.iloc[-1]:+.0f} of Σ {g.net.sum():+.0f} ({s['top_pair_share']:.0f} %) · top day {dc.index[-1]} {dc.iloc[-1]:+.0f} "
      f"({s['top_day_share']:.0f} %) · worst pair {pc.index[0]} {pc.iloc[0]:+.0f} · worst day {dc.index[0]} {dc.iloc[0]:+.0f} · pairs positive "
      f"{(pc > 0).mean() * 100:.0f} % of {len(pc)}")
    r = CS.loc[cell]
    P(f"shuffled-entry null (pre-funding, random bar in the same episode): real {r.real_nofund:+.2f} vs null {r.null_mean:+.2f} · p own {r.p_own:.3f} · "
      f"family-wise p (max-z over {len(CS)} cells) {r.p_fw:.3f}")


def tails(cell, k=10):
    g = X[(X.cell == cell)].sort_values("net").head(k)
    P(f"\n#### worst {k} — {cell}\n")
    P("| pair | entry (UTC) | block | net % | exit | max against within the hold (no stop) | what happened |"); P("|---|---|---|---|---|---|---|")
    for r in g.itertuples():
        e = eps[r.ep]; i = int(np.searchsorted(e["t"], r.t_ms - S.BAR)); hb = int(r.hold) * 12
        hh = e["h"][i + 1:i + 1 + hb]; mx = (np.nanmax(hh) / r.c - 1) * 100 if len(hh) else np.nan
        cl = e["c"][min(i + hb, len(e["c"]) - 1)]; endr = (1 - cl / r.c) * 100
        P(f"| {r.pair} | {r.entry_dt:%Y-%m-%d %H:%M} | {r.block} | {r.net:+.1f} | {r.reason} | +{mx:.0f} % | close at hold end {endr:+.1f} % for the short |")


def interactions(cell):
    RF, YF = S.real_fills(), S.replay_fills()
    g = X[X.cell == cell]
    P(f"\n#### interactions — {cell}: FRENZY long-family fills on the SAME pair that the short would block / that would block the short\n")
    P("| block | shorts | longs opened while the short is open (blocked) | their avg % | their Σ % | shorts refused (a long already open at entry) |")
    P("|---|---|---|---|---|---|")
    for b, src in (("1_live", RF), ("2_master", RF), ("3_year", YF)):
        x = g[g.block == b]
        if b != "3_year":
            x = x[x.real_ep]
        bl = []; ref = 0
        for r in x.itertuples():
            a, z = pd.to_datetime(r.t_ms, unit="ms"), pd.to_datetime(r.x_ms, unit="ms")
            f = src[src.pair == r.pair]
            bl.append(f[(f.opened > a) & (f.opened < z)].pnl_percentage)
            ref += int(((f.opened <= a) & (f.closed > a)).any())
        bb = pd.concat(bl) if bl else pd.Series(dtype=float)
        P(f"| {b} | {len(x)} | {len(bb)} | {bb.mean() if len(bb) else float('nan'):+.2f} | {bb.sum():+.1f} | {ref} |")


def sizing(cell, acct=2600.0, max_loss=0.05, max_open=3):
    g = X[(X.cell == cell)].sort_values("t_ms")
    loss99 = -np.percentile(g.net, 1); worst = -g.net.min()
    notional = acct * max_loss / max(loss99 / 100, 1e-9)
    P(f"\n#### sizing — {cell} at the ${acct:,.0f} account\n")
    P(f"loss at the 1st percentile {-loss99:+.1f} % · worst {-worst:+.1f} % → notional ${notional:,.0f} (1× of {notional / acct:.2f} the account) "
      f"so a 1st-percentile loss = 5 % of the account; worst trade then ${-worst / 100 * notional:,.0f}")
    for b in ("3_year", "all"):
        x = g if b == "all" else g[g.block == b]
        openl = []; taken = []
        for r in x.itertuples():
            openl = [z for z in openl if z > r.t_ms]
            if len(openl) >= max_open:
                continue
            openl.append(r.x_ms); taken.append(r)
        tk = pd.DataFrame(taken)
        pnl = tk.net / 100 * notional
        eq = pnl.cumsum(); dd = (eq - eq.cummax()).min()
        day = pnl.groupby(tk.day).sum()
        P(f"- {b}: taken {len(tk)} of {len(x)} (≤ {max_open} open) · Σ ${pnl.sum():+,.0f} · worst trade ${pnl.min():+,.0f} · worst day ${day.min():+,.0f} "
          f"({day.idxmin()}) · max drawdown ${dd:+,.0f} ({dd / acct * 100:+.1f} % of the account) · days {len(day)}")


def episodes_real():
    cells = [PRI, SEC, best]
    g = X[X.cell.isin(cells) & (X.block != "3_year")]
    eps_ids = sorted(set(g.ep))
    rows = []
    for k in eps_ids:
        e = eps[k]; on, off = e["on_i"], e["off_i"]
        d = dict(pair=e["pair"], spike=pd.to_datetime(e["spike_ts"], unit="ms"), off=pd.to_datetime(e["t"][min(off, len(e["t"]) - 1)] + S.BAR, unit="ms") if off < len(e["t"]) else "still flagged",
                 real=bool(g[g.ep == k].real_ep.iloc[0]), run_pct=(np.nanmax(e["h"][on:off]) / e["base"] - 1) * 100,
                 last_vs_vwap=(e["c"][min(off, len(e["c"]) - 1)] / e["vwap"][min(off - 1, len(e["c"]) - 1)] - 1) * 100)
        for c in cells:
            x = g[(g.ep == k) & (g.cell == c)]
            d[c] = f"{x.net.iloc[0]:+.1f} ({x.reason.iloc[0]}, {x.entry_dt.iloc[0]:%m-%d %H:%M})" if len(x) else ""
        rows.append(d)
    T = pd.DataFrame(rows).sort_values("spike")
    T.to_csv(OUT + "_episodes_real.csv", index=False)
    P("\n#### per-episode (blocks ① ②, walk episodes; real = bot FRENZY evidence)\n")
    P(md(T, '+.1f', index=False))


def today7():
    P("\n#### today's dashboard pairs (latest episode per pair, data to the last closed bar)\n")
    P("| pair | spike (UTC) | as of | hours | flagged then | last close vs VWAP | closes below VWAP in a row | signals fired (net % so far / final) |")
    P("|---|---|---|---|---|---|---|---|")
    for p in S.TODAY7:
        es = [(k, e) for k, e in enumerate(eps) if e["pair"] == p]
        if not es:
            P(f"| {p} | no verified shortlisted episode in the walk | | | | | |"); continue
        k, e = max(es, key=lambda x: x[1]["spike_ts"])
        last = len(e["t"]) - 1; on = e["on_i"]
        fl = bool(e["flag"][last]) if e["off_i"] > last else False
        vs = (e["c"][last] / e["vwap"][last] - 1) * 100 if not np.isnan(e["vwap"][last]) else np.nan
        bs = int(e["bstreak"][last])
        x = X[(X.ep == k) & (X.hold == 24) & (X.stop == 10.0) & (X.tp == "none")]
        sg = "; ".join(f"{r.entry} {r.entry_dt:%m-%d %H:%M} → {r.net:+.1f} ({r.reason})" for r in x.sort_values("t_ms").itertuples()) or "none"
        P(f"| {p} | {pd.to_datetime(e['spike_ts'], unit='ms'):%m-%d %H:%M} | {pd.to_datetime(e['t'][last] + S.BAR, unit='ms'):%m-%d %H:%M} | {(e['t'][last] - e['t'][on]) / 3.6e6:.0f} | {'yes' if fl else 'no'} | {vs:+.1f} % | {bs} | {sg} |")


if __name__ == "__main__":
    P(f"best (year, N ≥ 100, highest lower CI): {best}")
    P("\n### cell leaderboard (year block, N ≥ 100, by lower CI)\n")
    cols = ["N", "WR", "avg", "lo", "hi", "p2025", "p26H1", "p26H2", "lomo_min", "ex_top5", "top_pair_share", "top_day_share", "real_nofund", "null_mean", "p_own", "p_fw"]
    P(md(CS[CS.N >= 100].sort_values("lo", ascending=False)[cols].head(25)))
    P("\n### per entry, its best exit cell by lower CI\n")
    cs = CS.copy(); cs["entry"] = [c.split("|")[0] for c in cs.index]
    P(md(cs.sort_values("lo", ascending=False).groupby("entry").head(1)[cols]))
    P("\n### primary / secondary rows\n")
    P(md(CS.loc[[PRI, SEC]][cols]))
    for c in dict.fromkeys([PRI, SEC, best]):
        three_blocks(c)
    for c in dict.fromkeys([PRI, best]):
        robustness(c); tails(c); interactions(c); sizing(c)
    episodes_real()
    today7()
    keep = X[X.cell.isin([PRI, SEC, best])].drop(columns=["row"])
    keep.to_csv(OUT + "_trades.csv", index=False)
