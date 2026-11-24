#!/usr/bin/env python3
"""TP_RUNNER_2026-11-24 — before / after per sleeve (current batch on today's rules · master real fills · backtest DESIGN / HOLDOUT),
one size X = 0.61 × $10k, runner share p ∈ {50 %, 100 %}, each sleeve at its DESIGN-picked runner (and EXH also at the plateau pick)."""
import os, sys, importlib, glob, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
T = importlib.import_module("TP_HOLD_2026-11-24"); RU = importlib.import_module("TP_RUNNER_2026-11-24"); C = T.C
OUT = T.OUT; X, ACC = 0.61, 10_000
PICK = {"EXHAUSTION_SHORT": "none|5", "DISTRIBUTION_SHORT": "none|8", "FLATBASE_SHORT": "halfTP|5", "FRENZY_WILLY_HT": "none|3"}
SH = {"EXH": "EXHAUSTION_SHORT", "DIST": "DISTRIBUTION_SHORT", "FB": "FLATBASE_SHORT", "HT": "FRENZY_WILLY_HT"}
MS = lambda s: int((pd.to_datetime(s, utc=True, format="mixed") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1))


def dcr(df, pct_col, days):
    r = df.groupby("day")[pct_col].sum() * X / 100
    r = r.reindex(days, fill_value=0.0)
    return (np.prod(1 + r.values) ** (1 / len(days)) - 1) * 100


def stats(df, col, days):
    return dict(N=len(df), WR=100 * (df[col] > 0).mean() if len(df) else np.nan, avg=df[col].mean() if len(df) else np.nan,
                usd=(df[col] * X * ACC / 100).sum(), dcr=dcr(df, col, days) if len(df) else 0.0)


def line(name, b, a1, a2):
    f = lambda s: f"{s['N']:3d} · {s['WR']:3.0f}% · {s['avg']:+.2f} · ${s['usd']:+,.0f} · {s['dcr']:+.2f}%/d"
    print(f"  {name:20s} BEFORE {f(b)}  |  runner 50% {f(a1)}  |  runner 100% {f(a2)}")


def batch():
    F = pd.read_csv(os.path.join(ROOT, "reports", "BATCH_TODAY_RULES_2026-11-23_fills.csv"))
    F = F[F.kept.astype(bool)].copy()
    ex = pd.read_csv(os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-23_22-20-57.csv"), low_memory=False)
    ex["om"] = ex.opened_at.astype(str).str[:16]
    F["om"] = F.opened_at.astype(str).str[:16]
    F = F.merge(ex[["om", "pair", "entry_price", "exit_price", "direction"]].drop_duplicates(["om", "pair"]), on=["om", "pair"], how="left")
    F["day"] = pd.to_datetime(F.today_closed_at, utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    days = pd.date_range("2026-11-18", "2026-11-23").strftime("%Y-%m-%d")
    for p in (0.5, 1.0):
        F[f"after{p}"] = F.today_pct
    F["delta"] = 0.0
    for i, r in F.iterrows():
        if str(r.today_reason) not in ("TP", "FRENZY_TP") or not np.isfinite(r.entry_price):
            continue
        s = SH.get(r.sleeve)
        if s is None: continue
        d = -1 if str(r.direction).upper() == "SHORT" else 1
        E = float(r.entry_price)
        if np.isfinite(r.today_tp):
            tp = E * (1 + d * float(r.today_tp) / 100); tt = MS(r.today_closed_at)
        else:                                              # booked TP kept (exit not re-walked): the real TP fill
            tp = float(r.exit_price); tt = MS(r.closed_at)
        o = RU.outcomes(r.pair, tt, E, tp, d, T.atr15_pct(r.pair, tt // 60000 * 60000))
        if o is None: continue
        F.at[i, "delta"] = o[PICK[s]]
    for p in (0.5, 1.0):
        F[f"after{p}"] = F.today_pct + p * F.delta
    print("\n=== CURRENT BATCH (11-18 → 11-23, today's filters + exits, one size) ===")
    for s, g in F.groupby("sleeve"):
        line(f"{s} [{PICK.get(SH.get(s), '-')}]", stats(g, "today_pct", days), stats(g, "after0.5", days), stats(g, "after1.0", days))
    line("ALL", stats(F, "today_pct", days), stats(F, "after0.5", days), stats(F, "after1.0", days))
    print("  TP hits re-walked:", F[F.delta != 0].groupby("sleeve").delta.agg(["size", "mean"]).round(2).to_dict("index"))


def master():
    P = []
    for f in sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv"))):
        h = pd.read_csv(f, nrows=0).columns
        if "entry_strategy" not in h: continue
        x = pd.read_csv(f, low_memory=False)
        x = x[x.entry_strategy.isin(list(PICK)) & (x.status == "CLOSED")]
        if len(x): x["src"] = os.path.basename(f); P.append(x)
    M = pd.concat(P, ignore_index=True)
    if "is_probe" in M: M = M[~M.is_probe.fillna(False).astype(str).str.lower().isin(["true", "1"])]
    M["k"] = M.opened_at.astype(str).str[:19]
    M = M.sort_values("src").drop_duplicates(["k", "pair", "direction"], keep="last")
    M, _ = T.GK.split_operator_closed(M)
    M["pct"] = M.pnl / (M.quantity * M.entry_price) * 100
    M["t_touch"] = M.closed_at.map(MS)
    R = pd.read_csv(os.path.join(OUT, "runner_real.csv"))
    M = M.merge(R[["pair", "t_touch"] + list(set(PICK.values()))], on=["pair", "t_touch"], how="left")
    M["day"] = pd.to_datetime(M.closed_at, utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    print("\n=== MASTER (every real fill, as booked, one size; each sleeve over its own live days) ===")
    for s, g in M.groupby("entry_strategy"):
        dl = g.assign(dl=g.apply(lambda r: r[PICK[s]] if pd.notna(r.get(PICK[s])) else 0.0, axis=1)).dl
        g = g.assign(a5=g.pct + 0.5 * dl, a10=g.pct + dl)
        days = pd.date_range(g.day.min(), g.day.max()).strftime("%Y-%m-%d")
        line(f"{s[:16]} [{PICK[s]}]", stats(g, "pct", days), stats(g, "a5", days), stats(g, "a10", days))
        print(f"      TP hits with a runner read: {int(dl.ne(0).sum())} · avg runner Δ {dl[dl.ne(0)].mean():+.2f} pp")


def backtest():
    f = pd.read_csv(os.path.join(ROOT, "reports", "STATE_EXIT_2026-11-21_data", "fills_out.csv"), low_memory=False)
    f = f[f.kind == "BT"]
    base = {}
    for s, col in (("EXH", "tp7"), ("FB", "base_net"), ("HT", "tp6")):
        for per, g in f[f.sleeve == s].groupby("period"):
            base[(SH[s], per)] = (len(g), g[col].mean(), 100 * (g[col] > 0).mean())
    d = pd.read_csv(os.path.join(ROOT, "reports", "DIST_EXIT_2026-11-18_trades.csv.gz")); d = d[d.cell == "REF_B"]
    for per, g in d.groupby("block"):
        base[("DISTRIBUTION_SHORT", per)] = (len(g), g.net.mean(), 100 * (g.net > 0).mean())
    R = pd.read_csv(os.path.join(OUT, "runner_bt.csv"))
    print("\n=== BACKTEST (per fill, all fills; after = before + p × Σ runner Δ on the TP hits ÷ N) ===")
    for s in PICK:
        for per in ("DESIGN", "HOLDOUT"):
            if (s, per) not in base: continue
            n, b, wr = base[(s, per)]
            g = R[(R.sleeve == s) & (R.period == per)]
            add = g[PICK[s]].sum() / n
            print(f"  {s[:16]:16s} {per:7s} [{PICK[s]}] N {n:4d} · WR {wr:3.0f}% · BEFORE {b:+.2f}%/fill → runner 50% {b + 0.5 * add:+.2f} · runner 100% {b + add:+.2f}   ({len(g)} TP hits, runner Δ {g[PICK[s]].mean():+.2f} pp each)")


if __name__ == "__main__":
    batch(); master(); backtest()
