#!/usr/bin/env python3
"""FRENZY LONG SEARCH — rank every candidate by expected DAILY compound return (operator: many trades/day, profitable):
trades/day and coins/day over every calendar day of the data span, avg net, holdout avgs, Kelly at the one-sided 95 % Wilson WR floor,
one-size DCR (0.18 × equity notional per trade, P&L booked on the exit day, every calendar day, 0 on no-fill days), the empirical
log-optimal size (max mean log(1 + f·r), descriptive — fat tails), max open positions, max drawdown. Writes _rank.csv."""
import os, pickle, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_eval as EV, frenzy_long_search_flag_eval as FE   # noqa: E402
D = EV.D; OUT = EV.OUT; DAY = EV.DAY
T = pd.read_csv(os.path.join(D, "tape_daily.csv"), index_col=0); A = pd.read_csv(OUT + "_analogs.csv", index_col=0)


def dcr(d, x, f, d0, d1):
    days = np.arange(d0, d1 + 1)
    xd = (d[EXIT + "_x"].values // DAY)
    pnl = pd.Series(x * f / 100).groupby(xd).sum().reindex(days).fillna(0).values
    pnl = np.maximum(pnl, -0.99)
    eq = np.cumprod(1 + pnl)
    dd = (eq / np.maximum.accumulate(eq) - 1).min() * 100
    return (eq[-1] ** (1 / len(days)) - 1) * 100, dd


def maxopen(d):
    ev = np.r_[np.c_[d.entry_ms.values, np.ones(len(d))], np.c_[d[EXIT + "_x"].values, -np.ones(len(d))]]
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    return int(np.cumsum(ev[:, 1]).max()) if len(ev) else 0


def row(name, d, x, preg):
    global EXIT
    s = EV.stats(x, d)
    d0, d1 = int(d.day.min()), int(d.day.max())
    ncal = d1 - d0 + 1
    o = dict(candidate=name, prereg=preg, N=s["N"], trades_day=s["N"] / ncal, coins_day=d.groupby("day").pair.nunique().sum() / ncal,
             avg=s["avg"], med=s["med"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], best5_out=s["best5_out"], kelly_floor=s["kelly_floor"])
    for b in ("DESIGN", "HOLD-A", "HOLD-B"):
        m = (d.block == b).values
        o[b] = x[m].mean() if m.any() else np.nan
    o["dcr_018"], o["dd_018"] = dcr(d, x, 0.18, d0, d1)
    fk = max(0.0, s["kelly_floor"]) if s["kelly_floor"] == s["kelly_floor"] else 0.0
    o["dcr_kelly_floor"] = dcr(d, x, fk, d0, d1)[0] if fk > 0 else 0.0
    grid = np.linspace(0.005, 0.5, 100)
    lg = [np.mean(np.log1p(np.maximum(f * x / 100, -0.999))) for f in grid]
    o["logopt_f"] = grid[int(np.argmax(lg))] if max(lg) > 0 else 0.0
    o["max_open"] = maxopen(d)
    # realistic book: at most K = 10 open positions (first come), then the same reads
    dd_ = d.assign(_x=x).sort_values("entry_ms"); ox = []; keep = []
    for e_, x_ in zip(dd_.entry_ms.values, dd_[EXIT + "_x"].values):
        ox = [y for y in ox if y > e_]
        if len(ox) < 10:
            ox.append(x_); keep.append(True)
        else:
            keep.append(False)
    b = dd_[np.array(keep)]; xb = b._x.values
    sb = EV.stats(xb, b)
    o["K10_N"], o["K10_trades_day"], o["K10_avg"], o["K10_best5_out"], o["K10_kelly_floor"] = sb["N"], sb["N"] / ncal, sb["avg"], sb["best5_out"], sb["kelly_floor"]
    for b_ in ("DESIGN", "HOLD-A", "HOLD-B"):
        m = (b.block == b_).values
        o["K10_" + b_] = xb[m].mean() if m.any() else np.nan
    o["K10_dcr_018"], o["K10_dd_018"] = dcr(b, xb, 0.18, d0, d1)
    fk = max(0.0, sb["kelly_floor"]) if sb["kelly_floor"] == sb["kelly_floor"] else 0.0
    o["K10_dcr_kelly_floor"] = dcr(b, xb, fk, d0, d1)[0] if fk > 0 else 0.0
    return o


def main():
    global EXIT
    out = []
    E = pickle.load(open(os.path.join(D, "eval.pkl"), "rb"))
    for (fam, ck, ex), d in E["store"].items():
        EXIT = ex
        if fam == "BASE":
            continue
        out.append(row(f"{fam}_{ck}h · {ex}", d, d.net.values, "primary"))
    df, _ = EV.load()
    for fam, ck, ex in (("BASE", 24, "H168_Sx_T20"), ("BASE", 6, "H72_Sx_Tx"), ("BASE", 24, "H72_Sx_Tx"), ("SURVIVOR", 24, "H168_Sx_T20"), ("CARRY", 48, "H24_S10_Tx")):
        EXIT = ex
        d = EV.dedupe(df[(df.ck == ck) & df[fam] & df[ex + "_complete"]], ex)
        out.append(row(f"{fam}_{ck}h · {ex}", d, EV.net(d, ex), "post-hoc (exit picked after the grid read)"))
    F = FE.tag(pickle.load(open(os.path.join(D, "flag_rows.pkl"), "rb")), T, A)
    for ex in ("TR", "Q3", "Q5", "T4", "H72"):
        EXIT = ex
        d = EV.dedupe(F[F[ex + "_complete"]], ex)
        out.append(row(f"FLAG · {ex}", d, FE.netf(d, ex), "addendum A primary" if ex == "TR" else "addendum A"))
    S = FE.tag(pickle.load(open(os.path.join(D, "fast5_rows.pkl"), "rb")), T, A)
    for det in ("D3_5m", "D5_5m"):
        for ex in ("TR", "Q5", "T4", "H72"):
            EXIT = ex
            d = EV.dedupe(S[(S.det == det) & S[ex + "_complete"]], ex)
            out.append(row(f"FAST5 {det} · {ex}", d, FE.netf(d, ex), "addendum A robustness"))
    Rk = pd.DataFrame(out).sort_values(["K10_dcr_kelly_floor", "K10_dcr_018"], ascending=False)
    Rk.to_csv(OUT + "_rank.csv", index=False)
    pd.set_option("display.width", 300)
    print(Rk[["candidate", "N", "trades_day", "avg", "ci_lo", "kelly_floor", "K10_trades_day", "K10_avg", "K10_DESIGN", "K10_HOLD-A", "K10_HOLD-B", "K10_best5_out", "K10_kelly_floor", "K10_dcr_018", "K10_dd_018", "K10_dcr_kelly_floor"]].round(3).to_string())


if __name__ == "__main__":
    main()
