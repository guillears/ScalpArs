#!/usr/bin/env python3
"""Route B report 2026-11-21 — step 2: per-fill recorded state at entry and during the trade (DESCRIPTIVE ONLY, N tiny).

Inputs: reports/ROUTE_B_REPORT_2026-11-21_data/{features_all.pkl.gz, events.json} (from ROUTE_B_REPORT_2026-11-21_load.py, S3 read-only)
        + exports_archive/scalpars_orders_paper_*.csv (dedup key opened_at[:19], pair, direction; CLOSED) + the bot list (open positions).
Outputs: reports/ROUTE_B_REPORT_2026-11-21_data/{fills.csv, coverage.csv, inventory.json} + reports/ROUTE_B_REPORT_2026-11-21_chart.png
Usage: venv/bin/python -I scripts/ROUTE_B_REPORT_2026-11-21.py"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "ROUTE_B_REPORT_2026-11-21_data")
REC_START = pd.Timestamp("2026-11-19 23:23:30", tz="UTC")
MIN = 60_000


def load_features():
    F = pd.read_pickle(os.path.join(D, "features_all.pkl.gz"))
    book_cols = ["n_book", "mid", "mid_open", "mid_hi", "mid_lo", "spread_bps", "bid_usd_02", "ask_usd_02", "bid_usd_05", "ask_usd_05",
                 "bid_usd_1", "ask_usd_1", "imb_05", "reach_bid_pct", "reach_ask_pct", "cost_buy_2500", "cost_sell_2500",
                 "cost_buy_10000", "cost_sell_10000"]
    F = F.sort_values(["sym", "minute", "n_book"])
    g = F.groupby(["sym", "minute"], sort=False)
    book = g[book_cols].last()                                      # fuller book row (sorted by n_book)
    sums = g[["n_trades", "buy_usd", "sell_usd", "liq_long_usd", "liq_short_usd"]].sum(min_count=1)
    mx = g[["max_trade_usd"]].max()
    mk = g[["mark", "index", "funding"]].last()                      # last non-null
    out = pd.concat([book, sums, mx, mk], axis=1).reset_index()
    out["t"] = pd.to_datetime(out.minute, unit="ms", utc=True)
    return out


def load_fills(events):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d["src"] = os.path.basename(f)
        rows.append(d)
    a = pd.concat(rows, ignore_index=True)
    a = a[a.status == "CLOSED"].copy()
    a["k"] = a.opened_at.str[:19] + "|" + a.pair + "|" + a.direction
    a = a.sort_values("src").drop_duplicates("k", keep="last")
    a["open_t"] = pd.to_datetime(a.opened_at.str[:19], utc=True)
    a["close_t"] = pd.to_datetime(a.closed_at.str[:19], utc=True)
    keep = a[(a.close_t >= REC_START)].copy()
    keep["state"] = "CLOSED"
    # open positions from the bot's last list (not in any export yet)
    last = [e for e in events if e.get("ev") == "bot"][-1]
    op = []
    for pair, v in (last.get("open") or {}).items():
        k = v["o"][:19] + "|" + pair + "|" + v["d"]
        if k in set(keep.k):
            continue
        op.append(dict(pair=pair, direction=v["d"], entry_strategy=v["s"], opened_at=v["o"], open_t=pd.Timestamp(v["o"], tz="UTC"),
                       close_t=pd.Timestamp(last["r"], unit="ms", tz="UTC"), state="OPEN", k=k))
    return pd.concat([keep, pd.DataFrame(op)], ignore_index=True), last


def px(c, t0, t1):
    """mark-price path rows [t0, t1] for coin c (every coin has the 1 s mark stream)."""
    return c[(c.t >= t0) & (c.t <= t1)]


def per_fill(F, fills):
    out = []
    by = {s: g.sort_values("minute") for s, g in F[F.sym.isin(set(fills.pair))].groupby("sym")}
    for r in fills.itertuples():
        c = by.get(r.pair)
        sgn = -1 if r.direction == "SHORT" else 1
        E = r.open_t.floor("min")
        X = r.close_t.floor("min")
        row = dict(pair=r.pair, sleeve=r.entry_strategy, direction=r.direction, state=r.state, opened=str(r.open_t)[:16],
                   closed=str(r.close_t)[:16] if r.state == "CLOSED" else "open", hold_min=round((r.close_t - r.open_t).total_seconds() / 60),
                   pnl_pct=getattr(r, "pnl_percentage", np.nan), pnl_usd=getattr(r, "pnl", np.nan), peak_pct=getattr(r, "peak_pnl", np.nan),
                   close_reason=getattr(r, "close_reason", "OPEN"), funding_fee_usd=getattr(r, "funding_fee_usd", np.nan),
                   entry_price=getattr(r, "entry_price", np.nan),
                   notional=(getattr(r, "investment", np.nan) or np.nan) * (getattr(r, "leverage", np.nan) or np.nan),
                   paper_entry_slip=getattr(r, "entry_slippage_pct", np.nan), paper_exit_slip=getattr(r, "exit_slippage_pct", np.nan))
        if c is None:
            row.update(rec_entry=False, rec_through=0.0)
            out.append(row)
            continue
        cb = c[c.n_book > 0]
        span = pd.date_range(E, X, freq="min")
        row["rec_entry"] = bool(((cb.t >= E - pd.Timedelta(minutes=1)) & (cb.t <= E + pd.Timedelta(minutes=1))).any())
        row["rec_through"] = round(cb.t.isin(span).sum() / max(len(span), 1), 3)
        pre = c[(c.t >= E - pd.Timedelta(minutes=15)) & (c.t < E)]
        pre5 = c[(c.t >= E - pd.Timedelta(minutes=5)) & (c.t < E)]
        b1 = cb[(cb.t < E) & (cb.t >= E - pd.Timedelta(minutes=3))]
        b1 = b1.iloc[-1] if len(b1) else None
        adv_liq = "liq_short_usd" if sgn < 0 else "liq_long_usd"           # the side a move AGAINST us force-closes
        fav_liq = "liq_long_usd" if sgn < 0 else "liq_short_usd"
        adv_depth = "ask_usd_05" if sgn < 0 else "bid_usd_05"              # wall a move against us must eat (short: asks above)
        our_depth = "bid_usd_05" if sgn < 0 else "ask_usd_05"
        med15 = (cb.buy_usd + cb.sell_usd).median() * 15
        if b1 is not None:
            n = row["notional"] if row["notional"] == row["notional"] else 6000
            ein, eout = ("sell", "buy") if sgn < 0 else ("buy", "sell")
            interp = lambda side, rr: rr[f"cost_{side}_2500"] + (rr[f"cost_{side}_10000"] - rr[f"cost_{side}_2500"]) * (min(n, 10000) - 2500) / 7500
            row.update(spread_bps=b1.spread_bps, bid02=b1.bid_usd_02, ask02=b1.ask_usd_02, bid05=b1.bid_usd_05, ask05=b1.ask_usd_05,
                       bid1=b1.bid_usd_1, ask1=b1.ask_usd_1, imb05=b1.imb_05, cost_in_bps=interp(ein, b1),
                       adv_depth_vs_coin_med=b1[adv_depth] / cb[adv_depth].median())
        bs, ss = pre.buy_usd.sum(), pre.sell_usd.sum()
        bs5, ss5 = pre5.buy_usd.sum(), pre5.sell_usd.sum()
        row.update(taker_buy_share_15=bs / (bs + ss) if bs + ss else np.nan, taker_buy_share_5=bs5 / (bs5 + ss5) if bs5 + ss5 else np.nan,
                   flow15_vs_coin_med=(bs + ss) / med15 if med15 else np.nan,
                   liq_long_15=pre.liq_long_usd.sum(), liq_short_15=pre.liq_short_usd.sum())
        m = c[(c.t <= E)].dropna(subset=["mark"])
        if len(m):
            row["funding_pct_entry"] = m.funding.dropna().iloc[-1] * 100 if m.funding.notna().any() else np.nan
            p0 = m.mark.iloc[-1]
            for h in (15, 60):
                mh = m[m.t <= E - pd.Timedelta(minutes=h)]
                row[f"move_{h}m_pct"] = (p0 / mh.mark.iloc[-1] - 1) * 100 if len(mh) else np.nan
        # ---- during the trade
        tr = c[(c.t >= E) & (c.t <= X)].copy()
        if len(tr) and row["entry_price"] == row["entry_price"]:
            ep = row["entry_price"]
            hi = tr.mid_hi.fillna(tr.mark)
            lo = tr.mid_lo.fillna(tr.mark)
            fav = ((lo if sgn < 0 else hi) / ep - 1) * 100 * sgn
            adv = ((hi if sgn < 0 else lo) / ep - 1) * 100 * sgn
            ipk = fav.values.argmax()
            row.update(mfe_rec=fav.max(), mae_rec=adv.min(), peak_min=int((tr.t.iloc[ipk] - E).total_seconds() // 60))
            row["adv_liq_trade"] = tr[adv_liq].sum()
            row["fav_liq_trade"] = tr[fav_liq].sum()
            j = tr[adv_liq].values.argmax()
            row["adv_liq_max1m"] = tr[adv_liq].iloc[j]
            row["adv_liq_max_min_before_exit"] = int((X - tr.t.iloc[j]).total_seconds() // 60)
            row["adv_liq_coin_p99"] = c[adv_liq][c[adv_liq] > 0].quantile(0.99) if (c[adv_liq] > 0).sum() > 20 else np.nan
            trb = tr[tr.n_book > 0]
            if len(trb) and b1 is not None:
                last10 = trb[trb.t >= X - pd.Timedelta(minutes=10)]
                row["adv_depth_min_vs_entry"] = trb[adv_depth].min() / b1[adv_depth]
                row["adv_depth_last10_vs_entry"] = last10[adv_depth].median() / b1[adv_depth] if len(last10) else np.nan
                row["our_depth_last10_vs_entry"] = last10[our_depth].median() / b1[our_depth] if len(last10) else np.nan
            l15 = tr[tr.t >= X - pd.Timedelta(minutes=15)]
            b, s = l15.buy_usd.sum(), l15.sell_usd.sum()
            row["taker_buy_share_last15"] = b / (b + s) if b + s else np.nan
            # after the peak: 15 min of flow / liquidations
            pk_t = tr.t.iloc[ipk]
            ap = tr[(tr.t > pk_t) & (tr.t <= pk_t + pd.Timedelta(minutes=15))]
            b, s = ap.buy_usd.sum(), ap.sell_usd.sum()
            row.update(post_peak_taker_buy_share=b / (b + s) if b + s else np.nan, post_peak_adv_liq=ap[adv_liq].sum(),
                       post_peak_fav_liq=ap[fav_liq].sum())
            xb = cb[(cb.t >= X - pd.Timedelta(minutes=1)) & (cb.t <= X)]
            if len(xb) and row["notional"] == row["notional"]:
                xr = xb.iloc[-1]
                n = row["notional"]
                side = "buy" if sgn < 0 else "sell"
                row["cost_exit_bps"] = xr[f"cost_{side}_2500"] + (xr[f"cost_{side}_10000"] - xr[f"cost_{side}_2500"]) * (min(n, 10000) - 2500) / 7500
            fm = tr.funding.dropna()
            if len(fm):
                row["funding_pct_min_trade"] = fm.min() * 100
                row["funding_pct_max_trade"] = fm.max() * 100
        out.append(row)
    return pd.DataFrame(out)


def inventory(F, events):
    hb = F[F.n_book > 0]
    ev = [e for e in events if e.get("ev") in ("start", "stop", "disconnect", "reconnect")]
    return dict(first_minute=str(F.t.min()), last_minute=str(F.t.max()), hours=round((F.t.max() - F.t.min()).total_seconds() / 3600, 1),
                coins_mark=int(F.sym.nunique()), coins_book=int(hb.sym.nunique()), coin_minutes_book=int(len(hb)),
                book_snapshots=int(hb.n_book.sum()), trades=int(F.n_trades.sum()), liq_long_usd=float(F.liq_long_usd.sum()),
                liq_short_usd=float(F.liq_short_usd.sum()), n_gap_events=len(ev),
                gap_s=sum(float(e.get("gap_s") or 0) for e in ev if e["ev"] == "reconnect"))


def chart(F, fills, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [("RAREUSDT", "2026-11-20T20:00", "RARE · EXHAUSTION short 11-20 20:00 (+7.9 → stop −7.4)"),
             ("KERNELUSDT", "2026-11-20T20:00", "KERNEL · FLATBASE short 11-20 20:00 (stop −11.7)"),
             ("ENSOUSDT", "2026-11-21T02:30", "ENSO · WILLY_HT long 11-21 02:30 (+4.9 → stop −3.9)")]
    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
    fig, ax = plt.subplots(4, 3, figsize=(16, 12), sharex="col")
    for j, (pair, o, title) in enumerate(picks):
        f = fills[(fills.pair == pair) & (fills.opened_at.str[:16] == o)].iloc[0]
        sgn = -1 if f.direction == "SHORT" else 1
        E, X = f.open_t.floor("min"), f.close_t.floor("min")
        c = F[(F.sym == pair) & (F.t >= E - pd.Timedelta(minutes=60)) & (F.t <= X + pd.Timedelta(minutes=30))].sort_values("t")
        x = (c.t - E).dt.total_seconds() / 60
        price = c.mid.fillna(c.mark)
        ax[0, j].plot(x, (price / f.entry_price - 1) * 100 * sgn, color=BLUE, lw=1.6)
        ax[0, j].axhline(0, color=GREY, lw=0.8)
        ax[0, j].set_title(title, fontsize=10)
        ax[0, j].set_ylabel("P&L % in trade direction" if j == 0 else "")
        adv = "liq_short_usd" if sgn < 0 else "liq_long_usd"
        ax[1, j].bar(x, c[adv] / 1e3, width=0.9, color=ORANGE)
        ax[1, j].set_ylabel("liquidations AGAINST us\n$k per minute" if j == 0 else "")
        roll_b = c.buy_usd.rolling(15, min_periods=5).sum()
        roll_s = c.sell_usd.rolling(15, min_periods=5).sum()
        adv_share = (roll_b if sgn < 0 else roll_s) / (roll_b + roll_s)
        ax[2, j].plot(x, adv_share * 100, color=AQUA, lw=1.6)
        ax[2, j].axhline(50, color=GREY, lw=0.8)
        ax[2, j].set_ylabel("taker flow AGAINST us\n% of 15-min $" if j == 0 else "")
        dep = c["ask_usd_05" if sgn < 0 else "bid_usd_05"].rolling(5, min_periods=1).median()
        ax[3, j].plot(x, dep / 1e3, color=BLUE, lw=1.6)
        ax[3, j].set_ylabel("book wall AGAINST us\n$k within 0.5 % (5-min med)" if j == 0 else "")
        ax[3, j].set_xlabel("minutes from entry")
        for i in range(4):
            ax[i, j].axvline(0, color=GREY, lw=0.8, ls="--")
            ax[i, j].axvline((X - E).total_seconds() / 60, color=ORANGE, lw=0.8, ls="--")
            for s in ("top", "right"):
                ax[i, j].spines[s].set_visible(False)
            ax[i, j].grid(axis="y", color="#e8e8e6", lw=0.6)
    fig.suptitle("Route B event study — three operator cases (grey dashed = entry, orange dashed = exit). DESCRIPTIVE, N = 3.\nPrice = minute close (intraminute peaks not shown); depth = top-20 book levels, a lower bound where they reach < 0.5 % (KERNEL, ENSO).", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    events = json.load(open(os.path.join(D, "events.json")))
    F = load_features()
    fills, last = load_fills(events)
    T = per_fill(F, fills)
    T.to_csv(os.path.join(D, "fills.csv"), index=False)
    inv = inventory(F, events)
    json.dump(inv, open(os.path.join(D, "inventory.json"), "w"), indent=1)
    chart(F, fills, os.path.join(ROOT, "reports", "ROUTE_B_REPORT_2026-11-21_chart.png"))
    pd.set_option("display.width", 300, "display.max_columns", 80)
    print(json.dumps(inv, indent=1))
    print(T.T.to_string())


if __name__ == "__main__" and not os.environ.get("STEP3"):
    main()


# ---------------------------------------------------------------- step 3: data-quality reach + new-sleeve trigger COUNTS (no outcomes)
def usdt_only(F):
    return F[F.sym.str.endswith("USDT") & ~F.sym.str.contains("_")]


def reach_stats(F, pairs):
    b = F[(F.n_book > 0) & F.sym.isin(pairs)]
    g = b.groupby("sym").agg(reach_bid=("reach_bid_pct", "median"), reach_ask=("reach_ask_pct", "median"),
                             cost10_nan=("cost_buy_10000", lambda s: s.isna().mean()), spread=("spread_bps", "median"),
                             bid05=("bid_usd_05", "median"), ask05=("ask_usd_05", "median"))
    return g.round(3)


def episodes(mask_df, gap_min=60):
    """one possibility per coin per `gap_min` minutes (the first trigger minute opens it); returns rows + distinct 15-min windows."""
    out = []
    for s, g in mask_df.sort_values("minute").groupby("sym"):
        last = -10**18
        for m in g.minute:
            if m - last >= gap_min * MIN:
                out.append((s, m))
                last = m
    e = pd.DataFrame(out, columns=["sym", "minute"])
    e["win15"] = e.minute // (15 * MIN)
    return e


def trigger_counts(F):
    U = usdt_only(F)
    U = U[~U.sym.isin(["BTCUSDT", "ETHUSDT"])].sort_values(["sym", "minute"]).copy()
    hb = U[U.n_book > 0].copy()
    res = {}
    for side in ("liq_long_usd", "liq_short_usd"):
        r5 = hb.groupby("sym")[side].transform(lambda s: s.rolling(5, min_periods=1).sum())
        thr = hb.assign(r5=r5).groupby("sym").r5.transform(lambda s: s[s > 0].quantile(0.99) if (s > 0).sum() > 20 else np.inf)
        res[f"cascade_{side}"] = episodes(hb[(r5 >= thr) & (r5 >= 50_000)])
    imb3 = hb.groupby("sym").imb_05.transform(lambda s: s.rolling(3, min_periods=3).min())
    thin = (hb.bid_usd_05 + hb.ask_usd_05) < 40_000
    res["imbalance_bid_heavy_thin"] = episodes(hb[(imb3 >= 0.5) & thin])
    imb3n = hb.groupby("sym").imb_05.transform(lambda s: s.rolling(3, min_periods=3).max())
    res["imbalance_ask_heavy_thin"] = episodes(hb[(imb3n <= -0.5) & thin])
    med_ask = hb.groupby("sym").ask_usd_05.transform("median")
    med_bid = hb.groupby("sym").bid_usd_05.transform("median")
    res["ask_vacuum"] = episodes(hb[hb.ask_usd_05 < 0.35 * med_ask])
    res["bid_vacuum"] = episodes(hb[hb.bid_usd_05 < 0.35 * med_bid])
    res["funding_le_-0.5pct"] = episodes(U[U.funding <= -0.005], gap_min=24 * 60)
    res["funding_ge_+0.1pct"] = episodes(U[U.funding >= 0.001], gap_min=24 * 60)
    flow = hb.buy_usd.fillna(0) + hb.sell_usd.fillna(0)
    f15 = flow.groupby(hb.sym).transform(lambda s: s.rolling(15, min_periods=10).sum())
    b15 = hb.buy_usd.fillna(0).groupby(hb.sym).transform(lambda s: s.rolling(15, min_periods=10).sum())
    med15 = flow.groupby(hb.sym).transform("median") * 15
    res["taker_buy_climax"] = episodes(hb[(f15 >= 5 * med15) & (b15 / f15 >= 0.6)])
    res["taker_sell_climax"] = episodes(hb[(f15 >= 5 * med15) & (b15 / f15 <= 0.4)])
    days = (F.t.max() - F.t.min()).total_seconds() / 86400
    rows = []
    for k, e in res.items():
        rows.append(dict(trigger=k, possibilities=len(e), coins=e.sym.nunique() if len(e) else 0, windows15=e.win15.nunique() if len(e) else 0,
                         per_day=round(len(e) / days, 1), windows_per_day=round((e.win15.nunique() if len(e) else 0) / days, 1)))
    return pd.DataFrame(rows), days


def step3():
    F = load_features()
    fills = pd.read_csv(os.path.join(D, "fills.csv"))
    rs = reach_stats(F, set(fills.pair))
    rs.to_csv(os.path.join(D, "reach_fill_coins.csv"))
    tc, days = trigger_counts(F)
    tc.to_csv(os.path.join(D, "trigger_counts.csv"), index=False)
    U = usdt_only(F)
    liq = dict(liq_long_usdt=float(U.liq_long_usd.sum()), liq_short_usdt=float(U.liq_short_usd.sum()), coins_usdt_mark=int(U.sym.nunique()),
               coins_usdt_book=int(U[U.n_book > 0].sym.nunique()))
    bt = F[F.sym == "BTCUSDT"].sort_values("t").dropna(subset=["mark"])
    liq.update(btc_first=float(bt.mark.iloc[0]), btc_last=float(bt.mark.iloc[-1]), btc_hi=float(bt.mark.max()), btc_lo=float(bt.mark.min()))
    inv = json.load(open(os.path.join(D, "inventory.json")))
    inv.update(liq)
    json.dump(inv, open(os.path.join(D, "inventory.json"), "w"), indent=1)
    print(rs.to_string())
    print(tc.to_string(), days)
    print(liq)


if __name__ == "__main__" and os.environ.get("STEP3"):
    step3()
