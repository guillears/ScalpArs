#!/usr/bin/env python3
"""Route B report 2026-11-28 (rebuild of the 11-21 edition to date) — step 2: per-fill recorded state at entry and during the trade (DESCRIPTIVE ONLY, N tiny).

Inputs: reports/ROUTE_B_REPORT_2026-11-28_data/{features_all.pkl.gz, events.json} (from ROUTE_B_REPORT_2026-11-28_load.py, S3 read-only)
        + exports_archive/scalpars_orders_paper_2026-11-*.csv AND the newest ~/Downloads/scalpars_orders_paper_*.csv (dedup key opened_at[:19], pair, direction; CLOSED) + the bot list (open positions).
Outputs: reports/ROUTE_B_REPORT_2026-11-28_data/{fills.csv, coverage.csv, inventory.json} + reports/ROUTE_B_REPORT_2026-11-28_chart.png
Usage: venv/bin/python -I scripts/ROUTE_B_REPORT_2026-11-28.py"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "ROUTE_B_REPORT_2026-11-28_data")
REC_START = pd.Timestamp("2026-11-19 23:23:30", tz="UTC")
MIN = 60_000
FULL_START = pd.Timestamp("2026-11-21 22:02", tz="UTC")       # DL 611 deploy: full book (30 priority coins) + OI + ratios begin
FB_COLS = None  # filled by load_features


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
    extra = [c for c in F.columns if c.startswith("fb_") or c in ("n_fbook", "oi", "oi_chg_1m", "oi_chg_5m", "top_ls_acct", "top_ls_pos",
                                                                    "global_ls_acct", "taker_bs_ratio")]
    ex = g[extra].last()                                              # full book (DL 611) + open interest / positioning, last non-null
    out = pd.concat([book, sums, mx, mk, ex], axis=1).reset_index()
    out["t"] = pd.to_datetime(out.minute, unit="ms", utc=True)
    return out


def load_fills(events):
    rows = []
    dl = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")))
    files = sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-*.csv"))) + dl[-1:]
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["src"] = os.path.basename(f)
        rows.append(d)
    a = pd.concat(rows, ignore_index=True)
    a = a[a.status == "CLOSED"].copy()
    for pc in [c for c in a.columns if "probe" in c.lower()]:          # no probes (none expected in this window)
        a = a[~a[pc].fillna(False).astype(bool)]
    a = a[~a.entry_strategy.astype(str).str.endswith("_PROBE")]
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
        m = c[(c.t < E)].dropna(subset=["mark"])                       # strictly before the fill's minute (no look-ahead)
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
        # ---- full book (DL 611, priority coins) + open interest / positioning, where recorded
        fbm = c[c.get("n_fbook", pd.Series(0, index=c.index)).fillna(0) > 0] if "n_fbook" in c else c.iloc[0:0]
        row["fbook_through"] = round(fbm.t.isin(span).sum() / max(len(span), 1), 3)
        row["fbook_at_entry"] = bool(((fbm.t >= E - pd.Timedelta(minutes=1)) & (fbm.t <= E)).any())
        if "oi" in c:
            oim = c[c.oi.notna() & (c.t >= E) & (c.t <= X)]
            row["oi_through"] = round(oim.t.isin(span).sum() / max(len(span), 1), 3)
            if len(oim) > 1:
                row["oi_chg_trade_pct"] = (oim.oi.iloc[-1] / oim.oi.iloc[0] - 1) * 100
                row["oi_chg5_min_trade"] = oim.oi_chg_5m.min()
                row["oi_chg5_max_trade"] = oim.oi_chg_5m.max()
            rt = c[c.top_ls_acct.notna() & (c.t >= E) & (c.t <= X)]
            if len(rt):
                row.update(top_ls_acct_first=rt.top_ls_acct.iloc[0], top_ls_acct_last=rt.top_ls_acct.iloc[-1],
                           global_ls_first=rt.global_ls_acct.iloc[0], global_ls_last=rt.global_ls_acct.iloc[-1])
        if len(fbm):
            fx = fbm[(fbm.t >= X - pd.Timedelta(minutes=2)) & (fbm.t <= X)]
            side = "buy" if sgn < 0 else "sell"
            if len(fx):
                row["fb_cost_exit_6000_bps"] = fx[f"fb_cost_{side}_6000"].iloc[-1]
                row["fb_adv_usd_05_exit"] = fx["fb_ask_usd_05" if sgn < 0 else "fb_bid_usd_05"].iloc[-1]
                row["fb_adv_usd_2_exit"] = fx["fb_ask_usd_2" if sgn < 0 else "fb_bid_usd_2"].iloc[-1]
        out.append(row)
    return pd.DataFrame(out)


def inventory(F, events):
    hb = F[F.n_book > 0]
    ev = [e for e in events if e.get("ev") in ("start", "stop", "disconnect", "reconnect")]
    return dict(first_minute=str(F.t.min()), last_minute=str(F.t.max()), hours=round((F.t.max() - F.t.min()).total_seconds() / 3600, 1),
                coins_mark=int(F.sym.nunique()), coins_book=int(hb.sym.nunique()), coin_minutes_book=int(len(hb)),
                book_snapshots=int(hb.n_book.sum()), trades=int(F.n_trades.sum()), liq_long_usd=float(F.liq_long_usd.sum()),
                liq_short_usd=float(F.liq_short_usd.sum()), n_gap_events=len(ev),
                gap_s=sum(float(e.get("gap_s") or 0) for e in ev if e["ev"] == "reconnect"),
                coins_fbook=int(F[F.n_fbook.fillna(0) > 0].sym.nunique()), coin_minutes_fbook=int((F.n_fbook.fillna(0) > 0).sum()),
                coins_oi=int(F[F.oi.notna()].sym.nunique()), coin_minutes_oi=int(F.oi.notna().sum()),
                coins_ratio=int(F[F.top_ls_acct.notna()].sym.nunique()),
                book_coins_by_day={str(d): int(n) for d, n in hb.groupby(hb.t.dt.date).sym.nunique().items()},
                n_bot_lists=sum(1 for e in events if e.get("ev") == "bot"))


def chart(F, fills, path):
    """AGGREGATE event study (operator 11-23: summaries only, never per trade): per sleeve, winners vs losers, the median (and IQR band)
    of four recorded series anchored at the entry minute — P&L in trade direction · liquidations against us · taker share against us ·
    the book wall against us (vs its own entry value). DESCRIPTIVE, N tiny."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SL = [("FRENZY_WILLY_HT", "HT"), ("EXHAUSTION_SHORT", "EXH"), ("DISTRIBUTION_SHORT", "DIST"), ("FLATBASE_SHORT", "FB")]
    GREEN, RED, GREY = "#1baf7a", "#eb6834", "#52514e"
    PRE, POST = 60, 240
    fig, ax = plt.subplots(4, len(SL), figsize=(18, 12), sharex=True)
    for j, (sl, nm) in enumerate(SL):
        sub = fills[(fills.entry_strategy == sl) & (fills.state == "CLOSED")]
        counts = {}
        for lab, col, g in (("W", GREEN, sub[sub.pnl_percentage > 0]), ("L", RED, sub[sub.pnl_percentage <= 0])):
            S = {k: [] for k in range(4)}
            for f in g.itertuples():
                sgn = -1 if f.direction == "SHORT" else 1
                E = f.open_t.floor("min")
                c = F[(F.sym == f.pair) & (F.t >= E - pd.Timedelta(minutes=PRE)) & (F.t <= E + pd.Timedelta(minutes=POST))].sort_values("t")
                if not len(c) or not (f.entry_price == f.entry_price):
                    continue
                X = f.close_t.floor("min")
                idx = ((c.t - E).dt.total_seconds() // 60).astype(int).values
                live = (c.t <= X).values                                   # the trade's own minutes only (after the exit = NaN)
                price = c.mid.fillna(c.mark).values
                pnl = pd.Series(np.where(live, (price / f.entry_price - 1) * 100 * sgn, np.nan))
                if f.pnl_percentage == f.pnl_percentage:                   # after the exit: hold the BOOKED result (no survivor bias)
                    pnl[~live] = float(f.pnl_percentage)
                pnl = pnl.values
                adv = "liq_short_usd" if sgn < 0 else "liq_long_usd"
                rb, rs = c.buy_usd.rolling(15, min_periods=5).sum(), c.sell_usd.rolling(15, min_periods=5).sum()
                share = ((rb if sgn < 0 else rs) / (rb + rs)).values * 100
                dep = c["ask_usd_05" if sgn < 0 else "bid_usd_05"].rolling(5, min_periods=1).median()
                d0 = dep[c.t < E]                                          # the baseline strictly before the fill's minute
                rel = (dep / d0.iloc[-1]).values if len(d0) and d0.iloc[-1] > 0 else np.full(len(c), np.nan)
                after = ~live & (c.t > E).values                          # the coin's tape after OUR exit is masked (not our trade)
                liqv = np.where(after, np.nan, c[adv].values / 1e3)
                share = np.where(after, np.nan, share)
                rel = np.where(after, np.nan, rel)
                for k, v in enumerate((pnl, liqv, share, rel)):
                    S[k].append(pd.Series(v, index=idx).groupby(level=0).last().reindex(range(-PRE, POST + 1)))
            counts[lab] = len(S[0])
            for k in range(4):
                if not S[k]:
                    continue
                M = pd.concat(S[k], axis=1)
                x = M.index.values
                ax[k, j].plot(x, M.median(axis=1), color=col, lw=1.6, label=f"{lab} (n {len(S[k])})")
                if len(S[k]) >= 3:
                    ax[k, j].fill_between(x, M.quantile(0.25, axis=1), M.quantile(0.75, axis=1), color=col, alpha=0.15, lw=0)
        ax[0, j].set_title(f"{nm} — winners {counts.get('W', 0)} · losers {counts.get('L', 0)}", fontsize=10)
        for k, (yl, ref) in enumerate((("P&L % in trade direction\n(booked result held after exit)", 0), ("liquidations AGAINST us\n$k / min", None),
                                       ("taker flow AGAINST us\n% of 15-min $", 50), ("book wall AGAINST us\n× its entry value (0.5 %)", 1))):
            if j == 0:
                ax[k, j].set_ylabel(yl)
            if ref is not None:
                ax[k, j].axhline(ref, color=GREY, lw=0.8)
            ax[k, j].axvline(0, color=GREY, lw=0.8, ls="--")
            for sp in ("top", "right"):
                ax[k, j].spines[sp].set_visible(False)
            ax[k, j].grid(axis="y", color="#e8e8e6", lw=0.6)
        ax[3, j].set_xlabel("minutes from entry")
        ax[0, j].legend(fontsize=8, frameon=False)
    fig.suptitle("Route B event study — per sleeve (WILLY, 2 fills, not plotted), winners vs losers: median line, IQR band (n ≥ 3). "
                 "Grey dashed = entry. Tape panels end at each fill's exit; P&L holds the booked result. DESCRIPTIVE, N tiny, one BTC daily-trend sign.",
                 fontsize=10)
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
    chart(F, fills, os.path.join(ROOT, "reports", "ROUTE_B_REPORT_2026-11-28_chart.png"))
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
    fb = F[(F.n_fbook.fillna(0) > 0) & F.sym.isin(pairs)]                 # the full book (DL 611) on the same coins, where recorded
    if len(fb):
        h = fb.groupby("sym").agg(fb_minutes=("n_fbook", "size"), fb_reach_bid=("fb_reach_bid_pct", "median"),
                                  fb_reach_ask=("fb_reach_ask_pct", "median"), fb_cost6k_buy=("fb_cost_buy_6000", "median"),
                                  fb_cost6k_sell=("fb_cost_sell_6000", "median"), fb_cost10k_nan=("fb_cost_buy_10000", lambda s: s.isna().mean()),
                                  fb_bid05=("fb_bid_usd_05", "median"), fb_ask05=("fb_ask_usd_05", "median"),
                                  fb_bid2=("fb_bid_usd_2", "median"), fb_ask2=("fb_ask_usd_2", "median"))
        # the top-20 stream over the SAME minutes (like for like)
        bb = F[(F.n_book > 0) & (F.n_fbook.fillna(0) > 0) & F.sym.isin(pairs)].groupby("sym").agg(
            t20_reach_same=("reach_bid_pct", "median"), t20_cost10_nan_same=("cost_buy_10000", lambda s: s.isna().mean()))
        g = g.join(h, how="left").join(bb, how="left")
    return g.round(3)


def reach_all(F):
    """share of recorded coins with a short top-20 reach / no $10k cost (top-20 stream) — the 11-21 data-quality numbers, to date."""
    b = usdt_only(F)
    b = b[b.n_book > 0]
    g = b.groupby("sym").agg(reach=("reach_bid_pct", "median"), c10=("cost_buy_10000", lambda s: s.isna().mean()))
    fb = b[b.n_fbook.fillna(0) > 0].groupby("sym").agg(fr=("fb_reach_bid_pct", "median"), fc=("fb_cost_buy_10000", lambda s: s.isna().mean()))
    return dict(coins=len(g), reach_lt05=float((g.reach < 0.5).mean()), c10_nan_most=float((g.c10 > 0.5).mean()),
                fb_coins=len(fb), fb_reach_lt05=float((fb.fr < 0.5).mean()) if len(fb) else None,
                fb_c10_nan_most=float((fb.fc > 0.5).mean()) if len(fb) else None)


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
    # H7 (new data since DL 611): open-interest flush — OI down ≥ 3 % in 5 min while the price fell ≥ 2 % in 5 min (hot coins with OI)
    O = U[U.oi_chg_5m.notna()].copy()
    O["mv5"] = O.groupby("sym").mark.transform(lambda s: s / s.shift(5) - 1) * 100
    res["oi_flush_price_down"] = episodes(O[(O.oi_chg_5m <= -3) & (O.mv5 <= -2)])
    res["oi_flush_price_up"] = episodes(O[(O.oi_chg_5m <= -3) & (O.mv5 >= 2)])
    res["oi_build_price_up"] = episodes(O[(O.oi_chg_5m >= 5) & (O.mv5 >= 2)])
    days = (F.t.max() - F.t.min()).total_seconds() / 86400
    days_oi = (O.t.max() - O.t.min()).total_seconds() / 86400 if len(O) else np.nan
    rows = []
    for k, e in res.items():
        dd = days_oi if k.startswith("oi_") else days
        rows.append(dict(days=round(dd, 2), trigger=k, possibilities=len(e), coins=e.sym.nunique() if len(e) else 0, windows15=e.win15.nunique() if len(e) else 0,
                         per_day=round(len(e) / dd, 1), windows_per_day=round((e.win15.nunique() if len(e) else 0) / dd, 1)))
    return pd.DataFrame(rows), days


def step3():
    F = load_features()
    fills = pd.read_csv(os.path.join(D, "fills.csv"))
    rs = reach_stats(F, set(fills.pair))
    rs.to_csv(os.path.join(D, "reach_fill_coins.csv"))
    tc, days = trigger_counts(F)
    ra = reach_all(F)
    print(ra)
    tc.to_csv(os.path.join(D, "trigger_counts.csv"), index=False)
    U = usdt_only(F)
    liq = dict(liq_long_usdt=float(U.liq_long_usd.sum()), liq_short_usdt=float(U.liq_short_usd.sum()), coins_usdt_mark=int(U.sym.nunique()),
               coins_usdt_book=int(U[U.n_book > 0].sym.nunique()))
    bt = F[F.sym == "BTCUSDT"].sort_values("t").dropna(subset=["mark"])
    liq.update(btc_first=float(bt.mark.iloc[0]), btc_last=float(bt.mark.iloc[-1]), btc_hi=float(bt.mark.max()), btc_lo=float(bt.mark.min()))
    inv = json.load(open(os.path.join(D, "inventory.json")))
    bt["day"] = bt.t.dt.date
    liq["btc_by_day"] = {str(d): [round(float(x.mark.iloc[0])), round(float(x.mark.min())), round(float(x.mark.max())), round(float(x.mark.iloc[-1]))]
                         for d, x in bt.groupby("day")}
    liq["reach_all"] = ra
    Ud = U.assign(day=U.t.dt.date).groupby("day")[["liq_long_usd", "liq_short_usd"]].sum()
    liq["liq_by_day"] = {str(d): [round(float(r.liq_long_usd) / 1e6, 1), round(float(r.liq_short_usd) / 1e6, 1)] for d, r in Ud.iterrows()}
    inv.update(liq)
    json.dump(inv, open(os.path.join(D, "inventory.json"), "w"), indent=1)
    print(rs.to_string())
    print(tc.to_string(), days)
    print(liq)


if __name__ == "__main__" and os.environ.get("STEP3"):
    step3()
