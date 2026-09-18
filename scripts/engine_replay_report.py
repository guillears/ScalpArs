#!/usr/bin/env python3
"""Merge engine-replay chunks → Performance-by-Sleeve table with a COMPOUNDING ledger.

Each chunk ran the real engine on a ~$5k book; fills are re-priced here at the running
balance with the engine's OWN calculate_position_size (same config, same reserve /
leverage schedules) plus the engine's liquidity + gross caps, so P&L compounds exactly
the way the live book would (P&L% per notional is leverage-invariant and net of fees).

  venv/bin/python scripts/engine_replay_report.py --tags jan feb ... --balance 5000
"""
import os, sys, argparse, json
import pandas as pd, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./_replay_report_dummy.db"   # hard-set: never inherit a real DB URL
os.environ["DEBUG"] = "false"
OUT = os.path.join(ROOT, "reports", "backtest_cache", "replay")

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--balance", type=float, default=5000.0)
ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
ap.add_argument("--split", default="2026-06-17")     # in-sample boundary (first live batch day)
ap.add_argument("--out", default=None)
ap.add_argument("--exclude-sleeves", default="")
A = ap.parse_args()

import config
from services import trading_engine as te
ENG = te.trading_engine
ENG._bnb_burn_rate = 0.0
INV = config.trading_config.investment

def sleeve_of(strat, direction):
    s = strat if isinstance(strat, str) and strat else "MOMENTUM"
    if s.startswith("FLIP"): return "FLIP-short"
    if s == "MOMENTUM": return "MOM-long" if direction == "LONG" else "MOM-short"
    if s == "BULLRUN_LONG": return "BullRun-Long"
    if s == "BEARRUN_SHORT": return "BearRun-Short"
    if s == "SPIKE_CHASE": return "Spike-Chase"
    if s == "SPIKE_FADE": return "Spike-Fade"
    if s == "SPIKE_BOUNCE": return "Spike-Bounce"
    return s

frames = []
for t in A.tags:
    fp = os.path.join(OUT, f"{t}_orders.csv")
    df = pd.read_csv(fp, low_memory=False)
    meta = json.load(open(os.path.join(OUT, f"{t}_meta.json")))
    df["opened_at"] = pd.to_datetime(df["opened_at"], format="mixed")
    df["closed_at"] = pd.to_datetime(df["closed_at"], format="mixed")
    lo = pd.Timestamp(meta["start_ms"], unit="ms"); hi = pd.Timestamp(meta["end_ms"], unit="ms")
    df = df[(df.opened_at >= lo) & (df.opened_at < hi)]          # warm-up / follow-through excluded
    df["chunk"] = t
    frames.append(df)
o = pd.concat(frames, ignore_index=True)
o = o[o.status == "CLOSED"].copy()
if A.start: o = o[o.opened_at >= A.start]
if A.end: o = o[o.opened_at < A.end]
o["sleeve"] = [sleeve_of(s, d) for s, d in zip(o.entry_strategy, o.direction)]
src = o["cell_multiplier_source"].fillna("").astype(str)
o["is_probe"] = src.str.contains("PROBE")
if A.exclude_sleeves:
    o = o[~o.sleeve.isin(A.exclude_sleeves.split(","))]
o = o.sort_values("opened_at").reset_index(drop=True)
full = o[~o.is_probe].copy()

def compound(fills, balance0):
    """Chronological ledger: open at running balance with the engine's sizing, close at the
    replay's realized P&L%/notional. Returns (final_balance, priced fills, daily equity)."""
    ev = []
    for i, r in fills.iterrows():
        ev.append((r.opened_at, 1, i)); ev.append((r.closed_at, 0, i))
    ev.sort(key=lambda x: (x[0], x[1]))
    bal = balance0; open_pos = {}; priced = {}; skipped = 0
    equity_curve = []
    for ts, kind, i in ev:
        r = fills.loc[i]
        if kind == 1:
            open_margin = sum(p["inv"] for p in open_pos.values())
            open_notional = sum(p["notional"] for p in open_pos.values())
            available = bal - open_margin
            equity = bal
            inv, lev, _ = ENG.calculate_position_size(
                available, r.confidence, total_portfolio=equity,
                cell_multiplier=float(r.cell_multiplier or 1.0),
                cell_lev_multiplier=float(r.cell_lev_multiplier or 1.0),
                multiplier_target="both")
            if inv <= 0 or lev <= 0:
                skipped += 1; continue
            notional = inv * lev
            liq_pct = float(INV.max_notional_pct_of_pair_volume or 0)
            vol = r.get("entry_pair_volume_24h_usd")
            strat = r.entry_strategy if isinstance(r.entry_strategy, str) else ""
            if strat.startswith("SPIKE") and INV.spike_lowvol_liq_cap_pct and vol and 0 < vol < INV.spike_lowvol_threshold_usd:
                liq_pct = float(INV.spike_lowvol_liq_cap_pct)
            cap = None
            if liq_pct > 0 and vol and vol > 0: cap = liq_pct / 100.0 * vol
            if INV.max_notional_hard_ceiling: cap = INV.max_notional_hard_ceiling if cap is None else min(cap, INV.max_notional_hard_ceiling)
            if cap is not None and notional > cap: notional = cap
            if INV.max_gross_leverage:
                room = max(0.0, equity * INV.max_gross_leverage - open_notional)
                if room <= 0: skipped += 1; continue
                notional = min(notional, room)
            inv = notional / lev
            if inv < INV.min_investment_size: skipped += 1; continue
            open_pos[i] = {"inv": inv, "notional": notional, "lev": lev}
        else:
            p = open_pos.pop(i, None)
            if p is None: continue
            pnl = float(r.pnl_percentage or 0) / 100.0 * p["notional"]
            bal += pnl
            priced[i] = (p["inv"], p["lev"], p["notional"], pnl)
            equity_curve.append((ts, bal))
    return bal, priced, skipped, equity_curve

def table(df, label, days):
    rows = []
    for sl, g in df.groupby("sleeve"):
        fb, priced, sk, _ = compound(g, A.balance)
        pnl = sum(v[3] for v in priced.values())
        dcr = ((fb / A.balance) ** (1 / days) - 1) * 100 if days > 0 and fb > 0 else float("nan")
        rows.append({"Sleeve": sl, "N": len(g), "WR": f"{(g.pnl > 0).mean() * 100:.0f}%",
                     "Avg P&L%": f"{g.pnl_percentage.mean():+.3f}", "Net $ (replay $5k book)": f"{g.pnl.sum():+,.0f}",
                     "Net $ (compounded)": f"{pnl:+,.0f}", "Final $": f"{fb:,.0f}", "Daily compound": f"{dcr:+.3f}%", "skipped": sk})
    fb, priced, sk, curve = compound(df, A.balance)
    pnl = sum(v[3] for v in priced.values())
    dcr = ((fb / A.balance) ** (1 / days) - 1) * 100 if days > 0 and fb > 0 else float("nan")
    rows.append({"Sleeve": "TOTAL (shared book)", "N": len(df), "WR": f"{(df.pnl > 0).mean() * 100:.0f}%",
                 "Avg P&L%": f"{df.pnl_percentage.mean():+.3f}", "Net $ (replay $5k book)": f"{df.pnl.sum():+,.0f}",
                 "Net $ (compounded)": f"{pnl:+,.0f}", "Final $": f"{fb:,.0f}", "Daily compound": f"{dcr:+.3f}%", "skipped": sk})
    t = pd.DataFrame(rows)
    print(f"\n=== {label} — {days:.0f} days, start ${A.balance:,.0f} ===")
    print(t.to_string(index=False))
    return t, curve

lo, hi = full.opened_at.min(), full.opened_at.max()
if A.start: lo = pd.Timestamp(A.start)
if A.end: hi = pd.Timestamp(A.end)
days_all = (hi - lo).total_seconds() / 86400
out_lines = []
t_all, curve = table(full, f"FULL PERIOD {lo:%Y-%m-%d} → {hi:%Y-%m-%d}", days_all)
# headline: start → end
_fb, _pr, _sk, _cv = compound(full, A.balance)
print(f"\n>>> SHARED BOOK: started ${A.balance:,.0f} on {lo:%Y-%m-%d} → ended ${_fb:,.0f} on {hi:%Y-%m-%d} "
      f"({(_fb / A.balance - 1) * 100:+.1f}% over {days_all:.0f} days) — daily compound {((_fb / A.balance) ** (1 / days_all) - 1) * 100:+.3f}%")
for sl, g in full.groupby("sleeve"):
    fb_s, _, _, _ = compound(g, A.balance)
    print(f"    {sl:14s} own ledger: ${A.balance:,.0f} → ${fb_s:,.0f} ({(fb_s / A.balance - 1) * 100:+.1f}%) — daily compound {((fb_s / A.balance) ** (1 / days_all) - 1) * 100:+.3f}%")
if _cv:
    eqc = pd.DataFrame(_cv, columns=["ts", "equity"]).set_index("ts")
    print("\n=== SHARED BOOK, chained month by month (start-of-month → end-of-month balance) ===")
    prev = A.balance
    for mo_ in sorted(full.opened_at.dt.strftime("%Y-%m").unique()):
        m0 = pd.Timestamp(mo_ + "-01"); m1 = m0 + pd.offsets.MonthBegin(1)
        seg = eqc[(eqc.index >= m0) & (eqc.index < m1)]
        end = float(seg.equity.iloc[-1]) if len(seg) else prev
        d = max((min(m1, hi + pd.Timedelta(minutes=1)) - max(m0, lo)).total_seconds() / 86400, 1)
        dcr_m = ((end / prev) ** (1 / d) - 1) * 100 if prev > 0 and end > 0 else float("nan")
        print(f"    {mo_}: ${prev:,.0f} → ${end:,.0f} ({(end / prev - 1) * 100:+.1f}%, {d:.0f} days, daily compound {dcr_m:+.2f}%)")
        prev = end
sp = pd.Timestamp(A.split)
pre = full[full.opened_at < sp]; post = full[full.opened_at >= sp]
if len(pre): table(pre, f"OUT-OF-SAMPLE {lo:%Y-%m-%d} → {sp:%Y-%m-%d} (filters were NOT tuned on this)", (sp - lo).total_seconds() / 86400)
if len(post): table(post, f"IN-SAMPLE {sp:%Y-%m-%d} → {hi:%Y-%m-%d} (live batches the filters were tuned on)", (hi - sp).total_seconds() / 86400)
if o.is_probe.sum():
    pr = o[o.is_probe]
    print(f"\nprobe-sized fires excluded from the tables: {len(pr)} ({pr.pnl.sum():+.0f}$ at replay size)")
# monthly — daily compound return per sleeve per month (each cell = a fresh $balance0 ledger
# over that month's fills, DCR = (final/start)^(1/days in month) - 1), plus the TOTAL shared book
full["month"] = full.opened_at.dt.strftime("%Y-%m")
def dcr_of(df, days):
    fb, priced, sk, _ = compound(df, A.balance)
    return ((fb / A.balance) ** (1 / days) - 1) * 100 if days > 0 and fb > 0 else float("nan")
months = sorted(full.month.unique())
sleeves = sorted(full.sleeve.unique())
rows = []
for mo_ in months:
    g = full[full.month == mo_]
    m0 = pd.Timestamp(mo_ + "-01"); m1 = min(m0 + pd.offsets.MonthBegin(1), hi + pd.Timedelta(minutes=1))
    days = max((m1 - max(m0, lo)).total_seconds() / 86400, 1)
    row = {"month": mo_, "days": round(days)}
    for sl in sleeves:
        gs = g[g.sleeve == sl]
        row[sl] = f"{dcr_of(gs, days):+.2f}% ({len(gs)})" if len(gs) else "-"
    row["TOTAL"] = f"{dcr_of(g, days):+.2f}% ({len(g)})"
    rows.append(row)
yr = {"month": "YEAR", "days": round(days_all)}
for sl in sleeves:
    yr[sl] = f"{dcr_of(full[full.sleeve == sl], days_all):+.3f}%"
yr["TOTAL"] = f"{dcr_of(full, days_all):+.3f}%"
rows.append(yr)
print("\n=== DAILY COMPOUND RETURN per sleeve per month (own $5k ledger each; N in brackets) ===")
print(pd.DataFrame(rows).to_string(index=False))
mo = full.groupby(["month", "sleeve"]).agg(N=("pnl", "size"), WR=("pnl", lambda s: (s > 0).mean() * 100), pnl=("pnl", "sum"), avg=("pnl_percentage", "mean")).round(2)
print("\n=== monthly by sleeve (replay $5k book, not compounded) ===")
print(mo.unstack("sleeve").fillna(0).to_string())
if curve:
    eq = pd.DataFrame(curve, columns=["ts", "equity"]).set_index("ts")
    eq["peak"] = eq.equity.cummax(); eq["dd"] = eq.equity / eq.peak - 1
    print(f"\nshared book: max drawdown {eq.dd.min() * 100:.1f}% | end equity ${eq.equity.iloc[-1]:,.0f}")
if A.out:
    full.to_csv(A.out, index=False); print(f"\nfills saved → {A.out}")
