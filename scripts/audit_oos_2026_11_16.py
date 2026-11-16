"""OOS audit 2026-11-16 (operator): per sleeve ① raw master edge (no fitted filters) ② forward (as traded since 2026-11-11 15:11 UTC,
DL 527) ③ backtest yr6 per seed ④ live only. Read-only."""
import glob, sys
import numpy as np, pandas as pd

R = "/Users/guillearslanian/Downloads/NOFA AI/"
LIVE_NOW = "/Users/guillearslanian/Downloads/scalpars_orders_live_2026-11-16_15-45-38.csv"
rng = np.random.default_rng(7)
SLEEVES = [("Momentum long", "MOMENTUM", "LONG"), ("Momentum short", "MOMENTUM", "SHORT"), ("FAN flip", "FLIP:FAN_RATIO_GATE", "SHORT"),
           ("Spike fade", "SPIKE_FADE", "SHORT"), ("LITE", "FRENZY_LITE", "LONG"), ("LITE RETRY", "FRENZY_LITE_RETRY", "LONG"),
           ("WILLY", "FRENZY_WILLY", "LONG"), ("WILLY HT", "FRENZY_WILLY_HT", "LONG"), ("FRENZY_LONG", "FRENZY_LONG", "LONG"),
           ("Flatbase short", "FLATBASE_SHORT", "SHORT"), ("Bull-Run", "BULLRUN_LONG", "LONG"), ("Bear-Run", "BEARRUN_SHORT", "SHORT"),
           ("SURGE long", "SURGE_LONG", "LONG")]
BT_NAME = {"MOMENTUM|LONG": "MOMENTUM", "MOMENTUM|SHORT": "MOMENTUM"}
STRUCTURAL = ("SPIKE_BOUNCE_OFF", "FRENZY_WIDE_OFF", "SPIKE_CHASE_OFF", "SURGE_SHORT_OFF", "BULLRUN_V1", "MANUAL_EXEMPT", "PROBE_EXEMPT")


def probe_mask(d):
    p = pd.Series(False, index=d.index)
    if "is_probe" in d:
        p |= d.is_probe.astype(str).str.lower().isin(("true", "1", "1.0"))
    for c in ("cell_multiplier_source", "pattern_cell_source"):
        if c in d:
            p |= d[c].astype(str).str.upper().str.contains("PROBE")
    return p


def stats(p, t, usd=None):
    p = pd.to_numeric(p, errors="coerce"); ok = p.notna()
    p, t = p[ok].values, pd.to_datetime(pd.Series(t)[ok.values].astype(str).str[:19].str.replace("T", " ")).values
    n = len(p)
    if n == 0:
        return dict(N=0)
    w, l = p[p > 0], p[p <= 0]
    be = (abs(l.mean()) / (w.mean() + abs(l.mean())) * 100) if len(w) and len(l) else np.nan
    day = pd.Series(t).dt.strftime("%Y-%m-%d").values
    days = np.unique(day); g = {d_: p[day == d_] for d_ in days}
    bs = np.array([np.concatenate([g[days[i]] for i in rng.integers(0, len(days), len(days))]).mean() for _ in range(2000)])
    ts = np.sort(t.astype("datetime64[s]").astype(np.int64)); win = 1 + int((np.diff(ts) > 1800).sum()) if n > 1 else 1
    out = dict(N=n, WR=(p > 0).mean() * 100, avg=p.mean(), BE=be, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
               Ppos=(bs > 0).mean(), days=len(days), win=win)
    if usd is not None:
        out["usd"] = pd.to_numeric(pd.Series(usd), errors="coerce")[ok.values].sum()
    return out


def fmt(s):
    if not s or s.get("N", 0) == 0:
        return "—"
    return f"{s['N']} · {s['WR']:.0f}% · {s['avg']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] P+ {s['Ppos']:.2f} · {s['days']}d/{s['win']}w"


# ① raw master
m = pd.read_csv(R + "reports/MASTER_POOL_stacked.csv", low_memory=False)
m = m[~probe_mask(m)]
reason = m.stack_block_reason.astype(str)
keep = m.stack_keep.astype(str).eq("True")
m = m[keep | ~reason.str.startswith(STRUCTURAL)]
m["P"] = np.where(m.stack_keep.astype(str).eq("True"), pd.to_numeric(m.stack_pct, errors="coerce"), pd.to_numeric(m.pnl_percentage, errors="coerce"))
# ② forward / ④ live
def load(fs):
    out = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False); d["_f"] = f.split("/")[-1]; out.append(d)
    d = pd.concat(out, ignore_index=True)
    d = d[d.status.astype(str).eq("CLOSED") & ~d.entry_strategy.astype(str).eq("MANUAL")]
    d = d[~probe_mask(d)]
    d["_k"] = d.opened_at.astype(str).str[:19] + d.pair.astype(str) + d.direction.astype(str)
    return d.drop_duplicates("_k", keep="last")
live = load(sorted(glob.glob(R + "reports/BASELINE2[234]*_LIVE_orders.csv")) + [LIVE_NOW])
fwd = load([R + "reports/BASELINE26_batch1111-1114_orders.csv", LIVE_NOW])
fwd = fwd[fwd.opened_at.astype(str) >= "2026-11-11T15:11"]
# ③ backtest
bt = pd.read_csv(R + "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv", low_memory=False,
                 usecols=["entry_strategy", "direction", "status", "is_probe", "seed", "opened_at", "pair", "pnl_percentage"])
bt = bt[bt.status.eq("CLOSED") & ~bt.is_probe.astype(bool)]

rows = []
for name, es, dr in SLEEVES:
    a = m[m.entry_strategy.astype(str).eq(es) & m.direction.astype(str).eq(dr)]
    kept = a[a.stack_keep.astype(str).eq("True")]
    f = fwd[fwd.entry_strategy.astype(str).eq(es) & fwd.direction.astype(str).eq(dr)]
    lv = live[live.entry_strategy.astype(str).eq(es) & live.direction.astype(str).eq(dr)]
    r = dict(sleeve=name, raw=stats(a.P, a.opened_at), kept=stats(kept.P, kept.opened_at), fwd=stats(f.pnl_percentage, f.opened_at, f.pnl),
             live=stats(lv.pnl_percentage, lv.opened_at, lv.pnl))
    for sd in (1, 2, 3):
        b = bt[bt.entry_strategy.eq(es) & bt.direction.eq(dr) & bt.seed.eq(sd)]
        r[f"bt{sd}"] = stats(b.pnl_percentage, b.opened_at)
    r["blocks_ignored"] = a[~a.stack_keep.astype(str).eq("True")].stack_block_reason.astype(str).str.replace(r"\(.*", "", regex=True).value_counts().to_dict()
    rows.append(r)
pd.to_pickle(rows, "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/oos_rows.pkl")
for r in rows:
    print(f"\n## {r['sleeve']}")
    for k in ("raw", "kept", "fwd", "live", "bt1", "bt2", "bt3"):
        s = r[k]; extra = f" · ${s['usd']:,.0f}" if s.get("usd") is not None and s.get("N") else ""
        be = f" · BE {s['BE']:.0f}%" if s.get("N") and not np.isnan(s.get("BE", np.nan)) else ""
        print(f"  {k:5s} {fmt(s)}{be}{extra}")
    print("  ignored blocks:", {k: v for k, v in list(r['blocks_ignored'].items())[:8]})
