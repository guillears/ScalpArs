#!/usr/bin/env python3
"""H4 ask vacuum — the two FROZEN HOLDOUT lines of PREREG Addendum 9 (research only; operator 2026-11-24 "yes freeze both lines").

  H4-T  tradeable: H4_ask_vacuum openers whose 20-level ask $ within 0.5 % ≥ $12,000, exit TP +5 / stop −3 / 24 h, vs matched controls.
  H4-B  market switch (DAY units): the H4-T rows split by BTC's 4 h return at the trigger (> 0 = UP).
Only rows with trigger minute ≥ 2026-12-07 00:00 UTC count (HOLDOUT). Until the floors are met the status is COLLECTING; at the first read
with the floors met the verdict is PASS or FAIL (frozen, never re-fitted). Alert only — nothing arms.

Run (after build.py has refreshed the datasets):  venv/bin/python -I scripts/route_b_signals/h4_frozen_lines.py
Writes reports/H4_FROZEN_LINES_<UTC date>.md.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exits as X  # noqa: E402
import interim_q3 as I  # noqa: E402
import sources as SRC  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KEY = "H4_ask_vacuum"
HOLDOUT_START = pd.Timestamp("2026-12-07 00:00", tz="UTC")
ASK_MIN_USD = 12_000.0
TP, SL, HOLD = 5.0, 3.0, 1440
COST_OK = ("route_b_book", "book_unwalkable")
MIN_MS = 60_000
T_MIN_N, T_MIN_DAYS = 30, 10
B_MIN_DAYS = 10
N_BOOT = 2000
SEED = 20261207
EXCL = ("BTCUSDT", "ETHUSDT")
FUNDING_LAG_MS = 10 * MIN_MS                  # a settlement is published a few minutes after its time (sources.py)
STATE = os.path.join(ROOT, "reports", "H4_FROZEN_LINES_state.json")   # the first non-COLLECTING verdict per line, frozen


def holdout_rows(T):
    """the H4-T cohort rows (pure): key, HOLDOUT by trigger minute, a recorded book, an entry minute, ask $ ≥ the frozen floor."""
    m = (T.trigger == KEY) & (T.minute >= HOLDOUT_START.value // 1_000_000) & T.h_cost_src.isin(COST_OK) & T.h_entry_ms.notna()
    m &= (T.rb_ask_usd_05 >= ASK_MIN_USD) & ~T.pair.isin(EXCL)
    return T[m].copy()


def walk_net(k, e_ms, slip_in, slip_out, fund):
    """net % of one long from the open of minute e_ms under the frozen exit; NaN unless the 24 h path is complete. Pure."""
    k = k[k.open_ms >= e_ms]
    if len(k) < HOLD or int(k.open_ms.iloc[0]) != int(e_ms) or int(k.open_ms.iloc[HOLD - 1]) != int(e_ms) + (HOLD - 1) * MIN_MS:
        return math.nan                                               # path missing or not contiguous → not final
    k = k.iloc[:HOLD]
    o, h, l, c, t = (k[x].to_numpy(float) for x in ("o", "h", "l", "c", "open_ms"))
    w = X.walk(o, h, l, c, t, float(o[0]), 1, TP, SL, HOLD, funding=fund, entry_slip_pct=max(slip_in, 0), exit_slip_pct=max(slip_out, 0),
               entry_ms=int(e_ms))
    return w["net_pct"] if w["final"] else math.nan


def price(df, now_ms):
    out = []
    for r in df.itertuples():
        e = int(r.h_entry_ms)
        if now_ms < e + HOLD * MIN_MS + FUNDING_LAG_MS:               # the hold's last funding settlement may not be published yet
            out.append(math.nan)
            continue
        k = SRC.klines(r.pair, e, e + (HOLD + 5) * MIN_MS, now_ms)
        fund, ok = SRC.funding_history(r.pair, e, e + HOLD * MIN_MS, now_ms)
        out.append(walk_net(k, e, float(r.h_slip_entry_pct), float(r.h_slip_exit_pct), fund) if (k is not None and len(k) and ok)
                   else math.nan)
    return np.array(out, float)


def btc_ret4h(s, minute_ms):
    """BTC 4 h return % at a trigger minute m: close of m − 1 vs close of m − 241 (both must exist). s = BTC closes indexed by open_ms. Pure."""
    a, b = minute_ms - MIN_MS, minute_ms - 241 * MIN_MS
    if a not in s.index or b not in s.index:
        return math.nan
    return (s[a] / s[b] - 1) * 100


def boot_ci(groups, rng, n=N_BOOT):
    """95 % CI of the pooled mean, resampling whole groups. groups = list of arrays."""
    if len(groups) < 3:
        return (math.nan, math.nan)
    sums = np.array([g.sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    k = rng.integers(0, len(groups), (n, len(groups)))
    m = sums[k].sum(1) / np.maximum(cnts[k].sum(1), 1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def gates_T(rows, ctrl, rng):
    """H4-T status (pure given priced rows): rows need net, day, hour, pair, trigger_id; ctrl needs net, trigger_id."""
    r = rows.dropna(subset=["net"])
    st = dict(N=len(r), days=r.day.nunique())
    if st["N"] < T_MIN_N or st["days"] < T_MIN_DAYS:
        return dict(st, status="COLLECTING")
    x = r.net.to_numpy()
    st["mean"] = x.mean()
    st["ci"] = boot_ci([g.net.to_numpy() for _, g in r.groupby("hour")], rng)
    c = ctrl.dropna(subset=["net"])
    c = c[c.trigger_id.isin(r.trigger_id)]
    cm = c.groupby("trigger_id").net.mean()
    paired = r.set_index("trigger_id").net.reindex(cm.index) - cm
    hr = r.set_index("trigger_id").hour.reindex(cm.index)
    st["delta"] = paired.mean() if len(paired) else math.nan
    st["delta_ci"] = boot_ci([g.to_numpy() for _, g in paired.groupby(hr)], rng) if len(paired) else (math.nan, math.nan)
    kf = I.kelly_floor(x)
    if not np.isfinite(kf) and (x < 0).sum() == 0 and (x > 0).sum():   # no losers: Kelly → the Wilson WR floor itself
        kf = I.wilson_lo(int((x > 0).sum()), len(x))
    st["kelly_floor"] = kf
    tot = x.sum()
    st["top_coin_share"] = r.groupby("pair").net.sum().max() / tot if tot > 0 else math.nan
    st["top_day_share"] = r.groupby("day").net.sum().max() / tot if tot > 0 else math.nan
    st["loo_coin_min"] = min(r[r.pair != p].net.mean() for p in r.pair.unique())
    st["loo_day_min"] = min(r[r.day != d].net.mean() for d in r.day.unique())
    st["capacity_median_ask_usd"] = float(r.rb_ask_usd_05.median()) if "rb_ask_usd_05" in r else math.nan
    keep = 0.5 * st["mean"]                                           # §6 B2: leave-one-out keeps the sign AND ≥ 50 % of the effect
    ok = (st["ci"][0] > 0 and st["delta_ci"][0] > 0 and st["kelly_floor"] > 0 and st["top_coin_share"] < 0.5
          and st["top_day_share"] < 0.5 and st["loo_coin_min"] >= keep > 0 and st["loo_day_min"] >= keep)
    return dict(st, status=regime_label("PASS", r) if ok else "FAIL")


def regime_label(status, r):
    """B4: a pass seen under one BTC daily-trend sign only is capped at OBSERVE. Pure."""
    signs = set(r["btc_d_trend"].dropna().unique()) if "btc_d_trend" in r else set()
    return status if len(signs) >= 2 else status + " (one BTC regime → OBSERVE)"


def gates_B(rows, rng):
    """H4-B status in DAY units (pure given priced rows with net, day, btc4h)."""
    r = rows.dropna(subset=["net", "btc4h"])
    up = r[r.btc4h > 0].groupby("day").net.mean()
    dn = r[r.btc4h <= 0].groupby("day").net.mean()
    st = dict(days_up=len(up), days_down=len(dn))
    if len(up) < B_MIN_DAYS or len(dn) < B_MIN_DAYS:
        return dict(st, status="COLLECTING")
    u, d = up.to_numpy(), dn.to_numpy()
    days = np.array(sorted(set(up.index) | set(dn.index)))
    gaps, ups = [], []
    for _ in range(N_BOOT):                                          # JOINT day resample: a day can sit on both sides
        k = days[rng.integers(0, len(days), len(days))]
        a, b = up.reindex(k).dropna(), dn.reindex(k).dropna()
        if len(a) and len(b):
            gaps.append(a.mean() - b.mean())
        if len(a):
            ups.append(a.mean())
    st.update(gap=u.mean() - d.mean(), gap_ci=tuple(np.percentile(gaps, [2.5, 97.5])), up_mean=u.mean(),
              up_ci=tuple(np.percentile(ups, [2.5, 97.5])))
    loo = [up.drop(dy, errors="ignore").mean() - dn.drop(dy, errors="ignore").mean() for dy in set(up.index) | set(dn.index)]
    st["loo_gap_min"] = min(loo)
    ok = st["gap_ci"][0] > 0 and st["up_ci"][0] > 0 and st["loo_gap_min"] > 0
    return dict(st, status=regime_label("PASS", r) if ok else "FAIL")


def freeze(state, line, result, cutoff):
    """read ONCE: the first non-COLLECTING result per line is stored and returned on every later run (never re-judged). Pure."""
    if line in state:
        return state[line], state
    if result["status"] == "COLLECTING":
        return result, state
    state = dict(state)
    state[line] = dict({k: (list(v) if isinstance(v, tuple) else v) for k, v in result.items()}, frozen_at=cutoff)
    return state[line], state


def main():
    T, C, meta = I.load()
    now_ms = int(pd.Timestamp.now("UTC").value // 1_000_000)
    rows = holdout_rows(T)
    rng = np.random.default_rng(SEED)
    if len(rows):
        rows["net"] = price(rows, now_ms)
        rows["day"] = rows.minute.astype("int64") // I.DAY_MS
        rows["hour"] = (rows.signal_time.astype("datetime64[ms, UTC]").astype("int64")) // I.HOUR_MS
        ctrl = C[(C.trigger_id.isin(rows.trigger_id)) & (C.ctrl_k >= 0) & (C.ctrl_final == True) & C.h_entry_ms.notna()  # noqa: E712
                 & C.h_cost_src.isin(COST_OK)].copy()
        ctrl["net"] = price(ctrl, now_ms) if len(ctrl) else []
        lo, hi = int(rows.minute.min()) - 300 * MIN_MS, int(rows.minute.max()) + MIN_MS
        btc = SRC.klines("BTCUSDT", lo, hi, now_ms)
        bs = btc.set_index("open_ms").c if btc is not None and len(btc) else None
        rows["btc4h"] = [btc_ret4h(bs, int(m)) for m in rows.minute] if bs is not None else math.nan
    else:
        rows = rows.assign(net=[], day=[], hour=[], btc4h=[])
        ctrl = pd.DataFrame(columns=["trigger_id", "net"])
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    cutoff = str(meta.get("data_end"))
    gt, state = freeze(state, "H4-T", gates_T(rows, ctrl, rng), cutoff)
    gb, state = freeze(state, "H4-B", gates_B(rows, rng), cutoff)
    json.dump(state, open(STATE, "w"), indent=1, default=float)
    date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    f = lambda v: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (  # noqa: E731
        f"{v[0]:+.2f} … {v[1]:+.2f}" if isinstance(v, (tuple, list)) else (f"{v:+.2f}" if isinstance(v, float) else str(v)))
    L = [f"# H4 frozen lines (PREREG Addendum 9) — {date}", "",
         f"HOLDOUT from {HOLDOUT_START:%Y-%m-%d %H:%M} UTC · data to {meta.get('data_end')} · alert only, nothing arms.", "",
         "| line | status | " + " | ".join(k for k in gt if k != "status") + " |", "|---|---|" + "---|" * (len(gt) - 1),
         "| H4-T | **" + gt["status"] + "** | " + " | ".join(f(v) for k, v in gt.items() if k != "status") + " |", "",
         "| line | status | " + " | ".join(k for k in gb if k != "status") + " |", "|---|---|" + "---|" * (len(gb) - 1),
         "| H4-B | **" + gb["status"] + "** | " + " | ".join(f(v) for k, v in gb.items() if k != "status") + " |"]
    out = os.path.join(ROOT, "reports", f"H4_FROZEN_LINES_{date}.md")
    open(out, "w").write("\n".join(L) + "\n")
    print(out, gt["status"], gb["status"])


if __name__ == "__main__":
    main()
