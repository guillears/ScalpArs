#!/usr/bin/env python3
"""LITE GREEN-RETRY deep dive 2026-10-19 — block ③ (two-year backtest, 2025-01-01 → 2026-10-11). Research only, cache only (no network).

Variants and pricing are FROZEN in scripts/study_lite_green_retry_deep.py (docstring) — same names, same rules, same LITE C exit and entry ruler.

Universe = an engine-function enumeration of LITE stretch judgements (the scripts/study_b19_fzentry_lite_stretches.py construction, extended to
two years): per pair, 5m bars merged from reports/backtest_cache_2025/k5m_full + reports/backtest_cache/k5m_full + b19_5m (later wins);
spike LEADS = the walk's own lead rule (30-min return ≥ 5 % ∧ last-hour quote vol ≥ $2M ∧ ≥ 20× the normal hour, services.frenzy.normal_hour_usd);
every bar closing in [lead + 1h55, lead + 20 h] is walked once with the REAL services.frenzy.frenzy_walk (1,499 closed bars, today's thresholds,
frenzy_state_vol_mult 100) and kept when the episode is flagged (verified ∧ ≤ 96 h). LITE status = services.frenzy.frenzy_lite_status with
vol24 = services.frenzy.frenzy_vol24_at. Not modelled: the ≤ 25-pair FRENZY shortlist, scan latency / seeds (one deterministic path).
Concurrency: LITE slots (2) + pair-day cap (3, counting LITE fills + the replay's other FRENZY fills on that pair-day) simulated here; pair held
and the WILLY global hold come from the yr6 replay seed-1 fills (reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv — every NON-LITE position holds
its pair; WILLY fills hold globally). Bearish block = last closed UTC daily BTC candle < 0 ∧ BTC 5m EMA13 < EMA50. LITE C slope = BTC 1h EMA20
slope (last closed hour vs the one before). PARITY target: the replay's seed-1 FRENZY_LITE fills (pair, signal bar) — printed.
Read: per variant, per period (2025 / 2026-H1 / 2026-H2), per month, leave-one-month-out, day-block bootstrap, per pair / day concentration,
top-5 removed. Usage: venv/bin/python scripts/study_lite_green_retry_deep_bt.py walk|price|sim [workers]
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402  (also patches the 2025 tick / 1m caches via study_tick_maker_lock_common)
import study_lite_green_retry_deep as GR  # noqa: E402
import config  # noqa: E402
from services.frenzy import (frenzy_walk, normal_hour_usd, frenzy_flagged, frenzy_vol24_at, frenzy_lite_status,  # noqa: E402
                             FRENZY_LITE_JUDGES, FRENZY_LITE_READY)
TH0 = config.trading_config.thresholds; TC = config.trading_config
D = os.path.join(RC.SCR, "lgr", "bt"); os.makedirs(D, exist_ok=True)
MS5, MIN, H, WIN = 300_000, 60_000, 3_600_000, 1499
T0 = int(pd.Timestamp("2025-01-01").value // 1e6); T1 = int(pd.Timestamp("2026-10-12").value // 1e6)
C25, C26, CB = "reports/backtest_cache_2025/k5m_full", "reports/backtest_cache/k5m_full", "reports/backtest_cache/b19_5m"
SKIP = set()
for src in (getattr(TC, "pair_blacklist", ""), getattr(TC, "no_trade_pairs", ""), getattr(TH0, "frenzy_pair_blacklist", "")):
    SKIP |= {x.strip().upper() for x in str(src or "").split(",") if x.strip()}
TH = GR.EN.THX(TH0, frenzy_state_vol_mult=100.0, frenzy_lite_max_hours=16.75, frenzy_lite_enabled=True)


def load5(pair):
    fr = []
    for c in (C25, C26, CB):
        f = f"{c}/{pair}.csv"
        if os.path.exists(f):
            try:
                fr.append(pd.read_csv(f, usecols=["open_time", "o", "h", "l", "c", "vol"]))
            except Exception:
                pass
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    d["open_time"] = d.open_time.astype("int64")
    return d[(d.open_time >= T0 - 40 * 86_400_000) & (d.open_time < T1 + 86_400_000)].reset_index(drop=True)


def one(pair):
    d = load5(pair)
    if d is None or len(d) < 3000:
        return []
    ot = d.open_time.values; o, h, l, c, v = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "vol"))
    rows = [[int(a), float(b), float(e), float(f), float(g), float(x)] for a, b, e, f, g, x in zip(ot, o, h, l, c, v)]
    q = v * (h + l + c) / 3.0; q1h = pd.Series(q).rolling(12).sum().values
    hr = ot // H * H
    g = d.assign(hr=hr).groupby("hr").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("vol", "sum"), n=("o", "size"))
    g = g[g.n == 12]
    H1 = [[int(k), r.o, r.h, r.l, r.c, r.v] for k, r in zip(g.index, g.itertuples())]; hk = np.array([x[0] for x in H1], dtype="int64")
    ncache = {}

    def nh_at(j):
        k = int(np.searchsorted(hk, int(ot[j]) + MS5 - H, side="right"))
        k0 = max(0, k - 744); m = int(np.searchsorted(hk[k0:k] + H, int(ot[j]) - 24 * H, side="right"))
        key = (k0, k0 + m)
        if key not in ncache:
            ncache[key] = normal_hour_usd(H1[k0:k], int(ot[j]))
        return ncache[key]
    r30 = np.r_[np.full(6, np.nan), c[6:] / c[:-6] - 1] * 100
    cand = np.flatnonzero((ot >= T0 - 2 * 86_400_000) & (r30 >= float(TH0.frenzy_spike_ret_pct)) & (q1h >= float(TH0.frenzy_spike_min_hour_usd)))
    leads = [j for j in cand if (lambda nh: nh and q1h[j] / nh >= float(TH0.frenzy_spike_vol_mult))(nh_at(j))]
    if not leads:
        return []
    mark = np.zeros(len(ot), bool)
    for j in leads:                                   # bars closing in [lead close + 1h55, lead close + 20 h]
        a = j + 23; b = min(len(ot) - 1, j + 240)
        mark[a:b + 1] = True
    mark &= (ot + MS5 >= T0) & (ot + MS5 < T1)
    out = []
    for j in np.flatnonzero(mark):
        if j < 300:
            continue
        nh = nh_at(j)
        if not nh:
            continue
        sub = rows[max(0, j - WIN + 1):j + 1]
        ep = frenzy_walk(sub, nh, TH)
        if not (ep and frenzy_flagged(ep, TH)) or ep["hours"] > 20.5:
            continue
        out.append(dict(pair=pair, bar_open=int(ot[j]), t_close=int(ot[j]) + MS5, spike_ts=int(ep["spike_ts"]), hours=ep["hours"],
                        vol_mult=ep["vol_mult"], in_state=bool(ep["in_state"]), fresh_on=bool(ep["fresh_on"]), above_streak=int(ep["above_streak"]),
                        verified=bool(ep["verified"]), bar_red=bool(ep["bar_red"]), bar_ret=ep["bar_ret_pct"], vs_vwap=ep["vs_vwap_pct"],
                        vol24=frenzy_vol24_at(sub), c=float(c[j]), o=float(o[j])))
    return out


def step_walk(nw):
    pairs = sorted(({f[:-4] for f in os.listdir(C25) if f.endswith(".csv")} | {f[:-4] for f in os.listdir(C26) if f.endswith(".csv")}) - SKIP)
    pairs = [p for p in pairs if p.endswith("USDT") and p != "BTCUSDT"]
    with Pool(nw) as P:
        res = []
        for i, r in enumerate(P.imap_unordered(one, pairs, chunksize=1)):
            res += r
            if i % 50 == 0:
                print(i, len(pairs), len(res), flush=True)
    W = pd.DataFrame(res).sort_values(["t_close", "pair"]).reset_index(drop=True)
    W.to_pickle(f"{D}/walk.pkl"); print("walk rows", len(W), "pairs", W.pair.nunique())


# ───────────── BTC + replay occupancy ─────────────
def btc_frame():
    fr = [pd.read_csv(f, usecols=["open_time", "o", "c"]) for f in (f"{C25}/BTCUSDT.csv", "reports/backtest_cache/btc_5m.csv", f"{C26}/BTCUSDT.csv",
                                                                       f"{CB}/BTCUSDT.csv") if os.path.exists(f)]
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    d["open_time"] = d.open_time.astype("int64")
    c = d.c.values
    e13 = pd.Series(c).ewm(span=13, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    d["gap"] = (e13 - e50) / e50 * 100; d["tclose"] = d.open_time + MS5
    day = pd.to_datetime(d.open_time, unit="ms").dt.floor("D")
    dc = d.groupby(day).agg(o=("o", "first"), c=("c", "last"))
    dc["ret"] = (dc.c / dc.o - 1) * 100
    dd = pd.to_datetime(d.tclose, unit="ms").dt.floor("D") - pd.Timedelta(days=1)
    d["d1"] = dd.map(dc.ret).values
    hc = d.set_index(pd.to_datetime(d.open_time, unit="ms")).c.resample("1h").last().dropna()
    e20 = hc.ewm(span=20, adjust=False).mean(); sl = e20 - e20.shift(1)
    hclose = pd.Series(sl.values, index=hc.index + pd.Timedelta(hours=1))
    d["h1slope"] = hclose.reindex(pd.to_datetime(d.tclose, unit="ms"), method="ffill").values
    return d[["tclose", "gap", "d1", "h1slope"]].set_index("tclose")


def replay_fills():
    F = pd.read_csv("reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv", low_memory=False,
                    usecols=["pair", "direction", "entry_strategy", "seed", "opened_at", "closed_at", "pct_new", "status", "entry_frenzy_vol_mult",
                             "entry_frenzy_hours", "entry_btc_1h_slope", "close_reason"])
    F = F[(F.seed == 1) & (F.status == "CLOSED")].drop_duplicates(["opened_at", "pair", "direction"]).copy()
    F["t_open"] = pd.to_datetime(F.opened_at.str[:19]).dt.as_unit("ms").astype("int64")
    F["t_shut"] = pd.to_datetime(F.closed_at.str[:19]).dt.as_unit("ms").astype("int64")
    F["sig_close"] = (F.t_open - 1000) // MS5 * MS5
    return F.reset_index(drop=True)


# ───────────── pricing ─────────────
def keys_needed(W):
    """V0 judged READY bars + the retry superset (per green judged bar: first 3 red/flat same-stretch bars passing the LITE status, first 3
    red/flat same-stretch bars, first 3 strictly red bars ≤ 60 min) + RETRY_ANY candidates (first 3 READY bars of every judged stretch)."""
    need = set()
    for p, d in W.groupby("pair"):
        d = d.sort_values("t_close")
        judged = None; pend = []
        recs = list(d.itertuples())
        for i, r in enumerate(recs):
            if r.in_state:
                continue
            ep = dict(verified=r.verified, hours=r.hours, in_state=r.in_state, above_streak=r.above_streak, vol_mult=r.vol_mult,
                      last_bar_ts=r.bar_open, bar_red=r.bar_red, bar_ret_pct=r.bar_ret)
            ready, code, _ = frenzy_lite_status(ep, TH, r.vol24, judged)
            if code not in FRENZY_LITE_JUDGES:
                continue
            sid = int(r.bar_open) - (int(r.above_streak) - 1) * MS5
            judged = sid
            if ready:
                need.add((p, int(r.t_close), r.c))
            else:
                a = b = s = 0; need.add((p, int(r.t_close), r.c))   # the green bar itself (theory: entering ON the green bar)
                for x in recs[i + 1:]:
                    same = GR.stretch_id(x) == sid
                    if x.t_close - r.t_close <= 60 * MIN and x.bar_ret is not None and x.bar_ret < 0 and s < 3:
                        need.add((p, int(x.t_close), x.c)); s += 1
                    if same and x.bar_red:
                        if b < 3:
                            need.add((p, int(x.t_close), x.c)); b += 1
                        if a < 3 and not x.in_state:
                            e2 = dict(verified=x.verified, hours=x.hours, in_state=x.in_state, above_streak=x.above_streak, vol_mult=x.vol_mult,
                                      last_bar_ts=x.bar_open, bar_red=x.bar_red, bar_ret_pct=x.bar_ret)
                            if frenzy_lite_status(e2, TH, x.vol24, None)[0]:
                                need.add((p, int(x.t_close), x.c)); a += 1
                    if (not same and x.t_close - r.t_close > 60 * MIN) or x.t_close - r.t_close > 20 * H:
                        break
    N = pd.DataFrame(sorted(need), columns=["pair", "t_close", "c"])
    N["key"] = N.pair + "|" + N.t_close.astype(str)
    return N


def step_price(nw):
    W = pd.read_pickle(f"{D}/walk.pkl")
    N = keys_needed(W)
    N.to_pickle(f"{D}/need.pkl")
    print("bars to price", len(N), flush=True)
    done = pd.read_pickle(f"{D}/priced.pkl") if os.path.exists(f"{D}/priced.pkl") else {}
    todo = [(r.key, r.pair, r.t_close, r.c) for r in N.itertuples() if r.key not in done]
    with Pool(nw) as p:
        for i, (k, o, e) in enumerate(p.imap_unordered(GR.price_any, todo, chunksize=4)):
            done[k] = o if o is not None else {"err": e}
            if i % 1000 == 0:
                print(i, len(todo), flush=True)
                pd.to_pickle(done, f"{D}/priced.pkl")
    pd.to_pickle(done, f"{D}/priced.pkl")
    print("priced", len(done), "errors", sum(1 for x in done.values() if "err" in x))


# ───────────── simulation ─────────────
def simulate(W, PA, BT, RF_, v):
    X = W.sort_values(["t_close", "pair"])
    oth = RF_[RF_.entry_strategy != "FRENZY_LITE"]
    occ = {p: g[["t_open", "t_shut"]].values for p, g in oth.groupby("pair")}
    wl = RF_[RF_.entry_strategy == "FRENZY_WILLY"][["t_open", "t_shut"]].values
    wl = wl[np.argsort(wl[:, 0])]
    fz = RF_[RF_.entry_strategy.isin(["FRENZY_LONG", "FRENZY_WIDE", "FRENZY_WILLY"])]
    fzday = fz.assign(day=pd.to_datetime(fz.t_open, unit="ms").dt.strftime("%Y-%m-%d")).groupby(["pair", "day"]).t_open.apply(list).to_dict()
    opens = []
    judged, hold_first, pd_count, fills, log, pend = {}, {}, {}, [], [], {}
    bt_idx = BT.index.values
    judge_once = v.get("judge_once", True)
    retry = "win" in v

    def btc_at(tc):
        j = int(np.searchsorted(bt_idx, tc, side="right")) - 1
        return BT.iloc[j] if j >= 0 else None

    def busy(pair, t):
        a = occ.get(pair)
        if a is not None and np.any((a[:, 0] <= t) & (t < a[:, 1])):
            return True
        return any(p == pair and x <= t < y for p, x, y in opens)

    def nslots(t):
        return sum(1 for p, x, y in opens if x <= t < y)

    def willy_hold(t):
        k = int(np.searchsorted(wl[:, 0], t, side="right"))
        return bool(k and np.any(wl[max(0, k - 20):k, 1] > t))

    def daycount(pair, te, dkey):
        return pd_count.get(dkey, 0) + sum(1 for x in fzday.get(dkey, []) if x < te)

    def try_open(r, how, filters=True, extra=None):
        te = int(r.t_close) + 8_000
        pr = PA.get(f"{r.pair}|{int(r.t_close)}")
        b = btc_at(r.t_close)
        if filters and b is not None and np.isfinite(b.d1) and np.isfinite(b.gap) and b.d1 < 0 and b.gap < 0:
            return "BEARISH_DAY"
        if busy(r.pair, te):
            return "PAIR_HELD"
        if nslots(te) >= 2:
            return "SLOTS"
        dkey = (r.pair, pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"))
        if filters and daycount(r.pair, te, dkey) >= 3:
            return "PAIR_DAY_CAP"
        if pr is None or "err" in pr:
            return "UNPRICED"
        if filters and pr.get("disloc"):
            return "DISLOCATION"
        slope = b.h1slope if b is not None else np.nan
        tp = 3 if (np.isfinite(r.vol_mult) and r.vol_mult >= 20 and np.isfinite(slope) and slope > 0) else 2
        pct, why, mn = pr[f"lite{tp}_pct"], pr[f"lite{tp}_why"], pr[f"lite{tp}_min"]
        opens.append((r.pair, te, te + int((mn if np.isfinite(mn) else 720) * MIN)))
        if len(opens) > 50:
            opens[:] = [x for x in opens if x[2] > te - 13 * H]
        pd_count[dkey] = pd_count.get(dkey, 0) + 1
        row = dict(pair=r.pair, t_close=int(r.t_close), te=te, spike_ts=int(r.spike_ts), hours=r.hours, vol_mult=r.vol_mult, vs_vwap=r.vs_vwap,
                   bar_ret=r.bar_ret, tp=tp, pct=pct, why=why, exit_min=mn, mfe=pr["mfe"], n_bar=pr["n_bar"], usd=pct / 100 * RC.NOTIONAL["FRENZY_LITE"],
                   how=how, btc_d1=(b.d1 if b is not None else np.nan), btc_gap=(b.gap if b is not None else np.nan), btc_h1slope=slope,
                   disloc=bool(pr.get("disloc", False)))
        if extra:
            row.update(extra)
        fills.append(row)
        return "OPEN"

    for r in X.itertuples():
        t = int(r.t_close)
        if retry and r.pair in pend:
            g = pend[r.pair]
            sid_now = GR.stretch_id(r)
            age = t - g["t"]
            if (v["win"] is not None and age > v["win"] * MIN) or (v["same"] and sid_now != g["sid"]) or age > 20 * H:
                pend.pop(r.pair)
                log.append(dict(pair=r.pair, t_close=t, code="RETRY_EXPIRED", stretch=g["sid"]))
            else:
                qual = (r.bar_ret is not None and r.bar_ret < 0) if v["strict"] else bool(r.bar_red)
                if qual:
                    ok = True
                    if v["recheck"] is True:
                        ep = dict(verified=r.verified, hours=r.hours, in_state=r.in_state, above_streak=r.above_streak, vol_mult=r.vol_mult,
                                  last_bar_ts=r.bar_open, bar_red=r.bar_red, bar_ret_pct=r.bar_ret)
                        ok = (not r.in_state) and bool(frenzy_lite_status(ep, TH, r.vol24, None)[0])
                    if ok:
                        if willy_hold(t + 8_000):
                            g["hold"] = g.get("hold", 0) + 1
                            if g["hold"] > 6:
                                pend.pop(r.pair); log.append(dict(pair=r.pair, t_close=t, code="RETRY_HOLD_EXPIRED", stretch=g["sid"]))
                        else:
                            pend.pop(r.pair)
                            res = try_open(r, "RETRY", filters=(v["recheck"] in (True, "open")),
                                           extra=dict(green_t=g["t"], green_c=g["c"], green_ret=g["ret"], wait_min=age / MIN,
                                                      move_from_green=(r.c / g["c"] - 1) * 100, same_stretch=(sid_now == g["sid"])))
                            log.append(dict(pair=r.pair, t_close=t, code="RETRY_" + res, stretch=g["sid"]))
            if r.pair in pend:
                continue
        if r.in_state or r.above_streak < 12:          # frenzy_lite_status would return IN_STATE / NOT_ABOVE (need = 12) — skip the call
            continue
        ep = dict(verified=r.verified, hours=r.hours, in_state=r.in_state, above_streak=r.above_streak, vol_mult=r.vol_mult,
                  last_bar_ts=r.bar_open, bar_red=r.bar_red, bar_ret_pct=r.bar_ret)
        ready, code, _ = frenzy_lite_status(ep, TH, r.vol24, judged.get(r.pair))
        if code not in FRENZY_LITE_JUDGES:
            continue
        sid = int(r.bar_open) - (int(r.above_streak) - 1) * MS5
        if ready and willy_hold(t + 8_000):
            hf = hold_first.get(r.pair)
            if hf is None or hf[0] != sid:
                hold_first[r.pair] = (sid, t)
            elif t - hf[1] > 6 * MS5:
                judged[r.pair] = sid; log.append(dict(pair=r.pair, t_close=t, code="HOLD_EXPIRED", stretch=sid))
                continue
            log.append(dict(pair=r.pair, t_close=t, code="WILLY_HOLD", stretch=sid))
            continue
        if code == FRENZY_LITE_READY:
            res = try_open(r, "LITE")
            if judge_once or res == "OPEN":
                judged[r.pair] = sid
        else:
            res = code
            if judge_once:
                judged[r.pair] = sid
            if retry:
                pend[r.pair] = dict(sid=sid, t=t, c=float(r.c), ret=float(r.bar_ret))
        log.append(dict(pair=r.pair, t_close=t, code=res, stretch=sid, hours=r.hours, vol_mult=r.vol_mult, bar_ret=r.bar_ret))
    return pd.DataFrame(fills), pd.DataFrame(log)


def step_sim():
    W = pd.read_pickle(f"{D}/walk.pkl"); PA = pd.read_pickle(f"{D}/priced.pkl"); BT = btc_frame(); RF_ = replay_fills()
    allf, alll = [], []
    for v in GR.VARIANTS:
        F, Lg = simulate(W, PA, BT, RF_, v)
        F["variant"] = v["name"]; Lg["variant"] = v["name"]
        allf.append(F); alll.append(Lg)
        print(v["name"], len(F), round(F.pct.mean(), 3), round(F.usd.sum()), flush=True)
    pd.concat(allf, ignore_index=True).to_pickle(f"{D}/bt_fills.pkl")
    pd.concat(alll, ignore_index=True).to_pickle(f"{D}/bt_log.pkl")
    # parity: V0 fills vs the replay's seed-1 LITE fills
    F0 = allf[0]; R = RF_[RF_.entry_strategy == "FRENZY_LITE"]
    a = set(zip(F0.pair, F0.t_close)); b = set(zip(R.pair, R.sig_close))   # sig_close = the signal bar CLOSE ms
    bj = set(zip(alll[0].pair, alll[0].t_close))
    print(f"PARITY replay LITE fills {len(b)} · found as V0 fills {len(a & b)} ({len(a & b) / max(1, len(b)):.3f}) · found as any V0 judged bar "
          f"{len(bj & b)} ({len(bj & b) / max(1, len(b)):.3f}) · V0 fills not in replay {len(a - b)}")


if __name__ == "__main__":
    s = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    {"walk": lambda: step_walk(nw), "price": lambda: step_price(nw), "sim": step_sim}[s]()
