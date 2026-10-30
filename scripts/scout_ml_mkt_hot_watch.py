#!/usr/bin/env python3
"""👁 Scout — ML_MKT_HOT watch line (pre-registered 2026-10-30, DECISION_LOG 416; OBSERVE ONLY — never changes config, no bot API; public 1h klines only).

QUESTION   Are full-size momentum longs bought when most of the market has already risen over 4 hours flat or worse, while those bought in a soft market earn? Theory (post-hoc:
           the frozen pre-study hypothesis H5 said breadth 50–80 % would be BETTER): with most coins already up we buy late, with few buyers left.
EVIDENCE   (in-sample, NEVER counted) 180 full-size fills, master on today's stack + B25: zone 83 on 41 days · 64 % · −0.03 %/fill vs 97 · 78 % · +0.30, day-bootstrap P(zone < rest) 0.99. Master 157 · 71 % ·
           +0.13 → 84 · 76 % · +0.24, daily compound +0.155 → +0.156 (each trade = account); B25 23 · 74 % · +0.30 → 13 · 92 % · +0.71, +0.95 → +1.29. Backtest yr6 2026 at the 0.48 cut: seeds 1 / 2 / 3 zone −0.11 / −0.10 / −0.10 vs
           −0.03 / −0.01 / −0.04, daily compound improves in all 3. AGAINST: the 322-split family test does not beat shuffled data (p ≈ 0.65); clearer in the later half of the data; only about half the batches lean this way.
OTHER      (operator 10-30) the same split is also read on FRENZY_WILLY and FRENZY_WILLY_HT, each decided SEPARATELY on the same bar, and shown for FRENZY_LITE (DISPLAY ONLY, never decided: its master, B25 and backtest disagree). In-sample, hot vs not: WILLY 26 · 65 % · −0.30 vs 27 · 78 % · +0.54
           (backtest REVERSED: +0.28 vs −0.09) · WILLY_HT 6 · 50 % · −0.49 vs 6 · 100 % · +1.92 (B25 only) · LITE 24 · 54 % · +0.02 vs 17 · 76 % · +1.02 (backtest reversed: +0.28 vs −0.03). NOT read: FRENZY_LONG (master re-priced at today's −4 stop, hot 17 · 82 % · +0.53 vs 16 · 88 % · +0.81, still earning, blocking lowers compound; at the old −3 it was −0.32 vs +0.60) and LITE_RETRY (hot is the better side).
COHORT     CLOSED LONG fills of the read sleeves, full size (*_PROBE out), from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN, newest export wins; operator-closed out
           (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 416 commit + 10 min (forward only). breadth = the share of the frozen UNIVERSE (108 coins, the most frequent top-50 names at registration, BTC / ETH out) whose 1h close is above its
           close 4 h earlier, on the last closed 1h bar at the entry, ≥ 30 coins with data. zone = breadth > 0.50 · rest = ≤ 0.50 · no breadth → UNSCORED (never counted). P&L = the booked pnl_percentage.
FROZEN BAR (🔒 OPERATOR 10-30, DECLARED: "we block negative or breakeven … we should target much higher than breakeven" — deliberately NOT the CLAUDE.md expectancy bar, whose WR-below-breakeven and P(mean < 0) legs this line does not use): at the first prefix (by open
           time) where the ZONE reaches N ≥ 15 on ≥ 8 distinct UTC days, decided ONCE — ① zone avg ≥ rest avg → RETIRED; ② FILTER CANDIDATE (operator decides; never automatic) iff zone avg ≤ +0.05 %/fill ∧ day-clustered bootstrap P(zone avg < rest avg) ≥ 0.90 (4,000 resamples,
           seed 7) ∧ no single day and no single pair carries ≥ 50 % of the zone's gross loss; ③ else KEEP OBSERVING. Deferred until the newest export was downloaded ≥ the prefix end + 1,200 min, no unscored fill sits inside the prefix and the breadth fetch completed. The read shows before → after
           with the daily compound return. A live rule also needs the engine to compute breadth (not stamped today), the 30–50 % in-sample haircut and a pre-committed revert.
STATE      reports/SCOUT_ML_MKT_HOT.json (gitignored).
Usage: venv/bin/python -I scripts/scout_ml_mkt_hot_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (gross-loss share, state I/O)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)

DL = 416
HOT_CUT = 0.50
DECIDED = (("ML", "MOMENTUM"), ("FRENZY_WILLY", "FRENZY_WILLY"), ("FRENZY_WILLY_HT", "FRENZY_WILLY_HT"))   # (state key, entry_strategy), each decided on its own
DISPLAY = (("FRENZY_LITE", "FRENZY_LITE"),)                                                                # shown, never frozen
MIN_UNIVERSE = 97                  # a run with fewer universe coins (90 %) is incomplete: its breadth differs from the registered measure
MIN_COINS = 30
N_MIN, DAYS_MIN = 15, 8
AVG_MAX, P_MIN, SHARE_MAX = 0.05, 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = 1200
PAGE_BARS = 720
HOUR = 3_600_000
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_ML_MKT_HOT.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_ml_mkt_hot")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "close_reason", "cell_multiplier_source")
CANDIDATE = "FILTER CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): momentum longs zone 83 on 41 days · 64 % · −0.03 vs 97 · 78 % · +0.30, P(zone < rest) 0.99; master compound +0.155 → +0.156, B25 +0.95 → +1.29; backtest 3 seeds zone worse; "
                "family test not above luck; WILLY −0.30 vs +0.54 and LITE +0.02 vs +1.02 with the backtest REVERSED; WILLY_HT −0.49 vs +1.92 on 12 fills")
UNIVERSE = tuple((
    "1000BONKUSDT 1000PEPEUSDT 1000SHIBUSDT AAVEUSDT ACEUSDT ADAUSDT AKEUSDT ALLOUSDT APTUSDT ARBUSDT ARIAUSDT ASTERUSDT AVAXUSDT AXSUSDT BANKUSDT BCHUSDT "
    "BEATUSDT BERAUSDT BILLUSDT BNBUSDT BRUSDT BSBUSDT BTRUSDT BTWUSDT BULLAUSDT BUSDT CHIPUSDT CLOUSDT CRVUSDT CYSUSDT DASHUSDT DEXEUSDT DOGEUSDT DOTUSDT "
    "EDGEUSDT ENAUSDT ENSOUSDT ESPORTSUSDT FARTCOINUSDT FETUSDT FILUSDT GIGGLEUSDT HBARUSDT HEIUSDT HOMEUSDT HUSDT HYPEUSDT ICPUSDT INJUSDT IPUSDT JTOUSDT "
    "KAITOUSDT KITEUSDT LABUSDT LINKUSDT LITUSDT LTCUSDT LYNUSDT MAGMAUSDT NEARUSDT NOMUSDT ONDOUSDT ONUSDT OPNUSDT OPUSDT PAXGUSDT PENGUUSDT PIPPINUSDT "
    "PLAYUSDT PONSUSDT POWERUSDT PUMPUSDT RAVEUSDT RENDERUSDT REUSDT RIVERUSDT SAHARAUSDT SIRENUSDT SKYAIUSDT SOLUSDT SUIUSDT SYNUSDT TAOUSDT TONUSDT "
    "TRUMPUSDT TRXUSDT TUTUSDT UAIUSDT UBUSDT UNIUSDT USELESSUSDT USUSDT VELVETUSDT VIRTUALUSDT VVVUSDT WIFUSDT WLDUSDT WLFIUSDT XAUTUSDT XLMUSDT XMRUSDT "
    "XPLUSDT XRPUSDT ZAMAUSDT ZECUSDT ZROUSDT 币安人生USDT 龙虾USDT "
).split())


def load(paths=None):
    """→ (closed full-size fills of the read sleeves — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
            read.append(f)
    newest = None
    for f in read:
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str).isin([e for _, e in DECIDED + DISPLAY]) & (d.direction.astype(str) == "LONG")
          & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def breadth_series(closes):
    """DataFrame coin × 1h bar-CLOSE time → the share above their close 4 h earlier (bars with ≥ MIN_COINS coins with data). PURE."""
    r4 = closes / closes.shift(4) - 1
    n = r4.notna().sum(axis=1)
    return ((r4 > 0).sum(axis=1) / n)[n >= MIN_COINS]


def fetch_closes(t_min, now_ms, bud=None):
    """1h closes of the UNIVERSE from t_min − 8 h, public klines in fixed pages (complete pages are cached). → (closes indexed by bar close, complete?)."""
    bud = bud or LS.Budget()
    t0 = (int(t_min) // HOUR - 8) * HOUR
    out, complete = {}, True
    for s in UNIVERSE:
        rows, t = [], t0
        while t < now_ms:
            pg = LS.fapi_klines(s, "1h", t, PAGE_BARS, bud, CACHE, now_ms, fetch=True)
            if pg is None or getattr(bud, "errors", 0):       # a transient error's [] is not "no data": the run is incomplete, stop asking
                complete = False
                break
            rows += pg
            if len(pg) < PAGE_BARS:
                break
            t += PAGE_BARS * HOUR
        if not complete:
            break
        if rows:
            out[s] = pd.Series({pd.Timestamp(int(q[0]) + HOUR, unit="ms"): float(q[4]) for q in rows})
    return (pd.DataFrame(out).sort_index() if out else pd.DataFrame()), complete and len(out) >= MIN_UNIVERSE


def score(c, breadth):
    """fills + the breadth series → zone / rest / unscored on the last closed bar at each entry (a bar more than 2 h old = no reading). PURE."""
    c = c.copy()
    c["breadth"] = np.nan
    if len(c) and len(breadth):
        idx = breadth.index.values.astype("datetime64[ns]")
        o = c.o.values.astype("datetime64[ns]")
        i = np.searchsorted(idx, o, side="right") - 1
        ok = (i >= 0) & ((o - idx[np.clip(i, 0, len(idx) - 1)]) <= np.timedelta64(2, "h"))
        c.loc[ok, "breadth"] = breadth.values[i[ok]]
    c["zone"] = np.where(c.breadth.isna(), "unscored", np.where(c.breadth > HOT_CUT, "zone", "rest"))
    return c


def p_zone_below(zone, rest, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap P(zone avg < rest avg), resampling UTC days across both groups."""
    days = sorted(set(zone.day) | set(rest.day))
    z = {d: zone.pct.values[zone.day.values == d] for d in days}
    r = {d: rest.pct.values[rest.day.values == d] for d in days}
    rng, hit, tot = np.random.default_rng(seed), 0, 0
    for _ in range(n):
        pick = [days[i] for i in rng.integers(0, len(days), len(days))]
        a, b = np.concatenate([z[d] for d in pick]), np.concatenate([r[d] for d in pick])
        if len(a) and len(b):
            tot += 1
            hit += a.mean() < b.mean()
    return hit / tot if tot else None


def verdict(zone, rest):
    """zone / rest = counted fills (pct, day, pair) → (state, detail). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz, mr = float(zone.pct.mean()), (float(rest.pct.mean()) if len(rest) else float("nan"))
    p = p_zone_below(zone, rest) if len(rest) else None
    sd, sp = NF.gross_loss_share(zone.pct.values, zone.day.values), NF.gross_loss_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {(zone.pct > 0).mean():.0%} · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · P(zone < rest) "
           f"{(p if p is not None else float('nan')):.2f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross loss")
    if len(rest) and mz >= mr:
        return "RETIRED", det
    if mz <= AVG_MAX and p is not None and p >= P_MIN and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def before_after(sc):
    """before → after blocking the zone: N · WR · avg · the daily compound return with each trade = the account (comparison only). PURE."""
    if not len(sc):
        return "no scored fills"
    span = pd.date_range(sc.o.min().normalize(), sc.o.max().normalize(), freq="D").strftime("%Y-%m-%d")

    def dcr(g):
        return (((1 + (g.groupby("day").pct.sum() / 100).reindex(span).fillna(0)).prod()) ** (1 / len(span)) - 1) * 100
    k = sc[sc.zone == "rest"]
    a = f"{len(k)} · {(k.pct > 0).mean():.0%} · {k.pct.mean():+.3f} % · {dcr(k):+.3f} %/day" if len(k) else "0 fills"
    return f"before {len(sc)} · {(sc.pct > 0).mean():.0%} · {sc.pct.mean():+.3f} % · {dcr(sc):+.3f} %/day → after {a}"


def crossing_prefix(sc):
    """scored counted fills by open time → the prefix ending at the zone fill where the zone first reaches N ≥ 15 on ≥ 8 days."""
    seen, nz = set(), 0
    for k, (g, d) in enumerate(zip(sc.zone.values, sc.day.values), 1):
        if g != "zone":
            continue
        nz += 1
        seen.add(d)
        if nz >= N_MIN and len(seen) >= DAYS_MIN:
            return sc.iloc[:k]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | "
            f"{pd.to_numeric(g.pnl, errors='coerce').sum():+,.0f} |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, breadth_fn=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 416 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write.
    breadth_fn(t_min_ms, now_ms) → (breadth series, complete?); None → public klines."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 ML_MKT_HOT — bought with > {HOT_CUT:.0%} of the market up over 4 h: momentum longs, WILLY, WILLY_HT (LITE shown) (pre-registered 2026-10-30, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="ML_MKT_HOT watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    complete = True
    if len(keep):
        t_min = int(keep.o.min().value // 10**6)
        if breadth_fn is None:
            closes, complete = fetch_closes(t_min, now_ms)
            br = breadth_series(closes) if len(closes) and complete else pd.Series(dtype=float)   # a partial universe is not the registered measure → unscored
        else:
            br, complete = breadth_fn(t_min, now_ms)
        keep = score(keep, br)
    else:
        keep = keep.assign(zone=pd.Series(dtype=object), breadth=pd.Series(dtype=float))
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    L += [f"Cohort: CLOSED full-size fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · each sleeve read separately · "
          f"{_GK.operator_closed_line(ox)}.", ""]
    if not complete:
        L += ["⚠ breadth fetch incomplete this run (budget, cooldown or too few coins): fills are unscored until a later run; nothing freezes.", ""]
    changed = False
    for key, es in DECIDED + DISPLAY:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        sc = s[s.zone != "unscored"] if len(s) else s
        zone, rest, uns = sc[sc.zone == "zone"], sc[sc.zone == "rest"], s[s.zone == "unscored"]
        shown = (key, es) in DISPLAY
        L += [f"### {'MOMENTUM LONG' if es == 'MOMENTUM' else es}" + (" (display only, never decided)" if shown else ""), "", "| Group | N | days | WR | avg % | Σ$ as traded |", "|---|---|---|---|---|---|",
              _row(f"**zone (market hot, > {HOT_CUT:.0%} up)**", zone), _row("rest", rest), _row("UNSCORED (no breadth) — never counted", uns), ""]
        why = []
        rec = st.get(key) if (ok and not shown) else None
        damaged = rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec))
        rec = None if damaged else rec     # a damaged record never re-freezes silently and blocks only its own sleeve
        if damaged:
            L.append(f"⚠ {key}: frozen record damaged — operator restore needed; this sleeve does not freeze until then.")
        if not rec and ok and not shown and not damaged:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if not complete:
                    why.append("the breadth fetch stopped early this run")
                elif (uns.o <= last).any():
                    why.append("an unscored fill sits inside the prefix")
                elif newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                    why.append(f"newest export < the crossing fill + {WAIT_MIN} min (a fill could still be open)")
                else:
                    state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                    rec = dict(state=state, detail=det, before_after=before_after(pre), at=f"{last:%Y-%m-%d %H:%M} UTC",
                               run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                               keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{g}" for a, b, g in zip(pre.o, pre.pair, pre.zone)])
                    if state_path:
                        st[key] = rec
                        changed = True
        live_state, live_det = verdict(zone, rest)
        if rec and ok and state_path:
            L.append(f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
            L.append(f"Before → after at the crossing: {rec.get('before_after', '–')}")
            L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
        else:
            L.append(f"Now: ⏳ {live_state} ({live_det})." if not shown else f"Now: {live_det}.")
        L.append("Before → after now: " + before_after(sc) + ".")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        L.append("")
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, operator's, not the expectancy bar): at the first prefix where the zone reaches N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — zone avg ≥ rest avg → RETIRED; {CANDIDATE} iff zone avg ≤ {AVG_MAX:+.2f} % ∧ "
          f"P(zone < rest) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ {SHARE_MAX:.0%} of the zone's gross loss; else KEEP OBSERVING.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
