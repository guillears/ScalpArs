#!/usr/bin/env python3
"""👁 Scout — G51_BAND1_MOVING watch line (pre-registered 2026-11-04, DECISION_LOG 463, operator "why do nothing?"; OBSERVE ONLY — never
changes config, no bot API; public 1m klines through path_labels' budgeted fetcher + the shared 1m cache).

QUESTION   Gate 51 band ① (btc_rsi_adx_filter_long "50-55:99-100" = every momentum LONG blocked while BTC 5m RSI is in [50, 55)) refuses
           signals that win when BTC is MOVING. Does band ① pay when BTC's trailing 24 h range (high / low − 1 over the 24 h before the signal)
           is ≥ 2.5 %? If yes, the next step is a probe door (band ① open only on a moving BTC, 1×) — never a config change from this line.
EVIDENCE   (in-sample, NEVER counted) B25 Oct-29 → Nov-4, signals refused by this gate ALONE (decisions FAILS set = bare
           MACRO:BTC_RSI_ADX_CROSS), band from the 14-day CloudWatch log: band ① 440 episodes · 66 % · +0.02 % (≈ breakeven); 24 h range
           < 1.5 % 58 % −0.06 · 1.5-2.5 59 % −0.05 · 2.5-4 71 % +0.08 · ≥ 4 74 % +0.11 (89 of the ≥ 4 % episodes came from ONE day, Oct-30).
           Real band-① fills while it was open (Aug-18 → Sep-15): 8 · 88 % · −$191 — seven wins of ~+0.1 % and one −1.19 % at 2× (龙虾,
           blacklisted). Band ③ (55-60) 502 · 55 % · −0.10 and 70+ 192 · 64 % · −0.01 → those blocks pay; context rows only. Phase 1 (Aug-17
           replay) and today's log replay: the same direction. BTC range is MARKET-WIDE (window-units rule): the bar counts UTC days.
COHORT     decisions exports (export_glob: Downloads + exports_archive/) — FAILS rows, src MOMENTUM, dir LONG, gate exactly
           MACRO:BTC_RSI_ADX_CROSS (the ladder passed; this gate alone refused it), stamped with the scan's BTC RSI / ADX in `reason` (the engine
           writes it since DECISION_LOG 463 — the exact values the gate read; a kline recomputation matches only ~80 %). Unstamped rows → UNSCORED.
           Episodes: one per pair per 45 min (the Aug-17 audit rule). COUNTED: bucket ≥ the DECISION_LOG 463 commit + 10 min (forward only).
PRICING    PROXY ruler (never the live exit — the audit's model-free win test): entry = the open of the first 1m bar at the end of the 5-minute
           bucket (no look-ahead); WIN = +0.40 % before −0.70 % within 360 min (a bar touching both = LOSS), else the 360-min close. Breakeven
           WR = 0.70 / 1.10 = 63.6 %. Both legs need the full 360 bars (else PENDING, fetched on later runs).
ZONE       band ① (RSI 50-55) ∧ BTC 24 h range ≥ 2.5 % (FROZEN, never re-fit) · REST = band ① ∧ range < 2.5 %.
FROZEN BAR at the first prefix (by bucket time) where the zone reaches N ≥ 30 on ≥ 8 distinct UTC days, decided ONCE — ① zone avg ≤ 0 or
           ≤ rest avg → RETIRED; ② PROBE-DOOR CANDIDATE (operator decides; never automatic) iff zone WR ≥ 63.6 % ∧ day-clustered bootstrap
           P(mean < 0) ≤ 0.05 (4,000 resamples, seed 7) ∧ no single day and no single pair carries ≥ 50 % of the zone's gross gain; ③ else
           KEEP OBSERVING. Deferred while any zone episode of the prefix is PENDING. A probe door ships only after a before → after per batch
           with the daily compound return, the 30-50 % haircut and a pre-committed revert (its first 10 fills: WR < 61 % or Σ < 0 → closed).
STATE      reports/SCOUT_G51_BAND1.json (gitignored).
Usage: venv/bin/python -I scripts/scout_g51_band1_watch.py
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (day-bootstrap, state I/O)

DL = 463
GATE = "MACRO:BTC_RSI_ADX_CROSS"
BAND = (50.0, 55.0)
RANGE_CUT = 2.5
UP, DN, HOR_MIN = 0.40, -0.70, 360
BE_WR = 0.70 / 1.10
EPISODE_MIN = 45
NODATA_AFTER_MIN = 3 * 1440                                 # still PENDING 3 days after its horizon (exchange gap / delisted) → NODATA, never counted
N_MIN, DAYS_MIN = 30, 8
P_MAX, SHARE_MAX = 0.05, 0.50
BOOT_N, BOOT_SEED = 4000, 7
MIN = 60_000
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_G51_BAND1.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")
CACHE_VER = "v1"                                           # bump when the row filter changes → old per-export caches are never reused
CACHE_DIR = os.path.join(_ROOT, "reports", "backtest_cache", "scout_g51_band1")   # gitignored (reports/backtest_cache/)
COLS = ("t", "e", "pair", "dir", "gate", "src", "reason")
CANDIDATE = "PROBE-DOOR CANDIDATE (operator decides; never automatic)"
_RX = re.compile(r"btc_rsi=([-0-9.]+)\s+btc_adx=([-0-9.]+)")


def parse_reason(s):
    """'btc_rsi=52.10 btc_adx=31.90' → (52.1, 31.9); anything else → (nan, nan). PURE."""
    m = _RX.search(str(s)) if isinstance(s, str) else None
    return (float(m[1]), float(m[2])) if m else (float("nan"), float("nan"))


def band_of(rsi):
    """the gate-51 LONG band a BTC RSI falls in (labels only; the zone uses BAND). PURE."""
    if rsi != rsi:
        return "unscored"
    if BAND[0] <= rsi < BAND[1]:
        return "① 50-55"
    if 55.0 <= rsi < 60.0:
        return "③ 55-60"
    return "70+" if rsi >= 70.0 else "other"


def _read_export(f, cache_dir=CACHE_DIR):
    """one decisions export → its sole-refused momentum LONG rows (COLS, str). Parsed once: the small result is cached per export
    (name + size + whole-second mtime) — a 60 MB export is never re-read on later runs. cache_dir None → no cache. None when unreadable."""
    key = None
    if cache_dir:
        try:
            st = os.stat(f)
            key = os.path.join(cache_dir, f"{os.path.basename(f)}.{st.st_size}.{int(st.st_mtime)}.{CACHE_VER}.csv")
            if os.path.exists(key):
                return pd.read_csv(key, dtype=str, keep_default_na=False).reindex(columns=list(COLS))
        except Exception:
            key = None
    try:
        d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS, dtype=str)
    except Exception:
        return None
    if not {"t", "e", "pair", "dir", "gate", "src"} <= set(d.columns):
        return None
    d = d[((d.e == "FAILS") & (d.dir == "LONG") & (d.src == "MOMENTUM") & (d.gate == GATE)) | (d.e == "SCAN")].reindex(columns=list(COLS))
    if key:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = f"{key}.{os.getpid()}.tmp"
            d.to_csv(tmp, index=False)
            os.replace(tmp, key)
        except Exception:
            pass
    return d


def load(paths=None, since=None, cache_dir=CACHE_DIR):
    """→ sole-refused momentum LONG rows (t, pair, rsi, adx), one per (t, pair) — a stamped row wins. Exports stamped before `since` skipped."""
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        s = _EG.export_stamp(f)
        if since is not None and s:
            try:
                if pd.Timestamp(s) < since:
                    continue
            except ValueError:
                pass
        d = _read_export(f, cache_dir)
        if d is not None:
            fr.append(d)
    if not fr:
        return pd.DataFrame(columns=["t", "pair", "rsi", "adx"])
    d = pd.concat(fr, ignore_index=True)
    d = d[d.e != "SCAN"]
    d["reason"] = d.reason.replace("", np.nan)
    d["t"] = pd.to_datetime(d.t, format="ISO8601", errors="coerce")
    d = d[d.t.notna()]
    rr = [parse_reason(x) for x in d.reason.values]
    d["rsi"], d["adx"] = [a for a, _ in rr], [b for _, b in rr]
    d = d.sort_values(["t", "pair", "rsi"], na_position="first").drop_duplicates(["t", "pair"], keep="last")
    return d[["t", "pair", "rsi", "adx"]].reset_index(drop=True)


def scan_buckets(paths=None, since=None, cache_dir=CACHE_DIR):
    """→ the set of 5-minute buckets (Timestamps) the exports' SCAN heartbeats cover — journal coverage, so a day nobody exported is seen."""
    out = set()
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        s_ = _EG.export_stamp(f)
        if since is not None and s_:
            try:
                if pd.Timestamp(s_) < since:
                    continue
            except ValueError:
                pass
        d = _read_export(f, cache_dir)
        if d is None or not len(d):
            continue
        t = pd.to_datetime(d[d.e == "SCAN"].t, format="ISO8601", errors="coerce").dropna()
        out |= set(t.dt.floor("5min"))
    return out


def coverage_gaps(days, buckets, start, end, need=0.90):
    """UTC days whose [max(day, start), min(day end, end)) window has < need of its 5-minute buckets in `buckets`. PURE."""
    bad = []
    for d in sorted(set(days)):
        lo = max(pd.Timestamp(d), start.floor("5min"))
        hi = min(pd.Timestamp(d) + pd.Timedelta(days=1), end)
        want = pd.date_range(lo, hi, freq="5min", inclusive="left")
        if len(want) and sum(1 for x in want if x in buckets) / len(want) < need:
            bad.append(d)
    return bad


def episodes(rows, gap_min=EPISODE_MIN):
    """one row per pair per gap_min (from the episode's first bucket), across ALL bands — the Aug-17 audit rule (a band-③ refusal can
    therefore absorb a band-① one of the same pair a few minutes later). PURE."""
    keep, last = [], {}
    for r in rows.sort_values(["t", "pair"]).itertuples(index=False):
        p = last.get(r.pair)
        if p is not None and r.t - p < pd.Timedelta(minutes=gap_min):
            continue
        last[r.pair] = r.t
        keep.append(r)
    return pd.DataFrame(keep, columns=rows.columns)


def walk(bars, entry_ms, up=UP, dn=DN, hor=HOR_MIN):
    """bars (open_time, o, h, l, c) → (result W / L / F / PENDING, pct). Entry = the open of the bar at entry_ms. PURE."""
    if bars is None or not len(bars):
        return "PENDING", None
    b = bars[(bars.open_time >= entry_ms) & (bars.open_time < entry_ms + hor * MIN)]
    if not len(b) or int(b.open_time.iloc[0]) != int(entry_ms):
        return "PENDING", None
    e = float(b.o.iloc[0])
    if e != e:                                            # caches written before Oct-10 carry no open → the previous bar's close
        prev = bars[bars.open_time == entry_ms - MIN]
        if not len(prev):
            return "PENDING", None
        e = float(prev.c.iloc[0])
    ot = b.open_time.values.astype("int64")
    for i, (h, lo) in enumerate(zip(b.h.values, b.l.values)):
        if int(ot[i]) != int(entry_ms) + i * MIN:          # (463 review) a cache gap before a hit → PENDING (fetched), never a W / L across it
            return "PENDING", None
        if (lo / e - 1) * 100 <= dn:
            return "L", dn
        if (h / e - 1) * 100 >= up:
            return "W", up
    if len(b) < hor:
        return "PENDING", None
    return "F", round((float(b.c.iloc[-1]) / e - 1) * 100, 4)


def range24(btc, t_ms):
    """BTC (high / low − 1) % over the 1m bars of the 24 h before t_ms; None when < 95 % of them are cached. PURE."""
    if btc is None or not len(btc):
        return None
    w = btc[(btc.open_time >= t_ms - 1440 * MIN) & (btc.open_time < t_ms)]
    if len(w) < 0.95 * 1440:
        return None
    return round((float(w.h.max()) / float(w.l.min()) - 1) * 100, 4)


def price(ep, now_ms, fetch=True, budget=None, notes=None, attempts=None):
    """episodes → + range24, res, pct. Missing 1m windows fetched within the budget (path_labels' fetcher, shared cache)."""
    import path_labels as PL
    import scout_willy_timecap as WT
    notes = notes if notes is not None else []
    bud = budget if budget is not None else {}
    bud.setdefault("req", 0)
    bud.setdefault("used", 0)
    bud.setdefault("t0", time.monotonic())
    attempts = attempts if attempts is not None else {}
    out = []
    if not len(ep):
        return ep.assign(range24=[], res=[], pct=[])
    t_ms = [int(pd.Timestamp(t).value // 1_000_000) for t in ep.t]
    if fetch and not bud.get("stopped"):
        PL._fetch_window(WT, "BTCUSDT", min(t_ms) - 1440 * MIN, max(t_ms) + 5 * MIN, now_ms, bud, attempts)
    btc = WT.load_1m("BTCUSDT")
    for r, tm in zip(ep.itertuples(index=False), t_ms):
        e_ms = tm + 5 * MIN                                   # end of the 5-minute bucket: no look-ahead inside it
        res, pct = walk(WT.load_1m(r.pair), e_ms)
        if res == "PENDING" and fetch and not bud.get("stopped") and e_ms + HOR_MIN * MIN <= now_ms:
            if PL._fetch_window(WT, r.pair, e_ms, e_ms + HOR_MIN * MIN, now_ms, bud, attempts):
                res, pct = walk(WT.load_1m(r.pair), e_ms)
        if res == "PENDING" and now_ms - (e_ms + HOR_MIN * MIN) > NODATA_AFTER_MIN * MIN:
            res = "NODATA"
        out.append((range24(btc, tm), res, pct))
    if bud.get("stopped"):
        notes.append(f"fetching stopped: {bud['stopped']} — the rest catches up on later runs")
    e2 = ep.reset_index(drop=True).copy()
    e2["range24"], e2["res"], e2["pct"] = [a for a, _, _ in out], [b for _, b, _ in out], [c for _, _, c in out]
    return e2


def classify(ep):
    """+ band, day, zone (zone / rest / other band / unscored). PURE."""
    e = ep.copy()
    e["band"] = [band_of(x) for x in e.rsi]
    e["day"] = pd.to_datetime(e.t).dt.strftime("%Y-%m-%d")
    rng = pd.to_numeric(e.range24, errors="coerce")
    e["zone"] = np.where(e.band != "① 50-55", e.band, np.where(rng.isna(), "unscored", np.where(rng >= RANGE_CUT, "zone", "rest")))
    return e


def gain_share(pct, key):
    """the largest key's share of the summed WINNING-episode gain (losers ignored). PURE."""
    d = pd.DataFrame(dict(p=np.asarray(pct, dtype=float), k=np.asarray(key)))
    w = d[d.p > 0]
    tg = w.p.sum()
    return float(w.groupby("k").p.sum().max() / tg) if tg > 0 else 0.0


def verdict(zone, rest):
    """zone / rest = resolved episodes (pct, day, pair) → (state, detail). PURE."""
    n, nd = len(zone), zone.day.nunique() if len(zone) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"zone N {n}/{N_MIN} · {nd}/{DAYS_MIN} days"
    mz = float(zone.pct.mean())
    mr = float(rest.pct.mean()) if len(rest) else float("nan")
    wr = float((zone.res == "W").mean()) if "res" in zone else float((zone.pct > 0).mean())   # wins = the +0.40 hit (BE_WR's own ruler)
    p = NF.p_mean_neg(zone.pct.values, zone.day.values, BOOT_N, BOOT_SEED)
    sd, sp = gain_share(zone.pct.values, zone.day.values), gain_share(zone.pct.values, zone.pair.values)
    det = (f"zone N {n} · {nd} d · WR {wr:.0%} (breakeven {BE_WR:.1%}) · avg {mz:+.3f} % vs rest {mr:+.3f} % (N {len(rest)}) · "
           f"P(mean<0) {(p if p is not None else float('nan')):.3f} · top day {sd:.0%} / top pair {sp:.0%} of the zone's gross gain")
    if mz <= 0 or (len(rest) and mz <= mr):
        return "RETIRED", det
    if wr >= BE_WR and p is not None and p <= P_MAX and sd < SHARE_MAX and sp < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def crossing_prefix(sc):
    """classified episodes by time → the prefix ending at the zone episode where the zone first reaches N ≥ 30 on ≥ 8 days, else None."""
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
    r = g[g.res.isin(["W", "L", "F"])] if len(g) else g
    if not len(r):
        return f"| {lab} | {len(g)} | – | – | – | – |"
    pen = int((g.res == "PENDING").sum())
    return (f"| {lab} | {len(r)}{f' (+{pen} pending)' if pen else ''} | {r.day.nunique()} | {(r.res == 'W').mean():.0%} | "
            f"{r.pct.mean():+.3f} % | {r.pct.sum():+.1f} % |")


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, fetch=True, budget=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 463 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 G51_BAND1_MOVING — momentum LONGs refused ONLY by gate-51 band ① (BTC RSI 50-55), split by BTC 24 h range ≥ {RANGE_CUT:g} % "
         f"(pre-registered 2026-11-04, DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms is not None
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="G51_BAND1_MOVING watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    notes = []
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    attempts = st.setdefault("_attempts", {}) if ok and isinstance(st.get("_attempts", {}), dict) else {}
    rows = load(paths, since=since.floor("D"))
    rows = rows[rows.t >= since] if len(rows) else rows
    ep = episodes(rows) if len(rows) else rows.assign(range24=[], res=[], pct=[])
    ep = classify(price(ep, now_ms, fetch=fetch, budget=budget, notes=notes, attempts=attempts)) if len(ep) else \
        pd.DataFrame(columns=["t", "pair", "rsi", "adx", "range24", "res", "pct", "band", "day", "zone"])
    done = ep[ep.res.isin(["W", "L", "F"])] if len(ep) else ep
    zone, rest = done[done.zone == "zone"], done[done.zone == "rest"]
    L += [f"Cohort: sole-refused momentum LONG signals (decisions FAILS = bare {GATE}) since {since:%Y-%m-%d %H:%M} UTC "
          f"({'floor given' if given else f'DL {DL} commit + 10 min'}) · one per pair per {EPISODE_MIN} min · PROXY ruler +{UP:.2f} before "
          f"{DN:.2f} in {HOR_MIN} min (breakeven WR {BE_WR:.1%}) · N = refused SIGNALS, not trades (5 slots — the bot takes a few).", "",
          "| Group | N | days | WR | avg % | Σ % |", "|---|---|---|---|---|---|",
          _row(f"**zone: band ① ∧ BTC 24 h range ≥ {RANGE_CUT:g} %**", ep[ep.zone == "zone"] if len(ep) else ep),
          _row(f"rest: band ① ∧ range < {RANGE_CUT:g} %", ep[ep.zone == "rest"] if len(ep) else ep),
          _row("context: band ③ 55-60", ep[ep.zone == "③ 55-60"] if len(ep) else ep),
          _row("context: band 70+", ep[ep.zone == "70+"] if len(ep) else ep),
          _row("context: other RSI (< 50 or 60-70)", ep[ep.zone == "other"] if len(ep) else ep),
          _row("UNSCORED (no RSI stamp / BTC bars) — never counted", ep[ep.zone == "unscored"] if len(ep) else ep),
          f"NODATA (no 1m path 3 days after the horizon — never counted): {int((ep.res == 'NODATA').sum()) if len(ep) else 0}", ""]
    rec = st.get("BAND1") if ok else None
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        rec, ok = None, False          # a damaged record never re-freezes silently
    why = []
    if not rec and ok and len(ep):
        sc = ep[ep.zone.isin(["zone", "rest"]) & (ep.res != "NODATA")].sort_values(["t", "pair"])
        pre = crossing_prefix(sc)
        if pre is not None:
            end = pd.Timestamp(pre.t.max())
            b1 = ep[(ep.band == "① 50-55") & (pd.to_datetime(ep.t) <= end)]
            gaps = coverage_gaps(pre.day, scan_buckets(paths, since=since.floor("D")), since, end + pd.Timedelta(minutes=5))
            if (pre.res == "PENDING").any():
                why.append("a prefix episode is still PENDING (its 360-min path is not cached yet)")
            elif b1.range24.isna().any():
                why.append("a band-① episode before the crossing has no BTC 24 h range yet (BTC bars not cached)")
            elif gaps:
                why.append(f"decisions exports cover < 90 % of the 5-minute buckets on {', '.join(gaps)} (download a fresh export)")
            else:
                state, det = verdict(pre[pre.zone == "zone"], pre[pre.zone == "rest"])
                last = pd.Timestamp(pre.t.max())
                rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                           keys=[f"{pd.Timestamp(a):%Y-%m-%dT%H:%M}|{b}|{g}" for a, b, g in zip(pre.t, pre.pair, pre.zone)])
                if state_path:
                    st["BAND1"] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(zone, rest)
    if rec and ok and state_path:
        L.append(f"**Frozen verdict (crossing at {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if why:
        L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
    if ok and state_path and attempts:
        try:
            NF.du_save_state(st, state_path)               # fetch attempts persist (an exchange gap is asked at most twice, ever)
        except Exception:
            pass
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(str(state_path))}.*.bad); freezing skipped this run.")
    L += [f"Note: {n}" for n in notes]
    L += ["", "Evidence (in-sample, NEVER counted): B25 Oct-29 → Nov-4 band ① 440 · 66 % · +0.02 % — range < 2.5 % 204 · 59 % · −0.05 vs ≥ 2.5 % "
          "235 · 72 % · +0.09 (89 of the ≥ 4 % rows = Oct-30 alone); real band-① fills Aug-18 → Sep-15 8 · 88 % · −$191 (one 2× −1.19 %).",
          f"FROZEN bar (DL {DL}): first prefix with zone N ≥ {N_MIN} on ≥ {DAYS_MIN} UTC days, decided ONCE — zone avg ≤ 0 or ≤ rest → RETIRED; "
          f"{CANDIDATE} iff WR ≥ {BE_WR:.1%} ∧ day-bootstrap P(mean < 0) ≤ {P_MAX:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ no day / pair ≥ "
          f"{SHARE_MAX:.0%} of the gross gain; else KEEP OBSERVING. A probe door needs a before → after per batch with the daily compound "
          "return, the 30-50 % haircut and its pre-committed revert (first 10 fills: WR < 61 % or Σ < 0 → closed).", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
