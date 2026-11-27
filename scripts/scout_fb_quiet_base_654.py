#!/usr/bin/env python3
"""🤫 Scout — FLATBASE QUIET BASE observe line (DECISION_LOG 654, registered 2026-11-27, operator "yes build the observe line"; OBSERVE ONLY —
never changes config, no bot API).

WHY        operator 11-27: "it was supposed to short a coin that pumped, that was previously FLAT … see if the trigger is correct". Study
           reports/FLATBASE_THEORY_2026-11-27.md: today's FLAT test (14-day close range ≤ 44.4 %) does not measure flatness; the base's DAILY
           VOLATILITY does — backtest 545 kept fills, terciles +1.79 (81 % WR) / +1.13 / +0.48 %/fill, both blocks, 5 / 5 half-years, family-wise
           luck p 0.12; real master + batch closed: quiet 8 · 100 % · +4.65 vs other 11 · 73 % · +0.78. The noisy side is still profitable
           (+0.80 backtest) → NOT a filter: the candidate is a SIZE-UP of the quiet side (caps for losers, multipliers for winners).
LINE       quiet = entry_flatbase_base_vol_pct ≤ 3.732 (services.flatbase.QUIET_BASE_VOL_MAX_PCT, the backtest's quietest-tercile cut, FROZEN)
           · other = every other scored fill · no stamp → recomputed from public 5m klines at the fill's own base bar (the engine's exact
           function, base_vol14_from_bars) · still unreadable → UNSCORED, never counted.
COHORT     CLOSED FLATBASE_SHORT fills (orders exports, dedup (opened_at, pair), CLOSED wins), opened ≥ the DECISION_LOG 654 commit + 10 min
           (forward only — the 22 master + batch fills the line was found on are in-sample and never counted). Booked pnl_percentage.
FROZEN     each read freezes ONCE, at the first prefix (in open-time order) reaching its N, after every fill of the prefix has closed (an export
           ≥ the crossing fill + 72 h):
           ① RETIRE read at quiet N ≥ 15 (needs ≥ 5 other fills): quiet avg ≤ other avg → RETIRED (the theory failed forward) · else KEEP OBSERVING.
           ② PROMOTE read at quiet N ≥ 30 (only after ① = KEEP OBSERVING), the locked Pattern-W multiplier gate: quiet WR ≥ 70 % ∧ avg ≥ +0.10 %
              ∧ Σ $ > 0, AND the sleeve's edge clears (size-by-edge rule): one-sided 95 % Wilson WR floor of ALL counted fills ≥ their breakeven
              WR (|avg loss| ÷ (avg win + |avg loss|)) → MULTIPLIER CANDIDATE 1.5× (operator decides; never automatic) · else KEEP OBSERVING.
           Booked P&L → pinned to the FLATBASE exit keys in observe_registry (an exit change → ⚠ STALE the same day).
STATE      reports/SCOUT_FB_QUIET_BASE_654.json (gitignored).
Usage: venv/bin/python scripts/scout_fb_quiet_base_654.py"""
import json
import math
import os
import sys
import time
import urllib.request

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_ROOT = os.path.dirname(_D)
_ROOT in sys.path or sys.path.insert(0, _ROOT)

import export_glob as _EG                         # noqa: E402
import scout_b1h_negflank as NF                   # noqa: E402  du_load_state / du_save_state
import scout_gatekit as _GK                       # noqa: E402
from services.flatbase import (QUIET_BASE_VOL_MAX_PCT, base_vol14_from_bars, base_close_ms,   # noqa: E402
                               FLAT_DAYS, DAY_MS, BAR_MS)

DL = 654
LINE = "FB_QUIET_BASE"
SLEEVE = "FLATBASE_SHORT"
CUT = QUIET_BASE_VOL_MAX_PCT
RETIRE_N, RETIRE_OTHER_MIN = 15, 5
PROMOTE_N, PROMOTE_WR, PROMOTE_AVG = 30, 0.70, 0.10
HOLD_MIN = 72 * 60 + 60                           # 72 h max hold + 1 h (a time exit fires on the bar / poll after 72 h)
Z95 = 1.6449                                      # one-sided 95 %
FETCH_MAX = 12                                    # kline fallbacks per run (3 reads × weight 10 each)
STOP_REASONS_NOTE = "booked pnl_percentage (net of fees and funding as booked)"
STATE = os.path.join(_ROOT, "reports", "SCOUT_FB_QUIET_BASE_654.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_fb_quiet_base")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_*.csv")
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "pnl", "entry_frenzy_spike_at",
        "entry_flatbase_base_vol_pct")
REGISTRATION = ("in-sample at registration (NEVER counted): backtest 545 · quiet 182 · 81 % · +1.79 vs other 363 · +0.80 · real master + batch "
                "closed quiet 8 · 100 % · +4.65 vs other 11 · 73 % · +0.78 (reports/FLATBASE_THEORY_2026-11-27.md)")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def side_of(vol):
    """≤ cut → quiet · other finite → other · missing → unscored. Pure."""
    v = _f(vol)
    if v is None:
        return "unscored"
    return "quiet" if v <= CUT + 1e-12 else "other"


def wilson_floor(wins, n, z=Z95):
    """one-sided Wilson lower bound of a win rate. Pure."""
    if n <= 0:
        return None
    p = wins / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def breakeven_wr(pcts):
    """|avg loss| ÷ (avg win + |avg loss|) on the given fills — None without both a win and a loss. Pure."""
    w = [x for x in pcts if x > 0]
    l = [x for x in pcts if x <= 0]
    if not w or not l:
        return None
    aw, al = sum(w) / len(w), abs(sum(l) / len(l))
    return al / (aw + al) if (aw + al) > 0 else None


def load(paths=None):
    """→ (closed FLATBASE fills — dedup, CLOSED wins, newest export wins —, newest export time or None)."""
    fr, newest = [], None
    for f in (sorted(_EG.glob_exports(EXPORT_GLOB), key=os.path.getmtime) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            continue
        if not (d.entry_strategy.astype(str) == SLEEVE).any():
            continue                              # (review) only exports that hold Flatbase rows date the hold check
        fr.append(d.assign(_m=os.path.getmtime(f)))
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct", "usd"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str) == SLEEVE].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")   # the locked dedup key
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    global _LAST_OPEN
    _LAST_OPEN = list(d[~d._c & d.o.notna()].o)      # OPEN Flatbase rows in the exports (the freeze waits for them)
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["usd"] = pd.to_numeric(c.pnl, errors="coerce").fillna(0.0)
    c = c[c.pct.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


_LAST_OPEN = []


def opened_before(open_o, t):
    """any OPEN Flatbase fill opened at or before t. Pure."""
    return any(x <= t for x in (open_o or []))


def _fetch_5m(pair, start_ms, end_ms):
    """public Binance futures 5m klines [open, o, h, l, c, v] for [start, end] — ≤ 4 reads (15 d = 4,320 bars; weight 10 each). Raises on a
    network error, and when the used-weight header passes 800 / min (the research budget is 1,200)."""
    rows, s = [], int(start_ms)
    for _ in range(4):
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={s}&endTime={int(end_ms)}&limit=1500")
        with urllib.request.urlopen(url, timeout=20) as r:
            page = json.load(r)
            used = r.headers.get("X-MBX-USED-WEIGHT-1M")
        if used is not None and int(used) > 800:
            raise RuntimeError(f"used weight {used} > 800")
        if not page:
            break
        rows += [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in page]
        s = int(page[-1][0]) + BAR_MS
        if len(page) < 1500 or s > end_ms:
            break
    return rows


def fallback_vol(pair, spike_at, fetch=_fetch_5m, cache_dir=CACHE):
    """the engine's base_vol14_from_bars at the fill's own base bar (spike close − 30 min), from public 5m klines — cached per (pair, spike).
    None when unreadable. A start 1 day before the window lets the function see the listing reached back past it (the engine's rule)."""
    try:
        sp = int(pd.Timestamp(str(spike_at)[:19]).value // 10**6)
    except (TypeError, ValueError):
        return None
    bc = base_close_ms(sp)
    key = os.path.join(cache_dir, f"{pair}_{sp}.json")
    try:
        with open(key) as fh:
            v = json.load(fh).get("vol")
        return _f(v)
    except (FileNotFoundError, ValueError, OSError):
        pass
    rows = fetch(pair, bc - (FLAT_DAYS + 1) * DAY_MS, bc)
    v = base_vol14_from_bars(rows, bc)
    if not rows:
        return None                               # (review) an empty read is never cached — retried next run
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(key, "w") as fh:
            json.dump({"vol": v}, fh)
    except OSError:
        pass
    return v


def score(a, fetch=_fetch_5m, cache_dir=CACHE, fetch_max=FETCH_MAX):
    """→ (frame with vol / side / src, fallback note). Stamped fills use the stamp; others ≤ fetch_max kline fallbacks per run."""
    if not len(a):
        return a.assign(vol=[], side=[], src=[]), ""
    a = a.copy()
    vols, srcs, used, err = [], [], 0, None
    for r in a.itertuples():
        v = _f(r.entry_flatbase_base_vol_pct)
        if v is not None:
            vols.append(v); srcs.append("stamp"); continue
        try:
            sp = int(pd.Timestamp(str(r.entry_frenzy_spike_at)[:19]).value // 10**6)
        except (TypeError, ValueError):
            sp = None
        if sp is None:
            vols.append(None); srcs.append("none"); continue          # no episode stamp → permanently unscored
        cached = os.path.exists(os.path.join(cache_dir, f"{r.pair}_{sp}.json"))
        if not cached and (used >= fetch_max or err):
            vols.append(None); srcs.append("pending"); continue       # (review) not tried this run → blocks freezing past it
        try:
            v = fallback_vol(r.pair, r.entry_frenzy_spike_at, fetch, cache_dir)
        except Exception as e:                    # network down / 418 / 429 / weight → stop fetching this run
            err = f"{type(e).__name__}"
            v = None
        used += 0 if cached else 1
        done = cached or os.path.exists(os.path.join(cache_dir, f"{r.pair}_{sp}.json"))
        vols.append(v); srcs.append("klines" if v is not None else ("none" if done else "pending"))
    a["vol"] = vols
    a["src"] = srcs
    a["side"] = [side_of(v) for v in vols]
    return a, (f"kline fallback stopped ({err}) — later unstamped fills wait" if err else "")


def retire_verdict(q, o):
    """read ①. PURE."""
    if len(q) < RETIRE_N:
        return "COLLECTING", f"quiet N {len(q)}/{RETIRE_N}"
    if len(o) < RETIRE_OTHER_MIN:
        return "COLLECTING", f"quiet N {len(q)} · other N {len(o)}/{RETIRE_OTHER_MIN} — waits for a comparison"
    mq, mo = float(q.pct.mean()), float(o.pct.mean())
    det = f"quiet N {len(q)} · WR {(q.pct > 0).mean():.0%} · avg {mq:+.3f} % vs other N {len(o)} · avg {mo:+.3f} %"
    return ("RETIRED", det) if mq <= mo else ("KEEP OBSERVING", det)


def promote_verdict(q, allc):
    """read ② — the locked Pattern-W gate + the sleeve's edge (Wilson floor vs breakeven on ALL counted fills). PURE."""
    if len(q) < PROMOTE_N:
        return "COLLECTING", f"quiet N {len(q)}/{PROMOTE_N}"
    wr, avg, usd = float((q.pct > 0).mean()), float(q.pct.mean()), float(q.usd.sum())
    n, w = len(allc), int((allc.pct > 0).sum())
    fl, be = wilson_floor(w, n), breakeven_wr(list(allc.pct))
    edge = fl is not None and w > 0 and (be is None or fl >= be)
    det = (f"quiet N {len(q)} · WR {wr:.0%} (≥ {PROMOTE_WR:.0%}) · avg {avg:+.3f} % (≥ +{PROMOTE_AVG:.2f}) · Σ ${usd:+,.0f} (> 0) · sleeve "
           f"floor {('%.1f %%' % (fl * 100)) if fl is not None else '–'} vs breakeven {('%.1f %%' % (be * 100)) if be is not None else 'n/a (no loss)'} on {n} fills")
    if wr >= PROMOTE_WR and avg >= PROMOTE_AVG and usd > 0 and edge:
        return "MULTIPLIER CANDIDATE 1.5× (operator decides; never automatic)", det
    return "KEEP OBSERVING", det


def crossing_prefix(sc, n_min, other_min=0):
    """the first prefix with ≥ n_min quiet AND ≥ other_min other fills (the retire read needs a comparison). Pure."""
    k = m = 0
    for i, s in enumerate(sc.side.values, 1):
        k += s == "quiet"
        m += s == "other"
        if k >= n_min and m >= other_min:
            return sc.iloc[:i]
    return None


def _row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    return f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | {g.usd.sum():+,.0f} |"


def _valid(rec):
    return isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, fetch=_fetch_5m, cache_dir=CACHE):
    """→ markdown lines. since_ms None → the DECISION_LOG 654 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🤫 {LINE} — FLATBASE shorts on a quiet base (daily vol ≤ {CUT:g} %) · DECISION_LOG {DL} · OBSERVE only (size-up candidate, never a filter)", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label=f"{LINE} (DL {DL})")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a, newest = load(paths)
    open_o = list(_LAST_OPEN)
    s = a[a.o >= since] if len(a) else a
    s, note = score(s, fetch, cache_dir)
    sc = s[s.side != "unscored"] if len(s) else s
    q = sc[sc.side == "quiet"] if len(sc) else sc
    o = sc[sc.side == "other"] if len(sc) else sc
    u = s[s.side == "unscored"] if len(s) else s
    one = float(sc.pct.sum()) if len(sc) else 0.0
    up = float(sc.pct.sum() + 0.5 * q.pct.sum()) if len(sc) else 0.0
    L += [f"Cohort: CLOSED {SLEEVE} fills opened ≥ {since:%Y-%m-%d %H:%M} UTC "
          f"({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · {STOP_REASONS_NOTE}.", "",
          "| Group | N | days | WR | avg % | Σ $ |", "|---|---|---|---|---|---|",
          _row(f"quiet (base daily vol ≤ {CUT:g} %)", q), _row("other", o), _row("UNSCORED — never counted" + (f" ({int((s.src == 'pending').sum())} pending a kline read)" if len(s) and (s.src == 'pending').any() else ""), u), "",
          f"Before → after (Σ %/fill, one size): all 1× {one:+.2f} → quiet at 1.5× {up:+.2f} (Δ {up - one:+.2f})"
          + (f" · stamps {int((s.src == 'stamp').sum())} / klines {int((s.src == 'klines').sum())}" if len(s) else ""), ""]
    if note:
        L.append(f"⏸ {note}")
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    recs = {}
    for key, n_min, o_min, fn in ((f"{LINE}@retire", RETIRE_N, RETIRE_OTHER_MIN, lambda p: retire_verdict(p[p.side == "quiet"], p[p.side == "other"])),
                           (f"{LINE}@promote", PROMOTE_N, 0, lambda p: promote_verdict(p[p.side == "quiet"], p))):
        rec = st.get(key) if ok else None
        if rec is not None and not _valid(rec):
            L.append(f"⚠ frozen record {key} malformed — operator restore needed; no further read this run.")
            break
        if key.endswith("promote") and not (recs.get(f"{LINE}@retire") or {}).get("state") == "KEEP OBSERVING":
            break                                 # ② only after ① froze KEEP OBSERVING
        if rec is None and ok and len(sc):
            pre = crossing_prefix(sc, n_min, o_min)
            pend = (bool(((s.src == "pending") & (s.o <= pre.o.max())).any()) if len(s) else False) if pre is not None else False
            pend = pend or (pre is not None and opened_before(open_o, pre.o.max()))   # (review) a still-OPEN earlier fill → wait
            if pre is not None and not pend and newest is not None and newest >= pre.o.max() + pd.Timedelta(minutes=HOLD_MIN):
                state, det = fn(pre)
                rec = dict(state=state, detail=det, at=f"{pre.o.max():%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                if state_path:
                    st[key] = rec
                    NF.du_save_state(st, state_path)
        recs[key] = rec
        if not rec:
            break
    for key, rec in recs.items():
        if rec:
            L.append(f"Frozen {key.split('@')[1]} read (crossing {rec['at']}, frozen {rec['run_at']}): **{rec['state']}** ({rec['detail']})")
    rs, rd = retire_verdict(q, o)
    ps, pd_ = promote_verdict(q, sc)
    L += [f"Now (information only): retire read {rs} — {rd} · promote read {ps} — {pd_}", ""]
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"FROZEN: quiet = base daily vol ≤ {CUT:g} % · ① retire read at quiet N ≥ {RETIRE_N} (≥ {RETIRE_OTHER_MIN} other): quiet avg ≤ other → RETIRED · "
          f"② promote read at quiet N ≥ {PROMOTE_N}: WR ≥ {PROMOTE_WR:.0%} ∧ avg ≥ +{PROMOTE_AVG:.2f} % ∧ Σ $ > 0 ∧ sleeve Wilson floor ≥ breakeven "
          "→ MULTIPLIER CANDIDATE 1.5× (operator decides). Alert only.", REGISTRATION, ""]
    return L


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
