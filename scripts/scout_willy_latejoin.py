#!/usr/bin/env python3
"""🕰 Scout — WILLY LATE-JOIN refusals priced on WILLY's live exit (DECISION_LOG 366, operator 2026-10-24 after LIGHTUSDT; observe-only).

WHY             LIGHTUSDT 10-24: spike bar open 10:40 / close 10:45 UTC with 24 h quote volume ≈ $7M (< frenzy_min_volume_usd $20M) → it joined the FRENZY
                shortlist only at 13:05, when the pump itself pushed the volume over the floor, and WILLY refused entry A as 28 bars late.
                Operator: "add the late-join low-volume spikes to scout" — a LIVE record of what such late joiners would have made.
PRIOR           reports/WILLY_LATE_JOIN_2026-10-22.md (pre-registered): 720 backtest late joiners (L 4–12) · 72 % WR · −0.27 %/trade, day-boot
                95 % −0.43 … −0.11, negative in 2025 / 26-H1 / 26-H2 → no engine change; the study proposed exactly this observe line and its
                frozen gate (below). DECISION_LOG 350 (R_NOMISS) already re-admits the late reads that missed no red bar — this line prices
                the ones it still refuses. ⚠ So the gated L 4–12 cohort here is (almost all) _MISSED reads — a red bar had already closed —
                while the study's 720 pooled missed AND no-red-missed joiners, priced on the OLD +1.25 exit with 5,000 day resamples (4,000
                here): the study is a reference, not the same cohort.
COHORT          decisions-export journal rows e = WILLY, gate FRENZY_WILLY_A_NOMISS_MISSED ("read {L} bars after the spike · a red bar already
                closed") · _LATE ("… no red bar yet, the red wait from the spike is over") · _OFF (eligible while the switch was off) — emitted
                by services/trading_engine._frenzy_willy_nomiss_ok for every COLD pair read past the WILLY A late window. Counted from the
                DECISION_LOG 350 deploy + 10 min (scout_gatekit; not in git → NOT EVALUABLE). The journal is mode-agnostic (DL 294), so one
                event appears in paper AND live exports: dedup (t, pair, gate) then ONE row per (pair, spike bar), the first read kept; mode =
                the mode of the OLDEST export holding the row (≈ the bot's mode at the event; a column, pooled in the gate).
TIMING          the event t is the pass that judged the bar just closed: forming bar open B = floor(t, 5 min); join bar = [B − 5 min, B);
                the engine's L = (B − spike close) // 5 min (services.frenzy.frenzy_willy_late_bars) → spike bar open = B − (L + 1) × 5 min.
                B = floor(t) assumes the event was written inside its pass's bar: t − B > 120 s → status TIMING? (the bar may be one off) —
                held out of the gate, not priced. The derived spike bar is checked on the fetched 5m bars (spike_ok = it closed green).
PER REFUSAL     public Binance 5m klines (budgeted fetch, cached), then:
                ① L bucket 4–12 · 13–36 · 37+ (≤ 3 is the on-time window — never a refusal).
                ② vol24 = Σ quote volume of the 288 5m bars ending WITH the spike bar → LOWVOL (< frenzy_min_volume_usd) · LIQUID (≥ the floor:
                   the 15 % move leg or the 25-pair cap held it back) · UNREAD (< 288 bars, e.g. a new listing).
                ③ entry = WILLY's late-cold rule (the study's R1): armed at the join bar, the FIRST red 5m bar (close < open strictly; the join
                   bar counts) closing ≤ join close + frenzy_willy_red_max_wait_minutes → entry = its close × (1 + 0.10 %) at close + 8 s (the
                   live ruler: 8 s + 0.10 slip, FRENZY latency study / the late-join study). No red bar in the wait → EXPIRED (counted, not
                   priced). Window not closed yet → PENDING. _OFF rows (R_NOMISS-eligible, switch off) are priced on the same R1 rule.
                ④ exit = today's live WILLY V2b (DECISION_LOG 358): target T = max(+1.25, min(+8, net at H × 0.99)), H = highest 5m high from
                   the spike bar through the red bar, on the engine's fee convention (scout_willy_exit_v2.target_at_k: the config taker fee as
                   the entry fee — there is no fill — + the maker exit fee) · stop −4 · late TP +0.25 after 30 min · cap 90 → walk_v2 on the
                   hybrid tick → 1m net path (scout_frenzy_willy_exit.load_path; caches + scout_willy_timecap.ensure_data). An unresolved path →
                   UNRES (unscored; holds the freeze). The exit constants are PINNED (scout_willy_exit_v2 / willy_exit_era), never read from config.
                   The TP touch is tested on the taker-exit net path (scout_willy_timecap.net_pct, ≈ 0.03 % conservative vs a maker exit) —
                   inherited from the V2b scout. 1m-only paths are kept (no tick-provisional hold).
                ⑤ turnover routing (WILLY vs WILLY_HT ¼) is NOT modelled: no scout reads a supply / market-cap source offline (the study's
                   bapi_supply.json is an Oct-8 snapshot no scout uses) → every refusal is priced as a full position.
                ⑥ same pump twice: a refusal whose entry falls inside an earlier PRICED refusal's hold (entry → exit) on the same pair →
                   OVERLAP (not priced, out of the gate).
TERMINAL        a symbol answered 4xx on its 5m page is remembered (bad_symbols.json, never re-requested) → NODATA_FINAL; a NODATA (bars
                read, gap) / UNPRICED / UNRES row whose window closed > 7 days ago with its fetch final (the 1m page attempt done in
                scout_willy_timecap's attempts) → NODATA_FINAL / UNPRICED_FINAL / UNRES_FINAL. *_FINAL rows never hold the freeze and are
                never priced; they are counted in the gate text.
🔒 FROZEN GATE   (the study's pre-registered gate, L 4–12, all classes) at the first prefix (by spike time) holding ≥ 30 PRICED refusals on ≥ 10
                UTC days → RE-OPEN CANDIDATE iff avg > 0 ∧ the day-clustered bootstrap 95 % lower bound > 0 (4,000 resamples, seed 7); else KEEP
                (no change). A PENDING / NODATA / UNPRICED / UNRES row inside the prefix holds the freeze, and so does a prefix whose last spike
                is not ≤ the newest decisions-export stamp − 65 min (a refusal of an earlier spike could still arrive). Persisted once in
                reports/.willy_latejoin_gate.json — never re-fit. LOWVOL-only and L 13–36 / 37+ summaries are observe-only, no gate.
BLIND SPOTS     (printed — BLIND) engine-only guards are not replayed (WILLY / HT slot, pair already held, dislocation / lateness guards, the
                turnover refusal, the global hold, the episode ending inside the wait) · the 25-pair cap timing is not visible; LIQUID also
                covers restart / FRENZY-paused / pre-filter cold reads ("not the floor", cause unseen) · late reads with NO journal row (no-red
                check unreadable, late unreadable, stored state unread, WILLY or entry A off) · R_NOMISS entries killed at the red bar by the
                $ floor (FRENZY_WILLY_A_NOMISS_VOL24_LOW — counted in scout_willy_nomiss, priced nowhere) · late reads before the DL 350
                deploy were never journaled (CloudWatch only) · turnover routing (⑤) · 1m-only paths kept · TP on a taker-exit net path.
Usage           venv/bin/python scripts/scout_willy_latejoin.py [--no-fetch]
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402
import scout_willy_exit_v2 as V2                             # noqa: E402  the live V2b walker + the engine's target convention
import willy_exit_era as WEra                                # noqa: E402

DL, NOMISS_DL, LAG_MIN = 366, 350, 10
EVENTS = ("FRENZY_WILLY_A_NOMISS_MISSED", "FRENZY_WILLY_A_NOMISS_LATE", "FRENZY_WILLY_A_NOMISS_OFF")
_L_RE = re.compile(r"read\s+(\d+)\s+bars?\s+after the spike")
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
CONFIG = os.path.join(_R, "trading_config.json")
STATE = os.path.join(_R, "reports", ".willy_latejoin_gate.json")
CSV_OUT = os.path.join(_R, "reports", "SCOUT_WILLY_LATEJOIN.csv")
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_willy_latejoin", "5m")
BAD_SYMBOLS = os.path.join(_R, "reports", "backtest_cache", "scout_willy_latejoin", "bad_symbols.json")
MIN, BAR = 60_000, 300_000
VOL_BARS = 288                                               # rolling 24 h = 288 closed 5m bars (the shortlist's volume leg)
LATENCY_MS, SLIP_PCT = 8_000, 0.10                           # live ruler: red close + 8 s, + 0.10 % slip (late-join study / latency study)
BUCKETS = (("4-12", 4, 12), ("13-36", 13, 36), ("37+", 37, 10 ** 9))
GATE_BUCKET, GATE_KEY = "4-12", "LATEJOIN_L4_12"
GATE_N, GATE_DAYS = 30, 10
BOOT_N, BOOT_SEED = 4000, 7
HOLDS = ("PENDING", "NODATA", "UNPRICED", "UNRES", "OVERLAP?")   # outcome not known yet → holds the freeze of a prefix it belongs to
# (OVERLAP? = an earlier same-pair refusal is still unresolved, so whether this one overlaps its hold is not known yet)
FINALS = ("NODATA_FINAL", "UNPRICED_FINAL", "UNRES_FINAL")   # never resolvable → out of the gate (never held, never priced), counted
OUT = FINALS + ("TIMING?", "OVERLAP")                       # out of the gate, counted in its text
FINAL_AGE_MS = 7 * 86_400_000                                # a window closed > 7 days ago with its fetch final → *_FINAL
TIMING_MAX_S = 120                                           # t − floor(t, 5 min) > 120 s → the pass may belong to the previous bar
FREEZE_LAG_MS = 65 * 60_000                                  # freeze only when the prefix's last spike ≤ the newest export stamp − 65 min
BAD = "BAD_SYMBOL"                                           # fetch_5m sentinel: the symbol answered 4xx (remembered, never re-requested)
WAIT_DEF, FLOOR_DEF, TAKER_DEF = 60.0, 20e6, 0.00045
MAX_REQ_5M, WEIGHT_STOP, BUDGET_S, PACE_S, KL_MAX = 30, 900, 25.0, 0.25, 1500
MAX_REQ_1M = 10
PATH_MIN = max(WEra.OLD_HOLD, 120)                           # path minutes after the entry (the 90-min cap + slack, the V2b scout's span)
LIST_CAP = 20
_NET_BLOCKED = False                                         # tests: any fetch attempt raises
CSV_COLS = ("t", "mode", "pair", "gate", "L", "bucket", "spike_utc", "join_close_utc", "vol24_usd", "vol24_bars", "cls", "floor_usd",
            "wait_min", "status", "red_open_utc", "entry_utc", "entry_px", "spike_high", "target", "exit_why", "net_pct", "exit_utc", "path_src",
            "timing_off_s", "spike_ok", "day")


# ─────────────────────────── config / parsing ───────────────────────────
def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_cfg(path=CONFIG):
    """→ dict(wait min, floor $, taker rate) from trading_config.json (thresholds.frenzy_willy_red_max_wait_minutes /
    thresholds.frenzy_min_volume_usd / taker_fee — what the engine reads); missing / bad → the code defaults. Never raises."""
    try:
        with open(path, encoding="utf-8") as fh:
            c = json.load(fh) or {}
    except Exception:
        c = {}
    th = c.get("thresholds") or {}
    w, fl, tk = _f(th.get("frenzy_willy_red_max_wait_minutes")), _f(th.get("frenzy_min_volume_usd")), _f(c.get("taker_fee"))
    return dict(wait=w if w is not None and w >= 0 else WAIT_DEF, floor=fl if fl and fl > 0 else FLOOR_DEF,   # wait 0 = the join bar only (engine)
                taker=tk if tk is not None and 0 <= tk < 0.01 else TAKER_DEF)


def parse_L(reason):
    """'read 28 bars after the spike · …' → 28 · unreadable → None."""
    m = _L_RE.search(str(reason or ""))
    return int(m.group(1)) if m else None


def bucket(L):
    for name, a, b in BUCKETS:
        if a <= L <= b:
            return name
    return "<=3"


def event_bars(t_ms, L):
    """the engine's bar arithmetic for one read → dict(B forming bar open, join_open, join_close, spike_open, spike_close) in ms."""
    B = (int(t_ms) // BAR) * BAR
    return dict(B=B, join_open=B - BAR, join_close=B, spike_close=B - int(L) * BAR, spike_open=B - (int(L) + 1) * BAR)


def _mode(path):
    b = os.path.basename(str(path))
    return "live" if ("_live_" in b or "_LIVE_" in b) else "paper"


def _ms(ts):
    return int(pd.Timestamp(ts).value // 1_000_000)


def _stamp_ms(f):
    """the export's own filename stamp (UTC) — files without one (archived BASELINE copies) give None: their mtime is a copy time, not
    coverage, and must never move the freeze earlier."""
    st = EG.export_stamp(f)
    try:
        return _ms(st) if st else None
    except Exception:
        return None


def read_refusals(paths=None, since=None):
    """→ (refusals frame, n_bad, newest export stamp ms or None) — one row per (pair, spike bar), the first read kept; columns t, ts, t_ms,
    pair, gate, reason, L, mode, timing_off_s + event_bars(). Bad / unreadable rows are counted, never raised."""
    if paths is None:
        import glob
        paths = sorted({p for pat in DEC_PATTERNS for p in (EG.glob_exports(pat) if "scalpars_" in pat else glob.glob(pat))})
    fr, newest = [], None
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "gate", "reason"))
        except Exception:
            continue
        if not {"t", "e", "pair", "gate"} <= set(x.columns):
            continue
        sm = _stamp_ms(f)                                       # every readable journal counts toward "the newest export" (freeze timing)
        newest = sm if newest is None or (sm is not None and sm > newest) else newest
        x = x[(x.e.astype(str) == "WILLY") & x.gate.astype(str).isin(EVENTS)]
        if len(x):
            st = EG.export_stamp(f) or (pd.Timestamp(os.path.getmtime(f), unit="s").strftime("%Y-%m-%dT%H:%M:%S") if os.path.exists(f) else "9")
            fr.append(x.assign(mode=_mode(f), _st=st))
    cols = ["t", "ts", "t_ms", "pair", "gate", "reason", "L", "mode", "timing_off_s", "B", "join_open", "join_close", "spike_close", "spike_open"]
    if not fr:
        return pd.DataFrame(columns=cols), 0, newest
    d = pd.concat(fr, ignore_index=True)
    if "reason" not in d:
        d["reason"] = None
    d = d.sort_values("_st", kind="stable").drop_duplicates(["t", "pair", "gate"], keep="first")
    d["ts"] = pd.to_datetime(d.t.astype(str).str[:23].str.replace(" ", "T"), errors="coerce", format="ISO8601")
    d["L"] = [parse_L(r) for r in d.reason]
    ok = d.ts.notna() & d.L.notna() & d.pair.notna() & d.pair.astype(str).str.strip().str.len().gt(0)
    n_bad = int((~ok).sum())
    d = d[ok].copy()
    if since is not None:
        d = d[d.ts >= since]
    if not len(d):
        return pd.DataFrame(columns=cols), n_bad, newest
    d["L"] = d.L.astype(int)
    d["t_ms"] = [_ms(x) for x in d.ts]
    eb = pd.DataFrame([event_bars(t, L) for t, L in zip(d.t_ms, d.L)], index=d.index)
    d = pd.concat([d, eb], axis=1)
    d["timing_off_s"] = (d.t_ms - d.B) / 1000.0
    d = d.sort_values(["ts", "pair"], kind="stable").drop_duplicates(["pair", "spike_open"], keep="first")
    return d[cols].reset_index(drop=True), n_bad, newest


# ─────────────────────────── 5m klines (public, budgeted, cached) ───────────────────────────
class _Bud:
    def __init__(self, fetch=True, max_req=MAX_REQ_5M, budget_s=BUDGET_S):
        self.fetch, self.max_req, self.n, self.used, self.stop = fetch, max_req, 0, 0, None
        self.deadline = time.monotonic() + budget_s
        self.notes = []


def _k5_frame(rows):
    """Binance kline rows → DataFrame open_time, o, h, l, c, qv (quote volume, index 7)."""
    out = [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7])) for x in rows or [] if len(x) > 7]
    return pd.DataFrame(out, columns=["open_time", "o", "h", "l", "c", "qv"])


def _bad_symbols():
    try:
        with open(BAD_SYMBOLS) as fh:
            return dict(json.load(fh) or {})
    except Exception:
        return {}


def _mark_bad(pair, code):
    try:
        b = _bad_symbols()
        b[str(pair)] = int(code)
        os.makedirs(os.path.dirname(BAD_SYMBOLS), exist_ok=True)
        tmp = f"{BAD_SYMBOLS}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(b, fh)
        os.replace(tmp, BAD_SYMBOLS)
    except Exception:
        pass


def _fetch_chunk(pair, start, limit, now_ms, bud):
    """ONE 5m page (cached JSON once the page is complete) → rows, BAD (the symbol answered 4xx — remembered, never re-requested) or None
    (not fetched: budget / cooldown / a 403-418-429 stop / a transient failure)."""
    if str(pair) in _bad_symbols():
        return BAD
    p = os.path.join(CACHE, f"{pair}_{int(start)}_{int(limit)}.json")
    if os.path.exists(p):
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            pass
    if not bud.fetch or bud.stop:
        return None
    import scout_b1h_negflank as NF
    import scout_willy_timecap as WT
    if NF._cooldown_until() > now_ms:
        bud.stop = "rate-limit cooldown active"
        return None
    if bud.n >= bud.max_req:
        bud.stop = f"request cap {bud.max_req}/run reached — the rest catches up on later runs"
        return None
    if bud.used >= WEIGHT_STOP:
        bud.stop = f"used weight {bud.used} ≥ {WEIGHT_STOP}"
        return None
    left = bud.deadline - time.monotonic()
    if left < 1.0:
        bud.stop = "wall-clock budget spent"
        return None
    if _NET_BLOCKED:
        raise RuntimeError("network blocked (test)")
    bud.n += 1
    time.sleep(PACE_S)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={int(start)}&limit={int(limit)}"
    try:
        rows, headers = WT._threaded(WT._http_get, url, min(10.0, left), now_ms, f"{pair} 5m fetch")
    except NF.RateLimited as e:
        bud.stop = str(e)
        return None
    except Exception as e:
        code = getattr(e, "code", None)
        if code in (403, 451):                                  # WAF / IP / region block → stop every further request this run (like 418 / 429)
            bud.stop = f"Binance {code} (blocked) — stopped"
            return None
        bud.notes.append(f"{pair} 5m page failed ({code or str(e)[:60]})")
        if code == 400:                                         # invalid symbol (-1121, delisted) — the only 4xx remembered as permanent
            _mark_bad(pair, code)
            return BAD
        return None                                             # any other 4xx / 5xx / timeout: transient, retried next run
    bud.used = int((headers or {}).get("X-MBX-USED-WEIGHT-1M") or 0)
    rows = [x for x in rows if int(x[0]) + BAR <= now_ms]                      # closed bars only
    if rows and int(start) + int(limit) * BAR <= now_ms:                       # cache complete pages only
        os.makedirs(CACHE, exist_ok=True)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(rows, fh)
        os.replace(tmp, p)
    return rows


def fetch_5m(pair, start, end, now_ms, bud):
    """5m bars with open in [start, end) → DataFrame or None when a page could not be read this run."""
    fr, s = [], int(start)
    while s < int(end):
        lim = int(min(KL_MAX, (int(end) - s) // BAR))
        if lim <= 0:
            break
        rows = _fetch_chunk(pair, s, lim, now_ms, bud)
        if rows is None or rows is BAD or (isinstance(rows, str) and rows == BAD):
            return BAD if rows is not None else None
        fr.append(_k5_frame(rows))
        s += lim * BAR
    k = pd.concat(fr, ignore_index=True) if fr else _k5_frame([])
    return k.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


# ─────────────────────────── per-refusal math (pure) ───────────────────────────
def vol24(k, spike_open):
    """→ (Σ quote volume of the 288 bars ending with the spike bar, bars found). Fewer than 288 → (partial Σ, n) — the class is UNREAD."""
    if k is None or not len(k):
        return None, 0
    w = k[(k.open_time > int(spike_open) - VOL_BARS * BAR) & (k.open_time <= int(spike_open))]
    return (float(w.qv.sum()) if len(w) else None), int(len(w))


def vol_class(v, n, floor):
    if v is None or n < VOL_BARS:
        return "UNREAD"
    return "LOWVOL" if v < float(floor) else "LIQUID"


def red_entry(k, join_open, wait_min, now_ms):
    """the R1 late-cold entry: armed at the join bar, the first RED 5m bar (close < open; the join bar counts) whose close ≤ join close +
    wait → dict(status RED / EXPIRED / PENDING / NODATA, red_open, close). A gap in the bars before a red bar → NODATA (never guessed)."""
    jo = int(join_open)
    last_open = jo + BAR + int(round(float(wait_min) * MIN)) - BAR                # the last bar whose close ≤ join close + wait
    if k is None:
        return dict(status="NODATA", red_open=None, close=None)
    want = range(jo, last_open + 1, BAR)
    by = {int(r.open_time): r for r in k.itertuples()} if len(k) else {}
    for o in want:
        r = by.get(o)
        if r is None:
            if o + BAR > int(now_ms):
                return dict(status="PENDING", red_open=None, close=None)
            return dict(status="NODATA", red_open=None, close=None)
        if _f(r.o) and _f(r.c) and float(r.c) < float(r.o):
            return dict(status="RED", red_open=o, close=float(r.c))
    return dict(status="EXPIRED", red_open=None, close=None)


def spike_high(k, spike_open, red_open):
    """H = the highest 5m high from the spike bar through the red entry bar (frenzy_walk's peak) → float or None."""
    w = k[(k.open_time >= int(spike_open)) & (k.open_time <= int(red_open))] if k is not None and len(k) else None
    if w is None or not len(w):
        return None
    h = float(w.h.max())
    return h if math.isfinite(h) and h > 0 else None


def spike_ok(k, spike_open):
    """the derived spike bar on the fetched 5m bars: True = present and closed green (a FRENZY spike bar is an up bar) · False = present,
    not green (the bar arithmetic may be off) · None = not in the bars."""
    if k is None or isinstance(k, str) or not len(k):
        return None
    w = k[k.open_time == int(spike_open)]
    if not len(w):
        return None
    return bool(float(w.c.iloc[0]) > float(w.o.iloc[0]))


def exit_ms(P, tp):
    """the exit time of walk_v2(P, tp): the first event whose truncated path already decides the walk (binary search — the decision is
    monotone in the prefix); CAP → te + cap; UNRES → None."""
    w = V2.walk_v2(P, tp)
    if w["why"] == "UNRES":
        return None
    te = int(P["te"])
    if w["why"] == "CAP":
        return te + WEra.OLD_HOLD * MIN
    t = P["t"]
    lo, hi = 0, len(t) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        Q = {k: (v[:mid + 1] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
        Q["end_ms"] = int(t[mid])
        if V2.walk_v2(Q, tp)["why"] == "UNRES":
            lo = mid + 1
        else:
            hi = mid
    return int(t[lo])


def target(E, H, taker, maker=None):
    """the V2b target the engine would store: max(+1.25, min(+8, net at H × 0.99)) — scout_willy_exit_v2.target_at_k on a synthetic fill
    row (entry fee = taker × notional; quantity 1)."""
    return V2.target_at_k({"entry_price": E, WEra.HIGH_COL: H, "entry_fee": float(taker) * float(E), "quantity": 1.0}, V2.LIVE_K, maker)


def _utc(ms):
    return None if ms is None or (isinstance(ms, float) and not math.isfinite(ms)) else pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d %H:%M:%S")


def price_one(r, k, cfg, now_ms, path_fn):
    """one refusal → dict of computed columns. path_fn(pair, te, E) → net path P (load_path shape) or None. Never raises."""
    out = dict(t=str(r["t"]), mode=r["mode"], pair=r["pair"], gate=r["gate"], L=int(r["L"]), bucket=bucket(int(r["L"])),
               spike_utc=_utc(r["spike_open"]), join_close_utc=_utc(r["join_close"]), vol24_usd=None, vol24_bars=0, cls="UNREAD",
               floor_usd=cfg["floor"], wait_min=cfg["wait"], status="NODATA", red_open_utc=None, entry_utc=None, entry_px=None,
               spike_high=None, target=None, exit_why=None, net_pct=None, exit_utc=None, path_src=None,
               timing_off_s=_f(r.get("timing_off_s")), spike_ok=None,
               day=pd.Timestamp(int(r["spike_open"]), unit="ms").strftime("%Y-%m-%d"), _te=None, _E=None, _xm=None, _kread=k is not None)
    try:
        if isinstance(k, str):                                  # BAD: the symbol answered 4xx on its 5m page — never resolvable
            out["status"] = "NODATA_FINAL"
            return out
        v, n = vol24(k, r["spike_open"])
        out.update(vol24_usd=v, vol24_bars=n, cls=vol_class(v, n, cfg["floor"]), spike_ok=spike_ok(k, r["spike_open"]))
        if out["timing_off_s"] is not None and out["timing_off_s"] > TIMING_MAX_S:
            out["status"] = "TIMING?"                           # the event may belong to the previous pass's bar → never priced into the gate
            return out
        e = red_entry(k, r["join_open"], cfg["wait"], now_ms)
        out["status"] = e["status"]
        if e["status"] != "RED":
            return out
        E = float(e["close"]) * (1 + SLIP_PCT / 100.0)
        te = int(e["red_open"]) + BAR + LATENCY_MS
        H = spike_high(k, r["spike_open"], e["red_open"])
        T = target(E, H, cfg["taker"]) if H else None
        out.update(red_open_utc=_utc(e["red_open"]), entry_utc=_utc(te), entry_px=E, spike_high=H, target=T, _te=te, _E=E)
        if T is None:
            out["status"] = "UNPRICED"
            return out
        P = path_fn(r["pair"], te, E)
        if P is None or not len(P.get("t", [])):
            out["status"] = "UNPRICED"
            return out
        w = V2.walk_v2(P, T)                                    # the LIVE V2b exit (stop −4 · late +0.25 @30 · cap 90), pinned
        out.update(exit_why=w["why"], path_src=P.get("_src", "path"))
        if w["why"] == "UNRES" or not math.isfinite(w["pct"]):
            out["status"] = "UNRES"
            return out
        xm = exit_ms(P, T)
        out.update(status="PRICED", net_pct=round(float(w["pct"]), 4), exit_utc=_utc(xm), _xm=xm)
    except Exception as ex:                                     # a code / environment error is never "no data": it holds, never turns final
        out.update(status="NODATA", exit_why=f"error {str(ex)[:60]}", _kread=False)
    return out


def _default_path_fn():
    import scout_frenzy_willy_exit as FX
    tcache = {}

    def fn(pair, te, E):
        P = FX.load_path(pair, int(te), float(E), PATH_MIN, tcache)
        P["_src"] = FX.path_src(P)
        return P
    return fn


def _default_fetch_final():
    """(pair, te) → True when the 1m page attempt for that entry is final in scout_willy_timecap's attempts (done = fetched or given up)."""
    try:
        import scout_willy_timecap as WT
        done = WT.load_attempts()["done"]
    except Exception:
        done = {}

    def fn(pair, te):
        try:
            import scout_willy_timecap as WT
            v = done.get(WT._kk(pair, te))
            # a block / rate limit is never a final give-up (WT marks any 4xx done) — only a real fetch or a real give-up is
            return v is not None and not str(v).startswith(("HTTP 403", "HTTP 408", "HTTP 418", "HTTP 429", "HTTP 451"))
        except Exception:
            return False
    return fn


def finalize(x, r, cfg, now_ms, fetch_final):
    """a NODATA (bars were read — a gap / bad value, not a skipped fetch) / UNPRICED / UNRES row whose window closed > 7 days ago with
    its fetch final → *_FINAL (never resolvable: out of the gate, counted). Never raises."""
    try:
        st = x["status"]
        if st == "NODATA" and x.get("_kread"):
            end = int(r["join_close"]) + int(round(cfg["wait"] * MIN))
            if int(now_ms) - end > FINAL_AGE_MS:
                x["status"] = "NODATA_FINAL"
        elif st in ("UNPRICED", "UNRES"):
            te = x.get("_te")
            end = (int(te) if te is not None else int(r["join_close"]) + int(round(cfg["wait"] * MIN))) + PATH_MIN * MIN
            if int(now_ms) - end > FINAL_AGE_MS and (te is None or x.get("target") is None or fetch_final(x["pair"], int(te))):
                x["status"] = st + "_FINAL"
    except Exception:
        pass
    return x


def mark_overlaps(rows):
    """⑥ the same pump twice: a PRICED refusal whose entry falls inside an earlier PRICED refusal's hold (entry → exit) on the same pair →
    OVERLAP (not priced). In entry order; an OVERLAP row never blocks a later one."""
    held, open_ = {}, {}
    for x in rows:                                              # an earlier same-pair refusal with an entry but no resolved exit yet
        if x["status"] in ("UNPRICED", "UNRES") and x.get("_te") is not None:
            open_.setdefault(x["pair"], []).append(int(x["_te"]))
    for x in sorted((x for x in rows if x["status"] == "PRICED"), key=lambda x: (int(x["_te"]), str(x["pair"]))):
        if any(t < int(x["_te"]) for t in open_.get(x["pair"], [])):
            x.update(status="OVERLAP?", net_pct=None, exit_why="an earlier same-pair refusal is unresolved — overlap not known yet")
            continue
        iv = held.get(x["pair"])
        if iv and iv[0] <= int(x["_te"]) < iv[1]:
            x.update(status="OVERLAP", net_pct=None, exit_why=f"inside the {pd.Timestamp(iv[0], unit='ms'):%m-%d %H:%M} refusal's hold")
            continue
        held[x["pair"]] = (int(x["_te"]), int(x["_xm"]) if x.get("_xm") is not None else int(x["_te"]) + WEra.OLD_HOLD * MIN)
    return rows


def evaluate(ref, cfg, now_ms, k5_fn=None, path_fn=None, fetch=True, fetch_final=None):
    """→ (priced frame, notes). k5_fn(pair, start, end) → 5m DataFrame, None (not read this run) or BAD (default: the budgeted public
    fetch + cache). fetch_final(pair, te) → is the 1m page attempt final (default: scout_willy_timecap's attempts)."""
    notes = []
    bud = _Bud(fetch=fetch)
    if k5_fn is None:
        def k5_fn(pair, start, end):
            return fetch_5m(pair, start, end, now_ms, bud)
    own_path = path_fn is None
    if own_path:
        path_fn = _default_path_fn()
    recs, ks, rows = ref.to_dict("records"), [], []
    for r in recs:
        start = int(r["spike_open"]) - (VOL_BARS - 1) * BAR                    # 288 bars ending with the spike bar …
        end = int(r["join_close"]) + int(round(cfg["wait"] * MIN))             # … through the last bar the red wait allows
        try:
            k = k5_fn(r["pair"], start, min(end, (int(now_ms) // BAR) * BAR))
        except Exception as ex:
            notes.append(f"{r['pair']} 5m read failed ({str(ex)[:60]})")
            k = None
        ks.append(k)
        rows.append(price_one(r, k, cfg, now_ms, path_fn))
    # the 1m page catch-up for refusals still without a resolved path (shared caches + the budgeted fetch of the WILLY scouts)
    need = [i for i, x in enumerate(rows) if x["status"] in ("UNPRICED", "UNRES") and x["_te"] is not None and x["target"] is not None]
    if own_path and fetch and need and not bud.stop:            # a 5m block / rate limit stops the 1m catch-up too
        try:
            import scout_willy_timecap as WT
            fr = pd.DataFrame([dict(ts=pd.Timestamp(rows[i]["_te"], unit="ms"), pair=rows[i]["pair"], te=rows[i]["_te"], src=None) for i in need])
            notes += WT.ensure_data(fr, WEra.OLD_HOLD, int(now_ms), fetch=True, max_req=MAX_REQ_1M)
            path_fn = _default_path_fn()
            for i in need:
                rows[i] = price_one(recs[i], ks[i], cfg, now_ms, path_fn)
        except Exception as ex:
            notes.append(f"1m catch-up skipped ({str(ex)[:80]})")
    if bud.stop:
        notes.append(f"5m fetching stopped: {bud.stop}")
    notes += bud.notes
    if fetch_final is None:
        fetch_final = _default_fetch_final()
    for i, x in enumerate(rows):
        rows[i] = finalize(x, recs[i], cfg, now_ms, fetch_final)
    mark_overlaps(rows)
    xc = list(CSV_COLS) + ["_te", "_E", "_xm"]
    d = pd.DataFrame(rows, columns=xc) if rows else pd.DataFrame(columns=xc)
    d["net_pct"] = pd.to_numeric(d.net_pct, errors="coerce")
    d["_spk"] = pd.to_datetime(d.spike_utc, errors="coerce")
    return d.sort_values(["_spk", "pair"], kind="stable").reset_index(drop=True), notes


# ─────────────────────────── gate ───────────────────────────
def boot_lb(pct, day, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap: resample UTC days with replacement, pooled mean of their refusals → the 2.5th percentile (None < 2 days)."""
    g = pd.DataFrame(dict(p=np.asarray(pct, float), d=np.asarray(day))).groupby("d").p.agg(["sum", "count"])
    if len(g) < 2:
        return None
    s, c = g["sum"].values, g["count"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    return float(np.percentile(s[idx].sum(1) / c[idx].sum(1), 2.5))


def _out_txt(g):
    o = g[g.status.isin(OUT)] if len(g) else g
    if not len(o):
        return ""
    return " · out of the gate: " + ", ".join(f"{k} {int(v)}" for k, v in o.status.value_counts().sort_index().items())


def gate(d, newest_ms=None):
    """🔒 the frozen L 4–12 gate → (verdict, text, complete). Verdicts COLLECTING · HOLD · RE-OPEN CANDIDATE · KEEP. *_FINAL / TIMING? /
    OVERLAP rows are out (never held, never priced; counted). A decided prefix freezes only when its last spike ≤ newest_ms − 65 min
    (newest decisions-export stamp; unknown → HOLD). Never raises."""
    try:
        g = d[d.bucket == GATE_BUCKET].sort_values(["_spk", "pair"], kind="stable") if len(d) else d
        ox = _out_txt(g)
        g = g[~g.status.isin(OUT)] if len(g) else g
        pr = g[g.status == "PRICED"] if len(g) else g
        end = None
        n, days = 0, set()
        for i, r in enumerate(g.itertuples()):
            if r.status == "PRICED":
                n += 1
                days.add(r.day)
                if n >= GATE_N and len(days) >= GATE_DAYS:
                    end = i
                    break
        if end is None:
            return "COLLECTING", (f"⏳ {len(pr)}/{GATE_N} priced L 4–12 refusals on {pr.day.nunique() if len(pr) else 0}/{GATE_DAYS} UTC days "
                                  f"(expired {int((g.status == 'EXPIRED').sum()) if len(g) else 0} · held {int(g.status.isin(HOLDS).sum()) if len(g) else 0}{ox})"), False
        s = g.iloc[:end + 1]
        h = s[s.status.isin(HOLDS)]
        if len(h):
            return "HOLD", (f"prefix of {len(s)} L 4–12 refusals holds {len(h)} unresolved ("
                            + ", ".join(f"{x.pair} {str(x.spike_utc)[5:16]} {x.status}" for x in h.head(5).itertuples()) + f") — held{ox}"), False
        p = s[s.status == "PRICED"]
        avg, lb = float(p.net_pct.mean()), boot_lb(p.net_pct.values, p.day.values)
        ok = avg > 0 and lb is not None and lb > 0
        head = (f"first {len(p)} priced L 4–12 refusals on {p.day.nunique()} days ({p.day.min()} → {p.day.max()}): WR {(p.net_pct > 0).mean():.0%} · "
                f"avg {avg:+.3f} % · day-boot 95 % LB {'n/a' if lb is None else f'{lb:+.3f}'}{ox}")
        last = int(s._spk.max().value // 1_000_000)
        if newest_ms is None or last > int(newest_ms) - FREEZE_LAG_MS:
            nw = "unknown" if newest_ms is None else pd.Timestamp(int(newest_ms), unit="ms").strftime("%m-%d %H:%M")
            return "HOLD", (head + f" — would read {'RE-OPEN CANDIDATE' if ok else 'KEEP'}, not frozen yet: the prefix's last spike "
                            f"{pd.Timestamp(last, unit='ms'):%m-%d %H:%M} is not ≤ the newest decisions export ({nw}) − 65 min"), False
        return ("RE-OPEN CANDIDATE" if ok else "KEEP"), head + (" → RE-OPEN CANDIDATE: operator decides whether to re-open the late window "
                                                                  "(re-run the pre-registered study bar first)" if ok else " → KEEP (no change)"), True
    except Exception as e:
        return "ERROR", f"gate unreadable ({str(e)[:80]})", False


# ─────────────────────────── report ───────────────────────────
def summ(g):
    """N · priced · days · WR · avg · Σ · expired · held."""
    if not len(g):
        return "N 0"
    p = g[g.status == "PRICED"]
    s = (f"N {len(g)} · priced {len(p)}")
    if len(p):
        s += (f" on {p.day.nunique()} days · WR {(p.net_pct > 0).mean():.0%} · avg {p.net_pct.mean():+.3f} % · Σ {p.net_pct.sum():+.2f} pts")
    return s + (f" · expired {int((g.status == 'EXPIRED').sum())} · held {int(g.status.isin(HOLDS).sum())} · out "
                f"{int(g.status.isin(OUT).sum())}")


def comparator(since, now, paths=None):
    """same-period on-time WILLY-class fills (WILLY + HT, CLOSED, operator-closed out) from the orders exports, per book."""
    try:
        import scout_willy_hold_off as WHO
        a = WHO.load(paths)
        if not len(a):
            return ["Comparator: no orders export read"]
        c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since) & (a.o <= now) & a.es.isin(V2.STRATEGIES) & a.pct.notna()]
        c = _GK.split_operator_closed(c)[0] if len(c) else c
        out = []
        for lab, g in (("live", c[~c.paper] if len(c) else c), ("paper", c[c.paper] if len(c) else c)):
            out.append(f"{lab} N {len(g)}" + (f" · {g.o.dt.date.nunique()} days · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · "
                                               f"Σ {g.pct.sum():+.2f} pts" if len(g) else ""))
        return [f"Comparator — on-time WILLY + WILLY_HT fills, same period (≥ {since:%m-%d %H:%M} UTC, CLOSED, operator-closed out, booked "
                f"pnl_percentage; R_NOMISS fills included): " + " · ".join(out)]
    except Exception as e:
        return [f"Comparator unavailable ({str(e)[:80]})"]


def _fmt(v, f="{:+.2f}"):
    x = _f(v)
    return "n/a" if x is None else f.format(x)


def write_csv(d, path):
    if not path:
        return
    try:
        tmp = f"{path}.{os.getpid()}.tmp"
        d.reindex(columns=list(CSV_COLS)).to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        pass


BLIND = ("Could NOT test: engine-only guards (WILLY / HT slot, pair already held, dislocation / lateness guards, turnover refusal, global "
         "hold, the episode ending inside the wait) · the 25-pair shortlist cap timing — LIQUID also covers restart / FRENZY-paused / "
         "pre-filter cold reads ('not the volume floor', cause unseen) · turnover routing WILLY vs WILLY_HT ¼ (no offline supply source any "
         "scout uses — every refusal priced as a full position) · late reads with NO journal row (no-red check unreadable, late unreadable, "
         "stored state unread, WILLY or entry A off) · R_NOMISS entries killed at the red bar by the $ floor (FRENZY_WILLY_A_NOMISS_VOL24_LOW: "
         "counted in scout_willy_nomiss, priced nowhere) · _OFF rows are R_NOMISS-eligible but priced on R1 here · late reads before the "
         "DL 350 deploy (never journaled; CloudWatch only) · 1m-only paths kept (no tick-provisional hold) · TP touch tested on a taker-exit "
         "net path (≈ 0.03 % conservative, inherited from the V2b scout) · a refusal not yet in a decisions export (e.g. LIGHTUSDT 10-24 "
         "13:05 needs the next export).")


def run(now_ms=None, dec_paths=None, since_ms="auto", k5_fn=None, path_fn=None, fetch=True, state_path=STATE, csv_path=CSV_OUT,
        orders_paths=None, cfg=None, fetch_final=None):
    L = [f"## 🕰 WILLY LATE-JOIN (low-volume late spikes) — DECISION_LOG {DL}, observe-only", ""]
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if since_ms == "auto":
        since_ms = _GK.dl_deploy_ms(NOMISS_DL, lag_min=LAG_MIN, label="WILLY_LATEJOIN")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {NOMISS_DL} deploy commit is not in git; the refusal journal starts there.", BLIND, ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    cfg = cfg or read_cfg()
    ref, n_bad, newest_ms = read_refusals(dec_paths, since)
    d, notes = evaluate(ref, cfg, now_ms, k5_fn=k5_fn, path_fn=path_fn, fetch=fetch, fetch_final=fetch_final)
    write_csv(d, csv_path)
    L.append(f"Cohort: journal e = WILLY refusals {' / '.join(x.replace('FRENZY_WILLY_A_NOMISS', '') for x in EVENTS)} ≥ {since:%Y-%m-%d %H:%M} UTC "
             f"(DL {NOMISS_DL} deploy + {LAG_MIN} min) · {len(d)} (pair, spike bar) after dedup · paper {int((d['mode'] == 'paper').sum()) if len(d) else 0} / "
             f"live {int((d['mode'] == 'live').sum()) if len(d) else 0} (mode = oldest export holding the row; pooled in the gate)"
             + (f" · {n_bad} unreadable rows skipped" if n_bad else ""))
    L.append(f"Rule: armed at the join bar → first red 5m bar within {cfg['wait']:g} min · entry close × (1 + {SLIP_PCT:.2f} %) at close + "
             f"{LATENCY_MS // 1000} s · exit = live WILLY V2b (T = max(+{WEra.OLD_TP}, min(+{V2.TGT_MAX:g}, net at H × 0.99)) · stop −{WEra.OLD_STOP:g} · "
             f"late +{WEra.OLD_LATE[1]} @{WEra.OLD_LATE[0]:g} min · cap {WEra.OLD_HOLD}) · LOWVOL = vol24 at the spike < ${cfg['floor'] / 1e6:g}M")
    nt = int((d.status == "TIMING?").sum()) if len(d) else 0
    nsp = int((d.spike_ok == False).sum()) if len(d) else 0                     # noqa: E712 (None = not checked)
    L.append(f"Timing check: {nt} TIMING? (event > {TIMING_MAX_S} s into its bar — out of the gate) · spike bar not green on the fetched "
             f"5m bars: {nsp} (bar arithmetic suspect; listed, still counted) · status counts: "
             + (", ".join(f"{k} {int(v)}" for k, v in d.status.value_counts().sort_index().items()) if len(d) else "none"))
    L += ["", "| L bucket | class | refusals | priced | WR | avg % | Σ pts | expired | held | out |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b, _, _ in BUCKETS:
        for c in ("LOWVOL", "LIQUID", "UNREAD", "all"):
            g = d[(d.bucket == b) & ((d.cls == c) if c != "all" else True)] if len(d) else d
            if c != "all" and not len(g):
                continue
            p = g[g.status == "PRICED"] if len(g) else g
            L.append(f"| {b} | {c} | {len(g)} | {len(p)} | {(p.net_pct > 0).mean():.0%} | {p.net_pct.mean():+.3f} | {p.net_pct.sum():+.2f} | "
                     f"{int((g.status == 'EXPIRED').sum())} | {int(g.status.isin(HOLDS).sum())} | {int(g.status.isin(OUT).sum())} |" if len(p) else
                     f"| {b} | {c} | {len(g)} | 0 | – | – | – | {int((g.status == 'EXPIRED').sum()) if len(g) else 0} | "
                     f"{int(g.status.isin(HOLDS).sum()) if len(g) else 0} | {int(g.status.isin(OUT).sum()) if len(g) else 0} |")
    L.append("")
    v, txt, complete = gate(d, newest_ms)
    fv, ft = V2.frozen(state_path, GATE_KEY, v, txt, complete, True)
    L.append(f"🔒 FROZEN gate (L 4–12, all classes; the 2026-10-22 study's pre-registered gate: ≥ {GATE_N} priced on ≥ {GATE_DAYS} UTC days → "
             f"RE-OPEN CANDIDATE iff avg > 0 ∧ day-boot 95 % LB > 0, {BOOT_N:,} resamples seed {BOOT_SEED}; else KEEP; freezes only once the prefix's "
             f"last spike ≤ the newest decisions export − 65 min; never re-fit): {fv} — {ft}")
    L.append("Observe-only, no gate:")
    L.append(f"  LOWVOL only (all L): {summ(d[d.cls == 'LOWVOL'] if len(d) else d)}")
    L.append(f"  L 13–36 (LIGHTUSDT's case was L 28): {summ(d[d.bucket == '13-36'] if len(d) else d)}")
    L.append(f"  L 37+: {summ(d[d.bucket == '37+'] if len(d) else d)}")
    L.append("Study reference — NOT the same cohort (reports/WILLY_LATE_JOIN_2026-10-22.md, backtest R1-60 = L 4–12): 720 · 72 % WR · "
             "−0.267 %/trade (day-boot 5,000: −0.43 … −0.11) vs on-time −0.456 — it pooled missed AND no-red-missed joiners on the OLD +1.25 "
             "exit; here R_NOMISS takes the no-red-missed ones, so the gated L 4–12 rows are (almost all) _MISSED reads on the V2b exit "
             "(4,000 resamples). Live + master refusals then ①② 13 · Σ −0.9 pts.")
    L += comparator(since, pd.Timestamp(now_ms, unit="ms"), orders_paths)
    if len(d):
        L.append(f"Per refusal (last {LIST_CAP}): " + " · ".join(
            f"{r.pair} {str(r.spike_utc)[5:16]} L{r.L} vol24 {_fmt(None if r.vol24_usd is None else r.vol24_usd / 1e6, '${:.1f}M')} {r.cls} | "
            + (f"entry {str(r.entry_utc)[5:16]} T {_fmt(r.target)} {r.exit_why} {_fmt(r.net_pct)} %" if r.status == "PRICED"
               else f"{r.status}" + (f" ({r.exit_why})" if r.status == "OVERLAP" else ""))
            for r in d.tail(LIST_CAP).itertuples()))
    for n in notes[:6]:
        L.append(f"  ({n})")
    L.append(BLIND)
    if csv_path:
        L.append(f"CSV: {os.path.relpath(csv_path, _R) if csv_path.startswith(_R) else csv_path} (one row per refusal)")
    return L + [""]


def main():
    print("\n".join(run(fetch="--no-fetch" not in sys.argv)))


if __name__ == "__main__":
    main()
