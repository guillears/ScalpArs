#!/usr/bin/env python3
"""🚫🔭 Scout — EXH_NO_SPOT forward watch (DECISION_LOG 630, 2026-11-23, operator "yes build the no-spot scout"). OBSERVE / ALERT only:
never changes config, never calls the bot API; writes only reports/ (cache + frozen state, both gitignored).

QUESTION        EXHAUSTION_SHORT refuses every coin with no Binance spot market (EXH_NO_SPOT) BEFORE it reads any kline — the rule's edge is
                the spot share of volume falling, which a perp-only coin cannot have. The 11-18 backtests (NOSPOT_MCAP / SLEEVE / EXIT /
                HYBRID / BTCBEAR) found the trigger LOSES on no-spot coins (2,795 trades, −0.09 %/trade vs +0.43 with spot). This line asks
                the forward question on REAL live signals: would the no-spot setups that pass EVERY OTHER rule have paid under today's exit?
COHORT          e=EXHAUSTION journal rows with exh_block = EXH_NO_SPOT (both universes: FRENZY at exhaustion_min_vol_usd, WIDE at
                exhaustion_wide_min_vol_usd), read from the decisions EXPORTS (~/Downloads + exports_archive/ — the bot does not publish its
                journal; a signal is visible only once an export holding it exists), dedup (pair, signal time), signal time ≥ the
                DECISION_LOG 630 commit + 10 min (the rows the build was motivated by — 11-18 → 11-23 — are NOT counted). Signal-time order.
"PASSES ELSE"   (every RULE-level gate — the open path's refusals: slots, the depth guard, the late-order limit, a manual position, are
                not modelled) the engine skipped every other check on these rows, so the scout rebuilds them with the ENGINE's own code: the perp 5m window
                ending at T (services.exhaustion.KLINE_LIMIT bars, quote volume) → services.exhaustion.features(has_spot=False) → the
                re-read RSI cross · 24 h volume ≥ the row universe's floor · vol_decay ≥ exhaustion_vol_decay_min. Fail any → listed
                (OUT_NO_SIGNAL / OUT_VOL24 / OUT_VOL_DECAY / OUT_NO_DATA), never counted. Exit levels = services.exhaustion.exit_levels(ATR %).
ENTRY           the REAL first Binance trade at ≥ T + 8 s (aggTrades, the live latency ruler) moved 0.10 % against the short. Binance serves
                aggTrades for the last 2 days only, so the entry is fetched and CACHED on the first run that sees the signal: the binding
                constraint is a decisions EXPORT holding the signal AND a scout run within 2 days of it (the scout runs every 4 h); a row
                first seen > 2 days late → NO_ENTRY (listed, never priced with a proxy — the 11-21 BLESS fill sat 5.5 % below the candle
                close). The NO_ENTRY count therefore measures the export cadence.
EXIT            today's EXHAUSTION exit (pinned in observe_registry → ⚠ STALE on a change; the hold = exhaustion_max_hold_hours): the 1m
                bars from the first WHOLE minute after the entry (as scout_exh_exit_override.price_fill — the entry minute holds pre-fill
                prints) · stop first (gap-aware, taker 0.05 % + 0.10 % slip) · TP a resting maker 0.02 % filled when a 1m low trades
                through · else the close of the last minute of the hold (taker + slip) · taker 0.05 % on the entry. FUNDING priced: the
                settlements after the entry and up to the exit (a stop / TP: up to the exit minute's start — the intra-minute moment is
                unknown; a time exit: up to its end + 1 s) — a short earns + rate, pays a negative rate. One shadow per pair: a setup inside
                an earlier shadow's life → SHADOWED (listed, never counted); a setup that would have traded but has no price (NO_ENTRY / a gapped
                path) holds its coin for the whole hold, as live would.
CAVEATS         full size at aggTrade + 0.10 %: the depth guard (DL 583 / 584 / 588 — SHRINK / SKIP on the thinner side) is NOT applied,
                and no-spot coins are the thin ones (Route B 11-22: $10k costs 24–30 bps a side on H / BULLA / NAORIS) → the priced mean is
                an UPPER bound on what live would book.
GATE (frozen)   pre-committed 2026-11-23, read ONCE, alert-only: at the first prefix of ≥ 30 priced setups spanning ≥ 10 UTC days → the
                one-sided 80 % LOWER bound of the mean (day-block bootstrap, seed 630) > 0 → REOPEN (the no-spot study is re-opened on
                fresh evidence — NOT a switch: there is no toggle; any change goes through a full study + the locked bars) · else KEEP
                (the block stands). Frozen in reports/SCOUT_EXH_NOSPOT_630.json. Counted setups are always a signal-time PREFIX: pricing
                pauses at the first setup whose path is still open (later setups' features + entry are still captured) and stops on a
                network ceiling; a coin Binance answers 400 / 404 for (delisted) is NO_DATA for good, never a permanent pause. Shown
                beside (information): the kept EXHAUSTION fills over the priced window, price-% from real entry / exit, fees in (FRENZY-universe fills only — the shared loader drops WIDE).
NETWORK         fapi klines 5m (weight 10) + aggTrades (weight 20) + 1m klines (weight 10) + fundingRate (weight 1) per setup, ≥ 0.6 s
                apart, stops above 1,000 used weight / on 418 / 429 (scout_exh_exit_override.NetStop). Results cached in
                reports/.exh_nospot_630_cache.json (gitignored; written atomically, saved even when a run fails; an unreadable cache is
                never overwritten — the run reports NOT EVALUABLE).
Usage: venv/bin/python -I scripts/scout_exh_nospot_630.py"""
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)

import scout_exh_exit_override as XO          # noqa: E402  fetch_1m / path_bars / NetStop / TAKER / MAKER / UA / the weight ruler
from services import exhaustion as X          # noqa: E402  features / exit_levels / exhaustion_params / max_hold_minutes (the ENGINE)

DL = 630
BLOCK = "EXH_NO_SPOT"
LAT_MS = 8_000                                 # the live latency ruler (signal close → order)
SLIP = 0.10                                    # % against the short, on the entry and on a taker exit
AGG_WINDOW_MS = 2 * 86_400_000                 # Binance aggTrades: "search window is restricted to recent 2 days only"
GONE_HTTP = (400, 404)
GAP_GRACE_MS = 12 * 3_600_000                   # a short / gapped 1m answer is retried for 12 h (the hook runs every 4 h) before NO_DATA                         # Binance's answer for a delisted / unknown symbol → NO_DATA for good
GATE_N, GATE_DAYS = 30, 10
LB_Q, BOOT_N, BOOT_SEED = 0.80, 2000, 630
CACHE = os.path.join(_R, "reports", ".exh_nospot_630_cache.json")
STATE = os.path.join(_R, "reports", "SCOUT_EXH_NOSPOT_630.json")
CONFIG = os.path.join(_R, "trading_config.json")
OUT_CODES = ("OUT_NO_DATA", "OUT_NO_SIGNAL", "OUT_VOL24", "OUT_VOL_DECAY")
FINAL = ("PRICED", "NO_ENTRY", "NO_DATA") + OUT_CODES


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(ts):
    import pandas as pd
    t = pd.to_datetime(str(ts)[:19], errors="coerce")
    return None if pd.isna(t) else int(t.value // 1_000_000)


def _uni(v):
    return "WIDE" if str(v or "").strip().upper() == "WIDE" else "FRENZY"   # NaN / blank / anything else → FRENZY (the engine's default)


def _load(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except FileNotFoundError:
        return {}
    except Exception:
        return None                                               # unreadable → never freeze / overwrite over it


def _save(d, path):
    """atomic: a temp file in the same folder, then os.replace (a crash mid-write never corrupts the cache / the frozen state)."""
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def thresholds(path=CONFIG):
    """the live trading_config.json thresholds as an attribute object (exhaustion_params falls back to the study's value on a bad one)."""
    from types import SimpleNamespace
    try:
        with open(path) as f:
            th = (json.load(f) or {}).get("thresholds") or {}
    except Exception:
        th = {}
    return SimpleNamespace(**th)


def hold_minutes(th):
    """the live hold (exhaustion_max_hold_hours, via the engine's own reader)."""
    return int(X.max_hold_minutes(th))


# ── ① would the setup have passed every rule but the spot ones? (the engine's features) ───────────────────────────────────────────

def passes_else(f, universe, th):
    """features() dict → (True, None) or (False, OUT_*). Mirrors services.exhaustion.decide's order without the spot checks: data → the
    re-read cross → 24 h volume ≥ the universe floor → vol_decay. Pure."""
    p = X.exhaustion_params(th)
    if not f or not f.get("ok"):
        return False, "OUT_NO_DATA"
    if not f.get("cross"):
        return False, "OUT_NO_SIGNAL"
    floor = p["wide_min_vol"] if _uni(universe) == "WIDE" else p["min_vol"]
    if f.get("vol24") is None or f["vol24"] < floor:
        return False, "OUT_VOL24"
    vd = f.get("vol_decay")
    if vd is None or vd < p["vol_decay_min"]:
        return False, "OUT_VOL_DECAY"
    return True, None


# ── ② the walk (pure) ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def walk(entry, bars, tp_pct, sl_pct, hold, slip=SLIP):
    """SHORT filled at `entry` (already moved against the short) over 1m bars [(o, h, l, c)] from the first whole minute after the fill →
    (net % before funding, 'SL' / 'TP' / 'TIME', exit bar index) · (None, 'OPEN', None) when the path is short. Stop first (gap-aware:
    the worse of the stop and the bar open, taker + slip), TP maker when a low trades THROUGH, else the close of bar hold−1 (taker +
    slip). Entry taker fee. Pure."""
    e = _f(entry)
    if e is None or e <= 0 or not bars:
        return None, "NO_DATA", None
    tp_px, sl_px = e * (1 - tp_pct / 100.0), e * (1 + sl_pct / 100.0)
    for i, (o, h, l, c) in enumerate(bars[:hold]):
        if h >= sl_px:
            return (1 - max(sl_px, o) / e) * 100 - XO.TAKER - XO.TAKER - slip, "SL", i
        if l < tp_px:
            return (1 - tp_px / e) * 100 - XO.TAKER - XO.MAKER, "TP", i
    if len(bars) < hold:
        return None, "OPEN", None
    return (1 - bars[hold - 1][3] / e) * 100 - XO.TAKER - XO.TAKER - slip, "TIME", hold - 1


def funding_pct(rates, entry_ms, until_ms):
    """% the SHORT earns from the settlements in (entry_ms, until_ms]: Σ rate × 100 (a negative rate is paid). rates = [(time_ms, rate)]. Pure."""
    return sum(r * 100.0 for t, r in rates if entry_ms < t <= until_ms and r is not None and math.isfinite(r))


def funding_until(t1, i, why):
    """the last settlement time counted: a stop / TP hit inside minute i (the moment in the minute is unknown) → that minute's START;
    a time exit at the close of minute i → its END + 1 s (Binance stamps fundingTime a few ms after the hour). Pure."""
    start = t1 + i * 60_000
    return start if why in ("SL", "TP") else start + 60_000 + 1_000


# ── ③ network (each read paced + weight-guarded through XO's ruler) ─────────────────────────────────────────────────────────────────

def _over(u):
    return u is not None and str(u).isdigit() and int(u) > XO.WEIGHT_STOP


def _get(url):
    if XO._weight_hit[0]:
        raise XO.NetStop(f"used weight > {XO.WEIGHT_STOP}")
    wait = 0.6 - (time.time() - XO._last_call[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": XO.UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            XO._last_call[0] = time.time()
            used = r.headers.get("X-MBX-USED-WEIGHT-1M")
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        XO._last_call[0] = time.time()
        if e.code in (418, 429) or _over(e.headers.get("X-MBX-USED-WEIGHT-1M") if e.headers else None):
            raise XO.NetStop(f"HTTP {e.code}")
        raise
    if _over(used):
        XO._weight_hit[0] = True
    return out


def fetch_5m(pair, T):
    return _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&endTime={int(T) - 1}&limit={X.KLINE_LIMIT}")


def fetch_entry(pair, t_ms):
    """the first aggTrade at ≥ t_ms (searching 60 s) → price, or None (no print in that minute)."""
    a = _get(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={pair}&startTime={int(t_ms)}&endTime={int(t_ms) + 60_000}&limit=1")
    return _f(a[0]["p"]) if a else None


def fetch_funding(pair, a_ms, b_ms):
    rows = _get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}&startTime={int(a_ms)}&endTime={int(b_ms)}&limit=100")
    return [(int(r["fundingTime"]), _f(r.get("fundingRate"))) for r in rows or []]


NET = dict(fetch_5m=fetch_5m, fetch_entry=fetch_entry, fetch_1m=lambda p, t, n: XO.fetch_1m(p, t, n), fetch_funding=fetch_funding)


# ── ④ one setup ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def price_setup(r, th, cache, now_ms, net=None, capture_only=False):
    """one journal row (dict with pair, _t, _spike, exh_universe) → the cache entry under pair|T: status ∈ FINAL, or None = not finished
    yet (path open — retried next run). Each stage is cached as soon as it is known (the features verdict, the entry price), so the
    2-day aggTrades window is used once. capture_only: stop after the entry (a later setup while the prefix waits)."""
    net = net or NET
    pair, T = str(r["pair"]), int(r["_t"])
    c = cache.setdefault(f"{pair}|{T}", {})
    if c.get("status") in FINAL:
        return c
    if "ok" not in c:                                             # features → the "passes every other rule" verdict
        sp = r.get("_spike")
        rows = net["fetch_5m"](pair, T)
        if not rows or sp is None:
            c.update(status="OUT_NO_DATA", why="perp 5m klines empty / spike time unreadable")
            return c
        ft = X.features(rows, [], T, int(sp) - X.BAR_MS, False, th)   # the engine passes the spike bar's OPEN (spike close − 5 min)
        ok, code = passes_else(ft, r.get("exh_universe"), th)
        tp, sl = X.exit_levels(ft.get("atr_pct"), th)
        c.update(ok=ok, vol24=ft.get("vol24"), vol_decay=ft.get("vol_decay"), atr_pct=ft.get("atr_pct"), tp=tp, sl=sl)
        if not ok:
            c["status"] = code
            return c
        if tp is None or sl is None:
            c.update(status="OUT_NO_DATA", why="ATR undefined")
            return c
    if "entry" not in c:
        if now_ms - (T + LAT_MS) > AGG_WINDOW_MS - 600_000:
            c.update(status="NO_ENTRY", why="first seen after the 2-day aggTrades window")
            return c
        px = net["fetch_entry"](pair, T + LAT_MS)
        if px is None:
            c.update(status="NO_ENTRY", why="no trade printed in the minute after T + 8 s")
            return c
        c["entry"] = px * (1 - SLIP / 100.0)
    if capture_only:
        return None
    hold = hold_minutes(th)
    t1 = (T + LAT_MS) // 60_000 * 60_000 + 60_000                 # the first WHOLE minute after the fill
    if now_ms < t1 + hold * 60_000 + 60_000:
        return None                                               # the path is not over yet
    bars = XO.path_bars(net["fetch_1m"](pair, t1, hold), t1, hold)
    if bars is None:
        if now_ms > t1 + hold * 60_000 + GAP_GRACE_MS:
            c.update(status="NO_DATA", why="1m path gapped")
            return c
        return None
    pct, why, i = walk(c["entry"], bars, c["tp"], c["sl"], hold)
    if pct is None:
        if why == "NO_DATA":
            c.update(status="NO_DATA", why="bad entry")
            return c
        return None
    until = funding_until(t1, i, why)
    fund = funding_pct(net["fetch_funding"](pair, T, until), T + LAT_MS, until)
    c.update(status="PRICED", pct=pct + fund, pct_ex_funding=pct, funding=fund, exit=why, exit_ms=t1 + (i + 1) * 60_000)
    return c


# ── ⑤ the gate (pure) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def lower_bound(rows, q=LB_Q, n_boot=BOOT_N, seed=BOOT_SEED):
    """one-sided q LOWER bound of the mean by a day-block bootstrap (whole days resampled, deterministic seed). Pure."""
    import random
    by = {}
    for d, x in rows:
        by.setdefault(d, []).append(x)
    days = list(by.values())
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        means.append(sum(sum(v) for v in pick) / sum(len(v) for v in pick))
    means.sort()
    return means[max(0, int(round((1 - q) * len(means))) - 1)]   # round: (1 − 0.8) × 2000 is 399.99… in floats


def gate(rows):
    """rows = [(utc day, net %)] in signal-time order → (verdict, text, n used or None). The first prefix with ≥ GATE_N rows on ≥ GATE_DAYS
    days decides: 80 % lower bound > 0 → REOPEN · else KEEP. Before: COLLECTING. Pure."""
    rows = [(d, x) for d, x in rows if x is not None and math.isfinite(x)]
    rule = (f"gate: the first {GATE_N}+ priced setups on ≥ {GATE_DAYS} UTC days → the one-sided {LB_Q:.0%} lower bound of the mean "
            f"(day-block bootstrap) > 0 → REOPEN the no-spot study · else KEEP the block — read once, frozen")
    for n in range(GATE_N, len(rows) + 1):
        days = len({d for d, _ in rows[:n]})
        if days >= GATE_DAYS:
            cut = [x for _, x in rows[:n]]
            m = sum(cut) / n
            wr = 100.0 * sum(1 for x in cut if x > 0) / n
            lb = lower_bound(rows[:n])
            info = f"first {n} setups ({days} days): mean {m:+.3f} % · {LB_Q:.0%} lower bound {lb:+.3f} % · WR {wr:.0f} %"
            if lb > 0:
                return "REOPEN", f"🔴 {info} → no-spot setups paid: re-open the no-spot study (no toggle — a full study decides)", n
            return "KEEP", f"🟢 {info} → not shown to pay: the EXH_NO_SPOT block stands", n
    return "COLLECTING", f"⏳ {len(rows)}/{GATE_N} priced on {len({d for d, _ in rows})}/{GATE_DAYS} days — {rule}", None


# ── ⑥ cohort + report ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def cohort(journal, floor_ms):
    """EXH_NO_SPOT journal rows at / after floor_ms (both universes), dedup (pair, signal time), signal-time order. floor None → []."""
    if floor_ms is None or journal is None or not len(journal):
        return []
    out, seen = [], set()
    for r in journal.to_dict("records"):
        if str(r.get("exh_block") or "") != BLOCK:
            continue
        t = _ms(r.get("exh_signal_at"))
        if t is None or t < floor_ms or (str(r.get("pair")), t) in seen:
            continue
        seen.add((str(r.get("pair")), t))
        r["_t"], r["_spike"] = t, _ms(r.get("exh_spike_at"))
        out.append(r)
    return sorted(out, key=lambda r: (r["_t"], str(r.get("pair"))))


def kept_price_pct(fills, a_ms, b_ms):
    """the kept EXHAUSTION fills opened in [a, b + 24 h]: price-% from real entry / exit (short), taker in + taker out (an upper bound on
    fees for a maker TP). → list of floats."""
    import pandas as pd
    if fills is None or not len(fills) or "entry_price" not in fills or "exit_price" not in fills:
        return []
    o = pd.to_datetime(fills.opened_at.astype(str).str[:19], errors="coerce")
    k = fills[(o >= pd.Timestamp(a_ms, unit="ms")) & (o <= pd.Timestamp(b_ms, unit="ms") + pd.Timedelta(hours=24))]
    e, x = pd.to_numeric(k.entry_price, errors="coerce"), pd.to_numeric(k.exit_price, errors="coerce")
    v = ((1 - x / e) * 100 - 2 * XO.TAKER).dropna()
    return [float(z) for z in v if math.isfinite(z)]


def _tag(r):
    return f"{r.get('pair')} {str(r.get('exh_signal_at'))[:16]}"


def _walk_cohort(coh, th, cache, now_ms, net):
    """→ (rows [(day, net, _t)], lines, listed {status: [tags]}, shadowed [tags], halted text or None). Fills `cache` in place."""
    rows, lines, listed, shadowed = [], [], {}, []
    busy_until, halted, halt_i = {}, None, None
    for idx, r in enumerate(coh):
        tag, pair = _tag(r), str(r["pair"])
        if busy_until.get(pair, -1) >= r["_t"]:
            shadowed.append(tag)
            continue
        try:
            c = price_setup(r, th, cache, now_ms, net)
        except XO.NetStop as e:
            halted = f"{tag}: {e}"
            break
        except urllib.error.HTTPError as e:
            if e.code in GONE_HTTP:                               # delisted / unknown symbol → NO_DATA for good, the run goes on
                c = cache.setdefault(f"{pair}|{int(r['_t'])}", {})
                c.update(status="NO_DATA", why=f"HTTP {e.code} (symbol gone)")
            else:
                halted = f"{tag}: HTTP {e.code}"
                break
        except Exception as e:                                    # anything else: pause the prefix, never crash the 4-h scout
            halted = f"{tag}: {type(e).__name__}"
            break
        if c is None:
            halted, halt_i = f"{tag}: path not complete yet", idx
            break
        st = c.get("status")
        if st != "PRICED":
            listed.setdefault(st, []).append(tag)
            if c.get("ok") and st in ("NO_ENTRY", "NO_DATA"):   # it WOULD have traded, its life is unknown → hold the coin for the whole
                busy_until[pair] = int(r["_t"]) + LAT_MS + (hold_minutes(th) + 1) * 60_000   # hold (live = PAIR_HELD); never priced
            continue
        busy_until[pair] = int(c["exit_ms"])
        day = time.strftime("%Y-%m-%d", time.gmtime(r["_t"] / 1000))
        rows.append((day, float(c["pct"]), r["_t"]))
        lines.append(f"    {tag} [{_uni(r.get('exh_universe'))}] entry {c['entry']:.6g} · TP {c['tp']:.2f} / SL {c['sl']:.2f} · vol24 "
                     f"${(c.get('vol24') or 0) / 1e6:.1f}M · decay {c.get('vol_decay') or 0:.2f} → {c['pct']:+6.2f} % ({c['exit']}; funding "
                     f"{c['funding']:+.3f})")
    if halt_i is not None:                                        # the prefix waits, but the 2-day aggTrades window does not: capture the
        for r in coh[halt_i + 1:]:                                # later setups' features + entry now (never priced out of order)
            try:
                price_setup(r, th, cache, now_ms, net, capture_only=True)
            except urllib.error.HTTPError as e:
                if e.code in GONE_HTTP:
                    cache.setdefault(f"{r['pair']}|{int(r['_t'])}", {}).update(status="NO_DATA", why=f"HTTP {e.code} (symbol gone)")
                    continue
                break
            except Exception:
                break
    return rows, lines, listed, shadowed, halted


def report(journal, fills=None, now_ms=None, state_path=STATE, floor_ms="auto", th=None, net=None, cache_path=CACHE):
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    th = th if th is not None else thresholds()
    out = [f"🚫🔭 EXH_NO_SPOT forward watch (DECISION_LOG {DL}, observe) — no-spot EXHAUSTION setups that pass every other rule, priced on "
           f"today's exit (real trade at T + 8 s, 0.10 % slip, fees, funding; full size — no depth guard, an upper bound) · prior: "
           f"backtest 2,795 no-spot trades −0.09 %/trade (NOSPOT_MCAP 11-18); build sample 11-18 → 11-23 (not counted) 35 setups · "
           f"+0.18 % (last 2 days, real entries: 18 · −2.3 %)"]
    if floor_ms == "auto":
        import scout_gatekit as GK
        floor_ms = GK.dl_deploy_ms(DL, label=f"EXH_NO_SPOT watch (DECISION_LOG {DL})")
    if floor_ms is None:
        out.append(f"  ⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet: nothing counted")
        return "\n".join(out)
    cache = _load(cache_path)
    if cache is None:
        out.append(f"  ⚠ NOT EVALUABLE — the cache {os.path.basename(cache_path)} is unreadable: nothing priced, nothing overwritten "
                   f"(its saved entries cannot be fetched again after 2 days — repair or restore it)")
        return "\n".join(out)
    if time.time() - XO._last_call[0] > 60:                       # a ceiling another scout hit in this run still holds inside its minute
        XO._weight_hit[0] = False
    coh = cohort(journal, floor_ms)
    try:
        rows, lines, listed, shadowed, halted = _walk_cohort(coh, th, cache, now_ms, net)
    finally:
        _save(cache, cache_path)                                  # the captured entries survive any failure
    days_rows = [(d, x) for d, x, _ in rows]
    st = _load(state_path) if state_path else {}
    if st is not None and "verdict" not in st:
        v, txt, n = gate(days_rows)
        if n is not None:
            st = dict(verdict=v, detail=txt, n=n, run_at=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now_ms / 1000)))
            if state_path:
                _save(st, state_path)
    if st and st.get("verdict"):
        out.append(f"  FROZEN ({st.get('run_at')}): {st.get('verdict')} — {st.get('detail')} (alert only · operator decides)")
    else:
        v, txt, _ = gate(days_rows)
        out.append(f"  verdict: {v} — {txt}" + (" (⚠ state file unreadable — nothing frozen this run)" if st is None else ""))
    if rows:
        xs = [x for _, x, _ in rows]
        out.append(f"  running: N {len(xs)} · WR {100.0 * sum(1 for x in xs if x > 0) / len(xs):.0f} % · mean {sum(xs) / len(xs):+.3f} % · "
                   f"Σ {sum(xs):+.2f} % · {len({d for d, _, _ in rows})} days · {len({ln.split()[0] for ln in lines})} coins")
        kp = kept_price_pct(fills, min(t for _, _, t in rows), max(t for _, _, t in rows))
        if kp:
            out.append(f"  (information) kept EXHAUSTION fills (FRENZY universe) over the priced window: N {len(kp)} · WR "
                       f"{100.0 * sum(1 for x in kp if x > 0) / len(kp):.0f} % · mean {sum(kp) / len(kp):+.3f} % (price %, fees in, no funding)")
    else:
        out.append(f"  running: no priced setup yet ({len(coh)} no-spot signals in the cohort)")
    if halted:
        out.append(f"  ⏸ pricing paused at {halted} — later setups wait for the next run (the counted setups stay a prefix)")
    for code in OUT_CODES + ("NO_ENTRY", "NO_DATA"):
        if listed.get(code):
            out.append(f"  {code} ({len(listed[code])}, not counted): {', '.join(listed[code][:12])}{' …' if len(listed[code]) > 12 else ''}")
    if shadowed:
        out.append(f"  SHADOWED (inside an earlier priced shadow of the same coin, not counted): {', '.join(shadowed)}")
    out.append("  ⚠ each entry needs a decisions export holding the signal + a scout run within 2 days of it (aggTrades window) — "
               "NO_ENTRY counts measure the export cadence")
    out.extend(lines)
    return "\n".join(out)


def load_journal(paths=None):
    """the e=EXHAUSTION journal rows of every decisions export (both universes — the shared scout_exhaustion_short.load_journal drops WIDE
    and does not read exh_spike_at), dedup (pair, signal time), the newest export wins."""
    import pandas as pd
    import scout_exhaustion_short as SC
    cols = set(SC.DEC_COLS) | {"exh_spike_at"}
    paths = paths if paths is not None else sorted({p for pat in SC.DEC_PATTERNS for p in SC._EG.glob_exports(pat)}, key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if "e" in x and "exh_status" in x:
            fr.append(x[x.e.astype(str) == "EXHAUSTION"])
    if not fr:
        return pd.DataFrame(columns=sorted(cols))
    return pd.concat(fr, ignore_index=True).drop_duplicates(["pair", "exh_signal_at"], keep="last")


def load():
    import scout_exhaustion_short as SC
    return load_journal(), SC.load_fills(extra=("entry_price", "exit_price"))


def run(now_ms=None):
    j, f = load()
    return [f"## 🚫🔭 EXH_NO_SPOT forward watch (DL {DL})", "", "```", report(j, f, now_ms=now_ms), "```", ""]


def main():
    j, f = load()
    print(report(j, f))


if __name__ == "__main__":
    main()
