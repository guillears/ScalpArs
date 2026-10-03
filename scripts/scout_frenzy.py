#!/usr/bin/env python3
"""🔥 Scout — FRENZY watch (operator, 2026-10-03: "make sure scout considers everything in frenzy pairs").

READ-ONLY, public Binance data + the bot's own decision exports. Called by scripts/opportunity_scout.py every run (never breaks it).

PAIRS   every pair FRENZY can be watching: the bot's shortlist rule NOW (|24 h change| OR 24 h low→high range ≥ frenzy_shortlist_change_pct,
        24 h volume ≥ frenzy_min_volume_usd, eligibility filters, blacklists incl. frenzy_pair_blacklist) PLUS every pair the bot's decision
        exports show it followed in the last 26 h (FRENZY order-book rows, FRENZY_* refusals, FRENZY fills).
REPLAY  the bot's own pure rules (services.frenzy) on each CLOSED 5m bar of the last 24 h, each on the last 1,500 bars up to it (as live) and the
        pair's normal hour anchored at that bar → every FRESH setup (the first candle the bot may enter): FRENZY (ready) · WIDE (FRENZY refused
        only for ATR / a green candle) · — (refused by both). Market volume on that bar = services.frenzy.global_volume_ratio over the top-50
        pairs by 24 h volume (BTC / ETH / blacklisted included, as the live gate).
OUTCOME the FRENZY exit on 5m bars from the next bar's open, in fee-NET space as live (frenzy_exit_for on net P&L: stop at −stop net, trail armed
        at +arm net, closes give % of price below the best), a bar opening through the line fills at its open; low before high inside a bar
        (conservative); costs 0.11 % (0.09 fees + 0.02 slip). "open" = still running at the last closed bar. These are REPLAY outcomes.
BOT     what the bot recorded for THAT bar (journal BLOCK bucket = the signal close, + one bar for a retry pass; an OPEN within 3 min of the
        close), sleeve-specific: FRENZY ↔ FRENZY_LONG fill / FRENZY_* gate · WIDE ↔ FRENZY_WIDE fill / FRENZY_WIDE_* gate. A record one bar
        earlier / later is reported as "bot fired ±1 bar" (the replay and live disagree on the fresh bar).
STATUS  per setup: ok (bot recorded it) · ⚠ MISMATCH (the bot was following the pair then — FRENZY order-book rows within 10 min — the sleeve
        was live and on, its journal covers the bar, yet nothing was recorded: a code-path gap, check the server log) · NOT FOLLOWED (the
        bot was not watching the pair then: a shortlist miss, the MOVR class) · sleeve not live then (WIDE before its deploy) · no export.
        The market-volume gate is judged only on bars after its deploy; earlier bars say "gate not live".
"""
import glob
import os
import time
from types import SimpleNamespace

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAR = 300_000
CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY.csv")
COST = 0.11
# deploy times (UTC ms) of the live switches the replay judges — before them the bot could not have done it
WIDE_LIVE_MS = int(pd.Timestamp("2026-10-03 22:10", tz="UTC").value // 1_000_000)   # 0e6a5a0 pushed 21:57 UTC + deploy
GVOL_LIVE_MS = int(pd.Timestamp("2026-10-03 22:30", tz="UTC").value // 1_000_000)   # 0d79904 pushed 22:14 UTC + deploy
WINDOW = 1500                   # live frenzy_walk window (bars)


def _th(cfg):
    return SimpleNamespace(**cfg)


def _eligible(EX, retry, cfg):
    """{pair: dict(symbol, qv, chg, rng, coin_ok)} — tickers after the bot's eligibility filters (blacklists NOT applied here)."""
    mk = retry(EX.load_markets) or {}
    tk = retry(EX.fetch_tickers) or {}
    nl_days = int(cfg.get("new_listing_filter_days", 0) or 0)
    alpha = bool(cfg.get("alpha_subtype_filter_enabled", False))
    coin = bool(cfg.get("coin_underlying_only", False))
    now = time.time() * 1000
    out = {}
    for s, t in tk.items():
        if not s.endswith("/USDT:USDT") or t.get("last") is None:
            continue
        info = (mk.get(s, {}) or {}).get("info", {}) or {}
        if coin and info.get("underlyingType") not in (None, "COIN"):
            continue
        try:
            ob = info.get("onboardDate")
            if nl_days > 0 and ob is not None and now - int(ob) < nl_days * 86400_000:
                continue
        except (TypeError, ValueError):
            pass
        sub = info.get("underlyingSubType") or []
        if alpha and any("alpha" in str(x).lower() for x in (sub if isinstance(sub, list) else [sub])):
            continue
        hi, lo = t.get("high"), t.get("low")
        try:
            rng = (float(hi) / float(lo) - 1) * 100 if hi and lo and float(lo) > 0 and float(hi) >= float(lo) else 0.0
        except (TypeError, ValueError):
            rng = 0.0
        out[s.split("/")[0] + "USDT"] = dict(symbol=s, qv=float(t.get("quoteVolume") or 0), chg=float(t.get("percentage") or 0), rng=rng)
    return out


def _exports(now_ms, hours=27):
    """From the decision exports of the last `hours`: FRENZY gate / fill rows, FRENZY order-book minutes per pair, journal heartbeats."""
    rows, book, beats = [], {}, set()
    for f in sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))):
        if time.time() - os.path.getmtime(f) > 3 * 86400:
            continue
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "dir", "gate", "strategy", "src", "n"))
        except Exception:
            continue
        if not {"t", "e", "pair"} <= set(d.columns):
            continue
        for c in ("gate", "strategy", "src"):
            if c not in d:
                d[c] = None
        ms = (pd.to_datetime(d.t, utc=True, errors="coerce", format="ISO8601") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
        d = d.assign(ms=ms)[ms.notna() & (ms >= now_ms - hours * 3600_000)]
        if not len(d):
            continue
        d["ms"] = d.ms.astype("int64")
        beats |= set(d[d.e == "SCAN"].ms)
        b = d[(d.e == "BOOK") & (d.src.astype(str) == "FRENZY")]
        for p, g in b.groupby(b.pair.astype(str)):
            book.setdefault(p, set()).update(g.ms.tolist())
        g = d[(d.e == "BLOCK") & d.gate.astype(str).str.startswith("FRENZY")]
        o = d[(d.e == "OPEN") & d.strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE", "MANUAL"])]
        rows.append(pd.concat([g, o]))
    ev = pd.concat(rows, ignore_index=True).drop_duplicates(["t", "e", "pair", "gate", "strategy"]) if rows else pd.DataFrame(
        columns=["t", "e", "pair", "gate", "strategy", "ms"])
    return ev, book, beats


def _followed_pairs(ev, book):
    return set(book) | set(ev[ev.e.isin(["BLOCK"]) | ev.strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"])].pair.astype(str))


def _walk_exit(bars, i0, th):
    """FRENZY exit in fee-NET space from bars[i0] open → (net %, minutes, how). Low before high inside a bar; a bar opening through the line
    fills at its open."""
    stop = float(getattr(th, "frenzy_stop_pct", 3.0) or 3.0); arm = float(getattr(th, "frenzy_trail_arm_pct", 5.0) or 5.0)
    give = float(getattr(th, "frenzy_trail_giveback_pct", 1.5) or 1.5); cap_min = int(getattr(th, "frenzy_max_hold_minutes", 720) or 720)
    cap = max(1, cap_min // 5)
    if i0 >= len(bars):
        return None, 0, "no bar yet"
    e = float(bars[i0][1]); pk_px = e; armed = False
    net = lambda px: (px / e - 1) * 100 - COST
    for k in range(i0, min(len(bars), i0 + cap)):
        o, h, l, c = (float(x) for x in bars[k][1:5])
        line_px = pk_px * (1 - give / 100) if armed else e * (1 + (COST - stop) / 100)
        if l <= line_px:
            return net(min(o, line_px)), (k - i0 + 1) * 5, ("trail" if armed else "stop")
        pk_px = max(pk_px, h)
        armed = armed or net(pk_px) >= arm
    n = min(len(bars), i0 + cap) - i0
    return net(float(bars[i0 + n - 1][4])), n * 5, (f"{cap_min // 60} h cap" if n >= cap else "open")


def _klines(EX, retry, sym, tf, n, end_ms):
    """≥ n closed bars ending at end_ms (two calls when n > 1500)."""
    step = BAR if tf == "5m" else 3_600_000
    out = {}
    since = end_ms - (n - 1) * step
    while since <= end_ms and len(out) < n + 5:
        r = retry(EX.fetch_ohlcv, sym, tf, since=since, limit=1500) or []
        if not r:
            break
        for x in r:
            if int(x[0]) <= end_ms:
                out[int(x[0])] = x
        nxt = int(r[-1][0]) + step
        if nxt <= since or len(r) < 2:
            break
        since = nxt
        time.sleep(0.05)
    return [out[k] for k in sorted(out)]


def scan(EX, retry, cfg, last_closed, alts, btc_full, now_ms):
    """→ (setups, flagged now, pairs checked, shortlist now, followed). Never raises past the caller's try."""
    from services.frenzy import (frenzy_walk, frenzy_flagged, frenzy_long_status, frenzy_wide_ready, normal_hour_usd, global_volume_ratio,
                                  frenzy_di_spread, frenzy_adx_delta, frenzy_vol_trend)
    from services.indicators import calculate_indicators
    from services.surge import wilder_atr_pct
    th = _th(cfg)
    el = _eligible(EX, retry, cfg)
    chg = float(cfg.get("frenzy_shortlist_change_pct", 15.0) or 0); vmin = float(cfg.get("frenzy_min_volume_usd", 20e6) or 0)
    skip = {"BTCUSDT", "ETHUSDT"}
    for k in ("pair_blacklist", "no_trade_pairs", "frenzy_pair_blacklist"):
        skip |= {x.strip().upper() for x in str(cfg.get(k) or "").split(",") if x.strip()}
    short = [p for p, r in sorted(el.items(), key=lambda kv: (abs(kv[1]["chg"]) < chg, -max(abs(kv[1]["chg"]), kv[1]["rng"])))
             if p not in skip and p.isascii() and max(abs(r["chg"]), r["rng"]) >= chg and r["qv"] >= vmin][:25]
    ev, book, beats = _exports(now_ms)
    followed = _followed_pairs(ev, book)
    pairs = list(dict.fromkeys(short + [p for p in sorted(followed) if p in el and p not in skip]))[:45]
    # market volume: the live gate's universe — top-50 by 24 h volume, BTC / ETH / blacklisted included
    gv_src = {}
    for p in [p for p, _ in sorted(el.items(), key=lambda kv: -kv[1]["qv"])[:50]]:
        d = alts.get(p) if p != "BTCUSDT" else btc_full
        if d is not None and len(d) >= 400:
            gv_src[p] = [[int(t), r.o, r.h, r.l, r.c, r.v] for t, r in d.tail(700).iterrows()]
        else:
            r = retry(EX.fetch_ohlcv, el[p]["symbol"], "5m", limit=400) or []
            gv_src[p] = [x for x in r if int(x[0]) <= last_closed]
            time.sleep(0.05)
    gmax = float(cfg.get("frenzy_gvol_max", 0) or 0)
    on_long = bool(cfg.get("frenzy_long_enabled", False)); on_wide = bool(cfg.get("frenzy_wide_enabled", False))
    setups, flagged = [], []
    for p in pairs:
        sym = el[p]["symbol"]
        b5 = _klines(EX, retry, sym, "5m", WINDOW + 300, last_closed)
        h1 = _klines(EX, retry, sym, "1h", 800, last_closed)
        if len(b5) < 600 or len(h1) < 250:
            continue
        closed = b5; n = len(closed); nh_cache = {}

        def nh_at(sig):
            hk = sig // 3_600_000
            if hk not in nh_cache:
                nh_cache[hk] = normal_hour_usd(h1, sig)
            return nh_cache[hk]
        nh_now = nh_at(int(closed[-1][0]))
        ep_now = frenzy_walk(closed[-WINDOW:], nh_now, th) if nh_now else None
        if frenzy_flagged(ep_now, th):
            flagged.append(dict(pair=p, hours=ep_now.get("hours"), vs_avg=ep_now.get("vs_vwap_pct"), vol_x=ep_now.get("vol_mult"),
                                on=bool(ep_now.get("in_state")), followed=p in followed))
        for k in range(max(WINDOW // 2, n - 288), n + 1):
            sig = int(closed[k - 1][0]); close_ms = sig + BAR
            nh = nh_at(sig)
            if not nh:
                continue
            ep = frenzy_walk(closed[max(0, k - WINDOW):k], nh, th)
            if not (ep and ep.get("fresh_on") and frenzy_flagged(ep, th)):
                continue
            atr = wilder_atr_pct(closed[k - 300:k])
            vol24 = sum(float(r[5]) * (float(r[2]) + float(r[3]) + float(r[4])) / 3 for r in closed[max(0, k - 288):k])
            ready, code, text = frenzy_long_status(ep, atr, vol24, th)
            sleeve = "FRENZY" if ready else ("WIDE" if frenzy_wide_ready(ep, code, SimpleNamespace(**{**cfg, "frenzy_wide_enabled": True}), atr) else "")
            gv = global_volume_ratio(gv_src, sig)
            if gmax <= 0:
                gate = "off"
            elif close_ms < GVOL_LIVE_MS:
                gate = "gate not live"
            else:
                gate = "unread" if gv is None else ("pass" if gv < gmax else "BLOCK")
            pnl, mins, how = _walk_exit(closed, k, th)
            try:
                ind = calculate_indicators([list(r) for r in closed[k - 300:k]]) or {}
            except Exception:
                ind = {}
            _e = lambda a, b: (round((ind[a] / ind[b] - 1) * 100, 3) if ind.get(a) and ind.get(b) else None)
            # what the bot recorded for THIS bar, sleeve-specific
            pe = ev[ev.pair.astype(str) == p] if len(ev) else ev

            def recs(lo, hi):
                g = pe[(pe.ms >= lo) & (pe.ms <= hi)] if len(pe) else pe
                fl = g[(g.e == "OPEN")] if len(g) else g
                return (sorted(set(fl.strategy.astype(str))) if len(fl) else []), (sorted(set(g[g.e == "BLOCK"].gate.astype(str))) if len(g) else [])
            fills, gates = recs(close_ms, close_ms + BAR + 3 * 60_000)
            pf, pg = recs(close_ms - BAR, close_ms - 1)
            nf, ng = recs(close_ms + 2 * BAR, close_ms + 2 * BAR + 3 * 60_000)
            bot_fills = [x for x in fills if x != "MANUAL"]
            want_wide = sleeve == "WIDE"
            mine = [x for x in bot_fills if x == ("FRENZY_WIDE" if want_wide else "FRENZY_LONG")] + \
                   [x for x in gates if (x.startswith("FRENZY_WIDE") if want_wide else not x.startswith("FRENZY_WIDE"))]
            covered = any(abs(b - close_ms) <= 2 * BAR for b in beats)
            following = any(abs(t - close_ms) <= 10 * 60_000 for t in book.get(p, ())) or bool(fills or gates)
            live = (on_wide and close_ms >= WIDE_LIVE_MS) if want_wide else on_long
            if not covered:
                status = "no export"
            elif mine:
                status = "ok"
            elif (pf or pg or nf or ng) and not (fills or gates):
                status = "bot fired ±1 bar"
            elif not sleeve:
                status = "ok (refused by both rules)" if gates or not following else "⚠ MISMATCH (no refusal recorded)"
            elif not live:
                status = "sleeve not live then"
            elif not following:
                status = "NOT FOLLOWED (shortlist miss?)"
            else:
                status = "⚠ MISMATCH"
            bot = ", ".join(bot_fills + gates) or ("— (±1 bar: " + ", ".join(pf + pg + nf + ng) + ")" if (pf or pg or nf or ng) else "nothing recorded")
            setups.append(dict(pair=p, signal_close_utc=pd.Timestamp(close_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"), bar_ts=sig,
                               spike_close_utc=pd.Timestamp(int(ep["spike_ts"]), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                               hours=round(ep["hours"], 1), vol_x=round(ep["vol_mult"] or 0), vs_avg=round(ep.get("vs_vwap_pct") or 0, 2),
                               atr=(round(atr, 2) if atr is not None else None), candle=round(ep.get("bar_ret_pct") or 0, 3),
                               status_rule=code, rule_text=text, sleeve=sleeve or "—", gvol=(round(gv, 3) if gv is not None else None), gvol_gate=gate,
                               pnl=(round(pnl, 2) if pnl is not None else None), held_min=mins, exit=how, bot=bot,
                               manual=("manual" if "MANUAL" in fills else ""), check=status, mismatch=status.startswith("⚠"),
                               followed_then=following, shortlisted_now=p in short,
                               run_pct=round(ep.get("run_pct") or 0, 2), gain_pct=round(ep.get("gain_pct") or 0, 2),
                               off_peak_pct=round(ep.get("off_peak_pct") or 0, 2), vwap=ep.get("vwap"), price=ep.get("price"), vol24_usd=round(vol24),
                               rsi=(round(ind["rsi"], 1) if ind.get("rsi") is not None else None),
                               adx=(round(ind["adx"], 1) if ind.get("adx") is not None else None), adx_delta=frenzy_adx_delta(closed[k - 300:k]),
                               di_spread=frenzy_di_spread(closed[k - 300:k]), vol_trend=frenzy_vol_trend(closed[:k]),
                               gap5_8=_e("ema5", "ema8"), gap5_20=_e("ema5", "ema20"), run_ms=now_ms))
    return setups, flagged, pairs, short, followed


def save(setups, pairs, now_ms):
    """reports/SCOUT_FRENZY.csv — one row per pair × signal bar, the latest run wins. A stored row inside this run's 24 h window on a pair this
    run checked that the replay no longer finds is marked superseded (kept for audit, left out of the cohort table)."""
    new = pd.DataFrame(setups)
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    if len(old):
        if "superseded" not in old:
            old["superseded"] = False
        keys = set(zip(new.pair, new.bar_ts)) if len(new) else set()
        win = (old.bar_ts.astype("int64") >= now_ms - 24 * 3600_000 + BAR) & old.pair.isin(pairs)
        gone = win & ~pd.Series([(a, int(b)) in keys for a, b in zip(old.pair, old.bar_ts)], index=old.index)
        old.loc[gone, "superseded"] = True
    if len(new):
        new["superseded"] = False
    allr = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(allr):
        allr = allr.drop_duplicates(["pair", "bar_ts"], keep="last").sort_values("bar_ts")
        tmp = CSV + ".tmp"; allr.to_csv(tmp, index=False); os.replace(tmp, CSV)
    return allr


def _f(v, fmt, none="–"):
    try:
        return format(float(v), fmt) if v is not None and v == v else none
    except (TypeError, ValueError):
        return none


def lines(setups, flagged, pairs, short, followed, cfg, hist=None):
    gmax = float(cfg.get("frenzy_gvol_max", 0) or 0)
    L = ["## 🔥 FRENZY watch (every pair FRENZY can be watching — last 24 h)", "",
         f"{len(pairs)} pairs checked ({len(short)} on the bot's shortlist rule now · {len(followed)} the bot's exports show it followed). Replay of "
         f"the bot's own rules on closed 5m bars (1,500-bar window, normal hour at each bar). Market volume = top-50 by 24 h volume on the signal "
         f"bar; gate < {gmax:g}× ({'judged from its deploy' if gmax > 0 else 'OFF'}). Result = REPLAY of the FRENZY exit, fee-net, 5m bars.", ""]
    if flagged:
        L += ["**Flagged now:** " + " · ".join(f"{x['pair']} {_f(x['hours'], '.0f')} h{' ON' if x['on'] else ''} ({_f(x['vs_avg'], '+.1f')}% vs avg, "
                                                f"vol {_f(x['vol_x'], '.0f')}×){'' if x['followed'] else ' — not in the bot exports'}"
                                                for x in sorted(flagged, key=lambda x: x["hours"] or 0)), ""]
    if not setups:
        L += ["No fresh FRENZY setup in the last 24 h.", ""]
    else:
        L += ["| Signal close UTC | Pair | h after spike | vol × | vs avg | ATR | candle | RSI · ADX Δ | FRENZY rule | Sleeve | Market vol | Result (replay) | Bot recorded | Check |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in sorted(setups, key=lambda s: s["bar_ts"]):
            res = f"{_f(s['pnl'], '+.2f')}% {s['exit']} ({s['held_min']} min)" if s["pnl"] is not None else s["exit"]
            gv = f"{_f(s['gvol'], '.2f')}× {s['gvol_gate']}" if s["gvol"] is not None else f"unread ({s['gvol_gate']})"
            L.append(f"| {s['signal_close_utc'][5:]} | {s['pair']} | {s['hours']} | {s['vol_x']} | {_f(s['vs_avg'], '+.2f')}% | {_f(s['atr'], '.2f')}% | "
                     f"{_f(s['candle'], '+.3f')}% | {_f(s['rsi'], '.0f')} · {_f(s['adx_delta'], '+.1f')} | {s['status_rule']} | {s['sleeve']} | {gv} | {res} | "
                     f"{s['bot']}{' · manual' if s['manual'] else ''} | {s['check']} |")
        L.append("")
        mm = [s for s in setups if s["mismatch"] or s["check"].startswith("NOT FOLLOWED")]
        if mm:
            L += [f"**⚠ {len(mm)} to investigate:** " + "; ".join(f"{s['signal_close_utc'][11:]} {s['pair']} {s['sleeve']} — {s['check']}" for s in mm), ""]
    if hist is not None and len(hist):
        h = hist[hist.pnl.notna() & (hist.exit.astype(str) != "open") & (hist.get("superseded", False) != True)].copy()  # noqa: E712
        if len(h):
            g = pd.to_numeric(h.gvol, errors="coerce")
            h["g"] = g.map(lambda v: "unread" if v != v else ("pass" if gmax <= 0 or v < gmax else "BLOCK"))

            def row(nm, z):
                return f"| {nm} | {len(z)} | {(z.pnl > 0).mean() * 100:.0f}% | {z.pnl.mean():+.3f}% |" if len(z) else f"| {nm} | 0 | – | – |"
            L += [f"**All recorded setups so far (SCOUT_FRENZY.csv, replay outcomes, judged against today's gate < {gmax:g}×):**", "",
                  "| cohort | setups | won | avg % |", "|---|---|---|---|",
                  row("FRENZY · market vol below the gate", h[(h.sleeve == "FRENZY") & (h.g == "pass")]),
                  row("FRENZY · market vol at / above", h[(h.sleeve == "FRENZY") & (h.g == "BLOCK")]),
                  row("WIDE · market vol below the gate", h[(h.sleeve == "WIDE") & (h.g == "pass")]),
                  row("WIDE · market vol at / above", h[(h.sleeve == "WIDE") & (h.g == "BLOCK")]),
                  row("market vol unread", h[h.g == "unread"]),
                  row("refused by both rules", h[h.sleeve == "—"]), ""]
    return L


def note_items(setups, noted, now_ms):
    """(key, line) for the scout notes — each MISMATCH / NOT FOLLOWED setup once."""
    out = []
    for s in setups:
        if s["mismatch"] or s["check"].startswith("NOT FOLLOWED"):
            k = f"FZM|{s['pair']}|{s['bar_ts']}"
            if k not in noted:
                out.append((k, f"🔥⚠ FRENZY {s['check']} {s['signal_close_utc'][5:]} {s['pair']}: the bot's rules say {s['sleeve'] or 'refuse'} "
                               f"(market vol {_f(s['gvol'], '.2f')}×) — {'check the server log' if s['mismatch'] else 'the bot was not watching the pair'}"))
    return out
