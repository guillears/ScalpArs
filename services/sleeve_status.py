"""🧭 Per-pair SLEEVE STATUS for the pairs panel's BLOCK REASON column (operator 11-18; no trading logic) — pure.

One line per ACTIVE sleeve that has something on the pair (a sleeve switched off never appears): 🪫 EXH · 🌐 WIDE · 📉 DIST · 🩳 FB · 🎲 HT ·
🔥 LONG (FRENZY_LONG only while it is enabled). Built from what the engine already holds in memory (the EXHAUSTION / FLATBASE tracking maps,
the WILLY pending map, the last 15m RSI EXHAUSTION read) and the LAST journal decision per sleeve (exhaustion_signals / distribution_signals /
flatbase_episodes) — never an exchange read. The panel and both text exports format the times in the app's clock (frenzyOnClock).

A line = dict(sleeve, icon, state, rank, at_ms, reason, detail):
  (EXH / DIST journal rows count only for the tracked episode — spike ≥ its spike, same universe; the RSI read ≤ 30 min old)
  state  open · signal (a decision < 1 h old that opened / passed) · blocked (a refused decision) · armed · waiting · tracking
  rank   0 open trade · 1 a signal or block in the last hour · 2 an older block · 3 waiting / tracking   (the panel shows rank order, then newest)
"""
import re as _re
import logging
from typing import Dict, Iterable, List, Optional

HOUR_MS = 3_600_000

SLEEVE_OF_STRATEGY = {"EXHAUSTION_SHORT": "EXH", "DISTRIBUTION_SHORT": "DIST", "FLATBASE_SHORT": "FB", "FRENZY_WILLY_HT": "HT",
                      "FRENZY_WILLY": "WILLY", "FRENZY_LONG": "LONG"}
PLAIN_OFF_TXT = "plain WILLY entries OFF"   # 🎲 (DECISION_LOG 587) the engine's WILLY_PLAIN_OFF outcome text (willy_last)
# 🎲 (DECISION_LOG 646, operator: "HT: WILLY A opened" read as an HT block) a PLAIN FRENZY_WILLY open-path outcome in willy_last — the
# setup passed turnover, so it was never an HT candidate: "WILLY A opened" / "WILLY A refused: …" / "WILLY A already entered". The routing
# texts that DO concern HT ("WILLY A armed · …", "… red bar refused: turnover …", "(R_NOMISS) refused") never match. HT's own texts
# ("WILLY HT A …") live in willy_ht_last.
PLAIN_OUTCOME_RE = _re.compile(r"\bWILLY [AB] (?:opened\b|refused: |already entered)")
ICON = {"EXH": "🪫", "WIDE": "🌐", "DIST": "📉", "FB": "🩳", "HT": "🎲", "WILLY": "🎲", "LONG": "🔥"}
RSI_MAX_AGE_MS = 30 * 60_000
FB_STATE_TEXT = {"SHADOW": "shadow · turnover ≤ 1, no order", "MISSED": "missed · the S4 close was not judged",
                 "EXPIRED": "expired · no S4 signal in the window", "SKIPPED": "skipped", "REFUSED": "refused"}

EXH_REASON = {"EXH_NO_SPOT": "no spot", "EXH_SSC_OIVOL": "OI filter", "EXH_SSC_OIVOL_UNREAD": "OI unread", "EXH_VOL_DECAY": "vol_decay",
              "EXH_SPOT_SHARE": "spot share", "EXH_VOL24_LOW": "24 h volume", "EXH_PAIR_HELD": "pair held", "EXH_NO_DATA": "no data",
              "EXH_NO_SIGNAL": "no cross", "EXH_LATE": "late", "EXH_BOOK_FULL": "book full", "EXH_READ_CAP": "read cap", "EXH_FAILED": "error",
              "EXH_OPEN_REFUSED": "open refused", "EXH_SPOT_MISMATCH": "spot mismatch",
              "EXH_BEAR80": "market too bearish", "EXH_BEAR80_UNREAD": "breadth unread", "EXH_EARLY_SPIKE": "too soon after spike"}   # 🔤 (589 re-review 5) · 🌡⏱ (650)
DIST_REASON = {"DIST_POS_CHG": "top traders not cutting", "DIST_POS_UNREAD": "positions unread", "DIST_NOT_VSW": "volume < 5×",
               "DIST_VOL24_LOW": "24 h volume", "DIST_PAIR_HELD": "pair held", "DIST_NO_DATA": "no data", "DIST_LATE": "late",
               "DIST_BOOK_FULL": "book full", "DIST_RSI1H_UNREAD": "RSI1h unread", "DIST_FAILED": "error", "DIST_OPEN_REFUSED": "open refused"}


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _on(th, key, default=False) -> bool:
    return getattr(th, key, default) is True


def active_sleeves(th) -> Dict[str, bool]:
    """the sleeves switched on now (a switched-off sleeve never shows a line)."""
    exh = _on(th, 'exhaustion_short_enabled')
    return dict(EXH=exh, WIDE=exh and _on(th, 'exhaustion_wide_enabled'), DIST=_on(th, 'distribution_short_enabled'),
                FB=_on(th, 'flatbase_short_enabled'),
                HT=_on(th, 'frenzy_willy_enabled') and getattr(th, 'frenzy_willy_ht_enabled', True) is not False,
                LONG=_on(th, 'frenzy_long_enabled'))


def _decision_line(sleeve, last, now_ms, reason_map, rsi_cut=None) -> Optional[dict]:
    """the line of a journaled decision (OPENED / BLOCKED / REFUSED). None when there is none."""
    if not last:
        return None
    at = _f(last.get('at_ms'))
    st = str(last.get('status') or '')
    blk = str(last.get('block') or '')
    if blk == "DIST_RSI1H":
        r = _f(last.get('rsi1h'))
        why = f"RSI1h {r:.0f} < {rsi_cut:g}" if (r is not None and rsi_cut) else "RSI1h below the cut"
    elif blk == "DIST_NOT_VSW":   # (operator 11-18) show how close the candle came: "red candle 4.7× (needs 5×)"
        m = _re.search(r"quote volume ([0-9]+(?:\.[0-9]+)?)× the 20-candle mean \(need ≥ ([0-9]+(?:\.[0-9]+)?)×", str(last.get('reason') or ''))
        if not m:
            why = reason_map.get(blk, "volume < 5×")
        elif float(m.group(1)) > float(m.group(2)):   # the multiple clearly passed → the candle was not red (= at 2 dp is ambiguous → shown as a near-miss)
            why = f"candle not red ({float(m.group(1)):.2f}×)"
        else:
            why = f"red candle {float(m.group(1)):.2f}× (needs {float(m.group(2)):g}×)"
    else:
        why = reason_map.get(blk) or (str(last.get('reason') or '')[:40] if st != "OPENED" else "")
    fresh = at is not None and now_ms - at <= HOUR_MS
    if st == "OPENED":
        return dict(sleeve=sleeve, icon=ICON[sleeve], state="signal", rank=1 if fresh else 2, at_ms=at, reason="opened",
                    detail=str(last.get('reason') or ''))
    return dict(sleeve=sleeve, icon=ICON[sleeve], state="blocked", rank=1 if fresh else 2, at_ms=at, reason=why,
                detail=str(last.get('reason') or ''))


def utc_prefix_ms(txt, now_ms):
    """an engine text that starts "MM-DD HH:MM" (UTC, the bar time) → (ms, the text without the prefix); (None, text) without one. Pure."""
    import re
    from datetime import datetime, timezone
    t = str(txt or "")
    m = re.match(r"^(\d{2})-(\d{2}) (\d{2}):(\d{2})\s*", t)
    if not m:
        return None, t
    try:
        y = datetime.fromtimestamp(int(now_ms) / 1000, timezone.utc).year
        ms = int(datetime(y, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=timezone.utc).timestamp() * 1000)
        if ms > int(now_ms) + 2 * 86_400_000:   # a January read of a December bar
            ms = int(datetime(y - 1, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), tzinfo=timezone.utc).timestamp() * 1000)
        return ms, t[m.end():]
    except (ValueError, OverflowError):
        return None, t


def _same_episode(last, spike_ms, universe=None) -> bool:
    """a journal row belongs to the tracked episode: its spike ≥ the episode's spike (and the same universe when given)."""
    if not last:
        return False
    sp = _f(last.get('spike_ms'))
    if sp is None or spike_ms is None or sp < float(spike_ms) - 0.5:
        return False
    if universe is not None and str(last.get('universe') or "FRENZY") != universe:
        return False
    return True


def build(pair, now_ms, th, *, open_es=None, flag=None, exh_eps: Iterable = (), exh_rsi=None, exh_last=None, dist_last=None,
          fb_rec=None, fb_last=None, willy_pend=None) -> List[dict]:
    """→ the sorted lines for ONE pair (module doc). exh_eps = [(spike_ms, universe)] TRACKING · exh_rsi = (T_ms, rsi) the last 15m read ·
    exh_last / dist_last = the newest journal row as dict(at_ms, status, block, reason[, rsi1h]) · fb_rec = the in-memory FLATBASE record
    (status, confirm) · fb_last = dict(at_ms, status, reason) the ledger row · willy_pend = the WILLY pending entry (trig, armed, exp).
    Never raises (bad input → fewer lines)."""
    out: List[dict] = []
    try:
        act = active_sleeves(th)
        now_ms = int(now_ms)
        sl_open = SLEEVE_OF_STRATEGY.get(str(open_es or ""))
        if open_es:
            out.append(dict(sleeve=sl_open or "OPEN", icon=ICON.get(sl_open, "📌"), state="open", rank=0, at_ms=None,
                            reason=f"open ({open_es})", detail=f"an open {open_es} position holds the pair (one position per coin)"))
        eps = [(int(s), str(u or "FRENZY")) for s, u in (exh_eps or ())]
        fz_eps = [s for s, u in eps if u != "WIDE"]
        wd_eps = [s for s, u in eps if u == "WIDE"]
        # 🪫 EXH / 🌐 WIDE — one line, labelled by the episode the rule would judge (FRENZY first)
        if act["EXH"] and (fz_eps or (act["WIDE"] and wd_eps)) and sl_open != "EXH":
            lab = "EXH" if fz_eps else "WIDE"
            _sp = max(fz_eps) if fz_eps else max(wd_eps)
            ln = _decision_line(lab, exh_last, now_ms, EXH_REASON) if _same_episode(exh_last, _sp, "FRENZY" if fz_eps else "WIDE") else None
            if ln is None:
                _rt, r = (exh_rsi or (None, None))
                if _f(_rt) is None or now_ms - float(_rt) > RSI_MAX_AGE_MS:   # a stale 15m read is not shown
                    r = None
                ln = dict(sleeve=lab, icon=ICON[lab], state="tracking", rank=3, at_ms=None,
                          reason=(f"tracking · RSI15 {r:.0f} · waits cross ↓70" if _f(r) is not None else "tracking · waits RSI15 cross ↓70"),
                          detail="")
            out.append(ln)
        # 📉 DIST — FRENZY-universe episodes only
        if act["DIST"] and fz_eps and sl_open != "DIST":
            ln = (_decision_line("DIST", dist_last, now_ms, DIST_REASON, rsi_cut=_f(getattr(th, 'distribution_rsi1h_min', None)))
                  if _same_episode(dist_last, max(fz_eps)) else None)
            if ln is None:
                ln = dict(sleeve="DIST", icon=ICON["DIST"], state="waiting", rank=3, at_ms=None, reason="waits red ≥5× candle", detail="")
            out.append(ln)
        # 🩳 FB
        if act["FB"] and (fb_rec or fb_last) and sl_open != "FB":
            st = str((fb_rec or {}).get('status') or (fb_last or {}).get('status') or '')
            if fb_rec and fb_rec.get('confirm'):
                ln = dict(sleeve="FB", icon=ICON["FB"], state="armed", rank=3, at_ms=None, reason="confirm wait: 4 h close below the S4 close", detail="")
            elif st == "TRACKING":
                ln = dict(sleeve="FB", icon=ICON["FB"], state="tracking", rank=3, at_ms=None, reason="tracking · waits 4 h close below VWAP", detail="")
            elif st == "OPENED":
                at = _f((fb_last or {}).get('at_ms'))
                ln = dict(sleeve="FB", icon=ICON["FB"], state="signal", rank=1 if (at is not None and now_ms - at <= HOUR_MS) else 2, at_ms=at,
                          reason="opened", detail=str((fb_last or {}).get('reason') or ''))
            elif st in ("SHADOW", "MISSED", "EXPIRED"):
                at = _f((fb_last or {}).get('at_ms'))
                ln = dict(sleeve="FB", icon=ICON["FB"], state="blocked", rank=1 if (at is not None and now_ms - at <= HOUR_MS) else 2, at_ms=at,
                          reason=FB_STATE_TEXT[st], detail=str((fb_last or {}).get('reason') or ''))
            else:
                at = _f((fb_last or {}).get('at_ms'))
                why = str((fb_last or {}).get('reason') or st or '').split(' — ')[0][:40]
                ln = dict(sleeve="FB", icon=ICON["FB"], state="blocked", rank=1 if (at is not None and now_ms - at <= HOUR_MS) else 2,
                          at_ms=at, reason=(f"{st.lower()} · {why}" if why and why != st else st.lower()), detail=str((fb_last or {}).get('reason') or ''))
            out.append(ln)
        # 🎲 HT — the WILLY machinery's pending entry, else its last HT outcome text
        if act["HT"] and sl_open not in ("HT", "WILLY"):
            if willy_pend:
                exp = _f(willy_pend.get('exp_ms') if 'exp_ms' in willy_pend else willy_pend.get('exp'))
                mins = max(0, int((exp - now_ms) // 60_000)) if exp is not None else None
                out.append(dict(sleeve="HT", icon=ICON["HT"], state="armed", rank=3, at_ms=None,
                                reason=(f"armed {willy_pend.get('trig') or ''}, waits red ≤ {mins} min" if mins is not None else "armed, waits red").replace("armed , ", "armed, "),
                                detail=""))
            elif flag and (flag.get('willy_ht_last') or flag.get('willy_last')):
                cands = [utc_prefix_ms(flag.get(k), now_ms) for k in ('willy_ht_last', 'willy_last') if flag.get(k)]
                # (587 re-review 11) a "plain WILLY entries OFF" refusal is a WILLY outcome by design, never an HT block → not shown
                cands = [c for c in cands if PLAIN_OFF_TXT not in c[1]]
                cands = [c for c in cands if not PLAIN_OUTCOME_RE.search(c[1])]   # (646) a plain WILLY's own outcome is not an HT line
                if cands:
                    at, txt = max(cands, key=lambda c: (c[0] or 0))   # the NEWER of the two outcome texts (UTC prefix stripped: reason AND hover)
                    tail = txt.split("refused: ", 1)[1] if "refused: " in txt else txt.split(" · ")[-1]
                    bad = "refused" in txt
                    out.append(dict(sleeve="HT", icon=ICON["HT"], state="blocked" if bad else "waiting",
                                    rank=(1 if (bad and at is not None and now_ms - at <= HOUR_MS) else (2 if bad else 3)),
                                    at_ms=at, reason=tail.strip()[:56], detail=txt.strip()))
        # 🔥 FRENZY_LONG — only while it is enabled
        if act["LONG"] and flag and sl_open != "LONG":
            out.append(dict(sleeve="LONG", icon=ICON["LONG"], state="waiting", rank=3, at_ms=None, reason=str(flag.get('text') or '')[:48],
                            detail=str(flag.get('text') or '')))
    except Exception as e:   # (re-review 11) the lines built so far are kept; the failure is logged (debug), never raised
        logging.getLogger(__name__).debug(f"[SLEEVE_STATUS] {pair}: build stopped early ({str(e)[:120]})")
    out.sort(key=lambda d: (d.get("rank", 9), -(d.get("at_ms") or 0)))
    return out
