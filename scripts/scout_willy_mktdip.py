#!/usr/bin/env python3
"""👁 Scout — WILLY_MKTDIP observe line (DECISION_LOG 305, operator 2026-10-16: "register willy"). OBSERVE ONLY — never changes config.

Origin: reports/WILLY_CONT_2026-10-16.md (Q1, the one survivor of 28,646 tests, p_fw 0.010, walk-forward 0 confirmations, corner effect).
HYPOTHESIS: when WILLY's red entry candle falls on a MARKET-WIDE red 5m bar while the coin's own short trend is still strong, the dip is the
market's, not the pump's — so a trade that touches +1.25 keeps going to +2 (backtest: 45 of 46 touches reached +2 by minute 30 vs 76 %).
If ever armed it would change only those trades' TP from +1.25 to +2 (late TP kept). Master today's rules: +7.3 → +9.5 % / +$484 → +$631.

FROZEN DEFINITION (never re-fit)
  cohort   CLOSED FRENZY_WILLY fills opened ≥ COHORT_START (orders exports, paper + live via export_glob), dedup (opened_at, pair, direction).
  zone     br_up5 ≤ 0.2979 ∧ entry_ema_gap_8_13 > 1.393 (the study's exact cut), where br_up5 = share of the TOP-50 universe (reports/backtest_cache/
           universe_daily_top50.csv, last date ≤ the fill day − 1) whose LAST CLOSED 5m bar at entry closed above its open
           (bar opening at floor(entry, 5m) − 5m), EXCLUDING BTC / USDC / PAXG / XAUT / FDUSD as the study's grid does. ≥ 30 coins with data,
           else UNSCORED.
  touch    first 1m bar from the entry minute whose HIGH ≥ entry × 1.01341 (+1.25 net, the study's taker/taker basis) ;
  cont     a 1m HIGH ≥ entry × 1.02091 (+2 net) at or after the touch bar, before entry + 30 min and before any LOW ≤ the stop. Stop: a LOW ≤ entry × 0.96088 (−4 net) before the
           touch ends the walk (no touch); the entry-minute bar is included whole and a same-bar stop + touch counts as STOP. A fill whose
           30-min window is not complete yet stays PENDING. (Approximation of the study's walker on public 1m klines — declared, kept frozen.)
  universe the top-50 file's date used and its lag are printed; a lag > 2 days is flagged (refresh the universe file before the read).
GATE (observe only; even a CANDIDATE needs a reviewed study before any order change)
  CANDIDATE at ≥ 15 in-zone touches on ≥ 8 UTC days with a +2 rate ≥ 85 % AND the +2 rate of out-of-zone touches < 80 % (no
  out-of-zone touches yet → keep observing).
  RETIRE if the in-zone +2 rate < 78 % at N ≥ 10.
BINANCE  public fapi klines only; cache reports/backtest_cache/scout_willy_mktdip/; ≤ MAX_FILLS fills scored per run (≤ 51 requests each),
  paced, stop once X-MBX-USED-WEIGHT-1M ≥ 700, shared scout cooldown honoured / armed on 418-429.
Usage  venv/bin/python scripts/scout_willy_mktdip.py      (batch review; never calls the bot API)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG                                     # noqa: E402
import scout_b1h_negflank as NF                              # noqa: E402  (shared cooldown helpers)
import willy_exit_era as WEra                                # noqa: E402  🎲² (DECISION_LOG 356) exit-era split / header

COHORT_START = "2026-10-17T00:00:00"
BR_MAX, GAP_MIN, MIN_COINS = 0.2979, 1.393, 30           # the study's exact cell (study_willy_cont_followup.py)
TOUCH, CONT, STOP = 1.01341, 1.02091, 0.96088           # the study's net lines (taker fees both sides) in gross terms
EXCL = {"BTCUSDT", "USDCUSDT", "PAXGUSDT", "XAUTUSDT", "FDUSDUSDT"}   # dropped by the study's breadth grid (load_universe)
GATE_N, GATE_DAYS, CAND_RATE, REST_MAX, RETIRE_N, RETIRE_RATE = 15, 8, 0.85, 0.80, 10, 0.78
MAX_FILLS, WEIGHT_STOP, PACE_S = 6, 700, 0.35
M1, M5, DAY = 60_000, 300_000, 86_400_000
_ROOT = os.path.dirname(_D)
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_willy_mktdip")
UNIV = [os.path.join(_ROOT, "reports", p, "universe_daily_top50.csv") for p in ("backtest_cache_2025", "backtest_cache")]


def _ms(s):
    return int(pd.Timestamp(str(s)[:19]).value // 1_000_000)


def fills():
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in (
                "opened_at", "pair", "direction", "status", "entry_strategy", "entry_price", "pnl_percentage", "entry_ema_gap_8_13", "close_reason")
                + tuple(WEra.V2_COLS)))                         # DL 356: the V2 stamps ride the reader
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a = a[(a.entry_strategy.astype(str) == "FRENZY_WILLY") & (a.opened_at.astype(str) >= COHORT_START)]
    a = a.assign(_c=(a.status == "CLOSED").astype(int)).sort_values("_c").assign(_k=a.opened_at.astype(str).str[:19]).drop_duplicates(["_k", "pair", "direction"], keep="last")
    return a[a.status == "CLOSED"].reset_index(drop=True)


_UNIV_SEEN = {}


def top50(day_ms):
    fs = [pd.read_csv(f) for f in UNIV if os.path.exists(f)]
    if not fs:
        return []
    U = pd.concat(fs)
    d = U[U.date_ms <= day_ms]
    if not len(d):
        return []
    last = int(d.date_ms.max())
    _UNIV_SEEN[last] = max(_UNIV_SEEN.get(last, 0), (day_ms - last) // DAY)
    return sorted(d[d.date_ms == last].pair.unique())


class _Bud:
    used, stop = 0, False


def kl(sym, interval, start, limit, bud):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{sym}_{interval}_{start}_{limit}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    now = int(time.time() * 1000)
    if bud.stop or NF._cooldown_until() > now or bud.used >= WEIGHT_STOP:
        bud.stop = True
        return None
    time.sleep(PACE_S)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&startTime={start}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
            bud.stop = True
        return None if e.code in (418, 429) else []
    except Exception:
        return []                                              # one unreadable symbol / timeout = missing data, not a budget stop
    out = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows]
    if out and out[-1][0] + (M1 if interval == "1m" else M5) <= now:   # cache only complete windows
        with open(p, "w") as f:
            json.dump(out, f)
    return out


def breadth(te, bud):
    bar = (te // M5) * M5 - M5
    ups, n = 0, 0
    for s in top50((te // DAY) * DAY - DAY):
        if not s.isascii() or s in EXCL:
            continue                                           # non-ASCII listings are skipped (counted as no data)
        k = kl(s, "5m", bar, 1, bud)
        if k is None:
            return None
        if k and k[0][0] == bar:
            n += 1
            ups += k[0][4] > k[0][1]
    return ups / n if n >= MIN_COINS else np.nan


def path(pair, te, e, bud):
    if int(time.time() * 1000) < te + 31 * M1:
        return None                                            # window not complete yet → pending
    k = kl(pair, "1m", (te // M1) * M1, 31, bud)
    if k is None:
        return None
    if len(k) < 31 and not any(l <= e * STOP for _, _, _, l, _ in k):
        return None                                            # partial window without a stop → pending
    end = te + 30 * M1
    for i, (t, o, h, l, c) in enumerate(k):
        if t >= end:
            break
        if l <= e * STOP:
            return (False, False)
        if h >= e * TOUCH:
            for tt, _, hh, ll, _ in k[i:]:
                if tt >= end:
                    break
                if hh >= e * CONT:
                    return (True, True)
                if ll <= e * STOP:
                    return (True, False)                       # stopped before +2 (the study's t_2 requires +2 before any stop)
            return (True, False)
    return (False, False)


def main():
    a = fills()
    import scout_gatekit as _GK                                 # 🧰 (DECISION_LOG 331) operator-closed fills: not the bot's exit → out, listed
    a, opx = _GK.split_operator_closed(a) if len(a) else (a, a)
    a = a.reset_index(drop=True)
    print(f"WILLY_MKTDIP (DECISION_LOG 305, observe) · cohort FRENZY_WILLY closed fills opened ≥ {COHORT_START} UTC → {len(a)}"
          f" · {_GK.operator_closed_line(opx)}")
    print(WEra.era_line(a, "WILLY_MKTDIP", note=("touch / +2 are read on the 1m path with the frozen lines (never the booked exit), so V2 "
                                                 "fills stay in the gate; 'avg booked' counts old-exit fills only") if WEra.counts(a)[2] else None))   # DL 356 (c)
    if not len(a):
        return
    bud, rows, pend = _Bud(), [], 0
    for r in a.to_dict("records"):
        if len(rows) + pend >= MAX_FILLS and not os.path.exists(os.path.join(CACHE, f"{r['pair']}_1m_{(_ms(r['opened_at']) // M1) * M1}_31.json")):
            pend += 1
            continue
        te, e = _ms(r["opened_at"]), float(r["entry_price"])
        br, pt = breadth(te, bud), path(r["pair"], te, e, bud)
        if br is None or pt is None:
            pend += 1
            continue
        gap = pd.to_numeric(pd.Series([r.get("entry_ema_gap_8_13")]), errors="coerce").iloc[0]
        zone = "UNSCORED" if (np.isnan(br) or np.isnan(gap)) else ("ZONE" if (br <= BR_MAX and gap > GAP_MIN) else "REST")
        rows.append(dict(day=str(r["opened_at"])[:10], zone=zone, touch=pt[0], cont=pt[1], pct=pd.to_numeric(pd.Series([r.get("pnl_percentage")]), errors="coerce").iloc[0],
                         v2c=WEra.is_v2_changed(r)))                # DL 356: a V2 runner / floor fill booked another exit
    d = pd.DataFrame(rows)
    print(f"  scored {len(d)} · pending {pend} (budget / cooldown / window not complete — re-run later)")
    for last, lag in sorted(_UNIV_SEEN.items()):
        print(f"  top-50 universe used: {pd.Timestamp(last, unit='ms').date()} (max lag {lag} d){'  ⚠ STALE — refresh universe_daily_top50.csv' if lag > 2 else ''}")
    if not len(d):
        return
    for z, g in d.groupby("zone"):
        t = g[g.touch]
        gb = g[~g.v2c]                                          # DL 356: booked % of old-exit fills only (V2 runner / floor out, counted)
        print(f"  {z:9s} fills {len(g):3d} · touched +1.25 {len(t):3d} on {t.day.nunique()} days · reached +2 by 30 min "
              f"{int(t.cont.sum())} ({(t.cont.mean() if len(t) else float('nan')):.0%}) · avg booked {gb.pct.mean():+.2f} %"
              + (f" (old-exit fills; {int(g.v2c.sum())} V2 runner / floor out)" if g.v2c.any() else ""))
    zt, rt = d[(d.zone == "ZONE") & d.touch], d[(d.zone == "REST") & d.touch]
    zr, rr = (zt.cont.mean() if len(zt) else np.nan), (rt.cont.mean() if len(rt) else np.nan)
    if len(zt) >= RETIRE_N and zr < RETIRE_RATE:
        v = "RETIRE (in-zone +2 rate below 78 %)"
    elif len(zt) >= GATE_N and zt.day.nunique() >= GATE_DAYS:
        v = "CANDIDATE → reviewed study (operator decides)" if (zr >= CAND_RATE and rr < REST_MAX) else "keep observing (gate not met)"
    else:
        v = f"collecting ({len(zt)}/{GATE_N} in-zone touches on {zt.day.nunique()}/{GATE_DAYS} days)"
    print(f"  verdict: {v}")


if __name__ == "__main__":
    main()
