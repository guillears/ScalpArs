#!/usr/bin/env python3
"""🌡 Scout — HEAT-BLOCK × BTC REGIME observation (pre-registered 2026-10-05, operator; OBSERVE only — never changes config).

Question (operator: "there must be a macro BTC variable that says which rule makes sense in each period"): the full-year filter × regime
matrix (reports/FILTER_REGIME_MATRIX_2026-10-05.md, 40 gates × 21 BTC splits, 500-round shuffled-day null) found NO robust regime-conditional
filter. Its two closest heat-block near-misses are tracked here on the forward heat blocks, thresholds FROZEN now:

  VOLR   BTC daily quote volume of the last CLOSED day ÷ the mean of the 30 days before it.
         LOW ≤ 0.817 (year: the block removed WINNERS, +0.323 %/signal · 47 signals · 14 days · positive in both halves)
         HIGH > 1.13 (year: the block removed LOSERS, −0.108 % · 71 signals · 21 days) · gap CI [−0.738, +0.031] just missed.
  CHOP   BTC 72h efficiency (the engine's bull-run monitor 'eff': |Δ| ÷ Σ|step| over the last 864 CLOSED 5m bars) ≤ 0.007
         (year: chop +0.315 % vs non-chop −0.126 %, only 6 chop days).

SOURCE  the heat-blocked LONG signals the revert-gate tracker already re-prices with the live momentum-LONG exit (reports/SCOUT_REVERT_GATES.json):
        HEAT_ORIG = the live original rule's blocks (after the Oct-5 revert, DECISION_LOG 208) — the read; HEAT = the re-scope era's frozen
        blocks (Sep-25 → Oct-5) — context only. A positive re-priced signal = the block removed a winner.
UNIT    the DAY (both variables are market-wide and VOLR is constant within a day) — window-units rule; a day's value = the mean of its signals.
REVIEW  (≥ 8 days on EACH side for both splits) VOLR: ≥ 8 LOW days ∧ ≥ 8 HIGH days. Candidate only if mean(LOW day means) > 0 ∧ mean(HIGH day means) < 0 ∧ no single day ≥ 50 % of
        the LOW side's positive gain. CHOP: ≥ 8 chop ∧ ≥ 8 trend days; candidate only if mean(chop day means) > 0 ∧ mean(non-chop day means) < 0 ∧ the
        same concentration check. Never re-fit 0.817 / 1.13 / 0.007; a candidate still faces the locked promotion gates.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_JSON = os.path.join(ROOT, "reports", "SCOUT_REVERT_GATES.json")
MIN, H, DAY = 60_000, 3_600_000, 86_400_000
VOLR_LOW, VOLR_HIGH, EFF_CHOP, EFF_W = 0.817, 1.13, 0.007, 864
REVIEW_DAYS = 8
YEAR_REF = ("yr5 heat blocks: VOLR ≤ 0.817 +0.323 %/signal (47 · 14 days) · VOLR > 1.13 −0.108 % (71 · 21 days) · "
            "chop (eff ≤ 0.007) +0.315 % (6 days) vs non-chop −0.126 %")


# ─────────────────────────── pure readings (selftest) ───────────────────────────
def volr_at(d1, t_ms):
    """d1 = DataFrame(open_time, qv) of daily bars sorted; → qv(last day CLOSED by t) ÷ mean(qv of the 30 days before it), or None."""
    ot = d1.open_time.values
    i = int(np.searchsorted(ot, t_ms - DAY, side="right")) - 1   # last bar with open_time + DAY ≤ t
    if i < 30 or int(ot[i]) != (t_ms // DAY) * DAY - DAY:   # the day that closed just before t must be there — stale data is never read (review)
        return None
    prev = d1.qv.values[i - 30:i]
    m = float(np.mean(prev))
    return float(d1.qv.values[i] / m) if m > 0 else None


def eff_at(btc5, t_ms):
    """btc5 = ndarray [open_time, o, h, l, c] sorted; the engine's 72h efficiency over the last 864 bars CLOSED by t, or None."""
    if btc5 is None or not len(btc5):
        return None
    j = int(np.searchsorted(btc5[:, 0], t_ms - 5 * MIN, side="right"))   # bars [0, j) are closed by t
    if j < EFF_W or (t_ms // (5 * MIN)) * 5 * MIN - 5 * MIN - int(btc5[j - 1, 0]) > 5 * MIN:   # last closed bar ≤ 1 bar late (REST lag), else stale → None (review)
        return None
    w = btc5[j - EFF_W:j, 4]
    if (btc5[j - 1, 0] - btc5[j - EFF_W, 0]) != (EFF_W - 1) * 5 * MIN:   # a gap in the bars → no reading (never a guess)
        return None
    d = float(np.abs(np.diff(w)).sum())
    return float(abs(w[-1] - w[0]) / d) if d > 0 else 0.0


def volr_bucket(v):
    if v is None or not np.isfinite(v):
        return "unread"
    return "LOW" if v <= VOLR_LOW else ("HIGH" if v > VOLR_HIGH else "MID")


def chop_bucket(e):
    if e is None or not np.isfinite(e):
        return "unread"
    return "CHOP" if e <= EFF_CHOP else "TREND"


def day_means(rows, key):
    """rows: dicts with day, sim, <key> → {bucket: {day: mean}}; a day lands in the bucket of its signals (VOLR is per day; CHOP can
    change inside a day → the day is split by bucket)."""
    out = {}
    for r in rows:
        out.setdefault(r[key], {}).setdefault(r["day"], []).append(r["sim"])
    return {b: {d: float(np.mean(v)) for d, v in dd.items()} for b, dd in out.items()}


def decide(pos_side, neg_side, n_days=REVIEW_DAYS, need_neg_days=True):
    """→ ('collecting' | 'candidate' | 'no'), detail. pos_side / neg_side = {day: mean}."""
    np_, nn = len(pos_side), len(neg_side)
    if np_ < n_days or (need_neg_days and nn < n_days) or nn == 0:
        return "collecting", f"{np_}/{n_days} vs {nn}/{n_days if need_neg_days else 1} days"
    mp, mn = float(np.mean(list(pos_side.values()))), float(np.mean(list(neg_side.values())))
    gains = [v for v in pos_side.values() if v > 0]
    top = (max(gains) / sum(gains)) if gains else 1.0
    ok = mp > 0 and mn < 0 and top < 0.5
    return ("candidate" if ok else "no"), f"means {mp:+.3f} vs {mn:+.3f} · top day {top * 100:.0f} % of the gain"


# ─────────────────────────── data ───────────────────────────
def _get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


def btc_daily(now_ms, days=200):
    q = urllib.parse.urlencode(dict(symbol="BTCUSDT", interval="1d", startTime=now_ms - days * DAY, limit=days + 5))
    r = _get_json(f"https://fapi.binance.com/fapi/v1/klines?{q}") or []
    d = pd.DataFrame([(int(x[0]), float(x[7])) for x in r if int(x[0]) + DAY <= now_ms], columns=["open_time", "qv"])
    return d.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def _items(st, code):
    """priced items + the count still pending a price."""
    out, pend = [], 0
    for v in ((st.get("gates") or {}).get(code) or {}).get("items", {}).values():
        if v.get("sim1") is None:
            pend += 1
            continue
        out.append(dict(t=int(v["t"]), pair=str(v.get("pair")), sim=float(v["sim1"]), final=bool(v.get("final"))))
    return out, pend


def run(now_ms=None, btc5=None):
    """→ markdown lines. btc5 = a BTC 5m [open_time, o, h, l, c] array; None → scout_revert_gates.btc5m_array (cache + REST). Never raises past the caller."""
    now_ms = int(now_ms or time.time() * 1000)
    st = {}
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as fh:
            st = json.load(fh)
    if btc5 is None:
        import scout_revert_gates as _rg
        btc5 = _rg.btc5m_array(now_ms - 12 * DAY)
    d1 = btc_daily(now_ms)
    L = ["## 🌡 Heat block × BTC regime (pre-registered, OBSERVE only — never changes config)", "",
         f"Frozen splits: BTC daily volume ratio (last closed day ÷ prior 30-day mean) LOW ≤ {VOLR_LOW} / HIGH > {VOLR_HIGH} · BTC 72h efficiency "
         f"CHOP ≤ {EFF_CHOP}. Unit = the DAY. A positive re-priced block = the block removed a winner. Year reference: {YEAR_REF}. "
         f"Review at ≥ {REVIEW_DAYS} days per side: candidate only if the 'removes winners' side > 0 ∧ the other < 0 ∧ no day ≥ 50 % of the gain.", ""]
    now_v, now_e = volr_at(d1, now_ms), eff_at(btc5, now_ms)
    L.append(f"**Now:** BTC volume ratio {now_v:.2f} ({volr_bucket(now_v)})" if now_v is not None else "**Now:** volume ratio unread")
    L[-1] += f" · 72h efficiency {now_e:.3f} ({chop_bucket(now_e)})" if now_e is not None else " · efficiency unread"
    L.append("")
    L += ["| Heat blocks | Signals | VOLR LOW: days · mean of day means | MID | HIGH | CHOP: days · mean | TREND | unread |", "|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for code, label in (("HEAT_ORIG", "live original rule (from Oct-5) — the read"), ("HEAT", "re-scope era Sep-25→Oct-5 (frozen, context)")):
        rows = []
        its, pend = _items(st, code)
        for x in its:
            v, e = volr_at(d1, x["t"]), eff_at(btc5, x["t"])
            rows.append(dict(x, day=pd.to_datetime(x["t"], unit="ms").strftime("%Y-%m-%d"), vb=volr_bucket(v), cb=chop_bucket(e)))
        V, C = day_means([r for r in rows if r["vb"] != "unread"], "vb"), day_means([r for r in rows if r["cb"] != "unread"], "cb")
        cell = lambda m, b: (f"{len(m.get(b, {}))} · {np.mean(list(m[b].values())):+.3f}" if m.get(b) else "0")
        unread = sum(1 for r in rows if r["vb"] == "unread" or r["cb"] == "unread")
        L.append(f"| {label} | {len(rows)}{' (' + str(sum(1 for r in rows if not r['final'])) + ' on 1m)' if any(not r['final'] for r in rows) else ''}{f' (+{pend} pending)' if pend else ''} | "
                 f"{cell(V, 'LOW')} | {cell(V, 'MID')} | {cell(V, 'HIGH')} | {cell(C, 'CHOP')} | {cell(C, 'TREND')} | {unread} |")
        if code == "HEAT_ORIG":
            sv, dv = decide(V.get("LOW", {}), V.get("HIGH", {}))
            sc, dc = decide(C.get("CHOP", {}), C.get("TREND", {}))
            verdicts = [f"VOLR: {'📋 CANDIDATE — review' if sv == 'candidate' else ('❌ pattern not confirmed' if sv == 'no' else '⏳ collecting')} ({dv})",
                        f"CHOP: {'📋 CANDIDATE — review' if sc == 'candidate' else ('❌ pattern not confirmed' if sc == 'no' else '⏳ collecting')} ({dc})"]
    L += ["", "Live-rule verdicts: " + " · ".join(verdicts), ""]
    return L


# ─────────────────────────── self-test ───────────────────────────
def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    d1 = pd.DataFrame(dict(open_time=[i * DAY for i in range(40)], qv=[100.0] * 39 + [50.0]))
    chk(volr_at(d1, 40 * DAY) == 0.5, "VOLR = last closed day / mean of the 30 before")
    chk(volr_at(d1, 40 * DAY - 1) == 1.0, "the forming day is never read")
    chk(volr_at(d1, 20 * DAY) is None, "needs 30 prior days")
    chk(volr_at(d1, 45 * DAY) is None, "stale daily data (last closed day missing) → None")
    t = np.arange(900) * 5 * MIN
    up = np.column_stack([t, t * 0, t * 0, t * 0, 100 + np.arange(900.0)])
    chk(abs(eff_at(up, 900 * 5 * MIN) - 1.0) < 1e-12, "a straight line = efficiency 1")
    zz = up.copy(); zz[:, 4] = 100 + (np.arange(900) % 2)
    chk(eff_at(zz, 900 * 5 * MIN) < 0.002, "a zig-zag ≈ 0")
    chk(eff_at(up, 500 * 5 * MIN) is None, "needs 864 closed bars")
    chk(eff_at(up, 960 * 5 * MIN) is None, "stale 5m data (last closed bar > 1 bar late) → None")
    chk(eff_at(up, 901 * 5 * MIN) is not None, "one bar of REST lag tolerated")
    jmp = up.copy(); jmp[899, 4] = 0.0
    chk(eff_at(jmp, 900 * 5 * MIN) != eff_at(jmp, 900 * 5 * MIN - 1), "bar 899 is read only once it has closed")
    gap = np.delete(up, 600, axis=0)
    chk(eff_at(gap, 899 * 5 * MIN + 5 * MIN) is None, "a gap → unread")
    chk(volr_bucket(0.817) == "LOW" and volr_bucket(1.13) == "MID" and volr_bucket(1.131) == "HIGH" and volr_bucket(None) == "unread", "VOLR buckets")
    chk(chop_bucket(0.007) == "CHOP" and chop_bucket(0.0071) == "TREND", "chop bucket")
    pos = {f"d{i}": 0.2 for i in range(8)}; neg = {f"e{i}": -0.1 for i in range(8)}
    chk(decide(pos, neg)[0] == "candidate", "both signs + spread gain → candidate")
    chk(decide(dict(list(pos.items())[:7]), neg)[0] == "collecting", "7 days → collecting")
    conc = dict(pos); conc["d0"] = 5.0
    chk(decide(conc, neg)[0] == "no", "one day ≥ 50 % of the gain → no")
    chk(decide(pos, {k: 0.1 for k in neg})[0] == "no", "other side positive → no")
    dm = day_means([dict(day="a", sim=1.0, vb="LOW"), dict(day="a", sim=0.0, vb="LOW"), dict(day="b", sim=-1.0, vb="HIGH")], "vb")
    chk(dm == {"LOW": {"a": 0.5}, "HIGH": {"b": -1.0}}, "day means")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        print("\n".join(run()))
