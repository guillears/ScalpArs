#!/usr/bin/env python3
"""⏱ FRENZY MAX-DURATION READ — one row per FRENZY sleeve (DECISION_LOG 496; B25 plan ②z3 = DECISION_LOG 438 + the operator's re-ask
2026-11-07: "re-check the max duration per FRENZY sleeve in the B25 review, and a recommendation"). Read-only: no engine / config change,
no gate, no alert — the data the review's per-sleeve recommendation is built on.

Per sleeve (FRENZY_LITE · FRENZY_LITE_RETRY · FRENZY_LONG · FRENZY_WILLY · FRENZY_WILLY_HT, each apart — never pooled, operator 10-30):
  ① HOLD PROFILE — the live cap (trading_config.json, re-read every run), N · WR · avg %, hold median / p75 / p90 / longest, winners' vs
     losers' median hold, fills that reached the cap (MAX_HOLD_TIME) with their WR / avg, how many were still open at each X.
  ② SHORTER CAP AT X (X = 15 / 30 / 60 / 120 / 240 / 360 / 480 min, only X < the sleeve's cap): the cap fires at open + X + 8 s (live
     latency ruler); a fill still open then closes at market at the close of the last 1m bar that CLOSED by that moment (never a price from
     after it — so never past the fill's real close either), taker exit fee + 0.10 slip (operator's ≥ 0.10 ruler); a fill that closed
     before the cap fired keeps its booked % (Δ 0). No replica stop, no replica TP: the path before X is the real one, so the
     live-stopped-cohort rule (CLAUDE.md) holds by construction — only fills still open at X change, winners and losers alike. Per X:
     N changed · WR · avg % · Σ pts · Δ Σ vs booked · day-block bootstrap P(Δ > 0) · 🧮 leave-one-out · the 30–50 % in-sample haircut
     (favourable Δ only) · the CURRENT batch's daily compound return before → after on the real account path (review_math.dcr_account,
     each fill's actual notional, B25 start $2,900 + the 10-26 13:50 top-up).
  ③ LONG HOLDS — every fill held ≥ 4 h (operator 11-07: TAUSDT FRENZY_LITE #240 first), its exit and booked % vs a market close at 4 h.
  ④ PARITY — (a) the fee ruler re-books every market-closed fill from its own exit price, slip 0 · (b) the price ruler: each MAX_HOLD_TIME
     fill's own exit price vs this ruler's price at its real close. Either off → ② is not read.

EXIT ERAS: LITE / RETRY fills opened before the LITE C revert (DECISION_LOG 420 deploy + 10 min) booked the old +2 TP out of zone — their
rows are split pre / post, never mixed (ENTRY × EXIT rule ④).

Blocks: CURRENT BATCH (the newest orders export, fills opened ≥ sweep_separators.BATCH_START) and MASTER (reports/MASTER_POOL_stacked.csv,
stack_keep rows, WR / avg on stack_pct = today's stack). ⚠ The master shows the hold PROFILE only: most FRENZY master rows are re-priced on
today's stack, so their traded hold is not the hold under today's exit, and pricing only the unchanged rows would be one-sided (they are
mostly stops — the TP winners a cap would cut are the re-priced ones). A master ② needs a walk under today's TP / stop (review step).
Probes are out (exact sleeve names only; is_probe rows dropped).

1m bars: the shared caches (scout_willy_timecap.load_1m); --fetch fills missing windows through path_labels' budgeted fetcher (public
fapi klines, ≤ 120 requests / 60 s per run, stops at used weight 900, on a 418 / 429 cooldown or the first network error).

Usage: venv/bin/python -I scripts/scout_frenzy_duration.py [orders.csv] [--fetch] [--since 2026-10-23T18:30:00]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import review_math as RM  # noqa: E402

CONFIG = os.path.join(ROOT, "trading_config.json")
MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
SLEEVES = ("FRENZY_LITE", "FRENZY_LITE_RETRY", "FRENZY_LONG", "FRENZY_WILLY", "FRENZY_WILLY_HT")
LITE_FAMILY = ("FRENZY_LITE", "FRENZY_LITE_RETRY")
LITE_ERA_DL = 420                                                   # LITE C revert: +2 out of zone → +3 for every fill
CAP_KEY = {"FRENZY_LITE": "frenzy_max_hold_minutes", "FRENZY_LITE_RETRY": "frenzy_max_hold_minutes", "FRENZY_LONG": "frenzy_max_hold_minutes",
           "FRENZY_WILLY": "frenzy_willy_max_hold_minutes", "FRENZY_WILLY_HT": "frenzy_willy_max_hold_minutes"}
XS = (15, 30, 60, 120, 240, 360, 480)
LONG_HOLD_MIN = 240
TAKER, MAKER, SLIP = 0.045, 0.018, 0.10
LAT_MS = 8_000
MIN = 60_000
B25_START, B25_FLOWS = 2900.0, [("2026-10-26T13:50:00", 7100.0)]   # DECISION_LOG 375: paper $2,900 → $10,000 top-up, no reset
MARKET_EXITS = ("STOP_LOSS", "MAX_HOLD_TIME", "WILLY_FLOOR", "BACKSTOP")


# ───────────────────────────── pure math ─────────────────────────────
def caps(cfg_path=CONFIG):
    """→ {sleeve: cap minutes}, read as the engine reads them (services/trading_engine.py max-hold check): FRENZY = min(global
    max_holding_time_minutes, frenzy_max_hold_minutes) — 0 on either side → the other; WILLY / HT round(frenzy_willy_max_hold_minutes),
    < 1 → 120, never the global."""
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    th = cfg.get("thresholds", {})
    glob_mh = cfg.get("investment", {}).get("max_holding_time_minutes", 1200)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    out = {}
    for s, k in CAP_KEY.items():
        v = num(th.get(k))
        if k == "frenzy_willy_max_hold_minutes":
            out[s] = int(round(v)) if v >= 1 else 120
        else:
            g, fz = int(num(glob_mh)), int(v)
            out[s] = (min(g, fz) if g > 0 else fz) if fz > 0 else g
    return out


def to_ms(v):
    """a timestamp string (naive UTC or tz-aware) → epoch ms, None when unparseable."""
    try:
        t = pd.Timestamp(str(v).replace(" ", "T"))
    except Exception:
        return None
    if pd.isna(t):
        return None
    t = t.tz_convert(None) if t.tzinfo else t
    return int(t.value // 1_000_000)


def hold_minutes(opened, closed):
    a, b = to_ms(opened), to_ms(closed)
    return None if a is None or b is None else (b - a) / MIN


def entry_fee(order_type, fee_usd=None, notional=None):
    """entry fee % of notional: the fill's own entry_fee ÷ notional when both are known, else by order type (pure MAKER → maker rate,
    anything with a taker leg → taker)."""
    try:
        f, n = float(fee_usd), float(notional)
        if np.isfinite(f) and np.isfinite(n) and n > 0 and f >= 0:
            return f / n * 100.0
    except (TypeError, ValueError):
        pass
    return MAKER if str(order_type or "").strip().upper() == "MAKER" else TAKER


def net_pct(entry, px, direction, fee_in, slip=SLIP):
    """net % on the entry notional of a MARKET exit at px: direction-aware gross − entry fee − taker exit fee (on the exit notional) − slip."""
    sgn = -1.0 if str(direction).upper() == "SHORT" else 1.0
    r = float(px) / float(entry)
    return sgn * (r - 1.0) * 100.0 - fee_in - TAKER * r - slip


def cap_fires_ms(te, x):
    return int(te) + int(x) * MIN + LAT_MS


def open_at_cap(hold_min, x):
    """True when the fill was still open when a cap at x would have fired (x + the 8 s latency) — the only fills a shorter cap changes."""
    return hold_min is not None and not pd.isna(hold_min) and hold_min * MIN > int(x) * MIN + LAT_MS


def price_at(bars, t_ms):
    """the close of the last 1m bar that CLOSED by t_ms (open_time = floor(t) − 1 min) → float or None (bar missing). Never a price
    printed after t_ms — so for a fill still open at t, never one after its real close either."""
    if bars is None or not len(bars):
        return None
    m = (int(t_ms) // MIN) * MIN - MIN
    hit = bars.c.values[bars.open_time.values == m]
    return float(hit[-1]) if len(hit) else None


def cap_at(booked, hold_min, x, px_at_x, entry, direction, fee_in):
    """booked % under a cap at x minutes: unchanged when the fill closed before the cap fired; else the market close (None = unpriced)."""
    if hold_min is None or pd.isna(hold_min):
        return None
    if not open_at_cap(hold_min, x):
        return float(booked)
    if px_at_x is None or pd.isna(px_at_x):
        return None
    return net_pct(entry, px_at_x, direction, fee_in)


def boot_p(delta, days, n=4000, seed=7):
    """day-block bootstrap P(mean Δ > 0) — resample whole close days, never trades (one market day = one observation)."""
    d = pd.Series(list(delta), dtype=float)
    g = d.groupby(list(days)).apply(list).tolist()
    if len(g) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        pick = [g[i] for i in rng.integers(0, len(g), len(g))]
        hits += float(np.mean([x for b in pick for x in b])) > 0
    return hits / n


def haircut_text(mean_d):
    """the 30–50 % in-sample haircut applies to a FAVOURABLE Δ only (shrinking a harm would flatter it)."""
    if not np.isfinite(mean_d) or mean_d <= 0:
        return "n/a (Δ ≤ 0)"
    return f"{mean_d * 0.5:+.3f} … {mean_d * 0.7:+.3f}"


def profile(f, cap):
    """hold profile of one sleeve's fills (cols: pct, hold, close_reason) → dict."""
    h = pd.to_numeric(f.hold, errors="coerce")
    p = pd.to_numeric(f.pct, errors="coerce")
    w, l = h[p > 0], h[p <= 0]
    capd = f[f.close_reason.astype(str) == "MAX_HOLD_TIME"]
    cp = pd.to_numeric(capd.pct, errors="coerce")
    q = (lambda s, x: float(s.quantile(x)) if s.notna().any() else float("nan"))
    return dict(n=len(f), wr=float((p > 0).mean()) if len(p) else float("nan"), avg=float(p.mean()) if len(p) else float("nan"),
                cap=cap, med=q(h, 0.5), p75=q(h, 0.75), p90=q(h, 0.9), mx=float(h.max()) if h.notna().any() else float("nan"),
                w_med=q(w, 0.5), l_med=q(l, 0.5), n_cap=len(capd), cap_wr=float((cp > 0).mean()) if len(cp) else float("nan"),
                cap_avg=float(cp.mean()) if len(cp) else float("nan"),
                open_at={x: int(sum(open_at_cap(v, x) for v in h)) for x in XS if x < cap})


# ───────────────────────────── I/O ─────────────────────────────
def _bars(pair, te, hold, fetch, bud, attempts, now_ms, notes):
    import scout_willy_timecap as WT
    k = WT.load_1m(pair)
    if not fetch or bud.get("stopped"):
        return k
    want = [(cap_fires_ms(te, x) // MIN) * MIN - MIN for x in XS if open_at_cap(hold, x)] + [((te + int(hold * MIN)) // MIN) * MIN - MIN]
    have = set(k.open_time.values.astype("int64").tolist()) if len(k) else set()
    if all(w in have for w in want):
        return k
    try:
        import path_labels as PL
        if PL._fetch_window(WT, pair, (te // MIN) * MIN - MIN, max(want) + 2 * MIN, now_ms, bud, attempts):
            WT._K1.pop(pair, None)
            k = WT.load_1m(pair)
    except Exception as e:
        notes.append(f"{pair}: 1m fetch failed ({type(e).__name__}: {str(e)[:60]})")
    return k


def price_fills(f, fetch=False, notes=None, bud=None, now_ms=None):
    """f: rows with pair, direction, opened_at, hold, close_reason → + px_<x> (price when a cap at x fires; None = unpriced / not open)
    + px_close (the ruler's price at the real close, MAX_HOLD_TIME fills only — the price-ruler parity)."""
    notes = notes if notes is not None else []
    bud = bud if bud is not None else {}
    bud.setdefault("req", 0)
    bud.setdefault("used", 0)
    bud.setdefault("t0", time.monotonic())
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    attempts = {}
    cols = {x: [] for x in XS}
    pxc = []
    req0 = int(bud["req"])
    for r in f.to_dict("records"):
        te, hold = to_ms(r["opened_at"]), r["hold"]
        if te is None or hold is None or pd.isna(hold) or not open_at_cap(hold, min(XS)):
            for x in XS:
                cols[x].append(None)
            pxc.append(None)
            continue
        k = _bars(str(r["pair"]), te, float(hold), fetch, bud, attempts, now_ms, notes)
        for x in XS:
            cols[x].append(price_at(k, cap_fires_ms(te, x)) if open_at_cap(hold, x) else None)
        pxc.append(price_at(k, te + int(round(hold * MIN))) if str(r.get("close_reason")) == "MAX_HOLD_TIME" else None)
    out = f.copy()
    for x in XS:
        out[f"px_{x}"] = cols[x]
    out["px_close"] = pxc
    if bud.get("stopped") and not bud.get("_noted"):
        bud["_noted"] = True
        notes.append(f"1m fetching stopped: {bud['stopped']} — re-run with --fetch later")
    if bud["req"] > req0:
        notes.append(f"1m fetch: {bud['req'] - req0} request(s), used weight {bud['used']}")
    return out


def load_current(path, since, probes=False):
    """CLOSED fills opened ≥ since (deduped). probes=True keeps probe rows (the ACCOUNT path: their $ moved the real equity)."""
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.status.astype(str) == "CLOSED")]
    d = d[d.opened_at.astype(str).str.replace(" ", "T").str[:19] >= since]
    if "is_probe" in d.columns and not probes:
        d = d[~d.is_probe.fillna(False).astype(str).str.lower().isin(("true", "1", "1.0"))]
    return d.drop_duplicates(["opened_at", "pair", "direction"], keep="last")


def fill_key(df):
    return (df.opened_at.astype(str).str.replace(" ", "T").str[:19] + df.pair.astype(str) + df.direction.astype(str)).tolist()


def _norm(d, pct_col):
    f = d.copy()
    f["pct"] = pd.to_numeric(f[pct_col], errors="coerce")
    f["hold"] = [hold_minutes(a, b) for a, b in zip(f.opened_at, f.closed_at)]
    n = len(f)
    col = (lambda c: f[c] if c in f.columns else [None] * n)
    f["fee_in"] = [entry_fee(t, fu, nv) for t, fu, nv in zip(col("entry_order_type"), col("entry_fee"), col("notional_value"))]
    return f


# ───────────────────────────── report ─────────────────────────────
def _m(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    v = float(v)
    return f"{v:.0f} min" if v < 120 else f"{v / 60:.1f} h"


def _pct(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:+.2f}"


def parity_line(f):
    """④ (a) the fee ruler on the sleeve's own market-closed fills (slip 0) vs their booked % · (b) the price ruler on MAX_HOLD_TIME
    fills (px_close vs the real exit price). → (text, ok)."""
    m = f[f.close_reason.astype(str).isin(MARKET_EXITS)]
    m = m[pd.to_numeric(m.exit_price, errors="coerce").notna() & m.pct.notna()]
    if not len(m):
        a, ok = "fee ruler: no market-closed fill to check", True
    else:
        est = [net_pct(e, x, dr, fi, slip=0.0) for e, x, dr, fi in zip(m.entry_price, m.exit_price, m.direction, m.fee_in)]
        d = np.abs(np.array(est) - m.pct.values)
        ok = float(np.median(d)) <= 0.05
        a = f"fee ruler on {len(m)} market-closed fill(s), slip 0: median |Δ| {np.median(d):.3f} pts · max {d.max():.3f}"
    b = ""
    if "px_close" in f.columns:
        c = f[pd.to_numeric(f.px_close, errors="coerce").notna() & pd.to_numeric(f.exit_price, errors="coerce").notna()]
        if len(c):
            g = np.abs(c.px_close.astype(float).values / c.exit_price.astype(float).values - 1.0) * 100.0
            b = f" · price ruler on {len(c)} cap close(s): median |Δ| {np.median(g):.3f} % of price · max {g.max():.3f}"
            ok = ok and float(np.median(g)) <= 0.5
    return "parity — " + a + b + ("" if ok else " ⚠ RULER OFF — do not read ② until fixed"), ok


def sleeve_block(f, cap, title, account=None, priced=True):
    """markdown lines for one sleeve in one block. account = (acc frame with k + usd, t0, t1, keys) for the DCR column, else None.
    priced=False → the hold profile only (no ②, no 4 h price)."""
    L = [f"#### {title}", ""]
    if not len(f):
        return L + ["no fills", ""]
    p = profile(f, cap)
    L += [f"① cap **{_m(cap)}** · {p['n']} fills · WR {p['wr']:.0%} · avg {_pct(p['avg'])} % · hold median {_m(p['med'])} · p75 {_m(p['p75'])} · "
          f"p90 {_m(p['p90'])} · longest {_m(p['mx'])} · winners' median {_m(p['w_med'])} vs losers' {_m(p['l_med'])} · "
          f"reached the cap {p['n_cap']}" + (f" (WR {p['cap_wr']:.0%} · avg {_pct(p['cap_avg'])} %)" if p['n_cap'] else "")
          + " · still open at " + (", ".join(f"{_m(x)}: {n}" for x, n in p['open_at'].items()) or "–"), ""]
    xs = [x for x in XS if x < cap] if priced else []
    if xs:
        ptxt, _ok = parity_line(f)
        L += ["  " + ptxt, ""]
        days = pd.to_datetime(f.closed_at.astype(str).str.replace(" ", "T").str[:19], errors="coerce").dt.date.astype(str)
        lab = (f.pair.astype(str) + " " + f.opened_at.astype(str).str[5:16]).tolist()
        L += ["| ② cap at | changed | WR | avg % | Σ pts | Δ Σ vs booked | P(Δ>0) day-boot | after 30–50 % haircut (avg Δ) | 🧮 leave-one-out | DCR %/day before → after | unpriced |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        base = f.pct.values
        dcr0 = None
        for x in xs:
            cf = [cap_at(b, h, x, px, e, dr, fi) for b, h, px, e, dr, fi in zip(f.pct, f.hold, f[f"px_{x}"], f.entry_price, f.direction, f.fee_in)]
            unp = sum(v is None for v in cf)
            v = np.array([b if c is None else c for b, c in zip(base, cf)], dtype=float)   # unpriced → booked (Δ 0, counted)
            d = v - base
            ch = int(sum(1 for c, h in zip(cf, f.hold) if c is not None and open_at_cap(h, x)))
            pb = boot_p(d, days) if ch else float("nan")
            dcr = "–"
            if account is not None:
                acc, t0, t1, keys = account
                m = acc.k.isin(set(keys))
                if not (int(m.sum()) == len(f) == len(set(keys))):   # a missed key would print after == before silently
                    raise ValueError(f"{title}: DCR join matched {int(m.sum())} of {len(f)} fills ({len(set(keys))} distinct keys)")
                a = acc.copy()
                dmap = dict(zip(keys, d / 100.0 * pd.to_numeric(f.notional_value, errors="coerce").fillna(0.0).values))
                a["u1"] = a.usd + a.k.map(dmap).fillna(0.0)
                if dcr0 is None:
                    dcr0 = RM.dcr_account(a, m, "usd", B25_START, t0, t1, B25_FLOWS)
                dcr = f"{dcr0:+.3f} → {RM.dcr_account(a, m, 'u1', B25_START, t0, t1, B25_FLOWS):+.3f}"
            L.append(f"| {_m(x)} | {ch} | {(v > 0).mean():.0%} | {v.mean():+.2f} | {v.sum():+.2f} | {d.sum():+.2f} | "
                     f"{'–' if not np.isfinite(pb) else f'{pb:.2f}'} | {haircut_text(d.mean())} | "
                     f"{RM.loo_text(d, lab) if ch else '–'} | {dcr} | {unp} |")
        L.append("")
    lh = f[pd.to_numeric(f.hold, errors="coerce") >= LONG_HOLD_MIN]
    if len(lh):
        L += [f"③ held ≥ {_m(LONG_HOLD_MIN)}:", ""]
        for r in lh.sort_values("hold", ascending=False).to_dict("records"):
            at4 = (cap_at(r["pct"], r["hold"], LONG_HOLD_MIN, r.get(f"px_{LONG_HOLD_MIN}"), r["entry_price"], r["direction"], r["fee_in"])
                   if priced and LONG_HOLD_MIN < cap else None)
            man = " ⚠ operator-closed (MANUAL)" if str(r["close_reason"]).upper() == "MANUAL" else ""
            L.append(f"- {r['pair']}{(' #' + str(int(r['id']))) if pd.notna(r.get('id')) else ''} opened {str(r['opened_at'])[:16]} · "
                     f"held {_m(r['hold'])} · {r['close_reason']}{man} {_pct(r['pct'])} % · market close at 4 h "
                     + (f"{_pct(at4)} %" if at4 is not None else ('unpriced' if priced else 'not priced (master)')))
        L.append("")
    return L


def _era_ms():
    try:
        import scout_gatekit as GK
        return GK.dl_deploy_ms(LITE_ERA_DL, lag_min=10, label="FRENZY duration LITE era split")
    except Exception:
        return None


def main(argv):
    fetch = "--fetch" in argv
    since = argv[argv.index("--since") + 1] if "--since" in argv else None
    if since is None:
        import sweep_separators as SW
        since = SW.BATCH_START
    files = [a for a in argv[1:] if a.endswith(".csv")]
    if not files:
        import path_labels as PL
        ex = PL._exports()
        files = ex[-1:]
    cap = caps()
    era = _era_ms()
    notes, bud = [], {}
    out = ["# ⏱ FRENZY MAX DURATION — per sleeve (DECISION_LOG 496 · plan ②z3)", "",
           "Live caps: " + " · ".join(f"{s} {_m(c)}" for s, c in cap.items())
           + f" · ruler: cap fires at X + 8 s, market close at the last closed 1m bar, taker + {SLIP:.2f} slip", ""]
    if files:
        import sweep_separators as SW
        cur = load_current(files[0], since)
        acc = None
        if since[:19] == SW.BATCH_START[:19]:          # the $2,900 start + 10-26 top-up belong to B25 only → DCR only for that batch
            acc = load_current(files[0], since, probes=True).copy()
            acc["k"] = fill_key(acc)
            acc["usd"] = pd.to_numeric(acc.pnl, errors="coerce").fillna(0.0)
            t0 = pd.Timestamp(since)                      # every calendar day from the batch start counts (empty days too)
            t1 = pd.to_datetime(acc.closed_at, format="mixed").max()
        else:
            notes.append(f"--since {since} is not the current batch start ({SW.BATCH_START}) — DCR not printed (the account path is B25's)")
        out += [f"## CURRENT BATCH — {os.path.basename(files[0])}, fills opened ≥ {since[:16]} UTC (closed fills only; open trades are not in the export)", ""]
        for s in SLEEVES:
            f = _norm(cur[cur.entry_strategy.astype(str) == s], "pnl_percentage")
            if f.pct.isna().any():
                notes.append(f"{s}: {int(f.pct.isna().sum())} fill(s) without a booked % left out")
                f = f[f.pct.notna()]
            f = price_fills(f, fetch, notes, bud).reset_index(drop=True)
            parts = [(s, f)]
            if s in LITE_FAMILY and len(f):
                if era is None:
                    notes.append(f"{s}: DL {LITE_ERA_DL} deploy time not found — pre / post LITE C eras NOT split (read with care)")
                else:
                    post = pd.Series([(to_ms(o) or 0) >= era for o in f.opened_at])
                    lbl = f"{pd.Timestamp(era, unit='ms'):%m-%d %H:%M} UTC"
                    parts = [(f"{s} — after the LITE C revert (opened ≥ {lbl}, today's +3 TP)", f[post.values].reset_index(drop=True)),
                             (f"{s} — BEFORE the LITE C revert (old +2 TP out of zone — another exit era, context only)", f[~post.values].reset_index(drop=True))]
            for title, part in parts:
                out += sleeve_block(part, cap[s], title, (acc, t0, t1, fill_key(part)) if (len(part) and acc is not None) else None)
    else:
        out += ["## CURRENT BATCH — no orders export found", ""]
    if os.path.exists(MASTER):
        m = pd.read_csv(MASTER, low_memory=False)
        m = m[m.stack_keep.astype(str).str.lower().isin(("true", "1", "1.0"))]
        if "is_probe" in m.columns:
            m = m[~m.is_probe.fillna(False).astype(str).str.lower().isin(("true", "1", "1.0"))]
        out += ["## MASTER (stack_keep · WR / avg on stack_pct = today's stack · holds AS TRADED) — hold profile only: a shorter-cap "
                "read here needs a walk under today's TP / stop (the re-priced rows are the TP winners a cap would cut; pricing only the "
                "unchanged rows would be one-sided). DCR: no per-batch account path in the master.", ""]
        for s in SLEEVES:
            ms = m[m.entry_strategy.astype(str) == s]
            if not len(ms):
                out += [f"#### {s}", "", "no master fills", ""]
                continue
            same = (pd.to_numeric(ms.stack_pct, errors="coerce") - pd.to_numeric(ms.pnl_percentage, errors="coerce")).abs() < 1e-6
            f = _norm(ms, "stack_pct")
            f = f[f.pct.notna()].reset_index(drop=True)
            out += sleeve_block(f, cap[s], f"{s} — {len(f)} fills ({int((~same).sum())} re-priced on today's stack → their hold is the traded one), eras: "
                                + ", ".join(f"{k} {v}" for k, v in ms.era.astype(str).value_counts().items()), None, priced=False)
    out += ["---", "Not tested here (blind spots): funding over long holds · 1 s / tick paths (1m closes only) · a master shorter-cap read (needs a "
            "walk under today's exit) · a time-based TIGHTEN (not at TP by X → lower TP / tighter stop) · the entry × exit joint check with the "
            "pending entry lines · in the DCR the cap's Δ$ lands on the fill's real close day, not the cap's (wrong day only on long holds)."]
    out += [f"- {n}" for n in notes]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv)
