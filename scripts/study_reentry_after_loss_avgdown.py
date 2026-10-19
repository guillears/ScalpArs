#!/usr/bin/env python3
"""AVERAGING DOWN (in-trade martingale) per active sleeve — operator 2026-10-19 (refocused from "re-entry after a loss").

PRE-REGISTERED (frozen before any result was read):
  Variants (add while the ORIGINAL position is still open, once, at net −X % of the original leg, X = 1/3 · 1/2 · 2/3 of the sleeve stop S):
    AD1  add 1× the original notional · exit: fixed-TP sleeves → TP (sleeve's own TP %, net) measured from the NEW average price; the
         original stop PRICE and the original time cap are unchanged (so a failed add doubles the loss). Trail sleeves (no TP) → the add
         rides the original exit (co-exit at the original's actual exit price; the trail is keyed to the original entry).
    AD2  as AD1 with 2× (true martingale).
    AD3  add 1× · exit the whole position at break-even + fees from the new average (net 0) — the "rescue" exit — else the original exit
         (fixed sleeves: original stop / cap; trail sleeves: original exit).
  Costs: taker 0.045 % both sides on the add; add slippage 0.03 % on entry and on any non-TP exit (TP = resting line, booked at the line).
  Path: ticks where cached (reports/backtest_cache/ticks*, scout_frenzy_willy_exit.load_path) else 1m bars. 1m ASSUMPTIONS (stop-first):
    a bar touching the add line and the stop → add, then stop · a bar touching TP and stop → stop · no TP on the add bar itself · a bar
    whose open is already through the add / stop line fills at the open · the baseline exit bar touching the add line → no add.
  Trail-sleeve trigger: the LIVE-POLLED trough stamp (trough_pnl ≤ −X) must confirm the add (a 1m wick the bot never polled is not an
  add); the 1m path only times it (AD3 needs the time).
  Baseline: master = stack_pct (STACK 2026-10-19a, kept, non-probe); live = as-traded pnl_percentage; backtest = yr6 replay pct.
  Fixed-TP sleeves (FRENZY_WILLY / LITE / LONG) are compared against THIS walker's own baseline (parity vs the recorded pct reported).
  $ = % × today's notional at the $1,225 account (live B22/B23 sizes). Day-block bootstrap 95 % CI (4,000 reps).
Outputs reports/AVERAGE_DOWN_2026-10-19_rows.csv (+ _summary.csv, _seqmart.csv, _margin.csv). The md report is written by hand from these.
"""
import os, sys, json, time, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
import binance_cooldown as BC  # noqa: E402

T = 0.045
SLIP = 0.03
MIN = 60_000
ACCOUNT = 1225.0
MMR = 1.0           # maintenance margin % assumed for liquidation distance (alt brackets 0.5–2.5 %)
SLOT_MARGIN = 240.0  # typical per-slot margin at the $1,225 account (live B22/B23 WILLY / LITE margins $200–$270)
CACHE = os.path.join(ROOT, "reports", "backtest_cache", "reentry_after_loss")
OUT = os.path.join(ROOT, "reports", "AVERAGE_DOWN_2026-10-19")

# sleeve: kind, dir, stop S (net %, stop distance used to place X), tp, cap (min), today's notional $, leverage
SL = {
    "MOM_LONG":     dict(kind="trail", d=1,  S=0.70, N=250,   L=5),
    "MOM_SHORT":    dict(kind="trail", d=-1, S=0.70, N=4800,  L=20),
    "FLIP_SHORT":   dict(kind="trail", d=-1, S=1.20, N=2400,  L=10),
    "SPIKE_FADE":   dict(kind="trail", d=-1, S=1.50, N=11870, L=20),
    "BULLRUN_LONG": dict(kind="trail", d=1,  S=0.90, N=4800,  L=20),
    "SURGE_LONG":   dict(kind="trail", d=1,  S=1.20, N=2300,  L=10),
    "BEARRUN_SHORT": dict(kind="trail", d=-1, S=0.90, N=1200, L=5),
    "FRENZY_WILLY": dict(kind="fixed", d=1,  S=4.0, tp=1.25, cap=90, late=(30, 0.25), N=2300, L=10),
    "FRENZY_LITE":  dict(kind="fixed", d=1,  S=3.0, tp=None, cap=720, late=None, N=1950, L=8),
    "FRENZY_LONG":  dict(kind="fixed", d=1,  S=3.0, tp=1.5, cap=720, late=None, N=100, L=1),
}
MAP_MASTER = {("MOMENTUM", "LONG"): "MOM_LONG", ("MOMENTUM", "SHORT"): "MOM_SHORT", ("FLIP:FAN_RATIO_GATE", "SHORT"): "FLIP_SHORT",
              ("SPIKE_FADE", "SHORT"): "SPIKE_FADE", ("BULLRUN_LONG", "LONG"): "BULLRUN_LONG", ("SURGE_LONG", "LONG"): "SURGE_LONG",
              ("BEARRUN_SHORT", "SHORT"): "BEARRUN_SHORT", ("FRENZY_WILLY", "LONG"): "FRENZY_WILLY", ("FRENZY_LITE", "LONG"): "FRENZY_LITE",
              ("FRENZY_LONG", "LONG"): "FRENZY_LONG"}
MAP_BT = {"MOM-long": "MOM_LONG", "MOM-short": "MOM_SHORT", "FLIP-short": "FLIP_SHORT", "Spike-Fade": "SPIKE_FADE",
          "BullRun-Long": "BULLRUN_LONG", "SURGE_LONG": "SURGE_LONG", "BearRun-Short": "BEARRUN_SHORT", "FRENZY_WILLY": "FRENZY_WILLY",
          "FRENZY_LITE": "FRENZY_LITE", "FRENZY_LONG": "FRENZY_LONG"}
FRACS = ((1, 3), (1, 2), (2, 3))
VARIANTS = [(v, m, f) for v, m in (("AD1", 1), ("AD2", 2), ("AD3", 1)) for f in FRACS]


def lite_tp(r):
    try:
        return 3.0 if float(r.get("entry_frenzy_vol_mult")) >= 20 and float(r.get("entry_btc_1h_slope")) > 0 else 2.0
    except (TypeError, ValueError):
        return 2.0


# ───────────────────────── price helpers (ratio r = price / original entry) ─────────────────────────
def net_from_ratio(r, d):
    """original-leg net % of its notional at price ratio r."""
    return d * (r - 1) * 100 - T - T * r


def ratio_from_net(net, d, base=1.0):
    """price ratio (vs base) at which a position entered at base books `net` % (taker both sides)."""
    if d == 1:
        return base * (net + 100 + T) / (100 - T)
    return base * (100 - T - net) / (100 + T)


def pos_net(r, d, m, r_add, add_exit_slip):
    """whole-position P&L as % of the ORIGINAL notional at exit ratio r (original leg + m× add leg entered at r_add)."""
    leg0 = net_from_ratio(r, d)
    q1 = m / r_add
    leg1 = q1 * d * (r - r_add) * 100 - T * m - T * q1 * r - (SLIP * m * r / r_add if add_exit_slip else 0.0)
    return leg0 + leg1


# ───────────────────────── 1m data (local caches + own fetch) ─────────────────────────
_K = {}


def my1m(pair):
    if pair in _K:
        return _K[pair]
    fr = [WT.load_1m(pair)]
    p = os.path.join(CACHE, "1m", f"{pair}.csv")
    if os.path.exists(p):
        fr.append(WT._read_k1(p))
    k = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    _K[pair] = k
    return k


def covered(pair, a, b):
    k = my1m(pair)
    first = -(-int(a) // MIN) * MIN
    need = np.arange(first, (int(b) // MIN) * MIN, MIN)
    return np.isin(need, k.open_time.values).all() if len(need) else True


def fetch(pair, a, b):
    import requests
    os.makedirs(os.path.join(CACHE, "1m"), exist_ok=True)   # review: not at import (the scout imports this module)
    out = []
    s = -(-int(a) // MIN) * MIN
    while s < b:
        while BC.active():
            print("cooldown active — waiting 60 s", flush=True); time.sleep(60)
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=pair, interval="1m", startTime=s, limit=1000), timeout=20)
        if r.status_code in (418, 429):
            raise SystemExit(f"Binance {r.status_code} — stopping")
        w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
        if w > 700:
            print(f"used weight {w} — pause 30 s", flush=True); time.sleep(30)
        if r.status_code != 200:
            print(pair, r.status_code, r.text[:100]); break
        js = r.json()
        if not js:
            break
        out += [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in js]
        s = int(js[-1][0]) + MIN
        time.sleep(0.15)
    if out:
        p = os.path.join(CACHE, "1m", f"{pair}.csv")
        new = pd.DataFrame(out, columns=["open_time", "o", "h", "l", "c"])
        if os.path.exists(p):
            new = pd.concat([WT._read_k1(p), new]).drop_duplicates("open_time", keep="last").sort_values("open_time")
        new.to_csv(p, index=False)
        _K.pop(pair, None)


# ───────────────────────── path (ratio space) ─────────────────────────
def get_path(pair, te, E, horizon_min, tcache):
    """events after te: t, hi, lo, cl, op (price ratios vs a unit entry), bar flag."""
    P = FX.load_path(pair, int(te), float(E), int(horizon_min) + 1, tcache, k1m=my1m(pair))
    inv = lambda x: (np.asarray(x, float) + 100 + T) / (100 - T)  # noqa: E731  exact inverse of WT.net_pct → price / E
    return dict(t=P["t"], hi=inv(P["hi"]), lo=inv(P["lo"]), cl=inv(P["cl"]), op=inv(P["op"]), bar=P["bar"], end=P["end_ms"],
                src=FX.path_src(P))


def adv_fav(P, d):
    return (P["lo"], P["hi"]) if d == 1 else (P["hi"], P["lo"])


def first_idx(mask, start=0):
    i = np.flatnonzero(mask[start:])
    return int(i[0]) + start if len(i) else None


def through(x, line, d, adverse=True):
    """adverse: price moved against the position to/through `line`; else favourable."""
    if adverse:
        return x <= line if d == 1 else x >= line
    return x >= line if d == 1 else x <= line


def walk_fixed(P, d, te, S, tp, cap, late, add=None):
    """→ (pct of original notional, why, add info). add = (m, X, mode 'TP'|'BE') or None for the baseline."""
    n = int(np.searchsorted(P["t"], te + cap * MIN, side="right"))
    if n == 0:
        return np.nan, "UNRES", None
    full = P["end"] >= te + cap * MIN
    adv, fav = adv_fav(P, d)
    adv, fav, t, bar, op, cl = adv[:n], fav[:n], P["t"][:n], P["bar"][:n], P["op"][:n], P["cl"][:n]
    r_stop = ratio_from_net(-S, d)

    def tp_line(base, mins_arr, tpv):
        if late is None:
            return np.full(len(mins_arr), ratio_from_net(tpv, d, base))
        lt_min, lt_pct = late
        return np.where(mins_arr >= lt_min, ratio_from_net(lt_pct, d, base), ratio_from_net(tpv, d, base))

    mins = (t - te) / MIN
    # baseline exit index
    tl0 = tp_line(1.0, mins, tp)
    i_tp0 = first_idx(through(fav, tl0, d, False))
    i_st0 = first_idx(through(adv, r_stop, d, True))

    def stop_fill(i, line):
        if bar[i] and np.isfinite(op[i]) and through(op[i], line, d, True):
            return op[i]
        return line if bar[i] else adv[i]

    def base_result():
        if i_st0 is not None and (i_tp0 is None or i_st0 <= i_tp0):
            return net_from_ratio(stop_fill(i_st0, r_stop), d), "STOP", i_st0
        if i_tp0 is not None:
            r = tl0[i_tp0] if bar[i_tp0] else fav[i_tp0]
            return net_from_ratio(r, d), "TP", i_tp0
        return (net_from_ratio(cl[n - 1], d), "CAP", n - 1) if full else (np.nan, "UNRES", n - 1)
    b_pct, b_why, b_i = base_result()
    if b_why == "UNRES":
        return np.nan, "UNRES", None
    if add is None:
        return b_pct, b_why, None
    m, X, mode = add
    r_trig = ratio_from_net(-X, d)
    j = first_idx(through(adv, r_trig, d, True))
    if j is None or j > b_i or (j == b_i and b_why == "TP"):
        return b_pct, b_why, dict(trig=False)
    r_add = op[j] if (bar[j] and np.isfinite(op[j]) and through(op[j], r_trig, d, True)) else (r_trig if bar[j] else adv[j])
    r_add = r_add * (1 + d * SLIP / 100)
    q1 = m / r_add
    A = (1 + q1 * r_add) / (1 + q1)
    tl = tp_line(A, mins, 0.0 if mode == "BE" else tp)
    i_st = first_idx(through(adv, r_stop, d, True), j)
    i_tp = first_idx(through(fav, tl, d, False), j + 1)
    info = dict(trig=True, add_min=float(mins[j]), t_add=int(t[j]))
    if i_st is not None and (i_tp is None or i_st <= i_tp):
        return pos_net(stop_fill(i_st, r_stop), d, m, r_add, True), "STOP", info
    if i_tp is not None:
        r = tl[i_tp] if bar[i_tp] else fav[i_tp]
        return pos_net(r, d, m, r_add, False), ("BE" if mode == "BE" else "TP"), info
    if not full:
        return np.nan, "UNRES", info
    return pos_net(cl[n - 1], d, m, r_add, True), "CAP", info


def walk_trail(P, d, te, tx, base_pct, trough, trough_at, add):
    """co-exit with the original at its recorded exit; AD3 = BE exit from the new average before the original exit."""
    m, X, mode = add
    r_exit = ratio_from_net(base_pct, d)
    if not (np.isfinite(trough) and trough <= -X):
        return base_pct, "BASE", dict(trig=False)
    r_trig = ratio_from_net(-X, d)
    j = None
    if P is not None and len(P["t"]):
        k = int(np.searchsorted(P["t"], tx, side="left"))
        adv, fav = adv_fav(P, d)
        j = first_idx(through(adv[:k], r_trig, d, True))
    if j is not None:
        bar, op = P["bar"], P["op"]
        r_add = op[j] if (bar[j] and np.isfinite(op[j]) and through(op[j], r_trig, d, True)) else r_trig
        add_min = (P["t"][j] - te) / MIN
    else:
        r_add = r_trig
        add_min = (trough_at - te) / MIN if np.isfinite(trough_at) else np.nan
    r_add = r_add * (1 + d * SLIP / 100)
    info = dict(trig=True, add_min=float(add_min), t_add=int(te + (add_min if np.isfinite(add_min) else 0) * MIN), timed=j is not None)
    if mode == "BE":
        if j is None:
            return pos_net(r_exit, d, m, r_add, True), "COEXIT_UNTIMED", info
        q1 = m / r_add
        A = (1 + q1 * r_add) / (1 + q1)
        r_be = ratio_from_net(0.0, d, A)
        k = int(np.searchsorted(P["t"], tx, side="left"))
        fav = adv_fav(P, d)[1]
        i = first_idx(through(fav[:k], r_be, d, False), j + 1)
        if i is not None:
            return pos_net(r_be, d, m, r_add, False), "BE", info
    return pos_net(r_exit, d, m, r_add, True), "COEXIT", info


def ms(x):
    return int(pd.Timestamp(x).value // 1_000_000)


# ───────────────────────── cohorts ─────────────────────────
def load_master():
    d = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    d = d[(d.stack_keep == True) & (d.is_probe != True) & (d.status == "CLOSED")].copy()  # noqa: E712
    d["sleeve"] = [MAP_MASTER.get((s, dr)) for s, dr in zip(d.entry_strategy, d.direction)]
    d = d[d.sleeve.notna()].copy()
    d["block"] = "MASTER"; d["batch"] = d.era; d["base"] = d.stack_pct.astype(float)
    return d


def load_live():
    fs = [os.path.join(ROOT, "reports", "BASELINE22_batch1014-1016_LIVE_orders.csv"),
          os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-19_03-22-01.csv")]
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in fs if os.path.exists(f)], ignore_index=True)
    d = d[(d.status == "CLOSED") & (d.is_paper.astype(str).str.lower() == "false")].copy()
    d = d.drop_duplicates(["opened_at", "pair", "direction"])
    d["sleeve"] = [MAP_MASTER.get((s, dr)) for s, dr in zip(d.entry_strategy, d.direction)]
    d = d[d.sleeve.notna()].copy()
    d["block"] = "LIVE"; d["batch"] = np.where(d.opened_at < "2026-10-16T12:30", "B22", "B23"); d["base"] = d.pnl_percentage.astype(float)
    return d


def load_bt():
    F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_fills.csv"), low_memory=False)
    F["sleeve"] = F.sleeve.map(MAP_BT)
    F = F[F.sleeve.notna() & (F.year.astype(str) == "2026")].copy()
    F["block"] = "BT_s" + F.seed.astype(str); F["batch"] = F.month.astype(str); F["base"] = F.pct.astype(float)
    F["opened_at"] = F.t.astype(str)
    return F


COLS = ["block", "batch", "sleeve", "pair", "direction", "opened_at", "closed_at", "entry_price", "base", "trough_pnl", "trough_reached_at",
        "entry_frenzy_vol_mult", "entry_btc_1h_slope"]


def rows_of(df):
    out = []
    for r in df.reindex(columns=COLS).to_dict("records"):
        try:
            r["te"] = ms(r["opened_at"]); r["tx"] = ms(r["closed_at"])
        except Exception:
            continue
        try:
            r["ta"] = ms(r["trough_reached_at"]) if isinstance(r["trough_reached_at"], str) and r["trough_reached_at"] else np.nan
        except Exception:
            r["ta"] = np.nan
        r["day"] = str(r["opened_at"])[:10]
        out.append(r)
    return out


def horizon(r):
    c = SL[r["sleeve"]]
    if c["kind"] == "fixed":
        return c["cap"] + 2
    return max(2, (r["tx"] - r["te"]) / MIN + 2)


def job(rows):
    tc, out = {}, []
    for r in rows:
        c = SL[r["sleeve"]]; d = c["d"]
        rec = {k: r[k] for k in ("block", "batch", "sleeve", "pair", "opened_at", "day", "base", "trough_pnl")}
        try:
            P = get_path(r["pair"], r["te"], r["entry_price"], horizon(r), tc)
            rec["src"] = P["src"]
        except Exception as e:  # noqa: BLE001
            P = None; rec["src"] = f"ERR {e}"[:40]
        if c["kind"] == "fixed":
            tp = lite_tp(r) if r["sleeve"] == "FRENZY_LITE" else c["tp"]
            if P is None:
                continue
            b, why, _ = walk_fixed(P, d, r["te"], c["S"], tp, c["cap"], c["late"])
            rec["walk_base"], rec["walk_why"] = b, why
            if not np.isfinite(b):
                rec["unres"] = True; out.append(rec); continue
            ref = b
        else:
            ref = float(r["base"])
            rec["walk_base"] = ref
        rec["ref"] = ref
        for v, m, (a, bq) in VARIANTS:
            X = c["S"] * a / bq
            key = f"{v}_{a}{bq}"
            if c["kind"] == "fixed":
                pct, why, info = walk_fixed(P, d, r["te"], c["S"], tp, c["cap"], c["late"], add=(m, X, "BE" if v == "AD3" else "TP"))
            else:
                tr = float(r["trough_pnl"]) if pd.notna(r["trough_pnl"]) else np.nan
                pct, why, info = walk_trail(P, d, r["te"], r["tx"], ref, tr, r["ta"], (m, X, "BE" if v == "AD3" else "TP"))
            rec[key] = pct; rec[key + "_why"] = why
            rec[key + "_trig"] = bool(info and info.get("trig"))
            rec[key + "_tadd"] = info.get("t_add") if info and info.get("trig") else np.nan
        out.append(rec)
    return out


def ensure_1m(rows):
    need = {}
    for r in rows:
        if r["block"].startswith("BT"):
            continue
        a, b = r["te"], r["te"] + int(horizon(r)) * MIN
        if not covered(r["pair"], a, b):
            need.setdefault(r["pair"], []).append((a, b))
    print(f"fetching 1m for {sum(len(v) for v in need.values())} windows on {len(need)} pairs", flush=True)
    for p, ws in need.items():
        for a, b in ws:
            if not covered(p, a, b):
                fetch(p, a - 2 * MIN, b + 2 * MIN)


def main():
    M, Lv, B = load_master(), load_live(), load_bt()
    rows = rows_of(M) + rows_of(Lv) + rows_of(B)
    ensure_1m(rows)
    ps = sorted({r["pair"] for r in rows}); g = {p: k % 7 for k, p in enumerate(ps)}
    with Pool(7) as Pp:
        R = pd.DataFrame(sum(Pp.map(job, [[r for r in rows if g[r["pair"]] == k] for k in range(7)]), []))
    R.to_csv(OUT + "_rows.csv", index=False)
    print("rows", len(R), R.groupby(["block", "sleeve"]).size().to_dict())


if __name__ == "__main__":
    main()
