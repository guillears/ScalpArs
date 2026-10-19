#!/usr/bin/env python3
"""LITE GREEN-RETRY deep dive 2026-10-19 — block ① (real era) + ② (per batch). Research only: read-only on code / config / caches.

Operator idea (AGT 10-16 02:00): when LITE's ONE judgement of an above-VWAP stretch is refused for a GREEN signal candle
(FRENZY_LITE_GREEN_BAR), wait for the next red candle instead of giving up on the stretch.

Built on the engine replica of reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19 (scripts/study_frenzy_realfills_engine.py — 96 % of real LITE
fills matched on bar + sleeve): the same 5m walk at T = 100, the same LONG / LITE / WILLY-hold / pair-held / slots / pair-day cap / bearish /
dislocation sequencing, the same entry ruler (close + 8 s first print × 1.001) and the same LITE C exit (TP +3 if vol ≥ 20× ∧ BTC 1h EMA20
slope > 0 else +2, stop −3, 12 h, maker TP at the line, taker stop + 0.03 slip), $2,450 notional (8× at the $1,225 account).

FROZEN VARIANTS (written before any variant result was read). A green judged bar still JUDGES the stretch for the normal path; the variant only
adds ONE pending retry for that stretch, which ends at its first qualifying bar (filled or refused) or when the window closes:
  V0        today — the green bar consumes the stretch, no retry.
  S{N}r     N ∈ {5, 15, 30, 60} min / SXr (until the stretch breaks): the first red / flat bar (close ≤ open, LITE's own colour rule) of the
            SAME stretch (same stretch id = above-VWAP streak unbroken) closing ≤ green close + N min, RE-CHECKING LITE's status at that bar
            (verified flag, not in the FRENZY state, ≥ 12 closes above the spike VWAP, last-hour vol < 100×, 2 h ≤ age ≤ 16.75 h, 24 h vol ≥ $20M —
            services.frenzy.frenzy_lite_status) and the open filters (bearish-day block, pair held, LITE slots ≤ 2, pair-day cap 3, 1 %
            dislocation). A red bar that fails the LITE status keeps the wait open; a refusal at the open ends the retry (one judgement).
            The WILLY global hold refuses without judging (engine rule, ≤ 6 bars). gvol < 1 is NOT applied: frenzy_gvol_max = 0 (gate OFF today).
  S{N}n     same windows, NO re-check: the first red / flat bar of the same stretch inside the window is entered whatever the LITE status / bearish
            day / pair-day cap / dislocation say; only the physical constraints apply (pair held, LITE slots, WILLY hold).
  W60       WILLY-style: the first STRICTLY red bar (close < open) closing ≤ green close + 60 min, the stretch need NOT survive, LITE status not
            re-checked; open filters re-checked (bearish, pair held, slots, pair-day cap, dislocation).
  PRIMARY   SXr (= the operator's literal rule; = the Part F row 'green bar does NOT consume'). The rest is a family of 9 more cells → any
            'best cell' carries the 30–50 % haircut and needs LODO.
  REF       RETRY_ANY = 'retry every refused bar until a fill' (Part F judge_once=False) — the DL 243-class rule, for contrast only.
Read: per variant N · WR · avg % · Σ$ · Δ$ vs V0 [paired day-block CI] · P(Δ>0) · added / lost fills (LITE and LONG cannibalisation) · per batch.
Usage: venv/bin/python scripts/study_lite_green_retry_deep.py price|sim|report [workers]
"""
import os, sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_engine as EN  # noqa: E402
from services.frenzy import frenzy_lite_status, FRENZY_LITE_JUDGES, FRENZY_LITE_READY  # noqa: E402
RF = RC.RF
OUTD = os.path.join(RC.SCR, "lgr"); os.makedirs(OUTD, exist_ok=True)
OUT = "reports/LITE_GREEN_RETRY_DEEP_2026-10-19"
MS5, MIN, HR = 300_000, 60_000, 3_600_000
NOTL = RC.NOTIONAL
VARIANTS = [dict(name="V0")] + \
    [dict(name=f"S{n}r", win=n, same=True, recheck=True, strict=False) for n in (5, 15, 30, 60)] + \
    [dict(name="SXr", win=None, same=True, recheck=True, strict=False)] + \
    [dict(name=f"S{n}n", win=n, same=True, recheck=False, strict=False) for n in (5, 15, 30, 60)] + \
    [dict(name="SXn", win=None, same=True, recheck=False, strict=False)] + \
    [dict(name="W60", win=60, same=False, recheck="open", strict=True)] + \
    [dict(name="RETRY_ANY", judge_once=False)]


def price_any(a):
    """price_one without the dislocation early return: the path is priced from the first print whatever the move (flag kept)."""
    k, pair, tc, close = a
    te = int(tc) + 8_000
    try:
        P = RC.gross_path(pair, te, float(close), 720)
    except Exception as e:  # noqa: BLE001
        return k, None, repr(e)
    if P is None or not len(P["t"]):
        return k, None, "nopath"
    p0 = P["op"][0] if np.isfinite(P["op"][0]) else P["cl"][0]
    f = (1 + p0 / 100) * 1.001
    for x in ("hi", "lo", "op", "cl"):
        P[x] = ((1 + P[x] / 100) / f - 1) * 100
    P["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(P["hi"])[:-1]]
    o = dict(disloc=bool(abs(f - 1) > 0.01 + 0.001), move_pct=(f - 1) * 100, E=float(close) * f, n_bar=P["n_bar"],
             path_end_h=(P["end_ms"] - te) / HR)
    for tp in (2.0, 3.0):
        x = RC.walk(P, tp=tp, sl=3.0, cap=720)
        o[f"lite{int(tp)}_pct"], o[f"lite{int(tp)}_why"], o[f"lite{int(tp)}_min"] = x
    lx = RC.walk(P, tp=1.5, sl=3.0, cap=720)
    o.update(long_pct=lx[0], long_why=lx[1], long_min=lx[2])
    m, st, tst = RC.mfe_before(P, 720, 3.0)
    o.update(mfe=m, run5=m >= 5, t_stop3=tst)
    return k, o, None


def stretch_id(r):
    return int(r.bar_open) - (int(r.above_streak) - 1) * MS5 if int(r.above_streak) >= 1 else None


def retry_bars(W, LG):
    """superset of bars any variant can enter on: after every green judged bar (V0 log), the red / flat bars of the same stretch (first 8)
    and the strictly red bars ≤ 60 min (any stretch)."""
    X = W[W["T"] == 100.0].sort_values(["pair", "t_close"])
    G = LG[(LG.code == "FRENZY_LITE_GREEN_BAR")]
    byp = {p: g for p, g in X.groupby("pair")}
    need = []
    for g in G.itertuples():
        d = byp[g.pair]
        d = d[d.t_close > g.t_close]
        k = 0
        for r in d.itertuples():
            same = (stretch_id(r) == g.stretch)
            if r.t_close - g.t_close <= 60 * MIN and r.bar_ret < 0:
                need.append((r.pair, int(r.t_close), float(r.c)))
            if same and r.bar_red and k < 8:
                need.append((r.pair, int(r.t_close), float(r.c))); k += 1
            if not same and r.t_close - g.t_close > 60 * MIN:
                break
    N = pd.DataFrame(need, columns=["pair", "t_close", "c"]).drop_duplicates(["pair", "t_close"])
    N["key"] = N.pair + "|" + N.t_close.astype(str)
    return N


def simulate(W, PR, PA, BT, RP, v):
    """EN.simulate + the frozen retry variants (v). PR = the replica's priced red LITE signal bars, PA = price_any for retry bars."""
    th = EN.THX(EN.TH0, frenzy_state_vol_mult=100.0, frenzy_lite_max_hours=16.75, frenzy_lite_enabled=True)
    X = W[W["T"] == 100.0].sort_values(["t_close", "pair"])
    willy = RP[RP.entry_strategy == "FRENZY_WILLY"][["pair", "t_open", "t_shut"]].values
    opens = []
    judged, hold_first, pending_long, pd_count, fills, log, pend = {}, {}, {}, {}, [], [], {}
    bt_idx = BT.index.values
    judge_once = v.get("judge_once", True)
    retry = "win" in v

    def btc_at(tc):
        j = int(np.searchsorted(bt_idx, tc, side="right")) - 1
        return BT.iloc[j] if j >= 0 else None

    def busy(pair, t):
        if any(p == pair and a <= t < b for p, a, b in willy):
            return True
        return any(p == pair and a <= t < b for p, s, a, b in opens)

    def nslots(sleeve, t):
        return sum(1 for p, s, a, b in opens if s == sleeve and a <= t < b)

    def willy_hold(t):
        return any(a <= t < b for p, a, b in willy)

    def try_open(r, sleeve, how, filters=True, extra=None):
        te = int(r.t_close) + 8_000
        key = f"{r.pair}|{int(r.t_close)}"
        pr = PR.get(key)
        if how.startswith("RETRY") or pr is None or "err" in pr or pr.get("disloc"):
            pr = PA.get(key, pr)
        b = btc_at(r.t_close)
        if filters and b is not None and np.isfinite(b.d1) and np.isfinite(b.gap) and b.d1 < 0 and b.gap < 0:
            return "BEARISH_DAY"
        if busy(r.pair, te):
            return "PAIR_HELD"
        if nslots(sleeve, te) >= 2:
            return "SLOTS"
        dkey = (r.pair, pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"))
        if filters and pd_count.get(dkey, 0) >= 3:
            return "PAIR_DAY_CAP"
        if pr is None or "err" in pr:
            return "UNPRICED"
        if filters and pr.get("disloc"):
            return "DISLOCATION"
        if sleeve == "FRENZY_LONG":
            pct, why, mn = pr["long_pct"], pr["long_why"], pr["long_min"]; tp = 1.5
        else:
            slope = b.h1slope if b is not None else np.nan
            tp = 3 if (np.isfinite(r.vol_mult) and r.vol_mult >= 20 and np.isfinite(slope) and slope > 0) else 2
            pct, why, mn = pr[f"lite{tp}_pct"], pr[f"lite{tp}_why"], pr[f"lite{tp}_min"]
        opens.append((r.pair, sleeve, te, te + int((mn if np.isfinite(mn) else 720) * MIN)))
        pd_count[dkey] = pd_count.get(dkey, 0) + 1
        row = dict(pair=r.pair, sleeve=sleeve, t_close=int(r.t_close), te=te, era=r.era, spike_ts=int(r.spike_ts), hours=r.hours,
                   vol_mult=r.vol_mult, vs_vwap=r.vs_vwap, bar_ret=r.bar_ret, tp=tp, pct=pct, why=why, exit_min=mn, mfe=pr["mfe"],
                   n_bar=pr["n_bar"], usd=pct / 100 * NOTL[sleeve], how=how, btc_d1=(b.d1 if b is not None else np.nan),
                   btc_gap=(b.gap if b is not None else np.nan), btc_h1slope=(b.h1slope if b is not None else np.nan),
                   disloc=bool(pr.get("disloc", False)), E=pr.get("E"))
        if extra:
            row.update(extra)
        fills.append(row)
        return "OPEN"

    for r in X.itertuples():
        t = int(r.t_close)
        if r.pair in pending_long and not willy_hold(t + 8_000):
            on_t, on_c = pending_long.pop(r.pair)
            if t - on_t <= 6 * MS5 and abs(r.c / on_c - 1) <= 0.01 and r.in_state:
                res = try_open(r, "FRENZY_LONG", "CATCHUP")
                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LONG", code="CATCHUP_" + res))
        if r.fresh_on:
            if r.long_ready:
                if willy_hold(t + 8_000):
                    pending_long[r.pair] = (t, r.c); res = "WILLY_HOLD"
                else:
                    res = try_open(r, "FRENZY_LONG", "ON")
            else:
                res = r.long_code
            log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LONG", code=res))
        # ── pending green retry (variants) ──
        if retry and r.pair in pend:
            g = pend[r.pair]
            sid_now = stretch_id(r)
            age = t - g["t"]
            expired = (v["win"] is not None and age > v["win"] * MIN) or (v["same"] and sid_now != g["sid"])
            if expired:
                pend.pop(r.pair)
                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="RETRY_EXPIRED", stretch=g["sid"], green_t=g["t"]))
            else:
                qual = (r.bar_ret < 0) if v["strict"] else bool(r.bar_red)
                if qual:
                    ok = True
                    if v["recheck"] is True:
                        ep = EN.ep_of(r)
                        ready, code, _ = frenzy_lite_status(ep, th, r.vol24, None) if not r.in_state else (False, "IN_STATE", "")
                        ok = bool(ready)
                    if ok:
                        if willy_hold(t + 8_000):
                            g["hold"] = g.get("hold", 0) + 1
                            if g["hold"] > 6:
                                pend.pop(r.pair)
                                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="RETRY_HOLD_EXPIRED", stretch=g["sid"]))
                        else:
                            pend.pop(r.pair)
                            res = try_open(r, "FRENZY_LITE", "RETRY", filters=(v["recheck"] in (True, "open")),
                                           extra=dict(green_t=g["t"], green_c=g["c"], green_ret=g["ret"], wait_min=age / MIN,
                                                      move_from_green=(r.c / g["c"] - 1) * 100, same_stretch=(sid_now == g["sid"])))
                            log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="RETRY_" + res, stretch=g["sid"], green_t=g["t"]))
            if r.pair in pend:
                continue
        if not r.in_state:
            ep = EN.ep_of(r)
            ready, code, _ = frenzy_lite_status(ep, th, r.vol24, judged.get(r.pair))
            if code not in FRENZY_LITE_JUDGES:
                continue
            sid = int(r.bar_open) - (int(r.above_streak) - 1) * MS5
            if ready and willy_hold(t + 8_000):
                hf = hold_first.get(r.pair)
                if hf is None or hf[0] != sid:
                    hold_first[r.pair] = (sid, t)
                elif t - hf[1] > 6 * MS5:
                    judged[r.pair] = sid
                    log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="HOLD_EXPIRED", stretch=sid))
                    continue
                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="WILLY_HOLD", stretch=sid))
                continue
            if code == FRENZY_LITE_READY:
                res = try_open(r, "FRENZY_LITE", "LITE")
                if judge_once or res == "OPEN":
                    judged[r.pair] = sid
            else:   # GREEN_BAR
                res = code
                if judge_once:
                    judged[r.pair] = sid
                if retry:
                    pend[r.pair] = dict(sid=sid, t=t, c=float(r.c), ret=float(r.bar_ret))
            log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code=res, stretch=sid, hours=r.hours, vol_mult=r.vol_mult,
                            bar_ret=r.bar_ret, vs_vwap=r.vs_vwap, spike_ts=int(r.spike_ts)))
    return pd.DataFrame(fills), pd.DataFrame(log)


def load_all():
    W = EN.load_walk()
    PR = pd.read_pickle(f"{RF}/cands_priced.pkl"); BT = EN.btc_frame(); RP = EN.real_positions()
    return W, PR, BT, RP


def step_price(nw):
    W, PR, BT, RP = load_all()
    F0, L0 = simulate(W, PR, {}, BT, RP, VARIANTS[0])
    L0.to_pickle(f"{OUTD}/v0_log.pkl")
    N = retry_bars(W, L0)
    print("green judged bars", int((L0.code == "FRENZY_LITE_GREEN_BAR").sum()), "retry bars to price", len(N), flush=True)
    with Pool(nw) as p:
        res = p.map(price_any, [(r.key, r.pair, r.t_close, r.c) for r in N.itertuples()], chunksize=2)
    out = {k: (o if o is not None else {"err": e}) for k, o, e in res}
    pd.to_pickle(out, f"{OUTD}/retry_priced.pkl")
    print("priced", len(out), "errors", sum(1 for x in out.values() if "err" in x))


def step_sim():
    W, PR, BT, RP = load_all()
    PA = pd.read_pickle(f"{OUTD}/retry_priced.pkl")
    allf, alll = [], []
    for v in VARIANTS:
        F, Lg = simulate(W, PR, PA, BT, RP, v)
        F["variant"] = v["name"]; Lg["variant"] = v["name"]
        allf.append(F); alll.append(Lg)
        print(v["name"], len(F), F.groupby("sleeve").pct.agg(["size", "mean"]).round(3).to_dict(), flush=True)
    pd.concat(allf, ignore_index=True).to_pickle(f"{OUTD}/era_fills.pkl")
    pd.concat(alll, ignore_index=True).to_pickle(f"{OUTD}/era_log.pkl")


def step_green(nw):
    """theory: price the green judged bars themselves (entering ON the green bar) + parity of V0 LITE fills vs the real LITE fills."""
    L0 = pd.read_pickle(f"{OUTD}/v0_log.pkl"); W = EN.load_walk()
    G = L0[L0.code == "FRENZY_LITE_GREEN_BAR"][["pair", "t_close"]].merge(W[W["T"] == 100.0][["pair", "t_close", "c"]], on=["pair", "t_close"])
    G["key"] = G.pair + "|" + G.t_close.astype(str)
    with Pool(nw) as p:
        res = p.map(price_any, [(r.key, r.pair, r.t_close, r.c) for r in G.itertuples()])
    pd.to_pickle({k: (o if o is not None else {"err": e}) for k, o, e in res}, f"{OUTD}/green_priced.pkl")
    F = pd.read_pickle(f"{OUTD}/era_fills.pkl"); F0 = F[(F.variant == "V0") & (F.sleeve == "FRENZY_LITE")]
    RP = EN.real_positions(); RL = RP[RP.entry_strategy == "FRENZY_LITE"]
    a = set(zip(F0.pair, F0.t_close)); b = set(zip(RL.pair, RL.sig_close))
    b2 = set(zip(RL.pair, RL.sig_close + MS5)) | b
    hit = sum(1 for p_, s_ in zip(RL.pair, RL.sig_close) if (p_, s_) in a or (p_, s_ + MS5) in a)
    R = F[(F.variant == "SXr") & (F.how == "RETRY")]
    clash = sum(1 for k in zip(R.pair, R.t_close) if k in b2)
    print(f"PARITY real LITE fills {len(RL)} · replica V0 LITE fill on the same bar {hit} · retry bars that coincide with a real LITE fill {clash}")


if __name__ == "__main__":
    s = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if s == "price":
        step_price(nw)
    elif s == "sim":
        step_sim()
    elif s == "green":
        step_green(nw)
