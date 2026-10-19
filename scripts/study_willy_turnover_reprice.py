#!/usr/bin/env python3
"""📊 2026-10-19 research (WILLY turnover filter re-price) — READ-ONLY on code / config; cache-only (no network).

Re-prices the backtest WILLY A cohort of reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv (cohort willyA, built by
scripts/study_vol_mcap_cohorts.py: t0 = the first-flag 5m bar CLOSE; priced there by study_vol_mcap_willy_price.py with an IMMEDIATE
entry at t0 + 8 s under the OLD exit +1 / −3 / 60) under TODAY's live WILLY entry + exit:
  ENTRY (services/trading_engine._frenzy_willy_eval): armed on the trigger bar (open = t0 − 5 min); opens on the first CLOSED RED 5m bar
    (close < open; the trigger bar counts) whose open ≤ armed + frenzy_willy_red_max_wait_minutes (60) → entry = first aggTrades print ≥
    red close + 8 s (live ≈ 7.5 s), FRENZY dislocation guard (|print / last print before red close − 1| > 1 % → refused), × (1 + 0.035 %)
    slip (the old study's ruler). No tick day → the 1m ruler: open of the 1m bar starting at the red close × 1.00035 (src 1m).
    Episode-end expiry (unflag inside the wait) is NOT modelled (no flag state in the cohort).
  R at the red bar (engine reads R there): vol24 = Σ qvol of the 288 closed 5m bars before the red close; mcap = mcap_b × (red close /
    trigger-bar close) (mcap_b = the study's mcap at t0 = today's supply × t0 price). R_t0 (the study's R) kept for comparison.
  EXIT (trading_config.json thresholds, read at run time): TP +1.25 net · stop −4 net · 90-min cap · late TP +0.25 from minute 30 ·
    resting maker TP (book="live": a TP books exactly the line) → scout_willy_timecap.evaluate_late(book="live") on
    study_b19_fzexit_common.load_path (ticks first, 1m tail; bar touching both lines = stop) — the walker of build_willy_exit_cf.py (master).
  Also: IMMEDIATE entry (t0 + 8 s, the old study's entry) under today's exit — separates the entry change from the exit change.
Output: reports/WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv
Usage: venv/bin/python -I scripts/study_willy_turnover_reprice.py
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_b19_fzexit_common as FX          # noqa: E402
WT = FX.WT
C = os.path.join(ROOT, "reports", "backtest_cache")
IN = os.path.join(ROOT, "reports", "FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv")
OUT = os.path.join(ROOT, "reports", "WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv")
MIN, BAR = 60_000, 300_000
DELAY, SLIP_IN, DISLOC = 8_000, 0.035, 1.0

TH = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
TP = float(TH["frenzy_willy_tp_pct"]); SL = abs(float(TH["frenzy_willy_stop_pct"])); CAP = int(TH["frenzy_willy_max_hold_minutes"])
LATE_M = float(TH["frenzy_willy_late_tp_minutes"]); LATE_P = float(TH["frenzy_willy_late_tp_pct"])
WAIT = int(TH["frenzy_willy_red_max_wait_minutes"]) * MIN
BOOK = "live" if TH.get("resting_tp_willy_enabled") else "print"
RULE = f"+{TP:g}/-{SL:g}/{CAP}/late{LATE_M:g}@+{LATE_P:g}/{BOOK}"


def k5(pair, _c={}):
    if pair not in _c:
        _c.clear()
        d = pd.read_csv(os.path.join(C, "k5m_full", f"{pair}.csv")).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        _c[pair] = d
    return _c[pair]


def entry_at(pair, t_ms, tcache):
    """→ (E, te, src, st). Ticks: first print ≥ t_ms + 8 s with the dislocation guard; else the 1m bar open at t_ms."""
    d = f"{pd.Timestamp(t_ms, unit='ms'):%Y-%m-%d}"
    z = WT._load_day(pair, d, tcache)
    if z is not None and (z[2] is None or WT._covers(z[2], (t_ms - MIN, t_ms + MIN))):
        t, p = z[0], z[1]
        k = int(np.searchsorted(t, t_ms + DELAY, side="left"))
        pre = int(np.searchsorted(t, t_ms, side="left")) - 1
        if k < len(t) and pre >= 0 and t[k] - t_ms <= 60_000:
            raw, ref = float(p[k]), float(p[pre])
            if abs(raw / ref - 1) * 100 > DISLOC:
                return None, int(t[k]), "ticks", "disloc"
            return raw * (1 + SLIP_IN / 100), int(t[k]), "ticks", "ok"
    k1 = WT.load_1m(pair)
    m = k1[k1.open_time.values == (t_ms // MIN) * MIN]
    if len(m) and np.isfinite(m.o.iloc[0]):
        return float(m.o.iloc[0]) * (1 + SLIP_IN / 100), int((t_ms // MIN) * MIN), "1m", "ok"
    return None, None, "none", "noentry"


def walk(pair, E, te, tcache):
    P = FX.load_path(pair, te, E, CAP, tcache)
    if not len(P["t"]):
        return dict(why="UNRES", pct=np.nan, xmin=np.nan, path="none", dec_on_bar=None)
    ev = WT.evaluate_late(P, TP, SL, CAP, LATE_M, LATE_P, book=BOOK)
    path = "ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m")
    return dict(why=ev["why"], pct=float(ev["pct"]) if ev["why"] != "UNRES" else np.nan, xmin=ev["xmin"], path=path,
                dec_on_bar=bool(ev.get("on_bar")) if ev["why"] != "UNRES" else None, worst=ev.get("worst"))


def one(r, tcache):
    pair, t0 = r["pair"], int(r["t0"])
    k = k5(pair)
    o = k.open_time.values
    trig_open = t0 - BAR
    i0 = int(np.searchsorted(o, trig_open))
    out = dict(idx=r["idx"])
    if i0 >= len(o) or o[i0] != trig_open:
        out.update(st="no_trigger_bar"); return out
    trig_close_px = float(k.c.iloc[i0])
    j = None
    for i in range(i0, len(o)):
        if o[i] > trig_open + WAIT:
            break
        if i > i0 and o[i] != o[i - 1] + BAR:
            break                                           # kline gap: unknown bar → stop (conservative)
        if float(k.c.iloc[i]) < float(k.o.iloc[i]):
            j = i; break
    # immediate-entry variant (the old study's entry) under today's exit
    E0, te0, s0, st0 = entry_at(pair, t0, tcache)
    if st0 == "ok":
        w0 = walk(pair, E0, te0, tcache)
        out.update(imm_pct=w0["pct"], imm_why=w0["why"], imm_src=s0, imm_path=w0["path"])
    else:
        out.update(imm_pct=np.nan, imm_why=st0)
    if j is None:
        out.update(st="red_expired"); return out
    red_close = int(o[j]) + BAR
    lo = max(0, j + 1 - 288)
    vol24 = float(k.qvol.iloc[lo:j + 1].sum()) if (j + 1 - lo) == 288 else np.nan
    mcap_red = r["mcap_b"] * float(k.c.iloc[j]) / trig_close_px if np.isfinite(r["mcap_b"]) else np.nan
    out.update(wait_bars=j - i0, red_ret=(float(k.c.iloc[j]) / float(k.o.iloc[j]) - 1) * 100, t_red=red_close, vol24_red=vol24,
               mcap_red=mcap_red, R_red=vol24 / mcap_red if (np.isfinite(vol24) and np.isfinite(mcap_red) and mcap_red > 0) else np.nan)
    E, te, src, st = entry_at(pair, red_close, tcache)
    out.update(entry_src=src)
    if st != "ok":
        out.update(st=st); return out
    w = walk(pair, E, te, tcache)
    out.update(st="ok" if w["why"] != "UNRES" else "unres", te=te, E=E, **{k_: v for k_, v in w.items()})
    return out


def work(rows):
    tc, res = {}, []
    for r in rows:
        try:
            res.append(one(r, tc))
        except Exception as ex:  # noqa
            res.append(dict(idx=r["idx"], st=f"err:{type(ex).__name__}:{ex}"[:120]))
        if len(tc) > 12:
            tc.clear()
    return res


if __name__ == "__main__":
    D = pd.read_csv(IN)
    A = D[D.cohort == "willyA"].reset_index(drop=True)
    A["idx"] = A.index
    rows = A[["idx", "pair", "t0", "mcap_b"]].to_dict("records")
    pairs = sorted(A.pair.unique()); g = {p: n % 8 for n, p in enumerate(pairs)}
    chunks = [[r for r in rows if g[r["pair"]] == n] for n in range(8)]
    with Pool(8) as P:
        res = sum(P.map(work, chunks), [])
    R = pd.DataFrame(res).set_index("idx").sort_index()
    keep = ["pair", "t0", "day", "bear", "vol24", "mcap_b", "R", "pct", "why"]
    out = pd.concat([A[keep].rename(columns={"pct": "old_pct", "why": "old_why", "R": "R_t0", "vol24": "vol24_t0"}), R], axis=1)
    out["rule"] = RULE
    out.to_csv(OUT, index=False)
    print(RULE)
    print(out.st.value_counts(dropna=False).to_dict())
    ok = out[out.st == "ok"]
    print(f"priced {len(ok)} · WR {(ok.pct > 0).mean() * 100:.1f}% · avg {ok.pct.mean():+.3f} · {ok.why.value_counts().to_dict()}")
    print("path", ok.path.value_counts().to_dict(), "entry", ok.entry_src.value_counts().to_dict())
