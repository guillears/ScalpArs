#!/usr/bin/env python3
"""👁 Scout — WILLY_AVGDOWN observe line (DECISION_LOG 313, operator 2026-10-19: "add the watchlist yes"). OBSERVE ONLY — never changes config.

Origin: reports/AVERAGE_DOWN_2026-10-19.md (no sleeve earns averaging down; FRENZY_WILLY is the only candidate — master add leg +0.79 %
on 11 adds, backtest +0.19 on ~230 adds/seed, BT AD1 +$1,008…+1,059/seed with CI spanning 0, master gain all B19, BAT 35 %, live −$145
on 6 adds, worst trade −7.6 → −11.8 % of the account). AD2 (2× add) and sequence martingale are REJECTED — never a candidate here.
HYPOTHESIS: inside a frenzy, a WILLY dip to −2 % is close to a coin flip that sometimes reverts; adding 1× there pulls the average ~1 %
lower so the +1.25 TP (or the +0.25 late TP) becomes reachable — worth it only if the reverts pay for the doubled −4 stops.

FROZEN DEFINITION (never re-fit) — the walk IS the study's walker (study_reentry_after_loss_avgdown.walk_fixed, variant AD1 at X = 2.00)
  cohort   CLOSED FRENZY_WILLY fills opened ≥ COHORT_START (orders exports, paper + LIVE via export_glob), dedup (opened_at, pair, direction).
  path     the study's path: local ticks where cached, else public 1m klines (bars from the first whole minute after entry, t = bar close),
           horizon 90 + 2 min; prices on the taker/taker net basis (0.045 % each side).
  add      the FIRST time the fill touches NET −2.00 % → add 1× the original notional at that price (a bar opening through the line fills
           at its open) + 0.03 % slippage, taker 0.045 % on the add's entry and exit (+0.03 % slip on a non-TP exit). A bar touching the add
           line AND the stop → add, then stop (doubled loss); no TP on the add bar itself. The baseline exit bar touching the add line → no add if it was the TP bar.
  exit     merged position: TP +1.25 net from the NEW average (late TP +0.25 from minute 30, live DL 297), the ORIGINAL −4 stop PRICE, the
           90-min cap (close at market). 1m bars are stop-first (a bar touching TP and stop = stop); TP is booked at the line (resting maker).
  result   add leg = merged result − the original trade's own result on the same walk, in % of the ORIGINAL notional; $ at the fill's
           actual notional_value (fallback $2,300 = the study's 10× WILLY notional). Worst shadow trade = merged $ / ACCOUNT. The exports
           carry no balance column → ACCOUNT = $1,225 (stated in the output). Parity: walker baseline vs booked pnl_percentage, sign
           disagreements flagged.
GATE (frozen, pre-registered 2026-10-19, never re-fit)
  READ at N ≥ 30 shadow adds on ≥ 8 distinct UTC days. DISCUSS AD1 (operator decides; never AD2) only if add-leg mean > 0 with day-bootstrap
  P ≥ 0.95 ∧ no pair ≥ 50 % of the total gain ∧ total gain × 0.5 (haircut) > $0 ∧ no shadow trade worse than −12 % of the account.
  Otherwise at N ≥ 30 → DROP the line. Below N → "collecting (n/30 on d/8 days)".
  Worst trade (SIZING-INVARIANT, fixed pre-freeze by the deep review): merged % of notional × the reference WILLY notional $2,300 ÷ $1,225
  (≡ merged ≤ −6.39 % of notional fails) — never the fill's own notional ÷ a fixed account (would drift with the balance). The 50 %
  haircut is reported for the discussion; as a pass test it equals "total > 0" (already required). Conservative bias: TP exits are
  charged taker fees though WILLY TPs rest as maker orders (DL 296 / 310) — under-credits the add leg by ≈ 0.02 pp.
BINANCE  public fapi 1m klines only (1 request of weight 1 per fill); cache reports/backtest_cache/scout_willy_avgdown/; ≤ MAX_FILLS uncached
  fills per run, paced, stop once X-MBX-USED-WEIGHT-1M ≥ 700, shared scout cooldown honoured / armed on 418-429. A fill whose 92-min
  window is not complete yet stays PENDING.
Usage  venv/bin/python scripts/scout_willy_avgdown.py      (batch review; never calls the bot API)
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
import scout_frenzy_willy_exit as FX                         # noqa: E402  (the study's path builder)
import study_reentry_after_loss_avgdown as AD                # noqa: E402  (the study's walker — the scout and the study agree)
import willy_exit_era as WEra                                # noqa: E402  🎲² (DECISION_LOG 356) exit-era split / header

COHORT_START = "2026-10-19T13:00:00"
W = AD.SL["FRENZY_WILLY"]                                   # d 1 · S 4.0 · tp 1.25 · cap 90 · late (30, 0.25) · N 2300
ADD_X, ADD_M = 2.00, 1                                      # AD1 at net −2.00 % (frozen)
HORIZON = W["cap"] + 2                                      # the study's horizon(r) for fixed sleeves
ACCOUNT = 1225.0
GATE_N, GATE_DAYS, P_MIN, PAIR_MAX, HAIRCUT, WORST_ACCT = 30, 8, 0.95, 0.50, 0.5, -12.0
BOOT_REPS = 4000
MAX_FILLS, WEIGHT_STOP, PACE_S = 30, 700, 0.35
M1 = 60_000
_ROOT = os.path.dirname(_D)
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_willy_avgdown")
_EMPTY_K = pd.DataFrame({"open_time": pd.Series(dtype="int64"), "o": pd.Series(dtype=float), "h": pd.Series(dtype=float),
                         "l": pd.Series(dtype=float), "c": pd.Series(dtype=float)})


def _ms(s):
    return int(pd.Timestamp(str(s)[:19]).value // 1_000_000)


def _num(v):
    x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else np.nan


def fills():
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime):
        try:                                                    # export_glob adds the post-go-live LIVE exports (live rows only)
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in (
                "opened_at", "pair", "direction", "status", "entry_strategy", "entry_price", "pnl_percentage", "pnl", "notional_value",
                "is_paper", "close_reason") + tuple(WEra.V2_COLS)))   # DL 356: the V2 stamps ride the reader
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a = a[(a.entry_strategy.astype(str) == "FRENZY_WILLY") & (a.direction.astype(str).str.upper() == "LONG")   # the walker assumes d = +1
          & (a.opened_at.astype(str).str[:19] >= COHORT_START)]
    a = a.assign(_c=(a.status == "CLOSED").astype(int)).sort_values("_c").assign(_k=a.opened_at.astype(str).str[:19]).drop_duplicates(["_k", "pair", "direction"], keep="last")
    return a[a.status == "CLOSED"].reset_index(drop=True)


class _Bud:
    used, stop = 0, False


def _start(te):
    return -(-int(te) // M1) * M1                               # the study's first bar: the first whole minute at/after entry


def _cache_path(sym, start):
    return os.path.join(CACHE, f"{sym}_1m_{start}_{HORIZON + 1}.json")


def kl(sym, te, bud):
    """the fill's 1m window (HORIZON + 1 bars from the first whole minute) → DataFrame open_time/o/h/l/c, or None (budget / cooldown / pending)."""
    start, lim = _start(te), HORIZON + 1
    p = _cache_path(sym, start)
    if os.path.exists(p):
        with open(p) as f:
            return pd.DataFrame(json.load(f), columns=["open_time", "o", "h", "l", "c"])
    now = int(time.time() * 1000)
    if now < te + (HORIZON + 1) * M1:
        return None                                            # window not complete yet → pending
    if bud.stop or NF._cooldown_until() > now or bud.used >= WEIGHT_STOP:
        bud.stop = True
        return None
    time.sleep(PACE_S)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&startTime={start}&limit={lim}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
            bud.stop = True
            return None
        if e.code == 403:                                      # WAF / geo block: stop the run, never hammer
            bud.stop = True
            return None
        return _EMPTY_K.copy()
    except Exception:
        return _EMPTY_K.copy()                                 # one unreadable symbol / timeout = missing data, not a budget stop
    out = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows]
    if out and out[-1][0] + M1 <= now:                         # cache only complete windows
        os.makedirs(CACHE, exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f)
    return pd.DataFrame(out, columns=["open_time", "o", "h", "l", "c"]) if out else _EMPTY_K.copy()


def path_of(pair, te, E, k1m, tcache=None):
    """the study's get_path (ratio space vs entry) with our own 1m frame as the bar source (local ticks first, as the study)."""
    P = FX.load_path(pair, int(te), float(E), HORIZON + 1, {} if tcache is None else tcache, k1m=k1m)
    inv = lambda x: (np.asarray(x, float) + 100 + AD.T) / (100 - AD.T)  # noqa: E731
    return dict(t=P["t"], hi=inv(P["hi"]), lo=inv(P["lo"]), cl=inv(P["cl"]), op=inv(P["op"]), bar=P["bar"], end=P["end_ms"],
                src=FX.path_src(P))


def shadow(P, te):
    """→ dict(base, base_why, merged, merged_why, trig, add_min, leg) in % of the original notional; base / merged NaN when unresolved."""
    b, bw, _ = AD.walk_fixed(P, W["d"], te, W["S"], W["tp"], W["cap"], W["late"])
    if not np.isfinite(b):
        return dict(base=np.nan, base_why="UNRES", merged=np.nan, merged_why="UNRES", trig=False, add_min=np.nan, leg=np.nan)
    m, mw, info = AD.walk_fixed(P, W["d"], te, W["S"], W["tp"], W["cap"], W["late"], add=(ADD_M, ADD_X, "TP"))
    trig = bool(info and info.get("trig"))
    return dict(base=b, base_why=bw, merged=m, merged_why=mw, trig=trig, add_min=(info.get("add_min") if trig else np.nan),
                leg=(m - b) if np.isfinite(m) else np.nan)


def day_boot_p(vals, days, reps=BOOT_REPS, seed=0):
    """P(mean > 0) under a day-block bootstrap (resample UTC days with replacement, pool their adds)."""
    v, dd = np.asarray(vals, float), np.asarray(days)
    u = np.unique(dd)
    if not len(u):
        return np.nan
    groups = [v[dd == x] for x in u]
    sums, cnts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(reps, len(u)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float((means > 0).mean())


def verdict(adds, account=ACCOUNT):
    """adds = DataFrame of TRIGGERED shadow adds with day, pair, leg (%), leg_usd, merged_usd → (verdict string, detail dict)."""
    n, nd = len(adds), (adds.day.nunique() if len(adds) else 0)
    if n < GATE_N or nd < GATE_DAYS:
        return f"collecting ({n}/{GATE_N} on {nd}/{GATE_DAYS} days)", {}
    p = day_boot_p(adds.leg.values, adds.day.values)
    total = float(adds.leg_usd.sum())
    by_pair = adds.groupby("pair").leg_usd.sum()
    pair_share = float(by_pair.max() / total) if total > 0 else np.nan
    # deep review (pre-freeze): the worst check is SIZING-INVARIANT — merged % of the original notional × the reference WILLY notional
    # (W["N"] $2,300 = 10× at the $1,225 account) ÷ that account. merged_usd / a fixed $1,225 would drift toward DROP as the live
    # notional grows with the balance. −12 % of the account ↔ a merged trade of −6.39 % of notional (an add-then-stop is ≈ −6.2 %).
    worst = float((adds.merged * float(W["N"]) / account).min())
    det = dict(mean=float(adds.leg.mean()), p=p, total=total, pair=str(by_pair.idxmax()), pair_share=pair_share,
               haircut=total * HAIRCUT, worst_acct=worst)
    # the 50 % haircut is REPORTED (det['haircut']) for the discussion; as a pass test it equals total > 0 (already required), so it is
    # not a separate condition (deep review: it was a no-op)
    ok = (det["mean"] > 0 and p >= P_MIN and total > 0 and pair_share < PAIR_MAX and worst >= WORST_ACCT)
    return ("DISCUSS AD1 (operator decides; never AD2)" if ok else "DROP the line (frozen gate failed at N ≥ 30)"), det


def main():
    a = fills()
    import scout_gatekit as _GK                                 # 🧰 (DECISION_LOG 331) operator-closed fills: not the bot's exit → out, listed
    a, opx = _GK.split_operator_closed(a) if len(a) else (a, a)
    a = a.reset_index(drop=True)
    print(f"  {_GK.operator_closed_line(opx)}")
    print(WEra.era_line(a, "WILLY_AVGDOWN", note=("the shadow walks EVERY fill under the study's pinned exit (+1.25 / −4 / 90, late +0.25 @ 30 — "
                                                  "never the config), so V2 fills stay in the add-leg gate; V2 runner / floor fills are kept "
                                                  "OUT of the walker-vs-booked parity line") if WEra.counts(a)[2] else None))   # DL 356 (c)
    print(f"WILLY_AVGDOWN (DECISION_LOG 313, observe) · cohort FRENZY_WILLY closed fills opened ≥ {COHORT_START} UTC → {len(a)}"
          f"  (paper {int((a.is_paper.astype(str).str.lower() == 'true').sum()) if len(a) else 0} · live "
          f"{int((a.is_paper.astype(str).str.lower() == 'false').sum()) if len(a) else 0})")
    print(f"  shadow AD1: add 1× at net −{ADD_X:.2f} % · merged TP +{W['tp']} from the new average (late +{W['late'][1]} from min "
          f"{W['late'][0]}) · original −{W['S']:.0f} stop price · {W['cap']}-min cap · account ${ACCOUNT:,.0f} (no balance column in the exports)")
    if not len(a):
        print(f"  verdict: collecting (0/{GATE_N} on 0/{GATE_DAYS} days)")
        return
    bud, rows, pend, unres, fetched = _Bud(), [], 0, 0, 0
    for r in a.to_dict("records"):
        te, e = _ms(r["opened_at"]), float(r["entry_price"])
        cached = os.path.exists(_cache_path(r["pair"], _start(te)))
        if not cached and fetched >= MAX_FILLS:
            pend += 1
            continue
        k = kl(r["pair"], te, bud)
        fetched += 0 if cached else 1
        if k is None:
            pend += 1
            continue
        s = shadow(path_of(r["pair"], te, e, k), te)
        if not np.isfinite(s["base"]) or (s["trig"] and not np.isfinite(s["merged"])):
            unres += 1
            continue
        notional = _num(r.get("notional_value"))
        notional = notional if np.isfinite(notional) and notional > 0 else float(W["N"])
        booked = _num(r.get("pnl_percentage"))
        v2c = WEra.is_v2_changed(r)                             # DL 356: booked the V2 runner / floor exit → not a parity case
        rows.append(dict(day=str(r["opened_at"])[:10], opened_at=str(r["opened_at"])[:19], pair=r["pair"], notional=notional, booked=booked,
                         **s, leg_usd=(s["leg"] * notional / 100 if s["trig"] else 0.0), merged_usd=s["merged"] * notional / 100,
                         v2c=v2c, sign_flag=bool(not v2c and np.isfinite(booked) and np.sign(booked) != np.sign(s["base"]))))
    d = pd.DataFrame(rows)
    print(f"  scored {len(d)} · pending {pend} (budget / cooldown / window not complete — re-run later) · unresolved path {unres}")
    if not len(d):
        print(f"  verdict: collecting (0/{GATE_N} on 0/{GATE_DAYS} days)")
        return
    dp = d[~d.v2c]                                              # DL 356: V2 runner / floor fills out of the parity read
    gap = (dp.base - dp.booked).abs()
    print(f"  parity walker-vs-booked: mean |Δ| {gap.mean():.2f} pp · sign disagreements {int(dp.sign_flag.sum())}/{len(dp)}"
          + (f" · {int(d.v2c.sum())} V2 runner / floor fill(s) out (booked the V2 exit)" if d.v2c.any() else ""))
    for x in d[d.sign_flag].itertuples():
        print(f"    ⚠ {x.opened_at} {x.pair}: booked {x.booked:+.2f} vs walker {x.base:+.2f} ({x.base_why})")
    t = d[d.trig]
    print(f"  shadow adds {len(t)} of {len(d)} fills on {t.day.nunique()} days")
    for x in t.itertuples():
        print(f"    {x.opened_at} {x.pair:12s} add @ min {x.add_min:4.0f} · original {x.base:+.2f} ({x.base_why}) → merged {x.merged:+.2f} "
              f"({x.merged_why}) · add leg {x.leg:+.2f} % / ${x.leg_usd:+.2f} · merged {x.merged_usd / ACCOUNT * 100:+.1f} % acct")
    if len(t):
        print(f"  add leg: mean {t.leg.mean():+.3f} % · Σ ${t.leg_usd.sum():+.2f} (haircut ×0.5 ${t.leg_usd.sum() * HAIRCUT:+.2f}) · "
              f"worst merged {(t.merged_usd / ACCOUNT * 100).min():+.1f} % acct vs original {(t.base * t.notional / 100 / ACCOUNT * 100).min():+.1f} %")
    v, det = verdict(t)
    if det:
        print(f"  gate: mean {det['mean']:+.3f} % · day-boot P(>0) {det['p']:.3f} (≥ {P_MIN}) · top pair {det['pair']} "
              f"{det['pair_share']:.0%} of the gain (< 50 %) · haircut ${det['haircut']:+.2f} (> 0) · worst {det['worst_acct']:+.1f} % acct (≥ −12)")
    print(f"  verdict: {v}")


if __name__ == "__main__":
    main()
