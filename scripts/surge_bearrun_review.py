#!/usr/bin/env python3
"""⚡🐻 SURGE (LONG / SHORT) + BEAR-RUN deep review (operator 2026-10-04: "re-evaluate deeply SURGE entries and exits, long and
short, or just longs maybe. Same with BEAR RUN").

Read-only research. Never touches the bot, its config or services/*; the bot's own pure functions are imported and CALLED
(services.surge.surge_trigger / surge_pair_pick, services.trading_engine._bullrun_exit_for / surge_short_exit_for) so the
replica runs the exact live rules with the live trading_config.json values.

Stages (each caches its output under reports/backtest_cache/surge_review/):
  live     live SURGE / BEARRUN fills (archives + Downloads exports, dedup (opened_at, pair, direction), MANUAL out, cohort floors)
           + SURGE trigger funnel from the decision journals
  events   year replica of the SURGE triggers (BTC 5m, k5m_full) → per trigger: universe (top-20 tradeable by rolling 24 h quote
           volume: COIN, non-Alpha, listed ≥ 90 d, global blacklist / no-trade / side blacklist skipped), picks (surge_pair_pick,
           rank order, ≤ 4), refused pairs (ATR_LOW / NOT_LEADER) and a paired CONTROL (same picks, same clock time 24 h earlier)
  needticks  pair-days the walk needs that are missing from ticks_q/ticks → surge_review/need_ticks.csv (fetch with
           scripts/backtest_fetch_ticks.py <csv> --qty)
  walk     1-second last-trade path from the aggTrade ticks; entry = first print ≥ window open + LATENCY; every exit variant judged
           on NET % (price % − 0.09 fees, the bot's accounting), filled at the CROSSING print (never at the stop level)
  bear     BEARRUN_SHORT fills from the engine replays (yr4 valid for Bear-Run, yr5 when present) — re-walked on ticks
  report   all tables → reports/SURGE_BEARRUN_REVIEW_2026-10-04_tables.md (+ csv)
Usage: venv/bin/python scripts/surge_bearrun_review.py <stage> [...]"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
OUT = os.path.join(CACHE, "surge_review"); os.makedirs(OUT, exist_ok=True)
K5 = os.path.join(CACHE, "k5m_full")
BAR, MIN, DAY = 300_000, 60_000, 86_400_000
FEE = 0.09
PER_SECOND = "--per-second" in sys.argv
LATENCY_S = int(os.environ.get("SURGE_LATENCY_S", "60"))          # bot opens ~30-120 s after the window opens (yr5 replay + live: SAND 14:55:48 for a 14:55 window)
HOLD_MIN = 240
COHORT_START = pd.Timestamp("2026-09-30 15:30")   # SURGE cohort floor (memory: feedback_surge_cohort_floor)
YEAR0 = 1767225600000   # 2026-01-01
SPLIT = 1777593600000   # 2026-05-01 (halves, same split as the design grids)
CFG = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
TH = CFG.get("thresholds", CFG)
# research override (operator Oct-4: "1 % seems late") — SURGE_MOVE=<pct> re-runs every stage on its own cache dir
if os.environ.get("SURGE_MOVE"):
    TH = dict(TH); TH["surge_btc_move_pct"] = float(os.environ["SURGE_MOVE"])
    OUT = os.path.join(CACHE, f"surge_review_m{os.environ['SURGE_MOVE']}"); os.makedirs(OUT, exist_ok=True)
# SURGE_HIGH_TOL=<pct>|off — LONG 24h-high rule relaxed to close ≥ prior-24h-high × (1 − tol %) or dropped (operator: "a high from minutes ago blocks it")
HIGH_TOL = os.environ.get("SURGE_HIGH_TOL")
if HIGH_TOL:
    OUT = os.path.join(CACHE, f"surge_review_m{TH['surge_btc_move_pct']}_h{HIGH_TOL}"); os.makedirs(OUT, exist_ok=True)
# SURGE_VOLX=<mult> — BTC bar volume ≥ this × median (operator: "with volume 5× we should flex the 1 % to 0.3 %")
if os.environ.get("SURGE_VOLX"):
    TH = dict(TH); TH["surge_btc_vol_mult"] = float(os.environ["SURGE_VOLX"])
    OUT = OUT.rstrip("/") + f"_v{os.environ['SURGE_VOLX']}"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"
ON_FILL = os.environ.get("SURGE_SPACING_ON_FILL") == "1"   # Oct-4: the cooldown starts only after a trigger that FILLED (18:10 no-pick lesson)
if ON_FILL:
    OUT = OUT.rstrip("/") + "_f"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"
if os.environ.get("SURGE_PAIR_COOLDOWN_H"):
    OUT = OUT.rstrip("/") + f"_p{os.environ['SURGE_PAIR_COOLDOWN_H']}"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"
if os.environ.get("SURGE_WINDOW_BARS"):
    OUT = OUT.rstrip("/") + f"_w{os.environ['SURGE_WINDOW_BARS']}"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"
if os.environ.get("SURGE_GVOL_PRE"):
    OUT = OUT.rstrip("/") + f"_g{os.environ['SURGE_GVOL_PRE']}"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"
# SURGE_SPACING=<hours> — cooldown between LONG triggers (operator: "4 h cooldown makes no sense while we go up like crazy")
if os.environ.get("SURGE_SPACING"):
    TH = dict(TH); TH["surge_trigger_spacing_hours"] = float(os.environ["SURGE_SPACING"])
    OUT = OUT.rstrip("/") + f"_s{os.environ['SURGE_SPACING']}"; os.makedirs(OUT, exist_ok=True)
    if not HIGH_TOL:
        HIGH_TOL = "0"   # spacing ≠ the bot's → the fidelity check must not drop triggers


def _set(s):
    return {x.strip().upper() for x in str(s or "").split(",") if x.strip()}


# ----------------------------------------------------------------------------------------------------------------- live
def live():
    """Every live SURGE / BEARRUN fill ever exported (archives + Downloads), dedup on (opened_at, pair, direction)."""
    fs = sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE*.csv*"))) + \
        sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"))) + \
        [os.path.join(ROOT, "reports", "SURGE_FIRST_TRIGGER_0930_bugged_orders.csv")]
    rows = []
    for f in fs:
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in d:
            continue
        s = d[d.entry_strategy.astype(str).isin(["SURGE_LONG", "SURGE_SHORT", "BEARRUN_SHORT"])].copy()
        if len(s):
            s["src"] = os.path.basename(f); s["_mt"] = os.path.getmtime(f); rows.append(s)
    if not rows:
        return pd.DataFrame()
    a = pd.concat(rows).sort_values("_mt").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[a.status.astype(str) == "CLOSED"].copy()
    a["cohort"] = np.where((a.entry_strategy.str.startswith("SURGE")) & (pd.to_datetime(a.opened_at) < COHORT_START),
                           "EXCLUDED (pre-floor, EMA13 first-tick bug)", "counted")
    a.to_csv(os.path.join(OUT, "live_fills.csv"), index=False)
    # journal funnel: SURGE refusals / opens per trigger minute
    J = []
    for f in sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))):
        try:
            d = pd.read_csv(f, usecols=["t", "e", "pair", "dir", "gate", "strategy"], low_memory=False)
        except Exception:
            continue
        J.append(d[(d.gate.astype(str).str.startswith("SURGE")) | (d.strategy.astype(str).str.startswith("SURGE"))])
    if J:
        j = pd.concat(J).drop_duplicates()
        j["minute"] = j.t.astype(str).str[:16]
        j.to_csv(os.path.join(OUT, "live_journal_surge.csv"), index=False)
    return a


# --------------------------------------------------------------------------------------------------------------- events
def _load5(p):
    d = pd.read_csv(os.path.join(K5, p + ".csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    d["qv"] = d.vol * d.c
    return d


def _universe_pairs():
    ex = json.load(open(os.path.join(CACHE, "exchange_info.json")))
    skip = _set(CFG.get("pair_blacklist")) | _set(CFG.get("no_trade_pairs"))
    out = []
    for f in glob.glob(os.path.join(K5, "*.csv")):
        p = os.path.basename(f)[:-4]
        info = ex.get(p) or {}
        if not p.endswith("USDT") or p in skip:
            continue
        if info and info.get("underlyingType") != "COIN":
            continue
        if any("alpha" in str(x).lower() for x in (info.get("underlyingSubType") or [])):
            continue
        out.append((p, info.get("onboardDate")))
    return out


CTRL_MODE = False
_GV_D = None


def _gvol_at(t):
    """market volume ratio at bar t (the bot's global_volume_ratio: top-50 by rolling 24 h quote volume, base volume vs its 48-bar mean)."""
    global _GV_D
    if _GV_D is None:   # the LIVE gate's universe: every COIN, non-Alpha USDT perp INCLUDING blacklisted / no-trade pairs and BTC / ETH,
        _GV_D = {}      # minus listings younger than new_listing_filter_days at the bar (get_top_futures_pairs) — validated vs the scout's live reads
        ex = json.load(open(os.path.join(CACHE, "exchange_info.json")))
        for f in glob.glob(os.path.join(K5, "*.csv")):
            p = os.path.basename(f)[:-4]; info = ex.get(p) or {}
            if not p.endswith("USDT") or (info and info.get("underlyingType") != "COIN") or \
                    any("alpha" in str(x).lower() for x in (info.get("underlyingSubType") or [])):
                continue
            try:
                d = _load5(p)
            except Exception:
                continue
            d["v24"] = d.qv.rolling(288, min_periods=250).sum(); d["m48"] = d.vol.rolling(48).mean()
            _GV_D[p] = (d[["vol", "v24", "m48"]], info.get("onboardDate"))
    nd = float(CFG.get("new_listing_filter_days", 90) or 0)
    snap = [(d.at[t, "v24"], d.at[t, "vol"], d.at[t, "m48"]) for d, ob in _GV_D.values()
            if t in d.index and not (nd > 0 and ob and ob > t - nd * DAY)]
    snap = [x for x in snap if np.isfinite(x[0]) and np.isfinite(x[2]) and x[2] > 0]
    top = sorted(snap, key=lambda x: -x[0])[:50]
    if len(top) < 30:
        return None
    return sum(x[1] for x in top) / sum(x[2] for x in top)


def events():
    import services.surge as S
    from types import SimpleNamespace
    th = SimpleNamespace(**{k: v for k, v in TH.items() if k.startswith("surge_")})
    btc = _load5("BTCUSDT")
    rows_btc = btc.reset_index()[["open_time", "o", "h", "l", "c", "vol"]].values
    c = btc.c.values; ts = btc.index.values
    r30 = np.full(len(c), np.nan); r30[6:] = (c[6:] / c[:-6] - 1) * 100
    hi = pd.Series(btc.h.values).shift(1).rolling(288).max().values
    lo = pd.Series(btc.l.values).shift(1).rolling(288).min().values
    med = pd.Series(btc.qv.values).shift(1).rolling(288).median().values
    vm = btc.qv.values / med
    trig = {}
    gpre = float(os.environ.get("SURGE_GVOL_PRE", "0") or 0)   # Oct-4: LONG market-volume gate applied BEFORE the spacing (refused ≠ cooldown)
    for side in ("LONG", "SHORT"):
        need = float(TH["surge_btc_move_pct"])
        ok = (r30 >= need) if side == "LONG" else (r30 <= -need)
        if side == "LONG" and TH.get("surge_long_require_24h_high", True) and HIGH_TOL != "off":
            if HIGH_TOL and HIGH_TOL.startswith("x"):   # x<min>: the 24 h high EXCLUDING the last <min> minutes (the current move's own highs)
                k = int(HIGH_TOL[1:]) // 5
                ok &= c >= pd.Series(btc.h.values).shift(k + 1).rolling(288 - k).max().values
            else:
                ok &= c >= hi * (1 - float(HIGH_TOL or 0) / 100)
        if side == "SHORT" and TH.get("surge_short_require_24h_low", False):
            ok &= c <= lo
        ok &= vm >= float(TH["surge_btc_vol_mult"])
        if side == "LONG" and gpre > 0:
            ok &= np.array([((_gvol_at(int(x)) or 0) >= gpre) if o else False for x, o in zip(ts, ok)])
        out, last = [], -10**18
        for i in np.where(ok)[0]:
            if ts[i] < YEAR0 or (not ON_FILL and ts[i] - last < float(TH["surge_trigger_spacing_hours"]) * 3600_000):
                continue   # ON_FILL: every qualifying bar is a candidate; the spacing is applied after the picks (only a FILL starts it)
            # fidelity: the bot's own surge_trigger on the same 310-bar view must agree
            view = rows_btc[i - 300:i + 2].tolist()   # ... trigger bar, forming bar
            if HIGH_TOL and side == "LONG":   # relaxed high rule = not the bot's rule → skip the fidelity check, compute the readings here
                out.append(dict(side=side, bar_ts=int(ts[i]), close_ts=int(ts[i]) + BAR, btc_move=float(r30[i]), btc_vol_mult=float(vm[i])))
                last = ts[i]
                continue
            t = S.surge_trigger(view, th, side)
            if t is None or int(t["bar_ts"]) != int(ts[i]):
                print(f"  ! trigger mismatch {side} {pd.to_datetime(ts[i], unit='ms')}: replica fires, surge_trigger={t}")
                continue
            out.append(dict(side=side, bar_ts=int(ts[i]), close_ts=int(ts[i]) + BAR, btc_move=float(t["btc_move_pct"]),
                            btc_vol_mult=t["btc_vol_mult"]))
            last = ts[i]
        trig[side] = out
        print(f"{side}: {len(out)} triggers {pd.to_datetime(out[0]['bar_ts'], unit='ms') if out else ''} → "
              f"{pd.to_datetime(out[-1]['bar_ts'], unit='ms') if out else ''}")
    # universe data
    U = _universe_pairs()
    D = {}
    t_need = sorted({e["bar_ts"] for s in trig.values() for e in s} | {e["bar_ts"] - DAY for s in trig.values() for e in s})
    for p, onboard in U:
        try:
            d = _load5(p)
        except Exception:
            continue
        d["v24"] = d.qv.rolling(288, min_periods=250).sum()
        D[p] = (d, onboard)
    print(f"universe candidates loaded: {len(D)}")
    newdays = float(CFG.get("new_listing_filter_days", 90) or 0)
    th_full = SimpleNamespace(**{k: v for k, v in TH.items() if k.startswith("surge_")})

    def ranked(t, side):
        bl = _set(TH.get(f"surge_{side.lower()}_pair_blacklist"))
        vols = []
        for p, (d, onboard) in D.items():
            if p in bl or t not in d.index:
                continue
            if newdays > 0 and onboard and onboard > t - newdays * DAY:
                continue
            v = d.v24.get(t)
            if v is None or not np.isfinite(v):
                continue
            vols.append((p, v))
        return [p for p, _ in sorted(vols, key=lambda x: -x[1])[: int(TH["surge_universe_size"])]]

    def bars(p, t, n=100):
        d = D[p][0]
        i = d.index.searchsorted(t, side="right")
        w = d.iloc[max(0, i - n):i]
        return [[int(a), o, h, l, cc, v] for a, o, h, l, cc, v in zip(w.index, w.o, w.h, w.l, w.c, w.vol)]

    if CTRL_MODE:
        return ranked, bars, th_full, btc
    rows = []
    WB = int(os.environ.get("SURGE_WINDOW_BARS", "0") or 0)   # Oct-4 operator: "5 minutes alone is wrong" — stay active WB more bars
    r30_at = dict(zip(ts.tolist(), r30.tolist()))
    PCD = float(os.environ.get("SURGE_PAIR_COOLDOWN_H", "0") or 0) * 3600_000   # Oct-5 operator: "pairs are independent" — per-pair cooldown
    for side, evs in trig.items():
        last_fill = -10**18
        pair_last = {}
        for e in evs:
            if ON_FILL and e["bar_ts"] - last_fill < float(TH["surge_trigger_spacing_hours"]) * 3600_000:
                continue
            uni = ranked(e["bar_ts"], side)
            n_open = 0
            picked = set()
            for rank, p in enumerate(uni, 1):
                if PCD and e["bar_ts"] - pair_last.get(p, -10**18) < PCD:
                    continue   # this pair was bought < cooldown ago (per-pair cooldown); other pairs stay eligible
                ok, why, atr, pm = S.surge_pair_pick(bars(p, e["bar_ts"]), e["bar_ts"], e["btc_move"], th_full, side)
                status = "PICK" if ok else why
                if ok:
                    if n_open >= int(TH["surge_max_slots"]):
                        status = "SURGE_MAX_SLOTS"
                    else:
                        n_open += 1
                if status == "PICK":
                    picked.add(p); pair_last[p] = e["bar_ts"]
                rows.append(dict(side=side, trig=e["bar_ts"], btc_move=e["btc_move"], btc_vol_mult=e["btc_vol_mult"], pair=p,
                                 rank=rank, status=status, atr=atr, pair_move=pm, entry_bar=e["bar_ts"]))
            # later bars of the window: pairs not yet picked, judged on THAT bar (ATR, own 30-min move vs BTC's 30-min move at that bar)
            for k in range(1, WB + 1):
                bt = int(e["bar_ts"]) + k * BAR
                bm = r30_at.get(bt)
                if bm is None or not np.isfinite(bm) or n_open >= int(TH["surge_max_slots"]):
                    break
                for rank, p in enumerate(ranked(bt, side), 1):
                    if p in picked or n_open >= int(TH["surge_max_slots"]):
                        continue
                    ok, why, atr, pm = S.surge_pair_pick(bars(p, bt), bt, bm, th_full, side)
                    if ok:
                        n_open += 1; picked.add(p)
                        rows.append(dict(side=side, trig=e["bar_ts"], btc_move=e["btc_move"], btc_vol_mult=e["btc_vol_mult"], pair=p,
                                         rank=rank, status="PICK", atr=atr, pair_move=pm, entry_bar=bt))
            if picked:
                last_fill = e["bar_ts"]
    F = pd.DataFrame(rows)
    F["trig_at"] = pd.to_datetime(F.trig, unit="ms")
    F.to_csv(os.path.join(OUT, "events_picks.csv"), index=False)
    print(F.groupby(["side", "status"]).size())
    return F


def ctrl():
    """No-trigger CONTROL with FRESH selection (operator lesson: a control must not reuse pairs picked with hindsight): the same
    clock time 2 and 1 days before and 1 and 2 days after each trigger, universe + surge_pair_pick re-run at that moment with
    BTC's own 30-min move then (no trigger), ≤ 4 picks. Skips control moments within 4 h of any real trigger of the same side."""
    global CTRL_MODE
    import services.surge as S
    CTRL_MODE = True
    ranked, bars, th_full, btc = events()
    CTRL_MODE = False
    F = pd.read_csv(os.path.join(OUT, "events_picks.csv"))
    c = btc.c
    rows = []
    for side in ("LONG", "SHORT"):
        trigs = np.array(sorted(F[F.side == side].trig.unique()))
        for t0 in trigs:
            for off in (-2, -1, 1, 2):
                t = int(t0 + off * DAY)
                if np.abs(trigs - t).min() < 4 * 3600_000 or t not in c.index or t > c.index[-1] - 5 * 3600_000:
                    continue
                i = c.index.get_loc(t)
                bm = (c.iloc[i] / c.iloc[i - 6] - 1) * 100
                n = 0
                for rank, p in enumerate(ranked(t, side), 1):
                    ok, why, atr, pm = S.surge_pair_pick(bars(p, t), t, bm, th_full, side)
                    if ok and n < int(TH["surge_max_slots"]):
                        n += 1
                        rows.append(dict(side=side, trig=t0, ctrl_t=t, off=off, btc_move=bm, pair=p, rank=rank, atr=atr, pair_move=pm))
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "ctrl_picks.csv"), index=False)
    print(C.groupby(["side", "off"]).size())


# ------------------------------------------------------------------------------------------------------------ fill list
def entry_ms(side, trig_bar_ts):
    delay = float(TH["surge_long_entry_delay_min"] if side == "LONG" else TH["surge_short_entry_delay_min"])
    return int(trig_bar_ts) + BAR + int(delay * MIN) + LATENCY_S * 1000


def fill_list(include_refused=True):
    """Fills to walk: PICK (the live rule), CONTROL (same pair, 24 h earlier), REFUSED (ATR_LOW / NOT_LEADER / MAX_SLOTS)."""
    F = pd.read_csv(os.path.join(OUT, "events_picks.csv"))
    rows = []
    for r in F.itertuples():
        _eb = getattr(r, "entry_bar", None)
        t = entry_ms(r.side, int(_eb) if _eb is not None and pd.notna(_eb) else r.trig)
        if r.status == "PICK":
            rows.append(dict(kind="PICK", side=r.side, trig=r.trig, pair=r.pair, t_entry=t, atr=r.atr, rank=r.rank, pair_move=r.pair_move))
            rows.append(dict(kind="CONTROL_SAMEPAIR", side=r.side, trig=r.trig, pair=r.pair, t_entry=t - DAY, atr=r.atr, rank=r.rank,
                             pair_move=r.pair_move))   # biased (pair chosen with 24 h of hindsight) — kept only to show the bias
        elif include_refused:
            rows.append(dict(kind="REFUSED_" + str(r.status).replace("SURGE_", ""), side=r.side, trig=r.trig, pair=r.pair, t_entry=t,
                             atr=r.atr, rank=r.rank, pair_move=r.pair_move))
    cf = os.path.join(OUT, "ctrl_picks.csv")
    if os.path.exists(cf):
        for r in pd.read_csv(cf).itertuples():
            rows.append(dict(kind="CTRL", side=r.side, trig=r.trig, pair=r.pair, t_entry=entry_ms(r.side, r.ctrl_t), atr=r.atr,
                             rank=r.rank, pair_move=r.pair_move, ctrl_off=r.off))
    return pd.DataFrame(rows)


def _tick_path(pair, date):
    for d in ("ticks_q", "ticks"):
        f = os.path.join(CACHE, d, pair, f"{date}.npz")
        if os.path.exists(f):
            return f
    return None


def needticks():
    L = fill_list()
    need = set()
    for r in L.itertuples():
        for t in (r.t_entry, r.t_entry + HOLD_MIN * MIN):
            need.add((r.pair, pd.to_datetime(t, unit="ms").strftime("%Y-%m-%d")))
    miss = sorted(x for x in need if _tick_path(*x) is None and x[1] < "2026-10-04")
    kinds = L.assign(d=pd.to_datetime(L.t_entry, unit="ms").dt.strftime("%Y-%m-%d"))
    pd.DataFrame(miss, columns=["pair", "date"]).to_csv(os.path.join(OUT, "need_ticks.csv"), index=False)
    pk = {(r.pair, r.d) for r in kinds[kinds.kind.isin(["PICK", "CONTROL"])].itertuples()}
    print(f"pair-days needed {len(need)} · missing {len(miss)} (of which PICK/CONTROL {len(set(miss) & pk)})")



# ----------------------------------------------------------------------------------------------------------------- walk
_TICKS = {}


def _ticks(pair, date):
    k = (pair, date)
    if k not in _TICKS:
        if len(_TICKS) > 40:
            _TICKS.clear()
        f = _tick_path(pair, date)
        if f is None:
            _TICKS[k] = None
        else:
            z = np.load(f)
            _TICKS[k] = (z["t"], z["p"].astype(np.float64))
    return _TICKS[k]


def sec_path(pair, t0, t1):
    """price path (every price-changing print; --per-second = last trade per second) on (t0, t1]: (sec_ms array, price array) + the entry price (last trade ≤ t0, else first after)."""
    parts = []
    for d in sorted({pd.to_datetime(x, unit="ms").strftime("%Y-%m-%d") for x in (t0 - MIN, t0, t1)}):
        z = _ticks(pair, d)
        if z is not None:
            parts.append(z)
    if not parts:
        return None
    t = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
    o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    k = np.searchsorted(t, t0, side="right")
    if k > 0 and t0 - t[k - 1] <= 60_000:
        e = p[k - 1]
    elif k < len(t) and t[k] - t0 <= 60_000:
        e = p[k]
    else:
        return None
    m = (t > t0) & (t <= t1)
    tt, pp = t[m], p[m]
    if len(tt) < 10:
        return None
    if PER_SECOND:
        sec = tt // 1000
        last = np.r_[np.diff(sec) != 0, True]      # sensitivity only: last trade of each second
        return sec[last] * 1000, pp[last], e
    keep = np.r_[True, np.diff(pp) != 0]           # every PRINT that changes the price (the WS tracker sees each trade)
    return tt[keep], pp[keep], e


def _lines(fn, peak):
    """stop line per sample from a peak-only stop function, evaluated once per distinct peak (exact for the bot functions)."""
    u, inv = np.unique(peak, return_inverse=True)
    return np.array([fn(x) for x in u])[inv]


def run_exit(ts, pnl, t_entry, stop_fn, tp=None, hold=HOLD_MIN):
    """→ (net %, exit reason, minutes held, peak). Exit at the first sample whose NET P&L ≤ the stop line (filled at that print),
    or ≥ tp; else at the last print before the hold limit."""
    m = ts <= t_entry + hold * MIN
    ts, pnl = ts[m], pnl[m]
    if not len(pnl):
        return None
    peak = np.maximum.accumulate(np.maximum(pnl, 0.0))   # the bot's peak starts at 0 (peak_pnl defaults 0)
    line = _lines(stop_fn, peak)
    hit = pnl <= line
    if tp is not None:
        hit |= pnl >= tp
    j = int(np.argmax(hit)) if hit.any() else None
    if j is None:
        return float(pnl[-1]), "TIME", (ts[-1] - t_entry) / MIN, float(peak[-1])
    why = "TP" if (tp is not None and pnl[j] >= tp) else ("STOP" if line[j] < 0 else "TRAIL")
    return float(pnl[j]), why, (ts[j] - t_entry) / MIN, float(peak[j])


def variants(side, atr):
    import services.trading_engine as TE
    a = float(atr if atr and atr > 0 else TH["surge_atr_min_pct"])
    sl_live = min(-0.70, max(-1.5 * a, -1.2))
    V = {}
    def flat(sl):
        return lambda pk: sl
    if side == "LONG":
        V["LIVE"] = (lambda pk: TE._bullrun_exit_for(float("inf"), pk, a, trail_mult_override=1.0)[2], None, HOLD_MIN)
        for m in (0.5, 2.0):
            V[f"BR_TRAIL_{m}ATR"] = (lambda pk, m=m: TE._bullrun_exit_for(float("inf"), pk, a, trail_mult_override=m)[2], None, HOLD_MIN)
        def br(pk, arm=1.0, lock=0.2, mult=1.0, sl=sl_live):
            return max(lock, pk - mult * a) if pk >= arm else sl
        V["BR_ARM0.5_LOCK0.1"] = (lambda pk: br(pk, 0.5, 0.1), None, HOLD_MIN)
        V["BR_ARM2.0"] = (lambda pk: br(pk, 2.0, 0.2), None, HOLD_MIN)
        V["BR_NO_BE_LOCK"] = (lambda pk: max(sl_live, pk - a) if pk >= 1.0 else sl_live, None, HOLD_MIN)
        V["MOM_LONG_RUNNER"] = (lambda pk: max(pk - a, 0.10) if pk >= 0.40 else sl_live, None, HOLD_MIN)
        for sl in (-0.7, -2.0, -3.0, -5.0):   # the REAL exit (trail + lock + ladder) with only the pre-arm stop moved
            V[f"LIVE_SL{sl}"] = (lambda pk, sl=sl: TE._bullrun_exit_for(float("inf"), pk, a, trail_mult_override=1.0)[2]
                                 if pk >= float(TH.get("bullrun_be_arm_pct", 1.0)) else sl, None, HOLD_MIN)
    else:
        V["LIVE"] = (lambda pk: TE.surge_short_exit_for(float("inf"), pk, a)[2], None, HOLD_MIN)
        V["QUICK_DESIGN"] = (lambda pk: max(0.5 * pk, 0.10) if pk >= 0.30 else -0.50, None, 30)
        def st(pk, arm=0.40, frac=0.35, n=0.5, sl=sl_live):
            if pk < arm - 0.005:
                return sl
            gb = n * a
            if frac and pk > 0 and frac * pk < gb:
                gb = frac * pk
            return max(pk - gb, 0.0) if pk - gb >= 0 else sl
        V["ST_ARM0.8"] = (lambda pk: st(pk, arm=0.8), None, HOLD_MIN)
        V["ST_ARM1.2"] = (lambda pk: st(pk, arm=1.2), None, HOLD_MIN)
        V["ST_FRAC0.5"] = (lambda pk: st(pk, frac=0.5), None, HOLD_MIN)
        V["ST_TRAIL_0.5ATR"] = (lambda pk: st(pk, frac=0), None, HOLD_MIN)
        V["BR_MIRROR_1ATR"] = (lambda pk: max(0.2, pk - a) if pk >= 1.0 else sl_live, None, HOLD_MIN)
        for sl in (-0.7, -2.0, -3.0, -5.0):
            V[f"LIVE_SL{sl}"] = (lambda pk, sl=sl: TE.surge_short_exit_for(float("inf"), pk, a)[2] if pk >= 0.395 or pk >= 1.0 else sl, None, HOLD_MIN)
    base = V["LIVE"][0]
    for h in (15, 30, 60, 120):
        V[f"LIVE_HOLD{h}"] = (base, None, h)
    for tp in (0.5, 1.0, 2.0, 3.0, 5.0):
        for sl in (-0.7, -1.2, -2.0, -3.0):
            V[f"FIX_TP{tp}_SL{sl}"] = (flat(sl), tp, HOLD_MIN)
    V["HOLD_ONLY_240"] = (flat(-1e9), None, HOLD_MIN)
    return V


def walk(kinds=("PICK", "CONTROL_SAMEPAIR", "CTRL"), live_only_kinds=("REFUSED_ATR_LOW", "REFUSED_NOT_LEADER", "REFUSED_MAX_SLOTS")):
    L = fill_list()
    L = L[L.kind.isin(list(kinds) + list(live_only_kinds))]
    res = []
    for n, r in enumerate(L.itertuples()):
        sp = sec_path(r.pair, r.t_entry, r.t_entry + HOLD_MIN * MIN + 5000)
        if sp is None:
            continue
        ts, px, e = sp
        pnl = ((px / e - 1) * 100 if r.side == "LONG" else (e - px) / e * 100) - FEE
        V = variants(r.side, r.atr)
        if r.kind in live_only_kinds:
            V = {k: V[k] for k in ("LIVE", "HOLD_ONLY_240")}
        rec = dict(kind=r.kind, side=r.side, trig=r.trig, pair=r.pair, t_entry=r.t_entry, atr=r.atr, rank=r.rank,
                   pair_move=r.pair_move, entry_px=e, path_max=float(pnl.max()), path_min=float(pnl.min()))
        for name, (fn, tp, hold) in V.items():
            o = run_exit(ts, pnl, r.t_entry, fn, tp, hold)
            if o is None:
                continue
            rec[name] = o[0]
            if name == "LIVE":
                rec["LIVE_why"], rec["LIVE_min"], rec["LIVE_peak"] = o[1], o[2], o[3]
        res.append(rec)
        if n % 200 == 0:
            print(f"  {n}/{len(L)}", flush=True)
    W = pd.DataFrame(res)
    W.to_csv(os.path.join(OUT, f"walk{'_1s' if PER_SECOND else ''}{'' if LATENCY_S == 60 else f'_lat{LATENCY_S}'}.csv"), index=False)
    print(W.groupby(["side", "kind"]).LIVE.agg(["size", "mean", lambda x: (x > 0).mean()]))
    return W



# ------------------------------------------------------------------------------------------------------------- features
def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def features():
    """Regime (per trigger) + pair-level (per fill) features, all on CLOSED bars at the trigger bar (no look-ahead)."""
    W = pd.read_csv(os.path.join(OUT, "walk.csv"))
    btc = _load5("BTCUSDT")
    h1 = btc.c.groupby(btc.index // 3600_000).last()                     # 1h closes (bar labelled by hour start)
    e20h, e50h = pd.Series(_ema(h1.values, 20), h1.index), pd.Series(_ema(h1.values, 50), h1.index)
    c = btc.c; r5 = c.pct_change().abs()
    btc_atr = (pd.concat([btc.h - btc.l, (btc.h - c.shift()).abs(), (btc.l - c.shift()).abs()], axis=1).max(axis=1)
               .ewm(alpha=1 / 14, adjust=False).mean() / c * 100)
    trig = sorted(W.trig.unique())
    T = []
    # breadth: share of that moment's top-50 COIN universe whose 5m close > EMA20
    U = dict(_universe_pairs())
    D = {}
    for p in U:
        try:
            d = _load5(p)
        except Exception:
            continue
        d["v24"] = d.qv.rolling(288, min_periods=250).sum(); d["e20"] = _ema(d.c.values, 20)
        D[p] = d[["c", "v24", "e20"]]
    for t in trig:
        i = c.index.get_loc(t)
        hk = t // 3600_000 - 1                                             # last CLOSED 1h bar
        r24 = (c.iloc[i] / c.iloc[i - 288] - 1) * 100
        r72 = (c.iloc[i] / c.iloc[i - 864] - 1) * 100 if i >= 864 else np.nan
        eff24 = abs(np.log(c.iloc[i] / c.iloc[i - 288])) / np.log(1 + r5.iloc[i - 287:i + 1]).sum() if i >= 288 else np.nan
        hi30 = btc.h.iloc[max(0, i - 8640):i + 1].max()
        snap = [(p, d.at[t, "v24"], d.at[t, "c"] > d.at[t, "e20"]) for p, d in D.items() if t in d.index and np.isfinite(d.at[t, "v24"])]
        top = sorted(snap, key=lambda x: -x[1])[:50]
        T.append(dict(trig=t, btc_1h_slope=(e20h.get(hk) / e20h.get(hk - 1) - 1) * 100 if hk - 1 in e20h.index else np.nan,
                      btc_above_1h_e50=(h1.get(hk) - e50h.get(hk)) / e50h.get(hk) * 100 if hk in e50h.index else np.nan,
                      btc_r24=r24, btc_r72=r72, btc_eff24=eff24, btc_atr=btc_atr.iloc[i], btc_off30d_hi=(c.iloc[i] / hi30 - 1) * 100,
                      breadth_top50=np.mean([x[2] for x in top]) * 100 if top else np.nan,
                      hour=pd.to_datetime(t, unit="ms").hour, weekend=int(pd.to_datetime(t, unit="ms").weekday() >= 5)))
    T = pd.DataFrame(T)
    P = []
    for r in W[["pair", "trig"]].drop_duplicates().itertuples():
        d = D.get(r.pair)
        if d is None:
            try:
                d = _load5(r.pair)
            except Exception:
                continue
        full = _load5(r.pair) if r.pair not in D else None
        dd = full if full is not None else _load5(r.pair)
        i = dd.index.searchsorted(r.trig, side="right")
        w = dd.iloc[max(0, i - 400):i]
        if len(w) < 60:
            continue
        cc = w.c.values
        e5, e8, e13, e20, e50 = (_ema(cc, n)[-1] for n in (5, 8, 13, 20, 50))
        up = e5 > e8 > e13 > e20; dn = e5 < e8 < e13 < e20
        P.append(dict(pair=r.pair, trig=r.trig, pair_stack=1 if up else (-1 if dn else 0), pair_vs_e50=(cc[-1] / e50 - 1) * 100,
                      pair_r24=(cc[-1] / cc[-289] - 1) * 100 if len(cc) > 289 else np.nan,
                      pair_vol_ratio=w.vol.values[-1] / w.vol.values[-21:-1].mean() if w.vol.values[-21:-1].mean() > 0 else np.nan,
                      pair_r4h=(cc[-1] / cc[-49] - 1) * 100))
    P = pd.DataFrame(P)
    T.to_csv(os.path.join(OUT, "features_trigger.csv"), index=False); P.to_csv(os.path.join(OUT, "features_pair.csv"), index=False)
    print(T.describe().T[["count", "mean", "50%"]])



# ----------------------------------------------------------------------------------------------------------------- bear
REPLAY = os.path.join(CACHE, "replay")


def replay_fills(strategies, prefixes=("yr4", "yr5")):
    """Engine-replay fills (yr4 = Bear-Run valid / SURGE broken (B2); yr5 = corrected) with run / seed / chunk tags."""
    rows = []
    for pre in prefixes:
        for f in sorted(glob.glob(os.path.join(REPLAY, f"{pre}_*_orders.csv"))):
            d = pd.read_csv(f, low_memory=False)
            d = d[d.entry_strategy.isin(strategies)].copy()
            if not len(d):
                continue
            b = os.path.basename(f)
            d["run"] = pre; d["seed"] = b.split("_")[2]; d["chunk"] = b.split("_")[1]; rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bear_windows(prefixes=("yr4", "yr5")):
    """Bear monitor periods (union over seeds/chunks), merged like the ledger (≤ 180 min apart = one window)."""
    P = []
    for pre in prefixes:
        for f in glob.glob(os.path.join(REPLAY, f"{pre}_*_bear_periods.csv")):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if len(d):
                P.append(d[["started_at", "ended_at"]].assign(run=pre))
    P = pd.concat(P)
    P["s"] = pd.to_datetime(P.started_at); P["e"] = pd.to_datetime(P.ended_at).fillna(P.s + pd.Timedelta("6h"))
    P = P.sort_values("s")
    win, cur = [], None
    for r in P.itertuples():
        if cur is None or r.s > cur[1] + pd.Timedelta(minutes=float(TH.get("bearrun_window_merge_minutes", 180))):
            if cur:
                win.append(cur)
            cur = [r.s, r.e]
        else:
            cur[1] = max(cur[1], r.e)
    if cur:
        win.append(cur)
    return pd.DataFrame(win, columns=["w_start", "w_end"])


def bear():
    B = replay_fills(["BEARRUN_SHORT"])
    B = B[B.status == "CLOSED"].copy()
    B["t"] = pd.to_datetime(B.opened_at)
    Wn = bear_windows()
    B["window"] = [Wn[(Wn.w_start - pd.Timedelta("10min") <= t) & (t <= Wn.w_end + pd.Timedelta("4h"))].w_start.astype(str).min()
                   for t in B.t]
    B["exit_family"] = B.close_reason.astype(str).str.replace(r" L\d+", "", regex=True)
    # tick re-walk of every fill under the alternative exits (entry = the replay's own entry price/time)
    res = []
    for r in B.itertuples():
        t0 = int(r.t.value // 10**6)
        sp = sec_path(r.pair, t0, t0 + HOLD_MIN * MIN + 5000)
        if sp is None:
            res.append({}); continue
        ts, px, _ = sp
        e = float(r.entry_price)
        pnl = (e - px) / e * 100 - FEE
        V = variants("SHORT", float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else None)
        rec = {}
        for name in ("LIVE", "ST_ARM0.8", "ST_TRAIL_0.5ATR", "BR_MIRROR_1ATR", "LIVE_SL-0.7", "LIVE_SL-2.0", "LIVE_HOLD30",
                     "LIVE_HOLD60", "FIX_TP1.0_SL-1.2", "FIX_TP2.0_SL-1.2", "FIX_TP3.0_SL-2.0", "HOLD_ONLY_240"):
            fn, tp, hold = V[name]
            o = run_exit(ts, pnl, t0, fn, tp, hold)
            if o:
                rec["rw_" + ("NO_EMA13_STACK" if name == "LIVE" else name)] = o[0]
                if name == "LIVE":
                    rec["rw_why"] = o[1]
        res.append(rec)
    B = pd.concat([B.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    keep = [c for c in B.columns if c in ("run", "seed", "chunk", "pair", "opened_at", "closed_at", "entry_price", "exit_price", "pnl_percentage",
                                         "close_reason", "exit_family", "peak_pnl", "entry_atr_pct", "window") or c.startswith(("rw_", "entry_bear", "entry_btc", "entry_pair", "entry_rsi", "entry_adx", "entry_bull", "entry_global", "entry_gap", "entry_ema"))]
    B[keep].to_csv(os.path.join(OUT, "bear_fills.csv"), index=False)
    Wn.to_csv(os.path.join(OUT, "bear_windows.csv"), index=False)
    print(B.groupby(["run", "seed"]).pnl_percentage.agg(["size", "mean"]))
    print(B.groupby("exit_family").agg(n=("pnl_percentage", "size"), live=("pnl_percentage", "mean"), rw=("rw_NO_EMA13_STACK", "mean")))
    return B


def candle():
    """Why the design grid (+0.32 LONG / +0.22 SHORT per event) and this tick replica disagree: re-walk the SAME PICK fills with
    the design's simulator — 1m candles built from the same ticks, candle path (falling O→H→L→C, rising O→L→H→C), entry at the
    1m open, stop filled AT the stop level — and with the tick walker but entry at the window open (no latency)."""
    import services.trading_engine as TE
    W = pd.read_csv(os.path.join(OUT, "walk.csv"))
    W = W[W.kind == "PICK"]
    out = []
    for r in W.itertuples():
        t_open = r.t_entry - LATENCY_S * 1000
        sp = sec_path(r.pair, t_open, t_open + HOLD_MIN * MIN + 5000)
        if sp is None:
            continue
        ts, px, e0 = sp
        side = r.side
        fn = variants(side, r.atr)["LIVE"][0]
        sgn = 1 if side == "LONG" else -1
        # (a) tick walker, zero latency
        pnl0 = ((px / e0 - 1) * 100 if side == "LONG" else (e0 - px) / e0 * 100) - FEE
        a = run_exit(ts, pnl0, t_open, fn)
        # (b) design candle path on 1m bars from the same ticks
        m = ts // MIN
        df = pd.DataFrame({"m": m, "p": px}).groupby("m").p.agg(["first", "max", "min", "last"])
        e = float(df["first"].iloc[0]); peak = 0.0; res = None
        for o, h, l, c in df.values[: HOLD_MIN]:
            path = (o, h, l, c) if c < o else (o, l, h, c)
            for x in path:
                v = ((x / e - 1) * 100 if side == "LONG" else (e - x) / e * 100) - FEE
                stop = fn(peak)
                if v <= stop:
                    res = stop; break
                peak = max(peak, v)
            if res is not None:
                break
        if res is None:
            c = df["last"].values[min(HOLD_MIN, len(df)) - 1]
            res = ((c / e - 1) * 100 if side == "LONG" else (e - c) / e * 100) - FEE
        out.append(dict(side=side, trig=r.trig, pair=r.pair, tick_lat60=r.LIVE, tick_lat0=a[0] if a else np.nan, candle_1m=res))
    C = pd.DataFrame(out)
    C.to_csv(os.path.join(OUT, "candle_vs_tick.csv"), index=False)
    for side, g in C.groupby("side"):
        print(side, {k: round(g.groupby("trig")[k].mean().mean(), 3) for k in ("tick_lat60", "tick_lat0", "candle_1m")},
              "fill means", {k: round(g[k].mean(), 3) for k in ("tick_lat60", "tick_lat0", "candle_1m")})
    return C


# --------------------------------------------------------------------------------------------------------------- stats
TICKET = 922.24 * 20.0      # live SURGE ticket at 1×/1× (SAND 10-02: investment 922.24 × lev 20) → $ per 1 % = TICKET / 100


def boot_ci(x, n=5000, seed=7):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ev_line(g, col="LIVE"):
    """window units: one trigger = one observation (mean of its fills)."""
    g = g[np.isfinite(g[col])]
    e = g.groupby("trig")[col].mean()
    lo, hi = boot_ci(e.values)
    a, b = e[e.index < SPLIT], e[e.index >= SPLIT]
    top = e.sort_values(ascending=False)
    top3 = top.head(3).sum() / e.sum() if e.sum() > 0 else np.nan
    return dict(fills=len(g), triggers=len(e), wr=round((g[col] > 0).mean() * 100), avg_fill=round(g[col].mean(), 3),
                per_trig=round(e.mean(), 3), ci=f"[{lo:+.2f}, {hi:+.2f}]", ci_lo=lo, ci_hi=hi,
                jan_apr=round(a.mean(), 3) if len(a) else np.nan, may_oct=round(b.mean(), 3) if len(b) else np.nan,
                n_a=len(a), n_b=len(b), trig_pos=round((e > 0).mean() * 100), top3_share=round(top3, 2) if top3 == top3 else None,
                usd=round(g[col].sum() * TICKET / 100))


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{floatfmt}f}" if abs(v) < 100 else f"{v:,.0f}")
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(out)


def load_walk():
    W = pd.read_csv(os.path.join(OUT, "walk.csv"))
    T = pd.read_csv(os.path.join(OUT, "features_trigger.csv")); P = pd.read_csv(os.path.join(OUT, "features_pair.csv"))
    E = pd.read_csv(os.path.join(OUT, "events_picks.csv")).groupby(["side", "trig"])[["btc_move", "btc_vol_mult"]].first().reset_index()
    E["btc_move"] = E.btc_move.abs()   # magnitude of the trigger move (sign is the side)
    W = W.merge(T, on="trig", how="left").merge(P, on=["pair", "trig"], how="left").merge(E, on=["side", "trig"], how="left")
    W["month"] = pd.to_datetime(W.trig, unit="ms").dt.strftime("%Y-%m")
    W["half"] = np.where(W.trig < SPLIT, "Jan-Apr", "May-Oct")
    return W


def robust():
    """Screen-survivor checks (no arm before review): shuffled null for the WHOLE macro sweep (max |Δ| over every var ×
    granularity, outcomes permuted across triggers), dose-response and leave-one-month-out for the best LONG / SHORT cuts, and the
    tape context per month (BTC return / realized vol / trend) for checklist ④."""
    W = load_walk(); L = []
    for side, var in (("LONG", "btc_r72"), ("SHORT", "btc_1h_slope")):
        pk = W[(W.side == side) & (W.kind == "PICK")]
        e = pk.groupby("trig").agg(y=("LIVE", "mean"), **{v: (v, "first") for v in MACRO})
        def maxd(y):
            best = 0.0
            for v in MACRO:
                x = e[v]
                if x.nunique() < 3:
                    continue
                cuts = [x.median()] + ([0.0] if v in SIGNED else [])
                for c in cuts:
                    a, b = y[x > c], y[x <= c]
                    if len(a) >= 5 and len(b) >= 5:
                        best = max(best, abs(a.mean() - b.mean()))
                q1, q2 = x.quantile([1 / 3, 2 / 3]); a, b = y[x >= q2], y[x <= q1]
                if len(a) >= 5 and len(b) >= 5:
                    best = max(best, abs(a.mean() - b.mean()))
            return best
        obs = maxd(e.y)
        rng = np.random.default_rng(11)
        null = [maxd(pd.Series(rng.permutation(e.y.values), index=e.index)) for _ in range(500)]
        L.append(f"{side}: largest macro split |Δ| = {obs:.3f} per trigger; shuffled-null p (max over all macro splits) = "
                 f"{np.mean(np.array(null) >= obs):.3f} (500 permutations, {len(e)} triggers)")
        med = pk[var].median()
        good = pk[pk[var] <= med] if side == "LONG" else pk[pk[var] > 0]
        L.append(f"  best cut {var} {'≤ median ' + format(med, '+.2f') if side == 'LONG' else '> 0'}: per trigger "
                 f"{good.groupby('trig').LIVE.mean().mean():+.3f} over {good.trig.nunique()} triggers")
        lomo = []
        for m in sorted(good.month.unique()):
            x = good[good.month != m].groupby("trig").LIVE.mean()
            lomo.append(f"{m}:{x.mean():+.2f}")
        L.append("  leave-one-month-out: " + " ".join(lomo))
    # tape context
    btc = _load5("BTCUSDT"); c = btc.c
    m = pd.to_datetime(c.index, unit="ms").strftime("%Y-%m")
    rows = []
    pkL = W[(W.side == "LONG") & (W.kind == "PICK")]; pkS = W[(W.side == "SHORT") & (W.kind == "PICK")]
    for mm, x in c.groupby(m):
        if mm < "2026-01":
            continue
        r = x.pct_change()
        rows.append(dict(month=mm, btc_ret=round((x.iloc[-1] / x.iloc[0] - 1) * 100, 1), btc_rvol_5m=round(r.std() * 100, 3),
                         long_trig=pkL[pkL.month == mm].trig.nunique(), long_per_trig=round(pkL[pkL.month == mm].groupby("trig").LIVE.mean().mean(), 2),
                         short_trig=pkS[pkS.month == mm].trig.nunique(), short_per_trig=round(pkS[pkS.month == mm].groupby("trig").LIVE.mean().mean(), 2)))
    T = pd.DataFrame(rows)
    L.append("\n" + md(T))
    txt = "\n".join(L)
    open(os.path.join(OUT, "robust.txt"), "w").write(txt)
    print(txt)


def bear_sweep():
    """Sleeve-kill checklist ①② for BEARRUN_SHORT on the replay fills: every numeric entry_* stamp at sign (if signed) / median /
    outer terciles; replicate-window units (a window's fills averaged per replicate, then over replicates)."""
    B = pd.read_csv(os.path.join(OUT, "bear_fills.csv")); B["rep"] = B.run + "_" + B.seed
    cols = [c for c in B.columns if c.startswith("entry_") and pd.api.types.is_numeric_dtype(B[c]) and B[c].notna().mean() >= 0.8 and B[c].nunique() > 3]
    rows = []
    for c in cols:
        x = B[c]; grans = []
        if (x < 0).any() and (x > 0).any():
            grans.append(("sign", x > 0, x <= 0))
        grans.append(("median", x > x.median(), x <= x.median()))
        q1, q2 = x.quantile([1 / 3, 2 / 3]); grans.append(("terciles", x >= q2, x <= q1))
        for gname, mh, ml in grans:
            def wmean(m):
                g = B[m.fillna(False)]
                return g.groupby(["window", "rep"]).pnl_percentage.mean().groupby("window").mean()
            a, b = wmean(mh), wmean(ml)
            if len(a) < 2 or len(b) < 2:
                continue
            rows.append(dict(var=c, gran=gname, n_hi=f"{int(mh.sum())}f/{len(a)}w", hi=round(a.mean(), 3), n_lo=f"{int(ml.sum())}f/{len(b)}w",
                             lo=round(b.mean(), 3), delta=round(a.mean() - b.mean(), 3),
                             both_sides_ge3_windows="yes" if min(len(a), len(b)) >= 3 else ""))
    S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "bear_sweep.csv"), index=False)
    good = S[(S.both_sides_ge3_windows == "yes") & ((S.hi > 0.10) | (S.lo > 0.10))]
    print(f"{len(cols)} stamped entry columns · {len(S)} splits · splits with ≥3 windows on both sides and one side > +0.10 per window: {len(good)}")
    print(good.sort_values("delta").to_string(index=False))
    return S


def sweep2d(n_perm=300):
    """Exhaustive 2D quadrant screen (memory: exhaustive 2D + shuffled null + OOS before any 'no separator' claim). Every pair of
    MACRO + PAIR variables, split at 0 (signed) or the median; every quadrant with ≥ 8 triggers and ≥ 15 fills is scored by its
    per-trigger mean (window units). Null = the same max over all quadrants with fill outcomes permuted. OOS = the best quadrant's
    mean in each half."""
    W = load_walk(); res = []
    for side in ("LONG", "SHORT"):
        pk = W[(W.side == side) & (W.kind == "PICK")].reset_index(drop=True)
        vars_ = [v for v in MACRO + PAIRV if v in pk and pk[v].nunique() > 2 and v != "weekend"]
        bins = {v: ((pk[v] > 0) if v in SIGNED else (pk[v] > pk[v].median())).values for v in vars_}
        trig = pk.trig.values
        def scan(y):
            best = (-9, None)
            for a in range(len(vars_)):
                for b in range(a + 1, len(vars_)):
                    for qa in (True, False):
                        for qb in (True, False):
                            m = (bins[vars_[a]] == qa) & (bins[vars_[b]] == qb)
                            if m.sum() < 15:
                                continue
                            g = pd.Series(y[m]).groupby(trig[m]).mean()
                            if len(g) < 8:
                                continue
                            if g.mean() > best[0]:
                                best = (g.mean(), (vars_[a], qa, vars_[b], qb))
            return best
        y = pk.LIVE.values
        obs, key = scan(y)
        rng = np.random.default_rng(17)
        null = [scan(rng.permutation(y))[0] for _ in range(n_perm)]
        va, qa, vb, qb = key
        m = (bins[va] == qa) & (bins[vb] == qb)
        g = pk[m]
        e = g.groupby("trig").LIVE.mean(); lo, hi = boot_ci(e.values)
        halves = g.groupby("half").apply(lambda x: x.groupby("trig").LIVE.mean().mean()).round(3).to_dict()
        res.append(dict(side=side, quadrants_tested=f"{len(vars_) * (len(vars_) - 1) * 2}", best=f"{va}{'>' if qa else '≤'}{'0' if va in SIGNED else 'med'} ∧ {vb}{'>' if qb else '≤'}{'0' if vb in SIGNED else 'med'}",
                        fills=int(m.sum()), triggers=len(e), per_trig=round(obs, 3), ci=f"[{lo:+.2f},{hi:+.2f}]", halves=str(halves),
                        null_p=round(float(np.mean(np.array(null) >= obs)), 3), null_p95=round(float(np.percentile(null, 95)), 3)))
        print(res[-1], flush=True)
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "sweep2d.csv"), index=False)
    return R


def validate():
    """Fidelity gates for THIS tool against real fills (memory: validate against master).
    V1 replica trigger + universe + selection vs the live decision journal (every live SURGE trigger in the journals)
    V2 replica picks / outcomes vs the corrected engine replay yr5 (same triggers)
    V3 the live counted SURGE fill (SAND 10-02) re-walked from its real open time and price → the live exit
    V4 Bear-Run: replay fills closed by stop / trail / ladder re-walk to the same P&L (two-sided: Δ ≈ 0)"""
    out = []
    F = pd.read_csv(os.path.join(OUT, "events_picks.csv")); F["close"] = pd.to_datetime(F.trig + BAR, unit="ms")
    J = pd.read_csv(os.path.join(OUT, "live_journal_surge.csv"))
    J["t"] = pd.to_datetime(J.t.astype(str).str[:19])
    for (mn, d), j in J[J.gate.astype(str) != "SURGE_WINDOW_MISSED"].groupby(["minute", "dir"]):
        tm = pd.Timestamp(mn)
        delay = float(TH["surge_long_entry_delay_min"] if d == "LONG" else TH["surge_short_entry_delay_min"])
        f = F[(F.side == d) & ((F.close + pd.Timedelta(minutes=delay) - tm).abs() <= pd.Timedelta("5min"))]
        live = {r.pair: ("PICK" if r.e in ("OPEN", "POSITION") else r.gate) for r in j.itertuples()}
        if not len(f):
            out.append(f"V1 {mn} {d}: live trigger not in replica (live trigger bar differs from the replica's 4 h spacing chain — "
                       f"{'expected: sleeve deployed after the 13:30 trigger' if mn.startswith('2026-09-30') else 'CHECK'})"); continue
        rep = dict(zip(f.pair, f.status))
        same = sum(1 for p in live if rep.get(p) == live[p])
        out.append(f"V1 {mn} {d}: {same}/{len(live)} live pairs same status in replica; replica-only {sorted(set(rep) - set(live))}, live-only {sorted(set(live) - set(rep))}")
    W = pd.read_csv(os.path.join(OUT, "walk.csv")); P = W[W.kind == "PICK"].copy(); P["t"] = pd.to_datetime(P.t_entry, unit="ms")
    Y = replay_fills(["SURGE_LONG", "SURGE_SHORT"], prefixes=("yr5",))
    if len(Y):
        Y["t"] = pd.to_datetime(Y.opened_at); Y["side"] = Y.entry_strategy.str[6:]
        m = []
        for r in Y.itertuples():
            x = P[(P.pair == r.pair) & (P.side == r.side) & ((P.t - r.t).abs() < pd.Timedelta("10min"))]
            m.append((r.pnl_percentage, x.LIVE.iloc[0]) if len(x) else (r.pnl_percentage, np.nan))
        m = np.array(m, float); ok = np.isfinite(m[:, 1])
        out.append(f"V2 yr5 SURGE fills matched to a replica pick: {ok.sum()}/{len(m)} · sign agreement {np.mean(np.sign(m[ok, 0]) == np.sign(m[ok, 1])) * 100:.0f} % · "
                   f"mean yr5 {m[ok, 0].mean():+.3f} vs replica {m[ok, 1].mean():+.3f}")
    A = pd.read_csv(os.path.join(OUT, "live_fills.csv")); A = A[A.cohort == "counted"]
    for r in A.itertuples():
        t0 = int(pd.Timestamp(r.opened_at).value // 10**6); side = r.direction
        sp = sec_path(r.pair, t0, t0 + HOLD_MIN * MIN)
        if sp is None:
            out.append(f"V3 {r.pair}: no ticks"); continue
        ts, px, _ = sp; e = float(r.entry_price)
        pnl = ((px / e - 1) * 100 if side == "LONG" else (e - px) / e * 100) - FEE
        o = run_exit(ts, pnl, t0, variants(side, r.entry_atr_pct)["LIVE"][0])
        out.append(f"V3 live {r.entry_strategy} {r.pair} {r.opened_at}: live {r.pnl_percentage:+.3f} {r.close_reason} after "
                   f"{(pd.Timestamp(r.closed_at) - pd.Timestamp(r.opened_at)).total_seconds() / 60:.1f} min · replica {o[0]:+.3f} {o[1]} after {o[2]:.1f} min")
    bf = os.path.join(OUT, "bear_fills.csv")
    if os.path.exists(bf):
        B = pd.read_csv(bf); ne = B[B.exit_family != "EMA13_CROSS_EXIT"]; d = ne.rw_NO_EMA13_STACK - ne.pnl_percentage
        out.append(f"V4 Bear-Run stop/trail/ladder fills re-walked: n {len(ne)} · mean Δ {d.mean():+.3f} · median |Δ| {d.abs().median():.3f}")
    txt = "\n".join(out); open(os.path.join(OUT, "validate.txt"), "w").write(txt); print(txt)


# --------------------------------------------------------------------------------------------------------------- report
MACRO = ["btc_move", "btc_vol_mult", "btc_1h_slope", "btc_above_1h_e50", "btc_r24", "btc_r72", "btc_eff24", "btc_atr", "btc_off30d_hi",
         "breadth_top50", "hour", "weekend"]
PAIRV = ["atr", "rank", "pair_move", "pair_stack", "pair_vs_e50", "pair_r24", "pair_r4h", "pair_vol_ratio"]
SIGNED = {"btc_1h_slope", "btc_above_1h_e50", "btc_r24", "btc_r72", "pair_vs_e50", "pair_r24", "pair_r4h", "pair_move", "pair_stack"}


def _split_rows(g, col, unit_trigger):
    """sign (signed vars) · median · outer terciles; Δ = hi − lo in per-trigger means (trigger-clustered); halves consistency."""
    x = g[col]
    if x.notna().sum() < 10 or x.nunique() < 2:
        return []
    out = []
    grans = []
    if col in SIGNED:
        grans.append(("sign", x > 0, x <= 0))
    med = x.median(); grans.append(("median", x > med, x <= med))
    q1, q2 = x.quantile([1 / 3, 2 / 3]); grans.append(("terciles", x >= q2, x <= q1))
    for gran, mh, ml in grans:
        hi, lo = g[mh.fillna(False)], g[ml.fillna(False)]
        if len(hi) < 3 or len(lo) < 3:
            continue
        mh_ = hi.groupby("trig").LIVE.mean(); ml_ = lo.groupby("trig").LIVE.mean()
        d = mh_.mean() - ml_.mean()
        # cluster bootstrap of the difference (resample triggers within each arm)
        rng = np.random.default_rng(3); bs = []
        for _ in range(2000):
            bs.append(rng.choice(mh_.values, len(mh_)).mean() - rng.choice(ml_.values, len(ml_)).mean())
        lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
        da = (hi[hi.half == "Jan-Apr"].groupby("trig").LIVE.mean().mean() - lo[lo.half == "Jan-Apr"].groupby("trig").LIVE.mean().mean())
        db = (hi[hi.half == "May-Oct"].groupby("trig").LIVE.mean().mean() - lo[lo.half == "May-Oct"].groupby("trig").LIVE.mean().mean())
        out.append(dict(var=col, gran=gran, cut=round(float(0 if gran == "sign" else (med if gran == "median" else q2)), 3),
                        n_hi=f"{len(hi)}f/{len(mh_)}t", hi=round(mh_.mean(), 3), n_lo=f"{len(lo)}f/{len(ml_)}t", lo=round(ml_.mean(), 3),
                        delta=round(d, 3), ci=f"[{lo_ci:+.2f},{hi_ci:+.2f}]", d_jan_apr=round(da, 3) if da == da else np.nan,
                        d_may_oct=round(db, 3) if db == db else np.nan,
                        consistent="yes" if (da == da and db == db and np.sign(da) == np.sign(db) == np.sign(d)) else "",
                        ci_excl0="yes" if (lo_ci > 0 or hi_ci < 0) else ""))
    return out


def sweep(g):
    rows = []
    for c in MACRO + PAIRV:
        if c in g:
            rows += _split_rows(g, c, True)
    return pd.DataFrame(rows)


def report():
    W = load_walk()
    L = []
    def H(t):
        L.append("\n## " + t + "\n")
    for side in ("LONG", "SHORT"):
        g = W[(W.side == side)]
        H(f"SURGE_{side} — tick replica, live rules (entry +{LATENCY_S}s, live exit, 240 min, net of 0.09 %)")
        rows = []
        for k in ("PICK", "CTRL", "CONTROL_SAMEPAIR", "REFUSED_ATR_LOW", "REFUSED_NOT_LEADER", "REFUSED_MAX_SLOTS"):
            x = g[g.kind == k]
            if len(x):
                l = ev_line(x); l.pop("ci_lo"); l.pop("ci_hi"); rows.append(dict(cohort=k, **l))
        L.append(md(pd.DataFrame(rows)))
        pk = g[g.kind == "PICK"]; ct = g[g.kind == "CTRL"]
        d = pk.groupby("trig").LIVE.mean(); c = ct.groupby("trig").LIVE.mean()
        j = pd.concat([d, c], axis=1, keys=["pick", "ctrl"]).dropna()
        lo, hi = boot_ci((j.pick - j.ctrl).values)
        L.append(f"\nPaired PICK − CTRL (same trigger, fresh no-trigger selection ±1/2 days): {(j.pick - j.ctrl).mean():+.3f} per trigger, "
                 f"95 % CI [{lo:+.2f}, {hi:+.2f}], n={len(j)} triggers.")
        why = pk.groupby("LIVE_why").LIVE.agg(["size", "mean"]).round(3)
        wins, loss = pk[pk.LIVE > 0].LIVE.mean(), pk[pk.LIVE <= 0].LIVE.mean()
        L.append(f"\nExit mix (PICK): " + ", ".join(f"{k} {int(v['size'])} × {v['mean']:+.3f}" for k, v in why.iterrows()) +
                 f". Avg win {wins:+.3f} / avg loss {loss:+.3f} → breakeven WR {abs(loss) / (wins + abs(loss)) * 100:.0f} % vs actual "
                 f"{(pk.LIVE > 0).mean() * 100:.0f} %.")
        H(f"SURGE_{side} — per month (window units)")
        mm = []
        for m, x in pk.groupby("month"):
            e = x.groupby("trig").LIVE.mean(); cc = ct[ct.month == m].groupby("trig").LIVE.mean()
            mm.append(dict(month=m, triggers=len(e), fills=len(x), per_trig=round(e.mean(), 3), trig_pos=f"{(e > 0).sum()}/{len(e)}",
                           ctrl_per_trig=round(cc.mean(), 3) if len(cc) else np.nan, usd=round(x.LIVE.sum() * TICKET / 100)))
        L.append(md(pd.DataFrame(mm)))
        pc = pk.groupby("pair").LIVE.agg(["size", "sum"]).sort_values("sum")
        neg = pk[pk.LIVE < 0].LIVE.sum()
        L.append(f"\nPair concentration: {pk.pair.nunique()} pairs; worst 2 pairs carry {pc['sum'].head(2).sum() / neg * 100:.0f} % of the gross loss "
                 f"({', '.join(f'{p} {int(r['size'])}×{r['sum']:+.2f}' for p, r in pc.head(3).iterrows())}); best: "
                 f"{', '.join(f'{p} {int(r['size'])}×{r['sum']:+.2f}' for p, r in pc.tail(3).iterrows())}.")
        H(f"SURGE_{side} — exit variants on the same fills (Δ vs LIVE paired per trigger)")
        vs = [c for c in pk.columns if (c == "LIVE" or c.startswith(("BR_", "LIVE_", "MOM_", "FIX_", "HOLD_", "QUICK", "ST_")))
              and c not in ("LIVE_why", "LIVE_min", "LIVE_peak") and pk[c].notna().any()]
        rows = []
        for v in vs:
            l = ev_line(pk, v); dd = (pk[v] - pk.LIVE).groupby(pk.trig).mean(); lo, hi = boot_ci(dd.values)
            cl = ev_line(ct, v) if ct[v].notna().any() else {}
            rows.append(dict(variant=v, per_trig=l["per_trig"], ci=l["ci"], wr=l["wr"], jan_apr=l["jan_apr"], may_oct=l["may_oct"],
                             d_vs_live=round(dd.mean(), 3), d_ci=f"[{lo:+.2f},{hi:+.2f}]", ctrl_per_trig=cl.get("per_trig")))
        V = pd.DataFrame(rows).sort_values("per_trig", ascending=False)
        V.to_csv(os.path.join(OUT, f"variants_{side}.csv"), index=False)
        L.append(md(V))
        H(f"SURGE_{side} — stop width, two-sided (live-stopped cohort only)")
        st = pk[pk.LIVE_why == "STOP"]; ns = pk[pk.LIVE_why != "STOP"]
        rows = []
        for v in [c for c in vs if c.startswith("LIVE_SL")]:
            dv = st[v] - st.LIVE
            rows.append(dict(variant=v, live_stopped=len(st), saved=int((dv > 0.05).sum()), deeper=int((dv < -0.05).sum()),
                             d_stopped=round(dv.mean(), 3), d_not_stopped=round((ns[v] - ns.LIVE).mean(), 3),
                             not_stopped_changed=int(((ns[v] - ns.LIVE).abs() > 1e-9).sum()), d_all_per_trig=round((pk[v] - pk.LIVE).groupby(pk.trig).mean().mean(), 3)))
        L.append(md(pd.DataFrame(rows)))
        H(f"SURGE_{side} — separator sweep (checklist ①②): macro = trigger-level (window units), pair = fill-level, trigger-clustered CI")
        S = sweep(pk)
        S.to_csv(os.path.join(OUT, f"sweep_{side}.csv"), index=False)
        L.append(md(S))
        H(f"SURGE_{side} — EMA-stack watch (rebuilt from closed 5m bars for EVERY fill: +1 = EMA5>8>13>20, −1 = reverse)")
        rows = []
        for v, x in pk.groupby("pair_stack"):
            e = x.groupby("trig").LIVE.mean()
            rows.append(dict(stack=int(v), fills=len(x), triggers=len(e), wr=round((x.LIVE > 0).mean() * 100), avg_fill=round(x.LIVE.mean(), 3),
                             per_trig=round(e.mean(), 3)))
        L.append(md(pd.DataFrame(rows)))
        H(f"SURGE_{side} — uniform-degradation test (checklist ③): per-trigger mean by quarter × cohort")
        pk = pk.assign(q=pd.to_datetime(pk.trig, unit="ms").dt.quarter.map(lambda q: f"Q{q}"),
                       atr_t=pd.qcut(pk.atr, 3, labels=["atr_lo", "atr_mid", "atr_hi"]),
                       rank_h=np.where(pk["rank"] <= 10, "rank1-10", "rank11-20"))
        rows = []
        for coh in ("atr_t", "rank_h", "pair_stack"):
            for v, x in pk.groupby(coh, observed=True):
                r = dict(cohort=f"{coh}={v}")
                for q, y in x.groupby("q"):
                    r[q] = f"{y.groupby('trig').LIVE.mean().mean():+.2f} ({len(y)})"
                rows.append(r)
        L.append(md(pd.DataFrame(rows).fillna("")))
    # ---- design simulator vs ticks (same fills)
    cf = os.path.join(OUT, "candle_vs_tick.csv")
    if os.path.exists(cf):
        C = pd.read_csv(cf); H("Why the design grids looked positive — the SAME PICK fills under three simulators (per trigger / per fill)")
        rows = []
        for side, g in C.groupby("side"):
            for k, lab in (("candle_1m", "design: 1m candle path, entry at the window open, stop filled AT the level"),
                           ("tick_lat0", "ticks, entry at the window open (no latency), fill at the crossing print"),
                           ("tick_lat60", f"ticks, entry +{LATENCY_S}s (live-like), fill at the crossing print")):
                rows.append(dict(side=side, simulator=lab, per_trig=round(g.groupby("trig")[k].mean().mean(), 3), per_fill=round(g[k].mean(), 3),
                                 wr=round((g[k] > 0).mean() * 100)))
        L.append(md(pd.DataFrame(rows)))
    sf = os.path.join(OUT, "sweep2d.csv")
    if os.path.exists(sf):
        H("Exhaustive 2D quadrant screen + shuffled null + halves (no-separator claim)"); L.append(md(pd.read_csv(sf)))
    bs = os.path.join(OUT, "bear_sweep.csv")
    if os.path.exists(bs):
        H("BEARRUN_SHORT stamped entry_* sweep (replicate-window units; 7 windows → under-powered, listed for completeness)")
        L.append(md(pd.read_csv(bs)))
    vf = os.path.join(OUT, "validate.txt")
    if os.path.exists(vf):
        H("Fidelity checks of this tool vs real fills (stage validate)"); L.append("\n".join("- " + x for x in open(vf).read().splitlines()))
    rf = os.path.join(OUT, "robust.txt")
    if os.path.exists(rf):
        H("Sweep null, leave-one-month-out, tape context (checklist ④)"); L.append(open(rf).read())
    # ---- yr5 engine replay (corrected harness)
    Y = replay_fills(["SURGE_LONG", "SURGE_SHORT", "BEARRUN_SHORT"], prefixes=("yr5",))
    if len(Y):
        Y = Y[Y.status == "CLOSED"]
        H(f"yr5 engine replay (corrected harness, 3 seeds) — chunks available: {', '.join(sorted(Y.chunk.unique()))}")
        rows = []
        for st, g in Y.groupby("entry_strategy"):
            g = g.assign(trig=pd.to_datetime(g.opened_at).dt.floor("h"))
            e = g.groupby(["trig", "seed"]).pnl_percentage.mean().groupby("trig").mean()
            lo, hi = boot_ci(e.values)
            rows.append(dict(sleeve=st, fills_all_seeds=len(g), fills_per_seed=round(len(g) / g.seed.nunique(), 1), windows=len(e),
                             wr=round((g.pnl_percentage > 0).mean() * 100), avg_fill=round(g.pnl_percentage.mean(), 3),
                             per_window=round(e.mean(), 3), ci=f"[{lo:+.2f}, {hi:+.2f}]", windows_pos=f"{(e > 0).sum()}/{len(e)}",
                             exits=", ".join(f"{k} {v}" for k, v in g.close_reason.astype(str).str.replace(r" L\d+", "", regex=True).value_counts().items())))
        L.append(md(pd.DataFrame(rows)))
    # ---- Bear-Run (engine replays)
    bf = os.path.join(OUT, "bear_fills.csv")
    if os.path.exists(bf):
        B = pd.read_csv(bf); B["rep"] = B.run + "_" + B.seed
        H("BEARRUN_SHORT — engine replays (yr4 Jan→Oct valid for Bear-Run + yr5), WINDOW units (mean over replicates of each window)")
        w = B.groupby(["window", "rep"]).pnl_percentage.agg(["size", "mean"]).reset_index()
        T = w.groupby("window").agg(replicates=("rep", "nunique"), fills_per_rep=("size", "mean"), mean=("mean", "mean"),
                                    rep_min=("mean", "min"), rep_max=("mean", "max"))
        T["pairs"] = B.groupby("window").pair.apply(lambda x: " ".join(sorted(set(p.replace("USDT", "") for p in x))))
        T["exits"] = B.groupby("window").exit_family.apply(lambda x: ", ".join(f"{k} {v}" for k, v in x.value_counts().items()))
        L.append(md(T.reset_index()))
        rows = []
        for c in ["pnl_percentage"] + [c for c in B.columns if c.startswith("rw_") and c != "rw_why"]:
            ww = B.groupby(["window", "rep"])[c].mean().groupby("window").mean()
            rows.append(dict(exit=("AS REPLAYED (live momentum-short stack)" if c == "pnl_percentage" else c[3:]), per_window=round(ww.mean(), 3),
                             windows_pos=f"{(ww > 0).sum()}/{len(ww)}", ex_sep15=round(ww[[not i.startswith("2026-09-15") for i in ww.index]].mean(), 3)))
        L.append("\nExit re-walk on ticks (same entries), window units:\n\n" + md(pd.DataFrame(rows).sort_values("per_window", ascending=False)))
        y4 = B[B.run == "yr4"].sort_values("opened_at"); kb = []
        for sd, x in y4.groupby("seed"):
            f10 = x.pnl_percentage.head(10); ws = x.groupby("window").pnl_percentage.sum()
            kb.append(f"{sd}: first-10 WR {(f10 > 0).mean() * 100:.0f} % Σ {f10.sum():+.2f} · window signs {''.join('+' if v > 0 else '−' for v in ws.values)}")
        L.append("\nOwn kill bar replayed per yr4 seed (first 10 fills WR ≤ 45 % ∨ Σ < 0, or 2 consecutive net-negative windows): " + " | ".join(kb))
        ne = B[B.exit_family != "EMA13_CROSS_EXIT"]; d = ne.rw_NO_EMA13_STACK - ne.pnl_percentage
        L.append(f"\nRe-walk fidelity (two-sided): fills the replay closed by stop / trail / ladder re-walk to Δ {d.mean():+.3f} (median |Δ| {d.abs().median():.3f}, "
                 f"{(d.abs() > 0.1).sum()} of {len(ne)} beyond 0.10).")
    # ---- live
    lf = os.path.join(OUT, "live_fills.csv")
    if os.path.exists(lf):
        A = pd.read_csv(lf)
        H("LIVE fills (every export + archive, dedup (opened_at, pair, direction), MANUAL out)")
        L.append(md(A[["entry_strategy", "pair", "opened_at", "closed_at", "pnl_percentage", "pnl", "close_reason", "peak_pnl", "cohort", "src"]]))
    jf = os.path.join(OUT, "live_journal_surge.csv")
    if os.path.exists(jf):
        J = pd.read_csv(jf)
        H("LIVE SURGE funnel from the decision journals (every SURGE gate / open by entry-window minute)")
        L.append(md(J.drop_duplicates(["minute", "pair", "dir", "gate", "e"]).groupby(["minute", "dir", "e", "gate"], dropna=False).size().reset_index(name="pairs")))
    open(os.path.join(ROOT, "reports", "SURGE_BEARRUN_REVIEW_2026-10-04_tables.md"), "w").write(
        "# SURGE / BEAR-RUN review — generated tables (scripts/surge_bearrun_review.py report)\n" + "\n".join(L) + "\n")
    print("\n".join(L)[:20000])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    t0 = time.time()
    if stage == "live":
        a = live(); print(a[["src", "pair", "direction", "entry_strategy", "opened_at", "pnl_percentage", "close_reason", "cohort"]])
    elif stage == "events":
        events()
    elif stage == "needticks":
        needticks()
    elif stage == "walk":
        walk()
    elif stage == "ctrl":
        ctrl()
    elif stage == "bear":
        bear()
    elif stage == "candle":
        candle()
    elif stage == "report":
        report()
    elif stage == "robust":
        robust()
    elif stage == "validate":
        validate()
    elif stage == "bearsweep":
        bear_sweep()
    elif stage == "sweep2d":
        sweep2d()
    elif stage == "features":
        features()
    print(f"done in {time.time() - t0:.0f}s")


