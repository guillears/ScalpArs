#!/usr/bin/env python3
"""SPIKE_FADE recall trace (2026-10-08) — step 3: per-signal re-evaluation of the spike trigger + every fade gate.

Read-only. Inputs: reports/study_fade_trace_matchset.csv (step 1), the yr5 replay's OWN decision journals (SCAN instants with the
BTC readings the engine used, pair-tagged capacity blocks BOOK_FULL / PAIR_HELD / COOLDOWN / NO_BALANCE, OPEN lines), the replay
kline cache (k5m_full / k5m / k1m + k1m_ondemand + k1m_spike / k1d / k4h), real trades (replay cache ticks_q + the study's own
reports/study_fade_trace_ticks), live decision journals (~/Downloads/scalpars_decisions_paper_*.csv, Sep-28 → ) and the live
config history (reports/backtest_cache/replay/config_history_v2).

Two views of the FORMING 5m candle are rebuilt exactly like scripts/engine_replay.py (code_yr5_181131e) _ohlcv5:
  replay view = ticks_q (only when that pair-day file existed when the chunk ran) → else the 1m rebuild from COMPLETED 1m bars
                [candle open, floor-minute(t)) → else the last completed 5m candle plays the forming one;
  truth view  = real trades [candle open, t) (what the live bot's REST kline read sees).
Trigger = _spike_scanner_cycle legs on 100 bars: Wilder RSI(12) prev in [35,55], jump ≥ 25, candle ≥ +0.5 %, volume ≥ 5× prior-20.
Router (regime from the SCAN line's BTC ADX/RSI/EMA20-slope; chase regimes STRONG_BULL/HEALTHY_BULL; pair ADX > 30 → fade) and
gates in engine order: MAXVOL (replay rolling 24 h quote volume ≥ $20M) · BRSI (> 50) · BD13 (BTC above 5m EMA13) · FRESHBREAK
(RSI12 prev < 44 ∧ EMA13-50 gap > −0.40 %) · LAGGARD (pair 1d −DI14 > 16.1 ∧ BTC 4h EMA50/200 gap > 0).
Replay scanner decision instant = SCAN + 45 s (replay SCAN→OPEN median 54 s minus the 6 s open delay + latency; a ±15 s window is
also scanned). Live decision instant = opened_at − 2…40 s.
Outputs: reports/study_fade_trace_signals.csv (one row per signal × seed) and prints the parity checks.
Usage: venv/bin/python scripts/study_fade_trace_eval.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "reports", "backtest_cache", "replay", "code_yr5_181131e")
sys.path.insert(0, SNAP); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services.indicators import calculate_indicators, closed_wilder_ndi, closed_ema_gap_pct   # noqa: E402  (frozen engine code)
from services.regime import classify_btc_regime                                              # noqa: E402
import yr5_fills_trimmed as YT                                                               # noqa: E402

REP = os.path.join(ROOT, "reports")
CACHE = os.path.join(REP, "backtest_cache")
JDIR = os.path.join(SNAP, "reports", "backtest_cache", "replay", "year", "journal")
STUDY_TICKS = os.path.join(REP, "study_fade_trace_ticks")
CFG = json.load(open(os.path.join(CACHE, "replay", "frozen_config_yr5_181131e.json")))
TH = CFG["thresholds"]
EXINFO = json.load(open(os.path.join(CACHE, "exchange_info.json")))
BL = set(x.strip() for x in (CFG.get("pair_blacklist") or "").split(",") if x.strip())
NT = set(x.strip() for x in (CFG.get("no_trade_pairs") or "").split(",") if x.strip())
DAY, M5 = 86_400_000, 300_000
EVAL_OFF = 45_000
OFFS = (30_000, 37_000, 45_000, 52_000, 58_000, 65_000)
CAP_GATES = ("BOOK_FULL", "PAIR_HELD", "COOLDOWN", "NO_BALANCE", "WILLY_HOLD", "GLOBAL_HOLD", "GROSS_CAP_SKIP", "LIQ_CAP_SKIP", "BRACKET_CAP_SKIP")
W0, W1 = pd.Timestamp("2026-07-25"), pd.Timestamp("2026-10-05")


def tms(t):
    return int(pd.Timestamp(t).value // 1_000_000)


# ───────────────────────────── data access (mirrors engine_replay.py KlineServer / TickStore) ─────────────────────────────
class Data:
    def __init__(self):
        self.k5, self.k1, self.tk, self.hcache = {}, {}, {}, {}

    @staticmethod
    def _csv(path):
        d = pd.read_csv(path).drop_duplicates("open_time").sort_values("open_time")
        lo, hi = tms(W0) - 40 * DAY, tms(W1) + 2 * DAY
        return d[(d.open_time >= lo) & (d.open_time < hi)]

    def get5(self, pair):
        if pair not in self.k5:
            if pair == "BTCUSDT":
                fr = [pd.read_csv(os.path.join(CACHE, f)) for f in ("btc_5m.csv", "k5m_full/BTCUSDT.csv", "k5m/BTCUSDT.csv")
                      if os.path.exists(os.path.join(CACHE, f))]
                d = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time")
            else:
                fp = next((os.path.join(CACHE, s, f"{pair}.csv") for s in ("k5m_full", "k5m") if os.path.exists(os.path.join(CACHE, s, f"{pair}.csv"))), None)
                d = self._csv(fp) if fp else None
            self.k5[pair] = None if d is None or not len(d) else {
                "ts": d.open_time.values.astype(np.int64), "o": d.o.values.astype(float), "h": d.h.values.astype(float),
                "l": d.l.values.astype(float), "c": d.c.values.astype(float), "v": d.vol.values.astype(float), "q": d.qvol.values.astype(float)}
        return self.k5[pair]

    def get1(self, pair):
        if pair not in self.k1:
            fr = [pd.read_csv(os.path.join(CACHE, s, f"{pair}.csv")) for s in ("k1m", "k1m_ondemand", "k1m_spike")
                  if os.path.exists(os.path.join(CACHE, s, f"{pair}.csv"))]
            d = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time") if fr else None
            self.k1[pair] = None if d is None or not len(d) else {
                "ts": d.open_time.values.astype(np.int64), "o": d.o.values.astype(float), "h": d.h.values.astype(float),
                "l": d.l.values.astype(float), "c": d.c.values.astype(float), "v": d.vol.values.astype(float)}
        return self.k1[pair]

    def ticks(self, pair, day_ms, source):
        """source 'cache' = replay ticks_q only; 'truth' = ticks_q or the study's own archive."""
        key = (pair, day_ms, source)
        if key not in self.tk:
            ds = pd.Timestamp(day_ms, unit="ms").strftime("%Y-%m-%d")
            fps = [os.path.join(CACHE, "ticks_q", pair, f"{ds}.npz")]
            if source == "truth":
                fps.append(os.path.join(STUDY_TICKS, pair, f"{ds}.npz"))
            fp = next((f for f in fps if os.path.exists(f)), None)
            if fp is None:
                self.tk[key] = None
            else:
                z = np.load(fp)
                self.tk[key] = (z["t"].astype(np.int64), z["p"].astype(float), z["q"].astype(float), os.path.getmtime(fp))
        return self.tk[key]

    def forming_ticks(self, pair, t0, t1, source, run_start=None):
        d = self.ticks(pair, (t0 // DAY) * DAY, source)
        if d is None or (t1 - 1) // DAY != t0 // DAY:
            return None
        if source == "cache" and run_start is not None and d[3] > run_start:
            return None                                       # file written after this chunk ran → the replay did not have it
        t, p, q, _ = d
        a, b = np.searchsorted(t, t0), np.searchsorted(t, t1)
        if b <= a:
            return ()
        return (p[a], p[a:b].max(), p[a:b].min(), p[b - 1], q[a:b].sum())

    def ohlcv5(self, pair, t_ms, view, run_start=None, limit=100):
        """returns (bars, source) — view 'replay' or 'truth'."""
        s = self.get5(pair)
        if s is None:
            return [], "NO_5M"
        cur = (t_ms // M5) * M5
        n = int(np.searchsorted(s["ts"], cur))
        closed = lambda lo, hi: [[int(s["ts"][i]), s["o"][i], s["h"][i], s["l"][i], s["c"][i], s["v"][i]] for i in range(lo, hi)]
        fc = self.forming_ticks(pair, cur, t_ms, "cache" if view == "replay" else "truth", run_start)
        if fc is not None:
            out = closed(max(0, n - (limit - 1)), n)
            if fc:
                out.append([cur, *fc])
            elif out:
                px = out[-1][4]; out.append([cur, px, px, px, px, 0.0])
            return out, "TICKS"
        if view == "truth":
            return [], "NO_TRUTH_TICKS"
        s1 = self.get1(pair)
        has1 = s1 is not None and s1["ts"][0] <= cur and s1["ts"][-1] >= cur - 60_000
        if has1:
            out = closed(max(0, n - (limit - 1)), n)
            m_end = (t_ms // 60_000) * 60_000
            a, b = np.searchsorted(s1["ts"], cur), np.searchsorted(s1["ts"], m_end)
            if b > a:
                out.append([cur, s1["o"][a], s1["h"][a:b].max(), s1["l"][a:b].min(), s1["c"][b - 1], s1["v"][a:b].sum()])
            else:
                if not out:
                    return [], "NO_DATA"
                px = out[-1][4]; out.append([cur, px, px, px, px, 0.0])
            return out, f"1M({(b - a)}min)"
        return closed(max(0, n - limit), n), "CLOSED_ONLY"

    def vol24(self, pair, t_ms):
        s = self.get5(pair)
        if s is None:
            return np.nan
        i = int(np.searchsorted(s["ts"], t_ms - M5, side="right"))
        return float(s["q"][i - 288:i].sum()) if i >= 288 else np.nan

    def htf(self, pair, sub, tf_ms, t_ms):
        key = (pair, sub)
        if key not in self.hcache:
            fp = os.path.join(CACHE, sub, f"{pair}.csv")
            self.hcache[key] = pd.read_csv(fp).drop_duplicates("open_time").sort_values("open_time") if os.path.exists(fp) else None
        h = self.hcache[key]
        if h is None:
            return None
        cur = (t_ms // tf_ms) * tf_ms
        c = h[h.open_time < cur]
        bars = c[["open_time", "o", "h", "l", "c", "vol"]].values.tolist()
        return bars + [[cur, 0, 0, 0, 0, 0]]                        # dummy forming bar (both readers drop the last bar)


D = Data()


# ───────────────────────────── trigger legs (engine _spike_rsi12 + scanner legs) ─────────────────────────────
def rsi12(closes):
    if len(closes) < 20:
        return None, None
    au = ad = 0.0; rs = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]; u = ch if ch > 0 else 0.0; d = -ch if ch < 0 else 0.0
        if i == 1:
            au, ad = u, d
        else:
            au = (au * 11 + u) / 12.0; ad = (ad * 11 + d) / 12.0
        if i >= 12:
            rs.append(100.0 - 100.0 / (1 + au / ad) if ad > 0 else 100.0)
    return (rs[-1], rs[-2]) if len(rs) >= 2 else (None, None)


def legs(bars):
    if len(bars) < 22:
        return None
    closes = [float(b[4]) for b in bars]; vols = [float(b[5]) for b in bars]
    r, rp = rsi12(closes)
    if r is None:
        return None
    chg = (closes[-1] / closes[-2] - 1) * 100 if closes[-2] > 0 else np.nan
    av = sum(vols[-21:-1]) / 20.0
    vr = vols[-1] / av if av > 0 else 0.0
    ok = {"prev": 35 <= rp <= 55, "jump": (r - rp) >= 25, "chg": chg >= 0.5, "vol": vr >= 5}
    fail = [k for k in ("prev", "jump", "chg", "vol") if not ok[k]]
    return {"rsi": r, "rsi_prev": rp, "jump": r - rp, "chg": chg, "vr": vr, "trig": not fail, "fail": "+".join(fail), "cts": int(bars[-1][0])}


def btc_view(t_ms, run_start=None):
    bars, _ = D.ohlcv5("BTCUSDT", t_ms, "replay", run_start)
    ind = calculate_indicators(bars) if len(bars) >= 60 else None
    if not ind:
        return None
    return (bars[-1][4] - ind["ema13"]) / ind["ema13"] * 100 if ind.get("ema13") else None


_lag_cache = {}


def laggard(pair, t_ms):
    k = (pair, t_ms // 3_600_000)
    if k not in _lag_cache:
        d1 = D.htf(pair, "k1d", DAY, t_ms)
        ndi = closed_wilder_ndi(d1) if d1 else None
        b4 = D.htf("BTCUSDT", "k4h", 14_400_000, t_ms)
        gap = closed_ema_gap_pct(b4[-1001:], 50, 200) if b4 else None
        _lag_cache[k] = (ndi, gap)
    return _lag_cache[k]


def gates(pair, bars, lg, scan, t_ms, run_start, era_th=None):
    """first blocking gate in engine order (None = passes). era_th: thresholds dict to use instead of today's (live era check)."""
    th = era_th or TH
    ind = calculate_indicators(bars)
    adx = ind.get("adx") if ind else None
    reg = classify_btc_regime(scan.get("btc_adx"), scan.get("btc_rsi"), scan.get("btc_slope"))
    chase = set(x.strip() for x in (th.get("spike_chase_regimes") or "").split(",") if x.strip())
    if th.get("spike_regime_router_enabled"):
        reg_fade = (reg not in chase) if reg not in (None, "UNKNOWN") else True
    else:
        reg_fade = False
    info = {"adx": adx, "regime": reg, "btc_rsi": scan.get("btc_rsi")}
    if not (reg_fade or (adx is not None and adx > float(th.get("spike_chase_max_adx") or 30))):
        return "ROUTED_CHASE", info
    if not th.get("spike_fade_enabled", False):
        return "FADE_DISABLED", info
    v24 = D.vol24(pair, t_ms); info["vol24"] = v24
    vmax = float(th.get("spike_fade_max_vol_24h_usd") or 0)
    if vmax > 0 and v24 == v24 and v24 >= vmax:
        return "MAXVOL", info
    bmax = float(th.get("spike_fade_max_btc_rsi") or 0)
    if bmax > 0 and scan.get("btc_rsi") is not None and scan["btc_rsi"] > bmax:
        return "BRSI", info
    bd = th.get("spike_fade_max_btc_dist13")
    bd = 99.0 if bd is None else float(bd)
    if bd < 99:
        d13 = btc_view(int(scan["tms"]), run_start); info["bd13"] = d13
        if d13 is not None and d13 > bd:
            return "BD13", info
    rmin = float(th.get("spike_fade_fb_rsi_prev_min") or 0)
    gmin = th.get("spike_fade_fb_pgap_min"); gmin = -0.40 if gmin is None else float(gmin)
    if rmin > 0 and lg["rsi_prev"] < rmin and ind and ind.get("ema13") is not None and ind.get("ema50"):
        pg = (ind["ema13"] - ind["ema50"]) / ind["ema50"] * 100; info["pgap"] = pg
        if pg > gmin:
            return "FRESHBREAK", info
    nmin = float(th.get("spike_fade_lag_ndi_min") or 0)
    if nmin > 0:
        ndi, gap = laggard(pair, t_ms); info["ndi"], info["btc4h_gap"] = ndi, gap
        if ndi is not None and gap is not None and ndi > nmin and gap > float(th.get("spike_fade_lag_btc_gap_min") or 0):
            return "LAGGARD", info
    return None, info


# ───────────────────────────── replay journals ─────────────────────────────
def load_journals():
    J = {}
    for tag, chunk, seed, a, z, _w in YT.chunk_windows("yr5"):
        if z < W0 or a > W1:
            continue
        fp_orders = os.path.join(CACHE, "replay", f"{tag}_orders.csv")
        meta = json.load(open(os.path.join(CACHE, "replay", f"{tag}_meta.json")))
        run_start = os.path.getmtime(fp_orders) - float(meta.get("elapsed_s", 0)) - 600
        sc, bl, sp, op = [], [], [], []
        for f in sorted(glob.glob(os.path.join(JDIR, tag, "decisions-*.jsonl"))):
            for line in open(f):
                if '"SCAN"' in line:
                    d = json.loads(line); t = pd.Timestamp(d["t"])
                    if a <= t < z:
                        sc.append((t, d.get("btc_rsi"), d.get("btc_adx"), d.get("btc_slope")))
                elif '"BLOCK"' in line and ('SPIKE_FADE' in line or any(g in line for g in CAP_GATES)):
                    d = json.loads(line); t = pd.Timestamp(d["t"])
                    if a <= t < z:
                        (bl if d.get("pair") else sp).append((t, d.get("pair"), d.get("gate")))
                elif '"OPEN"' in line:
                    d = json.loads(line); t = pd.Timestamp(d["t"])
                    if a <= t < z:
                        op.append((t, d.get("pair"), d.get("strategy")))
        J.setdefault(seed, {"scan": [], "blk": [], "sp": [], "open": [], "chunks": []})
        J[seed]["scan"] += sc; J[seed]["blk"] += bl; J[seed]["sp"] += sp; J[seed]["open"] += op
        J[seed]["chunks"].append((a, z, run_start))
    for s in J:
        S = pd.DataFrame(J[s]["scan"], columns=["t", "btc_rsi", "btc_adx", "btc_slope"]).sort_values("t").reset_index(drop=True)
        S["tms"] = S.t.values.astype("datetime64[ms]").astype("int64")
        J[s]["scan"] = S
        J[s]["blk"] = pd.DataFrame(J[s]["blk"], columns=["t", "pair", "gate"])
        J[s]["sp"] = pd.DataFrame(J[s]["sp"], columns=["t", "pair", "gate"])
        J[s]["open"] = pd.DataFrame(J[s]["open"], columns=["t", "pair", "strategy"])
    return J


def run_start_for(J, seed, t):
    for a, z, rs in J[seed]["chunks"]:
        if a <= t < z:
            return rs
    return None


# ───────────────────────────── live journal (Sep-28 →) ─────────────────────────────
def load_live_journal():
    fs = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")))
    parts = []
    for f in fs:
        d = pd.read_csv(f, usecols=["t", "e", "pair", "dir", "gate", "n", "strategy"], low_memory=False)
        parts.append(d[d.e.isin(["SCAN", "BLOCK", "OPEN"])])
    L = pd.concat(parts).drop_duplicates()
    L["t"] = pd.to_datetime(L.t.astype(str).str[:23], format="mixed")
    return L


def era_thresholds(t):
    best = None
    for f in sorted(glob.glob(os.path.join(CACHE, "replay", "config_history_v2", "*_*.json"))):
        b = os.path.basename(f).split("_")[0]
        if not b.isdigit():
            continue
        if int(b) <= tms(t):
            best = f
    return json.load(open(best))["thresholds"] if best else None


# ───────────────────────────── per-signal evaluation ─────────────────────────────
def eval_instant(pair, t_ms, view, run_start):
    bars, src = D.ohlcv5(pair, t_ms, view, run_start)
    lg = legs(bars) if bars else None
    return bars, src, lg


def trigger_window(pair, cts):
    """seconds of the candle [cts, cts+300 s) in which the TRUTH trigger legs hold (1 s grid)."""
    hits = []
    for sec in range(1, 300):
        bars, src, lg = eval_instant(pair, cts + sec * 1000, "truth", None)
        if lg is None:
            return None, None
        if lg["trig"]:
            hits.append(sec)
    return len(hits), (hits[0] if hits else None)


# replay decision instant for a scanner pair: yr5 fade fills (Jul–Oct, N 210) give OPEN − SCAN = 25.3 s + 0.170 s × universe rank
# (r = 0.997 — the scanner walks its candidates in volume-rank order, 8 per batch); minus the ~7 s open path (6 s open delay +
# latency) → decision ≈ SCAN + 18.3 + 0.17 × rank, ± 4 s.
def central_off(rank):
    r = 180.0 if rank is None or rank != rank else float(rank)
    return 18_300 + 170.0 * r
GRID = tuple(range(20_000, 92_000, 3_000))


def replay_held(pair, t, seed, OR):
    o = OR[(OR.seed == seed) & (OR.pair == pair) & (OR.ta <= t) & (OR.tz > t)]
    return len(o) > 0


def trace(pair, T, J, seed, OR=None, rank=None):
    """every replay scan of `seed` in [T-6 min, T+4 min]: replay-view trigger on a 3 s grid of decision offsets (bits), first triggering
    scan per candle (central offsets first, else any grid offset), then universe floor, gates, capacity."""
    S = J[seed]["scan"]
    rs = run_start_for(J, seed, T)
    win = S[(S.t >= T - pd.Timedelta(minutes=6)) & (S.t <= T + pd.Timedelta(minutes=4))]
    rows, seen = [], set()
    for sc in win.itertuples():
        ev = {off: eval_instant(pair, int(sc.tms) + off, "replay", rs) for off in GRID}
        bits = "".join("X" if (ev[o][2] or {}).get("trig") else "." for o in GRID)
        c0 = central_off(rank)
        cen = [o for o in GRID if abs(o - c0) <= 4_000 and (ev[o][2] or {}).get("trig")]
        anyo = [o for o in GRID if (ev[o][2] or {}).get("trig")]
        off = cen[0] if cen else (anyo[0] if anyo else min(GRID, key=lambda o: abs(o - c0)))
        e = int(sc.tms) + off
        bars, src, lg = ev.get(off) or eval_instant(pair, e, "replay", rs)
        tbits = "".join("X" if (eval_instant(pair, int(sc.tms) + o, "truth", None)[2] or {}).get("trig") else "." for o in GRID[::2])
        tb, tsrc, tl = eval_instant(pair, e, "truth", None)
        r = {"scan_t": sc.t, "eval_t": pd.Timestamp(e, unit="ms"), "src": src, "bits": bits, "tbits": tbits,
             "rep_trig": bool(lg and lg["trig"]), "rep_central": bool(cen),
             "rep_fail": (lg or {}).get("fail", "nodata"), "rep_jump": (lg or {}).get("jump"), "rep_vr": (lg or {}).get("vr"),
             "rep_chg": (lg or {}).get("chg"), "rep_prev": (lg or {}).get("rsi_prev"), "cts": (lg or {}).get("cts"),
             "truth_src": tsrc, "truth_trig": "X" in tbits, "truth_fail": (tl or {}).get("fail", "nodata"),
             "truth_jump": (tl or {}).get("jump"), "truth_vr": (tl or {}).get("vr"), "truth_chg": (tl or {}).get("chg"),
             "vol24": D.vol24(pair, e), "held": replay_held(pair, sc.t, seed, OR) if OR is not None else None, "gate": None, "cap": None}
        if lg and lg["trig"]:
            key = lg["cts"]
            if key in seen:
                r["gate"] = "SEEN_THIS_CANDLE"
            else:
                seen.add(key)
                g, info = gates(pair, bars, lg, {"btc_rsi": sc.btc_rsi, "btc_adx": sc.btc_adx, "btc_slope": sc.btc_slope, "tms": sc.tms},
                                e, rs)
                r.update({"gate": g or "PASS", **{f"g_{k}": v for k, v in info.items()}})
                if not g:
                    B = J[seed]["blk"]
                    b = B[(B.pair == pair) & (B.t >= sc.t) & (B.t <= sc.t + pd.Timedelta(seconds=150))]
                    P = J[seed]["sp"]
                    pb = P[(P.t >= sc.t + pd.Timedelta(seconds=25)) & (P.t <= sc.t + pd.Timedelta(seconds=80))
                           & ~P.gate.astype(str).str.startswith("SPIKE_")]
                    r["cap"] = "|".join(sorted(set(b.gate))) if len(b) else None
                    r["cap_pairless"] = "|".join(sorted(set(pb.gate.astype(str)))) if len(pb) else None
        rows.append(r)
    return rows


def classify_live_only(rows, near_rep_min, pair):
    """first leg/gate that differs, per seed."""
    if not rows:
        return "NO_REPLAY_SCANS"
    if pair in BL or pair in NT:
        return "UNIVERSE_BLACKLIST"
    floor = float(TH.get("spike_scanner_min_vol_usd") or 0)
    if all((r["vol24"] == r["vol24"]) and r["vol24"] < floor for r in rows):
        return "UNIVERSE_VOL_FLOOR"
    trig = [r for r in rows if r["rep_trig"] and r["gate"] != "SEEN_THIS_CANDLE"]
    if trig:
        r = trig[0]
        if r["gate"] != "PASS":
            return f"GATE_{r['gate']}" + ("" if r["rep_central"] else "@EDGE")
        if r.get("held"):
            return "CAPACITY_REPLAY_HELD_PAIR"
        if r["cap"]:
            return f"CAPACITY_{r['cap']}"
        if not r["rep_central"]:
            return "SCAN_INSTANT_EDGE"     # replay view triggers only at off-centre decision offsets (1-min view boundary)
        if r.get("cap_pairless"):
            return f"CAPACITY_POSSIBLE_{r['cap_pairless']}"
        return "TRIGGER_PASS_NO_OPEN"
    if any(r["truth_trig"] for r in rows):
        return "DATA_STALE_CANDLE"        # real tape triggers at a replay scan instant, the replay's candle view did not
    if near_rep_min == near_rep_min and near_rep_min <= 60:
        return "FIRED_OTHER_TIME"
    return "SCAN_PHASE"                   # no replay scan instant (any view) falls inside the trigger window


def main():
    X = pd.read_csv(os.path.join(REP, "study_fade_trace_matchset.csv"), low_memory=False)
    X["t"] = pd.to_datetime(X.t)
    J = load_journals()
    OR = YT.load(trim=False, closed_only=False, drop_probes=False, drop_manual=False)
    OR["ta"] = OR.t; OR["tz"] = pd.to_datetime(OR.closed_at.astype(str).str[:23].str.replace("T", " "), format="mixed", errors="coerce").fillna(pd.Timestamp("2027-01-01"))
    OR = OR[(OR.ta >= W0 - pd.Timedelta(days=2)) & (OR.ta < W1)][["seed", "pair", "ta", "tz"]]
    LJ = load_live_journal()
    LJ_scan = LJ[LJ.e == "SCAN"].t.sort_values().values
    LJ0 = LJ.t.min()
    out = []
    # ---------- LIVE fades ----------
    for i, r in X[X.side == "LIVE"].iterrows():
        T = r.t
        live_dec = [T - pd.Timedelta(seconds=s) for s in (2, 5, 8, 12, 20, 30, 40)]
        lt = [eval_instant(r.pair, tms(x), "truth", None)[2] for x in live_dec]
        lr = eval_instant(r.pair, tms(T - pd.Timedelta(seconds=8)), "replay", run_start_for(J, 1, T))
        live_truth_trig = any(l and l["trig"] for l in lt)
        lt0 = next((l for l in lt if l), None)
        cts = (tms(T - pd.Timedelta(seconds=8)) // M5) * M5
        wsec, wfirst = trigger_window(r.pair, cts)
        base = {"side": "LIVE", "pair": r.pair, "t": T, "stack_keep": r.stack_keep, "era": r.era, "pct_live_raw": r.pct_live_raw,
                "pct_live_stack": r.pct_live_stack, "pct_rep_matched": r.pct_rep, "n_seeds": r.n_seeds, "m_seeds": r.m_seeds,
                "near_rep_min": r.near_rep_min, "live_truth_trig": live_truth_trig, "live_truth_fail": (lt0 or {}).get("fail", "nodata"),
                "live_truth_jump": (lt0 or {}).get("jump"), "live_truth_vr": (lt0 or {}).get("vr"), "live_truth_chg": (lt0 or {}).get("chg"),
                "live_truth_prev": (lt0 or {}).get("rsi_prev"),
                "live_rep_src": lr[1], "live_rep_trig": bool(lr[2] and lr[2]["trig"]), "trig_window_s": wsec, "trig_first_s": wfirst,
                "stamp_rsi": r.get("entry_rsi"), "stamp_vr": r.get("entry_pair_volume_ratio"), "stamp_vol24": r.get("entry_pair_volume_24h_usd"),
                "stamp_btc_rsi": r.get("entry_btc_rsi"), "stamp_regime": r.get("entry_btc_regime"), "stamp_adx": r.get("entry_adx")}
        for s in sorted(J):
            ms_ = [x for x in str(r.m_seeds).split(",") if x and x != "nan"]
            rows = trace(r.pair, T, J, s, OR, r.get("entry_pair_rank"))
            matched = str(s) in ms_
            cls = "MATCHED" if matched else classify_live_only(rows, r.near_rep_min, r.pair)
            fr = next((x for x in rows if x["rep_trig"]), rows[0] if rows else {})
            out.append({**base, "seed": s, "class": cls, "n_scans": len(rows),
                        "scan_srcs": "|".join(sorted(set(x["src"].split("(")[0] for x in rows))),
                        "any_truth_trig_at_rep_scan": any(x["truth_trig"] for x in rows),
                        "bits_all": " ".join(f"{x['scan_t'].strftime('%H:%M:%S')}:{x['bits']}/{x['tbits']}" for x in rows),
                        "min_vol24": min((x["vol24"] for x in rows), default=np.nan),
                        **{f"r_{k}": v for k, v in fr.items() if k not in ("scan_t",)}})
        print("LIVE", r.pair, T, out[-1]["class"], "| window", wsec, "| live_truth", live_truth_trig, flush=True)
    # ---------- REPLAY fades ----------
    for i, r in X[X.side == "REPLAY"].iterrows():
        T, s = r.t, int(r.seed)
        rs = run_start_for(J, s, T)
        S = J[s]["scan"]
        prev = S[S.t <= T].tail(2)
        sc = prev.iloc[-1] if len(prev) else None
        e = int(sc.tms) + EVAL_OFF if sc is not None else tms(T) - 10_000
        if e > tms(T):                                    # the decision preceded the open: use the scan before
            sc = prev.iloc[0]; e = int(sc.tms) + EVAL_OFF
        bars, src, lg = eval_instant(r.pair, e, "replay", rs)
        # parity: the trigger at the replay's own decision instant, scanning e-20 s … e+20 s for the minute boundary
        par = [eval_instant(r.pair, e + d, "replay", rs)[2] for d in range(-20_000, 21_000, 5_000)]
        sc_open = S[S.t <= T - pd.Timedelta(seconds=5)].tail(1)
        gbits = "".join("X" if (eval_instant(r.pair, int(sc_open.tms.iloc[0]) + o, "replay", rs)[2] or {}).get("trig") else "."
                        for o in GRID) if len(sc_open) else ""
        rep_trig = any(p and p["trig"] for p in par) or ("X" in gbits)
        _, tsrc, tl = eval_instant(r.pair, e, "truth", None)
        tl_any = any((eval_instant(r.pair, e + d, "truth", None)[2] or {}).get("trig") for d in range(-20_000, 21_000, 5_000))
        cts = (e // M5) * M5
        wsec, wfirst = trigger_window(r.pair, cts)
        th_era = era_thresholds(T)
        g_today, info = gates(r.pair, bars, lg, {"btc_rsi": sc.btc_rsi, "btc_adx": sc.btc_adx, "btc_slope": sc.btc_slope, "tms": sc.tms}, e, rs) \
            if (lg and sc is not None) else ("NA", {})
        g_era, _ = gates(r.pair, bars, lg, {"btc_rsi": sc.btc_rsi, "btc_adx": sc.btc_adx, "btc_slope": sc.btc_slope, "tms": sc.tms}, e, rs, th_era) \
            if (lg and sc is not None and th_era) else ("NA", {})
        # live journal (Sep-28 →): live scan instants in this candle, live truth trigger at them, live blocks for the pair / spike gates
        lj = {}
        if T >= LJ0:
            ls = LJ_scan[(LJ_scan >= np.datetime64(pd.Timestamp(cts, unit="ms") - pd.Timedelta(minutes=2))) &
                         (LJ_scan < np.datetime64(pd.Timestamp(cts + M5, unit="ms")))]
            lt = [eval_instant(r.pair, tms(pd.Timestamp(x)) + EVAL_OFF, "truth", None)[2] for x in ls]
            lb = LJ[(LJ.e == "BLOCK") & (LJ.t >= T - pd.Timedelta(minutes=6)) & (LJ.t <= T + pd.Timedelta(minutes=6))
                    & ((LJ.pair == r.pair) | LJ.gate.astype(str).str.startswith("SPIKE_FADE"))]
            lj = {"lj_scans": len(ls), "lj_truth_trig": any(x and x["trig"] for x in lt),
                  "lj_blocks": "|".join(sorted(set(lb.pair.astype(str) + ":" + lb.gate.astype(str))))}
        out.append({"side": "REPLAY", "pair": r.pair, "t": T, "seed": s, "live_up": r.live_up, "matched_live": r.matched_live,
                    "pct_rep": r.pct_rep, "rep_reason": r.rep_reason, "src": src, "parity_rep_trig": rep_trig, "gbits": gbits,
                    "rep_jump": (lg or {}).get("jump"), "rep_vr": (lg or {}).get("vr"), "rep_chg": (lg or {}).get("chg"),
                    "truth_src": tsrc, "truth_trig": bool(tl and tl["trig"]), "truth_trig_pm20": bool(tl_any),
                    "truth_fail": (tl or {}).get("fail", "nodata"), "truth_jump": (tl or {}).get("jump"), "truth_vr": (tl or {}).get("vr"),
                    "truth_chg": (tl or {}).get("chg"), "trig_window_s": wsec, "trig_first_s": wfirst,
                    "gate_today": g_today or "PASS", "gate_live_era": g_era or "PASS",
                    "era_brsi_max": (th_era or {}).get("spike_fade_max_btc_rsi"), "btc_rsi_scan": None if sc is None else sc.btc_rsi,
                    "stamp_btc_rsi": r.get("entry_btc_rsi"), "stamp_ndi": r.get("entry_pair_1d_ndi"), "calc_ndi": info.get("ndi"),
                    "stamp_btc4h": r.get("entry_btc_4h_ema50_200_gap_pct"), "calc_btc4h": info.get("btc4h_gap"),
                    "stamp_vol24": r.get("entry_pair_volume_24h_usd"), "calc_vol24": info.get("vol24"), **lj})
        print("REPLAY", r.pair, T, s, src, "parity", rep_trig, "truth", bool(tl and tl["trig"]), "era", g_era, flush=True)
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(REP, "study_fade_trace_signals_raw.csv"), index=False)
    print("wrote", len(O))


if __name__ == "__main__":
    main()
