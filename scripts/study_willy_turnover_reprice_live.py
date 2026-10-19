#!/usr/bin/env python3
"""📊 2026-10-19 research (WILLY turnover filter re-price) — LIVE side. READ-ONLY on code / config.

① the turnover-BLOCKED WILLY entries (reports/SCOUT_WILLY_TURNOVER_BLOCKED.csv, gate FRENZY_WILLY_TURNOVER) re-priced under TODAY's full live
   WILLY exit (trading_config.json: +1.25 / −4 / 90 / late +0.25 from min 30 / resting maker TP → scout_willy_timecap.evaluate_late
   book="live" on study_b19_fzexit_common.load_path, ticks first) — entry = the first aggTrades print at / after the refusal second (the
   scout's ruler, no slip; a +0.035 % slip variant too). mcap = 24 h futures quote volume (Σ the 24 closed 1h klines before the refusal,
   public fapi /fapi/v1/klines weight 1, cached in reports/WILLY_TURNOVER_REPRICE_2026-10-19_vol24cache.json; shared cooldown honoured,
   X-MBX-USED-WEIGHT-1M read, stop on 403 / 418 / 429) ÷ R (the bot's own two inputs — as DECISION_LOG 314).
② every closed FRENZY_WILLY fill since the DECISION_LOG 251 deploy (BASELINE19–22 + the newest ~/Downloads live export) re-walked the same
   way from its booked entry → the let-through (kept) live cohort + walker-vs-booked parity.
Output: reports/WILLY_TURNOVER_REPRICE_2026-10-19_live.csv
Usage: venv/bin/python -I scripts/study_willy_turnover_reprice_live.py
"""
import glob, json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_b19_fzexit_common as FX          # noqa: E402
import binance_cooldown as BC                 # noqa: E402
WT = FX.WT
MIN = 60_000
TH = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
TP = float(TH["frenzy_willy_tp_pct"]); SL = abs(float(TH["frenzy_willy_stop_pct"])); CAP = int(TH["frenzy_willy_max_hold_minutes"])
LATE_M = float(TH["frenzy_willy_late_tp_minutes"]); LATE_P = float(TH["frenzy_willy_late_tp_pct"])
BOOK = "live" if TH.get("resting_tp_willy_enabled") else "print"
VCACHE = os.path.join(ROOT, "reports", "WILLY_TURNOVER_REPRICE_2026-10-19_vol24cache.json")
OUT = os.path.join(ROOT, "reports", "WILLY_TURNOVER_REPRICE_2026-10-19_live.csv")
DEPLOY_251 = "2026-10-08T15:23"     # 84318b3 15:13:12 UTC + 10 min (scout convention)


def ms(s):
    return int(pd.Timestamp(str(s)[:19]).tz_localize("UTC").value // 1_000_000)


def walk(pair, te, E, tc):
    P = FX.load_path(pair, te, E, CAP, tc)
    if not len(P["t"]):
        return dict(cf_why="UNRES", cf_pct=np.nan, cf_path="none")
    ev = WT.evaluate_late(P, TP, SL, CAP, LATE_M, LATE_P, book=BOOK)
    return dict(cf_why=ev["why"], cf_pct=float(ev["pct"]) if ev["why"] != "UNRES" else np.nan, cf_min=ev["xmin"],
                cf_path="ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m"), cf_on_bar=ev.get("on_bar"))


def vol24(pair, t_ms, cache, used):
    k = f"{pair}|5m|{(t_ms // 300_000) * 300_000}"   # review 10-19: 5m window (the 1h one left out the spike hour); old 1h keys ignored
    if k in cache:
        return cache[k]
    if used.get("stop") or BC.active():
        used["stop"] = True
        return None
    end = (t_ms // 300_000) * 300_000
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=5m&startTime={end - 288 * 300_000}&endTime={end - 1}&limit=288"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows = json.loads(r.read().decode())
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
            used["w"] = w
            if w >= 700:
                used["stop"] = True
    except urllib.error.HTTPError as e:
        if e.code in (403, 418, 429):
            used["stop"] = True
        return None
    except Exception:
        return None
    time.sleep(0.3)
    v = float(sum(float(x[7]) for x in rows)) * 288.0 / len(rows) if len(rows) >= 240 else None
    cache[k] = v
    return v


def main():
    tc = {}
    # ① blocked
    B = pd.read_csv(os.path.join(ROOT, "reports", "SCOUT_WILLY_TURNOVER_BLOCKED.csv"))
    B = B[B.gate == "FRENZY_WILLY_TURNOVER"].sort_values("t").reset_index(drop=True)
    cache = json.load(open(VCACHE)) if os.path.exists(VCACHE) else {}
    used = {}
    rows = []
    for r in B.itertuples():
        t = ms(r.t)
        d = f"{pd.Timestamp(t, unit='ms'):%Y-%m-%d}"
        z = WT._load_day(r.pair, d, tc)
        E = te = None
        if z is not None:
            i = int(np.searchsorted(z[0], t, side="left"))
            if i < len(z[0]) and z[0][i] - t <= 60_000:
                E, te = float(z[1][i]), int(z[0][i]) - 1
        v = vol24(r.pair, t, cache, used)
        o = dict(kind="blocked", t=str(r.t)[:19], pair=r.pair, trig=r.trig, R=float(r.R), scout_pct=r.pct, scout_how=r.how, vol24=v,
                 mcap=(v / float(r.R)) if v else np.nan, E=E)
        if E is not None:
            o.update(walk(r.pair, te, E, tc))
            w2 = walk(r.pair, te, E * 1.00035, tc)
            o.update(cf_pct_slip=w2["cf_pct"])
        rows.append(o)
    json.dump(cache, open(VCACHE, "w"), indent=0)
    print("binance used-weight last", used.get("w"), "stop", used.get("stop"))
    # ② live WILLY fills
    fs = sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE19_*orders*.csv")) + glob.glob(os.path.join(ROOT, "reports", "BASELINE2[0-9]_*orders*.csv")))
    fs += sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_live_*.csv")))[-1:]
    fr = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False)
        d = d[(d.entry_strategy.astype(str) == "FRENZY_WILLY") & (d.status == "CLOSED")]
        fr.append(d.assign(src_file=os.path.basename(f)))
    W = pd.concat(fr, ignore_index=True)
    W = W[W.opened_at.astype(str).str[:16] >= DEPLOY_251]
    W["k"] = W.opened_at.astype(str).str[:19] + W.pair
    W = W.drop_duplicates("k", keep="last").sort_values("opened_at")
    for r in W.itertuples():
        te = ms(r.opened_at)
        o = dict(kind="fill", t=str(r.opened_at)[:19], pair=r.pair, trig=getattr(r, "entry_frenzy_willy_trigger", None),
                 R=pd.to_numeric(r.entry_vol_mcap_ratio, errors="coerce"), mcap=pd.to_numeric(r.entry_mcap_usd, errors="coerce"),
                 vol24=pd.to_numeric(r.entry_pair_volume_24h_usd, errors="coerce"), E=float(r.entry_price), booked_pct=float(r.pnl_percentage),
                 booked_why=r.close_reason, is_paper=r.is_paper, src_file=r.src_file, wait_bars=getattr(r, "entry_frenzy_willy_wait_bars", None))
        o.update(walk(r.pair, te, float(r.entry_price), tc))
        rows.append(o)
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
