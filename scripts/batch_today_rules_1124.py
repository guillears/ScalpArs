#!/usr/bin/env python3
"""Current paper batch (fills 2026-11-18 → 2026-11-24) re-priced on TODAY's rules — filters + exits + one size (DL 503 / 628: "every
batch read re-prices the whole batch on today's stack"). Research only; no engine / config change.

TODAY's FILTERS — a fill opened before a filter went live (its config commit + 60 min) is re-judged with the ENGINE's own function:
  · EXH_SSC_OIVOL (DL 582, 2026-11-18 15:06 UTC): ssc > cut ∧ oi_vol > cut; oi_vol unstamped → fail-open (the engine's rule)
  · WILLY_HT_NEAR_HIGH (DL 594, 2026-11-19 18:37 UTC): dip below the spike high < frenzy_willy_ht_min_dip_pct — the engine reads the
    DECISION price; the fill price is used here (flagged: the fill sits a few ticks from the decision)
  · FLATBASE_FUNDING_ABOVE_BASE (DL 601, 2026-11-20 17:27 UTC): entry funding stamp > cut (strict); unstamped → fail-open
  · WILLY_HT_BURST_REJECTION (DL 619, 2026-11-22 00:09 UTC): pre_vol15_vs60 + cloc15 on the 75 Binance 1m bars before the open minute
    (route_b_signals.today_stack, cached); unreadable → fail-open
TODAY's EXITS (route_b_signals.common.exit_specs from trading_config.json): a fill whose sleeve exit changed after it opened is re-walked on
its own 1m path from the REAL fill price (route_b_signals.exits.walk: stop on a touch, worse of level and minute close; TP = resting maker
0.05 % through; taker fee + 0.10 % exit slip on stops / time exits; real settled funding; FLATBASE stop counts funding paid). Unchanged
exits keep the BOOKED result. Changes: EXH TP cap 10 → 7 and HT TP +5 → +6 (2026-11-21 20:59 UTC), FLATBASE stop counts funding
(2026-11-21 02:43 UTC); HT ran WILLY's exit before 2026-11-18 22:56 UTC. DIST and plain WILLY exits unchanged in the batch.
ONE SIZE: every fill at today's size X = 0.61 × equity (0.5 invest × 5× for every sleeve in the batch) — r_d = Σ fill % × X ÷ 100 over
fills closed that UTC day, DCR over every calendar day 11-18 → 11-24 (scripts/best_case/one_size.dcr). "As traded" $ = the export's pnl.
Usage: venv/bin/python -I scripts/batch_today_rules_1124.py <orders export csv>   → reports/BATCH_TODAY_RULES_2026-11-24_{fills,summary,changes,days}.csv
"""
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "route_b_signals"))
sys.path.insert(0, os.path.join(HERE, "best_case"))
sys.path.insert(0, ROOT)
import common as C            # noqa: E402
import exits as X             # noqa: E402
import sources as SRC         # noqa: E402
import today_stack as TS      # noqa: E402
from build import pessimistic_fill_minute   # noqa: E402
import one_size as OS         # noqa: E402
from services.watch_filters import ht_dip_pct   # noqa: E402

UTC = lambda s: pd.Timestamp(s, tz="UTC")
MARGIN = pd.Timedelta(minutes=60)
X_SIZE = 0.61
ACCOUNT = 10_000.0
EXIT_SLIP = 0.10
EXIT_CHANGED_BEFORE = {"EXHAUSTION_SHORT": UTC("2026-11-21 20:59:49"), "FRENZY_WILLY_HT": UTC("2026-11-21 20:59:49"),
                       "FLATBASE_SHORT": UTC("2026-11-21 02:43:31")}
FILTERS = [("EXH_SSC_OIVOL", "EXHAUSTION_SHORT", UTC("2026-11-18 15:06:14")),
           ("WILLY_HT_NEAR_HIGH", "FRENZY_WILLY_HT", UTC("2026-11-19 18:37:02")),
           ("FLATBASE_FUNDING_ABOVE_BASE", "FLATBASE_SHORT", UTC("2026-11-20 17:27:18")),
           ("WILLY_HT_BURST_REJECTION", "FRENZY_WILLY_HT", UTC("2026-11-22 00:09:59"))]
SHORT = {"EXHAUSTION_SHORT": "EXH", "DISTRIBUTION_SHORT": "DIST", "FLATBASE_SHORT": "FB", "FRENZY_WILLY_HT": "HT", "FRENZY_WILLY": "WILLY"}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def filter_check(name, r, th, now_ms):
    """(state, text) zone = today refuses · rest · unread (fail-open)."""
    if name == "EXH_SSC_OIVOL":
        ssc, oiv = _num(r.entry_exh_spot_share_chg), _num(r.entry_exh_oi_vol)
        if ssc is None or oiv is None:
            return "unread", "oi_vol / ssc unstamped → fail-open"
        hit = ssc > float(th["exhaustion_ssc_oivol_ssc_min"]) and oiv > float(th["exhaustion_ssc_oivol_oivol_min"])
        return ("zone" if hit else "rest"), f"ssc {ssc:+.4f} · oi_vol {oiv:.3f}"
    if name == "WILLY_HT_NEAR_HIGH":
        dip = ht_dip_pct(r.entry_price, r.entry_frenzy_willy_spike_high)
        if dip is None:
            return "unread", "spike high unstamped → fail-open"
        cut = float(th["frenzy_willy_ht_min_dip_pct"])
        return ("zone" if dip < cut else "rest"), f"dip {dip:.2f} % vs {cut:g} % (fill price)"
    if name == "FLATBASE_FUNDING_ABOVE_BASE":
        st, txt = TS.fb_funding({"journal_funding_rate": r.entry_funding_rate}, th)
        return st, txt
    ms = int(UTC(r.opened_at[:19]).value // 1_000_000)
    rows = TS.fetch_bars_1m(r.pair, ms, now_ms, os.path.join(C.DATA, "today_stack"))
    if rows is None:
        return "nofetch", "1m bars not fetched"
    return TS.ht_burst(rows, ms)


def rewalk(r, spec, now_ms):
    side = 1 if r.direction == "LONG" else -1
    f_ms = int(UTC(r.opened_at[:19]).value // 1_000_000)
    m0 = f_ms // C.MIN_MS * C.MIN_MS
    k = SRC.klines(r.pair, m0, m0 + (spec["hold_min"] + 2) * C.MIN_MS, now_ms)
    if not len(k) or int(k.open_ms.iloc[0]) != m0:
        return None
    k = pessimistic_fill_minute(k.reset_index(drop=True), 0, float(r.entry_price), side)
    atr = _num(r.entry_exh_atr_pct) if spec["kind"] == "atr" else None
    tp, sl = C.levels_for(spec, atr)
    fund, ok = SRC.funding_history(r.pair, f_ms, f_ms + spec["hold_min"] * C.MIN_MS, now_ms)
    w = X.walk(k.o.to_numpy(), k.h.to_numpy(), k.l.to_numpy(), k.c.to_numpy(), k.open_ms.to_numpy(), float(r.entry_price), side, tp, sl,
               spec["hold_min"], funding=fund, stop_counts_funding=spec["stop_counts_funding"], entry_ms=f_ms)
    net = w["net_pct"] - (0.0 if w["reason"] == "TP" else EXIT_SLIP)
    return dict(today_reason=w["reason"], today_pct=net, today_final=bool(w["final"]) and ok, today_closed_at=pd.Timestamp(int(w["exit_ms"]) + C.MIN_MS, unit="ms", tz="UTC"),
                today_tp=tp, today_sl=sl)


def stats(p):
    p = pd.Series(p, dtype=float)
    if not len(p):
        return dict(N=0, WR=np.nan, avg=np.nan, sum=0.0)
    return dict(N=len(p), WR=(p > 0).mean() * 100, avg=p.mean(), sum=p.sum())


def main(path):
    th = C.load_thresholds()
    specs = C.exit_specs(th)
    now_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
    d = pd.read_csv(path)
    d = d[(d.status == "CLOSED") & ~d.get("is_probe", pd.Series(False, index=d.index)).fillna(False).astype(bool)].sort_values("opened_at")
    d = d.reset_index(drop=True)
    rows = []
    for r in d.itertuples():
        t_open = UTC(r.opened_at[:19])
        sl = r.entry_strategy
        out = dict(sleeve=SHORT.get(sl, sl), pair=r.pair, opened_at=r.opened_at[:16], closed_at=str(r.closed_at)[:19], booked_reason=r.close_reason,
                   booked_pct=float(r.pnl_percentage), booked_usd=float(r.pnl), filter_block="", filter_notes="", exit_rewalked=False)
        notes = []
        for name, fsl, armed in FILTERS:
            if fsl != sl or not t_open < armed + MARGIN:
                continue
            st, txt = filter_check(name, r, th, now_ms)
            notes.append(f"{name}: {st} ({txt})")
            if st == "zone" and not out["filter_block"]:
                out["filter_block"] = name
        out["filter_notes"] = " | ".join(notes)
        chg = EXIT_CHANGED_BEFORE.get(sl)
        if chg is not None and t_open < chg + MARGIN and specs.get(sl):
            w = rewalk(r, specs[sl], now_ms)
            if w is None:
                out.update(today_reason="NO_PATH", today_pct=np.nan, today_final=False, today_closed_at=out["closed_at"])
            else:
                out.update(w, exit_rewalked=True)
        else:
            out.update(today_reason=r.close_reason, today_pct=float(r.pnl_percentage), today_final=True, today_closed_at=str(r.closed_at)[:19])
        rows.append(out)
    F = pd.DataFrame(rows)
    F["kept"] = F.filter_block == ""
    F["today_closed_at"] = F.today_closed_at.astype(str)
    os.makedirs(C.REPORTS, exist_ok=True)
    F.to_csv(os.path.join(C.REPORTS, "BATCH_TODAY_RULES_2026-11-24_fills.csv"), index=False)
    t0, t1 = UTC(d.opened_at.min()[:19]), max(UTC(str(d.closed_at.max())[:19]), pd.to_datetime(F.today_closed_at, utc=True, format="mixed").max())
    order = ["HT", "EXH", "DIST", "FB", "WILLY"]
    tab, chg = [], []
    for s in order + ["ALL"]:
        g = F if s == "ALL" else F[F.sleeve == s]
        if not len(g):
            continue
        b, k = stats(g.booked_pct), stats(g[g.kept].today_pct)
        dcr_b = OS.dcr(g.closed_at, OS.frac(g.booked_pct, X_SIZE), np.ones(len(g), bool), t0, t1)
        dcr_a = OS.dcr(g.today_closed_at, OS.frac(g.today_pct, X_SIZE), g.kept.to_numpy(), t0, t1)
        kp = g[g.kept].today_pct
        best = kp.max() if len(kp) else np.nan
        conc = (g[g.kept].groupby("pair").today_pct.sum().abs().max() / kp.abs().sum() * 100) if len(kp) and kp.abs().sum() else np.nan
        tab.append(dict(sleeve=s, b_N=b["N"], b_WR=b["WR"], b_avg=b["avg"], b_sum=b["sum"], b_usd_traded=g.booked_usd.sum(),
                        b_usd_one=b["sum"] * X_SIZE * ACCOUNT / 100, b_dcr=dcr_b, a_N=k["N"], a_WR=k["WR"], a_avg=k["avg"], a_sum=k["sum"],
                        a_usd_one=k["sum"] * X_SIZE * ACCOUNT / 100, a_dcr=dcr_a, a_sum_wo_best=(k["sum"] - best) if len(kp) else np.nan,
                        top_pair_share=conc))
        blk = g[~g.kept]
        rw = g[g.kept & g.exit_rewalked]
        chg.append(dict(sleeve=s, blocked_N=len(blk), blocked_booked_sum=blk.booked_pct.sum(), blocked_today_sum=blk.today_pct.sum(),
                        blocked_by=", ".join(f"{k_} {v_}" for k_, v_ in blk.filter_block.value_counts().items()) or "–",
                        rewalked_N=len(rw), rewalked_booked_sum=rw.booked_pct.sum(), rewalked_today_sum=rw.today_pct.sum(),
                        not_final=int((~F.loc[g.index, "today_final"].astype(bool)).sum())))
    T, Cg = pd.DataFrame(tab), pd.DataFrame(chg)
    T.to_csv(os.path.join(C.REPORTS, "BATCH_TODAY_RULES_2026-11-24_summary.csv"), index=False)
    Cg.to_csv(os.path.join(C.REPORTS, "BATCH_TODAY_RULES_2026-11-24_changes.csv"), index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    print(Cg.round(2).to_string(index=False))
    # per UTC close day: before (booked, as traded $ + one-size % of the account) vs after (today's rules, one size)
    F["b_day"] = pd.to_datetime(F.closed_at, utc=True, format="mixed").dt.strftime("%m-%d")
    F["a_day"] = pd.to_datetime(F.today_closed_at, utc=True, format="mixed").dt.strftime("%m-%d")
    days = sorted(set(F.b_day) | set(F.a_day))
    dd = []
    for day in days:
        b = F[F.b_day == day]
        a = F[(F.a_day == day) & F.kept]
        dd.append(dict(day=day, b_N=len(b), b_usd_traded=b.booked_usd.sum(), b_pct_one=(b.booked_pct * X_SIZE).sum(),
                       a_N=len(a), a_pct_one=(a.today_pct * X_SIZE).sum(), a_usd_one=(a.today_pct * X_SIZE).sum() * ACCOUNT / 100))
    D = pd.DataFrame(dd)
    D.to_csv(os.path.join(C.REPORTS, "BATCH_TODAY_RULES_2026-11-24_days.csv"), index=False)
    print(D.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
