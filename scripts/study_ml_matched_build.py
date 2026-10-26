#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — step 2: build + price every recoverable refused matched long and its controls.

SHADOW cohorts (a refused momentum LONG that matched a C / W pattern — counter LONG_UNMATCHED_ONLY):
  PJUL   live PASS:/FADE LONG_UNMATCHED_ONLY phantoms 2026-07-06 → 07-26 (archived exports; labels + entry stamps recorded live)
  JSEP   live decision journal 2026-09-20 → 09-24 (the only local copy of the server journal: replay/year/journal/calib_live)
  CWOCT  live CloudWatch refusals 2026-10-12 → 10-26 (pair, time, C/W family; price + state rebuilt from klines)
  RPL    engine replay yr6 / yr6a (today's config at 01419d4, 2025-01-01 → 2026-10-11, 3 seeds), BLOCK lines from the journals
CONTROLS (same replica, same fee):
  CTL_LIVE  master KEEP non-probe momentum-long fills (+ B25 provisional) — the unmatched sleeve under today's stack
  CTL_RPL   yr6 / yr6a non-probe MOM-long fills (trimmed to each chunk window)
  MREAL     master non-probe MATCHED momentum longs that really traded (doors / re-enable cells) — descriptive

Labels for JSEP / CWOCT / RPL: the engine's own pattern functions on rebuilt inputs (study_ml_matched_common); PJUL keeps the live
label (entry_cohort) and is re-labelled with today's thresholds for the agreement check.
Post-block gates applied (engine order after LONG_UNMATCHED_ONLY): today's pair blacklist, LONG_HEAT_BLOCK, LONG_MEGACAP_BLOCK
(+ LONG_RSI_MOM_LOADX on PJUL, which predates that signal-side gate). LONG_CHOP_BURST not applied (needs the fill book) → listed.
Pricing: RP.BASE (today's ML exit) at a TAKER entry fee for every shadow AND control (a refused long's order type is unknown);
the control is also priced at its own fee (as in DL 380) for reference.
One shadow position per pair at a time (a refusal while the previous shadow on that pair is open is dropped).

    venv/bin/python -W ignore scripts/study_ml_matched_build.py [--jobs 8]
Outputs: reports/ML_MATCHED_SHADOW_2026-10-26_shadows.csv.gz, _controls.csv.gz
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_matched_common as M  # noqa: E402

RP, DD = M.RP, M.DD
ROOT = M.ROOT
HEAT_TH = SimpleNamespace(long_heat_block_enabled=True, long_heat_btc_slope_min=0.07, long_heat_btc_rsi_prev_min=64.0,
                          long_heat_bull_pct_min=80.0, long_heat_exempt_off30d_max=-10.0, long_megacap_rank_max=10)
LOADX_TH = SimpleNamespace(long_rsi_momentum_adx_max=21.0)


def _ms(s):
    return pd.to_datetime(s, format="mixed").values.astype("datetime64[ms]").astype(np.int64)


def gates(df):
    """today's post-block gates (+ blacklist). df needs pair, btc_slope, btc_rsi_prev, bull, off30d, rank (+ rsi, rsi_prev2, pair_adx for LOADX)."""
    from services.trading_engine import long_heat_eval, long_megacap_block
    from services.indicators import rsi_mom_loadx_block
    import build_master_pool as BMP
    out = []
    for r in df.itertuples():
        g = BMP.stack_pair_blacklist_reason("MOMENTUM", r.pair)
        if not g and long_heat_eval(HEAT_TH, r.btc_slope, r.btc_rsi_prev, r.bull, r.off30d)[1]:
            g = "LONG_HEAT_BLOCK"
        if not g and long_megacap_block(HEAT_TH, r.rank):
            g = "LONG_MEGACAP_BLOCK"
        if not g and getattr(r, "apply_loadx", False) and rsi_mom_loadx_block(LOADX_TH, r.rsi, r.rsi_prev2, r.pair_adx):
            g = "LONG_RSI_MOM_LOADX"
        out.append(g)
    return out


def off30_lookup():
    o = pd.read_csv(os.path.join(ROOT, "reports/btc_off30d_hourly.csv"))
    return dict(zip(o.hour_utc, o.btc_off30d_high_pct))


def rank_lookup():
    u = pd.concat([pd.read_csv(f) for f in (os.path.join(ROOT, "reports/backtest_cache/universe_daily_top50.csv"),
                                            os.path.join(ROOT, "reports/backtest_cache_2025/universe_daily_top50.csv"))])
    u = u.drop_duplicates(["date_ms", "pair"])
    return {(int(d), p): int(r) for d, p, r in zip(u.date_ms, u.pair, u["rank"])}


def prev_day_rank(R, pair, o_ms):
    d = (o_ms // 86_400_000) * 86_400_000
    for dd in (d, d - 86_400_000):    # the universe list dated d is built from day d-1's volume (as the regime study reads it)
        if (dd, pair) in R:
            return R[(dd, pair)]
    return None


# ───────────── cohorts ─────────────
def pjul():
    fs = (glob.glob(os.path.join(ROOT, "reports/phantoms_2026-07-0*.csv")) + [os.path.join(ROOT, "reports/scalpars_phantoms_paper_2026-07-30_13-38-44.csv")]
          + glob.glob(os.path.expanduser("~/Downloads/scalpars_phantoms_paper_2026-07-*.csv")))
    D = pd.concat([pd.read_csv(f, low_memory=False) for f in fs])
    D = D[D.source_filter.astype(str).str.contains("LONG_UNMATCHED_ONLY")].copy()
    D["is_pass"] = D.source_filter.astype(str).str.startswith("PASS:")
    D["k"] = D.pair + "|" + D.entry_at.astype(str).str[:19]
    D = D.sort_values(["is_pass", "pnl_pct"], ascending=[False, True]).drop_duplicates("k")   # PASS row = the long itself
    o = pd.DataFrame(dict(cohort="PJUL", seed=0, pair=D.pair.values, o_ms=_ms(D.entry_at.astype(str).str[:23]),
                          E=D.entry_price.astype(float).values, atr=D.entry_atr_pct.astype(float).values,
                          label_live=D.entry_cohort.astype(str).values,
                          july_pct=np.where(D.is_pass, D.pnl_pct, np.nan), july_exit=np.where(D.is_pass, D.exit_reason, None),
                          btc_rsi=D.entry_btc_rsi.values, btc_rsi_prev=D.entry_btc_rsi_prev.values, btc_adx=D.entry_btc_adx.values,
                          btc_adx_prev=D.entry_btc_adx_prev.values, btc_slope=D.entry_btc_ema20_slope.values, btc_gap=D.entry_btc_trend_gap_pct.values,
                          btc_atr=D.entry_btc_atr_pct.values, btc_1h_slope=D.entry_btc_1h_slope.values, bull=D.entry_bull_pct.values,
                          regime=D.entry_btc_regime.values, rank=D.entry_pair_rank.values, rsi=D.entry_rsi.values, rsi_prev2=D.entry_rsi_prev.values,
                          pair_adx=D.entry_adx.values, adx_delta=D.entry_adx_delta.values, stretch=D.entry_ema5_stretch.values,
                          rng_pos=D.entry_range_position.values, pair_gap=D.entry_pair_ema20_ema50_gap_pct.values,
                          ema20_slope=D.entry_ema20_slope.values, ema50_slope=D.entry_ema50_slope.values,
                          pair_vol_ratio=D.entry_pair_volume_ratio.values, apply_loadx=True))
    o["label"] = [M.labels(r.rng_pos, r.pair_gap, r.adx_delta, r.btc_rsi, r.btc_rsi_prev, r.btc_adx, r.btc_adx_prev, r.btc_gap,
                           r.stretch, r.pair_adx, r.btc_atr, r.ema20_slope, r.ema50_slope, r.rsi, r.pair_vol_ratio) for r in o.itertuples()]
    return o


def journal_rows(files, keep_window=None):
    rows = []
    for f in files:
        p = subprocess.run(["grep", "-E", '"LONG_UNMATCHED_ONLY"|"e":"SCAN"', f], capture_output=True, text=True)
        scan = {}
        for line in p.stdout.splitlines():
            ev = json.loads(line)
            if ev.get("e") == "SCAN":
                scan = ev; continue
            if ev.get("e") != "BLOCK" or ev.get("gate") != "LONG_UNMATCHED_ONLY":
                continue
            r = dict(t=ev["t"], pair=ev.get("pair"), room=ev.get("room"), scan_t=scan.get("t"))
            for k in ("btc_rsi", "btc_adx", "btc_adx_prev", "btc_slope", "off30d", "bull", "bear", "eff", "r72", "above", "off24h", "br_state"):
                r["scan_" + k] = scan.get(k)
            for k, v in (ev.get("ctx") or {}).items():
                r["ctx_" + k] = v
            rows.append(r)
    return pd.DataFrame(rows)


def from_journal(B, cohort, bs):
    """journal BLOCK rows (ctx + SCAN) → shadow rows with rebuilt labels."""
    out = []
    for _, r in B.iterrows():
        ts = r.scan_t if isinstance(r.scan_t, str) else r.t
        s = bs.at(int(pd.Timestamp(ts).value // 10**6))
        p = M.pair_fields_from_ctx(r)
        lab = M.labels(p["rng_pos"], p["pair_gap"], p["adx_delta"], r.scan_btc_rsi, s["btc_rsi_prev"], r.scan_btc_adx, r.scan_btc_adx_prev,
                       s["btc_gap"], p["stretch"], p["pair_adx"], s["btc_atr"], p["ema20_slope"], p["ema50_slope"], p["rsi"], p["pair_vol_ratio"])
        out.append(dict(cohort=cohort, seed=int(r.get("seed", 0) or 0), prefix=r.get("prefix", "live"), pair=r.pair,
                        o_ms=int(pd.Timestamp(r.t).value // 10**6), E=p["price"], atr=p["atr_pct"], label=lab,
                        btc_rsi=r.scan_btc_rsi, btc_rsi_prev=s["btc_rsi_prev"], btc_adx=r.scan_btc_adx, btc_adx_prev=r.scan_btc_adx_prev,
                        btc_slope=r.scan_btc_slope, btc_gap=s["btc_gap"], btc_atr=s["btc_atr"], bull=r.scan_bull, off30d=r.scan_off30d,
                        rsi=p["rsi"], rsi_prev2=p["rsi_prev2"], pair_adx=p["pair_adx"], apply_loadx=False))
    return pd.DataFrame(out)


def cwoct(bs):
    """CloudWatch refusals: entry = last closed 1m close before t; pair state = engine indicators on 100 5m bars (forming from 1m)."""
    from services.indicators import calculate_indicators
    C = pd.read_csv(M.PFX + "_cloudwatch_refusals.csv")
    out = []
    for r in C.itertuples():
        f5, f1 = os.path.join(M.FETCH, "k5m", r.pair + ".csv"), os.path.join(M.FETCH, "k1m", r.pair + ".csv")
        if not (os.path.exists(f5) and os.path.exists(f1)):
            continue
        k5 = pd.read_csv(f5).drop_duplicates("open_time").sort_values("open_time")
        k1 = pd.read_csv(f1).drop_duplicates("open_time").sort_values("open_time")
        cur5 = (r.o_ms // 300_000) * 300_000
        c5 = k5[k5.open_time < cur5].tail(99)
        m = k1[(k1.open_time >= cur5) & (k1.open_time <= r.o_ms - 60_000)]
        prev1 = k1[k1.open_time <= r.o_ms - 60_000]
        if not len(prev1) or len(c5) < 60:
            continue
        px = float(prev1.c.iloc[-1])
        bars = [[int(a), float(o), float(h), float(l), float(c), float(v)] for a, o, h, l, c, v in c5[["open_time", "o", "h", "l", "c", "vol"]].values]
        if len(m):
            bars.append([int(cur5), float(m.o.iloc[0]), float(m.h.max()), float(m.l.min()), px, float(m.vol.sum())])
        else:
            bars.append([int(cur5), px, px, px, px, 0.0])
        ind = calculate_indicators(bars[-100:])
        if not ind:
            continue
        x = {"ctx_" + k: ind.get(k) for k in ("price", "rsi", "rsi_prev2", "adx", "adx_prev1", "atr", "ema5", "ema13", "ema20", "ema50",
                                               "ema20_prev3", "ema50_prev12", "high_20", "low_20", "volume", "avg_volume")}
        p = M.pair_fields_from_ctx(x)
        s = bs.at(r.o_ms)
        lab = M.labels(p["rng_pos"], p["pair_gap"], p["adx_delta"], s["btc_rsi"], s["btc_rsi_prev"], s["btc_adx"], s["btc_adx_prev"],
                       s["btc_gap"], p["stretch"], p["pair_adx"], s["btc_atr"], p["ema20_slope"], p["ema50_slope"], p["rsi"], p["pair_vol_ratio"])
        out.append(dict(cohort="CWOCT", seed=0, prefix="live", pair=r.pair, o_ms=int(r.o_ms), E=px, atr=p["atr_pct"], label=lab,
                        family_log=r.family, conf=r.conf, btc_rsi=s["btc_rsi"], btc_rsi_prev=s["btc_rsi_prev"], btc_adx=s["btc_adx"],
                        btc_adx_prev=s["btc_adx_prev"], btc_slope=s["btc_slope"], btc_gap=s["btc_gap"], btc_atr=s["btc_atr"],
                        bull=None, off30d=None, rsi=p["rsi"], rsi_prev2=p["rsi_prev2"], pair_adx=p["pair_adx"], apply_loadx=False))
    return pd.DataFrame(out)


def family(lab):
    s = str(lab or "")
    c = any(t.startswith("C") for t in s.split("+") if t)
    w = any(t.startswith("W") for t in s.split("+") if t)
    return "C+W" if (c and w) else "C" if c else "W" if w else "NONE"


def to_fills(df, prefix):
    fx = []
    for i, r in df.iterrows():
        if not (pd.notna(r.E) and r.E > 0):
            continue
        rsi = RP.btc_rsi_at(int(r.o_ms))
        fx.append(dict(key=f"{prefix}{i}", pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.get("fee_in", RP.TAKER) or RP.TAKER),
                       atr=float(r.atr) if pd.notna(r.atr) and r.atr > 0 else None, rsi_entry=None if rsi is None else float(rsi)))
    return fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    bs = M.BtcState()
    off30 = off30_lookup()
    R = rank_lookup()

    # ── shadows
    P = pjul(); print("PJUL", len(P), flush=True)
    J = from_journal(journal_rows(sorted(glob.glob(os.path.join(ROOT, "reports/backtest_cache/replay/year/journal/calib_live/decisions-*.jsonl")))),
                     "JSEP", bs); print("JSEP", len(J), flush=True)
    C = cwoct(bs); print("CWOCT", len(C), flush=True)
    RB = pd.read_csv(M.PFX + "_replay_blocks.csv.gz", low_memory=False)
    RPL = from_journal(RB, "RPL", bs); print("RPL", len(RPL), flush=True)
    S = pd.concat([P, J, C, RPL], ignore_index=True)
    S["prefix"] = S.prefix.fillna("live")
    S["off30d"] = [v if pd.notna(v) else off30.get(pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d %H:00")) for v, t in zip(S.off30d, S.o_ms)]
    S["rank"] = [v if pd.notna(v) else prev_day_rank(R, p, t) for v, p, t in zip(S["rank"], S.pair, S.o_ms)]
    S["apply_loadx"] = S.apply_loadx.fillna(False).astype(bool)
    S["gate"] = gates(S)
    S["family"] = S.label.map(family)
    S["fee_in"] = RP.TAKER
    fx = to_fills(S, "S")
    print(f"pricing {len(fx)} shadows", flush=True)
    Rz = M.price(fx, a.jobs).set_index("key")
    S["key"] = [f"S{i}" for i in S.index]
    S = S.join(Rz, on="key")
    # one shadow per pair at a time, per cohort × seed (gated refusals never open → excluded before the episode pass)
    parts = []
    for (co, sd), g in S.groupby(["cohort", "seed"]):
        ok = g[(g.gate == "") & g.pct.notna()]
        e = M.episodes(ok)
        parts.append(e)
    S["episode"] = S.key.isin(pd.concat(parts).key)
    S.to_csv(M.PFX + "_shadows.csv.gz", index=False, compression="gzip")
    print(S.groupby("cohort").agg(n=("key", "size"), priced=("pct", "count"), episodes=("episode", "sum")), flush=True)

    # ── controls
    L = DD.cohort()
    L = L[L.block.isin(["MASTER", "B25"])].copy()
    L["fee_real"] = L.fee_in
    Lx = []
    for r in L.itertuples():
        f = DD.fill_dict(r); f["fee_in"] = RP.TAKER; f["key"] = "L" + r.key; Lx.append(f)
        f2 = DD.fill_dict(r); f2["key"] = "R" + r.key; Lx.append(f2)
    F = pd.read_csv(os.path.join(ROOT, "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False)
    F = F[(F.sleeve == "MOM-long") & (F.status == "CLOSED") & ~F.is_probe.fillna(False).astype(bool)].copy()
    F["o_ms"] = _ms(F.t.astype(str))
    for i, r in F.iterrows():
        rsi = r.entry_btc_rsi_closed if pd.notna(r.entry_btc_rsi_closed) else RP.btc_rsi_at(int(r.o_ms))
        Lx.append(dict(key=f"F{i}", pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price), fee_in=RP.TAKER,
                       atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) and r.entry_atr_pct > 0 else None,
                       rsi_entry=None if rsi is None or pd.isna(rsi) else float(rsi)))
    print(f"pricing {len(Lx)} control legs", flush=True)
    Rc = M.price(Lx, a.jobs).set_index("key")
    L["pct_taker"] = Rc.reindex("L" + L.key).pct.values
    L["pct_realfee"] = Rc.reindex("R" + L.key).pct.values
    L["reason_rep"] = Rc.reindex("L" + L.key).reason.values
    L["o_ms"] = L.o_ms.astype(np.int64)
    m = pd.read_csv(os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
    keep_cols = ["pair", "o_ms", "era", "block", "y", "pct_taker", "pct_realfee", "reason_rep", "reason_live", "atr", "entry_btc_regime",
                 "entry_btc_1h_slope", "entry_bull_pct", "is_door"]
    CL = L[[c for c in keep_cols if c in L.columns]].assign(cohort="CTL_LIVE", seed=0, prefix="live")
    F["pct_taker"] = Rc.reindex([f"F{i}" for i in F.index]).pct.values
    CR = pd.DataFrame(dict(cohort="CTL_RPL", seed=F.seed.values, prefix=F.prefix.values, pair=F.pair.values, o_ms=F.o_ms.values,
                           y=F.pct_new.values, pct_taker=F.pct_taker.values, atr=F.entry_atr_pct.values, entry_btc_regime=F.entry_btc_regime.values,
                           entry_btc_1h_slope=F.entry_btc_1h_slope.values, entry_bull_pct=F.entry_bull_pct.values,
                           cell=F.cell_multiplier_source.values))
    # MREAL: master non-probe matched momentum longs that traded
    mm = m[(m.direction == "LONG") & m.entry_strategy.astype(str).str.startswith("MOMENTUM") & (m.status == "CLOSED")
           & ~m.is_probe.fillna(False).astype(bool)].copy()
    tf = lambda s: s.astype(str).str.lower().eq("true")
    lab = []
    for _, r in mm.iterrows():
        on = [n for n in M.C_L if tf(pd.Series([r[f"entry_pattern_{n.lower()}_match"]])).iloc[0]] + \
             [n for n in M.W_L if tf(pd.Series([r[f"entry_pattern_{n.lower()}_match"]])).iloc[0]]
        lab.append("+".join(on))
    mm["label"] = lab
    mm = mm[mm.label != ""].copy()
    mm["o_ms"] = _ms(mm.opened_at.astype(str).str[:19])
    mx = [dict(key=f"M{i}", pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price), fee_in=RP.TAKER,
               atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) and r.entry_atr_pct > 0 else None,
               rsi_entry=(float(r.entry_btc_rsi_closed) if pd.notna(r.get("entry_btc_rsi_closed")) else RP.btc_rsi_at(int(r.o_ms))))
          for i, r in mm.iterrows()]
    Rm = M.price(mx, a.jobs).set_index("key")
    MR = pd.DataFrame(dict(cohort="MREAL", seed=0, prefix="live", pair=mm.pair.values, o_ms=mm.o_ms.values, era=mm.era.values,
                           label=mm.label.values, y=mm.pnl_percentage.values, y_stack=mm.stack_pct.values, stack_keep=mm.stack_keep.values,
                           pct_taker=Rm.reindex([f"M{i}" for i in mm.index]).pct.values, cell=mm.pattern_cell_source.values,
                           entry_btc_regime=mm.entry_btc_regime.values, atr=mm.entry_atr_pct.values))
    CT = pd.concat([CL, CR, MR], ignore_index=True)
    CT.to_csv(M.PFX + "_controls.csv.gz", index=False, compression="gzip")
    print(CT.groupby("cohort").agg(n=("pair", "size"), priced=("pct_taker", "count")), flush=True)


if __name__ == "__main__":
    main()
