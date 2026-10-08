#!/usr/bin/env python3
"""MOMENTUM LONG sleeve-kill checklist (2026-10-08) — frame builder. Read-only research; no network.

Inputs (all existing, nothing re-fetched):
  reports/NEGFLANK_2D_features.pkl  — every ML fill of both cohorts with the entry_* stamps + 47 kline rebuilds (k_* / d_*),
                                      built by scripts/study_negflank2d_features.py from study_ml_b18_common (master kept non-probe
                                      CLOSED MOMENTUM LONG incl. B1 flagged; yr5 trimmed ML fills, 3 seeds). Verified identical to
                                      today's MASTER_POOL_stacked (10-08c) kept set and stack_pct before use (130/130, max |Δpct| 0).
  reports/backtest_cache/funding/<PAIR>.csv — settled 8 h funding (rebuild: last settled rate at/before entry).
  reports/study_ml_trace_signals.csv + MASTER_POOL_stacked — live AS-TRADED ML fills per batch (BASE…B18) for the period split.
Adds: today's cell size (build_master_pool.today_size_scale — the frozen 10-08c rule), fixed-$3k-book $ at 1× / as-run / today,
cohort tags (cell, PVR class, crowd-sprint, pattern, heat, chop, NEGFLANK, BTC_HOT_MATURE, regime), period labels.
Out: reports/study_ml_checklist_frame.pkl, reports/study_ml_checklist_astraded.csv
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
import study_ml_b18_common as C
import build_master_pool as BMP
import yr5_fills_trimmed as YT

OUT = "reports/study_ml_checklist_frame.pkl"
n = lambda s: pd.to_numeric(s, errors="coerce")


def funding_last(F):
    out = pd.Series(np.nan, index=F.index)
    tms = (F.o.values.astype("datetime64[ms]").astype("int64"))
    for p, idx in F.groupby("pair").groups.items():
        f = f"reports/backtest_cache/funding/{p}.csv"
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f).sort_values("t")
        if not len(x):
            continue
        t = tms[F.index.get_indexer(idx)]
        j = np.searchsorted(x.t.values, t, side="right") - 1
        v = np.where(j >= 0, x.rate.values[np.clip(j, 0, None)], np.nan)
        out.loc[idx] = v
    return out


def main():
    D = pd.read_pickle("reports/NEGFLANK_2D_features.pkl")
    # guard: the pkl's master rows must still equal today's master kept set
    m = C.master(include_b1=True)
    k1 = dict(zip(zip(m.pair, m.opened_at.astype(str)), m.pct)); P = D[D.src == "master"]
    k2 = dict(zip(zip(P.pair, P.opened_at.astype(str)), P.pct))
    assert set(k1) == set(k2) and max(abs(k1[k] - k2[k]) for k in k1) < 1e-9, "feature pkl is stale vs MASTER_POOL_stacked"
    D = D.reset_index(drop=True).copy()
    D["k_funding_last"] = funding_last(D)
    # sizing
    cm = n(D.cell_multiplier).fillna(1.0); cl = n(D.cell_lev_multiplier).fillna(1.0)
    src = D.cell_multiplier_source.fillna("").astype(str)
    f = [BMP.today_size_scale("MOMENTUM", "LONG", s, c, g, b, p, l)
         for s, c, g, b, p, l in zip(src, cm, n(D.entry_global_volume_ratio), n(D.entry_btc_ema20_slope), n(D.entry_pair_volume_ratio), cl)]
    D["m_asrun"] = (cm * cl).clip(lower=0.1)
    D["m_today"] = D.m_asrun * np.array(f)
    BK, BASE = YT.BOOK_USD, YT.BASE_NOTIONAL_FRAC
    D["usd1"] = D.pct / 100 * BASE * BK
    D["usd_asrun"] = D.usd1 * D.m_asrun
    D["usd_today"] = D.usd1 * D.m_today
    # tags
    pvr = n(D.entry_pair_volume_ratio); gvr = n(D.entry_global_volume_ratio); b20 = n(D.entry_btc_ema20_slope)
    D["t_cell"] = src.replace("", "NONE")
    D["t_pvr"] = np.select([pvr < 0.68, pvr < 0.90, pvr >= 0.90], ["QUIET<0.68", "MID", "CROWD>=0.90"], "NA")
    D["t_sprint"] = (gvr > 0.74) & (b20 > 0.07)
    pc = [c for c in D.columns if c.startswith("entry_pattern_") and c.endswith("_match") and "any" not in c]
    D["t_pattern"] = np.where(n(D.entry_pattern_c_any_match) > 0, "C-match", np.where(n(D.entry_pattern_w_any_match) > 0, "W-match", "none"))
    D["t_heat"] = n(D.entry_long_heat_flags).fillna(-1).map(lambda v: "NA" if v < 0 else f"heat{int(v)}")
    eff = n(D.entry_btc_eff72).fillna(n(D.k_btc_eff72))
    D["t_chop"] = np.where(eff <= 0.007, "CHOP(eff72<=.007)", "trend")
    D["t_negflank"] = n(D.entry_btc_1h_slope).fillna(n(D.k_btc_slope1h)) <= -0.05
    adx, atr, ext = n(D.entry_btc_adx), n(D.entry_btc_atr_pct), n(D.entry_btc_dist_from_ema13_pct)
    D["t_hot"] = (adx >= 25) & ((atr >= 0.15) | (ext >= 0.20))
    D["t_regime"] = D.entry_btc_regime.astype(str)
    D["t_order"] = D.entry_order_type.astype(str)
    # periods
    D["month"] = D.o.dt.strftime("%Y-%m")
    D["half"] = np.where(D.o < "2026-05-19", "H1", "H2")      # yr5 calendar mid-point (Jan-04 → Oct-02)
    D["washed"] = D.wash | (D.washed30.fillna(False).astype(bool))
    D["seed"] = D.seed.fillna(0).astype(int)
    D.to_pickle(OUT); print("wrote", OUT, D.shape, D.groupby("src").size().to_dict())

    # live AS-TRADED ML fills per batch (BASE…B16 from the trace, B17/B18 from the master's non-probe rows)
    S = pd.read_csv("reports/study_ml_trace_signals.csv")
    a = S[["era", "pair", "opened_at", "pct_live", "pct_live_stack", "grp"]].rename(columns={"pct_live": "pct_traded"})
    H = C.history_master(); H = H[H.era.isin(["B17", "B18"])]
    b = pd.DataFrame({"era": H.era, "pair": H.pair, "opened_at": H.opened_at.astype(str),
                      "pct_traded": n(H.pnl_percentage), "pct_live_stack": np.where(H.stack_keep.astype(str).str.lower().isin(["true", "1"]), n(H.stack_pct), np.nan),
                      "grp": np.where(H.stack_keep.astype(str).str.lower().isin(["true", "1"]), "kept", "removed")})
    A = pd.concat([a, b], ignore_index=True)
    A["o"] = pd.to_datetime(A.opened_at.astype(str).str[:19].str.replace("T", " ")); A["day"] = A.o.dt.strftime("%Y-%m-%d")
    A.to_csv("reports/study_ml_checklist_astraded.csv", index=False); print("as-traded:", A.groupby("era").size().to_dict())


if __name__ == "__main__":
    main()
