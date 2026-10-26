#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — parity checks (FATAL rule: the cohort must be the engine's own cohort).

① KLINE REBUILD on REAL master fills: the CWOCT path (pair state = engine calculate_indicators on 100 5m bars with the forming bar from
   closed 1m bars; BTC state = study_ml_matched_common.BtcState) is run at the open time of every master non-probe momentum-long fill
   since 2026-09-20 that has the caches, and compared with the fill's own entry stamps and pattern flags (unmatched fills must rebuild
   as unmatched; matched door fills must rebuild their label).
② REPLICA on the same master KEEP fills (control) — exit reason (base names) and % vs the master outcome (DL 380 fidelity re-check).
Output: reports/ML_MATCHED_SHADOW_2026-10-26_parity.csv + printed summary.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_matched_common as M  # noqa: E402

RP, DD = M.RP, M.DD


def k5_pair(pair):
    fr = []
    for fp in (os.path.join(RP.CACHE, "k5m_full", f"{pair}.csv"), os.path.join(DD.PREP, "k5m_full", f"{pair}.csv"),
               os.path.join(M.FETCH, "k5m", f"{pair}.csv")):
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp))
    if not fr:
        return None
    k = pd.concat(fr)
    if "vol" not in k:
        return None
    return k.drop_duplicates("open_time").sort_values("open_time")


def rebuild(pair, o_ms, bs):
    from services.indicators import calculate_indicators
    k5 = k5_pair(pair)
    k1 = M._k1_ext(pair)
    if k5 is None or k1 is None:
        return None
    cur5 = (o_ms // 300_000) * 300_000
    c5 = k5[k5.open_time < cur5].tail(99)
    m = k1[(k1.open_time >= cur5) & (k1.open_time <= o_ms - 60_000)]
    prev1 = k1[k1.open_time <= o_ms - 60_000]
    if len(c5) < 60 or not len(prev1) or (cur5 - c5.open_time.iloc[-1]) > 300_000:
        return None
    px = float(prev1.c.iloc[-1])
    bars = [[int(a), float(o), float(h), float(l), float(c), float(v)] for a, o, h, l, c, v in c5[["open_time", "o", "h", "l", "c", "vol"]].values]
    vol_f = 0.0
    bars.append([int(cur5), float(m.o.iloc[0]) if len(m) else px, float(m.h.max()) if len(m) else px,
                 float(m.l.min()) if len(m) else px, px, vol_f])
    ind = calculate_indicators(bars[-100:])
    if not ind:
        return None
    x = {"ctx_" + k: ind.get(k) for k in ("price", "rsi", "rsi_prev2", "adx", "adx_prev1", "atr", "ema5", "ema13", "ema20", "ema50",
                                           "ema20_prev3", "ema50_prev12", "high_20", "low_20", "volume", "avg_volume")}
    p = M.pair_fields_from_ctx(x)
    s = bs.at(o_ms)
    p.update(s)
    p["label"] = M.labels(p["rng_pos"], p["pair_gap"], p["adx_delta"], s["btc_rsi"], s["btc_rsi_prev"], s["btc_adx"], s["btc_adx_prev"],
                          s["btc_gap"], p["stretch"], p["pair_adx"], s["btc_atr"], p["ema20_slope"], p["ema50_slope"], p["rsi"],
                          p["pair_vol_ratio"])
    return p


def main():
    m = pd.read_csv(os.path.join(M.ROOT, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[(m.direction == "LONG") & m.entry_strategy.astype(str).str.startswith("MOMENTUM") & (m.status == "CLOSED")
          & ~m.is_probe.fillna(False).astype(bool)].copy()
    m["o_ms"] = pd.to_datetime(m.opened_at.astype(str).str[:19]).values.astype("datetime64[ms]").astype(np.int64)
    m = m[m.o_ms >= int(pd.Timestamp("2026-09-20").value // 10**6)]
    tf = lambda v: str(v).lower() == "true"
    bs = M.BtcState()
    rows = []
    for _, r in m.iterrows():
        stamp = "+".join([n for n in M.C_L + M.W_L if tf(r[f"entry_pattern_{n.lower()}_match"])])
        p = rebuild(r.pair, int(r.o_ms), bs)
        if p is None:
            rows.append(dict(pair=r.pair, opened_at=r.opened_at, stamp_label=stamp, rebuilt=False)); continue
        rows.append(dict(pair=r.pair, opened_at=r.opened_at, stamp_label=stamp, rebuilt=True, rebuilt_label=p["label"],
                         d_rsi=p["rsi"] - r.entry_rsi, d_adx=p["pair_adx"] - r.entry_adx, d_rng=p["rng_pos"] - r.entry_range_position,
                         d_gap=p["pair_gap"] - r.entry_pair_ema20_ema50_gap_pct, d_stretch=p["stretch"] - r.entry_ema5_stretch,
                         d_atr=p["atr_pct"] - r.entry_atr_pct, d_btc_rsi=p["btc_rsi"] - r.entry_btc_rsi,
                         d_btc_adx=p["btc_adx"] - r.entry_btc_adx, d_btc_gap=(p["btc_gap"] - r.entry_btc_trend_gap_pct),
                         d_btc_atr=p["btc_atr"] - r.entry_btc_atr_pct))
    P = pd.DataFrame(rows)
    P.to_csv(M.PFX + "_parity.csv", index=False)
    R = P[P.rebuilt]
    print(f"master non-probe ML fills since 09-20: {len(P)} · rebuilt {len(R)}")
    fam = lambda s: ("C" if any(t.startswith("C") for t in str(s).split("+") if t) else "") + ("W" if any(t.startswith("W") for t in str(s).split("+") if t) else "")
    R = R.assign(fs=R.stamp_label.fillna("").map(fam), fr=R.rebuilt_label.fillna("").map(fam))
    print("family parity (stamp vs rebuilt):", round((R.fs == R.fr).mean() * 100, 1), "%  exact label:",
          round((R.stamp_label.fillna("") == R.rebuilt_label.fillna("")).mean() * 100, 1), "%")
    print(pd.crosstab(R.fs.replace("", "none"), R.fr.replace("", "none")))
    for c in [c for c in R.columns if c.startswith("d_")]:
        d = R[c].abs()
        print(f"  {c:12s} median |Δ| {d.median():.4f}  p90 {d.quantile(.9):.4f}")


if __name__ == "__main__":
    main()
