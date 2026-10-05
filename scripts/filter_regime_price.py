#!/usr/bin/env python3
"""FILTER x BTC-REGIME MATRIX (2026-10-05) step 2 — build the momentum-LONG refusal signals from the extract (filter_regime_extract.py)
and price each with the live momentum-LONG exit replica ON LOCAL DATA (scripts/ml_exit_optimize_yr5.py BASE via run_fill: local tick
archive, 1m fallback; entry = signal t + 60 s, taker in; ATR % = Wilder-14 on the closed 5m bars before entry from k5m_full, 1m-built
fallback; BTC RSI entry = the replica's live ruler). Read-only research.

Signal definition
  ladder + macro gates  = journal FAILS lines whose COMPLETE fail set is one gate (sole blockers; MACRO:<gate> = ladder PASS refused
                          only by the scan's FIRST macro veto)
  engine-chain gates    = journal BLOCK lines (the candidate passed the ladder; the gate is the FIRST chain gate that refused it —
                          later chain gates are not evaluated: necessary, not sufficient)
  one signal per gate x pair per 30 min (chained) per seed; de-duplicated across seeds by 5-min bucket (n_seeds kept).
  Gates with more than CAP signals are sampled down to CAP (uniform, fixed seed) — the day is the unit downstream.
Out: reports/FILTER_REGIME_MATRIX_signals_priced.csv
"""
import os, sys, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
S = os.environ.get("S", "/tmp")
OUT = os.path.join(ROOT, "reports", "FILTER_REGIME_MATRIX_signals_priced.csv")
CAP = 8000
CHAIN = {'FAN_RATIO_GATE', 'PAIR_ATR_MIN', 'PAIR_EMA_GAP_NOT_EXPANDING', 'PAIR_NO_TRADE', 'ADX_DELTA_BTC_ADX_CROSS', 'BTC_RSI_ATR_COND',
         'VOL_GATE', 'LONG_BTC1H_DEADBAND', 'BTC_ACCEL_CHASE_LONG', 'BTC_GAP_BTC_ADX_CROSS', 'LONG_UNMATCHED_ONLY', 'RNGPOS_ADX_DELTA_CROSS',
         'ENTRY_QUALITY_SCORE', 'RSI_SPIKE_GUARD', 'PAIR_RSI_ADX_CROSS', 'CALM3D_DMI', 'CALM3D_REENTRY', 'CALM3D_BTC_ATR_MIN',
         'LONG_MEGACAP_BLOCK', 'LONG_HEAT_BLOCK', 'LONG_CHOP_BURST', 'PAIR_ATR_MAX', 'BTC_SLOPE_MAX_GATE'}


def build_signals():
    B = pd.read_pickle(f"{S}/frm_block.pkl"); F = pd.read_pickle(f"{S}/frm_fails.pkl")
    b = B[B.gate.isin(CHAIN) & B.pair.notna()][["seed", "t", "gate", "pair", "room"]].copy()
    b["kind"] = "BLOCK(chain)"
    f = F[F.pair.notna() & (F.pair != "?") & (F.src == "MOMENTUM")].copy()
    f["gate"] = f.gates; f["room"] = True; f["kind"] = "FAILS(sole)"
    A = pd.concat([b, f[["seed", "t", "gate", "pair", "room", "kind"]]]).sort_values("t")
    A["gap"] = A.groupby(["seed", "gate", "pair"]).t.diff()
    E = A[A.gap.isna() | (A.gap > 30 * 60_000)].copy()
    E["bucket"] = E.t // 300_000 * 300_000
    U = E.groupby(["gate", "pair", "bucket"]).agg(t=("t", "min"), n_seeds=("seed", "nunique"), room=("room", "mean"),
                                                   kind=("kind", "first")).reset_index()
    n_all = U.groupby("gate").size().rename("n_signals_all")
    parts = []
    for g, d in U.groupby("gate"):
        parts.append(d.sample(CAP, random_state=20261005) if len(d) > CAP else d)
    V = pd.concat(parts).merge(n_all, left_on="gate", right_index=True)
    return V


_K5H = {}


def _atr(M, pair, ms):
    from ta.volatility import AverageTrueRange
    if pair not in _K5H:
        _K5H.clear()
        fp = f"{ROOT}/reports/backtest_cache/k5m_full/{pair}.csv"
        k = pd.read_csv(fp) if os.path.exists(fp) else None
        if k is None:
            k1 = M._k1(pair)
            if k1 is not None and len(k1):
                k = k1.assign(b=k1.open_time // 300_000 * 300_000).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"),
                                                                                    n=("o", "size")).reset_index().rename(columns={"b": "open_time"})
                k = k[k.n == 5]
        _K5H[pair] = k.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True) if k is not None else None
    k = _K5H[pair]
    if k is None:
        return None
    j = np.searchsorted(k.open_time.values, ms - 300_000, side="right")
    g = k.iloc[max(0, j - 200):j]
    if len(g) < 30:
        return None
    a = AverageTrueRange(high=g.h, low=g.l, close=g.c, window=14).average_true_range()
    v = float(a.iloc[-1] / g.c.iloc[-1] * 100)
    return v if np.isfinite(v) and v > 0 else None


def work(rows):
    import ml_exit_optimize_yr5 as M
    out = []
    for pair, t in rows:
        e = int(t) + 60_000
        try:
            bp = M.build_path(pair, e, 2 * 3600_000)
            if bp is None or not len(bp[0]):
                out.append((pair, t, np.nan, "NO_PATH", "none", np.nan)); continue
            fx = dict(key="x", pair=pair, o_ms=e, E=float(bp[1][0]), fee_in=M.TAKER, atr=_atr(M, pair, e), rsi_entry=M.btc_rsi_at(e))
            k, why, pct, tex, src, pk = M.run_fill(fx, {"BASE": M.BASE})["BASE"]
            out.append((pair, t, float(pct), str(why), src, (tex - e) / 60_000 if tex else np.nan))
        except Exception as ex:
            out.append((pair, t, np.nan, f"ERR:{str(ex)[:40]}", "err", np.nan))
    return out


def main():
    V = build_signals()
    K = V[["pair", "t"]].drop_duplicates()
    have = pd.read_csv(OUT) if os.path.exists(OUT) else None
    done = {}
    if have is not None:
        done = {(p, int(t)): (x, w, s, h) for p, t, x, w, s, h in zip(have.pair, have.t, have.pct, have.why, have.src, have.hold_min)}
    todo = [(p, int(t)) for p, t in zip(K.pair, K.t) if (p, int(t)) not in done]
    todo.sort()
    print(len(V), "signals", len(K), "unique pair-times", len(todo), "to price", flush=True)
    t0 = time.time()
    n = 400
    batches = [todo[i:i + n] for i in range(0, len(todo), n)]
    with ProcessPoolExecutor(9) as ex:
        for i, res in enumerate(ex.map(work, batches)):
            for pair, t, pct, why, src, h in res:
                done[(pair, t)] = (pct, why, src, h)
            if i % 25 == 0:
                print(i, len(batches), f"{time.time() - t0:.0f}s", flush=True)
    V["pct"] = [done[(p, int(t))][0] for p, t in zip(V.pair, V.t)]
    V["why"] = [done[(p, int(t))][1] for p, t in zip(V.pair, V.t)]
    V["src"] = [done[(p, int(t))][2] for p, t in zip(V.pair, V.t)]
    V["hold_min"] = [done[(p, int(t))][3] for p, t in zip(V.pair, V.t)]
    V.to_csv(OUT, index=False)
    print("coverage", V.pct.notna().mean(), V.src.value_counts().to_dict())


if __name__ == "__main__":
    main()
