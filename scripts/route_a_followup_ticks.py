#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP (PREREG reports/ROUTE_A_FOLLOWUP_2026-11-19_PREREG.md) — tick pass for Q2 and Q3.
For every sampled event (reports/ROUTE_A_TICKS_data/ev/*.npz) and checkpoint C0 / C1 / C2 / C5:
 Q2  PATH labels +X % before −X % (X = 1, 2, 3, 5) from the entry (first trade ≥ decision + 8 s): 1 s bars from ticks → 1m bars
     from ticks → local 5m (route_a_ticks_features.path_bars), 24 h, same bar both = loss.
 Q3  the 4 TP × 4 stop × 3 hold scalp grid at slip 0.10 / 0.20 / 0.30 %, exact trade order on ticks until sig + 10 min, then 1m
     bars from ticks (then 5m); same bar both = stop (primary) and TP (bound); maker TP strictly through by 1 tick; real funding.
 NULL the same grid at the 1m convention (entry = close of the decision minute's bar × (1 + 0.10 %), path from the next bar) for the
     observed decision minute AND 300 random minutes on the same coin-day (frozen seed).
Output reports/ROUTE_A_FOLLOWUP_data/ticks.pkl. Research only, local files, 0 API weight."""
import glob, itertools, json, os, pickle, sys
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import route_a_ticks_features as RF   # noqa: E402  (path_bars, TB with the 5m / funding sources)
TB = RF.TB
SRC = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); EV = os.path.join(SRC, "ev")
D = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_data"); os.makedirs(D, exist_ok=True)
SEC, MIN, H1, DAY = 1000, 60_000, 3_600_000, 86_400_000
CHK = (0, 60, 120, 300)
LAT = 8 * SEC
TPS = (0.5, 1.0, 1.5, 2.0); SLS = (0.5, 1.0, 2.0, 3.0); HOLDS = (5, 15, 60)
CELLS = list(itertools.product(TPS, SLS, HOLDS))
SLIPS = (0.10, 0.20, 0.30)
XS = (1, 2, 3, 5)
NNULL = 300
TICKS = json.load(open(os.path.join(R, "reports", "backtest_cache_2025", "exchange_ticks.json")))
G = {}


def tick_of(pair, p):
    if pair in TICKS and TICKS[pair] > 0:
        return float(TICKS[pair])
    u = np.unique(p[:20000]); d = np.diff(u); d = d[d > 1e-15]
    return float(d.min()) if len(d) else float(p[0]) * 1e-4


def fund_cost(ft, rates, a, b):
    if not len(ft):
        return 0.0
    i = int(np.searchsorted(ft, a, side="right")); j = int(np.searchsorted(ft, b, side="right"))
    return float(rates[i:j].sum())


def first(mask):
    k = int(np.argmax(mask)) if len(mask) else 0
    return k if len(mask) and mask[k] else -1


def grid_ticks(tt, pp, j, bars_after, P0, slip, tick, ft, rates):
    """scalp grid from entry index j. tt/pp ticks (cut at tick_end); bars_after = (start, o, h, l, c) 1m/5m bars after the ticks.
    returns net primary (48), net TP-first bound (48), kind (48: 1 TP, 2 SL, 3 TIME, 0 incomplete), where (48: 1 tick, 2 bar)"""
    s = slip / 100
    E = P0 * (1 + s); ent = int(tt[j])
    tseg_t = tt[j:]; tseg_p = pp[j:]
    bs, bo, bh, bl, bc, bend = bars_after
    out_n = np.full(len(CELLS), np.nan); out_b = np.full(len(CELLS), np.nan)
    out_k = np.zeros(len(CELLS), np.int8); out_w = np.zeros(len(CELLS), np.int8)
    for ci, (tp, sl, hd) in enumerate(CELLS):
        end = ent + hd * MIN
        tpx = np.ceil(E * (1 + tp / 100) / tick - 1e-9) * tick; fill_lvl = tpx + tick * 0.999
        slx = E * (1 - sl / 100)
        inwin = tseg_t < end
        kt = first((tseg_p >= fill_lvl) & inwin); ks = first((tseg_p <= slx) & inwin)
        res = None
        if kt >= 0 or ks >= 0:
            if ks < 0 or (kt >= 0 and kt < ks):
                res = ((tpx / E - 1) * 100 - 0.05 - 0.02, 1, int(tseg_t[kt]), 1); bnd = res
            else:
                xp = min(slx, tseg_p[ks]) * (1 - s)
                res = ((xp / E - 1) * 100 - 0.10, 2, int(tseg_t[ks]), 1); bnd = res
        elif tseg_t[-1] >= end or not len(bs):
            k = first(tseg_t >= end)
            xp = (tseg_p[k] if k >= 0 else tseg_p[-1]) * (1 - s)
            res = ((xp / E - 1) * 100 - 0.10, 3 if k >= 0 else 0, int(tseg_t[k] if k >= 0 else tseg_t[-1]), 1); bnd = res
        else:
            w = bend <= end + 1
            hit_t = (bh >= fill_lvl) & w; hit_s = (bl <= slx) & w
            kt = first(hit_t); ks = first(hit_s)
            if kt < 0 and ks < 0:
                last = np.where(w)[0]
                if len(last):
                    i = last[-1]; xp = bc[i] * (1 - s)
                    res = ((xp / E - 1) * 100 - 0.10, 3 if bend[i] >= end - MIN else 0, int(bend[i]), 2)
                else:
                    xp = tseg_p[-1] * (1 - s); res = ((xp / E - 1) * 100 - 0.10, 0, int(tseg_t[-1]), 2)
                bnd = res
            else:
                tp_r = ((tpx / E - 1) * 100 - 0.05 - 0.02, 1, int(bend[kt]) if kt >= 0 else 0, 2)
                sl_r = (((min(slx, bo[ks]) * (1 - s)) / E - 1) * 100 - 0.10, 2, int(bend[ks]) if ks >= 0 else 0, 2)
                if ks < 0:
                    res = bnd = tp_r
                elif kt < 0:
                    res = bnd = sl_r
                elif kt < ks:
                    res = bnd = tp_r
                elif ks < kt:
                    res = bnd = sl_r
                else:
                    res, bnd = sl_r, tp_r
        f = fund_cost(ft, rates, ent, res[2])
        out_n[ci] = res[0] - f; out_b[ci] = bnd[0] - fund_cost(ft, rates, ent, bnd[2]); out_k[ci] = res[1]; out_w[ci] = res[3]
    return out_n, out_b, out_k, out_w


def grid_1m(bs, bo, bh, bl, bc, i, slip, tick):
    """1m convention: entry = close of bar i × (1 + slip); bars after i; hold by bar start ≤ start_i + hd min; same bar = stop"""
    s = slip / 100
    E = bc[i] * (1 + s); st0 = bs[i]
    out = np.full(len(CELLS), np.nan)
    sub = slice(i + 1, i + 1 + 70)
    S, O, Hh, L, C = bs[sub], bo[sub], bh[sub], bl[sub], bc[sub]
    for ci, (tp, sl, hd) in enumerate(CELLS):
        w = S < st0 + (hd + 1) * MIN       # bars [st0 + 1 min, st0 + hd + 1 min)
        if not w.any():
            continue
        tpx = np.ceil(E * (1 + tp / 100) / tick - 1e-9) * tick
        kt = first((Hh >= tpx + tick * 0.999) & w); ks = first((L <= E * (1 - sl / 100)) & w)
        if ks >= 0 and (kt < 0 or ks <= kt):
            out[ci] = ((min(E * (1 - sl / 100), O[ks]) * (1 - s)) / E - 1) * 100 - 0.10
        elif kt >= 0:
            out[ci] = (tpx / E - 1) * 100 - 0.07
        else:
            out[ci] = (C[np.where(w)[0][-1]] * (1 - s) / E - 1) * 100 - 0.10
    return out


def work(pair):
    evs = G["E"][G["E"].pair == pair]
    x = TB.load(pair)
    k5 = None if x is None else (x[0], x[1], x[2], x[3], x[4])
    ft, fc, _ = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    out = []
    for sig in sorted(evs.sig_ms.unique()):
        f = os.path.join(EV, f"{pair}_{sig}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); t, p, bars = z["t"], z["p"], z["bars"]
        if len(t) < 50 or not len(bars):
            continue
        tick = tick_of(pair, p)
        tick_end = int(sig) + 10 * MIN
        # bars after the tick window: 1m from ticks, then local 5m
        b1 = bars[bars[:, 0] >= tick_end]
        parts = [np.c_[b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3], b1[:, 4], b1[:, 0] + MIN]] if len(b1) else []
        last_end = parts[0][-1, 5] if parts else tick_end
        if k5 is not None:
            g5 = k5[0]; jj = int(np.searchsorted(g5, last_end)); kk = min(len(g5), jj + 30)
            if kk > jj:
                parts.append(np.c_[g5[jj:kk], k5[1][jj:kk], k5[2][jj:kk], k5[3][jj:kk], k5[4][jj:kk], g5[jj:kk] + 300_000])
        BA = np.vstack(parts) if parts else np.zeros((0, 6))
        bars_after = tuple(BA[:, k] for k in range(6))
        # 1m-convention arrays (bars from ticks)
        bs, bo, bh, bl, bc = bars[:, 0].astype("int64"), bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4]
        # random minutes on the same coin-day
        d0 = (int(sig) // DAY) * DAY
        lo, hi = max(int(bs[0]), d0), min(d0 + DAY, int(bs[-1]) - 61 * MIN)
        cand = np.where((bs >= lo) & (bs <= hi))[0]
        rng = np.random.default_rng(int(sig) % (2 ** 32) + 7)
        null = np.full((NNULL, len(CELLS)), np.nan, np.float32)
        if len(cand) > 30:
            pick = rng.choice(cand, NNULL, replace=True)
            for r_, i in enumerate(pick):
                null[r_] = grid_1m(bs, bo, bh, bl, bc, int(i), 0.10, tick)
        for cs in CHK:
            T = int(sig) + cs * SEC
            j = int(np.searchsorted(t, T + LAT))
            if j >= len(t):
                continue
            P0 = p[j]; ent = int(t[j])
            r = dict(pair=pair, sig_ms=int(sig), chk=cs, entry_ms=ent, P0=P0, tick=tick, tick_bps=tick / P0 * 1e4,
                     tick_span_s=(tick_end - ent) / 1000)
            for sl_ in SLIPS:
                n_, b_, k_, w_ = grid_ticks(t, p, j, bars_after, P0, sl_, tick, ft, rates)
                r[f"net_{sl_:.2f}"] = n_; r[f"bnd_{sl_:.2f}"] = b_; r[f"kind_{sl_:.2f}"] = k_; r[f"where_{sl_:.2f}"] = w_
            # 1m convention, observed decision minute
            ib = np.where(bs == T)[0]
            r["net1m_obs"] = grid_1m(bs, bo, bh, bl, bc, int(ib[0]), 0.10, tick) if len(ib) and ib[0] < len(bs) - 2 else np.full(len(CELLS), np.nan)
            # Q2 PATH labels (24 h) from 1 s → 1m → 5m path, entry P0 (no slip, as Route A's PATH)
            pb = RF.path_bars(t, p, bars, k5, ent, int(t[-1]) + 1)
            if len(pb) >= 10:
                bh2, bl2, be2 = pb[:, 1], pb[:, 2], pb[:, 4]
                w = be2 <= ent + 24 * H1
                for X in XS:
                    up = np.where(bh2[w] >= P0 * (1 + X / 100))[0]; dn = np.where(bl2[w] <= P0 * (1 - X / 100))[0]
                    ui = up[0] if len(up) else 10 ** 9; di = dn[0] if len(dn) else 10 ** 9
                    r[f"path{X}"] = int(ui < di) if (ui < 10 ** 9 or di < 10 ** 9) else 0
            r["null_ix"] = len(out)
            out.append((r, null if cs == 0 else None))
    return out


def init(E):
    G["E"] = E


def main():
    F = pickle.load(open(os.path.join(SRC, "features.pkl"), "rb"))
    E = F[F.kind == "event"][["pair", "sig_ms"]].drop_duplicates()
    rows, nulls = [], {}
    with mp.Pool(8, initializer=init, initargs=(E,)) as P:
        for k, res in enumerate(P.imap_unordered(work, sorted(E.pair.unique()))):
            for r, nl in res:
                r.pop("null_ix")
                rows.append(r)
                if nl is not None:
                    nulls[(r["pair"], r["sig_ms"])] = nl
            if k % 50 == 0:
                print(k, len(rows), flush=True)
    Tk = pd.DataFrame(rows)
    pickle.dump(dict(T=Tk, null=nulls, cells=CELLS, slips=SLIPS), open(os.path.join(D, "ticks.pkl"), "wb"))
    print("rows", len(Tk), "events with null", len(nulls))


if __name__ == "__main__":
    main()
