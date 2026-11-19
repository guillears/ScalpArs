#!/usr/bin/env python3
"""FRENZY LONG SEARCH (PREREG reports/FRENZY_LONG_SEARCH_2026-11-19_PREREG.md) — BUILD.
Per pair: every validated-walk Frenzy episode (reports/UNIFIED_LONG_data/eps_all.pkl) → checkpoint entries at spike close + h
(h ∈ 6/24/48/72 h), entry at the next 5m open; lag-safe features at the signal close; 18 frozen long exits
(hold 24/72/168 h × stop none/20/10 % gap-aware × TP none/20 % maker) priced from the 5m path with real funding; hourly path
−24 h … +168 h for the event study; REACH / WORKED labels.
Output reports/FRENZY_LONG_SEARCH_data/rows/<PAIR>.pkl.  Research only (reads local caches; 0 Binance API weight).
Usage: venv/bin/python scripts/frenzy_long_search_build.py [--pairs A,B] [--procs 8]"""
import argparse, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import frenzy_ta_build as TB   # noqa: E402

D = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data"); os.makedirs(os.path.join(D, "rows"), exist_ok=True)
UD = os.path.join(R, "reports", "UNIFIED_LONG_data")
SSD = os.path.join(R, "reports", "SPOT_SHARE_DEEP_data"); EXD = os.path.join(R, "reports", "EXHAUSTION_EXPAND_data")
TB.K5M_SRC = [os.path.join(UD, "k5m"), os.path.join(EXD, "k5m"), os.path.join(SSD, "k5m")] + TB.K5M_SRC + [os.path.join(SSD, "k5m_jun")]
TB.FUND_SRC = TB.FUND_SRC + [os.path.join(SSD, "funding"), os.path.join(EXD, "funding"), os.path.join(UD, "funding")]
BAR, H1, DAY = 300_000, 3_600_000, 86_400_000
CHECK_H = (6, 24, 48, 72)
HOLDS = {24: 288, 72: 864, 168: 2016}
STOPS = (None, 20.0, 10.0)
TPS = (None, 20.0, 10.0, 5.0)   # 10 / 5 added post-hoc (exploratory, after the GAVE_BACK path read)
EXITS = [(hh, sl, tp) for hh in HOLDS for sl in STOPS for tp in TPS]
ENAME = lambda hh, sl, tp: f"H{hh}_S{'x' if sl is None else int(sl)}_T{'x' if tp is None else int(tp)}"
PATH_OFF = np.arange(-24, 169)          # hourly offsets vs the entry bar
FUND_MISS = 0.05                         # % per 8 h against the position on a missing pair-month


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def work(args):
    pair, eps = args
    out_f = os.path.join(D, "rows", pair + ".pkl")
    if os.path.exists(out_f):
        return pair, -1
    x = TB.load(pair)
    if x is None:
        pickle.dump(pd.DataFrame(), open(out_f, "wb")); return pair, 0
    t, o, h, l, c, v, qv = x
    n = len(t)
    real = qv > 0
    q24 = pd.Series(qv).rolling(288, min_periods=1).sum().values
    ft, fc, fmonths = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    rows = []
    for e in eps:
        sp = int(e["spike_ms"])                      # spike bar close
        i_sp = int(np.searchsorted(t, sp - BAR))     # spike bar index (open = close − 5 min)
        if i_sp >= n or t[i_sp] != sp - BAR:
            continue
        i_pre = i_sp - 12
        if i_pre < 0:
            continue
        pre = c[i_pre]                               # close 1 h before the spike close
        a0 = max(0, i_sp - 5)                        # 6-bar spike window start → anchored VWAP
        cq = np.cumsum(qv[a0:]); cv = np.cumsum(v[a0:])
        for hh in CHECK_H:
            i_sig = i_sp + hh * 12                   # signal bar: closes at spike close + h
            i_ent = i_sig + 1
            if i_ent + 1 >= n:
                continue
            sig_close_ms = int(t[i_sig]) + BAR
            if e["first_seen_ms"] < 0 or e["first_seen_ms"] > sig_close_ms:
                continue
            if q24[i_sig] < 20e6 or not real[i_sig] or not real[i_ent]:
                continue
            j = i_sig - a0
            vwap = cq[j] / cv[j] if cv[j] > 0 else np.nan
            mx = h[i_sp:i_sig + 1].max()
            first6 = qv[i_sp:i_sp + 72].sum(); last6 = qv[max(i_sp, i_sig - 71):i_sig + 1].sum()
            i24 = max(0, i_sig - 288)
            # funding at the signal close
            if len(ft):
                jf = int(np.searchsorted(ft, sig_close_ms, side="right")) - 1
                f_last = rates[jf] if jf >= 0 else np.nan
                ivl = (ft[jf] - ft[jf - 1]) / H1 if jf >= 1 else 8.0
                ivl = ivl if ivl > 0.5 else 8.0
                f8 = f_last * 8.0 / min(8.0, ivl) if np.isfinite(f_last) else np.nan
                k24 = int(np.searchsorted(ft, sig_close_ms - DAY, side="right"))
                f24 = rates[k24:jf + 1].sum() if jf >= k24 else 0.0
            else:
                f_last = f8 = f24 = np.nan; ivl = np.nan
            E0 = o[i_ent]
            r = dict(pair=pair, spike_ms=sp, ck=hh, sig_ms=sig_close_ms, entry_ms=int(t[i_ent]), entry_px=E0,
                     gain=(c[i_sig] / pre - 1) * 100, vs_vwap=(c[i_sig] / vwap - 1) * 100 if vwap else np.nan,
                     off_high=(c[i_sig] / mx - 1) * 100, vdec=last6 / first6 if first6 > 0 else np.nan,
                     ret24=(c[i_sig] / c[i24] - 1) * 100, ret6=(c[i_sig] / c[max(0, i_sig - 72)] - 1) * 100,
                     vol24=q24[i_sig], fund_last=f_last, fund8=f8, fund_ivl=ivl, fund24=f24,
                     still_flag=bool(e["off_ms"] > sig_close_ms))
            # path (hourly closes vs entry open)
            idx = i_ent + PATH_OFF * 12
            ok = (idx >= 0) & (idx < n)
            pth = np.full(len(PATH_OFF), np.nan, np.float32)
            pth[ok] = (c[idx[ok]] / E0 - 1) * 100
            r["path"] = pth
            # exits
            end_max = min(n - 1, i_ent + 2016 - 1)
            hi = h[i_ent:end_max + 1]; lo = l[i_ent:end_max + 1]; op = o[i_ent:end_max + 1]; cl = c[i_ent:end_max + 1]
            r["avail_h"] = len(cl) / 12.0
            r["reach168"] = (hi.max() / E0 - 1) * 100; r["mae168"] = (lo.min() / E0 - 1) * 100
            d0 = int(t[i_ent]) // DAY * DAY
            dstarts = d0 + DAY * np.arange(9)
            dmiss = np.array([pd.Timestamp(int(dd), unit="ms").strftime("%Y-%m") not in fmonths for dd in dstarts])
            for hold, nb in HOLDS.items():
                m = min(nb, len(cl))
                for sl in STOPS:
                    s_hit = np.where(lo[:m] <= E0 * (1 - sl / 100))[0] if sl else np.zeros(0, int)
                    s_i = int(s_hit[0]) if len(s_hit) else 10**9
                    for tp in TPS:
                        t_hit = np.where(hi[:m] >= E0 * (1 + tp / 100) * 1.0005)[0] if tp else np.zeros(0, int)
                        t_i = int(t_hit[0]) if len(t_hit) else 10**9
                        if s_i <= t_i and s_i < 10**9:
                            px = min(E0 * (1 - sl / 100), op[s_i]); kind = 1; xi = s_i
                        elif t_i < 10**9:
                            px = E0 * (1 + tp / 100); kind = 2; xi = t_i
                        else:
                            xi = m - 1; px = cl[xi]; kind = 0
                            if xi < 5 and False:
                                pass
                        if sl is None:   # catastrophe cap −95 %
                            px = max(px, E0 * 0.05)
                        x_ms = int(t[i_ent + xi]) + BAR
                        # funding: settlements in (entry_open, exit]
                        fc_ = 0.0
                        if len(ft):
                            a_ = int(np.searchsorted(ft, int(t[i_ent]), side="right")); b_ = int(np.searchsorted(ft, x_ms, side="right"))
                            fc_ = float(rates[a_:b_].sum())
                        # missing months penalty
                        held_d = (x_ms - int(t[i_ent])) / DAY
                        nd = int(np.searchsorted(dstarts, x_ms))          # calendar days touched
                        fpen = FUND_MISS * 3 * held_d * (dmiss[:nd].mean() if nd else 0.0)
                        nm = ENAME(hold, sl, tp)
                        r[nm + "_g"] = (px / E0 - 1) * 100          # gross vs the raw entry open
                        r[nm + "_k"] = kind                           # 0 time, 1 stop, 2 TP
                        r[nm + "_f"] = fc_ + fpen                     # funding paid by the long (% of notional; <0 = received)
                        r[nm + "_x"] = x_ms
                        r[nm + "_complete"] = bool(len(cl) >= nb or kind != 0)
            # WORKED / REACH vs the −10 % and −20 % stops
            for sl in (10.0, 20.0):
                sh = np.where(lo <= E0 * (1 - sl / 100))[0]; si = int(sh[0]) if len(sh) else len(lo)
                r[f"reach_s{int(sl)}"] = (hi[:max(1, si)].max() / E0 - 1) * 100
            rows.append(r)
    df = pd.DataFrame(rows)
    pickle.dump(df, open(out_f, "wb"))
    return pair, len(df)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    E = pickle.load(open(os.path.join(UD, "eps_all.pkl"), "rb"))
    by = {}
    for e in E:
        by.setdefault(e["pair"], []).append({k: e[k] for k in ("spike_ms", "on_ms", "off_ms", "first_seen_ms")})
    pairs = sorted(by) if not a.pairs else a.pairs.split(",")
    log("pairs", len(pairs), "episodes", sum(len(by[p]) for p in pairs))
    tot = 0
    with mp.Pool(a.procs) as pool:
        for k, (p, nrows) in enumerate(pool.imap_unordered(work, [(p, by[p]) for p in pairs])):
            tot += max(0, nrows)
            if k % 50 == 0:
                log(k, p, nrows, "total", tot)
    log("done rows", tot)


if __name__ == "__main__":
    main()
