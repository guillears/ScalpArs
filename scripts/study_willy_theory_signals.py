#!/usr/bin/env python3
"""WILLY THEORY — step 2: the FRENZY_WILLY entry-A emulator on the rebuilt episode bars + every alternative entry bar of the same
60-min window + outcome-blind features, and the PARITY check vs the replay's WILLY fills. No outcome is read here.

Emulator (mirror of TradingEngine._frenzy_willy_eval, entry A only, B off): the FIRST flagged bar of an episode (spike_ts newer than any
seen on the pair) arms (armed = that bar's close, exp = armed + 60 min); the pending entry opens on the first CLOSED RED 5m bar
(bar_ret < 0; the trigger bar counts) judged at bar_open ≤ exp while the episode is flagged, with turnover R = 24 h quote volume /
(supply proxy × close) < 1.0 (R ≥ 1 or unreadable → refused, stays armed). A newer episode on the pair expires the pending one.
No bearish-day block (WILLY has none). Slots / global hold / pair held are NOT emulated (signal level) — declared.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402

BAR = W.BAR_MS


def load_bars():
    B = pd.read_pickle(os.path.join(W.OUT, "bars_w.pkl"))
    B = B.drop_duplicates(["pair", "T"]).sort_values(["pair", "T"]).reset_index(drop=True)
    return B


def shortlist_T(p):
    """the engine's FRENZY shortlist at each pass (engine_replay._chg_rng parity): |24 h change| (last close vs the open 288 bars ago) OR
    24 h low→high range ≥ frenzy_shortlist_change_pct 15, AND 24 h quote volume ≥ frenzy_min_volume_usd $20M — on the 288 closed 5m bars
    ending at the bar just closed. The 25-pair cap is NOT emulated (declared). → set of bar opens T at which the pair is shortlisted."""
    d = W.load_5m(p)
    if d is None or len(d) < 300:
        return set()
    t = d.open_time.values.astype("int64"); o = d.o.values.astype(float); c = d.c.values.astype(float)
    hi = pd.Series(d.h.values.astype(float)).rolling(288, min_periods=288).max().values
    lo = pd.Series(d.l.values.astype(float)).rolling(288, min_periods=288).min().values
    qv = pd.Series(d.qvol.values.astype(float)).rolling(288, min_periods=288).sum().values
    op = np.full(len(t), np.nan); op[287:] = o[:len(t) - 287]
    with np.errstate(invalid="ignore", divide="ignore"):
        chg = (c / op - 1) * 100
        rng = (np.maximum(hi, c) / np.minimum(lo, c) - 1) * 100
        ok = ((np.abs(chg) >= 15.0) | (rng >= 15.0)) & (qv >= 20e6)
    return set(t[ok].tolist())


def emulate_pair(G, sup, SL):
    """G = one pair's flagged bars (sorted), SL = shortlisted bar opens. The engine judges the pair on bar T when it is shortlisted or was
    flagged on the previous bar (followed). Entry A: the first JUDGED flagged bar of a newer episode; warm (judged on the previous bar) →
    arm; cold → arm only if frenzy_willy_late_bars ≤ 3 (frenzy_willy_a_cold_max_bars), else the episode is seen without an A.
    → list of episode dicts (trigger, first red, entry …)."""
    out = []
    seen = None
    pend = None
    Tm = G["T"].values.astype("int64"); sp = G.spike_ts.values.astype("int64"); ret = G.bar_ret.values.astype(float)
    c = G.c.values.astype(float); v24 = G.vol24.values.astype(float)
    FLG = set(Tm.tolist())

    def judged(T):
        return T in SL or (T - BAR) in FLG

    def R_at(i):
        if sup is None or not np.isfinite(v24[i]) or not (c[i] > 0):
            return np.nan
        return v24[i] / (sup * c[i])

    for i in range(len(G)):
        s = int(sp[i]); T = int(Tm[i])
        if not judged(T):
            continue
        if seen is None or s > seen:                       # new flag — a pending older episode expires first
            if pend is not None:
                pend["end"] = "REPLACED"; out.append(pend); pend = None
            seen = s
            warm = judged(T - BAR)
            late = (T + BAR - s) // BAR
            if warm or late <= 3:
                pend = dict(spike_ts=s, T0=T, armed=T + BAR, exp=T + BAR + W.WAIT_BARS * BAR, warm=warm, late=late,
                            T_red1=None, R_red1=np.nan, T_ent=None, R_ent=np.nan, n_refused=0, end=None)
            else:
                out.append(dict(spike_ts=s, T0=T, warm=warm, late=late, end="COLD_LATE"))
        if pend is None or s != pend["spike_ts"]:
            continue
        if T + BAR > pend["exp"]:
            pend["end"] = "EXPIRED"; out.append(pend); pend = None
            continue
        if not (np.isfinite(ret[i]) and ret[i] < 0):
            continue
        R = R_at(i)
        if pend["T_red1"] is None:
            pend["T_red1"] = T; pend["R_red1"] = R
        if np.isfinite(R) and R < W.R_CUT:
            pend["T_ent"] = T; pend["R_ent"] = R; pend["end"] = "ENTRY"
            out.append(pend); pend = None
        else:
            pend["n_refused"] += 1
    if pend is not None:
        pend["end"] = "EXPIRED"; out.append(pend)
    return out


def main():
    B = load_bars()
    sup = W.supply_table()
    eps = []
    for p, G in B.groupby("pair", sort=False):
        for e in emulate_pair(G.sort_values("T"), sup.get(p), shortlist_T(p)):
            e["pair"] = p
            eps.append(e)
    E = pd.DataFrame(eps)
    E = E[E.end != "COLD_LATE"].reset_index(drop=True)
    print(f"episodes (A triggers) {len(E):,} · end {E.end.value_counts().to_dict()} · first red found {E.T_red1.notna().sum():,}")
    # ── features from the raw 5m series (outcome-blind)
    lst = W.listing_ms()
    rows, alts = [], []
    Bi = B.set_index(["pair", "T"])
    FL = set(zip(B.pair.values, B["T"].values.astype("int64").tolist()))
    for p, G in E.groupby("pair"):
        d = W.load_5m(p)
        t = d.open_time.values.astype("int64"); o = d.o.values.astype(float); h = d.h.values.astype(float); l = d.l.values.astype(float)
        c = d.c.values.astype(float); qv = d.qvol.values.astype(float)
        cq = np.concatenate([[0.0], np.cumsum(qv)])
        idx = {int(x): k for k, x in enumerate(t)}
        for e in G.to_dict("records"):
            on = idx.get(e["spike_ts"] - BAR)
            i0 = idx.get(e["T0"])
            if on is None or i0 is None or on < 18:
                continue
            r = dict(e)
            # spike shape (the 6 bars of the 30-min run ending at the spike bar)
            br = (c[on - 5:on + 1] / c[on - 6:on] - 1) * 100
            pos = br[br > 0]
            r["spike_ret6"] = (c[on] / c[on - 6] - 1) * 100
            r["spike_conc"] = float(br.max() / pos.sum()) if len(pos) and pos.sum() > 0 else np.nan     # 1 = one-candle spike
            rng = h[on] - l[on]
            r["spike_uwick"] = float((h[on] - max(o[on], c[on])) / rng) if rng > 0 else np.nan
            r["spike_bar_ret"] = (c[on] / o[on] - 1) * 100
            r["spike_q"] = qv[on]
            r["pre_base"] = float(np.min(l[max(0, on - 18):on - 5])) if on >= 6 else np.nan                 # low of the 1 h before the run
            r["trig_wait_min"] = (e["T0"] - (e["spike_ts"] - BAR)) / W.MIN                                     # spike bar open → trigger bar open
            r["listing_days"] = (e["T0"] - lst[p]) / W.DAY_MS if p in lst else np.nan
            # window bars T0 … T0 + 12 (raw 5m) — alternative entries
            red_k = 0
            for k in range(W.WAIT_BARS + 1):
                j = i0 + k
                if j >= len(t) or t[j] != e["T0"] + k * BAR:
                    break
                bret = (c[j] / o[j] - 1) * 100
                red = bret < 0
                red_k += int(red)
                flagged = (p, int(t[j])) in FL
                hs = float(h[on:j + 1].max())
                alts.append(dict(pair=p, spike_ts=e["spike_ts"], T0=e["T0"], T=int(t[j]), k=k, red=red, red_ord=(red_k if red else 0),
                                 bar_ret=bret, flagged=flagged, c=c[j], q=qv[j],
                                 vol24=float(cq[j + 1] - cq[j + 1 - 288]) if j + 1 >= 288 else np.nan,
                                 off_high=(c[j] / hs - 1) * 100, is_red1=(e["T_red1"] == int(t[j])), is_ent=(e["T_ent"] == int(t[j]))))
            if pd.notna(e["T_ent"]):
                e["T_ent"] = int(e["T_ent"]); j = idx.get(e["T_ent"])
                hs = float(h[on:j + 1].max())
                r["ent_bar_ret"] = (c[j] / o[j] - 1) * 100
                r["ent_q"] = qv[j]
                r["red_vs_spike_q"] = qv[j] / qv[on] if qv[on] > 0 else np.nan
                r["red_vs_prev_q"] = qv[j] / qv[j - 1] if qv[j - 1] > 0 else np.nan
                r["ent_c"] = c[j]
                r["off_high"] = (c[j] / hs - 1) * 100
                r["wait_bars"] = (e["T_ent"] - e["T0"]) // BAR
                r["min_from_spike"] = (e["T_ent"] + BAR - e["spike_ts"]) / W.MIN
                r["vs_base_pct"] = (c[j] / r["pre_base"] - 1) * 100 if r["pre_base"] > 0 else np.nan
                pk, bs = hs, r["pre_base"]
                r["retrace_frac"] = (pk - c[j]) / (pk - bs) if pk > bs else np.nan
                x = Bi.loc[(p, e["T_ent"])]
                for col in ("hours", "vol_mult", "vs_vwap", "vwap", "run_pct", "gain_pct", "off_peak", "above_share", "peak", "base",
                            "spike_volx", "vol24"):
                    r[col] = x[col]
            rows.append(r)
    S = pd.DataFrame(rows)
    A = pd.DataFrame(alts)
    S["era"] = [W.era_of(x) for x in S.T0]
    S["year"] = [W.year_of(x) for x in S.T0]
    S.to_pickle(W.SIG); A.to_pickle(W.ALT)
    print(f"signals: {len(S):,} episodes with features · entries {S.T_ent.notna().sum():,} · alt bars {len(A):,}")
    # ── parity vs the replay WILLY fills
    F = pd.read_csv(W.FILLS, low_memory=False, usecols=["sleeve", "status", "seed", "pair", "opened_at", "entry_price", "entry_frenzy_spike_at",
                                                        "entry_frenzy_willy_wait_bars", "entry_vol_mcap_ratio", "entry_frenzy_bar_ret_pct"])
    F = F[(F.sleeve == "FRENZY_WILLY") & (F.status == "CLOSED")].copy()
    F["te"] = W.ms(F.opened_at.astype(str)); F["T"] = (F.te // BAR) * BAR - BAR
    F["sp"] = W.ms(F.entry_frenzy_spike_at.astype(str))
    Sg = S[S.T_ent.notna()][["pair", "T_ent", "spike_ts", "wait_bars", "R_ent", "ent_bar_ret"]].rename(columns={"T_ent": "T"})
    Sg["T"] = Sg["T"].astype("int64")
    M = F.merge(Sg, on=["pair", "T"], how="left")
    M["hit"] = M.spike_ts.notna()
    for sd in (1, 2, 3):
        m = M[M.seed == sd]
        print(f"seed {sd}: replay fills {len(m)} · reproduced (same pair & red bar) {m.hit.sum()} = {m.hit.mean() * 100:.1f}% · "
              f"2025 {m[m['T'] < W.SPLIT26].hit.mean() * 100:.1f}% · 2026 {m[m['T'] >= W.SPLIT26].hit.mean() * 100:.1f}%")
    mh = M[M.hit]
    print(f"on matched: spike equal {np.mean(mh.sp == mh.spike_ts) * 100:.1f}% · wait bars equal {np.mean(mh.wait_bars == mh.entry_frenzy_willy_wait_bars) * 100:.1f}% · "
          f"R median ratio {np.nanmedian(mh.R_ent / mh.entry_vol_mcap_ratio):.3f} · bar_ret |Δ| median {np.nanmedian(abs(mh.ent_bar_ret - mh.entry_frenzy_bar_ret_pct)):.4f}")
    miss = M[(~M.hit) & (M.seed == 1)]
    S1 = S.assign(key=S.pair + "|" + S.spike_ts.astype("int64").astype(str))
    miss = miss.assign(key=miss.pair + "|" + miss.sp.astype(str))
    print(f"seed-1 misses {len(miss)}: episode rebuilt {miss.key.isin(S1.key).sum()} · pair absent from bars {(~miss.pair.isin(S.pair)).sum()}")
    print(miss.merge(S1[["key", "T0", "T_red1", "T_ent", "end", "R_red1"]], on="key", how="left")[
        ["pair", "opened_at", "entry_frenzy_willy_wait_bars", "entry_vol_mcap_ratio", "T0", "T_red1", "T_ent", "end", "R_red1"]].head(25).to_string())
    M.to_csv(os.path.join(W.OUT, "parity_rows.csv"), index=False)
    ent = S[S.T_ent.notna()]
    print(f"rebuilt entries per era: {ent.era.value_counts().to_dict()} · replay seed-1 fills per year: {F[F.seed == 1].assign(y=lambda x: np.where(x['T'] < W.SPLIT26, '2025', '2026')).y.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
