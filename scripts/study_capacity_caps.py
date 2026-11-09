#!/usr/bin/env python3
"""💧 CAPACITY × LIQUIDITY CAP per sleeve (DECISION_LOG 505, operator 2026-11-09: "for how many days could the 10 % be valid considering the
max cap per trade of 0.1 % volume or 0.5 % … starting balance 10,000 … per sleeve … build it, add it to the B25 review"). RESEARCH ONLY —
no engine / config change, no bot API, no network.

QUESTION  A position is today's size X (× account, best case one size, with the ② C3 re-size factors) but never more than cap × the coin's
          24 h volume (the engine's liquidity cap: 0.1 % most sleeves, 0.5 % spike trades). As the account compounds, positions hit the cap
          one sleeve after another and the daily compound falls. A higher cap keeps size longer but each order moves the price more.
          → per sleeve and per cap (0.1 / 0.25 / 0.5 / 1 %): the daily compound at each account size NET of price impact, the best cap,
          and the path from $10,000: how many days the platform keeps ≥ 10 / 7 / 5 %/day.
DATA      the best case's current-batch inputs (run run_best_case.py on the same export first): reports/best_case/zones.csv (fill % on
          today's exits + line zones), fb_combo.csv (Flatbase FB_CONFIRM), flip_exits.csv (FAN TP +2), best_dcr.csv (each sleeve's best
          line), best_dcr_out.txt line 1 (X) · the orders export (24 h volume stamp, close time, side) · every decisions export's BOOK
          rows (minute order-book depth at 0 / 0.25 / 0.5 / 1 / 2 % from mid, FRENZY pairs since 11-03).
IMPACT    walk the book: the average price move of an order of $Q on the side it takes (buy → asks), depth linear between the recorded
          levels, extrapolated past 2 % at the last slope, + half the spread. A fill with a book snapshot ≤ 15 min before its entry uses
          its own book; every other fill uses the square-root law impact % = k × √(Q ÷ 24 h volume) with k fitted (median) on the book
          fills. Paid on the entry and, for a losing / stopped fill, on the exit too (a resting TP fills as maker — no impact).
          Paper books every fill at the line, so the booked % carries no size impact: the whole modelled impact is subtracted.
CALIBRATION (operator 11-09 "calibrate it"): the model is scaled ONCE so it matches the REAL entry impact the live batches recorded —
          entry_fill_vs_book_pct (fill vs the book's FAR side at the order — + the books' median ½ spread = vs mid; live B22–B24, reports/BASELINE2[2-4]_*LIVE_orders.csv): factor s =
          median(live impact) ÷ median(model impact at the same order size ÷ 24 h volume). The uncalibrated book walk / k overcharged ≈ 3×
          (DL 505 deep review: the √ law overcharges SMALL orders ≈ 3× — real impact has a half-spread floor, not a √ shape — while the raw
          order-book walk already matches live at live size; the impact LEVEL at 0.1–1 % of volume is NOT identified by live data).
          ⚠ So the cap verdict is SIGN-SENSITIVE to that level → report ④ shows the path at scale 0.2 / 0.35 / 0.6 / 0.85 / 1.0.
LIMITS    current batch only (master has no books / today's sizes); the Binance leverage bracket and notional caps are not modelled (they bind
          too — LITE ≈ 35 % of fills); impact from a minute snapshot (books refill — conservative for a single order, optimistic if several
          sleeves hit one coin at once); the path compounds the batch's one-size daily rate at each account size (in-sample, 30–50 % cut).

Usage: venv/bin/python -I scripts/study_capacity_caps.py <orders export> [decisions export …]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, SC.replace("reports", "scripts"))
import one_size as OS  # noqa: E402

CAPS = (0.001, 0.0025, 0.005, 0.01)
EQUITY = (10e3, 25e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2.5e6, 5e6, 10e6)
LIVE_CAP = {"SPIKE_FADE": 0.005}                     # engine today: spike_lowvol_liq_cap_pct 0.5; everyone else 0.1 %
C3 = {"FRENZY_LITE_RETRY": 1.70, "FLATBASE_SHORT": 2.05, "FRENZY_LITE": 0.86, "MOMENTUM_LONG": 5.0}   # ② C3 (DL 503 / 504)
LEVELS = (0.0, 0.25, 0.5, 1.0, 2.0)
BOOK_TOL = pd.Timedelta(minutes=15)


def walk_impact(q, mid_spread_pct, depth):
    """$q against cumulative depth [(pct from mid, $ cumulative)] → average adverse move % (incl. half spread). PURE."""
    if not (q > 0):
        return 0.0
    pts = [((mid_spread_pct / 2.0 if p == 0 else p), d) for p, d in depth if d == d]   # bands are measured from mid; the top sits at ½ spread
    if len(pts) < 2:
        return float("nan")
    pts = sorted(pts)
    cost, prev_p, prev_d = 0.0, pts[0][0], 0.0
    for p, d in pts:
        if d <= prev_d:
            prev_p = p
            continue
        take = min(q, d) - prev_d
        if take > 0:
            frac = take / (d - prev_d)
            p_end = prev_p + (p - prev_p) * frac
            cost += take * (prev_p + p_end) / 2.0
        if q <= d:
            return cost / q
        prev_p, prev_d = p, d
    slope = (pts[-1][1] - pts[-2][1]) / max(pts[-1][0] - pts[-2][0], 1e-9)   # $ per % beyond the last level
    if not (slope > 0):
        return float("nan")   # (review) a flat last band would explode the extrapolation → the √ law takes the fill
    rest = q - prev_d
    p_end = prev_p + rest / max(slope, 1e-9)
    cost += rest * (prev_p + p_end) / 2.0
    return cost / q


def sqrt_impact(q, vol, k):
    """square-root law: k × √(q ÷ vol) %. PURE."""
    return float(k) * np.sqrt(np.maximum(q, 0.0) / np.maximum(vol, 1.0))


def load_books(paths):
    cols = ["t", "e", "pair", "ob_mid", "ob_spread_pct", "ob_top_bid_usd", "ob_top_ask_usd"] + \
           [f"ob_{s}_usd_{l}" for s in ("bid", "ask") for l in ("025", "05", "1", "2")]
    out = []
    for p in paths:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in cols, low_memory=False)
        except Exception:
            continue
        out.append(d[d.e == "BOOK"])
    if not out:
        return pd.DataFrame(columns=cols + ["bt"])
    b = pd.concat(out, ignore_index=True).drop_duplicates(["t", "pair"])
    b["bt"] = pd.to_datetime(b.t, format="mixed", utc=True).dt.tz_convert(None)
    return b.sort_values("bt")


def book_depth(r, side):
    """side 'ask' (buy) / 'bid' (sell) → [(pct, cumulative $)]."""
    top = r.get(f"ob_top_{side}_usd")
    return [(0.0, top)] + [(lv, r.get(f"ob_{side}_usd_{s}")) for lv, s in ((0.25, "025"), (0.5, "05"), (1.0, "1"), (2.0, "2"))]


def kept_fills(exp):
    """current-batch fills each sleeve's best line keeps, on today's exits → DataFrame(sl, pct, closed_at, opened_at, pair, dir, vol)."""
    B = pd.read_csv(os.path.join(SC, "best_dcr.csv")); B = B[B.scen == "B"].set_index("sleeve")
    Z = pd.read_csv(os.path.join(SC, "zones.csv")); Z = Z[Z.src == "B25"].copy()
    Z["k"] = pd.to_datetime(Z.o).dt.strftime("%Y-%m-%dT%H:%M:%S")
    A = pd.read_csv(exp, low_memory=False)
    A = A[(A.status == "CLOSED") & (pd.to_datetime(A.opened_at, format="mixed") >= pd.Timestamp("2026-10-23 18:30"))].copy()
    A["k"] = A.opened_at.astype(str).str[:19]
    A["sl"] = A.entry_strategy.astype(str).where(A.entry_strategy.astype(str) != "MOMENTUM", "MOMENTUM_" + A.direction.astype(str))
    A["vol"] = pd.to_numeric(A.entry_pair_volume_24h_usd, errors="coerce")
    meta = A[["k", "pair", "sl", "closed_at", "opened_at", "direction", "vol", "close_reason"]]
    rows = []
    for sl in sorted(A.sl.unique()):
        line = B.loc[sl, "lines"] if sl in B.index and isinstance(B.loc[sl, "lines"], str) else ""
        lines = [x for x in line.split(" + ") if x]
        if sl == "FLATBASE_SHORT" and "FB_CONFIRM" in lines:
            fb = pd.read_csv(os.path.join(SC, "fb_combo.csv")); fb = fb[fb.k >= "2026-10-23T18:30"]
            fb = fb[fb.cf510.notna() & (fb.cf510 != 0)]
            g = fb.rename(columns={"cf510": "pct"})[["k", "pair", "pct"]]
            g = g.merge(meta[meta.sl == sl], on=["k", "pair"], how="left")
            g["closed_at"] = (pd.to_datetime(fb.k.values) + pd.to_timedelta(fb.h_cf510.values, unit="m")).astype(str)   # h_cf510 counts from S4 incl. the wait (review)
        elif sl.startswith("FLIP") and any(x.startswith("EXIT ") for x in lines):
            ex = [x[5:] for x in lines if x.startswith("EXIT ")][0]
            fx = pd.read_csv(os.path.join(SC, "flip_exits.csv")); fx = fx[fx.src == "B25"]
            g = fx.rename(columns={ex: "pct"})[["k", "pair", "pct", f"h_{ex}"]].merge(meta[meta.sl == sl], on=["k", "pair"], how="left")
            g["closed_at"] = (pd.to_datetime(g.k) + pd.to_timedelta(g[f"h_{ex}"], unit="m")).astype(str)   # the re-priced exit's own hold (review)
        else:
            z = Z[Z.sl == sl]
            if len(z):
                keep = pd.Series(True, index=z.index)
                for ln in lines:
                    keep &= ~z[ln].fillna(False).astype(bool)
                g = z[keep][["k", "pair", "pct"]].merge(meta[meta.sl == sl], on=["k", "pair"], how="left")
            else:
                g = meta[meta.sl == sl].merge(A[["k", "pair", "pnl_percentage"]], on=["k", "pair"]).rename(columns={"pnl_percentage": "pct"})
        g = g.assign(sl=sl)
        rows.append(g)
    F = pd.concat(rows, ignore_index=True)
    F["pct"] = pd.to_numeric(F.pct, errors="coerce")
    return F[F.pct.notna() & F.closed_at.notna()].reset_index(drop=True), A


def attach_books(F, books):
    F = F.copy()
    F["o"] = pd.to_datetime(F.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    if not len(books):
        F["bt"] = pd.NaT
        return F
    m = pd.merge_asof(F.reset_index().sort_values("o"), books, left_on="o", right_on="bt", by="pair", direction="backward", tolerance=BOOK_TOL)
    return m.set_index("index").sort_index()


def fit_k(F, probe=(0.25, 0.5, 1.0, 2.0)):
    """median k of impact = k √(q / vol) over book-covered fills, at order sizes = probe × the fill's 0.1 % cap. → (k, n fills)."""
    ks = []
    for r in F[F.bt.notna() & (F.vol > 0)].to_dict("records"):
        side = "ask" if str(r["direction"]).upper() == "LONG" else "bid"
        for f in probe:
            q = f * 0.001 * r["vol"]
            imp = walk_impact(q, r.get("ob_spread_pct") or 0.0, book_depth(r, side))
            if imp == imp and imp > 0:
                ks.append(imp / np.sqrt(q / r["vol"]))
    return (float(np.median(ks)) if ks else 2.0), len(F[F.bt.notna()])


def calibrate(k, root=ROOT, half_spread=0.0):
    """live B22–B24 fills: s = median(entry_fill_vs_book_pct) ÷ median(k √(notional ÷ 24 h volume)) → (s, n, live median %, median q/vol)."""
    fs = sorted(glob.glob(os.path.join(root, "reports", "BASELINE2[2-4]_*LIVE_orders.csv")))
    if not fs:
        return 1.0, 0, float("nan"), float("nan")
    D = pd.concat([pd.read_csv(f, low_memory=False) for f in fs], ignore_index=True)
    D = D[D.status == "CLOSED"].drop_duplicates(["opened_at", "pair", "direction"])
    live = pd.to_numeric(D.get("entry_fill_vs_book_pct"), errors="coerce")
    qv = pd.to_numeric(D.notional_value, errors="coerce") / pd.to_numeric(D.get("entry_pair_volume_24h_usd"), errors="coerce")
    ok = live.notna() & (qv > 0)
    if ok.sum() < 10:
        return 1.0, int(ok.sum()), float("nan"), float("nan")
    model = sqrt_impact(qv[ok].values, 1.0, k)
    live_mid = max(float(live[ok].median()), 0.0) + float(half_spread)   # (review) entry_fill_vs_book_pct is vs the far side (exec_shadow) → + ½ spread = vs mid
    s = float(live_mid / np.median(model)) if np.median(model) > 0 else 1.0
    return s, int(ok.sum()), live_mid, float(qv[ok].median())


SCALE = 1.0   # set in main() from calibrate()


def impact_pct(r, q, k):
    side = "ask" if str(r["direction"]).upper() == "LONG" else "bid"
    if pd.notna(r.get("bt")):
        v = walk_impact(q, r.get("ob_spread_pct") or 0.0, book_depth(r, side))
        if v == v:
            return SCALE * v
    if not (r["vol"] == r["vol"] and r["vol"] > 0):
        raise ValueError(f"{r.get('sl')} {r.get('pair')} {r.get('k')}: no 24 h volume and no book — its impact can't be priced")
    return SCALE * sqrt_impact(q, r["vol"], k)


def sleeve_frac(F, sl, X, E, cap, k):
    """per-fill account fraction of sleeve sl at account E and cap (net of impact) → (fractions, mean share of size kept)."""
    g = F[F.sl == sl]
    xs = X[sl] * C3.get(sl, 1.0)
    out, kept = [], []
    for r in g.to_dict("records"):
        want = xs * E
        q = min(want, cap * r["vol"]) if r["vol"] == r["vol"] and r["vol"] > 0 else want
        kept.append(q / want if want > 0 else 1.0)
        imp = impact_pct(r, q, k)
        legs = 2.0 if (r["pct"] < 0 or "STOP" in str(r.get("close_reason", "")).upper()) else 1.0
        out.append((r["pct"] - legs * imp) * q / E / 100.0)
    return np.array(out, dtype=float), (float(np.mean(kept)) if kept else 1.0)


def dcr_of(F, frac_by_sl, t0, t1):
    idx = np.concatenate([F.index[F.sl == sl].values for sl in frac_by_sl]) if frac_by_sl else np.array([], int)
    f = np.concatenate([frac_by_sl[sl] for sl in frac_by_sl]) if frac_by_sl else np.array([])
    if not len(idx):
        return 0.0
    return OS.dcr(F.loc[idx, "closed_at"].values, f, np.ones(len(idx), bool), t0, t1)


def main(argv):
    global SCALE
    exp = argv[1]
    decs = argv[2:] or sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))
                              + glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_decisions_*.csv")))   # (review) Downloads gets wiped
    X = {k: float(v) for k, v in re.findall(r"([A-Z_:]+) ([0-9.]+)", open(os.path.join(SC, "best_dcr_out.txt")).readline().split("=", 1)[1])}
    F, A = kept_fills(exp)
    F = F[F.sl.isin(X) & ~F.sl.isin(("SURGE_LONG", "BEARRUN_SHORT"))].reset_index(drop=True)   # operator: SURGE / Bear-Run out
    books = load_books(decs)
    F = attach_books(F, books)
    k, nb = fit_k(F)
    _hs = float(pd.to_numeric(books.get("ob_spread_pct"), errors="coerce").median() / 2.0) if len(books) else 0.0
    SCALE, n_live, live_med, live_qv = calibrate(k, half_spread=_hs if _hs == _hs else 0.0)
    t0 = A.opened_at.min()
    t1 = pd.to_datetime(A.closed_at, format="mixed").max()
    sleeves = sorted(F.sl.unique())
    # per sleeve × cap × equity
    rows = []
    for sl in sleeves:
        for c in CAPS:
            for E in EQUITY:
                fr, kept = sleeve_frac(F, sl, X, E, c, k)
                rows.append(dict(sl=sl, cap=c, E=E, dcr=dcr_of(F, {sl: fr}, t0, t1), kept=kept))
    T = pd.DataFrame(rows)
    # platform at today's caps vs the per-sleeve best cap
    def platform(capmap):
        out = []
        for E in EQUITY:
            fr = {sl: sleeve_frac(F, sl, X, E, capmap(sl, E), k)[0] for sl in sleeves}
            out.append(dcr_of(F, fr, t0, t1))
        return np.array(out)
    live = platform(lambda sl, E: LIVE_CAP.get(sl, 0.001))
    def _pick(sl, E):   # (operator 11-09 review) a tie (within 0.01 %/day) keeps the smallest cap — never a bigger order for nothing
        t = T[(T.sl == sl) & (T.E == E)]
        top = t.dcr.max()
        return float(t[t.dcr >= top - 0.01].cap.min())
    best_c = {(sl, E): _pick(sl, E) for sl in sleeves for E in EQUITY}
    best = platform(lambda sl, E: best_c[(sl, E)])
    uni = {c: platform(lambda sl, E, c=c: c) for c in CAPS}
    # 🎯 (operator 11-09 "show me changing LITE and RETRY to 0.5 % and the rest as is" → "save this and add it to the B25 review")
    LR = ("FRENZY_LITE", "FRENZY_LITE_RETRY")
    lr05 = platform(lambda sl, E: 0.005 if sl in LR else LIVE_CAP.get(sl, 0.001))
    LRF = LR + ("FLATBASE_SHORT",)   # 🎯 (operator 11-09) + Flatbase — the recommendation: big coins, 1 % adds ≈ nothing over 0.5 %
    lrf05 = platform(lambda sl, E: 0.005 if sl in LRF else LIVE_CAP.get(sl, 0.001))
    # ⚠ (DL 505 deep review H1) sensitivity to the large-order impact level: today vs the recommendation at several scales
    _s0, SENS = SCALE, []
    for _sc in (0.2, 0.35, 0.6, 0.85, 1.0):
        SCALE = _sc
        SENS.append((_sc, platform(lambda sl, E: LIVE_CAP.get(sl, 0.001)), platform(lambda sl, E: 0.005 if sl in LRF else LIVE_CAP.get(sl, 0.001))))
    SCALE = _s0

    def path(curve, days=120):
        le = np.log(np.array(EQUITY)); E, out = 10_000.0, []
        for d in range(days):
            r = float(np.interp(np.log(max(E, EQUITY[0])), le, curve)) / 100.0
            out.append((d + 1, E, r))
            E *= (1 + r)
        return out

    def first_below(p, x):
        for d, E, r in p:
            if r < x:
                return d, E
        return None, None
    stamp = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})", os.path.basename(exp))
    stamp = stamp[1] if stamp else "latest"
    L = [f"# 💧 Capacity × liquidity cap per sleeve — export {stamp} (DECISION_LOG 505)", "",
         "Research only. Each sleeve at today's size × the ② C3 re-size (RETRY ×1.70 · Flatbase ×2.05 · LITE ×0.86 · Momentum Long ×5), its "
         "best-case line on, current batch, one size; positions capped at cap × the coin's 24 h volume; price impact subtracted (own order "
         f"book where one was recorded ≤ 15 min before entry — {nb} fills — else the square-root law fitted on those books, k = {k:.2f}), "
         f"**calibrated to the live fills** ({n_live} live B22–B24 entries: median fill vs book mid (far side + ½ median spread) {live_med:+.3f} % at order ÷ volume "
         f"{live_qv * 100:.4f} % → the model × {SCALE:.2f}; above that order size the √ law extrapolates). "
         f"Books from {len(decs)} decisions export(s). SURGE / Bear-Run left out (operator). The path assumes the ② C3 re-sizes (RETRY ×1.70 · Flatbase ×2.05 · LITE ×0.86 · ML ×5 — gated, not shipped). Calibration fills: WILLY 14 · LITE 10 · FRENZY_LONG 6 · SURGE 3 · RETRY 3 · Flatbase 1 — Momentum / FAN / Spike Fade have no live calibration fill and no books (√ law only). In-sample: 30–50 % cut. Not modelled: Binance leverage brackets / notional caps, the engine's max_notional_hard_ceiling $500k and max_gross_leverage 25 (they bind too — the $1M+ rows are optimistic), "
         "several sleeves hitting one coin at once.", "",
         "## ① The path from $10,000 (platform daily compound vs account size)", "",
         "| Caps (impact scale as calibrated) | at $10k | $50k | $250k | $1M | $5M | ≥ 7 %/day through (balance) | ≥ 5 %/day through | balance day 30 / 60 / 90 |",
         "|---|---|---|---|---|---|---|---|---|"]
    def row(name, curve):
        p = path(curve)
        cells = [f"{curve[EQUITY.index(e)]:+.2f}" for e in (10e3, 50e3, 250e3, 1e6, 5e6)]
        fb = []
        for x in (0.07, 0.05):
            d, E = first_below(p, x)
            fb.append("never (120 d)" if d is None else f"day {d - 1} (${E / 1e3:,.0f}k)")   # (review L1) the last day still at / above
        bal = " / ".join(f"${p[d - 1][1] / 1e3:,.0f}k" for d in (30, 60, 90))
        return f"| {name} | " + " | ".join(cells) + " | " + " | ".join(fb) + f" | {bal} |"
    L.append(row("today (0.1 %, fades 0.5 %)", live))
    for c in CAPS:
        L.append(row(f"all sleeves {c * 100:g} %", uni[c]))
    L.append(row("🎯 LITE + RETRY 0.5 %, rest as today", lr05))
    L.append(row("🎯 LITE + RETRY + Flatbase 0.5 %, rest as today", lrf05))
    L += ["", "## ④ ⚠ Sensitivity — the verdict depends on how much a BIG order moves the price (not measured live yet)", "",
          "Live fills only tell us the impact of small orders (≈ 0.004 % of volume). Scale = how much of the order-book / √-law impact a big "
          "order really pays (1.0 = the full minute-snapshot book walk, ≈ what the books say; 0.35 = this report's calibration; books refill "
          "within seconds, so the truth for one order is likely below 1.0 — but unknown).", "",
          "| Impact scale | Today's caps: ≥ 5 %/day through | day 90 | LITE + RETRY + Flatbase 0.5 %: ≥ 5 %/day through | day 90 | 0.5 % vs today |", "|---|---|---|---|---|---|"]
    for _sc, _t, _r in SENS:
        _pt, _pr = path(_t), path(_r)
        _dt, _ = first_below(_pt, 0.05); _dr, _ = first_below(_pr, 0.05)
        _bt, _bR = _pt[89][1], _pr[89][1]
        L.append(f"| {_sc:.2f}{' (this report)' if abs(_sc - 0.35) < 0.01 else ''} | {'never' if _dt is None else f'day {_dt - 1}'} | ${_bt / 1e3:,.0f}k | "
                 f"{'never' if _dr is None else f'day {_dr - 1}'} | ${_bR / 1e3:,.0f}k | {'**wins**' if _bR > _bt * 1.05 else ('loses' if _bR < _bt * 0.95 else 'about even')} |")
    L += ["", "→ 0.5 % beats today's caps only while big orders pay well under the full book walk. Decide it on LIVE slippage at larger "
          "sizes, not on this table: 🔒 a raised cap ships only with a pre-committed revert — its first 15 live fills' slippage vs the book "
          "above the level where this table says it loses → back to 0.1 %.",
          "", "## ② Per sleeve — daily compound (%/day, net of impact) and share of the wanted size kept, by cap and account size", "",
          "| Sleeve | Cap | " + " | ".join(f"${e / 1e3:,.0f}k" for e in EQUITY) + " |", "|---|---|" + "---|" * len(EQUITY)]
    for sl in sleeves:
        for c in CAPS:
            t = T[(T.sl == sl) & (T.cap == c)].set_index("E")
            L.append(f"| {sl} | {c * 100:g} % | " + " | ".join(f"{t.loc[e, 'dcr']:+.2f} ({t.loc[e, 'kept']:.0%})" for e in EQUITY) + " |")
    L += ["", "## ③ Best cap per sleeve (highest daily compound net of impact)", "",
          "| Sleeve | today | $25k | $100k | $250k | $1M | $5M |", "|---|---|---|---|---|---|---|"]
    for sl in sleeves:
        L.append(f"| {sl} | {LIVE_CAP.get(sl, 0.001) * 100:g} % | " + " | ".join(f"{best_c[(sl, e)] * 100:g} %" for e in (25e3, 100e3, 250e3, 1e6, 5e6)) + " |")
    L += ["", "Reading: a cap that wins only at sizes the account has not reached changes nothing today; a sleeve whose best cap rises with the "
          "account is capacity-bound (its size is worth more than the extra impact). Decide per sleeve at the review next to the ②y3 sizing "
          "study, then confirm on the first live fills (slippage vs order size — the DL 454 open decision)."]
    out = os.path.join(ROOT, "reports", f"CAPACITY_CAPS_{stamp}.md")
    open(out, "w").write("\n".join(L) + "\n")
    T.to_csv(os.path.join(ROOT, "reports", f"CAPACITY_CAPS_{stamp}.csv"), index=False)
    print("\n".join(L))
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv)
