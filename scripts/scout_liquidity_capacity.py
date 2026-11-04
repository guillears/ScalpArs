#!/usr/bin/env python3
"""💧📏 Scout — LIQUIDITY CAPS & CAPACITY per sleeve (DECISION_LOG 454, 2026-11-04 — operator: "In scout we are tracking the positions that
were capped? per sleeve?" → "yes build it"). OBSERVE-only, never changes config; orders exports + the master pool only, no network.

WHY        every order's notional is capped at a share of the coin's 24 h volume (max_notional_pct_of_pair_volume 0.1 % · spike trades
           spike_lowvol_liq_cap_pct 0.5 % · max_notional_hard_ceiling $500k). The cap protects the fill, but once it binds a sleeve stops
           growing with the account — the first part of the capacity study (scaling roadmap ③, DECISION_LOG 404 / 432).
COHORT     CLOSED full-size bot fills (MANUAL and probes out: is_probe / *_PROBE cells), each sleeve apart (MOMENTUM split LONG / SHORT).
           ① CURRENT BATCH = every orders export (paper + live) opened ≥ BATCH_START, dedup (opened_at, pair, direction), newest wins ·
           ② MASTER = reports/MASTER_POOL_stacked.csv kept rows, AS TRADED (pnl_percentage / notional_value — the size the cap produced).
TWO CAPS   the engine sizes in order (services/trading_engine.py ~14334–14460): desired (after cells / re-size / HT ¼) → ① VOLUME cap
           min(pct × 24 h vol, $500k) (+ the gross-leverage guard, off today) → ③ EXCHANGE LEVERAGE BRACKET (Binance's max position per
           leverage tier, entry_bracket_cap_notional / bracket_capped, DECISION_LOG 170) → notional_value. Deep review 11-04: the bracket
           cuts more fills than the volume cap (WILLY 8 bracket vs 1 volume), so both are shown apart.
COLUMNS    volume-capped = liquidity_capped ∧ the volume cap < desired (a gross-only throttle is counted apart) · bracket-capped =
           bracket_capped · cut split: volume = desired − min(desired, vol cap) · bracket / other = the rest down to notional (bracket, or a gross-only throttle) · $ left out = booked %
           × that cut / 100 (linear, no market impact: an UPPER bound; a capped loser's cut SAVED money and counts negative) · binds at ×K =
           limit / desired per fill (the account multiple at which the wanted size meets it; ≤ 1 = capped today) for the volume cap alone and
           for ANY cap (min of volume and bracket — the bracket read AT THE FILL'S LEVERAGE: a lower leverage holds more position for more
           margin, so 'any cap' is the conservative read; a bracket fill with no tier stamp = its traded notional) — median, and the share of fills
           capped (any cap) at ×2 / ×5 / ×10 the account.
SLIPPAGE   entry_slippage_pct of capped vs uncapped fills on LIVE fills only (paper simulates ≈ 0): master LIVE eras + live exports.
Usage      venv/bin/python -I scripts/scout_liquidity_capacity.py
"""
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_D)
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG  # noqa: E402

DL = 454
BATCH_START = "2026-10-23T18:30:00"                 # B25 (PAPER since 10-23 ~18:30 UTC) — move at each batch boundary
LIVE_ERAS = ("B22", "B23", "B24")                   # master batches traded LIVE (go-live 10-14; B22 · B23 · B24 archived as *_LIVE_orders)
MASTER = os.path.join(_ROOT, "reports", "MASTER_POOL_stacked.csv")
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
MULTS = (2, 5, 10)
_TRUE = ("true", "1", "1.0")


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


def _probe(d):
    m = d["is_probe"].astype(str).str.lower().isin(_TRUE) if "is_probe" in d else pd.Series(False, index=d.index)
    for c in ("cell_multiplier_source", "pattern_cell_source"):
        if c in d:
            m = m | d[c].astype(str).str.upper().str.contains("PROBE")
    return m


def prep(d):
    """→ the cohort frame with sleeve, vcap (volume cap bound), gross (gross-only throttle), bcap (bracket), pct, notional, desired, cap,
    limit (any cap), slip (pure)."""
    d = d[(d.status.astype(str) == "CLOSED") & ~_probe(d)].copy()
    es = d.entry_strategy.fillna("MOMENTUM").astype(str)
    d = d[es != "MANUAL"].copy()
    es = d.entry_strategy.fillna("MOMENTUM").astype(str)
    d["sleeve"] = es.where(es != "MOMENTUM", "MOMENTUM_" + d.direction.astype(str))
    lc = d.liquidity_capped.astype(str).str.lower().isin(_TRUE) if "liquidity_capped" in d else pd.Series(False, index=d.index)
    d["pct"] = _num(d, "pnl_percentage")
    d["notional"] = _num(d, "notional_value")
    d["desired"] = _num(d, "entry_desired_notional")
    d["cap"] = _num(d, "entry_liquidity_cap_notional")
    d["vcap"] = lc & (d.cap < d.desired - 0.01)
    d["gross"] = lc & ~d.vcap
    d["bcap"] = d.bracket_capped.astype(str).str.lower().isin(_TRUE) if "bracket_capped" in d else False
    brk = _num(d, "entry_bracket_cap_notional").where(lambda x: x > 0)
    lim = pd.concat([d.cap.where(d.cap > 0), brk], axis=1).min(axis=1)
    d["limit"] = lim.where(~(d.bcap & brk.isna()), pd.concat([lim, d.notional], axis=1).min(axis=1))   # a leverage-cut bracket: its notional
    d["capped"] = d.vcap | d.gross | d.bcap
    d["slip"] = _num(d, "entry_slippage_pct")
    return d


def sleeve_rows(d):
    """→ one dict per sleeve: n, capped, cut (median on capped), missed $, capped / uncapped WR · avg %, binds-at median, share capped
    at each account multiple (pure)."""
    out = []
    for sl, g in d.groupby("sleeve"):
        c, u = g[g.capped], g[~g.capped]
        h = g[g.desired.gt(0)]
        after_vol = np.minimum(h.desired, h.cap.where(h.cap > 0, np.inf))
        v_cut, b_cut = (h.desired - after_vol).clip(lower=0), (after_vol - h.notional).clip(lower=0)
        kv = (g.cap / g.desired)[g.desired.gt(0) & g.cap.gt(0)]
        k = (g.limit / g.desired)[g.desired.gt(0) & g.limit.gt(0)]
        vc, bc = h[h.vcap], h[h.bcap]
        out.append(dict(sleeve=sl, n=len(g), vcap=int(g.vcap.sum()), bcap=int(g.bcap.sum()), gross=int(g.gross.sum()), capped=len(c),
                        v_cutmed=(v_cut[vc.index] / vc.desired).median() if len(vc) else np.nan,
                        b_cutmed=(b_cut[bc.index] / bc.desired).median() if len(bc) else np.nan,
                        v_missed=float((h.pct * v_cut / 100).sum()), b_missed=float((h.pct * b_cut / 100).sum()),
                        c_stat=(len(c), (c.pct > 0).mean() if len(c) else np.nan, c.pct.mean() if len(c) else np.nan),
                        u_stat=(len(u), (u.pct > 0).mean() if len(u) else np.nan, u.pct.mean() if len(u) else np.nan),
                        binds_v=kv.median() if len(kv) else np.nan, binds=k.median() if len(k) else np.nan, scored=len(k),
                        at={m: (k < m).mean() if len(k) else np.nan for m in MULTS}))
    return sorted(out, key=lambda r: (np.nan_to_num(r["binds"], nan=1e9), -r["n"]))


def slip_rows(d):
    """live fills only → per sleeve (capped n · median slip, uncapped n · median slip) (pure)."""
    out = []
    for sl, g in d[d.slip.notna()].groupby("sleeve"):
        c, u = g[g.capped].slip, g[~g.capped].slip
        out.append((sl, len(c), c.median() if len(c) else np.nan, len(u), u.median() if len(u) else np.nan))
    return out


def _p(x, f="{:.0%}"):
    return "–" if x is None or x != x else f.format(x)


def _stat(t):
    n, wr, av = t
    return "0" if not n else f"{n} · {wr:.0%} · {av:+.2f} %"


def _x(k):
    return "–" if k != k else (f"×{k:.1f}" if k < 100 else "×100+")


def table(d, title):
    L = [f"### {title}", "", "| Sleeve | N | Volume-capped · bracket-capped · gross-only | Median cut: volume · bracket | $ left out: volume · bracket / other "
         "| Any cap: N · WR · avg | Uncapped: N · WR · avg | Binds at (median × account): volume · any cap | Capped (any) at ×2 · ×5 · ×10 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sleeve_rows(d):
        at = " · ".join(_p(r["at"][m]) for m in MULTS)
        L.append(f"| {r['sleeve']} | {r['n']} | {r['vcap']} · {r['bcap']} · {r['gross']} | {_p(r['v_cutmed'])} · {_p(r['b_cutmed'])} "
                 f"| {r['v_missed']:+,.0f} · {r['b_missed']:+,.0f} | {_stat(r['c_stat'])} | {_stat(r['u_stat'])} | {_x(r['binds_v'])} · {_x(r['binds'])} "
                 f"({r['scored']} scored) | {at} |")
    return L + [""]


def load_batch(paths=None):
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for i, f in enumerate(paths):
        try:
            x = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if {"opened_at", "status", "entry_strategy"} <= set(x.columns):
            _lv = (x.is_paper.astype(str).str.lower() == "false") if "is_paper" in x else pd.Series(EG.is_live_export(f), index=x.index)
            fr.append(x.copy().assign(_i=i, _live=_lv))   # per-row mode: a live row keeps its flag whatever export it came from
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a["opened_at"] = a.opened_at.astype(str).str.replace(" ", "T").str[:19]
    a = a[a.opened_at >= BATCH_START]
    return a.sort_values("_i", kind="stable").drop_duplicates(["opened_at", "pair", "direction"], keep="last")


def run(now_ms=None, paths=None, master=MASTER):
    L = [f"## 💧📏 LIQUIDITY CAPS & CAPACITY per sleeve (DECISION_LOG {DL}, observe only — the first part of the capacity study)", "",
         "Two caps cut an order: the VOLUME cap (0.1 % of the coin's 24 h volume, spike trades 0.5 %, $500k ceiling) and the exchange's LEVERAGE "
         "BRACKET (Binance's max position per leverage tier). 'Binds at ×K' = how many times today's account the sleeve can grow before its wanted size "
         "meets the cap (median fill; ≤ ×1 = capped today; the bracket at the fill's leverage — conservative, a lower leverage holds more for "
         "more margin). '$ left out' = booked % × the "
         "size the cap cut (linear, no market impact — an upper bound; a capped loser's cut saved money). Full-size fills only, each sleeve apart.", ""]
    b = load_batch(paths)
    if len(b):
        bd = prep(b)
        L += table(bd, f"① Current batch (opened ≥ {BATCH_START[:16]} UTC, as traded)")
    else:
        bd = pd.DataFrame()
        L += ["### ① Current batch", "", "No orders export found.", ""]
    md = pd.DataFrame()
    if master and os.path.exists(master):
        m = pd.read_csv(master, low_memory=False)
        m = m[m.stack_keep.astype(str).str.lower().isin(_TRUE)]
        md = prep(m)
        L += table(md, "② Master per batch, pooled (kept by today's stack, AS TRADED size and % — '× account' is relative to the size each fill "
                       "traded at, and older batches traded some sleeves bigger: read ① for today's account)")
    live = []
    if len(md):
        live.append(md[md.era.astype(str).isin(LIVE_ERAS)])
    if len(bd) and "_live" in bd:
        live.append(bd[bd._live.astype(bool)])
    lv = pd.concat(live, ignore_index=True) if live else pd.DataFrame()
    L += ["### ③ Entry slippage, LIVE fills only (paper fills simulate ≈ 0)", ""]
    rows = slip_rows(lv) if len(lv) else []
    if rows:
        L += ["| Sleeve | Capped: N · median entry slip % | Uncapped: N · median entry slip % |", "|---|---|---|"]
        L += [f"| {s} | {nc} · {_p(mc, '{:+.3f}')} | {nu} · {_p(mu, '{:+.3f}')} |" for s, nc, mc, nu, mu in rows]
    else:
        L.append("No live fills with a slippage stamp.")
    L += ["", f"Read at every batch review (plan ②y / capacity, DECISION_LOG {DL}): a sleeve whose cap binds below the size step it is "
          "offered (WILLY 8–10×, Flatbase 0.5, Momentum Long ×5 / ×10) gains less than the best-case re-size row shows. OPEN DECISION (operator "
          "11-04): the real cap PER SLEEVE — today 0.1 % for all and 0.5 % for spike trades, the 0.5 set by declared overrides (0.1 → 0.2 → 0.3 → "
          "0.5, 'unverifiable until live'); set each sleeve's cap from its LIVE slippage vs order size (③), never from paper fills.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
