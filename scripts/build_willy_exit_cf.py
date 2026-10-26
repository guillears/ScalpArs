#!/usr/bin/env python3
"""Master-pool support (DECISION_LOG 282): every archived FRENZY_WILLY / FRENZY_WILLY_HT fill re-walked under TODAY's WILLY exit on its own
price path — ticks first, then the 1m tail (scripts/study_b19_fzexit_common.load_path, tick ruler = first print through the line; 28/28 live
parity on B19). WILLY's exit is path-dependent, so build_master_pool reads this table (reports/WILLY_EXIT_CF.csv) instead of the as-traded
pct; an unpriced fill keeps its as-traded pct (the builder lists those) and the builder refuses a CF priced under another exit. Cache-only.
Re-run whenever the WILLY exit changes or a batch is archived.

RULES (--rule, default v2b):
  v2b    — 🎲²ᵇ the LIVE exit since DECISION_LOG 356 → 358 (STACK 2026-10-26a, DECISION_LOG 377): per-order target
           T = max(+tp, min(cap, net at H × (1 − k %))) — services.frenzy.frenzy_willy_high_target, the engine's own function (entry taker fee
           % of the fill = entry_fee / (entry_price × quantity) + the MAKER exit fee, trading_config.json maker_fee) — resting maker TP at T,
           stop −stop THROUGHOUT, the late TP (+late_pct from minute late_min) for EVERY trade, the cap, NO floor. Walked with
           scout_willy_exit_v2.walk_v2 (no floor) — the walker the WILLY_EXIT_V2b scout uses; at T = +tp it is evaluate_late(book="live")
           exactly. H = the episode's spike high = frenzy_walk's `peak` at the entry pass = the highest 5m high from the spike bar
           (entry_frenzy_spike_at = the spike bar's CLOSE time → open − 5 min) through the red signal bar (the last 5m bar CLOSED at the
           fill). Fills since DL 356 carry the stamp entry_frenzy_willy_spike_high (and the T they ran, frenzy_willy_tp_target_pct — the
           stamped T is used as is); older fills: H rebuilt from the caches — 5m klines (backtest_cache/k5m_full) → complete 1m bars → full
           tick days (reconstruct_high; validated on the stamped fills, see --validate). No H: priced only when the walk at T = +tp does not
           end on the TP (then no T ≥ +tp changes the exit — "invariant"); else NO_H → unpriced (as traded in the master, listed).
           Extra columns: target_pct · t_src (stamp / rebuilt / invariant / none) · spike_high · h_src (stamp / 5m / 1m / ticks / none) ·
           entry_fee_pct · pre356_pct / pre356_why (the PINNED pre-356 exit on the same path — the master before → after).
           exit_rule = "V2b:T=max(+tp,min(cap,net@H*(1-k)))/-stop/cap/late<min>@+<pct>/rtp/nofloor".
           (`net@H*(1-k)` = the net % at the PRICE H × (1 − k), as frenzy_willy_high_target computes it — not net % × (1 − k).)
  pre356 — the pre-356 exit kept reproducible (STACK ≤ 2026-10-25a): +tp / −stop / cap / late / rtp, scout_willy_timecap.evaluate_late
           (book="live") — STACK 2026-10-16a note: a TP on the +tp line books EXACTLY the line (resting limit), a late-line TP books the line,
           or the print when the first event after the late minute is already through it; stop = first print through −stop; cap = last
           event ≤ the cap. FRENZY_WILLY only (as before). → reports/WILLY_EXIT_CF_PRE356.csv (never read by the master).
           exit_rule = "+tp/-stop/cap[/late<min>@+<pct>][/rtp]".
Fill assumption (both): a touch of the line fills the resting limit (paper books ≥ 1 tick through; queue / partial fills not modelled —
optimistic by ≤ 1 tick).
--validate GLOB: H / T rebuild vs the stamps on every stamped WILLY / HT fill of the matching exports (read-only; nothing written).
Usage: venv/bin/python -I scripts/build_willy_exit_cf.py [--rule v2b|pre356] [--validate '~/Downloads/scalpars_orders_*.csv']
"""
import glob, json, math, os, sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
import study_b19_fzexit_common as FX
OUT = os.path.join(ROOT, "reports", "WILLY_EXIT_CF.csv")
OUT_PRE356 = os.path.join(ROOT, "reports", "WILLY_EXIT_CF_PRE356.csv")
K5_DIR = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full")
STRATS = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
MIN, BAR5 = 60_000, 300_000
MAKER_FEE_DEF = 0.00018


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ms(v):
    try:
        ts = pd.Timestamp(str(v))
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return int((ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).value // 1_000_000)


def load_th(path=None):
    cfg = json.load(open(path or os.path.join(ROOT, "trading_config.json")))
    th = dict(cfg["thresholds"])
    mk = _num(cfg.get("maker_fee"))
    th["_maker_fee"] = mk if (mk is not None and 0 <= mk < 0.01) else MAKER_FEE_DEF
    return th


def pre356_params(th):
    tp = float(th.get("frenzy_willy_tp_pct") or 1.0); st = abs(float(th.get("frenzy_willy_stop_pct") or 0)) or None
    cap = int(th.get("frenzy_willy_max_hold_minutes") or 120)
    late_min = float(th.get("frenzy_willy_late_tp_minutes") or 0); late_pct = float(th.get("frenzy_willy_late_tp_pct") or 0)
    late_on = late_min > 0 and 0 < late_pct < tp                     # services.frenzy.frenzy_willy_late_tp: ≤ 0 or ≥ the TP = off
    rtp = bool(th.get("resting_tp_willy_enabled"))                   # DECISION_LOG 296
    return tp, st, cap, late_on, late_min, late_pct, rtp


def rule_pre356(th):
    tp, st, cap, late_on, late_min, late_pct, rtp = pre356_params(th)
    return (f"+{tp:g}/-{st:g}/{cap}" if st else f"+{tp:g}/none/{cap}") + (f"/late{late_min:g}@+{late_pct:g}" if late_on else "") + ("/rtp" if rtp else "")


def v2b_on(th):
    return th.get("frenzy_willy_high_target_enabled", True) is True


def rule_v2b(th):
    """the V2b rule string (DECISION_LOG 358); with the high target switched off the live exit IS the pre-356 rule."""
    if not v2b_on(th):
        return rule_pre356(th)
    tp, st, cap, late_on, late_min, late_pct, rtp = pre356_params(th)
    k = float(th.get("frenzy_willy_high_k_pct", 1.0)); mx = float(th.get("frenzy_willy_high_tp_max_pct", 8.0))
    return (f"V2b:T=max({tp:g},min({mx:g},net@H*{1 - k / 100:g}))/" + (f"-{st:g}" if st else "none") + f"/{cap}"
            + (f"/late{late_min:g}@+{late_pct:g}" if late_on else "") + ("/rtp" if rtp else "") + "/nofloor")


def walk_pre356(P, th):
    tp, st, cap, late_on, late_min, late_pct, rtp = pre356_params(th)
    if not late_on and not rtp:
        return FX.evaluate(P, tp, st or 1e9, cap)                    # the pre-296/297 walker (print booking)
    lm, lp = (late_min, late_pct) if late_on else (float(cap) + 1.0, tp)
    return FX.WT.evaluate_late(P, tp, st or 1e9, cap, lm, lp, book="live" if rtp else "print")


def walk_v2b(P, T, th):
    """the V2b exit at target T on a net path → dict(why TP / LATE_TP / STOP / CAP / UNRES, pct, xmin) — scout_willy_exit_v2.walk_v2 (no
    floor) + the exit minute (the shortest event prefix on which the walk already decides; walk_v2 is prefix-monotone)."""
    import scout_willy_exit_v2 as SV
    tp, st, cap, late_on, late_min, late_pct, rtp = pre356_params(th)
    kw = dict(sl=st or 0, cap=cap, late=(late_min if late_on else None), ztp=(late_pct if late_on else None))
    w = SV.walk_v2(P, T, **kw)
    if w["why"] == "UNRES":
        return dict(why="UNRES", pct=float("nan"), xmin=float("nan"))
    te, t = int(P["te"]), P["t"]
    if w["why"] == "CAP":                                             # the cap closes at the cap minute (evaluate_late's convention)
        return dict(why="CAP", pct=float(w["pct"]), xmin=float(cap))
    lo_, hi_ = 1, int(np.searchsorted(t, te + cap * MIN, side="right"))
    while lo_ < hi_:                                                  # smallest m whose prefix already exits
        m = (lo_ + hi_) // 2
        Q = dict(P, t=t[:m], hi=P["hi"][:m], lo=P["lo"][:m], cl=P["cl"][:m], op=P["op"][:m], bar=P["bar"][:m], end_ms=0)
        if SV.walk_v2(Q, T, **kw)["why"] == "UNRES":
            lo_ = m + 1
        else:
            hi_ = m
    return dict(why=w["why"], pct=float(w["pct"]), xmin=(int(t[lo_ - 1]) - te) / MIN)


# ─────────────────────────── H (the spike high) ───────────────────────────
_K5 = {}


def _k5(pair):
    if pair not in _K5:
        f = os.path.join(K5_DIR, f"{pair}.csv")
        try:
            _K5[pair] = pd.read_csv(f, usecols=["open_time", "h"]) if os.path.exists(f) else None
        except Exception:
            _K5[pair] = None
    return _K5[pair]


def h_window(spike_at_ms, te_ms):
    """(first bar open, last bar open) of the engine's peak window: the spike bar (spike_at = its CLOSE time) through the last 5m bar CLOSED
    at the fill (the red signal bar). None when the window is empty / inverted."""
    b0, last = int(spike_at_ms) - BAR5, (int(te_ms) // BAR5) * BAR5 - BAR5
    return (b0, last) if last >= b0 else None


def reconstruct_high(pair, spike_at_ms, te_ms, tcache=None):
    """→ (H, source '5m' / '1m' / 'ticks') or (None, 'none'): max high over the h_window, from the FIRST cache that covers it completely —
    5m klines (backtest_cache/k5m_full) → 1m bars (scout_willy_timecap.load_1m: every minute present) → full tick days. Cache-only."""
    w = h_window(spike_at_ms, te_ms) if (spike_at_ms is not None and te_ms is not None) else None
    if w is None:
        return None, "none"
    b0, last = w
    k = _k5(pair)
    if k is not None:
        s = k[(k.open_time >= b0) & (k.open_time <= last)]
        if s.open_time.nunique() == (last - b0) // BAR5 + 1 and s.h.notna().all():
            return float(s.h.max()), "5m"
    try:
        k1 = FX.WT.load_1m(pair)
        s = k1[(k1.open_time >= b0) & (k1.open_time < last + BAR5)]
        if s.open_time.nunique() == (last + BAR5 - b0) // MIN and s.h.notna().all():
            return float(s.h.max()), "1m"
    except Exception:
        pass
    days = [f"{d:%Y-%m-%d}" for d in pd.date_range(pd.Timestamp(b0, unit='ms').normalize(),
                                                     pd.Timestamp(last + BAR5 - 1, unit='ms').normalize(), freq="D")]   # every day the window spans
    mx = None
    for d in days:
        z = FX.WT._load_day(pair, d, tcache if tcache is not None else {})
        if z is None or z[2] is not None:                            # a window slice cannot prove the whole bar range was seen
            return None, "none"
        t, p = z[0], z[1]
        m = (t >= b0) & (t < last + BAR5)
        if m.any():
            v = float(np.max(p[m]))
            mx = v if mx is None else max(mx, v)
    return (mx, "ticks") if mx is not None else (None, "none")


def entry_fee_pct(r):
    e, q, fee = _num(r.get("entry_price")), _num(r.get("quantity")), _num(r.get("entry_fee"))
    if not (e and e > 0 and q and q > 0) or fee is None or fee < 0:
        return None
    return fee / (e * q) * 100.0


def target_from_h(th, r, H):
    """the engine's T_used for this fill at spike high H (services.frenzy.frenzy_willy_high_target) → (T_used, T raw or None)."""
    from services.frenzy import frenzy_willy_high_target
    return frenzy_willy_high_target(SimpleNamespace(**{k: v for k, v in th.items() if not k.startswith("_")}),
                                    _num(r.get("entry_price")), H, entry_fee_pct(r), th["_maker_fee"])


def resolve_target(th, r, tcache=None):
    """→ dict(T, t_src, H, h_src, T_calc): the stamped T when the fill ran one (V2 / V2b era), else T from the stamped / rebuilt H."""
    H, h_src = _num(r.get("entry_frenzy_willy_spike_high")), "stamp"
    if not (H and H > 0):
        H, h_src = reconstruct_high(r.get("pair"), _ms(r.get("entry_frenzy_spike_at")), _ms(r.get("opened_at")), tcache)
    tc = None
    if H:
        tu, traw = target_from_h(th, r, H)
        tc = tu if traw is not None else None
    ts = _num(r.get("frenzy_willy_tp_target_pct"))
    if ts is not None and ts > 0:
        return dict(T=ts, t_src="stamp", H=H, h_src=h_src if H else "none", T_calc=tc)
    if tc is not None:
        return dict(T=tc, t_src="rebuilt", H=H, h_src=h_src, T_calc=tc)
    return dict(T=None, t_src="none", H=H, h_src=h_src if H else "none", T_calc=None)


def archived_fills(strats=STRATS):
    fr = []
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE[0-9]*_*orders*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" not in d:
            continue
        d = d[d.entry_strategy.astype(str).isin(strats) & (d.status == "CLOSED")]
        if len(d):
            fr.append(d)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def price_row(r, th, cache, tcache, rule):
    """one V2b CF row for a CLOSED WILLY / HT fill (dict-like r)."""
    tp, st, cap = pre356_params(th)[:3]
    te = _ms(r.get("opened_at"))
    E = _num(r.get("entry_price"))
    P = FX.load_path(r.get("pair"), te, E, cap, cache) if (te is not None and E and E > 0) else dict(t=[])
    src = "none" if not len(P["t"]) else ("ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m"))
    tg = resolve_target(th, r, tcache) if v2b_on(th) else dict(T=tp, t_src="switch_off", H=None, h_src="none", T_calc=None)
    out = dict(opened_at=str(r.get("opened_at"))[:19], pair=r.get("pair"), entry_strategy=str(r.get("entry_strategy")),
               live_pct=_num(r.get("pnl_percentage")), cf_pct=None, cf_why="UNRES", cf_min=float("nan"), src=src, exit_rule=rule,
               target_pct=tg["T"], t_src=tg["t_src"], spike_high=tg["H"], h_src=tg["h_src"], target_calc=tg["T_calc"],
               stamp_target=_num(r.get("frenzy_willy_tp_target_pct")), stamp_high=_num(r.get("entry_frenzy_willy_spike_high")),
               entry_fee_pct=entry_fee_pct(r), pre356_pct=None, pre356_why="UNRES")
    if not len(P["t"]):
        return out
    old = walk_pre356(P, th)
    out.update(pre356_pct=None if old["why"] == "UNRES" else round(float(old["pct"]), 4), pre356_why=old["why"])
    T = tg["T"]
    if T is None:                                                     # no H: the exit is T-invariant unless the +tp walk ends on its TP
        w = walk_v2b(P, tp, th)
        if w["why"] in ("UNRES", "TP"):
            out.update(cf_why="NO_H" if w["why"] == "TP" else "UNRES")
            return out
        out.update(t_src="invariant", target_pct=None)
    else:
        w = walk_v2b(P, T, th)
    if w["why"] != "UNRES":
        out.update(cf_pct=round(float(w["pct"]), 4), cf_why=w["why"], cf_min=w["xmin"])
    return out


def build_v2b(th=None):
    th = th or load_th()
    rule = rule_v2b(th)
    d = archived_fills()
    cache, tcache = {}, {}
    R = pd.DataFrame([price_row(r, th, cache, tcache, rule) for r in d.to_dict("records")])
    R.to_csv(OUT, index=False)
    print(f"{len(R)} WILLY / WILLY_HT fills → {OUT}  ({rule})")
    if len(R):
        un = R[R.cf_pct.isna()]
        print(f"  H: {dict(R.h_src.value_counts())} · T: {dict(R.t_src.value_counts())} · unpriced {len(un)}"
              + (": " + ", ".join(f"{a} {p} ({w})" for a, p, w in zip(un.opened_at, un.pair, un.cf_why)) if len(un) else ""))
        print(R[["opened_at", "pair", "entry_strategy", "live_pct", "pre356_pct", "pre356_why", "target_pct", "t_src", "h_src", "cf_pct",
                 "cf_why", "cf_min", "src"]].round(3).to_string(index=False))
    return R


def build_pre356(th=None):
    """the pre-356 table exactly as STACK ≤ 2026-10-25a priced it (FRENZY_WILLY only) — kept reproducible, never read by the master."""
    th = th or load_th()
    rule = rule_pre356(th)
    tp, st, cap = pre356_params(th)[:3]
    rows, cache = [], {}
    for r in archived_fills(("FRENZY_WILLY",)).itertuples():
        te = int(pd.Timestamp(r.opened_at).value // 1_000_000)
        P = FX.load_path(r.pair, te, float(r.entry_price), cap, cache)
        ev = walk_pre356(P, th) if len(P["t"]) else dict(why="UNRES", pct=float("nan"), xmin=float("nan"))
        src = "none" if not len(P["t"]) else ("ticks" if P["n_bar"] == 0 else ("1m" if P["n_tick"] == 0 else "ticks+1m"))
        rows.append(dict(opened_at=str(r.opened_at)[:19], pair=r.pair, entry_strategy="FRENZY_WILLY", live_pct=float(r.pnl_percentage),
                         cf_pct=None if ev["why"] == "UNRES" else round(float(ev["pct"]), 4), cf_why=ev["why"], cf_min=ev["xmin"], src=src,
                         exit_rule=rule))
    R = pd.DataFrame(rows)
    R.to_csv(OUT_PRE356, index=False)
    print(f"{len(R)} WILLY fills → {OUT_PRE356}  ({rule})")
    return R


def validate(pattern, th=None):
    """H / T rebuild vs the stamps on every stamped WILLY / HT fill in the exports matching pattern (dedup opened_at[:19] + pair)."""
    th = th or load_th()
    fr = []
    for f in sorted(glob.glob(os.path.expanduser(pattern))):
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" in d and "entry_frenzy_willy_spike_high" in d:
            fr.append(d[d.entry_strategy.astype(str).isin(STRATS)])
    if not fr:
        print("validate: no export with the stamp"); return None
    d = pd.concat(fr, ignore_index=True)
    d = d[pd.to_numeric(d.entry_frenzy_willy_spike_high, errors="coerce") > 0]
    d = d.assign(_k=d.opened_at.astype(str).str[:19] + d.pair.astype(str)).drop_duplicates("_k", keep="last")
    rows, tcache = [], {}
    for r in d.to_dict("records"):
        Hs = float(r["entry_frenzy_willy_spike_high"])
        H, src = reconstruct_high(r["pair"], _ms(r.get("entry_frenzy_spike_at")), _ms(r["opened_at"]), tcache)
        Ts = _num(r.get("frenzy_willy_tp_target_pct"))
        Tc = target_from_h(th, r, Hs)[0] if Ts is not None else None
        rows.append(dict(opened_at=str(r["opened_at"])[:19], pair=r["pair"], strat=r["entry_strategy"], H_stamp=Hs, H_rebuilt=H, h_src=src,
                         H_err_pct=(abs(H / Hs - 1) * 100 if H else None), T_stamp=Ts, T_from_stamped_H=Tc,
                         T_err=(abs(Tc - Ts) if (Tc is not None and Ts is not None) else None)))
    V = pd.DataFrame(rows)
    ok = V[V.H_rebuilt.notna()]
    print(f"validate: {len(V)} stamped fills · H rebuilt on {len(ok)} (no cache: {len(V) - len(ok)}) · H within 0.01 %: "
          f"{int((ok.H_err_pct < 0.01).sum())}/{len(ok)} · max H err {ok.H_err_pct.max() if len(ok) else float('nan'):.5f} % · "
          f"T (engine fn on the stamped H) max |err| {V.T_err.max():.5f}")
    print(V.round(6).to_string(index=False))
    return V


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if "--validate" in a:
        return validate(a[a.index("--validate") + 1])
    rule = a[a.index("--rule") + 1] if "--rule" in a else "v2b"
    if rule not in ("v2b", "pre356"):
        raise SystemExit(f"--rule {rule}: use v2b or pre356")
    return build_v2b() if rule == "v2b" else build_pre356()


if __name__ == "__main__":
    main()
