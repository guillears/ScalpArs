"""📈 CAPACITY PATH (DECISION_LOG 513) — days to $50k / $100k / $200k / $500k / $1M at a start %/day, with each sleeve's coin-volume caps,
the leverage schedule and the reserve schedule bending the rate as the account grows (operator 2026-11-09: "you are not considering the
capacity limit per sleeve" · "add the 5 % row").

Method (read-only, no network):
  · fills = CLOSED full-size bot fills (MANUAL + *_PROBE out) of the export opened ≥ --since (default the B25 10-26 13:50 UTC top-up).
  · each fill: wanted size as a share of the tradeable line at its open (equity at open = --start-equity + realised $ of EVERY fill closed
    between --since and its open), cap = min(entry_liquidity_cap_notional, entry_bracket_cap_notional).
  · at balance E: wanted = share × T(E) × min(lev, L(E)) ÷ lev (T = trading_config reserve schedule + fee reserve, L = the
    leverage_balance_schedule ceiling — 20× → 15× above $25k), traded = min(wanted, cap); a sleeve's "size kept" = Σ traded ÷ Σ share × T(E),
    RELATIVE to the start balance (today's rate already carries today's caps and leverage).
  · rate(E) = start rate × Σ_sleeves sleeve share × size kept(E) × [T(E)/E ÷ T(E0)/E0]; sleeve shares = the positive "best" values of the
    newest reports/best_case/BEST_CASE_*.md table ① (current batch), scaled to 1.
  · walk day by day from the day after the export: E ← E × (1 + rate(E)); the date is the day the target is first reached.
Approximations (not modelled): coin volumes move with the market (hot markets loosen caps) · each sleeve's cap profile uses ALL its fills
since --since, while its share is its best-line value (a subset of fills; table ① also counts from the batch start, not the top-up) ·
every start rate uses the same sleeve mix · paper-balance edits after --since are not in equity at open · the gross-notional cap · new
deep-coin sleeves.

Usage: venv/bin/python -I scripts/best_case/capacity_path.py <orders export.csv> [--rates 8,5] [--start 10000] [--since "2026-10-26 13:50"]
       [--start-equity 10100] [--no-reserve] [--best-case reports/best_case/BEST_CASE_….md] [--resize SLEEVE=f,…]
Saved runs: reports/best_case/CAPACITY_PATH_<date>.md.
"""
import argparse, glob, json, math, os, re, sys, warnings
from types import SimpleNamespace
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from services.trading_engine import _reserve_schedule_target as SCHED, _lookup_leverage_schedule as LEVSCHED
from services.capital_use import sleeve_of

NAMES = {"LITE RETRY": "FRENZY_LITE_RETRY", "WILLY": "FRENZY_WILLY", "Flatbase": "FLATBASE_SHORT", "FAN flips": "FLIP:FAN_RATIO_GATE",
         "Spike Fade": "SPIKE_FADE", "SURGE Long": "SURGE_LONG", "LITE": "FRENZY_LITE", "WILLY_HT": "FRENZY_WILLY_HT",
         "Momentum Long": "MOMENTUM_LONG", "FRENZY_LONG": "FRENZY_LONG", "Bear-Run Short": "BEARRUN_SHORT", "Momentum Short": "MOMENTUM_SHORT"}
TARGETS = (50e3, 100e3, 200e3, 500e3, 1e6)
BALANCES = (10e3, 25e3, 50e3, 100e3, 200e3, 500e3, 1e6, 2e6)
LOW_N = 10
MAX_DAYS = 10000


def best_case_shares(path, warn=print):
    """table ① 'Current batch, before → best' → {sleeve: best %/day} (the last number in the cell; 'no line helps' rows = their one value).
    Table ① only (①b / ①c reuse the sleeve names with other columns); a row whose name is not in NAMES is reported, never silently dropped."""
    out, on = {}, False
    for ln in open(path, encoding="utf-8"):
        if on and ln.startswith("## "):
            break
        if ln.startswith("## ① "):
            on = True
            continue
        if on and ln.startswith("| ") and not ln.startswith("| Sleeve") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            if len(c) < 3:
                continue
            if c[0] not in NAMES:
                warn(f"⚠ table ① row '{c[0]}' is not a known sleeve name — left out of the shares (add it to NAMES)")
                continue
            nums = re.findall(r"[+−-]?\d+(?:\.\d+)?", c[2].replace("**", ""))
            if nums and NAMES[c[0]] not in out:
                out[NAMES[c[0]]] = float(nums[-1].replace("−", "-"))
    return out


def tradeable_fn(no_reserve=False, inv=None):
    """balance → tradeable: the engine's schedule-mode reserve line (mirror of main._capuse_tradeable_fn). inv = the trading_config
    'investment' dict (default: read trading_config.json); a reserve_mode other than 'schedule' is refused, not silently mis-modelled."""
    if no_reserve:
        return lambda e: max(0.0, float(e))
    if inv is None:
        cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
        inv = cfg.get("investment", cfg)
    if inv.get("reserve_mode") != "schedule":
        raise SystemExit(f"reserve_mode '{inv.get('reserve_mode')}' — only 'schedule' is modelled; run with --no-reserve or extend tradeable_fn")
    sch = inv.get("reserve_schedule", "")
    fee_usd = max(0.0, float(inv.get("fee_reserve_usd", 0) or 0))
    fee_pct = max(0.0, float(inv.get("fee_reserve_pct", 0) or 0))

    def T(e):
        e = max(0.0, float(e or 0.0))
        t = SCHED(sch, e)
        t = t if (t is not None and t > 0) else None
        fee = max(fee_usd, fee_pct / 100 * (min(e, t) if t else e))
        return max(0.0, e - (max(e - t, fee) if t else fee))
    return T


def kept_at(fills, sl, E, T, lev_schedule=""):
    """share of a sleeve's wanted size the coin caps + leverage ceiling let through at balance E (1.0 when it has no fills). PURE."""
    g = fills[fills.sl == sl]
    if not len(g):
        return 1.0
    base = g.frac * T(E)
    lc = LEVSCHED(lev_schedule, E)
    w = base * (np.minimum(g.lev, lc) / g.lev if lc else 1.0)
    tr = np.where(g.cap.notna(), np.minimum(w, g.cap), w)
    return float(tr.sum() / base.sum()) if base.sum() > 0 else 1.0


def build(fills, share, T, start, lev_schedule=""):
    """fills = DataFrame with sl, frac (wanted ÷ T at open), cap (NaN = none), lev → (rate(E, R), rel(sleeve, E)). PURE."""
    kept = lambda sl, E: kept_at(fills, sl, E, T, lev_schedule)
    k0 = {s: kept(s, start) for s in share}
    rel = lambda s, E: kept(s, E) / k0[s] if k0[s] > 0 else 1.0
    r0 = T(start) / start

    def rate(E, R):
        return R * sum(share[s] * rel(s, E) for s in share) * (T(E) / E / r0)
    return rate, rel


def days_to(rate, R, start, targets=TARGETS, max_days=MAX_DAYS):
    """→ {target: first day the balance is ≥ target} (day 1 = the first day walked; a target ≤ start = 0)."""
    E, day, out = start, 0, {t: 0 for t in targets if t <= start}
    while day < max_days and len(out) < len(targets):
        E *= 1 + rate(E, R) / 100
        day += 1
        for t in targets:
            if t not in out and E >= t:
                out[t] = day
    return out


def load_fills(path, since, start_equity, T):
    d = pd.read_csv(path, low_memory=False)
    for c in ("opened_at", "closed_at"):
        d[c[0]] = pd.to_datetime(d[c], utc=True, errors="coerce", format="ISO8601").dt.tz_convert(None)
    s0 = pd.Timestamp(since)
    allc = d[(d.status.astype(str) == "CLOSED") & d.c.notna() & (d.c >= s0)]
    cl = allc[["c"]].assign(pnl=pd.to_numeric(allc.pnl, errors="coerce").fillna(0.0)).sort_values("c")
    man = d.entry_strategy.astype(str).str.upper().eq("MANUAL") | d.get("confidence", pd.Series("", index=d.index)).astype(str).str.upper().eq("MANUAL")
    probe = d.get("cell_multiplier_source", pd.Series("", index=d.index)).astype(str).str.endswith("_PROBE")
    d = d[(d.status.astype(str) == "CLOSED") & ~man & ~probe & (d.o >= s0)].copy()
    d["E_open"] = [start_equity + cl.pnl[cl.c < t].sum() for t in d.o]          # every fill closed since --since (manual / probe included)
    d["want"] = pd.to_numeric(d.entry_desired_notional, errors="coerce")
    d["cap"] = np.fmin(pd.to_numeric(d.entry_liquidity_cap_notional, errors="coerce"),
                       pd.to_numeric(d.entry_bracket_cap_notional, errors="coerce"))
    d["lev"] = pd.to_numeric(d.leverage, errors="coerce").clip(lower=1).fillna(1.0)
    d = d[d.want > 0].copy()
    d["frac"] = d.want / d.E_open.map(T)
    d["sl"] = [sleeve_of(SimpleNamespace(entry_strategy=(s if isinstance(s, str) and s else None), direction=r))
               for s, r in zip(d.entry_strategy, d.direction.astype(str))]
    return d, (d.c.max() if len(d) else pd.Timestamp.utcnow().tz_localize(None))


def parse_resize(txt):
    """"FRENZY_WILLY=2.05,FRENZY_WILLY_HT=4.09" → {sleeve: factor}; a bad item or a factor ≤ 0 stops the run. PURE."""
    out = {}
    for it in (x.strip() for x in str(txt or "").split(",")):
        if not it:
            continue
        k, _, v = it.partition("=")
        try:
            f = float(v)
        except ValueError:
            raise SystemExit(f"--resize item '{it}' — use SLEEVE=factor")
        if not k.strip() or not f > 0:
            raise SystemExit(f"--resize item '{it}' — the factor must be > 0")
        out[k.strip()] = f
    return out


def resize_shares(share, fills, resized, rsz, T, start, lev_schedule=""):
    """(DL 514 review) a re-sized sleeve's share of the start rate = its one-size share × f × (what its caps let through at the start at the
    new size ÷ at today's size) — a bigger size already capped at the start adds less than f. Not normalised. PURE."""
    out = {}
    for k, v in share.items():
        f = rsz.get(k, 1.0)
        if f != 1.0:
            k_old = kept_at(fills, k, start, T, lev_schedule)
            v = v * f * (kept_at(resized, k, start, T, lev_schedule) / k_old if k_old > 0 else 1.0)
        out[k] = v
    return out


def apply_resize(fills, rsz):
    """each re-sized sleeve's wanted size × its factor (the coin cap does not move). PURE (returns a copy)."""
    d = fills.copy()
    for sl, f in rsz.items():
        d.loc[d.sl == sl, "frac"] = d.loc[d.sl == sl, "frac"] * f
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--rates", default="8,5", help="start %%/day, comma list (operator 11-09: 8 = in-sample best case, 5 = luck-checked)")
    ap.add_argument("--start", type=float, default=10000.0)
    ap.add_argument("--since", default="2026-10-26 13:50")
    ap.add_argument("--start-equity", type=float, default=10100.0)
    ap.add_argument("--no-reserve", action="store_true")
    ap.add_argument("--best-case", default=None)
    ap.add_argument("--resize", default="", help="SLEEVE=factor,… — a sleeve re-sized to factor × today's size (DL 514: the best-case C5 "
                    "row, e.g. FRENZY_WILLY=2.05,FRENZY_WILLY_HT=4.09): its share of the rate AND its wanted size × factor (it meets its coin caps sooner)")
    a = ap.parse_args()
    bcs = sorted(glob.glob(os.path.join(ROOT, "reports/best_case/BEST_CASE_*.md")))
    bc = a.best_case or (bcs[-1] if bcs else None)
    if not bc:
        raise SystemExit("no reports/best_case/BEST_CASE_*.md — run scripts/best_case/run_best_case.py first or pass --best-case")
    rsz = parse_resize(a.resize)
    share = {k: v for k, v in best_case_shares(bc).items() if v > 0}
    if set(rsz) - set(share):   # (DL 514 review) a typo / a sleeve not in table ①'s positive rows would silently do nothing
        raise SystemExit(f"--resize {sorted(set(rsz) - set(share))} not among the positive table ① sleeves: {sorted(share)}")
    rates = [float(x) for x in a.rates.split(",") if x.strip()]
    T = tradeable_fn(a.no_reserve)
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    lev_sched = (cfg.get("investment", cfg) or {}).get("leverage_balance_schedule", "") or cfg.get("leverage_balance_schedule", "")
    d0, last = load_fills(a.export, a.since, a.start_equity, T)
    d = apply_resize(d0, rsz)
    share = resize_shares(share, d0, d, rsz, T, a.start, lev_sched)
    tot = sum(share.values())
    if tot <= 0:
        raise SystemExit(f"no positive sleeve in table ① of {bc}")
    share = {k: v / tot for k, v in share.items()}
    rate, rel = build(d, share, T, a.start, lev_sched)

    print(f"best case: {os.path.basename(bc)} · export {os.path.basename(a.export)} · fills since {a.since}: {len(d)} · "
          f"reserve {'OFF' if a.no_reserve else 'schedule'} · leverage schedule '{lev_sched or 'none'}' · "
          + " / ".join(f"{R:g}" for R in rates) + f" %/day at ${a.start:,.0f}")
    if rsz:
        print("re-sized (share + wanted size): " + " · ".join(f"{k} ×{v:g}" for k, v in rsz.items()))
    n = {s: int((d.sl == s).sum()) for s in share}
    print("sleeve shares of the rate: " + " · ".join(f"{s} {v:.0%} ({n[s]} fills{' ⚠ few' if n[s] < LOW_N else ''})" for s, v in share.items()))
    if any(v == 0 for v in n.values()):
        print("⚠ a sleeve with 0 fills is never capped here (optimistic): " + ", ".join(s for s, v in n.items() if v == 0))
    print(f"\n| Balance | " + " | ".join(f"Rate at {R:g} % start" for R in rates) + " | " + " | ".join(share) + " |")
    print("|---|" + "---|" * (len(rates) + len(share)))
    for E in BALANCES:
        print(f"| ${E:,.0f} | " + " | ".join(f"{rate(E, R):.2f}" for R in rates) + " | " + " | ".join(f"{rel(s, E):.0%}" for s in share) + " |")
    d1 = last.normalize() + pd.Timedelta(days=1)       # day 1 = the day after the export's last close
    for R in rates:
        capped = days_to(rate, R, a.start)
        plain = {t: (math.ceil(math.log(t / a.start) / math.log(1 + R / 100)) if t > a.start else 0) for t in TARGETS}
        print(f"\n**{R:g} %/day start** (day 1 = {d1:%b %-d, %Y})\n\n| Target | No caps, no reserve | With caps{'' if a.no_reserve else ' + reserve'}"
              f" + leverage schedule | ≈ Date reached |\n|---|---|---|---|")
        for t in TARGETS:
            c = capped.get(t)
            print(f"| ${t:,.0f} | {plain[t]} | **{c if c is not None else f'> {MAX_DAYS:,}'}** | "
                  f"{(d1 + pd.Timedelta(days=c - 1)).strftime('%b %-d, %Y') if c else '–'} |")


if __name__ == "__main__":
    main()
