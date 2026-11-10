"""⚙️ Re-size → the settings to type (operator 11-10: "be clear on INV size 1x, or what, and lev size 10x (0.5) for example").

Turns each best-case re-size factor (a × on the traded size, DL 502 / 503 / 504 / 514) into the two numbers the dashboard takes:
INVESTMENT (a share of one slot: (total − reserve) ÷ max_open_positions) and LEVERAGE (a whole number). The engine opens at
round(confidence leverage × lev setting) (trading_engine.compose_leverage; STRONG_BUY = 20×), and leverage_balance_schedule is only a
CEILING applied after ("0:20" since DL 517: a 20× ceiling; the old 25000:15 step cut only legs above 15×). Size = investment × leverage.

ONE RULE for every sleeve (the WILLY call agreed 11-09: "1× investment at 5×"; 0.05 steps, operator 11-10):
  1. keep today's leverage, scale the investment by the factor;
  2. up to ONE slot (an investment 1.00–1.05 is rounded down to 1.00 — the size lands ≤ 5 % under the target, never over the cap);
  3. above one slot, raise the leverage to the smallest whole number that keeps the investment ≤ 1 slot, then trim the investment
     to the size (never above the base leverage — then the investment stays above one slot, flagged ⚠);
  4. the investment is rounded DOWN to 0.05 (never over the target → never over the 5 % cap) and clamped to the engine's bounds
     [0.1, rsi_adx_multiplier_hard_cap] (a clamp that moves the size is flagged ⚠).
WILLY_HT is sized off WILLY (frenzy_willy_ht_size_frac × WILLY's investment, WILLY's leverage): its fraction is solved against
WILLY's NEW settings when WILLY moves in the same row (0.05 … 1.0; the engine reads ≤ 0 as 0.25, > 1 is refused).
Momentum Long: the engine rounds investment × leverage jointly (momentum_long_resize) and a pattern cell's multipliers apply on top.
PURE — no I/O except load_cfg.
"""
import json
import math

FIELDS = {   # sleeve → (investment setting, leverage setting)
    "FRENZY_LITE_RETRY": ("frenzy_lite_retry_invest_mult", "frenzy_lite_retry_lev_mult"),
    "FRENZY_LITE": ("frenzy_lite_invest_mult", "frenzy_lite_lev_mult"),
    "FLATBASE_SHORT": ("flatbase_invest_mult", "flatbase_lev_mult"),
    "MOMENTUM_LONG": ("momentum_long_invest_mult", "momentum_long_lev_mult"),
    "FRENZY_WILLY": ("frenzy_willy_invest_mult", "frenzy_willy_lev_mult"),
    "FRENZY_LONG": ("frenzy_long_invest_mult", "frenzy_long_lev_mult"),
}
HT = "FRENZY_WILLY_HT"
HT_FIELD = "frenzy_willy_ht_size_frac"
ROUND_TOL = 1.05
STEP = 0.05          # investment rounding, always DOWN (operator 11-10: "round down to 0.05")
INV_MIN = 0.1        # trading_engine: cell_mult = max(0.1, min(inv, rsi_adx_multiplier_hard_cap))


def load_cfg(path):
    """→ the flat settings dict: trading_config.json keeps the sleeve fields under 'thresholds', the schedule / slot count under
    'investment' and the leverage under 'confidence_levels' (kept as is)."""
    with open(path) as fh:
        c = json.load(fh)
    return {**c, **(c.get("investment") or {}), **(c.get("thresholds") or {})}


def base_leverage(cfg) -> float:
    """the leverage a STRONG_BUY signal opens at before the sleeve setting (confidence_levels.STRONG_BUY.leverage); 20 when unreadable."""
    try:
        v = float(cfg["confidence_levels"]["STRONG_BUY"]["leverage"])
        return v if v >= 1 else 20.0
    except (KeyError, TypeError, ValueError):
        return 20.0


def schedule_ceilings(cfg):
    """leverage_balance_schedule '0:20, 25000:15' → [(0, 20), (25000, 15)] ([] when unreadable)."""
    out = []
    for part in str(cfg.get("leverage_balance_schedule", "") or "").split(","):
        try:
            b, lv = part.split(":")
            out.append((float(b), float(lv)))
        except ValueError:
            continue
    return out


def slots(cfg) -> int:
    try:
        return int(cfg.get("max_open_positions", 4)) or 4
    except (TypeError, ValueError):
        return 4


def _lev(cfg, mult) -> int:
    return max(1, int(round(base_leverage(cfg) * float(mult))))


def _floor(v, step):
    return round(math.floor(v / step + 1e-9) * step, 4)


def today(cfg, sleeve, willy=None):
    """→ dict(inv, lev, inv_set, lev_set, inv_key, lev_key) of today's settings. inv = share of one slot."""
    if sleeve == HT:
        w = willy or today(cfg, "FRENZY_WILLY")
        frac = float(cfg.get(HT_FIELD, 0.25))
        return dict(inv=frac * w["inv"], lev=w["lev"], inv_set=frac, lev_set=w["lev_set"], inv_key=HT_FIELD, lev_key=w["lev_key"])
    ik, lk = FIELDS[sleeve]
    inv, lm = float(cfg[ik]), float(cfg[lk])
    return dict(inv=inv, lev=_lev(cfg, lm), inv_set=inv, lev_set=lm, inv_key=ik, lev_key=lk)


def recommend(cfg, sleeve, factor, willy_new=None, step=STEP):
    """→ dict(today, new, size_x, flag) — new = the settings after the ONE RULE (module doc); size_x = new size ÷ today's size;
    flag = '' or why the size misses the rule (⚠ in the table)."""
    base = base_leverage(cfg)
    t = today(cfg, sleeve)
    target = t["inv"] * t["lev"] * float(factor)
    if sleeve == HT:
        w = willy_new or today(cfg, "FRENZY_WILLY")
        raw = target / (w["inv"] * w["lev"])
        frac = max(step, min(_floor(min(raw, 1.0), step), 1.0))
        flag = "HT capped at WILLY's size (1.0)" if raw > ROUND_TOL else ("above target (min step)" if frac > raw + 1e-9 else "")
        new = dict(inv=frac * w["inv"], lev=w["lev"], inv_set=frac, lev_set=w["lev_set"], inv_key=HT_FIELD, lev_key=w["lev_key"])
        return dict(today=t, new=new, size_x=(new["inv"] * new["lev"]) / (t["inv"] * t["lev"]), flag=flag)
    if sleeve == "MOMENTUM_LONG":   # engine: notional = slot × cell × round(base × lev_mult × inv_mult) — jointly rounded (deep review 516)
        lev = min(int(base), max(1, int(math.floor(target + 1e-9))))
        flag = "UNMATCHED long cell ×1.5 → 1.5 slots of margin"
        if target > base:
            flag = f"capped at {int(base)}×; " + flag
        new = dict(inv=1.0, lev=lev, inv_set=1.0, lev_set=round(lev / base, 4), inv_key=t["inv_key"], lev_key=t["lev_key"])
        return dict(today=t, new=new, size_x=lev / (t["inv"] * t["lev"]), flag=flag)
    flag = ""
    lev = t["lev"]
    inv = target / lev
    if 1.0 < inv <= ROUND_TOL:
        inv = 1.0
    elif inv > ROUND_TOL:
        lev = min(int(base), math.ceil(target - 1e-9))
        inv = target / lev
        if inv > 1.0 + 1e-9:
            flag = f"at {int(base)}× the investment stays above one slot"
    inv = _floor(inv, step)
    hard = float(cfg.get("rsi_adx_multiplier_hard_cap", 2.0) or 2.0)
    if inv < INV_MIN or inv > hard:
        inv = min(max(inv, INV_MIN), hard)
        flag = f"engine clamps the investment to {INV_MIN:g}…{hard:g}"
    new = dict(inv=inv, lev=lev, inv_set=inv, lev_set=round(lev / base, 4), inv_key=t["inv_key"], lev_key=t["lev_key"])
    return dict(today=t, new=new, size_x=(inv * lev) / (t["inv"] * t["lev"]), flag=flag)


def _x(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _cell(s):
    return (f"**{_x(s['inv'])}× inv · {s['lev']}× lev** (`{s['inv_key']}` {_x(s['inv_set'])} · `{s['lev_key']}` {_x(s['lev_set'])})")


def table(cfg, rows, stops=None, step=STEP):
    """rows = [(row label, {sleeve: factor})] (WILLY_HT reads WILLY's new settings from the same row); stops = {sleeve: one stop today,
    % of the account} → markdown lines."""
    stops = stops or {}
    out = ["| Row | Sleeve | Re-size | Today | Recommended | Size vs today | One stop, % of account |", "|---|---|---|---|---|---|---|"]
    for lab, fac in rows:
        wn = recommend(cfg, "FRENZY_WILLY", fac["FRENZY_WILLY"], step=step)["new"] if "FRENZY_WILLY" in fac else None
        for sl, f in fac.items():
            if sl not in FIELDS and sl != HT:
                continue
            r = recommend(cfg, sl, f, willy_new=wn, step=step)
            st = stops.get(sl)
            d = 2 if st and st < 1 else 1
            stop = f"{st:.{d}f} → {st * r['size_x']:.{d}f}" if st else "–"
            size = f"×{r['size_x']:.2f}" + (f" ⚠ {r['flag']}" if r["flag"] else "")
            out.append(f"| {lab} | {sl} | ×{float(f):.2f} | {_cell(r['today'])} | {_cell(r['new'])} | {size} | {stop} |")
    return out
