import sys, os, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import scout_flatbase_short as FS, scout_flatbase_confirm_watch as CF, review_math as RM
from services.flatbase import price_path_exit
EXP = sys.argv[1]
now = int(time.time() * 1000)
c, _ = CF.load(); c["kk"] = c["_k"]
X = pd.read_csv(EXP, low_memory=False)
extra = X[X.entry_strategy == "FLATBASE_SHORT"][["opened_at", "pair", "entry_frenzy_run_pct", "entry_flatbase_off_peak_pct", "notional_value", "closed_at"]]
extra = extra.assign(kk=extra.opened_at.astype(str).str.replace(" ", "T").str[:19])
c = c.merge(extra.drop(columns=["opened_at", "closed_at"]), on=["kk", "pair"], how="left")
# B24 fill (master) lives in the archive, not in the B25 export
b24 = pd.read_csv(os.path.join(ROOT, "reports/BASELINE24_batch1023_LIVE_orders.csv"), low_memory=False)
b24 = b24[b24.entry_strategy == "FLATBASE_SHORT"]
for r in b24.itertuples():
    k = str(r.opened_at).replace(" ", "T")[:19]
    m = (c.kk == k) & (c.pair == r.pair)
    for col in ("entry_frenzy_run_pct", "entry_flatbase_off_peak_pct", "notional_value"):
        c.loc[m, col] = getattr(r, col)
bud = CF.LS.Budget()
# fills opened after the FB_CONFIRM ship (DL 428 + 10 min) were entered at the confirm bar already: their booked result IS the
# FB_CONFIRM outcome (cf510 = booked at today's +5 / −10) — shadow-confirming them again would ask for a SECOND 4-h close.
SHIP_MS = CF._GK.dl_deploy_ms(CF.DL_SHIP, lag_min=10, label="FB_CONFIRM ship")
if SHIP_MS is None:
    sys.exit("FB_CONFIRM ship time not found — cannot split S4-entered vs confirm-entered fills")
# 🎲 (DECISION_LOG 474) the OTHER entries the FB_CONFIRM study tried at registration (DL 417), priced so the luck test penalises the real
# search: waits of 1–24 h (enter at the open of the first 5m bar ≥ signal + W, today's +5 / −10) and selling 1–7 % higher (a resting short
# at E × (1 + x %) filled by the first bar whose high reaches it within 24 h, walked from the next bar; never filled = no trade = 0 %).
# Not priced: "a move against then a 1 h close back below" (stated in the luck note).
WAITS_H = (1, 2, 4, 8, 12, 24)
HIGHER_PCT = (1, 2, 3, 4, 5, 6, 7)
ALT_COLS = tuple(f"w{h}" for h in WAITS_H) + tuple(f"hi{x}" for x in HIGHER_PCT)
def th(tp, sl): return FS._th_tp(tp, sl)
def walk_from(E, rows, tp, sl):
    """(474) an alternative entry gets the same 72 h hold from ITS OWN entry — anchored on the first bar walked (rows < rows[0] + 72 h; the
    engine's TIME exit), never the signal's window (a late entry cut at signal + 72 h reads OPEN). → net % or NaN while still short."""
    if not rows:
        return np.nan
    rows = [b for b in rows if b[0] < rows[0][0] + 72 * 3600_000]
    x = price_path_exit(E, rows, th(tp, sl))
    if x is None or x["reason"] == "OPEN":
        return np.nan
    return float(x["net_pct"])
def bars_to(pair, te, need_ms, bars):
    """the fill's 5m bars extended (one more cached page from the last bar) until need_ms, or as far as the budget allows."""
    out = list(bars)
    while out and out[-1][0] + CF.M5 < need_ms:
        more = CF.fetch_bars(pair, out[-1][0] + CF.M5, now, bud)
        new = [q for q in (more or []) if q[0] > out[-1][0]]
        if not new:   # no page / budget stop / nothing newer — never loop on the same bars
            break
        out += new
    return out
def walk(E, bars, tp, sl, te):
    rows = [b for b in bars if b[0] < te + 72 * 3600_000]
    x = price_path_exit(E, rows, th(tp, sl))
    if x is None or x["reason"] == "OPEN":
        return np.nan, np.nan
    return float(x["net_pct"]), (x["bars"] - 0.5) * 5 + (rows[0][0] - te) / 60000
out = []
for r in c.itertuples():
    bars = CF.fetch_bars(r.pair, r.te, now, bud)
    if not bars:
        rec = dict(k=r.kk, pair=r.pair)
        if r.te >= SHIP_MS:   # confirm-entered: its CF is the booked result, no bars needed (471 deep review)
            rec.update(cf_why="CONFIRMED_LIVE", cf_510_ref=r.pct, cf510=r.pct, wait_min=0.0, booked=r.pct)
        out.append(rec); continue
    st, _ = FS.tp_path_start(r.te)
    after = [b for b in bars if b[0] >= st]
    rec = dict(k=r.kk, pair=r.pair, day=r.day, booked=r.pct, notional=r.notional_value,
               zone=FS.obs_pk_zone(r.entry_frenzy_run_pct, r.entry_flatbase_off_peak_pct),
               booked_min=(pd.Timestamp(r.closed_at) - r.o).total_seconds() / 60)
    rec["now510"], rec["h_now510"] = walk(r.E, after, 5, 10, r.te)
    rec["tp1sl4"], rec["h_tp1sl4"] = walk(r.E, after, 1, 4, r.te)
    if r.te < SHIP_MS:   # 🎲 (474) the alternatives tried at registration (S4-entered fills only — a confirm-entered fill has no S4 entry)
        bx = bars_to(r.pair, r.te, min(r.te + (24 + 72) * 3600_000, now), bars)   # the latest entry (24 h) + its own 72 h hold
        b = np.asarray(bx, float); t = b[:, 0].astype("int64")
        for h in WAITS_H:
            j = np.flatnonzero(t >= FS.tp_path_start(r.te + h * 3600_000)[0])   # the bar a fill at signal + W would start at (474 review)
            rec[f"w{h}"] = walk_from(float(b[j[0], 1]), [list(x) for x in bx[j[0]:]], 5, 10) if len(j) else np.nan
        for x in HIGHER_PCT:
            lim = r.E * (1 + x / 100)
            j = np.flatnonzero((t >= st) & (t < r.te + 24 * 3600_000) & (b[:, 2] >= lim))
            if not len(j):
                rec[f"hi{x}"] = 0.0 if (len(t) and t[-1] >= r.te + 24 * 3600_000) else np.nan   # never filled → no trade (0 %), or still open
            else:
                # the fill bar is stop-checked from the limit (its high can run through the stop after the fill), never TP-credited (its
                # low may come before the fill): a synthetic [t, lim, high, lim, close] first row (474 deep review)
                f0 = bx[j[0]]
                rec[f"hi{x}"] = walk_from(lim, [[f0[0], lim, f0[2], lim, f0[4]]] + [list(y) for y in bx[j[0] + 1:]], 5, 10)
    if r.te >= SHIP_MS:   # confirm-entered: the real fill is the CF
        rec["cf_why"], rec["cf_510_ref"] = "CONFIRMED_LIVE", r.pct
        rec["cf510"], rec["h_cf510"] = r.pct, rec["booked_min"]
        rec["cf14"], rec["h_cf14"] = rec["tp1sl4"], rec["h_tp1sl4"]
        rec["wait_min"] = 0.0
        out.append(rec); continue
    why, sh = CF.price_shadow(bars, r.te, r.E, now)
    rec["cf_why"], rec["cf_510_ref"] = why, sh
    if why in ("SKIP",):
        rec["cf510"] = rec["cf14"] = 0.0; rec["h_cf510"] = rec["h_cf14"] = 0.0
    elif why in ("PENDING", "UNPRICED"):
        rec["cf510"] = rec["cf14"] = np.nan
    else:
        b = np.asarray(bars, float); t = b[:, 0].astype("int64")
        hit = np.flatnonzero(((t + CF.M5) % CF.H4 == 0) & (t + CF.M5 <= r.te + 24 * 3600_000) & (t + CF.M5 > r.te) & (b[:, 4] < r.E))
        i0 = int(hit[0]) + 1; Ee = float(b[i0, 1])
        rows = [list(x) for x in bars[i0:]]
        rec["cf510"], rec["h_cf510"] = walk(Ee, rows, 5, 10, r.te); rec["cf14"], rec["h_cf14"] = walk(Ee, rows, 1, 4, r.te)
        rec["wait_min"] = (rows[0][0] - r.te) / 60000
    out.append(rec)
P = pd.DataFrame(out)
P["batch"] = np.where(P.k < "2026-10-23T18:30", "B24", "B25")
pd.set_option("display.width", 250)
print(P[["batch","booked_min","h_now510","h_tp1sl4","wait_min","h_cf510","h_cf14", "k", "pair", "zone", "booked", "now510", "tp1sl4", "cf_why", "cf_510_ref", "cf510", "cf14"]].round(2).to_string())
os.makedirs(os.path.join(ROOT, "reports", "best_case"), exist_ok=True)
P.to_csv(os.path.join(ROOT, "reports", "best_case", "fb_combo.csv"), index=False)
