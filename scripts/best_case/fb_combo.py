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
def th(tp, sl): return FS._th_tp(tp, sl)
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
        out.append(dict(k=r.kk, pair=r.pair)); continue
    st, _ = FS.tp_path_start(r.te)
    after = [b for b in bars if b[0] >= st]
    rec = dict(k=r.kk, pair=r.pair, day=r.day, booked=r.pct, notional=r.notional_value,
               zone=FS.obs_pk_zone(r.entry_frenzy_run_pct, r.entry_flatbase_off_peak_pct),
               booked_min=(pd.Timestamp(r.closed_at) - r.o).total_seconds() / 60)
    rec["now510"], rec["h_now510"] = walk(r.E, after, 5, 10, r.te)
    rec["tp1sl4"], rec["h_tp1sl4"] = walk(r.E, after, 1, 4, r.te)
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
