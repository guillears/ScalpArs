import sys, os, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_exit_shadows as ES
NOW = int(time.time() * 1000)
b, _ = ES.load([sys.argv[1]]); b = b[b.o >= pd.Timestamp("2026-10-23 18:30")].assign(src="B25", era="B25")
M = pd.read_csv(os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv"), low_memory=False)
M = M[(M.stack_keep == True) & (M.is_probe != True) & M.entry_strategy.astype(str).str.startswith("FLIP")]
os.makedirs(SC, exist_ok=True)
mp = os.path.join(SC, "master_flips.csv"); M.to_csv(mp, index=False)
m, _ = ES.load([mp])
m = m.merge(M.assign(_k=M.opened_at.astype(str).str.replace(" ", "T").str[:19])[["_k", "pair", "era", "stack_pct"]], on=["_k", "pair"], how="left").assign(src="MASTER")
P = pd.concat([b, m], ignore_index=True); P["kk"] = P["_k"]
bud = ES.LS.Budget(max_req=400) if "max_req" in ES.LS.Budget.__init__.__code__.co_varnames else ES.LS.Budget()
cache, out = {}, []
for r in P.itertuples():
    res = ES.price_fill(r, bud, NOW, cache, fetch=True)
    ex = res.get("exits") or {}
    out.append(dict(src=r.src, era=r.era, k=r.kk, pair=r.pair, booked=r.pct, stack=getattr(r, "stack_pct", np.nan), status=res["status"],
                    parity=res.get("parity"), **{x: ex.get(x, np.nan) for x in ES.FLIP_EXITS}))
D = pd.DataFrame(out); D.to_csv(os.path.join(SC, "flip_exits.csv"), index=False)
print(D.groupby(["src", "status"]).size())
for s, g in D[D.status == "PRICED"].groupby("src"):
    base = g["booked"] if s == "B25" else g["stack"]
    print(s, f"N {len(g)} · booked/stack {base.mean():+.3f} · " + " · ".join(f"{x} {g[x].mean():+.3f} (WR {(g[x] > 0).mean():.0%})" for x in ES.FLIP_EXITS), "· parity", g.parity.mean())
