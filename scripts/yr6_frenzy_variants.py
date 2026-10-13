"""FRENZY-only shared-book variants on yr6 2026 seed 1 re-priced to today (operator 2026-10-12/13): LONG / WIDE size and TP, LITE / WILLY alone.
Reuses scripts/yr6_reprice_today.py helpers. Output appended to reports/ENGINE_REPLAY_YR6_TODAY.txt (variants section)."""
import sys, os, runpy
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.argv=["x","--draws","10"]
# load the module's functions without re-running its tables: exec up to the data section
src=open(os.path.join(ROOT,"scripts","yr6_reprice_today.py")).read()
head=src.split("F = pd.read_csv(")[0]
g={"__file__":os.path.join(ROOT,"scripts","yr6_reprice_today.py"),"__name__":"v"}
exec(compile(head,"yr6_reprice_today","exec"),g)
pd=g["pd"]; np=g["np"]
F=pd.read_csv(os.path.join(ROOT,"reports/ENGINE_REPLAY_YR6_fills.csv"),low_memory=False)
F["t"]=pd.to_datetime(F.t,format="mixed"); F=F[(F.year.astype(str)=="2026")&(F.seed==1)&F.sleeve.str.startswith("FRENZY")].reset_index(drop=True)
def run(name, tp, lev, keep):
    g["TP_NEW"].clear(); g["TP_NEW"].update(tp); g["FZ_LEV"].clear(); g["FZ_LEV"].update(lev)
    G=F[F.sleeve.isin(keep)].copy(); G["pct_v"]=[ (tp[r.entry_strategy] if (r.entry_strategy in tp and tp[r.entry_strategy]<3 and pd.notna(r.peak_pnl) and r.peak_pnl>=tp[r.entry_strategy]) else r.pct) for r in G.itertuples()]
    b=g["book"](G,2900.0,"pct_v",True)
    print(f"{name:58} ${b['end']:>9,.0f}  mdd {b['mdd_pct']:.0f}%  worst day {b['worst_day_pct']:.0f}%  {b['dcr_pct']:+.3f}%/day  <20% {b['below20']}")
T3={"FRENZY_LONG":3.0,"FRENZY_WIDE":3.0,"FRENZY_LITE":3.0}; T15={"FRENZY_LONG":1.5,"FRENZY_WIDE":1.5,"FRENZY_LITE":3.0}
ALL=["FRENZY_LONG","FRENZY_WIDE","FRENZY_LITE","FRENZY_WILLY"]
run("DL 283 cfg: LONG/WIDE +1.5 10x · LITE +3 10x · WILLY 10x",T15,dict(FRENZY_LONG=.5,FRENZY_WIDE=.5,FRENZY_LITE=.5,FRENZY_WILLY=.5),ALL)
run("DL 286 cfg (today): LONG/WIDE +1.5 at 2x/1x · LITE +3 10x · WILLY 10x",T15,dict(FRENZY_LONG=.1,FRENZY_WIDE=.05,FRENZY_LITE=.5,FRENZY_WILLY=.5),ALL)
run("LONG/WIDE +3 at 2x/1x · LITE +3 10x · WILLY 10x",T3,dict(FRENZY_LONG=.1,FRENZY_WIDE=.05,FRENZY_LITE=.5,FRENZY_WILLY=.5),ALL)
run("LONG/WIDE +3 at 2x/1x · LITE +3 8x · WILLY 10x (= replay cfg)",T3,dict(FRENZY_LONG=.1,FRENZY_WIDE=.05,FRENZY_LITE=.4,FRENZY_WILLY=.5),ALL)
run("LITE +3 10x · WILLY 10x only",T15,dict(FRENZY_LITE=.5,FRENZY_WILLY=.5),["FRENZY_LITE","FRENZY_WILLY"])
run("LITE +3 10x only",T15,dict(FRENZY_LITE=.5),["FRENZY_LITE"])
run("WILLY 10x only",T15,dict(FRENZY_WILLY=.5),["FRENZY_WILLY"])
run("LONG/WIDE +1.5 10x only",T15,dict(FRENZY_LONG=.5,FRENZY_WIDE=.5),["FRENZY_LONG","FRENZY_WIDE"])
